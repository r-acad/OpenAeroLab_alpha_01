using KernelAbstractions
using StaticArrays
using Atomix

# ── RESTRICT COVERED-BLOCK BANDING (2026-06-12, 100-MLUPS item 2) ────────────
# The main restriction kernel launches dense over ALL parent blocks every
# coarse substep, but every cell of a parent block is a complete no-op unless
# its octant child block exists (each parent cell's 8 fine children live in
# exactly ONE of the parent block's 8 octant child blocks, so block-level
# membership "≥1 octant child active" is EXACT, not a superset). The AMR
# topology is static after setup (the mark-once lesson), so the covered-block
# list is computed once per child level on the host and the launch is banded
# to it. TRUE-bitwise: covered blocks run the identical per-cell math (no
# cross-block interaction in the kernel); skipped blocks did nothing.
# "0" reverts to the dense launch.
const _D3Q19_RESTRICT_COVERED_BAND =
    get(ENV, "LUDWIG_D3Q19_RESTRICT_COVERED_BAND", "1") != "0"
const _RESTRICT_COVERED_CACHE = IdDict{Any, Any}()
# WALL-DRAG FIX (2026-06-22, default OFF): in the FULL-coverage restriction path
# (cell_has_full_child_support), obstacle-adjacent coarse cells skip their SOLID
# children (line ~2122 `continue`) and average the remaining FLUID children — but
# still set frac=1.0, so the τ-rescaled non-equilibrium STRESS carrier is injected at
# FULL fluid strength into a cell that is only partly fluid. On a wall-resolving AMR
# case this over-drives the near-wall coarse cells (which feed the fine halo) and
# inflates the drag: converged cylinder Re=20 Cd 2.62 (3-level) vs 1.77 (single-level)
# vs 2.05 Henderson — the inflation scales with how close the L→L-1 interface sits to
# the wall (standoff 0.75D → 2.19). FIX: weight the restricted populations by the
# ACTUAL fluid coverage frac = fluid_children/8 and blend the (1-frac) solid portion
# with the parent's prior state — exactly what the partial-FRONTIER path already does
# (line ~3209 `frac = valid_children*0.125`) for block-absent cells, here extended to
# cells whose child BLOCK exists but whose child CELLS are partly solid. The macros
# (ρ, j) stay averaged over fluid children (correct); only the stress carrier is
# coverage-scaled. Gated; "1" enables. Bitwise no-op for cells with all 8 fluid
# children (frac stays 1.0).
const _D3Q19_RESTRICT_COVERAGE_FRAC =
    get(ENV, "LUDWIG_D3Q19_RESTRICT_COVERAGE_FRAC", "0") == "1"
# _AMR_FNEQ_POSTCOLL (AMR-SEAM FIX 1A) is defined in physics_interpolation.jl
# (included first) and reused here for the restriction-side correction.

function _restrict_covered_block_list(parent_level, child_level, backend)
    cached = get(_RESTRICT_COVERED_CACHE, child_level, nothing)
    cached !== nothing && return cached
    map_x = Array(parent_level.map_x)
    map_y = Array(parent_level.map_y)
    map_z = Array(parent_level.map_z)
    bp = Array(child_level.block_pointer)
    cdx, cdy, cdz = size(bp)
    n_parent_blocks = length(parent_level.active_block_coords)
    list = Int32[]
    for pb in 1:n_parent_blocks
        cbx0 = 2 * Int(map_x[pb]) - 1
        cby0 = 2 * Int(map_y[pb]) - 1
        cbz0 = 2 * Int(map_z[pb]) - 1
        covered = false
        for oz in 0:1, oy in 0:1, ox in 0:1
            cbx = cbx0 + ox; cby = cby0 + oy; cbz = cbz0 + oz
            if 1 <= cbx <= cdx && 1 <= cby <= cdy && 1 <= cbz <= cdz &&
               bp[cbx, cby, cbz] != 0
                covered = true
                break
            end
        end
        covered && push!(list, Int32(pb))
    end
    list_dev = KernelAbstractions.allocate(backend, Int32, length(list))
    length(list) > 0 && copyto!(list_dev, list)
    entry = (list = list_dev, n = length(list), n_total = n_parent_blocks)
    _RESTRICT_COVERED_CACHE[child_level] = entry
    return entry
end

@inline function lookup_child_cell(
    child_block_pointer,
    child_dim_x::Int32, child_dim_y::Int32, child_dim_z::Int32,
    block_size::Int32,
    fgx::Int32, fgy::Int32, fgz::Int32
)
    fgy = _pwrap_y(fgy, child_dim_y * block_size)   # periodic-Y AMR (identity when off)
    if fgx < Int32(1) || fgy < Int32(1) || fgz < Int32(1)
        return (Int32(0), Int32(0), Int32(0), Int32(0), false)
    end

    cbx = div(fgx - Int32(1), block_size) + Int32(1)
    cby = div(fgy - Int32(1), block_size) + Int32(1)
    cbz = div(fgz - Int32(1), block_size) + Int32(1)
    if cbx > child_dim_x || cby > child_dim_y || cbz > child_dim_z
        return (Int32(0), Int32(0), Int32(0), Int32(0), false)
    end

    cb_idx = child_block_pointer[cbx, cby, cbz]
    if cb_idx == Int32(0)
        return (Int32(0), Int32(0), Int32(0), Int32(0), false)
    end

    clx = mod(fgx - Int32(1), block_size) + Int32(1)
    cly = mod(fgy - Int32(1), block_size) + Int32(1)
    clz = mod(fgz - Int32(1), block_size) + Int32(1)
    return (clx, cly, clz, cb_idx, true)
end

@inline function lookup_parent_cell(
    parent_block_pointer,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    block_size::Int32,
    gx::Int32, gy::Int32, gz::Int32
)
    gy = _pwrap_y(gy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
    if gx < Int32(1) || gy < Int32(1) || gz < Int32(1)
        return (Int32(0), Int32(0), Int32(0), Int32(0), false)
    end

    pbx = div(gx - Int32(1), block_size) + Int32(1)
    pby = div(gy - Int32(1), block_size) + Int32(1)
    pbz = div(gz - Int32(1), block_size) + Int32(1)
    if pbx > parent_dim_x || pby > parent_dim_y || pbz > parent_dim_z
        return (Int32(0), Int32(0), Int32(0), Int32(0), false)
    end

    pb_idx = parent_block_pointer[pbx, pby, pbz]
    if pb_idx == Int32(0)
        return (Int32(0), Int32(0), Int32(0), Int32(0), false)
    end

    plx = mod(gx - Int32(1), block_size) + Int32(1)
    ply = mod(gy - Int32(1), block_size) + Int32(1)
    plz = mod(gz - Int32(1), block_size) + Int32(1)
    return (plx, ply, plz, pb_idx, true)
end

@inline function solve_shell_moment4(
    m00::Float32, m01::Float32, m02::Float32, m03::Float32,
    m11::Float32, m12::Float32, m13::Float32,
    m22::Float32, m23::Float32,
    m33::Float32,
    r0::Float32, r1::Float32, r2::Float32, r3::Float32
)
    eps = 1.0f-12

    a00 = m00 + eps
    a01 = m01; a02 = m02; a03 = m03; b0 = r0
    a10 = m01; a11 = m11 + eps; a12 = m12; a13 = m13; b1 = r1
    a20 = m02; a21 = m12; a22 = m22 + eps; a23 = m23; b2 = r2
    a30 = m03; a31 = m13; a32 = m23; a33 = m33 + eps; b3 = r3

    p0 = a00
    ok0 = abs(p0) > eps
    inv0 = ok0 ? 1.0f0 / p0 : 0.0f0
    f10 = a10 * inv0; f20 = a20 * inv0; f30 = a30 * inv0
    a11 -= f10 * a01; a12 -= f10 * a02; a13 -= f10 * a03; b1 -= f10 * b0
    a21 -= f20 * a01; a22 -= f20 * a02; a23 -= f20 * a03; b2 -= f20 * b0
    a31 -= f30 * a01; a32 -= f30 * a02; a33 -= f30 * a03; b3 -= f30 * b0

    p1 = a11
    ok1 = abs(p1) > eps
    inv1 = ok1 ? 1.0f0 / p1 : 0.0f0
    f21 = a21 * inv1; f31 = a31 * inv1
    a22 -= f21 * a12; a23 -= f21 * a13; b2 -= f21 * b1
    a32 -= f31 * a12; a33 -= f31 * a13; b3 -= f31 * b1

    p2 = a22
    ok2 = abs(p2) > eps
    inv2 = ok2 ? 1.0f0 / p2 : 0.0f0
    f32 = a32 * inv2
    a33 -= f32 * a23; b3 -= f32 * b2

    p3 = a33
    ok3 = abs(p3) > eps
    inv3 = ok3 ? 1.0f0 / p3 : 0.0f0

    x3 = b3 * inv3
    x2 = (b2 - a23 * x3) * inv2
    x1 = (b1 - a12 * x2 - a13 * x3) * inv1
    x0 = (b0 - a01 * x1 - a02 * x2 - a03 * x3) * inv0

    return (x0, x1, x2, x3, ok0 && ok1 && ok2 && ok3)
end

# ── ET-AMR restriction decode/encode helpers (STAGE 4, 2026-06-18) ────────────
# The restricted child cells are INTERIOR to the fine box, so the ET decode's
# downstream +c neighbour always exists (resolved via child_block_pointer block-
# crossing — the restriction kernel does not carry a neighbor_table). These mirror
# _et_load_l1!/_et_store_l1!'s INTERIOR mapping exactly, at the given phase.

# Resolve the (cell-local, block-idx) of the +c neighbour of child cell
# (clx,cly,clz) in block (cbx,cby,cbz), via child_block_pointer. Returns
# (ncl_x,ncl_y,ncl_z, ncb) with ncb<=0 if the neighbour block is absent.
@inline function _et_child_nbr(
    child_block_pointer, cbx::Int32, cby::Int32, cbz::Int32,
    clx::Int32, cly::Int32, clz::Int32,
    cxk::Int32, cyk::Int32, czk::Int32, bs::Int32,
    pdimx::Int32, pdimy::Int32, pdimz::Int32,
)
    sx = clx + cxk; sy = cly + cyk; sz = clz + czk
    if sx >= Int32(1) && sx <= bs && sy >= Int32(1) && sy <= bs && sz >= Int32(1) && sz <= bs
        ncb0 = child_block_pointer[cbx, cby, cbz]
        return sx, sy, sz, ncb0
    end
    nbx = cbx + (sx < Int32(1) ? Int32(-1) : (sx > bs ? Int32(1) : Int32(0)))
    nby = cby + (sy < Int32(1) ? Int32(-1) : (sy > bs ? Int32(1) : Int32(0)))
    nbz = cbz + (sz < Int32(1) ? Int32(-1) : (sz > bs ? Int32(1) : Int32(0)))
    if nbx < Int32(1) || nbx > pdimx || nby < Int32(1) || nby > pdimy || nbz < Int32(1) || nbz > pdimz
        return Int32(0), Int32(0), Int32(0), Int32(0)
    end
    ncb = child_block_pointer[nbx, nby, nbz]
    nsx = sx < Int32(1) ? sx + bs : (sx > bs ? sx - bs : sx)
    nsy = sy < Int32(1) ? sy + bs : (sy > bs ? sy - bs : sy)
    nsz = sz < Int32(1) ? sz + bs : (sz > bs ? sz - bs : sz)
    return nsx, nsy, nsz, ncb
end

# Decode the 19 conventional populations of an ET (Esoteric) child cell into cfloc.
# Interior mapping (verified): cfloc[kp]=f[own, p?kp:km... ] etc. — identical to
# _et_load_l1!'s interior path. If a downstream neighbour is unexpectedly absent
# (ragged frontier), that opp slot falls back to the own-cell value (harmless: the
# caller already gates restriction on full child support).
@inline function _et_decode_child_cell!(
    cfloc, child_f, child_block_pointer,
    cbx::Int32, cby::Int32, cbz::Int32, cb_idx::Int32,
    clx::Int32, cly::Int32, clz::Int32,
    et_pair_kp, et_pair_km, cx_arr, cy_arr, cz_arr,
    bs::Int32, p::Int32, pdimx::Int32, pdimy::Int32, pdimz::Int32,
)
    @inbounds begin
        cfloc[1] = child_f[clx, cly, clz, cb_idx, 1]
        for q in Int32(1):Int32(9)
            kp = et_pair_kp[q]; km = et_pair_km[q]
            cxk = cx_arr[kp]; cyk = cy_arr[kp]; czk = cz_arr[kp]
            cfloc[kp] = child_f[clx, cly, clz, cb_idx, p == Int32(0) ? km : kp]
            nlx, nly, nlz, ncb = _et_child_nbr(
                child_block_pointer, cbx, cby, cbz, clx, cly, clz,
                cxk, cyk, czk, bs, pdimx, pdimy, pdimz)
            cfloc[km] = ncb > Int32(0) ?
                child_f[nlx, nly, nlz, ncb, p == Int32(0) ? kp : km] :
                child_f[clx, cly, clz, cb_idx, p == Int32(0) ? kp : km]
        end
    end
    return nothing
end

# 5-moment Gram solve — mass + 3·momentum + ENERGY (|c|²). Extends
# solve_shell_moment4 with a 5th basis vector e5(c)=|c|² so the conservative
# reflux carries the registered ENERGY flux too, not just mass+momentum. This
# is the strict-ρE upgrade for the time-refined AMR interface (SOTA #2,
# 2026-05-29): at Ma≳1.2 the shock-heating energy crossing the interface must
# be conserved, which the 4-moment projection silently dropped. Symmetric 5×5
# Gauss elimination, same inline pattern as solve_shell_moment4. The basis is
# {1, cx, cy, cz, |c|²}; m4* are the energy-row Gram entries, r4 the energy
# residual moment. Returns (a0,ax,ay,az,a4, ok).
@inline function solve_shell_moment5(
    m00::Float32, m01::Float32, m02::Float32, m03::Float32, m04::Float32,
    m11::Float32, m12::Float32, m13::Float32, m14::Float32,
    m22::Float32, m23::Float32, m24::Float32,
    m33::Float32, m34::Float32,
    m44::Float32,
    r0::Float32, r1::Float32, r2::Float32, r3::Float32, r4::Float32
)
    eps = 1.0f-12

    a00 = m00 + eps; a01 = m01; a02 = m02; a03 = m03; a04 = m04; b0 = r0
    a10 = m01; a11 = m11 + eps; a12 = m12; a13 = m13; a14 = m14; b1 = r1
    a20 = m02; a21 = m12; a22 = m22 + eps; a23 = m23; a24 = m24; b2 = r2
    a30 = m03; a31 = m13; a32 = m23; a33 = m33 + eps; a34 = m34; b3 = r3
    a40 = m04; a41 = m14; a42 = m24; a43 = m34; a44 = m44 + eps; b4 = r4

    p0 = a00; ok0 = abs(p0) > eps; inv0 = ok0 ? 1.0f0 / p0 : 0.0f0
    f10 = a10*inv0; f20 = a20*inv0; f30 = a30*inv0; f40 = a40*inv0
    a11 -= f10*a01; a12 -= f10*a02; a13 -= f10*a03; a14 -= f10*a04; b1 -= f10*b0
    a21 -= f20*a01; a22 -= f20*a02; a23 -= f20*a03; a24 -= f20*a04; b2 -= f20*b0
    a31 -= f30*a01; a32 -= f30*a02; a33 -= f30*a03; a34 -= f30*a04; b3 -= f30*b0
    a41 -= f40*a01; a42 -= f40*a02; a43 -= f40*a03; a44 -= f40*a04; b4 -= f40*b0

    p1 = a11; ok1 = abs(p1) > eps; inv1 = ok1 ? 1.0f0 / p1 : 0.0f0
    f21 = a21*inv1; f31 = a31*inv1; f41 = a41*inv1
    a22 -= f21*a12; a23 -= f21*a13; a24 -= f21*a14; b2 -= f21*b1
    a32 -= f31*a12; a33 -= f31*a13; a34 -= f31*a14; b3 -= f31*b1
    a42 -= f41*a12; a43 -= f41*a13; a44 -= f41*a14; b4 -= f41*b1

    p2 = a22; ok2 = abs(p2) > eps; inv2 = ok2 ? 1.0f0 / p2 : 0.0f0
    f32 = a32*inv2; f42 = a42*inv2
    a33 -= f32*a23; a34 -= f32*a24; b3 -= f32*b2
    a43 -= f42*a23; a44 -= f42*a24; b4 -= f42*b2

    p3 = a33; ok3 = abs(p3) > eps; inv3 = ok3 ? 1.0f0 / p3 : 0.0f0
    f43 = a43*inv3
    a44 -= f43*a34; b4 -= f43*b3

    p4 = a44; ok4 = abs(p4) > eps; inv4 = ok4 ? 1.0f0 / p4 : 0.0f0

    x4 = b4 * inv4
    x3 = (b3 - a34*x4) * inv3
    x2 = (b2 - a23*x3 - a24*x4) * inv2
    x1 = (b1 - a12*x2 - a13*x3 - a14*x4) * inv1
    x0 = (b0 - a01*x1 - a02*x2 - a03*x3 - a04*x4) * inv0

    return (x0, x1, x2, x3, x4, ok0 && ok1 && ok2 && ok3 && ok4)
end

@inline function count_fine_children(
    child_block_pointer,
    child_dim_x::Int32, child_dim_y::Int32, child_dim_z::Int32,
    block_size::Int32,
    gx::Int32, gy::Int32, gz::Int32
)
    count = Int32(0)
    @inbounds for sx in Int32(0):Int32(1), sy in Int32(0):Int32(1), sz in Int32(0):Int32(1)
        fgx = Int32(2) * gx - Int32(1) + sx
        fgy = Int32(2) * gy - Int32(1) + sy
        fgz = Int32(2) * gz - Int32(1) + sz
        _, _, _, _, valid = lookup_child_cell(
            child_block_pointer,
            child_dim_x, child_dim_y, child_dim_z,
            block_size,
            fgx, fgy, fgz
        )
        count += valid ? Int32(1) : Int32(0)
    end
    return count
end

@inline function coarse_cell_has_fine_children(
    child_block_pointer,
    child_dim_x::Int32, child_dim_y::Int32, child_dim_z::Int32,
    block_size::Int32,
    gx::Int32, gy::Int32, gz::Int32
)
    return count_fine_children(
        child_block_pointer,
        child_dim_x, child_dim_y, child_dim_z,
        block_size,
        gx, gy, gz
    ) == Int32(8)
end

@inline function amr_gpu_face_budget_allocated(level::BlockLevel)
    return length(level.amr_face_budget) > 12
end

const AMR_POP_STREAM_REGISTER_BASE = Int32(12)
const AMR_POP_STREAM_REGISTER_STRIDE = Int32(3)
const AMR_POP_STREAM_REGISTER_MAX_DIRS = Int32(39)
const AMR_POP_STREAM_FACE_COUNT = Int32(6)
const AMR_POP_STREAM_FACE_BASE =
    AMR_POP_STREAM_REGISTER_BASE +
    AMR_POP_STREAM_REGISTER_STRIDE * AMR_POP_STREAM_REGISTER_MAX_DIRS
const AMR_POP_STREAM_FACE_COMPONENTS =
    AMR_POP_STREAM_FACE_COUNT *
    AMR_POP_STREAM_REGISTER_STRIDE *
    AMR_POP_STREAM_REGISTER_MAX_DIRS
const AMR_POP_STREAM_XFACE_GLOBAL_BASE =
    AMR_POP_STREAM_FACE_BASE + AMR_POP_STREAM_FACE_COMPONENTS
const AMR_POP_STREAM_XFACE_GLOBAL_STRIDE = Int32(5)
const AMR_POP_STREAM_XFACE_GLOBAL_COMPONENTS =
    Int32(2) * AMR_POP_STREAM_XFACE_GLOBAL_STRIDE

@inline function amr_population_stream_register_allocated(level::BlockLevel)
    return size(level.amr_face_budget, 5) >=
           Int(AMR_POP_STREAM_REGISTER_BASE +
               AMR_POP_STREAM_REGISTER_STRIDE * AMR_POP_STREAM_REGISTER_MAX_DIRS)
end

@inline function amr_population_stream_register_face_resolved_allocated(level::BlockLevel)
    return size(level.amr_face_budget, 5) >=
           Int(AMR_POP_STREAM_FACE_BASE + AMR_POP_STREAM_FACE_COMPONENTS)
end

@inline function amr_population_stream_xface_global_register_allocated(level::BlockLevel)
    return size(level.amr_face_budget, 5) >=
           Int(AMR_POP_STREAM_XFACE_GLOBAL_BASE + AMR_POP_STREAM_XFACE_GLOBAL_COMPONENTS)
end

@inline function amr_pop_stream_slot(k::Int32, offset::Int32)
    return AMR_POP_STREAM_REGISTER_BASE +
           (k - Int32(1)) * AMR_POP_STREAM_REGISTER_STRIDE +
           offset
end

@inline function amr_pop_stream_face_slot(face_id::Int32, k::Int32, offset::Int32)
    return AMR_POP_STREAM_FACE_BASE +
           ((face_id - Int32(1)) * AMR_POP_STREAM_REGISTER_MAX_DIRS +
            (k - Int32(1))) * AMR_POP_STREAM_REGISTER_STRIDE +
           offset
end

@inline function amr_pop_stream_xface_global_slot(face_id::Int32, offset::Int32)
    return AMR_POP_STREAM_XFACE_GLOBAL_BASE +
           (face_id - Int32(1)) * AMR_POP_STREAM_XFACE_GLOBAL_STRIDE +
           offset
end

@inline function amr_oriented_face_id(axis::Int32, normal_sign::Int32)
    if axis == Int32(1)
        return normal_sign > Int32(0) ? Int32(1) : Int32(2)
    elseif axis == Int32(2)
        return normal_sign > Int32(0) ? Int32(3) : Int32(4)
    end
    return normal_sign > Int32(0) ? Int32(5) : Int32(6)
end

@inline function link_step_component(d::Int32, step::Int32, steps::Int32)
    if d == Int32(0)
        return Int32(0)
    end
    ad = abs(d)
    if ad == steps
        return d > Int32(0) ? step : -step
    end
    return d > Int32(0) ? Int32(1) : Int32(-1)
end

@inline function coarse_link_crosses_fine_children(
    child_block_pointer,
    child_dim_x::Int32,
    child_dim_y::Int32,
    child_dim_z::Int32,
    block_size::Int32,
    gx::Int32,
    gy::Int32,
    gz::Int32,
    dx::Int32,
    dy::Int32,
    dz::Int32
)
    steps = max(max(abs(dx), abs(dy)), abs(dz))
    steps == Int32(0) && return false
    @inbounds for step in Int32(1):Int32(3)
        if step <= steps
            tx = gx + link_step_component(dx, step, steps)
            ty = gy + link_step_component(dy, step, steps)
            tz = gz + link_step_component(dz, step, steps)
            if coarse_cell_has_fine_children(
                child_block_pointer,
                child_dim_x,
                child_dim_y,
                child_dim_z,
                block_size,
                tx,
                ty,
                tz
            )
                return true
            end
        end
    end
    return false
end

@inline function first_fine_child_coarse_cell_along_link(
    child_block_pointer,
    child_dim_x::Int32,
    child_dim_y::Int32,
    child_dim_z::Int32,
    block_size::Int32,
    gx::Int32,
    gy::Int32,
    gz::Int32,
    dx::Int32,
    dy::Int32,
    dz::Int32
)
    steps = max(max(abs(dx), abs(dy)), abs(dz))
    steps == Int32(0) && return (gx, gy, gz, false)
    @inbounds for step in Int32(1):Int32(3)
        if step <= steps
            tx = gx + link_step_component(dx, step, steps)
            ty = gy + link_step_component(dy, step, steps)
            tz = gz + link_step_component(dz, step, steps)
            if coarse_cell_has_fine_children(
                child_block_pointer,
                child_dim_x,
                child_dim_y,
                child_dim_z,
                block_size,
                tx,
                ty,
                tz
            )
                return (tx, ty, tz, true)
            end
        end
    end
    return (gx, gy, gz, false)
end

@inline function interface_child_coord(coarse_g::Int32, dir::Int32, side_bit::Int32)
    if dir > Int32(0)
        return Int32(2) * coarse_g - Int32(1)
    elseif dir < Int32(0)
        return Int32(2) * coarse_g
    end
    return Int32(2) * coarse_g - Int32(1) + side_bit
end

function reset_amr_gpu_face_budgets!(grids)
    for level in grids
        amr_gpu_face_budget_allocated(level) || continue
        fill!(level.amr_face_budget, 0.0f0)
    end
    return nothing
end

@kernel function accumulate_amr_population_stream_register_kernel!(
    face_budget,
    parent_f,
    parent_obstacle,
    parent_map_x,
    parent_map_y,
    parent_map_z,
    child_block_pointer,
    child_obstacle,
    child_f,
    cx_arr,
    cy_arr,
    cz_arr,
    block_size::Int32,
    child_dim_x::Int32,
    child_dim_y::Int32,
    child_dim_z::Int32,
    n_dirs_flow::Int32,
    dt_weight::Float32,
    side::Int32,
    face_resolved::Int32
)
    lx, ly, lz, pb_idx = @index(Global, NTuple)
    bs = block_size

    gx = (parent_map_x[pb_idx] - Int32(1)) * bs + Int32(lx)
    gy = (parent_map_y[pb_idx] - Int32(1)) * bs + Int32(ly)
    gz = (parent_map_z[pb_idx] - Int32(1)) * bs + Int32(lz)

    active_parent = !parent_obstacle[lx, ly, lz, pb_idx] &&
                    count_fine_children(
                        child_block_pointer,
                        child_dim_x,
                        child_dim_y,
                        child_dim_z,
                        bs,
                        gx,
                        gy,
                        gz
                    ) == Int32(0)

    if active_parent
        @inbounds for kk in Int32(1):AMR_POP_STREAM_REGISTER_MAX_DIRS
            if kk <= n_dirs_flow
                dx = Int32(cx_arr[kk])
                dy = Int32(cy_arr[kk])
                dz = Int32(cz_arr[kk])
                dst_gx, dst_gy, dst_gz, valid_first_fine =
                    first_fine_child_coarse_cell_along_link(
                    child_block_pointer,
                    child_dim_x,
                    child_dim_y,
                    child_dim_z,
                    bs,
                    gx,
                    gy,
                    gz,
                    dx,
                    dy,
                    dz
                )
                if valid_first_fine
                    cross_x = dst_gx == gx ? Int32(0) : (dst_gx > gx ? Int32(1) : Int32(-1))
                    cross_y = dst_gy == gy ? Int32(0) : (dst_gy > gy ? Int32(1) : Int32(-1))
                    cross_z = dst_gz == gz ? Int32(0) : (dst_gz > gz ? Int32(1) : Int32(-1))
                    n_cross = (cross_x == Int32(0) ? Int32(0) : Int32(1)) +
                              (cross_y == Int32(0) ? Int32(0) : Int32(1)) +
                              (cross_z == Int32(0) ? Int32(0) : Int32(1))
                    split_weight = n_cross > Int32(0) ? 1.0f0 / Float32(n_cross) : 0.0f0
                    if side == Int32(1)
                        count_slot = amr_pop_stream_slot(kk, Int32(3))
                        coarse_slot = amr_pop_stream_slot(kk, Int32(1))
                        coarse_stream = dt_weight * parent_f[lx, ly, lz, pb_idx, kk]
                        face_budget[lx, ly, lz, pb_idx, count_slot] += dt_weight
                        face_budget[lx, ly, lz, pb_idx, coarse_slot] += coarse_stream
                        if face_resolved == Int32(1) && n_cross > Int32(0)
                            if cross_x != Int32(0)
                                face_id = amr_oriented_face_id(Int32(1), cross_x)
                                face_budget[lx, ly, lz, pb_idx,
                                            amr_pop_stream_face_slot(face_id, kk, Int32(3))] +=
                                    dt_weight * split_weight
                                face_budget[lx, ly, lz, pb_idx,
                                            amr_pop_stream_face_slot(face_id, kk, Int32(1))] +=
                                    coarse_stream * split_weight
                            end
                            if cross_y != Int32(0)
                                face_id = amr_oriented_face_id(Int32(2), cross_y)
                                face_budget[lx, ly, lz, pb_idx,
                                            amr_pop_stream_face_slot(face_id, kk, Int32(3))] +=
                                    dt_weight * split_weight
                                face_budget[lx, ly, lz, pb_idx,
                                            amr_pop_stream_face_slot(face_id, kk, Int32(1))] +=
                                    coarse_stream * split_weight
                            end
                            if cross_z != Int32(0)
                                face_id = amr_oriented_face_id(Int32(3), cross_z)
                                face_budget[lx, ly, lz, pb_idx,
                                            amr_pop_stream_face_slot(face_id, kk, Int32(3))] +=
                                    dt_weight * split_weight
                                face_budget[lx, ly, lz, pb_idx,
                                            amr_pop_stream_face_slot(face_id, kk, Int32(1))] +=
                                    coarse_stream * split_weight
                            end
                        end
                    else
                        fine_sum = 0.0f0
                        n_valid = Int32(0)
                        sx_max = dx == Int32(0) ? Int32(1) : Int32(0)
                        sy_max = dy == Int32(0) ? Int32(1) : Int32(0)
                        sz_max = dz == Int32(0) ? Int32(1) : Int32(0)
                        for sx in Int32(0):Int32(1)
                            if sx <= sx_max
                                for sy in Int32(0):Int32(1)
                                    if sy <= sy_max
                                        for sz in Int32(0):Int32(1)
                                            if sz <= sz_max
                                                fgx = interface_child_coord(dst_gx, dx, sx)
                                                fgy = interface_child_coord(dst_gy, dy, sy)
                                                fgz = interface_child_coord(dst_gz, dz, sz)
                                                clx, cly, clz, cb, valid_child = lookup_child_cell(
                                                    child_block_pointer,
                                                    child_dim_x,
                                                    child_dim_y,
                                                    child_dim_z,
                                                    bs,
                                                    fgx,
                                                    fgy,
                                                    fgz
                                                )
                                                if valid_child && !child_obstacle[clx, cly, clz, cb]
                                                    fine_sum += child_f[clx, cly, clz, cb, kk]
                                                    n_valid += Int32(1)
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end
                        if n_valid > Int32(0)
                            fine_slot = amr_pop_stream_slot(kk, Int32(2))
                            fine_stream = dt_weight * fine_sum / Float32(n_valid)
                            face_budget[lx, ly, lz, pb_idx, fine_slot] +=
                                fine_stream
                            if face_resolved == Int32(1) && n_cross > Int32(0)
                                if cross_x != Int32(0)
                                    face_id = amr_oriented_face_id(Int32(1), cross_x)
                                    face_budget[lx, ly, lz, pb_idx,
                                                amr_pop_stream_face_slot(face_id, kk, Int32(2))] +=
                                        fine_stream * split_weight
                                end
                                if cross_y != Int32(0)
                                    face_id = amr_oriented_face_id(Int32(2), cross_y)
                                    face_budget[lx, ly, lz, pb_idx,
                                                amr_pop_stream_face_slot(face_id, kk, Int32(2))] +=
                                        fine_stream * split_weight
                                end
                                if cross_z != Int32(0)
                                    face_id = amr_oriented_face_id(Int32(3), cross_z)
                                    face_budget[lx, ly, lz, pb_idx,
                                                amr_pop_stream_face_slot(face_id, kk, Int32(2))] +=
                                        fine_stream * split_weight
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

function accumulate_amr_population_stream_register!(
    parent_level::BlockLevel,
    child_level::BlockLevel;
    side::Symbol,
    dt_weight::Float32,
    parent_f_current=nothing,
    child_f_current=nothing,
    cx_arr,
    cy_arr,
    cz_arr
)
    amr_population_stream_register_allocated(parent_level) || return nothing
    side === :parent || side === :child ||
        error("AMR population stream register side must be :parent or :child")
    n_parent_blocks = length(parent_level.active_block_coords)
    n_parent_blocks == 0 && return nothing
    parent_f_data = parent_f_current === nothing ? parent_level.f : parent_f_current
    child_f_data = child_f_current === nothing ? child_level.f : child_f_current
    backend = get_backend(parent_level.rho)
    kernel! = accumulate_amr_population_stream_register_kernel!(backend)
    kernel!(
        parent_level.amr_face_budget,
        parent_f_data,
        parent_level.obstacle,
        parent_level.map_x,
        parent_level.map_y,
        parent_level.map_z,
        child_level.block_pointer,
        child_level.obstacle,
        child_f_data,
        cx_arr,
        cy_arr,
        cz_arr,
        Int32(BLOCK_SIZE),
        Int32(size(child_level.block_pointer, 1)),
        Int32(size(child_level.block_pointer, 2)),
        Int32(size(child_level.block_pointer, 3)),
        Int32(size(parent_f_data, 5)),
        dt_weight,
        side === :parent ? Int32(1) : Int32(2),
        (AMR_POPULATION_STREAM_REGISTER_FACE_RESOLVED &&
         amr_population_stream_register_face_resolved_allocated(parent_level)) ?
            Int32(1) : Int32(0);
        ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_parent_blocks)
    )
    return nothing
end

@kernel function apply_amr_population_stream_register_reflux_kernel!(
    face_budget,
    parent_rho,
    parent_vel,
    parent_f,
    parent_obstacle,
    parent_kinetic_energy,
    cx_arr,
    cy_arr,
    cz_arr,
    w_arr,
    n_dirs_flow::Int32,
    alpha::Float32,
    has_kinetic_energy::Int32
)
    lx, ly, lz, pb_idx = @index(Global, NTuple)

    if !parent_obstacle[lx, ly, lz, pb_idx]
        corrected = false
        raw_residual = MVector{39, Float32}(undef)
        active_residual = MVector{39, Int32}(undef)
        @inbounds for kk in Int32(1):AMR_POP_STREAM_REGISTER_MAX_DIRS
            raw_residual[kk] = 0.0f0
            active_residual[kk] = Int32(0)
        end

        m00 = 0.0f0
        m01 = 0.0f0
        m02 = 0.0f0
        m03 = 0.0f0
        m11 = 0.0f0
        m12 = 0.0f0
        m13 = 0.0f0
        m22 = 0.0f0
        m23 = 0.0f0
        m33 = 0.0f0
        # SOTA #2 (2026-05-29): energy-row Gram entries + residual for the
        # 5-moment conservative reflux (e5 = |c|²). m0..3 4 capture cross terms.
        m04 = 0.0f0; m14 = 0.0f0; m24 = 0.0f0; m34 = 0.0f0; m44 = 0.0f0
        r0 = 0.0f0
        r1 = 0.0f0
        r2 = 0.0f0
        r3 = 0.0f0
        r4 = 0.0f0
        active_dirs = Int32(0)

        @inbounds for kk in Int32(1):AMR_POP_STREAM_REGISTER_MAX_DIRS
            if kk <= n_dirs_flow
                coarse_slot = amr_pop_stream_slot(kk, Int32(1))
                fine_slot = amr_pop_stream_slot(kk, Int32(2))
                count_slot = amr_pop_stream_slot(kk, Int32(3))
                stream_time = face_budget[lx, ly, lz, pb_idx, count_slot]
                if stream_time > 1.0f-12
                    residual =
                        (face_budget[lx, ly, lz, pb_idx, fine_slot] -
                         face_budget[lx, ly, lz, pb_idx, coarse_slot]) /
                        stream_time
                    raw_residual[kk] = residual
                    cxk = Float32(cx_arr[kk])
                    cyk = Float32(cy_arr[kk])
                    czk = Float32(cz_arr[kk])
                    wk = max(Float32(w_arr[kk]), 1.0f-12)
                    # Conservative AMR reflux: project the accumulated fine-minus-coarse
                    # population stream residual onto the hydrodynamic subspace. The
                    # weighted Gram solve gives the minimum-population-change correction
                    # that carries exactly the registered mass and momentum flux.
                    m00 += wk
                    m01 += wk * cxk
                    m02 += wk * cyk
                    m03 += wk * czk
                    m11 += wk * cxk * cxk
                    m12 += wk * cxk * cyk
                    m13 += wk * cxk * czk
                    m22 += wk * cyk * cyk
                    m23 += wk * cyk * czk
                    m33 += wk * czk * czk
                    # Energy-row (e5 = |c|²) Gram entries + residual.
                    c2k = cxk * cxk + cyk * cyk + czk * czk
                    m04 += wk * c2k
                    m14 += wk * cxk * c2k
                    m24 += wk * cyk * c2k
                    m34 += wk * czk * c2k
                    m44 += wk * c2k * c2k
                    r0 += residual
                    r1 += residual * cxk
                    r2 += residual * cyk
                    r3 += residual * czk
                    r4 += residual * c2k
                    active_residual[kk] = Int32(1)
                    active_dirs += Int32(1)
                end
            end
        end

        # SOTA #2: 5-moment (mass+mom+ENERGY) projection when enough active
        # directions are present to constrain the energy mode; else fall back to
        # the 4-moment solve (a4=0, energy not corrected — old behavior).
        local a0, ax, ay, az, a4, ok_projection
        if active_dirs >= Int32(5)
            a0, ax, ay, az, a4, ok_projection = solve_shell_moment5(
                m00, m01, m02, m03, m04,
                m11, m12, m13, m14,
                m22, m23, m24,
                m33, m34,
                m44,
                r0, r1, r2, r3, r4
            )
        else
            a0, ax, ay, az, ok4 = solve_shell_moment4(
                m00, m01, m02, m03,
                m11, m12, m13,
                m22, m23,
                m33,
                r0, r1, r2, r3
            )
            a4 = 0.0f0
            ok_projection = ok4
        end

        if ok_projection && active_dirs >= Int32(4)
            pos_limiter = 1.0f0
            @inbounds for kk in Int32(1):AMR_POP_STREAM_REGISTER_MAX_DIRS
                if kk <= n_dirs_flow
                    if active_residual[kk] == Int32(1)
                        cxk = Float32(cx_arr[kk])
                        cyk = Float32(cy_arr[kk])
                        czk = Float32(cz_arr[kk])
                        c2k = cxk * cxk + cyk * cyk + czk * czk
                        hydro_part = a0 + ax * cxk + ay * cyk + az * czk + a4 * c2k
                        df = alpha * max(Float32(w_arr[kk]), 1.0f-12) * hydro_part
                        if df < 0.0f0
                            f_now = parent_f[lx, ly, lz, pb_idx, kk]
                            pos_limiter = min(
                                pos_limiter,
                                max((f_now - 1.0f-10) / (-df), 0.0f0)
                            )
                        end
                    end
                end
            end
            pos_limiter = min(pos_limiter, 1.0f0)

            @inbounds for kk in Int32(1):AMR_POP_STREAM_REGISTER_MAX_DIRS
                if kk <= n_dirs_flow
                    if active_residual[kk] == Int32(1)
                        cxk = Float32(cx_arr[kk])
                        cyk = Float32(cy_arr[kk])
                        czk = Float32(cz_arr[kk])
                        c2k = cxk * cxk + cyk * cyk + czk * czk
                        hydro_part = a0 + ax * cxk + ay * cyk + az * czk + a4 * c2k
                        df = pos_limiter * alpha * max(Float32(w_arr[kk]), 1.0f-12) * hydro_part
                        if abs(df) > 1.0f-12
                            parent_f[lx, ly, lz, pb_idx, kk] =
                                max(parent_f[lx, ly, lz, pb_idx, kk] + df, 1.0f-10)
                            corrected = true
                        end
                    end
                end
            end
        end

        if corrected
            rho_after = 0.0f0
            jx_after = 0.0f0
            jy_after = 0.0f0
            jz_after = 0.0f0
            ekin_after = 0.0f0
            @inbounds for kk in Int32(1):AMR_POP_STREAM_REGISTER_MAX_DIRS
                if kk <= n_dirs_flow
                    fval = parent_f[lx, ly, lz, pb_idx, kk]
                    cxk = Float32(cx_arr[kk])
                    cyk = Float32(cy_arr[kk])
                    czk = Float32(cz_arr[kk])
                    rho_after += fval
                    jx_after += fval * cxk
                    jy_after += fval * cyk
                    jz_after += fval * czk
                    ekin_after += fval * (cxk * cxk + cyk * cyk + czk * czk)
                end
            end
            rho_after = max(rho_after, 0.01f0)
            parent_rho[lx, ly, lz, pb_idx] = rho_after
            parent_vel[lx, ly, lz, pb_idx, 1] = jx_after / rho_after
            parent_vel[lx, ly, lz, pb_idx, 2] = jy_after / rho_after
            parent_vel[lx, ly, lz, pb_idx, 3] = jz_after / rho_after
            if has_kinetic_energy == Int32(1)
                parent_kinetic_energy[lx, ly, lz, pb_idx] = ekin_after
            end
        end
    end
end

function apply_amr_population_stream_register_reflux!(
    parent_level::BlockLevel,
    parent_level_idx::Int,
    parent_f_current,
    parent_vel_current,
    cx_arr,
    cy_arr,
    cz_arr,
    w_arr;
    alpha::Float32=1.0f0,
    max_parent_level::Int=2
)
    alpha > 0.0f0 || return nothing
    parent_level_idx <= max_parent_level || return nothing
    amr_population_stream_register_allocated(parent_level) || return nothing
    n_parent_blocks = length(parent_level.active_block_coords)
    n_parent_blocks == 0 && return nothing
    backend = get_backend(parent_level.rho)
    kernel! = apply_amr_population_stream_register_reflux_kernel!(backend)
    kernel!(
        parent_level.amr_face_budget,
        parent_level.rho,
        parent_vel_current,
        parent_f_current,
        parent_level.obstacle,
        parent_level.kinetic_energy,
        cx_arr,
        cy_arr,
        cz_arr,
        w_arr,
        Int32(size(parent_f_current, 5)),
        alpha,
        length(parent_level.kinetic_energy) > 1 ? Int32(1) : Int32(0);
        ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_parent_blocks)
    )
    return nothing
end

@kernel function apply_amr_population_stream_xface_register_reflux_kernel!(
    face_budget,
    parent_rho,
    parent_vel,
    parent_f,
    parent_obstacle,
    parent_kinetic_energy,
    cx_arr,
    cy_arr,
    cz_arr,
    n_dirs_flow::Int32,
    alpha::Float32,
    has_kinetic_energy::Int32
)
    lx, ly, lz, pb_idx = @index(Global, NTuple)

    if !parent_obstacle[lx, ly, lz, pb_idx]
        corrected = false
        raw_residual = MVector{39, Float32}(undef)
        @inbounds for kk in Int32(1):AMR_POP_STREAM_REGISTER_MAX_DIRS
            raw_residual[kk] = 0.0f0
        end

        @inbounds for face_id in Int32(1):Int32(2)
            normal_sign = face_id == Int32(1) ? 1.0f0 : -1.0f0
            m00 = 0.0f0
            m01 = 0.0f0
            m11 = 0.0f0
            r0 = 0.0f0
            r1 = 0.0f0
            active_dirs = Int32(0)

            for kk in Int32(1):AMR_POP_STREAM_REGISTER_MAX_DIRS
                raw_residual[kk] = 0.0f0
                if kk <= n_dirs_flow
                    coarse_slot = amr_pop_stream_face_slot(face_id, kk, Int32(1))
                    fine_slot = amr_pop_stream_face_slot(face_id, kk, Int32(2))
                    count_slot = amr_pop_stream_face_slot(face_id, kk, Int32(3))
                    stream_time = face_budget[lx, ly, lz, pb_idx, count_slot]
                    if stream_time > 1.0f-12
                        residual =
                            (face_budget[lx, ly, lz, pb_idx, fine_slot] -
                             face_budget[lx, ly, lz, pb_idx, coarse_slot]) /
                            stream_time
                        raw_residual[kk] = residual
                        cn = normal_sign * Float32(cx_arr[kk])
                        m00 += 1.0f0
                        m01 += cn
                        m11 += cn * cn
                        r0 += residual
                        r1 += residual * cn
                        active_dirs += Int32(1)
                    end
                end
            end

            eps = 1.0f-12
            a00 = m00 + eps
            a11 = m11 + eps
            det = a00 * a11 - m01 * m01
            if active_dirs >= Int32(2) && abs(det) > eps
                inv_det = 1.0f0 / det
                a0 = (r0 * a11 - r1 * m01) * inv_det
                an = (a00 * r1 - m01 * r0) * inv_det

                pos_limiter = 1.0f0
                for kk in Int32(1):AMR_POP_STREAM_REGISTER_MAX_DIRS
                    if kk <= n_dirs_flow
                        raw = raw_residual[kk]
                        if raw != 0.0f0
                            cn = normal_sign * Float32(cx_arr[kk])
                            neutral_part = a0 + an * cn
                            df = alpha * (raw - neutral_part)
                            if df < 0.0f0
                                f_now = parent_f[lx, ly, lz, pb_idx, kk]
                                pos_limiter = min(
                                    pos_limiter,
                                    max((f_now - 1.0f-10) / (-df), 0.0f0)
                                )
                            end
                        end
                    end
                end
                pos_limiter = min(pos_limiter, 1.0f0)

                for kk in Int32(1):AMR_POP_STREAM_REGISTER_MAX_DIRS
                    if kk <= n_dirs_flow
                        raw = raw_residual[kk]
                        if raw != 0.0f0
                            cn = normal_sign * Float32(cx_arr[kk])
                            neutral_part = a0 + an * cn
                            df = pos_limiter * alpha * (raw - neutral_part)
                            if abs(df) > 1.0f-12
                                parent_f[lx, ly, lz, pb_idx, kk] =
                                    max(parent_f[lx, ly, lz, pb_idx, kk] + df, 1.0f-10)
                                corrected = true
                            end
                        end
                    end
                end
            end
        end

        if corrected
            rho_after = 0.0f0
            jx_after = 0.0f0
            jy_after = 0.0f0
            jz_after = 0.0f0
            ekin_after = 0.0f0
            @inbounds for kk in Int32(1):AMR_POP_STREAM_REGISTER_MAX_DIRS
                if kk <= n_dirs_flow
                    fval = parent_f[lx, ly, lz, pb_idx, kk]
                    cxk = Float32(cx_arr[kk])
                    cyk = Float32(cy_arr[kk])
                    czk = Float32(cz_arr[kk])
                    rho_after += fval
                    jx_after += fval * cxk
                    jy_after += fval * cyk
                    jz_after += fval * czk
                    ekin_after += fval * (cxk * cxk + cyk * cyk + czk * czk)
                end
            end
            rho_after = max(rho_after, 0.01f0)
            parent_rho[lx, ly, lz, pb_idx] = rho_after
            parent_vel[lx, ly, lz, pb_idx, 1] = jx_after / rho_after
            parent_vel[lx, ly, lz, pb_idx, 2] = jy_after / rho_after
            parent_vel[lx, ly, lz, pb_idx, 3] = jz_after / rho_after
            if has_kinetic_energy == Int32(1)
                parent_kinetic_energy[lx, ly, lz, pb_idx] = ekin_after
            end
        end
    end
end

function apply_amr_population_stream_xface_register_reflux!(
    parent_level::BlockLevel,
    parent_level_idx::Int,
    parent_f_current,
    parent_vel_current,
    cx_arr,
    cy_arr,
    cz_arr;
    alpha::Float32=1.0f0,
    max_parent_level::Int=2
)
    alpha > 0.0f0 || return nothing
    parent_level_idx <= max_parent_level || return nothing
    amr_population_stream_register_face_resolved_allocated(parent_level) || return nothing
    n_parent_blocks = length(parent_level.active_block_coords)
    n_parent_blocks == 0 && return nothing
    backend = get_backend(parent_level.rho)
    kernel! = apply_amr_population_stream_xface_register_reflux_kernel!(backend)
    kernel!(
        parent_level.amr_face_budget,
        parent_level.rho,
        parent_vel_current,
        parent_f_current,
        parent_level.obstacle,
        parent_level.kinetic_energy,
        cx_arr,
        cy_arr,
        cz_arr,
        Int32(size(parent_f_current, 5)),
        alpha,
        length(parent_level.kinetic_energy) > 1 ? Int32(1) : Int32(0);
        ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_parent_blocks)
    )
    return nothing
end

@kernel function apply_amr_population_stream_xface_highpass_register_reflux_kernel!(
    face_budget,
    parent_rho,
    parent_vel,
    parent_f,
    parent_obstacle,
    parent_kinetic_energy,
    parent_block_pointer,
    parent_map_x,
    parent_map_y,
    parent_map_z,
    cx_arr,
    cy_arr,
    cz_arr,
    block_size::Int32,
    parent_dim_x::Int32,
    parent_dim_y::Int32,
    parent_dim_z::Int32,
    n_dirs_flow::Int32,
    alpha::Float32,
    has_kinetic_energy::Int32
)
    lx, ly, lz, pb_idx = @index(Global, NTuple)

    if !parent_obstacle[lx, ly, lz, pb_idx]
        gx = (parent_map_x[pb_idx] - Int32(1)) * block_size + Int32(lx)
        gy = (parent_map_y[pb_idx] - Int32(1)) * block_size + Int32(ly)
        gz = (parent_map_z[pb_idx] - Int32(1)) * block_size + Int32(lz)

        corrected = false

        @inbounds for face_id in Int32(1):Int32(2)
            for kk in Int32(1):AMR_POP_STREAM_REGISTER_MAX_DIRS
                if kk <= n_dirs_flow
                    coarse_slot = amr_pop_stream_face_slot(face_id, kk, Int32(1))
                    fine_slot = amr_pop_stream_face_slot(face_id, kk, Int32(2))
                    count_slot = amr_pop_stream_face_slot(face_id, kk, Int32(3))
                    stream_time = face_budget[lx, ly, lz, pb_idx, count_slot]
                    if stream_time > 1.0f-12
                        residual =
                            (face_budget[lx, ly, lz, pb_idx, fine_slot] -
                             face_budget[lx, ly, lz, pb_idx, coarse_slot]) /
                            stream_time

                        local_sum = 0.0f0
                        local_count = Int32(0)
                        for oy in Int32(-1):Int32(1), oz in Int32(-1):Int32(1)
                            nlx, nly, nlz, nb, valid_parent = lookup_parent_cell(
                                parent_block_pointer,
                                parent_dim_x, parent_dim_y, parent_dim_z,
                                block_size,
                                gx, gy + oy, gz + oz
                            )
                            if valid_parent && !parent_obstacle[nlx, nly, nlz, nb]
                                nbr_stream_time = face_budget[nlx, nly, nlz, nb, count_slot]
                                if nbr_stream_time > 1.0f-12
                                    nbr_residual =
                                        (face_budget[nlx, nly, nlz, nb, fine_slot] -
                                         face_budget[nlx, nly, nlz, nb, coarse_slot]) /
                                        nbr_stream_time
                                    local_sum += nbr_residual
                                    local_count += Int32(1)
                                end
                            end
                        end

                        if local_count >= Int32(3)
                            local_mean = local_sum / Float32(local_count)
                            df = alpha * (residual - local_mean)
                            if abs(df) > 1.0f-12
                                f_now = parent_f[lx, ly, lz, pb_idx, kk]
                                if df < 0.0f0
                                    df = max(df, -(f_now - 1.0f-10))
                                end
                                if abs(df) > 1.0f-12
                                    parent_f[lx, ly, lz, pb_idx, kk] =
                                        max(f_now + df, 1.0f-10)
                                    corrected = true
                                end
                            end
                        end
                    end
                end
            end
        end

        if corrected
            rho_after = 0.0f0
            jx_after = 0.0f0
            jy_after = 0.0f0
            jz_after = 0.0f0
            ekin_after = 0.0f0
            @inbounds for kk in Int32(1):AMR_POP_STREAM_REGISTER_MAX_DIRS
                if kk <= n_dirs_flow
                    fval = parent_f[lx, ly, lz, pb_idx, kk]
                    cxk = Float32(cx_arr[kk])
                    cyk = Float32(cy_arr[kk])
                    czk = Float32(cz_arr[kk])
                    rho_after += fval
                    jx_after += fval * cxk
                    jy_after += fval * cyk
                    jz_after += fval * czk
                    ekin_after += fval * (cxk * cxk + cyk * cyk + czk * czk)
                end
            end
            rho_after = max(rho_after, 0.01f0)
            parent_rho[lx, ly, lz, pb_idx] = rho_after
            parent_vel[lx, ly, lz, pb_idx, 1] = jx_after / rho_after
            parent_vel[lx, ly, lz, pb_idx, 2] = jy_after / rho_after
            parent_vel[lx, ly, lz, pb_idx, 3] = jz_after / rho_after
            if has_kinetic_energy == Int32(1)
                parent_kinetic_energy[lx, ly, lz, pb_idx] = ekin_after
            end
        end
    end
end

function apply_amr_population_stream_xface_highpass_register_reflux!(
    parent_level::BlockLevel,
    parent_level_idx::Int,
    parent_f_current,
    parent_vel_current,
    cx_arr,
    cy_arr,
    cz_arr;
    alpha::Float32=0.25f0,
    max_parent_level::Int=2
)
    alpha > 0.0f0 || return nothing
    parent_level_idx <= max_parent_level || return nothing
    amr_population_stream_register_face_resolved_allocated(parent_level) || return nothing
    n_parent_blocks = length(parent_level.active_block_coords)
    n_parent_blocks == 0 && return nothing
    backend = get_backend(parent_level.rho)
    kernel! = apply_amr_population_stream_xface_highpass_register_reflux_kernel!(backend)
    kernel!(
        parent_level.amr_face_budget,
        parent_level.rho,
        parent_vel_current,
        parent_f_current,
        parent_level.obstacle,
        parent_level.kinetic_energy,
        parent_level.block_pointer,
        parent_level.map_x,
        parent_level.map_y,
        parent_level.map_z,
        cx_arr,
        cy_arr,
        cz_arr,
        Int32(BLOCK_SIZE),
        Int32(size(parent_level.block_pointer, 1)),
        Int32(size(parent_level.block_pointer, 2)),
        Int32(size(parent_level.block_pointer, 3)),
        Int32(size(parent_f_current, 5)),
        alpha,
        length(parent_level.kinetic_energy) > 1 ? Int32(1) : Int32(0);
        ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_parent_blocks)
    )
    return nothing
end

@kernel function accumulate_amr_population_stream_xface_global_moments_kernel!(
    face_budget,
    parent_obstacle,
    cx_arr,
    n_dirs_flow::Int32
)
    lx, ly, lz, pb_idx = @index(Global, NTuple)

    if !parent_obstacle[lx, ly, lz, pb_idx]
        @inbounds for face_id in Int32(1):Int32(2)
            normal_sign = face_id == Int32(1) ? 1.0f0 : -1.0f0
            slot_m00 = amr_pop_stream_xface_global_slot(face_id, Int32(1))
            slot_m01 = amr_pop_stream_xface_global_slot(face_id, Int32(2))
            slot_m11 = amr_pop_stream_xface_global_slot(face_id, Int32(3))
            slot_r0 = amr_pop_stream_xface_global_slot(face_id, Int32(4))
            slot_r1 = amr_pop_stream_xface_global_slot(face_id, Int32(5))
            for kk in Int32(1):AMR_POP_STREAM_REGISTER_MAX_DIRS
                if kk <= n_dirs_flow
                    coarse_slot = amr_pop_stream_face_slot(face_id, kk, Int32(1))
                    fine_slot = amr_pop_stream_face_slot(face_id, kk, Int32(2))
                    count_slot = amr_pop_stream_face_slot(face_id, kk, Int32(3))
                    stream_time = face_budget[lx, ly, lz, pb_idx, count_slot]
                    if stream_time > 1.0f-12
                        residual =
                            (face_budget[lx, ly, lz, pb_idx, fine_slot] -
                             face_budget[lx, ly, lz, pb_idx, coarse_slot]) /
                            stream_time
                        cn = normal_sign * Float32(cx_arr[kk])
                        Atomix.@atomic face_budget[1, 1, 1, 1, slot_m00] += 1.0f0
                        Atomix.@atomic face_budget[1, 1, 1, 1, slot_m01] += cn
                        Atomix.@atomic face_budget[1, 1, 1, 1, slot_m11] += cn * cn
                        Atomix.@atomic face_budget[1, 1, 1, 1, slot_r0] += residual
                        Atomix.@atomic face_budget[1, 1, 1, 1, slot_r1] += residual * cn
                    end
                end
            end
        end
    end
end

@kernel function apply_amr_population_stream_xface_global_register_reflux_kernel!(
    face_budget,
    parent_rho,
    parent_vel,
    parent_f,
    parent_obstacle,
    parent_kinetic_energy,
    cx_arr,
    cy_arr,
    cz_arr,
    n_dirs_flow::Int32,
    alpha::Float32,
    has_kinetic_energy::Int32
)
    lx, ly, lz, pb_idx = @index(Global, NTuple)

    if !parent_obstacle[lx, ly, lz, pb_idx]
        corrected = false

        @inbounds for face_id in Int32(1):Int32(2)
            normal_sign = face_id == Int32(1) ? 1.0f0 : -1.0f0
            m00 = face_budget[1, 1, 1, 1, amr_pop_stream_xface_global_slot(face_id, Int32(1))]
            m01 = face_budget[1, 1, 1, 1, amr_pop_stream_xface_global_slot(face_id, Int32(2))]
            m11 = face_budget[1, 1, 1, 1, amr_pop_stream_xface_global_slot(face_id, Int32(3))]
            r0 = face_budget[1, 1, 1, 1, amr_pop_stream_xface_global_slot(face_id, Int32(4))]
            r1 = face_budget[1, 1, 1, 1, amr_pop_stream_xface_global_slot(face_id, Int32(5))]

            eps = 1.0f-12
            a00 = m00 + eps
            a11 = m11 + eps
            det = a00 * a11 - m01 * m01
            if m00 >= 2.0f0 && abs(det) > eps
                inv_det = 1.0f0 / det
                a0 = (r0 * a11 - r1 * m01) * inv_det
                an = (a00 * r1 - m01 * r0) * inv_det

                for kk in Int32(1):AMR_POP_STREAM_REGISTER_MAX_DIRS
                    if kk <= n_dirs_flow
                        coarse_slot = amr_pop_stream_face_slot(face_id, kk, Int32(1))
                        fine_slot = amr_pop_stream_face_slot(face_id, kk, Int32(2))
                        count_slot = amr_pop_stream_face_slot(face_id, kk, Int32(3))
                        stream_time = face_budget[lx, ly, lz, pb_idx, count_slot]
                        if stream_time > 1.0f-12
                            residual =
                                (face_budget[lx, ly, lz, pb_idx, fine_slot] -
                                 face_budget[lx, ly, lz, pb_idx, coarse_slot]) /
                                stream_time
                            cn = normal_sign * Float32(cx_arr[kk])
                            neutral_part = a0 + an * cn
                            df = alpha * (residual - neutral_part)
                            if abs(df) > 1.0f-12
                                f_now = parent_f[lx, ly, lz, pb_idx, kk]
                                if df < 0.0f0
                                    df = max(df, -(f_now - 1.0f-10))
                                end
                                if abs(df) > 1.0f-12
                                    parent_f[lx, ly, lz, pb_idx, kk] =
                                        max(f_now + df, 1.0f-10)
                                    corrected = true
                                end
                            end
                        end
                    end
                end
            end
        end

        if corrected
            rho_after = 0.0f0
            jx_after = 0.0f0
            jy_after = 0.0f0
            jz_after = 0.0f0
            ekin_after = 0.0f0
            @inbounds for kk in Int32(1):AMR_POP_STREAM_REGISTER_MAX_DIRS
                if kk <= n_dirs_flow
                    fval = parent_f[lx, ly, lz, pb_idx, kk]
                    cxk = Float32(cx_arr[kk])
                    cyk = Float32(cy_arr[kk])
                    czk = Float32(cz_arr[kk])
                    rho_after += fval
                    jx_after += fval * cxk
                    jy_after += fval * cyk
                    jz_after += fval * czk
                    ekin_after += fval * (cxk * cxk + cyk * cyk + czk * czk)
                end
            end
            rho_after = max(rho_after, 0.01f0)
            parent_rho[lx, ly, lz, pb_idx] = rho_after
            parent_vel[lx, ly, lz, pb_idx, 1] = jx_after / rho_after
            parent_vel[lx, ly, lz, pb_idx, 2] = jy_after / rho_after
            parent_vel[lx, ly, lz, pb_idx, 3] = jz_after / rho_after
            if has_kinetic_energy == Int32(1)
                parent_kinetic_energy[lx, ly, lz, pb_idx] = ekin_after
            end
        end
    end
end

function apply_amr_population_stream_xface_global_register_reflux!(
    parent_level::BlockLevel,
    parent_level_idx::Int,
    parent_f_current,
    parent_vel_current,
    cx_arr,
    cy_arr,
    cz_arr;
    alpha::Float32=1.0f0,
    max_parent_level::Int=2
)
    alpha > 0.0f0 || return nothing
    parent_level_idx <= max_parent_level || return nothing
    amr_population_stream_xface_global_register_allocated(parent_level) || return nothing
    n_parent_blocks = length(parent_level.active_block_coords)
    n_parent_blocks == 0 && return nothing
    backend = get_backend(parent_level.rho)

    accum_kernel! = accumulate_amr_population_stream_xface_global_moments_kernel!(backend)
    accum_kernel!(
        parent_level.amr_face_budget,
        parent_level.obstacle,
        cx_arr,
        Int32(size(parent_f_current, 5));
        ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_parent_blocks)
    )
    KernelAbstractions.synchronize(backend)

    apply_kernel! = apply_amr_population_stream_xface_global_register_reflux_kernel!(backend)
    apply_kernel!(
        parent_level.amr_face_budget,
        parent_level.rho,
        parent_vel_current,
        parent_f_current,
        parent_level.obstacle,
        parent_level.kinetic_energy,
        cx_arr,
        cy_arr,
        cz_arr,
        Int32(size(parent_f_current, 5)),
        alpha,
        length(parent_level.kinetic_energy) > 1 ? Int32(1) : Int32(0);
        ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_parent_blocks)
    )
    return nothing
end

@kernel function accumulate_amr_gpu_x_face_budget_kernel!(
    face_budget,
    parent_rho,
    parent_vel,
    parent_obstacle,
    parent_map_x,
    parent_map_y,
    parent_map_z,
    child_block_pointer,
    child_obstacle,
    child_rho,
    child_vel,
    block_size::Int32,
    child_dim_x::Int32,
    child_dim_y::Int32,
    child_dim_z::Int32,
    dt_weight::Float32,
    cs2_flow::Float32,
    side::Int32
)
    lx, ly, lz, pb_idx = @index(Global, NTuple)
    bs = block_size

    gx = (parent_map_x[pb_idx] - Int32(1)) * bs + Int32(lx)
    gy = (parent_map_y[pb_idx] - Int32(1)) * bs + Int32(ly)
    gz = (parent_map_z[pb_idx] - Int32(1)) * bs + Int32(lz)

    active_cell = !parent_obstacle[lx, ly, lz, pb_idx] &&
                  count_fine_children(child_block_pointer, child_dim_x, child_dim_y, child_dim_z,
                                      bs, gx, gy, gz) == Int32(0)

    if active_cell
        @inbounds for face_slot in Int32(0):Int32(1)
            sign_i = face_slot == Int32(0) ? Int32(1) : Int32(-1)
            if coarse_cell_has_fine_children(
                child_block_pointer, child_dim_x, child_dim_y, child_dim_z,
                bs, gx + sign_i, gy, gz
            )
                base = sign_i > Int32(0) ? Int32(0) : Int32(6)
                sign_f = Float32(sign_i)

                if side == Int32(1)
                    rho_p = max(parent_rho[lx, ly, lz, pb_idx], 0.01f0)
                    ux_p = parent_vel[lx, ly, lz, pb_idx, 1]
                    un_p = sign_f * ux_p
                    mass_flux = rho_p * un_p
                    mom_flux = rho_p * un_p * un_p + cs2_flow * rho_p
                    face_budget[lx, ly, lz, pb_idx, base + Int32(1)] += dt_weight * mass_flux
                    face_budget[lx, ly, lz, pb_idx, base + Int32(3)] += dt_weight * mom_flux
                    face_budget[lx, ly, lz, pb_idx, base + Int32(5)] += dt_weight
                else
                    face_fx0 = sign_i > Int32(0) ? Int32(2) * gx + Int32(1) : Int32(2) * gx - Int32(2)
                    face_fy0 = Int32(2) * gy - Int32(1)
                    face_fz0 = Int32(2) * gz - Int32(1)
                    mass_sum = 0.0f0
                    mom_sum = 0.0f0
                    n_valid = Int32(0)
                    for sy in Int32(0):Int32(1), sz in Int32(0):Int32(1)
                        clx, cly, clz, cb, valid_child = lookup_child_cell(
                            child_block_pointer, child_dim_x, child_dim_y, child_dim_z,
                            bs, face_fx0, face_fy0 + sy, face_fz0 + sz
                        )
                        if valid_child && !child_obstacle[clx, cly, clz, cb]
                            rho_c = max(child_rho[clx, cly, clz, cb], 0.01f0)
                            ux_c = child_vel[clx, cly, clz, cb, 1]
                            un_c = sign_f * ux_c
                            mass_sum += rho_c * un_c
                            mom_sum += rho_c * un_c * un_c + cs2_flow * rho_c
                            n_valid += Int32(1)
                        end
                    end
                    if n_valid == Int32(4)
                        inv_n = 0.25f0
                        face_budget[lx, ly, lz, pb_idx, base + Int32(2)] += dt_weight * mass_sum * inv_n
                        face_budget[lx, ly, lz, pb_idx, base + Int32(4)] += dt_weight * mom_sum * inv_n
                        face_budget[lx, ly, lz, pb_idx, base + Int32(6)] += dt_weight
                    end
                end
            end
        end
    end
end

function accumulate_amr_gpu_x_face_budget!(
    parent_level::BlockLevel,
    child_level::BlockLevel;
    side::Symbol,
    dt_weight::Float32,
    parent_vel_current=nothing,
    child_vel_current=nothing
)
    amr_gpu_face_budget_allocated(parent_level) || return nothing
    side === :parent || side === :child || error("AMR GPU face budget side must be :parent or :child")
    n_parent_blocks = length(parent_level.active_block_coords)
    n_parent_blocks == 0 && return nothing
    backend = get_backend(parent_level.rho)
    kernel! = accumulate_amr_gpu_x_face_budget_kernel!(backend)
    kernel!(
        parent_level.amr_face_budget,
        parent_level.rho,
        parent_vel_current === nothing ? parent_level.vel : parent_vel_current,
        parent_level.obstacle,
        parent_level.map_x,
        parent_level.map_y,
        parent_level.map_z,
        child_level.block_pointer,
        child_level.obstacle,
        child_level.rho,
        child_vel_current === nothing ? child_level.vel : child_vel_current,
        Int32(BLOCK_SIZE),
        Int32(size(child_level.block_pointer, 1)),
        Int32(size(child_level.block_pointer, 2)),
        Int32(size(child_level.block_pointer, 3)),
        dt_weight,
        Float32(CS2_PHYSICS),
        side === :parent ? Int32(1) : Int32(2);
        ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_parent_blocks)
    )
    return nothing
end

@kernel function apply_amr_gpu_x_face_budget_reflux_kernel!(
    face_budget,
    parent_rho,
    parent_vel,
    parent_f,
    parent_obstacle,
    parent_kinetic_energy,
    child_block_pointer,
    parent_map_x,
    parent_map_y,
    parent_map_z,
    cx_arr,
    cy_arr,
    cz_arr,
    w_arr,
    block_size::Int32,
    child_dim_x::Int32,
    child_dim_y::Int32,
    child_dim_z::Int32,
    n_dirs_flow::Int32,
    cs2_flow::Float32,
    shift_x::Float32,
    shift_y::Float32,
    shift_z::Float32,
    alpha::Float32,
    flux_du_cap_abs::Float32,
    flux_du_cap_rel::Float32,
    momflux_stress_cap_abs::Float32,
    momflux_stress_cap_rel::Float32,
    has_kinetic_energy::Int32
)
    lx, ly, lz, pb_idx = @index(Global, NTuple)
    bs = block_size

    gx = (parent_map_x[pb_idx] - Int32(1)) * bs + Int32(lx)
    gy = (parent_map_y[pb_idx] - Int32(1)) * bs + Int32(ly)
    gz = (parent_map_z[pb_idx] - Int32(1)) * bs + Int32(lz)

    do_apply = !parent_obstacle[lx, ly, lz, pb_idx] &&
               count_fine_children(child_block_pointer, child_dim_x, child_dim_y, child_dim_z,
                                   bs, gx, gy, gz) == Int32(0)

    shell_face_count = Int32(0)
    x_sign = Int32(0)
    if do_apply
        @inbounds for face_id in Int32(0):Int32(5)
            dx = face_id == Int32(0) ? Int32(1) :
                 face_id == Int32(1) ? Int32(-1) : Int32(0)
            dy = face_id == Int32(2) ? Int32(1) :
                 face_id == Int32(3) ? Int32(-1) : Int32(0)
            dz = face_id == Int32(4) ? Int32(1) :
                 face_id == Int32(5) ? Int32(-1) : Int32(0)
            if coarse_cell_has_fine_children(
                child_block_pointer, child_dim_x, child_dim_y, child_dim_z,
                bs, gx + dx, gy + dy, gz + dz
            )
                shell_face_count += Int32(1)
                if dy == Int32(0) && dz == Int32(0)
                    x_sign = dx
                end
            end
        end
    end
    do_apply = do_apply && shell_face_count == Int32(1) && x_sign != Int32(0)

    if do_apply
        base = x_sign > Int32(0) ? Int32(0) : Int32(6)
        parent_time = face_budget[lx, ly, lz, pb_idx, base + Int32(5)]
        child_time = face_budget[lx, ly, lz, pb_idx, base + Int32(6)]
        do_apply = parent_time > 0.0f0 && child_time > 0.0f0
    end

    if do_apply
    parent_mass_rate = face_budget[lx, ly, lz, pb_idx, base + Int32(1)] / parent_time
    child_mass_rate = face_budget[lx, ly, lz, pb_idx, base + Int32(2)] / child_time
    parent_mom_rate = face_budget[lx, ly, lz, pb_idx, base + Int32(3)] / parent_time
    child_mom_rate = face_budget[lx, ly, lz, pb_idx, base + Int32(4)] / child_time
    dmass_rate = parent_mass_rate - child_mass_rate
    dmom_rate = parent_mom_rate - child_mom_rate

    rho = max(parent_rho[lx, ly, lz, pb_idx], 0.01f0)
    ux = parent_vel[lx, ly, lz, pb_idx, 1]
    u_ref = max(abs(ux), 1.0f-3)
    mass_cap = rho * max(flux_du_cap_abs, flux_du_cap_rel * u_ref)
    mom_scale = max(max(abs(parent_mom_rate), abs(child_mom_rate)), cs2_flow * rho)
    mom_cap = max(momflux_stress_cap_abs, momflux_stress_cap_rel * mom_scale)

    target_mass = -alpha * clamp(dmass_rate, -mass_cap, mass_cap)
    target_mom = -alpha * clamp(dmom_rate, -mom_cap, mom_cap)
    do_projection = abs(target_mass) > 1.0f-12 || abs(target_mom) > 1.0f-12

    if do_projection
    sign_f = Float32(x_sign)
    m0 = 0.0f0
    m1 = 0.0f0
    m2 = 0.0f0
    m3 = 0.0f0
    m4 = 0.0f0
    @inbounds for kk in 1:39
        if kk <= n_dirs_flow
            cn = sign_f * (Float32(cx_arr[kk]) - shift_x)
            if cn > 0.0f0
                wk = Float32(w_arr[kk])
                cn2 = cn * cn
                m0 += wk
                m1 += wk * cn
                m2 += wk * cn2
                m3 += wk * cn2 * cn
                m4 += wk * cn2 * cn2
            end
        end
    end

    r0 = 0.0f0
    r1 = target_mass
    r2 = target_mom
    det =
        m0 * (m2 * m4 - m3 * m3) -
        m1 * (m1 * m4 - m3 * m2) +
        m2 * (m1 * m3 - m2 * m2)
    if abs(det) > 1.0f-12

    a = (
        r0 * (m2 * m4 - m3 * m3) -
        m1 * (r1 * m4 - m3 * r2) +
        m2 * (r1 * m3 - m2 * r2)
    ) / det
    b = (
        m0 * (r1 * m4 - m3 * r2) -
        r0 * (m1 * m4 - m3 * m2) +
        m2 * (m1 * r2 - r1 * m2)
    ) / det
    c = (
        m0 * (m2 * r2 - r1 * m3) -
        m1 * (m1 * r2 - r1 * m2) +
        r0 * (m1 * m3 - m2 * m2)
    ) / det

    pos_limiter = 1.0f0
    @inbounds for kk in 1:39
        if kk <= n_dirs_flow
            cn = sign_f * (Float32(cx_arr[kk]) - shift_x)
            if cn > 0.0f0
                wk = Float32(w_arr[kk])
                df = wk * (a + b * cn + c * cn * cn)
                if df < 0.0f0
                    f_now = parent_f[lx, ly, lz, pb_idx, kk]
                    pos_limiter = min(pos_limiter, max((f_now - 1.0f-10) / (-df), 0.0f0))
                end
            end
        end
    end
    pos_limiter = min(pos_limiter, 1.0f0)

    rho_after = 0.0f0
    jx_after = 0.0f0
    jy_after = 0.0f0
    jz_after = 0.0f0
    ekin_after = 0.0f0
    @inbounds for kk in 1:39
        if kk <= n_dirs_flow
            cxk = Float32(cx_arr[kk])
            cn = sign_f * (cxk - shift_x)
            if cn > 0.0f0
                fval = parent_f[lx, ly, lz, pb_idx, kk]
                wk = Float32(w_arr[kk])
                df = pos_limiter * wk * (a + b * cn + c * cn * cn)
                fval = max(fval + df, 1.0f-10)
                parent_f[lx, ly, lz, pb_idx, kk] = fval
            end
        end
    end

    # (2026-06-05) The plane-symmetry FIX 1 that was mistakenly pasted here was in a
    # DEAD never-launched kernel (apply_amr_gpu_x_face_budget_reflux_kernel!) → it
    # never ran. Removed. The correct, engaged version lives in
    # restrict_child_state_to_parent_kernel! just before its final shell macro write.

    @inbounds for kk in 1:39
        if kk <= n_dirs_flow
            cxk = Float32(cx_arr[kk])
            cyk = Float32(cy_arr[kk])
            czk = Float32(cz_arr[kk])
            fval = parent_f[lx, ly, lz, pb_idx, kk]
            rho_after += fval
            jx_after += fval * cxk
            jy_after += fval * cyk
            jz_after += fval * czk
            cxu = cxk - shift_x
            cyu = cyk - shift_y
            czu = czk - shift_z
            ekin_after += fval * (cxu * cxu + cyu * cyu + czu * czu)
        end
    end

    rho_after = max(rho_after, 0.01f0)
    parent_rho[lx, ly, lz, pb_idx] = rho_after
    parent_vel[lx, ly, lz, pb_idx, 1] = jx_after / rho_after
    parent_vel[lx, ly, lz, pb_idx, 2] = jy_after / rho_after
    parent_vel[lx, ly, lz, pb_idx, 3] = jz_after / rho_after
    if has_kinetic_energy == Int32(1)
        parent_kinetic_energy[lx, ly, lz, pb_idx] = ekin_after
    end
    end
    end
    end
end

function apply_amr_gpu_x_face_budget_reflux!(
    parent_level::BlockLevel,
    child_level::BlockLevel,
    parent_level_idx::Int,
    parent_f_current,
    parent_vel_current,
    cx_arr,
    cy_arr,
    cz_arr,
    w_arr;
    alpha::Float32=0.10f0,
    max_parent_level::Int=1
)
    alpha > 0.0f0 || return nothing
    parent_level_idx <= max_parent_level || return nothing
    amr_gpu_face_budget_allocated(parent_level) || return nothing
    n_parent_blocks = length(parent_level.active_block_coords)
    n_parent_blocks == 0 && return nothing
    backend = get_backend(parent_level.rho)
    kernel! = apply_amr_gpu_x_face_budget_reflux_kernel!(backend)
    kernel!(
        parent_level.amr_face_budget,
        parent_level.rho,
        parent_vel_current,
        parent_f_current,
        parent_level.obstacle,
        parent_level.kinetic_energy,
        child_level.block_pointer,
        parent_level.map_x,
        parent_level.map_y,
        parent_level.map_z,
        cx_arr,
        cy_arr,
        cz_arr,
        w_arr,
        Int32(BLOCK_SIZE),
        Int32(size(child_level.block_pointer, 1)),
        Int32(size(child_level.block_pointer, 2)),
        Int32(size(child_level.block_pointer, 3)),
        Int32(size(parent_f_current, 5)),
        Float32(CS2_PHYSICS),
        Float32(LATTICE_SHIFT_X),
        Float32(LATTICE_SHIFT_Y),
        Float32(LATTICE_SHIFT_Z),
        alpha,
        Float32(AMR_SHELL_FLUX_DU_CAP_ABS),
        Float32(AMR_SHELL_FLUX_DU_CAP_REL),
        Float32(AMR_SHELL_MOMFLUX_STRESS_CAP_ABS),
        Float32(AMR_SHELL_MOMFLUX_STRESS_CAP_REL),
        Int32(length(parent_level.kinetic_energy) > 1 ? 1 : 0);
        ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_parent_blocks)
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

@inline function project_second_order_stress_mode(
    wk::Float32,
    cx::Float32, cy::Float32, cz::Float32,
    cs2::Float32,
    pxx::Float32, pyy::Float32, pzz::Float32,
    pxy::Float32, pxz::Float32, pyz::Float32
)
    # AMR restriction/shell reconstruction should carry the deviatoric
    # viscous stress.  The isotropic trace of Π_neq is an energy/bulk mode; in
    # the current athermal flow solve it is not represented by a consistent
    # thermal population and appears as AMR-locked Mach/pressure ringing.
    tr = (pxx + pyy + pzz) * (1.0f0 / 3.0f0)
    pxx_d = pxx - tr
    pyy_d = pyy - tr
    pzz_d = pzz - tr
    qxx = cx * cx - cs2
    qyy = cy * cy - cs2
    qzz = cz * cz - cs2
    qxy = cx * cy
    qxz = cx * cz
    qyz = cy * cz
    contraction = qxx * pxx_d + qyy * pyy_d + qzz * pzz_d +
                  2.0f0 * (qxy * pxy + qxz * pxz + qyz * pyz)
    return wk * contraction / (2.0f0 * cs2 * cs2)
end

# Restrict covered parent cells from the resolved child state itself.
# The previous path averaged child macros and then rebuilt parent f/g from a
# reduced macro state, which discarded part of the fine-level non-equilibrium
# content and left visible AMR seams. Here we average the child populations
# directly, then recover the parent macros from that restricted population state.
#
# Lagrava-consistent f_neq filter (Lagrava et al., JCP 2012, Eq. 33):
# Instead of deriving f_neq from averaged populations minus a parent equilibrium,
# we compute f_neq_k at each child cell independently (using each child's own
# Ï, u) and then spatially average those f_neq_k over the 2Ã—2Ã—2 children that
# the parent covers. This is the discrete analogue of the box-low-pass filter
# Lagrava applies to f_neq on fineâ†’coarse decimation, and it removes an O(uÂ²Â·Î´uÂ²)
# nonlinearity error that the "averaged-f minus parent-eq" path silently added.
#
# Entropic mirror decomposition (BÃ¶sch/Chikatamarla/Karlin KBC):
# When the collision operator is entropic, the averaged f_neq is mirror-decomposed
# into s_k = Â½(fÌ„_neq_k + fÌ„_neq_kÌ„) (mirror-symmetric, viscous stress) and
# h_k = Â½(fÌ„_neq_k âˆ’ fÌ„_neq_kÌ„) (mirror-antisymmetric, higher-order). Only s is
# rescaled by Ï„_parent/Ï„_child â€” the hydrodynamic (shear) relaxation ratio â€”
# because h relaxes at the cell-dependent entropic rate 2Â·Î²Â·Î± that is not
# meaningful to propagate upscale. Dropping h at the interface avoids the
# checkerboard mode that the viscous Ï„-scaling would otherwise excite.
@kernel function restrict_child_state_to_parent_kernel!(
    parent_rho,
    parent_vel,
    parent_temperature,
    parent_kinetic_energy,
    parent_f_current,
    parent_g_current,
    parent_pi_neq,
    parent_map_x,
    parent_map_y,
    parent_map_z,
    child_block_pointer,
    child_obstacle,
    child_wall_dist,
    child_rho,
    child_vel,
    child_temperature,
    child_kinetic_energy,
    child_f_current,
    child_g_current,
    child_pi_neq,
    cx_arr,
    cy_arr,
    cz_arr,
    w_arr,
    opp_arr,
    block_size::Int32,
    child_dim_x::Int32,
    child_dim_y::Int32,
    child_dim_z::Int32,
    n_dirs_flow::Int32,
    n_dirs_thermal::Int32,
    restrict_thermal::Int32,
    reset_pi_neq::Int32,
    preserve_pi_neq::Int32,
    # PHASE 2b register diet (2026-06-12): these were RUNTIME Int32 flags, so
    # the D3Q39 13-moment machinery + entropic mirror paths were COMPILED INTO
    # the kernel even on pure-D3Q19 runs (dead-but-compiled register poison —
    # the ef261cb lesson at kernel scale). As Val type params the dead
    # lattice's branches constant-fold away; Int32 shims in the body keep the
    # diff minimal.
    ::Val{LATTICE_IS_D3Q39},
    flow_is_compressible::Int32,
    ::Val{FRAPOLLI_MODE},
    ::Val{MIRROR_ENTROPIC},
    cs2_flow::Float32,
    # SOTA A3 — thermodynamic EOS R = (γ-1)·cv. MUST match the value the
    # collision kernel passes as `eos_R_flow` so the restriction f_eq solves
    # use the IDENTICAL equilibrium across the AMR interface (no spurious p/ρ
    # step at T≠1). Sentinel -1 ⇒ cs² fallback (legacy linearized EOS); at
    # T=1 the eos_R term vanishes ⇒ freestream/non-thermodynamic byte-identical.
    eos_R_flow::Float32,
    cs2_thermal::Float32,
    cv_thermal::Float32,
    shift_x::Float32,
    shift_y::Float32,
    shift_z::Float32,
    tau_ratio_flow::Float32,
    tau_ratio_thermal::Float32,
    h_scale_restrict::Float32,
    # Pass 265 (Tier 1 #1e): per-cell moment-correction pass count.
    correction_passes::Int32,
    # Pass 265 (Tier 1 #1e, complementary). When 1, exact-correct f_eq
    # mass+momentum to (rho_parent, j_parent) via 4×4 Gram before
    # adding the stress part. Closes the Newton-solver tolerance gap.
    restriction_conservative_feq::Int32,
    # ── SOTA P1: conservative ρE restriction across AMR ─────────────────
    # When enabled (and Frapolli thermal is active), the 4-moment correction
    # below is upgraded to a 5-moment correction that ALSO enforces parent
    # kinetic energy E_kin_parent = (1/8)·Σ_children Σ_k f_k·|c|². Uses the
    # closed-form `apply_5momentum_correction_d3q39!` in physics_utils.jl.
    restriction_conservative_rhoe::Int32,
    # ── B1 Pillar 2: 13-moment-exact (mortar-element) restriction ───────
    # When enabled (D3Q39 only), the entire feq + stress-carrier construction
    # is replaced by a single solve that drives ALL 13 Hermite moments
    # (mass + 3·momentum + 6·stress + 3·energy-flux) of the parent f to
    # the τ-rescaled averaged child moments, to Float32 round-off. Strict
    # conservation: each fine→coarse hop preserves every moment to ~1e-7.
    # See `apply_13moment_restriction_d3q39!` in physics_utils.jl and
    # SOTA_ROADMAP.md "B1 — Mortar-element AMR coupling" pillar 2.
    restriction_conservative_13mom::Int32,
    # ── B3 LIVE (2026-05-28) — Restriction-side FP64 halo arithmetic ────
    # Symmetric companion to the prolongation-side FP64 promotion in
    # `interpolate_macro_quartic_5pt_fp64`. When 1, the 8-child
    # accumulation + moment computation (rho, j, E_kin) runs in Float64
    # internally — eliminates the ~8-50× FP32-ULP accumulation error
    # that bounds the seam residual on the COARSE side of every AMR
    # interface. Storage stays Float32 (zero VRAM impact); the
    # downstream 13-mom Newton solver still operates on Float32 targets
    # (cheaper, and the target accuracy is already at Float32-storage
    # resolution after the demotion). Off by default — opt in via
    # `refinement.fp64_halo: true` (same flag as the prolongation side).
    use_fp64_halo::Int32,
    # ── SYMMETRY-PLANE FIX (2026-06-04, FIX 1 — DOMINANT) ───────────────
    # The restriction OVERWRITES parent (incl. L1) plane-cell velocity every
    # coarse step, AFTER the flow kernel's plane-symmetrize — re-injecting an
    # asymmetric u_y that compounds (the runaway max|u_y| 7e-3→0.28). When
    # sym_y_plane==1 and this parent cell is on a free-slip y-plane (global
    # y==1 or y==parent_ny_global), symmetrize the reconstructed parent_f over
    # cy-pairs so Σf·cy=0 to round-off → parent u_y=0 exactly. Gated on the
    # host (SYMMETRIC_ANALYSIS && !GROUND); default 0 ⇒ byte-identical otherwise.
    sym_y_plane::Int32,
    parent_ny_global::Int32,
    # ── GOLD-STANDARD STRICT CONSERVATION (2026-06-17) — D3Q19 4-moment ──
    # The D3Q19 BGK reconstruction (feq + τ·fneq) is mass/momentum-exact in
    # exact arithmetic, but the post-floor max(…,1e-10) clip leaves a ~1e-7/cell
    # mass deficit. When 1 (D3Q19 only, full-coverage parents), a 4-moment
    # correction δf[k] = w_k·(a0 + a1·cx' + a2·cy' + a3·cz') drives Σf→ρ_parent
    # and Σ(c')·f→ρ·u_rel to FP32 round-off, via the existing solve_shell_moment4
    # Gram solve. Orthogonal to the traceless stress basis by D3Q19 isotropy →
    # does not perturb shear transport. Scalar-only locals (one MVector{19} stack
    # buffer); default 0 ⇒ byte-identical. See memory/d3q19-amr-continuity-anatomy.
    restriction_conservative_d3q19::Int32,
    # 100-MLUPS item 2 (2026-06-12): covered-block banding. `nothing` ⇒ dense
    # legacy launch (4th ndrange dim = parent block index); a device Int32
    # list ⇒ banded launch (4th dim = list position, remapped below). The
    # Nothing/array dispatch is compile-time (the nothing-singleton pattern).
    covered_list,
    # ── ET-AMR (Esoteric Pull) restriction (STAGE 4, 2026-06-18) ────────────
    # When child_is_et==1, child_f_current is in the PERMUTED Esoteric layout —
    # each child cell's 19 conventional populations are ET-DECODED (interior:
    # leader from own slot, opp from the +c_kp neighbour via child_block_pointer)
    # at child_phase before accumulation. When parent_is_et==1, the reconstructed
    # 19 parent populations are ET-ENCODED (leader→+c neighbour, opp→own) at
    # parent_phase instead of written conventionally. Both Val-gated so the non-ET
    # launch is byte-identical (the existing _ET_RESTRICT_VAL pattern). Phases are
    # the child/parent NEXT-load parities (verified empirically — STAGE 4a).
    ::Val{CHILD_IS_ET},
    ::Val{PARENT_IS_ET},
    child_phase::Int32,
    parent_phase::Int32,
    et_pair_kp,
    et_pair_km,
    # parent block grid for the ET parent-encode +c-neighbour scatter
    parent_block_pointer_for_encode,
    parent_dim_x_enc::Int32,
    parent_dim_y_enc::Int32,
    parent_dim_z_enc::Int32,
) where {LATTICE_IS_D3Q39, FRAPOLLI_MODE, MIRROR_ENTROPIC, CHILD_IS_ET, PARENT_IS_ET}
    # Val→Int32 shims: the body's flag tests below constant-fold under the
    # type params, pruning the dead lattice's code paths entirely.
    lattice_is_d3q39 = LATTICE_IS_D3Q39 ? Int32(1) : Int32(0)
    frapolli_mode = FRAPOLLI_MODE ? Int32(1) : Int32(0)
    mirror_decompose_entropic = MIRROR_ENTROPIC ? Int32(1) : Int32(0)
    lx, ly, lz, pb_raw = @index(Global, NTuple)
    pb_idx = covered_list === nothing ? Int32(pb_raw) :
             @inbounds(covered_list[pb_raw])
    bs = block_size

    pbx = parent_map_x[pb_idx]
    pby = parent_map_y[pb_idx]
    pbz = parent_map_z[pb_idx]

    child_base_x = Int32(2) * pbx - Int32(1)
    child_base_y = Int32(2) * pby - Int32(1)
    child_base_z = Int32(2) * pbz - Int32(1)

    # Conservative restriction is decided per coarse cell. A parent-block gate
    # leaves fully covered cells stale along refinement frontiers whenever one
    # neighboring child block is absent.
    cell_has_full_child_support = true
    @inbounds for sx in Int32(0):Int32(1), sy in Int32(0):Int32(1), sz in Int32(0):Int32(1)
        fx = Int32(2) * (Int32(lx) - Int32(1)) + Int32(1) + sx
        fy = Int32(2) * (Int32(ly) - Int32(1)) + Int32(1) + sy
        fz = Int32(2) * (Int32(lz) - Int32(1)) + Int32(1) + sz

        cbx = child_base_x + div(fx - Int32(1), bs)
        cby = child_base_y + div(fy - Int32(1), bs)
        cbz = child_base_z + div(fz - Int32(1), bs)
        if cbx < Int32(1) || cby < Int32(1) || cbz < Int32(1) ||
           cbx > child_dim_x || cby > child_dim_y || cbz > child_dim_z ||
           child_block_pointer[cbx, cby, cbz] == Int32(0)
            cell_has_full_child_support = false
        end
    end

    if cell_has_full_child_support
        # Partial restriction has been rolled back, so the live path is a full
        # overwrite whenever all 8 children exist.
        frac = 1.0f0
        one_m_frac = 0.0f0

        flow_avg = MVector{39, Float32}(undef)
        thermal_avg = MVector{39, Float32}(undef)
        @inbounds for k in 1:39
            flow_avg[k] = 0.0f0
            thermal_avg[k] = 0.0f0
        end
        # ET child decode scratch (only used when CHILD_IS_ET).
        cfloc = MVector{19, Float32}(undef)
        # ★ ET-AMR (STAGE 4): the child's Esoteric f, decoded per-direction, is the
        # NEXT-load (post-STREAM) state — its populations are shifted one cell per
        # direction, so Σ f·c gives the WRONG per-cell macros at GRADIENTS (the
        # box-edge ring). The CORRECT current-at-cell macros are the ones the child
        # ET step already computed + stored in child_rho/child_vel/child_pi_neq
        # (from the loaded pops, BEFORE the store shifted them). So for ET we average
        # the stored child MACROS directly and override rho/j/Π below.
        sum_crho = 0.0f0; sum_cjx = 0.0f0; sum_cjy = 0.0f0; sum_cjz = 0.0f0

        # B3 (2026-05-28) — Float64 accumulators for the 8-child reduction.
        # The dominant FP32-ULP error in restriction is summing 8 Float32
        # populations into a Float32 register (~8× ULP) + summing 39
        # directions to get rho/jx/jy/jz (~39× ULP) → ~50× FP32 ULP =
        # ~6e-6 on a unit ρ. Accumulating in Float64 and demoting once at
        # the end drops this to the FP32 storage limit (~1.2e-7). The
        # downstream Newton solver and moment correction stay Float32 —
        # their accuracy is already storage-bound after the demotion.
        flow_avg_f64 = MVector{39, Float64}(undef)
        thermal_avg_f64 = MVector{39, Float64}(undef)
        if use_fp64_halo == Int32(1)
            @inbounds for k in 1:39
                flow_avg_f64[k] = 0.0
                thermal_avg_f64[k] = 0.0
            end
        end

        sum_temp = 0.0f0
        sum_ekin = 0.0f0
        sum_pxx = 0.0f0
        sum_pyy = 0.0f0
        sum_pzz = 0.0f0
        sum_pxy = 0.0f0
        sum_pxz = 0.0f0
        sum_pyz = 0.0f0
        # B3 Float64 sibling accumulators for scalar sums.
        sum_temp_f64 = 0.0
        sum_ekin_f64 = 0.0
        sum_pxx_f64 = 0.0
        sum_pyy_f64 = 0.0
        sum_pzz_f64 = 0.0
        sum_pxy_f64 = 0.0
        sum_pxz_f64 = 0.0
        sum_pyz_f64 = 0.0
        fluid_children = Int32(0)

        @inbounds for sx in Int32(0):Int32(1), sy in Int32(0):Int32(1), sz in Int32(0):Int32(1)
            fx = Int32(2) * (Int32(lx) - Int32(1)) + Int32(1) + sx
            fy = Int32(2) * (Int32(ly) - Int32(1)) + Int32(1) + sy
            fz = Int32(2) * (Int32(lz) - Int32(1)) + Int32(1) + sz

        cbx = child_base_x + div(fx - Int32(1), bs)
        cby = child_base_y + div(fy - Int32(1), bs)
        cbz = child_base_z + div(fz - Int32(1), bs)
            clx = (fx - Int32(1)) % bs + Int32(1)
            cly = (fy - Int32(1)) % bs + Int32(1)
            clz = (fz - Int32(1)) % bs + Int32(1)

            cb_idx = child_block_pointer[cbx, cby, cbz]
            if child_obstacle[clx, cly, clz, cb_idx]
                continue
            end

            fluid_children += Int32(1)

            # ET-AMR (STAGE 4): accumulate the child's STORED current-at-cell macros
            # (the physically-correct state). The Esoteric child populations are the
            # post-stream (one-cell-shifted-per-direction) state, so summing them gives
            # wrong macros at gradients (the box-edge ring). child_rho/child_vel were
            # computed by the child ET step from the loaded pops = the cell's current
            # macros — average those instead. (cfloc decode dropped; was the wrong fix.)
            if CHILD_IS_ET
                _crho = child_rho[clx, cly, clz, cb_idx]
                sum_crho += _crho
                sum_cjx += _crho * child_vel[clx, cly, clz, cb_idx, 1]
                sum_cjy += _crho * child_vel[clx, cly, clz, cb_idx, 2]
                sum_cjz += _crho * child_vel[clx, cly, clz, cb_idx, 3]
            end

            if use_fp64_halo == Int32(1)
                # B3: Float64 accumulation path. Promote each read to Float64
                # before adding to the f64 accumulator. Keeps the Float32
                # accumulator zero — it's filled at the demotion step below.
                @inbounds for k in 1:39
                    if k <= n_dirs_flow
                        flow_avg_f64[k] += Float64(child_f_current[clx, cly, clz, cb_idx, k])
                    end
                    if restrict_thermal == Int32(1) && k <= n_dirs_thermal
                        thermal_avg_f64[k] += Float64(child_g_current[clx, cly, clz, cb_idx, k])
                    end
                end

                if restrict_thermal == Int32(1)
                    sum_temp_f64 += Float64(child_temperature[clx, cly, clz, cb_idx])
                    sum_ekin_f64 += Float64(child_kinetic_energy[clx, cly, clz, cb_idx])
                end

                if preserve_pi_neq == Int32(1)
                    sum_pxx_f64 += Float64(child_pi_neq[clx, cly, clz, cb_idx, 1])
                    sum_pyy_f64 += Float64(child_pi_neq[clx, cly, clz, cb_idx, 2])
                    sum_pzz_f64 += Float64(child_pi_neq[clx, cly, clz, cb_idx, 3])
                    sum_pxy_f64 += Float64(child_pi_neq[clx, cly, clz, cb_idx, 4])
                    sum_pxz_f64 += Float64(child_pi_neq[clx, cly, clz, cb_idx, 5])
                    sum_pyz_f64 += Float64(child_pi_neq[clx, cly, clz, cb_idx, 6])
                end
            else
                # Legacy FP32 path — preserved unchanged for backward compat.
                # ★ ET-AMR (2026-06-18 FIX): child_f_current is the PERMUTED Esoteric
                # buffer — reading it raw as flow_avg[k] is garbage (and the post-stream
                # decode is one-cell-shifted ⇒ wrong macros at gradients). Instead
                # RECONSTRUCT each child's POST-COLLISION conventional population from the
                # state the child ET step already stored: f_coll[k] = feq(child ρ,u) +
                # Grad(full child Π). This is EXACTLY what the child kernel computed and
                # scattered, so the averaged flow_avg matches what the production non-ET
                # path averages from stored conventional populations → ET then takes the
                # SAME legacy population reconstruction (feq + τ·(flow_avg−feq)), not the
                # deviatoric Π-projection. (Trace-keeping + feq-curvature consistent.)
                if CHILD_IS_ET && !LATTICE_IS_D3Q39
                    _cr  = max(child_rho[clx, cly, clz, cb_idx], 0.01f0)
                    _cux = child_vel[clx, cly, clz, cb_idx, 1]
                    _cuy = child_vel[clx, cly, clz, cb_idx, 2]
                    _cuz = child_vel[clx, cly, clz, cb_idx, 3]
                    _cpxx = child_pi_neq[clx, cly, clz, cb_idx, 1]
                    _cpyy = child_pi_neq[clx, cly, clz, cb_idx, 2]
                    _cpzz = child_pi_neq[clx, cly, clz, cb_idx, 3]
                    _cpxy = child_pi_neq[clx, cly, clz, cb_idx, 4]
                    _cpxz = child_pi_neq[clx, cly, clz, cb_idx, 5]
                    _cpyz = child_pi_neq[clx, cly, clz, cb_idx, 6]
                    _inv2cs4 = 0.5f0 / (cs2_flow * cs2_flow)
                    _usq = _cux*_cux + _cuy*_cuy + _cuz*_cuz
                    @inbounds for k in Int32(1):Int32(19)
                        if k <= n_dirs_flow
                            _cxk = Float32(cx_arr[k]); _cyk = Float32(cy_arr[k]); _czk = Float32(cz_arr[k])
                            _wk = Float32(w_arr[k])
                            _cu = _cxk*_cux + _cyk*_cuy + _czk*_cuz
                            _feq = _cr * _wk * (1.0f0 + 3.0f0*_cu + 4.5f0*_cu*_cu - 1.5f0*_usq)
                            _Qxx = _cxk*_cxk - cs2_flow; _Qyy = _cyk*_cyk - cs2_flow; _Qzz = _czk*_czk - cs2_flow
                            _fneq = _wk * _inv2cs4 * (
                                _cpxx*_Qxx + _cpyy*_Qyy + _cpzz*_Qzz +
                                2.0f0*(_cpxy*_cxk*_cyk + _cpyz*_cyk*_czk + _cpxz*_czk*_cxk))
                            flow_avg[k] += _feq + _fneq
                        end
                    end
                else
                @inbounds for k in 1:39
                    if k <= n_dirs_flow
                        flow_avg[k] += child_f_current[clx, cly, clz, cb_idx, k]
                    end
                    if restrict_thermal == Int32(1) && k <= n_dirs_thermal
                        thermal_avg[k] += child_g_current[clx, cly, clz, cb_idx, k]
                    end
                end
                end

                if restrict_thermal == Int32(1)
                    sum_temp += child_temperature[clx, cly, clz, cb_idx]
                    sum_ekin += child_kinetic_energy[clx, cly, clz, cb_idx]
                end

                if preserve_pi_neq == Int32(1)
                    sum_pxx += child_pi_neq[clx, cly, clz, cb_idx, 1]
                    sum_pyy += child_pi_neq[clx, cly, clz, cb_idx, 2]
                    sum_pzz += child_pi_neq[clx, cly, clz, cb_idx, 3]
                    sum_pxy += child_pi_neq[clx, cly, clz, cb_idx, 4]
                    sum_pxz += child_pi_neq[clx, cly, clz, cb_idx, 5]
                    sum_pyz += child_pi_neq[clx, cly, clz, cb_idx, 6]
                end
            end
        end

        # B3 demotion step: when FP64 accumulator was used, demote to FP32
        # for compatibility with downstream Float32 code. The averaging
        # division happens in Float64 too (extra precision in the divisor
        # 1/fluid_children, which can be 1, 2, 4, 8 → varies in ULP error).
        if use_fp64_halo == Int32(1)
            inv_children_f64 = 1.0 / Float64(fluid_children)
            @inbounds for k in 1:39
                if k <= n_dirs_flow
                    flow_avg[k] = Float32(flow_avg_f64[k])
                end
                if restrict_thermal == Int32(1) && k <= n_dirs_thermal
                    thermal_avg[k] = Float32(thermal_avg_f64[k])
                end
            end
            # Note: we keep flow_avg as the SUM (not yet averaged) — the
            # legacy code averages by `inv_children` in the next loop and
            # accumulates the moments in the same loop. To stay consistent,
            # we leave flow_avg with the FP64-summed value (demoted to F32)
            # so the subsequent average/moment code path is unchanged.
            sum_temp = Float32(sum_temp_f64)
            sum_ekin = Float32(sum_ekin_f64)
            sum_pxx = Float32(sum_pxx_f64)
            sum_pyy = Float32(sum_pyy_f64)
            sum_pzz = Float32(sum_pzz_f64)
            sum_pxy = Float32(sum_pxy_f64)
            sum_pxz = Float32(sum_pxz_f64)
            sum_pyz = Float32(sum_pyz_f64)
        end

        if fluid_children > Int32(0)
        inv_children = 1.0f0 / Float32(fluid_children)
        # WALL-DRAG FIX (see _D3Q19_RESTRICT_COVERAGE_FRAC at top of file): for an
        # obstacle-adjacent full-support cell with some SOLID children, weight the
        # restricted populations by the actual fluid coverage instead of full strength,
        # blending the solid portion with the parent's prior state. No-op when all 8
        # children are fluid (frac stays 1.0) or when the flag is off.
        if _D3Q19_RESTRICT_COVERAGE_FRAC && fluid_children < Int32(8)
            frac = Float32(fluid_children) * 0.125f0
            one_m_frac = 1.0f0 - frac
        end
        rho_raw = 0.0f0
        jx = 0.0f0
        jy = 0.0f0
        jz = 0.0f0
        temp_parent = 1.0f0
        pxx_parent = sum_pxx * inv_children
        pyy_parent = sum_pyy * inv_children
        pzz_parent = sum_pzz * inv_children
        pxy_parent = sum_pxy * inv_children
        pxz_parent = sum_pxz * inv_children
        pyz_parent = sum_pyz * inv_children

        if use_fp64_halo == Int32(1)
            # B3 — accumulate the 39-direction moment sums in Float64. With
            # 39 Float32 reads accumulating into a Float32 register, the
            # legacy path's rho/j moments carry ~39× FP32 ULP ≈ 4.6e-6.
            # Float64 accumulation drops this to the FP32 storage limit
            # at the demotion (~1.2e-7).
            inv_children_f64 = 1.0 / Float64(fluid_children)
            rho_raw_f64 = 0.0
            jx_f64 = 0.0
            jy_f64 = 0.0
            jz_f64 = 0.0
            @inbounds for k in 1:39
                if k <= n_dirs_flow
                    favg64 = flow_avg_f64[k] * inv_children_f64
                    flow_avg[k] = Float32(favg64)   # demote averaged f for downstream code
                    rho_raw_f64 += favg64
                    jx_f64 += favg64 * Float64(cx_arr[k])
                    jy_f64 += favg64 * Float64(cy_arr[k])
                    jz_f64 += favg64 * Float64(cz_arr[k])
                end
                if restrict_thermal == Int32(1) && k <= n_dirs_thermal
                    gavg64 = thermal_avg_f64[k] * inv_children_f64
                    thermal_avg[k] = Float32(gavg64)
                end
            end
            rho_raw = Float32(rho_raw_f64)
            jx = Float32(jx_f64)
            jy = Float32(jy_f64)
            jz = Float32(jz_f64)
        else
            # Legacy FP32 path — preserved unchanged.
            @inbounds for k in 1:39
                if k <= n_dirs_flow
                    favg = flow_avg[k] * inv_children
                    flow_avg[k] = favg
                    rho_raw += favg
                    jx += favg * Float32(cx_arr[k])
                    jy += favg * Float32(cy_arr[k])
                    jz += favg * Float32(cz_arr[k])
                end

                if restrict_thermal == Int32(1) && k <= n_dirs_thermal
                    gavg = thermal_avg[k] * inv_children
                    thermal_avg[k] = gavg
                end
            end
        end

        # ★ ET-AMR (STAGE 4): OVERRIDE rho/j with the averaged child STORED macros
        # (the correct current-at-cell state). The population-summed rho_raw/jx is the
        # post-stream shifted state (wrong at gradients → box-edge ring). Mass &
        # momentum are exactly conserved (linear average of child macros). The Π
        # stress carrier still comes from the averaged child_pi_neq (sum_p*), which is
        # already the per-cell post-collision stress the child step stored.
        if CHILD_IS_ET
            rho_raw = sum_crho * inv_children
            jx = sum_cjx * inv_children
            jy = sum_cjy * inv_children
            jz = sum_cjz * inv_children
        end

        rho_parent = max(rho_raw, 0.01f0)
        inv_rho = 1.0f0 / rho_parent
        ux_parent = jx * inv_rho
        uy_parent = jy * inv_rho
        uz_parent = jz * inv_rho

        parent_rho[lx, ly, lz, pb_idx] = rho_parent
        parent_vel[lx, ly, lz, pb_idx, 1] = ux_parent
        parent_vel[lx, ly, lz, pb_idx, 2] = uy_parent
        parent_vel[lx, ly, lz, pb_idx, 3] = uz_parent

        # â”€â”€ Ï„-rescale non-equilibrium part: f_coarse = f_eq + (Ï„_c-0.5)/(Ï„_f-0.5) Â· <f_neq>_fine â”€â”€
        # The reference equilibrium MUST match what the rest of the D3Q39 solver
        # uses, otherwise the polynomial vs. 13-moment-numerical mismatch biases
        # every direction in the same sign and concentrates into the mirror-symmetric
        # stress mode s. Collision kernel, prolongation, and shell sync all use the
        # numerical entropic equilibrium on D3Q39 â€” restriction must match. For
        # D3Q27 the polynomial is correct (that is the collision-kernel equilibrium
        # on D3Q27). Solve lambdas once per parent cell, reuse for all 39 directions.
        temp_for_geq = restrict_thermal == Int32(1) ? max(sum_temp * inv_children, 0.01f0) : 1.0f0

        # Restriction must use the SAME equilibrium family as the collision and
        # prolongation paths, otherwise the coarse<->fine round-trip injects a
        # deterministic seam: restriction writes parent f from a polynomial feq,
        # prolongation reads it back against the 13-moment entropic feq, and the
        # mismatch is reinterpreted as non-equilibrium stress at every subcycle.
        #
        # That inconsistency is harmless at very small u, but once the transonic
        # gradients sharpen it becomes a seed for checkerboard stress and density
        # teeth on the parent shell. D3Q27 still uses the polynomial equilibrium;
        # D3Q39 entropic must stay on the numerical 13-moment path here too.
        use_numerical_eq = lattice_is_d3q39 == Int32(1)
        ux_rel = ux_parent - shift_x
        uy_rel = uy_parent - shift_y
        uz_rel = uz_parent - shift_z
        T_parent_clamped = clamp(temp_for_geq, 0.1f0, 10.0f0)
        lambda_eq = use_numerical_eq ?
            solve_13mom_lambdas_d3q39(
                rho_parent, ux_rel, uy_rel, uz_rel,
                cx_arr, cy_arr, cz_arr, w_arr,
                cs2_flow, Int32(50), T_parent_clamped,
                shift_x, shift_y, shift_z,
                100.0f0, eos_R_flow          # SOTA A3 — match collision-kernel EOS R
            ) :
            (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
             0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
        mass_corr_eq = use_numerical_eq ?
            compute_13mom_mass_correction_d3q39(
                rho_parent, lambda_eq,
                cx_arr, cy_arr, cz_arr, w_arr,
                cs2_flow, shift_x, shift_y, shift_z
            ) : 1.0f0

        # ── Pass 265 (Tier 1 #1e) — exact mass+momentum correction of f_eq ──
        # When `restriction_conservative_feq` is enabled, materialize f_eq for
        # all 39 directions into a stack-local MVector, then apply a 4×4 Gram
        # correction that drives Σ f_eq → ρ_parent and Σ (c−shift)·f_eq → ρ·u_rel
        # to round-off. Without this, the 13-moment Newton solver leaves
        # ~1e-5 residual that propagates into the post-restriction seam Δρ.
        # The correction is moment-orthogonal to the 2nd-order Hermite stress
        # projection used below, so the stress carrier is unaffected.
        # When the flag is off, `f_eq_corr_valid` stays false and the per-k
        # loop falls back to the legacy eval_13mom_feq_single path.
        f_eq_corr = MVector{39, Float32}(undef)
        f_eq_corr_valid = false
        if restriction_conservative_feq == Int32(1) && use_numerical_eq
            @inbounds for k in Int32(1):Int32(39)
                cxk_eq = Float32(cx_arr[k]) - shift_x
                cyk_eq = Float32(cy_arr[k]) - shift_y
                czk_eq = Float32(cz_arr[k]) - shift_z
                wk_eq = Float32(w_arr[k])
                f_eq_corr[k] = mass_corr_eq * eval_13mom_feq_single(
                    rho_parent, lambda_eq,
                    cxk_eq, cyk_eq, czk_eq, wk_eq, cs2_flow
                )
            end
            # Targets in the SHIFTED frame (matches solve_13mom_lambdas_d3q39):
            #   j_rel = j_lab − ρ · shift
            jx_rel_target = jx - rho_parent * shift_x
            jy_rel_target = jy - rho_parent * shift_y
            jz_rel_target = jz - rho_parent * shift_z
            # SOTA P1: 5-moment (mass + 3·mom + E_kin) when conservative ρE is on
            # AND we have thermal data (sum_ekin populated). Otherwise fall back
            # to the legacy 4-moment correction.
            if restriction_conservative_rhoe == Int32(1) && restrict_thermal == Int32(1)
                # SHIFT-FRAME FIX (2026-05-29): kinetic_energy is STORED as the
                # UNSHIFTED moment Σf·|c_unshifted|² (see kernel_entropic_d3q39.jl
                # E_kin_unshifted). But apply_5momentum_correction_d3q39! computes
                # its internal energy moment in the SHIFTED frame (cxk = c − shift).
                # Convert to the shifted frame before passing:
                #   E_shifted = E_unshifted + 2·U·j_rel − |U|²·ρ
                # (inverse of E_unshifted = E_shifted − 2·U·j_rel + |U|²·ρ, with
                #  j_rel = j_shifted = j_lab − ρ·U). Identity when shift = 0.
                Ekin_unshifted_target = sum_ekin * inv_children
                U_dot_jrel = shift_x * jx_rel_target + shift_y * jy_rel_target + shift_z * jz_rel_target
                U_sq_shift = shift_x*shift_x + shift_y*shift_y + shift_z*shift_z
                Ekin_rel_target = Ekin_unshifted_target + 2.0f0 * U_dot_jrel - U_sq_shift * rho_parent
                apply_5momentum_correction_d3q39!(
                    f_eq_corr,
                    rho_parent, jx_rel_target, jy_rel_target, jz_rel_target,
                    Ekin_rel_target,
                    cx_arr, cy_arr, cz_arr, w_arr,
                    shift_x, shift_y, shift_z
                )
            else
                apply_4momentum_correction_d3q39!(
                    f_eq_corr,
                    rho_parent, jx_rel_target, jy_rel_target, jz_rel_target,
                    cx_arr, cy_arr, cz_arr, w_arr,
                    shift_x, shift_y, shift_z
                )
            end
            f_eq_corr_valid = true
        end

        # â”€â”€ Parent-side Î _neq (already averaged from children via preserve_pi_neq) â”€â”€
        # When available, use it as the stress-mode carrier â€” SAME recipe as the
        # prolongation's project_pi_neq_stress_mode path, so the round-trip becomes
        # symmetric: (coarse â†’ fine) uses Î  projection, (fine â†’ coarse) uses Î 
        # projection too. Population mirror decomposition is kept only as the
        # fallback for operators that don't track Î _neq (cumulant/BGK without
        # variable_nu_sgs). Dropping the h component on the way up is consistent
        # with dropping it on the way down; the fine Î± solver rebuilds it in one
        # collision step.
        # ET-AMR (STAGE 4, REVISED 2026-06-18): for D3Q19 ET, DO NOT force the
        # deviatoric Π-projection. The earlier STAGE-4 code forced use_pi_stress for ET
        # to avoid the post-stream-shifted flow_avg — but that made ET restriction use a
        # DIFFERENT reconstruction (feq + τ·project_DEVIATORIC(Π)) than the validated
        # production non-ET D3Q19 path (feq + τ·(flow_avg−feq), full fneq incl. trace +
        # feq-curvature). That divergence was the box-edge ring + interior blotches
        # (Z265 vs Z266 A/B). FIX: the ET accumulation loop above now reconstructs each
        # child's POST-COLLISION conventional population (feq(child macros) + Grad(full
        # child Π)) into flow_avg — exactly what production averages from stored conv
        # populations. So D3Q19 ET takes the SAME legacy population path as production.
        # The entropic D3Q39 ET path (LATTICE_IS_D3Q39) keeps the Π-carrier (its
        # mirror_decompose_entropic recipe is the validated one there).
        use_pi_stress = (mirror_decompose_entropic == Int32(1) ||
                         (CHILD_IS_ET && LATTICE_IS_D3Q39)) &&
                        preserve_pi_neq == Int32(1)
        # Keep shell populations donor-consistent whenever we have any
        # non-equilibrium carrier. A pure parent-shell equilibrium anchor
        # creates a level-dependent velocity state that shows up as vorticity
        # on AMR faces, most visibly on x-facing interfaces.
        shell_equilibrium_only = mirror_decompose_entropic == Int32(1) &&
                                 preserve_pi_neq == Int32(0) &&
                                 tau_ratio_flow <= 0.0f0

        # ══════════════════════════════════════════════════════════════════
        # B1 Pillar 2 — 13-moment-exact restriction (mortar-element coupling)
        # ══════════════════════════════════════════════════════════════════
        # When `restriction_conservative_13mom == 1` AND we're on D3Q39 AND
        # we have populations to read (any path), bypass the legacy
        # equilibrium + stress carrier construction. Instead, solve once
        # for the parent λ that matches all 13 averaged child moments
        # (τ-rescaled on the non-equilibrium portion) to round-off.
        #
        # This is the strict-conservation upgrade — replaces the partial
        # 4- or 5-moment correction + projected stress with a single
        # closed-form pass that conserves ALL 13 Hermite moments.
        #
        # When the flag is off, control falls through to the legacy path.
        used_13mom_path = false
        if restriction_conservative_13mom == Int32(1) && use_numerical_eq
            # Per-k τ-rescale of moments expects the same convention as the
            # existing code: tau_ratio_flow already encodes (τ_c-0.5)/(τ_f-0.5).
            f_13mom_out = MVector{39, Float32}(undef)
            # GOLD-STANDARD FIX (2026-06-25):
            #  - Thread the parent's ACTUAL T (temp_for_geq) and EOS R (eos_R_flow,
            #    the same coefficient the collision kernel uses) into the M_eq
            #    reference, so the thermal pressure is NOT mislabeled non-eq and
            #    amplified. eos_R_flow ≤ 0 sentinel → cs2 fallback inside helper,
            #    keeping athermal/linearized cases byte-identical.
            #  - Capture the realizability/convergence flag. When the off-axis
            #    oblique-shock+shear solve fails to converge OR realizes a spiked
            #    (non-realizable) distribution, ok==false: we DO NOT commit the
            #    13-moment populations and leave used_13mom_path=false so the per-k
            #    loop below reconstructs THIS cell with the guaranteed-positive
            #    legacy realizable path (feq + τ·projected-stress). Strict 13-moment
            #    conservation is preserved byte-identically in every converged cell.
            ok_13mom = apply_13moment_restriction_d3q39!(
                f_13mom_out, flow_avg, tau_ratio_flow,
                cx_arr, cy_arr, cz_arr, w_arr,
                cs2_flow,
                eos_R_flow,        # SOTA A3 — match collision-kernel EOS R (≤0 ⇒ cs2)
                shift_x, shift_y, shift_z,
                Int32(50),
                temp_for_geq       # parent T reference for the M_eq split
            )
            if ok_13mom
                # The 13-moment path is D3Q39-only (use_numerical_eq); the ET
                # encode (pfloc) is the D3Q19 path, so PARENT_IS_ET is never true
                # here — write directly to parent_f_current as before.
                @inbounds for k in 1:39
                    if k <= n_dirs_flow
                        f_prior_k = parent_f_current[lx, ly, lz, pb_idx, k]
                        parent_f_current[lx, ly, lz, pb_idx, k] =
                            max(frac * f_13mom_out[k] + one_m_frac * f_prior_k, 1.0f-10)
                    end
                end
                used_13mom_path = true
            end
            # else: graceful per-cell fallback — used_13mom_path stays false, the
            # legacy realizable reconstruction in the per-k loop handles this cell.
        end

        # SOTA #2: capture the prior parent Σg BEFORE the g-write loop overwrites
        # it, so the energy-conservation correction below has the exact shell-blend
        # reference: target Σg = frac·Σ(thermal_avg) + (1−frac)·Σg_prior_parent.
        prior_parent_sum_g = 0.0f0
        if restriction_conservative_rhoe == Int32(1) && restrict_thermal == Int32(1) &&
           frapolli_mode == Int32(1)
            @inbounds for k in Int32(1):n_dirs_thermal
                prior_parent_sum_g += parent_g_current[lx, ly, lz, pb_idx, k]
            end
        end

        # GOLD-STANDARD D3Q19 conservation, FRONTIER capture (2026-06-17): for
        # partial-coverage parents (frac<1) the per-k loop below writes the BLEND
        # frac·restricted + (1−frac)·prior, so the conserved target is the SAME
        # blend of invariants. Capture the prior parent's mass+momentum HERE,
        # before the loop overwrites parent_f_current. Mirrors prior_parent_sum_g
        # (SOTA #2). Only needed for D3Q19 + flag + frontier; same shifted frame.
        prior_rho_d19 = 0.0f0; prior_jx_d19 = 0.0f0
        prior_jy_d19 = 0.0f0;  prior_jz_d19 = 0.0f0
        if restriction_conservative_d3q19 == Int32(1) && !use_numerical_eq &&
           frac < 0.999f0
            # 2026-06-23: loop to n_dirs_flow (was hard-coded 19) so all 27 D3Q27
            # cumulant directions are captured. Same flag now serves D3Q19 + D3Q27.
            @inbounds for k in Int32(1):n_dirs_flow
                if k <= n_dirs_flow
                    fpk = parent_f_current[lx, ly, lz, pb_idx, k]
                    prior_rho_d19 += fpk
                    prior_jx_d19  += fpk * (Float32(cx_arr[k]) - shift_x)
                    prior_jy_d19  += fpk * (Float32(cy_arr[k]) - shift_y)
                    prior_jz_d19  += fpk * (Float32(cz_arr[k]) - shift_z)
                end
            end
        end

        # ET-AMR parent encode (STAGE 4): when PARENT_IS_ET, the flow reconstruction
        # + corrections below write/read a local pfloc[19] (conventional layout)
        # instead of parent_f_current; the final values are ET-SCATTERED into
        # parent_f_current once at the cell end (race-free, == _et_store_l1!). Seeded
        # to 0 (full-coverage frac=1 ⇒ the f_prior term is weighted 0 anyway).
        pfloc = MVector{19, Float32}(undef)
        if PARENT_IS_ET
            @inbounds for k in 1:19; pfloc[k] = 0.0f0; end
        end

        # ── PROPORTIONAL POSITIVITY GUARD for the use_pi_stress reconstruction ──
        # SPECKLE FIX (2026-06-25): the use_pi_stress flow reconstruction below
        # builds f_restricted_k = feq_k + tau_ratio_flow·stress_k, then floors each
        # direction with max(…,1e-10). The deviatoric stress projection has ZERO
        # 0th/1st moments on the lattice quadrature, so clipping an INDIVIDUAL
        # direction injects uncompensated mass/momentum → cell-scale checkerboard
        # speckle on the +x refinement-box face (where, under shift U=+1, the +x
        # high-order links carry the largest Π_xx and only there does
        # feq+τ·stress go negative and trip the floor).
        #
        # Port the PROVEN prolongation pattern (kernel_entropic_d3q39.jl:2392-2425):
        # a single PRE-PASS over all flow directions computes ONE scalar
        # stress_theta_restrict ∈ [0,1] = the minimum scale that keeps every
        # direction's feq_kk + θ·τ·stress_kk ≥ F_FLOOR, then the per-k write applies
        # that SAME θ uniformly. Damping the WHOLE stress carrier proportionally
        # preserves Σf and Σc·f moment balance (no per-direction imbalance ⇒ no
        # speckle). Freestream byte-identical: stress≈0 ⇒ θ=1 ⇒ identical to legacy.
        # Computed ONCE here (before the per-k loop), reused across all k.
        # GPU-safe: scalar-only accumulation, no heap allocation (feq_kk reuses the
        # already-materialized f_eq_corr MVector or the same eval_13mom_feq_single
        # path the per-k loop uses).
        stress_theta_restrict = 1.0f0
        if use_pi_stress
            @inbounds for kk in Int32(1):n_dirs_flow
                cxk_g = Float32(cx_arr[kk]) - shift_x
                cyk_g = Float32(cy_arr[kk]) - shift_y
                czk_g = Float32(cz_arr[kk]) - shift_z
                wk_g = Float32(w_arr[kk])
                feq_kk = if f_eq_corr_valid
                    f_eq_corr[kk]
                elseif use_numerical_eq
                    mass_corr_eq * eval_13mom_feq_single(
                        rho_parent, lambda_eq, cxk_g, cyk_g, czk_g, wk_g, cs2_flow
                    )
                else
                    calculate_equilibrium_auto(
                        rho_parent, ux_parent, uy_parent, uz_parent,
                        wk_g, cxk_g, cyk_g, czk_g,
                        flow_is_compressible
                    )
                end
                stress_kk = project_second_order_stress_mode(
                    wk_g, cxk_g, cyk_g, czk_g, cs2_flow,
                    pxx_parent, pyy_parent, pzz_parent,
                    pxy_parent, pxz_parent, pyz_parent
                )
                s_scaled = tau_ratio_flow * stress_kk
                if feq_kk + s_scaled < F_FLOOR
                    stress_theta_restrict = min(
                        stress_theta_restrict,
                        max((feq_kk - F_FLOOR) / max(-s_scaled, 1.0f-20), 0.0f0)
                    )
                end
            end
            stress_theta_restrict = clamp(stress_theta_restrict, 0.0f0, 1.0f0)
        end

        @inbounds for k in 1:39
            # B1 Pillar 2: flow side is fully handled by the 13-moment-exact
            # path above when active. The thermal side (k ≤ n_dirs_thermal)
            # still runs the legacy Frapolli/polynomial reconstruction below
            # — it depends on `lambda_eq` and `mass_corr_eq`, which are
            # rebuilt from (rho_parent, ux/uy/uz_parent) and thus consistent
            # with whichever flow path was used.
            if k <= n_dirs_flow && !used_13mom_path
                cxk = Float32(cx_arr[k]) - shift_x
                cyk = Float32(cy_arr[k]) - shift_y
                czk = Float32(cz_arr[k]) - shift_z
                wk = Float32(w_arr[k])
                # Pass 265 (Tier 1 #1e): use the 4-momentum-corrected f_eq when
                # the conservative path is active. Otherwise fall back to the
                # legacy mass-only-corrected Newton equilibrium.
                feq_k = if f_eq_corr_valid
                    f_eq_corr[k]
                elseif use_numerical_eq
                    mass_corr_eq * eval_13mom_feq_single(
                        rho_parent, lambda_eq, cxk, cyk, czk, wk, cs2_flow
                    )
                else
                    calculate_equilibrium_auto(
                        rho_parent, ux_parent, uy_parent, uz_parent,
                        wk, cxk, cyk, czk,
                        flow_is_compressible
                    )
                end

                # Restricted population: use the restricted macros (rho_restricted,
                # ux_restricted) to build feq, but the STRESS comes from the averaged
                # Î _neq (whose content is already restricted). The final parent_f is
                # a blend with the parent's PRIOR f_k weighted by frac (partial-
                # restriction v1.11), so cells with only some children inside L4 get
                # a proportional update toward the L4 state without erasing their
                # own evolved state.
                f_prior_k = PARENT_IS_ET ? pfloc[k] : parent_f_current[lx, ly, lz, pb_idx, k]
                # ISOLATION-REBUILD BASELINE (2026-06-25): reverted the restriction
                # population-mirror port; back to the as-found use_pi_stress gate.
                if use_pi_stress
                    # Moment-based reconstruction: second-order-Hermite projection
                    # of averaged Î _neq onto direction k. Symmetric with the
                    # prolongation (both paths now work in Î  space).
                    stress_k = project_second_order_stress_mode(
                        wk, cxk, cyk, czk, cs2_flow,
                        pxx_parent, pyy_parent, pzz_parent,
                        pxy_parent, pxz_parent, pyz_parent
                    )
                    # SPECKLE FIX (2026-06-25): scale the WHOLE stress carrier by the
                    # single proportional guard stress_theta_restrict (computed once
                    # per coarse cell in the pre-pass above) instead of clipping each
                    # direction independently. Preserves Σf and Σc·f moment balance.
                    f_restricted_k = feq_k + tau_ratio_flow * stress_theta_restrict * stress_k
                    # Outer floor kept as a backstop; θ already keeps it positive.
                    _f_rec_k = max(frac * f_restricted_k + one_m_frac * f_prior_k, 1.0f-10)
                    if PARENT_IS_ET
                        pfloc[k] = _f_rec_k
                    else
                        parent_f_current[lx, ly, lz, pb_idx, k] = _f_rec_k
                    end
                elseif mirror_decompose_entropic == Int32(1)
                    # Entropic without Î _neq (should be rare after STRESS_TENSOR_TRANSFER
                    # default-on): fall back to population mirror decomposition.
                    kopp = opp_arr[k]
                    cxk_b = Float32(cx_arr[kopp]) - shift_x
                    cyk_b = Float32(cy_arr[kopp]) - shift_y
                    czk_b = Float32(cz_arr[kopp]) - shift_z
                    wk_b = Float32(w_arr[kopp])
                    feq_kb = use_numerical_eq ?
                        mass_corr_eq * eval_13mom_feq_single(
                            rho_parent, lambda_eq, cxk_b, cyk_b, czk_b, wk_b, cs2_flow
                        ) :
                        calculate_equilibrium_auto(
                            rho_parent, ux_parent, uy_parent, uz_parent,
                            wk_b, cxk_b, cyk_b, czk_b,
                            flow_is_compressible
                        )
                    fneq_k = flow_avg[k] - feq_k
                    fneq_kb = flow_avg[kopp] - feq_kb
                    s_k = 0.5f0 * (fneq_k + fneq_kb)
                    h_k = 0.5f0 * (fneq_k - fneq_kb)
                    f_restricted_k = feq_k + tau_ratio_flow * s_k +
                                     h_scale_restrict * tau_ratio_flow * h_k
                    _f_rec_k = max(frac * f_restricted_k + one_m_frac * f_prior_k, 1.0f-10)
                    if PARENT_IS_ET
                        pfloc[k] = _f_rec_k
                    else
                        parent_f_current[lx, ly, lz, pb_idx, k] = _f_rec_k
                    end
                else
                    # Cumulant/BGK: legacy full f_neq rescaling.
                    fneq_k = flow_avg[k] - feq_k
                    f_restricted_k = feq_k + tau_ratio_flow * fneq_k
                    _f_rec_k = max(frac * f_restricted_k + one_m_frac * f_prior_k, 1.0f-10)
                    if PARENT_IS_ET
                        pfloc[k] = _f_rec_k
                    else
                        parent_f_current[lx, ly, lz, pb_idx, k] = _f_rec_k
                    end
                end
            end

            if restrict_thermal == Int32(1) && k <= n_dirs_thermal
                cxk = Float32(cx_arr[k]) - shift_x
                cyk = Float32(cy_arr[k]) - shift_y
                czk = Float32(cz_arr[k]) - shift_z
                wk = Float32(w_arr[k])
                # Thermal equilibrium: Frapolli g_eq = (Cv âˆ’ DÂ·csÂ²/2)Â·TÂ·f_eq_numerical
                # for D3Q39 compressible (uses the SAME 13-moment Î»'s already solved
                # above for the flow side, since Frapolli's thermal DDF reuses the
                # flow equilibrium scaled by a coeffÂ·T factor). For D3Q27 the
                # polynomial advection-diffusion form is correct as before. This
                # closes the Tier A-1 paper-quality gap â€” same class of fix as the
                # flow-side polynomialâ†’numerical eq switch we already made.
                geq_k = if use_numerical_eq && frapolli_mode == Int32(1)
                    # EOS-CONSISTENT g VARIANCE (2026-06-25 REGRESSION FIX): the
                    # Frapolli g_eq coefficient is (Cv − D·R/2) where R is the
                    # SAME EOS coefficient the thermal collision kernel uses. With
                    # eos_thermodynamic on, the collision/thermal kernels use
                    # coeff_g = Cv − 1.5·eos_R_flow (kernel_thermal.jl:151/394/448,
                    # AUDIT 2026-06-03). This restriction was left on the legacy
                    # LATTICE form Cv − 1.5·cs²_thermal, so at eos_R≠cs² it rebuilt
                    # every covered parent's g with a DIFFERENT variance than
                    # collision → Σg step → recovered T = (E_kin−ρu²+2Σg)/(2ρCv)
                    # step → p=(γ−1)ρCvT step at the refinement-box edge (a stable
                    # BULK refined-vs-coarse offset, ~+20% T at the D3Q39 Ma1.5
                    # case: 2.5−1.5·(2/3)=1.5 vs 2.5−1.5·1.0=1.0 ⇒ ΔΣg=0.5 ⇒
                    # ΔT≈0.2). Sentinel eos_R_flow≤0 ⇒ cs²_thermal fallback keeps
                    # the legacy linearized-EOS path byte-identical.
                    R_g = eos_R_flow > 0.0f0 ? eos_R_flow : cs2_thermal
                    coeff_g = cv_thermal - 1.5f0 * R_g
                    feq_flow_k = mass_corr_eq * eval_13mom_feq_single(
                        rho_parent, lambda_eq, cxk, cyk, czk, wk, cs2_flow
                    )
                    coeff_g * temp_for_geq * feq_flow_k
                else
                    inv_cs2_t = 1.0f0 / cs2_thermal
                    c_dot_u = cxk * ux_parent + cyk * uy_parent + czk * uz_parent
                    wk * temp_for_geq * (1.0f0 + c_dot_u * inv_cs2_t)
                end
                gneq_scaled = tau_ratio_thermal * (thermal_avg[k] - geq_k)
                g_restricted_k = geq_k + gneq_scaled
                g_prior_k = parent_g_current[lx, ly, lz, pb_idx, k]
                parent_g_current[lx, ly, lz, pb_idx, k] =
                    max(frac * g_restricted_k + one_m_frac * g_prior_k, 1.0f-10)
            end
        end

        # ══════════════════════════════════════════════════════════════════
        # GOLD-STANDARD STRICT CONSERVATION (2026-06-17) — D3Q19 4-moment
        # ══════════════════════════════════════════════════════════════════
        # Drive the just-written parent populations to EXACT mass + 3·momentum
        # (the complete invariant set for athermal D3Q19). The reconstruction
        # above is conservative in exact arithmetic (Identity 1: Σfeq=ρ_parent,
        # Σc·feq=ρu, Σfneq=0), but the per-k positivity floor max(…,1e-10) and
        # FP32 round-off leave a ~1e-7/cell residual. Solve a 4×4 weighted-Gram
        # system (the existing solve_shell_moment4) for the correction
        # δf[k] = w_k·(a0 + a1·cx' + a2·cy' + a3·cz') that exactly closes the
        # mass+momentum residual; by D3Q19 isotropy this basis is orthogonal to
        # the traceless stress modes, so shear transport is untouched.
        # Scope: ALL covered parents. Full-coverage (frac≈1) targets the child
        # invariants (ρ_parent, ρ_parent·u_rel) exactly; FRONTIER (frac<1) targets
        # the SAME blend the per-k loop writes: frac·child + (1−frac)·prior (the
        # prior_*_d19 captured above), so conservation holds without altering the
        # anti-band blend's intent. D3Q19-only (the numerical-eq/D3Q39 path has its
        # own 13-mom-exact restriction). Gated; default 0 ⇒ byte-identical.
        if restriction_conservative_d3q19 == Int32(1) && !use_numerical_eq
            # 2026-06-23: MVector sized for D3Q27 (27 dirs); D3Q19 uses [1:19]. The
            # `if k <= n_dirs_flow` guard keeps the D3Q19 result byte-identical.
            f_d19 = MVector{27, Float32}(undef)
            cur_rho = 0.0f0; cur_jx = 0.0f0; cur_jy = 0.0f0; cur_jz = 0.0f0
            m00 = 0.0f0; m01 = 0.0f0; m02 = 0.0f0; m03 = 0.0f0
            m11 = 0.0f0; m12 = 0.0f0; m13 = 0.0f0
            m22 = 0.0f0; m23 = 0.0f0; m33 = 0.0f0
            @inbounds for k in Int32(1):n_dirs_flow
                if k <= n_dirs_flow
                    fk = PARENT_IS_ET ? pfloc[k] : parent_f_current[lx, ly, lz, pb_idx, k]
                    f_d19[k] = fk
                    cxk = Float32(cx_arr[k]) - shift_x
                    cyk = Float32(cy_arr[k]) - shift_y
                    czk = Float32(cz_arr[k]) - shift_z
                    wk  = Float32(w_arr[k])
                    cur_rho += fk; cur_jx += fk*cxk; cur_jy += fk*cyk; cur_jz += fk*czk
                    m00 += wk;          m01 += wk*cxk;     m02 += wk*cyk;     m03 += wk*czk
                    m11 += wk*cxk*cxk;  m12 += wk*cxk*cyk; m13 += wk*cxk*czk
                    m22 += wk*cyk*cyk;  m23 += wk*cyk*czk
                    m33 += wk*czk*czk
                end
            end
            # Target = the BLEND the per-k loop wrote (relative frame). For
            # full-coverage frac=1 ⇒ collapses to the child invariants (ρ_parent,
            # ρ_parent·u_rel) exactly (= v1). For frontier frac<1 ⇒ blends with the
            # prior parent invariants captured above, matching the anti-band blend.
            ux_rel = ux_parent - shift_x
            uy_rel = uy_parent - shift_y
            uz_rel = uz_parent - shift_z
            tgt_rho = frac*rho_parent          + one_m_frac*prior_rho_d19
            tgt_jx  = frac*(rho_parent*ux_rel) + one_m_frac*prior_jx_d19
            tgt_jy  = frac*(rho_parent*uy_rel) + one_m_frac*prior_jy_d19
            tgt_jz  = frac*(rho_parent*uz_rel) + one_m_frac*prior_jz_d19
            er = tgt_rho - cur_rho
            ex = tgt_jx  - cur_jx
            ey = tgt_jy  - cur_jy
            ez = tgt_jz  - cur_jz
            if abs(er) > 1.0f-9 || abs(ex) > 1.0f-9 ||
               abs(ey) > 1.0f-9 || abs(ez) > 1.0f-9
                a0c, a1c, a2c, a3c, okm = solve_shell_moment4(
                    m00, m01, m02, m03, m11, m12, m13, m22, m23, m33,
                    er, ex, ey, ez)
                if okm
                    # 2026-06-23: loop to n_dirs_flow (was 19) to correct all 27
                    # D3Q27 directions. Byte-identical for D3Q19 (only 1:19 execute).
                    @inbounds for k in Int32(1):n_dirs_flow
                        if k <= n_dirs_flow
                            cxk = Float32(cx_arr[k]) - shift_x
                            cyk = Float32(cy_arr[k]) - shift_y
                            czk = Float32(cz_arr[k]) - shift_z
                            wk  = Float32(w_arr[k])
                            _f_corr_k = max(f_d19[k] + wk*(a0c + a1c*cxk + a2c*cyk + a3c*czk), 1.0f-10)
                            if PARENT_IS_ET
                                pfloc[k] = _f_corr_k
                            else
                                parent_f_current[lx, ly, lz, pb_idx, k] = _f_corr_k
                            end
                        end
                    end
                end
            end
        end

        # ══════════════════════════════════════════════════════════════════
        # SOTA #2 (2026-05-29) — conservative ρE across AMR (g-side closure)
        # ══════════════════════════════════════════════════════════════════
        # The energy carried across the AMR interface lives in Σg (Frapolli
        # closure: 2ρCvT = E_kin − ρ|u|² + 2Σg). The flow-side restriction
        # already conserves E_kin exactly (5-/13-moment correction above), but
        # the g reconstruction `g = geq + τ·(g_avg − geq)` rescales the
        # NON-equilibrium part, which shifts the zeroth g-moment (Σg) away from
        # the conservative target Σ(g_avg). That residual is the unconserved
        # internal-energy fraction — exactly what drifts when a shock crosses a
        # refinement boundary at Ma≳1.2.
        #
        # Fix: after the per-k g write, add a single scalar δ_g·w_k to every
        # parent g so that Σ_k g_parent = Σ_k thermal_avg[k] (= the average of
        # the 8 children's Σg, the strict-conservation target). Adding δ_g·w_k
        # shifts Σg by δ_g·Σw_k = δ_g (weights sum to 1) and leaves the g heat-
        # flux moment Σ c_k g untouched (Σ w_k c_k = 0 by lattice isotropy), so
        # the correction touches ONLY the conserved energy moment. The post-write
        # max(…, 1e-10) floor and the frac/one_m_frac shell blend are respected
        # by recomputing the realized Σg from the written values.
        # Gated by `restriction_conservative_rhoe` — the same flag that turns on
        # the flow-side E_kin conservation, so both halves of ρE move together.
        if restriction_conservative_rhoe == Int32(1) && restrict_thermal == Int32(1) &&
           frapolli_mode == Int32(1)
            # Conservative target Σg, blended identically to the populations:
            #   target = frac·Σ(thermal_avg) + (1−frac)·Σg_prior_parent
            children_sum_g = 0.0f0
            realized_sum_g = 0.0f0
            @inbounds for k in Int32(1):n_dirs_thermal
                children_sum_g += thermal_avg[k]
                realized_sum_g += parent_g_current[lx, ly, lz, pb_idx, k]
            end
            target_sum_g = frac * children_sum_g + one_m_frac * prior_parent_sum_g

            # GOLD-STANDARD GUARD (2026-06-25) — physical-T clamp on the energy
            # split. The Frapolli closure recovers parent T from Σg:
            #   T = (E_kin − ρ|u|²_lab + 2·Σg) / (2·ρ·Cv).
            # A bad averaged-child energy split (extreme oblique-shock cell) can
            # drive the implied T negative or huge. Clamp the conservative
            # target_sum_g to the value yielding T ∈ [0.1, 10]. This is INACTIVE
            # in every physical cell (the implied T already lies inside the
            # band ⇒ byte-identical strict ρE conservation); it only bounds the
            # pathological split, degrading gracefully while staying positive.
            if frapolli_mode == Int32(1)
                cv_safe_rhoe = max(cv_thermal, 1.0f-6)
                ekin_lab_rhoe = sum_ekin * inv_children
                uxl_rhoe = ux_parent - shift_x
                uyl_rhoe = uy_parent - shift_y
                uzl_rhoe = uz_parent - shift_z
                usq_lab_rhoe = uxl_rhoe*uxl_rhoe + uyl_rhoe*uyl_rhoe + uzl_rhoe*uzl_rhoe
                # 2·ρ·Cv·T = E_kin − ρ|u|²_lab + 2·Σg  ⇒ solve for the Σg that
                # corresponds to the T-band edges.
                two_rho_cv = 2.0f0 * rho_parent * cv_safe_rhoe
                base_e = ekin_lab_rhoe - rho_parent * usq_lab_rhoe
                sumg_for_Tmin = 0.5f0 * (two_rho_cv * 0.1f0 - base_e)
                sumg_for_Tmax = 0.5f0 * (two_rho_cv * 10.0f0 - base_e)
                target_sum_g = clamp(target_sum_g, sumg_for_Tmin, sumg_for_Tmax)
            end
            # Adding δ·w_k to the thermal directions shifts Σg by δ·Σ_{k≤nt} w_k.
            # On D3Q39 the thermal lattice = flow lattice (Σ w_k = 1 over all 39),
            # so Σ_{k≤nt} w_k = 1 and δ = (target − realized) exactly. Compute the
            # partial weight sum explicitly to stay correct if n_dirs_thermal < 39.
            wsum_thermal = 0.0f0
            @inbounds for k in Int32(1):n_dirs_thermal
                wsum_thermal += Float32(w_arr[k])
            end
            if wsum_thermal > 1.0f-6
                delta_g = (target_sum_g - realized_sum_g) / wsum_thermal
                @inbounds for k in Int32(1):n_dirs_thermal
                    wk_g = Float32(w_arr[k])
                    gv = parent_g_current[lx, ly, lz, pb_idx, k] + delta_g * wk_g
                    parent_g_current[lx, ly, lz, pb_idx, k] = max(gv, 1.0f-10)
                end
            end
        end

        # â”€â”€ Lagrava mass redistribution (restriction side) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # The parent populations above were rebuilt from feq + Ï„Â·stress
        # (or feq + Ï„Â·f_neq for the legacy path), then clipped by
        # max(â€¦, 1.0f-10). This sequence is mass-conservative only up to
        # the 13-moment solver tolerance and the occasional clipping â€”
        # small per-cell residuals (~1e-7 per population) accumulate to
        # ~1e-4 density offsets that are plainly visible as teeth at
        # every AMR interface in the schlieren diagnostic.
        #
        # Fix: sum the newly-written parent populations, compute the
        # deficit against the exact target parent_rho (= arithmetic
        # mean of 8 child Ï â€” set at line 249 above), and redistribute
        # the deficit across all 39 populations weighted by w_k. This
        # preserves momentum (Î£ w_k Â· c_k = 0 by lattice isotropy) and
        # higher-order moments to leading order, while driving the
        # first moment (Î£ f = Ï_target) to numerical equality.
        # Moment-consistent redistribution after reconstruction and positivity floors.
        # A weight-only correction fixes mass but can leave the parent populations
        # out of sync with the restricted child momentum.  Correct the conserved
        # mass and momentum moments, then refresh parent macros from the final f/g.
        # Pass187 REVERT: restored fixed 2-pass redistribution.
        # Pass186 extended to 8-pass iterate-to-convergence as a "gold-standard
        # moment conservation" step, but the iteration amplified an asymmetric
        # positivity-floor activation between +c_x and −c_x populations in
        # wake-aligned cells, producing visible streamwise stripes in velocity
        # highpass that weren't there in pass185 (fixed-2-pass). The honest
        # conclusion: the 2-pass form was already tuned to give "good enough"
        # mass+momentum convergence without driving asymmetric clipping into
        # accumulated bias. True gold-standard moment conservation needs to
        # come from a different mechanism (e.g., 13×13 linear solve per cell
        # with stress and energy moments, or cell-averaged Galerkin projection
        # at the restriction step) — not from more iterations of this scheme.
        # Conservative space-time AMR literature frames this as a constrained
        # least-squares population projection. The old isotropic closed form is
        # kept as a fallback, but the primary correction solves the actual
        # weighted Gram system for the active lattice directions.
        m00_corr = 0.0f0; m01_corr = 0.0f0; m02_corr = 0.0f0; m03_corr = 0.0f0
        m11_corr = 0.0f0; m12_corr = 0.0f0; m13_corr = 0.0f0
        m22_corr = 0.0f0; m23_corr = 0.0f0; m33_corr = 0.0f0
        @inbounds for k in Int32(1):Int32(n_dirs_flow)
            cxk_corr = Float32(cx_arr[k])
            cyk_corr = Float32(cy_arr[k])
            czk_corr = Float32(cz_arr[k])
            wk_corr = Float32(w_arr[k])
            m00_corr += wk_corr
            m01_corr += wk_corr * cxk_corr
            m02_corr += wk_corr * cyk_corr
            m03_corr += wk_corr * czk_corr
            m11_corr += wk_corr * cxk_corr * cxk_corr
            m12_corr += wk_corr * cxk_corr * cyk_corr
            m13_corr += wk_corr * cxk_corr * czk_corr
            m22_corr += wk_corr * cyk_corr * cyk_corr
            m23_corr += wk_corr * cyk_corr * czk_corr
            m33_corr += wk_corr * czk_corr * czk_corr
        end
        inv_cs2_corr = 1.0f0 / max(cs2_flow, 1.0f-6)
        # Pass 265 (Tier 1 #1e): pass count is now configurable via
        # advanced.refinement.restriction_correction_passes (default 2).
        for _corr_pass in Int32(1):correction_passes
            rho_reconstructed = 0.0f0
            jx_reconstructed = 0.0f0
            jy_reconstructed = 0.0f0
            jz_reconstructed = 0.0f0
            @inbounds for k in Int32(1):Int32(n_dirs_flow)
                fval = PARENT_IS_ET ? pfloc[k] : parent_f_current[lx, ly, lz, pb_idx, k]
                cxk_corr = Float32(cx_arr[k])
                cyk_corr = Float32(cy_arr[k])
                czk_corr = Float32(cz_arr[k])
                rho_reconstructed += fval
                jx_reconstructed += fval * cxk_corr
                jy_reconstructed += fval * cyk_corr
                jz_reconstructed += fval * czk_corr
            end

            deficit_rho = rho_parent - rho_reconstructed
            deficit_jx = jx - jx_reconstructed
            deficit_jy = jy - jy_reconstructed
            deficit_jz = jz - jz_reconstructed
            if abs(deficit_rho) > Float32(1e-12) ||
               abs(deficit_jx) > Float32(1e-12) ||
               abs(deficit_jy) > Float32(1e-12) ||
               abs(deficit_jz) > Float32(1e-12)
                c0_corr, c1_corr, c2_corr, c3_corr, corr_ok = solve_shell_moment4(
                    m00_corr, m01_corr, m02_corr, m03_corr,
                    m11_corr, m12_corr, m13_corr,
                    m22_corr, m23_corr,
                    m33_corr,
                    deficit_rho, deficit_jx, deficit_jy, deficit_jz
                )
                @inbounds for k in Int32(1):Int32(n_dirs_flow)
                    cxk_corr = Float32(cx_arr[k])
                    cyk_corr = Float32(cy_arr[k])
                    czk_corr = Float32(cz_arr[k])
                    delta = if corr_ok
                        Float32(w_arr[k]) *
                        (c0_corr + c1_corr * cxk_corr +
                         c2_corr * cyk_corr + c3_corr * czk_corr)
                    else
                        Float32(w_arr[k]) *
                        (deficit_rho +
                         inv_cs2_corr * (cxk_corr * deficit_jx +
                                         cyk_corr * deficit_jy +
                                         czk_corr * deficit_jz))
                    end
                    if PARENT_IS_ET
                        pfloc[k] = max(pfloc[k] + delta, 1.0f-10)
                    else
                        parent_f_current[lx, ly, lz, pb_idx, k] =
                            max(parent_f_current[lx, ly, lz, pb_idx, k] + delta, 1.0f-10)
                    end
                end
            end
        end

        # ── SYMMETRY-PLANE FIX (2026-06-05, FIX 1b — CORRECT LOCATION) ──────────
        # restrict_fine_to_parent PROVEN to be the sole source of the L1 y=0 plane
        # u_y runaway (restrict OFF ⇒ u_y≡0; ON ⇒ grows to 0.29). The plane parent
        # cell averages its on-plane child (jy≈0) with its one-cell-into-the-gradient
        # child (jy≠0) → biased Σf·cy → written to parent_vel below → fed back via
        # prolongation → grows. Symmetrize the reconstructed parent_f_current over
        # cy-pairs HERE, just before the final shell recompute, so jy_shell≡0 ⇒
        # uy_shell≡0 ⇒ the probe-read parent_vel[...,2] is exactly 0. (The earlier
        # FIX 1 was pasted into a DEAD never-launched kernel — this is the live path.)
        if sym_y_plane == Int32(1)
            parent_gy_sym = (pby - Int32(1)) * bs + Int32(ly)
            if parent_gy_sym == Int32(1) || parent_gy_sym == parent_ny_global
                @inbounds for kk in Int32(1):n_dirs_flow
                    cyk_sym = cy_arr[kk]
                    if cyk_sym != Int32(0)
                        cxk_sym = cx_arr[kk]; czk_sym = cz_arr[kk]
                        km = kk
                        for j in Int32(1):n_dirs_flow
                            if cx_arr[j] == cxk_sym && cy_arr[j] == -cyk_sym && cz_arr[j] == czk_sym
                                km = j; break
                            end
                        end
                        if km > kk
                            if PARENT_IS_ET
                                avg = 0.5f0 * (pfloc[kk] + pfloc[km])
                                pfloc[kk] = avg; pfloc[km] = avg
                            else
                                avg = 0.5f0 * (parent_f_current[lx, ly, lz, pb_idx, kk] +
                                               parent_f_current[lx, ly, lz, pb_idx, km])
                                parent_f_current[lx, ly, lz, pb_idx, kk] = avg
                                parent_f_current[lx, ly, lz, pb_idx, km] = avg
                            end
                        end
                    end
                end
            end
        end

        rho_shell_raw = 0.0f0
        jx_shell = 0.0f0
        jy_shell = 0.0f0
        jz_shell = 0.0f0
        g_sum_shell = 0.0f0
        ekin_shell = 0.0f0
        # ── ET-AMR parent ENCODE: scatter the reconstructed pfloc into the
        # Esoteric parent_f_current ONCE (race-free, == _et_store_l1!) at the
        # parent's next-load parity. Done BEFORE the shell recompute so the
        # macros below read the final stored state consistently (the shell loop
        # reads pfloc directly for ET — layout-independent moments).
        if PARENT_IS_ET
            @inbounds for q in Int32(1):Int32(9)
                kp = et_pair_kp[q]; km = et_pair_km[q]
                cxk_e = cx_arr[kp]; cyk_e = cy_arr[kp]; czk_e = cz_arr[kp]
                # opp -> own cell; leader -> +c neighbour (parent block grid)
                parent_f_current[lx, ly, lz, pb_idx, parent_phase == Int32(0) ? km : kp] = pfloc[km]
                nlx, nly, nlz, ncb = _et_child_nbr(
                    parent_block_pointer_for_encode, pbx, pby, pbz, Int32(lx), Int32(ly), Int32(lz),
                    cxk_e, cyk_e, czk_e, bs, parent_dim_x_enc, parent_dim_y_enc, parent_dim_z_enc)
                if ncb > Int32(0)
                    parent_f_current[nlx, nly, nlz, ncb, parent_phase == Int32(0) ? kp : km] = pfloc[kp]
                end
            end
            parent_f_current[lx, ly, lz, pb_idx, 1] = pfloc[1]
        end

        @inbounds for k in Int32(1):Int32(n_dirs_flow)
            fval = PARENT_IS_ET ? pfloc[k] : parent_f_current[lx, ly, lz, pb_idx, k]
            cxk_shell = Float32(cx_arr[k])
            cyk_shell = Float32(cy_arr[k])
            czk_shell = Float32(cz_arr[k])
            rho_shell_raw += fval
            jx_shell += fval * cxk_shell
            jy_shell += fval * cyk_shell
            jz_shell += fval * czk_shell
            cxu_shell = cxk_shell - shift_x
            cyu_shell = cyk_shell - shift_y
            czu_shell = czk_shell - shift_z
            ekin_shell += fval * (cxu_shell * cxu_shell +
                                  cyu_shell * cyu_shell +
                                  czu_shell * czu_shell)
        end

        rho_shell = max(rho_shell_raw, 0.01f0)
        inv_rho_shell = 1.0f0 / rho_shell
        ux_shell = jx_shell * inv_rho_shell
        uy_shell = jy_shell * inv_rho_shell
        uz_shell = jz_shell * inv_rho_shell
        parent_rho[lx, ly, lz, pb_idx] = rho_shell
        parent_vel[lx, ly, lz, pb_idx, 1] = ux_shell
        parent_vel[lx, ly, lz, pb_idx, 2] = uy_shell
        parent_vel[lx, ly, lz, pb_idx, 3] = uz_shell

        if restrict_thermal == Int32(1)
            @inbounds for k in Int32(1):Int32(n_dirs_thermal)
                g_sum_shell += parent_g_current[lx, ly, lz, pb_idx, k]
            end
            ux_lab = ux_shell - shift_x
            uy_lab = uy_shell - shift_y
            uz_lab = uz_shell - shift_z
            u_sq_lab = ux_lab * ux_lab + uy_lab * uy_lab + uz_lab * uz_lab
            temp_shell = if frapolli_mode == Int32(1)
                cv_safe = max(cv_thermal, 1.0f-6)
                (ekin_shell - rho_shell * u_sq_lab + 2.0f0 * g_sum_shell) /
                (2.0f0 * rho_shell * cv_safe)
            else
                g_sum_shell
            end

            parent_temperature[lx, ly, lz, pb_idx] =
                clamp(max(temp_shell, 0.01f0), 0.1f0, 10.0f0)
            parent_kinetic_energy[lx, ly, lz, pb_idx] = ekin_shell
        end

        if reset_pi_neq == Int32(1)
            if preserve_pi_neq == Int32(1)
                # Pi_neq also scales by Ï„-ratio (same reason as f_neq); blend with
                # prior for partial restriction.
                pxx_prior = parent_pi_neq[lx, ly, lz, pb_idx, 1]
                pyy_prior = parent_pi_neq[lx, ly, lz, pb_idx, 2]
                pzz_prior = parent_pi_neq[lx, ly, lz, pb_idx, 3]
                pxy_prior = parent_pi_neq[lx, ly, lz, pb_idx, 4]
                pxz_prior = parent_pi_neq[lx, ly, lz, pb_idx, 5]
                pyz_prior = parent_pi_neq[lx, ly, lz, pb_idx, 6]
                parent_pi_neq[lx, ly, lz, pb_idx, 1] =
                    frac * tau_ratio_flow * pxx_parent + one_m_frac * pxx_prior
                parent_pi_neq[lx, ly, lz, pb_idx, 2] =
                    frac * tau_ratio_flow * pyy_parent + one_m_frac * pyy_prior
                parent_pi_neq[lx, ly, lz, pb_idx, 3] =
                    frac * tau_ratio_flow * pzz_parent + one_m_frac * pzz_prior
                parent_pi_neq[lx, ly, lz, pb_idx, 4] =
                    frac * tau_ratio_flow * pxy_parent + one_m_frac * pxy_prior
                parent_pi_neq[lx, ly, lz, pb_idx, 5] =
                    frac * tau_ratio_flow * pxz_parent + one_m_frac * pxz_prior
                parent_pi_neq[lx, ly, lz, pb_idx, 6] =
                    frac * tau_ratio_flow * pyz_parent + one_m_frac * pyz_prior
            else
                @inbounds for comp in Int32(1):Int32(6)
                    parent_pi_neq[lx, ly, lz, pb_idx, comp] = 0.0f0
                end
            end
        end
        end
    end
end

@kernel function restrict_partial_child_state_to_parent_kernel!(
    parent_rho,
    parent_vel,
    parent_temperature,
    parent_kinetic_energy,
    parent_f_current,
    parent_g_current,
    parent_pi_neq,
    parent_obstacle,
    parent_map_x,
    parent_map_y,
    parent_map_z,
    child_block_pointer,
    child_obstacle,
    child_temperature,
    child_kinetic_energy,
    child_f_current,
    child_g_current,
    child_pi_neq,
    cx_arr,
    cy_arr,
    cz_arr,
    w_arr,
    block_size::Int32,
    child_dim_x::Int32,
    child_dim_y::Int32,
    child_dim_z::Int32,
    n_dirs_flow::Int32,
    n_dirs_thermal::Int32,
    restrict_thermal::Int32,
    reset_pi_neq::Int32,
    preserve_pi_neq::Int32,
    # PHASE 2b register diet (2026-06-12): Val type params — see the main
    # restriction kernel's note (same dead-lattice register-poison fix).
    ::Val{LATTICE_IS_D3Q39},
    flow_is_compressible::Int32,
    ::Val{FRAPOLLI_MODE},
    ::Val{MIRROR_ENTROPIC},
    cs2_flow::Float32,
    # SOTA A3 — thermodynamic EOS R = (γ-1)·cv. MUST match the value the
    # collision kernel passes as `eos_R_flow` so the restriction f_eq solves
    # use the IDENTICAL equilibrium across the AMR interface (no spurious p/ρ
    # step at T≠1). Sentinel -1 ⇒ cs² fallback (legacy linearized EOS); at
    # T=1 the eos_R term vanishes ⇒ freestream/non-thermodynamic byte-identical.
    eos_R_flow::Float32,
    cs2_thermal::Float32,
    cv_thermal::Float32,
    shift_x::Float32,
    shift_y::Float32,
    shift_z::Float32,
    tau_ratio_flow::Float32,
    tau_ratio_thermal::Float32
) where {LATTICE_IS_D3Q39, FRAPOLLI_MODE, MIRROR_ENTROPIC}
    lattice_is_d3q39 = LATTICE_IS_D3Q39 ? Int32(1) : Int32(0)
    frapolli_mode = FRAPOLLI_MODE ? Int32(1) : Int32(0)
    mirror_decompose_entropic = MIRROR_ENTROPIC ? Int32(1) : Int32(0)
    # The partial-frontier kernel is NEVER run under ET (it's auto-disabled for D3Q19 at
    # config-load AND gated `!child_is_et` at the call site). It has no CHILD_IS_ET Val
    # param, but its body references CHILD_IS_ET in the use_pi_stress expression below
    # (an incomplete ET-port carryover). Bind it to a local `false` so the reference is
    # defined (was a latent undefined-symbol landmine if the call-site guard ever changed).
    CHILD_IS_ET = false
    lx, ly, lz, pb_idx = @index(Global, NTuple)
    bs = block_size

    if !parent_obstacle[lx, ly, lz, pb_idx]
        pbx = parent_map_x[pb_idx]
        pby = parent_map_y[pb_idx]
        pbz = parent_map_z[pb_idx]

        child_base_x = Int32(2) * pbx - Int32(1)
        child_base_y = Int32(2) * pby - Int32(1)
        child_base_z = Int32(2) * pbz - Int32(1)

        child_lx = MVector{8, Int32}(undef)
        child_ly = MVector{8, Int32}(undef)
        child_lz = MVector{8, Int32}(undef)
        child_block = MVector{8, Int32}(undef)
        child_valid = MVector{8, Int32}(undef)
        child_sx = MVector{8, Int32}(undef)
        child_sy = MVector{8, Int32}(undef)
        child_sz = MVector{8, Int32}(undef)

        valid_children = Int32(0)
        child_slot = 1

        @inbounds for sx in Int32(0):Int32(1), sy in Int32(0):Int32(1), sz in Int32(0):Int32(1)
            fx = Int32(2) * (Int32(lx) - Int32(1)) + Int32(1) + sx
            fy = Int32(2) * (Int32(ly) - Int32(1)) + Int32(1) + sy
            fz = Int32(2) * (Int32(lz) - Int32(1)) + Int32(1) + sz

            child_sx[child_slot] = sx
            child_sy[child_slot] = sy
            child_sz[child_slot] = sz
            child_lx[child_slot] = Int32(0)
            child_ly[child_slot] = Int32(0)
            child_lz[child_slot] = Int32(0)
            child_block[child_slot] = Int32(0)
            child_valid[child_slot] = Int32(0)

        cbx = child_base_x + div(fx - Int32(1), bs)
        cby = child_base_y + div(fy - Int32(1), bs)
        cbz = child_base_z + div(fz - Int32(1), bs)

            if cbx >= Int32(1) && cbx <= child_dim_x &&
               cby >= Int32(1) && cby <= child_dim_y &&
               cbz >= Int32(1) && cbz <= child_dim_z
                cb_idx = child_block_pointer[cbx, cby, cbz]
                if cb_idx != Int32(0)
                    clx_tmp = (fx - Int32(1)) % bs + Int32(1)
                    cly_tmp = (fy - Int32(1)) % bs + Int32(1)
                    clz_tmp = (fz - Int32(1)) % bs + Int32(1)
                    if child_obstacle[clx_tmp, cly_tmp, clz_tmp, cb_idx]
                        child_slot += 1
                        continue
                    end
                    valid_children += Int32(1)
                    child_valid[child_slot] = Int32(1)
                    child_lx[child_slot] = clx_tmp
                    child_ly[child_slot] = cly_tmp
                    child_lz[child_slot] = clz_tmp
                    child_block[child_slot] = cb_idx
                end
            end

            child_slot += 1
        end

        if valid_children > Int32(0) && valid_children < Int32(8)
            # Frontier parents only have partial fine coverage. Even after the
            # missing octants are completed symmetrically, overwriting the whole
            # coarse cell with the child-completed macro state prints a visible
            # band on x-faces once streamwise gradients strengthen: the missing
            # half of the parent gets treated as if it had already reached the
            # refined state. Use the true fine coverage as the conserved-moment
            # blend factor, then rebuild the parent from a local equilibrium.
            # This keeps the unresolved half anchored to the evolved coarse
            # state while still importing the resolved child information.
            frac = Float32(valid_children) * 0.125f0
            one_m_frac = 1.0f0 - frac

            flow_avg = MVector{39, Float32}(undef)
            thermal_avg = MVector{39, Float32}(undef)
            @inbounds for k in 1:39
                flow_avg[k] = 0.0f0
                thermal_avg[k] = 0.0f0
            end

            sum_temp = 0.0f0
            sum_ekin_child = 0.0f0
            sum_pxx = 0.0f0
            sum_pyy = 0.0f0
            sum_pzz = 0.0f0
            sum_pxy = 0.0f0
            sum_pxz = 0.0f0
            sum_pyz = 0.0f0

            completed_slots = Int32(0)
            @inbounds for child_idx in 1:8
                best_dist = Int32(4)
                donor_count = Int32(0)
                @inbounds for donor_idx in 1:8
                    if child_valid[donor_idx] == Int32(1)
                        dist =
                            (child_sx[donor_idx] == child_sx[child_idx] ? Int32(0) : Int32(1)) +
                            (child_sy[donor_idx] == child_sy[child_idx] ? Int32(0) : Int32(1)) +
                            (child_sz[donor_idx] == child_sz[child_idx] ? Int32(0) : Int32(1))
                        if dist < best_dist
                            best_dist = dist
                            donor_count = Int32(1)
                        elseif dist == best_dist
                            donor_count += Int32(1)
                        end
                    end
                end

                donor_count == Int32(0) && continue
                inv_donor = 1.0f0 / Float32(donor_count)
                completed_slots += Int32(1)

                @inbounds for donor_idx in 1:8
                    if child_valid[donor_idx] == Int32(1)
                        dist =
                            (child_sx[donor_idx] == child_sx[child_idx] ? Int32(0) : Int32(1)) +
                            (child_sy[donor_idx] == child_sy[child_idx] ? Int32(0) : Int32(1)) +
                            (child_sz[donor_idx] == child_sz[child_idx] ? Int32(0) : Int32(1))
                        if dist == best_dist
                            clx = child_lx[donor_idx]
                            cly = child_ly[donor_idx]
                            clz = child_lz[donor_idx]
                            cb = child_block[donor_idx]

                            @inbounds for k in 1:39
                                if k <= n_dirs_flow
                                    flow_avg[k] += inv_donor * child_f_current[clx, cly, clz, cb, k]
                                end
                                if restrict_thermal == Int32(1) && k <= n_dirs_thermal
                                    thermal_avg[k] += inv_donor * child_g_current[clx, cly, clz, cb, k]
                                end
                            end

                            if restrict_thermal == Int32(1)
                                sum_temp += inv_donor * child_temperature[clx, cly, clz, cb]
                                sum_ekin_child += inv_donor * child_kinetic_energy[clx, cly, clz, cb]
                            end
                        end
                    end
                end
            end

            inv_valid = completed_slots > Int32(0) ?
                (1.0f0 / Float32(completed_slots)) :
                (1.0f0 / Float32(valid_children))

            rho_child = 0.0f0
            jx_child = 0.0f0
            jy_child = 0.0f0
            jz_child = 0.0f0
            @inbounds for k in 1:39
                if k <= n_dirs_flow
                    favg = flow_avg[k] * inv_valid
                    flow_avg[k] = favg
                    rho_child += favg
                    jx_child += favg * Float32(cx_arr[k])
                    jy_child += favg * Float32(cy_arr[k])
                    jz_child += favg * Float32(cz_arr[k])
                end
                if restrict_thermal == Int32(1) && k <= n_dirs_thermal
                    thermal_avg[k] *= inv_valid
                end
            end

            rho_child = max(rho_child, 0.01f0)
            rho_prior = max(parent_rho[lx, ly, lz, pb_idx], 0.01f0)
            ux_prior = parent_vel[lx, ly, lz, pb_idx, 1]
            uy_prior = parent_vel[lx, ly, lz, pb_idx, 2]
            uz_prior = parent_vel[lx, ly, lz, pb_idx, 3]
            jx_prior = rho_prior * ux_prior
            jy_prior = rho_prior * uy_prior
            jz_prior = rho_prior * uz_prior

            rho_target = max(frac * rho_child + one_m_frac * rho_prior, 0.01f0)
            jx_target = frac * jx_child + one_m_frac * jx_prior
            jy_target = frac * jy_child + one_m_frac * jy_prior
            jz_target = frac * jz_child + one_m_frac * jz_prior
            inv_rho_target = 1.0f0 / rho_target
            ux_target = jx_target * inv_rho_target
            uy_target = jy_target * inv_rho_target
            uz_target = jz_target * inv_rho_target

            T_prior = parent_temperature[lx, ly, lz, pb_idx]
            temp_child = restrict_thermal == Int32(1) ? max(sum_temp * inv_valid, 0.01f0) : 1.0f0
            temp_target = restrict_thermal == Int32(1) ?
                clamp(frac * temp_child + one_m_frac * T_prior, 0.1f0, 10.0f0) : 1.0f0

            # Partial frontier cells do NOT have a complete 2Ã—2Ã—2 child support,
            # so transporting non-equilibrium content here is under-resolved and
            # excites the rectangular AMR box imprint visible in schlieren. Keep
            # only coverage-blended conserved moments and rebuild the parent cell
            # from a local equilibrium. Full 8-child parents still use the richer
            # moment/stress transfer in `restrict_child_state_to_parent_kernel!`.
            # Missing octants are completed symmetrically from the nearest valid
            # child pattern, but the completed child estimate is still blended by
            # the actual fine coverage so the unresolved portion of the parent
            # remains tied to the coarse prior instead of snapping to the refined
            # state in one update.
            pxx_target = 0.0f0
            pyy_target = 0.0f0
            pzz_target = 0.0f0
            pxy_target = 0.0f0
            pxz_target = 0.0f0
            pyz_target = 0.0f0

            use_numerical_eq = lattice_is_d3q39 == Int32(1)
            ux_rel = ux_target - shift_x
            uy_rel = uy_target - shift_y
            uz_rel = uz_target - shift_z
            lambda_eq = use_numerical_eq ?
                solve_13mom_lambdas_d3q39(
                    rho_target, ux_rel, uy_rel, uz_rel,
                    cx_arr, cy_arr, cz_arr, w_arr,
                    cs2_flow, Int32(50), temp_target,
                    shift_x, shift_y, shift_z,
                    100.0f0, eos_R_flow          # SOTA A3 — match collision-kernel EOS R
                ) :
                (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
                 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
            mass_corr_eq = use_numerical_eq ?
                compute_13mom_mass_correction_d3q39(
                    rho_target, lambda_eq,
                    cx_arr, cy_arr, cz_arr, w_arr,
                    cs2_flow, shift_x, shift_y, shift_z
                ) : 1.0f0

            # SOTA #4 (2026-05-29): conserve kinetic energy on the partial-frontier
            # path too. Previously this kernel wrote pure f_eq(ρ,u,T) — mass +
            # momentum + temperature were carried, but the populations' |c|²
            # moment (E_kin) was whatever the 13-mom eq produced, NOT pinned to
            # the children's averaged E_kin. At Ma≳1.2 that drops shock-heating
            # energy at partial-coverage AMR cells. Fix: materialize f_eq, then
            # apply the closed-form 5-moment (mass+3·mom+E_kin) correction
            # toward the frac-blended children E_kin target — the same
            # apply_5momentum_correction_d3q39! the full-coverage kernel uses.
            apply_energy_corr = use_numerical_eq && restrict_thermal == Int32(1) &&
                                frapolli_mode == Int32(1)
            f_eq_part = MVector{39, Float32}(undef)
            @inbounds for k in 1:39
                if k <= n_dirs_flow
                    wk = Float32(w_arr[k])
                    cxk = Float32(cx_arr[k]) - shift_x
                    cyk = Float32(cy_arr[k]) - shift_y
                    czk = Float32(cz_arr[k]) - shift_z
                    feq_k = use_numerical_eq ?
                        mass_corr_eq * eval_13mom_feq_single(
                            rho_target, lambda_eq, cxk, cyk, czk, wk, cs2_flow
                        ) :
                        calculate_equilibrium_auto(
                            rho_target, ux_target, uy_target, uz_target,
                            wk, cxk, cyk, czk,
                            flow_is_compressible
                        )

                    if apply_energy_corr
                        f_eq_part[k] = feq_k
                    else
                        parent_f_current[lx, ly, lz, pb_idx, k] = max(feq_k, 1.0f-10)
                    end
                end

                if restrict_thermal == Int32(1) && k <= n_dirs_thermal
                    wk = Float32(w_arr[k])
                    cxk = Float32(cx_arr[k]) - shift_x
                    cyk = Float32(cy_arr[k]) - shift_y
                    czk = Float32(cz_arr[k]) - shift_z
                    geq_k = if use_numerical_eq && frapolli_mode == Int32(1)
                        # EOS-CONSISTENT g VARIANCE (2026-06-25): match collision-kernel
                        # coeff_g = Cv − 1.5·eos_R_flow; sentinel ≤0 ⇒ cs²_thermal (legacy).
                        R_g = eos_R_flow > 0.0f0 ? eos_R_flow : cs2_thermal
                        coeff_g = cv_thermal - 1.5f0 * R_g
                        feq_flow_k = mass_corr_eq * eval_13mom_feq_single(
                            rho_target, lambda_eq, cxk, cyk, czk, wk, cs2_flow
                        )
                        coeff_g * temp_target * feq_flow_k
                    else
                        inv_cs2_t = 1.0f0 / cs2_thermal
                        c_dot_u = cxk * ux_target + cyk * uy_target + czk * uz_target
                        wk * temp_target * (1.0f0 + c_dot_u * inv_cs2_t)
                    end
                    parent_g_current[lx, ly, lz, pb_idx, k] = max(geq_k, 1.0f-10)
                end
            end

            # SOTA #4: apply the 5-moment energy correction to the materialized
            # f_eq, then write. Target E_kin is the frac-blended children value
            # (in the shifted frame, matching the solver convention). The
            # subsequent 2-pass mass+momentum correction below leaves the energy
            # mode intact (it only touches ρ, j).
            if apply_energy_corr
                ekin_child_target = sum_ekin_child * inv_valid
                ekin_prior = parent_kinetic_energy[lx, ly, lz, pb_idx]
                ekin_unshifted_target = frac * ekin_child_target + one_m_frac * ekin_prior
                jx_rel_t = jx_target - rho_target * shift_x
                jy_rel_t = jy_target - rho_target * shift_y
                jz_rel_t = jz_target - rho_target * shift_z
                # SHIFT-FRAME FIX (2026-05-29): convert the UNSHIFTED-stored E_kin
                # to the SHIFTED frame the correction works in. Identity at shift=0.
                # See the matching conversion in restrict_child_state_to_parent_kernel!.
                U_dot_jrel_p = shift_x * jx_rel_t + shift_y * jy_rel_t + shift_z * jz_rel_t
                U_sq_shift_p = shift_x*shift_x + shift_y*shift_y + shift_z*shift_z
                ekin_target = ekin_unshifted_target + 2.0f0 * U_dot_jrel_p - U_sq_shift_p * rho_target
                apply_5momentum_correction_d3q39!(
                    f_eq_part,
                    rho_target, jx_rel_t, jy_rel_t, jz_rel_t,
                    ekin_target,
                    cx_arr, cy_arr, cz_arr, w_arr,
                    shift_x, shift_y, shift_z
                )
                @inbounds for k in Int32(1):n_dirs_flow
                    parent_f_current[lx, ly, lz, pb_idx, k] = max(f_eq_part[k], 1.0f-10)
                end
            end

            inv_cs2_corr = 1.0f0 / max(cs2_flow, 1.0f-6)
            for _corr_pass in Int32(1):Int32(2)
                rho_reconstructed = 0.0f0
                jx_reconstructed = 0.0f0
                jy_reconstructed = 0.0f0
                jz_reconstructed = 0.0f0
                @inbounds for k in Int32(1):Int32(n_dirs_flow)
                    fval = parent_f_current[lx, ly, lz, pb_idx, k]
                    cxk_corr = Float32(cx_arr[k])
                    cyk_corr = Float32(cy_arr[k])
                    czk_corr = Float32(cz_arr[k])
                    rho_reconstructed += fval
                    jx_reconstructed += fval * cxk_corr
                    jy_reconstructed += fval * cyk_corr
                    jz_reconstructed += fval * czk_corr
                end

                deficit_rho = rho_target - rho_reconstructed
                deficit_jx = jx_target - jx_reconstructed
                deficit_jy = jy_target - jy_reconstructed
                deficit_jz = jz_target - jz_reconstructed
                if abs(deficit_rho) > Float32(1e-12) ||
                   abs(deficit_jx) > Float32(1e-12) ||
                   abs(deficit_jy) > Float32(1e-12) ||
                   abs(deficit_jz) > Float32(1e-12)
                    @inbounds for k in Int32(1):Int32(n_dirs_flow)
                        cxk_corr = Float32(cx_arr[k])
                        cyk_corr = Float32(cy_arr[k])
                        czk_corr = Float32(cz_arr[k])
                        delta = Float32(w_arr[k]) *
                                (deficit_rho +
                                 inv_cs2_corr * (cxk_corr * deficit_jx +
                                                 cyk_corr * deficit_jy +
                                                 czk_corr * deficit_jz))
                        parent_f_current[lx, ly, lz, pb_idx, k] =
                            max(parent_f_current[lx, ly, lz, pb_idx, k] + delta, 1.0f-10)
                    end
                end
            end

            rho_raw = 0.0f0
            jx = 0.0f0
            jy = 0.0f0
            jz = 0.0f0
            g_sum = 0.0f0
            ekin_shell = 0.0f0

            @inbounds for k in 1:39
                if k <= n_dirs_flow
                    fval = parent_f_current[lx, ly, lz, pb_idx, k]
                    rho_raw += fval
                    jx += fval * Float32(cx_arr[k])
                    jy += fval * Float32(cy_arr[k])
                    jz += fval * Float32(cz_arr[k])
                    cxu = Float32(cx_arr[k]) - shift_x
                    cyu = Float32(cy_arr[k]) - shift_y
                    czu = Float32(cz_arr[k]) - shift_z
                    ekin_shell += fval * (cxu * cxu + cyu * cyu + czu * czu)
                end

                if restrict_thermal == Int32(1) && k <= n_dirs_thermal
                    g_sum += parent_g_current[lx, ly, lz, pb_idx, k]
                end
            end

            rho_shell = max(rho_raw, 0.01f0)
            inv_rho_shell = 1.0f0 / rho_shell
            ux_shell = jx * inv_rho_shell
            uy_shell = jy * inv_rho_shell
            uz_shell = jz * inv_rho_shell

            parent_rho[lx, ly, lz, pb_idx] = rho_shell
            parent_vel[lx, ly, lz, pb_idx, 1] = ux_shell
            parent_vel[lx, ly, lz, pb_idx, 2] = uy_shell
            parent_vel[lx, ly, lz, pb_idx, 3] = uz_shell

            if restrict_thermal == Int32(1)
                ux_lab = ux_shell - shift_x
                uy_lab = uy_shell - shift_y
                uz_lab = uz_shell - shift_z
                u_sq_lab = ux_lab * ux_lab + uy_lab * uy_lab + uz_lab * uz_lab
                temp_shell = if frapolli_mode == Int32(1)
                    cv_safe = max(cv_thermal, 1.0f-6)
                    (ekin_shell - rho_shell * u_sq_lab + 2.0f0 * g_sum) /
                    (2.0f0 * rho_shell * cv_safe)
                else
                    g_sum
                end

                parent_temperature[lx, ly, lz, pb_idx] =
                    clamp(max(temp_shell, 0.01f0), 0.1f0, 10.0f0)
                parent_kinetic_energy[lx, ly, lz, pb_idx] = ekin_shell
            end

            if reset_pi_neq == Int32(1)
                @inbounds for comp in Int32(1):Int32(6)
                    parent_pi_neq[lx, ly, lz, pb_idx, comp] = 0.0f0
                end
            end
        end
    end
end

@inline function blended_shell_population(old_f, candidate_f, alpha::Float32)
    f_new = max(candidate_f, 1.0f-10)
    return max((1.0f0 - alpha) * old_f + alpha * f_new, 1.0f-10)
end

@kernel function synchronize_parent_interface_shell_kernel!(
    parent_rho,
    parent_vel,
    parent_rho_old,
    parent_vel_old,
    parent_temperature,
    parent_kinetic_energy,
    parent_f_current,
    parent_g_current,
    parent_pi_neq,
    parent_face_budget,
    parent_obstacle,
    parent_amr_covered,
    parent_block_pointer,
    parent_map_x,
    parent_map_y,
    parent_map_z,
    child_block_pointer,
    child_obstacle,
    child_rho,
    child_vel,
    child_rho_old,
    child_vel_old,
    child_temperature,
    child_kinetic_energy,
    child_f_current,
    child_g_current,
    child_pi_neq,
    cx_arr,
    cy_arr,
    cz_arr,
    w_arr,
    opp_arr,
    block_size::Int32,
    parent_dim_x::Int32,
    parent_dim_y::Int32,
    parent_dim_z::Int32,
    child_dim_x::Int32,
    child_dim_y::Int32,
    child_dim_z::Int32,
    n_dirs_flow::Int32,
    n_dirs_thermal::Int32,
    restrict_thermal::Int32,
    preserve_pi_neq::Int32,
    mirror_decompose_entropic::Int32,
    flow_is_compressible::Int32,
    frapolli_mode::Int32,
    cs2_flow::Float32,
    # SOTA A3 — thermodynamic EOS R = (γ-1)·cv. MUST match the value the
    # collision kernel passes as `eos_R_flow` so the shell/donor f_eq solves
    # use the IDENTICAL equilibrium across the AMR interface (no spurious p/ρ
    # step at T≠1). Sentinel -1 ⇒ cs² fallback (legacy linearized EOS); at
    # T=1 the eos_R term vanishes ⇒ freestream/non-thermodynamic byte-identical.
    eos_R_flow::Float32,
    cs2_thermal::Float32,
    cv_thermal::Float32,
    shift_x::Float32,
    shift_y::Float32,
    shift_z::Float32,
    tau_ratio_flow::Float32,
    tau_ratio_thermal::Float32,
    h_scale_shell::Float32,
    apply_eq_shift::Int32,
    use_moment_shift::Int32,
    include_x_links::Int32,
    population_blend_x::Float32,
    rho_alpha_x_single::Float32,
    rho_alpha_x_multi::Float32,
    rho_alpha_t_single::Float32,
    rho_alpha_t_multi::Float32,
    vel_alpha_x_single::Float32,
    vel_alpha_x_multi::Float32,
    vel_alpha_t::Float32,
    vel_normal_only::Int32,
    drho_cap_x_abs::Float32,
    drho_cap_x_rel::Float32,
    drho_cap_t_abs::Float32,
    drho_cap_t_rel::Float32,
    du_cap_x_abs::Float32,
    du_cap_x_rel::Float32,
    du_cap_t_abs::Float32,
    du_cap_t_rel::Float32,
    flux_alpha_x::Float32,
    flux_alpha_t::Float32,
    flux_du_cap_abs::Float32,
    flux_du_cap_rel::Float32,
    momflux_alpha_x::Float32,
    momflux_drho_cap_abs::Float32,
    momflux_drho_cap_rel::Float32,
    momflux_stress_alpha_x::Float32,
    momflux_stress_cap_abs::Float32,
    momflux_stress_cap_rel::Float32,
    face_weight_x::Float32,
    face_pop_shift::Int32,
    face_pop_alpha::Float32,
    face_pop_side::Int32,
    face_stress_shift::Int32,
    face_stress_alpha::Float32,
    face_stress_rel_cap::Float32,
    face_stress_abs_cap::Float32,
    face_time_snapshots::Int32,
    coupled_reflux::Int32,
    coupled_reflux_alpha::Float32,
    paired_reflux::Int32,
    paired_reflux_alpha::Float32,
    paired_reflux_mass_alpha::Float32,
    paired_reflux_mom_alpha::Float32,
    paired_reflux_mom_alpha_initial::Float32,
    paired_reflux_mom_alpha_hold_steps::Int32,
    paired_reflux_parent_fraction::Float32,
    paired_reflux_mass_parent_fraction::Float32,
    paired_reflux_mass_cap_scale::Float32,
    paired_reflux_mass_cap_initial::Float32,
    paired_reflux_mass_cap_hold_steps::Int32,
    paired_reflux_face_conservative::Int32,
    paired_reflux_coupled_caps::Int32,
    paired_reflux_coupled_mass_caps::Int32,
    paired_reflux_macro_consistent_caps::Int32,
    paired_reflux_macro_pairwise_caps::Int32,
    paired_reflux_budget_mean_preserving::Int32,
    paired_reflux_shared_state::Int32,
    paired_reflux_shared_state_rho_alpha::Float32,
    paired_reflux_highpass::Int32,
    paired_reflux_highpass_blend::Float32,
    paired_reflux_lowpass::Int32,
    paired_reflux_lowpass_blend::Float32,
    paired_reflux_second_moment::Int32,
    paired_reflux_second_moment_blend::Float32,
    paired_reflux_post_stress::Int32,
    paired_reflux_post_stress_alpha::Float32,
    paired_reflux_post_stress_soft_limit::Int32,
    gpu_budget_reflux::Int32,
    gpu_budget_reflux_alpha::Float32,
    gpu_budget_reflux_mass_alpha::Float32,
    gpu_budget_reflux_mom_alpha::Float32,
    gpu_budget_reflux_drho_cap_abs::Float32,
    gpu_budget_reflux_drho_cap_rel::Float32,
    gpu_budget_reflux_mom_cap_abs::Float32,
    gpu_budget_reflux_mom_cap_rel::Float32,
    gpu_budget_reflux_neutral_mom::Int32,
    current_top_step::Int32,
    shell_recalibration::Int32,
    well_balanced_rebuild::Int32
)
    lx, ly, lz, pb_idx = @index(Global, NTuple)
    bs = block_size

    gx = (parent_map_x[pb_idx] - Int32(1)) * bs + Int32(lx)
    gy = (parent_map_y[pb_idx] - Int32(1)) * bs + Int32(ly)
    gz = (parent_map_z[pb_idx] - Int32(1)) * bs + Int32(lz)

    active_cell = !parent_obstacle[lx, ly, lz, pb_idx] &&
                  count_fine_children(
                      child_block_pointer,
                      child_dim_x, child_dim_y, child_dim_z,
                      bs,
                      gx, gy, gz
                  ) == Int32(0)

    if active_cell
        # Incoming shell update with a moment-consistent refined donor:
        # average the refined mother-cell children in macro / stress space,
        # rebuild the incoming coarse-shell population from feq + Ï„Â·Î , and only
        # fall back to raw population mirror decomposition when Î _neq is not
        # tracked. This keeps the shell compatible with the coarse entropic
        # equilibrium instead of importing unresolved child-grid content
        # direction-by-direction, which was printing a persistent x-face seam.
        have_updates = false
        # ET-AMR (STAGE 4): force the Π-CARRIER reconstruction (feq + τ·project(Π))
        # for ET, NOT the legacy flow_avg-fneq path. For ET the population average
        # flow_avg is the post-stream (shifted) state — wrong at gradients. The Π
        # carrier uses the averaged child_pi_neq (correct per-cell stress) + the
        # overridden child-macro feq. Works for D3Q19 (project_second_order_stress_mode
        # is the same 2nd-order Hermite projection prolongation uses). Needs preserve_pi_neq.
        use_pi_stress = (mirror_decompose_entropic == Int32(1) || CHILD_IS_ET) &&
                        preserve_pi_neq == Int32(1)
        # Keep shell populations donor-consistent whenever we have any
        # non-equilibrium carrier. A pure parent-shell equilibrium anchor
        # creates a level-dependent velocity state that shows up as vorticity
        # on AMR faces, most visibly on x-facing interfaces.
        shell_equilibrium_only = mirror_decompose_entropic == Int32(1) &&
                                 preserve_pi_neq == Int32(0) &&
                                 tau_ratio_flow <= 0.0f0
        use_numerical_eq = mirror_decompose_entropic == Int32(1) && n_dirs_flow == Int32(39)
        rho_shell_ref = max(parent_rho[lx, ly, lz, pb_idx], 0.01f0)
        ux_shell_ref = parent_vel[lx, ly, lz, pb_idx, 1]
        uy_shell_ref = parent_vel[lx, ly, lz, pb_idx, 2]
        uz_shell_ref = parent_vel[lx, ly, lz, pb_idx, 3]
        temp_shell_ref = restrict_thermal == Int32(1) ?
            clamp(max(parent_temperature[lx, ly, lz, pb_idx], 0.01f0), 0.1f0, 10.0f0) :
            1.0f0
        shell_lambda_eq = use_numerical_eq ?
            solve_13mom_lambdas_d3q39(
                rho_shell_ref,
                ux_shell_ref - shift_x,
                uy_shell_ref - shift_y,
                uz_shell_ref - shift_z,
                cx_arr, cy_arr, cz_arr, w_arr,
                cs2_flow, Int32(50), temp_shell_ref,
                shift_x, shift_y, shift_z,
                100.0f0, eos_R_flow          # SOTA A3 — match collision-kernel EOS R
            ) :
            (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
             0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
        shell_mass_corr_eq = use_numerical_eq ?
            compute_13mom_mass_correction_d3q39(
                rho_shell_ref, shell_lambda_eq,
                cx_arr, cy_arr, cz_arr, w_arr,
                cs2_flow, shift_x, shift_y, shift_z
            ) : 1.0f0

        @inbounds for k in 1:39
            cxk_int = Int32(cx_arr[k])
            cyk_int = Int32(cy_arr[k])
            czk_int = Int32(cz_arr[k])
            src_gx = gx - cxk_int
            src_gy = gy - cyk_int
            src_gz = gz - czk_int

            # Historical runs used transverse-only shell sync. The AMR
            # microscope can enable X-bearing links too, which directly tests
            # streamwise interface continuity instead of hiding those faces.
            if (include_x_links == Int32(0) && cxk_int != Int32(0)) ||
               (cxk_int == Int32(0) && cyk_int == Int32(0) && czk_int == Int32(0))
                continue
            end

            if coarse_cell_has_fine_children(
                child_block_pointer,
                child_dim_x, child_dim_y, child_dim_z,
                bs,
                src_gx, src_gy, src_gz
            )
                mother_fx0 = Int32(2) * src_gx - Int32(1)
                mother_fy0 = Int32(2) * src_gy - Int32(1)
                mother_fz0 = Int32(2) * src_gz - Int32(1)
                sum_rho = 0.0f0
                sum_jx = 0.0f0
                sum_jy = 0.0f0
                sum_jz = 0.0f0
                sum_temp = 0.0f0
                sum_f_k = 0.0f0
                sum_f_kb = 0.0f0
                sum_g_k = 0.0f0
                sum_pxx = 0.0f0
                sum_pyy = 0.0f0
                sum_pzz = 0.0f0
                sum_pxy = 0.0f0
                sum_pxz = 0.0f0
                sum_pyz = 0.0f0
                valid_children = Int32(0)
                kopp = opp_arr[k]

                # Pass152: shell populations must be restricted from the
                # fine subset that faces the shell cell, not the whole donor
                # parent volume. Axial transverse links use a 2x2 fine face;
                # transverse diagonals use the 2-cell fine edge.
                sx_first = cxk_int == Int32(0) ? Int32(0) : (cxk_int > Int32(0) ? Int32(1) : Int32(0))
                sx_last = cxk_int == Int32(0) ? Int32(1) : sx_first
                sy_first = cyk_int == Int32(0) ? Int32(0) : (cyk_int > Int32(0) ? Int32(1) : Int32(0))
                sy_last = cyk_int == Int32(0) ? Int32(1) : sy_first
                sz_first = czk_int == Int32(0) ? Int32(0) : (czk_int > Int32(0) ? Int32(1) : Int32(0))
                sz_last = czk_int == Int32(0) ? Int32(1) : sz_first

                @inbounds for sx in sx_first:sx_last, sy in sy_first:sy_last, sz in sz_first:sz_last
                    clx, cly, clz, cb, valid_child = lookup_child_cell(
                        child_block_pointer, child_dim_x, child_dim_y, child_dim_z,
                        bs,
                        mother_fx0 + sx,
                        mother_fy0 + sy,
                        mother_fz0 + sz
                    )
                    if valid_child && !child_obstacle[clx, cly, clz, cb]
                        rho_child = child_rho[clx, cly, clz, cb]
                        ux_child = child_vel[clx, cly, clz, cb, 1]
                        uy_child = child_vel[clx, cly, clz, cb, 2]
                        uz_child = child_vel[clx, cly, clz, cb, 3]
                        sum_rho += rho_child
                        sum_jx += rho_child * ux_child
                        sum_jy += rho_child * uy_child
                        sum_jz += rho_child * uz_child
                        if k <= n_dirs_flow
                            sum_f_k += child_f_current[clx, cly, clz, cb, k]
                            if mirror_decompose_entropic == Int32(1)
                                sum_f_kb += child_f_current[clx, cly, clz, cb, kopp]
                            end
                        end
                        if restrict_thermal == Int32(1)
                            sum_temp += child_temperature[clx, cly, clz, cb]
                            if k <= n_dirs_thermal
                                sum_g_k += child_g_current[clx, cly, clz, cb, k]
                            end
                        end
                        if preserve_pi_neq == Int32(1)
                            sum_pxx += child_pi_neq[clx, cly, clz, cb, 1]
                            sum_pyy += child_pi_neq[clx, cly, clz, cb, 2]
                            sum_pzz += child_pi_neq[clx, cly, clz, cb, 3]
                            sum_pxy += child_pi_neq[clx, cly, clz, cb, 4]
                            sum_pxz += child_pi_neq[clx, cly, clz, cb, 5]
                            sum_pyz += child_pi_neq[clx, cly, clz, cb, 6]
                        end
                        valid_children += Int32(1)
                    end
                end

                if valid_children > Int32(0)
                    inv_children = 1.0f0 / Float32(valid_children)

                    if k <= n_dirs_flow
                        rho_donor = max(sum_rho * inv_children, 0.01f0)
                        inv_rho_donor = 1.0f0 / rho_donor
                        ux_donor = sum_jx * inv_children * inv_rho_donor
                        uy_donor = sum_jy * inv_children * inv_rho_donor
                        uz_donor = sum_jz * inv_children * inv_rho_donor
                        temp_donor = restrict_thermal == Int32(1) ? max(sum_temp * inv_children, 0.01f0) : 1.0f0
                        temp_clamped = clamp(temp_donor, 0.1f0, 10.0f0)

                        cxk = Float32(cx_arr[k]) - shift_x
                        cyk = Float32(cy_arr[k]) - shift_y
                        czk = Float32(cz_arr[k]) - shift_z
                        wk = Float32(w_arr[k])
                        pop_blend_k = cxk_int == Int32(0) ? 1.0f0 : population_blend_x

                        feq_k = use_numerical_eq ?
                            shell_mass_corr_eq * eval_13mom_feq_single(
                                rho_shell_ref, shell_lambda_eq, cxk, cyk, czk, wk, cs2_flow
                            ) :
                            calculate_equilibrium_auto(
                                rho_shell_ref, ux_shell_ref, uy_shell_ref, uz_shell_ref,
                                wk, cxk, cyk, czk,
                                flow_is_compressible
                            )

                        if shell_equilibrium_only
                            parent_f_current[lx, ly, lz, pb_idx, k] =
                                blended_shell_population(parent_f_current[lx, ly, lz, pb_idx, k], feq_k, pop_blend_k)
                        elseif use_pi_stress
                            lambda_eq = use_numerical_eq ?
                                solve_13mom_lambdas_d3q39(
                                    rho_donor, ux_donor - shift_x, uy_donor - shift_y, uz_donor - shift_z,
                                    cx_arr, cy_arr, cz_arr, w_arr,
                                    cs2_flow, Int32(50), temp_clamped,
                                    shift_x, shift_y, shift_z,
                                    100.0f0, eos_R_flow          # SOTA A3 — match collision-kernel EOS R
                                ) :
                                (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
                                 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
                            mass_corr_eq = use_numerical_eq ?
                                compute_13mom_mass_correction_d3q39(
                                    rho_donor, lambda_eq,
                                    cx_arr, cy_arr, cz_arr, w_arr,
                                    cs2_flow, shift_x, shift_y, shift_z
                                ) : 1.0f0
                            donor_feq_k = use_numerical_eq ?
                                mass_corr_eq * eval_13mom_feq_single(
                                    rho_donor, lambda_eq, cxk, cyk, czk, wk, cs2_flow
                                ) :
                                calculate_equilibrium_auto(
                                    rho_donor, ux_donor, uy_donor, uz_donor,
                                    wk, cxk, cyk, czk,
                                    flow_is_compressible
                                )
                            pxx_donor = sum_pxx * inv_children
                            pyy_donor = sum_pyy * inv_children
                            pzz_donor = sum_pzz * inv_children
                            pxy_donor = sum_pxy * inv_children
                            pxz_donor = sum_pxz * inv_children
                            pyz_donor = sum_pyz * inv_children
                            stress_k = project_second_order_stress_mode(
                                wk, cxk, cyk, czk, cs2_flow,
                                pxx_donor, pyy_donor, pzz_donor,
                                pxy_donor, pxz_donor, pyz_donor
                            )
                            parent_f_current[lx, ly, lz, pb_idx, k] =
                                blended_shell_population(
                                    parent_f_current[lx, ly, lz, pb_idx, k],
                                    donor_feq_k + tau_ratio_flow * stress_k,
                                    pop_blend_k
                                )
                        elseif mirror_decompose_entropic == Int32(1)
                            cxk_b = Float32(cx_arr[kopp]) - shift_x
                            cyk_b = Float32(cy_arr[kopp]) - shift_y
                            czk_b = Float32(cz_arr[kopp]) - shift_z
                            wk_b = Float32(w_arr[kopp])
                            lambda_eq = use_numerical_eq ?
                                solve_13mom_lambdas_d3q39(
                                    rho_donor, ux_donor - shift_x, uy_donor - shift_y, uz_donor - shift_z,
                                    cx_arr, cy_arr, cz_arr, w_arr,
                                    cs2_flow, Int32(50), temp_clamped,
                                    shift_x, shift_y, shift_z,
                                    100.0f0, eos_R_flow          # SOTA A3 — match collision-kernel EOS R
                                ) :
                                (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
                                 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
                            mass_corr_eq = use_numerical_eq ?
                                compute_13mom_mass_correction_d3q39(
                                    rho_donor, lambda_eq,
                                    cx_arr, cy_arr, cz_arr, w_arr,
                                    cs2_flow, shift_x, shift_y, shift_z
                                ) : 1.0f0
                            donor_feq_k = use_numerical_eq ?
                                mass_corr_eq * eval_13mom_feq_single(
                                    rho_donor, lambda_eq, cxk, cyk, czk, wk, cs2_flow
                                ) :
                                calculate_equilibrium_auto(
                                    rho_donor, ux_donor, uy_donor, uz_donor,
                                    wk, cxk, cyk, czk,
                                    flow_is_compressible
                                )
                            feq_kb = use_numerical_eq ?
                                mass_corr_eq * eval_13mom_feq_single(
                                    rho_donor, lambda_eq, cxk_b, cyk_b, czk_b, wk_b, cs2_flow
                                ) :
                                calculate_equilibrium_auto(
                                    rho_donor, ux_donor, uy_donor, uz_donor,
                                    wk_b, cxk_b, cyk_b, czk_b,
                                    flow_is_compressible
                                )
                            favg_k = sum_f_k * inv_children
                            favg_kb = sum_f_kb * inv_children
                            s_k = 0.5f0 * ((favg_k - donor_feq_k) + (favg_kb - feq_kb))
                            h_k = 0.5f0 * ((favg_k - donor_feq_k) - (favg_kb - feq_kb))
                            parent_f_current[lx, ly, lz, pb_idx, k] =
                                blended_shell_population(
                                    parent_f_current[lx, ly, lz, pb_idx, k],
                                    donor_feq_k + tau_ratio_flow * s_k + h_scale_shell * tau_ratio_flow * h_k,
                                    pop_blend_k
                                )
                        else
                            lambda_eq = use_numerical_eq ?
                                solve_13mom_lambdas_d3q39(
                                    rho_donor, ux_donor - shift_x, uy_donor - shift_y, uz_donor - shift_z,
                                    cx_arr, cy_arr, cz_arr, w_arr,
                                    cs2_flow, Int32(50), temp_clamped,
                                    shift_x, shift_y, shift_z,
                                    100.0f0, eos_R_flow          # SOTA A3 — match collision-kernel EOS R
                                ) :
                                (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
                                 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
                            mass_corr_eq = use_numerical_eq ?
                                compute_13mom_mass_correction_d3q39(
                                    rho_donor, lambda_eq,
                                    cx_arr, cy_arr, cz_arr, w_arr,
                                    cs2_flow, shift_x, shift_y, shift_z
                                ) : 1.0f0
                            donor_feq_k = use_numerical_eq ?
                                mass_corr_eq * eval_13mom_feq_single(
                                    rho_donor, lambda_eq, cxk, cyk, czk, wk, cs2_flow
                                ) :
                                calculate_equilibrium_auto(
                                    rho_donor, ux_donor, uy_donor, uz_donor,
                                    wk, cxk, cyk, czk,
                                    flow_is_compressible
                                )
                            favg_k = sum_f_k * inv_children
                            parent_f_current[lx, ly, lz, pb_idx, k] =
                                blended_shell_population(
                                    parent_f_current[lx, ly, lz, pb_idx, k],
                                    donor_feq_k + tau_ratio_flow * (favg_k - donor_feq_k),
                                    pop_blend_k
                                )
                        end
                        have_updates = true
                    end

                    if restrict_thermal == Int32(1) && k <= n_dirs_thermal
                        rho_donor = max(sum_rho * inv_children, 0.01f0)
                        inv_rho_donor = 1.0f0 / rho_donor
                        ux_donor = sum_jx * inv_children * inv_rho_donor
                        uy_donor = sum_jy * inv_children * inv_rho_donor
                        uz_donor = sum_jz * inv_children * inv_rho_donor
                        temp_donor = max(sum_temp * inv_children, 0.01f0)
                        temp_clamped = clamp(temp_donor, 0.1f0, 10.0f0)

                        cxk = Float32(cx_arr[k]) - shift_x
                        cyk = Float32(cy_arr[k]) - shift_y
                        czk = Float32(cz_arr[k]) - shift_z
                        wk = Float32(w_arr[k])

                        lambda_eq = use_numerical_eq ?
                            solve_13mom_lambdas_d3q39(
                                rho_donor, ux_donor - shift_x, uy_donor - shift_y, uz_donor - shift_z,
                                cx_arr, cy_arr, cz_arr, w_arr,
                                cs2_flow, Int32(50), temp_clamped,
                                shift_x, shift_y, shift_z,
                                100.0f0, eos_R_flow          # SOTA A3 — match collision-kernel EOS R
                            ) :
                            (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
                             0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
                        mass_corr_eq = use_numerical_eq ?
                            compute_13mom_mass_correction_d3q39(
                                rho_donor, lambda_eq,
                                cx_arr, cy_arr, cz_arr, w_arr,
                                cs2_flow, shift_x, shift_y, shift_z
                            ) : 1.0f0

                        geq_shell_k = if use_numerical_eq && frapolli_mode == Int32(1)
                            # EOS-CONSISTENT g VARIANCE (2026-06-25): match collision-kernel
                            # coeff_g = Cv − 1.5·eos_R_flow; sentinel ≤0 ⇒ cs²_thermal (legacy).
                            R_g = eos_R_flow > 0.0f0 ? eos_R_flow : cs2_thermal
                            coeff_g = cv_thermal - 1.5f0 * R_g
                            feq_flow_k = shell_mass_corr_eq * eval_13mom_feq_single(
                                rho_shell_ref, shell_lambda_eq, cxk, cyk, czk, wk, cs2_flow
                            )
                            coeff_g * temp_shell_ref * feq_flow_k
                        else
                            inv_cs2_t = 1.0f0 / cs2_thermal
                            wk * temp_shell_ref * (1.0f0 + (cxk * ux_shell_ref + cyk * uy_shell_ref + czk * uz_shell_ref) * inv_cs2_t)
                        end
                        geq_k = if use_numerical_eq && frapolli_mode == Int32(1)
                            R_g = eos_R_flow > 0.0f0 ? eos_R_flow : cs2_thermal
                            coeff_g = cv_thermal - 1.5f0 * R_g
                            feq_flow_k = mass_corr_eq * eval_13mom_feq_single(
                                rho_donor, lambda_eq, cxk, cyk, czk, wk, cs2_flow
                            )
                            coeff_g * temp_clamped * feq_flow_k
                        else
                            inv_cs2_t = 1.0f0 / cs2_thermal
                            wk * temp_clamped * (1.0f0 + (cxk * ux_donor + cyk * uy_donor + czk * uz_donor) * inv_cs2_t)
                        end
                        gavg_k = sum_g_k * inv_children
                        parent_g_current[lx, ly, lz, pb_idx, k] =
                            max(geq_k + tau_ratio_thermal * (gavg_k - geq_k), 1.0f-10)
                    end
                end
            end
        end

        # â”€â”€ Pass 2: outgoing-population bidirectional sync â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # The loop above syncs f_k for direction k whose SOURCE (gx - c_k) has
        # fine children â€” i.e., populations incoming to the shell cell from
        # the refined neighbour. Without a matching outgoing pass, the shell
        # cell's populations streaming INTO the refined region (cx_k points
        # toward the box) remain coarse-self-evolved. Those unsynced pops
        # still contribute to the macro refresh below, so Ï, u inherit a
        # hybrid coarse/fine state. Prolongation then writes that hybrid
        # state into the fine halo â†’ a visible 1-cell column of lower Ï
        # at the +x (upstream) face of every refinement box, and the
        # corresponding box-outline in u (both derived from the same
        # populations).
        #
        # This second pass handles the direction where dst = gx + c_k has
        # fine children AND src = gx - c_k does NOT (to avoid double-syncing
        # populations already handled by the first loop, e.g. in a corner
        # sandwiched between two refinement boxes). It uses the fine
        # children at dst as the donor and reconstructs f_k via the same
        # feq + Ï„Â·stress formula the first loop uses. After this pass the
        # macro refresh produces a fully-consistent Ï, u â€” eliminating the
        # asymmetric upstream-face seam that flow convection (Pe â‰ˆ 80 along
        # x) otherwise stretches into a visible band.
        @inbounds for k in 1:39
            cxk_int = Int32(cx_arr[k])
            cyk_int = Int32(cy_arr[k])
            czk_int = Int32(cz_arr[k])
            if cxk_int == Int32(0) && cyk_int == Int32(0) && czk_int == Int32(0)
                continue
            end
            # Keep the outgoing pass symmetric with the incoming shell pass:
            # when AMR_SHELL_SYNC_INCLUDE_X is enabled, streamwise populations
            # must be refreshed too. Leaving them coarse-self-evolved gives a
            # half-synced parent shell and shows up as the dominant +/-x AMR
            # high-pass line in uniform/acoustic tests.
            if include_x_links == Int32(0) && cxk_int != Int32(0)
                continue
            end

            dst_gx = gx + cxk_int
            dst_gy = gy + cyk_int
            dst_gz = gz + czk_int
            src_gx = gx - cxk_int
            src_gy = gy - cyk_int
            src_gz = gz - czk_int

            # Only sync directions whose dst has children AND src does NOT.
            # The src-has-children case was handled by the first loop above.
            dst_has_children = coarse_cell_has_fine_children(
                child_block_pointer,
                child_dim_x, child_dim_y, child_dim_z,
                bs, dst_gx, dst_gy, dst_gz
            )
            dst_has_children || continue
            if coarse_cell_has_fine_children(
                child_block_pointer,
                child_dim_x, child_dim_y, child_dim_z,
                bs, src_gx, src_gy, src_gz
            )
                continue
            end

            mother_fx0 = Int32(2) * dst_gx - Int32(1)
            mother_fy0 = Int32(2) * dst_gy - Int32(1)
            mother_fz0 = Int32(2) * dst_gz - Int32(1)
            sum_rho = 0.0f0
            sum_jx = 0.0f0
            sum_jy = 0.0f0
            sum_jz = 0.0f0
            sum_temp = 0.0f0
            sum_f_k = 0.0f0
            sum_f_kb = 0.0f0
            sum_g_k = 0.0f0
            sum_pxx = 0.0f0
            sum_pyy = 0.0f0
            sum_pzz = 0.0f0
            sum_pxy = 0.0f0
            sum_pxz = 0.0f0
                sum_pyz = 0.0f0
                valid_children = Int32(0)
                kopp = opp_arr[k]

            # Destination-side donor: current shell cell is on the opposite
            # side of the fine parent from +c, so the facing fine subset uses
            # the opposite child offset from the source-side case above.
            sx_first = cxk_int == Int32(0) ? Int32(0) : (cxk_int > Int32(0) ? Int32(0) : Int32(1))
            sx_last = cxk_int == Int32(0) ? Int32(1) : sx_first
            sy_first = cyk_int == Int32(0) ? Int32(0) : (cyk_int > Int32(0) ? Int32(0) : Int32(1))
            sy_last = cyk_int == Int32(0) ? Int32(1) : sy_first
            sz_first = czk_int == Int32(0) ? Int32(0) : (czk_int > Int32(0) ? Int32(0) : Int32(1))
            sz_last = czk_int == Int32(0) ? Int32(1) : sz_first

            @inbounds for sx in sx_first:sx_last, sy in sy_first:sy_last, sz in sz_first:sz_last
                clx, cly, clz, cb, valid_child = lookup_child_cell(
                    child_block_pointer, child_dim_x, child_dim_y, child_dim_z,
                    bs,
                    mother_fx0 + sx,
                    mother_fy0 + sy,
                    mother_fz0 + sz
                    )
                    if valid_child && !child_obstacle[clx, cly, clz, cb]
                        rho_child = child_rho[clx, cly, clz, cb]
                        ux_child = child_vel[clx, cly, clz, cb, 1]
                        uy_child = child_vel[clx, cly, clz, cb, 2]
                    uz_child = child_vel[clx, cly, clz, cb, 3]
                    sum_rho += rho_child
                    sum_jx += rho_child * ux_child
                    sum_jy += rho_child * uy_child
                    sum_jz += rho_child * uz_child
                    if k <= n_dirs_flow
                        sum_f_k += child_f_current[clx, cly, clz, cb, k]
                        if mirror_decompose_entropic == Int32(1)
                            sum_f_kb += child_f_current[clx, cly, clz, cb, kopp]
                        end
                    end
                    if restrict_thermal == Int32(1)
                        sum_temp += child_temperature[clx, cly, clz, cb]
                        if k <= n_dirs_thermal
                            sum_g_k += child_g_current[clx, cly, clz, cb, k]
                        end
                    end
                    if preserve_pi_neq == Int32(1)
                        sum_pxx += child_pi_neq[clx, cly, clz, cb, 1]
                        sum_pyy += child_pi_neq[clx, cly, clz, cb, 2]
                        sum_pzz += child_pi_neq[clx, cly, clz, cb, 3]
                        sum_pxy += child_pi_neq[clx, cly, clz, cb, 4]
                        sum_pxz += child_pi_neq[clx, cly, clz, cb, 5]
                        sum_pyz += child_pi_neq[clx, cly, clz, cb, 6]
                    end
                    valid_children += Int32(1)
                end
            end

            valid_children > Int32(0) || continue
            inv_children = 1.0f0 / Float32(valid_children)

            if k <= n_dirs_flow
                rho_donor = max(sum_rho * inv_children, 0.01f0)
                inv_rho_donor = 1.0f0 / rho_donor
                ux_donor = sum_jx * inv_children * inv_rho_donor
                uy_donor = sum_jy * inv_children * inv_rho_donor
                uz_donor = sum_jz * inv_children * inv_rho_donor
                temp_donor = restrict_thermal == Int32(1) ? max(sum_temp * inv_children, 0.01f0) : 1.0f0
                temp_clamped = clamp(temp_donor, 0.1f0, 10.0f0)

                cxk = Float32(cx_arr[k]) - shift_x
                cyk = Float32(cy_arr[k]) - shift_y
                czk = Float32(cz_arr[k]) - shift_z
                wk = Float32(w_arr[k])
                pop_blend_k = cxk_int == Int32(0) ? 1.0f0 : population_blend_x

                feq_k = use_numerical_eq ?
                    shell_mass_corr_eq * eval_13mom_feq_single(
                        rho_shell_ref, shell_lambda_eq, cxk, cyk, czk, wk, cs2_flow
                    ) :
                    calculate_equilibrium_auto(
                        rho_shell_ref, ux_shell_ref, uy_shell_ref, uz_shell_ref,
                        wk, cxk, cyk, czk,
                        flow_is_compressible
                    )

                if shell_equilibrium_only
                    parent_f_current[lx, ly, lz, pb_idx, k] =
                        blended_shell_population(parent_f_current[lx, ly, lz, pb_idx, k], feq_k, pop_blend_k)
                elseif use_pi_stress
                    lambda_eq = use_numerical_eq ?
                        solve_13mom_lambdas_d3q39(
                            rho_donor, ux_donor - shift_x, uy_donor - shift_y, uz_donor - shift_z,
                            cx_arr, cy_arr, cz_arr, w_arr,
                            cs2_flow, Int32(50), temp_clamped,
                            shift_x, shift_y, shift_z,
                            100.0f0, eos_R_flow          # SOTA A3 — match collision-kernel EOS R
                        ) :
                        (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
                         0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
                    mass_corr_eq = use_numerical_eq ?
                        compute_13mom_mass_correction_d3q39(
                            rho_donor, lambda_eq,
                            cx_arr, cy_arr, cz_arr, w_arr,
                            cs2_flow, shift_x, shift_y, shift_z
                        ) : 1.0f0
                    donor_feq_k = use_numerical_eq ?
                        mass_corr_eq * eval_13mom_feq_single(
                            rho_donor, lambda_eq, cxk, cyk, czk, wk, cs2_flow
                        ) :
                        calculate_equilibrium_auto(
                            rho_donor, ux_donor, uy_donor, uz_donor,
                            wk, cxk, cyk, czk,
                            flow_is_compressible
                        )
                    pxx_donor = sum_pxx * inv_children
                    pyy_donor = sum_pyy * inv_children
                    pzz_donor = sum_pzz * inv_children
                    pxy_donor = sum_pxy * inv_children
                    pxz_donor = sum_pxz * inv_children
                    pyz_donor = sum_pyz * inv_children
                    stress_k = project_second_order_stress_mode(
                        wk, cxk, cyk, czk, cs2_flow,
                        pxx_donor, pyy_donor, pzz_donor,
                        pxy_donor, pxz_donor, pyz_donor
                    )
                    parent_f_current[lx, ly, lz, pb_idx, k] =
                        blended_shell_population(
                            parent_f_current[lx, ly, lz, pb_idx, k],
                            donor_feq_k + tau_ratio_flow * stress_k,
                            pop_blend_k
                        )
                elseif mirror_decompose_entropic == Int32(1)
                    cxk_b = Float32(cx_arr[kopp]) - shift_x
                    cyk_b = Float32(cy_arr[kopp]) - shift_y
                    czk_b = Float32(cz_arr[kopp]) - shift_z
                    wk_b = Float32(w_arr[kopp])
                    lambda_eq = use_numerical_eq ?
                        solve_13mom_lambdas_d3q39(
                            rho_donor, ux_donor - shift_x, uy_donor - shift_y, uz_donor - shift_z,
                            cx_arr, cy_arr, cz_arr, w_arr,
                            cs2_flow, Int32(50), temp_clamped,
                            shift_x, shift_y, shift_z,
                            100.0f0, eos_R_flow          # SOTA A3 — match collision-kernel EOS R
                        ) :
                        (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
                         0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
                    mass_corr_eq = use_numerical_eq ?
                        compute_13mom_mass_correction_d3q39(
                            rho_donor, lambda_eq,
                            cx_arr, cy_arr, cz_arr, w_arr,
                            cs2_flow, shift_x, shift_y, shift_z
                        ) : 1.0f0
                    donor_feq_k = use_numerical_eq ?
                        mass_corr_eq * eval_13mom_feq_single(
                            rho_donor, lambda_eq, cxk, cyk, czk, wk, cs2_flow
                        ) :
                        calculate_equilibrium_auto(
                            rho_donor, ux_donor, uy_donor, uz_donor,
                            wk, cxk, cyk, czk,
                            flow_is_compressible
                        )
                    feq_kb = use_numerical_eq ?
                        mass_corr_eq * eval_13mom_feq_single(
                            rho_donor, lambda_eq, cxk_b, cyk_b, czk_b, wk_b, cs2_flow
                        ) :
                        calculate_equilibrium_auto(
                            rho_donor, ux_donor, uy_donor, uz_donor,
                            wk_b, cxk_b, cyk_b, czk_b,
                            flow_is_compressible
                        )
                    favg_k = sum_f_k * inv_children
                    favg_kb = sum_f_kb * inv_children
                    s_k = 0.5f0 * ((favg_k - donor_feq_k) + (favg_kb - feq_kb))
                    h_k = 0.5f0 * ((favg_k - donor_feq_k) - (favg_kb - feq_kb))
                    parent_f_current[lx, ly, lz, pb_idx, k] =
                        blended_shell_population(
                            parent_f_current[lx, ly, lz, pb_idx, k],
                            donor_feq_k + tau_ratio_flow * s_k + h_scale_shell * tau_ratio_flow * h_k,
                            pop_blend_k
                        )
                else
                    lambda_eq = use_numerical_eq ?
                        solve_13mom_lambdas_d3q39(
                            rho_donor, ux_donor - shift_x, uy_donor - shift_y, uz_donor - shift_z,
                            cx_arr, cy_arr, cz_arr, w_arr,
                            cs2_flow, Int32(50), temp_clamped,
                            shift_x, shift_y, shift_z,
                            100.0f0, eos_R_flow          # SOTA A3 — match collision-kernel EOS R
                        ) :
                        (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
                         0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
                    mass_corr_eq = use_numerical_eq ?
                        compute_13mom_mass_correction_d3q39(
                            rho_donor, lambda_eq,
                            cx_arr, cy_arr, cz_arr, w_arr,
                            cs2_flow, shift_x, shift_y, shift_z
                        ) : 1.0f0
                    donor_feq_k = use_numerical_eq ?
                        mass_corr_eq * eval_13mom_feq_single(
                            rho_donor, lambda_eq, cxk, cyk, czk, wk, cs2_flow
                        ) :
                        calculate_equilibrium_auto(
                            rho_donor, ux_donor, uy_donor, uz_donor,
                            wk, cxk, cyk, czk,
                            flow_is_compressible
                        )
                    favg_k = sum_f_k * inv_children
                    parent_f_current[lx, ly, lz, pb_idx, k] =
                        blended_shell_population(
                            parent_f_current[lx, ly, lz, pb_idx, k],
                            donor_feq_k + tau_ratio_flow * (favg_k - donor_feq_k),
                            pop_blend_k
                        )
                end
                have_updates = true
            end

            if restrict_thermal == Int32(1) && k <= n_dirs_thermal
                rho_donor = max(sum_rho * inv_children, 0.01f0)
                inv_rho_donor = 1.0f0 / rho_donor
                ux_donor = sum_jx * inv_children * inv_rho_donor
                uy_donor = sum_jy * inv_children * inv_rho_donor
                uz_donor = sum_jz * inv_children * inv_rho_donor
                temp_donor = max(sum_temp * inv_children, 0.01f0)
                temp_clamped = clamp(temp_donor, 0.1f0, 10.0f0)

                cxk = Float32(cx_arr[k]) - shift_x
                cyk = Float32(cy_arr[k]) - shift_y
                czk = Float32(cz_arr[k]) - shift_z
                wk = Float32(w_arr[k])

                lambda_eq = use_numerical_eq ?
                    solve_13mom_lambdas_d3q39(
                        rho_donor, ux_donor - shift_x, uy_donor - shift_y, uz_donor - shift_z,
                        cx_arr, cy_arr, cz_arr, w_arr,
                        cs2_flow, Int32(50), temp_clamped,
                        shift_x, shift_y, shift_z,
                        100.0f0, eos_R_flow          # SOTA A3 — match collision-kernel EOS R
                    ) :
                    (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
                     0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
                mass_corr_eq = use_numerical_eq ?
                    compute_13mom_mass_correction_d3q39(
                        rho_donor, lambda_eq,
                        cx_arr, cy_arr, cz_arr, w_arr,
                        cs2_flow, shift_x, shift_y, shift_z
                    ) : 1.0f0

                geq_shell_k = if use_numerical_eq && frapolli_mode == Int32(1)
                    # EOS-CONSISTENT g VARIANCE (2026-06-25): match collision-kernel
                    # coeff_g = Cv − 1.5·eos_R_flow; sentinel ≤0 ⇒ cs²_thermal (legacy).
                    R_g = eos_R_flow > 0.0f0 ? eos_R_flow : cs2_thermal
                    coeff_g = cv_thermal - 1.5f0 * R_g
                    feq_flow_k = shell_mass_corr_eq * eval_13mom_feq_single(
                        rho_shell_ref, shell_lambda_eq, cxk, cyk, czk, wk, cs2_flow
                    )
                    coeff_g * temp_shell_ref * feq_flow_k
                else
                    inv_cs2_t = 1.0f0 / cs2_thermal
                    wk * temp_shell_ref * (1.0f0 + (cxk * ux_shell_ref + cyk * uy_shell_ref + czk * uz_shell_ref) * inv_cs2_t)
                end
                geq_k = if use_numerical_eq && frapolli_mode == Int32(1)
                    R_g = eos_R_flow > 0.0f0 ? eos_R_flow : cs2_thermal
                    coeff_g = cv_thermal - 1.5f0 * R_g
                    feq_flow_k = mass_corr_eq * eval_13mom_feq_single(
                        rho_donor, lambda_eq, cxk, cyk, czk, wk, cs2_flow
                    )
                    coeff_g * temp_clamped * feq_flow_k
                else
                    inv_cs2_t = 1.0f0 / cs2_thermal
                    wk * temp_clamped * (1.0f0 + (cxk * ux_donor + cyk * uy_donor + czk * uz_donor) * inv_cs2_t)
                end
                gavg_k = sum_g_k * inv_children
                parent_g_current[lx, ly, lz, pb_idx, k] =
                    max(geq_k + tau_ratio_thermal * (gavg_k - geq_k), 1.0f-10)
            end
        end

        # The population loops above intentionally skip every x-bearing D3Q39
        # link: earlier A/B runs showed that importing those oblique populations
        # worsens vorticity.  The macro closure below is different: it only
        # moves the parent shell toward the 2x2 fine face average and then
        # applies an equilibrium shift.  Therefore it must also run on pure
        # x-facing shell cells, where have_updates remains false by design.
        has_fine_face_neighbor = false
        @inbounds for face_id in Int32(0):Int32(5)
            dx = face_id == Int32(0) ? Int32(1) :
                 face_id == Int32(1) ? Int32(-1) : Int32(0)
            dy = face_id == Int32(2) ? Int32(1) :
                 face_id == Int32(3) ? Int32(-1) : Int32(0)
            dz = face_id == Int32(4) ? Int32(1) :
                 face_id == Int32(5) ? Int32(-1) : Int32(0)

            has_fine_face_neighbor =
                has_fine_face_neighbor ||
                coarse_cell_has_fine_children(
                    child_block_pointer,
                    child_dim_x, child_dim_y, child_dim_z,
                    bs, gx + dx, gy + dy, gz + dz
                )
        end

        if have_updates || has_fine_face_neighbor

            rho_raw = 0.0f0
            jx = 0.0f0
            jy = 0.0f0
            jz = 0.0f0
            g_sum = 0.0f0
            ekin_shell = 0.0f0

            @inbounds for k in 1:39
                if k <= n_dirs_flow
                    fval = parent_f_current[lx, ly, lz, pb_idx, k]
                    rho_raw += fval
                    jx += fval * Float32(cx_arr[k])
                    jy += fval * Float32(cy_arr[k])
                    jz += fval * Float32(cz_arr[k])
                    cxu = Float32(cx_arr[k]) - shift_x
                    cyu = Float32(cy_arr[k]) - shift_y
                    czu = Float32(cz_arr[k]) - shift_z
                    ekin_shell += fval * (cxu * cxu + cyu * cyu + czu * czu)
                end

                if restrict_thermal == Int32(1) && k <= n_dirs_thermal
                    g_sum += parent_g_current[lx, ly, lz, pb_idx, k]
                end
            end

            rho_shell = max(rho_raw, 0.01f0)
            inv_rho_shell = 1.0f0 / rho_shell
            ux_shell = jx * inv_rho_shell
            uy_shell = jy * inv_rho_shell
            uz_shell = jz * inv_rho_shell

            parent_rho[lx, ly, lz, pb_idx] = rho_shell
            parent_vel[lx, ly, lz, pb_idx, 1] = ux_shell
            parent_vel[lx, ly, lz, pb_idx, 2] = uy_shell
            parent_vel[lx, ly, lz, pb_idx, 3] = uz_shell

            if restrict_thermal == Int32(1)
                ux_lab = ux_shell - shift_x
                uy_lab = uy_shell - shift_y
                uz_lab = uz_shell - shift_z
                u_sq_lab = ux_lab * ux_lab + uy_lab * uy_lab + uz_lab * uz_lab
                temp_shell = if frapolli_mode == Int32(1)
                    cv_safe = max(cv_thermal, 1.0f-6)
                    (ekin_shell - rho_shell * u_sq_lab + 2.0f0 * g_sum) /
                    (2.0f0 * rho_shell * cv_safe)
                else
                    g_sum
                end

                parent_temperature[lx, ly, lz, pb_idx] =
                    clamp(max(temp_shell, 0.01f0), 0.1f0, 10.0f0)
                parent_kinetic_energy[lx, ly, lz, pb_idx] = ekin_shell
            end

            # â”€â”€ Direct fine-side macro force-sync â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # The macro refresh above computes Ï, u, T from Î£f_k over the
            # shell cell's 39 populations. Most of those populations are
            # now synced (both incoming-from-box and outgoing-to-box passes
            # above), but a few non-interface populations still reflect the
            # shell cell's own coarse evolution. Those residual pops drag
            # the refreshed macros back toward the coarse-evolved state,
            # leaving the ~1e-4 Ï gap that convection stretches into a
            # 1-cell dark column at +x faces.
            #
            # Fix: for each face-adjacent coarse neighbor that contains
            # fine children, average the 4 children at the shell-facing
            # face of that neighbor (the fine cells that directly touch
            # the AMR interface from the inside). Average these over all
            # face directions that see fine children, and OVERWRITE the
            # shell cell's macros. This makes the shell macros identical
            # to the fine side's view at the interface â€” prolongation
            # from this cell now produces a halo consistent with the
            # fine interior, eliminating the seam.
            #
            # Populations are left as computed by the earlier reconstruction
            # passes, so mass flux through the interface is still coupled
            # through the full distribution. Only the macro fields used by
            # prolongation are force-synced.
            fine_rho_sum = 0.0f0
            fine_jx_sum = 0.0f0
            fine_jy_sum = 0.0f0
            fine_jz_sum = 0.0f0
            fine_T_sum = 0.0f0
            fine_face_weight_sum = 0.0f0
            fine_face_count = Int32(0)
            fine_x_face_count = Int32(0)
            fine_y_face_count = Int32(0)
            fine_z_face_count = Int32(0)
            fine_jx_x_sum = 0.0f0
            fine_jy_y_sum = 0.0f0
            fine_jz_z_sum = 0.0f0
            fine_x_face_weight_sum = 0.0f0
            fine_y_face_weight_sum = 0.0f0
            fine_z_face_weight_sum = 0.0f0
            fine_jx_x_pos_sum = 0.0f0
            fine_jx_x_neg_sum = 0.0f0
            fine_rho_x_pos_sum = 0.0f0
            fine_rho_x_neg_sum = 0.0f0
            fine_jx_x_pos_old_sum = 0.0f0
            fine_jx_x_neg_old_sum = 0.0f0
            fine_rho_x_pos_old_sum = 0.0f0
            fine_rho_x_neg_old_sum = 0.0f0
            fine_x_pos_face_weight_sum = 0.0f0
            fine_x_neg_face_weight_sum = 0.0f0
            fine_pxx_sum = 0.0f0
            fine_pyy_sum = 0.0f0
            fine_pzz_sum = 0.0f0
            fine_pxy_sum = 0.0f0
            fine_pxz_sum = 0.0f0
            fine_pyz_sum = 0.0f0
            single_face_axis_candidate = Int32(0)
            single_face_sign_candidate = Int32(0)

            # Iterate over the 6 face neighbours. We encode the face in a loop
            # index `face_id âˆˆ 0..5` and compute (dx, dy, dz, axis, face_fx0,
            # face_fy0, face_fz0) inline. Everything accumulates directly into
            # the outer-scope fine_*_sum variables â€” no nested function with a
            # `return` statement, which is forbidden inside a KernelAbstractions
            # `@kernel function` body.
            @inbounds for face_id in Int32(0):Int32(5)
                dx = face_id == Int32(0) ? Int32(1) :
                     face_id == Int32(1) ? Int32(-1) : Int32(0)
                dy = face_id == Int32(2) ? Int32(1) :
                     face_id == Int32(3) ? Int32(-1) : Int32(0)
                dz = face_id == Int32(4) ? Int32(1) :
                     face_id == Int32(5) ? Int32(-1) : Int32(0)
                axis = face_id < Int32(2) ? Int32(1) :
                       face_id < Int32(4) ? Int32(2) : Int32(3)

                ngx = gx + dx
                ngy = gy + dy
                ngz = gz + dz

                if coarse_cell_has_fine_children(
                    child_block_pointer,
                    child_dim_x, child_dim_y, child_dim_z,
                    bs, ngx, ngy, ngz
                )
                    # face-origin fine coords on the shell-facing face of ngx,gy,gz.
                    # For a +x neighbour (dx=1), the shell-facing face is the
                    # neighbour's -x face at fine x = 2Â·ngx - 1 = 2gx + 1.
                    # For a -x neighbour (dx=-1), it's the neighbour's +x face
                    # at fine x = 2Â·ngx = 2gx - 2. Analogous for y/z.
                    face_fx0 = axis == Int32(1) ?
                        (dx == Int32(1) ? Int32(2) * gx + Int32(1) : Int32(2) * gx - Int32(2)) :
                        Int32(2) * gx - Int32(1)
                    face_fy0 = axis == Int32(2) ?
                        (dy == Int32(1) ? Int32(2) * gy + Int32(1) : Int32(2) * gy - Int32(2)) :
                        Int32(2) * gy - Int32(1)
                    face_fz0 = axis == Int32(3) ?
                        (dz == Int32(1) ? Int32(2) * gz + Int32(1) : Int32(2) * gz - Int32(2)) :
                        Int32(2) * gz - Int32(1)

                    sum_r = 0.0f0
                    sum_jx_ = 0.0f0
                    sum_jy_ = 0.0f0
                    sum_jz_ = 0.0f0
                    sum_r_old_ = 0.0f0
                    sum_jx_old_ = 0.0f0
                    sum_T_ = 0.0f0
                    sum_pxx_ = 0.0f0
                    sum_pyy_ = 0.0f0
                    sum_pzz_ = 0.0f0
                    sum_pxy_ = 0.0f0
                    sum_pxz_ = 0.0f0
                    sum_pyz_ = 0.0f0
                    n_valid = Int32(0)

                    # Iterate the 2Ã—2 children on the shell-facing face.
                    # axis=1 â†’ fx fixed, iterate fy, fz
                    # axis=2 â†’ fy fixed, iterate fx, fz
                    # axis=3 â†’ fz fixed, iterate fx, fy
                    for sa in Int32(0):Int32(1), sb in Int32(0):Int32(1)
                        fx = axis == Int32(1) ? face_fx0 :
                             axis == Int32(2) ? face_fx0 + sa : face_fx0 + sa
                        fy = axis == Int32(1) ? face_fy0 + sa :
                             axis == Int32(2) ? face_fy0 : face_fy0 + sb
                        fz = axis == Int32(1) ? face_fz0 + sb :
                             axis == Int32(2) ? face_fz0 + sb : face_fz0
                        clx, cly, clz, cb, valid_child = lookup_child_cell(
                            child_block_pointer, child_dim_x, child_dim_y, child_dim_z,
                            bs, fx, fy, fz
                        )
                        if valid_child && !child_obstacle[clx, cly, clz, cb]
                            rc = child_rho[clx, cly, clz, cb]
                            ucx = child_vel[clx, cly, clz, cb, 1]
                            ucy = child_vel[clx, cly, clz, cb, 2]
                            ucz = child_vel[clx, cly, clz, cb, 3]
                            sum_r += rc
                            sum_jx_ += rc * ucx
                            sum_jy_ += rc * ucy
                            sum_jz_ += rc * ucz
                            if coupled_reflux == Int32(1) && face_time_snapshots == Int32(1)
                                rc_old = child_rho_old[clx, cly, clz, cb]
                                ucx_old = child_vel_old[clx, cly, clz, cb, 1]
                                sum_r_old_ += rc_old
                                sum_jx_old_ += rc_old * ucx_old
                            end
                            if restrict_thermal == Int32(1)
                                sum_T_ += child_temperature[clx, cly, clz, cb]
                            end
                            if preserve_pi_neq == Int32(1)
                                sum_pxx_ += child_pi_neq[clx, cly, clz, cb, 1]
                                sum_pyy_ += child_pi_neq[clx, cly, clz, cb, 2]
                                sum_pzz_ += child_pi_neq[clx, cly, clz, cb, 3]
                                sum_pxy_ += child_pi_neq[clx, cly, clz, cb, 4]
                                sum_pxz_ += child_pi_neq[clx, cly, clz, cb, 5]
                                sum_pyz_ += child_pi_neq[clx, cly, clz, cb, 6]
                            end
                            n_valid += Int32(1)
                        end
                    end

                    if n_valid > Int32(0)
                        inv_nv = 1.0f0 / Float32(n_valid)
                        # Pass162: keep x-facing shells in this macro closure,
                        # but do not let them dominate the target. Pass161's
                        # stronger x contraction drove a streamwise pressure
                        # pump during ramp-up, visible as vertical velocity and
                        # vorticity striping around the wing.
                        face_weight = axis == Int32(1) ? face_weight_x : 1.0f0
                        fine_rho_sum += face_weight * sum_r * inv_nv
                        fine_jx_sum += face_weight * sum_jx_ * inv_nv
                        fine_jy_sum += face_weight * sum_jy_ * inv_nv
                        fine_jz_sum += face_weight * sum_jz_ * inv_nv
                        if restrict_thermal == Int32(1)
                            fine_T_sum += face_weight * sum_T_ * inv_nv
                        end
                        if preserve_pi_neq == Int32(1)
                            fine_pxx_sum += face_weight * sum_pxx_ * inv_nv
                            fine_pyy_sum += face_weight * sum_pyy_ * inv_nv
                            fine_pzz_sum += face_weight * sum_pzz_ * inv_nv
                            fine_pxy_sum += face_weight * sum_pxy_ * inv_nv
                            fine_pxz_sum += face_weight * sum_pxz_ * inv_nv
                            fine_pyz_sum += face_weight * sum_pyz_ * inv_nv
                        end
                        fine_face_weight_sum += face_weight
                        fine_face_count += Int32(1)
                        single_face_axis_candidate = axis
                        single_face_sign_candidate = axis == Int32(1) ? dx :
                                                     axis == Int32(2) ? dy : dz
                        if axis == Int32(1)
                            fine_x_face_count += Int32(1)
                            fine_jx_x_sum += face_weight * sum_jx_ * inv_nv
                            fine_x_face_weight_sum += face_weight
                            if dx > Int32(0)
                                fine_jx_x_pos_sum += face_weight * sum_jx_ * inv_nv
                                fine_rho_x_pos_sum += face_weight * sum_r * inv_nv
                                if coupled_reflux == Int32(1) && face_time_snapshots == Int32(1)
                                    fine_jx_x_pos_old_sum += face_weight * sum_jx_old_ * inv_nv
                                    fine_rho_x_pos_old_sum += face_weight * sum_r_old_ * inv_nv
                                end
                                fine_x_pos_face_weight_sum += face_weight
                            else
                                fine_jx_x_neg_sum += face_weight * sum_jx_ * inv_nv
                                fine_rho_x_neg_sum += face_weight * sum_r * inv_nv
                                if coupled_reflux == Int32(1) && face_time_snapshots == Int32(1)
                                    fine_jx_x_neg_old_sum += face_weight * sum_jx_old_ * inv_nv
                                    fine_rho_x_neg_old_sum += face_weight * sum_r_old_ * inv_nv
                                end
                                fine_x_neg_face_weight_sum += face_weight
                            end
                        elseif axis == Int32(2)
                            fine_y_face_count += Int32(1)
                            fine_jy_y_sum += face_weight * sum_jy_ * inv_nv
                            fine_y_face_weight_sum += face_weight
                        else
                            fine_z_face_count += Int32(1)
                            fine_jz_z_sum += face_weight * sum_jz_ * inv_nv
                            fine_z_face_weight_sum += face_weight
                        end
                    end
                end
            end

            # Apply force-sync as a bounded incremental state update, not a hard
            # overwrite. The previous version (hard overwrite) caused a
            # positive-feedback density blowup around step 1200 in pass55:
            #   â€¢ the overwrite replaced coarse Ï with the fine-side average
            #     every shell-sync call,
            #   â€¢ the new Ï drove the coarse cell's next collision and the
            #     prolongated halo,
            #   â€¢ the resulting fine state had a slightly different mean,
            #   â€¢ restriction â†’ next force-sync â†’ amplified mismatch â€¦
            # leading to mass drift growing 4% â†’ 50% in 100 steps and Ï_min
            # crashing to the 0.01 floor.
            #
            # Bounded blend protects against this:
            #   1. Compute the proposed change Î”Ï = Ï_fine_avg âˆ’ Ï_coarse_now
            #   2. If |Î”Ï/Ï_coarse_now| > MAX_REL: skip (anomalous â€” likely
            #      a wake/obstacle-adjacent shell cell where the fine-side
            #      average should not drive the coarse macro)
            #   3. Else apply Ï â† Ï + Î±Â·Î”Ï with Î± small enough that the
            #      prolongation feedback loop stays a contraction
            #
            # Keep this correction symmetric and bounded. A previous x-biased
            # relaxation (and direct T/Ekin updates below) drove the shell state
            # from the fine-side average without making the underlying thermal
            # populations conservative, which reintroduced the pass55/pass63
            # mass/thermal feedback near step ~1200.
            if fine_face_count > Int32(0) && fine_face_weight_sum > 0.0f0
                inv_faces = 1.0f0 / fine_face_weight_sum
                rho_fine_avg = max(fine_rho_sum * inv_faces, 0.01f0)
                ux_fine_avg = (fine_jx_sum * inv_faces) / rho_fine_avg
                uy_fine_avg = (fine_jy_sum * inv_faces) / rho_fine_avg
                uz_fine_avg = (fine_jz_sum * inv_faces) / rho_fine_avg
                rho_target = rho_fine_avg
                ux_target = ux_fine_avg
                uy_target = uy_fine_avg
                uz_target = uz_fine_avg
                temp_target = restrict_thermal == Int32(1) ?
                    clamp(max(fine_T_sum * inv_faces, 0.01f0), 0.1f0, 10.0f0) :
                    1.0f0
                pxx_fine_avg = preserve_pi_neq == Int32(1) ? fine_pxx_sum * inv_faces : 0.0f0
                pyy_fine_avg = preserve_pi_neq == Int32(1) ? fine_pyy_sum * inv_faces : 0.0f0
                pzz_fine_avg = preserve_pi_neq == Int32(1) ? fine_pzz_sum * inv_faces : 0.0f0
                pxy_fine_avg = preserve_pi_neq == Int32(1) ? fine_pxy_sum * inv_faces : 0.0f0
                pxz_fine_avg = preserve_pi_neq == Int32(1) ? fine_pxz_sum * inv_faces : 0.0f0
                pyz_fine_avg = preserve_pi_neq == Int32(1) ? fine_pyz_sum * inv_faces : 0.0f0

                single_face_shell = fine_face_count == Int32(1)
                has_x_face = fine_x_face_count > Int32(0)

                if well_balanced_rebuild == Int32(1)
                    # Rebuild the full parent shell distribution from the
                    # fine-side face average. This is the well-balanced AMR
                    # fixed-point path: a uniform moving D3Q39 equilibrium
                    # remains exactly uniform across coarse/fine interfaces.
                    rho_wb = rho_target
                    ux_wb = ux_target
                    uy_wb = uy_target
                    uz_wb = uz_target
                    temp_wb = temp_target

                    lambda_wb = use_numerical_eq ?
                        solve_13mom_lambdas_d3q39(
                            rho_wb,
                            ux_wb - shift_x,
                            uy_wb - shift_y,
                            uz_wb - shift_z,
                            cx_arr, cy_arr, cz_arr, w_arr,
                            cs2_flow, Int32(50), temp_wb,
                            shift_x, shift_y, shift_z,
                            100.0f0, eos_R_flow          # SOTA A3 — match collision-kernel EOS R
                        ) :
                        (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
                         0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
                    mass_corr_wb = use_numerical_eq ?
                        compute_13mom_mass_correction_d3q39(
                            rho_wb, lambda_wb,
                            cx_arr, cy_arr, cz_arr, w_arr,
                            cs2_flow, shift_x, shift_y, shift_z
                        ) : 1.0f0

                    pxx_wb = preserve_pi_neq == Int32(1) ? pxx_fine_avg : 0.0f0
                    pyy_wb = preserve_pi_neq == Int32(1) ? pyy_fine_avg : 0.0f0
                    pzz_wb = preserve_pi_neq == Int32(1) ? pzz_fine_avg : 0.0f0
                    pxy_wb = preserve_pi_neq == Int32(1) ? pxy_fine_avg : 0.0f0
                    pxz_wb = preserve_pi_neq == Int32(1) ? pxz_fine_avg : 0.0f0
                    pyz_wb = preserve_pi_neq == Int32(1) ? pyz_fine_avg : 0.0f0

                    pos_limiter_wb = 1.0f0
                    if preserve_pi_neq == Int32(1)
                        @inbounds for kk in 1:39
                            if kk <= n_dirs_flow
                                cxk = Float32(cx_arr[kk]) - shift_x
                                cyk = Float32(cy_arr[kk]) - shift_y
                                czk = Float32(cz_arr[kk]) - shift_z
                                wk = Float32(w_arr[kk])
                                feq_wb_k = use_numerical_eq ?
                                    mass_corr_wb * eval_13mom_feq_single(
                                        rho_wb, lambda_wb, cxk, cyk, czk, wk, cs2_flow
                                    ) :
                                    calculate_equilibrium_auto(
                                        rho_wb, ux_wb, uy_wb, uz_wb,
                                        wk, cxk, cyk, czk,
                                        flow_is_compressible
                                    )
                                stress_wb_k = tau_ratio_flow *
                                    project_second_order_stress_mode(
                                        wk, cxk, cyk, czk, cs2_flow,
                                        pxx_wb, pyy_wb, pzz_wb,
                                        pxy_wb, pxz_wb, pyz_wb
                                    )
                                if stress_wb_k < 0.0f0
                                    pos_limiter_wb = min(
                                        pos_limiter_wb,
                                        max((feq_wb_k - 1.0f-10) / (-stress_wb_k), 0.0f0)
                                    )
                                end
                            end
                        end
                        pos_limiter_wb = min(pos_limiter_wb, 1.0f0)
                    end

                    rho_rec_wb = 0.0f0
                    jx_rec_wb = 0.0f0
                    jy_rec_wb = 0.0f0
                    jz_rec_wb = 0.0f0
                    ekin_rec_wb = 0.0f0
                    @inbounds for kk in 1:39
                        if kk <= n_dirs_flow
                            cxk = Float32(cx_arr[kk]) - shift_x
                            cyk = Float32(cy_arr[kk]) - shift_y
                            czk = Float32(cz_arr[kk]) - shift_z
                            wk = Float32(w_arr[kk])
                            feq_wb_k = use_numerical_eq ?
                                mass_corr_wb * eval_13mom_feq_single(
                                    rho_wb, lambda_wb, cxk, cyk, czk, wk, cs2_flow
                                ) :
                                calculate_equilibrium_auto(
                                    rho_wb, ux_wb, uy_wb, uz_wb,
                                    wk, cxk, cyk, czk,
                                    flow_is_compressible
                                )
                            stress_wb_k = preserve_pi_neq == Int32(1) ?
                                pos_limiter_wb * tau_ratio_flow *
                                project_second_order_stress_mode(
                                    wk, cxk, cyk, czk, cs2_flow,
                                    pxx_wb, pyy_wb, pzz_wb,
                                    pxy_wb, pxz_wb, pyz_wb
                                ) :
                                0.0f0
                            f_wb_k = max(feq_wb_k + stress_wb_k, 1.0f-10)
                            parent_f_current[lx, ly, lz, pb_idx, kk] = f_wb_k
                            rho_rec_wb += f_wb_k
                            jx_rec_wb += f_wb_k * Float32(cx_arr[kk])
                            jy_rec_wb += f_wb_k * Float32(cy_arr[kk])
                            jz_rec_wb += f_wb_k * Float32(cz_arr[kk])
                            ekin_rec_wb += f_wb_k * (cxk * cxk + cyk * cyk + czk * czk)

                            if restrict_thermal == Int32(1) && kk <= n_dirs_thermal
                                geq_wb_k = if use_numerical_eq && frapolli_mode == Int32(1)
                                    R_g = eos_R_flow > 0.0f0 ? eos_R_flow : cs2_thermal
                                    coeff_g = cv_thermal - 1.5f0 * R_g
                                    coeff_g * temp_wb * feq_wb_k
                                else
                                    inv_cs2_t = 1.0f0 / cs2_thermal
                                    wk * temp_wb * (1.0f0 + (cxk * ux_wb + cyk * uy_wb + czk * uz_wb) * inv_cs2_t)
                                end
                                parent_g_current[lx, ly, lz, pb_idx, kk] = max(geq_wb_k, 1.0f-10)
                            end
                        end
                    end

                    inv_cs2_corr_wb = 1.0f0 / max(cs2_flow, 1.0f-6)
                    jx_target_wb = rho_wb * ux_wb
                    jy_target_wb = rho_wb * uy_wb
                    jz_target_wb = rho_wb * uz_wb
                    for _corr_pass_wb in Int32(1):Int32(2)
                        deficit_rho_wb = rho_wb - rho_rec_wb
                        deficit_jx_wb = jx_target_wb - jx_rec_wb
                        deficit_jy_wb = jy_target_wb - jy_rec_wb
                        deficit_jz_wb = jz_target_wb - jz_rec_wb
                        if abs(deficit_rho_wb) > 1.0f-12 ||
                           abs(deficit_jx_wb) > 1.0f-12 ||
                           abs(deficit_jy_wb) > 1.0f-12 ||
                           abs(deficit_jz_wb) > 1.0f-12
                            rho_rec_wb = 0.0f0
                            jx_rec_wb = 0.0f0
                            jy_rec_wb = 0.0f0
                            jz_rec_wb = 0.0f0
                            ekin_rec_wb = 0.0f0
                            @inbounds for kk in 1:39
                                if kk <= n_dirs_flow
                                    cxk_raw = Float32(cx_arr[kk])
                                    cyk_raw = Float32(cy_arr[kk])
                                    czk_raw = Float32(cz_arr[kk])
                                    cxk = cxk_raw - shift_x
                                    cyk = cyk_raw - shift_y
                                    czk = czk_raw - shift_z
                                    delta_wb = Float32(w_arr[kk]) *
                                        (deficit_rho_wb +
                                         inv_cs2_corr_wb * (
                                             cxk_raw * deficit_jx_wb +
                                             cyk_raw * deficit_jy_wb +
                                             czk_raw * deficit_jz_wb
                                         ))
                                    f_wb_k = max(parent_f_current[lx, ly, lz, pb_idx, kk] + delta_wb, 1.0f-10)
                                    parent_f_current[lx, ly, lz, pb_idx, kk] = f_wb_k
                                    rho_rec_wb += f_wb_k
                                    jx_rec_wb += f_wb_k * cxk_raw
                                    jy_rec_wb += f_wb_k * cyk_raw
                                    jz_rec_wb += f_wb_k * czk_raw
                                    ekin_rec_wb += f_wb_k * (cxk * cxk + cyk * cyk + czk * czk)
                                end
                            end
                        end
                    end

                    rho_rec_wb = max(rho_rec_wb, 0.01f0)
                    parent_rho[lx, ly, lz, pb_idx] = rho_rec_wb
                    parent_vel[lx, ly, lz, pb_idx, 1] = jx_rec_wb / rho_rec_wb
                    parent_vel[lx, ly, lz, pb_idx, 2] = jy_rec_wb / rho_rec_wb
                    parent_vel[lx, ly, lz, pb_idx, 3] = jz_rec_wb / rho_rec_wb
                    if restrict_thermal == Int32(1)
                        parent_temperature[lx, ly, lz, pb_idx] = temp_wb
                        parent_kinetic_energy[lx, ly, lz, pb_idx] = ekin_rec_wb
                    end
                    if preserve_pi_neq == Int32(1)
                        parent_pi_neq[lx, ly, lz, pb_idx, 1] = pos_limiter_wb * pxx_wb
                        parent_pi_neq[lx, ly, lz, pb_idx, 2] = pos_limiter_wb * pyy_wb
                        parent_pi_neq[lx, ly, lz, pb_idx, 3] = pos_limiter_wb * pzz_wb
                        parent_pi_neq[lx, ly, lz, pb_idx, 4] = pos_limiter_wb * pxy_wb
                        parent_pi_neq[lx, ly, lz, pb_idx, 5] = pos_limiter_wb * pxz_wb
                        parent_pi_neq[lx, ly, lz, pb_idx, 6] = pos_limiter_wb * pyz_wb
                    end
                else
                rho_coarse_now = max(parent_rho[lx, ly, lz, pb_idx], 0.01f0)
                rel_drho = abs(rho_fine_avg - rho_coarse_now) / rho_coarse_now

                ux_now = parent_vel[lx, ly, lz, pb_idx, 1]
                uy_now = parent_vel[lx, ly, lz, pb_idx, 2]
                uz_now = parent_vel[lx, ly, lz, pb_idx, 3]

                # Use a capped contraction instead of an all-or-nothing gate.
                # The old 0.1% density gate protected against hard-overwrite
                # feedback, but it also left genuine interface jumps untouched
                # once they were large enough to matter in schlieren/vorticity.
                # Here the shell always moves toward the fine face, but by a
                # bounded amount that cannot avalanche into the pass55 mass drift.
                # Pass153: the remaining YZ rectangle is dominated by broad
                # single-face parent shells (not corners). Pull those faces
                # harder toward the fine macro state while leaving multi-face
                # shell cells on the safer pass150 contraction.
                single_face_shell = fine_face_count == Int32(1)
                has_x_face = fine_x_face_count > Int32(0)
                drho = rho_target - rho_coarse_now
                drho_cap = has_x_face ?
                    max(drho_cap_x_abs, drho_cap_x_rel * rho_coarse_now) :
                    max(drho_cap_t_abs, drho_cap_t_rel * rho_coarse_now)
                drho_limited = clamp(drho, -drho_cap, drho_cap)
                # X-facing streamwise interfaces are pressure-sensitive during
                # the inlet ramp. Treat them as nearly pressure-neutral: the
                # conservative population/restriction path carries density,
                # while this macro update mainly trims velocity discontinuity.
                rho_blend_alpha = single_face_shell ?
                    (has_x_face ? rho_alpha_x_single : rho_alpha_t_single) :
                    (has_x_face ? rho_alpha_x_multi : rho_alpha_t_multi)
                parent_rho[lx, ly, lz, pb_idx] = max(
                    rho_coarse_now + rho_blend_alpha * drho_limited,
                    0.01f0
                )

                # Velocity continuity is the quantity that signed vorticity
                # exposes, so do not couple it to the density gate above. Keep
                # the update bounded: the shell follows the fine-side face
                # average gradually, and large shock/body mismatches are clipped
                # rather than fed back as a hard overwrite.
                dux = ux_target - ux_now
                duy = uy_target - uy_now
                duz = uz_target - uz_now
                du_mag = sqrt(dux*dux + duy*duy + duz*duz)
                u_now_mag = sqrt(ux_now*ux_now + uy_now*uy_now + uz_now*uz_now)
                u_fine_mag = sqrt(ux_target*ux_target + uy_target*uy_target + uz_target*uz_target)
                u_ref = max(max(u_now_mag, u_fine_mag), 1.0f-3)
                du_cap = has_x_face ? max(du_cap_x_abs, du_cap_x_rel * u_ref) :
                                      max(du_cap_t_abs, du_cap_t_rel * u_ref)
                du_limiter = du_mag > du_cap ? du_cap / max(du_mag, 1.0f-12) : 1.0f0
                vel_blend_alpha = has_x_face ?
                    (single_face_shell ? vel_alpha_x_single : vel_alpha_x_multi) :
                    vel_alpha_t
                dvx_shell = vel_blend_alpha * du_limiter * dux
                dvy_shell = vel_blend_alpha * du_limiter * duy
                dvz_shell = vel_blend_alpha * du_limiter * duz
                parent_vel[lx, ly, lz, pb_idx, 1] = ux_now +
                    (vel_normal_only == Int32(0) || fine_x_face_count > Int32(0) ? dvx_shell : 0.0f0)
                parent_vel[lx, ly, lz, pb_idx, 2] = uy_now +
                    (vel_normal_only == Int32(0) || fine_y_face_count > Int32(0) ? dvy_shell : 0.0f0)
                parent_vel[lx, ly, lz, pb_idx, 3] = uz_now +
                    (vel_normal_only == Int32(0) || fine_z_face_count > Int32(0) ? dvz_shell : 0.0f0)

                # Optional mass-flux closure for the active parent shell.
                # The regular velocity update targets <u>_fine. In compressible
                # transfer, the continuity defect is <rho*u_n>_fine -
                # rho_shell*u_n_shell. After the bounded density update above,
                # use the shell density to form a velocity target that matches
                # the fine-side normal mass flux. Keep this as a small capped
                # correction; default alpha=0 keeps legacy behavior.
                flux_alpha = has_x_face ? flux_alpha_x : flux_alpha_t
                if flux_alpha > 0.0f0
                    rho_flux = max(parent_rho[lx, ly, lz, pb_idx], 0.01f0)
                    flux_cap = max(flux_du_cap_abs, flux_du_cap_rel * u_ref)
                    if fine_x_face_count > Int32(0) && fine_x_face_weight_sum > 0.0f0
                        x_sign_for_flux = single_face_shell && single_face_axis_candidate == Int32(1) ?
                            single_face_sign_candidate : Int32(0)
                        if x_sign_for_flux > Int32(0) && fine_x_pos_face_weight_sum > 0.0f0
                            ux_flux_target = (fine_jx_x_pos_sum / fine_x_pos_face_weight_sum) / rho_flux
                        elseif x_sign_for_flux < Int32(0) && fine_x_neg_face_weight_sum > 0.0f0
                            ux_flux_target = (fine_jx_x_neg_sum / fine_x_neg_face_weight_sum) / rho_flux
                        elseif fine_x_pos_face_weight_sum > 0.0f0 && fine_x_neg_face_weight_sum == 0.0f0
                            ux_flux_target = (fine_jx_x_pos_sum / fine_x_pos_face_weight_sum) / rho_flux
                        elseif fine_x_neg_face_weight_sum > 0.0f0 && fine_x_pos_face_weight_sum == 0.0f0
                            ux_flux_target = (fine_jx_x_neg_sum / fine_x_neg_face_weight_sum) / rho_flux
                        else
                            ux_flux_target = parent_vel[lx, ly, lz, pb_idx, 1]
                        end
                        du_flux_x = clamp(ux_flux_target - parent_vel[lx, ly, lz, pb_idx, 1],
                                          -flux_cap, flux_cap)
                        parent_vel[lx, ly, lz, pb_idx, 1] += flux_alpha_x * du_flux_x
                    end
                    if fine_y_face_count > Int32(0) && fine_y_face_weight_sum > 0.0f0
                        uy_flux_target = (fine_jy_y_sum / fine_y_face_weight_sum) / rho_flux
                        du_flux_y = clamp(uy_flux_target - parent_vel[lx, ly, lz, pb_idx, 2],
                                          -flux_cap, flux_cap)
                        parent_vel[lx, ly, lz, pb_idx, 2] += flux_alpha_t * du_flux_y
                    end
                    if fine_z_face_count > Int32(0) && fine_z_face_weight_sum > 0.0f0
                        uz_flux_target = (fine_jz_z_sum / fine_z_face_weight_sum) / rho_flux
                        du_flux_z = clamp(uz_flux_target - parent_vel[lx, ly, lz, pb_idx, 3],
                                          -flux_cap, flux_cap)
                        parent_vel[lx, ly, lz, pb_idx, 3] += flux_alpha_t * du_flux_z
                    end
                end

                # Optional normal-momentum-flux closure for x faces. The mass
                # flux correction above adjusts u_n; this one targets
                # rho*u_n^2 + cs^2*rho with a small bounded density update,
                # so pressure continuity can improve without reopening broad
                # density relaxation. Skip corners/ambiguous sign cases.
                if momflux_alpha_x > 0.0f0 &&
                   single_face_shell &&
                   single_face_axis_candidate == Int32(1) &&
                   single_face_sign_candidate != Int32(0)

                    x_sign_for_mom = single_face_sign_candidate
                    face_w_mom = x_sign_for_mom > Int32(0) ?
                        fine_x_pos_face_weight_sum : fine_x_neg_face_weight_sum
                    if face_w_mom > 0.0f0
                        rho_face = x_sign_for_mom > Int32(0) ?
                            max(fine_rho_x_pos_sum / face_w_mom, 0.01f0) :
                            max(fine_rho_x_neg_sum / face_w_mom, 0.01f0)
                        jx_face = x_sign_for_mom > Int32(0) ?
                            fine_jx_x_pos_sum / face_w_mom :
                            fine_jx_x_neg_sum / face_w_mom
                        ux_face = jx_face / rho_face
                        ux_shell_mom = parent_vel[lx, ly, lz, pb_idx, 1]
                        rho_shell_mom = max(parent_rho[lx, ly, lz, pb_idx], 0.01f0)
                        mom_flux_face = rho_face * ux_face * ux_face + cs2_flow * rho_face
                        rho_mom_target = mom_flux_face / max(ux_shell_mom * ux_shell_mom + cs2_flow, 1.0f-6)
                        mom_drho_cap = max(momflux_drho_cap_abs, momflux_drho_cap_rel * rho_shell_mom)
                        drho_mom = clamp(rho_mom_target - rho_shell_mom, -mom_drho_cap, mom_drho_cap)
                        parent_rho[lx, ly, lz, pb_idx] = max(
                            rho_shell_mom + momflux_alpha_x * drho_mom,
                            0.01f0
                        )
                    end
                end

                # Conservative x normal-momentum-flux repair. This targets the
                # same fine-side `rho*u_x^2 + cs^2*rho` residual as the density
                # closure above, but injects it as a bounded Pxx stress mode:
                # zero zeroth moment, zero momentum, nonzero second moment.
                # It is therefore the principled anti-drift form of the repair.
                if momflux_stress_alpha_x > 0.0f0 &&
                   single_face_shell &&
                   single_face_axis_candidate == Int32(1) &&
                   single_face_sign_candidate != Int32(0)

                    x_sign_for_mom_stress = single_face_sign_candidate
                    face_w_mom_stress = x_sign_for_mom_stress > Int32(0) ?
                        fine_x_pos_face_weight_sum : fine_x_neg_face_weight_sum
                    if face_w_mom_stress > 0.0f0
                        rho_face_stress = x_sign_for_mom_stress > Int32(0) ?
                            max(fine_rho_x_pos_sum / face_w_mom_stress, 0.01f0) :
                            max(fine_rho_x_neg_sum / face_w_mom_stress, 0.01f0)
                        jx_face_stress = x_sign_for_mom_stress > Int32(0) ?
                            fine_jx_x_pos_sum / face_w_mom_stress :
                            fine_jx_x_neg_sum / face_w_mom_stress
                        ux_face_stress = jx_face_stress / rho_face_stress
                        rho_shell_stress = max(parent_rho[lx, ly, lz, pb_idx], 0.01f0)
                        ux_shell_stress = parent_vel[lx, ly, lz, pb_idx, 1]
                        mom_face_stress = rho_face_stress * ux_face_stress * ux_face_stress +
                                          cs2_flow * rho_face_stress
                        mom_shell_stress = rho_shell_stress * ux_shell_stress * ux_shell_stress +
                                           cs2_flow * rho_shell_stress
                        mom_scale = max(max(abs(mom_face_stress), abs(mom_shell_stress)), cs2_flow * rho_shell_stress)
                        dp_cap = max(momflux_stress_cap_abs, momflux_stress_cap_rel * mom_scale)
                        dpxx_mom = momflux_stress_alpha_x *
                                   clamp(mom_face_stress - mom_shell_stress, -dp_cap, dp_cap)

                        pos_limiter_mom = 1.0f0
                        @inbounds for kk in 1:39
                            if kk <= n_dirs_flow
                                cxk = Float32(cx_arr[kk]) - shift_x
                                cyk = Float32(cy_arr[kk]) - shift_y
                                czk = Float32(cz_arr[kk]) - shift_z
                                wk = Float32(w_arr[kk])
                                df_mom = project_second_order_stress_mode(
                                    wk, cxk, cyk, czk, cs2_flow,
                                    dpxx_mom, 0.0f0, 0.0f0,
                                    0.0f0, 0.0f0, 0.0f0
                                )
                                if df_mom < 0.0f0
                                    f_now_mom = parent_f_current[lx, ly, lz, pb_idx, kk]
                                    pos_limiter_mom = min(
                                        pos_limiter_mom,
                                        max((f_now_mom - 1.0f-10) / (-df_mom), 0.0f0)
                                    )
                                end
                            end
                        end
                        dpxx_mom *= min(pos_limiter_mom, 1.0f0)
                        @inbounds for kk in 1:39
                            if kk <= n_dirs_flow
                                cxk = Float32(cx_arr[kk]) - shift_x
                                cyk = Float32(cy_arr[kk]) - shift_y
                                czk = Float32(cz_arr[kk]) - shift_z
                                wk = Float32(w_arr[kk])
                                df_mom = project_second_order_stress_mode(
                                    wk, cxk, cyk, czk, cs2_flow,
                                    dpxx_mom, 0.0f0, 0.0f0,
                                    0.0f0, 0.0f0, 0.0f0
                                )
                                parent_f_current[lx, ly, lz, pb_idx, kk] =
                                    max(parent_f_current[lx, ly, lz, pb_idx, kk] + df_mom, 1.0f-10)
                            end
                        end
                        if preserve_pi_neq == Int32(1)
                            parent_pi_neq[lx, ly, lz, pb_idx, 1] += dpxx_mom
                        end
                    end
                end

                if restrict_thermal == Int32(1)
                    temp_now = clamp(max(parent_temperature[lx, ly, lz, pb_idx], 0.01f0), 0.1f0, 10.0f0)
                    dT = temp_target - temp_now
                    dT_cap = max(1.0f-3, 0.02f0 * max(abs(temp_now), 1.0f0))
                    dT_limiter = abs(dT) > dT_cap ? dT_cap / max(abs(dT), 1.0f-12) : 1.0f0
                    temp_blend_alpha = has_x_face ? 0.02f0 : 0.08f0
                    parent_temperature[lx, ly, lz, pb_idx] =
                        clamp(temp_now + temp_blend_alpha * dT_limiter * dT, 0.1f0, 10.0f0)
                end
                end

                # â”€â”€ Equilibrium-shift to restore Î£f = Ï_new after macro update â”€â”€â”€â”€â”€â”€
                # The face update above wrote new (Ï, u, T) into parent_rho/vel/temperature
                # without touching the populations. After this point Î£f (computed from
                # the unmodified parent_f) equals (rho_shell, ux_shell, â€¦) â€” the macros
                # we captured BEFORE the update â€” not the new macros now stored in the
                # macro arrays. Prolongation reads those macros and the populations
                # independently and runs them through a 13-moment Newton solver, so any
                # disagreement injects a phantom non-equilibrium into the fine halo
                # that flow convection (Pe â‰ˆ 80 along x) stretches into the visible
                # 1-cell streamwise band at every AMR boundary face.
                #
                # Standard NEBB-style fix: f_new = f_old + (feq_new âˆ’ feq_old). This
                # preserves f_neq exactly (= f âˆ’ feq, untouched by the shift), so the
                # carefully reconstructed donor stress from passes 1-2 above is kept,
                # while restoring Î£f_new = Ï_new and Î£(f_newÂ·c) = ÏÂ·u_new.
                #
                # Only runs in the non-equilibrium-only branch; the shell_equilibrium_only
                # branch below already rebuilds populations as feq from the new macros,
                # so consistency is automatic there.
                if apply_eq_shift == Int32(1) && !shell_equilibrium_only && well_balanced_rebuild == Int32(0)
                    rho_new_shell = max(parent_rho[lx, ly, lz, pb_idx], 0.01f0)
                    ux_new_shell = parent_vel[lx, ly, lz, pb_idx, 1]
                    uy_new_shell = parent_vel[lx, ly, lz, pb_idx, 2]
                    uz_new_shell = parent_vel[lx, ly, lz, pb_idx, 3]
                    temp_new_shell = restrict_thermal == Int32(1) ?
                        clamp(max(parent_temperature[lx, ly, lz, pb_idx], 0.01f0), 0.1f0, 10.0f0) :
                        1.0f0
                    # Old temperature: the populations now stored in parent_g_current
                    # were last touched by passes 1-2 + macro refresh, so they are
                    # consistent with `temp_shell` (the recovered T from macro refresh
                    # at line 1673-1683 above), not with the kernel-entry temp_shell_ref.
                    # In the non-thermal path temp_shell is unset, so we fall back to 1.0f0.
                    temp_old_shell = restrict_thermal == Int32(1) ?
                        clamp(max(temp_shell, 0.01f0), 0.1f0, 10.0f0) :
                        1.0f0

                    drho_eq = rho_new_shell - rho_shell
                    dux_eq = ux_new_shell - ux_shell
                    duy_eq = uy_new_shell - uy_shell
                    duz_eq = uz_new_shell - uz_shell
                    dT_eq = temp_new_shell - temp_old_shell

                    # Skip the shift when the update is below numerical noise.
                    macro_changed = abs(drho_eq) > 1.0f-9 ||
                                    abs(dux_eq) > 1.0f-9 ||
                                    abs(duy_eq) > 1.0f-9 ||
                                    abs(duz_eq) > 1.0f-9 ||
                                    abs(dT_eq) > 1.0f-9

                    if macro_changed
                        used_face_pop_shift = false
                        if face_pop_shift == Int32(1) &&
                           single_face_shell &&
                           vel_normal_only == Int32(1) &&
                           single_face_axis_candidate > Int32(0) &&
                           single_face_sign_candidate != Int32(0)

                            axis_pop = single_face_axis_candidate
                            sign_pop = Float32(single_face_sign_candidate)
                            side_pop = Float32(face_pop_side)
                            drho_pop = face_pop_alpha * drho_eq
                            djn_pop = face_pop_alpha * (
                                axis_pop == Int32(1) ? (rho_new_shell * ux_new_shell - rho_shell * ux_shell) :
                                axis_pop == Int32(2) ? (rho_new_shell * uy_new_shell - rho_shell * uy_shell) :
                                                       (rho_new_shell * uz_new_shell - rho_shell * uz_shell)
                            )

                            m0 = 0.0f0
                            m1 = 0.0f0
                            m2 = 0.0f0
                            @inbounds for kk in 1:39
                                if kk <= n_dirs_flow
                                    cn = sign_pop * (
                                        axis_pop == Int32(1) ? (Float32(cx_arr[kk]) - shift_x) :
                                        axis_pop == Int32(2) ? (Float32(cy_arr[kk]) - shift_y) :
                                                               (Float32(cz_arr[kk]) - shift_z)
                                    )
                                    if side_pop * cn > 0.0f0
                                        wk = Float32(w_arr[kk])
                                        m0 += wk
                                        m1 += wk * cn
                                        m2 += wk * cn * cn
                                    end
                                end
                            end

                            det = m0 * m2 - m1 * m1
                            if abs(det) > 1.0f-12
                                a_pop = (drho_pop * m2 - djn_pop * m1) / det
                                b_pop = (djn_pop * m0 - drho_pop * m1) / det
                                @inbounds for kk in 1:39
                                    if kk <= n_dirs_flow
                                        cn = sign_pop * (
                                            axis_pop == Int32(1) ? (Float32(cx_arr[kk]) - shift_x) :
                                            axis_pop == Int32(2) ? (Float32(cy_arr[kk]) - shift_y) :
                                                                   (Float32(cz_arr[kk]) - shift_z)
                                        )
                                        if side_pop * cn > 0.0f0
                                            wk = Float32(w_arr[kk])
                                            f_now_k = parent_f_current[lx, ly, lz, pb_idx, kk]
                                            parent_f_current[lx, ly, lz, pb_idx, kk] =
                                                max(f_now_k + wk * (a_pop + b_pop * cn), 1.0f-10)
                                        end
                                    end
                                end
                                rho_after = 0.0f0
                                jx_after = 0.0f0
                                jy_after = 0.0f0
                                jz_after = 0.0f0
                                @inbounds for kk in 1:39
                                    if kk <= n_dirs_flow
                                        fval = parent_f_current[lx, ly, lz, pb_idx, kk]
                                        rho_after += fval
                                        jx_after += fval * Float32(cx_arr[kk])
                                        jy_after += fval * Float32(cy_arr[kk])
                                        jz_after += fval * Float32(cz_arr[kk])
                                    end
                                end
                                rho_after = max(rho_after, 0.01f0)
                                parent_rho[lx, ly, lz, pb_idx] = rho_after
                                parent_vel[lx, ly, lz, pb_idx, 1] = jx_after / rho_after
                                parent_vel[lx, ly, lz, pb_idx, 2] = jy_after / rho_after
                                parent_vel[lx, ly, lz, pb_idx, 3] = jz_after / rho_after
                                used_face_pop_shift = true
                            end
                        end

                        # Solve 13-moment lambdas at OLD macros (matches the equilibrium
                        # used implicitly by the populations as they stand right now).
                        old_lambda_eq = (!used_face_pop_shift && use_numerical_eq) ?
                            solve_13mom_lambdas_d3q39(
                                rho_shell,
                                ux_shell - shift_x,
                                uy_shell - shift_y,
                                uz_shell - shift_z,
                                cx_arr, cy_arr, cz_arr, w_arr,
                                cs2_flow, Int32(50), temp_old_shell,
                                shift_x, shift_y, shift_z,
                                100.0f0, eos_R_flow          # SOTA A3 — match collision-kernel EOS R
                            ) :
                            (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
                             0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
                        old_mass_corr = (!used_face_pop_shift && use_numerical_eq) ?
                            compute_13mom_mass_correction_d3q39(
                                rho_shell, old_lambda_eq,
                                cx_arr, cy_arr, cz_arr, w_arr,
                                cs2_flow, shift_x, shift_y, shift_z
                            ) : 1.0f0
                        # Solve 13-moment lambdas at NEW (post-update) macros.
                        new_lambda_eq = (!used_face_pop_shift && use_numerical_eq) ?
                            solve_13mom_lambdas_d3q39(
                                rho_new_shell,
                                ux_new_shell - shift_x,
                                uy_new_shell - shift_y,
                                uz_new_shell - shift_z,
                                cx_arr, cy_arr, cz_arr, w_arr,
                                cs2_flow, Int32(50), temp_new_shell,
                                shift_x, shift_y, shift_z,
                                100.0f0, eos_R_flow          # SOTA A3 — match collision-kernel EOS R
                            ) :
                            (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
                             0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
                        new_mass_corr = (!used_face_pop_shift && use_numerical_eq) ?
                            compute_13mom_mass_correction_d3q39(
                                rho_new_shell, new_lambda_eq,
                                cx_arr, cy_arr, cz_arr, w_arr,
                                cs2_flow, shift_x, shift_y, shift_z
                            ) : 1.0f0

                        if shell_recalibration == Int32(1) &&
                           use_pi_stress &&
                           single_face_shell &&
                           !used_face_pop_shift

                            parent_pxx = parent_pi_neq[lx, ly, lz, pb_idx, 1]
                            parent_pyy = parent_pi_neq[lx, ly, lz, pb_idx, 2]
                            parent_pzz = parent_pi_neq[lx, ly, lz, pb_idx, 3]
                            parent_pxy = parent_pi_neq[lx, ly, lz, pb_idx, 4]
                            parent_pxz = parent_pi_neq[lx, ly, lz, pb_idx, 5]
                            parent_pyz = parent_pi_neq[lx, ly, lz, pb_idx, 6]

                            dpxx = pxx_fine_avg - parent_pxx
                            dpyy = pyy_fine_avg - parent_pyy
                            dpzz = pzz_fine_avg - parent_pzz
                            dpxy = pxy_fine_avg - parent_pxy
                            dpxz = pxz_fine_avg - parent_pxz
                            dpyz = pyz_fine_avg - parent_pyz
                            parent_pi_norm = sqrt(
                                parent_pxx*parent_pxx + parent_pyy*parent_pyy + parent_pzz*parent_pzz +
                                2.0f0 * (parent_pxy*parent_pxy + parent_pxz*parent_pxz + parent_pyz*parent_pyz)
                            )
                            fine_pi_norm = sqrt(
                                pxx_fine_avg*pxx_fine_avg + pyy_fine_avg*pyy_fine_avg + pzz_fine_avg*pzz_fine_avg +
                                2.0f0 * (pxy_fine_avg*pxy_fine_avg + pxz_fine_avg*pxz_fine_avg + pyz_fine_avg*pyz_fine_avg)
                            )
                            dpi_norm = sqrt(
                                dpxx*dpxx + dpyy*dpyy + dpzz*dpzz +
                                2.0f0 * (dpxy*dpxy + dpxz*dpxz + dpyz*dpyz)
                            )
                            pi_scale = max(max(parent_pi_norm, fine_pi_norm), 1.0f-8)
                            dpi_cap = max(face_stress_abs_cap, face_stress_rel_cap * pi_scale)
                            stress_limiter = dpi_norm > dpi_cap ? 0.0f0 : 1.0f0

                            pxx_target = parent_pxx + face_stress_alpha * stress_limiter * dpxx
                            pyy_target = parent_pyy + face_stress_alpha * stress_limiter * dpyy
                            pzz_target = parent_pzz + face_stress_alpha * stress_limiter * dpzz
                            pxy_target = parent_pxy + face_stress_alpha * stress_limiter * dpxy
                            pxz_target = parent_pxz + face_stress_alpha * stress_limiter * dpxz
                            pyz_target = parent_pyz + face_stress_alpha * stress_limiter * dpyz

                            pos_limiter_recal = 1.0f0
                            @inbounds for kk in 1:39
                                if kk <= n_dirs_flow
                                    cxk = Float32(cx_arr[kk]) - shift_x
                                    cyk = Float32(cy_arr[kk]) - shift_y
                                    czk = Float32(cz_arr[kk]) - shift_z
                                    wk = Float32(w_arr[kk])
                                    feq_new_k = use_numerical_eq ?
                                        new_mass_corr * eval_13mom_feq_single(
                                            rho_new_shell, new_lambda_eq, cxk, cyk, czk, wk, cs2_flow
                                        ) :
                                        calculate_equilibrium_auto(
                                            rho_new_shell, ux_new_shell, uy_new_shell, uz_new_shell,
                                            wk, cxk, cyk, czk,
                                            flow_is_compressible
                                        )
                                    stress_k = project_second_order_stress_mode(
                                        wk, cxk, cyk, czk, cs2_flow,
                                        pxx_target, pyy_target, pzz_target,
                                        pxy_target, pxz_target, pyz_target
                                    )
                                    df_stress = tau_ratio_flow * stress_k
                                    if df_stress < 0.0f0
                                        pos_limiter_recal = min(
                                            pos_limiter_recal,
                                            max((feq_new_k - 1.0f-10) / (-df_stress), 0.0f0)
                                        )
                                    end
                                end
                            end
                            pos_limiter_recal = min(pos_limiter_recal, 1.0f0)

                            rho_recal = 0.0f0
                            jx_recal = 0.0f0
                            jy_recal = 0.0f0
                            jz_recal = 0.0f0
                            @inbounds for kk in 1:39
                                if kk <= n_dirs_flow
                                    cxk = Float32(cx_arr[kk]) - shift_x
                                    cyk = Float32(cy_arr[kk]) - shift_y
                                    czk = Float32(cz_arr[kk]) - shift_z
                                    wk = Float32(w_arr[kk])
                                    feq_new_k = use_numerical_eq ?
                                        new_mass_corr * eval_13mom_feq_single(
                                            rho_new_shell, new_lambda_eq, cxk, cyk, czk, wk, cs2_flow
                                        ) :
                                        calculate_equilibrium_auto(
                                            rho_new_shell, ux_new_shell, uy_new_shell, uz_new_shell,
                                            wk, cxk, cyk, czk,
                                            flow_is_compressible
                                        )
                                    stress_k = project_second_order_stress_mode(
                                        wk, cxk, cyk, czk, cs2_flow,
                                        pxx_target, pyy_target, pzz_target,
                                        pxy_target, pxz_target, pyz_target
                                    )
                                    f_recal = max(
                                        feq_new_k + pos_limiter_recal * tau_ratio_flow * stress_k,
                                        1.0f-10
                                    )
                                    parent_f_current[lx, ly, lz, pb_idx, kk] = f_recal
                                    rho_recal += f_recal
                                    jx_recal += f_recal * Float32(cx_arr[kk])
                                    jy_recal += f_recal * Float32(cy_arr[kk])
                                    jz_recal += f_recal * Float32(cz_arr[kk])
                                end
                            end

                            rho_recal = max(rho_recal, 0.01f0)
                            parent_rho[lx, ly, lz, pb_idx] = rho_recal
                            parent_vel[lx, ly, lz, pb_idx, 1] = jx_recal / rho_recal
                            parent_vel[lx, ly, lz, pb_idx, 2] = jy_recal / rho_recal
                            parent_vel[lx, ly, lz, pb_idx, 3] = jz_recal / rho_recal
                            parent_pi_neq[lx, ly, lz, pb_idx, 1] = pxx_target
                            parent_pi_neq[lx, ly, lz, pb_idx, 2] = pyy_target
                            parent_pi_neq[lx, ly, lz, pb_idx, 3] = pzz_target
                            parent_pi_neq[lx, ly, lz, pb_idx, 4] = pxy_target
                            parent_pi_neq[lx, ly, lz, pb_idx, 5] = pxz_target
                            parent_pi_neq[lx, ly, lz, pb_idx, 6] = pyz_target

                        elseif used_face_pop_shift
                            nothing
                        elseif use_moment_shift == Int32(1)
                            m00 = 0.0f0; m01 = 0.0f0; m02 = 0.0f0; m03 = 0.0f0
                            m11 = 0.0f0; m12 = 0.0f0; m13 = 0.0f0
                            m22 = 0.0f0; m23 = 0.0f0; m33 = 0.0f0
                            @inbounds for kk in 1:39
                                if kk <= n_dirs_flow
                                    cxk = Float32(cx_arr[kk]) - shift_x
                                    cyk = Float32(cy_arr[kk]) - shift_y
                                    czk = Float32(cz_arr[kk]) - shift_z
                                    wk = Float32(w_arr[kk])
                                    m00 += wk
                                    m01 += wk * cxk; m02 += wk * cyk; m03 += wk * czk
                                    m11 += wk * cxk * cxk; m12 += wk * cxk * cyk; m13 += wk * cxk * czk
                                    m22 += wk * cyk * cyk; m23 += wk * cyk * czk
                                    m33 += wk * czk * czk
                                end
                            end

                            djx_eq = rho_new_shell * ux_new_shell - rho_shell * ux_shell
                            djy_eq = rho_new_shell * uy_new_shell - rho_shell * uy_shell
                            djz_eq = rho_new_shell * uz_new_shell - rho_shell * uz_shell
                            c0, c1, c2, c3, moment_ok = solve_shell_moment4(
                                m00, m01, m02, m03,
                                m11, m12, m13,
                                m22, m23,
                                m33,
                                drho_eq, djx_eq, djy_eq, djz_eq
                            )

                            if moment_ok
                                pos_limiter_shift = 1.0f0
                                @inbounds for kk in 1:39
                                    if kk <= n_dirs_flow
                                        cxk = Float32(cx_arr[kk]) - shift_x
                                        cyk = Float32(cy_arr[kk]) - shift_y
                                        czk = Float32(cz_arr[kk]) - shift_z
                                        wk = Float32(w_arr[kk])
                                        df_k = wk * (c0 + c1 * cxk + c2 * cyk + c3 * czk)
                                        if df_k < 0.0f0
                                            f_now_k = parent_f_current[lx, ly, lz, pb_idx, kk]
                                            pos_limiter_shift = min(
                                                pos_limiter_shift,
                                                max((f_now_k - 1.0f-10) / (-df_k), 0.0f0)
                                            )
                                        end
                                    end
                                end
                                pos_limiter_shift = min(pos_limiter_shift, 1.0f0)
                                @inbounds for kk in 1:39
                                    if kk <= n_dirs_flow
                                        cxk = Float32(cx_arr[kk]) - shift_x
                                        cyk = Float32(cy_arr[kk]) - shift_y
                                        czk = Float32(cz_arr[kk]) - shift_z
                                        wk = Float32(w_arr[kk])
                                        f_now_k = parent_f_current[lx, ly, lz, pb_idx, kk]
                                        df_k = pos_limiter_shift *
                                               wk * (c0 + c1 * cxk + c2 * cyk + c3 * czk)
                                        parent_f_current[lx, ly, lz, pb_idx, kk] =
                                            max(f_now_k + df_k, 1.0f-10)
                                    end
                                end
                            end
                        else
                            @inbounds for kk in 1:39
                                if kk <= n_dirs_flow
                                cxk = Float32(cx_arr[kk]) - shift_x
                                cyk = Float32(cy_arr[kk]) - shift_y
                                czk = Float32(cz_arr[kk]) - shift_z
                                wk = Float32(w_arr[kk])

                                feq_old_k = use_numerical_eq ?
                                    old_mass_corr * eval_13mom_feq_single(
                                        rho_shell, old_lambda_eq, cxk, cyk, czk, wk, cs2_flow
                                    ) :
                                    calculate_equilibrium_auto(
                                        rho_shell, ux_shell, uy_shell, uz_shell,
                                        wk, cxk, cyk, czk,
                                        flow_is_compressible
                                    )
                                feq_new_k = use_numerical_eq ?
                                    new_mass_corr * eval_13mom_feq_single(
                                        rho_new_shell, new_lambda_eq, cxk, cyk, czk, wk, cs2_flow
                                    ) :
                                    calculate_equilibrium_auto(
                                        rho_new_shell, ux_new_shell, uy_new_shell, uz_new_shell,
                                        wk, cxk, cyk, czk,
                                        flow_is_compressible
                                    )
                                f_now_k = parent_f_current[lx, ly, lz, pb_idx, kk]
                                parent_f_current[lx, ly, lz, pb_idx, kk] =
                                    max(f_now_k + (feq_new_k - feq_old_k), 1.0f-10)

                                if restrict_thermal == Int32(1) && kk <= n_dirs_thermal
                                    geq_old_k = if use_numerical_eq && frapolli_mode == Int32(1)
                                        R_g = eos_R_flow > 0.0f0 ? eos_R_flow : cs2_thermal
                                        coeff_g = cv_thermal - 1.5f0 * R_g
                                        coeff_g * temp_old_shell * feq_old_k
                                    else
                                        inv_cs2_t = 1.0f0 / cs2_thermal
                                        wk * temp_old_shell *
                                        (1.0f0 + (cxk * ux_shell + cyk * uy_shell + czk * uz_shell) * inv_cs2_t)
                                    end
                                    geq_new_k = if use_numerical_eq && frapolli_mode == Int32(1)
                                        R_g = eos_R_flow > 0.0f0 ? eos_R_flow : cs2_thermal
                                        coeff_g = cv_thermal - 1.5f0 * R_g
                                        coeff_g * temp_new_shell * feq_new_k
                                    else
                                        inv_cs2_t = 1.0f0 / cs2_thermal
                                        wk * temp_new_shell *
                                        (1.0f0 + (cxk * ux_new_shell + cyk * uy_new_shell + czk * uz_new_shell) * inv_cs2_t)
                                    end
                                    g_now_k = parent_g_current[lx, ly, lz, pb_idx, kk]
                                    parent_g_current[lx, ly, lz, pb_idx, kk] =
                                        max(g_now_k + (geq_new_k - geq_old_k), 1.0f-10)
                                end
                                end
                            end
                        end

                        if shell_recalibration == Int32(0) &&
                           face_stress_shift == Int32(1) &&
                           preserve_pi_neq == Int32(1) &&
                           single_face_shell
                            parent_pxx = parent_pi_neq[lx, ly, lz, pb_idx, 1]
                            parent_pyy = parent_pi_neq[lx, ly, lz, pb_idx, 2]
                            parent_pzz = parent_pi_neq[lx, ly, lz, pb_idx, 3]
                            parent_pxy = parent_pi_neq[lx, ly, lz, pb_idx, 4]
                            parent_pxz = parent_pi_neq[lx, ly, lz, pb_idx, 5]
                            parent_pyz = parent_pi_neq[lx, ly, lz, pb_idx, 6]
                            dpxx = pxx_fine_avg - parent_pxx
                            dpyy = pyy_fine_avg - parent_pyy
                            dpzz = pzz_fine_avg - parent_pzz
                            dpxy = pxy_fine_avg - parent_pxy
                            dpxz = pxz_fine_avg - parent_pxz
                            dpyz = pyz_fine_avg - parent_pyz
                            parent_pi_norm = sqrt(
                                parent_pxx*parent_pxx + parent_pyy*parent_pyy + parent_pzz*parent_pzz +
                                2.0f0 * (parent_pxy*parent_pxy + parent_pxz*parent_pxz + parent_pyz*parent_pyz)
                            )
                            fine_pi_norm = sqrt(
                                pxx_fine_avg*pxx_fine_avg + pyy_fine_avg*pyy_fine_avg + pzz_fine_avg*pzz_fine_avg +
                                2.0f0 * (pxy_fine_avg*pxy_fine_avg + pxz_fine_avg*pxz_fine_avg + pyz_fine_avg*pyz_fine_avg)
                            )
                            dpi_norm = sqrt(
                                dpxx*dpxx + dpyy*dpyy + dpzz*dpzz +
                                2.0f0 * (dpxy*dpxy + dpxz*dpxz + dpyz*dpyz)
                            )
                            pi_scale = max(max(parent_pi_norm, fine_pi_norm), 1.0f-8)
                            dpi_cap = max(face_stress_abs_cap, face_stress_rel_cap * pi_scale)
                            # This projection is a smooth-interface repair, not
                            # a shock/wall stress overwrite. If the fine/coarse
                            # stress mismatch is not locally mild, leave the
                            # conservative restriction/prolongation path in
                            # charge; soft limiting proved too permissive in
                            # ONERA pass201.
                            stress_limiter = dpi_norm > dpi_cap ? 0.0f0 : 1.0f0
                            dpi_xx = face_stress_alpha * stress_limiter * dpxx
                            dpi_yy = face_stress_alpha * stress_limiter * dpyy
                            dpi_zz = face_stress_alpha * stress_limiter * dpzz
                            dpi_xy = face_stress_alpha * stress_limiter * dpxy
                            dpi_xz = face_stress_alpha * stress_limiter * dpxz
                            dpi_yz = face_stress_alpha * stress_limiter * dpyz
                            pos_limiter = 1.0f0
                            @inbounds for kk in 1:39
                                if kk <= n_dirs_flow
                                    cxk = Float32(cx_arr[kk]) - shift_x
                                    cyk = Float32(cy_arr[kk]) - shift_y
                                    czk = Float32(cz_arr[kk]) - shift_z
                                    wk = Float32(w_arr[kk])
                                    df_stress = project_second_order_stress_mode(
                                        wk, cxk, cyk, czk, cs2_flow,
                                        dpi_xx, dpi_yy, dpi_zz,
                                        dpi_xy, dpi_xz, dpi_yz
                                    )
                                    if df_stress < 0.0f0
                                        f_now_k = parent_f_current[lx, ly, lz, pb_idx, kk]
                                        pos_limiter = min(pos_limiter, max((f_now_k - 1.0f-10) / (-df_stress), 0.0f0))
                                    end
                                end
                            end
                            pos_limiter = min(pos_limiter, 1.0f0)
                            dpi_xx *= pos_limiter
                            dpi_yy *= pos_limiter
                            dpi_zz *= pos_limiter
                            dpi_xy *= pos_limiter
                            dpi_xz *= pos_limiter
                            dpi_yz *= pos_limiter
                            @inbounds for kk in 1:39
                                if kk <= n_dirs_flow
                                    cxk = Float32(cx_arr[kk]) - shift_x
                                    cyk = Float32(cy_arr[kk]) - shift_y
                                    czk = Float32(cz_arr[kk]) - shift_z
                                    wk = Float32(w_arr[kk])
                                    df_stress = project_second_order_stress_mode(
                                        wk, cxk, cyk, czk, cs2_flow,
                                        dpi_xx, dpi_yy, dpi_zz,
                                        dpi_xy, dpi_xz, dpi_yz
                                    )
                                    parent_f_current[lx, ly, lz, pb_idx, kk] =
                                        max(parent_f_current[lx, ly, lz, pb_idx, kk] + df_stress, 1.0f-10)
                                end
                            end
                            parent_pi_neq[lx, ly, lz, pb_idx, 1] += dpi_xx
                            parent_pi_neq[lx, ly, lz, pb_idx, 2] += dpi_yy
                            parent_pi_neq[lx, ly, lz, pb_idx, 3] += dpi_zz
                            parent_pi_neq[lx, ly, lz, pb_idx, 4] += dpi_xy
                            parent_pi_neq[lx, ly, lz, pb_idx, 5] += dpi_xz
                            parent_pi_neq[lx, ly, lz, pb_idx, 6] += dpi_yz
                        end

                        # Coupled face-time reflux prototype. The previous
                        # latest-state version improved mass bias but damaged
                        # normal momentum. Use a trapezoidal coarse-step budget
                        # from old/current parent and child face states, then
                        # apply one constrained half-range population projection.
                        # The paired variant applies the opposite macro-flux
                        # increment to the four fine cells on the same face,
                        # which is the conservative transfer form we need for
                        # the space-time flux gate.
                        if ((coupled_reflux == Int32(1) && coupled_reflux_alpha > 0.0f0) ||
                            (paired_reflux == Int32(1) && paired_reflux_alpha > 0.0f0) ||
                            (gpu_budget_reflux == Int32(1) && gpu_budget_reflux_alpha > 0.0f0)) &&
                           single_face_shell &&
                           single_face_axis_candidate == Int32(1) &&
                           single_face_sign_candidate != Int32(0)

                            x_sign_reflux = single_face_sign_candidate
                            use_gpu_budget_reflux = gpu_budget_reflux == Int32(1) && gpu_budget_reflux_alpha > 0.0f0
                            use_paired_reflux = paired_reflux == Int32(1) &&
                                                paired_reflux_alpha > 0.0f0 &&
                                                !use_gpu_budget_reflux
                            alpha_reflux = use_gpu_budget_reflux ? gpu_budget_reflux_alpha :
                                (use_paired_reflux ? paired_reflux_alpha : coupled_reflux_alpha)
                            paired_mom_alpha_eff_reflux = paired_reflux_mom_alpha
                            if use_paired_reflux &&
                               paired_reflux_mom_alpha_hold_steps > Int32(0) &&
                               current_top_step > Int32(0) &&
                               current_top_step <= paired_reflux_mom_alpha_hold_steps
                                paired_mom_alpha_eff_reflux =
                                    paired_reflux_mom_alpha_initial
                            end
                            face_w_reflux = x_sign_reflux > Int32(0) ?
                                fine_x_pos_face_weight_sum : fine_x_neg_face_weight_sum
                            if face_w_reflux > 0.0f0
                                rho_fine_reflux = x_sign_reflux > Int32(0) ?
                                    max(fine_rho_x_pos_sum / face_w_reflux, 0.01f0) :
                                    max(fine_rho_x_neg_sum / face_w_reflux, 0.01f0)
                                jx_fine_reflux = x_sign_reflux > Int32(0) ?
                                    fine_jx_x_pos_sum / face_w_reflux :
                                    fine_jx_x_neg_sum / face_w_reflux
                                ux_fine_reflux = jx_fine_reflux / rho_fine_reflux

                                rho_reflux = max(parent_rho[lx, ly, lz, pb_idx], 0.01f0)
                                ux_reflux = parent_vel[lx, ly, lz, pb_idx, 1]
                                jx_reflux = rho_reflux * ux_reflux

                                mom_fine_reflux = rho_fine_reflux * ux_fine_reflux * ux_fine_reflux +
                                                  cs2_flow * rho_fine_reflux
                                mom_parent_reflux = rho_reflux * ux_reflux * ux_reflux +
                                                    cs2_flow * rho_reflux

                                drho_raw_reflux = 0.0f0
                                djx_raw_reflux = jx_fine_reflux - jx_reflux
                                dq_raw_reflux = mom_fine_reflux - mom_parent_reflux
                                child_drho_raw_reflux = 0.0f0
                                child_djx_raw_reflux = 0.0f0
                                child_dq_raw_reflux = 0.0f0
                                if use_gpu_budget_reflux
                                    budget_base_reflux = x_sign_reflux > Int32(0) ? Int32(0) : Int32(6)
                                    parent_time_reflux = parent_face_budget[lx, ly, lz, pb_idx, budget_base_reflux + Int32(5)]
                                    child_time_reflux = parent_face_budget[lx, ly, lz, pb_idx, budget_base_reflux + Int32(6)]
                                    if parent_time_reflux > 0.0f0 && child_time_reflux > 0.0f0
                                        parent_mass_rate_reflux =
                                            parent_face_budget[lx, ly, lz, pb_idx, budget_base_reflux + Int32(1)] /
                                            parent_time_reflux
                                        child_mass_rate_reflux =
                                            parent_face_budget[lx, ly, lz, pb_idx, budget_base_reflux + Int32(2)] /
                                            child_time_reflux
                                        parent_mom_rate_reflux =
                                            parent_face_budget[lx, ly, lz, pb_idx, budget_base_reflux + Int32(3)] /
                                            parent_time_reflux
                                        child_mom_rate_reflux =
                                            parent_face_budget[lx, ly, lz, pb_idx, budget_base_reflux + Int32(4)] /
                                            child_time_reflux
                                        normal_mass_resid_reflux =
                                            parent_mass_rate_reflux - child_mass_rate_reflux
                                        normal_mom_resid_reflux =
                                            parent_mom_rate_reflux - child_mom_rate_reflux
                                        # The budget stores normal fluxes. Convert the
                                        # desired normal mass-flux correction back to x
                                        # momentum so the existing half-range projector
                                        # can keep using its x-moment basis.
                                        normal_mass_correction_reflux =
                                            -gpu_budget_reflux_mass_alpha *
                                            normal_mass_resid_reflux
                                        djx_raw_reflux = Float32(x_sign_reflux) *
                                            normal_mass_correction_reflux
                                        dq_macro_raw_reflux = gpu_budget_reflux_neutral_mom == Int32(1) ?
                                            0.0f0 :
                                            -gpu_budget_reflux_mom_alpha *
                                            normal_mom_resid_reflux
                                        un_reflux = Float32(x_sign_reflux) * ux_reflux
                                        denom_reflux = cs2_flow - un_reflux * un_reflux
                                        if abs(denom_reflux) > 1.0f-6
                                            drho_raw_reflux =
                                                (dq_macro_raw_reflux -
                                                 2.0f0 * un_reflux * normal_mass_correction_reflux) /
                                                denom_reflux
                                        end
                                        # The AMR gate's normal momentum flux is
                                        # macroscopic (rho*u_n^2 + cs2*rho). The
                                        # second-moment/stress slot does not move
                                        # that diagnostic, so keep it neutral here.
                                        dq_raw_reflux = 0.0f0
                                    else
                                        drho_raw_reflux = 0.0f0
                                        djx_raw_reflux = 0.0f0
                                        dq_raw_reflux = 0.0f0
                                    end
                                elseif face_time_snapshots == Int32(1)
                                    rho_fine_old_reflux = x_sign_reflux > Int32(0) ?
                                        max(fine_rho_x_pos_old_sum / face_w_reflux, 0.01f0) :
                                        max(fine_rho_x_neg_old_sum / face_w_reflux, 0.01f0)
                                    jx_fine_old_reflux = x_sign_reflux > Int32(0) ?
                                        fine_jx_x_pos_old_sum / face_w_reflux :
                                        fine_jx_x_neg_old_sum / face_w_reflux
                                    ux_fine_old_reflux = jx_fine_old_reflux / rho_fine_old_reflux
                                    rho_old_reflux = max(parent_rho_old[lx, ly, lz, pb_idx], 0.01f0)
                                    ux_old_reflux = parent_vel_old[lx, ly, lz, pb_idx, 1]
                                    jx_old_reflux = rho_old_reflux * ux_old_reflux
                                    mom_fine_old_reflux =
                                        rho_fine_old_reflux * ux_fine_old_reflux * ux_fine_old_reflux +
                                        cs2_flow * rho_fine_old_reflux
                                    mom_parent_old_reflux =
                                        rho_old_reflux * ux_old_reflux * ux_old_reflux +
                                        cs2_flow * rho_old_reflux
                                    djx_raw_reflux = 0.5f0 * (
                                        djx_raw_reflux + (jx_fine_old_reflux - jx_old_reflux)
                                    )
                                    dq_raw_reflux = 0.5f0 * (
                                        dq_raw_reflux + (mom_fine_old_reflux - mom_parent_old_reflux)
                                    )
                                end

                                if use_paired_reflux &&
                                   (paired_reflux_highpass == Int32(1) ||
                                    paired_reflux_lowpass == Int32(1))
                                    hp_blend_reflux =
                                        paired_reflux_highpass == Int32(1) ?
                                        clamp(paired_reflux_highpass_blend, 0.0f0, 1.0f0) :
                                        0.0f0
                                    lp_blend_reflux =
                                        paired_reflux_lowpass == Int32(1) ?
                                        clamp(paired_reflux_lowpass_blend, 0.0f0, 1.0f0) :
                                        0.0f0
                                    if hp_blend_reflux > 0.0f0 || lp_blend_reflux > 0.0f0
                                        sum_djx_hp_reflux = 0.0f0
                                        sum_dq_hp_reflux = 0.0f0
                                        n_hp_reflux = Int32(0)
                                        @inbounds for oy_reflux in Int32(-1):Int32(1),
                                                      oz_reflux in Int32(-1):Int32(1)
                                            ngx_reflux = gx
                                            ngy_reflux = gy + oy_reflux
                                            ngz_reflux = gz + oz_reflux
                                            nlx_reflux, nly_reflux, nlz_reflux,
                                            nb_reflux, valid_parent_reflux =
                                                lookup_parent_cell(
                                                    parent_block_pointer,
                                                    parent_dim_x, parent_dim_y, parent_dim_z,
                                                    bs,
                                                    ngx_reflux, ngy_reflux, ngz_reflux
                                                )
                                            if valid_parent_reflux &&
                                               !parent_obstacle[nlx_reflux, nly_reflux, nlz_reflux, nb_reflux] &&
                                               count_fine_children(
                                                   child_block_pointer,
                                                   child_dim_x, child_dim_y, child_dim_z,
                                                   bs,
                                                   ngx_reflux, ngy_reflux, ngz_reflux
                                               ) == Int32(0) &&
                                               coarse_cell_has_fine_children(
                                                   child_block_pointer,
                                                   child_dim_x, child_dim_y, child_dim_z,
                                                   bs,
                                                   ngx_reflux + x_sign_reflux,
                                                   ngy_reflux,
                                                   ngz_reflux
                                               )
                                                face_fx0_hp_reflux =
                                                    x_sign_reflux > Int32(0) ?
                                                    Int32(2) * ngx_reflux + Int32(1) :
                                                    Int32(2) * ngx_reflux - Int32(2)
                                                face_fy0_hp_reflux =
                                                    Int32(2) * ngy_reflux - Int32(1)
                                                face_fz0_hp_reflux =
                                                    Int32(2) * ngz_reflux - Int32(1)
                                                sum_r_hp_reflux = 0.0f0
                                                sum_jx_hp_reflux = 0.0f0
                                                n_child_hp_reflux = Int32(0)
                                                for sy_hp_reflux in Int32(0):Int32(1),
                                                    sz_hp_reflux in Int32(0):Int32(1)
                                                    clx_hp_reflux, cly_hp_reflux,
                                                    clz_hp_reflux, cb_hp_reflux,
                                                    valid_child_hp_reflux =
                                                        lookup_child_cell(
                                                            child_block_pointer,
                                                            child_dim_x, child_dim_y, child_dim_z,
                                                            bs,
                                                            face_fx0_hp_reflux,
                                                            face_fy0_hp_reflux + sy_hp_reflux,
                                                            face_fz0_hp_reflux + sz_hp_reflux
                                                        )
                                                    if valid_child_hp_reflux &&
                                                       !child_obstacle[
                                                           clx_hp_reflux, cly_hp_reflux,
                                                           clz_hp_reflux, cb_hp_reflux
                                                       ]
                                                        rho_child_hp_reflux = max(
                                                            child_rho[
                                                                clx_hp_reflux, cly_hp_reflux,
                                                                clz_hp_reflux, cb_hp_reflux
                                                            ],
                                                            0.01f0
                                                        )
                                                        ux_child_hp_reflux =
                                                            child_vel[
                                                                clx_hp_reflux, cly_hp_reflux,
                                                                clz_hp_reflux, cb_hp_reflux, 1
                                                            ]
                                                        sum_r_hp_reflux += rho_child_hp_reflux
                                                        sum_jx_hp_reflux +=
                                                            rho_child_hp_reflux *
                                                            ux_child_hp_reflux
                                                        n_child_hp_reflux += Int32(1)
                                                    end
                                                end
                                                if n_child_hp_reflux == Int32(4)
                                                    rho_parent_hp_reflux = max(
                                                        parent_rho[
                                                            nlx_reflux, nly_reflux,
                                                            nlz_reflux, nb_reflux
                                                        ],
                                                        0.01f0
                                                    )
                                                    ux_parent_hp_reflux =
                                                        parent_vel[
                                                            nlx_reflux, nly_reflux,
                                                            nlz_reflux, nb_reflux, 1
                                                        ]
                                                    jx_parent_hp_reflux =
                                                        rho_parent_hp_reflux *
                                                        ux_parent_hp_reflux
                                                    rho_fine_hp_reflux =
                                                        max(0.25f0 * sum_r_hp_reflux, 0.01f0)
                                                    jx_fine_hp_reflux =
                                                        0.25f0 * sum_jx_hp_reflux
                                                    ux_fine_hp_reflux =
                                                        jx_fine_hp_reflux / rho_fine_hp_reflux
                                                    mom_parent_hp_reflux =
                                                        rho_parent_hp_reflux *
                                                        ux_parent_hp_reflux *
                                                        ux_parent_hp_reflux +
                                                        cs2_flow * rho_parent_hp_reflux
                                                    mom_fine_hp_reflux =
                                                        rho_fine_hp_reflux *
                                                        ux_fine_hp_reflux *
                                                        ux_fine_hp_reflux +
                                                        cs2_flow * rho_fine_hp_reflux
                                                    sum_djx_hp_reflux +=
                                                        jx_fine_hp_reflux - jx_parent_hp_reflux
                                                    sum_dq_hp_reflux +=
                                                        mom_fine_hp_reflux - mom_parent_hp_reflux
                                                    n_hp_reflux += Int32(1)
                                                end
                                            end
                                        end
                                        if n_hp_reflux > Int32(0)
                                            inv_hp_reflux = 1.0f0 / Float32(n_hp_reflux)
                                            mean_djx_hp_reflux =
                                                sum_djx_hp_reflux * inv_hp_reflux
                                            mean_dq_hp_reflux =
                                                sum_dq_hp_reflux * inv_hp_reflux
                                            if lp_blend_reflux > 0.0f0
                                                djx_raw_reflux =
                                                    (1.0f0 - lp_blend_reflux) * djx_raw_reflux +
                                                    lp_blend_reflux * mean_djx_hp_reflux
                                                dq_raw_reflux =
                                                    (1.0f0 - lp_blend_reflux) * dq_raw_reflux +
                                                    lp_blend_reflux * mean_dq_hp_reflux
                                            else
                                                djx_raw_reflux =
                                                    (1.0f0 - hp_blend_reflux) * djx_raw_reflux +
                                                    hp_blend_reflux *
                                                    (djx_raw_reflux - mean_djx_hp_reflux)
                                                dq_raw_reflux =
                                                    (1.0f0 - hp_blend_reflux) * dq_raw_reflux +
                                                    hp_blend_reflux *
                                                    (dq_raw_reflux - mean_dq_hp_reflux)
                                            end
                                        end
                                    end
                                end

                                if use_paired_reflux
                                    if paired_reflux_shared_state == Int32(1)
                                        shared_drho_delta_reflux = rho_fine_reflux - rho_reflux
                                        shared_djn_delta_reflux =
                                            Float32(x_sign_reflux) * (jx_fine_reflux - jx_reflux)
                                        shared_dq_delta_reflux = mom_fine_reflux - mom_parent_reflux

                                        drho_raw_reflux =
                                            0.5f0 * paired_reflux_shared_state_rho_alpha *
                                            shared_drho_delta_reflux
                                        child_drho_raw_reflux = -drho_raw_reflux
                                        parent_djn_reflux =
                                            0.5f0 * paired_reflux_mass_alpha *
                                            shared_djn_delta_reflux
                                        child_djn_reflux = -parent_djn_reflux
                                        parent_dq_reflux =
                                            0.5f0 * paired_mom_alpha_eff_reflux *
                                            shared_dq_delta_reflux
                                        child_dq_reflux = -parent_dq_reflux
                                        djx_raw_reflux =
                                            Float32(x_sign_reflux) * parent_djn_reflux
                                        child_djx_raw_reflux =
                                            Float32(x_sign_reflux) * child_djn_reflux
                                        dq_raw_reflux = parent_dq_reflux
                                        child_dq_raw_reflux = child_dq_reflux
                                    else
                                        parent_frac_reflux = clamp(
                                            paired_reflux_parent_fraction,
                                            0.0f0,
                                            1.0f0
                                        )
                                        child_frac_reflux = 1.0f0 - parent_frac_reflux
                                        mass_parent_frac_reflux = clamp(
                                            paired_reflux_mass_parent_fraction,
                                            0.0f0,
                                            1.0f0
                                        )
                                        mass_child_frac_reflux =
                                            1.0f0 - mass_parent_frac_reflux
                                        normal_mass_delta_reflux =
                                            Float32(x_sign_reflux) * djx_raw_reflux
                                        normal_mom_delta_reflux = dq_raw_reflux
                                        if paired_reflux_budget_mean_preserving == Int32(1)
                                            budget_base_paired = x_sign_reflux > Int32(0) ? Int32(0) : Int32(6)
                                            parent_time_paired =
                                                parent_face_budget[
                                                    lx, ly, lz, pb_idx,
                                                    budget_base_paired + Int32(5)
                                                ]
                                            child_time_paired =
                                                parent_face_budget[
                                                    lx, ly, lz, pb_idx,
                                                    budget_base_paired + Int32(6)
                                                ]
                                            if parent_time_paired > 0.0f0 && child_time_paired > 0.0f0
                                                parent_mass_rate_paired =
                                                    parent_face_budget[
                                                        lx, ly, lz, pb_idx,
                                                        budget_base_paired + Int32(1)
                                                    ] / parent_time_paired
                                                child_mass_rate_paired =
                                                    parent_face_budget[
                                                        lx, ly, lz, pb_idx,
                                                        budget_base_paired + Int32(2)
                                                    ] / child_time_paired
                                                parent_mom_rate_paired =
                                                    parent_face_budget[
                                                        lx, ly, lz, pb_idx,
                                                        budget_base_paired + Int32(3)
                                                    ] / parent_time_paired
                                                child_mom_rate_paired =
                                                    parent_face_budget[
                                                        lx, ly, lz, pb_idx,
                                                        budget_base_paired + Int32(4)
                                                    ] / child_time_paired
                                                normal_mass_delta_reflux -=
                                                    parent_mass_rate_paired - child_mass_rate_paired
                                                normal_mom_delta_reflux -=
                                                    parent_mom_rate_paired - child_mom_rate_paired
                                            else
                                                normal_mass_delta_reflux = 0.0f0
                                                normal_mom_delta_reflux = 0.0f0
                                            end
                                        end

                                        if paired_reflux_face_conservative == Int32(1)
                                            parent_djn_reflux =
                                                0.5f0 * paired_reflux_mass_alpha *
                                                normal_mass_delta_reflux
                                            parent_dq_reflux =
                                                0.5f0 * paired_mom_alpha_eff_reflux *
                                                normal_mom_delta_reflux
                                            child_djn_reflux = -parent_djn_reflux
                                            child_dq_reflux = -parent_dq_reflux
                                        else
                                            parent_djn_reflux =
                                                paired_reflux_mass_alpha *
                                                mass_parent_frac_reflux *
                                                normal_mass_delta_reflux
                                            parent_dq_reflux =
                                                paired_mom_alpha_eff_reflux *
                                                parent_frac_reflux * normal_mom_delta_reflux
                                            child_djn_reflux =
                                                -paired_reflux_mass_alpha *
                                                mass_child_frac_reflux *
                                                normal_mass_delta_reflux
                                            child_dq_reflux =
                                                -paired_mom_alpha_eff_reflux *
                                                child_frac_reflux * normal_mom_delta_reflux
                                        end

                                        un_parent_reflux = Float32(x_sign_reflux) * ux_reflux
                                        denom_parent_reflux =
                                            cs2_flow - un_parent_reflux * un_parent_reflux
                                        drho_raw_reflux = abs(denom_parent_reflux) > 1.0f-6 ?
                                            (parent_dq_reflux -
                                             2.0f0 * un_parent_reflux * parent_djn_reflux) /
                                            denom_parent_reflux :
                                            0.0f0
                                        djx_raw_reflux =
                                            Float32(x_sign_reflux) * parent_djn_reflux

                                        un_child_reflux =
                                            Float32(x_sign_reflux) * ux_fine_reflux
                                        denom_child_reflux =
                                            cs2_flow - un_child_reflux * un_child_reflux
                                        child_drho_raw_reflux = abs(denom_child_reflux) > 1.0f-6 ?
                                            (child_dq_reflux -
                                             2.0f0 * un_child_reflux * child_djn_reflux) /
                                            denom_child_reflux :
                                            0.0f0
                                        child_djx_raw_reflux =
                                            Float32(x_sign_reflux) * child_djn_reflux

                                        # If we change rho and rho*u to realize a
                                        # pressure-like flux but force the raw
                                        # second moment to remain fixed, the
                                        # difference is injected as a spurious
                                        # Pi_neq jump. The consistent option gives
                                        # the projector the same normal second
                                        # moment implied by the paired macro-flux.
                                        second_moment_blend_reflux =
                                            paired_reflux_second_moment == Int32(1) ?
                                            clamp(paired_reflux_second_moment_blend, 0.0f0, 1.0f0) :
                                            0.0f0
                                        dq_raw_reflux =
                                            second_moment_blend_reflux * parent_dq_reflux
                                        child_dq_raw_reflux =
                                            second_moment_blend_reflux * child_dq_reflux
                                    end
                                end

                                u_scale_reflux = max(max(abs(ux_reflux), abs(ux_fine_reflux)), 1.0f-3)
                                paired_mass_cap_eff_reflux = clamp(
                                    paired_reflux_mass_cap_scale,
                                    0.0f0,
                                    1.0f0
                                )
                                if use_paired_reflux &&
                                   paired_reflux_mass_alpha > 0.0f0 &&
                                   paired_reflux_mass_cap_hold_steps > Int32(0) &&
                                   current_top_step > Int32(0) &&
                                   current_top_step <= paired_reflux_mass_cap_hold_steps
                                    paired_mass_cap_eff_reflux = clamp(
                                        paired_reflux_mass_cap_initial,
                                        0.0f0,
                                        1.0f0
                                    )
                                end
                                drho_cap_abs_reflux = use_gpu_budget_reflux ?
                                    gpu_budget_reflux_drho_cap_abs : drho_cap_x_abs
                                drho_cap_rel_reflux = use_gpu_budget_reflux ?
                                    gpu_budget_reflux_drho_cap_rel : drho_cap_x_rel
                                paired_drho_cap_scale_reflux = use_paired_reflux ?
                                    max(1.0f0, abs(paired_mom_alpha_eff_reflux)) :
                                    1.0f0
                                drho_cap_reflux = paired_drho_cap_scale_reflux *
                                    max(drho_cap_abs_reflux,
                                        drho_cap_rel_reflux * rho_reflux)
                                dj_cap_reflux = rho_reflux *
                                    max(flux_du_cap_abs, flux_du_cap_rel * u_scale_reflux)
                                if use_paired_reflux &&
                                   paired_reflux_mass_alpha > 0.0f0
                                    dj_cap_reflux *= paired_mass_cap_eff_reflux
                                end
                                mom_scale_reflux = max(max(abs(mom_fine_reflux), abs(mom_parent_reflux)),
                                                       cs2_flow * rho_reflux)
                                dq_cap_abs_reflux = use_gpu_budget_reflux ?
                                    gpu_budget_reflux_mom_cap_abs : momflux_stress_cap_abs
                                dq_cap_rel_reflux = use_gpu_budget_reflux ?
                                    gpu_budget_reflux_mom_cap_rel : momflux_stress_cap_rel
                                dq_cap_reflux = max(dq_cap_abs_reflux,
                                                    dq_cap_rel_reflux * mom_scale_reflux)
                                child_drho_reflux = 0.0f0
                                child_djx_reflux = 0.0f0
                                child_dq_reflux = 0.0f0
                                if use_paired_reflux
                                    child_drho_cap_reflux = paired_drho_cap_scale_reflux * max(
                                        drho_cap_abs_reflux,
                                        drho_cap_rel_reflux * rho_fine_reflux
                                    )
                                    child_dj_cap_reflux = rho_fine_reflux *
                                        max(flux_du_cap_abs, flux_du_cap_rel * u_scale_reflux)
                                    if paired_reflux_mass_alpha > 0.0f0
                                        child_dj_cap_reflux *= paired_mass_cap_eff_reflux
                                    end
                                    child_mom_scale_reflux =
                                        max(abs(mom_fine_reflux), cs2_flow * rho_fine_reflux)
                                    child_dq_cap_reflux = max(
                                        dq_cap_abs_reflux,
                                        dq_cap_rel_reflux * child_mom_scale_reflux
                                    )
                                    if paired_reflux_macro_consistent_caps == Int32(1) &&
                                       paired_reflux_face_conservative == Int32(0) &&
                                       paired_reflux_coupled_caps == Int32(0) &&
                                       paired_reflux_coupled_mass_caps == Int32(0)
                                        parent_macro_scale_reflux = 1.0f0
                                        child_macro_scale_reflux = 1.0f0
                                        if paired_reflux_macro_pairwise_caps == Int32(1)
                                            pair_macro_scale_reflux = 1.0f0
                                            if abs(drho_raw_reflux) > drho_cap_reflux
                                                pair_macro_scale_reflux = min(
                                                    pair_macro_scale_reflux,
                                                    drho_cap_reflux / max(abs(drho_raw_reflux), 1.0f-12)
                                                )
                                            end
                                            if abs(djx_raw_reflux) > dj_cap_reflux
                                                pair_macro_scale_reflux = min(
                                                    pair_macro_scale_reflux,
                                                    dj_cap_reflux / max(abs(djx_raw_reflux), 1.0f-12)
                                                )
                                            end
                                            if abs(dq_raw_reflux) > dq_cap_reflux
                                                pair_macro_scale_reflux = min(
                                                    pair_macro_scale_reflux,
                                                    dq_cap_reflux / max(abs(dq_raw_reflux), 1.0f-12)
                                                )
                                            end
                                            if abs(child_drho_raw_reflux) > child_drho_cap_reflux
                                                pair_macro_scale_reflux = min(
                                                    pair_macro_scale_reflux,
                                                    child_drho_cap_reflux /
                                                    max(abs(child_drho_raw_reflux), 1.0f-12)
                                                )
                                            end
                                            if abs(child_djx_raw_reflux) > child_dj_cap_reflux
                                                pair_macro_scale_reflux = min(
                                                    pair_macro_scale_reflux,
                                                    child_dj_cap_reflux /
                                                    max(abs(child_djx_raw_reflux), 1.0f-12)
                                                )
                                            end
                                            if abs(child_dq_raw_reflux) > child_dq_cap_reflux
                                                pair_macro_scale_reflux = min(
                                                    pair_macro_scale_reflux,
                                                    child_dq_cap_reflux /
                                                    max(abs(child_dq_raw_reflux), 1.0f-12)
                                                )
                                            end
                                            parent_macro_scale_reflux = pair_macro_scale_reflux
                                            child_macro_scale_reflux = pair_macro_scale_reflux
                                        else
                                            if abs(drho_raw_reflux) > drho_cap_reflux
                                                parent_macro_scale_reflux = min(
                                                    parent_macro_scale_reflux,
                                                    drho_cap_reflux / max(abs(drho_raw_reflux), 1.0f-12)
                                                )
                                            end
                                            if abs(djx_raw_reflux) > dj_cap_reflux
                                                parent_macro_scale_reflux = min(
                                                    parent_macro_scale_reflux,
                                                    dj_cap_reflux / max(abs(djx_raw_reflux), 1.0f-12)
                                                )
                                            end

                                            if abs(child_drho_raw_reflux) > child_drho_cap_reflux
                                                child_macro_scale_reflux = min(
                                                    child_macro_scale_reflux,
                                                    child_drho_cap_reflux /
                                                    max(abs(child_drho_raw_reflux), 1.0f-12)
                                                )
                                            end
                                            if abs(child_djx_raw_reflux) > child_dj_cap_reflux
                                                child_macro_scale_reflux = min(
                                                    child_macro_scale_reflux,
                                                    child_dj_cap_reflux /
                                                    max(abs(child_djx_raw_reflux), 1.0f-12)
                                                )
                                            end
                                        end

                                        drho_reflux =
                                            alpha_reflux * parent_macro_scale_reflux * drho_raw_reflux
                                        djx_reflux =
                                            alpha_reflux * parent_macro_scale_reflux * djx_raw_reflux
                                        dq_reflux =
                                            alpha_reflux * parent_macro_scale_reflux * dq_raw_reflux
                                        child_drho_reflux =
                                            alpha_reflux * child_macro_scale_reflux * child_drho_raw_reflux
                                        child_djx_reflux =
                                            alpha_reflux * child_macro_scale_reflux * child_djx_raw_reflux
                                        child_dq_reflux =
                                            alpha_reflux * child_macro_scale_reflux * child_dq_raw_reflux
                                    elseif paired_reflux_face_conservative == Int32(1) ||
                                       paired_reflux_coupled_caps == Int32(1) ||
                                       paired_reflux_coupled_mass_caps == Int32(1)
                                        drho_scale_reflux = 1.0f0
                                        if abs(drho_raw_reflux) > drho_cap_reflux
                                            drho_scale_reflux = min(
                                                drho_scale_reflux,
                                                drho_cap_reflux / max(abs(drho_raw_reflux), 1.0f-12)
                                            )
                                        end
                                        if abs(child_drho_raw_reflux) > child_drho_cap_reflux
                                            drho_scale_reflux = min(
                                                drho_scale_reflux,
                                                child_drho_cap_reflux /
                                                max(abs(child_drho_raw_reflux), 1.0f-12)
                                            )
                                        end

                                        dj_scale_reflux = 1.0f0
                                        if abs(djx_raw_reflux) > dj_cap_reflux
                                            dj_scale_reflux = min(
                                                dj_scale_reflux,
                                                dj_cap_reflux / max(abs(djx_raw_reflux), 1.0f-12)
                                            )
                                        end
                                        if abs(child_djx_raw_reflux) > child_dj_cap_reflux
                                            dj_scale_reflux = min(
                                                dj_scale_reflux,
                                                child_dj_cap_reflux /
                                                max(abs(child_djx_raw_reflux), 1.0f-12)
                                            )
                                        end

                                        drho_reflux = alpha_reflux * drho_scale_reflux * drho_raw_reflux
                                        djx_reflux = alpha_reflux * dj_scale_reflux * djx_raw_reflux
                                        child_drho_reflux =
                                            alpha_reflux * drho_scale_reflux * child_drho_raw_reflux
                                        child_djx_reflux =
                                            alpha_reflux * dj_scale_reflux * child_djx_raw_reflux
                                        if paired_reflux_coupled_mass_caps == Int32(1) &&
                                           paired_reflux_coupled_caps == Int32(0) &&
                                           paired_reflux_face_conservative == Int32(0)
                                            dq_reflux = alpha_reflux *
                                                clamp(dq_raw_reflux,
                                                      -dq_cap_reflux,
                                                      dq_cap_reflux)
                                            child_dq_reflux = alpha_reflux *
                                                clamp(child_dq_raw_reflux,
                                                      -child_dq_cap_reflux,
                                                      child_dq_cap_reflux)
                                        else
                                            dq_scale_reflux = 1.0f0
                                            if abs(dq_raw_reflux) > dq_cap_reflux
                                                dq_scale_reflux = min(
                                                    dq_scale_reflux,
                                                    dq_cap_reflux / max(abs(dq_raw_reflux), 1.0f-12)
                                                )
                                            end
                                            if abs(child_dq_raw_reflux) > child_dq_cap_reflux
                                                dq_scale_reflux = min(
                                                    dq_scale_reflux,
                                                    child_dq_cap_reflux /
                                                    max(abs(child_dq_raw_reflux), 1.0f-12)
                                                )
                                            end
                                            dq_reflux = alpha_reflux * dq_scale_reflux * dq_raw_reflux
                                            child_dq_reflux =
                                                alpha_reflux * dq_scale_reflux * child_dq_raw_reflux
                                        end
                                    else
                                        child_drho_reflux = alpha_reflux *
                                            clamp(child_drho_raw_reflux,
                                                  -child_drho_cap_reflux,
                                                  child_drho_cap_reflux)
                                        child_djx_reflux = alpha_reflux *
                                            clamp(child_djx_raw_reflux,
                                                  -child_dj_cap_reflux,
                                                  child_dj_cap_reflux)
                                        child_dq_reflux = alpha_reflux *
                                            clamp(child_dq_raw_reflux,
                                                  -child_dq_cap_reflux,
                                                  child_dq_cap_reflux)
                                        drho_reflux = alpha_reflux *
                                            clamp(drho_raw_reflux, -drho_cap_reflux, drho_cap_reflux)
                                        djx_reflux = alpha_reflux *
                                            clamp(djx_raw_reflux, -dj_cap_reflux, dj_cap_reflux)
                                        dq_reflux = alpha_reflux *
                                            clamp(dq_raw_reflux, -dq_cap_reflux, dq_cap_reflux)
                                    end
                                else
                                    drho_reflux = alpha_reflux *
                                        clamp(drho_raw_reflux, -drho_cap_reflux, drho_cap_reflux)
                                    djx_reflux = alpha_reflux *
                                        clamp(djx_raw_reflux, -dj_cap_reflux, dj_cap_reflux)
                                    dq_reflux = alpha_reflux *
                                        clamp(dq_raw_reflux, -dq_cap_reflux, dq_cap_reflux)
                                end

                                if abs(drho_reflux) > 1.0f-12 ||
                                   abs(djx_reflux) > 1.0f-12 ||
                                   abs(dq_reflux) > 1.0f-12
                                    sign_reflux = Float32(x_sign_reflux)
                                    m0_reflux = 0.0f0
                                    m1_reflux = 0.0f0
                                    m2_reflux = 0.0f0
                                    m3_reflux = 0.0f0
                                    m4_reflux = 0.0f0
                                    @inbounds for kk in 1:39
                                        if kk <= n_dirs_flow
                                            cn_reflux = sign_reflux * (Float32(cx_arr[kk]) - shift_x)
                                            if cn_reflux > 0.0f0
                                                wk_reflux = Float32(w_arr[kk])
                                                cn2_reflux = cn_reflux * cn_reflux
                                                m0_reflux += wk_reflux
                                                m1_reflux += wk_reflux * cn_reflux
                                                m2_reflux += wk_reflux * cn2_reflux
                                                m3_reflux += wk_reflux * cn2_reflux * cn_reflux
                                                m4_reflux += wk_reflux * cn2_reflux * cn2_reflux
                                            end
                                        end
                                    end

                                    r0_reflux = drho_reflux
                                    r1_reflux = sign_reflux * (djx_reflux - shift_x * drho_reflux)
                                    r2_reflux = dq_reflux
                                    det_reflux =
                                        m0_reflux * (m2_reflux * m4_reflux - m3_reflux * m3_reflux) -
                                        m1_reflux * (m1_reflux * m4_reflux - m3_reflux * m2_reflux) +
                                        m2_reflux * (m1_reflux * m3_reflux - m2_reflux * m2_reflux)

                                    if abs(det_reflux) > 1.0f-12
                                        a_reflux = (
                                            r0_reflux * (m2_reflux * m4_reflux - m3_reflux * m3_reflux) -
                                            m1_reflux * (r1_reflux * m4_reflux - m3_reflux * r2_reflux) +
                                            m2_reflux * (r1_reflux * m3_reflux - m2_reflux * r2_reflux)
                                        ) / det_reflux
                                        b_reflux = (
                                            m0_reflux * (r1_reflux * m4_reflux - m3_reflux * r2_reflux) -
                                            r0_reflux * (m1_reflux * m4_reflux - m3_reflux * m2_reflux) +
                                            m2_reflux * (m1_reflux * r2_reflux - r1_reflux * m2_reflux)
                                        ) / det_reflux
                                        c_reflux = (
                                            m0_reflux * (m2_reflux * r2_reflux - r1_reflux * m3_reflux) -
                                            m1_reflux * (m1_reflux * r2_reflux - r1_reflux * m2_reflux) +
                                            r0_reflux * (m1_reflux * m3_reflux - m2_reflux * m2_reflux)
                                        ) / det_reflux

                                        pos_limiter_reflux = 1.0f0
                                        @inbounds for kk in 1:39
                                            if kk <= n_dirs_flow
                                                cn_reflux = sign_reflux * (Float32(cx_arr[kk]) - shift_x)
                                                if cn_reflux > 0.0f0
                                                    wk_reflux = Float32(w_arr[kk])
                                                    df_reflux = wk_reflux *
                                                        (a_reflux + b_reflux * cn_reflux +
                                                         c_reflux * cn_reflux * cn_reflux)
                                                    if df_reflux < 0.0f0
                                                        f_now_reflux = parent_f_current[lx, ly, lz, pb_idx, kk]
                                                        pos_limiter_reflux = min(
                                                            pos_limiter_reflux,
                                                            max((f_now_reflux - 1.0f-10) / (-df_reflux), 0.0f0)
                                                        )
                                                    end
                                                end
                                            end
                                        end
                                        pos_limiter_reflux = min(pos_limiter_reflux, 1.0f0)

                                        rho_after_reflux = 0.0f0
                                        jx_after_reflux = 0.0f0
                                        jy_after_reflux = 0.0f0
                                        jz_after_reflux = 0.0f0
                                        ekin_after_reflux = 0.0f0
                                        @inbounds for kk in 1:39
                                            if kk <= n_dirs_flow
                                                cx_reflux = Float32(cx_arr[kk])
                                                cy_reflux = Float32(cy_arr[kk])
                                                cz_reflux = Float32(cz_arr[kk])
                                                cn_reflux = sign_reflux * (cx_reflux - shift_x)
                                                fval_reflux = parent_f_current[lx, ly, lz, pb_idx, kk]
                                                if cn_reflux > 0.0f0
                                                    wk_reflux = Float32(w_arr[kk])
                                                    df_reflux = pos_limiter_reflux * wk_reflux *
                                                        (a_reflux + b_reflux * cn_reflux +
                                                         c_reflux * cn_reflux * cn_reflux)
                                                    fval_reflux = max(fval_reflux + df_reflux, 1.0f-10)
                                                    parent_f_current[lx, ly, lz, pb_idx, kk] = fval_reflux
                                                end
                                                rho_after_reflux += fval_reflux
                                                jx_after_reflux += fval_reflux * cx_reflux
                                                jy_after_reflux += fval_reflux * cy_reflux
                                                jz_after_reflux += fval_reflux * cz_reflux
                                                cxu_reflux = cx_reflux - shift_x
                                                cyu_reflux = cy_reflux - shift_y
                                                czu_reflux = cz_reflux - shift_z
                                                ekin_after_reflux += fval_reflux *
                                                    (cxu_reflux * cxu_reflux +
                                                     cyu_reflux * cyu_reflux +
                                                     czu_reflux * czu_reflux)
                                            end
                                        end

                                        rho_after_reflux = max(rho_after_reflux, 0.01f0)
                                        parent_rho[lx, ly, lz, pb_idx] = rho_after_reflux
                                        parent_vel[lx, ly, lz, pb_idx, 1] = jx_after_reflux / rho_after_reflux
                                        parent_vel[lx, ly, lz, pb_idx, 2] = jy_after_reflux / rho_after_reflux
                                        parent_vel[lx, ly, lz, pb_idx, 3] = jz_after_reflux / rho_after_reflux
                                        if restrict_thermal == Int32(1)
                                            parent_kinetic_energy[lx, ly, lz, pb_idx] = ekin_after_reflux
                                        end

                                        if use_paired_reflux &&
                                           paired_reflux_post_stress == Int32(1) &&
                                           face_stress_shift == Int32(1) &&
                                           preserve_pi_neq == Int32(1)
                                            post_stress_alpha_reflux = clamp(
                                                paired_reflux_post_stress_alpha,
                                                0.0f0,
                                                1.0f0
                                            )
                                            if post_stress_alpha_reflux > 0.0f0
                                                ux_post_stress = parent_vel[lx, ly, lz, pb_idx, 1]
                                                uy_post_stress = parent_vel[lx, ly, lz, pb_idx, 2]
                                                uz_post_stress = parent_vel[lx, ly, lz, pb_idx, 3]
                                                temp_post_stress = restrict_thermal == Int32(1) ?
                                                    clamp(
                                                        max(parent_temperature[lx, ly, lz, pb_idx], 0.01f0),
                                                        0.1f0,
                                                        10.0f0
                                                    ) :
                                                    1.0f0
                                                lambda_post_stress = use_numerical_eq ?
                                                    solve_13mom_lambdas_d3q39(
                                                        rho_after_reflux,
                                                        ux_post_stress - shift_x,
                                                        uy_post_stress - shift_y,
                                                        uz_post_stress - shift_z,
                                                        cx_arr, cy_arr, cz_arr, w_arr,
                                                        cs2_flow, Int32(50), temp_post_stress,
                                                        shift_x, shift_y, shift_z,
                                                        100.0f0, eos_R_flow          # SOTA A3 — match collision-kernel EOS R
                                                    ) :
                                                    (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
                                                     0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
                                                mass_corr_post_stress = use_numerical_eq ?
                                                    compute_13mom_mass_correction_d3q39(
                                                        rho_after_reflux,
                                                        lambda_post_stress,
                                                        cx_arr, cy_arr, cz_arr, w_arr,
                                                        cs2_flow, shift_x, shift_y, shift_z
                                                    ) :
                                                    1.0f0

                                                pxx_post_stress = 0.0f0
                                                pyy_post_stress = 0.0f0
                                                pzz_post_stress = 0.0f0
                                                pxy_post_stress = 0.0f0
                                                pxz_post_stress = 0.0f0
                                                pyz_post_stress = 0.0f0
                                                @inbounds for kk in 1:39
                                                    if kk <= n_dirs_flow
                                                        cxk_post = Float32(cx_arr[kk]) - shift_x
                                                        cyk_post = Float32(cy_arr[kk]) - shift_y
                                                        czk_post = Float32(cz_arr[kk]) - shift_z
                                                        wk_post = Float32(w_arr[kk])
                                                        feq_post = use_numerical_eq ?
                                                            mass_corr_post_stress * eval_13mom_feq_single(
                                                                rho_after_reflux,
                                                                lambda_post_stress,
                                                                cxk_post, cyk_post, czk_post,
                                                                wk_post, cs2_flow
                                                            ) :
                                                            calculate_equilibrium_auto(
                                                                rho_after_reflux,
                                                                ux_post_stress,
                                                                uy_post_stress,
                                                                uz_post_stress,
                                                                wk_post, cxk_post, cyk_post, czk_post,
                                                                flow_is_compressible
                                                            )
                                                        fneq_post =
                                                            parent_f_current[lx, ly, lz, pb_idx, kk] -
                                                            feq_post
                                                        pxx_post_stress += cxk_post * cxk_post * fneq_post
                                                        pyy_post_stress += cyk_post * cyk_post * fneq_post
                                                        pzz_post_stress += czk_post * czk_post * fneq_post
                                                        pxy_post_stress += cxk_post * cyk_post * fneq_post
                                                        pxz_post_stress += cxk_post * czk_post * fneq_post
                                                        pyz_post_stress += cyk_post * czk_post * fneq_post
                                                    end
                                                end

                                                dpxx_post = pxx_fine_avg - pxx_post_stress
                                                dpyy_post = pyy_fine_avg - pyy_post_stress
                                                dpzz_post = pzz_fine_avg - pzz_post_stress
                                                dpxy_post = pxy_fine_avg - pxy_post_stress
                                                dpxz_post = pxz_fine_avg - pxz_post_stress
                                                dpyz_post = pyz_fine_avg - pyz_post_stress
                                                parent_pi_norm_post = sqrt(
                                                    pxx_post_stress*pxx_post_stress +
                                                    pyy_post_stress*pyy_post_stress +
                                                    pzz_post_stress*pzz_post_stress +
                                                    2.0f0 * (
                                                        pxy_post_stress*pxy_post_stress +
                                                        pxz_post_stress*pxz_post_stress +
                                                        pyz_post_stress*pyz_post_stress
                                                    )
                                                )
                                                fine_pi_norm_post = sqrt(
                                                    pxx_fine_avg*pxx_fine_avg +
                                                    pyy_fine_avg*pyy_fine_avg +
                                                    pzz_fine_avg*pzz_fine_avg +
                                                    2.0f0 * (
                                                        pxy_fine_avg*pxy_fine_avg +
                                                        pxz_fine_avg*pxz_fine_avg +
                                                        pyz_fine_avg*pyz_fine_avg
                                                    )
                                                )
                                                dpi_norm_post = sqrt(
                                                    dpxx_post*dpxx_post +
                                                    dpyy_post*dpyy_post +
                                                    dpzz_post*dpzz_post +
                                                    2.0f0 * (
                                                        dpxy_post*dpxy_post +
                                                        dpxz_post*dpxz_post +
                                                        dpyz_post*dpyz_post
                                                    )
                                                )
                                                pi_scale_post = max(
                                                    max(parent_pi_norm_post, fine_pi_norm_post),
                                                    1.0f-8
                                                )
                                                dpi_cap_post = max(
                                                    face_stress_abs_cap,
                                                    face_stress_rel_cap * pi_scale_post
                                                )
                                                stress_limiter_post =
                                                    dpi_norm_post > dpi_cap_post ?
                                                    (paired_reflux_post_stress_soft_limit == Int32(1) ?
                                                     dpi_cap_post / max(dpi_norm_post, 1.0f-12) :
                                                     0.0f0) :
                                                    1.0f0
                                                dpi_xx_post =
                                                    post_stress_alpha_reflux *
                                                    stress_limiter_post * dpxx_post
                                                dpi_yy_post =
                                                    post_stress_alpha_reflux *
                                                    stress_limiter_post * dpyy_post
                                                dpi_zz_post =
                                                    post_stress_alpha_reflux *
                                                    stress_limiter_post * dpzz_post
                                                dpi_xy_post =
                                                    post_stress_alpha_reflux *
                                                    stress_limiter_post * dpxy_post
                                                dpi_xz_post =
                                                    post_stress_alpha_reflux *
                                                    stress_limiter_post * dpxz_post
                                                dpi_yz_post =
                                                    post_stress_alpha_reflux *
                                                    stress_limiter_post * dpyz_post

                                                pos_limiter_post = 1.0f0
                                                @inbounds for kk in 1:39
                                                    if kk <= n_dirs_flow
                                                        cxk_post = Float32(cx_arr[kk]) - shift_x
                                                        cyk_post = Float32(cy_arr[kk]) - shift_y
                                                        czk_post = Float32(cz_arr[kk]) - shift_z
                                                        wk_post = Float32(w_arr[kk])
                                                        df_post = project_second_order_stress_mode(
                                                            wk_post,
                                                            cxk_post, cyk_post, czk_post,
                                                            cs2_flow,
                                                            dpi_xx_post, dpi_yy_post, dpi_zz_post,
                                                            dpi_xy_post, dpi_xz_post, dpi_yz_post
                                                        )
                                                        if df_post < 0.0f0
                                                            f_now_post =
                                                                parent_f_current[
                                                                    lx, ly, lz, pb_idx, kk
                                                                ]
                                                            pos_limiter_post = min(
                                                                pos_limiter_post,
                                                                max((f_now_post - 1.0f-10) / (-df_post), 0.0f0)
                                                            )
                                                        end
                                                    end
                                                end
                                                pos_limiter_post = min(pos_limiter_post, 1.0f0)
                                                dpi_xx_post *= pos_limiter_post
                                                dpi_yy_post *= pos_limiter_post
                                                dpi_zz_post *= pos_limiter_post
                                                dpi_xy_post *= pos_limiter_post
                                                dpi_xz_post *= pos_limiter_post
                                                dpi_yz_post *= pos_limiter_post

                                                @inbounds for kk in 1:39
                                                    if kk <= n_dirs_flow
                                                        cxk_post = Float32(cx_arr[kk]) - shift_x
                                                        cyk_post = Float32(cy_arr[kk]) - shift_y
                                                        czk_post = Float32(cz_arr[kk]) - shift_z
                                                        wk_post = Float32(w_arr[kk])
                                                        df_post = project_second_order_stress_mode(
                                                            wk_post,
                                                            cxk_post, cyk_post, czk_post,
                                                            cs2_flow,
                                                            dpi_xx_post, dpi_yy_post, dpi_zz_post,
                                                            dpi_xy_post, dpi_xz_post, dpi_yz_post
                                                        )
                                                        parent_f_current[lx, ly, lz, pb_idx, kk] =
                                                            max(
                                                                parent_f_current[
                                                                    lx, ly, lz, pb_idx, kk
                                                                ] + df_post,
                                                                1.0f-10
                                                            )
                                                    end
                                                end
                                                parent_pi_neq[lx, ly, lz, pb_idx, 1] =
                                                    pxx_post_stress + dpi_xx_post
                                                parent_pi_neq[lx, ly, lz, pb_idx, 2] =
                                                    pyy_post_stress + dpi_yy_post
                                                parent_pi_neq[lx, ly, lz, pb_idx, 3] =
                                                    pzz_post_stress + dpi_zz_post
                                                parent_pi_neq[lx, ly, lz, pb_idx, 4] =
                                                    pxy_post_stress + dpi_xy_post
                                                parent_pi_neq[lx, ly, lz, pb_idx, 5] =
                                                    pxz_post_stress + dpi_xz_post
                                                parent_pi_neq[lx, ly, lz, pb_idx, 6] =
                                                    pyz_post_stress + dpi_yz_post
                                            end
                                        end
                                    end
                                end

                                if use_paired_reflux &&
                                   (abs(child_drho_reflux) > 1.0f-12 ||
                                    abs(child_djx_reflux) > 1.0f-12 ||
                                    abs(child_dq_reflux) > 1.0f-12)

                                    child_sign_reflux = -Float32(x_sign_reflux)
                                    child_m0_reflux = 0.0f0
                                    child_m1_reflux = 0.0f0
                                    child_m2_reflux = 0.0f0
                                    child_m3_reflux = 0.0f0
                                    child_m4_reflux = 0.0f0
                                    @inbounds for kk in 1:39
                                        if kk <= n_dirs_flow
                                            child_cn_reflux =
                                                child_sign_reflux * (Float32(cx_arr[kk]) - shift_x)
                                            if child_cn_reflux > 0.0f0
                                                wk_child_reflux = Float32(w_arr[kk])
                                                child_cn2_reflux = child_cn_reflux * child_cn_reflux
                                                child_m0_reflux += wk_child_reflux
                                                child_m1_reflux += wk_child_reflux * child_cn_reflux
                                                child_m2_reflux += wk_child_reflux * child_cn2_reflux
                                                child_m3_reflux += wk_child_reflux *
                                                    child_cn2_reflux * child_cn_reflux
                                                child_m4_reflux += wk_child_reflux *
                                                    child_cn2_reflux * child_cn2_reflux
                                            end
                                        end
                                    end

                                    child_r0_reflux = child_drho_reflux
                                    child_r1_reflux = child_sign_reflux *
                                        (child_djx_reflux - shift_x * child_drho_reflux)
                                    child_r2_reflux = child_dq_reflux
                                    child_det_reflux =
                                        child_m0_reflux * (child_m2_reflux * child_m4_reflux -
                                                           child_m3_reflux * child_m3_reflux) -
                                        child_m1_reflux * (child_m1_reflux * child_m4_reflux -
                                                           child_m3_reflux * child_m2_reflux) +
                                        child_m2_reflux * (child_m1_reflux * child_m3_reflux -
                                                           child_m2_reflux * child_m2_reflux)

                                    if abs(child_det_reflux) > 1.0f-12
                                        child_a_reflux = (
                                            child_r0_reflux * (child_m2_reflux * child_m4_reflux -
                                                               child_m3_reflux * child_m3_reflux) -
                                            child_m1_reflux * (child_r1_reflux * child_m4_reflux -
                                                               child_m3_reflux * child_r2_reflux) +
                                            child_m2_reflux * (child_r1_reflux * child_m3_reflux -
                                                               child_m2_reflux * child_r2_reflux)
                                        ) / child_det_reflux
                                        child_b_reflux = (
                                            child_m0_reflux * (child_r1_reflux * child_m4_reflux -
                                                               child_m3_reflux * child_r2_reflux) -
                                            child_r0_reflux * (child_m1_reflux * child_m4_reflux -
                                                               child_m3_reflux * child_m2_reflux) +
                                            child_m2_reflux * (child_m1_reflux * child_r2_reflux -
                                                               child_r1_reflux * child_m2_reflux)
                                        ) / child_det_reflux
                                        child_c_reflux = (
                                            child_m0_reflux * (child_m2_reflux * child_r2_reflux -
                                                               child_r1_reflux * child_m3_reflux) -
                                            child_m1_reflux * (child_m1_reflux * child_r2_reflux -
                                                               child_r1_reflux * child_m2_reflux) +
                                            child_r0_reflux * (child_m1_reflux * child_m3_reflux -
                                                               child_m2_reflux * child_m2_reflux)
                                        ) / child_det_reflux

                                        child_face_fx0_reflux = x_sign_reflux > Int32(0) ?
                                            Int32(2) * gx + Int32(1) :
                                            Int32(2) * gx - Int32(2)
                                        child_face_fy0_reflux = Int32(2) * gy - Int32(1)
                                        child_face_fz0_reflux = Int32(2) * gz - Int32(1)
                                        child_pos_limiter_reflux = 1.0f0

                                        @inbounds for sy_child_reflux in Int32(0):Int32(1),
                                                       sz_child_reflux in Int32(0):Int32(1)
                                            clx_reflux, cly_reflux, clz_reflux, cb_reflux, valid_child_reflux =
                                                lookup_child_cell(
                                                    child_block_pointer,
                                                    child_dim_x,
                                                    child_dim_y,
                                                    child_dim_z,
                                                    bs,
                                                    child_face_fx0_reflux,
                                                    child_face_fy0_reflux + sy_child_reflux,
                                                    child_face_fz0_reflux + sz_child_reflux
                                                )
                                            if valid_child_reflux &&
                                               !child_obstacle[clx_reflux, cly_reflux, clz_reflux, cb_reflux]
                                                @inbounds for kk in 1:39
                                                    if kk <= n_dirs_flow
                                                        child_cn_reflux =
                                                            child_sign_reflux *
                                                            (Float32(cx_arr[kk]) - shift_x)
                                                        if child_cn_reflux > 0.0f0
                                                            wk_child_reflux = Float32(w_arr[kk])
                                                            df_child_reflux = wk_child_reflux *
                                                                (child_a_reflux +
                                                                 child_b_reflux * child_cn_reflux +
                                                                 child_c_reflux *
                                                                 child_cn_reflux * child_cn_reflux)
                                                            if df_child_reflux < 0.0f0
                                                                f_now_child_reflux =
                                                                    child_f_current[
                                                                        clx_reflux,
                                                                        cly_reflux,
                                                                        clz_reflux,
                                                                        cb_reflux,
                                                                        kk
                                                                    ]
                                                                child_pos_limiter_reflux = min(
                                                                    child_pos_limiter_reflux,
                                                                    max((f_now_child_reflux - 1.0f-10) /
                                                                        (-df_child_reflux), 0.0f0)
                                                                )
                                                            end
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                        child_pos_limiter_reflux =
                                            min(child_pos_limiter_reflux, 1.0f0)

                                        @inbounds for sy_child_reflux in Int32(0):Int32(1),
                                                       sz_child_reflux in Int32(0):Int32(1)
                                            clx_reflux, cly_reflux, clz_reflux, cb_reflux, valid_child_reflux =
                                                lookup_child_cell(
                                                    child_block_pointer,
                                                    child_dim_x,
                                                    child_dim_y,
                                                    child_dim_z,
                                                    bs,
                                                    child_face_fx0_reflux,
                                                    child_face_fy0_reflux + sy_child_reflux,
                                                    child_face_fz0_reflux + sz_child_reflux
                                                )
                                            if valid_child_reflux &&
                                               !child_obstacle[clx_reflux, cly_reflux, clz_reflux, cb_reflux]
                                                rho_child_after_reflux = 0.0f0
                                                jx_child_after_reflux = 0.0f0
                                                jy_child_after_reflux = 0.0f0
                                                jz_child_after_reflux = 0.0f0
                                                ekin_child_after_reflux = 0.0f0
                                                @inbounds for kk in 1:39
                                                    if kk <= n_dirs_flow
                                                        cx_child_reflux = Float32(cx_arr[kk])
                                                        cy_child_reflux = Float32(cy_arr[kk])
                                                        cz_child_reflux = Float32(cz_arr[kk])
                                                        child_cn_reflux =
                                                            child_sign_reflux *
                                                            (cx_child_reflux - shift_x)
                                                        fval_child_reflux =
                                                            child_f_current[
                                                                clx_reflux,
                                                                cly_reflux,
                                                                clz_reflux,
                                                                cb_reflux,
                                                                kk
                                                            ]
                                                        if child_cn_reflux > 0.0f0
                                                            wk_child_reflux = Float32(w_arr[kk])
                                                            df_child_reflux =
                                                                child_pos_limiter_reflux *
                                                                wk_child_reflux *
                                                                (child_a_reflux +
                                                                 child_b_reflux * child_cn_reflux +
                                                                 child_c_reflux *
                                                                 child_cn_reflux * child_cn_reflux)
                                                            fval_child_reflux =
                                                                max(fval_child_reflux + df_child_reflux,
                                                                    1.0f-10)
                                                            child_f_current[
                                                                clx_reflux,
                                                                cly_reflux,
                                                                clz_reflux,
                                                                cb_reflux,
                                                                kk
                                                            ] = fval_child_reflux
                                                        end
                                                        rho_child_after_reflux += fval_child_reflux
                                                        jx_child_after_reflux +=
                                                            fval_child_reflux * cx_child_reflux
                                                        jy_child_after_reflux +=
                                                            fval_child_reflux * cy_child_reflux
                                                        jz_child_after_reflux +=
                                                            fval_child_reflux * cz_child_reflux
                                                        cxu_child_reflux = cx_child_reflux - shift_x
                                                        cyu_child_reflux = cy_child_reflux - shift_y
                                                        czu_child_reflux = cz_child_reflux - shift_z
                                                        ekin_child_after_reflux +=
                                                            fval_child_reflux *
                                                            (cxu_child_reflux * cxu_child_reflux +
                                                             cyu_child_reflux * cyu_child_reflux +
                                                             czu_child_reflux * czu_child_reflux)
                                                    end
                                                end
                                                rho_child_after_reflux =
                                                    max(rho_child_after_reflux, 0.01f0)
                                                child_rho[
                                                    clx_reflux,
                                                    cly_reflux,
                                                    clz_reflux,
                                                    cb_reflux
                                                ] = rho_child_after_reflux
                                                child_vel[
                                                    clx_reflux,
                                                    cly_reflux,
                                                    clz_reflux,
                                                    cb_reflux,
                                                    1
                                                ] = jx_child_after_reflux / rho_child_after_reflux
                                                child_vel[
                                                    clx_reflux,
                                                    cly_reflux,
                                                    clz_reflux,
                                                    cb_reflux,
                                                    2
                                                ] = jy_child_after_reflux / rho_child_after_reflux
                                                child_vel[
                                                    clx_reflux,
                                                    cly_reflux,
                                                    clz_reflux,
                                                    cb_reflux,
                                                    3
                                                ] = jz_child_after_reflux / rho_child_after_reflux
                                                if restrict_thermal == Int32(1)
                                                    child_kinetic_energy[
                                                        clx_reflux,
                                                        cly_reflux,
                                                        clz_reflux,
                                                        cb_reflux
                                                    ] = ekin_child_after_reflux
                                                end
                                            end
                                        end
                                    end
                                end
                            end
                        end

                        # Refresh kinetic energy (used by Frapolli T recovery) so that
                        # downstream consumers see the post-shift moments. The shift
                        # changes Î£fÂ·cÂ² by exactly Î£(feq_newÂ·cÂ² âˆ’ feq_oldÂ·cÂ²), which
                        # we recompute here from the populations themselves to avoid
                        # accumulated floating-point drift.
                        if restrict_thermal == Int32(1)
                            ekin_post = 0.0f0
                            @inbounds for kk in 1:39
                                if kk <= n_dirs_flow
                                    cxu = Float32(cx_arr[kk]) - shift_x
                                    cyu = Float32(cy_arr[kk]) - shift_y
                                    czu = Float32(cz_arr[kk]) - shift_z
                                    fval = parent_f_current[lx, ly, lz, pb_idx, kk]
                                    ekin_post += fval * (cxu * cxu + cyu * cyu + czu * czu)
                                end
                            end
                            parent_kinetic_energy[lx, ly, lz, pb_idx] = ekin_post
                        end
                    end
                end

                if shell_equilibrium_only
                    rho_eq_shell = max(parent_rho[lx, ly, lz, pb_idx], 0.01f0)
                    ux_eq_shell = parent_vel[lx, ly, lz, pb_idx, 1]
                    uy_eq_shell = parent_vel[lx, ly, lz, pb_idx, 2]
                    uz_eq_shell = parent_vel[lx, ly, lz, pb_idx, 3]
                    temp_eq_shell = restrict_thermal == Int32(1) ?
                        clamp(max(parent_temperature[lx, ly, lz, pb_idx], 0.01f0), 0.1f0, 10.0f0) :
                        1.0f0

                    lambda_eq_shell = use_numerical_eq ?
                        solve_13mom_lambdas_d3q39(
                            rho_eq_shell,
                            ux_eq_shell - shift_x,
                            uy_eq_shell - shift_y,
                            uz_eq_shell - shift_z,
                            cx_arr, cy_arr, cz_arr, w_arr,
                            cs2_flow, Int32(50), temp_eq_shell,
                            shift_x, shift_y, shift_z,
                            100.0f0, eos_R_flow          # SOTA A3 — match collision-kernel EOS R
                        ) :
                        (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
                         0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
                    mass_corr_eq_shell = use_numerical_eq ?
                        compute_13mom_mass_correction_d3q39(
                            rho_eq_shell, lambda_eq_shell,
                            cx_arr, cy_arr, cz_arr, w_arr,
                            cs2_flow, shift_x, shift_y, shift_z
                        ) : 1.0f0

                    ekin_eq_shell = 0.0f0
                    @inbounds for kk in 1:39
                        if kk <= n_dirs_flow
                            cxu = Float32(cx_arr[kk]) - shift_x
                            cyu = Float32(cy_arr[kk]) - shift_y
                            czu = Float32(cz_arr[kk]) - shift_z
                            wk = Float32(w_arr[kk])
                            feq_final = use_numerical_eq ?
                                mass_corr_eq_shell * eval_13mom_feq_single(
                                    rho_eq_shell, lambda_eq_shell, cxu, cyu, czu, wk, cs2_flow
                                ) :
                                calculate_equilibrium_auto(
                                    rho_eq_shell, ux_eq_shell, uy_eq_shell, uz_eq_shell,
                                    wk, cxu, cyu, czu,
                                    flow_is_compressible
                                )
                            parent_f_current[lx, ly, lz, pb_idx, kk] = max(feq_final, 1.0f-10)
                            ekin_eq_shell += feq_final * (cxu * cxu + cyu * cyu + czu * czu)

                            if restrict_thermal == Int32(1) && kk <= n_dirs_thermal
                                geq_final = if use_numerical_eq && frapolli_mode == Int32(1)
                                    R_g = eos_R_flow > 0.0f0 ? eos_R_flow : cs2_thermal
                                    coeff_g = cv_thermal - 1.5f0 * R_g
                                    coeff_g * temp_eq_shell * feq_final
                                else
                                    inv_cs2_t = 1.0f0 / cs2_thermal
                                    wk * temp_eq_shell *
                                    (1.0f0 + (cxu * ux_eq_shell + cyu * uy_eq_shell + czu * uz_eq_shell) * inv_cs2_t)
                                end
                                parent_g_current[lx, ly, lz, pb_idx, kk] = max(geq_final, 1.0f-10)
                            end
                        end
                    end

                    parent_kinetic_energy[lx, ly, lz, pb_idx] = ekin_eq_shell
                    if preserve_pi_neq == Int32(1)
                        for comp in 1:6
                            parent_pi_neq[lx, ly, lz, pb_idx, comp] = 0.0f0
                        end
                    end
                end

            end
        end
    end
end

@kernel function damp_parent_interface_shell_highpass_kernel!(
    parent_rho,
    parent_rho_src,
    parent_vel,
    parent_vel_src,
    parent_f_current,
    parent_obstacle,
    parent_block_pointer,
    parent_map_x,
    parent_map_y,
    parent_map_z,
    child_block_pointer,
    cx_arr,
    cy_arr,
    cz_arr,
    w_arr,
    block_size::Int32,
    parent_dim_x::Int32,
    parent_dim_y::Int32,
    parent_dim_z::Int32,
    child_dim_x::Int32,
    child_dim_y::Int32,
    child_dim_z::Int32,
    n_dirs_flow::Int32,
    shift_x::Float32,
    shift_y::Float32,
    shift_z::Float32,
    alpha::Float32,
    rho_alpha::Float32,
    drho_cap_abs::Float32,
    drho_cap_rel::Float32,
    du_cap_abs::Float32,
    du_cap_rel::Float32,
    shell_depth::Int32,
    transverse_only::Int32
)
    lx, ly, lz, pb_idx = @index(Global, NTuple)
    bs = block_size

    gx = (parent_map_x[pb_idx] - Int32(1)) * bs + Int32(lx)
    gy = (parent_map_y[pb_idx] - Int32(1)) * bs + Int32(ly)
    gz = (parent_map_z[pb_idx] - Int32(1)) * bs + Int32(lz)

    active_cell = (alpha > 0.0f0 || rho_alpha > 0.0f0) &&
                  !parent_obstacle[lx, ly, lz, pb_idx] &&
                  count_fine_children(
                      child_block_pointer,
                      child_dim_x, child_dim_y, child_dim_z,
                      bs,
                      gx, gy, gz
                  ) == Int32(0)

    if active_cell
        near_x_face = false
        near_transverse_face = false
        @inbounds for d in Int32(1):shell_depth
            near_x_face |= coarse_cell_has_fine_children(
                child_block_pointer,
                child_dim_x, child_dim_y, child_dim_z,
                bs,
                gx - d, gy, gz
            ) || coarse_cell_has_fine_children(
                child_block_pointer,
                child_dim_x, child_dim_y, child_dim_z,
                bs,
                gx + d, gy, gz
            )
            near_transverse_face |= coarse_cell_has_fine_children(
                child_block_pointer,
                child_dim_x, child_dim_y, child_dim_z,
                bs,
                gx, gy - d, gz
            ) || coarse_cell_has_fine_children(
                child_block_pointer,
                child_dim_x, child_dim_y, child_dim_z,
                bs,
                gx, gy + d, gz
            ) || coarse_cell_has_fine_children(
                child_block_pointer,
                child_dim_x, child_dim_y, child_dim_z,
                bs,
                gx, gy, gz - d
            ) || coarse_cell_has_fine_children(
                child_block_pointer,
                child_dim_x, child_dim_y, child_dim_z,
                bs,
                gx, gy, gz + d
            )
        end

        shell_eligible = transverse_only == Int32(1) ?
            near_transverse_face :
            (near_x_face || near_transverse_face)

        if shell_eligible
            sum_rho = 0.0f0
            sum_ux = 0.0f0
            sum_uy = 0.0f0
            sum_uz = 0.0f0
            n_sample = Int32(0)

            @inbounds for dir in Int32(1):Int32(6)
                ox = dir == Int32(1) ? Int32(-1) : (dir == Int32(2) ? Int32(1) : Int32(0))
                oy = dir == Int32(3) ? Int32(-1) : (dir == Int32(4) ? Int32(1) : Int32(0))
                oz = dir == Int32(5) ? Int32(-1) : (dir == Int32(6) ? Int32(1) : Int32(0))
                ngx = gx + ox
                ngy = gy + oy
                ngz = gz + oz

                nlx, nly, nlz, nb_idx, valid_parent = lookup_parent_cell(
                    parent_block_pointer,
                    parent_dim_x, parent_dim_y, parent_dim_z,
                    bs,
                    ngx, ngy, ngz
                )
                if valid_parent &&
                   !parent_obstacle[nlx, nly, nlz, nb_idx] &&
                   count_fine_children(
                       child_block_pointer,
                       child_dim_x, child_dim_y, child_dim_z,
                       bs,
                       ngx, ngy, ngz
                   ) == Int32(0)
                    sum_rho += parent_rho_src[nlx, nly, nlz, nb_idx]
                    sum_ux += parent_vel_src[nlx, nly, nlz, nb_idx, 1]
                    sum_uy += parent_vel_src[nlx, nly, nlz, nb_idx, 2]
                    sum_uz += parent_vel_src[nlx, nly, nlz, nb_idx, 3]
                    n_sample += Int32(1)
                end
            end

            if n_sample >= Int32(3)
                rho_old = max(parent_rho_src[lx, ly, lz, pb_idx], 0.01f0)
                ux_old = parent_vel_src[lx, ly, lz, pb_idx, 1]
                uy_old = parent_vel_src[lx, ly, lz, pb_idx, 2]
                uz_old = parent_vel_src[lx, ly, lz, pb_idx, 3]
                inv_sample = 1.0f0 / Float32(n_sample)

                rho_avg = max(sum_rho * inv_sample, 0.01f0)
                drho = rho_alpha * (rho_avg - rho_old)
                drho_cap = max(drho_cap_abs, drho_cap_rel * rho_old)
                if drho_cap > 0.0f0
                    drho = clamp(drho, -drho_cap, drho_cap)
                end

                dux = alpha * (sum_ux * inv_sample - ux_old)
                duy = alpha * (sum_uy * inv_sample - uy_old)
                duz = alpha * (sum_uz * inv_sample - uz_old)

                u_scale = sqrt(ux_old*ux_old + uy_old*uy_old + uz_old*uz_old)
                du_cap = max(du_cap_abs, du_cap_rel * max(u_scale, 1.0f-8))
                du_norm = sqrt(dux*dux + duy*duy + duz*duz)
                du_limiter = (du_cap > 0.0f0 && du_norm > du_cap) ?
                    du_cap / max(du_norm, 1.0f-12) :
                    1.0f0

                ux_new = ux_old + du_limiter * dux
                uy_new = uy_old + du_limiter * duy
                uz_new = uz_old + du_limiter * duz
                rho_new = max(rho_old + drho, 0.01f0)
                drho = rho_new - rho_old

                changed = abs(drho) > 1.0f-10 ||
                          abs(ux_new - ux_old) > 1.0f-10 ||
                          abs(uy_new - uy_old) > 1.0f-10 ||
                          abs(uz_new - uz_old) > 1.0f-10

                if changed
                    m00 = 0.0f0; m01 = 0.0f0; m02 = 0.0f0; m03 = 0.0f0
                    m11 = 0.0f0; m12 = 0.0f0; m13 = 0.0f0
                    m22 = 0.0f0; m23 = 0.0f0
                    m33 = 0.0f0
                    @inbounds for kk in Int32(1):n_dirs_flow
                        cxk = Float32(cx_arr[kk]) - shift_x
                        cyk = Float32(cy_arr[kk]) - shift_y
                        czk = Float32(cz_arr[kk]) - shift_z
                        wk = Float32(w_arr[kk])
                        m00 += wk
                        m01 += wk * cxk; m02 += wk * cyk; m03 += wk * czk
                        m11 += wk * cxk * cxk; m12 += wk * cxk * cyk; m13 += wk * cxk * czk
                        m22 += wk * cyk * cyk; m23 += wk * cyk * czk
                        m33 += wk * czk * czk
                    end

                    djx = rho_new * ux_new - rho_old * ux_old
                    djy = rho_new * uy_new - rho_old * uy_old
                    djz = rho_new * uz_new - rho_old * uz_old
                    c0, c1, c2, c3, moment_ok = solve_shell_moment4(
                        m00, m01, m02, m03,
                        m11, m12, m13,
                        m22, m23,
                        m33,
                        drho, djx, djy, djz
                    )

                    if moment_ok
                        pos_limiter = 1.0f0
                        @inbounds for kk in Int32(1):n_dirs_flow
                            cxk = Float32(cx_arr[kk]) - shift_x
                            cyk = Float32(cy_arr[kk]) - shift_y
                            czk = Float32(cz_arr[kk]) - shift_z
                            wk = Float32(w_arr[kk])
                            df_k = wk * (c0 + c1 * cxk + c2 * cyk + c3 * czk)
                            if df_k < 0.0f0
                                f_now_k = parent_f_current[lx, ly, lz, pb_idx, kk]
                                pos_limiter = min(
                                    pos_limiter,
                                    max((f_now_k - 1.0f-10) / (-df_k), 0.0f0)
                                )
                            end
                        end
                        pos_limiter = min(pos_limiter, 1.0f0)

                        @inbounds for kk in Int32(1):n_dirs_flow
                            cxk = Float32(cx_arr[kk]) - shift_x
                            cyk = Float32(cy_arr[kk]) - shift_y
                            czk = Float32(cz_arr[kk]) - shift_z
                            wk = Float32(w_arr[kk])
                            df_k = pos_limiter * wk * (c0 + c1 * cxk + c2 * cyk + c3 * czk)
                            parent_f_current[lx, ly, lz, pb_idx, kk] =
                                max(parent_f_current[lx, ly, lz, pb_idx, kk] + df_k, 1.0f-10)
                        end
                    end

                    rho_after = 0.0f0
                    jx_after = 0.0f0
                    jy_after = 0.0f0
                    jz_after = 0.0f0
                    @inbounds for kk in Int32(1):n_dirs_flow
                        fval = parent_f_current[lx, ly, lz, pb_idx, kk]
                        cx_raw = Float32(cx_arr[kk])
                        cy_raw = Float32(cy_arr[kk])
                        cz_raw = Float32(cz_arr[kk])
                        rho_after += fval
                        jx_after += fval * cx_raw
                        jy_after += fval * cy_raw
                        jz_after += fval * cz_raw
                    end

                    rho_after = max(rho_after, 0.01f0)
                    parent_rho[lx, ly, lz, pb_idx] = rho_after
                    parent_vel[lx, ly, lz, pb_idx, 1] = jx_after / rho_after
                    parent_vel[lx, ly, lz, pb_idx, 2] = jy_after / rho_after
                    parent_vel[lx, ly, lz, pb_idx, 3] = jz_after / rho_after
                end
            end
        end
    end
end

function apply_amr_shell_highpass_damper!(
    parent_level::BlockLevel,
    parent_f_current,
    parent_vel_current,
    child_level::BlockLevel,
    cx_arr,
    cy_arr,
    cz_arr,
    w_arr,
    backend
)
    n_parent_blocks = length(parent_level.active_block_coords)
    if n_parent_blocks == 0 ||
       !AMR_SHELL_HIGHPASS_DAMPER ||
       (AMR_SHELL_HIGHPASS_DAMPER_ALPHA <= 0.0f0 &&
        AMR_SHELL_HIGHPASS_DAMPER_RHO_ALPHA <= 0.0f0)
        return nothing
    end

    rho_snapshot = similar(parent_level.rho)
    copyto!(rho_snapshot, parent_level.rho)
    vel_snapshot = similar(parent_vel_current)
    copyto!(vel_snapshot, parent_vel_current)
    KernelAbstractions.synchronize(backend)

    damper_kernel! = damp_parent_interface_shell_highpass_kernel!(backend)
    damper_kernel!(
        parent_level.rho,
        rho_snapshot,
        parent_vel_current,
        vel_snapshot,
        parent_f_current,
        parent_level.obstacle,
        parent_level.block_pointer,
        parent_level.map_x,
        parent_level.map_y,
        parent_level.map_z,
        child_level.block_pointer,
        cx_arr,
        cy_arr,
        cz_arr,
        w_arr,
        Int32(BLOCK_SIZE),
        Int32(size(parent_level.block_pointer, 1)),
        Int32(size(parent_level.block_pointer, 2)),
        Int32(size(parent_level.block_pointer, 3)),
        Int32(size(child_level.block_pointer, 1)),
        Int32(size(child_level.block_pointer, 2)),
        Int32(size(child_level.block_pointer, 3)),
        Int32(size(parent_f_current, 5)),
        Float32(LATTICE_SHIFT_X),
        Float32(LATTICE_SHIFT_Y),
        Float32(LATTICE_SHIFT_Z),
        Float32(AMR_SHELL_HIGHPASS_DAMPER_ALPHA),
        Float32(AMR_SHELL_HIGHPASS_DAMPER_RHO_ALPHA),
        Float32(AMR_SHELL_HIGHPASS_DAMPER_DRHO_CAP_ABS),
        Float32(AMR_SHELL_HIGHPASS_DAMPER_DRHO_CAP_REL),
        Float32(AMR_SHELL_HIGHPASS_DAMPER_CAP_ABS),
        Float32(AMR_SHELL_HIGHPASS_DAMPER_CAP_REL),
        Int32(AMR_SHELL_HIGHPASS_DAMPER_DEPTH),
        Int32(AMR_SHELL_HIGHPASS_DAMPER_TRANSVERSE_ONLY ? 1 : 0);
        ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_parent_blocks)
    )
    KernelAbstractions.synchronize(backend)
    return nothing
end

function restrict_fine_to_parent!(
    parent_level::BlockLevel,
    parent_f_current,
    parent_vel_current,
    child_level::BlockLevel,
    cx_arr,
    cy_arr,
    cz_arr,
    w_arr,
    opp_arr;
    parent_g_current=nothing,
    child_f_current=nothing,
    child_g_current=nothing,
    current_step::Int=0,
)
    # (ET-AMR / Esoteric-Twist restriction path removed 2026-06-24 with the D3Q19
    #  retirement; restriction is always conventional now.)
    child_is_et = false
    parent_is_et = false
    child_phase = Int32(0)
    parent_phase = Int32(0)
    n_parent_blocks = length(parent_level.active_block_coords)
    n_parent_blocks == 0 && return

    backend = get_backend(parent_level.rho)
    parent_g_arr = parent_g_current === nothing ? parent_level.g : parent_g_current
    child_f_arr = child_f_current === nothing ? child_level.f : child_f_current
    child_g_arr = child_g_current === nothing ? child_level.g : child_g_current

    restrict_thermal = Int32(parent_g_current !== nothing &&
                             child_g_arr !== nothing &&
                             has_thermal(parent_level) &&
                             has_thermal(child_level))
    reset_pi_neq = Int32(has_pi_neq(parent_level))
    preserve_pi_neq = Int32(has_pi_neq(parent_level) && has_pi_neq(child_level))
    # PHASE 2b (2026-06-12): Val-typed so the restriction kernels constant-
    # fold the dead lattice's code paths away (register diet; see the kernel
    # signatures). Both kernel calls below pass these variables positionally.
    lattice_is_d3q39 = Val(LATTICE_TYPE == :d3q39)
    flow_is_compressible = Int32(COMPRESSIBLE_MODE ? 1 : 0)
    frapolli_mode = Val(DDF_ENABLED && LATTICE_TYPE == :d3q39)
    mirror_decompose_entropic = Val(COLLISION_OPERATOR == :entropic)
    # For the restriction, match the coarseâ†’fine convention: drop h at the
    # interface. The existing INTERFACE_ENTROPIC_H_SCALE parameter controls
    # the symmetric prolongation; mirroring its default here keeps round-trip
    # symmetry and avoids the h-checkerboard mode on both sides.
    h_scale_restrict_val = Float32(INTERFACE_ENTROPIC_H_SCALE)
    level_divisor = 2 ^ max(parent_level.level_id - 1, 0)
    parent_top_step = current_step > 0 ? max(1, fld(current_step, level_divisor)) : 0

    # Ï„-rescaling for fineâ†’coarse f_neq. Matches the convention used by the
    # coarseâ†’fine prolongation in physics_interpolation.jl (raw Ï„ ratio,
    # not (Ï„-0.5)/(Ï„-0.5)). Inverse direction here: scale = Ï„_c / Ï„_f > 1.
    # Live convention: match coarseâ†’fine interpolation by scaling the
    # non-equilibrium carrier with the Chapman-Enskog factor (Ï„ - 0.5).
    tau_parent_ce = max(parent_level.tau - 0.5f0, 1.0f-6)
    tau_child_ce = max(child_level.tau - 0.5f0, 1.0f-6)
    tau_parent_g_ce = max(parent_level.tau_g - 0.5f0, 1.0f-6)
    tau_child_g_ce = max(child_level.tau_g - 0.5f0, 1.0f-6)
    tau_ratio_flow_val = Float32(clamp(tau_parent_ce / tau_child_ce, 0.01f0, 100.0f0))
    if LEVEL_VISCOSITY_MODE === :physical
        # advanced.refinement.level_viscosity: "physical" (lbm-taufix, 28 Oct 2026): the pre-collision non-equilibrium
        # scales as tau·dt·S_phys, so parent/child = (tau_p dt_p)/(tau_c dt_c) = 2 tau_p/tau_c (dt_p = 2 dt_c). The
        # lattice-viscosity ratio above equals 2 only under the legacy level rule. The post-collision correction
        # below applies on top, as before.
        tau_ratio_flow_val = Float32(clamp(2.0f0 * Float32(parent_level.tau) / max(Float32(child_level.tau), 0.500001f0),
                                           0.01f0, 100.0f0))
    end
    # AMR-SEAM FIX 1A (2026-06-16): the stored child f_neq is POST-collision (carries
    # an extra (1−ω_child) factor). The (τ_c−0.5)/(τ_f−0.5) ratio alone is the
    # PRE-collision rescaling; transferring post→post requires the additional
    # (1−ω_parent)/(1−ω_child) correction (Lagrava 2012 §3.5 / Rohde 2006). Without
    # it the parent stress is over-injected coherently every coarse step — the
    # AMR box-edge seam, worst at the τ-clamped finest interface where (1−ω)
    # diverges fastest. Freestream-byte-identical (f_neq≡0 ⇒ scale irrelevant).
    # Gated behind LUDWIG_D3Q19_AMR_FNEQ_POSTCOLL (default ON for D3Q19 BGK; set 0
    # to revert to the legacy uncorrected ratio for an A/B).
    if _AMR_FNEQ_POSTCOLL && (LATTICE_TYPE == :d3q19 || LATTICE_TYPE == :d3q27)
        omega_p = 1.0f0 / max(Float32(parent_level.tau), 0.500001f0)
        omega_c = 1.0f0 / max(Float32(child_level.tau), 0.500001f0)
        one_m_wp = 1.0f0 - omega_p
        one_m_wc = 1.0f0 - omega_c
        if abs(one_m_wc) > 1.0f-6
            postcoll_corr = one_m_wp / one_m_wc
            tau_ratio_flow_val = Float32(clamp(tau_ratio_flow_val * postcoll_corr, 0.01f0, 100.0f0))
        end
    end
    tau_ratio_thermal_val = AMR_THERMAL_NONEQ_TRANSFER ?
        Float32(clamp(tau_parent_g_ce / tau_child_g_ce, 0.01f0, 100.0f0)) :
        0.0f0

    # 100-MLUPS item 2 (2026-06-12): banded launch over covered parent blocks
    # only (static topology, cached). Membership is exact — see the cache
    # helper at the top of this file. Falls back to dense when the env flag
    # is off or the list isn't smaller than the dense range.
    _restrict_band = nothing
    _restrict_ndrange4 = Int(n_parent_blocks)
    if _D3Q19_RESTRICT_COVERED_BAND
        _band_entry = _restrict_covered_block_list(parent_level, child_level, backend)
        # n==0 cannot occur for a real child level; fall back to the dense
        # launch in that case (a no-op pass) rather than skipping the
        # function's trailing work (refresh_pi_neq! etc.).
        if 0 < _band_entry.n < _band_entry.n_total
            _restrict_band = _band_entry.list
            _restrict_ndrange4 = _band_entry.n
        end
    end

    if _PHASE_TIMING; KernelAbstractions.synchronize(backend); _tr0 = time(); end
    kernel! = restrict_child_state_to_parent_kernel!(backend)
    kernel!(
        parent_level.rho,
        parent_vel_current,
        parent_level.temperature,
        parent_level.kinetic_energy,
        parent_f_current,
        parent_g_arr,
        parent_level.pi_neq,
        parent_level.map_x,
        parent_level.map_y,
        parent_level.map_z,
        child_level.block_pointer,
        child_level.obstacle,
        child_level.wall_dist,
        child_level.rho,
        child_level.vel,
        child_level.temperature,
        child_level.kinetic_energy,
        child_f_arr,
        child_g_arr,
        child_level.pi_neq,
        cx_arr,
        cy_arr,
        cz_arr,
        w_arr,
        opp_arr,
        Int32(BLOCK_SIZE),
        Int32(size(child_level.block_pointer, 1)),
        Int32(size(child_level.block_pointer, 2)),
        Int32(size(child_level.block_pointer, 3)),
        Int32(size(parent_f_current, 5)),
        Int32(size(parent_g_arr, 5)),
        restrict_thermal,
        reset_pi_neq,
        preserve_pi_neq,
        lattice_is_d3q39,
        flow_is_compressible,
        frapolli_mode,
        mirror_decompose_entropic,
        Float32(CS2_PHYSICS),
        # SOTA A3 — eos_R_flow: same expression as the collision kernel
        # (timestep.jl) so collision, restriction, and prolongation all use the
        # identical EOS. Sentinel -1 ⇒ cs² fallback (byte-identical legacy).
        EOS_THERMODYNAMIC ? Float32((DDF_GAMMA - 1.0) * DDF_CV) : Float32(-1.0f0),
        Float32(CS2_PHYSICS),
        Float32(DDF_CV),
        Float32(LATTICE_SHIFT_X),
        Float32(LATTICE_SHIFT_Y),
        Float32(LATTICE_SHIFT_Z),
        tau_ratio_flow_val,
        tau_ratio_thermal_val,
        h_scale_restrict_val,
        # Pass 265 (Tier 1 #1e): moment-correction pass count, replaces hardcoded 2.
        AMR_RESTRICTION_CORRECTION_PASSES,
        # Pass 265 (Tier 1 #1e, complementary): conservative f_eq flag.
        AMR_RESTRICTION_CONSERVATIVE_FEQ ? Int32(1) : Int32(0),
        # SOTA P1 (2026-05-27): conservative ρE — extends the 4-moment f_eq
        # correction to 5 moments (mass + 3·momentum + kinetic energy).
        AMR_RESTRICTION_CONSERVATIVE_RHOE ? Int32(1) : Int32(0),
        # B1 Pillar 2 (2026-05-27): 13-moment-exact restriction. Replaces
        # the f_eq + stress-carrier construction with a closed-form solve
        # that drives ALL 13 Hermite moments (mass + momentum + stress +
        # energy flux) of the parent populations to round-off exactness.
        # Strict-conservation upgrade over P1 (which only covers 5).
        AMR_RESTRICTION_CONSERVATIVE_13MOM ? Int32(1) : Int32(0),
        # B3 LIVE (2026-05-28): restriction-side FP64 halo arithmetic.
        # When 1, the 8-child accumulation + 39-direction moment sum run
        # in Float64 internally (storage stays FP32, zero VRAM cost).
        # Symmetric companion to the prolongation-side FP64 promotion in
        # `interpolate_macro_quartic_5pt_fp64`. Closes the round-trip
        # coarse↔fine AMR cycle at FP64 arithmetic precision.
        INTERFACE_FP64_HALO ? Int32(1) : Int32(0),
        # SYMMETRY-PLANE FIX (2026-06-04, FIX 1): host-side gate (module consts,
        # read here in the driver — NOT inside the kernel → GPU-safe). When the
        # half-domain mirror plane is active, the kernel zeroes parent u_y at plane
        # cells so the restriction can't re-inject the runaway asymmetry. Off ⇒
        # byte-identical (dead branch in the kernel).
        (SYMMETRIC_ANALYSIS && !GROUND_ANALYSIS) ? Int32(1) : Int32(0),
        Int32(size(parent_level.block_pointer, 2) * BLOCK_SIZE),
        # GOLD-STANDARD STRICT CONSERVATION (2026-06-17): D3Q19 4-moment exact
        # restriction. Host gate — D3Q19 only (the D3Q39 path has its own
        # 13-mom-exact restriction) AND the flag on. Off ⇒ byte-identical.
        # 2026-06-23: fires for D3Q19 (its flag) OR D3Q27 cumulant (the new flag).
        # Both use the polynomial-eq reconstruction + the n_dirs-generic 4-moment
        # Gram correction in the kernel body. D3Q39 has its own 13-mom-exact path.
        ((LATTICE_TYPE == :d3q19 && AMR_RESTRICTION_CONSERVATIVE_D3Q19) ||
         (LATTICE_TYPE == :d3q27 && AMR_RESTRICTION_CONSERVATIVE_D3Q27)) ? Int32(1) : Int32(0),
        _restrict_band,
        # ET-AMR Val-gated decode/encode slots — ET retired 2026-06-24, always Val(false)
        # so the kernel's ET branches const-fold away; pair arrays = opp_arr stand-in.
        Val(child_is_et),
        Val(parent_is_et),
        child_phase,
        parent_phase,
        opp_arr,
        opp_arr,
        parent_level.block_pointer,
        Int32(size(parent_level.block_pointer, 1)),
        Int32(size(parent_level.block_pointer, 2)),
        Int32(size(parent_level.block_pointer, 3));
        ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, _restrict_ndrange4)
    )
    # PERF (2026-06-04): removed host synchronize — the next op is another GPU
    # kernel (partial_kernel! / subsequent level work) on the same backend, which
    # KernelAbstractions auto-orders. No host read between. Confirmed removable by
    # the sync-classification workflow (classifier+verifier agreed).

    # ★ ET-AMR (2026-06-18): the partial-frontier / shell-sync / damper kernels below
    # are NOT ET-aware — they read child_f_arr AS conventional populations, but under ET
    # that buffer is the PERMUTED Esoteric layout (garbage when read directly), and they
    # write into parent_f_current (the decoded f_temp) at the box-frontier cells. That was
    # the dominant box-edge ring + interior speckle (Z265/Z267). The main restriction
    # kernel above already handles ALL covered parents incl. the partial-coverage frontier
    # (the frac<1 blend), so for ET we SKIP these secondary refinements rather than corrupt
    # the frontier. (Non-ET path unchanged.)
    if AMR_PARTIAL_FRONTIER_SYNC && !child_is_et
        partial_kernel! = restrict_partial_child_state_to_parent_kernel!(backend)
        partial_kernel!(
            parent_level.rho,
            parent_vel_current,
            parent_level.temperature,
            parent_level.kinetic_energy,
            parent_f_current,
            parent_g_arr,
            parent_level.pi_neq,
            parent_level.obstacle,
            parent_level.map_x,
            parent_level.map_y,
            parent_level.map_z,
            child_level.block_pointer,
            child_level.obstacle,
            child_level.temperature,
            child_level.kinetic_energy,
            child_f_arr,
            child_g_arr,
            child_level.pi_neq,
            cx_arr,
            cy_arr,
            cz_arr,
            w_arr,
            Int32(BLOCK_SIZE),
            Int32(size(child_level.block_pointer, 1)),
            Int32(size(child_level.block_pointer, 2)),
            Int32(size(child_level.block_pointer, 3)),
            Int32(size(parent_f_current, 5)),
            Int32(size(parent_g_arr, 5)),
            restrict_thermal,
            reset_pi_neq,
            preserve_pi_neq,
            lattice_is_d3q39,
            flow_is_compressible,
            frapolli_mode,
            mirror_decompose_entropic,
            Float32(CS2_PHYSICS),
            # SOTA A3 — eos_R_flow: same expression as the collision kernel
            # (timestep.jl) so collision, restriction, and prolongation all use
            # the identical EOS. Sentinel -1 ⇒ cs² fallback (byte-identical).
            EOS_THERMODYNAMIC ? Float32((DDF_GAMMA - 1.0) * DDF_CV) : Float32(-1.0f0),
            Float32(CS2_PHYSICS),
            Float32(DDF_CV),
            Float32(LATTICE_SHIFT_X),
            Float32(LATTICE_SHIFT_Y),
            Float32(LATTICE_SHIFT_Z),
            tau_ratio_flow_val,
            tau_ratio_thermal_val;
            ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_parent_blocks)
        )
        KernelAbstractions.synchronize(backend)
    end

    if AMR_INTERFACE_SHELL_SYNC && !child_is_et   # ET: not ET-aware, skip (see note above)
        shell_restrict_thermal = Int32(
            restrict_thermal == Int32(1) && DDF_FEEDBACK_TO_FLOW ? 1 : 0
        )
        shell_kernel! = synchronize_parent_interface_shell_kernel!(backend)
        shell_kernel!(
            parent_level.rho,
            parent_vel_current,
            parent_level.rho_old,
            parent_level.vel_old,
            parent_level.temperature,
            parent_level.kinetic_energy,
            parent_f_current,
            parent_g_arr,
            parent_level.pi_neq,
            parent_level.amr_face_budget,
            parent_level.obstacle,
            parent_level.amr_covered,
            parent_level.block_pointer,
            parent_level.map_x,
            parent_level.map_y,
            parent_level.map_z,
            child_level.block_pointer,
            child_level.obstacle,
            child_level.rho,
            child_level.vel,
            child_level.rho_old,
            child_level.vel_old,
            child_level.temperature,
            child_level.kinetic_energy,
            child_f_arr,
            child_g_arr,
            child_level.pi_neq,
            cx_arr,
            cy_arr,
            cz_arr,
            w_arr,
            opp_arr,
            Int32(BLOCK_SIZE),
            Int32(size(parent_level.block_pointer, 1)),
            Int32(size(parent_level.block_pointer, 2)),
            Int32(size(parent_level.block_pointer, 3)),
            Int32(size(child_level.block_pointer, 1)),
            Int32(size(child_level.block_pointer, 2)),
            Int32(size(child_level.block_pointer, 3)),
            Int32(size(parent_f_current, 5)),
            Int32(size(parent_g_arr, 5)),
            shell_restrict_thermal,
            preserve_pi_neq,
            mirror_decompose_entropic,
            flow_is_compressible,
            frapolli_mode,
            Float32(CS2_PHYSICS),
            # SOTA A3 — eos_R_flow: same expression as the collision kernel
            # (timestep.jl) so collision, restriction, and prolongation all use
            # the identical EOS. Sentinel -1 ⇒ cs² fallback (byte-identical).
            EOS_THERMODYNAMIC ? Float32((DDF_GAMMA - 1.0) * DDF_CV) : Float32(-1.0f0),
            Float32(LATTICE_TYPE == :d3q39 ? 2.0 / 3.0 : 1.0 / 3.0),
            Float32(DDF_CV),
            Float32(LATTICE_SHIFT_X),
            Float32(LATTICE_SHIFT_Y),
            Float32(LATTICE_SHIFT_Z),
            tau_ratio_flow_val,
            tau_ratio_thermal_val,
            h_scale_restrict_val,
            Int32(AMR_SHELL_EQ_SHIFT ? 1 : 0),
            Int32(AMR_SHELL_MOMENT_SHIFT ? 1 : 0),
            Int32(AMR_SHELL_SYNC_INCLUDE_X ? 1 : 0),
            Float32(AMR_SHELL_POPULATION_BLEND_X),
            Float32(AMR_SHELL_RHO_ALPHA_X_SINGLE),
            Float32(AMR_SHELL_RHO_ALPHA_X_MULTI),
            Float32(AMR_SHELL_RHO_ALPHA_T_SINGLE),
            Float32(AMR_SHELL_RHO_ALPHA_T_MULTI),
            Float32(AMR_SHELL_VEL_ALPHA_X_SINGLE),
            Float32(AMR_SHELL_VEL_ALPHA_X_MULTI),
            Float32(AMR_SHELL_VEL_ALPHA_T),
            Int32(AMR_SHELL_VEL_NORMAL_ONLY ? 1 : 0),
            Float32(AMR_SHELL_DRHO_CAP_X_ABS),
            Float32(AMR_SHELL_DRHO_CAP_X_REL),
            Float32(AMR_SHELL_DRHO_CAP_T_ABS),
            Float32(AMR_SHELL_DRHO_CAP_T_REL),
            Float32(AMR_SHELL_DU_CAP_X_ABS),
            Float32(AMR_SHELL_DU_CAP_X_REL),
            Float32(AMR_SHELL_DU_CAP_T_ABS),
            Float32(AMR_SHELL_DU_CAP_T_REL),
            Float32(AMR_SHELL_FLUX_ALPHA_X),
            Float32(AMR_SHELL_FLUX_ALPHA_T),
            Float32(AMR_SHELL_FLUX_DU_CAP_ABS),
            Float32(AMR_SHELL_FLUX_DU_CAP_REL),
            Float32(AMR_SHELL_MOMFLUX_ALPHA_X),
            Float32(AMR_SHELL_MOMFLUX_DRHO_CAP_ABS),
            Float32(AMR_SHELL_MOMFLUX_DRHO_CAP_REL),
            Float32(AMR_SHELL_MOMFLUX_STRESS_ALPHA_X),
            Float32(AMR_SHELL_MOMFLUX_STRESS_CAP_ABS),
            Float32(AMR_SHELL_MOMFLUX_STRESS_CAP_REL),
            Float32(AMR_SHELL_FACE_WEIGHT_X),
            Int32(AMR_SHELL_FACE_POP_SHIFT ? 1 : 0),
            Float32(AMR_SHELL_FACE_POP_ALPHA),
            Int32(AMR_SHELL_FACE_POP_SIDE),
            Int32(AMR_SHELL_FACE_STRESS_SHIFT ? 1 : 0),
            Float32(AMR_SHELL_FACE_STRESS_ALPHA),
            Float32(AMR_SHELL_FACE_STRESS_REL_CAP),
            Float32(AMR_SHELL_FACE_STRESS_ABS_CAP),
            Int32((length(parent_level.rho_old) > 1 && length(child_level.rho_old) > 1) ? 1 : 0),
            Int32(AMR_SHELL_COUPLED_REFLUX ? 1 : 0),
            Float32(AMR_SHELL_COUPLED_REFLUX_ALPHA),
            Int32(AMR_SHELL_PAIRED_REFLUX ? 1 : 0),
            Float32(AMR_SHELL_PAIRED_REFLUX_ALPHA),
            Float32(AMR_SHELL_PAIRED_REFLUX_MASS_ALPHA),
            Float32(AMR_SHELL_PAIRED_REFLUX_MOM_ALPHA),
            Float32(AMR_SHELL_PAIRED_REFLUX_MOM_ALPHA_INITIAL),
            Int32(AMR_SHELL_PAIRED_REFLUX_MOM_ALPHA_HOLD_STEPS),
            Float32(AMR_SHELL_PAIRED_REFLUX_PARENT_FRACTION),
            Float32(AMR_SHELL_PAIRED_REFLUX_MASS_PARENT_FRACTION),
            Float32(AMR_SHELL_PAIRED_REFLUX_MASS_CAP_SCALE),
            Float32(AMR_SHELL_PAIRED_REFLUX_MASS_CAP_INITIAL),
            Int32(AMR_SHELL_PAIRED_REFLUX_MASS_CAP_HOLD_STEPS),
            Int32(AMR_SHELL_PAIRED_REFLUX_FACE_CONSERVATIVE ? 1 : 0),
            Int32(AMR_SHELL_PAIRED_REFLUX_COUPLED_CAPS ? 1 : 0),
            Int32(AMR_SHELL_PAIRED_REFLUX_COUPLED_MASS_CAPS ? 1 : 0),
            Int32(AMR_SHELL_PAIRED_REFLUX_MACRO_CONSISTENT_CAPS ? 1 : 0),
            Int32(AMR_SHELL_PAIRED_REFLUX_MACRO_PAIRWISE_CAPS ? 1 : 0),
            Int32(AMR_SHELL_PAIRED_REFLUX_BUDGET_MEAN_PRESERVING ? 1 : 0),
            Int32(AMR_SHELL_PAIRED_REFLUX_SHARED_STATE ? 1 : 0),
            Float32(AMR_SHELL_PAIRED_REFLUX_SHARED_STATE_RHO_ALPHA),
            Int32(AMR_SHELL_PAIRED_REFLUX_HIGHPASS ? 1 : 0),
            Float32(AMR_SHELL_PAIRED_REFLUX_HIGHPASS_BLEND),
            Int32(AMR_SHELL_PAIRED_REFLUX_LOWPASS ? 1 : 0),
            Float32(AMR_SHELL_PAIRED_REFLUX_LOWPASS_BLEND),
            Int32(AMR_SHELL_PAIRED_REFLUX_SECOND_MOMENT ? 1 : 0),
            Float32(AMR_SHELL_PAIRED_REFLUX_SECOND_MOMENT_BLEND),
            Int32(AMR_SHELL_PAIRED_REFLUX_POST_STRESS ? 1 : 0),
            Float32(AMR_SHELL_PAIRED_REFLUX_POST_STRESS_ALPHA),
            Int32(AMR_SHELL_PAIRED_REFLUX_POST_STRESS_SOFT_LIMIT ? 1 : 0),
            Int32((AMR_GPU_FACE_BUDGET_REFLUX &&
                   parent_level.level_id <= AMR_GPU_FACE_BUDGET_REFLUX_MAX_PARENT_LEVEL) ? 1 : 0),
            Float32(AMR_GPU_FACE_BUDGET_REFLUX_ALPHA),
            Float32(AMR_GPU_FACE_BUDGET_REFLUX_MASS_ALPHA),
            Float32(AMR_GPU_FACE_BUDGET_REFLUX_MOM_ALPHA),
            Float32(AMR_GPU_FACE_BUDGET_REFLUX_DRHO_CAP_ABS),
            Float32(AMR_GPU_FACE_BUDGET_REFLUX_DRHO_CAP_REL),
            Float32(AMR_GPU_FACE_BUDGET_REFLUX_MOM_CAP_ABS),
            Float32(AMR_GPU_FACE_BUDGET_REFLUX_MOM_CAP_REL),
            Int32(AMR_GPU_FACE_BUDGET_REFLUX_NEUTRAL_MOM ? 1 : 0),
            Int32(parent_top_step),
            Int32(AMR_SHELL_RECALIBRATION ? 1 : 0),
            Int32(AMR_SHELL_WELL_BALANCED_REBUILD ? 1 : 0);
            ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_parent_blocks)
        )
        KernelAbstractions.synchronize(backend)
    end

    apply_amr_shell_highpass_damper!(
        parent_level,
        parent_f_current,
        parent_vel_current,
        child_level,
        cx_arr,
        cy_arr,
        cz_arr,
        w_arr,
        backend
    )

    if AMR_POST_RESTRICT_SPONGE
        # This is a solver-specific stabilizing patch, not part of the
        # reference fineâ†’coarse coupling operators. Keep it configurable so
        # AMR debugging can isolate the core restriction/prolongation pair.
        _post_restrict_sponge_enforce!(parent_level, parent_f_current, parent_vel_current,
                                       cx_arr, cy_arr, cz_arr, w_arr, backend;
                                       parent_g_current=parent_g_arr,
                                       restrict_thermal=(restrict_thermal == Int32(1)))
    end

    if has_pi_neq(parent_level)
        # 100-MLUPS item 2b (2026-06-12): the restriction only changed COVERED
        # parent cells, so the refresh recompute at uncovered cells is a
        # value-identical no-op (same f ⇒ same Π), and the restrict-side
        # scratch write at uncovered cells is dead before the next post-step
        # refresh overwrites it. Adjudicated by the bitwise seam gates.
        refresh_pi_neq!(parent_level, parent_f_current, cx_arr, cy_arr, cz_arr, w_arr, backend;
                        shift_x=Float32(LATTICE_SHIFT_X),
                        shift_y=Float32(LATTICE_SHIFT_Y),
                        shift_z=Float32(LATTICE_SHIFT_Z),
                        covered_band=_restrict_band)
    end

    if _PHASE_TIMING; KernelAbstractions.synchronize(backend); _phase_op_add!(_T_OP_RESTRICT, time() - _tr0); end
    return nothing
end

# â”€â”€â”€ Post-restriction sponge enforcement kernel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# For parent cells inside the sponge zone (sponge > threshold),
# blend the restricted populations toward freestream equilibrium
# proportionally to sponge strength.  This prevents the restriction
# from injecting stale non-equilibrium into the sponge zone, which
# otherwise causes banding at AMR interfaces near the outlet.
@kernel function _post_restrict_sponge_kernel!(
    parent_rho,
    parent_vel,
    parent_temperature,
    parent_f,
    parent_g,
    parent_sponge,
    parent_map_x,
    parent_map_y,
    parent_map_z,
    cx_arr,
    cy_arr,
    cz_arr,
    w_arr,
    block_size::Int32,
    n_dirs_flow::Int32,
    n_dirs_thermal::Int32,
    u_inlet::Float32,
    t_inlet::Float32,
    do_thermal::Int32,
    cs2_val::Float32,
    shift_x::Float32,
    shift_y::Float32,
    shift_z::Float32
)
    lx, ly, lz, pb_idx = @index(Global, NTuple)

    sp = parent_sponge[lx, ly, lz, pb_idx]

    if sp >= 0.05f0
        # Freestream target
        rho_tgt = 1.0f0
        ux_tgt = u_inlet
        uy_tgt = 0.0f0
        uz_tgt = 0.0f0

        # Read current restricted macroscopic
        rho_cur = parent_rho[lx, ly, lz, pb_idx]
        ux_cur = parent_vel[lx, ly, lz, pb_idx, 1]
        uy_cur = parent_vel[lx, ly, lz, pb_idx, 2]
        uz_cur = parent_vel[lx, ly, lz, pb_idx, 3]

        # Blend macroscopic toward freestream at sponge rate
        rho_new = rho_cur * (1.0f0 - sp) + rho_tgt * sp
        ux_new = ux_cur * (1.0f0 - sp) + ux_tgt * sp
        uy_new = uy_cur * (1.0f0 - sp)
        uz_new = uz_cur * (1.0f0 - sp)

        parent_rho[lx, ly, lz, pb_idx] = rho_new
        parent_vel[lx, ly, lz, pb_idx, 1] = ux_new
        parent_vel[lx, ly, lz, pb_idx, 2] = uy_new
        parent_vel[lx, ly, lz, pb_idx, 3] = uz_new

        # Rebuild distributions from blended macroscopic (equilibrium)
        inv_cs2 = 1.0f0 / cs2_val
        @inbounds for k in Int32(1):n_dirs_flow
            cx_k = Float32(cx_arr[k]) - shift_x
            cy_k = Float32(cy_arr[k]) - shift_y
            cz_k = Float32(cz_arr[k]) - shift_z
            wk = Float32(w_arr[k])
            c_dot_u = cx_k * ux_new + cy_k * uy_new + cz_k * uz_new
            u2 = ux_new * ux_new + uy_new * uy_new + uz_new * uz_new
            feq = rho_new * wk * (1.0f0 + c_dot_u * inv_cs2 +
                  0.5f0 * (c_dot_u * c_dot_u) * inv_cs2 * inv_cs2 -
                  0.5f0 * u2 * inv_cs2)
            feq = max(feq, 1.0f-10)

            # Blend: inside sponge, push f toward equilibrium at target state
            f_cur = parent_f[lx, ly, lz, pb_idx, k]
            parent_f[lx, ly, lz, pb_idx, k] = f_cur * (1.0f0 - sp) + feq * sp
        end

        # Thermal DDF: same sponge enforcement
        if do_thermal == Int32(1)
            t_cur = parent_temperature[lx, ly, lz, pb_idx]
            t_new = t_cur * (1.0f0 - sp) + t_inlet * sp
            parent_temperature[lx, ly, lz, pb_idx] = t_new
            @inbounds for k in Int32(1):n_dirs_thermal
                cx_k = Float32(cx_arr[k]) - shift_x
                cy_k = Float32(cy_arr[k]) - shift_y
                cz_k = Float32(cz_arr[k]) - shift_z
                wk = Float32(w_arr[k])
                c_dot_u = cx_k * ux_new + cy_k * uy_new + cz_k * uz_new
                geq = wk * t_new * (1.0f0 + c_dot_u * inv_cs2)
                geq = max(geq, 1.0f-10)
                g_cur = parent_g[lx, ly, lz, pb_idx, k]
                parent_g[lx, ly, lz, pb_idx, k] = g_cur * (1.0f0 - sp) + geq * sp
            end
        end
    end
end

function _post_restrict_sponge_enforce!(parent_level, parent_f_current, parent_vel_current,
                                        cx_arr, cy_arr, cz_arr, w_arr, backend;
                                        parent_g_current=nothing, restrict_thermal::Bool=false)
    n_parent_blocks = length(parent_level.active_block_coords)
    n_parent_blocks == 0 && return

    parent_g_arr = parent_g_current !== nothing ? parent_g_current : parent_level.g
    n_dirs_flow = Int32(size(parent_f_current, 5))
    n_dirs_thermal = Int32(size(parent_g_arr, 5))
    do_thermal = Int32(restrict_thermal && has_thermal(parent_level) ? 1 : 0)

    kernel! = _post_restrict_sponge_kernel!(backend)
    kernel!(
        parent_level.rho,
        parent_vel_current,
        parent_level.temperature,
        parent_f_current,
        parent_g_arr,
        parent_level.sponge,
        parent_level.map_x,
        parent_level.map_y,
        parent_level.map_z,
        cx_arr,
        cy_arr,
        cz_arr,
        w_arr,
        Int32(BLOCK_SIZE),
        n_dirs_flow,
        n_dirs_thermal,
        U_INLET_CURRENT[],  # Ramped inlet velocity â€” respects startup ramp
        Float32(DDF_T_INLET),
        do_thermal,
        Float32(CS2_PHYSICS),
        Float32(LATTICE_SHIFT_X),
        Float32(LATTICE_SHIFT_Y),
        Float32(LATTICE_SHIFT_Z);
        ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_parent_blocks)
    )
    KernelAbstractions.synchronize(backend)
end
