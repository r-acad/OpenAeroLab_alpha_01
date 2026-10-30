"""
    compute_feather_weights!(feather_weight, block_pointer, feather_depth; x_only=false)

Populate a per-cell feather weight in [0, 1] over a `feather_depth`-cell-thick
band adjacent to each face of a block whose neighbor block is inactive
(i.e., outside the fine level's active region). Weight is close to 1.0 at
the boundary face and decays with a smoothstep profile inward.
The collision kernel uses this as a multiplier on an extra viscous
dissipation contribution that damps the O(1e-4) density seam the
prolongation-streaming mix injects at the halo, preventing it from
propagating into the flow region of interest.

When `x_only=true`, only the streamwise `:xm` / `:xp` faces are marked.
This targets the persistent x-normal AMR imprint without adding dissipation
on the already-smooth transverse interfaces.

`block_pointer` is the (bx_max, by_max, bz_max) array where >0 means
active. This runs on CPU at setup (one pass per level) — no hot path.
"""
# ── Per-block Chebyshev (L∞) distance feather (CORNER-QUAD FIX 2026-06-10) ────
# The legacy feather measured distance to the AMR boundary FACE-wise only:
# fine-side ramps were applied per block face with a missing neighbor, and the
# coarse-side companion probed for refined cells along the 6 AXIAL directions.
# Both miss DIAGONAL adjacency: a cell just outside (or just inside) a box
# CORNER/EDGE found no boundary along any single axis/face and kept weight 0,
# punching a ~feather_depth-sized notch of missing dissipation into the feather
# band at every refinement-box corner — visible for years as the "isolated quad
# + stepped interface line" artifact at box corners in density/velocity slices
# (one per corner, every nesting level; coarse side via the coarse_side scale,
# fine side at the inside corners of the non-convex L4+ shells).
#
# Fix: per-cell smoothstep over the CHEBYSHEV distance to the "other side"
# (fine side: the inactive region; coarse side: the child-supported region).
# On flat faces this reproduces the legacy values exactly (axial == Chebyshev
# there); at corners/edges it wraps the band around continuously. Computed per
# active block on a (BLOCK_SIZE + 2·depth)³ local window via iterative ±1
# dilation (depth rounds × 3 separable axis passes) — exact within the window
# because any source farther than `depth` gets weight 0 anyway.

"One ±1 Chebyshev dilation round of `src` into `dst` (both n³ windows). x_only dilates along x."
function _feather_dilate_round!(dst::Array{Bool, 3}, tmp::Array{Bool, 3},
                                src::Array{Bool, 3}; x_only::Bool=false)
    n1, n2, n3 = size(src)
    @inbounds for z in 1:n3, y in 1:n2, x in 1:n1
        tmp[x, y, z] = src[x, y, z] ||
                       (x > 1 && src[x-1, y, z]) || (x < n1 && src[x+1, y, z])
    end
    if x_only
        copyto!(dst, tmp)
        return nothing
    end
    @inbounds for z in 1:n3, y in 1:n2, x in 1:n1
        dst[x, y, z] = tmp[x, y, z] ||
                       (y > 1 && tmp[x, y-1, z]) || (y < n2 && tmp[x, y+1, z])
    end
    @inbounds for z in 1:n3, y in 1:n2, x in 1:n1
        tmp[x, y, z] = dst[x, y, z] ||
                       (z > 1 && dst[x, y, z-1]) || (z < n3 && dst[x, y, z+1])
    end
    copyto!(dst, tmp)
    return nothing
end

# Apply smoothstep(Chebyshev distance − 1) weights to one block from a window
# source mask. `src` covers global cells (g0 .+ (1:n)) per axis with the block's
# cells at window offset `depth`. Round r of the dilation reaches exactly the
# cells at Chebyshev distance ≤ r, so a cell first reached at round r gets
# w = scale·smoothstep(1 − (r−1)/depth) — identical to the legacy face/axial
# ramp on flat boundaries. `accumulate` keeps existing weights via max().
function _feather_apply_window!(feather_weight::AbstractArray{Float32, 4},
                                b_idx::Int, src::Array{Bool, 3},
                                buf_a::Array{Bool, 3}, buf_b::Array{Bool, 3},
                                feather_depth::Int, scale::Float32;
                                x_only::Bool=false)
    bs = Int(BLOCK_SIZE)
    d = feather_depth
    inv_d = 1.0f0 / Float32(d)
    cur = buf_a
    copyto!(cur, src)
    other = buf_b
    @inbounds for r in 1:d
        _feather_dilate_round!(other, src, cur; x_only=x_only)  # src reused as scratch
        t = 1.0f0 - Float32(r - 1) * inv_d                       # t ∈ (0, 1]
        w = scale * t * t * (3.0f0 - 2.0f0 * t)
        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            wx = lx + d; wy = ly + d; wz = lz + d
            if other[wx, wy, wz] && !cur[wx, wy, wz]
                if w > feather_weight[lx, ly, lz, b_idx]
                    feather_weight[lx, ly, lz, b_idx] = w
                end
            end
        end
        cur, other = other, cur
    end
    return nothing
end

function compute_feather_weights!(feather_weight::AbstractArray{Float32, 4},
                                  block_pointer::AbstractArray{Int32, 3},
                                  feather_depth::Int;
                                  x_only::Bool=false)
    fill!(feather_weight, 0.0f0)
    feather_depth <= 0 && return
    bx_max, by_max, bz_max = size(block_pointer)
    bs = Int(BLOCK_SIZE)
    d = feather_depth
    n = bs + 2 * d
    src = Array{Bool, 3}(undef, n, n, n)
    buf_a = similar(src); buf_b = similar(src)

    # Source mask: the INACTIVE region (out-of-grid counts as inactive — same
    # convention as the legacy bx==1/bx_max edge test).
    _feather_pyamr = periodic_y_amr_active()   # periodic-Y AMR: the Y faces are not interfaces
    @inline function _inactive(gx::Int, gy_raw::Int, gz::Int)
        gy = _feather_pyamr ? mod1(gy_raw, by_max * bs) : gy_raw
        (gx < 1 || gy < 1 || gz < 1) && return true
        nbx = div(gx - 1, bs) + 1
        nby = div(gy - 1, bs) + 1
        nbz = div(gz - 1, bs) + 1
        (nbx > bx_max || nby > by_max || nbz > bz_max) && return true
        return block_pointer[nbx, nby, nbz] == Int32(0)
    end

    @inbounds for bz in 1:bz_max, by in 1:by_max, bx in 1:bx_max
        b_idx = Int(block_pointer[bx, by, bz])
        b_idx == 0 && continue
        gx0 = (bx - 1) * bs - d   # window cell (wx,wy,wz) ↔ global g0 + w
        gy0 = (by - 1) * bs - d
        gz0 = (bz - 1) * bs - d
        any_src = false
        for wz in 1:n, wy in 1:n, wx in 1:n
            s = _inactive(gx0 + wx, gy0 + wy, gz0 + wz)
            src[wx, wy, wz] = s
            any_src |= s
        end
        any_src || continue   # fully interior block — no feather
        _feather_apply_window!(feather_weight, b_idx, src, buf_a, buf_b,
                               d, 1.0f0; x_only=x_only)
    end
    return nothing
end

"""
    compute_coarse_side_feather_weights!(feather_weight, parent_block_pointer,
                                         child_block_pointer, feather_depth; x_only=false)

Pass183 (Tier 3 / item 14 sub-task): coarse-side companion to
`compute_feather_weights!`. Marks parent cells on the COARSE side of an AMR
boundary so the existing feather-band damping mechanism in the collision
kernel applies symmetrically to both levels.

The fine-side feather (existing function above) catches the L+1 cells whose
block has no neighbor block — these are the cells immediately inside each
fine refinement zone. But the L cells immediately OUTSIDE the same boundary
(parent cells with no refined neighbor that sit adjacent to a refined zone)
are NOT damped. The persistent ρ-gradient differential between adjacent
levels manifests as the bright halos parked on the coarse side of each AMR
rectangle in YZ schlieren slices. This function damps those parent cells.

Criterion: a parent block (bx,by,bz) has "refined neighbors" if any of its
2×2×2 child block coordinates (2*bx-1+dbx, 2*by-1+dby, 2*bz-1+dbz) exist on
the child level. For such blocks, apply feather across all 6 faces (same
shape as fine-side feather: smoothstep ramp from 1.0 at face → 0 at depth).

ACCUMULATES into the existing `feather_weight` array via max() — does NOT
clear. Call after `compute_feather_weights!` so both contributions add.
"""
function compute_coarse_side_feather_weights!(feather_weight::AbstractArray{Float32, 4},
                                              parent_block_pointer::AbstractArray{Int32, 3},
                                              child_block_pointer::AbstractArray{Int32, 3},
                                              feather_depth::Int;
                                              x_only::Bool=false,
                                              scale::Float32=1.0f0)
    (feather_depth <= 0 || scale <= 0.0f0) && return
    pbx_max, pby_max, pbz_max = size(parent_block_pointer)
    cbx_max, cby_max, cbz_max = size(child_block_pointer)
    bs = Int(BLOCK_SIZE)
    d = feather_depth
    n = bs + 2 * d
    src = Array{Bool, 3}(undef, n, n, n)
    buf_a = similar(src); buf_b = similar(src)

    # CORNER-QUAD FIX (2026-06-10): per-cell Chebyshev distance to the child-
    # supported region (see the fine-side comment above). The legacy 6-axial
    # probe left zero-weight notches at every refinement-box corner/edge.
    #
    # A parent cell's 8 children all live in ONE child block per axis pair
    # (child cells 2g−1, 2g share block (g−1)÷4 + 1), so full-child-support
    # reduces to a single child-block-activity lookup.
    half_bs = bs ÷ 2
    _feather_pyamr = periodic_y_amr_active()   # periodic-Y AMR: wrap the parent Y index
    @inline function _supported(gx::Int, gy_raw::Int, gz::Int)
        gy = _feather_pyamr ? mod1(gy_raw, pby_max * bs) : gy_raw
        (gx < 1 || gy < 1 || gz < 1) && return false
        cbx = div(gx - 1, half_bs) + 1   # == div(2g−2, bs)+1: block of children 2g−1, 2g
        cby = div(gy - 1, half_bs) + 1
        cbz = div(gz - 1, half_bs) + 1
        (cbx > cbx_max || cby > cby_max || cbz > cbz_max) && return false
        return child_block_pointer[cbx, cby, cbz] > Int32(0)
    end

    @inbounds for bz in 1:pbz_max, by in 1:pby_max, bx in 1:pbx_max
        b_idx = Int(parent_block_pointer[bx, by, bz])
        b_idx == 0 && continue
        gx0 = (bx - 1) * bs - d
        gy0 = (by - 1) * bs - d
        gz0 = (bz - 1) * bs - d
        any_src = false
        all_src = true
        for wz in 1:n, wy in 1:n, wx in 1:n
            s = _supported(gx0 + wx, gy0 + wy, gz0 + wz)
            src[wx, wy, wz] = s
            any_src |= s
            all_src &= s
        end
        (!any_src || all_src) && continue   # no interface within reach of this block
        _feather_apply_window!(feather_weight, b_idx, src, buf_a, buf_b,
                               d, scale; x_only=x_only)
    end
    return nothing
end

@inline function _clamp01(x::Float64)
    return min(max(x, 0.0), 1.0)
end

@inline function _smoothstep01(x::Float64)
    t = _clamp01(x)
    return t * t * (3.0 - 2.0 * t)
end

@inline function _wake_window_scale(x::Float64,
                                    wake_start_x::Float64,
                                    wake_end_x::Float64)
    wake_length = max(wake_end_x - wake_start_x, eps(Float64))
    ramp_length = max(0.15 * wake_length, eps(Float64))

    if x < wake_start_x || x > wake_end_x
        return 0.0
    elseif x <= wake_start_x + ramp_length
        return _smoothstep01((x - wake_start_x) / ramp_length)
    elseif x >= wake_end_x - ramp_length
        return _smoothstep01((wake_end_x - x) / ramp_length)
    else
        return 1.0
    end
end

function block_intersects_tapered_wake(b_min_x::Float64, b_max_x::Float64,
                                       b_min_y::Float64, b_max_y::Float64,
                                       b_min_z::Float64, b_max_z::Float64,
                                       wake_start_x::Float64, wake_end_x::Float64,
                                       wake_center_y::Float64, wake_center_z::Float64,
                                       wake_width::Float64, wake_height::Float64;
                                       periodic_ly::Float64=0.0,
                                       span_full::Bool=false)
    center_x = 0.5 * (b_min_x + b_max_x)
    center_y = 0.5 * (b_min_y + b_max_y)
    center_z = 0.5 * (b_min_z + b_max_z)
    half_x = 0.5 * (b_max_x - b_min_x)
    half_y = 0.5 * (b_max_y - b_min_y)
    half_z = 0.5 * (b_max_z - b_min_z)

    x_candidates = Float64[
        center_x,
        clamp(wake_start_x, b_min_x, b_max_x),
        clamp(wake_end_x, b_min_x, b_max_x),
    ]

    base_radius_y = 0.5 * wake_width
    base_radius_z = 0.5 * wake_height
    # Periodic-Y AMR (27 Oct 2026): a body spanning the whole period has a
    # whole-period wake (span_full ⇒ no Y restriction, Y-uniform wake blocks);
    # otherwise the Y gap is the minimum-image distance across the seam.
    dy_c = center_y - wake_center_y
    if periodic_ly > 0.0
        dy_c -= periodic_ly * round(dy_c / periodic_ly)
    end
    dy_gap = span_full ? 0.0 : max(0.0, abs(dy_c) - half_y)
    dz_gap = max(0.0, abs(center_z - wake_center_z) - half_z)

    for x in x_candidates
        scale = _wake_window_scale(x, wake_start_x - half_x, wake_end_x + half_x)
        if scale <= 0.0
            continue
        end

        radius_y = max(base_radius_y * scale, half_y)
        radius_z = max(base_radius_z * scale, half_z)
        metric = (dy_gap / radius_y)^2 + (dz_gap / radius_z)^2
        if metric <= 1.0
            return true
        end
    end

    return false
end

"""
    _periodic_body_y_arc(mesh, offset_y, Ly; nbins=8192) -> (span_full, center, width, straddling)

Periodic-Y AMR (27 Oct 2026). The body's footprint on the periodic Y circle:
the union of its placed triangles' Y ranges taken mod Ly, on `nbins` bins.
`span_full` when the union covers the whole circle (an extrusion through the
period); otherwise the covered arc is the complement of the longest uncovered
run, with its `center` (in [0, Ly)) and `width`. `straddling` counts triangles
that cross y = 0 or y = Ly (only meaningful for a spanning body: forces keep or
drop whole triangles by centroid). `below` / `above` count triangles reaching
below y = 0 / above y = Ly (a body on the seam given as two copies one period
apart has both; one copy alone has one). `gap` is the longest uncovered run (m);
a gap of at most `gap_tol` (callers pass one finest cell) counts as spanning,
`gap_closed = true` (an extrusion slightly shorter than the snapped period).
"""
function _periodic_body_y_arc(mesh::Geometry.SolverMesh, offset_y::Float64, Ly::Float64; nbins::Int=8192,
                              gap_tol::Float64=0.0)
    covered = falses(nbins)
    straddling = 0; below = 0; above = 0
    # STL coordinates are Float32: a vertex meant to lie ON y = 0 or y = Ly is off by a few Float32 ulps.
    tol = 4.0 * Float64(eps(Float32)) * max(Ly, abs(offset_y) + Ly, 1.0)
    full = false
    for tri in mesh.triangles
        y1 = Float64(tri[1][2]) + offset_y; y2 = Float64(tri[2][2]) + offset_y; y3 = Float64(tri[3][2]) + offset_y
        lo = min(y1, y2, y3); hi = max(y1, y2, y3)
        ((lo < -tol && hi > tol) || (lo < Ly - tol && hi > Ly + tol)) && (straddling += 1)
        lo < -tol && (below += 1)
        hi > Ly + tol && (above += 1)
        full && continue
        if hi - lo >= Ly
            full = true
            continue
        end
        b0 = floor(Int, lo / Ly * nbins); b1 = floor(Int, hi / Ly * nbins)
        for b in b0:b1
            covered[mod1(b + 1, nbins)] = true
        end
    end
    (full || all(covered)) && return (span_full = true, center = Ly / 2, width = Ly, straddling = straddling,
                                      below = below, above = above, gap = 0.0, gap_closed = false)
    any(covered) || return (span_full = false, center = Ly / 2, width = 0.0, straddling = straddling,
                            below = below, above = above, gap = Ly, gap_closed = false)
    # longest circular run of uncovered bins
    best_len = 0; best_start = 0
    start = findfirst(covered)            # begin scanning just after a covered bin
    run = 0; run_start = 0
    for k in 1:nbins
        i = mod1(start + k, nbins)
        if !covered[i]
            run == 0 && (run_start = i)
            run += 1
            if run > best_len
                best_len = run; best_start = run_start
            end
        else
            run = 0
        end
    end
    gap = best_len / nbins * Ly
    gap <= gap_tol && return (span_full = true, center = Ly / 2, width = Ly, straddling = straddling,
                              below = below, above = above, gap = gap, gap_closed = true)
    arc_first = mod1(best_start + best_len, nbins)      # first covered bin after the gap
    width = (nbins - best_len) / nbins * Ly
    center = mod((arc_first - 1) / nbins * Ly + width / 2, Ly)
    return (span_full = false, center = center, width = width, straddling = straddling,
            below = below, above = above, gap = gap, gap_closed = false)
end

function setup_multilevel_domain(mesh::Geometry.SolverMesh, params)
    println("\n" * "="^70)
    println("[Setup] Multi-Level Domain Generation")
    if REFINEMENT_STRATEGY == :geometry_first
        println("        Strategy: Geometry-First (Exact Triangle-Block SAT)")
    else
        println("        Strategy: Topology-Legacy (Inherited Obstacles)")
    end
    if BOUNDARY_METHOD == :bouzidi
        println("        Boundary: Bouzidi IBM (finest $BOUZIDI_LEVELS level(s))")
    else
        println("        Boundary: Simple Bounce-Back")
    end
    println("        Temporal Interpolation: $(TEMPORAL_INTERPOLATION ? "ENABLED" : "DISABLED")")
    println("="^70)
    
    num_levels = params.num_levels
    mesh_offset = SVector(params.mesh_offset[1], params.mesh_offset[2], params.mesh_offset[3])
    
    placed_mesh_min = SVector(params.mesh_min...) + mesh_offset
    placed_mesh_max = SVector(params.mesh_max...) + mesh_offset
    
    wake_start_x = placed_mesh_max[1] - (params.reference_length * 0.1)
    wake_end_x = placed_mesh_max[1] + (params.reference_length * WAKE_REFINEMENT_LENGTH)
    
    wake_center_y = (placed_mesh_min[2] + placed_mesh_max[2]) / 2.0
    wake_center_z = (placed_mesh_min[3] + placed_mesh_max[3]) / 2.0
    
    wake_width = (placed_mesh_max[2] - placed_mesh_min[2]) * WAKE_REFINEMENT_WIDTH_FACTOR
    wake_height = (placed_mesh_max[3] - placed_mesh_min[3]) * WAKE_REFINEMENT_HEIGHT_FACTOR
    
    wake_min_y = wake_center_y - wake_width/2.0
    wake_max_y = wake_center_y + wake_width/2.0
    wake_min_z = wake_center_z - wake_height/2.0
    wake_max_z = wake_center_z + wake_height/2.0

    # Periodic-Y AMR (27 Oct 2026): the wake test uses the minimum-image Y gap,
    # and a body spanning the whole period gets a whole-period (Y-uniform) wake.
    # The body's Y footprint is measured ON THE PERIODIC CIRCLE (the union of its
    # triangles' Y ranges mod Ly), not from the bounding box: a body on the seam
    # given as two copies one period apart has a bounding box wider than the
    # period but occupies only a short arc around the seam.
    _pyamr = periodic_y_amr_active()
    _wake_ly = _pyamr ? Float64(params.domain_size[2]) : 0.0
    _wake_span_full = false
    if _pyamr
        _dx_finest = params.dx_coarse / 2.0^(max(num_levels, 1) - 1)
        arc = _periodic_body_y_arc(mesh, Float64(mesh_offset[2]), _wake_ly; gap_tol=_dx_finest)
        _wake_span_full = arc.span_full
        if arc.gap_closed
            @warn @sprintf("[Domain] Periodic-Y AMR: the body covers the period except a %.3e m gap (at most one finest cell, %.3e m); it is treated as spanning (whole-period wake). The solid may still leave a one-cell slit at the seam: extrude the STL past both periodic faces.", arc.gap, _dx_finest)
        end
        if !arc.span_full && xor(arc.below > 0, arc.above > 0)
            @warn "[Domain] Periodic-Y AMR: $(arc.below > 0 ? arc.below : arc.above) surface triangles reach " *
                  "$(arc.below > 0 ? "below y = 0" : "above y = Ly") with no copy one period away. The part outside " *
                  "[0, Ly) is outside the domain (not voxelised, no force): for a body on the seam give two copies " *
                  "one period apart, or move it inside the period (domain.periodic_y_body_shift)."
        end
        if !arc.span_full
            wake_center_y = arc.center
            wake_width = arc.width * WAKE_REFINEMENT_WIDTH_FACTOR
            wake_min_y = wake_center_y - wake_width/2.0
            wake_max_y = wake_center_y + wake_width/2.0
        end
        @printf("[Domain] Periodic-Y AMR: Y period %.6f m (%d coarse blocks); body %s the period%s%s\n",
                params.domain_size[2], params.by_max,
                _wake_span_full ? "spans" : "does not span",
                _wake_span_full ? "" : @sprintf(" (Y arc centre %.6f m, width %.6f m)", arc.center, arc.width),
                ENABLE_WAKE_REFINEMENT ? (_wake_span_full ? " → whole-period wake" : " → minimum-image wake") : "")
        if _wake_span_full && arc.straddling > 0
            @warn "[Domain] Periodic-Y AMR: $(arc.straddling) surface triangles of the spanning body cross a periodic " *
                  "face (y = 0 or y = Ly). Surface forces keep or drop a whole triangle by its centroid, so their " *
                  "overspan area is miscounted: split the extruded STL at y = 0 and y = Ly (as the app's extrusion does)."
        end
        REFERENCE_AREA_CONFIG > 0.0 ||
            @warn "[Domain] Periodic-Y AMR: reference_area_of_full_model is not set; the default is taken from the " *
                  "STL bounding box (overspan / periodic copies included). Set it explicitly (span × chord)."
    end

    outlet_sponge_thickness = min(params.domain_size[1],
                                  params.reference_length * max(Float64(SPONGE_THICKNESS), 0.08))
    outlet_sponge_start = params.domain_size[1] - outlet_sponge_thickness

    if ENABLE_WAKE_REFINEMENT
        @printf("[Domain] Wake refinement envelope: x=[%.6f, %.6f], y=[%.6f, %.6f], z=[%.6f, %.6f]\n",
                wake_start_x, wake_end_x, wake_min_y, wake_max_y, wake_min_z, wake_max_z)
    end
    @printf("[Domain] Outlet sponge zone: x=[%.6f, %.6f] (thickness %.6f m = %.3f L_ref)\n",
            outlet_sponge_start, params.domain_size[1], outlet_sponge_thickness,
            outlet_sponge_thickness / params.reference_length)
    if ENABLE_WAKE_REFINEMENT
        overlap = wake_end_x - outlet_sponge_start
        if overlap > 0.0
            @printf("[Domain] Warning: wake refinement overlaps outlet sponge by %.6f m\n", overlap)
        else
            @printf("[Domain] Wake refinement clears outlet sponge by %.6f m\n", -overlap)
        end
    end
    
    grids = Vector{BlockLevel}()

    for lvl in 1:num_levels
        println("\n--- Level $lvl ---")

        scale = 2^(lvl - 1)
        dx = params.dx_coarse / scale
        dt = 1.0f0 / Float32(scale)
        tau = params.tau_levels[lvl]

        bx_max = params.bx_max * scale
        by_max = params.by_max * scale
        bz_max = params.bz_max * scale

        active_set = Set{Tuple{Int,Int,Int}}()

        if lvl == 1
            println("  Generating Full Wind Tunnel for Level 1...")
            for bz in 1:bz_max, by in 1:by_max, bx in 1:bx_max
                push!(active_set, (bx, by, bz))
            end
            println("  Full domain: $(bx_max)×$(by_max)×$(bz_max) = $(length(active_set)) blocks")
        else
            prev_level = grids[lvl-1]
            prev_scale = 2^(lvl - 2)
            prev_dx = params.dx_coarse / prev_scale
            prev_bs_phys = BLOCK_SIZE * prev_dx


            if REFINEMENT_STRATEGY == :geometry_first
                surface_blocks = get_active_blocks_for_level(mesh, dx, mesh_offset, bx_max, by_max, bz_max)
                union!(active_set, surface_blocks)
                
                if ENABLE_WAKE_REFINEMENT && (WAKE_MAX_LEVEL <= 0 || lvl <= WAKE_MAX_LEVEL)
                    # Geometric, ISOTROPIC wake pad — chord-anchored.
                    # (WAKE_MAX_LEVEL caps the wake below the finest level — Fix.)
                    #
                    # Old scheme: wake_pad = (num_levels − lvl) · (BLOCK_SIZE · dx_lvl)
                    # doubled per level by construction (dx doubles) but was
                    # grid-sized rather than physics-sized and dropped to zero
                    # at the finest level, producing the ONERAM6 step-1800
                    # bullseye at the L2→L1 cliff.
                    #
                    # v1 (replaced): anisotropic pad (0.5·down, 0.2·up, 0.3·lat)
                    # captured wake physics but made AMR boxes much longer in x
                    # than in z. The slice diagnostic showed this caused the
                    # "teeth only on horizontal interfaces" appearance: the
                    # horizontal edges of each refinement box are 3-4× longer
                    # than the vertical edges, so the same per-pixel 1e-4 seam
                    # reads as a prominent continuous line on horizontal
                    # interfaces but is barely noticeable on short vertical
                    # interfaces.
                    #
                    # v2 (current): isotropic pad at 0.33·L_stream·growth.
                    # Chosen to preserve total box volume at each level
                    # vs. v1 (cubed-root of v1's (0.7, 0.6, 0.6) per-axis
                    # factors ≈ 0.33 across the board), so VRAM stays within
                    # ~10 % of the anisotropic case. Horizontal and vertical
                    # AMR box edges now have similar length → the teeth's
                    # preferential appearance on horizontal interfaces goes
                    # away; any residual seam is equally visible (or absent)
                    # on all four sides.
                    depth = max(0, num_levels - lvl)
                    growth = 2.0^depth - 1.0
                    L_stream = params.mesh_extent[1]
                    pad_iso  = 0.33 * L_stream * growth
                    pad_down = pad_iso
                    pad_up   = pad_iso
                    pad_lat  = pad_iso

                    wake_start_x_lvl = max(0.0, wake_start_x - pad_up)
                    wake_end_x_lvl   = min(params.domain_size[1], wake_end_x + pad_down)
                    wake_min_y_lvl   = max(0.0, wake_min_y - pad_lat)
                    wake_max_y_lvl   = min(params.domain_size[2], wake_max_y + pad_lat)
                    wake_min_z_lvl   = max(0.0, wake_min_z - pad_lat)
                    wake_max_z_lvl   = min(params.domain_size[3], wake_max_z + pad_lat)

                    @printf("  Wake target L%d: x=[%.6f, %.6f], y=[%.6f, %.6f], z=[%.6f, %.6f] (pad_down=%.3f m, lat=%.3f m, tapered)\n",
                            lvl, wake_start_x_lvl, wake_end_x_lvl,
                            wake_min_y_lvl, wake_max_y_lvl,
                            wake_min_z_lvl, wake_max_z_lvl,
                            pad_down, pad_lat)

                    for (cbx, cby, cbz) in prev_level.active_block_coords
                        b_min_x = (cbx - 1) * prev_bs_phys
                        b_max_x = cbx * prev_bs_phys
                        b_min_y = (cby - 1) * prev_bs_phys
                        b_max_y = cby * prev_bs_phys
                        b_min_z = (cbz - 1) * prev_bs_phys
                        b_max_z = cbz * prev_bs_phys

                        if block_intersects_tapered_wake(b_min_x, b_max_x, b_min_y, b_max_y, b_min_z, b_max_z,
                                                         wake_start_x_lvl, wake_end_x_lvl,
                                                         wake_center_y, wake_center_z,
                                                         _pyamr ? (wake_width + 2.0 * pad_lat) :
                                                                  (wake_max_y_lvl - wake_min_y_lvl),
                                                         wake_max_z_lvl - wake_min_z_lvl;
                                                         periodic_ly=_wake_ly, span_full=_wake_span_full)
                            for dbz in 0:1, dby in 0:1, dbx in 0:1
                                fbx = 2*cbx - 1 + dbx
                                fby = 2*cby - 1 + dby
                                fbz = 2*cbz - 1 + dbz
                                if fbx >= 1 && fbx <= bx_max && fby >= 1 && fby <= by_max && fbz >= 1 && fbz <= bz_max
                                    push!(active_set, (fbx, fby, fbz))
                                end
                            end
                        end
                    end
                end
                
                prev_level_set = Set(grids[lvl-1].active_block_coords)
                orphans = 0
                final_set = Set{Tuple{Int,Int,Int}}()
                for (bx, by, bz) in active_set
                    pbx = (bx + 1) ÷ 2
                    pby = (by + 1) ÷ 2
                    pbz = (bz + 1) ÷ 2
                    if (pbx, pby, pbz) in prev_level_set
                        push!(final_set, (bx, by, bz))
                    else
                        orphans += 1
                    end
                end
                active_set = final_set
            else
                for (b_idx, (cbx, cby, cbz)) in enumerate(prev_level.active_block_coords)
                    is_surface = any(view(prev_level.obstacle, :, :, :, b_idx))
                    
                    is_wake = false
                    if ENABLE_WAKE_REFINEMENT && !is_surface
                        b_min_x = (cbx - 1) * prev_bs_phys
                        b_max_x = cbx * prev_bs_phys
                        b_min_y = (cby - 1) * prev_bs_phys
                        b_max_y = cby * prev_bs_phys
                        b_min_z = (cbz - 1) * prev_bs_phys
                        b_max_z = cbz * prev_bs_phys
                        
                        is_wake = block_intersects_tapered_wake(b_min_x, b_max_x, b_min_y, b_max_y, b_min_z, b_max_z,
                                                                wake_start_x, wake_end_x,
                                                                wake_center_y, wake_center_z,
                                                                wake_width, wake_height;
                                                                periodic_ly=_wake_ly, span_full=_wake_span_full)
                    end
                    
                    if is_surface || is_wake
                        for dbz in 0:1, dby in 0:1, dbx in 0:1
                            fbx = 2*cbx - 1 + dbx
                            fby = 2*cby - 1 + dby
                            fbz = 2*cbz - 1 + dbz
                            if fbx >= 1 && fbx <= bx_max && fby >= 1 && fby <= by_max && fbz >= 1 && fbz <= bz_max
                                push!(active_set, (fbx, fby, fbz))
                            end
                        end
                    end
                end
            end

            if REFINEMENT_STRATEGY == :geometry_first
                prev_level_set = Set(prev_level.active_block_coords)
                active_set = Set{Tuple{Int,Int,Int}}()

                raw_surface_blocks = get_active_blocks_for_level(mesh, dx, mesh_offset, bx_max, by_max, bz_max)
                surface_blocks = Set{Tuple{Int,Int,Int}}()
                rejected_surface_blocks = 0
                for (bx, by, bz) in raw_surface_blocks
                    if (fld(bx + 1, 2), fld(by + 1, 2), fld(bz + 1, 2)) in prev_level_set
                        push!(surface_blocks, (bx, by, bz))
                    else
                        rejected_surface_blocks += 1
                    end
                end

                # Per-level halo thickness. :distance mode converts a physical band
                # (meters) to block layers at THIS level's resolution, so coarse
                # levels (big blocks) get thinner shells. :blocks mode = legacy
                # fixed counts. block_phys = BLOCK_SIZE · dx (this level).
                local m_up, m_dn, m_sp, m_u, m_d
                if REFINEMENT_BAND_MODE == :distance
                    block_phys = BLOCK_SIZE * dx
                    to_layers(dist) = max(0, ceil(Int, Float64(dist) / max(block_phys, 1e-12)))
                    m_up = to_layers(REFINEMENT_BAND_UPSTREAM)
                    m_dn = to_layers(REFINEMENT_BAND_DOWNSTREAM)
                    m_sp = to_layers(REFINEMENT_BAND_SPAN)
                    m_u  = to_layers(REFINEMENT_BAND_UP)
                    m_d  = to_layers(REFINEMENT_BAND_DOWN)
                    println("  Band(distance)→layers L$lvl: up=$m_up dn=$m_dn span=$m_sp up=$m_u dn=$m_d (block_phys=$(round(block_phys,digits=4)) m)")
                else
                    m_up, m_dn, m_sp, m_u, m_d = REFINEMENT_MARGIN_UPSTREAM, REFINEMENT_MARGIN_DOWNSTREAM,
                                                 REFINEMENT_MARGIN_SPAN, REFINEMENT_MARGIN_UP, REFINEMENT_MARGIN_DOWN
                end
                body_blocks = copy(surface_blocks)
                add_directional_halo_blocks_with_siblings!(
                    body_blocks,
                    m_up, m_dn, m_sp, m_u, m_d,
                    bx_max, by_max, bz_max;
                    parent_set=prev_level_set,
                )

                wake_blocks = Set{Tuple{Int,Int,Int}}()
                if ENABLE_WAKE_REFINEMENT && (WAKE_MAX_LEVEL <= 0 || lvl <= WAKE_MAX_LEVEL)
                    # Mirror of the pass-1 isotropic pad (see comment above).
                    # WAKE_MAX_LEVEL>0 caps the wake slab below this level so the
                    # FINEST level emerges body-only (shape-hugging) — see Fix.
                    depth = max(0, num_levels - lvl)
                    growth = 2.0^depth - 1.0
                    L_stream = params.mesh_extent[1]
                    pad_iso  = 0.33 * L_stream * growth
                    # Mesh-efficiency (2026-05-30): the isotropic pad balloons the
                    # COARSE levels (large growth) up-/down-/laterally equally. A wake
                    # is a downstream, laterally-thin structure: padding it upstream
                    # just re-covers the body halo, and padding it laterally at coarse
                    # levels wastes cells. WAKE_ANISOTROPIC_PAD keeps the full
                    # downstream reach but scales the upstream/lateral pad down.
                    # Gated + reversible; the isotropic path is the `false` default.
                    if WAKE_ANISOTROPIC_PAD
                        pad_down = pad_iso
                        pad_up   = pad_iso * WAKE_PAD_UPSTREAM_SCALE
                        pad_lat  = pad_iso * WAKE_PAD_LATERAL_SCALE
                    else
                        pad_down = pad_iso
                        pad_up   = pad_iso
                        pad_lat  = pad_iso
                    end

                    wake_start_x_lvl = max(0.0, wake_start_x - pad_up)
                    wake_end_x_lvl   = min(params.domain_size[1], wake_end_x + pad_down)
                    wake_min_y_lvl   = max(0.0, wake_min_y - pad_lat)
                    wake_max_y_lvl   = min(params.domain_size[2], wake_max_y + pad_lat)
                    wake_min_z_lvl   = max(0.0, wake_min_z - pad_lat)
                    wake_max_z_lvl   = min(params.domain_size[3], wake_max_z + pad_lat)

                    for (cbx, cby, cbz) in prev_level.active_block_coords
                        b_min_x = (cbx - 1) * prev_bs_phys
                        b_max_x = cbx * prev_bs_phys
                        b_min_y = (cby - 1) * prev_bs_phys
                        b_max_y = cby * prev_bs_phys
                        b_min_z = (cbz - 1) * prev_bs_phys
                        b_max_z = cbz * prev_bs_phys

                        if block_intersects_tapered_wake(b_min_x, b_max_x, b_min_y, b_max_y, b_min_z, b_max_z,
                                                         wake_start_x_lvl, wake_end_x_lvl,
                                                         wake_center_y, wake_center_z,
                                                         _pyamr ? (wake_width + 2.0 * pad_lat) :
                                                                  (wake_max_y_lvl - wake_min_y_lvl),
                                                         wake_max_z_lvl - wake_min_z_lvl;
                                                         periodic_ly=_wake_ly, span_full=_wake_span_full)
                            for dbz in 0:1, dby in 0:1, dbx in 0:1
                                fbx = 2*cbx - 1 + dbx
                                fby = 2*cby - 1 + dby
                                fbz = 2*cbz - 1 + dbz
                                if fbx >= 1 && fbx <= bx_max && fby >= 1 && fby <= by_max && fbz >= 1 && fbz <= bz_max
                                    push!(wake_blocks, (fbx, fby, fbz))
                                end
                            end
                        end
                    end
                end

                union!(active_set, body_blocks)
                union!(active_set, wake_blocks)
                orphan_count = filter_orphan_blocks!(active_set, prev_level_set)

                println("  Surface seed blocks: $(length(surface_blocks))" *
                        (rejected_surface_blocks > 0 ? " (rejected $rejected_surface_blocks orphan candidates)" : ""))
                println("  Body halo blocks: +$(length(body_blocks) - length(surface_blocks)) " *
                        "[upstream=$(m_up), downstream=$(m_dn), span=$(m_sp), up=$(m_u), down=$(m_d)]")
                (ENABLE_WAKE_REFINEMENT && (WAKE_MAX_LEVEL <= 0 || lvl <= WAKE_MAX_LEVEL)) &&
                    println("  Wake blocks: $(length(wake_blocks))")
                (ENABLE_WAKE_REFINEMENT && WAKE_MAX_LEVEL > 0 && lvl > WAKE_MAX_LEVEL) &&
                    @warn "[Wake] Level $lvl wake refinement SKIPPED (wake_max_level=$WAKE_MAX_LEVEL): " *
                          "downstream wake resolved one level coarser here. Cd/wake-spectrum will differ from an uncapped run."
                orphan_count > 0 && println("  Removed $orphan_count orphan fine blocks after union")
            end
        end
        
        n_before_halo = length(active_set)
        
        if lvl > 1 && REFINEMENT_STRATEGY != :geometry_first
            add_directional_halo_blocks_with_siblings!(
                active_set,
                REFINEMENT_MARGIN_UPSTREAM,
                REFINEMENT_MARGIN_DOWNSTREAM,
                REFINEMENT_MARGIN_SPAN,
                REFINEMENT_MARGIN_UP,
                REFINEMENT_MARGIN_DOWN,
                bx_max, by_max, bz_max;
                parent_set=Set(grids[lvl-1].active_block_coords),
            )
        elseif lvl == 1
            add_halo_blocks_with_siblings!(active_set, REFINEMENT_MARGIN, bx_max, by_max, bz_max)
        end
        ensure_complete_parent_coverage!(active_set, bx_max, by_max, bz_max)

        # ── Wall-buffer: forbid refinement from touching the domain walls ──
        # Without this, teal (refined-and-restricted) and yellow (unrefined)
        # cells both touch the walls depending on where the geometry/wake
        # halo happens to end, producing asymmetric treatment at the
        # boundary. Filtering here ensures the coarsest level is the only
        # one adjacent to the walls, so all four/six walls see identical
        # physics. The check uses block index 1, 2, N-1, N (i.e. all 8
        # children of a wall-adjacent parent) so that removal is sibling-
        # complete and parent coverage remains valid.
        if lvl > 1
            n_before_wall_buf = length(active_set)
            filter!(bc -> !is_wall_adjacent(bc[1], bc[2], bc[3], bx_max, by_max, bz_max;
                                            skip_y_min=HALF_Y_DOMAIN),
                    active_set)
            n_removed_wall_buf = n_before_wall_buf - length(active_set)
            if n_removed_wall_buf > 0
                println("  Wall buffer: removed $n_removed_wall_buf blocks touching domain boundary")
            end
        end

        n_after_halo = length(active_set)
        if lvl > 1
            println("  Added $(n_after_halo - n_before_halo) post-selection closure blocks")
        end
        
        sorted_blocks = sort(collect(active_set))
        n_blocks = length(sorted_blocks)

        if n_blocks > 0
            x_min = (minimum(c[1] for c in sorted_blocks) - 1) * BLOCK_SIZE * dx
            x_max =  maximum(c[1] for c in sorted_blocks)      * BLOCK_SIZE * dx
            y_min = (minimum(c[2] for c in sorted_blocks) - 1) * BLOCK_SIZE * dx
            y_max =  maximum(c[2] for c in sorted_blocks)      * BLOCK_SIZE * dx
            z_min = (minimum(c[3] for c in sorted_blocks) - 1) * BLOCK_SIZE * dx
            z_max =  maximum(c[3] for c in sorted_blocks)      * BLOCK_SIZE * dx
            @printf("  Active extent: x=[%.6f, %.6f], y=[%.6f, %.6f], z=[%.6f, %.6f]\n",
                    x_min, x_max, y_min, y_max, z_min, z_max)
        end
        
        nb_table = build_neighbor_table(
            sorted_blocks, bx_max, by_max, bz_max;
            periodic_y=SPAN_PERIODIC_Y && (NUM_LEVELS_CONFIG == 1 || PERIODIC_Y_AMR))
        
        bs = BLOCK_SIZE
        obstacle_arr = falses(bs, bs, bs, n_blocks)
        sponge_arr = zeros(Float32, bs, bs, bs, n_blocks)
        wall_dist_arr = fill(100.0f0, bs, bs, bs, n_blocks)
        
        block_ptr = zeros(Int32, bx_max, by_max, bz_max)
        for (idx, (bx, by, bz)) in enumerate(sorted_blocks)
            block_ptr[bx, by, bz] = Int32(idx)
        end
        
        if REFINEMENT_MARKER_ONLY
            println("[Domain] Refinement marker only: skipping obstacle voxelization/flood fill.")
        else
            voxelize_blocks!(obstacle_arr, sorted_blocks, mesh, dx, mesh_offset)
            perform_flood_fill!(obstacle_arr, sorted_blocks, block_ptr, bx_max, by_max, bz_max)
            if BOUZIDI_Q_GEOMETRY == :sphere_analytic
                apply_analytic_sphere_obstacle!(obstacle_arr, sorted_blocks, mesh, dx, mesh_offset)
            end
        end
        
        apply_sponge!(sponge_arr, sorted_blocks, params, scale)
        
        if WALL_MODEL_ENABLED
            compute_wall_distances!(wall_dist_arr, sorted_blocks, obstacle_arr, mesh, dx, mesh_offset)
        end
        
        use_bouzidi = should_use_bouzidi(lvl, num_levels, BOUNDARY_METHOD, BOUZIDI_LEVELS)
        
        bouzidi_q_map = nothing
        bouzidi_cell_block = nothing
        bouzidi_cell_x = nothing
        bouzidi_cell_y = nothing
        bouzidi_cell_z = nothing
        bouzidi_tri_map = nothing
        n_boundary_cells = 0
        
        if use_bouzidi
            println("[Bouzidi] Computing Q-map and surface mapping for Level $lvl...")
            
            q_map_cpu, cell_block, cell_x, cell_y, cell_z, tri_map, n_b = 
                compute_bouzidi_qmap_sparse(sorted_blocks, mesh, dx, mesh_offset, BLOCK_SIZE, obstacle_arr;
                                            q_geometry=BOUZIDI_Q_GEOMETRY)
            
            bouzidi_q_map = q_map_cpu
            bouzidi_cell_block = cell_block
            bouzidi_cell_x = cell_x
            bouzidi_cell_y = cell_y
            bouzidi_cell_z = cell_z
            bouzidi_tri_map = tri_map
            n_boundary_cells = n_b
            
            println("[Bouzidi] Level $lvl: $n_boundary_cells boundary cells")

            if BOUZIDI_SYMMETRY_REPORT && n_boundary_cells > 0
                report_bouzidi_link_symmetry(
                    Int(lvl),
                    bouzidi_q_map, bouzidi_cell_block,
                    bouzidi_cell_x, bouzidi_cell_y, bouzidi_cell_z,
                    Int(n_boundary_cells),
                    obstacle_arr, nb_table, sorted_blocks,
                    Int(BLOCK_SIZE),
                    C_X, C_Y, C_Z, OPP,
                    Q_MIN_THRESHOLD,
                )
            end

            if BOUZIDI_FF_AUDIT && n_boundary_cells > 0
                report_bouzidi_ff_fallback(
                    Int(lvl),
                    bouzidi_q_map, bouzidi_cell_block,
                    bouzidi_cell_x, bouzidi_cell_y, bouzidi_cell_z,
                    Int(n_boundary_cells),
                    obstacle_arr, nb_table, sorted_blocks,
                    Int(BLOCK_SIZE),
                    C_X, C_Y, C_Z, OPP,
                    Q_MIN_THRESHOLD,
                    OUT_DIR,
                )
            end

            if BOUZIDI_MISSED_LINK_AUDIT && n_boundary_cells > 0
                report_bouzidi_missed_links(
                    Int(lvl),
                    bouzidi_q_map, bouzidi_cell_block,
                    bouzidi_cell_x, bouzidi_cell_y, bouzidi_cell_z,
                    Int(n_boundary_cells),
                    obstacle_arr, nb_table, sorted_blocks,
                    Int(BLOCK_SIZE),
                    C_X, C_Y, C_Z, OPP,
                    Q_MIN_THRESHOLD,
                    OUT_DIR,
                )
            end
        end


        level = BlockLevel(lvl, sorted_blocks, nb_table, Float32(dx), dt, tau;
                           bouzidi_q_map = bouzidi_q_map,
                           bouzidi_cell_block = bouzidi_cell_block,
                           bouzidi_cell_x = bouzidi_cell_x,
                           bouzidi_cell_y = bouzidi_cell_y,
                           bouzidi_cell_z = bouzidi_cell_z,
                           bouzidi_tri_map = bouzidi_tri_map,
                           n_boundary_cells = n_boundary_cells,
                           enable_temporal_interpolation = TEMPORAL_INTERPOLATION,
                           enable_macro_temporal_interpolation = INTERFACE_MACRO_TEMPORAL_INTERPOLATION,
                           has_children = (lvl < num_levels),
                           periodic_grid_by = periodic_y_amr_active() ? by_max : 0)

        level.obstacle .= obstacle_arr
        level.sponge .= sponge_arr
        level.wall_dist .= wall_dist_arr

        # Populate AMR-interface feather band (extra viscous dissipation
        # at cells adjacent to inactive-block neighbors). No-op when
        # FEATHER_DEPTH = 0 or this is the coarsest level (L1 has no
        # inactive-block neighbors on the level-to-level sense — its
        # "outside" is the domain boundary, which is handled by sponge).
        amr_weight_depth = max(FEATHER_DEPTH, AMR_REGULARIZATION_DEPTH)
        if amr_weight_depth > 0 && lvl > 1
            compute_feather_weights!(level.feather_weight,
                                     level.block_pointer,
                                     amr_weight_depth;
                                     x_only=FEATHER_X_ONLY)
            n_feather = count(level.feather_weight .> 0.0f0)
            n_total = length(level.feather_weight)
            @printf("  AMR interface weight%s: %d/%d cells (%.1f%%), max weight = %.3f\n",
                    FEATHER_X_ONLY ? " [x-only]" : "",
                    n_feather, n_total,
                    100.0 * n_feather / max(n_total, 1),
                    maximum(level.feather_weight))
        end

        push!(grids, level)
        
        n_voxels = n_blocks * BLOCK_SIZE^3
        @printf("  Total: %d blocks, %.2f M voxels\n", n_blocks, n_voxels/1e6)
        if use_bouzidi
            println("  Boundary: Bouzidi IBM ($n_boundary_cells cells, sparse)")
        else
            println("  Boundary: Simple bounce-back")
        end
    end
    
    # ═══════════════════════════════════════════════════════════════════
    # Level grading pass — ensure no block at level N+2 borders level N
    # without an intermediate level N+1 buffer.
    #
    # Order matters: strip wall-adjacent refined blocks FIRST, then run
    # grading LAST and unconditionally. Grading is non-negotiable (a 2:1
    # jump causes density banding), while the wall buffer is a soft
    # preference. If grading needs to add a block that sits in the wall
    # buffer, we keep it.
    # ═══════════════════════════════════════════════════════════════════
    if num_levels >= 2
        level_sets = [Set(grids[l].active_block_coords) for l in 1:num_levels]
        bx_maxes = [params.bx_max * 2^(l - 1) for l in 1:num_levels]
        by_maxes = [params.by_max * 2^(l - 1) for l in 1:num_levels]
        bz_maxes = [params.bz_max * 2^(l - 1) for l in 1:num_levels]

        # ── Obstacle-standoff precompute (Fix A). Capture the finest-level
        #    surface footprint ONCE so the closure-loop standoff pass dilates a
        #    FIXED seed (idempotent, terminating). marker-only geometry is an
        #    AMR marker not a wall → standoff is meaningless, skip.
        #
        # CORRECTNESS FLOOR (2026-06-29): a real-wall (non-marker) obstacle must
        # NEVER be touched or crossed by an AMR box edge — a box edge cutting the
        # body leaves the surface partly resolved at a coarser level and breaks the
        # wall BC. Without this floor, body_shell_blocks=0 skipped the standoff
        # ENTIRELY, so the finest box was placed by the raw geometry footprint +
        # directional margins and could sit flush against (or clip) the surface
        # (observed on the Ma1.5 sphere: finest box offset downstream, upstream
        # edge on the sphere). We therefore guarantee a minimum isotropic
        # separation for every real wall, regardless of the knob. An explicit
        # body_shell_blocks ≥ the floor is respected as-is (tunes a HEALTHIER
        # margin); marker-only geometry still gets no standoff (it is not a wall).
        standoff_n = REFINEMENT_MARKER_ONLY ? 0 :
                     max(REFINEMENT_BODY_SHELL_BLOCKS, MIN_OBSTACLE_STANDOFF_BLOCKS)
        surface_fine = Set{Tuple{Int,Int,Int}}()
        if standoff_n > 0 && !REFINEMENT_MARKER_ONLY
            dx_fine = params.dx_coarse / 2.0^(num_levels - 1)
            surface_fine = get_active_blocks_for_level(
                mesh, dx_fine, mesh_offset,
                bx_maxes[num_levels], by_maxes[num_levels], bz_maxes[num_levels])
            println("[Standoff] Finest surface footprint: $(length(surface_fine)) blocks; " *
                    "enforcing >=$standoff_n-block isotropic separation")
        end

        # ── Sharp-edge (feature-edge) local refinement precompute (2026-06-24).
        #    Detect STL edges where adjacent triangles fold by more than the
        #    threshold (wing LE/TE, creases, corners) and capture the finest-level
        #    footprint of those triangles ONCE, so the closure-loop pass dilates a
        #    FIXED seed (idempotent). marker-only geometry isn't a wall → skip.
        sharp_fine = Set{Tuple{Int,Int,Int}}()
        if SHARP_EDGE_REFINE && num_levels >= 2 && !REFINEMENT_MARKER_ONLY
            dx_fine_s = params.dx_coarse / 2.0^(num_levels - 1)
            sharp_tris = detect_sharp_edge_triangles(mesh, SHARP_EDGE_ANGLE_THRESHOLD_DEG)
            sharp_fine = get_sharp_edge_blocks_for_level(
                mesh, sharp_tris, dx_fine_s, mesh_offset,
                bx_maxes[num_levels], by_maxes[num_levels], bz_maxes[num_levels])
            println("[SharpEdge] $(length(sharp_tris)) sharp-edge triangles " *
                    "(>$(SHARP_EDGE_ANGLE_THRESHOLD_DEG)° fold) → $(length(sharp_fine)) finest blocks; " *
                    "enforcing +$SHARP_EDGE_MARGIN_BLOCKS-block local refinement")
        end

        # Pass 1: strip wall-adjacent refined blocks (cascades orphans to finer).
        wall_removed = strip_wall_adjacent_refinement!(level_sets, bx_maxes, by_maxes, bz_maxes)
        if wall_removed > 0
            println("\n[WallBuffer] Stripped $wall_removed wall-adjacent refined blocks before grading")
        end

        # Pass 2: close the interpolation stencil and then enforce grading.
        # The donor halo is coarser than the child level, so it is a cheap way
        # to keep high-order prolongation from falling back to one-sided edge
        # substitutes at AMR box boundaries. D3Q39 reaches three fine cells
        # across an interface and the cubic prolongation needs four parent
        # samples, so use two parent-block layers instead of the bare minimum.
        donor_added = 0
        grading_added = 0
        standoff_added = 0
        symmetry_added = 0
        sharp_added = 0
        span_added = 0
        if num_levels >= 2
            for closure_pass in 1:8
                # Standoff FIRST: force the guaranteed obstacle shell + cascade
                # ancestors, THEN let donor + grading close stencils/jumps around
                # whatever the standoff added (joint fixed point).
                added_standoff = enforce_obstacle_standoff!(
                    level_sets, bx_maxes, by_maxes, bz_maxes, surface_fine;
                    standoff_blocks=standoff_n)
                # Sharp-edge local refinement: concentrate EXTRA finest blocks on the
                # high-curvature feature edges (LE/TE), on top of the uniform shell.
                added_sharp = SHARP_EDGE_REFINE ?
                    enforce_sharp_edge_refinement!(
                        level_sets, bx_maxes, by_maxes, bz_maxes, sharp_fine;
                        margin_blocks=SHARP_EDGE_MARGIN_BLOCKS) : 0
                added_donor = ensure_prolongation_donor_halos!(
                    level_sets, bx_maxes, by_maxes, bz_maxes;
                    layers=2
                )
                added_grading = num_levels >= 3 ?
                    ensure_level_grading!(level_sets, bx_maxes, by_maxes, bz_maxes) : 0
                added_symmetry = AMR_ENFORCE_Z_SYMMETRY ?
                    enforce_z_mirror_symmetry!(level_sets, bx_maxes, by_maxes, bz_maxes;
                                               start_level=2) : 0
                added_symmetry_y = AMR_ENFORCE_Y_SYMMETRY ?
                    enforce_y_mirror_symmetry!(level_sets, bx_maxes, by_maxes, bz_maxes;
                                               start_level=2) : 0
                # Periodic-Y AMR option: extrude every refined level over the period.
                added_span = SPAN_UNIFORM_LEVELS ?
                    enforce_span_uniform_levels!(level_sets, bx_maxes, by_maxes, bz_maxes) : 0
                standoff_added += added_standoff
                sharp_added += added_sharp
                donor_added += added_donor
                grading_added += added_grading
                symmetry_added += added_symmetry + added_symmetry_y
                span_added += added_span
                (added_standoff == 0 && added_sharp == 0 && added_donor == 0 &&
                 added_grading == 0 && added_symmetry == 0 && added_symmetry_y == 0 &&
                 added_span == 0) && break
            end
        end
        if span_added > 0
            println("[SpanUniform] Added $span_added blocks so every refined level spans the whole Y period")
        end
        if standoff_added > 0
            println("[Standoff] Force-added $standoff_added blocks (+ancestors) for the guaranteed obstacle shell")
        end
        if sharp_added > 0
            println("[SharpEdge] Force-added $sharp_added blocks (+ancestors) for local feature-edge refinement")
        end
        if donor_added > 0
            println("[DonorHalo] Added $donor_added coarser blocks so prolongation stencils have symmetric parent support")
        end
        if grading_added > 0
            println("[Grading] Added $grading_added blocks across intermediate levels to enforce max 1-level jumps")
        end
        if symmetry_added > 0
            println("[Symmetry] Added $symmetry_added mirror AMR blocks (+ancestors) for topology closure (z and/or y)")
        end

        # Wake-waste diagnostic (Fix B): how much bigger is the finest level than
        # the surface+standoff shell that should hug the body? Ratio≈1 ⇒ tight hug.
        if standoff_n > 0 && !isempty(surface_fine)
            n_fine    = length(level_sets[num_levels])
            n_surface = length(surface_fine)
            shell_set = dilate_blocks_isotropic(surface_fine, standoff_n,
                          bx_maxes[num_levels], by_maxes[num_levels], bz_maxes[num_levels])
            ensure_complete_parent_coverage!(shell_set, bx_maxes[num_levels],
                          by_maxes[num_levels], bz_maxes[num_levels])
            n_shell = length(shell_set)
            shell_ratio = n_fine / max(n_shell, 1)
            @printf("[Standoff] Finest level: %d blocks | surface %d | N=%d shell %d | excess-over-shell %.2fx\n",
                    n_fine, n_surface, standoff_n, n_shell, shell_ratio)
            if shell_ratio > 1.5
                @warn "[Standoff] Finest level is $(round(shell_ratio,digits=1))x larger than the " *
                      "surface+standoff shell — a wake/box slab still dominates the finest level. " *
                      "Set advanced.refinement.wake_max_level=$(num_levels-1) to cap the wake below the finest level."
            end
        end

        if donor_added > 0 || grading_added > 0 || wall_removed > 0 ||
           standoff_added > 0 || symmetry_added > 0 || sharp_added > 0 || span_added > 0
            # Rebuild any level whose block set changed. Compare coordinates,
            # not just counts: a same-count remove/add leaves stale block
            # pointers and neighbor tables.
            for lvl in 2:num_levels
                old_coords = Set(grids[lvl].active_block_coords)
                new_coords = level_sets[lvl]
                new_coords == old_coords && continue
                if length(new_coords) == length(old_coords)
                    println("[Mesh] Level $lvl block count unchanged, but coordinates changed; rebuilding stale AMR maps")
                end

                println("[Mesh] Rebuilding level $lvl: $(length(old_coords)) → $(length(new_coords)) blocks ($(length(new_coords) - length(old_coords) >= 0 ? "+" : "")$(length(new_coords) - length(old_coords)))")

                scale = 2^(lvl - 1)
                dx = params.dx_coarse / scale
                dt_lvl = 1.0f0 / Float32(scale)
                tau_lvl = params.tau_levels[lvl]
                bx_max_lvl = bx_maxes[lvl]
                by_max_lvl = by_maxes[lvl]
                bz_max_lvl = bz_maxes[lvl]
                bs = BLOCK_SIZE

                sorted_blocks = sort(collect(new_coords))
                n_blocks = length(sorted_blocks)

                nb_table = build_neighbor_table(
                    sorted_blocks, bx_max_lvl, by_max_lvl, bz_max_lvl;
                    periodic_y=SPAN_PERIODIC_Y && (NUM_LEVELS_CONFIG == 1 || PERIODIC_Y_AMR))
                obstacle_arr = falses(bs, bs, bs, n_blocks)
                sponge_arr = zeros(Float32, bs, bs, bs, n_blocks)
                wall_dist_arr = fill(100.0f0, bs, bs, bs, n_blocks)

                block_ptr = zeros(Int32, bx_max_lvl, by_max_lvl, bz_max_lvl)
                for (idx, (bx, by, bz)) in enumerate(sorted_blocks)
                    block_ptr[bx, by, bz] = Int32(idx)
                end

                if REFINEMENT_MARKER_ONLY
                    println("[Domain] Refinement marker only: skipping obstacle voxelization/flood fill.")
                else
                    voxelize_blocks!(obstacle_arr, sorted_blocks, mesh, dx, mesh_offset)
                    perform_flood_fill!(obstacle_arr, sorted_blocks, block_ptr, bx_max_lvl, by_max_lvl, bz_max_lvl)
                    if BOUZIDI_Q_GEOMETRY == :sphere_analytic
                        apply_analytic_sphere_obstacle!(obstacle_arr, sorted_blocks, mesh, dx, mesh_offset)
                    end
                end
                apply_sponge!(sponge_arr, sorted_blocks, params, scale)

                if WALL_MODEL_ENABLED
                    compute_wall_distances!(wall_dist_arr, sorted_blocks, obstacle_arr, mesh, dx, mesh_offset)
                end

                use_bouzidi = should_use_bouzidi(lvl, num_levels, BOUNDARY_METHOD, BOUZIDI_LEVELS)
                bouzidi_q_map = nothing
                bouzidi_cell_block = nothing
                bouzidi_cell_x = nothing
                bouzidi_cell_y = nothing
                bouzidi_cell_z = nothing
                bouzidi_tri_map = nothing
                n_boundary_cells = 0

                if use_bouzidi
                    q_map_cpu, cell_block, cell_x, cell_y, cell_z, tri_map, n_b =
                        compute_bouzidi_qmap_sparse(sorted_blocks, mesh, dx, mesh_offset, BLOCK_SIZE, obstacle_arr;
                                                    q_geometry=BOUZIDI_Q_GEOMETRY)
                    bouzidi_q_map = q_map_cpu
                    bouzidi_cell_block = cell_block
                    bouzidi_cell_x = cell_x
                    bouzidi_cell_y = cell_y
                    bouzidi_cell_z = cell_z
                    bouzidi_tri_map = tri_map
                    n_boundary_cells = n_b
                end

                level = BlockLevel(lvl, sorted_blocks, nb_table, Float32(dx), dt_lvl, tau_lvl;
                                   bouzidi_q_map = bouzidi_q_map,
                                   bouzidi_cell_block = bouzidi_cell_block,
                                   bouzidi_cell_x = bouzidi_cell_x,
                                   bouzidi_cell_y = bouzidi_cell_y,
                                   bouzidi_cell_z = bouzidi_cell_z,
                                   bouzidi_tri_map = bouzidi_tri_map,
                                   n_boundary_cells = n_boundary_cells,
                                   enable_temporal_interpolation = TEMPORAL_INTERPOLATION,
                                   enable_macro_temporal_interpolation = INTERFACE_MACRO_TEMPORAL_INTERPOLATION,
                                   has_children = (lvl < num_levels),
                                   periodic_grid_by = periodic_y_amr_active() ? by_max_lvl : 0)

                level.obstacle .= obstacle_arr
                level.sponge .= sponge_arr
                level.wall_dist .= wall_dist_arr

                grids[lvl] = level

                n_voxels = n_blocks * bs^3
                @printf("  Rebuilt: %d blocks, %.2f M voxels\n", n_blocks, n_voxels / 1e6)
            end
        else
            println("\n[Grading] No changes needed — mesh is already graded and wall-buffered")
        end
    end

    # Recompute AMR-interface feather weights after the final topology pass.
    # Levels can be rebuilt by grading above; a rebuilt BlockLevel starts with
    # a zero feather array, so doing this once here guarantees every refined
    # level has interface weights that match the actual block boundary.
    amr_weight_depth = max(FEATHER_DEPTH, AMR_REGULARIZATION_DEPTH)
    if amr_weight_depth > 0
        println("\n[AMR] Recomputing interface weights on final AMR topology...")
        # Fine-side feather (cells on the L+1 side whose block has an
        # inactive neighbor — the outer perimeter of each fine refinement
        # zone).
        for lvl in 2:num_levels
            compute_feather_weights!(grids[lvl].feather_weight,
                                     grids[lvl].block_pointer,
                                     amr_weight_depth;
                                     x_only=FEATHER_X_ONLY)
            n_feather = count(grids[lvl].feather_weight .> 0.0f0)
            n_total = length(grids[lvl].feather_weight)
            @printf("  Level %d%s [fine-side]: %d/%d cells (%.1f%%), max w = %.3f\n",
                    lvl,
                    FEATHER_X_ONLY ? " x-only" : "",
                    n_feather, n_total,
                    100.0 * n_feather / max(n_total, 1),
                    maximum(grids[lvl].feather_weight))
        end
        # Pass185 REVERT: coarse-side feather DISABLED. Pass183 introduced
        # `compute_coarse_side_feather_weights!` (above, still defined for
        # future use) but the resulting symmetric damping caused horizontal
        # stripes in the velocity highpass slice and visibly wider black
        # bands at AMR boundaries in the YZ schlieren. The fine-side-only
        # feather (pass174f, the historical setting) was the right behavior.
        # The function remains available — re-enable by uncommenting the
        # block below if a future iteration figures out the proper coarse-
        # side damping magnitude.
        if COARSE_SIDE_FEATHER_SCALE > 0.0f0
            for lvl in 1:(num_levels - 1)
                compute_coarse_side_feather_weights!(grids[lvl].feather_weight,
                                                     grids[lvl].block_pointer,
                                                     grids[lvl + 1].block_pointer,
                                                     amr_weight_depth;
                                                     x_only=FEATHER_X_ONLY,
                                                     scale=Float32(COARSE_SIDE_FEATHER_SCALE))
                n_feather = count(grids[lvl].feather_weight .> 0.0f0)
                n_total = length(grids[lvl].feather_weight)
                @printf("  Level %d coarse-side scaled feather: %d/%d cells (%.1f%%), scale=%.3f, max w = %.3f\n",
                        lvl, n_feather, n_total,
                        100.0 * n_feather / max(n_total, 1),
                        Float64(COARSE_SIDE_FEATHER_SCALE),
                        maximum(grids[lvl].feather_weight))
            end
        end
    end

    println("\n[Verify] Checking parent coverage...")
    for lvl in 2:num_levels
        fine_set = Set(grids[lvl].active_block_coords)
        coarse_set = Set(grids[lvl-1].active_block_coords)
        missing = 0
        for (fbx, fby, fbz) in fine_set
            pbx = (fbx + 1) ÷ 2
            pby = (fby + 1) ÷ 2
            pbz = (fbz + 1) ÷ 2
            if !((pbx, pby, pbz) in coarse_set)
                missing += 1
            end
        end
        println("  Level $lvl: $(missing == 0 ? "✓" : "⚠") (missing parents: $missing)")
    end

    # Verify grading: no 2-level jumps between spatial neighbors
    println("[Verify] Checking level grading (max 1-level jump)...")
    for lvl in 3:num_levels
        fine_set = Set(grids[lvl].active_block_coords)
        mid_set = Set(grids[lvl-1].active_block_coords)
        coarse_set = Set(grids[lvl-2].active_block_coords)
        violations = 0
        for (fbx, fby, fbz) in fine_set
            pbx = fld(fbx + 1, 2)
            pby = fld(fby + 1, 2)
            pbz = fld(fbz + 1, 2)
            gpbx = fld(pbx + 1, 2)
            gpby = fld(pby + 1, 2)
            gpbz = fld(pbz + 1, 2)
            for dz in -1:1, dy in -1:1, dx in -1:1
                (dx == 0 && dy == 0 && dz == 0) && continue
                ngpx = gpbx + dx
                ngpy = _topo_wrap_by(gpby + dy, params.by_max * 2^(lvl - 3))
                ngpz = gpbz + dz
                (ngpx, ngpy, ngpz) in coarse_set || continue
                has_mid = false
                for dbz in 0:1, dby in 0:1, dbx in 0:1
                    mcx = 2 * ngpx - 1 + dbx
                    mcy = 2 * ngpy - 1 + dby
                    mcz = 2 * ngpz - 1 + dbz
                    if (mcx, mcy, mcz) in mid_set
                        has_mid = true
                        break
                    end
                end
                has_mid || (violations += 1)
            end
        end
        println("  Level $lvl↔$(lvl-2): $(violations == 0 ? "✓" : "⚠ $violations violations")")
    end

    covered_counts = compute_all_amr_covered_masks!(grids)
    println("[Verify] Covered coarse cells by level...")
    for lvl in 1:(num_levels - 1)
        n_total = length(grids[lvl].amr_covered)
        n_cov = covered_counts[lvl]
        if n_total > 1
            @printf("  Level %d: %d/%d cells covered by L%d (%.1f%%)\n",
                    lvl, n_cov, n_total, lvl + 1, 100.0 * n_cov / max(n_total, 1))
        end
    end

    return grids
end

function validate_half_domain_mesh!(mesh::Geometry.SolverMesh)
    !HALF_Y_DOMAIN && return

    min_y = mesh.min_bounds[2]
    max_y = mesh.max_bounds[2]
    extent_x = mesh.max_bounds[1] - mesh.min_bounds[1]
    extent_y = max_y - min_y
    extent_z = mesh.max_bounds[3] - mesh.min_bounds[3]
    tol = max(1.0e-8, maximum((extent_x, extent_y, extent_z)) * 1.0e-6)

    if SYMMETRIC_ANALYSIS && min_y < -tol
        # Full-symmetric STL (spans both sides of y=0): allowed. Bouzidi
        # ray-casting uses the watertight full STL for correct inside/outside
        # classification; the half-domain (HALF_Y_DOMAIN=true) only allocates
        # cells in y>=0, and the slip BC at y=0 handles the symmetry plane.
        # This path used to work in earlier versions of the code.
        @printf("[Domain] symmetric_analysis=true with full STL (y=[%.3f, %.3f]); using watertight STL for Bouzidi, half-domain restricted to y>=0.\n", min_y, max_y)
    end

    if GROUND_ANALYSIS
        extent_y <= tol && error("symmetric_analysis: ground produced a near-zero y extent after rotation; check the STL orientation and scale.")
        @printf("[Domain] Ground mode bounds before anchoring: y=[%.6f, %.6f]\n", min_y, max_y)
    elseif abs(min_y) > tol
        @printf("[Domain] Half-domain placement: shifting mesh min-y %.6e to y=0\n", min_y)
    end
end

function print_placed_mesh_bounds(bounds, mesh_offset)
    placed_min = (bounds.min_bounds[1] + mesh_offset[1],
                  bounds.min_bounds[2] + mesh_offset[2],
                  bounds.min_bounds[3] + mesh_offset[3])
    placed_max = (bounds.max_bounds[1] + mesh_offset[1],
                  bounds.max_bounds[2] + mesh_offset[2],
                  bounds.max_bounds[3] + mesh_offset[3])

    @printf("[Domain] Placed mesh bounds: x=[%.6f, %.6f], y=[%.6f, %.6f], z=[%.6f, %.6f]\n",
            placed_min[1], placed_max[1], placed_min[2], placed_max[2], placed_min[3], placed_max[3])

    if HALF_Y_DOMAIN
        boundary_name = GROUND_ANALYSIS ? "no-slip ground wall" : "slip symmetry plane"
        @printf("[Domain] Half-domain y-min boundary: %s at y=%.6f\n", boundary_name, placed_min[2])
        if abs(placed_min[2]) > 1.0e-6
            @printf("[Domain] Warning: half-domain placement expected y_min=0, got %.6e\n", placed_min[2])
        end
    end
end

"""
    setup_multilevel_domain(runtime, claimed_output; num_levels=NUM_LEVELS_CONFIG)

Build a production domain from the immutable STL bytes captured in a
`RuntimeSelection`. No STL pathname is opened here. Subdivided diagnostics are
confined to the already-claimed fresh output directory and source-indexed so
same-named inputs cannot collide.
"""
function setup_multilevel_domain(runtime::CaseIDContract.RuntimeSelection,
                                 claimed_output::CaseIDContract.ClaimedOutput;
                                 num_levels=NUM_LEVELS_CONFIG,
                                 root_dir::AbstractString=dirname(dirname(runtime.case.case_dir)))
    CaseIDContract.verify_runtime_inputs(root_dir, runtime)
    CaseIDContract.verify_claimed_output(runtime, claimed_output)
    isempty(runtime.stls) && error("runtime STL selection must not be empty")

    meshes = Geometry.SolverMesh[]
    triangle_counts = Int[]
    source_names = String[]
    total_sources = length(runtime.stls)

    for (i, snapshot) in enumerate(runtime.stls)
        CaseIDContract.verify_claimed_output(runtime, claimed_output)
        source_name = basename(snapshot.path)
        push!(source_names, source_name)
        if total_sources > 1
            println("[Domain] Loading captured STL [$i/$total_sources]: $(snapshot.declared)")
        else
            println("[Domain] Loading captured STL: $(snapshot.declared)")
        end

        subdivision_output = if Float64(MINIMUM_FACET_SIZE) > 0.0
            indexed_name = "subdivided_$(lpad(string(i), 3, '0'))_$(source_name)"
            joinpath(claimed_output.target.path, indexed_name)
        else
            nothing
        end
        mesh = Geometry.load_mesh_bytes(snapshot.bytes;
                                        source_name=snapshot.declared,
                                        scale=Float64(STL_SCALE),
                                        min_facet_size=Float64(MINIMUM_FACET_SIZE))
        if subdivision_output !== nothing
            CaseIDContract.verify_claimed_output(runtime, claimed_output)
            Geometry.write_stl_binary_exclusive(subdivision_output, mesh)
        end
        push!(meshes, mesh)
        push!(triangle_counts, length(mesh.triangles))
    end

    mesh = Geometry.merge_meshes(meshes)
    object_names = Vector{String}(undef, length(mesh.triangles))
    offset = 0
    for (i, source_name) in enumerate(source_names)
        name = replace(source_name, r"\.[sS][tT][lL]$" => "")
        for j in 1:triangle_counts[i]
            object_names[offset + j] = name
        end
        offset += triangle_counts[i]
    end

    unrotated_mesh = mesh
    unrotated_centers = copy(mesh.centers)
    mesh = Geometry.rotate_mesh(mesh, ALPHA_DEG, BETA_DEG;
                                ground_mode=GROUND_ANALYSIS,
                                ground_up_axis=GROUND_UP_AXIS)
    if SYMMETRIC_ANALYSIS && mesh.min_bounds[2] < -1.0e-6
        mesh = Geometry.auto_slice_half_y(mesh)
    end

    validate_half_domain_mesh!(mesh)
    bounds = Geometry.compute_mesh_bounds(mesh)
    compute_domain_from_mesh(Tuple(bounds.min_bounds), Tuple(bounds.max_bounds))
    params = get_domain_params()
    print_placed_mesh_bounds(bounds, params.mesh_offset)
    print_domain_summary()

    return setup_multilevel_domain(mesh, params), mesh, unrotated_mesh,
           unrotated_centers, object_names
end
