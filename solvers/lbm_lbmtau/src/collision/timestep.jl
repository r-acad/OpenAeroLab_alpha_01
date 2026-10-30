using StaticArrays

# Global mass-conservation controller baseline (2026-05-29). Captured once
# after MASS_CONTROL_BASELINE_STEP; <=0 means "not yet captured". Module-level
# Ref so it persists across perform_timestep_v2! calls within a run.
# 2026-05-30 — RENAMED from _MASS_BASELINE to avoid a GLOBAL NAME COLLISION with
# diagnostics.jl's `const _MASS_BASELINE` (both files are included into the same
# module, so the two consts aliased the SAME Ref). The diagnostics module writes
# its ACTIVE-mass baseline (~162) into it, which the controller then read as its
# own baseline while measuring the full-array sum (~665600) → surplus_rel≈3973 →
# s saturated to the drain clamp every step → spurious −3.6% mass drain. The
# unique name keeps the two baselines independent.
const _MASS_CTRL_BASELINE = Ref{Float64}(0.0)

# 2026-05-30 — one-shot GPU register/occupancy probe for the flow kernel.
# When LUDWIG_KERNEL_PROBE=1, the first flow-kernel launch reports registers/thread,
# spill (local mem) bytes, and theoretical occupancy via CUDA.jl introspection
# (same data as Nsight Compute's register page; no ncu install needed). Reversible
# diagnostic — guarded so it fires exactly once, then the run proceeds normally.
const _KERNEL_PROBE_DONE = Ref{Bool}(false)
const _KERNEL_PROBE_DONE_L1 = Ref{Bool}(false)    # probe fires once per variant
const _KERNEL_PROBE_DONE_AMR = Ref{Bool}(false)
const _KERNEL_PROBE_DONE_AMR_INTERIOR = Ref{Bool}(false)
const _KERNEL_PROBE_DONE_AMR_SAMELEVEL_SHELL = Ref{Bool}(false)
const _KERNEL_PROBE_DONE_AMR_PHYSICAL_BOUNDARY = Ref{Bool}(false)
const _KERNEL_PROBE_DONE_AMR_HALO_MARK = Ref{Bool}(false)
const _KERNEL_PROBE_DONE_AMR_HALO_STAGE = Ref{Bool}(false)
const _KERNEL_PROBE_DONE_AMR_HALO_GENERIC_STAGE = Ref{Bool}(false)
const _KERNEL_PROBE_DONE_AMR_HALO_STAGE_DIR = Ref{Bool}(false)
const _KERNEL_PROBE_DONE_AMR_HALO_STAGE_PI_DIR = Ref{Bool}(false)
const _KERNEL_PROBE_DONE_AMR_HALO = Ref{Bool}(false)
const _KERNEL_PROBE_DONE_AMR_RESIDUAL_BOUNDARY = Ref{Bool}(false)
const _KERNEL_PROBE_ENABLED = get(ENV, "LUDWIG_KERNEL_PROBE", "0") == "1"
# (Dead D3Q19 halo-stage/residual-split env consts removed 2026-06-24.)
# SHARED-BSPLINE TILE (2026-06-10): when "1" AND the bspline-always macro mode
# is active (LUDWIG_D3Q19_BSPLINE_MACRO_TIME_CONSISTENT=0), the dense PI-dir
# stage kernel swaps to the workgroup-cooperative @localmem tile variant
# (~21× fewer global reads on the pristine-mode hot path). Default OFF.
const _D3Q19_BSPLINE_TILE = get(ENV, "LUDWIG_D3Q19_BSPLINE_TILE", "0") == "1"
# RESIDUAL PARENT-FILL (2026-06-10): when "1", the residual-boundary kernel
# prolongates from the parent for directions whose source lies outside the
# refinement box (instead of returning rest-state w_k — the box-edge∩domain-
# plane seam bug, see HANDOVER_2026-06-10). Default OFF = legacy byte-identical.
const _D3Q19_RESIDUAL_PARENT_FILL = get(ENV, "LUDWIG_D3Q19_RESIDUAL_PARENT_FILL", "0") == "1"
# RANGED RESIDUAL LAUNCH (2026-06-10b): launch the residual-boundary kernel only
# over DOMAIN-BOUNDARY blocks — the only blocks that can contain cells with
# is_domain_shell=true, which is the kernel's entire process_cell set
# (RESIDUAL_ONLY_VAL=true). Bit-identical by construction (threads in interior
# blocks were early-out no-ops). Implemented as a BLOCK-ID LIST indirection
# (single launch over (8,8,8,n_boundary), kernel maps b_local → list[b_local],
# same pattern as the sparse-PI staging): a per-interval sub-launch variant was
# measured 3.5× SLOWER than dense (16 intervals × 120 substeps of host launch
# overhead on the 70+-arg ABI). Removes the fat-ABI full-ndrange idle-thread
# cost in RESIDUAL_PARENT_FILL (pristine) mode.
# Default ON; "0" restores the full-ndrange launch.
# (Dead D3Q19 residual-ranged + mirror-reflect-fix env consts removed 2026-06-24.)
# (Dead D3Q19 halo-PI tile / corner-tile / src-dedup / cubic-Π-dispatch env consts removed 2026-06-24.)
# 100-MLUPS pineq lever (2026-06-13): FUSED-Π — the post-step refresh
# (~3.3 s/batch, a separate 19-read dense pass) is folded into the kernels
# that write the FINAL f for each cell. Because SVV is the last f-writer on
# every SVV-active level (and conserves ρ+momentum so post-SVV ρ/u ==
# stored), the SVV epilogue computes level.pi_neq for every SVV-active fluid
# cell; the 5 collision kernels compute it for obstacle cells + cells on
# SVV-inactive levels. The post-step refresh is then skipped (the restrict-
# side banded refresh stays).
# DEFAULT ON 2026-06-13: ULP-equivalent (not bitwise — cross-kernel
# FMA-context, union-tile class). USER FIELD-CHECK PASSED on Z197 (production
# SR=235 sphere): max per-pixel diff ≤2/255 on every field, ULP-noise follows
# flow gradients with NO seam/corner structure (CASE_LOGS/fieldcheck_*). Seam
# baselines re-pinned to the fused-ON values. "0" reverts to the dense refresh.
const _D3Q19_FUSE_PINEQ = get(ENV, "LUDWIG_D3Q19_FUSE_PINEQ", "1") != "0"
# FUSED-Π Bouzidi mop-up (2026-06-13): Bouzidi correction modifies body-cell f
# AFTER the SVV epilogue writes their Π, so those cells' fused Π is stale. The
# Bouzidi cells are a static flat list (level.bouzidi_cell_block); cache the
# UNIQUE sorted block ids once per level and re-run the (banded) refresh over
# them after Bouzidi — idempotent (post-Bouzidi f ⇒ exact Π for those blocks),
# bitwise-correct, leaves the SVV-fused Π untouched elsewhere. Device list cached.
const _D3Q19_BOUZIDI_BLOCK_CACHE = IdDict{Any, Any}()
function _d3q19_bouzidi_block_list(level, backend)
    cached = get(_D3Q19_BOUZIDI_BLOCK_CACHE, level, nothing)
    cached !== nothing && return cached
    blocks_host = Array(level.bouzidi_cell_block)
    n = level.n_boundary_cells
    uniq = sort!(unique(Int32.(@view blocks_host[1:n])))
    dev = KernelAbstractions.allocate(backend, Int32, length(uniq))
    length(uniq) > 0 && copyto!(dev, uniq)
    entry = (list = dev, n = length(uniq))
    _D3Q19_BOUZIDI_BLOCK_CACHE[level] = entry
    return entry
end
# (Dead D3Q19 mark-once / halo-hoist / residual-stage-fill env consts + caches removed 2026-06-24.)
# z-PARITY FIX attempt 2 (2026-06-11, round 7): EQUILIBRIUM-BLEND DAMPING at
# plane∩box-edge cells. λ ∈ [0,1] blend of the boundary-filled populations
# toward the cell's local equilibrium (prev-step ρ, u — parity-neutral by
# lattice symmetry), per pull class:
#   _PFILL — parent-prolongated box-void pulls at cells ON a free-slip mirror
#            row. The principled analogue of the quick preset's rest-state w_k
#            injection (the ONLY parity-clean arm in the round-4 truth table,
#            HANDOVER_2026-06-10c §6) with correct local macros.
#   _SPEC  — wrong-cell same-cell-specular COMBINED pulls whose reflected
#            source has no same-level data.
# Default 0.0 = OFF (bitwise legacy). Gates: Z201/Z203 MirrorProbe step-30/60
# ≤ ~1e-5 class AND Z156/Z160 pristine seam not regressed.
const _D3Q19_MIRROR_EQBLEND_PFILL =
    something(tryparse(Float32, get(ENV, "LUDWIG_D3Q19_MIRROR_EQBLEND_PFILL", "0")), 0.0f0)
const _D3Q19_MIRROR_EQBLEND_SPEC =
    something(tryparse(Float32, get(ENV, "LUDWIG_D3Q19_MIRROR_EQBLEND_SPEC", "0")), 0.0f0)
# Blend TARGET for the eqblend lever. The Z201 probe showed feq(local) damps
# only ~2-3× — the cell's own macros re-inject the parity perturbation (self-
# coupling gain ∂feq/∂u). State-independent targets break the loop the way
# quick's constant w_k does, without its wrong-macros seam bug:
#   local      (0) — feq(prev-step cell ρ,u). Z201-clean in B-bundle, but a
#                    ZERO-GRADIENT fill: lags the streamwise ramp along the
#                    upstream box edge → the round-7b corner quad (Z197).
#   freestream (1) — feq(ρ=1, u=(u_current,0,0)): fully state-independent.
#   parenteq   (2) — equilibrium-only parent fill (entropic_mode=1 in
#                    interpolate_with_rescaling): feq at the C4-interpolated
#                    parent macros, state-independent w.r.t. the fine level.
#                    λ acts as a BINARY gate in this mode. Z201-DIRTY.
#   localgrad  (3) — round 7b, REFUTED ON Z201 (kept for the record): feq at
#                    gradient-extrapolated local macros. BOTH variants are
#                    unstable — v2a (2ρ_cell − ρ_opp, own-gain 2) DIVERGED
#                    (ρ→1e11 by step 60); v2b (cell + (ρ_{+c} − ρ_{+2c}),
#                    own-gain 1) reached order-1 corruption by step 30. ANY
#                    spatial-difference content in the ring fill re-excites
#                    the amplification loop; only the gain-1 cell-anchored
#                    `local` (and constant `freestream`) fills are stable.
const _D3Q19_MIRROR_EQBLEND_TARGET = let s = lowercase(get(ENV, "LUDWIG_D3Q19_MIRROR_EQBLEND_TARGET", "local"))
    s == "freestream" ? Int32(1) : s == "parenteq" ? Int32(2) : s == "localgrad" ? Int32(3) : Int32(0)
end
# Damping SCOPE (round 7b): which ring cells get the PFILL blend.
#   row    (0) — every cell on a free-slip mirror row (round-7 behavior).
#                Parity-clean, but the feq fill along ENTIRE box faces lags
#                the streamwise ramp → the visible upstream-edge line + the
#                user-reported corner quad (Z197 step-50).
#   corner (1) — only ring cells with BOX-VOID pulls in ≥ 2 axis directions.
#                REFUTED ON Z201 (kept for the record): 3.15e-03 ≈ the
#                no-lever B-bundle — the amplification loop runs along the
#                ENTIRE face∩plane ring (the argmax just moves one cell off
#                the damped corner). Full-row damping is REQUIRED for parity.
const _D3Q19_MIRROR_EQBLEND_SCOPE =
    lowercase(get(ENV, "LUDWIG_D3Q19_MIRROR_EQBLEND_SCOPE", "row")) == "corner" ? Int32(1) : Int32(0)
if _D3Q19_MIRROR_EQBLEND_PFILL > 0.0f0 || _D3Q19_MIRROR_EQBLEND_SPEC > 0.0f0
    println("[D3Q19] mirror EQUILIBRIUM-BLEND lever ACTIVE: " *
            "spec=$(_D3Q19_MIRROR_EQBLEND_SPEC) pfill=$(_D3Q19_MIRROR_EQBLEND_PFILL) " *
            "target=$(_D3Q19_MIRROR_EQBLEND_TARGET) (0=local,1=freestream,2=parenteq; " *
            "plane∩box-edge damping, round 7)")
end
# (Dead D3Q19 L1-cell-split / halo-fallback / pi-sparse / generic-stage env consts +
#  interval/ready/ET caches removed 2026-06-24.)
# Forced-occupancy register cap (LUDWIG_MAXREGS=N). 0 = off (normal KA launch).
# N≤128 → compiler fits ≥2 blocks/SM (more occupancy, more spill). SPEED experiment.
const _MAXREGS_CAP = parse(Int, get(ENV, "LUDWIG_MAXREGS", "0"))
# ── Per-phase timers (LUDWIG_PHASE_TIMING=1) — find where the per-step time goes
# (flow kernel vs thermal kernel vs AMR restrict). Accumulate seconds; main loop
# prints + resets. Reveals whether the flow-kernel register work was even the
# right target, or if thermal/AMR/sub-cycling dominates. ──
const _PHASE_TIMING = get(ENV, "LUDWIG_PHASE_TIMING", "0") == "1"
const _T_FLOW = Ref{Float64}(0.0)
const _T_THERMAL = Ref{Float64}(0.0)
const _T_BC = Ref{Float64}(0.0)
const _T_FLOW_CALLS = Ref{Int}(0)
const _PHASE_LEVEL_MAX = 12
const _T_SPLIT_L1 = zeros(Float64, _PHASE_LEVEL_MAX)
const _T_SPLIT_AMR_INTERIOR = zeros(Float64, _PHASE_LEVEL_MAX)
const _T_SPLIT_AMR_SAMELEVEL = zeros(Float64, _PHASE_LEVEL_MAX)
const _T_SPLIT_AMR_PHYSICAL = zeros(Float64, _PHASE_LEVEL_MAX)
const _T_SPLIT_AMR_RESIDUAL = zeros(Float64, _PHASE_LEVEL_MAX)
const _T_SPLIT_AMR_HALO_MARK = zeros(Float64, _PHASE_LEVEL_MAX)
const _T_SPLIT_AMR_HALO_STAGE = zeros(Float64, _PHASE_LEVEL_MAX)
const _T_SPLIT_AMR_HALO_STAGE_DIR = zeros(Float64, _PHASE_LEVEL_MAX)
const _T_SPLIT_AMR_HALO_STAGE_PI_DIR = zeros(Float64, _PHASE_LEVEL_MAX)
const _T_SPLIT_AMR_HALO_COLLIDE = zeros(Float64, _PHASE_LEVEL_MAX)
const _T_SPLIT_AMR_HALO_GENERIC = zeros(Float64, _PHASE_LEVEL_MAX)
const _T_SPLIT_AMR_HALO_FALLBACK = zeros(Float64, _PHASE_LEVEL_MAX)
const _T_SPLIT_SVV = zeros(Float64, _PHASE_LEVEL_MAX)
const _T_SPLIT_WALL = zeros(Float64, _PHASE_LEVEL_MAX)
# Coarse-grained non-flow phase accumulators (2026-06-09 profiling). Track the
# big per-substep operations that live in the "AMR/BC/other" remainder so the
# bottleneck inside that ~46% bucket is visible. Gated on _PHASE_TIMING.
const _T_OP_RESTRICT = Ref{Float64}(0.0)
const _T_OP_SPLINE = Ref{Float64}(0.0)
const _T_OP_BOUZIDI = Ref{Float64}(0.0)
const _T_OP_PINEQ = Ref{Float64}(0.0)
const _T_OP_COPYOLD = Ref{Float64}(0.0)
@inline function _phase_op_add!(bucket::Ref{Float64}, dt)
    if _PHASE_TIMING; bucket[] += dt; end
    return nothing
end

@inline function _phase_level_idx(level_id)
    return min(max(Int(level_id), 1), _PHASE_LEVEL_MAX)
end

@inline function _phase_add!(bucket, level_id, dt)
    bucket[_phase_level_idx(level_id)] += dt
    return nothing
end

function _phase_reset_split_timers!()
    fill!(_T_SPLIT_L1, 0.0)
    fill!(_T_SPLIT_AMR_INTERIOR, 0.0)
    fill!(_T_SPLIT_AMR_SAMELEVEL, 0.0)
    fill!(_T_SPLIT_AMR_PHYSICAL, 0.0)
    fill!(_T_SPLIT_AMR_RESIDUAL, 0.0)
    fill!(_T_SPLIT_AMR_HALO_MARK, 0.0)
    fill!(_T_SPLIT_AMR_HALO_STAGE, 0.0)
    fill!(_T_SPLIT_AMR_HALO_STAGE_DIR, 0.0)
    fill!(_T_SPLIT_AMR_HALO_STAGE_PI_DIR, 0.0)
    fill!(_T_SPLIT_AMR_HALO_COLLIDE, 0.0)
    fill!(_T_SPLIT_AMR_HALO_GENERIC, 0.0)
    fill!(_T_SPLIT_AMR_HALO_FALLBACK, 0.0)
    fill!(_T_SPLIT_SVV, 0.0)
    fill!(_T_SPLIT_WALL, 0.0)
    return nothing
end

function _phase_print_split_timers!()
    labels = (
        ("L1", _T_SPLIT_L1),
        ("amr_int", _T_SPLIT_AMR_INTERIOR),
        ("same", _T_SPLIT_AMR_SAMELEVEL),
        ("phys", _T_SPLIT_AMR_PHYSICAL),
        ("resid", _T_SPLIT_AMR_RESIDUAL),
        ("h_mark", _T_SPLIT_AMR_HALO_MARK),
        ("h_stage", _T_SPLIT_AMR_HALO_STAGE),
        ("h_dir", _T_SPLIT_AMR_HALO_STAGE_DIR),
        ("h_pi", _T_SPLIT_AMR_HALO_STAGE_PI_DIR),
        ("h_coll", _T_SPLIT_AMR_HALO_COLLIDE),
        ("h_gen", _T_SPLIT_AMR_HALO_GENERIC),
        ("h_fb", _T_SPLIT_AMR_HALO_FALLBACK),
        ("svv", _T_SPLIT_SVV),
        ("wall", _T_SPLIT_WALL),
    )
    printed = false
    for lvl in 1:_PHASE_LEVEL_MAX
        level_total = 0.0
        for (_, bucket) in labels
            level_total += bucket[lvl]
        end
        if level_total > 1.0e-6
            if !printed
                println("    [SplitTiming] per-level kernel seconds:")
                printed = true
            end
            @printf("      L%-2d total=%7.2fs", lvl, level_total)
            for (name, bucket) in labels
                val = bucket[lvl]
                if val > 1.0e-4
                    @printf(" %s=%.2f", name, val)
                end
            end
            println()
        end
    end
    return nothing
end

@inline function _host_d3q19_samelevel_shell_ready(
    ix::Int, iy::Int, iz::Int,
    gx::Int, gy::Int, gz::Int,
    b_idx::Int,
    block_size::Int,
    nx_global::Int, ny_global::Int, nz_global::Int,
    neighbor_table,
)
    is_block_shell =
        ix == 1 || ix == block_size ||
        iy == 1 || iy == block_size ||
        iz == 1 || iz == block_size
    is_block_shell || return false
    if gx <= 1 || gx >= nx_global ||
       gy <= 1 || gy >= ny_global ||
       gz <= 1 || gz >= nz_global
        return false
    end
    @inbounds for k in 1:19
        sx = ix - Int(C_X[k])
        sy = iy - Int(C_Y[k])
        sz = iz - Int(C_Z[k])
        if sx < 1 || sx > block_size ||
           sy < 1 || sy > block_size ||
           sz < 1 || sz > block_size
            nb_off_x = sx < 1 ? -1 : (sx > block_size ? 1 : 0)
            nb_off_y = sy < 1 ? -1 : (sy > block_size ? 1 : 0)
            nb_off_z = sz < 1 ? -1 : (sz > block_size ? 1 : 0)
            dir_idx = (nb_off_x + 1) + (nb_off_y + 1) * 3 + (nb_off_z + 1) * 9 + 1
            neighbor_table[b_idx, dir_idx] > Int32(0) || return false
        end
    end
    return true
end

@inline function _host_d3q19_bspline_macro_ready(
    parent_ptr,
    parent_dim_x::Int, parent_dim_y::Int, parent_dim_z::Int,
    fine_gx::Int, fine_gy::Int, fine_gz::Int,
    block_size::Int,
)
    p0x = div(fine_gx + 1, 2)
    p0y = div(fine_gy + 1, 2)
    p0z = div(fine_gz + 1, 2)
    @inbounds for iz in -3:3
        pgz = p0z + iz
        pbz = div(pgz - 1, block_size) + 1
        (1 <= pbz <= parent_dim_z) || return false
        for iy in -3:3
            pgy = p0y + iy
            pby = div(pgy - 1, block_size) + 1
            (1 <= pby <= parent_dim_y) || return false
            for ix in -3:3
                pgx = p0x + ix
                pbx = div(pgx - 1, block_size) + 1
                (1 <= pbx <= parent_dim_x) || return false
                parent_ptr[pbx, pby, pbz] > Int32(0) || return false
            end
        end
    end
    return true
end

@inline function _host_d3q19_bspline_halo_ready(
    ix::Int, iy::Int, iz::Int,
    gx::Int, gy::Int, gz::Int,
    b_idx::Int,
    block_size::Int,
    neighbor_table,
    parent_ptr,
    parent_dim_x::Int, parent_dim_y::Int, parent_dim_z::Int,
)
    @inbounds for k in 1:19
        cx = Int(C_X[k])
        cy = Int(C_Y[k])
        cz = Int(C_Z[k])
        sx = ix - cx
        sy = iy - cy
        sz = iz - cz
        if sx < 1 || sx > block_size ||
           sy < 1 || sy > block_size ||
           sz < 1 || sz > block_size
            nb_off_x = sx < 1 ? -1 : (sx > block_size ? 1 : 0)
            nb_off_y = sy < 1 ? -1 : (sy > block_size ? 1 : 0)
            nb_off_z = sz < 1 ? -1 : (sz > block_size ? 1 : 0)
            dir_idx = (nb_off_x + 1) + (nb_off_y + 1) * 3 + (nb_off_z + 1) * 9 + 1
            if neighbor_table[b_idx, dir_idx] <= Int32(0)
                _host_d3q19_bspline_macro_ready(
                    parent_ptr,
                    parent_dim_x, parent_dim_y, parent_dim_z,
                    gx - cx, gy - cy, gz - cz,
                    block_size,
                ) || return false
            end
        end
    end
    return true
end

# (D3Q19 halo-fallback / residual / pi-ready host-helper cluster removed 2026-06-24 — closed dead cluster after D3Q19 retirement.)
@kernel function _update_block_shift_kernel!(block_shift, block_shift_next, vel, obstacle, amr_covered, sponge, wall_dist,
                                             local_shift_wall_guard, map_x,
                                             n_blocks::Int32, block_size::Int32,
                                             u_max::Int32, threshold::Float32,
                                             hysteresis::Float32, u_ref::Float32,
                                              reference_shift_x::Int32,
                                              wall_guard_cells::Float32,
                                              wall_zero_guard_cells::Float32,
                                              zero_allowed_x_min_cells::Float32,
                                              wake_zero_deficit_frac::Float32,
                                              high_speed_frac_trigger::Float32,
                                              geometric_band_cells::Float32)
    b = @index(Global)
    if b <= n_blocks
        sum_ux = 0.0f0
        sum_uy = 0.0f0
        sum_uz = 0.0f0
        count = Int32(0)
        high_x_count = Int32(0)
        high_yp_count = Int32(0)
        high_ym_count = Int32(0)
        high_zp_count = Int32(0)
        high_zm_count = Int32(0)
        max_yp = 0.0f0
        max_ym = 0.0f0
        max_zp = 0.0f0
        max_zm = 0.0f0
        deficit_count = Int32(0)
        min_ux = 1.0f9
        max_ux = -1.0f9
        min_wall_dist = 1.0f9
        max_sponge = 0.0f0
        high_cut = threshold + hysteresis
        trans_cut = max(0.48f0, threshold - 0.08f0)
        deficit_cut = max(threshold, 0.85f0 * u_ref)
        @inbounds for lz in Int32(1):block_size, ly in Int32(1):block_size, lx in Int32(1):block_size
            if !obstacle[lx, ly, lz, b] && !amr_covered[lx, ly, lz, b]
                sp_cell = sponge[lx, ly, lz, b]
                max_sponge = max(max_sponge, sp_cell)
                ux_cell = vel[lx, ly, lz, b, Int32(1)]
                uy_cell = vel[lx, ly, lz, b, Int32(2)]
                uz_cell = vel[lx, ly, lz, b, Int32(3)]
                sum_ux += ux_cell
                sum_uy += uy_cell
                sum_uz += uz_cell
                min_ux = min(min_ux, ux_cell)
                max_ux = max(max_ux, ux_cell)
                max_yp = max(max_yp, uy_cell)
                max_ym = max(max_ym, -uy_cell)
                max_zp = max(max_zp, uz_cell)
                max_zm = max(max_zm, -uz_cell)
                ux_cell >= high_cut && (high_x_count += Int32(1))
                uy_cell >= trans_cut && (high_yp_count += Int32(1))
                uy_cell <= -trans_cut && (high_ym_count += Int32(1))
                uz_cell >= trans_cut && (high_zp_count += Int32(1))
                uz_cell <= -trans_cut && (high_zm_count += Int32(1))
                ux_cell < deficit_cut && (deficit_count += Int32(1))
                wd = wall_dist[lx, ly, lz, b]
                if wd == wd && wd > 0.0f0
                    min_wall_dist = min(min_wall_dist, wd)
                end
                count += Int32(1)
            end
        end

        if count > 0
            ux_mean = sum_ux / Float32(count)
            uy_mean = sum_uy / Float32(count)
            uz_mean = sum_uz / Float32(count)
            high_x_frac = Float32(high_x_count) / Float32(count)
            high_yp_frac = Float32(high_yp_count) / Float32(count)
            high_ym_frac = Float32(high_ym_count) / Float32(count)
            high_zp_frac = Float32(high_zp_count) / Float32(count)
            high_zm_frac = Float32(high_zm_count) / Float32(count)
            deficit_frac = Float32(deficit_count) / Float32(count)
            prev_code = block_shift[b]
            prev_x = local_shift_decode_x_d3q39(prev_code)
            prev_y = local_shift_decode_y_d3q39(prev_code)
            prev_z = local_shift_decode_z_d3q39(prev_code)
            next_code = Int32(0)
            # Optional stationary-wall guard. A zero width must be a true
            # local-shift run: forcing every Bouzidi-owning block to the
            # reference frame creates a square U=1 collar around curved bodies.
            owns_wall_links = local_shift_wall_guard[b] > Int32(0)
            near_stationary_wall = wall_guard_cells > 0.0f0 &&
                                   (owns_wall_links || min_wall_dist <= wall_guard_cells)
            near_stationary_wall_zero = wall_zero_guard_cells > 0.0f0 &&
                                        (owns_wall_links || min_wall_dist <= wall_zero_guard_cells)
            if near_stationary_wall_zero || max_sponge > 0.02f0
                next_code = Int32(0)
            elseif near_stationary_wall
                next_code = local_shift_encode3_d3q39(clamp(reference_shift_x, Int32(0), u_max), Int32(0), Int32(0))
            else
                coherent_x_frac = max(high_speed_frac_trigger, 0.12f0)
                x_pocket_frac = max(0.018f0, 0.15f0 * coherent_x_frac)
                x_pocket_speed = max(0.74f0, high_cut + 0.06f0)
                x_pocket_mean_floor = max(0.34f0, 0.62f0 * max(u_ref, 0.1f0))
                trans_on_frac = max(0.18f0, min(high_speed_frac_trigger, 0.35f0))
                trans_keep_frac = max(0.10f0, 0.65f0 * trans_on_frac)
                transverse_mean_floor = max(0.22f0, 0.40f0 * max(u_ref, 0.1f0))
                trans_pocket_frac = max(0.004f0, 0.025f0 * trans_on_frac)
                trans_pocket_speed = max(0.56f0, threshold - 0.02f0)
                trans_pocket_mean_floor = -0.035f0

                best_score = 0.0f0
                best_code = Int32(0)
                # Transverse shoulder spikes are sub-block stress/positivity events,
                # not coherent advection-frame changes. Promoting an entire 8^3
                # block to Uy/Uz=±1 creates the vertical frame seams seen in Z110/Z111.
                # Keep local shifting streamwise-only until transverse relabel has a
                # conservative cell-local transition path.
                enable_transverse_shift = false

                if u_ref >= max(0.35f0, threshold - hysteresis) &&
                   ux_mean >= threshold &&
                   high_x_frac >= coherent_x_frac &&
                   deficit_frac < 0.35f0
                    best_score = high_x_frac * max(ux_mean, 0.0f0)
                    best_code = local_shift_encode3_d3q39(Int32(1), Int32(0), Int32(0))
                end
                x_pocket_score = high_x_frac * max_ux
                if u_ref >= max(0.42f0, threshold - hysteresis) &&
                   high_x_frac >= x_pocket_frac &&
                   max_ux >= x_pocket_speed &&
                   ux_mean >= x_pocket_mean_floor &&
                   deficit_frac < 0.38f0 &&
                   x_pocket_score > best_score
                    best_score = x_pocket_score
                    best_code = local_shift_encode3_d3q39(Int32(1), Int32(0), Int32(0))
                end

                score_yp = high_yp_frac * max(uy_mean, 0.0f0)
                if enable_transverse_shift && u_ref >= 0.35f0 && high_yp_frac >= trans_on_frac &&
                   uy_mean >= transverse_mean_floor && score_yp > best_score
                    best_score = score_yp
                    best_code = local_shift_encode3_d3q39(Int32(0), Int32(1), Int32(0))
                end
                pocket_score_yp = high_yp_frac * max_yp
                if enable_transverse_shift && u_ref >= 0.42f0 && high_yp_frac >= trans_pocket_frac &&
                   max_yp >= trans_pocket_speed && uy_mean >= trans_pocket_mean_floor &&
                   pocket_score_yp > best_score
                    best_score = pocket_score_yp
                    best_code = local_shift_encode3_d3q39(Int32(0), Int32(1), Int32(0))
                end
                score_ym = high_ym_frac * max(-uy_mean, 0.0f0)
                if enable_transverse_shift && u_ref >= 0.35f0 && high_ym_frac >= trans_on_frac &&
                   -uy_mean >= transverse_mean_floor && score_ym > best_score
                    best_score = score_ym
                    best_code = local_shift_encode3_d3q39(Int32(0), Int32(-1), Int32(0))
                end
                pocket_score_ym = high_ym_frac * max_ym
                if enable_transverse_shift && u_ref >= 0.42f0 && high_ym_frac >= trans_pocket_frac &&
                   max_ym >= trans_pocket_speed && -uy_mean >= trans_pocket_mean_floor &&
                   pocket_score_ym > best_score
                    best_score = pocket_score_ym
                    best_code = local_shift_encode3_d3q39(Int32(0), Int32(-1), Int32(0))
                end
                score_zp = high_zp_frac * max(uz_mean, 0.0f0)
                if enable_transverse_shift && u_ref >= 0.35f0 && high_zp_frac >= trans_on_frac &&
                   uz_mean >= transverse_mean_floor && score_zp > best_score
                    best_score = score_zp
                    best_code = local_shift_encode3_d3q39(Int32(0), Int32(0), Int32(1))
                end
                pocket_score_zp = high_zp_frac * max_zp
                if enable_transverse_shift && u_ref >= 0.42f0 && high_zp_frac >= trans_pocket_frac &&
                   max_zp >= trans_pocket_speed && uz_mean >= trans_pocket_mean_floor &&
                   pocket_score_zp > best_score
                    best_score = pocket_score_zp
                    best_code = local_shift_encode3_d3q39(Int32(0), Int32(0), Int32(1))
                end
                score_zm = high_zm_frac * max(-uz_mean, 0.0f0)
                if enable_transverse_shift && u_ref >= 0.35f0 && high_zm_frac >= trans_on_frac &&
                   -uz_mean >= transverse_mean_floor && score_zm > best_score
                    best_score = score_zm
                    best_code = local_shift_encode3_d3q39(Int32(0), Int32(0), Int32(-1))
                end
                pocket_score_zm = high_zm_frac * max_zm
                if enable_transverse_shift && u_ref >= 0.42f0 && high_zm_frac >= trans_pocket_frac &&
                   max_zm >= trans_pocket_speed && -uz_mean >= trans_pocket_mean_floor &&
                   pocket_score_zm > best_score
                    best_score = pocket_score_zm
                    best_code = local_shift_encode3_d3q39(Int32(0), Int32(0), Int32(-1))
                end

                keep_prev = false
                if prev_x == Int32(1)
                    keep_prev = (
                        high_x_frac >= max(0.08f0, 0.50f0 * coherent_x_frac) &&
                        ux_mean >= threshold - hysteresis &&
                        deficit_frac < 0.45f0
                    ) || (
                        high_x_frac >= 0.70f0 * x_pocket_frac &&
                        max_ux >= x_pocket_speed - hysteresis &&
                        ux_mean >= x_pocket_mean_floor - 0.05f0 &&
                        deficit_frac < 0.45f0
                    )
                elseif prev_y == Int32(1)
                    keep_prev = high_yp_frac >= trans_keep_frac && uy_mean >= 0.75f0 * transverse_mean_floor
                elseif prev_y == Int32(-1)
                    keep_prev = high_ym_frac >= trans_keep_frac && -uy_mean >= 0.75f0 * transverse_mean_floor
                elseif prev_z == Int32(1)
                    keep_prev = high_zp_frac >= trans_keep_frac && uz_mean >= 0.75f0 * transverse_mean_floor
                elseif prev_z == Int32(-1)
                    keep_prev = high_zm_frac >= trans_keep_frac && -uz_mean >= 0.75f0 * transverse_mean_floor
                end

                next_code = (best_score > 0.0f0) ? best_code : (keep_prev ? prev_code : Int32(0))
            end

            zero_allowed = Int32(1)
            if zero_allowed_x_min_cells > 0.0f0
                block_center_x = Float32((map_x[b] - Int32(1)) * block_size) + 0.5f0 * Float32(block_size)
                zero_allowed = block_center_x >= zero_allowed_x_min_cells ? Int32(1) : Int32(0)
            end

            slow_core = min_ux < max(0.20f0, threshold - hysteresis)
            wake_mean_low = ux_mean < max(0.30f0, 0.75f0 * u_ref)
            wake_deficit_enough = deficit_frac >= wake_zero_deficit_frac
            wake_deficit_strong = deficit_frac >= max(0.25f0, 2.0f0 * wake_zero_deficit_frac)
            x_pocket_zero_protect =
                deficit_frac < 0.30f0 &&
                ux_mean >= max(0.34f0, 0.62f0 * max(u_ref, 0.1f0)) &&
                high_x_frac >= max(0.018f0, 0.15f0 * max(high_speed_frac_trigger, 0.12f0)) &&
                max_ux >= max(0.74f0, high_cut + 0.06f0)
            if zero_allowed == Int32(1) && wake_zero_deficit_frac > 0.0f0 &&
               slow_core && (wake_deficit_strong || (wake_deficit_enough && wake_mean_low)) &&
               high_x_frac < 0.70f0 &&
               !x_pocket_zero_protect &&
               local_shift_decode_x_d3q39(next_code) != Int32(0)
                next_code = Int32(0)
            end

            if zero_allowed == Int32(0) && u_ref >= high_cut
                ref_frame = clamp(reference_shift_x, Int32(0), u_max)
                if ref_frame > Int32(0) && next_code == Int32(0)
                    next_code = local_shift_encode3_d3q39(ref_frame, Int32(0), Int32(0))
                end
            end
            # ── STATIC GEOMETRIC FRAME MAP (2026-06-27) ──
            # When geometric_band_cells > 0, IGNORE all the velocity-driven scoring above
            # and assign the frame PURELY by geometry: U=0 in a fixed shell of `band` cells
            # around the body (covers the bow-shock standoff + the subsonic stagnation), and
            # U=u_max everywhere else (the supersonic freestream and far wake). This breaks the
            # dynamic map's upstream-marching collapse — the supersonic stream can no longer
            # flip pre-shock blocks to U=0, so the shock forms stably at the band's outer edge
            # (tune `band` ≈ the natural bow-shock standoff). The velocity scoring still runs
            # (cheap, per-block) but its result is discarded here.
            if geometric_band_cells > 0.0f0
                in_band = (min_wall_dist <= geometric_band_cells) ||
                          (max_sponge > 0.02f0) ||
                          (local_shift_wall_guard[b] > Int32(0))
                # GEOMETRY-BOUNDED MACH GATE (2026-06-29): assign U=0 only where the block is
                # geometrically near the body AND locally SUBSONIC (mean speed < `threshold`,
                # set to ~the lattice sound speed). This keeps pre-shock SUPERSONIC blocks
                # inside the band as U=1 — otherwise their rest-frame u_rel≈u_lattice sits at
                # the positive-equilibrium manifold edge and seeds the grid-scale speckle. The
                # geometric band still BOUNDS the U=0 region: a far-upstream block, even if it
                # slows, is outside the band and can never flip to U=0, so the dynamic map's
                # upstream-feedback collapse cannot occur. The U=0/U=1 boundary lands at the
                # sonic line. threshold <= 0 ⇒ legacy pure-geometric (every in-band block U=0).
                blk_speed = sqrt(ux_mean * ux_mean + uy_mean * uy_mean + uz_mean * uz_mean)
                u0_here = in_band && (threshold <= 0.0f0 || blk_speed < threshold)
                next_code = u0_here ? Int32(0) :
                            local_shift_encode3_d3q39(clamp(u_max, Int32(0), u_max), Int32(0), Int32(0))
            end
            block_shift_next[b] = next_code
        else
            block_shift_next[b] = Int32(0)
        end
    end
end

@kernel function _dilate_transverse_block_shift_kernel!(block_shift_next, block_shift_dilated,
                                                        neighbor_table, sponge, wall_dist,
                                                        local_shift_wall_guard,
                                                        n_blocks::Int32, block_size::Int32,
                                                        wall_zero_guard_cells::Float32)
    b = @index(Global)
    if b <= n_blocks
        @inbounds begin
            min_wall_dist = 1.0f9
            max_sponge = 0.0f0
            for lz in Int32(1):block_size, ly in Int32(1):block_size, lx in Int32(1):block_size
                max_sponge = max(max_sponge, sponge[lx, ly, lz, b])
                wd = wall_dist[lx, ly, lz, b]
                if wd == wd && wd > 0.0f0
                    min_wall_dist = min(min_wall_dist, wd)
                end
            end

            owns_wall_links = local_shift_wall_guard[b] > Int32(0)
            protected = max_sponge > 0.02f0 ||
                        (wall_zero_guard_cells > 0.0f0 &&
                         (owns_wall_links || min_wall_dist <= wall_zero_guard_cells))
            if protected
                block_shift_dilated[b] = Int32(0)
            else
                code = block_shift_next[b]
                code_y = local_shift_decode_y_d3q39(code)
                code_z = local_shift_decode_z_d3q39(code)
                if code_y != Int32(0) || code_z != Int32(0)
                    block_shift_dilated[b] = code
                else
                    count_yp = Int32(0)
                    count_ym = Int32(0)
                    count_zp = Int32(0)
                    count_zm = Int32(0)
                    for ii in Int32(1):Int32(6)
                        dir_idx = ii == Int32(1) ? Int32(5)  :
                                  ii == Int32(2) ? Int32(11) :
                                  ii == Int32(3) ? Int32(13) :
                                  ii == Int32(4) ? Int32(15) :
                                  ii == Int32(5) ? Int32(17) : Int32(23)
                        nb = neighbor_table[b, dir_idx]
                        if nb > Int32(0)
                            ncode = block_shift_next[nb]
                            ny = local_shift_decode_y_d3q39(ncode)
                            nz = local_shift_decode_z_d3q39(ncode)
                            if ny == Int32(1) && nz == Int32(0)
                                count_yp += Int32(1)
                            elseif ny == Int32(-1) && nz == Int32(0)
                                count_ym += Int32(1)
                            elseif nz == Int32(1) && ny == Int32(0)
                                count_zp += Int32(1)
                            elseif nz == Int32(-1) && ny == Int32(0)
                                count_zm += Int32(1)
                            end
                        end
                    end

                    best_count = count_yp
                    best_code = local_shift_encode3_d3q39(Int32(0), Int32(1), Int32(0))
                    if count_ym > best_count
                        best_count = count_ym
                        best_code = local_shift_encode3_d3q39(Int32(0), Int32(-1), Int32(0))
                    end
                    if count_zp > best_count
                        best_count = count_zp
                        best_code = local_shift_encode3_d3q39(Int32(0), Int32(0), Int32(1))
                    end
                    if count_zm > best_count
                        best_count = count_zm
                        best_code = local_shift_encode3_d3q39(Int32(0), Int32(0), Int32(-1))
                    end
                    block_shift_dilated[b] = (code == Int32(0) && best_count > Int32(0)) ? best_code : code
                end
            end
        end
    end
end

@kernel function _relabel_local_lattice_shift_kernel!(f_arr, block_shift, block_shift_next,
                                                      rho_arr, vel_arr, obstacle, amr_covered,
                                                      n_blocks::Int32, block_size::Int32,
                                                      cx_arr, cy_arr, cz_arr, w_arr,
                                                      cs2_val::Float32, T_ref::Float32)
    x, y, z, b = @index(Global, NTuple)
    if b <= n_blocks
        @inbounds begin
                old_shift = block_shift[b]
                new_shift = block_shift_next[b]
                if old_shift != new_shift && !obstacle[x, y, z, b] && !amr_covered[x, y, z, b]
                old_sx = local_shift_decode_x_d3q39(old_shift)
                old_sy = local_shift_decode_y_d3q39(old_shift)
                old_sz = local_shift_decode_z_d3q39(old_shift)
                new_sx = local_shift_decode_x_d3q39(new_shift)
                new_sy = local_shift_decode_y_d3q39(new_shift)
                new_sz = local_shift_decode_z_d3q39(new_shift)
                rho_l = max(rho_arr[x, y, z, b], 0.01f0)
                ux_l = vel_arr[x, y, z, b, Int32(1)]
                uy_l = vel_arr[x, y, z, b, Int32(2)]
                uz_l = vel_arr[x, y, z, b, Int32(3)]
                if !(ux_l == ux_l) || !(uy_l == uy_l) || !(uz_l == uz_l)
                    ux_l = Float32(new_sx)
                    uy_l = Float32(new_sy)
                    uz_l = Float32(new_sz)
                end

                rho_t = 0.0f0
                jx_t = 0.0f0
                jy_t = 0.0f0
                jz_t = 0.0f0
                pxx_t = 0.0f0
                pyy_t = 0.0f0
                pzz_t = 0.0f0
                pxy_t = 0.0f0
                pxz_t = 0.0f0
                pyz_t = 0.0f0
                array_shift_x = Float32(cx_arr[Int32(1)])
                array_shift_y = Float32(cy_arr[Int32(1)])
                array_shift_z = Float32(cz_arr[Int32(1)])
                old_Ux = Float32(old_sx)
                old_Uy = Float32(old_sy)
                old_Uz = Float32(old_sz)
                for k in Int32(1):Int32(39)
                    val_old = max(f_arr[x, y, z, b, k], 1.0f-10)
                    qx_old = Float32(cx_arr[k]) - array_shift_x
                    qy_old = Float32(cy_arr[k]) - array_shift_y
                    qz_old = Float32(cz_arr[k]) - array_shift_z
                    cx_old = qx_old + old_Ux
                    cy_old = qy_old + old_Uy
                    cz_old = qz_old + old_Uz
                    rho_t += val_old
                    jx_t += val_old * cx_old
                    jy_t += val_old * cy_old
                    jz_t += val_old * cz_old
                    pxx_t += val_old * cx_old * cx_old
                    pyy_t += val_old * cy_old * cy_old
                    pzz_t += val_old * cz_old * cz_old
                    pxy_t += val_old * cx_old * cy_old
                    pxz_t += val_old * cx_old * cz_old
                    pyz_t += val_old * cy_old * cz_old
                end
                if !(rho_t == rho_t) || rho_t < 0.01f0
                    rho_t = rho_l
                    jx_t = rho_l * ux_l
                    jy_t = rho_l * uy_l
                    jz_t = rho_l * uz_l
                    pxx_t = rho_l * (ux_l*ux_l + cs2_val)
                    pyy_t = rho_l * (uy_l*uy_l + cs2_val)
                    pzz_t = rho_l * (uz_l*uz_l + cs2_val)
                    pxy_t = rho_l * ux_l * uy_l
                    pxz_t = rho_l * ux_l * uz_l
                    pyz_t = rho_l * uy_l * uz_l
                end
                inv_rho_t = 1.0f0 / max(rho_t, 0.01f0)
                ux_t = jx_t * inv_rho_t
                uy_t = jy_t * inv_rho_t
                uz_t = jz_t * inv_rho_t

                f_new = MVector{39, Float32}(undef)
                rho_new = 0.0f0
                mx_new = 0.0f0
                my_new = 0.0f0
                mz_new = 0.0f0
                qxx_new = 0.0f0
                qyy_new = 0.0f0
                qzz_new = 0.0f0
                qxy_new = 0.0f0
                qxz_new = 0.0f0
                qyz_new = 0.0f0
                delta_x = new_sx - old_sx
                delta_y = new_sy - old_sy
                delta_z = new_sz - old_sz
                transverse_recenter = (delta_y != Int32(0)) || (delta_z != Int32(0))
                lambda_recenter = (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
                                   0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
                mass_corr_recenter = 1.0f0
                if transverse_recenter
                    lambda_recenter = solve_13mom_lambdas_d3q39(
                        rho_t,
                        ux_t - Float32(new_sx),
                        uy_t - Float32(new_sy),
                        uz_t - Float32(new_sz),
                        cx_arr, cy_arr, cz_arr, w_arr,
                        cs2_val, Int32(120), T_ref,
                        array_shift_x, array_shift_y, array_shift_z)
                    mass_corr_recenter = compute_13mom_mass_correction_d3q39(
                        rho_t, lambda_recenter,
                        cx_arr, cy_arr, cz_arr, w_arr,
                        cs2_val, array_shift_x, array_shift_y, array_shift_z)
                end
                for k in Int32(1):Int32(39)
                    if transverse_recenter
                        qx_eq = Float32(cx_arr[k]) - array_shift_x
                        qy_eq = Float32(cy_arr[k]) - array_shift_y
                        qz_eq = Float32(cz_arr[k]) - array_shift_z
                        val = mass_corr_recenter * eval_13mom_feq_single(
                            rho_t, lambda_recenter,
                            qx_eq, qy_eq, qz_eq,
                            Float32(w_arr[k]), cs2_val)
                    else
                        km = local_shift_remap_d3q39(cx_arr, cy_arr, cz_arr, k, delta_x, delta_y, delta_z)
                        val = km > Int32(0) ?
                              f_arr[x, y, z, b, km] :
                              local_shift_fallback_feq3_d3q39(
                                  rho_t, ux_t, uy_t, uz_t,
                                  Float32(new_sx), Float32(new_sy), Float32(new_sz),
                                  k, cx_arr, cy_arr, cz_arr, w_arr, cs2_val, T_ref)
                    end
                    val = max(val, 1.0f-10)
                    f_new[k] = val
                    qx = Float32(cx_arr[k]) - array_shift_x
                    qy = Float32(cy_arr[k]) - array_shift_y
                    qz = Float32(cz_arr[k]) - array_shift_z
                    rho_new += val
                    mx_new += val * qx
                    my_new += val * qy
                    mz_new += val * qz
                    qxx_new += val * qx * qx
                    qyy_new += val * qy * qy
                    qzz_new += val * qz * qz
                    qxy_new += val * qx * qy
                    qxz_new += val * qx * qz
                    qyz_new += val * qy * qz
                end

                Ux = Float32(new_sx)
                Uy = Float32(new_sy)
                Uz = Float32(new_sz)
                mx_t = jx_t - Ux * rho_t
                my_t = jy_t - Uy * rho_t
                mz_t = jz_t - Uz * rho_t
                qxx_t = pxx_t - 2.0f0 * Ux * jx_t + Ux * Ux * rho_t
                qyy_t = pyy_t - 2.0f0 * Uy * jy_t + Uy * Uy * rho_t
                qzz_t = pzz_t - 2.0f0 * Uz * jz_t + Uz * Uz * rho_t
                qxy_t = pxy_t - Ux * jy_t - Uy * jx_t + Ux * Uy * rho_t
                qxz_t = pxz_t - Ux * jz_t - Uz * jx_t + Ux * Uz * rho_t
                qyz_t = pyz_t - Uy * jz_t - Uz * jy_t + Uy * Uz * rho_t

                drho = rho_t - rho_new
                dmx = mx_t - mx_new
                dmy = my_t - my_new
                dmz = mz_t - mz_new
                dqxx = qxx_t - qxx_new
                dqyy = qyy_t - qyy_new
                dqzz = qzz_t - qzz_new
                dqxy = qxy_t - qxy_new
                dqxz = qxz_t - qxz_new
                dqyz = qyz_t - qyz_new
                if transverse_recenter
                    stress_relabel_scale = 0.10f0
                    dqxx *= stress_relabel_scale
                    dqyy *= stress_relabel_scale
                    dqzz *= stress_relabel_scale
                    dqxy *= stress_relabel_scale
                    dqxz *= stress_relabel_scale
                    dqyz *= stress_relabel_scale
                end
                inv_cs2 = 1.0f0 / max(cs2_val, 1.0f-6)
                inv_cs4 = inv_cs2 * inv_cs2
                inv_2cs4 = 0.5f0 * inv_cs4

                limiter = 1.0f0
                for k in Int32(1):Int32(39)
                    qx = Float32(cx_arr[k]) - array_shift_x
                    qy = Float32(cy_arr[k]) - array_shift_y
                    qz = Float32(cz_arr[k]) - array_shift_z
                    df = Float32(w_arr[k]) * (
                        drho +
                        inv_cs2 * (dmx*qx + dmy*qy + dmz*qz) +
                        inv_2cs4 * (dqxx*(qx*qx - cs2_val) +
                                     dqyy*(qy*qy - cs2_val) +
                                     dqzz*(qz*qz - cs2_val)) +
                        inv_cs4 * (dqxy*qx*qy + dqxz*qx*qz + dqyz*qy*qz)
                    )
                    if df < 0.0f0
                        limiter = min(limiter, max((f_new[k] - 1.0f-10) / (-df + 1.0f-20), 0.0f0))
                    end
                end
                if transverse_recenter
                    limiter = min(limiter, 0.65f0)
                end

                for k in Int32(1):Int32(39)
                    qx = Float32(cx_arr[k]) - array_shift_x
                    qy = Float32(cy_arr[k]) - array_shift_y
                    qz = Float32(cz_arr[k]) - array_shift_z
                    df = limiter * Float32(w_arr[k]) * (
                        drho +
                        inv_cs2 * (dmx*qx + dmy*qy + dmz*qz) +
                        inv_2cs4 * (dqxx*(qx*qx - cs2_val) +
                                     dqyy*(qy*qy - cs2_val) +
                                     dqzz*(qz*qz - cs2_val)) +
                        inv_cs4 * (dqxy*qx*qy + dqxz*qx*qz + dqyz*qy*qz)
                    )
                    f_arr[x, y, z, b, k] = max(f_new[k] + df, 1.0f-10)
                end
            end
        end
    end
end

function update_local_lattice_shift!(level, backend, vel_current;
                                     f_current=nothing, rho_current=level.rho,
                                     f_history=nothing, rho_history=nothing, vel_history=nothing,
                                     u_frame_ref=0.0f0)
    n_blocks = length(level.active_block_coords)
    if n_blocks == 0 || !LOCAL_LATTICE_SHIFT_ENABLED
        return
    end
    kernel! = _update_block_shift_kernel!(backend)
    kernel!(level.block_shift, level.block_shift_next, vel_current, level.obstacle, level.amr_covered, level.sponge, level.wall_dist,
            level.local_shift_wall_guard, level.map_x,
            Int32(n_blocks), Int32(BLOCK_SIZE),
            LOCAL_LATTICE_SHIFT_U_MAX,
            LOCAL_LATTICE_SHIFT_THRESHOLD,
            LOCAL_LATTICE_SHIFT_HYSTERESIS,
            Float32(u_frame_ref),
            LATTICE_SHIFT_X,
            LOCAL_LATTICE_SHIFT_WALL_GUARD_CELLS,
            LOCAL_LATTICE_SHIFT_WALL_ZERO_GUARD_CELLS,
            LOCAL_LATTICE_SHIFT_ZERO_ALLOWED_X_MIN_CELLS,
            LOCAL_LATTICE_SHIFT_WAKE_ZERO_DEFICIT_FRAC,
            LOCAL_LATTICE_SHIFT_HIGH_SPEED_FRAC,
            LOCAL_LATTICE_SHIFT_GEOMETRIC_BAND_CELLS;
            ndrange=n_blocks)
    KernelAbstractions.synchronize(backend)

    if f_current !== nothing
        relabel! = _relabel_local_lattice_shift_kernel!(backend)
        relabel!(f_current, level.block_shift, level.block_shift_next,
                 rho_current, vel_current, level.obstacle, level.amr_covered,
                 Int32(n_blocks), Int32(BLOCK_SIZE),
                 C_X, C_Y, C_Z, W,
                 CS2_PHYSICS,
                 DDF_ENABLED ? Float32(DDF_T_INLET) : 1.0f0;
                 ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks))
        KernelAbstractions.synchronize(backend)

        if f_history !== nothing
            relabel!(f_history, level.block_shift, level.block_shift_next,
                     rho_history === nothing ? rho_current : rho_history,
                     vel_history === nothing ? vel_current : vel_history,
                     level.obstacle, level.amr_covered,
                     Int32(n_blocks), Int32(BLOCK_SIZE),
                     C_X, C_Y, C_Z, W,
                     CS2_PHYSICS,
                     DDF_ENABLED ? Float32(DDF_T_INLET) : 1.0f0;
                     ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks))
            KernelAbstractions.synchronize(backend)
        end
    end

    copyto!(level.block_shift, level.block_shift_next)
    KernelAbstractions.synchronize(backend)
end

function initialize_local_lattice_shift!(level, backend, vel_current; u_frame_ref=Float32(U_TARGET))
    n_blocks = length(level.active_block_coords)
    if n_blocks == 0 || !LOCAL_LATTICE_SHIFT_ENABLED
        return
    end

    kernel! = _update_block_shift_kernel!(backend)
    kernel!(level.block_shift, level.block_shift_next, vel_current, level.obstacle, level.amr_covered, level.sponge, level.wall_dist,
            level.local_shift_wall_guard, level.map_x,
            Int32(n_blocks), Int32(BLOCK_SIZE),
            LOCAL_LATTICE_SHIFT_U_MAX,
            LOCAL_LATTICE_SHIFT_THRESHOLD,
            LOCAL_LATTICE_SHIFT_HYSTERESIS,
            Float32(u_frame_ref),
            LATTICE_SHIFT_X,
            LOCAL_LATTICE_SHIFT_WALL_GUARD_CELLS,
            LOCAL_LATTICE_SHIFT_WALL_ZERO_GUARD_CELLS,
            LOCAL_LATTICE_SHIFT_ZERO_ALLOWED_X_MIN_CELLS,
            LOCAL_LATTICE_SHIFT_WAKE_ZERO_DEFICIT_FRAC,
            LOCAL_LATTICE_SHIFT_HIGH_SPEED_FRAC,
            LOCAL_LATTICE_SHIFT_GEOMETRIC_BAND_CELLS;
            ndrange=n_blocks)
    KernelAbstractions.synchronize(backend)

    relabel! = _relabel_local_lattice_shift_kernel!(backend)
    relabel!(level.f, level.block_shift, level.block_shift_next,
             level.rho, vel_current, level.obstacle, level.amr_covered,
             Int32(n_blocks), Int32(BLOCK_SIZE),
             C_X, C_Y, C_Z, W,
             CS2_PHYSICS,
             DDF_ENABLED ? Float32(DDF_T_INLET) : 1.0f0;
             ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks))
    KernelAbstractions.synchronize(backend)

    relabel!(level.f_temp, level.block_shift, level.block_shift_next,
             level.rho, vel_current, level.obstacle, level.amr_covered,
             Int32(n_blocks), Int32(BLOCK_SIZE),
             C_X, C_Y, C_Z, W,
             CS2_PHYSICS,
             DDF_ENABLED ? Float32(DDF_T_INLET) : 1.0f0;
             ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks))
    KernelAbstractions.synchronize(backend)

    if length(level.f_old) > N_DIRS
        relabel!(level.f_old, level.block_shift, level.block_shift_next,
                 level.rho, vel_current, level.obstacle, level.amr_covered,
                 Int32(n_blocks), Int32(BLOCK_SIZE),
                 C_X, C_Y, C_Z, W,
                 CS2_PHYSICS,
                 DDF_ENABLED ? Float32(DDF_T_INLET) : 1.0f0;
                 ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks))
        KernelAbstractions.synchronize(backend)
    end

    copyto!(level.block_shift, level.block_shift_next)
    KernelAbstractions.synchronize(backend)
end

function perform_timestep_v2!(
    level,
    parent_f, parent_rho, parent_vel, parent_ptr,
    parent_f_old, parent_rho_old, parent_vel_old,
    parent_tau::Float32,
    f_out, f_in, vel_out, vel_in,
    f_post_collision, u_curr,
    cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
    domain_nx, domain_ny, domain_nz,
    wall_model_active::Bool, c_wale_val::Float32, nu_sgs_bg::Float32,
    timestep::Int, inlet_turbulence::Float32,
    temporal_weight::Float32, use_temporal_interp::Bool, sponge_blend_dist::Bool;
    # Thermal DDF (Phase 2/4b) — optional keyword arguments
    g_out=nothing, g_in=nothing, g_post_collision_arr=nothing,
    parent_g=nothing, parent_temperature=nothing,
    parent_g_old=nothing, parent_temperature_old=nothing,
    parent_pi_neq=nothing,
    parent_pi_neq_old=nothing,
    # C2/C4 quintic B-spline prolongation (2026-06-04): parent level's precomputed
    # B-spline macro coefficients (filled by compute_spline_coeffs!). nothing → stub.
    parent_rho_spline_coeff=nothing, parent_vel_spline_coeff=nothing,
    parent_vel_thermal=nothing, parent_vel_old_thermal=nothing,
    parent_kinetic_energy=nothing,
    parent_tau_g::Float32=0.5f0,
    # Pass178 (Tier A item A2 — quadratic temporal interpolation):
    # Older snapshots of parent macros (state at t = -dt_c, one parent step
    # before parent_rho_old). When non-nothing, the AMR halo prolongation
    # uses quadratic Lagrange in time through {older, old, new} at times
    # {-dt_c, 0, +dt_c}. Default nothing → linear blend (backward-compatible).
    parent_rho_older=nothing, parent_vel_older=nothing,
    parent_temperature_older=nothing,
    # B1 P3 (2026-05-28) — cubic Hermite temporal: oldest parent snapshots
    # at t = -2·dt_c. Combined with {older, old, new} gives 4 anchors for
    # cubic Lagrange in time. Default nothing → quadratic / linear fallback.
    parent_rho_oldest=nothing, parent_vel_oldest=nothing,
    parent_temperature_oldest=nothing,
    # B1 P3 RK4-equivalent (2026-05-28) — quintic Lagrange temporal: oldest2
    # parent snapshots at t = -3·dt_c. Combined with {oldest, older, old, new}
    # gives 5 anchors. Default nothing → cubic / quadratic / linear fallback.
    parent_rho_oldest2=nothing, parent_vel_oldest2=nothing,
    parent_temperature_oldest2=nothing,
    # Multi-GPU (Strategy C) — block offset and range override
    block_offset::Int32=Int32(0),
    ndrange_blocks::Int=0
)
    backend = get_backend(f_in)
    n_blocks = length(level.active_block_coords)
    if n_blocks == 0; return; end

    is_l1 = (parent_f === nothing)

    if is_l1
        p_f = f_in; p_rho = level.rho; p_vel = level.vel; p_ptr = level.block_pointer
        p_f_old = f_in; p_rho_old = level.rho; p_vel_old = level.vel
        px, py, pz = Int32(1), Int32(1), Int32(1)
    else
        p_f = parent_f; p_rho = parent_rho; p_vel = parent_vel; p_ptr = parent_ptr
        p_f_old = parent_f_old; p_rho_old = parent_rho_old; p_vel_old = parent_vel_old
        px, py, pz = Int32(size(p_ptr, 1)), Int32(size(p_ptr, 2)), Int32(size(p_ptr, 3))
    end
    # Pass180 (Tier A item A2 — quadratic temporal): resolve "older" parent
    # snapshot arrays. When the parent supplies them AND macro_temporal mode
    # is on, the AMR halo interpolation uses quadratic Lagrange in time
    # through {parent_*_older, parent_*_old, parent_*_new}. Defaults to
    # parent_rho_old/vel_old (degrades quadratic to linear) when older not
    # supplied or is_l1.
    if is_l1 || parent_rho_older === nothing
        p_rho_older = p_rho_old
        p_vel_older = p_vel_old
    else
        p_rho_older = parent_rho_older
        p_vel_older = parent_vel_older
    end
    # B1 P3 — resolve oldest snapshots. When the parent doesn't supply them,
    # fall back to `_older` (degrades cubic-Hermite to quadratic). When the
    # parent doesn't supply `_older` either, falls back further to `_old`
    # (degrades cubic to linear).
    if is_l1 || parent_rho_oldest === nothing
        p_rho_oldest = p_rho_older
        p_vel_oldest = p_vel_older
    else
        p_rho_oldest = parent_rho_oldest
        p_vel_oldest = parent_vel_oldest
    end
    # B1 P3 RK4-equivalent — resolve oldest2 snapshots. When the parent doesn't
    # supply them, fall back to `_oldest` (degrades quintic to cubic), then to
    # `_older`/`_old` (cubic→quadratic→linear) via the same chain.
    if is_l1 || parent_rho_oldest2 === nothing
        p_rho_oldest2 = p_rho_oldest
        p_vel_oldest2 = p_vel_oldest
    else
        p_rho_oldest2 = parent_rho_oldest2
        p_vel_oldest2 = parent_vel_oldest2
    end
    # Pass182: quadratic temporal is now ALWAYS-ON safe via the bounded-
    # overshoot limiter inside interpolate_entropic_mirror_d3q39 (see
    # physics_interpolation.jl ~L765). The pass180 vorticity bleed during ramp
    # came from the negative w_older=-0.125 weight extrapolating against the
    # trend; the per-cell soft clamp limits any quadratic result to
    # [min(old,new) − δ, max(old,new) + δ] where δ = 10% of |old−new|, so any
    # spurious extrapolation collapses back to bounded interpolation. No
    # step-gate needed — the limiter handles ramp transients automatically.
    use_quadratic_temporal_flow =
        (!is_l1 && INTERFACE_MACRO_TEMPORAL_INTERPOLATION &&
         parent_rho_older !== nothing && parent_vel_older !== nothing &&
         length(parent_rho_older) > 1) ? Int32(1) : Int32(0)
    # B1 P3 — cubic Hermite is active only when INTERFACE_CUBIC_TEMPORAL is
    # set AND the parent supplied non-stub oldest arrays. Otherwise the
    # interpolator falls back to the quadratic (or linear) path.
    # HIGH-ORDER AMR-CONTINUITY SCHEDULE (2026-06-23): decide at RUNTIME whether to
    # pay the ~3-4× cost of C4/C3 this step. The kernel just receives 0/1 flags, so
    # this gates without any recompile.
    #   :always       → always on
    #   :off          → never (pure C0 trilinear, fastest)
    #   :steady_state → on once the flow has developed: u_curr/U_TARGET ≥ threshold
    #                   (ramp essentially complete) AND past the settle delay.
    _hi_order_active =
        INTERFACE_HIGH_ORDER_SCHEDULE == :always ? true :
        INTERFACE_HIGH_ORDER_SCHEDULE == :off ? false :
        # :steady_state — escalate near convergence.
        (U_TARGET > 0.0f0 &&
         (u_curr / U_TARGET) >= INTERFACE_HIGH_ORDER_U_THRESHOLD &&
         Int32(timestep) >= (Int32(RAMP_STEPS) + INTERFACE_HIGH_ORDER_SETTLE_STEPS))
    _hi_order_i = _hi_order_active ? Int32(1) : Int32(0)

    use_cubic_temporal_flow =
        (_hi_order_active && !is_l1 && INTERFACE_CUBIC_TEMPORAL && INTERFACE_MACRO_TEMPORAL_INTERPOLATION &&
         parent_rho_oldest !== nothing && parent_vel_oldest !== nothing &&
         length(parent_rho_oldest) > 1) ? Int32(1) : Int32(0)

    # B1 P3 RK4-equivalent — quintic Lagrange is active only when both the cubic
    # AND quintic flags are set AND the parent supplied non-stub oldest2 arrays.
    # Otherwise the interpolator falls back to cubic / quadratic / linear.
    use_quintic_temporal_flow =
        (_hi_order_active && !is_l1 && INTERFACE_QUINTIC_TEMPORAL && INTERFACE_CUBIC_TEMPORAL &&
         INTERFACE_MACRO_TEMPORAL_INTERPOLATION &&
         parent_rho_oldest2 !== nothing && parent_vel_oldest2 !== nothing &&
         length(parent_rho_oldest2) > 1) ? Int32(1) : Int32(0)

    use_macro_temporal_interp_flow =
        (!is_l1 && INTERFACE_MACRO_TEMPORAL_INTERPOLATION &&
         length(p_rho_old) > 1 && length(p_vel_old) > 1) ? Int32(1) : Int32(0)

    scale = 1 << (level.level_id - 1)
    nx_g, ny_g, nz_g = Int32(domain_nx * scale), Int32(domain_ny * scale), Int32(domain_nz * scale)
    # AMR-velocity-step fix (2026-05-28). The previous setting `amr_sgs_scale =
    # 1.0` (uniform lattice ν across levels) made physical ν scale with dx
    # — L1 had 8× the physical SGS viscosity of L4. The dissipation
    # discontinuity at each AMR interface printed onto velocity as a visible
    # step (red lump at L1↔L2 inlet side; stripes at wake L1↔L2; ~3-7%
    # velocity jump in the L2 box vs L1 freestream in case 030).
    #
    # Acoustic-LBM scaling: ν_phys = ν_lat · dx · cs. For UNIFORM physical ν
    # across levels (which is what produces a coupling-clean velocity field),
    # ν_lat must scale as 1/dx. That means: small lattice ν at coarse levels,
    # large at fine.
    #
    # With 2:1 refinement: dx_L1 = 2^(num_levels - 1)·dx_finest. So:
    #   amr_sgs_scale(level_id) = (1/2)^(NUM_LEVELS - level_id)
    # Gives 0.125, 0.25, 0.5, 1.0 for L1..L4 in a 4-level case. The finest
    # level (L4) keeps the reference SGS strength; coarser levels get
    # progressively less to compensate for their bigger dx.
    #
    # The earlier Pass173 attempt used (1/2)^(level_id - 1) — BACKWARDS
    # direction (L1=1.0, L4=0.125), which over-damped the coarse levels
    # because their lattice ν was the largest AND multiplied by the biggest
    # dx. That was 64× worse than uniform. The CORRECT direction (this fix)
    # gives uniform physical ν.
    # 2026-05-29 REVERT to uniform (= May-25 backup value). The level-dependent
    # `(0.5)^(NUM_LEVELS-level_id)` was introduced 2026-05-28 to equalize PHYSICAL
    # ν across levels, but it gives the COARSEST level (L1) the LEAST lattice
    # dissipation — and L1 is the most under-resolved (Re=1.9e7, biggest cells),
    # so it needs MORE, not less. Two confirmed harms vs the backup's uniform=1.0:
    #   (1) a 2× ν jump at every AMR interface → the bright |∇ρ| rim + Mach step
    #       at L1↔L2 (L1 = global ν minimum, so its interface is worst); the
    #       backup with =1.0 had NO such rim (verified by direct comparison).
    #   (2) starves the under-resolved coarse wake of the dissipation that kept
    #       the backup alive longer.
    # The "uniform physical ν" argument is sound only for a RESOLVED flow; at this
    # Re the coarse level is grossly under-resolved and uniform LATTICE ν is the
    # stabilizing + interface-clean choice. Restoring backup behavior.
    # Legacy level-dependent form (reverted): Float32(0.5 ^ (NUM_LEVELS_CONFIG - level.level_id))
    amr_sgs_scale = Float32(1.0)

    # Phase 2: Read collision parameters from globals
    # For cumulant: these are the actual omega values.
    # For entropic: the _adaptive_omega4 slot encodes the alpha-solver type,
    #               all other cumulant params are unused (zeroed out by the kernel).
    if COLLISION_OPERATOR == :entropic
        omega_bulk_p  = 0.0f0
        omega_3_p     = 0.0f0
        omega_4_p     = 0.0f0
        # Encode entropic solver type: 0=bisection, 1=newton, 2=approximate, 3=taylor
        adaptive_om4  = ENTROPIC_ALPHA_SOLVER == :newton      ? Int32(1) :
                        ENTROPIC_ALPHA_SOLVER == :approximate ? Int32(2) :
                        ENTROPIC_ALPHA_SOLVER == :taylor      ? Int32(3) : Int32(0)
        lambda_p      = ENTROPIC_SIGMA   # Reuse lambda slot for h-damping sigma
        limiter_i     = Int32(0)
    else
        omega_bulk_p  = CUMULANT_OMEGA_BULK
        omega_3_p     = CUMULANT_OMEGA_3
        omega_4_p     = CUMULANT_OMEGA_4
        adaptive_om4  = CUMULANT_ADAPTIVE_OMEGA4 ? Int32(1) : Int32(0)
        lambda_p      = CUMULANT_LAMBDA_PARAM
        # Encode limiter: 0=none, 1=factored, 2=positivity
        limiter_i     = CUMULANT_LIMITER == :factored ? Int32(1) :
                        CUMULANT_LIMITER == :positivity ? Int32(2) : Int32(0)
    end
    # Compressibility correction flag
    comp_corr_i   = COMPRESSIBILITY_CORRECTION ? Int32(1) : Int32(0)

    kernel! = get_collision_kernel(COLLISION_OPERATOR, backend, is_l1)

    # ── Phase 5: Pre-compute kernel DISABLED/DELETED (2026-06-24) ──
    # The D3Q39 collision kernel computes O(u⁴) polynomial equilibrium INLINE
    # from post-streaming ρ, u. The old pre-compute kernel (kernel_equilibrium_d3q39.jl,
    # deleted) used stale previous-timestep ρ, u → transonic diagonal-stripe artifacts.

    # ── Phase 5: Pre-compute inlet/outlet boundary equilibrium on CPU ──
    # The inlet/outlet BCs need equilibrium at u_inlet velocity. For D3Q39,
    # always use Newton-Raphson on CPU (13-moment exact, trivially fast for 39 values).
    # With shifted lattice, polynomial can produce negative values; without shift,
    # NR still provides higher accuracy than polynomial at transonic speeds.
    if LATTICE_TYPE == :d3q39
        cx_s, cy_s, cz_s, w_s, _, _ = build_d3q39_static()
        n_inlet_frames = LOCAL_LATTICE_SHIFT_ENABLED ? (Int(LOCAL_LATTICE_SHIFT_U_MAX) + 1) : 1
        f_eq_inlet_cpu = Vector{Float32}(undef, 39 * n_inlet_frames)
        for frame_u in 0:(n_inlet_frames - 1)
            # Compute inlet equilibrium: rho=1, u=(u_inlet,0,0), relative to the frame.
            frame_shift_x = LOCAL_LATTICE_SHIFT_ENABLED ? Float32(frame_u) : Float32(LATTICE_SHIFT_X)
            u_rel_inlet = u_curr - frame_shift_x
            uy_rel_inlet = -Float32(LATTICE_SHIFT_Y)
            uz_rel_inlet = -Float32(LATTICE_SHIFT_Z)
            f_eq_frame = MVector{39, Float32}(undef)
            calculate_equilibrium_numerical_d3q39!(
                f_eq_frame, 1.0f0, u_rel_inlet, uy_rel_inlet, uz_rel_inlet,
                cx_s, cy_s, cz_s, w_s, CS2_PHYSICS, Int32(200),
                DDF_ENABLED ? Float32(DDF_T_INLET) : 1.0f0
            )
            mass_correct_feq39!(f_eq_frame, 1.0f0)
            for k in 1:39
                f_eq_inlet_cpu[frame_u * 39 + k] = max(f_eq_frame[k], Float32(1e-10))
            end
        end
        f_eq_inlet_gpu = adapt(backend, f_eq_inlet_cpu)

        # Stationary walls in a shifted lattice are moving walls in the lattice
        # frame: u_wall,rel = -U_frame. Precompute rho=1 numerical equilibria so
        # Bouzidi/YMLS can correct the ordinary no-slip-at-U=0 reconstruction.
        max_wall_frame = max(abs(Int(LATTICE_SHIFT_X)),
                             LOCAL_LATTICE_SHIFT_ENABLED ? Int(LOCAL_LATTICE_SHIFT_U_MAX) : 0)
        n_wall_frames = max_wall_frame + 1
        f_eq_wall_cpu = Vector{Float32}(undef, 39 * n_wall_frames)
        for frame_u in 0:(n_wall_frames - 1)
            f_eq_frame = MVector{39, Float32}(undef)
            calculate_equilibrium_numerical_d3q39!(
                f_eq_frame, 1.0f0, -Float32(frame_u), -Float32(LATTICE_SHIFT_Y), -Float32(LATTICE_SHIFT_Z),
                cx_s, cy_s, cz_s, w_s, CS2_PHYSICS, Int32(200),
                DDF_ENABLED ? Float32(DDF_T_INLET) : 1.0f0
            )
            mass_correct_feq39!(f_eq_frame, 1.0f0)
            for k in 1:39
                f_eq_wall_cpu[frame_u * 39 + k] = max(f_eq_frame[k], Float32(1e-10))
            end
        end
        f_eq_wall_gpu = adapt(backend, f_eq_wall_cpu)
        wall_eq_frame_count = Int32(n_wall_frames)
    else
        # Dummy array (unused for D3Q27)
        f_eq_inlet_gpu = adapt(backend, zeros(Float32, 39))
        f_eq_wall_gpu = adapt(backend, zeros(Float32, 39))
        wall_eq_frame_count = Int32(1)
    end

    parent_temperature_flow = parent_temperature !== nothing ? parent_temperature :
                              ((DDF_ENABLED && has_thermal(level)) ? level.temperature : level.rho)
    parent_temperature_old_flow = parent_temperature_old !== nothing ? parent_temperature_old :
                                  ((DDF_ENABLED && has_thermal(level)) ? level.temperature_old : level.rho)
    parent_pi_neq_flow = parent_pi_neq !== nothing ? parent_pi_neq : level.pi_neq
    parent_pi_neq_old_flow = parent_pi_neq_old !== nothing ? parent_pi_neq_old : parent_pi_neq_flow
    use_parent_pi_neq_flow =
        (!is_l1 && has_pi_neq(level) && parent_pi_neq !== nothing && length(parent_pi_neq) > 6) ? Int32(1) : Int32(0)
    use_parent_pi_neq_old_flow =
        (use_parent_pi_neq_flow == Int32(1) &&
         parent_pi_neq_old !== nothing &&
         length(parent_pi_neq_old_flow) == length(parent_pi_neq_flow)) ? Int32(1) : Int32(0)
    interface_entropic_h_scale_flow =
        (!is_l1 && COLLISION_OPERATOR == :entropic && LATTICE_TYPE == :d3q39) ? Float32(INTERFACE_ENTROPIC_H_SCALE) : 0.0f0

    # (ET/AA-pattern refined-level dispatch removed 2026-06-24 with the D3Q19 retirement.)

    # (D3Q19 RR-BGK kernel handles + AMR kargs setup removed 2026-06-24 with the D3Q19 retirement.)
    _kargs = (
        f_out, f_in, f_post_collision,
        level.rho, vel_out, vel_in, level.rho_old,
        level.obstacle, level.sponge, level.wall_dist, level.amr_covered,
        level.neighbor_table,
        level.map_x, level.map_y, level.map_z,
        p_f, p_rho, p_vel, p_ptr,
        p_f_old, p_rho_old, p_vel_old,
        parent_temperature_flow, parent_temperature_old_flow, parent_pi_neq_flow,
        px, py, pz,
        level.tau, parent_tau, c_wale_val, nu_sgs_bg, amr_sgs_scale,
        CS2_PHYSICS, CS4_PHYSICS,
        Val(is_l1),
        GROUND_ANALYSIS ? Int32(1) : Int32(0),
        Int32(Z_WALL_BC),   # advanced.boundary.z_walls (lbm-zwalls): 0 legacy, 1 slip, 2 no-slip
        Int32(LEVEL_VISC_PHYSICAL),   # advanced.refinement.level_viscosity (lbm-taufix): 0 legacy, 1 physical
        nx_g, ny_g, nz_g, u_curr,
        Int32(n_blocks), Int32(BLOCK_SIZE),
        cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
        Val(wall_model_active),
        Float32(WALL_MODEL_MAX_DISTANCE_CELLS),
        # NOTE (2026-06-23): the duplicate early Larsson arg that lived here was an
        # erroneous copy-paste from the D3Q19 branches — neither the D3Q27 cumulant
        # nor the D3Q39 entropic kernel has a slot at this position (the real Larsson
        # flag is passed later, mapping to _wall_model_larsson_pgr_i). It caused a
        # MethodError on stream_collide_cumulant! / stream_collide_entropic_d3q39!.
        # Removed so this shared branch matches the kernel signature contract.
        Int32(timestep % 1000000),
        (level.bouzidi_enabled && level.n_boundary_cells > 0) ? Int32(1) : Int32(0),
        inlet_turbulence,
        Float32(temporal_weight),
        use_temporal_interp ? Int32(1) : Int32(0),
        use_macro_temporal_interp_flow,
        use_parent_pi_neq_flow,
        interface_entropic_h_scale_flow,
        sponge_blend_dist ? Int32(1) : Int32(0),
        # Phase 2: new cumulant parameters
        omega_bulk_p,
        omega_3_p,
        omega_4_p,
        adaptive_om4,
        lambda_p,
        limiter_i,
        # Compressibility correction
        comp_corr_i,
        # Phase 4: two-way thermal coupling — temperature feeds back to flow equilibrium
        (DDF_ENABLED && has_thermal(level)) ? level.temperature : level.rho,  # temperature_arr (dummy=rho if no thermal)
        DDF_ENABLED ? Float32(DDF_T_INLET) : 1.0f0,                          # T_inlet
        (DDF_ENABLED && DDF_FEEDBACK_TO_FLOW && LATTICE_TYPE == :d3q39 && has_thermal(level)) ? Int32(1) : Int32(0),  # thermal_coupling flag
        DDF_ENABLED ? Float32(DDF_FEEDBACK_RELAXATION) : 1.0f0,             # feedback_relaxation
        (DDF_SHOCK_TUBE_COUPLING && INITIAL_CONDITION_TYPE === :riemann_x) ? Int32(1) : Int32(0),  # AUDIT: shock_tube_thermal — ungate feedback for Riemann
        # Phase 4b: Frapolli E_kin output for thermal energy conservation
        (DDF_ENABLED && has_thermal(level)) ? level.kinetic_energy : level.rho,  # kinetic_energy_out (dummy=rho if no thermal)
        (DDF_ENABLED && LATTICE_TYPE == :d3q39) ? Float32(DDF_CV) : 0.0f0,      # Cv_thermal (0.0 = no E_kin output)
        # Phase 5: Pre-computed inlet/outlet equilibrium (D3Q39 shifted lattice)
        f_eq_inlet_gpu,  # 39-element GPU array/table of inlet equilibrium
        level.block_shift,
        LOCAL_LATTICE_SHIFT_ENABLED ? Int32(1) : Int32(0),
        TRANSVERSE_FREESTREAM_BOUNDARY ? Int32(1) : Int32(0),
        Float32(TRANSVERSE_FREESTREAM_BLEND),
        OUTLET_FREESTREAM_BOUNDARY ? Int32(1) : Int32(0),
        Int32(OUTLET_BOUNDARY_MODE),
        Float32(OUTLET_FREESTREAM_BLEND),
        Int32(INLET_DENSITY_MODE),
        Float32(INLET_NEQ_BLEND),   # non-reflecting inlet (2026-05-29); 1.0=legacy reflective
        # Multi-GPU block offset (0 for single-GPU)
        block_offset,
        # Phase B: variable ν_sgs from Π_neq
        level.pi_neq,
        VARIABLE_NU_SGS ? Int32(1) : Int32(0),
        Float32(VARIABLE_NU_SGS_C_PI),
        Float32(VARIABLE_NU_SGS_FLOOR),
        # Phase B4: adaptive sigma from |Π_neq|
        VARIABLE_SIGMA ? Int32(1) : Int32(0),
        Float32(VARIABLE_SIGMA_MIN),
        Float32(VARIABLE_SIGMA_MAX),
        Float32(VARIABLE_SIGMA_PI_REF),
        # Phase C1: Xu-Sagaut sponge mode
        SPONGE_XU_SAGAUT ? Int32(1) : Int32(0),
        # Tier A-2: Knudsen-adaptive τ (Latt 2020). Applied on top of variable_nu_sgs
        # via max(); when that's off, it is the sole adaptive-τ mechanism.
        KNUDSEN_ADAPTIVE_TAU ? Int32(1) : Int32(0),
        # ── AMR feather-band extra dissipation ──
        # feather_weight is populated at setup on levels with an inactive-block
        # neighbor. FEATHER_DEPTH uses it for optional viscosity; the AMR
        # regularization depth uses it only for mismatch-triggered h cleanup.
        level.feather_weight,
        FEATHER_NU_MAX,
        # ── L1-L2 discontinuity fix (2026-05-28) ──
        # Pass185 had gated feather to `level_id > 1` after pass183's symmetric
        # damping caused velocity-highpass streaks. With case 014's clean
        # baseline we now see the OPPOSITE artifact: a clear density step at
        # the L1-L2 interface (visible as vertical bands in the density slice
        # at x ≈ 100 px and 440 px). The step is the L1 cell-average prolongation
        # error (L1 cells are 832mm → can only represent gradients at L1
        # resolution; L2 inherits the step). The fix is symmetric dissipation
        # on BOTH sides of the L1-L2 interface — fine side (L2) was already
        # handled, coarse side (L1) was gated out.
        #
        # Dropping the `level_id > 1` gate means L1 cells WITH nonzero
        # feather_weight get the extra ν. feather_weight on L1 is zero unless
        # `coarse_side_feather_scale > 0` (Pass185-disabled, opt-in per case).
        # So this change is a NO-OP for cases that don't enable coarse-side
        # feather; for cases that do, it lets the dissipation reach the L1
        # cells facing an L2 zone. Pass185's "streak" artifact came from
        # pass183's larger ν_max + always-on feather; tuning ν_max/depth
        # appropriately per case keeps the L1 path safe.
        (FEATHER_DEPTH > 0 && FEATHER_NU_MAX > 0.0f0) ? Int32(1) : Int32(0),
        ((FEATHER_DEPTH > 0 || AMR_REGULARIZATION_DEPTH > 0)) ? Int32(1) : Int32(0),
        # ── Recursive Regularization (RR-LBM, Latt-Chopard 2006 / Coreixas 2017 /
        #    Schukmann 2025) — post-collision Hermite-projection regularization.
        # Gated per-level: applied on level_id >= RR_LBM_MIN_LEVEL when enabled, OR
        # the FINEST level only (level_id == NUM_LEVELS_CONFIG, the actual runtime level
        # count) when min_level: "finest" — robust to AMR level clamping.
        # Only the D3Q39 entropic kernel uses it; other kernels accept-and-ignore.
        Val(RR_LBM_ENABLED && (RR_LBM_FINEST_ONLY ?
                               (level.level_id == Int32(NUM_LEVELS_CONFIG)) :
                               (level.level_id >= RR_LBM_MIN_LEVEL))),
        # ── Spectral Vanishing Viscosity (SVV) — Y-Z bi-Laplacian (Tadmor 1989 /
        #    Karamanos-Karniadakis 2000 / Pasquetti 2006).
        # Damps spatial high-wavenumber modes (e.g., (-1)^(y+z) checkerboard)
        # that have nonzero macroscopic moments and escape Hermite projection.
        # Gated per-level: applied on level_id >= SVV_YZ_MIN_LEVEL when enabled.
        # epsilon controls strength: k⁴ scaling means ε=0.002 → 6.4%/step damping
        # at Y-Z Nyquist, 0.14%/step at k=π/4 (well-resolved). Pass 0 to disable.
        (SVV_YZ_ENABLED && level.level_id >= SVV_YZ_MIN_LEVEL) ? SVV_YZ_EPSILON : 0.0f0,
        # SOTA #5: streamwise-X SVV — gated to COARSE levels (level_id <=
        # SVV_X_MAX_LEVEL, default 1 = L1 only), where the streamwise lattice
        # banding lives and the lattice ν is smallest. Inverse gating vs Y-Z.
        (SVV_X_ENABLED && level.level_id <= SVV_X_MAX_LEVEL) ? SVV_X_EPSILON : 0.0f0,
        # ── SOTA A3: Two-way EOS coupling — thermodynamic R = (γ-1)·cv ──
        # When EOS_THERMODYNAMIC=true, the 13-moment Newton solver uses the
        # full ideal-gas EOS p = (γ-1)·ρ·cv·T instead of the linearized
        # p = ρ·cs²·T (which implicitly assumes (γ-1)·cv = cs²). Required for
        # any defensible Ma > 1.2 result. Sentinel -1 triggers legacy behavior
        # in the solver (falls back to cs²). For air (γ=1.4, cv=2.5):
        # (γ-1)·cv = 1.0 vs cs²=2/3 — 50% relative pressure difference at T=1.
        EOS_THERMODYNAMIC ? Float32((DDF_GAMMA - 1.0) * DDF_CV) : Float32(-1.0f0),
        # ── SOTA B2: Dynamic SGS (Germano-Lilly) ───────────────────────
        # When enabled, the entropic kernel replaces the hand-tuned `c_pi`
        # with a per-cell coefficient computed inline via Germano-Lilly.
        # Requires variable_nu_sgs to be enabled too; this flag has no effect
        # otherwise.
        Val(DYNAMIC_SGS_ENABLED),
        # ── SOTA B6: Compressible wall model (Van Driest transformation) ──
        # When enabled and wall_model is active, the equilibrium wall model is
        # transformed by ρ_local/ρ_wall to account for compressibility (Van
        # Driest 1956). Required for high-Re compressible BLs. The wall
        # temperature is from `thermal.t_wall`.
        WALL_MODEL_COMPRESSIBLE ? Int32(1) : Int32(0),
        Float32(DDF_T_WALL),
        # ── SOTA B6-ext: Larsson 2010 NEQ-WM pressure-gradient correction ──
        # Adds (y/(ρ_w·u_τ²))·∂p/∂x_stream to the log-law residual. Requires
        # the Van Driest compressible wall model (WALL_MODEL_COMPRESSIBLE).
        # Stub Int32(0) preserves legacy behavior.
        WALL_MODEL_LARSSON_PGR ? Int32(1) : Int32(0),
        # ── HRR-hybrid stress reconstruction (Astoul 2020 / Schukmann 2025) ──
        # Blends a fraction (1-σ_HRR) of the projection-based 2nd-order non-eq
        # coefficient with a finite-difference Chapman-Enskog continuum stress.
        # Only the D3Q39 entropic kernel currently implements this; other kernels
        # accept and ignore the parameters via leading-underscore stubs.
        # Pass 261: σ resolved per level — HRR_SIGMA_PER_LEVEL[level_id] when present,
        # otherwise HRR_SIGMA. Lets us add extra stress damping on the finest level
        # only (kills the L3 Nyquist ring without softening L1/L2).
        Val(HRR_HYBRID_ENABLED),
        (length(HRR_SIGMA_PER_LEVEL) >= level.level_id) ?
            HRR_SIGMA_PER_LEVEL[level.level_id] : HRR_SIGMA,
        AMR_HALO_FULLCELL_RECOMPOSITION ? Int32(1) : Int32(0),
        Float32(AMR_HALO_FULLCELL_MACRO_ALPHA),
        Float32(AMR_HALO_FULLCELL_POP_ALPHA),
        AMR_PROLONGATION_MONOTONE_VELOCITY ? Int32(1) : Int32(0),
        AMR_HALO_TEMPORAL_MACRO_CLOSURE ? Int32(1) : Int32(0),
        AMR_HALO_EQ_PROJECTION ? Int32(1) : Int32(0),
        AMR_HALO_CONSTRAINED_COMPLETION ? Int32(1) : Int32(0),
        # Pass180: quadratic temporal interpolation at AMR halos
        p_rho_older, p_vel_older, parent_temperature_older,
        use_quadratic_temporal_flow,
        # B1 P3 (2026-05-28): cubic Hermite temporal interpolation at AMR halos
        p_rho_oldest, p_vel_oldest, parent_temperature_oldest,
        Val(use_cubic_temporal_flow == Int32(1)),
        # B1 P3 RK4-equivalent (2026-05-28): quintic Lagrange temporal at AMR halos
        p_rho_oldest2, p_vel_oldest2, parent_temperature_oldest2,
        Val(use_quintic_temporal_flow == Int32(1)),
        # A1 LIVE (2026-05-28) — SgsAndFeaturesPack bundle. 11 scattered scalars
        # (6 Float32 thresholds + 5 Int32 feature flags) packed into a 44-byte
        # isbits struct. Body locals shadow the per-scalar args inside the
        # D3Q39 entropic kernel; LLVM DAE drops the unused args from PTX.
        SgsAndFeaturesPack(
            Float32(VARIABLE_NU_SGS_C_PI),
            Float32(VARIABLE_NU_SGS_FLOOR),
            Float32(VARIABLE_SIGMA_MIN),
            Float32(VARIABLE_SIGMA_MAX),
            Float32(VARIABLE_SIGMA_PI_REF),
            FEATHER_NU_MAX,
            VARIABLE_NU_SGS ? Int32(1) : Int32(0),
            VARIABLE_SIGMA ? Int32(1) : Int32(0),
            SPONGE_XU_SAGAUT ? Int32(1) : Int32(0),
            KNUDSEN_ADAPTIVE_TAU ? Int32(1) : Int32(0),
            (FEATHER_DEPTH > 0 && FEATHER_NU_MAX > 0.0f0) ? Int32(1) : Int32(0),
        ),
        # AUDIT FIX (2026-06-03) — MacroBoundsPack: configurable realizability
        # clamp bounds (legacy 0.42/2.10/0.35/2.40 by default; canonical
        # low-density cases override so Sod ρ_R=0.125 is not clamped to 0.42).
        MacroBoundsPack(
            Float32(MACRO_DENSITY_FLOOR),
            Float32(MACRO_DENSITY_CEIL),
            Float32(MACRO_DENSITY_FLOOR_NEAR),
            Float32(MACRO_DENSITY_CEIL_NEAR),
        ),
        # B4 LIVE (2026-05-28) — Newton warm-start cache (per-level 5D Float32).
        # Init: NaN sentinel → polynomial init on first step; on subsequent steps
        # holds converged λ from previous step → Newton starts near solution.
        level.lambda_cache,
        # SOTA #3 (2026-05-29): Ducros shock-sensor dissipation. Active only on
        # the entropic D3Q39 path; stubs on the others. Default-on for compressible.
        Val(SHOCK_SENSOR_ENABLED && LATTICE_TYPE == :d3q39),
        Float32(SHOCK_SENSOR_NU_MAX),
        Float32(SHOCK_SENSOR_THRESH_LO),
        Float32(SHOCK_SENSOR_THRESH_HI),
        # Level-independent lattice-ν floor (2026-05-29) — unscaled by level so
        # coarse levels (L1) get enough lattice dissipation to kill streamwise
        # checkerboard modes. Same value on all levels.
        Float32(NU_SGS_LATTICE_FLOOR),
        ADAPTIVE_OPERATOR ? Float32(ADAPTIVE_MA_THRESH) : 0.0f0,
        # C2/C4 quintic B-spline prolongation (2026-06-04): parent spline coeffs +
        # gate. Active on D3Q27-cumulant AND D3Q39-entropic AMR levels (2026-06-27:
        # the B-spline macro path is lattice-agnostic; the D3Q39 entropic interp
        # (interpolate_entropic_mirror_d3q39) now forwards + consumes these coeffs
        # via the same @noinline interpolate_macro_bspline helper) with the feature
        # on and coeffs present.
        (parent_rho_spline_coeff === nothing ? level.rho : parent_rho_spline_coeff),
        (parent_vel_spline_coeff === nothing ? level.vel : parent_vel_spline_coeff),
        (_hi_order_active && !is_l1 &&
         (LATTICE_TYPE == :d3q27 || LATTICE_TYPE == :d3q39) &&
         QUINTIC_BSPLINE_PROLONGATION &&
         parent_rho_spline_coeff !== nothing) ? Int32(1) : Int32(0),
        # Gendre 2017 AMR-interface normal-gradient correction (2026-06-28).
        # D3Q39-entropic only; only meaningful on AMR levels (!is_l1, where parent
        # halo cells exist). Val-gated ⇒ dead-code-eliminated off ⇒ byte-identical.
        Val(GENDRE_NORMAL_GRADIENT && !is_l1 && LATTICE_TYPE == :d3q39),
    )
    _kndrange = (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, ndrange_blocks > 0 ? ndrange_blocks : n_blocks)
    # (D3Q19 halo-dir / pi-dense ndranges removed 2026-06-24 with the D3Q19 retirement.)

    # ── One-shot GPU register/occupancy probe (LUDWIG_KERNEL_PROBE=1) ──────
    # Reports registers/thread, spill (local-mem) bytes, and theoretical
    # occupancy for the flow kernel — the same data Nsight Compute's register
    # page shows, via CUDA.jl introspection (no ncu install needed). Fires once.
    # (D3Q19 AMR KernelProbe blocks removed 2026-06-24 with the D3Q19 retirement.)
    _probe_this = _KERNEL_PROBE_ENABLED && backend isa CUDABackend &&
                  (is_l1 ? !_KERNEL_PROBE_DONE_L1[] : !_KERNEL_PROBE_DONE_AMR[])
    if _probe_this
        try
            _ndr, _wgs, _itsp, _dyn = KernelAbstractions.launch_config(kernel!, _kndrange, nothing)
            _ctx = KernelAbstractions.mkcontext(kernel!, _ndr, _itsp)
            _hk = CUDA.@cuda launch=false always_inline=backend.always_inline kernel!.f(_ctx, _kargs...)
            _regs = CUDA.registers(_hk)
            _mem  = CUDA.memory(_hk)
            _maxt = CUDA.maxthreads(_hk)
            println("┌─ [KernelProbe] flow kernel = $(kernel!.f)  (level_id=$(level.level_id), is_l1=$(is_l1))")
            println("│  registers/thread = $(_regs)   (RTX 4060 cap = 255; >168 limits occupancy)")
            println("│  spill local-mem  = $(_mem.local) bytes/thread   (>0 ⇒ register spilling — perf killer)")
            println("│  shared = $(_mem.shared) B   constant = $(_mem.constant) B   maxthreads = $(_maxt)")
            _blk = BLOCK_SIZE^3
            println("│  block = $(_blk) threads ⇒ regs/block = $(_regs*_blk) (65536 regs/SM on Ada)")
            println("│  occupancy-limited-by-regs: $(_regs*_blk > 65536 ? "YES (regs exceed SM budget for 1 block)" : "no")")
            # ── OCCUPANCY: blocks/SM the GPU can run with _regs at the REAL
            # launch thread count (_maxt, since KA caps the block to maxthreads). ──
            try
                _thr = _maxt                                            # actual threads/block
                _ab = CUDA.active_blocks(_hk.fun, _thr)                 # concurrent blocks/SM
                _occ = _ab * _thr / 1536.0                             # Ada: 1536 threads/SM max
                _peakbw = CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)*1e3 *
                          (CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)/8) * 2 / 1e9
                println("│  ── OCCUPANCY ────────────────────────────────────")
                println("│  launch threads/block = $(_thr)   concurrent blocks/SM = $(_ab)")
                println("│  achieved occupancy ≈ $(round(_occ*100,digits=1))%  (warps/SM = $(_ab*_thr÷32) of 48)")
                println("│  peak DRAM BW ≈ $(round(_peakbw,digits=0)) GB/s; 32 MB L2 (data may be L2-resident)")
                println("│  → for true MLUPS/BW use the solver's compute-only MLUPS column (C4 metric);")
                println("│    in-kernel micro-timing is unreliable here (async + L2 effects).")
                println("│  LIKELY BOTTLENECK: $(_occ < 0.25 ? "OCCUPANCY/register (<25% occ from spill — but LBM moves 80+ floats/cell so MEMORY likely co-limits)" : "memory/compute (occupancy OK)")")
                println("│  → for REAL throughput use the solver's compute-only MLUPS (main.jl C4 metric,")
                println("│    times the full recursive timestep). In-kernel re-launch timing is NOT")
                println("│    representative (cache-resident, no streaming dependency).")
            catch _re
                println("│  [occupancy] skipped: $(_re)")
            end
            println("└─ probe done ($(is_l1 ? "L1" : "AMR/L>1") variant); run continues normally.")
        catch _e
            println("[KernelProbe] failed: $(_e)")
        end
        if is_l1; _KERNEL_PROBE_DONE_L1[] = true; else _KERNEL_PROBE_DONE_AMR[] = true; end
    end

    # NOTE 2026-05-30: a LUDWIG_MAXREGS forced-occupancy launch was tested here
    # (cap regs/thread → ≥2 blocks/SM). The launch wiring works (cudaconvert'd
    # HostKernel call), BUT ptxas HANGS recompiling the legacy wide kernel under ANY
    # maxregs cap (tested 128 AND 200 — both: GPU 0%, compile never finishes).
    # So the maxregs occupancy lever is BLOCKED by kernel SIZE, not by my code.
    # The real path to higher occupancy is smaller purpose-built kernels; the
    # D3Q19 C4 variant above is the first split. Reverted to the clean launch.
    if _PHASE_TIMING
        KernelAbstractions.synchronize(backend); _tf = time()
        # (D3Q19 cell-split launch path removed 2026-06-24; conventional monolithic launch.)
        _tk = time()
        kernel!(_kargs...; ndrange=_kndrange)
        KernelAbstractions.synchronize(backend); _phase_add!(_T_SPLIT_L1, level.level_id, time() - _tk)
        KernelAbstractions.synchronize(backend)
        _T_FLOW[] += time() - _tf; _T_FLOW_CALLS[] += 1
    else
        # PERF (2026-06-04, deep-sync workflow): removed the per-flow-kernel host
        # synchronize — THE highest-frequency barrier (~63×/step in a 6-level
        # 2:1 sub-cycle). Everything that follows on this backend is more GPU
        # work (NSCBC outlet kernel, mass rescale broadcast, Bouzidi correction,
        # thermal DDF, then the child level's kernels), all auto-ordered by
        # KernelAbstractions. The only host reads in the window are the
        # MASS_CONTROL `sum(level.rho)` reductions (≈1286/1336), which are GPU
        # reductions that copy their scalar back with their OWN synchronizing
        # copyto! — so they do not depend on this barrier. Byte-identical on the
        # single-GPU same-backend path. Verified removable by classifier+verifier.
        # (D3Q19 cell-split launch path removed 2026-06-24; conventional monolithic launch.)
        kernel!(_kargs...; ndrange=_kndrange)
    end

    # (LUDWIG_DEBUG_SPLIT_DIFF D3Q19 diagnostic removed 2026-06-24 with the D3Q19 retirement.)
    # ── D3Q19 SVV high-k damping (separate post-collision kernel) ──────
    # OCCUPANCY (2026-06-05): SVV runs here as a standalone pass, NOT inline in
    # the collision kernel, so the main kernel sheds the f_stored-lifetime + SVV
    # temporaries that had pinned it at 255 regs / 16.7% occupancy. Same k⁴
    # bi-Laplacian + per-cell 4-moment conservation as before; gated per-level on
    # the same epsilon expressions used in _kargs (lines ~1075/1079). KA orders it
    # after the flow kernel on the same backend — no host sync needed.
    # 2026-06-23: standalone SVV post-collision pass for D3Q19 AND D3Q27 cumulant.
    # The k⁴ bi-Laplacian is operator-independent (acts on post-collision pops) and
    # the inline helper is now MVector-generic. For a sharp-edged body at high Re,
    # the cumulant adaptive-ω₄ alone does NOT kill the (-1)^(x+z) Nyquist mode at the
    # sharp LE/TE + AMR wake crossings; SVV's ~47× k⁴ selectivity removes it without
    # touching resolved scales (NOT global dissipation). Default off ⇒ no-op.
    if LATTICE_TYPE == :d3q27
        _svv_yz_eps = (SVV_YZ_ENABLED && level.level_id >= SVV_YZ_MIN_LEVEL) ? SVV_YZ_EPSILON : 0.0f0
        _svv_x_eps  = (SVV_X_ENABLED  && level.level_id <= SVV_X_MAX_LEVEL)  ? SVV_X_EPSILON  : 0.0f0
        if _svv_yz_eps > 0.0f0 || _svv_x_eps > 0.0f0
            if _PHASE_TIMING
                KernelAbstractions.synchronize(backend)
                _tsvv = time()
            end
            apply_svv_d3q19_postcollision_launch!(
                f_out, f_in, f_post_collision, level.obstacle, level.neighbor_table,
                BLOCK_SIZE, n_blocks,
                cx_gpu, cy_gpu, cz_gpu, w_gpu,
                Float32(CS2_LATTICE), Int(N_DIRS),
                # 2026-10-27 (lbm-periodic-amr, out-of-bounds fix): write f_post_collision only when it is the
                # full-size array. On a level without boundary cells it is the 1x1x1x1xN stub (blocks.jl), and the
                # unconditional write went past its end on every SVV step (CPU: GC corruption; GPU: silent device
                # memory writes). Full-size arrays: unchanged (flag 1, as before).
                _svv_x_eps, _svv_yz_eps, (size(f_post_collision, 1) == BLOCK_SIZE ? Int32(1) : Int32(0)),
                backend;
                # FUSED-Π (2026-06-13): the SVV epilogue writes level.pi_neq for
                # every SVV-active fluid cell (post-f-final). It MUST read the
                # SAME macro arrays the legacy refresh reads — level.rho/level.vel
                # (NOT vel_out, which may be the ping-pong vel_temp; the refresh
                # hardcodes level.vel, so to match bitwise we read it too). NOTE
                # cs2: the SVV kernel's cs2_val IS CS2_LATTICE; for D3Q19 ==
                # CS2_PHYSICS == 1/3, so the fused feq matches the refresh.
                rho_macro = (_D3Q19_FUSE_PINEQ && has_pi_neq(level)) ? level.rho : nothing,
                vel_macro = (_D3Q19_FUSE_PINEQ && has_pi_neq(level)) ? level.vel : nothing,
                pi_neq_out = (_D3Q19_FUSE_PINEQ && has_pi_neq(level)) ? level.pi_neq : nothing,
                fuse_pineq = _D3Q19_FUSE_PINEQ && has_pi_neq(level))
            if _PHASE_TIMING
                KernelAbstractions.synchronize(backend)
                _phase_add!(_T_SPLIT_SVV, level.level_id, time() - _tsvv)
            end
        end
    end

    # ── SOTA A4: NSCBC outflow BC ──────────────────────────────────────
    # Apply the Poinsot-Lele characteristic outflow BC on the +x face of
    # L1 cells only. Replaces the legacy sponge-only outlet with a true
    # non-reflecting BC. Subsonic outflow: LODI pressure relaxation.
    # Supersonic outflow: pure extrapolation. See kernel_nscbc_outlet.jl.
    if OUTLET_NSCBC_ENABLED && level.level_id == 1
        # ── Global mass-conservation controller (2026-05-29) ──────────────
        # ROOT-CAUSE FIX for the recurring mass pump (see memory
        # project_mass_pump_root_cause). The BC system is over-constrained for
        # mass: hard-velocity inlet + impermeable slip lateral walls +
        # pressure-only NSCBC outlet ⇒ nothing enforces ∮ρu·n dA = 0, so during
        # the ramp the inlet injects mass faster than the pressure-relaxing
        # outlet vents it → monotonic domain-mass growth → blow-up.
        #
        # Fix: a proportional controller on the global L1 domain mass. Measure
        # the domain mass surplus relative to a baseline (captured once, after
        # the first few steps), and lower the outlet pressure target in
        # proportion so the EXISTING, validated LODI relaxation bleeds exactly
        # the excess. This is conservation-driven (responds to the real surplus)
        # and acts only through the physically-correct exit (the outlet), not a
        # blunt domain-wide ρ rescale. Gated by MASS_CONTROL_ENABLED.
        p_target_eff = Float32(OUTLET_NSCBC_P_TARGET)
        if MASS_CONTROL_ENABLED && MASS_CONTROL_MODE == :nscbc_ptarget
            domain_mass = Float64(sum(level.rho))   # cheap GPU reduction
            if _MASS_CTRL_BASELINE[] <= 0.0 && timestep >= MASS_CONTROL_BASELINE_STEP
                _MASS_CTRL_BASELINE[] = domain_mass       # capture baseline once
            end
            if _MASS_CTRL_BASELINE[] > 0.0
                surplus_rel = (domain_mass - _MASS_CTRL_BASELINE[]) / _MASS_CTRL_BASELINE[]
                # Proportional pull: high mass ⇒ lower outlet target ⇒ more outflux.
                # Bounded to ±20% of p_target so a transient can't invert the outlet.
                dp = clamp(MASS_CONTROL_GAIN * Float32(surplus_rel),
                           -0.2f0 * Float32(OUTLET_NSCBC_P_TARGET),
                            0.2f0 * Float32(OUTLET_NSCBC_P_TARGET))
                p_target_eff = Float32(OUTLET_NSCBC_P_TARGET) - dp
            end
        end
        apply_nscbc_outlet!(
            level, backend,
            cx_gpu, cy_gpu, cz_gpu, w_gpu;
            p_target = p_target_eff,
            sigma_relax = OUTLET_NSCBC_SIGMA,
            neq_blend = OUTLET_NSCBC_NEQ_BLEND,
            eos_R = EOS_THERMODYNAMIC ?
                Float32((DDF_GAMMA - 1.0) * DDF_CV) :
                Float32(-1.0f0),
            # 2026-05-28: pass γ so the LODI sound speed is c = √(γ·R·T)
            # (true thermodynamic) rather than the legacy √(R·T) which
            # under-estimated K_relax by √γ ≈ 1.18 at γ=1.4.
            gamma_eos = DDF_ENABLED ? Float32(DDF_GAMMA) : 1.0f0
        )
    end

    # ── Mass-conservation controller — :rescale mode (2026-05-30) ──────────
    # Decoupled from the outlet BC. The NSCBC characteristic outlet detonates
    # at Ma=0.84 (corner interaction with lateral walls — cases 044/044a both
    # blew up @step152 with NSCBC on, while the plain zerogradient-outlet 042
    # stayed stable). This path instead bleeds the global mass surplus by a
    # gentle uniform rescale of the L1 populations. Because f scales ρ AND
    # momentum j together, the velocity u = j/ρ is UNCHANGED — only the excess
    # mass is removed. Unconditionally stable: no BC, no characteristics, no
    # corner cells. Runs on L1 only, once per step.
    #
    # PROPORTIONAL controller (045-v2): the correction is a small FRACTION of the
    # current surplus, not a fixed-rate ramp toward the target. 045-v1 used
    # s = clamp(M_target/M_now, 1±ε) with ε=1e-3, which saturated the clamp every
    # step (the 0.1%/step cap is ~50× the pump rate) and monotonically over-drained
    # to ΔM=-6.7% by step 152 — stable, but wrong target tracking. The P-form
    #     s = 1 − k·surplus_rel        (k = MASS_CONTROL_GAIN, surplus_rel = ΔM/M)
    # is self-limiting: when ΔM≈0 the correction →0 (no overshoot); when ΔM grows
    # the correction grows. It exponentially decays the surplus to 0 with time
    # constant ~1/k steps. ε remains only as a hard safety rail.
    if MASS_CONTROL_ENABLED && MASS_CONTROL_MODE == :rescale && level.level_id == 1
        domain_mass = Float64(sum(level.rho))
        if _MASS_CTRL_BASELINE[] <= 0.0 && timestep >= MASS_CONTROL_BASELINE_STEP
            _MASS_CTRL_BASELINE[] = domain_mass
        end
        if _MASS_CTRL_BASELINE[] > 0.0 && domain_mass > 0.0
            surplus_rel = (domain_mass - _MASS_CTRL_BASELINE[]) / _MASS_CTRL_BASELINE[]
            # Cheap throttled mass log (the sum is already computed above — no extra
            # GPU pull, unlike the full compressible_audit which hangs the 4-level run).
            if timestep % 50 == 0
                @printf("          [MassCtrl] step=%d  surplus=%+.4f%%  (rescale eps=%.3f)\n",
                        timestep, 100.0 * surplus_rel, Float64(MASS_CONTROL_RESCALE_EPS))
            end
            k    = Float64(MASS_CONTROL_GAIN)               # fraction of surplus per step
            eps  = Float64(MASS_CONTROL_RESCALE_EPS)        # hard per-step safety clamp
            s    = LBM_F(clamp(1.0 - k * surplus_rel, 1.0 - eps, 1.0 + eps))
            if s != one(LBM_F)
                # Scale the LIVE post-collision buffer (f_out — the same array
                # the NSCBC/Bouzidi kernels read here), NOT level.f, which may be
                # a stale ping-pong buffer at this point. ρ→s·ρ, j→s·j ⇒ u
                # unchanged; bleeds only the excess mass.
                f_out .*= s
            end
        end
    end

    shifted_wall_scale_eff =
        (BOUZIDI_SHIFTED_WALL_CORRECTION && LATTICE_TYPE == :d3q39) ?
        BOUZIDI_SHIFTED_WALL_CORRECTION_SCALE : 0.0f0
    if shifted_wall_scale_eff > 0.0f0 && BOUZIDI_SHIFTED_WALL_CORRECTION_RAMP_STEPS > Int32(0)
        ramp = min(1.0f0, Float32(timestep) / Float32(max(BOUZIDI_SHIFTED_WALL_CORRECTION_RAMP_STEPS, Int32(1))))
        shifted_wall_scale_eff *= ramp
    end

    if level.bouzidi_enabled && level.n_boundary_cells > 0
        if _PHASE_TIMING
            KernelAbstractions.synchronize(backend)
            _twall = time()
        end
        wall_local_shift_active =
            (LOCAL_LATTICE_SHIFT_ENABLED && LATTICE_TYPE == :d3q39) ? Int32(1) : Int32(0)
        # Note: Bouzidi processes ALL boundary cells even in multi-GPU mode,
        # because boundary cells are indexed by position, not by block partition.
        # The cell_offset=0 default is correct since both GPUs have full arrays.
        #
        # B7 — gold-standard wall BC ladder. Dispatch on BC_SCHEME:
        #   :grad             → Grad-reconstruction (regularized) wall — q-free,
        #                       normal-free, axis-agnostic; rebuilds all wall-cell
        #                       pops from (ρ, u_wall, Π_neq). Cures the link-wise
        #                       curved-wall Nyquist ring + axis-tangent hot spots.
        #   :bouzidi_linear   → legacy 2-point linear Bouzidi (BFL 2001)
        #   :ymls_quadratic   → 3-point quadratic (BFL 2001 / YMLS 2003)
        #   :marson_enriched  → YMLS + Marson 2021 isotropy-correcting term
        if BC_SCHEME == :grad
            apply_grad_wall_correction!(
                f_out, f_post_collision, level.rho, vel_out, level.wall_dist,
                level.bouzidi_q_map,
                level.bouzidi_cell_block, level.bouzidi_cell_x,
                level.bouzidi_cell_y, level.bouzidi_cell_z,
                level.n_boundary_cells, BLOCK_SIZE,
                cx_gpu, cy_gpu, cz_gpu, w_gpu,
                Float32(CS2_LATTICE), Float32(CS4_LATTICE), Int(N_DIRS), backend;
                omega_wall = 1.0f0 / Float32(level.tau),
                grad_pineq_scale = GRAD_WALL_PINEQ_SCALE,
                grad_vel_mode = GRAD_WALL_VELOCITY_MODE,
                grad_blend_dx = 1.0f0,
                write_post_collision = Int32(1))   # this branch only runs when bouzidi_enabled && n_boundary>0, i.e. the flow kernel already wrote f_post_collision; keep forces/restriction consistent with the Grad wall.
        elseif BC_SCHEME == :bouzidi_linear
            apply_bouzidi_correction!(
                f_out, f_post_collision, level.obstacle,
                level.bouzidi_q_map, level.bouzidi_cell_block,
                level.bouzidi_cell_x, level.bouzidi_cell_y, level.bouzidi_cell_z,
                level.n_boundary_cells, level.neighbor_table, BLOCK_SIZE,
                cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, Q_MIN_THRESHOLD, backend;
                wall_eq_arr = f_eq_wall_gpu,
                wall_eq_frame_count = wall_eq_frame_count,
                shifted_wall_correction_scale = shifted_wall_scale_eff,
                shifted_wall_correction_cap_rel = BOUZIDI_SHIFTED_WALL_CORRECTION_CAP_REL,
                shifted_wall_correction_cap_abs = BOUZIDI_SHIFTED_WALL_CORRECTION_CAP_ABS,
                wall_pop_limiter_rel_cap = BOUZIDI_WALL_POP_LIMITER_REL_CAP,
                wall_pop_limiter_abs_cap = BOUZIDI_WALL_POP_LIMITER_ABS_CAP,
                local_shift_active = wall_local_shift_active,
                block_shift_arr = level.block_shift,
                ff_fallback_mode = BOUZIDI_FF_FALLBACK_MODE
            )
        else
            apply_bouzidi_ymls_correction!(
                f_out, f_post_collision, level.obstacle,
                level.bouzidi_q_map, level.bouzidi_cell_block,
                level.bouzidi_cell_x, level.bouzidi_cell_y, level.bouzidi_cell_z,
                level.n_boundary_cells, level.neighbor_table, BLOCK_SIZE,
                cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, Q_MIN_THRESHOLD, backend;
                ff_fallback_mode = BOUZIDI_FF_FALLBACK_MODE,
                wall_eq_arr = f_eq_wall_gpu,
                wall_eq_frame_count = wall_eq_frame_count,
                shifted_wall_correction_scale = shifted_wall_scale_eff,
                shifted_wall_correction_cap_rel = BOUZIDI_SHIFTED_WALL_CORRECTION_CAP_REL,
                shifted_wall_correction_cap_abs = BOUZIDI_SHIFTED_WALL_CORRECTION_CAP_ABS,
                wall_pop_limiter_rel_cap = BOUZIDI_WALL_POP_LIMITER_REL_CAP,
                wall_pop_limiter_abs_cap = BOUZIDI_WALL_POP_LIMITER_ABS_CAP,
                local_shift_active = wall_local_shift_active,
                block_shift_arr = level.block_shift,
                marson_enriched = (BC_SCHEME == :marson_enriched) ? Int32(1) : Int32(0)
            )
        end
        # PERF (2026-06-04): removed host synchronize — the next op is the
        # missed-link-fix kernel (same backend, KA-ordered) or the thermal/next
        # level kernel. No host read of f_out between. Verified removable.

        if BOUZIDI_MISSED_LINK_FIX_MODE != Int32(0)
            apply_bouzidi_missed_link_fix!(
                f_out, f_post_collision, level.obstacle,
                level.bouzidi_q_map, level.bouzidi_cell_block,
                level.bouzidi_cell_x, level.bouzidi_cell_y, level.bouzidi_cell_z,
                level.n_boundary_cells, level.neighbor_table, BLOCK_SIZE,
                cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, Q_MIN_THRESHOLD, backend;
                wall_eq_arr = f_eq_wall_gpu,
                wall_eq_frame_count = wall_eq_frame_count,
                shifted_wall_correction_scale = shifted_wall_scale_eff,
                shifted_wall_correction_cap_rel = BOUZIDI_SHIFTED_WALL_CORRECTION_CAP_REL,
                shifted_wall_correction_cap_abs = BOUZIDI_SHIFTED_WALL_CORRECTION_CAP_ABS,
                local_shift_active = wall_local_shift_active,
                block_shift_arr = level.block_shift,
                missed_link_fix_mode = BOUZIDI_MISSED_LINK_FIX_MODE,
                missed_link_eq_blend = BOUZIDI_MISSED_LINK_EQ_BLEND,
                missed_link_delta_rel_cap = BOUZIDI_MISSED_LINK_DELTA_REL_CAP,
                missed_link_delta_abs_cap = BOUZIDI_MISSED_LINK_DELTA_ABS_CAP
            )
            # PERF (2026-06-04): removed host synchronize — what follows is more
            # GPU kernel work on the same backend (thermal DDF / next level step),
            # which KernelAbstractions auto-orders. No host read here. Confirmed
            # removable by the sync-classification workflow (classifier+verifier).
        end
        # FUSED-Π Bouzidi mop-up (2026-06-13): the Bouzidi correction just
        # modified body-cell f AFTER SVV wrote their fused Π. Re-run the refresh
        # over the (cached, unique) Bouzidi-touched blocks so their Π reflects
        # the post-Bouzidi f. Idempotent ⇒ bitwise for those blocks; the
        # SVV-fused Π elsewhere is untouched. copy_scratch=false: the scratch
        # rotation already ran pre-timestep; re-rotating here would overwrite
        # the previous-step value with the new one.
        if _D3Q19_FUSE_PINEQ && has_pi_neq(level) && level.n_boundary_cells > 0
            _bz = _d3q19_bouzidi_block_list(level, backend)
            if _bz.n > 0
                refresh_pi_neq!(level, f_out, cx_gpu, cy_gpu, cz_gpu, w_gpu, backend;
                                shift_x=Float32(LATTICE_SHIFT_X),
                                shift_y=Float32(LATTICE_SHIFT_Y),
                                shift_z=Float32(LATTICE_SHIFT_Z),
                                covered_band=_bz.list,
                                copy_scratch=false)
            end
        end
    end

    # ── Thermal DDF kernel (Phase 2/4b Frapolli) ──
    if _PHASE_TIMING && level.bouzidi_enabled && level.n_boundary_cells > 0
        KernelAbstractions.synchronize(backend)
        _phase_add!(_T_SPLIT_WALL, level.level_id, time() - _twall)
    end

    update_local_lattice_shift!(level, backend, vel_out;
                                f_current=f_out,
                                rho_current=level.rho,
                                f_history=(length(level.f_old) > N_DIRS ? level.f_old : nothing),
                                rho_history=(length(level.rho_old) > 1 ? level.rho_old : level.rho),
                                vel_history=(length(level.vel_old) > 3 ? level.vel_old : vel_out),
                                u_frame_ref=u_curr)

    if DDF_ENABLED && g_out !== nothing && g_in !== nothing && has_thermal(level)
        # Parent thermal data
        if is_l1
            pg = g_in; pT = level.temperature; pvel_t = level.vel; pke_t = level.kinetic_energy; pptr_t = level.block_pointer
            pg_old = g_in; pT_old = level.temperature; pvel_old_t = level.vel
            pgx, pgy, pgz = Int32(1), Int32(1), Int32(1)
        else
            pg = parent_g; pT = parent_temperature; pvel_t = parent_vel_thermal !== nothing ? parent_vel_thermal : p_vel
            pke_t = parent_kinetic_energy !== nothing ? parent_kinetic_energy : p_rho
            pg_old = parent_g_old; pT_old = parent_temperature_old
            pvel_old_t = parent_vel_old_thermal !== nothing ? parent_vel_old_thermal : p_vel_old
            pptr_t = p_ptr
            pgx, pgy, pgz = px, py, pz
        end

        # Thermal wall BC flag
        wall_bc_adiabatic_i = DDF_WALL_BC == :adiabatic ? Int32(1) : Int32(0)

        # Phase 4b: thermal kernel uses SAME lattice arrays as flow (D3Q39 or D3Q27)
        # No separate D3Q27 thermal arrays needed anymore.
        # Frapolli model parameters
        is_frapolli = (LATTICE_TYPE == :d3q39)
        cs2_thermal = is_frapolli ? Float32(2.0/3.0) : Float32(1.0/3.0)
        use_macro_temporal_interp_thermal =
            (!is_l1 && INTERFACE_MACRO_TEMPORAL_INTERPOLATION &&
             parent_temperature_old !== nothing && parent_vel_old_thermal !== nothing &&
             length(parent_temperature_old) > 1 && length(parent_vel_old_thermal) > 1 &&
             length(p_rho_old) > 1) ? Int32(1) : Int32(0)

        _tt = _PHASE_TIMING ? time() : 0.0
        thermal_kernel! = stream_collide_thermal!(backend)
        thermal_kernel!(
            g_out, g_in, g_post_collision_arr !== nothing ? g_post_collision_arr : level.g_post_collision,
            level.temperature, level.rho, vel_out,
            level.obstacle, level.sponge, level.wall_dist,
            level.neighbor_table,
            level.map_x, level.map_y, level.map_z,
            pg, pT, p_rho, pvel_t, pke_t, pptr_t,
            pg_old, pT_old, p_rho_old, pvel_old_t,
            pgx, pgy, pgz,
            level.tau_g,
            parent_tau_g,
            level.tau,
            parent_tau,
            is_l1 ? Int32(1) : Int32(0),
            GROUND_ANALYSIS ? Int32(1) : Int32(0),
            nx_g, ny_g, nz_g, u_curr,
            Float32(DDF_T_INLET),
            Float32(DDF_T_WALL),
            wall_bc_adiabatic_i,
            Int32(n_blocks), Int32(BLOCK_SIZE),
            cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
            (level.bouzidi_enabled && level.n_boundary_cells > 0) ? Int32(1) : Int32(0),
            Float32(temporal_weight),
            use_temporal_interp ? Int32(1) : Int32(0),
            use_macro_temporal_interp_thermal,
            sponge_blend_dist ? Int32(1) : Int32(0),
            # Phase 4b: SGS thermal diffusivity parameters
            c_wale_val,
            nu_sgs_bg,
            amr_sgs_scale,
            Float32(DDF_TURBULENT_PRANDTL),
            # Phase 4b: Frapolli model parameters
            N_DIRS_THERMAL,                                              # n_dirs_thermal
            is_frapolli ? level.kinetic_energy : level.rho,              # kinetic_energy_arr (dummy=rho for D3Q27)
            is_frapolli ? Float32(DDF_CV) : 0.0f0,                      # Cv (0.0 = legacy mode)
            cs2_thermal,                                                  # cs2_thermal
            Float32(LATTICE_SHIFT_X),                                    # shift_x
            Float32(LATTICE_SHIFT_Y),                                    # shift_y
            Float32(LATTICE_SHIFT_Z),                                    # shift_z
            f_eq_inlet_gpu,                                              # numerical D3Q39 freestream equilibrium
            # Multi-GPU block offset (0 for single-GPU)
            block_offset,
            # Phase B4: adaptive sigma_even for thermal kernel (uses same pi_neq field)
            level.pi_neq,
            VARIABLE_SIGMA_THERMAL ? Int32(1) : Int32(0),
            Float32(VARIABLE_SIGMA_TH_MIN),
            Float32(VARIABLE_SIGMA_TH_MAX),
            Float32(VARIABLE_SIGMA_TH_PI_REF),
            # AMR feather-band thermal cleanup mirrors the flow-side frontier
            # damping, but acts on thermal diffusivity and odd heat-flux modes.
            level.feather_weight,
            FEATHER_NU_MAX,
            # 2026-05-28 — `level_id > 1` gate removed for symmetric AMR
            # interface treatment (see matching flow-side change at L274).
            # The feather_weight array itself encodes which cells need
            # damping; gating on level_id was redundant + biased.
            (FEATHER_DEPTH > 0 && FEATHER_NU_MAX > 0.0f0) ? Int32(1) : Int32(0),
            # SOTA #1: total-energy source terms (pressure-work + viscous heating).
            # Active only for Frapolli/D3Q39 when ENERGY_SOURCE_TERMS is set.
            (ENERGY_SOURCE_TERMS && is_frapolli) ? Int32(1) : Int32(0),
            Float32(DDF_GAMMA),
            # AUDIT FIX (2026-06-03): effective flow-EOS pressure coefficient for
            # the energy-source pressure-work term. Must match what the flow
            # equilibrium uses: (γ−1)·Cv when EOS_THERMODYNAMIC is on, else the
            # linearized lattice cs² (cs2_thermal). Keeps −p·∇·u consistent with
            # the momentum-side pressure.
            EOS_THERMODYNAMIC ? Float32((DDF_GAMMA - 1.0) * DDF_CV) : cs2_thermal,
            # AUDIT FIX (2026-06-03): explicit T-recovery cap for high-T Riemann
            # states (0 = auto formula; aero unchanged).
            Float32(DDF_T_ADMISSIBLE_MAX),
            # Intensive-T advection fix (2026-06-29): rescale streamed g by ρ_local/ρ_up.
            THERMAL_INTENSIVE_ADVECTION ? Int32(1) : Int32(0),
            ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, ndrange_blocks > 0 ? ndrange_blocks : n_blocks)
        )
        KernelAbstractions.synchronize(backend)
        if _PHASE_TIMING; _T_THERMAL[] += time() - _tt; end

        # Thermal wall links need the thermal wall condition, not the no-slip
        # momentum interpolation. For adiabatic walls, reflect the outgoing
        # thermal population to enforce zero normal heat flux.
        if level.bouzidi_enabled && level.n_boundary_cells > 0
            if wall_bc_adiabatic_i == Int32(1)
                apply_thermal_adiabatic_bouzidi_correction!(
                    g_out, g_post_collision_arr !== nothing ? g_post_collision_arr : level.g_post_collision,
                    level.bouzidi_q_map, level.bouzidi_cell_block,
                    level.bouzidi_cell_x, level.bouzidi_cell_y, level.bouzidi_cell_z,
                    level.n_boundary_cells, opp_gpu,
                    Q_MIN_THRESHOLD, N_DIRS_THERMAL, backend
                )
            else
                # Pass193: thermal isothermal Bouzidi keeps the legacy fallback
                # (mode=0). The macroscopic-Nyquist injection diagnosed in
                # PASS192_DIAGNOSIS.md is a momentum-side artifact; the
                # thermal-side fallback w_k * Σg would imply a "wall temperature
                # equal to the local fluid bulk temperature" which is not the
                # isothermal wall semantics. Revisit only if pass193 confirms
                # the flow-side fix works and an A/B isolates a thermal-side
                # residual.
                apply_bouzidi_correction!(
                    g_out, g_post_collision_arr !== nothing ? g_post_collision_arr : level.g_post_collision,
                    level.obstacle,
                    level.bouzidi_q_map, level.bouzidi_cell_block,
                    level.bouzidi_cell_x, level.bouzidi_cell_y, level.bouzidi_cell_z,
                    level.n_boundary_cells, level.neighbor_table, BLOCK_SIZE,
                    cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, Q_MIN_THRESHOLD, backend;
                    ff_fallback_mode = Int32(0)
                )
            end
            KernelAbstractions.synchronize(backend)
        end

        # Thermal population regularization is applied inside stream_collide_thermal!:
        # even modes use the partial-mirror sigma path, and odd heat-flux modes get
        # a small AMR-aware cleanup in refinement-frontier/wall bands. The old full
        # post-kernel g regularizer remains disabled because it overwrote useful
        # thermal non-equilibrium everywhere.
    end
end

"""
Build GPU-resident lattice arrays for D3Q27 (default).
"""
function build_lattice_arrays_gpu_d3q27(backend)
    cx, cy, cz, w = Int32[], Int32[], Int32[], Float32[]

    for dz in -1:1, dy in -1:1, dx in -1:1
        push!(cx, Int32(dx)); push!(cy, Int32(dy)); push!(cz, Int32(dz))
        d2 = dx^2 + dy^2 + dz^2
        push!(w, d2==0 ? 8f0/27f0 : d2==1 ? 2f0/27f0 : d2==2 ? 1f0/54f0 : 1f0/216f0)
    end

    opp, mirror_y, mirror_z = zeros(Int32, 27), zeros(Int32, 27), zeros(Int32, 27)
    for i in 1:27, j in 1:27
        if cx[j]==-cx[i] && cy[j]==-cy[i] && cz[j]==-cz[i]; opp[i] = Int32(j); end
        if cx[j]==cx[i] && cy[j]==-cy[i] && cz[j]==cz[i]; mirror_y[i] = Int32(j); end
        if cx[j]==cx[i] && cy[j]==cy[i] && cz[j]==-cz[i]; mirror_z[i] = Int32(j); end
    end

    return (adapt(backend, cx), adapt(backend, cy), adapt(backend, cz), adapt(backend, w),
            adapt(backend, opp), adapt(backend, mirror_y), adapt(backend, mirror_z))
end

"""
Build GPU-resident lattice arrays for D3Q19 (standard incompressible-LBM).
Uses the static D3Q19 data from `build_d3q19_static()` in lattice.jl.
No Galilean shift support (D3Q19 is athermal/incompressible — Ma_lattice is
bounded by ~0.1 and the shifted-velocity-set extension that exists for D3Q39
would have no benefit here).
"""
function build_lattice_arrays_gpu_d3q19(backend)
    cx_s, cy_s, cz_s, w_s, opp_s, my_s = build_d3q19_static()
    n = 19
    cx  = Vector{Int32}(cx_s)
    cy  = Vector{Int32}(cy_s)
    cz  = Vector{Int32}(cz_s)
    w   = Vector{Float32}(w_s)
    opp = Vector{Int32}(opp_s)
    mirror_y = Vector{Int32}(my_s)

    # mirror_z mapping built on the velocity set (same convention as D3Q27/D3Q39)
    mirror_z = zeros(Int32, n)
    for i in 1:n, j in 1:n
        if cx_s[j]==cx_s[i] && cy_s[j]==cy_s[i] && cz_s[j]==-cz_s[i]
            mirror_z[i] = Int32(j)
        end
    end

    return (adapt(backend, cx), adapt(backend, cy), adapt(backend, cz), adapt(backend, w),
            adapt(backend, opp), adapt(backend, mirror_y), adapt(backend, mirror_z))
end

"""
Build GPU-resident lattice arrays for D3Q39 Variant B (Frapolli/Karlin).
Uses the static lattice data from lattice.jl.
Applies Galilean shift from LATTICE_SHIFT_X/Y/Z globals (Phase 3.2).
Weights, OPP, and mirror mappings are index-based and invariant under the shift.
"""
function build_lattice_arrays_gpu_d3q39(backend)
    # Build from the static D3Q39 definition
    cx_s, cy_s, cz_s, w_s, opp_s, my_s = build_d3q39_static()

    n = 39
    # Apply Galilean shift to velocities (weights, OPP, mirror_y are invariant)
    cx  = Vector{Int32}(cx_s) .+ LATTICE_SHIFT_X
    cy  = Vector{Int32}(cy_s) .+ LATTICE_SHIFT_Y
    cz  = Vector{Int32}(cz_s) .+ LATTICE_SHIFT_Z
    w   = Vector{Float32}(w_s)
    opp = Vector{Int32}(opp_s)
    mirror_y = Vector{Int32}(my_s)

    # Build mirror_z mapping on UNSHIFTED velocities (index-based, shift-invariant)
    mirror_z = zeros(Int32, n)
    for i in 1:n, j in 1:n
        if cx_s[j]==cx_s[i] && cy_s[j]==cy_s[i] && cz_s[j]==-cz_s[i]
            mirror_z[i] = Int32(j)
        end
    end

    return (adapt(backend, cx), adapt(backend, cy), adapt(backend, cz), adapt(backend, w),
            adapt(backend, opp), adapt(backend, mirror_y), adapt(backend, mirror_z))
end

"""
Dispatch: build correct GPU lattice arrays based on LATTICE_TYPE.
"""
function build_lattice_arrays_gpu(backend)
    if LATTICE_TYPE == :d3q39
        return build_lattice_arrays_gpu_d3q39(backend)
    elseif LATTICE_TYPE == :d3q19
        return build_lattice_arrays_gpu_d3q19(backend)
    else
        return build_lattice_arrays_gpu_d3q27(backend)
    end
end
