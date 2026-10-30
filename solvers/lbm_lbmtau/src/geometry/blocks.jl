// # FILE: .\src\blocks.jl
"""
BLOCKS.JL - Block-Level Data Structures with Temporal Interpolation Support

Added f_old, rho_old, vel_old arrays for temporal interpolation at refinement interfaces.
This enables smooth boundary conditions for fine grids instead of discontinuous jumps.
"""

using KernelAbstractions
using CUDA
using Adapt
using StaticArrays

const BLOCK_SIZE = 8

mutable struct BlockLevel{T_P4, T_P5, T_BlockPtr, T_NbTable, T_Obst, T_F16, T_I8, T_IntVec, T_MapVec, T_TriMap}
    level_id::Int
    dx::Float64
    dt::Float32
    tau::Float32

    grid_dim_x::Int
    grid_dim_y::Int
    grid_dim_z::Int

    block_pointer::T_BlockPtr
    active_block_coords::Vector{Tuple{Int, Int, Int}}

    # -- MACROSCOPIC VARS --
    rho::T_P4
    vel::T_P5
    vel_temp::T_P5

    # -- TEMPORAL INTERPOLATION STORAGE --
    # Pass176 (Tier A item A2 - quadratic temporal interpolation):
    # `*_older` arrays hold the parent's state from ONE parent step prior to
    # `*_old`. Combined with `_old` (current parent step start) and the
    # current arrays (= new, parent step end), they give 3 anchor points for
    # quadratic Lagrange temporal interpolation at fine sub-cycle times,
    # raising temporal continuity from C⁰ to C¹. Only macros are tracked
    # in `older` to keep memory cost ~40MB at SR=256 (full population older
    # would cost ~310MB which doesn't fit the 4 GB target).
    #
    # B1 Pillar 3 (2026-05-28) — CUBIC HERMITE temporal interpolation.
    # `*_oldest` adds a fourth anchor at t = -2·dt_c, completing the chain
    # {oldest, older, old, new} at times {-2·dt_c, -dt_c, 0, +dt_c}. The
    # cubic Lagrange polynomial through these four points evaluated at the
    # fine sub-cycle time t = tw·dt_c raises temporal continuity from C¹ to
    # C². Per the SOTA_ROADMAP standing directive on AMR coupling: this is
    # the temporal piece of the gold-standard mortar-element scheme.
    # Allocated only when INTERFACE_CUBIC_TEMPORAL is set; otherwise the
    # *_oldest arrays are 1×1×1×1 stubs and the kernel falls back to the
    # existing quadratic / linear paths.
    # B1 P3 RK4-equivalent (2026-05-28) — 5th historical anchor at t = -3·dt_c.
    # Extends the cubic Hermite chain to a 5-anchor quintic Lagrange in time
    # {oldest2, oldest, older, old, new} at {-3·dt_c, -2·dt_c, -dt_c, 0, +dt_c}.
    # Raises temporal accuracy at the AMR halo from C² (cubic, B1 P3) to C³
    # (quintic). The "RK4-consistent sub-cycling" half of P3 in the standing
    # directive — pragmatically equivalent for AMR-coupling purposes to a
    # 4-stage RK4 collision restructure (which would be a multi-week BGK
    # operator refactor; see CLAUDE.md "B1 P3 RK4" section). Allocated only
    # when INTERFACE_QUINTIC_TEMPORAL is set; otherwise 1×1×1×1 stubs.
    rho_oldest2::T_P4
    vel_oldest2::T_P5
    rho_oldest::T_P4
    vel_oldest::T_P5
    rho_older::T_P4
    vel_older::T_P5
    rho_old::T_P4
    vel_old::T_P5

    # -- DISTRIBUTIONS --
    f::T_P5
    f_temp::T_P5
    f_post_collision::T_P5
    f_old::T_P5

    # -- THERMAL DDF (Phase 2/4b Frapolli) --
    tau_g::Float32
    g::T_P5
    g_temp::T_P5
    g_post_collision::T_P5
    g_old::T_P5
    temperature::T_P4
    temperature_old::T_P4
    temperature_older::T_P4   # Pass176: 2-step-old temperature for quadratic temporal blend
    temperature_oldest::T_P4  # B1 P3 (2026-05-28): 3-step-old temperature for cubic Hermite temporal blend (stub when INTERFACE_CUBIC_TEMPORAL=false)
    temperature_oldest2::T_P4 # B1 P3 RK4-equivalent (2026-05-28): 4-step-old temperature for quintic Lagrange temporal blend (stub when INTERFACE_QUINTIC_TEMPORAL=false)
    kinetic_energy::T_P4        # Phase 4b: Σf·|c|² from flow kernel (for energy-based T recovery)
    temperature_filter_scratch::T_P4   # Scratch buffer for post-recovery T laplacian filter (stub when disabled)

    # -- Π_neq (Phase B): persistent non-equilibrium stress tensor field --
    # Layout (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 6): components (xx,yy,zz,xy,xz,yz).
    # Allocated only when VARIABLE_NU_SGS=true OR STRESS_TENSOR_TRANSFER=true; otherwise a stub.
    # Refreshed every timestep (post-collision) and consumed by the next collision
    # and by the cross-level prolongation (moment-based stress carrier).
    pi_neq::T_P5
    # Scratch buffer for the Lagrava q-direction filter (item 5, paper-quality).
    # Same shape as pi_neq when allocated; filter reads pi_neq → writes scratch,
    # then pi_neq is overwritten with the filtered field. Allocated only when
    # pi_neq itself is full-sized.
    pi_neq_scratch::T_P5

    # -- C2/C4 quintic B-spline prolongation coefficients (2026-06-04) --
    # B-spline coefficients c of the level's macros (c = A⁻¹·P, computed by a
    # per-coarse-step prefilter pass), consumed by child prolongation for a
    # C4-continuous + exactly-2:1-conservative coarse→fine macro interpolation.
    # Layout matches pi_neq (block-structured, BS,BS,BS,n_blocks[,3]). Allocated
    # full-size only when QUINTIC_BSPLINE_PROLONGATION; else a stub. The *_scratch
    # arrays ping-pong the 3 separable axis passes (x→y→z), like pi_neq/_scratch.
    rho_spline_coeff::T_P4
    rho_spline_scratch::T_P4
    vel_spline_coeff::T_P5
    vel_spline_scratch::T_P5

    # -- Lever 1b: once-per-coarse-step prolonged-halo macro+Π cache (2026-06-20) --
    # The AMR coarse→fine halo prolongation (macro B-spline + Π trilinear) is LINEAR
    # in the parent state and re-run every fine substep with only the scalar
    # temporal_weight changing. These cache the FULLY-PROLONGED fine-halo macros
    # (rho,ux,uy,uz) and Π (6 comps) at the two coarse-step anchors (new-time, old-
    # time), computed ONCE per coarse step. Each fine substep then reads the 2 anchors
    # + scalar-lerps + does the nonlinear feq+stress reconstruction — replacing the
    # per-substep 6-tap tile eval + prefilter. Indexed by FINE cell (this level's own
    # block grid), so allocated full-size only on levels WITH A PARENT (level_id>1)
    # when the hoist is active; else a 1×1×1×1×1 stub. Layout (BS,BS,BS,n_blocks,comp).
    halo_macro_new::T_P5      # 4 comps: rho,ux,uy,uz at new-time anchor
    halo_macro_old::T_P5      # 4 comps: rho,ux,uy,uz at old-time anchor
    halo_pi_new::T_P5         # 6 comps: pxx,pyy,pzz,pxy,pxz,pyz at new-time
    halo_pi_old::T_P5         # 6 comps: … at old-time

    # -- Tier B4: Newton solver warm-start cache (2026-05-28) --
    # Per-cell storage for the 13-mom Newton solver's converged λ from the
    # PREVIOUS timestep. Read at Newton entry as init guess; written back
    # after Newton completes. Cells in steady state get λ that's already
    # near-converged, Newton finishes in 1-3 iters instead of 5-10.
    # Layout: (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 13).
    # Initial value: NaN sentinel — Newton solver detects this and falls
    # back to the polynomial init guess for the very first step.
    lambda_cache::T_P5

    # -- GEOMETRY / SPONGE --
    wall_dist::T_P4
    obstacle::T_Obst
    sponge::T_P4
    amr_covered::T_Obst

    # -- AMR-interface feather band --
    # Per-cell Float32 in [0, 1]. Nonzero only in an N-cell-thick band
    # adjacent to the fine-level's outer boundary (cells whose block has
    # at least one inactive neighbor block). 1.0 at the boundary face,
    # decays linearly to 0 at distance `feather_depth`. Collision kernel
    # uses this as a weight on an extra viscous dissipation contribution
    # (layered via max() with WALE / variable_ν_sgs / Knudsen-τ) so the
    # O(1e-4) density seam that the prolongation-streaming mix injects
    # at the halo is damped out before it can propagate inward into the
    # wake or stagnation region. Allocated as a 1×1×1×1 stub when
    # feather_depth=0 to avoid memory overhead on cases that disable it.
    feather_weight::T_P4

    # -- D3Q19 staged AMR halo populations --
    # Scratch packet for the staged coarse/fine halo split. Components 1:19
    # hold the pulled D3Q19 populations; component 20 is a finite/NaN cell-valid
    # sentinel. The small halo collision kernel consumes the packet without
    # carrying the prolongation machinery in its register frame. Stubbed unless
    # the D3Q19 AMR split is active on an AMR child level.
    amr_halo_pop_scratch::T_P5

    # -- AMR GPU face budget --
    # Optional x-face space-time register. Component layout:
    # x+ parent mass, child mass, parent mom, child mom, parent time, child time,
    # then the same six quantities for x-. Allocated only when a guarded
    # GPU face-budget path needs the register.
    amr_face_budget::T_P5

    # -- Local shifted lattice --
    # Per-block integer Galilean x-shift used by D3Q39 local-frame streaming.
    # `block_shift_next` is filled from the latest velocity field, then the
    # populations are relabelled before it becomes the active frame.
    block_shift::T_IntVec
    block_shift_next::T_IntVec
    # Blocks that own Bouzidi wall-link cells. This supports the optional
    # compatibility wall guard; normal local-shift runs now use per-frame q-maps.
    local_shift_wall_guard::T_IntVec

    # -- CONNECTIVITY --
    neighbor_table::T_NbTable
    map_x::T_MapVec
    map_y::T_MapVec
    map_z::T_MapVec

    # -- BOUZIDI IBM DATA --
    bouzidi_enabled::Bool
    bouzidi_q_map::T_F16
    bouzidi_cell_block::T_IntVec
    bouzidi_cell_x::T_I8
    bouzidi_cell_y::T_I8
    bouzidi_cell_z::T_I8
    # MAPPING TO SURFACE TRIANGLES (New)
    bouzidi_tri_map::T_TriMap
    n_boundary_cells::Int
end

function Adapt.adapt_structure(to, level::BlockLevel)
    BlockLevel(
        level.level_id, level.dx, level.dt, level.tau,
        level.grid_dim_x, level.grid_dim_y, level.grid_dim_z,
        adapt(to, level.block_pointer),
        level.active_block_coords,
        adapt(to, level.rho), adapt(to, level.vel), adapt(to, level.vel_temp),
        adapt(to, level.rho_oldest2), adapt(to, level.vel_oldest2),  # B1 P3 RK4 (quintic anchor at t=-3·dt_c)
        adapt(to, level.rho_oldest), adapt(to, level.vel_oldest),    # B1 P3
        adapt(to, level.rho_older), adapt(to, level.vel_older),
        adapt(to, level.rho_old), adapt(to, level.vel_old),
        adapt(to, level.f), adapt(to, level.f_temp),
        adapt(to, level.f_post_collision), adapt(to, level.f_old),
        level.tau_g,
        adapt(to, level.g), adapt(to, level.g_temp),
        adapt(to, level.g_post_collision), adapt(to, level.g_old),
        adapt(to, level.temperature), adapt(to, level.temperature_old), adapt(to, level.temperature_older),
        adapt(to, level.temperature_oldest),    # B1 P3
        adapt(to, level.temperature_oldest2),   # B1 P3 RK4 (quintic anchor for T)
        adapt(to, level.kinetic_energy),
        adapt(to, level.temperature_filter_scratch),
        adapt(to, level.pi_neq),
        adapt(to, level.pi_neq_scratch),
        adapt(to, level.rho_spline_coeff), adapt(to, level.rho_spline_scratch),  # C2/C4 B-spline
        adapt(to, level.vel_spline_coeff), adapt(to, level.vel_spline_scratch),
        adapt(to, level.halo_macro_new), adapt(to, level.halo_macro_old),  # lever 1b hoist cache
        adapt(to, level.halo_pi_new), adapt(to, level.halo_pi_old),
        adapt(to, level.lambda_cache),     # B4 (2026-05-28) — Newton warm-start cache
        adapt(to, level.wall_dist), adapt(to, level.obstacle), adapt(to, level.sponge),
        adapt(to, level.amr_covered),
        adapt(to, level.feather_weight),
        adapt(to, level.amr_halo_pop_scratch),
        adapt(to, level.amr_face_budget),
        adapt(to, level.block_shift),
        adapt(to, level.block_shift_next),
        adapt(to, level.local_shift_wall_guard),
        adapt(to, level.neighbor_table),
        adapt(to, level.map_x), adapt(to, level.map_y), adapt(to, level.map_z),
        level.bouzidi_enabled,
        adapt(to, level.bouzidi_q_map),
        adapt(to, level.bouzidi_cell_block),
        adapt(to, level.bouzidi_cell_x), adapt(to, level.bouzidi_cell_y), adapt(to, level.bouzidi_cell_z),
        adapt(to, level.bouzidi_tri_map),
        level.n_boundary_cells
    )
end

function BlockLevel(level_id::Int,
                    active_coords::Vector{Tuple{Int, Int, Int}},
                    neighbor_table::Matrix{Int32},
                    dx::AbstractFloat, dt::AbstractFloat, tau::AbstractFloat;
                    bouzidi_q_map = nothing,
                    bouzidi_cell_block = nothing,
                    bouzidi_cell_x = nothing,
                    bouzidi_cell_y = nothing,
                    bouzidi_cell_z = nothing,
                    bouzidi_tri_map = nothing,
                    n_boundary_cells = 0,
                    enable_temporal_interpolation = true,
                    enable_macro_temporal_interpolation = false,
                    # PERF (2026-06-04): the finest level is never an AMR
                    # prolongation SOURCE (it has no children to prolongate to),
                    # so its B-spline spline-coefficient arrays are never written
                    # or read. Pass has_children=false for the finest level to
                    # allocate the 1×1×1×1 stub instead (saves ~133 MiB on the
                    # Z132 A250 case, typically the largest block count). Default
                    # true preserves behavior for all existing callers.
                    has_children = true,
                    # Periodic-Y AMR (27 Oct 2026): when > 0, the block_pointer's Y
                    # extent is the level's FULL grid height (the periodic period in
                    # blocks) instead of the highest active block. Every kernel takes
                    # its Y period from size(block_pointer, 2) (parent_dim_y,
                    # grid_dim_y, the B-spline prefilter's by_max), so a level that
                    # does not reach the top of the domain must still report the true
                    # period. 0 (default) ⇒ legacy sizing, byte-identical.
                    periodic_grid_by::Int = 0)

    n_blocks = length(active_coords)

    if isempty(active_coords)
        bx_max, by_max, bz_max = 0, 0, 0
    else
        bx_max = maximum(c[1] for c in active_coords)
        by_max = maximum(c[2] for c in active_coords)
        bz_max = maximum(c[3] for c in active_coords)
    end
    if periodic_grid_by > 0 && n_blocks > 0
        by_max >= 1 && by_max <= periodic_grid_by ||
            error("BlockLevel $level_id: active Y block $by_max outside the periodic grid of $periodic_grid_by blocks")
        by_max = periodic_grid_by
    end
    
    block_pointer = fill(Int32(0), bx_max, by_max, bz_max)
    for (i, (bx, by, bz)) in enumerate(active_coords)
        block_pointer[bx, by, bz] = Int32(i)
    end

    # --- Allocation ---
    rho = ones(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    vel = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 3)
    vel_temp = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 3)
    
    # Temporal interpolation
    if (enable_temporal_interpolation || enable_macro_temporal_interpolation) && n_blocks > 0
        rho_old = ones(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
        vel_old = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 3)
        # Pass176: `_older` arrays for quadratic temporal interp. Initialized
        # to same as `_old` so the first 2 parent steps fall back to linear
        # blend (the quadratic correction term is proportional to the second
        # time derivative of the parent state, which is small at startup).
        rho_older = ones(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
        vel_older = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 3)
        # B1 P3 (2026-05-28) — cubic Hermite temporal interpolation anchors.
        # Same allocation pattern as `_older`, initialized to `_old` value so
        # the first 3 parent steps fall back to quadratic blend (the cubic
        # correction term is proportional to ∂³/∂t³ of the parent state).
        if INTERFACE_CUBIC_TEMPORAL
            rho_oldest = ones(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
            vel_oldest = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 3)
        else
            rho_oldest = ones(LBM_F, 1, 1, 1, 1)
            vel_oldest = zeros(LBM_F, 1, 1, 1, 1, 3)
        end
        # B1 P3 RK4-equivalent quintic Lagrange 5th anchor (t=-3·dt_c). Allocated
        # full-size only when BOTH the quintic flag AND the cubic flag are on (the
        # quintic shares the cubic anchor chain). The copy_to_old! rotation
        # self-activates once these are full-size (it is guarded on length>1).
        # Initialized to `_old` value so the first 4 parent steps fall back to
        # cubic → quadratic → linear.
        if INTERFACE_QUINTIC_TEMPORAL && INTERFACE_CUBIC_TEMPORAL
            rho_oldest2 = ones(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
            vel_oldest2 = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 3)
        else
            rho_oldest2 = ones(LBM_F, 1, 1, 1, 1)
            vel_oldest2 = zeros(LBM_F, 1, 1, 1, 1, 3)
        end
    else
        rho_old = ones(LBM_F, 1, 1, 1, 1)
        vel_old = zeros(LBM_F, 1, 1, 1, 1, 3)
        rho_older = ones(LBM_F, 1, 1, 1, 1)
        vel_older = zeros(LBM_F, 1, 1, 1, 1, 3)
        rho_oldest = ones(LBM_F, 1, 1, 1, 1)
        vel_oldest = zeros(LBM_F, 1, 1, 1, 1, 3)
        rho_oldest2 = ones(LBM_F, 1, 1, 1, 1)
        vel_oldest2 = zeros(LBM_F, 1, 1, 1, 1, 3)
    end
    
    # Distributions — N_DIRS directions (27 for D3Q27, 39 for D3Q39)
    n_dirs = Int(N_DIRS)
    f = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, n_dirs)
    f_temp = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, n_dirs)

    if n_boundary_cells > 0
        f_post_collision = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, n_dirs)
    else
        f_post_collision = zeros(LBM_F, 1, 1, 1, 1, n_dirs)
    end

    if enable_temporal_interpolation && n_blocks > 0
        f_old = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, n_dirs)
    else
        f_old = zeros(LBM_F, 1, 1, 1, 1, n_dirs)
    end

    # Thermal DDF arrays (Phase 2/4b Frapolli)
    # N_DIRS_THERMAL = 39 for D3Q39 flow (Frapolli model), 27 for D3Q27 flow
    n_dirs_g = Int(N_DIRS_THERMAL)
    tau_g = 0.0f0
    if DDF_ENABLED && n_blocks > 0
        g = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, n_dirs_g)
        g_temp = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, n_dirs_g)
        temperature = fill(LBM_F(DDF_T_INITIAL), BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
        temperature_old = fill(LBM_F(DDF_T_INITIAL), BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
        temperature_older = fill(LBM_F(DDF_T_INITIAL), BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)  # Pass176: quadratic temporal anchor
        if INTERFACE_CUBIC_TEMPORAL
            temperature_oldest = fill(LBM_F(DDF_T_INITIAL), BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)  # B1 P3: cubic Hermite temporal anchor
        else
            temperature_oldest = ones(LBM_F, 1, 1, 1, 1)
        end
        # B1 P3 RK4 quintic temperature anchor (t=-3·dt_c). Full-size only when
        # both quintic + cubic flags are on; otherwise 1×1×1×1 stub.
        if INTERFACE_QUINTIC_TEMPORAL && INTERFACE_CUBIC_TEMPORAL
            temperature_oldest2 = fill(LBM_F(DDF_T_INITIAL), BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
        else
            temperature_oldest2 = ones(LBM_F, 1, 1, 1, 1)
        end
        kinetic_energy = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
        if THERMAL_FILTER_ENABLED
            temperature_filter_scratch = fill(LBM_F(DDF_T_INITIAL), BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
        else
            temperature_filter_scratch = ones(LBM_F, 1, 1, 1, 1)
        end
        if n_boundary_cells > 0
            g_post_collision = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, n_dirs_g)
        else
            g_post_collision = zeros(LBM_F, 1, 1, 1, 1, n_dirs_g)
        end
        if enable_temporal_interpolation
            g_old = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, n_dirs_g)
        else
            g_old = zeros(LBM_F, 1, 1, 1, 1, n_dirs_g)
        end
    else
        g = zeros(LBM_F, 1, 1, 1, 1, n_dirs_g)
        g_temp = zeros(LBM_F, 1, 1, 1, 1, n_dirs_g)
        g_post_collision = zeros(LBM_F, 1, 1, 1, 1, n_dirs_g)
        g_old = zeros(LBM_F, 1, 1, 1, 1, n_dirs_g)
        temperature = ones(LBM_F, 1, 1, 1, 1)
        temperature_old = ones(LBM_F, 1, 1, 1, 1)
        temperature_older = ones(LBM_F, 1, 1, 1, 1)  # Pass176: stub when DDF disabled
        temperature_oldest = ones(LBM_F, 1, 1, 1, 1)  # B1 P3: stub when DDF disabled
        temperature_oldest2 = ones(LBM_F, 1, 1, 1, 1) # B1 P3 RK4: stub when DDF disabled
        kinetic_energy = zeros(LBM_F, 1, 1, 1, 1)
        temperature_filter_scratch = ones(LBM_F, 1, 1, 1, 1)
    end

    # Π_neq persistent field. Allocated when:
    #   - variable ν_sgs is active (per-cell adaptive eddy viscosity), OR
    #   - Π_neq is used as the coarse↔fine stress carrier (STRESS_TENSOR_TRANSFER,
    #     default true for entropic). Prolongation reads parent Π_neq and projects
    #     it onto the lattice directions to reconstruct fine populations without
    #     interpolating raw f_k — significantly more stable at high Mach.
    _alloc_pi_neq = (VARIABLE_NU_SGS || STRESS_TENSOR_TRANSFER) && n_blocks > 0
    if _alloc_pi_neq
        pi_neq = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 6)
        pi_neq_scratch = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 6)
    else
        pi_neq = zeros(LBM_F, 1, 1, 1, 1, 6)
        pi_neq_scratch = zeros(LBM_F, 1, 1, 1, 1, 6)
    end

    # C2/C4 quintic B-spline prolongation coefficient arrays (2026-06-04).
    # Full-size only when the flag is on; else 1×1×1×1 stubs. The flag global is
    # QUINTIC_BSPLINE_PROLONGATION (config_loader). Use a safe isdefined guard so
    # blocks.jl loads even before config sets it.
    # has_children gate: the finest level is never a prolongation source, so it
    # never needs spline coeffs. Skip its (potentially largest) spline arrays.
    _alloc_spline = (@isdefined(QUINTIC_BSPLINE_PROLONGATION) && QUINTIC_BSPLINE_PROLONGATION) && n_blocks > 0 && has_children
    if _alloc_spline
        rho_spline_coeff   = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
        rho_spline_scratch = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
        vel_spline_coeff   = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 3)
        vel_spline_scratch = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 3)
    else
        rho_spline_coeff   = zeros(LBM_F, 1, 1, 1, 1)
        rho_spline_scratch = zeros(LBM_F, 1, 1, 1, 1)
        vel_spline_coeff   = zeros(LBM_F, 1, 1, 1, 1, 3)
        vel_spline_scratch = zeros(LBM_F, 1, 1, 1, 1, 3)
    end

    # Lever 1b (2026-06-20) — once-per-coarse-step prolonged-halo macro+Π cache.
    # Full-size only when the hoist is active (env LUDWIG_D3Q19_HALO_HOIST) and the
    # level has blocks; else 1×1×1×1×1 stubs. Indexed by THIS level's fine cells, so
    # it is meaningful on refined levels (a parent fills it); L1 allocates the stub
    # via the same gate but never fills/reads it (it has no parent). Use an env read
    # with a default-OFF so blocks.jl loads regardless of config-global init order.
    # GHOST-PADDED (BLOCK_SIZE+2)³ (2026-06-20): the consumer reads the cache at the
    # streaming SOURCE cell (cell − c); for D3Q19 (|c|≤1) the source can be one fine
    # cell OUTSIDE the block on each face, so the cache carries a 1-cell ghost ring.
    # Local padded index lp ∈ 1..BLOCK_SIZE+2 maps to source local s = lp−1 ∈
    # 0..BLOCK_SIZE+1 (s=0 ghost-low, s=BLOCK_SIZE+1 ghost-high). The +2 (not +4) is
    # sufficient ONLY because D3Q19 velocities are |c|≤1 on every axis.
    _alloc_halo_hoist = (get(ENV, "LUDWIG_D3Q19_HALO_HOIST", "0") == "1") && n_blocks > 0
    _hp = BLOCK_SIZE + 2
    if _alloc_halo_hoist
        halo_macro_new = zeros(LBM_F, _hp, _hp, _hp, n_blocks, 4)
        halo_macro_old = zeros(LBM_F, _hp, _hp, _hp, n_blocks, 4)
        halo_pi_new    = zeros(LBM_F, _hp, _hp, _hp, n_blocks, 6)
        halo_pi_old    = zeros(LBM_F, _hp, _hp, _hp, n_blocks, 6)
    else
        halo_macro_new = zeros(LBM_F, 1, 1, 1, 1, 4)
        halo_macro_old = zeros(LBM_F, 1, 1, 1, 1, 4)
        halo_pi_new    = zeros(LBM_F, 1, 1, 1, 1, 6)
        halo_pi_old    = zeros(LBM_F, 1, 1, 1, 1, 6)
    end

    # B4 (2026-05-28) — Newton warm-start cache for the 13-mom solver.
    # Initialized to NaN; the solver detects this sentinel and falls back to
    # polynomial init on the very first step. After Newton runs, the kernel
    # writes the converged λ back; subsequent steps read it as warm-start.
    #
    # PERF (2026-06-04, profiling workflow): this 13-component array is consumed
    # ONLY by the D3Q39 entropic Newton kernel. On the D3Q19 / D3Q27 paths it is
    # the unused stub param `_lambda_cache_arr` (never dereferenced). Allocating
    # it full-size on every level regardless of lattice cost ~528 MiB on the
    # Z132 A250 D3Q19 case (6.4% of 8 GB) for nothing. Gate on the D3Q39 numerical
    # equilibrium so non-D3Q39 runs get the 1×1×1×1×13 stub. Byte-identical:
    # the D3Q39 path keeps the full array (gate true); other paths never read it.
    _alloc_lambda = (@isdefined(NUMERICAL_EQUILIBRIUM) && NUMERICAL_EQUILIBRIUM) && n_blocks > 0
    if _alloc_lambda
        lambda_cache = fill(LBM_F(NaN), BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 13)
    else
        lambda_cache = fill(LBM_F(NaN), 1, 1, 1, 1, 13)
    end

    # Geometry arrays
    wall_dist = fill(LBM_F(100.0), BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    obstacle = zeros(Bool, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    sponge = zeros(LBM_F, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    amr_covered = n_blocks > 0 ? fill(false, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks) :
                                 fill(false, 1, 1, 1, 1)
    # AMR feather band — populated post-allocation by compute_feather_weights!
    # when FEATHER_DEPTH > 0. Allocated even when disabled so the kernel
    # argument list is uniform; a zero field means no extra dissipation.
    feather_weight = zeros(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, max(n_blocks, 1))
    # (D3Q19 AMR halo-pop scratch retired 2026-06-24 — only the D3Q19 staged-halo path
    #  used it; now always a stub.)
    amr_halo_pop_scratch = zeros(LBM_F, 1, 1, 1, 1, 20)
    amr_face_budget_components = AMR_POPULATION_STREAM_REGISTER ?
        (AMR_POPULATION_STREAM_REGISTER_FACE_RESOLVED ?
            12 + 3 * 39 + 6 * 3 * 39 +
                (AMR_POPULATION_STREAM_REGISTER_XFACE_GLOBAL_APPLY ? 10 : 0) :
            12 + 3 * 39) :
        12
    _alloc_amr_face_budget =
        (AMR_GPU_FACE_BUDGET_REFLUX ||
         AMR_SHELL_PAIRED_REFLUX_BUDGET_MEAN_PRESERVING ||
         AMR_POPULATION_STREAM_REGISTER) &&
        n_blocks > 0
    amr_face_budget = _alloc_amr_face_budget ?
        zeros(Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, amr_face_budget_components) :
        zeros(Float32, 1, 1, 1, 1, amr_face_budget_components)
    block_shift = zeros(Int32, max(n_blocks, 1))
    block_shift_next = zeros(Int32, max(n_blocks, 1))
    local_shift_wall_guard = zeros(Int32, max(n_blocks, 1))
    
    map_x = Int32[c[1] for c in active_coords]
    map_y = Int32[c[2] for c in active_coords]
    map_z = Int32[c[3] for c in active_coords]
    
    bouzidi_enabled = (n_boundary_cells > 0) && (bouzidi_q_map !== nothing)
    
    if bouzidi_enabled
        q_map_final = convert(Array{Float16, 5}, bouzidi_q_map)
        c_block_final = convert(Vector{Int32}, bouzidi_cell_block)
        c_x_final = convert(Vector{Int8}, bouzidi_cell_x)
        c_y_final = convert(Vector{Int8}, bouzidi_cell_y)
        c_z_final = convert(Vector{Int8}, bouzidi_cell_z)
        # Convert tri map to int32 
        tri_map_final = convert(Array{Int32, 5}, bouzidi_tri_map)
        @inbounds for ci in 1:min(Int(n_boundary_cells), length(c_block_final))
            b = Int(c_block_final[ci])
            if 1 <= b <= n_blocks
                local_shift_wall_guard[b] = Int32(1)
            end
        end
    else
        q_map_final = zeros(Float16, 1, 1, 1, 1, n_dirs)
        c_block_final = Int32[]
        c_x_final = Int8[]
        c_y_final = Int8[]
        c_z_final = Int8[]
        tri_map_final = zeros(Int32, 1, 1, 1, 1, n_dirs)
    end

    return BlockLevel(
        level_id, Float64(dx), Float32(dt), Float32(tau),
        bx_max, by_max, bz_max,
        block_pointer, active_coords,
        rho, vel, vel_temp,
        rho_oldest2, vel_oldest2, # B1 P3 RK4 (2026-05-28): quintic Lagrange anchors (4-step-old)
        rho_oldest, vel_oldest,  # B1 P3 (2026-05-28): cubic Hermite temporal anchors (3-step-old)
        rho_older, vel_older,    # Pass176: quadratic temporal anchors (2-step-old)
        rho_old, vel_old,
        f, f_temp, f_post_collision, f_old,
        tau_g, g, g_temp, g_post_collision, g_old,
        temperature, temperature_old, temperature_older,   # Pass176: quadratic temporal anchor for T
        temperature_oldest,      # B1 P3: cubic Hermite temporal anchor for T
        temperature_oldest2,     # B1 P3 RK4: quintic Lagrange temporal anchor for T
        kinetic_energy,
        temperature_filter_scratch,
        pi_neq,
        pi_neq_scratch,
        rho_spline_coeff, rho_spline_scratch, vel_spline_coeff, vel_spline_scratch,  # C2/C4 B-spline
        halo_macro_new, halo_macro_old, halo_pi_new, halo_pi_old,  # lever 1b hoist cache
        lambda_cache,           # B4 (2026-05-28) — Newton warm-start cache
        wall_dist, obstacle, sponge,
        amr_covered,
        feather_weight,
        amr_halo_pop_scratch,
        amr_face_budget,
        block_shift,
        block_shift_next,
        local_shift_wall_guard,
        neighbor_table, map_x, map_y, map_z,
        bouzidi_enabled,
        q_map_final, c_block_final, c_x_final, c_y_final, c_z_final, tri_map_final,
        n_boundary_cells
    )
end

mutable struct SolverMesh
    levels::Vector{BlockLevel}
    domain_dims::Tuple{Int, Int, Int}
end

"""
Copy current state to old arrays for temporal interpolation.
Call BEFORE advancing the coarse level.
Optionally pass `g_current` for thermal DDF temporal storage.
"""
function copy_to_old!(level::BlockLevel, f_current, vel_current; g_current=nothing)
    # OpTiming (2026-06-12): the per-substep temporal-anchor rotation —
    # including the FULL 19-direction f_old copy — was invisible to
    # PhaseTiming; wire the dead _T_OP_COPYOLD bucket (PHASE_TIMING only).
    _tco = 0.0
    if _PHASE_TIMING
        KernelAbstractions.synchronize(KernelAbstractions.get_backend(level.rho)); _tco = time()
    end
    # Pass176 (Tier A item A2 - quadratic temporal interpolation):
    # Rotate the snapshot chain BEFORE updating `_old`:
    #   `_older` (2 steps ago) <- `_old` (1 step ago)
    #   `_old`   (1 step ago)  <- `current`
    # Combined with the current arrays = `_new` (this step's end state, after
    # collision), this gives 3 anchor points {older, old, new} at times
    # {-dt_c, 0, +dt_c} for quadratic Lagrange temporal interpolation at fine
    # sub-cycle times in the AMR halo prolongation. The quadratic correction
    # term is proportional to the second time derivative of the parent state,
    # which is zero for steady flow and small for slowly-evolving fields.
    if length(level.rho_old) > 1
        # B1 P3 (2026-05-28) — extended rotation chain for cubic Hermite:
        #   `_oldest2` (4 steps ago) <- `_oldest` (3 steps ago)  [B1 P3 RK4 — happens FIRST]
        #   `_oldest`  (3 steps ago) <- `_older`  (2 steps ago)
        #   `_older`   (2 steps ago) <- `_old`    (1 step ago)
        #   `_old`     (1 step ago)  <- current
        # Each rotation happens only if its destination array is live (non-stub).
        # B1 P3 RK4 (2026-05-28) — quintic Lagrange rotation: `_oldest2` <- `_oldest`.
        if length(level.rho_oldest2) > 1 && length(level.rho_oldest) > 1
            copyto!(level.rho_oldest2, level.rho_oldest)
            copyto!(level.vel_oldest2, level.vel_oldest)
        end
        if length(level.rho_oldest) > 1 && length(level.rho_older) > 1
            copyto!(level.rho_oldest, level.rho_older)
            copyto!(level.vel_oldest, level.vel_older)
        end
        # Rotation: `_older` <- `_old` (must happen before `_old` is overwritten)
        if length(level.rho_older) > 1
            copyto!(level.rho_older, level.rho_old)
            copyto!(level.vel_older, level.vel_old)
        end
        copyto!(level.rho_old, level.rho)
        copyto!(level.vel_old, vel_current)
    end
    if length(level.f_old) > N_DIRS
        copyto!(level.f_old, f_current)
    end
    # Thermal DDF temporal storage (N_DIRS_THERMAL directions: 27 for D3Q27, 39 for Frapolli D3Q39)
    if g_current !== nothing && has_thermal(level)
        if length(level.g_old) > N_DIRS_THERMAL
            copyto!(level.g_old, g_current)
        end
        # Same rotation for the thermal field.
        # B1 P3 RK4 extended chain: oldest2 <- oldest <- older <- old <- current
        if length(level.temperature_oldest2) > 1 && length(level.temperature_oldest) > 1
            copyto!(level.temperature_oldest2, level.temperature_oldest)
        end
        if length(level.temperature_oldest) > 1 && length(level.temperature_older) > 1
            copyto!(level.temperature_oldest, level.temperature_older)
        end
        if length(level.temperature_older) > 1
            copyto!(level.temperature_older, level.temperature_old)
        end
        copyto!(level.temperature_old, level.temperature)
    end
    if _PHASE_TIMING
        KernelAbstractions.synchronize(KernelAbstractions.get_backend(level.rho))
        _T_OP_COPYOLD[] += time() - _tco
    end
end

"""Check if level has temporal interpolation storage allocated."""
has_temporal_storage(level::BlockLevel) = length(level.f_old) > N_DIRS

"""Check if level has macro-only temporal storage allocated (rho_old/vel_old)."""
has_macro_temporal_storage(level::BlockLevel) = length(level.rho_old) > 1

"""Check if level has thermal DDF arrays allocated (not stubs). Uses N_DIRS_THERMAL (27 or 39)."""
has_thermal(level::BlockLevel) = length(level.g) > N_DIRS_THERMAL

"""Check if level has the persistent Π_neq field allocated (not a 1×1×1×1×6 stub)."""
has_pi_neq(level::BlockLevel) = length(level.pi_neq) > 6
# C2/C4 B-spline: full-size when allocated (>1 cell), stub (=1) otherwise.
has_spline_coeff(level::BlockLevel) = length(level.rho_spline_coeff) > 1
# Lever 1b hoist cache: full-size (4 comps × >1 cell) when allocated, stub otherwise.

"""Check if level has the thermal T-filter scratch buffer allocated (not a 1×1×1×1 stub)."""
has_thermal_filter(level::BlockLevel) = length(level.temperature_filter_scratch) > 1

"""
Mark coarse cells that are topologically covered by the next finer level.

Each coarse cell belongs to one of the 2×2×2 child octants of its parent block.
If that child block exists on the finer level, the coarse cell is fully covered
and should not evolve as an independent coarse fluid cell during the coarse step.

The implementation below deliberately uses the stricter restriction criterion:
all 8 fine cells corresponding to the coarse cell must be backed by active
child blocks. That keeps AMR ownership and fine-to-parent restriction aligned
at corners and partial frontiers.
"""
function compute_amr_covered_mask!(parent_level::BlockLevel, child_level::BlockLevel)
    length(parent_level.amr_covered) > 1 && fill!(parent_level.amr_covered, false)

    n_parent_blocks = length(parent_level.active_block_coords)
    n_parent_blocks == 0 && return 0

    child_ptr = child_level.block_pointer
    child_dim_x, child_dim_y, child_dim_z = size(child_ptr)
    covered_cells = 0

    @inline function child_cell_active(fgx::Int, fgy::Int, fgz::Int)
        fgx < 1 && return false
        fgy < 1 && return false
        fgz < 1 && return false
        cbx = div(fgx - 1, BLOCK_SIZE) + 1
        cby = div(fgy - 1, BLOCK_SIZE) + 1
        cbz = div(fgz - 1, BLOCK_SIZE) + 1
        if cbx < 1 || cbx > child_dim_x ||
           cby < 1 || cby > child_dim_y ||
           cbz < 1 || cbz > child_dim_z
            return false
        end
        return child_ptr[cbx, cby, cbz] != Int32(0)
    end

    @inline function coarse_cell_has_full_child_support(gx::Int, gy::Int, gz::Int)
        @inbounds for sz in 0:1, sy in 0:1, sx in 0:1
            fgx = 2 * gx - 1 + sx
            fgy = 2 * gy - 1 + sy
            fgz = 2 * gz - 1 + sz
            child_cell_active(fgx, fgy, fgz) || return false
        end
        return true
    end

    @inbounds for (pb_idx, (pbx, pby, pbz)) in enumerate(parent_level.active_block_coords)
        for z in 1:BLOCK_SIZE
            gz = (pbz - 1) * BLOCK_SIZE + z
            for y in 1:BLOCK_SIZE
                gy = (pby - 1) * BLOCK_SIZE + y
                for x in 1:BLOCK_SIZE
                    gx = (pbx - 1) * BLOCK_SIZE + x
                    covered = coarse_cell_has_full_child_support(gx, gy, gz)
                    parent_level.amr_covered[x, y, z, pb_idx] = covered
                    covered_cells += covered ? 1 : 0
                end
            end
        end
    end

    return covered_cells
end

"""Populate the covered-by-fine mask for the whole AMR hierarchy."""
function compute_all_amr_covered_masks!(grids::Vector{BlockLevel})
    n_levels = length(grids)
    counts = zeros(Int, n_levels)

    for lvl in 1:n_levels
        length(grids[lvl].amr_covered) > 1 && fill!(grids[lvl].amr_covered, false)
    end

    for lvl in 1:(n_levels - 1)
        counts[lvl] = compute_amr_covered_mask!(grids[lvl], grids[lvl + 1])
    end

    return counts
end
