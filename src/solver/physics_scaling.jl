using Printf
include(joinpath(@__DIR__,"grid_quantization.jl"))


# Note: CS2_PHYSICS and CS4_PHYSICS are defined in dependencies.jl as mutable globals
# They are set at startup based on LATTICE_TYPE (D3Q27: 1/3, D3Q39: 2/3)

mutable struct DomainParameters
    initialized::Bool
    num_levels::Int
    mesh_min::Tuple{Float64, Float64, Float64}
    mesh_max::Tuple{Float64, Float64, Float64}
    mesh_center::Tuple{Float64, Float64, Float64}
    mesh_extent::Tuple{Float64, Float64, Float64}
    reference_length::Float64
    reference_chord::Float64
    reference_area::Float64
    moment_center::Tuple{Float64, Float64, Float64}
    domain_min::Tuple{Float64, Float64, Float64}
    domain_max::Tuple{Float64, Float64, Float64}
    domain_size::Tuple{Float64, Float64, Float64}
    mesh_offset::Tuple{Float64, Float64, Float64}
    dx_fine::Float64
    dx_coarse::Float64
    dx_levels::Vector{Float64}
    nx_coarse::Int
    ny_coarse::Int
    nz_coarse::Int
    bx_max::Int
    by_max::Int
    bz_max::Int
    l_char::Float64
    nu_lattice::Float64
    tau_coarse::Float32
    tau_levels::Vector{Float32}
    cs2::Float32
    cs4::Float32
    re_number::Float64
    u_physical::Float64
    rho_physical::Float64
    nu_physical::Float64
    length_scale::Float64
    time_scale::Float64
    velocity_scale::Float64
    force_scale::Float64
    tau_fine::Float64
    tau_margin_percent::Float64
    wall_model_active::Bool
    y_plus_first_cell::Float64
    estimated_memory_gb::Float64
    # Thermal DDF (Phase 2)
    tau_g_fine::Float64
    tau_g_levels::Vector{Float32}
end

global DOMAIN_PARAMS = DomainParameters(
    false, 0, (0.0,0.0,0.0), (0.0,0.0,0.0), (0.0,0.0,0.0), (0.0,0.0,0.0),
    0.0, 0.0, 0.0, (0.0,0.0,0.0), (0.0,0.0,0.0), (0.0,0.0,0.0), (0.0,0.0,0.0), (0.0,0.0,0.0),
    0.0, 0.0, Float64[], 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0f0, Float32[], 1f0/3f0, 1f0/9f0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false, 0.0, 0.0,
    0.0, Float32[]
)

function compute_tau_for_levels(Re::Float64, ref_length::Float64, resolution::Int, n_levels::Int, u_lattice::Float32)
    nu_lattice_fine = Float64(u_lattice) * resolution / Re
    return nu_lattice_fine / Float64(CS2_PHYSICS) + 0.5
end

function compute_max_levels_for_domain(domain_size::Float64, dx_fine::Float64, block_size::Int, min_blocks::Int)
    # Two caps:
    #   hard_cap: the coarsest level fits ≥ 1 block per axis (= block_size cells).
    #             Below this streaming is ill-defined.
    #   soft_cap: the coarsest level fits ≥ min_blocks blocks per axis. This is
    #             the "healthy pyramid" rule — the outer mesh needs a few coarse
    #             cells to absorb the wake, run sponges, and let the interpolation
    #             stencil (cubic Lagrange, 4 cells wide) sit entirely inside the
    #             parent grid. An L1 that is 5×2×3 blocks, as with the ONERAM6
    #             case that produced rectangular density patches at every
    #             refinement-level transition, has no such buffer.
    #
    # We return `soft_cap` by default: num_levels is silently clamped to the
    # healthy pyramid limit. Users who really need deeper refinement in a tight
    # domain should lower `advanced.high_re.min_coarse_blocks` explicitly, which
    # is the unambiguous opt-in.
    hard_ratio = _roundoff_snap_grid_integer(domain_size / (dx_fine * block_size))
    soft_ratio = _roundoff_snap_grid_integer(domain_size / (dx_fine * min_blocks * block_size))

    hard_cap = hard_ratio < 1.0 ? 1 : Int(floor(1 + log2(hard_ratio)))
    soft_cap = soft_ratio < 1.0 ? 1 : Int(floor(1 + log2(soft_ratio)))

    if NUM_LEVELS_CONFIG > soft_cap
        @info "[Levels] Clamping num_levels $(NUM_LEVELS_CONFIG) → $(soft_cap) to satisfy min_coarse_blocks=$(min_blocks) (hard cap $(hard_cap)). Lower min_coarse_blocks to recover deeper refinement."
    end

    return soft_cap
end

"""
    periodic_y_max_levels(domain_y, dx_fine, block_size) -> Int

Periodic-Y AMR (lbm-app-int, 27 Oct 2026): the most levels the Y period allows (one coarsest block, no
healthy-pyramid rule on the periodic axis). This cap MUST agree with the period refusal in
`compute_domain_from_mesh`, which snaps the period up to whole coarsest blocks and accepts a requested span that
lies within ONE finest cell below such a period. The cap is therefore read on the largest span that refusal still
accepts, `domain_y + 0.999 dx_fine` (0.999 keeps it strictly inside the one-cell tolerance). Read on the raw span
(the code before this fix), a span such as the app's `(span_cells - 0.5)/N` lost a level whenever it was exactly
one coarsest block. The adjustment raises the cap by at most one level, and only for a span within 0.999 dx_fine
below a whole coarsest block of the raised level count, which the refusal then accepts; the non-periodic cap is
not affected.
"""
function periodic_y_max_levels(domain_y::Float64, dx_fine::Float64, block_size::Int)
    return compute_max_levels_for_domain(domain_y + 0.999 * dx_fine, dx_fine, block_size, 1)
end

function print_re_analysis(Re::Float64, ref_length::Float64, resolution::Int, u_lattice::Float32;
                           tau_override::Union{Nothing,Float64}=nothing,
                           scaling_note::String="")
    tau = tau_override === nothing ? compute_tau_for_levels(Re, ref_length, resolution, 1, u_lattice) :
                                     tau_override
    println("\n[High-Re] ═══════════════════════════════════════════════════════")
    @printf("[High-Re] Re = %.2e, L = %.4f m, N = %d cells/L, U_lat = %.3f\n", Re, ref_length, resolution, u_lattice)
    !isempty(scaling_note) && println("[High-Re] Scaling note: $scaling_note")
    @printf("[High-Re] Computed τ_fine = %.6f\n", tau)
    @printf("[High-Re] Effective τ_min (with ν_sgs_bg) = %.6f\n", max(tau, 0.5 + Float64(NU_SGS_BACKGROUND)/Float64(CS2_PHYSICS)))
    if tau < 0.501
        if NU_SGS_BACKGROUND > 0.0f0
            println("[High-Re] ⚠ τ ≈ 0.5 (stability limit) - background ν_sgs active")
        else
            println("[High-Re] ⚠ τ ≈ 0.5 (stability limit) - relying on tau_min / entropic stability")
        end
    end
    println("[High-Re] ═══════════════════════════════════════════════════════\n")
end

# ────────────────────────────────────────────────────────────────────
# estimate_bytes_per_cell — shared between pre-run sizer and post-run
# diagnostics so the two can never drift. Derives the actual per-cell
# byte footprint from the globals set by config_loader.jl. Assumes
# Float32 distributions / macros, Bool obstacle, Int32 connectivity.
# ────────────────────────────────────────────────────────────────────
function estimate_bytes_per_cell()
    # Flow
    n_dirs_flow = Int(N_DIRS)
    bytes = 0
    bytes += n_dirs_flow * 4 * 2            # f, f_temp
    bytes += n_dirs_flow * 4                # f_post_collision (always allocated on bouzidi levels; a safe overcount on pure-interior levels)
    if TEMPORAL_INTERPOLATION
        bytes += n_dirs_flow * 4            # f_old
    end
    bytes += 4                              # rho
    bytes += 12                             # vel (3·Float32)
    bytes += 12                             # vel_temp
    if TEMPORAL_INTERPOLATION
        bytes += 4 + 12                     # rho_old + vel_old
    end
    if STRESS_TENSOR_TRANSFER || VARIABLE_NU_SGS
        bytes += 24 * 2                     # pi_neq + pi_neq_scratch
    end
    # Geometry per cell
    bytes += 1                              # obstacle (Bool)
    bytes += 4                              # sponge
    bytes += 4                              # wall_dist
    # Connectivity per-cell share is ~O(12 B) amortised across BLOCK_SIZE³ cells
    bytes += 2
    # Bouzidi: near-wall only; assume 5 % cell share ≈ 40 B/cell weighted
    if BOUNDARY_METHOD == :bouzidi
        bytes += 5
    end
    # Thermal DDF
    if DDF_ENABLED
        n_dirs_g = Int(N_DIRS_THERMAL)
        bytes += n_dirs_g * 4 * 2           # g, g_temp
        bytes += n_dirs_g * 4               # g_post_collision
        if TEMPORAL_INTERPOLATION
            bytes += n_dirs_g * 4           # g_old
        end
        bytes += 4 + 4 + 4                  # temperature, temperature_old, kinetic_energy
        if THERMAL_FILTER_ENABLED
            bytes += 4                      # temperature_filter_scratch
        end
    end
    return bytes
end

# ────────────────────────────────────────────────────────────────────
# estimate_active_blocks_per_level — analytic projection of the block
# count at each level, used by the VRAM-fit loop BEFORE the expensive
# SAT voxelizer runs. Accurate to roughly ±25 % for typical aircraft /
# ground-vehicle geometries. Monotone in domain extents, so the fit
# loop converges cleanly.
#
# Level 1: always fills the wind tunnel → bx_coarse · by_coarse · bz_coarse.
# Level L > 1: body-AABB inflated by body_halo_chords, plus a wake box
# whose extent decays with depth (reuses the Fix 2 geometric growth so
# the estimate matches the actual builder's output).
# ────────────────────────────────────────────────────────────────────
function estimate_active_blocks_per_level(
    num_levels::Int, dx_coarse::Float64, block_size::Int,
    nx_coarse::Int, ny_coarse::Int, nz_coarse::Int,
    mesh_min, mesh_max,
    body_halo_chords::Float64,
    wake_end_base_x::Float64, wake_start_base_x::Float64,
)
    L_stream = mesh_max[1] - mesh_min[1]
    L_stream <= 0.0 && (L_stream = dx_coarse)   # safety
    counts = Int[]
    for lvl in 1:num_levels
        scale = 2^(lvl - 1)
        dx_lvl = dx_coarse / scale
        bs_phys = block_size * dx_lvl
        if lvl == 1
            push!(counts, nx_coarse * ny_coarse * nz_coarse ÷ block_size^3)
        else
            # Body AABB inflated by halo in chord units
            halo = body_halo_chords * L_stream
            bx = max(1.0, (mesh_max[1] - mesh_min[1] + 2halo) / bs_phys)
            by = max(1.0, (mesh_max[2] - mesh_min[2] + 2halo) / bs_phys)
            bz = max(1.0, (mesh_max[3] - mesh_min[3] + 2halo) / bs_phys)
            body_blocks = ceil(Int, bx) * ceil(Int, by) * ceil(Int, bz)

            # Wake box — geometric growth mirroring geometry/domain.jl Fix 2
            depth = num_levels - lvl
            growth = 2.0^depth - 1.0
            pad_down = 0.5 * L_stream * growth
            pad_lat  = 0.3 * L_stream * growth
            wake_len_x = max(0.0, (wake_end_base_x - wake_start_base_x) + pad_down)
            wake_y = max(0.0, 2 * pad_lat + 0.2 * L_stream)
            wake_z = max(0.0, 2 * pad_lat + 0.2 * L_stream)
            wake_blocks = ceil(Int, wake_len_x / bs_phys) *
                          ceil(Int, wake_y / bs_phys) *
                          ceil(Int, wake_z / bs_phys)

            # Assume ~30 % overlap between body halo and wake box
            push!(counts, round(Int, 0.7 * (body_blocks + wake_blocks)))
        end
    end
    return counts
end

# ────────────────────────────────────────────────────────────────────
# compute_domain_auto_extents — the "min_dx + num_levels + VRAM" sizer.
# Produces (num_levels, nx_coarse, ny_coarse, nz_coarse, dx_fine,
# domain_x, domain_y, domain_z, SR_equivalent) and writes them into the
# legacy globals so the rest of compute_domain_from_mesh stays
# unchanged.
#
# Uses a simple monotone fit: seeds extents from chord-relative defaults,
# estimates VRAM, shrinks downstream→lateral→upstream proportionally if
# over budget, and grows downstream (up to its cap) if well under budget.
# Converges in ≤ 5 iterations for all realistic cases.
# ────────────────────────────────────────────────────────────────────
function compute_domain_auto_extents(mesh_min, mesh_max)
    MESH_MIN_DX > 0.0 || return nothing
    bs = BLOCK_SIZE_CONFIG
    L_stream = mesh_max[1] - mesh_min[1]
    L_span_y = mesh_max[2] - mesh_min[2]
    L_span_z = mesh_max[3] - mesh_min[3]
    L_chord  = REFERENCE_CHORD_CONFIG > 0 ? REFERENCE_CHORD_CONFIG : L_stream

    # Seed extents in chord units — prefer NEW chord-relative knobs,
    # fall back to legacy absolute knobs scaled by reference_length.
    chord_up_0   = DOMAIN_UPSTREAM_CHORDS   > 0 ? DOMAIN_UPSTREAM_CHORDS   : 3.0
    chord_down_0 = DOMAIN_DOWNSTREAM_CHORDS > 0 ? DOMAIN_DOWNSTREAM_CHORDS : 10.0
    chord_lat_0  = DOMAIN_LATERAL_CHORDS    > 0 ? DOMAIN_LATERAL_CHORDS    : 4.0
    chord_ht_0   = DOMAIN_HEIGHT_CHORDS     > 0 ? DOMAIN_HEIGHT_CHORDS     : 4.0

    # The legacy absolute domain.{upstream,...} keys now act as caps.
    cap_up   = DOMAIN_UPSTREAM   > 0 ? DOMAIN_UPSTREAM   * max(REFERENCE_LENGTH_FOR_MESHING, L_stream) : Inf
    cap_down = DOMAIN_DOWNSTREAM > 0 ? DOMAIN_DOWNSTREAM * max(REFERENCE_LENGTH_FOR_MESHING, L_stream) : Inf
    cap_lat  = DOMAIN_LATERAL    > 0 ? DOMAIN_LATERAL    * max(REFERENCE_LENGTH_FOR_MESHING, L_stream) : Inf
    cap_ht   = DOMAIN_HEIGHT     > 0 ? DOMAIN_HEIGHT     * max(REFERENCE_LENGTH_FOR_MESHING, L_stream) : Inf

    chord_up   = chord_up_0
    chord_down = chord_down_0
    chord_lat  = chord_lat_0
    chord_ht   = chord_ht_0

    # VRAM budget — if not provided and CUDA is live, take 90 % of free
    budget_gb = MESH_VRAM_BUDGET_GB
    if budget_gb <= 0.0
        try
            free_bytes, _ = CUDA.Mem.info()
            budget_gb = 0.9 * free_bytes / 1024^3
            @info "[AutoMesh] VRAM budget auto-detected: $(round(budget_gb, digits=2)) GB (90 % of free)"
        catch
            budget_gb = 8.0
            @info "[AutoMesh] CUDA unavailable; using default 8 GB budget"
        end
    end
    safety = max(1.0, MESH_SAFETY_FACTOR)
    budget_bytes = budget_gb * 1024^3 / safety

    # Initial num_levels — user-provided, capped below by min_coarse_blocks fit
    num_levels_try = max(1, NUM_LEVELS_CONFIG)

    est_bytes = 0.0
    est_counts = Int[]

    function project_counts(num_levels_local)
        dx_fine   = MESH_MIN_DX
        dx_coarse = dx_fine * 2.0^(num_levels_local - 1)

        ext_x = (chord_up + chord_down) * L_chord + L_stream
        if HALF_Y_DOMAIN
            ext_y = chord_lat * L_chord + L_span_y
        else
            ext_y = 2 * chord_lat * L_chord + L_span_y
        end
        ext_z = 2 * chord_ht * L_chord + L_span_z

        ext_x = min(ext_x, L_stream + cap_up + cap_down)
        ext_y = min(ext_y, L_span_y + 2 * cap_lat)
        ext_z = min(ext_z, L_span_z + 2 * cap_ht)

        # Snap to L1 cell grid, enforce min_coarse_blocks per axis at L1
        min_cells = MIN_COARSE_BLOCKS * bs
        nx = bs*_stable_grid_block_count(ext_x,dx_coarse,bs;minimum_blocks=MIN_COARSE_BLOCKS)
        ny = bs*_stable_grid_block_count(ext_y,dx_coarse,bs;minimum_blocks=MIN_COARSE_BLOCKS)
        nz = bs*_stable_grid_block_count(ext_z,dx_coarse,bs;minimum_blocks=MIN_COARSE_BLOCKS)

        domain_x = nx * dx_coarse
        wake_start = chord_up * L_chord + L_stream - 0.1 * L_chord
        wake_end   = min(domain_x, wake_start + 0.1 * L_chord)
        counts = estimate_active_blocks_per_level(
            num_levels_local, dx_coarse, bs, nx, ny, nz,
            (chord_up * L_chord, chord_lat * L_chord, chord_ht * L_chord),
            (chord_up * L_chord + L_stream, chord_lat * L_chord + L_span_y, chord_ht * L_chord + L_span_z),
            DOMAIN_BODY_HALO_CHORDS,
            wake_end, wake_start,
        )
        bytes_per_cell = estimate_bytes_per_cell()
        total_bytes = sum(counts) * bs^3 * bytes_per_cell
        return counts, total_bytes
    end

    # Clamp num_levels to satisfy min_coarse_blocks (same as Fix 1 but on
    # the auto-sized extents).
    while num_levels_try > 1
        dx_coarse_try = MESH_MIN_DX * 2.0^(num_levels_try - 1)
        min_side = min(
            (chord_up + chord_down) * L_chord + L_stream,
            L_span_y + 2 * chord_lat * L_chord,
            L_span_z + 2 * chord_ht * L_chord,
        )
        if min_side >= MIN_COARSE_BLOCKS * bs * dx_coarse_try
            break
        end
        num_levels_try -= 1
    end
    if num_levels_try < NUM_LEVELS_CONFIG
        @info "[AutoMesh] num_levels $(NUM_LEVELS_CONFIG) → $(num_levels_try) (min_coarse_blocks=$(MIN_COARSE_BLOCKS))"
    end

    # Monotone fit loop — at most 8 iterations (shrink → expand)
    for iter in 1:8
        est_counts, est_bytes = project_counts(num_levels_try)
        if est_bytes <= budget_bytes
            break
        end
        # Over budget — shrink downstream first, then lateral, then upstream
        over = est_bytes / budget_bytes
        shrink = clamp(sqrt(1.0 / over), 0.5, 0.99)
        if chord_down > 2.0
            chord_down = max(2.0, chord_down * shrink)
        elseif chord_lat > 1.5
            chord_lat = max(1.5, chord_lat * shrink)
            chord_ht  = max(1.5, chord_ht  * shrink)
        elseif chord_up > 1.5
            chord_up = max(1.5, chord_up * shrink)
        else
            # We have hit the floor — last resort: drop a level
            if num_levels_try > 1
                num_levels_try -= 1
                @info "[AutoMesh] Iter $iter: dropping num_levels to $(num_levels_try) to fit VRAM"
            else
                @warn "[AutoMesh] Cannot fit VRAM even at minimum extents & 1 level. Budget=$(round(budget_gb,digits=2)) GB; estimate=$(round(est_bytes/1024^3,digits=2)) GB. Consider larger min_dx."
                break
            end
        end
    end

    # Grow downstream if well under budget (≤ 60 %) and user used defaults
    if est_bytes < 0.6 * budget_bytes && DOMAIN_DOWNSTREAM_CHORDS <= 0.0
        grow_cap = min(2.0 * chord_down_0, cap_down / L_chord)
        while chord_down < grow_cap
            trial_chord_down = min(grow_cap, chord_down * 1.25)
            save = chord_down; chord_down = trial_chord_down
            trial_counts, trial_bytes = project_counts(num_levels_try)
            if trial_bytes > 0.8 * budget_bytes
                chord_down = save   # roll back
                break
            end
            est_counts, est_bytes = trial_counts, trial_bytes
        end
    end

    # Commit legacy globals so the rest of compute_domain_from_mesh sees
    # consistent values. dx_coarse recomputed deterministically.
    dx_fine   = MESH_MIN_DX
    dx_coarse = dx_fine * 2.0^(num_levels_try - 1)

    # Write back globals expected by the legacy path
    global NUM_LEVELS_CONFIG    = num_levels_try
    global SURFACE_RESOLUTION   = max(1, Int(round(max(L_stream, L_chord) / dx_fine)))
    global DOMAIN_UPSTREAM      = chord_up   * L_chord / max(REFERENCE_LENGTH_FOR_MESHING, L_stream)
    global DOMAIN_DOWNSTREAM    = chord_down * L_chord / max(REFERENCE_LENGTH_FOR_MESHING, L_stream)
    global DOMAIN_LATERAL       = chord_lat  * L_chord / max(REFERENCE_LENGTH_FOR_MESHING, L_stream)
    global DOMAIN_HEIGHT        = chord_ht   * L_chord / max(REFERENCE_LENGTH_FOR_MESHING, L_stream)

    println("\n[AutoMesh] ═══════════════════════════════════════════════════")
    @printf("[AutoMesh] min_dx = %.4f m, num_levels = %d (budget %.2f GB · safety %.2f)\n",
            MESH_MIN_DX, num_levels_try, budget_gb, safety)
    @printf("[AutoMesh] Extents (chord units): up=%.1f, down=%.1f, lat=%.1f, ht=%.1f\n",
            chord_up, chord_down, chord_lat, chord_ht)
    @printf("[AutoMesh] dx_fine=%.4g m, dx_coarse=%.4g m, L1 floor=%d blocks/axis\n",
            dx_fine, dx_coarse, MIN_COARSE_BLOCKS)
    @printf("[AutoMesh] Active-block estimate per level: %s (total %.2fM cells)\n",
            join(est_counts, ", "),
            sum(est_counts) * Float64(bs)^3 / 1e6)
    @printf("[AutoMesh] Estimated VRAM: %.2f GB / %.2f GB budget\n",
            est_bytes / 1024^3, budget_gb)
    println("[AutoMesh] ═══════════════════════════════════════════════════\n")

    return num_levels_try
end

"Return the mesh offset that places its AABB center at an absolute coordinate inside the resolved domain."
function object_center_offset_m(mesh_min::Float64, mesh_max::Float64,
                                domain_size::Float64, target::Float64, key::String)
    isfinite(target) && isfinite(domain_size) && domain_size > 0 &&
        isfinite(mesh_min) && isfinite(mesh_max) && mesh_max >= mesh_min ||
        error("advanced.domain.$key requires finite body bounds, target, and positive domain size")
    center = (mesh_min + mesh_max) / 2
    half_extent = (mesh_max - mesh_min) / 2
    lowest = half_extent
    highest = domain_size - half_extent
    tolerance = max(1.0e-10 * domain_size, 1.0e-12)
    if highest < lowest - tolerance || target < lowest - tolerance || target > highest + tolerance
        error("advanced.domain.$key=$(target) m does not fit the body inside the resolved domain. " *
              "Valid AABB-center range is [$(round(lowest, digits=6)), $(round(highest, digits=6))] m " *
              "for a body extent of $(round(mesh_max - mesh_min, digits=6)) m and domain size $(round(domain_size, digits=6)) m.")
    end
    return target - center
end

function compute_domain_from_mesh(mesh_min::Tuple{Float64,Float64,Float64}, mesh_max::Tuple{Float64,Float64,Float64})
    # Optional auto-sizer. When mesh.min_dx is set in YAML, this rewrites
    # the legacy DOMAIN_* / SURFACE_RESOLUTION / NUM_LEVELS_CONFIG globals
    # to values that fit the VRAM budget; otherwise it's a no-op and the
    # legacy path below runs unchanged.
    compute_domain_auto_extents(mesh_min, mesh_max)
    mesh_center = ((mesh_min[1]+mesh_max[1])/2, (mesh_min[2]+mesh_max[2])/2, (mesh_min[3]+mesh_max[3])/2)
    mesh_extent = (mesh_max[1]-mesh_min[1], mesh_max[2]-mesh_min[2], mesh_max[3]-mesh_min[3])
    preserve_symmetry_y=SYMMETRIC_ANALYSIS && isdefined(@__MODULE__,:DOMAIN_PRESERVE_SYMMETRY_Y) && DOMAIN_PRESERVE_SYMMETRY_Y
    preserve_symmetry_y && mesh_max[2]<=0 && error("The preserved model lies entirely outside the positive-Y symmetry domain")
    
    ref_length = REFERENCE_LENGTH_FOR_MESHING > 0 ? REFERENCE_LENGTH_FOR_MESHING :
                 (REFERENCE_DIMENSION == :x ? mesh_extent[1] : REFERENCE_DIMENSION == :y ? mesh_extent[2] :
                  REFERENCE_DIMENSION == :z ? mesh_extent[3] : maximum(mesh_extent))
    
    ref_chord = REFERENCE_CHORD_CONFIG > 0 ? REFERENCE_CHORD_CONFIG : mesh_extent[1]
    ref_area = REFERENCE_AREA_CONFIG > 0 ? REFERENCE_AREA_CONFIG : 
               (SYMMETRIC_ANALYSIS ? mesh_extent[2]*mesh_extent[3]*2 : mesh_extent[2]*mesh_extent[3])
    
    moment_center_rel = (Float64(MOMENT_CENTER_CONFIG[1]), Float64(MOMENT_CENTER_CONFIG[2]), Float64(MOMENT_CENTER_CONFIG[3]))
    
    u_phys, nu_phys, rho_phys = FLOW_VELOCITY, FLUID_KINEMATIC_VISCOSITY, FLUID_DENSITY
    re_number = u_phys * ref_length / nu_phys
    dx_fine = ref_length / SURFACE_RESOLUTION

    scale_u_phys = abs(u_phys)
    scale_u_lattice = abs(Float64(U_TARGET))
    scaling_note = ""
    if scale_u_phys <= eps(Float64) || scale_u_lattice <= eps(Float64)
        if COMPRESSIBLE_MODE
            scale_u_phys = SPEED_OF_SOUND_PHYSICAL
            scale_u_lattice = sqrt(Float64(CS2_PHYSICS))
            scaling_note = "zero freestream: using acoustic scale c_phys -> c_lattice for viscosity/time scaling"
        else
            scale_u_phys = max(SPEED_OF_SOUND_PHYSICAL, eps(Float64))
            scale_u_lattice = max(scale_u_lattice, 0.08)
            scaling_note = "zero freestream: using fallback finite reference scale for viscosity/time scaling"
        end
    end

    velocity_scale = scale_u_phys / scale_u_lattice
    time_scale = dx_fine / velocity_scale
    nu_lattice_fine = nu_phys * time_scale / (dx_fine^2)
    tau_fine_computed = 0.5 + nu_lattice_fine / Float64(CS2_PHYSICS)
    tau_fine = max(tau_fine_computed, Float64(TAU_MIN))

    print_re_analysis(re_number, ref_length, SURFACE_RESOLUTION, Float32(scale_u_lattice);
                      tau_override=tau_fine_computed, scaling_note=scaling_note)

    # Report the numerical floor that actually controls resolved wake dynamics.
    # The physical viscosity can imply tau ~= 0.5, but tau_min and SGS floors set
    # a much lower effective Reynolds number on the lattice.
    lattice_ref_length = ref_length / dx_fine
    nu_tau_floor = max(tau_fine - 0.5, 0.0) * Float64(CS2_PHYSICS)
    nu_sgs_floor = max(Float64(NU_SGS_BACKGROUND),
                       VARIABLE_NU_SGS ? Float64(VARIABLE_NU_SGS_FLOOR) : 0.0)
    nu_eff_floor = nu_tau_floor + nu_sgs_floor
    if nu_eff_floor > 0.0 && scale_u_lattice > 0.0
        re_eff_floor = scale_u_lattice * lattice_ref_length / nu_eff_floor
        @printf("[High-Re] Effective lattice floor: nu_tau=%.6g, nu_sgs_floor=%.6g, L/dx=%.1f -> Re_floor≈%.3e\n",
                nu_tau_floor, nu_sgs_floor, lattice_ref_length, re_eff_floor)
        if tau_fine > tau_fine_computed + 1.0e-8
            @printf("[High-Re] tau_min clip active: tau_physical=%.9f -> tau_used=%.9f\n",
                    tau_fine_computed, tau_fine)
        end
    end
    
    domain_x = ref_length * (DOMAIN_UPSTREAM + DOMAIN_DOWNSTREAM) + mesh_extent[1]
    if GROUND_ANALYSIS
        domain_y = mesh_extent[2] + ref_length * DOMAIN_HEIGHT
        domain_z = mesh_extent[3] + 2 * ref_length * DOMAIN_LATERAL
    elseif HALF_Y_DOMAIN
        domain_y = (preserve_symmetry_y ? mesh_max[2] : mesh_extent[2]) + ref_length * DOMAIN_LATERAL
        domain_z = mesh_extent[3] + 2 * ref_length * DOMAIN_HEIGHT
    else
        domain_y = mesh_extent[2] + 2 * ref_length * DOMAIN_LATERAL
        domain_z = mesh_extent[3] + 2 * ref_length * DOMAIN_HEIGHT
    end
    if DOMAIN_SPAN_OVERRIDE > 0.0
        @printf("[Domain] Y-span override: active domain_y %.6f -> %.6f m (mesh y extent %.6f m; overspan geometry allowed)\n",
                domain_y, DOMAIN_SPAN_OVERRIDE, mesh_extent[2])
        domain_y = DOMAIN_SPAN_OVERRIDE
    end
    
    min_domain = min(domain_x, domain_y, domain_z)
    if !(PERIODIC_Y_AMR && SPAN_PERIODIC_Y)
        max_levels_domain = compute_max_levels_for_domain(min_domain, dx_fine, BLOCK_SIZE_CONFIG, MIN_COARSE_BLOCKS)
    else
        # Periodic-Y AMR (27 Oct 2026): the healthy-pyramid rule (min_coarse_blocks
        # coarse blocks per axis, so that interpolation stencils sit inside the
        # parent grid) does not apply to the periodic axis — its stencils wrap. A
        # thin span needs only ONE coarsest block in Y; x and z keep the rule.
        max_levels_domain = min(
            compute_max_levels_for_domain(min(domain_x, domain_z), dx_fine, BLOCK_SIZE_CONFIG, MIN_COARSE_BLOCKS),
            periodic_y_max_levels(domain_y, dx_fine, BLOCK_SIZE_CONFIG))
    end

    num_levels = NUM_LEVELS_CONFIG > 0 ? min(NUM_LEVELS_CONFIG, max_levels_domain) :
                 (AUTO_LEVELS ? min(max_levels_domain, MAX_LEVELS) : min(8, max_levels_domain))
    if PERIODIC_Y_AMR && SPAN_PERIODIC_Y && NUM_LEVELS_CONFIG > 0 && num_levels < NUM_LEVELS_CONFIG
        @warn "[Domain] Periodic-Y AMR: num_levels clamped $(NUM_LEVELS_CONFIG) → $(num_levels) by the domain size " *
              "(x/z need min_coarse_blocks=$(MIN_COARSE_BLOCKS) coarsest blocks; the Y period needs 1)."
    end
    
    dx_coarse = dx_fine * 2^(num_levels - 1)
    dx_levels = [dx_fine * 2^(num_levels - lvl) for lvl in 1:num_levels]
    
    nx_coarse = BLOCK_SIZE_CONFIG*_stable_grid_block_count(domain_x,dx_coarse,BLOCK_SIZE_CONFIG)
    ny_coarse = BLOCK_SIZE_CONFIG*_stable_grid_block_count(domain_y,dx_coarse,BLOCK_SIZE_CONFIG)
    nz_coarse = BLOCK_SIZE_CONFIG*_stable_grid_block_count(domain_z,dx_coarse,BLOCK_SIZE_CONFIG)
    
    domain_x, domain_y, domain_z = nx_coarse*dx_coarse, ny_coarse*dx_coarse, nz_coarse*dx_coarse
    bx_max, by_max, bz_max = nx_coarse÷BLOCK_SIZE_CONFIG, ny_coarse÷BLOCK_SIZE_CONFIG, nz_coarse÷BLOCK_SIZE_CONFIG

    # ── Periodic-Y AMR: the period must be what was asked for (27 Oct 2026) ──
    # The Y extent is snapped to whole COARSEST blocks (BLOCK_SIZE·dx_coarse =
    # BLOCK_SIZE·2^(L-1) finest cells), so every finer level tiles the period
    # exactly. A requested span (span_override) that does not land on that
    # quantum would be silently re-sized to a different period — refuse instead,
    # naming the nearest valid spans.
    if PERIODIC_Y_AMR && SPAN_PERIODIC_Y
        period_quantum = BLOCK_SIZE_CONFIG * dx_coarse
        if DOMAIN_SPAN_OVERRIDE > 0.0 && abs(domain_y - DOMAIN_SPAN_OVERRIDE) >= dx_fine
            n_lo = max(1, floor(Int, DOMAIN_SPAN_OVERRIDE / period_quantum + 1.0e-9))
            error(@sprintf("periodic_y_amr: the requested span (advanced.domain.span_override = %.6g m = %.2f finest cells) is not a whole number of coarsest blocks. With %d level(s) the period must be a multiple of BLOCK_SIZE·2^(levels-1) = %d finest cells = %.6g m; it would have been re-sized to %.6g m. Use span_override = %.6g m or %.6g m (or change surface_resolution / num_levels).",
                           DOMAIN_SPAN_OVERRIDE, DOMAIN_SPAN_OVERRIDE / dx_fine, num_levels,
                           BLOCK_SIZE_CONFIG * 2^(num_levels - 1), period_quantum, domain_y,
                           n_lo * period_quantum, (n_lo + 1) * period_quantum))
        end
        @printf("[Domain] Periodic-Y AMR period: %.6g m = %d coarsest blocks = %d finest cells (%d levels)\n",
                domain_y, by_max, round(Int, domain_y / dx_fine), num_levels)
    end

    # ── Resolved-domain transparency + quasi-2D reference-area footgun guard ──
    # (2026-06-19) The span override (and any domain dim) is BLOCK-SNAPPED above to a
    # multiple of BLOCK_SIZE·dx_coarse, so the RESOLVED domain_y can differ from the
    # requested value (e.g. span_override 0.1667 → 0.16666 raw → 9 cells → snapped 16
    # cells = 0.3333, a silent 2×). For a quasi-2D run, Cd normalizes by
    # reference_area = ref_chord × SPANNED_Y; if the user set reference_area to the
    # REQUESTED (pre-snap) span, Cd is wrong by the snap ratio. Print the resolved
    # domain dims always, and WARN when a span_override case's configured reference
    # area disagrees with ref_chord × resolved_domain_y by >2%.
    @printf("[Domain] Resolved grid: domain = %.5f × %.5f × %.5f m  (%d × %d × %d coarse cells, dx_coarse=%.6f)\n",
            domain_x, domain_y, domain_z, nx_coarse, ny_coarse, nz_coarse, dx_coarse)
    if DOMAIN_SPAN_OVERRIDE > 0.0 && REFERENCE_AREA_CONFIG > 0.0
        _implied_area = ref_chord * domain_y
        _area_ratio = REFERENCE_AREA_CONFIG / max(_implied_area, 1e-12)
        if abs(_area_ratio - 1.0) > 0.02
            @warn @sprintf("[Domain] reference_area_of_full_model=%.5f disagrees with ref_chord×resolved_span_Y=%.5f×%.5f=%.5f (ratio %.3f). For a quasi-2D span-override case the reference area should equal ref_chord × the RESOLVED (block-snapped) Y span — your Cd will be off by this ratio. Set reference_area_of_full_model = %.5f.",
                           REFERENCE_AREA_CONFIG, ref_chord, domain_y, _implied_area, _area_ratio, _implied_area)
        end
    end

    if DOMAIN_OBJECT_CENTER_X_M !== nothing
        mesh_x_offset = object_center_offset_m(mesh_min[1], mesh_max[1], domain_x,
                                               DOMAIN_OBJECT_CENTER_X_M, "object_center_x_m")
        @printf("[Domain] Streamwise placement: object AABB center x=%.6f m\n", DOMAIN_OBJECT_CENTER_X_M)
    elseif DOMAIN_OBJECT_CENTER_X_FRACTION === nothing
        mesh_x_offset = ref_length * DOMAIN_UPSTREAM - mesh_min[1]
    else
        target_center_fraction = Float64(DOMAIN_OBJECT_CENTER_X_FRACTION)
        center_to_min = mesh_center[1] - mesh_min[1]
        center_to_max = mesh_max[1] - mesh_center[1]
        min_center_fraction = center_to_min / domain_x
        max_center_fraction = 1.0 - center_to_max / domain_x

        if target_center_fraction < min_center_fraction || target_center_fraction > max_center_fraction
            error("advanced.domain.object_center_x_fraction=$(target_center_fraction) does not fit inside the current x-domain. " *
                  "Valid range is [$(round(min_center_fraction, digits=4)), $(round(max_center_fraction, digits=4))]. " *
                  "Increase upstream/downstream or choose a center fraction inside that range.")
        end

        mesh_x_offset = target_center_fraction * domain_x - mesh_center[1]
        placed_min_x = mesh_min[1] + mesh_x_offset
        placed_max_x = mesh_max[1] + mesh_x_offset
        @printf("[Domain] Streamwise placement: object center at x/Lx = %.3f -> x=[%.6f, %.6f]\n",
                target_center_fraction, placed_min_x, placed_max_x)
    end

    if DOMAIN_OBJECT_CENTER_Y_M !== nothing && SYMMETRIC_ANALYSIS
        error("advanced.domain.object_center_y_m cannot move a body off the y=0 symmetry plane; omit it")
    end
    if DOMAIN_OBJECT_CENTER_Y_M !== nothing && PERIODIC_Y_BODY_SHIFT != 0.0
        error("advanced.domain.object_center_y_m and periodic_y_body_shift cannot both be set")
    end
    mesh_y = DOMAIN_OBJECT_CENTER_Y_M === nothing ?
             (preserve_symmetry_y ? 0.0 : HALF_Y_DOMAIN ? -mesh_min[2] : domain_y/2 - mesh_center[2]) :
             object_center_offset_m(mesh_min[2], mesh_max[2], domain_y,
                                    DOMAIN_OBJECT_CENTER_Y_M, "object_center_y_m")
    if DOMAIN_OBJECT_CENTER_Y_M !== nothing
        @printf("[Domain] Transverse placement: object AABB center y=%.6f m%s\n",
                DOMAIN_OBJECT_CENTER_Y_M,
                GROUND_ANALYSIS ? @sprintf(" (ground clearance %.6f m)", mesh_min[2] + mesh_y) : "")
    end
    if PERIODIC_Y_AMR && PERIODIC_Y_BODY_SHIFT != 0.0
        # periodic-Y AMR: optional body placement along the periodic axis
        mesh_y += PERIODIC_Y_BODY_SHIFT
        @printf("[Domain] Periodic-Y AMR: body shifted along Y by %.6g m\n", PERIODIC_Y_BODY_SHIFT)
    end
    mesh_z = DOMAIN_OBJECT_CENTER_Z_M === nothing ? domain_z/2 - mesh_center[3] :
             object_center_offset_m(mesh_min[3], mesh_max[3], domain_z,
                                    DOMAIN_OBJECT_CENTER_Z_M, "object_center_z_m")
    DOMAIN_OBJECT_CENTER_Z_M === nothing ||
        @printf("[Domain] Vertical placement: object AABB center z=%.6f m\n", DOMAIN_OBJECT_CENTER_Z_M)
    mesh_offset = (mesh_x_offset, mesh_y, mesh_z)
    
    length_scale = dx_fine
    
    tau_levels = Float32[]
    for lvl in 1:num_levels
        if LEVEL_VISCOSITY_MODE === :physical
            # advanced.refinement.level_viscosity: "physical" (lbm-taufix, 28 Oct 2026). dt ∝ dx on every level (each
            # child takes two steps per parent step), so the same physical viscosity needs nu_lat,l = nu_lat,f/2^(L-l).
            # The finest level's tau is the reference (it already carries the TAU_MIN floor); a coarser level whose
            # physical tau would fall below TAU_MIN is floored there, and the resulting viscosity excess is reported.
            tau_phys_lvl = lvl == num_levels ? tau_fine : 0.5 + (tau_fine - 0.5) / 2.0^(num_levels - lvl)
            tau_lvl = max(tau_phys_lvl, Float64(TAU_MIN))
            # (VA3) the floor report gives both ratios: to the finest level's viscosity and to physical (from the
            # finest level's computed, unfloored tau), with the legacy rule's two figures.
            nu_ref_phys = max(tau_fine_computed - 0.5, 1e-12)
            if lvl == num_levels
                if tau_fine > tau_fine_computed + 1.0e-8
                    @printf("[Levels] level %d (finest): physical tau %.6f is below tau_min %.6f: floored; its viscosity is %.3f x physical\n",
                            lvl, tau_fine_computed, Float64(TAU_MIN), (tau_fine - 0.5) / nu_ref_phys)
                end
            elseif tau_lvl > tau_phys_lvl
                f_lvl = 2.0^(num_levels - lvl); f_leg = 4.0^(num_levels - lvl)
                @printf("[Levels] level %d: physical tau %.6f is below tau_min %.6f: floored; its viscosity is %.3f x the finest level's and %.3f x physical (legacy rule: %.0f x and %.3f x)\n",
                        lvl, 0.5 + (tau_fine_computed - 0.5) / f_lvl, Float64(TAU_MIN),
                        (tau_lvl - 0.5) * f_lvl / max(tau_fine - 0.5, 1e-12), (tau_lvl - 0.5) * f_lvl / nu_ref_phys,
                        f_leg, f_leg * (tau_fine - 0.5) / nu_ref_phys)
            end
        else
            tau_lvl = lvl == num_levels ? tau_fine : 0.5 + (tau_fine - 0.5) * 2.0^(num_levels - lvl)
        end
        push!(tau_levels, Float32(tau_lvl))
    end
    if LEVEL_VISCOSITY_MODE === :physical
        # (VA3, VA5) the level coupling's combined f_neq factor, prolongation side: the pre-collision ratio
        # tau_f/(2 tau_c) times the post-collision correction (1 - 1/tau_f)/(1 - 1/tau_c), i.e. (tau_f - 1)/(2 (tau_c - 1))
        # (the restriction's is its reciprocal). It must be finite and inside the [0.01, 100] clamps of
        # interpolate_with_rescaling / amr_coupling.jl, or the coupling silently changes it: negative when the levels
        # straddle tau = 1, not finite at tau = 1 exactly (the coarse post-collision f_neq vanishes and the guard skips
        # the correction), clamped near tau = 1.
        for lvl in 1:num_levels-1
            tc, tf = tau_levels[lvl], tau_levels[lvl+1]
            fac = (Float64(tf) - 1.0) / (2.0 * (Float64(tc) - 1.0))
            if !isfinite(fac) || !(0.01 <= fac <= 100.0)
                why = (tc - 1.0f0) * (tf - 1.0f0) < 0.0f0 ? "straddle tau = 1" :
                      (tc == 1.0f0 || tf == 1.0f0) ? "have a level at exactly tau = 1" : "are too close to tau = 1"
                error(@sprintf("advanced.refinement.level_viscosity: \"physical\": levels %d and %d %s (tau %.6f and %.6f): the level coupling's combined non-equilibrium factor (tau_f - 1)/(2 (tau_c - 1)) = %.6g is not finite or lies outside its [0.01, 100] clamps (the post-collision ratio (1 - 1/tau_c)/(1 - 1/tau_f) cannot be carried). Use the legacy rule (key absent), or a resolution or viscosity that keeps every level clear of tau = 1.",
                               lvl, lvl + 1, why, tc, tf, fac))
            end
        end
    end

    force_scale = rho_phys * length_scale^4 / time_scale^2
    moment_center_phys = (mesh_min[1] + mesh_offset[1] + moment_center_rel[1]*ref_chord,
                          mesh_center[2] + mesh_offset[2] + moment_center_rel[2]*ref_chord,
                          mesh_center[3] + mesh_offset[3] + moment_center_rel[3]*ref_chord)
    
    # Thermal DDF: compute tau_g levels from kappa = nu/Pr
    # Phase 4b (Frapolli): When D3Q39 flow, thermal g also uses D3Q39 → cs²_thermal = 2/3.
    # When D3Q27 flow, thermal uses D3Q27 → cs²_thermal = 1/3.
    tau_g_fine = 0.0
    tau_g_levels = Float32[]
    if DDF_ENABLED
        local cs2_thermal = (LATTICE_TYPE == :d3q39) ? 2.0 / 3.0 : 1.0 / 3.0
        kappa_lattice = nu_lattice_fine / DDF_PRANDTL
        tau_g_fine = 0.5 + kappa_lattice / cs2_thermal
        for lvl in 1:num_levels
            tau_g_lvl = lvl == num_levels ? tau_g_fine : 0.5 + (tau_g_fine - 0.5) * 2.0^(num_levels - lvl)
            push!(tau_g_levels, Float32(tau_g_lvl))
        end
        @printf("[Thermal] Frapolli model: cs²_thermal = %.4f (%s)\n",
                cs2_thermal, LATTICE_TYPE == :d3q39 ? "D3Q39 — same as flow" : "D3Q27")
        @printf("[Thermal] κ_lattice = %.6f (ν/Pr = %.6f / %.3f)\n",
                kappa_lattice, nu_lattice_fine, DDF_PRANDTL)
        @printf("[Thermal] τ_g_fine  = %.6f (molecular — SGS κ_sgs = ν_eddy/Pr_t added at runtime)\n", tau_g_fine)
        @printf("[Thermal] Pr_t = %.3f (turbulent Prandtl number for SGS thermal diffusivity)\n", DDF_TURBULENT_PRANDTL)
    end

    bytes_per_cell = TEMPORAL_INTERPOLATION ? 220 : 160
    if DDF_ENABLED
        bytes_per_cell += 220  # g, g_temp, g_old, g_post_collision, temperature, temperature_old
    end
    total_cells_est = bx_max * by_max * bz_max * BLOCK_SIZE_CONFIG^3
    for lvl in 2:num_levels; total_cells_est += Int(ceil(total_cells_est * 0.08)); end
    estimated_memory_gb = total_cells_est * bytes_per_cell / 1e9
    
    p = DOMAIN_PARAMS
    p.initialized = true
    p.num_levels = num_levels
    p.mesh_min, p.mesh_max, p.mesh_center, p.mesh_extent = mesh_min, mesh_max, mesh_center, mesh_extent
    p.reference_length, p.reference_chord, p.reference_area = ref_length, ref_chord, ref_area
    p.moment_center = moment_center_phys
    p.domain_min, p.domain_max = (0.0, 0.0, 0.0), (domain_x, domain_y, domain_z)
    p.domain_size, p.mesh_offset = (domain_x, domain_y, domain_z), mesh_offset
    p.dx_fine, p.dx_coarse, p.dx_levels = dx_fine, dx_coarse, dx_levels
    p.nx_coarse, p.ny_coarse, p.nz_coarse = nx_coarse, ny_coarse, nz_coarse
    p.bx_max, p.by_max, p.bz_max = bx_max, by_max, bz_max
    p.l_char, p.nu_lattice = ref_length/dx_coarse, nu_lattice_fine
    p.tau_coarse, p.tau_levels = tau_levels[1], tau_levels
    p.cs2, p.cs4 = CS2_PHYSICS, CS4_PHYSICS
    p.re_number, p.u_physical, p.rho_physical, p.nu_physical = re_number, u_phys, rho_phys, nu_phys
    p.length_scale, p.time_scale, p.velocity_scale, p.force_scale = length_scale, time_scale, velocity_scale, force_scale
    p.tau_fine, p.tau_margin_percent = tau_fine, (tau_fine - 0.5)/0.5*100
    p.wall_model_active, p.y_plus_first_cell = WALL_MODEL_ENABLED, 0.0
    p.estimated_memory_gb = estimated_memory_gb
    p.tau_g_fine, p.tau_g_levels = tau_g_fine, tau_g_levels
    
    return p
end

function print_domain_summary()
    !DOMAIN_PARAMS.initialized && (println("Domain not initialized."); return)
    p = DOMAIN_PARAMS
    println("╔══════════════════════════════════════════════════════════════════╗")
    println("║                 LBM DOMAIN CONFIGURATION SUMMARY                 ║")
    println("╠══════════════════════════════════════════════════════════════════╣")
    @printf("║  Case: %-40s      ║\n", basename(CASE_DIR))
    @printf("║  Re = %-12.0f | %d levels | Est. %.2f GB                    ║\n", p.re_number, p.num_levels, p.estimated_memory_gb)
    println("╚══════════════════════════════════════════════════════════════════╝")
end

get_domain_params() = DOMAIN_PARAMS.initialized ? DOMAIN_PARAMS : error("Domain not initialized")

# The physical time of one SOLVER step (lbm-taufix VA3, 28 Oct 2026; finding W7). One solver step is one step of the
# COARSEST level (`recursive_step!(grids, 1, t, ...)` once per step; each finer level takes two substeps per parent
# step), while `time_scale` is the FINEST level's step. So a solver step lasts time_scale * 2^(L-1). Every conversion
# of a step count to physical time (time_phys, domain_length_covered -> STEPS, the summaries, the plane files) uses
# this; with one level it is time_scale * 1.0, exactly time_scale (single-level outputs are byte-identical).
solver_step_time(p) = p.time_scale * 2.0^(p.num_levels - 1)
get_num_levels() = DOMAIN_PARAMS.initialized ? DOMAIN_PARAMS.num_levels : NUM_LEVELS_CONFIG
