# ⛔ HALT GUARD — must precede imports, output, configuration mutation, allocation, and evolution.
include(joinpath(dirname(@__DIR__), "tools", "halt_guard.jl"))

# SOLVER-PRESET BOOTSTRAP (2026-06-10b): must run BEFORE dependencies.jl is
# included — the pristine-mode D3Q19 seam gates it sets are compile-time consts
# baked into the GPU kernels at module load. See preset_bootstrap.jl.
include("io/case_id_contract.jl")
include("io/batch_outcome_contract.jl")
using .BatchOutcomeContract
# lbm-io-opts (27 Oct 2026): LUDWIG_CASES_FILE / LUDWIG_CASE or --cases-file / --case choose the
# queue (src/io/case_id_contract.jl); with none of them it is <root>/cases_to_run.yaml as before.
# Arguments are read only when this file is the program (an including script keeps its ARGS).
const STARTUP_LAUNCH_SELECTOR = CaseIDContract.resolve_launch_selector(
    ENV, CaseIDContract.program_is(PROGRAM_FILE, @__FILE__) ? ARGS : String[])
STARTUP_LAUNCH_SELECTOR.kind === :default ||
    println("[Cases] Queue from $(STARTUP_LAUNCH_SELECTOR.source) = $(STARTUP_LAUNCH_SELECTOR.value)")
const STARTUP_SELECTION = CaseIDContract.capture_startup_selection(joinpath(@__DIR__, "..");
                                                                   selector=STARTUP_LAUNCH_SELECTOR)
const STARTUP_SUMMARY_SINK = CaseIDContract.capture_summary_sink(STARTUP_SELECTION)
include("preset_bootstrap.jl")

function _capture_startup_runtime_selections(snapshot::CaseIDContract.StartupSelection)
    CaseIDContract.verify_startup_selection(snapshot)
    root_dir = snapshot.root_dir
    runtimes = CaseIDContract.RuntimeSelection[]
    for selection in snapshot.selections
        parsed = YAML.load(selection.config_text)
        parsed isa AbstractDict || error("Every selected config.yaml must parse to a mapping")
        push!(runtimes, CaseIDContract.capture_runtime_selection(root_dir, selection, parsed))
    end
    CaseIDContract.verify_startup_selection(snapshot)
    foreach(runtime -> CaseIDContract.verify_runtime_selection(root_dir, runtime), runtimes)
    return Tuple(runtimes)
end

# Capture config-derived STL bytes and the still-absent output targets before
# preset environment mutation, heavyweight dependencies, logging, or GPU work.
const STARTUP_RUNTIME_SELECTIONS = _capture_startup_runtime_selections(STARTUP_SELECTION)
preset_bootstrap!(STARTUP_SELECTION)

ENV["LC_ALL"] = "C"

include("dependencies.jl")


const SIMULATION_START_TIME = Ref(time())

walltime_str() = begin
    e = time() - SIMULATION_START_TIME[]
    @sprintf("%02d:%02d:%05.2f", floor(Int, e/3600), floor(Int, (e%3600)/60), e%60)
end

log_walltime(msg::String) = println("[$(walltime_str())] $msg")

# ──────────────────────────────────────────────────────────────────────────────
# ASYNC SLICE EXPORT TASK POOL (perf 2026-05-28)
# ──────────────────────────────────────────────────────────────────────────────
# Slice rendering + PNG IO takes 5-30 sec per slice plane × 3 planes.
# Before this change, those ran SYNCHRONOUSLY in the main loop, blocking the
# next GPU batch by ~20-60 sec total. With this change, we extract the slice
# data synchronously (Array(::CuArray) auto-syncs GPU), then dispatch the
# RENDERING (which is pure CPU) to a background Task. The GPU batch continues
# in parallel.
#
# Race-safety: extract_slice_data inside export_slice_image grabs the GPU
# arrays via Array(::CuArray). That's a SYNCHRONIZING device-to-host copy —
# it waits for outstanding GPU work to complete before the copy starts.
# The Task only reads CPU-local copies after that point; no GPU access from
# the Task. Safe in practice; documented assumption.
#
# wait_pending_slice_exports() must be called at end-of-simulation (and
# any other point we need a hard sync) to ensure all PNG writes finish
# before the program exits.
const _PENDING_SLICE_TASKS = Task[]

# 2026-05-28 — Slice-rendering cairo race fix. Plots.jl/GR/cairo's PNG writer
# (`cairo_surface_write_to_png`) is NOT thread-safe; concurrent calls from
# multiple Threads.@spawn-ed slice exports caused `print device already
# activated` + `Assertion failed: !surface->finished` → SIGABRT. The mutex
# serializes the savefig() calls while still allowing the GPU→host copies
# (Array(::CuArray)) and any numpy-like array post-processing to run async.
# The work_fn typically does both — for now we just lock the whole work_fn,
# which gives correctness at the cost of full serialization. Acceptable since
# slice IO is a small fraction of total wall time.
const _SLICE_RENDER_LOCK = ReentrantLock()

# ── Failure diagnostics (2026-05-29, user request) ─────────────────────────
# Writes full stacktraces + a per-batch heartbeat to a dedicated error-log file
# (separate from stdout, which can be lost/buffered on a hard exit). For HARD
# crashes (segfault / SIGABRT from C libs / OOM-kill) that bypass Julia's
# try/catch, the heartbeat is the key signal: the last line in the file is the
# last batch/stage that completed → tells us exactly where it died even with no
# Julia exception. ERROR_LOG_PATH is set per-case in solve_main().
const ERROR_LOG_PATH = Ref{String}("")

function _errlog(msg::AbstractString)
    path = ERROR_LOG_PATH[]
    isempty(path) && return
    try
        open(path, "a") do io
            println(io, "[", Dates.format(Dates.now(), "HH:MM:SS.sss"), "] ", msg)
            flush(io)
        end
    catch
        # never let logging itself crash the run
    end
end

# Heartbeat: flush "alive at step N (phase)" to the error log + an fsync so it
# survives a hard crash on the very next instruction.
function _heartbeat(step::Int, phase::AbstractString)
    _errlog("HEARTBEAT step=$step phase=$phase")
end

# Dump a caught exception with full backtrace to BOTH stdout and the error log.
function _log_exception(context::AbstractString, e, bt)
    _errlog("EXCEPTION in $context: $e")
    try
        io = IOBuffer()
        showerror(io, e, bt)
        _errlog(String(take!(io)))
    catch
    end
    println("!!! EXCEPTION in $context: $e")
    showerror(stdout, e, bt); println()
    flush(stdout)
end

"""
    spawn_slice_export!(label, slice_step, work_fn)

Run the slice-rendering/IO work `work_fn` on a background Task. Tracks the
Task in `_PENDING_SLICE_TASKS` so it can be waited on later.

Falls back to synchronous execution when `nthreads() == 1` (no benefit
from spawning).

Cairo-race-safe (2026-05-28): all work_fn invocations serialized through
`_SLICE_RENDER_LOCK`. The background Task still benefits from overlapping
with subsequent GPU kernels — the main thread doesn't wait on the lock.
"""
function spawn_slice_export!(label::String, slice_step::Int, work_fn::Function)
    if Threads.nthreads() <= 1
        # No multithreading available; run synchronously.
        lock(_SLICE_RENDER_LOCK) do
            run_export_stage_sync(label, slice_step, work_fn)
        end
        return
    end
    task = Threads.@spawn begin
        lock(_SLICE_RENDER_LOCK) do
            run_export_stage_sync($label, $slice_step, $work_fn)
        end
    end
    push!(_PENDING_SLICE_TASKS, task)
    # Reap completed tasks to avoid unbounded growth across long runs.
    filter!(t -> !istaskdone(t), _PENDING_SLICE_TASKS)
end

"""
    wait_pending_slice_exports()

Block until all background slice-export Tasks have completed.
"""
function wait_pending_slice_exports()
    isempty(_PENDING_SLICE_TASKS) && return
    pending = length(_PENDING_SLICE_TASKS)
    println("[SliceAsync] Waiting for $pending pending slice export task(s)...")
    for task in _PENDING_SLICE_TASKS
        try
            wait(task)
        catch e
            println("[SliceAsync] Warning: slice task failed: $e")
        end
    end
    empty!(_PENDING_SLICE_TASKS)
    println("[SliceAsync] All pending slice exports complete.")
end

wait_for_slice_tasks!() = wait_pending_slice_exports()

"""
    run_export_stage_sync(label, step, work_fn)

Synchronous wrapper that mirrors the original `run_export_stage` semantics
(label printing, timing, catch-and-warn). Used internally by
`spawn_slice_export!` for both the threaded and non-threaded paths.
"""
function run_export_stage_sync(label::String, step::Int, work_fn::Function)
    stage_start = time()
    println("[$label] START step $step")
    flush(stdout)
    try
        work_fn()
        stage_dt = time() - stage_start
        println("[$label] DONE step $step in $(round(stage_dt, digits=2)) s")
        flush(stdout)
        return true
    catch e
        stage_dt = time() - stage_start
        println("[$label] Warning: stage failed at step $step after $(round(stage_dt, digits=2)) s: $e")
        flush(stdout)
        # User request 2026-05-29: capture the FULL stacktrace to the error log
        # (the one-line warning above hid the real cause of slice-export deaths).
        _log_exception("slice-export [$label] step $step", e, catch_backtrace())
        return false
    end
end

@inline function ramp_shape(x::Float32, profile::Symbol)
    s = clamp(x, 0.0f0, 1.0f0)
    if profile === :linear
        return s
    elseif profile === :smoothstep
        return s * s * (3.0f0 - 2.0f0 * s)
    elseif profile === :smootherstep
        return s * s * s * (s * (s * 6.0f0 - 15.0f0) + 10.0f0)
    elseif profile === :smoothstep7
        s2 = s * s
        s4 = s2 * s2
        return s4 * (35.0f0 + s * (-84.0f0 + s * (70.0f0 - 20.0f0 * s)))
    else
        return 0.5f0 * (1.0f0 - cos(Float32(pi) * s))
    end
end

@inline function ramp_progress(batch_end::Int, ramp_steps::Int, profile::Symbol)
    ramp_steps <= 0 && return 1.0f0
    batch_end >= ramp_steps && return 1.0f0
    return ramp_shape(Float32(batch_end) / Float32(ramp_steps), profile)
end

function force_cleanup()
    if CUDA.functional()
        GC.gc(true); GC.gc(true); CUDA.reclaim()
        free_mem, total_mem = CUDA.memory_info()
        @printf("[Memory] VRAM: %.2f GB free / %.2f GB total\n", free_mem/1024^3, total_mem/1024^3)
        return free_mem
    end
    return 0
end


function solve_main(runtime_selection::CaseIDContract.RuntimeSelection,
                    output_claim::CaseIDContract.ClaimedOutput)
    CaseIDContract.verify_runtime_config(runtime_selection, CFG)
    CaseIDContract.verify_runtime_inputs(STARTUP_SELECTION.root_dir, runtime_selection)
    CaseIDContract.verify_claimed_output(runtime_selection, output_claim)
    SIMULATION_START_TIME[] = time()
    sim_start_datetime = Dates.now()

    println("\n" * "="^70)
    # PonD per-cell conservative kinetic-FV path (opt-in via advanced.numerics.pond.enabled).
    # The FV solver evolves conserved moments via shared face fluxes, structurally distinct from
    # the LBM stream-collide loop below, so it dispatches HERE — not through get_collision_kernel.
    # Same-level cross-block coupling is validated (src/collision/pond/pond_fv3d_blocks.jl); the
    # production-case path (geometry mask + boundaries-from-moments on the block/AMR domain +
    # surface drag) is the active integration. The current production path runs the VALIDATED
    # single-level uniform-grid driver on a config-declared grid + immersed body (cross-block/AMR
    # execution on the block domain is the next step; substrate validated in pond_fv3d_blocks.jl).
    POND_ENABLED && LBMIO_BUILD_ONLY &&   # lbm-io-opts: the PonD path has its own domain; refuse rather than run it
        error("basic.simulation.build_only is not supported with advanced.numerics.pond.enabled")
    if POND_ENABLED
        _pdir = joinpath(@__DIR__, "collision", "pond")
        include(joinpath(_pdir, "pond_grad.jl"))
        include(joinpath(_pdir, "pond_fv3d_gpu.jl"))
        include(joinpath(_pdir, "pond_fv3d_dugks.jl"))
        include(joinpath(_pdir, "pond_fv3d_driver.jl"))
        include(joinpath(_pdir, "pond_case_runner.jl"))
        # invokelatest: the files were just include()d (newer world age) ⇒ Julia 1.12 forbids a
        # direct call from this (older) world. invokelatest bridges it; non-pond runs never load these.
        return Base.invokelatest(run_pond_fv_case!, runtime_selection, output_claim)
    end
    lattice_label = LATTICE_TYPE == :d3q39 ? "D3Q39" :
                    LATTICE_TYPE == :d3q19 ? "D3Q19" :
                    LATTICE_TYPE == :pond  ? "PonD-FV" : "D3Q27"
    if COMPRESSIBLE_MODE
        solver_mode = "$(lattice_label) | COMPRESSIBLE SUPERSONIC"
    elseif COMPRESSIBILITY_CORRECTION
        solver_mode = "$(lattice_label) | Compressible"
    else
        solver_mode = "$(lattice_label) | Incompressible"
    end
    println("    LBM SOLVER | $(solver_mode) | WALE LES | SURFACE FORCE METHOD")
    println("    Case: $(basename(CASE_DIR)) | $(sim_start_datetime)")
    println("="^70)
    
    # Initialize lattice type and speed of sound
    initialize_lattice!(LATTICE_TYPE;
                        shift_x=LATTICE_SHIFT_X, shift_y=LATTICE_SHIFT_Y, shift_z=LATTICE_SHIFT_Z)
    global CS2_PHYSICS = CS2_LATTICE
    global CS4_PHYSICS = CS2_PHYSICS * CS2_PHYSICS

    c_wale = haskey(CFG["advanced"]["numerics"], "c_wale") ? Float32(CFG["advanced"]["numerics"]["c_wale"]) : C_SMAGO
    inlet_turb = INLET_TURBULENCE_INTENSITY
    nu_sgs_bg = NU_SGS_BACKGROUND
    use_temporal = TEMPORAL_INTERPOLATION
    sponge_blend = SPONGE_BLEND_DISTRIBUTIONS
    
    println("\n[Config] Stability Settings:")
    @printf("          Background Î½_sgs: %.6f â†’ Ï„_eff_min â‰ˆ %.4f\n", nu_sgs_bg, 0.5 + Float64(nu_sgs_bg)/Float64(CS2_PHYSICS))
    println("          Sponge f-blending: $sponge_blend")
    println("          Temporal interpolation: $use_temporal")
    
    CaseIDContract.verify_runtime_inputs(STARTUP_SELECTION.root_dir, runtime_selection)
    CaseIDContract.verify_claimed_output(runtime_selection, output_claim)
    # lbm-io-opts (27 Oct 2026): basic.simulation.build_only: domain build + summary, then stop
    # (before any backend selection, GPU call or time step). src/io/lbm_io_opts.jl.
    if LBMIO_BUILD_ONLY
        return lbmio_build_only_case(runtime_selection, output_claim, sim_start_datetime)
    end
    force_cleanup()

    gpu_device_name = CUDA.functional() ? string(CUDA.name(CUDA.device())) : "CPU"
    backend = CUDA.functional() ? (println("[Backend] CUDA: $(gpu_device_name)"); CUDABackend()) : (println("[Backend] CPU"); CPU())
    if CUDA.functional(); CUDA.allowscalar(false); end
    
    output_dir = output_claim.target.path
    CaseIDContract.verify_claimed_output(runtime_selection, output_claim)

    # Provenance: capture solver version, git hash, host, GPUs, and full
    # resolved config so an external reader can reproduce the run. Written
    # before any compute so even crashed runs have a provenance file.
    print_provenance_banner(stdout)
    write_provenance(output_dir, basename(CASE_DIR), CFG;
                     extra=Dict(
                         "lattice_type" => string(LATTICE_TYPE),
                         "collision_operator" => string(COLLISION_OPERATOR),
                         "compressible_mode" => COMPRESSIBLE_MODE,
                         "compressibility_correction" => COMPRESSIBILITY_CORRECTION,
                         "ddf_enabled" => DDF_ENABLED,
                         "hrr_hybrid_enabled" => HRR_HYBRID_ENABLED,
                         "hrr_sigma" => HRR_SIGMA,
                         "hrr_sigma_per_level" => HRR_SIGMA_PER_LEVEL,
                          "amr_interface_shell_sync" => AMR_INTERFACE_SHELL_SYNC,
                          "amr_shell_well_balanced_rebuild" => AMR_SHELL_WELL_BALANCED_REBUILD,
                          "amr_shell_sync_include_x" => AMR_SHELL_SYNC_INCLUDE_X,
                         "amr_shell_vel_normal_only" => AMR_SHELL_VEL_NORMAL_ONLY,
                         "amr_shell_flux_alpha_x" => AMR_SHELL_FLUX_ALPHA_X,
                         "amr_shell_flux_alpha_t" => AMR_SHELL_FLUX_ALPHA_T,
                         "amr_shell_flux_du_cap_abs" => AMR_SHELL_FLUX_DU_CAP_ABS,
                         "amr_shell_flux_du_cap_rel" => AMR_SHELL_FLUX_DU_CAP_REL,
                         "amr_shell_momflux_alpha_x" => AMR_SHELL_MOMFLUX_ALPHA_X,
                         "amr_shell_momflux_stress_alpha_x" => AMR_SHELL_MOMFLUX_STRESS_ALPHA_X,
                         "amr_shell_momflux_stress_cap_abs" => AMR_SHELL_MOMFLUX_STRESS_CAP_ABS,
                         "amr_shell_momflux_stress_cap_rel" => AMR_SHELL_MOMFLUX_STRESS_CAP_REL,
                         "amr_shell_highpass_damper_rho_alpha" => AMR_SHELL_HIGHPASS_DAMPER_RHO_ALPHA,
                         "amr_shell_highpass_damper_drho_cap_abs" => AMR_SHELL_HIGHPASS_DAMPER_DRHO_CAP_ABS,
                         "amr_shell_highpass_damper_drho_cap_rel" => AMR_SHELL_HIGHPASS_DAMPER_DRHO_CAP_REL,
                         "amr_halo_fullcell_recomposition" => AMR_HALO_FULLCELL_RECOMPOSITION,
                         "amr_halo_fullcell_macro_alpha" => AMR_HALO_FULLCELL_MACRO_ALPHA,
                         "amr_halo_fullcell_pop_alpha" => AMR_HALO_FULLCELL_POP_ALPHA,
                         "amr_prolongation_monotone_velocity" => AMR_PROLONGATION_MONOTONE_VELOCITY,
                         "amr_halo_temporal_macro_closure" => AMR_HALO_TEMPORAL_MACRO_CLOSURE,
                         "amr_halo_eq_projection" => AMR_HALO_EQ_PROJECTION,
                         "amr_halo_constrained_completion" => AMR_HALO_CONSTRAINED_COMPLETION,
                         "coarse_side_feather_scale" => COARSE_SIDE_FEATHER_SCALE,
                         "interface_temporal_weight_bias" => INTERFACE_TEMPORAL_WEIGHT_BIAS,
                         "interface_temporal_weight_bias_rise_u_start" => INTERFACE_TEMPORAL_WEIGHT_BIAS_RISE_U_START,
                         "interface_temporal_weight_bias_rise_u_end" => INTERFACE_TEMPORAL_WEIGHT_BIAS_RISE_U_END,
                         "interface_temporal_weight_bias_decay_u_start" => INTERFACE_TEMPORAL_WEIGHT_BIAS_DECAY_U_START,
                         "interface_temporal_weight_bias_decay_u_end" => INTERFACE_TEMPORAL_WEIGHT_BIAS_DECAY_U_END,
                         "amr_shell_eq_shift" => AMR_SHELL_EQ_SHIFT,
                         "amr_restriction_pi_neq_filter" => AMR_RESTRICTION_PI_NEQ_FILTER,
                         "sim_start" => string(sim_start_datetime),
                     ))

    function run_startup_stage(stage_fn::Function, label::String)
        log_walltime("$label START")
        _errlog("STARTUP ENTER $label")
        flush(stdout)
        flush(stderr)
        t_stage = time()
        result = stage_fn()
        dt_stage = time() - t_stage
        log_walltime("$label DONE in $(round(dt_stage, digits=2)) s")
        _errlog("STARTUP DONE $label dt=$(round(dt_stage, digits=2))")
        flush(stdout)
        flush(stderr)
        return result
    end

    run_startup_stage(label::String, stage_fn::Function) = run_startup_stage(stage_fn, label)

    # All per-run CSVs (convergence, forces, AMR diagnostics, resource history)
    # live in <output_dir>/csv/. ParaView VTK files live in <output_dir>/VTK/.
    # 2D slice PNGs live in <output_dir>/SLICES/. Keeps the RESULTS root clean.
    csv_root    = joinpath(output_dir, "csv")
    vtk_dir     = joinpath(output_dir, "VTK")
    slices_dir  = joinpath(output_dir, "SLICES")
    for d in (csv_root, vtk_dir, slices_dir)
        isdir(d) || mkpath(d)
    end

    csv_path = joinpath(csv_root, "convergence.csv")
    open(csv_path, "w") do io; println(io, "Step,Time_phys_s,Domain_travel_pct,u_inlet_lattice,Rho_min,MLUPS,Cd,Cl,Cs"); end

    force_csv = joinpath(csv_root, "forces.csv")
    if FORCE_COMPUTATION_ENABLED; write_force_csv_header(force_csv); end

    # Lightweight resource-usage history: one row per diagnostic tick with
    # iteration, absolute wall-clock time the row was written
    # (ISO-8601 `YYYY-MM-DD HH:MM:SS.fff` local time), elapsed seconds since
    # solver start, integer seconds since the previous row, integer iterations
    # per second over that row interval, VRAM used (MiB), and process RAM
    # working set (MiB).
    # 2026-05-28 user request — replaced the previous "seconds since start"
    # primary column with a real wall-clock timestamp; elapsed seconds kept
    # as a secondary column for back-compat / quick deltas.
    resource_csv = joinpath(csv_root, "resource_history.csv")
    open(resource_csv, "w") do io
        println(io, "Step,Wallclock_time,Elapsed_s,Delta_s,Iter_per_s,VRAM_used_MiB,VRAM_total_MiB,RAM_used_MiB,Total_elements")
    end

    amr_interface_csv = joinpath(csv_root, "amr_interface_history.csv")
    if AMR_INTERFACE_DIAGNOSTICS; write_amr_interface_csv_header(amr_interface_csv); end
    amr_highpass_csv = joinpath(csv_root, "amr_highpass_history.csv")
    if AMR_INTERFACE_DIAGNOSTICS; write_amr_highpass_csv_header(amr_highpass_csv); end
    # Pass 260 — per-block modal decomposition. Separates block-scale standing
    # mode from 1-cell Nyquist checkerboard to disambiguate the L3 interior
    # instability source.
    amr_block_modal_csv = joinpath(csv_root, "amr_block_modal_history.csv")
    if AMR_INTERFACE_DIAGNOSTICS; write_amr_block_modal_csv_header(amr_block_modal_csv); end
    # SYMMETRY-PLANE PROBE (2026-06-04): signed u_y residual at y=0 (the only
    # diagnostic that proves a mirror plane is free-slip). Header auto-written on
    # first append by print_symmetry_plane_residual.
    sym_plane_csv = joinpath(csv_root, "symmetry_plane_residual.csv")
    amr_face_register_csv = joinpath(csv_root, "amr_face_register_history.csv")
    if AMR_FACE_REGISTER_DIAGNOSTICS; write_amr_face_register_csv_header(amr_face_register_csv); end
    amr_population_stream_register_csv = joinpath(csv_root, "amr_population_stream_register_history.csv")
    if AMR_FACE_REGISTER_DIAGNOSTICS && AMR_POPULATION_STREAM_REGISTER
        write_amr_population_stream_register_csv_header(amr_population_stream_register_csv)
    end
    amr_population_stream_face_register_csv = joinpath(csv_root, "amr_population_stream_face_register_history.csv")
    if AMR_FACE_REGISTER_DIAGNOSTICS &&
       AMR_POPULATION_STREAM_REGISTER &&
       AMR_POPULATION_STREAM_REGISTER_FACE_RESOLVED
        write_amr_population_stream_face_register_csv_header(amr_population_stream_face_register_csv)
    end
    amr_runtime_face_budget_csv = joinpath(csv_root, "amr_runtime_face_budget_history.csv")
    if AMR_RUNTIME_FACE_BUDGET_DIAGNOSTICS; write_amr_runtime_face_budget_csv_header(amr_runtime_face_budget_csv); end

    # Pass 190 — Compressible audit diagnostic (read-only, default off).
    # Header written here so the file exists from step 0 even if the run
    # fails before the first diagnostic tick. Number of level columns is
    # NUM_LEVELS_CONFIG which is locked at this point.
    compressible_audit_csv = joinpath(csv_root, "compressible_audit_history.csv")
    if COMPRESSIBLE_AUDIT_DIAGNOSTICS
        write_compressible_audit_csv_header(compressible_audit_csv, NUM_LEVELS_CONFIG)
    end

    transonic_wake_csv = joinpath(csv_root, "transonic_wake_history.csv")
    if TRANSONIC_WAKE_DIAGNOSTICS
        write_transonic_wake_csv_header(transonic_wake_csv)
    end
    stability_extrema_csv = joinpath(csv_root, "stability_extrema_history.csv")
    if STABILITY_ABORT_ENABLED
        write_stability_extrema_csv_header(stability_extrema_csv)
    end
    reset_stability_abort_baseline!()
    
    CaseIDContract.verify_runtime_inputs(STARTUP_SELECTION.root_dir, runtime_selection)
    CaseIDContract.verify_claimed_output(runtime_selection, output_claim)
    cpu_grids, geometry_mesh, unrotated_mesh, unrotated_centers, object_names =
        run_startup_stage("Domain build") do
            setup_multilevel_domain(runtime_selection, output_claim;
                                    num_levels=NUM_LEVELS_CONFIG)
        end

    params = get_domain_params()

    # Export AMR block structure for standalone Babylon.js visualization.
    # Written BEFORE the GPU upload so the viewer is available even if the
    # simulation fails to reach step 1.
    try
        export_amr_viz(cpu_grids, params.mesh_offset, output_dir;
                       mesh=geometry_mesh, case_name=basename(CASE_DIR))
    catch e
        @printf("[AMR Viz] Warning: export failed: %s\n", e)
    end
    domain_nx, domain_ny, domain_nz = params.nx_coarse, params.ny_coarse, params.nz_coarse

    # â”€â”€ Compute STEPS from domain_length_covered if specified â”€â”€
    if DOMAIN_LENGTH_COVERED > 0.0
        domain_x_phys = params.domain_size[1]
        physical_time_needed = DOMAIN_LENGTH_COVERED * domain_x_phys / params.u_physical
        # lbm-taufix VA3 (W7): one solver step is one COARSEST step; solver_step_time = time_scale * 2^(L-1)
        # (= time_scale with one level, so single-level runs are unchanged).
        dt_step = solver_step_time(params)
        computed_steps = Int(ceil(physical_time_needed / dt_step))
        # Round up to nearest multiple of DIAG_FREQ for clean output
        computed_steps = Int(ceil(computed_steps / DIAG_FREQ)) * DIAG_FREQ
        global STEPS = computed_steps
        @printf("[Info] domain_length_covered = %.2f â†’ physical time = %.4f s â†’ STEPS = %d\n",
                DOMAIN_LENGTH_COVERED, physical_time_needed, STEPS)
    end
    if STEPS <= 0
        error("STEPS must be > 0. Set either 'steps' or 'domain_length_covered' in config.")
    end

    Ma_lattice = Float64(U_TARGET) / sqrt(Float64(CS2_PHYSICS))
    Ma_physical = params.u_physical / 343.0
    @printf("[Info] Re = %.0f, Ma_phys = %.4f, Ma_lat = %.4f\n", params.re_number, Ma_physical, Ma_lattice)
    println("[Info] Ï„_levels = $(join([@sprintf("%.6f", t) for t in params.tau_levels], ", "))")
    if COMPRESSIBILITY_CORRECTION
        println("[Info] Compressible solver active: O(uâ´) equilibrium + thermal DDF")
    end
    if DDF_ENABLED && COMPRESSIBLE_MODE && !COMPRESSIBILITY_CORRECTION
        println("[Info] *** THERMAL DDF auto-enabled for compressible mode (Phase 4: two-way coupling) ***")
    end
    
    # â”€â”€ Multi-GPU setup (Strategy C) â”€â”€
    local mgpu = nothing
    if MULTI_GPU_ENABLED && CUDA.functional() && CUDA.ndevices() >= 2
        log_walltime("Multi-GPU mode: setting up $(CUDA.ndevices()) GPUs (Strategy C)...")
        mgpu = run_startup_stage("Multi-GPU setup") do
            setup_multi_gpu(cpu_grids, GPU_SPLIT_LEVEL)
        end
        # Use GPU 0's grids as the primary reference (for diagnostics, forces, VTK)
        grids = mgpu.grids[1]
        cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu = mgpu.lattice_arrays[1]
        CUDA.device!(0)
        gpu_mesh = run_startup_stage("Mesh upload") do
            Geometry.upload_mesh_to_gpu(geometry_mesh, backend)
        end
        cpu_grids = nothing; GC.gc()
    else
        if MULTI_GPU_ENABLED
            println("[MultiGPU] WARNING: multi_gpu enabled but not enough GPUs ($(CUDA.functional() ? CUDA.ndevices() : 0)). Falling back to single-GPU.")
        end
        grids = run_startup_stage("Grid upload") do
            [adapt(backend, g) for g in cpu_grids]
        end
        gpu_mesh = run_startup_stage("Mesh upload") do
            Geometry.upload_mesh_to_gpu(geometry_mesh, backend)
        end
        cpu_grids = nothing; GC.gc()
        cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu =
            run_startup_stage("Lattice array upload") do
                build_lattice_arrays_gpu(backend)
            end
    end

    # Phase 4b: Thermal DDF now uses the SAME lattice as flow (Frapolli model)
    # No separate D3Q27 thermal arrays needed â€” g uses D3Q39 when flow is D3Q39
    if DDF_ENABLED && LATTICE_TYPE == :d3q39
        println("[Thermal] Frapolli model: g uses same D3Q39 lattice as flow ($(N_DIRS_THERMAL) directions)")
    elseif DDF_ENABLED
        println("[Thermal] D3Q27 advection-diffusion model ($(N_DIRS_THERMAL) directions)")
    end

    if mgpu !== nothing
        run_startup_stage("Equilibrium initialization") do
            # Multi-GPU: initialize equilibrium on both GPUs
            init_equilibrium_multi_gpu!(mgpu, params)
            nothing
        end
    else
        # Single-GPU: original initialization
        @kernel function init_eq!(f, f_temp, f_old, rho, vel, vel_temp, rho_old, vel_old, W,
                                  cx_arr, cy_arr, cz_arr,
                                  has_old::Int32, n_dirs::Int32,
                                  rho_init::Float32,
                                  u_init_x::Float32, u_init_y::Float32, u_init_z::Float32,
                                  cs2::Float32, cs4::Float32)
            x, y, z, b = @index(Global, NTuple)
            @inbounds begin
                usq = u_init_x*u_init_x + u_init_y*u_init_y + u_init_z*u_init_z
                inv_cs2 = 1.0f0 / cs2
                inv_2cs4 = 0.5f0 / cs4
                inv_2cs2 = 0.5f0 / cs2
                for k in 1:n_dirs
                    cx = Float32(cx_arr[k])
                    cy = Float32(cy_arr[k])
                    cz = Float32(cz_arr[k])
                    cu = cx*u_init_x + cy*u_init_y + cz*u_init_z
                    feq = rho_init * W[k] * (1.0f0 + inv_cs2*cu + inv_2cs4*cu*cu - inv_2cs2*usq)
                    f[x, y, z, b, k] = feq
                    f_temp[x, y, z, b, k] = feq
                end
                rho[x, y, z, b] = rho_init
                vel[x, y, z, b, 1] = u_init_x
                vel[x, y, z, b, 2] = u_init_y
                vel[x, y, z, b, 3] = u_init_z
                vel_temp[x, y, z, b, 1] = u_init_x
                vel_temp[x, y, z, b, 2] = u_init_y
                vel_temp[x, y, z, b, 3] = u_init_z
                if has_old == Int32(1)
                    for k in 1:n_dirs
                        cx = Float32(cx_arr[k])
                        cy = Float32(cy_arr[k])
                        cz = Float32(cz_arr[k])
                        cu = cx*u_init_x + cy*u_init_y + cz*u_init_z
                        f_old[x, y, z, b, k] = rho_init * W[k] *
                            (1.0f0 + inv_cs2*cu + inv_2cs4*cu*cu - inv_2cs2*usq)
                    end
                    rho_old[x, y, z, b] = rho_init
                    vel_old[x, y, z, b, 1] = u_init_x
                    vel_old[x, y, z, b, 2] = u_init_y
                    vel_old[x, y, z, b, 3] = u_init_z
                end
            end
        end

        # Pass 192a — Riemann initial condition along X. Each cell is
        # assigned the "left" or "right" state by comparing its global
        # lattice-units X index against `split_idx_x`. Populations are
        # initialized to the 2nd-order polynomial equilibrium of the
        # selected state (the first collision will refine to the proper
        # higher-order / Frapolli form). For Sod where u=0 on both sides
        # this reduces to f_k = ρ · w_k, which is exact for either lattice.
        @kernel function init_eq_riemann_x!(f, f_temp, f_old, rho, vel, vel_temp, rho_old, vel_old,
                                            cx_arr, cy_arr, cz_arr, W, map_x,
                                            has_old::Int32, n_dirs::Int32,
                                            split_idx_x::Int32, block_size::Int32,
                                            rho_L::Float32, ux_L::Float32, uy_L::Float32, uz_L::Float32, T_L::Float32,
                                            rho_R::Float32, ux_R::Float32, uy_R::Float32, uz_R::Float32, T_R::Float32,
                                            cs2::Float32, cs4::Float32, eos_R_ic::Float32)
            x, y, z, b = @index(Global, NTuple)
            @inbounds begin
                gx = (map_x[b] - Int32(1)) * block_size + Int32(x)
                left = gx <= split_idx_x
                rho_l = left ? rho_L : rho_R
                ux_l  = left ? ux_L  : ux_R
                uy_l  = left ? uy_L  : uy_R
                uz_l  = left ? uz_L  : uz_R
                T_l   = left ? T_L   : T_R
                usq = ux_l*ux_l + uy_l*uy_l + uz_l*uz_l
                inv_cs2 = 1.0f0 / cs2
                inv_2cs4 = 0.5f0 / cs4
                inv_2cs2 = 0.5f0 / cs2
                # AUDIT FIX Bug B (2026-06-03): the 2nd-order polynomial eq encodes
                # T=1 in its second moment (Σf|c|²=ρ·D·cs²=2ρ at u=0). Add the
                # closed-form Hermite-2 (trace) correction g_T·(|c|²−D·cs²) so the
                # SECOND moment encodes the per-side T:  Σf|c|² = ρ|u|² + 2ρT — the
                # kinetic split the Frapolli closure T=(E_kin−ρ|u|²+2Σg)/(2ρCv)
                # requires. g_T = 0.75·(T−1) is the exact D3Q39 solution (S2=2,
                # S4=20/3). Mass & momentum untouched (Σw(|c|²−D·cs²)=0; term even
                # in c). Without it the Sod right state recovers T≈0.96 not 0.8.
                # AUDIT FIX (2026-06-03): eos_R-consistent Hermite-2 amplitude.
                # (9/8)·eos_R·(T−1): for D3Q39 (S2=2, S4=20/3) this makes the f
                # second-moment contribution 3ρ·eos_R·(T−1), so E_kin matches the
                # eos_R-convention closure. eos_R_ic=cs² ⇒ 0.75(T−1) (legacy, aero
                # byte-identical).
                g_T = 1.125f0 * eos_R_ic * (T_l - 1.0f0)
                D_cs2 = 3.0f0 * cs2                 # = 2 for D3Q39
                for k in 1:n_dirs
                    cx = Float32(cx_arr[k])
                    cy = Float32(cy_arr[k])
                    cz = Float32(cz_arr[k])
                    cu = cx*ux_l + cy*uy_l + cz*uz_l
                    csq = cx*cx + cy*cy + cz*cz
                    poly = 1.0f0 + inv_cs2*cu + inv_2cs4*cu*cu - inv_2cs2*usq
                    feq = max(rho_l * W[k] * (poly + g_T * (csq - D_cs2)), 1.0f-10)
                    f[x, y, z, b, k]      = feq
                    f_temp[x, y, z, b, k] = feq
                end
                rho[x, y, z, b]       = rho_l
                vel[x, y, z, b, 1]    = ux_l
                vel[x, y, z, b, 2]    = uy_l
                vel[x, y, z, b, 3]    = uz_l
                vel_temp[x, y, z, b, 1] = ux_l
                vel_temp[x, y, z, b, 2] = uy_l
                vel_temp[x, y, z, b, 3] = uz_l
                if has_old == Int32(1)
                    for k in 1:n_dirs
                        cx = Float32(cx_arr[k])
                        cy = Float32(cy_arr[k])
                        cz = Float32(cz_arr[k])
                        cu = cx*ux_l + cy*uy_l + cz*uz_l
                        csq = cx*cx + cy*cy + cz*cz
                        poly = 1.0f0 + inv_cs2*cu + inv_2cs4*cu*cu - inv_2cs2*usq
                        f_old[x, y, z, b, k] = max(rho_l * W[k] * (poly + g_T * (csq - D_cs2)), 1.0f-10)
                    end
                    rho_old[x, y, z, b]    = rho_l
                    vel_old[x, y, z, b, 1] = ux_l
                    vel_old[x, y, z, b, 2] = uy_l
                    vel_old[x, y, z, b, 3] = uz_l
                end
            end
        end

        @kernel function init_eq_from_feq!(f, f_temp, f_old, rho, vel, vel_temp, rho_old, vel_old, f_eq_arr,
                                           has_old::Int32, n_dirs::Int32,
                                           rho_init::Float32,
                                           u_init_x::Float32, u_init_y::Float32, u_init_z::Float32)
            x, y, z, b = @index(Global, NTuple)
            @inbounds begin
                for k in 1:n_dirs
                    feq = f_eq_arr[k]
                    f[x, y, z, b, k] = feq
                    f_temp[x, y, z, b, k] = feq
                    if has_old == Int32(1)
                        f_old[x, y, z, b, k] = feq
                    end
                end
                rho[x, y, z, b] = rho_init
                vel[x, y, z, b, 1] = u_init_x
                vel[x, y, z, b, 2] = u_init_y
                vel[x, y, z, b, 3] = u_init_z
                vel_temp[x, y, z, b, 1] = u_init_x
                vel_temp[x, y, z, b, 2] = u_init_y
                vel_temp[x, y, z, b, 3] = u_init_z
                if has_old == Int32(1)
                    rho_old[x, y, z, b] = rho_init
                    vel_old[x, y, z, b, 1] = u_init_x
                    vel_old[x, y, z, b, 2] = u_init_y
                    vel_old[x, y, z, b, 3] = u_init_z
                end
            end
        end

        @kernel function init_eq_gaussian_axis!(f, f_temp, f_old, rho, vel, vel_temp, rho_old, vel_old,
                                             cx_arr, cy_arr, cz_arr, W, map_x, map_y, map_z,
                                             has_old::Int32, n_dirs::Int32,
                                             block_size::Int32, dx::Float32,
                                             offset_x::Float32, offset_y::Float32, offset_z::Float32,
                                             rho_base::Float32, ux_base::Float32, uy_base::Float32, uz_base::Float32,
                                             center_x::Float32, center_y::Float32, center_z::Float32,
                                             sigma_x::Float32, sigma_y::Float32, sigma_z::Float32,
                                             axis::Int32, ellipsoidal::Int32,
                                             rho_amp::Float32, ux_amp::Float32, uy_amp::Float32, uz_amp::Float32,
                                             cs2::Float32, cs4::Float32,
                                             period_y::Float32)   # periodic-Y AMR: >0 ⇒ minimum-image Y distance
            x, y, z, b = @index(Global, NTuple)
            @inbounds begin
                gx = (map_x[b] - Int32(1)) * block_size + Int32(x)
                gy = (map_y[b] - Int32(1)) * block_size + Int32(y)
                gz = (map_z[b] - Int32(1)) * block_size + Int32(z)
                x_phys = offset_x + (Float32(gx) - 0.5f0) * dx
                y_phys = offset_y + (Float32(gy) - 0.5f0) * dx
                z_phys = offset_z + (Float32(gz) - 0.5f0) * dx
                y_rel = y_phys - center_y
                if period_y > 0.0f0
                    y_rel -= period_y * round(y_rel / period_y)
                end
                coord = axis == Int32(2) ? y_phys : (axis == Int32(3) ? z_phys : x_phys)
                center = axis == Int32(2) ? center_y : (axis == Int32(3) ? center_z : center_x)
                sigma = axis == Int32(2) ? sigma_y : (axis == Int32(3) ? sigma_z : sigma_x)
                xi = (axis == Int32(2) ? y_rel : (coord - center)) / max(sigma, 1.0f-12)
                xi_x = (x_phys - center_x) / max(sigma_x, 1.0f-12)
                xi_y = y_rel / max(sigma_y, 1.0f-12)
                xi_z = (z_phys - center_z) / max(sigma_z, 1.0f-12)
                r2 = ellipsoidal == Int32(1) ? (xi_x*xi_x + xi_y*xi_y + xi_z*xi_z) : (xi * xi)
                g = exp(-0.5f0 * r2)
                rho_l = max(rho_base + rho_amp * g, 0.01f0)
                ux_l = ux_base + ux_amp * g
                uy_l = uy_base + uy_amp * g
                uz_l = uz_base + uz_amp * g
                usq = ux_l*ux_l + uy_l*uy_l + uz_l*uz_l
                inv_cs2 = 1.0f0 / cs2
                inv_2cs4 = 0.5f0 / cs4
                inv_2cs2 = 0.5f0 / cs2
                for k in 1:n_dirs
                    cx = Float32(cx_arr[k])
                    cy = Float32(cy_arr[k])
                    cz = Float32(cz_arr[k])
                    cu = cx*ux_l + cy*uy_l + cz*uz_l
                    feq = rho_l * W[k] * (1.0f0 + inv_cs2*cu + inv_2cs4*cu*cu - inv_2cs2*usq)
                    f[x, y, z, b, k] = feq
                    f_temp[x, y, z, b, k] = feq
                end
                rho[x, y, z, b] = rho_l
                vel[x, y, z, b, 1] = ux_l
                vel[x, y, z, b, 2] = uy_l
                vel[x, y, z, b, 3] = uz_l
                vel_temp[x, y, z, b, 1] = ux_l
                vel_temp[x, y, z, b, 2] = uy_l
                vel_temp[x, y, z, b, 3] = uz_l
                if has_old == Int32(1)
                    for k in 1:n_dirs
                        cx = Float32(cx_arr[k])
                        cy = Float32(cy_arr[k])
                        cz = Float32(cz_arr[k])
                        cu = cx*ux_l + cy*uy_l + cz*uz_l
                        f_old[x, y, z, b, k] = rho_l * W[k] *
                            (1.0f0 + inv_cs2*cu + inv_2cs4*cu*cu - inv_2cs2*usq)
                    end
                    rho_old[x, y, z, b] = rho_l
                    vel_old[x, y, z, b, 1] = ux_l
                    vel_old[x, y, z, b, 2] = uy_l
                    vel_old[x, y, z, b, 3] = uz_l
                end
            end
        end

        # ── Spiegel isentropic vortex IC (Gate 2 — AMR validation) ──
        # Initializes ρ, u, T from the exact Spiegel/isentropic vortex
        # solution superimposed on a uniform freestream:
        #   r²   = ((x - x_c)/R)² + ((y - y_c)/R)²
        #   δu_x = -((y - y_c)/R) · A · exp((1 - r²)/2)
        #   δu_y =  ((x - x_c)/R) · A · exp((1 - r²)/2)
        #   δT   = -(γ-1)·A²/2 · exp(1 - r²)
        #   T    = T_∞ + δT
        #   ρ    = ρ_∞ · (T/T_∞)^(1/(γ-1))   (isentropic)
        # u_lab = freestream + δu. Outside the support of the gaussian
        # decay (r ≫ 1), the IC reduces to the uniform freestream.
        # Populations are set to the 2nd-order polynomial equilibrium of
        # (ρ, u) — the first entropic collision refines to the proper
        # 13-moment numerical equilibrium.
        @kernel function init_eq_spiegel_vortex!(f, f_temp, f_old, rho, vel, vel_temp, rho_old, vel_old,
                                                  cx_arr, cy_arr, cz_arr, W, map_x, map_y, map_z,
                                                  has_old::Int32, n_dirs::Int32,
                                                  block_size::Int32, dx::Float32,
                                                  offset_x::Float32, offset_y::Float32, offset_z::Float32,
                                                  rho_inf::Float32, T_inf::Float32, gamma::Float32,
                                                  u_inf_x::Float32, u_inf_y::Float32, u_inf_z::Float32,
                                                  center_x::Float32, center_y::Float32, center_z::Float32,
                                                  radius::Float32, amplitude::Float32,
                                                  cs2::Float32, cs4::Float32,
                                                  period_y::Float32)   # periodic-Y AMR: >0 ⇒ minimum-image Y distance
            x, y, z, b = @index(Global, NTuple)
            @inbounds begin
                gx = (map_x[b] - Int32(1)) * block_size + Int32(x)
                gy = (map_y[b] - Int32(1)) * block_size + Int32(y)
                gz = (map_z[b] - Int32(1)) * block_size + Int32(z)
                x_phys = offset_x + (Float32(gx) - 0.5f0) * dx
                y_phys = offset_y + (Float32(gy) - 0.5f0) * dx
                z_phys = offset_z + (Float32(gz) - 0.5f0) * dx
                inv_R = 1.0f0 / max(radius, 1.0f-12)
                xi = (x_phys - center_x) * inv_R
                y_rel = y_phys - center_y
                if period_y > 0.0f0
                    y_rel -= period_y * round(y_rel / period_y)
                end
                eta = y_rel * inv_R
                r2 = xi*xi + eta*eta
                env = exp(0.5f0 * (1.0f0 - r2))      # exp((1-r²)/2)
                env2 = exp(1.0f0 - r2)               # exp(1 - r²)
                du_x = -eta * amplitude * env
                du_y =  xi  * amplitude * env
                dT   = -(gamma - 1.0f0) * 0.5f0 * amplitude * amplitude * env2
                T_l  = T_inf + dT
                T_l  = max(T_l, 0.01f0)
                # Isentropic: ρ/ρ_∞ = (T/T_∞)^(1/(γ-1))
                rho_l = rho_inf * (T_l / max(T_inf, 1.0f-6))^(1.0f0 / (gamma - 1.0f0))
                rho_l = max(rho_l, 0.01f0)
                ux_l = u_inf_x + du_x
                uy_l = u_inf_y + du_y
                uz_l = u_inf_z
                usq = ux_l*ux_l + uy_l*uy_l + uz_l*uz_l
                inv_cs2 = 1.0f0 / cs2
                inv_2cs4 = 0.5f0 / cs4
                inv_2cs2 = 0.5f0 / cs2
                for k in 1:n_dirs
                    cx = Float32(cx_arr[k])
                    cy = Float32(cy_arr[k])
                    cz = Float32(cz_arr[k])
                    cu = cx*ux_l + cy*uy_l + cz*uz_l
                    feq = rho_l * W[k] * (1.0f0 + inv_cs2*cu + inv_2cs4*cu*cu - inv_2cs2*usq)
                    f[x, y, z, b, k] = feq
                    f_temp[x, y, z, b, k] = feq
                end
                rho[x, y, z, b] = rho_l
                vel[x, y, z, b, 1] = ux_l
                vel[x, y, z, b, 2] = uy_l
                vel[x, y, z, b, 3] = uz_l
                vel_temp[x, y, z, b, 1] = ux_l
                vel_temp[x, y, z, b, 2] = uy_l
                vel_temp[x, y, z, b, 3] = uz_l
                if has_old == Int32(1)
                    for k in 1:n_dirs
                        cx = Float32(cx_arr[k])
                        cy = Float32(cy_arr[k])
                        cz = Float32(cz_arr[k])
                        cu = cx*ux_l + cy*uy_l + cz*uz_l
                        f_old[x, y, z, b, k] = rho_l * W[k] *
                            (1.0f0 + inv_cs2*cu + inv_2cs4*cu*cu - inv_2cs2*usq)
                    end
                    rho_old[x, y, z, b] = rho_l
                    vel_old[x, y, z, b, 1] = ux_l
                    vel_old[x, y, z, b, 2] = uy_l
                    vel_old[x, y, z, b, 3] = uz_l
                end
            end
        end

        @kernel function init_thermal_eq!(g, g_temp, g_old, g_post_collision,
                                           temperature, temperature_old, kinetic_energy,
                                           W, T_init::Float32, has_old::Int32, has_bouzidi::Int32,
                                           n_dirs_g::Int32, coeff_g::Float32)
            x, y, z, b = @index(Global, NTuple)
            @inbounds begin
                for k in 1:n_dirs_g
                    g_eq = coeff_g * T_init * W[k]
                    g[x, y, z, b, k] = g_eq
                    g_temp[x, y, z, b, k] = g_eq
                end
                temperature[x, y, z, b] = T_init
                temperature_old[x, y, z, b] = T_init
                kinetic_energy[x, y, z, b] = 0.0f0
                if has_old == Int32(1)
                    for k in 1:n_dirs_g; g_old[x, y, z, b, k] = coeff_g * T_init * W[k]; end
                end
                if has_bouzidi == Int32(1)
                    for k in 1:n_dirs_g; g_post_collision[x, y, z, b, k] = coeff_g * T_init * W[k]; end
                end
            end
        end

        # Pass 192a — Riemann initial condition for the thermal DDF.
        # Splits on the same X index as init_eq_riemann_x! so f and g are
        # consistent. kinetic_energy is initialized to ρ·|u|² (consistent
        # with the flow's f_eq second moment for the chosen u state), so
        # the first Frapolli T-recovery on the next step is well-posed
        # without a transient.
        @kernel function init_thermal_eq_riemann_x!(g, g_temp, g_old, g_post_collision,
                                                     temperature, temperature_old, kinetic_energy,
                                                     W, map_x,
                                                     has_old::Int32, has_bouzidi::Int32,
                                                     n_dirs_g::Int32, coeff_g::Float32,
                                                     split_idx_x::Int32, block_size::Int32,
                                                     T_L::Float32, T_R::Float32,
                                                     rho_L::Float32, ux_L::Float32, uy_L::Float32, uz_L::Float32,
                                                     rho_R::Float32, ux_R::Float32, uy_R::Float32, uz_R::Float32,
                                                     cs2_ic::Float32, eos_R_ic::Float32)
            x, y, z, b = @index(Global, NTuple)
            @inbounds begin
                gx = (map_x[b] - Int32(1)) * block_size + Int32(x)
                left = gx <= split_idx_x
                T_l   = left ? T_L   : T_R
                rho_l = left ? rho_L : rho_R
                ux_l  = left ? ux_L  : ux_R
                uy_l  = left ? uy_L  : uy_R
                uz_l  = left ? uz_L  : uz_R
                # AUDIT FIX (2026-06-03): Frapolli g_eq = coeff_g·T·f_eq_k, and
                # Σf_eq_k = ρ, so Σg = coeff_g·T·ρ. The previous IC dropped the ρ
                # factor (g_eq = coeff_g·T·W[k]), which is masked when ρ=1 but is
                # 8× too large for the Sod right state (ρ=0.125) — making the
                # Frapolli T-recovery return ≈1.0 instead of 0.8. Multiply by ρ.
                for k in 1:n_dirs_g
                    g_eq = coeff_g * T_l * rho_l * W[k]
                    g[x, y, z, b, k]      = g_eq
                    g_temp[x, y, z, b, k] = g_eq
                end
                temperature[x, y, z, b]     = T_l
                temperature_old[x, y, z, b] = T_l
                # E_kin = Σf_k|c_k|² = ρ|u|² + 2ρT at equilibrium (D3Q39, the
                # closure subtracts ρ|u|² then divides by 2ρCv). Seed the full
                # second moment so step-1 T-recovery is consistent with T_l.
                # AUDIT FIX (2026-06-03): eos_R-consistent E_kin seed. The Frapolli
                # closure expects E_kin = ρ|u|² + D·ρ·cs² + D·ρ·eos_R·(T−1) (D=3).
                # eos_R_ic=cs² ⇒ ρ|u|² + 2ρT (legacy, aero byte-identical).
                kinetic_energy[x, y, z, b]  = rho_l * (ux_l*ux_l + uy_l*uy_l + uz_l*uz_l) +
                                              3.0f0 * rho_l * cs2_ic +
                                              3.0f0 * rho_l * eos_R_ic * (T_l - 1.0f0)
                if has_old == Int32(1)
                    for k in 1:n_dirs_g; g_old[x, y, z, b, k] = coeff_g * T_l * rho_l * W[k]; end
                end
                if has_bouzidi == Int32(1)
                    for k in 1:n_dirs_g; g_post_collision[x, y, z, b, k] = coeff_g * T_l * rho_l * W[k]; end
                end
            end
        end

        @kernel function init_thermal_eq_gaussian_axis!(g, g_temp, g_old, g_post_collision,
                                                      temperature, temperature_old, kinetic_energy,
                                                      W, map_x, map_y, map_z,
                                                      has_old::Int32, has_bouzidi::Int32,
                                                      n_dirs_g::Int32, coeff_g::Float32,
                                                      block_size::Int32, dx::Float32,
                                                      offset_x::Float32, offset_y::Float32, offset_z::Float32,
                                                      T_base::Float32,
                                                      center_x::Float32, center_y::Float32, center_z::Float32,
                                                      sigma_x::Float32, sigma_y::Float32, sigma_z::Float32,
                                                      axis::Int32, ellipsoidal::Int32,
                                                      T_amp::Float32, rho_base::Float32,
                                                      rho_amp::Float32,
                                                      ux_base::Float32, ux_amp::Float32,
                                                      uy_base::Float32, uy_amp::Float32,
                                                      uz_base::Float32, uz_amp::Float32,
                                                      period_y::Float32)   # periodic-Y AMR: >0 ⇒ minimum-image Y distance
            x, y, z, b = @index(Global, NTuple)
            @inbounds begin
                gx = (map_x[b] - Int32(1)) * block_size + Int32(x)
                gy = (map_y[b] - Int32(1)) * block_size + Int32(y)
                gz = (map_z[b] - Int32(1)) * block_size + Int32(z)
                x_phys = offset_x + (Float32(gx) - 0.5f0) * dx
                y_phys = offset_y + (Float32(gy) - 0.5f0) * dx
                z_phys = offset_z + (Float32(gz) - 0.5f0) * dx
                y_rel = y_phys - center_y
                if period_y > 0.0f0
                    y_rel -= period_y * round(y_rel / period_y)
                end
                coord = axis == Int32(2) ? y_phys : (axis == Int32(3) ? z_phys : x_phys)
                center = axis == Int32(2) ? center_y : (axis == Int32(3) ? center_z : center_x)
                sigma = axis == Int32(2) ? sigma_y : (axis == Int32(3) ? sigma_z : sigma_x)
                xi = (axis == Int32(2) ? y_rel : (coord - center)) / max(sigma, 1.0f-12)
                xi_x = (x_phys - center_x) / max(sigma_x, 1.0f-12)
                xi_y = y_rel / max(sigma_y, 1.0f-12)
                xi_z = (z_phys - center_z) / max(sigma_z, 1.0f-12)
                r2 = ellipsoidal == Int32(1) ? (xi_x*xi_x + xi_y*xi_y + xi_z*xi_z) : (xi * xi)
                pulse = exp(-0.5f0 * r2)
                T_l = max(T_base + T_amp * pulse, 0.01f0)
                rho_l = max(rho_base + rho_amp * pulse, 0.01f0)
                ux_l = ux_base + ux_amp * pulse
                uy_l = uy_base + uy_amp * pulse
                uz_l = uz_base + uz_amp * pulse
                for k in 1:n_dirs_g
                    g_eq = coeff_g * T_l * W[k]
                    g[x, y, z, b, k] = g_eq
                    g_temp[x, y, z, b, k] = g_eq
                end
                temperature[x, y, z, b] = T_l
                temperature_old[x, y, z, b] = T_l
                kinetic_energy[x, y, z, b] = rho_l * (ux_l*ux_l + uy_l*uy_l + uz_l*uz_l)
                if has_old == Int32(1)
                    for k in 1:n_dirs_g; g_old[x, y, z, b, k] = coeff_g * T_l * W[k]; end
                end
                if has_bouzidi == Int32(1)
                    for k in 1:n_dirs_g; g_post_collision[x, y, z, b, k] = coeff_g * T_l * W[k]; end
                end
            end
        end  # init_thermal_eq_gaussian_axis!

        # Thermal IC for the Spiegel isentropic vortex — sets the per-cell
        # temperature/g-field to the SAME isentropic δT the flow IC imposes
        # (init_eq_spiegel_vortex!). Without this the vortex δT was discarded and
        # the thermal field initialized to uniform T, making the vortex IC
        # thermodynamically inconsistent (rho varied but T did not).
        @kernel function init_thermal_eq_spiegel_vortex!(g, g_temp, g_old, g_post_collision,
                                                          temperature, temperature_old, kinetic_energy,
                                                          W, map_x, map_y, map_z,
                                                          has_old::Int32, has_bouzidi::Int32,
                                                          n_dirs_g::Int32, coeff_g::Float32,
                                                          block_size::Int32, dx::Float32,
                                                          offset_x::Float32, offset_y::Float32, offset_z::Float32,
                                                          rho_inf::Float32, T_inf::Float32, gamma::Float32,
                                                          u_inf_x::Float32, u_inf_y::Float32, u_inf_z::Float32,
                                                          center_x::Float32, center_y::Float32, center_z::Float32,
                                                          radius::Float32, amplitude::Float32,
                                                          period_y::Float32)   # periodic-Y AMR: >0 ⇒ minimum-image Y distance
            x, y, z, b = @index(Global, NTuple)
            @inbounds begin
                gx = (map_x[b] - Int32(1)) * block_size + Int32(x)
                gy = (map_y[b] - Int32(1)) * block_size + Int32(y)
                x_phys = offset_x + (Float32(gx) - 0.5f0) * dx
                y_phys = offset_y + (Float32(gy) - 0.5f0) * dx
                inv_R = 1.0f0 / max(radius, 1.0f-12)
                xi = (x_phys - center_x) * inv_R
                y_rel = y_phys - center_y
                if period_y > 0.0f0
                    y_rel -= period_y * round(y_rel / period_y)
                end
                eta = y_rel * inv_R
                r2 = xi*xi + eta*eta
                env = exp(0.5f0 * (1.0f0 - r2))
                env2 = exp(1.0f0 - r2)
                du_x = -eta * amplitude * env
                du_y =  xi  * amplitude * env
                dT   = -(gamma - 1.0f0) * 0.5f0 * amplitude * amplitude * env2
                T_l  = max(T_inf + dT, 0.01f0)
                rho_l = max(rho_inf * (T_l / max(T_inf, 1.0f-6))^(1.0f0 / (gamma - 1.0f0)), 0.01f0)
                ux_l = u_inf_x + du_x
                uy_l = u_inf_y + du_y
                uz_l = u_inf_z
                for k in 1:n_dirs_g
                    g_eq = coeff_g * T_l * W[k]
                    g[x, y, z, b, k] = g_eq
                    g_temp[x, y, z, b, k] = g_eq
                end
                temperature[x, y, z, b] = T_l
                temperature_old[x, y, z, b] = T_l
                kinetic_energy[x, y, z, b] = rho_l * (ux_l*ux_l + uy_l*uy_l + uz_l*uz_l)
                if has_old == Int32(1)
                    for k in 1:n_dirs_g; g_old[x, y, z, b, k] = coeff_g * T_l * W[k]; end
                end
                if has_bouzidi == Int32(1)
                    for k in 1:n_dirs_g; g_post_collision[x, y, z, b, k] = coeff_g * T_l * W[k]; end
                end
            end
        end  # init_thermal_eq_spiegel_vortex!

        run_startup_stage("Equilibrium initialization") do
            # Pass 192a — Riemann IC dispatch. :uniform (default) keeps the
            # legacy single-state initialization; :riemann_x splits along X
            # at IC_RIEMANN_SPLIT_X (physical meters) and assigns left/right
            # states from config.
            is_riemann = (INITIAL_CONDITION_TYPE === :riemann_x)
            is_gaussian = INITIAL_CONDITION_TYPE in (:gaussian_x, :gaussian_y, :gaussian_z)
            is_spiegel  = (INITIAL_CONDITION_TYPE === :spiegel_vortex)
            use_numerical_uniform_d3q39 = (INITIAL_CONDITION_TYPE === :uniform) && (LATTICE_TYPE == :d3q39)
            f_eq_uniform_gpu = nothing
            if use_numerical_uniform_d3q39
                cx_s, cy_s, cz_s, w_s, _, _ = build_d3q39_static()
                f_eq_uniform_cpu = MVector{39, Float32}(undef)
                ic_shift_x = LOCAL_LATTICE_SHIFT_ENABLED ? 0.0 : Float64(LATTICE_SHIFT_X)
                ic_shift_y = LOCAL_LATTICE_SHIFT_ENABLED ? 0.0 : Float64(LATTICE_SHIFT_Y)
                ic_shift_z = LOCAL_LATTICE_SHIFT_ENABLED ? 0.0 : Float64(LATTICE_SHIFT_Z)
                calculate_equilibrium_numerical_d3q39!(
                    f_eq_uniform_cpu,
                    Float32(IC_UNIFORM_RHO),
                    Float32(IC_UNIFORM_UX - ic_shift_x),
                    Float32(IC_UNIFORM_UY - ic_shift_y),
                    Float32(IC_UNIFORM_UZ - ic_shift_z),
                    cx_s, cy_s, cz_s, w_s, CS2_PHYSICS, Int32(200),
                    DDF_ENABLED ? Float32(DDF_T_INLET) : 1.0f0
                )
                mass_correct_feq39!(f_eq_uniform_cpu, Float32(IC_UNIFORM_RHO))
                for k in 1:39
                    f_eq_uniform_cpu[k] = max(f_eq_uniform_cpu[k], Float32(1e-10))
                end
                f_eq_uniform_gpu = adapt(backend, Vector{Float32}(f_eq_uniform_cpu))
                @printf("[Init]    D3Q39 numerical uniform equilibrium: rho=%.6f u=(%.6f,%.6f,%.6f)\n",
                        IC_UNIFORM_RHO, IC_UNIFORM_UX, IC_UNIFORM_UY, IC_UNIFORM_UZ)
            end
            mesh_offset_x = Float64(params.mesh_offset[1])
            mesh_offset_y = Float64(params.mesh_offset[2])
            mesh_offset_z = Float64(params.mesh_offset[3])
            # Periodic-Y AMR (27 Oct 2026): localized ICs (gaussian / spiegel) use the
            # minimum-image Y distance, so a pulse centred on the seam is continuous
            # across it. 0 ⇒ legacy (non-periodic) distance, bitwise unchanged.
            ic_period_y = periodic_y_amr_active() ? Float32(params.domain_size[2]) : 0.0f0
            for lvl in 1:length(grids)
                level = grids[lvl]
                n = length(level.active_block_coords)
                if n > 0
                    has_old = has_temporal_storage(level) ? Int32(1) : Int32(0)
                    # Per-level split index in lattice units. Cell with global
                    # lattice index gx has center at (gx-0.5)·dx + offset, so
                    # the left set is gx <= floor((split_x_phys - offset)/dx + 0.5).
                    split_cont = (IC_RIEMANN_SPLIT_X - mesh_offset_x) / level.dx + 0.5
                    split_idx_x = Int32(floor(split_cont))
                    # Macro arrays store lab-frame velocity. For shifted D3Q39,
                    # the distribution equilibrium above is built in the relative
                    # frame u_rel = u_lab - U_shift so physical rest stays rest.
                    u_init_x = Float32(IC_UNIFORM_UX)
                    u_init_y = Float32(IC_UNIFORM_UY)
                    u_init_z = Float32(IC_UNIFORM_UZ)
                    # AUDIT FIX (2026-06-03): eos-R-consistent IC energy seeding.
                    # The IC seeds flow E_kin, thermal Σg, and the flow-f Hermite-2
                    # amplitude on the lattice-cs² convention, but the (already-fixed)
                    # Frapolli closure uses eos_R=(γ-1)cv when eos_thermodynamic is on.
                    # The closure is a NEUTRAL map (preserves seeded energy), so the
                    # convention mismatch bakes excess energy into a raised steady T
                    # (Sod right state 0.8→1.0). Seed all three on the SAME eos_R as
                    # the closure. When eos_thermodynamic=false (aero default),
                    # eos_R_ic=cs² → byte-identical to the prior seeding.
                    eos_R_ic = EOS_THERMODYNAMIC ? Float32((DDF_GAMMA - 1.0) * DDF_CV) : Float32(CS2_PHYSICS)
                    if is_riemann
                        init_eq_riemann_x!(backend)(
                            level.f, level.f_temp, level.f_old, level.rho, level.vel, level.vel_temp,
                            level.rho_old, level.vel_old,
                            cx_gpu, cy_gpu, cz_gpu, w_gpu, level.map_x,
                            has_old, N_DIRS,
                            split_idx_x, Int32(BLOCK_SIZE),
                            Float32(IC_RIEMANN_LEFT_RHO),  Float32(IC_RIEMANN_LEFT_UX),
                            Float32(IC_RIEMANN_LEFT_UY),   Float32(IC_RIEMANN_LEFT_UZ),  Float32(IC_RIEMANN_LEFT_T),
                            Float32(IC_RIEMANN_RIGHT_RHO), Float32(IC_RIEMANN_RIGHT_UX),
                            Float32(IC_RIEMANN_RIGHT_UY),  Float32(IC_RIEMANN_RIGHT_UZ), Float32(IC_RIEMANN_RIGHT_T),
                            Float32(CS2_PHYSICS), Float32(CS4_PHYSICS), eos_R_ic,
                            ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n))
                    elseif is_gaussian
                        init_eq_gaussian_axis!(backend)(
                            level.f, level.f_temp, level.f_old, level.rho, level.vel, level.vel_temp,
                            level.rho_old, level.vel_old,
                            cx_gpu, cy_gpu, cz_gpu, w_gpu, level.map_x, level.map_y, level.map_z,
                            has_old, N_DIRS,
                            Int32(BLOCK_SIZE), Float32(level.dx),
                            0.0f0, 0.0f0, 0.0f0,
                            Float32(IC_UNIFORM_RHO),
                            u_init_x, u_init_y, u_init_z,
                            Float32(IC_GAUSSIAN_CENTER_X), Float32(IC_GAUSSIAN_CENTER_Y), Float32(IC_GAUSSIAN_CENTER_Z),
                            Float32(IC_GAUSSIAN_SIGMA_X), Float32(IC_GAUSSIAN_SIGMA_Y), Float32(IC_GAUSSIAN_SIGMA_Z),
                            Int32(IC_GAUSSIAN_AXIS), IC_GAUSSIAN_ELLIPSOIDAL ? Int32(1) : Int32(0),
                            Float32(IC_GAUSSIAN_RHO_AMPLITUDE),
                            Float32(IC_GAUSSIAN_UX_AMPLITUDE), Float32(IC_GAUSSIAN_UY_AMPLITUDE), Float32(IC_GAUSSIAN_UZ_AMPLITUDE),
                            Float32(CS2_PHYSICS), Float32(CS4_PHYSICS),
                            ic_period_y,
                            ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n))
                    elseif is_spiegel
                        init_eq_spiegel_vortex!(backend)(
                            level.f, level.f_temp, level.f_old, level.rho, level.vel, level.vel_temp,
                            level.rho_old, level.vel_old,
                            cx_gpu, cy_gpu, cz_gpu, w_gpu, level.map_x, level.map_y, level.map_z,
                            has_old, N_DIRS,
                            Int32(BLOCK_SIZE), Float32(level.dx),
                            0.0f0, 0.0f0, 0.0f0,
                            Float32(IC_UNIFORM_RHO), Float32(IC_UNIFORM_T), Float32(DDF_GAMMA),
                            u_init_x, u_init_y, u_init_z,
                            Float32(IC_SPIEGEL_CENTER_X), Float32(IC_SPIEGEL_CENTER_Y), Float32(IC_SPIEGEL_CENTER_Z),
                            Float32(IC_SPIEGEL_RADIUS), Float32(IC_SPIEGEL_AMPLITUDE),
                            Float32(CS2_PHYSICS), Float32(CS4_PHYSICS),
                            ic_period_y,
                            ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n))
                    else
                        if use_numerical_uniform_d3q39
                            init_eq_from_feq!(backend)(level.f, level.f_temp, level.f_old,
                                                       level.rho, level.vel, level.vel_temp,
                                                       level.rho_old, level.vel_old,
                                                       f_eq_uniform_gpu,
                                                       has_old, N_DIRS, Float32(IC_UNIFORM_RHO),
                                                       u_init_x, u_init_y, u_init_z,
                                                       ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n))
                        else
                            init_eq!(backend)(level.f, level.f_temp, level.f_old,
                                              level.rho, level.vel, level.vel_temp,
                                              level.rho_old, level.vel_old, w_gpu,
                                              cx_gpu, cy_gpu, cz_gpu,
                                              has_old, N_DIRS, Float32(IC_UNIFORM_RHO),
                                              u_init_x, u_init_y, u_init_z,
                                              Float32(CS2_PHYSICS), Float32(CS4_PHYSICS),
                                              ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n))
                        end
                    end

                    if DDF_ENABLED && has_thermal(level)
                        level.tau_g = params.tau_g_levels[lvl]
                        has_old_g = (length(level.g_old) > N_DIRS_THERMAL) ? Int32(1) : Int32(0)
                        has_bouz_g = (level.bouzidi_enabled && level.n_boundary_cells > 0) ? Int32(1) : Int32(0)
                        coeff_g_init = (LATTICE_TYPE == :d3q39) ? Float32(DDF_CV - 1.5 * Float64(eos_R_ic)) : 1.0f0   # AUDIT 2026-06-03: eos_R-consistent g seed
                        if is_riemann
                            init_thermal_eq_riemann_x!(backend)(
                                level.g, level.g_temp, level.g_old, level.g_post_collision,
                                level.temperature, level.temperature_old, level.kinetic_energy,
                                w_gpu, level.map_x,
                                has_old_g, has_bouz_g,
                                N_DIRS_THERMAL, coeff_g_init,
                                split_idx_x, Int32(BLOCK_SIZE),
                                Float32(IC_RIEMANN_LEFT_T), Float32(IC_RIEMANN_RIGHT_T),
                                Float32(IC_RIEMANN_LEFT_RHO),  Float32(IC_RIEMANN_LEFT_UX),
                                Float32(IC_RIEMANN_LEFT_UY),   Float32(IC_RIEMANN_LEFT_UZ),
                                Float32(IC_RIEMANN_RIGHT_RHO), Float32(IC_RIEMANN_RIGHT_UX),
                                Float32(IC_RIEMANN_RIGHT_UY),  Float32(IC_RIEMANN_RIGHT_UZ),
                                Float32(CS2_PHYSICS), eos_R_ic,
                                ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n))
                        elseif is_gaussian
                            init_thermal_eq_gaussian_axis!(backend)(
                                level.g, level.g_temp, level.g_old, level.g_post_collision,
                                level.temperature, level.temperature_old, level.kinetic_energy,
                                w_gpu, level.map_x, level.map_y, level.map_z,
                                has_old_g, has_bouz_g,
                                N_DIRS_THERMAL, coeff_g_init,
                                Int32(BLOCK_SIZE), Float32(level.dx),
                                0.0f0, 0.0f0, 0.0f0,
                                Float32(IC_UNIFORM_T),
                                Float32(IC_GAUSSIAN_CENTER_X), Float32(IC_GAUSSIAN_CENTER_Y), Float32(IC_GAUSSIAN_CENTER_Z),
                                Float32(IC_GAUSSIAN_SIGMA_X), Float32(IC_GAUSSIAN_SIGMA_Y), Float32(IC_GAUSSIAN_SIGMA_Z),
                                Int32(IC_GAUSSIAN_AXIS), IC_GAUSSIAN_ELLIPSOIDAL ? Int32(1) : Int32(0),
                                Float32(IC_GAUSSIAN_T_AMPLITUDE),
                                Float32(IC_UNIFORM_RHO),
                                Float32(IC_GAUSSIAN_RHO_AMPLITUDE),
                                u_init_x, Float32(IC_GAUSSIAN_UX_AMPLITUDE),
                                u_init_y, Float32(IC_GAUSSIAN_UY_AMPLITUDE),
                                u_init_z, Float32(IC_GAUSSIAN_UZ_AMPLITUDE),
                                ic_period_y,
                                ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n))
                        elseif is_spiegel
                            init_thermal_eq_spiegel_vortex!(backend)(
                                level.g, level.g_temp, level.g_old, level.g_post_collision,
                                level.temperature, level.temperature_old, level.kinetic_energy,
                                w_gpu, level.map_x, level.map_y, level.map_z,
                                has_old_g, has_bouz_g,
                                N_DIRS_THERMAL, coeff_g_init,
                                Int32(BLOCK_SIZE), Float32(level.dx),
                                0.0f0, 0.0f0, 0.0f0,
                                Float32(IC_UNIFORM_RHO), Float32(IC_UNIFORM_T), Float32(DDF_GAMMA),
                                u_init_x, u_init_y, u_init_z,
                                Float32(IC_SPIEGEL_CENTER_X), Float32(IC_SPIEGEL_CENTER_Y), Float32(IC_SPIEGEL_CENTER_Z),
                                Float32(IC_SPIEGEL_RADIUS), Float32(IC_SPIEGEL_AMPLITUDE),
                                ic_period_y,
                                ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n))
                        else
                            init_thermal_eq!(backend)(level.g, level.g_temp, level.g_old, level.g_post_collision,
                                                      level.temperature, level.temperature_old, level.kinetic_energy,
                                                      w_gpu, Float32(IC_UNIFORM_T), has_old_g, has_bouz_g,
                                                      N_DIRS_THERMAL, coeff_g_init,
                                                      ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n))
                        end
                    end
                end
            end
            KernelAbstractions.synchronize(backend)
            # (ET/AA-pattern one-shot encode removed 2026-06-24 with the D3Q19 retirement.)
            nothing
        end
    end

    total_cells = sum(length(g.active_block_coords) * BLOCK_SIZE^3 for g in grids)
    println("[Info] Total cells: $(round(total_cells/1e6, digits=2)) M")

    print_vram_breakdown(grids)
    
    force_data = nothing
    force_accm = nothing
    if FORCE_COMPUTATION_ENABLED
        force_data = ForceData(gpu_mesh.n_triangles, backend; 
                               rho_ref=params.rho_physical, 
                               u_ref=params.u_physical,
                               area_ref=params.reference_area, 
                               chord_ref=params.reference_chord,
                               moment_center=params.moment_center, 
                               force_scale=params.force_scale,
                               length_scale=params.length_scale, 
                               symmetric=SYMMETRIC_ANALYSIS)
        
        force_accm = ForceAccumulator(force_data)
        
        println("[Forces] Initialized surface stress method for $(gpu_mesh.n_triangles) triangles")
        println("          Reference: Ï=$(round(params.rho_physical, digits=3)) kg/mÂ³, U=$(round(params.u_physical, digits=2)) m/s")
        println("          Reference: A=$(round(params.reference_area, digits=4)) mÂ², L=$(round(params.reference_chord, digits=4)) m")
        flush(stdout)
        flush(stderr)
    end
    
    domain_length_x = params.domain_size[1]

    # Use finest level for force probing â€” it has the best block coverage near the geometry.
    force_level = grids[end]
    force_search_radius = max(8, ceil(Int, 0.12 / force_level.dx))
    println("[Forces] Probing on level $(length(grids)) (dx=$(round(force_level.dx, digits=6)) m), search_radius = $(force_search_radius) cells ($(round(force_search_radius * force_level.dx, digits=4)) m physical)")
    flush(stdout)
    flush(stderr)

    crossed_frequency(step_end::Int, batch_size::Int, freq::Int) =
        freq > 0 && (step_end % freq < batch_size || step_end == STEPS)

    function run_export_stage(stage_fn::Function, label::String, export_step::Int)
        stage_start = time()
        println("[$label] START step $export_step")
        flush(stdout)
        flush(stderr)
        try
            stage_fn()
            stage_dt = time() - stage_start
            println("[$label] DONE step $export_step in $(round(stage_dt, digits=2)) s")
            flush(stdout)
            flush(stderr)
            return true
        catch e
            stage_dt = time() - stage_start
            println("[$label] Warning: stage failed at step $export_step after $(round(stage_dt, digits=2)) s: $e")
            flush(stdout)
            flush(stderr)
            return false
        end
    end

    run_export_stage(label::String, export_step::Int, stage_fn::Function) =
        run_export_stage(stage_fn, label, export_step)

    function export_surface_outputs!(export_step::Int; recompute_if_needed::Bool=true)
        if force_data === nothing
            return
        end

        if recompute_if_needed && (export_step % DIAG_FREQ != 0)
            compute_aerodynamics!(force_data, force_level, gpu_mesh, backend, params;
                                  search_radius=force_search_radius,
                                  cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu)
            if export_step >= RAMP_STEPS
                accumulate_forces!(force_accm, force_data)
            end
        end

        surf_filename = @sprintf("%s/surface_%06d", vtk_dir, export_step)
        save_surface_vtk(surf_filename, force_data, geometry_mesh, params;
                         object_names=object_names)

        surf_bin_filename = @sprintf("%s/surface_%06d.lbmp", vtk_dir, export_step)
        export_surface_pressure_bin(surf_bin_filename, force_data, unrotated_mesh, params;
                                    object_names=object_names,
                                    use_domain_offset=false)

        if NET_FORCE_EXPORT_ENABLED
            net_Fx, net_Fy, net_Fz = compute_net_aerodynamics!(
                force_data, force_level, gpu_mesh, backend, params;
                search_radius=force_search_radius,
                cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu
            )
            export_net_triangle_forces_csv(
                @sprintf("%s/net_forces_%06d.csv", csv_root, export_step),
                net_Fx, net_Fy, net_Fz,
                gpu_mesh, params.mesh_offset;
                unrotated_centers=unrotated_centers,
                alpha_deg=ALPHA_DEG,
                beta_deg=BETA_DEG,
                ground_mode=GROUND_ANALYSIS,
                ground_up_axis=GROUND_UP_AXIS,
                object_names=object_names
            )
            export_net_triangle_forces_vtk(
                @sprintf("%s/net_forces_%06d", vtk_dir, export_step),
                net_Fx, net_Fy, net_Fz,
                gpu_mesh, params.mesh_offset
            )
        end
    end

    if LOCAL_LATTICE_SHIFT_ENABLED
        for level in grids
            initialize_local_lattice_shift!(level, backend, level.vel;
                                            u_frame_ref=Float32(IC_UNIFORM_UX))
        end
    end

    # One-shot export of geometric fields (AMR owner level + slice sampling mode).
    # These don't change across timesteps, so we dump them here and skip them
    # during the time loop to save per-iteration I/O.
    if OUTPUT_STARTUP_SLICES && LBMIO_SLICE_CUSTOM
        # lbm-io-opts: the geometric set on the selected planes only (nothing when png=false)
        lbmio_sync_png_slices!("Startup", 0, grids, slices_dir, backend, :geometric, nothing;
                               cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu)
    elseif OUTPUT_STARTUP_SLICES
        run_export_stage("StartupLevelSlice", 0) do
            export_slice_image(0, grids, slices_dir, backend; field=:geometric,
                               cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu)
        end
        run_export_stage("StartupLevelSliceY025", 0) do
            export_slice_image(0, grids, slices_dir, backend; field=:geometric,
                               cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu,
                               y_override=0.25, fname_suffix="_Y0p25")
        end
        run_export_stage("StartupLevelSliceYZ", 0) do
            export_slice_image(0, grids, slices_dir, backend; field=:geometric,
                               cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu,
                               plane_override=:yz, fname_suffix="_YZ")
        end
    else
        println("[Startup] Skipping geometric startup slices (output_startup_slices=false)")
    end
    if OUTPUT_STARTUP_SLICES && LBMIO_EXPORT_DATA   # lbm-io-opts: step-0 numeric plane(s)
        run_export_stage("StartupSliceData", 0) do
            lbmio_export_planes!(0, 0, grids, slices_dir)
        end
    end
    if OUTPUT_STARTUP_VOLUME
        run_export_stage("StartupLevelVolume", 0) do
            export_merged_mesh_sync(0, grids, vtk_dir, backend;
                                    cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu,
                                    include_level=true)
        end
    else
        println("[Startup] Skipping geometric startup volume (output_startup_volume=false)")
    end

    log_walltime("LBM Analysis STARTED")
    println()
    @printf("%8s | %10s | %7s | %7s | %7s | %17s | %8s | %8s | %8s\n",
            "Step", "Time[s]", "%Domain", "u_in", "Ï_min", "MLUPS", "Cd", "Cl", "Cs")
    println(repeat("-", 112))
    flush(stdout)
    flush(stderr)

    t_start_sim = time()
    last_diag = time()
    batch = GPU_ASYNC_DEPTH
    resource_prev_step = Ref{Int}(0)
    resource_prev_wall_s = Ref{Float64}(t_start_sim - SIMULATION_START_TIME[])

    # C4 perf (2026-05-28): compute-only MLUPS accumulator. The legacy
    # `mlups = total_cells * DIAG_FREQ / (now_t - last_diag)` formula counts
    # ALL wall time between diag ticks — including slice exports, force
    # computation, AMR diagnostics, and surface IO. With slice_output_freq
    # at 20, slice IO dominates and gives a misleading ~0.1 MLUPS reading
    # vs the true compute throughput of ~30 MLUPS. `compute_dt_accumulator`
    # collects ONLY the time spent in `execute_timestep_batch!`, so the
    # compute-only metric reflects the actual kernel throughput.
    compute_dt_accumulator = 0.0
    last_diag_compute_dt = 0.0

    t = 1
    while t <= STEPS
        batch_end = min(t + batch - 1, STEPS)
        actual = batch_end - t + 1
        # Diagnostic: log every batch if BATCH_DIAG env var set, else first 3 batches + every 10th
        verbose_diag = get(ENV, "LUDWIG_BATCH_DIAG", "0") == "1"
        batch_logging = (STEPS <= 32 || t == 1 || verbose_diag || t <= 24 || (t % 80 == 1))
        # C4 perf: always time the batch, regardless of `batch_logging`.
        # The compute-only MLUPS metric needs a complete time series.
        batch_timer = time()
        # Heartbeat BEFORE compute (always, every batch) → if a hard crash
        # (segfault/SIGABRT/OOM) hits during the kernel, the last error-log line
        # is "ENTER batch t:batch_end", pinpointing the failing step with no
        # Julia exception. (2026-05-29, user request.)
        _heartbeat(t, "ENTER-batch->$batch_end")
        if batch_logging
            log_walltime("Timestep batch START (steps $t:$batch_end)")
            flush(stdout)
            flush(stderr)
        end

        prog_raw = ramp_progress(batch_end, RAMP_STEPS, RAMP_PROFILE)
        ramp_floor = clamp(RAMP_START_FRACTION, 0.0f0, 1.0f0)
        prog = ramp_floor + (1.0f0 - ramp_floor) * prog_raw
        # u_curr is the LAB-FRAME inlet velocity. The shifted D3Q39 lattice
        # changes the discrete velocities used inside the equilibrium/streaming
        # operators; it must not add a physical offset to the inlet ramp.
        u_curr = U_TARGET * prog
        U_INLET_CURRENT[] = u_curr

        if mgpu !== nothing
            execute_timestep_batch_multigpu!(mgpu, t, actual, u_curr,
                                             domain_nx, domain_ny, domain_nz,
                                             params.wall_model_active, c_wale, nu_sgs_bg,
                                             inlet_turb, use_temporal, sponge_blend)
        else
            execute_timestep_batch!(grids, t, actual, u_curr,
                                    cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                                    domain_nx, domain_ny, domain_nz,
                                    params.wall_model_active, c_wale, nu_sgs_bg,
                                    inlet_turb, use_temporal, sponge_blend)
        end
        _heartbeat(batch_end, "DONE-batch")  # compute survived this batch
        if mgpu === nothing
            materialize_current_state_for_io!(grids, batch_end)
            # Opt-in raw per-level state dump (default off; 27 Oct 2026).
            if RAW_STATE_DUMP_FREQ > 0 && (batch_end % RAW_STATE_DUMP_FREQ == 0 || batch_end == STEPS)
                dump_raw_state(grids, output_dir, batch_end)
            end
        end
        # Per-batch mirror probe (LUDWIG_MIRROR_PROBE_FREQ=N): catch the FIRST
        # parity-violating step (use gpu async_depth: 1 for per-step batches).
        if _MIRROR_PROBE_FREQ > 0 && mgpu === nothing && batch_end % _MIRROR_PROBE_FREQ == 0
            KernelAbstractions.synchronize(backend)
            dump_z_mirror_residual(grids, domain_nz, batch_end)
        end

        if mgpu === nothing
            KernelAbstractions.synchronize(backend)
        end
        batch_dt = time() - batch_timer
        compute_dt_accumulator += batch_dt
        if batch_logging
            log_walltime("Timestep batch DONE (steps $t:$batch_end) in $(round(batch_dt, digits=2)) s")
            flush(stdout)
            flush(stderr)
        end
        if _PHASE_TIMING
            _tot = _T_FLOW[] + _T_THERMAL[]
            @printf("    [PhaseTiming] flow=%.2fs thermal=%.2fs (%d flow calls) | flow=%.0f%% thermal=%.0f%% of (flow+thermal) | batch_dt=%.2fs ⇒ AMR/BC/other=%.0f%%\n",
                    _T_FLOW[], _T_THERMAL[], _T_FLOW_CALLS[],
                    _tot>0 ? 100*_T_FLOW[]/_tot : 0, _tot>0 ? 100*_T_THERMAL[]/_tot : 0,
                    batch_dt, batch_dt>0 ? 100*(batch_dt-_tot)/batch_dt : 0)
            _phase_print_split_timers!()
            @printf("    [OpTiming] restrict=%.2fs spline_prefilter=%.2fs pineq_refresh=%.2fs copy_to_old=%.2fs (per-substep GPU ops, synced)\n",
                    _T_OP_RESTRICT[], _T_OP_SPLINE[], _T_OP_PINEQ[], _T_OP_COPYOLD[])
            flush(stdout)
            _T_FLOW[] = 0.0; _T_THERMAL[] = 0.0; _T_BC[] = 0.0; _T_FLOW_CALLS[] = 0
            _T_OP_RESTRICT[] = 0.0; _T_OP_SPLINE[] = 0.0; _T_OP_BOUZIDI[] = 0.0
            _T_OP_PINEQ[] = 0.0; _T_OP_COPYOLD[] = 0.0
            _phase_reset_split_timers!()
        end
        
        # Resource sampler: every ~5 simulated steps, independent of DIAG_FREQ.
        # Emits at most one row per completed batch, using the latest crossed
        # 5-step boundary. That keeps Delta_s/Iter_per_s tied to real wall time
        # between appended rows rather than multiple synthetic rows stamped at
        # the same batch-end instant.
        let res_freq = 5
            first_mark = ((t - 1) ÷ res_freq + 1) * res_freq
            mark = (batch_end ÷ res_freq) * res_freq
            if mark >= first_mark && mark > resource_prev_step[]
                try
                    wall_now = now()                              # local clock instant
                    wall_s = time() - SIMULATION_START_TIME[]      # elapsed seconds since start
                    delta_s_raw = max(0.0, wall_s - resource_prev_wall_s[])
                    delta_iter = max(0, mark - resource_prev_step[])
                    delta_s_int = Int(round(delta_s_raw))
                    iter_per_s = delta_s_raw > 0.0 ? Int(round(delta_iter / delta_s_raw)) : 0
                    # ISO-8601-ish local timestamp with millisecond precision:
                    #   YYYY-MM-DD HH:MM:SS.fff
                    wall_ts = Dates.format(wall_now, "yyyy-mm-dd HH:MM:SS.sss")
                    vram_used_mib = 0.0
                    vram_total_mib = 0.0
                    if CUDA.functional()
                        free_b, total_b = CUDA.memory_info()
                        vram_used_mib = (total_b - free_b) / 1024^2
                        vram_total_mib = total_b / 1024^2
                    end
                    ram_used_mib = try
                        Float64(Sys.maxrss()) / 1024^2
                    catch
                        NaN
                    end
                    open(resource_csv, "a") do io
                        @printf(io, "%d,%s,%.3f,%d,%d,%.1f,%.1f,%.1f,%d\n",
                                mark, wall_ts, wall_s, delta_s_int, iter_per_s,
                                vram_used_mib, vram_total_mib, ram_used_mib, total_cells)
                    end
                    resource_prev_step[] = mark
                    resource_prev_wall_s[] = wall_s
                catch
                    # Never let resource bookkeeping kill the run.
                end
            end
        end

        if crossed_frequency(batch_end, actual, DIAG_FREQ)
            diag_step = (batch_end ÷ DIAG_FREQ) * DIAG_FREQ
            if diag_step >= t && diag_step <= batch_end
                _heartbeat(diag_step, "ENTER-diagnostics")
                _heartbeat(diag_step, "ENTER-flow-stats-L1")
                stats = compute_flow_stats(grids[1])
                _heartbeat(diag_step, "DONE-flow-stats-L1")

                # Per-level diagnostics: show u_max and u_mean for each level.
                # u_max catches halo/boundary cells; u_mean reveals the interior bulk.
                # If u_max is high but u_mean is low → flow is at boundary only, interior dead.
                # If both are high → flow fully populated in level.
                print("          Per-level: ")
                for lvl_idx in 1:length(grids)
                    try
                        _heartbeat(diag_step, "ENTER-level-stats-L$lvl_idx")
                        lvl_stats = compute_flow_stats(grids[lvl_idx])
                        @printf("L%d[u_mean=%.4f rho_mean=%.4f T_mean=%.4f Ekin=%.4f gsum=%.4f rho_max=%.3f]  ",
                                lvl_idx, lvl_stats.u_mean, Float64(lvl_stats.rho_mean), Float64(lvl_stats.T_mean),
                                Float64(lvl_stats.ekin_lattice_mean), Float64(lvl_stats.gsum_mean), Float64(lvl_stats.rho_max))
                        _heartbeat(diag_step, "DONE-level-stats-L$lvl_idx")
                    catch e
                        @printf("L%d[stats failed]  ", lvl_idx)
                        _heartbeat(diag_step, "FAILED-level-stats-L$lvl_idx")
                    end
                end
                println()
                if AMR_INTERFACE_DIAGNOSTICS
                    try
                        full_amr_diag = should_emit_full_amr_diagnostics(batch_end)
                        print_amr_interface_stats(grids; csv_path=amr_interface_csv, step=batch_end, full=full_amr_diag)
                        print_amr_highpass_stats(grids; csv_path=amr_highpass_csv, step=batch_end)
                        # Pass 260: block-modal decomposition (block-standing vs cell-Nyquist).
                        print_amr_block_modal_stats(grids; csv_path=amr_block_modal_csv, step=batch_end)
                    catch e
                        println("          [AMR] diagnostics failed: $e")
                    end
                end
                # SYMMETRY-PLANE PROBE (2026-06-04): prove the y=0 mirror plane is
                # free-slip via the signed normal velocity u_y — the ONLY direct test
                # (rho/|u| diagnostics are blind to it). Fires only in half-domain
                # mirror mode; cheap (one GPU→CPU vel pullback at diag cadence).
                if SYMMETRIC_ANALYSIS
                    try
                        print_symmetry_plane_residual(grids, u_curr; step=batch_end, csv_path=sym_plane_csv)
                    catch e
                        println("          [SymPlane] probe failed: $e")
                    end
                end
                if AMR_FACE_REGISTER_DIAGNOSTICS
                    try
                        write_amr_face_register(grids; csv_path=amr_face_register_csv, step=batch_end)
                        if AMR_POPULATION_STREAM_REGISTER
                            write_amr_population_stream_register(
                                grids,
                                cx_gpu,
                                cy_gpu,
                                cz_gpu;
                                csv_path=amr_population_stream_register_csv,
                                step=batch_end,
                            )
                            if AMR_POPULATION_STREAM_REGISTER_FACE_RESOLVED
                                write_amr_population_stream_face_register(
                                    grids,
                                    cx_gpu,
                                    cy_gpu,
                                    cz_gpu;
                                    csv_path=amr_population_stream_face_register_csv,
                                    step=batch_end,
                                )
                            end
                        end
                    catch e
                        println("          [AMR-FACE] register failed: $e")
                    end
                end
                if AMR_RUNTIME_FACE_BUDGET_DIAGNOSTICS
                    try
                        write_amr_runtime_face_budget(csv_path=amr_runtime_face_budget_csv, step=batch_end)
                    catch e
                        println("          [AMR-BUDGET] runtime face budget failed: $e")
                    end
                end
                if MASS_CONSERVATION_DIAGNOSTICS
                    _heartbeat(diag_step, "ENTER-mass-diagnostics")
                    try
                        print_mass_conservation(grids, batch_end)
                    catch e
                        println("          [Mass] diagnostic failed: $e")
                    end
                    _heartbeat(diag_step, "DONE-mass-diagnostics")
                end
                if COMPRESSIBLE_AUDIT_DIAGNOSTICS
                    try
                        print_compressible_audit(grids, batch_end;
                                                  csv_path=compressible_audit_csv)
                    catch e
                        println("          [CompAudit] diagnostic failed: $e")
                    end
                end
                wake_stats = nothing
                if TRANSONIC_WAKE_DIAGNOSTICS
                    _heartbeat(diag_step, "ENTER-transonic-wake")
                    try
                        wake_stats = print_transonic_wake_diagnostics(grids, batch_end, u_curr;
                                                                      csv_path=transonic_wake_csv)
                    catch e
                        println("          [Wake] diagnostic failed: $e")
                    end
                    _heartbeat(diag_step, "DONE-transonic-wake")
                end
                if STABILITY_ABORT_ENABLED
                    try
                        print_stability_extrema_diagnostics(grids, batch_end;
                                                           csv_path=stability_extrema_csv)
                    catch e
                        println("          [Extrema] diagnostic failed: $e")
                    end
                end
                flush(stdout)
                _heartbeat(diag_step, "DONE-diagnostic-copies")

                now_t = time()
                # C4 perf (2026-05-28): MLUPS now reports TRUE compute-only
                # throughput (kernel time only, excluding slice/force/AMR-diag
                # IO). The legacy wall-time-inclusive metric was dominated by
                # slice IO when `slice_output_freq` was small (~0.1 MLUPS with
                # slice_freq=20 vs ~30 MLUPS true compute on RTX 4060 D3Q39
                # entropic). For the wall-clock-inclusive "throughput including
                # IO" metric, compare to `mlups_inclusive`.
                compute_dt_this_diag = compute_dt_accumulator - last_diag_compute_dt
                last_diag_compute_dt = compute_dt_accumulator
                mlups = compute_dt_this_diag > 1.0e-6 ?
                        (total_cells * DIAG_FREQ) / (compute_dt_this_diag * 1e6) :
                        0.0
                mlups_inclusive = (total_cells * DIAG_FREQ) / ((now_t - last_diag) * 1e6)
                last_diag = now_t
                # lbm-taufix VA3 (W7): physical time per SOLVER step (one coarsest step = 2^(L-1) finest steps);
                # forces.csv, convergence.csv and the console Time / %Domain columns all take it from here.
                time_phys = Float64(diag_step) * solver_step_time(params)

                cd_str, cl_str, cs_str = "N/A", "N/A", "N/A"
                if force_data !== nothing
                    compute_aerodynamics!(force_data, force_level, gpu_mesh, backend, params;
                                          search_radius=force_search_radius,
                                          cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu)

                    accumulate_forces!(force_accm, force_data)
                    # Running per-triangle and per-scalar means (Tier A-3).
                    # Only post-ramp so the means reflect steady-state physics.
                    if batch_end >= RAMP_STEPS
                        accumulate_force_averages!(force_data)
                    end

                    cd_str = @sprintf("%.4f", force_data.Cd)
                    cl_str = @sprintf("%.4f", force_data.Cl)
                    cs_str = @sprintf("%.4f", force_data.Cs)
                    append_force_csv(force_csv, diag_step, time_phys, force_data, u_curr)
                end

                domain_travel_pct = domain_length_x > 0 ? 100.0 * (time_phys * params.u_physical) / domain_length_x : 0.0

                @printf("%8d | %10.4f | %6.1f%% | %.4f | %.4f | %6.1f (incl %5.1f) | %8s | %8s | %8s\n",
                        diag_step, time_phys, domain_travel_pct, u_curr, stats.rho_min, mlups, mlups_inclusive, cd_str, cl_str, cs_str)
                # FREESTREAM-COLLAPSE PROBE (2026-06-27): unambiguous bulk indicators (lattice units).
                # rho_min above is dominated by a 1-cell outlet artifact; rho_mean/v_max show the BULK.
                @printf("          [PROBE] L1 rho_mean=%.5f rho_max=%.5f v_max=%.5f (u_inlet=%.5f)\n",
                        Float64(stats.rho_mean), Float64(stats.rho_max), Float64(stats.v_max), u_curr)
                open(csv_path, "a") do io
                    @printf(io, "%d,%.6e,%.6e,%.4f,%.6f,%.1f,%s,%s,%s\n",
                            diag_step, time_phys, domain_travel_pct, u_curr, stats.rho_min, mlups, cd_str, cl_str, cs_str)
                end
                flush(stdout)
                flush(stderr)

                abort_eval = evaluate_stability_abort(grids, diag_step; wake_stats=wake_stats)
                if abort_eval.abort
                    abort_msg = @sprintf("[StabilityAbort] step=%d mass=%.9e mass_rel=%+.6e :: %s",
                                         diag_step, abort_eval.mass, abort_eval.mass_rel, abort_eval.message)
                    println(abort_msg)
                    _errlog(abort_msg)
                    flush(stdout)
                    flush(stderr)
                    error(abort_msg)
                end
                _heartbeat(diag_step, "DONE-diagnostics")
            end
        end
        
        if false && batch_end % OUTPUT_FREQ < actual
            out_step = (batch_end ÷ OUTPUT_FREQ) * OUTPUT_FREQ
            if out_step >= t && out_step <= batch_end
                export_merged_mesh_sync(out_step, grids, output_dir, backend;
                                         cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu)

                # â”€â”€ 2D slice export (Mach, density, velocity PNG) â”€â”€
                try
                    export_slice_image(out_step, grids, output_dir, backend; field=:all,
                                       cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu)
                catch e
                    println("[Slice] Warning: slice export failed: $e")
                end
                try
                    export_thermal_interface_diagnostics(out_step, grids, output_dir)
                catch e
                    println("[Diag] Warning: thermal interface diagnostics failed: $e")
                end

                if force_data !== nothing && out_step > RAMP_STEPS
                    if out_step != ((out_step ÷ DIAG_FREQ) * DIAG_FREQ)
                        compute_aerodynamics!(force_data, force_level, gpu_mesh, backend, params;
                                              search_radius=force_search_radius,
                                              cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu)

                        accumulate_forces!(force_accm, force_data)
                    # Running per-triangle and per-scalar means (Tier A-3).
                    # Only post-ramp so the means reflect steady-state physics.
                    if batch_end >= RAMP_STEPS
                        accumulate_force_averages!(force_data)
                    end
                    end
                    
                    surf_filename = @sprintf("%s/surface_%06d", vtk_dir, out_step)
                    save_surface_vtk(surf_filename, force_data, geometry_mesh, params;
                                     object_names=object_names)

                    # â”€â”€ Binary surface pressure export (for web viewer) â”€â”€
                    # Uses unrotated mesh (STL frame, no domain offset) so that
                    # pressures and net force vectors share the same coordinate system.
                    surf_bin_filename = @sprintf("%s/surface_%06d.lbmp", vtk_dir, out_step)
                    export_surface_pressure_bin(surf_bin_filename, force_data, unrotated_mesh, params;
                                                object_names=object_names,
                                                use_domain_offset=false)

                    # â”€â”€ Net surface force export (two-sided) â”€â”€
                    if NET_FORCE_EXPORT_ENABLED
                        net_Fx, net_Fy, net_Fz = compute_net_aerodynamics!(
                            force_data, force_level, gpu_mesh, backend, params;
                            search_radius=force_search_radius,
                            cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu
                        )
                        export_net_triangle_forces_csv(
                            @sprintf("%s/net_forces_%06d.csv", csv_root, out_step),
                            net_Fx, net_Fy, net_Fz,
                            gpu_mesh, params.mesh_offset;
                            unrotated_centers=unrotated_centers,
                            alpha_deg=ALPHA_DEG,
                            beta_deg=BETA_DEG,
                            ground_mode=GROUND_ANALYSIS,
                            ground_up_axis=GROUND_UP_AXIS,
                            object_names=object_names
                        )
                        export_net_triangle_forces_vtk(
                            @sprintf("%s/net_forces_%06d", vtk_dir, out_step),
                            net_Fx, net_Fy, net_Fz,
                            gpu_mesh, params.mesh_offset
                        )
                    end
                end
            end
        end
        
        if crossed_frequency(batch_end, actual, SLICE_OUTPUT_FREQ)
            slice_step = (batch_end ÷ SLICE_OUTPUT_FREQ) * SLICE_OUTPUT_FREQ
            if slice_step >= t && slice_step <= batch_end
                # Async slice export (perf opt 2026-05-28). Each call spawns a
                # background Task; the GPU batch continues immediately.
                # Race-safety relies on `Array(::CuArray)` inside extract_slice_data
                # auto-syncing the device before the host copy. See
                # `spawn_slice_export!` docstring for the assumption.
                # Perf opt 2026-05-28: per-iteration dumps use the lean
                # `:production` field-set (density / velocity / schlieren / mach)
                # — drops to 4 fields × 3 planes = 12 PNGs per slice tick
                # (from 10 × 3 = 30). The full `:all` set is still emitted at
                # FinalSlice. Cuts ~60% of slice rendering wall-time on long runs.
                # lbm-io-opts (27 Oct 2026): numeric planes first, synchronously, from the state
                # of this batch (state step = batch_end), before any background PNG task runs.
                if LBMIO_EXPORT_DATA
                    run_export_stage("SliceData", slice_step) do
                        lbmio_export_planes!(slice_step, batch_end, grids, slices_dir)
                    end
                end
                if LBMIO_SLICE_CUSTOM   # lbm-io-opts: selected planes / fields; no PNG when png=false
                    lbmio_spawn_png_slices!(slice_step, grids, slices_dir, backend, SLICE_FIELD_SET;
                                            cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu)
                else   # legacy PNG set (options absent): the block below is unchanged and not re-indented
                spawn_slice_export!("Slice", slice_step, () -> begin
                    export_slice_image(slice_step, grids, slices_dir, backend; field=SLICE_FIELD_SET,
                                       cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu)
                end)
                spawn_slice_export!("SliceY025", slice_step, () -> begin
                    export_slice_image(slice_step, grids, slices_dir, backend; field=SLICE_FIELD_SET,
                                       cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu,
                                       y_override=0.25, fname_suffix="_Y0p25")
                end)
                spawn_slice_export!("SliceYZ", slice_step, () -> begin
                    export_slice_image(slice_step, grids, slices_dir, backend; field=SLICE_FIELD_SET,
                                       cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu,
                                       plane_override=:yz, fname_suffix="_YZ")
                end)
                if SLICE_FIELD_SET == :all && (AMR_INTERFACE_DIAGNOSTICS || AMR_SEAM_DIAGNOSTIC_ALL_STEPS)
                    spawn_slice_export!("SliceAMRDebug", slice_step, () -> begin
                        export_slice_image(slice_step, grids, slices_dir, backend; field=:amr_debug,
                                           cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu)
                    end)
                    spawn_slice_export!("SliceAMRDebugY025", slice_step, () -> begin
                        export_slice_image(slice_step, grids, slices_dir, backend; field=:amr_debug,
                                           cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu,
                                           y_override=0.25, fname_suffix="_Y0p25")
                    end)
                    spawn_slice_export!("SliceAMRDebugYZ", slice_step, () -> begin
                        export_slice_image(slice_step, grids, slices_dir, backend; field=:amr_debug,
                                           cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu,
                                           plane_override=:yz, fname_suffix="_YZ")
                    end)
                end
                if SLICE_FIELD_SET == :all && (AMR_INTERFACE_DIAGNOSTICS || AMR_SEAM_DIAGNOSTIC_ALL_STEPS)
                    wait_for_slice_tasks!()
                end
                end   # lbm-io-opts: legacy PNG set
                if THERMAL_INTERFACE_DIAGNOSTICS
                    run_export_stage("Diag", slice_step) do
                        export_thermal_interface_diagnostics(slice_step, grids, output_dir)
                    end
                end
                # ── Raw-field z-mirror residual probe (LUDWIG_MIRROR_PROBE=1) ──
                # Direct parity measurement on the GPU arrays — the slice-based
                # diagnostics CANNOT answer parity questions (one-sided pixel
                # stencil, see io_slice.jl ~L297 + diagnostics.jl probe header).
                if _MIRROR_PROBE
                    run_export_stage("MirrorProbe", slice_step) do
                        dump_z_mirror_residual(grids, domain_nz, slice_step)
                    end
                end
                # ── AMR validation gates 1+3 (13-mom conservation + H-theorem) ──
                # Walks every (parent, child) pair and dumps per-hop CSVs.
                # Cheap when AMR_VALIDATION_GATES=false (early return inside).
                if AMR_VALIDATION_GATES &&
                   (AMR_VALIDATION_FREQ <= 1 ||
                    slice_step % AMR_VALIDATION_FREQ == 0)
                    run_export_stage("AMR-validation", slice_step) do
                        for lvl in 1:(length(grids) - 1)
                            parent_level = grids[lvl]
                            child_level  = grids[lvl + 1]
                            length(parent_level.active_block_coords) == 0 && continue
                            length(child_level.active_block_coords)  == 0 && continue
                            try
                                dump_amr_validation_diagnostic!(
                                    parent_level, child_level,
                                    slices_dir, slice_step,
                                    cx_gpu, cy_gpu, cz_gpu, w_gpu,
                                    CS2_PHYSICS;
                                    shift_x=Float64(LATTICE_SHIFT_X),
                                    shift_y=Float64(LATTICE_SHIFT_Y),
                                    shift_z=Float64(LATTICE_SHIFT_Z))
                            catch e
                                println("          [AMR-validation] L$lvl→L$(lvl+1) failed: $e")
                            end
                        end
                    end
                end
                # ── Gate 2 — vortex metrics (Spiegel isentropic vortex test) ──
                if VORTEX_METRICS_ENABLED
                    run_export_stage("Vortex", slice_step) do
                        try
                            U_inlet_lat = u_curr
                            dump_vortex_metrics_diagnostic!(
                                grids, slices_dir, slice_step,
                                Float64(U_inlet_lat) + Float64(LATTICE_SHIFT_X);
                                U_inlet_y=Float64(LATTICE_SHIFT_Y),
                                U_inlet_z=Float64(LATTICE_SHIFT_Z))
                            # AUDIT ADD (2026-06-03): full XY field dump for the
                            # vortex convergence study (operator-accuracy benchmark).
                            eos_R_vtx = EOS_THERMODYNAMIC ? Float64((DDF_GAMMA - 1.0) * DDF_CV) : -1.0
                            dump_vortex_field_diagnostic!(
                                grids, slices_dir, slice_step,
                                Float64(params.mesh_offset[1]), Float64(params.mesh_offset[2]);
                                eos_R=eos_R_vtx)
                        catch e
                            println("          [Vortex] metrics failed: $e")
                        end
                    end
                end
                # ── Sod shock-tube centerline profile (case 050 validation) ──
                if SOD_CENTERLINE_ENABLED
                    run_export_stage("Sod", slice_step) do
                        try
                            eos_R_sod = EOS_THERMODYNAMIC ? Float64((DDF_GAMMA - 1.0) * DDF_CV) : -1.0
                            dump_sod_centerline_diagnostic!(
                                grids, slices_dir, slice_step,
                                Float64(params.mesh_offset[1]); eos_R=eos_R_sod)
                        catch e
                            println("          [Sod] centerline failed: $e")
                        end
                    end
                end
            end
        end

        if force_data !== nothing && crossed_frequency(batch_end, actual, FORCE_OUTPUT_FREQ)
            surface_step = (batch_end ÷ FORCE_OUTPUT_FREQ) * FORCE_OUTPUT_FREQ
            if surface_step >= t && surface_step <= batch_end
                run_export_stage("Surface", surface_step) do
                    need_surface_recompute = surface_step < RAMP_STEPS ||
                                             (surface_step % DIAG_FREQ != 0)
                    export_surface_outputs!(surface_step; recompute_if_needed=need_surface_recompute)
                end
            end
        end

        if crossed_frequency(batch_end, actual, OUTPUT_FREQ)
            flow_step = (batch_end ÷ OUTPUT_FREQ) * OUTPUT_FREQ
            if flow_step >= t && flow_step <= batch_end
                run_export_stage("Flow", flow_step) do
                    export_merged_mesh_sync(flow_step, grids, vtk_dir, backend;
                                            cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu,
                                            include_level=false)
                end
            end
        end

        t = batch_end + 1
    end

    # â”€â”€ Final output: ensure results are written even if STEPS < OUTPUT_FREQ â”€â”€
    # Check whether the last step already triggered an output
    last_output_step = (STEPS ÷ OUTPUT_FREQ) * OUTPUT_FREQ
    if false && (last_output_step != STEPS || STEPS < OUTPUT_FREQ)
        final_step = STEPS
        log_walltime("Writing final output at step $final_step...")

        try
            export_merged_mesh_sync(final_step, grids, vtk_dir, backend;
                                     cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu)
        catch e
            println("[Warning] Final VTK export failed: $e")
        end

        try
            export_slice_image(final_step, grids, slices_dir, backend; field=:all,
                               cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu)
        catch e
            println("[Warning] Final slice export failed: $e")
        end
        try
            export_thermal_interface_diagnostics(final_step, grids, output_dir)
        catch e
            println("[Warning] Final thermal interface diagnostics failed: $e")
        end

        if force_data !== nothing && final_step > RAMP_STEPS
            # Ensure aerodynamics are computed for the final state
            compute_aerodynamics!(force_data, force_level, gpu_mesh, backend, params;
                                  search_radius=force_search_radius,
                                  cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu)

            try
                surf_filename = @sprintf("%s/surface_%06d", vtk_dir, final_step)
                save_surface_vtk(surf_filename, force_data, geometry_mesh, params;
                                 object_names=object_names)

                surf_bin_filename = @sprintf("%s/surface_%06d.lbmp", vtk_dir, final_step)
                export_surface_pressure_bin(surf_bin_filename, force_data, unrotated_mesh, params;
                                            object_names=object_names,
                                            use_domain_offset=false)
            catch e
                println("[Warning] Final surface export failed: $e")
            end

            if NET_FORCE_EXPORT_ENABLED
                try
                    net_Fx, net_Fy, net_Fz = compute_net_aerodynamics!(
                        force_data, force_level, gpu_mesh, backend, params;
                        search_radius=force_search_radius,
                        cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu
                    )
                    export_net_triangle_forces_csv(
                        @sprintf("%s/net_forces_%06d.csv", csv_root, final_step),
                        net_Fx, net_Fy, net_Fz,
                        gpu_mesh, params.mesh_offset;
                        unrotated_centers=unrotated_centers,
                        alpha_deg=ALPHA_DEG,
                        beta_deg=BETA_DEG,
                        ground_mode=GROUND_ANALYSIS,
                        ground_up_axis=GROUND_UP_AXIS,
                        object_names=object_names
                    )
                    export_net_triangle_forces_vtk(
                        @sprintf("%s/net_forces_%06d", vtk_dir, final_step),
                        net_Fx, net_Fy, net_Fz,
                        gpu_mesh, params.mesh_offset
                    )
                catch e
                    println("[Warning] Final net forces export failed: $e")
                end
            end
        end
    end

    final_step = STEPS
    need_final_flow = OUTPUT_FREQ <= 0 ||
                      (OUTPUT_FREQ > 0 && STEPS > OUTPUT_FREQ &&
                       (STEPS ÷ OUTPUT_FREQ) * OUTPUT_FREQ != STEPS)
    need_final_slice = SLICE_OUTPUT_FREQ <= 0 ||
                       (SLICE_OUTPUT_FREQ > 0 && STEPS > SLICE_OUTPUT_FREQ &&
                        (STEPS ÷ SLICE_OUTPUT_FREQ) * SLICE_OUTPUT_FREQ != STEPS)
    need_final_surface = force_data !== nothing && final_step >= RAMP_STEPS &&
                         (FORCE_OUTPUT_FREQ <= 0 ||
                          (FORCE_OUTPUT_FREQ > 0 && STEPS > FORCE_OUTPUT_FREQ &&
                           (STEPS ÷ FORCE_OUTPUT_FREQ) * FORCE_OUTPUT_FREQ != STEPS))

    if need_final_flow || need_final_slice || need_final_surface
        log_walltime("Writing final output at step $final_step...")
    end

    # Wait for any background async slice exports to finish before doing the
    # final synchronous slice. Avoids race on file system + ensures consistent
    # output ordering.
    wait_pending_slice_exports()

    if need_final_slice && LBMIO_EXPORT_DATA   # lbm-io-opts: numeric planes of the final state
        run_export_stage("FinalSliceData", final_step) do
            lbmio_export_planes!(final_step, final_step, grids, slices_dir)
        end
    end
    if need_final_slice && LBMIO_SLICE_CUSTOM   # lbm-io-opts: selected planes / fields (none when png=false)
        lbmio_sync_png_slices!("Final", final_step, grids, slices_dir, backend, :all, LBMIO_SLICE_FIELDS;
                               cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu)
        if THERMAL_INTERFACE_DIAGNOSTICS
            run_export_stage("FinalDiag", final_step) do
                export_thermal_interface_diagnostics(final_step, grids, output_dir)
            end
        end
    elseif need_final_slice
        run_export_stage("FinalSlice", final_step) do
            export_slice_image(final_step, grids, slices_dir, backend; field=:all,
                               cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu)
        end
        run_export_stage("FinalSliceY025", final_step) do
            export_slice_image(final_step, grids, slices_dir, backend; field=:all,
                               cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu,
                               y_override=0.25, fname_suffix="_Y0p25")
        end
        run_export_stage("FinalSliceYZ", final_step) do
            export_slice_image(final_step, grids, slices_dir, backend; field=:all,
                               cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu,
                               plane_override=:yz, fname_suffix="_YZ")
        end
        if THERMAL_INTERFACE_DIAGNOSTICS
            run_export_stage("FinalDiag", final_step) do
                export_thermal_interface_diagnostics(final_step, grids, output_dir)
            end
        end
    end

    if need_final_surface
        run_export_stage("FinalSurface", final_step) do
            export_surface_outputs!(final_step; recompute_if_needed=true)
        end
    end

    if need_final_flow
        run_export_stage("FinalFlow", final_step) do
            export_merged_mesh_sync(final_step, grids, vtk_dir, backend;
                                    cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu,
                                    include_level=false)
        end
    end

    total_time = time() - t_start_sim
    println("\n" * "="^70)
    @printf("    SIMULATION COMPLETE | Wall time: %.1f s | Performance: %.1f MLUPS\n", total_time, (total_cells * STEPS) / (total_time * 1e6))
    println("="^70)
    flush(stdout)
    flush(stderr)
    
    if force_accm !== nothing
        print_averaged_summary(force_accm; re_number=params.re_number,
                               ma_physical=Ma_physical, ma_lattice=Ma_lattice,
                               compressibility_correction=COMPRESSIBILITY_CORRECTION)
    elseif force_data !== nothing
        print_force_summary(force_data; re_number=params.re_number,
                            ma_physical=Ma_physical, ma_lattice=Ma_lattice,
                            compressibility_correction=COMPRESSIBILITY_CORRECTION)
    end

    # â”€â”€ Write comprehensive analysis summary â”€â”€
    sim_end_datetime = Dates.now()
    try
        write_analysis_summary(output_dir, params, force_data, force_accm,
                               total_cells, total_time,
                               sim_start_datetime, sim_end_datetime;
                               gpu_name=gpu_device_name)
    catch e
        println("[Warning] Failed to write analysis summary: $e")
    end

    # â”€â”€ Build case result for multi-case summary â”€â”€
    case_result = build_case_result(params, force_data, force_accm,
                                     total_cells, total_time,
                                     sim_start_datetime, sim_end_datetime)

    grids = nothing
    force_cleanup()

    return case_result
end


function run_all_cases()
    snapshot = CaseIDContract.verify_startup_selection(STARTUP_SELECTION)
    length(STARTUP_RUNTIME_SELECTIONS) == length(snapshot.selections) ||
        error("Captured runtime selection count differs from the case queue")

    batch_start_datetime = Dates.now()
    case_results = BatchOutcomeContract.CaseOutcome[]

    println("="^70)
    println("      MULTI-CASE EXECUTION: $(length(snapshot.selections)) cases")
    println("="^70)
    for (i, runtime_selection) in enumerate(STARTUP_RUNTIME_SELECTIONS)
        CaseIDContract.verify_startup_selection(snapshot)
        CaseIDContract.verify_runtime_selection(snapshot.root_dir, runtime_selection)
        selection = runtime_selection.case
        case_name = selection.case_id
        println("\n>>> CASE $i/$(length(snapshot.selections)): $case_name")
        ERROR_LOG_PATH[] = ""
        case_output_claimed = false
        case_output_path = ""
        case_start_datetime = Dates.now()
        try
            output_claim = CaseIDContract.claim_output_directory(
                snapshot.root_dir, runtime_selection)
            case_output_claimed = true
            case_output_path = output_claim.target.path
            CaseIDContract.verify_claimed_output(runtime_selection, output_claim)
            ERROR_LOG_PATH[] = joinpath(output_claim.target.path, "ERROR_LOG.txt")
            open(ERROR_LOG_PATH[], "w") do io
                println(io, "=== LUDWIG run error log — case $case_name — started ", Dates.now(), " ===")
            end
            _errlog("CASE START $case_name")
            load_case_configuration(runtime_selection)
            CaseIDContract.verify_runtime_inputs(snapshot.root_dir, runtime_selection)
            CaseIDContract.verify_claimed_output(runtime_selection, output_claim)
            result = solve_main(runtime_selection, output_claim)
            result = BatchOutcomeContract.require_case_outcome(result, case_name)
            push!(case_results, result)
            if BatchOutcomeContract.case_completed(result)
                _errlog("CASE COMPLETED $case_name")
            else
                _errlog("CASE FAILED $case_name: $(result.error_msg)")
            end
        catch e
            _log_exception("case $case_name", e, catch_backtrace())
            failed = BatchOutcomeContract.failed_case_outcome(case_name, e;
                start_time=case_start_datetime, end_time=Dates.now(),
                output_dir=case_output_path)
            push!(case_results, failed)
            _errlog("CASE FAILED $case_name: $(failed.error_msg)")
        end
        GC.gc(true)
        # lbm-io-opts: a build-only case allocated nothing on the GPU; CUDA.reclaim() would create a context
        if case_output_claimed && CUDA.functional(); LBMIO_BUILD_ONLY || CUDA.reclaim(); end
    end

    batch_end_datetime = Dates.now()
    batch = BatchOutcomeContract.build_batch_outcome(
        case_results, batch_start_datetime, batch_end_datetime;
        expected_count=length(snapshot.selections))

    # â”€â”€ Write multi-case summary â”€â”€
    # CASE_LOGS was captured before run-side mutation. The final name is
    # exclusively claimed and never overwritten; summary failure fails the batch.
    return BatchOutcomeContract.finalize_batch!(batch) do published_batch
        summary_sink = CaseIDContract.claim_summary_sink(snapshot, STARTUP_SUMMARY_SINK)
        summary_filename = "cases_summary_$(Dates.format(batch_start_datetime, "yyyymmdd_HHMMSS")).md"
        summary_claim = CaseIDContract.write_summary_file_exclusive(
            summary_sink, summary_filename) do io, claim
            write_cases_summary(io, claim.path, published_batch)
        end
        println("[CasesSummary] Multi-case summary written to: $(summary_claim.path)")
        return summary_claim
    end
end

run_all_cases()


