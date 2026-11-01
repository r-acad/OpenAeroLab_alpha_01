# =====================================================================================================================
# lbm_case.jl -- settings, estimates and the generated case of the "LBM 3D (LUDWIG)" backend (27 Oct 2026). Pure Julia
# (Base, Printf, SHA, TOML, Dates). Included by the app's CFD backend (lbm_backend.jl) and by the worker (lbm_worker.jl).
#
# The LBM solver is LUDWIG (src/main.jl) run AS IT IS from a pinned, read-only git worktree (the snapshot). No solver
# file is changed. Which snapshot, and how the app drives it, is the solver PROFILE (LBM_PROFILES below; the one-line
# switch is LBM_PROFILE_DEFAULT):
#   "lbmint" (S1027_LBMINT, lbm-app-int 48ea3c52, merged as cav-rebuild-20260922 6dcfe299; the default since 27 Oct):
#            everything of "lbmio" below, plus the solver's periodic-Y AMR (pamr: lbm_levels 1-7 in this app) and its level-cap fix.
#   "lbmio"  (S1027_LBMIO, lbm-io-opts 21738159): the case lives in the app's own run folder,
#            <point>/lbm_root/CASES/lbmapp_<job>_<point>, and is selected with `main.jl --case <dir>` (no queue file is
#            read or written anywhere); the solver writes numeric mid-plane files (advanced.slice.export_data,
#            SLICES/step_N/plane_xz_mid_N.lbmslice) and no slice PNG (advanced.slice.png false); the domain check is the
#            solver's own basic.simulation.build_only mode (build_only_summary.toml, step-0 obstacle plane).
#   "lbmapp" (S1027_LBMAPP, 64104953, the first path): the case is CASES/Z<n>_lbmapp_<job>_<point> in the snapshot and
#            the snapshot's own cases_to_run.yaml points at it for the run; the app reads the solver's PNG slices; the
#            domain check stops the solver by its recorded pid once the domain is built.
# Both: `julia --project=<snapshot>/src <snapshot>/src/main.jl`, the solver's RESULTS/csv/forces.csv, convergence.csv,
# amr_viz/amr_data.json. Single level (num_levels 1) on both, because their spanwise periodicity
# (advanced.boundary.periodic_y) applies to single-level runs only. A profile with `pamr = true` (the solver with
# periodic-Y AMR, branch lbm-periodic-amr, merged with lbm-io-opts; its pinned snapshot is to come) also takes
# lbm_levels 2-7: advanced.boundary.periodic_y_amr, the span snapped to whole coarsest blocks, span-uniform levels, a
# near-body shell, a wake zone to the second-finest level and the gold-standard level coupling
# (W_LBMPAMR/memory/project_docs/_records/2026-10-27_lbm_periodic_amr/README.md section 5); and its single-level cases use
# periodic_y_amr too (LBM_PAMR_SINGLE_LEVEL; exactly span-invariant where the legacy periodic_y is not, ibid. T9).
# =====================================================================================================================
module AirfoilLBMCase

using Printf, SHA, TOML, Dates

export lbm_settings, lbm_estimate, lbm_case_yaml, LBM_DEFAULTS

"The solver snapshots the app can drive (see the header). `io`: the lbm-io-opts keys and the --case selector exist."
const LBM_VENDOR_SNAPSHOT = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "solvers", "lbm_lbmtau"))
const LBM_LEGACY_ROOT = normpath(get(ENV, "LUDWIG_LEGACY_SNAPSHOT_ROOT",
    joinpath(@__DIR__, "..", "..", "..", "..", "..", "02_PRIVATE_PROJECT_DEVELOPMENT", "legacy_snapshots")))
const LBM_PROFILES = Dict{String,NamedTuple}(
    # "lbmzw" (28 Oct): the merged solver with the corrected top/bottom walls, advanced.boundary.z_walls
    # "slip" | "no_slip" (branch lbm-zwalls, two independent reviews MERGE, GPU gates passed; merged into the main line as
    # 94c879c7). Its src/ is identical to 31271739; the merge also carries the lbm-app-int work, so every feature of
    # "lbmint" is in it. It has the walls only: the solver's time-reporting fix and the level-viscosity option are still on
    # the branch lbm-taufix (merged later as "lbmtau"), hence timefix=false, levelvisc=false here.
    "lbmzw" => (snapshot=joinpath(LBM_LEGACY_ROOT, "S1028_LBMZW"), commit="de5c387a14aa6de098c1d6bb35c2b5fc3df6bcba",
                manifest="lbm_snapshot_manifest_lbmzw.toml", io=true, pamr=true, zwalls=true, levelvisc=false, timefix=false,
                label="the merged solver with the corrected top/bottom walls (z_walls slip | no_slip), numeric mid-plane, 1-4 levels"),
    # "lbmtau" (28 Oct): the walls plus the solver's time-reporting fix (its time_s and Time_phys_s are per solver step =
    # 2^(L-1) finest steps; the plane header gains time_per_step_s: timefix) and the opt-in physical level viscosity
    # (advanced.refinement.level_viscosity "physical": levelvisc, available, OFF by default in the app: its stability gate GA
    # has not passed). Reviewed MERGE, GPU merge gates passed, merged into the main line as ca99784c; src/ identical to the
    # tested 7a2acfc0; it contains lbm-zwalls too.
    "lbmtau" => (snapshot=LBM_VENDOR_SNAPSHOT, commit="6e50352d7eb9511d880d6507bc40fc2043a7cb06",
                 manifest="lbm_snapshot_manifest_lbmtau.toml", io=true, pamr=true, zwalls=true, levelvisc=true, timefix=true,
                 label="the corrected walls, the solver's time per solver step, and the opt-in physical level viscosity (lbm-taufix)"),
    "lbmint" => (snapshot=joinpath(LBM_LEGACY_ROOT, "S1027_LBMINT"), commit="48ea3c522447643b05ff56c79713ed0edb8c57cc",
                 manifest="lbm_snapshot_manifest_lbmint.toml", io=true, pamr=true, zwalls=false, levelvisc=false,
                 label="the merged solver (lbm-app-int; io options + periodic-Y AMR + the level-cap fix): numeric mid-plane, --case selector, build-only domain check, 1-4 levels"),
    "lbmio" => (snapshot=joinpath(LBM_LEGACY_ROOT, "S1027_LBMIO"), commit="21738159377e884729b6f24988e7054486b0c91c",
                manifest="lbm_snapshot_manifest_lbmio.toml", io=true, pamr=false, zwalls=false, levelvisc=false,
                label="lbm-io-opts: numeric mid-plane export, --case selector, build-only domain check"),
    "lbmapp" => (snapshot=joinpath(LBM_LEGACY_ROOT, "S1027_LBMAPP"), commit="6410495396f9046919621e3969f0bb3e756a36b1",
                 manifest="lbm_snapshot_manifest.toml", io=false, pamr=false, zwalls=false, levelvisc=false,
                 label="the solver at 64104953: snapshot queue file, slice PNGs"),
)
# THE switch between solver snapshots (one line: "lbmzw" goes back to the walls without the time fix, "lbmint" to the merged
# solver without the corrected walls,
# "lbmio" to the io options alone, "lbmapp" to the first path).
# AIRFOIL_LBM_PROFILE overrides it for a test; a worker always uses the profile its input.toml names (an input without
# one is "lbmapp", the first path).
const LBM_PROFILE_DEFAULT = "lbmtau"         # since 28 Oct: its CPU checks (domain checks 1-2 levels, 2-level runs with the viscosity option on and off, no-slip, the time check) passed
lbm_profile_name() = get(ENV, "AIRFOIL_LBM_PROFILE", LBM_PROFILE_DEFAULT)
function lbm_profile(name::AbstractString=lbm_profile_name())
    haskey(LBM_PROFILES, name) || throw(ArgumentError("unknown LBM solver profile $(repr(name)) (known: $(join(sort(collect(keys(LBM_PROFILES))), ", ")))"))
    LBM_PROFILES[name]
end
const LBM_MAIN_CHECKOUT = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
"The snapshot of a profile (AIRFOIL_LBM_SNAPSHOT overrides the path for a development test only)."
lbm_snapshot(name::AbstractString=lbm_profile_name()) = normpath(get(ENV, "AIRFOIL_LBM_SNAPSHOT", lbm_profile(name).snapshot))
lbm_commit(name::AbstractString=lbm_profile_name()) = lbm_profile(name).commit
lbm_manifest(name::AbstractString=lbm_profile_name()) = joinpath(@__DIR__, "..", "..", "manifests", lbm_profile(name).manifest)
lbm_io(name::AbstractString=lbm_profile_name()) = lbm_profile(name).io
"Does a profile's solver take several refinement levels with the periodic span (periodic-Y AMR)?"
lbm_pamr(name::AbstractString=lbm_profile_name()) = get(lbm_profile(name), :pamr, false) === true
"""Does a profile's solver take advanced.boundary.z_walls ("slip" | "no_slip"; branch lbm-zwalls)? Without it the top and
bottom faces are the solver's mirror (free-slip) walls, and slip is written as advanced.domain.sponge_transverse_strength
0 (no freestream sponge on the z faces; on a periodic-Y case the key acts on the z faces only, apply_sponge!)."""
lbm_zwalls(name::AbstractString=lbm_profile_name()) = get(lbm_profile(name), :zwalls, false) === true
"""Does a profile's solver take advanced.refinement.level_viscosity "physical" (branch lbm-taufix)? Without it each
coarse level l of L carries 4^(L-l) x the physical viscosity (the legacy level rule, measured 4.005 x: the level-
viscosity record, W_LBMTAU memory/project_docs/_records/2026-10-28_lbm_level_viscosity)."""
lbm_levelvisc(name::AbstractString=lbm_profile_name()) = get(lbm_profile(name), :levelvisc, false) === true
"The standing caveat of the LBM drag (measured 28 Oct 2026, W_LBMWM memory/project_docs/_records/2026-10-28_lbm_wall_model): shown in every result."
const LBM_FRICTION_NOTE = "Friction (viscous) drag from the LBM surface-force routine is underestimated (measured 0.22-0.71 of exact in a laminar channel test); drag values are mostly pressure drag. Under investigation."
"The second standing caveat of the LBM results (review of the wall-model audit, 28 Oct 2026): shown next to the friction note everywhere."
const LBM_WALL_POSITION_NOTE = "The LBM curved-wall conditions place the effective wall about 1.1-1.3 cells inside the fluid (measured in laminar channel tests; independent of resolution), so bodies are effectively about that much thicker on each side and surface shear is misread. Root cause identified (a write-location defect in the linear Bouzidi family, a double relaxation in the grad condition, and the solid-cell marking); fixes are in development. LBM results are development-only until then."
"The `lbm_wall_model` control's label: the solver's wall-model switch does nothing in the solver as shipped (its audit found several defects)."
const LBM_WALL_MODEL_LABEL = "Wall model (inactive in the current solver: not validated)"
const LBM_LEVELVISC_REFUSAL = "The physical viscosity on coarse levels needs the solver update (lbm-taufix), which is not in the app's solver snapshot; use the legacy rule."
const LBM_LEVELVISC_TOOLTIP = "Experimental. The solver's legacy rule gives each coarse level 4^(L-l) x the physical viscosity (measured 4.005 x one level up). This option cuts it to the physical value where the stability floor tau >= 0.501 allows: at 128 cells per chord and Re 2e4, 1.3 x and 2.6 x physical on the two coarse levels of three. Its app-length stability screen (24 c/U) did not pass: the growth of the C_L fluctuation amplitude differed from the legacy twin's (1.5 x against 0.96 x); no numerical instability was seen. In that screen the peak velocity on the coarsest level after 8 c/U was 9-20 % above the legacy twin's. Results with it are not validated. Off by default; it has no effect with one level."
const LBM_NOSLIP_REFUSAL = "No-slip top and bottom walls need the solver update (lbm-zwalls), which is not in the app's solver snapshot; use slip walls."
const LBM_ZWALLS_NOTE = "Either wall type makes the domain a tunnel, not free air: the lift depends on the domain height (blockage and streamline curvature), more so for small heights. No-slip walls also grow a boundary layer along each wall from the inlet; whether that is wanted depends on what the case represents. Slip walls are an exact specular reflection, and the subgrid models (WALE, SVV) keep a small non-mirrored ghost at the faces (1 to 3.4e-4 U), so slip is nearly, not exactly, a symmetry plane."
"On a periodic-AMR profile, single-level cases use periodic_y_amr as well (exactly span-invariant; record T9)."
const LBM_PAMR_SINGLE_LEVEL = true
const LBM_LEVELS_MAX = 7
# The first path's names, kept for its worker and tests.
const LBM_SNAPSHOT_DEFAULT = LBM_PROFILES["lbmapp"].snapshot
const LBM_COMMIT_PINNED = LBM_PROFILES["lbmapp"].commit
const LBM_MANIFEST = joinpath(@__DIR__, "..", "..", "manifests", LBM_PROFILES["lbmapp"].manifest)
"The solver files whose content is pinned (relative to the snapshot root): the LBM driver and every directory main.jl includes."
const LBM_PINNED_DIRS = ("src/collision", "src/geometry", "src/solver", "src/io", "src/forces")
const LBM_PINNED_FILES = ("src/main.jl", "src/dependencies.jl", "src/preset_bootstrap.jl", "src/Project.toml",
                          "src/Manifest.toml", "src/LocalPreferences.toml", "tools/halt_guard.jl")

const FFMPEG_DEFAULT = "ffmpeg"
lbm_ffmpeg() = get(ENV, "AIRFOIL_FFMPEG", FFMPEG_DEFAULT)

# Physical carrier values (the LBM takes SI inputs; only their ratios matter): chord 1 m, U = 0.1 x 343 m/s, nu = U c / Re.
const LBM_CHORD_M = 1.0
const LBM_SPEED_OF_SOUND = 343.0
const LBM_PHYSICAL_MACH = 0.1
const LBM_U_LATTICE_DEFAULT = 0.08          # the solver's default lattice speed (lattice Mach 0.08*sqrt(3) = 0.139)
const LBM_TAU_MIN = 0.501                   # the solver's tau floor (advanced.numerics.tau_min default)
const LBM_GPU_MEMORY_GB = 10.0              # fallback for offline estimates
const LBM_GPU_LIMIT_GB = 9.0                # fallback for offline estimates

"Detected GPU memory with 1 GB total and 0.7 GB currently free reserved."
function lbm_gpu_capacity()
    raw = try
        read(`nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits`, String)
    catch
        ""
    end
    lines = filter(!isempty, strip.(split(strip(raw), '\n')))
    line = isempty(lines) ? "" : first(lines)
    parts = strip.(split(line, ','))
    if length(parts) >= 2
        free_mb = tryparse(Float64, parts[1]); total_mb = tryparse(Float64, parts[2])
        if free_mb !== nothing && total_mb !== nothing && total_mb > 0
            total = total_mb/1024; free = free_mb/1024
            return (total_gb=total, free_gb=free, budget_gb=max(0.0, min(total-1.0, free-0.7)), available=true)
        end
    end
    (total_gb=LBM_GPU_MEMORY_GB, free_gb=LBM_GPU_MEMORY_GB,
     budget_gb=LBM_GPU_LIMIT_GB, available=false)
end
# Memory and time MEASURED on the RTX 3080 (27 Oct 2026), on the app's own cases, by the integration check of the merged
# solver (branch lbm-app-int, 41b95853; W_LBMINT memory/project_docs/_records/2026-10-27_lbm_app_int/README.md section 2):
# - allocation per cell is a property of the level: 462 B on a level without wall cells, 700 B on the finest (wall)
#   level, 528 B for one level (with temporal_interpolation: false, as the lbmio cases write it); the solver's printed
#   "VRAM breakdown" is about half of that (it omits f_old, f_post_collision, the temporal anchors, the B-spline
#   coefficients...). One level with temporal interpolation on (the "lbmapp" path): 668 B/cell, measured by the
#   solver's build-only summary on the CPU (27 Oct, this work);
# - device peak = allocation + 0.6-0.7 GB (CUDA context and pool): the refusal limit applies to allocation + 0.7 GB;
# - time per SOLVER step. One solver step is one step of the COARSEST level (src/main.jl calls recursive_step! once per
#   step on level 1, and each level calls its child twice: solver_control.jl), so a solver step advances 2^(L-1) finest
#   steps and does W = sum over levels of cells_l 2^(l-1) cell updates (finding W7 of the level-viscosity record,
#   W_LBMTAU memory/project_docs/_records/2026-10-28_lbm_level_viscosity, checked against the source on 28 Oct). Measured per solver step:
#   one level 116 M updates/s (1.67 M cells); 2 levels 0.079 s at W 2.28 M (1.97 M cells) and 0.058 s at W 0.86 M (the
#   app's 2-level GPU run, 0.66 M cells): 0.0452 s + W / 67.4 M/s; 3 levels 0.18 s at W 1.87 M (the app's 3-level default,
#   0.74 M cells), 0.306 s at W 3.74 M (1.47 M cells) and 1.04 s at W 25.7 M (periodic-AMR T8, 12.4 M cells): least
#   squares 0.143 s + W / 28.5 M/s (within 15 %). Launch-bound at these sizes: extra levels save memory, not time.
#   4 levels: not measured (the 3-level form with twice its fixed cost);
# - one numeric mid-plane export: 0.37-0.39 s for 384 x 272 nodes, 1.37-1.65 s for 768 x 544 (about 3.7 us per node),
#   2.7-5.3 s at step 0; the first batch compiles for about 15-32 s.
const LBM_BYTES_PER_CELL_IO = 528.0         # one level, "lbmio" (temporal interpolation off), measured on the GPU
const LBM_BYTES_PER_CELL_APP = 668.0        # one level, "lbmapp" (temporal interpolation on), build-only summary
const LBM_BYTES_PER_CELL = LBM_BYTES_PER_CELL_APP
const LBM_FIXED_GB = 0.7                    # CUDA context and memory pool on top of the allocation (measured 0.6-0.7 GB)
const LBM_MLUPS = 116.0                     # one level, measured (million lattice updates per second)
const LBM_PNG_SECONDS = 1.2                 # one slice PNG render ("lbmapp": the solver renders 7 fields x 3 planes per frame)
const LBM_EXPORT_SECONDS_PER_NODE = 3.7e-6  # one numeric mid-plane export ("lbmio"), per plane node, measured
const LBM_EXPORT_SECONDS = 0.38             # the same at 384 x 272 nodes (kept for the estimate's basis text)
const LBM_AMR_BYTES_PER_CELL = 462.0        # a level without wall cells, measured
const LBM_AMR_BYTES_PER_WALL_CELL = 700.0   # the finest (wall) level, measured
"""Seconds per SOLVER step (one coarsest-level step) with L levels for W = sum_l cells_l 2^(l-1) cell updates per solver
step (measured; see above)."""
function lbm_amr_step_seconds(L::Integer, W::Real)
    L <= 1 && return W/(LBM_MLUPS*1e6)
    L == 2 && return 0.0452 + W/67.4e6
    fixed = L == 3 ? 0.143 : 0.143*2.0^(L - 3)
    fixed + W/28.5e6
end
"Cell updates per solver step: each level l of L marches 2^(l-1) steps per coarsest step."
lbm_updates_per_solver_step(level_cells::AbstractVector) = sum(c*2.0^(l - 1) for (l, c) in enumerate(level_cells))
"""Solver steps per convective time c/U: one solver step is one coarsest-level step, dt_coarse = 2^(L-1) dt_finest, and
U dt_finest = u_lattice dx_finest, so c/U is N/(u_lattice 2^(L-1)) solver steps (N finest cells per chord)."""
lbm_steps_per_ctu(N::Real, u_lattice::Real, levels::Integer) = N/(u_lattice*2^(levels - 1))
# Cells per level with several levels: scaled from record T8 (Z513), NACA 0012 at 4 deg, span 0.25 c, 3 levels, 384
# cells/c finest, domain 6 x 0.25 x 4.17 c: 5.53 / 3.69 / 3.19 M cells on L1 / L2 / L3 (lbm_amr_level_cells).
const LBM_AMR_T8 = (N=384, finest_area=0.2253, next_area=2.085, perimeter=2.04, area=0.0822, shell_blocks=2)

const LBM_AMR_REFUSAL = "The selected LBM solver profile does not support periodic-Y AMR. Select the pinned lbmtau profile to use 2 to 7 refinement levels."
const LBM_DEFAULTS = Dict{String,Any}("lbm_span_ratio"=>0.25, "lbm_cells_per_chord"=>128, "lbm_ctu"=>5.0,
    "lbm_ramp_ctu"=>0.5, "lbm_frames"=>60, "lbm_upstream"=>2.0, "lbm_downstream"=>3.0, "lbm_height"=>2.0,
    "lbm_plate_cells"=>2.0, "lbm_wall_model"=>false, "lbm_u_lattice"=>LBM_U_LATTICE_DEFAULT,
    "lbm_lattice"=>"d3q27", "reynolds"=>20000.0)

_g(d, k, default) = haskey(d, k) ? d[k] : (haskey(d, Symbol(k)) ? d[Symbol(k)] : default)
function _num(d, k, default, lo, hi; integer::Bool=false)
    v = _g(d, k, default)
    v isa AbstractString && (v = something(tryparse(Float64, v), NaN))
    (v isa Real && !(v isa Bool) && isfinite(v)) || throw(ArgumentError("$k must be a finite number"))
    lo <= v <= hi || throw(ArgumentError("$k must lie in [$lo, $hi] (got $v)"))
    integer && (isinteger(v) || throw(ArgumentError("$k must be a whole number")))
    integer ? Int(v) : Float64(v)
end
function _bool(d, k, default)
    v = _g(d, k, default)
    v isa Bool || throw(ArgumentError("$k must be boolean"))
    v
end

"""    lbm_settings(flow) -> Dict

Validated LBM settings of a request's flow map (ArgumentError with the reason on a bad value). The span is rounded to
whole 8-cell lattice blocks (the solver snaps every domain dimension to blocks): span_cells = 8 round(ratio N / 8)."""
function lbm_settings(flow::AbstractDict; pamr::Bool=lbm_pamr(), zwalls::Bool=lbm_zwalls(), levelvisc::Bool=lbm_levelvisc())
    lattice = String(_g(flow, "lbm_lattice", "d3q27"))
    lattice in ("d3q27", "d3q39") || throw(ArgumentError("lbm_lattice must be d3q27 or d3q39"))
    physical_mach = lattice == "d3q39" ? _num(flow, "mach", 0.3, 0.01, 1.6) : LBM_PHYSICAL_MACH
    cs2 = lattice == "d3q39" ? 2.0/3.0 : 1.0/3.0
    tau_min = lattice == "d3q39" ? 0.505 : LBM_TAU_MIN
    N = _num(flow, "lbm_cells_per_chord", LBM_DEFAULTS["lbm_cells_per_chord"], 16, 1024; integer=true)
    ratio = _num(flow, "lbm_span_ratio", LBM_DEFAULTS["lbm_span_ratio"], 0.05, 2.0)
    # Refinement: the resolution control is the FINEST level's cells per chord. Several levels need a periodic-AMR
    # profile (lbm_pamr); the span is then a whole number of the coarsest level's 8-cell blocks (8 2^(levels-1) finest
    # cells), as the solver requires (it refuses any other span).
    levels = _num(flow, "lbm_levels", 1, 1, LBM_LEVELS_MAX; integer=true)
    levels == 1 || pamr || throw(ArgumentError(LBM_AMR_REFUSAL))
    levels > 1 && N < 8*2^(levels - 1) && throw(ArgumentError("lbm_cells_per_chord $(N) leaves $(N/2^(levels - 1)) cells per chord on the coarsest of $(levels) levels: use at least $(8*2^(levels - 1)) cells per chord or fewer levels"))
    quantum = 8*2^(levels - 1)
    span_cells = quantum*max(1, round(Int, ratio*N/quantum))
    re = _num(flow, "reynolds", LBM_DEFAULTS["reynolds"], 100.0, 1e6)
    ctu = _num(flow, "lbm_ctu", LBM_DEFAULTS["lbm_ctu"], 0.05, 50.0)
    ramp = _num(flow, "lbm_ramp_ctu", LBM_DEFAULTS["lbm_ramp_ctu"], 0.0, 10.0)
    ramp < ctu || throw(ArgumentError("lbm_ramp_ctu must be shorter than lbm_ctu"))
    frames = _num(flow, "lbm_frames", LBM_DEFAULTS["lbm_frames"], 2, 400; integer=true)
    up = _num(flow, "lbm_upstream", LBM_DEFAULTS["lbm_upstream"], 0.5, 10.0)
    down = _num(flow, "lbm_downstream", LBM_DEFAULTS["lbm_downstream"], 1.0, 20.0)
    ht = _num(flow, "lbm_height", LBM_DEFAULTS["lbm_height"], 0.5, 10.0)
    plate = _num(flow, "lbm_plate_cells", LBM_DEFAULTS["lbm_plate_cells"], 1.5, 8.0)
    ulat = lattice == "d3q39" ? physical_mach*sqrt(cs2) :
        _num(flow, "lbm_u_lattice", LBM_DEFAULTS["lbm_u_lattice"], 0.02, 0.15)
    wall = _bool(flow, "lbm_wall_model", LBM_DEFAULTS["lbm_wall_model"])
    shell = _num(flow, "lbm_shell_blocks", 2, 1, 16; integer=true)
    # the top and bottom (z) walls: "slip" (default) or "no_slip" (needs the lbm-zwalls solver option)
    zw = _g(flow, "lbm_z_walls", "slip")
    (zw isa AbstractString && zw in ("slip", "no_slip")) || throw(ArgumentError("lbm_z_walls must be \"slip\" or \"no_slip\""))
    zw = String(zw)
    lattice == "d3q39" && zw == "no_slip" && throw(ArgumentError("D3Q39 does not support the pinned solver's no-slip z_walls option; select slip"))
    zw == "no_slip" && !zwalls && throw(ArgumentError(LBM_NOSLIP_REFUSAL))
    # the coarse levels' viscosity: "legacy" (default) or "physical" (experimental; needs the lbm-taufix option; with one
    # level there are no coarse levels and it has no effect)
    lv = _g(flow, "lbm_level_viscosity", "legacy")
    (lv isa AbstractString && lv in ("legacy", "physical")) || throw(ArgumentError("lbm_level_viscosity must be \"legacy\" or \"physical\""))
    lv = String(lv)
    lv == "physical" && !levelvisc && throw(ArgumentError(LBM_LEVELVISC_REFUSAL))
    # the numeric plane's extent: "auto" (the whole domain on one level, the body's region on several) or "domain"
    pw = _g(flow, "lbm_plane_window", "auto")
    (pw isa AbstractString && pw in ("auto", "domain")) || throw(ArgumentError("lbm_plane_window must be \"auto\" or \"domain\""))
    wake = _bool(flow, "lbm_wake_zone", true)
    check = _bool(flow, "lbm_domain_check", false)       # API-only (the app's tests): the solver's domain build on the CPU, no march
    cpu = _bool(flow, "lbm_cpu_run", false)              # API-only (the app's tests, "lbmio"): the whole run on the CPU backend, no GPU lock
    preferences=_g(flow,"output_preferences",Dict{String,Any}())
    preferences isa AbstractDict || throw(ArgumentError("output_preferences must be a map"))
    solver_png = get(preferences,"requested",false) ? get(preferences,"slice_png",false) : _bool(flow, "lbm_solver_png", false)
    (check && cpu) && throw(ArgumentError("lbm_domain_check and lbm_cpu_run exclude each other"))
    cap = _num(flow, "lbm_wall_cap_s", 0.0, 0.0, 172800.0)
    lockwait = _num(flow, "lbm_max_lock_wait_s", 0.0, 0.0, 604800.0)   # API-only: give up waiting for the GPU lock after this (0: never)
    settle = _num(flow, "lbm_lock_settle_s", 0.0, 0.0, 600.0)           # API-only: take the lock only after seeing it free for this long (not inside another owner's chain of jobs)
    steps_per_ctu = lbm_steps_per_ctu(N, ulat, levels)     # SOLVER steps (coarsest level) per c/U
    nsteps0 = max(frames, round(Int, ctu*steps_per_ctu))
    # The slice cadence is a whole number of the solver's 8-step GPU batches (advanced.gpu.async_depth 8): a frame then
    # records the state of its own step (the lbm-io-opts header's state_step == step).
    requested_display = _g(flow,"lbm_display_every",_g(flow,"display_every",nothing))
    if requested_display === nothing
        slice_every = 8*cld(cld(nsteps0, frames), 8)
        steps = slice_every*frames
    else
        interval=_num(flow,"lbm_display_every",requested_display,1,1_000_000;integer=true)
        slice_every=interval
        steps=gcd(8,interval)*cld(nsteps0,gcd(8,interval))
        frames=cld(steps,slice_every)
    end
    diag = requested_display===nothing ? max(5, round(Int, steps_per_ctu/16)) : slice_every
    diag = min(diag, steps)
    ramp_steps = round(Int, ramp*steps_per_ctu)
    nu_lat = ulat*N/re
    tau = lattice == "d3q39" ? 0.5 + nu_lat/cs2 : 0.5 + 3nu_lat
    tau_used = max(tau, tau_min)
    re_eff = lattice == "d3q39" ? ulat*N/((tau_used - 0.5)*cs2) : ulat*N/((tau_used - 0.5)/3)
    Dict{String,Any}("cells_per_chord"=>N, "span_ratio_requested"=>ratio, "span_cells"=>span_cells,
        "span"=>span_cells/N, "reynolds"=>re, "ctu"=>steps/steps_per_ctu, "ctu_requested"=>ctu, "ramp_ctu"=>ramp,
        "frames"=>frames, "upstream"=>up, "downstream"=>down, "height"=>ht, "plate_cells"=>plate, "plate_thickness"=>plate/N,
        "u_lattice"=>ulat, "lattice"=>lattice, "physical_mach"=>physical_mach, "cs2"=>cs2,
        "tau_min"=>tau_min, "wall_model"=>wall, "domain_check"=>check, "cpu_run"=>cpu, "solver_png"=>solver_png,
        "wall_cap_s"=>cap, "max_lock_wait_s"=>lockwait, "lock_settle_s"=>settle,
        "levels"=>levels, "shell_blocks"=>shell, "wake_zone"=>wake, "finest_cells_per_chord"=>N,
        "span_quantum_cells"=>quantum, "pamr"=>pamr, "z_walls"=>zw,
        "zwalls_option"=>lattice == "d3q27" && zwalls, "plane_window"=>String(pw),
        "level_viscosity"=>(levels > 1 && lv == "physical") ? "physical" : "legacy", "level_viscosity_requested"=>lv, "levelvisc_option"=>levelvisc,
        "coarsest_cells_per_chord"=>N/2^(levels - 1),
        "steps"=>steps, "steps_per_ctu"=>steps_per_ctu, "slice_every"=>slice_every, "diag_every"=>diag,
        "display_every_requested"=>requested_display,"display_every_applied"=>slice_every,"output_preferences"=>Dict{String,Any}(String(k)=>v for (k,v) in pairs(preferences)),
        "async_depth"=>requested_display===nothing ? 8 : gcd(8,slice_every),
        "finest_steps_per_solver_step"=>2^(levels - 1), "step_unit"=>"solver step = one coarsest-level step = $(2^(levels - 1)) finest step(s)",
        "ramp_steps"=>ramp_steps, "tau"=>tau, "tau_used"=>tau_used, "tau_clipped"=>tau < tau_min,
        "reynolds_effective"=>re_eff, "lattice_mach"=>ulat/sqrt(cs2), "velocity_m_s"=>physical_mach*LBM_SPEED_OF_SOUND,
        "kinematic_viscosity"=>physical_mach*LBM_SPEED_OF_SOUND*LBM_CHORD_M/re)
end

"First-cell y+ estimate (flat plate at the chord Reynolds number: laminar below 5e5, turbulent above), cell centre dx/2."
function lbm_yplus(re, N)
    cf = re < 5e5 ? 1.328/sqrt(re) : 0.026*re^(-1/7)
    0.5/N*re*sqrt(cf/2)
end

"""Estimated cells of each level of a periodic-AMR case (levels > 1), coarsest first, scaled from record T8
(LBM_AMR_T8): level 1 is the whole coarse domain (exact); the finest level covers the section plus a shell of
s["shell_blocks"] finest blocks (its extra area scales with the section's perimeter and the shell thickness); the
second-finest holds the wake (wake_length 1.5 c) and its extra area is taken as T8's; each further level adds a halo of
two of its own blocks around a region of perimeter 6.2 c. Every refined level spans the whole period
(span_uniform_levels). An estimate: the solver's own build gives the counts."""
function lbm_amr_level_cells(s::AbstractDict, lat; perimeter::Real=LBM_AMR_T8.perimeter, area::Real=LBM_AMR_T8.area)
    L = s["levels"]; N = s["cells_per_chord"]
    L == 1 && return [lat.ncx*lat.ncy*lat.ncz]
    t = LBM_AMR_T8
    dom = lat.Lx*lat.Lz
    shell = s["shell_blocks"]*8/N; shell8 = t.shell_blocks*8/t.N
    k = (t.finest_area - t.area)/(t.perimeter*shell8)
    A = zeros(L)
    A[L] = min(dom, area + k*perimeter*shell)
    for l in L-1:-1:2
        A[l] = l == L - 1 ? min(dom, A[L] + (s["wake_zone"] ? 1.0 : 0.3)*(t.next_area - t.finest_area)) :
                            min(dom, A[l+1] + 6.2*16*2.0^(L - l)/N)
    end
    cells = Int[lat.ncx*lat.ncy*lat.ncz]
    for l in 2:L
        Nl = N/2^(L - l)
        push!(cells, round(Int, A[l]*Nl^2*(s["span_cells"]/2^(L - l))))
    end
    cells
end

"""Cell count, memory and time estimates of a lattice (nx, ny, nz) with settings `s`. With `level_cells` (several
levels, from lbm_amr_level_cells) the periodic-AMR figures of record T8: memory per level, and the time from the
level-weighted updates per finest step at T8's rate. Memory: LBM_BYTES_PER_CELL per
cell + LBM_FIXED_GB; time: LBM_MLUPS sustained, plus the slice output: on "lbmio" one numeric mid-plane export per
frame (synchronous, LBM_EXPORT_SECONDS) and, with lbm_solver_png, three PNGs per frame; on "lbmapp" the solver's slice
rendering (7 production fields on three planes per frame, rendered on the CPU in a background task)."""
function lbm_estimate(s::AbstractDict, nx::Integer, ny::Integer, nz::Integer; io::Bool=lbm_io(),
                      level_cells::Union{Nothing,AbstractVector}=nothing, plane_nodes::Union{Nothing,Integer}=nothing)
    amr = level_cells !== nothing && length(level_cells) > 1
    cells = amr ? sum(level_cells) : nx*ny*nz
    L = amr ? length(level_cells) : 1
    bpc = io ? LBM_BYTES_PER_CELL_IO : LBM_BYTES_PER_CELL_APP
    factor = get(s, "lattice", "d3q27") == "d3q39" ? 3.0 : 1.0
    fixed_gb = factor > 1 ? 1.6 : LBM_FIXED_GB
    gb = amr ? factor*sum(c*(l == L ? LBM_AMR_BYTES_PER_WALL_CELL : LBM_AMR_BYTES_PER_CELL) for (l, c) in enumerate(level_cells))/1e9 + fixed_gb :
               factor*cells*bpc/1e9 + fixed_gb
    updates = amr ? lbm_updates_per_solver_step(level_cells) : Float64(cells)     # per solver (coarsest) step
    step_s = (factor > 1 ? 50.0 : 1.0)*(amr ? lbm_amr_step_seconds(L, updates) : cells/(LBM_MLUPS*1e6))
    march = step_s*s["steps"]
    scale = clamp(nx*nz/(768*536), 0.3, 3.0)
    pngs = io ? (get(s, "solver_png", false) === true ? 3s["frames"] + 3 : 0) : s["frames"]*21 + 6
    render = pngs*LBM_PNG_SECONDS*scale
    nodes = plane_nodes === nothing ? nx*nz : plane_nodes
    export_s = io ? (s["frames"] + 1)*(0.05 + LBM_EXPORT_SECONDS_PER_NODE*nodes) + 5.0 : 0.0   # + step 0 (slower, once)
    startup = 600.0                                      # Julia start, the solver's kernel compilation, domain build
    limit = get(s, "gpu_budget_gb", LBM_GPU_LIMIT_GB)
    total = get(s, "gpu_memory_gb", LBM_GPU_MEMORY_GB)
    Dict{String,Any}("cells"=>cells, "memory_gb"=>gb, "memory_limit_gb"=>limit, "gpu_memory_gb"=>total,
        "fits"=>gb <= limit, "seconds_per_step"=>step_s, "seconds_per_ctu"=>step_s*s["steps_per_ctu"],
        "march_seconds"=>march, "slice_render_seconds"=>render, "export_seconds"=>export_s, "startup_seconds"=>startup,
        "total_seconds"=>startup + export_s + max(march, render) + 0.2min(march, render), "pngs"=>pngs,
        "yplus_estimate"=>lbm_yplus(s["reynolds"], s["cells_per_chord"]),
        "levels"=>L, "level_cells"=>amr ? collect(level_cells) : [cells], "updates_per_solver_step"=>updates,
        "plane_nodes"=>nodes, "memory_basis"=>factor == 1.0 ? "measured on D3Q27 integration cases: allocation per cell + $(LBM_FIXED_GB) GB" :
            "conservative D3Q39 flow + thermal allowance: 3 x D3Q27 per-cell allocation + 1.6 GB; time is a rough extrapolation from the short GPU check",
        "basis"=>factor > 1 ? "D3Q39 flow + thermal: conservative 3 x D3Q27 per-cell memory plus 1.6 GB device overhead; runtime is a rough extrapolation from a short GPU check, not a measured app-length rate" :
               amr ? @sprintf("%d levels: %.0f B/cell (levels without wall cells) and %.0f B/cell (finest), measured, + %.1f GB; cells per level scaled from periodic-AMR T8; %.3g s per solver step (one coarsest step; measured rates at these sizes%s); %.2f s per numeric mid-plane frame",
                               L, LBM_AMR_BYTES_PER_CELL, LBM_AMR_BYTES_PER_WALL_CELL, LBM_FIXED_GB, step_s, L > 3 ? "; 4+ levels extrapolated" : "", 0.05 + LBM_EXPORT_SECONDS_PER_NODE*nodes) :
               io ? @sprintf("%.0f B/cell + %.1f GB, measured; %.0f MLUPS, measured; %.2f s per numeric mid-plane frame%s", bpc, LBM_FIXED_GB, LBM_MLUPS, 0.05 + LBM_EXPORT_SECONDS_PER_NODE*nodes,
                               pngs > 0 ? @sprintf(", %.1f s per solver PNG", LBM_PNG_SECONDS) : "") :
                      @sprintf("%.0f B/cell + %.1f GB; %.0f MLUPS; %.1f s per slice PNG", bpc, LBM_FIXED_GB, LBM_MLUPS, LBM_PNG_SECONDS))
end

"A cheap conservative allowance before triangulating the extruded section."
function lbm_prebuild_estimate(s::AbstractDict)
    levels = s["levels"]
    N = s["cells_per_chord"]
    f = 2^(levels - 1)
    coarse_x = 8*ceil(Int, (s["upstream"] + s["downstream"] + 1.15)*N/(8*f))
    coarse_z = 8*ceil(Int, (2*s["height"] + 0.4)*N/(8*f))
    coarse_y = s["span_cells"] ÷ f
    lat = (ncx=coarse_x, ncy=coarse_y, ncz=coarse_z,
           Lx=coarse_x*f/N, Lz=coarse_z*f/N)
    level_cells = levels == 1 ? nothing : lbm_amr_level_cells(s, lat)
    lbm_estimate(s, coarse_x*f, coarse_y*f, coarse_z*f; level_cells=level_cells)
end

"Choose the densest 8-cell increment whose conservative allocation estimate fits a user VRAM target."
function lbm_fit_cells_for_vram(flow::AbstractDict; capacity=lbm_gpu_capacity())
    percent_raw=_g(flow,"lbm_target_vram_percent",_g(flow,"target_vram_percent",nothing))
    percent=percent_raw===nothing ? nothing : _num(flow,"lbm_target_vram_percent",percent_raw,1,100)
    requested=percent===nothing ? _num(flow,"lbm_target_vram_gb",capacity.budget_gb,0.5,capacity.budget_gb) : capacity.free_gb*percent/100
    target=min(requested,capacity.budget_gb)
    target>=0.5 || throw(ArgumentError("The percentage target leaves less than 0.5 GB after the GPU reserve; raise the target or free GPU memory"))
    levels = _num(flow, "lbm_levels", 1, 1, LBM_LEVELS_MAX; integer=true)
    levels == 1 || lbm_pamr() || throw(ArgumentError(LBM_AMR_REFUSAL))
    first_n = levels == 1 ? 16 : 8*2^(levels - 1)
    best = nothing
    best_estimate = nothing
    for N in first_n:8:1024
        request = Dict{String,Any}(String(k)=>v for (k,v) in pairs(flow))
        request["lbm_cells_per_chord"] = N
        s = lbm_settings(request)
        s["gpu_budget_gb"] = capacity.budget_gb
        s["gpu_memory_gb"] = capacity.total_gb
        estimate = lbm_prebuild_estimate(s)
        if estimate["memory_gb"] <= target
            best = N
            best_estimate = estimate["memory_gb"]
        end
    end
    best === nothing && throw(ArgumentError(
        "No $(levels)-level lattice fits the $(round(target,digits=2)) GB VRAM target. Reduce refinement levels or raise the target."))
    (cells_per_chord=best, target_gb=target, estimated_gb=best_estimate,target_percent=percent,requested_gb=requested)
end

_f(x) = @sprintf("%.12g", x)

"""The refinement block of the case. Single level: the block size only. Several levels (periodic-Y AMR; the keys of the
record's section 5 and of its measured case Z513): every refined level spanning the period, a finest-level shell of
s["shell_blocks"] blocks around the body, the wake refined up to the second-finest level (the finest stays shape-
hugging), one-block margins, and the gold-standard level coupling (quintic B-spline prolongation, cubic + quintic
temporal anchors)."""
function _lbm_refinement_yaml(s::AbstractDict)
    L = s["levels"]
    L == 1 && return ["  refinement:", "    block_size: 8", "    symmetric_analysis: false"]
    get(s, "pamr", false) === true || throw(ArgumentError(LBM_AMR_REFUSAL))
    ["  refinement:", "    block_size: 8", "    symmetric_analysis: false", "    strategy: \"geometry_first\"", "    marker_only: false",
     "    margin: 1", "    margin_upstream: 1", "    margin_downstream: 1", "    margin_span: 1", "    margin_up: 1", "    margin_down: 1",
     "    span_uniform_levels: true", "    body_shell_blocks: $(s["shell_blocks"])",
     "    wake_enabled: $(s["wake_zone"])", "    wake_length: 1.5", "    wake_max_level: $(L - 1)",
     "    quintic_bspline_prolongation: true", "    interface_cubic_temporal: true", "    interface_quintic_temporal: true",
     (get(s, "level_viscosity", "legacy") == "physical" ? ["    level_viscosity: \"physical\""] : String[])...]
end

"""The tau and the viscosity relative to the physical one of each level (coarsest first) of settings `s`, by the solver's
rule: the finest tau 0.5 + 3 nu_lat floored at tau_min; a coarse level l of L takes 0.5 + (tau_f - 0.5) 2^(L-l) (the legacy
rule, 4^(L-l) x the physical viscosity) or, with level_viscosity "physical", 0.5 + (tau_f - 0.5) / 2^(L-l), floored at
tau_min (the floor then leaves it above physical). Returns a vector of (level, tau, ratio, floored)."""
function lbm_level_taus(s::AbstractDict)
    L = s["levels"]
    tau_min = get(s, "tau_min", LBM_TAU_MIN)
    tf_phys = s["tau"]; tf = max(tf_phys, tau_min)
    out = NamedTuple[]
    for l in 1:L
        k = 2.0^(L - l)
        phys = (tf_phys - 0.5)/k                          # the physical lattice viscosity x 3 at level l
        t = get(s, "level_viscosity", "legacy") == "physical" ? 0.5 + (tf - 0.5)/k : 0.5 + (tf - 0.5)*k
        floored = t < tau_min
        t = max(t, tau_min)
        push!(out, (level=l, tau=t, ratio=(t - 0.5)/phys, floored=floored || (l == L && tf_phys < tau_min)))
    end
    out
end
"Does a case use the solver's periodic-Y AMR (periodic_y_amr)? Several levels, or one level on a periodic-AMR profile."
lbm_uses_pamr(s::AbstractDict) = get(s, "pamr", false) === true && (s["levels"] > 1 || LBM_PAMR_SINGLE_LEVEL)

"The numeric mid-plane fields the app asks the solver for (\"lbmio\"; density is pressure / cs^2 here and is left out)."
const LBM_EXPORT_FIELDS = ("pressure", "u", "v", "w", "velocity_magnitude", "vorticity")
"The solver PNG fields written with lbm_solver_png (\"lbmio\")."
const LBM_SOLVER_PNG_FIELDS = ("velocity", "vorticity", "pressure")

"""The slice block of the case. "lbmio": one plane, xz_mid (the mid-span XZ plane through the STL centre), numeric files
of LBM_EXPORT_FIELDS at every slice step on the whole domain at the finest spacing, and no slice PNG unless
s["solver_png"] (then the three LBM_SOLVER_PNG_FIELDS on that plane only). "lbmapp": the solver's production PNG set."""
function _lbm_slice_yaml(s::AbstractDict, io::Bool; window=nothing)
    lines = ["  slice:", "    mode: \"production\"", "    cosmetic_filtering: false"]
    io || return lines
    q(v) = "[" * join(["\"$(x)\"" for x in v], ", ") * "]"
    append!(lines, ["    planes: [\"xz_mid\"]",
                    get(s, "solver_png", false) === true ? "    fields: $(q(LBM_SOLVER_PNG_FIELDS))" : "    png: false",
                    "    export_data: true",
                    "    export_fields: $(q(LBM_EXPORT_FIELDS))",
                    window === nothing ? "    export_window: \"domain\"" : "    export_window: [" * join(_f.(window), ", ") * "]",
                    "    export_level: \"finest\""])
end
"""The numeric plane's window for a several-level case (domain frame, metres, x then z): the body's box placed in the
domain (`placed_min`, `placed_max` as (x, z)) widened by 0.6 c upstream, 2 c downstream and 0.8 c above and below,
clipped to the domain (Lx, Lz); the far field is coarse there and would only repeat coarse cells at the finest
spacing. Single level: nothing (the whole domain)."""
function lbm_export_window(s::AbstractDict, placed_min, placed_max, Lx::Real, Lz::Real)
    (s["levels"] == 1 || get(s, "plane_window", "auto") == "domain") && return nothing
    [max(0.0, placed_min[1] - 0.6), min(Lx, placed_max[1] + 2.0), max(0.0, placed_min[2] - 0.8), min(Lz, placed_max[2] + 0.8)]
end

"""The case's config.yaml. `alpha_app` is the app's angle of attack (nose up positive); the solver's basic.alpha is its
negative (lbm_geometry.jl header). `span_len` is the resolved span (chords). Every setting the solver would otherwise
take from a default that matters here is written explicitly. `io` (the profile's lbm-io-opts keys): the numeric
mid-plane slice block and, for a domain check, basic.simulation.build_only."""
function lbm_case_yaml(s::AbstractDict, alpha_app::Real; stl::AbstractString="lbm_section.stl", note::AbstractString="",
                       io::Bool=lbm_io(), export_window=nothing)
    pamr = lbm_uses_pamr(s)
    d3q39 = get(s, "lattice", "d3q27") == "d3q39"
    L = s["levels"]
    N = s["cells_per_chord"]; span_len = s["span_cells"]/N
    nm = join(fill("1.0", 7), ", ")
    lines = String[
        "# Airfoil Lab LBM 3D case, generated $(Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS"))Z by the app (apps/airfoil_web/src/lbm/lbm_case.jl).",
        "# $(note)",
        "# Section extruded along Y by $(s["span_cells"]) finest cells = $(_f(span_len)) c, spanwise periodic ($(L == 1 ? "single level" : "$(L) levels, periodic-Y AMR")$(pamr && L == 1 ? ", periodic_y_amr" : "")), chord 1 m.",
        "# basic.alpha = -alpha: LUDWIG's positive alpha raises the trailing edge (lbm_geometry.jl).",
        "basic:",
        "  stl_file: [\"$(stl)\"]",
        "  stl_scale: 1.0",
        "  alpha: $(_f(-Float64(alpha_app)))",
        "  beta: 0.0",
        "  surface_resolution: $(N)",
        "  num_levels: $(s["levels"])",
        "  reference_area_of_full_model: $(_f(span_len*LBM_CHORD_M))",
        "  reference_chord: 1.0",
        "  reference_length_for_meshing: 1.0",
        "  reference_dimension: \"x\"",
        "  minimum_facet_size: 0.0",
        "  subdivision: false",
        "  fluid:",
        "    density: 1.0",
        "    kinematic_viscosity: $(_f(s["kinematic_viscosity"]))",
        "    speed_of_sound: $(_f(LBM_SPEED_OF_SOUND))",
        "  flow:",
        "    velocity: $(_f(s["velocity_m_s"]))",
        "  simulation:",
        "    steps: $(s["steps"])",
        "    ramp_steps: $(s["ramp_steps"])",
        "    output_freq: $(get(get(s,"output_preferences",Dict()),"volume_vtk",false) ? s["slice_every"] : s["steps"] + 1)",
        "    slice_output_freq: $(s["slice_every"])",
        "    output_startup_slices: true",
        "    output_startup_volume: false",
        (io && s["domain_check"] === true ? ["    build_only: true"] : String[])...,
        "    output_dir: \"RESULTS\"",
        "    output_fields:",
        "      density: true",
        "      velocity: true",
        "      velocity_magnitude: true",
        "      pressure: true",
        "      mach: true",
        "      temperature: $d3q39",
        "      schlieren: true",
        "      wall_distance: false",
        "      obstacle: true",
        "      level: true",
        "      bouzidi: false",
        "advanced:",
        "  solver_preset: \"vanilla\"",
        "  numerics:",
        "    lattice: \"$(s["lattice"])\"",
        "    compressible_mode: $d3q39",
        "    compressibility_correction: false",
        "    collision_operator: \"$(d3q39 ? "entropic" : "cumulant")\"",
        (d3q39 ? ["    eos_thermodynamic: true", "    shifted_lattice:", "      auto: false",
                  "      shift_x: $(s["physical_mach"] >= 1.2 ? 1 : 0)", "      shift_y: 0", "      shift_z: 0"] : String[])...,
        "    adaptive_omega_4: true",
        "    limiter: \"factored\"",
        "    u_lattice: $(_f(s["u_lattice"]))",
        "    c_wale: 0.5",
        "    tau_min: $(_f(s["tau_min"]))",
        "    tau_safety_factor: 1.0",
        "    inlet_turbulence_intensity: 0.0",
        "    nu_sgs_background: $(d3q39 && s["physical_mach"] >= 1.2 ? 0.02 : 0.0)",
        "    sponge_blend_distributions: false",
        # several levels: the temporal coupling of the levels; one level ("lbmio"): off, nothing reads its history arrays
        (L > 1 ? ["    temporal_interpolation: true", "    interface_macro_temporal_interpolation: true"] :
                 io ? ["    temporal_interpolation: false"] : String[])...,
        "    recursive_regularization:",
        "      enabled: false",
        "      min_level: 1",
        "    variable_nu_sgs:",
        "      enabled: $d3q39",
        "    svv_yz:",
        "      enabled: true",
        "      epsilon: 0.002",
        "      min_level: 1",
        "    svv_x:",
        "      enabled: true",
        "      epsilon: 0.002",
        "      max_level: 6",
        "    hrr_hybridization: false",
        "    hrr_sigma_per_level: [$(d3q39 ? join(fill("0.98", 7), ", ") : nm)]",
        "  thermal:",
        "    enabled: $d3q39",
        (d3q39 ? ["    feedback_to_flow: true", "    gamma: 1.4", "    cv: 2.5",
                  "    wall_bc: \"adiabatic\"", "  initial_condition:", "    type: \"uniform\"",
                  "    uniform:", "      use_freestream_velocity: true", "      rho: 1.0"] : String[])...,
        "  high_re:",
        "    auto_levels: false",
        "    wall_model:",
        "      enabled: $(s["wall_model"])",
        "      type: \"equilibrium\"",
        "      y_plus_target: 100.0",
        "  domain:",
        "    upstream: $(_f(s["upstream"]))",
        "    downstream: $(_f(s["downstream"]))",
        "    lateral: 0.5",
        "    height: $(_f(s["height"]))",
        "    span_override: $(_f((s["span_cells"] - 0.5)/N))",
        "    sponge_thickness: 0.20",
        # slip top and bottom walls without the solver's lbm-zwalls option: the mirror walls with no z-face sponge
        (get(s, "zwalls_option", false) === true ? String[] : ["    sponge_transverse_strength: 0.0"])...,
        _lbm_refinement_yaml(s)...,
        "  boundary:",
        "    method: \"bouzidi\"",
        "    bc_scheme: \"grad\"",
        "    grad_velocity_mode: \"q_blend\"",
        "    bouzidi_levels: 1",
        "    use_float16_qmap: false",
        "    q_min_threshold: 0.02",
        "    ff_fallback: \"wall_eq\"",
        "    missed_link_fix: \"bounded_blend\"",
        "    q_geometry: \"stl_ray\"",
        "    outlet_mode: \"extrapolate\"",
        "    inlet_density: \"local\"",
        "    periodic_y: true",
        (get(s, "zwalls_option", false) === true ? ["    z_walls: \"$(s["z_walls"])\""] : String[])...,
        (pamr ? ["    periodic_y_amr: true"] : String[])...,
        "    outlet_nscbc:",
        "      enabled: false",
        "  forces:",
        "    enabled: true",
        "    output_freq: $(get(get(s,"output_preferences",Dict()),"requested",false) && (get(s["output_preferences"],"surface_vtk",false)||get(s["output_preferences"],"surface_lbmp",false)) ? s["slice_every"] : s["steps"])",
        "  diagnostics:",
        "    freq: $(s["diag_every"])",
        "    stability_check: true",
        "    print_tau_warning: true",
        "    mass_conservation_diagnostics: false",
        "    amr_interface_diagnostics: false",
        "    amr_seam_all_steps: false",
        _lbm_slice_yaml(s, io; window=export_window)...,
        "  gpu:",
        "    async_depth: $(get(s,"async_depth",8))",
        ""]
    join(lines, "\n")
end

# ---------------------------------------------------------------------------------------------------- the snapshot
_head(snap) = begin
    normpath(snap) == LBM_VENDOR_SNAPSHOT && return LBM_PROFILES["lbmtau"].commit
    gitfile = joinpath(snap, ".git")
    if isfile(gitfile)
        gd = strip(replace(read(gitfile, String), "gitdir:" => ""))
        String(strip(read(joinpath(gd, "HEAD"), String)))
    else
        String(strip(read(`git -C $snap rev-parse HEAD`, String)))
    end
end
"The pinned solver files of a snapshot (relative paths, sorted)."
function lbm_pinned_files(snap::AbstractString)
    files = String[f for f in LBM_PINNED_FILES]
    for d in LBM_PINNED_DIRS
        root = joinpath(snap, d)
        for (dir, _, fs) in walkdir(root), f in fs
            endswith(f, ".jl") || continue
            push!(files, replace(relpath(joinpath(dir, f), snap), '\\' => '/'))
        end
    end
    sort!(unique!(files))
end
"Write the manifest of a freshly made snapshot (once, by the developer): commit and SHA-256 of every pinned file."
function lbm_write_manifest(snap::AbstractString=lbm_snapshot(); path::AbstractString=lbm_manifest())
    head = _head(snap)
    m = Dict{String,Any}("commit"=>head, "snapshot"=>replace(snap, '\\' => '/'), "written_utc"=>Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SSZ"),
        "sha256"=>Dict(f=>bytes2hex(open(sha256, joinpath(snap, f))) for f in lbm_pinned_files(snap)))
    open(io -> TOML.print(io, m), path, "w")
    path
end
"""The snapshot of a profile and its commit, refusing a snapshot that is missing, at another commit, or with a modified
solver file (SHA-256 against the profile's manifest). AIRFOIL_LBM_SNAPSHOT overrides the path for a development test
only. The lbm-io-opts profile also needs its reader, src/io/lbm_slice_format.jl (pinned with src/io)."""
function lbm_snapshot_check(name::AbstractString=lbm_profile_name())
    prof = lbm_profile(name)
    get(prof, :pending, false) === true && throw(ArgumentError("the LBM solver profile $(name) is prepared but its snapshot is not pinned yet"))
    snap = lbm_snapshot(name)
    isfile(joinpath(snap, "src", "main.jl")) || throw(ArgumentError(
        "the LBM solver snapshot is missing: $snap (create it with git worktree add --detach $snap $(prof.commit[1:8]))"))
    prof.io && !isfile(joinpath(snap, "src", "io", "lbm_slice_format.jl")) && throw(ArgumentError(
        "the LBM solver snapshot $snap has no src/io/lbm_slice_format.jl: it is not an lbm-io-opts snapshot (profile $(name))"))
    head = _head(snap)
    overridden = haskey(ENV, "AIRFOIL_LBM_SNAPSHOT")
    overridden || head == prof.commit || throw(ArgumentError("the LBM solver snapshot $snap is at $head, not the pinned $(prof.commit) (profile $(name))"))
    if !overridden
        m = TOML.parsefile(lbm_manifest(name))
        m["commit"] == head || throw(ArgumentError("the LBM snapshot manifest is for $(m["commit"]), the snapshot is at $head"))
        changed = [f for (f, h) in m["sha256"] if !isfile(joinpath(snap, f)) || bytes2hex(open(sha256, joinpath(snap, f))) != h]
        isempty(changed) || throw(ArgumentError("the LBM solver snapshot $snap has modified solver files ($(join(sort(changed), ", "))); refusing to run"))
    end
    halt = [p for p in (joinpath(snap, "SOLVER_HALT"), joinpath(LBM_MAIN_CHECKOUT, "SOLVER_HALT")) if isfile(p)]
    isempty(halt) || throw(ArgumentError("SOLVER_HALT is present ($(join(halt, ", "))): the LBM will not be started"))
    snap, head
end

"The next sequential case number: one above the highest Z### in the main checkout's CASES and the snapshot's CASES."
function lbm_next_case_number(snap::AbstractString=lbm_snapshot())
    hi = 0
    for root in (joinpath(LBM_MAIN_CHECKOUT, "CASES"), joinpath(snap, "CASES"))
        isdir(root) || continue
        for f in readdir(root)
            m = match(r"^Z(\d+)", f)
            m === nothing || (hi = max(hi, parse(Int, m.captures[1])))
        end
    end
    hi + 1
end

end # module AirfoilLBMCase
