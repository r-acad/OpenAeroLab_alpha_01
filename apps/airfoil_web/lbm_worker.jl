# =====================================================================================================================
# lbm_worker.jl -- one point of the "LBM 3D (LUDWIG)" backend (27 Oct 2026). Plain Julia (stdlib only):
#   julia --startup-file=no lbm_worker.jl <point_dir>/input.toml
#
# The app server has already written, into the point directory, the solver case (config.yaml + lbm_section.stl, from
# lbm_case.jl / lbm_geometry.jl) and input.toml, which names the solver profile (lbm_case.jl; an input without one is the
# first path, "lbmapp"). This worker:
#   1. takes the private development root's shared GPU lock (mkdir GPU_LOCK.d, owner.txt; it waits
#      and retries every 60 s while another owner holds it and never breaks a lock), then waits while another julia
#      process is still on the GPU (nvidia-smi), at most 10 times;
#   2. places the case:
#        "lbmio":  in the point directory itself, <point>/lbm_root/CASES/lbmapp_<job>_<point>, run with
#                  `main.jl --case <that folder>` (no queue file anywhere is read or written; the solver's batch summary
#                  goes to <point>/lbm_root/CASE_LOGS);
#        "lbmapp": in the pinned LBM snapshot as CASES/Z<n>_lbmapp_<job>_<point>, the SNAPSHOT's own cases_to_run.yaml
#                  pointing at it for the run (restored after; the main checkout's root files are never touched);
#   3. runs the solver as it is: julia --project=<snapshot>/src <snapshot>/src/main.jl, its output going straight to
#      <point_dir>/solver.log (the solver never writes into this worker's pipe), its pid recorded in lbm_solver.pid;
#   4. follows the solver: forces.csv rows become ITERATION lines (live CL/CD charts); each new mid-plane frame becomes
#      an LBM_NFRAME line ("lbmio": the solver's numeric plane converted to the app's viewer files, num/<field>/s<step>.f16)
#      or an LBM_FRAME line ("lbmapp": copies of the solver's velocity and vorticity PNGs); selected log lines are
#      forwarded; a wall cap (fault guard) stops the solver by its recorded pid;
#   5. releases the GPU lock as soon as the solver exits, then assembles the results and writes result.toml:
#        "lbmio":  one H.264 MP4 per video field (|u|/U, vorticity, Cp) rendered from the numeric planes on a FIXED
#                  colour scale (lbm_render.jl), a PNG poster of the last frame of each;
#        "lbmapp": the solver's PNG frames as JPEG copies and one MP4 per field.
# Domain check (input domain_check = true; an API-only option of the app's tests): no GPU lock and CUDA hidden from the
# solver (CUDA_VISIBLE_DEVICES=-1).
#   "lbmio":  the solver's own build-only mode (basic.simulation.build_only): it builds the domain on the CPU (STL,
#             voxelization, flood fill, Bouzidi q-maps), writes build_only_summary.toml, amr_viz/amr_data.json and the
#             step-0 mid-span plane with the obstacle mask, and exits by itself;
#   "lbmapp": the solver builds its domain on the CPU and writes its step-0 geometry slices; the worker then stops it by
#             its recorded pid (that snapshot's CPU backend could not march).
# CPU run (input cpu_run = true; "lbmio" only, API-only for the app's tests): the whole run on the solver's CPU backend,
# no GPU lock, CUDA hidden (the lbm-io-opts snapshot's CPU backend marches; slow, for small test lattices).
# =====================================================================================================================
using TOML, Printf, Dates
include(joinpath(@__DIR__, "gpu_lock.jl"))
include(joinpath(@__DIR__, "lbm_case.jl"))
include(joinpath(@__DIR__, "lbm_render.jl"))
using .AirfoilGPULock: holder, utc_now, LOCK_DIR, OWNER_FILE, default_poll
using .AirfoilLBMCase
using .AirfoilLBMRender

# `--assemble <input.toml>` (the app's tests only): assemble frames, videos and result.toml from an existing solver case
# (input keys case_dir, log) without running anything.
const ASSEMBLE_ONLY = length(ARGS) >= 2 && ARGS[1] == "--assemble"
const CFG = TOML.parsefile(ASSEMBLE_ONLY ? ARGS[2] : ARGS[1])
const OUT = normpath(CFG["outdir"])
const SMOKE = get(CFG, "domain_check", false) === true
const FIELDS_VIEW = ("velocity", "vorticity", "pressure", "density", "mach", "schlieren", "velocity_highpass")
const FIELDS_VIDEO = ("velocity", "vorticity", "pressure")
const THREADS = clamp(get(CFG, "threads", 8), 1, 8)
const PROFILE = String(get(CFG, "profile", "lbmapp"))
const IOP = AirfoilLBMCase.lbm_io(PROFILE)                     # the lbm-io-opts path
const CPU_RUN = IOP && get(CFG, "cpu_run", false) === true
const NUM_VIDEO = ("umag", "vort", "cp")                      # "lbmio": the fields with a video
const TIMEFIX = get(AirfoilLBMCase.lbm_profile(PROFILE), :timefix, false) === true   # the solver's time_s is per solver step
const TIME_MISMATCHES = String[]                                # frames whose solver time disagrees with the step count
const TIME_NOMINAL_DIFF = Float64[]                             # |solver time - nominal step-based time| / nominal, per frame
# "lbmio": the solver's own reader of its numeric plane files (Base + TOML), from the pinned snapshot.
IOP && include(joinpath(AirfoilLBMCase.lbm_snapshot(PROFILE), "src", "io", "lbm_slice_format.jl"))

# Writes to the server's pipe must never kill the worker (the server may restart): the run and the lock cleanup go on.
emit(args...) = try
    println(stdout, args...); flush(stdout)
catch
end
note(msg) = emit("[LBM] ", msg)

# ------------------------------------------------------------------------------------------------- the GPU lock
mutable struct LBMLock
    dir::String; owner::String; held::Bool
end
function lbm_release!(l::LBMLock)
    l.held || return false
    l.held = false
    f = joinpath(l.dir, OWNER_FILE)
    if isfile(f) && strip(read(f, String)) == l.owner
        rm(l.dir; recursive=true, force=true)
        emit("GPU_RELEASED,", replace(l.owner, ',' => ';'))
        return true
    end
    false
end
function lbm_acquire!(what::AbstractString; dir::AbstractString=LOCK_DIR, poll::Real=default_poll(), report::Real=60, max_wait::Real=Inf,
                      settle::Real=0)
    owner = "Airfoil Lab LBM 3D worker (app server job): $(what); pid $(getpid()); started $(utc_now())"
    last_holder = ""; last_print = -Inf; t0 = time(); free_since = nothing
    while true
        if settle > 0
            # the gap guard: take the lock only after it has been continuously free for `settle` seconds, so a short gap
            # between two jobs of another owner's chain is never used
            if isdir(dir)
                free_since = nothing
            else
                free_since === nothing && (free_since = time(); emit("GPU_WAIT,nobody (the lock is free; waiting ", round(Int, settle), " s to be sure it is not a gap in another owner's chain)"))
                if time() - free_since < settle
                    sleep(min(poll, max(1.0, settle/3)))
                    continue
                end
            end
        end
        created = try
            mkdir(dir); true
        catch err
            isdir(dir) || rethrow()
            false
        end
        if created
            write(joinpath(dir, OWNER_FILE), owner * "\n")
            l = LBMLock(String(dir), owner, true)
            atexit(() -> lbm_release!(l))
            emit("GPU_ACQUIRED,", replace(owner, ',' => ';'))
            return l
        end
        h = holder(dir)
        if h != last_holder || time() - last_print >= report
            emit("GPU_WAIT,", replace(h, ',' => ';'))
            last_holder = h; last_print = time()
        end
        time() - t0 > max_wait && error("gave up after waiting $(round(Int, time() - t0)) s for the GPU lock (held by $(holder(dir))); nothing was run")
        sleep(poll)
    end
end
"Julia processes on the GPU (nvidia-smi's compute-application list), as pid => name; empty when none or unreadable."
function gpu_julia_processes()
    out = try
        read(`nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader`, String)
    catch
        return Dict{Int,String}()
    end
    d = Dict{Int,String}()
    for ln in split(out, '\n')
        v = split(ln, ',')
        length(v) >= 2 || continue
        occursin("julia", lowercase(v[2])) || continue
        p = tryparse(Int, strip(v[1])); p === nothing || (d[p] = strip(v[2]))
    end
    d
end
function gpu_memory_used_mib()
    out = try read(`nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits`, String) catch; return (NaN, NaN) end
    v = split(strip(first(split(out, '\n'))), ',')
    length(v) >= 2 || return (NaN, NaN)
    (something(tryparse(Float64, strip(v[1])), NaN), something(tryparse(Float64, strip(v[2])), NaN))
end

# ------------------------------------------------------------------------------------------------- helpers
function csv_rows(path)
    isfile(path) || return Dict{String,String}[]
    lines = readlines(path)
    length(lines) >= 2 || return Dict{String,String}[]
    hdr = split(lines[1], ',')
    [Dict(String(k) => String(v) for (k, v) in zip(hdr, split(l, ','))) for l in lines[2:end] if count(==(','), l) == length(hdr) - 1]
end
fparse(s) = something(tryparse(Float64, strip(String(s))), NaN)
const FORWARD = r"^\s*(\[(Domain|Info|Memory|Regime|High-Re|Boundary|Init|Geometry|Stability|Backend|StabilityAbort|Summary|Warning|Error|BuildOnly|Cases|Levels|Standoff)\]|!!!|ERROR|Error|SIMULATION COMPLETE|\s*Cd  =|\s*Cl  =|\s*\d+ \| )"

# ------------------------------------------------------------------------------------------------- numeric planes ("lbmio")
const NUM = joinpath(OUT, "num")
const NUM_INDEX = Dict{Int,Dict{String,Any}}()      # step => frame record
const NUM_GRID = Ref{Any}(nothing)                  # the plane of the first file read (grid, header excerpt, solid count)
const NUM_HEADER_KEYS = ("plane", "plane_tag", "axis1", "axis2", "normal_axis", "normal_coord_m", "normal_target_m",
    "normal_index_finest", "plane_moved_to_cross_obstacle", "plane_crosses_obstacle", "frame", "mesh_offset_m",
    "domain_size_m", "coarse_grid_cells", "num_levels", "level_dx_m", "levels_used", "spacing_level", "combine_rule",
    "origin_m", "spacing_m", "window_m", "window_mode", "time_scale_s", "velocity_scale_m_s", "u_ref_m_s",
    "reference_chord_m", "pressure_ref_lattice", "dynamic_pressure_ref_lattice", "u_target_lattice", "case", "solver",
    "lattice", "collision_operator", "format", "format_version")

"""Convert one of the solver's plane files (SLICES/step_N/plane_xz_mid_N.lbmslice) into the app's viewer files:
num/mask.u8 once (n1 x n2 bytes, i along x fastest: 0 fluid, 1 solid, 2 no level), num/level.u8 (the level of each
node) and, for a frame (st > 0, a plane
with fields), num/<key>/s<step>.f16 for every field of AirfoilLBMRender.LBM_NUM_FIELDS (Float16 little-endian, same
layout, normalised; NaN outside the fluid). Returns the frame record, or nothing for a plane without fields."""
function convert_plane!(path::AbstractString, st::Integer)
    h, a = LBMSliceFormat.read_lbmslice(path)
    n1 = Int(h["n1"]); n2 = Int(h["n2"])
    obs = a["obstacle"]; lev = a["level"]
    mask = UInt8[lev[i] == 0x00 ? 0x02 : obs[i] == 0x01 ? 0x01 : 0x00 for i in eachindex(obs)]
    if NUM_GRID[] === nothing
        mkpath(NUM)
        write(joinpath(NUM, "mask.u8"), mask)
        write(joinpath(NUM, "level.u8"), vec(lev))         # the level that supplied each node (several levels)
        g = Dict{String,Any}(k => h[k] for k in NUM_HEADER_KEYS if haskey(h, k))
        g["n1"] = n1; g["n2"] = n2; g["solid_nodes"] = count(==(0x01), mask); g["no_level_nodes"] = count(==(0x02), mask)
        g["first_file"] = replace(path, '\\' => '/'); g["first_step"] = st
        NUM_GRID[] = g
    else
        g = NUM_GRID[]
        (g["n1"], g["n2"]) == (n1, n2) || error("the solver's plane changed size: $(g["n1"]) x $(g["n2"]) -> $n1 x $n2 ($path)")
        count(==(0x01), mask) == g["solid_nodes"] || note("The solver's mid-plane obstacle mask differs between steps ($path).")
    end
    (st > 0 && haskey(a, "u")) || return nothing
    # The time always comes from the step count (CFG["steps_per_ctu"] counts SOLVER steps, one coarsest step each). The
    # solver's own time_s is per its time bookkeeping: before the lbm-taufix fix it is state_step x the FINEST dt, 2^(L-1)
    # x too small with L levels (W7), and is not compared; on a profile with the fix (timefix: time_s per solver step,
    # and the header's time_per_step_s) it must equal the step-based time to 1e-9 relative, else a mismatch is recorded.
    spc = Float64(CFG["steps_per_ctu"])
    ctu = h["state_step"]/spc
    t_app = ctu*h["reference_chord_m"]/h["u_ref_m_s"]
    rec = Dict{String,Any}("step"=>st, "state_step"=>h["state_step"], "solver_time_s"=>h["time_s"],
        "time_s"=>t_app, "ctu"=>ctu, "files"=>Dict{String,Any}(),
        "min"=>Dict{String,Any}(), "max"=>Dict{String,Any}())
    haskey(h, "time_per_step_s") && (rec["solver_time_per_step_s"] = h["time_per_step_s"])
    if TIMEFIX
        # The solver's lattice speed is Float32: 0.08 is 0.07999999821 there, so its physical time per step is 2.2e-8 (relative)
        # off the nominal N/0.08 count. The check uses the solver's own Float32 speed (1e-9 relative); the nominal
        # difference is recorded (the app shows the nominal times).
        ulat = Float64(get(CFG, "u_lattice", 0.08)); ulat32 = Float64(Float32(ulat))
        spc_s = spc*ulat/ulat32
        t_exp = h["state_step"]/spc_s*h["reference_chord_m"]/h["u_ref_m_s"]
        dt_exp = h["reference_chord_m"]/h["u_ref_m_s"]/spc_s
        bad = String[]
        abs(h["time_s"] - t_exp) <= 1e-9*max(abs(t_exp), eps()) ||
            push!(bad, @sprintf("time_s %.12g s against the step-based %.12g s", h["time_s"], t_exp))
        haskey(h, "time_per_step_s") && !(abs(h["time_per_step_s"] - dt_exp) <= 1e-9*dt_exp) &&
            push!(bad, @sprintf("time_per_step_s %.12g s against the step-based %.12g s", h["time_per_step_s"], dt_exp))
        isempty(bad) || push!(TIME_MISMATCHES, @sprintf("step %d: ", st) * join(bad, "; "))
        t_app > 0 && push!(TIME_NOMINAL_DIFF, abs(h["time_s"] - t_app)/t_app)
        rec["solver_time_checked"] = isempty(bad)
    end
    for f in LBM_NUM_FIELDS
        haskey(a, f.src) || continue
        src = a[f.src]
        off, fac = f.lin(h)
        out = Vector{Float16}(undef, n1*n2); lo = Inf; hi = -Inf
        @inbounds for i in eachindex(out)
            v = mask[i] == 0x00 ? (Float64(src[i]) - off)*fac : NaN
            if isfinite(v)
                out[i] = Float16(clamp(v, -6.0e4, 6.0e4)); lo = min(lo, v); hi = max(hi, v)
            else
                out[i] = Float16(NaN)
            end
        end
        mkpath(joinpath(NUM, f.key))
        rel = @sprintf("num/%s/s%07d.f16", f.key, st)
        write(joinpath(OUT, rel), out)
        rec["files"][f.key] = rel; rec["min"][f.key] = lo; rec["max"][f.key] = hi
    end
    NUM_INDEX[st] = rec
    rec
end
"Convert every plane file of the solver case not converted yet (frames in step order); emit LBM_NFRAME lines when `live`."
function scan_planes!(slices::AbstractString, seen::Set{Int}; live::Bool=true)
    isdir(slices) || return 0
    steps = sort([parse(Int, m.captures[1]) for m in (match(r"^step_(\d+)$", d) for d in readdir(slices)) if m !== nothing])
    n = 0
    for st in steps
        st in seen && continue
        f = joinpath(slices, @sprintf("step_%06d", st), @sprintf("plane_xz_mid_%06d.lbmslice", st))
        isfile(f) || continue                       # the solver writes each file whole (a .partial renamed into place)
        push!(seen, st)
        rec = try
            convert_plane!(f, st)
        catch err
            note("Could not read the solver's plane $(f): " * first(sprint(showerror, err), 400)); nothing
        end
        rec === nothing && continue
        n += 1
        g = NUM_GRID[]
        live && emit(@sprintf("LBM_NFRAME,%d,%.6f,%d,%d,%.9g,%.9g,%.9g", st, rec["ctu"], g["n1"], g["n2"],
                              g["origin_m"][1], g["origin_m"][2], g["spacing_m"]))
    end
    n
end

# ------------------------------------------------------------------------------------------------- the run
function main()
    started = time()
    snap, head = AirfoilLBMCase.lbm_snapshot_check(PROFILE)
    for f in ("config.yaml", "lbm_section.stl")
        isfile(joinpath(OUT, f)) || error("the point directory has no $f")
    end
    gpu_lock = nothing
    baseline_mib = NaN
    if SMOKE
        note(IOP ? "Domain check (CPU, no GPU lock): the solver's build-only mode builds its domain, writes its summary and the step-0 mid-span obstacle plane, and exits; no march." :
                   "Domain check (CPU, no GPU lock): the solver builds its domain and step-0 slices, then it is stopped.")
    elseif CPU_RUN
        note("CPU run (the app's tests): the solver marches on its CPU backend with $(THREADS) threads; no GPU lock, CUDA hidden.")
    else
        mw = Float64(get(CFG, "max_lock_wait_s", 0.0))
        gpu_lock = lbm_acquire!("output $(OUT)"; max_wait=mw > 0 ? mw : Inf, settle=Float64(get(CFG, "lock_settle_s", 0.0)))
        for attempt in 1:10
            others = gpu_julia_processes()
            isempty(others) && break
            attempt == 10 && error("the GPU lock is ours but julia processes are still on the GPU: $(join(["$(v) (pid $(k))" for (k, v) in others], ", "))")
            emit("GPU_WAIT,the lock is ours but another julia process is still on the GPU: ", replace(join(["$(v) pid $(k)" for (k, v) in others], "; "), ',' => ';'))
            sleep(60)
        end
        baseline_mib = gpu_memory_used_mib()[1]
    end
    case_dir = ""; cases_file = joinpath(snap, "cases_to_run.yaml"); cases_saved = nothing
    proc = nothing; outcome = "failed"; message = ""
    peak_mib = baseline_mib
    rows_seen = 0; frames_seen = Set{Int}(); log_pos = 0
    forces = joinpath(OUT, "solver.log")   # placeholder, set below
    try
        if IOP
            # ---- the case in the point directory, selected with --case (lbm-io-opts)
            case_id = "lbmapp_" * String(CFG["case_suffix"])
            occursin(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", case_id) || error("invalid case id $case_id")
            root = joinpath(OUT, "lbm_root")
            case_dir = joinpath(root, "CASES", case_id)
            ispath(case_dir) && error("the case folder already exists: $case_dir")
            mkpath(case_dir)
            cp(joinpath(OUT, "config.yaml"), joinpath(case_dir, "config.yaml"))
            cp(joinpath(OUT, "lbm_section.stl"), joinpath(case_dir, "lbm_section.stl"))
            note("Case $(case_id) in the point folder ($(case_dir)), run by the pinned LBM snapshot $(snap) (commit $(head[1:8]), profile $(PROFILE)) with --case: no queue file is read or written.")
        else
            # ---- the case in the snapshot
            n = AirfoilLBMCase.lbm_next_case_number(snap)
            case_id = @sprintf("Z%03d_lbmapp_%s", n, CFG["case_suffix"])
            occursin(r"^[A-Za-z0-9_.-]+$", case_id) || error("invalid case id $case_id")
            case_dir = joinpath(snap, "CASES", case_id)
            ispath(case_dir) && error("the case folder already exists: $case_dir")
            mkpath(case_dir)
            cp(joinpath(OUT, "config.yaml"), joinpath(case_dir, "config.yaml"))
            cp(joinpath(OUT, "lbm_section.stl"), joinpath(case_dir, "lbm_section.stl"))
            cases_saved = isfile(cases_file) ? read(cases_file, String) : nothing
            write(cases_file, "case_folders:\n  - \"$(case_id)\"\n")
            note("Case $(case_id) in the pinned LBM snapshot $(snap) (commit $(head[1:8])); the snapshot's cases_to_run.yaml points at it for this run.")
        end
        open(joinpath(OUT, "lbm_case.toml"), "w") do io
            TOML.print(io, Dict("case_id"=>case_id, "case_dir"=>replace(case_dir, '\\' => '/'), "snapshot"=>replace(snap, '\\' => '/'),
                "commit"=>head, "profile"=>PROFILE, "started_utc"=>utc_now(), "worker_pid"=>getpid(), "domain_check"=>SMOKE, "cpu_run"=>CPU_RUN))
        end
        emit("LBM_CASE,", case_id)
        # ---- launch the solver as it is
        julia = joinpath(Sys.BINDIR, Base.julia_exename())
        env = Dict{String,String}(k => v for (k, v) in ENV if !startswith(k, "LUDWIG_") && !(k in ("JULIA_PROJECT", "JULIA_LOAD_PATH", "JULIA_NUM_THREADS")))
        (SMOKE || CPU_RUN) && (env["CUDA_VISIBLE_DEVICES"] = "-1")
        main_jl = joinpath(snap, "src", "main.jl")
        cmd = IOP ? Cmd(`$julia --startup-file=no --project=$(joinpath(snap, "src")) -t $(THREADS) --check-bounds=no -O3 $(main_jl) --case $(case_dir)`;
                        dir=dirname(dirname(case_dir)), env=env) :
                    Cmd(`$julia --startup-file=no --project=$(joinpath(snap, "src")) -t $(THREADS) --check-bounds=no -O3 $(main_jl)`;
                        dir=snap, env=env)
        logpath = joinpath(OUT, "solver.log")
        logio = open(logpath, "w")
        proc = run(pipeline(cmd; stdout=logio, stderr=logio); wait=false)
        pid = getpid(proc)
        write(joinpath(OUT, "lbm_solver.pid"), string(pid))
        note(SMOKE ? "Solver started (pid $(pid)) for the domain check (CPU); its log is solver.log." :
             CPU_RUN ? "Solver started (pid $(pid)): CPU run of $(CFG["steps"]) steps; its log is solver.log." :
                       "Solver started (pid $(pid)): GPU run of $(CFG["steps"]) steps; its log is solver.log.")
        forces = joinpath(case_dir, "RESULTS", "csv", "forces.csv")
        slices = joinpath(case_dir, "RESULTS", "SLICES")
        livedir = IOP ? "" : mkpath(joinpath(OUT, "live"))
        cap = Float64(CFG["wall_cap_s"])
        t_launch = time(); last_mem = 0.0
        check_png = Ref(0); check_t = Ref(time())
        while true
            alive = process_running(proc)
            # forward selected log lines
            if isfile(logpath)
                sz = filesize(logpath)
                if sz > log_pos
                    chunk = open(logpath, "r") do io
                        seek(io, log_pos); read(io, sz - log_pos)
                    end
                    txt = String(chunk)
                    cut = findlast('\n', txt)
                    if cut !== nothing
                        for ln in split(txt[1:cut], '\n')
                            l = rstrip(ln)
                            occursin(FORWARD, l) && emit(l)
                        end
                        log_pos += ncodeunits(txt[1:cut])
                    end
                end
            end
            # forces -> live charts
            rows = csv_rows(forces)
            for r in rows[rows_seen+1:end]
                st = tryparse(Int, r["Step"]); cd = fparse(r["Cd"]); cl = fparse(r["Cl"])
                st === nothing && continue
                if isfinite(cd) && isfinite(cl)
                    prev = rows_seen >= 1 ? rows[rows_seen] : nothing
                    res = prev === nothing ? 1.0 : abs(cl - fparse(prev["Cl"])) + abs(cd - fparse(prev["Cd"]))
                    emit(@sprintf("ITERATION,%d,%d,%.6e,%.6e,%.6e,%.8f,%.8f", st, CFG["steps"], max(res, 1e-12), max(res, 1e-12), max(res, 1e-12), cl, cd))
                end
                rows_seen += 1
            end
            # new mid-plane frames -> live view
            if IOP
                scan_planes!(slices, frames_seen)
            elseif isdir(slices)
                for d in readdir(slices)
                    m = match(r"^step_(\d+)$", d); m === nothing && continue
                    st = parse(Int, m.captures[1]); st == 0 && continue
                    st in frames_seen && continue
                    f = joinpath(slices, d, @sprintf("slice_velocity_%06d.png", st))
                    g = joinpath(slices, d, @sprintf("slice_vorticity_%06d.png", st))
                    (isfile(f) && isfile(g)) || continue
                    push!(frames_seen, st)
                    for (fld, src) in (("velocity", f), ("vorticity", g))
                        dst = joinpath(livedir, @sprintf("%s_%06d.png", fld, st))
                        try cp(src, dst; force=true) catch end
                    end
                    # keep the newest three of each
                    for fld in ("velocity", "vorticity")
                        old = sort(filter(x -> startswith(x, fld * "_"), readdir(livedir)))
                        length(old) > 3 && foreach(x -> rm(joinpath(livedir, x); force=true), old[1:end-3])
                    end
                    emit(@sprintf("LBM_FRAME,%d,live/velocity_%06d.png,live/vorticity_%06d.png", st, st, st))
                end
            end
            if gpu_lock !== nothing && time() - last_mem > 15
                peak_mib = max(peak_mib, gpu_memory_used_mib()[1]); last_mem = time()
            end
            alive || break
            if SMOKE && !IOP
                # the domain check ends once the domain record and the step-0 slices are written and stable for 10 s
                s0 = joinpath(slices, "step_000000")
                n0 = isdir(s0) ? count(f -> endswith(f, ".png"), readdir(s0)) : 0
                if isfile(joinpath(case_dir, "RESULTS", "amr_viz", "amr_data.json")) && n0 > 0
                    n0 == check_png[] || (check_png[] = n0; check_t[] = time())
                    if time() - check_t[] > 10
                        note("Domain check: domain record and $(n0) step-0 slices written; stopping the solver (pid $(pid)).")
                        kill(proc); sleep(3); process_running(proc) && kill(proc, Base.SIGKILL)
                        outcome = "domain_check"
                        break
                    end
                end
            end
            if cap > 0 && time() - t_launch > cap
                note(@sprintf("Wall cap (%.0f s, a fault guard) reached: stopping the solver (pid %d).", cap, pid))
                kill(proc); sleep(5); process_running(proc) && kill(proc, Base.SIGKILL)
                outcome = "wall_cap"
                break
            end
            sleep(2)
        end
        wait(proc)
        close(logio)
        gpu_lock === nothing || lbm_release!(gpu_lock)
        IOP && scan_planes!(slices, frames_seen)          # the planes written after the last poll
        log = read(logpath, String)
        complete = occursin("SIMULATION COMPLETE", log)
        if SMOKE && IOP
            ok = isfile(joinpath(case_dir, "RESULTS", "build_only_summary.toml")) && NUM_GRID[] !== nothing
            outcome = outcome == "wall_cap" ? "wall_cap" : ok ? "domain_check" : "failed"
            outcome == "failed" && (message = "the solver's build-only run did not write its summary and step-0 plane; see solver.log")
        elseif SMOKE
            outcome = isfile(joinpath(case_dir, "RESULTS", "amr_viz", "amr_data.json")) ? "domain_check" : "failed"
            outcome == "failed" && (message = "the solver stopped before writing its domain record; see solver.log")
        elseif outcome != "wall_cap"
            outcome = complete && rows_seen > 0 ? "completed" : "failed"
        end
        if outcome == "failed" && isempty(message)
            m = match(r"(?m)^.*(StabilityAbort|EXCEPTION in|ERROR:|OutOfGPUMemory|out of memory).*$", log)
            message = m === nothing ? "the solver exited without completing (exit code $(proc.exitcode)); see solver.log" : strip(m.match)
        end
        note(@sprintf("Solver finished: %s after %.0f s (%d force samples, %d mid-plane frames).", outcome, time() - t_launch, rows_seen,
                      IOP ? length(NUM_INDEX) : length(frames_seen)))
        # ---- frames and videos
        assemble_outputs(case_dir, log, outcome, message, head, snap, time() - t_launch, baseline_mib, peak_mib)
    catch err
        message = sprint(showerror, err)
        note("Worker error: " * first(message, 2000))
        if proc !== nothing && process_running(proc)
            note("Stopping the solver (pid $(getpid(proc))).")
            kill(proc); sleep(5); process_running(proc) && kill(proc, Base.SIGKILL)
        end
        isfile(joinpath(OUT, "result.toml")) || open(joinpath(OUT, "result.toml"), "w") do io
            TOML.print(io, Dict("outcome"=>"failed", "message"=>message, "case_dir"=>replace(case_dir, '\\' => '/'), "profile"=>PROFILE))
        end
        rethrow()
    finally
        gpu_lock === nothing || lbm_release!(gpu_lock)
        cases_saved === nothing || (try write(cases_file, cases_saved) catch end)
        note(@sprintf("Worker done in %.0f s.", time() - started))
    end
end

const FFLOG = Ref{Any}(devnull)

"""The numeric mid-plane results ("lbmio"): the index of the viewer files written by convert_plane! (grid, fields with
their fixed colour ranges, steps, times, file lists, per-frame ranges), and for each field of NUM_VIDEO an H.264 MP4
rendered on its fixed colour scale (lbm_render.jl) and a PNG poster of its last frame."""
function assemble_numeric(have_ff::Bool, ff::AbstractString)
    g = NUM_GRID[]
    g === nothing && return Dict{String,Any}(), Dict{String,Any}(), Dict{String,Any}()
    steps = sort(collect(keys(NUM_INDEX)))
    n1, n2 = g["n1"], g["n2"]
    has(key) = !isempty(steps) && all(haskey(NUM_INDEX[st]["files"], key) for st in steps)
    flds = [f for f in LBM_NUM_FIELDS if has(f.key)]
    numeric = Dict{String,Any}("grid"=>g, "mask"=>"num/mask.u8", "level"=>"num/level.u8", "steps"=>steps,
        "fields"=>[Dict{String,Any}("key"=>f.key, "label"=>f.label, "cmap"=>f.cmap, "range"=>collect(f.range), "mid"=>f.mid,
                                    "units"=>f.units, "source"=>f.src) for f in flds],
        "ctu"=>[NUM_INDEX[st]["ctu"] for st in steps], "time_s"=>[NUM_INDEX[st]["time_s"] for st in steps],
        "state_steps"=>[NUM_INDEX[st]["state_step"] for st in steps],
        "state_step_matches"=>all(NUM_INDEX[st]["state_step"] == st for st in steps),
        "files"=>Dict{String,Any}(f.key=>[NUM_INDEX[st]["files"][f.key] for st in steps] for f in flds),
        "min"=>Dict{String,Any}(f.key=>[NUM_INDEX[st]["min"][f.key] for st in steps] for f in flds),
        "max"=>Dict{String,Any}(f.key=>[NUM_INDEX[st]["max"][f.key] for st in steps] for f in flds),
        "dtype"=>"float16", "mask_dtype"=>"uint8",
        "time_check"=>Dict{String,Any}("compared"=>TIMEFIX, "mismatches"=>copy(TIME_MISMATCHES),
            "max_relative_difference_to_nominal"=>isempty(TIME_NOMINAL_DIFF) ? 0.0 : maximum(TIME_NOMINAL_DIFF),
            "rule"=>TIMEFIX ? "the solver's time_s (and time_per_step_s) against the step-based time computed with the solver's Float32 lattice speed, 1e-9 relative; the nominal times the app shows differ from the solver's by the Float32 rounding of u_lattice (2.2e-8 relative for 0.08)" :
                              "not compared: this solver's time_s counts finest steps (2^(L-1) x too small with L levels, W7); the app's times come from the step count"),
        "layout"=>"n1 x n2 values, i along x (axis1) fastest, little-endian; node (i, k) centre at origin_m + ((i - 0.5), (k - 0.5)) * spacing_m (domain frame); mask 0 fluid, 1 solid, 2 no level; field values NaN outside the fluid")
    videos = Dict{String,Any}(); posters = Dict{String,Any}()
    (have_ff && !isempty(steps)) || return numeric, videos, posters
    mask = reshape(read(joinpath(NUM, "mask.u8")), n1, n2)
    lvf = joinpath(NUM, "level.u8")
    levels = isfile(lvf) && filesize(lvf) == n1*n2 ? reshape(read(lvf), n1, n2) : nothing
    scale = max(1, fld(1600, n1))
    title = String(get(CFG, "title", "LBM 3D (LUDWIG)"))
    for key in NUM_VIDEO
        has(key) || continue
        f = lbm_field(key)
        tmp = mkpath(joinpath(OUT, "tmp_num_" * key))
        W = 0
        for (k, st) in enumerate(steps)
            rec = NUM_INDEX[st]
            vals = Float32.(reshape(reinterpret(Float16, read(joinpath(OUT, rec["files"][key]))), n1, n2))
            sub = @sprintf("%s   mid-span plane (y = %.4g m)   t = %.3f c/U   step %d", f.label, get(g, "normal_coord_m", NaN), rec["ctu"], st)
            img = render_frame(vals, mask, f, title, sub; scale=scale, levels=levels)
            W = size(img, 2)
            write_ppm(joinpath(tmp, @sprintf("%05d.ppm", k)), img)
        end
        tw = min(1920, W); tw -= isodd(tw)
        png = "poster_$(key).png"
        success(pipeline(`$ff -hide_banner -loglevel error -y -i $(joinpath(tmp, @sprintf("%05d.ppm", length(steps)))) $(joinpath(OUT, png))`;
                         stdout=FFLOG[], stderr=FFLOG[])) && (posters[key] = png)
        if length(steps) >= 2
            fps = clamp(round(Int, length(steps)/8), 4, 24)
            mp4 = "video_$(key).mp4"
            okv = success(pipeline(`$ff -hide_banner -loglevel error -y -framerate $fps -i $(joinpath(tmp, "%05d.ppm")) -vf scale=$(tw):-2:flags=lanczos,format=yuv420p -c:v libx264 -preset medium -crf 20 -movflags +faststart $(joinpath(OUT, mp4))`;
                                   stdout=FFLOG[], stderr=FFLOG[]))
            okv && (videos[key] = Dict{String,Any}("file"=>mp4, "frames"=>length(steps), "fps"=>fps,
                "codec"=>"H.264 (libx264), yuv420p, CRF 20, $(tw) px wide", "colour_range"=>collect(f.range), "colour_map"=>f.cmap,
                "label"=>f.label, "source"=>"numeric mid-plane, fixed colour scale"))
        end
        rm(tmp; recursive=true, force=true)
    end
    numeric, videos, posters
end

"Mid-plane frames (JPEG copies at 1600 px of the solver's PNGs, when it wrote any), the numeric mid-plane results
(\"lbmio\": assemble_numeric), one MP4 per video field, and result.toml."
function assemble_outputs(case_dir, log, outcome, message, head, snap, march_wall, baseline_mib, peak_mib)
    FFLOG[] = open(joinpath(OUT, "ffmpeg.log"), "a")
    res = joinpath(case_dir, "RESULTS")
    slices = joinpath(res, "SLICES")
    IOP && NUM_GRID[] === nothing && scan_planes!(slices, Set{Int}(); live=false)   # --assemble: nothing was followed live
    steps = isdir(slices) ? sort([parse(Int, m.captures[1]) for m in (match(r"^step_(\d+)$", d) for d in readdir(slices)) if m !== nothing]) : Int[]
    ff = AirfoilLBMCase.lbm_ffmpeg()
    have_ff = isfile(ff)
    frames = Dict{String,Any}(); videos = Dict{String,Any}()
    framedir = mkpath(joinpath(OUT, "frames"))
    for fld in FIELDS_VIEW
        list = [(st, joinpath(slices, @sprintf("step_%06d", st), @sprintf("slice_%s_%06d.png", fld, st))) for st in steps if st > 0]
        filter!(x -> isfile(x[2]), list)
        isempty(list) && continue
        tmp = mkpath(joinpath(OUT, "tmp_" * fld))
        for (k, (_, src)) in enumerate(list)
            cp(src, joinpath(tmp, @sprintf("%05d.png", k)); force=true)
        end
        dst = mkpath(joinpath(framedir, fld))
        ok = have_ff && success(pipeline(`$ff -hide_banner -loglevel error -y -i $(joinpath(tmp, "%05d.png")) -vf scale=1600:-2:flags=lanczos -q:v 3 $(joinpath(dst, "%05d.jpg"))`;
                                           stdout=FFLOG[], stderr=FFLOG[]))
        if !ok    # no ffmpeg: keep the solver's PNGs themselves
            for (k, (_, src)) in enumerate(list)
                cp(src, joinpath(dst, @sprintf("%05d.png", k)); force=true)
            end
        end
        ext = ok ? "jpg" : "png"
        frames[fld] = Dict("steps"=>[st for (st, _) in list], "files"=>[@sprintf("frames/%s/%05d.%s", fld, k, ext) for k in 1:length(list)],
                           "source"=>[replace(src, '\\' => '/') for (_, src) in list])
        if have_ff && fld in FIELDS_VIDEO && length(list) >= 2
            mp4 = joinpath(OUT, "video_$(fld).mp4")
            fps = clamp(round(Int, length(list)/8), 4, 24)
            okv = success(pipeline(`$ff -hide_banner -loglevel error -y -framerate $fps -i $(joinpath(tmp, "%05d.png")) -vf scale=1920:-2:flags=lanczos,format=yuv420p -c:v libx264 -preset medium -crf 20 -movflags +faststart $mp4`;
                                   stdout=FFLOG[], stderr=FFLOG[]))
            okv && (videos[fld] = Dict("file"=>"video_$(fld).mp4", "frames"=>length(list), "fps"=>fps, "codec"=>"H.264 (libx264), yuv420p, CRF 20, 1920 px wide"))
        end
        rm(tmp; recursive=true, force=true)
    end
    numeric, nvideos, posters = IOP ? assemble_numeric(have_ff, ff) : (Dict{String,Any}(), Dict{String,Any}(), Dict{String,Any}())
    merge!(videos, nvideos)
    build = Dict{String,Any}()
    bs = joinpath(res, "build_only_summary.toml")
    isfile(bs) && (build = TOML.parsefile(bs))
    # the solver's mid-span geometry slices at step 0 (obstacle mask, level map)
    startup = Dict{String,Any}()
    for nm in ("level", "wall_dist", "amr_edge")
        src = joinpath(slices, "step_000000", "slice_$(nm)_000000.png")
        isfile(src) && (cp(src, joinpath(OUT, "startup_$(nm).png"); force=true); startup[nm] = "startup_$(nm).png")
    end
    # the solver's own domain record (amr_viz/amr_data.json header) and log lines
    lattice = Dict{String,Any}()
    aj = joinpath(res, "amr_viz", "amr_data.json")
    if isfile(aj)
        head_txt = String(open(io -> read(io, 4000), aj))
        for k in ("domain_nx", "domain_ny", "domain_nz", "n_levels", "block_size", "n_triangles")
            m = match(Regex("\"$(k)\":(\\d+)"), head_txt); m === nothing || (lattice[k] = parse(Int, m.captures[1]))
        end
        m = match(r"\"mesh_offset\":\[([^\]]+)\]", head_txt); m === nothing || (lattice["mesh_offset"] = fparse.(split(m.captures[1], ',')))
    end
    if NUM_GRID[] !== nothing        # "lbmio": the plane header's own record of the domain
        g = NUM_GRID[]
        haskey(g, "coarse_grid_cells") && (lattice["coarse_grid_cells"] = g["coarse_grid_cells"])
        haskey(g, "mesh_offset_m") && (lattice["mesh_offset_plane"] = g["mesh_offset_m"])
        haskey(g, "domain_size_m") && (lattice["domain_size_m"] = g["domain_size_m"])
        haskey(g, "num_levels") && (lattice["num_levels"] = g["num_levels"])
    end
    m = match(r"\[Info\] Total cells: ([0-9.]+) M", log); m === nothing || (lattice["total_cells_millions"] = fparse(m.captures[1]))
    m = match(r"\[Domain\] Resolved grid: domain = ([0-9.]+) × ([0-9.]+) × ([0-9.]+) m\s+\((\d+) × (\d+) × (\d+) coarse cells", log)
    m === nothing || (lattice["resolved_grid"] = m.match)
    perf = Dict{String,Any}()
    m = match(r"SIMULATION COMPLETE \| Wall time: ([0-9.]+) s \| Performance: ([0-9.]+) MLUPS", log)
    m === nothing || (perf["solver_wall_s"] = fparse(m.captures[1]); perf["mlups_inclusive"] = fparse(m.captures[2]))
    conv = csv_rows(joinpath(res, "csv", "convergence.csv"))
    ml = [fparse(r["MLUPS"]) for r in conv]
    ml = filter(x -> isfinite(x) && x > 0, ml)
    if length(ml) >= 2
        s = sort(ml[2:end]); perf["mlups_compute_median"] = s[cld(length(s), 2)]
    end
    perf["march_wall_s"] = march_wall
    m = match(r"Timestep batch START \(steps 1:", log)
    perf["gpu_baseline_mib"] = baseline_mib; perf["gpu_peak_mib"] = peak_mib
    perf["gpu_run_mib"] = isfinite(peak_mib) && isfinite(baseline_mib) ? peak_mib - baseline_mib : NaN
    vram = [fparse(x.captures[1]) for x in eachmatch(r"\[Memory\] VRAM: ([0-9.]+) GB free", log)]
    isempty(vram) || (perf["solver_min_free_gb"] = minimum(vram))
    # forces
    rows = csv_rows(joinpath(res, "csv", "forces.csv"))
    hist = [(parse(Int, r["Step"]), fparse(r["Cl"]), fparse(r["Cd"]), fparse(r["Cs"]), fparse(r["Cmy"]), fparse(r["Cdp"]), fparse(r["Cdv"])) for r in rows if tryparse(Int, r["Step"]) !== nothing]
    summ = Dict{String,Any}()
    for (k, rx) in (("cd_solver_mean", r"Averaged Coefficients:\s*\n\s*Cd\s*=\s*([-+0-9.eE]+)"), ("cl_solver_mean", r"Averaged Coefficients:[^\n]*\n[^\n]*\n\s*Cl\s*=\s*([-+0-9.eE]+)"))
        m = match(rx, log); m === nothing || (summ[k] = fparse(m.captures[1]))
    end
    m = match(r"\((\d+) samples after ramp\)", log); m === nothing || (summ["solver_mean_samples"] = parse(Int, m.captures[1]))
    write(joinpath(OUT, "forces_history_NOTE.txt"), AirfoilLBMCase.LBM_FRICTION_NOTE * "\n" * AirfoilLBMCase.LBM_WALL_POSITION_NOTE * "\nColumns cdp and cdv are the pressure and viscous parts of cd from that routine; cdv is the underestimated one.\n")
    open(joinpath(OUT, "forces_history.csv"), "w") do io
        println(io, "step,ctu,cl,cd,cs,cmy,cdp,cdv")
        for h in hist
            @printf(io, "%d,%.6f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n", h[1], h[1]/CFG["steps_per_ctu"], h[2], h[3], h[4], h[5], h[6], h[7])
        end
    end
    result = Dict{String,Any}("outcome"=>outcome, "message"=>message, "commit"=>head, "snapshot"=>replace(snap, '\\' => '/'),
        "case_dir"=>replace(case_dir, '\\' => '/'), "steps"=>CFG["steps"], "steps_per_ctu"=>CFG["steps_per_ctu"],
        "force_samples"=>length(hist), "frames"=>frames, "videos"=>videos, "startup_slices"=>startup, "lattice_solver"=>lattice,
        "performance"=>perf, "solver_summary"=>summ, "domain_check"=>SMOKE, "ffmpeg"=>have_ff ? ff : "missing",
        "slice_steps"=>[s for s in steps if s > 0], "friction_drag_note"=>AirfoilLBMCase.LBM_FRICTION_NOTE, "wall_position_note"=>AirfoilLBMCase.LBM_WALL_POSITION_NOTE, "profile"=>PROFILE, "cpu_run"=>CPU_RUN, "numeric"=>numeric,
        "posters"=>posters, "build_only"=>build)
    if !isempty(hist)
        result["last_step"] = hist[end][1]; result["cl_last"] = hist[end][2]; result["cd_last"] = hist[end][3]
    end
    open(io -> TOML.print(io, result), joinpath(OUT, "result.toml"), "w")
    try close(FFLOG[]) catch end
    outcome == "failed" && error(isempty(message) ? "LBM run outcome: $(outcome)" : message)
end

if ASSEMBLE_ONLY
    assemble_outputs(CFG["case_dir"], read(CFG["log"], String), get(CFG, "outcome", "completed"), "", AirfoilLBMCase.lbm_commit(PROFILE),
                     AirfoilLBMCase.lbm_snapshot(PROFILE), 0.0, NaN, NaN)
else
    main()
end
