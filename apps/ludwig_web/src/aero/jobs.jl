include(joinpath(@__DIR__,"..","workbench_revision.jl"))
include(joinpath(@__DIR__,"run_monitoring.jl"))

_job_dir(id::String) = joinpath(JOBS_ROOT, id)
_state_path(id::String) = joinpath(_job_dir(id), "state.json")
_now_utc() = Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SSZ")

function _load_job(id::String)
    lock(JOB_LOCK) do
        file = _state_path(id)
        isdir(_job_dir(id)) || throw(ArgumentError("Unknown Aero Lab job"))
        for attempt in 1:5
            try
                parsed=_plain(JSON3.read(read(file,String)))
                parsed isa AbstractDict || error("The case state is not a JSON object")
                return parsed
            catch error
                # Older app instances may still use remove-then-rename. A short
                # retry also covers transient Windows sharing/antivirus locks.
                attempt==5 && rethrow(error)
                sleep(.01*attempt)
            end
        end
    end
end

function _save_job(state::AbstractDict)
    lock(JOB_LOCK) do
    id = _id(state["job_id"])
    mkpath(_job_dir(id))
    state["updated_utc"] = _now_utc()
    if !haskey(state,"activity_log") && haskey(state,"message")
        _append_activity!(state,state["message"])
    end
    if !haskey(state,"mesh_progress") && get(state,"status","")=="meshing"
        _mesh_progress!(state,1,"queued","Mesh request accepted")
    end
    tmp = joinpath(_job_dir(id), "state.$(uuid4()).tmp")
    try
        write(tmp, JSON3.write(state))
        for attempt in 1:5
            try
                # rename replaces the complete file atomically. mv(force=true)
                # first removes its destination and exposes a missing-state gap.
                Base.Filesystem.rename(tmp,_state_path(id))
                return state
            catch error
                attempt==5 && rethrow(error)
                sleep(.01*attempt)
            end
        end
    finally
        isfile(tmp) && rm(tmp;force=true)
    end
    end
end

function _patch_job(id::String, changes::AbstractDict)
    lock(JOB_LOCK) do
        state = _load_job(id)
        previous_status=get(state,"status",nothing);previous_message=get(state,"message",nothing)
        for (key, value) in pairs(changes)
            state[String(key)] = value
        end
        if get(state,"status",nothing)!=previous_status || get(state,"message",nothing)!=previous_message
            _append_activity!(state,get(state,"message",get(state,"status","Updated")))
        end
        get(state,"status","")=="mesh_ready" && _mesh_progress!(state,100,"ready","Mesh ready for inspection")
        get(state,"status","")=="mesh_failed" && _mesh_progress!(state,0,"failed",get(state,"message","Mesh failed"))
        get(state,"status","")=="cancelled" && _mesh_progress!(state,0,"cancelled",get(state,"message","Cancelled"))
        _save_job(state)
    end
end

_vram_fitting(case) = get(case,"mesh_sizing",
    get(case,"target_vram_gb",nothing)!==nothing || get(case,"target_vram_percent",nothing)!==nothing ? "vram" : "manual") == "vram"

"Use the admission budget, not free VRAM after this run has allocated its arrays."
function _budget_exceeded(state,actual=nothing)
    _q19_case(get(state,"case",Dict())) && return false
    estimate=get(state,"estimate",Dict{String,Any}())
    actual===nothing && get(estimate,"measured",nothing)===false && return false
    gpu=get(estimate,"gpu",Dict{String,Any}())
    get(gpu,"available",false) || return false
    budget=get(estimate,"budget_gb",nothing)
    memory=actual===nothing ? get(estimate,"memory_gb",nothing) : get(actual,"memory_gb",nothing)
    budget isa Real && memory isa Real && isfinite(budget) && isfinite(memory) || return false
    actual===nothing && _vram_fitting(state["case"]) && return false
    Float64(memory)>Float64(budget)
end

function _cancel_for_gpu_budget!(id::String,state,actual=nothing)
    state["status"] in ("meshing","sizing","mesh_sizing","waiting_for_gpu","running","queued","starting","mesh_ready") || return false
    _budget_exceeded(state,actual) || return false
    estimate=state["estimate"]
    memory=actual===nothing ? estimate["memory_gb"] : actual["memory_gb"]
    message="Above GPU budget: estimated $(round(memory,digits=2)) / $(round(estimate["budget_gb"],digits=2)) GB. Job cancelled; increase surface cell size or revise the VRAM target."
    changes=Dict{String,Any}("status"=>"cancelled","message"=>message,
        "cancel_reason"=>"gpu_budget_exceeded","memory_budget_exceeded"=>true)
    if actual!==nothing
        _apply_measured_mesh_estimate!(state,actual)
        changes["actual"]=actual;changes["estimate"]=state["estimate"]
    end
    _patch_job(id,changes)
    merge!(state,changes)
    process=lock(JOB_LOCK) do
        get(JOB_PROCESSES,id,nothing)
    end
    process===nothing || try
        kill(process)
    catch
    end
    true
end

function _gpu_memory()
    raw = try
        read(`nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits`, String)
    catch
        ""
    end
    lines = filter(x -> !isempty(x), strip.(split(strip(raw), '\n')))
    isempty(lines) && return Dict("available" => false, "free_gb" => 0.0,
                                  "total_gb" => 0.0, "budget_gb" => 0.0)
    parts = strip.(split(first(lines), ','))
    length(parts) >= 2 || return Dict("available" => false, "free_gb" => 0.0,
                                     "total_gb" => 0.0, "budget_gb" => 0.0)
    free_mb = tryparse(Float64, parts[1]); total_mb = tryparse(Float64, parts[2])
    (free_mb === nothing || total_mb === nothing) &&
        return Dict("available" => false, "free_gb" => 0.0,
                    "total_gb" => 0.0, "budget_gb" => 0.0)
    free_gb = free_mb / 1024
    Dict("available" => true, "free_gb" => free_gb, "total_gb" => total_mb / 1024,
         "budget_gb" => max(0.0, min(total_mb / 1024 - 1.0, free_gb - 0.7)))
end

"Lightweight launcher identity; deliberately does not query GPU state."
function server_info()
    _response(200,Dict("ok"=>true,"api_revision"=>WORKBENCH_API_REVISION,
        "repository_id"=>_workbench_repository_id(PUBLIC_ROOT),"max_amr_levels"=>7))
end

function capabilities()
    solvers = [Dict("id" => name, "label" => label,
                    "available" => name == "q19_octree" ? _q19_available() : (!Sys.isapple() && isfile(joinpath(root, "src", "main.jl"))),
                    "reason" => name == "q19_octree" ? "CPU-only D3Q19 lattice Boltzmann on a cell-level octree mesh; no GPU is used. Athermal flow, Mach at most 0.3. Mesh creation, CPU runs and saved result viewing are available. Development; not validated." :
                                Sys.isapple() ? "The CUDA solver is not available on macOS. Model placement and saved result viewing remain available." : name=="pond_thermal" ? "Experimental thermal finite-volume solver on the same Cartesian AMR mesh. Inviscid slip or laminar no-slip walls; no turbulence model or validated aircraft predictions." : "",
                    "placement" => name != "lbmtau",
                    "web_lattices" => name == "main" ? ["d3q27", "d3q39"] : name == "q19_octree" ? ["d3q19"] : ["d3q27"])
               for (name, label, root) in (("main", "Main LBM solver", MAIN_SOLVER),
                                            ("lbmtau", "Pinned lbmtau snapshot", PINNED_SOLVER),
                                            ("pond_thermal", "Thermal kinetic FV (experimental)", MAIN_SOLVER),
                                            ("q19_octree", "D3Q19 CPU octree (development)", Q19_SOLVER))]
    _response(200, Dict("ok" => true, "solvers" => solvers,
                        "api_revision" => WORKBENCH_API_REVISION,
                        "repository_id" => _workbench_repository_id(PUBLIC_ROOT),
                        "features" => Dict("studies"=>true,"free_air_domain_expansion"=>true,
                                           "volume_cuts"=>true,"max_amr_levels"=>7,"pond_thermal"=>true,
                                           "q19_octree"=>_q19_available(),"max_octree_levels"=>Q19_MAX_LEVELS),
                        "web_lattices" => [Dict("lattice" => "d3q27", "collision_operator" => "cumulant", "max_levels" => 7),
                                           Dict("lattice" => "d3q39", "collision_operator" => "entropic",
                                                "solver" => "main", "max_mach" => 1.6, "max_levels" => 7),
                                           Dict("lattice"=>"d3q19","collision_operator"=>"rr","solver"=>"q19_octree","max_mach"=>Q19_MAX_MACH,"max_levels"=>Q19_MAX_LEVELS,"device"=>"CPU")],
                        "gpu" => _gpu_memory(),
                        "outputs" => Dict("default_folder" => "",
                            "formats" => ["surface_vtk","surface_lbmp","volume_vtk","numeric_slices","slice_png","csv_history"],
                            "report_formats" => ["html"], "chart_formats" => ["png","svg","pdf"], "render_formats" => ["png"],
                            "note" => "Choose a destination before Run. A timestamped study folder retains the setup and native data; selected formats are grouped for export."),
                        "formats" => ["stl", "obj", "glb"],
                        "max_model_bytes" => MAX_STL_BYTES,
                        "note" => "OBJ and GLB are converted to binary STL in the browser. Cut planes are selected before a run."))
end

function _solver_command(state::AbstractDict, kind::String; case_dir_override=nothing)
    q19 = _q19_case(state["case"])
    solver_root = state["case"]["solver"] == "lbmtau" ? PINNED_SOLVER : MAIN_SOLVER
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    # the CPU octree solver is its own CUDA-free package: project and entry under src/octree_q19
    project = q19 ? Q19_SOLVER : joinpath(solver_root, "src")
    main = q19 ? joinpath(Q19_SOLVER, "main.jl") : joinpath(project, "main.jl")
    case_dir = case_dir_override === nothing ? _case_dir(state["job_id"], kind) : case_dir_override
    threads = q19 ? _q19_threads(state["case"]) : max(1, min(Threads.nthreads(), 8))
    env = Dict{String,String}(k => v for (k, v) in ENV
        if !startswith(k, "LUDWIG_") && !(k in ("JULIA_PROJECT", "JULIA_LOAD_PATH", "JULIA_NUM_THREADS")))
    (kind == "mesh" || q19) && (env["CUDA_VISIBLE_DEVICES"] = "-1")
    Cmd(`$julia --startup-file=no --project=$project -t $threads --check-bounds=no -O3 $main --case $case_dir`;
        dir=RUNTIME_ROOT, env=env)
end

"Run a build-only probe with a distinct case directory so the solver never overwrites an earlier result."
function _mesh_probe!(id::String, state::AbstractDict, case::AbstractDict, index::Int)
    probe = joinpath(CASES_ROOT, "$(id)_probe_$(index)")
    mkpath(probe)
    cp(joinpath(_case_dir(id, "mesh"), "model.stl"), joinpath(probe, "model.stl"))
    write(joinpath(probe, "config.yaml"), _yaml_case(case, case["bounds"]; build_only=true))
    cmd = _solver_command(state, "mesh"; case_dir_override=probe)
    _patch_job(id,Dict("mesh_log_path"=>joinpath(probe,"solver.log"),
        "mesh_progress_base"=>5+10*(index-1),"mesh_progress_span"=>10))
    proc = open(joinpath(probe, "solver.log"), "w") do io
        child = run(pipeline(cmd; stdout=io, stderr=io); wait=false)
        lock(JOB_LOCK) do
            JOB_PROCESSES[id] = child
        end
        wait(child)
        child
    end
    delete!(JOB_PROCESSES, id)
    _refresh_mesh_progress!(id,_load_job(id))
    _load_job(id)["status"] == "cancelled" && return nothing
    success(proc) || error("VRAM sizing probe $index failed at cell size $(case["surface_dx"]) m; see probe_$index/solver.log")
    summary_path = joinpath(probe, "RESULTS", "build_only_summary.toml")
    isfile(summary_path) || error("VRAM sizing probe $index produced no build summary")
    summary = _validate_build_summary(_read_build_summary(summary_path))
    Int(summary["num_levels"]) == case["levels"] ||
        error("VRAM sizing probe $index used $(summary["num_levels"]) levels instead of $(case["levels"]); inspect the resolved domain and native solver log")
    gpu_memory_estimate_gb(summary, _memory_lattice(case))
end

"Select the finest built mesh whose conservative GPU estimate stays under the requested target."
function _fit_mesh_to_vram!(id::String, state::AbstractDict)
    original = state["case"]
    target = Float64(original["target_vram_gb"])
    levels = Int(original["levels"])
    # Free-air domains can grow with a coarser trial mesh. Ground and symmetry
    # domains retain their prescribed physical boundaries and fixed-domain cap.
    dx_max = _mesh_dx_cap(original)
    candidate(dx) = begin
        checked = _mesh_candidate(original,dx)
        limits=_mesh_limits(checked)
        limits["valid"] || throw(ArgumentError(join(limits["reasons"]," ")))
        checked
    end
    gpu=get(state["estimate"],"gpu",_gpu_memory())
    dx = min(Float64(original["surface_dx"]), dx_max)
    try
        candidate(dx)
    catch
        seed=_coarser_mesh_choice(original,gpu,nothing)
        seed===nothing && error("No mesh retaining $levels levels fits the current CPU build limits; increase surface cell size or revise a fixed section")
        dx=Float64(seed.case["surface_dx"])
    end
    best = nothing
    under_dx = nothing
    over_dx = nothing
    for index in 1:6
        _load_job(id)["status"] == "cancelled" && return nothing
        current = try
            candidate(dx)
        catch e
            best === nothing && rethrow(e)
            break
        end
        # A bounding-box volume can overestimate a geometry-first surface mesh
        # by an order of magnitude. Valid CPU guards permit a measurement at
        # the requested dx before any GPU-budget-driven coarsening.
        _patch_job(id, Dict("message" => "VRAM sizing probe $index/6: finest cell $(round(dx,sigdigits=4)) m",
            "mesh_progress"=>Dict("percent"=>Float64(5+10*(index-1)),"phase"=>"sizing",
                "detail"=>"Measuring VRAM candidate $index/6","estimated"=>true)))
        measured = try
            _mesh_probe!(id, state, current, index)
        catch error
            next_dx=min(dx_max,dx*1.5)
            _patch_job(id,Dict("message"=>"VRAM sizing probe $index failed: $(sprint(showerror,error)). Retrying with a coarser mesh."))
            if next_dx<=dx*1.01
                best===nothing && rethrow(error)
                break
            end
            over_dx=dx;dx=next_dx
            continue
        end
        measured === nothing && return nothing
        if measured <= target
            if best === nothing || current["surface_dx"] < best.case["surface_dx"]
                best = (case=current, memory=measured)
            end
            under_dx = dx
            measured >= 0.95target && break
        else
            over_dx = dx
        end
        next_dx = if under_dx !== nothing && over_dx !== nothing
            sqrt(under_dx*over_dx)
        elseif measured > target
            # Leave measured headroom rather than approaching the target from
            # above indefinitely during the bounded six-probe search.
            dx*clamp((measured/(.85target))^0.4,1.12,1.8)
        else
            clamp(dx*(measured/target)^0.4, dx*0.65, dx*1.6)
        end
        next_dx = min(next_dx, dx_max)
        abs(next_dx/dx - 1) < 0.01 && break
        dx = next_dx
    end
    best === nothing && error("No measured mesh at the requested $(levels) levels fit the $(round(target,digits=2)) GB target within six probes; raise the VRAM target or request a coarser surface cell size")
    selected = best.case
    requested_dx=get(original,"requested_surface_dx",original["surface_dx"])
    selected["requested_surface_dx"]=requested_dx
    selected["mesh_adjustment"]=Dict("requested_surface_dx"=>requested_dx,"selected_surface_dx"=>selected["surface_dx"],
        "reason"=>"Automatic sizing used measured CPU-build counts to stay within the requested GPU budget.",
        "estimated_memory_gb"=>best.memory,"budget_gb"=>target)
    root = _case_dir(id, "mesh")
    write(joinpath(root, "config.yaml"), _yaml_case(selected, selected["bounds"]; build_only=true))
    estimate = _estimate(selected;gpu=gpu)
    # Measured active-block allocation replaces the envelope for admission.
    estimate["envelope_memory_gb"]=estimate["memory_gb"]
    estimate["memory_gb"]=best.memory;estimate["fits"]=best.memory<=target
    estimate["measured"]=true
    estimate["basis"]="measured CPU build-only active-block counts and conservative GPU per-cell allocation"
    estimate["budget_gb"] = target
    if get(selected,"target_vram_percent",nothing) !== nothing
        estimate["target_vram_percent"] = selected["target_vram_percent"]
        estimate["target_vram_requested_gb"] = get(selected,"target_vram_requested_gb",target)
        estimate["target_vram_effective_gb"] = target
    end
    _patch_job(id, Dict("case" => selected, "estimate" => estimate,
                        "message" => "Selected finest cell $(round(selected["surface_dx"],sigdigits=5)) m; estimated $(round(best.memory,digits=2)) / $(round(target,digits=2)) GB"))
    selected
end

function _run_solver(id::String, kind::String)
    state = _load_job(id)
    dir = _case_dir(id, kind)
    logpath = joinpath(dir, "solver.log")
    lock_handle = nothing
    q19 = _q19_case(state["case"])
    try
        if kind == "run" && !q19          # the CPU octree solver never takes the shared GPU lock
            mkpath(dirname(AirfoilGPULock.LOCK_DIR))
            _patch_job(id, Dict("status" => "waiting_for_gpu", "message" => "Waiting for the shared GPU"))
            deadline = time() + 180.0
            last_owner = ""
            while true
                _load_job(id)["status"] == "cancelled" && return
                try
                    lock_handle = AirfoilGPULock.acquire!("Aero Lab $id";
                        poll=1.0, report=60.0, max_wait=1.0, io=devnull)
                    break
                catch e
                    isdir(AirfoilGPULock.LOCK_DIR) || rethrow(e)
                    owner = AirfoilGPULock.holder()
                    if owner != last_owner
                        _patch_job(id, Dict("message" => "Waiting for the shared GPU (held by $owner)"))
                        last_owner = owner
                    end
                    time() <= deadline || error("GPU lock remained held for 3 minutes by $owner. Check the shared GPU_LOCK.d owner before retrying")
                end
            end
        end
        _load_job(id)["status"] == "cancelled" && return
        _patch_job(id, Dict("status" => kind == "mesh" ? "meshing" : "running",
                            "message" => kind == "mesh" ? (q19 ? "Building the octree mesh on CPU" : "Building the AMR mesh on CPU") : (q19 ? "Solver running on CPU" : "Solver running on GPU")))
        if kind == "mesh" && _vram_fitting(state["case"])
            _fit_mesh_to_vram!(id, state) === nothing && return
            state = _load_job(id)
        end
        if kind == "mesh"
            base=_vram_fitting(state["case"]) ? 75.0 : 0.0
            _patch_job(id,Dict("mesh_log_path"=>logpath,"mesh_progress_base"=>base,"mesh_progress_span"=>100-base,
                "message"=>"Building the selected mesh at $(round(state["case"]["surface_dx"],sigdigits=6)) m surface cell size"))
        end
        cmd = _solver_command(state, kind)
        proc = open(logpath, "w") do io
            child = run(pipeline(cmd; stdout=io, stderr=io); wait=false)
            lock(JOB_LOCK) do
                JOB_PROCESSES[id] = child
            end
            _patch_job(id,Dict("solver_pid"=>Int(getpid(child)),"solver_started_utc"=>_now_utc()))
            wait(child)
            child
        end
        delete!(JOB_PROCESSES, id)
        _patch_job(id,Dict("solver_exit_code"=>proc.exitcode,
            (kind=="run" ? "run_finished_utc" : "mesh_finished_utc")=>_now_utc()))
        kind=="mesh" && _refresh_mesh_progress!(id,_load_job(id))
        _load_job(id)["status"] == "cancelled" && return
        if success(proc)
            if kind == "mesh"
                actual = build_summary(id; lattice=_memory_lattice(state["case"]),allow_incomplete=false)
                actual === nothing && error("Solver exited without a build-only summary")
                Int(actual["num_levels"]) == state["case"]["levels"] ||
                    error("Built mesh has $(actual["num_levels"]) AMR levels instead of $(state["case"]["levels"]); inspect the resolved domain and solver log")
                _assess_mesh!(actual, state["case"])
                measured_state=_load_job(id)
                _apply_measured_mesh_estimate!(measured_state,actual)
                _cancel_for_gpu_budget!(id,measured_state,actual) && return
                adjustment = get(state["case"],"domain_adjustment",nothing)
                note = adjustment isa AbstractDict && get(adjustment,"expanded",false) ? " "*adjustment["message"] : ""
                _patch_job(id, Dict("status" => "mesh_ready", "message" => (q19 ? "Mesh built; inspect the domain and RAM estimate." : "Mesh built; inspect the domain and VRAM estimate.")*note,
                                    "actual" => actual,"estimate"=>measured_state["estimate"]))
            else
                if _q19_case(state["case"])
                    summary_path=joinpath(dir,"RESULTS","octree_run_summary.toml")
                    isfile(summary_path) || error("The CPU octree solver exited without its run summary; inspect solver.log")
                    summary=TOML.parsefile(summary_path)
                    get(summary,"status","")=="completed_steps" && get(summary,"steps",-1)==state["case"]["steps"] ||
                        error("The CPU octree solver did not finish the requested steps; inspect its run summary and solver.log")
                    _patch_job(id,Dict("status"=>"completed","converged"=>false,"solver_summary"=>summary,
                        "message"=>"Requested CPU octree steps completed. Development solver; convergence and validation are not established."))
                elseif _pond_case(state["case"])
                    summary_path=joinpath(dir,"RESULTS","thermal_fv_summary.toml")
                    isfile(summary_path) || error("Thermal kinetic FV exited without its final diagnostic summary; inspect solver.log")
                    summary=TOML.parsefile(summary_path)
                    get(summary,"status","")=="completed_steps" &&
                        get(summary,"steps",-1)==state["case"]["steps"] ||
                        error("Thermal kinetic FV did not finish the requested iterations; inspect its diagnostic summary and solver.log")
                    _patch_job(id,Dict("status"=>"completed","converged"=>false,
                        "solver_summary"=>summary,"message"=>"Requested thermal kinetic FV steps completed. Steady convergence was not assessed; this is an experimental inviscid/laminar result."))
                else
                    _patch_job(id, Dict("status" => "completed", "message" => "Solver completed"))
                end
            end
            try
                _sync_exports!(id;force=true)
                _launch_postprocessing!(id,kind)
            catch e
                _patch_job(id,Dict("export_warning" => sprint(showerror,e)))
            end
        else
            error("Solver exited with code $(proc.exitcode); see solver.log")
        end
    catch e
        if _load_job(id)["status"] != "cancelled"
            _patch_job(id, Dict("status" => kind == "mesh" ? "mesh_failed" : "failed",
                                "message" => sprint(showerror, e)))
        end
        @error "Aero Lab solver task failed" id kind exception=(e, catch_backtrace())
    finally
        lock(JOB_LOCK) do
            pop!(JOB_PROCESSES, id, nothing)
        end
        lock_handle === nothing || AirfoilGPULock.release!(lock_handle)
        try _sync_exports!(id;force=true) catch;end
    end
end

function _resolve_vram_target!(case,estimate,gpu)
    if _q19_case(case)
        estimate["budget_gb"]=nothing;estimate["fits"]=nothing
        return nothing
    end
    target = get(case, "target_vram_gb", nothing)
    percent = get(case,"target_vram_percent",nothing)
    if !gpu["available"] && !_vram_fitting(case)
        estimate["budget_gb"]=nothing
        estimate["fits"]=nothing
        return nothing
    end
    if percent !== nothing
        gpu["available"] || throw(ArgumentError("VRAM target sizing requires an available NVIDIA GPU"))
        requested = gpu["free_gb"]*Float64(percent)/100
        target = min(requested,gpu["budget_gb"])
        target > 0 || throw(ArgumentError("No safe GPU memory is currently available"))
        case["target_vram_gb"] = target
        case["target_vram_requested_gb"] = requested
        case["target_vram_available_gb"] = gpu["free_gb"]
        estimate["target_vram_percent"] = percent
        estimate["target_vram_requested_gb"] = requested
        estimate["target_vram_effective_gb"] = target
    end
    if target !== nothing
        gpu["available"] || throw(ArgumentError("VRAM target sizing requires an available NVIDIA GPU"))
        target <= gpu["budget_gb"] || throw(ArgumentError(
            "Target VRAM $(round(target,digits=2)) GB exceeds the current safe budget $(round(gpu["budget_gb"],digits=2)) GB"))
        estimate["budget_gb"] = target
        estimate["fits"] = _vram_fitting(case) ? nothing : estimate["memory_gb"] <= target
    end
    target
end

function create_mesh(req::HTTP.Request)
    case, triangles = _validate_case(_body(req);enforce_mesh_limits=false)
    gpu=_gpu_memory()
    target_info=Dict{String,Any}("memory_gb"=>Inf)
    target=_resolve_vram_target!(case,target_info,gpu)
    budget=gpu["available"] && !_q19_case(case) ? (target===nothing ? gpu["budget_gb"] : target) : nothing   # the CPU solver has no GPU budget
    limits=_mesh_limits(case)
    if !_vram_fitting(case) && !limits["valid"]
        proposal=_mesh_proposal(case,gpu,budget)
        return _response(422,Dict("ok"=>false,"error"=>join(limits["reasons"]," "),
            "mesh_proposal"=>proposal,"mesh_limits"=>limits))
    end
    if _vram_fitting(case)
        minimum_overhead=get(case,"lattice","d3q27")=="d3q39" ? 1.6 : .7
        if budget!==nothing && budget<=minimum_overhead
            proposal=_mesh_proposal(case,gpu,budget;reason="The VRAM target is below the solver's minimum device overhead.")
            return _response(422,Dict("ok"=>false,"error"=>proposal["reason"],"mesh_proposal"=>proposal))
        end
        seed=_coarser_mesh_choice(case,gpu,nothing)
        if seed===nothing
            proposal=_mesh_proposal(case,gpu,budget;reason="No mesh retaining the requested levels fits the build limits and GPU target.")
            return _response(422,Dict("ok"=>false,"error"=>proposal["reason"],"mesh_proposal"=>proposal))
        end
        requested=case["surface_dx"];case=seed.case
        case["requested_surface_dx"]=requested
        case["mesh_adjustment"]=Dict("requested_surface_dx"=>requested,"selected_surface_dx"=>case["surface_dx"],
            "reason"=>"The initial automatic candidate respects the CPU build limits. GPU sizing starts from requested resolution when valid and uses measured active-block counts.")
    end
    estimate = _estimate(case;gpu=gpu)
    _resolve_vram_target!(case,estimate,gpu)
    estimate["fits"]=nothing
    estimate["budget_assessment"]="pending_build"
    id = string(uuid4())
    export_directory = nothing # Persistent output is chosen and reserved at Run admission.
    mkpath(CASES_ROOT)
    mkpath(_job_dir(id))
    dir = _case_dir(id, "mesh")
    mkpath(dir)
    solver_triangles = _solver_model_triangles(case,triangles)
    _write_stl(joinpath(dir, "model.stl"), solver_triangles)
    write(joinpath(dir, "config.yaml"), _yaml_case(case, case["bounds"]; build_only=true))
    state = Dict{String,Any}(
        "job_id" => id, "status" => "meshing", "message" => "Building the AMR mesh on CPU",
        "case" => case, "estimate" => estimate, "actual" => nothing,
        "export_directory" => export_directory,
        "mesh_case_dir" => dir, "run_case_dir" => _case_dir(id, "run"),
        "created_utc" => _now_utc(),"activity_log"=>Any[])
    _append_activity!(state,"Requested surface cell size $(get(case,"requested_surface_dx",case["surface_dx"])) m; $(case["levels"]) AMR levels; $(_vram_fitting(case) ? "automatic VRAM sizing" : "manual sizing").")
    if haskey(case,"mesh_adjustment") && case["surface_dx"]!=case["requested_surface_dx"]
        _append_activity!(state,"Initial automatic candidate: $(round(case["surface_dx"],sigdigits=6)) m. "*case["mesh_adjustment"]["reason"])
    end
    if !_vram_fitting(case) && budget!==nothing && estimate["memory_gb"]>budget
        state["mesh_proposal"]=_mesh_proposal(case,gpu,budget;reason="The requested mesh exceeds the available GPU budget.")
    end
    _save_job(state)
    _cancel_for_gpu_budget!(id,state) && return _response(200,merge(state,Dict("ok"=>true)))
    @async _run_solver(id, "mesh")
    _response(202, Dict("ok" => true, "job_id" => id, "status" => "meshing",
                        "estimate" => estimate, "url" => "/api/aero/jobs/$id"))
end

function _run_memory_estimate(state, actual)
    actual === nothing && return Inf
    Float64(actual["memory_gb"])
end

function _assess_mesh!(actual, case)
    levels = get(actual, "levels", Any[])
    isempty(levels) && return actual
    finest = last(levels)
    wall = Int(get(finest, "bouzidi_boundary_cells", 0))
    dx = Float64(get(finest, "dx_m", case["surface_dx"]))
    source_area = Float64(get(case, "surface_area_m2", 0.0))
    source_triangles = Int(get(case, "triangles", 0))
    used_triangles = Int(get(actual, "geometry_triangles", source_triangles))
    area = source_area > 0 ?
        (source_triangles > 0 ? source_area * min(1.0, used_triangles/source_triangles) : source_area) : nothing
    surface_cells = area !== nothing && dx > 0 ? area / dx^2 : nothing
    density = surface_cells !== nothing && surface_cells > 0 ? wall / surface_cells : nothing
    sparse = surface_cells !== nothing && density !== nothing && surface_cells >= 100 && density < 0.05
    severe = sparse && wall < 16
    message = severe ? "Very few finest-level wall links were resolved for this surface. Reduce surface_dx, inspect STL watertightness and thin features, then rebuild before running." :
              sparse ? "Wall-link coverage is sparse at the requested surface cell size; inspect the mesh and pressure surface before trusting the result." : ""
    audit = get(case,"geometry_audit",nothing)
    ambiguous = audit isa AbstractDict && (get(audit,"nonmanifold_edges",0)>0 || get(audit,"orientation_conflicts",0)>0 ||
        (!get(audit,"outward_orientation_verified",false) && !get(audit,"symmetry_boundary_verified",false)))
    geometry_handling = _geometry_handling(get(case,"geometry_handling","closed_solid"))
    open_allowed = geometry_handling == "two_sided_open"
    geometry_blocked = ambiguous && !open_allowed
    geometry_warning = !ambiguous ? "" : open_allowed ?
        "Two-sided / intentional open surface is enabled. The geometry audit is advisory: outside orientation and solid interior are unverified. Wall intersections use both triangle sides without duplicating faces; openings and nonmanifold or intersecting faces can change the filled solid and computed forces." :
        "The geometry has open, nonmanifold or inconsistently oriented faces. Choose Solver geometry: Two-sided / intentional open surface to run knowingly with this geometry, or repair the model. A matching minimum-Y symmetry opening is accepted automatically."
    isempty(geometry_warning) || (message = isempty(message) ? geometry_warning : message*" "*geometry_warning)
    actual["finest_wall_cells"] = wall
    actual["surface_area_m2"] = area
    actual["surface_cells_estimate"] = surface_cells
    actual["wall_cells_per_surface_cell"] = density
    actual["run_blocked_by_mesh_quality"] = severe
    actual["run_blocked_by_geometry"] = geometry_blocked
    actual["geometry_handling"] = geometry_handling
    actual["geometry_warning"] = geometry_warning
    actual["geometry_override_active"] = ambiguous && open_allowed
    actual["geometry_intersections_two_sided"] = true
    actual["quality_warning"] = message
    actual["quality"] = Dict("runnable" => !severe&&!geometry_blocked, "warning" => message)
    actual
end

function _apply_run_output_updates!(state,raw)
    isempty(raw) && return state
    if haskey(raw,"reference")
        state["case"]["reference"]=_reference_dimensions(raw["reference"],state["case"]["domain"]["boundary"])
    end
    if haskey(raw,"outputs")
        state["case"]["outputs"] = _validate_outputs(raw["outputs"])
    end
    if haskey(raw,"display_every")
        every = _integer(raw["display_every"],"display_every";lo=1,hi=1_000_000)
        state["case"]["display_every"] = every
        state["case"]["effective_display_every"] = min(every,state["case"]["steps"])
    end
    if haskey(raw,"target_vram_percent")
        state["case"]["target_vram_percent"]=_number(raw["target_vram_percent"],"target_vram_percent";lo=1,hi=100)
    end
    if haskey(raw,"geometry_handling")
        state["case"]["geometry_handling"] = _geometry_handling(raw["geometry_handling"])
    end
    if haskey(raw,"source_fileinfo")
        info=raw["source_fileinfo"]
        info isa AbstractString && length(info)<=2000 && !occursin('\0',info) || throw(ArgumentError("source_fileinfo must be text containing at most 2000 characters"))
        state["source_fileinfo"]=String(info)
    end
    if haskey(raw,"cuts")
        cuts=raw["cuts"]
        cuts isa AbstractVector && length(cuts)<=6 || throw(ArgumentError("cuts must contain at most six planes"))
        parsed=Dict{String,Any}[]
        for (index,cut) in enumerate(cuts)
            cut isa AbstractDict || throw(ArgumentError("cuts[$index] must be an object"))
            axis=string(get(cut,"axis",""))
            axis in ("x","y","z") || throw(ArgumentError("cuts[$index].axis must be x, y, or z"))
            position=_number(get(cut,"position",nothing),"cuts[$index].position";lo=0,hi=1e5)
            push!(parsed,Dict("axis"=>axis,"position"=>position))
        end
        state["case"]["cuts"]=parsed
    end
    state
end

function start_run(id::String,req::Union{Nothing,HTTP.Request}=nothing)
    updates = req === nothing || isempty(req.body) ? Dict{String,Any}() : _body(req)
    lock(JOB_LOCK) do
        state = _load_job(id)
        state["status"] == "mesh_ready" || throw(ArgumentError("Build and inspect the mesh before running"))
        _apply_run_output_updates!(state,updates)
        _require_run_output_folder(state["case"])
        state["study_name"]=_study_name(get(updates,"study_name",get(state,"study_name","Aero study")))
        actual = build_summary(id; lattice=_memory_lattice(state["case"]),allow_incomplete=false)
        actual === nothing && throw(ArgumentError("Build-only summary is missing"))
        for (index,cut) in enumerate(state["case"]["cuts"])
            axis=findfirst(==(cut["axis"]),("x","y","z"))
            limit=Float64(actual["domain_size_m"][axis])
            cut["position"]<=limit+1e-8*max(1.,limit) || throw(ArgumentError("Cut plane $index lies outside the built domain on axis $(cut["axis"])"))
        end
        _assess_mesh!(actual, state["case"])
        actual["quality"]["runnable"] || throw(ArgumentError(actual["quality"]["warning"]))
        memory = _run_memory_estimate(state, actual)
        if _q19_case(state["case"])
            # the CPU octree solver runs in RAM: no GPU, no lock; refuse only what the machine cannot hold
            ram_gb = Sys.total_memory() / 1e9
            available_ram=Sys.free_memory()/1e9;ram_budget=0.8min(ram_gb,available_ram)
            memory <= ram_budget || throw(ArgumentError(
                "Estimated run memory $(round(memory,digits=2)) GB exceeds the current RAM allowance ($(round(ram_budget,digits=1)) GB, 80% of available RAM); increase surface cell size, reduce levels or close other work"))
            state["estimate"]["budget_gb"]=nothing
        else
            gpu = _gpu_memory()
            gpu["available"] || throw(ArgumentError("No NVIDIA GPU is available for this solver"))
            target = get(state["case"], "target_vram_gb", nothing)
            budget = target === nothing ? gpu["budget_gb"] : min(gpu["budget_gb"], Float64(target))
            percent = get(state["case"],"target_vram_percent",nothing)
            if percent !== nothing
                requested=gpu["free_gb"]*Float64(percent)/100
                budget=min(gpu["budget_gb"],requested)
                state["case"]["target_vram_gb"]=budget
                state["case"]["target_vram_requested_gb"]=requested
                state["case"]["target_vram_available_gb"]=gpu["free_gb"]
            end
            state["estimate"]["budget_gb"]=budget
            state["estimate"]["gpu"]=gpu
            memory <= budget || throw(ArgumentError(
                "Estimated run memory $(round(memory,digits=2)) GB exceeds the current GPU budget $(round(budget,digits=2)) GB; increase surface cell size or reduce domain/levels"))
        end
        mesh_source=_case_dir(id,"mesh")
        destination=_prepare_export_directory(id,state["case"];study_name=state["study_name"])
        mesh_dir=joinpath(destination,"native","CASES","mesh")
        _copy_case_tree!(mesh_source,mesh_dir)
        run_dir = joinpath(destination,"native","CASES","run")
        mkpath(run_dir)
        cp(joinpath(mesh_dir, "model.stl"), joinpath(run_dir, "model.stl"))
        write(joinpath(run_dir, "config.yaml"), _yaml_case(state["case"], state["case"]["bounds"]; build_only=false))
        state["export_directory"]=destination;state["mesh_case_dir"]=mesh_dir;state["run_case_dir"]=run_dir
        state["run_started_utc"]=_now_utc()
        state["solver_pid"]=nothing
        _save_job(state)
        state["study_snapshot_path"]=_write_run_study_snapshot!(state,destination;request=updates)
        _save_job(state)
        _sync_exports!(id;force=true)
        _patch_job(id, Dict("status" => _q19_case(state["case"]) ? "starting" : "waiting_for_gpu",
                            "message" => _q19_case(state["case"]) ? "Starting the CPU octree solver" : "Waiting for the shared GPU",
                            "run_memory_estimate_gb" => memory))
    end
    @async _run_solver(id, "run")
    job_response(id)
end

function cancel_job(id::String,req::Union{Nothing,HTTP.Request}=nothing)
    raw=req===nothing || isempty(req.body) ? Dict{String,Any}() : _body(req)
    reason=String(get(raw,"reason","user"))
    reason in ("user","gpu_budget_exceeded") || throw(ArgumentError("Unknown cancellation reason"))
    state = _load_job(id)
    state["status"]=="cancelled" && return job_response(id)
    state["status"] in ("meshing", "sizing", "mesh_sizing", "waiting_for_gpu", "running", "queued", "starting") ||
        throw(ArgumentError("Job is not active"))
    changes=Dict{String,Any}("status"=>"cancelled","cancel_reason"=>reason,
        "message"=>reason=="gpu_budget_exceeded" ? "Above GPU budget. Job cancelled; increase surface cell size or revise the VRAM target." : "Cancelled by user")
    reason=="gpu_budget_exceeded" && (changes["memory_budget_exceeded"]=true)
    _patch_job(id,changes)
    proc = lock(JOB_LOCK) do
        get(JOB_PROCESSES, id, nothing)
    end
    proc === nothing || try
        kill(proc)
    catch
    end
    job_response(id)
end

function _progress(id::String, state)
    total = state["case"]["steps"]
    history=_run_history(id)
    step=maximum(Int(round(get(row,"Step",0))) for row in (history.convergence,history.resources,history.forces))
    Dict("step" => step, "total" => total,
         "percent" => min(100.0, 100.0*step/max(total,1)))
end

function job_response(id::String)
    state = _load_job(id)
    _refresh_mesh_progress!(id,state)
    if !haskey(state,"mesh_progress")
        ready=state["status"] in ("mesh_ready","completed","running","waiting_for_gpu")
        _mesh_progress!(state,ready ? 100 : 1,ready ? "ready" : "queued",ready ? "Mesh ready" : get(state,"message","Preparing mesh"))
    end
    haskey(state,"activity_log") || (state["activity_log"]=Any[])
    try
        _sync_exports!(id)
    catch error
        state["export_warning"] = sprint(showerror,error)
    end
    actual = try
        build_summary(id; lattice=_memory_lattice(state["case"]))
    catch error
        error isa ArgumentError || rethrow()
        # Inspection must remain usable even if a completed output is damaged.
        # Run admission still reads and validates the summary strictly.
        state["inspection_error"]=sprint(showerror,error)
        state["message"]=state["inspection_error"]
        state["status"]=="mesh_ready" && (state["status"]="mesh_failed")
        nothing
    end
    if actual===nothing && state["status"] in ("meshing","sizing","mesh_sizing")
        previous=get(state,"actual",nothing)
        if previous isa AbstractDict && get(previous,"num_levels",nothing)==state["case"]["levels"]
            actual=previous
        end
        state["inspection_pending"]=true
    end
    actual === nothing || _assess_mesh!(actual, state["case"])
    actual === nothing || _apply_measured_mesh_estimate!(state,actual)
    _cancel_for_gpu_budget!(id,state,actual)
    state["actual"] = actual
    state["progress"] = _progress(id, state)
    state["monitoring"] = try
        _run_monitoring(id,state)
    catch error
        # Hardware telemetry is optional; it must not hide a completed field or
        # turn an already-admitted solver launch into an HTTP failure.
        Dict{String,Any}("available"=>false,"error"=>sprint(showerror,error),
            "iteration"=>state["progress"]["step"],"total_iterations"=>state["case"]["steps"],
            "progress_percent"=>state["progress"]["percent"],"phase"=>Dict("key"=>state["status"],"label"=>state["status"],"detail"=>"Live resource monitoring is temporarily unavailable"))
    end
    state["artifacts"] = discover_artifacts(id)
    state["exports"] = _export_response(id,state)
    state["ok"] = true
    _response(200, state)
end
