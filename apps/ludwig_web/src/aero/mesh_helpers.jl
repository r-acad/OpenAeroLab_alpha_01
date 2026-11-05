const MESH_CELL_LIMIT=30_000_000.0
const MESH_ACTIVITY_LIMIT=80

function _apply_measured_mesh_estimate!(state,actual)
    estimate=get!(state,"estimate",Dict{String,Any}())
    if get(estimate,"measured",nothing)!==true && haskey(estimate,"memory_gb")
        estimate["envelope_memory_gb"]=estimate["memory_gb"]
    end
    estimate["memory_gb"]=Float64(actual["memory_gb"])
    estimate["measured"]=true
    estimate["budget_assessment"]="measured_build"
    estimate["basis"]=get(actual,"memory_basis","measured active-block counts and conservative GPU per-cell allocation")
    budget=get(estimate,"budget_gb",nothing)
    estimate["fits"]=budget isa Real ? estimate["memory_gb"]<=budget : nothing
    state
end

"Check allocations before integer grid counts or a CPU mesh process are created."
function _mesh_limits(case; available_ram_bytes=Sys.free_memory())
    ext=Float64.(case["bounds"]["max"]).-Float64.(case["bounds"]["min"])
    body=_resolved_body_extent(case["domain"],ext)
    lengths=_domain_lengths(case["domain"],ext)
    dx=Float64(case["surface_dx"]);levels=Int(case["levels"]);h0=dx*2.0^(levels-1)
    coarse=prod(8.0*_stable_grid_block_count(length,h0,8;minimum_blocks=4) for length in lengths)
    fine=_q19_case(case) ? _q19_cells_estimate(case,0,get(case,"surface_area_m2",0.0),dx,levels) : levels==1 ? 0.0 : prod(body./dx .+ 16)
    pyramid=levels==1 || all(_roundoff_snap_grid_integer(length/(32h0))>=1 for length in lengths)
    reasons=String[]
    pyramid || push!(reasons,"The fixed domain cannot retain $levels AMR levels at this cell size.")
    coarse<=MESH_CELL_LIMIT || push!(reasons,"The coarse lattice exceeds 30 million cells.")
    result=Dict{String,Any}("coarse_cells"=>coarse,
        "fine_envelope_cells"=>fine,"pyramid_fits"=>pyramid)
    if _q19_case(case)
        # A CPU octree is not subject to the GPU mesher's fixed fine-cell cap.
        # Account for its dense refinement masks as well as the surface and wake
        # storage. This is an admission estimate, not a replacement for the native
        # measured run allowance checked again immediately before Run.
        o=get(case,"octree",_q19_settings(Dict{String,Any}()))
        wake_level=max(1,levels-o["wake_level_offset"])
        wake_volume=prod(min.(lengths,[o["wake_length"]*body[1],1.2body[2],1.2body[3]]))
        wake_cells=sum((wake_volume/(dx*2.0^(levels-l))^3 for l in 2:wake_level);init=0.)
        mask_bytes=coarse*sum(8.0^k for k in 0:levels-1)/8
        # Construction stores topology, marking work and serialized mesh copies;
        # it does not allocate the population buffers used by a solver run.
        # Two mask pyramids cover refinement/balancing scratch. The separate
        # native run estimate includes populations and coupling buffers.
        build_bytes=256*(coarse+fine+wake_cells)+2mask_bytes+512_000_000
        budget=.8min(Float64(Sys.total_memory()),Float64(available_ram_bytes))
        result["build_memory_estimate_gb"]=build_bytes/1e9
        result["build_ram_budget_gb"]=budget/1e9
        result["dense_mask_bytes"]=mask_bytes
        build_bytes<=budget || push!(reasons,"The CPU octree build allowance ($(round(build_bytes/1e9,digits=2)) GB, including refinement masks and wake) exceeds 80% of available RAM ($(round(budget/1e9,digits=2)) GB). Increase surface cell size or reduce domain/levels.")
    else
        fine<=MESH_CELL_LIMIT || push!(reasons,"The conservative near-body fine-cell envelope exceeds 30 million cells.")
    end
    result["valid"]=isempty(reasons);result["reasons"]=reasons
    result
end

function _mesh_candidate(original,dx;levels=Int(original["levels"]))
    current=deepcopy(original)
    current["surface_dx"]=Float64(dx);current["levels"]=levels
    current["domain"]=deepcopy(get(original,"requested_domain",original["domain"]))
    ext=Float64.(current["bounds"]["max"]).-Float64.(current["bounds"]["min"])
    tx=get(get(current,"transform",Dict()),"translation",[0.,0.,0.])[1]
    adjustment=_expand_minimum_domain!(current["domain"],ext,dx,levels;translation_x=tx)
    current["domain_adjustment"]=adjustment
    current["requested_domain"]=adjustment["requested_domain"]
    current
end

function _mesh_dx_cap(case,levels=Int(case["levels"]))
    _domain_sizing(get(case["domain"],"sizing","minimum"))!="fixed" && return 100.0
    levels==1 && return 100.0
    ext=Float64.(case["bounds"]["max"]).-Float64.(case["bounds"]["min"])
    min(100.0,.999*minimum(_domain_lengths(case["domain"],ext))/(32*2.0^(levels-1)))
end

"Find the nearest coarser feasible request, preserving physical geometry and prescribed boundaries."
function _coarser_mesh_choice(original,gpu,budget;allow_fewer_levels::Bool=false)
    allow_fewer_levels &= _domain_sizing(get(original["domain"],"sizing","minimum"))=="fixed"
    requested=Float64(original["surface_dx"])
    maxlevels=Int(original["levels"])
    for levels in (allow_fewer_levels ? (maxlevels:-1:1) : (maxlevels:maxlevels))
        cap=_mesh_dx_cap(original,levels)
        requested<=cap || continue
        function feasible(dx)
            candidate=_mesh_candidate(original,dx;levels=levels)
            _mesh_limits(candidate)["valid"] || return nothing
            estimate=_estimate(candidate;gpu=gpu)
            budget!==nothing && estimate["memory_gb"]>budget && return nothing
            (case=candidate,estimate=estimate)
        end
        first=feasible(requested)
        first===nothing || return first
        lo=requested;hi=min(cap,max(requested*1.5,1e-5));best=feasible(hi)
        while best===nothing && hi<cap
            lo=hi;hi=min(cap,hi*1.5);best=feasible(hi)
        end
        best===nothing && continue
        for _ in 1:36
            mid=sqrt(lo*hi);trial=feasible(mid)
            if trial===nothing;lo=mid;else;hi=mid;best=trial;end
        end
        # Keep a little distance from integer/block and roundoff thresholds.
        return something(feasible(min(cap,hi*1.002)),best)
    end
    nothing
end

function _mesh_proposal(case,gpu,budget;reason=nothing)
    choice=_coarser_mesh_choice(case,gpu,budget;allow_fewer_levels=true)
    requested=Float64(case["surface_dx"])
    why=reason===nothing ? join(_mesh_limits(case)["reasons"]," ") : String(reason)
    if choice===nothing
        return Dict{String,Any}("available"=>false,"reason"=>why,"requested_surface_dx"=>requested,
            "budget_gb"=>budget,"message"=>_q19_case(case) ? "No coarser octree mesh preserving this geometry and domain fits the current build limits. Increase surface cell size or revise fixed domain dimensions; this CPU solver uses RAM, not VRAM." : "No coarser mesh preserving this geometry and domain fits the current limits and GPU budget. Raise the VRAM target, reduce the requested domain, or select another device.")
    end
    proposed=choice.case;dx=proposed["surface_dx"]
    message="Suggested surface cell size $(round(dx,sigdigits=6)) m ($(case["levels"]) → $(proposed["levels"]) AMR levels); estimated $(round(choice.estimate["memory_gb"],digits=2)) GB"
    budget===nothing || (message*=" / $(round(budget,digits=2)) GB target")
    message*=". The model scale and shape are preserved. "
    message*=_domain_sizing(get(case["domain"],"sizing","minimum"))!="fixed" ?
        "All requested AMR levels are retained; physical clearances remain minima and far faces may grow. " :
        "The fixed section clearances are preserved; native block snapping is reported after building. "
    message*=_q19_case(case) ? "Apply this proposal, then inspect the CPU RAM estimate." : "Apply this proposal or choose automatic VRAM sizing."
    Dict{String,Any}("available"=>true,"reason"=>why,"requested_surface_dx"=>requested,
        "suggested_surface_dx"=>dx,"suggested_levels"=>proposed["levels"],
        "estimated_memory_gb"=>choice.estimate["memory_gb"],"budget_gb"=>budget,
        "case"=>proposed,"estimate"=>choice.estimate,
        "changes"=>Dict("surface_dx"=>dx,"levels"=>proposed["levels"]),"message"=>message)
end

function _append_activity!(state,message;status=get(state,"status","unknown"),percent=nothing)
    entries=get!(state,"activity_log",Any[])
    text=first(String(message),min(length(String(message)),1200))
    if !isempty(entries) && get(last(entries),"message","")==text && get(last(entries),"status","")==status
        return state
    end
    seq=isempty(entries) ? 1 : Int(get(last(entries),"seq",0))+1
    entry=Dict{String,Any}("seq"=>seq,"at"=>_now_utc(),"status"=>status,"message"=>text)
    percent===nothing || (entry["percent"]=percent)
    push!(entries,entry)
    length(entries)>MESH_ACTIVITY_LIMIT && deleteat!(entries,1:length(entries)-MESH_ACTIVITY_LIMIT)
    state
end

function _mesh_progress!(state,percent,phase,detail)
    previous=get(state,"mesh_progress",Dict{String,Any}())
    value=clamp(max(Float64(get(previous,"percent",0)),Float64(percent)),0,100)
    state["mesh_progress"]=Dict("percent"=>round(value,digits=1),"phase"=>phase,"detail"=>detail,"estimated"=>true)
    state
end

function _mesh_log_milestone(line,levels)
    # the CPU octree solver's build-only log
    occursin("[BuildOnly] LUDWIG D3Q19 CPU octree",line) && return (5.,"compile","Starting the CPU octree mesh generator")
    occursin("[BuildOnly] geometry:",line) && return (10.,"geometry","Geometry placed; refining the octree around it")
    m=match(r"\[BuildOnly\] L(\d+): dx=",line)
    m===nothing || return (15+70*parse(Int,m[1])/max(levels,1),"level_build","Octree level $(m[1])/$levels counted")
    occursin("[BuildOnly] mesh built in",line) && return (88.,"export","Octree built; writing mesh.oct and the inspection data")
    occursin("[SliceData]",line) && return (96.,"export","Level and obstacle planes written")
    occursin("[BuildOnly] done in",line) && return (99.,"summary","Mesh counts and memory summary written")
    occursin("[BuildOnly] basic.simulation.build_only",line) && return (5.,"compile","Initializing the CPU-only native geometry builder")
    occursin("[Domain] Loading captured STL",line) && return (8.,"geometry","Loading the immutable geometry for this build")
    m=match(r"--- Level (\d+) ---",line)
    m===nothing || return (12+43*(parse(Int,m[1])-1)/max(levels,1),"level_build","Building AMR level $(m[1])/$levels")
    m=match(r"\[Mesh\] Rebuilding level (\d+)",line)
    m===nothing || return (60+23*parse(Int,m[1])/max(levels,1),"topology_rebuild","Rebuilding AMR level $(m[1])/$levels after topology closure")
    occursin("[Setup] Multi-Level",line) && return (10.,"domain","Preparing the multilevel domain")
    occursin("[Standoff]",line) && return (58.,"surface_shell","Checking the near-surface refinement shell")
    (occursin("[DonorHalo]",line)||occursin("[Grading]",line)) && return (60.,"topology","Closing parent support and AMR grading")
    occursin("[Verify]",line) && return (88.,"verify","Verifying parent coverage and AMR level grading")
    occursin("[BuildOnly] domain build done",line) && return (91.,"export","Domain built; exporting inspection data")
    occursin("[AMR Viz] Wrote",line) && return (94.,"export","AMR inspection mesh exported")
    (occursin("[BuildOnly]",line)&&occursin("summary:",line)) && return (99.,"summary","Mesh counts and allocation summary written")
    nothing
end

"Consume bounded new solver-log text. Percentages identify completed stages, not elapsed-time predictions."
function _refresh_mesh_progress!(id,state)
    lock(JOB_LOCK) do
        _consume_mesh_progress!(id,state)
    end
end

function _consume_mesh_progress!(id,state)
    # A poll must never write an older meshing state over a terminal transition.
    fresh=_load_job(id);empty!(state);merge!(state,fresh)
    state["status"] in ("meshing","sizing","mesh_sizing") || return state
    logpath=get(state,"mesh_log_path",joinpath(_case_dir(id,"mesh"),"solver.log"))
    isfile(logpath) || return state
    count=filesize(logpath)
    cursor=get(state,"mesh_log_cursor_path","")==logpath ? Int(get(state,"mesh_log_cursor",0)) : 0
    cursor=0<=cursor<=count ? cursor : 0
    count==cursor && return state
    start=max(cursor,count-65_536)
    chunk=open(logpath,"r") do io;seek(io,start);String(read(io));end
    lines=split(chunk,'\n');start>cursor && !isempty(lines) && popfirst!(lines)
    levels=Int(state["case"]["levels"])
    base=Float64(get(state,"mesh_progress_base",0));span=Float64(get(state,"mesh_progress_span",100))
    for line in lines
        milestone=_mesh_log_milestone(line,levels)
        milestone===nothing && continue
        percent,phase,detail=milestone
        _mesh_progress!(state,base+span*percent/100,phase,detail)
        _append_activity!(state,detail;percent=state["mesh_progress"]["percent"])
    end
    state["mesh_log_cursor_path"]=logpath;state["mesh_log_cursor"]=count
    _save_job(state)
    state
end
