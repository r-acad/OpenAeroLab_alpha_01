using Serialization,TOML

"API-only checked initial guess; exact native continuation remains a separate driver operation."
function _lowspeed_initial_id(flow,backends)
    haskey(flow,"lowspeed_initial_job_id") || return nothing
    "lowspeed" in backends || throw(ArgumentError("lowspeed_initial_job_id requires the low-speed CFD backend"))
    value=flow["lowspeed_initial_job_id"]
    value isa AbstractString || throw(ArgumentError("lowspeed_initial_job_id must be a saved job UUID"))
    id=try string(UUID(value)) catch;throw(ArgumentError("lowspeed_initial_job_id must be a saved job UUID"));end
    lowercase(String(value))==id || throw(ArgumentError("lowspeed_initial_job_id must be a canonical job UUID"))
    id
end

function _initial_confined(path,root)
    ispath(path) && isdir(root) || throw(ArgumentError("Saved low-speed initial-flow file is missing"))
    file=realpath(path);base=realpath(root)
    canonical(p)=Sys.iswindows() ? lowercase(p) : p
    startswith(canonical(file),canonical(normpath(base))*string(Sys.iswindows() ? '\\' : '/')) ||
        throw(ArgumentError("Saved low-speed initial flow escapes its job directory"))
    file
end

function _initial_same_airfoil(a,b)
    get(a,"kind",nothing)==get(b,"kind",nothing) || return false
    if haskey(a,"coordinates") || haskey(b,"coordinates")
        return haskey(a,"coordinates")&&haskey(b,"coordinates")&&a["coordinates"]==b["coordinates"]
    end
    get(a,"id",nothing)==get(b,"id",nothing)
end

function _lowspeed_initial_source(case)
    id=_lowspeed_initial_id(case["flow"],case["backends"]);id===nothing && return nothing
    job=_get_job(id);job===nothing && (job=_saved_job(id))
    job===nothing && throw(ArgumentError("Initial-flow job $id was not found"))
    get(job,"status","") in ("completed","partial","failed","cancelled") ||
        throw(ArgumentError("Initial-flow job must be terminal before its saved state can be used"))
    request=haskey(job,"case") ? _job_request(job["case"]) : get(job,"request",nothing)
    request isa AbstractDict && "lowspeed" in get(request,"backends",String[]) ||
        throw(ArgumentError("Initial-flow job has no low-speed CFD request"))
    _initial_same_airfoil(case["airfoil"],get(request,"airfoil",Dict())) ||
        throw(ArgumentError("Initial-flow job uses a different source airfoil"))
    for flow in (case["flow"],get(request,"flow",Dict()))
        get(flow,"cfd_streamline_align",false)===false ||
            throw(ArgumentError("Saved initial flow is not supported with development streamline alignment"))
    end
    current=CFDBackend.lowspeed_configuration(case["flow"])
    previous=CFDBackend.lowspeed_configuration(get(request,"flow",Dict()))
    for key in ("reynolds","tu","mut_ratio","resolution")
        current[key]==previous[key] || throw(ArgumentError("Initial-flow job has incompatible $key"))
    end
    root=_safe_job_dir(id)
    isdir(root) || throw(ArgumentError("Initial-flow job directory is missing"))
    registered=AirfoilResultLocations.registered_directory(RUN_DIR,id)
    allowed=registered!==nothing && normpath(registered)==normpath(root) ||
        any(base->isdir(base)&&try _initial_confined(root,base);true catch;false end,(RUN_DIR,LEGACY_RUN_DIR))
    allowed || throw(ArgumentError("Initial-flow job is outside saved result storage"))
    lowdir=joinpath(root,"lowspeed")
    isdir(lowdir) || throw(ArgumentError("Initial-flow job has no low-speed result directory"))
    _initial_confined(lowdir,root)
    points=String[]
    for name in sort(readdir(lowdir))
        occursin(r"^\d{3}_alpha_[+-]\d+\.\d+$",name) || continue
        dir=joinpath(lowdir,name);isdir(dir)||continue
        _initial_confined(dir,root)
        all(file->isfile(joinpath(dir,file)),("state_checkpoint.jls","input.toml","result.toml")) || continue
        for file in ("state_checkpoint.jls","input.toml","result.toml");_initial_confined(joinpath(dir,file),root);end
        input=TOML.parsefile(joinpath(dir,"input.toml"));angle=get(input,"alpha",nothing)
        angle isa Real && isfinite(angle) && abs(Float64(angle)-first(case["alphas"]))<=2.5 || continue
        push!(points,dir)
    end
    isempty(points) && throw(ArgumentError("Initial-flow job has no complete saved checkpoint within 2.5 degrees of the first angle"))
    sort!(points;by=dir->(abs(Float64(TOML.parsefile(joinpath(dir,"input.toml"))["alpha"])-first(case["alphas"])),basename(dir)))
    (;id,root,points)
end

function _initial_loaded_source(snap)
    io=IOBuffer();dir=joinpath(snap,"src","pbcav")
    for file in sort!(filter(f->endswith(f,".jl"),readdir(dir)))
        write(io,file);write(io,read(joinpath(dir,file)))
    end
    bytes2hex(sha256(take!(io)))
end

function _lowspeed_initial_target(case,path,outdir)
    flow=CFDBackend._lowspeed_resolve_flow(case["airfoil"],path,case["flow"])
    alpha=first(case["alphas"]);dir=joinpath(outdir,"lowspeed",@sprintf("%03d_alpha_%+.3f",1,alpha));mkpath(dir)
    ch=CFDBackend._lowspeed_ch_selected(flow)
    if ch
        ls=CFDBackend.lowspeed_ch_configuration(flow)
        snap,head=CFDBackend._lowspeed_ch_snapshot_check(baffle=haskey(ls,"baffle"),refine=haskey(get(ls,"baffle",Dict()),"refine"))
        foil,_,gap=CFDBackend._lowspeed_ch_geometry(case,path,dir,flow)
        knobs=CFDBackend._lowspeed_ch_knobs(ls,gap)
        input=CFDBackend._lowspeed_ch_input(snap,head,foil,ls,knobs,alpha,dir)
    else
        snap,head=CFDBackend._lowspeed_snapshot_check();ls=CFDBackend.lowspeed_configuration(flow)
        foil,_=CFDBackend._lowspeed_geometry(case,path,dir,snap)
        input=CFDBackend._lowspeed_input(snap,head,foil,ls,alpha,dir)
    end
    (;ch,ls,input,head,source=_initial_loaded_source(snap),foil_sha=bytes2hex(open(sha256,foil)))
end

function _check_lowspeed_initial_point(dir,target,alpha)
    input=TOML.parsefile(joinpath(dir,"input.toml"))
    result=TOML.parsefile(joinpath(dir,"result.toml"))
    input["commit"]==target.head || throw(ArgumentError("solver revision differs"))
    get(result,"source_sha256",nothing)==target.source || throw(ArgumentError("loaded solver source differs"))
    input["resolution"]==target.input["resolution"] || throw(ArgumentError("mesh resolution differs"))
    get(input,"streamline",nothing)===nothing || throw(ArgumentError("streamline alignment differs"))
    oldknobs=get(input,"knobs",Dict());newknobs=target.input["knobs"]
    meshkey(key)=startswith(key,"PB_MESH")||startswith(key,"PB_BAFFLE")||startswith(key,"PB_FARFIELD")||startswith(key,"PB_CHANNEL")
    for key in union(filter(meshkey,collect(keys(oldknobs))),filter(meshkey,collect(keys(newknobs))))
        get(oldknobs,key,nothing)==get(newknobs,key,nothing) || throw(ArgumentError("mesh or baffle setting $key differs"))
    end
    if !target.ch
        for key in ("mesh","le_ds");get(input,key,nothing)==get(target.input,key,nothing)||throw(ArgumentError("mesh $key differs"));end
    end
    # This file is only deserialized after resolving a terminal generated job and
    # confining every path to its result directory. No user-supplied file path is accepted.
    saved=open(deserialize,joinpath(dir,"state_checkpoint.jls"))
    saved isa AbstractDict || throw(ArgumentError("saved state is not a checkpoint map"))
    get(saved,"commit",nothing)==target.head && get(saved,"source_sha256",nothing)==target.source ||
        throw(ArgumentError("checkpoint solver source differs"))
    get(saved,"topology",nothing)==(target.ch ? "conformal_C_H_v1" : nothing) || throw(ArgumentError("checkpoint topology differs"))
    saved["foil_sha256"]==target.foil_sha || throw(ArgumentError("CFD airfoil coordinates differ"))
    get(saved,"baffle","")==get(newknobs,"PB_BAFFLE","") || throw(ArgumentError("checkpoint baffle differs"))
    for (key,value) in (("reynolds",target.ls["reynolds"]),("tu",target.ls["tu"]),("mut_ratio",target.ls["mut_ratio"]))
        get(saved,key,nothing)==value || throw(ArgumentError("checkpoint $key differs"))
    end
    saved["alpha_deg"] isa Real && isfinite(saved["alpha_deg"]) && abs(saved["alpha_deg"]-alpha)<=2.5 ||
        throw(ArgumentError("checkpoint incidence differs by more than 2.5 degrees"))
    all(key->get(saved,key,nothing) isa Real && isfinite(saved[key]),("gamma","dt","dt_prev")) && saved["dt"]>0 && saved["dt_prev"]>0 ||
        throw(ArgumentError("checkpoint timestep or circulation is invalid"))
    for key in ("ux","uy","p","rho","q","rk","rw","mut","F","Fb")
        data=get(saved,key,nothing)
        data isa Array{Float64,3} && size(data,2)==1 && size(data,3)==3 && size(data,1)>0 && all(isfinite,data) ||
            throw(ArgumentError("checkpoint state $key is invalid"))
    end
    all(key->size(saved[key])==size(saved["rho"]),("ux","uy","p","q","rk","rw","mut")) ||
        throw(ArgumentError("checkpoint cell-state sizes differ"))
    all(>(0),saved["rho"])&&all(iszero,saved["q"])&&all(key->all(>=(0),saved[key]),("rk","rw","mut")) ||
        throw(ArgumentError("checkpoint physical state is invalid"))
    # Worker still checks exact generated C/H mesh hashes and all destination sizes.
    for key in ("mesh_hashes",target.ch ? "strip_hashes" : "mesh_hashes")
        get(saved,key,nothing) isa AbstractVector && all(x->x isa AbstractString,saved[key]) || throw(ArgumentError("checkpoint mesh receipt is missing"))
    end
    Float64(saved["alpha_deg"])
end

function _resolve_lowspeed_initial(case,path,outdir)
    source=_lowspeed_initial_source(case);source===nothing && return nothing
    target=_lowspeed_initial_target(case,path,outdir);reasons=String[]
    for dir in source.points
        angle=try _check_lowspeed_initial_point(dir,target,first(case["alphas"])) catch err
            push!(reasons,"$(basename(dir)): $(sprint(showerror,err))");continue
        end
        checkpoint=joinpath(dir,"state_checkpoint.jls")
        provenance=Dict{String,Any}("kind"=>"checked_initial_flow","job_id"=>source.id,"source_alpha_deg"=>angle,
            "checkpoint_sha256"=>bytes2hex(open(sha256,checkpoint)),"source_sha256"=>target.source,
            "checkpoint"=>"/api/jobs/$(source.id)/files/lowspeed/$(basename(dir))/state_checkpoint.jls",
            "meaning"=>"Checked initial flow only: the worker resets pseudo-time and step count. This is not exact native continuation.")
        return (;dir,provenance)
    end
    throw(ArgumentError("Initial-flow job has no compatible checkpoint: "*join(reasons,"; ")))
end
