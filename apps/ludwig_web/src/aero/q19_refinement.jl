# Experimental staged restart admission. This is not live, flux-conservative
# AMR: the solver transfers a completed checkpoint to a separately built mesh.

"Read and cross-check the fixed header of an atomically published Q19 checkpoint."
function _q19_checkpoint_step(path::AbstractString)
    isfile(path) && !islink(path) || throw(ArgumentError("The source checkpoint is not a regular file"))
    open(path,"r") do io
        String(read(io,8)) == "OCTQ19C\0" || throw(ArgumentError("Invalid source checkpoint magic"))
        read(io,Int64) == 1 || throw(ArgumentError("Unsupported source checkpoint version"))
        read(io,Int64) in (4,8) || throw(ArgumentError("Invalid source checkpoint precision"))
        length(read(io,128)) == 128 || throw(ArgumentError("Truncated source checkpoint"))
        step=read(io,Int64)
        step>=0 || throw(ArgumentError("Invalid source checkpoint step"))
        Int(step)
    end
end

function _q19_refinement_source(id::String, state::AbstractDict, target_actual::AbstractDict, updates::AbstractDict;
                                load_job::Function=_load_job, available_ram::Real=Sys.free_memory(), total_ram::Real=Sys.total_memory())
    source_id=get(updates,"refinement_source_job_id",nothing)
    source_id===nothing && return nothing
    _q19_case(state["case"]) || throw(ArgumentError("Refinement restart is available only for D3Q19"))
    get(updates,"accept_nonconservative_restart",false) === true || throw(ArgumentError("Acknowledge that the experimental refinement transfer is not exactly conservative"))
    source_id=_id(source_id);source_id != id || throw(ArgumentError("Refinement source and target must be different jobs"))
    source=load_job(source_id)
    get(source,"status",nothing)=="completed" || throw(ArgumentError("The source stage must complete before a refinement restart"))
    scase=get(source,"case",Dict());_q19_case(scase) || throw(ArgumentError("The source stage must use D3Q19"))
    sactual=get(source,"actual",nothing);sactual isa AbstractDict || throw(ArgumentError("Source mesh summary is missing"))
    _require_current_q19_mesh(scase,sactual)
    td=state["case"]; sn=Int(sactual["num_levels"]);tn=Int(target_actual["num_levels"])
    tn>sn || throw(ArgumentError("A refinement target needs more levels than its source"))
    Int(td["steps"])>Int(scase["steps"]) || throw(ArgumentError("Target step limit must exceed the completed source step"))
    for key in ("flow","transform","domain")
        get(scase,key,nothing)==get(td,key,nothing) || throw(ArgumentError("Source and target $key settings must match"))
    end
    for key in ("domain_size_m","mesh_offset_m")
        a=get(sactual,key,nothing);b=get(target_actual,key,nothing)
        a isa AbstractVector && b isa AbstractVector && length(a)==3 && length(b)==3 && all(isapprox.(Float64.(a),Float64.(b);rtol=0,atol=1e-9)) ||
            throw(ArgumentError("Source and target $key must match"))
    end
    sd=Float64(sactual["levels"][1]["dx_m"]);tdx=Float64(target_actual["levels"][1]["dx_m"])
    isapprox(sd,tdx;rtol=1e-12,atol=0) || throw(ArgumentError("Source and target coarse cell sizes must match"))
    source_dir=get(source,"run_case_dir",nothing)
    source_dir isa AbstractString && isabspath(source_dir) && isdir(source_dir) && !islink(source_dir) || throw(ArgumentError("Source run directory is unavailable"))
    source_dir=realpath(source_dir)
    isfile(joinpath(source_dir,"config.yaml")) && isfile(joinpath(source_dir,"model.stl")) || throw(ArgumentError("Source case files are missing"))
    source_config=read(joinpath(source_dir,"config.yaml"),String)
    reference_match=match(r"(?m)^  reference_length_for_meshing:\s*([^\s#]+)",source_config)
    reference_match===nothing && throw(ArgumentError("Source case does not record its meshing reference length"))
    source_reference=tryparse(Float64,reference_match.captures[1])
    target_reference=get(get(td,"octree",Dict()),"stage_reference_length_m",nothing)
    source_reference isa Real && isfinite(source_reference) && source_reference>0 && target_reference isa Real && isfinite(target_reference) &&
        isapprox(source_reference,target_reference;rtol=1e-12,atol=0) || throw(ArgumentError("Target must retain the source meshing reference length and physical sponge thickness"))
    mesh_dir=get(source,"mesh_case_dir",nothing)
    mesh_dir isa AbstractString && isfile(joinpath(mesh_dir,"mesh.oct")) || throw(ArgumentError("Source mesh is missing"))
    target_model=joinpath(_case_dir(id,"mesh"),"model.stl")
    isfile(target_model) && sha256(read(target_model))==sha256(read(joinpath(source_dir,"model.stl"))) || throw(ArgumentError("Source and target geometry must match"))
    ckdir=joinpath(source_dir,"RESULTS","checkpoints")
    isdir(ckdir) || throw(ArgumentError("Source stage has no checkpoint directory"))
    files=sort(filter(f->occursin(r"^checkpoint_\d{9}\.bin$",f),readdir(ckdir)))
    isempty(files) && throw(ArgumentError("Source stage has no published checkpoint"))
    ck=joinpath(ckdir,files[end]);step=_q19_checkpoint_step(ck)
    step==parse(Int,files[end][12:20])==Int(scase["steps"]) || throw(ArgumentError("Source checkpoint is not the completed stage step"))
    smem=get(sactual,"estimated_run_bytes_total",nothing);tmem=get(target_actual,"estimated_run_bytes_total",nothing)
    smem isa Real && tmem isa Real && all(isfinite,(smem,tmem,available_ram,total_ram)) && smem>0 && tmem>0 || throw(ArgumentError("Both stage memory estimates must be available"))
    reserve=max(8e9,.15total_ram);budget=min(.8available_ram,available_ram-reserve)
    smem+tmem<=budget || throw(ArgumentError("Insufficient RAM for both stage states and the required reserve"))
    Dict{String,Any}("source_job_id"=>source_id,"source_case_dir"=>source_dir,"source_checkpoint"=>realpath(ck),"source_step"=>step,
        "source_checkpoint_bytes"=>filesize(ck),"source_checkpoint_mtime"=>stat(ck).mtime,"source_run_estimate_bytes"=>smem,"target_run_estimate_bytes"=>tmem,
        "stage_reference_length_m"=>source_reference,"accepted_nonconservative"=>true,"note"=>"Experimental startup transfer; geometric fluid volume and mass may change. This is not conservative dynamic AMR.")
end
