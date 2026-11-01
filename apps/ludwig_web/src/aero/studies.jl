"""Portable Aero studies: complete editable setup, embedded geometry, and inert result references.

Saved studies deliberately live outside TEMP_DELETEABLE. Result fields are pointers:
import never reads a supplied result path or starts/resumes a solver automatically.
"""
const STUDIES_ROOT = joinpath(PRIVATE_ROOT, "studies", "aero")
const STUDY_SCHEMA = "org.ludwig.aero-study"
const STUDY_VERSION = 1
const MAX_STUDY_BYTES = 2*4*cld(MAX_STL_BYTES,3) + 2_000_000
const STUDY_SETUP_KEYS = ("solver","lattice","transform","domain","surface_dx","levels","flow","steps","cuts")
const RECENT_STUDIES_FILE = joinpath(PRIVATE_ROOT,"studies","recent_studies.json")
const RECENT_STUDIES_LOCK = ReentrantLock()
const MAX_RECENT_STUDIES = 48

"Native files known to this app; result or study imports cannot supply arbitrary open paths."
function _recent_study_path(value)
    value isa AbstractString || return nothing
    path=String(value)
    isabspath(path) && length(path)<=2000 && !occursin('\0',path) || return nothing
    startswith(replace(path,'\\'=>'/'),"//") && return nothing
    lowercase(splitext(path)[2])==".ludwig-study" && isfile(path) && !islink(path) || return nothing
    realpath(path)
end

_recent_study_id(path)=bytes2hex(sha256(Sys.iswindows() ? lowercase(path) : path))

function _read_recent_studies()
    isfile(RECENT_STUDIES_FILE) || return Any[]
    try
        filesize(RECENT_STUDIES_FILE)<=256_000 || return Any[]
        raw=_plain(JSON3.read(read(RECENT_STUDIES_FILE,String)))
        raw isa AbstractDict && get(raw,"version",nothing)==1 || return Any[]
        [item for item in get(raw,"studies",Any[]) if item isa AbstractDict && get(item,"id",nothing) isa AbstractString]
    catch
        Any[]
    end
end

function _write_recent_studies(entries)
    mkpath(dirname(RECENT_STUDIES_FILE))
    temporary=RECENT_STUDIES_FILE*".$(uuid4()).tmp"
    try
        write(temporary,JSON3.write(Dict("version"=>1,"studies"=>entries)))
        Base.Filesystem.rename(temporary,RECENT_STUDIES_FILE)
    finally
        isfile(temporary) && rm(temporary)
    end
end

function _remember_study(path;name=nothing,results=nothing,kind=nothing,used=time())
    canonical=_recent_study_path(path)
    canonical===nothing && return nothing
    lock(RECENT_STUDIES_LOCK) do
        id=_recent_study_id(canonical)
        entries=_read_recent_studies()
        previous=findfirst(item->get(item,"id",nothing)==id,entries)
        item=previous===nothing ? Dict{String,Any}() : Dict{String,Any}(entries[previous])
        item["id"]=id;item["path"]=canonical;item["last_used_unix"]=used
        kind===nothing ? get!(item,"kind","study") : (item["kind"]=kind)
        name===nothing || (item["name"]=String(name))
        get!(item,"name",splitext(basename(canonical))[1])
        results===nothing || (item["results_hint"]=get(results,"available",false) ? "Results linked" :
            get(results,"source_job_id",nothing)!==nothing ? "Saved result references" : "Setup and geometry")
        get!(item,"results_hint","Setup and geometry")
        filter!(entry->get(entry,"id",nothing)!=id,entries);push!(entries,item)
        sort!(entries;by=entry->get(entry,"last_used_unix",0.),rev=true)
        _write_recent_studies(first(entries,min(length(entries),MAX_RECENT_STUDIES)))
        item
    end
end

"Seed the initial picker from named studies, without traversing disposable work trees."
function _seed_recent_studies!()
    root=joinpath(PRIVATE_ROOT,"studies");isdir(root) || return
    known=Set(get(item,"id","") for item in _read_recent_studies())
    candidates=String[];folders=[(root,0)];visited=0
    while !isempty(folders) && visited<100 && length(candidates)<500
        directory,depth=popfirst!(folders);visited+=1
        paths=try readdir(directory;join=true) catch; String[] end
        for path in paths
            islink(path) && continue
            if isdir(path)
                depth<2 && push!(folders,(path,depth+1))
            elseif lowercase(splitext(path)[2])==".ludwig-study"
                # UUID retained/download copies are implementation history, not user destinations.
                occursin(r"^[0-9a-f-]{36}\.ludwig-study$"i,basename(path)) && continue
                push!(candidates,path);length(candidates)>=500 && break
            end
        end
    end
    sort!(candidates;by=path->try stat(path).mtime catch; 0. end,rev=true)
    metadata_budget=64_000_000
    for path in first(candidates,min(length(candidates),MAX_RECENT_STUDIES))
        canonical=_recent_study_path(path);canonical===nothing && continue
        _recent_study_id(canonical) in known && continue
        title=splitext(basename(path))[1]
        lowercase(title) in ("aero study","study") && dirname(path)!=root && (title=basename(dirname(path))*" / "*title)
        refs=nothing
        # JSON3 objects retain lazy geometry strings; do not materialize the embedded model.
        # Bound the first-install metadata work even if many large studies are present.
        size=try filesize(canonical) catch; continue end
        if size<=min(metadata_budget,32_000_000)
            metadata_budget-=size
            try
                document=JSON3.read(read(canonical,String))
                get(document,"schema",nothing)==STUDY_SCHEMA || continue
                saved_name=_study_name(get(document,"name",title))
                lowercase(saved_name) in ("aero study","study") || (title=saved_name)
                value=get(document,"results",nothing)
                value isa AbstractDict && (refs=Dict("available"=>get(value,"available",false),"source_job_id"=>get(value,"source_job_id",nothing)))
            catch
                continue
            end
        end
        _remember_study(canonical;name=title,results=refs,used=stat(canonical).mtime)
    end
end

function recent_studies()
    lock(RECENT_STUDIES_LOCK) do
        _seed_recent_studies!()
        entries=Any[]
        for item in _read_recent_studies()
            path=_recent_study_path(get(item,"path",nothing));path===nothing && continue
            get(item,"id",nothing)==_recent_study_id(path) || continue
            information=try stat(path) catch; continue end
            push!(entries,merge(item,Dict("path"=>path,"modified_utc"=>string(unix2datetime(information.mtime))*"Z",
                "size_bytes"=>information.size)))
        end
        sort!(entries;by=item->get(item,"last_used_unix",0.),rev=true)
        _response(200,Dict("ok"=>true,"studies"=>entries))
    end
end

function open_recent_study(req::HTTP.Request)
    NativeDialogs._validate_request(req)
    raw=_body(req);id=get(raw,"id",nothing)
    id isa AbstractString && occursin(r"^[0-9a-f]{64}$",id) || throw(ArgumentError("Choose a study from the recent list"))
    item=lock(RECENT_STUDIES_LOCK) do
        entries=_read_recent_studies();index=findfirst(entry->get(entry,"id",nothing)==id,entries)
        index===nothing ? nothing : entries[index]
    end
    item===nothing && throw(ArgumentError("This study is no longer in the recent list; use Browse other study"))
    path=_recent_study_path(get(item,"path",nothing))
    path!==nothing && _recent_study_id(path)==id || throw(ArgumentError("This study has moved or was deleted; use Browse other study"))
    _open_study_path(path)
end

function _check_study_setup(setup)
    setup isa AbstractDict || throw(ArgumentError("Study must contain a complete case object"))
    encoded=try JSON3.write(setup) catch
        throw(ArgumentError("Study setup must contain finite JSON-compatible values"))
    end
    ncodeunits(encoded)<=1_000_000 ||
        throw(ArgumentError("Study must contain a complete case object smaller than 1 MB"))
    _id(get(setup,"model_id",""))
    for key in STUDY_SETUP_KEYS
        haskey(setup,key) || throw(ArgumentError("Study setup is missing $key"))
    end
    for (section,keys) in (("transform",("scale","rotation_deg","translation")),
                           ("domain",("boundary","upstream","downstream","lateral","height")),
                           ("flow",("velocity","density","kinematic_viscosity","speed_of_sound")))
        value=setup[section]
        value isa AbstractDict && all(key->haskey(value,key),keys) ||
            throw(ArgumentError("Study setup has an incomplete $section section"))
    end
    setup["cuts"] isa AbstractVector || throw(ArgumentError("Study cuts must be an array"))
    get(setup["domain"],"symmetry_center_y",false) isa Bool ||
        throw(ArgumentError("Study domain.symmetry_center_y must be true or false"))
    _domain_sizing(get(setup["domain"],"sizing","minimum"))
    _domain_proportions(get(setup["domain"],"proportions",[1.,1.,1.]))
    _geometry_handling(get(setup,"geometry_handling","closed_solid"))
    haskey(setup,"surface_triangle_max_edge_m") &&
        _number(setup["surface_triangle_max_edge_m"],"Surface triangle edge length";lo=1e-4,hi=100.)
    setup
end

function _study_body(req::HTTP.Request)
    length(req.body) <= MAX_STUDY_BYTES || throw(ArgumentError("Study exceeds the 136 MB file limit"))
    parsed = try _plain(JSON3.read(String(req.body))) catch
        throw(ArgumentError("Study must be a valid JSON object"))
    end
    parsed isa AbstractDict || throw(ArgumentError("Study must be a JSON object"))
    parsed
end

function _study_name(value)
    value isa AbstractString || throw(ArgumentError("Study name must be text"))
    name = strip(String(value))
    1 <= length(name) <= 160 || throw(ArgumentError("Study name must contain 1 to 160 characters"))
    all(c -> !iscntrl(c), name) || throw(ArgumentError("Study name must not contain control characters"))
    name
end

function _study_filename(name)
    stem = replace(String(name), r"[^A-Za-z0-9_. -]" => "_")
    stem = strip(stem, [' ', '.'])
    isempty(stem) && (stem = "Aero study")
    first(stem,min(length(stem),100))*".ludwig-study"
end

function _study_geometry_name(value, fallback)
    value isa AbstractString || throw(ArgumentError("Geometry file name must be text"))
    name = String(value)
    # A file name is metadata, never a path used for import writes.
    name = last(split(replace(name,'\\'=>'/'),'/'))
    name = replace(name, r"[\x00-\x1f<>:\"/\\|?*]" => "_")
    name = strip(name,[' ','.'])
    isempty(name) ? fallback : first(name,min(length(name),180))
end

function _geometry_document(bytes::Vector{UInt8}, name::AbstractString, media_type::AbstractString)
    Dict{String,Any}("name"=>name,"media_type"=>media_type,"encoding"=>"base64",
        "size_bytes"=>length(bytes),"sha256"=>bytes2hex(sha256(bytes)),"data"=>base64encode(bytes))
end

function _decode_study_geometry(value; canonical=false)
    value isa AbstractDict || throw(ArgumentError("Study geometry must be an object"))
    get(value,"encoding",nothing)=="base64" || throw(ArgumentError("Study geometry must use base64 encoding"))
    data = get(value,"data",nothing)
    data isa AbstractString && ncodeunits(data)<=4*cld(MAX_STL_BYTES,3)+4 ||
        throw(ArgumentError("Each embedded geometry file must be at most 50 MB"))
    bytes = try base64decode(data) catch
        throw(ArgumentError("Study geometry contains invalid base64"))
    end
    0 < length(bytes) <= MAX_STL_BYTES || throw(ArgumentError("Each embedded geometry file must contain 1 byte to 50 MB"))
    digest = bytes2hex(sha256(bytes))
    haskey(value,"sha256") && lowercase(string(value["sha256"])) != digest &&
        throw(ArgumentError("Study geometry checksum does not match its content"))
    haskey(value,"size_bytes") && value["size_bytes"] != length(bytes) &&
        throw(ArgumentError("Study geometry length does not match its metadata"))
    canonical && _stl_triangles(bytes)
    name = _study_geometry_name(get(value,"name",canonical ? "model.stl" : "source_geometry"),"source_geometry")
    media = string(get(value,"media_type",canonical ? "model/stl" : "application/octet-stream"))
    ncodeunits(media)<=160 && !occursin(r"[\r\n]",media) || throw(ArgumentError("Invalid geometry media type"))
    bytes, _geometry_document(bytes,name,media)
end

"The optional centering flag was absent from legacy studies; omission means false."
function _same_study_domain(a,b)
    a isa AbstractDict && b isa AbstractDict || return a==b
    normalized(value)=begin
        result=merge(Dict{String,Any}(String(k)=>v for (k,v) in pairs(value)),
            Dict{String,Any}("symmetry_center_y"=>get(value,"symmetry_center_y",false),
                "sizing"=>get(value,"sizing","minimum")))
        result["proportions"]=result["sizing"]=="proportions" ? _effective_domain_proportions(value) : [1.,1.,1.]
        result
    end
    left=normalized(a);right=normalized(b)
    ratios_left=pop!(left,"proportions");ratios_right=pop!(right,"proportions")
    left==right && isapprox(ratios_left,ratios_right;rtol=1e-12,atol=1e-14)
end

function _same_study_setup(a::AbstractDict,b::AbstractDict)
    get(a,"surface_triangle_max_edge_m",0.0)==get(b,"surface_triangle_max_edge_m",0.0) &&
    _geometry_handling(get(a,"geometry_handling","closed_solid")) ==
        _geometry_handling(get(b,"geometry_handling","closed_solid")) &&
    all(key -> key=="cuts" ? true : key=="domain" ?
        (_same_study_domain(get(a,key,nothing),get(b,key,nothing)) ||
         _same_study_domain(get(a,"requested_domain",nothing),get(b,key,nothing))) :
        key=="flow" ? merge(Dict("alpha"=>0.,"beta"=>0.),get(a,key,Dict()))==merge(Dict("alpha"=>0.,"beta"=>0.),get(b,key,Dict())) :
        get(a,key,nothing)==get(b,key,nothing), STUDY_SETUP_KEYS)
end

function _study_results(job_id, setup, canonical_bytes)
    job_id === nothing && return nothing
    id = _id(job_id)
    isfile(_state_path(id)) || return Dict("source_job_id"=>id,"compatible"=>false,"available"=>false,"locations"=>Any[])
    job = _load_job(id)
    job_case = get(job,"case",Dict{String,Any}())
    model_file = joinpath(MODEL_ROOT,string(get(job_case,"model_id","")),"source.stl")
    compatible = _same_study_setup(job_case,setup) && isfile(model_file) && sha256(read(model_file))==sha256(canonical_bytes)
    locations = Any[]
    for kind in ("mesh","run")
        directory = _case_dir(id,kind)
        isdir(directory) && push!(locations,Dict("kind"=>kind,"path"=>directory,"available"=>true))
    end
    destination = get(job,"export_directory",nothing)
    destination isa AbstractString && isdir(destination) &&
        push!(locations,Dict("kind"=>"exports","path"=>destination,"available"=>true))
    planes_match=get(job_case,"cuts",Any[])==get(setup,"cuts",Any[])
    Dict{String,Any}("source_job_id"=>id,"compatible"=>compatible,"recorded_planes_match"=>planes_match,"available"=>!isempty(locations),
        "status"=>get(job,"status","unknown"),"updated_utc"=>get(job,"updated_utc",nothing),"locations"=>locations,
        "note"=>planes_match ? "Result locations are references; result files are not embedded in this study." :
            "Surface and recorded fields remain valid for this model and flow. The requested recording planes changed; use the available recorded planes, a saved volume field, or record the new planes in a future run.")
end

function _persist_study(document::AbstractDict)
    id = string(uuid4())
    mkpath(STUDIES_ROOT)
    path = joinpath(STUDIES_ROOT,id*".ludwig-study")
    # UUID names make each save immutable; names and embedded file names are metadata only.
    write(path,JSON3.write(document))
    id,path
end

function _study_save_target(raw,name)
    folder=get(raw,"folder",nothing)
    folder===nothing && return nothing
    folder isa AbstractString && !isempty(strip(folder)) || throw(ArgumentError("Choose a study folder"))
    directory=_output_folder(String(folder);create=true)
    file=get(raw,"file_name",_study_filename(name))
    file isa AbstractString || throw(ArgumentError("Study file name must be text"))
    file=strip(String(file))
    1<=length(file)<=180 && !occursin(r"[\x00-\x1f<>:\"/\\|?*]",file) && !endswith(file,".") ||
        throw(ArgumentError("Choose a file name without path separators or reserved characters"))
    lowercase(splitext(file)[2])==".ludwig-study" || (file*=".ludwig-study")
    occursin(r"^(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)",file) && throw(ArgumentError("This file name is reserved by Windows"))
    target=joinpath(directory,file)
    islink(target) && throw(ArgumentError("The chosen study file is a symbolic link"))
    ispath(target) && !isfile(target) && throw(ArgumentError("The chosen study path is not a regular file"))
    if isfile(target)
        digest=bytes2hex(sha256(read(target)))
        if get(raw,"overwrite",false)!==true || get(raw,"overwrite_sha256",nothing)!=digest
            return _response(409,Dict("ok"=>false,"error"=>"A study file already exists at the chosen location.",
                "path"=>target,"existing_sha256"=>digest,"file_name"=>file))
        end
    end
    target
end

function _write_selected_study(target,document;overwrite=false)
    temporary=target*".$(uuid4()).tmp"
    try
        write(temporary,JSON3.write(document))
        !overwrite && ispath(target) && throw(ArgumentError("The chosen study file was created while saving; choose another name"))
        Base.Filesystem.rename(temporary,target)
    finally
        isfile(temporary) && rm(temporary)
    end
    target
end

function save_study(req::HTTP.Request)
    raw = _study_body(req)
    setup = _check_study_setup(get(raw,"case",nothing))
    model_id = _id(get(setup,"model_id",""))
    _load_model(model_id) # audits legacy models and verifies that this model exists
    canonical_bytes = read(joinpath(MODEL_ROOT,model_id,"source.stl"))
    name = _study_name(get(raw,"name","Aero study"))
    target=_study_save_target(raw,name)
    target isa HTTP.Response && return target
    view = get(raw,"view",Dict{String,Any}())
    view isa AbstractDict && ncodeunits(JSON3.write(view))<=500_000 ||
        throw(ArgumentError("Study view must be an object smaller than 500 kB"))
    original = if haskey(raw,"source_geometry") && raw["source_geometry"] !== nothing
        last(_decode_study_geometry(raw["source_geometry"]))
    else
        preserved_source=joinpath(MODEL_ROOT,model_id,"study_source_geometry.json")
        original_file=joinpath(MODEL_ROOT,model_id,"original.stl")
        display_source=_stored_model_source_geometry(model_id)
        isfile(preserved_source) ? _plain(JSON3.read(read(preserved_source,String))) :
            display_source!==nothing ? display_source :
            isfile(original_file) ? _geometry_document(read(original_file),"original.stl","model/stl") : nothing
    end
    results = _study_results(get(raw,"job_id",nothing),setup,canonical_bytes)
    if results === nothing && get(raw,"results",nothing) !== nothing
        results = first(_import_result_references(raw["results"],setup,canonical_bytes))
    end
    document = Dict{String,Any}("schema"=>STUDY_SCHEMA,"version"=>STUDY_VERSION,"workbench"=>"aero",
        "name"=>name,"saved_utc"=>_now_utc(),"case"=>_plain(setup),"view"=>_plain(view),
        "geometry"=>_geometry_document(canonical_bytes,"model.stl","model/stl"),
        "source_geometry"=>original,"results"=>results)
    target===nothing || _write_selected_study(target,document;overwrite=get(raw,"overwrite",false)===true)
    id,path = _persist_study(document)
    target===nothing || _remember_study(target;name,results)
    _response(201,Dict("ok"=>true,"study_id"=>id,"name"=>name,"file_name"=>target===nothing ? _study_filename(name) : basename(target),
        "url"=>"/api/aero/studies/$id/file","storage_path"=>path,"saved_path"=>target,"folder"=>target===nothing ? nothing : dirname(target),
        "saved_sha256"=>target===nothing ? nothing : bytes2hex(sha256(read(target)))))
end

function study_file(id::String)
    path=joinpath(STUDIES_ROOT,_id(id)*".ludwig-study")
    isfile(path) || return _bad("Unknown study",404)
    document=_plain(JSON3.read(read(path,String)))
    HTTP.Response(200,["Content-Type"=>"application/json; charset=utf-8","Cache-Control"=>"no-store",
        "X-Content-Type-Options"=>"nosniff","Content-Disposition"=>"attachment; filename=\"$(_study_filename(document["name"]))\""],read(path))
end

function _import_result_references(value, setup, canonical_bytes)
    value === nothing && return nothing,nothing
    value isa AbstractDict || throw(ArgumentError("Study results must be an object or null"))
    # Do not read, copy or publish arbitrary paths from an imported document.
    refs = _plain(value)
    ncodeunits(JSON3.write(refs))<=500_000 || throw(ArgumentError("Study result references exceed 500 kB"))
    refs["available"]=false
    refs["compatible"]=false
    refs["locations"] = [merge(_plain(item),Dict("available"=>false)) for item in get(refs,"locations",Any[]) if item isa AbstractDict]
    id=get(refs,"source_job_id",nothing)
    local_id = id isa AbstractString ? (try _id(id) catch; nothing end) : nothing
    local_job = local_id !== nothing && isfile(_state_path(local_id)) ? local_id : nothing
    if local_job !== nothing
        known = _study_results(local_job,setup,canonical_bytes)
        known !== nothing && get(known,"compatible",false) && return known,local_job
    end
    if local_job===nothing && local_id!==nothing
        recovered=_restore_persistent_study_results(refs,setup,canonical_bytes)
        recovered===nothing || return recovered,local_id
    end
    refs["note"]="Result references are unavailable or belong to a different setup on this computer; build a new mesh to run this study."
    refs,nothing
end

function _restore_persistent_study_results(refs,setup,canonical_bytes)
    id=_id(refs["source_job_id"]);digest=bytes2hex(sha256(canonical_bytes))
    candidates=String[]
    locations=get(refs,"locations",Any[])
    locations isa AbstractVector || return nothing
    for item in Iterators.take(locations,32)
        item isa AbstractDict || continue
        path=get(item,"path",nothing);path isa AbstractString && isabspath(path) || continue
        # Portable documents cannot silently initiate network-share access.
        startswith(replace(path,'\\'=>'/'),"//") && continue
        current=normpath(path)
        for _ in 0:4
            push!(candidates,joinpath(current,"run_manifest.json"))
            parent=dirname(current);parent==current && break;current=parent
        end
    end
    for file in unique(candidates)
        try
            isfile(file) && !islink(file) && filesize(file)<8_000_000 || continue
            manifest=_plain(JSON3.read(read(file,String)))
            get(manifest,"schema","")=="org.ludwig.aero-run" && get(manifest,"version",0)==1 &&
                get(manifest,"job_id",nothing)==id && get(manifest,"canonical_geometry_sha256",nothing)==digest || continue
            previous=manifest["case"]
            _same_study_setup(previous,setup) || continue
            root=realpath(dirname(file));run=String(manifest["run_case_dir"]);mesh=String(manifest["mesh_case_dir"])
            inside(path)=begin
                isdir(path) || return false
                relative=relpath(realpath(path),root)
                !isabspath(relative) && all(part->part!="..",splitpath(relative))
            end
            inside(run) && inside(mesh) || continue
            previous=deepcopy(previous);previous["model_id"]=setup["model_id"]
            status=String(get(manifest,"status","interrupted"))
            status in ("completed","failed","cancelled") || (status="interrupted")
            state=Dict{String,Any}("job_id"=>id,"status"=>status,"message"=>"Restored saved result files from the study output folder.",
                "case"=>previous,"estimate"=>Dict("gpu"=>Dict("available"=>false),"fits"=>nothing),
                "mesh_case_dir"=>mesh,"run_case_dir"=>run,"export_directory"=>root,
                "study_name"=>get(manifest,"study_name","Aero study"),"study_geometry_sha256"=>digest,
                "study_snapshot_path"=>get(manifest,"study_file",nothing),"created_utc"=>get(manifest,"updated_utc",_now_utc()))
            _save_job(state)
            return _study_results(id,setup,canonical_bytes)
        catch
            # A foreign/missing reference must not prevent opening its portable setup.
        end
    end
    nothing
end

function import_study(req::HTTP.Request;source_path=nothing,source_sha256=nothing)
    raw = _study_body(req)
    get(raw,"schema",nothing)==STUDY_SCHEMA && get(raw,"version",nothing)==STUDY_VERSION && get(raw,"workbench",nothing)=="aero" ||
        throw(ArgumentError("Unsupported study format or version"))
    name = _study_name(get(raw,"name","Aero study"))
    setup=_check_study_setup(get(raw,"case",nothing))
    view=get(raw,"view",Dict{String,Any}())
    view isa AbstractDict && ncodeunits(JSON3.write(view))<=500_000 || throw(ArgumentError("Study view must be an object smaller than 500 kB"))
    canonical_bytes,canonical = _decode_study_geometry(get(raw,"geometry",nothing);canonical=true)
    source = get(raw,"source_geometry",nothing)
    source === nothing || (source=last(_decode_study_geometry(source)))
    # The ordinary upload path applies the current winding audit to portable geometry.
    uploaded=upload_model(HTTP.Request("POST","/api/aero/models",[],canonical_bytes))
    model=_plain(JSON3.read(String(uploaded.body)))
    setup=_plain(setup);setup["model_id"]=model["model_id"]
    results,resume_id = _import_result_references(get(raw,"results",nothing),setup,canonical_bytes)
    # Resume is offered only for a locally verified, identical setup. The returned
    # editable study case remains the saved setup rather than the job's current case.
    document=Dict{String,Any}("schema"=>STUDY_SCHEMA,"version"=>STUDY_VERSION,"workbench"=>"aero",
        "name"=>name,"saved_utc"=>get(raw,"saved_utc",_now_utc()),"imported_utc"=>_now_utc(),
        "case"=>setup,"view"=>_plain(view),"geometry"=>canonical,"source_geometry"=>source,"results"=>results)
    id,path=_persist_study(document)
    # Retain the optional original file without using its supplied name as a path.
    if source !== nothing
        write(joinpath(MODEL_ROOT,model["model_id"],"study_source_geometry.json"),JSON3.write(source))
    end
    _response(201,Dict("ok"=>true,"study_id"=>id,"name"=>name,"file_name"=>_study_filename(name),
        "url"=>"/api/aero/studies/$id/file","storage_path"=>path,"case"=>setup,"view"=>_plain(view),
        "model"=>model,"source_geometry"=>source===nothing ? nothing : Dict(k=>v for (k,v) in source if k!="data"),
        "results"=>results,"job_id"=>resume_id,"source_path"=>source_path,"source_sha256"=>source_sha256))
end

function _open_study_path(path)
    filesize(path)<=MAX_STUDY_BYTES || throw(ArgumentError("Study exceeds the 136 MB file limit"));bytes=read(path)
    response=import_study(HTTP.Request("POST","/api/aero/studies/import",[],copy(bytes));
        source_path=path,source_sha256=bytes2hex(sha256(bytes)))
    loaded=_plain(JSON3.read(String(copy(response.body))))
    _remember_study(path;name=get(loaded,"name",nothing),results=get(loaded,"results",nothing))
    response
end

function open_study_native(req::HTTP.Request;chooser=NativeDialogs.choose_file)
    selected=chooser(req,joinpath(PRIVATE_ROOT,"TEMP_DELETEABLE","native_dialogs");mode="open")
    get(selected,"cancelled",false) && return _response(200,selected)
    _open_study_path(String(selected["path"]))
end

function choose_study_save_path(req::HTTP.Request;chooser=NativeDialogs.choose_file)
    selected=chooser(req,joinpath(PRIVATE_ROOT,"TEMP_DELETEABLE","native_dialogs");mode="save")
    get(selected,"cancelled",false) && return _response(200,selected)
    path=String(selected["path"])
    _response(200,Dict("ok"=>true,"cancelled"=>false,"path"=>path,"folder"=>dirname(path),"file_name"=>basename(path),
        "existing_sha256"=>isfile(path) ? bytes2hex(sha256(read(path))) : nothing))
end

"Capture the exact model/setup/view and persistent result locations before a Run."
function _write_run_study_snapshot!(state,destination;request=Dict())
    setup=deepcopy(state["case"])
    metadata=get(request,"study_snapshot",Dict{String,Any}())
    metadata isa AbstractDict || throw(ArgumentError("study_snapshot must be an object"))
    view=get(metadata,"view",Dict{String,Any}())
    view isa AbstractDict && ncodeunits(JSON3.write(view))<=500_000 || throw(ArgumentError("Study view is too large"))
    name=_study_name(get(state,"study_name",get(metadata,"name","Aero study")))
    canonical=read(joinpath(MODEL_ROOT,setup["model_id"],"source.stl"))
    state["study_geometry_sha256"]=bytes2hex(sha256(canonical))
    refs=_study_results(state["job_id"],setup,canonical)
    document=Dict{String,Any}("schema"=>STUDY_SCHEMA,"version"=>STUDY_VERSION,"workbench"=>"aero",
        "name"=>name,"saved_utc"=>_now_utc(),"case"=>setup,"view"=>_plain(view),
        "geometry"=>_geometry_document(canonical,"model.stl","model/stl"),
        "source_geometry"=>_stored_model_source_geometry(setup["model_id"]),"results"=>refs)
    path=joinpath(destination,_export_study_stem(name)*".ludwig-study")
    _write_selected_study(path,document;overwrite=false)
    _remember_study(path;name,results=refs,kind="run snapshot")
    path
end

"Update a run-owned study once its solver reaches a terminal state."
function _refresh_run_study_snapshot!(id,state,destination)
    status=get(state,"status","")
    status in ("completed","failed","cancelled") || return
    path=get(state,"study_snapshot_path",nothing)
    path isa AbstractString && isfile(path) && !islink(path) || return
    # Never rewrite a user-associated file or a path outside this run directory.
    realpath(dirname(path))==realpath(destination) || return
    filesize(path)<=MAX_STUDY_BYTES || throw(ArgumentError("The run study snapshot exceeds its file limit"))
    document=_plain(JSON3.read(read(path,String)))
    get(document,"schema",nothing)==STUDY_SCHEMA && get(document,"version",nothing)==STUDY_VERSION || return
    previous=get(document,"results",nothing)
    previous isa AbstractDict && get(previous,"source_job_id",nothing)==id || return
    get(previous,"status",nothing)==status && return
    setup=get(document,"case",nothing)
    setup isa AbstractDict && _same_study_setup(state["case"],setup) || return
    canonical,_=_decode_study_geometry(document["geometry"];canonical=true)
    digest=bytes2hex(sha256(canonical))
    digest==get(state,"study_geometry_sha256",nothing) || return
    refs=_study_results(id,setup,canonical)
    refs isa AbstractDict || return
    document["results"]=refs;document["results_updated_utc"]=_now_utc()
    _write_selected_study(path,document;overwrite=true)
    _remember_study(path;name=get(document,"name",nothing),results=refs,kind="run snapshot")
end
