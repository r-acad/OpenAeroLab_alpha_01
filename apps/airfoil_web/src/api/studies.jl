# Airfoil study documents: setup and source coordinates, with references to
# separately stored results. Opening a document never schedules a solver.
const AIRFOIL_STUDY_SCHEMA = "org.ludwig.airfoil-study"
const AIRFOIL_STUDY_DIR = joinpath(PRIVATE_ROOT,"studies","Airfoil")
const AIRFOIL_STUDY_INDEX = joinpath(AIRFOIL_STUDY_DIR,"recent_studies.json")
const AIRFOIL_STUDY_LOCK = ReentrantLock()
const AIRFOIL_STUDY_LIMIT = 8*1024*1024
struct AirfoilStudyConflict <: Exception
    message::String
end
Base.showerror(io::IO,e::AirfoilStudyConflict) = print(io,e.message)
_study_json(code,data) = HTTP.Response(code,["Content-Type"=>"application/json; charset=utf-8",
    "Cache-Control"=>"no-store"],JSON3.write(_clean(data)))
_study_hash(path) = isfile(path) ? bytes2hex(sha256(read(path))) : nothing

function _study_path(value; existing=false)
    value isa AbstractString && 1 <= length(value) <= 2000 && !occursin('\0',value) ||
        throw(ArgumentError("Choose a study file"))
    isabspath(value) || throw(ArgumentError("A study needs an absolute file path"))
    path = normpath(String(value))
    endswith(lowercase(path),".ludwig-study") || throw(ArgumentError("Use a .ludwig-study filename"))
    existing && !isfile(path) && throw(ArgumentError("Study file is no longer available: $path"))
    path
end

function _study_input(req)
    startswith(lowercase(_header(req,"Content-Type")),"application/json") ||
        throw(ArgumentError("Study actions require JSON"))
    origin = _header(req,"Origin")
    if !isempty(origin)
        uri = HTTP.URI(origin)
        uri.scheme == "http" && lowercase(String(uri.host)) in ("localhost","127.0.0.1","::1") &&
            lowercase(origin) == "http://"*lowercase(_header(req,"Host")) ||
            throw(ArgumentError("Use study actions from the local application"))
    end
    length(req.body) <= AIRFOIL_STUDY_LIMIT || throw(ArgumentError("Study is too large (8 MB limit)"))
    data = JSON3.read(String(copy(req.body)),Dict{String,Any})
    data isa AbstractDict || throw(ArgumentError("Study action must be an object"))
    data
end

function _study_atomic_write(path,data)
    mkpath(dirname(path))
    CFDBackend.FullFieldArtifact.FieldSnapshots.atomic_write(path,
        Vector{UInt8}(codeunits(JSON3.write(_clean(data))));reserve_bytes=0)
end

function _study_recent()
    isfile(AIRFOIL_STUDY_INDEX) || return Any[]
    try
        items = JSON3.read(read(AIRFOIL_STUDY_INDEX,String),Vector{Dict{String,Any}})
        [merge(item,Dict("available"=>isfile(String(get(item,"path",""))))) for item in items]
    catch
        Any[]
    end
end
function _study_remember(path,document)
    items = filter(item->normpath(String(get(item,"path",""))) != path,_study_recent())
    pushfirst!(items,Dict("path"=>path,"name"=>document["name"],"airfoil"=>document["request"]["airfoil"]["id"],
        "opened_at"=>string(now(UTC))*"Z"))
    _study_atomic_write(AIRFOIL_STUDY_INDEX,items[1:min(end,30)])
end

function _study_result_link(value)
    value === nothing && return nothing
    value isa AbstractDict || throw(ArgumentError("Invalid study result link"))
    id = String(get(value,"job_id",""))
    occursin(r"^[0-9a-f-]{36}$",id) || throw(ArgumentError("Invalid study result job id"))
    directory = String(get(value,"directory",""))
    length(directory)<=2000 && !occursin('\0',directory) || throw(ArgumentError("Invalid results location"))
    # Pointers are descriptive; an imported file cannot grant arbitrary file access.
    Dict("job_id"=>id,"directory"=>directory,"url"=>"/api/jobs/$id")
end

function _study_validate_document(document)
    document isa AbstractDict && get(document,"schema","") == AIRFOIL_STUDY_SCHEMA ||
        throw(ArgumentError("This is not an Airfoil Lab study. Open Aero Lab studies in Aero Lab."))
    get(document,"version",0) == 1 || throw(ArgumentError("Unsupported Airfoil study version"))
    request = get(document,"request",nothing)
    request isa AbstractDict || throw(ArgumentError("Study has no analysis setup"))
    geometry = get(document,"geometry",nothing)
    geometry isa AbstractDict || throw(ArgumentError("Study has no embedded source geometry"))
    coordinates = AirfoilJXBackend.study_coordinates(geometry)
    foil = get(request,"airfoil",Dict())
    if get(foil,"kind","") == "uiuc"
        request = merge(request,Dict("airfoil"=>merge(foil,Dict("coordinates"=>coordinates))))
    end
    case = _validated_case(request)
    # Bundled profiles must retain their identity; do not silently accept changed
    # coordinates under a benchmark name and then offer its reference overlay.
    resolved = AirfoilJXBackend.resolve_airfoil(case["airfoil"])
    length(coordinates["x"]) == length(resolved.airfoil.x) &&
        all(isapprox.(coordinates["x"],resolved.airfoil.x;atol=1e-12,rtol=1e-12)) &&
        all(isapprox.(coordinates["y"],resolved.airfoil.y;atol=1e-12,rtol=1e-12)) ||
        throw(ArgumentError("Embedded coordinates do not match the named airfoil"))
    controls = get(document,"controls",Dict{String,Any}())
    controls isa AbstractDict && length(controls)<=350 || throw(ArgumentError("Invalid saved controls"))
    all(k isa AbstractString && length(k)<=100 && (v isa Bool || v isa AbstractString && length(v)<=4000)
        for (k,v) in controls) || throw(ArgumentError("Invalid saved control values"))
    name = get(document,"name","Airfoil study")
    name isa AbstractString && 1<=length(name)<=160 || throw(ArgumentError("Invalid study name"))
    merge(document,Dict("request"=>_job_request(case),"geometry"=>merge(geometry,coordinates),
        "controls"=>controls,"results"=>_study_result_link(get(document,"results",nothing))))
end

function _study_load(path)
    path = _study_path(path;existing=true)
    filesize(path)<=AIRFOIL_STUDY_LIMIT || throw(ArgumentError("Study exceeds 8 MB"))
    bytes = read(path)
    etag = bytes2hex(sha256(bytes))
    document = _study_validate_document(JSON3.read(String(bytes),Dict{String,Any}))
    link = document["results"]
    available = link !== nothing && (_get_job(link["job_id"]) !== nothing || isfile(joinpath(_safe_job_dir(link["job_id"]),JOB_RECORD_FILE)))
    _study_remember(path,document)
    Dict("ok"=>true,"path"=>path,"etag"=>etag,"study"=>document,"results_available"=>available,
        "message"=>link !== nothing && !available ? "Setup loaded. Linked results are missing from this installation; no analysis was started. Results location: $(link["directory"])" : "Study loaded")
end

function _study_save(data)
    path = _study_path(get(data,"path",""))
    isdir(dirname(path)) || throw(ArgumentError("The selected study folder is unavailable"))
    actual = _study_hash(path)
    expected = get(data,"etag",nothing)
    actual == expected || throw(AirfoilStudyConflict("This file changed outside this window. Reopen it or use Save study as to preserve both versions."))
    request = get(data,"request",nothing)
    case = _validated_case(request)
    resolved = AirfoilJXBackend.resolve_airfoil(case["airfoil"])
    foil = resolved.airfoil
    link = _study_result_link(get(data,"results",nothing))
    if link !== nothing
        link["directory"] = _safe_job_dir(link["job_id"])
    end
    document = _study_validate_document(Dict{String,Any}("schema"=>AIRFOIL_STUDY_SCHEMA,"version"=>1,
        "workbench"=>"airfoil","name"=>splitext(basename(path))[1],"saved_at"=>string(now(UTC))*"Z",
        "request"=>_job_request(case),"geometry"=>Dict("name"=>foil.name,"x"=>foil.x,"y"=>foil.y,
            "source_url"=>get(resolved.metadata,"source_url",nothing),"attribution"=>get(resolved.metadata,"attribution",nothing)),
        "controls"=>get(data,"controls",Dict()),"view"=>get(data,"view",Dict()),"results"=>link))
    # Geometry resolution may take time (e.g. first UIUC download). Recheck
    # immediately before replacing the file in case an external editor saved it.
    _study_hash(path) == expected || throw(AirfoilStudyConflict("This study changed while saving. Reopen it or use Save study as."))
    _study_atomic_write(path,document)
    _study_remember(path,document)
    Dict("ok"=>true,"path"=>path,"etag"=>_study_hash(path),"study"=>document,
        "message"=>"Study saved. Source coordinates are embedded; solver results remain in their linked folder.")
end

function _airfoil_study_route(req,path)
    try
        if req.method == "GET" && path == "/api/airfoil/studies/recent"
            return _study_json(200,lock(()->Dict("items"=>_study_recent()),AIRFOIL_STUDY_LOCK))
        elseif req.method == "POST"
            data = _study_input(req)
            if path in ("/api/airfoil/studies/choose-open","/api/airfoil/studies/choose-save")
                mkpath(AIRFOIL_STUDY_DIR)
                chosen = NativeDialogs.choose_file(req,joinpath(PRIVATE_ROOT,"TEMP_DELETEABLE","native_dialogs");
                    mode=endswith(path,"choose-save") ? "save" : "open")
                if !get(chosen,"cancelled",true)
                    chosen["etag"] = _study_hash(chosen["path"])
                end
                return _study_json(200,chosen)
            elseif path == "/api/airfoil/studies/open"
                return _study_json(200,lock(()->_study_load(get(data,"path","")),AIRFOIL_STUDY_LOCK))
            elseif path == "/api/airfoil/studies/save"
                return _study_json(200,lock(()->_study_save(data),AIRFOIL_STUDY_LOCK))
            end
        end
        _study_json(404,Dict("error"=>"Unknown study action"))
    catch err
        _study_json(err isa AirfoilStudyConflict ? 409 : err isa ArgumentError ? 400 : 500,
            Dict("ok"=>false,"error"=>sprint(showerror,err)))
    end
end
