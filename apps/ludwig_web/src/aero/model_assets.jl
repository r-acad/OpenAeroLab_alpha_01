# Preserve original display assets separately from the canonical solver STL.
const MAX_MODEL_ASSET_BYTES = 50_000_084

function _model_asset_name(value)
    name=basename(replace(String(value),'\\'=>'/'))
    1<=length(name)<=255 && !any(iscntrl,name) || throw(ArgumentError("Invalid original model file name"))
    lowercase(splitext(name)[2]) in (".glb",".obj",".stl") ||
        throw(ArgumentError("Original model must be GLB, OBJ or STL"))
    name
end

function _recover_study_model_source(id,directory)
    canonical=joinpath(directory,"source.stl")
    isfile(canonical)&&isdir(STUDIES_ROOT) || return nothing
    digest=bytes2hex(sha256(read(canonical)))
    files=filter(p->endswith(lowercase(p),".ludwig-study"),readdir(STUDIES_ROOT;join=true))
    sort!(files;by=p->stat(p).mtime,rev=true)
    for path in files
        filesize(path)<=MAX_STUDY_BYTES || continue
        document=try _plain(JSON3.read(read(path,String))) catch; continue end
        setup=get(document,"case",nothing);geometry=get(document,"geometry",nothing);source=get(document,"source_geometry",nothing)
        setup isa AbstractDict && get(setup,"model_id",nothing)==id && geometry isa AbstractDict &&
            get(geometry,"sha256",nothing)==digest && source isa AbstractDict || continue
        recovered=try last(_decode_study_geometry(source)) catch; continue end
        lock(JOB_LOCK) do
            target=joinpath(directory,"study_source_geometry.json")
            temporary=target*".$(uuid4()).tmp";write(temporary,JSON3.write(recovered));Base.Filesystem.rename(temporary,target)
        end
        return recovered
    end
    nothing
end

function _stored_model_source_geometry(id::AbstractString)
    directory=joinpath(MODEL_ROOT,_id(id))
    preserved=joinpath(directory,"study_source_geometry.json")
    isfile(preserved) && return _plain(JSON3.read(read(preserved,String)))
    metadata=joinpath(directory,"display_source.json");source=joinpath(directory,"display_source.bin")
    isfile(metadata)&&isfile(source) || return _recover_study_model_source(id,directory)
    info=_plain(JSON3.read(read(metadata,String)))
    filesize(source)<=MAX_MODEL_ASSET_BYTES || throw(ArgumentError("Original model exceeds the source size limit"))
    bytes=read(source)
    bytes2hex(sha256(bytes))==info["sha256"] || throw(ArgumentError("Original model file is incomplete or changed"))
    _geometry_document(bytes,info["name"],info["media_type"])
end

function save_model_asset(id::String,req::HTTP.Request)
    _load_model(id)
    params=HTTP.queryparams(HTTP.URI(req.target))
    name=_model_asset_name(get(params,"name","original.glb"));extension=lowercase(splitext(name)[2])
    bytes=Vector{UInt8}(req.body)
    0<length(bytes)<=MAX_MODEL_ASSET_BYTES || throw(ArgumentError("Original model must contain at most 50 MB"))
    if extension==".glb"
        length(bytes)>=12 && String(bytes[1:4])=="glTF" || throw(ArgumentError("Invalid GLB source header"))
        version=ltoh(reinterpret(UInt32,bytes[5:8])[1]);size=ltoh(reinterpret(UInt32,bytes[9:12])[1])
        version==2 && Int(size)==length(bytes) || throw(ArgumentError("Unsupported or incomplete GLB source"))
    end
    media=extension==".glb" ? "model/gltf-binary" : extension==".obj" ? "model/obj" : "model/stl"
    info=Dict("name"=>name,"media_type"=>media,"bytes"=>length(bytes),"sha256"=>bytes2hex(sha256(bytes)))
    lock(JOB_LOCK) do
        directory=joinpath(MODEL_ROOT,id);token=string(uuid4())
        temporary=joinpath(directory,"display_source.$token.tmp")
        write(temporary,bytes);Base.Filesystem.rename(temporary,joinpath(directory,"display_source.bin"))
        temporary=joinpath(directory,"display_source_metadata.$token.tmp")
        write(temporary,JSON3.write(info));Base.Filesystem.rename(temporary,joinpath(directory,"display_source.json"))
    end
    _response(201,merge(info,Dict("ok"=>true,"url"=>"/api/aero/models/$id/source")))
end

function model_asset(id::String)
    document=_stored_model_source_geometry(id)
    document===nothing && return _bad("Original appearance is unavailable for this model. Load the GLB file to restore its materials and textures.",404)
    _response(200,merge(document,Dict("ok"=>true)))
end
