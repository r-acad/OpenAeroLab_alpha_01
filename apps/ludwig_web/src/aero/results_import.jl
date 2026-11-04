# Imported results are visualization data, never executable case input.
const RESULTS_IMPORT_ROOT=joinpath(RUNTIME_ROOT,"imported_results")
const RESULT_EXTENSIONS=(".lbmp",".lbmslice",".vtu",".vtp",".vtk",".json",".toml")
const RESULT_IMPORT_CACHE=Dict{String,Any}()
const RESULT_IMPORT_LOCK=ReentrantLock()
const MAX_RESULT_BYTES=512_000_000
const MAX_RESULT_SURFACE_TRIANGLES=5_000_000

_result_dir(id)=joinpath(RESULTS_IMPORT_ROOT,_id(id))
_result_metadata(id)=joinpath(_result_dir(id),"dataset.json")
function _load_result_metadata(id)
    path=_result_metadata(id);isfile(path) || throw(ArgumentError("Unknown imported results dataset"))
    _plain(JSON3.read(read(path,String)))
end
function _result_key(name)
    key=lowercase(replace(String(name),r"[^A-Za-z0-9]"=>""))
    get(Dict("pressurecoefficient"=>"cp","coefficientofpressure"=>"cp","cp"=>"cp",
        "pressure"=>"pressure","p"=>"pressure","wallshearstress"=>"shear","shearstress"=>"shear","shear"=>"shear",
        "velocity"=>"velocity","velocitymagnitude"=>"velocity","density"=>"density","rho"=>"density",
        "mach"=>"mach","temperature"=>"temperature","vorticity"=>"vorticity"),key,key)
end
function _result_bounds(vertices)
    isempty(vertices) && throw(ArgumentError("Results contain no coordinates"))
    Dict("min"=>[minimum(vertices[a:3:end]) for a in 1:3],"max"=>[maximum(vertices[a:3:end]) for a in 1:3])
end
function _result_surface(vertices,fields;names=String[])
    all(isfinite,vertices) || throw(ArgumentError("Result geometry contains nonfinite coordinates"))
    length(vertices)%9==0 && 0<length(vertices)<=9MAX_RESULT_SURFACE_TRIANGLES || throw(ArgumentError("Invalid or excessive result triangle count"))
    all(values->length(values)==length(vertices)÷9,values(fields)) || throw(ArgumentError("Surface field values do not match the result triangle count"))
    clean=Dict(String(name)=>[isfinite(v) ? Float64(v) : nothing for v in values] for (name,values) in fields)
    Dict{String,Any}("vertices"=>Float64.(vertices),"fields"=>clean,"names"=>names,"triangles"=>length(vertices)÷9,"bounds"=>_result_bounds(vertices))
end
function _result_lbmp_frame_metadata(path,n)
    metadata_path=path*".toml";isfile(metadata_path) || return nothing
    filesize(metadata_path)<=65_536 || throw(ArgumentError("Native surface coordinate metadata is too large"))
    metadata=try TOML.parsefile(metadata_path) catch
        throw(ArgumentError("Native surface coordinate metadata is incomplete or unavailable; retry after the export completes"))
    end
    get(metadata,"format",nothing)=="LBMP" && get(metadata,"format_version",nothing)==1 &&
        get(metadata,"triangles",nothing) isa Integer && !(metadata["triangles"] isa Bool) && metadata["triangles"]==n ||
        throw(ArgumentError("Native surface coordinate metadata does not match the LBMP file"))
    frame=get(metadata,"geometry_frame",nothing);frame in ("solver_local","domain") ||
        throw(ArgumentError("Unsupported native surface geometry frame"))
    get(metadata,"triangle_order",nothing)=="solver_geometry" ||
        throw(ArgumentError("Unsupported native surface face ordering"))
    offset=collect(_vec3(get(metadata,"mesh_offset_m",nothing),"native surface mesh offset"))
    Dict("geometry_frame"=>frame,"mesh_offset_m"=>offset)
end

function _read_result_lbmp(path)
    open(path,"r") do io
        String(read(io,4))=="LBMP" || throw(ArgumentError("Invalid native surface file"))
        version=Int(ltoh(read(io,UInt32)));n=Int(ltoh(read(io,UInt32)));objects=Int(ltoh(read(io,UInt32)))
        version==1 && 0<n<=MAX_RESULT_SURFACE_TRIANGLES && 0<=objects<=10000 || throw(ArgumentError("Unsupported native surface version or size"))
        16+64n+4objects<=filesize(path)<=MAX_RESULT_BYTES || throw(ArgumentError("Native surface file is incomplete or exceeds the interactive file limit"))
        names=String[]
        for _ in 1:objects
            count=Int(ltoh(read(io,UInt32)));count<=4096 && position(io)+count<=filesize(path) || throw(ArgumentError("Invalid surface name"));push!(names,String(read(io,count)))
        end
        position(io)+4n*16<=filesize(path) || throw(ArgumentError("Native surface file is incomplete"))
        floats(count)=read!(io,Vector{Float32}(undef,count))
        vertices=floats(9n);floats(3n)
        pressure=floats(n);shear=floats(n);cp=floats(n)
        surface=_result_surface(vertices,Dict("pressure"=>Float64.(pressure),"shear"=>Float64.(shear),"cp"=>Float64.(cp));names)
        metadata=_result_lbmp_frame_metadata(path,n)
        surface["native_model_coordinates"]=metadata===nothing || metadata["geometry_frame"]=="solver_local"
        surface["geometry_frame"]=metadata===nothing ? "legacy_unverified" : metadata["geometry_frame"]
        metadata===nothing || (surface["native_mesh_offset"]=metadata["mesh_offset_m"])
        surface["field_units"]=Dict("pressure"=>"Pa","shear"=>"Pa","cp"=>"1");surface
    end
end

function _result_xml_arrays(xml,section,header,compressed)
    found=occursin(Regex("<"*section*"\\b"),xml)
    found || return Dict{String,Any}()
    Dict(name=>(components=parse(Int,_vtk_attribute(spec.attributes,"NumberOfComponents","1")),values=_vtk_array(spec,header,compressed))
        for (name,spec) in _vtk_arrays(_vtk_section(xml,section)))
end
function _triangulate_result(points,polygons,cellfields,pointfields)
    vertices=Float64[];owners=Int[];corners=NTuple{3,Int}[]
    for (owner,polygon) in enumerate(polygons)
        3<=length(polygon)<=10000 || throw(ArgumentError("Invalid result polygon"))
        all(i->1<=i<=size(points,2),polygon) || throw(ArgumentError("Result polygon index is invalid"))
        for i in 2:length(polygon)-1
            tri=(polygon[1],polygon[i],polygon[i+1]);push!(corners,tri);push!(owners,owner)
            length(owners)<=MAX_RESULT_SURFACE_TRIANGLES || throw(ArgumentError("Excessive result triangle count; load a smaller surface snapshot"))
            for point in tri;append!(vertices,@view points[:,point]);end
        end
    end
    fields=Dict{String,Any}()
    for (association,arrays) in ((:point,pointfields),(:cell,cellfields))
        for (name,spec) in arrays
            components=spec.components;values=spec.values;key=_result_key(name)
            count=association===:cell ? length(polygons) : size(points,2)
            components>0 && length(values)==count*components || throw(ArgumentError("Result field $name does not match its mesh"))
            array=reshape(values,components,:)
            if components==1
                fields[key]=[association===:cell ? Float64(array[1,owners[t]]) : sum(Float64(array[1,p]) for p in corners[t])/3 for t in eachindex(corners)]
            elseif components==3
                vectors=[association===:cell ? Float64.(array[:,owners[t]]) : [sum(Float64(array[k,p]) for p in corners[t])/3 for k in 1:3] for t in eachindex(corners)]
                fields[key]=[sqrt(sum(abs2,v)) for v in vectors]
                if key=="velocity"
                    for (k,label) in enumerate(("u","v","w"));fields[label]=[v[k] for v in vectors];end
                end
            end
        end
    end
    _result_surface(vertices,fields)
end
function _read_result_xml_surface(path)
    xml=read(path,String);root=match(r"<VTKFile\b([^>]*)>",xml)
    root===nothing && throw(ArgumentError("Invalid VTK XML result"))
    _vtk_attribute(root[1],"byte_order","LittleEndian")=="LittleEndian" || throw(ArgumentError("Unsupported VTK byte order"))
    compressor=_vtk_attribute(root[1],"compressor");compressor in ("","vtkZLibDataCompressor") || throw(ArgumentError("Unsupported VTK compressor"))
    header=_vtk_attribute(root[1],"header_type","UInt32");compressed=!isempty(compressor)
    arrays=_result_xml_arrays(xml,"Points",header,compressed);isempty(arrays) && throw(ArgumentError("VTK results have no points"))
    points=reshape(first(values(arrays)).values,3,:)
    polydata=_vtk_attribute(root[1],"type")=="PolyData"
    cells=_result_xml_arrays(xml,polydata ? "Polys" : "Cells",header,compressed)
    haskey(cells,"connectivity") && haskey(cells,"offsets") || throw(ArgumentError("VTK surface topology is missing"))
    offsets=cells["offsets"].values;connectivity=cells["connectivity"].values
    if !polydata
        kinds=get(cells,"types",nothing);kinds===nothing && throw(ArgumentError("VTK cell types are missing"))
        all(k->k in (5,7,9),kinds.values) || throw(ArgumentError("VTK surface import supports triangle, quad and polygon cells"))
    end
    polygons=Vector{Int}[];previous=0
    for stop in offsets
        last=Int(stop);previous<last<=length(connectivity) || throw(ArgumentError("Invalid VTK polygon offsets"))
        push!(polygons,Int.(connectivity[previous+1:last]).+1);previous=last
    end
    _triangulate_result(points,polygons,_result_xml_arrays(xml,"CellData",header,compressed),_result_xml_arrays(xml,"PointData",header,compressed))
end

function _read_result_legacy_surface(path)
    text=read(path,String);lines=split(text,'\n');length(lines)>=4 && strip(lines[3])=="ASCII" || throw(ArgumentError("Legacy VTK import requires ASCII; save binary files as inline VTU/VTP"))
    tokens=split(join(lines[4:end]," "));cursor=1;points=nothing;polygons=Vector{Int}[];cellfields=Dict{String,Any}();pointfields=Dict{String,Any}();association=:cell;count=0
    take()=begin cursor<=length(tokens) || throw(ArgumentError("Legacy VTK file is incomplete"));v=tokens[cursor];cursor+=1;v end
    number()=parse(Float64,take());integer()=parse(Int,take())
    while cursor<=length(tokens)
        word=take()
        if word=="DATASET";take()
        elseif word=="POINTS"
            n=integer();take();0<n<=3_000_000 || throw(ArgumentError("Excessive VTK point count"));points=reshape([number() for _ in 1:3n],3,:)
        elseif word in ("POLYGONS","CELLS")
            n=integer();integer();0<n<=MAX_RESULT_SURFACE_TRIANGLES || throw(ArgumentError("Excessive VTK cell count"))
            for _ in 1:n;size=integer();3<=size<=10000 || throw(ArgumentError("Invalid VTK polygon"));push!(polygons,[integer()+1 for _ in 1:size]);end
        elseif word=="CELL_TYPES"
            n=integer();all(k->k in (5,7,9),[integer() for _ in 1:n]) || throw(ArgumentError("Legacy VTK must contain surface polygons"))
        elseif word in ("CELL_DATA","POINT_DATA");association=word=="CELL_DATA" ? :cell : :point;count=integer()
        elseif word=="SCALARS"
            name=take();take();components=1
            cursor<=length(tokens) && tryparse(Int,tokens[cursor])!==nothing && (components=integer())
            take()=="LOOKUP_TABLE" || throw(ArgumentError("Missing VTK lookup table"));take()
            arrays=association===:cell ? cellfields : pointfields;arrays[name]=(components=components,values=[number() for _ in 1:count*components])
        elseif word=="VECTORS"
            name=take();take();arrays=association===:cell ? cellfields : pointfields;arrays[name]=(components=3,values=[number() for _ in 1:count*3])
        elseif word=="FIELD"
            take();n=integer();arrays=association===:cell ? cellfields : pointfields
            for _ in 1:n;name=take();components=integer();tuples=integer();take();values=[number() for _ in 1:components*tuples];tuples==count && (arrays[name]=(components=components,values=values));end
        else
            throw(ArgumentError("Unsupported legacy VTK section $word"))
        end
    end
    points===nothing && throw(ArgumentError("VTK results contain no points"))
    _triangulate_result(points,polygons,cellfields,pointfields)
end

function _result_input_files(path)
    path isa AbstractString && isabspath(path) || throw(ArgumentError("Choose an absolute result file or folder path"))
    path=normpath(abspath(path));ispath(path) || throw(ArgumentError("Result path does not exist"))
    if isfile(path);return [path];end
    files=String[]
    for (directory,dirs,names) in walkdir(path;follow_symlinks=false)
        length(splitpath(relpath(directory,path)))>4 && (empty!(dirs);continue)
        filter!(d->!islink(joinpath(directory,d)),dirs)
        for name in names
            lowercase(splitext(name)[2]) in RESULT_EXTENSIONS && push!(files,joinpath(directory,name))
        end
        length(files)<=10000 || throw(ArgumentError("Result folder has too many files; choose a narrower folder"))
    end
    sort!(files;by=p->(mtime(p),p),rev=true)
    chosen=String[];seen=Set{String}()
    for file in files
        ext=lowercase(splitext(file)[2]);kind=ext in (".lbmp",".vtp",".vtk") ? "surface" : ext==".vtu" ? (startswith(lowercase(basename(file)),"surface") ? "surface" : "volume") : ext==".json" ? "amr" : ext==".toml" ? "summary" : basename(file)
        kind in seen && continue
        ext==".json" && lowercase(basename(file))!="amr_data.json" && continue
        ext==".toml" && lowercase(basename(file))!="build_only_summary.toml" && continue
        push!(chosen,file);push!(seen,kind);length(chosen)>=24 && break
    end
    isempty(chosen) && throw(ArgumentError("Folder has no supported native or VTK results"))
    chosen
end

function _import_result_options(raw)
    coordinates=string(get(raw,"coordinates","auto"));coordinates in ("auto","domain","model") || throw(ArgumentError("Result coordinates must be auto, domain or model"))
    offset=haskey(raw,"mesh_offset") ? collect(_vec3(raw["mesh_offset"],"mesh_offset")) : nothing
    fallback=get(raw,"fallback_mesh_offset",nothing)
    fallback===nothing || (fallback=collect(_vec3(fallback,"fallback_mesh_offset")))
    job_id=get(raw,"job_id",nothing);case=nothing
    if job_id!==nothing
        job=_load_job(_id(job_id));case=get(raw,"use_case_scaling",false)===true ? get(job,"case",nothing) : nothing
        if offset===nothing
            amr=get(discover_artifacts(job_id),"amr",nothing)
            for kind in ("run","mesh")
                path=joinpath(_case_dir(job_id,kind),"RESULTS","amr_viz","amr_data.json")
                isfile(path) || continue;offset=_plain(JSON3.read(read(path,String)))["mesh_offset"];break
            end
        end
    end
    explicit=get(raw,"offset_explicit",false) in (true,"true")
    Dict{String,Any}("coordinates"=>coordinates,"mesh_offset"=>offset,"fallback_mesh_offset"=>fallback,"offset_explicit"=>explicit,"job_id"=>job_id,"case"=>case)
end

function _build_result_metadata!(id,files,options)
    directory=_result_dir(id);surface=nothing;volume=nothing;cuts=Any[];amr=nothing;summary=nothing;warnings=String[];source_paths=String[]
    for item in files
        path=joinpath(directory,item["stored_name"]);ext=lowercase(splitext(path)[2]);push!(source_paths,item["original_path"])
        if ext==".json"
            data=_plain(JSON3.read(read(path,String)));haskey(data,"levels") && haskey(data,"domain_size") || throw(ArgumentError("JSON result must be amr_data.json"));amr=data;item["kind"]="mesh"
        elseif ext==".toml";summary=TOML.parsefile(path);item["kind"]="summary"
        elseif ext==".lbmp";surface=_read_result_lbmp(path);item["kind"]="surface";surface["file"]=item["stored_name"]
        elseif ext==".lbmslice"
            h=_slice_header(path);get(h,"with_fields",false) || throw(ArgumentError("Native slice has no saved fields"));push!(cuts,Dict("file"=>item["stored_name"],"axis"=>h["normal_axis"],"position"=>h["normal_coord_m"],"step"=>h["step"],"fields"=>[_result_key(spec["name"]) for spec in h["arrays"]]));item["kind"]="cut"
        elseif ext in (".vtu",".vtp")
            try
                grid=_read_volume_cut_grid(path);volume=Dict("file"=>item["stored_name"],"bounds"=>Dict("min"=>grid["min"],"max"=>grid["max"]),"fields"=>collect(keys(grid["fields"])));item["kind"]="volume"
            catch volume_error
                ext==".vtp" || occursin("axis-aligned voxel/hexahedron",sprint(showerror,volume_error)) || rethrow()
                surface=_read_result_xml_surface(path);surface["file"]=item["stored_name"];item["kind"]="surface"
            end
        elseif ext==".vtk";surface=_read_result_legacy_surface(path);surface["file"]=item["stored_name"];item["kind"]="surface"
        end
    end
    coordinates=options["coordinates"]=="auto" ? (surface!==nothing && get(surface,"native_model_coordinates",false) ? "model" : "domain") : options["coordinates"]
    offset=options["mesh_offset"]
    if surface!==nothing && haskey(surface,"native_mesh_offset") &&
        (offset===nothing || options["coordinates"]=="auto" && !get(options,"offset_explicit",false))
        offset=surface["native_mesh_offset"]
    end
    amr!==nothing && (offset===nothing || options["coordinates"]=="auto" && !get(options,"offset_explicit",false)) && (offset=get(amr,"mesh_offset",nothing))
    offset===nothing && (offset=get(options,"fallback_mesh_offset",nothing))
    if coordinates=="model" && surface!==nothing
        offset===nothing && throw(ArgumentError("Model-coordinate results need a mesh offset. Load matching AMR data, select a matching simulation, or specify the X/Y/Z offset."))
    end
    offset=coordinates=="domain" || offset===nothing ? [0.,0.,0.] : Float64.(offset)
    bounds=amr!==nothing ? Dict("min"=>[0.,0.,0.],"max"=>amr["domain_size"]) : volume!==nothing ? volume["bounds"] :
        surface!==nothing ? Dict("min"=>surface["bounds"]["min"].+offset,"max"=>surface["bounds"]["max"].+offset) : nothing
    if bounds===nothing && !isempty(cuts)
        h=_slice_header(joinpath(directory,cuts[1]["file"]))
        raw_origin=Float64.(h["origin_m"]);spacing=Float64(h["spacing_m"]);axes=Dict("x"=>1,"y"=>2,"z"=>3)
        a1=axes[h["axis1"]];a2=axes[h["axis2"]];normal=axes[h["normal_axis"]]
        origin=length(raw_origin)==3 ? raw_origin : zeros(3)
        if length(raw_origin)==2
            origin[a1]=raw_origin[1];origin[a2]=raw_origin[2];origin[normal]=Float64(h["normal_coord_m"])
        elseif length(raw_origin)!=3
            throw(ArgumentError("Native slice origin must contain two plane coordinates or three domain coordinates"))
        end
        hi=copy(origin)
        hi[a1]+=h["n1"]*spacing;hi[a2]+=h["n2"]*spacing;bounds=Dict("min"=>origin,"max"=>hi)
    end
    bounds===nothing && throw(ArgumentError("Results contain no surface, volume or numeric slice"))
    fields=surface===nothing ? String[] : collect(keys(surface["fields"]))
    options["case"]===nothing && volume!==nothing && push!(warnings,"Imported VTK values are shown as stored; no lattice-to-physical scaling was assumed.")
    amr===nothing && push!(warnings,"No AMR metadata was loaded; mesh-level outlines are unavailable for this dataset.")
    named=findfirst(item->get(item,"kind","") in ("surface","volume","cut"),files)
    display_name=named===nothing ? basename(first(source_paths)) : files[named]["name"]
    meta=Dict{String,Any}("ok"=>true,"id"=>id,"dataset_id"=>id,"name"=>display_name,"files"=>files,"source_paths"=>source_paths,"coordinate_system"=>coordinates,
        "domain_bounds"=>bounds,"surface_offset"=>offset,"fields"=>fields,"volume_fields"=>volume===nothing ? String[] : volume["fields"],
        "surface"=>surface===nothing ? nothing : Dict("file"=>surface["file"],"triangles"=>surface["triangles"]),"volume"=>volume,"cuts"=>cuts,"options"=>options,"warnings"=>warnings,
        "field_units"=>surface===nothing ? Dict{String,String}() : get(surface,"field_units",Dict(name=>"as stored" for name in keys(surface["fields"]))),
        "surface_url"=>surface===nothing ? nothing : "/api/aero/results/$id/surface","cut_url"=>"/api/aero/results/$id/cut","amr_url"=>amr===nothing ? nothing : "/api/aero/results/$id/amr")
    write(_result_metadata(id),JSON3.write(meta));empty!(RESULT_IMPORT_CACHE);RESULT_IMPORT_CACHE[id]=(surface=surface,amr=amr)
    meta
end

function _store_result_sources(paths,raw;uploaded=nothing)
    id=get(raw,"dataset_id",nothing);existing=id===nothing ? nothing : _load_result_metadata(_id(id));id=id===nothing ? string(uuid4()) : _id(id)
    directory=_result_dir(id);mkpath(directory);files=existing===nothing ? Any[] : existing["files"]
    options=existing===nothing ? _import_result_options(raw) : existing["options"]
    for (name,source,bytes) in paths
        ext=lowercase(splitext(name)[2]);ext in RESULT_EXTENSIONS || throw(ArgumentError("Unsupported result extension $ext"))
        count=bytes===nothing ? filesize(source) : length(bytes)
        limit=bytes===nothing && ext==".lbmslice" ? 16_000_000_000 : MAX_RESULT_BYTES
        0<count<=limit || throw(ArgumentError("Result file exceeds the interactive import limit (512 MB for uploads; 16 GB for local native planes)"))
        stored="source_$(length(files)+1)$ext";destination=joinpath(directory,stored)
        bytes===nothing ? cp(source,destination;force=false) : write(destination,bytes)
        item=Dict{String,Any}("name"=>basename(name),"original_path"=>source,"stored_name"=>stored,"bytes"=>count)
        if ext==".lbmp" && bytes===nothing && isfile(source*".toml")
            filesize(source*".toml")<=65_536 || throw(ArgumentError("Native surface coordinate metadata is too large"))
            cp(source*".toml",destination*".toml";force=false);item["geometry_metadata_file"]=stored*".toml"
        end
        push!(files,item)
    end
    _build_result_metadata!(id,files,options)
end
function import_results(req::HTTP.Request)
    raw=_body(req);paths=_result_input_files(get(raw,"path",nothing))
    lock(RESULT_IMPORT_LOCK) do
        try
            _response(201,_store_result_sources([(basename(p),p,nothing) for p in paths],raw))
        catch error
            error isa Union{EOFError,BoundsError,DimensionMismatch,KeyError} && throw(ArgumentError("Saved result file is incomplete or has unsupported topology"))
            error isa ErrorException && startswith(sprint(showerror,error),"lbmslice:") && throw(ArgumentError(sprint(showerror,error)))
            rethrow()
        end
    end
end
function upload_results(req::HTTP.Request)
    params=HTTP.queryparams(HTTP.URI(req.target));name=String(get(params,"name",""));raw=Dict{String,Any}(String(k)=>v for (k,v) in params)
    haskey(raw,"mesh_offset") && (raw["mesh_offset"]=[parse(Float64,v) for v in split(raw["mesh_offset"],',')])
    haskey(raw,"fallback_mesh_offset") && (raw["fallback_mesh_offset"]=[parse(Float64,v) for v in split(raw["fallback_mesh_offset"],',')])
    lock(RESULT_IMPORT_LOCK) do
        try
            _response(201,_store_result_sources([(name,"Uploaded: "*basename(name),Vector{UInt8}(req.body))],raw))
        catch error
            error isa Union{EOFError,BoundsError,DimensionMismatch,KeyError} && throw(ArgumentError("Saved result file is incomplete or has unsupported topology"))
            error isa ErrorException && startswith(sprint(showerror,error),"lbmslice:") && throw(ArgumentError(sprint(showerror,error)))
            rethrow()
        end
    end
end
function imported_result_response(id)
    _response(200,_load_result_metadata(id))
end
function _result_cached(id)
    get!(RESULT_IMPORT_CACHE,id) do
        meta=_load_result_metadata(id);surface=get(meta,"surface",nothing);amrfile=findfirst(f->get(f,"kind","")=="mesh",meta["files"])
        data=surface===nothing ? nothing : begin
            path=joinpath(_result_dir(id),surface["file"]);ext=lowercase(splitext(path)[2]);ext==".lbmp" ? _read_result_lbmp(path) : ext==".vtk" ? _read_result_legacy_surface(path) : _read_result_xml_surface(path)
        end
        (surface=data,amr=amrfile===nothing ? nothing : _plain(JSON3.read(read(joinpath(_result_dir(id),meta["files"][amrfile]["stored_name"]),String))))
    end
end
function imported_surface(id)
    meta=_load_result_metadata(id);data=_result_cached(id).surface;data===nothing && return _bad("Imported dataset has no surface",404)
    payload=copy(data);payload["surface_offset"]=meta["surface_offset"];payload["coordinates"]=meta["options"]["coordinates"];payload["ok"]=true
    payload["offset"]=meta["surface_offset"];payload["coordinate_system"]=meta["coordinate_system"];payload["field_units"]=meta["field_units"]
    _response(200,payload)
end
function imported_amr(id)
    amr=_result_cached(id).amr;amr===nothing && return _bad("Imported dataset has no AMR metadata",404);_response(200,amr)
end
function _imported_volume_plane(path,axis,position,field,case)
    lock(VOLUME_CUT_LOCK) do
        key=(path,stat(path).mtime,filesize(path));cached=VOLUME_CUT_CACHE[]
        grid=if cached!==nothing && cached.key==key
            cached.grid
        else
            parsed=_read_volume_cut_grid(path);VOLUME_CUT_CACHE[]=(key=key,grid=parsed);parsed
        end
        _volume_sample_plane(grid,axis,position,field,case;resolution=384)
    end
end
function imported_cut(id,req)
    meta=_load_result_metadata(id);params=HTTP.queryparams(HTTP.URI(req.target));axis=String(get(params,"axis","z"));axis in ("x","y","z") || throw(ArgumentError("Choose X, Y or Z cut normal"))
    position=tryparse(Float64,String(get(params,"position","")));position!==nothing && isfinite(position) || throw(ArgumentError("Cut position must be finite"))
    field=String(get(params,"field","velocity"));volume=get(meta,"volume",nothing)
    if volume!==nothing
        case=meta["options"]["case"]
        stored=case===nothing;stored && (case=Dict("lattice"=>"d3q27","flow"=>Dict("velocity"=>.08,"speed_of_sound"=>1.)))
        result=_imported_volume_plane(joinpath(_result_dir(id),volume["file"]),axis,position,field,case)
        result===nothing || begin
            stored && (result["units"]=field=="mach" ? "1" : "as stored")
            result["dataset_id"]=id;return _response(200,result)
        end
    end
    choices=filter(c->c["axis"]==axis && abs(c["position"]-position)<=1e-7*max(1,abs(position)),meta["cuts"])
    isempty(choices) && return _bad("No imported volume or recorded plane contains this requested field and position",404)
    cut=last(sort(choices;by=c->c["step"]));h,arrays,key,stride=_sample_numeric_slice(joinpath(_result_dir(id),cut["file"]),field)
    sampled=arrays[key]
    owner_levels=haskey(arrays,"level") ? Int.(vec(arrays["level"])) : Int[]
    origin=[Float64(x)+(1-stride)*Float64(h["spacing_m"])/2 for x in h["origin_m"]]
    units="";for spec in h["arrays"];spec["name"]==key && (units=String(spec["units"]));end
    _response(200,Dict("ok"=>true,"source"=>"native_slice","field"=>field,"units"=>units,"axis1"=>h["axis1"],"axis2"=>h["axis2"],"normal_axis"=>axis,"normal_coord_m"=>position,"origin_m"=>origin,"spacing_m"=>h["spacing_m"]*stride,"n1"=>size(sampled,1),"n2"=>size(sampled,2),"values"=>[isfinite(v) ? Float64(v) : nothing for v in vec(sampled)],"owner_levels"=>owner_levels,"step"=>h["step"],"dataset_id"=>id))
end
