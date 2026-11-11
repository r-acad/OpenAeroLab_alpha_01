# Read-only access to the solver's case artifacts for Aero Lab.
include(joinpath(@__DIR__,"wall_surface.jl"))

_case_dir(id::AbstractString, kind::AbstractString) = begin
    _id(id)
    kind == "export" && return begin
        destination = get(_load_job(String(id)),"export_directory",nothing)
        destination isa AbstractString || throw(ArgumentError("Job has no selected export folder"))
        String(destination)
    end
    kind in ("mesh", "run") || throw(ArgumentError("Unknown case artifact kind"))
    if isdefined(@__MODULE__,:_state_path) && isfile(_state_path(String(id)))
        pointer=get(_load_job(String(id)),kind=="mesh" ? "mesh_case_dir" : "run_case_dir",nothing)
        if pointer isa AbstractString
            isabspath(pointer) || throw(ArgumentError("Stored case folder must be an absolute path"))
            return normpath(String(pointer))
        end
    end
    joinpath(CASES_ROOT, "$(id)_$(kind)")
end

_artifact_url(id, kind, relative) =
    "/api/aero/jobs/$(id)/files/$(kind)/" * replace(String(relative), '\\' => '/')

function _case_file(id::AbstractString, kind::AbstractString, relative::AbstractString)
    root = _case_dir(id, kind)
    rel = String(relative)
    # HTTP routing may leave percent escapes intact. Reject them, backslashes,
    # drive prefixes and dot segments before constructing a filesystem path.
    occursin(r"^[A-Za-z0-9_./-]+$", rel) || throw(ArgumentError("Invalid artifact path"))
    parts = split(rel, '/'; keepempty=true)
    all(p -> !isempty(p) && p != "." && p != "..", parts) ||
        throw(ArgumentError("Invalid artifact path"))
    path = joinpath(root, parts...)
    isfile(path) || return nothing
    # A case-local symlink must not make the file endpoint escape this job.
    inside = try
        relative_real = relpath(realpath(path), realpath(root))
        !isabspath(relative_real) && all(p -> p != "..", splitpath(relative_real))
    catch
        false
    end
    inside || throw(ArgumentError("Artifact path escapes the job directory"))
    path
end

function job_file(id::AbstractString, kind::AbstractString, relative::AbstractString)
    file = _case_file(id, kind, relative)
    file === nothing && return _bad("Artifact not found", 404)
    mime = get(Dict(
        ".json" => "application/json; charset=utf-8", ".html" => "text/html; charset=utf-8",
        ".ludwig-study"=>"application/json; charset=utf-8",
        ".svg" => "image/svg+xml", ".pdf" => "application/pdf", ".md" => "text/markdown; charset=utf-8",
        ".png" => "image/png", ".stl" => "model/stl",
        ".lbmp" => "application/octet-stream", ".lbmslice" => "application/octet-stream",
        ".toml" => "text/plain; charset=utf-8", ".yaml" => "text/plain; charset=utf-8",
        ".csv" => "text/csv; charset=utf-8", ".log" => "text/plain; charset=utf-8",
        ".vtk" => "text/plain; charset=utf-8", ".vtu" => "application/xml",
        ".vtp" => "application/xml"), lowercase(splitext(file)[2]), "application/octet-stream")
    HTTP.Response(200, ["Content-Type" => mime, "Cache-Control" => "no-store",
                        "X-Content-Type-Options" => "nosniff"], read(file))
end

function job_file(id::AbstractString, relative::AbstractString)
    parts = split(String(relative), '/'; limit=2)
    length(parts) == 2 || throw(ArgumentError("Artifact path must begin with mesh/ or run/"))
    job_file(id, parts[1], parts[2])
end

"Return the solver's built mesh counts and memory, or nothing until its summary exists."
function gpu_memory_estimate_gb(summary::AbstractDict, lattice::AbstractString="d3q27")
    levels = summary["levels"]
    isempty(levels) && throw(ArgumentError("Build summary has no mesh levels"))
    # the CPU octree solver reports its stored slots directly: RAM, no device overhead
    lattice == "q19_octree" && return haskey(summary,"estimated_run_bytes_total") ? Float64(summary["estimated_run_bytes_total"])/1e9 : 2.5Float64(summary["bytes_allocated_total"])/1e9+0.75
    calibrated = if length(levels) == 1
        668.0 * Float64(levels[1]["cells"]) / 1e9 + 0.7
    else
        sum(Float64(level["cells"]) * (i == length(levels) ? 700.0 : 462.0)
            for (i, level) in enumerate(levels)) / 1e9 + 0.7
    end
    # D3Q39 adds 39-direction flow and thermal populations. Until measured on
    # this app's geometry, reserve three times the calibrated D3Q27 arrays.
    lattice == "d3q39" && (calibrated = 3.0 * (calibrated - 0.7) + 1.6)
    cpu_grid_gb = Float64(summary["bytes_allocated_total"]) / 1e9
    # CPU summarysize is not GPU allocation, but it provides an independent
    # lower-bound proxy if a new solver option adds arrays beyond calibration.
    # Force maps and uploaded surface data are not part of CPU grid summarysize.
    # Reserve 128 bytes per actual (possibly symmetry-clipped) sampled facet.
    surface_gb=128.0*Float64(get(summary,"geometry_triangles",0))/1e9
    estimate=max(calibrated, 1.15 * cpu_grid_gb + (lattice == "d3q39" ? 1.6 : 0.7)) + surface_gb
    # Count all allocated levels (including covered parents), so this remains
    # conservative before the finite-volume leaf/face adapter has run.
    lattice=="pond_thermal" && (estimate=max(estimate,1800.0*sum(Float64(level["cells"]) for level in levels)/1e9+.7+surface_gb))
    estimate
end

function _validate_build_summary(raw::AbstractDict)
    count_value(value) = value isa Integer && !(value isa Bool) && value >= 0
    positive_count(value) = count_value(value) && value > 0
    finite_value(value) = value isa Real && !(value isa Bool) && isfinite(value)
    require(test,message) = test || throw(ArgumentError(message))
    for key in ("total_cells","total_blocks","bytes_allocated_total","num_levels","levels","domain_size_m","mesh_offset_m","coarse_grid_cells")
        require(haskey(raw,key),"missing $key")
    end
    for key in ("total_cells","total_blocks","bytes_allocated_total","num_levels")
        require(positive_count(raw[key]),"$key must be a positive integer")
    end
    octree = get(raw,"mesh","") == "octree"
    require(raw["num_levels"] <= (octree ? Q19_MAX_LEVELS : 7),"num_levels must be from 1 to $(octree ? Q19_MAX_LEVELS : 7)")
    for key in ("domain_size_m","mesh_offset_m","coarse_grid_cells")
        values=raw[key]
        require(values isa AbstractVector && length(values)==3,"$key must have three entries")
        require(all(finite_value,values),"$key contains non-finite coordinates")
        key=="mesh_offset_m" || require(all(x->x>0,values),"$key must be positive")
        key=="coarse_grid_cells" && require(all(positive_count,values),"coarse_grid_cells must be integers")
    end
    levels=raw["levels"]
    require(levels isa AbstractVector && length(levels)==raw["num_levels"],"levels does not match num_levels")
    for (i,level) in enumerate(levels)
        require(level isa AbstractDict,"level $i must be a table")
        for key in ("level","cells","active_blocks","bytes_allocated","fluid_cells","obstacle_cells","bouzidi_boundary_cells","dx_m")
            require(haskey(level,key),"level $i is missing $key")
        end
        require(positive_count(level["level"]) && level["level"]==i,"level numbering must be consecutive")
        for key in ("cells","active_blocks","bytes_allocated")
            require(positive_count(level[key]),"level $i $key must be a positive integer")
        end
        for key in ("fluid_cells","obstacle_cells","bouzidi_boundary_cells")
            require(count_value(level[key]) && level[key]<=level["cells"],"level $i $key is outside the cell count")
        end
        require(Int128(level["fluid_cells"])+Int128(level["obstacle_cells"])==level["cells"],"level $i fluid and obstacle counts do not add up")
        require(finite_value(level["dx_m"]) && level["dx_m"]>0,"level $i dx_m must be finite and positive")
        i==1 || require(isapprox(levels[i-1]["dx_m"],2*level["dx_m"];rtol=1e-5),"level $i spacing is not half its parent")
    end
    for (total,key) in (("total_cells","cells"),("total_blocks","active_blocks"),("bytes_allocated_total","bytes_allocated"))
        require(sum(Int128(level[key]) for level in levels)==raw[total],"$total does not match the level counts")
    end
    raw
end

function _build_summary_pending(id::AbstractString)
    get(_load_job(String(id)),"status","") in ("meshing","sizing","mesh_sizing")
end

function _read_build_summary(path::AbstractString)
    for attempt in 1:6
        contents=try
            read(path,String)
        catch error
            error isa Union{Base.IOError,SystemError} && attempt<6 || rethrow()
            sleep(.02*attempt)
            continue
        end
        # Close the file before parsing, so Windows can replace it while the
        # immutable snapshot is being checked by another status request.
        return TOML.parse(contents)
    end
end

function build_summary(id::AbstractString; lattice::AbstractString="d3q27",allow_incomplete::Union{Nothing,Bool}=nothing)
    path = _case_file(id, "mesh", "RESULTS/build_only_summary.toml")
    path === nothing && return nothing
    raw = try
        _validate_build_summary(_read_build_summary(path))
    catch error
        error isa Union{TOML.ParserError,ArgumentError,EOFError,Base.IOError,SystemError} || rethrow()
        pending=allow_incomplete===nothing ? _build_summary_pending(id) : allow_incomplete
        pending && return nothing
        throw(ArgumentError("The build-only mesh summary is incomplete or invalid ($(sprint(showerror,error))). Rebuild this mesh before running; inspect $path and the mesh solver log."))
    end
    result = Dict{String,Any}(String(k) => v for (k, v) in raw)
    result["cells"] = Int(raw["total_cells"])
    result["cpu_grid_gb"] = Float64(raw["bytes_allocated_total"]) / 1e9
    result["memory_gb"] = gpu_memory_estimate_gb(raw, lattice)
    result["surface_geometry_allowance_gb"]=128.0*Float64(get(raw,"geometry_triangles",0))/1e9
    result["memory_basis"] = lattice == "q19_octree" ?
        "CPU octree solver: conservative RAM allowance for mesh, population/density/velocity and AMR transfer buffers, thread forces and bounded exports; includes reserve and process overhead. No GPU." : lattice == "pond_thermal" ?
        "provisional thermal kinetic FV allowance: 1800 bytes per allocated Cartesian cell for leaf states, shared faces and work buffers, plus 0.7 GB overhead and surface sampling; at least the LBM build-memory estimate. Measured allocations remain subject to validation" : lattice == "d3q39" ?
        "conservative D3Q39 flow + thermal allowance: 3 x calibrated D3Q27 per-cell costs, plus 1.6 GB overhead; at least 1.15 x CPU grid bytes plus overhead; plus 128 bytes per surface sampling triangle" :
        "estimated GPU run memory from active cells and measured per-cell costs, with 0.7 GB device overhead; at least 1.15 x CPU grid bytes plus overhead; plus 128 bytes per surface sampling triangle"
    result
end

function _slice_header(path::AbstractString)
    open(path, "r") do io
        String(read(io, 8)) == LBMSliceFormat.LBMSLICE_MAGIC ||
            throw(ArgumentError("Invalid numeric cut file"))
        hlen = Int(ltoh(read(io, UInt64)))
        data_offset = Int(ltoh(read(io, UInt64)))
        0 < hlen <= 1_000_000 && data_offset >= 24 + hlen && data_offset <= filesize(path) ||
            throw(ArgumentError("Invalid numeric cut header"))
        header=TOML.parse(String(read(io, hlen)))
        arrays=get(header,"arrays",Any[])
        isempty(arrays) && throw(ArgumentError("Numeric cut has no arrays"))
        required=maximum(Int(get(array,"offset",-1))+Int(get(array,"nbytes",-1)) for array in arrays)
        required>=0 && data_offset+required<=filesize(path) || throw(ArgumentError("Numeric cut data is incomplete"))
        header
    end
end

"Temporary files from atomic result writers are incomplete until published."
function _artifact_in_progress(path::AbstractString)
    name=lowercase(basename(path))
    ext=lowercase(splitext(name)[2])
    ext in (".partial",".tmp") || occursin(r"\.tmp\.[0-9]+\.[0-9a-f]+$",name)
end

function _artifact_complete(path::AbstractString)
    isfile(path) && !islink(path) || return false
    name=lowercase(basename(path));ext=lowercase(splitext(name)[2])
    _artifact_in_progress(name) && return false
    filesize(path)>0 || return false
    try
        if ext==".lbmw"
            _wall_surface_metadata(path);return true
        elseif ext==".lbmwg"
            _wall_geometry_header(path);return true
        elseif ext==".lbmp"
            return open(path,"r") do io
                String(read(io,4))=="LBMP" || return false
                version=Int(ltoh(read(io,UInt32)));count=Int(ltoh(read(io,UInt32)));objects=Int(ltoh(read(io,UInt32)))
                version==1 && 0<count<=MAX_WEB_SURFACE_TRIANGLES && 0<=objects<=10000 || return false
                for _ in 1:objects
                    bytes=Int(ltoh(read(io,UInt32)));0<=bytes<=4096 || return false
                    skip(io,bytes)
                end
                position(io)+64count==filesize(path)
            end
        elseif ext==".lbmslice"
            _slice_header(path);return true
        elseif ext in (".vtu",".vtp",".vti",".vtm",".pvtu",".pvd",".svg")
            tail=open(path,"r") do io;seek(io,max(0,filesize(path)-1024));String(read(io));end
            return occursin(ext==".svg" ? "</svg>" : "</VTKFile>",tail)
        elseif ext==".png"
            filesize(path)>=20 || return false
            return open(path,"r") do io
                read(io,8)==UInt8[0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a] || return false
                seek(io,filesize(path)-12)
                read(io,12)==UInt8[0,0,0,0,0x49,0x45,0x4e,0x44,0xae,0x42,0x60,0x82]
            end
        end
        true
    catch
        false
    end
end

function _surface_frame_metadata(path)
    lowercase(splitext(path)[2])==".lbmw" && return _wall_surface_metadata(path)
    result=Dict{String,Any}("geometry_frame"=>"legacy_unverified","geometry_binding"=>"amr_if_triangle_count_matches")
    sidecar=path*".toml"
    if isfile(sidecar)
        try
            metadata=TOML.parsefile(sidecar)
            frame=get(metadata,"geometry_frame","")
            frame in ("solver_local","domain","model_local") || return result
            result["geometry_frame"]=frame;result["geometry_binding"]="native_vertices"
            offset=get(metadata,"mesh_offset_m",nothing)
            offset isa AbstractVector && length(offset)==3 && all(x->x isa Real && isfinite(x),offset) && (result["mesh_offset_m"]=Float64.(offset))
            get(metadata,"triangles",nothing) isa Integer && (result["triangles"]=metadata["triangles"])
        catch
        end
    end
    result
end

function _artifact_catalog_entry(id,phase,root,path)
    relative=replace(relpath(path,root),'\\'=>'/');name=basename(path);ext=lowercase(splitext(name)[2])
    category=_export_category(relative)
    kind=_catalog_kind(category)
    (endswith(lowercase(name),".lbmp.toml") || endswith(lowercase(name),".lbmw.toml") || ext==".lbmwg") && (kind="setup")
    matchstep=match(r"(?:_|step_)(\d{6,10})(?:\.[^.]+)?$",name)
    step=matchstep===nothing ? nothing : parse(Int,matchstep[1])
    quantity=match(r"velocity|pressure|vorticity|density|mach|temperature|cp|shear",lowercase(name))
    entry=Dict{String,Any}("name"=>relative,"url"=>_artifact_url(id,phase,relative),"kind"=>kind,
        "phase"=>phase,"step"=>step,"format"=>replace(ext,"."=>""),"bytes"=>filesize(path))
    quantity===nothing || (entry["quantity"]=quantity.match)
    ext in (".lbmp",".lbmw") && merge!(entry,_surface_frame_metadata(path))
    ext==".lbmw" && (entry["geometry_url"]=_artifact_url(id,phase,replace(joinpath(dirname(relative),entry["geometry_file"]),'\\'=>'/')))
    if ext==".lbmslice"
        header=_slice_header(path)
        entry["has_fields"]=get(header,"with_fields",false)
        entry["fields"]=[String(array["name"]) for array in header["arrays"]]
        entry["axis"]=get(header,"normal_axis",nothing)
        entry["position"]=get(header,"normal_coord_m",nothing)
        phase=="run" && (entry["json_url"]="/api/aero/jobs/$id/slice/$name")
    end
    entry
end

"Discover mesh, surface and recorded cut files without loading their large arrays."
function discover_artifacts(id::AbstractString)
    _id(id)
    mesh = _case_dir(id, "mesh")
    run = _case_dir(id, "run")
    result = Dict{String,Any}("model" => nothing, "amr" => nothing,
                              "summary" => nothing, "surface_latest" => nothing,
                              "cuts" => Dict{String,Any}[],"surfaces"=>Any[],"volumes"=>Any[],"images"=>Any[],"files"=>Any[],"in_progress_files"=>Any[])
    for (kind, dir) in (("mesh", mesh), ("run", run))
        model = joinpath(dir, "model.stl")
        isfile(model) && result["model"] === nothing &&
            (result["model"] = _artifact_url(id, kind, "model.stl"))
        amr = joinpath(dir, "RESULTS", "amr_viz", "amr_data.json")
        isfile(amr) && (result["amr"] = _artifact_url(id, kind, "RESULTS/amr_viz/amr_data.json"))
        if isdir(dir)
            for (folder,dirs,files) in walkdir(dir;follow_symlinks=false)
                filter!(name->!islink(joinpath(folder,name)),dirs)
                for file in files
                    path=joinpath(folder,file)
                    if isfile(path) && !islink(path) && _artifact_in_progress(file)
                        push!(result["in_progress_files"],Dict("phase"=>kind,"name"=>replace(relpath(path,dir),'\\'=>'/')))
                        continue
                    end
                    _artifact_complete(path) || continue
                    entry=try _artifact_catalog_entry(id,kind,dir,path) catch;continue;end
                    push!(result["files"],entry)
                    category=entry["kind"]
                    category=="surface" && push!(result["surfaces"],entry)
                    category=="volume" && push!(result["volumes"],entry)
                    category=="image" && push!(result["images"],entry)
                end
            end
        end
    end
    summary = joinpath(mesh, "RESULTS", "build_only_summary.toml")
    isfile(summary) && (result["summary"] = _artifact_url(id, "mesh", "RESULTS/build_only_summary.toml"))

    vtk = joinpath(run, "RESULTS", "VTK")
    if isdir(vtk)
        surfaces = Tuple{Int,String}[];native=Tuple{Int,String}[]
        for f in readdir(vtk)
            m = match(r"^surface_(\d+)\.lbmp$", f)
            m === nothing || (_artifact_complete(joinpath(vtk,f)) && push!(surfaces, (parse(Int, m.captures[1]), f)))
            w=match(r"^surface_(\d+)\.lbmw$",f)
            w===nothing || (_artifact_complete(joinpath(vtk,f)) && push!(native,(parse(Int,w.captures[1]),f)))
        end
        if !isempty(surfaces)
            sort!(surfaces; by=first)
            _, latest = last(surfaces)
            result["surface_latest"] = _artifact_url(id, "run", "RESULTS/VTK/$latest")
            result["surface_latest_info"]=merge(Dict("url"=>result["surface_latest"]),_surface_frame_metadata(joinpath(vtk,latest)))
            result["surface_facet_latest"]=result["surface_latest"];result["surface_facet_latest_info"]=result["surface_latest_info"]
        end
        if !isempty(native)
            sort!(native;by=first);_,latest=last(native)
            url=_artifact_url(id,"run","RESULTS/VTK/$latest")
            info=_artifact_catalog_entry(id,"run",_case_dir(id,"run"),joinpath(vtk,latest))
            result["surface_native_latest"]=url;result["surface_native_latest_info"]=info
            result["wall_surface_latest_info"]=info
            result["surface_latest"]=url;result["surface_latest_info"]=info
        end
    end

    slices = joinpath(run, "RESULTS", "SLICES")
    if isdir(slices)
        cut_files = Tuple{Int,String,String}[]
        for step_dir in readdir(slices)
            m = match(r"^step_(\d{6,10})$", step_dir)
            m === nothing && continue
            dir = joinpath(slices, step_dir)
            isdir(dir) || continue
            for file in readdir(dir)
                occursin(r"^plane_[A-Za-z0-9_]+_\d{6,10}\.lbmslice$", file) || continue
                push!(cut_files, (parse(Int, m.captures[1]), step_dir, file))
            end
        end
        sort!(cut_files; by=x -> (x[1], x[3]))
        for (step, step_dir, file) in last(cut_files, min(length(cut_files), 120))
            path = joinpath(slices, step_dir, file)
            h = try _slice_header(path) catch; continue end
            get(h, "with_fields", false) === true || continue
            axis = String(get(h, "normal_axis", ""))
            axis in ("x", "y", "z") || continue
            position = Float64(get(h, "normal_coord_m", NaN))
            isfinite(position) || continue
            relative = "RESULTS/SLICES/$step_dir/$file"
            push!(result["cuts"], Dict{String,Any}(
                "axis" => axis, "position" => position, "step" => step,
                "url" => _artifact_url(id, "run", relative),
                "json_url" => "/api/aero/jobs/$id/slice/$file"))
        end
    end
    for key in ("files","surfaces","volumes","images")
        sort!(result[key];by=entry->(get(entry,"step",nothing)===nothing ? -1 : entry["step"],entry["name"]))
    end
    result["wall_surfaces"]=[entry for entry in result["surfaces"] if get(entry,"format",nothing)=="lbmw"]
    state=_load_job(String(id));opts=get(get(state,"case",Dict()),"outputs",Dict())
    availability=Dict{String,Any}()
    for (label,key,reason) in (("surface","surfaces","No completed surface field has been written yet."),
                              ("volume","volumes",get(opts,"volume_vtk",false) ? "No completed volume snapshot has been written yet." : "Volume snapshots were not requested for this analysis. Use the recorded field planes or enable volume VTK for a future run."),
                              ("cuts","cuts","No recorded plane with flow fields is available yet."),
                              ("images","images","No complete rendered image is available yet."))
        entries=result[key]
        steps=[get(entry,"step",nothing) for entry in entries if get(entry,"step",nothing)!==nothing]
        availability[label]=Dict("available"=>!isempty(entries),"count"=>length(entries),
            "latest_step"=>isempty(steps) ? nothing : maximum(steps),"reason"=>isempty(entries) ? reason : "")
    end
    result["availability"]=availability
    result["available"]=availability
    result
end

function _slice_path(id::AbstractString, file::AbstractString)
    m = match(r"^plane_[A-Za-z0-9_]+_(\d{6,10})\.lbmslice$", String(file))
    m === nothing && throw(ArgumentError("Invalid numeric cut name"))
    step = parse(Int, m.captures[1])
    relative = "RESULTS/SLICES/" * @sprintf("step_%06d/", step) * String(file)
    _case_file(id, "run", relative)
end

"Read only selected columns of one field and its AMR owner level, bounded for multi-GB native planes."
function _sample_numeric_slice(path::AbstractString,selected::AbstractString;max_samples=500_000)
    h=_slice_header(path)
    get(h,"with_fields",false)===true || throw(ArgumentError("No flow fields in this cut yet"))
    key=selected=="velocity" ? "velocity_magnitude" : selected=="vorticity" ? "vorticity_$(h["normal_axis"])" : String(selected)
    specs=Dict(String(s["name"])=>s for s in h["arrays"])
    haskey(specs,key) || throw(ArgumentError("Requested field is unavailable in this cut"))
    n1=Int(h["n1"]);n2=Int(h["n2"])
    0<n1<=10_000_000 && 0<n2<=10_000_000 && Int128(n1)*n2<=10_000_000_000 || throw(ArgumentError("Invalid numeric cut dimensions"))
    stride=max(1,ceil(Int,sqrt(Float64(n1)*n2/max_samples)))
    while cld(n1,stride)*cld(n2,stride)>max_samples;stride+=1;end
    arrays=Dict{String,Matrix}()
    open(path,"r") do io
        seek(io,16);base=Int(ltoh(read(io,UInt64)))
        for name in (key,"level")
            haskey(specs,name) || continue
            spec=specs[name];T=spec["dtype"]=="float32" ? Float32 : spec["dtype"]=="uint8" ? UInt8 : throw(ArgumentError("Unsupported cut array type"))
            offset=Int(spec["offset"]);bytes=Int(spec["nbytes"])
            offset>=0 && bytes==n1*n2*sizeof(T) && base+offset+bytes<=filesize(path) || throw(ArgumentError("Invalid cut array bounds"))
            column=Vector{T}(undef,n1);sampled=Matrix{T}(undef,cld(n1,stride),cld(n2,stride))
            for (jj,j) in enumerate(1:stride:n2)
                seek(io,base+offset+(j-1)*n1*sizeof(T));read!(io,column)
                for (ii,i) in enumerate(1:stride:n1);sampled[ii,jj]=column[i];end
            end
            arrays[name]=sampled
        end
    end
    h,arrays,key,stride
end

function slice_json(id::AbstractString, file::AbstractString, field::AbstractString)
    selected = String(field)
    selected in ("velocity", "density", "pressure", "vorticity", "u", "v", "w", "mach", "temperature") ||
        throw(ArgumentError("Unknown cut field"))
    path = _slice_path(id, file)
    path === nothing && return _bad("Numeric cut not found", 404)
    h, arrays, key, stride = try
        _sample_numeric_slice(path,selected)
    catch error
        return _bad("Numeric cut is unavailable: "*sprint(showerror,error),409)
    end
    sampled = arrays[key]
    values = Any[isfinite(x) ? Float64(x) : nothing for x in vec(sampled)]
    owner_levels = haskey(arrays,"level") ? Int.(vec(arrays["level"])) : Int[]
    raw_origin = h["origin_m"]
    spacing = Float64(h["spacing_m"])
    origin = [Float64(x) + (1 - stride)*spacing/2 for x in raw_origin]
    units = ""
    for t in h["arrays"]
        if t["name"] == key
            units = String(t["units"])
            break
        end
    end
    _response(200, Dict{String,Any}(
        "ok" => true, "field" => selected, "source_field" => key,
        "units" => units, "n1" => size(sampled,1), "n2" => size(sampled,2),
        "axis1" => h["axis1"], "axis2" => h["axis2"],
        "normal_axis" => h["normal_axis"], "normal_coord_m" => h["normal_coord_m"],
        "origin_m" => origin, "spacing_m" => spacing*stride,
        "decimation_stride" => stride, "step" => h["step"],
        "values" => values, "owner_levels" => owner_levels))
end

function slice_json(id::AbstractString, file::AbstractString, req::HTTP.Request)
    params = HTTP.queryparams(HTTP.URI(req.target))
    slice_json(id, file, String(get(params, "field", "velocity")))
end
