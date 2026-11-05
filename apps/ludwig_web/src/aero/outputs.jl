const EXPORT_ROOT = joinpath(PRIVATE_ROOT, "results", "exports")
const OUTPUT_DEFAULTS = Dict{String,Any}(
    "folder" => "", "surface_vtk" => true, "surface_lbmp" => false,
    "volume_vtk" => false, "numeric_slices" => true, "slice_png" => true,
    "csv_history" => true, "report" => false, "charts" => false,
    "renders_3d" => false, "report_format" => "html", "chart_format" => "png",
    "render_format" => "png", "surface_quantity" => "cp", "camera" => "perspective",
    "image_width" => 1600, "image_height" => 1000)
const EXPORT_SYNC_TIMES = Dict{String,Float64}()
const POSTPROCESS_PROCESSES = Dict{Tuple{String,String},Any}()
_readable_directory(path) = try isdir(path) catch; false end

function _output_folder(raw; create::Bool=false)
    raw isa AbstractString || throw(ArgumentError("outputs.folder must be a folder path"))
    text = strip(String(raw))
    length(text) <= 2000 && !occursin('\0', text) || throw(ArgumentError("Invalid output folder"))
    folder = isempty(text) ? EXPORT_ROOT : text
    isabspath(folder) || throw(ArgumentError("Output folder must be an absolute path"))
    folder = normpath(abspath(expanduser(folder)))
    try
        isfile(folder) && throw(ArgumentError("Output folder names a file"))
        create && mkpath(folder)
    catch error
        error isa ArgumentError && rethrow()
        throw(ArgumentError("Output folder is unavailable or cannot be created"))
    end
    folder
end

function _validate_outputs(raw)
    raw isa AbstractDict || throw(ArgumentError("outputs must be an object"))
    result = copy(OUTPUT_DEFAULTS)
    for key in ("surface_vtk", "surface_lbmp", "volume_vtk", "numeric_slices", "slice_png",
                "csv_history", "report", "charts", "renders_3d")
        value = get(raw, key, result[key])
        value isa Bool || throw(ArgumentError("outputs.$key must be true or false"))
        result[key] = value
    end
    folder=get(raw,"folder","")
    folder isa AbstractString || throw(ArgumentError("outputs.folder must be a folder path"))
    selected=!isempty(strip(folder))
    result["folder"] = selected ? _output_folder(folder) : ""
    result["folder_selected"] = selected
    for (key, choices) in (("report_format", ("html",)), ("chart_format", ("png", "svg", "pdf")),
                           ("render_format", ("png",)), ("surface_quantity", ("cp", "pressure", "shear")),
                           ("camera", ("perspective", "top", "front", "side")))
        value = string(get(raw, key, result[key]))
        value in choices || throw(ArgumentError("outputs.$key must be one of $(join(choices, ", "))"))
        result[key] = value
    end
    result["image_width"] = _integer(get(raw, "image_width", 1600), "outputs.image_width"; lo=320, hi=7680)
    result["image_height"] = _integer(get(raw, "image_height", 1000), "outputs.image_height"; lo=240, hi=4320)
    result
end

function output_folders(req::HTTP.Request)
    params = HTTP.queryparams(HTTP.URI(req.target))
    folder = _output_folder(string(get(params, "path", "")))
    if folder == EXPORT_ROOT && !isdir(folder); mkpath(folder); end
    _readable_directory(folder) || throw(ArgumentError("Folder does not exist or is unavailable"))
    directories = Dict{String,String}[]
    entries = try readdir(folder) catch; throw(ArgumentError("Folder cannot be read")) end
    for name in entries
        child = joinpath(folder, name)
        _readable_directory(child) && !islink(child) && push!(directories, Dict("name" => name, "path" => child))
    end
    roots = Sys.iswindows() ? [Dict("name" => "$(letter):\\", "path" => "$(letter):\\")
        for letter in 'A':'Z' if _readable_directory("$(letter):\\")] : [Dict("name" => "/", "path" => "/")]
    _response(200, Dict("ok" => true, "path" => folder,
        "parent" => dirname(folder) == folder ? nothing : dirname(folder),
        "roots" => roots, "directories" => directories))
end

function _require_run_output_folder(case)
    opts=get(case,"outputs",nothing)
    opts isa AbstractDict || throw(ArgumentError("Choose an output folder before running the analysis"))
    folder=get(opts,"folder","")
    folder isa AbstractString && !isempty(strip(folder)) ||
        throw(ArgumentError("Choose an output folder before running the analysis"))
    selected=get(opts,"folder_selected",normpath(String(folder))!=normpath(EXPORT_ROOT))
    selected===true || throw(ArgumentError("Choose an output folder before running the analysis; the old temporary default is not a selected destination"))
    _output_folder(folder;create=true)
end

function _export_study_stem(name)
    value=replace(strip(String(name)),r"[^A-Za-z0-9_.-]"=>"_")
    value=strip(value,['.','_'])
    isempty(value) && (value="Aero_study")
    first(value,min(length(value),80))
end

function _prepare_export_directory(id::String, case;study_name="Aero study",timestamp=Dates.now(Dates.UTC))
    opts = get(case, "outputs", nothing)
    opts === nothing && return nothing
    root = _require_run_output_folder(case)
    stamp=Dates.format(timestamp,dateformat"yyyymmdd_HHMMSS")*"_"*lpad(string(Dates.millisecond(timestamp)),3,'0')*"Z"
    base=stamp*"_"*_export_study_stem(study_name)*"_"*first(id,8)
    destination = joinpath(root,base)
    for kind in ("mesh","run")
        relative = relpath(destination,_case_dir(id,kind))
        (relative == "." || (!isabspath(relative) && first(splitpath(relative)) != "..")) &&
            throw(ArgumentError("The selected export folder cannot be inside this job's working case"))
    end
    # A fresh job owns exactly one new subfolder. Existing folders are never reused.
    for suffix in 0:99
        candidate=suffix==0 ? destination : destination*"_"*string(suffix)
        ispath(candidate) && continue
        try
            mkdir(candidate)
            return candidate
        catch error
            ispath(candidate) && continue
            rethrow(error)
        end
    end
    throw(ArgumentError("Could not reserve a unique timestamped output folder"))
end

"""Copy a stable file snapshot without libuv sendfile's large-file limits.

Only the initial source length is copied: logs may continue growing while an
export is made. A truncated source or failed write removes the incomplete file.
"""
function _copy_regular_file!(source,destination;buffer_bytes::Integer=8*1024^2)
    isfile(source) || throw(ArgumentError("The source file is unavailable"))
    ispath(destination) && throw(ArgumentError("The destination file already exists"))
    buffer_bytes>0 || throw(ArgumentError("The copy buffer must be positive"))
    expected=filesize(source);remaining=expected;created=false
    buffer=Vector{UInt8}(undef,min(buffer_bytes,max(1,expected)))
    try
        open(source,"r") do input
            open(destination,"w") do output
                created=true
                while remaining>0
                    wanted=min(length(buffer),remaining)
                    count=readbytes!(input,buffer,wanted)
                    count==wanted || error("The source file was truncated during copying")
                    write(output,view(buffer,1:count))==count || error("Incomplete file write")
                    remaining-=count
                end
            end
        end
        filesize(destination)==expected || error("Incomplete file copy")
    catch
        created && isfile(destination) && rm(destination;force=true)
        rethrow()
    end
    destination
end

function _copy_case_tree!(source,destination)
    isdir(source) || throw(ArgumentError("The prepared mesh folder is unavailable"))
    mkpath(destination)
    for (folder,dirs,files) in walkdir(source;follow_symlinks=false)
        filter!(name->!islink(joinpath(folder,name)),dirs)
        for name in files
            original=joinpath(folder,name);islink(original) && continue
            relative=relpath(original,source)
            target=joinpath(destination,relative);mkpath(dirname(target))
            _copy_regular_file!(original,target)
        end
    end
    destination
end

function _export_category(relative)
    name=lowercase(basename(relative));ext=lowercase(splitext(name)[2])
    ext==".lbmp" && return "surfaces"
    endswith(name,".lbmp.toml") && return "surfaces"
    startswith(name,"surface_") && ext in (".vtu",".vtp",".vtk",".pvtu",".pvd") && return "surfaces"
    ext in (".vtu",".vtp",".vtk",".vti",".vtm",".pvtu",".pvd") && return "volumes"
    ext==".lbmslice" && return "numeric_cuts"
    ext in (".png",".svg",".jpg",".jpeg") && return "images"
    ext==".csv" && return "histories"
    name=="thermal_state.jls" && return "checkpoints"
    (startswith(name,"checkpoint_") && ext==".bin") && return "checkpoints"
    name in ("thermal_fv_summary.toml","octree_run_summary.toml") && return "diagnostics"
    ext in (".log",".txt") && return "logs"
    name in ("build_only_summary.toml","amr_data.json","mesh.oct") && return "mesh"
    "setup"
end

_typed_export_relative(kind,relative)=joinpath("results",_export_category(relative),kind,relative)
_catalog_kind(category)=get(Dict("surfaces"=>"surface","volumes"=>"volume","numeric_cuts"=>"cut",
    "images"=>"image","histories"=>"history","logs"=>"log"),category,"setup")

function _copy_export_snapshot!(original,target;immutable=false)
    mkpath(dirname(target));islink(target) && error("An export destination was replaced by a symbolic link")
    temporary=target*".$(uuid4()).partial"
    try
        if immutable && isdefined(Base.Filesystem,:hardlink)
            try Base.Filesystem.hardlink(original,temporary) catch;_copy_regular_file!(original,temporary);end
        else
            _copy_regular_file!(original,temporary)
        end
        Base.Filesystem.rename(temporary,target)
    finally
        isfile(temporary) && rm(temporary;force=true)
    end
end

function _write_export_manifest!(id,state,destination)
    _refresh_run_study_snapshot!(id,state,destination)
    document=Dict{String,Any}("schema"=>"org.ludwig.aero-run","version"=>1,"job_id"=>id,
        "study_name"=>get(state,"study_name","Aero study"),"status"=>get(state,"status","unknown"),
        "updated_utc"=>_now_utc(),"case"=>state["case"],
        "run_case_dir"=>get(state,"run_case_dir",_case_dir(id,"run")),
        "mesh_case_dir"=>get(state,"mesh_case_dir",_case_dir(id,"mesh")),
        "study_file"=>get(state,"study_snapshot_path",nothing),
        "canonical_geometry_sha256"=>get(state,"study_geometry_sha256",nothing),
        "source_fileinfo"=>get(state,"source_fileinfo",nothing),
        "note"=>"The complete native solver output and study snapshot are retained in this folder. Runtime job metadata may be restored from this manifest without running the solver.")
    target=joinpath(destination,"run_manifest.json");temporary=target*".$(uuid4()).partial"
    try
        write(temporary,JSON3.write(document));Base.Filesystem.rename(temporary,target)
    finally
        isfile(temporary) && rm(temporary;force=true)
    end
end

function _selected_export(relative::String, opts)
    name = lowercase(basename(relative)); ext = lowercase(splitext(name)[2])
    startswith(name, "surface_") && ext in (".vtu", ".vtk", ".vtp", ".pvtu", ".pvd") && return opts["surface_vtk"]
    ext == ".lbmp" && return opts["surface_lbmp"]
    endswith(name,".lbmp.toml") && return opts["surface_lbmp"]
    ext == ".lbmslice" && return opts["numeric_slices"]
    ext == ".png" && return opts["slice_png"]
    ext == ".csv" && return opts["csv_history"]
    ext in (".vtu", ".vtk", ".vtp", ".vti", ".vtm", ".pvtu", ".pvd") && return opts["volume_vtk"]
    name in ("build_only_summary.toml", "amr_data.json", "thermal_fv_summary.toml", "thermal_state.jls", "octree_run_summary.toml", "mesh.oct") && return true
    false
end

function _sync_exports!(id::String; force::Bool=false)
    state = _load_job(id)
    destination = get(state, "export_directory", nothing)
    destination === nothing && return
    opts = state["case"]["outputs"]
    now = time()
    !force && now - get(EXPORT_SYNC_TIMES, id, 0.0) < 5 && return
    EXPORT_SYNC_TIMES[id] = now
    isdir(destination) && !islink(destination) || error("The selected export job folder is unavailable")
    for kind in ("mesh", "run")
        source = _case_dir(id, kind)
        isdir(source) || continue
        for (folder, dirs, files) in walkdir(source; follow_symlinks=false)
            filter!(name -> !islink(joinpath(folder, name)), dirs)
            for name in files
                original = joinpath(folder, name); islink(original) && continue
                relative = relpath(original, source)
                keep = name in ("model.stl", "config.yaml","solver.log") || _selected_export(relative, opts)
                keep || continue
                _artifact_complete(original) || continue
                target = joinpath(destination,_typed_export_relative(kind,relative))
                if isfile(target) && filesize(target) == filesize(original) && mtime(target) >= mtime(original); continue; end
                _copy_export_snapshot!(original,target;immutable=lowercase(splitext(name)[2]) in (".lbmp",".lbmslice",".vtu",".vtp",".vtk",".png"))
            end
        end
    end
    _write_export_manifest!(id,state,destination)
end

function _export_response(id::String, state)
    directory = get(state, "export_directory", nothing)
    directory === nothing && return Dict("status" => "not_configured", "folder" => nothing, "files" => Any[], "warnings" => Any[])
    warnings = Any[get(state, "export_warning", "")]; filter!(x -> !isempty(string(x)), warnings)
    post = get(state, "postprocessing", Dict())
    for (_, entry) in pairs(post)
        get(entry, "status", "") == "failed" && push!(warnings, get(entry, "message", "Makie postprocessing failed"))
    end
    for kind in ("mesh","run")
        manifest = joinpath(directory,"postprocess",kind,"manifest.json")
        isfile(manifest) || (manifest=joinpath(directory,"makie_$kind","manifest.json"))
        if isfile(manifest)
            try
                details = _plain(JSON3.read(read(manifest,String)))
                append!(warnings,get(details,"warnings",Any[]))
            catch
            end
        end
    end
    files = Dict{String,Any}[]
    if isdir(directory)
        for (folder, dirs, names) in walkdir(directory; follow_symlinks=false)
            filter!(name -> !islink(joinpath(folder,name)), dirs)
            for name in names
                path = joinpath(folder,name); islink(path) && continue
                _artifact_complete(path) || continue
                relative = replace(relpath(path,directory), '\\' => '/')
                occursin(r"^[A-Za-z0-9_./-]+$", relative) || continue
                entry=Dict{String,Any}("name" => relative, "url" => _artifact_url(id,"export",relative), "bytes" => filesize(path),
                    "kind"=>_catalog_kind(_export_category(relative)),"format"=>replace(lowercase(splitext(name)[2]),"."=>""))
                source=match(r"^results/[^/]+/(mesh|run)/(.*)$",relative)
                source===nothing && (source=match(r"^(mesh|run)/(.*)$",relative))
                source===nothing || (entry["source_url"]=_artifact_url(id,source[1],source[2]);entry["phase"]=source[1])
                native=match(r"^native_(mesh|run)/(.*)$",relative)
                native===nothing && (native=match(r"^native/CASES/(mesh|run)/(.*)$",relative))
                native===nothing && (native=match(r"^native/CASES/[A-Za-z0-9_-]+_(mesh|run)/(.*)$",relative))
                native===nothing || (entry["source_url"]=_artifact_url(id,native[1],native[2]);entry["phase"]=native[1])
                origin=source===nothing ? native : source
                if origin!==nothing
                    original=joinpath(_case_dir(id,origin[1]),split(origin[2],'/')...)
                    if isfile(original)
                        metadata=try _artifact_catalog_entry(id,origin[1],_case_dir(id,origin[1]),original) catch;Dict();end
                        for (key,value) in metadata
                            key in ("url","name") || (entry[key]=value)
                        end
                    end
                end
                push!(files,entry)
            end
        end
    end
    running = any(get(entry,"status","") == "running" for (_,entry) in pairs(post))
    phase = get(state,"status","")
    options = get(get(state,"case",Dict()),"outputs",OUTPUT_DEFAULTS)
    requested = any(get(options,key,false) for key in ("report","charts","renders_3d"))
    pending = requested && ((phase == "mesh_ready" && !haskey(post,"mesh")) || (phase == "completed" && !haskey(post,"run")))
    active = phase in ("meshing","waiting_for_gpu","running","sizing","mesh_sizing")
    Dict("status" => running || pending ? "generating" : active ? "collecting" : "ready", "folder" => directory,
         "files" => files, "warnings" => unique(warnings), "postprocessing" => post,
         "study_file"=>get(state,"study_snapshot_path",nothing),"retains_native_results"=>isdir(joinpath(directory,"native","CASES")))
end

function _launch_postprocessing!(id::String, kind::String)
    state = _load_job(id); opts = get(state["case"], "outputs", OUTPUT_DEFAULTS)
    any(opts[key] for key in ("report", "charts", "renders_3d")) || return
    destination = get(state, "export_directory", nothing); destination === nothing && return
    request = Dict{String,Any}("source_directory" => _case_dir(id,kind),
        "destination_directory" => joinpath(destination,"postprocess",kind), "title" => "$(get(state,"study_name","Aero study")) — $kind",
        "report" => opts["report"], "charts" => opts["charts"], "render_3d" => opts["renders_3d"],
        "chart_format" => opts["chart_format"], "surface_quantity" => opts["surface_quantity"],
        "camera" => opts["camera"], "image_width" => opts["image_width"], "image_height" => opts["image_height"],
        "metadata" => Dict("job_id" => id, "stage" => kind, "case" => state["case"]))
    post = get(state,"postprocessing",Dict{String,Any}())
    try
        mkpath(joinpath(destination,"logs"))
        process = LudwigPostprocess.launch(request; log_path=joinpath(destination,"logs","makie_$kind.log"), wait=false)
        POSTPROCESS_PROCESSES[(id,kind)] = process
        post[kind] = Dict("status" => "running", "message" => "Generating requested Makie outputs")
        _patch_job(id,Dict("postprocessing" => post))
        @async begin
            wait(process)
            current = _load_job(id); entries = get(current,"postprocessing",Dict{String,Any}())
            entries[kind] = Dict("status" => success(process) ? "completed" : "failed",
                "message" => success(process) ? "Makie outputs generated" : "Makie generation failed; inspect makie_$kind.log")
            _patch_job(id,Dict("postprocessing" => entries))
            pop!(POSTPROCESS_PROCESSES,(id,kind),nothing)
        end
    catch error
        post[kind] = Dict("status" => "failed", "message" => sprint(showerror,error))
        _patch_job(id,Dict("postprocessing" => post))
    end
end
