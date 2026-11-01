# Export preferences are separate from private working artifacts required by the GUI.
const AIRFOIL_POSTPROCESS_LAUNCHER=joinpath(ROOT,"tools","postprocess","PostprocessLauncher.jl")
isfile(AIRFOIL_POSTPROCESS_LAUNCHER) && include(AIRFOIL_POSTPROCESS_LAUNCHER)
const AIRFOIL_OUTPUT_DEFAULTS = Dict{String,Any}(
    "folder"=>"", "surface_vtk"=>true, "surface_lbmp"=>false,
    "volume_vtk"=>false, "numeric_slices"=>true, "slice_png"=>true,
    "csv_history"=>true, "report"=>false, "charts"=>false, "renders_3d"=>false,
    "report_format"=>"html", "chart_format"=>"png", "render_format"=>"png",
    "surface_quantity"=>"cp", "camera"=>"perspective", "image_width"=>1600, "image_height"=>1000)

function _output_preferences(data::AbstractDict)
    supplied = haskey(data,"outputs") || haskey(data,"output_directory")
    raw = get(data,"outputs",Dict{String,Any}())
    raw isa AbstractDict || throw(ArgumentError("outputs must be an object"))
    settings = copy(AIRFOIL_OUTPUT_DEFAULTS)
    for key in ("surface_vtk","surface_lbmp","volume_vtk","numeric_slices","slice_png","csv_history","report","charts","renders_3d")
        value=get(raw,key,settings[key]);value isa Bool || throw(ArgumentError("outputs.$key must be boolean"));settings[key]=value
    end
    for (key,choices) in (("report_format",("html",)),("chart_format",("png","svg","pdf")),
                          ("render_format",("png","svg","pdf")),("surface_quantity",("cp","pressure","shear")),
                          ("camera",("perspective","top","front","side")))
        value=String(get(raw,key,settings[key]));value in choices || throw(ArgumentError("Unsupported outputs.$key: $value"));settings[key]=value
    end
    for key in ("image_width","image_height")
        value=get(raw,key,settings[key]);value isa Real && !(value isa Bool) && isfinite(value) && isinteger(value) && 320<=value<=4096 || throw(ArgumentError("outputs.$key must be an integer in [320,4096]"));settings[key]=Int(value)
    end
    folder=get(raw,"folder",get(data,"output_directory",""))
    folder isa AbstractString || throw(ArgumentError("outputs.folder must be a path string"))
    folder=strip(String(folder));ncodeunits(folder)<=4096 && !occursin('\0',folder) || throw(ArgumentError("Invalid output folder"))
    isempty(folder) || isabspath(folder) || throw(ArgumentError("Choose an absolute output folder"))
    isempty(folder) || !isfile(folder) || throw(ArgumentError("The output folder is a file"))
    settings["folder"]=isempty(folder) ? "" : normpath(folder)
    settings["requested"]=supplied
    settings
end

function _output_capabilities()
    common=Dict("report"=>true,"charts"=>true,"csv_history"=>true)
    Dict("jxfoil"=>merge(common,Dict("surface_vtk"=>false,"surface_lbmp"=>false,"volume_vtk"=>false,"numeric_slices"=>false,"slice_png"=>false,"renders_3d"=>false,
        "display_note"=>"Panel/boundary-layer results are published per completed operating point.")),
      "xfoil"=>merge(common,Dict("surface_vtk"=>false,"surface_lbmp"=>false,"volume_vtk"=>false,"numeric_slices"=>false,"slice_png"=>false,"renders_3d"=>false,
        "display_note"=>"XFOIL results are published per completed operating point.")),
      "lbm"=>merge(common,Dict("surface_vtk"=>true,"surface_lbmp"=>true,"volume_vtk"=>true,"numeric_slices"=>true,"slice_png"=>true,"renders_3d"=>true,
        "display_note"=>"Field output follows the exact requested iteration interval; GPU batch depth is reduced when necessary.")),
      "transonic"=>merge(common,Dict("surface_vtk"=>true,"surface_lbmp"=>false,"volume_vtk"=>false,"numeric_slices"=>true,"slice_png"=>false,"renders_3d"=>false,
        "display_note"=>"Iteration history follows the requested interval. Field snapshots are supported by the native compressible path.")),
      "lowspeed"=>merge(common,Dict("surface_vtk"=>true,"surface_lbmp"=>false,"volume_vtk"=>false,"numeric_slices"=>true,"slice_png"=>false,"renders_3d"=>false,
        "display_note"=>"Sharp C-mesh fields follow the requested interval; the pinned C+H path exposes its final field only.")),
      "cavitation"=>merge(common,Dict("surface_vtk"=>true,"surface_lbmp"=>false,"volume_vtk"=>false,"numeric_slices"=>true,"slice_png"=>true,"renders_3d"=>false,
        "display_note"=>"The requested interval controls available history/frame output; final fields remain available.")))
end

function _output_folders(path::AbstractString)
    requested=strip(String(path)); isempty(requested) && (requested=PRIVATE_ROOT)
    ncodeunits(requested)<=4096 && !occursin('\0',requested) && isabspath(requested) || throw(ArgumentError("Choose an absolute directory"))
    isdir(requested) || throw(ArgumentError("Output folder does not exist; enter a new path directly to create it when exporting"))
    current=realpath(requested)
    children=Any[]
    for name in try readdir(current) catch; String[] end
        startswith(name,".") && continue
        candidate=joinpath(current,name)
        (try isdir(candidate) catch; false end) || continue
        push!(children,Dict("name"=>name,"path"=>candidate));length(children)>=500 && break
    end
    roots=Any[Dict("name"=>"Private project","path"=>PRIVATE_ROOT),Dict("name"=>"Home","path"=>homedir())]
    if Sys.iswindows()
        for letter in 'C':'Z'
            drive=string(letter,":\\");(try isdir(drive) catch; false end) && push!(roots,Dict("name"=>drive,"path"=>drive))
        end
    else
        push!(roots,Dict("name"=>"Filesystem","path"=>"/"))
    end
    Dict("ok"=>true,"path"=>current,"parent"=>dirname(current),"roots"=>roots,"directories"=>children)
end

function _chart_series(points, backend, key)
    good=[p for p in points if get(p,"backend","")==backend && get(p,"alpha_deg",nothing) isa Real && get(p,key,nothing) isa Real]
    sort!(good;by=p->p["alpha_deg"])
    Dict("label"=>backend,"x"=>[p["alpha_deg"] for p in good],"y"=>[p[key] for p in good])
end

function _airfoil_chart_data(payload, case)
    points=get(payload,"results",Any[]);charts=Any[]
    for (quantity,label) in (("cl","CL"),("cd","CD"),("cm","CM"))
        series=[_chart_series(points,b,quantity) for b in case["backends"]]
        filter!(s->!isempty(s["x"]),series)
        isempty(series) || push!(charts,Dict("name"=>"polar_"*quantity,"title"=>label*" versus angle","xlabel"=>"Angle [deg]","ylabel"=>label,"series"=>series))
    end
    for (i,p) in enumerate(points)
        cp=get(p,"cp",Dict()); cp isa AbstractDict || continue
        series=Any[]
        for side in ("upper","lower")
            x=Float64[];y=Float64[]
            for row in get(cp,side,Any[])
                xv=row isa AbstractDict ? get(row,"x",nothing) : row isa AbstractVector && length(row)>=2 ? row[1] : nothing
                yv=row isa AbstractDict ? get(row,"cp",get(row,"cp_inviscid",nothing)) : row isa AbstractVector && length(row)>=2 ? row[2] : nothing
                xv isa Real && yv isa Real && isfinite(xv) && isfinite(yv) || continue
                push!(x,Float64(xv));push!(y,Float64(yv))
            end
            isempty(x) || push!(series,Dict("label"=>side,"x"=>x,"y"=>y))
        end
        isempty(series) || push!(charts,Dict("name"=>"cp_"*string(i),"title"=>"$(get(p,"backend","")) Cp at $(get(p,"alpha_deg","?")) deg","xlabel"=>"x/c","ylabel"=>"Cp","yreverse"=>true,"series"=>series))
    end
    Dict("charts"=>charts,"metadata"=>Dict("airfoil"=>case["airfoil"],"flow"=>case["flow"]))
end

function _write_results_csv(payload, dir)
    rows=get(payload,"results",Any[])
    open(joinpath(dir,"polar.csv"),"w") do io
        println(io,"backend,alpha_deg,cl,cd,cm,status")
        for p in rows
            println(io,join([replace(string(get(p,key,"")),','=>';','\n'=>' ') for key in ("backend","alpha_deg","cl","cd","cm","status")],','))
        end
    end
    for (index,p) in enumerate(rows)
        cp=get(p,"cp",nothing);cp isa AbstractDict || continue
        open(joinpath(dir,"surface_cp_$(index).csv"),"w") do io
            println(io,"backend,alpha_deg,surface,x,y,cp")
            for side in ("upper","lower"),row in get(cp,side,Any[])
                x=row isa AbstractDict ? get(row,"x","") : row isa AbstractVector && length(row)>=2 ? row[1] : ""
                y=row isa AbstractDict ? get(row,"y","") : ""
                value=row isa AbstractDict ? get(row,"cp",get(row,"cp_inviscid","")) : row isa AbstractVector && length(row)>=2 ? row[2] : ""
                println(io,join((get(p,"backend",""),get(p,"alpha_deg",""),side,x,y,value),','))
            end
        end
    end
end

function _write_sample_vtk(payload, dir)
    outputs=String[]
    for (index,p) in enumerate(get(payload,"results",Any[]))
        field=get(p,"fields",Dict());field isa AbstractDict || continue
        samples=get(field,"samples",nothing);samples isa AbstractDict || continue
        x=get(samples,"x",nothing);y=get(samples,"y",nothing)
        x isa AbstractVector && y isa AbstractVector && length(x)==length(y) && !isempty(x) || continue
        all(v->v isa Real && isfinite(v),x) && all(v->v isa Real && isfinite(v),y) || continue
        file=joinpath(dir,"sampled_field_$(index).vtk")
        open(file,"w") do io
            println(io,"# vtk DataFile Version 3.0\nAirfoil Lab sampled 2D field (points, no invented volume cells)\nASCII\nDATASET POLYDATA")
            n=length(x);println(io,"POINTS $n double")
            for i in eachindex(x);println(io,"$(x[i]) $(y[i]) 0");end
            println(io,"VERTICES $n $(2n)")
            for i in 0:n-1;println(io,"1 $i");end
            println(io,"POINT_DATA $n")
            for (key,values) in samples
                key in ("x","y") && continue
                values isa AbstractVector && length(values)==n && all(v->v isa Real && isfinite(v),values) || continue
                name=replace(String(key),r"[^A-Za-z0-9_]"=>"_");println(io,"SCALARS $name double 1\nLOOKUP_TABLE default")
                for value in values;println(io,value);end
            end
        end
        push!(outputs,file)
    end
    outputs
end

function _selected_export(path, settings)
    name=lowercase(basename(path));ext=lowercase(splitext(path)[2]);full=lowercase(path)
    ext in (".vtk",".vtu",".vtp") && return occursin("surface",name) ? settings["surface_vtk"] : settings["volume_vtk"]
    ext==".lbmp" && return settings["surface_lbmp"]
    ext==".lbmslice" && return settings["numeric_slices"]
    ext in (".msgpack",".f16") && (occursin("field",name)||occursin("mesh",name)||occursin("num",full)) && return settings["numeric_slices"]
    ext in (".png",".jpg",".jpeg") && return settings["slice_png"]
    ext==".csv" && return settings["csv_history"]
    false
end

function _finalize_outputs!(id, case, payload, outdir)
    settings=get(case,"outputs",Dict{String,Any}());get(settings,"requested",false) || return
    _put_job(id,Dict("message"=>"Preparing selected output files and reports"))
    exportdir=joinpath(outdir,"exports");mkpath(exportdir);warnings=String[]
    for (directory,dirs,files) in walkdir(outdir)
        filter!(name->name!="exports" && !islink(joinpath(directory,name)),dirs)
        for name in files
            file=joinpath(directory,name);islink(file) && continue
            _selected_export(file,settings) || continue
            relative=relpath(file,outdir);target=joinpath(exportdir,relative);mkpath(dirname(target));cp(file,target;force=true)
        end
    end
    settings["csv_history"] && _write_results_csv(payload,exportdir)
    settings["surface_vtk"] && !("lbm" in case["backends"]) && _write_sample_vtk(payload,exportdir)
    if settings["report"] || settings["charts"] || settings["renders_3d"]
        try
            isdefined(@__MODULE__,:LudwigPostprocess) || error("The Makie postprocessor is not installed; restart the app after installing it")
            chartfile=joinpath(outdir,"airfoil_charts.json");write(chartfile,JSON3.write(_clean(_airfoil_chart_data(payload,case))))
            request=Dict{String,Any}("source_directory"=>outdir,"destination_directory"=>exportdir,
                "title"=>"Airfoil Lab — $(case["airfoil"]["id"])","report"=>settings["report"],"charts"=>settings["charts"],
                "render_3d"=>settings["renders_3d"],"chart_format"=>settings["chart_format"],"render_format"=>settings["render_format"],
                "surface_quantity"=>settings["surface_quantity"],"camera"=>settings["camera"],"image_width"=>settings["image_width"],
                "image_height"=>settings["image_height"],"airfoil_results"=>chartfile,"metadata"=>Dict("display_every"=>get(case,"display_every",nothing)))
            process=Base.invokelatest(LudwigPostprocess.launch,request;log_path=joinpath(outdir,"postprocess.log"),wait=true)
            success(process) || push!(warnings,"Makie postprocessing reported a failure; inspect postprocess.log")
        catch error
            push!(warnings,"Makie output: "*sprint(showerror,error))
        end
    end
    manifest=joinpath(exportdir,"manifest.json")
    if isfile(manifest)
        try
            value=JSON3.read(read(manifest,String));append!(warnings,String.(get(value,"warnings",String[])))
        catch; end
    end
    files=Any[]
    for (directory,_,names) in walkdir(exportdir),name in names
        file=joinpath(directory,name);islink(file) && continue
        relative=relpath(file,exportdir);push!(files,Dict("name"=>replace(relative,'\\'=>'/'),"url"=>"/api/jobs/$id/files/exports/"*replace(relative,'\\'=>'/')))
    end
    for (key,extensions) in (("surface_vtk",(".vtk",".vtu",".vtp")),("surface_lbmp",(".lbmp",)),("volume_vtk",(".vtu",".vtk")),("numeric_slices",(".lbmslice",".msgpack",".f16")),("slice_png",(".png",)))
        settings[key] && !any(f->any(ext->endswith(lowercase(f["name"]),ext),extensions),files) && push!(warnings,"No $key artifact was produced by the selected solver path.")
    end
    destination=exportdir
    if !isempty(settings["folder"])
        try
            mkpath(settings["folder"]);destination=joinpath(realpath(settings["folder"]),"airfoil_"*id)
            source_root=lowercase(abspath(exportdir))*(Sys.iswindows() ? "\\" : "/")
            startswith(lowercase(abspath(destination)),source_root) && throw(ArgumentError("The selected folder cannot be inside this job's export directory"))
            ispath(destination) && throw(ArgumentError("Refusing to replace an existing output job folder"))
            cp(exportdir,destination;force=false)
        catch error
            destination=exportdir;push!(warnings,"Output folder copy: "*sprint(showerror,error)*". Files remain downloadable in the private job folder.")
        end
    end
    output_settings=merge(settings,Dict("display_every_requested"=>get(case,"display_every",nothing)))
    _put_job(id,Dict("exports"=>Dict("status"=>isempty(warnings) ? "completed" : "partial","folder"=>destination,"files"=>files,"warnings"=>warnings),"output_settings"=>output_settings))
end

function _safe_finalize_outputs!(id,case,payload,outdir)
    try
        _finalize_outputs!(id,case,payload,outdir)
    catch error
        message=sprint(showerror,error)
        _put_job(id,Dict("exports"=>Dict("status"=>"failed","folder"=>outdir,"files"=>Any[],"warnings"=>[message])))
        _append_log(id,"outputs","Optional file export failed: "*message)
    end
end
