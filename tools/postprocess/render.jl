#!/usr/bin/env julia
# Headless output generation shared by both web workbenches.
using CairoMakie, GeometryBasics, JSON3, TOML, Dates, Statistics, Printf

include(joinpath(@__DIR__, "..", "..", "src", "io", "lbm_slice_format.jl"))
using .LBMSliceFormat

const COLORS = ["#167d9a", "#d6813e", "#5b9072", "#926bac", "#ba5658"]

plain(x::AbstractDict) = Dict{String,Any}(String(k)=>plain(v) for (k,v) in pairs(x))
plain(x::AbstractVector) = [plain(v) for v in x]
plain(x) = x
safe_name(s) = strip(replace(lowercase(String(s)),r"[^a-z0-9_-]+"=>"_"),'_')
escape_html(x) = replace(string(x),'&'=>"&amp;",'<'=>"&lt;",'>'=>"&gt;",'"'=>"&quot;",'\''=>"&#39;")
finite_number(x) = x isa Real && !(x isa Bool) && isfinite(x)

function request(path)
    raw = endswith(lowercase(path),".json") ? plain(JSON3.read(read(path,String))) : TOML.parsefile(path)
    source = abspath(String(raw["source_directory"])); isdir(source) || error("Source directory does not exist")
    dest = abspath(String(raw["destination_directory"])); mkpath(dest)
    format = lowercase(String(get(raw,"chart_format","png")))
    format in ("png","svg","pdf") || error("Unsupported chart format")
    render_format = lowercase(String(get(raw,"render_format","png")))
    render_format in ("png","svg","pdf") || error("Unsupported render format")
    quantity = lowercase(String(get(raw,"surface_quantity","cp")))
    quantity in ("cp","pressure","shear") || error("Unsupported surface quantity")
    camera = lowercase(String(get(raw,"camera","perspective")))
    camera in ("perspective","top","front","side") || error("Unsupported camera")
    width=Int(get(raw,"image_width",1400)); height=Int(get(raw,"image_height",900))
    480<=width<=4096 && 320<=height<=4096 || error("Image dimensions must be between 480×320 and 4096×4096")
    merge(raw,Dict("source_directory"=>source,"destination_directory"=>dest,"chart_format"=>format,"render_format"=>render_format,
                   "surface_quantity"=>quantity,"camera"=>camera,"image_width"=>width,"image_height"=>height))
end

function source_files(source,dest)
    found=String[]
    for (root,dirs,files) in walkdir(source;follow_symlinks=false)
        filter!(d->begin
            p=abspath(joinpath(root,d)); p!=dest && !startswith(p,dest*string(Base.Filesystem.path_separator)) &&
            !(d in (".git","node_modules",".julia","makie"))
        end,dirs)
        for name in files
            path=joinpath(root,name); islink(path) && continue
            push!(found,path)
            length(found)<=100_000 || error("Postprocess source contains too many files")
        end
    end
    found
end

function range_for(values)
    a=sort!(Float64[x for x in values if finite_number(x)])
    isempty(a) && return (-1.0,1.0)
    lo=a[max(1,round(Int,.02*length(a)))]; hi=a[max(1,round(Int,.98*length(a)))]
    hi>lo ? (lo,hi) : (lo-max(abs(lo)*.01,1e-5),hi+max(abs(hi)*.01,1e-5))
end

function save_artifact!(fig,req,manifest,name,kind,title;format=req["chart_format"])
    filename=safe_name(name)*"."*format
    CairoMakie.save(joinpath(req["destination_directory"],filename),fig;px_per_unit=1)
    push!(manifest["artifacts"],Dict("path"=>filename,"kind"=>kind,"title"=>title,
          "mime"=>get(Dict("png"=>"image/png","svg"=>"image/svg+xml","pdf"=>"application/pdf"),format,"application/octet-stream")))
    filename
end

function numeric_csv(path)
    lines=readlines(path); isempty(lines) && return (String[],Vector{Vector{Float64}}())
    firstline=findfirst(l->!isempty(strip(l))&&!startswith(strip(l),'#'),lines)
    firstline===nothing && return (String[],Vector{Vector{Float64}}())
    sep=count(==(';'),lines[firstline])>count(==(','),lines[firstline]) ? ';' : ','
    names=strip.(replace.(split(lines[firstline],sep),'"'=>"")); cols=[Float64[] for _ in names]
    selected=lines[firstline+1:end]; stride=max(1,cld(length(selected),200_000))
    for line in selected[1:stride:end]
        vals=split(strip(line),sep); length(vals)==length(names) || continue
        parsed=[tryparse(Float64,strip(v)) for v in vals]
        all(v->v!==nothing,parsed) || continue
        for i in eachindex(cols); push!(cols[i],parsed[i]); end
    end
    names,cols
end

function history_chart!(req,manifest,files)
    candidates=filter(p->lowercase(basename(p)) in ("convergence.csv","forces.csv","history.csv"),files)
    isempty(candidates) && return false
    sort!(candidates;by=p->(lowercase(basename(p))=="convergence.csv",stat(p).mtime))
    path=last(candidates); names,cols=numeric_csv(path)
    (isempty(cols)||isempty(first(cols))) && return false
    key=lowercase.(replace.(names,r"[^A-Za-z0-9]"=>""))
    ix=something(findfirst(k->k in ("step","iteration","iterations","iter"),key),1)
    selected=Int[]
    for wanted in (("cd","drag"),("cl","lift"),("cs","cm"),("rhomin","residual","massresidual"))
        j=findfirst(k->k in wanted,key); j===nothing||j==ix||push!(selected,j)
    end
    isempty(selected) && append!(selected,first(filter(!=(ix),collect(eachindex(cols))),min(3,length(cols)-1)))
    isempty(selected) && return false
    fig=Figure(size=(req["image_width"],req["image_height"]),backgroundcolor=:white,fontsize=15)
    Label(fig[0,1],get(req,"title","Ludwig")*" · saved iteration history",fontsize=21,font=:bold)
    for (row,j) in enumerate(selected)
        ax=Axis(fig[row,1],xlabel=names[ix],ylabel=names[j],title=names[j],xgridcolor=(:gray,.16),ygridcolor=(:gray,.16))
        lines!(ax,cols[ix],cols[j];color=COLORS[mod1(row,length(COLORS))],linewidth=2)
    end
    Label(fig[length(selected)+1,1],"Saved data only; a flat history alone does not establish convergence.",fontsize=11,color=:gray40)
    save_artifact!(fig,req,manifest,"chart_iteration_history","chart","Iteration history")
    manifest["last_sample"]=Dict(names[i]=>last(cols[i]) for i in eachindex(names) if !isempty(cols[i])&&isfinite(last(cols[i])))
    true
end

function normalized_charts!(req,manifest)
    path=get(req,"airfoil_results",nothing); path===nothing && return false
    isfile(String(path)) || (push!(manifest["warnings"],"The requested normalized airfoil data file is unavailable.");return false)
    raw=plain(JSON3.read(read(String(path),String))); charts=get(raw,"charts",Any[]); made=false
    for (i,item) in enumerate(first(charts,min(length(charts),12)))
        fig=Figure(size=(req["image_width"],req["image_height"]),backgroundcolor=:white,fontsize=16)
        ax=Axis(fig[1,1],title=String(get(item,"title","Airfoil result")),xlabel=String(get(item,"xlabel","x")),
                ylabel=String(get(item,"ylabel","y")),yreversed=get(item,"yreverse",false)===true,
                xgridcolor=(:gray,.16),ygridcolor=(:gray,.16))
        count=0
        for series in get(item,"series",Any[])
            x=get(series,"x",Any[]); y=get(series,"y",Any[]); length(x)==length(y)||continue
            valid=findall(k->finite_number(x[k])&&finite_number(y[k]),eachindex(x)); isempty(valid)&&continue
            count+=1
            lines!(ax,Float64[x[k] for k in valid],Float64[y[k] for k in valid];label=String(get(series,"label","Series $count")),
                   color=COLORS[mod1(count,length(COLORS))],linewidth=2)
            scatter!(ax,Float64[x[k] for k in valid],Float64[y[k] for k in valid];color=COLORS[mod1(count,length(COLORS))],markersize=5)
        end
        count>0||continue;count>1&&axislegend(ax;position=:rb)
        save_artifact!(fig,req,manifest,"chart_"*String(get(item,"name","airfoil_$i")),"chart",String(get(item,"title","Airfoil result")))
        made=true
    end
    made
end

function lbmp(path)
    open(path,"r") do io
        String(read(io,4))=="LBMP"||error("Invalid surface pressure file")
        ltoh(read(io,UInt32))==1||error("Unsupported LBMP version")
        n=Int(ltoh(read(io,UInt32)));objects=Int(ltoh(read(io,UInt32)))
        0<n<=1_000_000 && 0<=objects<=10_000||error("Surface mesh size is unsupported")
        for _ in 1:objects
            len=Int(ltoh(read(io,UInt32)));0<=len<=4096||error("Invalid surface object name");skip(io,len)
        end
        required=(9+3+4)*4*n;filesize(path)-position(io)>=required||error("Surface file is incomplete")
        xyz=read!(io,Vector{Float32}(undef,9*n));skip(io,3*n*4)
        pressure=read!(io,Vector{Float32}(undef,n));shear=read!(io,Vector{Float32}(undef,n));cp=read!(io,Vector{Float32}(undef,n))
        xyz,Dict("cp"=>cp,"pressure"=>pressure,"shear"=>shear)
    end
end

function stl_vertices(path)
    open(path,"r") do io
        filesize(path)>=84||error("STL file is too short")
        seek(io,80);n=Int(ltoh(read(io,UInt32)))
        0<n<=1_000_000 && filesize(path)==84+50*n||error("Only binary STL is supported for Makie rendering")
        xyz=Vector{Float32}(undef,9*n)
        for i in 1:n
            skip(io,12);read!(io,view(xyz,9*(i-1)+1:9*i));skip(io,2)
        end
        xyz
    end
end

function surface_render!(req,manifest,files)
    surfaces=filter(p->occursin(r"^surface_\d+\.lbmp$",basename(p)),files)
    xyz=nothing;values=nothing;quantity=req["surface_quantity"];subtitle="Geometry preview; no surface flow field available"
    if !isempty(surfaces)
        sort!(surfaces;by=p->(parse(Int,match(r"surface_(\d+)",basename(p)).captures[1]),stat(p).mtime))
        xyz,fields=lbmp(last(surfaces));values=fields[quantity]
        subtitle=quantity=="cp" ? "Pressure coefficient Cp" : quantity=="pressure" ? "Gauge pressure [Pa]" : "Wall shear magnitude [Pa]"
    else
        stls=filter(p->lowercase(splitext(p)[2])==".stl",files)
        sort!(stls;by=p->(lowercase(basename(p))=="model.stl",stat(p).mtime))
        if !isempty(stls); xyz=stl_vertices(last(stls))
        else
            amrs=filter(p->basename(p)=="amr_data.json",files)
            if !isempty(amrs)
                a=plain(JSON3.read(read(last(amrs),String)));xyz=Float32.(a["geometry"]["vertices"])
                subtitle="Solver-placed geometry preview; no surface flow field available"
            end
        end
    end
    xyz===nothing&&return false
    triangles=length(xyz)÷9;indices=1:triangles
    points=Point3f[];faces=TriangleFace{Int}[];colors=Float32[]
    for (out,i) in enumerate(indices)
        for v in 0:2
            k=9*(i-1)+3*v; p=Point3f(xyz[k+1],xyz[k+2],xyz[k+3]);all(isfinite,p)||error("Nonfinite surface coordinate")
            push!(points,p);values===nothing||push!(colors,isfinite(values[i]) ? values[i] : NaN32)
        end
        push!(faces,TriangleFace(3*out-2,3*out-1,3*out))
    end
    angles=Dict("perspective"=>(7pi/4,pi/6),"top"=>(-pi/2,pi/2-.001),"front"=>(pi,.001),"side"=>(-pi/2,.001))
    az,el=angles[req["camera"]]
    fig=Figure(size=(req["image_width"],req["image_height"]),backgroundcolor=:white,fontsize=15)
    ax=Axis3(fig[1,1],title=get(req,"title","Ludwig")*"\n"*subtitle,xlabel="Model X [m]",ylabel="Model Y [m]",zlabel="Model Z [m]",
             aspect=:data,viewmode=:fitzoom,azimuth=az,elevation=el)
    if values===nothing
        mesh!(ax,points,faces;color="#77a6b8",shading=true)
    else
        picture=mesh!(ax,points,faces;color=colors,colormap=:turbo,colorrange=range_for(values),shading=false)
        Colorbar(fig[1,2],picture;label=subtitle,width=18)
    end
    footer=values===nothing ? fig[2,1] : fig[2,1:2]
    Label(footer,"Source/model coordinates · all $triangles triangles shown · CairoMakie",fontsize=11,color=:gray40)
    save_artifact!(fig,req,manifest,"render_surface_"*(values===nothing ? "geometry" : quantity),"render_3d",subtitle;format=String(get(req,"render_format","png")))
    values===nothing&&push!(manifest["warnings"],"The 3D render shows geometry only because no saved LBMP surface field was available.")
    true
end

function slice_chart!(req,manifest,files)
    paths=filter(p->endswith(p,".lbmslice"),files);isempty(paths)&&return false
    sort!(paths;by=p->stat(p).mtime)
    header=nothing;arrays=nothing
    for path in reverse(paths)
        h,a=LBMSliceFormat.read_lbmslice(path)
        if get(h,"with_fields",false)===true;header=h;arrays=a;break;end
    end
    arrays===nothing&&return false
    fields=[k for k in ("velocity_magnitude","pressure","density") if haskey(arrays,k)];isempty(fields)&&return false
    fields=first(fields,min(length(fields),2));fig=Figure(size=(req["image_width"],req["image_height"]),backgroundcolor=:white,fontsize=15)
    Label(fig[0,1:2],get(req,"title","Ludwig")*" · recorded cut",fontsize=19,font=:bold)
    h=Float64(header["spacing_m"]);o=Float64.(header["origin_m"]);stride=max(1,ceil(Int,sqrt(header["n1"]*header["n2"]/250_000)))
    x=o[1].+((collect(1:stride:header["n1"])).-.5).*h;y=o[2].+((collect(1:stride:header["n2"])).-.5).*h
    for (i,field) in enumerate(fields)
        values=Float64.(arrays[field][1:stride:end,1:stride:end])
        if haskey(arrays,"obstacle");values[arrays["obstacle"][1:stride:end,1:stride:end].>0].=NaN;end
        unit="";for a in header["arrays"];a["name"]==field&&(unit=a["units"]);end
        ax=Axis(fig[i,1],title="$field [$unit] · step $(header["step"])",xlabel=uppercase(header["axis1"])*" [m]",ylabel=uppercase(header["axis2"])*" [m]",aspect=DataAspect())
        p=heatmap!(ax,x,y,values;colormap=:turbo,colorrange=range_for(values));Colorbar(fig[i,2],p;width=16)
    end
    Label(fig[length(fields)+1,1:2],"$(uppercase(header["normal_axis"])) = $(header["normal_coord_m"]) m · saved data at step $(header["step"]) · CairoMakie",fontsize=11,color=:gray40)
    save_artifact!(fig,req,manifest,"chart_recorded_cut","chart","Recorded cut plane")
    true
end

function html_report!(req,manifest)
    dest=req["destination_directory"];title=String(get(req,"title","Ludwig results"))
    open(joinpath(dest,"report.html"),"w") do io
        print(io,"<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>",escape_html(title),"</title><style>body{font:16px/1.6 system-ui;margin:40px auto;max-width:1100px;padding:0 24px;color:#173542}h1{line-height:1.2}h2{margin-top:32px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #d8e3e8;text-align:left;padding:7px}img,object{max-width:100%;height:auto}aside{padding:14px;background:#f6ead7;border-radius:8px}small{color:#647c87}</style><h1>",escape_html(title),"</h1><small>Generated ",escape_html(manifest["generated_utc"])," with CairoMakie</small>")
        print(io,"<p>This report describes saved solver artifacts. It does not independently establish mesh adequacy or solution convergence.</p>")
        metadata=get(req,"metadata",Dict());isempty(metadata)||print(io,"<h2>Case metadata</h2><table>",join(["<tr><th>$(escape_html(k))</th><td>$(escape_html(v))</td></tr>" for (k,v) in sort!(collect(metadata);by=first)]),"</table>")
        if !isempty(manifest["warnings"]);print(io,"<h2>Notes</h2><aside><ul>",join(["<li>$(escape_html(w))</li>" for w in manifest["warnings"]]),"</ul></aside>");end
        if haskey(manifest,"last_sample");print(io,"<h2>Last saved history sample</h2><table>",join(["<tr><th>$(escape_html(k))</th><td>$(escape_html(v))</td></tr>" for (k,v) in sort!(collect(manifest["last_sample"]);by=first)]),"</table>");end
        for artifact in manifest["artifacts"]
            filename=artifact["path"];print(io,"<h2>",escape_html(artifact["title"]),"</h2><p><a href=\"",escape_html(filename),"\">Download ",escape_html(filename),"</a></p>")
            artifact["mime"] in ("image/png","image/svg+xml")&&print(io,"<img src=\"",escape_html(filename),"\" alt=\"",escape_html(artifact["title"]),"\">")
        end
        print(io,"</html>")
    end
    push!(manifest["artifacts"],Dict("path"=>"report.html","kind"=>"report","title"=>"HTML report","mime"=>"text/html"))
end

function main(args)
    length(args)==1||error("Usage: render.jl REQUEST.toml|REQUEST.json")
    req=request(args[1]);manifest=Dict{String,Any}("ok"=>true,"engine"=>"CairoMakie","generated_utc"=>Dates.format(now(UTC),"yyyy-mm-ddTHH:MM:SSZ"),
       "title"=>get(req,"title","Ludwig results"),"artifacts"=>Any[],"warnings"=>String[])
    files=source_files(req["source_directory"],req["destination_directory"])
    if get(req,"charts",false)===true
        made=false
        for (name,action) in (("airfoil charts",()->normalized_charts!(req,manifest)),("iteration history",()->history_chart!(req,manifest,files)),("recorded cut",()->slice_chart!(req,manifest,files)))
            try made=action()||made catch e;push!(manifest["warnings"],"Could not generate $name: "*sprint(showerror,e));end
        end
        made||push!(manifest["warnings"],"No saved numeric history, normalized chart data, or flow cut was available for charts.")
    end
    if get(req,"render_3d",false)===true
        try surface_render!(req,manifest,files)||push!(manifest["warnings"],"No binary STL, AMR geometry, or LBMP surface was available for the 3D render.")
        catch e;push!(manifest["warnings"],"Could not generate the 3D render: "*sprint(showerror,e));end
    end
    get(req,"report",false)===true&&html_report!(req,manifest)
    open(joinpath(req["destination_directory"],"manifest.json"),"w") do io;JSON3.write(io,manifest);end
    println("MAKIE_POSTPROCESS_COMPLETED artifacts=$(length(manifest["artifacts"])) warnings=$(length(manifest["warnings"]))")
    for warning in manifest["warnings"];println("NOTE: ",warning);end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(ARGS)
end
