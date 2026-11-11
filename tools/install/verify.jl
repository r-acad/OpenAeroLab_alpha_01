# Each environment is checked in its own process, matching solver worker isolation.
project=only(ARGS)
function verify_cuda!()
    available=CUDA.functional()
    println("CUDA functional: ",available)
    if available
        runtime=CUDA.runtime_version();toolkit=CUDA.toolkit_version
        println("CUDA runtime: ",runtime,"; compiled toolkit: ",toolkit)
        if toolkit===nothing || Base.thisminor(runtime)!=Base.thisminor(toolkit)
            if Sys.iswindows() && runtime>=v"13" && toolkit!==nothing && toolkit>=v"13"
                println(stderr,"CUDA runtime and compiled toolkit versions differ. On Windows CUDA 13 the runtime version can report the display driver's version; CUDA.jl 5.10 may flag this as an error (upstream JuliaGPU/CUDA.jl#3139). Review driver and toolkit compatibility before relying on solver results.")
            else
                println(stderr,"CUDA runtime and compiled toolkit versions differ; review the CUDA.jl diagnostic and driver/toolkit compatibility before relying on solver results.")
            end
        end
        sum(CUDA.ones(Float32,16))==16 || error("CUDA device smoke check failed")
        println("CUDA device smoke check passed; this is not a full solver validation")
    end
end
if project=="third_party/JXfoil"
    using HTTP, JSON3, JXfoil
    include(joinpath(@__DIR__,"..","..","apps","ludwig_web","src","server.jl"))
    for route in ("/", "/aero-lab", "/airfoil-lab", "/api/aero/server-info")
        response=LudwigWeb.app_handler(HTTP.Request("GET",route))
        response.status==200 || error("Workbench smoke check failed: $route ($(response.status))")
    end
elseif project=="src/octree_q19"
    using OctreeQ19, WriteVTK, YAML
    println("CPU octree D3Q19 package loaded; no GPU is required.")
elseif project=="tools/postprocess"
    using CairoMakie, GeometryBasics, JSON3
    fig=Figure(size=(160,100));ax=Axis(fig[1,1]);lines!(ax,[0,1],[0,1])
    scratch=joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(@__DIR__,"..","..","..","02_PRIVATE_PROJECT_DEVELOPMENT")),"TEMP_DELETEABLE","setup")
    mkpath(scratch)
    mktempdir(scratch) do dir; save(joinpath(dir,"setup-check.png"),fig);end
elseif project=="solvers/pbcav_s1030/src/pbcav"
    using CUDA, KernelAbstractions, StaticArrays, Adapt, Atomix
    include(joinpath(@__DIR__,"..","..",project,"PBCav.jl"))
    verify_cuda!()
else
    using CUDA, KernelAbstractions, StaticArrays, WriteVTK, YAML, Plots, Adapt, Atomix
    include(joinpath(@__DIR__,"gr_runtime.jl"))
    configure_gr_windows_path!(Plots)
    # Imports alone do not catch missing native GR plugin dependencies on Windows.
    scratch=joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(@__DIR__,"..","..","..","02_PRIVATE_PROJECT_DEVELOPMENT")),"TEMP_DELETEABLE","setup")
    mkpath(scratch)
    mktempdir(scratch) do dir
        figure=Plots.plot([0,1,2],[0,1,0];legend=false)
        png=joinpath(dir,"gr-check.png");svg=joinpath(dir,"gr-check.svg")
        Plots.savefig(figure,png);Plots.savefig(figure,svg)
        data=read(png)
        length(data)>100 && data[1:8]==UInt8[0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a] || error("GR PNG renderer did not produce a valid PNG")
        occursin("<svg",read(svg,String)) || error("GR SVG renderer did not produce a valid SVG")
        println("Verified GR PNG and SVG exports")
    end
    verify_cuda!()
end
println("Verified ",project)
