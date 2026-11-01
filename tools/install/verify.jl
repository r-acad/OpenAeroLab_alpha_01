# Each environment is checked in its own process, matching solver worker isolation.
project=only(ARGS)
if project=="third_party/JXfoil"
    using HTTP, JSON3, JXfoil
    include(joinpath(@__DIR__,"..","..","apps","ludwig_web","src","server.jl"))
    for route in ("/", "/aero-lab", "/airfoil-lab", "/api/aero/server-info")
        response=LudwigWeb.app_handler(HTTP.Request("GET",route))
        response.status==200 || error("Workbench smoke check failed: $route ($(response.status))")
    end
elseif project=="tools/postprocess"
    using CairoMakie, GeometryBasics, JSON3
    fig=Figure(size=(160,100));ax=Axis(fig[1,1]);lines!(ax,[0,1],[0,1])
    scratch=joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(@__DIR__,"..","..","..","02_PRIVATE_PROJECT_DEVELOPMENT")),"TEMP_DELETEABLE","setup")
    mkpath(scratch)
    mktempdir(scratch) do dir; save(joinpath(dir,"setup-check.png"),fig);end
elseif project=="solvers/pbcav_s1030/src/pbcav"
    using CUDA, KernelAbstractions, StaticArrays, Adapt, Atomix
    include(joinpath(@__DIR__,"..","..",project,"PBCav.jl"))
    println("CUDA functional: ",CUDA.functional())
    CUDA.functional() && (sum(CUDA.ones(Float32,16))==16 || error("CUDA device check failed"))
else
    using CUDA, KernelAbstractions, StaticArrays, WriteVTK, YAML, Plots, Adapt, Atomix
    println("CUDA functional: ",CUDA.functional())
    CUDA.functional() && (sum(CUDA.ones(Float32,16))==16 || error("CUDA device check failed"))
end
println("Verified ",project)
