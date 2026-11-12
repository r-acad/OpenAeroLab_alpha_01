using Test,Printf
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
include(joinpath(ROOT,"apps","airfoil_web","src","backends","cfd_backend.jl"))
const B=CFDBackend
const SCRATCH=joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(dirname(ROOT),"02_PRIVATE_PROJECT_DEVELOPMENT")),"TEMP_DELETEABLE","lowspeed_auto_tests")
mkpath(SCRATCH)
@testset "Imported geometry selects low-speed topology and matching starts" begin
    mktempdir(SCRATCH) do dir
        af=B.CMeshGen.naca4_points(4412;n=401)
        source=Dict{String,Any}("kind"=>"dat","id"=>"section.dat")
        path=joinpath(dir,"section.dat")
        writefoil(x,y)=open(path,"w") do io
            println(io,"XFOIL input")
            for k in reverse(eachindex(x));@printf(io,"%.17g %.17g\n",x[k],y[k]);end
        end
        writefoil(af.x,af.y)
        original=read(path)
        clean=Dict{String,Any}("reynolds"=>3e6,"alpha_deg"=>12.)
        resolved=B._lowspeed_resolve_flow(source,path,clean)
        @test resolved["lowspeed_topology"]=="ch_native"
        @test !haskey(clean,"lowspeed_topology")
        flap=merge(copy(clean),Dict{String,Any}("gurney_height"=>.01,"gurney_mode"=>"lower"))
        resolved_flap=B._lowspeed_resolve_flow(source,path,flap)
        c=B.lowspeed_configuration(resolved);g=B.lowspeed_configuration(resolved_flap)
        @test (c["mesh"],c["start"],c["mut_base"])==(g["mesh"],g["start"],g["mut_base"])
        @test c["start"]=="ramp" && c["mut_base"]=="committed"
        @test haskey(g,"baffle") && !haskey(c,"baffle")
        @test read(path)==original
        x=copy(af.x);y=copy(af.y)
        x[1]=x[end]=(x[1]+x[end])/2;y[1]=y[end]=(y[1]+y[end])/2
        writefoil(x,y)
        @test B._lowspeed_resolve_flow(source,path,clean)["lowspeed_topology"]=="sharp"
        @test_throws ArgumentError B._lowspeed_resolve_flow(source,path,flap)
        @test_throws ArgumentError B.lowspeed_configuration(Dict("lowspeed_topology"=>"unknown"))
    end
end
