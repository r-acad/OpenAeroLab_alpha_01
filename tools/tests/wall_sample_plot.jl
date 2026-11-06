# Binary identity and within-facet pressure variation; no GUI or solver launch.
using Test,SHA
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
include(joinpath(ROOT,"tools","postprocess","WallSamplePlots.jl"))
const W=WallSamplePlots
const P=joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(ROOT,"..","02_PRIVATE_PROJECT_DEVELOPMENT")),"TEMP_DELETEABLE","wall_sample_plot_tests")
mkpath(P);const FIXTURE=mktempdir(P)
geometry=joinpath(FIXTURE,"surface.lbmwg");field=joinpath(FIXTURE,"surface_000012.lbmw")
open(geometry,"w") do io
    write(io,codeunits("LBMWGEO\0"),UInt32(1),UInt32(36),UInt64(3))
    for x in (-1f0,0f0,1f0)
        write(io,Float32[x,0,0],Float32[0,1,0],.1f0,UInt32(1),UInt8(8),UInt8(2),UInt16(1))
    end
end
digest=bytes2hex(open(sha256,geometry))
open(field,"w") do io
    write(io,codeunits("LBMWFLD\0"),UInt32(1),UInt32(1),UInt64(3),codeunits(digest),Int64(12),Float64[.01,1,1,1],Float32[-1,0,1])
end
@testset "Native wall sample plot retains within-facet resolution" begin
    frame=W.read_frame(field)
    @test frame.count==3&&frame.step==12&&frame.geometry_sha256==digest
    @test frame.pressure==Float32[-1,0,1]
    parents=UInt32[];W.foreach_sample(geometry) do _,_,_,_,parent,_,_,_;push!(parents,parent);end
    @test parents==[1,1,1]
    image=W.rasterize(geometry,frame;width=600,height=400)
    @test image.finite_samples==3&&image.painted_pixels>0
    @test extrema(filter(isfinite,vec(image.values)))==(-1f0,1f0)
    @test !all(image.covered) # no invented pressure in the background/holes
    unsupported=joinpath(FIXTURE,"unsupported.lbmwg");cp(geometry,unsupported)
    open(unsupported,"r+") do io;seek(io,24+34);write(io,UInt16(0));end
    unsupported_frame=merge(frame,(geometry_sha256=bytes2hex(open(sha256,unsupported)),))
    @test W.rasterize(unsupported,unsupported_frame;width=600,height=400).finite_samples==2
    wrong=merge(frame,(geometry_sha256=repeat("0",64),))
    @test_throws ErrorException W.rasterize(geometry,wrong;width=600,height=400)
    @test_throws ErrorException W.foreach_sample((args...)->nothing,geometry;expected_count=4)
    truncated=joinpath(FIXTURE,"incomplete.lbmw");write(truncated,read(field)[1:end-1])
    @test_throws ErrorException W.read_frame(truncated)
end
println("Wall sample plot fixtures: ",FIXTURE)
if "--render" in ARGS
    result=W.plot_wall_samples(geometry,field,joinpath(FIXTURE,"native_wall_fixture.png"))
    @test isfile(joinpath(FIXTURE,"native_wall_fixture.png"))
end
