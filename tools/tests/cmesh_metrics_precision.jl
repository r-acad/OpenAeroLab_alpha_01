using Test,StaticArrays
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
const LBM_F=Float32
include(joinpath(ROOT,"src","collision","lattice.jl"))
include(joinpath(ROOT,"src","cmesh","cmesh_gen.jl"))
include(joinpath(ROOT,"src","cmesh","cmesh_metrics.jl"))

@testset "Metric precision is explicit and retains original coordinates" begin
    X=[.1234567890123+.017123456789i+.000123456789j+.000001234567i*j for i in 0:3,j in 0:3]
    Y=[.00023456789i+.003123456789j+.000002345678i*j for i in 0:3,j in 0:3]
    originalX=copy(X);originalY=copy(Y)
    default=CMeshMetrics.build_metrics(X,Y,0,3)
    explicit32=CMeshMetrics.build_metrics(X,Y,0,3;scalar_type=Float32)
    full64=CMeshMetrics.build_metrics(X,Y,0,3;scalar_type=Float64)
    for name in (:Six,:Siy,:Sjx,:Sjy,:vol,:xc,:yc,:wdist,:gammat)
        @test eltype(getfield(default,name))==Float32
        @test isequal(getfield(default,name),getfield(explicit32,name))
        @test eltype(getfield(full64,name))==Float64
    end
    @test isequal(X,originalX)&&isequal(Y,originalY)
    @test any(full64.xc .!= Float64.(default.xc))
    @test any(full64.Six .!= Float64.(default.Six))
    @test any(full64.vol .!= Float64.(default.vol))
    @test CMeshMetrics.gcl_check(full64)[1]<1e-13
    @test all(>(0),full64.vol)
    @test_throws ArgumentError CMeshMetrics.build_metrics(X,Y,0,3;scalar_type=Float16)
    object=CMeshGen.CMesh(3,3,0,3,X,Y,.003,ones(4))
    @test isequal(CMeshMetrics.build_metrics(object;scalar_type=Float64).vol,full64.vol)

    # Exercise both channel/strip forwarding paths with an exact small U-grid.
    XC=[1. 1.;-1. -2.;-1. -2.;1. 1.]
    YC=[-1. -2.;-1. -2.;1. 2.;1. 2.]
    channel=CMeshGen.ChannelSpec(height=4.,xi_in=-2.,xi_out=3.,angle_deg=0.)
    kinds=[CMeshGen.OUTER_LOWER_WALL,CMeshGen.OUTER_INLET,CMeshGen.OUTER_UPPER_WALL]
    xs=[Float64(i) for i in 1:3,j in 1:3];ys=[Float64(j-2) for i in 1:3,j in 1:3]
    strip=CMeshGen.StripBlock(2,2,1,xs,ys,[0.,.5,1.],ones(4))
    mesh=CMeshGen.ChannelMesh(3,1,0,3,XC,YC,.1,ones(4),channel,kinds,strip)
    cm=CMeshMetrics.build_metrics(mesh;scalar_type=Float64)
    sm=CMeshMetrics.build_strip_metrics(mesh;scalar_type=Float64)
    @test eltype(cm.vol)==Float64 && cm.channel_provenance
    @test eltype(sm.vol)==Float64 && sm.channel_provenance
    @test sm.outer_face_kind==fill(CMeshGen.OUTER_STRIP_INTERFACE,2)
    @test all(==(1.),sm.gammat)
    @test all(==(1.),sm.vol)
end

@testset "D3Q39 weights are constructed in requested precision" begin
    default=build_d3q39_static();single=build_d3q39_static(Float32);double=build_d3q39_static(Float64)
    @test isequal(default,single)
    @test eltype(default[4])==Float32
    @test eltype(double[4])==Float64
    @test double[4][1] === Float64(1//12)
    @test double[4][1] != Float64(default[4][1])
    @test all(>(0),double[4])
    @test abs(sum(double[4])-1)<=1e-15
    for axis in 1:3
        @test abs(sum(double[4][k]*double[axis][k]^2 for k in 1:39)-2/3)<=1e-15
    end
    @test abs(sum(double[4][k]*double[1][k]*double[2][k] for k in 1:39))<=1e-15
    @test double[1:3]==default[1:3]
    @test double[5:6]==default[5:6]
end
