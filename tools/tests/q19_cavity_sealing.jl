# julia -t 4 --project=src/octree_q19 tools/tests/q19_cavity_sealing.jl
using Test,OctreeQ19
const G=OctreeQ19
include(joinpath(@__DIR__,"q19_octree_geometry_helpers.jl"))

quad_x(x,y0,y1,z0,z1) = G.Tri[
    ((x,y0,z0),(x,y1,z0),(x,y1,z1)),
    ((x,y0,z0),(x,y1,z1),(x,y0,z1))]
quad_z(z,x0,x1,y0,y1) = G.Tri[
    ((x0,y0,z),(x1,y0,z),(x1,y1,z)),
    ((x0,y0,z),(x1,y1,z),(x0,y1,z))]
const PANELS=vcat(quad_x(2.,.75,1.25,.75,1.25),quad_z(2.,.75,1.25,.75,1.25))
const P=(1.0625,.9375,1.1875)
const BVH=G.build_bvh(PANELS)
const STACK=Vector{Int32}(undef,2length(BVH.lo)+10)
kind_at(mesh,p) = (ls=G.locate_leaf(mesh,p); @assert ls[1]>0; mesh.levels[ls[1]].kind[G.slot_k(ls[2]),G.slot_oct(ls[2])])

@testset "No detached phantom solid from open-surface ray projections" begin
    pos=[G.ray_crossings(BVH,P,a,STACK) for a in 1:3]
    neg=[G.ray_crossings(BVH,P,a,STACK,Val(-1)) for a in 1:3]
    @test isodd(pos[1])&&!isodd(pos[2])&&isodd(pos[3])
    @test all(iseven,neg)
    @test G.is_inside(BVH,P,STACK) # old majority-only rule made this exterior point solid
    @test G.cavity_seal_decision(BVH,P,STACK,false)==2
    @test G.cavity_seal_decision(BVH,P,STACK,true)==1
    mesh=G.generate(PANELS,G.MeshSpec(domain=(4.,4.,4.),levels=1,dx_finest=.125,band_fine=4,band=3,seal_cavities=true))
    @test kind_at(mesh,P)==G.K_FLUID
    @test mesh.counts["reports"]["cavity_unbracketed_parity_cells"]>0
    @test mesh.counts["cavity_seal_policy_version"]==2
end

@testset "Open cavity with independent enclosing surfaces remains sealed" begin
    sphere=sphere_tris(16,32;r=.5,c=(2.,2.,2.))
    open_sphere=[t for t in sphere if !all(v[3]>2.42 for v in t)]
    bvh=G.build_bvh(open_sphere);stack=Vector{Int32}(undef,2length(bvh.lo)+10)
    center=(2.025,2.025,2.025)
    @test G.cavity_seal_decision(bvh,center,stack,false)==1
    settings=G.MeshSpec(domain=(4.,4.,4.),levels=2,dx_finest=.05,band_fine=4,band=3,seal_cavities=true)
    mesh=G.generate(open_sphere,settings)
    @test kind_at(mesh,center)==G.K_SOLID
    @test kind_at(mesh,(2.025,2.025,2.825))==G.K_FLUID
    @test mesh.counts["reports"]["sealed_cells"]>0
end
