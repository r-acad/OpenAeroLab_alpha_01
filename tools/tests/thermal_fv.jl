# Run from the repository: julia --startup-file=no --project=src tools/tests/thermal_fv.jl
# In-memory CPU regressions; no private fixtures, generated files, or CUDA required.
using Test, Random
const REPOSITORY = dirname(dirname(@__DIR__))
include(joinpath(REPOSITORY,"src/collision/pond/pond_lbm_mesh.jl"))
include(joinpath(REPOSITORY,"src/collision/pond/pond_thermal_fv.jl"))
using .PondLBMMesh, .PondThermalFV
const Flux = PondThermalFV.PonDThermalFlux

@testset "Maxwellian flux: Euler consistency and orientation" begin
    rng=MersenneTwister(20261102)
    for T in (Float32,Float64), axis in 1:3, direction in (-1,1), _ in 1:40
        gamma=T(1.4);rho=T(.1+2rand(rng));p=T(.1+2rand(rng))
        q=(rho,T(2rand(rng)-1),T(2rand(rng)-1),T(2rand(rng)-1),p)
        E=p/(gamma-1)+rho*sum(abs2,q[2:4])/2
        un=q[axis+1]*direction
        expected=(rho*un,ntuple(a->rho*q[a+1]*un+(a==axis ? direction*p : zero(T)),3)...,(E+p)*un)
        tolerance=T===Float32 ? 3e-6 : 3e-14
        @test all(isapprox.(Flux.kinetic_flux(q,q,axis,direction,gamma),expected;rtol=tolerance,atol=tolerance))
        r=(rho*T(.7),-q[2],q[3]*T(.5),q[4]*T(1.2),p*T(1.1))
        forward=Flux.kinetic_flux(q,r,axis,direction,gamma)
        backward=Flux.kinetic_flux(r,q,axis,-direction,gamma)
        @test all(isapprox.(forward,.-backward;rtol=tolerance,atol=tolerance))
    end
end

function block_fixture(dx,coords)
    obstacle=falses(2,2,2,length(coords))
    (;dx,active_block_coords=coords,obstacle,wall_dist=fill(2.5dx,size(obstacle)),amr_covered=falses(size(obstacle)))
end

@testset "Shared-face AMR: closure, uniform preservation and conservation" begin
    coarse=block_fixture(1.,[(i,j,k) for k in 1:2 for j in 1:2 for i in 1:2])
    # Refine one parent cell at the periodic X boundary: coarse/fine neighbours
    # occur both inside the domain and across its periodic seam.
    fine=block_fixture(.5,[(4,3,3)])
    coarse.amr_covered[2,1,1,8]=true
    periodic=(true,true,true)
    mesh=build_pond_lbm_mesh([coarse,fine];block_size=2,periodic)
    @test length(mesh.x)==71
    @test sum(mesh.volume)==64
    @test all(mesh.neighbor.>0)
    @test mesh.metadata["max_relative_face_closure"]<1e-14
    @test mesh.metadata["max_relative_face_area_error"]<1e-14
    visits=zeros(Int,length(mesh.owner))
    for i in eachindex(mesh.x), k in mesh.cell_face_offsets[i]:mesh.cell_face_offsets[i+1]-1
        sf=mesh.cell_faces[k]
        @test (sf>0 ? mesh.owner[abs(sf)] : mesh.neighbor[abs(sf)])==i
        visits[abs(sf)]+=sign(sf)
    end
    @test all(iszero,visits)
    @test all(find_leaf(mesh,mesh.x[i],mesh.y[i],mesh.z[i])==i for i in eachindex(mesh.x))
    far=(1.,.8,.15,-.05,1/1.4)
    s=ThermalFV(mesh;T=Float64,periodic,freestream=far)
    initial=copy(s.q)
    for _ in 1:30;advance!(s);end
    @test maximum(abs,s.q-initial)<1e-12
    @test s.rejected_steps==0
    for mu in (0.,.002)
        perturbation(i,x,y,z)=(1+.05sinpi(x/2)*cospi(y/2),.8+.02sinpi(z/2),.15,-.05,1/1.4)
        s=ThermalFV(mesh;T=Float64,periodic,initial=perturbation,freestream=far,mu)
        before=conserved_totals(s)
        for _ in 1:30;advance!(s);end
        after=conserved_totals(s)
        @test maximum(abs.(after-before)./max.(abs.(before),1.))<1e-12
        P=primitives(s)
        @test all(isfinite,P)
        @test minimum(P[:,1])>0 && minimum(P[:,5])>0
        @test s.rejected_steps==0
    end
    coarse.amr_covered[2,1,1,8]=false
    @test_throws ArgumentError build_pond_lbm_mesh([coarse,fine];block_size=2,periodic)
end
