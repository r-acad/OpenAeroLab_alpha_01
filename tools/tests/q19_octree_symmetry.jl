# A body crossing the Y=0 symmetry plane must retain its finest wall cells.
# Run: julia -t 4 --project=src/octree_q19 tools/tests/q19_octree_symmetry.jl
using Test, OctreeQ19
const G=OctreeQ19
include(joinpath(@__DIR__,"q19_octree_geometry_helpers.jl"))
const FACES=(:inlet,:outlet,:symmetry,:freestream,:freestream,:freestream)

function check_boundary_topology(mesh)
    fine_boundary_cells=0
    for (l,Lv) in enumerate(mesh.levels)
        for o in 1:Lv.nocts,k in 1:8
            kind=Lv.kind[k,o]
            if kind==G.K_FLUID
                @test all(!iszero,Lv.onbr[:,o])
            elseif kind==G.K_BOUNDARY && l>1
                fine_boundary_cells+=1
                @test G.slot_cell(Lv,G.slot_id(k,o))[2]<0
                @test Lv.face[k,o]==G.face_bit(G.FACE_YMIN)
            end
        end
        if l>1
            parent=mesh.levels[l-1]
            for i in eachindex(Lv.ghost_slot)
                @test isapprox(sum(Lv.ghost_w[:,i]),1.;atol=1e-6)
                @test all(d==0 || parent.kind[G.slot_k(d),G.slot_oct(d)] in (G.K_FLUID,G.K_GHOST,G.K_COVERED) for d in Lv.ghost_donor[:,i])
            end
        end
    end
    @test fine_boundary_cells>0
end

@testset "Finest surface cells reach a reflecting symmetry plane" begin
    tri=sphere_tris(24,48;r=.4,c=(1.5,0.,1.5))
    spec=G.MeshSpec(domain=(4.,2.,3.2),levels=3,dx_finest=.05,band_fine=4,band=3,faces=FACES)
    mesh=G.generate(tri,spec)
    @test mesh.counts["symmetry_boundary_refinement"]==1
    check_boundary_topology(mesh)
    @test G.locate_leaf(mesh,(1.925,.025,1.525))[1]==3
    l,s=G.locate_leaf(mesh,(1.5,.025,1.5))
    @test mesh.levels[l].kind[G.slot_k(s),G.slot_oct(s)]==G.K_SOLID
    fine=mesh.levels[end]
    half_wall_count=count(s->G.slot_cell(fine,s)[2]==0,fine.cut_slot)
    @test half_wall_count>0
    full_tri=sphere_tris(24,48;r=.4,c=(1.5,2.,1.5))
    full_spec=G.MeshSpec(domain=(4.,4.,3.2),levels=3,dx_finest=.05,band_fine=4,band=3,
        faces=(:inlet,:outlet,:freestream,:freestream,:freestream,:freestream))
    full=G.generate(full_tri,full_spec).levels[end]
    @test half_wall_count==count(s->G.slot_cell(full,s)[2]==40,full.cut_slot)
    @test mesh.counts["reports"]["leak_cells"]==0
    # Positive-Y half-sphere volume and all leaf cells partition the half-domain.
    per=mesh.counts["per_level"]
    @test isapprox(sum(p["solid"]*p["dx_m"]^3 for p in per),2pi*.4^3/3;rtol=.08)
    @test isapprox(sum((p["fluid"]+p["solid"])*p["dx_m"]^3 for p in per),prod(mesh.domain);rtol=1e-12)
    cfg=G.test_case_config(spec;nu_lattice=.05,u_lattice=.02,precision=Float64)
    state=G.State(mesh,cfg,Float64)
    state.u_inlet=.02
    G.initialise!(state,p->(1.,.02,0.,0.));G.couple_initial!(state)
    for _ in 1:12;G.advance!(state);end
    for (Lv,S) in zip(mesh.levels,state.levels)
        values=[S.rho[k,o] for o in S.real_octs,k in 1:8 if Lv.kind[k,o]==G.K_FLUID]
        @test all(isfinite,values)
        @test minimum(values)>.9 && maximum(values)<1.1
        @test all(isfinite,S.f)
    end
end

@testset "Tangential uniform flow through an AMR patch touching symmetry" begin
    region=G.Region(:box,3;lo=(.8,0.,.8),hi=(2.2,.6,2.2))
    spec=G.MeshSpec(domain=(4.,2.,3.2),levels=3,dx_finest=.05,band_fine=4,band=3,faces=FACES,regions=[region])
    mesh=G.generate(G.Tri[],spec)
    check_boundary_topology(mesh)
    cfg=G.test_case_config(spec;nu_lattice=.05,u_lattice=.04,precision=Float64)
    state=G.State(mesh,cfg,Float64)
    state.u_inlet=.04
    G.initialise!(state,p->(1.,.04,0.,0.));G.couple_initial!(state)
    initial=G.total_mass(state)
    for _ in 1:24;G.advance!(state);end
    @test abs(G.total_mass(state)-initial)<1e-10
    error=0.
    for (Lv,S) in zip(mesh.levels,state.levels),o in S.real_octs,k in 1:8
        Lv.kind[k,o]==G.K_FLUID || continue
        error=max(error,abs(S.rho[k,o]-1.),abs(S.u[k,1,o]-.04),abs(S.u[k,2,o]),abs(S.u[k,3,o]))
    end
    @test error<1e-12
end
