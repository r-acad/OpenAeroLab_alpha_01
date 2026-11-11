using Test, OctreeQ19, SHA, TOML
const G=OctreeQ19
include(joinpath(@__DIR__,"q19_octree_geometry_helpers.jl"))

@testset "refinement rejects invalid populations" begin
    @test G.valid_refinement_populations(ntuple(_->0f0,Val(G.Q)))
    @test !G.valid_refinement_populations(ntuple(i->i==1 ? Inf32 : 0f0,Val(G.Q)))
    @test !G.valid_refinement_populations(ntuple(i->i==2 ? -1f0 : 0f0,Val(G.Q)))
    # Large finite, oppositely signed stresses must not enter the next stage.
    extreme=ntuple(i->i==2 ? floatmax(Float32)/4 : i==3 ? -floatmax(Float32)/4 : 0f0,Val(G.Q))
    @test all(isfinite,extreme)
    @test !G.valid_refinement_populations(extreme)
end

function stage(spec; nu=.01, T=Float64, tris=G.Tri[])
    cfg=G.test_case_config(spec;nu_lattice=nu,precision=T)
    G.State(G.generate(tris,spec),cfg,T)
end

@testset "refinement startup: uniform and evolved periodic flow" begin
    for T in (Float32,Float64)
        coarse=G.MeshSpec(domain=(16.,16.,16.),periodic=(true,true,true),levels=1,dx_finest=2.)
        fine=G.MeshSpec(domain=(16.,16.,16.),periodic=(true,true,true),levels=2,dx_finest=1.,regions=[G.Region(:box,2;lo=(4.,4.,4.),hi=(12.,12.,12.))])
        src=stage(coarse;nu=.005,T=T);dst=stage(fine;nu=.01,T=T)
        G.initialise!(src,p->(1.0,0.02*sin(2pi*p[3]/16),0.0,0.0))
        for _ in 1:5; G.advance!(src); end
        mass=G.total_mass(src)*src.scaling.dx_finest^3
        momentum=G.total_momentum(src).*src.scaling.dx_finest^3
        report=G.transfer_refined!(dst,src)
        @test report["source_step"]==5
        @test report["visible_fallback_cells"]==0
        @test dst.step==src.step
        unchanged=0
        for (l,L) in enumerate(dst.mesh.levels), o in dst.levels[l].real_octs, k in 1:8
            L.kind[k,o]==G.K_FLUID || continue
            p=G.slot_centre(L,G.slot_id(k,o));sl,ss=G.locate_leaf(src.mesh,p)
            sl>0 && src.mesh.levels[sl].dx==L.dx || continue
            sk=G.slot_k(ss);so=G.slot_oct(ss)
            @test dst.levels[l].f[k,:,o]==src.levels[sl].f[sk,:,so]
            unchanged+=1
        end
        @test unchanged>0
        @test isapprox(G.total_mass(dst)*dst.scaling.dx_finest^3,mass;rtol=T===Float32 ? 2e-6 : 1e-12)
        @test all(isapprox.(G.total_momentum(dst).*dst.scaling.dx_finest^3,momentum;atol=T===Float32 ? 1e-3 : 1e-10))
        for _ in 1:5; G.advance!(dst); end
        @test all(isfinite,G.state_extrema(dst))
        @test isapprox(G.total_mass(dst)*dst.scaling.dx_finest^3,mass;rtol=5e-6)
        @test_throws ErrorException G.check_refinement_start(dst,src)
    end
end

@testset "refinement startup does not transfer through a thin panel" begin
    a=(7.2,4.,4.);b=(7.2,12.,4.);c=(7.2,12.,12.);d=(7.2,4.,12.)
    tris=G.Tri[(a,b,c),(a,c,d)]
    coarse=G.MeshSpec(domain=(16.,16.,16.),periodic=(true,true,true),levels=1,dx_finest=2.)
    fine=G.MeshSpec(domain=(16.,16.,16.),periodic=(true,true,true),levels=2,dx_finest=1.,regions=[G.Region(:box,2;lo=(4.,4.,4.),hi=(12.,12.,12.))])
    src=stage(coarse;nu=.005,tris);dst=stage(fine;nu=.01,tris)
    G.initialise!(src,p->(1.0,p[1]>7.2 ? 0.02 : -0.02,0.,0.))
    report=G.transfer_refined!(dst,src)
    l,s=G.locate_leaf(dst.mesh,(7.5,8.5,8.5));@test l==2
    @test dst.levels[l].u[G.slot_k(s),1,G.slot_oct(s)]>0
    @test report["cross_surface_donors_rejected"]>0
    # Last-resort wall equilibrium is an explicit startup guess and is only
    # available within a quarter cell of real geometry.
    bvh=G.build_bvh(tris);stack=G.new_stack();S=dst.levels[l];k=G.slot_k(s);o=G.slot_oct(s)
    distance=G.refinement_wall_seed!(S,k,o,(7.21,8.5,8.5),1.,bvh,stack)
    @test distance≈.01
    @test all(iszero,S.f[k,:,o]) && all(iszero,S.g[k,:,o])
    @test S.rho[k,o]==1 && all(iszero,S.u[k,:,o])
    @test_throws ErrorException G.refinement_wall_seed!(S,k,o,(7.5,8.5,8.5),1.,bvh,stack)
end

@testset "refinement startup: visible donors near a closed body" begin
    tris=sphere_tris(10,20;r=3.5,c=(8.,8.,8.))
    coarse=G.MeshSpec(domain=(16.,16.,16.),periodic=(true,true,true),levels=1,dx_finest=2.,seal_cavities=true)
    fine=G.MeshSpec(domain=(16.,16.,16.),periodic=(true,true,true),levels=2,dx_finest=1.,seal_cavities=true)
    src=stage(coarse;nu=.005,tris);dst=stage(fine;nu=.01,tris)
    G.initialise_uniform!(src,1.,.08,0,0)
    report=G.transfer_refined!(dst,src)
    @test report["visible_fallback_cells"]==96
    @test report["maximum_fallback_distance_m"]<4
    @test report["source_fluid_volume_m3"]!=report["target_fluid_volume_m3"]
    @test isapprox(report["source_mass_lattice_m3"],report["source_fluid_volume_m3"];rtol=1e-12)
    @test isapprox(report["target_mass_lattice_m3"],report["target_fluid_volume_m3"];rtol=1e-12)
    @test all(isfinite,G.state_extrema(dst))
    for _ in 1:3; G.advance!(dst); end
    @test all(isfinite,G.state_extrema(dst))
end

@testset "completed checkpoint starts a new case without changing the source" begin
    private=get(ENV,"LUDWIG_PRIVATE_ROOT",normpath(joinpath(@__DIR__,"..","..","..","02_PRIVATE_PROJECT_DEVELOPMENT")))
    scratch=joinpath(private,"TEMP_DELETEABLE","q19_refinement_start");mkpath(scratch)
    mktempdir(scratch;prefix="handoff_") do root
        srcdir=joinpath(root,"coarse");dstdir=joinpath(root,"fine");mkpath(srcdir);mkpath(dstdir)
        model=joinpath(srcdir,"model.stl")
        open(model,"w") do io
            println(io,"solid sphere")
            for tri in sphere_tris(6,12;r=.25)
                println(io,"facet normal 0 0 0\nouter loop")
                for v in tri; println(io,"vertex ",join(v," ")); end
                println(io,"endloop\nendfacet")
            end
            println(io,"endsolid sphere")
        end
        cp(model,joinpath(dstdir,"model.stl"))
        function config(levels,dx,steps)
            """
            basic:
              stl_file: [model.stl]
              reference_chord: 1.0
              reference_length_for_meshing: 1.0
              fluid: {density: 1.225, kinematic_viscosity: 0.001, speed_of_sound: 343.0}
              flow: {velocity: 1.0}
              simulation: {build_only: false, steps: $steps, ramp_steps: 0, output_freq: 999, slice_output_freq: 999}
            advanced:
              domain: {upstream: 1.0, downstream: 1.0, lateral: 1.0, height: 1.0}
              refinement: {symmetric_analysis: false}
              forces: {output_freq: 1}
              slice: {planes: []}
            octree_q19:
              precision: f64
              mesh: {levels: $levels, finest_dx_m: $dx, band_fine: 3, band: 3, wake: {length: 0.0}}
              numerics: {collision: rr, sgs: none, u_lattice: 0.05}
              checkpoint: {every_minutes: 30, keep: 1}
              outputs: {surface_lbmp: false, surface_vtk: false}
            """
        end
        write(joinpath(srcdir,"config.yaml"),config(1,.25,2))
        write(joinpath(dstdir,"config.yaml"),config(2,.125,3))
        src=G.load_case(srcdir);dst=G.load_case(dstdir)
        srcstate=G.run_case(src)
        cpfile=joinpath(srcdir,"RESULTS","checkpoints","checkpoint_000000002.bin")
        @test isfile(cpfile)
        digest=bytes2hex(open(sha256,cpfile))
        restored=G.State(srcstate.mesh,src,Float64)
        histories=G.open_histories(joinpath(root,"resume_history"))
        try
            @test G.resume_checkpoint!(restored,joinpath(srcdir,"RESULTS"),histories)==2
            @test restored.levels[1].f==srcstate.levels[1].f
            @test restored.levels[1].rho==srcstate.levels[1].rho
        finally
            G.close_histories(histories)
        end
        corrupted=joinpath(root,"checkpoint_with_trailing_byte.bin");cp(cpfile,corrupted)
        open(corrupted,"a") do io;write(io,UInt8(0));end
        @test_throws ErrorException G.read_checkpoint!(restored,corrupted,nothing)
        meshfile=joinpath(srcdir,"mesh.oct");meshdigest=bytes2hex(open(sha256,meshfile))
        changed=G.CaseConfig(ntuple(i->fieldname(G.CaseConfig,i)==:config_sha ? "different-config" : getfield(src,i),fieldcount(G.CaseConfig))...)
        @test_throws ErrorException G.run_case(changed;from_case=srcdir)
        @test bytes2hex(open(sha256,meshfile))==meshdigest
        @test bytes2hex(open(sha256,cpfile))==digest
        # A refinement target must already have a matching mesh; never rebuild
        # a mismatched target implicitly when this mode is selected.
        @test_throws ErrorException G.run_case(dst;from_case=srcdir)
        @test !isfile(joinpath(dstdir,"mesh.oct"))
        G.build_only_case(dst)
        @test_throws ErrorException G.run_case(dst;from_case=srcdir,expected_checkpoint=cpfile,expected_checkpoint_bytes=filesize(cpfile)+1,expected_checkpoint_mtime=stat(cpfile).mtime)
        @test !isdir(joinpath(dstdir,"RESULTS","csv"))
        @test_throws ErrorException G.run_case(dst;from_case=srcdir,expected_checkpoint=cpfile,expected_checkpoint_bytes=filesize(cpfile),expected_checkpoint_mtime=stat(cpfile).mtime+1)
        @test_throws ErrorException G.run_case(dst;from_case=srcdir,expected_checkpoint=corrupted,expected_checkpoint_bytes=filesize(corrupted),expected_checkpoint_mtime=stat(corrupted).mtime)
        @test !isdir(joinpath(dstdir,"RESULTS","csv"))
        @test_throws ErrorException G.run_case(dst;from_case=srcdir,expected_checkpoint=cpfile)
        out=G.run_case(dst;from_case=srcdir,expected_checkpoint=cpfile,expected_checkpoint_bytes=filesize(cpfile),expected_checkpoint_mtime=stat(cpfile).mtime)
        @test out.step==3
        @test bytes2hex(open(sha256,cpfile))==digest
        receipt=TOML.parsefile(joinpath(dstdir,"RESULTS","refinement_start.toml"))
        @test receipt["source_step"]==2
        @test receipt["source_checkpoint"]==abspath(cpfile)
        @test isapprox(receipt["source_physical_time_s"],2*src.mesh.dx_finest*src.numerics.u_lattice/src.flow.velocity)
        @test_throws ErrorException G.run_case(dst;from_case=srcdir)
        @test_throws ErrorException G.run_case(dst;resume=true,from_case=srcdir)
        @test bytes2hex(open(sha256,cpfile))==digest
    end
end
