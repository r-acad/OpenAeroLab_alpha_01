# Positivity safeguard for under-resolved wall and AMR states.
# julia --project=src/octree_q19 tools/tests/q19_positivity.jl
using Test, OctreeQ19
const G = OctreeQ19

@testset "Q19 kinetic positivity blend" begin
    rho = 1.03f0; ux = 0.08f0; uy = 0f0; uz = 0f0
    eq = ntuple(i -> G.feq_shifted(i, rho, ux, uy, uz, ux^2), Val(G.Q))
    before = G.POSITIVITY_LIMITED_CELLS[]
    @test G.positive_shifted(eq, eq) === eq
    @test G.POSITIVITY_LIMITED_CELLS[] == before

    # A symmetric pair loses population while the rest direction gains the
    # same mass. Momentum stays fixed, so relaxation toward equilibrium can
    # restore positivity without changing the conserved moments.
    disturbed = ntuple(i -> i == 1 ? eq[i] + 0.16f0 : i == 2 || i == 3 ? eq[i] - 0.08f0 : eq[i], Val(G.Q))
    @test any(disturbed[i] + Float32(G.WT[i]) < 0 for i in 1:G.Q)
    repaired = G.positive_shifted(disturbed, eq)
    @test all(isfinite(repaired[i]) && repaired[i] + Float32(G.WT[i]) > 0 for i in 1:G.Q)
    @test isapprox(G.macros_shifted(repaired)[1], G.macros_shifted(disturbed)[1]; atol=2f-7)
    @test all(isapprox(G.macros_shifted(repaired)[i], G.macros_shifted(disturbed)[i]; atol=2f-7) for i in 2:4)
    @test G.POSITIVITY_LIMITED_CELLS[] == before + 1
end

@testset "Q19 forced tuple conserves post-force moments" begin
    rho = 1.01f0; ux = 0.04f0; uy = -0.02f0; uz = 0.01f0
    eq = ntuple(i -> G.feq_shifted(i, rho, ux, uy, uz, ux^2 + uy^2 + uz^2), Val(G.Q))
    disturbed = ntuple(i -> i == 1 ? eq[i] + 0.14f0 : i == 2 || i == 3 ? eq[i] - 0.07f0 : eq[i], Val(G.Q))
    repaired = G.positive_shifted_forced(disturbed)
    @test all(repaired[i] + Float32(G.WT[i]) > 0 for i in 1:G.Q)
    @test all(isapprox(G.macros_shifted(repaired)[i], G.macros_shifted(disturbed)[i]; atol=2f-7) for i in 1:4)
end

@testset "Q19 checkpoint repair" begin
    spec = G.MeshSpec(domain=(4.0, 4.0, 4.0), periodic=(true, true, true), levels=1, dx_finest=1.0)
    mesh = G.generate(G.Tri[], spec)
    cfg = G.test_case_config(spec; nu_lattice=0.01)
    st = G.State(mesh, cfg, Float32)
    G.initialise_uniform!(st, 1.0, 0.0, 0.0, 0.0)
    o = Int(st.levels[1].real_octs[1]); k = 1
    for buffer in (st.levels[1].f, st.levels[1].g)
        buffer[k, 1, o] = 0.16f0
        buffer[k, 2, o] = -0.08f0
        buffer[k, 3, o] = -0.08f0
    end
    before = ntuple(i -> st.levels[1].f[k,i,o], Val(G.Q))
    repaired = G.repair_state_positivity!(st)
    after = ntuple(i -> st.levels[1].f[k,i,o], Val(G.Q))
    @test repaired == 2
    @test all(st.levels[1].f[k,i,o] + Float32(G.WT[i]) > 0 for i in 1:G.Q)
    @test all(st.levels[1].g[k,i,o] + Float32(G.WT[i]) > 0 for i in 1:G.Q)
    @test all(isapprox(G.macros_shifted(after)[i], G.macros_shifted(before)[i]; atol=2f-7) for i in 1:4)
end
println("q19_positivity: all tests passed")
