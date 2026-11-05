using Test, OctreeQ19
const G = OctreeQ19

spec = G.MeshSpec(domain=(32.0, 32.0, 32.0), periodic=(true, true, true),
    levels=2, dx_finest=1.0,
    regions=[G.Region(:box, 2; lo=(8.0, 8.0, 8.0), hi=(24.0, 24.0, 24.0))])
mesh = G.generate(G.Tri[], spec)
cfg = G.test_case_config(spec; precision=Float32)

@testset "uniform fast start matches generic field start through AMR" begin
    generic = G.State(mesh, cfg, Float32)
    fast = G.State(mesh, cfg, Float32)
    G.initialise!(generic, p -> (1.0, 0.08, 0.0, 0.0))
    G.initialise_uniform!(fast, 1.0, 0.08, 0.0, 0.0)
    @test generic.mass0 == fast.mass0
    for l in eachindex(mesh.levels)
        @test generic.levels[l].f == fast.levels[l].f
        @test generic.levels[l].g == fast.levels[l].g
        @test generic.levels[l].u == fast.levels[l].u
        @test generic.levels[l].rho == fast.levels[l].rho
    end
    G.advance!(generic)
    G.advance!(fast)
    for l in eachindex(mesh.levels)
        @test generic.levels[l].f == fast.levels[l].f
        @test generic.levels[l].u == fast.levels[l].u
        @test generic.levels[l].rho == fast.levels[l].rho
    end
end
