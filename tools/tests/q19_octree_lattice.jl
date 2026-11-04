# Gate G0.1 and G0.2 of the CPU octree D3Q19 solver: its lattice tables against the main solver's
# `build_d3q19_static` (src/collision/lattice.jl), and the lattice identities. CPU only, no CUDA, seconds.
#   julia --project=src/octree_q19 tools/tests/q19_octree_lattice.jl
using Test
root = normpath(joinpath(@__DIR__, "..", ".."))
using OctreeQ19
const L = OctreeQ19

# The main solver's builder needs its precision alias and StaticArrays, which comes from the main `src` environment
# stacked behind this package's environment (no CUDA is loaded: only lattice.jl is included).
push!(LOAD_PATH, joinpath(root, "src"))
module MainLattice
const LBM_F = Float32
const F = Float32
include(joinpath(normpath(joinpath(@__DIR__, "..", "..")), "src", "collision", "lattice.jl"))
end

@testset "G0.1 tables against build_d3q19_static" begin
    cx, cy, cz, w, opp, my = MainLattice.build_d3q19_static()
    @test length(cx) == 19
    main = Dict((Int(cx[i]), Int(cy[i]), Int(cz[i])) => Float64(w[i]) for i in 1:19)
    ours = Dict((L.CX[i], L.CY[i], L.CZ[i]) => L.WT[i] for i in 1:19)
    @test Set(keys(main)) == Set(keys(ours))
    for (k, v) in ours
        @test isapprox(main[k], v; atol=1e-7)
    end
    # opposite and y-mirror tables agree as maps on velocities
    for i in 1:19
        j = L.OPP[i]
        @test (L.CX[j], L.CY[j], L.CZ[j]) == (-L.CX[i], -L.CY[i], -L.CZ[i])
        m = L.MIRROR_Y[i]
        @test (L.CX[m], L.CY[m], L.CZ[m]) == (L.CX[i], -L.CY[i], L.CZ[i])
        im = findfirst(k -> (Int(cx[k]), Int(cy[k]), Int(cz[k])) == (L.CX[i], L.CY[i], L.CZ[i]), 1:19)
        @test (Int(cx[opp[im]]), Int(cy[opp[im]]), Int(cz[opp[im]])) == (-L.CX[i], -L.CY[i], -L.CZ[i])
        @test (Int(cx[my[im]]), Int(cy[my[im]]), Int(cz[my[im]])) == (L.CX[i], -L.CY[i], L.CZ[i])
    end
    @test all(L.OPP[L.OPP[i]] == i for i in 1:19)
    @test all(L.MIRROR_X[L.MIRROR_X[i]] == i for i in 1:19)
    @test all(L.MIRROR_Z[L.MIRROR_Z[i]] == i for i in 1:19)
    @test all(L.OPP[i] == i + 1 for i in 2:2:18)   # adjacent opposite pairs
end

@testset "G0.2 lattice identities" begin
    w = collect(L.WT); c = [(L.CX[i], L.CY[i], L.CZ[i]) for i in 1:19]
    @test sum(w) ≈ 1 atol=1e-15
    for a in 1:3, b in 1:3
        @test sum(w[i] * c[i][a] * c[i][b] for i in 1:19) ≈ (a == b ? 1 / 3 : 0) atol=1e-15
    end
    # fourth-order isotropy: sum w c_a c_b c_c c_d = cs^4 (d_ab d_cd + d_ac d_bd + d_ad d_bc)
    d(a, b) = a == b ? 1 : 0
    for a in 1:3, b in 1:3, cc in 1:3, dd in 1:3
        expected = (1 / 9) * (d(a, b) * d(cc, dd) + d(a, cc) * d(b, dd) + d(a, dd) * d(b, cc))
        @test sum(w[i] * c[i][a] * c[i][b] * c[i][cc] * c[i][dd] for i in 1:19) ≈ expected atol=1e-15
    end
    # odd moments vanish
    for a in 1:3
        @test sum(w[i] * c[i][a] for i in 1:19) ≈ 0 atol=1e-15
        for b in 1:3, cc in 1:3
            @test sum(w[i] * c[i][a] * c[i][b] * c[i][cc] for i in 1:19) ≈ 0 atol=1e-15
        end
    end
    # equilibrium moments: density and momentum exact, stress to second order
    rho = 1.07; u = (0.05, -0.03, 0.02); usq = sum(abs2, u)
    f = ntuple(i -> L.feq(i, rho, u..., usq), 19)
    r, ux, uy, uz = L.macros(f)
    @test r ≈ rho atol=1e-14
    @test (ux, uy, uz) == (u...,) || all(isapprox.((ux, uy, uz), u; atol=1e-14))
    p = L.pi_neq(f, rho, u...)
    @test all(abs.(p) .< 1e-14)
    # the regularised non-equilibrium of a pure a2 state reproduces a2 (G0.5 projection consistency)
    a2 = (0.01, -0.004, -0.006, 0.003, -0.002, 0.001)
    fneq = ntuple(i -> L.fneq_regularised(i, a2, 0.0, 0.0, 0.0), 19)
    g = ntuple(i -> f[i] + fneq[i], 19)
    p2 = L.pi_neq(g, rho, u...)
    @test all(isapprox.(p2, a2; atol=1e-12))
    @test sum(fneq) ≈ 0 atol=1e-15
    @test all(abs(sum(fneq[i] * c[i][a] for i in 1:19)) < 1e-15 for a in 1:3)
end
println("q19_octree_lattice: all tests passed")
