# Gates G0.3, G0.4, G1.1 and G1.2 of the CPU octree D3Q19 solver on a periodic box, one level:
# rest state, mass and momentum conservation, shear-wave viscosity, Taylor-Green convergence order. CPU, seconds.
#   julia -t 8 --project=src/octree_q19 tools/tests/q19_octree_step.jl
using Test, Printf
using OctreeQ19
const G = OctreeQ19
const ONLY = get(ENV, "Q19_TEST_ONLY", "")          # run only the testsets whose name contains this
want(name) = isempty(ONLY) || occursin(ONLY, name)

function periodic_box(n::Int; dx=1.0, nu=0.01, sgs=:none, precision=Float32, collision=:rr)
    spec = G.MeshSpec(domain=(n * dx, n * dx, n * dx), periodic=(true, true, true), levels=1, dx_finest=dx)
    mesh = G.generate(G.Tri[], spec)
    cfg = G.test_case_config(spec; nu_lattice=nu, sgs=sgs, precision=precision, collision=collision)
    mesh, cfg, G.State(mesh, cfg, precision)
end

function kinetic_energy(st)
    Lv = st.mesh.levels[1]; S = st.levels[1]; ke = 0.0
    for o in S.real_octs, k in 1:8
        Lv.kind[k, o] == G.K_FLUID || continue
        ke += 0.5 * Float64(S.rho[k, o]) * (Float64(S.u[k, 1, o])^2 + Float64(S.u[k, 2, o])^2 + Float64(S.u[k, 3, o])^2)
    end
    ke
end

want("G0.3") && @testset "G0.3 rest state" begin
    mesh, cfg, st = periodic_box(16)
    @test mesh.counts["fluid_total"] == 16^3
    G.initialise!(st, p -> (1.0, 0.0, 0.0, 0.0))
    for _ in 1:50; G.advance!(st); end
    S = st.levels[1]
    @test all(S.f[k, i, o] == 0 for o in S.real_octs, i in 1:19, k in 1:8)      # shifted populations stay exactly zero
    @test all(S.u[k, a, o] == 0 for o in S.real_octs, a in 1:3, k in 1:8)
    @test G.total_mass(st) == 16^3
    # the step allocates nothing per cell (only the thread tasks): the closure-boxing trap of the prototype
    alloc = @allocated G.advance!(st)
    @test alloc < 64 * 1024
end

want("G0.4") && @testset "G0.4 conservation, RR and HRR, FP32 and FP64" begin
    for (precision, tol) in ((Float32, 2e-6), (Float64, 1e-12)), collision in (:rr, :hrr)
        mesh, cfg, st = periodic_box(16; precision=precision, collision=collision)
        G.initialise!(st, p -> (1.0 + 0.01 * sin(2π * p[1] / 16) * cos(2π * p[3] / 16), 0.03 * sin(2π * p[2] / 16), 0.02 * cos(2π * p[3] / 16), 0.01 * sin(2π * p[1] / 16)))
        m0 = G.total_mass(st); p0 = G.total_momentum(st)
        for _ in 1:100; G.advance!(st); end
        @test abs(G.total_mass(st) - m0) / m0 < tol
        @test maximum(abs.(G.total_momentum(st) .- p0)) / (16^3 * 0.03) < tol
        @test all(isfinite, st.levels[1].f)
    end
end

want("G1.1") && @testset "G1.1 shear-wave viscosity" begin
    # u_x = U0 sin(k z), k = 2 pi / n: amplitude decays as exp(-nu k^2 t). The registered size is 64^3 (tau 0.52
    # and 0.505); the 32^3 run shows the second-order discretisation error of the measured viscosity.
    # The decay rate is fitted between two later times, which excludes the initial layer (the start is an
    # equilibrium state whose non-equilibrium part settles over a few steps).
    function measured_nu(n, nu, precision; steps)
        U0 = 0.01
        mesh, cfg, st = periodic_box(n; nu=nu, precision=precision)
        G.initialise!(st, p -> (1.0, U0 * sin(2π * p[3] / n), 0.0, 0.0))
        s1 = steps ÷ 5
        for _ in 1:s1; G.advance!(st); end
        ke1 = kinetic_energy(st)                                   # KE is proportional to the squared amplitude
        for _ in s1+1:steps; G.advance!(st); end
        ke2 = kinetic_energy(st)
        k = 2π / n
        -log(ke2 / ke1) / (2 * k^2 * (steps - s1))
    end
    for (nu, steps64) in ((0.02 / 3, 1500), (0.005 / 3, 6000))      # tau 0.52 and 0.505
        e32 = abs(measured_nu(32, nu, Float64; steps=steps64 ÷ 4) - nu) / nu
        for (precision, tol) in ((Float32, 0.01), (Float64, 0.005))
            nm = measured_nu(64, nu, precision; steps=steps64)
            e64 = abs(nm - nu) / nu
            @printf("shear wave tau %.3f %s: nu %.6f measured %.6f, error %.3f %% at 64^3 (%.3f %% at 32^3)\n", 3nu + 0.5, precision, nu, nm, 100 * e64, 100 * e32)
            @test e64 < tol
            precision == Float64 && @test e32 / e64 > 3.0          # second order in the resolution
        end
    end
end

want("G1.2") && @testset "G1.2 Taylor-Green order" begin
    # u = U0 sin(kx) cos(ky), v = -U0 cos(kx) sin(ky): KE decays as exp(-4 nu k^2 t); diffusive scaling of the step
    # count. The amplitude keeps the O(Ma^2) compressibility error (about 3 Ma^2) below the discretisation errors.
    nu = 0.01; U0 = 0.002
    errors = Float64[]
    for n in (16, 32, 64)
        mesh, cfg, st = periodic_box(n; nu=nu, precision=Float64)
        k = 2π / n
        # the consistent pressure field of the vortex is set with the velocity (rho = 1 + 3 p)
        G.initialise!(st, p -> (1.0 + 3 * (U0^2 / 4) * (cos(2k * p[1]) + cos(2k * p[2])), U0 * sin(k * p[1]) * cos(k * p[2]), -U0 * cos(k * p[1]) * sin(k * p[2]), 0.0))
        steps = 125 * (n ÷ 16)^2; s1 = steps ÷ 5
        for _ in 1:s1; G.advance!(st); end
        ke1 = kinetic_energy(st)
        for _ in s1+1:steps; G.advance!(st); end
        ke2 = kinetic_energy(st)
        expected = exp(-4 * nu * k^2 * (steps - s1))
        err = abs(ke2 / ke1 - expected) / expected
        push!(errors, err)
        @printf("Taylor-Green n=%d: KE ratio %.6f expected %.6f (error %.3e)\n", n, ke2 / ke1, expected, err)
    end
    order = log(errors[1] / errors[2]) / log(2)
    order2 = log(errors[2] / errors[3]) / log(2)
    @printf("Taylor-Green orders: %.2f (16->32), %.2f (32->64)\n", order, order2)
    @test errors[3] < 0.01
    @test min(order, order2) >= 1.8
end
println("q19_octree_step: all tests passed")
