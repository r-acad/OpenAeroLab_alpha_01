# Gate G0.7 of the CPU octree D3Q19 solver: the wall laws, their inversion, and the unit conversions, against
# hand values. CPU, seconds.
#   julia --project=src/octree_q19 tools/tests/q19_octree_walls.jl
using Test
using OctreeQ19
const G = OctreeQ19

@testset "laws of the wall" begin
    # Reichardt: u+ ~ y+ in the sublayer, the log law far out; monotone
    @test isapprox(G.uplus_reichardt(0.5), 0.5; rtol=0.05)
    @test isapprox(G.uplus_reichardt(1000.0), log(1000) / 0.41 + 5.2; rtol=0.02)
    ys = 10 .^ range(-1, 4; length=200)
    @test all(diff(G.uplus_reichardt.(ys)) .> 0)
    # the derivative matches a finite difference
    for yp in (0.3, 5.0, 30.0, 500.0)
        fd = (G.uplus_reichardt(yp * 1.0001) - G.uplus_reichardt(yp * 0.9999)) / (yp * 0.0002)
        @test isapprox(G.duplus_reichardt(yp), fd; rtol=1e-4)
    end
    # Spalding: y+(u+) inverted exactly; sublayer and log limits
    for yp in (0.2, 3.0, 11.0, 50.0, 2000.0)
        up = G.uplus_spalding(yp)
        @test isapprox(G.yplus_spalding(up), yp; rtol=1e-6)
    end
    @test isapprox(G.uplus_spalding(0.3), 0.3; rtol=0.01)
    @test isapprox(G.uplus_spalding(3000.0), log(3000) / 0.41 + 5.2; rtol=0.01)
    for T in (Float32,Float64), yp in T.((1e-15,1e-7,.001,.1,1.,100.))
        up=G.uplus_spalding(yp)
        @test isapprox(G.yplus_spalding(up),yp;rtol=T===Float32 ? 3e-6 : 1e-7)
        @test isfinite(G.duplus_spalding(yp)) && 0<G.duplus_spalding(yp)<=1
    end
    # Werner-Wengle continuity at y+ = 11.81
    @test isapprox(G.uplus_werner_wengle(11.81 - 1e-9), G.uplus_werner_wengle(11.81 + 1e-9); rtol=1e-3)
    @test_throws ArgumentError G.uplus(:unknown, 1.0)
    @test_throws ArgumentError G.uplus(:reichardt, -1.0)
    @test_throws ArgumentError G.uplus(:reichardt, Inf)
    # Single precision must retain the viscous limit at small wall distances;
    # 1-exp(-x) previously lost enough precision to corrupt inversion.
    for yp in (1f-7, 1f-5, 5f-4, .01f0)
        @test isapprox(G.uplus_reichardt(yp), Float32(G.uplus_reichardt(Float64(yp))); rtol=3e-6)
    end
end

@testset "u_tau inversion" begin
    for law in (:reichardt, :spalding, :werner_wengle), (y, nu, ut_true) in ((2.0, 1e-3, 0.02), (0.5, 2e-3, 0.004), (10.0, 1e-4, 0.05))
        yp = y * ut_true / nu
        u = ut_true * G.uplus(law, yp)
        ut = G.u_tau_from_law(law, u, y, nu)
        @test isapprox(ut, ut_true; rtol=1e-5)
        # the warm start converges to the same value
        @test isapprox(G.u_tau_from_law(law, u, y, nu; guess=1.3ut_true), ut_true; rtol=1e-5)
    end
    @test G.u_tau_from_law(:reichardt, 0.0, 1.0, 1e-3) == 0
    for T in (Float32, Float64), law in (:reichardt, :spalding, :werner_wengle)
        # Independent residual: reconstruct the requested speed over a broad
        # range of wall units, including the very-small-y+ FP32 regression.
        for (u,y,nu) in ((1e-8,.25,.01), (1e-5,.05,1e-4), (.08,1.,1e-6), (.02,2.,1e-9))
            speed,distance,viscosity=T(u),T(y),T(nu)
            ut=G.u_tau_from_law(law,speed,distance,viscosity)
            @test isfinite(ut) && ut>0
            @test isapprox(ut*G.uplus(law,distance*ut/viscosity),speed;rtol=T===Float32 ? 3e-6 : 1e-7)
        end
        @test_throws ArgumentError G.u_tau_from_law(law,T(.1),zero(T),T(.01))
        @test_throws ArgumentError G.u_tau_from_law(law,T(.1),T(1),zero(T))
        @test_throws ArgumentError G.u_tau_from_law(law,T(NaN),T(1),T(.01))
        @test_throws ArgumentError G.u_tau_from_law(law,T(.1),T(1),T(.01);guess=T(Inf))
        # A very small but representable speed must not be raised by an
        # arbitrary absolute u_tau floor (the former 1e-12 produced 1e6 error).
        tiny=T(1e-30);tiny_ut=G.u_tau_from_law(law,tiny,one(T),one(T))
        @test isapprox(tiny_ut*G.uplus(law,tiny_ut),tiny;rtol=T===Float32 ? 3e-6 : 1e-7)
        # Inputs whose intermediate wall units overflow must fail explicitly,
        # not return a misleading finite friction velocity.
        @test_throws ErrorException G.u_tau_from_law(law,one(T),floatmax(T),floatmin(T))
    end
    @test_throws ArgumentError G.u_tau_from_law(:unknown, .1, 1., .01)
    # The rounded Werner–Wengle piecewise constants produce a tiny jump;
    # an input in its gap has no exact root and must never silently succeed.
    jump=8.3*11.81^(1/7)-11.81
    @test_throws ErrorException G.u_tau_from_law(:werner_wengle,11.81*(11.81+jump/2),1.,1.)
    @test G.tau_wall(1.05, 0.02) ≈ 1.05 * 4e-4
    # the gradient in the viscous sublayer is u_tau^2 / nu, in the log layer u_tau / (kappa y)
    @test isapprox(G.wall_gradient(:reichardt, 0.001, 0.5, 1e-3), 0.001^2 / 1e-3; rtol=0.05)
    @test isapprox(G.wall_gradient(:reichardt, 0.05, 20.0, 1e-4), 0.05 / (0.41 * 20.0); rtol=0.1)
    @test_throws ArgumentError G.wall_gradient(:unknown, .01, 1., .01)
    @test_throws ArgumentError G.wall_gradient(:reichardt, .01, 0., .01)
    @test_throws ArgumentError G.wall_gradient(:reichardt, -.01, 1., .01)
    for T in (Float32,Float64), law in (:reichardt,:spalding,:werner_wengle)
        @test_throws ErrorException G.wall_gradient(law,one(T),floatmax(T),floatmin(T))
    end
end

@testset "unit conversions (the GPU solver's defect: metres read as cells)" begin
    # a wall cell of level 3 in a 3-level mesh: dx_1 = 0.04 m, dx_3 = 0.01 m; U = 30 m/s at u_lat 0.08
    dx3 = 0.01; vs = 30 / 0.08; dt3 = dx3 / vs
    nu_si = 1.5e-5
    nu_lat3 = nu_si * dt3 / dx3^2
    @test isapprox(G.lattice_to_si_viscosity(nu_lat3, dx3, dt3), nu_si; rtol=1e-12)
    @test G.metres_to_cells(0.025, dx3) == 2.5
    u_tau_lat = 0.004; u_tau_si = G.lattice_to_si_velocity(u_tau_lat, vs)
    @test u_tau_si == 1.5
    # y+ computed in SI equals y+ computed in the level's lattice units
    y_m = 0.015
    yp_si = G.yplus_si(y_m, u_tau_si, nu_si)
    yp_lat = G.metres_to_cells(y_m, dx3) * u_tau_lat / nu_lat3
    @test isapprox(yp_si, yp_lat; rtol=1e-12)
    @test isapprox(yp_si, 1500.0; rtol=1e-12)
    # and it differs from the defective reading (metres as cells) by the factor 1 / dx
    @test !isapprox(y_m * u_tau_lat / nu_lat3, yp_lat; rtol=0.5)
end
println("q19_octree_walls: all tests passed")
