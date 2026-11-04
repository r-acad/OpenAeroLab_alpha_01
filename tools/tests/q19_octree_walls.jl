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
    # Werner-Wengle continuity at y+ = 11.81
    @test isapprox(G.uplus_werner_wengle(11.81 - 1e-9), G.uplus_werner_wengle(11.81 + 1e-9); rtol=1e-3)
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
    @test G.tau_wall(1.05, 0.02) ≈ 1.05 * 4e-4
    # the gradient in the viscous sublayer is u_tau^2 / nu, in the log layer u_tau / (kappa y)
    @test isapprox(G.wall_gradient(:reichardt, 0.001, 0.5, 1e-3), 0.001^2 / 1e-3; rtol=0.05)
    @test isapprox(G.wall_gradient(:reichardt, 0.05, 20.0, 1e-4), 0.05 / (0.41 * 20.0); rtol=0.1)
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
