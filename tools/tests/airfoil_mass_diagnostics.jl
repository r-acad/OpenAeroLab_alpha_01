using Test

include(joinpath(@__DIR__, "..", "..", "apps", "airfoil_web", "src", "diagnostics", "mass_diagnostics.jl"))
using .CFDMassDiagnostics

@testset "outer boundary numerical flux accounting" begin
    fi = zeros(3, 2)
    fj = zeros(2, 3)
    fi[1, :] .= (3, -1)
    fi[3, :] .= (2, -2)
    fj[:, 3] .= (4, -3)
    dc = fi[2:3, :] .- fi[1:2, :] .+ fj[:, 2:3] .- fj[:, 1:2]
    volume = ones(2, 2)
    result = assess_mass_balance(dc, volume, fi, fj, volume, volume;
                                 niw=0, nif=2, nbase=0,
                                 rho_inf=1, speed_inf=2, chord=1)
    @test result["outer_boundary_numerical_flux_incoming"] == 8
    @test result["outer_boundary_numerical_flux_outgoing"] == 7
    @test result["outer_boundary_net_over_incoming_flux"] == -1/8
    @test result["outer_boundary_net_over_absolute_flux"] == -1/15
    @test result["boundary_total_over_freestream_flux"] == -1/2
    @test result["telescoping_error_over_freestream_flux"] == 0

    fi .= 0; fj .= 0; dc .= 0
    stationary = assess_mass_balance(dc, volume, fi, fj, volume, volume;
                                     niw=0, nif=2, nbase=0,
                                     rho_inf=1, speed_inf=2, chord=1)
    @test stationary["outer_boundary_net_over_incoming_flux"] === nothing
    @test stationary["outer_boundary_net_over_absolute_flux"] === nothing

    # Inner wake/body faces are tracked in the full-domain net, but must not
    # be mistaken for farfield throughput or alter the outer-only ratio.
    fi = zeros(5, 2); fj = zeros(4, 3); volume = ones(4, 2)
    fi[1, 1] = 5; fi[5, 1] = 3
    fj[:, 1] .= (2, 3, 4, 5)
    dc = fi[2:5, :] .- fi[1:4, :] .+ fj[:, 2:3] .- fj[:, 1:2]
    wake = assess_mass_balance(dc, volume, fi, fj, volume, volume;
                               niw=1, nif=2, nbase=0,
                               rho_inf=1, speed_inf=2, chord=1)
    @test wake["outer_boundary_net_over_incoming_flux"] == -2/5
    @test wake["boundary_total_over_freestream_flux"] == -8
    @test wake["telescoping_error_over_freestream_flux"] == 0
end
