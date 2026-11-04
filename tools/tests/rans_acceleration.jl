using Test

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(ROOT, "apps", "airfoil_web", "src", "backends", "cfd_backend.jl"))
using .CFDBackend

@testset "steady RANS acceleration routing" begin
    for family in (:compressible, :compressible_ch, :lowspeed, :lowspeed_ch, :species_air)
        standard = CFDBackend.RANSAcceleration.policy(false, family)
        @test !standard.irsj && !standard.species_line && standard.pseudo_dt == 0.25
        @test_throws ArgumentError CFDBackend.RANSAcceleration.policy(true, family; physical=true)
    end
    @test CFDBackend.RANSAcceleration.policy(true, :compressible).irsj
    @test CFDBackend.RANSAcceleration.policy(true, :compressible_ch).irsj
    @test CFDBackend.RANSAcceleration.policy(true, :species_air).species_line

    base = Dict{String,Any}("reynolds"=>3e6)
    faster = merge(copy(base), Dict{String,Any}("rans_acceleration"=>true))
    sharp = CFDBackend.lowspeed_configuration(faster)
    @test CFDBackend._lowspeed_input("snapshot", "commit", "foil.dat", sharp, 0.0, @__DIR__)["dt"] == 0.5
    ch = CFDBackend.lowspeed_ch_configuration(faster)
    @test CFDBackend._lowspeed_ch_knobs(ch, 0.002)["PB_DT"] == "0.5"

    case = Dict{String,Any}("airfoil"=>Dict{String,Any}("kind"=>"naca4", "id"=>"0012"))
    gas = Dict{String,Any}("rans_acceleration"=>true, "cfd_mesh"=>"aplus_nested_refined")
    cfg = CFDBackend._ch_transonic_config(case, "foil.dat", gas, 0.0, @__DIR__)
    @test cfg["irsj"] == true
    @test cfg["rans_acceleration"] == true
end
