using Test

const SOLVER = normpath(joinpath(@__DIR__, "..", "..", "solvers", "pbcav_s1030"))
include(joinpath(SOLVER, "src", "pbcav", "PBCav.jl"))
using .PBCav

@testset "free air RANS linear solver controls" begin
    base = Dict("PB_DOMAIN" => "freeair", "PB_RE" => "3e6")
    k, _ = PBCav.read_knobs(base)
    cfg = PBCav.freeair_config(k)
    @test (cfg.cpl_eta, cfg.cpl_sweeps, cfg.cpl_vcycles) == (0.1, 1, 1)

    tuned = merge(base, Dict("PB_CPL_ETA" => "0.35", "PB_CPL_SWEEPS" => "2",
                             "PB_CPL_VCYCLES" => "3"))
    k, _ = PBCav.read_knobs(tuned)
    cfg = PBCav.freeair_config(k)
    @test (cfg.cpl_eta, cfg.cpl_sweeps, cfg.cpl_vcycles) == (0.35, 2, 3)
    @test occursin("cpl_eta=0.35 cpl_sweeps=2 cpl_vcycles=3",
                   PBCav.freeair_config_line(cfg, PBCav.FreeStream(), k))

    for bad in (Dict("PB_CPL_ETA" => "1"), Dict("PB_CPL_SWEEPS" => "0"),
                Dict("PB_CPL_VCYCLES" => "9"))
        @test_throws ArgumentError PBCav.read_knobs(merge(base, bad))
    end
    @test_throws ArgumentError PBCav.read_knobs(Dict("PB_CPL_ETA" => "0.35"))

end
