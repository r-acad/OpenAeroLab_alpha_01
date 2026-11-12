using Test

const SOLVER = normpath(joinpath(@__DIR__, "..", "..", "solvers", "pbcav_s1030"))
include(joinpath(SOLVER, "src", "pbcav", "PBCav.jl"))
using .PBCav

@testset "free air RANS linear solver controls" begin
    base = Dict("PB_DOMAIN" => "freeair", "PB_RE" => "3e6")
    k, _ = PBCav.read_knobs(base)
    cfg = PBCav.freeair_config(k)
    @test (cfg.cpl_eta, cfg.cpl_sweeps, cfg.cpl_vcycles) == (0.1, 1, 1)
    @test (cfg.cpl_restart, cfg.cpl_maxit) == (30, 300)
    native=PBCav.PBConfig(mu=1e-3,dt=.02,nonorth=1.)
    @test (native.cpl_restart,native.cpl_maxit)==(30,300)
    @test occursin("cpl_restart=30 cpl_maxit=300", PBCav.freeair_config_line(cfg, PBCav.FreeStream(), k))

    tuned = merge(base, Dict("PB_CPL_ETA" => "0.35", "PB_CPL_SWEEPS" => "2",
                             "PB_CPL_VCYCLES" => "3", "PB_CPL_RESTART" => "60", "PB_CPL_MAXIT" => "900"))
    k, _ = PBCav.read_knobs(tuned)
    cfg = PBCav.freeair_config(k)
    @test (cfg.cpl_eta, cfg.cpl_sweeps, cfg.cpl_vcycles) == (0.35, 2, 3)
    @test (cfg.cpl_restart, cfg.cpl_maxit) == (60, 900)
    @test occursin("cpl_restart=60 cpl_maxit=900", PBCav.freeair_config_line(cfg, PBCav.FreeStream(), k))
    @test occursin("cpl_eta=0.35 cpl_sweeps=2 cpl_vcycles=3",
                   PBCav.freeair_config_line(cfg, PBCav.FreeStream(), k))

    for bad in (Dict("PB_CPL_ETA" => "1"), Dict("PB_CPL_SWEEPS" => "0"),
                Dict("PB_CPL_VCYCLES" => "9"), Dict("PB_CPL_RESTART" => "0"), Dict("PB_CPL_RESTART" => "121"),
                Dict("PB_CPL_MAXIT" => "0"), Dict("PB_CPL_MAXIT" => "10001"), Dict("PB_CPL_RESTART" => "30.5"),
                Dict("PB_CPL_MAXIT" => "300.5"))
        @test_throws ArgumentError PBCav.read_knobs(merge(base, bad))
    end
    @test_throws ArgumentError PBCav.read_knobs(Dict("PB_CPL_ETA" => "0.35"))
    for (name,lo,hi) in (("PB_CPL_RESTART",1,120),("PB_CPL_MAXIT",1,10000))
        for value in (lo,hi)
            edge,_=PBCav.read_knobs(merge(base,Dict(name=>string(value))))
            field=name=="PB_CPL_RESTART" ? :cpl_restart : :cpl_maxit
            @test getproperty(PBCav.freeair_config(edge),field)==value
        end
        @test_throws ArgumentError PBCav.read_knobs(Dict("PB_DOMAIN"=>"tunnel",name=>string(lo)))
        @test haskey(PBCav.freeair_restart_knobs(k),name)
    end

end
