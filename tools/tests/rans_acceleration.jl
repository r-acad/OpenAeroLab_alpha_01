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

    automatic=CFDBackend.lowspeed_configuration(Dict{String,Any}("reynolds"=>3e6))
    @test automatic["topology"]=="ch"
    @test automatic["start"]=="ramp"
    @test automatic["mut_base"]=="committed"
    base = Dict{String,Any}("reynolds"=>3e6,"lowspeed_topology"=>"sharp")
    faster = merge(copy(base), Dict{String,Any}("rans_acceleration"=>true))
    sharp = CFDBackend.lowspeed_configuration(faster)
    sharp_input = CFDBackend._lowspeed_input("snapshot", "commit", "foil.dat", sharp, 0.0, @__DIR__)
    @test sharp_input["dt"] == 0.5
    @test sharp_input["knobs"]["PB_CPU_LAUNCH"] == "balanced"
    @test sharp_input["knobs"]["PB_TURB_STEP_RELAX"]=="0.5"
    @test sharp_input["steady_tol"] == 1e-6
    @test sharp_input["turb_steady_tol"] == 1e-7
    ch = CFDBackend.lowspeed_ch_configuration(faster)
    ch_knobs = CFDBackend._lowspeed_ch_knobs(ch, 0.002)
    @test ch_knobs["PB_DT"] == "0.5"
    @test ch_knobs["PB_CPU_LAUNCH"] == "balanced"
    @test ch_knobs["PB_TURB_STEP_RELAX"]=="0.5"
    @test ch_knobs["PB_CPL_AU"]=="0.95"
    overridden=CFDBackend.lowspeed_configuration(merge(copy(base),Dict{String,Any}("lowspeed_turb_step_relax"=>1.)))
    @test CFDBackend._lowspeed_input("snapshot","commit","foil.dat",overridden,0.,@__DIR__)["knobs"]["PB_TURB_STEP_RELAX"]=="1.0"
    tuned=merge(copy(base),Dict{String,Any}("lowspeed_cpl_au"=>0.95))
    @test CFDBackend._lowspeed_input("snapshot","commit","foil.dat",CFDBackend.lowspeed_configuration(tuned),0.,@__DIR__)["knobs"]["PB_CPL_AU"]=="0.95"
    @test CFDBackend._lowspeed_ch_knobs(CFDBackend.lowspeed_ch_configuration(tuned),0.002)["PB_CPL_AU"]=="0.95"
    @test_throws ArgumentError CFDBackend.lowspeed_configuration(merge(copy(base),Dict{String,Any}("lowspeed_cpl_au"=>0.)))
    @test parse(Float64,ch_knobs["PB_STEADY_TOL"]) == 1e-6
    @test parse(Float64,ch_knobs["PB_TURB_STEADY_TOL"]) == 1e-7
    relaxed=merge(copy(base),Dict{String,Any}("lowspeed_steady_tol"=>1e-4,"lowspeed_turb_steady_tol"=>1e-4))
    relaxed_sharp=CFDBackend.lowspeed_configuration(relaxed)
    @test CFDBackend._lowspeed_input("snapshot", "commit", "foil.dat", relaxed_sharp, 0.0, @__DIR__)["steady_tol"] == 1e-4
    relaxed_ch=CFDBackend.lowspeed_ch_configuration(merge(copy(relaxed),Dict{String,Any}("lowspeed_topology"=>"ch_native")))
    @test parse(Float64,CFDBackend._lowspeed_ch_knobs(relaxed_ch,0.002)["PB_TURB_STEADY_TOL"]) == 1e-4
    @test_throws ArgumentError CFDBackend.lowspeed_configuration(merge(copy(base),Dict{String,Any}("lowspeed_steady_tol"=>1e-2)))
    gpu = merge(copy(faster), Dict{String,Any}("lowspeed_gpu"=>true))
    @test CFDBackend._lowspeed_input("snapshot", "commit", "foil.dat", CFDBackend.lowspeed_configuration(gpu), 0.0, @__DIR__)["knobs"]["PB_CPU_LAUNCH"] == "fixed"
    @test CFDBackend._lowspeed_ch_knobs(CFDBackend.lowspeed_ch_configuration(gpu), 0.002)["PB_CPU_LAUNCH"] == "fixed"
    baseline = merge(copy(base), Dict{String,Any}("lowspeed_topology"=>"ch_native", "lowspeed_start"=>"ramp", "lowspeed_turb_mut_base"=>"committed"))
    control = CFDBackend.lowspeed_configuration(baseline)
    @test control["ch_source"] == "native"
    @test CFDBackend._lowspeed_ch_knobs(control, 0.002)["PB_START"] == "ramp"
    @test CFDBackend._lowspeed_ch_knobs(control, 0.002)["PB_TURB_MUT_BASE"] == "committed"
    refined = CFDBackend.lowspeed_configuration(merge(copy(baseline), Dict{String,Any}("lowspeed_resolution"=>"fine")))
    @test CFDBackend._lowspeed_ch_knobs(refined, 0.002)["PB_MESH_LEVEL"] == "2"
    @test CFDBackend._lowspeed_ch_knobs(control, 0.002)["PB_MESH_LEVEL"] == "1"
    message=CFDBackend._lowspeed_convergence_message(Dict("outcome"=>"INCOMPLETE", "steps"=>200),
        Dict("status"=>"loads_stable_flow_unsettled", "reason"=>"equation residual limits have not passed", "converged"=>false))
    @test occursin("Recent load stability does not establish the final CL/CD",message)

    case = Dict{String,Any}("airfoil"=>Dict{String,Any}("kind"=>"naca4", "id"=>"0012"))
    gas = Dict{String,Any}("rans_acceleration"=>true, "cfd_mesh"=>"aplus_nested_refined")
    cfg = CFDBackend._ch_transonic_config(case, "foil.dat", gas, 0.0, @__DIR__)
    @test cfg["irsj"] == true
    @test cfg["rans_acceleration"] == true
end

@testset "Low-speed Krylov defaults, request overrides and capabilities" begin
    function request_knobs(flow)
        settings=CFDBackend.lowspeed_configuration(flow)
        if get(settings,"topology",nothing)=="ch"
            return settings,CFDBackend._lowspeed_ch_knobs(settings,0.002)
        end
        settings,CFDBackend._lowspeed_input("snapshot","commit","foil.dat",settings,12.,@__DIR__)["knobs"]
    end
    routes=(
        (Dict{String,Any}("lowspeed_topology"=>"sharp"),30,"0.8"),
        (Dict{String,Any}("lowspeed_topology"=>"ch_native"),60,"0.95"),
        (Dict{String,Any}("gurney_height"=>0.01,"gurney_mode"=>"lower"),60,"0.95"),
        (Dict{String,Any}(),60,"0.95"))
    for (route,restart,au) in routes
        flow=merge(Dict{String,Any}("reynolds"=>3e6),route)
        settings,knobs=request_knobs(flow)
        @test knobs["PB_CPL_RESTART"]==string(restart)
        @test knobs["PB_CPL_MAXIT"]=="300"
        @test knobs["PB_CPL_AU"]==au
        @test knobs["PB_TURB_STEP_RELAX"]=="0.5"
        @test settings["steady_tol"]==1e-6
        @test settings["turb_steady_tol"]==1e-7
        _,overridden=request_knobs(merge(flow,Dict("lowspeed_cpl_restart"=>45,"lowspeed_cpl_maxit"=>900)))
        @test overridden["PB_CPL_RESTART"]=="45"
        @test overridden["PB_CPL_MAXIT"]=="900"
        for (key,value) in (("lowspeed_cpl_restart",0),("lowspeed_cpl_restart",121),
                            ("lowspeed_cpl_restart",30.5),("lowspeed_cpl_restart",true),
                            ("lowspeed_cpl_maxit",0),("lowspeed_cpl_maxit",10001),
                            ("lowspeed_cpl_maxit",1.5),("lowspeed_cpl_maxit",false),
                            ("lowspeed_cpl_maxit",NaN),("lowspeed_cpl_restart",Inf),
                            ("lowspeed_cpl_restart",nothing),("lowspeed_cpl_restart",[60]),
                            ("lowspeed_cpl_restart",Dict("value"=>60)),
                            ("lowspeed_cpl_maxit",nothing),("lowspeed_cpl_maxit",[300]),
                            ("lowspeed_cpl_maxit",Dict("value"=>300)))
            @test_throws ArgumentError CFDBackend.lowspeed_configuration(merge(flow,Dict(key=>value)))
        end
    end
    for (restart,maxit) in ((1,1),(120,10000))
        controls=CFDBackend.lowspeed_iteration_controls(Dict("lowspeed_cpl_restart"=>restart,"lowspeed_cpl_maxit"=>maxit))
        @test controls["PB_CPL_RESTART"]==string(restart)
        @test controls["PB_CPL_MAXIT"]==string(maxit)
    end
    capability=CFDBackend.lowspeed_capabilities()["krylov"]
    @test capability["request_keys"]==["lowspeed_cpl_restart","lowspeed_cpl_maxit"]
    @test capability["restart_bounds"]==[1,120]
    @test capability["maxit_bounds"]==[1,10000]
    @test capability["true_residual_required"]===true
    for (route,restart) in (("native",30),("sharp",30),("ch_native",60))
        @test capability["defaults"][route]==Dict("restart"=>restart,"maxit"=>300)
    end
end
