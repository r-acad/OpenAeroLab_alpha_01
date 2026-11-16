using Test
const REPOSITORY=dirname(dirname(@__DIR__))
const PRIVATE=get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(dirname(REPOSITORY),"02_PRIVATE_PROJECT_DEVELOPMENT"))
const SCRATCH=joinpath(PRIVATE,"TEMP_DELETEABLE","experimental_admission_tests")
mkpath(SCRATCH);ENV["LUDWIG_PRIVATE_ROOT"]=mktempdir(SCRATCH)
ENV["AIRFOIL_WEB_WARM_PREVIEW"]="0"
include(joinpath(REPOSITORY,"apps","airfoil_web","src","server.jl"))
const A=AirfoilWeb;const B=A.CFDBackend
request(extra=Dict{String,Any}();id="2412",backends=["transonic"])=Dict{String,Any}(
    "airfoil"=>Dict("kind"=>"naca4","id"=>id),"backends"=>backends,
    "flow"=>merge(Dict{String,Any}("mach"=>.3,"reynolds"=>3e6,"alpha_deg"=>4.),extra))
message(f)=try f();"" catch err;sprint(showerror,err) end

@testset "Every local asset referenced by the actual page is served" begin
    html=read(joinpath(REPOSITORY,"apps","airfoil_web","static","index.html"),String)
    paths=unique([String(m.captures[1]) for m in eachmatch(r"(?:src|href)=\"(/static/[^\"]+)\"",html)])
    @test "/static/scripts/workflow.js" in paths
    @test !("/static/scripts/research_settings.js" in paths)
    for path in paths
        response=A.app_handler(A.HTTP.Request("GET",path))
        @test response.status==200
        @test !isempty(response.body)
    end
    # Files and endpoints that were removed are not served under any name.
    for gone in ("/static/scripts/research_settings.js","/static/app.js","/static/app.css","/api/campaign","/api/campaign/files/stall_reference.png")
        @test A.app_handler(A.HTTP.Request("GET",gone)).status==404
    end
end

@testset "Iteration methods: the three the page offers, nothing else" begin
    @test B.RANSAcceleration.METHODS==("standard","irsj","newton")
    health=A.AirfoilWire.decode(copy(A.app_handler(A.HTTP.Request("GET","/api/health")).body),:msgpack)
    @test health["rans_methods"]==["standard","irsj","newton"]
    for method in ("standard","irsj","newton")
        @test A._validated_case(request(Dict("rans_method"=>method)))["flow"]["rans_method"]==method
    end
    for method in ("line_implicit","jfnk_native","jfnk_krylov","unknown")
        text=message(()->A._validated_case(request(Dict("rans_method"=>method))))
        @test occursin("rans_method must be one of standard, irsj, newton",text)&&occursin(method,text)
    end
    cfg=Dict{String,Any}("steps"=>5000,"time_mode"=>"steady")
    B.RANSAcceleration.configure_compressible!(cfg,Dict("rans_method"=>"irsj"))
    @test cfg["irsj"]===true&&cfg["rans_acceleration"]===true&&!any(startswith("implicit_"),keys(cfg))
    @test_throws ArgumentError B.RANSAcceleration.configure_compressible!(Dict{String,Any}("time_mode"=>"physical"),Dict("rans_method"=>"irsj"))
end

@testset "Withdrawn options: the value the app now always uses loads, any other value is refused" begin
    plain=A._validated_case(request())["flow"]
    @test isempty(intersect(keys(plain),keys(A.REMOVED_FLOW_KEYS)))        # the server adds none of them
    for (key,value) in (("cfd_te_topology","auto"),("cfd_mesh","aplus_nested_refined"),("lowspeed_topology","auto"),
                        ("cfd_irsj",false),("cfd_sfd_enabled",false),("lbm_wall_model",false))
        accepted=A._validated_case(request(Dict{String,Any}(key=>value)))
        @test !haskey(accepted["flow"],key)
        @test accepted["flow"]==plain
    end
    @test !haskey(A._validated_case(request(Dict("rans_method"=>"newton","cfd_mesh"=>"marched")))["flow"],"cfd_mesh")
    for (key,value) in (("cfd_te_topology","aplus"),("cfd_mesh","aplus_uniform_quality"),("cfd_mesh","aplus_uniform_fixed"),
                        ("cfd_mesh","cavitation_aplus_farfield"),("cfd_mesh","legacy_sharp_cmesh"),("cfd_mesh","marched"),
                        ("cfd_n_base",4),("cfd_irsj",true),("cfd_sfd_enabled",true),("lbm_wall_model",true),
                        ("lowspeed_topology","sharp"),("lowspeed_topology","ch_native"),
                        ("cfd_implicit_cfl",10.),("cfd_implicit_globalization","pseudo_transient"))
        text=message(()->A._validated_case(request(Dict{String,Any}(key=>value))))
        @test occursin("$key = $value is no longer an option of Airfoil Lab",text)
    end
    # A study round trip keeps a current request unchanged.
    saved=A.JSON3.read(A.JSON3.write(A._clean(A._validated_case(request(Dict("rans_method"=>"irsj"))))),Dict{String,Any})
    @test A._validated_case(saved)["flow"]["rans_method"]=="irsj"
    @test message(()->A._validated_case(request(Dict("lowspeed_resolution"=>"demo");backends=["lowspeed"])))==
        "ArgumentError: lowspeed_resolution must be standard or fine (or smoke for tests)"
end

@testset "Trailing-edge topology follows the section" begin
    sharp=joinpath(REPOSITORY,"apps","airfoil_web","data","references","rae2822_agard.dat")
    naca=request()
    @test B._auto_ch_selected(naca,sharp,naca["flow"])                                   # NACA four-digit equations: finite edge, C+H
    closed=Dict("airfoil"=>Dict("kind"=>"rae2822","id"=>"rae2822"))
    @test !B._auto_ch_selected(closed,sharp,Dict{String,Any}())                          # closed edge: one C block
    @test B._auto_ch_selected(closed,sharp,Dict{String,Any}("cfd_te_thickness"=>.004))   # a set thickness: C+H
    # The single C block refuses a finite edge instead of closing it or building a base fan.
    af=B.CMeshGen.naca4_points(2412)
    text=message(()->B._build_transonic_mesh(af;n_foil=64,n_wake=16,n_normal=32,dn1=1e-5,alpha_deg=0.,trip_upper=0.,trip_lower=0.))
    @test occursin("needs a closed trailing edge",text)&&occursin("C+H",text)
    # Species on a finite edge: the free-fluid domain runs on the C+H pair in physical time (species_ch_backend.jl).
    species(extra)=merge(Dict{String,Any}("mach"=>.2,"reynolds"=>3e6,"alpha_deg"=>2.),extra)
    water=species(Dict("cav_fluid"=>"water","cav_domain"=>"free_fluid","sigma"=>1.2))
    air=species(Dict("cav_fluid"=>"air","cav_domain"=>"free_fluid","cav_air_mode"=>"physical"))
    @test B.species_finite_edge_refusal(water)===nothing&&B.species_finite_edge_refusal(air)===nothing
    for flow in (water,air)
        cfg=B._species_ch_config(naca,sharp,flow,2.,SCRATCH)
        @test cfg["mesh_mode"]==B.SPECIES_CH_MESH=="conformal_C_H_v1"&&cfg["naca4_code"]==2412&&cfg["cav_domain"]=="free_fluid"
        @test cfg["cav_air_mode"]=="physical"&&cfg["cav_stage_clock_margin"]==.99&&cfg["cfd_farfield_angle_deg"]==0.
    end
    # What the pair does not cover is refused by name, at Create mesh or at Run, before any worker: air in the tunnel
    # (one C block), the steady air iteration (one C block; a Run setting, so Create mesh still shows the mesh).
    tunnel_air=species(Dict("cav_fluid"=>"air","cav_domain"=>"tunnel","cav_air_mode"=>"physical"))
    steady_air=species(Dict("cav_fluid"=>"air","cav_domain"=>"free_fluid"))
    run(flow)=message(()->B._transonic_point(naca,sharp,flow,2.,mktempdir(SCRATCH),_->nothing,_->nothing;species_backend=true))
    @test message(()->B.preview_species_mesh(sharp,tunnel_air,mktempdir(SCRATCH);source=naca["airfoil"]))=="ArgumentError: "*B.SPECIES_CH_TUNNEL_AIR
    @test run(tunnel_air)=="ArgumentError: "*B.SPECIES_CH_TUNNEL_AIR
    @test run(steady_air)=="ArgumentError: "*B.SPECIES_CH_STEADY_AIR
    @test B.species_finite_edge_refusal(steady_air;run=false)===nothing
    @test occursin("the C+H pair runs in physical time",run(merge(water,Dict("rans_acceleration"=>true))))
    @test occursin("cfd_n_foil sizes the single C block",run(merge(water,Dict("cfd_n_foil"=>128))))
    # The page states the same two refusals in the same words.
    page=read(joinpath(REPOSITORY,"apps","airfoil_web","static","scripts","app.js"),String)
    @test occursin(B.SPECIES_CH_TUNNEL_AIR,page)&&occursin(B.SPECIES_CH_STEADY_AIR,page)
end

@testset "C+H compressible options: time mode, trips, viscosity law" begin
    naca=request();path=joinpath(REPOSITORY,"apps","airfoil_web","data","references","rae2822_agard.dat")   # the NACA code decides
    config(extra=Dict{String,Any}())=B._ch_transonic_config(naca,path,merge(naca["flow"],Dict{String,Any}("cfd_gpu"=>false),extra),4.,mktempdir(SCRATCH))
    # A request without the options is the driver's previous configuration: steady, constant kinematic, fully turbulent.
    plain=config()
    @test (plain["time_mode"],plain["viscosity_model"],plain["transition_model"],plain["trip_upper"],plain["trip_lower"])==("steady","constant_kinematic","fully_turbulent",0.,0.)
    @test plain["history_stride"]==25&&!haskey(plain,"duration_ctu")&&!haskey(plain,"freestream_temperature_k")
    @test !B._ch_worker_settings(plain).physical
    # Trips are applied per side unless the request states the fully turbulent model, which records them as requested.
    tripped=config(Dict{String,Any}("trip_upper"=>.07,"trip_lower"=>.2))
    @test (tripped["trip_upper"],tripped["trip_lower"],tripped["transition_model"])==(.07,.2,"prescribed_trips")
    turbulent=config(Dict{String,Any}("trip_upper"=>.07,"trip_lower"=>.2,"cfd_transition_model"=>"fully_turbulent"))
    @test (turbulent["trip_upper"],turbulent["trip_lower"],turbulent["transition_model"])==(0.,0.,"fully_turbulent")
    @test (turbulent["requested_trip_upper"],turbulent["requested_trip_lower"])==(.07,.2)
    one=B._ch_worker_settings(config(Dict{String,Any}("trip_upper"=>.1)))
    @test (one.trip_upper,one.trip_lower)==(.1,0.)
    # Viscosity laws
    law=config(Dict{String,Any}("cfd_viscosity_model"=>"sutherland","cfd_freestream_temperature_k"=>271.))
    @test law["viscosity_model"]=="sutherland" && law["freestream_temperature_k"]==271.0 && B._ch_worker_settings(law).temperature_k==271.0
    @test config(Dict{String,Any}("cfd_viscosity_model"=>"constant_dynamic"))["viscosity_model"]=="constant_dynamic"
    @test occursin("must be constant_kinematic, sutherland, or constant_dynamic",message(()->config(Dict{String,Any}("cfd_viscosity_model"=>"power_law"))))
    # IRS-J forms its coefficient with a constant kinematic viscosity: refused with a law, by the app and by the worker.
    @test occursin("constant kinematic viscosity only",message(()->config(Dict{String,Any}("cfd_viscosity_model"=>"sutherland","rans_method"=>"irsj"))))
    @test occursin("IRS-J",message(()->B._ch_worker_settings(merge(law,Dict{String,Any}("irsj"=>true)))))
    # Physical time: its duration settings travel, smart stop is off, and the steady accelerations are refused.
    timed=config(Dict{String,Any}("cfd_time_mode"=>"physical","cfd_duration_ctu"=>40.,"cfd_burnin_ctu"=>10.,"cfd_block_ctu"=>2.))
    @test (timed["time_mode"],timed["duration_ctu"],timed["burnin_ctu"],timed["block_ctu"],timed["force_sample_dt_ctu"],timed["auto_stop"],timed["history_stride"])==("physical",40.,10.,2.,.1,false,400)
    settings=B._ch_worker_settings(timed)
    @test settings.physical && settings.duration_ctu==40.0 && settings.burnin_ctu==10.0 && !settings.irsj
    @test occursin("Burn-in",message(()->config(Dict{String,Any}("cfd_time_mode"=>"physical","cfd_duration_ctu"=>5.,"cfd_burnin_ctu"=>5.))))
    @test occursin("steady",message(()->config(Dict{String,Any}("cfd_time_mode"=>"physical","rans_method"=>"irsj"))))
    @test occursin("steady pseudo-time only",message(()->config(Dict{String,Any}("cfd_time_mode"=>"physical","rans_acceleration"=>true))))
    @test occursin("steady pseudo-time only",message(()->B._ch_worker_settings(merge(timed,Dict{String,Any}("irsj"=>true)))))
    @test occursin("trip locations",message(()->B._ch_worker_settings(merge(plain,Dict{String,Any}("trip_upper"=>1.5)))))
    @test occursin("time_mode",message(()->B._ch_worker_settings(merge(plain,Dict{String,Any}("time_mode"=>"dual")))))
end

@testset "Validation presets run on the current topologies" begin
    items=A.AirfoilReferenceData.benchmark_catalog()
    @test [item["id"] for item in items]==["rae2822-case9","rae2822-case6",
        "oat15a-2.5","oat15a-3.0","oat15a-3.1","oat15a-3.25","oat15a-3.5","oat15a-3.9","naca0012-stall"]
    for item in items
        case=A._validated_case(item["case"])
        @test isempty(intersect(keys(item["case"]["flow"]),keys(A.REMOVED_FLOW_KEYS)))
        @test case["backends"]==item["case"]["backends"]
        @test !occursin(r"(?i)a-plus|aplus",item["numerics_note"])
    end
    # ONERA OAT15A has a finite trailing edge: its six cases run on C+H with the settings they carry (trips at 0.07 c,
    # the Sutherland law at 271 K, physical time from 3.25 degrees).
    oat=joinpath(REPOSITORY,"apps","airfoil_web","data","references","onera_oat15a.dat")
    for item in filter(item->item["kind"]=="oat15a",items)
        case=A._validated_case(item["case"]);flow=case["flow"];alpha=flow["alpha_deg"]
        @test B._auto_ch_selected(case,oat,flow)
        @test item["unsteady"]==(alpha>=3.25)
        @test !isempty(item["reference"]["points"])&&item["reference"]["conditions"]["trip_upper"]==.07
        cfg=B._ch_transonic_config(case,oat,flow,alpha,mktempdir(SCRATCH))
        @test (cfg["trip_upper"],cfg["trip_lower"],cfg["transition_model"])==(.07,.07,"prescribed_trips")
        @test cfg["viscosity_model"]=="sutherland"&&cfg["freestream_temperature_k"]==271.
        @test cfg["time_mode"]==(item["unsteady"] ? "physical" : "steady")&&cfg["mach"]==.73&&cfg["reynolds"]==3e6&&cfg["resolution"]=="fine"
        item["unsteady"] && @test (cfg["duration_ctu"],cfg["burnin_ctu"],cfg["auto_stop"],cfg["irsj"])==(100.,20.,false,false)
        @test B._ch_worker_settings(cfg).physical==item["unsteady"]
    end
    for preset in A.CavitationReferences.catalog()
        @test isempty(intersect(keys(preset["flow"]),keys(A.REMOVED_FLOW_KEYS)))
        @test A._validated_case(Dict("airfoil"=>Dict("kind"=>"cavitation_reference","id"=>preset["id"])))["backends"]==["cavitation"]
    end
end
