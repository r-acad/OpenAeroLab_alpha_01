using Test,UUIDs,TOML,Serialization
const REPOSITORY=dirname(dirname(@__DIR__))
const SCRATCH=joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(dirname(REPOSITORY),"02_PRIVATE_PROJECT_DEVELOPMENT")),"TEMP_DELETEABLE","lowspeed_initial_flow_tests")
mkpath(SCRATCH);ENV["LUDWIG_PRIVATE_ROOT"]=mktempdir(SCRATCH)
include(joinpath(REPOSITORY,"apps","airfoil_web","src","server.jl"))
const A=AirfoilWeb;const B=A.CFDBackend

request(;alpha=12.,seed=nothing,extra=Dict{String,Any}())=Dict{String,Any}(
    "airfoil"=>Dict("kind"=>"naca4","id"=>"4412"),"backends"=>["lowspeed"],
    "flow"=>merge(Dict{String,Any}("reynolds"=>3e6,"alpha_deg"=>alpha,"lowspeed_resolution"=>"standard"),
        seed===nothing ? Dict() : Dict("lowspeed_initial_job_id"=>seed),extra))
function fixture_job(;status="cancelled",angles=(10.,12.),results_parent=nothing)
    case=A._validated_case(request());id=string(uuid4())
    results_parent===nothing || A.AirfoilResultLocations.create_directory(A.RUN_DIR,id;parent=results_parent)
    root=A._safe_job_dir(id);mkpath(root)
    job=Dict{String,Any}("id"=>id,"status"=>status,"case"=>case)
    A.JOBS[id]=job
    foil=A.AirfoilJXBackend.resolve_airfoil(case["airfoil"]).airfoil
    path=joinpath(root,"airfoil.dat");A._save_geometry(foil,path)
    target=A._lowspeed_initial_target(case,path,root)
    for (index,alpha) in enumerate(angles)
        dir=joinpath(root,"lowspeed",A.@sprintf("%03d_alpha_%+.3f",index,alpha));mkpath(dir)
        input=deepcopy(target.input);input["alpha"]=alpha
        open(io->TOML.print(io,input),joinpath(dir,"input.toml"),"w")
        open(io->TOML.print(io,Dict("source_sha256"=>target.source)),joinpath(dir,"result.toml"),"w")
        saved=Dict{String,Any}("commit"=>target.head,"source_sha256"=>target.source,
            "topology"=>"conformal_C_H_v1","foil_sha256"=>target.foil_sha,"mesh_hashes"=>["c1","c2"],
            "strip_hashes"=>["h1","h2"],"baffle"=>"","reynolds"=>3e6,"tu"=>.01,"mut_ratio"=>10.,
            "alpha_deg"=>alpha,"gamma"=>.5,"dt"=>.02,"dt_prev"=>.01)
        for key in ("ux","uy","p","rho","q","rk","rw","mut","F","Fb")
            saved[key]=fill(key=="rho"||key=="rw" ? 1. : 0.,4,1,3)
        end
        open(io->serialize(io,saved),joinpath(dir,"state_checkpoint.jls"),"w")
    end
    (;id,root,case,target,path)
end

@testset "Required saved flow is refused by actual CPU workers before any cold projection or march" begin
    for topology in ("sharp","ch_native")
        dir=mktempdir(SCRATCH)
        flow=Dict{String,Any}("reynolds"=>3e6,"alpha_deg"=>12.,"lowspeed_topology"=>topology,
            "lowspeed_resolution"=>topology=="sharp" ? "smoke" : "demo","lowspeed_max_steps"=>1)
        case=A._validated_case(Dict("airfoil"=>Dict("kind"=>"naca4","id"=>"4412"),"backends"=>["lowspeed"],"flow"=>flow))
        source=A.AirfoilJXBackend.resolve_airfoil(case["airfoil"]).airfoil
        path=joinpath(dir,"airfoil.dat");A._save_geometry(source,path)
        snap,head=B._lowspeed_snapshot_check();ls=B.lowspeed_configuration(flow)
        if topology=="sharp"
            foil,_=B._lowspeed_geometry(case,path,dir,snap)
            cfg=B._lowspeed_input(snap,head,foil,ls,12.,dir)
            bad=joinpath(dir,"bad_checkpoint.jls")
            open(io->serialize(io,Dict("commit"=>"different solver revision")),bad,"w")
            cfg["warm_start"]=bad
        else
            foil,_,gap=B._lowspeed_ch_geometry(case,path,dir,flow)
            cfg=B._lowspeed_ch_input(snap,head,foil,ls,B._lowspeed_ch_knobs(ls,gap),12.,dir)
            # A required seed with no offered file must fail before the ordinary
            # worker's fresh ramp/projection fallback is entered.
        end
        cfg["require_warm_start"]=true
        input=joinpath(dir,"input.toml");open(io->TOML.print(io,cfg),input,"w")
        cmd=topology=="sharp" ? B._lowspeed_cmd(snap,input) : B._lowspeed_ch_cmd(snap,input)
        log=joinpath(dir,"worker.log")
        process=run(pipeline(ignorestatus(cmd);stdout=log,stderr=log))
        text=read(log,String)
        @test !success(process)
        @test occursin(r"(?i)required saved initial[- ]flow",text)
        @test !occursin("PB FREEAIR START:",text)
        @test !isfile(joinpath(dir,"loads_history.csv"))
        @test !isfile(joinpath(dir,"state_checkpoint.pbrestart"))
    end
end
post(data)=A.app_handler(A.HTTP.Request("POST","/api/jobs",["Content-Type"=>"application/msgpack","Accept"=>"application/msgpack"],A.AirfoilWire.encode(data,:msgpack)))
body(response)=A.AirfoilWire.decode(copy(response.body),:msgpack)

@testset "Initial-flow API accepts only confined terminal low-speed jobs" begin
    @test A._lowspeed_initial_id(Dict(),["lowspeed"])===nothing
    for bad in ("../outside","",42,string(uuid4())*"/file")
        @test_throws ArgumentError A._validated_case(request(seed=bad))
    end
    invalid=request(seed=string(uuid4()));invalid["backends"]=["jxfoil"]
    @test_throws ArgumentError A._validated_case(invalid)
    @test post(request(seed=string(uuid4()))).status==400
    f=fixture_job(status="running")
    @test post(request(seed=f.id)).status==400
    A.JOBS[f.id]["status"]="cancelled"
    @test A._lowspeed_initial_source(A._validated_case(request(seed=f.id))).id==f.id
    for extra in (Dict("reynolds"=>4e6),Dict("lowspeed_tu"=>.02),Dict("lowspeed_mut_ratio"=>20.),Dict("lowspeed_resolution"=>"fine"))
        @test post(request(seed=f.id,extra=extra)).status==400
    end
    wrong=request(seed=f.id);wrong["airfoil"]["id"]="0012"
    @test post(wrong).status==400
    far=request(alpha=20.,seed=f.id)
    @test post(far).status==400
    outside=joinpath(dirname(f.root),"outside.txt");write(outside,"unrelated")
    @test_throws ArgumentError A._initial_confined(outside,f.root)
    @test A._initial_confined(joinpath(f.root,"airfoil.dat"),f.root)==realpath(joinpath(f.root,"airfoil.dat"))
    registered=fixture_job(results_parent=mktempdir(SCRATCH))
    selected=A._lowspeed_initial_source(A._validated_case(request(seed=registered.id)))
    @test selected.id==registered.id
    @test selected.root==registered.root
    @test A._initial_confined(joinpath(first(selected.points),"state_checkpoint.jls"),selected.root)==realpath(joinpath(first(selected.points),"state_checkpoint.jls"))
    health=A.AirfoilWire.decode(copy(A.app_handler(A.HTTP.Request("GET","/api/health")).body),:msgpack)
    @test health["api_revision"]>=25
    @test health["lowspeed_initial_flow"]["checked_job_seed"]===true
    @test health["lowspeed_initial_flow"]["exact_continuation"]===false
end

@testset "Closest compatible state, mesh/source checks and initial-flow provenance" begin
    f=fixture_job();case=A._validated_case(request(seed=f.id,extra=Dict("lowspeed_cpl_au"=>.95)))
    out=mktempdir(SCRATCH)
    seed=A._resolve_lowspeed_initial(case,f.path,out)
    @test basename(seed.dir)=="002_alpha_+12.000"
    @test seed.provenance["kind"]=="checked_initial_flow"
    @test seed.provenance["job_id"]==f.id
    @test occursin("not exact native",seed.provenance["meaning"])
    @test seed.provenance["checkpoint_sha256"]==bytes2hex(A.sha256(read(joinpath(seed.dir,"state_checkpoint.jls"))))
    near=A._validated_case(request(alpha=11.,seed=f.id))
    @test basename(A._resolve_lowspeed_initial(near,f.path,mktempdir(SCRATCH)).dir)=="001_alpha_+10.000"
    checkpoint=joinpath(seed.dir,"state_checkpoint.jls");original=read(checkpoint)
    saved=open(deserialize,checkpoint);saved["source_sha256"]="changed"
    open(io->serialize(io,saved),checkpoint,"w")
    @test basename(A._resolve_lowspeed_initial(case,f.path,mktempdir(SCRATCH)).dir)=="001_alpha_+10.000"
    write(checkpoint,original)
    for mutate! in (s->(s["foil_sha256"]="different"),s->(s["baffle"]="gurney:side=lower,h=0.01"),
            s->(s["reynolds"]=4e6),s->(s["dt_prev"]=0.),s->(s["ux"][1]=NaN),s->(s["mut"][1]=-.1))
        saved=open(deserialize,checkpoint);mutate!(saved);open(io->serialize(io,saved),checkpoint,"w")
        @test_throws ArgumentError A._check_lowspeed_initial_point(seed.dir,f.target,12.)
        write(checkpoint,original)
    end
    flap=A._validated_case(request(seed=f.id,extra=Dict("gurney_height"=>.01,"gurney_mode"=>"lower")))
    @test_throws ArgumentError A._resolve_lowspeed_initial(flap,f.path,mktempdir(SCRATCH))
    A.JOBS[f.id]["case"]["backends"]=["jxfoil"]
    @test post(request(seed=f.id)).status==400
    A.JOBS[f.id]["case"]["backends"]=["lowspeed"]
    # Exercise the real POST path without launching a numerical worker.
    @eval A _start_worker()=nothing
    response=post(request(seed=f.id,extra=Dict("lowspeed_cpl_au"=>.95)))
    @test response.status==202
    queued=A._get_job(body(response)["id"])
    @test queued["case"]["flow"]["lowspeed_initial_job_id"]==f.id
end

@testset "Seed is mandatory for first point; later points use the preceding angle" begin
    @eval B begin
        const INITIAL_TEST_WARMS=Any[]
        const INITIAL_TEST_REQUIRED=Bool[]
        const INITIAL_TEST_REFUSE=Ref(false)
        function _lowspeed_point(case,path,flow,alpha,dir,progress,log,warmdir=nothing;warm_required::Bool=false)
            push!(INITIAL_TEST_WARMS,warmdir)
            push!(INITIAL_TEST_REQUIRED,warm_required)
            write(joinpath(dir,"state_checkpoint.jls"),"generated fake fixture")
            Dict{String,Any}("status"=>"completed","alpha_deg"=>alpha,
                "diagnostics"=>Dict("warm_start_source"=>INITIAL_TEST_REFUSE[] ? "" : joinpath(warmdir,"state_checkpoint.jls")))
        end
    end
    f=fixture_job();firstdir=joinpath(f.root,"lowspeed","002_alpha_+12.000")
    case=A._validated_case(request());case["airfoil"]["path"]=f.path;case["alphas"]=[12.,13.]
    case["sweep"]=Dict("enabled"=>true,"start"=>12.,"stop"=>13.,"step"=>1.)
    output=B.run_case(case,"lowspeed",mktempdir(SCRATCH);initial_warmdir=firstdir)
    @test length(output["points"])==2
    @test B.INITIAL_TEST_WARMS[1]==firstdir
    @test basename(B.INITIAL_TEST_WARMS[2])=="001_alpha_+12.000"
    @test B.INITIAL_TEST_REQUIRED==[true,false]
    empty!(B.INITIAL_TEST_WARMS);B.INITIAL_TEST_REFUSE[]=true
    err=try B.run_case(case,"lowspeed",mktempdir(SCRATCH);initial_warmdir=firstdir);nothing catch e;e end
    @test err isa ArgumentError
    @test occursin("cold fallback is not accepted",sprint(showerror,err))
    @test length(B.INITIAL_TEST_WARMS)==1
    @test_throws ArgumentError B.run_case(case,"transonic",mktempdir(SCRATCH);initial_warmdir=firstdir)
end
