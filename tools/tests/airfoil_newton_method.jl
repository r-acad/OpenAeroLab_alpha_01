# julia --startup-file=no --project=third_party/JXfoil tools/tests/airfoil_newton_method.jl
# The opt-in Implicit Newton (CPU) steady method of Airfoil Lab's compressible backend and its marched mesh profile:
# request validation, the worker configuration, Create mesh for both topologies (pure geometry) and the worker's
# progress, layout and status rules. No solver is loaded and no flow is marched here; the march itself is covered by
# tools/tests/cmesh_steady_newton.jl and cmesh_steady_newton_pair.jl.
using Test, TOML
const REPOSITORY=dirname(dirname(@__DIR__))
const PRIVATE=get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(dirname(REPOSITORY),"02_PRIVATE_PROJECT_DEVELOPMENT"))
const SCRATCH=joinpath(PRIVATE,"TEMP_DELETEABLE","airfoil_newton_method_tests")
mkpath(SCRATCH);ENV["LUDWIG_PRIVATE_ROOT"]=mktempdir(SCRATCH)
ENV["AIRFOIL_WEB_WARM_PREVIEW"]="0"
include(joinpath(REPOSITORY,"apps","airfoil_web","src","server.jl"))
const A=AirfoilWeb
const B=A.CFDBackend

request(flow;backends=["transonic"],airfoil=Dict("kind"=>"naca4","id"=>"0012"))=Dict{String,Any}(
    "airfoil"=>airfoil,"backends"=>backends,
    "flow"=>merge(Dict{String,Any}("mach"=>.5,"reynolds"=>3e6,"alpha_deg"=>2.),flow))
message(action)=try action();"" catch err;sprint(showerror,err) end

"A NACA 0012 with a closed trailing edge (the 0.1036 coefficient) as a coordinate file, upper surface first."
function closed_naca0012(path;n=121)
    yt(x)=5*.12*(.2969*sqrt(x)-.126x-.3516x^2+.2843x^3-.1036x^4)
    xs=[.5*(1-cospi(k/(n-1))) for k in 0:n-1]
    open(path,"w") do io
        println(io,"NACA 0012, closed trailing edge")
        for x in reverse(xs);println(io,x," ",x==1 ? 0. : yt(x));end
        for x in xs[2:end];println(io,x," ",x==1 ? 0. : -yt(x));end
    end
    path
end
const DIR=mktempdir(SCRATCH)
const SHARP=closed_naca0012(joinpath(DIR,"naca0012_closed.dat"))
const SHARP_CASE=Dict("airfoil"=>Dict("kind"=>"dat","id"=>"naca0012_closed.dat"))
const NACA_CASE=Dict("airfoil"=>Dict("kind"=>"naca4","id"=>"0012"))
const NEWTON_FLOW=Dict{String,Any}("rans_method"=>"newton")

@testset "Method name, label and routing" begin
    R=B.RANSAcceleration
    @test "newton" in R.METHODS
    cfg=R.configure_compressible!(Dict{String,Any}("steps"=>300,"time_mode"=>"steady"),Dict("rans_method"=>"newton"))
    @test cfg["rans_method"]=="newton"&&cfg["rans_acceleration_method"]==R.NEWTON_LABEL&&!cfg["irsj"]
    @test !haskey(cfg,"implicit_cfl")&&!haskey(cfg,"sfd_start")
    @test_throws ArgumentError R.configure_compressible!(Dict{String,Any}("steps"=>300,"time_mode"=>"physical"),Dict("rans_method"=>"newton"))
    @test_throws ArgumentError R.configure_compressible!(Dict{String,Any}("steps"=>300,"time_mode"=>"steady"),Dict("rans_method"=>"newton");coupled=true)
    # the explicit worker never runs under this method's name
    backend=read(joinpath(REPOSITORY,"apps","airfoil_web","src","backends","cfd_backend.jl"),String)
    @test occursin("\"--newton-worker\"",backend)&&occursin("CFDBackend._newton_worker(ARGS[2])",backend)
    worker=first(findfirst("include(joinpath(@__DIR__,\"..\",\"workers\",\"newton_worker.jl\"))",backend))
    @test first(findfirst("const _RANS_RESTART_SOURCE_LOAD_BEFORE",backend))<worker<first(findfirst("IS_WORKER && _rans_restart_bind_source_snapshot!",backend))
end

@testset "Request validation" begin
    accepted=A._validated_case(request(NEWTON_FLOW))
    @test accepted["flow"]["rans_method"]=="newton"
    # the request names the method only: the mesh is the method's own, and no mesh or topology key travels with it
    @test !haskey(accepted["flow"],"cfd_mesh")&&!haskey(accepted["flow"],"cfd_te_topology")
    # a study saved by the earlier page (the method with its mesh name) still loads; the name is dropped
    @test !haskey(A._validated_case(request(Dict("rans_method"=>"newton","cfd_mesh"=>"marched")))["flow"],"cfd_mesh")
    # the marched mesh cannot be asked for with another method
    for method in ("standard","irsj")
        @test occursin("cfd_mesh = marched is no longer an option",message(()->A._validated_case(request(Dict("rans_method"=>method,"cfd_mesh"=>"marched")))))
    end
    @test_throws ArgumentError A._validated_case(request(merge(NEWTON_FLOW,Dict("cfd_time_mode"=>"physical"))))
    @test_throws ArgumentError A._validated_case(request(NEWTON_FLOW;backends=["lowspeed"]))
    @test_throws ArgumentError A._validated_case(request(NEWTON_FLOW;backends=["transonic","cavitation"]))
    @test_throws ArgumentError A._validated_case(request(merge(NEWTON_FLOW,Dict("cfd_viscosity_model"=>"powerlaw"))))
    @test_throws ArgumentError A._validated_case(request(merge(NEWTON_FLOW,Dict("cfd_resolution"=>"coarse"))))
    @test_throws ArgumentError A._validated_case(request(merge(NEWTON_FLOW,Dict("gurney_height"=>0.01))))
    for model in ("sutherland","constant_dynamic","constant_kinematic")
        @test A._validated_case(request(merge(NEWTON_FLOW,Dict("cfd_viscosity_model"=>model))))["flow"]["cfd_viscosity_model"]==model
    end
    # a saved study reopens
    roundtrip=A.JSON3.read(A.JSON3.write(A._clean(accepted)),Dict{String,Any})
    @test A._validated_case(roundtrip)["flow"]["rans_method"]=="newton"
    # a request that does not name the method is not routed to it
    @test !haskey(A._validated_case(request(Dict{String,Any}()))["flow"],"rans_method")
    health=A.decode(Vector{UInt8}(A.app_handler(A.HTTP.Request("GET","/api/health")).body),:msgpack)
    @test "newton" in health["rans_methods"]&&health["api_revision"]>=29
end

@testset "Worker configuration" begin
    flow=A._validated_case(request(merge(NEWTON_FLOW,Dict("cfd_gpu"=>true,"cfd_steps"=>500000,
        "cfd_viscosity_model"=>"sutherland","cfd_freestream_temperature_k"=>271.))))["flow"]
    cfg=B._newton_config(SHARP_CASE,SHARP,flow,2.,DIR)
    # CPU and Float64 whatever the GPU switch says; the worker loader reads the gpu key
    @test cfg["gpu"]===false&&cfg["requested_gpu"]===true&&cfg["precision"]=="float64"
    @test cfg["rans_method"]=="newton"&&cfg["mesh_mode"]=="marched"&&cfg["time_mode"]=="steady"
    @test cfg["newton_topology"]=="c"&&cfg["source_te_gap"]<=1e-8&&cfg["cfd_te_gap"]<=1e-8
    @test (cfg["n_foil"],cfg["n_wake"],cfg["n_normal"])==(768,128,96)
    @test cfg["te_ds"]==1e-3&&cfg["wake_aspect"]==200.
    # step budget: min(cfd_steps, 600) on the fine level, 200 on each coarse level
    @test cfg["steps"]==600&&cfg["newton_nsteps"]==[200,200,600]&&cfg["requested_cfd_steps"]==500000
    @test cfg["newton_levels"]==[4,2,1]&&cfg["newton_cflmax"]==[1e3,1e3,1e3]
    @test cfg["newton_load_tol"]==[1e-3,3e-4,1e-4]&&cfg["newton_window"]==[10,15,20]
    @test cfg["newton_lintol"]==.3&&cfg["newton_maxit"]==60&&cfg["newton_growth"]==1.5
    @test cfg["newton_every"]==2&&cfg["newton_vortex_relax"]==.3
    @test cfg["limiter"]=="venkat_chord"&&cfg["limiter_smooth_coefficient"]==.2
    @test cfg["tu_inf"]==.001&&cfg["mu_t_ratio"]==.1
    @test cfg["trip_upper"]==.03&&cfg["trip_lower"]==.05&&!cfg["fully_turbulent"]     # the app's defaults
    @test cfg["viscosity_model"]=="sutherland"&&cfg["freestream_temperature_k"]==271.
    @test cfg["mach"]==.5&&cfg["reynolds"]==3e6&&cfg["alpha_deg"]==2.0&&cfg["cfd_farfield_angle_deg"]==2.
    @test !haskey(cfg,"naca4_code")&&!haskey(cfg,"warm_start_requested")&&!isempty(cfg["numerical_source_sha256"])
    path=joinpath(DIR,"input.toml");open(io->TOML.print(io,cfg),path,"w")
    parsed=TOML.parsefile(path)
    @test parsed["gpu"]===false&&parsed["newton_nsteps"]==[200,200,600]&&parsed["newton_load_tol"]==[1e-3,3e-4,1e-4]

    without_steps=delete!(copy(flow),"cfd_steps")
    @test B._newton_config(SHARP_CASE,SHARP,without_steps,2.,DIR)["newton_nsteps"]==[200,200,300]
    @test B._newton_config(SHARP_CASE,SHARP,merge(flow,Dict("cfd_steps"=>150)),2.,DIR)["newton_nsteps"]==[200,200,150]
    for (resolution,sizes) in (("standard",(768,128,96)),("fine",(1152,192,128)),("very_fine",(1536,256,160)))
        c=B._newton_config(SHARP_CASE,SHARP,merge(flow,Dict("cfd_resolution"=>resolution)),2.,DIR)
        @test (c["n_foil"],c["n_wake"],c["n_normal"])==sizes&&all(n->n%4==0,sizes)
    end
    @test_throws ArgumentError B._newton_config(SHARP_CASE,SHARP,merge(flow,Dict("cfd_n_foil"=>770)),2.,DIR)
    tripped=B._newton_config(SHARP_CASE,SHARP,merge(flow,Dict("trip_upper"=>.07,"trip_lower"=>.1)),2.,DIR)
    @test tripped["trip_upper"]==.07&&tripped["trip_lower"]==.1
    for model in ("constant_dynamic","constant_kinematic")
        @test B._newton_config(SHARP_CASE,SHARP,merge(flow,Dict("cfd_viscosity_model"=>model)),2.,DIR)["viscosity_model"]==model
    end
    # a sweep keeps the mesh of its first angle (the mesh-angle rule of the single-block compressible mesh)
    swept=B._newton_config(SHARP_CASE,SHARP,flow,4.,DIR;mesh_anchor=0.)
    @test swept["alpha_deg"]==4.0&&swept["cfd_farfield_angle_deg"]==0.0&&swept["polar_mesh_anchor_deg"]==0.

    # topology from the section after the app's conditioning
    naca=B._newton_config(NACA_CASE,SHARP,flow,2.,DIR)        # the equation NACA sections have a finite trailing edge
    @test naca["newton_topology"]=="c_plus_h"&&naca["naca4_code"]==12&&naca["cfd_te_gap"]>1e-3
    thick=B._newton_config(SHARP_CASE,SHARP,merge(flow,Dict("cfd_te_thickness"=>.005)),2.,DIR)
    @test thick["newton_topology"]=="c_plus_h"&&thick["cfd_te_thickness"]==.005&&isapprox(thick["cfd_te_gap"],.005;atol=1e-9)
    # every point starts cold; an offered checkpoint is recorded, not used
    warm=mktempdir(SCRATCH);write(joinpath(warm,"state_checkpoint.jls"),"not read")
    @test B._newton_config(SHARP_CASE,SHARP,flow,2.,DIR;warmdir=warm)["warm_start_requested"]==joinpath(warm,"state_checkpoint.jls")
    @test B._newton_threads() in (string(max(1,Sys.CPU_THREADS>=4 ? Sys.CPU_THREADS÷2 : Sys.CPU_THREADS)),get(ENV,"LUDWIG_NEWTON_THREADS","auto"))   # one per physical core unless set
end

@testset "Create mesh shows the marched mesh" begin
    flow=A._validated_case(request(merge(NEWTON_FLOW,Dict("cfd_resolution"=>"standard"))))["flow"]
    # closed trailing edge: one block, the single-block payload
    dir=mktempdir(SCRATCH)
    sharp=B.preview_mesh(SHARP,flow,dir;source=SHARP_CASE["airfoil"])
    m=sharp.mesh
    @test m["kind"]=="structured"&&(m["full_ni"],m["full_nj"])==(1025,97)
    @test m["ni"]>1&&m["nj"]>1&&length(m["x"])==m["ni"]*m["nj"]==length(m["y"])
    @test m["wall_i_start"]==129&&m["wall_i_end"]==897&&m["base_faces_per_side"]==0&&m["resolution"]=="standard"
    @test sharp.metadata["mesh_topology"]=="marched_C"&&sharp.metadata["cells"]==1024*96&&sharp.metadata["n_base"]==0
    re=3e6;dn1=.7/(re*sqrt((.026*re^(-1/7))/2))               # the wall spacing of the other compressible meshes
    @test sharp.metadata["wall_dn1"]==dn1&&isapprox(sharp.metadata["estimated_yplus"],.7;rtol=1e-12)
    @test sharp.metadata["farfield_angle_deg"]==2.0&&sharp.metadata["minimum_cell_area"]>0
    @test isfile(joinpath(dir,"mesh_full.msgpack"))&&isfile(joinpath(dir,"mesh_preview.msgpack"))
    @test haskey(m["quality"],"geometric_gate")

    # finite trailing edge: C+H, the payload kind of the app's C+H preview
    dir=mktempdir(SCRATCH)
    blunt=B.preview_mesh(SHARP,flow,dir;source=NACA_CASE["airfoil"])
    m=blunt.mesh
    @test m["kind"]=="structured_blocks"&&[b["id"] for b in m["blocks"]]==["C","H"]
    @test all(b->b["ni"]>1&&b["nj"]>1&&length(b["x"])==b["ni"]*b["nj"]==length(b["y"]),m["blocks"])
    c,h=m["blocks"]
    @test (c["full_ni"],c["full_nj"])==(1025,97)&&h["full_ni"]==129&&(h["full_nj"]-1)%4==0
    @test m["total_cells"]==blunt.metadata["cells"]==1024*96+128*(h["full_nj"]-1)
    @test blunt.metadata["mesh_topology"]=="marched_C_H"&&blunt.metadata["cfd_te_gap"]>1e-3
    @test m["quality_passed"] isa Bool&&m["quality_failures"] isa AbstractVector&&m["solver_supported"]===true
    full=A.decode(read(joinpath(dir,"mesh_full.msgpack")),:msgpack)
    grid(b)=(reshape(Float64.(b["x"]),b["ni"],b["nj"]),reshape(Float64.(b["y"]),b["ni"],b["nj"]))
    (cx,cy),(hx,hy)=grid.(full["blocks"]);nw=128;nf=768
    # the app's strip layout: row 1 on the lower leg, the last row on the upper leg, column 1 on the base
    @test all(hx[k,1]==cx[nw+2-k,1]&&hy[k,1]==cy[nw+2-k,1]&&hx[k,end]==cx[nw+nf+k,1]&&hy[k,end]==cy[nw+nf+k,1] for k in 1:nw+1)
    @test hypot(hx[1,end]-hx[1,1],hy[1,end]-hy[1,1])≈blunt.metadata["cfd_te_gap"]
    # a thickness override on the closed section gives the same topology
    thick=B.preview_mesh(SHARP,merge(flow,Dict("cfd_te_thickness"=>.005)),mktempdir(SCRATCH);source=SHARP_CASE["airfoil"])
    @test thick.mesh["kind"]=="structured_blocks"&&isapprox(thick.metadata["cfd_te_gap"],.005;atol=1e-9)

    # through the server's Create mesh request
    preview(flow)=A._preview_request(Dict("airfoil"=>Dict("kind"=>"naca4","id"=>"0012"),"mesh"=>true,"backends"=>["transonic"],
        "flow"=>merge(Dict{String,Any}("mach"=>.5,"reynolds"=>3e6,"alpha_deg"=>2.,"cfd_resolution"=>"standard"),flow)))
    case,want=preview(NEWTON_FLOW)
    payload=A._make_preview(case,want)
    @test want&&payload["mesh"]["kind"]=="structured_blocks"&&startswith(payload["mesh"]["artifact"],"/api/previews/")
    @test payload["geometry_metadata"]["mesh_topology"]=="marched_C_H"
    @test occursin("cfd_mesh = marched is no longer an option",message(()->preview(Dict("cfd_mesh"=>"marched"))))
end

@testset "Progress line, strip layout and status" begin
    # ITERATION,n,total,residual_scaled,residual_linf,residual_l2,cl,cd,mass_net_frac,cm from the march's six residuals
    six=[1e-5,4e-5,3e-5,2e-5,1e-9,7e-3]
    values=split(B._newton_iteration_line(42,700,six,.3,.01,2e-7,-.02),',')
    @test values[1]=="ITERATION"&&length(values)==10&&parse.(Int,values[2:3])==[42,700]
    numbers=parse.(Float64,values[4:10])
    @test numbers==[1e-5,7e-3,4e-5,.3,.01,2e-7,-.02]      # continuity; turbulence; largest flow equation; loads; mass; moment
    # the server's reader accepts the line and keeps every row of this worker
    events=Any[];julia=joinpath(Sys.BINDIR,Base.julia_exename())
    script="println(\"ITERATION,3,700,1e-5,7e-3,4e-5,0.3,0.01,2e-7,-0.02\");println(\"ITERATION,5,700,9e-6,6e-3,3e-5,0.31,0.011,1e-7,-0.021\")"
    @test B._log_process(`$julia --startup-file=no -e $script`,joinpath(DIR,"fake_solver.log"),event->push!(events,event),identity,
        "transonic",2.;every_history_row=true)
    rows=[e for e in events if get(e,"event","")=="iteration"]
    @test [e["iteration"] for e in rows]==[3,5]&&all(e->e["every_history_row"]===true&&e["total_iterations"]==700,rows)
    @test rows[1]["residual_scaled"]==1e-5&&rows[1]["residual_linf"]==7e-3&&rows[1]["residual_l2"]==4e-5&&rows[1]["cm"]==-.02
    empty!(events)
    B._log_process(`$julia --startup-file=no -e $script`,joinpath(DIR,"fake_solver.log"),event->push!(events,event),identity,"transonic",2.)
    @test !any(e->haskey(e,"every_history_row"),events)     # every other worker's events are unchanged

    # the solver stores the strip with the base as its wall; the fields are written back in the app's layout
    nw,ns=6,4;X=[100k+j for k in 1:nw+1,j in 1:ns+1]            # the app's strip nodes (k downstream, j across)
    hn=permutedims(X[:,end:-1:1])                                # newton_pair_strip_nodes
    @test all(begin
        i,kk=B._newton_strip_cell(ns,k,j)
        Set((hn[i,kk],hn[i+1,kk],hn[i,kk+1],hn[i+1,kk+1]))==Set((X[k,j],X[k+1,j],X[k,j+1],X[k+1,j+1]))
    end for k in 1:nw,j in 1:ns)

    # a point is eligible only when the march reports settled loads on the fine level
    settled=B._newton_guard(true;steps=120,budget=300,load_tol=1e-4,window=20)
    open_=B._newton_guard(false;steps=300,budget=300,load_tol=1e-4,window=20)
    @test settled["status"]=="early_stop_eligible"&&settled["early_stop_eligible"]&&settled["stopped_early"]&&settled["termination"]=="loads_settled"
    @test open_["status"]=="loads_not_settled"&&!open_["early_stop_eligible"]&&!open_["stopped_early"]&&open_["termination"]=="iteration_limit"
    # the job outcome follows the point status the march reported
    point(status,guard)=Dict{String,Any}("backend"=>"transonic","status"=>status,"cl"=>.3,"cd"=>.01,
        "diagnostics"=>Dict{String,Any}("rans_method"=>"newton","time_mode"=>"steady","convergence_guard"=>guard))
    done=point("completed",settled);unfinished=point("not_converged",open_)
    @test A._point_outcome([done])[1]=="completed"
    @test A._point_outcome([unfinished])[1]!="completed"
end
