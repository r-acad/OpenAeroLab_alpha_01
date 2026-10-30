# Application adapter for the native conservative, two-block thermal gas/SST
# operator. This file is included at CFDBackend module scope, after the solver
# modules have loaded in the isolated worker. No native flux or stage is changed.
if IS_WORKER && ARGS[1] == "--ch-transonic-worker"
    include(joinpath(@__DIR__,"runtime","gas_strip_prototype.jl"))
    include(joinpath(@__DIR__,"runtime","gas_strip_sst_prototype.jl"))
    include(joinpath(@__DIR__,"runtime","gas_strip_sst_steady.jl"))
end

function _ch_worker_settings(cfg::AbstractDict)
    String(get(cfg,"time_mode","steady"))=="steady" ||
        throw(ArgumentError("C+H currently supports steady local pseudo-time diagnostics; physical-time coupling is not enabled"))
    String(get(cfg,"viscosity_model","constant_kinematic"))=="constant_kinematic" ||
        throw(ArgumentError("C+H conservative gas/SST currently requires constant kinematic viscosity"))
    String(get(cfg,"resolution","standard")) in ("standard","fine","very_fine") ||
        throw(ArgumentError("C+H resolution must be standard, fine, or very_fine"))
    for key in ("sst_nonorth_correction","characteristic_i_outlet","refresh_flow_bc_before_sst","sfd_enabled")
        Bool(get(cfg,key,false)) && throw(ArgumentError("C+H does not support $key; no legacy single-block operation is substituted"))
    end
    Int(get(cfg,"mass_probe_stride",0))==0 || throw(ArgumentError("The old single-block SFD mass-stage probe is unavailable for C+H"))
    !Bool(get(cfg,"mesh_custom_dimensions_requested",false)) ||
        throw(ArgumentError("C+H uses the same resolution profiles as Create Mesh; custom legacy block dimensions are unavailable"))
    for key in ("trip_upper","trip_lower")
        Float64(get(cfg,key,0.))==0 || throw(ArgumentError("C+H conservative SST is fully turbulent; an imposed $key is not supported"))
    end
    mach=Float64(cfg["mach"]);re=Float64(cfg["reynolds"]);alpha=Float64(cfg["alpha_deg"])
    all(isfinite,(mach,re,alpha)) && 0<mach<=3 && re>=1e3 && abs(alpha)<=90 ||
        throw(ArgumentError("Invalid C+H Mach, Reynolds number, or angle"))
    steps=Int(cfg["steps"]);cfl=Float64(get(cfg,"cfl",.1))
    0<steps<=2_000_000 && isfinite(cfl) && 0<cfl<=.5 ||
        throw(ArgumentError("C+H requires 1..2000000 iterations and 0 < CFL <= 0.5"))
    stride=Int(get(cfg,"history_stride",25));stride>0 || throw(ArgumentError("history_stride must be positive"))
    wall_seconds=Float64(get(cfg,"wall_seconds",Inf))
    wall_seconds>0 && !isnan(wall_seconds) || throw(ArgumentError("wall_seconds must be positive"))
    maxdt=Float64(get(cfg,"max_pseudo_dt",.01));isfinite(maxdt) && maxdt>0 || throw(ArgumentError("max_pseudo_dt must be finite and positive"))
    window=Int(get(cfg,"convergence_window",2000));window>=100 || throw(ArgumentError("convergence_window must be at least 100 iterations"))
    (mach=mach,reynolds=re,alpha=alpha,steps=steps,cfl=cfl,stride=stride,
        gpu=Bool(cfg["gpu"]),wall_seconds=wall_seconds,max_pseudo_dt=maxdt,
        auto_stop=Bool(get(cfg,"auto_stop",true)),window=window,irsj=Bool(get(cfg,"irsj",false)))
end

function _ch_worker_source_hashes()
    roots=[@__FILE__,joinpath(@__DIR__,"cfd_backend.jl"),joinpath(@__DIR__,"ch_transonic_backend.jl"),joinpath(@__DIR__,"full_field_artifact.jl"),
        joinpath(@__DIR__,"protocol.jl"),joinpath(@__DIR__,"convergence.jl"),
        joinpath(@__DIR__,"ch_mesh.jl"),joinpath(@__DIR__,"transonic_mesh.jl"),
        joinpath(SRC,"collision","lattice.jl"),joinpath(SRC,"cmesh","cmesh_gen.jl"),
        joinpath(SRC,"cmesh","cmesh_metrics.jl"),joinpath(SRC,"cmesh","cmesh_solver_ka.jl"),
        [joinpath(@__DIR__,"runtime",p) for p in ("gas_strip_prototype.jl","gas_strip_sst_prototype.jl","gas_strip_sst_steady.jl")]...]
    seen=Set{String}();pending=copy(roots)
    while !isempty(pending)
        p=abspath(pop!(pending));p in seen && continue;push!(seen,p)
        for m in eachmatch(r"include\s*\(\s*joinpath\(\s*@__DIR__\s*,([^)]*)\)\s*\)",read(p,String))
            parts=[x.captures[1] for x in eachmatch(r"\"([^\"]+)\"",m.captures[1])]
            q=normpath(joinpath(dirname(p),parts...));isfile(q) && push!(pending,q)
        end
    end
    Dict(relpath(p,ROOT)=>bytes2hex(sha256(read(p))) for p in sort!(collect(seen)))
end

# Calculate force and moment from the very same native face fluxes used in the
# residual, including the straight H-block base; no separate traction formula.
function _ch_wall_loads(flow,m;faces_path=nothing)
    c,h=flow.c,flow.h;r=c.NIW+1:c.NIW+c.NIF
    cx,cy=map(a->Float64.(Array(view(a,r,1))),(c.Fj_mx,c.Fj_my))
    hx,hy=map(a->Float64.(Array(view(a,1,:))),(h.Fi_mx,h.Fi_my))
    force_x=-sum(cx)-sum(hx);force_y=-sum(cy)-sum(hy)
    speed=hypot(Float64(c.uxi),Float64(c.uyi));q=.5Float64(c.rho_inf)*speed^2
    ca,sa=Float64(c.ca),Float64(c.sa);moment=0.
    rows=Any[]
    for (k,i) in enumerate(r)
        x=.5(m.X[i,1]+m.X[i+1,1]);y=.5(m.Y[i,1]+m.Y[i+1,1])
        moment+=_aerodynamic_cm(x,y,-cx[k],-cy[k],q)
        faces_path===nothing || push!(rows,("C",i,x,y,-cx[k],-cy[k]))
    end
    for j in eachindex(hx)
        x=.5(m.strip.X[1,j]+m.strip.X[1,j+1]);y=.5(m.strip.Y[1,j]+m.strip.Y[1,j+1])
        moment+=_aerodynamic_cm(x,y,-hx[j],-hy[j],q)
        faces_path===nothing || push!(rows,("H",j,x,y,-hx[j],-hy[j]))
    end
    if faces_path!==nothing
        open(faces_path,"w") do io
            println(io,"block,index,x,y,fx,fy")
            for row in rows;println(io,join(row,','));end
        end
    end
    (cl=(-force_x*sa+force_y*ca)/q,cd=(force_x*ca+force_y*sa)/q,cm=moment,
        fx=force_x,fy=force_y,base_fx=-sum(hx),base_fy=-sum(hy))
end

function _ch_equation_norms(coupled)
    P=GasStripPrototype
    linf=zeros(6);l2=zeros(6);integrated=zeros(6);volume=0.
    # Device reductions retain all cells, rather than subsampling the residual.
    for q in (coupled.c,coupled.h)
        s=q.solver;vol=Float64.(s.vol);volume+=Float64(sum(vol))
        for (k,rr) in enumerate((P.residuals(s)...,q.turbulence.dck,q.turbulence.dcw))
            raw=Float64.(rr)./vol
            linf[k]=max(linf[k],Float64(maximum(abs,raw)))
            l2[k]+=Float64(sum(raw.*raw.*vol));integrated[k]+=Float64(sum(Float64.(rr)))
        end
    end
    (linf=linf,l2=sqrt.(l2./volume),integrated=integrated)
end

function _ch_equation_reference(c)
    rho=Float64(c.rho_inf);u=hypot(Float64(c.uxi),Float64(c.uyi))
    # Chord is normalized to one by the geometry generator. These are scales,
    # not replacements for the raw six-equation residuals retained separately.
    [rho*u,rho*u*u,rho*u*u,rho*u*(Float64(c.eint)*Float64(c.T_inf)+.5u*u),
        rho*u*max(Float64(c.k_inf),1e-12),rho*u*max(Float64(c.w_inf),1e-12)]
end

# The zero-step state is retained as the six-equation reduction reference and
# sent to the live viewer, but the production monitor admits accepted steps
# 1:max_steps only. Do not give it a fictitious accepted iteration.
function _ch_observe_iteration!(monitor,sample)
    sample.iteration==0 && return (stop=false,abort=false,status="initial_reference",
        reason="Initial state recorded; waiting for the first accepted iteration",
        metrics=Dict{String,Float64}())
    CFDConvergence.observe!(monitor,sample)
end

function _ch_write_fields(coupled,m,outdir,iteration)
    G=CMeshSolverKA
    mesh_path=joinpath(outdir,"mesh_full.msgpack");meshsha=bytes2hex(sha256(read(mesh_path)))
    blocks=Any[];fullblocks=Any[];total=0;bytes=0
    for (id,q) in zip(("C","H"),(coupled.c,coupled.h))
        s=q.solver;g=s.g;ir=g+1:g+s.NI;jr=g+1:g+s.NJ
        rho,u,v,t,nut=map(a->Float64.(Array(view(a,ir,jr))),(s.rho,s.ux,s.uy,s.T,s.nut))
        xc,yc=map(a->Float64.(Array(a)),(s.xc,s.yc))
        vort=Float64.(Array(s.gvx)).-Float64.(Array(s.guy))
        pinf=Float64(s.rho_inf)*Float64(G.CS2)*Float64(s.T_inf)
        qinf=.5Float64(s.rho_inf)*hypot(Float64(s.uxi),Float64(s.uyi))^2
        gamma=(5.0+Float64(s.drot))/(3.0+Float64(s.drot))
        at(i,j)=(xc[i,j],yc[i,j],hypot(u[i,j],v[i,j])/sqrt(gamma*Float64(G.CS2)*t[i,j]),
            (rho[i,j]*Float64(G.CS2)*t[i,j]-pinf)/qinf,rho[i,j]*Float64(G.CS2)*t[i,j],
            rho[i,j],u[i,j],v[i,j],t[i,j],vort[i,j],nut[i,j])
        stride=max(1,ceil(Int,sqrt(s.NI*s.NJ/18000)))
        ii=sort!(unique(vcat(collect(1:stride:s.NI),[s.NI])))
        jj=sort!(unique(vcat(collect(1:stride:s.NJ),[s.NJ])))
        samples=Dict(name=>Float64[] for name in FullFieldArtifact.COLUMNS)
        for j in jj,i in ii
            values=at(i,j)
            for (name,value) in zip(FullFieldArtifact.COLUMNS,values)
                isfinite(value) || error("Nonfinite C+H $id $name at ($i,$j)")
                push!(samples[name],value)
            end
        end
        filename="field_"*id*"_full.msgpack"
        full=FullFieldArtifact.write_full_field(at,joinpath(outdir,filename),s.NI,s.NJ,mesh_path,iteration)
        total+=full["cell_count"];bytes+=full["bytes"]
        push!(blocks,Dict{String,Any}("id"=>id,"kind"=>"structured_cells","ni"=>length(ii),"nj"=>length(jj),
            "samples"=>samples,"sample_shape"=>[length(ii),length(jj)],"sample_stride"=>stride,
            "i_indices"=>ii,"j_indices"=>jj,"full_ni"=>s.NI,"full_nj"=>s.NJ,
            "iteration"=>iteration,"mesh_sha256"=>meshsha,"cell_order"=>"j-major, i-fastest","full_artifact"=>filename))
        push!(fullblocks,Dict("id"=>id,"artifact"=>filename,"ni"=>s.NI,"nj"=>s.NJ))
        println("CH_FIELDS: saved ",id," block: ",s.NI*s.NJ," cells, ",full["part_count"]," binary part(s)");flush(stdout)
    end
    common=Dict{String,Any}("kind"=>"structured_blocks","cell_order"=>"j-major, i-fastest",
        "mesh_sha256"=>meshsha,"iteration"=>iteration,"total_cells"=>total,
        "units"=>"solver units","coordinate_units"=>"chord",
        "quantity_units"=>Dict("x"=>"x/c","y"=>"y/c","mach"=>"dimensionless","cp"=>"dimensionless",
            "pressure"=>"native solver pressure","density"=>"rho/rho_inf",
            "u"=>"native solver velocity","v"=>"native solver velocity","temperature"=>"T/T_inf",
            "vorticity"=>"native solver inverse time","nut"=>"native solver kinematic viscosity"))
    write(joinpath(outdir,"fields_blocks.msgpack"),AirfoilWire.encode(merge(common,Dict("blocks"=>blocks,"full_artifact"=>"fields_blocks_full.msgpack")),:msgpack))
    write(joinpath(outdir,"fields_blocks_full.msgpack"),AirfoilWire.encode(merge(common,Dict("blocks"=>fullblocks)),:msgpack))
    (;meshsha,total,bytes)
end

function _ch_transonic_worker(cfg_path::AbstractString)
    cfg=TOML.parsefile(cfg_path);settings=_ch_worker_settings(cfg)
    G=CMeshSolverKA;P=GasStripPrototype;S=GasStripSSTPrototype;L=GasStripSSTSteady
    outdir=String(cfg["outdir"]);mkpath(outdir);started=time();source_before=_ch_worker_source_hashes()
    println("C+H: preparing the exact Create Mesh geometry; native thermal gas + conservative SST, fully turbulent, no SFD");flush(stdout)
    af=haskey(cfg,"naca4_code") ? CMeshGen.naca4_points(cfg["naca4_code"]) : CMeshGen.read_airfoil_dat(cfg["geometry"])
    meshflow=Dict{String,Any}("reynolds"=>settings.reynolds,
        "cfd_te_thickness"=>get(cfg,"cfd_te_thickness",0.),"cfd_farfield_angle_deg"=>get(cfg,"cfd_farfield_angle_deg",0.))
    # Opt-in stall-mesh knobs (21 Oct 2026), passed only when requested, exactly as Create Mesh reads them.
    for key in ("cfd_stall_layer","cfd_le_ds_scale","cfd_ch_pair_factor","cfd_ch_te_multiple","cfd_ch_foil_cells","cfd_ch_surface_distribution","cfd_ch_h_cross")
        haskey(cfg,key) && (meshflow[key]=cfg[key])
    end
    built=_build_ch_mesh(af,meshflow;resolution=String(get(cfg,"resolution","standard")))
    # Streamline alignment (opt-in, development): the morph Create mesh applied, on the same hashed path file.
    aligned=haskey(cfg,"streamline") ? _ch_streamline_apply(built,cfg["streamline"]) : nothing
    aligned===nothing || (built=(;mesh=aligned.mesh,metadata=aligned.metadata);println(aligned.line);flush(stdout))
    m=built.mesh;metadata=built.metadata
    metadata["solver_supported"]=true
    metadata["solver_support_message"]="Native conservative two-block thermal gas/SST diagnostic; aerodynamic convergence and stall parity remain unqualified"
    metadata["geometry_support"]=haskey(cfg,"naca4_code") ? "NACA four-digit equations and natural C2 surface evaluation, identical to Create Mesh" : "Natural C2 surface evaluation, identical to Create Mesh"
    _write_ch_mesh_artifacts(m,metadata,outdir,String(get(cfg,"resolution","standard"));streamlines=aligned===nothing ? nothing : aligned.streamlines)
    println("MESH_READY,mesh_preview.msgpack");flush(stdout)
    println("C+H mesh generated: ",m.NI*m.NJ+m.strip.NI*m.strip.NJ," cells in two conforming blocks; air geometry gates ",metadata["quality_passed"] ? "passed" : "FAILED (diagnostic only)");flush(stdout)
    # Geometry is visible before CUDA initialization and before the first step.
    settings.gpu && !CUDA.functional() && error("CUDA was requested but is unavailable; no CPU fallback was used")
    backend=settings.gpu ? CUDA.CUDABackend() : KernelAbstractions.CPU()
    println("C+H initializing ",settings.gpu ? "GPU: "*CUDA.name(CUDA.device()) : "KernelAbstractions CPU"," state");flush(stdout)
    base=(m.strip.X[1,1],m.strip.Y[1,1],m.strip.X[1,end],m.strip.Y[1,end])
    mc=CMeshMetrics._build_metrics(m.X,m.Y,m.NIW,m.NIF,fill(UInt8(0),m.NI);extra_segments=[base])
    mh=CMeshMetrics._build_metrics(m.strip.X,m.strip.Y,m.NIW,0,fill(UInt8(4),m.strip.NI);channel_provenance=true,wall_segments=[base])
    cx,cy,cz,w,_,_=build_d3q39_static()
    build(met)=G.build_sol_ka(backend,met;drot=2f0,cxv=cx,cyv=cy,czv=cz,wv=w)
    c,h=build(mc),build(mh);G.GLOBAL_DT[]=0
    for s in (c,h)
        G.set_freestream_ka!(s,settings.mach,settings.alpha;pg_correction=false)
        s.fullvisc=Int32(1);s.nu_lam=Float32(hypot(s.uxi,s.uyi)/settings.reynolds)
        s.recon=Int32(1);s.wall_flux=true
    end
    if settings.irsj
        # Steady pseudo-time only: the physical-time pair steps refuse a block with irsj set.
        P.STEADY_IRSJ[]=true
        for s in (c,h);s.irsj=Int32(1);end
        println("C+H j-line implicit residual smoothing (IRS-J): on (steady local pseudo-time; both blocks)");flush(stdout)
    end
    flow=P.build_pair(c,h);coupled=S.build_coupled(flow;tu_inf=Float64(get(cfg,"tu_inf",.001)),mu_t_ratio=Float64(get(cfg,"mu_t_ratio",.1)))
    pilot=L.build_steady(coupled);scales=_ch_equation_reference(c)
    speed=hypot(Float64(c.uxi),Float64(c.uyi));massscale=Float64(c.rho_inf)*speed
    wsize=settings.window;confirm=max(100,wsize÷4)
    monitor=CFDConvergence.Monitor(CFDConvergence.Config(sfd_start=0,max_steps=settings.steps,
        window_iterations=wsize,min_post_sfd=2wsize,min_samples_per_window=20,
        confirmation_iterations=confirm,confirmation_checks=10,
        residual_scaled_limit=Float64(get(cfg,"mass_ceiling",.1)),
        mass_net_limit=Float64(get(cfg,"mass_net_limit",.001)),
        stop_on_stall=settings.auto_stop,stall_window_iterations=2wsize,stall_min_post_sfd=4wsize,
        stall_min_samples=30,stall_confirmation_iterations=wsize,stall_confirmation_checks=20))
    names=["mass","momentum_x","momentum_y","energy","rho_k","rho_omega"]
    history=Any[];six_since=0;six_checks=0;firstnorms=nothing;lastnorms=nothing
    final_load=nothing;final_decision=nothing;status="step_limit";failure="";rejections=0
    clock_min=Inf;clock_max=0.;last_record_time=time();march_started=time()
    history_io=open(joinpath(outdir,"iteration_history.csv"),"w")
    equation_io=open(joinpath(outdir,"equation_residual_history.csv"),"w")
    println(history_io,"iteration,residual_linf,residual_scaled,residual_l2,cl,cd,mass_net_frac,cm");flush(history_io)
    println(equation_io,join(vcat(["iteration"],["raw_"*n*"_Linf" for n in names],["raw_"*n*"_volume_L2" for n in names]),','));flush(equation_io)
    function record!(n)
        norms=_ch_equation_norms(coupled);load=_ch_wall_loads(flow,m)
        all(isfinite,vcat(norms.linf,norms.l2,[load.cl,load.cd,load.cm])) || error("Nonfinite C+H accepted state diagnostics")
        massnet=norms.integrated[1]/massscale
        raw=max(Float64(maximum(abs,c.dc_r)),Float64(maximum(abs,h.dc_r)))
        sample=(iteration=n,residual_scaled=norms.linf[1]/scales[1],residual_l2=norms.l2[1]/scales[1],cl=load.cl,cd=load.cd,mass_net_frac=massnet)
        push!(history,merge(sample,(residual_linf=raw,cm=load.cm)))
        @printf(history_io,"%d,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g\n",n,raw,sample.residual_scaled,sample.residual_l2,load.cl,load.cd,massnet,load.cm);flush(history_io)
        println(equation_io,join(vcat([n],norms.linf,norms.l2),','));flush(equation_io)
        @printf("ITERATION,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",n,settings.steps,sample.residual_scaled,raw,sample.residual_l2,load.cl,load.cd,massnet,load.cm)
        flush(stdout)
        decision=_ch_observe_iteration!(monitor,sample)
        norms,load,decision
    end
    try
        firstnorms,final_load,final_decision=record!(0);lastnorms=firstnorms
        println("C+H marching on ",settings.gpu ? "GPU" : "CPU",": shared six-equation flux, native conservative SSPRK3, CFL=",settings.cfl,"; each record is flushed live");flush(stdout)
        for n in 1:settings.steps
            isfile(joinpath(outdir,"STOP")) && (status="cooperatively_stopped";break)
            time()-march_started>=settings.wall_seconds && (status="wall_limit";break)
            accepted=L.step!(pilot,settings.cfl;max_pseudo_dt=settings.max_pseudo_dt,clock_margin=.9)
            rejections+=accepted.rejections;clock_min=min(clock_min,accepted.local_dt_min);clock_max=max(clock_max,accepted.local_dt_max)
            if n==1 || n%settings.stride==0 || n==settings.steps || time()-last_record_time>=2
                lastnorms,final_load,final_decision=record!(n);last_record_time=time()
                scaled_linf=lastnorms.linf./scales;scaled_l2=lastnorms.l2./scales
                ratio=lastnorms.linf./max.(firstnorms.linf,eps(Float64))
                six_ok=all(scaled_linf.<=.075) && all(scaled_l2.<=.015) && all(ratio.<=.1)
                if six_ok;six_since==0 && (six_since=n);six_checks+=1;else;six_since=0;six_checks=0;end
                if settings.auto_stop && final_decision.abort;status="diverged_stop";break;end
                if settings.auto_stop && final_decision.stop
                    if final_decision.status=="stalled_stop";status="stalled_stop";break;end
                    if six_ok && six_since>0 && n-six_since>=confirm && six_checks>=10
                        status="equation_and_force_guard_passed";break
                    end
                end
            end
        end
    catch err
        status="numerical_rejection";failure=sprint(showerror,err,catch_backtrace())
        println(stderr,"C+H numerical diagnostic stopped at accepted n=",pilot.iterations,": ",failure);flush(stderr)
    finally
        # An atomic failed native step restores the last accepted state. Export
        # that state as explicitly unqualified diagnostic evidence if possible.
        S.residual!(coupled)
        if isempty(history) || last(history).iteration!=pilot.iterations
            lastnorms,final_load,final_decision=record!(pilot.iterations)
        end
        close(history_io);close(equation_io)
    end
    pilot.iterations>=1 || error("C+H could not accept a single iteration: "*failure)
    gas=P.balance(flow);scalar=S.scalar_balance(coupled)
    gas.interface.exact && all(e->abs(e.face64_telescoping_error)<1e-7,gas.equations) &&
        all(e->abs(e.face64_telescoping_error)<1e-7,scalar) || error("C+H final common-flux/independent finite-volume closure failed")
    # The SST adapter gives both sides the same authoritative scalar face flux.
    ac,ah=coupled.c.turbulence,coupled.h.turbulence
    scalar_common=true
    for (a,b) in ((ac.Fjk,ah.Fjk),(ac.Fjw,ah.Fjw))
        ca=Array(view(a,:,1));hb=Array(view(b,:,1));ht=Array(view(b,:,h.NJ+1))
        scalar_common &= all(ca[c.NIW+1-k]==-hb[k] && ca[c.NIW+c.NIF+k]==ht[k] for k in 1:c.NIW)
    end
    scalar_common || error("C+H SST interface flux was not common")
    cpfile=joinpath(outdir,"surface_cp.csv")
    ridx=c.NIW+1:c.NIW+c.NIF;ghost_ridx=(first(ridx)+c.g):(last(ridx)+c.g)
    wr,wt=map(a->Float64.(Array(view(a,ghost_ridx,c.g+1))),(c.rho,c.T))
    wallx=[.5(m.X[i,1]+m.X[i+1,1]) for i in ridx];ile=argmin(wallx)
    pinf=Float64(c.rho_inf)*Float64(G.CS2)*Float64(c.T_inf);qinf=.5Float64(c.rho_inf)*speed^2
    open(cpfile,"w") do io
        println(io,"surface,x,cp")
        for k in eachindex(wallx);println(io,k>ile ? "upper" : "lower",',',wallx[k],',',(wr[k]*Float64(G.CS2)*wt[k]-pinf)/qinf);end
    end
    final_load=_ch_wall_loads(flow,m;faces_path=joinpath(outdir,"wall_force_faces.csv"))
    field=_ch_write_fields(coupled,m,outdir,pilot.iterations)
    source_after=_ch_worker_source_hashes();source_before==source_after || error("Native C+H worker source changed during this run")
    guard=CFDConvergence.convergence_report(monitor)
    converged=status=="equation_and_force_guard_passed"
    guard["continuity_force_guard_status"]=get(guard,"status","unavailable")
    guard["status"]=converged ? "six_equation_guard_passed" : status
    guard["reason"]=converged ? "All six scaled raw equations, signed global mass balance, and force windows passed with confirmation margin; mesh/physical parity remain unqualified" :
        "C+H diagnostic stopped with status $status; no aerodynamic convergence or stall qualification is claimed"
    guard["all_six_equations_confirmed"]=converged
    scaled_linf=lastnorms.linf./scales;scaled_l2=lastnorms.l2./scales
    tail=history[max(1,end-39):end]
    meta=Dict{String,Any}("mesh_mode"=>"conformal_C_H_v1","status"=>status,"failure"=>failure,
        "cl"=>final_load.cl,"cd"=>final_load.cd,"cm"=>final_load.cm,"steps"=>pilot.iterations,"max_steps"=>settings.steps,
        "gpu"=>settings.gpu,"device"=>settings.gpu ? CUDA.name(CUDA.device()) : "KernelAbstractions CPU",
        "time_mode"=>"steady","clock"=>"steady local pseudo-time","elapsed_ctu"=>0.,"physical_elapsed"=>0.,
        "physical_clock_unchanged"=>coupled.elapsed==0 && flow.elapsed==0,"coefficient_scope"=>"final native wall-flux diagnostic, including H-base; not a validated aerodynamic polar",
        "field_time_scope"=>"final accepted steady iterate","time_integration_scope"=>"Native shared six-equation conservative SSPRK3 with unequal local pseudo-clocks; no unsteady stall claim",
        "sst_model"=>"SST2003m, fully turbulent; no imposed transition trip","trip_upper"=>0.,"trip_lower"=>0.,
        "requested_trip_upper"=>Float64(get(cfg,"requested_trip_upper",0.)),"requested_trip_lower"=>Float64(get(cfg,"requested_trip_lower",0.)),"sfd_engaged"=>false,
        "rk_weight_closure"=>true,"rk_weight_closure_implementation"=>"native conservative SSPRK3 gas/SST stage construction; not the old single-block correction switch",
        "sst_wall_metric_correction"=>true,"sst_wall_metric_implementation"=>"native conservative SST face-wall distance; no legacy single-block switch",
        "viscosity_model"=>"constant_kinematic","reconstruction"=>1,"full_viscous_stress"=>true,"irsj"=>settings.irsj,
        "mach"=>settings.mach,"reynolds"=>settings.reynolds,"alpha_deg"=>settings.alpha,"cfl"=>settings.cfl,
        "geometry"=>metadata,"mesh_cells"=>field.total,"mesh_sha256"=>field.meshsha,
        "cp_csv"=>basename(cpfile),"iteration_csv"=>"iteration_history.csv","equation_residual_csv"=>"equation_residual_history.csv",
        "wall_force_csv"=>"wall_force_faces.csv","force_q_inf"=>qinf,"force_ca"=>Float64(c.ca),"force_sa"=>Float64(c.sa),
        "fields_blocks_msgpack"=>"fields_blocks.msgpack","field_full_msgpack"=>"fields_blocks_full.msgpack",
        "full_field_cells"=>field.total,"full_field_mesh_sha256"=>field.meshsha,"full_field_bytes"=>field.bytes,
        "residual_linf"=>last(history).residual_linf,"residual_scaled"=>last(history).residual_scaled,"residual_l2"=>last(history).residual_l2,
        "residual_ratio"=>lastnorms.linf[1]/max(firstnorms.linf[1],eps(Float64)),"residual_reference_iteration"=>0,
        "mass_net_frac"=>last(history).mass_net_frac,"mass_net_scope"=>"signed sum of native C+H continuity residual / rho_inf U_inf chord",
        "residual_scope"=>"all six native conservative gas/SST equations; raw per-volume and explicitly nondimensionalized norms retained",
        "equation_names"=>names,"raw_all6_Linf"=>lastnorms.linf,"raw_all6_volume_L2"=>lastnorms.l2,
        "equation_reference_scales"=>scales,"scaled_all6_Linf"=>scaled_linf,"scaled_all6_volume_L2"=>scaled_l2,
        "convergence_guard"=>guard,"aerodynamic_converged"=>converged,"numerically_stationary"=>converged,
        "physical_sanity"=>isfinite(final_load.cl) && isfinite(final_load.cd) && isfinite(final_load.cm),
        "physical_sanity_scope"=>"finite diagnostics only; no validation","load_drift"=>maximum(r.cl for r in tail)-minimum(r.cl for r in tail),
        "drag_drift"=>maximum(r.cd for r in tail)-minimum(r.cd for r in tail),
        "common_six_equation_flux"=>true,"common_gas_interface_exact"=>gas.interface.exact,"common_sst_interface_exact"=>scalar_common,
        "face64_gas_closure_error"=>[e.face64_telescoping_error for e in gas.equations],
        "face64_sst_closure_error"=>[e.face64_telescoping_error for e in scalar],
        "rejected_atomic_trials"=>rejections,"accepted_local_clock_min"=>clock_min,"accepted_local_clock_max"=>clock_max,
        "source_sha256_before"=>source_before,"source_sha256_after"=>source_after,"source_changed_during_run"=>false,
        "loaded_source_scope"=>"Worker records source hashes before and after this run; sources were not loaded from a frozen campaign snapshot",
        "elapsed_wall_seconds"=>time()-started,"elapsed_march_wall_seconds"=>time()-march_started,
        "full_campaign_released"=>false,"stall_validated"=>false,"parity_qualified"=>false)
    open(joinpath(outdir,"result.toml"),"w") do io;TOML.print(io,meta);end
    println("C+H worker complete: n=",pilot.iterations," status=",status,"; actual two-block fields and diagnostics saved");flush(stdout)
    nothing
end
