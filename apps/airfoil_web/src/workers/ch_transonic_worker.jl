# Application adapter for the native conservative, two-block thermal gas/SST
# operator. This file is included at CFDBackend module scope, after the solver
# modules have loaded in the isolated worker. No native flux or stage is changed.
if IS_WORKER && ARGS[1] == "--ch-transonic-worker"
    include(joinpath(@__DIR__,"..","runtime","gas_strip_prototype.jl"))
    include(joinpath(@__DIR__,"..","runtime","gas_strip_sst_prototype.jl"))
    include(joinpath(@__DIR__,"..","runtime","gas_strip_sst_steady.jl"))
end

function _ch_worker_settings(cfg::AbstractDict)
    time_mode=String(get(cfg,"time_mode","steady"))
    time_mode in ("steady","physical") || throw(ArgumentError("C+H time_mode must be steady or physical"))
    physical=time_mode=="physical"
    viscosity=String(get(cfg,"viscosity_model","constant_kinematic"))
    viscosity in ("constant_kinematic","sutherland","constant_dynamic") ||
        throw(ArgumentError("C+H viscosity_model must be constant_kinematic, sutherland, or constant_dynamic"))
    temperature=Float64(get(cfg,"freestream_temperature_k",288.15))
    viscosity!="sutherland" || (isfinite(temperature) && 100<=temperature<=2000) ||
        throw(ArgumentError("The Sutherland law needs a freestream static temperature of 100 to 2000 K"))
    irsj=Bool(get(cfg,"irsj",false))
    !(irsj && physical) || throw(ArgumentError("Implicit residual smoothing (IRS-J) applies to steady pseudo-time only; the physical-time step refuses it"))
    !(irsj && viscosity!="constant_kinematic") ||
        throw(ArgumentError("Implicit residual smoothing (IRS-J) forms its coefficient with a constant kinematic viscosity; it is not available with the $viscosity law"))
    String(get(cfg,"resolution","standard")) in ("standard","fine","very_fine") ||
        throw(ArgumentError("C+H resolution must be standard, fine, or very_fine"))
    for key in ("sst_nonorth_correction","characteristic_i_outlet","refresh_flow_bc_before_sst","sfd_enabled")
        Bool(get(cfg,key,false)) && throw(ArgumentError("C+H does not support $key; no legacy single-block operation is substituted"))
    end
    Int(get(cfg,"mass_probe_stride",0))==0 || throw(ArgumentError("The old single-block SFD mass-stage probe is unavailable for C+H"))
    !Bool(get(cfg,"mesh_custom_dimensions_requested",false)) ||
        throw(ArgumentError("C+H uses the same resolution profiles as Create Mesh; custom legacy block dimensions are unavailable"))
    # Prescribed trips, x/c per side; 0 leaves that side fully turbulent.
    trip_upper=Float64(get(cfg,"trip_upper",0.));trip_lower=Float64(get(cfg,"trip_lower",0.))
    all(x->isfinite(x) && 0<=x<=1,(trip_upper,trip_lower)) || throw(ArgumentError("C+H trip locations must lie in 0..1 chord (0: no trip on that side)"))
    # Physical time: the run ends at duration_ctu (or the step or wall-clock limit); forces are sampled at least every
    # force_sample_dt_ctu and averaged after burnin_ctu in blocks of block_ctu (CFDTimeStatistics).
    duration=Float64(get(cfg,"duration_ctu",100.));burnin=Float64(get(cfg,"burnin_ctu",50.))
    block=Float64(get(cfg,"block_ctu",5.));sample=Float64(get(cfg,"force_sample_dt_ctu",.1))
    if physical
        isfinite(duration) && duration>0 && isfinite(burnin) && 0<=burnin<duration && isfinite(block) && block>0 ||
            throw(ArgumentError("physical duration, burn-in and block duration are invalid"))
        isfinite(sample) && sample>0 || throw(ArgumentError("physical force sampling interval is invalid"))
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
        auto_stop=Bool(get(cfg,"auto_stop",true)),window=window,irsj=irsj,
        physical=physical,duration_ctu=duration,burnin_ctu=burnin,block_ctu=block,force_sample_dt_ctu=sample,
        viscosity=viscosity,temperature_k=temperature,trip_upper=trip_upper,trip_lower=trip_lower)
end

function _ch_worker_source_hashes()
    roots=[@__FILE__,joinpath(@__DIR__,"..","backends","cfd_backend.jl"),joinpath(@__DIR__,"..","backends","ch_transonic_backend.jl"),joinpath(@__DIR__,"..","full_field_artifact.jl"),
        joinpath(@__DIR__,"..","..","..","common","FieldSnapshots.jl"),
        joinpath(@__DIR__,"..","protocol.jl"),joinpath(@__DIR__,"..","diagnostics","convergence.jl"),
        joinpath(@__DIR__,"..","mesh","ch_mesh.jl"),joinpath(@__DIR__,"..","mesh","transonic_mesh.jl"),
        joinpath(SRC,"collision","lattice.jl"),joinpath(SRC,"cmesh","cmesh_gen.jl"),
        joinpath(SRC,"cmesh","cmesh_metrics.jl"),joinpath(SRC,"compressible","cmesh_solver_ka.jl"),
        [joinpath(@__DIR__,"..","runtime",p) for p in ("gas_strip_prototype.jl","gas_strip_sst_prototype.jl","gas_strip_sst_steady.jl")]...]
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

function _ch_equation_norm_work(coupled)
    blocks=map((coupled.c,coupled.h)) do q
        vol=Float64.(q.solver.vol)
        (vol=vol,raw=similar(vol),weighted=similar(vol))
    end
    (blocks=blocks,volume=sum(Float64(sum(b.vol)) for b in blocks))
end

function _ch_equation_norms(coupled,normwork=_ch_equation_norm_work(coupled))
    P=GasStripPrototype
    linf=zeros(6);l2=zeros(6);integrated=zeros(6)
    # Device reductions retain all cells, rather than subsampling the residual.
    for (q,b) in zip((coupled.c,coupled.h),normwork.blocks)
        s=q.solver
        for (k,rr) in enumerate((P.residuals(s)...,q.turbulence.dck,q.turbulence.dcw))
            @. b.raw=Float64(rr)/b.vol
            linf[k]=max(linf[k],Float64(maximum(abs,b.raw)))
            @. b.weighted=b.raw*b.raw*b.vol
            l2[k]+=Float64(sum(b.weighted))
            @. b.weighted=Float64(rr)
            integrated[k]+=Float64(sum(b.weighted))
        end
    end
    (linf=linf,l2=sqrt.(l2./normwork.volume),integrated=integrated)
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

function _ch_write_fields(coupled,m,outdir,iteration;prefix="")
    G=CMeshSolverKA
    # One block at a time, as before: a block's host copies are made when the writer below reaches it.
    function source(q)
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
        at
    end
    _write_block_fields([(id=id,ni=q.solver.NI,nj=q.solver.NJ,source=()->source(q))
        for (id,q) in zip(("C","H"),(coupled.c,coupled.h))],outdir,iteration;prefix)
end

"""Write the cell fields of a two-block (C, H) solution: the sampled preview, each block's full field and the manifests.

`sources` holds (id, ni, nj, source) per block; `source()` returns `at(i, j)` with the values of
`FullFieldArtifact.COLUMNS` at cell (i, j) of that block in the layout of its mesh block. This is the writer of the
coupled C+H worker, shared with the two-block Implicit Newton worker (newton_worker.jl).
"""
function _write_block_fields(sources,outdir,iteration;prefix="")
    mesh_path=joinpath(outdir,"mesh_full.msgpack");meshsha=bytes2hex(sha256(read(mesh_path)))
    blocks=Any[];fullblocks=Any[];total=0;bytes=0
    for b in sources
        id=b.id;ni=b.ni;nj=b.nj;at=b.source()
        stride=max(1,ceil(Int,sqrt(ni*nj/18000)))
        ii=sort!(unique(vcat(collect(1:stride:ni),[ni])))
        jj=sort!(unique(vcat(collect(1:stride:nj),[nj])))
        samples=Dict(name=>Float64[] for name in FullFieldArtifact.COLUMNS)
        for j in jj,i in ii
            values=at(i,j)
            for (name,value) in zip(FullFieldArtifact.COLUMNS,values)
                isfinite(value) || error("Nonfinite C+H $id $name at ($i,$j)")
                push!(samples[name],value)
            end
        end
        filename=prefix*"field_"*id*"_full.msgpack"
        full=FullFieldArtifact.write_full_field(at,joinpath(outdir,filename),ni,nj,mesh_path,iteration)
        total+=full["cell_count"];bytes+=full["bytes"]
        push!(blocks,Dict{String,Any}("id"=>id,"kind"=>"structured_cells","ni"=>length(ii),"nj"=>length(jj),
            "samples"=>samples,"sample_shape"=>[length(ii),length(jj)],"sample_stride"=>stride,
            "i_indices"=>ii,"j_indices"=>jj,"full_ni"=>ni,"full_nj"=>nj,
            "iteration"=>iteration,"mesh_sha256"=>meshsha,"cell_order"=>"j-major, i-fastest","full_artifact"=>filename))
        push!(fullblocks,Dict("id"=>id,"artifact"=>filename,"ni"=>ni,"nj"=>nj))
        println("CH_FIELDS: saved ",id," block: ",ni*nj," cells, ",full["part_count"]," binary part(s)");flush(stdout)
    end
    common=Dict{String,Any}("kind"=>"structured_blocks","cell_order"=>"j-major, i-fastest",
        "mesh_sha256"=>meshsha,"iteration"=>iteration,"total_cells"=>total,
        "units"=>"solver units","coordinate_units"=>"chord",
        "quantity_units"=>Dict("x"=>"x/c","y"=>"y/c","mach"=>"dimensionless","cp"=>"dimensionless",
            "pressure"=>"native solver pressure","density"=>"rho/rho_inf",
            "u"=>"native solver velocity","v"=>"native solver velocity","temperature"=>"T/T_inf",
            "vorticity"=>"native solver inverse time","nut"=>"native solver kinematic viscosity"))
    mainfile=isempty(prefix) ? "fields_blocks.msgpack" : @sprintf("live_field_%06d.msgpack",iteration)
    fullfile=prefix*"fields_blocks_full.msgpack"
    FullFieldArtifact.FieldSnapshots.atomic_write(joinpath(outdir,fullfile),AirfoilWire.encode(merge(common,Dict("blocks"=>fullblocks)),:msgpack))
    FullFieldArtifact.FieldSnapshots.atomic_write(joinpath(outdir,mainfile),AirfoilWire.encode(merge(common,Dict("blocks"=>blocks,"full_artifact"=>fullfile)),:msgpack))
    (;meshsha,total,bytes)
end

# What the trip intermittency of the C block's metrics does on this mesh, per side (the sides and chord stations are
# those of CMeshMetrics' wall-order rule): the number of foil faces without SST production (intermittency 0) and the
# chord station of the first face with full production (intermittency 1).
function _ch_trip_faces(m,metrics)
    first_wall=m.NIW+1;last_wall=m.NIW+m.NIF+1
    tex=.5(m.X[first_wall,1]+m.X[last_wall,1]);tey=.5(m.Y[first_wall,1]+m.Y[last_wall,1])
    le=argmax(i->(m.X[i,1]-tex)^2+(m.Y[i,1]-tey)^2,first_wall:last_wall)
    cx,cy=tex-m.X[le,1],tey-m.Y[le,1]
    station(i)=((.5(m.X[i,1]+m.X[i+1,1])-m.X[le,1])*cx+(.5(m.Y[i,1]+m.Y[i+1,1])-m.Y[le,1])*cy)/(cx^2+cy^2)
    result=Dict{String,Any}()
    for (side,faces) in (("lower",first_wall:le-1),("upper",le:last_wall-1))
        result[side*"_laminar_faces"]=count(i->metrics.gammat[i,1]==0,faces)
        full=[station(i) for i in faces if metrics.gammat[i,1]==1]
        result[side*"_turbulent_from_x"]=isempty(full) ? 1. : minimum(full)
    end
    result
end

function _ch_transonic_worker(cfg_path::AbstractString)
    cfg=TOML.parsefile(cfg_path);settings=_ch_worker_settings(cfg)
    G=CMeshSolverKA;P=GasStripPrototype;S=GasStripSSTPrototype;L=GasStripSSTSteady
    outdir=String(cfg["outdir"]);mkpath(outdir);started=time();source_before=_ch_worker_source_hashes()
    physical=settings.physical;tripped=settings.trip_upper>0 || settings.trip_lower>0
    println("C+H: preparing the exact Create Mesh geometry; native thermal gas + conservative SST, ",
        tripped ? "prescribed trips (upper x/c=$(settings.trip_upper), lower x/c=$(settings.trip_lower); 0: none)" : "fully turbulent",
        ", ",replace(settings.viscosity,'_'=>' ')," viscosity, ",physical ? "physical time" : "steady pseudo-time",", no SFD");flush(stdout)
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
    # Trip intermittency of the C block's foil columns (CMeshMetrics): 0 ahead of each side's station along the chord
    # line, a linear ramp over 0.02 c, then 1. A side without a trip gets a station ahead of the leading edge, which
    # makes it 1 on that whole side. The SST reads the field only when a trip is prescribed (build_coupled below).
    station(x)=x>0 ? x : -1.
    mc=CMeshMetrics._build_metrics(m.X,m.Y,m.NIW,m.NIF,fill(UInt8(0),m.NI);extra_segments=[base],
        trip_upper=station(settings.trip_upper),trip_lower=station(settings.trip_lower),trip_side=:wall_order)
    mh=CMeshMetrics._build_metrics(m.strip.X,m.strip.Y,m.NIW,0,fill(UInt8(4),m.strip.NI);channel_provenance=true,wall_segments=[base])
    trip_faces=_ch_trip_faces(m,mc)
    cx,cy,cz,w,_,_=build_d3q39_static()
    build(met)=G.build_sol_ka(backend,met;drot=2f0,cxv=cx,cyv=cy,czv=cz,wv=w)
    c,h=build(mc),build(mh);G.GLOBAL_DT[]=0
    for s in (c,h)
        G.set_freestream_ka!(s,settings.mach,settings.alpha;pg_correction=false)
        s.fullvisc=Int32(1);s.nu_lam=Float32(hypot(s.uxi,s.uyi)/settings.reynolds)
        s.recon=Int32(1);s.wall_flux=true
        # nu_lam stays the freestream kinematic viscosity; the law makes mu = rho_inf nu_lam f(T) in both blocks.
        settings.viscosity=="sutherland" && G.set_air_sutherland_ka!(s,settings.temperature_k)
        settings.viscosity=="constant_dynamic" && G.set_air_constant_dynamic_ka!(s)
    end
    println("C+H viscosity: ",settings.viscosity=="sutherland" ? @sprintf("Sutherland air, T_inf=%.2f K, S=110.4 K",settings.temperature_k) :
        settings.viscosity=="constant_dynamic" ? "constant molecular dynamic" : "constant kinematic");flush(stdout)
    tripped && println("C+H trips: ",trip_faces["upper_laminar_faces"]," upper and ",trip_faces["lower_laminar_faces"],
        " lower foil faces ahead of the trip have no SST production; fully turbulent from x/c=",
        trip_faces["upper_turbulent_from_x"]," (upper) and ",trip_faces["lower_turbulent_from_x"]," (lower)");flush(stdout)
    if settings.irsj
        # Steady pseudo-time only: the physical-time pair steps refuse a block with irsj set.
        P.STEADY_IRSJ[]=true
        for s in (c,h);s.irsj=Int32(1);end
        println("C+H j-line implicit residual smoothing (IRS-J): on (steady local pseudo-time; both blocks)");flush(stdout)
    end
    flow=P.build_pair(c,h);coupled=S.build_coupled(flow;tu_inf=Float64(get(cfg,"tu_inf",.001)),mu_t_ratio=Float64(get(cfg,"mu_t_ratio",.1)),
        intermittency=tripped)
    # A physical-time run starts from the freestream, as the single-block worker's does: a steady checkpoint is neither
    # read nor written.
    warm_source=physical ? "" : _rans_load_restart!(cfg,joinpath(outdir,"mesh_full.msgpack"),"conformal_C_H_v1",
        [(name="C",solver=c,extras=(rk=coupled.c.turbulence.rk,rw=coupled.c.turbulence.rw)),
         (name="H",solver=h,extras=(rk=coupled.h.turbulence.rk,rw=coupled.h.turbulence.rw))])
    isempty(warm_source) || S.residual!(coupled)
    # Steady: the local pseudo-time relaxation. Physical: S.step! on the pair itself, one global step for both blocks.
    pilot=physical ? nothing : L.build_steady(coupled);scales=_ch_equation_reference(c)
    accepted_steps()=physical ? coupled.accepted_steps : pilot.iterations
    speed=hypot(Float64(c.uxi),Float64(c.uyi));massscale=Float64(c.rho_inf)*speed
    # Convective time units (CTU): the solver's time times U_inf/chord, with the chord of the meshed wall.
    wall_nodes=view(m.X,c.NIW+1:c.NIW+c.NIF+1,1);ctu=speed/(maximum(wall_nodes)-minimum(wall_nodes))
    elapsed_ctu=0.;next_force_sample_ctu=settings.force_sample_dt_ctu;dt_ctu_min=Inf;dt_ctu_max=0.
    physical_duration_reached=false;last_dt_ctu=0.;last_clock_line=0.
    physical_history=Dict{String,Any}[]
    physical_io=physical ? open(joinpath(outdir,"physical_force_history.csv"),"w") : nothing
    physical && (println(physical_io,"iteration,t_ctu,applied_dt_ctu,cl,cd,cm");flush(physical_io))
    wsize=settings.window;confirm=max(100,wsize÷4)
    monitor=CFDConvergence.Monitor(CFDConvergence.Config(sfd_start=0,max_steps=settings.steps,
        window_iterations=wsize,min_post_sfd=2wsize,min_samples_per_window=20,
        confirmation_iterations=confirm,confirmation_checks=10,
        residual_scaled_limit=Float64(get(cfg,"mass_ceiling",.1)),
        mass_net_limit=Float64(get(cfg,"mass_net_limit",.001)),
        stop_on_stall=settings.auto_stop && !physical,stall_window_iterations=2wsize,stall_min_post_sfd=4wsize,
        stall_min_samples=30,stall_confirmation_iterations=wsize,stall_confirmation_checks=20))
    names=["mass","momentum_x","momentum_y","energy","rho_k","rho_omega"]
    history=Any[];six_since=0;six_checks=0;firstnorms=nothing;lastnorms=nothing
    final_load=nothing;final_decision=nothing;status="step_limit";failure="";rejections=0
    clock_min=Inf;clock_max=0.;last_record_time=time();march_started=time()
    profile_stages=settings.gpu && Bool(get(cfg,"profile_stages",false))
    profile_step_seconds=Ref(0.0);profile_record_seconds=Ref(0.0)
    profile_step_samples=Ref(0);profile_record_samples=Ref(0)
    history_io=open(joinpath(outdir,"iteration_history.csv"),"w")
    equation_io=open(joinpath(outdir,"equation_residual_history.csv"),"w")
    normwork=_ch_equation_norm_work(coupled)
    println(history_io,"iteration,residual_linf,residual_scaled,residual_l2,cl,cd,mass_net_frac,cm");flush(history_io)
    println(equation_io,join(vcat(["iteration"],["raw_"*n*"_Linf" for n in names],["raw_"*n*"_volume_L2" for n in names]),','));flush(equation_io)
    function record!(n)
        norms=_ch_equation_norms(coupled,normwork);load=_ch_wall_loads(flow,m)
        all(isfinite,vcat(norms.linf,norms.l2,[load.cl,load.cd,load.cm])) || error("Nonfinite C+H accepted state diagnostics")
        massnet=norms.integrated[1]/massscale
        raw=max(Float64(maximum(abs,c.dc_r)),Float64(maximum(abs,h.dc_r)))
        sample=(iteration=n,residual_scaled=norms.linf[1]/scales[1],residual_l2=norms.l2[1]/scales[1],cl=load.cl,cd=load.cd,mass_net_frac=massnet)
        push!(history,merge(sample,(residual_linf=raw,cm=load.cm)))
        @printf(history_io,"%d,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g\n",n,raw,sample.residual_scaled,sample.residual_l2,load.cl,load.cd,massnet,load.cm);flush(history_io)
        println(equation_io,join(vcat([n],norms.linf,norms.l2),','));flush(equation_io)
        if physical
            # The force sample at the pair's own clock; the statistics integrate these on the actual, nonuniform times.
            push!(physical_history,Dict{String,Any}("iteration"=>n,"t_ctu"=>elapsed_ctu,"cl"=>load.cl,"cd"=>load.cd,"cm"=>load.cm))
            @printf(physical_io,"%d,%.16g,%.16g,%.12g,%.12g,%.12g\n",n,elapsed_ctu,last_dt_ctu,load.cl,load.cd,load.cm);flush(physical_io)
            next_force_sample_ctu=elapsed_ctu+settings.force_sample_dt_ctu
            if n>=1 && (n==1 || physical_duration_reached || time()-last_clock_line>=5)
                @printf("PHYSICAL_TIME,%d,%.12g,%.12g,%.12g\n",n,elapsed_ctu,settings.duration_ctu,last_dt_ctu);last_clock_line=time()
            end
        end
        @printf("ITERATION,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",n,settings.steps,sample.residual_scaled,raw,sample.residual_l2,load.cl,load.cd,massnet,load.cm)
        flush(stdout)
        decision=_ch_observe_iteration!(monitor,sample)
        norms,load,decision
    end
    function timed_record!(n)
        t=profile_stages ? time() : 0.0
        result=record!(n)
        if profile_stages
            CUDA.synchronize()
            profile_record_seconds[]+=time()-t
            profile_record_samples[]+=1
        end
        result
    end
    try
        firstnorms,final_load,final_decision=timed_record!(0);lastnorms=firstnorms
        println("C+H marching on ",settings.gpu ? "GPU" : "CPU",": shared six-equation flux, native conservative SSPRK3, CFL=",settings.cfl,
            physical ? "; physical time, one global step for both blocks, $(settings.duration_ctu) CTU requested" : "",
            "; each record is flushed live");flush(stdout)
        snapshots=FullFieldArtifact.FieldSnapshots.Schedule(get(cfg,"snapshot_interval_seconds",600.))
        println("Field snapshots: every ",snapshots.interval/60," wall-clock minutes (0 = disabled); final fields are saved separately.")
        for n in 1:settings.steps
            isfile(joinpath(outdir,"STOP")) && (status="cooperatively_stopped";break)
            time()-march_started>=settings.wall_seconds && (status="wall_limit";break)
            step_start=profile_stages ? time() : 0.0
            if physical
                # The last step is shortened to land on the requested duration.
                accepted=S.step!(coupled,settings.cfl;max_dt=(settings.duration_ctu-elapsed_ctu)/ctu)
                last_dt_ctu=accepted.dt*ctu;elapsed_ctu=coupled.elapsed*ctu
                dt_ctu_min=min(dt_ctu_min,last_dt_ctu);dt_ctu_max=max(dt_ctu_max,last_dt_ctu)
                physical_duration_reached=elapsed_ctu>=settings.duration_ctu*(1-1e-7)
                rejections+=accepted.rejections;clock_min=min(clock_min,accepted.dt);clock_max=max(clock_max,accepted.dt)
            else
                accepted=L.step!(pilot,settings.cfl;max_pseudo_dt=settings.max_pseudo_dt,clock_margin=.9)
                rejections+=accepted.rejections;clock_min=min(clock_min,accepted.local_dt_min);clock_max=max(clock_max,accepted.local_dt_max)
            end
            if profile_stages
                CUDA.synchronize()
                profile_step_seconds[]+=time()-step_start
                profile_step_samples[]+=1
            end
            FullFieldArtifact.FieldSnapshots.save_if_due!(snapshots) do
                _ch_write_fields(coupled,m,outdir,n;prefix=@sprintf("live_%06d_",n))
                println("FIELD_SNAPSHOT,",@sprintf("live_field_%06d.msgpack",n),",",n);flush(stdout)
            end
            if physical
                if n==1 || n%settings.stride==0 || elapsed_ctu>=next_force_sample_ctu || physical_duration_reached ||
                   n==settings.steps || time()-last_record_time>=2
                    lastnorms,final_load,final_decision=timed_record!(n);last_record_time=time()
                    if settings.auto_stop && final_decision.abort;status="diverged_stop";break;end
                end
                physical_duration_reached && (status="physical_duration_reached";break)
            elseif n==1 || n%settings.stride==0 || n==settings.steps || time()-last_record_time>=2
                lastnorms,final_load,final_decision=timed_record!(n);last_record_time=time()
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
        println(stderr,"C+H numerical diagnostic stopped at accepted n=",accepted_steps(),": ",failure);flush(stderr)
    finally
        # An atomic failed native step restores the last accepted state. Export
        # that state as explicitly unqualified diagnostic evidence if possible.
        S.residual!(coupled)
        if isempty(history) || last(history).iteration!=accepted_steps()
            lastnorms,final_load,final_decision=timed_record!(accepted_steps())
        end
        close(history_io);close(equation_io);physical && close(physical_io)
    end
    accepted_steps()>=1 || error("C+H could not accept a single iteration: "*failure)
    # Force means of a physical-time run after its burn-in, with their sampling qualification. The final coefficients
    # and fields stay those of the last instant.
    time_stats=physical ? CFDTimeStatistics.time_statistics(physical_history;burn_in_ctu=settings.burnin_ctu,
            block_duration_ctu=settings.block_ctu,requested_duration_ctu=settings.duration_ctu,
            time_step_refined=Bool(get(cfg,"physical_time_refined",false))) :
        Dict{String,Any}("available"=>false,"reason"=>"steady pseudo-time has no physical averaging clock","physical_mean_eligible"=>false)
    if physical
        @printf("C+H physical final: t=%.12g/%.12g CTU, %d steps, sampling=%s, final field is instantaneous\n",
            elapsed_ctu,settings.duration_ctu,accepted_steps(),time_stats["status"]);flush(stdout)
    end
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
    field=_ch_write_fields(coupled,m,outdir,accepted_steps())
    source_after=_ch_worker_source_hashes();source_before==source_after || error("Native C+H worker source changed during this run")
    guard=CFDConvergence.convergence_report(monitor)
    converged=status=="equation_and_force_guard_passed"
    guard["continuity_force_guard_status"]=get(guard,"status","unavailable")
    guard["status"]=converged ? "six_equation_guard_passed" : status
    guard["reason"]=converged ? "All six scaled raw equations, signed global mass balance, and force windows passed with confirmation margin; mesh/physical parity remain unqualified" :
        "C+H diagnostic stopped with status $status; no aerodynamic convergence or stall qualification is claimed"
    if physical
        # The steady stop guard does not judge a time-accurate march: its duration and force statistics do.
        guard["status"]=physical_duration_reached ? "physical_time_complete" : "physical_time_incomplete"
        guard["reason"]="Physical-time run ended with status $status at $(elapsed_ctu) of $(settings.duration_ctu) CTU; the force statistics give the sampling status, and no steady convergence is claimed"
    end
    guard["all_six_equations_confirmed"]=converged
    scaled_linf=lastnorms.linf./scales;scaled_l2=lastnorms.l2./scales
    tail=history[max(1,end-39):end]
    meta=Dict{String,Any}("mesh_mode"=>"conformal_C_H_v1","status"=>status,"failure"=>failure,
        "stage_profile"=>Dict("enabled"=>profile_stages,"step_seconds"=>profile_step_seconds[],
            "record_seconds"=>profile_record_seconds[],"step_samples"=>profile_step_samples[],
            "record_samples"=>profile_record_samples[],
            "scope"=>"CUDA synchronized accepted-step and live diagnostic timings; opt-in diagnostic mode"),
        "cl"=>final_load.cl,"cd"=>final_load.cd,"cm"=>final_load.cm,"steps"=>accepted_steps(),"max_steps"=>settings.steps,
        "gpu"=>settings.gpu,"device"=>settings.gpu ? CUDA.name(CUDA.device()) : "KernelAbstractions CPU",
        "time_mode"=>physical ? "physical" : "steady","clock"=>physical ? "physical time" : "steady local pseudo-time",
        "elapsed_ctu"=>elapsed_ctu,"physical_elapsed"=>coupled.elapsed,
        "physical_clock_unchanged"=>coupled.elapsed==0 && flow.elapsed==0,
        "coefficient_scope"=>physical ? "final instantaneous native wall-flux coefficients, including H-base; the force means are the time_statistics entries" :
            "final native wall-flux diagnostic, including H-base; not a validated aerodynamic polar",
        "field_time_scope"=>physical ? "instantaneous final state" : "final accepted steady iterate",
        "time_integration_scope"=>physical ? "Native shared six-equation conservative SSPRK3 with one global timestep for both blocks; timestep refinement and unsteady validation require separate evidence" :
            "Native shared six-equation conservative SSPRK3 with unequal local pseudo-clocks; no unsteady stall claim",
        "requested_duration_ctu"=>settings.duration_ctu,"burnin_ctu"=>settings.burnin_ctu,"block_ctu"=>settings.block_ctu,
        "time_statistics"=>time_stats,"physical_duration_reached"=>physical_duration_reached,
        "physical_time_refined"=>Bool(get(cfg,"physical_time_refined",false)),
        "physical_clock_scope"=>physical ? "sum of the accepted global timesteps of the pair, multiplied by U_inf/chord" : "no physical clock",
        "physical_dt_ctu_min"=>physical ? dt_ctu_min : 0.,"physical_dt_ctu_max"=>physical ? dt_ctu_max : 0.,"global_time_step"=>physical,
        # The transition treatment actually applied: the trip stations (0: none on that side) and what they did on this mesh.
        "sst_model"=>tripped ? "SST2003m with prescribed trips: both productions multiplied by the trip intermittency (0 ahead of the station, linear ramp over 0.02 c, then 1)" :
            "SST2003m, fully turbulent; no imposed transition trip",
        "transition_model"=>tripped ? "prescribed_trips" : "fully_turbulent",
        "trip_upper"=>settings.trip_upper,"trip_lower"=>settings.trip_lower,"trip_ramp_chord"=>tripped ? .02 : 0.,"trip_faces"=>trip_faces,
        "requested_trip_upper"=>Float64(get(cfg,"requested_trip_upper",settings.trip_upper)),"requested_trip_lower"=>Float64(get(cfg,"requested_trip_lower",settings.trip_lower)),"sfd_engaged"=>false,
        "rk_weight_closure"=>true,"rk_weight_closure_implementation"=>"native conservative SSPRK3 gas/SST stage construction; not the old single-block correction switch",
        "sst_wall_metric_correction"=>true,"sst_wall_metric_implementation"=>"native conservative SST face-wall distance; no legacy single-block switch",
        "viscosity_model"=>settings.viscosity,"freestream_temperature_k"=>settings.temperature_k,
        "freestream_temperature_used_for_viscosity"=>settings.viscosity=="sutherland",
        "sutherland_constant_k"=>settings.viscosity=="sutherland" ? 110.4 : 0.,
        "viscosity_scope"=>settings.viscosity=="constant_kinematic" ? "constant kinematic viscosity U_inf c/Re in the gas and SST equations" :
            "molecular viscosity mu = rho_inf nu_inf f(T) in the gas stresses and heat flux, the SST diffusion, blending functions and wall omega, and both clocks; f = 1 for constant dynamic",
        "reconstruction"=>1,"full_viscous_stress"=>true,"irsj"=>settings.irsj,
        "rans_acceleration"=>Bool(get(cfg,"rans_acceleration",false)),
        "rans_acceleration_method"=>String(get(cfg,"rans_acceleration_method","standard")),
        "mach"=>settings.mach,"reynolds"=>settings.reynolds,"alpha_deg"=>settings.alpha,"cfl"=>settings.cfl,
        "geometry"=>metadata,"mesh_cells"=>field.total,"mesh_sha256"=>field.meshsha,
        "warm_start_source"=>warm_source,
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
    physical && (meta["physical_force_csv"]="physical_force_history.csv")
    open(joinpath(outdir,"result.toml"),"w") do io;TOML.print(io,meta);end
    if !physical && status in ("step_limit", "wall_limit", "equation_and_force_guard_passed", "stalled_stop")
        _rans_save_restart(cfg,joinpath(outdir,"mesh_full.msgpack"),"conformal_C_H_v1",
            [(name="C",solver=c,extras=(rk=coupled.c.turbulence.rk,rw=coupled.c.turbulence.rw)),
             (name="H",solver=h,extras=(rk=coupled.h.turbulence.rk,rw=coupled.h.turbulence.rw))])
    end
    println("C+H worker complete: n=",accepted_steps()," status=",status,"; actual two-block fields and diagnostics saved");flush(stdout)
    nothing
end
