# Worker of the Implicit Newton (CPU) steady method (flag --newton-worker). OPT-IN, a development result, not validated.
# Included at CFDBackend module scope after the solver modules; the solver functions it calls are the public
# pseudo-transient Newton-Krylov march (src/compressible/cmesh_steady_newton_ka.jl, one block) and its two-block form
# (cmesh_steady_newton_pair_ka.jl). CPU, Float64, cold start with mesh sequencing; no checkpoint is read or written.
#
# Solver set-up: that of the benchmark drivers of 13-14 Nov 2026 (thermal MUSCL with the chord-scaled smooth limiter,
# full viscous terms, SST with Tu 0.1 % and an eddy-viscosity ratio 0.1, far-field point vortex relaxed towards the lift,
# levels 4-2-1, CFL ramp to 1000, GMRES to 0.3 with 60 vectors).
#
# PROGRESS. `MESH_READY,mesh_preview.msgpack` after the mesh files are written, then one line per diagnostic row of
# the march (every second step, the first and the last step of a level):
#     ITERATION,n,total,residual_scaled,residual_linf,residual_l2,cl,cd,mass_net_frac,cm
#   n                 one counter over the three mesh levels: the steps of the finished levels plus the step of this one
#   total             the steps of the finished levels plus the step budgets of this and the remaining levels
#   residual_scaled   RMS of the continuity residual per volume, divided by the freestream density (the plotted curve)
#   residual_l2       the largest of the four flow-equation RMS residuals (mass, x and y momentum, energy), each
#                     divided by its freestream scale
#   residual_linf     the larger of the two turbulence-equation RMS residuals (k, omega), unscaled. The march computes
#                     no maximum norm; this slot carries the turbulence residual instead.
#   cl, cd            the march's loads (wall pressure and friction; with two blocks the base is included)
#   mass_net_frac     the march's mass imbalance |sum of the continuity residual| / (rho_inf U_inf c); it has no sign
#   cm                moment about the quarter chord from the same wall pressure and friction, nose-up positive
# All six residuals of every row, with the level, the CFL and the linear iterations, are in newton_history.csv.
#
# STATUS. The march stops a level when CL and CD have stayed within a relative tolerance over a window of steps. The
# result is marked settled (convergence_guard.status = "early_stop_eligible", so the point comes out "completed") only
# if that happened on the fine level; otherwise the status is "loads_not_settled" and the point is "not_converged".
# The six equation residuals are recorded, without qualified limits; they do not enter the status.

const NEWTON_EQUATIONS=["mass","momentum_x","momentum_y","energy","k","omega"]

"The three residual slots of the ITERATION line from the march's six RMS residuals (see the file header)."
_newton_residual_slots(six)=(scaled=Float64(six[1]),l2=Float64(maximum(six[1:4])),linf=Float64(maximum(six[5:6])))

function _newton_iteration_line(n::Integer,total::Integer,six,cl,cd,mass,cm)
    r=_newton_residual_slots(six)
    @sprintf("ITERATION,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g",n,total,r.scaled,r.linf,r.l2,cl,cd,mass,cm)
end

"""The stop guard of a Newton point. `settled` is the march's own report for the fine level (loads_stable); nothing
else can make a point eligible."""
function _newton_guard(settled::Bool;steps::Integer,budget::Integer,load_tol::Real,window::Integer)
    criterion="CL and CD spans over the last $window Newton steps below $load_tol, relative to max(|CL|, 0.1) and max(|CD|, 0.001), on the fine mesh level"
    Dict{String,Any}(
        "status"=>settled ? "early_stop_eligible" : "loads_not_settled",
        "early_stop_eligible"=>settled,
        "reason"=>settled ?
            "CL and CD settled on the fine mesh after $steps steps ($criterion). The six equation residuals are recorded without qualified limits." :
            "CL and CD did not settle within the $budget fine-level steps allowed ($criterion). Fields and coefficients are a saved diagnostic state.",
        "termination"=>settled ? "loads_settled" : "iteration_limit",
        "stopped_early"=>settled && steps<budget,"auto_stop_enabled"=>true,
        "criterion"=>criterion,
        "scope"=>"load history of the fine mesh level only; not equation convergence, mesh independence or validation")
end

"Cell (i, k) of the strip in the solver's base-as-wall layout that is cell (k, j) of the app's strip layout (NS cross cells)."
_newton_strip_cell(NS::Integer,k::Integer,j::Integer)=(NS+1-j,k)

"A prepared Float64 CPU block on the node set `nd`. `trip = false`: no trip (the strip behind the base)."
function _newton_block(M,nd,cfg;trip::Bool=true)
    T=Float64
    met=trip ?
        CMeshMetrics.build_metrics(nd.X,nd.Y,nd.NIW,nd.NIF;trip_upper=Float64(cfg["trip_upper"]),
            trip_lower=Float64(cfg["trip_lower"]),trip_side=:wall_order,scalar_type=T) :
        CMeshMetrics.build_metrics(nd.X,nd.Y,nd.NIW,nd.NIF;trip_upper=0.,trip_lower=0.,trip_side=:y_sign,scalar_type=T)
    cx,cy,cz,qw,_,_=build_d3q39_static(T)
    s=M.build_sol_ka(KernelAbstractions.CPU(),met;drot=T(2),cxv=cx,cyv=cy,czv=cz,wv=qw,scalar_type=T)
    s.wall_flux=true;s.fullvisc=Int32(1)
    M.set_airfoil_muscl_ka!(s,nd.X,nd.Y;limiter=Symbol(cfg["limiter"]),
        smooth_coefficient=Float32(cfg["limiter_smooth_coefficient"]))
    M.set_freestream_ka!(s,Float64(cfg["mach"]),Float64(cfg["alpha_deg"]))
    s.nu_lam=hypot(s.uxi,s.uyi)/Float64(cfg["reynolds"])
    model=String(cfg["viscosity_model"])
    model=="sutherland" ? M.set_air_sutherland_ka!(s,Float64(cfg["freestream_temperature_k"])) :
    model=="constant_dynamic" ? M.set_air_constant_dynamic_ka!(s) :
    model=="constant_kinematic" ? nothing : throw(ArgumentError("unknown viscosity model $model"))
    M.set_sst_ka!(s;tu_inf=Float64(cfg["tu_inf"]),mu_t_ratio=Float64(cfg["mu_t_ratio"]))
    M.applyBC_ka!(s,Int32(2));M.prepare_dt_ka!(s,.45f0)
    M.sst_step_ka!(s;corrected_wall_metric=true)
    s
end

"""Wall faces of a block (nodes X, Y): position, pressure coefficient and the pressure and friction forces on the
body, with the arithmetic of the solver's own load integral (loads_ka: no-slip gas wall, full viscous terms)."""
function _newton_wall(M,s,X,Y)
    g=s.g;faces=s.NIW+1:s.NIW+s.NIF;gr=(first(faces)+g):(last(faces)+g)
    rw,uw,vw,tw=map(a->Float64.(Array(view(a,gr,g+1))),(s.rho,s.ux,s.uy,s.T))
    cs2=Float64(M._gas_cs2(eltype(s.rho)))
    speed=hypot(Float64(s.uxi),Float64(s.uyi));q=.5*Float64(s.rho_inf)*speed^2
    pinf=cs2*Float64(s.rho_inf)*Float64(s.T_inf)
    n=length(faces);x=zeros(n);y=zeros(n);cp=zeros(n)
    fxp=zeros(n);fyp=zeros(n);fxf=zeros(n);fyf=zeros(n);moment=0.
    for (k,i) in enumerate(faces)
        sx=Float64(s.Sjx1_h[i]);sy=Float64(s.Sjy1_h[i]);sf=hypot(sx,sy)
        pressure=rw[k]*cs2*tw[k]
        d1=.5*Float64(s.vol1_h[i])/sf+1e-30
        mu=(s.air_sutherland!=Int32(0) ? Float64(s.rho_inf)*Float64(s.nu_lam) : rw[k]*Float64(s.nu_lam))*
           Float64(M._mufac(tw[k],s.suth))*sf/d1
        tx,ty=M._wall_stokes_traction(uw[k],vw[k],sx/sf,sy/sf,mu)
        x[k]=.5*(X[i,1]+X[i+1,1]);y[k]=.5*(Y[i,1]+Y[i+1,1]);cp[k]=(pressure-pinf)/q
        fxp[k]=-pressure*sx;fyp[k]=-pressure*sy;fxf[k]=Float64(tx);fyf[k]=Float64(ty)
        moment+=_aerodynamic_cm(x[k],y[k],fxp[k]+fxf[k],fyp[k]+fyf[k],q)
    end
    ca,sa=Float64(s.ca),Float64(s.sa)
    coefficients(fx,fy)=((fy*ca-fx*sa)/q,(fx*ca+fy*sa)/q)
    clp,cdp=coefficients(sum(fxp),sum(fyp));clf,cdf=coefficients(sum(fxf),sum(fyf))
    # excess pressure (relative to the freestream) per face: the region split of the result file
    excess(k)=(-(cp[k]*q)*Float64(s.Sjx1_h[faces[k]]),-(cp[k]*q)*Float64(s.Sjy1_h[faces[k]]))
    (;faces,x,y,cp,fxp,fyp,fxf,fyf,cm=moment,cl_pressure=clp,cd_pressure=cdp,cl_friction=clf,cd_friction=cdf,
      cl=clp+clf,cd=cdp+cdf,q,ca,sa,coefficients,excess)
end

"`at(i, j)`: the values of FullFieldArtifact.COLUMNS at cell (i, j) of a block, vorticity from the cell centres."
function _newton_cell_fields(M,s)
    g=s.g;ni=s.NI;nj=s.NJ;ir=g+1:g+ni;jr=g+1:g+nj
    rho,u,v,t,nut=map(a->Float64.(Array(view(a,ir,jr))),(s.rho,s.ux,s.uy,s.T,s.nut))
    xc,yc=map(a->Float64.(Array(a)),(s.xc,s.yc))
    cs2=Float64(M._gas_cs2(eltype(s.rho)));gamma=(5.0+Float64(s.drot))/(3.0+Float64(s.drot))
    pinf=cs2*Float64(s.rho_inf)*Float64(s.T_inf)
    q=.5*Float64(s.rho_inf)*hypot(Float64(s.uxi),Float64(s.uyi))^2
    function vorticity(i,j)
        im=max(1,i-1);ip=min(ni,i+1);jm=max(1,j-1);jp=min(nj,j+1)
        xi=xc[ip,j]-xc[im,j];yi=yc[ip,j]-yc[im,j];xj=xc[i,jp]-xc[i,jm];yj=yc[i,jp]-yc[i,jm]
        ui=u[ip,j]-u[im,j];uj=u[i,jp]-u[i,jm];vi=v[ip,j]-v[im,j];vj=v[i,jp]-v[i,jm];jac=xi*yj-xj*yi
        abs(jac)<1e-20 ? 0. : ((vi*yj-vj*yi)-(-ui*xj+uj*xi))/jac
    end
    at(i,j)=(xc[i,j],yc[i,j],hypot(u[i,j],v[i,j])/sqrt(gamma*cs2*t[i,j]),(rho[i,j]*cs2*t[i,j]-pinf)/q,
        rho[i,j]*cs2*t[i,j],rho[i,j],u[i,j],v[i,j],t[i,j],vorticity(i,j),nut[i,j])
    minimum(t)>0 && all(isfinite,rho) && all(isfinite,u) && all(isfinite,v) ||
        error("the Newton march returned a nonfinite or nonpositive flow state")
    at
end

function _newton_worker(cfg_path::AbstractString)
    cfg=TOML.parsefile(cfg_path);started=time()
    String(get(cfg,"rans_method",""))==NEWTON_METHOD || throw(ArgumentError("the Newton worker runs rans_method = newton only"))
    Bool(cfg["gpu"]) && throw(ArgumentError("The Implicit Newton method runs on the CPU in Float64; gpu must be false"))
    String(get(cfg,"time_mode","steady"))=="steady" || throw(ArgumentError("The Implicit Newton method is a steady method"))
    M=CMeshSolverKA
    outdir=String(cfg["outdir"]);mkpath(outdir)
    resolution=String(get(cfg,"resolution","custom"))
    re=Float64(cfg["reynolds"]);mach=Float64(cfg["mach"]);alpha=Float64(cfg["alpha_deg"])
    levels=Tuple(Int.(cfg["newton_levels"]));nsteps=Tuple(Int.(cfg["newton_nsteps"]))
    cflmax=Tuple(Float64.(cfg["newton_cflmax"]));load_tol=Tuple(Float64.(cfg["newton_load_tol"]))
    window=Tuple(Int.(cfg["newton_window"]))
    length(levels)==length(nsteps)==length(cflmax)==length(load_tol)==length(window) && last(levels)==1 ||
        throw(ArgumentError("the Newton level settings must have one entry per level and end on the fine mesh"))
    viscosity_model=String(cfg["viscosity_model"])
    println("NEWTON: Implicit Newton (CPU, Float64), $(Threads.nthreads()) threads; development method, not validated");flush(stdout)

    # ---- mesh: the mesh Create mesh shows -----------------------------------------------------------------
    af=haskey(cfg,"naca4_code") ? CMeshGen.naca4_points(cfg["naca4_code"]) : CMeshGen.read_airfoil_dat(cfg["geometry"])
    tmesh=time()
    built=_build_newton_mesh(af;reynolds=re,alpha_deg=alpha,farfield_angle_deg=Float64(cfg["cfd_farfield_angle_deg"]),
        cfd_te_thickness=Float64(get(cfg,"cfd_te_thickness",0.0)),n_foil=Int(cfg["n_foil"]),n_wake=Int(cfg["n_wake"]),
        n_normal=Int(cfg["n_normal"]),te_ds=Float64(cfg["te_ds"]),wake_aspect=Float64(cfg["wake_aspect"]))
    mesh=built.mesh;metadata=built.metadata;two=mesh.strip!==nothing
    (String(get(cfg,"newton_topology",two ? "c_plus_h" : "c"))=="c_plus_h")==two ||
        error("the marched mesh has $(two ? "two blocks" : "one block"), the request expected the other topology")
    FullFieldArtifact.field_layout(mesh.NI,mesh.NJ)       # impossible export sizes fail before the march
    two && FullFieldArtifact.field_layout(mesh.strip.NI,mesh.strip.NJ)
    metadata["geometry_support"]=_newton_geometry_support(haskey(cfg,"naca4_code"))
    _newton_write_mesh_artifacts(mesh,metadata,outdir,resolution)
    println("MESH_READY,mesh_preview.msgpack");flush(stdout)
    tmesh=time()-tmesh
    cells=metadata["cells"]
    @printf("NEWTON mesh: marched, %s; C %d x %d%s = %d cells; dn1=%.3e, estimated y+=%.2f (%.1f s)\n",
        two ? "two blocks (C+H)" : "one block",mesh.NI,mesh.NJ,
        two ? @sprintf(" + strip %d x %d",mesh.strip.NI,mesh.strip.NJ) : "",cells,mesh.dn1,metadata["estimated_yplus"],tmesh)
    @printf("NEWTON case: Mach %.4g, Re %.4g, alpha %.4g deg; trips %s; viscosity %s; fine-level budget %d steps\n",
        mach,re,alpha,Bool(get(cfg,"fully_turbulent",false)) ? "none (fully turbulent)" :
            @sprintf("upper %.3f, lower %.3f",cfg["trip_upper"],cfg["trip_lower"]),viscosity_model,nsteps[end])
    haskey(cfg,"warm_start_requested") &&
        println("NEWTON: a checkpoint of the previous angle was offered; this method starts cold and does not read it")
    flush(stdout)

    # ---- march, with one progress row per diagnostic row --------------------------------------------------
    history=NamedTuple[]
    history_io=open(joinpath(outdir,"iteration_history.csv"),"w")
    newton_io=open(joinpath(outdir,"newton_history.csv"),"w")
    println(history_io,"iteration,residual_linf,residual_scaled,residual_l2,cl,cd,cm,mass_net_frac")
    println(newton_io,"iteration,level_coarsening,level_step,seconds,cfl,linear_iterations,",
        join(NEWTON_EQUATIONS.*"_l2",','),",dk,dw,mass,cl,cd,cm")
    flush(history_io);flush(newton_io)
    current=Ref{Any}(nothing)                 # the blocks of the level being marched: (s, X, Y) each
    level=Ref(0);offset=Ref(0);last_step=Ref(0);rows=Ref(0)
    function on_row(k,f,row)
        if k!=level[]
            offset[]+=last_step[];level[]=k;last_step[]=0;rows[]=0
            @printf("NEWTON level %d of %d (node spacing %d of the fine mesh): %d cells\n",k,length(levels),f,
                sum(b.s.NI*b.s.NJ for b in current[]));flush(stdout)
        end
        last_step[]=row.step;rows[]+=1
        n=offset[]+row.step;total=offset[]+sum(nsteps[k:end])
        cm=sum(_newton_wall(M,b.s,b.X,b.Y).cm for b in current[])
        slots=_newton_residual_slots(row.six)
        all(isfinite,(slots...,row.cl,row.cd,row.mass,cm)) ||
            error("the Newton march produced nonfinite diagnostics at step $n")
        push!(history,(iteration=n,level=f,residual_scaled=slots.scaled,residual_l2=slots.l2,residual_linf=slots.linf,
            cl=row.cl,cd=row.cd,cm=cm,mass_net_frac=row.mass,six=Float64.(row.six)))
        @printf(history_io,"%d,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g\n",n,slots.linf,slots.scaled,slots.l2,row.cl,row.cd,cm,row.mass)
        println(newton_io,join((n,f,row.step,row.seconds,row.cfl,row.its,row.six...,row.dk,row.dw,row.mass,row.cl,row.cd,cm),','))
        flush(history_io);flush(newton_io)
        println(_newton_iteration_line(n,total,row.six,row.cl,row.cd,row.mass,cm))
        rows[]%10==1 && @printf("NEWTON step %d (level step %d): CFL %.0f, %d linear iterations, residuals %s, CL=%.6f CD=%.6f\n",
            n,row.step,row.cfl,row.its,string(round.(row.six;sigdigits=3)),row.cl,row.cd)
        flush(stdout)
    end
    marching=time()
    kwargs=(;levels,nsteps,cflmax,load_tol,window,lintol=Float64(cfg["newton_lintol"]),maxit=Int(cfg["newton_maxit"]),
        growth=Float64(cfg["newton_growth"]),every=Int(cfg["newton_every"]),
        vortex_relax=Float64(cfg["newton_vortex_relax"]),quiet=true,on_row)
    blocks=try
        if two
            pending=Ref{Any}(nothing)
            make_c(nd)=(s=_newton_block(M,nd,cfg);pending[]=(;s,X=nd.X,Y=nd.Y);s)
            make_h(nd)=(s=_newton_block(M,nd,cfg;trip=false);current[]=[pending[],(;s,X=nd.X,Y=nd.Y)];s)
            result=M.newton_pair_cold_run((X=mesh.X,Y=mesh.Y,NIW=mesh.NIW,NIF=mesh.NIF,NI=mesh.NI,NJ=mesh.NJ,strip=mesh.strip),
                make_c,make_h;kwargs...)
            (;run=result,c=result.p.c,h=result.p.h)
        else
            make(nd)=(s=_newton_block(M,nd,cfg);current[]=[(;s,X=nd.X,Y=nd.Y)];s)
            result=M.newton_cold_run((X=mesh.X,Y=mesh.Y,NIW=mesh.NIW,NIF=mesh.NIF,NI=mesh.NI,NJ=mesh.NJ),make;kwargs...)
            (;run=result,c=result.s,h=nothing)
        end
    finally
        close(history_io);close(newton_io)
    end
    marching=time()-marching
    summary=blocks.run.summary;c=blocks.c;h=blocks.h
    isempty(history) && error("the Newton march recorded no step")
    fine=summary["levels"][end]
    settled=fine["loads_stable"]===true
    total_steps=last(history).iteration
    total_steps==sum(Int(l["steps"]) for l in summary["levels"]) ||
        error("the progress counter ($total_steps) differs from the steps of the levels")
    for l in summary["levels"]
        @printf("NEWTON level 1/%d: %d cells, %d steps, loads settled=%s, %.1f s, CL=%.6f CD=%.6f\n",
            l["coarsening"],l["cells"],l["steps"],string(l["loads_stable"]),l["seconds"],l["CL"],l["CD"])
    end
    flush(stdout)

    # ---- loads, surface pressure, fields ------------------------------------------------------------------
    wall=_newton_wall(M,c,mesh.X,mesh.Y)
    hn=two ? M.newton_pair_strip_nodes(mesh.strip) : nothing
    base=two ? _newton_wall(M,h,hn.X,hn.Y) : nothing
    cl,cd=two ? M.newton_pair_loads(blocks.run.p) : Float64.(M.loads_ka(c))
    cl_wall=wall.cl+(two ? base.cl : 0.);cd_wall=wall.cd+(two ? base.cd : 0.)
    isfinite(cl) && isfinite(cd) || error("the Newton march returned nonfinite loads")
    abs(cl_wall-cl)<=1e-8*max(1.,abs(cl)) && abs(cd_wall-cd)<=1e-8*max(1.,abs(cd)) ||
        error("the wall-load decomposition ($cl_wall, $cd_wall) disagrees with the solver's loads ($cl, $cd)")
    cm=wall.cm+(two ? base.cm : 0.)
    ile=argmin(wall.x)                       # wall faces run lower trailing edge -> nose -> upper trailing edge
    cpfile=joinpath(outdir,"surface_cp.csv")
    open(cpfile,"w") do io
        println(io,"surface,x,cp")
        for k in eachindex(wall.x);println(io,k>ile ? "upper" : "lower",',',wall.x[k],',',wall.cp[k]);end
    end
    # excess-pressure contributions of the upper surface, the lower surface and the base (two blocks: the strip's wall)
    fxr=zeros(3);fyr=zeros(3);cmr=zeros(3)
    for k in eachindex(wall.x)
        r=k>ile ? 1 : 2;ex,ey=wall.excess(k);fxr[r]+=ex;fyr[r]+=ey;cmr[r]+=_aerodynamic_cm(wall.x[k],wall.y[k],ex,ey,wall.q)
    end
    if two
        for k in eachindex(base.x)
            ex,ey=base.excess(k);fxr[3]+=ex;fyr[3]+=ey;cmr[3]+=_aerodynamic_cm(base.x[k],base.y[k],ex,ey,base.q)
        end
    end
    region=[wall.coefficients(fxr[r],fyr[r]) for r in 1:3]
    cl_pressure=wall.cl_pressure+(two ? base.cl_pressure : 0.);cd_pressure=wall.cd_pressure+(two ? base.cd_pressure : 0.)
    cl_friction=wall.cl_friction+(two ? base.cl_friction : 0.);cd_friction=wall.cd_friction+(two ? base.cd_friction : 0.)
    @printf("NEWTON force split: Cl_pressure=%+.5f Cl_friction=%+.5f Cd_pressure=%+.6f Cd_friction=%+.6f; excess-pressure drag upper=%+.6f lower=%+.6f base=%+.6f\n",
        cl_pressure,cl_friction,cd_pressure,cd_friction,region[1][2],region[2][2],region[3][2]);flush(stdout)

    # continuity residual: where it peaks (both blocks), and the reduction from the freshly initialised fine level
    peak=(value=-Inf,block="C",i=0,j=0);near=0;ncell=0
    scaled_mass=[(id,abs.(Array(M._airfoil_residual(s)[1]))./Array(s.vol)) for (id,s) in (two ? (("C",c),("H",h)) : (("C",c),))]
    for (id,a) in scaled_mass
        k=argmax(a);a[k]>peak.value && (peak=(value=a[k],block=id,i=k[1],j=k[2]))
    end
    for (_,a) in scaled_mass;near+=count(>=(0.99*peak.value),a);ncell+=length(a);end
    six=Float64.(summary["final_six_residuals"]);over_cold=Float64.(summary["final_six_over_cold"])
    limited=sum(count(<(0.99),Array(M._AIRFOIL_MUSCL[s].theta)[1:s.NI,1:s.NJ]) for s in (two ? (c,h) : (c,)))

    files=Dict{String,Any}()
    if two
        NS=mesh.strip.NJ
        open(joinpath(outdir,"wall_force_faces.csv"),"w") do io
            println(io,"block,index,x,y,fx,fy")
            for (k,i) in enumerate(wall.faces)
                println(io,join(("C",i,wall.x[k],wall.y[k],wall.fxp[k]+wall.fxf[k],wall.fyp[k]+wall.fyf[k]),','))
            end
            # the base faces in the app's strip order: face j across the strip from the lower leg
            for j in 1:NS
                k=_newton_strip_cell(NS,1,j)[1]
                println(io,join(("H",j,base.x[k],base.y[k],base.fxp[k]+base.fxf[k],base.fyp[k]+base.fyf[k]),','))
            end
        end
        atc=_newton_cell_fields(M,c);ath=_newton_cell_fields(M,h)
        # the strip's cells are written in the app's layout (i downstream, j across from the lower leg)
        field=_write_block_fields([(id="C",ni=c.NI,nj=c.NJ,source=()->atc),
            (id="H",ni=mesh.strip.NI,nj=NS,source=()->((k,j)->ath(_newton_strip_cell(NS,k,j)...)))],outdir,total_steps)
        merge!(files,Dict("fields_blocks_msgpack"=>"fields_blocks.msgpack","field_full_msgpack"=>"fields_blocks_full.msgpack",
            "full_field_cells"=>field.total,"full_field_mesh_sha256"=>field.meshsha,"full_field_bytes"=>field.bytes,
            "mesh_sha256"=>field.meshsha,"wall_force_csv"=>"wall_force_faces.csv",
            "force_q_inf"=>wall.q,"force_ca"=>wall.ca,"force_sa"=>wall.sa,
            "mass_residual_peak_block"=>peak.block))
        field.total==cells || error("the block fields hold $(field.total) cells, the mesh $cells")
    else
        at=_newton_cell_fields(M,c);ni=c.NI;nj=c.NJ
        # the sampled preview of the single-block points (cfd_backend.jl): about 18000 cells, last row and column kept
        stride=max(1,ceil(Int,sqrt(ni*nj/18000)))
        sample_i=sort!(unique(vcat(collect(1:stride:ni),[ni])));sample_j=sort!(unique(vcat(collect(1:stride:nj),[nj])))
        open(joinpath(outdir,"field_samples.csv"),"w") do io
            println(io,"x,y,mach,cp,pressure,density,u,v,temperature,vorticity,nut")
            for j in sample_j,i in sample_i;println(io,join(at(i,j),','));end
        end
        full=FullFieldArtifact.write_full_field(at,joinpath(outdir,"field_full.msgpack"),ni,nj,
            joinpath(outdir,"mesh_full.msgpack"),total_steps)
        merge!(files,Dict("field_csv"=>"field_samples.csv","field_full_msgpack"=>full["filename"],
            "full_field_cells"=>full["cell_count"],"full_field_mesh_sha256"=>full["mesh_sha256"],
            "full_field_bytes"=>full["bytes"],"sample_stride"=>stride,
            "sampled_ni"=>length(sample_i),"sampled_nj"=>length(sample_j)))
    end
    @printf("NEWTON fields saved: %d cells, final n=%d\n",cells,total_steps);flush(stdout)

    # ---- result -------------------------------------------------------------------------------------------
    guard=_newton_guard(settled;steps=Int(fine["steps"]),budget=nsteps[end],load_tol=load_tol[end],window=window[end])
    tail=history[max(1,end-window[end]÷Int(cfg["newton_every"])):end]
    cl_span=maximum(r.cl for r in tail)-minimum(r.cl for r in tail)
    cd_span=maximum(r.cd for r in tail)-minimum(r.cd for r in tail)
    cl_limit=load_tol[end]*max(abs(cl),0.1);cd_limit=load_tol[end]*max(abs(cd),0.001)
    physical_sanity=cd>=0.0 && cd<=1.0 && abs(cl)<=3.0
    cold_start="Cold start with mesh sequencing at every angle; no checkpoint is read or written."*
        (haskey(cfg,"warm_start_requested") ? " A checkpoint of the previous angle was offered and not used." : "")
    kinematic_note=viscosity_model=="constant_kinematic" ?
        " Constant kinematic viscosity is covered by the method's small test cases only; the benchmark runs used Sutherland or constant dynamic viscosity." : ""
    notes=String[
        "Implicit Newton (CPU): pseudo-transient Newton-Krylov march in Float64 with SST, on the marched regular mesh ($(two ? "two blocks, C+H: the finite trailing-edge base is a wall of the strip block and is included in the forces" : "one C block, closed trailing edge")), mesh sequencing over $(length(levels)) levels. Development result, not validated."*kinematic_note,
        metadata["transform"]=="none" ? "The CFD geometry preserves the source trailing edge." :
            "CFD geometry change: $(metadata["transform"]). XFoil uses the source section; account for the recorded geometry difference when comparing results.",
        Bool(get(cfg,"fully_turbulent",false)) ? "Fully turbulent SST on the whole surface." :
            "Forced trips on the surface at x/c $(cfg["trip_upper"]) (upper) and $(cfg["trip_lower"]) (lower)"*(two ? "; the strip behind the base has no trip." : "."),
        cold_start,
        settled ? "CL and CD settled on the fine mesh (the method's stopping rule). That is a load-history statement: the six equation residuals are recorded without qualified limits, and mesh independence is not shown." :
            "CL and CD did not settle within the fine-level step budget: this point is not converged. The saved fields and coefficients are a diagnostic state.",
        physical_sanity ? "Loose force bounds passed; compare Cp and loads with XFoil/reference data." :
            "Loose force bounds failed; do not use these coefficients as a polar datum."]
    meta=Dict{String,Any}("cl"=>cl,"cd"=>cd,"cm"=>cm,"steps"=>total_steps,"max_steps"=>sum(nsteps),
        # The status of this point is the march's own load rule, see the file header.
        "rans_method"=>NEWTON_METHOD,
        "rans_acceleration"=>true,"rans_acceleration_method"=>String(get(cfg,"rans_acceleration_method","newton")),
        "irsj"=>false,"warm_start_source"=>"",
        "warm_start_requested"=>String(get(cfg,"warm_start_requested","")),"start"=>cold_start,
        "polar_mesh_anchor_deg"=>get(cfg,"polar_mesh_anchor_deg",cfg["cfd_farfield_angle_deg"]),
        "time_mode"=>"steady","clock"=>"steady pseudo-time (implicit Newton march, local CFL ramp to $(Int(cflmax[end])))",
        "elapsed_ctu"=>0.0,"field_time_scope"=>"final steady iterate",
        "convergence_guard"=>guard,"numerically_stationary"=>settled,"aerodynamic_converged"=>settled,
        "steady_assessment"=>Dict{String,Any}("status"=>guard["status"],"criteria_passed"=>settled,
            "reason"=>guard["reason"],"scope"=>guard["scope"],"gates"=>Dict("loads_stable"=>settled)),
        "convergence_summary"=>settled ?
            "CL and CD settled on the fine mesh (relative span below $(load_tol[end]) over $(window[end]) Newton steps); the six equation residuals are recorded without qualified limits; mesh independence and physical validation remain unverified" :
            "not converged: "*guard["reason"],
        "validation_summary"=>"development result, not validated (Implicit Newton, marched mesh)",
        "point_notes"=>notes,
        "cl_pressure"=>cl_pressure,"cl_friction"=>cl_friction,"cd_pressure"=>cd_pressure,"cd_friction"=>cd_friction,
        "cl_pressure_upper"=>region[1][1],"cl_pressure_lower"=>region[2][1],"cl_pressure_base"=>region[3][1],
        "cd_pressure_upper"=>region[1][2],"cd_pressure_lower"=>region[2][2],"cd_pressure_base"=>region[3][2],
        "cm_pressure_upper"=>cmr[1],"cm_pressure_lower"=>cmr[2],"cm_pressure_base"=>cmr[3],
        "pressure_region_scope"=>"excess wall pressure relative to freestream; lower, upper and finite-TE base",
        "residual_scaled"=>last(history).residual_scaled,"residual_l2"=>last(history).residual_l2,
        "residual_linf"=>last(history).residual_linf,
        "residual_scope"=>"Newton march: residual_scaled is the RMS continuity residual (freestream-scaled), residual_l2 the largest of the four flow-equation RMS residuals, residual_linf the larger of the k and omega equation RMS residuals; all six are in newton_history.csv",
        "residual_ratio"=>over_cold[1],"residual_reference_iteration"=>0,
        "residual_reference_scope"=>"continuity RMS residual relative to the freshly initialised fine level (uniform freestream)",
        "mass_net_frac"=>Float64(summary["final_mass_imbalance"]),
        "mass_net_scope"=>"|sum of the final continuity residual| / (rho_inf*U_inf*chord), both blocks; unsigned",
        "mass_residual_peak_cell"=>[peak.i,peak.j],"mass_residual_fraction_near_peak"=>near/ncell,
        "mesh_cells"=>cells,
        "gpu"=>false,"precision"=>"float64","reconstruction"=>Int(c.recon),
        "reconstruction_scope"=>"second-order thermal MUSCL, limiter $(cfg["limiter"]) with smoothing coefficient $(cfg["limiter_smooth_coefficient"])",
        "full_viscous_stress"=>c.fullvisc==Int32(1),"sfd_engaged"=>false,"sfd_start"=>0,
        "rk_weight_closure"=>false,"sst_wall_metric_correction"=>true,
        "residual_sampling_state"=>"final_accepted_newton_state",
        "viscosity_model"=>viscosity_model,"freestream_temperature_k"=>Float64(cfg["freestream_temperature_k"]),
        "freestream_temperature_used_for_viscosity"=>viscosity_model=="sutherland",
        "estimated_yplus"=>metadata["estimated_yplus"],"mesh_mode"=>NEWTON_MESH,
        "geometry"=>metadata,"trip_upper"=>cfg["trip_upper"],"trip_lower"=>cfg["trip_lower"],
        "mach"=>mach,"reynolds"=>re,"alpha_deg"=>alpha,
        "stationarity_scope"=>"Newton march: CL and CD spans over the last $(window[end]) fine-level steps, relative to max(|CL|, 0.1) and max(|CD|, 0.001), below $(load_tol[end]); there is no separate mean-shift test (the mean limits repeat the span limits); not equation convergence",
        "stationarity_window_iterations"=>window[end],
        "stationarity_cl_span_limit"=>cl_limit,"stationarity_cd_span_limit"=>cd_limit,
        "stationarity_cl_mean_limit"=>cl_limit,"stationarity_cd_mean_limit"=>cd_limit,
        "load_drift"=>cl_span,"drag_drift"=>cd_span,
        "load_drift_scope"=>"span over the recorded rows of the last $(window[end]) steps (a row every $(cfg["newton_every"]) steps)",
        "physical_sanity"=>physical_sanity,"physical_sanity_scope"=>"loose force bounds only; not experimental validation",
        "cp_csv"=>basename(cpfile),"iteration_csv"=>"iteration_history.csv","newton_history_csv"=>"newton_history.csv",
        "newton"=>Dict{String,Any}(
            "blocks"=>two ? 2 : 1,"fine_loads_stable"=>settled,
            "levels"=>[Dict{String,Any}("coarsening"=>Int(l["coarsening"]),"cells"=>Int(l["cells"]),"steps"=>Int(l["steps"]),
                "step_budget"=>nsteps[k],"loads_stable"=>l["loads_stable"]===true,"extrapolations"=>Int(l["extrapolations"]),
                "seconds"=>Float64(l["seconds"]),"cl"=>Float64(l["CL"]),"cd"=>Float64(l["CD"]),
                "cold_reference_six"=>Float64.(l["cold_reference_six"])) for (k,l) in enumerate(summary["levels"])],
            "equation_names"=>NEWTON_EQUATIONS,"final_six_residuals"=>six,"final_six_over_cold"=>over_cold,
            "six_scope"=>"RMS residuals per volume; the four flow equations divided by their freestream scales, k and omega unscaled; over_cold is relative to the freshly initialised fine level",
            "final_mass_imbalance"=>Float64(summary["final_mass_imbalance"]),
            "vortex_circulation"=>Float64(c.gam),"limiter_cells_below_0.99"=>limited,
            "solve_seconds"=>Float64(summary["solve_seconds"]),"march_wall_seconds"=>marching,"mesh_seconds"=>tmesh,
            "threads"=>Threads.nthreads(),
            "settings"=>Dict{String,Any}(k=>cfg[k] for k in ("newton_levels","newton_nsteps","newton_cflmax","newton_load_tol",
                "newton_window","newton_lintol","newton_maxit","newton_growth","newton_every","newton_vortex_relax",
                "limiter","limiter_smooth_coefficient","tu_inf","mu_t_ratio","te_ds","wake_aspect","requested_cfd_steps")),
            "source_sha256"=>get(cfg,"numerical_source_sha256",Dict{String,Any}())),
        "elapsed_wall_seconds"=>time()-started)
    if two
        meta["base_cl"]=base.cl;meta["base_cd"]=base.cd;meta["mesh_topology"]="marched_C_H"
        meta["base_load_scope"]="the strip's wall (the base) with the absolute wall pressure, as in the solver's load integral: the ambient pressure on the base cancels against the rest of the contour. The base drag relative to the freestream pressure is cd_pressure_base."
    end
    merge!(meta,files)
    open(joinpath(outdir,"result.toml"),"w") do io;TOML.print(io,meta);end
    @printf("NEWTON final: Cl=%+.6f Cd=%+.6f Cm=%+.6f; %d steps (%s); loads settled on the fine level=%s; residuals/cold %s; %.1f s\n",
        cl,cd,cm,total_steps,join((string(l["steps"]) for l in summary["levels"]),"+"),string(settled),
        string(round.(over_cold;sigdigits=3)),time()-started)
    println("NEWTON worker complete")
    nothing
end
