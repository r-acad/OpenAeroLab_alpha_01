# Pseudo-transient Newton-Krylov march for the steady 2D airfoil (single-block C or A+ mesh, plain SST).
# EXPERIMENTAL AND OPT-IN: no default solver path, worker or app setting calls it; the explicit and the existing
# implicit methods are unchanged. Included into module CMeshSolverKA.
#
# One coupled iteration (newton_rans_step!):
#   1. flow: a Newton step on the true residual (second-order kinetic flux, full viscous terms) at a local
#      pseudo-time step of CFL m.cfl, the Jacobian applied by forward finite differences, GMRES to a loose
#      tolerance, a physical step limit (a fraction of density and energy), no residual-norm line search;
#      preconditioner: a two-level cycle on the assembled first-order block matrix, rebuilt when the CFL changes
#      or every `lag` steps (cmesh_steady_newton_linear_ka.jl);
#   2. turbulence: one implicit SST step at the same local pseudo-time (cmesh_steady_newton_sst_ka.jl).
# newton_rans_march! ramps the CFL geometrically, applies reduced-rank extrapolation once the ramp is over and
# stops when CL and CD have settled. newton_cold_run adds mesh sequencing by node skipping.
#
# REQUIREMENTS AND LIMITS. Float64 state on the CPU (GMRES and the coarse solve are host code). The MUSCL
# limiter must be smooth: the cell-size-scaled Venkatakrishnan constant (limiter = :venkat) is switched by
# round-off-level variations and makes the linearisation useless. Subsonic cases run without a limiter
# (newton_unlimited_reconstruction!(s)); cases with shocks use set_airfoil_muscl_ka!(...; limiter = :venkat_chord),
# whose smoothing scale is a fixed fraction of the freestream magnitudes. Its coefficient matters: 0.05 also limits
# smooth boundary-layer profiles slightly (RAE 2822 Case 9: 13 % of the cells, drag 3.7 % higher), 0.2 acts at the
# shock only. newton_rans_march!(...; vortex_relax) updates the far-field point vortex from the lift.
# Development results only: not validated against reference data.
import Printf
include(joinpath(@__DIR__, "cmesh_steady_newton_linear_ka.jl"))
include(joinpath(@__DIR__, "cmesh_steady_newton_sst_ka.jl"))

"Make the airfoil MUSCL reconstruction effectively unlimited (Venkatakrishnan smoothing coefficient `kappa`); subsonic flow only."
function newton_unlimited_reconstruction!(s::KSol;kappa=1000)
    w=_AIRFOIL_MUSCL[s]
    _AIRFOIL_MUSCL[s]=AirfoilMUSCLWorkspace(w.geometry,w.states,w.theta,w.status,w.limiter,Float32(kappa),w.previous_recon)
    s
end

mutable struct NewtonRANS
    kw;L;coarse;sst;scalar;d0;invdt;backup;base
    cfl::Float64;it::Int;rows::Vector{Any};since_build::Int;built_cfl::Float64
end
function newton_rans_init(s::KSol;cfl=20.,agg=(4,4))
    kw=build_airfoil_kinetic_block_workspace(s;block_length=min(64,s.NI),block_scalar_type=Float64,fullvisc_block=true)
    L=newton_lines_init(s,kw)
    NewtonRANS(kw,L,agg===nothing ? nothing : newton_coarse_init(s;fi=agg[1],fj=agg[2]),newton_sst_init(s),newton_scalar_init(s,L),
        similar(s.dtf),similar(s.dtf),map(copy,(s.rho,s.ux,s.uy,s.T)),ntuple(_->similar(s.dtf),4),cfl,0,Any[],div(typemax(Int),2),NaN)
end

"Flow Newton step at pseudo-CFL m.cfl with frozen eddy viscosity. The preconditioner is rebuilt when the CFL changed or every `lag` steps."
function newton_flow_step!(s::KSol,m::NewtonRANS;lintol=0.05,maxit=40,fraction=0.2,lag=3,pre=1,post=1,blocks=:kinetic,delta=0.3)
    NI,NJ=s.NI,s.NJ;state=(s.rho,s.ux,s.uy,s.T);F4(v)=newton_fields(v,NI,NJ)
    t0=time()
    _raw_residual_ka!(s,Int32(2),Int32(1))
    for (b,d) in zip(m.backup,state);copyto!(b,d);end
    for (b,r) in zip(m.base,_airfoil_residual(s));copyto!(b,r);end
    l2=newton_flow_l2(s)
    newton_d0!(m.d0,s,m.cfl)
    if m.since_build>=lag||m.built_cfl!=m.cfl
        blocks===:roe ? newton_roe_build!(s,m.kw,m.d0;delta) : _airfoil_kinetic_build!(s,m.kw,m.d0)
        newton_lines_factor!(m.L,s,m.kw)
        m.coarse===nothing||newton_coarse_build!(m.coarse,s,m.kw)
        m.since_build=0;m.built_cfl=m.cfl
    end
    m.since_build+=1
    tbuild=time()-t0;t0=time()
    b=vcat(map(vec,m.base)...)
    A!(y,x)=(newton_fd_jv_fwd!(F4(y),F4(x),s,m.backup,m.base,m.d0);y)
    P!(z,v)=(m.coarse===nothing ? newton_line_precond!(F4(z),F4(v),s,m.kw,m.L;sweeps=pre+post) :
        newton_twogrid_precond!(F4(z),F4(v),s,m.kw,m.L,m.coarse;pre,post);z)
    x,its,hist=newton_gmres(A!,P!,b;tol=lintol,restart=maxit,maxit)
    delta=F4(x);tlin=time()-t0
    alpha=Float64(newton_physical_alpha(s,m.backup,delta,fraction));ok=false
    for _ in 1:6
        newton_trial!(s,m.backup,delta,alpha);applyBC_ka!(s,Int32(2))
        ok=_airfoil_physical_state(state...);ok&&break
        alpha/=2
    end
    ok||for (d,bk) in zip(state,m.backup);copyto!(d,bk);end
    (;l2,its,lin=hist[end],alpha,ok,tbuild,tlin)
end

"One coupled iteration: flow Newton step, then `sst_inner` implicit SST steps at pseudo-time dt_flow*turb_ratio."
function newton_rans_step!(s::KSol,m::NewtonRANS;turb_ratio=1.0,sst_inner=1,sst_relax=1.0,nut_relax=1.0,
        sst_tol=1e-3,kwargs...)
    r=newton_flow_step!(s,m;kwargs...)
    @. m.invdt=m.d0/s.vol/turb_ratio
    t0=time();t=nothing
    for _ in 1:sst_inner
        t=newton_sst_implicit_lines!(s,m.sst,m.scalar,m.invdt;relax=sst_relax,nut_relax,tol=sst_tol)
    end
    m.it+=1
    cl,cd=loads_ka(s)
    merge(r,(;it=m.it,cfl=m.cfl,dk=t.dkmax,dw=t.dwmax,tsst=time()-t0,cl=Float64(cl),cd=Float64(cd)))
end

"""March with a geometric CFL ramp. Loads every step; the six residuals every `every` steps.
Stops when CL and CD have stayed within `load_tol` (relative to max(|CL|,0.1), max(|CD|,0.001)) over the last
`window` steps and at least `min_steps` were taken. CSV rows per diagnostic if `csv` is given."""
function newton_rans_march!(s::KSol,m::NewtonRANS;nsteps=100,cflmax=1e3,growth=1.5,budget=Inf,every=5,csv=nothing,
        label="",ref=nothing,window=20,load_tol=1e-4,min_steps=window,quiet=false,rre=nothing,vortex_relax=0.0,
        kwargs...)
    t0=time();io=csv===nothing ? nothing : open(csv,"w")
    io===nothing||println(io,"step,seconds,cfl,its,rho,mx,my,E,k,omega,dk,dw,mass,CL,CD")
    loads=Tuple{Float64,Float64}[];converged=false;nlocal=0
    xs=Vector{Float64}[];scale=rre===nothing ? nothing : newton_state_scale(s);since=0;nrre=0
    try
        for n in 1:nsteps
            r=newton_rans_step!(s,m;kwargs...);nlocal+=1
            # Far-field point vortex (the Prandtl-Glauert corrected circulation of the free-stream boundary): relaxed
            # towards the current lift after every step when vortex_relax > 0 (0, the default: left as it is).
            vortex_relax>0&&r.ok&&(s.gam+=oftype(s.gam,vortex_relax)*(oftype(s.gam,0.5*hypot(s.uxi,s.uyi)*r.cl)-s.gam))
            if rre!==nothing&&r.cfl>=cflmax&&r.ok&&r.alpha>=0.99
                # Reduced-rank extrapolation once the CFL ramp is over: q+2 iterates every `cycle` steps.
                push!(xs,newton_state_vec(s));length(xs)>rre[1]+2&&popfirst!(xs);since+=1
                if since>=rre[2]&&length(xs)==rre[1]+2
                    newton_state_set!(s,newton_rre(xs,scale))&&(nrre+=1);empty!(xs);since=0
                    cl2,cd2=loads_ka(s);r=merge(r,(;cl=Float64(cl2),cd=Float64(cd2)))
                end
            elseif rre!==nothing
                empty!(xs);since=0
            end
            push!(loads,(r.cl,r.cd));length(loads)>window&&popfirst!(loads)
            if length(loads)==window&&nlocal>=min_steps
                cls=first.(loads);cds=last.(loads)
                converged=(maximum(cls)-minimum(cls))/max(abs(r.cl),0.1)<load_tol&&
                          (maximum(cds)-minimum(cds))/max(abs(r.cd),0.001)<load_tol
            end
            if n%every==0||n==1||converged||n==nsteps
                d=steady_airfoil_diagnostics_ka(s;corrected_wall_metric=true)
                six=[d.flow_l2...,d.turb_k_equation_l2,d.turb_omega_equation_l2]
                row=(;step=m.it,seconds=time()-t0,cfl=r.cfl,its=r.its,six,dk=r.dk,dw=r.dw,
                    mass=d.normalized_mass_imbalance,cl=r.cl,cd=r.cd)
                push!(m.rows,row)
                io===nothing||(println(io,join((row.step,row.seconds,row.cfl,row.its,six...,row.dk,row.dw,row.mass,row.cl,row.cd),','));flush(io))
                if !quiet
                    Printf.@printf("%s%4d t=%6.1f cfl=%6.0f its=%2d a=%.2f [build %.2f lin %.2f sst %.2f] six%s=%s mass=%.1e CL=%.6f CD=%.6f\n",
                        label,m.it,time()-t0,r.cfl,r.its,r.alpha,r.tbuild,r.tlin,r.tsst,ref===nothing ? "" : "/cold",
                        string(round.(ref===nothing ? six : six./ref;sigdigits=3)),row.mass,row.cl,row.cd)
                    flush(stdout)
                end
            end
            converged&&break
            m.cfl=r.ok&&r.alpha>=0.99 ? min(cflmax,m.cfl*growth) : max(1.,m.cfl*(r.ok ? max(0.5,r.alpha) : 0.25))
            time()-t0>budget&&break
        end
    finally
        io===nothing||close(io)
    end
    (;converged,steps=nlocal,seconds=time()-t0,extrapolations=nrre)
end


# ---- reduced-rank extrapolation -------------------------------------------------------------------------
"State vector: conservative flow variables and log(k), log(omega), log(nu_t) of the interior cells."
function newton_state_vec(s::KSol)
    g=s.g;ir=g+1:g+s.NI;jr=g+1:g+s.NJ;e=Float64(s.eint)
    r=Array(view(s.rho,ir,jr));u=Array(view(s.ux,ir,jr));v=Array(view(s.uy,ir,jr));T=Array(view(s.T,ir,jr))
    vcat(vec(r),vec(r.*u),vec(r.*v),vec(@. r*(e*T+(u*u+v*v)/2)),
        vec(log.(max.(Array(view(s.tk,ir,jr)),1e-300))),vec(log.(max.(Array(view(s.tom,ir,jr)),1e-300))),
        vec(log.(max.(Array(view(s.nut,ir,jr)),1e-300))))
end
function newton_state_scale(s::KSol)
    n=s.NI*s.NJ;r=Float64(s.rho_inf);u=max(hypot(Float64(s.uxi),Float64(s.uyi)),1e-3);E=r*(Float64(s.eint*s.T_inf)+u*u/2)
    vcat(fill(1/r,n),fill(1/(r*u),n),fill(1/(r*u),n),fill(1/E,n),fill(1.0,3n))
end
"Write a state vector back; returns false (state untouched) if it is not physical."
function newton_state_set!(s::KSol,x)
    NI,NJ,g=s.NI,s.NJ,s.g;n=NI*NJ;e=Float64(s.eint);F(k)=reshape(view(x,(k-1)*n+1:k*n),NI,NJ)
    r=F(1);all(isfinite,x)&&minimum(r)>1e-4||return false
    u=F(2)./r;v=F(3)./r;E=F(4);T=@. (E/r-(u*u+v*v)/2)/e
    minimum(T)>1e-3||return false
    ir=g+1:g+NI;jr=g+1:g+NJ
    view(s.rho,ir,jr).=r;view(s.ux,ir,jr).=u;view(s.uy,ir,jr).=v;view(s.T,ir,jr).=T
    view(s.tk,ir,jr).=clamp.(exp.(F(5)),1e-12,1.0);view(s.tom,ir,jr).=max.(exp.(F(6)),1e-6)
    view(s.nut,ir,jr).=min.(exp.(F(7)),0.15)
    applyBC_ka!(s,Int32(2));_newton_sst_bc!(s)
    true
end

"RRE from iterates xs[1..q+2]: s = x1 + U*xi with xi minimising |u1 + W*xi| in the scaled norm."
function newton_rre(xs::Vector{Vector{Float64}},scale)
    q=length(xs)-2;q>=1||throw(ArgumentError("RRE needs at least three iterates"))
    U=[xs[j+1].-xs[j] for j in 1:q+1]
    W=hcat([(U[j+1].-U[j]).*scale for j in 1:q]...)
    xi=-(W\(U[1].*scale))
    out=copy(xs[1])
    for j in 1:q;out.+=xi[j].*U[j];end
    out
end


# ---- mesh sequencing ------------------------------------------------------------------------------------
# A coarse level keeps every f-th node of the fine mesh (NI, NJ, NIW, NIF divisible by f); a settled coarse state is
# interpolated to the next finer level as its initial guess. Host code, once per level.
function newton_coarsen_nodes(nodes,f::Int)
    f==1&&return nodes
    all(x->x%f==0,(nodes.NI,nodes.NJ,nodes.NIW,nodes.NIF))||throw(ArgumentError("mesh not divisible by $f"))
    (X=nodes.X[1:f:end,1:f:end],Y=nodes.Y[1:f:end,1:f:end],NIW=nodes.NIW÷f,NIF=nodes.NIF÷f,NI=nodes.NI÷f,NJ=nodes.NJ÷f)
end

"Interpolate the coarse solver's state (level spacing ratio f) onto the fine solver; both on the CPU."
function newton_prolong!(sf,sc,f::Int)
    NIc,NJc,NIWc,NIFc,gc=sc.NI,sc.NJ,sc.NIW,sc.NIF,sc.g
    NI,NJ,NIW,NIF,g=sf.NI,sf.NJ,sf.NIW,sf.NIF,sf.g
    (NI,NJ)==(f*NIc,f*NJc)||throw(ArgumentError("levels do not nest"))
    wall(I)=NIWc<I<=NIWc+NIFc
    dc=Array(sc.wdist);df=Array(sf.wdist)
    for (name,mode) in ((:rho,:lin),(:ux,:vel),(:uy,:vel),(:T,:lin),(:tk,:logk),(:tom,:logw),(:nut,:lognut))
        C=Array(getproperty(sc,name));F=Array(getproperty(sf,name))
        logmode=mode in (:logk,:logw,:lognut)
        tiny=1e-30
        # padded coarse array, index offsets +1: P[I+1,J+1] for I=0..NIc+1, J=0..NJc+1
        P=zeros(NIc+2,NJc+2)
        val(I,J)=(v=Float64(C[I+gc,J+gc]);logmode ? log(max(v,tiny)) : v)
        for J in 1:NJc,I in 1:NIc;P[I+1,J+1]=val(I,J);end
        for I in 1:NIc
            P[I+1,NJc+2]=P[I+1,NJc+1]
            P[I+1,1]=wall(I) ? P[I+1,2] : val(NIc+1-I,1)       # wake: the cut partner; wall: handled below
        end
        P[1,:].=P[2,:];P[end,:].=P[end-1,:]
        for j in 1:NJ,i in 1:NI
            xi=(i-0.5)/f+0.5;eta=(j-0.5)/f+0.5
            I0=clamp(floor(Int,xi),0,NIc);tx=xi-I0;J0=clamp(floor(Int,eta),0,NJc);ty=eta-J0
            v=(1-tx)*(1-ty)*P[I0+1,J0+1]+tx*(1-ty)*P[I0+2,J0+1]+(1-tx)*ty*P[I0+1,J0+2]+tx*ty*P[I0+2,J0+2]
            if J0==0&&NIW<i<=NIW+NIF
                # Below the first coarse cell centre on the foil: sublayer scaling with the wall distance.
                Ic=clamp(round(Int,xi),NIWc+1,NIWc+NIFc)
                q=clamp(Float64(df[i,j])/max(Float64(dc[Ic,1]),1e-30),0.0,1.0)
                row=(1-tx)*P[I0+1,2]+tx*P[I0+2,2]
                v=mode===:vel ? row*q : mode===:logw ? row-2log(max(q,1e-6)) : mode===:logk ? row+2log(max(q,1e-6)) :
                  mode===:lognut ? row+4log(max(q,1e-6)) : row
            end
            F[i+g,j+g]=logmode ? exp(v) : v
        end
        copyto!(getproperty(sf,name),F)
    end
    applyBC_ka!(sf,Int32(2));_newton_sst_bc!(sf)
    sf
end


# ---- cold start: sequencing and the march on each level -------------------------------------------------
# `nodes` = (X, Y, NIW, NIF, NI, NJ) of the fine mesh; `make_solver(nodes)` returns a prepared KSol for a level.
function newton_cold_run(nodes,make_solver;levels=(4,2,1),nsteps=(120,120,160),cflmax=(1e3,1e3,1e3),
        load_tol=(1e-3,3e-4,1e-4),window=(10,15,20),rre=(3,8),blocks=:roe,delta=0.3,outdir=nothing,every=10,
        lintol=0.05,maxit=40,quiet=false,vortex_relax=0.0,growth=1.5)
    prev=nothing;pf=0;t00=time();out=nothing;summary=Dict{String,Any}();lv=Any[]
    for (k,f) in enumerate(levels)
        t0=time();nd=newton_coarsen_nodes(nodes,f)
        s=make_solver(nd)
        d=steady_airfoil_diagnostics_ka(s;corrected_wall_metric=true)
        ref=[d.flow_l2...,d.turb_k_equation_l2,d.turb_omega_equation_l2]
        prev===nothing||(newton_prolong!(s,prev,pf÷f);s.gam=oftype(s.gam,prev.gam))
        m=newton_rans_init(s;cfl=prev===nothing ? 5. : 20.,agg=(4÷f,4÷f))
        setup=time()-t0
        quiet||Printf.@printf("LEVEL 1/%d: %d x %d = %d cells, setup %.1fs\n",f,nd.NI,nd.NJ,nd.NI*nd.NJ,setup)
        r=newton_rans_march!(s,m;nsteps=nsteps[k],cflmax=cflmax[k],growth,every,vortex_relax,label="L$f ",ref,quiet,
            load_tol=load_tol[k],window=window[k],rre,blocks,delta,lintol,maxit,
            csv=outdir===nothing ? nothing : joinpath(outdir,"history_L$f.csv"))
        cl,cd=loads_ka(s)
        quiet||Printf.@printf("LEVEL 1/%d done: loads stable=%s, %d steps, %d extrapolations, %.1fs level, %.1fs total, CL=%.6f CD=%.6f\n",
            f,r.converged,r.steps,r.extrapolations,time()-t0,time()-t00,cl,cd)
        flush(stdout)
        push!(lv,Dict("coarsening"=>f,"cells"=>nd.NI*nd.NJ,"steps"=>r.steps,"loads_stable"=>r.converged,
            "extrapolations"=>r.extrapolations,"seconds"=>time()-t0,"setup_seconds"=>setup,"CL"=>Float64(cl),"CD"=>Float64(cd),
            "cold_reference_six"=>ref))
        prev=s;pf=f;out=(;s,m,ref)
    end
    d=steady_airfoil_diagnostics_ka(out.s;corrected_wall_metric=true)
    six=[d.flow_l2...,d.turb_k_equation_l2,d.turb_omega_equation_l2]
    summary["levels"]=lv;summary["solve_seconds"]=time()-t00
    summary["final_six_residuals"]=six;summary["final_six_over_cold"]=six./out.ref
    summary["final_CL"]=Float64(d.cl);summary["final_CD"]=Float64(d.cd)
    summary["final_mass_imbalance"]=d.normalized_mass_imbalance
    summary["final_omega_wall_constraint"]=d.turb_omega_wall_constraint_relative_linf
    (;s=out.s,m=out.m,ref=out.ref,summary)
end
