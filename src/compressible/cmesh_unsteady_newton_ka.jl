# BDF dual time stepping on the Newton-Krylov airfoil march (single-block C mesh; 18 Nov 2026). EXPERIMENTAL AND
# OPT-IN: the steady march is unchanged unless a `NewtonTime` is passed (keyword `dual` of newton_flow_step! and
# newton_rans_step!). Included into module CMeshSolverKA from cmesh_steady_newton_ka.jl.
#
# The march's residual R is the flux balance of a cell (V dU/dt = -R; its update is U <- U - alpha delta).
# Physical time is discretised by BDF2 (BDF1 on the first step after a start or restart without history):
#     V (a0 U^(n+1) - a1 U^n - a2 U^(n-1)) + R(U^(n+1)) = 0,   BDF2: a0 = 3/(2 dt), a1 = 2/dt, a2 = -1/(2 dt);
#                                                             BDF1: a0 = 1/dt,     a1 = 1/dt, a2 = 0.
# Each physical step is solved by inner Newton iterations in pseudo-time: the residual R is replaced by
# R* = R + V (a0 U - a1 U^n - a2 U^(n-1)) and a0 V is added to the pseudo-time diagonal, so the finite-difference
# Jacobian product, the preconditioner and the step limit all act on the unsteady system. The SST equations (k and omega
# form) take the same BDF terms in their point-implicit systems. U is conservative (rho, rho u, rho v, rho E) with
# rho E = rho (eint T + |u|^2 / 2), the form of the march's increments.
#
# The far-field point vortex is held at its value at the start (it is not relaxed toward an oscillating lift).
using Serialization

"State of the physical-time integration: step, coefficients' order and the two previous time levels (interior cells)."
mutable struct NewtonTime{A}
    dt::Float64; t::Float64; step::Int; order::Int
    un::NTuple{4,A}; um::NTuple{4,A}
    kn::A; km::A; wn::A; wm::A
end

"Conservative interior state of `s` into the four arrays `U` (size NI x NJ)."
function newton_time_conserved!(U,s::KSol)
    g=s.g;ir=g+1:g+s.NI;jr=g+1:g+s.NJ;e=s.eint
    r=view(s.rho,ir,jr);u=view(s.ux,ir,jr);v=view(s.uy,ir,jr);T=view(s.T,ir,jr)
    @. U[1]=r;@. U[2]=r*u;@. U[3]=r*v;@. U[4]=r*(e*T+(u*u+v*v)/2)
    U
end

"""Start the physical-time integration from the current state of `s` with time step `dt` (in the solver's time units, c/U
scaled as the solver's velocities). The first step is BDF1; `t0` sets the starting time."""
function newton_time_init(s::KSol;dt,t0=0.0)
    dt>0&&isfinite(dt)||throw(ArgumentError("dt must be positive and finite"))
    mk()=similar(s.dtf);un=ntuple(_->mk(),4);um=ntuple(_->mk(),4)
    newton_time_conserved!(un,s);for k in 1:4;copyto!(um[k],un[k]);end
    g=s.g;ir=g+1:g+s.NI;jr=g+1:g+s.NJ
    kn=copy(view(s.tk,ir,jr));wn=copy(view(s.tom,ir,jr))
    NewtonTime(Float64(dt),Float64(t0),0,1,un,um,typeof(kn)(kn),typeof(kn)(copy(kn)),typeof(kn)(wn),typeof(kn)(copy(wn)))
end

"(a0, a1, a2) of the current step: BDF1 when `order` is 1, BDF2 otherwise."
newton_time_coeffs(tm::NewtonTime)=tm.order==1 ? (1/tm.dt,1/tm.dt,0.0) : (1.5/tm.dt,2/tm.dt,-0.5/tm.dt)

"BDF terms of the SST systems for `newton_sst_implicit_lines!(...; bdf)`."
function newton_time_sst_terms(tm::NewtonTime)
    a0,a1,a2=newton_time_coeffs(tm)
    (;a0,a1,a2,kn=tm.kn,km=tm.km,wn=tm.wn,wm=tm.wm)
end

"""Unsteady right-hand side of the current state: returns (R*, norms) with R* = `base` + V (a0 U - a1 U^n - a2 U^(n-1))
(new arrays; `base`, the raw residual, is left unchanged because the finite-difference Jacobian product differences against
it) and the scaled norms of R* per equation (the scaling of `newton_flow_l2`); adds a0 V to the diagonal `d0` in place."""
function newton_time_rhs!(d0,base,s::KSol,tm::NewtonTime)
    a0,a1,a2=newton_time_coeffs(tm);V=s.vol
    U=ntuple(_->similar(s.dtf),4);newton_time_conserved!(U,s)
    rhs=ntuple(k->base[k].+V.*(a0.*U[k].-a1.*tm.un[k].-a2.*tm.um[k]),4)
    @. d0+=a0*V
    r=max(Float64(s.rho_inf),1e-12);u=max(hypot(Float64(s.uxi),Float64(s.uyi)),1e-3)
    e=max(Float64(s.eint*s.T_inf),u*u);scales=(r,r*u,r*u,r*e);n=length(V)
    rhs,ntuple(k->sqrt(mapreduce((a,v)->abs2(Float64(a)/v),+,rhs[k],V)/n)/scales[k],4)
end

"Advance the history after a converged physical step: U^(n-1) <- U^n, U^n <- current state, t += dt, order -> 2."
function newton_time_advance!(tm::NewtonTime,s::KSol)
    for k in 1:4;copyto!(tm.um[k],tm.un[k]);end
    newton_time_conserved!(tm.un,s)
    g=s.g;ir=g+1:g+s.NI;jr=g+1:g+s.NJ
    copyto!(tm.km,tm.kn);copyto!(tm.kn,view(s.tk,ir,jr))
    copyto!(tm.wm,tm.wn);copyto!(tm.wn,view(s.tom,ir,jr))
    tm.t+=tm.dt;tm.step+=1;tm.order=2
    tm
end

"""One physical time step: inner coupled Newton iterations (`newton_rans_step!` with `dual = tm`) at pseudo-CFL `cfl` until
the largest scaled norm of the unsteady flow residual has fallen by `inner_tol` from its first inner value or below
`inner_abs`, or `inner_max` iterations are reached; then the history advances. Returns the step's record."""
function newton_time_step!(s::KSol,m::NewtonRANS,tm::NewtonTime;cfl=1e4,inner_max=20,inner_tol=1e-3,inner_abs=0.0,
        blocks=:roe,delta=0.3,lintol=0.05,maxit=40,lag=3,fraction=0.2,kwargs...)
    m.cfl=cfl;m.since_build=div(typemax(Int),2)               # rebuild the preconditioner with this step's diagonal
    first=NaN;last=NaN;its=0;lin=0;k=0;ok=true
    for kk in 1:inner_max
        k=kk
        r=newton_rans_step!(s,m;dual=tm,blocks,delta,lintol,maxit,lag,fraction,kwargs...)
        lt=maximum(r.l2t);its+=r.its;ok&=r.ok
        kk==1&&(first=lt);last=lt
        (lt<=inner_tol*first||lt<=inner_abs)&&kk>1&&break
    end
    # residual of the final state (the loop's norms are those before each update)
    _raw_residual_ka!(s,Int32(2),Int32(1))
    for (b,r) in zip(m.base,_airfoil_residual(s));copyto!(b,r);end
    newton_d0!(m.d0,s,m.cfl);final=maximum(newton_time_rhs!(m.d0,m.base,s,tm)[2])
    newton_time_advance!(tm,s)
    cl,cd=loads_ka(s)
    (;step=tm.step,t=tm.t,cl=Float64(cl),cd=Float64(cd),inner=k,gmres=its,first,final,drop=final/first,ok)
end

"""March `nsteps` physical steps (`newton_time_step!`), writing a CSV row per step (`csv`) and calling `on_step(record)`.
`checkpoint` (path) and `every_checkpoint`: write a restart file (`newton_time_save`) every so many steps and at the end."""
function newton_time_march!(s::KSol,m::NewtonRANS,tm::NewtonTime;nsteps=100,csv=nothing,on_step=nothing,quiet=false,
        checkpoint=nothing,every_checkpoint=100,kwargs...)
    io=csv===nothing ? nothing : open(csv,isfile(csv) ? "a" : "w")
    io===nothing||position(io)>0||println(io,"step,t,CL,CD,inner,gmres,first,final,ok")
    recs=Any[];t0=time()
    try
        for n in 1:nsteps
            r=newton_time_step!(s,m,tm;kwargs...)
            push!(recs,r)
            io===nothing||(println(io,join((r.step,r.t,r.cl,r.cd,r.inner,r.gmres,r.first,r.final,r.ok),','));flush(io))
            on_step===nothing||on_step(r)
            quiet||Printf.@printf("T %6d t=%10.5f CL=%.6f CD=%.6f inner=%2d gmres=%4d res %.2e -> %.2e %s\n",
                r.step,r.t,r.cl,r.cd,r.inner,r.gmres,r.first,r.final,r.ok ? "" : "LIMITED")
            checkpoint===nothing||(r.step%every_checkpoint==0||n==nsteps)&&newton_time_save(checkpoint,s,tm)
        end
    finally
        io===nothing||close(io)
    end
    (;steps=length(recs),seconds=time()-t0,records=recs)
end

"Write a restart file: the state of `s` (with ghosts), the far-field vortex and the time history."
function newton_time_save(path,s::KSol,tm::NewtonTime)
    st=Dict{Symbol,Any}(:rho=>Array(s.rho),:ux=>Array(s.ux),:uy=>Array(s.uy),:T=>Array(s.T),:tk=>Array(s.tk),:tom=>Array(s.tom),
        :nut=>Array(s.nut),:gam=>s.gam,:dt=>tm.dt,:t=>tm.t,:step=>tm.step,:order=>tm.order,
        :un=>map(Array,tm.un),:um=>map(Array,tm.um),:kn=>Array(tm.kn),:km=>Array(tm.km),:wn=>Array(tm.wn),:wm=>Array(tm.wm))
    tmp=path*".tmp";open(io->serialize(io,st),tmp,"w");mv(tmp,path;force=true)
    path
end

"Read a restart file written by `newton_time_save` into `s`; returns the `NewtonTime` to continue with."
function newton_time_load!(path,s::KSol)
    st=open(deserialize,path)
    for (f,k) in ((s.rho,:rho),(s.ux,:ux),(s.uy,:uy),(s.T,:T),(s.tk,:tk),(s.tom,:tom),(s.nut,:nut));copyto!(f,st[k]);end
    s.gam=oftype(s.gam,st[:gam])
    tm=newton_time_init(s;dt=st[:dt],t0=st[:t])
    for k in 1:4;copyto!(tm.un[k],st[:un][k]);copyto!(tm.um[k],st[:um][k]);end
    copyto!(tm.kn,st[:kn]);copyto!(tm.km,st[:km]);copyto!(tm.wn,st[:wn]);copyto!(tm.wm,st[:wm])
    tm.step=st[:step];tm.order=st[:order]
    applyBC_ka!(s,Int32(2));_newton_sst_bc!(s)
    tm
end
