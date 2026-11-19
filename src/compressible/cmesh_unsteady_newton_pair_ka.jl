# BDF dual time stepping on the two-block (C + strip) Newton march of a blunt trailing edge (18 Nov 2026). EXPERIMENTAL
# AND OPT-IN: the steady pair march is unchanged unless a pair of `NewtonTime` is passed (keyword `dual` of
# newton_pair_flow_step!, newton_pair_step! and newton_pair_sst_step!). Included into module CMeshSolverKA after
# cmesh_steady_newton_pair_ka.jl. The time discretisation is that of the single block (cmesh_unsteady_newton_ka.jl): each
# block carries its own history, both advance together, and the inner iterations solve the coupled unsteady system of
# both blocks (the interface terms are those of the steady pair). The far-field point vortex is held at its start value.

"Physical-time state of the pair: one `NewtonTime` per block (C, strip), kept at the same step."
struct NewtonPairTime{A,B}
    c::NewtonTime{A}; h::NewtonTime{B}
end
newton_pair_time_init(p::NewtonPair;dt,t0=0.0)=NewtonPairTime(newton_time_init(p.c;dt,t0),newton_time_init(p.h;dt,t0))

"""One physical time step of the pair: inner coupled iterations (`newton_pair_step!` with `dual`) at pseudo-CFL `cfl` until
the largest scaled norm of the unsteady flow residual (over both blocks) has fallen by `inner_tol` from its first inner value
or below `inner_abs`, or `inner_max` iterations are reached; then both histories advance. Returns the step's record."""
function newton_pair_time_step!(p::NewtonPair,m::NewtonPairState,tm::NewtonPairTime;cfl=1e4,inner_max=20,inner_tol=1e-3,
        inner_abs=0.0,delta=0.3,lintol=0.05,maxit=40,lag=3,fraction=0.2,cfl_adapt=false,cfl_min=1.0,cfl_growth=1.5,kwargs...)
    m.cfl=cfl;m.since_build=div(typemax(Int),2)               # rebuild the preconditioner with this step's diagonal
    dual=(tm.c,tm.h);first=NaN;its=0;k=0;ok=true
    for kk in 1:inner_max
        k=kk
        r=newton_pair_step!(p,m;dual,delta,lintol,maxit,lag,fraction,kwargs...)
        lt=maximum(r.l2t);its+=r.its;ok&=r.ok
        kk==1&&(first=lt)
        (lt<=inner_tol*first||lt<=inner_abs)&&kk>1&&break
        # opt-in inner CFL control, as newton_time_step! (19 Nov 2026; default off: the inner CFL stays `cfl`)
        cfl_adapt&&(m.cfl=newton_time_next_cfl(m.cfl,r,cfl,cfl_min,cfl_growth))
    end
    # residual of the final state (the loop's norms are those before each update)
    newton_pair_residual!(p)
    for (s,base) in ((p.c,m.basec),(p.h,m.baseh)),(b,r) in zip(base,_airfoil_residual(s));copyto!(b,r);end
    newton_d0!(m.d0c,p.c,m.cfl);newton_d0!(m.d0h,p.h,m.cfl)
    final=maximum(max.(newton_time_rhs!(m.d0c,m.basec,p.c,tm.c)[2],newton_time_rhs!(m.d0h,m.baseh,p.h,tm.h)[2]))
    newton_time_advance!(tm.c,p.c);newton_time_advance!(tm.h,p.h)
    cl,cd=newton_pair_loads(p)
    (;step=tm.c.step,t=tm.c.t,cl=Float64(cl),cd=Float64(cd),inner=k,gmres=its,first,final,drop=final/first,ok,cfl=m.cfl)
end

"""March `nsteps` physical steps of the pair (`newton_pair_time_step!`), as `newton_time_march!`: a CSV row per step
(`csv`, appended to when it exists), `on_step(record)`, and a restart file (`checkpoint`) every `every_checkpoint` steps and at
the end."""
function newton_pair_time_march!(p::NewtonPair,m::NewtonPairState,tm::NewtonPairTime;nsteps=100,csv=nothing,on_step=nothing,
        quiet=false,checkpoint=nothing,every_checkpoint=100,kwargs...)
    io=csv===nothing ? nothing : open(csv,isfile(csv) ? "a" : "w")
    io===nothing||position(io)>0||println(io,"step,t,CL,CD,inner,gmres,first,final,ok")
    recs=Any[];t0=time()
    try
        for n in 1:nsteps
            r=newton_pair_time_step!(p,m,tm;kwargs...)
            push!(recs,r)
            io===nothing||(println(io,join((r.step,r.t,r.cl,r.cd,r.inner,r.gmres,r.first,r.final,r.ok),','));flush(io))
            on_step===nothing||on_step(r)
            quiet||Printf.@printf("T %6d t=%10.5f CL=%.6f CD=%.6f inner=%2d gmres=%4d res %.2e -> %.2e %s\n",
                r.step,r.t,r.cl,r.cd,r.inner,r.gmres,r.first,r.final,r.ok ? "" : "LIMITED")
            checkpoint===nothing||(r.step%every_checkpoint==0||n==nsteps)&&newton_pair_time_save(checkpoint,p,tm)
        end
    finally
        io===nothing||close(io)
    end
    (;steps=length(recs),seconds=time()-t0,records=recs)
end

_newton_time_block_dict(s::KSol,tm::NewtonTime)=Dict{Symbol,Any}(:rho=>Array(s.rho),:ux=>Array(s.ux),:uy=>Array(s.uy),:T=>Array(s.T),
    :tk=>Array(s.tk),:tom=>Array(s.tom),:nut=>Array(s.nut),:gam=>s.gam,:un=>map(Array,tm.un),:um=>map(Array,tm.um),
    :kn=>Array(tm.kn),:km=>Array(tm.km),:wn=>Array(tm.wn),:wm=>Array(tm.wm))
function _newton_time_block_load!(s::KSol,b,dt,t,step,order)
    for (f,k) in ((s.rho,:rho),(s.ux,:ux),(s.uy,:uy),(s.T,:T),(s.tk,:tk),(s.tom,:tom),(s.nut,:nut));copyto!(f,b[k]);end
    s.gam=oftype(s.gam,b[:gam])
    tm=newton_time_init(s;dt,t0=t)
    for k in 1:4;copyto!(tm.un[k],b[:un][k]);copyto!(tm.um[k],b[:um][k]);end
    copyto!(tm.kn,b[:kn]);copyto!(tm.km,b[:km]);copyto!(tm.wn,b[:wn]);copyto!(tm.wm,b[:wm])
    tm.step=step;tm.order=order
    tm
end

"Write a restart file of the pair: both blocks' states (with ghosts), the far-field vortex and both time histories."
function newton_pair_time_save(path,p::NewtonPair,tm::NewtonPairTime)
    st=Dict{Symbol,Any}(:c=>_newton_time_block_dict(p.c,tm.c),:h=>_newton_time_block_dict(p.h,tm.h),
        :dt=>tm.c.dt,:t=>tm.c.t,:step=>tm.c.step,:order=>tm.c.order)
    tmp=path*".tmp";open(io->serialize(io,st),tmp,"w");mv(tmp,path;force=true)
    path
end

"Read a restart file written by `newton_pair_time_save` into the pair; returns the `NewtonPairTime` to continue with."
function newton_pair_time_load!(path,p::NewtonPair)
    st=open(deserialize,path)
    tc=_newton_time_block_load!(p.c,st[:c],st[:dt],st[:t],st[:step],st[:order])
    th=_newton_time_block_load!(p.h,st[:h],st[:dt],st[:t],st[:step],st[:order])
    newton_pair_bc!(p);newton_pair_sst_bc!(p)
    NewtonPairTime(tc,th)
end
