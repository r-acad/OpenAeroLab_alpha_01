# Experimental pure thermal-gas / conservative SST-2003m C+H adapter.
# Shared native gas energy flux remains thermal; no mixture EOS projection,
# vapour transport, Reboud correction, SFD or local pseudo-time is called.
module GasStripSSTPrototype
using KernelAbstractions,SHA
import ..CMeshSolverKA
import ..GasStripPrototype
const G=CMeshSolverKA
const P=GasStripPrototype
const TRIP_YPLUS_MAX=30.0f0
struct StageReject <: Exception
    reason::String
end
Base.showerror(io::IO,e::StageReject)=print(io,e.reason)
_reject_stage(c,stage,kind,message,retryable)=throw(StageReject("SST stage $stage $kind: $message"))
_wall_function(s)=Int32(0)
_strip_ctx(q)=q.ctx
_block_label(q)=" gas $(q.name) block"
include(joinpath(@__DIR__,"..","..","..","..","src","cmesh","cmesh_species_sst.jl"))
export CoupledPair,build_coupled,residual!,step!,scalar_balance,inventory

struct SSTBlock{S,A,T,C,U}
    solver::S
    turbulence::T
    ctx::C
    status::A
    source_dt::A
    pressure::A
    tem::NamedTuple{(:alpha,:rho_l_sat,:rho_v_sat),Tuple{A,Float32,Float32}}
    reconstruction::Nothing
    wallmode::Int32
    sst_climit::Float32
    reboud_n::Float32
    reboud_form::Int32
    reboud_prod::Int32
    trip::NTuple{2,Float32}
    conserved::NamedTuple{(:Utrial,),Tuple{U}}
    name::String
end

mutable struct CoupledPair{P,B}
    flow::P
    c::B
    h::B
    accepted_steps::Int
    elapsed::Float64
    last_rejections::Int
end

function block(s,ctx,name)
    _admit_sst(s)
    a=_build_sst(s,nothing)
    z()=fill!(similar(s.rho,s.NI,s.NJ),0.0f0)
    ghost()=fill!(similar(s.rho),0.0f0)
    packed=fill!(similar(s.rho,Float64,s.NI,s.NJ,4),0.0)
    SSTBlock(s,a,ctx,z(),z(),ghost(),(alpha=ghost(),rho_l_sat=1f0,rho_v_sat=0f0),
        nothing,Int32(2),0.0f0,0.0f0,Int32(0),Int32(0),(0.0f0,0.0f0),(Utrial=packed,),name)
end

function admit(p::CoupledPair)
    P.admit_settings(p.flow.c,p.flow.h,p.flow.wallmode;allow_sst=true)
    p.flow.wallmode==2 || throw(ArgumentError("Conservative SST adapter requires no-slip wall"))
    for q in (p.c,p.h)
        s=q.solver
        s.sst && s.nu_lam>0 && s.fullvisc==1 && all(iszero,(s.keps,s.lag,s.earsm,s.rhok,s.fbm_cdelta,s.pans_fk)) || throw(ArgumentError("Unsupported SST extension or stress mode"))
        s.suth==0 && s.air_sutherland==0 || throw(ArgumentError("Conservative SST proof requires constant kinematic viscosity; local variable/dynamic viscosity coupling is not admitted"))
        q.reboud_n==0 && q.reboud_form==0 && q.reboud_prod==0 && q.trip==(0f0,0f0) || throw(ArgumentError("This proof is fully turbulent pure gas; no Reboud or imposed trip"))
    end
    p.c.solver.k_inf===p.h.solver.k_inf && p.c.solver.w_inf===p.h.solver.w_inf || throw(ArgumentError("SST inflow settings differ"))
end

function build_coupled(p::P.GasPair;tu_inf::Real=.001,mu_t_ratio::Real=.1)
    P.admit_settings(p.c,p.h,p.wallmode)
    p.iterations==0 && p.elapsed==0 || throw(ArgumentError("SST adapter requires a fresh gas pair; no hidden clock-origin reset"))
    all(s->s.suth==0 && s.air_sutherland==0,(p.c,p.h)) || throw(ArgumentError("SST adapter requires constant kinematic viscosity"))
    p.wallmode==2 || throw(ArgumentError("SST adapter requires no-slip walls"))
    0<tu_inf<.1 && mu_t_ratio>0 && isfinite(mu_t_ratio) || throw(ArgumentError("Invalid SST freestream"))
    for s in (p.c,p.h);G.set_sst_ka!(s;tu_inf=Float64(tu_inf),mu_t_ratio=Float64(mu_t_ratio));end
    sig()=fill!(similar(p.c.rho,Float32,p.c.NIW,2,2),0.0f0)
    c=block(p.c,(p.ctx_c...,sig_nb=sig()),"C")
    h=block(p.h,(p.ctx_h...,sig_nb=sig()),"H")
    result=CoupledPair(p,c,h,0,0.0,0)
    admit(result);residual!(result)
    result
end

@kernel function pressure_cache!(pressure,@Const(r),@Const(t),NX::Int32,NY::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NX && j<=NY;pressure[i,j]=r[i,j]*(2f0/3f0)*t[i,j];end
end

@kernel function exchange_sst!(kc,wc,nc,kh,wh,nh,sigc,sigh,sc_k,sc_w,sh_k,sh_w,
                              NW::Int32,NF::Int32,NS::Int32,g::Int32,mode::Int32)
    st,k=@index(Global,NTuple)
    @inbounds if st<=NW && k<=g
        s=Int32(st);q=Int32(k);il=NW+1-s;iu=NW+NF+s;ic=il+g;uc=iu+g;ih=s+g
        ghost=g+1-q;own=g+q;up=NS+g+1-q;top=NS+g+q
        if mode==0
            kc[ic,ghost]=kh[ih,own];wc[ic,ghost]=wh[ih,own]
            kc[uc,ghost]=kh[ih,up];wc[uc,ghost]=wh[ih,up]
            kh[ih,ghost]=kc[ic,own];wh[ih,ghost]=wc[ic,own]
            kh[ih,top]=kc[uc,own];wh[ih,top]=wc[uc,own]
        else
            nc[ic,ghost]=nh[ih,own];nc[uc,ghost]=nh[ih,up]
            nh[ih,ghost]=nc[ic,own];nh[ih,top]=nc[uc,own]
            if q==1
                sigc[s,1,1]=sh_k[s,1];sigc[s,2,1]=sh_w[s,1]
                sigc[s,1,2]=sh_k[s,NS];sigc[s,2,2]=sh_w[s,NS]
                sigh[s,1,1]=sc_k[il,1];sigh[s,2,1]=sc_w[il,1]
                sigh[s,1,2]=sc_k[iu,1];sigh[s,2,2]=sc_w[iu,1]
            end
        end
    end
end

function exchange_sst!(p::CoupledPair,mode)
    c,h=p.c.solver,p.h.solver;a,b=p.c.turbulence,p.h.turbulence
    exchange_sst!(c.backend)(c.tk,c.tom,c.nut,h.tk,h.tom,h.nut,p.c.ctx.sig_nb,p.h.ctx.sig_nb,
        a.sigk,a.sigw,b.sigk,b.sigw,Int32(c.NIW),Int32(c.NIF),Int32(h.NJ),Int32(c.g),Int32(mode);ndrange=(c.NIW,c.g))
end

function prepare!(p::CoupledPair)
    for q in (p.c,p.h)
        s=q.solver;a=q.turbulence
        _k_sst_recover!(s.backend)(s.tk,s.tom,a.rk,a.rw,s.rho,Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=(s.NI,s.NJ))
        _sst_bc!(q)
    end
    exchange_sst!(p,0)
    for q in (p.c,p.h)
        s=q.solver;a=q.turbulence;fill!(q.status,0f0)
        pressure_cache!(s.backend)(q.pressure,s.rho,s.T,Int32(size(s.rho,1)),Int32(size(s.rho,2));ndrange=size(s.rho))
        _k_sst_species_coeff!(s.backend)(s.nut,a.sigk,a.sigw,a.sk,a.sw,a.rate,q.status,s.rho,q.tem.alpha,q.pressure,s.tk,s.tom,
            s.ux,s.uy,s.wdist,s.gammat,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,s.nu_lam,0f0,Int32(0),Int32(0),1f0,0f0,
            (0f0,0f0,0f0,0f0),Int32(s.NI),Int32(s.NJ),Int32(s.g),Int32(s.NIW),Int32(s.NIF),Int32(0),q.ctx;ndrange=(s.NI,s.NJ))
        maximum(q.status)==0 || throw(StageReject("Nonfinite SST coefficient/source in $(q.name)"))
    end
    exchange_sst!(p,1)
    for q in (p.c,p.h)
        _sst_bc!(q;nut_only=true);_sst_owned_status!(q)
        maximum(q.status)==0 || throw(StageReject("Changed SST geometry or inconsistent owned state in $(q.name)"))
    end
end

function scalar_flux!(q)
    s=q.solver;a=q.turbulence;ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g)
    _k_sst_species_flux_i!(s.backend)(a.Fik,a.Fiw,s.Fi_r,a.rk,a.rw,s.rho,s.tk,s.tom,s.nut,a.sigk,a.sigw,
        s.xc,s.yc,s.Six,s.Siy,s.nu_lam,s.k_inf,s.w_inf,nothing,nothing,nothing,nothing,ni,nj,g,0f0,q.ctx;ndrange=(s.NI+1,s.NJ))
    _k_sst_species_flux_j!(s.backend)(a.Fjk,a.Fjw,s.Fj_r,a.rk,a.rw,s.rho,s.tk,s.tom,s.nut,a.sigk,a.sigw,
        s.xc,s.yc,s.Sjx,s.Sjy,s.vol,s.outer_face_kind,s.nu_lam,s.k_inf,s.w_inf,Int32(2),nothing,nothing,nothing,nothing,
        ni,nj,Int32(s.NIW),Int32(s.NIF),g,Int32(0),0f0,q.ctx;ndrange=(s.NI,s.NJ+1))
end

function scalar_residual!(q)
    s=q.solver;a=q.turbulence
    _k_sst_species_residual!(s.backend)(a.dck,a.dcw,q.source_dt,a.Fik,a.Fiw,a.Fjk,a.Fjw,s.Fi_r,s.Fj_r,
        a.sk,a.sw,a.rate,a.rk,a.rw,nothing,nothing,s.rho,s.nut,a.sigk,a.sigw,s.xc,s.yc,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,s.nu_lam,Int32(2),
        Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF),Int32(s.g),Int32(0),0f0,q.ctx;ndrange=(s.NI,s.NJ))
    cap=minimum(q.source_dt)
    cap>0 || throw(StageReject("Invalid SST transport/source clock in $(q.name)"))
    Float64(cap)
end

function residual!(p::CoupledPair)
    admit(p);f=p.flow;P.exchange!(f);prepare!(p)
    for (q,ctx) in ((p.c,p.c.ctx),(p.h,p.h.ctx))
        s=q.solver
        G._raw_residual_body_ka!(s,Int32(2),Int32(1),false,nothing;strip_ctx=ctx,store_divergence=false,outlet_sides=ctx.sides)
    end
    P.common_flux!(f.c.backend)(P.faces_j(f.c),P.faces_j(f.h),Int32(f.c.NIW),Int32(f.c.NIF),Int32(f.h.NJ);ndrange=f.c.NIW)
    for s in (f.c,f.h)
        G._k_residual!(s.backend)(P.residuals(s)...,P.faces_i(s)...,P.faces_j(s)...,Int32(s.NI),Int32(s.NJ);ndrange=(s.NI,s.NJ))
    end
    scalar_flux!(p.c);scalar_flux!(p.h)
    a,b=p.c.turbulence,p.h.turbulence
    P.common_flux!(f.c.backend)((a.Fjk,a.Fjw),(b.Fjk,b.Fjw),Int32(f.c.NIW),Int32(f.c.NIF),Int32(f.h.NJ);ndrange=f.c.NIW)
    min(scalar_residual!(p.c),scalar_residual!(p.h))
end

@kernel function pack_gas!(U,@Const(r),@Const(u),@Const(v),@Const(t),eint::Float32,NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        rr=Float64(r[i+g,j+g]);uu=Float64(u[i+g,j+g]);vv=Float64(v[i+g,j+g]);tt=Float64(t[i+g,j+g])
        U[i,j,1]=rr;U[i,j,2]=rr*uu;U[i,j,3]=rr*vv;U[i,j,4]=rr*(Float64(eint)*tt+.5*(uu*uu+vv*vv))
    end
end

function inventory(p::CoupledPair)
    k=0.;w=0.
    for q in (p.c,p.h)
        v=Float64.(Array(q.solver.vol));a=q.turbulence
        k+=sum(Float64.(Array(a.rk)).*v);w+=sum(Float64.(Array(a.rw)).*v)
    end
    vcat(P.inventory(p.flow),[k,w])
end

function scalar_balance(p::CoupledPair)
    result=[]
    for (name,ik,jk,src,res) in ((:rho_k,:Fik,:Fjk,:sk,:dck),(:rho_omega,:Fiw,:Fjw,:sw,:dcw))
        div=0.;bound=0.;source=0.;stored=0.
        for q in (p.c,p.h)
            a=q.turbulence;fi=Float64.(Array(getfield(a,ik)));fj=Float64.(Array(getfield(a,jk)))
            div+=sum(fi[2:end,:].-fi[1:end-1,:].+fj[:,2:end].-fj[:,1:end-1])
            bound+=sum(fi[end,:])-sum(fi[1,:])+sum(fj[:,end])-sum(fj[:,1]);source+=sum(Float64.(Array(getfield(a,src))))
            stored+=sum(Float64.(Array(getfield(a,res))))
        end
        push!(result,(equation=name,face64_divergence=div,boundary=bound,source=source,face64_telescoping_error=div-bound,stored_residual=stored,storage_difference=stored-(div-source)))
    end
    result
end

function backup(p)
    map((p.c,p.h)) do q
        s=q.solver;a=q.turbulence
        map(copy,(s.rho,s.ux,s.uy,s.T,s.tk,s.tom,s.nut,a.rk,a.rw))
    end
end
function restore!(p,saved,old_dt)
    for (q,b) in zip((p.c,p.h),saved)
        s=q.solver;a=q.turbulence
        for (dst,src) in zip((s.rho,s.ux,s.uy,s.T,s.tk,s.tom,s.nut,a.rk,a.rw),b);copyto!(dst,src);end
    end
    p.c.solver.last_global_dt,p.h.solver.last_global_dt=old_dt
end

function step!(p::CoupledPair,cfl::Real;max_dt::Real=Inf,max_retries::Int=6,stage_observer=nothing)
    p.c.solver.irsj==0 && p.h.solver.irsj==0 || throw(ArgumentError("IRS-J is a steady pseudo-time option; the physical step refuses it"))
    admit(p);isfinite(cfl) && 0<cfl<=.5 && max_dt>0 || throw(ArgumentError("Invalid physical clock setting"))
    saved=backup(p);old_dt=(p.c.solver.last_global_dt,p.h.solver.last_global_dt);limit=Float64(max_dt);attempted_dt=NaN
    for retry in 0:max_retries
        restore!(p,saved,old_dt)
        try
            cap=residual!(p)
            for q in (p.c,p.h);G.prepare_dt_ka!(q.solver,Float32(cfl));end
            dt=min(limit,cap,Float64(minimum(p.c.solver.dtf)),Float64(minimum(p.h.solver.dtf)))
            d=Float32(dt);Float64(d)>dt && (d=prevfloat(d));isfinite(d) && d>0 || throw(StageReject("Physical clock cannot advance"))
            attempted_dt=Float64(d)
            for q in (p.c,p.h);fill!(q.solver.dtf,d);P.snapshot!(q.solver);_sst_snapshot!(q);end
            floors=zeros(Float64,2)
            for (stage,(weight,other)) in enumerate(((1f0,0f0),(.25f0,.75f0),(2f0/3f0,1f0-2f0/3f0)))
                cap=residual!(p)
                Float64(d)<=cap*(1+8eps(Float32)) || throw(StageReject("SST stage clock decreased"))
                stage_observer===nothing || stage_observer(p,stage,:residual)
                for q in (p.c,p.h);P.update!(q.solver,weight,other);end
                for q in (p.c,p.h)
                    s=q.solver
                    all(isfinite,s.rho) && all(isfinite,s.ux) && all(isfinite,s.uy) && all(isfinite,s.T) && minimum(s.rho)>1f-4 && minimum(s.T)>1f-3 && maximum(hypot.(s.ux,s.uy))<3.49f0 || throw(StageReject("Native gas safeguard activated"))
                    pack_gas!(s.backend)(q.conserved.Utrial,s.rho,s.ux,s.uy,s.T,s.eint,Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=(s.NI,s.NJ))
                    _sst_trial!(q,weight;retryable=true,stage=stage)
                end
                for q in (p.c,p.h);_sst_commit!(q);end
                factor=stage==1 ? 1/6 : stage==2 ? 2/3 : 1.
                for q in (p.c,p.h)
                    floors[1]+=factor*sum(Float64.(Array(q.turbulence.kfloor)))
                    floors[2]+=factor*sum(Float64.(Array(q.turbulence.wpin)))
                end
                stage_observer===nothing || stage_observer(p,stage,:committed)
            end
            residual!(p)
            for q in (p.c,p.h);q.solver.last_global_dt=d;end
            p.elapsed+=Float64(d);p.accepted_steps+=1;p.last_rejections=retry
            p.flow.elapsed+=Float64(d);p.flow.iterations+=1
            return (dt=Float64(d),rejections=retry,weighted_k_floor=floors[1],weighted_omega_pin=floors[2])
        catch err
            restore!(p,saved,old_dt)
            if err isa StageReject && retry<max_retries
                isfinite(attempted_dt) && attempted_dt>0 || rethrow()
                limit=attempted_dt/2
                continue
            end
            rethrow()
        end
    end
    error("Unreachable physical trial exhaustion")
end
end
