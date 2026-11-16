# Experimental pure thermal-gas / conservative SST-2003m C+H adapter.
# Shared native gas energy flux remains thermal; no mixture EOS projection,
# vapour transport, Reboud correction, SFD or local pseudo-time is called.
#
# Two options, both off by default:
# - Viscosity law. The blocks carry CMeshSolverKA's law (set_air_sutherland_ka!, set_air_constant_dynamic_ka!). The gas
#   kernels take it natively; the SST transport then runs the `gas_*` kernels below, which are the species kernels'
#   pair branches with the local viscosity mu(T)/rho in place of the constant. With constant kinematic viscosity the
#   species kernels are called exactly as before.
# - Prescribed trips. build_coupled(...; intermittency=true) multiplies both SST productions on the C block by the
#   metrics' trip intermittency `gammat` (0 ahead of the trip station of each side, a linear ramp over 0.02 c, then 1),
#   the single-block solver's mechanism. Without it `gammat` is not read and the model is fully turbulent.
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
    intermittency::Bool   # the SST productions of this block are multiplied by the metrics' trip intermittency
end

mutable struct CoupledPair{P,B}
    flow::P
    c::B
    h::B
    accepted_steps::Int
    elapsed::Float64
    last_rejections::Int
end

function block(s,ctx,name,intermittency)
    _admit_sst(s)
    a=_build_sst(s,nothing)
    z()=fill!(similar(s.rho,s.NI,s.NJ),0.0f0)
    ghost()=fill!(similar(s.rho),0.0f0)
    packed=fill!(similar(s.rho,Float64,s.NI,s.NJ,4),0.0)
    SSTBlock(s,a,ctx,z(),z(),ghost(),(alpha=ghost(),rho_l_sat=1f0,rho_v_sat=0f0),
        nothing,Int32(2),0.0f0,0.0f0,Int32(0),Int32(0),(0.0f0,0.0f0),(Utrial=packed,),name,intermittency)
end

# The legacy positive `suth` without a law (a kinematic Sutherland factor in the gas flux only) has no SST counterpart.
_sst_law_admitted(s)=s.air_sutherland!=0 || s.suth==0

function admit(p::CoupledPair)
    P.admit_settings(p.flow.c,p.flow.h,p.flow.wallmode;allow_sst=true)
    p.flow.wallmode==2 || throw(ArgumentError("Conservative SST adapter requires no-slip wall"))
    for q in (p.c,p.h)
        s=q.solver
        s.sst && s.nu_lam>0 && s.fullvisc==1 && all(iszero,(s.keps,s.lag,s.earsm,s.rhok,s.fbm_cdelta,s.pans_fk)) || throw(ArgumentError("Unsupported SST extension or stress mode"))
        _sst_law_admitted(s) || throw(ArgumentError("Conservative SST takes constant kinematic viscosity or a viscosity law (Sutherland, constant dynamic); a bare Sutherland ratio is not admitted"))
        # q.trip is the species path's k-floor trigger, which this pure-gas pair never uses: a prescribed trip is the
        # production intermittency selected at build_coupled.
        q.reboud_n==0 && q.reboud_form==0 && q.reboud_prod==0 && q.trip==(0f0,0f0) || throw(ArgumentError("Pure gas pair: no Reboud correction and no k-floor trip"))
    end
    !p.h.intermittency || throw(ArgumentError("The H strip has only the base wall; trip intermittency applies to the C block"))
    p.c.solver.k_inf===p.h.solver.k_inf && p.c.solver.w_inf===p.h.solver.w_inf || throw(ArgumentError("SST inflow settings differ"))
end

function build_coupled(p::P.GasPair;tu_inf::Real=.001,mu_t_ratio::Real=.1,intermittency::Bool=false)
    P.admit_settings(p.c,p.h,p.wallmode)
    p.iterations==0 && p.elapsed==0 || throw(ArgumentError("SST adapter requires a fresh gas pair; no hidden clock-origin reset"))
    all(_sst_law_admitted,(p.c,p.h)) || throw(ArgumentError("SST adapter takes constant kinematic viscosity or a viscosity law; a bare Sutherland ratio is not admitted"))
    p.wallmode==2 || throw(ArgumentError("SST adapter requires no-slip walls"))
    0<tu_inf<.1 && mu_t_ratio>0 && isfinite(mu_t_ratio) || throw(ArgumentError("Invalid SST freestream"))
    for s in (p.c,p.h);G.set_sst_ka!(s;tu_inf=Float64(tu_inf),mu_t_ratio=Float64(mu_t_ratio));end
    sig()=fill!(similar(p.c.rho,Float32,p.c.NIW,2,2),0.0f0)
    c=block(p.c,(p.ctx_c...,sig_nb=sig()),"C",intermittency)
    h=block(p.h,(p.ctx_h...,sig_nb=sig()),"H",false)
    result=CoupledPair(p,c,h,0,0.0,0)
    admit(result);residual!(result)
    result
end

# The viscosity law of a block in the form the kernels take (CMeshSolverKA._air_nu_local): freestream kinematic viscosity,
# freestream density, Sutherland temperature ratio and mode (0 constant kinematic, 1 Sutherland, 2 constant dynamic).
viscosity_law(s)=(nu=s.nu_lam,rho_inf=s.rho_inf,suth=s.suth,mode=s.air_sutherland)
variable_viscosity(q)=q.solver.air_sutherland!=Int32(0)
# Local kinematic molecular viscosity mu(T)/rho; the constant itself with law 0.
@inline nu_at(r,t,l)=G._air_nu_local(r,t,l.nu,l.rho_inf,l.suth,l.mode)

# SST-2003m coefficients of a gas block with a viscosity law and/or trip intermittency: _k_sst_species_coeff! without its
# wall-function and Reboud arms, with the cell's own viscosity in the blending functions and `gammat` on both productions.
@kernel function gas_coeff!(nut,sigk,sigw,sk,sw,rate,status,@Const(rho),@Const(T),@Const(tk),@Const(tom),
                            @Const(ux),@Const(uy),@Const(wdist),@Const(gammat),
                            @Const(Six),@Const(Siy),@Const(Sjx),@Const(Sjy),@Const(vol),
                            l,trip::Int32,NI::Int32,NJ::Int32,g::Int32,ctx)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        ia=i+g;ja=j+g
        uxg,uyg=_sst_gradient(ux,i,j,g,Six,Siy,Sjx,Sjy,vol)
        vxg,vyg=_sst_gradient(uy,i,j,g,Six,Siy,Sjx,Sjy,vol)
        kx,ky=_sst_gradient(tk,i,j,g,Six,Siy,Sjx,Sjy,vol)
        wx,wy=_sst_gradient(tom,i,j,g,Six,Siy,Sjx,Sjy,vol)
        if G._strip_gg_cell(i,j,NJ,ctx)
            sg8=_strip_sst_gradients(ux,uy,tk,tom,Six,Siy,Sjx,Sjy,vol,Int32(i),Int32(j),NJ,g,ctx)
            uxg=sg8[1];uyg=sg8[2];vxg=sg8[3];vyg=sg8[4];kx=sg8[5];ky=sg8[6];wx=sg8[7];wy=sg8[8]
        end
        n,ak,aw,Sk,Sw,R=_sst_coefficients(tk[ia,ja],tom[ia,ja],max(wdist[i,j],1.0f-8),nu_at(rho[ia,ja],T[ia,ja],l),
            rho[ia,ja],uxg,uyg,vxg,vyg,kx*wx+ky*wy,1.0f0,trip==Int32(1) ? gammat[i,j] : 1.0f0)
        rv=rho[ia,ja]*vol[i,j];Sk*=rv;Sw*=rv
        good=isfinite(n) && n>=0.0f0 && isfinite(ak) && isfinite(aw) &&
            isfinite(Sk) && isfinite(Sw) && isfinite(R) && R>=0.0f0
        nut[ia,ja]=n
        sigk[i,j]=ak;sigw[i,j]=aw;sk[i,j]=Sk;sw[i,j]=Sw;rate[i,j]=R
        status[i,j]=good ? 0.0f0 : 1.0f0
    end
end

# The wall value of omega, 60 nu_w/(beta1 d1^2), with the wall cell's own viscosity. Rewrites the omega ghosts that
# _k_sst_ghosts! formed with the freestream value: the C block's foil rows and the strip's base columns over its
# interior rows (the strip's corner ghosts hold interface-row values and carry no wall value).
@kernel function gas_wall_omega!(tom,@Const(rho),@Const(T),@Const(Six),@Const(Siy),@Const(Sjx),@Const(Sjy),@Const(vol),
                                 l,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32,block::Int32)
    n,k=@index(Global,NTuple)
    @inbounds if k<=g
        if block==Int32(0) && NIW<n<=NIW+NIF
            i=Int32(n);ia=i+g
            S=sqrt(Sjx[i,1]^2+Sjy[i,1]^2);d1=0.5f0*vol[i,1]/S+1.0f-30
            tom[ia,g+Int32(1)-Int32(k)]=2.0f0*_sst_wall_omega(nu_at(rho[ia,g+1],T[ia,g+1],l),d1)-tom[ia,g+Int32(k)]
        elseif block==Int32(1) && n<=NJ
            j=Int32(n);jj=j+g
            S=sqrt(Six[1,j]^2+Siy[1,j]^2);d1=0.5f0*vol[1,j]/S+1.0f-30
            tom[g+Int32(1)-Int32(k),jj]=2.0f0*_sst_wall_omega(nu_at(rho[g+1,jj],T[g+1,jj],l),d1)-tom[g+Int32(k),jj]
        end
    end
end

# Two-point SST diffusion conductances of a pair block with a viscosity law: _sst_i_diff_strip and _sst_j_diff_strip of
# cmesh_species_sst.jl, operand for operand, with the molecular part mu(T_f)/rho_f at the face (the face viscosity of the
# gas flux) and the wall cell's own viscosity at the two walls.
@inline function gas_i_diff(rho,T,nut,sigk,sigw,xc,yc,Six,Siy,l,i,j,NI,g,ctx)
    ja=j+g
    if ctx.iwall==Int32(1) && i==1
        ia=g+Int32(1);S=sqrt(Six[1,j]^2+Siy[1,j]^2)
        dn=0.5f0*ctx.vol[1,j]/S+1.0f-30;v=rho[ia,ja]*nu_at(rho[ia,ja],T[ia,ja],l)*S/dn
        return (v,v) # the base wall: wall eddy viscosity is zero
    elseif 1<i<=NI
        ia=i+g
        dx=xc[i,j]-xc[i-1,j];dy=yc[i,j]-yc[i-1,j]
        dn=sqrt(dx*dx+dy*dy)+1.0f-30;S=sqrt(Six[i,j]^2+Siy[i,j]^2)
        rf=0.5f0*(rho[ia-1,ja]+rho[ia,ja]);factor=rf*S/dn
        nu=nu_at(rf,0.5f0*(T[ia-1,ja]+T[ia,ja]),l)
        return (factor*(nu+0.5f0*(sigk[i-1,j]*nut[ia-1,ja]+sigk[i,j]*nut[ia,ja])),
                factor*(nu+0.5f0*(sigw[i-1,j]*nut[ia-1,ja]+sigw[i,j]*nut[ia,ja])))
    end
    (0.0f0,0.0f0) # no diffusive flux at the open i ends
end
@inline function gas_j_diff(rho,T,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,l,i,j,NI,NJ,NIW,NIF,g,ctx)
    ia=i+g;S=sqrt(Sjx[i,j]^2+Sjy[i,j]^2)
    if j==1 && (i<=NIW || i>NIW+NIF)
        # an interface face of the C block's leg columns or of the strip: the other block's cell is the ghost
        st,fc=G._strip_station(i,NIW,NIF,ctx.block)
        dx=xc[i,1]-ctx.c_nb[st,1,fc];dy=yc[i,1]-ctx.c_nb[st,2,fc]
        dn=sqrt(dx*dx+dy*dy)+1.0f-30
        rf=0.5f0*(rho[ia,g]+rho[ia,g+1]);factor=rf*S/dn
        nu=nu_at(rf,0.5f0*(T[ia,g]+T[ia,g+1]),l)
        return (factor*(nu+0.5f0*(ctx.sig_nb[st,1,fc]*nut[ia,g]+sigk[i,1]*nut[ia,g+1])),
                factor*(nu+0.5f0*(ctx.sig_nb[st,2,fc]*nut[ia,g]+sigw[i,1]*nut[ia,g+1])))
    elseif ctx.block==Int32(1) && j==NJ+1
        # the strip's upper interface
        dx=ctx.c_nb[i,1,2]-xc[i,NJ];dy=ctx.c_nb[i,2,2]-yc[i,NJ]
        dn=sqrt(dx*dx+dy*dy)+1.0f-30
        rf=0.5f0*(rho[ia,NJ+g]+rho[ia,NJ+g+1]);factor=rf*S/dn
        nu=nu_at(rf,0.5f0*(T[ia,NJ+g]+T[ia,NJ+g+1]),l)
        return (factor*(nu+0.5f0*(sigk[i,NJ]*nut[ia,NJ+g]+ctx.sig_nb[i,1,2]*nut[ia,NJ+g+1])),
                factor*(nu+0.5f0*(sigw[i,NJ]*nut[ia,NJ+g]+ctx.sig_nb[i,2,2]*nut[ia,NJ+g+1])))
    elseif 1<j<=NJ
        ja=j+g
        dx=xc[i,j]-xc[i,j-1];dy=yc[i,j]-yc[i,j-1]
        dn=sqrt(dx*dx+dy*dy)+1.0f-30;rf=0.5f0*(rho[ia,ja-1]+rho[ia,ja]);factor=rf*S/dn
        nu=nu_at(rf,0.5f0*(T[ia,ja-1]+T[ia,ja]),l)
        return (factor*(nu+0.5f0*(sigk[i,j-1]*nut[ia,ja-1]+sigk[i,j]*nut[ia,ja])),
                factor*(nu+0.5f0*(sigw[i,j-1]*nut[ia,ja-1]+sigw[i,j]*nut[ia,ja])))
    elseif j==1
        dn=0.5f0*vol[i,1]/S+1.0f-30;v=rho[ia,g+1]*nu_at(rho[ia,g+1],T[ia,g+1],l)*S/dn
        return (v,v) # the foil wall: wall eddy viscosity is zero
    end
    (0.0f0,0.0f0) # outer boundary
end

# Scalar face fluxes of a pair block with a viscosity law: the two-point branches of _k_sst_species_flux_i!/_j! (first-order
# donor advection on the carrier mass flux, no reconstruction or non-orthogonal correction, as this adapter calls them).
@kernel function gas_flux_i!(Fk,Fw,@Const(Fmass),@Const(rk),@Const(rw),@Const(rho),@Const(T),@Const(tk),@Const(tom),@Const(nut),
                             @Const(sigk),@Const(sigw),@Const(xc),@Const(yc),@Const(Six),@Const(Siy),
                             l,ki::Float32,wi::Float32,NI::Int32,NJ::Int32,g::Int32,ctx)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI+1 && j<=NJ
        ia=i+g;ja=j+g;f=Fmass[i,j];id=f>=0.0f0 ? ia-1 : ia
        wall=ctx.iwall==Int32(1) && i==1
        incoming=!wall && ((i==1 && f>0.0f0) || (i==NI+1 && f<0.0f0))
        kd=incoming ? ki : _sst_donor(rk,tk,rho,id,ja,NI,NJ,g)
        wd=incoming ? wi : _sst_donor(rw,tom,rho,id,ja,NI,NJ,g)
        ck,cw=gas_i_diff(rho,T,nut,sigk,sigw,xc,yc,Six,Siy,l,i,j,NI,g,ctx)
        kl=tk[ia-1,ja];wl=tom[ia-1,ja]
        if wall
            S=sqrt(Six[1,j]^2+Siy[1,j]^2);dn=0.5f0*ctx.vol[1,j]/S+1.0f-30
            kl=0.0f0;wl=_sst_wall_omega(nu_at(rho[ia,ja],T[ia,ja],l),dn)
        end
        Fk[i,j]=f*kd-ck*(tk[ia,ja]-kl)
        Fw[i,j]=f*wd-cw*(tom[ia,ja]-wl)
    end
end
@kernel function gas_flux_j!(Fk,Fw,@Const(Fmass),@Const(rk),@Const(rw),@Const(rho),@Const(T),@Const(tk),@Const(tom),@Const(nut),
                             @Const(sigk),@Const(sigw),@Const(xc),@Const(yc),@Const(Sjx),@Const(Sjy),@Const(vol),@Const(kinds),
                             l,ki::Float32,wi::Float32,NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32,ctx)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ+1
        ia=i+g;ja=j+g;f=Fmass[i,j];jd=f>=0.0f0 ? ja-1 : ja
        incoming=j==NJ+1 && kinds[i]==0x01 && f<0.0f0
        kd=incoming ? ki : _sst_donor(rk,tk,rho,ia,jd,NI,NJ,g)
        wd=incoming ? wi : _sst_donor(rw,tom,rho,ia,jd,NI,NJ,g)
        kl=tk[ia,ja-1];wl=tom[ia,ja-1]
        if j==1 && NIW<i<=NIW+NIF
            S=sqrt(Sjx[i,1]^2+Sjy[i,1]^2);dn=0.5f0*vol[i,1]/S+1.0f-30
            kl=0.0f0;wl=_sst_wall_omega(nu_at(rho[ia,g+1],T[ia,g+1],l),dn)
        end
        ck,cw=gas_j_diff(rho,T,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,l,i,j,NI,NJ,NIW,NIF,g,ctx)
        Fk[i,j]=f*kd-ck*(tk[ia,ja]-kl)
        Fw[i,j]=f*wd-cw*(tom[ia,ja]-wl)
    end
end

# Finite-volume scalar residual and positivity clock of a pair block with a viscosity law: _k_sst_species_residual! in the
# form this adapter calls it (no reconstructed faces, no stored diffusion terms), with the conductances above.
@kernel function gas_scalar_residual!(dck,dcw,cap,@Const(Fik),@Const(Fiw),@Const(Fjk),@Const(Fjw),@Const(Fmi),@Const(Fmj),
                                      @Const(sk),@Const(sw),@Const(rate),@Const(rho),@Const(T),@Const(nut),@Const(sigk),@Const(sigw),
                                      @Const(xc),@Const(yc),@Const(Six),@Const(Siy),@Const(Sjx),@Const(Sjy),@Const(vol),
                                      l,NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32,ctx)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        dck[i,j]=Fik[i+1,j]-Fik[i,j]+Fjk[i,j+1]-Fjk[i,j]-sk[i,j]
        dcw[i,j]=Fiw[i+1,j]-Fiw[i,j]+Fjw[i,j+1]-Fjw[i,j]-sw[i,j]
        ek,ew=gas_i_diff(rho,T,nut,sigk,sigw,xc,yc,Six,Siy,l,i+1,j,NI,g,ctx)
        wk,ww=gas_i_diff(rho,T,nut,sigk,sigw,xc,yc,Six,Siy,l,i,j,NI,g,ctx)
        nk,nw=gas_j_diff(rho,T,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,l,i,j+1,NI,NJ,NIW,NIF,g,ctx)
        skc,swc=gas_j_diff(rho,T,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,l,i,j,NI,NJ,NIW,NIF,g,ctx)
        out=max(Fmi[i+1,j],0.0f0)+max(-Fmi[i,j],0.0f0)+max(Fmj[i,j+1],0.0f0)+max(-Fmj[i,j],0.0f0)
        mass=rho[i+g,j+g]*vol[i,j]
        R=(out+max(ek+wk+nk+skc,ew+ww+nw+swc))/mass+rate[i,j]
        good=isfinite(mass) && mass>0.0f0 && isfinite(R) && R>=0.0f0 && isfinite(dck[i,j]) && isfinite(dcw[i,j])
        cap[i,j]=good ? (R>0.0f0 ? 0.25f0/R : Inf32) : 0.0f0
    end
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
        if variable_viscosity(q)
            gas_wall_omega!(s.backend)(s.tom,s.rho,s.T,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,viscosity_law(s),
                Int32(s.NJ),Int32(s.NIW),Int32(s.NIF),Int32(s.g),q.ctx.block;ndrange=(max(s.NI,s.NJ),s.g))
        end
    end
    exchange_sst!(p,0)
    for q in (p.c,p.h)
        s=q.solver;a=q.turbulence;fill!(q.status,0f0)
        if variable_viscosity(q) || q.intermittency
            gas_coeff!(s.backend)(s.nut,a.sigk,a.sigw,a.sk,a.sw,a.rate,q.status,s.rho,s.T,s.tk,s.tom,
                s.ux,s.uy,s.wdist,s.gammat,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,viscosity_law(s),Int32(q.intermittency),
                Int32(s.NI),Int32(s.NJ),Int32(s.g),q.ctx;ndrange=(s.NI,s.NJ))
        else
            pressure_cache!(s.backend)(q.pressure,s.rho,s.T,Int32(size(s.rho,1)),Int32(size(s.rho,2));ndrange=size(s.rho))
            # wall function off (0); production form 0, the strain-squared default of the species SST (its study option
            # TOPO_SPECIES_SST_PRODUCTION is for cavitating mixtures and is not read for this pure-gas pair)
            _k_sst_species_coeff!(s.backend)(s.nut,a.sigk,a.sigw,a.sk,a.sw,a.rate,q.status,s.rho,q.tem.alpha,q.pressure,s.tk,s.tom,
                s.ux,s.uy,s.wdist,s.gammat,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,s.nu_lam,0f0,Int32(0),Int32(0),1f0,0f0,
                (0f0,0f0,0f0,0f0),Int32(s.NI),Int32(s.NJ),Int32(s.g),Int32(s.NIW),Int32(s.NIF),Int32(0),Int32(0),q.ctx;ndrange=(s.NI,s.NJ))
        end
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
    if variable_viscosity(q)
        l=viscosity_law(s)
        gas_flux_i!(s.backend)(a.Fik,a.Fiw,s.Fi_r,a.rk,a.rw,s.rho,s.T,s.tk,s.tom,s.nut,a.sigk,a.sigw,
            s.xc,s.yc,s.Six,s.Siy,l,s.k_inf,s.w_inf,ni,nj,g,q.ctx;ndrange=(s.NI+1,s.NJ))
        gas_flux_j!(s.backend)(a.Fjk,a.Fjw,s.Fj_r,a.rk,a.rw,s.rho,s.T,s.tk,s.tom,s.nut,a.sigk,a.sigw,
            s.xc,s.yc,s.Sjx,s.Sjy,s.vol,s.outer_face_kind,l,s.k_inf,s.w_inf,ni,nj,Int32(s.NIW),Int32(s.NIF),g,q.ctx;ndrange=(s.NI,s.NJ+1))
        return nothing
    end
    _k_sst_species_flux_i!(s.backend)(a.Fik,a.Fiw,s.Fi_r,a.rk,a.rw,s.rho,s.tk,s.tom,s.nut,a.sigk,a.sigw,
        s.xc,s.yc,s.Six,s.Siy,s.nu_lam,s.k_inf,s.w_inf,nothing,nothing,nothing,nothing,ni,nj,g,0f0,q.ctx;ndrange=(s.NI+1,s.NJ))
    _k_sst_species_flux_j!(s.backend)(a.Fjk,a.Fjw,s.Fj_r,a.rk,a.rw,s.rho,s.tk,s.tom,s.nut,a.sigk,a.sigw,
        s.xc,s.yc,s.Sjx,s.Sjy,s.vol,s.outer_face_kind,s.nu_lam,s.k_inf,s.w_inf,Int32(2),nothing,nothing,nothing,nothing,
        ni,nj,Int32(s.NIW),Int32(s.NIF),g,Int32(0),0f0,q.ctx;ndrange=(s.NI,s.NJ+1))
end

function scalar_residual!(q)
    s=q.solver;a=q.turbulence
    if variable_viscosity(q)
        gas_scalar_residual!(s.backend)(a.dck,a.dcw,q.source_dt,a.Fik,a.Fiw,a.Fjk,a.Fjw,s.Fi_r,s.Fj_r,
            a.sk,a.sw,a.rate,s.rho,s.T,s.nut,a.sigk,a.sigw,s.xc,s.yc,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,viscosity_law(s),
            Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF),Int32(s.g),q.ctx;ndrange=(s.NI,s.NJ))
    else
        # clock factor 0.25, the kernel's fixed value before the species study option TOPO_SPECIES_SST_CLOCK_FACTOR
        _k_sst_species_residual!(s.backend)(a.dck,a.dcw,q.source_dt,a.Fik,a.Fiw,a.Fjk,a.Fjw,s.Fi_r,s.Fj_r,
            a.sk,a.sw,a.rate,a.rk,a.rw,nothing,nothing,s.rho,s.nut,a.sigk,a.sigw,s.xc,s.yc,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,s.nu_lam,Int32(2),
            Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF),Int32(s.g),Int32(0),0f0,0.25f0,q.ctx;ndrange=(s.NI,s.NJ))
    end
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
