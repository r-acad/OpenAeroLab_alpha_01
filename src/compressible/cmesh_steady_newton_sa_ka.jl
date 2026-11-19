# Spalart-Allmaras one-equation model for the Newton-Krylov airfoil march, single block and C+H pair (19 Nov 2026).
# EXPERIMENTAL AND OPT-IN: nothing changes unless set_newton_sa!(s) registered the solver; the SST paths are untouched.
# Included into module CMeshSolverKA from cmesh_steady_newton_ka.jl (after the pair file, whose stencil geometry it uses).
#
# Model: the standard SA model with the ft2 term (Spalart and Allmaras 1994; the "SA" of the NASA Turbulence Modeling
# Resource), with the modified vorticity of the TMR's note 1(c), which keeps S-tilde positive. Fully turbulent, except that
# production is multiplied by the solver's trip field gammat, as in the SST equations of this march. Kinematic
# (non-conservative) form, as those SST equations:
#   D nt/Dt = cb1 (1 - ft2) St nt - (cw1 fw - cb1 ft2 / kappa^2) (nt / d)^2 + (1/sigma) [div((nu + nt) grad nt) + cb2 |grad nt|^2]
#   nu_t = nt fv1,  fv1 = chi^3 / (chi^3 + cv1^3),  chi = nt / nu,  nu the local laminar viscosity.
# nt is kept at or above 1e-12 (the floor of k in the SST step), so the march's logarithmic state vector stays finite.
#
# Storage: nt in s.tk; k_inf = nt_inf = nut_ratio * nu_inf (3 by default, the TMR's free-stream value); w_inf =
# nt_inf / nu_t,inf, and s.tom = w_inf. The public boundary kernels (wall ghost = -interior, wake-cut copy, far field
# k_inf, far-field nu_t = k_inf / w_inf) then give the SA boundary values unchanged. The -2/3 rho k stress must be off.
#
# Discretisation: the scalar system of the SST step with isk = 1 (wall: nt = 0 on the face), solved by the same GMRES with
# line relaxation. Upwind advection; diffusion (nu + nt_face) / sigma as an M-matrix; production and the cb2 term explicit
# (non-negative sources); destruction in Newton form when its coefficient is positive (2 c nt0 / d^2 on the diagonal,
# c nt0^2 / d^2 in the source), otherwise a non-negative explicit source. At a fixed point the discrete SA equation holds.

const _SA_CB1=0.1355;const _SA_SIGMA=2/3;const _SA_CB2=0.622;const _SA_KAPPA=0.41
const _SA_CW2=0.3;const _SA_CW3=2.0;const _SA_CV1=7.1;const _SA_CT3=1.2;const _SA_CT4=0.5;const _SA_CV2=0.7;const _SA_CV3=0.9
const _SA_CW1=_SA_CB1/_SA_KAPPA^2+(1+_SA_CB2)/_SA_SIGMA
const _SA_FLOOR=1e-12

"fv1 of the SA model at chi = nt / nu."
@inline _sa_fv1(chi)=(c3=chi*chi*chi;c3/(c3+oftype(chi,_SA_CV1)^3))

# Coefficients of the SA scalar system: neighbour coefficients aW..aN, implicit diagonal dg, explicit source src and the
# local laminar viscosity nuc (for nu_t afterwards). Stencil geometry from arrays (NewtonPairSSTGeo), as the pair SST kernel.
@kernel function _k_newton_sa_coeffs!(aW,aE,aS,aN,dg,src,nuc,
    @Const(nt),@Const(rho),@Const(T),@Const(ux),@Const(uy),
    @Const(Six),@Const(Siy),@Const(Sjx),@Const(Sjy),@Const(vol),@Const(wdist),@Const(gammat),
    @Const(mxxi),@Const(myxi),@Const(mxet),@Const(myet),@Const(dE),@Const(dW),@Const(dN),@Const(dS),
    nu_inf,rho_inf,suth,air_sutherland::Int32,NI::Int32,NJ::Int32,g::Int32,cb2split::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI&&j<=NJ
        ia=i+g;ja=j+g
        nu=_air_nu_local(rho[ia,ja],T[ia,ja],nu_inf,rho_inf,suth,air_sutherland)
        R=typeof(nu);z=zero(R)
        cb1=R(_SA_CB1);sig=R(_SA_SIGMA);cb2=R(_SA_CB2);kap=R(_SA_KAPPA);cw1=R(_SA_CW1);cw2=R(_SA_CW2);cw3=R(_SA_CW3)
        ct3=R(_SA_CT3);ct4=R(_SA_CT4);cv2=R(_SA_CV2);cv3=R(_SA_CV3)
        n0=max(nt[ia,ja],z);d=max(wdist[i,j],R(1e-8))
        xxi=mxxi[i,j];yxi=myxi[i,j];xet=mxet[i,j];yet=myet[i,j]
        J=xxi*yet-xet*yxi;J=abs(J)<R(1e-30) ? R(1e-30) : J
        duxi=(ux[ia+1,ja]-ux[ia-1,ja])/2;duet=(ux[ia,ja+1]-ux[ia,ja-1])/2
        dvxi=(uy[ia+1,ja]-uy[ia-1,ja])/2;dvet=(uy[ia,ja+1]-uy[ia,ja-1])/2
        dudy=(-duxi*xet+duet*xxi)/J;dvdx=(dvxi*yet-dvet*yxi)/J
        Om=abs(dudy-dvdx)
        dnxi=(nt[ia+1,ja]-nt[ia-1,ja])/2;dnet=(nt[ia,ja+1]-nt[ia,ja-1])/2
        dndx=(dnxi*yet-dnet*yxi)/J;dndy=(-dnxi*xet+dnet*xxi)/J
        chi=n0/nu;fv1=_sa_fv1(chi);fv2=1-chi/(1+chi*fv1);ft2=ct3*exp(-ct4*chi*chi)
        kd2=kap*kap*d*d;Sb=n0*fv2/kd2
        St=Sb>=-cv2*Om ? Om+Sb : Om+Om*(cv2*cv2*Om+cv3*Sb)/((cv3-2*cv2)*Om-Sb)
        r=St>z ? min(n0/(St*kd2),R(10)) : R(10)
        gg=r+cw2*(r^6-r);fw=gg*((1+cw3^6)/(gg^6+cw3^6))^(one(R)/6)
        gm=gammat[i,j];P=cb1*(1-ft2)*St*n0*gm
        cdes=cw1*fw-cb1/(kap*kap)*ft2;D0=cdes*n0*n0/(d*d)
        # Exact derivatives with respect to nt (fw through r, ft2 and S-tilde through chi). Without the fw and r terms the
        # first wall cells alternate between two values from step to step, frozen flow or not (measured 19 Nov 2026).
        cv13=R(_SA_CV1)^3
        dfv1=3*chi*chi*cv13/(chi*chi*chi+cv13)^2/nu                        # d fv1 / d nt
        dfv2=-(1/nu-chi*dfv1)/(1+chi*fv1)^2                                 # d fv2 / d nt
        dSb=(fv2+n0*dfv2)/kd2
        dSt=Sb>=-cv2*Om ? dSb : Om*(cv3*((cv3-2*cv2)*Om-Sb)+(cv2*cv2*Om+cv3*Sb))/((cv3-2*cv2)*Om-Sb)^2*dSb
        dr=(St>z&&n0/(St*kd2)<R(10)) ? (1-n0*dSt/St)/(St*kd2) : z
        dfw=((1+cw3^6)/(gg^6+cw3^6))^(one(R)/6)*cw3^6/(gg^6+cw3^6)*(1+cw2*(6*r^5-1))*dr
        dft2=-2*ct4*chi*ft2/nu
        dD=(2*cdes*n0+(cw1*dfw-cb1/(kap*kap)*dft2)*n0*n0)/(d*d)
        dP=cb1*gm*((1-ft2)*(St+n0*dSt)-dft2*St*n0)
        lin=max(-dP,z)                                   # a production that falls with nt enters the diagonal
        if cdes>z
            # Newton form of the destruction, with a diagonal of at least D0 / nt so the source stays non-negative
            Dp=max(dD,D0/n0);dg[i,j]=Dp+lin;s=P+lin*n0+Dp*n0-D0
        else
            dg[i,j]=lin;s=P+lin*n0-D0
        end
        iv=1/vol[i,j]
        SfE=hypot(Six[i+1,j],Siy[i+1,j]);SfW=hypot(Six[i,j],Siy[i,j])
        SfN=hypot(Sjx[i,j+1],Sjy[i,j+1]);SfS=hypot(Sjx[i,j],Sjy[i,j])
        nE=max((nt[ia,ja]+nt[ia+1,ja])/2,z);nW=max((nt[ia,ja]+nt[ia-1,ja])/2,z)
        nN=max((nt[ia,ja]+nt[ia,ja+1])/2,z);nS=max((nt[ia,ja]+nt[ia,ja-1])/2,z)
        gE=SfE/dE[i,j]*iv;gW=SfW/dW[i,j]*iv;gN=SfN/dN[i,j]*iv;gS=SfS/dS[i,j]*iv
        if cb2split==Int32(1)
            # The form of CFL3D and FUN3D: (1+cb2)/sigma div((nu+nt) grad nt) - cb2/sigma (nu+nt_P) lap nt. A negative
            # neighbour coefficient (a steep front) is taken explicitly at the current nt, so the matrix stays an M-matrix.
            cP=cb2*(nu+n0);ex=z
            dE_=((1+cb2)*(nu+nE)-cP)/sig*gE;dW_=((1+cb2)*(nu+nW)-cP)/sig*gW
            dN_=((1+cb2)*(nu+nN)-cP)/sig*gN;dS_=((1+cb2)*(nu+nS)-cP)/sig*gS
            if dE_<z;ex+=dE_*(nt[ia+1,ja]-nt[ia,ja]);dE_=z;end
            if dW_<z;ex+=dW_*(nt[ia-1,ja]-nt[ia,ja]);dW_=z;end
            if dN_<z;ex+=dN_*(nt[ia,ja+1]-nt[ia,ja]);dN_=z;end
            if dS_<z;ex+=dS_*(nt[ia,ja-1]-nt[ia,ja]);dS_=z;end
            src[i,j]=s+ex
        else
            dE_=(nu+nE)/sig*gE;dW_=(nu+nW)/sig*gW;dN_=(nu+nN)/sig*gN;dS_=(nu+nS)/sig*gS
            src[i,j]=s+cb2/sig*(dndx*dndx+dndy*dndy)
        end
        u0=ux[ia,ja];v0=uy[ia,ja]
        Uci=(u0*(Six[i,j]+Six[i+1,j])/2+v0*(Siy[i,j]+Siy[i+1,j])/2)*iv
        Ucj=(u0*(Sjx[i,j]+Sjx[i,j+1])/2+v0*(Sjy[i,j]+Sjy[i,j+1])/2)*iv
        aE[i,j]=dE_+max(-Uci,z);aW[i,j]=dW_+max(Uci,z);aN[i,j]=dN_+max(-Ucj,z);aS[i,j]=dS_+max(Ucj,z)
        nuc[i,j]=nu
    end
end

"Settings and work arrays of a solver registered for SA (stencil geometry and the arrays of the residual probe)."
struct NewtonSAState{G,C,Q};geo::G;coef::C;scal::Q;nut_ratio::Float64;cb2::Symbol;end
const _NEWTON_SA=WeakKeyDict{Any,Any}()

"True when `s` was registered for the Spalart-Allmaras model by set_newton_sa!."
newton_sa_on(s)=haskey(_NEWTON_SA,s)

"""
    set_newton_sa!(s; nut_ratio=3.0, cb2=:source) -> s

Register the prepared solver `s` (SST settings on, see set_sst_ka!) for the Spalart-Allmaras model of the Newton march:
the turbulence step of newton_rans_step! and newton_pair_step! then solves SA instead of SST. Sets the free-stream
nt_inf = nut_ratio * nu_inf and fills the field with it. Call it on every solver of a cold start (the `make_solver`
functions) and before newton_pair_build or newton_time_load!.
`cb2`: the discrete form of the cb2 term. `:source` (the first form, 18 Nov 2026): cb2/sigma |grad nt|^2 from central
differences as an explicit source beside the (nu + nt)/sigma diffusion. `:split` (19 Nov 2026): the form of CFL3D and
FUN3D, (1+cb2)/sigma div((nu+nt) grad nt) - cb2/sigma (nu+nt) lap nt, implicit (negative neighbour coefficients explicit).
"""
function set_newton_sa!(s::KSol;nut_ratio::Real=3.0,cb2::Symbol=:source)
    cb2 in (:source,:split)||throw(ArgumentError("cb2 must be :source or :split"))
    s.sst||throw(ArgumentError("SA uses the eddy-viscosity path of the SST solver: call set_sst_ka! first"))
    s.keps==0&&s.rhok==Int32(0)&&s.earsm==Int32(0)&&s.lag==Int32(0)||
        throw(ArgumentError("SA needs the plain eddy-viscosity solver (no k-epsilon, -2/3 rho k stress, EARSM or lag model)"))
    nut_ratio>0||throw(ArgumentError("nut_ratio must be positive"))
    R=eltype(s.rho);nt_inf=Float64(nut_ratio)*Float64(s.nu_lam);nut_inf=nt_inf*_sa_fv1(Float64(nut_ratio))
    s.k_inf=R(nt_inf);s.w_inf=R(nt_inf/nut_inf)
    fill!(s.tk,s.k_inf);fill!(s.tom,s.w_inf);fill!(s.nut,R(nut_inf));fill!(s.tnl,R(nut_inf))
    scal=NewtonScalar(ntuple(_->similar(s.dtf),8)...,Int32[],0)
    _NEWTON_SA[s]=NewtonSAState(_newton_pair_sst_geo(s),newton_sst_init(s),scal,Float64(nut_ratio),cb2)
    _newton_sst_bc!(s)
    s
end

function _newton_sa_coeffs!(s::KSol,c::NewtonSST,q::NewtonPairSSTGeo)
    _k_newton_sa_coeffs!(s.backend)(c.akW,c.akE,c.akS,c.akN,c.dk,c.sk,c.sf2,
        s.tk,s.rho,s.T,s.ux,s.uy,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,s.wdist,s.gammat,
        q.xxi,q.yxi,q.xet,q.yet,q.dE,q.dW,q.dN,q.dS,s.nu_lam,s.rho_inf,s.suth,s.air_sutherland,
        Int32(s.NI),Int32(s.NJ),Int32(s.g),Int32(_NEWTON_SA[s].cb2===:split);ndrange=(s.NI,s.NJ))
end
function _newton_sa_system!(s::KSol,c::NewtonSST,q,invdt)
    _k_newton_sst_system!(s.backend)(q.cW,q.cE,q.cS,q.cN,q.cP,q.rhs,c.akW,c.akE,c.akS,c.akN,c.dk,c.sk,s.tk,invdt,
        s.rho,s.T,s.wdist,eltype(q.cP)(s.k_inf),Int32(1),s.nu_lam,s.rho_inf,s.suth,s.air_sutherland,
        Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF),Int32(s.g);ndrange=(s.NI,s.NJ))
end
# nu_t = nt fv1(nt / nu) on the interior, relaxed by nut_relax and capped as in the SST step.
function _newton_sa_nut!(nutv,n,nuc,nut_relax)
    @inbounds for k in eachindex(n)
        nu=Float64(nuc[k]);nt=n[k]*_sa_fv1(n[k]/nu)
        nutv[k]+=nut_relax*(min(nt,0.15)-nutv[k])
    end
end

"""Implicit SA step with line relaxation (the single-block counterpart of newton_sst_implicit_lines!, same arguments).
`bdf`: as there (`newton_time_sst_terms`); its k levels hold nt. Returns the fields of the SST step (dw, wits and wlin 0)."""
function newton_sa_implicit_lines!(s::KSol,c::NewtonSST,q::NewtonScalar,invdt;relax=1.0,nut_relax=1.0,tol=1e-3,bdf=nothing)
    st=_NEWTON_SA[s];NI,NJ,g=s.NI,s.NJ,s.g;bk=s.backend
    _newton_sst_bc!(s);_newton_sa_coeffs!(s,c,st.geo)
    ir=g+1:g+NI;jr=g+1:g+NJ;nold=Array(view(s.tk,ir,jr))
    _newton_sa_system!(s,c,q,invdt)
    bdf===nothing||_k_newton_sst_bdf!(bk)(q.cP,q.rhs,bdf.kn,bdf.km,eltype(q.cP)(bdf.a0),eltype(q.cP)(bdf.a1),eltype(q.cP)(bdf.a2),
        Int32(1),Int32(NI),Int32(NJ),Int32(s.NIW),Int32(s.NIF);ndrange=(NI,NJ))
    nnew,its,lin=newton_scalar_solve!(s,q,nold;tol)
    nnew.=clamp.(nold.+relax.*(nnew.-nold),_SA_FLOOR,1.0)
    dmax=maximum(abs.(nnew.-nold)./max.(abs.(nold),Float64(s.k_inf)))
    view(s.tk,ir,jr).=nnew
    _newton_sa_nut!(view(s.nut,ir,jr),nnew,c.sf2,nut_relax)
    _newton_sst_bc!(s)
    (;dkmax=dmax,dwmax=0.0,kits=its,wits=0,klin=lin,wlin=0.0)
end

"RMS of the steady SA equation residual (no pseudo-time term) of a registered single-block solver."
function newton_sa_equation_l2(s::KSol)
    st=_NEWTON_SA[s];g=s.g;ir=g+1:g+s.NI;jr=g+1:g+s.NJ;q=st.scal
    _newton_sst_bc!(s);_newton_sa_coeffs!(s,st.coef,st.geo)
    _newton_sa_system!(s,st.coef,q,fill!(similar(s.dtf),0))
    _k_newton_scalar_residual!(s.backend)(q.r,q.rhs,Array(view(s.tk,ir,jr)),q.cW,q.cE,q.cS,q.cN,q.cP,
        Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF);ndrange=(s.NI,s.NJ))
    sqrt(sum(abs2,q.r)/length(q.r))
end

# ---- the C+H pair ---------------------------------------------------------------------------------------
"""Implicit SA step on both blocks of the pair (the counterpart of newton_pair_sst_step!, same arguments; both blocks must be
registered). The interface terms are those of the SST k system."""
function newton_pair_sa_step!(p::NewtonPair,m::NewtonPairState;relax=1.0,nut_relax=1.0,tol=1e-3,bdf=nothing)
    c,h=p.c,p.h
    newton_sa_on(h)||throw(ArgumentError("register both blocks of the pair with set_newton_sa!"))
    newton_pair_sst_bc!(p);_newton_sa_coeffs!(c,m.sstc,p.gc);_newton_sa_coeffs!(h,m.ssth,p.gh)
    int(s,a)=Array(view(a,s.g+1:s.g+s.NI,s.g+1:s.g+s.NJ))
    nc,nh=int(c,c.tk),int(h,h.tk)
    _newton_pair_sst_systems!(p,m,1,m.invdtc,m.invdth)
    bdf===nothing||for (s,q,b) in ((c,m.scalarc,bdf[1]),(h,m.scalarh,bdf[2]))
        T=eltype(q.cP)
        _k_newton_sst_bdf!(s.backend)(q.cP,q.rhs,b.kn,b.km,T(b.a0),T(b.a1),T(b.a2),Int32(1),Int32(s.NI),Int32(s.NJ),
            Int32(s.NIW),Int32(s.NIF);ndrange=(s.NI,s.NJ))
    end
    ncn,nhn,its,lin=_newton_pair_scalar_solve!(p,m,nc,nh;tol)
    dmax=0.0
    for (s,cf,no,nn) in ((c,m.sstc,nc,ncn),(h,m.ssth,nh,nhn))
        nn.=clamp.(no.+relax.*(nn.-no),_SA_FLOOR,1.0)
        dmax=max(dmax,maximum(abs.(nn.-no)./max.(abs.(no),Float64(s.k_inf))))
        ir=s.g+1:s.g+s.NI;jr=s.g+1:s.g+s.NJ
        view(s.tk,ir,jr).=nn;_newton_sa_nut!(view(s.nut,ir,jr),nn,cf.sf2,nut_relax)
    end
    newton_pair_sst_bc!(p)
    (;dkmax=dmax,dwmax=0.0,kits=its,wits=0,klin=lin,wlin=0.0)
end
"RMS of the steady SA equation residual over both blocks of the pair, and 0 in place of the omega entry."
function newton_pair_sa_equation_l2(p::NewtonPair,m::NewtonPairState)
    c,h=p.c,p.h
    newton_pair_sst_bc!(p);_newton_sa_coeffs!(c,m.sstc,p.gc);_newton_sa_coeffs!(h,m.ssth,p.gh)
    int(s,a)=Array(view(a,s.g+1:s.g+s.NI,s.g+1:s.g+s.NJ));zc=fill!(similar(c.dtf),0);zh=fill!(similar(h.dtf),0)
    _newton_pair_sst_systems!(p,m,1,zc,zh);rc=similar(zc);rh=similar(zh)
    _newton_pair_scalar_residual!(rc,rh,m.scalarc.rhs,m.scalarh.rhs,int(c,c.tk),int(h,h.tk),p,m)
    [sqrt((sum(abs2,rc)+sum(abs2,rh))/(length(rc)+length(rh))),0.0]
end
