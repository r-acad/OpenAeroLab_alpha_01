# Implicit SST k-omega step of the Newton-Krylov airfoil march (cmesh_steady_newton_ka.jl). Experimental and opt-in:
# nothing here is called by the default solver paths. Included into module CMeshSolverKA.
#
# Backward Euler with a local pseudo-time diagonal on the same discrete equations as the explicit sweep
# (_k_sst_sweep!, plain SST branch, orthogonal diffusion): at a fixed point the public raw residual is zero.
# Destruction is implicit, production explicit, advection upwind and diffusion an M-matrix, so k and omega stay
# positive. Wall omega is prescribed in the first two rows. The two scalar systems are solved by GMRES with
# alternating zebra line relaxation; the eddy viscosity is then algebraic.

@kernel function _k_newton_sst_coeffs!(akW,akE,akS,akN,awW,awE,awS,awN,dk,sk,dw,sw,sf2,
    @Const(tk),@Const(tom),@Const(nut),@Const(rho),@Const(T),@Const(ux),@Const(uy),@Const(xc),@Const(yc),
    @Const(Six),@Const(Siy),@Const(Sjx),@Const(Sjy),@Const(vol),@Const(wdist),@Const(gammat),
    nu_inf,rho_inf,suth,air_sutherland::Int32,NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32,
    corrected_wall_metric::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI&&j<=NJ
        βs=0.09f0;a1=0.31f0
        σk1=0.85f0;σw1=0.5f0;β1=0.075f0;γ1=0.5532f0
        σk2=1.0f0;σw2=0.856f0;β2=0.0828f0;γ2=0.4403f0
        ia=i+g;ja=j+g
        nu=_air_nu_local(rho[ia,ja],T[ia,ja],nu_inf,rho_inf,suth,air_sutherland)
        k0=tk[ia,ja];w0=max(tom[ia,ja],1.0f-6);d=max(wdist[i,j],1.0f-8)
        xxi,yxi,xet,yet=_sst_index_metrics(xc,yc,Sjx,Sjy,wdist,i,j,NI,NJ,NIW,NIF,corrected_wall_metric)
        J=xxi*yet-xet*yxi;J=abs(J)<1.0f-30 ? oftype(J,1.0f-30) : J
        duxi=0.5f0*(ux[ia+1,ja]-ux[ia-1,ja]);duet=0.5f0*(ux[ia,ja+1]-ux[ia,ja-1])
        dvxi=0.5f0*(uy[ia+1,ja]-uy[ia-1,ja]);dvet=0.5f0*(uy[ia,ja+1]-uy[ia,ja-1])
        dudx=(duxi*yet-duet*yxi)/J;dudy=(-duxi*xet+duet*xxi)/J
        dvdx=(dvxi*yet-dvet*yxi)/J;dvdy=(-dvxi*xet+dvet*xxi)/J
        Sxy=0.5f0*(dudy+dvdx);S2=2.0f0*(dudx*dudx+dvdy*dvdy+2.0f0*Sxy*Sxy);Ssr=sqrt(S2)
        Ω=abs(dudy-dvdx)
        sqk=sqrt(max(k0,0.0f0))
        dkxi=0.5f0*(tk[ia+1,ja]-tk[ia-1,ja]);dket=0.5f0*(tk[ia,ja+1]-tk[ia,ja-1])
        dwxi=0.5f0*(tom[ia+1,ja]-tom[ia-1,ja]);dwet=0.5f0*(tom[ia,ja+1]-tom[ia,ja-1])
        dkdx=(dkxi*yet-dket*yxi)/J;dkdy=(-dkxi*xet+dket*xxi)/J
        dwdx=(dwxi*yet-dwet*yxi)/J;dwdy=(-dwxi*xet+dwet*xxi)/J
        gkw=dkdx*dwdx+dkdy*dwdy
        CDkw=max(2.0f0*σw2/w0*gkw,1.0f-10)
        arg1=min(max(sqk/(βs*w0*d),500.0f0*nu/(d*d*w0)),4.0f0*σw2*k0/(CDkw*d*d))
        arg1=min(arg1,4.0f0);F1=tanh(arg1^4)
        arg2=min(max(2.0f0*sqk/(βs*w0*d),500.0f0*nu/(d*d*w0)),4.0f0);F2=tanh(arg2^2)
        nut_t=a1*k0/max(a1*w0,Ssr*F2)
        pkcap=10.0f0*βs*k0*w0
        Pk=min(nut_t*Ssr*Ω,pkcap)*gammat[i,j]
        σk=F1*σk1+(1.0f0-F1)*σk2;σw=F1*σw1+(1.0f0-F1)*σw2
        βc=F1*β1+(1.0f0-F1)*β2;γc=F1*γ1+(1.0f0-F1)*γ2
        iv=1/vol[i,j]
        SfE=hypot(Six[i+1,j],Siy[i+1,j])
        dnE=i+1<=NI ? hypot(xc[i+1,j]-xc[i,j],yc[i+1,j]-yc[i,j])+1.0f-30 : vol[i,j]/SfE+1.0f-30
        nfE=0.5f0*(nut[ia,ja]+nut[ia+1,ja])
        SfW=hypot(Six[i,j],Siy[i,j])
        dnW=i-1>=1 ? hypot(xc[i-1,j]-xc[i,j],yc[i-1,j]-yc[i,j])+1.0f-30 : vol[i,j]/SfW+1.0f-30
        nfW=0.5f0*(nut[ia,ja]+nut[ia-1,ja])
        SfN=hypot(Sjx[i,j+1],Sjy[i,j+1])
        dnN=j+1<=NJ ? hypot(xc[i,j+1]-xc[i,j],yc[i,j+1]-yc[i,j])+1.0f-30 : vol[i,j]/SfN+1.0f-30
        nfN=0.5f0*(nut[ia,ja]+nut[ia,ja+1])
        SfS=hypot(Sjx[i,j],Sjy[i,j])
        dnS=if j>1
            hypot(xc[i,j-1]-xc[i,j],yc[i,j-1]-yc[i,j])+1.0f-30
        elseif i<=NIW||i>NIW+NIF
            ip=NI+Int32(1)-i
            hypot(xc[ip,1]-xc[i,1],yc[ip,1]-yc[i,1])+1.0f-30
        else
            vol[i,j]/SfS+1.0f-30
        end
        nfS=0.5f0*(nut[ia,ja]+nut[ia,ja-1])
        u0=ux[ia,ja];v0=uy[ia,ja]
        Uci=(u0*0.5f0*(Six[i,j]+Six[i+1,j])+v0*0.5f0*(Siy[i,j]+Siy[i+1,j]))*iv
        Ucj=(u0*0.5f0*(Sjx[i,j]+Sjx[i,j+1])+v0*0.5f0*(Sjy[i,j]+Sjy[i,j+1]))*iv
        cW=max(Uci,zero(Uci));cE=max(-Uci,zero(Uci));cS=max(Ucj,zero(Ucj));cN=max(-Ucj,zero(Ucj))
        akE[i,j]=(nu+σk*nfE)*SfE/dnE*iv+cE;awE[i,j]=(nu+σw*nfE)*SfE/dnE*iv+cE
        akW[i,j]=(nu+σk*nfW)*SfW/dnW*iv+cW;awW[i,j]=(nu+σw*nfW)*SfW/dnW*iv+cW
        akN[i,j]=(nu+σk*nfN)*SfN/dnN*iv+cN;awN[i,j]=(nu+σw*nfN)*SfN/dnN*iv+cN
        akS[i,j]=(nu+σk*nfS)*SfS/dnS*iv+cS;awS[i,j]=(nu+σw*nfS)*SfS/dnS*iv+cS
        dk[i,j]=βs*w0;sk[i,j]=Pk
        cross=2.0f0*(1.0f0-F1)*σw2/w0*gkw
        prodw=γc/max(nut_t,1.0f-10)*Pk
        # Newton form of the destruction beta*w^2 -> (2 beta w0) w - beta w0^2; negative cross diffusion implicit.
        dw[i,j]=2.0f0*βc*w0+(cross<0 ? -cross/w0 : zero(cross))
        sw[i,j]=prodw+βc*w0*w0+(cross>0 ? cross : zero(cross))
        sf2[i,j]=Ssr*F2
    end
end

struct NewtonSST{A}
    akW::A;akE::A;akS::A;akN::A;awW::A;awE::A;awS::A;awN::A;dk::A;sk::A;dw::A;sw::A;sf2::A
end
newton_sst_init(s::KSol)=NewtonSST(ntuple(_->similar(s.dtf),13)...)

function _newton_sst_bc!(s::KSol)
    b=s.backend;NI,NJ,g=Int32(s.NI),Int32(s.NJ),Int32(s.g)
    _k_sst_bc!(b)(s.tk,s.tom,s.nut,s.tnl,s.k_inf,s.w_inf,s.keps,NI,NJ,Int32(s.NIW),Int32(s.NIF),g;ndrange=(s.NI,s.g))
    _k_sst_bc_io!(b)(s.tk,s.tom,s.nut,s.tnl,NI,NJ,g;ndrange=(s.NJ+2s.g,s.g))
end



# ---- scalar line solver ---------------------------------------------------------------------------------
struct NewtonScalar{A}
    cW::A;cE::A;cS::A;cN::A;cP::A;rhs::A;r::A;work::A
    jcolor::Vector{Int32};njcolor::Int
end
newton_scalar_init(s::KSol,L::NewtonLines)=NewtonScalar(ntuple(_->similar(s.dtf),8)...,L.jcolor,L.njcolor)

# Assemble one scalar system from the SST coefficients (same boundary logic as the LU prototype).
@kernel function _k_newton_sst_system!(cW,cE,cS,cN,cP,rhs,@Const(aW),@Const(aE),@Const(aS),@Const(aN),@Const(D),@Const(S),
    @Const(phi),@Const(invdt),@Const(rho),@Const(T),@Const(wdist),far,isk::Int32,
    nu_inf,rho_inf,suth,air_sutherland::Int32,NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI&&j<=NJ
        z=zero(eltype(cP))
        if isk==Int32(0)&&j<=2&&NIW<i<=NIW+NIF
            nu=_air_nu_local(rho[i+g,j+g],T[i+g,j+g],nu_inf,rho_inf,suth,air_sutherland)
            d=max(wdist[i,j],oftype(nu,1e-8))
            cW[i,j]=z;cE[i,j]=z;cS[i,j]=z;cN[i,j]=z;cP[i,j]=one(z);rhs[i,j]=6*nu/(oftype(nu,0.075)*d*d)
        else
            diag=invdt[i,j]+D[i,j];r=invdt[i,j]*phi[i+g,j+g]+S[i,j]
            w=z;e=z;so=z;no=z
            if i>1;diag+=aW[i,j];w=-aW[i,j];end
            if i<NI;diag+=aE[i,j];e=-aE[i,j];end
            diag+=aN[i,j]
            if j<NJ;no=-aN[i,j];else;r+=aN[i,j]*far;end
            if j>1||i<=NIW||i>NIW+NIF
                diag+=aS[i,j];so=-aS[i,j]
            elseif isk==Int32(1)
                diag+=2*aS[i,j]
            end
            cW[i,j]=w;cE[i,j]=e;cS[i,j]=so;cN[i,j]=no;cP[i,j]=diag;rhs[i,j]=r
        end
    end
end

@inline function _newton_scalar_south(x,i,j,NI,NIW,NIF)
    @inbounds j>1 ? x[i,j-Int32(1)] : ((i<=NIW||i>NIW+NIF) ? x[NI+Int32(1)-i,Int32(1)] : zero(eltype(x)))
end

# r = b - A x
@kernel function _k_newton_scalar_residual!(r,@Const(b),@Const(x),@Const(cW),@Const(cE),@Const(cS),@Const(cN),@Const(cP),
    NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI&&j<=NJ
        acc=cP[i,j]*x[i,j]+cS[i,j]*_newton_scalar_south(x,i,j,NI,NIW,NIF)
        if i>1;acc+=cW[i,j]*x[i-Int32(1),j];end
        if i<NI;acc+=cE[i,j]*x[i+Int32(1),j];end
        if j<NJ;acc+=cN[i,j]*x[i,j+Int32(1)];end
        r[i,j]=b[i,j]-acc
    end
end

# Cell m of j-line l and its (previous, next) off-diagonal coefficients along the line.
@inline function _newton_scalar_cut_cell(l,m,cS,cN,NI,NJ,NIW,NIF)
    @inbounds if l<=NIF
        i=NIW+l;j=m
        (i,j,j>1 ? cS[i,j] : zero(eltype(cS)),j<NJ ? cN[i,j] : zero(eltype(cS)))
    else
        w=l-NIF;ip=NI+Int32(1)-w
        if m<=NJ
            j=NJ+Int32(1)-m
            (ip,j,j<NJ ? cN[ip,j] : zero(eltype(cS)),cS[ip,j])
        else
            j=m-NJ
            (w,j,cS[w,j],j<NJ ? cN[w,j] : zero(eltype(cS)))
        end
    end
end

# x += T_j^-1 r on the j-lines of one colour (Thomas; r and work are overwritten on those lines).
@kernel function _k_newton_scalar_jline!(x,r,work,@Const(cS),@Const(cN),@Const(cP),@Const(color),c::Int32,
    NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32)
    l=@index(Global,Linear)
    @inbounds if l<=NIW+NIF&&color[l]==c
        count=l<=NIF ? NJ : Int32(2)*NJ
        cp=zero(eltype(x));dp=zero(eltype(x))
        for m in Int32(1):count
            i,j,lo,hi=_newton_scalar_cut_cell(l,m,cS,cN,NI,NJ,NIW,NIF)
            den=cP[i,j]-lo*cp
            cp=hi/den;dp=(r[i,j]-lo*dp)/den
            work[i,j]=cp;r[i,j]=dp
        end
        next=zero(eltype(x))
        for m in count:Int32(-1):Int32(1)
            i,j,_,_=_newton_scalar_cut_cell(l,m,cS,cN,NI,NJ,NIW,NIF)
            next=r[i,j]-work[i,j]*next
            x[i,j]+=next
        end
    end
end

@kernel function _k_newton_scalar_iline!(x,r,work,@Const(cW),@Const(cE),@Const(cP),c::Int32,NI::Int32,NJ::Int32)
    j=@index(Global,Linear)
    @inbounds if j<=NJ&&(j&Int32(1))==c
        cp=zero(eltype(x));dp=zero(eltype(x))
        for i in Int32(1):NI
            lo=i>1 ? cW[i,j] : zero(eltype(x))
            den=cP[i,j]-lo*cp
            cp=(i<NI ? cE[i,j] : zero(eltype(x)))/den;dp=(r[i,j]-lo*dp)/den
            work[i,j]=cp;r[i,j]=dp
        end
        next=zero(eltype(x))
        for i in NI:Int32(-1):Int32(1)
            next=r[i,j]-work[i,j]*next
            x[i,j]+=next
        end
    end
end

function newton_scalar_precond!(x,b,s::KSol,q::NewtonScalar;sweeps::Int=1)
    bk=s.backend;ni,nj,niw,nif=Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF)
    fill!(x,0);first=true
    resid!()=if first
        copyto!(q.r,b);first=false
    else
        _k_newton_scalar_residual!(bk)(q.r,b,x,q.cW,q.cE,q.cS,q.cN,q.cP,ni,nj,niw,nif;ndrange=(s.NI,s.NJ))
    end
    for _ in 1:sweeps
        for c in 0:q.njcolor-1
            resid!()
            _k_newton_scalar_jline!(bk)(x,q.r,q.work,q.cS,q.cN,q.cP,q.jcolor,Int32(c),ni,nj,niw,nif;ndrange=s.NIW+s.NIF)
        end
        for c in 0:1
            resid!()
            _k_newton_scalar_iline!(bk)(x,q.r,q.work,q.cW,q.cE,q.cP,Int32(c),ni,nj;ndrange=s.NJ)
        end
    end
    x
end

"Solve the assembled scalar system from the initial guess phi0 (NI,NJ); returns (solution, iterations, relative residual)."
function newton_scalar_solve!(s::KSol,q::NewtonScalar,phi0;tol=1e-3,maxit=60,sweeps=1)
    NI,NJ=s.NI,s.NJ;bk=s.backend;ni,nj,niw,nif=Int32(NI),Int32(NJ),Int32(s.NIW),Int32(s.NIF)
    R2(v)=reshape(v,NI,NJ)
    # Solve for the correction e: A e = rhs - A phi0, so the tolerance is relative to the initial defect.
    b0=similar(phi0)
    _k_newton_scalar_residual!(bk)(b0,q.rhs,phi0,q.cW,q.cE,q.cS,q.cN,q.cP,ni,nj,niw,nif;ndrange=(NI,NJ))
    zero0=fill!(similar(phi0),0)
    A!(y,x)=(_k_newton_scalar_residual!(bk)(R2(y),zero0,R2(x),q.cW,q.cE,q.cS,q.cN,q.cP,ni,nj,niw,nif;ndrange=(NI,NJ));y.*=-1;y)
    P!(z,v)=(newton_scalar_precond!(R2(z),R2(v),s,q;sweeps);z)
    e,its,hist=newton_gmres(A!,P!,vec(Array(b0));tol,restart=maxit,maxit)
    phi0.+R2(e),its,hist[end]
end

"Implicit SST step with the line-relaxation solver (same equations as newton_sst_implicit!)."
function newton_sst_implicit_lines!(s::KSol,c::NewtonSST,q::NewtonScalar,invdt;relax=1.0,nut_relax=1.0,tol=1e-3,
        corrected_wall_metric::Bool=true)
    s.sst&&s.keps==0&&s.lag==Int32(0)&&s.earsm==Int32(0)||throw(ArgumentError("implicit SST prototype: plain SST only"))
    NI,NJ,NIW,NIF,g=s.NI,s.NJ,s.NIW,s.NIF,s.g;bk=s.backend
    _newton_sst_bc!(s)
    _k_newton_sst_coeffs!(bk)(c.akW,c.akE,c.akS,c.akN,c.awW,c.awE,c.awS,c.awN,c.dk,c.sk,c.dw,c.sw,c.sf2,
        s.tk,s.tom,s.nut,s.rho,s.T,s.ux,s.uy,s.xc,s.yc,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,s.wdist,s.gammat,
        s.nu_lam,s.rho_inf,s.suth,s.air_sutherland,Int32(NI),Int32(NJ),Int32(NIW),Int32(NIF),Int32(g),
        Int32(corrected_wall_metric);ndrange=(NI,NJ))
    ir=g+1:g+NI;jr=g+1:g+NJ
    kold=Array(view(s.tk,ir,jr));wold=Array(view(s.tom,ir,jr))
    system!(aW,aE,aS,aN,D,S,phi,far,isk)=_k_newton_sst_system!(bk)(q.cW,q.cE,q.cS,q.cN,q.cP,q.rhs,aW,aE,aS,aN,D,S,phi,invdt,
        s.rho,s.T,s.wdist,eltype(q.cP)(far),Int32(isk),s.nu_lam,s.rho_inf,s.suth,s.air_sutherland,
        Int32(NI),Int32(NJ),Int32(NIW),Int32(NIF),Int32(g);ndrange=(NI,NJ))
    system!(c.akW,c.akE,c.akS,c.akN,c.dk,c.sk,s.tk,s.k_inf,1)
    knew,kits,klin=newton_scalar_solve!(s,q,kold;tol)
    system!(c.awW,c.awE,c.awS,c.awN,c.dw,c.sw,s.tom,s.w_inf,0)
    wnew,wits,wlin=newton_scalar_solve!(s,q,wold;tol)
    knew.=clamp.(kold.+relax.*(knew.-kold),1e-12,1.0);wnew.=max.(wold.+relax.*(wnew.-wold),1e-6)
    dkmax=maximum(abs.(knew.-kold)./max.(abs.(kold),Float64(s.k_inf),1e-12))
    dwmax=maximum(abs.(wnew.-wold)./max.(abs.(wold),Float64(s.w_inf),1e-6))
    view(s.tk,ir,jr).=knew;view(s.tom,ir,jr).=wnew
    nutv=view(s.nut,ir,jr);sf2=c.sf2
    @inbounds for j in 1:NJ,i in 1:NI
        wall=j<=2&&NIW<i<=NIW+NIF
        nt=wall ? knew[i,j]/max(wnew[i,j],1e-6) : 0.31*knew[i,j]/max(0.31*wnew[i,j],Float64(sf2[i,j]))
        nutv[i,j]+=nut_relax*(min(nt,0.15)-nutv[i,j])
    end
    _newton_sst_bc!(s)
    (;dkmax,dwmax,kits,wits,klin,wlin)
end
