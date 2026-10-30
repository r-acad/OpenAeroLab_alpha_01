# ==============================================================================
# Particles-on-Demand — 3D THERMAL conservative kinetic-FV  [full 3D flow; GPU]
# ==============================================================================
# The genuine-3D extension of pond_fv2d_thermal.jl: 5 conserved variables (ρ, ρu, ρv, ρw, ρE) on an
# NX×NY×NZ grid with SIX face fluxes (±x,±y,±z). Same D3Q39 set (already 3D, cs²=2/3, |c|≤3), same
# 3rd-order thermal Hermite equilibrium (now with the z peculiar velocity), and the same √θ PonD gauge
# (supersonic) + adiabatic kinetic wall. Minmod reconstruction (robust default). γ=5/3 (drot=0) or
# general-γ via drot. Reuses _clampi/_mm from the shared pond files.
# ==============================================================================

using KernelAbstractions

if !@isdefined(TGAS_D); const TGAS_D = 3.0f0; end             # internal DOF (shared with the 2D thermal file)
if !@isdefined(TGAS_GAMMA); const TGAS_GAMMA = 5.0f0/3.0f0; end

# 3D 3rd-order thermal Hermite equilibrium in the (co-moving/gauge) frame (ufx,ufy,ufz), reduced by
# isq=1/√θ_f, invth=1/θ_f (gauge). Fixed frame ⇒ isq=invth=1 (byte-identical to no-gauge).
@inline function _feqT3_gpu(rho, ux, uy, uz, T, ufx, ufy, ufz, cxk, cyk, czk, wk, cs2, isq, invth)
    px=(ux-ufx)*isq; py=(uy-ufy)*isq; pz=(uz-ufz)*isq
    inv_cs2=1.0f0/cs2; inv_2cs4=0.5f0/(cs2*cs2); inv_6cs6=1.0f0/(6.0f0*cs2*cs2*cs2)
    cp = px*cxk + py*cyk + pz*czk
    p2 = px*px + py*py + pz*pz
    csq = cxk*cxk + cyk*cyk + czk*czk
    th = T*invth - 1.0f0
    f  = rho
    f += rho*cp*inv_cs2
    f += rho*(cp*cp - cs2*p2)*inv_2cs4
    f += rho*th*0.5f0*(csq*inv_cs2 - TGAS_D)
    f += rho*(cp*cp*cp - 3.0f0*cs2*cp*p2)*inv_6cs6
    f += rho*th*cp*(csq - (TGAS_D+2.0f0)*cs2)*inv_2cs4
    return max(wk*f, 1.0f-12)
end

# face reconstruction: DRP κ=1/3 (low-dissipation) in smooth flow, blended to minmod at shocks by the
# sensor s (s=1 ⇒ minmod). Same low-dispersion reconstruction validated in 1D/2D; drp=0 ⇒ pure minmod.
@inline function _faceL3(s, fLL, fL, fR)
    hi = (-fLL + 5.0f0*fL + 2.0f0*fR)*0.16666667f0
    lo = fL + 0.5f0*_mm(fL-fLL, fR-fL)
    return (1.0f0-s)*hi + s*lo
end
@inline function _faceR3(s, fL, fR, fRR)
    hi = (2.0f0*fL + 5.0f0*fR - fRR)*0.16666667f0
    lo = fR - 0.5f0*_mm(fR-fL, fRR-fR)
    return (1.0f0-s)*hi + s*lo
end

# 3D thermal face flux (Fρ,Fmx,Fmy,Fmz,FE) along axis ax∈{1,2,3} from L/R primitives. Full feature parity
# with the 2D flux: sensor-gated DRP κ=1/3 (drp) / minmod, √θ gauge (gauge), SHOCK-AWARE Roe frame
# (shock_aware), and thin-layer Smagorinsky SGS eddy-viscosity (csmag>0 ⇒ LES).
@inline function _fvflux3T(rhoL,uxL,uyL,uzL,TL, rhoR,uxR,uyR,uzR,TR,
                           rhoLL,uxLL,uyLL,uzLL,TLL, rhoRR,uxRR,uyRR,uzRR,TRR,
                           ax, cx_arr,cy_arr,cz_arr,w_arr, cs2, drot, csmag, gauge, shock_aware, drp, nu_mol=0.0f0)
    ufx=0.5f0*(uxL+uxR); ufy=0.5f0*(uyL+uyR); ufz=0.5f0*(uzL+uzR); erc=0.5f0*drot*cs2
    s = 1.0f0
    if drp > 0
        pLm=rhoL*cs2*TL; pRm=rhoR*cs2*TR
        sj = max(abs(pRm-pLm)/min(pLm,pRm), abs(rhoR-rhoL)/min(rhoL,rhoR))
        s = sj<=0.1f0 ? 0.0f0 : (sj>=0.4f0 ? 1.0f0 : (sj-0.1f0)*(1.0f0/0.3f0))
    end
    rhoLf=max(_faceL3(s,rhoLL,rhoL,rhoR),1.0f-4); rhoRf=max(_faceR3(s,rhoL,rhoR,rhoRR),1.0f-4)
    uxLf=_faceL3(s,uxLL,uxL,uxR); uxRf=_faceR3(s,uxL,uxR,uxRR)
    uyLf=_faceL3(s,uyLL,uyL,uyR); uyRf=_faceR3(s,uyL,uyR,uyRR)
    uzLf=_faceL3(s,uzLL,uzL,uzR); uzRf=_faceR3(s,uzL,uzR,uzRR)
    TLf=max(_faceL3(s,TLL,TL,TR),1.0f-3); TRf=max(_faceR3(s,TL,TR,TRR),1.0f-3)
    sq=1.0f0; isq=1.0f0; invth=1.0f0
    if gauge > 0
        if shock_aware > 0     # ROE-averaged frame (√ρ-weighted velocity + enthalpy) — entropy-consistent
            sL=sqrt(rhoLf); sR=sqrt(rhoRf); wR=sR/(sL+sR); wL=1.0f0-wR
            cp=0.5f0*(5.0f0+drot)*cs2
            hL=cp*TLf+0.5f0*(uxLf*uxLf+uyLf*uyLf+uzLf*uzLf); hR=cp*TRf+0.5f0*(uxRf*uxRf+uyRf*uyRf+uzRf*uzRf)
            ufx=wL*uxLf+wR*uxRf; ufy=wL*uyLf+wR*uyRf; ufz=wL*uzLf+wR*uzRf
            hroe=wL*hL+wR*hR; θ=max((hroe-0.5f0*(ufx*ufx+ufy*ufy+ufz*ufz))/cp,1.0f-3)
        else
            θ=max(0.5f0*(TLf+TRf),1.0f-3)
        end
        sq=sqrt(θ); isq=1.0f0/sq; invth=1.0f0/θ
    end
    Frho=0.0f0;Fmx=0.0f0;Fmy=0.0f0;Fmz=0.0f0;FE=0.0f0
    @inbounds for k in 1:39
        cxk=cx_arr[k]; cyk=cy_arr[k]; czk=cz_arr[k]; wk=w_arr[k]
        vx=sq*cxk+ufx; vy=sq*cyk+ufy; vz=sq*czk+ufz
        vn = ax==1 ? vx : (ax==2 ? vy : vz)
        if vn>=0.0f0; fk=_feqT3_gpu(rhoLf,uxLf,uyLf,uzLf,TLf,ufx,ufy,ufz,cxk,cyk,czk,wk,cs2,isq,invth); Ts=TLf
        else;         fk=_feqT3_gpu(rhoRf,uxRf,uyRf,uzRf,TRf,ufx,ufy,ufz,cxk,cyk,czk,wk,cs2,isq,invth); Ts=TRf; end
        ke=0.5f0*(vx*vx+vy*vy+vz*vz)+erc*Ts
        Frho+=vn*fk; Fmx+=vn*vx*fk; Fmy+=vn*vy*fk; Fmz+=vn*vz*fk; FE+=vn*ke*fk
    end
    if csmag > 0.0f0 || nu_mol > 0.0f0     # viscous flux: MOLECULAR ν_mol (sets Re=UD/ν) + Smagorinsky SGS ν_sgs=(Cs Δ)²|∂ₙu|
        du=uxR-uxL; dv=uyR-uyL; dw=uzR-uzL; rf=0.5f0*(rhoL+rhoR)
        nu = nu_mol + (csmag>0.0f0 ? csmag*csmag*sqrt(du*du+dv*dv+dw*dw) : 0.0f0)
        Fmx+=-rf*nu*du; Fmy+=-rf*nu*dv; Fmz+=-rf*nu*dw; FE+=-rf*nu*(ufx*du+ufy*dv+ufz*dw)
    end
    return Frho,Fmx,Fmy,Fmz,FE
end

@kernel function pond_fv3dT_kernel!(rho_o,ux_o,uy_o,uz_o,T_o,
        @Const(rho),@Const(ux),@Const(uy),@Const(uz),@Const(Temp),@Const(chi),
        @Const(cx_arr),@Const(cy_arr),@Const(cz_arr),@Const(w_arr),
        cs2::Float32, a::Float32, drot::Float32, csmag::Float32, gauge::Int32, wall::Int32, shock_aware::Int32, drp::Int32, nu_mol::Float32, NX::Int32, NY::Int32, NZ::Int32)
    i,j,l = @index(Global, NTuple)
    nx=Int(NX); ny=Int(NY); nz=Int(NZ); eint=0.5f0*(3.0f0+drot)*cs2
    @inbounds begin
        iL=_clampi(i-1,nx);iR=_clampi(i+1,nx);iRR=_clampi(i+2,nx);iLL=_clampi(i-2,nx)
        jD=_clampi(j-1,ny);jU=_clampi(j+1,ny);jUU=_clampi(j+2,ny);jDD=_clampi(j-2,ny)
        kB=_clampi(l-1,nz);kF=_clampi(l+1,nz);kFF=_clampi(l+2,nz);kBB=_clampi(l-2,nz)
        FxR=_fvflux3T(rho[i,j,l],ux[i,j,l],uy[i,j,l],uz[i,j,l],Temp[i,j,l], rho[iR,j,l],ux[iR,j,l],uy[iR,j,l],uz[iR,j,l],Temp[iR,j,l],
            rho[iL,j,l],ux[iL,j,l],uy[iL,j,l],uz[iL,j,l],Temp[iL,j,l], rho[iRR,j,l],ux[iRR,j,l],uy[iRR,j,l],uz[iRR,j,l],Temp[iRR,j,l], 1, cx_arr,cy_arr,cz_arr,w_arr,cs2,drot,csmag,gauge,shock_aware,drp,nu_mol)
        FxL=_fvflux3T(rho[iL,j,l],ux[iL,j,l],uy[iL,j,l],uz[iL,j,l],Temp[iL,j,l], rho[i,j,l],ux[i,j,l],uy[i,j,l],uz[i,j,l],Temp[i,j,l],
            rho[iLL,j,l],ux[iLL,j,l],uy[iLL,j,l],uz[iLL,j,l],Temp[iLL,j,l], rho[iR,j,l],ux[iR,j,l],uy[iR,j,l],uz[iR,j,l],Temp[iR,j,l], 1, cx_arr,cy_arr,cz_arr,w_arr,cs2,drot,csmag,gauge,shock_aware,drp,nu_mol)
        FyR=_fvflux3T(rho[i,j,l],ux[i,j,l],uy[i,j,l],uz[i,j,l],Temp[i,j,l], rho[i,jU,l],ux[i,jU,l],uy[i,jU,l],uz[i,jU,l],Temp[i,jU,l],
            rho[i,jD,l],ux[i,jD,l],uy[i,jD,l],uz[i,jD,l],Temp[i,jD,l], rho[i,jUU,l],ux[i,jUU,l],uy[i,jUU,l],uz[i,jUU,l],Temp[i,jUU,l], 2, cx_arr,cy_arr,cz_arr,w_arr,cs2,drot,csmag,gauge,shock_aware,drp,nu_mol)
        FyL=_fvflux3T(rho[i,jD,l],ux[i,jD,l],uy[i,jD,l],uz[i,jD,l],Temp[i,jD,l], rho[i,j,l],ux[i,j,l],uy[i,j,l],uz[i,j,l],Temp[i,j,l],
            rho[i,jDD,l],ux[i,jDD,l],uy[i,jDD,l],uz[i,jDD,l],Temp[i,jDD,l], rho[i,jU,l],ux[i,jU,l],uy[i,jU,l],uz[i,jU,l],Temp[i,jU,l], 2, cx_arr,cy_arr,cz_arr,w_arr,cs2,drot,csmag,gauge,shock_aware,drp,nu_mol)
        FzR=_fvflux3T(rho[i,j,l],ux[i,j,l],uy[i,j,l],uz[i,j,l],Temp[i,j,l], rho[i,j,kF],ux[i,j,kF],uy[i,j,kF],uz[i,j,kF],Temp[i,j,kF],
            rho[i,j,kB],ux[i,j,kB],uy[i,j,kB],uz[i,j,kB],Temp[i,j,kB], rho[i,j,kFF],ux[i,j,kFF],uy[i,j,kFF],uz[i,j,kFF],Temp[i,j,kFF], 3, cx_arr,cy_arr,cz_arr,w_arr,cs2,drot,csmag,gauge,shock_aware,drp,nu_mol)
        FzL=_fvflux3T(rho[i,j,kB],ux[i,j,kB],uy[i,j,kB],uz[i,j,kB],Temp[i,j,kB], rho[i,j,l],ux[i,j,l],uy[i,j,l],uz[i,j,l],Temp[i,j,l],
            rho[i,j,kBB],ux[i,j,kBB],uy[i,j,kBB],uz[i,j,kBB],Temp[i,j,kBB], rho[i,j,kF],ux[i,j,kF],uy[i,j,kF],uz[i,j,kF],Temp[i,j,kF], 3, cx_arr,cy_arr,cz_arr,w_arr,cs2,drot,csmag,gauge,shock_aware,drp,nu_mol)
        # adiabatic kinetic wall: fluid cell bordering solid ⇒ reflective face flux (0 mass, 0 energy, p_w momentum)
        if wall==1 && chi[i,j,l]<=0.5f0
            pw=rho[i,j,l]*cs2*Temp[i,j,l]
            if chi[iR,j,l]>0.5f0; FxR=(0.0f0,pw,0.0f0,0.0f0,0.0f0); end
            if chi[iL,j,l]>0.5f0; FxL=(0.0f0,pw,0.0f0,0.0f0,0.0f0); end
            if chi[i,jU,l]>0.5f0; FyR=(0.0f0,0.0f0,pw,0.0f0,0.0f0); end
            if chi[i,jD,l]>0.5f0; FyL=(0.0f0,0.0f0,pw,0.0f0,0.0f0); end
            if chi[i,j,kF]>0.5f0; FzR=(0.0f0,0.0f0,0.0f0,pw,0.0f0); end
            if chi[i,j,kB]>0.5f0; FzL=(0.0f0,0.0f0,0.0f0,pw,0.0f0); end
        end
        u0x=ux[i,j,l];u0y=uy[i,j,l];u0z=uz[i,j,l];r0=rho[i,j,l]
        Ec=r0*(eint*Temp[i,j,l]+0.5f0*(u0x*u0x+u0y*u0y+u0z*u0z))
        r  = r0      - a*((FxR[1]-FxL[1])+(FyR[1]-FyL[1])+(FzR[1]-FzL[1]))
        mx = r0*u0x  - a*((FxR[2]-FxL[2])+(FyR[2]-FyL[2])+(FzR[2]-FzL[2]))
        my = r0*u0y  - a*((FxR[3]-FxL[3])+(FyR[3]-FyL[3])+(FzR[3]-FzL[3]))
        mz = r0*u0z  - a*((FxR[4]-FxL[4])+(FyR[4]-FyL[4])+(FzR[4]-FzL[4]))
        e  = Ec      - a*((FxR[5]-FxL[5])+(FyR[5]-FyL[5])+(FzR[5]-FzL[5]))
        r = r < 1.0f-3 ? 1.0f-3 : r
        ux2=mx/r;uy2=my/r;uz2=mz/r; sp=sqrt(ux2*ux2+uy2*uy2+uz2*uz2)
        if sp>3.5f0; fscl=3.5f0/sp; ux2*=fscl;uy2*=fscl;uz2*=fscl; end
        T2=(e/r-0.5f0*(ux2*ux2+uy2*uy2+uz2*uz2))/eint; T2=T2<1.0f-3 ? 1.0f-3 : T2
        rho_o[i,j,l]=r; ux_o[i,j,l]=ux2; uy_o[i,j,l]=uy2; uz_o[i,j,l]=uz2; T_o[i,j,l]=T2
    end
end

# freestream farfield overwrite on the 4 lateral faces (y,z edges); inlet (x=1..ng) + outlet (x>nx-ng)
@kernel function _bc3T!(rho,ux,uy,uz,T, rho_inf::Float32,Uin::Float32,Tinf::Float32, NX::Int32,NY::Int32,NZ::Int32,ng::Int32)
    i,j,l=@index(Global,NTuple); nx=Int(NX);ny=Int(NY);nz=Int(NZ);g=Int(ng)
    @inbounds begin
        if i<=g; rho[i,j,l]=rho_inf;ux[i,j,l]=Uin;uy[i,j,l]=0f0;uz[i,j,l]=0f0;T[i,j,l]=Tinf
        elseif i>nx-g; s=nx-g; rho[i,j,l]=rho[s,j,l];ux[i,j,l]=ux[s,j,l];uy[i,j,l]=uy[s,j,l];uz[i,j,l]=uz[s,j,l];T[i,j,l]=T[s,j,l]
        elseif (j<=g)||(j>ny-g)||(l<=g)||(l>nz-g); rho[i,j,l]=rho_inf;ux[i,j,l]=Uin;uy[i,j,l]=0f0;uz[i,j,l]=0f0;T[i,j,l]=Tinf end
    end
end
# α-weighted (cut-fraction) penalization: damp u by the SOLID FRACTION χ∈[0,1] ⇒ the effective wall sits at
# χ=½ (the true surface, sub-cell) rather than the staircase. Binary χ∈{0,1} reproduces the old behaviour.
@kernel function _penalize3!(ux,uy,uz, @Const(chi), rate::Float32)
    i,j,l=@index(Global,NTuple)
    @inbounds begin
        c=chi[i,j,l]
        if c>0.0f0; f=1f0/(1f0+c*rate); ux[i,j,l]*=f;uy[i,j,l]*=f;uz[i,j,l]*=f; end
    end
end
# k⁶-selective SVV (∇⁶ hyperviscosity) on the MOMENTUM — damps the 2Δ grid-scale mode that the
# low-dissipation DRP reconstruction permits at high resolution, WITHOUT touching the resolved scales
# (k⁶ ⇒ 47× more damping at 2Δ than at 4Δ). Mass and T pass through unchanged ⇒ conservative + safe at
# shocks. Double-buffered. Enables DRP κ=1/3 to run stably on fine grids.
@inline _d6(a,b,c,d,e,f,g) = a - 6.0f0*b + 15.0f0*c - 20.0f0*d + 15.0f0*e - 6.0f0*f + g
@kernel function _svv3T!(rho_o,ux_o,uy_o,uz_o,T_o, @Const(rho),@Const(ux),@Const(uy),@Const(uz),@Const(Temp), eps::Float32, NX::Int32,NY::Int32,NZ::Int32)
    i,j,l=@index(Global,NTuple); nx=Int(NX);ny=Int(NY);nz=Int(NZ)
    @inbounds begin
        a1=_clampi(i-1,nx);a2=_clampi(i-2,nx);a3=_clampi(i-3,nx);b1=_clampi(i+1,nx);b2=_clampi(i+2,nx);b3=_clampi(i+3,nx)
        c1=_clampi(j-1,ny);c2=_clampi(j-2,ny);c3=_clampi(j-3,ny);e1=_clampi(j+1,ny);e2=_clampi(j+2,ny);e3=_clampi(j+3,ny)
        p1=_clampi(l-1,nz);p2=_clampi(l-2,nz);p3=_clampi(l-3,nz);q1=_clampi(l+1,nz);q2=_clampi(l+2,nz);q3=_clampi(l+3,nz)
        r0=rho[i,j,l]; mx0=r0*ux[i,j,l]; my0=r0*uy[i,j,l]; mz0=r0*uz[i,j,l]
        hmx=_d6(rho[a3,j,l]*ux[a3,j,l],rho[a2,j,l]*ux[a2,j,l],rho[a1,j,l]*ux[a1,j,l],mx0,rho[b1,j,l]*ux[b1,j,l],rho[b2,j,l]*ux[b2,j,l],rho[b3,j,l]*ux[b3,j,l])+
            _d6(rho[i,c3,l]*ux[i,c3,l],rho[i,c2,l]*ux[i,c2,l],rho[i,c1,l]*ux[i,c1,l],mx0,rho[i,e1,l]*ux[i,e1,l],rho[i,e2,l]*ux[i,e2,l],rho[i,e3,l]*ux[i,e3,l])+
            _d6(rho[i,j,p3]*ux[i,j,p3],rho[i,j,p2]*ux[i,j,p2],rho[i,j,p1]*ux[i,j,p1],mx0,rho[i,j,q1]*ux[i,j,q1],rho[i,j,q2]*ux[i,j,q2],rho[i,j,q3]*ux[i,j,q3])
        hmy=_d6(rho[a3,j,l]*uy[a3,j,l],rho[a2,j,l]*uy[a2,j,l],rho[a1,j,l]*uy[a1,j,l],my0,rho[b1,j,l]*uy[b1,j,l],rho[b2,j,l]*uy[b2,j,l],rho[b3,j,l]*uy[b3,j,l])+
            _d6(rho[i,c3,l]*uy[i,c3,l],rho[i,c2,l]*uy[i,c2,l],rho[i,c1,l]*uy[i,c1,l],my0,rho[i,e1,l]*uy[i,e1,l],rho[i,e2,l]*uy[i,e2,l],rho[i,e3,l]*uy[i,e3,l])+
            _d6(rho[i,j,p3]*uy[i,j,p3],rho[i,j,p2]*uy[i,j,p2],rho[i,j,p1]*uy[i,j,p1],my0,rho[i,j,q1]*uy[i,j,q1],rho[i,j,q2]*uy[i,j,q2],rho[i,j,q3]*uy[i,j,q3])
        hmz=_d6(rho[a3,j,l]*uz[a3,j,l],rho[a2,j,l]*uz[a2,j,l],rho[a1,j,l]*uz[a1,j,l],mz0,rho[b1,j,l]*uz[b1,j,l],rho[b2,j,l]*uz[b2,j,l],rho[b3,j,l]*uz[b3,j,l])+
            _d6(rho[i,c3,l]*uz[i,c3,l],rho[i,c2,l]*uz[i,c2,l],rho[i,c1,l]*uz[i,c1,l],mz0,rho[i,e1,l]*uz[i,e1,l],rho[i,e2,l]*uz[i,e2,l],rho[i,e3,l]*uz[i,e3,l])+
            _d6(rho[i,j,p3]*uz[i,j,p3],rho[i,j,p2]*uz[i,j,p2],rho[i,j,p1]*uz[i,j,p1],mz0,rho[i,j,q1]*uz[i,j,q1],rho[i,j,q2]*uz[i,j,q2],rho[i,j,q3]*uz[i,j,q3])
        rho_o[i,j,l]=r0; T_o[i,j,l]=Temp[i,j,l]
        ux_o[i,j,l]=(mx0+eps*hmx)/r0; uy_o[i,j,l]=(my0+eps*hmy)/r0; uz_o[i,j,l]=(mz0+eps*hmz)/r0
    end
end
# absorbing sponge: relax toward freestream in a band near the 4 LATERAL (y,z) faces and the OUTLET (+x),
# NOT the inlet. Quadratic ramp. Damps the oblique-wave reflections that otherwise criss-cross the wake.
@kernel function _sponge3T!(rho,ux,uy,uz,T, ri::Float32,Ui::Float32,Ti::Float32, NX::Int32,NY::Int32,NZ::Int32, sw::Int32, smax::Float32)
    i,j,l=@index(Global,NTuple); nx=Int(NX);ny=Int(NY);nz=Int(NZ); w=Int(sw)
    @inbounds begin
        dmin=min(j-1, ny-j, l-1, nz-l, nx-i)          # dist to nearest lateral or outlet boundary
        if dmin<w
            t=1.0f0-Float32(dmin)/Float32(w); σ=smax*t*t
            rho[i,j,l]+=σ*(ri-rho[i,j,l]); ux[i,j,l]+=σ*(Ui-ux[i,j,l]); uy[i,j,l]+=σ*(0f0-uy[i,j,l]); uz[i,j,l]+=σ*(0f0-uz[i,j,l]); T[i,j,l]+=σ*(Ti-T[i,j,l])
        end
    end
end

mutable struct PonD3DT{A,V}
    backend; rho::A;ux::A;uy::A;uz::A;T::A; rhon::A;uxn::A;uyn::A;uzn::A;Tn::A; chi::A
    cxa::V;cya::V;cza::V;wa::V
    NX::Int;NY::Int;NZ::Int; dx::Float32;dt::Float32;cs2::Float32
    rho_inf::Float32;U_inf::Float32;T_inf::Float32;eta::Float32;ng::Int;drot::Float32;gauge::Int32;wall::Int32;drp::Int32
    sponge_w::Int;sponge_max::Float32;svv_eps::Float32;csmag::Float32;shock_aware::Int32;nu_mol::Float32
end

function pond_fv3dT_driver(backend,dev,NX,NY,NZ; dx,dt,cs2,rho_inf,U_inf,T_inf,chi_host,
                           eta=1f-3,ng=3,gamma=5/3,gauge_theta=false,adiabatic_wall=false,drp=false,
                           sponge_w=0,sponge_max=0.5f0,svv_eps=0f0,cs_smag=0f0,shock_aware=false,nu_mol=0f0,cxv,cyv,czv,wv)
    o(v)=dev(fill(Float32(v),NX,NY,NZ)); z()=dev(zeros(Float32,NX,NY,NZ))
    drot=Float32(2/(gamma-1)-3)
    ux0h=[chi_host[i,j,l]>0.5f0 ? 0f0 : Float32(U_inf) for i in 1:NX, j in 1:NY, l in 1:NZ]  # zero u INSIDE the body (no startup surface jump)
    PonD3DT{typeof(z()),typeof(dev(Float32.(cxv)))}(backend, o(rho_inf),dev(ux0h),z(),z(),o(T_inf), z(),z(),z(),z(),z(), dev(chi_host),
        dev(Float32.(cxv)),dev(Float32.(cyv)),dev(Float32.(czv)),dev(Float32.(wv)),
        NX,NY,NZ, Float32(dx),Float32(dt),Float32(cs2), Float32(rho_inf),Float32(U_inf),Float32(T_inf),Float32(eta),ng,drot,
        Int32(gauge_theta ? 1 : 0),Int32(adiabatic_wall ? 1 : 0),Int32(drp ? 1 : 0), Int(sponge_w),Float32(sponge_max),Float32(svv_eps),Float32(cs_smag),Int32(shock_aware ? 1 : 0),Float32(nu_mol))
end

function pond_fv3dT_advance!(d::PonD3DT,t::Int)
    nd=(d.NX,d.NY,d.NZ)
    pond_fv3dT_kernel!(d.backend)(d.rhon,d.uxn,d.uyn,d.uzn,d.Tn, d.rho,d.ux,d.uy,d.uz,d.T,d.chi, d.cxa,d.cya,d.cza,d.wa,
        d.cs2,d.dt/d.dx,d.drot,d.csmag,d.gauge,d.wall,d.shock_aware,d.drp,d.nu_mol, Int32(d.NX),Int32(d.NY),Int32(d.NZ); ndrange=nd)
    KernelAbstractions.synchronize(d.backend)
    d.rho,d.rhon=d.rhon,d.rho; d.ux,d.uxn=d.uxn,d.ux; d.uy,d.uyn=d.uyn,d.uy; d.uz,d.uzn=d.uzn,d.uz; d.T,d.Tn=d.Tn,d.T
    if d.svv_eps > 0f0     # k⁶ SVV: damp the 2Δ grid-scale mode so the low-dissipation DRP runs stable at high res
        _svv3T!(d.backend)(d.rhon,d.uxn,d.uyn,d.uzn,d.Tn, d.rho,d.ux,d.uy,d.uz,d.T, d.svv_eps, Int32(d.NX),Int32(d.NY),Int32(d.NZ); ndrange=nd)
        KernelAbstractions.synchronize(d.backend)
        d.rho,d.rhon=d.rhon,d.rho; d.ux,d.uxn=d.uxn,d.ux; d.uy,d.uyn=d.uyn,d.uy; d.uz,d.uzn=d.uzn,d.uz; d.T,d.Tn=d.Tn,d.T
    end
    _penalize3!(d.backend)(d.ux,d.uy,d.uz, d.chi, Float32(d.dt/d.eta); ndrange=nd); KernelAbstractions.synchronize(d.backend)
    if d.sponge_w > 0
        _sponge3T!(d.backend)(d.rho,d.ux,d.uy,d.uz,d.T, d.rho_inf,d.U_inf,d.T_inf, Int32(d.NX),Int32(d.NY),Int32(d.NZ), Int32(d.sponge_w),d.sponge_max; ndrange=nd)
        KernelAbstractions.synchronize(d.backend)
    end
    _bc3T!(d.backend)(d.rho,d.ux,d.uy,d.uz,d.T, d.rho_inf,d.U_inf,d.T_inf, Int32(d.NX),Int32(d.NY),Int32(d.NZ),Int32(d.ng); ndrange=nd)
    KernelAbstractions.synchronize(d.backend); return nothing
end
