# ==============================================================================
# Particles-on-Demand — 3D WENO5 (WENO-Z) interface reconstruction  [less-dispersive flux]
# ==============================================================================
# Attacks the RIPPLES AT THEIR SOURCE (the dispersion of the reconstruction) instead of
# damping them after the fact (SVV). The DUGKS viscous flux (pond_fv3d_dugks.jl) reconstructs
# the interface state with a 2nd-order CENTRAL average rb0=½(L+R) — accurate but DISPERSIVE:
# its phase error grows like O(k²Δx²), so the under-resolved 3-5Δ Mach-wave fronts radiate
# spurious trailing wiggles (the "ripples"). LBM's exact streaming has NONE of this because the
# advection is dispersion-free; the FV analogue is a HIGH-ORDER, NON-OSCILLATORY reconstruction.
#
# This module reconstructs the L/R interface states with 5th-order WENO-Z (Borges 2008):
#   • 5th-order accurate in smooth flow ⇒ phase error O(k⁵Δx⁵) — ~3 orders tighter than central
#     at the resolved scales ⇒ the Mach waves are carried cleanly (low DISPERSION, low DISSIPATION);
#   • essentially NON-OSCILLATORY at discontinuities — the nonlinear weights down-bias the stencil
#     that straddles a jump ⇒ NO Gibbs wiggles at the under-resolved fronts where ripples nucleate.
# The high-order non-oscillatory interface value 0.5(ρ_Lf+ρ_Rf) replaces the central rb0 and feeds
# the SAME DUGKS τ-machinery (controlled molecular ν) + LES Smagorinsky (the wake Re is unchanged).
# A 4th-order central normal gradient (vs the minmod, which clips extrema) feeds the foot transport.
# Because WENO5 is non-oscillatory, the velocity-smoothness ripple sensor AND the SVV are no longer
# needed — only a density-jump shock blend to KFVS is kept for bow-shock robustness.
#
# Reuses _feq3_gpu, _clampi3 from pond_fv3d_gpu.jl (must be included first).
# Gated by the driver's `use_weno` flag; the DUGKS/inviscid paths are untouched.
# ==============================================================================

using KernelAbstractions

# ---- WENO-Z 5th-order reconstruction (Borges, Carmona, Costa, Don 2008, JCP 227) -------------
# Stencil cells {a,b,c,d,e} = {i-2,i-1,i,i+1,i+2}; returns the value at the RIGHT face of the
# center cell c (the i+½ face), left-biased. The 3 candidate 3rd-order reconstructions are blended
# by nonlinear WENO-Z weights: smooth ⇒ recovers the optimal 5th-order central stencil; near a
# jump ⇒ the straddling sub-stencils get ~zero weight ⇒ non-oscillatory. τ5=|β0−β2| is the WENO-Z
# global smoothness reference (sharper, less dissipative than classic WENO-JS at smooth extrema).
@inline function _weno5z(a, b, c, d, e)
    p0 = ( 2.0f0*a - 7.0f0*b + 11.0f0*c) * (1.0f0/6.0f0)
    p1 = (-1.0f0*b + 5.0f0*c +  2.0f0*d) * (1.0f0/6.0f0)
    p2 = ( 2.0f0*c + 5.0f0*d -  1.0f0*e) * (1.0f0/6.0f0)
    b0 = (13.0f0/12.0f0)*(a-2.0f0*b+c)*(a-2.0f0*b+c) + 0.25f0*(a-4.0f0*b+3.0f0*c)*(a-4.0f0*b+3.0f0*c)
    b1 = (13.0f0/12.0f0)*(b-2.0f0*c+d)*(b-2.0f0*c+d) + 0.25f0*(b-d)*(b-d)
    b2 = (13.0f0/12.0f0)*(c-2.0f0*d+e)*(c-2.0f0*d+e) + 0.25f0*(3.0f0*c-4.0f0*d+e)*(3.0f0*c-4.0f0*d+e)
    ε  = 1.0f-12
    τ5 = abs(b0 - b2)
    a0 = 0.1f0*(1.0f0 + (τ5/(b0+ε))*(τ5/(b0+ε)))
    a1 = 0.6f0*(1.0f0 + (τ5/(b1+ε))*(τ5/(b1+ε)))
    a2 = 0.3f0*(1.0f0 + (τ5/(b2+ε))*(τ5/(b2+ε)))
    inv = 1.0f0/(a0+a1+a2)
    return (a0*p0 + a1*p1 + a2*p2)*inv
end

# ---- DUGKS kinetic integration from a PRE-RECONSTRUCTED interface state + frame + normal grad --
# Same f̄ transform / two-pass moment recovery as _dugks_flux3, but the spatial reconstruction
# (rb0, gradients, frame) is supplied by the caller (WENO5 + 4th-order central). h=dt/2.
@inline function _dugks_kinetic(rb0,uxb0,uyb0,uzb0, ufx,ufy,ufz, gρ,gux,guy,guz,
                                ax, cx_arr,cy_arr,cz_arr,w_arr, cs2, h, tau_eff)
    coef=(tau_eff+0.5f0*h)/cs2; w1=2.0f0*tau_eff/(2.0f0*tau_eff+h); w2=h/(2.0f0*tau_eff+h)
    rb=0.0f0; jx=0.0f0; jy=0.0f0; jz=0.0f0
    @inbounds for k in 1:39
        cxk=cx_arr[k]; cyk=cy_arr[k]; czk=cz_arr[k]; wk=w_arr[k]
        vx=cxk+ufx; vy=cyk+ufy; vz=czk+ufz
        vn = ax==1 ? vx : (ax==2 ? vy : vz)
        cax = ax==1 ? cxk : (ax==2 ? cyk : czk)
        ρf=rb0 - vn*h*gρ; ρf = ρf<1.0f-4 ? 1.0f-4 : ρf
        uxf=uxb0 - vn*h*gux; uyf=uyb0 - vn*h*guy; uzf=uzb0 - vn*h*guz
        Qx=cax*cxk-(ax==1 ? cs2 : 0.0f0); Qy=cax*cyk-(ax==2 ? cs2 : 0.0f0); Qz=cax*czk-(ax==3 ? cs2 : 0.0f0)
        fb=_feq3_gpu(ρf,uxf,uyf,uzf,ufx,ufy,ufz,cxk,cyk,czk,wk,cs2) - coef*wk*ρf*(Qx*gux+Qy*guy+Qz*guz)
        rb+=fb; jx+=vx*fb; jy+=vy*fb; jz+=vz*fb
    end
    rb = rb<1.0f-4 ? 1.0f-4 : rb; uxb=jx/rb; uyb=jy/rb; uzb=jz/rb
    Fρ=0.0f0; Fmx=0.0f0; Fmy=0.0f0; Fmz=0.0f0
    @inbounds for k in 1:39
        cxk=cx_arr[k]; cyk=cy_arr[k]; czk=cz_arr[k]; wk=w_arr[k]
        vx=cxk+ufx; vy=cyk+ufy; vz=czk+ufz
        vn = ax==1 ? vx : (ax==2 ? vy : vz)
        cax = ax==1 ? cxk : (ax==2 ? cyk : czk)
        ρf=rb0 - vn*h*gρ; ρf = ρf<1.0f-4 ? 1.0f-4 : ρf
        uxf=uxb0 - vn*h*gux; uyf=uyb0 - vn*h*guy; uzf=uzb0 - vn*h*guz
        Qx=cax*cxk-(ax==1 ? cs2 : 0.0f0); Qy=cax*cyk-(ax==2 ? cs2 : 0.0f0); Qz=cax*czk-(ax==3 ? cs2 : 0.0f0)
        fbar=_feq3_gpu(ρf,uxf,uyf,uzf,ufx,ufy,ufz,cxk,cyk,czk,wk,cs2) - coef*wk*ρf*(Qx*gux+Qy*guy+Qz*guz)
        feqb=_feq3_gpu(rb,uxb,uyb,uzb,ufx,ufy,ufz,cxk,cyk,czk,wk,cs2)
        fk = w1*fbar + w2*feqb
        Fρ+=vn*fk; Fmx+=vn*vx*fk; Fmy+=vn*vy*fk; Fmz+=vn*vz*fk
    end
    return Fρ,Fmx,Fmy,Fmz
end

# ---- robust KFVS upwind flux from EXPLICIT (WENO-reconstructed) L/R face states ----
@inline function _kfvs_states(rhoLf,uxLf,uyLf,uzLf, rhoRf,uxRf,uyRf,uzRf, ax, cx_arr,cy_arr,cz_arr,w_arr, cs2)
    ufx=0.5f0*(uxLf+uxRf); ufy=0.5f0*(uyLf+uyRf); ufz=0.5f0*(uzLf+uzRf)
    Frho=0.0f0; Fmx=0.0f0; Fmy=0.0f0; Fmz=0.0f0
    @inbounds for k in 1:39
        cxk=cx_arr[k]; cyk=cy_arr[k]; czk=cz_arr[k]; wk=w_arr[k]
        vx=cxk+ufx; vy=cyk+ufy; vz=czk+ufz
        vn = ax==1 ? vx : (ax==2 ? vy : vz)
        fk = vn>=0.0f0 ? _feq3_gpu(rhoLf,uxLf,uyLf,uzLf,ufx,ufy,ufz,cxk,cyk,czk,wk,cs2) :
                         _feq3_gpu(rhoRf,uxRf,uyRf,uzRf,ufx,ufy,ufz,cxk,cyk,czk,wk,cs2)
        Frho+=vn*fk; Fmx+=vn*vx*fk; Fmy+=vn*vy*fk; Fmz+=vn*vz*fk
    end
    return Frho,Fmx,Fmy,Fmz
end

# ---- WENO5 face flux: reconstruct L/R via WENO-Z, DUGKS kinetic flux, density-jump KFVS blend ----
# Stencils per variable = 6 cells {i-2,i-1,i,i+1,i+2,i+3}; the face is i+½ (between the L1 and R1 cell).
@inline function _flux_blend_weno(rL2,rL1,r0,rR1,rR2,rR3,
                                  xL2,xL1,x0,xR1,xR2,xR3,
                                  yL2,yL1,y0,yR1,yR2,yR3,
                                  zL2,zL1,z0,zR1,zR2,zR3,
                                  ax, cx,cy,cz,w, cs2, dx, h, tau, cs_smag)
    # WENO-Z left/right biased face states at i+½. Left value: center cell r0 (i), right face.
    # Right value: center cell rR1 (i+1), left face = mirror stencil.
    ρLf=_weno5z(rL2,rL1,r0,rR1,rR2);   ρRf=_weno5z(rR3,rR2,rR1,r0,rL1)
    uxLf=_weno5z(xL2,xL1,x0,xR1,xR2);  uxRf=_weno5z(xR3,xR2,xR1,x0,xL1)
    uyLf=_weno5z(yL2,yL1,y0,yR1,yR2);  uyRf=_weno5z(yR3,yR2,yR1,y0,yL1)
    uzLf=_weno5z(zL2,zL1,z0,zR1,zR2);  uzRf=_weno5z(zR3,zR2,zR1,z0,zL1)
    ρLf = ρLf<1.0f-4 ? 1.0f-4 : ρLf;   ρRf = ρRf<1.0f-4 ? 1.0f-4 : ρRf
    # high-order, non-oscillatory interface state + co-moving frame
    ufx=0.5f0*(uxLf+uxRf); ufy=0.5f0*(uyLf+uyRf); ufz=0.5f0*(uzLf+uzRf)
    rb0=0.5f0*(ρLf+ρRf)
    # MINMOD-limited (TVD) normal gradient for the foot transport — robust at the shock + under-
    # resolution (an unlimited high-order gradient injects grid-scale noise where the grid can't
    # resolve the strain). The DISPERSION fix is the WENO5 interface VALUE rb0 above; the foot
    # gradient is a small τ-scaled correction, so the limiter here costs little accuracy.
    invdx=1.0f0/dx
    gρ =_glim(rL1,r0,rR1,rR2)*invdx
    gux=_glim(xL1,x0,xR1,xR2)*invdx; guy=_glim(yL1,y0,yR1,yR2)*invdx; guz=_glim(zL1,z0,zR1,zR2)*invdx
    # LES Smagorinsky eddy viscosity (sets the wake Re together with τ_mol); cs_smag=0 ⇒ off
    Smag=sqrt(2.0f0*(gux*gux+guy*guy+guz*guz))
    tau_eff=tau + (cs_smag*cs_smag*dx*dx*Smag)/cs2
    Fd1,Fd2,Fd3,Fd4=_dugks_kinetic(rb0,ufx,ufy,ufz, ufx,ufy,ufz, gρ,gux,guy,guz, ax, cx,cy,cz,w, cs2, h, tau_eff)
    # Robustness blend IDENTICAL to the central path (_flux_blend): density-jump SHOCK sensor +
    # velocity-smoothness sensor → KFVS. At physical Re (ν_mol≈0) the low-dissipation kinetic flux
    # NEEDS this sink at the slip-wall / shock or it goes unstable (WENO order alone is not enough);
    # keeping it makes the WENO path a fair drop-in where ONLY the interface VALUE rb0 changed.
    jump=abs(ρRf-ρLf)/(ρLf<ρRf ? ρLf : ρRf)
    sj = jump<=0.1f0 ? 0.0f0 : (jump>=0.4f0 ? 1.0f0 : (jump-0.1f0)*(1.0f0/0.3f0))
    sc=_smooth_sensor(xL1,x0,xR1,xR2); sc=max(sc,_smooth_sensor(yL1,y0,yR1,yR2)); sc=max(sc,_smooth_sensor(zL1,z0,zR1,zR2))
    s_rip = sc<=0.5f0 ? 0.0f0 : (sc>=1.0f0 ? 0.5f0 : (sc-0.5f0))
    s = sj; s = s>s_rip ? s : s_rip; s = s>0.03f0 ? s : 0.03f0
    Fk1,Fk2,Fk3,Fk4=_kfvs_states(ρLf,uxLf,uyLf,uzLf, ρRf,uxRf,uyRf,uzRf, ax, cx,cy,cz,w, cs2)
    om=1.0f0-s
    return om*Fd1+s*Fk1, om*Fd2+s*Fk2, om*Fd3+s*Fk3, om*Fd4+s*Fk4
end

# ---- per-axis WENO kernels (±3 stencil; mirror pond_visc_{x,y,z}_kernel!) --------------------
@kernel function pond_weno_x_kernel!(rho_o,mx_o,my_o,mz_o, @Const(rho),@Const(ux),@Const(uy),@Const(uz),
                                     @Const(cx),@Const(cy),@Const(cz),@Const(w),
                                     cs2::Float32,a::Float32,dx::Float32,h::Float32,tau::Float32,cs_smag::Float32, NX::Int32,NY::Int32,NZ::Int32)
    i,j,k=@index(Global,NTuple); nx=Int(NX)
    @inbounds begin
        iL=_clampi3(i-1,nx); iR=_clampi3(i+1,nx); iLL=_clampi3(i-2,nx); iRR=_clampi3(i+2,nx); iLLL=_clampi3(i-3,nx); iRRR=_clampi3(i+3,nx)
        FrR,FmxR,FmyR,FmzR=_flux_blend_weno(
            rho[iLL,j,k],rho[iL,j,k],rho[i,j,k],rho[iR,j,k],rho[iRR,j,k],rho[iRRR,j,k],
            ux[iLL,j,k],ux[iL,j,k],ux[i,j,k],ux[iR,j,k],ux[iRR,j,k],ux[iRRR,j,k],
            uy[iLL,j,k],uy[iL,j,k],uy[i,j,k],uy[iR,j,k],uy[iRR,j,k],uy[iRRR,j,k],
            uz[iLL,j,k],uz[iL,j,k],uz[i,j,k],uz[iR,j,k],uz[iRR,j,k],uz[iRRR,j,k],
            1,cx,cy,cz,w,cs2,dx,h,tau,cs_smag)
        FrL,FmxL,FmyL,FmzL=_flux_blend_weno(
            rho[iLLL,j,k],rho[iLL,j,k],rho[iL,j,k],rho[i,j,k],rho[iR,j,k],rho[iRR,j,k],
            ux[iLLL,j,k],ux[iLL,j,k],ux[iL,j,k],ux[i,j,k],ux[iR,j,k],ux[iRR,j,k],
            uy[iLLL,j,k],uy[iLL,j,k],uy[iL,j,k],uy[i,j,k],uy[iR,j,k],uy[iRR,j,k],
            uz[iLLL,j,k],uz[iLL,j,k],uz[iL,j,k],uz[i,j,k],uz[iR,j,k],uz[iRR,j,k],
            1,cx,cy,cz,w,cs2,dx,h,tau,cs_smag)
        rho_o[i,j,k]=rho[i,j,k]-a*(FrR-FrL); mx_o[i,j,k]=rho[i,j,k]*ux[i,j,k]-a*(FmxR-FmxL)
        my_o[i,j,k]=rho[i,j,k]*uy[i,j,k]-a*(FmyR-FmyL); mz_o[i,j,k]=rho[i,j,k]*uz[i,j,k]-a*(FmzR-FmzL)
    end
end
@kernel function pond_weno_y_kernel!(rho_o,mx_o,my_o,mz_o, @Const(rho),@Const(ux),@Const(uy),@Const(uz),
                                     @Const(cx),@Const(cy),@Const(cz),@Const(w),
                                     cs2::Float32,a::Float32,dx::Float32,h::Float32,tau::Float32,cs_smag::Float32, NX::Int32,NY::Int32,NZ::Int32)
    i,j,k=@index(Global,NTuple); ny=Int(NY)
    @inbounds begin
        jD=_clampi3(j-1,ny); jU=_clampi3(j+1,ny); jDD=_clampi3(j-2,ny); jUU=_clampi3(j+2,ny); jDDD=_clampi3(j-3,ny); jUUU=_clampi3(j+3,ny)
        FrT,FmxT,FmyT,FmzT=_flux_blend_weno(
            rho[i,jDD,k],rho[i,jD,k],rho[i,j,k],rho[i,jU,k],rho[i,jUU,k],rho[i,jUUU,k],
            ux[i,jDD,k],ux[i,jD,k],ux[i,j,k],ux[i,jU,k],ux[i,jUU,k],ux[i,jUUU,k],
            uy[i,jDD,k],uy[i,jD,k],uy[i,j,k],uy[i,jU,k],uy[i,jUU,k],uy[i,jUUU,k],
            uz[i,jDD,k],uz[i,jD,k],uz[i,j,k],uz[i,jU,k],uz[i,jUU,k],uz[i,jUUU,k],
            2,cx,cy,cz,w,cs2,dx,h,tau,cs_smag)
        FrB,FmxB,FmyB,FmzB=_flux_blend_weno(
            rho[i,jDDD,k],rho[i,jDD,k],rho[i,jD,k],rho[i,j,k],rho[i,jU,k],rho[i,jUU,k],
            ux[i,jDDD,k],ux[i,jDD,k],ux[i,jD,k],ux[i,j,k],ux[i,jU,k],ux[i,jUU,k],
            uy[i,jDDD,k],uy[i,jDD,k],uy[i,jD,k],uy[i,j,k],uy[i,jU,k],uy[i,jUU,k],
            uz[i,jDDD,k],uz[i,jDD,k],uz[i,jD,k],uz[i,j,k],uz[i,jU,k],uz[i,jUU,k],
            2,cx,cy,cz,w,cs2,dx,h,tau,cs_smag)
        rho_o[i,j,k]-=a*(FrT-FrB); mx_o[i,j,k]-=a*(FmxT-FmxB); my_o[i,j,k]-=a*(FmyT-FmyB); mz_o[i,j,k]-=a*(FmzT-FmzB)
    end
end
@kernel function pond_weno_z_kernel!(rho_o,mx_o,my_o,mz_o, @Const(rho),@Const(ux),@Const(uy),@Const(uz),
                                     @Const(cx),@Const(cy),@Const(cz),@Const(w),
                                     cs2::Float32,a::Float32,dx::Float32,h::Float32,tau::Float32,cs_smag::Float32, NX::Int32,NY::Int32,NZ::Int32)
    i,j,k=@index(Global,NTuple); nz=Int(NZ)
    @inbounds begin
        kB=_clampi3(k-1,nz); kF=_clampi3(k+1,nz); kBB=_clampi3(k-2,nz); kFF=_clampi3(k+2,nz); kBBB=_clampi3(k-3,nz); kFFF=_clampi3(k+3,nz)
        FrF,FmxF,FmyF,FmzF=_flux_blend_weno(
            rho[i,j,kBB],rho[i,j,kB],rho[i,j,k],rho[i,j,kF],rho[i,j,kFF],rho[i,j,kFFF],
            ux[i,j,kBB],ux[i,j,kB],ux[i,j,k],ux[i,j,kF],ux[i,j,kFF],ux[i,j,kFFF],
            uy[i,j,kBB],uy[i,j,kB],uy[i,j,k],uy[i,j,kF],uy[i,j,kFF],uy[i,j,kFFF],
            uz[i,j,kBB],uz[i,j,kB],uz[i,j,k],uz[i,j,kF],uz[i,j,kFF],uz[i,j,kFFF],
            3,cx,cy,cz,w,cs2,dx,h,tau,cs_smag)
        FrK,FmxK,FmyK,FmzK=_flux_blend_weno(
            rho[i,j,kBBB],rho[i,j,kBB],rho[i,j,kB],rho[i,j,k],rho[i,j,kF],rho[i,j,kFF],
            ux[i,j,kBBB],ux[i,j,kBB],ux[i,j,kB],ux[i,j,k],ux[i,j,kF],ux[i,j,kFF],
            uy[i,j,kBBB],uy[i,j,kBB],uy[i,j,kB],uy[i,j,k],uy[i,j,kF],uy[i,j,kFF],
            uz[i,j,kBBB],uz[i,j,kBB],uz[i,j,kB],uz[i,j,k],uz[i,j,kF],uz[i,j,kFF],
            3,cx,cy,cz,w,cs2,dx,h,tau,cs_smag)
        r=rho_o[i,j,k]-a*(FrF-FrK); mx=mx_o[i,j,k]-a*(FmxF-FmxK); my=my_o[i,j,k]-a*(FmyF-FmyK); mz=mz_o[i,j,k]-a*(FmzF-FmzK)
        r = r<1.0f-4 ? 1.0f-4 : r
        rho_o[i,j,k]=r; mx_o[i,j,k]=mx/r; my_o[i,j,k]=my/r; mz_o[i,j,k]=mz/r
    end
end

# host WENO step (mirrors pond_fv3d_step_visc!): WENO5 reconstruction + DUGKS τ-viscosity + LES.
function pond_fv3d_step_weno!(backend, rho_o,ux_o,uy_o,uz_o, rho,ux,uy,uz, cx,cy,cz,w,
                              cs2::Float32, dt::Float32, dx::Float32, tau::Float32, cs_smag::Float32, NX,NY,NZ)
    a=dt/dx; h=dt*0.5f0; nd=(Int(NX),Int(NY),Int(NZ)); nx=Int32(NX);ny=Int32(NY);nz=Int32(NZ)
    pond_weno_x_kernel!(backend)(rho_o,ux_o,uy_o,uz_o, rho,ux,uy,uz, cx,cy,cz,w, cs2,Float32(a),dx,h,tau,cs_smag, nx,ny,nz; ndrange=nd)
    KernelAbstractions.synchronize(backend)
    pond_weno_y_kernel!(backend)(rho_o,ux_o,uy_o,uz_o, rho,ux,uy,uz, cx,cy,cz,w, cs2,Float32(a),dx,h,tau,cs_smag, nx,ny,nz; ndrange=nd)
    KernelAbstractions.synchronize(backend)
    pond_weno_z_kernel!(backend)(rho_o,ux_o,uy_o,uz_o, rho,ux,uy,uz, cx,cy,cz,w, cs2,Float32(a),dx,h,tau,cs_smag, nx,ny,nz; ndrange=nd)
    KernelAbstractions.synchronize(backend)
    return nothing
end
