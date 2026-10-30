# ==============================================================================
# Particles-on-Demand — 3D DUGKS VISCOUS kinetic-FV  [controlled physical viscosity]
# ==============================================================================
# Viscous replacement for the inviscid per-axis flux kernels (pond_fv3d_gpu.jl). The inviscid
# KFVS flux has ZERO physical viscosity and a large (~0.083) NUMERICAL viscosity ⇒ effective
# Re ~ 350 ⇒ over-damped wake. This DUGKS flux (Guo-Xu-Wang 2015):
#   • reconstructs the face populations at the characteristic foot x_b − c·h FROM THE INTERFACE
#     (central state + MINMOD-limited normal gradient) ⇒ ~0 numerical viscosity in smooth flow
#     (the wake's effective Re is set by the PHYSICS), limiter only dissipates AT the shock;
#   • injects a controlled physical viscosity via the relaxation time τ through the f̄ transform:
#     f = w1·f̄ + w2·f_eq, w1=2τ/(2τ+h), w2=h/(2τ+h), f̄_neq = −((τ+h/2)/cs²)·w·ρ·Q_{ax,β}·∂_ax u_β.
#     ν set by τ (1D-calibrated, tools/test_pond_dugks1d.jl). Only the NORMAL velocity gradient is
#     resolved per face (thin-layer viscous flux — the dominant wake dissipation). h = dt/2.
# Gated by τ>0; the inviscid kernels in pond_fv3d_gpu.jl are untouched (existing results unchanged).
# Reuses _mm3 and _feq3_gpu from pond_fv3d_gpu.jl (must be included first).
# ==============================================================================

using KernelAbstractions

# face-limited normal gradient: central (R−L) limited against the one-sided diffs ⇒ ~central in
# smooth flow, →0 at a shock (shock-robust). Returns the slope (per unit dx).
# AUDIT FIX (2026-07-07): 3rd arg was aR-aRR (inverted) ⇒ gradient was IDENTICALLY ZERO on smooth
# monotone profiles (foot-trace + Chapman-Enskog viscous term silently OFF). See _glimT in
# pond_fv2d_thermal.jl for the full analysis.
@inline _glim(aLL,aL,aR,aRR) = _mm3(aR-aL, _mm3(aL-aLL, aRR-aR))

# DUGKS face flux along axis ax (1=x,2=y,3=z). Returns (Fρ,Fmx,Fmy,Fmz). τ>0 sets ν; h=dt/2.
@inline function _dugks_flux3(rhoL,uxL,uyL,uzL, rhoR,uxR,uyR,uzR,
                              rhoLL,uxLL,uyLL,uzLL, rhoRR,uxRR,uyRR,uzRR,
                              ax, cx_arr, cy_arr, cz_arr, w_arr, cs2, dx, h, tau, cs_smag)
    ufx=0.5f0*(uxL+uxR); ufy=0.5f0*(uyL+uyR); ufz=0.5f0*(uzL+uzR)
    rb0=0.5f0*(rhoL+rhoR); uxb0=0.5f0*(uxL+uxR); uyb0=0.5f0*(uyL+uyR); uzb0=0.5f0*(uzL+uzR)
    invdx=1.0f0/dx
    gρ =_glim(rhoLL,rhoL,rhoR,rhoRR)*invdx
    gux=_glim(uxLL,uxL,uxR,uxRR)*invdx; guy=_glim(uyLL,uyL,uyR,uyRR)*invdx; guz=_glim(uzLL,uzL,uzR,uzRR)*invdx
    # LES: Smagorinsky eddy viscosity ν_sgs=(Cs·Δ)²|S| (strain ~ normal gradients) ⇒ τ_eff = τ_mol+ν_sgs/cs².
    # Models the UNRESOLVED scales (VRAM-constrained) AND damps grid-scale ripples physically (∝ strain,
    # vanishes in smooth freestream) — so τ_mol sets the target (Ma-consistent) Re. cs_smag=0 ⇒ off.
    Smag = sqrt(2.0f0*(gux*gux+guy*guy+guz*guz))
    tau_eff = tau + (cs_smag*cs_smag*dx*dx*Smag)/cs2
    coef=(tau_eff+0.5f0*h)/cs2; w1=2.0f0*tau_eff/(2.0f0*tau_eff+h); w2=h/(2.0f0*tau_eff+h)
    # ---- pass 1: conserved moments of f̄(foot) ----
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
    # ---- pass 2: recover interface f = w1·f̄ + w2·f_eq(interface) and accumulate flux ----
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

# smoothness sensor: 2nd undivided difference vs 1st — ~0 on smooth/linear data, O(1) on a
# grid-scale (odd-even) oscillation. Targets the dispersive ripples the central scheme sustains.
@inline function _smooth_sensor(aLL,aL,aR,aRR)
    num = abs(aLL-2.0f0*aL+aR) + abs(aL-2.0f0*aR+aRR)
    den = abs(aL-aLL)+abs(aR-aL)+abs(aRR-aR) + 1.0f-6
    return num/den
end

# Blend: DUGKS (low dissipation, controlled ν) where the flow is smooth, robust inviscid KFVS
# (_fvflux3) where it is NOT. Two sensors feed the blend s∈[0,1]: a density-jump term (the bow
# SHOCK) and a velocity-SMOOTHNESS term (kills the grid-scale dispersive RIPPLES the central
# reconstruction would otherwise sustain) — plus a tiny baseline. KFVS only where needed ⇒ the
# wake's effective Re stays set by τ.
@inline function _flux_blend(rhoL,uxL,uyL,uzL, rhoR,uxR,uyR,uzR,
                             rhoLL,uxLL,uyLL,uzLL, rhoRR,uxRR,uyRR,uzRR,
                             ax, cx,cy,cz,w, cs2, dx, h, tau, cs_smag)
    # density-jump SHOCK sensor → robust KFVS across the bow shock (+ a tiny baseline). The grid-scale
    # VELOCITY ripples are now handled PHYSICALLY by the LES Smagorinsky ν_sgs inside _dugks_flux3
    # (∝ strain) rather than an ad-hoc smoothness sensor, so τ_mol can hold the target Re in the wake.
    # a small velocity-curvature term cleans the grid-scale freestream ripples the very-low-ν DUGKS
    # sustains (the SGS strain-estimate alone under-damps them); the SGS still sets the wake Re.
    jump = abs(rhoR-rhoL)/(rhoL<rhoR ? rhoL : rhoR)
    sj = jump<=0.1f0 ? 0.0f0 : (jump>=0.4f0 ? 1.0f0 : (jump-0.1f0)/0.3f0)
    sc = _smooth_sensor(uxLL,uxL,uxR,uxRR); sc=max(sc,_smooth_sensor(uyLL,uyL,uyR,uyRR)); sc=max(sc,_smooth_sensor(uzLL,uzL,uzR,uzRR))
    s_rip = sc<=0.5f0 ? 0.0f0 : (sc>=1.0f0 ? 0.5f0 : (sc-0.5f0))   # only the STRONGEST grid-scale wiggles, capped 0.5
    s = sj; s = s>s_rip ? s : s_rip; s = s>0.03f0 ? s : 0.03f0
    Fd1,Fd2,Fd3,Fd4 = _dugks_flux3(rhoL,uxL,uyL,uzL, rhoR,uxR,uyR,uzR, rhoLL,uxLL,uyLL,uzLL, rhoRR,uxRR,uyRR,uzRR, ax,cx,cy,cz,w,cs2,dx,h,tau,cs_smag)
    s == 0.0f0 && return Fd1,Fd2,Fd3,Fd4
    Fk1,Fk2,Fk3,Fk4 = _fvflux3(rhoL,uxL,uyL,uzL, rhoR,uxR,uyR,uzR, rhoLL,uxLL,uyLL,uzLL, rhoRR,uxRR,uyRR,uzRR, ax,cx,cy,cz,w,cs2)
    om=1.0f0-s
    return om*Fd1+s*Fk1, om*Fd2+s*Fk2, om*Fd3+s*Fk3, om*Fd4+s*Fk4
end

# ---- per-axis viscous kernels (mirror pond_flux_{x,y,z}_kernel! but with the DUGKS flux) ----
@kernel function pond_visc_x_kernel!(rho_o,mx_o,my_o,mz_o, @Const(rho),@Const(ux),@Const(uy),@Const(uz),
                                     @Const(cx),@Const(cy),@Const(cz),@Const(w),
                                     cs2::Float32,a::Float32,dx::Float32,h::Float32,tau::Float32,cs_smag::Float32, NX::Int32,NY::Int32,NZ::Int32)
    i,j,k=@index(Global,NTuple); nx=Int(NX)
    @inbounds begin
        iL=_clampi3(i-1,nx); iR=_clampi3(i+1,nx); iRR=_clampi3(i+2,nx); iLL=_clampi3(i-2,nx)
        FrR,FmxR,FmyR,FmzR=_flux_blend(rho[i,j,k],ux[i,j,k],uy[i,j,k],uz[i,j,k], rho[iR,j,k],ux[iR,j,k],uy[iR,j,k],uz[iR,j,k],
            rho[iL,j,k],ux[iL,j,k],uy[iL,j,k],uz[iL,j,k], rho[iRR,j,k],ux[iRR,j,k],uy[iRR,j,k],uz[iRR,j,k], 1,cx,cy,cz,w,cs2,dx,h,tau,cs_smag)
        FrL,FmxL,FmyL,FmzL=_flux_blend(rho[iL,j,k],ux[iL,j,k],uy[iL,j,k],uz[iL,j,k], rho[i,j,k],ux[i,j,k],uy[i,j,k],uz[i,j,k],
            rho[iLL,j,k],ux[iLL,j,k],uy[iLL,j,k],uz[iLL,j,k], rho[iR,j,k],ux[iR,j,k],uy[iR,j,k],uz[iR,j,k], 1,cx,cy,cz,w,cs2,dx,h,tau,cs_smag)
        rho_o[i,j,k]=rho[i,j,k]-a*(FrR-FrL); mx_o[i,j,k]=rho[i,j,k]*ux[i,j,k]-a*(FmxR-FmxL)
        my_o[i,j,k]=rho[i,j,k]*uy[i,j,k]-a*(FmyR-FmyL); mz_o[i,j,k]=rho[i,j,k]*uz[i,j,k]-a*(FmzR-FmzL)
    end
end
@kernel function pond_visc_y_kernel!(rho_o,mx_o,my_o,mz_o, @Const(rho),@Const(ux),@Const(uy),@Const(uz),
                                     @Const(cx),@Const(cy),@Const(cz),@Const(w),
                                     cs2::Float32,a::Float32,dx::Float32,h::Float32,tau::Float32,cs_smag::Float32, NX::Int32,NY::Int32,NZ::Int32)
    i,j,k=@index(Global,NTuple); ny=Int(NY)
    @inbounds begin
        jD=_clampi3(j-1,ny); jU=_clampi3(j+1,ny); jUU=_clampi3(j+2,ny); jDD=_clampi3(j-2,ny)
        FrT,FmxT,FmyT,FmzT=_flux_blend(rho[i,j,k],ux[i,j,k],uy[i,j,k],uz[i,j,k], rho[i,jU,k],ux[i,jU,k],uy[i,jU,k],uz[i,jU,k],
            rho[i,jD,k],ux[i,jD,k],uy[i,jD,k],uz[i,jD,k], rho[i,jUU,k],ux[i,jUU,k],uy[i,jUU,k],uz[i,jUU,k], 2,cx,cy,cz,w,cs2,dx,h,tau,cs_smag)
        FrB,FmxB,FmyB,FmzB=_flux_blend(rho[i,jD,k],ux[i,jD,k],uy[i,jD,k],uz[i,jD,k], rho[i,j,k],ux[i,j,k],uy[i,j,k],uz[i,j,k],
            rho[i,jDD,k],ux[i,jDD,k],uy[i,jDD,k],uz[i,jDD,k], rho[i,jU,k],ux[i,jU,k],uy[i,jU,k],uz[i,jU,k], 2,cx,cy,cz,w,cs2,dx,h,tau,cs_smag)
        rho_o[i,j,k]-=a*(FrT-FrB); mx_o[i,j,k]-=a*(FmxT-FmxB); my_o[i,j,k]-=a*(FmyT-FmyB); mz_o[i,j,k]-=a*(FmzT-FmzB)
    end
end
@kernel function pond_visc_z_kernel!(rho_o,mx_o,my_o,mz_o, @Const(rho),@Const(ux),@Const(uy),@Const(uz),
                                     @Const(cx),@Const(cy),@Const(cz),@Const(w),
                                     cs2::Float32,a::Float32,dx::Float32,h::Float32,tau::Float32,cs_smag::Float32, NX::Int32,NY::Int32,NZ::Int32)
    i,j,k=@index(Global,NTuple); nz=Int(NZ)
    @inbounds begin
        kB=_clampi3(k-1,nz); kF=_clampi3(k+1,nz); kFF=_clampi3(k+2,nz); kBB=_clampi3(k-2,nz)
        FrF,FmxF,FmyF,FmzF=_flux_blend(rho[i,j,k],ux[i,j,k],uy[i,j,k],uz[i,j,k], rho[i,j,kF],ux[i,j,kF],uy[i,j,kF],uz[i,j,kF],
            rho[i,j,kB],ux[i,j,kB],uy[i,j,kB],uz[i,j,kB], rho[i,j,kFF],ux[i,j,kFF],uy[i,j,kFF],uz[i,j,kFF], 3,cx,cy,cz,w,cs2,dx,h,tau,cs_smag)
        FrK,FmxK,FmyK,FmzK=_flux_blend(rho[i,j,kB],ux[i,j,kB],uy[i,j,kB],uz[i,j,kB], rho[i,j,k],ux[i,j,k],uy[i,j,k],uz[i,j,k],
            rho[i,j,kBB],ux[i,j,kBB],uy[i,j,kBB],uz[i,j,kBB], rho[i,j,kF],ux[i,j,kF],uy[i,j,kF],uz[i,j,kF], 3,cx,cy,cz,w,cs2,dx,h,tau,cs_smag)
        r=rho_o[i,j,k]-a*(FrF-FrK); mx=mx_o[i,j,k]-a*(FmxF-FmxK); my=my_o[i,j,k]-a*(FmyF-FmyK); mz=mz_o[i,j,k]-a*(FmzF-FmzK)
        # ρ floor 1e-2: in STRONG expansion (Ma1.5 cylinder shoulders) a cell can momentarily overshoot
        # ρ→~0; recovering u=m/ρ then explodes and NaNs the wake. Physical min ρ here is ~0.2, so a 1e-2
        # floor only catches the non-physical near-zero seeds (byte-identical where ρ stays >1e-2, e.g.
        # the sphere's ρ_min 0.42). + a speed cap (≈Ma 3.7) as a backstop against low-ρ runaway.
        r = r<1.0f-2 ? 1.0f-2 : r
        ux=mx/r; uy=my/r; uz=mz/r
        sp=sqrt(ux*ux+uy*uy+uz*uz)
        if sp>3.0f0; f=3.0f0/sp; ux*=f; uy*=f; uz*=f; end
        rho_o[i,j,k]=r; mx_o[i,j,k]=ux; my_o[i,j,k]=uy; mz_o[i,j,k]=uz
    end
end

# host viscous step (mirrors pond_fv3d_step!): τ sets the MOLECULAR ν (target Re); cs_smag>0 adds the
# Smagorinsky LES eddy viscosity (models unresolved scales + damps ripples). dt, dx as usual; h=dt/2.
function pond_fv3d_step_visc!(backend, rho_o,ux_o,uy_o,uz_o, rho,ux,uy,uz, cx,cy,cz,w,
                              cs2::Float32, dt::Float32, dx::Float32, tau::Float32, cs_smag::Float32, NX,NY,NZ)
    a=dt/dx; h=dt*0.5f0; nd=(Int(NX),Int(NY),Int(NZ)); nx=Int32(NX);ny=Int32(NY);nz=Int32(NZ)
    pond_visc_x_kernel!(backend)(rho_o,ux_o,uy_o,uz_o, rho,ux,uy,uz, cx,cy,cz,w, cs2,Float32(a),dx,h,tau,cs_smag, nx,ny,nz; ndrange=nd)
    KernelAbstractions.synchronize(backend)
    pond_visc_y_kernel!(backend)(rho_o,ux_o,uy_o,uz_o, rho,ux,uy,uz, cx,cy,cz,w, cs2,Float32(a),dx,h,tau,cs_smag, nx,ny,nz; ndrange=nd)
    KernelAbstractions.synchronize(backend)
    pond_visc_z_kernel!(backend)(rho_o,ux_o,uy_o,uz_o, rho,ux,uy,uz, cx,cy,cz,w, cs2,Float32(a),dx,h,tau,cs_smag, nx,ny,nz; ndrange=nd)
    KernelAbstractions.synchronize(backend)
    return nothing
end
