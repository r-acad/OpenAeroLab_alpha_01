# ==============================================================================
# Particles-on-Demand — PURE 2D THERMAL conservative kinetic-FV  [energy equation / real T]
# ==============================================================================
# Upgrades the isothermal 2D solver (pond_fv2d_driver.jl) to a genuinely COMPRESSIBLE THERMAL gas:
# carries a 4th conserved variable — total energy density ρE — so temperature is a real, evolving
# field and the shocks are ADIABATIC (Rankine–Hugoniot, with shock heating + total-temperature rise),
# NOT isothermal. The marginalized D3Q39 set is treated as a 3D gas in 2D flow ⇒ D=3 internal DOF,
# γ=(D+2)/D=5/3, R=cs²=2/3, sound speed a=√(γRT).
#
# Thermal Hermite equilibrium = the isothermal Grad equilibrium + a temperature term:
#   f_eq_k = w_k ρ[ 1 + (c·p)/cs² + ((c·p)²−cs²|p|²)/(2cs⁴) + (θ−1)/(2cs²)(|c|²/cs² − D) ],   p=u−u_f, θ=T.
# Its moments give Σf=ρ, Σf c=ρp, stress Σf cc=ρpp+ρcs²θ I=ρpp+pI (p=ρRT, THERMAL pressure!), and
# energy Σf|c|²/2=ρ(|p|²/2+(D/2)cs²θ). The lab energy flux Σ v_n(|v|²/2)f (v=c+u_f, including the z
# thermal component c_z) is the enthalpy flux (ρE+p)u_n ⇒ the compressible Euler energy equation.
# Co-moving per-face frame u_f keeps the equilibrium accurate at supersonic u (the PonD trick).
# Reuses _mm/_clampi/_penalize2!/pond_cyl2d_chi from pond_fv_gpu.jl + pond_fv2d_driver.jl.
# ==============================================================================

using KernelAbstractions

# MC (monotonized-central) TVD slope limiter: non-oscillatory (no ripples) but allows up to 2× steeper
# slopes than minmod ⇒ markedly LOWER numerical dissipation on the resolved (wake/turbulence) scales.
@inline function _mc(a,b)
    a*b<=0.0f0 && return 0.0f0
    s = a>0.0f0 ? 1.0f0 : -1.0f0
    return s*min(0.5f0*abs(a+b), 2.0f0*abs(a), 2.0f0*abs(b))
end

const TGAS_D = 3.0f0          # internal DOF (3D gas, 2D flow)
const TGAS_GAMMA = 5.0f0/3.0f0

# 3rd-order THERMAL Hermite equilibrium population k at (ρ,ux,uy,T) in co-moving face frame (ufx,ufy).
# f_eq/w = ρ[ 1 + (c·p)/cs²                                              (mass+momentum)
#            + ((c·p)²−cs²|p|²)/(2cs⁴)                                   (2nd-order velocity / stress)
#            + (θ−1)(|c|²/cs²−D)/2                                       (2nd-order THERMAL → p=ρRT)
#            + ((c·p)³−3cs²(c·p)|p|²)/(6cs⁶)                             (3rd-order velocity, O(u³))
#            + (θ−1)(c·p)(|c|²−(D+2)cs²)/(2cs⁴) ]                        (3rd-order HEAT-FLUX coupling)
# p = u−u_f (peculiar velocity), θ=T, D=3, cs²=R T_0. The 2nd-order thermal coefficient makes the
# energy moment ρ(½|p|²+(D/2)cs²θ) exact ⇒ T=ρE/ρ−½|u|² is self-consistent; the 3rd-order heat-flux
# term fixes the energy-flux closure at strong gradients (the shock) — without it T_stag over-heats.
#
# √θ GAUGE (full PonD λ={u,T}, Dorschner-Bösch-Karlin PRL 121 130602 2018): pass isq=1/√θ_f, invth=1/θ_f
# (θ_f = face gauge temperature) ⇒ the equilibrium is expanded around the LOCAL gauge, so the Hermite
# arguments (reduced peculiar velocity px̂=(u−u_f)/√θ, reduced thermal tĥ=T/θ−1) stay SMALL even across
# strong shocks ⇒ lifts the fixed-lattice a<3 sound-speed cap WITHOUT enlarging the velocity set. The
# FIXED frame (current scheme) is the special case isq=1, invth=1 (px̂=u−u_f, tĥ=T−1) — byte-identical.
@inline function _feqT2_gpu(rho, ux, uy, T, ufx, ufy, cxk, cyk, czk, wk, cs2, isq, invth)
    px = (ux - ufx)*isq; py = (uy - ufy)*isq
    inv_cs2 = 1.0f0/cs2; inv_2cs4 = 0.5f0/(cs2*cs2); inv_6cs6 = 1.0f0/(6.0f0*cs2*cs2*cs2)
    cp = px*cxk + py*cyk                                  # c·p̂
    p2 = px*px + py*py                                    # |p̂|²
    csq = cxk*cxk + cyk*cyk + czk*czk                     # |c|² (full 3D lattice)
    th = T*invth - 1.0f0                                  # reduced thermal deviation T/θ − 1
    f  = rho
    f += rho*cp*inv_cs2                                                   # momentum
    f += rho*(cp*cp - cs2*p2)*inv_2cs4                                    # 2nd-order velocity
    f += rho*th*0.5f0*(csq*inv_cs2 - TGAS_D)                              # 2nd-order thermal (corrected factor)
    f += rho*(cp*cp*cp - 3.0f0*cs2*cp*p2)*inv_6cs6                        # 3rd-order velocity
    f += rho*th*cp*(csq - (TGAS_D+2.0f0)*cs2)*inv_2cs4                    # 3rd-order heat-flux coupling
    return max(wk*f, 1.0f-12)
end

# LEFT/RIGHT face reconstruction, blended by the shock sensor s (s=1 ⇒ minmod at shocks). The smooth
# (1-s) branch is either MC (drp=0) or a DRP κ=1/3 3rd-order upwind-biased reconstruction (drp=1) — the
# lowest-L1, lowest-dispersion linear reconstruction on the available 4-point stencil {LL,L,R,RR}. Note:
# the residual PHASE (dispersion) error is set by the kinetic flux, NOT the reconstruction (verified: it
# is identical for minmod/MC/κ=1/3), so DRP lowers the amplitude/L1 error, not the phase lag. drp=0 ⇒
# byte-identical MC/minmod. floor is applied by the caller.
@inline function _faceL(drp, s, fLL, fL, fR)
    hi = drp==Int32(1) ? (-fLL + 5.0f0*fL + 2.0f0*fR)*0.16666667f0 : fL + 0.5f0*_mc(fL-fLL, fR-fL)
    lo = fL + 0.5f0*_mm(fL-fLL, fR-fL)
    return (1.0f0-s)*hi + s*lo
end
@inline function _faceR(drp, s, fL, fR, fRR)
    hi = drp==Int32(1) ? (2.0f0*fL + 5.0f0*fR - fRR)*0.16666667f0 : fR - 0.5f0*_mc(fR-fL, fRR-fR)
    lo = fR - 0.5f0*_mm(fR-fL, fRR-fR)
    return (1.0f0-s)*hi + s*lo
end

# WENO5-Z (Borges 2008) LEFT-biased reconstruction of the face value at cell c's right edge, from the
# 5-point stencil {a,b,c,d,e}. Low dissipation AND low dispersion on smooth data (5th order) + essentially
# non-oscillatory at jumps (nonlinear weights down-bias the straddling stencil ⇒ no Gibbs ripple). Feed the
# stencil in DECREASING index order to get the RIGHT-biased state at the same face (mirror symmetry).
@inline function _weno5z2(a,b,c,d,e)
    p0=(2.0f0*a-7.0f0*b+11.0f0*c)*0.16666667f0; p1=(-b+5.0f0*c+2.0f0*d)*0.16666667f0; p2=(2.0f0*c+5.0f0*d-e)*0.16666667f0
    t0=a-2.0f0*b+c; s0=a-4.0f0*b+3.0f0*c; β0=1.0833333f0*t0*t0+0.25f0*s0*s0
    t1=b-2.0f0*c+d; β1=1.0833333f0*t1*t1+0.25f0*(b-d)*(b-d)
    t2=c-2.0f0*d+e; s2=3.0f0*c-4.0f0*d+e; β2=1.0833333f0*t2*t2+0.25f0*s2*s2
    ε=1.0f-12; τ=abs(β0-β2); r0=τ/(β0+ε); r1=τ/(β1+ε); r2=τ/(β2+ε)
    a0=0.1f0*(1.0f0+r0*r0); a1=0.6f0*(1.0f0+r1*r1); a2=0.3f0*(1.0f0+r2*r2)
    return (a0*p0+a1*p1+a2*p2)/(a0+a1+a2)
end
# blend: WENO5-Z where smooth, → 2nd-order minmod at shocks/walls (sensor s∈[0,1], s=1 at a jump) for
# robustness (positivity, immersed-wall discontinuity). c = the upwind cell; face is c's right edge.
@inline _wenoblend(s,a,b,c,d,e) = (1.0f0-s)*_weno5z2(a,b,c,d,e) + s*(c+0.5f0*_mm(c-b,d-c))

# thermal face flux (Fρ,Fmx,Fmy,FE) along axis ax from L/R primitives (ρ,u,T) + slope neighbours.
# `drot` = internal (rotational) DOF beyond the lattice's 3 translational ⇒ general γ: for air
# drot=2 (D_eff=5, γ=1.4); drot=0 recovers the monatomic lattice (γ=5/3, byte-identical). The
# rotational energy e_rot=(drot/2)·R·T advects with the populations ⇒ added into the energy flux.
# ---- Roadmap 1.2: Dynamic Smagorinsky (Germano–Lilly) — per-cell c_d, no tuned constant -----------------
# Replaces the fixed Cs² with a coefficient computed inline from the local flow via the Germano identity:
# auto-zeros in laminar/freestream, rises in turbulent shear. A separate pass fills a c_d field (like the LBM
# kernel_dynamic_sgs.jl); the flux then reads the face-averaged c_d. Simplified local form (Premnath et al.
# 2009): 3×3 top-hat test filter (width ratio α²), deviatoric Leonard stress L, modeled stress
# M = |S|S − α²|S̃|S̃ (GRID-minus-TEST, grid term kept unfiltered), Lilly least-squares c_d = ½⟨L:M⟩/⟨M:M⟩,
# clamped [0, 0.04]. (The M sign convention matters: test-minus-grid requires −½ — see the AUDIT FIX below.)
@inline function _tf3(ux,uy,a,b,nx,ny)          # 3×3 top-hat filtered (ũx,ũy) at (a,b)
    fx=0.0f0; fy=0.0f0
    @inbounds for dj in -1:1, di in -1:1
        p=_clampi(a+di,nx); q=_clampi(b+dj,ny); fx+=ux[p,q]; fy+=uy[p,q]
    end
    return fx*(1.0f0/9.0f0), fy*(1.0f0/9.0f0)
end
# Pass 1: per-cell Germano contractions L:M and M:M (stored raw; the division is deferred to pass 2 so the
# numerator/denominator can be spatially averaged — this is what makes the local dynamic procedure stable).
@kernel function _dyn_lmmm2T!(lm, mm, @Const(ux), @Const(uy), alpha2::Float32, NX::Int32, NY::Int32)
    i,j=@index(Global,NTuple); nx=Int(NX); ny=Int(NY)
    @inbounds begin
        iR=_clampi(i+1,nx);iL=_clampi(i-1,nx);jU=_clampi(j+1,ny);jD=_clampi(j-1,ny)
        fux=0.0f0;fuy=0.0f0;fxx=0.0f0;fxy=0.0f0;fyy=0.0f0
        for dj in -1:1, di in -1:1
            p=_clampi(i+di,nx); q=_clampi(j+dj,ny); u=ux[p,q]; v=uy[p,q]
            fux+=u;fuy+=v;fxx+=u*u;fxy+=u*v;fyy+=v*v
        end
        fux/=9.0f0;fuy/=9.0f0;fxx/=9.0f0;fxy/=9.0f0;fyy/=9.0f0
        Lxx=fxx-fux*fux; Lyy=fyy-fuy*fuy; Lxy=fxy-fux*fuy
        Lkk=0.5f0*(Lxx+Lyy); Lxx-=Lkk; Lyy-=Lkk                          # deviatoric
        dudx=0.5f0*(ux[iR,j]-ux[iL,j]); dudy=0.5f0*(ux[i,jU]-ux[i,jD])
        dvdx=0.5f0*(uy[iR,j]-uy[iL,j]); dvdy=0.5f0*(uy[i,jU]-uy[i,jD])
        Sxx=dudx; Syy=dvdy; Sxy=0.5f0*(dudy+dvdx); Smag=sqrt(2.0f0*(Sxx*Sxx+2.0f0*Sxy*Sxy+Syy*Syy))
        uxE,uyE=_tf3(ux,uy,iR,j,nx,ny); uxW,uyW=_tf3(ux,uy,iL,j,nx,ny)
        uxN,uyN=_tf3(ux,uy,i,jU,nx,ny); uxS,uyS=_tf3(ux,uy,i,jD,nx,ny)
        Txx=0.5f0*(uxE-uxW); Tyy=0.5f0*(uyN-uyS); Txy=0.5f0*(0.5f0*(uxN-uxS)+0.5f0*(uyE-uyW))
        Tmag=sqrt(2.0f0*(Txx*Txx+2.0f0*Txy*Txy+Tyy*Tyy))
        # AUDIT FIX (2026-07-07): M must be GRID-minus-TEST for the +½·⟨L:M⟩/⟨M:M⟩ Lilly solution.
        # Derivation: τ^d=−2c_d·Δ²|S|S and T^d=−2c_d·Δ̂²|S̃|S̃ give L^d = T−τ̃ = −2c_d·(α²|S̃|S̃−|S|S)
        # = +2c_d·M_thisform ⇒ c_d = +½⟨L:M⟩/⟨M:M⟩. The old test-minus-grid M returned c_d = −c_d_true,
        # which the [0,0.04] clamp + I_LM≥0 clip pinned to ~0 exactly where the SGS should act (the
        # dynamic model was silently inert in forward-cascade turbulence, active only on backscatter).
        # NOTE: the sibling LBM kernel_dynamic_sgs.jl carries the same sign misconception — separate fix.
        Mxx=Smag*Sxx-alpha2*Tmag*Txx; Myy=Smag*Syy-alpha2*Tmag*Tyy; Mxy=Smag*Sxy-alpha2*Tmag*Txy
        lm[i,j]=Lxx*Mxx+2.0f0*Lxy*Mxy+Lyy*Myy
        mm[i,j]=Mxx*Mxx+2.0f0*Mxy*Mxy+Myy*Myy
    end
end
# Pass 2: c_d = ½·⟨L:M⟩/⟨M:M⟩ with a 3×3 average of BOTH contractions (Lilly least-squares, local-averaged).
# In uniform flow ⟨L:M⟩→0 faster than the averaged denominator ⇒ c_d→0 (no spurious freestream SGS); in the
# turbulent wake the coherent L:M survives the average ⇒ c_d rises. Clamp [0, Cs²≈0.2²].
@kernel function _dyn_cd_avg2T!(cd, @Const(lm), @Const(mm), mmmax::Float32, NX::Int32, NY::Int32)
    i,j=@index(Global,NTuple); nx=Int(NX); ny=Int(NY)
    @inbounds begin
        slm=0.0f0; smm=0.0f0
        for dj in -1:1, di in -1:1
            p=_clampi(i+di,nx); q=_clampi(j+dj,ny); slm+=lm[p,q]; smm+=mm[p,q]
        end
        slm*=(1.0f0/9.0f0); smm*=(1.0f0/9.0f0)
        # mild scale-aware denominator floor (∝ max M:M) — prevents freestream noise-pinning without killing
        # the wake. NOTE: the local procedure is noise-limited (see roadmap 1.2); Lagrangian averaging is the
        # production-robust upgrade. c_d clamped [0, Cs²≈0.2²].
        cdv=0.5f0*slm/(smm+1.0f-4*mmmax)
        cd[i,j]= cdv<0.0f0 ? 0.0f0 : (cdv>0.04f0 ? 0.04f0 : cdv)
    end
end
# Roadmap 1.2 (robust): LAGRANGIAN dynamic averaging (Meneveau-Lund-Cabot 1996). Two memory fields I_LM, I_MM
# are relaxed toward the instantaneous L:M, M:M AND advected along pathlines: I^{n+1}(x)=ε·(local)+(1−ε)·I^n(x−u·dt).
# Each cell carries its OWN ⟨L:M⟩/⟨M:M⟩ history ⇒ freestream noise averages to ~0 while the wake's coherent
# signal survives — no global reference, immune to the body-surface strain peak that defeats the local form.
# ε = (dt/T)/(1+dt/T), T = θ·Δ·(I_LM·I_MM)^{−1/8} (the local turbulence timescale). I_LM clipped ≥0 (no
# backscatter); prod≤0 ⇒ ε=1 (seed the field with the local value). c_d = ½·I_LM/I_MM, clamp [0, Cs²≈0.2²].
@kernel function _dyn_lagr2T!(ILMo, IMMo, cd, @Const(ILM), @Const(IMM), @Const(lm), @Const(mm),
                              @Const(ux), @Const(uy), adv::Float32, dt::Float32, theta::Float32, NX::Int32, NY::Int32)
    i,j=@index(Global,NTuple); nx=Int(NX); ny=Int(NY)
    @inbounds begin
        LM=lm[i,j]; MM=mm[i,j]
        xd=Float32(i)-ux[i,j]*adv; yd=Float32(j)-uy[i,j]*adv       # departure point x − u·dt (cells)
        # NaN GUARD (2026-07-07): floor(Int, NaN) THROWS on device — a single NaN velocity anywhere used
        # to kill the whole run with an opaque KernelException (this kernel was the de-facto NaN canary).
        # Fall back to the cell's own position; the real NaN (if any) still surfaces in diagnostics/loads.
        isfinite(xd) || (xd=Float32(i)); isfinite(yd) || (yd=Float32(j))
        xc = xd<1.0f0 ? 1.0f0 : (xd>Float32(nx) ? Float32(nx) : xd)
        yc = yd<1.0f0 ? 1.0f0 : (yd>Float32(ny) ? Float32(ny) : yd)
        i0=floor(Int,xc); j0=floor(Int,yc); i1=min(i0+1,nx); j1=min(j0+1,ny); fx=xc-Float32(i0); fy=yc-Float32(j0)
        ilm_d=(1.0f0-fx)*(1.0f0-fy)*ILM[i0,j0]+fx*(1.0f0-fy)*ILM[i1,j0]+(1.0f0-fx)*fy*ILM[i0,j1]+fx*fy*ILM[i1,j1]
        imm_d=(1.0f0-fx)*(1.0f0-fy)*IMM[i0,j0]+fx*(1.0f0-fy)*IMM[i1,j0]+(1.0f0-fx)*fy*IMM[i0,j1]+fx*fy*IMM[i1,j1]
        prod=ilm_d*imm_d
        eps = prod<=0.0f0 ? 1.0f0 : (r=dt/max(theta*(prod^(-0.125f0)),1.0f-6); r/(1.0f0+r))
        ilm_n=eps*LM+(1.0f0-eps)*ilm_d; imm_n=eps*MM+(1.0f0-eps)*imm_d
        ilm_n= ilm_n<0.0f0 ? 0.0f0 : ilm_n                         # clip backscatter (I_LM ≥ 0)
        imm_n= imm_n<1.0f-30 ? 1.0f-30 : imm_n
        ILMo[i,j]=ilm_n; IMMo[i,j]=imm_n
        cdv=0.5f0*ilm_n/imm_n
        cd[i,j]= cdv<0.0f0 ? 0.0f0 : (cdv>0.04f0 ? 0.04f0 : cdv)
    end
end

@inline function _fvflux2T(rhoL,uxL,uyL,TL, rhoR,uxR,uyR,TR,
                           rhoLL,uxLL,uyLL,TLL, rhoRR,uxRR,uyRR,TRR,
                           rhoLLL,uxLLL,uyLLL,TLLL, rhoRRR,uxRRR,uyRRR,TRRR,
                           ax, cx_arr, cy_arr, cz_arr, w_arr, cs2, drot, csmag, gauge, shock_aware, drp, weno, nu_coeff)
    ufx = 0.5f0*(uxL+uxR); ufy = 0.5f0*(uyL+uyR); erc = 0.5f0*drot*cs2   # e_rot/mass = (drot/2)·R·T, R=cs²
    # Reconstruction: csmag=0 ⇒ minmod (robust, clean DEFAULT, validated on Sod/oblique-shock/general-γ).
    # csmag>0 ⇒ "LES mode": MC limiter (up to 2× LESS dissipative — 1D peak retained →1.000 vs minmod
    # 0.90→0.99, resolves the wake/turbulence) + a Smagorinsky SGS eddy-viscosity (below) that damps the
    # under-resolved grid-scale strain ⇒ clean low-dissipation at high Reynolds. Uniform branch (no divergence).
    les = csmag > 0.0f0
    # s = blend weight toward minmod: NOT-LES/WENO ⇒ s=1 (pure minmod, byte-identical to the validated default);
    # LES/WENO ⇒ s from a shock/contact sensor so the low-dissipation branch is used where smooth, minmod at shocks.
    s = 1.0f0; sp = 1.0f0
    if les || weno == Int32(1)
        pLm=rhoL*cs2*TL; pRm=rhoR*cs2*TR
        spj = abs(pRm-pLm)/min(pLm,pRm)                             # PRESSURE-only jump ⇒ true shocks (NOT contacts/shear)
        sj = max(spj, abs(rhoR-rhoL)/min(rhoL,rhoR))                # combined (shocks+contacts) ⇒ reconstruction→minmod
        s  = sj<=0.1f0  ? 0.0f0 : (sj>=0.4f0  ? 1.0f0 : (sj-0.1f0)*(1.0f0/0.3f0))
        # LES gate ramps only for STRONG shocks (|Δp|/p∈[0.8,2.5]): the bow shock (~6.6 at Ma2.5) gates LES off,
        # but the turbulent-wake eddy SHOCKLETS (~0.1-0.4) do NOT — so firm LES fires in the wake and damps the
        # shear-layer speckle. (The earlier [0.1,0.4] ramp wrongly gated LES off across the whole shocklet wake.)
        sp = spj<=0.8f0 ? 0.0f0 : (spj>=2.5f0 ? 1.0f0 : (spj-0.8f0)*(1.0f0/1.7f0))
    end
    if weno == Int32(1)     # WENO5-Z (low-dissipation, non-oscillatory) — the genuine less-dissipation-AND-clean path
        rhoLf = max(_wenoblend(s,rhoLLL,rhoLL,rhoL,rhoR,rhoRR), 1.0f-4); rhoRf = max(_wenoblend(s,rhoRRR,rhoRR,rhoR,rhoL,rhoLL), 1.0f-4)
        uxLf = _wenoblend(s,uxLLL,uxLL,uxL,uxR,uxRR); uxRf = _wenoblend(s,uxRRR,uxRR,uxR,uxL,uxLL)
        uyLf = _wenoblend(s,uyLLL,uyLL,uyL,uyR,uyRR); uyRf = _wenoblend(s,uyRRR,uyRR,uyR,uyL,uyLL)
        TLf  = max(_wenoblend(s,TLLL,TLL,TL,TR,TRR), 1.0f-3); TRf  = max(_wenoblend(s,TRRR,TRR,TR,TL,TLL), 1.0f-3)
    else
        rhoLf = max(_faceL(drp,s,rhoLL,rhoL,rhoR), 1.0f-4); rhoRf = max(_faceR(drp,s,rhoL,rhoR,rhoRR), 1.0f-4)
        uxLf = _faceL(drp,s,uxLL,uxL,uxR); uxRf = _faceR(drp,s,uxL,uxR,uxRR)
        uyLf = _faceL(drp,s,uyLL,uyL,uyR); uyRf = _faceR(drp,s,uyL,uyR,uyRR)
        TLf  = max(_faceL(drp,s,TLL,TL,TR), 1.0f-3); TRf = max(_faceR(drp,s,TL,TR,TRR), 1.0f-3)
    end
    # √θ GAUGE: gauge>0 ⇒ scale velocities v_k=√θ_f·c_k+u_f (θ_f=face temp) + expand equilibrium around
    # the gauge (isq=1/√θ, invth=1/θ). gauge=0 ⇒ sq=isq=invth=1 ⇒ FIXED frame, byte-identical.
    # SHOCK-AWARE gauge: shock_aware>0 ⇒ the frame (u_f,θ_f) is the ROE average of the reconstructed face
    # states (√ρ-weighted velocity + enthalpy), which is the entropy-consistent linearization frame for a
    # single shock/contact — it reduces the KFVS entropy generation (post-shock over-heating) vs the plain
    # arithmetic average. Falls back to the arithmetic gauge when off. Only active together with the gauge.
    sq = 1.0f0; isq = 1.0f0; invth = 1.0f0
    if gauge > 0
        if shock_aware > 0
            sL=sqrt(rhoLf); sR=sqrt(rhoRf); wR=sR/(sL+sR); wL=1.0f0-wR
            cp = 0.5f0*(5.0f0+drot)*cs2                                  # c_p = c_v+R = (D_eff/2+1)cs²
            hL = cp*TLf + 0.5f0*(uxLf*uxLf+uyLf*uyLf); hR = cp*TRf + 0.5f0*(uxRf*uxRf+uyRf*uyRf)
            ufx = wL*uxLf + wR*uxRf; ufy = wL*uyLf + wR*uyRf
            hroe = wL*hL + wR*hR; θ = max((hroe - 0.5f0*(ufx*ufx+ufy*ufy))/cp, 1.0f-3)
        else
            θ = max(0.5f0*(TLf+TRf), 1.0f-3)
        end
        sq = sqrt(θ); isq = 1.0f0/sq; invth = 1.0f0/θ
    end
    Frho=0.0f0; Fmx=0.0f0; Fmy=0.0f0; FE=0.0f0
    @inbounds for k in 1:39
        cxk=cx_arr[k]; cyk=cy_arr[k]; czk=cz_arr[k]; wk=w_arr[k]
        vx = sq*cxk+ufx; vy = sq*cyk+ufy; vz = sq*czk; vn = ax==1 ? vx : vy
        if vn>=0.0f0; fk=_feqT2_gpu(rhoLf,uxLf,uyLf,TLf,ufx,ufy,cxk,cyk,czk,wk,cs2,isq,invth); Ts=TLf
        else;         fk=_feqT2_gpu(rhoRf,uxRf,uyRf,TRf,ufx,ufy,cxk,cyk,czk,wk,cs2,isq,invth); Ts=TRf; end
        ke = 0.5f0*(vx*vx + vy*vy + vz*vz) + erc*Ts   # translational+kinetic + advected ROTATIONAL energy
        Frho += vn*fk; Fmx += vn*vx*fk; Fmy += vn*vy*fk; FE += vn*ke*fk
    end
    if les    # SHOCK-GATED SGS eddy viscosity ν_sgs = (1-sp)·c·Δ²|∂ₙu| (Δ=1). c = Cs² (static) OR Germano-Lilly c_d.
        # Gate on the PRESSURE-only sensor sp (NOT the combined s): a shear layer/contact has a DENSITY jump but no
        # pressure jump, so keying on s wrongly switched LES OFF in the shear layers — exactly where the grid-scale
        # speckle lives. sp→1 only at true shocks (keeps the bow shock sharp); sp≈0 in shear layers ⇒ FULL LES drain.
        du=uxR-uxL; dv=uyR-uyL; rf=0.5f0*(rhoL+rhoR); nu=(1.0f0-sp)*nu_coeff*sqrt(du*du+dv*dv)
        Fmx += -rf*nu*du; Fmy += -rf*nu*dv; FE += -rf*nu*(ufx*du+ufy*dv)
    end
    return Frho, Fmx, Fmy, FE
end

# ============================================================================================================
# G1 — 2D-THERMAL DUGKS (gas-kinetic, collision-IN-flux) — the low-dissipation gas-kinetic flux.
# ============================================================================================================
# Port of _dugks_flux3 (3D isothermal) to the thermal solver's frame, carrying the energy moment ρE (real T).
# WHY: the KFVS flux above is COLLISIONLESS (equilibrium upwind-split) ⇒ O(Δx·|c|) numerical viscosity that
# SMEARS the transonic suction pocket. DUGKS builds the interface distribution f = w1·f̄ + w2·f_eq, where
#   f̄ = f_eq(foot) − coef·w·ρ·(Q·∇u)          [Chapman-Enskog viscous stress IN the flux, coef=(τ_eff+h/2)/cs²]
#   w1 = 2τ_eff/(2τ_eff+h),  w2 = h/(2τ_eff+h),  h = dt/2
# so the PHYSICAL Navier-Stokes viscosity comes from τ_eff (=ν/cs²) and the numerical dissipation is O(τ) small.
# Fixed frame (no √θ gauge — accurate to Ma≲1.6 unshifted, fine for Ma 0.73 + a weak pocket); the gauge is a
# documented refinement. Energy: translational 2nd moment of the interface f + advected rotational erc·T_iface.
# HONESTY NOTES (audit 2026-07-07): (a) the CE term carries only FACE-NORMAL gradients (thin-layer form, like
# the 3D template) — the delivered x-momentum viscous operator is ∂x(2μ∂x u)+∂y(μ∂y u): the conjugate shear
# (∂y(μ∂x v)) and the −⅔μ∇·u trace are missing and the normal stress is doubled. Dominant BL terms ARE captured.
# (b) h is passed as dt/(2dx) while τ=ν/cs² is absolute time ⇒ consistent ONLY for dx=1 (all current callers).
# (c) the foot-trace itself contributes O(h) extra viscosity on top of cs²τ (the 3D path calibrates this with a
# table; the 2D path does not yet) — validate ν_eff on a 1D diffusion gate before trusting Re-sensitive results.
# face-gradient limiter: central (R−L) limited against the two one-sided OUTER slopes (L−LL and RR−R).
# AUDIT FIX (2026-07-07): the 3rd argument was inverted (aR-aRR = −(RR−R)); on ANY smooth monotone profile
# the inner minmod then saw opposite signs and returned 0, so the face gradient was IDENTICALLY ZERO in
# smooth flow — the DUGKS foot-trace upwinding AND the Chapman-Enskog viscous stress were silently OFF
# (the flux degenerated to a non-dissipative central equilibrium flux; the _sfsen ripple guard was masking
# it). Same fix applied to _glim in pond_fv3d_dugks.jl (the 3D template the bug was ported from).
@inline _glimT(aLL,aL,aR,aRR) = _mm(aR-aL, _mm(aL-aLL, aRR-aR))
@inline function _sfsen(aLL,aL,aR,aRR)   # smoothness sensor: 2nd/1st undivided difference (~0 smooth, O(1) on 2Δ wiggle)
    num=abs(aLL-2.0f0*aL+aR)+abs(aL-2.0f0*aR+aRR); den=abs(aL-aLL)+abs(aR-aL)+abs(aRR-aR)+1.0f-6; return num/den
end
@inline function _dugks_flux2T(rhoL,uxL,uyL,TL, rhoR,uxR,uyR,TR,
                               rhoLL,uxLL,uyLL,TLL, rhoRR,uxRR,uyRR,TRR,
                               ax, cx_arr,cy_arr,cz_arr,w_arr, cs2, drot, h, tau, sgs_coef)
    ufx=0.5f0*(uxL+uxR); ufy=0.5f0*(uyL+uyR); erc=0.5f0*drot*cs2; eint=0.5f0*(3.0f0+drot)*cs2
    rb0=0.5f0*(rhoL+rhoR); Tb0=0.5f0*(TL+TR)
    gρ=_glimT(rhoLL,rhoL,rhoR,rhoRR); gux=_glimT(uxLL,uxL,uxR,uxRR); guy=_glimT(uyLL,uyL,uyR,uyRR); gT=_glimT(TLL,TL,TR,TRR)
    # sgs_coef = the ALREADY-SQUARED SGS coefficient (static Cs² or the per-face Germano-Lilly c_d).
    # AUDIT FIX (2026-07-07): was the raw cs_smag squared here — the dynamic c_d never reached the DUGKS
    # branch (the dominant smooth-flow flux path), silently degrading dynamic_sgs=1 to static Cs².
    Smag=sqrt(2.0f0*(gux*gux+guy*guy)); tau_eff=tau + (sgs_coef*Smag)/cs2          # dx=1
    coef=(tau_eff+0.5f0*h)/cs2; w1=2.0f0*tau_eff/(2.0f0*tau_eff+h); w2=h/(2.0f0*tau_eff+h)
    # pass 1: interface conserved moments (mass, momentum, energy) of the transported f̄(foot)
    rb=0.0f0; jx=0.0f0; jy=0.0f0; en=0.0f0
    @inbounds for k in 1:39
        cxk=cx_arr[k]; cyk=cy_arr[k]; czk=cz_arr[k]; wk=w_arr[k]
        vx=cxk+ufx; vy=cyk+ufy; vz=czk; vn = ax==1 ? vx : vy; cax = ax==1 ? cxk : cyk
        ρf=rb0-vn*h*gρ; ρf=ρf<1.0f-4 ? 1.0f-4 : ρf
        uxf=ufx-vn*h*gux; uyf=ufy-vn*h*guy; Tf=Tb0-vn*h*gT; Tf=Tf<1.0f-3 ? 1.0f-3 : Tf
        Qx=cax*cxk-(ax==1 ? cs2 : 0.0f0); Qy=cax*cyk-(ax==2 ? cs2 : 0.0f0)
        fb=_feqT2_gpu(ρf,uxf,uyf,Tf,ufx,ufy,cxk,cyk,czk,wk,cs2,1.0f0,1.0f0) - coef*wk*ρf*(Qx*gux+Qy*guy)
        rb+=fb; jx+=vx*fb; jy+=vy*fb; en+=(0.5f0*(vx*vx+vy*vy+vz*vz)+erc*Tf)*fb
    end
    rb=rb<1.0f-4 ? 1.0f-4 : rb; uxb=jx/rb; uyb=jy/rb
    Tb=(en/rb-0.5f0*(uxb*uxb+uyb*uyb))/eint; Tb=Tb<1.0f-3 ? 1.0f-3 : Tb
    # pass 2: recover the interface distribution f = w1·f̄ + w2·f_eq(interface) and accumulate the flux
    Frho=0.0f0; Fmx=0.0f0; Fmy=0.0f0; FEt=0.0f0
    @inbounds for k in 1:39
        cxk=cx_arr[k]; cyk=cy_arr[k]; czk=cz_arr[k]; wk=w_arr[k]
        vx=cxk+ufx; vy=cyk+ufy; vz=czk; vn = ax==1 ? vx : vy; cax = ax==1 ? cxk : cyk
        ρf=rb0-vn*h*gρ; ρf=ρf<1.0f-4 ? 1.0f-4 : ρf
        uxf=ufx-vn*h*gux; uyf=ufy-vn*h*guy; Tf=Tb0-vn*h*gT; Tf=Tf<1.0f-3 ? 1.0f-3 : Tf
        Qx=cax*cxk-(ax==1 ? cs2 : 0.0f0); Qy=cax*cyk-(ax==2 ? cs2 : 0.0f0)
        fbar=_feqT2_gpu(ρf,uxf,uyf,Tf,ufx,ufy,cxk,cyk,czk,wk,cs2,1.0f0,1.0f0) - coef*wk*ρf*(Qx*gux+Qy*guy)
        feqb=_feqT2_gpu(rb,uxb,uyb,Tb,ufx,ufy,cxk,cyk,czk,wk,cs2,1.0f0,1.0f0)
        fk=w1*fbar+w2*feqb
        Frho+=vn*fk; Fmx+=vn*vx*fk; Fmy+=vn*vy*fk; FEt+=vn*0.5f0*(vx*vx+vy*vy+vz*vz)*fk
    end
    return Frho, Fmx, Fmy, FEt+erc*Tb*Frho     # energy flux = translational 2nd moment + advected rotational erc·T
end

# G1 blend: DUGKS (low-dissipation gas-kinetic, physical ν from τ) where smooth; robust KFVS (_fvflux2T,
# WENO5) across shocks via a pressure-jump sensor. dugks==0 ⇒ pure KFVS (byte-identical to the legacy flux).
@inline function _flux_blend2T(rhoL,uxL,uyL,TL, rhoR,uxR,uyR,TR, rhoLL,uxLL,uyLL,TLL, rhoRR,uxRR,uyRR,TRR,
                               rhoLLL,uxLLL,uyLLL,TLLL, rhoRRR,uxRRR,uyRRR,TRRR,
                               ax, cx,cy,cz,w, cs2, drot, csmag, gauge, shock_aware, drp, weno, nu_coeff, dugks, h, tau)
    if dugks == Int32(0)
        return _fvflux2T(rhoL,uxL,uyL,TL, rhoR,uxR,uyR,TR, rhoLL,uxLL,uyLL,TLL, rhoRR,uxRR,uyRR,TRR,
                         rhoLLL,uxLLL,uyLLL,TLLL, rhoRRR,uxRRR,uyRRR,TRRR, ax,cx,cy,cz,w,cs2,drot,csmag,gauge,shock_aware,drp,weno,nu_coeff)
    end
    pL=rhoL*cs2*TL; pR=rhoR*cs2*TR; jump=abs(pR-pL)/min(pL,pR)                 # PRESSURE-jump ⇒ true shocks
    sj = jump<=0.1f0 ? 0.0f0 : (jump>=0.5f0 ? 1.0f0 : (jump-0.1f0)*(1.0f0/0.4f0))
    # grid-scale ripple sensor (2nd/1st undivided difference): →KFVS on the odd-even wiggles the very-low-ν
    # DUGKS (w2≈1 at high Re) would otherwise sustain. Capped 0.5 so smooth regions stay DUGKS (low dissipation).
    scu=_sfsen(uxLL,uxL,uxR,uxRR); scv=_sfsen(uyLL,uyL,uyR,uyRR); sc = scu>scv ? scu : scv
    s_rip = sc<=0.5f0 ? 0.0f0 : (sc>=1.0f0 ? 0.5f0 : (sc-0.5f0))
    s = sj>s_rip ? sj : s_rip
    Fd1,Fd2,Fd3,Fd4 = _dugks_flux2T(rhoL,uxL,uyL,TL, rhoR,uxR,uyR,TR, rhoLL,uxLL,uyLL,TLL, rhoRR,uxRR,uyRR,TRR, ax,cx,cy,cz,w,cs2,drot,h,tau,nu_coeff)
    s == 0.0f0 && return Fd1,Fd2,Fd3,Fd4
    Fk1,Fk2,Fk3,Fk4 = _fvflux2T(rhoL,uxL,uyL,TL, rhoR,uxR,uyR,TR, rhoLL,uxLL,uyLL,TLL, rhoRR,uxRR,uyRR,TRR,
                                rhoLLL,uxLLL,uyLLL,TLLL, rhoRRR,uxRRR,uyRRR,TRRR, ax,cx,cy,cz,w,cs2,drot,csmag,gauge,shock_aware,drp,weno,nu_coeff)
    om=1.0f0-s
    return om*Fd1+s*Fk1, om*Fd2+s*Fk2, om*Fd3+s*Fk3, om*Fd4+s*Fk4
end

# rectangular thermal kernel: 4 conserved (ρ,ρu,ρv,ρE); recovers primitives (ρ,ux,uy,T).
@kernel function pond_fv2dT_rect_kernel!(rho_o, ux_o, uy_o, T_o, @Const(rho), @Const(ux), @Const(uy), @Const(Temp),
                                         @Const(chi), @Const(cx_arr), @Const(cy_arr), @Const(cz_arr), @Const(w_arr),
                                         cs2::Float32, a::Float32, drot::Float32, csmag::Float32, gauge::Int32, wall::Int32, shock_aware::Int32, drp::Int32, @Const(cd_arr), dyn::Int32, weno::Int32, @Const(sdf), cutcell::Int32, @Const(lam), @Const(aE), @Const(aW), @Const(aN), @Const(aS), dc_r, dc_mx, dc_my, dc_e, frd::Int32, dugks::Int32, dtau::Float32, hflux::Float32, flux_sgs::Int32, split::Int32, NX::Int32, NY::Int32)
    i, j = @index(Global, NTuple)
    nx = Int(NX); ny = Int(NY)
    eint = 0.5f0*(3.0f0+drot)*cs2       # internal-energy coeff = (D_eff/2)·R, R=cs²; D_eff=3+drot
    # PHASE 0 step 3 (fixed-h DUGKS, panel BLOCKER): the half-step h is a PARAMETER, decoupled from the
    # update dt. Explicit path passes h_nom=0.5·dt/dx (bitwise ≡ the old in-kernel 0.5f0*a — power-of-two
    # scaling is exact); the steady tier freezes h_nom per level so the DUGKS flux keeps its discrete
    # solution while pseudo-dt varies. dtau = τ = ν/cs² (nucal solves with the SAME h_nom).
    dhalf = hflux
    @inbounds begin
        iL=_clampi(i-1,nx); iR=_clampi(i+1,nx); iRR=_clampi(i+2,nx); iLL=_clampi(i-2,nx)
        jD=_clampi(j-1,ny); jU=_clampi(j+1,ny); jUU=_clampi(j+2,ny); jDD=_clampi(j-2,ny)
        iLLL=_clampi(i-3,nx); iRRR=_clampi(i+3,nx); jDDD=_clampi(j-3,ny); jUUU=_clampi(j+3,ny)   # ±3 stencil for WENO5
        # per-face SGS coefficient: static Cs² (dyn=0) or the face-average of the Germano-Lilly c_d field (dyn=1).
        # flux_sgs=0 (DDES-solo, opt-in): zero the FLUX-level SGS (KFVS face term + DUGKS τ_eff Smagorinsky) so
        # the SA-DDES tensor stress is the SOLE subgrid closure (canonical DDES — no double-counted dissipation);
        # csmag>0 still selects the low-dissipation LES reconstruction branch inside _fvflux2T.
        sgsf = flux_sgs==Int32(1) ? 1.0f0 : 0.0f0
        csq=csmag*csmag; cc=(dyn==Int32(1)) ? cd_arr[i,j] : csq
        nuR=sgsf*((dyn==Int32(1)) ? 0.5f0*(cc+cd_arr[iR,j]) : csq)
        nuL=sgsf*((dyn==Int32(1)) ? 0.5f0*(cc+cd_arr[iL,j]) : csq)
        nuT=sgsf*((dyn==Int32(1)) ? 0.5f0*(cc+cd_arr[i,jU]) : csq)
        nuB=sgsf*((dyn==Int32(1)) ? 0.5f0*(cc+cd_arr[i,jD]) : csq)
        FrR,FmxR,FmyR,FER = _flux_blend2T(rho[i,j],ux[i,j],uy[i,j],Temp[i,j], rho[iR,j],ux[iR,j],uy[iR,j],Temp[iR,j],
            rho[iL,j],ux[iL,j],uy[iL,j],Temp[iL,j], rho[iRR,j],ux[iRR,j],uy[iRR,j],Temp[iRR,j],
            rho[iLL,j],ux[iLL,j],uy[iLL,j],Temp[iLL,j], rho[iRRR,j],ux[iRRR,j],uy[iRRR,j],Temp[iRRR,j], 1, cx_arr,cy_arr,cz_arr,w_arr,cs2,drot,csmag,gauge,shock_aware,drp,weno, nuR, dugks, dhalf, dtau)
        FrL,FmxL,FmyL,FEL = _flux_blend2T(rho[iL,j],ux[iL,j],uy[iL,j],Temp[iL,j], rho[i,j],ux[i,j],uy[i,j],Temp[i,j],
            rho[iLL,j],ux[iLL,j],uy[iLL,j],Temp[iLL,j], rho[iR,j],ux[iR,j],uy[iR,j],Temp[iR,j],
            rho[iLLL,j],ux[iLLL,j],uy[iLLL,j],Temp[iLLL,j], rho[iRR,j],ux[iRR,j],uy[iRR,j],Temp[iRR,j], 1, cx_arr,cy_arr,cz_arr,w_arr,cs2,drot,csmag,gauge,shock_aware,drp,weno, nuL, dugks, dhalf, dtau)
        FrT,FmxT,FmyT,FET = _flux_blend2T(rho[i,j],ux[i,j],uy[i,j],Temp[i,j], rho[i,jU],ux[i,jU],uy[i,jU],Temp[i,jU],
            rho[i,jD],ux[i,jD],uy[i,jD],Temp[i,jD], rho[i,jUU],ux[i,jUU],uy[i,jUU],Temp[i,jUU],
            rho[i,jDD],ux[i,jDD],uy[i,jDD],Temp[i,jDD], rho[i,jUUU],ux[i,jUUU],uy[i,jUUU],Temp[i,jUUU], 2, cx_arr,cy_arr,cz_arr,w_arr,cs2,drot,csmag,gauge,shock_aware,drp,weno, nuT, dugks, dhalf, dtau)
        FrB,FmxB,FmyB,FEB = _flux_blend2T(rho[i,jD],ux[i,jD],uy[i,jD],Temp[i,jD], rho[i,j],ux[i,j],uy[i,j],Temp[i,j],
            rho[i,jDD],ux[i,jDD],uy[i,jDD],Temp[i,jDD], rho[i,jU],ux[i,jU],uy[i,jU],Temp[i,jU],
            rho[i,jDDD],ux[i,jDDD],uy[i,jDDD],Temp[i,jDDD], rho[i,jUU],ux[i,jUU],uy[i,jUU],Temp[i,jUU], 2, cx_arr,cy_arr,cz_arr,w_arr,cs2,drot,csmag,gauge,shock_aware,drp,weno, nuB, dugks, dhalf, dtau)
        # ADIABATIC KINETIC WALL (consistent reflective flux): a fluid-cell face bordering the solid
        # (χ>0.5) is a wall. Replace that face's flux by the CONSISTENT inviscid-wall value — zero mass,
        # zero energy (∂T/∂n=0, adiabatic), momentum = wall pressure p_w=ρRT only. This is thermodynamically
        # consistent (mass and its enthalpy are zeroed together, unlike masking energy alone), supplies the
        # correct surface pressure force, and removes the enthalpy pile-up behind the body that acts as a hot
        # reservoir and over-heats the stagnation region. Solid cells are untouched (handled by penalization).
        if cutcell != Int32(1) && wall == 1 && chi[i,j] <= 0.5f0     # legacy staircase kinetic wall (wall==1 mode)
            pw = rho[i,j]*cs2*Temp[i,j]         # fluid-side wall pressure (R=cs²)
            if chi[iR,j]>0.5f0; FrR=0.0f0; FmxR=pw; FmyR=0.0f0; FER=0.0f0; end
            if chi[iL,j]>0.5f0; FrL=0.0f0; FmxL=pw; FmyL=0.0f0; FEL=0.0f0; end
            if chi[i,jU]>0.5f0; FrT=0.0f0; FmyT=pw; FmxT=0.0f0; FET=0.0f0; end
            if chi[i,jD]>0.5f0; FrB=0.0f0; FmyB=pw; FmxB=0.0f0; FEB=0.0f0; end
        end
        u0x=ux[i,j]; u0y=uy[i,j]; r0=rho[i,j]
        Ec = r0*(eint*Temp[i,j] + 0.5f0*(u0x*u0x+u0y*u0y))            # ρE = ρ((D_eff/2)R T + ½|u|²)
        r=r0; mx=r0*u0x; my=r0*u0y; e=Ec
        if cutcell == Int32(1)
            # CONSERVATIVE CUT-CELL EMBEDDED-BOUNDARY UPDATE: Λ·dU/dt = −[(αE·FE−αW·FW)+(αN·FN−αS·FS)+F_wall].
            # Faces weighted by their fluid APERTURE; the impermeable wall carries {0 mass, 0 energy (adiabatic),
            # momentum = p_w·(A·n̂)} with A·n̂ = (−(αE−αW), −(αN−αS)) from the discrete divergence identity (∮n̂ dl=0
            # ⇒ a uniform pressure field yields ZERO net force, exactly). Dividing by Λ makes it strictly conservative.
            Λc=lam[i,j]
            if Λc > 1f-6
                αe=aE[i,j];αw=aW[i,j];αn=aN[i,j];αs=aS[i,j]; pw=r0*cs2*Temp[i,j]
                divr =(αe*FrR  - αw*FrL)  + (αn*FrT  - αs*FrB)       # aperture-weighted flux divergence (extensive/dx)
                divmx=(αe*FmxR - αw*FmxL) + (αn*FmxT - αs*FmxB) + pw*(-(αe-αw))
                divmy=(αe*FmyR - αw*FmyL) + (αn*FmyT - αs*FmyB) + pw*(-(αn-αs))
                dive =(αe*FER  - αw*FEL)  + (αn*FET  - αs*FEB)
                if frd == Int32(1)
                    invΛ=1.0f0/Λc                                    # store conservative divergence Dc=divF/Λ; the hybrid pass does the update
                    dc_r[i,j]=divr*invΛ; dc_mx[i,j]=divmx*invΛ; dc_my[i,j]=divmy*invΛ; dc_e[i,j]=dive*invΛ
                    # passthrough (r,mx,my,e already = U^n) — _frd_hybrid2T! overwrites with U^n − a·D_hybrid
                else
                    aΛ = a/max(Λc, 0.05f0)   # legacy Λ-floor scaffold (small-cell crash guard; superseded by frd)
                    r=r0-aΛ*divr; mx=r0*u0x-aΛ*divmx; my=r0*u0y-aΛ*divmy; e=Ec-aΛ*dive
                end
            end                            # Λ≈0 (solid): leave (r0,…) — ghost fill + redistribution handle it
        else
            # PHASE 0 (implicit roadmap, 2026-07-10): the rectangular flux divergence, named. This IS the
            # 4-component RATE R=divF (dx=1; dt factored out) of the flux part of the operator — the same
            # role Dc plays on the cut-cell path above.
            divr  = (FrR-FrL)  + (FrT-FrB)
            divmx = (FmxR-FmxL)+ (FmxT-FmxB)
            divmy = (FmyR-FmyL)+ (FmyT-FmyB)
            dive  = (FER-FEL)  + (FET-FEB)
            if split == Int32(1)
                # residual!/update! split: store R; pond_fv2dT_rect_update! applies U^{n+1}=U^n−a·R.
                # Bitwise-identical to the fused path below (an FP32 store/load round-trip is lossless
                # and the update kernel repeats the exact same expressions in the same order).
                dc_r[i,j]=divr; dc_mx[i,j]=divmx; dc_my[i,j]=divmy; dc_e[i,j]=dive
                # passthrough (r,mx,my,e stay U^n) — the update kernel overwrites every cell
            else
                r  = r0          - a*divr
                mx = r0*u0x      - a*divmx
                my = r0*u0y      - a*divmy
                e  = Ec          - a*dive
            end
        end
        r = r < 1.0f-3 ? 1.0f-3 : r
        ux2=mx/r; uy2=my/r; sp=sqrt(ux2*ux2+uy2*uy2)
        if sp>3.5f0; f=3.5f0/sp; ux2*=f; uy2*=f; end
        T2 = (e/r - 0.5f0*(ux2*ux2+uy2*uy2))/eint; T2 = T2 < 1.0f-3 ? 1.0f-3 : T2   # T = (ρE/ρ−½|u|²)/((D_eff/2)R)
        rho_o[i,j]=r; ux_o[i,j]=ux2; uy_o[i,j]=uy2; T_o[i,j]=T2
    end
end

# PHASE 0 (implicit roadmap step 1, 2026-07-10) — the trivial explicit update for the RECTANGULAR path:
# U^{n+1} = U^n − a·R, consuming the rate R stored by pond_fv2dT_rect_kernel! (split=1) in dc_*.
# Bitwise-identical to the kernel's fused else-branch: same conserved build, same op order, same
# floors/speed-cap, and the dc_* round-trip through global memory is an exact FP32 store/load.
# The cut-cell path needs NO counterpart — with frd=1 the flux kernel already stores Dc=divF/Λ and
# _frd_merge2T!/_frd_hybrid2T! IS the update. For MERGED sliver groups the steady-tier group rate is
# R_g = ΣΛ·Dc/ΣΛ (identical for all members); on the merged subspace (members equal — which the merge
# write guarantees at every step end) the transient Λ-averaging projection in _frd_merge2T! is the
# identity, so R_g=0 ⟺ the group is steady. See docs/2026-07-10_PHASE0_residual_split.md.
@kernel function pond_fv2dT_rect_update!(rho_o, ux_o, uy_o, T_o, @Const(rho), @Const(ux), @Const(uy), @Const(Temp),
                                         @Const(dc_r), @Const(dc_mx), @Const(dc_my), @Const(dc_e),
                                         cs2::Float32, a::Float32, drot::Float32)
    i,j=@index(Global,NTuple); eint=0.5f0*(3.0f0+drot)*cs2
    @inbounds begin
        r0=rho[i,j]; u0x=ux[i,j]; u0y=uy[i,j]
        Ec = r0*(eint*Temp[i,j] + 0.5f0*(u0x*u0x+u0y*u0y))
        r  = r0          - a*dc_r[i,j]
        mx = r0*u0x      - a*dc_mx[i,j]
        my = r0*u0y      - a*dc_my[i,j]
        e  = Ec          - a*dc_e[i,j]
        r = r < 1.0f-3 ? 1.0f-3 : r
        ux2=mx/r; uy2=my/r; sp=sqrt(ux2*ux2+uy2*uy2)
        if sp>3.5f0; f=3.5f0/sp; ux2*=f; uy2*=f; end
        T2 = (e/r - 0.5f0*(ux2*ux2+uy2*uy2))/eint; T2 = T2 < 1.0f-3 ? 1.0f-3 : T2
        rho_o[i,j]=r; ux_o[i,j]=ux2; uy_o[i,j]=uy2; T_o[i,j]=T2
    end
end

# V1 FLUX REDISTRIBUTION — hybrid-divergence pass (Berger/Colella-style, AMReX EB). For each fluid cell (Λ>0),
# form the volume-weighted neighbourhood-average divergence Dnc, blend D = Λ·Dc + (1−Λ)·Dnc (full cells Λ=1 ⇒
# D=Dc exactly, so they are unchanged), and apply the update U^{n+1}=U^n − a·D. Stable for small Λ
# because the (1−Λ) share uses the SMOOTH neighbourhood divergence instead of dividing the flux by a tiny volume
# — this removes the Λ-floor mass pile-up (the LE ρ-spike). Reads OLD prims (U^n) + the stored Dc; writes prims.
# HONESTY NOTE (audit 2026-07-07): the hybrid divergence alone is NOT strictly conservative — each cut cell
# leaks δM = a·Λ(1−Λ)·(Dc−Dnc) per step. Full Berger/Colella/AMReX-EB conservation requires REDISTRIBUTING that
# defect to the Λ-weighted neighbours (the Colella-Graves step = roadmap G6, not yet implemented). V1 is a
# small-cell STABILIZATION, adequate near steady state where Dc≈Dnc.
@kernel function _frd_hybrid2T!(rho_o, ux_o, uy_o, T_o, @Const(rho), @Const(ux), @Const(uy), @Const(Temp),
                                @Const(lam), @Const(dc_r), @Const(dc_mx), @Const(dc_my), @Const(dc_e),
                                cs2::Float32, a::Float32, drot::Float32, NX::Int32, NY::Int32)
    i,j=@index(Global,NTuple); nx=Int(NX); ny=Int(NY); eint=0.5f0*(3.0f0+drot)*cs2
    @inbounds begin
        Λc=lam[i,j]
        if Λc > 1f-6
            iL=_clampi(i-1,nx); iR=_clampi(i+1,nx); jD=_clampi(j-1,ny); jU=_clampi(j+1,ny)
            sw=Λc; sr=Λc*dc_r[i,j]; smx=Λc*dc_mx[i,j]; smy=Λc*dc_my[i,j]; se=Λc*dc_e[i,j]   # self + fluid neighbours
            Ln=lam[iL,j]; if Ln>1f-6; sw+=Ln; sr+=Ln*dc_r[iL,j]; smx+=Ln*dc_mx[iL,j]; smy+=Ln*dc_my[iL,j]; se+=Ln*dc_e[iL,j]; end
            Ln=lam[iR,j]; if Ln>1f-6; sw+=Ln; sr+=Ln*dc_r[iR,j]; smx+=Ln*dc_mx[iR,j]; smy+=Ln*dc_my[iR,j]; se+=Ln*dc_e[iR,j]; end
            Ln=lam[i,jD]; if Ln>1f-6; sw+=Ln; sr+=Ln*dc_r[i,jD]; smx+=Ln*dc_mx[i,jD]; smy+=Ln*dc_my[i,jD]; se+=Ln*dc_e[i,jD]; end
            Ln=lam[i,jU]; if Ln>1f-6; sw+=Ln; sr+=Ln*dc_r[i,jU]; smx+=Ln*dc_mx[i,jU]; smy+=Ln*dc_my[i,jU]; se+=Ln*dc_e[i,jU]; end
            invsw=1.0f0/sw; om=1.0f0-Λc
            Dr =Λc*dc_r[i,j] +om*sr*invsw;  Dmx=Λc*dc_mx[i,j]+om*smx*invsw
            Dmy=Λc*dc_my[i,j]+om*smy*invsw; De =Λc*dc_e[i,j] +om*se*invsw
            r0=rho[i,j]; u0x=ux[i,j]; u0y=uy[i,j]; Ec=r0*(eint*Temp[i,j]+0.5f0*(u0x*u0x+u0y*u0y))
            r=r0-a*Dr; mx=r0*u0x-a*Dmx; my=r0*u0y-a*Dmy; e=Ec-a*De
            r = r<1.0f-3 ? 1.0f-3 : r
            ux2=mx/r; uy2=my/r; sp=sqrt(ux2*ux2+uy2*uy2); if sp>3.5f0; f=3.5f0/sp; ux2*=f; uy2*=f; end
            T2=(e/r-0.5f0*(ux2*ux2+uy2*uy2))/eint; T2=T2<1.0f-3 ? 1.0f-3 : T2
            rho_o[i,j]=r; ux_o[i,j]=ux2; uy_o[i,j]=uy2; T_o[i,j]=T2
        end     # Λ≈0 solid: leave rho_o = passthrough U^n (from the rect kernel)
    end
end

# V2 COLELLA-GRAVES STATE REDISTRIBUTION (G6, 2026-07-07). The V1 hybrid divergence above is stable but NOT
# conservative: each cut cell leaks δM = a·Λ(1−Λ)·(Dc−Dnc) per step (watched live as the monotonic LE ρ-spike
# ρ_max 2.5→3.3 on the RAE2822 ghost run). The Berger/Colella/AMReX-EB cure: REDISTRIBUTE that defect to the
# cut cell's Λ-weighted neighbourhood — total Λ-weighted conserved change then telescopes to round-off.
# Pass 1 (_frd_defect2T!): per CUT cell, recompute Dnc (same 5-cell gather as the hybrid pass) and store
#   δM_c = a·Λc(1−Λc)(Dc−Dnc) per conserved variable (zero for full/solid cells).
# Pass 2 (_frd_cg2T!): GATHER form (race-free): each fluid cell i collects, from every donor c in its 5-cell
#   neighbourhood with δM_c≠0, the density increment δM_c/W_c where W_c = ΣΛ over the DONOR's neighbourhood
#   (recomputed per donor, identical for all its receivers ⇒ Σ_i Λ_i·δM_c/W_c = δM_c exactly — conservative
#   regardless of edge clamping). Update is applied to the POST-hybrid conserved state, in place (each thread
#   writes only its own cell; reads only dm_*/lam + its own prims).
@kernel function _frd_defect2T!(dm_r, dm_mx, dm_my, dm_e, @Const(lam), @Const(dc_r), @Const(dc_mx), @Const(dc_my), @Const(dc_e),
                                a::Float32, NX::Int32, NY::Int32)
    i,j=@index(Global,NTuple); nx=Int(NX); ny=Int(NY)
    @inbounds begin
        Λc=lam[i,j]
        if Λc>1f-6 && Λc<1f0-1f-6      # CUT cells only (full cells: D_hyb=Dc ⇒ zero defect)
            iL=_clampi(i-1,nx); iR=_clampi(i+1,nx); jD=_clampi(j-1,ny); jU=_clampi(j+1,ny)
            sw=Λc; sr=Λc*dc_r[i,j]; smx=Λc*dc_mx[i,j]; smy=Λc*dc_my[i,j]; se=Λc*dc_e[i,j]
            Ln=lam[iL,j]; if Ln>1f-6; sw+=Ln; sr+=Ln*dc_r[iL,j]; smx+=Ln*dc_mx[iL,j]; smy+=Ln*dc_my[iL,j]; se+=Ln*dc_e[iL,j]; end
            Ln=lam[iR,j]; if Ln>1f-6; sw+=Ln; sr+=Ln*dc_r[iR,j]; smx+=Ln*dc_mx[iR,j]; smy+=Ln*dc_my[iR,j]; se+=Ln*dc_e[iR,j]; end
            Ln=lam[i,jD]; if Ln>1f-6; sw+=Ln; sr+=Ln*dc_r[i,jD]; smx+=Ln*dc_mx[i,jD]; smy+=Ln*dc_my[i,jD]; se+=Ln*dc_e[i,jD]; end
            Ln=lam[i,jU]; if Ln>1f-6; sw+=Ln; sr+=Ln*dc_r[i,jU]; smx+=Ln*dc_mx[i,jU]; smy+=Ln*dc_my[i,jU]; se+=Ln*dc_e[i,jU]; end
            invsw=1.0f0/sw
            # δM = −a·Λ(1−Λ)(Dc−Dnc): the removal the hybrid pass FAILED to apply (sign: hybrid under-removes
            # when Dc>Dnc — the LE pile-up — so the redistributed increment must be NEGATIVE there).
            cf=-a*Λc*(1.0f0-Λc)
            dm_r[i,j] =cf*(dc_r[i,j] -sr*invsw);  dm_mx[i,j]=cf*(dc_mx[i,j]-smx*invsw)
            dm_my[i,j]=cf*(dc_my[i,j]-smy*invsw); dm_e[i,j] =cf*(dc_e[i,j] -se*invsw)
        else
            dm_r[i,j]=0f0; dm_mx[i,j]=0f0; dm_my[i,j]=0f0; dm_e[i,j]=0f0
        end
    end
end
@kernel function _frd_cg2T!(rho, ux, uy, T, @Const(lam), @Const(dm_r), @Const(dm_mx), @Const(dm_my), @Const(dm_e),
                            cs2::Float32, drot::Float32, NX::Int32, NY::Int32)
    i,j=@index(Global,NTuple); nx=Int(NX); ny=Int(NY); eint=0.5f0*(3.0f0+drot)*cs2
    @inbounds begin
        Λi=lam[i,j]
        if Λi > 1f-6
            ar=0f0; amx=0f0; amy=0f0; ae=0f0
            # donors: self + 4 cardinal neighbours; each donor's W recomputed over ITS OWN neighbourhood
            for (di,dj) in ((0,0),(-1,0),(1,0),(0,-1),(0,1))
                c1=_clampi(i+di,nx); c2=_clampi(j+dj,ny)
                δr=dm_r[c1,c2]
                if δr!=0f0 || dm_mx[c1,c2]!=0f0 || dm_my[c1,c2]!=0f0 || dm_e[c1,c2]!=0f0
                    dL=_clampi(c1-1,nx); dR=_clampi(c1+1,nx); dD=_clampi(c2-1,ny); dU=_clampi(c2+1,ny)
                    W=lam[c1,c2]+lam[dL,c2]+lam[dR,c2]+lam[c1,dD]+lam[c1,dU]
                    invW=1.0f0/max(W,1f-6)
                    ar+=δr*invW; amx+=dm_mx[c1,c2]*invW; amy+=dm_my[c1,c2]*invW; ae+=dm_e[c1,c2]*invW
                end
            end
            if ar!=0f0 || amx!=0f0 || amy!=0f0 || ae!=0f0
                r0=rho[i,j]; u0=ux[i,j]; v0=uy[i,j]
                Ec=r0*(eint*T[i,j]+0.5f0*(u0*u0+v0*v0))
                r=r0+ar; r = r<1.0f-3 ? 1.0f-3 : r
                mx=r0*u0+amx; my=r0*v0+amy; e=Ec+ae
                u2=mx/r; v2=my/r; sp=sqrt(u2*u2+v2*v2); if sp>3.5f0; f=3.5f0/sp; u2*=f; v2*=f; end
                T2=(e/r-0.5f0*(u2*u2+v2*v2))/eint; T2 = T2<1.0f-3 ? 1.0f-3 : T2
                rho[i,j]=r; ux[i,j]=u2; uy[i,j]=v2; T[i,j]=T2
            end
        end
    end
end

# V3 CUT-CELL SMALL-CELL MERGING (Berger-Helzel cell linking, G6 — 2026-07-07). The CG A/B proved the
# sliver defect is PERSISTENT-SIGNED (inconsistent local flux balance at stagnation slivers), so neither
# the Λ-floor, the hybrid, nor redistribution can be both stable AND conservative. MERGING is: every
# sliver (Λ<ΛMIN) is linked to its largest-Λ neighbour (host-precomputed, chains resolved, static
# geometry); each group is updated as ONE control volume with the strictly conservative aperture-weighted
# divergence — U_m = [ΣΛ_g·U_g − a·ΣΛ_g·Dc_g]/ΣΛ_g (Λ·Dc = divF, extensive) — and the merged state is
# written to every member. Unmerged cells get the plain conservative U − a·Dc (stable: Λ ≥ ΛMIN by
# construction). Strictly conservative, locally consistent, no tuning. Replaces hybrid+CG when active.
function pond_cutcell_merge_map(Λh::Matrix{Float32}, aEh::Matrix{Float32}, aWh::Matrix{Float32},
                                aNh::Matrix{Float32}, aSh::Matrix{Float32};
                                lam_min::Float32=0.6f0, maxlinks::Int=8, amin::Float32=0.1f0)
    NX,NY=size(Λh); lin(i,j)=Int32(i+(j-1)*NX)
    tgt=zeros(Int32,NX,NY); norph=0
    for j in 1:NY, i in 1:NX
        (1f-6<Λh[i,j] && Λh[i,j]<lam_min) || continue
        ci,cj=i,j; ok=false
        for _ in 1:6            # walk uphill in Λ through OPEN faces only (aperture>amin — at a sharp TE
                                # the max-Λ neighbour may be across the solid wedge; linking through it
                                # teleports mass through the wall ⇒ the step-1200 TE blowup)
            bi,bj,bl=ci,cj,Λh[ci,cj]
            for (di,dj,ap) in ((1,0,aEh[ci,cj]),(-1,0,aWh[ci,cj]),(0,1,aNh[ci,cj]),(0,-1,aSh[ci,cj]))
                ap>amin || continue
                ni=clamp(ci+di,1,NX); nj=clamp(cj+dj,1,NY)
                Λh[ni,nj]>bl && ((bi,bj,bl)=(ni,nj,Λh[ni,nj]))
            end
            (bi,bj)==(ci,cj) && break
            ci,cj=bi,bj
            if Λh[ci,cj]>=lam_min; ok=true; break; end
        end
        if ok && (ci,cj)!=(i,j); tgt[i,j]=lin(ci,cj); else; norph+=1; end
    end
    links=zeros(Int32,NX,NY,maxlinks); nlk=zeros(Int32,NX,NY); nover=0
    for j in 1:NY, i in 1:NX
        t=tgt[i,j]; t==Int32(0) && continue
        ti=Int(t); tx=((ti-1)%NX)+1; ty=((ti-1)÷NX)+1
        c=Int(nlk[tx,ty])+1
        if c<=maxlinks; links[tx,ty,c]=lin(i,j); nlk[tx,ty]=Int32(c)
        else; tgt[i,j]=Int32(0); nover+=1; norph+=1; end
    end
    nm=count(!=(Int32(0)),tgt)
    return tgt,links,nlk,nm,nover,norph
end
@kernel function _frd_merge2T!(rho_o,ux_o,uy_o,T_o, @Const(rho),@Const(ux),@Const(uy),@Const(Temp),
        @Const(lam), @Const(dc_r),@Const(dc_mx),@Const(dc_my),@Const(dc_e),
        @Const(tgt), @Const(links), @Const(nlk),
        cs2::Float32, a::Float32, drot::Float32, NX::Int32, NY::Int32)
    i,j=@index(Global,NTuple); nx=Int(NX); eint=0.5f0*(3.0f0+drot)*cs2
    @inbounds begin
        Λc=lam[i,j]
        if Λc>1f-6 && tgt[i,j]==Int32(0)          # unmerged cell or a merge TARGET (slivers written by their target)
            r0=rho[i,j]; u0=ux[i,j]; v0=uy[i,j]
            Ec=r0*(eint*Temp[i,j]+0.5f0*(u0*u0+v0*v0))
            # ORPHAN sliver (walk failed at init — isolated TE-wedge cells): the raw conservative 1/Λ update
            # would run at Λ_c⁻¹× CFL (instant blowup). Bounded legacy Λ-floor instead: scale the divergence
            # by min(1, Λ/0.5) — locally non-conservative for the FEW reported orphans, globally negligible.
            sc = Λc<0.6f0 ? min(1.0f0, 2.0f0*Λc) : 1.0f0
            sw=Λc
            sr =Λc*(r0    - a*sc*dc_r[i,j]);  smx=Λc*(r0*u0 - a*sc*dc_mx[i,j])
            smy=Λc*(r0*v0 - a*sc*dc_my[i,j]); se =Λc*(Ec    - a*sc*dc_e[i,j])
            nl=Int(nlk[i,j])
            for k in 1:nl
                li=Int(links[i,j,k]); ii=((li-1)%nx)+1; jj=((li-1)÷nx)+1
                Λs=lam[ii,jj]; rs=rho[ii,jj]; us=ux[ii,jj]; vs=uy[ii,jj]
                Es=rs*(eint*Temp[ii,jj]+0.5f0*(us*us+vs*vs))
                sw+=Λs
                sr +=Λs*(rs    - a*dc_r[ii,jj]);  smx+=Λs*(rs*us - a*dc_mx[ii,jj])
                smy+=Λs*(rs*vs - a*dc_my[ii,jj]); se +=Λs*(Es    - a*dc_e[ii,jj])
            end
            invw=1.0f0/sw
            r=sr*invw; r = r<1.0f-3 ? 1.0f-3 : r
            u2=smx*invw/r; v2=smy*invw/r; sp=sqrt(u2*u2+v2*v2); if sp>3.5f0; f=3.5f0/sp; u2*=f; v2*=f; end
            T2=(se*invw/r-0.5f0*(u2*u2+v2*v2))/eint; T2 = T2<1.0f-3 ? 1.0f-3 : T2
            rho_o[i,j]=r; ux_o[i,j]=u2; uy_o[i,j]=v2; T_o[i,j]=T2
            for k in 1:nl                          # unique writer: each sliver has exactly one target
                li=Int(links[i,j,k]); ii=((li-1)%nx)+1; jj=((li-1)÷nx)+1
                rho_o[ii,jj]=r; ux_o[ii,jj]=u2; uy_o[ii,jj]=v2; T_o[ii,jj]=T2
            end
        end
    end
end

# SPALART–ALLMARAS one-equation RANS turbulence model (Spalart-Allmaras 1992, standard form). Transports ν̃:
#   Dν̃/Dt = cb1·S̃·ν̃ − cw1·fw·(ν̃/d)² + (1/σ)[∇·((ν+ν̃)∇ν̃) + cb2|∇ν̃|²]
# with the eddy viscosity ν_t = ν̃·f_v1. Wall distance d = |SDF|. Explicit update; ν̃≥0; ν̃=0 in the solid (wall
# BC). Standard for external aero (it's what keeps the BL ATTACHED — the missing piece the M0.3/M0.73 laminar
# separation exposed). Writes ν_t·dt into nuw (=the `wm` field) ⇒ applied as ∇²(ρu) diffusion in _svvT_kernel!.
@kernel function _sa_kernel!(nut_o, nuw_o, @Const(nut), @Const(ux), @Const(uy), @Const(wdist), @Const(chi),
                             nu_lat::Float32, dt::Float32, ddes::Int32, cdes::Float32, NX::Int32, NY::Int32)
    i,j=@index(Global,NTuple); nx=Int(NX); ny=Int(NY)
    cb1=0.1355f0; σ=2.0f0/3.0f0; cb2=0.622f0; κ=0.41f0; cv1=7.1f0; cw2=0.3f0; cw3=2.0f0
    cw1=cb1/(κ*κ)+(1.0f0+cb2)/σ; cv13=cv1*cv1*cv1; cw36=cw3^6
    @inbounds begin
        if chi[i,j] > 0.5f0
            nut_o[i,j]=0.0f0; nuw_o[i,j]=0.0f0
        else
            iL=_clampi(i-1,nx); iR=_clampi(i+1,nx); jD=_clampi(j-1,ny); jU=_clampi(j+1,ny)
            ν̃=nut[i,j]
            dndx=0.5f0*(nut[iR,j]-nut[iL,j]); dndy=0.5f0*(nut[i,jU]-nut[i,jD])
            lap=nut[iR,j]+nut[iL,j]+nut[i,jU]+nut[i,jD]-4.0f0*ν̃; gnsq=dndx*dndx+dndy*dndy
            dudx=0.5f0*(ux[iR,j]-ux[iL,j]); dvdy=0.5f0*(uy[i,jU]-uy[i,jD])
            dvdx=0.5f0*(uy[iR,j]-uy[iL,j]); dudy=0.5f0*(ux[i,jU]-ux[i,jD]); S=abs(dvdx-dudy)   # |vorticity|
            adv = (ux[i,j]>=0.0f0 ? ux[i,j]*(ν̃-nut[iL,j]) : ux[i,j]*(nut[iR,j]-ν̃)) +          # 1st-order upwind
                  (uy[i,j]>=0.0f0 ? uy[i,j]*(ν̃-nut[i,jD]) : uy[i,j]*(nut[i,jU]-ν̃))
            d=max(wdist[i,j],1.0f0); χ=ν̃/max(nu_lat,1.0f-12); χ3=χ*χ*χ; fv1=χ3/(χ3+cv13)
            fv2=1.0f0-χ/(1.0f0+χ*fv1)
            # DDES (Spalart 2006): shield the attached BL (f_d→0 ⇒ RANS, d̃=d) and switch to LES (d̃=C_DES·Δ,
            # Δ=1 cell) in the wake ⇒ ν_t is grid-limited there (resolved LES / buffet), not over-grown RANS.
            dd = d
            if ddes == Int32(1)
                gmag=sqrt(dudx*dudx+dudy*dudy+dvdx*dvdx+dvdy*dvdy)
                rd=(ν̃+nu_lat)/(max(gmag,1.0f-10)*κ*κ*d*d)
                t8=8.0f0*rd; fd=1.0f0-tanh(t8*t8*t8)
                # AUDIT FIX (2026-07-07): the floor was 1.0, which clamped the LES branch to d̃=1 cell and made
                # every C_DES<1 unreachable (the ddes_cdes knob was inert; effective C_DES·Δ was always 1.0).
                # Floor now only guards positivity. PV_CDES=1.0 reproduces the pre-fix behavior exactly.
                dd = d - fd*max(0.0f0, d - cdes); dd = dd<0.05f0 ? 0.05f0 : dd
            end
            κ2d2=κ*κ*dd*dd; Stil=S+ν̃/κ2d2*fv2; Stil = Stil<0.3f0*S ? 0.3f0*S : Stil; Stil=max(Stil,1.0f-8)
            r=ν̃/(Stil*κ2d2); r = r>10.0f0 ? 10.0f0 : r; g=r+cw2*(r^6-r)
            fw=g*((1.0f0+cw36)/(g^6+cw36))^(1.0f0/6.0f0)
            prod=cb1*Stil*ν̃; dest=cw1*fw*(ν̃/dd)*(ν̃/dd)
            diff=(1.0f0/σ)*((nu_lat+ν̃)*lap+(1.0f0+cb2)*gnsq)
            ν̃n=ν̃+dt*(-adv+prod-dest+diff); ν̃n = ν̃n<0.0f0 ? 0.0f0 : ν̃n
            nut_o[i,j]=ν̃n; nuw_o[i,j]=dt*ν̃n*fv1                                                 # ν_t·dt for the turbulent stress
        end
    end
end

# PROPER deviatoric turbulent viscous-stress divergence (face-based, conservative) — the CORRECT way to apply
# an eddy viscosity ν_t, replacing the isotropic ∇²(ρu) Laplacian that DECAMBERED the outer suction. The
# Navier-Stokes turbulent stress τ_ij = ρν_t(∂_i u_j + ∂_j u_i − ⅔δ_ij ∇·u) concentrates the momentum transport
# in the SHEARED boundary layer (large ∂u/∂n) and nearly vanishes in the outer near-irrotational suction, so a
# turbulent BL is energized (delays separation) WITHOUT flattening the suction peak (no lift loss). `nuw` carries
# ν_t·dt (from SA/DDES). Momentum-conservative (flux differences of face stresses). Pressure-gated (spares shocks).
@kernel function _eddy_stress2T!(ux_o, uy_o, @Const(rho),@Const(ux),@Const(uy),@Const(Temp),@Const(nuw),
                                 cs2::Float32, gate_k::Float32, NX::Int32, NY::Int32)
    i,j=@index(Global,NTuple); nx=Int(NX); ny=Int(NY)
    @inbounds begin
        iL=_clampi(i-1,nx); iR=_clampi(i+1,nx); jD=_clampi(j-1,ny); jU=_clampi(j+1,ny)
        # face μ = ρ·ν_t·dt averaged to the face
        m0=rho[i,j]*nuw[i,j]
        mE=0.5f0*(m0+rho[iR,j]*nuw[iR,j]); mW=0.5f0*(m0+rho[iL,j]*nuw[iL,j])
        mN=0.5f0*(m0+rho[i,jU]*nuw[i,jU]); mS=0.5f0*(m0+rho[i,jD]*nuw[i,jD])
        # velocity gradients at the four faces (tangential grads via 4-point diagonal averages)
        # East (i+½,j)
        dudx_E=ux[iR,j]-ux[i,j]; dvdx_E=uy[iR,j]-uy[i,j]
        dudy_E=0.25f0*(ux[i,jU]+ux[iR,jU]-ux[i,jD]-ux[iR,jD]); dvdy_E=0.25f0*(uy[i,jU]+uy[iR,jU]-uy[i,jD]-uy[iR,jD])
        # West (i−½,j)
        dudx_W=ux[i,j]-ux[iL,j]; dvdx_W=uy[i,j]-uy[iL,j]
        dudy_W=0.25f0*(ux[iL,jU]+ux[i,jU]-ux[iL,jD]-ux[i,jD]); dvdy_W=0.25f0*(uy[iL,jU]+uy[i,jU]-uy[iL,jD]-uy[i,jD])
        # North (i,j+½)
        dudy_N=ux[i,jU]-ux[i,j]; dvdy_N=uy[i,jU]-uy[i,j]
        dudx_N=0.25f0*(ux[iR,j]+ux[iR,jU]-ux[iL,j]-ux[iL,jU]); dvdx_N=0.25f0*(uy[iR,j]+uy[iR,jU]-uy[iL,j]-uy[iL,jU])
        # South (i,j−½)
        dudy_S=ux[i,j]-ux[i,jD]; dvdy_S=uy[i,j]-uy[i,jD]
        dudx_S=0.25f0*(ux[iR,jD]+ux[iR,j]-ux[iL,jD]-ux[iL,j]); dvdx_S=0.25f0*(uy[iR,jD]+uy[iR,j]-uy[iL,jD]-uy[iL,j])
        c23=2.0f0/3.0f0; c43=4.0f0/3.0f0
        τxx_E=mE*(c43*dudx_E-c23*dvdy_E); τxx_W=mW*(c43*dudx_W-c23*dvdy_W)
        τxy_N=mN*(dudy_N+dvdx_N);         τxy_S=mS*(dudy_S+dvdx_S)
        τxy_E=mE*(dudy_E+dvdx_E);         τxy_W=mW*(dudy_W+dvdx_W)
        τyy_N=mN*(c43*dvdy_N-c23*dudx_N); τyy_S=mS*(c43*dvdy_S-c23*dudx_S)
        Fx=(τxx_E-τxx_W)+(τxy_N-τxy_S); Fy=(τxy_E-τxy_W)+(τyy_N-τyy_S)
        # pressure gate (spare shocks): 1 in smooth/shear, →0 at a true pressure jump.
        # gate_k = steepness (legacy 12; 0 ⇒ UNGATED). SBLI NOTE (2026-07-10): under a RANS closure the
        # gate suppresses the turbulent stress exactly at the SHOCK FOOT — the mechanism that spreads the
        # wall-pressure rise over the λ-foot (the experimental Cp "softness" the sharp-shock plots lack).
        # The gate was an LES-era protection for the INVISCID shock; with ν_t from SST it is likely
        # over-protective. gate_k=12 = byte-identical legacy; A/B via PV_EDDYGATE before changing defaults.
        gate=1f0
        if gate_k > 0f0
            pc=rho[i,j]*cs2*Temp[i,j]; pim=rho[iL,j]*cs2*Temp[iL,j]; pip=rho[iR,j]*cs2*Temp[iR,j]
            pjm=rho[i,jD]*cs2*Temp[i,jD]; pjp=rho[i,jU]*cs2*Temp[i,jU]
            sx=abs(pip-2f0*pc+pim)/(abs(pip)+2f0*abs(pc)+abs(pim)+1f-6); sy=abs(pjp-2f0*pc+pjm)/(abs(pjp)+2f0*abs(pc)+abs(pjm)+1f-6)
            gate=clamp(1f0-gate_k*max(sx,sy),0f0,1f0)
        end
        r0=rho[i,j]
        ux_o[i,j]=(r0*ux[i,j]+gate*Fx)/r0; uy_o[i,j]=(r0*uy[i,j]+gate*Fy)/r0
    end
end

# PG-AWARE LOG-LAW u_τ SOLVE (shared by WMLES + the SST wall-function band, 2026-07-08). Newton on
#   u_mag = u_τ·((1/κ)·ln y⁺ + B) + Δu_pg,   Δu_pg = d·G/u_τ  (Larsson 2010 / Hickel-Touber-Bodart-Larsson
# 2012 §3.2 pressure-gradient correction; G = (∇p·û)/ρ = per-unit-mass STREAMWISE pressure gradient).
# The equilibrium log-law is exact only at ∂p/∂s≈0 — at the shock foot and in the recovery region the
# PG term restores the wall stress (adverse G>0 ⇒ Δu_pg>0 ⇒ SMALLER u_τ for the same u_mag). Schur-bounded
# |Δu_pg| ≤ 0.5·u_mag so a stagnation-point gradient can never run the solve away. pgr=0 ⇒ legacy exactly.
@inline function _utau_loglaw_pg(umag::Float32, d::Float32, ν::Float32, G::Float32, pgr::Int32)
    κ=0.41f0; B=5.2f0
    uτ=sqrt(umag*ν/d)                                          # laminar seed u_τ=√(ν·u/d)
    lim=0.5f0*umag
    for _ in 1:10
        yp=d*uτ/ν; yp = yp<1.1f0 ? 1.1f0 : yp
        lg=log(yp)
        pg=0.0f0; dpg=0.0f0
        if pgr==Int32(1)
            pg = d*G/uτ
            if pg>lim; pg=lim; elseif pg<-lim; pg=-lim; else; dpg=-pg/uτ; end   # d(d·G/uτ)/duτ = −pg/uτ (0 at the clamp)
        end
        fn=uτ*((1.0f0/κ)*lg+B)+pg-umag; df=(1.0f0/κ)*(lg+1.0f0)+B+dpg
        # REVIEW FIX (2026-07-09, adversarially CONFIRMED by direct Float32 execution): for adverse G the
        # dpg term is negative and unbounded as uτ shrinks — df can hit EXACTLY 0.0f0 (reproduced:
        # base 18.740892 + dpg −18.740892) ⇒ uτ=Inf ⇒ next iter Inf−Inf/Inf=NaN, and NaN pierces EVERY
        # downstream clamp (NaN comparisons are false) into wm/tk/tom ⇒ permanent field NaN. Expected
        # O(0.1-10) events over a 200k-step production. Floor the derivative (base term ≥7.87 ⇒ pgr=0
        # byte-identical) and cap uτ at umag (physically u_τ ≪ u_mag).
        df = df<1.0f0 ? 1.0f0 : df
        uτ=uτ-fn/df; uτ = uτ<1.0f-6 ? 1.0f-6 : (uτ>umag ? umag : uτ)
    end
    return uτ
end

# WALL-MODELED LES — equilibrium (log-law) wall model. The first cells sit in the log layer (not the y⁺~1
# sublayer); the near-wall eddy viscosity is set from the LOCAL log-law friction velocity u_τ (solved by
# Newton from u_mag & wall distance d), ν_t = κ·u_τ·d·(1−e^{−d⁺/A⁺})² (Van Driest), WINDOWED to d<hmatch so it
# stays a THIN near-wall layer ⇒ the correct turbulent wall stress WITHOUT resolving the sublayer, and far less
# decambering than SA's thick spread on a coarse grid. Algebraic (no transport). Writes ν_t·dt into `wm`.
# 2026-07-08: u_τ solve upgraded to the PG-aware form above (pgr=1 ⇒ Larsson streamwise-∂p/∂s correction,
# computed from central-difference p=ρ·cs²·T; pgr=0 ⇒ byte-identical legacy log-law).
@kernel function _wmles_kernel!(nuw_o, @Const(ux), @Const(uy), @Const(rho), @Const(Temp), @Const(wdist), @Const(chi),
                                nu_lat::Float32, dt::Float32, hmatch::Float32, @Const(gtrip), trip_on::Int32,
                                cs2::Float32, pgr::Int32, NX::Int32, NY::Int32)
    i,j=@index(Global,NTuple); κ=0.41f0; Ap=26.0f0; nx=Int(NX); ny=Int(NY)
    @inbounds begin
        d=wdist[i,j]; umag=sqrt(ux[i,j]*ux[i,j]+uy[i,j]*uy[i,j])
        # PRESCRIBED TRANSITION (2026-07-07): γ_trip∈[0,1] gates the wall-stress ν_t — laminar (γ=0)
        # ahead of the trip station, turbulent aft (Case 9 trips at x/c=0.03; free transition is
        # unmodeled — full γ-Reθ transport is the SST/G3 milestone; this field is its plumbing).
        γt = trip_on==Int32(1) ? gtrip[i,j] : 1.0f0
        if chi[i,j]>0.5f0 || d>=hmatch || d<0.5f0 || umag<1.0f-6 || γt<=0.0f0
            nuw_o[i,j]=0.0f0
        else
            ν=max(nu_lat,1.0f-9)
            G=0.0f0
            if pgr==Int32(1)                                    # streamwise per-unit-mass ∂p/∂s (Larsson PG term)
                iL=_clampi(i-1,nx); iR=_clampi(i+1,nx); jD=_clampi(j-1,ny); jU=_clampi(j+1,ny)
                dpdx=0.5f0*(rho[iR,j]*Temp[iR,j]-rho[iL,j]*Temp[iL,j])*cs2
                dpdy=0.5f0*(rho[i,jU]*Temp[i,jU]-rho[i,jD]*Temp[i,jD])*cs2
                G=(dpdx*ux[i,j]+dpdy*uy[i,j])/(umag*max(rho[i,j],1.0f-3))
            end
            uτ=_utau_loglaw_pg(umag,d,ν,G,pgr)
            # ν_t = κ·u_τ·ℓ_mix (log-layer), but CAP the mixing length ℓ_mix=min(d, 0.3·hmatch) so ν_t PLATEAUS
            # in the log layer instead of growing unbounded to the BL edge (the flaw that over-diffused the outer
            # flow); Van Driest sublayer damping + a Gaussian window confine it to a THIN near-wall band.
            dp=d*uτ/ν; vd=1.0f0-exp(-dp/Ap); ℓ=d<0.3f0*hmatch ? d : 0.3f0*hmatch
            nu_w=γt*dt*κ*uτ*ℓ*vd*vd*exp(-(d/hmatch)*(d/hmatch))
            nuw_o[i,j]=isfinite(nu_w) ? nu_w : 0.0f0                # NaN-rejecting write (defense in depth)
        end
    end
end

# ============================================================================================================
# SST k-ω (Menter 2003) — the calibrated two-equation closure (2026-07-08). Replaces the equilibrium WMLES
# ν_t with TRANSPORT: k and ω carry history and pressure-gradient response through the shock and into the
# recovery region — the physics the algebraic log-law cannot represent (the remaining rooftop/Cm gap owner).
#   Dk/Dt = P_k − β*·k·ω + ∇·((ν+σ_k ν_t)∇k)
#   Dω/Dt = γ·S² − β·ω²  + ∇·((ν+σ_ω ν_t)∇ω) + 2(1−F1)·σ_ω2/ω·∇k·∇ω
#   ν_t = a1·k / max(a1·ω, S·F2),  P_k = min(ν_t·S², 10·β*·k·ω)·γ_trip   (trip-gated production)
# WALL-FUNCTION BC (first cells at y⁺~50-150 — sublayer unresolved by design): in the band d<2.5 impose the
# analytic log-layer values k=γ·u_τ²/√β*, ω=√(ω_vis²+ω_log²) with ω_log=u_τ/(√β*·κ·d), ω_vis=6ν/(β1·d²)
# (Menter blend) and u_τ from the PG-AWARE log-law solve above — so the Larsson correction acts on the k-ω
# state, not just an algebraic ν_t. These BCs reproduce ν_t=κ·u_τ·d in the log layer exactly (k/ω identity),
# i.e. SST ⊇ WMLES near the wall, + outer-layer transport. Numerics like SA: 1st-order upwind advection,
# conservative face-averaged variable-coefficient diffusion, explicit update, positivity floors. Per-unit-mass
# (incompressible-form) equations — standard for transonic aero closures at these density ratios.
# Fine-box borders: inflow edges (left/top/bottom) pinned to freestream (k∞,ω∞); outflow (right) zero-gradient.
# Solid (χ>0.5 during penalized startup, sdf≤0 after χ-off): k=0, ν_t=0 (band cells are imposed, so solid
# stencil reads never reach the transport region). Writes ν_t·dt into `wm` — applied via _eddy_stress2T!
# (the strain-consistent deviatoric tensor, the no-decamber path DDES validated), NOT the ∇² Laplacian.
# PV_SSTBAND (distribution campaign E2, 2026-07-12): the wall-function band height in FINE cells
# (default 2.5 = legacy, byte-identical). The band imposes the analytic log-layer k,ω — its height
# sets where the model hands over to transport, i.e. the imposed-νt depth (the BL-fullness lever:
# the laminar A/B showed excess νt drives the shock-heavy distribution). Read once per process.
const _SST_BAND_REF = Ref{Float32}(-1f0)
function _sst_band()
    if _SST_BAND_REF[] < 0f0
        _SST_BAND_REF[] = parse(Float32, get(ENV, "PV_SSTBAND", "2.5"))
    end
    return _SST_BAND_REF[]
end
# PV_KATO (distribution campaign E3, 2026-07-12): Kato-Launder production P_k = ν_t·S·Ω instead of
# ν_t·S² (Kato & Launder 1993). At shocks and at the LE stagnation point the strain S is large but
# the rotation Ω≈0 — the S² form manufactures spurious k exactly there (the "shock-generated
# turbulence" defect: E0-E2 showed peakM 1.41-1.45 immune to every ν_t lever while the laminar
# contrast sits at 1.24 ≈ experiment). S·Ω vanishes in irrotational straining, is identical to S²
# in the log-layer (S≈Ω there), and is the standard transonic fix (CFX/FUN3D option). Default OFF.
const _SST_KATO_REF = Ref{Int}(-1)
function _sst_kato()
    if _SST_KATO_REF[] < 0
        _SST_KATO_REF[] = get(ENV, "PV_KATO", "0") == "1" ? 1 : 0
    end
    return _SST_KATO_REF[] == 1
end

@kernel function _sst_kernel!(tk_o, tom_o, nuw_o, @Const(tk), @Const(tom), @Const(ux), @Const(uy),
                              @Const(rho), @Const(Temp), @Const(wdist), @Const(chi), @Const(sdf),
                              @Const(gtrip), trip_on::Int32, nu_lat::Float32, dt::Float32, cs2::Float32,
                              kinf::Float32, winf::Float32, uin::Float32, pgr::Int32, ng::Int32,
                              @Const(hk), @Const(hom), nested_kw::Int32, band::Float32, kato::Int32,
                              NX::Int32, NY::Int32)
    i,j=@index(Global,NTuple); nx=Int(NX); ny=Int(NY); g=Int(ng)
    βs=0.09f0; sβs=0.3f0; κ=0.41f0; a1=0.31f0                    # β*, √β*
    σk1=0.85f0; σw1=0.5f0; β1=0.075f0; γ1=0.5532f0               # inner (k-ω) set
    σk2=1.0f0;  σw2=0.856f0; β2=0.0828f0; γ2=0.4403f0            # outer (k-ε) set
    NUTMAX=0.15f0                                                # explicit-stability clamp (phys peak ~0.07; was 0.3 — tightened after the chord-300 TE A/B)
    @inbounds begin
        if chi[i,j]>0.5f0 || sdf[i,j]<=0.0f0                      # solid (penalized startup OR inside the SDF wall)
            tk_o[i,j]=0.0f0; tom_o[i,j]=winf; nuw_o[i,j]=0.0f0
        elseif nested_kw==Int32(1) && (i<=g || i>nx-g || j<=g || j>ny-g)
            # NESTED k/ω halo (REVIEW FIX 2026-07-09): pin the parent-prolonged k/ω on ALL edges — the
            # freestream pin laminarized a patch's mid-BL inflow (k∞~7e-7 vs incoming BL k~1e-2).
            kb=hk[i,j]; wb=hom[i,j]
            kb = kb<1.0f-12 ? 1.0f-12 : kb; wb = wb<1.0f-6 ? 1.0f-6 : wb
            if !(isfinite(kb) && isfinite(wb)); kb=kinf; wb=winf; end
            tk_o[i,j]=kb; tom_o[i,j]=wb
            nut=kb/wb; nuw_o[i,j]=dt*(nut>NUTMAX ? NUTMAX : nut)
        elseif i<=g || j<=g || j>ny-g                             # inflow/far-field borders ⇒ freestream
            tk_o[i,j]=kinf; tom_o[i,j]=winf; nuw_o[i,j]=dt*kinf/max(winf,1.0f-6)
        elseif i>nx-g                                             # outflow border ⇒ zero-gradient (wake exits cleanly)
            tk_o[i,j]=tk[nx-g,j]; tom_o[i,j]=tom[nx-g,j]; nuw_o[i,j]=dt*min(tk[nx-g,j]/max(tom[nx-g,j],1.0f-6),NUTMAX)
        else
            d=wdist[i,j]; ν=max(nu_lat,1.0f-9)
            umag=sqrt(ux[i,j]*ux[i,j]+uy[i,j]*uy[i,j])
            γt = trip_on==Int32(1) ? gtrip[i,j] : 1.0f0
            if d<band                                             # WALL-FUNCTION band: impose analytic log-layer k,ω
                dh = d<0.5f0 ? 0.5f0 : d
                # BAND VELOCITY CAP (2026-07-08, chord-300 TE A/B): during a TE transient band cells can see
                # umag~2-3 — outside the log-law's validity — and the band faithfully converts that into
                # u_τ~0.15, k~0.07, i.e. a ν_t injection right where the flow is most marginal (the SST+small-
                # patch NaN). Cap the log-law input at 1.5·U∞: bounds k_bc at its physical scale; a no-op for
                # any attached-BL state (edge velocity ≤ ~1.4·U∞ on this airfoil).
                ucap=1.5f0*uin; umag = umag>ucap ? ucap : umag
                uτ=0.0f0
                if umag>=1.0f-6
                    G=0.0f0
                    if pgr==Int32(1)
                        iL=_clampi(i-1,nx); iR=_clampi(i+1,nx); jD=_clampi(j-1,ny); jU=_clampi(j+1,ny)
                        dpdx=0.5f0*(rho[iR,j]*Temp[iR,j]-rho[iL,j]*Temp[iL,j])*cs2
                        dpdy=0.5f0*(rho[i,jU]*Temp[i,jU]-rho[i,jD]*Temp[i,jD])*cs2
                        G=(dpdx*ux[i,j]+dpdy*uy[i,j])/(umag*max(rho[i,j],1.0f-3))
                    end
                    uτ=_utau_loglaw_pg(umag,dh,ν,G,pgr)
                end
                wvis=6.0f0*ν/(β1*dh*dh); wlog=uτ/(sβs*κ*dh)
                ω_bc=sqrt(wvis*wvis+wlog*wlog); ω_bc = ω_bc<1.0f-6 ? 1.0f-6 : ω_bc
                k_bc=γt*uτ*uτ/sβs
                # NaN-rejecting writes (defense in depth): a NaN here would spread permanently via the tensor
                if !(isfinite(k_bc) && isfinite(ω_bc)); k_bc=0.0f0; ω_bc=winf; end
                tk_o[i,j]=k_bc; tom_o[i,j]=ω_bc
                nut=k_bc/ω_bc; nut = isfinite(nut) ? (nut>NUTMAX ? NUTMAX : nut) : 0.0f0
                nuw_o[i,j]=dt*nut
            else                                                  # TRANSPORT region
                iL=_clampi(i-1,nx); iR=_clampi(i+1,nx); jD=_clampi(j-1,ny); jU=_clampi(j+1,ny)
                k0=tk[i,j]; w0=tom[i,j]; w0 = w0<1.0f-6 ? 1.0f-6 : w0
                # strain rate S=√(2S_ij S_ij) from central differences
                dudx=0.5f0*(ux[iR,j]-ux[iL,j]); dudy=0.5f0*(ux[i,jU]-ux[i,jD])
                dvdx=0.5f0*(uy[iR,j]-uy[iL,j]); dvdy=0.5f0*(uy[i,jU]-uy[i,jD])
                Sxy=0.5f0*(dudy+dvdx); S2=2.0f0*(dudx*dudx+dvdy*dvdy+2.0f0*Sxy*Sxy); S=sqrt(S2)
                # blending functions (wall distance d, cells)
                dkdx=0.5f0*(tk[iR,j]-tk[iL,j]); dkdy=0.5f0*(tk[i,jU]-tk[i,jD])
                dwdx=0.5f0*(tom[iR,j]-tom[iL,j]); dwdy=0.5f0*(tom[i,jU]-tom[i,jD])
                gkw=dkdx*dwdx+dkdy*dwdy
                CDkw=2.0f0*σw2/w0*gkw; CDkw = CDkw<1.0f-10 ? 1.0f-10 : CDkw
                sqk=sqrt(k0<0.0f0 ? 0.0f0 : k0)
                arg1=sqk/(βs*w0*d); t2=500.0f0*ν/(d*d*w0); arg1 = arg1<t2 ? t2 : arg1
                t3=4.0f0*σw2*k0/(CDkw*d*d); arg1 = arg1<t3 ? arg1 : t3
                arg1 = arg1>4.0f0 ? 4.0f0 : arg1                  # tanh(x⁴) saturated beyond ~2.5 — avoids overflow
                a14=arg1*arg1*arg1*arg1; F1=tanh(a14)
                arg2=2.0f0*sqk/(βs*w0*d); arg2 = arg2<t2 ? t2 : arg2; arg2 = arg2>4.0f0 ? 4.0f0 : arg2
                F2=tanh(arg2*arg2)
                nut=a1*k0/max(a1*w0, S*F2); nut = nut>NUTMAX ? NUTMAX : nut
                # production (limited, trip-gated)
                Pk = kato==Int32(1) ? nut*S*abs(dudy-dvdx) : nut*S2   # Kato-Launder: S·Ω kills spurious P_k at shocks/stagnation
                plim=10.0f0*βs*k0*w0; Pk = Pk>plim ? plim : Pk; Pk*=γt
                # blended constants
                σk=F1*σk1+(1.0f0-F1)*σk2; σw=F1*σw1+(1.0f0-F1)*σw2
                βc=F1*β1+(1.0f0-F1)*β2;  γc=F1*γ1+(1.0f0-F1)*γ2
                # 1st-order upwind advection (same scheme as SA)
                advk = (ux[i,j]>=0.0f0 ? ux[i,j]*(k0-tk[iL,j]) : ux[i,j]*(tk[iR,j]-k0)) +
                       (uy[i,j]>=0.0f0 ? uy[i,j]*(k0-tk[i,jD]) : uy[i,j]*(tk[i,jU]-k0))
                advw = (ux[i,j]>=0.0f0 ? ux[i,j]*(w0-tom[iL,j]) : ux[i,j]*(tom[iR,j]-w0)) +
                       (uy[i,j]>=0.0f0 ? uy[i,j]*(w0-tom[i,jD]) : uy[i,j]*(tom[i,jU]-w0))
                # conservative face-averaged variable-coefficient diffusion; face ν_t ≈ ½(k/ω)_c + ½(k/ω)_nb
                ntE=0.5f0*(nut+min(max(tk[iR,j],0f0)/max(tom[iR,j],1f-6),NUTMAX)); ntW=0.5f0*(nut+min(max(tk[iL,j],0f0)/max(tom[iL,j],1f-6),NUTMAX))
                ntN=0.5f0*(nut+min(max(tk[i,jU],0f0)/max(tom[i,jU],1f-6),NUTMAX)); ntS=0.5f0*(nut+min(max(tk[i,jD],0f0)/max(tom[i,jD],1f-6),NUTMAX))
                Dk=(ν+σk*ntE)*(tk[iR,j]-k0)-(ν+σk*ntW)*(k0-tk[iL,j])+(ν+σk*ntN)*(tk[i,jU]-k0)-(ν+σk*ntS)*(k0-tk[i,jD])
                Dw=(ν+σw*ntE)*(tom[iR,j]-w0)-(ν+σw*ntW)*(w0-tom[iL,j])+(ν+σw*ntN)*(tom[i,jU]-w0)-(ν+σw*ntS)*(w0-tom[i,jD])
                kn=k0+dt*(-advk+Pk-βs*k0*w0+Dk)
                wn=w0+dt*(-advw+γc*S2-βc*w0*w0+Dw+2.0f0*(1.0f0-F1)*σw2/w0*gkw)
                kn = kn<1.0f-12 ? 1.0f-12 : (kn>0.5f0 ? 0.5f0 : kn)
                wn = wn<1.0f-6 ? 1.0f-6 : wn
                if !(isfinite(kn) && isfinite(wn)); kn=kinf; wn=winf; end    # NaN-rejecting write (clamps pass NaN through)
                tk_o[i,j]=kn; tom_o[i,j]=wn
                nuw_o[i,j]=isfinite(nut) ? dt*nut : 0.0f0
            end
        end
    end
end

# thermal BC kernels (carry T).
# POINT-VORTEX FAR-FIELD CORRECTION (2026-07-07, the "circulation-starved domain" fix): a lifting airfoil's
# own circulation deflects the flow by Δα ≈ Cl·c/(4π·r) even far away (~1-1.8° at 2 chords for Cl 0.8) —
# Dirichlet-clamping the boundaries to (U∞,0) suppresses it and cuts the EFFECTIVE angle of attack by that
# much (the standard 2D small-domain error; the cure used by CFL3D/FUN3D-class codes). When gam≠0 the
# boundary state becomes freestream + the clockwise-positive point vortex at (xv,yv):
#   (du,dv) = gam/(2π r²)·(y−yv, −(x−xv)),  Γ_clockwise = ½·U·c·Cl  (Kutta-Joukovski, lift +y).
# gam, xv, yv are in the GRID'S OWN cell/velocity units (Γ scales as 1/dx across levels). gam=0 ⇒ exact
# legacy behavior. The caller feeds gam back from the computed Cl (under-relaxed).
@inline function _fvortex(i,j, gam::Float32, xv::Float32, yv::Float32)
    dx=Float32(i)-xv; dy=Float32(j)-yv; r2=dx*dx+dy*dy+1.0f-6
    c=gam/(6.2831853f0*r2); return c*dy, -c*dx
end
# PRANDTL-GLAUERT vortex (Thomas & Salas 1986; SHOCK WORKSTREAM S2 finding 2026-07-13): the
# INCOMPRESSIBLE vortex above is ANGULAR-BLIND — the correct compressible far-field runs from ×β
# (streamwise) to ×1/β (transverse), a 1/β²≈2.14× streamwise-to-transverse SPREAD it flattens to
# ×1 everywhere; in particular it under-weights the transverse UPWASH by 1/β≈1.46× at M∞=0.73.
# Lead hypothesis for the +5.5 % Cl domain-height sensitivity S2 measured (10.7c→16c); a correct
# far-field should make height nearly irrelevant. Compressible form: the tangential perturbation is
# q_θ = Γ/(2πr) · β/(1 − M∞²·sin²θ), θ from the freestream axis. Both forms preserve total Γ (∮=Γ);
# the difference is the ANGULAR distribution, not the net circulation.
# m2=0, beta=1 reproduces the incompressible form EXACTLY (the PV_PGV=0 legacy gate).
@inline function _fvortex_pg(i,j, gam::Float32, xv::Float32, yv::Float32, m2::Float32, beta::Float32)
    dx=Float32(i)-xv; dy=Float32(j)-yv; r2=dx*dx+dy*dy+1.0f-6
    F=beta/(1.0f0 - m2*(dy*dy/r2))
    c=gam*F/(6.2831853f0*r2); return c*dy, -c*dx
end
@kernel function _inlet2T_x!(rho,ux,uy,T, rho_inf::Float32, Uin::Float32, Tinf::Float32, ng::Int32, gam::Float32, xv::Float32, yv::Float32, m2::Float32, beta::Float32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=ng
        du,dv=_fvortex_pg(i,j,gam,xv,yv,m2,beta)
        rho[i,j]=rho_inf; ux[i,j]=Uin+du; uy[i,j]=dv; T[i,j]=Tinf
    end
end
# SUBSONIC RESERVOIR (total-pressure) inlet: fix the stagnation state (ρ0,T0); the inlet Mach SELF-ADJUSTS from
# the interior static pressure so the mass flux is FREE (no backup) — the physical subsonic-inlet BC that lets a
# choked nozzle actually choke. Reads p_i just inside; M²=(2/(γ−1))((p0/p_i)^((γ−1)/γ)−1), clamped subsonic.
@kernel function _inlet_res2T_x!(rho,ux,uy,T, rho0::Float32, T0::Float32, cs2::Float32, drot::Float32, ng::Int32)
    i,j=@index(Global,NTuple); g=Int(ng)
    @inbounds if i<=g
        Deff=3.0f0+drot; γ=(Deff+2.0f0)/Deff; p0=rho0*cs2*T0
        pin=rho[g+1,j]*cs2*T[g+1,j]; pr=max(p0/max(pin,1f-4), 1.0f0)
        M2=(2.0f0/(γ-1.0f0))*(pr^((γ-1.0f0)/γ)-1.0f0); M2 = M2<0f0 ? 0f0 : (M2>0.98f0 ? 0.98f0 : M2)
        Tt=T0/(1.0f0+0.5f0*(γ-1.0f0)*M2); ρt=rho0*(Tt/T0)^(1.0f0/(γ-1.0f0)); a=sqrt(γ*cs2*Tt)
        rho[i,j]=ρt; ux[i,j]=sqrt(M2)*a; uy[i,j]=0.0f0; T[i,j]=Tt
    end
end
@kernel function _outlet2T_x!(rho,ux,uy,T, NX::Int32, ng::Int32)
    i,j=@index(Global,NTuple); nx=Int(NX)
    @inbounds if i>nx-ng; s=nx-ng; rho[i,j]=rho[s,j]; ux[i,j]=ux[s,j]; uy[i,j]=uy[s,j]; T[i,j]=T[s,j]; end
end
@kernel function _farfield2T_y!(rho,ux,uy,T, rho_inf::Float32, Uin::Float32, Tinf::Float32, NY::Int32, ng::Int32, gam::Float32, xv::Float32, yv::Float32, m2::Float32, beta::Float32)
    i,j=@index(Global,NTuple); ny=Int(NY)
    @inbounds if (j<=ng)||(j>ny-ng)
        du,dv=_fvortex_pg(i,j,gam,xv,yv,m2,beta)
        rho[i,j]=rho_inf; ux[i,j]=Uin+du; uy[i,j]=dv; T[i,j]=Tinf
    end
end
# NESTED fine-box halo pin: overwrite the ng-border (all four edges) of the live arrays with the prolongation-
# filled halo buffers h*. This is the fine box's boundary condition (the coarse base supplies h* by prolongation),
# re-imposed after every RK sub-stage exactly where applyBC! re-imposes the domain BCs for a normal grid.
@kernel function _pin_halo2T!(rho,ux,uy,T, @Const(hrho),@Const(hux),@Const(huy),@Const(hT), NX::Int32,NY::Int32, ng::Int32)
    i,j=@index(Global,NTuple); nx=Int(NX); ny=Int(NY); g=Int(ng)
    @inbounds if (i<=g)||(i>nx-g)||(j<=g)||(j>ny-g)
        rho[i,j]=hrho[i,j]; ux[i,j]=hux[i,j]; uy[i,j]=huy[i,j]; T[i,j]=hT[i,j]
    end
end
# ABSORBING far-field layer at the top/bottom (y) walls. A wave that reaches a hard/abruptly-forced boundary
# reflects; a good sponge ramps the pull toward the freestream state SMOOTHLY from ~0 at the interior edge to
# strong at the wall so the wave is dissipated gradually with almost no reflection. Key fixes vs the old
# version: (1) CUBIC onset (t³) instead of quadratic ⇒ the interior edge is far gentler (∂σ/∂d ≈ 0 there),
# which is what kills the reflected bands emanating from the walls; (2) the OUTERMOST cells are pinned exactly
# to the freestream (σ→smax_wall) so the reconstruction sees a clean far state. Width is set wide by the caller.
@kernel function _sponge2T_y!(rho,ux,uy,T, rho_inf::Float32, Uin::Float32, Tinf::Float32, NY::Int32, sw::Int32, smax::Float32, gam::Float32, xv::Float32, yv::Float32, m2::Float32, beta::Float32)
    i,j=@index(Global,NTuple); ny=Int(NY); w=Int(sw)
    @inbounds begin
        dy = j-1 < ny-j ? j-1 : ny-j
        if dy<w
            t=1.0f0-Float32(dy)/Float32(w); σ=smax*t*t*t                 # cubic ⇒ gentle onset, no reflection
            if dy<=1; σ=1.0f0; end                                       # pin the 2 wall rows exactly to freestream (clean BC)
            du,dv=_fvortex_pg(i,j,gam,xv,yv,m2,beta)                     # PG-corrected far-field target (gam=0 ⇒ legacy; m2=0,beta=1 ⇒ incompressible)
            rho[i,j]+=σ*(rho_inf-rho[i,j]); ux[i,j]+=σ*(Uin+du-ux[i,j]); uy[i,j]+=σ*(dv-uy[i,j]); T[i,j]+=σ*(Tinf-T[i,j])
        end
    end
end

# ---- Roadmap 1.1: NSCBC characteristic non-reflecting far-field (Riemann-invariant form) ------------------
# The sponge (above) works but at high Re its forced-freestream inner edge becomes Kelvin-Helmholtz unstable
# (the top/bottom bands seen in the 160M video). This is the principled cure: a compressible characteristic
# far-field (FUN3D/OVERFLOW class). At the top/bottom (y) walls, decompose the boundary-NORMAL flow into an
# OUTGOING Riemann invariant (from the interior) and an INCOMING one (from the freestream), then take entropy
# + the TANGENTIAL velocity from whichever side the normal flow actually comes from. Because the tangential
# velocity is left FREE (never forced to U), there is no shear layer at the wall ⇒ no roll-up, and outgoing
# acoustic/vortical waves leave without reflection. Supersonic-normal branches fall back to full extrapolation
# (outflow) or full freestream (inflow). p = ρ·R·T, c² = γ·R·T with R = cs2, γ = (D_eff+2)/D_eff.
@kernel function _nscbc2T_y!(rho,ux,uy,T, rho_inf::Float32, Uin::Float32, Tinf::Float32,
                             gam::Float32, Rgas::Float32, NY::Int32, ng::Int32)
    i,j=@index(Global,NTuple); ny=Int(NY); g=Int(ng)
    @inbounds if (j<=g)||(j>ny-g)
        top = j > ny-g
        sy  = top ? 1.0f0 : -1.0f0
        js  = top ? ny-g : g+1                        # first interior row just inside the ghost band
        rd=rho[i,js]; uxd=ux[i,js]; uyd=uy[i,js]; Td=T[i,js]
        gm1  = gam-1.0f0
        cd   = sqrt(gam*Rgas*(Td   > 1.0f-4 ? Td   : 1.0f-4))
        cinf = sqrt(gam*Rgas*(Tinf > 1.0f-4 ? Tinf : 1.0f-4))
        qnd  = sy*uyd                                 # outward boundary-normal velocity (interior side)
        rb=rd; uxb=uxd; uyb=uyd; Tb=Td
        if qnd >= cd                                  # supersonic OUTFLOW ⇒ full extrapolation
            # (defaults above already hold the interior state)
        elseif qnd <= -cd                             # supersonic INFLOW ⇒ full freestream
            rb=rho_inf; uxb=Uin; uyb=0.0f0; Tb=Tinf
        else                                          # subsonic normal ⇒ Riemann split
            Rout = qnd + 2.0f0*cd/gm1                 # outgoing invariant (interior)
            Rin  = 0.0f0 - 2.0f0*cinf/gm1             # incoming invariant (freestream, qn_inf = 0)
            qnb  = 0.5f0*(Rout+Rin)
            cb   = 0.25f0*gm1*(Rout-Rin); cb = cb < 1.0f-4 ? 1.0f-4 : cb
            local sB::Float32
            if qnb >= 0.0f0                            # OUTFLOW: entropy + tangential vel from interior
                sB = (rd*Rgas*Td)/(rd^gam); uxb = uxd
            else                                      # INFLOW: entropy + tangential vel from freestream
                sB = (rho_inf*Rgas*Tinf)/(rho_inf^gam); uxb = Uin
            end
            rb  = (cb*cb/(gam*sB))^(1.0f0/gm1)         # ρ from c² = γ·s·ρ^(γ−1)
            Tb  = (sB*rb^gam)/(rb*Rgas)                # T from p = s·ρ^γ = ρ·R·T
            uyb = sy*qnb
        end
        rho[i,j]=rb; ux[i,j]=uxb; uy[i,j]=uyb; T[i,j]=Tb
    end
end

# conservative ∇⁶ SVV hyperviscosity (k⁶-selective ⇒ kills the 2Δ grid-scale ripples/texture that the
# low-dissipation MC reconstruction leaves, WITHOUT touching the resolved scales). Applied to ρ and the
# momentum ρu,ρv (both flux divergences ⇒ mass+momentum conserved). Double-buffered (no race). LES-mode only.
@inline _d6(a,b,c,d,e,f,g) = a - 6.0f0*b + 15.0f0*c - 20.0f0*d + 15.0f0*e - 6.0f0*f + g
@kernel function _svvT_kernel!(rho_o, ux_o, uy_o, @Const(rho), @Const(ux), @Const(uy), @Const(Temp), eps::Float32, eps_rho::Float32, nu_mol::Float32, @Const(nuw), wm_scale::Float32, wm_on::Int32, cs2::Float32, NX::Int32, NY::Int32)
    i, j = @index(Global, NTuple); nx=Int(NX); ny=Int(NY)
    @inbounds begin
        a1=_clampi(i-1,nx);a2=_clampi(i-2,nx);a3=_clampi(i-3,nx);b1=_clampi(i+1,nx);b2=_clampi(i+2,nx);b3=_clampi(i+3,nx)
        c1=_clampi(j-1,ny);c2=_clampi(j-2,ny);c3=_clampi(j-3,ny);e1=_clampi(j+1,ny);e2=_clampi(j+2,ny);e3=_clampi(j+3,ny)
        r0=rho[i,j]; mx0=r0*ux[i,j]; my0=r0*uy[i,j]
        # MOMENTUM k⁶ hyperdiffusion (smooths the velocity/vorticity grid-scale texture)
        hmx= _d6(rho[a3,j]*ux[a3,j],rho[a2,j]*ux[a2,j],rho[a1,j]*ux[a1,j],mx0,rho[b1,j]*ux[b1,j],rho[b2,j]*ux[b2,j],rho[b3,j]*ux[b3,j]) +
             _d6(rho[i,c3]*ux[i,c3],rho[i,c2]*ux[i,c2],rho[i,c1]*ux[i,c1],mx0,rho[i,e1]*ux[i,e1],rho[i,e2]*ux[i,e2],rho[i,e3]*ux[i,e3])
        hmy= _d6(rho[a3,j]*uy[a3,j],rho[a2,j]*uy[a2,j],rho[a1,j]*uy[a1,j],my0,rho[b1,j]*uy[b1,j],rho[b2,j]*uy[b2,j],rho[b3,j]*uy[b3,j]) +
             _d6(rho[i,c3]*uy[i,c3],rho[i,c2]*uy[i,c2],rho[i,c1]*uy[i,c1],my0,rho[i,e1]*uy[i,e1],rho[i,e2]*uy[i,e2],rho[i,e3]*uy[i,e3])
        # PRESSURE gate (shared by molecular ν + density SVV): 1 in smooth/shear/wake, →0 at true shocks.
        gate=1.0f0
        if eps_rho > 0.0f0 || nu_mol > 0.0f0 || wm_on == Int32(1)
            pc=r0*cs2*Temp[i,j]; pim=rho[a1,j]*cs2*Temp[a1,j]; pip=rho[b1,j]*cs2*Temp[b1,j]
            pjm=rho[i,c1]*cs2*Temp[i,c1]; pjp=rho[i,e1]*cs2*Temp[i,e1]
            sx=abs(pip-2f0*pc+pim)/(abs(pip)+2f0*abs(pc)+abs(pim)+1f-6); sy=abs(pjp-2f0*pc+pjm)/(abs(pjp)+2f0*abs(pc)+abs(pjm)+1f-6)
            gate=clamp(1f0-12f0*max(sx,sy),0f0,1f0)
        end
        # MOLECULAR VISCOSITY: pressure-gated ∇²(ρu) momentum diffusion — sets the PHYSICAL Reynolds number in the
        # wake/shear (Re=U·D/ν, target Ma·1e4 per [[re-ma-consistency-les]]) so the wake is at its intended
        # turbulence level, NOT the ~150k effective Re of SGS-only. Gate spares the bow shock (captured inviscidly).
        # nu_mol carries dt (caller passes dt·ν). This is the DUGKS-molecular-ν half of the WMLES recipe (LES was
        # the other half); WITHOUT it the solver had no real viscosity ⇒ the wake ran far too turbulent.
        lmx=0.0f0; lmy=0.0f0
        nue = nu_mol + (wm_on==Int32(1) ? wm_scale*nuw[i,j] : 0.0f0)   # molecular ν + wall-model eddy ν(d) (direct ∇², RANS-style)
        if nue > 0.0f0
            lmx=(rho[a1,j]*ux[a1,j]+rho[b1,j]*ux[b1,j]+rho[i,c1]*ux[i,c1]+rho[i,e1]*ux[i,e1]) - 4.0f0*mx0
            lmy=(rho[a1,j]*uy[a1,j]+rho[b1,j]*uy[b1,j]+rho[i,c1]*uy[i,c1]+rho[i,e1]*uy[i,e1]) - 4.0f0*my0
        end
        # DENSITY k⁶ hyperdiffusion, pressure-gated (damps shear-layer ρ speckle in schlieren; spares the shock).
        # Computed FIRST — AUDIT FIX (2026-07-07): the momentum was divided by the OLD r0 while rho_o got the
        # UPDATED rn, silently rescaling the stored momentum rho_o·ux_o by rn/r0 (violating the conservation
        # claim whenever svv_rho>0, the active RAE2822 config). Divide by rn instead.
        rn = r0
        if eps_rho > 0.0f0
            hr= _d6(rho[a3,j],rho[a2,j],rho[a1,j],r0,rho[b1,j],rho[b2,j],rho[b3,j]) +
                _d6(rho[i,c3],rho[i,c2],rho[i,c1],r0,rho[i,e1],rho[i,e2],rho[i,e3])
            rn=r0+eps_rho*gate*hr; rn = rn < 0.3f0*r0 ? 0.3f0*r0 : rn   # floor: startup-transient safety (ρ→0 NaNs the divide)
        end
        rho_o[i,j]=rn
        ux_o[i,j]=(mx0+eps*hmx+nue*gate*lmx)/rn; uy_o[i,j]=(my0+eps*hmy+nue*gate*lmy)/rn
    end
end

# ---- SSP-RK3 time integration (WENO path) --------------------------------------------------------------
# A high-order spatial reconstruction (WENO5-Z) is LINEARLY UNSTABLE under single-step forward Euler — it
# amplifies grid-scale energy (verified: 8800× 2Δ-energy blow-up on a smooth resolved wave at CFL~0.5).
# The Shu-Osher 3rd-order Strong-Stability-Preserving Runge-Kutta restores stability at normal CFL while
# preserving WENO's low dissipation (104% peak retention at CFL~0.12; with RK3, at full CFL). The RK stage
# BLENDS must be done on the CONSERVED variables (ρ,ρu,ρv,ρE) — blending primitives (ρ,u,T) is NOT
# conservative. The rect kernel already outputs the forward-Euler stage result K(u)=u+dt·L(u) in primitives;
# these helpers snapshot U^n→conserved and combine a·U^n + b·conserved(K(u_stage)) → primitives.
@kernel function _prim2cons2T!(r_o, mx_o, my_o, E_o, @Const(rho), @Const(ux), @Const(uy), @Const(T), eint::Float32)
    i,j=@index(Global,NTuple)
    @inbounds begin
        r=rho[i,j]; u=ux[i,j]; v=uy[i,j]
        r_o[i,j]=r; mx_o[i,j]=r*u; my_o[i,j]=r*v; E_o[i,j]=r*(eint*T[i,j]+0.5f0*(u*u+v*v))
    end
end
# U_out(prim) = a·U^n(conserved snapshot rk_*) + b·conserved(raw prim stage); write back to (rho,ux,uy,T).
@kernel function _rkcombine2T!(rho, ux, uy, T, @Const(rkr), @Const(rkmx), @Const(rkmy), @Const(rkE),
                               @Const(rr), @Const(ru), @Const(rv), @Const(rT), a::Float32, b::Float32, eint::Float32)
    i,j=@index(Global,NTuple)
    @inbounds begin
        r2=rr[i,j]; u2=ru[i,j]; v2=rv[i,j]                                   # raw stage primitives
        cr = a*rkr[i,j]  + b*r2
        cmx= a*rkmx[i,j] + b*(r2*u2)
        cmy= a*rkmy[i,j] + b*(r2*v2)
        cE = a*rkE[i,j]  + b*(r2*(eint*rT[i,j]+0.5f0*(u2*u2+v2*v2)))
        r = cr < 1.0f-3 ? 1.0f-3 : cr
        ux2=cmx/r; uy2=cmy/r; sp=sqrt(ux2*ux2+uy2*uy2)
        if sp>3.5f0; f=3.5f0/sp; ux2*=f; uy2*=f; end
        T2=(cE/r - 0.5f0*(ux2*ux2+uy2*uy2))/eint; T2 = T2 < 1.0f-3 ? 1.0f-3 : T2
        rho[i,j]=r; ux[i,j]=ux2; uy[i,j]=uy2; T[i,j]=T2
    end
end

# ---- Roadmap 2.2: SDF ghost-cell curved wall (removes the immersed-boundary staircase) -------------------
# Volume penalisation snaps the wall to whole cells ⇒ a stair-stepped body. Instead, mirror the fluid state
# across the TRUE surface (the signed-distance isosurface) using the SDF normal: for a near-surface SOLID cell
# at depth d=|φ|, sample the fluid at the IMAGE point (cell + 2d·n̂, mirrored into the fluid) and set the ghost
# to the no-slip reflection (ρ,T copied; u ← −u_image ⇒ u=0 at the wall midpoint). The flux then sees a smooth
# curved wall at the φ=0 isosurface, 2nd-order, no staircase. Runs BEFORE the flux; χ must be 0 (no penalisation
# and no χ wall-flux) so the ghosts alone impose the wall.
@inline function _bilin2f(A,x,y,nx,ny)
    xc = x<1f0 ? 1f0 : (x>Float32(nx) ? Float32(nx) : x); yc = y<1f0 ? 1f0 : (y>Float32(ny) ? Float32(ny) : y)
    i0=floor(Int,xc);j0=floor(Int,yc);i1=min(i0+1,nx);j1=min(j0+1,ny);fx=xc-i0;fy=yc-j0
    @inbounds (1f0-fx)*(1f0-fy)*A[i0,j0]+fx*(1f0-fy)*A[i1,j0]+(1f0-fx)*fy*A[i0,j1]+fx*fy*A[i1,j1]
end
# FLUID-ONLY bilinear (mask cells with sdf≤0 out of the stencil + renormalize) ⇒ the image-point sample
# never reads solid/ghost data (the gold-standard fix vs plain bilinear). Falls back to nearest if all-solid.
@inline function _bilin2f_fluid(A,sdfm,x,y,nx,ny)
    xc = x<1f0 ? 1f0 : (x>Float32(nx) ? Float32(nx) : x); yc = y<1f0 ? 1f0 : (y>Float32(ny) ? Float32(ny) : y)
    i0=floor(Int,xc);j0=floor(Int,yc);i1=min(i0+1,nx);j1=min(j0+1,ny);fx=xc-i0;fy=yc-j0
    @inbounds begin
        m00 = sdfm[i0,j0]>0f0 ? 1f0 : 0f0; m10 = sdfm[i1,j0]>0f0 ? 1f0 : 0f0
        m01 = sdfm[i0,j1]>0f0 ? 1f0 : 0f0; m11 = sdfm[i1,j1]>0f0 ? 1f0 : 0f0
        w00=(1f0-fx)*(1f0-fy)*m00; w10=fx*(1f0-fy)*m10; w01=(1f0-fx)*fy*m01; w11=fx*fy*m11
        wsum=w00+w10+w01+w11
        wsum>1f-6 ? (w00*A[i0,j0]+w10*A[i1,j0]+w01*A[i0,j1]+w11*A[i1,j1])/wsum :
                    (m00>0f0 ? A[i0,j0] : (m10>0f0 ? A[i1,j0] : (m01>0f0 ? A[i0,j1] : A[i1,j1])))
    end
end
# GOLD-STANDARD 2nd-order image-point ghost-cell wall. For each near-surface SOLID cell (−band<φ≤0),
# mirror the FLUID state at the image point x_I = x + 2|φ|n̂ (fluid-only interp) and apply the wall BC:
# ρ,T copied (∂/∂n=0, adiabatic + ∂p/∂n≈0); velocity reflected — SLIP flips only the wall-NORMAL comp
# (impermeable, tangential free ⇒ inviscid/Euler loads), NO-SLIP flips the whole vector (u=0 at wall).
# The flux stencil then sees a smooth curved wall at φ=0 (no staircase) ⇒ clean surface Cp for loads.
@kernel function _ghost_wall2T!(rho,ux,uy,T, @Const(sdf),@Const(wnx),@Const(wny),@Const(lam), band::Float32, slip::Int32, NX::Int32,NY::Int32)
    i,j=@index(Global,NTuple); nx=Int(NX);ny=Int(NY)
    @inbounds begin
        φ=sdf[i,j]
        if φ<=0f0 && φ>-band && lam[i,j]<1f-4    # ghost band — DEEP-SOLID cells only (Λ≈0): a stencil fill for the
                                                 # reconstruction. CUT cells (Λ>0) are part of the fluid domain and
                                                 # ACCUMULATE via the cut-cell update — NEVER reset them (that would
                                                 # discard their mass ⇒ the body would act as a sink, not a wall).
            n1=wnx[i,j]; n2=wny[i,j]; d=-φ
            ix=Float32(i)+2f0*d*n1; iy=Float32(j)+2f0*d*n2
            ρI=_bilin2f_fluid(rho,sdf,ix,iy,nx,ny); TI=_bilin2f_fluid(T,sdf,ix,iy,nx,ny)
            uI=_bilin2f_fluid(ux,sdf,ix,iy,nx,ny); vI=_bilin2f_fluid(uy,sdf,ix,iy,nx,ny)
            # CONSTANT EXTRAPOLATION (copy the mirror-point fluid), NOT reflection: a smooth continuation for the
            # reconstruction stencil. A velocity REFLECTION (slip/no-slip mirror) is the textbook inviscid-wall ghost
            # for a Riemann solver, but through the KFVS/gauge kinetic flux the reflected normal velocity feeds a
            # stagnation HOT SPOT (T→20). The cut-cell wall FLUX imposes impermeability separately, so the fill only
            # needs to be smooth, not a mirror. (slip param retained for signature/future.)
            rho[i,j]=ρI; T[i,j]=(TI<1f-3 ? 1f-3 : TI); ux[i,j]=uI; uy[i,j]=vI
        elseif φ<=-band                          # deep interior ⇒ inert (quiescent), isolated from the flow
            ux[i,j]=0f0; uy[i,j]=0f0; rho[i,j]=1f0; T[i,j]=1f0
        end
    end
end

# ---- CUT-CELL embedded-boundary geometry (gold-standard body-conforming wall) --------------------------------
# From the cell-centred SDF (φ>0 fluid), compute per cell: the fluid VOLUME fraction Λ∈[0,1] and the four face
# APERTURES αE,αW,αN,αS∈[0,1] (fraction of each face open to fluid). Corner φ = bilinear of the 4 cell centres;
# edge apertures = exact linear φ=0 crossing on each edge; Λ = sub-sampled fluid area of the bilinear φ. General
# (works for cylinder, airfoil, any SDF). The wall segment A·n̂ is NOT stored — it follows EXACTLY from the discrete
# divergence identity A·n̂ = (−(αE−αW), −(αN−αS)) in the kernel, guaranteeing ∮n̂ dl=0 (discrete conservation).
function pond_cutcell_geometry_2d(sdf::AbstractMatrix{<:Real}, NX::Int, NY::Int; nsub::Int=8)
    Λ=zeros(Float32,NX,NY); αE=zeros(Float32,NX,NY); αW=zeros(Float32,NX,NY); αN=zeros(Float32,NX,NY); αS=zeros(Float32,NX,NY)
    cc(i,j)=Float32(sdf[clamp(i,1,NX),clamp(j,1,NY)])
    corner(i,j)=0.25f0*(cc(i,j)+cc(i+1,j)+cc(i,j+1)+cc(i+1,j+1))          # φ at continuous corner (i+½, j+½)
    ef(a,b)= (a>=0f0 && b>=0f0) ? 1f0 : ((a<0f0 && b<0f0) ? 0f0 : (a>=0f0 ? a/(a-b) : b/(b-a)))  # fluid fraction of an edge
    @inbounds for j in 1:NY, i in 1:NX
        φSW=corner(i-1,j-1); φSE=corner(i,j-1); φNW=corner(i-1,j); φNE=corner(i,j)
        if φSW>0f0 && φSE>0f0 && φNW>0f0 && φNE>0f0
            Λ[i,j]=1f0; αE[i,j]=1f0; αW[i,j]=1f0; αN[i,j]=1f0; αS[i,j]=1f0   # all-fluid
        elseif φSW<0f0 && φSE<0f0 && φNW<0f0 && φNE<0f0
            # all-solid ⇒ 0 (defaults)
        else                                                                # CUT cell
            αW[i,j]=ef(φSW,φNW); αE[i,j]=ef(φSE,φNE); αS[i,j]=ef(φSW,φSE); αN[i,j]=ef(φNW,φNE)
            cnt=0
            for jj in 0:nsub-1, ii in 0:nsub-1
                fx=(ii+0.5f0)/nsub; fy=(jj+0.5f0)/nsub
                φ=(1f0-fx)*(1f0-fy)*φSW+fx*(1f0-fy)*φSE+(1f0-fx)*fy*φNW+fx*fy*φNE
                φ>0f0 && (cnt+=1)
            end
            Λ[i,j]=Float32(cnt)/Float32(nsub*nsub)
        end
    end
    return Λ,αE,αW,αN,αS
end

mutable struct PonD2DT{A,V}
    backend
    rho::A; ux::A; uy::A; T::A; rhon::A; uxn::A; uyn::A; Tn::A; chi::A
    cxa::V; cya::V; cza::V; wa::V
    NX::Int; NY::Int; dx::Float32; dt::Float32; cs2::Float32
    rho_inf::Float32; U_inf::Float32; T_inf::Float32; eta::Float32
    sponge_w::Int; sponge_max::Float32; ng::Int; drot::Float32; csmag::Float32; gauge::Int32; wall::Int32; shock_aware::Int32; drp::Int32
    ff_mode::Int32   # y far-field: 0=hard-clamp freestream, 1=absorbing sponge, 2=NSCBC Riemann (roadmap 1.1)
    cd::A            # per-cell Germano-Lilly dynamic SGS coefficient (roadmap 1.2); 1×1 stub when off
    sgs_lm::A; sgs_mm::A            # instantaneous L:M, M:M scratch; 1×1 stubs when off
    sgs_ilm::A; sgs_imm::A          # Lagrangian-averaged I_LM, I_MM (persistent memory); ping-pong ↓
    sgs_ilmn::A; sgs_immn::A
    dyn_sgs::Int32   # 0 ⇒ static Cs²; 1 ⇒ dynamic (Lagrangian Germano-Lilly); needs csmag>0 (LES mode)
    svv_eps::Float32 # k⁶-selective SVV strength on momentum (configurable; default 0.004; stable < ~0.0078)
    weno::Int32      # 0 ⇒ minmod/MC reconstruction; 1 ⇒ WENO5-Z (low-dissipation, non-oscillatory)
    svv_passes::Int32 # # of SVV sub-passes/step (default 1). k⁶ is weak at 3-4Δ; N passes at a stable ε give
                      # N× cumulative grid-scale damping (each pass separately under the Nyquist limit) — kills the
                      # under-resolved shear-cone vorticity stipple while sparing the smooth shock (low-k) + resolved scales.
    svv_rho::Float32  # pressure-gated k⁶ DENSITY hyperdiffusion strength (default 0 = off). Damps the shear-layer
                      # density speckle (visible in schlieren |∇ρ| and, via the flux, vorticity) while sparing the shock.
    nu_mol::Float32   # dt·ν molecular viscosity (pressure-gated ∇² momentum diffusion). Sets the physical Reynolds
                      # number Re=U·D/ν in the wake (default 0 = off ⇒ SGS-only, effective Re ~150k = over-turbulent).
    rk_r::A; rk_mx::A; rk_my::A; rk_E::A   # SSP-RK3 conserved U^n snapshot (WENO path); 1×1 stubs when weno off
    sdf::A; wnx::A; wny::A                  # signed distance (cells; >0 fluid) + outward unit normal — ghost-cell reconstruction fill
    ghost_wall::Int32                       # 1 ⇒ cut-cell embedded-boundary wall active (replaces staircase penalization/wall-flux); 1×1 stubs when off
    wall_slip::Int32                        # 1 ⇒ slip (inviscid/Euler loads); 0 ⇒ no-slip
    lam::A; aE::A; aW::A; aN::A; aS::A       # CUT-CELL: fluid volume fraction Λ + face apertures (conservative embedded boundary)
    wm::A; wall_model::Int32                 # algebraic wall model: mixing-length ℓ² field (from SDF wall distance), ADDED to the SGS coeff cd each step ⇒ energizes the near-wall BL ⇒ delays laminar separation. Needs dyn_sgs=1.
    dc_r::A; dc_mx::A; dc_my::A; dc_e::A; frd::Int32   # V1 FLUX REDISTRIBUTION: conservative divergence Dc (per conserved var) stored by the flux kernel, then a hybrid-divergence pass D=Λ·Dc+(1−Λ)·Dnc fixes the cut-cell small-cell instability (replaces the Λ-floor). 1×1 stubs when off.
    inlet_res::Int32; inlet_rho0::Float32; inlet_T0::Float32   # subsonic reservoir (total-pressure) inlet: mass flux free to self-adjust (vs the over-specified fixed-ρUT inlet)
    nutilde::A; nutilde_n::A; wdist::A; sa::Int32; sa_nulat::Float32; sa_nuinf::Float32   # SPALART-ALLMARAS RANS: ν̃ transport field (+ping-pong scratch), wall distance d=|SDF| (cells), flag, molecular ν (lattice units), freestream ν̃=3ν. Writes the eddy viscosity ν_t=ν̃·f_v1 into `wm` each step ⇒ applied as ∇² momentum diffusion via the existing spatial-ν path.
    wmles::Int32; wm_hmatch::Float32   # WALL-MODELED LES: equilibrium log-law wall model — near-wall ν_t from the local friction velocity u_τ, confined (windowed) to d<wm_hmatch cells. Shares wdist + the `wm` spatial-ν path. Thinner than SA ⇒ less decambering on the coarse grid.
    nested::Int32; hrho::A; hux::A; huy::A; hT::A   # NESTED FINE-BOX AMR: when 1, this grid is a fine L2 box nested in a coarse base — applyBC! PINS the prolongation-filled ghost halo (h*) into the ng-border each RK stage instead of the domain inlet/outlet/farfield BCs. The coarse driver refreshes h* by prolongation once per coarse step. 1×1 stubs when off.
    ddes::Int32; ddes_cdes::Float32                 # DDES (Spalart 2006, SA-based): 1 ⇒ (a) SA runs with the DDES length-scale + f_d shielding (RANS attached BL, LES wake), AND (b) the eddy viscosity is applied through the PROPER deviatoric viscous-stress tensor `_eddy_stress2T!` (strain-consistent ⇒ no decamber) instead of the isotropic ∇²(ρu) Laplacian. cdes = C_DES (Δ=1 cell). Implies sa=1.
    dugks::Int32                                    # G1 gas-kinetic flux: 1 ⇒ the face flux uses the DUGKS interface distribution (collision-IN-flux, physical ν from τ=ν/cs², O(τ) numerical dissipation) blended with KFVS at shocks — un-smears the transonic pocket. The molecular-ν ∇² pass is then disabled (ν is carried by τ in the flux). 0 ⇒ pure KFVS (byte-identical).
    ddes_solo::Int32                                # 1 (with ddes=1) ⇒ zero the FLUX-level SGS (KFVS face term + DUGKS τ_eff Smagorinsky) so the SA-DDES tensor stress is the SOLE subgrid closure (canonical DDES, no double-count). 0 ⇒ legacy: static Smagorinsky in the flux + DDES stress simultaneously.
    dugks_nucal::Int32                              # 1 ⇒ G1(c) ν-calibration: solve the foot-trace quadratic for τ so the DELIVERED DUGKS viscosity equals the target ν (legacy τ=ν/cs² over-delivers ~2-3×; gate: tools/pond_dugks_nu1d.jl). 0 ⇒ legacy.
    ff_gam::Float32; ff_xv::Float32; ff_yv::Float32 # POINT-VORTEX far-field correction (this grid's cell units, clockwise-positive Γ at (xv,yv)); 0 ⇒ legacy clamped freestream. MUTABLE — the runner feeds Γ back from the computed Cl. Applied by inlet/farfield/sponge (coarsest level only — nested levels pin halos).
    ff_m2::Float32; ff_beta::Float32                # PRANDTL-GLAUERT compressible vortex (PV_PGV): m2=M∞², beta=√(1−M∞²). q_θ = Γ/(2πr)·β/(1−M∞²sin²θ). DEFAULT (0, 1) reproduces the incompressible _fvortex EXACTLY (byte-identical legacy gate). Driver sets these on L1 when PV_PGV=1 (2026-07-13 lead-hypothesis fix for the +5.5% Cl domain-height sensitivity S2 measured — the incompressible vortex under-weights transverse upwash by 1/β≈1.46× at M0.73).
    dm_r::A; dm_mx::A; dm_my::A; dm_e::A; frd_cg::Int32   # V2 COLELLA-GRAVES redistribution (G6): per-cut-cell hybrid defect δM (4 conserved vars) + flag. Restores cut-cell conservation to round-off (kills the LE ρ-spike the V1 hybrid leaks). 1×1 stubs when off.
    mg_tgt::Any; mg_links::Any; mg_nlk::Any; frd_merge::Int32   # V3 CELL MERGING (G6, Berger-Helzel linking): sliver→target map + reverse links (Int32 device arrays; Any-typed — concrete at kernel launch). Replaces hybrid+CG when 1: strictly conservative AND locally consistent.
    gtrip::A; trip_on::Int32                                    # PRESCRIBED-TRANSITION intermittency γ∈[0,1] (laminar ahead of the trip, turbulent aft) gating the wall-stress ν_t; 1×1 stub when off. Full γ-Reθ transport = SST/G3 milestone.
    tk::A; tkn::A; tom::A; tomn::A                              # SST k-ω (Menter 2003): k + ω transport fields (+ping-pong). 1×1 stubs when off.
    sst::Int32; sst_kinf::Float32; sst_winf::Float32            # SST flag + freestream (k∞,ω∞). ν_t·dt → `wm`, applied via the deviatoric tensor (like DDES). Replaces WMLES (priority over it).
    pgr::Int32                                                  # 1 ⇒ Larsson pressure-gradient correction in the log-law u_τ solve (WMLES ν_t + SST wall-function band)
    hk::A; hom::A                                               # NESTED+SST: prolongation-filled k/ω halo (REVIEW FIX 2026-07-09 — freestream-pinned borders laminarized a patch's mid-BL inflow). 1×1 stubs unless nested && sst.
    sst_halo::Int32                                             # 1 ⇒ pin borders from hk/hom (set by the TREE BUILDER only when the PARENT also runs SST — L4's parent L3 has none, so L4 keeps its legacy freestream/zero-grad borders ⇒ byte-identity)
    eddy_gate::Float32                                          # _eddy_stress2T! pressure-gate steepness (12=legacy; 0=UNGATED ⇒ SBLI-permitting — the λ-foot wall-pressure spread; PV_EDDYGATE A/B 2026-07-10)
    split::Int32                                                # PHASE 0 (implicit roadmap): 1 ⇒ residual!/update! split on the RECTANGULAR path — the flux kernel stores R=divF into dc_* and pond_fv2dT_rect_update! applies U−a·R (bitwise ≡ fused; gate tools/pond_split_gate.jl). Inert on the cut-cell path (frd already stores Dc and merge/hybrid IS the update). dc_* full-size when set.
    h_nom::Float32                                              # PHASE 0 step 3 (fixed-h DUGKS, panel BLOCKER): the DUGKS half-step h fed to the flux kernel AND the nucal τ quadratic. Default (h_flux=0 in the ctor) resolves to 0.5·dt/dx — bitwise ≡ the legacy in-kernel 0.5f0*a. The steady tier freezes it per level, decoupled from pseudo-dt.
    fas::Int32                                                  # FAS MG (2026-07-14): 1 ⇒ τ-corrected coarse smoothing on the rectangular sub-chain (PV_FAS); 0 ⇒ off (byte-identical). Requires split_residual (dc_* full-size).
    fas_tau_r::A; fas_tau_mx::A; fas_tau_my::A; fas_tau_e::A    # frozen FAS defect τ = R_H(I·U_h) − I·R_h(U_h) (CONSERVED rate; nonzero only on fluid-covered coarse cells). 1×1 stubs when off.
    fas_ub_r::A;  fas_ub_mx::A;  fas_ub_my::A;  fas_ub_e::A     # baseline Ū = I·U_h (CONSERVED) snapshot for the correction dU = U_H − Ū. 1×1 stubs when off.
end

# `gamma` sets the EOS: drot = 2/(γ−1) − 3 internal DOF beyond the lattice's 3 (γ=5/3 ⇒ drot=0;
# air γ=1.4 ⇒ drot=2). Default γ=5/3 is byte-identical to the monatomic scheme.
function pond_fv2dT_driver(backend, dev, NX, NY; dx, dt, cs2, rho_inf, U_inf, T_inf, chi_host,
                           eta=1f-3, sponge_w=0, sponge_max=0.5f0, ng=3, u_init_freestream=true,
                           gamma=5/3, cs_smag=0.0f0, gauge_theta=false, adiabatic_wall=false, shock_aware=false, drp=false,
                           farfield=:sponge, dynamic_sgs=false, svv_eps=0.004f0, svv_passes=1, svv_rho=0f0, nu_mol=0f0, weno=false,
                           wall_model=false, wm_band=20f0, wm_strength=1f0, flux_redist=false, flux_redist_cg=false, cutcell_merge=false, trip_host=nothing, spalart_allmaras=false,
                           wmles=false, wm_hmatch=12f0, ddes=false, ddes_cdes=0.65f0, dugks=false, ddes_solo=false, dugks_nucal=false,
                           sst=false, sst_tu=0.001f0, sst_nutr=0.1f0, pgr=false, eddy_gate=12f0, split_residual=false, h_flux=0f0, fas=false,
                           inlet_reservoir=false, inlet_rho0=1f0, inlet_T0=1f0, nested=false,
                           sdf_host=nothing, wnx_host=nothing, wny_host=nothing, ghost_wall=false, wall_slip=false, cxv, cyv, czv, wv)
    ux0 = u_init_freestream ? Float32(U_inf) : 0.0f0
    rho=dev(fill(Float32(rho_inf),NX,NY)); ux=dev(fill(ux0,NX,NY)); uy=dev(zeros(Float32,NX,NY)); T=dev(fill(Float32(T_inf),NX,NY))
    rhon=dev(zeros(Float32,NX,NY)); uxn=dev(zeros(Float32,NX,NY)); uyn=dev(zeros(Float32,NX,NY)); Tn=dev(zeros(Float32,NX,NY))
    chi=dev(chi_host); cxa=dev(Float32.(cxv)); cya=dev(Float32.(cyv)); cza=dev(Float32.(czv)); wa=dev(Float32.(wv))
    drot = Float32(2/(gamma-1) - 3)
    ffm = farfield === :nscbc ? Int32(2) : (farfield === :sponge ? Int32(1) : Int32(0))
    dyn = (dynamic_sgs && Float32(cs_smag) > 0.0f0) ? Int32(1) : Int32(0)   # dynamic needs LES mode (csmag>0)
    mkc()= dev(zeros(Float32, dyn==Int32(1) ? NX : 1, dyn==Int32(1) ? NY : 1))
    cd=mkc(); sgs_lm=mkc(); sgs_mm=mkc(); sgs_ilm=mkc(); sgs_imm=mkc(); sgs_ilmn=mkc(); sgs_immn=mkc()
    wf=Int32(weno ? 1 : 0)
    mkr()= dev(zeros(Float32, weno ? NX : 1, weno ? NY : 1))   # SSP-RK3 conserved snapshot (WENO path only)
    rk_r=mkr(); rk_mx=mkr(); rk_my=mkr(); rk_E=mkr()
    # AUDIT FIX (2026-07-07): ghost_wall without sdf_host used to build a full-size zero SDF with 1×1
    # lam/aperture stubs — every cell then satisfied the ghost-band test and lam[i,j]/aE[i,j] were read
    # OUT OF BOUNDS under @inbounds (silent GPU memory corruption). Degrade with a warning instead.
    (ghost_wall && sdf_host===nothing) && @warn "ghost_wall needs sdf_host (SDF) — IGNORED"
    gw=Int32((ghost_wall && sdf_host!==nothing) ? 1 : 0)
    (sst && sdf_host===nothing) && @warn "SST needs sdf_host (wall distance + solid mask) — IGNORED"
    ssto=Int32((sst && sdf_host!==nothing) ? 1 : 0)              # SST also needs the signed SDF on device (solid test after χ-off)
    need_sdf = gw==Int32(1) || ssto==Int32(1)
    sdf = need_sdf && sdf_host!==nothing ? dev(Float32.(sdf_host)) : dev(zeros(Float32, need_sdf ? NX : 1, need_sdf ? NY : 1))
    wnx = gw==Int32(1) && wnx_host!==nothing ? dev(Float32.(wnx_host)) : dev(zeros(Float32, gw==Int32(1) ? NX : 1, gw==Int32(1) ? NY : 1))
    wny = gw==Int32(1) && wny_host!==nothing ? dev(Float32.(wny_host)) : dev(zeros(Float32, gw==Int32(1) ? NX : 1, gw==Int32(1) ? NY : 1))
    if gw==Int32(1) && sdf_host!==nothing        # CUT-CELL embedded-boundary geometry from the SDF
        Λh,aEh,aWh,aNh,aSh = pond_cutcell_geometry_2d(Float32.(sdf_host), Int(NX), Int(NY))
        lam=dev(Λh); aE=dev(aEh); aW=dev(aWh); aN=dev(aNh); aS=dev(aSh)
        ncut=count(x->0f0<x<1f0, Λh); minlam=minimum(x for x in Λh if x>0f0)
        println("    CUT-CELL geometry: ", ncut, " cut cells, min Λ=", round(minlam,digits=4))
    else
        lam=dev(ones(Float32,1,1)); aE=dev(ones(Float32,1,1)); aW=dev(ones(Float32,1,1)); aN=dev(ones(Float32,1,1)); aS=dev(ones(Float32,1,1))
    end
    # ALGEBRAIC WALL MODEL: mixing-length ℓ(d)=κ·d·(1−e^{−d/A})·e^{−(d/band)²} from the SDF wall distance d=|φ|
    # (Van Driest-damped inner layer, Gaussian-windowed to the BL band so it never touches the pocket/freestream).
    # ℓ² is ADDED to the SGS coeff cd each step ⇒ turbulent-like near-wall momentum transport ⇒ fuller BL profile
    # ⇒ resists the adverse-gradient (laminar) separation aft of max thickness. Needs dyn_sgs=1 (piggybacks on cd).
    # ALGEBRAIC WALL MODEL — a wall-distance eddy viscosity ν_wall(d) applied as a DIRECT ∇²(ρu) diffusion (the
    # molecular-ν path), NOT a Smagorinsky |S|-scaled term. Direct diffusion fills the near-wall velocity profile
    # (turbulent-like, fuller) ⇒ resists the adverse-gradient laminar separation, WITHOUT the shear-scaled form's
    # pathology (which dumped viscosity at the wall/pocket and decambered the suction side). Profile:
    # ν_wall(d) = ν_peak·(1−e^{−d/A})·e^{−(d/band)²} — Van Driest-damped, Gaussian-windowed to the BL band so it
    # never touches the pocket/freestream. ν_peak = wm_strength (caller passes dt·ν_wall = dt·U·c/Re_wall).
    dds = Int32((ddes && sdf_host!==nothing) ? 1 : 0)              # DDES ⇒ SA transport + tensor-stress application
    sa = Int32(((spalart_allmaras || ddes) && sdf_host!==nothing) ? 1 : 0)   # SA (ddes implies SA); writes ν_t·dt into `wm`
    wml = Int32((wmles && sdf_host!==nothing) ? 1 : 0)             # WMLES log-law wall model (also writes `wm`)
    ((spalart_allmaras||wmles) && sdf_host===nothing) && @warn "SA/WMLES need sdf_host (wall distance) — IGNORED"
    (wall_model && (sa==Int32(1)||wml==Int32(1))) && @warn "static wall_model ν_wall(d) profile is OVERWRITTEN by SA/WMLES ν_t every step — combination not supported (pick one)"
    (wml==Int32(1) && sa==Int32(1)) && @warn "WMLES is IGNORED when SA/DDES is active (SA has priority in advance!)"
    (ssto==Int32(1) && sa==Int32(1)) && @warn "SST + SA/DDES both active — SA has priority in advance!; SST IGNORED"
    if ssto==Int32(1) && wml==Int32(1)
        @warn "WMLES is REPLACED by SST (SST wall-function band ⊇ the log-law ν_t) — WMLES disabled"
        wml=Int32(0)
    end
    wmf = Int32((wall_model || sa==Int32(1) || wml==Int32(1)) ? 1 : 0)   # _svvT applies `wm` when static wall model OR SA OR WMLES (SST applies via the tensor, not here)
    if wall_model && sdf_host!==nothing
        Ad=2.0f0; wmh=zeros(Float32,NX,NY)
        @inbounds for j in 1:NY, i in 1:NX
            dphi=Float32(sdf_host[i,j])
            dphi>0f0 && (wmh[i,j]=Float32(wm_strength)*(1f0-exp(-dphi/Ad))*exp(-(dphi/Float32(wm_band))^2))
        end
        wm=dev(wmh); println("    WALL MODEL: ν_wall(d) direct-∇² (band=", wm_band, " cells, dt·ν_peak=", round(Float64(wm_strength),sigdigits=3), "), max=", round(maximum(wmh),digits=5))
    elseif sa==Int32(1) || wml==Int32(1) || ssto==Int32(1)
        wm=dev(zeros(Float32,NX,NY))                               # SA/WMLES/SST fill ν_t·dt here each step
    else
        wm=dev(zeros(Float32,1,1))
    end
    # SPALART–ALLMARAS / WMLES / SST fields: ν̃ (init to freestream), scratch, wall distance d=|SDF| (cells). ν in lattice units.
    sa_nulat = Float32(nu_mol > 0f0 ? nu_mol/dt : 1f-4); sa_nuinf = Float32(3.0*sa_nulat)
    if sa==Int32(1) || wml==Int32(1) || ssto==Int32(1)
        wdist=dev(Float32.(abs.(sdf_host)))
        nutilde = sa==Int32(1) ? dev(fill(sa_nuinf,NX,NY)) : dev(zeros(Float32,1,1)); nutilde_n = sa==Int32(1) ? dev(zeros(Float32,NX,NY)) : dev(zeros(Float32,1,1))
        sa==Int32(1) && println("    ", dds==Int32(1) ? "DDES (SA-based): DDES length-scale + f_d shielding + TENSOR-stress application (no decamber)" : "SPALART-ALLMARAS", " ON (ν_lat=", round(Float64(sa_nulat),sigdigits=3), ", ν̃∞=", round(Float64(sa_nuinf),sigdigits=3), dds==Int32(1) ? string(", C_DES=",round(Float64(ddes_cdes),sigdigits=3)) : "", ")")
        wml==Int32(1) && println("    WMLES: equilibrium log-law wall model ON (hmatch=", wm_hmatch, " cells, ν_lat=", round(Float64(sa_nulat),sigdigits=3), pgr ? ", Larsson PG" : "", ")")
    else
        wdist=dev(zeros(Float32,1,1)); nutilde=dev(zeros(Float32,1,1)); nutilde_n=dev(zeros(Float32,1,1))
    end
    # SST k-ω (Menter 2003): k,ω fields init to freestream — k∞=1.5(Tu·U)², ω∞=k∞/(ν_ratio·ν) (ν_t∞ = ν_ratio·ν)
    sst_kinf = Float32(1.5*(Float64(sst_tu)*abs(Float64(U_inf)))^2); sst_kinf = max(sst_kinf, 1f-10)
    sst_winf = Float32(sst_kinf/(max(Float64(sst_nutr),1e-3)*Float64(sa_nulat))); sst_winf = max(sst_winf, 1f-4)
    if ssto==Int32(1) && sa==Int32(1); ssto=Int32(0); end          # SA priority (warned above)
    if ssto==Int32(1)
        tk=dev(fill(sst_kinf,NX,NY)); tkn=dev(zeros(Float32,NX,NY))
        tom=dev(fill(sst_winf,NX,NY)); tomn=dev(zeros(Float32,NX,NY))
        println("    SST k-ω (Menter 2003): wall-function band d<", _sst_band(), " Pk=", _sst_kato() ? "Kato-Launder(S·Ω)" : "S²", " (PG-aware u_τ: ", pgr ? "ON" : "off",
                "), trip-gated P_k, ν_t via tensor stress; k∞=", round(Float64(sst_kinf),sigdigits=3),
                " ω∞=", round(Float64(sst_winf),sigdigits=3), " ν_lat=", round(Float64(sa_nulat),sigdigits=3))
    else
        tk=dev(zeros(Float32,1,1)); tkn=dev(zeros(Float32,1,1)); tom=dev(zeros(Float32,1,1)); tomn=dev(zeros(Float32,1,1))
    end
    # V1 FLUX REDISTRIBUTION storage — only with the cut-cell wall (gw) active.
    # PHASE 0: the residual!/update! split reuses dc_* as the rate storage on the RECT path, so the
    # arrays are also full-size when split_residual is requested (rect mode has no Λ/frd).
    frd = Int32((flux_redist && gw==Int32(1)) ? 1 : 0)
    spl = Int32(split_residual ? 1 : 0)
    (spl==Int32(1) && gw==Int32(1) && frd!=Int32(1)) && @warn "split_residual on the cut-cell LEGACY Λ-floor path (ghost_wall without flux_redist) is INERT — enable flux_redist"
    mkfrd()= dev(zeros(Float32, (frd==Int32(1)||spl==Int32(1)) ? NX : 1, (frd==Int32(1)||spl==Int32(1)) ? NY : 1))
    dc_r=mkfrd(); dc_mx=mkfrd(); dc_my=mkfrd(); dc_e=mkfrd()
    frd==Int32(1) && println("    FLUX REDISTRIBUTION: hybrid-divergence cut-cell stabilization ON (replaces Λ-floor)")
    spl==Int32(1) && println("    PHASE-0 SPLIT: residual!/update! split ON (rect path: R stored in dc_*, trivial update kernel)")
    # FAS MG (2026-07-14): τ + baseline-Ū arrays — full-size only when fas && split (residual needed for τ).
    fasf = Int32((fas && spl==Int32(1)) ? 1 : 0)
    (fas && spl!=Int32(1)) && @warn "fas needs split_residual (dc_* full-size for R(U)) — IGNORED"
    mkfas() = dev(zeros(Float32, fasf==Int32(1) ? NX : 1, fasf==Int32(1) ? NY : 1))
    fas_tau_r=mkfas(); fas_tau_mx=mkfas(); fas_tau_my=mkfas(); fas_tau_e=mkfas()
    fas_ub_r =mkfas(); fas_ub_mx =mkfas(); fas_ub_my =mkfas(); fas_ub_e =mkfas()
    fasf==Int32(1) && println("    FAS MG: τ-corrected coarse smoothing ON (rectangular sub-chain)")
    # PHASE 0 step 3: resolve the DUGKS half-step. (0.5f0*dt)/dx ≡ 0.5f0*(dt/dx) bitwise (exact halving),
    # so the default reproduces the legacy in-kernel derivation exactly.
    hnom = Float32(h_flux) > 0f0 ? Float32(h_flux) : 0.5f0*Float32(dt)/Float32(dx)
    Float32(h_flux) > 0f0 && println("    PHASE-0 FIXED-h DUGKS: h_nom=", hnom, " FROZEN (decoupled from update dt=", Float32(dt), ")")
    (flux_redist && gw!=Int32(1)) && @warn "flux_redist needs the cut-cell wall (ghost_wall=true) — IGNORED"
    # V2 COLELLA-GRAVES defect redistribution (G6) — requires the V1 hybrid pass
    fcg = Int32((flux_redist_cg && frd==Int32(1)) ? 1 : 0)
    mkcg()= dev(zeros(Float32, fcg==Int32(1) ? NX : 1, fcg==Int32(1) ? NY : 1))
    dm_r=mkcg(); dm_mx=mkcg(); dm_my=mkcg(); dm_e=mkcg()
    fcg==Int32(1) && println("    COLELLA-GRAVES: cut-cell defect redistribution ON (conservative to round-off)")
    (flux_redist_cg && frd!=Int32(1)) && @warn "flux_redist_cg needs flux_redist + ghost_wall — IGNORED"
    # V3 CELL MERGING (G6, Berger-Helzel linking) — requires the cut-cell wall + Dc storage (frd)
    fmg = Int32((cutcell_merge && frd==Int32(1)) ? 1 : 0)
    if fmg==Int32(1)
        tgt_h,links_h,nlk_h,nm,nover,norph = pond_cutcell_merge_map(Λh,aEh,aWh,aNh,aSh)  # from the cut-cell geometry block
        mg_tgt=dev(tgt_h); mg_links=dev(links_h); mg_nlk=dev(nlk_h)
        println("    CELL MERGING: ", nm, " slivers linked (Λ<0.6, aperture-connected)",
                norph>0 ? "; $norph ORPHANS on Λ-floor fallback" : "", nover>0 ? " [$nover link-overflow]" : "")
    else
        mg_tgt=dev(zeros(Int32,1,1)); mg_links=dev(zeros(Int32,1,1,1)); mg_nlk=dev(zeros(Int32,1,1))
    end
    (cutcell_merge && frd!=Int32(1)) && @warn "cutcell_merge needs ghost_wall + flux_redist — IGNORED"
    # PRESCRIBED-TRANSITION intermittency field (gates the wall-stress ν_t; γ-Reθ transport = SST/G3 slot)
    tro = Int32(trip_host!==nothing ? 1 : 0)
    gtrip = tro==Int32(1) ? dev(Float32.(trip_host)) : dev(zeros(Float32,1,1))
    tro==Int32(1) && println("    TRIP: prescribed-transition intermittency ON (laminar ahead of the trip station)")
    # NESTED fine-box: the ng-border halo buffers pinned each RK stage (filled by the coarse driver's prolongation)
    nst = Int32(nested ? 1 : 0)
    mkh()= dev(zeros(Float32, nst==Int32(1) ? NX : 1, nst==Int32(1) ? NY : 1))
    hrho=mkh(); hux=mkh(); huy=mkh(); hT=mkh()
    nst==Int32(1) && println("    NESTED fine-box: ng=", ng, " ghost halo pinned by prolongation (domain BCs OFF)")
    # NESTED+SST k/ω halo (REVIEW FIX 2026-07-09): filled by the tree step when the PARENT also runs SST;
    # the SST kernel's border branch then pins these instead of freestream (which laminarized a TE patch's
    # mid-BL inflow, k∞~7e-7 vs BL k~1e-2). Init to freestream so a non-SST parent degrades gracefully.
    hk  = (nst==Int32(1) && ssto==Int32(1)) ? dev(fill(sst_kinf,NX,NY)) : dev(zeros(Float32,1,1))
    hom = (nst==Int32(1) && ssto==Int32(1)) ? dev(fill(sst_winf,NX,NY)) : dev(zeros(Float32,1,1))
    return PonD2DT{typeof(rho),typeof(cxa)}(backend, rho,ux,uy,T, rhon,uxn,uyn,Tn, chi, cxa,cya,cza,wa,
        NX,NY, Float32(dx),Float32(dt),Float32(cs2), Float32(rho_inf),Float32(U_inf),Float32(T_inf),Float32(eta),
        Int(sponge_w), Float32(sponge_max), ng, drot, Float32(cs_smag), Int32(gauge_theta ? 1 : 0), Int32(adiabatic_wall ? 1 : 0), Int32(shock_aware ? 1 : 0), Int32(drp ? 1 : 0), ffm, cd, sgs_lm, sgs_mm, sgs_ilm, sgs_imm, sgs_ilmn, sgs_immn, dyn, Float32(svv_eps), wf, Int32(max(1,svv_passes)), Float32(svv_rho), Float32(nu_mol), rk_r,rk_mx,rk_my,rk_E,
        sdf,wnx,wny, gw, Int32(wall_slip ? 1 : 0), lam,aE,aW,aN,aS, wm, wmf, dc_r,dc_mx,dc_my,dc_e, frd,
        Int32(inlet_reservoir ? 1 : 0), Float32(inlet_rho0), Float32(inlet_T0),
        nutilde, nutilde_n, wdist, sa, sa_nulat, sa_nuinf, wml, Float32(wm_hmatch),
        nst, hrho, hux, huy, hT, dds, Float32(ddes_cdes), Int32(dugks ? 1 : 0), Int32(ddes_solo ? 1 : 0), Int32(dugks_nucal ? 1 : 0),
        0.0f0, 0.0f0, 0.0f0, 0.0f0, 1.0f0, dm_r, dm_mx, dm_my, dm_e, fcg, mg_tgt, mg_links, mg_nlk, fmg, gtrip, tro,
        tk, tkn, tom, tomn, ssto, sst_kinf, sst_winf, Int32(pgr ? 1 : 0), hk, hom, Int32(0), Float32(eddy_gate), spl, hnom,
        fasf, fas_tau_r,fas_tau_mx,fas_tau_my,fas_tau_e, fas_ub_r,fas_ub_mx,fas_ub_my,fas_ub_e)
end

# TIER-2 SYNC ELISION (2026-07-14, memory/project_docs/2026-07-14_BRUTE_SPEED_program.md): on a single CUDABackend from a
# single task, ALL kernels launch on ONE CUDA stream and execute in-order — the per-kernel device syncs in
# the steady tier are never REQUIRED for correctness (host readbacks — cnorm's mapreduce, Array() copies —
# are self-synchronizing in CUDA.jl). Each sync'd launch costs 50-100 µs on Windows WDDM (flushes the
# command buffer, kills CPU enqueue-ahead). PV_NOSYNC=1 elides them (bit-identical result REQUIRED — same
# stream, same launch order); default 0 = the legacy per-kernel sync (also the debug mode: kernel faults
# surface at the offending launch instead of the next readback). VALID ONLY under single-task/single-stream
# launch order — if per-level streams/tasks are ever introduced, PV_NOSYNC must revert to 0.
const _NOSYNC_REF = Ref(-1)
@inline function _psync(b)
    if _NOSYNC_REF[] < 0
        _NOSYNC_REF[] = get(ENV, "PV_NOSYNC", "0") == "1" ? 1 : 0
    end
    _NOSYNC_REF[] == 1 ? nothing : KernelAbstractions.synchronize(b)
end

# PHASE 0 shared derivations — advance! and pond_fv2dT_residual! MUST compute these identically
# (review finding: duplicated derivations silently de-synchronize R(U) from the stepped operator;
# a Newton on a drifted R converges to a subtly wrong steady state with no gate to catch it).
_pond2T_wallp(d::PonD2DT)   = d.ghost_wall==Int32(1) ? Int32(0) : d.wall   # ghost-cell wall ⇒ staircase kinetic wall OFF
_pond2T_fluxsgs(d::PonD2DT) = (d.ddes==Int32(1) && d.ddes_solo==Int32(1)) ? Int32(0) : Int32(1)   # DDES-solo: SA-DDES is the sole SGS
function _pond2T_ghost!(d::PonD2DT, nd)    # image-point ghost fill (gold-standard curved wall) — internal sub-step of R
    d.ghost_wall==Int32(1) || return
    _ghost_wall2T!(d.backend)(d.rho,d.ux,d.uy,d.T, d.sdf,d.wnx,d.wny,d.lam, 4.0f0, d.wall_slip, Int32(d.NX),Int32(d.NY); ndrange=nd)
    _psync(d.backend)
end

# G1: DUGKS τ (the flux carries the molecular ν). PHASE 0: extracted from advance! so the residual
# operator computes the SAME τ (same expressions, same order ⇒ byte-identical).
function _pond2T_dtau(d::PonD2DT)
    dtau = 0.0f0
    if d.dugks==Int32(1)
        nu = d.nu_mol/d.dt                                   # ν in lattice units (nu_mol carries dt·ν)
        if d.dugks_nucal==Int32(1) && nu > 0f0
            # G1(c) ν-CALIBRATION (gate `tools/pond_dugks_nu1d.jl`, 2026-07-07): the foot-trace adds O(h)
            # viscosity on top of cs²τ — delivered ν_eff ≈ cs²(τ + 2τh/(2τ+h)) (measured 1.94× target at
            # τ/h=0.1; naive τ=ν/cs² over-delivers up to 3× for τ≪h ⇒ effective Re up to 3× LOW). Solve
            # 2cs²τ² + (3cs²h − 2ν)τ − νh = 0 for τ so the DELIVERED viscosity hits the target ν.
            # PHASE 0 step 3: h is the SAME frozen h_nom the flux kernel uses (default ≡ 0.5·dt/dx bitwise).
            h = d.h_nom
            b = 3f0*d.cs2*h - 2f0*nu
            dtau = (-b + sqrt(b*b + 8f0*d.cs2*nu*h))/(4f0*d.cs2)
        else
            dtau = nu/d.cs2                                  # legacy: τ = ν/cs² (over-delivers, see above)
        end
    end
    return dtau
end

function pond_fv2dT_advance!(d::PonD2DT, t::Int)
    nd=(d.NX,d.NY)
    dtau = _pond2T_dtau(d)
    if d.dyn_sgs == Int32(1)     # roadmap 1.2: Germano-Lilly c_d field (α²=8, 3×3 top-hat) — 2 passes before flux
        # AUDIT FIX (2026-07-07): α² 4→8. The equal-weight 3×3 box filter has per-axis variance ⅔Δ² ⇒
        # moment-matched test-filter width Δ̂²=12·(⅔)=8Δ² (Lund). α²=4 under-weighted the M test term ~2×,
        # inflating c_d ~2.3× (was masked by the sign inversion above).
        _dyn_lmmm2T!(d.backend)(d.sgs_lm, d.sgs_mm, d.ux, d.uy, 8.0f0, Int32(d.NX),Int32(d.NY); ndrange=nd)
        KernelAbstractions.synchronize(d.backend)
        _dyn_lagr2T!(d.backend)(d.sgs_ilmn, d.sgs_immn, d.cd, d.sgs_ilm, d.sgs_imm, d.sgs_lm, d.sgs_mm,
            d.ux, d.uy, Float32(d.dt/d.dx), d.dt, 1.5f0, Int32(d.NX),Int32(d.NY); ndrange=nd)
        KernelAbstractions.synchronize(d.backend)
        d.sgs_ilm,d.sgs_ilmn = d.sgs_ilmn,d.sgs_ilm; d.sgs_imm,d.sgs_immn = d.sgs_immn,d.sgs_imm   # ping-pong memory
    end
    # gold-standard image-point ghost-cell wall: refresh the ghost (near-surface solid) cells from the
    # CURRENT fluid state so the reconstruction stencil sees a smooth curved wall (band=4 ≥ WENO ±3 stencil).
    ghost!() = _pond2T_ghost!(d, nd)
    wallp = _pond2T_wallp(d)
    # forward-Euler spatial operator K(u)=u+dt·L(u): reads (rho..), writes (rhon..). No swap. Ghosts refreshed first.
    flux_sgs = _pond2T_fluxsgs(d)
    # guard (review finding): a runtime-mutated split flag must never point the kernel's split-store
    # at the 1×1 dc stubs (@inbounds ⇒ silent memory corruption). Construct with split_residual=true.
    spl_eff = (d.split==Int32(1) && size(d.dc_r,1)==d.NX && size(d.dc_r,2)==d.NY) ? Int32(1) : Int32(0)
    function flux!()
        ghost!()
        pond_fv2dT_rect_kernel!(d.backend)(d.rhon,d.uxn,d.uyn,d.Tn, d.rho,d.ux,d.uy,d.T, d.chi, d.cxa,d.cya,d.cza,d.wa,
            d.cs2, d.dt/d.dx, d.drot, d.csmag, d.gauge, wallp, d.shock_aware, d.drp, d.cd, d.dyn_sgs, d.weno, d.sdf, d.ghost_wall, d.lam,d.aE,d.aW,d.aN,d.aS, d.dc_r,d.dc_mx,d.dc_my,d.dc_e, d.frd, d.dugks, dtau, d.h_nom, flux_sgs, spl_eff, Int32(d.NX),Int32(d.NY); ndrange=nd)
        KernelAbstractions.synchronize(d.backend)
        if spl_eff == Int32(1) && d.ghost_wall != Int32(1)
            # PHASE 0: rect-path residual!/update! split — the flux kernel stored R=divF in dc_*;
            # apply the trivial update U^{n+1}=U^n−a·R (bitwise ≡ the fused else-branch).
            pond_fv2dT_rect_update!(d.backend)(d.rhon,d.uxn,d.uyn,d.Tn, d.rho,d.ux,d.uy,d.T,
                d.dc_r,d.dc_mx,d.dc_my,d.dc_e, d.cs2, d.dt/d.dx, d.drot; ndrange=nd)
            KernelAbstractions.synchronize(d.backend)
        end
        if d.frd == Int32(1)
            if d.frd_merge == Int32(1)   # V3 (G6): CELL MERGING — strictly conservative + locally consistent (replaces hybrid/CG)
                _frd_merge2T!(d.backend)(d.rhon,d.uxn,d.uyn,d.Tn, d.rho,d.ux,d.uy,d.T, d.lam,
                    d.dc_r,d.dc_mx,d.dc_my,d.dc_e, d.mg_tgt,d.mg_links,d.mg_nlk, d.cs2, d.dt/d.dx, d.drot, Int32(d.NX),Int32(d.NY); ndrange=nd)
                KernelAbstractions.synchronize(d.backend)
            else                          # V1: hybrid-divergence pass overwrites cut+fluid cells with U^n − a·D_hybrid
                _frd_hybrid2T!(d.backend)(d.rhon,d.uxn,d.uyn,d.Tn, d.rho,d.ux,d.uy,d.T, d.lam, d.dc_r,d.dc_mx,d.dc_my,d.dc_e, d.cs2, d.dt/d.dx, d.drot, Int32(d.NX),Int32(d.NY); ndrange=nd)
                KernelAbstractions.synchronize(d.backend)
                if d.frd_cg == Int32(1)   # V2 (G6): Colella-Graves defect redistribution — conservation to round-off (UNSTABLE at stagnation slivers — see docs; superseded by merging)
                    _frd_defect2T!(d.backend)(d.dm_r,d.dm_mx,d.dm_my,d.dm_e, d.lam, d.dc_r,d.dc_mx,d.dc_my,d.dc_e, d.dt/d.dx, Int32(d.NX),Int32(d.NY); ndrange=nd)
                    KernelAbstractions.synchronize(d.backend)
                    _frd_cg2T!(d.backend)(d.rhon,d.uxn,d.uyn,d.Tn, d.lam, d.dm_r,d.dm_mx,d.dm_my,d.dm_e, d.cs2, d.drot, Int32(d.NX),Int32(d.NY); ndrange=nd)
                    KernelAbstractions.synchronize(d.backend)
                end
            end
        end
    end
    function swapmain!()
        d.rho,d.rhon=d.rhon,d.rho; d.ux,d.uxn=d.uxn,d.ux; d.uy,d.uyn=d.uyn,d.uy; d.T,d.Tn=d.Tn,d.T
    end
    function applyBC!()          # Dirichlet inlet + y far-field + transmissive outlet on (rho..)
        if d.nested == Int32(1)  # NESTED fine box: the prolongation halo IS the BC — pin it, skip domain BCs
            _pin_halo2T!(d.backend)(d.rho,d.ux,d.uy,d.T, d.hrho,d.hux,d.huy,d.hT, Int32(d.NX),Int32(d.NY), Int32(d.ng); ndrange=nd)
            KernelAbstractions.synchronize(d.backend); return
        end
        if d.inlet_res == Int32(1)
            _inlet_res2T_x!(d.backend)(d.rho,d.ux,d.uy,d.T, d.inlet_rho0, d.inlet_T0, d.cs2, d.drot, Int32(d.ng); ndrange=nd)
        else
            _inlet2T_x!(d.backend)(d.rho,d.ux,d.uy,d.T, d.rho_inf, d.U_inf, d.T_inf, Int32(d.ng), d.ff_gam, d.ff_xv, d.ff_yv, d.ff_m2, d.ff_beta; ndrange=nd)
        end
        KernelAbstractions.synchronize(d.backend)
        if d.ff_mode == Int32(2)
            Deff = 3.0f0 + d.drot; gam = (Deff+2.0f0)/Deff
            _nscbc2T_y!(d.backend)(d.rho,d.ux,d.uy,d.T, d.rho_inf, d.U_inf, d.T_inf, gam, d.cs2, Int32(d.NY), Int32(d.ng); ndrange=nd)
        elseif d.ff_mode == Int32(1) && d.sponge_w>0
            _sponge2T_y!(d.backend)(d.rho,d.ux,d.uy,d.T, d.rho_inf, d.U_inf, d.T_inf, Int32(d.NY), Int32(d.sponge_w), d.sponge_max, d.ff_gam, d.ff_xv, d.ff_yv, d.ff_m2, d.ff_beta; ndrange=nd)
        else
            _farfield2T_y!(d.backend)(d.rho,d.ux,d.uy,d.T, d.rho_inf, d.U_inf, d.T_inf, Int32(d.NY), Int32(d.ng), d.ff_gam, d.ff_xv, d.ff_yv, d.ff_m2, d.ff_beta; ndrange=nd)
        end
        KernelAbstractions.synchronize(d.backend)
        _outlet2T_x!(d.backend)(d.rho,d.ux,d.uy,d.T, Int32(d.NX), Int32(d.ng); ndrange=nd); KernelAbstractions.synchronize(d.backend)
    end
    if d.weno == Int32(1)
        # SSP-RK3 (Shu-Osher): high-order WENO5-Z is forward-Euler-UNSTABLE (amplifies grid-scale energy);
        # RK3 restores stability at full CFL while keeping WENO's low dissipation. Blends on CONSERVED vars.
        eint = 0.5f0*(3.0f0+d.drot)*d.cs2
        _prim2cons2T!(d.backend)(d.rk_r,d.rk_mx,d.rk_my,d.rk_E, d.rho,d.ux,d.uy,d.T, eint; ndrange=nd); KernelAbstractions.synchronize(d.backend)
        flux!(); swapmain!(); applyBC!()                              # stage 1: u1 = K(u^n)
        flux!()                                                       # stage 2: raw = K(u1) → (rhon..)
        _rkcombine2T!(d.backend)(d.rho,d.ux,d.uy,d.T, d.rk_r,d.rk_mx,d.rk_my,d.rk_E, d.rhon,d.uxn,d.uyn,d.Tn, 0.75f0,0.25f0, eint; ndrange=nd)
        KernelAbstractions.synchronize(d.backend); applyBC!()         # u2 = ¾u^n + ¼raw
        flux!()                                                       # stage 3: raw = K(u2) → (rhon..)
        _rkcombine2T!(d.backend)(d.rho,d.ux,d.uy,d.T, d.rk_r,d.rk_mx,d.rk_my,d.rk_E, d.rhon,d.uxn,d.uyn,d.Tn, 0.33333334f0,0.6666667f0, eint; ndrange=nd)
        KernelAbstractions.synchronize(d.backend)                     # u^{n+1} = ⅓u^n + ⅔raw
    else
        flux!(); swapmain!()                                          # minmod/MC: single-step forward Euler
    end
    svv_ran = (d.csmag > 0.0f0 || d.weno == Int32(1)) && (d.svv_eps > 0.0f0 || d.svv_rho > 0.0f0 || d.nu_mol > 0.0f0 || d.wall_model == Int32(1))
    if svv_ran   # SVV/molecular-ν/wall-model post pass
        ghost!()                                                       # refresh solid ghosts so the SVV's near-wall neighbour reads are clean (not RK garbage)
        numol_pass = (d.dugks==Int32(1) ? 0.0f0 : d.nu_mol)/Float32(d.svv_passes)   # DUGKS carries molecular ν via τ in the flux ⇒ no ∇² here (avoid double-count)
        wm_scale = 1.0f0/Float32(d.svv_passes)                         # wall-model ν also applied ÷passes (N passes ⇒ once-equivalent)
        wm_on_svv = (d.ddes==Int32(1) || d.sst==Int32(1)) ? Int32(0) : d.wall_model   # DDES/SST apply the eddy ν via the TENSOR kernel below, not the ∇² Laplacian here
        for _ in 1:d.svv_passes                                        # N passes ⇒ N× cumulative grid-scale damping (each pass Nyquist-stable)
            _svvT_kernel!(d.backend)(d.rhon,d.uxn,d.uyn, d.rho,d.ux,d.uy,d.T, d.svv_eps, d.svv_rho, numol_pass, d.wm, wm_scale, wm_on_svv, d.cs2, Int32(d.NX),Int32(d.NY); ndrange=nd)
            KernelAbstractions.synchronize(d.backend)
            d.rho,d.rhon=d.rhon,d.rho; d.ux,d.uxn=d.uxn,d.ux; d.uy,d.uyn=d.uyn,d.uy   # T unchanged by SVV (read-only for the pressure gate) ⇒ no swap
        end
    elseif d.wall_model == Int32(1) && d.ddes != Int32(1) && d.sst != Int32(1)
        # AUDIT FIX (2026-07-07): SA/WMLES/static-wall-model ν_t was silently never APPLIED when
        # csmag==0 && weno==0 (the model kernels kept transporting ν̃ and the banners printed). Apply
        # the wm ∇² pass alone (eps=eps_rho=nu_mol=0) so the closure acts on every configuration.
        ghost!()
        _svvT_kernel!(d.backend)(d.rhon,d.uxn,d.uyn, d.rho,d.ux,d.uy,d.T, 0.0f0, 0.0f0, 0.0f0, d.wm, 1.0f0, d.wall_model, d.cs2, Int32(d.NX),Int32(d.NY); ndrange=nd)
        KernelAbstractions.synchronize(d.backend)
        d.rho,d.rhon=d.rhon,d.rho; d.ux,d.uxn=d.uxn,d.ux; d.uy,d.uyn=d.uyn,d.uy
    end
    if d.ddes == Int32(1) || d.sst == Int32(1)   # DDES/SST: apply the eddy viscosity through the PROPER deviatoric viscous stress (strain-consistent ⇒ no decamber).
        # AUDIT FIX (2026-07-07): was nested inside the SVV gate ⇒ silently skipped when csmag==0 && weno==0.
        svv_ran || ghost!()
        _eddy_stress2T!(d.backend)(d.uxn,d.uyn, d.rho,d.ux,d.uy,d.T, d.wm, d.cs2, d.eddy_gate, Int32(d.NX),Int32(d.NY); ndrange=nd)
        KernelAbstractions.synchronize(d.backend)
        d.ux,d.uxn=d.uxn,d.ux; d.uy,d.uyn=d.uyn,d.uy                # ρ,T unchanged by the eddy stress ⇒ no swap
    end
    _penalize2!(d.backend)(d.ux,d.uy, d.chi, Float32(d.dt/d.eta); ndrange=nd); KernelAbstractions.synchronize(d.backend)  # volume penalization (Brinkman) — the primary blocker; cut-cell wall flux refines the surface pressure
    ghost!()                    # leave the solid ghosts consistent for the next step's first flux + for output/rendering
    applyBC!()
    if d.sa == Int32(1)         # SPALART-ALLMARAS / DDES: transport ν̃ with the updated velocity; write ν_t·dt into wm for the next step
        _sa_kernel!(d.backend)(d.nutilde_n, d.wm, d.nutilde, d.ux,d.uy, d.wdist, d.chi, d.sa_nulat, d.dt, d.ddes, d.ddes_cdes, Int32(d.NX),Int32(d.NY); ndrange=nd)
        KernelAbstractions.synchronize(d.backend)
        d.nutilde, d.nutilde_n = d.nutilde_n, d.nutilde
    elseif d.sst == Int32(1)    # SST k-ω: transport k,ω with the updated state; write ν_t·dt into wm for the next step
        _sst_kernel!(d.backend)(d.tkn, d.tomn, d.wm, d.tk, d.tom, d.ux,d.uy, d.rho,d.T, d.wdist, d.chi, d.sdf,
            d.gtrip, d.trip_on, d.sa_nulat, d.dt, d.cs2, d.sst_kinf, d.sst_winf, abs(d.U_inf), d.pgr, Int32(d.ng),
            d.hk, d.hom, d.sst_halo, _sst_band(), Int32(_sst_kato() ? 1 : 0), Int32(d.NX),Int32(d.NY); ndrange=nd)
        KernelAbstractions.synchronize(d.backend)
        d.tk, d.tkn = d.tkn, d.tk; d.tom, d.tomn = d.tomn, d.tom
    elseif d.wmles == Int32(1)  # WMLES: algebraic log-law near-wall eddy viscosity → wm for the next step (trip-gated, PG-aware)
        _wmles_kernel!(d.backend)(d.wm, d.ux,d.uy, d.rho,d.T, d.wdist, d.chi, d.sa_nulat, d.dt, d.wm_hmatch, d.gtrip, d.trip_on, d.cs2, d.pgr, Int32(d.NX),Int32(d.NY); ndrange=nd)
        KernelAbstractions.synchronize(d.backend)
    end
    return nothing
end

# PHASE 0 (implicit roadmap step 1, 2026-07-10): R(U) as a callable GPU operator — fills d.dc_* with
# the 4-component RATE of the flux-divergence core of the operator (dt factored out; dx=1):
#   rect grids  (ghost_wall off):     R = divF            [construct with split_residual=true]
#   cut-cell    (ghost_wall + frd):   R = Dc = divF/Λ     [the existing frd storage]
# For MERGED sliver groups (frd_merge) the steady-tier group rate is R_g = ΣΛ·Dc/ΣΛ, identical for all
# members (see pond_fv2dT_rect_update! header + memory/project_docs/2026-07-10_PHASE0_residual_split.md).
# SCOPE (honest): this is the FLUX core only. Per-step passes NOT yet in R — SVV (dt-independent
# per-pass fraction!), eddy-stress (ν_t·dt inside wm), Brinkman penalization (dt/eta), SST transport,
# BC/halo pins — the Phase-0 rate-ization audit (roadmap step 2) re-expresses or excludes each.
# ★ R(U) is NOT a pure function of U on dyn-SGS grids: the face-SGS term reads d.cd, which advance!
# refreshes from U^n BEFORE its flux and this function does NOT — d.cd enters as FROZEN (one-step-
# lagged) input state. Likewise ng-border/sponge cells carry pinned-BC junk rates. Any ‖R‖ diagnostic
# must freeze/refresh cd deliberately and mask the borders, or it will not be reproducible/zero.
# Refreshes solid ghosts first (ghost fill = internal sub-step of R, per the roadmap).
# Side effect: d.rhon/uxn/uyn/Tn are scratch (kernel passthrough) after this call.
function pond_fv2dT_residual!(d::PonD2DT)
    nd=(d.NX,d.NY)
    (size(d.dc_r,1)==d.NX && size(d.dc_r,2)==d.NY) ||
        error("pond_fv2dT_residual!: dc_* are 1×1 stubs — construct with split_residual=true (rect) or flux_redist+ghost_wall (cut-cell)")
    (d.ghost_wall==Int32(1) && d.frd!=Int32(1)) &&
        error("pond_fv2dT_residual!: cut-cell legacy Λ-floor path has no stored rate — enable flux_redist")
    dtau = _pond2T_dtau(d)
    _pond2T_ghost!(d, nd)
    wallp = _pond2T_wallp(d)
    flux_sgs = _pond2T_fluxsgs(d)
    pond_fv2dT_rect_kernel!(d.backend)(d.rhon,d.uxn,d.uyn,d.Tn, d.rho,d.ux,d.uy,d.T, d.chi, d.cxa,d.cya,d.cza,d.wa,
        d.cs2, d.dt/d.dx, d.drot, d.csmag, d.gauge, wallp, d.shock_aware, d.drp, d.cd, d.dyn_sgs, d.weno, d.sdf, d.ghost_wall, d.lam,d.aE,d.aW,d.aN,d.aS, d.dc_r,d.dc_mx,d.dc_my,d.dc_e, d.frd, d.dugks, dtau, d.h_nom, flux_sgs, Int32(1), Int32(d.NX),Int32(d.NY); ndrange=nd)
    _psync(d.backend)
    return nothing
end
