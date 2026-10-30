# =====================================================================================================
# PHASE 1 (implicit roadmap, 2026-07-11) — the STEADY TIER: local-time-stepping pseudo-time driver.
#
# Pseudo-time march to the steady state: every cell advances at its LOCAL stability limit
# (dtf[i,j] = CFL/(3·√T + |u|), dx=1 — the D3Q39 max particle speed with the gauge) instead of the
# shared physical dt (which runs L1 at CFL≈dt_fine/S≈0.06 — the coarse levels are where LTS wins).
# The DUGKS flux keeps its FROZEN h_nom (Phase-0 step 3, panel BLOCKER): the discrete solution the
# steady state converges to is the fixed-h operator's, decoupled from the pseudo-step.
#
# RATE-CONSISTENCY (Phase-0 step-2 audit, implemented here): the per-step passes that are NOT rates
# (SVV per-pass fraction, eddy-stress force carrying ν_t·dt_ref in wm, Brinkman multiplicative
# penalization) are scaled by dt_local/dt_ref so the composed map's FIXED POINT matches the shared-dt
# operator's — without the scaling, LTS would converge to a (slightly) different answer per cell-size.
# SST k/ω stays on the uniform dt_ref pseudo-clock (its own fixed point is dt-independent; the
# loosely-coupled implicit treatment is the NEXT Phase-1 item). dyn-SGS runs as in advance! (its
# freeze-vs-off policy decision is flagged in the Phase-0 doc).
#
# The steady advance mirrors advance!'s SSP-RK3 structure (WENO5-Z is linearly unstable under forward
# Euler — the RK3 requirement carries over to pseudo-time; LTS inside RK stages is standard practice).
# The explicit production path is UNTOUCHED: everything here is new siblings, opt-in via PV_STEADY.
# =====================================================================================================

# ---- per-cell pseudo-dt field ------------------------------------------------------------------------
# PV_SSTPI gate (engineering block A): 1 (default) = steady tier uses the LTS/point-implicit SST sibling;
# 0 = legacy uniform-clock explicit kernel. Read once per process (kernel choice must not flip mid-run).
const _SST_PI_REF = Ref{Int}(-1)
function _sst_pi_on()
    if _SST_PI_REF[] < 0
        _SST_PI_REF[] = get(ENV, "PV_SSTPI", "1") == "0" ? 0 : 1
    end
    return _SST_PI_REF[] == 1
end

@kernel function _dtfield2T!(dtf, @Const(ux), @Const(uy), @Const(T), cfl::Float32)
    i,j=@index(Global,NTuple)
    @inbounds begin
        sp=sqrt(ux[i,j]*ux[i,j]+uy[i,j]*uy[i,j])
        c =3.0f0*sqrt(max(T[i,j],1.0f-3))            # max discrete speed 3·√θ (gauge-scaled set)
        dtf[i,j]=cfl/max(c+sp, 3.0f0)                # REVIEW L4: floor at the unshifted set speed (gauge-off safety)
    end
end

# ---- LTS rect update: U^{n+1} = U^n − a(i,j)·R (the split-path trivial update, per-cell dt) ------------
@kernel function pond_fv2dT_rect_update_lts!(rho_o, ux_o, uy_o, T_o, @Const(rho), @Const(ux), @Const(uy), @Const(Temp),
                                             @Const(dc_r), @Const(dc_mx), @Const(dc_my), @Const(dc_e),
                                             @Const(a_arr), cs2::Float32, drot::Float32)
    i,j=@index(Global,NTuple); eint=0.5f0*(3.0f0+drot)*cs2
    @inbounds begin
        a=a_arr[i,j]
        r0=rho[i,j]; u0x=ux[i,j]; u0y=uy[i,j]
        Ec = r0*(eint*Temp[i,j] + 0.5f0*(u0x*u0x+u0y*u0y))
        r  = r0     - a*dc_r[i,j]
        mx = r0*u0x - a*dc_mx[i,j]
        my = r0*u0y - a*dc_my[i,j]
        e  = Ec     - a*dc_e[i,j]
        r = r < 1.0f-3 ? 1.0f-3 : r
        ux2=mx/r; uy2=my/r; sp=sqrt(ux2*ux2+uy2*uy2)
        if sp>3.5f0; f=3.5f0/sp; ux2*=f; uy2*=f; end
        T2 = (e/r - 0.5f0*(ux2*ux2+uy2*uy2))/eint; T2 = T2 < 1.0f-3 ? 1.0f-3 : T2
        rho_o[i,j]=r; ux_o[i,j]=ux2; uy_o[i,j]=uy2; T_o[i,j]=T2
    end
end

# ---- FAS MG (2026-07-14): τ-source subtract — the smoother relaxes R_H(U)=τ by seeing dc ← dc−τ --------
# Runs AFTER the flux kernel stores R=divF into dc_* and BEFORE either LTS update kernel, so BOTH update
# paths (rect + merge) see the τ-corrected residual without modification. τ is ZERO everywhere except
# fluid-covered coarse cells (written masked by _fas_tau_finalize!), so the full-grid subtract is
# self-masking: borders/solid/leaf cells subtract exact 0.0f0 (an FP32 no-op). Gated on d.fas==1 at the
# call site ⇒ never launches in legacy runs. At the FAS fixed point dc_*≡τ ⇒ subtract yields 0 ⇒ the
# update moves nothing (the fixed-point-invariance requirement).
@kernel function _fas_axpy_sub!(dc_r, dc_mx, dc_my, dc_e,
        @Const(tr), @Const(tmx), @Const(tmy), @Const(te))
    i,j=@index(Global,NTuple)
    @inbounds begin
        dc_r[i,j]  -= tr[i,j];  dc_mx[i,j] -= tmx[i,j]
        dc_my[i,j] -= tmy[i,j]; dc_e[i,j]  -= te[i,j]
    end
end

# ---- LTS merge update: the V3 group update with the GROUP's dt = the HOST cell's dt --------------------
# (roadmap Phase 1: "group pseudo-dt keyed off the merge map" — one dt per merged control volume.)
@kernel function _frd_merge2T_lts!(rho_o,ux_o,uy_o,T_o, @Const(rho),@Const(ux),@Const(uy),@Const(Temp),
        @Const(lam), @Const(dc_r),@Const(dc_mx),@Const(dc_my),@Const(dc_e),
        @Const(tgt), @Const(links), @Const(nlk), @Const(a_arr),
        cs2::Float32, drot::Float32, NX::Int32, NY::Int32)
    i,j=@index(Global,NTuple); nx=Int(NX); eint=0.5f0*(3.0f0+drot)*cs2
    @inbounds begin
        Λc=lam[i,j]
        if Λc>1f-6 && tgt[i,j]==Int32(0)
            a=a_arr[i,j]                                       # host's dt = the group's dt
            r0=rho[i,j]; u0=ux[i,j]; v0=uy[i,j]
            Ec=r0*(eint*Temp[i,j]+0.5f0*(u0*u0+v0*v0))
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
            for k in 1:nl
                li=Int(links[i,j,k]); ii=((li-1)%nx)+1; jj=((li-1)÷nx)+1
                rho_o[ii,jj]=r; ux_o[ii,jj]=u2; uy_o[ii,jj]=v2; T_o[ii,jj]=T2
            end
        end
    end
end

# ---- LTS SVV/molecular-ν: per-pass strengths scaled by dt_local/dt_ref (rate consistency) --------------
@kernel function _svvT_lts!(rho_o, ux_o, uy_o, @Const(rho), @Const(ux), @Const(uy), @Const(Temp),
                            eps::Float32, eps_rho::Float32, nu_mol::Float32, @Const(nuw), wm_scale::Float32,
                            wm_on::Int32, @Const(a_arr), inv_dtref::Float32, cs2::Float32, NX::Int32, NY::Int32)
    i, j = @index(Global, NTuple); nx=Int(NX); ny=Int(NY)
    @inbounds begin
        rr=a_arr[i,j]*inv_dtref                                # dt_local/dt_ref
        # rate-consistent scaling CAPPED at the k⁶ per-pass stability limit (ε ≲ 0.0078 Nyquist —
        # struct doc): on coarse levels dt_local/dt_ref ≈ S ≈ 6-8 and uncapped ε·rr EXPLODES in
        # O(100) iters (first-launch NaN, 2026-07-11). The cap under-delivers SVV rate on coarse
        # levels relative to the shared-dt operator — a documented fixed-point deviation, bounded by
        # the Phase-1 gate's explicit-reference comparison (SVV is grid-scale-selective; the far
        # field it under-smooths is smooth).
        epsl=min(eps*rr, 7.5f-3); eps_rhol=min(eps_rho*rr, 7.5f-3)
        a1=_clampi(i-1,nx);a2=_clampi(i-2,nx);a3=_clampi(i-3,nx);b1=_clampi(i+1,nx);b2=_clampi(i+2,nx);b3=_clampi(i+3,nx)
        c1=_clampi(j-1,ny);c2=_clampi(j-2,ny);c3=_clampi(j-3,ny);e1=_clampi(j+1,ny);e2=_clampi(j+2,ny);e3=_clampi(j+3,ny)
        r0=rho[i,j]; mx0=r0*ux[i,j]; my0=r0*uy[i,j]
        hmx= _d6(rho[a3,j]*ux[a3,j],rho[a2,j]*ux[a2,j],rho[a1,j]*ux[a1,j],mx0,rho[b1,j]*ux[b1,j],rho[b2,j]*ux[b2,j],rho[b3,j]*ux[b3,j]) +
             _d6(rho[i,c3]*ux[i,c3],rho[i,c2]*ux[i,c2],rho[i,c1]*ux[i,c1],mx0,rho[i,e1]*ux[i,e1],rho[i,e2]*ux[i,e2],rho[i,e3]*ux[i,e3])
        hmy= _d6(rho[a3,j]*uy[a3,j],rho[a2,j]*uy[a2,j],rho[a1,j]*uy[a1,j],my0,rho[b1,j]*uy[b1,j],rho[b2,j]*uy[b2,j],rho[b3,j]*uy[b3,j]) +
             _d6(rho[i,c3]*uy[i,c3],rho[i,c2]*uy[i,c2],rho[i,c1]*uy[i,c1],my0,rho[i,e1]*uy[i,e1],rho[i,e2]*uy[i,e2],rho[i,e3]*uy[i,e3])
        gate=1.0f0
        if eps_rhol > 0.0f0 || nu_mol > 0.0f0 || wm_on == Int32(1)
            pc=r0*cs2*Temp[i,j]; pim=rho[a1,j]*cs2*Temp[a1,j]; pip=rho[b1,j]*cs2*Temp[b1,j]
            pjm=rho[i,c1]*cs2*Temp[i,c1]; pjp=rho[i,e1]*cs2*Temp[i,e1]
            sx=abs(pip-2f0*pc+pim)/(abs(pip)+2f0*abs(pc)+abs(pim)+1f-6); sy=abs(pjp-2f0*pc+pjm)/(abs(pjp)+2f0*abs(pc)+abs(pjm)+1f-6)
            gate=clamp(1f0-12f0*max(sx,sy),0f0,1f0)
        end
        lmx=0.0f0; lmy=0.0f0
        nue = min((nu_mol + (wm_on==Int32(1) ? wm_scale*nuw[i,j] : 0.0f0))*rr, 0.2f0)   # REVIEW L3: explicit-diffusion cap
        if nue > 0.0f0
            lmx=(rho[a1,j]*ux[a1,j]+rho[b1,j]*ux[b1,j]+rho[i,c1]*ux[i,c1]+rho[i,e1]*ux[i,e1]) - 4.0f0*mx0
            lmy=(rho[a1,j]*uy[a1,j]+rho[b1,j]*uy[b1,j]+rho[i,c1]*uy[i,c1]+rho[i,e1]*uy[i,e1]) - 4.0f0*my0
        end
        rn = r0
        if eps_rhol > 0.0f0
            hr= _d6(rho[a3,j],rho[a2,j],rho[a1,j],r0,rho[b1,j],rho[b2,j],rho[b3,j]) +
                _d6(rho[i,c3],rho[i,c2],rho[i,c1],r0,rho[i,e1],rho[i,e2],rho[i,e3])
            rn=r0+eps_rhol*gate*hr; rn = rn < 0.3f0*r0 ? 0.3f0*r0 : rn
        end
        rho_o[i,j]=rn
        ux_o[i,j]=(mx0+epsl*hmx+nue*gate*lmx)/rn; uy_o[i,j]=(my0+epsl*hmy+nue*gate*lmy)/rn
    end
end

# ---- LTS eddy stress: force scaled by dt_local/dt_ref (wm carries ν_t·dt_ref) --------------------------
@kernel function _eddy_stress2T_lts!(ux_o, uy_o, @Const(rho),@Const(ux),@Const(uy),@Const(Temp),@Const(nuw),
                                     @Const(a_arr), inv_dtref::Float32, cs2::Float32, gate_k::Float32, NX::Int32, NY::Int32)
    i,j=@index(Global,NTuple); nx=Int(NX); ny=Int(NY)
    @inbounds begin
        iL=_clampi(i-1,nx); iR=_clampi(i+1,nx); jD=_clampi(j-1,ny); jU=_clampi(j+1,ny)
        m0=rho[i,j]*nuw[i,j]
        mE=0.5f0*(m0+rho[iR,j]*nuw[iR,j]); mW=0.5f0*(m0+rho[iL,j]*nuw[iL,j])
        mN=0.5f0*(m0+rho[i,jU]*nuw[i,jU]); mS=0.5f0*(m0+rho[i,jD]*nuw[i,jD])
        dudx_E=ux[iR,j]-ux[i,j]; dvdx_E=uy[iR,j]-uy[i,j]
        dudy_E=0.25f0*(ux[i,jU]+ux[iR,jU]-ux[i,jD]-ux[iR,jD]); dvdy_E=0.25f0*(uy[i,jU]+uy[iR,jU]-uy[i,jD]-uy[iR,jD])
        dudx_W=ux[i,j]-ux[iL,j]; dvdx_W=uy[i,j]-uy[iL,j]
        dudy_W=0.25f0*(ux[iL,jU]+ux[i,jU]-ux[iL,jD]-ux[i,jD]); dvdy_W=0.25f0*(uy[iL,jU]+uy[i,jU]-uy[iL,jD]-uy[i,jD])
        dudy_N=ux[i,jU]-ux[i,j]; dvdy_N=uy[i,jU]-uy[i,j]
        dudx_N=0.25f0*(ux[iR,j]+ux[iR,jU]-ux[iL,j]-ux[iL,jU]); dvdx_N=0.25f0*(uy[iR,j]+uy[iR,jU]-uy[iL,j]-uy[iL,jU])
        dudy_S=ux[i,j]-ux[i,jD]; dvdy_S=uy[i,j]-uy[i,jD]
        dudx_S=0.25f0*(ux[iR,jD]+ux[iR,j]-ux[iL,jD]-ux[iL,j]); dvdx_S=0.25f0*(uy[iR,jD]+uy[iR,j]-uy[iL,jD]-uy[iL,j])
        c23=2.0f0/3.0f0; c43=4.0f0/3.0f0
        τxx_E=mE*(c43*dudx_E-c23*dvdy_E); τxx_W=mW*(c43*dudx_W-c23*dvdy_W)
        τxy_N=mN*(dudy_N+dvdx_N);         τxy_S=mS*(dudy_S+dvdx_S)
        τxy_E=mE*(dudy_E+dvdx_E);         τxy_W=mW*(dudy_W+dvdx_W)
        τyy_N=mN*(c43*dvdy_N-c23*dudx_N); τyy_S=mS*(c43*dvdy_S-c23*dudx_S)
        Fx=(τxx_E-τxx_W)+(τxy_N-τxy_S); Fy=(τxy_E-τxy_W)+(τyy_N-τyy_S)
        gate=1f0
        if gate_k > 0f0
            pc=rho[i,j]*cs2*Temp[i,j]; pim=rho[iL,j]*cs2*Temp[iL,j]; pip=rho[iR,j]*cs2*Temp[iR,j]
            pjm=rho[i,jD]*cs2*Temp[i,jD]; pjp=rho[i,jU]*cs2*Temp[i,jU]
            sx=abs(pip-2f0*pc+pim)/(abs(pip)+2f0*abs(pc)+abs(pim)+1f-6); sy=abs(pjp-2f0*pc+pjm)/(abs(pjp)+2f0*abs(pc)+abs(pjm)+1f-6)
            gate=clamp(1f0-gate_k*max(sx,sy),0f0,1f0)
        end
        rrl=a_arr[i,j]*inv_dtref
        m0d=rho[i,j]*nuw[i,j]                                  # REVIEW L3: cap the scaled diffusion number nu_t*dt_loc
        gl=gate*(m0d*rrl>0.2f0 ? 0.2f0/max(m0d,1f-9) : rrl)
        r0=rho[i,j]
        ux_o[i,j]=(r0*ux[i,j]+gl*Fx)/r0; uy_o[i,j]=(r0*uy[i,j]+gl*Fy)/r0
    end
end

# ---- LTS Brinkman penalization ------------------------------------------------------------------------
@kernel function _penalize2_lts!(ux,uy, @Const(chi), @Const(a_arr), inv_eta::Float32)
    i,j=@index(Global,NTuple)
    @inbounds begin; c=chi[i,j]; if c>0.0f0; f=1.0f0/(1.0f0+a_arr[i,j]*inv_eta*c); ux[i,j]*=f; uy[i,j]*=f; end; end
end

# ---- LTS + POINT-IMPLICIT SST k/ω (engineering block A, 2026-07-12) ------------------------------------
# Sibling of _sst_kernel! (pond_fv2d_thermal.jl) for the steady tier. Three changes, same fixed point:
#   (1) LOCAL pseudo-clock: the transport update advances at dtl = dt·(dtf[i,j]·inv_dtref) — the same
#       per-cell dt the flow uses (dx=1 ⇒ dtl = dtf). Advection stays upwind-CFL-stable because dtf is
#       acoustic-CFL-built and |u| < 3√T+|u|; diffusion number ≤ NUTMAX·cfl/3 ≈ 0.04.
#   (2) POINT-IMPLICIT destruction (Menter/CFL3D practice): β*kω and βω² are treated as β*·k_new·ω_old and
#       β·ω_new·ω_old (divide by the diagonal), and the ω cross-diffusion term is sign-split (positive →
#       explicit RHS, negative → added to the diagonal /ω). Removes the destruction stiffness that forced
#       the k/ω clock to dt_ref; positivity-preserving by construction. At convergence the (1+dtl·β·ω)
#       denominator multiplies a zero residual ⇒ the FIXED POINT IS IDENTICAL to the explicit kernel.
#   (3) ν_t UNDER-RELAXATION (roadmap): the wm write moves toward its target by ≤ 20%/iter (+ an additive
#       seed floor 1e-3·dt so it can leave zero; full ν_t reached in ~30-40 iters). Damps the νt-injection
#       spikes the band manufactures during transients (the SST+small-patch NaN class). Fixed point: wm =
#       dt·ν_t unchanged. Solid cells stay HARD zero (stencil correctness) — no relaxation there.
# wm convention unchanged: nuw_o carries ν_t·dt_ref; _eddy_stress2T_lts! rescales to the local clock.
# nuw_o is read-then-written at [i,j] only (own-cell, no @Const aliasing — restrict-safe).
@kernel function _sst_kernel_lts!(tk_o, tom_o, nuw_o, @Const(tk), @Const(tom), @Const(ux), @Const(uy),
                              @Const(rho), @Const(Temp), @Const(wdist), @Const(chi), @Const(sdf),
                              @Const(gtrip), trip_on::Int32, nu_lat::Float32, dt::Float32, cs2::Float32,
                              kinf::Float32, winf::Float32, uin::Float32, pgr::Int32, ng::Int32,
                              @Const(hk), @Const(hom), nested_kw::Int32, @Const(a_arr), inv_dtref::Float32,
                              band::Float32, kato::Int32, NX::Int32, NY::Int32)
    i,j=@index(Global,NTuple); nx=Int(NX); ny=Int(NY); g=Int(ng)
    βs=0.09f0; sβs=0.3f0; κ=0.41f0; a1=0.31f0
    σk1=0.85f0; σw1=0.5f0; β1=0.075f0; γ1=0.5532f0
    σk2=1.0f0;  σw2=0.856f0; β2=0.0828f0; γ2=0.4403f0
    NUTMAX=0.15f0
    @inbounds begin
        if chi[i,j]>0.5f0 || sdf[i,j]<=0.0f0                      # solid: hard zero, NO under-relax
            tk_o[i,j]=0.0f0; tom_o[i,j]=winf; nuw_o[i,j]=0.0f0
        else
            nuw_t=0.0f0                                           # branch-local ν_t·dt target
            if nested_kw==Int32(1) && (i<=g || i>nx-g || j<=g || j>ny-g)
                kb=hk[i,j]; wb=hom[i,j]
                kb = kb<1.0f-12 ? 1.0f-12 : kb; wb = wb<1.0f-6 ? 1.0f-6 : wb
                if !(isfinite(kb) && isfinite(wb)); kb=kinf; wb=winf; end
                tk_o[i,j]=kb; tom_o[i,j]=wb
                nut=kb/wb; nuw_t=dt*(nut>NUTMAX ? NUTMAX : nut)
            elseif i<=g || j<=g || j>ny-g
                tk_o[i,j]=kinf; tom_o[i,j]=winf; nuw_t=dt*kinf/max(winf,1.0f-6)
            elseif i>nx-g
                tk_o[i,j]=tk[nx-g,j]; tom_o[i,j]=tom[nx-g,j]; nuw_t=dt*min(tk[nx-g,j]/max(tom[nx-g,j],1.0f-6),NUTMAX)
            else
                d=wdist[i,j]; ν=max(nu_lat,1.0f-9)
                umag=sqrt(ux[i,j]*ux[i,j]+uy[i,j]*uy[i,j])
                γt = trip_on==Int32(1) ? gtrip[i,j] : 1.0f0
                if d<band                                         # WALL-FUNCTION band (imposition unchanged; height = PV_SSTBAND)
                    dh = d<0.5f0 ? 0.5f0 : d
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
                    if !(isfinite(k_bc) && isfinite(ω_bc)); k_bc=0.0f0; ω_bc=winf; end
                    tk_o[i,j]=k_bc; tom_o[i,j]=ω_bc
                    nut=k_bc/ω_bc; nut = isfinite(nut) ? (nut>NUTMAX ? NUTMAX : nut) : 0.0f0
                    nuw_t=dt*nut
                else                                              # TRANSPORT: LTS clock + point-implicit destruction
                    iL=_clampi(i-1,nx); iR=_clampi(i+1,nx); jD=_clampi(j-1,ny); jU=_clampi(j+1,ny)
                    k0=tk[i,j]; w0=tom[i,j]; w0 = w0<1.0f-6 ? 1.0f-6 : w0
                    dudx=0.5f0*(ux[iR,j]-ux[iL,j]); dudy=0.5f0*(ux[i,jU]-ux[i,jD])
                    dvdx=0.5f0*(uy[iR,j]-uy[iL,j]); dvdy=0.5f0*(uy[i,jU]-uy[i,jD])
                    Sxy=0.5f0*(dudy+dvdx); S2=2.0f0*(dudx*dudx+dvdy*dvdy+2.0f0*Sxy*Sxy); S=sqrt(S2)
                    dkdx=0.5f0*(tk[iR,j]-tk[iL,j]); dkdy=0.5f0*(tk[i,jU]-tk[i,jD])
                    dwdx=0.5f0*(tom[iR,j]-tom[iL,j]); dwdy=0.5f0*(tom[i,jU]-tom[i,jD])
                    gkw=dkdx*dwdx+dkdy*dwdy
                    CDkw=2.0f0*σw2/w0*gkw; CDkw = CDkw<1.0f-10 ? 1.0f-10 : CDkw
                    sqk=sqrt(k0<0.0f0 ? 0.0f0 : k0)
                    arg1=sqk/(βs*w0*d); t2=500.0f0*ν/(d*d*w0); arg1 = arg1<t2 ? t2 : arg1
                    t3=4.0f0*σw2*k0/(CDkw*d*d); arg1 = arg1<t3 ? arg1 : t3
                    arg1 = arg1>4.0f0 ? 4.0f0 : arg1
                    a14=arg1*arg1*arg1*arg1; F1=tanh(a14)
                    arg2=2.0f0*sqk/(βs*w0*d); arg2 = arg2<t2 ? t2 : arg2; arg2 = arg2>4.0f0 ? 4.0f0 : arg2
                    F2=tanh(arg2*arg2)
                    nut=a1*k0/max(a1*w0, S*F2); nut = nut>NUTMAX ? NUTMAX : nut
                    Pk = kato==Int32(1) ? nut*S*abs(dudy-dvdx) : nut*S2   # Kato-Launder: S·Ω kills spurious P_k at shocks/stagnation
                    plim=10.0f0*βs*k0*w0; Pk = Pk>plim ? plim : Pk; Pk*=γt
                    σk=F1*σk1+(1.0f0-F1)*σk2; σw=F1*σw1+(1.0f0-F1)*σw2
                    βc=F1*β1+(1.0f0-F1)*β2;  γc=F1*γ1+(1.0f0-F1)*γ2
                    advk = (ux[i,j]>=0.0f0 ? ux[i,j]*(k0-tk[iL,j]) : ux[i,j]*(tk[iR,j]-k0)) +
                           (uy[i,j]>=0.0f0 ? uy[i,j]*(k0-tk[i,jD]) : uy[i,j]*(tk[i,jU]-k0))
                    advw = (ux[i,j]>=0.0f0 ? ux[i,j]*(w0-tom[iL,j]) : ux[i,j]*(tom[iR,j]-w0)) +
                           (uy[i,j]>=0.0f0 ? uy[i,j]*(w0-tom[i,jD]) : uy[i,j]*(tom[i,jU]-w0))
                    ntE=0.5f0*(nut+min(max(tk[iR,j],0f0)/max(tom[iR,j],1f-6),NUTMAX)); ntW=0.5f0*(nut+min(max(tk[iL,j],0f0)/max(tom[iL,j],1f-6),NUTMAX))
                    ntN=0.5f0*(nut+min(max(tk[i,jU],0f0)/max(tom[i,jU],1f-6),NUTMAX)); ntS=0.5f0*(nut+min(max(tk[i,jD],0f0)/max(tom[i,jD],1f-6),NUTMAX))
                    Dk=(ν+σk*ntE)*(tk[iR,j]-k0)-(ν+σk*ntW)*(k0-tk[iL,j])+(ν+σk*ntN)*(tk[i,jU]-k0)-(ν+σk*ntS)*(k0-tk[i,jD])
                    Dw=(ν+σw*ntE)*(tom[iR,j]-w0)-(ν+σw*ntW)*(w0-tom[iL,j])+(ν+σw*ntN)*(tom[i,jU]-w0)-(ν+σw*ntS)*(w0-tom[i,jD])
                    dtl = dt*(a_arr[i,j]*inv_dtref)               # local pseudo-dt (dx=1 ⇒ = dtf[i,j])
                    cdt = 2.0f0*(1.0f0-F1)*σw2/w0*gkw             # ω cross-diffusion, sign-split
                    kn = (k0 + dtl*(-advk+Pk+Dk)) / (1.0f0 + dtl*βs*w0)
                    den = 1.0f0 + dtl*βc*w0 + (cdt<0.0f0 ? dtl*(-cdt)/w0 : 0.0f0)
                    wn = (w0 + dtl*(-advw+γc*S2+Dw+(cdt>0.0f0 ? cdt : 0.0f0))) / den
                    kn = kn<1.0f-12 ? 1.0f-12 : (kn>0.5f0 ? 0.5f0 : kn)
                    wn = wn<1.0f-6 ? 1.0f-6 : wn
                    if !(isfinite(kn) && isfinite(wn)); kn=kinf; wn=winf; end
                    tk_o[i,j]=kn; tom_o[i,j]=wn
                    nuw_t=isfinite(nut) ? dt*nut : 0.0f0
                end
            end
            old=nuw_o[i,j]; old = (isfinite(old) && old>=0.0f0) ? old : 0.0f0
            cap=0.2f0*old + 1.0f-3*dt                             # ≤20%/iter + seed floor (0→NUTMAX in ~35 iters)
            dnu=nuw_t-old; dnu = dnu>cap ? cap : (dnu<-cap ? -cap : dnu)
            nuw_o[i,j]=old+dnu
        end
    end
end

# ---- standalone BC application (shared by the steady advance and the Newton wrapper) ------------------
function pond_fv2dT_applyBC_steady!(d::PonD2DT)
    nd=(d.NX,d.NY)
    if d.nested == Int32(1)
        _pin_halo2T!(d.backend)(d.rho,d.ux,d.uy,d.T, d.hrho,d.hux,d.huy,d.hT, Int32(d.NX),Int32(d.NY), Int32(d.ng); ndrange=nd)
        _psync(d.backend); return
    end
    if d.inlet_res == Int32(1)
        _inlet_res2T_x!(d.backend)(d.rho,d.ux,d.uy,d.T, d.inlet_rho0, d.inlet_T0, d.cs2, d.drot, Int32(d.ng); ndrange=nd)
    else
        _inlet2T_x!(d.backend)(d.rho,d.ux,d.uy,d.T, d.rho_inf, d.U_inf, d.T_inf, Int32(d.ng), d.ff_gam, d.ff_xv, d.ff_yv, d.ff_m2, d.ff_beta; ndrange=nd)
    end
    _psync(d.backend)
    if d.ff_mode == Int32(2)
        Deff = 3.0f0 + d.drot; gam = (Deff+2.0f0)/Deff
        _nscbc2T_y!(d.backend)(d.rho,d.ux,d.uy,d.T, d.rho_inf, d.U_inf, d.T_inf, gam, d.cs2, Int32(d.NY), Int32(d.ng); ndrange=nd)
    elseif d.ff_mode == Int32(1) && d.sponge_w>0
        _sponge2T_y!(d.backend)(d.rho,d.ux,d.uy,d.T, d.rho_inf, d.U_inf, d.T_inf, Int32(d.NY), Int32(d.sponge_w), d.sponge_max, d.ff_gam, d.ff_xv, d.ff_yv, d.ff_m2, d.ff_beta; ndrange=nd)
    else
        _farfield2T_y!(d.backend)(d.rho,d.ux,d.uy,d.T, d.rho_inf, d.U_inf, d.T_inf, Int32(d.NY), Int32(d.ng), d.ff_gam, d.ff_xv, d.ff_yv, d.ff_m2, d.ff_beta; ndrange=nd)
    end
    _psync(d.backend)
    _outlet2T_x!(d.backend)(d.rho,d.ux,d.uy,d.T, Int32(d.NX), Int32(d.ng); ndrange=nd); _psync(d.backend)
    return nothing
end

# ---- the steady (pseudo-time) advance: advance!'s RK3 pipeline with the LTS kernels --------------------
# dtf: per-cell pseudo-dt device array for this level (refreshed here each call at the given CFL).
function pond_fv2dT_steady_advance!(d::PonD2DT, dtf, cfl::Float32, t::Int)
    nd=(d.NX,d.NY)
    # REVIEW L1/L2 guards: split storage must be real (silent-OOB class), and only the SST-or-none
    # closure set has LTS siblings (SA/DDES/WMLES transport would silently freeze).
    @assert size(d.dc_r,1)==d.NX && size(d.dc_r,2)==d.NY "steady_advance!: dc_* stubs — build with split_residual=true"
    @assert d.sa==Int32(0) && d.wmles==Int32(0) "steady tier supports SST or no closure (SA/DDES/WMLES have no LTS path)"
    dtau = _pond2T_dtau(d)
    _dtfield2T!(d.backend)(dtf, d.ux,d.uy,d.T, cfl; ndrange=nd); _psync(d.backend)
    if d.dyn_sgs == Int32(1)
        _dyn_lmmm2T!(d.backend)(d.sgs_lm, d.sgs_mm, d.ux, d.uy, 8.0f0, Int32(d.NX),Int32(d.NY); ndrange=nd)
        _psync(d.backend)
        _dyn_lagr2T!(d.backend)(d.sgs_ilmn, d.sgs_immn, d.cd, d.sgs_ilm, d.sgs_imm, d.sgs_lm, d.sgs_mm,
            d.ux, d.uy, Float32(d.dt/d.dx), d.dt, 1.5f0, Int32(d.NX),Int32(d.NY); ndrange=nd)
        _psync(d.backend)
        d.sgs_ilm,d.sgs_ilmn = d.sgs_ilmn,d.sgs_ilm; d.sgs_imm,d.sgs_immn = d.sgs_immn,d.sgs_imm
    end
    ghost!() = _pond2T_ghost!(d, nd)
    wallp = _pond2T_wallp(d)
    flux_sgs = _pond2T_fluxsgs(d)
    inv_dtref = Float32(d.dx/d.dt)                       # 1/dt_ref for the rate-consistent pass scaling
    function flux!()                                     # split residual + LTS update (no fused path here)
        ghost!()
        pond_fv2dT_rect_kernel!(d.backend)(d.rhon,d.uxn,d.uyn,d.Tn, d.rho,d.ux,d.uy,d.T, d.chi, d.cxa,d.cya,d.cza,d.wa,
            d.cs2, d.dt/d.dx, d.drot, d.csmag, d.gauge, wallp, d.shock_aware, d.drp, d.cd, d.dyn_sgs, d.weno, d.sdf, d.ghost_wall, d.lam,d.aE,d.aW,d.aN,d.aS, d.dc_r,d.dc_mx,d.dc_my,d.dc_e, d.frd, d.dugks, dtau, d.h_nom, flux_sgs, Int32(1), Int32(d.NX),Int32(d.NY); ndrange=nd)
        _psync(d.backend)
        if d.fas == Int32(1)                              # FAS: dc ← dc − τ (smoother relaxes R=τ; τ≡0 off-covered ⇒ self-masking)
            _fas_axpy_sub!(d.backend)(d.dc_r,d.dc_mx,d.dc_my,d.dc_e,
                d.fas_tau_r,d.fas_tau_mx,d.fas_tau_my,d.fas_tau_e; ndrange=nd)
            _psync(d.backend)
        end
        if d.ghost_wall == Int32(1)
            @assert d.frd==Int32(1) && d.frd_merge==Int32(1) "steady tier requires flux_redist+cutcell_merge on the cut-cell level"
            _frd_merge2T_lts!(d.backend)(d.rhon,d.uxn,d.uyn,d.Tn, d.rho,d.ux,d.uy,d.T, d.lam,
                d.dc_r,d.dc_mx,d.dc_my,d.dc_e, d.mg_tgt,d.mg_links,d.mg_nlk, dtf, d.cs2, d.drot, Int32(d.NX),Int32(d.NY); ndrange=nd)
        else
            pond_fv2dT_rect_update_lts!(d.backend)(d.rhon,d.uxn,d.uyn,d.Tn, d.rho,d.ux,d.uy,d.T,
                d.dc_r,d.dc_mx,d.dc_my,d.dc_e, dtf, d.cs2, d.drot; ndrange=nd)
        end
        _psync(d.backend)
    end
    function swapmain!()
        d.rho,d.rhon=d.rhon,d.rho; d.ux,d.uxn=d.uxn,d.ux; d.uy,d.uyn=d.uyn,d.uy; d.T,d.Tn=d.Tn,d.T
        return nothing
    end
    applyBC!() = pond_fv2dT_applyBC_steady!(d)
    if d.weno == Int32(1)                                 # SSP-RK3 in pseudo-time (WENO needs it)
        eint = 0.5f0*(3.0f0+d.drot)*d.cs2
        _prim2cons2T!(d.backend)(d.rk_r,d.rk_mx,d.rk_my,d.rk_E, d.rho,d.ux,d.uy,d.T, eint; ndrange=nd); _psync(d.backend)
        flux!(); swapmain!(); applyBC!()
        flux!()
        _rkcombine2T!(d.backend)(d.rho,d.ux,d.uy,d.T, d.rk_r,d.rk_mx,d.rk_my,d.rk_E, d.rhon,d.uxn,d.uyn,d.Tn, 0.75f0,0.25f0, eint; ndrange=nd)
        _psync(d.backend); applyBC!()
        flux!()
        _rkcombine2T!(d.backend)(d.rho,d.ux,d.uy,d.T, d.rk_r,d.rk_mx,d.rk_my,d.rk_E, d.rhon,d.uxn,d.uyn,d.Tn, 0.33333334f0,0.6666667f0, eint; ndrange=nd)
        _psync(d.backend)
    else
        flux!(); swapmain!()
    end
    svv_ran = (d.csmag > 0.0f0 || d.weno == Int32(1)) && (d.svv_eps > 0.0f0 || d.svv_rho > 0.0f0 || d.nu_mol > 0.0f0 || d.wall_model == Int32(1))
    if svv_ran
        ghost!()
        numol_pass = (d.dugks==Int32(1) ? 0.0f0 : d.nu_mol)/Float32(d.svv_passes)
        wm_scale = 1.0f0/Float32(d.svv_passes)
        wm_on_svv = (d.ddes==Int32(1) || d.sst==Int32(1)) ? Int32(0) : d.wall_model
        for _ in 1:d.svv_passes
            _svvT_lts!(d.backend)(d.rhon,d.uxn,d.uyn, d.rho,d.ux,d.uy,d.T, d.svv_eps, d.svv_rho, numol_pass, d.wm, wm_scale, wm_on_svv, dtf, inv_dtref, d.cs2, Int32(d.NX),Int32(d.NY); ndrange=nd)
            _psync(d.backend)
            d.rho,d.rhon=d.rhon,d.rho; d.ux,d.uxn=d.uxn,d.ux; d.uy,d.uyn=d.uyn,d.uy
        end
    end
    if d.ddes == Int32(1) || d.sst == Int32(1)
        svv_ran || ghost!()
        _eddy_stress2T_lts!(d.backend)(d.uxn,d.uyn, d.rho,d.ux,d.uy,d.T, d.wm, dtf, inv_dtref, d.cs2, d.eddy_gate, Int32(d.NX),Int32(d.NY); ndrange=nd)
        _psync(d.backend)
        d.ux,d.uxn=d.uxn,d.ux; d.uy,d.uyn=d.uyn,d.uy
    end
    _penalize2_lts!(d.backend)(d.ux,d.uy, d.chi, dtf, Float32(1.0/d.eta); ndrange=nd); _psync(d.backend)
    ghost!()
    applyBC!()
    if d.sst == Int32(1)
        if _sst_pi_on()                                   # engineering block A: LTS clock + point-implicit destruction + ν_t under-relax
            _sst_kernel_lts!(d.backend)(d.tkn, d.tomn, d.wm, d.tk, d.tom, d.ux,d.uy, d.rho,d.T, d.wdist, d.chi, d.sdf,
                d.gtrip, d.trip_on, d.sa_nulat, d.dt, d.cs2, d.sst_kinf, d.sst_winf, abs(d.U_inf), d.pgr, Int32(d.ng),
                d.hk, d.hom, d.sst_halo, dtf, inv_dtref, _sst_band(), Int32(_sst_kato() ? 1 : 0), Int32(d.NX),Int32(d.NY); ndrange=nd)
        else                                              # PV_SSTPI=0: legacy uniform dt_ref pseudo-clock (fixed point identical)
            _sst_kernel!(d.backend)(d.tkn, d.tomn, d.wm, d.tk, d.tom, d.ux,d.uy, d.rho,d.T, d.wdist, d.chi, d.sdf,
                d.gtrip, d.trip_on, d.sa_nulat, d.dt, d.cs2, d.sst_kinf, d.sst_winf, abs(d.U_inf), d.pgr, Int32(d.ng),
                d.hk, d.hom, d.sst_halo, _sst_band(), Int32(_sst_kato() ? 1 : 0), Int32(d.NX),Int32(d.NY); ndrange=nd)
        end
        _psync(d.backend)
        d.tk, d.tkn = d.tkn, d.tk; d.tom, d.tomn = d.tomn, d.tom
    end
    return nothing
end

# ---- steady tree step: same phases as pond_fv2dT_tree_step! with the steady advance -------------------
function pond_fv2dT_steady_tree_step!(G::Vector{<:PonD2DT}, T::Vector{PondTreeNode}, dtfs::Vector, cfl::Float32, t::Int)
    n=length(G)
    @assert all(nd->nd.substeps==1, T) "steady tier: sub-cycled patches unsupported (PV_PATCH=0)"
    for i in 2:n
        T[i].parent==0 && continue
        _tree_prolong_pin!(G[i], G[T[i].parent], T[i].pbox)
        if G[i].sst==Int32(1) && G[T[i].parent].sst==Int32(1) && size(G[i].hk,1)>1
            G[i].sst_halo=Int32(1)
            ci0,cj0,_,_=T[i].pbox
            prolong_kw_l1l2_2dT!(G[i].backend)(G[i].hk,G[i].hom, G[T[i].parent].tk,G[T[i].parent].tom,
                Int32(ci0),Int32(cj0), Int32(G[i].ng), Int32(G[T[i].parent].NX),Int32(G[T[i].parent].NY), Int32(G[i].NX),Int32(G[i].NY); ndrange=(Int(G[i].NX),Int(G[i].NY)))
            _psync(G[i].backend)
        end
    end
    for i in n:-1:1; pond_fv2dT_steady_advance!(G[i], dtfs[i], cfl, t); end
    for i in n:-1:2; _tree_restrict!(G[i], G[T[i].parent], T[i].pbox); end
    return nothing
end

# ==============================================================================
# FAS pseudo-time MULTIGRID V-cycle (2026-07-14, docs/2026-07-14_FAS_phase1_plan.md — PV_FAS)
# Replaces the "smooth-all + overwrite-restrict" pass with a τ-corrected FAS(ν1,ν2) V-cycle over the
# RECTANGULAR sub-chain L[1..m]; the cut-cell leaf tail L[m+1..n] keeps the exact legacy path (smooth +
# overwrite-restrict — Phase-1 exclusion; extension = Phase-1b). Coarse levels relax R_H(U_H)=τ_H with
# τ_H = R_H(I·U_h) − I·(R_h(U_h)−τ_h)  (the τ_h term = correct 3+-level composition: an intermediate
# level restricts the residual of ITS OWN τ-corrected equation, not raw R). The coarse correction
# dU = U_H − Ū (Ū snapshotted at restriction time, incl. a 2-cell stencil rim) is prolonged ADDITIVELY
# into the fine interior. FIXED POINT: at a converged fine state R−τ≡0 cascades down the chain ⇒ every
# τ_H exactly cancels the coarse defect ⇒ dU≡0 ⇒ identical converged state to PV_FAS=0, reached faster
# (the parent works the fine level's low-frequency error at its larger local dt).
# ==============================================================================
function pond_fv2dT_fas_tree_step!(G::Vector{<:PonD2DT}, T::Vector{PondTreeNode}, dtfs::Vector, cfl::Float32, t::Int;
                                   nu1::Int=2, nu2::Int=1, nuc::Int=6, rratio::Float32=2f0)
    n=length(G)
    @assert all(nd->nd.substeps==1, T) "FAS steady tier: sub-cycled patches unsupported"
    # m = deepest rectangular (non-cut-cell) level of the chain; the leaf tail m+1..n runs the legacy path.
    m=n; for i in 1:n; if G[i].ghost_wall==Int32(1); m=i-1; break; end; end
    @assert m>=2 "FAS needs ≥2 rectangular levels (m=$m)"
    for i in 1:m; @assert G[i].fas==Int32(1) "FAS: rect level $i built without fas=true"; end
    smooth(i,k) = (for _ in 1:k; pond_fv2dT_steady_advance!(G[i], dtfs[i], cfl, t); end)

    # phase 1: pin every child halo from its parent's current state (identical to the legacy step)
    for i in 2:n
        T[i].parent==0 && continue
        _tree_prolong_pin!(G[i], G[T[i].parent], T[i].pbox)
        if G[i].sst==Int32(1) && G[T[i].parent].sst==Int32(1) && size(G[i].hk,1)>1
            G[i].sst_halo=Int32(1)
            ci0,cj0,_,_=T[i].pbox
            prolong_kw_l1l2_2dT!(G[i].backend)(G[i].hk,G[i].hom, G[T[i].parent].tk,G[T[i].parent].tom,
                Int32(ci0),Int32(cj0), Int32(G[i].ng), Int32(G[T[i].parent].NX),Int32(G[T[i].parent].NY), Int32(G[i].NX),Int32(G[i].NY); ndrange=(Int(G[i].NX),Int(G[i].NY)))
            _psync(G[i].backend)
        end
    end

    # cut-cell leaf tail: EXISTING path (smooth + overwrite-restrict into L[m])
    for i in n:-1:m+1; smooth(i, nu1+nu2); end
    for i in n:-1:m+1; _tree_restrict!(G[i], G[T[i].parent], T[i].pbox); end

    # ---- DOWN-LEG (fine→coarse) over L[m..2]: pre-smooth, restrict U and (R−τ), form τ on the parent
    for i in m:-1:2
        p=T[i].parent; ci0,cj0,nbx,nby=T[i].pbox
        smooth(i, nu1)                                            # ν1 pre-smooth (AXPY inside relaxes R=τ_i)
        pond_fv2dT_residual!(G[i])                                # dc_* ← R_i(U_i)  (state untouched)
        _fas_axpy_sub!(G[i].backend)(G[i].dc_r,G[i].dc_mx,G[i].dc_my,G[i].dc_e,
            G[i].fas_tau_r,G[i].fas_tau_mx,G[i].fas_tau_my,G[i].fas_tau_e; ndrange=(Int(G[i].NX),Int(G[i].NY)))
        _psync(G[i].backend)              # dc_* ← R_i − τ_i (τ_m≡0 ⇒ no-op at the chain leaf)
        _fas_restrict_rate!(G[p].backend)(G[p].fas_tau_r,G[p].fas_tau_mx,G[p].fas_tau_my,G[p].fas_tau_e,
            G[i].dc_r,G[i].dc_mx,G[i].dc_my,G[i].dc_e,
            Int32(ci0),Int32(cj0),Int32(G[i].ng),rratio; ndrange=(nbx,nby))
        _psync(G[p].backend)              # stage I·(R_i−τ_i) into parent τ slots
        _tree_restrict!(G[i], G[p], T[i].pbox)                    # U_p(covered) ← I·U_i (existing overwrite)
        eint=0.5f0*(3.0f0+G[p].drot)*G[p].cs2
        _fas_snapshot_ub!(G[p].backend)(G[p].fas_ub_r,G[p].fas_ub_mx,G[p].fas_ub_my,G[p].fas_ub_e,
            G[p].rho,G[p].ux,G[p].uy,G[p].T, eint, Int32(ci0),Int32(cj0),
            Int32(G[p].NX),Int32(G[p].NY); ndrange=(nbx+4,nby+4))
        _psync(G[p].backend)              # Ū_p (covered + 2-cell rim), BEFORE coarse smoothing
        pond_fv2dT_residual!(G[p])                                # dc_* ← R_p(I·U_i)  (p already holds I·U_i)
        _fas_tau_finalize!(G[p].backend)(G[p].fas_tau_r,G[p].fas_tau_mx,G[p].fas_tau_my,G[p].fas_tau_e,
            G[p].dc_r,G[p].dc_mx,G[p].dc_my,G[p].dc_e,
            Int32(ci0),Int32(cj0); ndrange=(nbx,nby))
        _psync(G[p].backend)              # τ_p = R_p(I·U_i) − I·(R_i−τ_i)
    end

    # ---- COARSE SOLVE (root): relax R_1(U_1)=τ_1
    smooth(1, nuc)

    # ---- UP-LEG (coarse→fine) over L[2..m]: additive interior correction, re-pin halo, post-smooth
    for i in 2:m
        p=T[i].parent; ci0,cj0,_,_=T[i].pbox
        eint=0.5f0*(3.0f0+G[p].drot)*G[p].cs2
        _fas_prolong_add_interior!(G[i].backend)(G[i].rho,G[i].ux,G[i].uy,G[i].T,
            G[p].rho,G[p].ux,G[p].uy,G[p].T,
            G[p].fas_ub_r,G[p].fas_ub_mx,G[p].fas_ub_my,G[p].fas_ub_e,
            Int32(ci0),Int32(cj0),Int32(G[i].ng),eint,
            Int32(G[p].NX),Int32(G[p].NY),Int32(G[i].NX),Int32(G[i].NY); ndrange=(Int(G[i].NX),Int(G[i].NY)))
        _psync(G[i].backend)              # U_i(interior) += P(U_p − Ū_p)
        _tree_prolong_pin!(G[i], G[p], T[i].pbox)                 # re-pin halo from the corrected parent
        smooth(i, nu2)                                            # ν2 post-smooth
    end
    return nothing
end
