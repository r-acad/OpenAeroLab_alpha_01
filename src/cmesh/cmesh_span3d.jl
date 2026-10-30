# ---------------------------------------------------------------------------------------------
# PATH STATUS: **NOT PRODUCTION** -- quasi-3D spanwise extrusion (July 2026 experiment).
# Preserved under the working rule 'preserve old solvers and experiments'. It is NOT
# reachable from the production driver, so results obtained here are NOT statements about
# the production path. Map: docs/CAVITATION_CODE_MAP.md.
# ---------------------------------------------------------------------------------------------
# CMESH quasi-3D — spanwise extrusion of the validated 2D C-mesh KFVS solver. Companion module, ZERO
# surgery on cmesh_solver_ka.jl. See docs/2026-07-18_SPANWISE_extrusion_plan.md. The 2D cross-section is
# extruded into NK cells over z in [0,Lz), uniform dz, PERIODIC in z. Purpose: capture the 3D re-entrant
# jet / cloud shedding that 2D-URANS provably cannot (commits ff56426, 2ba26f5).
#
# State (ρ, ρu, ρv, ρw, ρE), p = ρ·cs²·T, cs²=2/3. The 2D solver carried the D3Q39 cz-velocities only in
# the internal energy (the drot device); 3D promotes span to a real transported momentum ρw. Flux = the
# rotated-frame KFVS extended to 3D (adds the pz peculiar-velocity + Fmz component + w-shift of vz) for the
# curvilinear i/j faces, plus a plain Cartesian KFVS for the uniform spanwise k-faces. At w≡0 and no
# spanwise variation every 3D addition vanishes/telescopes ⇒ the solver reduces EXACTLY to 2D (Gate G1).
#
# v1 scope was: laminar (nu_lam thin-layer), minmod reconstruction, HEM barotropic cavitation hook
# (per-cell, via the caller). SUPERSEDED — now landed: SST-3D (+ F1 gammat trip, Reboud), F2 full viscous
# stress tensor with Green-Gauss gradients, BDF2 dual-time, the TEM finite-rate cavitation model, the
# acoustic filter, and (2026-07-26) WENO5-Z reconstruction on ALL THREE flux directions — i/j via
# _fvflux_ij_3d and the spanwise k via _fvflux_k_3d, both selected by s.recon (0=minmod legacy, 1=WENO).
module CMeshSpan3D

using KernelAbstractions
const KA = KernelAbstractions
include(joinpath(@__DIR__, "cmesh_flux_core.jl"))   # CS2, TGAS_D, _mm, _faceL_mm, _faceR_mm, _mufac

export Span3D, build_span3d, set_farfield_mixture3d!, set_freestream_3d!, step_3d!, loads_3d, span_uniformity, rstar_report_3d, WG3
export hem_apply_3d!, seed_span_perturb!, cavity_len_3d, hist_cycle_3d!, seed_hist_3d!
export set_sst_3d!, sst_step_3d!, acoustic_filter_3d!

const WG3 = (64, 1, 1)

# ---- 3D equilibrium + fluxes (frame-agnostic peculiar velocity; reduces to _feqT2 at pz=0) ----------
@inline function _feqT2_3d(rho, un, ut, uz, T, ufn, uft, ufz, cxk, cyk, czk, wk)
    px = un - ufn; py = ut - uft; pz = uz - ufz
    inv_cs2 = 1.0f0/CS2; inv_2cs4 = 0.5f0/(CS2*CS2); inv_6cs6 = 1.0f0/(6.0f0*CS2*CS2*CS2)
    cp = px*cxk + py*cyk + pz*czk
    p2 = px*px + py*py + pz*pz
    csq = cxk*cxk + cyk*cyk + czk*czk
    th = T - 1.0f0
    f  = rho
    f += rho*cp*inv_cs2
    f += rho*(cp*cp - CS2*p2)*inv_2cs4
    f += rho*th*0.5f0*(csq*inv_cs2 - TGAS_D)
    f += rho*(cp*cp*cp - 3.0f0*CS2*cp*p2)*inv_6cs6
    f += rho*th*cp*(csq - (TGAS_D+2.0f0)*CS2)*inv_2cs4
    return max(wk*f, 1.0f-12)
end

# curvilinear i/j face (normal in x-y plane): rotate u,v into (un,ut); w = spanwise tangent uz.
# 6-STATE stencil, recon-selectable (2026-07-26 WENO PORT — the 3D was v1 pure-minmod while the certified
# 2D shedder runs recon=1 WENO5-Z; minmod over-dissipates the sharp cavity-closure/re-entrant-jet gradients
# so the 3D cavity never CUTS -> overgrows to supercavity. Mirrors _fvflux_normal6 exactly, so at span-
# uniform w=0 the 3D i/j flux == the 2D flux (G1)). recon: 0=minmod (bitwise legacy), 1=WENO5-Z blend.
# Returns (Frho,Fmx,Fmy,Fmz,FE)·S (global-frame momentum, face-area scaled).
@inline function _fvflux_ij_3d(rLLL,uxLLL,uyLLL,wLLL,TLLL, rLL,uxLL,uyLL,wLL,TLL, rL,uxL,uyL,wL,TL,
                               rR,uxR,uyR,wR,TR, rRR,uxRR,uyRR,wRR,TRR, rRRR,uxRRR,uyRRR,wRRR,TRRR,
                               nx::Float32, ny::Float32, S::Float32, cx_arr, cy_arr, cz_arr, w_arr, drot::Float32, recon::Int32)
    unLLL=uxLLL*nx+uyLLL*ny; utLLL=-uxLLL*ny+uyLLL*nx
    unLL =uxLL*nx+uyLL*ny;   utLL =-uxLL*ny+uyLL*nx
    unL  =uxL*nx+uyL*ny;     utL  =-uxL*ny+uyL*nx
    unR  =uxR*nx+uyR*ny;     utR  =-uxR*ny+uyR*nx
    unRR =uxRR*nx+uyRR*ny;   utRR =-uxRR*ny+uyRR*nx
    unRRR=uxRRR*nx+uyRRR*ny; utRRR=-uxRRR*ny+uyRRR*nx
    local rLf::Float32, rRf::Float32, unLf::Float32, unRf::Float32, utLf::Float32, utRf::Float32, uzLf::Float32, uzRf::Float32, TLf::Float32, TRf::Float32
    if recon == Int32(1)
        pLm=rL*CS2*TL; pRm=rR*CS2*TR
        spj=abs(pRm-pLm)/min(pLm,pRm); sj=max(spj, abs(rR-rL)/min(rL,rR))
        sb= sj<=0.1f0 ? 0.0f0 : (sj>=0.4f0 ? 1.0f0 : (sj-0.1f0)*(1.0f0/0.3f0))
        rLf=max(_wenoblend(sb,rLLL,rLL,rL,rR,rRR),1.0f-4);  rRf=max(_wenoblend(sb,rRRR,rRR,rR,rL,rLL),1.0f-4)
        unLf=_wenoblend(sb,unLLL,unLL,unL,unR,unRR);         unRf=_wenoblend(sb,unRRR,unRR,unR,unL,unLL)
        utLf=_wenoblend(sb,utLLL,utLL,utL,utR,utRR);         utRf=_wenoblend(sb,utRRR,utRR,utR,utL,utLL)
        uzLf=_wenoblend(sb,wLLL,wLL,wL,wR,wRR);              uzRf=_wenoblend(sb,wRRR,wRR,wR,wL,wLL)
        TLf=max(_wenoblend(sb,TLLL,TLL,TL,TR,TRR),1.0f-3);   TRf=max(_wenoblend(sb,TRRR,TRR,TR,TL,TLL),1.0f-3)
    else
        rLf=max(_faceL_mm(rLL,rL,rR),1.0f-4); rRf=max(_faceR_mm(rL,rR,rRR),1.0f-4)
        unLf=_faceL_mm(unLL,unL,unR); unRf=_faceR_mm(unL,unR,unRR)
        utLf=_faceL_mm(utLL,utL,utR); utRf=_faceR_mm(utL,utR,utRR)
        uzLf=_faceL_mm(wLL,wL,wR);    uzRf=_faceR_mm(wL,wR,wRR)
        TLf=max(_faceL_mm(TLL,TL,TR),1.0f-3); TRf=max(_faceR_mm(TL,TR,TRR),1.0f-3)
    end
    ufn=0.5f0*(unLf+unRf); uft=0.5f0*(utLf+utRf); ufz=0.5f0*(uzLf+uzRf); erc=0.5f0*drot*CS2
    Frho=0.0f0; Fmn=0.0f0; Fmt=0.0f0; Fmz=0.0f0; FE=0.0f0
    @inbounds for k in 1:39
        cxk=cx_arr[k]; cyk=cy_arr[k]; czk=cz_arr[k]; wk=w_arr[k]
        vn=cxk+ufn; vt=cyk+uft; vz=czk+ufz
        if vn>=0.0f0
            fk=_feqT2_3d(rLf,unLf,utLf,uzLf,TLf,ufn,uft,ufz,cxk,cyk,czk,wk); Ts=TLf
        else
            fk=_feqT2_3d(rRf,unRf,utRf,uzRf,TRf,ufn,uft,ufz,cxk,cyk,czk,wk); Ts=TRf
        end
        ke=0.5f0*(vn*vn+vt*vt+vz*vz)+erc*Ts
        Frho+=vn*fk; Fmn+=vn*vn*fk; Fmt+=vn*vt*fk; Fmz+=vn*vz*fk; FE+=vn*ke*fk
    end
    Fmx=nx*Fmn-ny*Fmt; Fmy=ny*Fmn+nx*Fmt
    return Frho*S, Fmx*S, Fmy*S, Fmz*S, FE*S
end

# spanwise k face (Cartesian, normal +z): NORMAL uses czk; tangents x,y. Raw flux (÷dz in the update).
# 6-STATE stencil, recon-selectable (2026-07-26 WENO PORT, part 2 — completes the 2D-parity reconstruction
# started on the i/j flux). The i/j port cured the cavity-CUTTING dissipation; this one cures the SPANWISE
# over-dissipation that damps the cellular structure the 3D shedding is supposed to develop (the handover's
# step 3). Stencil ka-3..ka+2 fits exactly: g=3 and _k_bc_span_3d! fills all 3 z-ghost layers periodically.
# G1-preserving: at span-uniform all 6 states are equal, the sensor gives sb=0, and WENO5-Z reproduces a
# constant EXACTLY (each candidate stencil sums to the cell value), so the k-flux is unchanged there.
# recon: 0=minmod (bitwise legacy), 1=WENO5-Z blend.
@inline function _fvflux_k_3d(rLLL,uxLLL,uyLLL,wLLL,TLLL, rLL,uxLL,uyLL,wLL,TLL, rL,uxL,uyL,wL,TL,
                              rR,uxR,uyR,wR,TR, rRR,uxRR,uyRR,wRR,TRR, rRRR,uxRRR,uyRRR,wRRR,TRRR,
                              cx_arr, cy_arr, cz_arr, w_arr, drot::Float32, recon::Int32)
    local rLf::Float32, rRf::Float32, uLf::Float32, uRf::Float32, vLf::Float32, vRf::Float32, wLf::Float32, wRf::Float32, TLf::Float32, TRf::Float32
    if recon == Int32(1)
        pLm=rL*CS2*TL; pRm=rR*CS2*TR
        spj=abs(pRm-pLm)/min(pLm,pRm); sj=max(spj, abs(rR-rL)/min(rL,rR))
        sb= sj<=0.1f0 ? 0.0f0 : (sj>=0.4f0 ? 1.0f0 : (sj-0.1f0)*(1.0f0/0.3f0))
        rLf=max(_wenoblend(sb,rLLL,rLL,rL,rR,rRR),1.0f-4);  rRf=max(_wenoblend(sb,rRRR,rRR,rR,rL,rLL),1.0f-4)
        uLf=_wenoblend(sb,uxLLL,uxLL,uxL,uxR,uxRR);          uRf=_wenoblend(sb,uxRRR,uxRR,uxR,uxL,uxLL)
        vLf=_wenoblend(sb,uyLLL,uyLL,uyL,uyR,uyRR);          vRf=_wenoblend(sb,uyRRR,uyRR,uyR,uyL,uyLL)
        wLf=_wenoblend(sb,wLLL,wLL,wL,wR,wRR);               wRf=_wenoblend(sb,wRRR,wRR,wR,wL,wLL)
        TLf=max(_wenoblend(sb,TLLL,TLL,TL,TR,TRR),1.0f-3);   TRf=max(_wenoblend(sb,TRRR,TRR,TR,TL,TLL),1.0f-3)
    else
        rLf=max(_faceL_mm(rLL,rL,rR),1.0f-4); rRf=max(_faceR_mm(rL,rR,rRR),1.0f-4)
        uLf=_faceL_mm(uxLL,uxL,uxR); uRf=_faceR_mm(uxL,uxR,uxRR)
        vLf=_faceL_mm(uyLL,uyL,uyR); vRf=_faceR_mm(uyL,uyR,uyRR)
        wLf=_faceL_mm(wLL,wL,wR);    wRf=_faceR_mm(wL,wR,wRR)
        TLf=max(_faceL_mm(TLL,TL,TR),1.0f-3); TRf=max(_faceR_mm(TL,TR,TRR),1.0f-3)
    end
    ufx=0.5f0*(uLf+uRf); ufy=0.5f0*(vLf+vRf); ufz=0.5f0*(wLf+wRf); erc=0.5f0*drot*CS2
    Frho=0.0f0; Fmx=0.0f0; Fmy=0.0f0; Fmz=0.0f0; FE=0.0f0
    @inbounds for k in 1:39
        cxk=cx_arr[k]; cyk=cy_arr[k]; czk=cz_arr[k]; wk=w_arr[k]
        vn=czk+ufz; vx=cxk+ufx; vy=cyk+ufy
        if vn>=0.0f0
            fk=_feqT2_3d(rLf,uLf,vLf,wLf,TLf,ufx,ufy,ufz,cxk,cyk,czk,wk); Ts=TLf
        else
            fk=_feqT2_3d(rRf,uRf,vRf,wRf,TRf,ufx,ufy,ufz,cxk,cyk,czk,wk); Ts=TRf
        end
        ke=0.5f0*(vn*vn+vx*vx+vy*vy)+erc*Ts
        Frho+=vn*fk; Fmx+=vn*vx*fk; Fmy+=vn*vy*fk; Fmz+=vn*vn*fk; FE+=vn*ke*fk
    end
    return Frho, Fmx, Fmy, Fmz, FE
end

# ---- state ------------------------------------------------------------------------------------------
mutable struct Span3D{A3, A2, V}
    NI::Int; NJ::Int; NK::Int; NIW::Int; NIF::Int; g::Int
    dz::Float32
    rho::A3; ux::A3; uy::A3; uz::A3; T::A3
    rk_r::A3; rk_mx::A3; rk_my::A3; rk_mz::A3; rk_E::A3
    Fi_r::A3; Fi_mx::A3; Fi_my::A3; Fi_mz::A3; Fi_e::A3
    Fj_r::A3; Fj_mx::A3; Fj_my::A3; Fj_mz::A3; Fj_e::A3
    Fk_r::A3; Fk_mx::A3; Fk_my::A3; Fk_mz::A3; Fk_e::A3
    dtf::A3
    un_r::A3; un_mx::A3; un_my::A3; un_mz::A3; un_E::A3   # dual-time BDF2 history U^n     (interior-sized)
    um_r::A3; um_mx::A3; um_my::A3; um_mz::A3; um_E::A3   #                      U^(n-1)   (inert unless dtp>0)
    tk::A3; tom::A3; nut::A3                              # k-omega SST: turbulent KE, specific dissipation, eddy visc
    knew::A3; wnew::A3; nutl::A3                          # SST sweep scratch (ghosted; commit copies back)
    wdist::A2                                             # wall distance (2D cross-section, reused per k-plane)
    gammat::A2                                             # trip intermittency (F1 2026-07-26: SST parity with the 2D shedder -- laminar LE ahead of the trip; the 3D previously ran fully turbulent from the LE)
    Six::A2; Siy::A2; Sjx::A2; Sjy::A2; vol::A2; xc::A2; yc::A2
    cxa::V; cya::V; cza::V; wa::V
    drot::Float32; eint::Float32
    rho_inf::Float32; uxi::Float32; uyi::Float32; T_inf::Float32; nu_lam::Float32; ca::Float32; sa::Float32
    sst::Bool; k_inf::Float32; w_inf::Float32
    Sjx1_h::Vector{Float32}; Sjy1_h::Vector{Float32}; vol1_h::Vector{Float32}
    wall_flux::Bool
    # PDP-KFVS pressure-split (mirrors KSol): p_mix = rho*cs2*T_band + pexc. cav LEVEL:
    # 0 = off (bitwise legacy), 1 = pexc split (breaks the 3:1 cap), 2 = +Layer C (preconditioned a_m).
    pexc::A3; am::A3
    cav::Int32
    # ★ Tier-C 3D mixture far-field (2026-07-23, 2D ff_* twin): barotropic characteristics on the
    # ambient mixture sound speed; REQUIRED whenever the TEM parameterisation makes the true liquid
    # pressure differ from the lattice ambient (eps_pv). ff_mix=0 = bitwise legacy ideal-gas branch.
    ff_mix::Int32; ff_p_ref::Float32; ff_a2m::Float32; ff_Tmin::Float32; ff_Tmax::Float32
    # ★ ACOUSTIC FILTER (2026-07-26, 3D port of the 2D acoustic_filter_ka!): ADI (i then j, per k-plane)
    # line-implicit low-pass of the pseudo-time DENSITY INCREMENT delta = rho - rk_r, gated to the railed
    # negative-compressibility core (gap = cs2*T - a_m^2 > 0), coeff = the GAP Kf = max(0, cs2*Tf - a2mf).
    # Cures the FAST negative-compressibility divergence (the numerical instability that hit 3D too). Filters
    # the increment (fixed-point-exact), interface-gated (no cross-interface smoothing), momentum invariant
    # (u,v,w *= rho_pre/rho_new). acou=0 => bitwise legacy. Metrics are the shared 2D cross-section; the span
    # (k) direction is periodic and not swept (the negative-compressibility instability is in the i-j plane).
    acou::Int32; acou_kfac::Float32
    acou_d::A3; acou_bp::A3; acou_an::A3                  # interior-sized scratch: delta + Thomas bp + off-diag
    # ★ FULL 3D VISCOUS STRESS (F2 2026-07-26): 0 = thin-layer face-normal only (bitwise legacy), 1 = full
    # tau = mu(grad u + grad u^T - 2/3 div I) incl the spanwise cross-derivative shear that carries the
    # re-entrant-jet closure. Green-Gauss cell gradients (ghosted; x,y per k-plane via shared metrics, z
    # central) precomputed each RK substage. Off => the 12 arrays are unused and cost only their alloc.
    fullvisc::Int32
    recon::Int32                                          # convective reconstruction: 0 = minmod (v1 legacy), 1 = WENO5-Z (2D-parity, cuts the cavity)
    gux::A3; guy::A3; guz::A3; gvx::A3; gvy::A3; gvz::A3
    gwx::A3; gwy::A3; gwz::A3; gTx::A3; gTy::A3; gTz::A3
    backend
end

_z3(b, nx, ny, nz) = (a = KA.allocate(b, Float32, nx, ny, nz); fill!(a, 0.0f0); a)

function build_span3d(backend, met, NK::Int, Lz::Float64; drot::Float32=2.0f0, cxv, cyv, czv, wv)
    if hasproperty(met, :outer_face_kind) && any(kind -> kind != 0x00, met.outer_face_kind)
        throw(ArgumentError("segmented outer-channel boundaries are not implemented in Span3D"))
    end
    NI, NJ, g = met.NI, met.NJ, 3
    dev2(x) = (a = KA.allocate(backend, Float32, size(x)...); copyto!(a, Float32.(x)); a)
    devv(x) = (a = KA.allocate(backend, Float32, length(x)); copyto!(a, collect(Float32.(x))); a)
    NX, NY, NZ = NI+2g, NJ+2g, NK+2g
    z3() = _z3(backend, NX, NY, NZ)
    zi() = _z3(backend, NI+1, NJ, NK)     # i-faces (interior k)
    zj() = _z3(backend, NI, NJ+1, NK)     # j-faces
    zk() = _z3(backend, NI, NJ, NK+1)     # k-faces
    eint = 0.5f0*(3.0f0+drot)*CS2
    Span3D{typeof(z3()), typeof(dev2(met.vol)), typeof(devv(cxv))}(
        NI, NJ, NK, met.NIW, met.NIF, g, Float32(Lz/NK),
        z3(), z3(), z3(), z3(), z3(),
        z3(), z3(), z3(), z3(), z3(),
        zi(), zi(), zi(), zi(), zi(),
        zj(), zj(), zj(), zj(), zj(),
        zk(), zk(), zk(), zk(), zk(),
        _z3(backend, NI, NJ, NK),
        _z3(backend, NI, NJ, NK), _z3(backend, NI, NJ, NK), _z3(backend, NI, NJ, NK), _z3(backend, NI, NJ, NK), _z3(backend, NI, NJ, NK),
        _z3(backend, NI, NJ, NK), _z3(backend, NI, NJ, NK), _z3(backend, NI, NJ, NK), _z3(backend, NI, NJ, NK), _z3(backend, NI, NJ, NK),
        z3(), z3(), z3(),                                    # tk, tom, nut
        z3(), z3(), z3(),                                    # knew, wnew, nutl (SST scratch)
        dev2(met.wdist), dev2(met.gammat),                   # wdist, gammat (F1 SST parity)
        dev2(met.Six), dev2(met.Siy), dev2(met.Sjx), dev2(met.Sjy), dev2(met.vol), dev2(met.xc), dev2(met.yc),
        devv(cxv), devv(cyv), devv(czv), devv(wv),
        drot, eint,
        1.0f0, 0.0f0, 0.0f0, 1.0f0, 0.0f0, 1.0f0, 0.0f0,
        false, 0.0f0, 0.0f0,                                 # sst, k_inf, w_inf
        Float32.(met.Sjx[:,1]), Float32.(met.Sjy[:,1]), Float32.(met.vol[:,1]),
        false,
        z3(), z3(), Int32(0),                                # pexc, am, cav (PDP split; off by default)
        Int32(0), 0.0f0, 0.0f0, 0.3f0, 3.0f0,                # ff_mix, ff_p_ref, ff_a2m, ff_Tmin, ff_Tmax
        Int32(0), 1.0f0,                                     # acou (OFF => bitwise legacy), acou_kfac
        _z3(backend, NI, NJ, NK), _z3(backend, NI, NJ, NK), _z3(backend, NI, NJ, NK),  # acou_d, acou_bp, acou_an
        Int32(0), Int32(0),                                  # fullvisc, recon (both OFF => bitwise legacy: thin-layer + minmod)
        z3(), z3(), z3(), z3(), z3(), z3(), z3(), z3(), z3(), z3(), z3(), z3(),  # gux..gTz (ghosted Green-Gauss grads)
        backend)
end

function set_freestream_3d!(s::Span3D, mach::Float64, alpha_deg::Float64)
    a = sqrt((1 + 2/(3+s.drot)) * CS2)
    U = Float32(mach*a)
    s.ca = Float32(cosd(alpha_deg)); s.sa = Float32(sind(alpha_deg))
    s.uxi = U*s.ca; s.uyi = U*s.sa; s.rho_inf = 1.0f0; s.T_inf = 1.0f0
    fill!(s.rho, s.rho_inf); fill!(s.ux, s.uxi); fill!(s.uy, s.uyi); fill!(s.uz, 0.0f0); fill!(s.T, s.T_inf)
    nothing
end

# ★ ORDERING GUARD (2026-07-26). w_inf is defined by the freestream eddy-viscosity RATIO
# nu_t_inf = mu_t_ratio*nu_lam, so nu_lam MUST already be set when this is called. It was NOT in the
# production 3D shedding driver (set_sst_3d! ran before s.nu_lam = U/RE), which divided by the
# constructor default nu_lam = 0 and silently produced w_inf = Inf => nu_t_inf = 0, i.e. the far-field
# BC re-imposed omega = Inf on the j-ghosts EVERY sweep. Measured (tools/probe_sst_parity.jl): 387560
# cells still hold Inf after 300 LTS steps and peak nu_t is 86x lower than the correctly-initialised
# run. The 2D twin (set_sst_ka!, cmesh_shed_topology.jl:144) always ordered this correctly, so this was
# a live 2D-vs-3D parity break in the turbulence model — exactly the class the 3D-shedding handover's
# step-1 audit was hunting. Hard error: a silent Inf is far more expensive than a stopped script.
function set_sst_3d!(s::Span3D; tu_inf::Float64=0.001, mu_t_ratio::Float64=0.1)
    s.nu_lam > 0.0f0 || error("set_sst_3d!: s.nu_lam must be set (>0) BEFORE this call — w_inf = " *
                              "k_inf/(mu_t_ratio*nu_lam) is Inf otherwise (got nu_lam=$(s.nu_lam)). " *
                              "Assign s.nu_lam = U/Re first, then call set_sst_3d!.")
    U = sqrt(Float64(s.uxi)^2 + Float64(s.uyi)^2)
    s.k_inf = Float32(1.5*(tu_inf*U)^2)
    s.w_inf = Float32(max(s.k_inf/(mu_t_ratio*Float64(s.nu_lam)), 1e-6))
    fill!(s.tk, s.k_inf); fill!(s.tom, s.w_inf); fill!(s.nut, s.k_inf/s.w_inf)
    s.sst = true
    nothing
end

# ---- BCs --------------------------------------------------------------------------------------------
@kernel function _k_bc_wall_3d!(rho, ux, uy, uz, T, @Const(Sjx), @Const(Sjy),
                                NI::Int32, NIW::Int32, NIF::Int32, g::Int32)
    i, kk, k = @index(Global, NTuple)              # i in 1..NI, kk in 1..g (ghost layer), k in 1..NK plane
    @inbounds if i <= NI && kk <= g
        ka = k+g
        if i <= NIW || i > NIW+NIF                  # wake cut: partner exchange
            ip = NI+1-i
            rho[i+g, g+1-kk, ka] = rho[ip+g, kk+g, ka]; ux[i+g, g+1-kk, ka] = ux[ip+g, kk+g, ka]
            uy[i+g, g+1-kk, ka]  = uy[ip+g, kk+g, ka];  uz[i+g, g+1-kk, ka] = uz[ip+g, kk+g, ka]
            T[i+g, g+1-kk, ka]   = T[ip+g, kk+g, ka]
        else                                        # noslip wall: rho/T mirror, velocity antisymmetric
            rho[i+g, g+1-kk, ka] = rho[i+g, g+kk, ka]; ux[i+g, g+1-kk, ka] = -ux[i+g, g+kk, ka]
            uy[i+g, g+1-kk, ka]  = -uy[i+g, g+kk, ka]; uz[i+g, g+1-kk, ka] = -uz[i+g, g+kk, ka]
            T[i+g, g+1-kk, ka]   = T[i+g, g+kk, ka]
        end
    end
end

@kernel function _k_bc_farfield_3d!(rho, ux, uy, uz, T, @Const(Sjx), @Const(Sjy), @Const(pexc),
                                    rho_inf::Float32, uxi::Float32, uyi::Float32, T_inf::Float32,
                                    drot::Float32, ff_mix::Int32, ff_p_ref::Float32, ff_a2m::Float32,
                                    ff_Tmin::Float32, ff_Tmax::Float32,
                                    NI::Int32, NJ::Int32, g::Int32)
    i, k = @index(Global, NTuple)
    @inbounds if i <= NI
        ka = k+g
        gam_eos = (5.0f0 + drot)/(3.0f0 + drot); gm1 = gam_eos - 1.0f0; Rgas = CS2
        Sx = Sjx[i,NJ+1]; Sy = Sjy[i,NJ+1]; Sm = sqrt(Sx*Sx+Sy*Sy)
        nxb = Sx/Sm; nyb = Sy/Sm; txb = -nyb; tyb = nxb
        rd = rho[i+g, NJ+g, ka]; uxd = ux[i+g, NJ+g, ka]; uyd = uy[i+g, NJ+g, ka]; Td = T[i+g, NJ+g, ka]
        cd_ = sqrt(gam_eos*Rgas*max(Td, 1.0f-4)); cinf = sqrt(gam_eos*Rgas*T_inf)
        qnd = uxd*nxb + uyd*nyb; qni = uxi*nxb + uyi*nyb
        rb = rd; uxb = uxd; uyb = uyd; Tb = Td
        if ff_mix == Int32(1)
            # ★ BAROTROPIC MIXTURE FAR FIELD (2D STAGE-1e twin — see cmesh_solver_ka.jl for the full
            # rationale): under PDP the carrier T is a railed pedestal with no thermodynamic content;
            # the correct invariants are the linear-acoustic pair about the MIXTURE ambient.
            am_r = sqrt(ff_a2m); Zc = rho_inf*am_r
            p_d = rd*CS2*Td + pexc[i+g, NJ+g, ka]
            if qnd <= -am_r                                  # supersonic inflow: impose ambient
                rb = rho_inf; uxb = uxi; uyb = uyi
                Tbb = ff_p_ref/(rho_inf*CS2)
                Tb = Tbb < ff_Tmin ? ff_Tmin : (Tbb > ff_Tmax ? ff_Tmax : Tbb)
            elseif qnd < am_r                                # subsonic: 1 out + 1 in
                Rout = qnd + p_d/Zc
                Rin  = qni - ff_p_ref/Zc
                qnb  = 0.5f0*(Rout + Rin)
                p_b  = 0.5f0*Zc*(Rout - Rin)
                p_b  = p_b < 1.0f-8 ? 1.0f-8 : p_b
                rbb  = rho_inf + (p_b - ff_p_ref)/ff_a2m
                rb   = rbb < 1.0f-4 ? 1.0f-4 : rbb
                utb  = qnb >= 0.0f0 ? (uxd*txb + uyd*tyb) : (uxi*txb + uyi*tyb)
                uxb  = qnb*nxb + utb*txb; uyb = qnb*nyb + utb*tyb
                Tbb  = p_b/(rb*CS2)
                Tb   = Tbb < ff_Tmin ? ff_Tmin : (Tbb > ff_Tmax ? ff_Tmax : Tbb)
            end
            # supersonic outflow: extrapolate (keep interior rb/uxb/uyb/Tb)
        elseif qnd <= -cd_
            rb = rho_inf; uxb = uxi; uyb = uyi; Tb = T_inf
        elseif qnd < cd_
            Rout = qnd + 2.0f0*cd_/gm1; Rin = qni - 2.0f0*cinf/gm1
            qnb = 0.5f0*(Rout+Rin); cb = 0.25f0*gm1*(Rout-Rin); cb = cb < 1.0f-4 ? 1.0f-4 : cb
            sB = 0.0f0; utb = 0.0f0
            if qnb >= 0.0f0
                sB = (rd*Rgas*Td)/(rd^gam_eos); utb = uxd*txb + uyd*tyb
            else
                sB = (rho_inf*Rgas*T_inf)/(rho_inf^gam_eos); utb = uxi*txb + uyi*tyb
            end
            rb = (cb*cb/(gam_eos*sB))^(1.0f0/gm1); Tb = (sB*rb^gam_eos)/(rb*Rgas)
            uxb = qnb*nxb + utb*txb; uyb = qnb*nyb + utb*tyb
        end
        for kk in 1:g
            rho[i+g, NJ+g+kk, ka] = rb; ux[i+g, NJ+g+kk, ka] = uxb
            uy[i+g, NJ+g+kk, ka] = uyb; uz[i+g, NJ+g+kk, ka] = 0.0f0; T[i+g, NJ+g+kk, ka] = Tb
        end
    end
end

@kernel function _k_bc_io_3d!(rho, ux, uy, uz, T, NI::Int32, NJ::Int32, g::Int32)
    jj, kk, k = @index(Global, NTuple)             # jj over all rows incl ghosts, kk in 1..g, k in 1..NK
    @inbounds if jj <= NJ+2g && kk <= g
        ka = k+g
        rho[g+1-kk, jj, ka] = rho[g+1, jj, ka]; ux[g+1-kk, jj, ka] = ux[g+1, jj, ka]
        uy[g+1-kk, jj, ka]  = uy[g+1, jj, ka];  uz[g+1-kk, jj, ka] = uz[g+1, jj, ka]; T[g+1-kk, jj, ka] = T[g+1, jj, ka]
        rho[NI+g+kk, jj, ka] = rho[NI+g, jj, ka]; ux[NI+g+kk, jj, ka] = ux[NI+g, jj, ka]
        uy[NI+g+kk, jj, ka]  = uy[NI+g, jj, ka]; uz[NI+g+kk, jj, ka] = uz[NI+g, jj, ka]; T[NI+g+kk, jj, ka] = T[NI+g, jj, ka]
    end
end

# spanwise periodic ghost planes (copies full i-j plane incl i/j ghosts)
@kernel function _k_bc_span_3d!(rho, ux, uy, uz, T, NX::Int32, NY::Int32, NK::Int32, g::Int32)
    i, j, kk = @index(Global, NTuple)
    @inbounds if i <= NX && j <= NY && kk <= g
        rho[i,j,g+1-kk] = rho[i,j,g+NK+1-kk]; ux[i,j,g+1-kk] = ux[i,j,g+NK+1-kk]
        uy[i,j,g+1-kk]  = uy[i,j,g+NK+1-kk];  uz[i,j,g+1-kk] = uz[i,j,g+NK+1-kk]; T[i,j,g+1-kk] = T[i,j,g+NK+1-kk]
        rho[i,j,g+NK+kk] = rho[i,j,g+kk]; ux[i,j,g+NK+kk] = ux[i,j,g+kk]
        uy[i,j,g+NK+kk]  = uy[i,j,g+kk];  uz[i,j,g+NK+kk] = uz[i,j,g+kk]; T[i,j,g+NK+kk] = T[i,j,g+kk]
    end
end

"""Tier-C mixture far-field (2D set_farfield_mixture! twin): hold the ambient at the MIXTURE
reference (p_ref, a2m from tem_ambient3d) with barotropic characteristics + the T pedestal band."""
function set_farfield_mixture3d!(s::Span3D; p_ref::Float64, a2m::Float64,
                                 Tmin::Float32=0.3f0, Tmax::Float32=3.0f0)
    a2m > 0.0 || error("set_farfield_mixture3d!: a2m must be > 0")
    p_ref > 0.0 || error("set_farfield_mixture3d!: p_ref must be > 0")
    s.ff_mix = Int32(1); s.ff_p_ref = Float32(p_ref); s.ff_a2m = Float32(a2m)
    s.ff_Tmin = Tmin; s.ff_Tmax = Tmax
    nothing
end

function applyBC_3d!(s::Span3D)
    b = s.backend; NI, NJ, NK, g = Int32(s.NI), Int32(s.NJ), Int32(s.NK), Int32(s.g)
    NX, NY = Int32(s.NI+2s.g), Int32(s.NJ+2s.g)
    _k_bc_wall_3d!(b, WG3)(s.rho, s.ux, s.uy, s.uz, s.T, s.Sjx, s.Sjy, NI, Int32(s.NIW), Int32(s.NIF), g; ndrange=(s.NI, s.g, s.NK))
    _k_bc_farfield_3d!(b, (64,1))(s.rho, s.ux, s.uy, s.uz, s.T, s.Sjx, s.Sjy, s.pexc,
                       s.rho_inf, s.uxi, s.uyi, s.T_inf, s.drot,
                       s.ff_mix, s.ff_p_ref, s.ff_a2m, s.ff_Tmin, s.ff_Tmax,
                       NI, NJ, g; ndrange=(s.NI, s.NK))
    _k_bc_io_3d!(b, WG3)(s.rho, s.ux, s.uy, s.uz, s.T, NI, NJ, g; ndrange=(s.NJ+2s.g, s.g, s.NK))
    _k_bc_span_3d!(b, WG3)(s.rho, s.ux, s.uy, s.uz, s.T, NX, NY, NK, g; ndrange=(s.NI+2s.g, s.NJ+2s.g, s.g))
    nothing
end

# ---- Green-Gauss cell gradients (F2 fullvisc prerequisite) ------------------------------------------
# x,y via the shared 2D metrics per k-plane (same _gg_grad as the 2D solver); z via central difference
# (uniform dz). Writes GHOSTED [ia,ja,ka] at interior cells; the k-ghosts are filled periodically by
# _k_grad_kghost_3d! so the k-face reads at k=1 / k=NK+1 land on the correct span images. Ghost primitives
# must be filled (applyBC_3d!) before this runs — they are, in-stream, at each RK substage.
@kernel function _k_gradients_3d!(gux,guy,guz, gvx,gvy,gvz, gwx,gwy,gwz, gTx,gTy,gTz,
                                  @Const(ux),@Const(uy),@Const(uz),@Const(T),
                                  @Const(Six),@Const(Siy),@Const(Sjx),@Const(Sjy),@Const(vol),
                                  dz::Float32, NI::Int32, NJ::Int32, NK::Int32, g::Int32)
    i,j,k = @index(Global, NTuple)
    @inbounds if i<=NI && j<=NJ && k<=NK
        ia=i+g; ja=j+g; ka=k+g; iv=1.0f0/vol[i,j]; izz=0.5f0/dz
        SxW=Six[i,j];SyW=Siy[i,j];SxE=Six[i+1,j];SyE=Siy[i+1,j]
        SxS=Sjx[i,j];SyS=Sjy[i,j];SxN=Sjx[i,j+1];SyN=Sjy[i,j+1]
        r=_gg_grad(ux[ia-1,ja,ka],ux[ia+1,ja,ka],ux[ia,ja-1,ka],ux[ia,ja+1,ka],ux[ia,ja,ka],SxW,SyW,SxE,SyE,SxS,SyS,SxN,SyN,iv)
        gux[ia,ja,ka]=r[1]; guy[ia,ja,ka]=r[2]; guz[ia,ja,ka]=izz*(ux[ia,ja,ka+1]-ux[ia,ja,ka-1])
        r=_gg_grad(uy[ia-1,ja,ka],uy[ia+1,ja,ka],uy[ia,ja-1,ka],uy[ia,ja+1,ka],uy[ia,ja,ka],SxW,SyW,SxE,SyE,SxS,SyS,SxN,SyN,iv)
        gvx[ia,ja,ka]=r[1]; gvy[ia,ja,ka]=r[2]; gvz[ia,ja,ka]=izz*(uy[ia,ja,ka+1]-uy[ia,ja,ka-1])
        r=_gg_grad(uz[ia-1,ja,ka],uz[ia+1,ja,ka],uz[ia,ja-1,ka],uz[ia,ja+1,ka],uz[ia,ja,ka],SxW,SyW,SxE,SyE,SxS,SyS,SxN,SyN,iv)
        gwx[ia,ja,ka]=r[1]; gwy[ia,ja,ka]=r[2]; gwz[ia,ja,ka]=izz*(uz[ia,ja,ka+1]-uz[ia,ja,ka-1])
        r=_gg_grad(T[ia-1,ja,ka],T[ia+1,ja,ka],T[ia,ja-1,ka],T[ia,ja+1,ka],T[ia,ja,ka],SxW,SyW,SxE,SyE,SxS,SyS,SxN,SyN,iv)
        gTx[ia,ja,ka]=r[1]; gTy[ia,ja,ka]=r[2]; gTz[ia,ja,ka]=izz*(T[ia,ja,ka+1]-T[ia,ja,ka-1])
    end
end

# periodic-k ghost fill of the 12 gradient arrays (mirrors _k_sst_span_3d!) so the k-face fullvisc reads land.
@kernel function _k_grad_kghost_3d!(gux,guy,guz, gvx,gvy,gvz, gwx,gwy,gwz, gTx,gTy,gTz,
                                    NX::Int32, NY::Int32, NK::Int32, g::Int32)
    i,j,kk = @index(Global, NTuple)
    @inbounds if i<=NX && j<=NY && kk<=g
        lo=g+1-kk; slo=g+NK+1-kk; hi=g+NK+kk; shi=g+kk
        gux[i,j,lo]=gux[i,j,slo]; gux[i,j,hi]=gux[i,j,shi]; guy[i,j,lo]=guy[i,j,slo]; guy[i,j,hi]=guy[i,j,shi]; guz[i,j,lo]=guz[i,j,slo]; guz[i,j,hi]=guz[i,j,shi]
        gvx[i,j,lo]=gvx[i,j,slo]; gvx[i,j,hi]=gvx[i,j,shi]; gvy[i,j,lo]=gvy[i,j,slo]; gvy[i,j,hi]=gvy[i,j,shi]; gvz[i,j,lo]=gvz[i,j,slo]; gvz[i,j,hi]=gvz[i,j,shi]
        gwx[i,j,lo]=gwx[i,j,slo]; gwx[i,j,hi]=gwx[i,j,shi]; gwy[i,j,lo]=gwy[i,j,slo]; gwy[i,j,hi]=gwy[i,j,shi]; gwz[i,j,lo]=gwz[i,j,slo]; gwz[i,j,hi]=gwz[i,j,shi]
        gTx[i,j,lo]=gTx[i,j,slo]; gTx[i,j,hi]=gTx[i,j,shi]; gTy[i,j,lo]=gTy[i,j,slo]; gTy[i,j,hi]=gTy[i,j,shi]; gTz[i,j,lo]=gTz[i,j,slo]; gTz[i,j,hi]=gTz[i,j,shi]
    end
end

# ---- flux kernels ----------------------------------------------------------------------------------
@kernel function _k_flux_i_3d!(Fr, Fmx, Fmy, Fmz, Fe, @Const(rho), @Const(ux), @Const(uy), @Const(uz), @Const(T),
                               @Const(nut), @Const(Six), @Const(Siy), @Const(xc), @Const(yc), @Const(cxa), @Const(cya), @Const(cza), @Const(wa),
                               @Const(gux),@Const(guy),@Const(guz),@Const(gvx),@Const(gvy),@Const(gvz),@Const(gwx),@Const(gwy),@Const(gwz),@Const(gTx),@Const(gTy),
                               drot::Float32, nu_lam::Float32, sst_on::Int32, fullvisc::Int32, recon::Int32, @Const(pexc), @Const(am), cav::Int32,
                               NI::Int32, NJ::Int32, NK::Int32, g::Int32)
    i, j, k = @index(Global, NTuple)
    @inbounds if i <= NI+1 && j <= NJ && k <= NK
        nxr=Six[i,j]; nyr=Siy[i,j]; S=sqrt(nxr*nxr+nyr*nyr); nx=nxr/S; ny=nyr/S
        ia=i+g; ja=j+g; ka=k+g
        fr,fmx,fmy,fmz,fe = _fvflux_ij_3d(
            rho[ia-3,ja,ka],ux[ia-3,ja,ka],uy[ia-3,ja,ka],uz[ia-3,ja,ka],T[ia-3,ja,ka],
            rho[ia-2,ja,ka],ux[ia-2,ja,ka],uy[ia-2,ja,ka],uz[ia-2,ja,ka],T[ia-2,ja,ka],
            rho[ia-1,ja,ka],ux[ia-1,ja,ka],uy[ia-1,ja,ka],uz[ia-1,ja,ka],T[ia-1,ja,ka],
            rho[ia,  ja,ka],ux[ia,  ja,ka],uy[ia,  ja,ka],uz[ia,  ja,ka],T[ia,  ja,ka],
            rho[ia+1,ja,ka],ux[ia+1,ja,ka],uy[ia+1,ja,ka],uz[ia+1,ja,ka],T[ia+1,ja,ka],
            rho[ia+2,ja,ka],ux[ia+2,ja,ka],uy[ia+2,ja,ka],uz[ia+2,ja,ka],T[ia+2,ja,ka],
            nx, ny, S, cxa, cya, cza, wa, drot, recon)
        if nu_lam > 0.0f0 && i >= 2 && i <= NI       # laminar + SST eddy viscosity (interior i-faces)
            rf=0.5f0*(rho[ia-1,ja,ka]+rho[ia,ja,ka]); cpg=0.5f0*(5.0f0+drot)*CS2
            nut_f = sst_on == Int32(1) ? 0.5f0*(nut[ia-1,ja,ka]+nut[ia,ja,ka]) : 0.0f0
            dxc=xc[i,j]-xc[i-1,j]; dyc=yc[i,j]-yc[i-1,j]
            if fullvisc == Int32(1)                  # F2: full 3D stress tensor (adds tau_xy, tau_xz, tau_yz)
                mueff=rf*(nu_lam+nut_f); keff=rf*cpg*(nu_lam/0.71f0+nut_f/0.9f0)
                Fvx,Fvy,Fvz,Fve = _visc_face_ij_3d(
                    gux[ia-1,ja,ka],guy[ia-1,ja,ka],guz[ia-1,ja,ka],gvx[ia-1,ja,ka],gvy[ia-1,ja,ka],gvz[ia-1,ja,ka],gwx[ia-1,ja,ka],gwy[ia-1,ja,ka],gwz[ia-1,ja,ka],gTx[ia-1,ja,ka],gTy[ia-1,ja,ka],
                    gux[ia,ja,ka],guy[ia,ja,ka],guz[ia,ja,ka],gvx[ia,ja,ka],gvy[ia,ja,ka],gvz[ia,ja,ka],gwx[ia,ja,ka],gwy[ia,ja,ka],gwz[ia,ja,ka],gTx[ia,ja,ka],gTy[ia,ja,ka],
                    ux[ia-1,ja,ka],uy[ia-1,ja,ka],uz[ia-1,ja,ka],T[ia-1,ja,ka], ux[ia,ja,ka],uy[ia,ja,ka],uz[ia,ja,ka],T[ia,ja,ka],
                    dxc,dyc, nx,ny,S, mueff,keff)
                fmx-=Fvx; fmy-=Fvy; fmz-=Fvz; fe-=Fve
            else                                     # thin-layer (bitwise legacy)
                dn=sqrt(dxc*dxc+dyc*dyc)+1.0f-30; mu=rf*(nu_lam+nut_f)*S/dn
                du=ux[ia,ja,ka]-ux[ia-1,ja,ka]; dv=uy[ia,ja,ka]-uy[ia-1,ja,ka]; dw=uz[ia,ja,ka]-uz[ia-1,ja,ka]
                dT=T[ia,ja,ka]-T[ia-1,ja,ka]
                ufx=0.5f0*(ux[ia-1,ja,ka]+ux[ia,ja,ka]); ufy=0.5f0*(uy[ia-1,ja,ka]+uy[ia,ja,ka]); ufz=0.5f0*(uz[ia-1,ja,ka]+uz[ia,ja,ka])
                fmx-=mu*du; fmy-=mu*dv; fmz-=mu*dw
                fe -= mu*(ufx*du+ufy*dv+ufz*dw) + rf*cpg*(nu_lam/0.71f0 + nut_f/0.9f0)*S/dn*dT
            end
        end
        if cav >= Int32(1)                          # PDP pressure-split: excess p_mix as a face force (GCL-conservative)
            pexf = 0.5f0*(pexc[ia-1,ja,ka] + pexc[ia,ja,ka])
            fmx += pexf*nxr; fmy += pexf*nyr
            ufx2=0.5f0*(ux[ia-1,ja,ka]+ux[ia,ja,ka]); ufy2=0.5f0*(uy[ia-1,ja,ka]+uy[ia,ja,ka])
            fe  += pexf*(ufx2*nxr + ufy2*nyr)
        end
        if cav >= Int32(2)                          # Layer C: preconditioned a_m AUSM+-up
            pL = rho[ia-1,ja,ka]*CS2*T[ia-1,ja,ka] + pexc[ia-1,ja,ka]
            pR = rho[ia,  ja,ka]*CS2*T[ia,  ja,ka] + pexc[ia,  ja,ka]
            unL = ux[ia-1,ja,ka]*nx + uy[ia-1,ja,ka]*ny; unR = ux[ia,ja,ka]*nx + uy[ia,ja,ka]*ny
            amf = 0.5f0*(am[ia-1,ja,ka] + am[ia,ja,ka])
            amp = max(amf, 0.5f0*abs(unL+unR), 0.05f0)
            fr  += -0.25f0*(pR - pL)/amp*S
            rf2  = 0.5f0*(rho[ia-1,ja,ka]+rho[ia,ja,ka])
            pu   = -0.5f0*rf2*amp*(unR - unL)
            fmx += pu*nxr; fmy += pu*nyr
        end
        Fr[i,j,k]=fr; Fmx[i,j,k]=fmx; Fmy[i,j,k]=fmy; Fmz[i,j,k]=fmz; Fe[i,j,k]=fe
    end
end

@kernel function _k_flux_j_3d!(Fr, Fmx, Fmy, Fmz, Fe, @Const(rho), @Const(ux), @Const(uy), @Const(uz), @Const(T),
                               @Const(nut), @Const(Sjx), @Const(Sjy), @Const(xc), @Const(yc), @Const(vol),
                               @Const(cxa), @Const(cya), @Const(cza), @Const(wa),
                               @Const(gux),@Const(guy),@Const(guz),@Const(gvx),@Const(gvy),@Const(gvz),@Const(gwx),@Const(gwy),@Const(gwz),@Const(gTx),@Const(gTy),
                               drot::Float32, nu_lam::Float32, sst_on::Int32, wallflux::Int32, fullvisc::Int32, recon::Int32, @Const(pexc), @Const(am), cav::Int32,
                               NI::Int32, NJ::Int32, NK::Int32, NIW::Int32, NIF::Int32, g::Int32)
    i, j, k = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ+1 && k <= NK
        nxr=Sjx[i,j]; nyr=Sjy[i,j]; S=sqrt(nxr*nxr+nyr*nyr); nx=nxr/S; ny=nyr/S
        ia=i+g; ja=j+g; ka=k+g
        if wallflux == Int32(1) && j == 1 && i > NIW && i <= NIW+NIF
            pw = rho[ia,g+1,ka]*CS2*T[ia,g+1,ka] + (cav >= Int32(1) ? pexc[ia,g+1,ka] : 0.0f0)   # PDP: wall carries p_mix
            fr=0.0f0; fmx=pw*nxr; fmy=pw*nyr; fmz=0.0f0; fe=0.0f0
            if nu_lam > 0.0f0
                d1=0.5f0*vol[i,1]/S+1.0f-30; muw=rho[ia,g+1,ka]*nu_lam*S/d1
                fmx-=muw*ux[ia,g+1,ka]; fmy-=muw*uy[ia,g+1,ka]; fmz-=muw*uz[ia,g+1,ka]
            end
            Fr[i,j,k]=fr; Fmx[i,j,k]=fmx; Fmy[i,j,k]=fmy; Fmz[i,j,k]=fmz; Fe[i,j,k]=fe
        else
            fr,fmx,fmy,fmz,fe = _fvflux_ij_3d(
                rho[ia,ja-3,ka],ux[ia,ja-3,ka],uy[ia,ja-3,ka],uz[ia,ja-3,ka],T[ia,ja-3,ka],
                rho[ia,ja-2,ka],ux[ia,ja-2,ka],uy[ia,ja-2,ka],uz[ia,ja-2,ka],T[ia,ja-2,ka],
                rho[ia,ja-1,ka],ux[ia,ja-1,ka],uy[ia,ja-1,ka],uz[ia,ja-1,ka],T[ia,ja-1,ka],
                rho[ia,ja,  ka],ux[ia,ja,  ka],uy[ia,ja,  ka],uz[ia,ja,  ka],T[ia,ja,  ka],
                rho[ia,ja+1,ka],ux[ia,ja+1,ka],uy[ia,ja+1,ka],uz[ia,ja+1,ka],T[ia,ja+1,ka],
                rho[ia,ja+2,ka],ux[ia,ja+2,ka],uy[ia,ja+2,ka],uz[ia,ja+2,ka],T[ia,ja+2,ka],
                nx, ny, S, cxa, cya, cza, wa, drot, recon)
            if nu_lam > 0.0f0 && ((j >= 2 && j <= NJ) || (j == 1 && (i <= NIW || i > NIW+NIF)))
                rf=0.5f0*(rho[ia,ja-1,ka]+rho[ia,ja,ka]); cpg=0.5f0*(5.0f0+drot)*CS2
                nut_f = sst_on == Int32(1) ? 0.5f0*(nut[ia,ja-1,ka]+nut[ia,ja,ka]) : 0.0f0
                if fullvisc == Int32(1) && j >= 2    # F2 full stress (interior j-faces only; j==1 cut stays thin-layer)
                    mueff=rf*(nu_lam+nut_f); keff=rf*cpg*(nu_lam/0.71f0+nut_f/0.9f0)
                    dxc=xc[i,j]-xc[i,j-1]; dyc=yc[i,j]-yc[i,j-1]
                    Fvx,Fvy,Fvz,Fve = _visc_face_ij_3d(
                        gux[ia,ja-1,ka],guy[ia,ja-1,ka],guz[ia,ja-1,ka],gvx[ia,ja-1,ka],gvy[ia,ja-1,ka],gvz[ia,ja-1,ka],gwx[ia,ja-1,ka],gwy[ia,ja-1,ka],gwz[ia,ja-1,ka],gTx[ia,ja-1,ka],gTy[ia,ja-1,ka],
                        gux[ia,ja,ka],guy[ia,ja,ka],guz[ia,ja,ka],gvx[ia,ja,ka],gvy[ia,ja,ka],gvz[ia,ja,ka],gwx[ia,ja,ka],gwy[ia,ja,ka],gwz[ia,ja,ka],gTx[ia,ja,ka],gTy[ia,ja,ka],
                        ux[ia,ja-1,ka],uy[ia,ja-1,ka],uz[ia,ja-1,ka],T[ia,ja-1,ka], ux[ia,ja,ka],uy[ia,ja,ka],uz[ia,ja,ka],T[ia,ja,ka],
                        dxc,dyc, nx,ny,S, mueff,keff)
                    fmx-=Fvx; fmy-=Fvy; fmz-=Fvz; fe-=Fve
                else                                 # thin-layer (wall/cut face, or fullvisc off)
                    local dn::Float32
                    if j == 1; dn = vol[i,1]/S + 1.0f-30; else
                        dxc=xc[i,j]-xc[i,j-1]; dyc=yc[i,j]-yc[i,j-1]; dn=sqrt(dxc*dxc+dyc*dyc)+1.0f-30; end
                    mu=rf*(nu_lam+nut_f)*S/dn
                    du=ux[ia,ja,ka]-ux[ia,ja-1,ka]; dv=uy[ia,ja,ka]-uy[ia,ja-1,ka]; dw=uz[ia,ja,ka]-uz[ia,ja-1,ka]; dT=T[ia,ja,ka]-T[ia,ja-1,ka]
                    ufx=0.5f0*(ux[ia,ja-1,ka]+ux[ia,ja,ka]); ufy=0.5f0*(uy[ia,ja-1,ka]+uy[ia,ja,ka]); ufz=0.5f0*(uz[ia,ja-1,ka]+uz[ia,ja,ka])
                    fmx-=mu*du; fmy-=mu*dv; fmz-=mu*dw
                    fe -= mu*(ufx*du+ufy*dv+ufz*dw) + rf*cpg*(nu_lam/0.71f0 + nut_f/0.9f0)*S/dn*dT
                end
            end
            if cav >= Int32(1)                      # PDP pressure-split
                pexf = 0.5f0*(pexc[ia,ja-1,ka] + pexc[ia,ja,ka])
                fmx += pexf*nxr; fmy += pexf*nyr
                ufx2=0.5f0*(ux[ia,ja-1,ka]+ux[ia,ja,ka]); ufy2=0.5f0*(uy[ia,ja-1,ka]+uy[ia,ja,ka])
                fe  += pexf*(ufx2*nxr + ufy2*nyr)
            end
            if cav >= Int32(2)                      # Layer C
                pL = rho[ia,ja-1,ka]*CS2*T[ia,ja-1,ka] + pexc[ia,ja-1,ka]
                pR = rho[ia,ja,  ka]*CS2*T[ia,ja,  ka] + pexc[ia,ja,  ka]
                unL = ux[ia,ja-1,ka]*nx + uy[ia,ja-1,ka]*ny; unR = ux[ia,ja,ka]*nx + uy[ia,ja,ka]*ny
                amf = 0.5f0*(am[ia,ja-1,ka] + am[ia,ja,ka])
                amp = max(amf, 0.5f0*abs(unL+unR), 0.05f0)
                fr  += -0.25f0*(pR - pL)/amp*S
                rf2  = 0.5f0*(rho[ia,ja-1,ka]+rho[ia,ja,ka])
                pu   = -0.5f0*rf2*amp*(unR - unL)
                fmx += pu*nxr; fmy += pu*nyr
            end
            Fr[i,j,k]=fr; Fmx[i,j,k]=fmx; Fmy[i,j,k]=fmy; Fmz[i,j,k]=fmz; Fe[i,j,k]=fe
        end
    end
end

@kernel function _k_flux_k_3d!(Fr, Fmx, Fmy, Fmz, Fe, @Const(rho), @Const(ux), @Const(uy), @Const(uz), @Const(T),
                               @Const(nut), @Const(cxa), @Const(cya), @Const(cza), @Const(wa),
                               @Const(gux),@Const(gvy),@Const(gwx),@Const(gwy),@Const(guz),@Const(gvz),@Const(gwz),@Const(gTz),
                               drot::Float32, nu_lam::Float32, sst_on::Int32, fullvisc::Int32, recon::Int32, dz::Float32, @Const(pexc), @Const(am), cav::Int32,
                               NI::Int32, NJ::Int32, NK::Int32, g::Int32)
    i, j, k = @index(Global, NTuple)                # k in 1..NK+1 (spanwise faces)
    @inbounds if i <= NI && j <= NJ && k <= NK+1
        ia=i+g; ja=j+g; ka=k+g                       # face between plane k-1 (ka-1) and k (ka)
        fr,fmx,fmy,fmz,fe = _fvflux_k_3d(
            rho[ia,ja,ka-3],ux[ia,ja,ka-3],uy[ia,ja,ka-3],uz[ia,ja,ka-3],T[ia,ja,ka-3],
            rho[ia,ja,ka-2],ux[ia,ja,ka-2],uy[ia,ja,ka-2],uz[ia,ja,ka-2],T[ia,ja,ka-2],
            rho[ia,ja,ka-1],ux[ia,ja,ka-1],uy[ia,ja,ka-1],uz[ia,ja,ka-1],T[ia,ja,ka-1],
            rho[ia,ja,ka  ],ux[ia,ja,ka  ],uy[ia,ja,ka  ],uz[ia,ja,ka  ],T[ia,ja,ka  ],
            rho[ia,ja,ka+1],ux[ia,ja,ka+1],uy[ia,ja,ka+1],uz[ia,ja,ka+1],T[ia,ja,ka+1],
            rho[ia,ja,ka+2],ux[ia,ja,ka+2],uy[ia,ja,ka+2],uz[ia,ja,ka+2],T[ia,ja,ka+2],
            cxa, cya, cza, wa, drot, recon)
        if nu_lam > 0.0f0                            # laminar + SST eddy viscosity in z (uniform dz)
            rf=0.5f0*(rho[ia,ja,ka-1]+rho[ia,ja,ka]); cpg=0.5f0*(5.0f0+drot)*CS2
            nut_f = sst_on == Int32(1) ? 0.5f0*(nut[ia,ja,ka-1]+nut[ia,ja,ka]) : 0.0f0
            if fullvisc == Int32(1)                  # F2 full stress in z (tau_xz, tau_yz, tau_zz)
                mueff=rf*(nu_lam+nut_f); keff=rf*cpg*(nu_lam/0.71f0+nut_f/0.9f0)
                Fvx,Fvy,Fvz,Fve = _visc_face_k_3d(
                    gux[ia,ja,ka-1],gvy[ia,ja,ka-1],gwx[ia,ja,ka-1],gwy[ia,ja,ka-1],guz[ia,ja,ka-1],gvz[ia,ja,ka-1],gwz[ia,ja,ka-1],gTz[ia,ja,ka-1],
                    gux[ia,ja,ka],gvy[ia,ja,ka],gwx[ia,ja,ka],gwy[ia,ja,ka],guz[ia,ja,ka],gvz[ia,ja,ka],gwz[ia,ja,ka],gTz[ia,ja,ka],
                    ux[ia,ja,ka-1],uy[ia,ja,ka-1],uz[ia,ja,ka-1],T[ia,ja,ka-1], ux[ia,ja,ka],uy[ia,ja,ka],uz[ia,ja,ka],T[ia,ja,ka],
                    dz, mueff, keff)
                fmx-=Fvx; fmy-=Fvy; fmz-=Fvz; fe-=Fve
            else                                     # thin-layer (bitwise legacy)
                mu=rf*(nu_lam+nut_f)/dz
                du=ux[ia,ja,ka]-ux[ia,ja,ka-1]; dv=uy[ia,ja,ka]-uy[ia,ja,ka-1]; dw=uz[ia,ja,ka]-uz[ia,ja,ka-1]; dT=T[ia,ja,ka]-T[ia,ja,ka-1]
                ufx=0.5f0*(ux[ia,ja,ka-1]+ux[ia,ja,ka]); ufy=0.5f0*(uy[ia,ja,ka-1]+uy[ia,ja,ka]); ufz=0.5f0*(uz[ia,ja,ka-1]+uz[ia,ja,ka])
                fmx-=mu*du; fmy-=mu*dv; fmz-=mu*dw
                fe -= mu*(ufx*du+ufy*dv+ufz*dw) + rf*cpg*(nu_lam/0.71f0 + nut_f/0.9f0)/dz*dT
            end
        end
        # PDP on the SPANWISE faces. NOTE: the k-flux is PER-UNIT-AREA (the residual uses b=dt/dz, not
        # dt/vol), so the excess pressure enters fmz directly with NO area factor.
        if cav >= Int32(1)
            pexf = 0.5f0*(pexc[ia,ja,ka-1] + pexc[ia,ja,ka])
            fmz += pexf
            fe  += pexf*0.5f0*(uz[ia,ja,ka-1]+uz[ia,ja,ka])
        end
        if cav >= Int32(2)                          # Layer C in z
            pL = rho[ia,ja,ka-1]*CS2*T[ia,ja,ka-1] + pexc[ia,ja,ka-1]
            pR = rho[ia,ja,ka  ]*CS2*T[ia,ja,ka  ] + pexc[ia,ja,ka  ]
            wLz = uz[ia,ja,ka-1]; wRz = uz[ia,ja,ka]
            amf = 0.5f0*(am[ia,ja,ka-1] + am[ia,ja,ka])
            amp = max(amf, 0.5f0*abs(wLz+wRz), 0.05f0)
            fr  += -0.25f0*(pR - pL)/amp
            fmz += -0.5f0*0.5f0*(rho[ia,ja,ka-1]+rho[ia,ja,ka])*amp*(wRz - wLz)
        end
        Fr[i,j,k]=fr; Fmx[i,j,k]=fmx; Fmy[i,j,k]=fmy; Fmz[i,j,k]=fmz; Fe[i,j,k]=fe
    end
end

# ★ ACOUSTIC FILTER kernels (3D port). Per k-plane ADI line-implicit low-pass of the density increment
# delta=rho-rk_r; off-diag beta = kfac*dtl^2*Kf*|S|^2/vol^2, Kf=(both cells railed) ? max(0,cs2*Tf-a2mf):0
# (a2m=am^2). Pure Neumann Helmholtz (zero row-sums => mass-conserving); railed-core gate zeros the
# off-diagonal across the vapour-liquid interface. Reconstruct rho=rk_r+delta_filt (floor 1e-4), hold
# momentum invariant (u,v,w *= rho_pre/rho_new). Metrics Six/Siy/Sjx/Sjy/vol are the shared 2D cross-section.
@kernel function _k_acou3d_i!(rho, ux, uy, uz, @Const(rk_r), @Const(am), @Const(T),
                              @Const(Six), @Const(Siy), @Const(vol), @Const(dtf),
                              delta, aE, bp, kfac::Float32, NI::Int32, NJ::Int32, NK::Int32, g::Int32)
    j, k = @index(Global, NTuple)
    @inbounds if j <= NJ && k <= NK
        ja = j + g; ka = k + g
        for i in 1:NI
            ia = i + g
            delta[i,j,k] = rho[ia,ja,ka] - rk_r[ia,ja,ka]
            if i < NI
                gapP = CS2*T[ia,ja,ka]   - am[ia,ja,ka]*am[ia,ja,ka]
                gapN = CS2*T[ia+1,ja,ka] - am[ia+1,ja,ka]*am[ia+1,ja,ka]
                if gapP > 0.0f0 && gapN > 0.0f0
                    Tf   = 0.5f0*(T[ia,ja,ka] + T[ia+1,ja,ka])
                    a2mf = 0.5f0*(am[ia,ja,ka]*am[ia,ja,ka] + am[ia+1,ja,ka]*am[ia+1,ja,ka])
                    Kf   = CS2*Tf - a2mf; Kf = Kf > 0.0f0 ? Kf : 0.0f0
                    Sf2  = Six[i+1,j]^2 + Siy[i+1,j]^2
                    dtl  = 0.5f0*(dtf[i,j,k] + dtf[i+1,j,k])
                    aE[i,j,k] = kfac*dtl*dtl*Kf*Sf2/(vol[i,j]*vol[i,j])
                else
                    aE[i,j,k] = 0.0f0
                end
            else
                aE[i,j,k] = 0.0f0
            end
        end
        bp[1,j,k] = 1.0f0 + aE[1,j,k]
        for i in 2:NI
            aW = aE[i-1,j,k]; w = (-aW)/bp[i-1,j,k]
            bp[i,j,k] = (1.0f0 + aW + aE[i,j,k]) - w*(-aW)
            delta[i,j,k] = delta[i,j,k] - w*delta[i-1,j,k]
        end
        delta[NI,j,k] = delta[NI,j,k]/bp[NI,j,k]
        for i in NI-1:-1:1
            delta[i,j,k] = (delta[i,j,k] + aE[i,j,k]*delta[i+1,j,k])/bp[i,j,k]
        end
        for i in 1:NI
            ia = i + g
            r_pre = rho[ia,ja,ka]
            r_raw = rk_r[ia,ja,ka] + delta[i,j,k]
            r_new = r_raw < 1.0f-4 ? 1.0f-4 : r_raw
            sc = r_raw < 1.0f-4 ? 1.0f0 : r_pre/r_new
            rho[ia,ja,ka] = r_new; ux[ia,ja,ka] *= sc; uy[ia,ja,ka] *= sc; uz[ia,ja,ka] *= sc
        end
    end
end

@kernel function _k_acou3d_j!(rho, ux, uy, uz, @Const(rk_r), @Const(am), @Const(T),
                              @Const(Sjx), @Const(Sjy), @Const(vol), @Const(dtf),
                              delta, aN, bp, kfac::Float32, NI::Int32, NJ::Int32, NK::Int32, g::Int32)
    i, k = @index(Global, NTuple)
    @inbounds if i <= NI && k <= NK
        ia = i + g; ka = k + g
        for j in 1:NJ
            ja = j + g
            delta[i,j,k] = rho[ia,ja,ka] - rk_r[ia,ja,ka]
            if j < NJ
                gapP = CS2*T[ia,ja,ka]   - am[ia,ja,ka]*am[ia,ja,ka]
                gapN = CS2*T[ia,ja+1,ka] - am[ia,ja+1,ka]*am[ia,ja+1,ka]
                if gapP > 0.0f0 && gapN > 0.0f0
                    Tf   = 0.5f0*(T[ia,ja,ka] + T[ia,ja+1,ka])
                    a2mf = 0.5f0*(am[ia,ja,ka]*am[ia,ja,ka] + am[ia,ja+1,ka]*am[ia,ja+1,ka])
                    Kf   = CS2*Tf - a2mf; Kf = Kf > 0.0f0 ? Kf : 0.0f0
                    Sf2  = Sjx[i,j+1]^2 + Sjy[i,j+1]^2
                    dtl  = 0.5f0*(dtf[i,j,k] + dtf[i,j+1,k])
                    aN[i,j,k] = kfac*dtl*dtl*Kf*Sf2/(vol[i,j]*vol[i,j])
                else
                    aN[i,j,k] = 0.0f0
                end
            else
                aN[i,j,k] = 0.0f0
            end
        end
        bp[i,1,k] = 1.0f0 + aN[i,1,k]
        for j in 2:NJ
            aS = aN[i,j-1,k]; w = (-aS)/bp[i,j-1,k]
            bp[i,j,k] = (1.0f0 + aS + aN[i,j,k]) - w*(-aS)
            delta[i,j,k] = delta[i,j,k] - w*delta[i,j-1,k]
        end
        delta[i,NJ,k] = delta[i,NJ,k]/bp[i,NJ,k]
        for j in NJ-1:-1:1
            delta[i,j,k] = (delta[i,j,k] + aN[i,j,k]*delta[i,j+1,k])/bp[i,j,k]
        end
        for j in 1:NJ
            ja = j + g
            r_pre = rho[ia,ja,ka]
            r_raw = rk_r[ia,ja,ka] + delta[i,j,k]
            r_new = r_raw < 1.0f-4 ? 1.0f-4 : r_raw
            sc = r_raw < 1.0f-4 ? 1.0f0 : r_pre/r_new
            rho[ia,ja,ka] = r_new; ux[ia,ja,ka] *= sc; uy[ia,ja,ka] *= sc; uz[ia,ja,ka] *= sc
        end
    end
end

# Driver: called BETWEEN step_3d! and tem3d_step!. acou==0 => bitwise legacy.
function acoustic_filter_3d!(s::Span3D)
    s.acou == Int32(0) && return nothing
    b = s.backend; NI, NJ, NK, g = Int32(s.NI), Int32(s.NJ), Int32(s.NK), Int32(s.g)
    _k_acou3d_i!(b, (64, 1))(s.rho, s.ux, s.uy, s.uz, s.rk_r, s.am, s.T, s.Six, s.Siy, s.vol, s.dtf,
                         s.acou_d, s.acou_an, s.acou_bp, s.acou_kfac, NI, NJ, NK, g; ndrange=(s.NJ, s.NK))
    _k_acou3d_j!(b, (64, 1))(s.rho, s.ux, s.uy, s.uz, s.rk_r, s.am, s.T, s.Sjx, s.Sjy, s.vol, s.dtf,
                         s.acou_d, s.acou_an, s.acou_bp, s.acou_kfac, NI, NJ, NK, g; ndrange=(s.NI, s.NK))
    nothing
end

@kernel function _k_snapshot_3d!(rk_r, rk_mx, rk_my, rk_mz, rk_E, @Const(rho), @Const(ux), @Const(uy), @Const(uz), @Const(T),
                                 eint::Float32, NI::Int32, NJ::Int32, NK::Int32, g::Int32)
    i, j, k = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ && k <= NK
        ia=i+g; ja=j+g; ka=k+g; r=rho[ia,ja,ka]; u=ux[ia,ja,ka]; v=uy[ia,ja,ka]; wv=uz[ia,ja,ka]
        rk_r[ia,ja,ka]=r; rk_mx[ia,ja,ka]=r*u; rk_my[ia,ja,ka]=r*v; rk_mz[ia,ja,ka]=r*wv
        rk_E[ia,ja,ka]=r*(eint*T[ia,ja,ka]+0.5f0*(u*u+v*v+wv*wv))
    end
end

@kernel function _k_dtfield_3d!(dtf, @Const(ux), @Const(uy), @Const(uz), @Const(T), @Const(nut),
                                @Const(Six), @Const(Siy), @Const(Sjx), @Const(Sjy), @Const(vol),
                                dz::Float32, cfl::Float32, nu_lam::Float32, sst_on::Int32, zcfl::Int32, NI::Int32, NJ::Int32, NK::Int32, g::Int32)
    i, j, k = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ && k <= NK
        ia=i+g; ja=j+g; ka=k+g; u=ux[ia,ja,ka]; v=uy[ia,ja,ka]; wv=uz[ia,ja,ka]; c=3.0f0*sqrt(max(T[ia,ja,ka],1.0f-3))
        SiE=sqrt(Six[i+1,j]^2+Siy[i+1,j]^2); SiW=sqrt(Six[i,j]^2+Siy[i,j]^2)
        SjN=sqrt(Sjx[i,j+1]^2+Sjy[i,j+1]^2); SjS=sqrt(Sjx[i,j]^2+Sjy[i,j]^2)
        lam_i=(abs(u*Six[i,j]+v*Siy[i,j])+abs(u*Six[i+1,j]+v*Siy[i+1,j]))*0.5f0+c*0.5f0*(SiE+SiW)
        lam_j=(abs(u*Sjx[i,j]+v*Sjy[i,j])+abs(u*Sjx[i,j+1]+v*Sjy[i,j+1]))*0.5f0+c*0.5f0*(SjN+SjS)
        vol2=vol[i,j]
        # zcfl=0 (consistency gate): drop the spanwise CFL term ⇒ dtf ≡ the 2D dtf (legitimate: at w≡0 &
        # spanwise-uniform the z-flux is identically zero, so the z-wave constraint is vacuous → byte match)
        zt = zcfl == Int32(1) ? (abs(wv)+c)*vol2/dz : 0.0f0
        dt = cfl*vol2/max(lam_i+lam_j + zt, 1.0f-30)
        nueff = nu_lam + (sst_on == Int32(1) ? nut[ia,ja,ka] : 0.0f0)   # eddy visc MUST bound the viscous CFL
        if nueff > 0.0f0
            dn_ij = vol2/max(max(SiE,SiW), max(SjN,SjS))
            dn_min = zcfl == Int32(1) ? min(dn_ij, dz) : dn_ij
            dt = min(dt, 0.2f0*dn_min*dn_min/nueff)
        end
        dtf[i,j,k]=dt
    end
end

@kernel function _k_update_3d!(rho, ux, uy, uz, T, @Const(rk_r), @Const(rk_mx), @Const(rk_my), @Const(rk_mz), @Const(rk_E),
                               @Const(Fi_r), @Const(Fi_mx), @Const(Fi_my), @Const(Fi_mz), @Const(Fi_e),
                               @Const(Fj_r), @Const(Fj_mx), @Const(Fj_my), @Const(Fj_mz), @Const(Fj_e),
                               @Const(Fk_r), @Const(Fk_mx), @Const(Fk_my), @Const(Fk_mz), @Const(Fk_e),
                               @Const(un_r), @Const(un_mx), @Const(un_my), @Const(un_mz), @Const(un_E),
                               @Const(um_r), @Const(um_mx), @Const(um_my), @Const(um_mz), @Const(um_E),
                               @Const(dtf), @Const(vol), eint::Float32, dz::Float32, dtp::Float32, wst::Float32, bst::Float32,
                               NI::Int32, NJ::Int32, NK::Int32, g::Int32)
    i, j, k = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ && k <= NK
        ia=i+g; ja=j+g; ka=k+g
        r0=rho[ia,ja,ka]; u0=ux[ia,ja,ka]; v0=uy[ia,ja,ka]; w0=uz[ia,ja,ka]
        E0=r0*(eint*T[ia,ja,ka]+0.5f0*(u0*u0+v0*v0+w0*w0))
        a=dtf[i,j,k]/vol[i,j]; b=dtf[i,j,k]/dz
        dcr =Fi_r[i+1,j,k]-Fi_r[i,j,k]  + Fj_r[i,j+1,k]-Fj_r[i,j,k]
        dcmx=Fi_mx[i+1,j,k]-Fi_mx[i,j,k]+ Fj_mx[i,j+1,k]-Fj_mx[i,j,k]
        dcmy=Fi_my[i+1,j,k]-Fi_my[i,j,k]+ Fj_my[i,j+1,k]-Fj_my[i,j,k]
        dcmz=Fi_mz[i+1,j,k]-Fi_mz[i,j,k]+ Fj_mz[i,j+1,k]-Fj_mz[i,j,k]
        dce =Fi_e[i+1,j,k]-Fi_e[i,j,k]  + Fj_e[i,j+1,k]-Fj_e[i,j,k]
        r  = r0    - a*dcr  - b*(Fk_r[i,j,k+1]-Fk_r[i,j,k])
        mx = r0*u0 - a*dcmx - b*(Fk_mx[i,j,k+1]-Fk_mx[i,j,k])
        my = r0*v0 - a*dcmy - b*(Fk_my[i,j,k+1]-Fk_my[i,j,k])
        mz = r0*w0 - a*dcmz - b*(Fk_mz[i,j,k+1]-Fk_mz[i,j,k])
        e  = E0    - a*dce  - b*(Fk_e[i,j,k+1]-Fk_e[i,j,k])
        if dtp > 0.0f0
            # dual-time BDF2 point-implicit source (mirrors the 2D _k_update! line-for-line)
            dtl=dtf[i,j,k]; fac=1.0f0/(1.0f0 + 1.5f0*dtl/dtp); h=dtl/(2.0f0*dtp)
            r  = (r  + h*(4.0f0*un_r[i,j,k]  - um_r[i,j,k] ))*fac
            mx = (mx + h*(4.0f0*un_mx[i,j,k] - um_mx[i,j,k]))*fac
            my = (my + h*(4.0f0*un_my[i,j,k] - um_my[i,j,k]))*fac
            mz = (mz + h*(4.0f0*un_mz[i,j,k] - um_mz[i,j,k]))*fac
            e  = (e  + h*(4.0f0*un_E[i,j,k]  - um_E[i,j,k] ))*fac
        end
        r  = bst*rk_r[ia,ja,ka]  + wst*r
        mx = bst*rk_mx[ia,ja,ka] + wst*mx
        my = bst*rk_my[ia,ja,ka] + wst*my
        mz = bst*rk_mz[ia,ja,ka] + wst*mz
        e  = bst*rk_E[ia,ja,ka]  + wst*e
        # density floor 1e-3 -> 1e-5: 1e-3 is EXACTLY a 1000:1 mixture's vapor density and CLIPPED the
        # cavity core (see the 2D twin). NO-OP for non-cavitating runs => bitwise-legacy.
        r = r < 1.0f-4 ? 1.0f-4 : r
        u2=mx/r; v2=my/r; w2=mz/r; sp=sqrt(u2*u2+v2*v2+w2*w2)
        if sp>3.5f0; f=3.5f0/sp; u2*=f; v2*=f; w2*=f; end
        T2=(e/r-0.5f0*(u2*u2+v2*v2+w2*w2))/eint; T2 = T2<1.0f-3 ? 1.0f-3 : T2
        rho[ia,ja,ka]=r; ux[ia,ja,ka]=u2; uy[ia,ja,ka]=v2; uz[ia,ja,ka]=w2; T[ia,ja,ka]=T2
    end
end

# ---- driver -----------------------------------------------------------------------------------------
function step_3d!(s::Span3D, cfl::Float32; zcfl::Bool=true, dtp::Float32=0.0f0)
    b = s.backend; NI, NJ, NK, g = Int32(s.NI), Int32(s.NJ), Int32(s.NK), Int32(s.g)
    wf = s.wall_flux ? Int32(1) : Int32(0); sston = s.sst ? Int32(1) : Int32(0)
    _k_dtfield_3d!(b, WG3)(s.dtf, s.ux, s.uy, s.uz, s.T, s.nut, s.Six, s.Siy, s.Sjx, s.Sjy, s.vol,
                           s.dz, cfl, s.nu_lam, sston, zcfl ? Int32(1) : Int32(0), NI, NJ, NK, g; ndrange=(s.NI, s.NJ, s.NK))
    _k_snapshot_3d!(b, WG3)(s.rk_r, s.rk_mx, s.rk_my, s.rk_mz, s.rk_E, s.rho, s.ux, s.uy, s.uz, s.T, s.eint, NI, NJ, NK, g; ndrange=(s.NI, s.NJ, s.NK))
    for (wst, bst) in ((1.0f0, 0.0f0), (0.25f0, 0.75f0), (2.0f0/3.0f0, 1.0f0/3.0f0))
        applyBC_3d!(s)
        if s.nu_lam > 0.0f0 && s.fullvisc == Int32(1)   # F2: precompute Green-Gauss gradients (ghosts filled above)
            _k_gradients_3d!(b, WG3)(s.gux,s.guy,s.guz,s.gvx,s.gvy,s.gvz,s.gwx,s.gwy,s.gwz,s.gTx,s.gTy,s.gTz,
                                     s.ux,s.uy,s.uz,s.T, s.Six,s.Siy,s.Sjx,s.Sjy,s.vol, s.dz, NI, NJ, NK, g; ndrange=(s.NI, s.NJ, s.NK))
            _k_grad_kghost_3d!(b, WG3)(s.gux,s.guy,s.guz,s.gvx,s.gvy,s.gvz,s.gwx,s.gwy,s.gwz,s.gTx,s.gTy,s.gTz,
                                       Int32(s.NI+2s.g), Int32(s.NJ+2s.g), NK, g; ndrange=(s.NI+2s.g, s.NJ+2s.g, s.g))
        end
        _k_flux_i_3d!(b, WG3)(s.Fi_r, s.Fi_mx, s.Fi_my, s.Fi_mz, s.Fi_e, s.rho, s.ux, s.uy, s.uz, s.T,
                              s.nut, s.Six, s.Siy, s.xc, s.yc, s.cxa, s.cya, s.cza, s.wa,
                              s.gux,s.guy,s.guz,s.gvx,s.gvy,s.gvz,s.gwx,s.gwy,s.gwz,s.gTx,s.gTy,
                              s.drot, s.nu_lam, sston, s.fullvisc, s.recon, s.pexc, s.am, s.cav, NI, NJ, NK, g; ndrange=(s.NI+1, s.NJ, s.NK))
        _k_flux_j_3d!(b, WG3)(s.Fj_r, s.Fj_mx, s.Fj_my, s.Fj_mz, s.Fj_e, s.rho, s.ux, s.uy, s.uz, s.T,
                              s.nut, s.Sjx, s.Sjy, s.xc, s.yc, s.vol, s.cxa, s.cya, s.cza, s.wa,
                              s.gux,s.guy,s.guz,s.gvx,s.gvy,s.gvz,s.gwx,s.gwy,s.gwz,s.gTx,s.gTy,
                              s.drot, s.nu_lam, sston, wf, s.fullvisc, s.recon, s.pexc, s.am, s.cav, NI, NJ, NK, Int32(s.NIW), Int32(s.NIF), g; ndrange=(s.NI, s.NJ+1, s.NK))
        _k_flux_k_3d!(b, WG3)(s.Fk_r, s.Fk_mx, s.Fk_my, s.Fk_mz, s.Fk_e, s.rho, s.ux, s.uy, s.uz, s.T,
                              s.nut, s.cxa, s.cya, s.cza, s.wa,
                              s.gux,s.gvy,s.gwx,s.gwy,s.guz,s.gvz,s.gwz,s.gTz,
                              s.drot, s.nu_lam, sston, s.fullvisc, s.recon, s.dz, s.pexc, s.am, s.cav, NI, NJ, NK, g; ndrange=(s.NI, s.NJ, s.NK+1))
        _k_update_3d!(b, WG3)(s.rho, s.ux, s.uy, s.uz, s.T, s.rk_r, s.rk_mx, s.rk_my, s.rk_mz, s.rk_E,
                              s.Fi_r, s.Fi_mx, s.Fi_my, s.Fi_mz, s.Fi_e, s.Fj_r, s.Fj_mx, s.Fj_my, s.Fj_mz, s.Fj_e,
                              s.Fk_r, s.Fk_mx, s.Fk_my, s.Fk_mz, s.Fk_e,
                              s.un_r, s.un_mx, s.un_my, s.un_mz, s.un_E, s.um_r, s.um_mx, s.um_my, s.um_mz, s.um_E,
                              s.dtf, s.vol, s.eint, s.dz, dtp, wst, bst, NI, NJ, NK, g; ndrange=(s.NI, s.NJ, s.NK))
    end
    nothing
end

# dual-time BDF2 history rotation (um <- un, un <- current conserved). Call ONCE per physical step.
@kernel function _k_histcycle_3d!(un_r, un_mx, un_my, un_mz, un_E, um_r, um_mx, um_my, um_mz, um_E,
                                  @Const(rho), @Const(ux), @Const(uy), @Const(uz), @Const(T),
                                  eint::Float32, NI::Int32, NJ::Int32, NK::Int32, g::Int32)
    i, j, k = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ && k <= NK
        um_r[i,j,k]=un_r[i,j,k]; um_mx[i,j,k]=un_mx[i,j,k]; um_my[i,j,k]=un_my[i,j,k]; um_mz[i,j,k]=un_mz[i,j,k]; um_E[i,j,k]=un_E[i,j,k]
        ia=i+g; ja=j+g; ka=k+g; r=rho[ia,ja,ka]; u=ux[ia,ja,ka]; v=uy[ia,ja,ka]; wv=uz[ia,ja,ka]
        un_r[i,j,k]=r; un_mx[i,j,k]=r*u; un_my[i,j,k]=r*v; un_mz[i,j,k]=r*wv
        un_E[i,j,k]=r*(eint*T[ia,ja,ka]+0.5f0*(u*u+v*v+wv*wv))
    end
end

function hist_cycle_3d!(s::Span3D)
    b = s.backend; NI, NJ, NK, g = Int32(s.NI), Int32(s.NJ), Int32(s.NK), Int32(s.g)
    _k_histcycle_3d!(b, WG3)(s.un_r, s.un_mx, s.un_my, s.un_mz, s.un_E, s.um_r, s.um_mx, s.um_my, s.um_mz, s.um_E,
                             s.rho, s.ux, s.uy, s.uz, s.T, s.eint, NI, NJ, NK, g; ndrange=(s.NI, s.NJ, s.NK))
    nothing
end
"""Seed both BDF2 anchors from the current (developed) field before the dual-time phase begins."""
seed_hist_3d!(s::Span3D) = (hist_cycle_3d!(s); hist_cycle_3d!(s))

# spanwise-averaged loads (Cl, Cd) — wall-row pressure + no-slip shear, averaged over k
function loads_3d(s::Span3D)
    g = s.g; KA.synchronize(s.backend)
    r3 = Array(s.rho); T3 = Array(s.T); u3 = Array(s.ux); v3 = Array(s.uy)
    px3 = s.cav >= Int32(1) ? Array(s.pexc) : nothing        # PDP: wall carries p_mix = rho*cs2*T + pexc
    Fx=0.0; Fy=0.0
    for k in 1:s.NK
        ka=k+g
        for (n,i) in enumerate(s.NIW+1:s.NIW+s.NIF)
            ia=i+g; pw=Float64(r3[ia,g+1,ka])*Float64(CS2)*Float64(T3[ia,g+1,ka]) +
                       (px3 === nothing ? 0.0 : Float64(px3[ia,g+1,ka]))
            Fx -= pw*Float64(s.Sjx1_h[i]); Fy -= pw*Float64(s.Sjy1_h[i])
            if s.nu_lam > 0.0f0
                Sf=sqrt(Float64(s.Sjx1_h[i])^2+Float64(s.Sjy1_h[i])^2); d1=0.5*Float64(s.vol1_h[i])/Sf+1e-30
                muw=Float64(r3[ia,g+1,ka])*Float64(s.nu_lam)*Sf/d1
                Fx += muw*Float64(u3[ia,g+1,ka]); Fy += muw*Float64(v3[ia,g+1,ka])
            end
        end
    end
    Fx/=s.NK; Fy/=s.NK
    U2=Float64(s.uxi)^2+Float64(s.uyi)^2; q=0.5*Float64(s.rho_inf)*U2
    Cl=(Fy*Float64(s.ca)-Fx*Float64(s.sa))/q; Cd=(Fx*Float64(s.ca)+Fy*Float64(s.sa))/q
    (Cl, Cd)
end

# max spanwise (k) variation of a field — the G1 consistency metric (should stay ~machine noise)
function span_uniformity(s::Span3D)
    g=s.g; KA.synchronize(s.backend); r=Array(s.rho); w=Array(s.uz)
    dmax=0.0; wmax=0.0
    for i in g+1:g+s.NI, j in g+1:g+s.NJ
        rmn=Inf; rmx=-Inf
        for k in g+1:g+s.NK
            v=Float64(r[i,j,k]); rmn=min(rmn,v); rmx=max(rmx,v); wmax=max(wmax, abs(Float64(w[i,j,k])))
        end
        dmax=max(dmax, rmx-rmn)
    end
    (dmax, wmax)   # (max spanwise rho spread, max |w|)
end

# ---- dual-time pseudo-residual (rstar) — 3D port of _k_rstar!/rstar_report_ka (convergence gate) -----
# The BDF2-in-physical-time residual of CONTINUITY, scaled by the local pseudo-step dtf so it is the
# dimensionless pseudo-time increment (like the 2D twin). At pseudo-convergence rstar -> 0: the inner
# loop has satisfied the BDF2 physical equation. This is the MANDATORY gate the 3D driver lacked -- a
# span-uniform steady state can read "no shedding" simply because the inner loop never converged. The
# spatial divergence adds the k-face term (Fk_r) the 2D lacks; at span-uniform w=0 it telescopes to zero
# and this reduces EXACTLY to the 2D residual. Fluxes/rho are read as the last step_3d! substage left
# them (the same mild post-substage inconsistency the 2D reporter carries).
@kernel function _k_rstar_3d!(out, @Const(rho), @Const(Fi_r), @Const(Fj_r), @Const(Fk_r),
                              @Const(un_r), @Const(um_r), @Const(dtf), @Const(vol),
                              dz::Float32, dtp::Float32, NI::Int32, NJ::Int32, NK::Int32, g::Int32)
    i, j, k = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ && k <= NK
        ia=i+g; ja=j+g; ka=k+g
        dcr = (Fi_r[i+1,j,k]-Fi_r[i,j,k] + Fj_r[i,j+1,k]-Fj_r[i,j,k])/vol[i,j] +
              (Fk_r[i,j,k+1]-Fk_r[i,j,k])/dz
        out[i,j,k] = dtf[i,j,k]*(dcr + (3.0f0*rho[ia,ja,ka] - 4.0f0*un_r[i,j,k] + um_r[i,j,k])/(2.0f0*dtp))
    end
end

# scratch is an interior-sized A3 the caller owns (e.g. similar(s.un_r)). Returns (Linf, L2, frac_at_max).
# L2 is the honest field measure; L-inf in a cavity is owned by one near-wall cell (2D lesson).
function rstar_report_3d(s::Span3D, dtp::Float32, scratch)
    b=s.backend; NI,NJ,NK,g = Int32(s.NI),Int32(s.NJ),Int32(s.NK),Int32(s.g)
    _k_rstar_3d!(b, WG3)(scratch, s.rho, s.Fi_r, s.Fj_r, s.Fk_r, s.un_r, s.um_r, s.dtf, s.vol,
                         s.dz, dtp, NI, NJ, NK, g; ndrange=(s.NI, s.NJ, s.NK))
    KA.synchronize(b)
    r = abs.(Array(scratch))
    linf=maximum(r); l2=sqrt(sum(abs2, r)/length(r)); frac=count(>=(0.99f0*linf), r)/length(r)
    (linf, l2, frac)
end

# ---- k-omega SST (3D, plain: no lag/gammat/EARSM) — ports cmesh_solver_ka _k_sst_sweep!/_commit! -----
# Curvilinear (i,j) gradients via the xi/eta Jacobian (per k-plane, reused metrics) + Cartesian (z)
# central differences. Full 3D strain S^2 = 2(uxx^2+vyy^2+wzz^2) + (uy+vx)^2 + (uz+wx)^2 + (vz+wy)^2 and
# 3D vorticity magnitude. At w=0 & spanwise-uniform the z-terms vanish => reduces to the 2D SST (G1-SST).
@kernel function _k_sst_bc_3d!(tk, tom, nut, k_inf::Float32, w_inf::Float32,
                               NI::Int32, NIW::Int32, NIF::Int32, NJ::Int32, g::Int32)
    i, kk, k = @index(Global, NTuple)
    @inbounds if i <= NI && kk <= g
        ka = k+g
        if i <= NIW || i > NIW+NIF                      # wake cut: partner exchange
            ip = NI+1-i
            tk[i+g,g+1-kk,ka]=tk[ip+g,kk+g,ka]; tom[i+g,g+1-kk,ka]=tom[ip+g,kk+g,ka]; nut[i+g,g+1-kk,ka]=nut[ip+g,kk+g,ka]
        else                                            # wall: k antisym (->0), omega mirror, nut->0
            tk[i+g,g+1-kk,ka]=-tk[i+g,g+kk,ka]; tom[i+g,g+1-kk,ka]=tom[i+g,g+kk,ka]; nut[i+g,g+1-kk,ka]=0.0f0
        end
        tk[i+g,NJ+g+kk,ka]=k_inf; tom[i+g,NJ+g+kk,ka]=w_inf; nut[i+g,NJ+g+kk,ka]=k_inf/w_inf   # far-field
    end
end

@kernel function _k_sst_bc_io_3d!(tk, tom, nut, NI::Int32, NJ::Int32, g::Int32)
    jj, kk, k = @index(Global, NTuple)
    @inbounds if jj <= NJ+2g && kk <= g
        ka = k+g
        tk[g+1-kk,jj,ka]=tk[g+1,jj,ka]; tom[g+1-kk,jj,ka]=tom[g+1,jj,ka]; nut[g+1-kk,jj,ka]=nut[g+1,jj,ka]
        tk[NI+g+kk,jj,ka]=tk[NI+g,jj,ka]; tom[NI+g+kk,jj,ka]=tom[NI+g,jj,ka]; nut[NI+g+kk,jj,ka]=nut[NI+g,jj,ka]
    end
end

@kernel function _k_sst_span_3d!(tk, tom, nut, NX::Int32, NY::Int32, NK::Int32, g::Int32)
    i, j, kk = @index(Global, NTuple)
    @inbounds if i <= NX && j <= NY && kk <= g
        tk[i,j,g+1-kk]=tk[i,j,g+NK+1-kk]; tom[i,j,g+1-kk]=tom[i,j,g+NK+1-kk]; nut[i,j,g+1-kk]=nut[i,j,g+NK+1-kk]
        tk[i,j,g+NK+kk]=tk[i,j,g+kk]; tom[i,j,g+NK+kk]=tom[i,j,g+kk]; nut[i,j,g+NK+kk]=nut[i,j,g+kk]
    end
end

@kernel function _k_sst_sweep_3d!(knew, wnew, nutl, @Const(tk), @Const(tom), @Const(nut),
                                  @Const(ux), @Const(uy), @Const(uz), @Const(xc), @Const(yc),
                                  @Const(Six), @Const(Siy), @Const(Sjx), @Const(Sjy), @Const(vol), @Const(wdist), @Const(gammat), @Const(dtf),
                                  @Const(amarr),
                                  nu::Float32, k_inf::Float32, w_inf::Float32, dz::Float32,
                                  wilcox_xi::Float32, wilcox_mt0::Float32,
                                  NI::Int32, NJ::Int32, NK::Int32, g::Int32)
    i, j, k = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ && k <= NK
        βs=0.09f0; a1=0.31f0
        σk1=0.85f0;σw1=0.5f0;β1=0.075f0;γ1=0.5532f0; σk2=1.0f0;σw2=0.856f0;β2=0.0828f0;γ2=0.4403f0
        ia=i+g; ja=j+g; ka=k+g
        k0=tk[ia,ja,ka]; w0=max(tom[ia,ja,ka],1.0f-6); d=max(wdist[i,j],1.0f-8)
        im=i>1 ? i-1 : 1; ip2=i<NI ? i+1 : NI; jm=j>1 ? j-1 : 1; jp=j<NJ ? j+1 : NJ
        xxi=0.5f0*(xc[ip2,j]-xc[im,j]); yxi=0.5f0*(yc[ip2,j]-yc[im,j])
        xet=0.5f0*(xc[i,jp]-xc[i,jm]); yet=0.5f0*(yc[i,jp]-yc[i,jm])
        J=xxi*yet-xet*yxi; J = abs(J)<1.0f-30 ? 1.0f-30 : J
        duxi=0.5f0*(ux[ia+1,ja,ka]-ux[ia-1,ja,ka]); duet=0.5f0*(ux[ia,ja+1,ka]-ux[ia,ja-1,ka])
        dvxi=0.5f0*(uy[ia+1,ja,ka]-uy[ia-1,ja,ka]); dvet=0.5f0*(uy[ia,ja+1,ka]-uy[ia,ja-1,ka])
        dsxi=0.5f0*(uz[ia+1,ja,ka]-uz[ia-1,ja,ka]); dset=0.5f0*(uz[ia,ja+1,ka]-uz[ia,ja-1,ka])
        dudx=(duxi*yet-duet*yxi)/J; dudy=(-duxi*xet+duet*xxi)/J
        dvdx=(dvxi*yet-dvet*yxi)/J; dvdy=(-dvxi*xet+dvet*xxi)/J
        dwdx=(dsxi*yet-dset*yxi)/J; dwdy=(-dsxi*xet+dset*xxi)/J
        dudz=0.5f0*(ux[ia,ja,ka+1]-ux[ia,ja,ka-1])/dz
        dvdz=0.5f0*(uy[ia,ja,ka+1]-uy[ia,ja,ka-1])/dz
        dwdz=0.5f0*(uz[ia,ja,ka+1]-uz[ia,ja,ka-1])/dz
        Sxy=0.5f0*(dudy+dvdx); Sxz=0.5f0*(dudz+dwdx); Syz=0.5f0*(dvdz+dwdy)
        S2=2.0f0*(dudx*dudx+dvdy*dvdy+dwdz*dwdz)+4.0f0*(Sxy*Sxy+Sxz*Sxz+Syz*Syz); Ssr=sqrt(S2)
        Ox=dwdy-dvdz; Oy=dudz-dwdx; Oz=dvdx-dudy; Ω=sqrt(Ox*Ox+Oy*Oy+Oz*Oz)
        sqk=sqrt(max(k0,0.0f0))
        dkxi=0.5f0*(tk[ia+1,ja,ka]-tk[ia-1,ja,ka]); dket=0.5f0*(tk[ia,ja+1,ka]-tk[ia,ja-1,ka])
        dqxi=0.5f0*(tom[ia+1,ja,ka]-tom[ia-1,ja,ka]); dqet=0.5f0*(tom[ia,ja+1,ka]-tom[ia,ja-1,ka])
        dkdx=(dkxi*yet-dket*yxi)/J; dkdy=(-dkxi*xet+dket*xxi)/J
        dqdx=(dqxi*yet-dqet*yxi)/J; dqdy=(-dqxi*xet+dqet*xxi)/J
        dkdz=0.5f0*(tk[ia,ja,ka+1]-tk[ia,ja,ka-1])/dz; dqdz=0.5f0*(tom[ia,ja,ka+1]-tom[ia,ja,ka-1])/dz
        gkw=dkdx*dqdx+dkdy*dqdy+dkdz*dqdz
        CDkw=max(2.0f0*σw2/w0*gkw, 1.0f-10)
        arg1=min(max(sqk/(βs*w0*d), 500.0f0*nu/(d*d*w0)), 4.0f0*σw2*k0/(CDkw*d*d)); arg1=min(arg1,4.0f0); F1=tanh(arg1^4)
        arg2=min(max(2.0f0*sqk/(βs*w0*d), 500.0f0*nu/(d*d*w0)),4.0f0); F2=tanh(arg2^2)
        nut_t=a1*k0/max(a1*w0, Ssr*F2)
        Pk=min(nut_t*Ssr*Ω, 10.0f0*βs*k0*w0)*gammat[i,j]   # F1: trip intermittency (laminar LE), parity with 2D solver_ka.jl:1149
        σk=F1*σk1+(1.0f0-F1)*σk2; σw=F1*σw1+(1.0f0-F1)*σw2; βc=F1*β1+(1.0f0-F1)*β2; γc=F1*γ1+(1.0f0-F1)*γ2
        Dk=0.0f0; Dw=0.0f0
        SfE=sqrt(Six[i+1,j]^2+Siy[i+1,j]^2); dnE=i+1<=NI ? sqrt((xc[i+1,j]-xc[i,j])^2+(yc[i+1,j]-yc[i,j])^2)+1.0f-30 : vol[i,j]/SfE+1.0f-30
        nfE=0.5f0*(nut[ia,ja,ka]+nut[ia+1,ja,ka]); Dk+=(nu+σk*nfE)*SfE/dnE*(tk[ia+1,ja,ka]-k0); Dw+=(nu+σw*nfE)*SfE/dnE*(tom[ia+1,ja,ka]-w0)
        SfW=sqrt(Six[i,j]^2+Siy[i,j]^2); dnW=i-1>=1 ? sqrt((xc[i-1,j]-xc[i,j])^2+(yc[i-1,j]-yc[i,j])^2)+1.0f-30 : vol[i,j]/SfW+1.0f-30
        nfW=0.5f0*(nut[ia,ja,ka]+nut[ia-1,ja,ka]); Dk+=(nu+σk*nfW)*SfW/dnW*(tk[ia-1,ja,ka]-k0); Dw+=(nu+σw*nfW)*SfW/dnW*(tom[ia-1,ja,ka]-w0)
        SfN=sqrt(Sjx[i,j+1]^2+Sjy[i,j+1]^2); dnN=j+1<=NJ ? sqrt((xc[i,j+1]-xc[i,j])^2+(yc[i,j+1]-yc[i,j])^2)+1.0f-30 : vol[i,j]/SfN+1.0f-30
        nfN=0.5f0*(nut[ia,ja,ka]+nut[ia,ja+1,ka]); Dk+=(nu+σk*nfN)*SfN/dnN*(tk[ia,ja+1,ka]-k0); Dw+=(nu+σw*nfN)*SfN/dnN*(tom[ia,ja+1,ka]-w0)
        SfS=sqrt(Sjx[i,j]^2+Sjy[i,j]^2); dnS=j-1>=1 ? sqrt((xc[i,j-1]-xc[i,j])^2+(yc[i,j-1]-yc[i,j])^2)+1.0f-30 : vol[i,j]/SfS+1.0f-30
        nfS=0.5f0*(nut[ia,ja,ka]+nut[ia,ja-1,ka]); Dk+=(nu+σk*nfS)*SfS/dnS*(tk[ia,ja-1,ka]-k0); Dw+=(nu+σw*nfS)*SfS/dnS*(tom[ia,ja-1,ka]-w0)
        Dk/=vol[i,j]; Dw/=vol[i,j]
        nfP=0.5f0*(nut[ia,ja,ka]+nut[ia,ja,ka+1]); nfM=0.5f0*(nut[ia,ja,ka]+nut[ia,ja,ka-1]); iz2=1.0f0/(dz*dz)
        Dk += (nu+σk*nfP)*iz2*(tk[ia,ja,ka+1]-k0) + (nu+σk*nfM)*iz2*(tk[ia,ja,ka-1]-k0)
        Dw += (nu+σw*nfP)*iz2*(tom[ia,ja,ka+1]-w0) + (nu+σw*nfM)*iz2*(tom[ia,ja,ka-1]-w0)
        u0=ux[ia,ja,ka]; v0=uy[ia,ja,ka]; wv=uz[ia,ja,ka]
        Uci=(u0*0.5f0*(Six[i,j]+Six[i+1,j])+v0*0.5f0*(Siy[i,j]+Siy[i+1,j]))/vol[i,j]
        Ucj=(u0*0.5f0*(Sjx[i,j]+Sjx[i,j+1])+v0*0.5f0*(Sjy[i,j]+Sjy[i,j+1]))/vol[i,j]
        advk=(Uci>=0.0f0 ? Uci*(k0-tk[ia-1,ja,ka]) : Uci*(tk[ia+1,ja,ka]-k0)) +
             (Ucj>=0.0f0 ? Ucj*(k0-tk[ia,ja-1,ka]) : Ucj*(tk[ia,ja+1,ka]-k0)) +
             (wv>=0.0f0 ? wv*(k0-tk[ia,ja,ka-1])/dz : wv*(tk[ia,ja,ka+1]-k0)/dz)
        advw=(Uci>=0.0f0 ? Uci*(w0-tom[ia-1,ja,ka]) : Uci*(tom[ia+1,ja,ka]-w0)) +
             (Ucj>=0.0f0 ? Ucj*(w0-tom[ia,ja-1,ka]) : Ucj*(tom[ia,ja+1,ka]-w0)) +
             (wv>=0.0f0 ? wv*(w0-tom[ia,ja,ka-1])/dz : wv*(tom[ia,ja,ka+1]-w0)/dz)
        dtl=dtf[i,j,k]
        # ★ Wilcox dilatation dissipation (3D twin of the 2D _k_sst_sweep! block, 2026-07-23):
        # beta* -> beta*(1 + xi*max(Mt^2 - Mt0^2, 0)), Mt^2 = 2k/a_m^2 on the MIXTURE sound speed —
        # self-localizes to the cavity/closure; inert in liquid; bitwise when wilcox_xi = 0.
        βs_eff = βs
        if wilcox_xi > 0.0f0
            amloc = amarr[ia,ja,ka]
            mt2 = 2.0f0*max(k0, 0.0f0)/max(amloc*amloc, 1.0f-8)
            fmt = mt2 - wilcox_mt0*wilcox_mt0
            if fmt > 0.0f0
                βs_eff = βs*(1.0f0 + wilcox_xi*fmt)
            end
        end
        kn=(k0+dtl*(Pk-advk+Dk))/(1.0f0+dtl*βs_eff*w0)
        wn=(w0+dtl*(γc/max(nut_t,1.0f-10)*Pk-advw+Dw+2.0f0*(1.0f0-F1)*σw2/w0*gkw))/(1.0f0+dtl*βc*w0)
        kn=clamp(kn,1.0f-12,1.0f0); wn=max(wn,1.0f-6)
        if !(isfinite(kn)&&isfinite(wn)); kn=k_inf; wn=w_inf; end
        knew[ia,ja,ka]=kn; wnew[ia,ja,ka]=wn
        nutl[ia,ja,ka]=a1*kn/max(a1*wn, Ssr*F2)
    end
end

@kernel function _k_sst_commit_3d!(tk, tom, nut, @Const(knew), @Const(wnew), @Const(nutl), @Const(wdist), @Const(rho),
                                   nu::Float32, reboud_n::Float32, reboud_rhov::Float32,
                                   NI::Int32, NJ::Int32, NK::Int32, NIW::Int32, NIF::Int32, g::Int32)
    i, j, k = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ && k <= NK
        β1=0.075f0; ia=i+g; ja=j+g; ka=k+g
        kn=knew[ia,ja,ka]; wn=wnew[ia,ja,ka]; wallrow = j <= 2 && NIW < i <= NIW+NIF
        local nut_t::Float32
        if wallrow
            d=max(wdist[i,j],1.0f-8); wn=6.0f0*nu/(β1*d*d); nut_t=kn/max(wn,1.0f-6)
        else
            nut_t=nutl[ia,ja,ka]
        end
        if reboud_n > 0.0f0                          # Reboud 1998 cavitation eddy-viscosity reduction (see 2D)
            r = rho[ia,ja,ka]
            av = (1.0f0 - r)/(1.0f0 - reboud_rhov); av = av < 0.0f0 ? 0.0f0 : (av > 1.0f0 ? 1.0f0 : av)
            nut_t *= (1.0f0-av)^(reboud_n-1.0f0)     # large-density-ratio limit of f(rho)/rho
        end
        tk[ia,ja,ka]=kn; tom[ia,ja,ka]=wn
        nut[ia,ja,ka] += 0.2f0*(min(nut_t,0.15f0) - nut[ia,ja,ka])
    end
end

"""One k-omega SST sub-iteration (BC -> sweep -> commit). Call once per solver step, AFTER step_3d!."""
function sst_step_3d!(s::Span3D; reboud_n::Float32=0.0f0, reboud_rhov::Float32=0.33f0,
                      wilcox_xi::Float32=0.0f0, wilcox_mt0::Float32=0.25f0)
    s.sst || return
    b = s.backend; NI, NJ, NK, g = Int32(s.NI), Int32(s.NJ), Int32(s.NK), Int32(s.g)
    NX, NY = Int32(s.NI+2s.g), Int32(s.NJ+2s.g)
    _k_sst_bc_3d!(b, WG3)(s.tk, s.tom, s.nut, s.k_inf, s.w_inf, NI, Int32(s.NIW), Int32(s.NIF), NJ, g; ndrange=(s.NI, s.g, s.NK))
    _k_sst_bc_io_3d!(b, WG3)(s.tk, s.tom, s.nut, NI, NJ, g; ndrange=(s.NJ+2s.g, s.g, s.NK))
    _k_sst_span_3d!(b, WG3)(s.tk, s.tom, s.nut, NX, NY, NK, g; ndrange=(s.NI+2s.g, s.NJ+2s.g, s.g))
    _k_sst_sweep_3d!(b, WG3)(s.knew, s.wnew, s.nutl, s.tk, s.tom, s.nut, s.ux, s.uy, s.uz, s.xc, s.yc,
                             s.Six, s.Siy, s.Sjx, s.Sjy, s.vol, s.wdist, s.gammat, s.dtf, s.am,
                             s.nu_lam, s.k_inf, s.w_inf, s.dz, wilcox_xi, wilcox_mt0,
                             NI, NJ, NK, g; ndrange=(s.NI, s.NJ, s.NK))
    _k_sst_commit_3d!(b, WG3)(s.tk, s.tom, s.nut, s.knew, s.wnew, s.nutl, s.wdist, s.rho,
                              s.nu_lam, reboud_n, reboud_rhov, NI, NJ, NK, Int32(s.NIW), Int32(s.NIF), g; ndrange=(s.NI, s.NJ, s.NK))
    nothing
end

# ---- HEM barotropic cavitation (per-cell T-carrier, 3D loop) — mirrors CMeshCav._k_hem_eos! ---------
@kernel function _k_hem_3d!(T, pexc, am, @Const(rho), rho_inf::Float32, p_inf::Float32, p_v::Float32,
                            a2l::Float32, cmin2::Float32, psm::Float32, Tmin::Float32, Tmax::Float32,
                            NI::Int32, NJ::Int32, NK::Int32, g::Int32)
    i, j, k = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ && k <= NK
        ia=i+g; ja=j+g; ka=k+g; r=rho[ia,ja,ka]
        pl = p_inf + a2l*(r - rho_inf); pvp = p_v + cmin2*r; d = pl - pvp
        sq = sqrt(d*d + psm*psm)
        p = 0.5f0*(pl + pvp + sq)
        Tn = p/(r*CS2); Tn = Tn < Tmin ? Tmin : (Tn > Tmax ? Tmax : Tn)   # in-band pedestal (byte-identical)
        T[ia,ja,ka] = Tn
        pexc[ia,ja,ka] = p - r*CS2*Tn                     # PDP excess (breaks the 3:1 cap when cav>=1)
        dpdr = 0.5f0*(a2l + cmin2 + (a2l - cmin2)*d/sq)   # mixture sound speed for Layer C (cav>=2)
        am[ia,ja,ka] = sqrt(dpdr < cmin2 ? cmin2 : dpdr)
    end
end

function hem_apply_3d!(s::Span3D; p_v::Float32, a2l::Float32, cmin2::Float32, psm::Float32,
                       Tmin::Float32=0.3f0, Tmax::Float32=3.0f0)
    b = s.backend; NI, NJ, NK, g = Int32(s.NI), Int32(s.NJ), Int32(s.NK), Int32(s.g)
    p_inf = s.rho_inf*CS2*s.T_inf
    _k_hem_3d!(b, WG3)(s.T, s.pexc, s.am, s.rho, s.rho_inf, p_inf, p_v, a2l, cmin2, psm, Tmin, Tmax, NI, NJ, NK, g; ndrange=(s.NI, s.NJ, s.NK))
    nothing
end

# seed a small spanwise w-perturbation w += amp*U*sin(2*pi*m*z/Lz), only where the cavity is (rho<rho_thr)
# — breaks the 2D symmetry so a spanwise instability (if any) can grow from a defined mode.
@kernel function _k_seed_perturb!(uz, @Const(rho), amp::Float32, mwave::Float32, dz::Float32, Lz::Float32,
                                  rho_thr::Float32, NI::Int32, NJ::Int32, NK::Int32, g::Int32)
    i, j, k = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ && k <= NK
        ia=i+g; ja=j+g; ka=k+g
        if rho[ia,ja,ka] < rho_thr
            z = (Float32(k)-0.5f0)*dz
            uz[ia,ja,ka] += amp*sin(6.2831853f0*mwave*z/Lz)
        end
    end
end

function seed_span_perturb!(s::Span3D; amp::Float32, mwave::Float32=2.0f0, rho_thr::Float32=0.9f0)
    b = s.backend; NI, NJ, NK, g = Int32(s.NI), Int32(s.NJ), Int32(s.NK), Int32(s.g)
    Lz = s.dz*Float32(s.NK)
    _k_seed_perturb!(b, WG3)(s.uz, s.rho, amp, mwave, s.dz, Lz, rho_thr, NI, NJ, NK, g; ndrange=(s.NI, s.NJ, s.NK))
    nothing
end

# spanwise-mean cavity length on the suction wall row (x/c where mean rho < rho_thr, i > LE)
function cavity_len_3d(s::Span3D, xc1_h::Vector{Float64}, i_le::Int, rho_thr::Float64)
    g=s.g; KA.synchronize(s.backend); r=Array(s.rho); lc=0.0
    for i in s.NIW+1:s.NIW+s.NIF
        i <= i_le && continue
        rm=0.0; for k in g+1:g+s.NK; rm += Float64(r[i+g, g+1, k]); end; rm/=s.NK
        rm < rho_thr && (lc = max(lc, xc1_h[i]))
    end
    lc
end

end # module
