# CMESH 3D cavitation Tier B — TRANSPORT-EQUATION MODEL (TEM), 3D twin of CMeshCavTEM.
#
# WHY THIS EXISTS (measured, 2026-07-18): with the PDP pressure-split removing the 3:1 density cap, the
# HEM BAROTROPE becomes BISTABLE in 3D — sigma=1.6 gives NO cavity, sigma<=1.25 gives SUPERCAVITATION
# (l/c=0.999, lift destroyed); there is no stable partial cavity in between. Root cause: the barotrope's
# vapor branch p ~ p_v + cmin2*rho is FLAT, so nothing bounds cavity density from below and the cavity
# evacuates to the numerical floor the instant inception occurs. A PARTIAL cavity is the prerequisite for
# re-entrant-jet cloud shedding, so shedding is impossible with the barotrope.
#
# The TEM fixes this BY CONSTRUCTION: rho_sat(alpha) = (1-a)*rho_l_sat + a*rho_v_sat bounds the mixture
# density at rho_v_sat independently of any numerical floor, so a partial cavity is a genuine equilibrium.
# Plus the finite production/destruction rate supplies the metastable LAG that drives shedding.
#
# Formulation + the eos=1 WALLIS harmonic stiffness fix are identical to the 2D module (see
# src/cmesh/cmesh_cav_tem.jl for the full derivation and the stiffness rationale). 3D differences ONLY:
#   - alpha/alpha_new are (NI+2g, NJ+2g, NK+2g); BDF2 history un/um are (NI, NJ, NK)
#   - the sweep gains a SPANWISE advection term (uniform dz => contravariant velocity is simply uz/dz)
#   - the BC gains SPANWISE-PERIODIC ghost planes (matching _k_bc_span_3d! in cmesh_span3d.jl)
# The commit exports pexc + am exactly as the 2D twin, so it plugs straight into the ported 3D flux
# (Span3D.cav >= 1 injects pexc; >= 2 adds Layer C).
module CMeshCavTEM3D

# ★★★ 2026-07-31 — THE DIVERGENT EOS COPY IS RETIRED. `_a2m_mix3d` below used to carry its own
# transliteration of the mixture stiffness, and the audit's conclusion after the SECOND EOS defect was
# that both came from exactly that: 2D and 3D each holding a copy, and the copies drifting. It now
# DELEGATES to the shared single source. `_pmix_vl`, the local-pressure vapour term, is implemented as
# the selectable `vaploc=1` branch without another copy; that branch is still unvalidated and
# unauthorized for 3D production. Delegation proven bitwise: with ncg=0 and dk_cmin2=0 `_a2m_mix`
# collapses operation-for-operation onto the old body, and tools/_attic/retired_verifiers/verify_eos_shared.jl checks the
# shared module against the 2D original over 49,600 states.
include(joinpath(@__DIR__, "cmesh_eos_mix.jl"))
using .CMeshEOSMix: _a2m_mix, _pmix_vb, _pmix_vl
include(joinpath(@__DIR__, "cmesh_cav_config_contract.jl"))
using .CMeshCavConfigContract: validate_vaploc

using KernelAbstractions
const KA = KernelAbstractions

export TEM3D, build_tem3d, tem_step3d!, tem_hist_cycle3d!, tem_seed3d!, tem_ratio3d, tem_timescales3d, tem_ambient3d

const TEM_CS2 = 0.6666667f0
const WG3 = (64, 4, 1)

mutable struct TEM3D{A3}
    alpha::A3; alpha_new::A3          # (NI+2g, NJ+2g, NK+2g)
    un::A3; um::A3                    # (NI, NJ, NK) dual-time BDF2 history
    p_v::Float32; a2l::Float32
    rho_l_sat::Float32; rho_v_sat::Float32
    kprod::Float32; kdest::Float32
    Tmin::Float32; Tmax::Float32
    a2v::Float32; psm::Float32
    eos::Int32
    wallis::Int32                     # ★ Tier-C port (2026-07-23): density-weighted Wallis a2m (2D Stage 1d twin)
    src_jac::Int32                    # ★ P1 port (2026-07-23): 0 legacy / 3 kink-local Jacobian floor (2D production)
    omega::Float32                    # ★ segregated-coupling under-relaxation (2D-proven load-bearing at 0.2)
    # ★★★ LOCAL-PRESSURE VAPOUR TERM, PORTED FROM 2D (2026-08-01). >0 replaces the frozen-p_v Wallis
    # vapour term a/p_v -- valid only AT saturation density -- with the correct isothermal identity
    # rho_v*a_v^2 = p at the LOCAL pressure, solved in closed form by the SHARED `_pmix_vl`.
    # 0 = bitwise legacy (the defective form this file has warned about since 2026-07-30).
    # ⚠ NOT a duplicate: this calls the same shared helper the 2D path calls. The audit rule was
    # "extract ONE shared helper instead of copying" -- that helper exists, and this uses it.
    # APPENDED AT THE END: the constructor is positional.
    vaploc::Float32
end

# Mixture stiffness a_m^2 — twin of the 2D _a2m_mix (cmesh_cav_tem.jl): wallis=0 legacy mass-weighted
# harmonic (bitwise with the old inline form); wallis=1 TRUE density-weighted Wallis, the Tier-C branch
# that collapses a_m at low void and puts the crossover knee in reach.
#
# ⚠⚠⚠ THE wallis>=1, vaploc=0 LEGACY BRANCH BELOW IS THE DEFECT REPAIRED IN 2D ON 2026-07-30.
# Its vapour term is written `a/p_v`, i.e. the identity rho_v*a_v^2 = p_v, which holds ONLY at
# saturation density. For an isothermal vapour the correct identity is rho_v*a_v^2 = p at the LOCAL
# pressure; using the frozen p_v below rho_v_sat is wrong in the direction that UNBOUNDS the pressure.
# The shared local-pressure branch (`vaploc=1`, `_pmix_vl`) solves the resulting implicit EOS in closed
# form and the ideal-gas vapour limit falls out of the algebra. See the CLAUDE.md top banner and
# memory/project_docs/2026-07-29_AUDIT_ROUND2_VERDICT.md §5b-§5c.
#
# MEASURED, at the 3D Tier-C constants (eps_pv=0.1, wallis=1, ratio=100, a2l_frac=1.4, sigma=1),
# by transliterating THIS function and the correct root side by side in Float32 (321,201 states,
# alpha 0..1 x rho 1e-4..1 log-spaced):
#   * alpha=1 limit here is  p = 2*p_v - p_v*rho_v_sat/rho , which crosses ZERO at rho = rho_v_sat/2
#     = 5.0e-3 and reaches -98 p_v at rho = 1e-4. The correct root gives p = a2v*rho = 0.1867*rho,
#     positive for every rho > 0.
#   * p < 0 in 273,516 / 321,201 states = 85.2% of the map. Correct root: 0.0%. (The 2D figure for the
#     same census was 84% -> 0, so this is the same defect at the same strength.)
#   * ON the saturation line rho = rho_sat(alpha) the two agree to 8.0e-6 (Float32 round-off) for every
#     alpha — property P5. That is WHY this went unnoticed: it is invisible wherever the cavity sits at
#     saturation and only bites once a cell is evacuated below rho_v_sat/2.
# The 3D path is DORMANT (the 3D campaign is retracted, 2026-07-27). The `vaploc=1` implementation has
# not been GPU/physics validated and is not production authority. All known entry points independently
# quarantine before imports and pin historical `vaploc=0.0`; tools/test_tem3d_quarantine.jl checks that
# census. Constructor diagnostics remain defense in depth for a future classified caller or direct use.
#
# ★ RETIRED AS A COPY (2026-07-31). The body below used to be an independent transliteration; it now
# DELEGATES to the shared `_a2m_mix`, so 2D and 3D can no longer drift apart. The delegation is
# BITWISE: with `ncg = 0` and `dk_cmin2 = 0` the shared function takes its `wallis` branch and
# evaluates the identical two expressions in the identical association --
#     wallis>=1 :  S = (1-a)/(rho_l_sat*a2l) + a/p_v ;  1/(r*S)
#     wallis==0 :  1/((1-a)/a2l + a/a2v)
# -- and the `dk_cmin2 > 0` branch that precedes them is not taken. This preserves the historical
# `vaploc=0` arithmetic. `vaploc=1` routes through the shared `_pmix_vl`; implementation availability
# is not validation or authority, so the quarantined caller census must remain at explicit 0.0.
@inline _a2m_mix3d(a::Float32, r::Float32, a2l::Float32, a2v::Float32,
                   rho_l_sat::Float32, p_v::Float32, wallis::Int32) =
    _a2m_mix(a, r, a2l, a2v, rho_l_sat, p_v, wallis, 0.0f0, 0.0f0, 0.0f0)

"""Build a 3D TEM state for Span3D solver `s`. `vfloor=false` (PDP mode) lets rho_v_sat be the REAL vapor
density (the pexc split carries the excess pressure). `eos=1` (default) = Wallis harmonic mixture
stiffness + smooth-max vapor floor — REQUIRED (eos=0 is the stiff legacy form kept only for A/B).
`vaploc` is required and binary: 0 preserves the known-unsound legacy diagnostic; 1 selects an
implemented shared branch that is not yet validated or authorized for 3D production."""
function build_tem3d(s; sigma, Cprod::Float64=1.0, Cdest::Float64=0.05, ratio::Float64=1000.0,
                     vfloor::Bool=false, Tmin::Float32=0.3f0, Tmax::Float32=3.0f0,
                     eos::Int=1, cmin2_frac::Float64=1e-3, psm_frac::Float64=0.05,
                     a2l_frac::Float64=1.4,
                     eps_pv::Float64=-1.0, wallis::Int=0, rate_ref_pv::Bool=false,
                     src_jac::Int=0, omega::Float64=1.0, vaploc::Float64)
    vaploc = validate_vaploc(vaploc)
    U = sqrt(Float64(s.uxi)^2 + Float64(s.uyi)^2)
    q = 0.5*Float64(s.rho_inf)*U^2
    p_inf = Float64(s.rho_inf)*Float64(TEM_CS2)*Float64(s.T_inf)
    # ★ Tier-C port (2026-07-23, 2D twin semantics): eps_pv >= 0 => p_v = eps*sigma*q (swept WITH
    # sigma); eps_pv < 0 = legacy welded p_v (bitwise default).
    p_v = eps_pv >= 0.0 ? eps_pv*sigma*q : p_inf - sigma*q
    a2l = a2l_frac*Float64(TEM_CS2)   # softening a2l is the ONLY lever on void fraction (see 2D twin)
    # Reference the liquid saturation density to p_ref = p_v + sigma*q, NOT the lattice ambient p_inf
    # (see the 2D twin). Byte-identical here TODAY because this build still welds p_v = p_inf - sigma*q;
    # it is the form that stays correct when eps_pv/wallis are ported over from 2D.
    rho_l_sat = Float64(s.rho_inf) - sigma*q/a2l
    rho_v_req = Float64(s.rho_inf)/ratio
    rho_v_floor = 1.02*p_v/(Float64(Tmax)*Float64(TEM_CS2))
    rho_v_sat = vfloor ? max(rho_v_req, rho_v_floor) : rho_v_req
    # PRESSURE-driven rates: source is k*(p_v - p), so k = C*U/q (the density form carried an
    # extra a2l because it multiplied a density difference). Timescales are unchanged.
    # rate_ref_pv=true references the rate to p_v (tau = 1/C CTU, p_v-independent — the 2D Tier-C
    # structural fix); false = legacy k = C*U/q (bitwise default).
    krate = rate_ref_pv ? U/max(p_v, 1e-30) : U/q
    kprod = Cprod*krate; kdest = Cdest*krate
    # VAPOR-BRANCH STIFFNESS — the fix that actually BOUNDS the cavity density.
    # MEASURED failure of the old form p_vap = p_v + a2v*rho (a2v = 1e-3*a2l, nearly flat): as rho -> 0 it
    # tends to p_v, i.e. a CONSTANT, so there is no pressure deficit to pull mass back and the cavity
    # evacuates without bound (3D barotrope AND 3D TEM both supercavitated to the numerical floor; note
    # rho_sat(alpha) bounds the equilibrium ALPHA, not rho — at alpha=1 evaporation acts on (1-alpha)=0 so
    # there is no restoring source either).
    # PHYSICAL form: sub-saturated vapor is an ideal gas, p_vap = (p_v/rho_v_sat)*rho -> 0 as rho -> 0.
    # Now under-dense vapor is UNDER-pressured, the surrounding liquid pushes mass back in, and rho is
    # bounded near rho_v_sat by pressure — not by a numerical floor.
    # NOTE this makes a2v = p_v/rho_v_sat, which is STIFF for extreme ratios (1000:1 -> a2v ~ 644), hence
    # ratio ~ 100 is the practical default (a2v ~ 64; still 30x the old 3:1 cap, and consistent with the
    # ~8-100 EFFECTIVE ratio real partial cavities run at).
    a2v = p_v/rho_v_sat
    # psm must stay << BOTH p_v and sigma*q (2D Tier-C rule; with eps_pv, p_v = 0.1*sigma*q is the
    # smaller scale). min() keeps the legacy branch (p_v >> sigma*q never happens there) bitwise.
    psm = psm_frac*min(p_v, sigma*q)
    # Defense in depth (2026-07-30, reconciled 2026-08-11). Known 3D entry points now exit at an
    # independent OPEN/unsound quarantine before construction, and every classified caller pins 0.0 as
    # checked by tools/test_tem3d_quarantine.jl. Keep this constructor diagnostic for any future
    # classified caller or direct module use; it does not authorize either branch for production.
    if wallis >= 1 && eos >= 1 && vaploc <= 0.0
        @warn """3D MIXTURE EOS IS THE KNOWN-DEFECTIVE FROZEN-p_v FORM (wallis=$wallis, eos=$eos).
        The Wallis vapour term uses a/p_v (valid only AT saturation density) instead of a/p at the
        local pressure. Measured at these constants: p crosses zero at rho = rho_v_sat/2 = $(Float32(rho_v_sat)/2)
        and is NEGATIVE over 85.2% of the (alpha,rho) map; the alpha=1 branch is unbounded below
        (-98 p_v at rho=1e-4). It agrees with the correct root to 8e-6 ON the saturation line, so a
        cavity that never evacuates below rho_v_sat/2 is unaffected -- and one that does is not physics.
        => Treat any 3D CAVITATING result from THIS build (vaploc=0) as UNSOUND. Current callers retain
        it only as quarantined historical diagnostic code.
        vaploc=1 calls the SAME shared `_pmix_vl` as 2D, not a copy, but that 3D branch has no current
        GPU/physics validation or production authority."""
    elseif wallis >= 1 && eos >= 1
        @info "3D mixture EOS: shared local-pressure branch IMPLEMENTED but NOT VALIDATED/AUTHORIZED (vaploc=$vaploc)."
    end
    NI, NJ, NK, g = s.NI, s.NJ, s.NK, s.g
    mk(nx,ny,nz) = (a = KA.allocate(s.backend, Float32, nx, ny, nz); fill!(a, 0.0f0); a)
    TEM3D(mk(NI+2g,NJ+2g,NK+2g), mk(NI+2g,NJ+2g,NK+2g), mk(NI,NJ,NK), mk(NI,NJ,NK),
          Float32(p_v), Float32(a2l), Float32(rho_l_sat), Float32(rho_v_sat),
          Float32(kprod), Float32(kdest), Tmin, Tmax, Float32(a2v), Float32(psm), Int32(eos),
          Int32(wallis), Int32(src_jac), Float32(omega), Float32(vaploc))
end

tem_ratio3d(t::TEM3D, s) = Float64(s.rho_inf)/Float64(t.rho_v_sat)

"""Ambient mixture state (p_ref, dp/drho) at (rho_inf, alpha=0) through the SAME _a2m_mix3d +
smooth-max the commit uses (2D tem_ambient twin). Feed straight into set_farfield_mixture3d!."""
function tem_ambient3d(t::TEM3D, s)
    r = Float32(s.rho_inf)
    a2m = _a2m_mix3d(0.0f0, r, t.a2l, t.a2v, t.rho_l_sat, t.p_v, t.wallis)
    pmx = Float64(t.p_v) + Float64(a2m)*(Float64(r) - Float64(t.rho_l_sat))
    if t.eos >= Int32(1)
        sq = sqrt(pmx*pmx + Float64(t.psm)^2)
        (0.5*(pmx + sq), 0.5*Float64(a2m)*(1.0 + pmx/sq))
    else
        (pmx, Float64(a2m))
    end
end

"""(tau_evap, tau_cond) in CHORD-TRAVEL UNITS. Shedding needs tau_cond ~ 0.2-1 CTU vs the ~2 CTU period;
the 2D sweep measured an amplitude RESONANCE at tau_cond ~ 0.4 CTU (=> Cdest ~ 0.05, Cprod ~ 1)."""
function tem_timescales3d(t::TEM3D, s)
    drho = Float64(t.rho_l_sat) - Float64(t.rho_v_sat)
    U = sqrt(Float64(s.uxi)^2 + Float64(s.uyi)^2)
    a2 = Float64(t.a2l)
    (U/(Float64(t.kprod)*a2*drho), U/(Float64(t.kdest)*a2*drho))
end

# ---- BC: wall zero-gradient / wake-cut partner / far-field liquid (per k-plane) ---------------------
@kernel function _k_tem3d_bc_j!(alpha, NI::Int32, NIW::Int32, NIF::Int32, NJ::Int32, NK::Int32, g::Int32)
    i, kk, k = @index(Global, NTuple)
    @inbounds if i <= NI && kk <= g && k <= NK
        ka = k + g
        if i <= NIW || i > NIW+NIF                 # wake cut: partner exchange
            ip = NI+1-i
            alpha[i+g, g+1-kk, ka] = alpha[ip+g, kk+g, ka]
        else                                       # solid wall: zero-gradient
            alpha[i+g, g+1-kk, ka] = alpha[i+g, g+kk, ka]
        end
        alpha[i+g, NJ+g+kk, ka] = 0.0f0            # far-field: pure liquid
    end
end

@kernel function _k_tem3d_bc_i!(alpha, NI::Int32, NJ::Int32, NK::Int32, g::Int32)
    jj, kk, k = @index(Global, NTuple)
    @inbounds if jj <= NJ+2g && kk <= g && k <= NK
        ka = k + g
        alpha[g+1-kk, jj, ka]  = alpha[g+1, jj, ka]      # i-inlet/outlet: zero-gradient
        alpha[NI+g+kk, jj, ka] = alpha[NI+g, jj, ka]
    end
end

# spanwise PERIODIC ghost planes (matches _k_bc_span_3d! in cmesh_span3d.jl)
@kernel function _k_tem3d_bc_span!(alpha, NX::Int32, NY::Int32, NK::Int32, g::Int32)
    i, j, kk = @index(Global, NTuple)
    @inbounds if i <= NX && j <= NY && kk <= g
        alpha[i, j, g+1-kk]  = alpha[i, j, g+NK+1-kk]
        alpha[i, j, g+NK+kk] = alpha[i, j, g+kk]
    end
end

# ---- sweep: upwind advection (i,j contravariant + k spanwise) + point-implicit Merkle + BDF2 --------
@kernel function _k_tem3d_sweep!(alpha_new, @Const(alpha), @Const(rho), @Const(ux), @Const(uy), @Const(uz),
                                 @Const(Six), @Const(Siy), @Const(Sjx), @Const(Sjy), @Const(vol),
                                 @Const(dtf), @Const(un), @Const(um),
                                 rho_l_sat::Float32, rho_v_sat::Float32, kprod::Float32, kdest::Float32,
                                 p_v::Float32, a2l::Float32, a2v::Float32, psm::Float32, wallis::Int32, vaploc::Float32,
                                 src_jac::Int32,
                                 dz::Float32, dtp::Float32, NI::Int32, NJ::Int32, NK::Int32, g::Int32)
    i, j, k = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ && k <= NK
        ia=i+g; ja=j+g; ka=k+g
        a0 = alpha[ia,ja,ka]; r = rho[ia,ja,ka]
        rho_sat = rho_l_sat*(1.0f0-a0) + rho_v_sat*a0
        # PRESSURE-DRIVEN Merkle source — see the 2D twin for the full rationale (density-driven ties the
        # void to liquid compressibility and caps alpha ~ 3%; pressure-driven decouples them).
        # ★★★ EOS SITE 1/3 (3D). vaploc>0 = the EXACT local-pressure-vapour solve via the SHARED
        # `_pmix_vl`; vaploc==0 takes the else-branch, which is the VERBATIM legacy pair => bitwise.
        local pmx::Float32
        if vaploc > 0.0f0
            pmx, _, _ = _pmix_vl(p_v, a0, r, rho_sat, a2l, rho_l_sat, rho_v_sat)
        else
        a2m = _a2m_mix3d(a0, r, a2l, a2v, rho_l_sat, p_v, wallis)
        pmx = p_v + a2m*(r - rho_sat)
        end
        pl  = 0.5f0*(pmx + sqrt(pmx*pmx + psm*psm))
        dpv = p_v - pl
        E = dpv > 0.0f0 ? kprod*dpv : 0.0f0
        D = dpv < 0.0f0 ? kdest*(-dpv) : 0.0f0
        u0=ux[ia,ja,ka]; v0=uy[ia,ja,ka]; w0=uz[ia,ja,ka]
        Uci = (u0*0.5f0*(Six[i,j]+Six[i+1,j]) + v0*0.5f0*(Siy[i,j]+Siy[i+1,j]))/vol[i,j]
        Ucj = (u0*0.5f0*(Sjx[i,j]+Sjx[i,j+1]) + v0*0.5f0*(Sjy[i,j]+Sjy[i,j+1]))/vol[i,j]
        Uck = w0/dz                                       # SPANWISE (uniform dz)
        adv = (Uci >= 0.0f0 ? Uci*(a0-alpha[ia-1,ja,ka]) : Uci*(alpha[ia+1,ja,ka]-a0)) +
              (Ucj >= 0.0f0 ? Ucj*(a0-alpha[ia,ja-1,ka]) : Ucj*(alpha[ia,ja+1,ka]-a0)) +
              (Uck >= 0.0f0 ? Uck*(a0-alpha[ia,ja,ka-1]) : Uck*(alpha[ia,ja,ka+1]-a0))
        dtl = dtf[i,j,k]
        num = a0 + dtl*(E - adv)
        den = 1.0f0 + dtl*(E + D)
        # ★ P1 3D PORT (2026-07-23): the src_jac=3 kink-local Jacobian floor — 2D-PROVEN load-bearing
        # for dense-cavity stability (the legacy diagonal's hole at dpv==0 NaN'd the 2D sigma=0.75
        # marches; mode 3 = local floor passed both 2D gates). Consistency: num += dtl*a0*jacp pairs
        # the den term so the fixed point is untouched (the 2D a0*J discipline).
        if src_jac >= Int32(1)
            wE = dpv > 0.0f0 ? 1.0f0 : 0.0f0
            wD = dpv < 0.0f0 ? 1.0f0 : 0.0f0
            dpda = a2m*(rho_l_sat - rho_v_sat)
            jacp = if src_jac >= Int32(3)
                wband = 1.0f0 - min(abs(dpv)/(0.10f0*p_v), 1.0f0)
                max(dpda*(kprod*wE*(1.0f0-a0) + kdest*wD*a0),
                    wband*dpda*max(kprod*(1.0f0-a0), kdest*a0))
            else
                dpda*(kprod*wE*(1.0f0-a0) + kdest*wD*a0)
            end
            num += dtl*a0*jacp
            den += dtl*jacp
        end
        if dtp > 0.0f0                                    # dual-time BDF2
            h = dtl/(2.0f0*dtp)
            num += h*(4.0f0*un[i,j,k] - um[i,j,k])
            den += 1.5f0*dtl/dtp
        end
        an = num/den
        alpha_new[ia,ja,ka] = an < 0.0f0 ? 0.0f0 : (an > 1.0f0 ? 1.0f0 : an)
    end
end

# ---- commit: alpha <- alpha_new; mixture pressure via the PDP split (T pedestal + pexc) -------------
@kernel function _k_tem3d_commit!(alpha, T, pexc, am, @Const(alpha_new), @Const(rho),
                                  p_v::Float32, a2l::Float32, rho_l_sat::Float32, rho_v_sat::Float32,
                                  a2v::Float32, psm::Float32, eos::Int32, wallis::Int32, vaploc::Float32, omega::Float32,
                                  Tmin::Float32, Tmax::Float32, NI::Int32, NJ::Int32, NK::Int32, g::Int32)
    i, j, k = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ && k <= NK
        ia=i+g; ja=j+g; ka=k+g
        # ★ omega under-relaxation (2D-proven: omega=1 saturates/NaNs segregated dense-cavity marches)
        a0c = alpha[ia,ja,ka]
        a = omega >= 1.0f0 ? alpha_new[ia,ja,ka] : (a0c + omega*(alpha_new[ia,ja,ka] - a0c))
        alpha[ia,ja,ka] = a
        r = rho[ia,ja,ka]
        rho_sat = rho_l_sat*(1.0f0-a) + rho_v_sat*a
        local p::Float32, dpdr::Float32
        if eos >= Int32(1)
            # WALLIS harmonic mixture stiffness. With a2v = p_v/rho_v_sat this ALREADY reduces to the
            # ideal-gas vapor law at alpha=1:  p = p_v + a2v*(r - rho_v_sat) = p_v*r/rho_v_sat  ->  0 as
            # r -> 0, which is exactly the restoring pressure that BOUNDS the cavity density. A separate
            # vapor-plateau branch is therefore redundant - and actively WRONG under a smooth-MAX, because
            # p_vap = a2v*r reaches ~64 in the liquid (a2v=64, r=1) and swamps p_mix~0.67, which drove the
            # whole domain to vapor (measured: Vv 6.6 -> 2218, Cl -> 0). The smooth-max here is now only a
            # POSITIVITY GUARD: smooth max(pmx, 0), which still protects the metastable low-alpha/low-rho
            # corner where the stiff-liquid branch would go strongly negative.
            # ★★★ EOS SITE 2/3 (3D). Same branch structure as site 1; the smooth-max is applied to
            # whichever root the branch produced, so the positivity guard is identical either way.
            local pmx::Float32, dpdr_raw::Float32
            if vaploc > 0.0f0
                pmx, dpdr_raw, _ = _pmix_vl(p_v, a, r, rho_sat, a2l, rho_l_sat, rho_v_sat)
            else
                a2m = _a2m_mix3d(a, r, a2l, a2v, rho_l_sat, p_v, wallis)
                pmx = p_v + a2m*(r - rho_sat)
                dpdr_raw = a2m
            end
            sq  = sqrt(pmx*pmx + psm*psm)
            p    = 0.5f0*(pmx + sq)
            dpdr = 0.5f0*dpdr_raw*(1.0f0 + pmx/sq)
        else
            p = p_v + a2l*(r - rho_sat); dpdr = a2l        # legacy raw-linear (STIFF)
        end
        Tn = p/(r*TEM_CS2)
        Tn = Tn < Tmin ? Tmin : (Tn > Tmax ? Tmax : Tn)
        T[ia,ja,ka] = Tn
        pexc[ia,ja,ka] = p - r*TEM_CS2*Tn                  # PDP excess (Span3D.cav>=1 injects it)
        amf2 = 1.0f-4*a2l                                  # STAGE 1b: positivity-scale floor, NOT a2v
        am[ia,ja,ka] = sqrt(dpdr < amf2 ? amf2 : dpdr)     # mixture sound speed (Layer C)
    end
end

# ★ STAGE-1a 3D PORT (2026-07-23 — closes the 2026-07-18 handover's "pexc ghost fix NOT YET PORTED
# TO 3D" item, the sole correctness blocker for spanwise cav>=1 runs): pexc/am evaluated over the
# FULL ghosted array from the ACTUAL ghost (alpha, rho, T) that the flow/alpha BC sweeps already
# fill — so the i/j/k face fluxes (including the spanwise PERIODIC seam, cmesh_span3d.jl k-flux)
# never read the allocate-zeroed pexc ghosts. Under cav>=1 the interior pexc is O(p_inf); an
# unfilled ghost is a spurious O(p_inf) jump exactly where a spanwise-varying cavity lives.
# Interior write is identical to the commit's. Mirrors _k_pexc_ghost! (cmesh_cav_tem.jl).
@kernel function _k_tem3d_pexc_ghost!(pexc, am, @Const(alpha), @Const(rho), @Const(T),
                                      p_v::Float32, a2l::Float32, rho_l_sat::Float32, rho_v_sat::Float32,
                                      a2v::Float32, psm::Float32, eos::Int32, wallis::Int32, vaploc::Float32,
                                      NX::Int32, NY::Int32, NZ::Int32)
    i, j, k = @index(Global, NTuple)
    @inbounds if i <= NX && j <= NY && k <= NZ
        a = alpha[i,j,k]; a = a < 0.0f0 ? 0.0f0 : (a > 1.0f0 ? 1.0f0 : a)
        r = rho[i,j,k]; r = r < 1.0f-6 ? 1.0f-6 : r
        rho_sat = rho_l_sat*(1.0f0-a) + rho_v_sat*a
        local p::Float32, dpdr::Float32
        if eos >= Int32(1)
            # ★★★ EOS SITE 3/3 (3D) -- the GHOST bands. Must match sites 1-2 exactly or the ghosts
            # carry a different equation of state from the interior, which is its own class of bug.
            local pmx::Float32, dpdr_raw::Float32
            if vaploc > 0.0f0
                pmx, dpdr_raw, _ = _pmix_vl(p_v, a, r, rho_sat, a2l, rho_l_sat, rho_v_sat)
            else
                a2m = _a2m_mix3d(a, r, a2l, a2v, rho_l_sat, p_v, wallis)
                pmx = p_v + a2m*(r - rho_sat)
                dpdr_raw = a2m
            end
            sq  = sqrt(pmx*pmx + psm*psm)
            p = 0.5f0*(pmx + sq); dpdr = 0.5f0*dpdr_raw*(1.0f0 + pmx/sq)
        else
            p = p_v + a2l*(r - rho_sat); dpdr = a2l
        end
        pexc[i,j,k] = p - r*TEM_CS2*T[i,j,k]
        amf2 = 1.0f-4*a2l
        am[i,j,k] = sqrt(dpdr < amf2 ? amf2 : dpdr)
    end
end

@kernel function _k_tem3d_hist!(un, um, @Const(alpha), NI::Int32, NJ::Int32, NK::Int32, g::Int32)
    i, j, k = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ && k <= NK
        um[i,j,k] = un[i,j,k]; un[i,j,k] = alpha[i+g,j+g,k+g]
    end
end

"""One post-step 3D TEM sub-iteration (call every inner iteration AFTER step_3d!, mirrors hem_apply_3d!)."""
function tem_step3d!(s, t::TEM3D; dtp::Float32=0.0f0)
    b = s.backend; NI, NJ, NK, g = Int32(s.NI), Int32(s.NJ), Int32(s.NK), Int32(s.g)
    NX = Int32(s.NI+2s.g); NY = Int32(s.NJ+2s.g)
    _k_tem3d_bc_j!(b, WG3)(t.alpha, NI, Int32(s.NIW), Int32(s.NIF), NJ, NK, g; ndrange=(s.NI, s.g, s.NK))
    _k_tem3d_bc_i!(b, WG3)(t.alpha, NI, NJ, NK, g; ndrange=(s.NJ+2s.g, s.g, s.NK))
    _k_tem3d_bc_span!(b, WG3)(t.alpha, NX, NY, NK, g; ndrange=(s.NI+2s.g, s.NJ+2s.g, s.g))
    _k_tem3d_sweep!(b, WG3)(t.alpha_new, t.alpha, s.rho, s.ux, s.uy, s.uz,
                            s.Six, s.Siy, s.Sjx, s.Sjy, s.vol, s.dtf, t.un, t.um,
                            t.rho_l_sat, t.rho_v_sat, t.kprod, t.kdest,
                            t.p_v, t.a2l, t.a2v, t.psm, t.wallis, t.vaploc, t.src_jac, s.dz, dtp,
                            NI, NJ, NK, g; ndrange=(s.NI, s.NJ, s.NK))
    _k_tem3d_commit!(b, WG3)(t.alpha, s.T, s.pexc, s.am, t.alpha_new, s.rho,
                             t.p_v, t.a2l, t.rho_l_sat, t.rho_v_sat, t.a2v, t.psm, t.eos, t.wallis, t.vaploc, t.omega,
                             t.Tmin, t.Tmax, NI, NJ, NK, g; ndrange=(s.NI, s.NJ, s.NK))
    # STAGE-1a 3D: fill pexc/am over the FULL ghosted array (incl the spanwise periodic seam)
    NZ = Int32(s.NK+2s.g)
    _k_tem3d_pexc_ghost!(b, WG3)(s.pexc, s.am, t.alpha, s.rho, s.T,
                                 t.p_v, t.a2l, t.rho_l_sat, t.rho_v_sat, t.a2v, t.psm, t.eos, t.wallis, t.vaploc,
                                 NX, NY, NZ; ndrange=(s.NI+2s.g, s.NJ+2s.g, s.NK+2s.g))
    nothing
end

"""Rotate the BDF2 history (once per physical step, after the inner loop)."""
function tem_hist_cycle3d!(t::TEM3D, s)
    b = s.backend
    _k_tem3d_hist!(b, WG3)(t.un, t.um, t.alpha, Int32(s.NI), Int32(s.NJ), Int32(s.NK), Int32(s.g);
                           ndrange=(s.NI, s.NJ, s.NK))
    nothing
end

tem_seed3d!(t::TEM3D, s) = (tem_hist_cycle3d!(t, s); tem_hist_cycle3d!(t, s))

end # module
