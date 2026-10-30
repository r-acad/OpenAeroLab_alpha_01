# Shared scalar mixture EOS helpers. Local-pressure roots admit same-type Float32/Float64 inputs.
# Pressure-root algebra, reference-composition semantics and phase admissibility
# are distinct from source integration and experimental material calibration.
# See memory/project_docs/2026-09-08_COMPOSITION_SEMANTICS_AUDIT.md and the reference audit.
module CMeshEOSMix
export _a2m_mix, _pmix_vb, _pmix_vl, _pmix_vl_d, _rhomix_vl

# Algebraic inverse of the guarded local-pressure root. The raw pressure is
# x=p-psm^2/(4p); require p>psm/2 and a finite positive density denominator.
# This low-level helper does not validate those caller obligations.
@inline function _rhomix_vl(p::T, a::T, p_v::T, a2l::T,
                            rho_l_sat::T, rho_v_sat::T, psm::T,
                            lq::T=zero(T)) where {T<:Union{Float32,Float64}}
    x = p - psm*psm/(4.0f0*p)
    L = (1.0f0-a)/(rho_l_sat*a2l)
    aef = a + (1.0f0-a)*lq
    rho_sat = rho_l_sat*(1.0f0-a) + rho_v_sat*a
    den = 1.0f0 - (x-p_v)*(L + aef/x)
    rho_sat/den
end

# Reference-composition stiffness options. wallis=0 uses a harmonic blend;
# wallis>=1 uses 1/[rho*((1-a)/(rl*a2l)+a/pv)], adding ncg/pv to the inner
# compliance sum only when ncg>0. The optional DK
# profile takes precedence and is capped by interpolated endpoint stiffnesses.
# These reference-pressure options are not the local-pressure derivative below.
# Their presence in the API does not establish physical gas or cavitation validity.
@inline function _a2m_mix(a::Float32, r::Float32, a2l::Float32, a2v::Float32,
                          rho_l_sat::Float32, p_v::Float32, wallis::Int32, ncg::Float32,
                          dk_cmin2::Float32, dk_a2v::Float32)
    if dk_cmin2 > 0.0f0
        a2_liq = rho_l_sat*a2l/r
        ends = (1.0f0-a)*a2_liq + a*dk_a2v
        core = dk_cmin2/(2.0f0*sqrt(max(a*(1.0f0-a), 1.0f-12)))
        return min(core, ends)
    elseif wallis >= Int32(1)
        S = (1.0f0-a)/(rho_l_sat*a2l) + a/p_v
        if ncg > 0.0f0
            S += ncg/p_v
        end
        return 1.0f0/(r*S)
    else
        return 1.0f0/((1.0f0-a)/a2l + a/a2v)
    end
end

# Optional bound on the legacy linearized pressure, using max(p_linear,a2v*rho).
# The returned density coefficient freezes the supplied a2m; it is not the total
# derivative when a2m itself depends on rho. The third value disables alpha
# sensitivity on the active bound. This bound can also dominate liquid states;
# it is a diagnostic option and is off in the current local-pressure path.
@inline function _pmix_vb(p_v::Float32, a2m::Float32, r::Float32, rho_sat::Float32,
                          a2v::Float32, vapbound::Float32)
    pmx = p_v + a2m*(r - rho_sat)
    if vapbound > 0.0f0
        pvap = a2v*r
        if pvap > pmx
            return (pvap, a2v, 0.0f0)
        end
    end
    return (pmx, a2m, 1.0f0)
end

# BEGIN _pmix_vl (historical extraction marker)
# Nonnegative root of rho*L*p^2 + B*p - rho*a_eff*pv=0, before the caller's
# smooth pressure guard. L=(1-a)/(rl*a2l), a_eff=a+(1-a)*lq and
# B=rho_sat-rho*(1-a)*(1-lq)-rho*pv*L. rho_sat=(1-a)*rl+a*rv.
# For rho>0, positive material constants, a in[0,1] and lq>=0, the discriminant
# is a sum of nonnegative terms. The stable companion-root formula avoids
# cancellation between B and sqrt(discriminant); it does not remove Float32 coefficient
# cancellation or sensitivity to stored rho/a. At a=1 the equation is linear.
# A zero raw root is possible at a=lq=0 and sufficiently low rho. That extension
# is outside the positive-pressure phase-volume interpretation used by species.
# Callers must validate the required domain; this helper has no input guards.
#
# dpdr follows implicit differentiation of this root (with the denominator
# safeguard shown below). dpda is subsequently CLIPPED for legacy source and p3
# iterations; it is a bounded iteration coefficient, which differs from the true
# EOS alpha derivative where clipping is active. A finite-step source update
# therefore needs its own accuracy test.
@inline function _pmix_vl(p_v::T, a::T, r::T, rho_sat::T,
                          a2l::T, rho_l_sat::T, rho_v_sat::T,
                          lq::T = zero(T)) where {T<:Union{Float32,Float64}}
    Lp = 1.0f0/(rho_l_sat*a2l)                   # = -dL/da, the liquid Wallis coefficient
    L  = (1.0f0 - a)*Lp                          # liquid term of S; independent of p
    aef = a + (1.0f0 - a)*lq                     # a_eff; == a EXACTLY at lq = 0 (bitwise legacy)
    A  = r*L                                     # >= 0
    B  = rho_sat - r*(1.0f0 - a)*(1.0f0 - lq) - r*p_v*L
    C  = -r*aef*p_v                              # <= 0 for a in [0,1] and lq >= 0
    disc = B*B - 4.0f0*A*C                       # == B*B + 4*A*|C| : sum of non-negatives (P1)
    sq   = sqrt(disc)                            # == +dF/dp at the non-negative root
    local p::T
    if A > 0.0f0
        if B >= 0.0f0
            q = -0.5f0*(B + sq)                  # <= 0 ; the non-cancelling companion
            p = q < 0.0f0 ? C/q : 0.0f0          # q == 0 <=> B == 0 and C == 0 <=> double root at 0
        else
            q = -0.5f0*(B - sq)                  # >  0
            p = q/A
        end
    else
        p = B > 0.0f0 ? (-C/B) : 0.0f0           # a == 1: LINEAR, B = rho_v_sat > 0 => p = a2v*r (P3)
    end
    den  = sq > 1.0f-30 ? sq : 1.0f-30
    dpdr = p*rho_sat/(r*den)
    dpda = ((rho_l_sat - rho_v_sat)*p - r*(p - p_v)*((1.0f0 - lq) - Lp*p))/den
    # Keep only the returned alpha coefficient in [0,max(dpdr*(rl-rv),0)].
    # Pressure and dpdr are unchanged by this clamp. The unclipped alpha
    # derivative can reverse sign; this clamp does not prove phase ordering.
    cap  = dpdr*(rho_l_sat - rho_v_sat)                    # >= 0 wherever r > 0 (dpdr >= 0, drho_lv > 0)
    cap  = max(cap, 0.0f0)                                 # unconditional: never clamp UP into negative
    dpda = dpda < 0.0f0 ? 0.0f0 : (dpda > cap ? cap : dpda)
    return (p, dpdr, dpda)
end

# END _pmix_vl

# Single-precision evaluation of the same root (2026-09-12, speed-first on FP32 GPUs).
# The only cancelling quantity of the local-pressure EOS is the liquid density excess
# over saturation, rho_l_sat - r: at low Mach it is a small fraction of r and carries
# the whole pressure. The caller forms d = rho_l_sat - r in Float64 from the
# authoritative state and passes it; every other operation is Float32. With
#   rho_sat = rho_l_sat (1-a) + rho_v_sat a,
#   B = rho_sat - r (1-a)(1-lq) - r p_v L = (1-a)(d + r lq) + rho_v_sat a - r p_v L,
# the quadratic, its root selection and the derivatives are those of _pmix_vl.
@inline function _pmix_vl_d(p_v::Float32, a::Float32, r::Float32, d::Float32,
                            a2l::Float32, rho_l_sat::Float32, rho_v_sat::Float32, lq::Float32)
    Lp = 1.0f0/(rho_l_sat*a2l)
    L  = (1.0f0 - a)*Lp
    aef = a + (1.0f0 - a)*lq
    rho_sat = rho_l_sat*(1.0f0 - a) + rho_v_sat*a
    A  = r*L
    B  = (1.0f0 - a)*(d + r*lq) + rho_v_sat*a - r*p_v*L
    C  = -r*aef*p_v
    disc = B*B - 4.0f0*A*C
    sq   = sqrt(disc)
    local p::Float32
    if A > 0.0f0
        if B >= 0.0f0
            q = -0.5f0*(B + sq)
            p = q < 0.0f0 ? C/q : 0.0f0
        else
            q = -0.5f0*(B - sq)
            p = q/A
        end
    else
        p = B > 0.0f0 ? (-C/B) : 0.0f0
    end
    den  = sq > 1.0f-30 ? sq : 1.0f-30
    dpdr = p*rho_sat/(r*den)
    dpda = ((rho_l_sat - rho_v_sat)*p - r*(p - p_v)*((1.0f0 - lq) - Lp*p))/den
    cap  = dpdr*(rho_l_sat - rho_v_sat)
    cap  = max(cap, 0.0f0)
    dpda = dpda < 0.0f0 ? 0.0f0 : (dpda > cap ? cap : dpda)
    return (p, dpdr, dpda)
end
end # module CMeshEOSMix
