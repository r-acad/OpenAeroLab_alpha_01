struct BLModelParams
    sccon::Float32
    gacon::Float32
    gbcon::Float32
    gccon::Float32
    dlcon::Float32
    ctrcon::Float32
    ctrcex::Float32
    duxcon::Float32
    ctcon::Float32
    cffac::Float32
    hvrat::Float32
    gamma::Float32
end

function BLModelParams(;
    sccon::Real = 5.6f0,
    gacon::Real = 6.70f0,
    gbcon::Real = 0.75f0,
    gccon::Real = 18.0f0,
    dlcon::Real = 0.9f0,
    ctrcon::Real = 1.8f0,
    ctrcex::Real = 3.3f0,
    duxcon::Real = 1.0f0,
    cffac::Real = 1.0f0,
    hvrat::Real = 0.35f0,
    gamma::Real = 1.4f0,
)
    ctcon = 0.5f0 / (Float32(gacon)^2 * Float32(gbcon))
    return BLModelParams(
        Float32(sccon),
        Float32(gacon),
        Float32(gbcon),
        Float32(gccon),
        Float32(dlcon),
        Float32(ctrcon),
        Float32(ctrcex),
        Float32(duxcon),
        ctcon,
        Float32(cffac),
        Float32(hvrat),
        Float32(gamma),
    )
end

struct BLFlowParams
    qinfbl::Float32
    gm1bl::Float32
    hstinv::Float32
    hstinv_ms::Float32
    rstbl::Float32
    rstbl_ms::Float32
    hvrat::Float32
    reybl::Float32
    reybl_ms::Float32
    reybl_re::Float32
end

function BLFlowParams(;
    qinfbl::Real = 1.0f0,
    gamma::Real = 1.4f0,
    hstinv::Real = 0.15f0,
    hstinv_ms::Real = 0.0f0,
    rstbl::Real = 1.0f0,
    rstbl_ms::Real = 0.0f0,
    hvrat::Real = 0.35f0,
    reybl::Real = 1.0f6,
    reybl_ms::Real = 0.0f0,
    reybl_re::Real = 1.0f0,
)
    return BLFlowParams(
        Float32(qinfbl),
        Float32(gamma) - 1.0f0,
        Float32(hstinv),
        Float32(hstinv_ms),
        Float32(rstbl),
        Float32(rstbl_ms),
        Float32(hvrat),
        Float32(reybl),
        Float32(reybl_ms),
        Float32(reybl_re),
    )
end

function _xfoil_bl_flow_params(
    reinf::Real;
    minf::Real = 0.0,
    qinf::Real = 1.0,
    model::BLModelParams = BLModelParams(),
    gamma::Real = model.gamma,
)
    qinfbl = max(Float32(qinf), 1f-12)
    minff = max(Float32(minf), 0.0f0)
    gm1 = Float32(gamma) - 1.0f0
    den = 1.0f0 + 0.5f0 * gm1 * minff^2

    rstbl = den^(1.0f0 / gm1)
    rstbl_ms = 0.5f0 * rstbl / den

    hstinv = gm1 * (minff / qinfbl)^2 / den
    hstinv_ms = gm1 * (1.0f0 / qinfbl)^2 / den - 0.5f0 * gm1 * hstinv / den

    herat = max(1.0f0 - 0.5f0 * qinfbl^2 * hstinv, 1f-12)
    herat_ms = -0.5f0 * qinfbl^2 * hstinv_ms
    reyfac = sqrt(herat^3) * (1.0f0 + model.hvrat) / (herat + model.hvrat)
    reybl = Float32(reinf) * reyfac
    reybl_ms = reybl * (1.5f0 / herat - 1.0f0 / (herat + model.hvrat)) * herat_ms

    return BLFlowParams(
        qinfbl = qinfbl,
        gamma = gamma,
        hstinv = hstinv,
        hstinv_ms = hstinv_ms,
        rstbl = rstbl,
        rstbl_ms = rstbl_ms,
        hvrat = model.hvrat,
        reybl = reybl,
        reybl_ms = reybl_ms,
        reybl_re = reyfac,
    )
end

struct BLKinematicState{T<:Real}
    u2::T
    t2::T
    d2::T
    m2::T
    r2::T
    h2::T
    hk2::T
    rt2::T
    herat::T
    v2::T
end

struct BLClosureState{T<:Real}
    ityp::Symbol
    kin::BLKinematicState{T}
    hc2::T
    hs2::T
    us2::T
    cf2::T
    cq2::T
end

# ── Type-preserving helper: converts literal to match first argument's type ──
@inline _T(x::Real) = typeof(float(x))
@inline _c(ref, val) = oftype(float(ref), val)

function hkin(h::Real, msq::Real)
    hf = float(h)
    mf = float(msq)
    return (hf - _c(hf, 0.29) * mf) / (_c(hf, 1) + _c(hf, 0.113) * mf)
end

function dil(hk::Real, rt::Real)
    hkf = float(hk)
    rtf = max(float(rt), _c(hkf, 1e-12))
    if hkf < _c(hkf, 4)
        return (_c(hkf, 0.00205) * (_c(hkf, 4) - hkf)^_c(hkf, 5.5) + _c(hkf, 0.207)) / rtf
    else
        hkb = hkf - _c(hkf, 4)
        den = _c(hkf, 1) + _c(hkf, 0.02) * hkb^2
        return (-_c(hkf, 0.0016) * hkb^2 / den + _c(hkf, 0.207)) / rtf
    end
end

function hsl(hk::Real, rt::Real, msq::Real)
    hkf = float(hk)
    if hkf < _c(hkf, 4.35)
        tmp = hkf - _c(hkf, 4.35)
        hs =
            _c(hkf, 0.0111) * tmp^2 / (hkf + _c(hkf, 1)) -
            _c(hkf, 0.0278) * tmp^3 / (hkf + _c(hkf, 1)) +
            _c(hkf, 1.528) -
            _c(hkf, 0.0002) * (tmp * hkf)^2
    else
        hs = _c(hkf, 0.015) * (hkf - _c(hkf, 4.35))^2 / hkf + _c(hkf, 1.528)
    end
    return hs
end

function dilw(hk::Real, rt::Real)
    hkf = float(hk)
    rtf = max(float(rt), _c(hkf, 1e-12))
    hs = hsl(hkf, rtf, _c(hkf, 0))
    hs = max(hs, _c(hkf, 1e-12))
    rcd = _c(hkf, 1.10) * (_c(hkf, 1) - _c(hkf, 1) / hkf)^2 / hkf
    return _c(hkf, 2) * rcd / (hs * rtf)
end

function cfl(hk::Real, rt::Real, msq::Real)
    hkf = float(hk)
    rtf = max(float(rt), _c(hkf, 1e-12))
    if hkf < _c(hkf, 5.5)
        tmp = (_c(hkf, 5.5) - hkf)^3 / (hkf + _c(hkf, 1))
        return (_c(hkf, 0.0727) * tmp - _c(hkf, 0.07)) / rtf
    else
        tmp = _c(hkf, 1) - _c(hkf, 1) / (hkf - _c(hkf, 4.5))
        return (_c(hkf, 0.015) * tmp^2 - _c(hkf, 0.07)) / rtf
    end
end

function dit(hs::Real, us::Real, cf::Real, st::Real)
    hsf = max(float(hs), _c(hs, 1e-12))
    usf = float(us)
    cff = float(cf)
    stf = float(st)
    return (_c(hsf, 0.5) * cff * usf + stf * stf * (_c(hsf, 1) - usf)) * _c(hsf, 2) / hsf
end

function hst(hk::Real, rt::Real, msq::Real)
    hkf = float(hk)
    rtf = max(float(rt), _c(hkf, 1e-12))
    msqf = float(msq)
    hsmin = _c(hkf, 1.500)
    dhsinf = _c(hkf, 0.015)

    if rtf > _c(hkf, 400)
        ho = _c(hkf, 3) + _c(hkf, 400) / rtf
    else
        ho = _c(hkf, 4)
    end

    rtz = rtf > _c(hkf, 200) ? rtf : _c(hkf, 200)

    hs =
        if hkf < ho
            hr = (ho - hkf) / (ho - _c(hkf, 1))
            (_c(hkf, 2) - hsmin - _c(hkf, 4) / rtz) * hr^2 * _c(hkf, 1.5) / (hkf + _c(hkf, 0.5)) + hsmin + _c(hkf, 4) / rtz
        else
            grt = log(rtz)
            hdif = hkf - ho
            rtmp = hkf - ho + _c(hkf, 4) / grt
            htmp = _c(hkf, 0.007) * grt / rtmp^2 + dhsinf / hkf
            hdif^2 * htmp + hsmin + _c(hkf, 4) / rtz
        end

    fm = _c(hkf, 1) + _c(hkf, 0.014) * msqf
    return (hs + _c(hkf, 0.028) * msqf) / fm
end

function cft(hk::Real, rt::Real, msq::Real; cffac::Real = 1.0f0, gamma::Real = 1.4f0)
    hkf = float(hk)
    rtf = max(float(rt), _c(hkf, 1e-12))
    msqf = max(float(msq), _c(hkf, 0))
    gm1 = float(gamma) - _c(hkf, 1)

    fc = sqrt(_c(hkf, 1) + _c(hkf, 0.5) * gm1 * msqf)
    grt = log(max(rtf / fc, _c(hkf, 1e-12)))
    grt = max(grt, _c(hkf, 3))

    gex = -_c(hkf, 1.74) - _c(hkf, 0.31) * hkf
    arg = max(-_c(hkf, 20), -_c(hkf, 1.33) * hkf)
    thk = tanh(_c(hkf, 4) - hkf / _c(hkf, 0.875))

    cfo = float(cffac) * _c(hkf, 0.3) * exp(arg) * (grt / _c(hkf, 2.3026))^gex
    return (cfo + _c(hkf, 1.1e-4) * (thk - _c(hkf, 1))) / fc
end

function hct(hk::Real, msq::Real)
    hkf = float(hk)
    msqf = float(msq)
    return msqf * (_c(hkf, 0.064) / (hkf - _c(hkf, 0.8)) + _c(hkf, 0.251))
end

function initial_ctau(hk::Real, model::BLModelParams = BLModelParams())
    hkf = float(hk)
    if hkf <= _c(hkf, 1)
        return _c(hkf, 0)
    end
    return max(_c(hkf, 0), model.ctrcon * exp(-model.ctrcex / (hkf - _c(hkf, 1))))
end

# ============================================================================
# Analytic Jacobian-returning versions of closure functions.
# These match the Fortran BLVAR/BLKIN subroutines exactly, returning both the
# value and its partial derivatives.
# ============================================================================

"""
    hkin_jac(h, msq) → (hk, hk_h, hk_msq)

Kinematic shape parameter with analytic derivatives (Fortran HKIN).
"""
function hkin_jac(h::Real, msq::Real)
    T0 = promote_type(typeof(h), typeof(msq))
    T = T0 <: AbstractFloat ? T0 : Float64

    hf = T(h)
    mf = T(msq)
    den = _c(hf, 1) + _c(hf, 0.113) * mf
    hk = (hf - _c(hf, 0.29) * mf) / den
    hk_h = _c(hf, 1) / den
    hk_msq = (-_c(hf, 0.29) - _c(hf, 0.113) * hk) / den
    return hk, hk_h, hk_msq
end

"""
    hct_jac(hk, msq) → (hc, hc_hk, hc_msq)

Density shape parameter with analytic derivatives (Fortran HCT).
"""
function hct_jac(hk::Real, msq::Real)
    hkf = float(hk)
    mf = float(msq)
    hc = mf * (_c(hkf, 0.064) / (hkf - _c(hkf, 0.8)) + _c(hkf, 0.251))
    hc_hk = mf * (-_c(hkf, 0.064) / (hkf - _c(hkf, 0.8))^2)
    hc_msq = _c(hkf, 0.064) / (hkf - _c(hkf, 0.8)) + _c(hkf, 0.251)
    return hc, hc_hk, hc_msq
end

"""
    hsl_jac(hk, rt, msq) → (hs, hs_hk, hs_rt, hs_msq)

Laminar energy thickness shape parameter with analytic derivatives (Fortran HSL).
"""
function hsl_jac(hk::Real, rt::Real, msq::Real)
    hkf = float(hk)
    if hkf < _c(hkf, 4.35)
        tmp = hkf - _c(hkf, 4.35)
        hs = _c(hkf, 0.0111) * tmp^2 / (hkf + _c(hkf, 1)) -
             _c(hkf, 0.0278) * tmp^3 / (hkf + _c(hkf, 1)) +
             _c(hkf, 1.528) -
             _c(hkf, 0.0002) * (tmp * hkf)^2
        hs_hk = _c(hkf, 0.0111) * (_c(hkf, 2) * tmp - tmp^2 / (hkf + _c(hkf, 1))) / (hkf + _c(hkf, 1)) -
                _c(hkf, 0.0278) * (_c(hkf, 3) * tmp^2 - tmp^3 / (hkf + _c(hkf, 1))) / (hkf + _c(hkf, 1)) -
                _c(hkf, 0.0002) * _c(hkf, 2) * tmp * hkf * (tmp + hkf)
    else
        hs = _c(hkf, 0.015) * (hkf - _c(hkf, 4.35))^2 / hkf + _c(hkf, 1.528)
        hs_hk = _c(hkf, 0.015) * _c(hkf, 2) * (hkf - _c(hkf, 4.35)) / hkf -
                _c(hkf, 0.015) * (hkf - _c(hkf, 4.35))^2 / hkf^2
    end
    z = _c(hkf, 0)
    return hs, hs_hk, z, z
end

"""
    hst_jac(hk, rt, msq) → (hs, hs_hk, hs_rt, hs_msq)

Turbulent energy thickness shape parameter with analytic derivatives (Fortran HST).
"""
function hst_jac(hk::Real, rt::Real, msq::Real)
    hkf = float(hk)
    rtf = max(float(rt), _c(hkf, 1e-12))
    msqf = float(msq)
    hsmin = _c(hkf, 1.500)
    dhsinf = _c(hkf, 0.015)

    if rtf > _c(hkf, 400)
        ho = _c(hkf, 3) + _c(hkf, 400) / rtf
        ho_rt = -_c(hkf, 400) / rtf^2
    else
        ho = _c(hkf, 4)
        ho_rt = _c(hkf, 0)
    end

    if rtf > _c(hkf, 200)
        rtz = rtf
        rtz_rt = _c(hkf, 1)
    else
        rtz = _c(hkf, 200)
        rtz_rt = _c(hkf, 0)
    end

    if hkf < ho
        hr = (ho - hkf) / (ho - _c(hkf, 1))
        hr_hk = -_c(hkf, 1) / (ho - _c(hkf, 1))
        hr_rt = (_c(hkf, 1) - hr) / (ho - _c(hkf, 1)) * ho_rt
        hs = (_c(hkf, 2) - hsmin - _c(hkf, 4) / rtz) * hr^2 * _c(hkf, 1.5) / (hkf + _c(hkf, 0.5)) + hsmin + _c(hkf, 4) / rtz
        hs_hk = -(_c(hkf, 2) - hsmin - _c(hkf, 4) / rtz) * hr^2 * _c(hkf, 1.5) / (hkf + _c(hkf, 0.5))^2 +
                (_c(hkf, 2) - hsmin - _c(hkf, 4) / rtz) * hr * _c(hkf, 2) * _c(hkf, 1.5) / (hkf + _c(hkf, 0.5)) * hr_hk
        hs_rt = (_c(hkf, 2) - hsmin - _c(hkf, 4) / rtz) * hr * _c(hkf, 2) * _c(hkf, 1.5) / (hkf + _c(hkf, 0.5)) * hr_rt +
                (hr^2 * _c(hkf, 1.5) / (hkf + _c(hkf, 0.5)) - _c(hkf, 1)) * _c(hkf, 4) / rtz^2 * rtz_rt
    else
        grt = log(rtz)
        hdif = hkf - ho
        rtmp = hkf - ho + _c(hkf, 4) / grt
        htmp = _c(hkf, 0.007) * grt / rtmp^2 + dhsinf / hkf
        htmp_hk = -_c(hkf, 0.014) * grt / rtmp^3 - dhsinf / hkf^2
        htmp_rt = -_c(hkf, 0.014) * grt / rtmp^3 * (-ho_rt - _c(hkf, 4) / grt^2 / rtz * rtz_rt) +
                  _c(hkf, 0.007) / rtmp^2 / rtz * rtz_rt
        hs = hdif^2 * htmp + hsmin + _c(hkf, 4) / rtz
        hs_hk = hdif * _c(hkf, 2) * htmp + hdif^2 * htmp_hk
        hs_rt = hdif^2 * htmp_rt - _c(hkf, 4) / rtz^2 * rtz_rt +
                hdif * _c(hkf, 2) * htmp * (-ho_rt)
    end

    fm = _c(hkf, 1) + _c(hkf, 0.014) * msqf
    hs_out = (hs + _c(hkf, 0.028) * msqf) / fm
    hs_hk_out = hs_hk / fm
    hs_rt_out = hs_rt / fm
    hs_msq_out = _c(hkf, 0.028) / fm - _c(hkf, 0.014) * hs_out / fm
    return hs_out, hs_hk_out, hs_rt_out, hs_msq_out
end

"""
    cfl_jac(hk, rt, msq) → (cf, cf_hk, cf_rt, cf_msq)

Laminar skin friction with analytic derivatives (Fortran CFL).
"""
function cfl_jac(hk::Real, rt::Real, msq::Real)
    hkf = float(hk)
    rtf = max(float(rt), _c(hkf, 1e-12))
    if hkf < _c(hkf, 5.5)
        tmp = (_c(hkf, 5.5) - hkf)^3 / (hkf + _c(hkf, 1))
        cf = (_c(hkf, 0.0727) * tmp - _c(hkf, 0.07)) / rtf
        cf_hk = (-_c(hkf, 0.0727) * tmp * _c(hkf, 3) / (_c(hkf, 5.5) - hkf) - _c(hkf, 0.0727) * tmp / (hkf + _c(hkf, 1))) / rtf
    else
        tmp = _c(hkf, 1) - _c(hkf, 1) / (hkf - _c(hkf, 4.5))
        cf = (_c(hkf, 0.015) * tmp^2 - _c(hkf, 0.07)) / rtf
        cf_hk = (_c(hkf, 0.015) * tmp * _c(hkf, 2) / (hkf - _c(hkf, 4.5))^2) / rtf
    end
    cf_rt = -cf / rtf
    cf_msq = _c(hkf, 0)
    return cf, cf_hk, cf_rt, cf_msq
end

"""
    cft_jac(hk, rt, msq; cffac, gamma) → (cf, cf_hk, cf_rt, cf_msq)

Turbulent skin friction with analytic derivatives (Fortran CFT).
"""
function cft_jac(hk::Real, rt::Real, msq::Real; cffac::Real = 1.0f0, gamma::Real = 1.4f0)
    hkf = float(hk)
    rtf = max(float(rt), _c(hkf, 1e-12))
    msqf = max(float(msq), _c(hkf, 0))
    gm1 = float(gamma) - _c(hkf, 1)

    fc = sqrt(_c(hkf, 1) + _c(hkf, 0.5) * gm1 * msqf)
    grt = log(max(rtf / fc, _c(hkf, 1e-12)))
    grt = max(grt, _c(hkf, 3))

    gex = -_c(hkf, 1.74) - _c(hkf, 0.31) * hkf
    arg = max(-_c(hkf, 20), -_c(hkf, 1.33) * hkf)
    thk = tanh(_c(hkf, 4) - hkf / _c(hkf, 0.875))

    cfo = float(cffac) * _c(hkf, 0.3) * exp(arg) * (grt / _c(hkf, 2.3026))^gex
    cf = (cfo + _c(hkf, 1.1e-4) * (thk - _c(hkf, 1))) / fc
    cf_hk = (-_c(hkf, 1.33) * cfo - _c(hkf, 0.31) * log(grt / _c(hkf, 2.3026)) * cfo -
             _c(hkf, 1.1e-4) * (_c(hkf, 1) - thk^2) / _c(hkf, 0.875)) / fc
    cf_rt = gex * cfo / (fc * grt) / rtf
    cf_msq = gex * cfo / (fc * grt) * (-_c(hkf, 0.25) * gm1 / fc^2) - _c(hkf, 0.25) * gm1 * cf / fc^2
    return cf, cf_hk, cf_rt, cf_msq
end

"""
    dil_jac(hk, rt) → (di, di_hk, di_rt)

Laminar dissipation function with analytic derivatives (Fortran DIL).
"""
function dil_jac(hk::Real, rt::Real)
    hkf = float(hk)
    rtf = max(float(rt), _c(hkf, 1e-12))
    if hkf < _c(hkf, 4)
        di = (_c(hkf, 0.00205) * (_c(hkf, 4) - hkf)^_c(hkf, 5.5) + _c(hkf, 0.207)) / rtf
        di_hk = (-_c(hkf, 0.00205) * _c(hkf, 5.5) * (_c(hkf, 4) - hkf)^_c(hkf, 4.5)) / rtf
    else
        hkb = hkf - _c(hkf, 4)
        den = _c(hkf, 1) + _c(hkf, 0.02) * hkb^2
        di = (-_c(hkf, 0.0016) * hkb^2 / den + _c(hkf, 0.207)) / rtf
        di_hk = (-_c(hkf, 0.0016) * _c(hkf, 2) * hkb * (_c(hkf, 1) / den - _c(hkf, 0.02) * hkb^2 / den^2)) / rtf
    end
    di_rt = -di / rtf
    return di, di_hk, di_rt
end

"""
    dilw_jac(hk, rt) → (di, di_hk, di_rt)

Laminar wake dissipation function with analytic derivatives (Fortran DILW).
"""
function dilw_jac(hk::Real, rt::Real)
    hkf = float(hk)
    rtf = max(float(rt), _c(hkf, 1e-12))
    hs, hs_hk, _, _ = hsl_jac(hkf, rtf, _c(hkf, 0))

    rcd = _c(hkf, 1.10) * (_c(hkf, 1) - _c(hkf, 1) / hkf)^2 / hkf
    rcd_hk = -_c(hkf, 1.10) * (_c(hkf, 1) - _c(hkf, 1) / hkf) * _c(hkf, 2) / hkf^3 - rcd / hkf

    di = _c(hkf, 2) * rcd / (hs * rtf)
    di_hk = _c(hkf, 2) * rcd_hk / (hs * rtf) - (di / hs) * hs_hk
    di_rt = -di / rtf
    return di, di_hk, di_rt
end

"""
    dampl_jac(hk, th, rt) → (ax, ax_hk, ax_th, ax_rt)

Envelope e^n amplification rate with analytic derivatives (Fortran DAMPL).
"""
function dampl_jac(hk::Real, th::Real, rt::Real)
    T0 = promote_type(typeof(hk), typeof(th), typeof(rt))
    T = T0 <: AbstractFloat ? T0 : Float64
    roundstep(x) = T(x)

    hkf = roundstep(hk)
    thf = max(roundstep(th), _c(hkf, 1e-12))
    rtf = max(roundstep(rt), _c(hkf, 1e-12))
    dgr = _c(hkf, 0.08)

    hmi = roundstep(_c(hkf, 1) / roundstep(hkf - _c(hkf, 1)))
    hmi_hk = roundstep(-roundstep(hmi^2))

    aa = roundstep(_c(hkf, 2.492) * roundstep(hmi^_c(hkf, 0.43)))
    aa_hk = roundstep(roundstep(aa / hmi) * _c(hkf, 0.43) * hmi_hk)

    bb = roundstep(tanh(roundstep(_c(hkf, 14) * hmi - _c(hkf, 9.24))))
    bb_hk = roundstep(roundstep(_c(hkf, 1) - roundstep(bb^2)) * _c(hkf, 14) * hmi_hk)

    grcrit = roundstep(aa + _c(hkf, 0.7) * roundstep(bb + _c(hkf, 1)))
    grc_hk = roundstep(aa_hk + _c(hkf, 0.7) * bb_hk)

    gr = roundstep(log10(rtf))
    gr_rt = roundstep(_c(hkf, 1) / roundstep(_c(hkf, 2.3025851) * rtf))

    if gr < grcrit - dgr
        z = _c(hkf, 0)
        return z, z, z, z
    end

    rnorm = roundstep(roundstep(gr - roundstep(grcrit - dgr)) / roundstep(_c(hkf, 2) * dgr))
    rn_hk = roundstep(-grc_hk / roundstep(_c(hkf, 2) * dgr))
    rn_rt = roundstep(gr_rt / roundstep(_c(hkf, 2) * dgr))

    if rnorm >= _c(hkf, 1)
        rfac = _c(hkf, 1)
        rfac_hk = _c(hkf, 0)
        rfac_rt = _c(hkf, 0)
    else
        rfac = roundstep(_c(hkf, 3) * roundstep(rnorm^2) - _c(hkf, 2) * roundstep(rnorm^3))
        rfac_rn = roundstep(_c(hkf, 6) * rnorm - _c(hkf, 6) * roundstep(rnorm^2))
        rfac_hk = roundstep(rfac_rn * rn_hk)
        rfac_rt = roundstep(rfac_rn * rn_rt)
    end

    arg = roundstep(_c(hkf, 3.87) * hmi - _c(hkf, 2.52))
    arg_hk = roundstep(_c(hkf, 3.87) * hmi_hk)

    ex = roundstep(exp(-roundstep(arg^2)))
    ex_hk = roundstep(ex * roundstep(-_c(hkf, 2) * arg * arg_hk))

    dadr = roundstep(_c(hkf, 0.028) * roundstep(hkf - _c(hkf, 1)) - _c(hkf, 0.0345) * ex)
    dadr_hk = roundstep(_c(hkf, 0.028) - _c(hkf, 0.0345) * ex_hk)

    af = roundstep(
        -_c(hkf, 0.05) +
        _c(hkf, 2.7) * hmi -
        _c(hkf, 5.5) * roundstep(hmi^2) +
        _c(hkf, 3) * roundstep(hmi^3),
    )
    af_hmi = roundstep(_c(hkf, 2.7) - _c(hkf, 11) * hmi + _c(hkf, 9) * roundstep(hmi^2))
    af_hk = roundstep(af_hmi * hmi_hk)

    ax_base = roundstep(af * dadr / thf)
    ax = roundstep(ax_base * rfac)
    ax_hk = roundstep(roundstep(roundstep(af_hk * dadr / thf) + roundstep(af * dadr_hk / thf)) * rfac + ax_base * rfac_hk)
    ax_th = roundstep(-ax / thf)
    ax_rt = roundstep(ax_base * rfac_rt)

    return ax, ax_hk, ax_th, ax_rt
end

"""
    dampl2_jac(hk, th, rt) → (ax, ax_hk, ax_th, ax_rt)

Modified envelope e^n amplification rate with analytic derivatives (Fortran DAMPL2).
"""
function dampl2_jac(hk::Real, th::Real, rt::Real)
    T0 = promote_type(typeof(hk), typeof(th), typeof(rt))
    T = T0 <: AbstractFloat ? T0 : Float64
    roundstep(x) = T(x)

    hkf = max(roundstep(hk), _c(T(hk), 1.0001))
    thf = max(roundstep(th), _c(hkf, 1e-12))
    rtf = max(roundstep(rt), _c(hkf, 1e-12))
    dgr = _c(hkf, 0.08)
    hk1c = _c(hkf, 3.5)
    hk2c = _c(hkf, 4)

    hmi = roundstep(_c(hkf, 1) / roundstep(hkf - _c(hkf, 1)))
    hmi_hk = roundstep(-roundstep(hmi^2))

    aa = roundstep(_c(hkf, 2.492) * roundstep(hmi^_c(hkf, 0.43)))
    aa_hk = roundstep(roundstep(aa / hmi) * _c(hkf, 0.43) * hmi_hk)

    bb = roundstep(tanh(roundstep(_c(hkf, 14) * hmi - _c(hkf, 9.24))))
    bb_hk = roundstep(roundstep(_c(hkf, 1) - roundstep(bb^2)) * _c(hkf, 14) * hmi_hk)

    grc = roundstep(aa + _c(hkf, 0.7) * roundstep(bb + _c(hkf, 1)))
    grc_hk = roundstep(aa_hk + _c(hkf, 0.7) * bb_hk)

    gr = roundstep(log10(rtf))
    gr_rt = roundstep(_c(hkf, 1) / roundstep(_c(hkf, 2.3025851) * rtf))

    if gr < grc - dgr
        z = _c(hkf, 0)
        return z, z, z, z
    end

    rnorm = roundstep(roundstep(gr - roundstep(grc - dgr)) / roundstep(_c(hkf, 2) * dgr))
    rn_hk = roundstep(-grc_hk / roundstep(_c(hkf, 2) * dgr))
    rn_rt = roundstep(gr_rt / roundstep(_c(hkf, 2) * dgr))

    if rnorm >= _c(hkf, 1)
        rfac = _c(hkf, 1)
        rfac_hk = _c(hkf, 0)
        rfac_rt = _c(hkf, 0)
    else
        rfac = roundstep(_c(hkf, 3) * roundstep(rnorm^2) - _c(hkf, 2) * roundstep(rnorm^3))
        rfac_rn = roundstep(_c(hkf, 6) * rnorm - _c(hkf, 6) * roundstep(rnorm^2))
        rfac_hk = roundstep(rfac_rn * rn_hk)
        rfac_rt = roundstep(rfac_rn * rn_rt)
    end

    arg = roundstep(_c(hkf, 3.87) * hmi - _c(hkf, 2.52))
    arg_hk = roundstep(_c(hkf, 3.87) * hmi_hk)

    ex = roundstep(exp(-roundstep(arg^2)))
    ex_hk = roundstep(ex * roundstep(-_c(hkf, 2) * arg * arg_hk))

    dadr = roundstep(_c(hkf, 0.028) * roundstep(hkf - _c(hkf, 1)) - _c(hkf, 0.0345) * ex)
    dadr_hk = roundstep(_c(hkf, 0.028) - _c(hkf, 0.0345) * ex_hk)

    brg = roundstep(-_c(hkf, 20) * hmi)
    ebrg = roundstep(exp(brg))
    af = roundstep(
        -_c(hkf, 0.05) +
        _c(hkf, 2.7) * hmi -
        _c(hkf, 5.5) * roundstep(hmi^2) +
        _c(hkf, 3) * roundstep(hmi^3) +
        _c(hkf, 0.1) * ebrg,
    )
    af_hmi = roundstep(_c(hkf, 2.7) - _c(hkf, 11) * hmi + _c(hkf, 9) * roundstep(hmi^2) - _c(hkf, 2) * ebrg)
    af_hk = roundstep(af_hmi * hmi_hk)

    ax1_base = roundstep(af * dadr / thf)
    ax1 = roundstep(ax1_base * rfac)
    ax1_hk = roundstep(roundstep(roundstep(af_hk * dadr / thf) + roundstep(af * dadr_hk / thf)) * rfac + ax1_base * rfac_hk)
    ax1_th = roundstep(-ax1 / thf)
    ax1_rt = roundstep(ax1_base * rfac_rt)

    if hkf < hk1c
        return ax1, ax1_hk, ax1_th, ax1_rt
    end

    hnorm = roundstep(roundstep(hkf - hk1c) / roundstep(hk2c - hk1c))
    hn_hk = roundstep(_c(hkf, 1) / roundstep(hk2c - hk1c))

    if hnorm >= _c(hkf, 1)
        hfac = _c(hkf, 1)
        hf_hk = _c(hkf, 0)
    else
        hfac = roundstep(_c(hkf, 3) * roundstep(hnorm^2) - _c(hkf, 2) * roundstep(hnorm^3))
        hf_hk = roundstep(roundstep(_c(hkf, 6) * hnorm - _c(hkf, 6) * roundstep(hnorm^2)) * hn_hk)
    end

    gr0_exp = roundstep(exp(roundstep(-_c(hkf, 0.15) * roundstep(hkf - _c(hkf, 5)))))
    gr0 = roundstep(_c(hkf, 0.30) + _c(hkf, 0.35) * gr0_exp)
    gr0_hk = roundstep(-_c(hkf, 0.35) * gr0_exp * _c(hkf, 0.15))

    tnr = roundstep(tanh(roundstep(_c(hkf, 1.2) * roundstep(gr - gr0))))
    tnr_rt = roundstep(roundstep(_c(hkf, 1) - roundstep(tnr^2)) * _c(hkf, 1.2) * gr_rt)
    tnr_hk = roundstep(-roundstep(_c(hkf, 1) - roundstep(tnr^2)) * _c(hkf, 1.2) * gr0_hk)

    hkxm1 = roundstep(hkf - _c(hkf, 1))
    ax2 = roundstep(roundstep(_c(hkf, 0.086) * tnr - _c(hkf, 0.25) / roundstep(hkxm1^_c(hkf, 1.5))) / thf)
    ax2_hk = roundstep(roundstep(_c(hkf, 0.086) * tnr_hk + _c(hkf, 1.5) * _c(hkf, 0.25) / roundstep(hkxm1^_c(hkf, 2.5))) / thf)
    ax2_rt = roundstep(roundstep(_c(hkf, 0.086) * tnr_rt) / thf)
    ax2_th = roundstep(-ax2 / thf)

    if ax2 < _c(hkf, 0)
        ax2 = _c(hkf, 0)
        ax2_hk = _c(hkf, 0)
        ax2_rt = _c(hkf, 0)
        ax2_th = _c(hkf, 0)
    end

    ax = roundstep(hfac * ax2 + roundstep(_c(hkf, 1) - hfac) * ax1)
    ax_hk = roundstep(hfac * ax2_hk + roundstep(_c(hkf, 1) - hfac) * ax1_hk + hf_hk * roundstep(ax2 - ax1))
    ax_rt = roundstep(hfac * ax2_rt + roundstep(_c(hkf, 1) - hfac) * ax1_rt)
    ax_th = roundstep(hfac * ax2_th + roundstep(_c(hkf, 1) - hfac) * ax1_th)

    return ax, ax_hk, ax_th, ax_rt
end

function blkin_state(
    u2::Real,
    t2::Real,
    d2::Real,
    flow::BLFlowParams = BLFlowParams(),
)
    u2f = float(u2)
    t2f = max(float(t2), _c(u2f, 1e-12))
    d2f = max(float(d2), _c(u2f, 1e-12))

    denom = flow.gm1bl * (_c(u2f, 1) - _c(u2f, 0.5) * u2f * u2f * flow.hstinv)
    denom = abs(denom) < _c(u2f, 1e-12) ? sign(denom + _c(u2f, 1e-12)) * _c(u2f, 1e-12) : denom

    m2 = u2f * u2f * flow.hstinv / denom
    tr2 = _c(u2f, 1) + _c(u2f, 0.5) * flow.gm1bl * m2
    tr2 = max(tr2, _c(u2f, 1e-12))

    r2 = flow.rstbl * tr2^(-_c(u2f, 1) / flow.gm1bl)
    h2 = d2f / t2f

    herat = _c(u2f, 1) - _c(u2f, 0.5) * u2f * u2f * flow.hstinv
    herat = max(herat, _c(u2f, 1e-12))
    v2 = sqrt(herat^3) * (_c(u2f, 1) + flow.hvrat) / (herat + flow.hvrat) / flow.reybl
    v2 = max(v2, _c(u2f, 1e-16))

    hk2 = hkin(h2, m2)
    rt2 = r2 * u2f * t2f / v2

    return BLKinematicState(u2f, t2f, d2f, m2, r2, h2, hk2, rt2, herat, v2)
end

function blvar_closure(
    u2::Real,
    t2::Real,
    d2::Real;
    ityp::Symbol = :laminar,
    flow::BLFlowParams = BLFlowParams(),
    model::BLModelParams = BLModelParams(),
)
    ityp in (:laminar, :turbulent, :wake) ||
        throw(ArgumentError("ityp must be :laminar, :turbulent, or :wake"))

    kin0 = blkin_state(u2, t2, d2, flow)
    hk2 =
        ityp == :wake ? max(kin0.hk2, _c(kin0.u2, 1.00005)) :
        max(kin0.hk2, _c(kin0.u2, 1.05))

    kin = BLKinematicState(
        kin0.u2,
        kin0.t2,
        kin0.d2,
        kin0.m2,
        kin0.r2,
        kin0.h2,
        hk2,
        kin0.rt2,
        kin0.herat,
        kin0.v2,
    )

    hc2 = hct(hk2, kin.m2)
    hs2 = ityp == :laminar ? hsl(hk2, kin.rt2, kin.m2) : hst(hk2, kin.rt2, kin.m2)

    h2 = max(kin.h2, _c(kin.u2, 1e-12))
    us2 = _c(kin.u2, 0.5) * hs2 * (_c(kin.u2, 1) - (hk2 - _c(kin.u2, 1)) / (model.gbcon * h2))
    if ityp != :wake && us2 > _c(kin.u2, 0.95)
        us2 = _c(kin.u2, 0.98)
    elseif ityp == :wake && us2 > _c(kin.u2, 0.99995)
        us2 = _c(kin.u2, 0.99995)
    end

    cf2 =
        if ityp == :wake
            _c(kin.u2, 0)
        elseif ityp == :laminar
            cfl(hk2, kin.rt2, kin.m2)
        else
            cftv = cft(hk2, kin.rt2, kin.m2; cffac = model.cffac, gamma = model.gamma)
            cflv = cfl(hk2, kin.rt2, kin.m2)
            max(cftv, cflv)
        end

    rt2 = max(kin.rt2, _c(kin.u2, 1e-12))
    hk2s = max(hk2, _c(kin.u2, 1e-12))
    hkb = hk2s - _c(kin.u2, 1)
    hkc =
        if ityp == :turbulent
            max(hk2s - _c(kin.u2, 1) - model.gccon / rt2, _c(kin.u2, 0.01))
        else
            hk2s - _c(kin.u2, 1)
        end
    usb = max(_c(kin.u2, 1) - us2, _c(kin.u2, 1e-8))
    cqarg = model.ctcon * hs2 * hkb * hkc^2 / (usb * h2 * hk2s^2)
    cq2 = cqarg <= _c(kin.u2, 0) ? _c(kin.u2, 0) : sqrt(cqarg)

    return BLClosureState(ityp, kin, hc2, hs2, us2, cf2, cq2)
end
