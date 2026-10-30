struct BLPrimaryState{T<:Real}
    x::T
    u::T
    t::T
    d::T
    s::T
    ampl::T
    dw::T
end

# Keyword constructor
function BLPrimaryState(
    x::Real,
    u::Real,
    t::Real,
    d::Real;
    s::Real = zero(float(x)),
    ampl::Real = zero(float(x)),
    dw::Real = zero(float(x)),
)
    T = promote_type(typeof(float(x)), typeof(float(u)), typeof(float(t)),
                     typeof(float(d)), typeof(float(s)), typeof(float(ampl)),
                     typeof(float(dw)))
    return BLPrimaryState{T}(
        T(float(x)),
        T(float(u)),
        T(float(t)),
        T(float(d)),
        T(float(s)),
        T(float(ampl)),
        T(float(dw)),
    )
end

# Positional constructor with automatic type promotion (handles mixed Float32/Float64)
function BLPrimaryState(x::Real, u::Real, t::Real, d::Real, s::Real, ampl::Real, dw::Real)
    T = promote_type(typeof(float(x)), typeof(float(u)), typeof(float(t)),
                     typeof(float(d)), typeof(float(s)), typeof(float(ampl)),
                     typeof(float(dw)))
    return BLPrimaryState{T}(T(x), T(u), T(t), T(d), T(s), T(ampl), T(dw))
end

struct BLSecondaryState{T<:Real}
    h::T
    m::T
    hk::T
    hs::T
    hc::T
    rt::T
    cf::T
    cq::T
    di::T
    de::T
    us::T
end

function _ityp_symbol(ityp::Symbol)
    ityp in (:similarity, :laminar, :turbulent, :wake) ||
        throw(ArgumentError("Invalid ityp symbol: $ityp"))
    return ityp
end

function _ityp_symbol(ityp::Integer)
    ityp == 0 && return :similarity
    ityp == 1 && return :laminar
    ityp == 2 && return :turbulent
    ityp == 3 && return :wake
    throw(ArgumentError("Invalid ityp code: $ityp"))
end

function dampl(hk::Real, th::Real, rt::Real)
    hkf = float(hk)
    thf = max(float(th), _c(hkf, 1e-12))
    rtf = max(float(rt), _c(hkf, 1e-12))
    dgr = _c(hkf, 0.08)

    hmi = _c(hkf, 1) / (hkf - _c(hkf, 1))
    aa = _c(hkf, 2.492) * hmi^_c(hkf, 0.43)
    bb = tanh(_c(hkf, 14) * hmi - _c(hkf, 9.24))
    grcrit = aa + _c(hkf, 0.7) * (bb + _c(hkf, 1))

    gr = log10(rtf)
    if gr < grcrit - dgr
        return _c(hkf, 0)
    end

    rfac =
        if gr >= grcrit + dgr
            _c(hkf, 1)
        else
            rnorm = (gr - (grcrit - dgr)) / (_c(hkf, 2) * dgr)
            _c(hkf, 3) * rnorm^2 - _c(hkf, 2) * rnorm^3
        end

    arg = _c(hkf, 3.87) * hmi - _c(hkf, 2.52)
    ex = exp(-(arg^2))
    dadr = _c(hkf, 0.028) * (hkf - _c(hkf, 1)) - _c(hkf, 0.0345) * ex

    af = -_c(hkf, 0.05) + _c(hkf, 2.7) * hmi - _c(hkf, 5.5) * hmi^2 + _c(hkf, 3) * hmi^3
    ax = af * dadr / thf * rfac
    return max(ax, _c(hkf, 0))
end

function dampl2(hk::Real, th::Real, rt::Real)
    hkf = max(float(hk), _c(hk, 1.0001))
    thf = max(float(th), _c(hkf, 1e-12))
    rtf = max(float(rt), _c(hkf, 1e-12))
    dgr = _c(hkf, 0.08)
    hk1 = _c(hkf, 3.5)
    hk2 = _c(hkf, 4)

    hmi = _c(hkf, 1) / (hkf - _c(hkf, 1))
    aa = _c(hkf, 2.492) * hmi^_c(hkf, 0.43)
    bb = tanh(_c(hkf, 14) * hmi - _c(hkf, 9.24))
    grcrit = aa + _c(hkf, 0.7) * (bb + _c(hkf, 1))

    gr = log10(rtf)
    if gr < grcrit - dgr
        return _c(hkf, 0)
    end

    rfac =
        if gr >= grcrit + dgr
            _c(hkf, 1)
        else
            rnorm = (gr - (grcrit - dgr)) / (_c(hkf, 2) * dgr)
            _c(hkf, 3) * rnorm^2 - _c(hkf, 2) * rnorm^3
        end

    arg = _c(hkf, 3.87) * hmi - _c(hkf, 2.52)
    ex = exp(-(arg^2))
    dadr = _c(hkf, 0.028) * (hkf - _c(hkf, 1)) - _c(hkf, 0.0345) * ex

    brg = -_c(hkf, 20) * hmi
    af = -_c(hkf, 0.05) + _c(hkf, 2.7) * hmi - _c(hkf, 5.5) * hmi^2 + _c(hkf, 3) * hmi^3 + _c(hkf, 0.1) * exp(brg)
    ax1 = max(af * dadr / thf * rfac, _c(hkf, 0))

    if hkf < hk1
        return ax1
    end

    hnorm = (hkf - hk1) / (hk2 - hk1)
    hfac =
        if hnorm >= _c(hkf, 1)
            _c(hkf, 1)
        else
            _c(hkf, 3) * hnorm^2 - _c(hkf, 2) * hnorm^3
        end

    gr0 = _c(hkf, 0.30) + _c(hkf, 0.35) * exp(-_c(hkf, 0.15) * (hkf - _c(hkf, 5)))
    tnr = tanh(_c(hkf, 1.2) * (gr - gr0))
    ax2 = (_c(hkf, 0.086) * tnr - _c(hkf, 0.25) / (hkf - _c(hkf, 1))^_c(hkf, 1.5)) / thf
    ax2 = max(ax2, _c(hkf, 0))

    return hfac * ax2 + (_c(hkf, 1) - hfac) * ax1
end

function axset(
    hk1::Real,
    t1::Real,
    rt1::Real,
    a1::Real,
    hk2::Real,
    t2::Real,
    rt2::Real,
    a2::Real,
    acrit::Real;
    idampv::Integer = 0,
)
    ax1 =
        idampv == 0 ? dampl(hk1, t1, rt1) :
        dampl2(hk1, t1, rt1)
    ax2 =
        idampv == 0 ? dampl(hk2, t2, rt2) :
        dampl2(hk2, t2, rt2)

    T = typeof(float(hk1))
    axsq = T(0.5) * (ax1^2 + ax2^2)
    axa = axsq <= zero(T) ? zero(T) : sqrt(axsq)

    arg = min(T(20) * (T(acrit) - T(0.5) * (T(a1) + T(a2))), T(20))
    exn = arg <= zero(T) ? one(T) : exp(-arg)
    dax = exn * T(0.002) / (max(T(t1) + T(t2), T(1e-12)))

    return axa + dax
end

function blsecondary_state(
    primary::BLPrimaryState,
    ityp::Union{Symbol,Integer};
    flow::BLFlowParams = BLFlowParams(),
    model::BLModelParams = BLModelParams(),
)
    mode = _ityp_symbol(ityp)
    closure_mode = mode == :similarity ? :laminar : mode

    # Fortran BLPRV: D2 = DSI - DSWAKI (gap-subtracted for H computation)
    d_eff = primary.d - primary.dw
    cl = blvar_closure(
        primary.u,
        primary.t,
        d_eff;
        ityp = closure_mode,
        flow = flow,
        model = model,
    )
    kin = cl.kin

    hk = cl.kin.hk2
    hs = cl.hs2
    us = cl.us2
    rt = max(cl.kin.rt2, _c(hk, 1e-12))
    cf = cl.cf2
    s = max(primary.s, _c(hk, 0))

    di =
        if mode == :laminar || mode == :similarity
            dil(hk, rt)
        elseif mode == :turbulent
            cf2t = cft(hk, rt, kin.m2; cffac = model.cffac, gamma = model.gamma)
            dit_wall = (_c(hk, 0.5) * cf2t * us) * _c(hk, 2) / max(hs, _c(hk, 1e-12))

            grt = log(rt)
            hmin = _c(hk, 1) + _c(hk, 2.1) / max(grt, _c(hk, 1e-12))
            fl = (hk - _c(hk, 1)) / max(hmin - _c(hk, 1), _c(hk, 1e-12))
            dfac = _c(hk, 0.5) + _c(hk, 0.5) * tanh(fl)

            dit_wall *= dfac

            dd_outer = s^2 * (_c(hk, 0.995) - us) * _c(hk, 2) / max(hs, _c(hk, 1e-12))
            dd_lam = _c(hk, 0.15) * (_c(hk, 0.995) - us)^2 / rt * _c(hk, 2) / max(hs, _c(hk, 1e-12))

            max(dit_wall + dd_outer + dd_lam, dil(hk, rt))
        else
            dd_outer = s^2 * (_c(hk, 0.995) - us) * _c(hk, 2) / max(hs, _c(hk, 1e-12))
            dd_lam = _c(hk, 0.15) * (_c(hk, 0.995) - us)^2 / rt * _c(hk, 2) / max(hs, _c(hk, 1e-12))
            _c(hk, 2) * max(dd_outer + dd_lam, dilw(hk, rt))
        end

    hdmx = _c(hk, 12)
    de = (_c(hk, 3.15) + _c(hk, 1.72) / max(hk - _c(hk, 1), _c(hk, 1e-8))) * primary.t + primary.d
    if de > hdmx * primary.t
        de = hdmx * primary.t
    end

    return BLSecondaryState(
        kin.h2,
        kin.m2,
        hk,
        hs,
        cl.hc2,
        rt,
        cf,
        cl.cq2,
        max(di, _c(hk, 0)),
        max(de, _c(hk, 0)),
        us,
    )
end

function blmid_cfm(
    sec1::BLSecondaryState,
    sec2::BLSecondaryState,
    ityp::Union{Symbol,Integer};
    model::BLModelParams = BLModelParams(),
)
    mode = _ityp_symbol(ityp)
    if mode == :wake
        return _c(sec1.hk, 0)
    end

    hka = _c(sec1.hk, 0.5) * (sec1.hk + sec2.hk)
    rta = _c(sec1.hk, 0.5) * (sec1.rt + sec2.rt)
    ma = _c(sec1.hk, 0.5) * (sec1.m + sec2.m)

    if mode == :laminar || mode == :similarity
        return cfl(hka, rta, ma)
    end

    cftv = cft(hka, rta, ma; cffac = model.cffac, gamma = model.gamma)
    cflv = cfl(hka, rta, ma)
    return max(cftv, cflv)
end

function _upwinding(
    sec1::BLSecondaryState,
    sec2::BLSecondaryState,
    mode::Symbol;
    hupwt::Real = 1.0f0,
)
    hk1 = max(sec1.hk, _c(sec1.hk, 1.0001))
    hk2 = max(sec2.hk, _c(sec2.hk, 1.0001))

    hdcon = _c(hk2, 5) * float(hupwt) / hk2^2
    if mode == :wake
        hdcon = float(hupwt) / hk2^2
    end

    arg = abs((hk2 - _c(hk2, 1)) / max(hk1 - _c(hk1, 1), _c(hk1, 1e-8)))
    hl = log(max(arg, _c(hk2, 1e-12)))
    hlsq = min(hl^2, _c(hk2, 15))
    ehh = exp(-hlsq * hdcon)
    return _c(hk2, 1) - _c(hk2, 0.5) * ehh
end

function bldif_residuals(
    ityp::Union{Symbol,Integer},
    p1::BLPrimaryState,
    p2::BLPrimaryState,
    s1::BLSecondaryState,
    s2::BLSecondaryState;
    model::BLModelParams = BLModelParams(),
    acrit::Real = 9.0f0,
    idampv::Integer = 0,
    bule::Real = 1.0f0,
    hupwt::Real = 1.0f0,
)
    mode = _ityp_symbol(ityp)
    T = typeof(float(p1.x))
    eps = T(1e-12)

    xlog, ulog, tlog, hlog =
        if mode == :similarity
            (one(T), T(bule), T(0.5) * (one(T) - T(bule)), zero(T))
        else
            (
                log(max(p2.x, eps) / max(p1.x, eps)),
                log(max(p2.u, eps) / max(p1.u, eps)),
                log(max(p2.t, eps) / max(p1.t, eps)),
                log(max(s2.hs, eps) / max(s1.hs, eps)),
            )
        end

    upw = _upwinding(s1, s2, mode; hupwt = float(hupwt))

    rez1 =
        if mode == :similarity
            -p2.ampl
        elseif mode == :laminar
            ax = axset(
                s1.hk,
                p1.t,
                s1.rt,
                p1.ampl,
                s2.hk,
                p2.t,
                s2.rt,
                p2.ampl,
                T(acrit);
                idampv = idampv,
            )
            p2.ampl - p1.ampl - ax * (p2.x - p1.x)
        else
            sa = (one(T) - upw) * p1.s + upw * p2.s
            cqa = (one(T) - upw) * s1.cq + upw * s2.cq
            cfa = (one(T) - upw) * s1.cf + upw * s2.cf
            hka = (one(T) - upw) * s1.hk + upw * s2.hk
            usa = T(0.5) * (s1.us + s2.us)
            rta = T(0.5) * (s1.rt + s2.rt)
            dea = T(0.5) * (s1.de + s2.de)
            da = max(T(0.5) * ((p1.d - p1.dw) + (p2.d - p2.dw)), eps)
            ald = mode == :wake ? model.dlcon : one(T)

            hkc =
                if mode == :turbulent
                    max(hka - one(T) - model.gccon / max(rta, eps), T(0.01))
                else
                    hka - one(T)
                end

            hr = hkc / max(model.gacon * ald * hka, eps)
            uq = (T(0.5) * cfa - hr^2) / max(model.gbcon * da, eps)

            scc = model.sccon * T(1.333) / max(one(T) + usa, eps)
            slog = log(max(p2.s, eps) / max(p1.s, eps))
            dxi = p2.x - p1.x

            scc * (cqa - sa * ald) * dxi - dea * T(2) * slog +
            dea * T(2) * (uq * dxi - ulog) * model.duxcon
        end

    cfm = blmid_cfm(s1, s2, mode; model = model)
    ha = T(0.5) * (s1.h + s2.h)
    ma = T(0.5) * (s1.m + s2.m)
    xa = T(0.5) * (p1.x + p2.x)
    ta = max(T(0.5) * (p1.t + p2.t), eps)
    hwa = T(0.5) * (p1.dw / max(p1.t, eps) + p2.dw / max(p2.t, eps))

    cfx_m =
        T(0.5) * cfm * xa / ta +
        T(0.25) * (s1.cf * p1.x / max(p1.t, eps) + s2.cf * p2.x / max(p2.t, eps))
    btmp_m = ha + T(2) - ma + hwa
    rez2 = tlog + btmp_m * ulog - xlog * T(0.5) * cfx_m

    xot1 = p1.x / max(p1.t, eps)
    xot2 = p2.x / max(p2.t, eps)
    hsa = max(T(0.5) * (s1.hs + s2.hs), eps)
    hca = T(0.5) * (s1.hc + s2.hc)

    dix = (one(T) - upw) * s1.di * xot1 + upw * s2.di * xot2
    cfx_h = (one(T) - upw) * s1.cf * xot1 + upw * s2.cf * xot2
    btmp_h = T(2) * hca / hsa + one(T) - ha - hwa
    rez3 = hlog + btmp_h * ulog + xlog * (T(0.5) * cfx_h - dix)

    return (rez1, rez2, rez3)
end
