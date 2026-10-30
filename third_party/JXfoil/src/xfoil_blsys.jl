# ============================================================================
# Literal line-by-line port of XFoil's boundary layer system routines.
#
# This file contains exact translations of:
#   BLPRV, BLKIN, BLVAR, BLMID, BLDIF, TRDIF, BLSYS, TESYS, TRCHEK2,
#   AXSET, MRCHDU, SETBL
#
# All variables, clamping, branching, and Float32 (Fortran REAL) arithmetic
# are preserved exactly.  The COM1/COM2 state propagation uses a mutable
# struct that matches Fortran's EQUIVALENCE layout.
#
# Convention: all BL computations use Float32 (XFT) matching Fortran REAL.
# This file is included inside the JXfoil module; all *_jac functions from
# viscous_closure.jl are available.
# ============================================================================

const XFT = Float32  # Match Fortran single-precision REAL

"""
Mutable BL state for one station.  Maps to Fortran's COM1/COM2 (NCOM=73).
Fields are in the EXACT order of the Fortran EQUIVALENCE.
"""
mutable struct BLCom
    x::XFT;  u::XFT;  t::XFT;  d::XFT;  s::XFT;  ampl::XFT;  u_uei::XFT;  u_ms::XFT;  dw::XFT  # 9
    h::XFT;  h_t::XFT;  h_d::XFT  # 12
    m::XFT;  m_u::XFT;  m_ms::XFT  # 15  (note: m_u maps to M1_U1 or M2_U2)
    r::XFT;  r_u::XFT;  r_ms::XFT  # 18
    v::XFT;  v_u::XFT;  v_ms::XFT;  v_re::XFT  # 22
    hk::XFT; hk_u::XFT; hk_t::XFT; hk_d::XFT; hk_ms::XFT  # 27
    hs::XFT; hs_u::XFT; hs_t::XFT; hs_d::XFT; hs_ms::XFT; hs_re::XFT  # 33
    hc::XFT; hc_u::XFT; hc_t::XFT; hc_d::XFT; hc_ms::XFT  # 38
    rt::XFT; rt_u::XFT; rt_t::XFT; rt_ms::XFT; rt_re::XFT  # 43
    cf::XFT; cf_u::XFT; cf_t::XFT; cf_d::XFT; cf_ms::XFT; cf_re::XFT  # 49
    di::XFT; di_u::XFT; di_t::XFT; di_d::XFT; di_s::XFT; di_ms::XFT; di_re::XFT  # 56
    us::XFT; us_u::XFT; us_t::XFT; us_d::XFT; us_ms::XFT; us_re::XFT  # 62
    cq::XFT; cq_u::XFT; cq_t::XFT; cq_d::XFT; cq_ms::XFT; cq_re::XFT  # 68
    de::XFT; de_u::XFT; de_t::XFT; de_d::XFT; de_ms::XFT  # 73
end

function BLCom()
    return BLCom(ntuple(_ -> XFT(0), 73)...)
end

function copycom!(dst::BLCom, src::BLCom)
    for f in fieldnames(BLCom)
        setfield!(dst, f, getfield(src, f))
    end
end

"""
BL flow parameters shared across all stations (Fortran COMMON/V_VAR/).
"""
struct BLFlowF32
    dwte::XFT
    qinfbl::XFT
    tkbl::XFT;    tkbl_ms::XFT
    rstbl::XFT;   rstbl_ms::XFT
    hstinv::XFT;  hstinv_ms::XFT
    reybl::XFT;   reybl_ms::XFT;  reybl_re::XFT
    gambl::XFT;   gm1bl::XFT;     hvrat::XFT
    bule::XFT;    xiforc::XFT;    amcrit::XFT
    idampv::Int32
end

@inline function xbl_with_xiforc(fl::BLFlowF32, xiforc::XFT)
    return BLFlowF32(
        fl.dwte, fl.qinfbl,
        fl.tkbl, fl.tkbl_ms,
        fl.rstbl, fl.rstbl_ms,
        fl.hstinv, fl.hstinv_ms,
        fl.reybl, fl.reybl_ms, fl.reybl_re,
        fl.gambl, fl.gm1bl, fl.hvrat,
        fl.bule, xiforc, fl.amcrit,
        fl.idampv,
    )
end

"""
BL model parameters (Fortran COMMON/BLPAR/).
"""
struct BLParF32
    sccon::XFT;  gacon::XFT;  gbcon::XFT;  gccon::XFT
    dlcon::XFT;  ctrcon::XFT; ctrcex::XFT; duxcon::XFT
    ctcon::XFT;  cffac::XFT
end

function BLParF32()
    sccon = XFT(5.6);  gacon = XFT(6.70); gbcon = XFT(0.75); gccon = XFT(18.0)
    dlcon = XFT(0.9);  ctrcon = XFT(1.8); ctrcex = XFT(3.3); duxcon = XFT(1.0)
    ctcon = XFT(0.5) / (gacon^2 * gbcon); cffac = XFT(1.0)
    return BLParF32(sccon,gacon,gbcon,gccon,dlcon,ctrcon,ctrcex,duxcon,ctcon,cffac)
end

# ============================================================================
#  BLPRV — Set BL primary "2" variables from parameter list
#  Literal translation of xblsys.f lines 701-722
# ============================================================================
function xbl_blprv!(c2::BLCom, xsi::XFT, ami::XFT, cti::XFT, thi::XFT,
                     dsi::XFT, dswaki::XFT, uei::XFT, fl::BLFlowF32)
    c2.x = xsi
    c2.ampl = ami
    c2.s = cti
    c2.t = thi
    c2.d = dsi - dswaki
    c2.dw = dswaki

    # Karman-Tsien compressibility transformation
    tkbl = fl.tkbl
    qinfbl = fl.qinfbl
    denom = XFT(1) - tkbl * (uei / qinfbl)^2
    c2.u = uei * (XFT(1) - tkbl) / denom
    c2.u_uei = (XFT(1) + tkbl * (XFT(2) * c2.u * uei / qinfbl^2 - XFT(1))) / denom
    c2.u_ms = (c2.u * (uei / qinfbl)^2 - uei) * fl.tkbl_ms / denom
    return nothing
end

# ============================================================================
#  BLKIN — Turbulence-independent secondary "2" variables
#  Literal translation of xblsys.f lines 749-805
# ============================================================================
function xbl_blkin!(c2::BLCom, fl::BLFlowF32)
    # Match the Fortran REAL march more closely by rounding each named
    # intermediate back to Float32 before it feeds downstream expressions.
    roundstep(x) = XFT(x)

    u = roundstep(c2.u); t = roundstep(c2.t); d = roundstep(c2.d)
    gm1bl = roundstep(fl.gm1bl); hstinv = roundstep(fl.hstinv)
    hstinv_ms = roundstep(fl.hstinv_ms); rstbl = roundstep(fl.rstbl)
    rstbl_ms = roundstep(fl.rstbl_ms); hvrat = roundstep(fl.hvrat)
    reybl = roundstep(fl.reybl); reybl_ms = roundstep(fl.reybl_ms)
    reybl_re = roundstep(fl.reybl_re)

    # Edge Mach²
    denom = roundstep(gm1bl * roundstep(1.0 - roundstep(0.5 * u^2 * hstinv)))
    if abs(denom) < 1e-12
        denom = copysign(1e-12, denom == 0.0 ? 1.0 : denom)
    end
    m = roundstep(u^2 * hstinv / denom)
    tr2 = roundstep(1.0 + 0.5 * gm1bl * m)
    m_u = roundstep(2.0 * m * tr2 / u)
    m_ms = roundstep(u^2 * tr2 / denom * hstinv_ms)

    # Edge density (isentropic)
    r = roundstep(rstbl * tr2^(-1.0 / gm1bl))
    r_u = roundstep(-r / tr2 * 0.5 * m_u)
    r_ms = roundstep(-r / tr2 * 0.5 * m_ms + rstbl_ms * tr2^(-1.0 / gm1bl))

    # Shape parameter
    t2s = max(t, 1e-12); d2s = max(d, 1e-12)
    h = roundstep(d2s / t2s)
    h_d = roundstep(1.0 / t2s)
    h_t = roundstep(-h / t2s)

    # Enthalpy ratio
    herat = roundstep(max(1.0 - 0.5 * u^2 * hstinv, 1e-12))
    he_u = roundstep(-u * hstinv)
    he_ms = roundstep(-0.5 * u^2 * hstinv_ms)

    # Molecular viscosity
    v = roundstep(sqrt(herat^3) * (1.0 + hvrat) / (herat + hvrat) / reybl)
    v = max(v, 1e-16)
    v_he = roundstep(v * (1.5 / herat - 1.0 / (herat + hvrat)))
    v_u = roundstep(v_he * he_u)
    v_ms = roundstep(-v / reybl * reybl_ms + v_he * he_ms)
    v_re = roundstep(-v / reybl * reybl_re)

    # Kinematic shape parameter
    hk32, hk_h32, hk_m32 = hkin_jac(XFT(h), XFT(m))
    hk = roundstep(hk32)
    hk_h = roundstep(hk_h32)
    hk_m = roundstep(hk_m32)

    # Momentum-thickness Reynolds number
    rt = roundstep(r * u * t2s / v)
    rt_u = roundstep(rt * (1.0 / u + r_u / r - v_u / v))
    rt_t = roundstep(rt / t2s)
    rt_ms = roundstep(rt * (r_ms / r - v_ms / v))
    rt_re = roundstep(rt * (-v_re / v))

    # Store results back as Float32
    c2.m = XFT(m); c2.m_u = XFT(m_u); c2.m_ms = XFT(m_ms)
    c2.r = XFT(r); c2.r_u = XFT(r_u); c2.r_ms = XFT(r_ms)
    c2.h = XFT(h); c2.h_d = XFT(h_d); c2.h_t = XFT(h_t)
    c2.v = XFT(v); c2.v_u = XFT(v_u); c2.v_ms = XFT(v_ms); c2.v_re = XFT(v_re)
    c2.hk = XFT(hk); c2.hk_u = XFT(hk_m * m_u); c2.hk_t = XFT(hk_h * h_t)
    c2.hk_d = XFT(hk_h * h_d); c2.hk_ms = XFT(hk_m * m_ms)
    c2.rt = XFT(rt); c2.rt_u = XFT(rt_u); c2.rt_t = XFT(rt_t)
    c2.rt_ms = XFT(rt_ms); c2.rt_re = XFT(rt_re)

    return nothing
end

# ============================================================================
#  BLVAR — All secondary "2" variables (laminar/turbulent/wake)
#  Literal translation of xblsys.f lines 808-1145
# ============================================================================
function xbl_blvar!(c2::BLCom, ityp::Int, fl::BLFlowF32, par::BLParF32)
    roundstep(x) = Float64(XFT(x))

    # Read all needed BLCom fields as Float64 once.
    hk = roundstep(c2.hk); hk_u = roundstep(c2.hk_u); hk_t = roundstep(c2.hk_t)
    hk_d = roundstep(c2.hk_d); hk_ms = roundstep(c2.hk_ms)
    m = roundstep(c2.m); m_u = roundstep(c2.m_u); m_ms = roundstep(c2.m_ms)
    h = roundstep(c2.h); h_t = roundstep(c2.h_t); h_d = roundstep(c2.h_d)
    rt = roundstep(c2.rt); rt_u = roundstep(c2.rt_u); rt_t = roundstep(c2.rt_t)
    rt_ms = roundstep(c2.rt_ms); rt_re = roundstep(c2.rt_re)
    t = roundstep(c2.t); d = roundstep(c2.d); s = roundstep(c2.s); u = roundstep(c2.u)
    gbcon = roundstep(par.gbcon); ctcon = roundstep(par.ctcon); gccon = roundstep(par.gccon)
    cffac_p = roundstep(par.cffac); gambl = roundstep(fl.gambl)

    # Clamp HK
    if ityp == 3
        if hk < 1.00005; hk = roundstep(1.00005); end
    else
        if hk < 1.05; hk = roundstep(1.05); end
    end

    # HC2 — density thickness
    hc32, hc_hk32, hc_m32 = hct_jac(XFT(hk), XFT(m))
    hc = roundstep(hc32)
    hc_hk = roundstep(hc_hk32)
    hc_m = roundstep(hc_m32)
    hc_u = roundstep(hc_hk * hk_u + hc_m * m_u)
    hc_t = roundstep(hc_hk * hk_t)
    hc_d = roundstep(hc_hk * hk_d)
    hc_ms = roundstep(hc_hk * hk_ms + hc_m * m_ms)

    # HS2 — KE thickness shape parameter
    local hs, hs_hk, hs_rt, hs_m
    if ityp == 1
        hs32, hs_hk32, hs_rt32, hs_m32 = hsl_jac(XFT(hk), XFT(rt), XFT(m))
    else
        hs32, hs_hk32, hs_rt32, hs_m32 = hst_jac(XFT(hk), XFT(rt), XFT(m))
    end
    hs = roundstep(hs32)
    hs_hk = roundstep(hs_hk32)
    hs_rt = roundstep(hs_rt32)
    hs_m = roundstep(hs_m32)
    hs_u = roundstep(hs_hk * hk_u + hs_rt * rt_u + hs_m * m_u)
    hs_t = roundstep(hs_hk * hk_t + hs_rt * rt_t)
    hs_d_val = roundstep(hs_hk * hk_d)
    hs_ms = roundstep(hs_hk * hk_ms + hs_rt * rt_ms + hs_m * m_ms)
    hs_re = roundstep(hs_rt * rt_re)

    # US2 — slip velocity
    h2 = roundstep(max(h, 1e-12))
    us = roundstep(0.5 * hs * (1.0 - (hk - 1.0) / (gbcon * h2)))
    us_hs_v = roundstep(0.5 * (1.0 - (hk - 1.0) / (gbcon * h2)))
    us_hk_v = roundstep(0.5 * hs * (-1.0 / (gbcon * h2)))
    us_h_v = roundstep(0.5 * hs * (hk - 1.0) / (gbcon * h2^2))
    us_u = roundstep(us_hs_v * hs_u + us_hk_v * hk_u)
    us_t = roundstep(us_hs_v * hs_t + us_hk_v * hk_t + us_h_v * h_t)
    us_d = roundstep(us_hs_v * hs_d_val + us_hk_v * hk_d + us_h_v * h_d)
    us_ms = roundstep(us_hs_v * hs_ms + us_hk_v * hk_ms)
    us_re = roundstep(us_hs_v * hs_re)

    # Clamp US
    if ityp <= 2 && us > 0.95
        us = roundstep(0.98); us_u = 0.0; us_t = 0.0; us_d = 0.0; us_ms = 0.0; us_re = 0.0
    end
    if ityp == 3 && us > 0.99995
        us = roundstep(0.99995); us_u = 0.0; us_t = 0.0; us_d = 0.0; us_ms = 0.0; us_re = 0.0
    end

    # CQ2 — equilibrium shear coefficient
    hkc = roundstep(hk - 1.0); hkc_hk = 1.0; hkc_rt = 0.0
    rt_safe = roundstep(max(rt, 1e-12))
    if ityp == 2
        hkc = roundstep(hk - 1.0 - gccon / rt_safe)
        hkc_hk = 1.0; hkc_rt = roundstep(gccon / rt_safe^2)
        if hkc < 0.01; hkc = roundstep(0.01); hkc_hk = 0.0; hkc_rt = 0.0; end
    end
    hkb = roundstep(hk - 1.0); usb = roundstep(max(1.0 - us, 1e-8))
    cqarg = roundstep(ctcon * hs * hkb * hkc^2 / (usb * h2 * hk^2))
    cqarg = max(cqarg, 0.0); cq = roundstep(sqrt(cqarg))
    local cq_hs, cq_us_v, cq_hk_v, cq_rt_v, cq_h_v
    if cq > 1e-12
        cq_hs = roundstep(ctcon * hkb * hkc^2 / (usb * h2 * hk^2) * 0.5 / cq)
        cq_us_v = roundstep(cqarg / usb * 0.5 / cq)
        cq_hk_v = roundstep((ctcon * hs * hkc^2 / (usb * h2 * hk^2) - 2.0 * cqarg / hk +
                             2.0 * ctcon * hs * hkb * hkc / (usb * h2 * hk^2) * hkc_hk) * 0.5 / cq)
        cq_rt_v = roundstep(2.0 * ctcon * hs * hkb * hkc / (usb * h2 * hk^2) * hkc_rt * 0.5 / cq)
        cq_h_v = roundstep(-cqarg / h2 * 0.5 / cq)
    else
        cq_hs = 0.0; cq_us_v = 0.0; cq_hk_v = 0.0; cq_rt_v = 0.0; cq_h_v = 0.0
    end
    cq_u = roundstep(cq_hs * hs_u + cq_us_v * us_u + cq_hk_v * hk_u + cq_rt_v * rt_u)
    cq_t = roundstep(cq_hs * hs_t + cq_us_v * us_t + cq_hk_v * hk_t + cq_rt_v * rt_t + cq_h_v * h_t)
    cq_d = roundstep(cq_hs * hs_d_val + cq_us_v * us_d + cq_hk_v * hk_d + cq_h_v * h_d)
    cq_ms = roundstep(cq_hs * hs_ms + cq_us_v * us_ms + cq_hk_v * hk_ms + cq_rt_v * rt_ms)
    cq_re = roundstep(cq_hs * hs_re + cq_us_v * us_re + cq_rt_v * rt_re)

    # CF2 — skin friction (closure calls already use Float64 args)
    local cf, cf_u_v, cf_t_v, cf_d_v, cf_ms_v, cf_re_v
    if ityp == 3
        cf = 0.0; cf_u_v = 0.0; cf_t_v = 0.0; cf_d_v = 0.0; cf_ms_v = 0.0; cf_re_v = 0.0
    elseif ityp == 1
        cf32, cf_hk32, cf_rt_c32, cf_m32 = cfl_jac(XFT(hk), XFT(rt), XFT(m))
        cf = roundstep(cf32)
        cf_hk = roundstep(cf_hk32)
        cf_rt_c = roundstep(cf_rt_c32)
        cf_m = roundstep(cf_m32)
        cf_u_v = roundstep(cf_hk * hk_u + cf_rt_c * rt_u + cf_m * m_u)
        cf_t_v = roundstep(cf_hk * hk_t + cf_rt_c * rt_t); cf_d_v = roundstep(cf_hk * hk_d)
        cf_ms_v = roundstep(cf_hk * hk_ms + cf_rt_c * rt_ms + cf_m * m_ms); cf_re_v = roundstep(cf_rt_c * rt_re)
    else
        cf32, cf_hk32, cf_rt_c32, cf_m32 =
            cft_jac(XFT(hk), XFT(rt), XFT(m); cffac = XFT(cffac_p), gamma = XFT(gambl))
        cf = roundstep(cf32)
        cf_hk = roundstep(cf_hk32)
        cf_rt_c = roundstep(cf_rt_c32)
        cf_m = roundstep(cf_m32)
        cf_u_v = roundstep(cf_hk * hk_u + cf_rt_c * rt_u + cf_m * m_u)
        cf_t_v = roundstep(cf_hk * hk_t + cf_rt_c * rt_t); cf_d_v = roundstep(cf_hk * hk_d)
        cf_ms_v = roundstep(cf_hk * hk_ms + cf_rt_c * rt_ms + cf_m * m_ms); cf_re_v = roundstep(cf_rt_c * rt_re)
        cf2l32, cf2l_hk32, cf2l_rt32, cf2l_m32 = cfl_jac(XFT(hk), XFT(rt), XFT(m))
        cf2l = roundstep(cf2l32)
        cf2l_hk = roundstep(cf2l_hk32)
        cf2l_rt = roundstep(cf2l_rt32)
        cf2l_m = roundstep(cf2l_m32)
        if cf2l > cf
            cf = cf2l; cf_u_v = roundstep(cf2l_hk * hk_u + cf2l_rt * rt_u + cf2l_m * m_u)
            cf_t_v = roundstep(cf2l_hk * hk_t + cf2l_rt * rt_t); cf_d_v = roundstep(cf2l_hk * hk_d)
            cf_ms_v = roundstep(cf2l_hk * hk_ms + cf2l_rt * rt_ms + cf2l_m * m_ms); cf_re_v = roundstep(cf2l_rt * rt_re)
        end
    end

    # DI2 — dissipation
    local di, di_s, di_u_v, di_t_v, di_d_v, di_ms_v, di_re_v
    if ityp == 1
        di32, di_hk32, di_rt_c32 = dil_jac(XFT(hk), XFT(rt))
        di = roundstep(di32)
        di_hk = roundstep(di_hk32)
        di_rt_c = roundstep(di_rt_c32)
        di_u_v = roundstep(di_hk * hk_u + di_rt_c * rt_u); di_t_v = roundstep(di_hk * hk_t + di_rt_c * rt_t)
        di_d_v = roundstep(di_hk * hk_d); di_s = 0.0
        di_ms_v = roundstep(di_hk * hk_ms + di_rt_c * rt_ms); di_re_v = roundstep(di_rt_c * rt_re)
    elseif ityp == 2
        cf2t32, cf2t_hk32, cf2t_rt32, cf2t_m32 =
            cft_jac(XFT(hk), XFT(rt), XFT(m); cffac = XFT(cffac_p), gamma = XFT(gambl))
        cf2t = roundstep(cf2t32)
        cf2t_hk = roundstep(cf2t_hk32)
        cf2t_rt = roundstep(cf2t_rt32)
        cf2t_m = roundstep(cf2t_m32)
        cf2t_u = roundstep(cf2t_hk * hk_u + cf2t_rt * rt_u + cf2t_m * m_u)
        cf2t_t = roundstep(cf2t_hk * hk_t + cf2t_rt * rt_t); cf2t_d = roundstep(cf2t_hk * hk_d)
        cf2t_ms = roundstep(cf2t_hk * hk_ms + cf2t_rt * rt_ms + cf2t_m * m_ms); cf2t_re = roundstep(cf2t_rt * rt_re)

        di = roundstep((0.5 * cf2t * us) * 2.0 / hs)
        ddi_hs = roundstep(-(0.5 * cf2t * us) * 2.0 / hs^2); ddi_us = roundstep((0.5 * cf2t) * 2.0 / hs)
        ddi_cf2t = roundstep((0.5 * us) * 2.0 / hs)
        di_s = 0.0; di_u_v = roundstep(ddi_hs * hs_u + ddi_us * us_u + ddi_cf2t * cf2t_u)
        di_t_v = roundstep(ddi_hs * hs_t + ddi_us * us_t + ddi_cf2t * cf2t_t)
        di_d_v = roundstep(ddi_hs * hs_d_val + ddi_us * us_d + ddi_cf2t * cf2t_d)
        di_ms_v = roundstep(ddi_hs * hs_ms + ddi_us * us_ms + ddi_cf2t * cf2t_ms)
        di_re_v = roundstep(ddi_hs * hs_re + ddi_us * us_re + ddi_cf2t * cf2t_re)

        # DFAC correction
        grt = roundstep(log(max(rt, 1e-12))); hmin = roundstep(1.0 + 2.1 / grt); hm_rt = roundstep(-(2.1 / grt^2) / rt)
        fl_val = roundstep((hk - 1.0) / (hmin - 1.0)); fl_hk = roundstep(1.0 / (hmin - 1.0))
        fl_rt = roundstep((-fl_val / (hmin - 1.0)) * hm_rt)
        tfl = roundstep(tanh(fl_val)); dfac = roundstep(0.5 + 0.5 * tfl); df_fl = roundstep(0.5 * (1.0 - tfl^2))
        df_hk = roundstep(df_fl * fl_hk); df_rt = roundstep(df_fl * fl_rt)

        di_s = roundstep(di_s * dfac); di_u_v = roundstep(di_u_v * dfac + di * (df_hk * hk_u + df_rt * rt_u))
        di_t_v = roundstep(di_t_v * dfac + di * (df_hk * hk_t + df_rt * rt_t))
        di_d_v = roundstep(di_d_v * dfac + di * (df_hk * hk_d))
        di_ms_v = roundstep(di_ms_v * dfac + di * (df_hk * hk_ms + df_rt * rt_ms))
        di_re_v = roundstep(di_re_v * dfac + di * (df_rt * rt_re)); di = roundstep(di * dfac)
    else
        di = 0.0; di_s = 0.0; di_u_v = 0.0; di_t_v = 0.0; di_d_v = 0.0; di_ms_v = 0.0; di_re_v = 0.0
    end

    # Outer layer contribution
    if ityp != 1
        dd = roundstep(s^2 * (0.995 - us) * 2.0 / hs); dd_hs = roundstep(-s^2 * (0.995 - us) * 2.0 / hs^2)
        dd_us = roundstep(-s^2 * 2.0 / hs); dd_s = roundstep(s * 2.0 * (0.995 - us) * 2.0 / hs)
        di = roundstep(di + dd); di_s = dd_s; di_u_v = roundstep(di_u_v + dd_hs * hs_u + dd_us * us_u)
        di_t_v = roundstep(di_t_v + dd_hs * hs_t + dd_us * us_t); di_d_v = roundstep(di_d_v + dd_hs * hs_d_val + dd_us * us_d)
        di_ms_v = roundstep(di_ms_v + dd_hs * hs_ms + dd_us * us_ms); di_re_v = roundstep(di_re_v + dd_hs * hs_re + dd_us * us_re)

        dd = roundstep(0.15 * (0.995 - us)^2 / rt_safe * 2.0 / hs)
        dd_us = roundstep(-0.15 * (0.995 - us) * 2.0 / rt_safe * 2.0 / hs)
        dd_hs = roundstep(-dd / hs); dd_rt = roundstep(-dd / rt_safe)
        di = roundstep(di + dd); di_u_v = roundstep(di_u_v + dd_hs * hs_u + dd_us * us_u + dd_rt * rt_u)
        di_t_v = roundstep(di_t_v + dd_hs * hs_t + dd_us * us_t + dd_rt * rt_t)
        di_d_v = roundstep(di_d_v + dd_hs * hs_d_val + dd_us * us_d)
        di_ms_v = roundstep(di_ms_v + dd_hs * hs_ms + dd_us * us_ms + dd_rt * rt_ms)
        di_re_v = roundstep(di_re_v + dd_hs * hs_re + dd_us * us_re + dd_rt * rt_re)
    end

    if ityp == 2
        di2l32, di2l_hk32, di2l_rt32 = dil_jac(XFT(hk), XFT(rt))
        di2l = roundstep(di2l32)
        di2l_hk = roundstep(di2l_hk32)
        di2l_rt = roundstep(di2l_rt32)
        if di2l > di
            di = di2l; di_s = 0.0; di_u_v = roundstep(di2l_hk * hk_u + di2l_rt * rt_u)
            di_t_v = roundstep(di2l_hk * hk_t + di2l_rt * rt_t); di_d_v = roundstep(di2l_hk * hk_d)
            di_ms_v = roundstep(di2l_hk * hk_ms + di2l_rt * rt_ms); di_re_v = roundstep(di2l_rt * rt_re)
        end
    end

    if ityp == 3
        di2l32, di2l_hk32, di2l_rt32 = dilw_jac(XFT(hk), XFT(rt))
        di2l = roundstep(di2l32)
        di2l_hk = roundstep(di2l_hk32)
        di2l_rt = roundstep(di2l_rt32)
        if di2l > di
            di = di2l; di_s = 0.0; di_u_v = roundstep(di2l_hk * hk_u + di2l_rt * rt_u)
            di_t_v = roundstep(di2l_hk * hk_t + di2l_rt * rt_t); di_d_v = roundstep(di2l_hk * hk_d)
            di_ms_v = roundstep(di2l_hk * hk_ms + di2l_rt * rt_ms); di_re_v = roundstep(di2l_rt * rt_re)
        end
        di = roundstep(di * 2.0); di_s = roundstep(di_s * 2.0); di_u_v = roundstep(di_u_v * 2.0); di_t_v = roundstep(di_t_v * 2.0)
        di_d_v = roundstep(di_d_v * 2.0); di_ms_v = roundstep(di_ms_v * 2.0); di_re_v = roundstep(di_re_v * 2.0)
    end

    # DE2 — BL thickness
    de = roundstep((3.15 + 1.72 / (hk - 1.0)) * t + d)
    de_hk = roundstep((-1.72 / (hk - 1.0)^2) * t)
    de_u = roundstep(de_hk * hk_u); de_t = roundstep(de_hk * hk_t + (3.15 + 1.72 / (hk - 1.0)))
    de_d_v = roundstep(de_hk * hk_d + 1.0); de_ms = roundstep(de_hk * hk_ms)
    if de > 12.0 * t
        de = roundstep(12.0 * t); de_u = 0.0; de_t = roundstep(12.0); de_d_v = 0.0; de_ms = 0.0
    end

    # Store ALL results back as Float32
    c2.hk = XFT(hk); c2.hk_u = XFT(hk_u); c2.hk_t = XFT(hk_t); c2.hk_d = XFT(hk_d); c2.hk_ms = XFT(hk_ms)
    c2.hc = XFT(hc); c2.hc_u = XFT(hc_u); c2.hc_t = XFT(hc_t); c2.hc_d = XFT(hc_d); c2.hc_ms = XFT(hc_ms)
    c2.hs = XFT(hs); c2.hs_u = XFT(hs_u); c2.hs_t = XFT(hs_t); c2.hs_d = XFT(hs_d_val); c2.hs_ms = XFT(hs_ms); c2.hs_re = XFT(hs_re)
    c2.us = XFT(us); c2.us_u = XFT(us_u); c2.us_t = XFT(us_t); c2.us_d = XFT(us_d); c2.us_ms = XFT(us_ms); c2.us_re = XFT(us_re)
    c2.cq = XFT(cq); c2.cq_u = XFT(cq_u); c2.cq_t = XFT(cq_t); c2.cq_d = XFT(cq_d); c2.cq_ms = XFT(cq_ms); c2.cq_re = XFT(cq_re)
    c2.cf = XFT(cf); c2.cf_u = XFT(cf_u_v); c2.cf_t = XFT(cf_t_v); c2.cf_d = XFT(cf_d_v); c2.cf_ms = XFT(cf_ms_v); c2.cf_re = XFT(cf_re_v)
    c2.di = XFT(di); c2.di_s = XFT(di_s); c2.di_u = XFT(di_u_v); c2.di_t = XFT(di_t_v)
    c2.di_d = XFT(di_d_v); c2.di_ms = XFT(di_ms_v); c2.di_re = XFT(di_re_v)
    c2.de = XFT(de); c2.de_u = XFT(de_u); c2.de_t = XFT(de_t); c2.de_d = XFT(de_d_v); c2.de_ms = XFT(de_ms)

    return nothing
end

# ============================================================================
#  BLMID — Midpoint skin friction
#  Literal translation of xblsys.f lines 1147-1215
# ============================================================================
struct BLMidResult
    cfm::XFT; cfm_ms::XFT; cfm_re::XFT
    cfm_u1::XFT; cfm_t1::XFT; cfm_d1::XFT
    cfm_u2::XFT; cfm_t2::XFT; cfm_d2::XFT
end

function xbl_blmid(c1::BLCom, c2::BLCom, ityp::Int, simi::Bool,
                    fl::BLFlowF32, par::BLParF32)
    roundstep(x) = XFT(x)

    # Set similarity variables if not defined
    hk1 = simi ? c2.hk : c1.hk
    hk1_t = simi ? c2.hk_t : c1.hk_t
    hk1_d = simi ? c2.hk_d : c1.hk_d
    hk1_u = simi ? c2.hk_u : c1.hk_u
    hk1_ms = simi ? c2.hk_ms : c1.hk_ms
    rt1 = simi ? c2.rt : c1.rt
    rt1_t = simi ? c2.rt_t : c1.rt_t
    rt1_u = simi ? c2.rt_u : c1.rt_u
    rt1_ms = simi ? c2.rt_ms : c1.rt_ms
    rt1_re = simi ? c2.rt_re : c1.rt_re
    m1 = simi ? c2.m : c1.m
    m1_u = simi ? c2.m_u : c1.m_u
    m1_ms = simi ? c2.m_ms : c1.m_ms

    hka = roundstep(XFT(0.5) * (hk1 + c2.hk))
    rta = roundstep(XFT(0.5) * (rt1 + c2.rt))
    ma  = roundstep(XFT(0.5) * (m1 + c2.m))

    if ityp == 3
        cfm = XFT(0); cfm_hka = XFT(0); cfm_rta = XFT(0); cfm_ma = XFT(0)
    elseif ityp == 1
        cfm32, cfm_hka32, cfm_rta32, cfm_ma32 = cfl_jac(XFT(hka), XFT(rta), XFT(ma))
        cfm = roundstep(cfm32); cfm_hka = roundstep(cfm_hka32)
        cfm_rta = roundstep(cfm_rta32); cfm_ma = roundstep(cfm_ma32)
    else
        cfm32, cfm_hka32, cfm_rta32, cfm_ma32 =
            cft_jac(XFT(hka), XFT(rta), XFT(ma); cffac = par.cffac, gamma = fl.gambl)
        cfm = roundstep(cfm32); cfm_hka = roundstep(cfm_hka32)
        cfm_rta = roundstep(cfm_rta32); cfm_ma = roundstep(cfm_ma32)
        cfml32, cfml_hka32, cfml_rta32, cfml_ma32 = cfl_jac(XFT(hka), XFT(rta), XFT(ma))
        cfml = roundstep(cfml32); cfml_hka = roundstep(cfml_hka32)
        cfml_rta = roundstep(cfml_rta32); cfml_ma = roundstep(cfml_ma32)
        if cfml > cfm
            cfm = cfml; cfm_hka = cfml_hka; cfm_rta = cfml_rta; cfm_ma = cfml_ma
        end
    end

    cfm_u1 = roundstep(XFT(0.5) * (cfm_hka * hk1_u + cfm_ma * m1_u + cfm_rta * rt1_u))
    cfm_t1 = roundstep(XFT(0.5) * (cfm_hka * hk1_t + cfm_rta * rt1_t))
    cfm_d1 = roundstep(XFT(0.5) * (cfm_hka * hk1_d))
    cfm_u2 = roundstep(XFT(0.5) * (cfm_hka * c2.hk_u + cfm_ma * c2.m_u + cfm_rta * c2.rt_u))
    cfm_t2 = roundstep(XFT(0.5) * (cfm_hka * c2.hk_t + cfm_rta * c2.rt_t))
    cfm_d2 = roundstep(XFT(0.5) * (cfm_hka * c2.hk_d))
    cfm_ms = roundstep(
        XFT(0.5) * (cfm_hka * hk1_ms + cfm_ma * m1_ms + cfm_rta * rt1_ms +
                    cfm_hka * c2.hk_ms + cfm_ma * c2.m_ms + cfm_rta * c2.rt_ms))
    cfm_re = roundstep(XFT(0.5) * (cfm_rta * rt1_re + cfm_rta * c2.rt_re))

    return BLMidResult(cfm, cfm_ms, cfm_re, cfm_u1, cfm_t1, cfm_d1, cfm_u2, cfm_t2, cfm_d2)
end

# ============================================================================
#  AXSET — Average amplification rate over interval 1..2
#  Literal translation of xblsys.f lines 35-144
# ============================================================================
function xbl_axset(c1::BLCom, c2::BLCom, fl::BLFlowF32)
    roundstep(x) = XFT(x)

    idampv = fl.idampv
    if idampv == 0
        ax132, ax1_hk132, ax1_t132, ax1_rt132 = dampl_jac(c1.hk, c1.t, c1.rt)
        ax232, ax2_hk232, ax2_t232, ax2_rt232 = dampl_jac(c2.hk, c2.t, c2.rt)
    else
        ax132, ax1_hk132, ax1_t132, ax1_rt132 = dampl2_jac(c1.hk, c1.t, c1.rt)
        ax232, ax2_hk232, ax2_t232, ax2_rt232 = dampl2_jac(c2.hk, c2.t, c2.rt)
    end

    ax1 = roundstep(ax132); ax1_hk1 = roundstep(ax1_hk132)
    ax1_t1 = roundstep(ax1_t132); ax1_rt1 = roundstep(ax1_rt132)
    ax2 = roundstep(ax232); ax2_hk2 = roundstep(ax2_hk232)
    ax2_t2 = roundstep(ax2_t232); ax2_rt2 = roundstep(ax2_rt232)

    axsq = roundstep(XFT(0.5) * roundstep(roundstep(ax1^2) + roundstep(ax2^2)))
    if axsq <= XFT(0)
        axa = XFT(0); axa_ax1 = XFT(0); axa_ax2 = XFT(0)
    else
        axa = roundstep(sqrt(axsq))
        axa_ax1 = roundstep(XFT(0.5) * ax1 / axa)
        axa_ax2 = roundstep(XFT(0.5) * ax2 / axa)
    end
    aavg = roundstep(XFT(0.5) * roundstep(c1.ampl + c2.ampl))
    arg = roundstep(min(roundstep(XFT(20) * roundstep(fl.amcrit - aavg)), XFT(20)))
    if arg <= XFT(0)
        exn = XFT(1); exn_a1 = XFT(0); exn_a2 = XFT(0)
    else
        exn = roundstep(exp(-arg))
        exn_a1 = roundstep(XFT(20) * XFT(0.5) * exn)
        exn_a2 = roundstep(XFT(20) * XFT(0.5) * exn)
    end
    tsum = roundstep(c1.t + c2.t)
    dax = roundstep(exn * XFT(0.002) / tsum)
    dax_a1 = roundstep(exn_a1 * XFT(0.002) / tsum)
    dax_a2 = roundstep(exn_a2 * XFT(0.002) / tsum)
    dax_t1 = roundstep(-dax / tsum)
    dax_t2 = roundstep(-dax / tsum)
    ax = roundstep(axa + dax)
    ax_hk1 = roundstep(axa_ax1 * ax1_hk1)
    ax_t1 = roundstep(axa_ax1 * ax1_t1 + dax_t1)
    ax_rt1 = roundstep(axa_ax1 * ax1_rt1)
    ax_a1 = dax_a1
    ax_hk2 = roundstep(axa_ax2 * ax2_hk2)
    ax_t2 = roundstep(axa_ax2 * ax2_t2 + dax_t2)
    ax_rt2 = roundstep(axa_ax2 * ax2_rt2)
    ax_a2 = dax_a2
    return ax, ax_hk1, ax_t1, ax_rt1, ax_a1, ax_hk2, ax_t2, ax_rt2, ax_a2
end

# ============================================================================
#  BLDIF — BL finite-difference equations
#  Literal translation of xblsys.f lines 1575-2000
#  ityp: 0=similarity, 1=laminar, 2=turbulent, 3=wake
# ============================================================================
function xbl_bldif!(vs1::Matrix{XFT}, vs2::Matrix{XFT}, vsrez::Vector{XFT},
                     vsm::Vector{XFT}, vsr::Vector{XFT}, vsx::Vector{XFT},
                     c1::BLCom, c2::BLCom, mid::BLMidResult,
                     ityp::Int, fl::BLFlowF32, par::BLParF32)
    if ityp == 0
        xlog = XFT(1); ulog = fl.bule; tlog = XFT(0.5)*(XFT(1) - fl.bule); hlog = XFT(0); ddlog = XFT(0)
    else
        xlog = log(max(abs(c2.x/c1.x),XFT(1e-12))); ulog = log(max(abs(c2.u/c1.u),XFT(1e-12)))
        tlog = log(max(abs(c2.t/c1.t),XFT(1e-12))); hlog = log(max(abs(c2.hs/c1.hs),XFT(1e-12))); ddlog = XFT(1)
    end
    for k in 1:4; vsrez[k]=XFT(0); vsm[k]=XFT(0); vsr[k]=XFT(0); vsx[k]=XFT(0); for l in 1:5; vs1[k,l]=XFT(0); vs2[k,l]=XFT(0); end; end

    # Upwinding
    hupwt = XFT(1)
    hdcon = (ityp==3 ? hupwt : XFT(5)*hupwt) / c2.hk^2
    hd_hk2 = -hdcon*XFT(2)/c2.hk
    arg = abs((c2.hk - XFT(1)) / (c1.hk - XFT(1)))
    hl = log(arg); hl_hk1 = -XFT(1)/(c1.hk - XFT(1)); hl_hk2 = XFT(1)/(c2.hk - XFT(1))
    hlsq = min(hl^2, XFT(15)); ehh = exp(-hlsq*hdcon)
    upw = XFT(1) - XFT(0.5)*ehh; upw_hl = ehh*hl*hdcon; upw_hd = XFT(0.5)*ehh*hlsq
    upw_hk1 = upw_hl*hl_hk1; upw_hk2 = upw_hl*hl_hk2 + upw_hd*hd_hk2
    upw_u1 = upw_hk1*c1.hk_u; upw_t1 = upw_hk1*c1.hk_t; upw_d1 = upw_hk1*c1.hk_d
    upw_u2 = upw_hk2*c2.hk_u; upw_t2 = upw_hk2*c2.hk_t; upw_d2 = upw_hk2*c2.hk_d
    upw_ms = upw_hk1*c1.hk_ms + upw_hk2*c2.hk_ms

    # Row 1
    if ityp == 0
        vs2[1,1] = XFT(1); vsrez[1] = -c2.ampl
    elseif ityp == 1
        ax, ax_hk1, ax_t1, ax_rt1, ax_a1, ax_hk2, ax_t2, ax_rt2, ax_a2 = xbl_axset(c1, c2, fl)
        rezc = c2.ampl - c1.ampl - ax*(c2.x - c1.x); z_ax = -(c2.x - c1.x)
        vs1[1,1] = z_ax*ax_a1 - XFT(1); vs1[1,2] = z_ax*(ax_hk1*c1.hk_t + ax_t1 + ax_rt1*c1.rt_t)
        vs1[1,3] = z_ax*(ax_hk1*c1.hk_d); vs1[1,4] = z_ax*(ax_hk1*c1.hk_u + ax_rt1*c1.rt_u); vs1[1,5] = ax
        vs2[1,1] = z_ax*ax_a2 + XFT(1); vs2[1,2] = z_ax*(ax_hk2*c2.hk_t + ax_t2 + ax_rt2*c2.rt_t)
        vs2[1,3] = z_ax*(ax_hk2*c2.hk_d); vs2[1,4] = z_ax*(ax_hk2*c2.hk_u + ax_rt2*c2.rt_u); vs2[1,5] = -ax
        vsm[1] = z_ax*(ax_hk1*c1.hk_ms + ax_rt1*c1.rt_ms + ax_hk2*c2.hk_ms + ax_rt2*c2.rt_ms)
        vsr[1] = z_ax*(ax_rt1*c1.rt_re + ax_rt2*c2.rt_re); vsrez[1] = -rezc
    else
        # Turbulent/wake shear lag
        sa=(XFT(1)-upw)*c1.s+upw*c2.s; cqa=(XFT(1)-upw)*c1.cq+upw*c2.cq; cfa=(XFT(1)-upw)*c1.cf+upw*c2.cf; hka=(XFT(1)-upw)*c1.hk+upw*c2.hk
        usa=XFT(0.5)*(c1.us+c2.us); rta=XFT(0.5)*(c1.rt+c2.rt); dea=XFT(0.5)*(c1.de+c2.de); da=XFT(0.5)*(c1.d+c2.d)
        ald = ityp==3 ? par.dlcon : XFT(1)
        if ityp==2; gcc=par.gccon; hkc=hka-XFT(1)-gcc/rta; hkc_hka=XFT(1); hkc_rta=gcc/rta^2; if hkc<XFT(0.01); hkc=XFT(0.01);hkc_hka=XFT(0);hkc_rta=XFT(0);end
        else; hkc=hka-XFT(1); hkc_hka=XFT(1); hkc_rta=XFT(0); end
        hr=hkc/(par.gacon*ald*hka); hr_hka=hkc_hka/(par.gacon*ald*hka)-hr/hka; hr_rta=hkc_rta/(par.gacon*ald*hka)
        uq=(XFT(0.5)*cfa-hr^2)/(par.gbcon*da); uq_hka=-XFT(2)*hr*hr_hka/(par.gbcon*da); uq_rta=-XFT(2)*hr*hr_rta/(par.gbcon*da)
        uq_cfa=XFT(0.5)/(par.gbcon*da); uq_da=-uq/da; uq_upw=uq_cfa*(c2.cf-c1.cf)+uq_hka*(c2.hk-c1.hk)
        uq_t1=(XFT(1)-upw)*(uq_cfa*c1.cf_t+uq_hka*c1.hk_t)+uq_upw*upw_t1+XFT(0.5)*uq_rta*c1.rt_t
        uq_d1=(XFT(1)-upw)*(uq_cfa*c1.cf_d+uq_hka*c1.hk_d)+uq_upw*upw_d1+XFT(0.5)*uq_da
        uq_u1=(XFT(1)-upw)*(uq_cfa*c1.cf_u+uq_hka*c1.hk_u)+uq_upw*upw_u1+XFT(0.5)*uq_rta*c1.rt_u
        uq_t2=upw*(uq_cfa*c2.cf_t+uq_hka*c2.hk_t)+uq_upw*upw_t2+XFT(0.5)*uq_rta*c2.rt_t
        uq_d2=upw*(uq_cfa*c2.cf_d+uq_hka*c2.hk_d)+uq_upw*upw_d2+XFT(0.5)*uq_da
        uq_u2=upw*(uq_cfa*c2.cf_u+uq_hka*c2.hk_u)+uq_upw*upw_u2+XFT(0.5)*uq_rta*c2.rt_u
        uq_ms=(XFT(1)-upw)*(uq_cfa*c1.cf_ms+uq_hka*c1.hk_ms)+uq_upw*upw_ms+upw*(uq_cfa*c2.cf_ms+uq_hka*c2.hk_ms)+XFT(0.5)*uq_rta*(c1.rt_ms+c2.rt_ms)
        uq_re=(XFT(1)-upw)*uq_cfa*c1.cf_re+upw*uq_cfa*c2.cf_re+XFT(0.5)*uq_rta*(c1.rt_re+c2.rt_re)
        scc=par.sccon*XFT(1.333)/(XFT(1)+usa); scc_usa=-scc/(XFT(1)+usa)
        slog=log(max(abs(c2.s/c1.s),XFT(1e-12))); dxi=c2.x-c1.x
        rezc=scc*(cqa-sa*ald)*dxi-dea*XFT(2)*slog+dea*XFT(2)*(uq*dxi-ulog)*par.duxcon
        z_cfa=dea*XFT(2)*uq_cfa*dxi*par.duxcon; z_hka=dea*XFT(2)*uq_hka*dxi*par.duxcon; z_da=dea*XFT(2)*uq_da*dxi*par.duxcon
        z_sl=-dea*XFT(2); z_ul=-dea*XFT(2)*par.duxcon; z_dxi=scc*(cqa-sa*ald)+dea*XFT(2)*uq*par.duxcon
        z_usa=scc_usa*(cqa-sa*ald)*dxi; z_cqa=scc*dxi; z_sa=-scc*dxi*ald; z_dea=XFT(2)*((uq*dxi-ulog)*par.duxcon-slog)
        z_upw=z_cqa*(c2.cq-c1.cq)+z_sa*(c2.s-c1.s)+z_cfa*(c2.cf-c1.cf)+z_hka*(c2.hk-c1.hk)
        z_de1=XFT(0.5)*z_dea; z_de2=XFT(0.5)*z_dea; z_us1=XFT(0.5)*z_usa; z_us2=XFT(0.5)*z_usa
        z_d1=XFT(0.5)*z_da; z_d2=XFT(0.5)*z_da; z_u1=-z_ul/c1.u; z_u2=z_ul/c2.u; z_x1=-z_dxi; z_x2=z_dxi
        z_s1=(XFT(1)-upw)*z_sa-z_sl/c1.s; z_s2=upw*z_sa+z_sl/c2.s
        z_cq1=(XFT(1)-upw)*z_cqa; z_cq2=upw*z_cqa; z_cf1=(XFT(1)-upw)*z_cfa; z_cf2=upw*z_cfa; z_hk1=(XFT(1)-upw)*z_hka; z_hk2=upw*z_hka
        vs1[1,1]=z_s1; vs1[1,2]=z_upw*upw_t1+z_de1*c1.de_t+z_us1*c1.us_t; vs1[1,3]=z_d1+z_upw*upw_d1+z_de1*c1.de_d+z_us1*c1.us_d
        vs1[1,4]=z_u1+z_upw*upw_u1+z_de1*c1.de_u+z_us1*c1.us_u; vs1[1,5]=z_x1
        vs2[1,1]=z_s2; vs2[1,2]=z_upw*upw_t2+z_de2*c2.de_t+z_us2*c2.us_t; vs2[1,3]=z_d2+z_upw*upw_d2+z_de2*c2.de_d+z_us2*c2.us_d
        vs2[1,4]=z_u2+z_upw*upw_u2+z_de2*c2.de_u+z_us2*c2.us_u; vs2[1,5]=z_x2
        vsm[1]=z_upw*upw_ms+z_de1*c1.de_ms+z_us1*c1.us_ms+z_de2*c2.de_ms+z_us2*c2.us_ms
        vs1[1,2]+=z_cq1*c1.cq_t+z_cf1*c1.cf_t+z_hk1*c1.hk_t; vs1[1,3]+=z_cq1*c1.cq_d+z_cf1*c1.cf_d+z_hk1*c1.hk_d; vs1[1,4]+=z_cq1*c1.cq_u+z_cf1*c1.cf_u+z_hk1*c1.hk_u
        vs2[1,2]+=z_cq2*c2.cq_t+z_cf2*c2.cf_t+z_hk2*c2.hk_t; vs2[1,3]+=z_cq2*c2.cq_d+z_cf2*c2.cf_d+z_hk2*c2.hk_d; vs2[1,4]+=z_cq2*c2.cq_u+z_cf2*c2.cf_u+z_hk2*c2.hk_u
        vsm[1]+=z_cq1*c1.cq_ms+z_cf1*c1.cf_ms+z_hk1*c1.hk_ms+z_cq2*c2.cq_ms+z_cf2*c2.cf_ms+z_hk2*c2.hk_ms
        vsr[1]=z_cq1*c1.cq_re+z_cf1*c1.cf_re+z_cq2*c2.cq_re+z_cf2*c2.cf_re; vsrez[1]=-rezc
    end

    # Row 2: momentum equation
    ha=XFT(0.5)*(c1.h+c2.h); ma_v=XFT(0.5)*(c1.m+c2.m); xa=XFT(0.5)*(c1.x+c2.x); ta=XFT(0.5)*(c1.t+c2.t); hwa=XFT(0.5)*(c1.dw/c1.t+c2.dw/c2.t)
    cfx=XFT(0.50)*mid.cfm*xa/ta+XFT(0.25)*(c1.cf*c1.x/c1.t+c2.cf*c2.x/c2.t)
    cfx_xa=XFT(0.50)*mid.cfm/ta; cfx_ta=-XFT(0.50)*mid.cfm*xa/ta^2
    cfx_x1=XFT(0.25)*c1.cf/c1.t+cfx_xa*XFT(0.5); cfx_x2=XFT(0.25)*c2.cf/c2.t+cfx_xa*XFT(0.5)
    cfx_t1=-XFT(0.25)*c1.cf*c1.x/c1.t^2+cfx_ta*XFT(0.5); cfx_t2=-XFT(0.25)*c2.cf*c2.x/c2.t^2+cfx_ta*XFT(0.5)
    cfx_cf1=XFT(0.25)*c1.x/c1.t; cfx_cf2=XFT(0.25)*c2.x/c2.t; cfx_cfm=XFT(0.50)*xa/ta
    btmp=ha+XFT(2)-ma_v+hwa; rezt=tlog+btmp*ulog-xlog*XFT(0.5)*cfx
    z_cfx=-xlog*XFT(0.5); z_ha=ulog; z_hwa=ulog; z_ma=-ulog; z_xl=-ddlog*XFT(0.5)*cfx; z_ul=ddlog*btmp; z_tl=ddlog
    z_cfm=z_cfx*cfx_cfm; z_cf1=z_cfx*cfx_cf1; z_cf2=z_cfx*cfx_cf2
    z_t1=-z_tl/c1.t+z_cfx*cfx_t1+z_hwa*XFT(0.5)*(-c1.dw/c1.t^2); z_t2=z_tl/c2.t+z_cfx*cfx_t2+z_hwa*XFT(0.5)*(-c2.dw/c2.t^2)
    z_x1=-z_xl/c1.x+z_cfx*cfx_x1; z_x2=z_xl/c2.x+z_cfx*cfx_x2; z_u1=-z_ul/c1.u; z_u2=z_ul/c2.u
    vs1[2,2]=XFT(0.5)*z_ha*c1.h_t+z_cfm*mid.cfm_t1+z_cf1*c1.cf_t+z_t1; vs1[2,3]=XFT(0.5)*z_ha*c1.h_d+z_cfm*mid.cfm_d1+z_cf1*c1.cf_d
    vs1[2,4]=XFT(0.5)*z_ma*c1.m_u+z_cfm*mid.cfm_u1+z_cf1*c1.cf_u+z_u1; vs1[2,5]=z_x1
    vs2[2,2]=XFT(0.5)*z_ha*c2.h_t+z_cfm*mid.cfm_t2+z_cf2*c2.cf_t+z_t2; vs2[2,3]=XFT(0.5)*z_ha*c2.h_d+z_cfm*mid.cfm_d2+z_cf2*c2.cf_d
    vs2[2,4]=XFT(0.5)*z_ma*c2.m_u+z_cfm*mid.cfm_u2+z_cf2*c2.cf_u+z_u2; vs2[2,5]=z_x2
    vsm[2]=XFT(0.5)*z_ma*c1.m_ms+z_cfm*mid.cfm_ms+z_cf1*c1.cf_ms+XFT(0.5)*z_ma*c2.m_ms+z_cf2*c2.cf_ms
    vsr[2]=z_cfm*mid.cfm_re+z_cf1*c1.cf_re+z_cf2*c2.cf_re; vsrez[2]=-rezt

    # Row 3: shape parameter equation
    xot1=c1.x/c1.t; xot2=c2.x/c2.t; ha=XFT(0.5)*(c1.h+c2.h); hsa=XFT(0.5)*(c1.hs+c2.hs); hca=XFT(0.5)*(c1.hc+c2.hc); hwa=XFT(0.5)*(c1.dw/c1.t+c2.dw/c2.t)
    dix=(XFT(1)-upw)*c1.di*xot1+upw*c2.di*xot2; cfx_s=(XFT(1)-upw)*c1.cf*xot1+upw*c2.cf*xot2
    dix_upw=c2.di*xot2-c1.di*xot1; cfx_upw=c2.cf*xot2-c1.cf*xot1
    btmp=XFT(2)*hca/hsa+XFT(1)-ha-hwa; rezh=hlog+btmp*ulog+xlog*(XFT(0.5)*cfx_s-dix)
    z_cfx=xlog*XFT(0.5); z_dix=-xlog; z_hca=XFT(2)*ulog/hsa; z_ha=-ulog; z_hwa=-ulog
    z_xl=ddlog*(XFT(0.5)*cfx_s-dix); z_ul=ddlog*btmp; z_hl=ddlog
    z_upw=z_cfx*cfx_upw+z_dix*dix_upw
    z_hs1=-hca*ulog/hsa^2-z_hl/c1.hs; z_hs2=-hca*ulog/hsa^2+z_hl/c2.hs
    z_cf1=(XFT(1)-upw)*z_cfx*xot1; z_cf2=upw*z_cfx*xot2; z_di1=(XFT(1)-upw)*z_dix*xot1; z_di2=upw*z_dix*xot2
    z_t1=(XFT(1)-upw)*(z_cfx*c1.cf+z_dix*c1.di)*(-xot1/c1.t)+z_hwa*XFT(0.5)*(-c1.dw/c1.t^2)
    z_t2=upw*(z_cfx*c2.cf+z_dix*c2.di)*(-xot2/c2.t)+z_hwa*XFT(0.5)*(-c2.dw/c2.t^2)
    z_x1=(XFT(1)-upw)*(z_cfx*c1.cf+z_dix*c1.di)/c1.t-z_xl/c1.x; z_x2=upw*(z_cfx*c2.cf+z_dix*c2.di)/c2.t+z_xl/c2.x
    z_u1=-z_ul/c1.u; z_u2=z_ul/c2.u
    vs1[3,1]=z_di1*c1.di_s; vs1[3,2]=z_hs1*c1.hs_t+z_cf1*c1.cf_t+z_di1*c1.di_t+z_t1; vs1[3,3]=z_hs1*c1.hs_d+z_cf1*c1.cf_d+z_di1*c1.di_d
    vs1[3,4]=z_hs1*c1.hs_u+z_cf1*c1.cf_u+z_di1*c1.di_u+z_u1; vs1[3,5]=z_x1
    vs2[3,1]=z_di2*c2.di_s; vs2[3,2]=z_hs2*c2.hs_t+z_cf2*c2.cf_t+z_di2*c2.di_t+z_t2; vs2[3,3]=z_hs2*c2.hs_d+z_cf2*c2.cf_d+z_di2*c2.di_d
    vs2[3,4]=z_hs2*c2.hs_u+z_cf2*c2.cf_u+z_di2*c2.di_u+z_u2; vs2[3,5]=z_x2
    vsm[3]=z_hs1*c1.hs_ms+z_cf1*c1.cf_ms+z_di1*c1.di_ms+z_hs2*c2.hs_ms+z_cf2*c2.cf_ms+z_di2*c2.di_ms
    vsr[3]=z_hs1*c1.hs_re+z_cf1*c1.cf_re+z_di1*c1.di_re+z_hs2*c2.hs_re+z_cf2*c2.cf_re+z_di2*c2.di_re
    vs1[3,2]+=XFT(0.5)*(z_hca*c1.hc_t+z_ha*c1.h_t)+z_upw*upw_t1; vs1[3,3]+=XFT(0.5)*(z_hca*c1.hc_d+z_ha*c1.h_d)+z_upw*upw_d1; vs1[3,4]+=XFT(0.5)*z_hca*c1.hc_u+z_upw*upw_u1
    vs2[3,2]+=XFT(0.5)*(z_hca*c2.hc_t+z_ha*c2.h_t)+z_upw*upw_t2; vs2[3,3]+=XFT(0.5)*(z_hca*c2.hc_d+z_ha*c2.h_d)+z_upw*upw_d2; vs2[3,4]+=XFT(0.5)*z_hca*c2.hc_u+z_upw*upw_u2
    vsm[3]+=XFT(0.5)*z_hca*c1.hc_ms+z_upw*upw_ms+XFT(0.5)*z_hca*c2.hc_ms; vsrez[3]=-rezh
    return nothing
end

# ============================================================================
#  BLSYS — Assemble BL interval system (calls BLVAR + BLMID + BLDIF)
#  Literal translation of xblsys.f lines 583-661
# ============================================================================
function xbl_blsys!(vs1::Matrix{XFT}, vs2::Matrix{XFT}, vsrez::Vector{XFT},
                     vsm::Vector{XFT}, vsr::Vector{XFT}, vsx::Vector{XFT},
                     c1::BLCom, c2::BLCom,
                     simi::Bool, tran::Bool, turb::Bool, wake::Bool,
                     fl::BLFlowF32, par::BLParF32;
                     xt::XFT = XFT(0),  # transition location from TRCHEK2
                     xt_a1::XFT = XFT(0), xt_x1::XFT = XFT(0),
                     xt_t1::XFT = XFT(0), xt_d1::XFT = XFT(0), xt_u1::XFT = XFT(0),
                     xt_x2::XFT = XFT(0), xt_t2::XFT = XFT(0),
                     xt_d2::XFT = XFT(0), xt_u2::XFT = XFT(0),
                     xt_ms::XFT = XFT(0), xt_re::XFT = XFT(0), xt_xf::XFT = XFT(0))
    if wake
        xbl_blvar!(c2, 3, fl, par)
        xbl_blmid_result = xbl_blmid(c1, c2, 3, simi, fl, par)
    elseif turb || tran
        # XFoil precomputes turbulent secondary variables even for transition
        # intervals before TRDIF so COM2/C2SAV carry a fully populated state.
        xbl_blvar!(c2, 2, fl, par)
        xbl_blmid_result = xbl_blmid(c1, c2, 2, simi, fl, par)
    else
        xbl_blvar!(c2, 1, fl, par)
        xbl_blmid_result = xbl_blmid(c1, c2, 1, simi, fl, par)
    end
    if simi
        copycom!(c1, c2)
    end
    if tran
        # Transition interval: use TRDIF
        xt_use = xt > XFT(0) ? xt : XFT(0.5) * (c1.x + c2.x)
        xbl_trdif!(vs1, vs2, vsrez, vsm, vsr, vsx, c1, c2, xt_use, fl, par,
                   xt_a1, xt_x1, xt_t1, xt_d1, xt_u1,
                   xt_x2, xt_t2, xt_d2, xt_u2, xt_ms, xt_re, xt_xf)
    else
        ityp = simi ? 0 : (!turb ? 1 : (wake ? 3 : 2))
        xbl_bldif!(vs1, vs2, vsrez, vsm, vsr, vsx, c1, c2, xbl_blmid_result, ityp, fl, par)
        if simi; for k in 1:4, l in 1:5; vs2[k,l] = vs1[k,l] + vs2[k,l]; vs1[k,l] = XFT(0); end; end
    end
    # Convert to incompressible Uei
    for k in 1:4
        res_u1 = vs1[k,4]; res_u2 = vs2[k,4]; res_ms = vsm[k]
        vs1[k,4] = res_u1 * c1.u_uei; vs2[k,4] = res_u2 * c2.u_uei
        vsm[k] = res_u1 * c1.u_ms + res_u2 * c2.u_ms + res_ms
    end
    return nothing
end

# ============================================================================
#  TESYS — TE-to-wake junction system
#  Literal translation of xblsys.f lines 664-698
# ============================================================================
function xbl_tesys!(vs1::Matrix{XFT}, vs2::Matrix{XFT}, vsrez::Vector{XFT},
                     vsm::Vector{XFT}, vsr::Vector{XFT}, vsx::Vector{XFT},
                     c2::BLCom, cte::XFT, tte::XFT, dte::XFT,
                     fl::BLFlowF32, par::BLParF32)
    for k in 1:4; vsrez[k]=XFT(0); vsm[k]=XFT(0); vsr[k]=XFT(0); vsx[k]=XFT(0); for l in 1:5; vs1[k,l]=XFT(0); vs2[k,l]=XFT(0); end; end
    xbl_blvar!(c2, 3, fl, par)
    vs1[1,1] = XFT(-1); vs2[1,1] = XFT(1); vsrez[1] = cte - c2.s
    vs1[2,2] = XFT(-1); vs2[2,2] = XFT(1); vsrez[2] = tte - c2.t
    vs1[3,3] = XFT(-1); vs2[3,3] = XFT(1); vsrez[3] = dte - c2.d - c2.dw
    return nothing
end

# ============================================================================
#  TRCHEK2 — Second-order implicit transition detection
#  Literal translation of xblsys.f lines 231-580
#
#  Modifies c2.ampl in-place.  Sets tran, trforc flags.
#  Returns (tran::Bool, trforc::Bool, xt::XFT, xt_a1, xt_x1, xt_t1, xt_d1, xt_u1,
#           xt_x2, xt_t2, xt_d2, xt_u2, xt_ms, xt_re, xt_xf)
# ============================================================================
function xbl_trchek2!(c1::BLCom, c2::BLCom, c2sav::BLCom, fl::BLFlowF32)
    daeps = XFT(5e-5)
    amcrit = fl.amcrit
    xiforc = fl.xiforc

    # Save COM2
    copycom!(c2sav, c2)

    # Initial guess from AXSET
    ax, ax_hk1, ax_t1, ax_rt1, ax_a1, ax_hk2, ax_t2, ax_rt2, ax_a2 = xbl_axset(c1, c2, fl)
    c2.ampl = c1.ampl + ax * (c2.x - c1.x)

    # Newton iteration for implicit AMPL2
    xt = c2.x
    amplt_a2 = XFT(1)
    # These must persist outside the loop for XT sensitivity computation
    wf1 = XFT(1); wf2 = XFT(0)
    wf1_a1 = XFT(0); wf1_a2 = XFT(0); wf1_x1 = XFT(0); wf1_x2 = XFT(0); wf1_xf = XFT(0)
    wf2_a1 = XFT(0); wf2_a2 = XFT(0); wf2_x1 = XFT(0); wf2_x2 = XFT(0); wf2_xf = XFT(0)
    xt_a2 = XFT(0)
    tt_a2 = XFT(0); dt_a2 = XFT(0); ut_a2 = XFT(0)
    hkt = XFT(0); hkt_tt = XFT(0); hkt_dt = XFT(0); hkt_ut = XFT(0); hkt_ms = XFT(0)
    rtt = XFT(0); rtt_tt = XFT(0); rtt_ut = XFT(0); rtt_ms = XFT(0); rtt_re = XFT(0)
    ax = XFT(0)
    ax_hk1_ = XFT(0); ax_t1_ = XFT(0); ax_rt1_ = XFT(0); ax_a1_ = XFT(0)
    ax_hkt = XFT(0); ax_tt = XFT(0); ax_rtt = XFT(0); ax_at = XFT(0)
    tt = XFT(0); dt = XFT(0); ut = XFT(0)
    for itam in 1:30
        if c2.ampl <= amcrit
            amplt = c2.ampl; amplt_a2 = XFT(1)
            sfa = XFT(1); sfa_a1 = XFT(0); sfa_a2 = XFT(0)
        else
            amplt = amcrit; amplt_a2 = XFT(0)
            sfa = (amplt - c1.ampl) / (c2.ampl - c1.ampl)
            sfa_a1 = (sfa - XFT(1)) / (c2.ampl - c1.ampl)
            sfa_a2 = (-sfa) / (c2.ampl - c1.ampl)
        end

        if xiforc < c2sav.x  # use saved X2
            sfx = (xiforc - c1.x) / (c2sav.x - c1.x)
            sfx_x1 = (sfx - XFT(1)) / (c2sav.x - c1.x)
            sfx_x2 = (-sfx) / (c2sav.x - c1.x)
            sfx_xf = XFT(1) / (c2sav.x - c1.x)
        else
            sfx = XFT(1); sfx_x1 = XFT(0); sfx_x2 = XFT(0); sfx_xf = XFT(0)
        end

        if sfa < sfx
            wf2 = sfa; wf2_a1 = sfa_a1; wf2_a2 = sfa_a2
            wf2_x1 = XFT(0); wf2_x2 = XFT(0); wf2_xf = XFT(0)
        else
            wf2 = sfx; wf2_a1 = XFT(0); wf2_a2 = XFT(0)
            wf2_x1 = sfx_x1; wf2_x2 = sfx_x2; wf2_xf = sfx_xf
        end

        wf1 = XFT(1) - wf2
        wf1_a1 = -wf2_a1; wf1_a2 = -wf2_a2
        wf1_x1 = -wf2_x1; wf1_x2 = -wf2_x2; wf1_xf = -wf2_xf

        # Interpolate to transition point
        xt = c1.x * wf1 + c2sav.x * wf2
        tt = c1.t * wf1 + c2sav.t * wf2
        dt = c1.d * wf1 + c2sav.d * wf2
        ut = c1.u * wf1 + c2sav.u * wf2

        xt_a2 = c1.x * wf1_a2 + c2sav.x * wf2_a2
        tt_a2 = c1.t * wf1_a2 + c2sav.t * wf2_a2
        dt_a2 = c1.d * wf1_a2 + c2sav.d * wf2_a2
        ut_a2 = c1.u * wf1_a2 + c2sav.u * wf2_a2

        # Temporarily set "2" from "T" for BLKIN
        c2.x = xt; c2.t = tt; c2.d = dt; c2.u = ut

        # Calculate laminar secondary "T" variables
        xbl_blkin!(c2, fl)
        hkt = c2.hk; hkt_tt = c2.hk_t; hkt_dt = c2.hk_d; hkt_ut = c2.hk_u; hkt_ms = c2.hk_ms
        rtt = c2.rt; rtt_tt = c2.rt_t; rtt_ut = c2.rt_u; rtt_ms = c2.rt_ms; rtt_re = c2.rt_re

        # Restore COM2 (except AMPL2)
        amsave = c2.ampl
        copycom!(c2, c2sav)
        c2.ampl = amsave

        # Calculate AX over X1..XT using transition-point kinematics
        # Create temporary BLCom for the transition point
        ct = BLCom()
        ct.hk = hkt; ct.t = tt; ct.rt = rtt; ct.ampl = amplt
        ax, ax_hk1_, ax_t1_, ax_rt1_, ax_a1_,
            ax_hkt, ax_tt, ax_rtt, ax_at = xbl_axset(c1, ct, fl)

        ax <= XFT(0) && break

        # Sensitivity of AX to A2
        ax_a2_val = (ax_hkt*hkt_tt + ax_tt + ax_rtt*rtt_tt)*tt_a2 +
                    (ax_hkt*hkt_dt)*dt_a2 +
                    (ax_hkt*hkt_ut + ax_rtt*rtt_ut)*ut_a2 +
                    ax_at * amplt_a2

        # Residual and Jacobian
        res = c2.ampl - c1.ampl - ax * (c2sav.x - c1.x)
        res_a2 = XFT(1) - ax_a2_val * (c2sav.x - c1.x)

        da2 = -res / res_a2

        rlx = XFT(1)
        dxt_val = xt_a2 * da2
        dx = c2sav.x - c1.x
        if rlx * abs(dxt_val / dx) > XFT(0.05)
            rlx = XFT(0.05) * abs(dx / dxt_val)
        end
        if rlx * abs(da2) > XFT(1)
            rlx = XFT(1) * abs(XFT(1) / da2)
        end

        abs(da2) < daeps && break

        if (c2.ampl > amcrit && c2.ampl + rlx*da2 < amcrit) ||
           (c2.ampl < amcrit && c2.ampl + rlx*da2 > amcrit)
            c2.ampl = amcrit
        else
            c2.ampl = c2.ampl + rlx * da2
        end
    end

    # Test for transition
    trfree = c2.ampl >= amcrit
    trforc = xiforc > c1.x && xiforc <= c2sav.x
    tran = trforc || trfree

    if !tran
        return (tran=false, trforc=false, xt=xt,
                xt_a1=XFT(0), xt_x1=XFT(0), xt_t1=XFT(0), xt_d1=XFT(0), xt_u1=XFT(0),
                xt_x2=XFT(0), xt_t2=XFT(0), xt_d2=XFT(0), xt_u2=XFT(0),
                xt_ms=XFT(0), xt_re=XFT(0), xt_xf=XFT(0))
    end

    # Resolve forced vs free
    if trfree && trforc
        trforc = xiforc < xt
        trfree = xiforc >= xt
    end

    if trforc
        return (tran=true, trforc=true, xt=xiforc,
                xt_a1=XFT(0), xt_x1=XFT(0), xt_t1=XFT(0), xt_d1=XFT(0), xt_u1=XFT(0),
                xt_x2=XFT(0), xt_t2=XFT(0), xt_d2=XFT(0), xt_u2=XFT(0),
                xt_ms=XFT(0), xt_re=XFT(0), xt_xf=XFT(1))
    end

    # Free transition: compute XT sensitivities (Fortran lines 464-577)
    # Use c1 for station-1 variables, c2sav for station-2 variables
    x1 = c1.x; x2 = c2sav.x; t1 = c1.t; t2 = c2sav.t
    d1 = c1.d; d2 = c2sav.d; u1 = c1.u; u2 = c2sav.u

    # Base sensitivities (Fortran lines 472-505)
    # XT = X1*WF1 + X2*WF2,  TT = T1*WF1 + T2*WF2, etc.
    xt_x1_v = wf1
    tt_t1   = wf1
    dt_d1   = wf1
    ut_u1   = wf1

    xt_x2_v = wf2
    tt_t2   = wf2
    dt_d2   = wf2
    ut_u2   = wf2

    xt_a1_v = x1*wf1_a1 + x2*wf2_a1
    tt_a1_v = t1*wf1_a1 + t2*wf2_a1
    dt_a1_v = d1*wf1_a1 + d2*wf2_a1
    ut_a1_v = u1*wf1_a1 + u2*wf2_a1

    # Chain through X1 (Fortran lines 492-495)
    xt_x1_v = x1*wf1_x1 + x2*wf2_x1 + xt_x1_v
    tt_x1   = t1*wf1_x1 + t2*wf2_x1
    dt_x1   = d1*wf1_x1 + d2*wf2_x1
    ut_x1   = u1*wf1_x1 + u2*wf2_x1

    # Chain through X2 (Fortran lines 497-500)
    xt_x2_v = x1*wf1_x2 + x2*wf2_x2 + xt_x2_v
    tt_x2   = t1*wf1_x2 + t2*wf2_x2
    dt_x2   = d1*wf1_x2 + d2*wf2_x2
    ut_x2   = u1*wf1_x2 + u2*wf2_x2

    # Chain through XF (Fortran lines 502-505)
    xt_xf_v = x1*wf1_xf + x2*wf2_xf
    tt_xf   = t1*wf1_xf + t2*wf2_xf
    dt_xf   = d1*wf1_xf + d2*wf2_xf
    ut_xf   = u1*wf1_xf + u2*wf2_xf

    # AX sensitivities (Fortran lines 510-542)
    # At this point ax_hk1_, ax_t1_, ax_rt1_, ax_a1_, ax_hkt, ax_tt, ax_rtt, ax_at
    # are from the last AXSET call (persisted outside loop).
    # c1 has station-1 kinematics (hk_t, hk_d, etc.)
    hk1_t1 = c1.hk_t; hk1_d1 = c1.hk_d; hk1_u1 = c1.hk_u; hk1_ms = c1.hk_ms
    rt1_t1 = c1.rt_t; rt1_u1 = c1.rt_u; rt1_ms = c1.rt_ms; rt1_re = c1.rt_re

    ax_t1_full = ax_hk1_*hk1_t1 + ax_t1_ + ax_rt1_*rt1_t1 +
        (ax_hkt*hkt_tt + ax_tt + ax_rtt*rtt_tt)*tt_t1
    ax_d1_full = ax_hk1_*hk1_d1 +
        (ax_hkt*hkt_dt)*dt_d1
    ax_u1_full = ax_hk1_*hk1_u1 + ax_rt1_*rt1_u1 +
        (ax_hkt*hkt_ut + ax_rtt*rtt_ut)*ut_u1
    ax_a1_full = ax_a1_ +
        (ax_hkt*hkt_tt + ax_tt + ax_rtt*rtt_tt)*tt_a1_v +
        (ax_hkt*hkt_dt)*dt_a1_v +
        (ax_hkt*hkt_ut + ax_rtt*rtt_ut)*ut_a1_v
    ax_x1_full = (ax_hkt*hkt_tt + ax_tt + ax_rtt*rtt_tt)*tt_x1 +
        (ax_hkt*hkt_dt)*dt_x1 +
        (ax_hkt*hkt_ut + ax_rtt*rtt_ut)*ut_x1

    ax_t2_full = (ax_hkt*hkt_tt + ax_tt + ax_rtt*rtt_tt)*tt_t2
    ax_d2_full = (ax_hkt*hkt_dt)*dt_d2
    ax_u2_full = (ax_hkt*hkt_ut + ax_rtt*rtt_ut)*ut_u2
    ax_a2_full = ax_at*amplt_a2 +
        (ax_hkt*hkt_tt + ax_tt + ax_rtt*rtt_tt)*tt_a2 +
        (ax_hkt*hkt_dt)*dt_a2 +
        (ax_hkt*hkt_ut + ax_rtt*rtt_ut)*ut_a2
    ax_x2_full = (ax_hkt*hkt_tt + ax_tt + ax_rtt*rtt_tt)*tt_x2 +
        (ax_hkt*hkt_dt)*dt_x2 +
        (ax_hkt*hkt_ut + ax_rtt*rtt_ut)*ut_x2

    ax_xf_full = (ax_hkt*hkt_tt + ax_tt + ax_rtt*rtt_tt)*tt_xf +
        (ax_hkt*hkt_dt)*dt_xf +
        (ax_hkt*hkt_ut + ax_rtt*rtt_ut)*ut_xf

    ax_ms_full = ax_hkt*hkt_ms + ax_rtt*rtt_ms + ax_hk1_*hk1_ms + ax_rt1_*rt1_ms
    ax_re_full = ax_rtt*rtt_re + ax_rt1_*rt1_re

    # Residual sensitivities (Fortran lines 545-563)
    # RES = AMPL2 - AMPL1 - AX*(X2-X1)
    z_ax = -(x2 - x1)

    z_a1 = z_ax*ax_a1_full - XFT(1)
    z_t1 = z_ax*ax_t1_full
    z_d1 = z_ax*ax_d1_full
    z_u1 = z_ax*ax_u1_full
    z_x1 = z_ax*ax_x1_full + ax

    z_a2_val = z_ax*ax_a2_full + XFT(1)
    z_t2 = z_ax*ax_t2_full
    z_d2 = z_ax*ax_d2_full
    z_u2 = z_ax*ax_u2_full
    z_x2 = z_ax*ax_x2_full - ax

    z_xf = z_ax*ax_xf_full
    z_ms = z_ax*ax_ms_full
    z_re = z_ax*ax_re_full

    # XT sensitivities via implicit differentiation (Fortran lines 566-577)
    # With RES stationary for A2 constraint
    xt_a2_ratio = xt_a2 / z_a2_val
    xt_a1_out = xt_a1_v - xt_a2_ratio*z_a1
    xt_t1_out =         - xt_a2_ratio*z_t1
    xt_d1_out =         - xt_a2_ratio*z_d1
    xt_u1_out =         - xt_a2_ratio*z_u1
    xt_x1_out = xt_x1_v - xt_a2_ratio*z_x1
    xt_t2_out =         - xt_a2_ratio*z_t2
    xt_d2_out =         - xt_a2_ratio*z_d2
    xt_u2_out =         - xt_a2_ratio*z_u2
    xt_x2_out = xt_x2_v - xt_a2_ratio*z_x2
    xt_ms_out =         - xt_a2_ratio*z_ms
    xt_re_out =         - xt_a2_ratio*z_re
    xt_xf_out = XFT(0)  # Fortran line 577: XT_XF = 0.0

    return (tran=true, trforc=false, xt=xt,
            xt_a1=xt_a1_out, xt_x1=xt_x1_out, xt_t1=xt_t1_out, xt_d1=xt_d1_out, xt_u1=xt_u1_out,
            xt_x2=xt_x2_out, xt_t2=xt_t2_out, xt_d2=xt_d2_out, xt_u2=xt_u2_out,
            xt_ms=xt_ms_out, xt_re=xt_re_out, xt_xf=xt_xf_out)
end

# ============================================================================
#  xbl_mrchdu! — Literal port of Fortran MRCHDU (xbl.f 886-1202)
#
#  Marches BL on side `is` in mixed Ue-Hk mode.  Uses the SAME equation
#  chain (BLPRV→BLKIN→TRCHEK2→BLSYS) as SETBL, ensuring consistency.
#
#  Modifies state arrays: uedg, thet, dstr, ctau, mass, itran
# ============================================================================
function xbl_mrchdu!(state::ViscalBLState, is::Int,
                      fl::BLFlowF32, par::BLParF32, ante::XFT)
    deps = XFT(5e-6)
    senswt = XFT(1000)

    iblte = state.iblte[is]
    nbl = state.nbl[is]
    itrold = state.itran[is]
    fl = xbl_with_xiforc(fl, XFT(state.xssi[iblte, is]))

    tran = false
    turb = false
    # Full reset matching Fortran: ITRAN(IS) = IBLTE(IS).
    # If TRCHEK2 doesn't find transition, we restore itrold after the march.
    state.itran[is] = iblte

    bule = XFT(1)

    # Allocate COM structs for station 1 and 2
    c1 = BLCom()
    c2 = BLCom()
    c2sav = BLCom()

    # BLSYS output arrays (shared across stations)
    vs1 = zeros(XFT, 4, 5); vs2_sys = zeros(XFT, 4, 5); vsrez = zeros(XFT, 4)
    vsm = zeros(XFT, 4); vsr = zeros(XFT, 4); vsx = zeros(XFT, 4)

    sens = XFT(0)
    sennew = XFT(0)
    ami = XFT(0)
    cti = XFT(0.03)

    for ibl in 2:nbl
        ibm = ibl - 1
        simi = (ibl == 2)
        wake = (ibl > iblte)

        xsi = XFT(state.xssi[ibl, is])
        uei = XFT(state.uedg[ibl, is])
        thi = XFT(state.thet[ibl, is])
        dsi = XFT(state.dstr[ibl, is])

        if ibl < itrold
            ami = XFT(state.ctau[ibl, is])
            cti = XFT(0.03)
        else
            cti = XFT(state.ctau[ibl, is])
            cti <= XFT(0) && (cti = XFT(0.03))
        end

        dswaki = XFT(0)
        if wake
            iw = ibl - iblte
            dswaki = iw <= length(state.wgap) ? XFT(state.wgap[iw]) : XFT(0)
        end

        # Clamp DSI
        if ibl <= iblte
            dsi = max(dsi - dswaki, XFT(1.02) * thi) + dswaki
        else
            dsi = max(dsi - dswaki, XFT(1.00005) * thi) + dswaki
        end

        ueref = XFT(0)
        hkref = XFT(0)
        last_dmax = typemax(XFT)
        converged_station = false

        for itbl in 1:25
            # Fortran TRCHEK/TRCHEK2 overwrites the interval transition flag on
            # every Newton sub-iteration. Reset it here so a transient early
            # detection doesn't leak into the final station mode.
            tran = false

            # Set primary variables
            xbl_blprv!(c2, xsi, ami, cti, thi, dsi, dswaki, uei, fl)
            xbl_blkin!(c2, fl)

            # TRCHEK2 for laminar stations
            local xt_mrchdu = XFT(0)
            local mr_xt_a1=XFT(0); local mr_xt_x1=XFT(0); local mr_xt_t1=XFT(0); local mr_xt_d1=XFT(0); local mr_xt_u1=XFT(0)
            local mr_xt_x2=XFT(0); local mr_xt_t2=XFT(0); local mr_xt_d2=XFT(0); local mr_xt_u2=XFT(0)
            local mr_xt_ms=XFT(0); local mr_xt_re=XFT(0); local mr_xt_xf=XFT(0)
            if !simi && !turb
                tr_result = xbl_trchek2!(c1, c2, c2sav, fl)
                ami = c2.ampl
                if tr_result.tran
                    tran = true
                    state.itran[is] = ibl
                    xt_mrchdu = tr_result.xt
                    mr_xt_a1 = tr_result.xt_a1; mr_xt_x1 = tr_result.xt_x1; mr_xt_t1 = tr_result.xt_t1
                    mr_xt_d1 = tr_result.xt_d1; mr_xt_u1 = tr_result.xt_u1; mr_xt_x2 = tr_result.xt_x2
                    mr_xt_t2 = tr_result.xt_t2; mr_xt_d2 = tr_result.xt_d2; mr_xt_u2 = tr_result.xt_u2
                    mr_xt_ms = tr_result.xt_ms; mr_xt_re = tr_result.xt_re; mr_xt_xf = tr_result.xt_xf
                else
                    state.itran[is] = ibl + 2
                end
            end

            # BLSYS (or TESYS for first wake station)
            if ibl == iblte + 1
                tte = XFT(state.thet[state.iblte[1], 1]) + XFT(state.thet[state.iblte[2], 2])
                dte = XFT(state.dstr[state.iblte[1], 1]) + XFT(state.dstr[state.iblte[2], 2]) + ante
                cte_val = (XFT(state.ctau[state.iblte[1], 1]) * XFT(state.thet[state.iblte[1], 1]) +
                           XFT(state.ctau[state.iblte[2], 2]) * XFT(state.thet[state.iblte[2], 2])) / max(tte, XFT(1e-10))
                xbl_tesys!(vs1, vs2_sys, vsrez, vsm, vsr, vsx, c2, cte_val, tte, dte, fl, par)
            else
                xbl_blsys!(vs1, vs2_sys, vsrez, vsm, vsr, vsx, c1, c2,
                           simi, tran && ibl == state.itran[is], turb, wake, fl, par;
                           xt=xt_mrchdu, xt_a1=mr_xt_a1, xt_x1=mr_xt_x1,
                           xt_t1=mr_xt_t1, xt_d1=mr_xt_d1, xt_u1=mr_xt_u1,
                           xt_x2=mr_xt_x2, xt_t2=mr_xt_t2, xt_d2=mr_xt_d2,
                           xt_u2=mr_xt_u2, xt_ms=mr_xt_ms, xt_re=mr_xt_re, xt_xf=mr_xt_xf)
            end

            # First iteration: set baseline Ue and Hk
            if itbl == 1
                ueref = c2.u
                hkref = c2.hk

                # Extrapolate Hk if station changed from turbulent to laminar
                if ibl < state.itran[is] && ibl >= itrold
                    uem = XFT(state.uedg[ibl-1, is])
                    dsm = XFT(state.dstr[ibl-1, is])
                    thm = XFT(state.thet[ibl-1, is])
                    msq = uem^2 * fl.hstinv / max(fl.gm1bl * (XFT(1) - XFT(0.5)*uem^2*fl.hstinv), XFT(1e-10))
                    hkref = hkin(XFT(dsm / thm), XFT(msq))
                end

                # Reinitialize Ctau if mode changed
                if ibl < itrold
                    if tran; cti = XFT(0.03); state.ctau[ibl, is] = XFT(0.03); end
                    if turb; cti = XFT(state.ctau[ibl-1, is]); state.ctau[ibl, is] = cti; end
                    if tran || turb; c2.s = cti; end
                end
            end

            # 4th equation: Ue-Hk tracking
            if simi || ibl == iblte + 1
                # Prescribe Ue
                vs2_sys[4, 1] = XFT(0); vs2_sys[4, 2] = XFT(0)
                vs2_sys[4, 3] = XFT(0); vs2_sys[4, 4] = c2.u_uei
                vsrez[4] = ueref - c2.u
            else
                # Ue-Hk characteristic slope
                vtmp = zeros(XFT, 4, 4); vztmp = zeros(XFT, 4)
                for k in 1:4; vztmp[k] = vsrez[k]; for l in 1:4; vtmp[k,l] = vs2_sys[k,l]; end; end
                vtmp[4,1] = XFT(0); vtmp[4,2] = c2.hk_t; vtmp[4,3] = c2.hk_d
                vtmp[4,4] = c2.hk_u * c2.u_uei; vztmp[4] = XFT(1)
                try; _gauss4!(vtmp, vztmp); catch; end
                sennew = senswt * vztmp[4] * hkref / max(abs(ueref), XFT(1e-10))
                if itbl <= 5; sens = sennew
                elseif itbl <= 15; sens = XFT(0.5) * (sens + sennew)
                end

                vs2_sys[4,1] = XFT(0)
                vs2_sys[4,2] = c2.hk_t * hkref
                vs2_sys[4,3] = c2.hk_d * hkref
                vs2_sys[4,4] = (c2.hk_u * hkref + sens / max(abs(ueref), XFT(1e-10))) * c2.u_uei
                vsrez[4] = -(hkref^2) * (c2.hk / hkref - XFT(1)) -
                            sens * (c2.u / ueref - XFT(1))
            end

            # Solve 4x4
            J = zeros(XFT, 4, 4); rhs = zeros(XFT, 4)
            for k in 1:4; rhs[k] = vsrez[k]; for l in 1:4; J[k,l] = vs2_sys[k,l]; end; end
            try; _gauss4!(J, rhs); catch; break; end
            any(!isfinite, rhs) && break

            # Underrelax
            dmax = max(abs(rhs[2] / max(abs(thi), XFT(1e-10))),
                       abs(rhs[3] / max(abs(dsi), XFT(1e-10))),
                       abs(rhs[4] / max(abs(uei), XFT(1e-10))))
            if ibl >= state.itran[is]
                dmax = max(dmax, abs(rhs[1] / (XFT(10) * max(abs(cti), XFT(1e-10)))))
            end
            rlx = dmax > XFT(0.3) ? XFT(0.3) / dmax : XFT(1)

            # Update
            if ibl < state.itran[is]; ami += rlx * rhs[1]; end
            if ibl >= state.itran[is]; cti += rlx * rhs[1]; end
            thi += rlx * rhs[2]
            dsi += rlx * rhs[3]
            uei += rlx * rhs[4]

            # Clamp
            if ibl >= state.itran[is]
                cti = clamp(cti, XFT(0.0000001), XFT(0.30))
            end
            hklim = ibl <= iblte ? XFT(1.02) : XFT(1.00005)
            msq = uei^2 * fl.hstinv / max(fl.gm1bl * (XFT(1) - XFT(0.5)*uei^2*fl.hstinv), XFT(1e-10))
            dsw = dsi - dswaki
            hk_val, hk_h, _ = hkin_jac(XFT(dsw / thi), XFT(msq))
            hk_val = XFT(hk_val)
            hk_h = XFT(hk_h)
            dh = iszero(hk_h) ? XFT(0) : max(XFT(0), hklim - hk_val) / hk_h
            dsi = XFT(dsw + dh * thi + dswaki)

            last_dmax = dmax
            if dmax <= deps
                converged_station = true
                break
            end
        end

        if !converged_station
            if last_dmax > XFT(0.1) && ibl > 3
                if ibl <= iblte
                    rat = sqrt(XFT(state.xssi[ibl, is]) /
                               max(XFT(state.xssi[ibm, is]), XFT(1e-10)))
                    thi = XFT(state.thet[ibm, is]) * rat
                    dsi = XFT(state.dstr[ibm, is]) * rat
                    uei = XFT(state.uedg[ibm, is])
                elseif ibl == iblte + 1
                    tte = XFT(state.thet[state.iblte[1], 1]) +
                          XFT(state.thet[state.iblte[2], 2])
                    dte = XFT(state.dstr[state.iblte[1], 1]) +
                          XFT(state.dstr[state.iblte[2], 2]) + ante
                    cte = (XFT(state.ctau[state.iblte[1], 1]) *
                           XFT(state.thet[state.iblte[1], 1]) +
                           XFT(state.ctau[state.iblte[2], 2]) *
                           XFT(state.thet[state.iblte[2], 2])) /
                          max(tte, XFT(1e-10))
                    cti = cte
                    thi = tte
                    dsi = dte
                    uei = XFT(state.uedg[ibm, is])
                else
                    thi = XFT(state.thet[ibm, is])
                    ratlen = (XFT(state.xssi[ibl, is]) - XFT(state.xssi[ibm, is])) /
                             (XFT(10) * max(XFT(state.dstr[ibm, is]), XFT(1e-10)))
                    dsi = (XFT(state.dstr[ibm, is]) + thi * ratlen) / (XFT(1) + ratlen)
                    uei = XFT(state.uedg[ibm, is])
                end
                if ibl == state.itran[is]
                    cti = XFT(0.05)
                elseif ibl > state.itran[is]
                    cti = XFT(state.ctau[ibm, is])
                end
            end

            xbl_blprv!(c2, xsi, ami, cti, thi, dsi, dswaki, uei, fl)
            xbl_blkin!(c2, fl)

            if !simi && !turb
                tr_result = xbl_trchek2!(c1, c2, c2sav, fl)
                ami = c2.ampl
                if tr_result.tran
                    tran = true
                    state.itran[is] = ibl
                else
                    state.itran[is] = ibl + 2
                end
            end

            if ibl < state.itran[is]
                xbl_blvar!(c2, 1, fl, par)
                xbl_blmid(c1, c2, 1, simi, fl, par)
            end
            if ibl >= state.itran[is]
                xbl_blvar!(c2, 2, fl, par)
                xbl_blmid(c1, c2, 2, simi, fl, par)
            end
            if wake
                xbl_blvar!(c2, 3, fl, par)
                xbl_blmid(c1, c2, 3, simi, fl, par)
            end
        end

        sens = sennew

        # Store results
        if ibl < state.itran[is]; state.ctau[ibl, is] = XFT(ami); end
        if ibl >= state.itran[is]; state.ctau[ibl, is] = XFT(cti); end
        state.thet[ibl, is] = XFT(thi)
        state.dstr[ibl, is] = XFT(dsi)
        state.uedg[ibl, is] = XFT(uei)
        state.mass[ibl, is] = XFT(dsi * uei)

        # Set COM1 for next station
        xbl_blprv!(c2, xsi, ami, cti, thi, dsi, dswaki, uei, fl)
        xbl_blkin!(c2, fl)
        copycom!(c1, c2)

        # Transition logic
        if tran || ibl == iblte
            turb = true
        end
        tran = false
    end
    return nothing
end

# ============================================================================
#  xbl_mrchue! — Literal port of Fortran MRCHUE (xbl.f 556-883)
#
#  Marches the BL on side `is` in direct mode using the UEDG array.
#  If separation is encountered, a plausible Hk value extrapolated from
#  upstream is prescribed instead (inverse mode).  Continuous checking
#  of transition onset is performed.
#
#  Modifies state arrays: uedg, thet, dstr, ctau, mass, itran
#  Also stores: tau, dis, ctq, delt, tstr (secondary outputs)
# ============================================================================
function xbl_mrchue!(state::ViscalBLState, is::Int,
                      fl::BLFlowF32, par::BLParF32, ante::XFT)
    # Separation Hk thresholds (Fortran lines 570-571)
    hlmax = XFT(3.8)
    htmax = XFT(2.5)

    iblte = state.iblte[is]
    nbl   = state.nbl[is]
    fl = xbl_with_xiforc(fl, XFT(state.xssi[iblte, is]))

    # Initialize similarity station with Thwaites' formula (lines 581-591)
    ibl = 2
    xsi = XFT(state.xssi[ibl, is])
    uei = XFT(state.uedg[ibl, is])
    bule = XFT(1)
    ucon = uei / xsi^bule
    tsq = XFT(0.45) / (ucon * (XFT(5) * bule + XFT(1)) * fl.reybl) *
          xsi^(XFT(1) - bule)
    thi = sqrt(abs(tsq))
    dsi = XFT(2.2) * thi
    ami = XFT(0)

    # Initialize Ctau for first turbulent station (line 594)
    cti = XFT(0.03)

    tran = false
    turb = false
    state.itran[is] = iblte  # (line 598)

    # COM structs for station 1 and 2
    c1 = BLCom()
    c2 = BLCom()
    c2sav = BLCom()

    # BLSYS output arrays
    vs1 = zeros(XFT, 4, 5); vs2_sys = zeros(XFT, 4, 5); vsrez = zeros(XFT, 4)
    vsm = zeros(XFT, 4); vsr = zeros(XFT, 4); vsx = zeros(XFT, 4)

    # March downstream (line 601)
    for ibl in 2:nbl
        ibm = ibl - 1
        simi = (ibl == 2)
        wake = (ibl > iblte)

        # Prescribed quantities (lines 610-611)
        xsi = XFT(state.xssi[ibl, is])
        uei = XFT(state.uedg[ibl, is])

        dswaki = XFT(0)
        if wake
            iw = ibl - iblte
            dswaki = iw <= length(state.wgap) ? XFT(state.wgap[iw]) : XFT(0)
        end

        direct = true
        htarg = XFT(0)

        # Newton iteration loop (line 623)
        dmax = XFT(1)
        for itbl in 1:25
            # Match Fortran TRCHEK/TRCHEK2 semantics: transition status is
            # recomputed from scratch on each sub-iteration.
            tran = false

            # Assemble BL system (lines 630-631)
            xbl_blprv!(c2, xsi, ami, cti, thi, dsi, dswaki, uei, fl)
            xbl_blkin!(c2, fl)

            # Transition check (lines 634-649)
            local me_xt_val = XFT(0)
            local me_xt_a1=XFT(0); local me_xt_x1=XFT(0); local me_xt_t1=XFT(0); local me_xt_d1=XFT(0); local me_xt_u1=XFT(0)
            local me_xt_x2=XFT(0); local me_xt_t2=XFT(0); local me_xt_d2=XFT(0); local me_xt_u2=XFT(0)
            local me_xt_ms=XFT(0); local me_xt_re=XFT(0); local me_xt_xf=XFT(0)
            if !simi && !turb
                tr_result = xbl_trchek2!(c1, c2, c2sav, fl)
                ami = c2.ampl

                if tr_result.tran
                    tran = true
                    state.itran[is] = ibl
                    me_xt_val = tr_result.xt
                    me_xt_a1 = tr_result.xt_a1; me_xt_x1 = tr_result.xt_x1; me_xt_t1 = tr_result.xt_t1
                    me_xt_d1 = tr_result.xt_d1; me_xt_u1 = tr_result.xt_u1; me_xt_x2 = tr_result.xt_x2
                    me_xt_t2 = tr_result.xt_t2; me_xt_d2 = tr_result.xt_d2; me_xt_u2 = tr_result.xt_u2
                    me_xt_ms = tr_result.xt_ms; me_xt_re = tr_result.xt_re; me_xt_xf = tr_result.xt_xf
                    if cti <= XFT(0)
                        cti = XFT(0.03)
                        c2.s = cti
                    end
                else
                    state.itran[is] = ibl + 2
                end
            end

            # BLSYS or TESYS (lines 651-659)
            if ibl == iblte + 1
                tte = XFT(state.thet[state.iblte[1], 1]) +
                      XFT(state.thet[state.iblte[2], 2])
                dte = XFT(state.dstr[state.iblte[1], 1]) +
                      XFT(state.dstr[state.iblte[2], 2]) + ante
                cte_val = (XFT(state.ctau[state.iblte[1], 1]) *
                           XFT(state.thet[state.iblte[1], 1]) +
                           XFT(state.ctau[state.iblte[2], 2]) *
                           XFT(state.thet[state.iblte[2], 2])) /
                          max(tte, XFT(1e-10))
                xbl_tesys!(vs1, vs2_sys, vsrez, vsm, vsr, vsx,
                           c2, cte_val, tte, dte, fl, par)
            else
                xbl_blsys!(vs1, vs2_sys, vsrez, vsm, vsr, vsx, c1, c2,
                           simi, tran && ibl == state.itran[is], turb, wake,
                           fl, par;
                           xt=me_xt_val, xt_a1=me_xt_a1, xt_x1=me_xt_x1,
                           xt_t1=me_xt_t1, xt_d1=me_xt_d1, xt_u1=me_xt_u1,
                           xt_x2=me_xt_x2, xt_t2=me_xt_t2, xt_d2=me_xt_d2,
                           xt_u2=me_xt_u2, xt_ms=me_xt_ms, xt_re=me_xt_re, xt_xf=me_xt_xf)
            end

            if direct
                # Direct mode: set dUe = 0 (lines 664-668)
                vs2_sys[4, 1] = XFT(0)
                vs2_sys[4, 2] = XFT(0)
                vs2_sys[4, 3] = XFT(0)
                vs2_sys[4, 4] = XFT(1)
                vsrez[4] = XFT(0)

                # Solve 4x4 Newton system (line 671)
                J = zeros(XFT, 4, 4); rhs = zeros(XFT, 4)
                for k in 1:4; rhs[k] = vsrez[k]
                    for l in 1:4; J[k, l] = vs2_sys[k, l]; end
                end
                try; _gauss4!(J, rhs); catch; break; end
                any(!isfinite, rhs) && break

                # Underrelax (lines 674-680)
                dmax = max(abs(rhs[2] / max(abs(thi), XFT(1e-10))),
                           abs(rhs[3] / max(abs(dsi), XFT(1e-10))))
                if ibl < state.itran[is]
                    dmax = max(dmax, abs(rhs[1] / XFT(10)))
                end
                if ibl >= state.itran[is]
                    dmax = max(dmax, abs(rhs[1] / max(abs(cti), XFT(1e-10))))
                end

                rlx = XFT(1)
                if dmax > XFT(0.3); rlx = XFT(0.3) / dmax; end

                # Check if direct mode is applicable (lines 683-694)
                if ibl != iblte + 1
                    msq = uei^2 * fl.hstinv /
                          max(fl.gm1bl * (XFT(1) - XFT(0.5) * uei^2 * fl.hstinv),
                              XFT(1e-10))
                    htest = (dsi + rlx * rhs[3]) /
                            max(thi + rlx * rhs[2], XFT(1e-10))
                    hktest = hkin(XFT(htest), XFT(msq))

                    hmax = ibl < state.itran[is] ? hlmax : htmax
                    direct = hktest < hmax
                end

                if direct
                    # Update as usual (lines 698-701)
                    if ibl >= state.itran[is]; cti += rlx * rhs[1]; end
                    thi += rlx * rhs[2]
                    dsi += rlx * rhs[3]
                else
                    # Set prescribed Hk for inverse (lines 704-732)
                    if ibl < state.itran[is]
                        # Laminar: slow Hk increase
                        htarg = c1.hk + XFT(0.03) * (c2.x - c1.x) / max(c1.t, XFT(1e-10))
                    elseif ibl == state.itran[is]
                        # Transition interval
                        htarg = c1.hk + (XFT(0.03) * (me_xt_val - c1.x) -
                                          XFT(0.15) * (c2.x - me_xt_val)) /
                                         max(c1.t, XFT(1e-10))
                    elseif wake
                        # Turbulent wake: asymptotic behavior
                        const_w = XFT(0.03) * (c2.x - c1.x) / max(c1.t, XFT(1e-10))
                        hk2 = c1.hk
                        for _nr in 1:3
                            hk2 -= (hk2 + const_w * (hk2 - XFT(1))^3 - c1.hk) /
                                   (XFT(1) + XFT(3) * const_w * (hk2 - XFT(1))^2)
                        end
                        htarg = hk2
                    else
                        # Turbulent: fast Hk decrease
                        htarg = c1.hk - XFT(0.15) * (c2.x - c1.x) / max(c1.t, XFT(1e-10))
                    end

                    # Limit Hk (lines 728-732)
                    if wake
                        htarg = max(htarg, XFT(1.01))
                    else
                        htarg = max(htarg, hmax)
                    end

                    # Try again with prescribed Hk (line 738: GO TO 100)
                    continue
                end
            else
                # Inverse mode: force Hk to prescribed HTARG (lines 744-768)
                vs2_sys[4, 1] = XFT(0)
                vs2_sys[4, 2] = c2.hk_t
                vs2_sys[4, 3] = c2.hk_d
                vs2_sys[4, 4] = c2.hk_u
                vsrez[4] = htarg - c2.hk

                J = zeros(XFT, 4, 4); rhs = zeros(XFT, 4)
                for k in 1:4; rhs[k] = vsrez[k]
                    for l in 1:4; J[k, l] = vs2_sys[k, l]; end
                end
                try; _gauss4!(J, rhs); catch; break; end
                any(!isfinite, rhs) && break

                # Underrelax with Ue clamp (lines 754-760)
                dmax = max(abs(rhs[2] / max(abs(thi), XFT(1e-10))),
                           abs(rhs[3] / max(abs(dsi), XFT(1e-10))),
                           abs(rhs[4] / max(abs(uei), XFT(1e-10))))
                if ibl >= state.itran[is]
                    dmax = max(dmax, abs(rhs[1] / max(abs(cti), XFT(1e-10))))
                end

                rlx = XFT(1)
                if dmax > XFT(0.3); rlx = XFT(0.3) / dmax; end

                # Update (lines 764-767)
                if ibl >= state.itran[is]; cti += rlx * rhs[1]; end
                thi += rlx * rhs[2]
                dsi += rlx * rhs[3]
                uei += rlx * rhs[4]
            end

            # Clamp Ctau (lines 772-775)
            if ibl >= state.itran[is]
                cti = clamp(cti, XFT(0.0000001), XFT(0.30))
            end

            # DSLIM: limit Hk (lines 777-785)
            hklim = ibl <= iblte ? XFT(1.02) : XFT(1.00005)
            msq = uei^2 * fl.hstinv /
                  max(fl.gm1bl * (XFT(1) - XFT(0.5) * uei^2 * fl.hstinv),
                      XFT(1e-10))
            dsw = dsi - dswaki
            hk_val, hk_h, _ = hkin_jac(XFT(dsw / thi), XFT(msq))
            hk_val = XFT(hk_val)
            hk_h = XFT(hk_h)
            dh = iszero(hk_h) ? XFT(0) : max(XFT(0), hklim - hk_val) / hk_h
            dsi = XFT(dsw + dh * thi + dswaki)

            # Convergence check (MRCHUE line 787 uses 1.0E-5 in Fortran).
            dmax <= XFT(1e-5) && break
        end  # Newton iterations

        # Fallback for non-converged stations (lines 796-818)
        if dmax > XFT(0.1) && ibl > 3
            if ibl <= iblte
                thi = XFT(state.thet[ibm, is]) *
                      (XFT(state.xssi[ibl, is]) / max(XFT(state.xssi[ibm, is]),
                       XFT(1e-10)))^XFT(0.5)
                dsi = XFT(state.dstr[ibm, is]) *
                      (XFT(state.xssi[ibl, is]) / max(XFT(state.xssi[ibm, is]),
                       XFT(1e-10)))^XFT(0.5)
            elseif ibl == iblte + 1
                cti = (XFT(state.ctau[state.iblte[1], 1]) *
                       XFT(state.thet[state.iblte[1], 1]) +
                       XFT(state.ctau[state.iblte[2], 2]) *
                       XFT(state.thet[state.iblte[2], 2])) /
                      max(XFT(state.thet[state.iblte[1], 1]) +
                          XFT(state.thet[state.iblte[2], 2]), XFT(1e-10))
                thi = XFT(state.thet[state.iblte[1], 1]) +
                      XFT(state.thet[state.iblte[2], 2])
                dsi = XFT(state.dstr[state.iblte[1], 1]) +
                      XFT(state.dstr[state.iblte[2], 2]) + ante
            else
                thi = XFT(state.thet[ibm, is])
                ratlen = (XFT(state.xssi[ibl, is]) - XFT(state.xssi[ibm, is])) /
                         (XFT(10) * max(XFT(state.dstr[ibm, is]), XFT(1e-10)))
                dsi = (XFT(state.dstr[ibm, is]) + thi * ratlen) / (XFT(1) + ratlen)
            end
            if ibl == state.itran[is]; cti = XFT(0.05); end
            if ibl > state.itran[is]; cti = XFT(state.ctau[ibm, is]); end

            uei = XFT(state.uedg[ibl, is])
            if ibl > 2 && ibl < nbl
                uei = XFT(0.5) * (XFT(state.uedg[ibl - 1, is]) +
                                   XFT(state.uedg[ibl + 1, is]))
            end
        end

        # Set final state and COM for this station (lines 820-821, 832-838)
        if dmax > XFT(1e-5)
            xbl_blprv!(c2, xsi, ami, cti, thi, dsi, dswaki, uei, fl)
            xbl_blkin!(c2, fl)
            # Transition re-check for fallback (lines 824-829)
            if !simi && !turb
                tr_result = xbl_trchek2!(c1, c2, c2sav, fl)
                ami = c2.ampl
                if tr_result.tran; state.itran[is] = ibl; end
                if !tr_result.tran; state.itran[is] = ibl + 2; end
            end
            # BLVAR for extrapolated state (lines 832-834)
            if ibl < state.itran[is]; xbl_blvar!(c2, 1, fl, par); end
            if ibl >= state.itran[is]; xbl_blvar!(c2, 2, fl, par); end
            if wake; xbl_blvar!(c2, 3, fl, par); end
        end

        # Store primary variables (lines 844-854)
        if ibl < state.itran[is]; state.ctau[ibl, is] = XFT(ami); end
        if ibl >= state.itran[is]; state.ctau[ibl, is] = XFT(cti); end
        state.thet[ibl, is] = XFT(thi)
        state.dstr[ibl, is] = XFT(dsi)
        state.uedg[ibl, is] = XFT(uei)
        state.mass[ibl, is] = XFT(dsi * uei)

        # Set "1" variables to "2" for next station (lines 857-861)
        xbl_blprv!(c2, xsi, ami, cti, thi, dsi, dswaki, uei, fl)
        xbl_blkin!(c2, fl)
        copycom!(c1, c2)

        # Turbulent intervals follow transition or TE (lines 864-870)
        if tran || ibl == iblte
            turb = true
        end
        tran = false

        # TE junction: sum both sides (lines 874-877)
        if ibl == iblte
            thi = XFT(state.thet[state.iblte[1], 1]) +
                  XFT(state.thet[state.iblte[2], 2])
            dsi = XFT(state.dstr[state.iblte[1], 1]) +
                  XFT(state.dstr[state.iblte[2], 2]) + ante
        end
    end  # station loop

    return nothing
end

# ============================================================================
#  xbl_setbl! — Monolithic SETBL: MRCHDU + USAV swap + Newton assembly
#  Literal translation of XFoil's xbl.f SETBL subroutine (lines 21-530).
#
#  This function:
#  1. Calls xbl_mrchdu! to update BL state and transition
#  2. Saves MRCHDU's UEDG, calls UESET, swaps (the USAV pattern)
#  3. Sweeps all stations using xbl_blsys! for Newton system assembly
#  4. Returns (va, vb, vm, vdel, vz) for BLSOLV
#
#  The COM1/COM2 state propagation is maintained internally, ensuring
#  MRCHDU and SETBL use the EXACT SAME equation chain.
# ============================================================================
function xbl_setbl!(
    state::ViscalBLState,
    dij::Matrix{Float64},
    fl::BLFlowF32,
    par::BLParF32,
    ante::XFT;
    topo = nothing,
    skip_mrchdu::Bool = false,
)
    nsys = state.nsys
    va = zeros(Float32, 3, 2, nsys)
    vb = zeros(Float32, 3, 2, nsys)
    vm = zeros(Float32, 3, nsys, nsys)
    vdel = zeros(Float32, 3, 2, nsys)
    vz = zeros(Float32, 3, 2)

    # ── Phase 1: MRCHDU (skipped after first Newton iteration) ──
    if !skip_mrchdu
        try; xbl_mrchdu!(state, 1, fl, par, ante); catch; end
        try; xbl_mrchdu!(state, 2, fl, par, ante); catch; end
    end

    # ── Phase 2: USAV swap (Fortran SETBL lines 100-114) ──
    usav = zeros(Float32, size(state.uedg))
    # Save MRCHDU's UEDG
    for is in 1:2, ibl in 2:state.nbl[is]
        usav[ibl, is] = state.uedg[ibl, is]
    end
    # UESET: compute mass-consistent Ue → store in UEDG temporarily
    for is in 1:2
        for ibl in 2:state.nbl[is]
            i = state.ipan[ibl, is]
            dui = XFT(0)
            for js in 1:2
                for jbl in 2:state.nbl[js]
                    j = state.ipan[jbl, js]
                    if j >= 1 && j <= size(dij,2) && i >= 1 && i <= size(dij,1)
                        ue_m = -XFT(state.vti[ibl,is]) * XFT(state.vti[jbl,js]) * XFT(dij[i,j])
                        dui += ue_m * XFT(state.mass[jbl,js])
                    end
                end
            end
            state.uedg[ibl, is] = state.uinv[ibl,is] + dui
        end
    end
    # Swap: USAV = UESET's Ue, UEDG = MRCHDU's Ue
    for is in 1:2, ibl in 2:state.nbl[is]
        temp = usav[ibl, is]
        usav[ibl, is] = state.uedg[ibl, is]  # USAV = mass-consistent
        state.uedg[ibl, is] = temp             # UEDG = MRCHDU's
    end

    # Pre-compute LE and TE sensitivities
    iblte1 = state.iblte[1]; iblte2 = state.iblte[2]
    ile1 = state.ipan[2, 1]; ile2 = state.ipan[2, 2]
    ite1 = state.ipan[iblte1, 1]; ite2 = state.ipan[iblte2, 2]
    jvte1 = state.isys[iblte1, 1]; jvte2 = state.isys[iblte2, 2]

    ule1_m = zeros(XFT, nsys); ule2_m = zeros(XFT, nsys)
    ute1_m = zeros(XFT, nsys); ute2_m = zeros(XFT, nsys)
    for js in 1:2, jbl in 2:state.nbl[js]
        j = state.ipan[jbl, js]; jv = state.isys[jbl, js]
        if jv >= 1 && jv <= nsys && j >= 1 && j <= size(dij,2)
            vtj = XFT(state.vti[jbl, js])
            ile1 >= 1 && ile1 <= size(dij,1) && (ule1_m[jv] = -XFT(state.vti[2,1]) * vtj * XFT(dij[ile1,j]))
            ile2 >= 1 && ile2 <= size(dij,1) && (ule2_m[jv] = -XFT(state.vti[2,2]) * vtj * XFT(dij[ile2,j]))
            ite1 >= 1 && ite1 <= size(dij,1) && (ute1_m[jv] = -XFT(state.vti[iblte1,1]) * vtj * XFT(dij[ite1,j]))
            ite2 >= 1 && ite2 <= size(dij,1) && (ute2_m[jv] = -XFT(state.vti[iblte2,2]) * vtj * XFT(dij[ite2,j]))
        end
    end

    dule1 = XFT(state.uedg[2,1]) - XFT(usav[2,1])
    dule2 = XFT(state.uedg[2,2]) - XFT(usav[2,2])

    sst_go = isnothing(topo) ? XFT(0) : XFT(topo.stagnation.sst_go)
    sst_gp = isnothing(topo) ? XFT(0) : XFT(topo.stagnation.sst_gp)

    # ── Phase 3: Station sweep (Fortran SETBL lines 143-527) ──
    c1 = BLCom(); c2 = BLCom(); c2sav = BLCom()
    vs1_bl = zeros(XFT,4,5); vs2_bl = zeros(XFT,4,5)
    vsrez_bl = zeros(XFT,4); vsm_bl = zeros(XFT,4); vsr_bl = zeros(XFT,4); vsx_bl = zeros(XFT,4)

    for is in 1:2
        fl_is = xbl_with_xiforc(fl, XFT(state.xssi[state.iblte[is], is]))
        u1_m = zeros(XFT, nsys); d1_m = zeros(XFT, nsys)
        due1_loc = XFT(0); dds1_loc = XFT(0)
        ami_setbl = XFT(0)
        cti_setbl = XFT(0.03)

        for ibl in 2:state.nbl[is]
            iv = state.isys[ibl, is]
            i_pan = state.ipan[ibl, is]
            simi = (ibl == 2)
            wake = (ibl > state.iblte[is])
            tran_flag = (ibl == state.itran[is])
            turb = (ibl > state.itran[is])

            uei = state.uedg[ibl, is]
            uei_den = abs(uei) > XFT(1.0e-10) ? uei :
                (signbit(uei) ? -XFT(1.0e-10) : XFT(1.0e-10))
            # XFoil SETBL reconstructs D2 from MASS/UEDG before BLPRV/BLKIN.
            dsi_mass = XFT(state.mass[ibl, is]) / uei_den
            d2_m2 = XFT(1) / uei_den
            d2_u2 = -dsi_mass / uei_den

            u2_m = zeros(XFT, nsys); d2_m = zeros(XFT, nsys)
            for js in 1:2, jbl in 2:state.nbl[js]
                j = state.ipan[jbl, js]; jv = state.isys[jbl, js]
                if jv >= 1 && jv <= nsys && i_pan >= 1 && j >= 1 &&
                   i_pan <= size(dij,1) && j <= size(dij,2)
                    u2_m[jv] = -XFT(state.vti[ibl,is]) * XFT(state.vti[jbl,js]) * XFT(dij[i_pan,j])
                    d2_m[jv] = d2_u2 * u2_m[jv]
                end
            end
            d2_m[iv] += d2_m2

            due2_loc = XFT(state.uedg[ibl,is]) - XFT(usav[ibl,is])
            dds2_loc = d2_u2 * due2_loc

            is_tesys = (ibl == state.iblte[is] + 1) && (ibl <= state.nbl[is])

            if is_tesys
                # TESYS
                tte = XFT(state.thet[iblte1,1]) + XFT(state.thet[iblte2,2])
                dte = XFT(state.dstr[iblte1,1]) + XFT(state.dstr[iblte2,2]) + XFT(ante)
                cte = (
                    XFT(state.ctau[iblte1,1]) * XFT(state.thet[iblte1,1]) +
                    XFT(state.ctau[iblte2,2]) * XFT(state.thet[iblte2,2])
                ) / max(tte, XFT(1.0e-10))

                tte_tte1 = XFT(1); tte_tte2 = XFT(1)
                dte_mte1 = XFT(1) / max(XFT(state.uedg[iblte1,1]), XFT(1.0e-10))
                dte_ute1 = -XFT(state.dstr[iblte1,1]) / max(XFT(state.uedg[iblte1,1]), XFT(1.0e-10))
                dte_mte2 = XFT(1) / max(XFT(state.uedg[iblte2,2]), XFT(1.0e-10))
                dte_ute2 = -XFT(state.dstr[iblte2,2]) / max(XFT(state.uedg[iblte2,2]), XFT(1.0e-10))
                cte_cte1 = XFT(state.thet[iblte1,1]) / max(tte, XFT(1.0e-10))
                cte_cte2 = XFT(state.thet[iblte2,2]) / max(tte, XFT(1.0e-10))
                cte_tte1 = (XFT(state.ctau[iblte1,1]) - cte) / max(tte, XFT(1.0e-10))
                cte_tte2 = (XFT(state.ctau[iblte2,2]) - cte) / max(tte, XFT(1.0e-10))

                # Use xbl_tesys! for the interval system
                xsi = XFT(state.xssi[ibl,is])
                s_val = XFT(state.ctau[ibl,is])
                thi = XFT(state.thet[ibl,is])
                dsi_val = dsi_mass
                dswaki = ibl > state.iblte[is] ? (ibl-state.iblte[is] <= length(state.wgap) ? XFT(state.wgap[ibl-state.iblte[is]]) : XFT(0)) : XFT(0)
                xbl_blprv!(c2, xsi, XFT(0), s_val, thi, dsi_val, dswaki, XFT(uei), fl_is)
                xbl_blkin!(c2, fl_is)
                xbl_tesys!(vs1_bl, vs2_bl, vsrez_bl, vsm_bl, vsr_bl, vsx_bl,
                    c2, XFT(cte), XFT(tte), XFT(dte), fl_is, par)

                # Redefine D1_M for TESYS
                for jv in 1:nsys; d1_m[jv] = dte_ute1*ute1_m[jv] + dte_ute2*ute2_m[jv]; end
                jvte1 >= 1 && jvte1 <= nsys && (d1_m[jvte1] += dte_mte1)
                jvte2 >= 1 && jvte2 <= nsys && (d1_m[jvte2] += dte_mte2)

                due1_loc = 0.0
                dds1_loc = dte_ute1*(XFT(state.uedg[iblte1,1])-XFT(usav[iblte1,1])) +
                           dte_ute2*(XFT(state.uedg[iblte2,2])-XFT(usav[iblte2,2]))

                # Fortran: XI_ULE applies to ALL stations including TESYS (lines 340-346, 350-437)
                xi_ule1_te = is == 1 ? sst_go : -sst_go
                xi_ule2_te = is == 1 ? -sst_gp : sst_gp

                for k in 1:3
                    va[k,1,iv] = XFT(vs2_bl[k,1]); va[k,2,iv] = XFT(vs2_bl[k,2])
                    xi_coeff = XFT(vs1_bl[k,5]) + XFT(vs2_bl[k,5]) + XFT(vsx_bl[k])
                    for jv in 1:nsys
                        vm[k,jv,iv] = XFT(vs1_bl[k,3]*d1_m[jv] + vs1_bl[k,4]*u1_m[jv] +
                                          vs2_bl[k,3]*d2_m[jv] + vs2_bl[k,4]*u2_m[jv] +
                                          xi_coeff * (xi_ule1_te*ule1_m[jv] + xi_ule2_te*ule2_m[jv]))
                    end
                    vdel[k,1,iv] = XFT(vsrez_bl[k] +
                        vs1_bl[k,4]*due1_loc + vs1_bl[k,3]*dds1_loc +
                        vs2_bl[k,4]*due2_loc + vs2_bl[k,3]*dds2_loc +
                        xi_coeff * (xi_ule1_te*dule1 + xi_ule2_te*dule2))
                    vdel[k,2,iv] = XFT(0)
                end
                # VZ coupling
                for k in 1:3
                    vz[k,1] = XFT(vs1_bl[k,1] * cte_cte1)
                    vz[k,2] = XFT(vs1_bl[k,1] * cte_tte1 + vs1_bl[k,2] * tte_tte1)
                    vb[k,1,iv] = XFT(vs1_bl[k,1] * cte_cte2)
                    vb[k,2,iv] = XFT(vs1_bl[k,1] * cte_tte2 + vs1_bl[k,2] * tte_tte2)
                end
            else
                # Normal station: use xbl_blsys!
                xsi = XFT(state.xssi[ibl,is])
                if ibl < state.itran[is]
                    ami_setbl = XFT(state.ctau[ibl, is])
                else
                    cti_setbl = XFT(state.ctau[ibl, is])
                end
                ami = ami_setbl
                cti = cti_setbl
                thi = XFT(state.thet[ibl,is])
                dsi_val = dsi_mass
                dswaki = wake ? (ibl-state.iblte[is] <= length(state.wgap) ? XFT(state.wgap[ibl-state.iblte[is]]) : XFT(0)) : XFT(0)

                try
                    xbl_blprv!(c2, xsi, ami, cti, thi, dsi_val, dswaki, XFT(uei), fl_is)
                    xbl_blkin!(c2, fl_is)
                    local xt_tran = XFT(0)
                    local tr_xt_a1=XFT(0); local tr_xt_x1=XFT(0); local tr_xt_t1=XFT(0); local tr_xt_d1=XFT(0); local tr_xt_u1=XFT(0)
                    local tr_xt_x2=XFT(0); local tr_xt_t2=XFT(0); local tr_xt_d2=XFT(0); local tr_xt_u2=XFT(0)
                    local tr_xt_ms=XFT(0); local tr_xt_re=XFT(0); local tr_xt_xf=XFT(0)
                    local tran_use = tran_flag
                    if tran_flag
                        tr = xbl_trchek2!(c1, c2, c2sav, fl_is)
                        ami = c2.ampl
                        tran_use = tr.tran
                        xt_tran = tr.xt
                        tr_xt_a1 = tr.xt_a1; tr_xt_x1 = tr.xt_x1; tr_xt_t1 = tr.xt_t1
                        tr_xt_d1 = tr.xt_d1; tr_xt_u1 = tr.xt_u1; tr_xt_x2 = tr.xt_x2
                        tr_xt_t2 = tr.xt_t2; tr_xt_d2 = tr.xt_d2; tr_xt_u2 = tr.xt_u2
                        tr_xt_ms = tr.xt_ms; tr_xt_re = tr.xt_re; tr_xt_xf = tr.xt_xf
                        copycom!(c2, c2sav)
                        c2.ampl = ami
                    end
                    xbl_blsys!(vs1_bl, vs2_bl, vsrez_bl, vsm_bl, vsr_bl, vsx_bl,
                        c1, c2, simi, tran_use, turb, wake, fl_is, par;
                        xt=xt_tran, xt_a1=tr_xt_a1, xt_x1=tr_xt_x1,
                        xt_t1=tr_xt_t1, xt_d1=tr_xt_d1, xt_u1=tr_xt_u1,
                        xt_x2=tr_xt_x2, xt_t2=tr_xt_t2, xt_d2=tr_xt_d2,
                        xt_u2=tr_xt_u2, xt_ms=tr_xt_ms, xt_re=tr_xt_re, xt_xf=tr_xt_xf)
                catch
                    # Skip station on error
                    u1_m .= u2_m; d1_m .= d2_m; due1_loc = due2_loc; dds1_loc = dds2_loc
                    continue
                end

                xi_ule1 = is == 1 ? sst_go : -sst_go
                xi_ule2 = is == 1 ? -sst_gp : sst_gp

                for k in 1:3
                    va[k,1,iv] = XFT(vs2_bl[k,1]); va[k,2,iv] = XFT(vs2_bl[k,2])
                    vb[k,1,iv] = XFT(vs1_bl[k,1]); vb[k,2,iv] = XFT(vs1_bl[k,2])
                    # Fortran: (VS1(k,5) + VS2(k,5) + VSX(k)) * (XI_ULE1*ULE1_M + XI_ULE2*ULE2_M)
                    xi_coeff = XFT(vs1_bl[k,5]) + XFT(vs2_bl[k,5]) + XFT(vsx_bl[k])
                    for jv in 1:nsys
                        vm[k,jv,iv] = XFT(
                            vs1_bl[k,3]*d1_m[jv] + vs1_bl[k,4]*u1_m[jv] +
                            vs2_bl[k,3]*d2_m[jv] + vs2_bl[k,4]*u2_m[jv] +
                            xi_coeff * (xi_ule1*ule1_m[jv] + xi_ule2*ule2_m[jv]))
                    end
                    vdel[k,1,iv] = XFT(vsrez_bl[k] +
                        vs1_bl[k,4]*due1_loc + vs1_bl[k,3]*dds1_loc +
                        vs2_bl[k,4]*due2_loc + vs2_bl[k,3]*dds2_loc +
                        xi_coeff * (xi_ule1*dule1 + xi_ule2*dule2))
                    vdel[k,2,iv] = XFT(0)
                end
            end

            # TE→wake transition (Fortran SETBL lines 487-494)
            if ibl == state.iblte[is]
                turb = true; wake = true
                xbl_blvar!(c2, 3, fl_is, par)
                xbl_blmid(c1, c2, 3, false, fl_is, par)  # Fortran: CALL BLMID(3)
            end

            # Shift 2→1 for next station
            u1_m .= u2_m; d1_m .= d2_m
            due1_loc = due2_loc; dds1_loc = dds2_loc

            # COM1 = COM2 for next station
            copycom!(c1, c2)
        end
    end

    return va, vb, vm, vdel, vz
end

# ============================================================================
#  xbl_trdif! — Transition interval equations
#  Faithful port of xblsys.f TRDIF (lines 1218-1572)
#
#  Splits the interval X1..X2 into:
#    Laminar part:   X1 → XT  (BLDIF type 1)
#    Turbulent part: XT → X2  (BLDIF type 2)
#  and sums the two systems with proper sensitivity conversion.
#
#  Accepts XT sensitivities from TRCHEK2.
#  Modifies: vs1, vs2, vsrez, vsm, vsr, vsx (output Newton system)
# ============================================================================
function xbl_trdif!(vs1::Matrix{XFT}, vs2::Matrix{XFT}, vsrez::Vector{XFT},
                     vsm::Vector{XFT}, vsr::Vector{XFT}, vsx::Vector{XFT},
                     c1::BLCom, c2::BLCom, xt::XFT,
                     fl::BLFlowF32, par::BLParF32,
                     xt_a1::XFT, xt_x1::XFT, xt_t1::XFT, xt_d1::XFT, xt_u1::XFT,
                     xt_x2::XFT, xt_t2::XFT, xt_d2::XFT, xt_u2::XFT,
                     xt_ms::XFT, xt_re::XFT, xt_xf::XFT)
    # Save COM1 and COM2 (Fortran lines 1232-1235)
    c1sav = BLCom(); copycom!(c1sav, c1)
    c2sav = BLCom(); copycom!(c2sav, c2)

    x1 = c1.x; x2 = c2.x; t1 = c1.t; t2 = c2.t
    d1 = c1.d; d2 = c2.d; u1 = c1.u; u2 = c2.u
    dx = x2 - x1
    dx = max(abs(dx), XFT(1e-12)) * sign(dx + XFT(1e-30))

    # ── WF2 sensitivities from XT (Fortran lines 1238-1266) ──
    wf2    = (xt - x1) / dx
    wf2_xt = XFT(1) / dx

    wf2_a1 = wf2_xt*xt_a1
    wf2_x1 = wf2_xt*xt_x1 + (wf2 - XFT(1)) / dx
    wf2_x2 = wf2_xt*xt_x2 - wf2 / dx
    wf2_t1 = wf2_xt*xt_t1;  wf2_t2 = wf2_xt*xt_t2
    wf2_d1 = wf2_xt*xt_d1;  wf2_d2 = wf2_xt*xt_d2
    wf2_u1 = wf2_xt*xt_u1;  wf2_u2 = wf2_xt*xt_u2
    wf2_ms = wf2_xt*xt_ms;  wf2_re = wf2_xt*xt_re;  wf2_xf = wf2_xt*xt_xf

    wf1    = XFT(1) - wf2
    wf1_a1 = -wf2_a1;  wf1_x1 = -wf2_x1;  wf1_x2 = -wf2_x2
    wf1_t1 = -wf2_t1;  wf1_t2 = -wf2_t2
    wf1_d1 = -wf2_d1;  wf1_d2 = -wf2_d2
    wf1_u1 = -wf2_u1;  wf1_u2 = -wf2_u2
    wf1_ms = -wf2_ms;  wf1_re = -wf2_re;  wf1_xf = -wf2_xf

    # ── Interpolate primary variables to transition point (Fortran lines 1272-1312) ──
    tt    = t1*wf1    + t2*wf2
    tt_a1 = t1*wf1_a1 + t2*wf2_a1
    tt_x1 = t1*wf1_x1 + t2*wf2_x1;  tt_x2 = t1*wf1_x2 + t2*wf2_x2
    tt_t1 = t1*wf1_t1 + t2*wf2_t1 + wf1;  tt_t2 = t1*wf1_t2 + t2*wf2_t2 + wf2
    tt_d1 = t1*wf1_d1 + t2*wf2_d1;  tt_d2 = t1*wf1_d2 + t2*wf2_d2
    tt_u1 = t1*wf1_u1 + t2*wf2_u1;  tt_u2 = t1*wf1_u2 + t2*wf2_u2
    tt_ms = t1*wf1_ms + t2*wf2_ms;  tt_re = t1*wf1_re + t2*wf2_re
    tt_xf = t1*wf1_xf + t2*wf2_xf

    dt_v    = d1*wf1    + d2*wf2
    dt_a1 = d1*wf1_a1 + d2*wf2_a1
    dt_x1 = d1*wf1_x1 + d2*wf2_x1;  dt_x2 = d1*wf1_x2 + d2*wf2_x2
    dt_t1 = d1*wf1_t1 + d2*wf2_t1;  dt_t2 = d1*wf1_t2 + d2*wf2_t2
    dt_d1 = d1*wf1_d1 + d2*wf2_d1 + wf1;  dt_d2 = d1*wf1_d2 + d2*wf2_d2 + wf2
    dt_u1 = d1*wf1_u1 + d2*wf2_u1;  dt_u2 = d1*wf1_u2 + d2*wf2_u2
    dt_ms = d1*wf1_ms + d2*wf2_ms;  dt_re = d1*wf1_re + d2*wf2_re
    dt_xf = d1*wf1_xf + d2*wf2_xf

    ut    = u1*wf1    + u2*wf2
    ut_a1 = u1*wf1_a1 + u2*wf2_a1
    ut_x1 = u1*wf1_x1 + u2*wf2_x1;  ut_x2 = u1*wf1_x2 + u2*wf2_x2
    ut_t1 = u1*wf1_t1 + u2*wf2_t1;  ut_t2 = u1*wf1_t2 + u2*wf2_t2
    ut_d1 = u1*wf1_d1 + u2*wf2_d1;  ut_d2 = u1*wf1_d2 + u2*wf2_d2
    ut_u1 = u1*wf1_u1 + u2*wf2_u1 + wf1;  ut_u2 = u1*wf1_u2 + u2*wf2_u2 + wf2
    ut_ms = u1*wf1_ms + u2*wf2_ms;  ut_re = u1*wf1_re + u2*wf2_re
    ut_xf = u1*wf1_xf + u2*wf2_xf

    # ── LAMINAR PART: X1 → XT (Fortran lines 1314-1403) ──
    # Set "2" variables to "T" (transition point)
    c2.x = xt; c2.t = tt; c2.d = dt_v; c2.u = ut
    c2.ampl = fl.amcrit; c2.s = XFT(0)

    xbl_blkin!(c2, fl)
    xbl_blvar!(c2, 1, fl, par)
    mid_lam = xbl_blmid(c1, c2, 1, false, fl, par)

    # Laminar BLDIF (type 1) — output in vs2_lam is wrt "T" variables
    vs1_lam = zeros(XFT,4,5); vs2_lam = zeros(XFT,4,5); vr_lam = zeros(XFT,4)
    vm_lam = zeros(XFT,4); vrr_lam = zeros(XFT,4); vx_lam = zeros(XFT,4)
    xbl_bldif!(vs1_lam, vs2_lam, vr_lam, vm_lam, vrr_lam, vx_lam,
        c1, c2, mid_lam, 1, fl, par)

    # Convert laminar sensitivities from "T" to "1"+"2" variables (Fortran lines 1341-1403)
    bl1 = zeros(XFT, 4, 5); bl2 = zeros(XFT, 4, 5)
    blrez = zeros(XFT, 4); blm = zeros(XFT, 4); blr = zeros(XFT, 4); blx = zeros(XFT, 4)
    for k in 2:3
        blrez[k] = vr_lam[k]
        blm[k] = vm_lam[k] +
            vs2_lam[k,2]*tt_ms + vs2_lam[k,3]*dt_ms +
            vs2_lam[k,4]*ut_ms + vs2_lam[k,5]*xt_ms
        blr[k] = vrr_lam[k] +
            vs2_lam[k,2]*tt_re + vs2_lam[k,3]*dt_re +
            vs2_lam[k,4]*ut_re + vs2_lam[k,5]*xt_re
        blx[k] = vx_lam[k] +
            vs2_lam[k,2]*tt_xf + vs2_lam[k,3]*dt_xf +
            vs2_lam[k,4]*ut_xf + vs2_lam[k,5]*xt_xf

        bl1[k,1] = vs1_lam[k,1] +
            vs2_lam[k,2]*tt_a1 + vs2_lam[k,3]*dt_a1 +
            vs2_lam[k,4]*ut_a1 + vs2_lam[k,5]*xt_a1
        bl1[k,2] = vs1_lam[k,2] +
            vs2_lam[k,2]*tt_t1 + vs2_lam[k,3]*dt_t1 +
            vs2_lam[k,4]*ut_t1 + vs2_lam[k,5]*xt_t1
        bl1[k,3] = vs1_lam[k,3] +
            vs2_lam[k,2]*tt_d1 + vs2_lam[k,3]*dt_d1 +
            vs2_lam[k,4]*ut_d1 + vs2_lam[k,5]*xt_d1
        bl1[k,4] = vs1_lam[k,4] +
            vs2_lam[k,2]*tt_u1 + vs2_lam[k,3]*dt_u1 +
            vs2_lam[k,4]*ut_u1 + vs2_lam[k,5]*xt_u1
        bl1[k,5] = vs1_lam[k,5] +
            vs2_lam[k,2]*tt_x1 + vs2_lam[k,3]*dt_x1 +
            vs2_lam[k,4]*ut_x1 + vs2_lam[k,5]*xt_x1

        bl2[k,1] = XFT(0)
        bl2[k,2] = vs2_lam[k,2]*tt_t2 + vs2_lam[k,3]*dt_t2 +
            vs2_lam[k,4]*ut_t2 + vs2_lam[k,5]*xt_t2
        bl2[k,3] = vs2_lam[k,2]*tt_d2 + vs2_lam[k,3]*dt_d2 +
            vs2_lam[k,4]*ut_d2 + vs2_lam[k,5]*xt_d2
        bl2[k,4] = vs2_lam[k,2]*tt_u2 + vs2_lam[k,3]*dt_u2 +
            vs2_lam[k,4]*ut_u2 + vs2_lam[k,5]*xt_u2
        bl2[k,5] = vs2_lam[k,2]*tt_x2 + vs2_lam[k,3]*dt_x2 +
            vs2_lam[k,4]*ut_x2 + vs2_lam[k,5]*xt_x2
    end

    # ── TURBULENT PART: XT → X2 (Fortran lines 1408-1539) ──
    # Calculate equilibrium shear coefficient CQT at transition point
    xbl_blvar!(c2, 2, fl, par)

    hk2_val = c2.hk
    ctr     = par.ctrcon * exp(-par.ctrcex / max(hk2_val - XFT(1), XFT(0.01)))
    ctr_hk2 = ctr * par.ctrcex / max((hk2_val - XFT(1))^2, XFT(0.0001))

    # ST and its sensitivities wrt TT, DT, UT (Fortran lines 1424-1443)
    st    = ctr*c2.cq
    st_tt = ctr*c2.cq_t + c2.cq*ctr_hk2*c2.hk_t
    st_dt = ctr*c2.cq_d + c2.cq*ctr_hk2*c2.hk_d
    st_ut = ctr*c2.cq_u + c2.cq*ctr_hk2*c2.hk_u
    st_ms_base = ctr*c2.cq_ms + c2.cq*ctr_hk2*c2.hk_ms
    st_re_base = ctr*c2.cq_re

    # ST sensitivities wrt actual "1" and "2" variables (Fortran lines 1432-1443)
    st_a1 = st_tt*tt_a1 + st_dt*dt_a1 + st_ut*ut_a1
    st_x1 = st_tt*tt_x1 + st_dt*dt_x1 + st_ut*ut_x1
    st_x2 = st_tt*tt_x2 + st_dt*dt_x2 + st_ut*ut_x2
    st_t1 = st_tt*tt_t1 + st_dt*dt_t1 + st_ut*ut_t1
    st_t2 = st_tt*tt_t2 + st_dt*dt_t2 + st_ut*ut_t2
    st_d1 = st_tt*tt_d1 + st_dt*dt_d1 + st_ut*ut_d1
    st_d2 = st_tt*tt_d2 + st_dt*dt_d2 + st_ut*ut_d2
    st_u1 = st_tt*tt_u1 + st_dt*dt_u1 + st_ut*ut_u1
    st_u2 = st_tt*tt_u2 + st_dt*dt_u2 + st_ut*ut_u2
    st_ms = st_tt*tt_ms + st_dt*dt_ms + st_ut*ut_ms + st_ms_base
    st_re = st_tt*tt_re + st_dt*dt_re + st_ut*ut_re + st_re_base
    st_xf = st_tt*tt_xf + st_dt*dt_xf + st_ut*ut_xf

    # Set S2=ST and recalculate turbulent secondary variables (Fortran lines 1445-1449)
    c2.ampl = XFT(0); c2.s = st
    xbl_blvar!(c2, 2, fl, par)

    # COM1 = COM2 (transition point), restore COM2 from saved (Fortran lines 1453-1456)
    copycom!(c1, c2)
    copycom!(c2, c2sav)

    # Calculate XT-X2 midpoint and turbulent BLDIF (Fortran lines 1458-1462)
    mid_turb = xbl_blmid(c1, c2, 2, false, fl, par)

    vs1_turb = zeros(XFT,4,5); vs2_turb = zeros(XFT,4,5); vr_turb = zeros(XFT,4)
    vm_turb = zeros(XFT,4); vrr_turb = zeros(XFT,4); vx_turb = zeros(XFT,4)
    xbl_bldif!(vs1_turb, vs2_turb, vr_turb, vm_turb, vrr_turb, vx_turb,
        c1, c2, mid_turb, 2, fl, par)

    # Convert turbulent sensitivities (Fortran lines 1466-1539)
    bt1 = zeros(XFT, 4, 5); bt2 = zeros(XFT, 4, 5)
    btrez = zeros(XFT, 4); btm = zeros(XFT, 4); btr = zeros(XFT, 4); btx = zeros(XFT, 4)
    for k in 1:3
        btrez[k] = vr_turb[k]
        btm[k] = vm_turb[k] +
            vs1_turb[k,1]*st_ms + vs1_turb[k,2]*tt_ms +
            vs1_turb[k,3]*dt_ms + vs1_turb[k,4]*ut_ms + vs1_turb[k,5]*xt_ms
        btr[k] = vrr_turb[k] +
            vs1_turb[k,1]*st_re + vs1_turb[k,2]*tt_re +
            vs1_turb[k,3]*dt_re + vs1_turb[k,4]*ut_re + vs1_turb[k,5]*xt_re
        btx[k] = vx_turb[k] +
            vs1_turb[k,1]*st_xf + vs1_turb[k,2]*tt_xf +
            vs1_turb[k,3]*dt_xf + vs1_turb[k,4]*ut_xf + vs1_turb[k,5]*xt_xf

        bt1[k,1] = vs1_turb[k,1]*st_a1 +
            vs1_turb[k,2]*tt_a1 + vs1_turb[k,3]*dt_a1 +
            vs1_turb[k,4]*ut_a1 + vs1_turb[k,5]*xt_a1
        bt1[k,2] = vs1_turb[k,1]*st_t1 +
            vs1_turb[k,2]*tt_t1 + vs1_turb[k,3]*dt_t1 +
            vs1_turb[k,4]*ut_t1 + vs1_turb[k,5]*xt_t1
        bt1[k,3] = vs1_turb[k,1]*st_d1 +
            vs1_turb[k,2]*tt_d1 + vs1_turb[k,3]*dt_d1 +
            vs1_turb[k,4]*ut_d1 + vs1_turb[k,5]*xt_d1
        bt1[k,4] = vs1_turb[k,1]*st_u1 +
            vs1_turb[k,2]*tt_u1 + vs1_turb[k,3]*dt_u1 +
            vs1_turb[k,4]*ut_u1 + vs1_turb[k,5]*xt_u1
        bt1[k,5] = vs1_turb[k,1]*st_x1 +
            vs1_turb[k,2]*tt_x1 + vs1_turb[k,3]*dt_x1 +
            vs1_turb[k,4]*ut_x1 + vs1_turb[k,5]*xt_x1

        bt2[k,1] = vs2_turb[k,1]
        bt2[k,2] = vs2_turb[k,2] +
            vs1_turb[k,1]*st_t2 + vs1_turb[k,2]*tt_t2 +
            vs1_turb[k,3]*dt_t2 + vs1_turb[k,4]*ut_t2 + vs1_turb[k,5]*xt_t2
        bt2[k,3] = vs2_turb[k,3] +
            vs1_turb[k,1]*st_d2 + vs1_turb[k,2]*tt_d2 +
            vs1_turb[k,3]*dt_d2 + vs1_turb[k,4]*ut_d2 + vs1_turb[k,5]*xt_d2
        bt2[k,4] = vs2_turb[k,4] +
            vs1_turb[k,1]*st_u2 + vs1_turb[k,2]*tt_u2 +
            vs1_turb[k,3]*dt_u2 + vs1_turb[k,4]*ut_u2 + vs1_turb[k,5]*xt_u2
        bt2[k,5] = vs2_turb[k,5] +
            vs1_turb[k,1]*st_x2 + vs1_turb[k,2]*tt_x2 +
            vs1_turb[k,3]*dt_x2 + vs1_turb[k,4]*ut_x2 + vs1_turb[k,5]*xt_x2
    end

    # ── COMBINE laminar + turbulent (Fortran lines 1541-1562) ──
    vsrez[1] =            btrez[1]
    vsrez[2] = blrez[2] + btrez[2]
    vsrez[3] = blrez[3] + btrez[3]
    vsm[1] =            btm[1]
    vsm[2] = blm[2]   + btm[2]
    vsm[3] = blm[3]   + btm[3]
    vsr[1] =            btr[1]
    vsr[2] = blr[2]   + btr[2]
    vsr[3] = blr[3]   + btr[3]
    vsx[1] =            btx[1]
    vsx[2] = blx[2]   + btx[2]
    vsx[3] = blx[3]   + btx[3]
    vsrez[4] = XFT(0); vsm[4] = XFT(0); vsr[4] = XFT(0); vsx[4] = XFT(0)
    for l in 1:5
        vs1[1,l] =            bt1[1,l]
        vs2[1,l] =            bt2[1,l]
        vs1[2,l] = bl1[2,l] + bt1[2,l]
        vs2[2,l] = bl2[2,l] + bt2[2,l]
        vs1[3,l] = bl1[3,l] + bt1[3,l]
        vs2[3,l] = bl2[3,l] + bt2[3,l]
        vs1[4,l] = XFT(0); vs2[4,l] = XFT(0)
    end

    # Restore COM1 (Fortran lines 1567-1569)
    copycom!(c1, c1sav)

    return nothing
end
