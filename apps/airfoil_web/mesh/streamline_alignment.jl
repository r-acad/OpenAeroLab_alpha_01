# =====================================================================================================================
# streamline_alignment.jl -- streamlines from JXfoil for a streamline-aligned CFD mesh (development, 23 Oct 2026).
#
# Phase 1 of the "streamline-aligned mesh" work item (STREAMLINE_ALIGNMENT_DESIGN.md). Runs in the app's server project
# (the JXfoil project). It returns, at one incidence and Reynolds number (Mach 0):
#   * the WAKE PATH from the trailing-edge midpoint (the base midpoint of a finite base) about 3 chords downstream,
#     then a C2 blend of its direction into the freestream direction;
#   * the STAGNATION POINT and the stagnation streamline about 2 chords upstream, then a C2 blend into the upstream
#     freestream direction.
#
# What JXfoil provides and what is used (read-only; nothing in the JXfoil project is changed):
#   * the inviscid linear-vorticity panel solution (`solve_inviscid_case`) and its influence functions
#     (`_psilin_vortex_derivatives`, XFOIL's PSILIN), from which the velocity at any field point is evaluated here in
#     Float64 (JXfoil itself evaluates XFOIL's wake in Float32);
#   * XFOIL's own wake trajectory (`_wake_station_trace`, XFOIL's XYWAKE: forward-Euler steps along the inviscid
#     streamline, one chord, n/8+2 points). It is the reference for the first chord only; the path is integrated here
#     with RK4 to 3 chords instead, because XYWAKE's 22 Euler points are too coarse for a mesh and too short;
#   * the viscous solution (`solve_coupled_case`, the app's JXfoil call) for the convergence gate, the trailing-edge
#     separation gate and the displacement thickness. XFOIL's viscous model does NOT displace the wake path: its wake
#     panels stay on the inviscid streamline at the analysis incidence. The "viscous" path here is therefore built from
#     the classical displacement body (airfoil + delta*): the inviscid panel solution around it carries the viscous
#     circulation and the displacement surfaces' trailing-edge bisector; its wake streamline, moved to start at the
#     geometric trailing edge (the offset relaxed over `relax_length`), is the viscous wake path.
#
# Every result records which field was used (`source`): "viscous_displacement_body" or "inviscid" (explicit, or as a
# labelled fallback "inviscid_fallback" with the viscous refusal reason). A viscous request is REFUSED (StreamlineRefusal)
# when JXfoil does not converge or when the trailing-edge separation exceeds `max_te_separation` (beyond stall the
# attached-wake model is not a description of the flow). Geometry only: no CFD file is read or written.
# =====================================================================================================================
module AirfoilStreamlineAlignment

const WORKSPACE_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const JXFOIL_ROOT = joinpath(WORKSPACE_ROOT, "third_party", "JXfoil")
isfile(joinpath(JXFOIL_ROOT, "src", "JXfoil.jl")) || error("JXfoil source tree was not found at $JXFOIL_ROOT")
JXFOIL_ROOT in LOAD_PATH || pushfirst!(LOAD_PATH, JXFOIL_ROOT)
import JXfoil
using SHA
using TOML

export StreamlineRefusal, PanelField, panel_field, field_velocity, trace_streamline, extract_streamlines,
       streamline_hash, write_streamline_file, read_streamline_file, STREAMLINE_FORMAT

const STREAMLINE_FORMAT = "airfoil_lab_streamlines_v1"

"Raised when a requested viscous path cannot be produced; `reason` is the user-facing sentence."
struct StreamlineRefusal <: Exception
    reason::String
end
Base.showerror(io::IO, e::StreamlineRefusal) = print(io, "streamline alignment refused: ", e.reason)

_smooth5(t) = (t = clamp(t, 0.0, 1.0); t^3*(10 - 15t + 6t^2))   # C2 quintic smoothstep 0 -> 1

# ---- the Float64 panel field ------------------------------------------------------------------------------------------
"""
    PanelField

The velocity field of one JXfoil inviscid solution: freestream (cos a, sin a) plus XFOIL's linear-vorticity sheet with
node strengths `gam` (JXfoil's `qnode`) and its trailing-edge panel (scs, sds), evaluated in Float64.
"""
struct PanelField
    x::Vector{Float64}; y::Vector{Float64}; s::Vector{Float64}; apanel::Vector{Float64}
    xp::Vector{Float64}; yp::Vector{Float64}
    scs::Float64; sds::Float64; sharp::Bool; xte::Float64; yte::Float64
    gam::Vector{Float64}; ca::Float64; sa::Float64; alpha_deg::Float64; cl::Float64
end

function panel_field(inv::JXfoil.CaseResult)
    x = Float64.(inv.xnode); y = Float64.(inv.ynode)
    length(x) == length(y) == length(inv.qnode) || throw(ArgumentError("JXfoil case geometry and strengths differ in length"))
    s = JXfoil._scalc(x, y)
    nx, ny, xp, yp = JXfoil._ncalc(x, y, s)
    te = JXfoil._tecalc(x, y, xp, yp)
    ap = JXfoil._apcalc(x, y, nx, ny; sharp=te.sharp)
    a = deg2rad(inv.alpha_deg)
    PanelField(x, y, Float64.(s), Float64.(ap), Float64.(xp), Float64.(yp), Float64(te.scs), Float64(te.sds), te.sharp,
               Float64(te.xte), Float64(te.yte), Float64.(inv.qnode), cos(a), sin(a), Float64(inv.alpha_deg), Float64(inv.cl))
end

"Velocity (u, v) of the field at (x, y): u = dpsi/dy, v = -dpsi/dx, with XFOIL's field-point (io = 0) PSILIN branch."
function field_velocity(F::PanelField, x::Real, y::Real)
    xf = Float64(x); yf = Float64(y)
    _, dqx = JXfoil._psilin_vortex_derivatives(0, xf, yf, 1.0, 0.0, F.x, F.y, F.s, F.apanel; scs=F.scs, sds=F.sds)
    _, dqy = JXfoil._psilin_vortex_derivatives(0, xf, yf, 0.0, 1.0, F.x, F.y, F.s, F.apanel; scs=F.scs, sds=F.sds)
    psx = -F.sa; psy = F.ca
    @inbounds for i in eachindex(F.gam)
        psx += dqx[i]*F.gam[i]; psy += dqy[i]*F.gam[i]
    end
    (psy, -psx)
end

# ---- RK4 streamline in arc length -------------------------------------------------------------------------------------
"""
    trace_streamline(F, x0, y0, len; direction=+1, h0=2e-4, growth=1.04, hmax=0.01, refine=1) -> (x, y, s)

Classical RK4 on dX/ds = direction * V/|V| (unit tangent, so s is arc length) from (x0, y0) over `len` chords. The step
grows geometrically from `h0` by `growth` to `hmax`; `refine = 2^k` halves every step k times (the convergence test).
Refuses a vanishing velocity (a stagnation point on the path).
"""
function trace_streamline(F::PanelField, x0::Real, y0::Real, len::Real; direction::Integer=1, h0::Real=2e-4,
                          growth::Real=1.04, hmax::Real=0.01, refine::Integer=1)
    refine >= 1 || throw(ArgumentError("refine must be >= 1"))
    h = Float64(h0)/refine; hm = Float64(hmax)/refine; g = Float64(growth)^(1/refine)
    dirn = Float64(sign(direction))
    function unit(px, py)
        u, v = field_velocity(F, px, py); q = hypot(u, v)
        q > 1e-10 || throw(StreamlineRefusal("the streamline reached a stagnation point at ($(px), $(py))"))
        (dirn*u/q, dirn*v/q)
    end
    x = Float64(x0); y = Float64(y0); s = 0.0; L = Float64(len)
    xs = [x]; ys = [y]; ss = [s]
    while s < L - 1e-13
        hh = min(h, L - s)
        k1 = unit(x, y)
        k2 = unit(x + 0.5hh*k1[1], y + 0.5hh*k1[2])
        k3 = unit(x + 0.5hh*k2[1], y + 0.5hh*k2[2])
        k4 = unit(x + hh*k3[1], y + hh*k3[2])
        x += hh/6*(k1[1] + 2k2[1] + 2k3[1] + k4[1]); y += hh/6*(k1[2] + 2k2[2] + 2k3[2] + k4[2])
        s += hh; push!(xs, x); push!(ys, y); push!(ss, s)
        h = min(hm, h*g)
    end
    (x=xs, y=ys, s=ss)
end

"Point of a polyline (x, y, s) at arc parameter `q` (linear interpolation; clamped)."
function _at(p, q::Real)
    s = p.s; q <= s[1] && return (p.x[1], p.y[1]); q >= s[end] && return (p.x[end], p.y[end])
    k = searchsortedlast(s, q); k = clamp(k, 1, length(s) - 1)
    t = (q - s[k])/(s[k+1] - s[k])
    ((1 - t)*p.x[k] + t*p.x[k+1], (1 - t)*p.y[k] + t*p.y[k+1])
end

"Direction (radians) of a polyline's last segment."
_end_angle(p) = atan(p.y[end] - p.y[end-1], p.x[end] - p.x[end-1])

"""Append a C2 blend of the path direction from its end angle into `theta_inf` over `blend` chords, then a straight
segment to `total` chords of arc length. Steps of `ds`."""
function _blend_to(p, theta_inf::Float64, blend::Float64, total::Float64; ds::Float64=0.01)
    xs = copy(p.x); ys = copy(p.y); ss = copy(p.s)
    th0 = _end_angle(p)
    dth = rem2pi(theta_inf - th0, RoundNearest)
    s = ss[end]; s0 = s; x = xs[end]; y = ys[end]
    while s < total - 1e-12
        h = min(ds, total - s)
        # midpoint rule on the blended angle (exact for the straight part)
        tm = th0 + dth*_smooth5((s + 0.5h - s0)/blend)
        x += h*cos(tm); y += h*sin(tm); s += h
        push!(xs, x); push!(ys, y); push!(ss, s)
    end
    (x=xs, y=ys, s=ss)
end

# ---- surface helpers --------------------------------------------------------------------------------------------------
"Outward unit normal of the panel polyline at node i (counterclockwise traversal: n = (t_y, -t_x))."
function _node_normal(x, y, i)
    n = length(x)
    a = max(1, i - 1); b = min(n, i + 1)
    tx = x[b] - x[a]; ty = y[b] - y[a]; t = hypot(tx, ty)
    (ty/t, -tx/t)
end

"Stagnation point of a field: XFOIL's STFIND on the field's own node strengths; returns the point, arc length and normal."
function _stagnation(F::PanelField, inv::JXfoil.CaseResult)
    stag, s = JXfoil._xfoil_stfind(inv)
    i = stag.ist; t = (stag.sst - s[i])/(s[i+1] - s[i])
    x = (1 - t)*F.x[i] + t*F.x[i+1]; y = (1 - t)*F.y[i] + t*F.y[i+1]
    tx = F.x[i+1] - F.x[i]; ty = F.y[i+1] - F.y[i]; tl = hypot(tx, ty)
    (x=x, y=y, s=Float64(stag.sst), ist=i, nx=ty/tl, ny=-tx/tl)
end

"Closest point of the closed panel polyline (x, y) to (px, py)."
function _closest_on_polyline(x, y, px, py)
    best = (Inf, px, py, 0)
    for i in 1:length(x)-1
        ax, ay = x[i], y[i]; bx, by = x[i+1], y[i+1]
        dx = bx - ax; dy = by - ay; L2 = dx*dx + dy*dy
        t = L2 > 0 ? clamp(((px - ax)*dx + (py - ay)*dy)/L2, 0.0, 1.0) : 0.0
        qx = ax + t*dx; qy = ay + t*dy; d = hypot(px - qx, py - qy)
        d < best[1] && (best = (d, qx, qy, i))
    end
    (d=best[1], x=best[2], y=best[3], i=best[4])
end

"TE bisector direction of XFOIL's wake start (XYWAKE's first normal), radians."
function _te_bisector(F::PanelField)
    sx = 0.5*(F.yp[end] - F.yp[1]); sy = 0.5*(F.xp[1] - F.xp[end]); sm = hypot(sx, sy)
    nx = sx/sm; ny = sy/sm
    atan(nx, -ny)            # tangent t = (-ny, nx)
end

# ---- trailing-edge separation from the viscous march ------------------------------------------------------------------
"Chord fraction, ending at the trailing edge, over which the skin friction of one surface is negative."
function _te_separation(surf)
    n = length(surf.xc); n >= 2 || return 0.0
    cf = [Float64(st.cf) for st in surf.secondary]
    xs = Float64.(surf.xc)
    k = n
    while k >= 1 && cf[k] < 0
        k -= 1
    end
    k == n && return 0.0
    k == 0 && return 1.0
    xsep = xs[k] + (xs[k+1] - xs[k])*(cf[k]/(cf[k] - cf[k+1]))     # linear zero crossing
    max(0.0, maximum(xs) - xsep)
end

# ---- the displacement body --------------------------------------------------------------------------------------------
"""
    _panel_dstar(r) -> (dstar, stagnation_arc, side_mismatch)

delta* at every panel node of a viscous result, from the boundary-layer stations by arc length measured from the VISCOUS
stagnation point. XFOIL's stations sit on panel nodes with xi = |s - s_stag|; the upper side's last station is node 1
(xi = s_stag) and the lower side's is node n (xi = s_total - s_stag), so s_stag follows from the stations themselves.
(JXfoil's `_panel_dstar_from_surfaces` uses the inviscid solution's stagnation point instead; when the viscous one has
moved by a panel, as at 12 deg, it gives the second node the trailing-edge value and distorts the displaced trailing
edge: found on NACA 4412, 23 Oct 2026.)
"""
function _panel_dstar(r; te_dstar::Symbol=:raw)
    inv = r.inviscid
    x = Float64.(inv.xnode); y = Float64.(inv.ynode); n = length(x)
    s = zeros(n); for i in 2:n; s[i] = s[i-1] + hypot(x[i] - x[i-1], y[i] - y[i-1]); end
    xu = Float64.(r.upper.x); du = [max(Float64(p.d), 0.0) for p in r.upper.primary]
    xl = Float64.(r.lower.x); dl = [max(Float64(p.d), 0.0) for p in r.lower.primary]
    su = xu[end]; sl = s[end] - xl[end]
    mismatch = abs(su - sl)
    mismatch <= 1e-4*s[end] || throw(StreamlineRefusal("the viscous stations do not close around the section " *
        "(upper and lower stagnation arcs differ by $(mismatch)): cannot place delta* on the panel nodes"))
    sst = 0.5*(su + sl)
    lin(xs, ds, q) = (q <= xs[1] ? ds[1] : q >= xs[end] ? ds[end] :
                      (k = searchsortedlast(xs, q); t = (q - xs[k])/(xs[k+1] - xs[k]); (1 - t)*ds[k] + t*ds[k+1]))
    d = [s[i] <= sst ? lin(xu, du, su - s[i]) : lin(xl, dl, s[i] - sl) for i in 1:n]
    raw = copy(d)
    te_dstar in (:raw, :sharp_last_node, :fit) || throw(ArgumentError("te_dstar must be :raw, :sharp_last_node or :fit"))
    fits = Any[]
    if te_dstar === :fit
        fits = [_smooth_te_dstar!(d, s, side, sst) for side in (:upper, :lower)]
    elseif te_dstar === :sharp_last_node && hypot(x[1] - x[n], y[1] - y[n]) < 1e-4*(maximum(x) - minimum(x))
        # a sharp edge: the trailing-edge node's delta* by linear extrapolation of the two stations before it
        d[1] = max(0.0, d[2] + (d[2] - d[3])*(s[2] - s[1])/(s[3] - s[2]))
        d[n] = max(0.0, d[n-1] + (d[n-1] - d[n-2])*(s[n] - s[n-1])/(s[n-1] - s[n-2]))
        fits = Any[Dict("side"=>"both", "treatment"=>"sharp_last_node", "dstar_te_upper_raw"=>raw[1], "dstar_te_lower_raw"=>raw[n])]
    end
    (dstar=d, dstar_raw=raw, stagnation_arc=sst, side_mismatch=mismatch, te_fits=fits)
end

"""
The trailing-edge displacement slope at boundary-layer-thickness scale. Over the last L = clamp(5 delta*_TE, 0.02, 0.15)
chords of one side delta* is replaced by its least-squares line in the distance q from the trailing edge (fitted on
q <= 2L, blended back to the stations C2 over L < q < 2L). XFOIL's last panel at a sharp, finite-angle edge carries a
local delta* jump (22% over a 0.002 c panel on the NASA TMR NACA 0012 at 4 deg, the adverse gradient into the edge's
stagnation point) that a spline end tangent turns into a spurious several-degree rotation of the departure: a
displacement surface is meaningful only at the layer's own thickness (about 5 delta*).
NOT THE DEFAULT (te_dstar = :fit): measured on NACA 0012/4412 and their sharp forms at 4-12 deg, the line under-represents
the convex delta* growth into the edge and the body's CL exceeds JXfoil's viscous CL by up to 12.5%; the default
:sharp_last_node (the stations as they are; at a sharp edge only the edge node's value replaced by the linear
extrapolation of the two before it) closes CL within 2.3% in all twelve cases.
"""
function _smooth_te_dstar!(d, s, side::Symbol, sst)
    n = length(d)
    idx = side === :upper ? [i for i in 1:n if s[i] <= sst] : [i for i in n:-1:1 if s[i] > sst]
    q = side === :upper ? [s[i] for i in idx] : [s[n] - s[i] for i in idx]
    dte = d[idx[1]]
    L = clamp(5dte, 0.02, 0.15)
    sel = [k for k in eachindex(q) if q[k] <= 2L]
    length(sel) >= 3 || return Dict("side"=>String(side), "fit_length"=>L, "applied"=>false)
    qs = q[sel]; ds = d[idx[sel]]
    qm = sum(qs)/length(qs); dm = sum(ds)/length(ds)
    b = sum((qs .- qm).*(ds .- dm))/sum((qs .- qm).^2); a = dm - b*qm
    for k in eachindex(q)
        q[k] >= 2L && break
        w = q[k] <= L ? 1.0 : 1.0 - _smooth5((q[k] - L)/L)
        d[idx[k]] = max(0.0, w*(a + b*q[k]) + (1 - w)*d[idx[k]])
    end
    Dict("side"=>String(side), "fit_length"=>L, "applied"=>true, "dstar_te_raw"=>dte, "dstar_te_fit"=>max(0.0, a),
         "slope_dstar_dq"=>b)
end

"""The displacement body of a converged viscous result: every panel node moved outward by its delta* (`_panel_dstar`),
solved inviscid at the same incidence on the same nodes (no repaneling)."""
function _displacement_body(r, alpha_deg; te_dstar::Symbol=:raw)
    inv = r.inviscid
    pd = _panel_dstar(r; te_dstar); dstar = pd.dstar
    x = Float64.(inv.xnode); y = Float64.(inv.ynode)
    xd = similar(x); yd = similar(y)
    for i in eachindex(x)
        nx, ny = _node_normal(x, y, i)
        xd[i] = x[i] + dstar[i]*nx; yd[i] = y[i] + dstar[i]*ny
    end
    body = JXfoil.Airfoil("displacement body", xd, yd)
    invd = JXfoil.solve_inviscid_case(body, alpha_deg; repanel=false)
    (inv=invd, dstar=dstar, x=xd, y=yd, dstar_te_upper=dstar[1], dstar_te_lower=dstar[end],
     viscous_stagnation_arc=pd.stagnation_arc, station_side_mismatch=pd.side_mismatch, te_fits=pd.te_fits,
     dstar_te_upper_raw=pd.dstar_raw[1], dstar_te_lower_raw=pd.dstar_raw[end])
end

# ---- the composed paths -----------------------------------------------------------------------------------------------
function _wake_path(F::PanelField, alpha_deg; wake_length, wake_blend, total, h0, growth, hmax, refine, start_offset=1e-4)
    th = _te_bisector(F)
    x0 = F.xte + start_offset*cos(th); y0 = F.yte + start_offset*sin(th)
    p = trace_streamline(F, x0, y0, wake_length - start_offset; direction=1, h0, growth, hmax, refine)
    p = (x=vcat(F.xte, p.x), y=vcat(F.yte, p.y), s=vcat(0.0, p.s .+ start_offset))
    traced_end = length(p.s)
    q = _blend_to(p, deg2rad(alpha_deg), wake_blend, total)
    (path=q, traced_points=traced_end, bisector=th)
end

function _stagnation_path(F::PanelField, st, alpha_deg; upstream_length, upstream_blend, total, h0, growth, hmax, refine,
                          start_offset=1e-3)
    x0 = st.x + start_offset*st.nx; y0 = st.y + start_offset*st.ny
    p = trace_streamline(F, x0, y0, upstream_length - start_offset; direction=-1, h0, growth, hmax, refine)
    p = (x=vcat(st.x, p.x), y=vcat(st.y, p.y), s=vcat(0.0, p.s .+ start_offset))
    traced_end = length(p.s)
    q = _blend_to(p, deg2rad(alpha_deg) + pi, upstream_blend, total)
    (path=q, traced_points=traced_end)
end

"Angle (deg) of a path's chord between arc lengths a and b, relative to the freestream angle alpha."
function _segment_angle_deg(p, a, b, alpha_deg)
    xa, ya = _at(p, a); xb, yb = _at(p, b)
    rad2deg(rem2pi(atan(yb - ya, xb - xa) - deg2rad(alpha_deg), RoundNearest))
end

"""
    extract_streamlines(foil, alpha_deg; reynolds=3e6, ncrit=9, mode=:viscous, allow_inviscid_fallback=false, ...)
        -> Dict{String,Any}

The wake path and the stagnation streamline of `foil` (a JXfoil.Airfoil, chord on x in [0, 1]) at `alpha_deg`, Mach 0.

`mode=:viscous`: JXfoil's coupled solve (as the app calls it); refused unless converged and its trailing-edge
separation is at most `max_te_separation` chords on each surface; the paths are the displacement body's streamlines,
the wake moved to start at the geometric trailing edge (offset relaxed C2 over `relax_length`). With
`allow_inviscid_fallback=true` a refusal returns the inviscid paths instead, labelled "inviscid_fallback" with the reason.
`mode=:inviscid`: the inviscid panel solution's streamlines (XFOIL's own wake definition).
`te_dstar` (viscous only): the displacement thickness at the trailing edge, :sharp_last_node (default), :raw or :fit
(see `_smooth_te_dstar!` for the measured choice).

Lengths in chords: the wake is traced `wake_length` (3) then blended to the freestream direction over `wake_blend` (1)
and continued straight to `wake_total` (40); the stagnation streamline is traced `upstream_length` (2), blended over
`upstream_blend` (0.5) and continued to `upstream_total` (40). The path points are in the airfoil's own frame
(the foil is not rotated; the freestream is at alpha).
"""
function extract_streamlines(foil::JXfoil.Airfoil, alpha_deg::Real; reynolds::Real=3e6, ncrit::Real=9.0,
                             mode::Symbol=:viscous, allow_inviscid_fallback::Bool=false, repanel::Bool=true,
                             viscal_max_iter::Integer=80, max_te_separation::Real=0.10,
                             wake_length::Real=3.0, wake_blend::Real=1.0, wake_total::Real=40.0,
                             upstream_length::Real=2.0, upstream_blend::Real=0.5, upstream_total::Real=40.0,
                             relax_length::Real=0.5, h0::Real=2e-4, growth::Real=1.04, hmax::Real=0.01, refine::Integer=1,
                             viscous_result=nothing, te_dstar::Symbol=:sharp_last_node)
    a = Float64(alpha_deg)
    isfinite(a) && abs(a) <= 25 || throw(ArgumentError("alpha must be finite and within +-25 degrees"))
    mode in (:viscous, :inviscid) || throw(ArgumentError("mode must be :viscous or :inviscid"))
    out = Dict{String,Any}("format"=>STREAMLINE_FORMAT, "alpha_deg"=>a, "reynolds"=>Float64(reynolds),
        "ncrit"=>Float64(ncrit), "mach"=>0.0, "requested_mode"=>String(mode), "airfoil_name"=>foil.name,
        "airfoil_points"=>length(foil.x), "repanel"=>repanel,
        "lengths"=>Dict("wake_traced"=>Float64(wake_length), "wake_blend"=>Float64(wake_blend),
            "wake_total"=>Float64(wake_total), "upstream_traced"=>Float64(upstream_length),
            "upstream_blend"=>Float64(upstream_blend), "upstream_total"=>Float64(upstream_total),
            "wake_offset_relax"=>Float64(relax_length)),
        "integration"=>Dict("method"=>"RK4 in arc length on the unit velocity direction", "h0"=>Float64(h0),
            "growth"=>Float64(growth), "hmax"=>Float64(hmax), "refine"=>Int(refine)))
    inv0 = JXfoil.solve_inviscid_case(foil, a; repanel)
    out["cl_inviscid"] = Float64(inv0.cl)
    refusal = ""; r = nothing; db = nothing
    if mode === :viscous
        r = viscous_result === nothing ? JXfoil.solve_coupled_case(foil, a; reynolds, ncrit, repanel, viscal_max_iter) :
                                         viscous_result
        p = r.provenance
        out["viscous"] = Dict{String,Any}("converged"=>p.converged, "status"=>string(p.status),
            "backend"=>string(p.backend), "iterations"=>p.iterations, "cl"=>Float64(r.cl), "cd"=>Float64(r.cd),
            "cm_c4"=>Float64(r.cm_c4), "top_xtr"=>Float64(r.top_xtr), "bot_xtr"=>Float64(r.bot_xtr))
        sep_u = _te_separation(r.upper); sep_l = _te_separation(r.lower)
        out["viscous"]["te_separation_upper"] = sep_u; out["viscous"]["te_separation_lower"] = sep_l
        if !p.converged
            refusal = "JXfoil's viscous solution did not converge at alpha $(a) deg (status $(p.status)); " *
                      "the wake of a non-converged (for example stalled) case is not a reliable mesh target"
        elseif max(sep_u, sep_l) > max_te_separation
            refusal = "JXfoil's viscous solution has trailing-edge separation over $(round(max(sep_u, sep_l); sigdigits=3)) c " *
                      "(limit $(max_te_separation) c) at alpha $(a) deg: beyond the attached-flow wake model"
        else
            db = _displacement_body(r, a; te_dstar)
        end
    end
    kw = (; h0, growth, hmax, refine)
    if mode === :viscous && isempty(refusal)
        F = panel_field(db.inv)
        source = "viscous_displacement_body"
        out["cl_field"] = Float64(db.inv.cl)
        out["displacement_body"] = Dict("dstar_te_upper"=>db.dstar_te_upper, "dstar_te_lower"=>db.dstar_te_lower,
            "cl_displacement_body"=>Float64(db.inv.cl), "cl_viscous"=>Float64(r.cl),
            "cl_ratio_body_over_viscous"=>Float64(db.inv.cl/r.cl), "viscous_stagnation_arc"=>db.viscous_stagnation_arc,
            "station_side_mismatch"=>db.station_side_mismatch, "te_dstar_fits"=>db.te_fits,
            "dstar_te_upper_raw"=>db.dstar_te_upper_raw, "dstar_te_lower_raw"=>db.dstar_te_lower_raw,
            "dstar_mapping"=>"stations by arc length from the viscous stagnation point (_panel_dstar)",
            "te_dstar"=>String(te_dstar))
        stinv = db.inv
    elseif mode === :viscous && !allow_inviscid_fallback
        throw(StreamlineRefusal(refusal))
    else
        F = panel_field(inv0)
        source = mode === :viscous ? "inviscid_fallback" : "inviscid"
        isempty(refusal) || (out["fallback_reason"] = refusal)
        out["cl_field"] = Float64(inv0.cl)
        stinv = inv0
    end
    out["source"] = source
    out["source_label"] = source == "viscous_displacement_body" ?
        "viscous (JXfoil coupled solution; inviscid streamlines of its displacement body)" :
        source == "inviscid" ? "inviscid (JXfoil panel solution; XFOIL's own wake definition, no viscous effect)" :
        "inviscid FALLBACK (the viscous path was refused: " * refusal * ")"
    # -- wake
    w = _wake_path(F, a; wake_length=Float64(wake_length), wake_blend=Float64(wake_blend), total=Float64(wake_total), kw...)
    wx = copy(w.path.x); wy = copy(w.path.y); ws = copy(w.path.s)
    geo = panel_field(inv0)             # the geometric section (JXfoil's paneling) for the TE midpoint and the surface
    te_shift = (geo.xte - F.xte, geo.yte - F.yte)
    for k in eachindex(wx)
        f = 1 - _smooth5(ws[k]/Float64(relax_length))
        source == "viscous_displacement_body" || (f = 1.0)
        wx[k] += f*te_shift[1]; wy[k] += f*te_shift[2]
    end
    wx[1] = geo.xte; wy[1] = geo.yte
    wake = (x=wx, y=wy, s=ws)
    # -- stagnation
    st = _stagnation(F, stinv)
    sp = _stagnation_path(F, st, a; upstream_length=Float64(upstream_length), upstream_blend=Float64(upstream_blend),
                          total=Float64(upstream_total), kw...)
    surf = _closest_on_polyline(geo.x, geo.y, st.x, st.y)
    sx = copy(sp.path.x); sy = copy(sp.path.y)
    sx[1] = surf.x; sy[1] = surf.y            # the stagnation point on the geometric section (delta* there ~ 1e-5 c)
    stag = (x=sx, y=sy, s=sp.path.s)
    # -- diagnostics
    dep = rad2deg(rem2pi(atan(wake.y[3] - wake.y[2], wake.x[3] - wake.x[2]), RoundNearest))
    xyw = try
        tr = JXfoil._wake_station_trace(stinv; waklen=1.0)
        (x=Float64.(tr.xw), y=Float64.(tr.yw))
    catch
        nothing
    end
    out["te_point"] = [geo.xte, geo.yte]
    out["te_point_field"] = [F.xte, F.yte]
    out["te_sharp"] = geo.sharp
    out["te_bisector_deg"] = rad2deg(_te_bisector(F))
    out["te_bisector_geometric_deg"] = rad2deg(_te_bisector(geo))
    out["te_departure_deg"] = dep
    out["wake_angle_rel_freestream_deg"] = Dict(string(L)=>_segment_angle_deg(wake, L - 0.25, L, a)
                                                for L in (0.5, 1.0, 2.0, 3.0))
    out["wake_traced_points"] = w.traced_points
    out["stagnation"] = Dict("x"=>surf.x, "y"=>surf.y, "x_field"=>st.x, "y_field"=>st.y,
        "arc_from_upper_te"=>st.s, "panel"=>st.ist, "normal"=>[st.nx, st.ny])
    out["stagnation_traced_points"] = sp.traced_points
    if xyw !== nothing
        # XFOIL's XYWAKE (Euler, Float32) against this path over the first chord: the maximum normal distance
        dmax = 0.0
        for k in eachindex(xyw.x)
            c = _closest_on_polyline(wx, wy, xyw.x[k], xyw.y[k])   # distance to the traced wake polyline
            dmax = max(dmax, c.d)
        end
        out["xfoil_xywake"] = Dict("x"=>xyw.x, "y"=>xyw.y, "max_distance_to_path"=>dmax,
            "note"=>"XFOIL's own wake (forward Euler on the same field, Float32, 1 chord) for comparison; not used")
    end
    out["wake"] = Dict("x"=>wake.x, "y"=>wake.y, "s"=>wake.s)
    out["stagnation_streamline"] = Dict("x"=>stag.x, "y"=>stag.y, "s"=>stag.s)
    out["sha256"] = streamline_hash(out)
    out
end

# ---- receipts and the path file ---------------------------------------------------------------------------------------
"SHA-256 of the path arrays and the incidence (the bytes of the Float64 values, in a fixed order)."
function streamline_hash(d::AbstractDict)
    io = IOBuffer()
    write(io, STREAMLINE_FORMAT)
    write(io, Float64(d["alpha_deg"]))
    for key in ("wake", "stagnation_streamline"), f in ("x", "y", "s")
        for v in d[key][f]; write(io, Float64(v)); end
    end
    for v in d["te_point"]; write(io, Float64(v)); end
    bytes2hex(sha256(take!(io)))
end

"Write the paths as TOML (Julia prints Float64 with round-trip precision) and check the hash on read-back."
function write_streamline_file(path::AbstractString, d::AbstractDict)
    clean(x::AbstractDict) = Dict{String,Any}(string(k)=>clean(v) for (k, v) in pairs(x) if v !== nothing)
    clean(x::AbstractVector) = [clean(v) for v in x]
    clean(x::Symbol) = String(x)
    clean(x) = x
    open(io -> TOML.print(io, clean(d)), path, "w")
    back = read_streamline_file(path)
    back["sha256"] == d["sha256"] || error("streamline file did not round-trip its hash")
    path
end

function read_streamline_file(path::AbstractString)
    d = TOML.parsefile(path)
    get(d, "format", "") == STREAMLINE_FORMAT || throw(ArgumentError("not a $(STREAMLINE_FORMAT) file: $path"))
    for key in ("wake", "stagnation_streamline"), f in ("x", "y", "s")
        d[key][f] = Float64.(d[key][f])
    end
    d["te_point"] = Float64.(d["te_point"])
    h = streamline_hash(d)
    h == d["sha256"] || throw(ArgumentError("streamline file hash mismatch: $path"))
    d
end

end # module
