# =====================================================================================================================
# mesh_streamline_morph.jl -- smooth node displacement of a structured C or C+H mesh onto JXfoil streamlines
# (development, 23 Oct 2026; design: STREAMLINE_ALIGNMENT_DESIGN.md).
#
# Pure Julia (Base and the SHA standard library). Node arrays in, new node arrays out: no solver, no JXfoil, no CMeshGen
# types, so the same code can serve the app's compressible C+H path and, copied or included, a solver that builds its own
# mesh. Inputs:
#   C block   X, Y :: (NI+1) x (NJ+1), NI = 2 NIW + NIF; row j = 1 is the wall and the wake cut(s), j = NJ+1 the outer
#             boundary; columns 1 and NI+1 lie on the outlet; foil columns NIW+1 .. NIW+NIF+1;
#   H block   Xs, Ys :: (NIW+1) x (NS+1) or `nothing` (sharp C mesh); column 1 is the finite base, row 1 the lower leg
#             (= C row 1, i = NIW+1 .. 1), row NS+1 the upper leg (= C row 1, i = NIW+NIF+1 .. NI+1);
#   paths     the wake path (x, y from the trailing-edge point downstream) and the stagnation streamline (x, y from the
#             stagnation point upstream), as written by streamline_alignment.jl.
#
# TRAILING EDGE (shear form). Frame: P0 = the trailing-edge point (the sharp edge node, or the base midpoint), e_s = the
# direction of the current cut to the outlet, e_n its normal; s, eta a node's coordinates. The current cut (the H block's
# centreline, or the sharp cut row) has offset eta_ref(s), the path eta_path(s), D = eta_path - eta_ref. Every C node is
# moved along e_n by
#        d(s, eta) = A E(s) core(s) w(|eta - eta_ref(s)|)
#   core(s)  = Deff(s) m(s) - J(s) + J_end smooth5((s - s_surf)/te_lag_relax)    (core = Deff beyond the two ramps)
#     m(s)   = smooth5((s - s_surf)/te_start_ramp): the cut's SLOPE turns from the old cut's into the path's over
#              te_start_ramp behind the furthest surface point s_surf, so nothing at the edge moves (the edge's columns
#              keep their first spacing) and there is no overshoot;
#     J(s)   = integral from s_surf of Deff m' (its end value J_end, the lag of that turn, is removed over te_lag_relax);
#     Deff   = D up to te_hold, then (te_far = :freeze) D(te_hold) + D'(te_hold) Phi(s - te_hold), Phi' = 1 -
#              smooth5(./te_freeze_length): the offset's slope decays C2 to zero (no kink), or (:follow) D itself;
#   E(s)     = 1 to te_hold, a C2 quintic decay to 0 at te_outlet_margin before the outlet (outlet band untouched);
#   w        = 1 inside the plateau (the base height: the H block legs move rigidly), a C2 quintic decay to 0 over
#              W = max(te_width_min, te_width_factor * max|A E core|) (the cross-wake stretch is at most 1.875/factor).
# The H block's straight columns are rebuilt from the moved legs at their original cross parameters (the generator's P5);
# its legs are assigned the C row-1 nodes bitwise. The displacement is one pure function of a node's coordinates, so a
# sharp mesh's two coincident cut rows stay bitwise equal.
#
# LEADING EDGE (column frame). The stagnation point is projected on the wall; the nearest foil column i_s (re-indexing:
# no wall node moves) is mapped onto the stagnation streamline: its signed offset normal to itself, Dperp(l) =
# ((the streamline point closest to C_is(l)) - C_is(l)) . n_is(l), frozen at l = le_hold beyond it. Every foil column i is
# moved SIDEWAYS, normal to itself (n_ij, the left normal of its own j-direction), by
#        d(i, j) = A_le * T(sigma_i - sigma_st) * R(l_ij) * Dperp(l_ij) * n_ij
# (a local rotation of the leading-edge fan: each column's wall-normal intervals change only at second order),
# R = 0 within le_ramp_start of the wall (wall rows and first spacing untouched), a C2 rise to 1 at le_ramp_end, 1 to
# le_hold, a C2 decay to 0 at le_end; T = 1 within le_plateau of the stagnation point (wall arc length), a C2 decay to 0
# over le_width. Surface nodes are never moved (sliding them along the exact surface was rejected: design note).
# =====================================================================================================================
module MeshStreamlineMorph

using SHA
using TOML

export MorphParams, streamline_morph, smooth5, te_frame, morph_hash, MORPH_TAG, params_dict, params_from, admit_morph,
       read_streamline_paths, streamline_paths_hash, bad_corners, aligned_wake_path

const MORPH_TAG = "StreamlineMorphV1"

"C2 quintic smoothstep: 0 for t <= 0, 1 for t >= 1 (exactly)."
smooth5(t::Float64) = (t = clamp(t, 0.0, 1.0); t*t*t*(10.0 - 15.0t + 6.0t*t))
"Integral of (1 - smooth5(v/L)) dv from 0 to u: u - L(5/2 t^4 - 3 t^5 + t^6), t = u/L; L/2 beyond L."
_phi(u::Float64, L::Float64) = u <= 0 ? u : u >= L ? 0.5L : (t = u/L; u - L*t^4*(2.5 - 3t + t*t))
"Derivative of smooth5(t) in t."
_dsmooth5(t::Float64) = (0.0 < t < 1.0) ? 30.0*t*t*(1.0 - t)^2 : 0.0

"""
    MorphParams

Every length in chords. Defaults are the development settings of 23 Oct 2026 (design note, section 4).
"""
Base.@kwdef struct MorphParams
    te::Bool = true
    le::Bool = true
    te_amplitude::Float64 = 1.0
    te_start_ramp::Float64 = 0.02        # the cut's slope turns into the path's over this length behind the edge
    te_lag_relax::Float64 = 0.25         # the turn's lag is removed over this length
    te_hold::Float64 = 4.0               # full alignment from the edge to here (the path is traced 3 c, blended to 4 c)
    te_far::Symbol = :freeze             # :freeze (the offset's slope decays to 0 beyond te_hold) or :follow (the path)
    te_freeze_length::Float64 = 4.0      # the C2 slope decay of :freeze
    te_outlet_margin::Float64 = 5.0      # the decay reaches 0 this far before the outlet (outlet band untouched)
    te_plateau::Float64 = -1.0           # < 0: the base height (C+H) or 0 (sharp)
    te_width_min::Float64 = 0.5
    te_width_factor::Float64 = 16.0
    le_amplitude::Float64 = 1.0
    le_ramp_start::Float64 = 0.01
    le_ramp_end::Float64 = 0.25
    le_hold::Float64 = 1.0
    le_end::Float64 = 5.0
    le_plateau::Float64 = 0.02
    le_width::Float64 = 0.30
    # the wake path is moved to start exactly at the mesh's trailing-edge point; a start further away than this is refused
    # (a different section); with te_path_shift_relax > 0 the move fades C2 to zero over that arc length of the path (the
    # Gurney plate: the clean section's path moved from its trailing-edge midpoint to the plate base's midpoint)
    te_path_shift_max::Float64 = 1e-3
    te_path_shift_relax::Float64 = 0.0
end

params_dict(p::MorphParams) = Dict{String,Any}(string(f)=>(getfield(p, f) isa Symbol ? String(getfield(p, f)) : getfield(p, f))
                                               for f in fieldnames(MorphParams))

"MorphParams from a Dict of field names (unknown names refused), on top of `base`."
function params_from(d::AbstractDict; base::MorphParams=MorphParams())
    v = Dict{Symbol,Any}(f=>getfield(base, f) for f in fieldnames(MorphParams))
    for (k, x) in d
        s = Symbol(k); haskey(v, s) || throw(ArgumentError("unknown morph parameter $k"))
        v[s] = s === :te_far ? Symbol(x) : (fieldtype(MorphParams, s) === Bool ? Bool(x) : Float64(x))
    end
    MorphParams(; v...)
end

# ---- 1-D cubic Hermite (finite-difference slopes; linear extrapolation) ----------------------------------------------
struct Hermite1D
    x::Vector{Float64}; y::Vector{Float64}; m::Vector{Float64}
end
function Hermite1D(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    n = length(x); n >= 2 && n == length(y) || throw(ArgumentError("Hermite1D needs two or more matching points"))
    xs = Float64.(x); ys = Float64.(y)
    all(diff(xs) .> 0) || throw(ArgumentError("Hermite1D abscissae must increase strictly"))
    m = similar(xs)
    m[1] = (ys[2] - ys[1])/(xs[2] - xs[1]); m[n] = (ys[n] - ys[n-1])/(xs[n] - xs[n-1])
    for k in 2:n-1
        h0 = xs[k] - xs[k-1]; h1 = xs[k+1] - xs[k]
        d0 = (ys[k] - ys[k-1])/h0; d1 = (ys[k+1] - ys[k])/h1
        m[k] = (h1*d0 + h0*d1)/(h0 + h1)
    end
    Hermite1D(xs, ys, m)
end
function (h::Hermite1D)(q::Float64)
    x = h.x; n = length(x)
    q <= x[1] && return h.y[1] + h.m[1]*(q - x[1])
    q >= x[n] && return h.y[n] + h.m[n]*(q - x[n])
    k = searchsortedlast(x, q); k = clamp(k, 1, n - 1)
    dx = x[k+1] - x[k]; t = (q - x[k])/dx
    t2 = t*t; t3 = t2*t
    (2t3 - 3t2 + 1)*h.y[k] + (t3 - 2t2 + t)*dx*h.m[k] + (-2t3 + 3t2)*h.y[k+1] + (t3 - t2)*dx*h.m[k+1]
end

"Arc-length parametrised planar curve (Hermite per component)."
struct ArcCurve
    L::Float64; fx::Hermite1D; fy::Hermite1D; x::Vector{Float64}; y::Vector{Float64}; s::Vector{Float64}
end
function ArcCurve(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    xs = Float64[x[1]]; ys = Float64[y[1]]; ss = [0.0]
    for k in 2:length(x)
        d = hypot(x[k] - xs[end], y[k] - ys[end]); d > 1e-14 || continue
        push!(xs, x[k]); push!(ys, y[k]); push!(ss, ss[end] + d)
    end
    ArcCurve(ss[end], Hermite1D(ss, xs), Hermite1D(ss, ys), xs, ys, ss)
end

"Closest point of curve `c` to (px, py), searching only the arc window [lo, hi] (the curves compared stay close)."
function closest_point(c::ArcCurve, px, py, lo, hi)
    k0 = max(1, searchsortedlast(c.s, lo)); k1 = min(length(c.s) - 1, searchsortedlast(c.s, hi))
    best = (Inf, c.x[k0], c.y[k0])
    for k in k0:k1
        ax, ay = c.x[k], c.y[k]; dx = c.x[k+1] - ax; dy = c.y[k+1] - ay; L2 = dx*dx + dy*dy
        t = L2 > 0 ? clamp(((px - ax)*dx + (py - ay)*dy)/L2, 0.0, 1.0) : 0.0
        qx = ax + t*dx; qy = ay + t*dy; d = hypot(px - qx, py - qy)
        d < best[1] && (best = (d, qx, qy))
    end
    (best[2], best[3])
end
(c::ArcCurve)(l::Float64) = (c.fx(l), c.fy(l))

"Distance from (px, py) to a polyline."
function polyline_distance(xs, ys, px, py)
    d = Inf
    for k in 1:length(xs)-1
        ax, ay = xs[k], ys[k]; dx = xs[k+1] - ax; dy = ys[k+1] - ay; L2 = dx*dx + dy*dy
        t = L2 > 0 ? clamp(((px - ax)*dx + (py - ay)*dy)/L2, 0.0, 1.0) : 0.0
        d = min(d, hypot(px - ax - t*dx, py - ay - t*dy))
    end
    d
end

# ---- the trailing-edge frame ------------------------------------------------------------------------------------------
"""
    te_frame(X, Y, NIW, NIF, Xs, Ys) -> NamedTuple

P0 (the trailing-edge point), e_s (towards the outlet along the current cut), e_n, the current cut centreline points
(from the edge to the outlet), and the base height (0 for a sharp mesh).
"""
function te_frame(X, Y, NIW::Int, NIF::Int, Xs, Ys)
    NI = size(X, 1) - 1
    if Xs === nothing
        P0 = (X[NIW+1, 1], Y[NIW+1, 1]); far = (X[1, 1], Y[1, 1])
        cx = [X[NIW+2-k, 1] for k in 1:NIW+1]; cy = [Y[NIW+2-k, 1] for k in 1:NIW+1]
        gap = 0.0
    else
        P0 = (0.5*(X[NIW+1, 1] + X[NIW+NIF+1, 1]), 0.5*(Y[NIW+1, 1] + Y[NIW+NIF+1, 1]))
        far = (0.5*(X[1, 1] + X[NI+1, 1]), 0.5*(Y[1, 1] + Y[NI+1, 1]))
        cx = [0.5*(Xs[k, 1] + Xs[k, end]) for k in 1:NIW+1]; cy = [0.5*(Ys[k, 1] + Ys[k, end]) for k in 1:NIW+1]
        gap = hypot(X[NIW+NIF+1, 1] - X[NIW+1, 1], Y[NIW+NIF+1, 1] - Y[NIW+1, 1])
    end
    ex = far[1] - P0[1]; ey = far[2] - P0[2]; el = hypot(ex, ey)
    es = (ex/el, ey/el); en = (-es[2], es[1])
    (P0=P0, es=es, en=en, cut_x=cx, cut_y=cy, gap=gap)
end

@inline _s(fr, x, y) = (x - fr.P0[1])*fr.es[1] + (y - fr.P0[2])*fr.es[2]
@inline _eta(fr, x, y) = (x - fr.P0[1])*fr.en[1] + (y - fr.P0[2])*fr.en[2]

"The wake path moved to start exactly at the mesh's trailing-edge point P0 (rigidly, or fading over te_path_shift_relax)."
function _shifted_path(fr, wake, p::MorphParams)
    wx = Float64.(wake["x"]); wy = Float64.(wake["y"])
    shift = (fr.P0[1] - wx[1], fr.P0[2] - wy[1])
    hypot(shift...) <= p.te_path_shift_max || throw(ArgumentError("the wake path starts $(hypot(shift...)) c from the mesh's trailing edge (limit $(p.te_path_shift_max) c): different sections"))
    if p.te_path_shift_relax > 0
        sarc = 0.0; ox = copy(wx); oy = copy(wy)
        for k in eachindex(wx)
            k > 1 && (sarc += hypot(ox[k] - ox[k-1], oy[k] - oy[k-1]))
            f = 1.0 - smooth5(sarc/p.te_path_shift_relax)
            wx[k] += f*shift[1]; wy[k] += f*shift[2]
        end
        wx[1] = fr.P0[1]; wy[1] = fr.P0[2]
    else
        wx .+= shift[1]; wy .+= shift[2]
    end
    wx, wy, shift
end

"The wake path the morph aligns the mesh with (moved to the mesh's trailing-edge point), for drawing."
function aligned_wake_path(X, Y, NIW::Int, NIF::Int, Xs, Ys, wake; params::MorphParams=MorphParams())
    wx, wy, _ = _shifted_path(te_frame(X, Y, NIW, NIF, Xs, Ys), wake, params)
    (x=wx, y=wy)
end

# ---- the trailing-edge morph ------------------------------------------------------------------------------------------
function _te_field(X, Y, NIW, NIF, Xs, Ys, wake, p::MorphParams)
    NI = size(X, 1) - 1; NJ = size(X, 2) - 1
    p.te_far in (:freeze, :follow) || throw(ArgumentError("te_far must be :freeze or :follow"))
    fr = te_frame(X, Y, NIW, NIF, Xs, Ys)
    cs = [_s(fr, fr.cut_x[k], fr.cut_y[k]) for k in eachindex(fr.cut_x)]
    ce = [_eta(fr, fr.cut_x[k], fr.cut_y[k]) for k in eachindex(fr.cut_x)]
    all(diff(cs) .> 0) || throw(ArgumentError("the current wake cut is not monotonic along its own outlet direction"))
    eref = Hermite1D(cs, ce)
    wx, wy, shift = _shifted_path(fr, wake, p)
    ps = [_s(fr, wx[k], wy[k]) for k in eachindex(wx)]; pe = [_eta(fr, wx[k], wy[k]) for k in eachindex(wx)]
    all(diff(ps) .> -1e-9) || throw(ArgumentError("the wake path turns back against the cut direction (not monotonic in s): not alignable by this morph"))
    keep = [1]
    for k in 2:length(ps); ps[k] > ps[keep[end]] + 1e-12 && push!(keep, k); end
    epath = Hermite1D(ps[keep], pe[keep])
    s_surf = 0.0
    for i in NIW+1:NIW+NIF+1; s_surf = max(s_surf, _s(fr, X[i, 1], Y[i, 1])); end
    if Xs !== nothing
        for j in axes(Xs, 2); s_surf = max(s_surf, _s(fr, Xs[1, j], Ys[1, j])); end
    end
    s_out = Inf
    for j in 1:NJ+1; s_out = min(s_out, _s(fr, X[1, j], Y[1, j]), _s(fr, X[NI+1, j], Y[NI+1, j])); end
    if Xs !== nothing
        for j in axes(Xs, 2); s_out = min(s_out, _s(fr, Xs[end, j], Ys[end, j])); end
    end
    s_hold = p.te_hold; s_end = s_out - p.te_outlet_margin
    s_end > s_hold + 1 || throw(ArgumentError("the outlet ($(s_out) c behind the edge) is too close for the trailing-edge morph"))
    D(s) = epath(s) - eref(s)
    dD = (D(s_hold + 1e-3) - D(s_hold - 1e-3))/2e-3
    Deff(s) = (p.te_far === :follow || s <= s_hold) ? D(s) : D(s_hold) + dD*_phi(s - s_hold, p.te_freeze_length)
    Lm = p.te_start_ramp; Lc = p.te_lag_relax
    # J(s) = int_{s_surf}^{s} Deff(u) m'(u) du on [s_surf, s_surf + Lm], composite Simpson, tabulated
    nJ = 4000; uu = range(s_surf, s_surf + Lm; length=nJ + 1); fJ = [Deff(u)*_dsmooth5((u - s_surf)/Lm)/Lm for u in uu]
    Jt = zeros(nJ + 1); h = Lm/nJ
    for k in 1:nJ; Jt[k+1] = Jt[k] + 0.5h*(fJ[k] + fJ[k+1]); end    # trapezoid at 4,000 panels (error ~1e-10 of D)
    Jh = Hermite1D(collect(uu), Jt); Jend = Jt[end]
    J(s) = s <= s_surf ? 0.0 : s >= s_surf + Lm ? Jend : Jh(s)
    E(s) = s <= s_hold ? 1.0 : 1.0 - smooth5((s - s_hold)/(s_end - s_hold))
    core(s) = Deff(s)*smooth5((s - s_surf)/Lm) - J(s) + Jend*smooth5((s - s_surf)/Lc)
    delta(s) = (s <= s_surf || s >= s_end) ? 0.0 : p.te_amplitude*E(s)*core(s)
    # the width follows the UNIT-amplitude offset, so a reduced amplitude reduces the distortion in proportion
    smax = 0.0
    for q in range(s_surf, s_end; length=20001); smax = max(smax, abs(E(q)*core(q))); end
    W = max(p.te_width_min, p.te_width_factor*smax)
    smax *= abs(p.te_amplitude)
    P = p.te_plateau < 0 ? fr.gap : p.te_plateau
    room = Inf
    for i in vcat(1:NIW+1, NIW+NIF+1:NI+1)
        x = X[i, NJ+1]; y = Y[i, NJ+1]; s = _s(fr, x, y)
        s_surf < s < s_end && (room = min(room, abs(_eta(fr, x, y) - eref(s))))
    end
    wcap = isfinite(room) ? 0.8room - P : Inf
    capped = W > wcap
    capped && (W = wcap)
    W > 0 || throw(ArgumentError("no room for the trailing-edge decay width"))
    function disp(x::Float64, y::Float64)
        s = _s(fr, x, y)
        (s <= s_surf || s >= s_end) && return (0.0, 0.0)
        er = abs(_eta(fr, x, y) - eref(s))
        w = er <= P ? 1.0 : 1.0 - smooth5((er - P)/W)
        w == 0.0 && return (0.0, 0.0)
        d = delta(s)*w
        (d*fr.en[1], d*fr.en[2])
    end
    info = Dict{String,Any}("te_point"=>[fr.P0...], "cut_direction"=>[fr.es...], "base_height"=>fr.gap,
        "path_start_shift"=>hypot(shift...), "path_shift_relax"=>p.te_path_shift_relax, "s_surface_max"=>s_surf, "s_outlet"=>s_out, "s_hold"=>s_hold,
        "s_decay_end"=>s_end, "max_offset"=>smax, "width"=>W, "plateau"=>P, "width_capped_by_outer_boundary"=>capped,
        "outer_room"=>room, "turn_lag_removed"=>Jend, "offset_slope_at_hold"=>dD,
        "offset_at"=>Dict(string(q)=>delta(Float64(q)) for q in (0.01, 0.05, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0, 16.0)),
        "far_policy"=>String(p.te_far))
    (disp=disp, info=info, frame=fr, epath=epath, eref=eref, delta=delta, s_surf=s_surf)
end

# ---- the leading-edge morph -------------------------------------------------------------------------------------------
function _le_field(X, Y, NIW, NIF, stag, p::MorphParams)
    ifs = NIW+1:NIW+NIF+1
    sig = zeros(length(ifs))
    for k in 2:length(ifs)
        i = ifs[k]; sig[k] = sig[k-1] + hypot(X[i, 1] - X[i-1, 1], Y[i, 1] - Y[i-1, 1])
    end
    sx = Float64.(stag["x"]); sy = Float64.(stag["y"])
    best = (Inf, 0.0)
    for k in 1:length(ifs)-1
        a = ifs[k]; b = ifs[k+1]
        dx = X[b, 1] - X[a, 1]; dy = Y[b, 1] - Y[a, 1]; L2 = dx*dx + dy*dy
        t = clamp(((sx[1] - X[a, 1])*dx + (sy[1] - Y[a, 1])*dy)/L2, 0.0, 1.0)
        d = hypot(X[a, 1] + t*dx - sx[1], Y[a, 1] + t*dy - sy[1])
        d < best[1] && (best = (d, sig[k] + t*(sig[k+1] - sig[k])))
    end
    best[1] <= 1e-3 || throw(ArgumentError("the stagnation point lies $(best[1]) c from the mesh wall (limit 1e-3 c): different sections"))
    sig_st = best[2]
    ks = argmin(abs.(sig .- sig_st)); is = ifs[ks]
    shift = (X[is, 1] - sx[1], Y[is, 1] - sy[1])
    st = ArcCurve(sx .+ shift[1], sy .+ shift[2])
    cs = ArcCurve(view(X, is, :), view(Y, is, :))
    lh = p.le_hold
    st.L > lh + 1 && cs.L > p.le_end || throw(ArgumentError("the stagnation streamline or the stagnation column is shorter than the leading-edge morph"))
    # Delta(l): from the column point at arc length l to the CLOSEST point of the streamline (a motion nearly normal to the
    # column, so the column's wall-normal intervals change only at second order); tabulated on 2,001 stations to le_hold
    # and frozen beyond it
    lg = collect(range(0.0, lh; length=2001))
    dxg = similar(lg); dyg = similar(lg)
    for (k, l) in enumerate(lg)
        cx, cy = cs(l); qx, qy = closest_point(st, cx, cy, max(0.0, l - 1.0), l + 1.0)
        dxg[k] = qx - cx; dyg[k] = qy - cy
    end
    dxg[1] = 0.0; dyg[1] = 0.0                       # the column starts on the (shifted) stagnation point
    hx = Hermite1D(lg, dxg); hy = Hermite1D(lg, dyg)
    Delta(l) = (lq = min(l, lh); (hx(lq), hy(lq)))
    # the SIGNED offset normal to the stagnation column (left normal of its j-direction): every column in the window is
    # moved sideways, normal to itself, by T R times this offset -- a local rotation of the fan, which keeps each column's
    # wall-normal intervals to second order (translating the fan by one vector moved nodes along the neighbouring columns)
    # The offset is measured ALONG the column's normal line to where that line crosses the streamline (so the column's
    # node lands on it); the closest-point projection is the fallback where the line misses the arc window.
    hh = 1e-6
    dng = zeros(length(lg))
    for (k, l) in enumerate(lg)
        a = cs(max(l - hh, 0.0)); b = cs(min(l + hh, cs.L)); tx = b[1] - a[1]; ty = b[2] - a[2]; tn = hypot(tx, ty)
        nx = -ty/tn; ny = tx/tn; cx, cy = cs(l)
        dng[k] = dxg[k]*nx + dyg[k]*ny
        k == 1 && (dng[k] = 0.0; continue)
        best = Inf; sbest = 0.0
        k0 = max(1, searchsortedlast(st.s, max(0.0, l - 1.0))); k1 = min(length(st.s) - 1, searchsortedlast(st.s, l + 1.0))
        for q in k0:k1
            ax, ay = st.x[q], st.y[q]; ex = st.x[q+1] - ax; ey = st.y[q+1] - ay
            den = nx*ey - ny*ex; abs(den) < 1e-14 && continue
            tau = ((ax - cx)*ey - (ay - cy)*ex)/den          # C + tau n = A + u E
            u = ((ax - cx)*ny - (ay - cy)*nx)/den
            0.0 <= u <= 1.0 && abs(tau) < abs(best) && (best = tau; sbest = st.s[q] + u*(st.s[q+1] - st.s[q]))
        end
        if isfinite(best)
            # Newton on the streamline's own cubic Hermite curve (the representation the column uses): find s with
            # (st(s) - C) . t = 0, then tau = (st(s) - C) . n. Below 1e-13 c the offset is round-off and is set to 0,
            # so a streamline that IS the column leaves it bitwise unchanged.
            tx1 = tx/tn; ty1 = ty/tn; sq = sbest
            for _ in 1:8
                p0 = st(sq); f = (p0[1] - cx)*tx1 + (p0[2] - cy)*ty1
                pa = st(sq + 1e-7); pb = st(sq - 1e-7); fp = ((pa[1] - pb[1])*tx1 + (pa[2] - pb[2])*ty1)/2e-7
                abs(fp) < 1e-12 && break
                ds = f/fp; sq -= ds; abs(ds) < 1e-15 && break
            end
            pq = st(sq); tau = (pq[1] - cx)*nx + (pq[2] - cy)*ny
            dng[k] = abs(tau) < 1e-13 ? 0.0 : tau
        end
    end
    hn = Hermite1D(lg, dng)
    Dperp(l) = hn(min(l, lh))
    R(l) = l <= p.le_ramp_start ? 0.0 : l < p.le_ramp_end ? smooth5((l - p.le_ramp_start)/(p.le_ramp_end - p.le_ramp_start)) :
           l <= lh ? 1.0 : 1.0 - smooth5((l - lh)/(p.le_end - lh))
    T(dsig) = (a = abs(dsig); a <= p.le_plateau ? 1.0 : 1.0 - smooth5((a - p.le_plateau)/p.le_width))
    (sig[ks] - sig[1] > p.le_plateau + p.le_width + 0.05 && sig[end] - sig[ks] > p.le_plateau + p.le_width + 0.05) ||
        throw(ArgumentError("the leading-edge window reaches the trailing edge"))
    before = Dict{String,Any}(string(l)=>polyline_distance(cs.x, cs.y, st(Float64(l))...) for l in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0))
    info = Dict{String,Any}("stagnation_point"=>[sx[1], sy[1]], "stagnation_wall_arc"=>sig_st, "stagnation_column"=>is,
        "column_wall_offset"=>hypot(shift...), "wall_spacing_at_column"=>hypot(X[is+1, 1] - X[is, 1], Y[is+1, 1] - Y[is, 1]),
        "geometric_le_column"=>ifs[argmin(X[ifs, 1])], "streamline_distance_to_column_before"=>before,
        "max_arc_offset_ramp_end_to_hold"=>maximum(hypot(Delta(l)...) for l in range(p.le_ramp_end, lh; length=200)))
    (sig=sig, sig_st=sig_st, is=is, Delta=Delta, Dperp=Dperp, R=R, T=T, info=info, st=st)
end

# ---- the composed morph -----------------------------------------------------------------------------------------------
"""
    streamline_morph(X, Y, NIW, NIF, Xs, Ys, paths; params=MorphParams()) -> (X, Y, Xs, Ys, report)

New node arrays (inputs are not modified). `paths` holds "wake" and "stagnation_streamline" dicts with "x", "y" vectors
(a streamline file read back by `read_streamline_file`). `Xs, Ys` are `nothing` for a sharp C mesh.
"""
function streamline_morph(X::AbstractMatrix{Float64}, Y::AbstractMatrix{Float64}, NIW::Int, NIF::Int, Xs, Ys,
                          paths::AbstractDict; params::MorphParams=MorphParams())
    p = params
    NI = size(X, 1) - 1; NJ = size(X, 2) - 1
    NI == 2NIW + NIF && size(Y) == size(X) || throw(ArgumentError("inconsistent C block dimensions"))
    (Xs === nothing) == (Ys === nothing) || throw(ArgumentError("both or neither strip arrays"))
    Xs === nothing || (size(Xs, 1) == NIW + 1 && size(Ys) == size(Xs)) || throw(ArgumentError("inconsistent H block dimensions"))
    X2 = copy(X); Y2 = copy(Y)
    Xs2 = Xs === nothing ? nothing : copy(Xs); Ys2 = Ys === nothing ? nothing : copy(Ys)
    report = Dict{String,Any}("tag"=>MORPH_TAG, "params"=>params_dict(p))
    haskey(paths, "sha256") && (report["streamline_sha256"] = paths["sha256"])
    moved = falses(NI+1, NJ+1); te_max = 0.0; le_max = 0.0; overlap = 0
    if p.te
        F = _te_field(X, Y, NIW, NIF, Xs, Ys, paths["wake"], p)
        for j in 1:NJ+1, i in 1:NI+1
            (j == 1 && NIW+1 <= i <= NIW+NIF+1) && continue          # the airfoil surface, never moved
            dx, dy = F.disp(X[i, j], Y[i, j])
            (dx == 0.0 && dy == 0.0) && continue
            j == NJ+1 && throw(ArgumentError("the trailing-edge morph reached the outer boundary"))
            X2[i, j] += dx; Y2[i, j] += dy; moved[i, j] = true; te_max = max(te_max, hypot(dx, dy))
        end
        if Xs !== nothing
            # the legs ARE the C row-1 nodes (bitwise); each column whose legs moved is rebuilt straight at its original
            # cross parameters; the base column and columns with unmoved legs keep their bits
            NS = size(Xs, 2) - 1
            for k in 2:NIW+1
                il = NIW+2-k; iu = NIW+NIF+k
                xl, yl, xu, yu = Xs[k, 1], Ys[k, 1], Xs[k, end], Ys[k, end]
                xl2, yl2, xu2, yu2 = X2[il, 1], Y2[il, 1], X2[iu, 1], Y2[iu, 1]
                (xl2 === xl && yl2 === yl && xu2 === xu && yu2 === yu) && continue
                dx = xu - xl; dy = yu - yl; W2 = dx*dx + dy*dy
                Xs2[k, 1] = xl2; Ys2[k, 1] = yl2; Xs2[k, end] = xu2; Ys2[k, end] = yu2
                for j in 2:NS
                    t = ((Xs[k, j] - xl)*dx + (Ys[k, j] - yl)*dy)/W2
                    Xs2[k, j] = xl2 + t*(xu2 - xl2); Ys2[k, j] = yl2 + t*(yu2 - yl2)
                end
            end
        end
        report["te"] = F.info
        fr = F.frame
        cutx = Xs === nothing ? [X2[NIW+2-k, 1] for k in 1:NIW+1] : [0.5*(Xs2[k, 1] + Xs2[k, end]) for k in 1:NIW+1]
        cuty = Xs === nothing ? [Y2[NIW+2-k, 1] for k in 1:NIW+1] : [0.5*(Ys2[k, 1] + Ys2[k, end]) for k in 1:NIW+1]
        err = 0.0; err_before = 0.0; ang_after = 0.0; ang_before = 0.0
        lo = F.s_surf + p.te_lag_relax
        for k in eachindex(cutx)
            s = _s(fr, cutx[k], cuty[k]); (lo < s <= p.te_hold) || continue
            err = max(err, abs(_eta(fr, cutx[k], cuty[k]) - F.epath(s)))
            err_before = max(err_before, abs(_eta(fr, fr.cut_x[k], fr.cut_y[k]) - F.epath(s)))
            if k < length(cutx)
                pa = atan(F.epath(s + 1e-4) - F.epath(s - 1e-4), 2e-4)
                ca = atan(_eta(fr, cutx[k+1], cuty[k+1]) - _eta(fr, cutx[k], cuty[k]), _s(fr, cutx[k+1], cuty[k+1]) - s)
                cb = atan(_eta(fr, fr.cut_x[k+1], fr.cut_y[k+1]) - _eta(fr, fr.cut_x[k], fr.cut_y[k]), _s(fr, fr.cut_x[k+1], fr.cut_y[k+1]) - s)
                ang_after = max(ang_after, abs(rad2deg(ca - pa))); ang_before = max(ang_before, abs(rad2deg(cb - pa)))
            end
        end
        report["te"]["cut_minus_path_max_before"] = err_before
        report["te"]["cut_minus_path_max_after"] = err
        report["te"]["cut_angle_minus_path_angle_max_deg_before"] = ang_before
        report["te"]["cut_angle_minus_path_angle_max_deg_after"] = ang_after
        report["te"]["alignment_window"] = [lo, p.te_hold]
        report["te"]["max_displacement"] = te_max
    end
    if p.le
        L = _le_field(X, Y, NIW, NIF, paths["stagnation_streamline"], p)
        ifs = NIW+1:NIW+NIF+1
        for (k, i) in enumerate(ifs)
            t = L.T(L.sig[k] - L.sig_st); t == 0.0 && continue
            l = 0.0
            for j in 2:NJ+1
                l += hypot(X[i, j] - X[i, j-1], Y[i, j] - Y[i, j-1])
                r = L.R(l); r == 0.0 && (l > p.le_end ? break : continue)
                j == NJ+1 && throw(ArgumentError("the leading-edge morph reached the outer boundary"))
                tx = X[i, j+1] - X[i, j-1]; ty = Y[i, j+1] - Y[i, j-1]; tn = hypot(tx, ty)
                d = p.le_amplitude*t*r*L.Dperp(l)
                moved[i, j] && (overlap += 1)
                X2[i, j] += d*(-ty/tn); Y2[i, j] += d*(tx/tn); le_max = max(le_max, abs(d))
            end
        end
        report["le"] = L.info
        report["le"]["streamline_distance_to_column_after"] = Dict{String,Any}(string(l)=>
            polyline_distance(view(X2, L.is, :), view(Y2, L.is, :), L.st(Float64(l))...) for l in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0))
        report["le"]["max_displacement"] = le_max
    end
    report["nodes_moved_by_both"] = overlap
    # invariants: the surface, the outer boundary, the interfaces
    for i in NIW+1:NIW+NIF+1
        (X2[i, 1] === X[i, 1] && Y2[i, 1] === Y[i, 1]) || error("a surface node moved")
    end
    for i in 1:NI+1
        (X2[i, NJ+1] === X[i, NJ+1] && Y2[i, NJ+1] === Y[i, NJ+1]) || error("an outer-boundary node moved")
    end
    for j in 1:NJ+1
        (X2[1, j] === X[1, j] && Y2[1, j] === Y[1, j] && X2[NI+1, j] === X[NI+1, j] && Y2[NI+1, j] === Y[NI+1, j]) ||
            error("an outlet node moved")
    end
    if Xs !== nothing
        for j in axes(Xs, 2)
            (Xs2[1, j] === Xs[1, j] && Ys2[1, j] === Ys[1, j]) || error("a base node moved")
            (Xs2[end, j] === Xs[end, j] && Ys2[end, j] === Ys[end, j]) || error("an H-block outlet node moved")
        end
        lower = all(Xs2[k, 1] === X2[NIW+2-k, 1] && Ys2[k, 1] === Y2[NIW+2-k, 1] for k in 1:NIW+1)
        upper = all(Xs2[k, end] === X2[NIW+NIF+k, 1] && Ys2[k, end] === Y2[NIW+NIF+k, 1] for k in 1:NIW+1)
        lower && upper || error("the morph broke the bitwise C/H interface identity")
        report["interfaces_bitwise"] = true
    else
        report["cut_rows_bitwise"] = all(X2[k, 1] === X2[NI+2-k, 1] && Y2[k, 1] === Y2[NI+2-k, 1] for k in 1:NIW+1)
        report["cut_rows_bitwise"] || error("the morph separated the sharp mesh's two cut rows")
    end
    fs(Xa, Ya, i) = hypot(Xa[i, 2] - Xa[i, 1], Ya[i, 2] - Ya[i, 1])
    report["foil_first_spacing_max_rel_change"] = maximum(abs(fs(X2, Y2, i)/fs(X, Y, i) - 1) for i in NIW+1:NIW+NIF+1)
    report["wake_first_spacing_max_rel_change"] = maximum(abs(fs(X2, Y2, i)/fs(X, Y, i) - 1) for i in vcat(1:NIW, NIW+NIF+2:NI+1))
    report["nodes_moved"] = count(moved) + (p.le ? 0 : 0)
    report["morphed_sha256"] = morph_hash(X2, Y2, Xs2, Ys2)
    report["source_sha256"] = morph_hash(X, Y, Xs, Ys)
    (X2, Y2, Xs2, Ys2, report)
end

"SHA-256 of the node bytes (C x, C y, then H x, H y when present), column-major Float64: the morph receipt."
function morph_hash(X, Y, Xs, Ys)
    io = IOBuffer()
    for A in (Xs === nothing ? (X, Y) : (X, Y, Xs, Ys)); write(io, vec(Float64.(A))); end
    bytes2hex(sha256(take!(io)))
end

# ---- the path file without JXfoil (a solver worker reads it) ---------------------------------------------------------
const STREAMLINE_FORMAT = "airfoil_lab_streamlines_v1"

"The path file's SHA-256 (the same bytes as streamline_alignment.jl's streamline_hash; a test pins the equality)."
function streamline_paths_hash(d::AbstractDict)
    io = IOBuffer()
    write(io, STREAMLINE_FORMAT)
    write(io, Float64(d["alpha_deg"]))
    for key in ("wake", "stagnation_streamline"), f in ("x", "y", "s")
        for v in d[key][f]; write(io, Float64(v)); end
    end
    for v in d["te_point"]; write(io, Float64(v)); end
    bytes2hex(sha256(take!(io)))
end

"Read a path file (TOML) and refuse it unless its arrays hash to its recorded SHA-256 (and to `expected`, when given)."
function read_streamline_paths(path::AbstractString; expected::Union{Nothing,AbstractString}=nothing)
    d = TOML.parsefile(path)
    get(d, "format", "") == STREAMLINE_FORMAT || throw(ArgumentError("not a $(STREAMLINE_FORMAT) file: $path"))
    for key in ("wake", "stagnation_streamline"), f in ("x", "y", "s"); d[key][f] = Float64.(d[key][f]); end
    d["te_point"] = Float64.(d["te_point"])
    h = streamline_paths_hash(d)
    h == d["sha256"] || throw(ArgumentError("streamline path file hash mismatch: $path"))
    expected === nothing || h == expected || throw(ArgumentError("streamline path file $path is not the one requested ($h, expected $expected)"))
    d
end

# ---- convexity and admission -------------------------------------------------------------------------------------------
"Corners of a block whose signed corner sine is not positive (inverted or non-convex cells), and the smallest sine."
function bad_corners(X, Y)
    NI, NJ = size(X) .- 1; nbad = 0; worst = Inf
    for j in 1:NJ, i in 1:NI
        px = (X[i,j], X[i+1,j], X[i+1,j+1], X[i,j+1]); py = (Y[i,j], Y[i+1,j], Y[i+1,j+1], Y[i,j+1])
        for k in 1:4
            a = mod1(k-1, 4); c = mod1(k+1, 4)
            e1x = px[c] - px[k]; e1y = py[c] - py[k]; e2x = px[a] - px[k]; e2y = py[a] - py[k]
            sn = (e1x*e2y - e1y*e2x)/(hypot(e1x, e1y)*hypot(e2x, e2y))
            worst = min(worst, sn); sn <= 0 && (nbad += 1)
        end
    end
    (count=nbad, min_sine=worst)
end

const LOWER_BOUND_GATES = ("G2i", "G2c'", "G2l'")      # value >= limit; every other gate is value <= limit
_gate_worse(name, before, after) = name in LOWER_BOUND_GATES ? after < before*(1 - 1e-9) : after > before*(1 + 1e-9)

"""
    admit_morph(X, Y, NIW, NIF, Xs, Ys, paths; quality, rule=:strict, params=MorphParams()) -> (X, Y, Xs, Ys, report)

The morph with an admission rule. `quality(X, Y, Xs, Ys)` returns a Dict with "gates" (a vector of Dicts with "gate",
"value", "limit", "pass") and, optionally, "postcheck_error" (a String). Both rules refuse a post-check error, a new
non-convex corner (either block) and any change of the wall first spacing (1e-9 relative); the morph itself already errors
on a moved surface/outer/outlet node or a broken C/H interface.
- `:strict`: every gate that passed before passes after, and a gate that failed before gets no worse (the design note's rule).
- `:demo`: gates are only REPORTED (each gate's value before and after, and whether it got worse); nothing is refused for
  them. Development use only: such a mesh is labelled "demo admission".
The loop (deterministic): the default parameters; the decay widths x1.5, x2.25; at the widest widths each part (TE, LE)
refused on its own gets the largest admitted amplitude by bisection (7 halvings); the parts combined and scaled by 0.8 until
admitted; else the unchanged mesh (amplitude 0). report["admission"] records every attempt and the gate changes.
"""
function admit_morph(X, Y, NIW::Int, NIF::Int, Xs, Ys, paths::AbstractDict; quality, rule::Symbol=:strict,
                     params::MorphParams=MorphParams())
    rule in (:strict, :demo) || throw(ArgumentError("admission rule must be strict or demo"))
    corners(Xa, Ya, Xsa, Ysa) = (c = bad_corners(Xa, Ya); Xsa === nothing ? c :
        (h = bad_corners(Xsa, Ysa); (count=c.count + h.count, min_sine=min(c.min_sine, h.min_sine))))
    q0 = quality(X, Y, Xs, Ys); c0 = corners(X, Y, Xs, Ys)
    g0 = Dict(g["gate"]=>g for g in q0["gates"])
    log = Any[]
    function check(q, c, rep)
        haskey(q, "postcheck_error") && return "post-check: " * q["postcheck_error"]
        c.count > c0.count && return "$(c.count - c0.count) new non-convex corners"
        rep["foil_first_spacing_max_rel_change"] > 1e-9 && return "the wall first spacing changed"
        rule === :demo && return ""
        bad = String[]
        for g in q["gates"]
            b = get(g0, g["gate"], nothing); b === nothing && continue
            if b["pass"]
                g["pass"] || push!(bad, "$(g["gate"]) now fails ($(round(g["value"]; sigdigits=6)) against $(g["limit"]))")
            elseif _gate_worse(g["gate"], b["value"], g["value"])
                push!(bad, "$(g["gate"]) already failing, worse ($(round(b["value"]; sigdigits=6)) -> $(round(g["value"]; sigdigits=6)))")
            end
        end
        join(bad, "; ")
    end
    function attempt(p, label)
        X2, Y2, Xs2, Ys2, rep = streamline_morph(X, Y, NIW, NIF, Xs, Ys, paths; params=p)
        q = quality(X2, Y2, Xs2, Ys2); c = corners(X2, Y2, Xs2, Ys2)
        why = check(q, c, rep)
        push!(log, Dict{String,Any}("attempt"=>length(log) + 1, "stage"=>label, "te_amplitude"=>p.te_amplitude,
            "le_amplitude"=>p.le_amplitude, "te_width_factor"=>p.te_width_factor, "le_width"=>p.le_width,
            "admitted"=>isempty(why), "reason"=>why))
        (X=X2, Y=Y2, Xs=Xs2, Ys=Ys2, rep=rep, q=q, c=c, p=p, ok=isempty(why))
    end
    widen(f) = (d = Dict{Symbol,Any}(k=>getfield(params, k) for k in fieldnames(MorphParams));
                d[:te_width_factor] = f*params.te_width_factor; d[:le_width] = min(0.45, f*params.le_width);
                d[:le_ramp_end] = min(0.5, f*params.le_ramp_end); d[:le_end] = f*params.le_end; MorphParams(; d...))
    amp(p; te=p.te_amplitude, le=p.le_amplitude) = (d = Dict{Symbol,Any}(k=>getfield(p, k) for k in fieldnames(MorphParams));
                d[:te_amplitude] = te; d[:le_amplitude] = le; MorphParams(; d...))
    r = nothing; p = params
    for (f, label) in ((1.0, "default"), (1.5, "widths x1.5"), (2.25, "widths x2.25"))
        p = widen(f); r = attempt(p, label); r.ok && break
    end
    if !r.ok
        a = Dict{Symbol,Float64}()
        for part in (:te, :le)
            solo = part === :te ? amp(p; le=0.0) : amp(p; te=0.0)
            r1 = attempt(solo, "$(part) alone, amplitude 1")
            if r1.ok; a[part] = 1.0; continue; end
            lo, hi = 0.0, 1.0
            for _ in 1:7
                m = 0.5(lo + hi)
                ra = attempt(part === :te ? amp(solo; te=m) : amp(solo; le=m), "$(part) alone, bisection $(m)")
                ra.ok ? (lo = m) : (hi = m)
            end
            a[part] = lo
        end
        scale = 1.0; r = nothing
        for _ in 0:6
            (a[:te] > 0 || a[:le] > 0) || break
            r = attempt(amp(p; te=scale*a[:te], le=scale*a[:le]), "combined, scale $(scale)")
            r.ok && break
            scale *= 0.8
        end
        (r === nothing || !r.ok) && (r = attempt(amp(p; te=0.0, le=0.0), "unchanged mesh (amplitude 0)"))
    end
    gates = Any[]
    for g in r.q["gates"]
        b = get(g0, g["gate"], nothing); b === nothing && continue
        push!(gates, Dict{String,Any}("gate"=>g["gate"], "limit"=>g["limit"], "before"=>b["value"], "after"=>g["value"],
            "pass_before"=>b["pass"], "pass_after"=>g["pass"], "worse"=>_gate_worse(g["gate"], b["value"], g["value"])))
    end
    rep = r.rep
    rep["admission"] = Dict{String,Any}("rule"=>String(rule), "stage"=>log[end]["stage"], "te_amplitude"=>r.p.te_amplitude,
        "le_amplitude"=>r.p.le_amplitude, "attempts"=>log, "gate_changes"=>gates,
        "gates_worse"=>[g["gate"] for g in gates if g["worse"]], "gates_newly_failing"=>[g["gate"] for g in gates if g["pass_before"] && !g["pass_after"]],
        "non_convex_corners_before"=>c0.count, "non_convex_corners_after"=>r.c.count,
        "min_corner_sine_before"=>c0.min_sine, "min_corner_sine_after"=>r.c.min_sine,
        "label"=>rule === :demo ? "DEMO ADMISSION (development): convex cells, conforming C/H interfaces and an unchanged wall first spacing are required; every gate's change is reported, none refuses" :
                                  "strict admission: every gate that passed still passes, and no failing gate got worse")
    rep["params_used"] = params_dict(r.p)
    (r.X, r.Y, r.Xs, r.Ys, rep)
end

end # module
