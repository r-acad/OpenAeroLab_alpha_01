# RETIRED 28 Oct 2026: the plate-INSERTION mesh (the reshaped-mesh Gurney flap) is no longer used by the app. The Gurney flap is a
# baffle in the section's own C+H mesh (PBCav PB_BAFFLE; lowspeed_ch_backend.jl). This file is not included by cfd_backend.jl any
# more; it stays here only so that the archived validation scripts under validation/gurney_demo_20261023 and
# validation/gurney_smooth_20261024 still load it. Do not use it for new work.
# Gurney flap C + H mesh (23 Oct 2026): the plate inserted into a clean conforming C + H mesh by a LOCAL rebuild.
# Pure Julia Base: node arrays in, node arrays out. The compressible app path (ch_mesh.jl) and the low-speed PBCav worker
# (lowspeed_ch_worker.jl, in the snapshot's environment) both include this file; neither version 1's generator nor PBCav
# is changed, and with the flap off nothing here runs.
#
# The plate (gurney.jl): zero thickness, on the trailing-edge station x = x_TE, from the lower trailing-edge point down
# h_l and/or from the upper one up h_u. Topology (the user's design): the C block's wall row runs
#     lower outlet -> lower leg -> lower TIP -> up the plate's front face -> lower ROOT -> lower surface -> ... (mirror)
# so each plate front face is a run of C wall faces between two EXACT corner nodes (tip and root; nothing is splined
# across them), and the H block's upstream wall (column 1) is the straight base x = x_TE from the lower end to the upper
# end of the C wall row: the plate's back face(s) plus the native gap. The plate is the line itself: C upstream, H behind.
#
# Construction, per plate side (the upper side is the lower one's mirror image), from a clean C + H mesh of the same
# section (its native base, built by the backend's own generator):
#   1. the plate's front-face nodes: a two-sided distribution from the leg's first spacing at the tip to the foil's
#      first spacing at the root, adjacent ratios <= 1.15;
#   2. the lower leg is the clean leg translated by -h_l (the interface starts at the tip); the C columns below it carry the
#      translation rigidly through the frozen layer and let it decay to zero at the window's top row J_w;
#   3. a window of C columns around the corner (leg columns within L of the tip, the plate, foil columns within L of
#      the root) is rebuilt below row J_w: rows 2..J_f by offsets along wall normals blended through each corner over
#      R_b = 3 delta_f (the concave root's rays converge beyond the frozen depth delta_f <= h/6; the convex tip's rays fan),
#      rows J_f..J_w by Winslow's equations on the computational grid stretched as the clean corner column (so the frozen
#      layer's spacing continues), with the frozen row J_f, the clean row J_w and the window's two side columns fixed;
#   4. above J_w the clean rows are resampled over the window by a smooth index map that spreads the plate's extra
#      columns (bilinear within clean cells, so no fold can be introduced there);
#   5. the H block is rebuilt by transfinite interpolation between the two legs with one cross distribution, clustered
#      toward both legs (first interval the clean H block's), on the straight base.
# Every cell of both blocks is then required strictly convex (the solvers' own test); otherwise the build refuses.
module AirfoilGurneyMesh

using Printf

export gurney_plate_mesh, gurney_h_block

const RATIO = 1.15                 # adjacent-interval ratio of the plate faces and the base (G2f/G2k limit 1.2)

_smooth(t) = (t = clamp(t, 0.0, 1.0); t*t*t*(t*(6t - 15) + 10))
_unit(v) = (n = hypot(v[1], v[2]); (v[1]/n, v[2]/n))
_rot90(t) = (-t[2], t[1])          # the C fluid lies on this side of the wall row (i increasing)
function _slerp(a, b, w)
    ta = atan(a[2], a[1]); tb = atan(b[2], b[1])
    d = rem2pi(tb - ta, RoundNearest)
    (cos(ta + w*d), sin(ta + w*d))
end

"""Intervals of a two-sided distribution of length L: first `a`, last `b`, adjacent ratio <= r, intervals <= dmax."""
function two_sided(L::Float64, a::Float64, b::Float64; r::Float64=RATIO, dmax::Float64=Inf, even::Bool=false)
    (L > 0 && a > 0 && b > 0) || throw(ArgumentError("two_sided needs positive lengths"))
    tot(n, cap) = sum(min(a*r^k, b*r^(n - 1 - k), cap) for k in 0:n-1)
    n = 1
    while tot(n, Inf) < L
        n += 1; n <= 20000 || throw(ArgumentError("two_sided: more than 20000 intervals"))
    end
    even && isodd(n) && (n += 1)
    while true
        lo, hi = 0.0, max(L, a, b)
        tot(n, hi) < L && (hi = 1e9)
        for _ in 1:200
            mid = 0.5(lo + hi); tot(n, mid) < L ? (lo = mid) : (hi = mid)
        end
        cap = hi
        if cap <= dmax || n > 20000
            s = [min(a*r^k, b*r^(n - 1 - k), cap) for k in 0:n-1]
            return s .* (L/sum(s))
        end
        n += even ? 2 : 1
    end
end

"""Vinokur's two-sided stretching (version 1's `_vinokur`, src/cmesh/cmesh_gen.jl, copied so this file stays standalone):
node fractions 0..1 of n intervals whose first and last intervals are s0 and s1 (fractions of the total)."""
function vinokur(n::Int, s0::Float64, s1::Float64)
    s0 = clamp(s0, 1e-12, 1 - 1e-12); s1 = clamp(s1, 1e-12, 1 - 1e-12)
    A = sqrt(s1/s0); B = 1/(n*sqrt(s0*s1))
    f(d) = B > 1 ? sinh(d)/d - B : sin(d)/d - B
    lo, hi = B > 1 ? (1e-8, 1.0) : (1e-8, pi - 1e-8)
    if B > 1
        while f(hi) < 0; hi *= 2; end
    end
    d = 0.0
    if abs(B - 1) >= 1e-6
        for _ in 1:200
            d = 0.5*(lo + hi)
            (f(d) > 0) == (B > 1) ? (hi = d) : (lo = d)
            hi - lo < 1e-14*max(1.0, hi) && break
        end
        d = 0.5*(lo + hi)
    end
    t = zeros(n + 1)
    for k in 0:n
        xi = k/n
        u = d == 0.0 ? xi : (B > 1 ? 0.5*(1 + tanh(d*(xi - 0.5))/tanh(d/2)) : 0.5*(1 + tan(d*(xi - 0.5))/tan(d/2)))
        t[k+1] = u/(A + (1 - A)*u)
    end
    t[1] = 0.0; t[end] = 1.0
    t
end

"""Intervals of a fixed count n over length L, first `a` and last `b`: min(a r^k, b r^(n-1-k)) with r solved."""
function fixed_two_sided(n::Int, L::Float64, a::Float64, b::Float64)
    tot(r) = sum(min(a*r^k, b*r^(n - 1 - k)) for k in 0:n-1)
    lo, hi = 1e-6, 1.0
    while tot(hi) < L; hi *= 2; hi < 1e6 || break; end
    for _ in 1:200; mid = 0.5(lo + hi); tot(mid) < L ? (lo = mid) : (hi = mid); end
    s = [min(a*hi^k, b*hi^(n - 1 - k)) for k in 0:n-1]
    s .* (L/sum(s))
end

_colfirst(X, Y, i) = _unit((X[i, 2] - X[i, 1], Y[i, 2] - Y[i, 1]))
function _coldist(X, Y, i)
    nj = size(X, 2); d = zeros(nj)
    for j in 2:nj; d[j] = d[j-1] + hypot(X[i, j] - X[i, j-1], Y[i, j] - Y[i, j-1]); end
    d
end

"""Strict convexity of every cell of a node block: the four corner cross products share one sign. Returns
(n_bad, worst normalised corner value, (i, j) of the worst)."""
function convexity(X::AbstractMatrix, Y::AbstractMatrix; irange=1:size(X, 1)-1, jrange=1:size(X, 2)-1)
    sgn = 0.0
    nbad = 0; worst = Inf; where = (0, 0)
    for j in jrange, i in irange
        px = (X[i, j], X[i+1, j], X[i+1, j+1], X[i, j+1]); py = (Y[i, j], Y[i+1, j], Y[i+1, j+1], Y[i, j+1])
        vmin = Inf; vmax = -Inf
        for k in 1:4
            kp = k == 4 ? 1 : k + 1; km = k == 1 ? 4 : k - 1
            ax = px[kp] - px[k]; ay = py[kp] - py[k]; bx = px[km] - px[k]; by = py[km] - py[k]
            v = (ax*by - ay*bx)/max(hypot(ax, ay)*hypot(bx, by), 1e-300)
            vmin = min(vmin, v); vmax = max(vmax, v)
        end
        sgn == 0.0 && (sgn = vmin > 0 ? 1.0 : -1.0)
        q = sgn > 0 ? vmin : -vmax
        q <= 0 && (nbad += 1)
        q < worst && (worst = q; where = (i, j))
    end
    (nbad, worst, where)
end

"""Winslow's equations on rows j0..j1 (interior j0+1..j1-1) and columns i0..i1 (interior i0+1..i1-1), the computational
grid uniform in i and stretched in j by `eta` (eta[j] for j in j0..j1). Gauss-Seidel with relaxation `omega`; the
boundary rows and columns are fixed. Returns the number of sweeps and the last largest move relative to the local cell."""
function winslow!(X, Y, i0, i1, j0, j1, eta; sweeps::Int=4000, omega::Float64=0.7, tol::Float64=1e-7, after_sweep=nothing,
                  P=nothing)
    last = 0.0; n = 0
    for sweep in 1:sweeps
        n = sweep; last = 0.0
        for j in j0+1:j1-1
            hm = eta[j] - eta[j-1]; hp = eta[j+1] - eta[j]; hs = hm + hp
            for i in i0+1:i1-1
                xx = (X[i+1, j] - X[i-1, j])/2; yx = (Y[i+1, j] - Y[i-1, j])/2
                xe = (X[i, j+1] - X[i, j-1])/hs; ye = (Y[i, j+1] - Y[i, j-1])/hs
                al = xe^2 + ye^2; be = xx*xe + yx*ye; ga = xx^2 + yx^2
                xxe = (X[i+1, j+1] - X[i+1, j-1] - X[i-1, j+1] + X[i-1, j-1])/(2hs)
                yxe = (Y[i+1, j+1] - Y[i+1, j-1] - Y[i-1, j+1] + Y[i-1, j-1])/(2hs)
                cg = 2ga/hs
                den = 2al + cg*(1/hp + 1/hm)
                sx = 0.0; sy = 0.0
                if P !== nothing
                    pp = P[i-i0+1, j]
                    pp != 0 && (sx = al*pp*xx; sy = al*pp*yx)
                end
                xn = (al*(X[i+1, j] + X[i-1, j]) + cg*(X[i, j+1]/hp + X[i, j-1]/hm) - 2be*xxe + sx)/den
                yn = (al*(Y[i+1, j] + Y[i-1, j]) + cg*(Y[i, j+1]/hp + Y[i, j-1]/hm) - 2be*yxe + sy)/den
                dx = omega*(xn - X[i, j]); dy = omega*(yn - Y[i, j])
                X[i, j] += dx; Y[i, j] += dy
                loc = min(hypot(X[i, j+1] - X[i, j], Y[i, j+1] - Y[i, j]), hypot(X[i+1, j] - X[i, j], Y[i+1, j] - Y[i, j]))
                last = max(last, hypot(dx, dy)/max(loc, 1e-300))
            end
        end
        after_sweep === nothing || after_sweep()
        last < tol && break
    end
    (n, last)
end

"""One plate on the LOWER side of a C block (the upper side is handled on the mirror image). `src` holds, per column,
the fractional column index of the clean mesh it was taken from (carried for the outer-face tags)."""
function _insert_lower(X::Matrix{Float64}, Y::Matrix{Float64}, NIW::Int, NIF::Int, h::Float64, src::Vector{Float64};
                       ratio::Float64=RATIO, frozen_fraction::Float64=1/6, window_depth::Float64=6.0, window_width::Float64=1.5,
                       align_passes::Int=4, winslow_eta::Symbol=:uniform, near::Symbol=:elliptic, march_depth::Float64=2.5,
                       control_iters::Int=400, control_sweeps::Int=20, control_relax::Float64=0.08, control_max::Float64=50.0,
                       control_decay::Float64=0.05, control_row::Int=4, redistribute::Bool=true, side_blend::Float64=3.0, leg_return::Float64=4.0, elliptic_frozen::Int=0, root_relief::Float64=1.0, tip_relief::Float64=0.0, window_foil_max::Float64=0.4, window_columns_ratio::Float64=2.0, plate_cells_multiple::Int=1,
                       offset_sigma0::Float64=0.25, offset_L::Float64=1.0, offset_sigma_max::Float64=4.0,
                       march_blend::Float64=0.45, march_normal::Float64=0.5, march_passes::Int=3, march_diffusion::Float64=0.5,
                       march_relax::Float64=0.3, march_relax_depth::Float64=1.0)
    NI = size(X, 1) - 1; NJ = size(X, 2) - 1; c = NIW + 1
    xte, yl = X[c, 1], Y[c, 1]
    dc = _coldist(X, Y, c)
    # 1. the plate's front face (tip -> root), exact corners
    ds_r = hypot(X[c+1, 1] - xte, Y[c+1, 1] - yl); ds_t = hypot(X[c-1, 1] - xte, Y[c-1, 1] - yl)
    sp = two_sided(h, ds_t, ds_r; r=ratio, dmax=max(h/8, ds_t, ds_r))
    if plate_cells_multiple > 1 && length(sp) % plate_cells_multiple != 0
        nm = plate_cells_multiple*cld(length(sp), plate_cells_multiple)
        sp = fixed_two_sided(nm, h, ds_t, ds_r)
    end
    m = length(sp)
    py = [yl - h; yl - h .+ cumsum(sp)]; py[end] = yl
    # 2. depths: frozen J_f (delta_f <= h/6), window top J_w (>= 5 h), blending radius R_b
    Jf = 2
    dc[2] <= frozen_fraction*h || throw(ArgumentError(@sprintf(
        "the plate of %.4g c needs a first wall spacing of at most %.3g c on this mesh (it has %.3g c)", h, frozen_fraction*h, dc[2])))
    if near === :elliptic
        Jf = 3
    elseif near === :march || near === :offset
        while Jf < NJ - 10 && dc[Jf] < march_depth*h; Jf += 1; end
    else
        while Jf + 1 <= NJ && dc[Jf+1] <= frozen_fraction*h; Jf += 1; end
    end
    Jw = Jf + 6
    while Jw < NJ - 2 && dc[Jw] < window_depth*h; Jw += 1; end
    Jw <= NJ - 2 || throw(ArgumentError("the mesh is too shallow for the plate's window"))
    df = dc[Jf]; Rb = near !== :frozen ? march_blend*h : min(3df, 0.45h)
    # 3. the window's columns (clean indices ia .. ibc; new ia .. ib = ibc + m)
    Lw = window_width*dc[Jw]
    ia = c - 2; ibc = c + 2
    local nc
    for _ in 1:40
        ia = c - 2
        while ia > 2 && X[ia, 1] - xte < Lw; ia -= 1; end
        ibc = c + 2; s = 0.0
        while ibc < NIW + NIF - 2 && s < min(Lw, window_foil_max)
            s += hypot(X[ibc+1, 1] - X[ibc, 1], Y[ibc+1, 1] - Y[ibc, 1]); ibc += 1
        end
        nc = ibc - ia
        nc >= window_columns_ratio*m && break
        Lw *= 1.25
    end
    nc >= 1.2m || throw(ArgumentError("the mesh has too few columns near the trailing edge for the plate's $(m) cells"))
    ib = ibc + m; NI2 = NI + m; Lnew = ib - ia
    A = 2m/Lnew
    phi(ip) = (t = ip - ia; ia + t - A*(t/2 - Lnew/(4pi)*sin(2pi*t/Lnew)))
    X2 = zeros(NI2 + 1, NJ + 1); Y2 = zeros(NI2 + 1, NJ + 1); src2 = zeros(NI2 + 1)
    # leg columns (clean i <= ia): the translation, rigid to J_f and decaying to zero at J_w (optionally faded out along
    # the wake over leg_return chords beyond the window; by default the whole leg carries it to the outlet)
    xw = X[ia, 1] - xte
    for i in 1:ia
        g = isfinite(leg_return) ? 1 - _smooth((X[i, 1] - xte - xw)/leg_return) : 1.0
        for j in 1:NJ+1
            # the same fraction on every leg column at a given row (the corner column's distances), so the long thin
            # wake cells are not sheared column against column; the per-row decrement is 1.875 h/D_w <= 0.31 of the
            # corner column's interval
            w = j <= Jf ? 1.0 : j >= Jw ? 0.0 : 1 - _smooth((dc[j] - dc[Jf])/(dc[Jw] - dc[Jf]))
            X2[i, j] = X[i, j]; Y2[i, j] = Y[i, j] - h*g*w
        end
        src2[i] = src[i]
    end
    for ip in ib:NI2+1                         # columns beyond the window: the clean ones
        X2[ip, :] .= X[ip-m, :]; Y2[ip, :] .= Y[ip-m, :]; src2[ip] = src[ip-m]
    end
    # rows >= J_w over the window: the clean rows at a fractional clean index q per new column (bilinear within clean
    # cells, so no fold can be introduced there); q starts as the smooth spreading phi and then SLIDES along row J_w so
    # that the window's columns arrive at it orthogonally (a Neumann condition: the plate's columns choose where they land)
    q = [phi(ip) for ip in ia:ib]; q[1] = ia; q[end] = ibc
    rowpt(j, qq) = (k = clamp(floor(Int, qq), 1, NI); f = qq - k; ((1 - f)*X[k, j] + f*X[k+1, j], (1 - f)*Y[k, j] + f*Y[k+1, j]))
    # above J_w the columns' clean indices relax from the slid q (which the window's top row needs) to a smoothed copy
    # (its increments diffused, the range kept) over a further 2 D_w: the far-field cells of the window are then not
    # bunched where the sliding's monotonicity floor held two columns together
    function setrows!(jfrom; smooth_outer::Bool=false)
        qs = copy(q)
        if smooth_outer
            dq = diff(q)
            for _ in 1:200
                d2 = copy(dq)
                for k in 1:length(dq)          # (the clean increment 1 outside both ends of the window)
                    d2[k] = 0.25(k == 1 ? 1.0 : dq[k-1]) + 0.5dq[k] + 0.25(k == length(dq) ? 1.0 : dq[k+1])
                end
                dq = d2 .* ((q[end] - q[1])/sum(d2))
            end
            qs = [q[1]; q[1] .+ cumsum(dq)]; qs[end] = q[end]
        end
        for n in 2:length(q)-1, j in jfrom:NJ+1
            ip = ia + n - 1
            b = smooth_outer ? _smooth((dc[min(j, NJ+1)] - dc[Jw])/(2*(dc[Jw] - dc[1]))) : 0.0
            p = rowpt(j, (1 - b)*q[n] + b*qs[n]); X2[ip, j] = p[1]; Y2[ip, j] = p[2]
        end
    end
    setrows!(Jw)
    epsq = 0.2*(ibc - ia)/(ib - ia)
    function top!()
        for n in 2:length(q)-1
            ip = ia + n - 1; px = X2[ip, Jw-1]; pyy = Y2[ip, Jw-1]
            bd = Inf; bq = q[n]; k0 = floor(Int, q[n])
            for k in max(ia, k0 - 8):min(ibc - 1, k0 + 8)
                ax, ay = X[k, Jw], Y[k, Jw]; bx, by = X[k+1, Jw], Y[k+1, Jw]
                t = clamp(((px - ax)*(bx - ax) + (pyy - ay)*(by - ay))/((bx - ax)^2 + (by - ay)^2), 0.0, 1.0)
                dd = hypot(ax + t*(bx - ax) - px, ay + t*(by - ay) - pyy)
                dd < bd && (bd = dd; bq = k + t)
            end
            # near the window's two ends the landing follows the smooth spreading phi (increment 1 at the ends, so the
            # window's outer cells grade into the clean ones); in its middle the orthogonal projection
            we = 1 - _smooth(min(n - 1, length(q) - n)/(0.2*length(q)))
            q[n] = 0.5q[n] + 0.5*((1 - we)*bq + we*phi(ip))
        end
        for n in 2:length(q)-1; q[n] = max(q[n], q[n-1] + epsq); end
        for n in length(q)-1:-1:2; q[n] = min(q[n], q[n+1] - epsq); end
        for n in 2:length(q)-1
            ip = ia + n - 1; p = rowpt(Jw, q[n]); X2[ip, Jw] = p[1]; Y2[ip, Jw] = p[2]
        end
    end
    # the new wall row, each window column's wall normal (blended through the tip and the root over R_b) and depth profile
    ntip = _unit((-1.0, -1.0)); nplate = (-1.0, 0.0)
    nfoil = _rot90(_unit((X[c+1, 1] - xte, Y[c+1, 1] - yl)))
    nroot = _unit((nplate[1] + nfoil[1], nplate[2] + nfoil[2]))
    nwall = Dict{Int,NTuple{2,Float64}}(); nwall0 = Dict{Int,NTuple{2,Float64}}(); dcol = Dict{Int,Vector{Float64}}()
    for ip in ia+1:ib-1
        if ip < c                              # leg column (translated); fanned toward the tip within R_b
            i = ip; s = X[i, 1] - xte
            X2[ip, 1] = X[i, 1]; Y2[ip, 1] = Y[i, 1] - h
            Rl = side_blend*h; nwall[ip] = s < Rl ? _slerp(ntip, _colfirst(X, Y, i), _smooth(s/Rl)) : _colfirst(X, Y, i)
            dcol[ip] = _coldist(X, Y, i)
        elseif ip <= c + m                     # the plate's front face
            k = ip - c; y0 = py[k+1]; st = y0 - (yl - h); sr = yl - y0
            X2[ip, 1] = xte; Y2[ip, 1] = y0
            nwall[ip] = ip == c ? ntip : ip == c + m ? nroot :
                 st < Rb ? _slerp(ntip, nplate, _smooth(st/Rb)) : sr < Rb ? _slerp(nroot, nplate, _smooth(sr/Rb)) : nplate
            dcol[ip] = dc
        else                                   # foil column; fanned toward the root within R_b
            i = ip - m; s = 0.0
            for k in c:i-1; s += hypot(X[k+1, 1] - X[k, 1], Y[k+1, 1] - Y[k, 1]); end
            X2[ip, 1] = X[i, 1]; Y2[ip, 1] = Y[i, 1]
            Rf = side_blend*h; nwall[ip] = s < Rf ? _slerp(nroot, _colfirst(X, Y, i), _smooth(s/Rf)) : _colfirst(X, Y, i)
            dcol[ip] = _coldist(X, Y, i)
        end
    end
    X2[c, 1] = xte; Y2[c, 1] = yl - h; X2[c+m, 1] = xte; Y2[c+m, 1] = yl
    nwall[ia] = _colfirst(X2, Y2, ia); nwall[ib] = _colfirst(X2, Y2, ib)
    # the wall distance of each window column from the root and from the tip (for the control's corner relief)
    droot = Dict{Int,Float64}(); dtip = Dict{Int,Float64}()
    for ip in ia+1:ib-1
        if ip < c
            droot[ip] = h + X[ip, 1] - xte; dtip[ip] = X[ip, 1] - xte
        elseif ip <= c + m
            droot[ip] = yl - py[ip-c+1]; dtip[ip] = py[ip-c+1] - (yl - h)
        else
            droot[ip] = sum(hypot(X[k+1, 1] - X[k, 1], Y[k+1, 1] - Y[k, 1]) for k in c:ip-m-1; init=0.0); dtip[ip] = h + droot[ip]
        end
    end
    for ip in ia+1:ib-1
        nwall0[ip] = ip < c ? _colfirst(X, Y, ip) : ip == c ? ntip : ip < c + m ? nplate : ip == c + m ? nroot : _colfirst(X, Y, ip - m)
    end
    # the frozen rows 2..J_f: each column a curve from its wall node, leaving along its wall normal and turning (smoothstep
    # in depth, so the first rows stay orthogonal) to the direction its Winslow column takes at J_f (none at first)
    function frozen!(dirs)
        for ip in ia+1:ib-1
            x = X2[ip, 1]; y = Y2[ip, 1]; d = dcol[ip]; n0 = nwall[ip]
            for j in 2:Jf
                dm = 0.5(d[j-1] + d[j])
                dr = dirs === nothing ? n0 : _slerp(n0, dirs[ip], _smooth(dm/d[Jf]))
                x += (d[j] - d[j-1])*dr[1]; y += (d[j] - d[j-1])*dr[2]
                X2[ip, j] = x; Y2[ip, j] = y
            end
        end
    end
    # near=:march -- rows 2..J_f marched from the wall (an explicit hyperbolic construction): each node steps its own
    # column's clean interval along the front's normal, blended with its column's previous direction, the directions
    # diffused along the front in proportion to step/spacing (the concave root's columns spread instead of crossing, the
    # convex tip's fan), and the positions relaxed along the front with a weight that grows from zero at the wall (the
    # first rows keep the wall's tangential distribution, so the near-wall cells stay orthogonal)
    function march!()
        nI = ib - ia + 1
        dirs = Vector{NTuple{2,Float64}}(undef, nI); prev = Vector{NTuple{2,Float64}}(undef, nI)
        for j in 1:Jf-1
            for n in 2:nI-1
                ip = ia + n - 1
                if j == 1
                    dirs[n] = nwall[ip]
                else
                    t = _unit((X2[ip+1, j] - X2[ip-1, j], Y2[ip+1, j] - Y2[ip-1, j])); fn = _rot90(t)
                    pd = _unit((X2[ip, j] - X2[ip, j-1], Y2[ip, j] - Y2[ip, j-1]))
                    dirs[n] = _unit((march_normal*fn[1] + (1 - march_normal)*pd[1], march_normal*fn[2] + (1 - march_normal)*pd[2]))
                end
            end
            dirs[1] = _unit((X2[ia, j+1] - X2[ia, j], Y2[ia, j+1] - Y2[ia, j])); dirs[nI] = _unit((X2[ib, j+1] - X2[ib, j], Y2[ib, j+1] - Y2[ib, j]))
            if j >= 2
                for _ in 1:march_passes
                    copyto!(prev, dirs)
                    for n in 2:nI-1
                        ip = ia + n - 1
                        dsl = hypot(X2[ip, j] - X2[ip-1, j], Y2[ip, j] - Y2[ip-1, j]); dsr = hypot(X2[ip+1, j] - X2[ip, j], Y2[ip+1, j] - Y2[ip, j])
                        st = dcol[ip][j+1] - dcol[ip][j]
                        nu = min(0.45, march_diffusion*st/max(min(dsl, dsr), 1e-300))
                        dirs[n] = _unit((prev[n][1] + nu*(prev[n-1][1] + prev[n+1][1] - 2prev[n][1]),
                                         prev[n][2] + nu*(prev[n-1][2] + prev[n+1][2] - 2prev[n][2])))
                    end
                end
            end
            for n in 2:nI-1
                ip = ia + n - 1; st = dcol[ip][j+1] - dcol[ip][j]
                X2[ip, j+1] = X2[ip, j] + st*dirs[n][1]; Y2[ip, j+1] = Y2[ip, j] + st*dirs[n][2]
            end
            mu = march_relax*_smooth(dc[j+1]/(march_relax_depth*h))
            if mu > 0
                for _ in 1:2
                    for n in 2:nI-1
                        ip = ia + n - 1
                        mx = 0.5(X2[ip-1, j+1] + X2[ip+1, j+1]) - X2[ip, j+1]; my = 0.5(Y2[ip-1, j+1] + Y2[ip+1, j+1]) - Y2[ip, j+1]
                        t = _unit((X2[ip+1, j+1] - X2[ip-1, j+1], Y2[ip+1, j+1] - Y2[ip-1, j+1]))
                        a = mu*(mx*t[1] + my*t[2])
                        X2[ip, j+1] += a*t[1]; Y2[ip, j+1] += a*t[2]
                    end
                end
            end
        end
    end
    # near=:offset -- rows 2..J_f as offsets along a normal field smoothed along the wall at a scale that grows with depth:
    # sigma(d) = sigma0 (1 + d/L)^2 (capped at sigma_max), L = sigma0. At the concave root the columns converge at the
    # rate |d theta/ds| of the smoothed field; a Gaussian-smoothed jump of 1.70 rad has a peak slope 0.68/sigma, and the
    # integral of 0.68/sigma over all depths is 0.68 L/sigma0 = 0.68 < 1, so the columns cannot meet; once sigma exceeds
    # the plate the tip's -90 deg and the root's +97 deg turns cancel and the field is smooth. At the wall the field is
    # the wall normal smoothed over sigma0 = h/4 (orthogonal along the middle of the plate and away from the corners).
    function offset!()
        nI = ib - ia + 1
        sw = zeros(nI); th0 = zeros(nI)
        for n in 2:nI
            sw[n] = sw[n-1] + hypot(X2[ia+n-1, 1] - X2[ia+n-2, 1], Y2[ia+n-1, 1] - Y2[ia+n-2, 1])
        end
        for n in 1:nI
            ip = ia + n - 1
            v = n == 1 ? _colfirst(X2, Y2, ia) : n == nI ? _colfirst(X2, Y2, ib) : nwall0[ip]
            th0[n] = atan(v[2], v[1])
            n > 1 && (th0[n] = th0[n-1] + rem2pi(th0[n] - th0[n-1], RoundNearest))
        end
        dsw = [n == 1 ? 0.5(sw[2] - sw[1]) : n == nI ? 0.5(sw[nI] - sw[nI-1]) : 0.5(sw[n+1] - sw[n-1]) for n in 1:nI]
        sig0 = offset_sigma0*h; Lsig = offset_L*sig0; smax = offset_sigma_max*h
        thd = similar(th0)
        for n in 2:nI-1
            ip = ia + n - 1; d = dcol[ip]
            x = X2[ip, 1]; y = Y2[ip, 1]
            for j in 2:Jf
                dm = 0.5(d[j-1] + d[j])
                sg = min(sig0*(1 + dm/Lsig)^2, smax, max(min(sw[n], sw[nI] - sw[n])/3, 1e-12))
                num = 0.0; den = 0.0
                for k in 1:nI
                    z = (sw[k] - sw[n])/sg
                    abs(z) > 5 && continue
                    w = exp(-0.5z*z)*dsw[k]; num += w*th0[k]; den += w
                end
                t = num/den
                x += (d[j] - d[j-1])*cos(t); y += (d[j] - d[j-1])*sin(t)
                X2[ip, j] = x; Y2[ip, j] = y
            end
        end
    end
    # near=:elliptic -- rows 2..J_w-1 of the window from Winslow's equations on the computational grid stretched as the
    # clean corner column, with Sorenson's control functions (GRAPE): P and Q at the wall are solved, under-relaxed, so
    # that the columns leave every wall node along its (corner-blended) normal, and decay into the interior; the top row
    # slides along the clean row J_w (orthogonal arrival). Then each column's nodes are redistributed along it, two-sided
    # (its wall spacing, the clean interval at J_w).
    function elliptic!()
        eta = zeros(NJ + 1); for j in 1:Jw; eta[j] = (dc[j] - dc[1])/(dc[Jw] - dc[1]); end
        for ip in ia+1:ib-1, j in 2:Jw-1
            u = (ip - ia)/(ib - ia); v = eta[j]
            X2[ip, j] = (1 - v)*X2[ip, 1] + v*X2[ip, Jw] + (1 - u)*X2[ia, j] + u*X2[ib, j] -
                        ((1 - u)*(1 - v)*X2[ia, 1] + u*(1 - v)*X2[ib, 1] + (1 - u)*v*X2[ia, Jw] + u*v*X2[ib, Jw])
            Y2[ip, j] = (1 - v)*Y2[ip, 1] + v*Y2[ip, Jw] + (1 - u)*Y2[ia, j] + u*Y2[ib, j] -
                        ((1 - u)*(1 - v)*Y2[ia, 1] + u*(1 - v)*Y2[ib, 1] + (1 - u)*v*Y2[ia, Jw] + u*v*Y2[ib, Jw])
        end
        # the first rows 2..K (elliptic_frozen) fixed at offsets along the corner-blended wall normals: around the tip's
        # 270 deg corner Winslow's own first nodes (rows one wall spacing thick, columns several wider) can fold the
        # cells that touch it; fixed, they fan exactly, and the solve starts above them
        # (auto, elliptic_frozen = 0: two rows, three where the tip's first wall cells are flatter than 1:6 -- measured: the
        # compressible profile's parent level, 8:1, folds the tip's cells with two; the PBCav demo mesh, 5:1, is better with two)
        K = max(1, min(elliptic_frozen > 0 ? elliptic_frozen : (ds_t/dc[2] > 6 ? 3 : 2), Jw - 4))
        for ip in ia+1:ib-1, j in 2:K
            X2[ip, j] = X2[ip, 1] + dcol[ip][j]*nwall[ip][1]; Y2[ip, j] = Y2[ip, 1] + dcol[ip][j]*nwall[ip][2]
        end
        nI = ib - ia + 1
        pb = zeros(nI); P = zeros(nI, Jw)
        for it in 1:control_iters
            # each wall node's departure angle from its (corner-blended) normal, at the control row; the source
            # alpha p x_xi shifts a column toward higher i for p > 0, so p falls where the column leans that way
            emax = 0.0
            for n in 2:nI-1
                ip = ia + n - 1
                dx = X2[ip, control_row] - X2[ip, 1]; dy = Y2[ip, control_row] - Y2[ip, 1]
                nn = nwall[ip]; tt = (nn[2], -nn[1])            # tt: the wall's +i direction (n = rot90(t))
                e = atan(dx*tt[1] + dy*tt[2], dx*nn[1] + dy*nn[2])
                emax = max(emax, abs(e))
                gain = (root_relief > 0 ? _smooth(droot[ip]/(root_relief*h)) : 1.0)*(tip_relief > 0 ? _smooth(dtip[ip]/(tip_relief*h)) : 1.0)
                pb[n] = clamp(pb[n] - control_relax*gain*e, -control_max, control_max)
            end
            haskey(ENV, "GDEBUG") && it % 20 == 1 && println("it ", it, " max departure ", round(rad2deg(emax), digits=2), " deg, max |p| ", round(maximum(abs, pb), digits=3))
            for n in 2:nI-1, j in 2:Jw-1
                P[n, j] = pb[n]*exp(-(eta[j] - eta[1])/control_decay)
            end
            winslow!(X2, Y2, ia, ib, K, Jw, eta; sweeps=control_sweeps, omega=1.0, tol=1e-12, after_sweep=top!, P=P)
        end
        setrows!(Jw; smooth_outer=true)
        for n in 2:length(q)-1
            ip = ia + n - 1; k = clamp(floor(Int, q[n]), 1, NI); f = q[n] - k; src2[ip] = (1 - f)*src[k] + f*src[k+1]
        end
        for ip in ia+1:ib-1
            redistribute || break
            cx = X2[ip, 1:Jw]; cy = Y2[ip, 1:Jw]
            sc = zeros(Jw); for k in 2:Jw; sc[k] = sc[k-1] + hypot(cx[k] - cx[k-1], cy[k] - cy[k-1]); end
            a = dcol[ip][2]
            b = hypot(X2[ip, Jw+1] - X2[ip, Jw], Y2[ip, Jw+1] - Y2[ip, Jw])/1.15
            gv = vinokur(Jw - 1, a/sc[end], b/sc[end])
            # far from the plate the column keeps its clean column's own shape (adjacent long, thin leg cells must not
            # differ); within h of the plate Vinokur's ends (the wall spacing, the clean interval at J_w); in between, a
            # smooth blend of the two (a convex combination of monotone fractions is monotone)
            sd = ip < c ? X[ip, 1] - xte : ip <= c + m ? 0.0 : sum(hypot(X[k+1, 1] - X[k, 1], Y[k+1, 1] - Y[k, 1]) for k in c:ip-m-1; init=0.0)
            w = 1 - _smooth((sd - h)/(2h))
            dsh = ip < c ? dcol[ip] : ip <= c + m ? dc : dcol[ip]
            # (a leg column carries the translation: its wall moved by h, row j by h w_j, as outside the window)
            sh = ip < c ? [h*(1 - (j <= Jf ? 1.0 : j >= Jw ? 0.0 : 1 - _smooth((dc[j] - dc[Jf])/(dc[Jw] - dc[Jf])))) for j in 1:Jw] : zeros(Jw)
            gc = [(dsh[j] - dsh[1] - sh[j])/(dsh[Jw] - dsh[1] - sh[Jw]) for j in 1:Jw]
            g = w .* gv .+ (1 - w) .* gc
            for j in 2:Jw-1
                t = g[j]*sc[end]
                k = clamp(searchsortedlast(sc, t), 1, Jw - 1); f = (t - sc[k])/(sc[k+1] - sc[k])
                X2[ip, j] = (1 - f)*cx[k] + f*cx[k+1]; Y2[ip, j] = (1 - f)*cy[k] + f*cy[k+1]
            end
        end
    end
    ws = (0, 0.0)
    if near === :elliptic
        elliptic!()
    else
    near === :offset ? offset!() : near === :march ? march!() : frozen!(nothing)
    # rows J_f..J_w over the window: a Coons start, then Winslow's equations on a UNIFORM computational grid (the harmonic
    # map: its columns cannot cross, and near the concave root its rows spread instead of creasing), alternating with the
    # frozen layer's alignment to the columns it meets
    etau = zeros(NJ + 1); for j in Jf:Jw; etau[j] = winslow_eta === :stretched ? (dc[j] - dc[Jf])/(dc[Jw] - dc[Jf]) : (j - Jf)/(Jw - Jf); end
    for ip in ia+1:ib-1, j in Jf+1:Jw-1
        u = (ip - ia)/(ib - ia); v = etau[j]
        X2[ip, j] = (1 - v)*X2[ip, Jf] + v*X2[ip, Jw] + (1 - u)*X2[ia, j] + u*X2[ib, j] -
                    ((1 - u)*(1 - v)*X2[ia, Jf] + u*(1 - v)*X2[ib, Jf] + (1 - u)*v*X2[ia, Jw] + u*v*X2[ib, Jw])
        Y2[ip, j] = (1 - v)*Y2[ip, Jf] + v*Y2[ip, Jw] + (1 - u)*Y2[ia, j] + u*Y2[ib, j] -
                    ((1 - u)*(1 - v)*Y2[ia, Jf] + u*(1 - v)*Y2[ib, Jf] + (1 - u)*v*Y2[ia, Jw] + u*v*Y2[ib, Jw])
    end
    for outer in 1:(near !== :frozen ? 1 : align_passes)
        ws = winslow!(X2, Y2, ia, ib, Jf, Jw, etau; sweeps=8000, omega=1.0, tol=1e-9, after_sweep=top!)
        (outer == align_passes || near !== :frozen) && break
        frozen!(Dict(ip => _unit((X2[ip, Jf+1] - X2[ip, Jf], Y2[ip, Jf+1] - Y2[ip, Jf])) for ip in ia+1:ib-1))
    end
    setrows!(Jw)
    for n in 2:length(q)-1
        ip = ia + n - 1; k = clamp(floor(Int, q[n]), 1, NI); f = q[n] - k; src2[ip] = (1 - f)*src[k] + f*src[k+1]
    end
    # each window column's nodes J_f..J_w redistributed along it, two-sided: the frozen layer's next interval, the clean one
    rf = (dc[Jf] - dc[Jf-1])/(dc[Jf-1] - dc[Jf-2])
    for ip in ia+1:ib-1
        cx = X2[ip, Jf:Jw]; cy = Y2[ip, Jf:Jw]
        sc = zeros(length(cx)); for k in 2:length(cx); sc[k] = sc[k-1] + hypot(cx[k] - cx[k-1], cy[k] - cy[k-1]); end
        a = hypot(X2[ip, Jf] - X2[ip, Jf-1], Y2[ip, Jf] - Y2[ip, Jf-1])*rf
        b = hypot(X2[ip, Jw+1] - X2[ip, Jw], Y2[ip, Jw+1] - Y2[ip, Jw])/rf
        sp2 = diff(vinokur(Jw - Jf, a/sc[end], b/sc[end])) .* sc[end]
        t = 0.0
        for j in Jf+1:Jw-1
            t += sp2[j-Jf]
            k = clamp(searchsortedlast(sc, t), 1, length(sc) - 1); f = (t - sc[k])/(sc[k+1] - sc[k])
            X2[ip, j] = (1 - f)*cx[k] + f*cx[k+1]; Y2[ip, j] = (1 - f)*cy[k] + f*cy[k+1]
        end
    end
    end
    info = Dict{String,Any}("plate_height"=>h, "plate_cells"=>m, "plate_first_spacing_tip"=>sp[1], "plate_first_spacing_root"=>sp[end],
        "plate_max_spacing"=>maximum(sp), "frozen_rows"=>Jf, "frozen_depth"=>df, "window_top_row"=>Jw, "window_depth"=>dc[Jw],
        "blend_radius"=>Rb, "leg_return_chords"=>leg_return, "window_columns"=>[ia, ib], "window_clean_columns"=>nc, "winslow_sweeps"=>ws[1], "winslow_last_move"=>ws[2],
        "tip_index"=>c, "root_index"=>c + m)
    (X2, Y2, NIF + m, src2, info)
end

_mirror(X, Y) = (reverse(X; dims=1), -reverse(Y; dims=1))

"""    gurney_h_block(X, Y, NIW, NIF; s1, ratio) -> (Xs, Ys, NS, t)

The H block behind the base x = x_TE, from the C block's two legs: the base's cross distribution (Vinokur's shape, the count
of 1.15 ramps from the first interval s1 at both ends, intervals <= base/8) at the base, blended over 2 base heights into
each station's (its ends equal to the C block's first radial interval at the two leg columns there)."""
function gurney_h_block(X::AbstractMatrix, Y::AbstractMatrix, NIW::Int, NIF::Int; s1::Float64, ratio::Float64=RATIO,
                        h_shape::Symbol=:vinokur, base_blend_follow::Bool=false)
    # the H block: the two legs, one cross distribution clustered toward both, on the straight base
    lo = (X[NIW+1, 1], Y[NIW+1, 1]); up = (X[NIW+NIF+1, 1], Y[NIW+NIF+1, 1])
    Lb = up[2] - lo[2]
    cs = two_sided(Lb, s1, s1; r=ratio, dmax=Lb/8, even=true)
    NS = length(cs)
    t = vinokur(NS, s1/Lb, s1/Lb)          # the count from the 1.15 ramps, the shape Vinokur's (as every station's)
    # station by station: the base's distribution at the base; downstream, first and last intervals equal to the C
    # block's first radial interval at the two leg columns there (so the C and H cells match across both interfaces;
    # when that exceeds the uniform interval, Vinokur's sine branch makes the middle intervals the smaller ones)
    Xs = zeros(NIW + 1, NS + 1); Ys = zeros(NIW + 1, NS + 1)
    # the ends' intervals: the C block's first radial interval at each leg column, times a factor kappa <= 1 where NS such
    # intervals would not fit (kappa = 0.9 L/(NS a)); kappa is made non-increasing downstream and smoothed, so the C and
    # H cells across each interface keep the same relative growth (long thin wake cells are sheared by any mismatch in
    # it) and no station needs Vinokur's sine branch (tiny middle cells: aspect ratios in the thousands at the outlet)
    al = zeros(NIW + 1); au = zeros(NIW + 1); Ls = zeros(NIW + 1)
    for k in 1:NIW+1
        il = NIW + 2 - k; iu = NIW + NIF + k
        Ls[k] = hypot(X[iu, 1] - X[il, 1], Y[iu, 1] - Y[il, 1])
        al[k] = hypot(X[il, 2] - X[il, 1], Y[il, 2] - Y[il, 1]); au[k] = hypot(X[iu, 2] - X[iu, 1], Y[iu, 2] - Y[iu, 1])
    end
    kap = ones(NIW + 1)   # (kappa < 1 was tried: it sheared the interface cells, theta 84 deg; recorded, not used)
    false && (kap = [min(1.0, 0.9Ls[k]/(NS*max(al[k], au[k]))) for k in 1:NIW+1])
    for k in 2:NIW+1; kap[k] = min(kap[k], kap[k-1]); end
    for _ in 1:20
        k2 = copy(kap)
        for k in 2:NIW; k2[k] = min(kap[k], 0.25kap[k-1] + 0.5kap[k] + 0.25kap[k+1]); end
        kap = k2
    end
    # the base distribution is blended into the stations' over 2 base heights, or less where the legs' first spacing
    # grows faster (by 20 % within that length: then the blend ends there, so the H cells follow the C cells' ramp)
    Lbl = 2Lb
    for k in 2:NIW+1
        if base_blend_follow && max(al[k], au[k]) > 1.2*max(al[1], au[1], s1)
            Lbl = min(Lbl, max(0.5(X[NIW+2-k, 1] + X[NIW+NIF+k, 1]) - lo[1], 1e-12)); break
        end
    end
    for k in 1:NIW+1
        il = NIW + 2 - k; iu = NIW + NIF + k
        xl, yl = X[il, 1], Y[il, 1]; xu, yu = X[iu, 1], Y[iu, 1]
        Lk = Ls[k]
        # the base's own distribution near the base (the H cells there are short streamwise, so any change of the cross
        # distribution from one station to the next shears them), blended into each station's over 2 base heights
        if h_shape === :ramps
            # every station from one family (the base's included): two geometric ramps from the C block's first radial
            # interval at each leg, their ratio solved for the station's height with NS cells -- the shape varies only as
            # smoothly as the legs' first spacing does (the host's own graded strip follows that ramp too)
            sp = fixed_two_sided(NS, Lk, al[k], au[k])
            tk = [0.0; cumsum(sp) ./ Lk]; tk[end] = 1.0
        else
            bk = _smooth((0.5(xl + xu) - lo[1])/Lbl)
            tk = bk == 0 ? t : (1 - bk) .* t .+ bk .* vinokur(NS, kap[k]*al[k]/Lk, kap[k]*au[k]/Lk)
        end
        for j in 1:NS+1
            Xs[k, j] = xl + tk[j]*(xu - xl); Ys[k, j] = yl + tk[j]*(yu - yl)
        end
        Xs[k, 1] = xl; Ys[k, 1] = yl; Xs[k, NS+1] = xu; Ys[k, NS+1] = yu
    end
    for j in 1:NS+1; Xs[1, j] = lo[1]; end    # the base column exactly on x = x_TE
    (Xs, Ys, NS, t)
end


"""    gurney_plate_mesh(X, Y, NIW, NIF, Xs, Ys; h_lower, h_upper) -> NamedTuple

The C + H mesh of the section with the plate(s), from the clean C + H node arrays of the same section (C block
(NI+1) x (NJ+1) with NIW wake columns per side and NIF surface cells; H block (NIW+1) x (NS+1)). Returns
`(X, Y, NIW, NIF, Xs, Ys, NS, src, lower, upper, convexity)`: the new C block (NIF grows by the plate cells), the rebuilt H
block, the clean column each new C column was resampled from (fractional, for the outer-face tags), each side's
construction record, and the convexity census of both blocks. Refuses (ArgumentError) when a cell is not strictly convex."""
function gurney_plate_mesh(X0::AbstractMatrix, Y0::AbstractMatrix, NIW::Int, NIF::Int, Xs0::AbstractMatrix, Ys0::AbstractMatrix;
                           h_lower::Real, h_upper::Real, ratio::Float64=RATIO, cross_first::Union{Nothing,Float64}=nothing,
                           refuse::Bool=true, h_shape::Symbol=:vinokur, base_blend_follow::Bool=false, kw...)
    X = Float64.(X0); Y = Float64.(Y0); NI = size(X, 1) - 1
    src = collect(1.0:NI+1)
    xte_l = X[NIW+1, 1]; xte_u = X[NIW+NIF+1, 1]
    abs(xte_l - xte_u) <= 1e-9 || throw(ArgumentError(@sprintf("the clean mesh's trailing-edge corners are %.2e c apart in x: the base is not one straight line", abs(xte_l - xte_u))))
    X[NIW+NIF+1, 1] = xte_l                 # (an ulp or so: the base and both plates on ONE x station, exactly)
    lower = nothing; upper = nothing
    if h_lower > 0
        X, Y, NIF, src, lower = _insert_lower(X, Y, NIW, NIF, Float64(h_lower), src; ratio, kw...)
    end
    if h_upper > 0
        Xm, Ym = _mirror(X, Y)
        Xm, Ym, NIF, srcm, upper = _insert_lower(Xm, Ym, NIW, NIF, Float64(h_upper), reverse(src); ratio, kw...)
        X, Y = _mirror(Xm, Ym); src = reverse(srcm)
        NI2 = size(X, 1) - 1
        upper["tip_index"] = NI2 + 2 - upper["tip_index"]; upper["root_index"] = NI2 + 2 - upper["root_index"]
        upper["window_columns"] = reverse(NI2 + 2 .- upper["window_columns"])
    end
    s1 = cross_first === nothing ? hypot(Xs0[1, 2] - Xs0[1, 1], Ys0[1, 2] - Ys0[1, 1]) : cross_first
    Xs, Ys, NS, t = gurney_h_block(X, Y, NIW, NIF; s1, ratio, h_shape, base_blend_follow)
    cC = convexity(X, Y); cH = convexity(Xs, Ys)
    conv = Dict{String,Any}("C_nonconvex"=>cC[1], "C_worst"=>cC[2], "C_worst_at"=>collect(cC[3]),
                            "H_nonconvex"=>cH[1], "H_worst"=>cH[2], "H_worst_at"=>collect(cH[3]))
    !refuse || (cC[1] == 0 && cH[1] == 0) || throw(ArgumentError(@sprintf(
        "the Gurney plate mesh has %d non-convex C cells (worst at %s) and %d non-convex H cells: refused", cC[1], string(cC[3]), cH[1])))
    (; X, Y, NIW, NIF, Xs, Ys, NS, t, src, lower, upper, convexity=conv)
end

end # module AirfoilGurneyMesh
