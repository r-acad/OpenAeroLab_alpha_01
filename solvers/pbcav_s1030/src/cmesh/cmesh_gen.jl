# ============================================================================================
# CMESH2D generator (2026-07-14, product Phase A) — body-fitted C-topology mesh around a
# point-defined airfoil. Pipeline: .dat -> arc-length spline resample (LE/TE clustered) ->
# inner boundary (lower wake -> foil TE-lower->LE->TE-upper -> upper wake) -> C-shaped outer
# boundary -> normal-blended TFI with per-column geometric wall-normal clustering (first
# spacing from a y+ target) -> optional Winslow elliptic smoothing (BL layers frozen) ->
# quality report (folds, wall orthogonality, achieved y+, growth) + plots + binary save.
#
# Topology & indexing (NODES X,Y :: (NI+1)x(NJ+1), CELLS NI x NJ):
#   i = 1..NIW+1          : LOWER wake leg, outlet -> TE (x decreasing, below the wake line)
#   i = NIW+1..NIW+NIF+1  : airfoil surface, TE-lower -> LE -> TE-upper (clockwise wrap)
#   i = NIW+NIF+1..NI+1   : UPPER wake leg, TE -> outlet
#   j = 1 (wall/wake cut) .. NJ+1 (outer C boundary);  NI = 2*NIW + NIF
#   WAKE CUT: node (i,1) and node (NI+2-i, 1) are the SAME physical point for i in 1..NIW+1
#   (the solver exchanges ghosts across the cut; the mesh guarantees coincidence exactly).
#
# TRAILING-EDGE OPTIONS (2 Oct 2026). Plan docs/2026-10-01_BLUNT_TRAILING_EDGE_PLAN.md; construction decision and
# integration spec docs/_records/2026-10-02_blunt_te/stage1/DECISION.md; integration record
# docs/_records/2026-10-02_blunt_te/stage1/INTEGRATION.md. Five keywords of generate_cmesh, every one inert at its
# default (the generator is then bitwise what it was, with one exception: G10, an open trailing edge in free air, which
# was meshed with non-coincident wake partners, is now refused; tools/test_cmesh_archived_receipts.jl is the external lock):
#   n_base > 0   blunt (finite-thickness) trailing edge, CHANNEL branch only: the midpoint split. The base (the straight
#                segment between the two resampled corners) is cut at its midpoint m into two halves of n_base uniform
#                cells folded into the foil index range, NIF = n_foil + 2 n_base:
#                  foil nodes = [m .. lower corner] + surface (lower corner .. LE .. upper corner) + [upper corner .. m]
#                Both junction nodes are m, so the wake-cut coincidence above holds bitwise, and the base faces are walls
#                by the index test. One construction, TE_BASE_CONSTRUCTION ("aplus_v1", see the helpers' header).
#   te_ds > 0    the surface interval at the trailing edge (at the two corners when n_base > 0), met as the Euclidean
#                chord between the first two resampled nodes of each side to 1e-9 relative (channel branch: with
#                n_base = 0 it is the sharp control's trailing-edge interval).
#   wake_near_ds, wake_near_len, wake_near_growth   the two-segment near wake (channel branch): the streamwise wake
#                interval held at or below wake_near_ds over wake_near_len behind the edge, then geometric to the outlet.
#                The wake starts at the trailing-edge interval (the base interval gap/(2 n_base) with a base; without
#                one te_ds, or the default distribution's), so a cap below it is refused (G14), never exceeded silently.
#
# STRIP TOPOLOGY (5 Oct 2026, docs/2026-10-05_STRIP_TOPOLOGY_PLAN.md section 6; record docs/_records/2026-10-05_strip_topology/
# stage1/README.md). One more keyword, te_topology = :aplus (the construction above, the default) or :strip, channel branch
# only, with n_base >= 2 (G-S1):
#   :strip       the base's cells are extruded downstream as a SECOND BLOCK, the strip, and the C block is the SHARP
#                construction wrapped around foil plus strip: NIF = n_foil (no fold), the open resample's ends are the two
#                corners, and the two wake legs leave the corners along the channel direction with their own station
#                vectors (each leg rescaled to its own length), 2 n_base strip cells apart. The strip (ChannelMesh.strip, a
#                StripBlock) is the linear transfinite interpolation between the legs: column 1 is the base exactly, column
#                NIW+1 lies on the outlet plane, and its two edges ARE the legs' row j = 1 bitwise (P1'). The legs' first
#                wall-normal spacing ramps from dn1 at the corner to the strip's edge interval gap/(2 n_base), or keeps dn1
#                where that interval is at or below dn1 inside G8''s slack (the per-column dn1_i of section 6.3, entering the
#                five dn1 sites through _column_schedule, whose ramp columns shrink back to the sharp schedule at the
#                column's own ratio). No A-plus trailing-edge machinery runs (tecols === nothing).
#                Construction TE_STRIP_CONSTRUCTION ("strip_v1"). With :aplus every statement of the file runs as before, the
#                per-column dn1_i being fill(dn1, NI+1): the same Float64 in the same operations, bitwise.
# ============================================================================================

module CMeshGen

export AirfoilPts, CMesh, ChannelSpec, ChannelMesh, StripBlock,
       OUTER_FARFIELD, OUTER_INLET, OUTER_LOWER_WALL, OUTER_UPPER_WALL, OUTER_STRIP_INTERFACE,
       read_airfoil_dat, generate_cmesh, mesh_quality, channel_quality, save_cmesh,
       TE_BASE_CONSTRUCTION, TE_STRIP_CONSTRUCTION, te_base_construction, te_construction_name, corner_sine_min

const OUTER_FARFIELD  = UInt8(0)
const OUTER_INLET     = UInt8(1)
const OUTER_LOWER_WALL = UInt8(2)
const OUTER_UPPER_WALL = UInt8(3)
const OUTER_STRIP_INTERFACE = UInt8(4)     # the strip block's outer tag (strip topology): its j = NJ+1 faces are the upper
                                           # interface with the C block, never a channel face; the kind of all its columns

struct AirfoilPts
    x::Vector{Float64}; y::Vector{Float64}    # ordered TE-lower -> LE -> TE-upper (the wrap order)
end

struct CMesh
    NI::Int; NJ::Int; NIW::Int; NIF::Int
    X::Matrix{Float64}; Y::Matrix{Float64}    # nodes (NI+1)x(NJ+1)
    dn1::Float64                              # first wall-normal spacing actually used (chords)
    growth::Vector{Float64}                   # per-column geometric ratio used
end

"""
Physical definition of a straight two-dimensional channel in foil-fixed coordinates.

`ref` is the physical point whose streamwise coordinate is `xi_ref`; `streamwise` is the
unit vector parallel to the tunnel walls.  Thus

    P(xi, eta) = ref + (xi - xi_ref)*streamwise + eta*(-streamwise[2], streamwise[1])

and the walls are `eta = +/-height/2`.  The Leroux CAV2003 setup used by the confinement
campaign is represented by `ChannelSpec(angle_deg=8)`: H/c=1.28, inlet xi=-4/3, outlet
xi=5, with the foil mid-chord `(0.5, 0)` at channel mid-height.
"""
struct ChannelSpec
    height::Float64
    xi_in::Float64
    xi_out::Float64
    xi_ref::Float64
    ref::NTuple{2,Float64}
    streamwise::NTuple{2,Float64}

    function ChannelSpec(height::Real, xi_in::Real, xi_out::Real, xi_ref::Real,
                         ref::Tuple{<:Real,<:Real}, streamwise::Tuple{<:Real,<:Real})
        h = Float64(height); xin = Float64(xi_in); xout = Float64(xi_out)
        xiref = Float64(xi_ref); pref = (Float64(ref[1]), Float64(ref[2]))
        es = (Float64(streamwise[1]), Float64(streamwise[2]))
        all(isfinite, (h, xin, xout, xiref, pref..., es...)) ||
            throw(ArgumentError("ChannelSpec values must all be finite"))
        h > 0 || throw(ArgumentError("ChannelSpec height must be positive (got $h)"))
        xin < xiref < xout || throw(ArgumentError(
            "ChannelSpec requires xi_in < xi_ref < xi_out (got $xin, $xiref, $xout)"))
        enrm = hypot(es...)
        enrm > 0 || throw(ArgumentError("ChannelSpec streamwise direction must be nonzero"))
        esn = (es[1]/enrm, es[2]/enrm)
        new(h, xin, xout, xiref, pref, esn)
    end
end

function ChannelSpec(; height::Real=1.28, xi_in::Real=-4/3, xi_out::Real=5.0,
                     xi_ref::Real=0.5, ref::Tuple{<:Real,<:Real}=(0.5, 0.0),
                     angle_deg::Real=8.0)
    a = Float64(angle_deg)
    isfinite(a) || throw(ArgumentError("ChannelSpec angle_deg must be finite"))
    ChannelSpec(height, xi_in, xi_out, xi_ref, ref, (cosd(a), sind(a)))
end

"""
The strip block of the strip trailing-edge topology (docs/2026-10-05_STRIP_TOPOLOGY_PLAN.md, section 2): the H block
behind the base. Nodes `X, Y :: (NI+1) x (NJ+1)` with `NI = n_wake` cells along the wake (`i = 1` beside the base, column
`NI+1` on the outlet plane) and `NJ = 2 n_base` cells across (row `j = 1` on the lower leg, row `NJ+1` on the upper leg).
`t` is the cross parameter of the transfinite interpolation (0 at the lower corner, 1 at the upper); `dn1_i` is the C
block's per-column first wall-normal spacing (length `2 n_wake + NIF + 1`): dn1 on every surface column and the legs'
ramp of section 6.3.
"""
struct StripBlock
    NI::Int; NJ::Int; n_base::Int
    X::Matrix{Float64}; Y::Matrix{Float64}
    t::Vector{Float64}
    dn1_i::Vector{Float64}
end

"""A C-mesh carrying the channel geometry and one boundary-kind tag per outer j-face (and, with the strip topology, the
strip block behind the base; `nothing` on every other mesh)."""
struct ChannelMesh
    NI::Int; NJ::Int; NIW::Int; NIF::Int
    X::Matrix{Float64}; Y::Matrix{Float64}
    dn1::Float64
    growth::Vector{Float64}
    channel::ChannelSpec
    outer_face_kind::Vector{UInt8}             # length NI; OUTER_* constants above
    strip::Union{Nothing,StripBlock}           # the strip block (te_topology = :strip, 5 Oct 2026); nothing otherwise
end
# The ten-argument constructor of every strip-free mesh: the generator's default and A-plus paths and the fixed-outer
# control (tools/cmesh_fixed_outer_mesh.jl) call it as before the strip field existed.
ChannelMesh(NI, NJ, NIW, NIF, X, Y, dn1, growth, channel, outer_face_kind) =
    ChannelMesh(NI, NJ, NIW, NIF, X, Y, dn1, growth, channel, outer_face_kind, nothing)

# ---- .dat reader: auto-detect Selig (TE->LE->TE single loop) vs dual-surface (upper LE->TE, lower LE->TE)
function read_airfoil_dat(path::AbstractString)
    pts = Tuple{Float64,Float64}[]
    for ln in eachline(path)
        s = strip(ln); (isempty(s) || startswith(s, "#")) && continue
        v = split(s)
        length(v) < 2 && continue
        xy = tryparse.(Float64, (v[1], v[2]))
        any(isnothing, xy) && continue        # header/name line
        push!(pts, (xy[1], xy[2]))
    end
    @assert length(pts) >= 20 "airfoil dat: too few points ($(length(pts)))"
    xs = first.(pts); ys = last.(pts)
    # find a restart of x near 0 after progress toward 1 => dual-surface format
    split_at = 0
    for k in 3:length(pts)-2
        if xs[k] < 0.05 && xs[k-1] > 0.5; split_at = k; break; end
    end
    if split_at == 0 && xs[1] > 0.5
        # Selig loop TE->LE->TE: first half runs TE->LE over the upper surface (convention), second LE->TE lower.
        # Reorder to TE-lower -> LE -> TE-upper: reverse the whole loop.
        ile = argmin(xs)
        up = collect(zip(xs[1:ile], ys[1:ile]))            # TE -> LE (upper)
        lo = collect(zip(xs[ile:end], ys[ile:end]))        # LE -> TE (lower)
        ordered = vcat(reverse(lo), up[2:end] |> reverse)  # TE-lower -> LE -> ... hmm; build explicitly below
        # explicit: TE-lower->LE = reverse(lo); LE->TE-upper = reverse(up)
        ordered = vcat(reverse(lo), reverse(up)[2:end])
        return AirfoilPts(first.(ordered), last.(ordered))
    end
    @assert split_at > 0 "airfoil dat: unrecognized format (expected Selig or dual-surface)"
    up = collect(zip(xs[1:split_at-1], ys[1:split_at-1]))  # upper LE->TE
    lo = collect(zip(xs[split_at:end], ys[split_at:end]))  # lower LE->TE
    ordered = vcat(reverse(lo), up[2:end])                 # TE-lower->LE (rev lower), then LE->TE-upper (skip dup LE)
    return AirfoilPts(first.(ordered), last.(ordered))
end

# ---- Catmull-Rom on the ordered point set, parametrized by cumulative arc length -----------------------
function _arclen(x, y)
    s = zeros(length(x))
    for k in 2:length(x); s[k] = s[k-1] + hypot(x[k]-x[k-1], y[k]-y[k-1]); end
    s
end
function _cr_eval(px, py, s, squery)         # Catmull-Rom sample at arc-length squery (s must be increasing)
    n = length(px)
    k = clamp(searchsortedlast(s, squery), 1, n-1)
    t = (squery - s[k]) / max(s[k+1]-s[k], 1e-30)
    i0 = max(k-1,1); i3 = min(k+2,n)
    b(u0,u1,u2,u3) = 0.5*((2u1) + (-u0+u2)*t + (2u0-5u1+4u2-u3)*t^2 + (-u0+3u1-3u2+u3)*t^3)
    (b(px[i0],px[k],px[k+1],px[i3]), b(py[i0],py[k],py[k+1],py[i3]))
end

# double-sided cosine clustering blended with uniform: xi in [0,1] -> t in [0,1], clustering both ends
_cluster(xi, b) = (1-b)*xi + b*(1 - cos(pi*xi))/2

# Two-sided stretching with prescribed end spacings (Vinokur 1983, J. Comput. Phys. 50:215): n intervals on [0,1] whose
# first interval is about s0 and last about s1 (fractions of the length). Used by the independent leading-edge spacing
# (speed-up plan W5, 30 Sep 2026); the default distribution never calls it.
function _vinokur(n::Int, s0::Float64, s1::Float64)
    (n >= 2 && 0 < s0 < 1 && 0 < s1 < 1) || throw(ArgumentError("vinokur: n >= 2 and end spacings in (0,1) required"))
    A = sqrt(s1/s0); B = 1/(n*sqrt(s0*s1))
    f(d) = B > 1 ? sinh(d)/d - B : sin(d)/d - B
    lo, hi = B > 1 ? (1e-8, 1.0) : (1e-8, pi - 1e-8)
    if B > 1
        while f(hi) < 0; hi *= 2; end
    end
    local d
    if abs(B - 1) < 1e-6
        d = 0.0
    else
        for _ in 1:200
            d = 0.5*(lo + hi)
            (f(d) > 0) == (B > 1 ? true : false) ? (hi = d) : (lo = d)
            hi - lo < 1e-14*max(1.0, hi) && break
        end
        d = 0.5*(lo + hi)
    end
    t = zeros(n+1)
    for k in 0:n
        xi = k/n
        u = d == 0.0 ? xi : (B > 1 ? 0.5*(1 + tanh(d*(xi - 0.5))/tanh(d/2)) : 0.5*(1 + tan(d*(xi - 0.5))/tan(d/2)))
        t[k+1] = u/(A + (1 - A)*u)
    end
    t[1] = 0.0; t[end] = 1.0
    t
end

# geometric growth ratio r solving dn1*(r^N - 1)/(r - 1) = L  (bisection; r=1 -> uniform)
function _solve_growth2(dn1, N, L)
    L <= dn1*N && return 1.0
    f(r) = dn1*(r^N - 1)/(r - 1) - L
    lo, hi = 1.0 + 1e-12, 3.0
    f(hi) < 0 && return 3.0
    for _ in 1:200
        mid = 0.5*(lo+hi)
        if f(mid) > 0; hi = mid; else; lo = mid; end
    end
    0.5*(lo+hi)
end

# Realised (coordinate-derived) adjacent radial-cell spacing ratios.  This is intentionally
# independent of the nominal per-column `growth` vector: smoothing and later repairs can change
# the distances represented by that construction-time schedule.
function _radial_spacing_stats(X, Y, NI, NJ)
    NJ < 2 && return (ratio_max=1.0, ratio_symmetric_max=1.0,
                      ratio_location=(i=0,j=0), ratio_symmetric_location=(i=0,j=0))
    rmax = 0.0; rsmax = 0.0
    rloc = (i=0,j=0); rsloc = (i=0,j=0)
    for i in 1:NI+1, j in 1:NJ-1
        d1 = hypot(X[i,j+1]-X[i,j], Y[i,j+1]-Y[i,j])
        d2 = hypot(X[i,j+2]-X[i,j+1], Y[i,j+2]-Y[i,j+1])
        ratio = d2/max(d1, 1e-30)
        symmetric = max(ratio, 1/max(ratio,1e-30))
        if ratio > rmax
            rmax = ratio; rloc = (i=i,j=j)
        end
        if symmetric > rsmax
            rsmax = symmetric; rsloc = (i=i,j=j)
        end
    end
    (ratio_max=rmax, ratio_symmetric_max=rsmax,
     ratio_location=rloc, ratio_symmetric_location=rsloc)
end

# The solver's reconstruction (cmesh_species_reconstruction.jl:11-21) refuses any cell whose bilinear map is not
# a bijection: all four corner Jacobians positive, i.e. the quadrilateral is strictly convex with consistent
# orientation. A mesh that passes the generator's old signed-area test but fails this one builds and then kills
# the run at its first step (measured 22 Sep 2026 at n_foil 416). The generator now applies the same test.
@inline function _corner_jacobians(X, Y, i, j)
    x1, y1 = X[i,j],     Y[i,j];     x2, y2 = X[i+1,j],   Y[i+1,j]
    x3, y3 = X[i+1,j+1], Y[i+1,j+1]; x4, y4 = X[i,j+1],   Y[i,j+1]
    ((x2-x1)*(y4-y1) - (y2-y1)*(x4-x1),      # at corner 1: edge 1->2 x edge 1->4
     (x3-x2)*(y1-y2) - (y3-y2)*(x1-x2),      # at corner 2: edge 2->3 x edge 2->1
     (x4-x3)*(y2-y3) - (y4-y3)*(x2-x3),      # at corner 3: edge 3->4 x edge 3->2
     (x1-x4)*(y3-y4) - (y1-y4)*(x3-x4))      # at corner 4: edge 4->1 x edge 4->3
end
"Count the cells that are not strictly convex, and the worst corner Jacobian, over j in jrange."
function _convexity_defects(X, Y, NI, jrange)
    nbad = 0; worst = Inf; where = (0, 0)
    for j in jrange, i in 1:NI
        m = minimum(_corner_jacobians(X, Y, i, j))
        m <= 0 && (nbad += 1)
        m < worst && (worst = m; where = (i, j))
    end
    (nbad, worst, where)
end

# ---- trailing-edge options (2 Oct 2026): construction, constants and helpers -----------------------------------------
# The blunt construction is the stage-1 prototype P2 "A-plus" (docs/_records/2026-10-02_blunt_te/stage1/algebraic/
# cmesh_gen_alg.jl, sha256 387ade96...) with the exact-chord te_ds and the two-segment near wake added (the measured P2x
# of docs/_records/2026-10-02_blunt_te/stage1/DECISION.md, sections 2.6 and 4). Stage 0's corner-bisector rays on the
# midpoint-split topology, made robust under the Winslow smoothing by: one junction-balanced direction per base half
# (_base_dir_maxmin); a wall fan of two first surface intervals at each corner (_te_wall_normals!); depth-filtered curved
# columns near the edge (_te_curves); a chi-tapered frozen core with a screened Winslow ramp above it (_te_winslow_sweep);
# a deeper anchor for large gaps; the base columns' outer targets downstream of the edge (_te_outer_downstream). Every
# piece runs only when n_base > 0. P2's tuned options are the constants below. ANY change to them, or to the
# construction's arithmetic, must bump TE_BASE_CONSTRUCTION ("aplus_v2", ...): every receipt names it, and the wall
# instruments refuse a run whose receipt names another version.
const TE_BASE_CONSTRUCTION = "aplus_v1"
const _TE_FAN_CELLS  = 2.0    # P2 fan_cells: corner wall fan, from the base direction to the surface normal, over 2 first surface intervals
const _TE_CURVE_C    = 0.4    # P2 curve_c: Gaussian filter width of the curved columns per unit depth (wall arc length)
const _TE_CURVE_WIN  = 0.15   # P2 curve_win: curved-column window, chords of wall distance from the base (cosine taper to twice it)
const _TE_FREEZE_H   = 5.0    # P2 te_freeze_h (freeze_taper): Winslow-frozen core of a curved column, chi * 5 h of construction arc length
const _TE_SOFT_H     = 5.0    # P2 te_soft: the screened ramp above the frozen core, 5 h
const _TE_BD_RATIO   = 1.5    # P2 te_bd_ratio: curved-column anchor depth bd + chi max(0, 1.5 (5 h + 5 h) - bd)
const _TE_OUTER_DS   = 1.0    # P2 outer_base = :downstream, outer_base_ds: base targets 1.0 ds_join apart, downstream of the edge
const _TE_MIN_SINE   = 0.2    # quality floor (P4): smallest corner sine of a delivered channel mesh with any trailing-edge option
const _TE_MIN_SWEEPS = 80     # sweep floor (G5) with n_base > 0: the construction was measured at 80-400 Winslow sweeps
const _TE_MIN_WALL   = 0.10   # anchor guard (G13): every curved column's anchor at least 0.10 c (production's anchor depth)
                              # inside the channel walls (INTEGRATION.md: measured margin in narrow channels)

# The strip topology's construction name (plan section 6.2, item 8). ANY change to the strip's arithmetic (_strip_block, the
# legs' ramp _leg_first_spacing, the leg station rule) must bump it ("strip_v2", ...): the receipt names it and the wall
# binder refuses a run whose receipt names another version for its topology. A change that leaves every mesh the
# construction has delivered bitwise keeps the name: the ramp's target max(edge, dn1) (stage-1 review gate D1) changes
# only the inputs it refused before, those whose edge interval lies inside G8''s slack below dn1.
const TE_STRIP_CONSTRUCTION = "strip_v1"

"The construction a trailing-edge topology names in the receipt: aplus_v1 for :aplus, strip_v1 for :strip; refuses any other."
function te_construction_name(topology::Symbol)
    topology === :aplus && return TE_BASE_CONSTRUCTION
    topology === :strip && return TE_STRIP_CONSTRUCTION
    throw(ArgumentError("te_topology must be :aplus or :strip (got $topology)"))
end

"""
    te_base_construction(gap; topology=:aplus) -> NamedTuple

The construction's name and, for the A-plus construction, its gap-dependent depths (chords) for the driver's receipt. The
strip construction (`topology = :strip`) has no depths: its tuple is `(construction = "strip_v1",)`.
"""
function te_base_construction(gap::Real; topology::Symbol=:aplus)
    topology === :strip && return (construction=TE_STRIP_CONSTRUCTION,)
    (construction=te_construction_name(topology),
     anchor_depth=max(0.10, _TE_BD_RATIO*(_TE_FREEZE_H*gap + _TE_SOFT_H*gap)), freeze_depth=_TE_FREEZE_H*gap,
     ramp_depth=_TE_SOFT_H*gap)
end

"Smallest corner sine over cells irange x jrange (the stage-1 harness's cell_sine; production's corner order)."
function corner_sine_min(X, Y, irange, jrange)
    smin = Inf; at = (0, 0)
    for j in jrange, i in irange
        p = ((X[i,j], Y[i,j]), (X[i+1,j], Y[i+1,j]), (X[i+1,j+1], Y[i+1,j+1]), (X[i,j+1], Y[i,j+1]))
        for c in 1:4
            a = p[c]; nx = p[mod1(c+1, 4)]; pv = p[mod1(c-1, 4)]
            e1 = (nx[1]-a[1], nx[2]-a[2]); e2 = (pv[1]-a[1], pv[2]-a[2])
            den = hypot(e1...)*hypot(e2...)
            s = den > 0 ? (e1[1]*e2[2] - e1[2]*e2[1])/den : 0.0
            s < smin && (smin = s; at = (i, j))
        end
    end
    (smin, at)
end

# The trailing-edge interval te_ds met exactly as the EUCLIDEAN CHORD between the resampled nodes. side = :lower: t runs
# TE (0) -> LE (1) over arc [s_a, s_a + L] and the TE interval is the first; side = :upper: t runs LE (0) -> TE (1) and the
# TE interval is the last. Bisection in log space on the Vinokur TE end parameter; the LE end request is kept as given.
# (A root find on the Vinokur parameter alone misses by up to 15%: the Catmull-Rom resample's speed is not 1 at the end.)
function _vinokur_te_chord(af, s_all, s_a::Float64, L::Float64, n::Int, te_ds::Float64, le_frac::Float64, side::Symbol)
    p_te = side === :lower ? _cr_eval(af.x, af.y, s_all, s_a) : _cr_eval(af.x, af.y, s_all, s_a + L)
    tv(x) = side === :lower ? _vinokur(n, x, le_frac) : _vinokur(n, le_frac, x)
    chord(x) = begin
        t = tv(x)
        p = _cr_eval(af.x, af.y, s_all, side === :lower ? s_a + t[2]*L : s_a + t[end-1]*L)
        hypot(p[1]-p_te[1], p[2]-p_te[2])
    end
    x0 = te_ds/L
    lo = log(x0/16); hi = log(min(16x0, 0.9))
    (chord(exp(lo)) < te_ds < chord(exp(hi))) || throw(ArgumentError("te_ds=$te_ds is not bracketed on the $side side"))
    for _ in 1:300
        mid = 0.5*(lo + hi)
        chord(exp(mid)) > te_ds ? (hi = mid) : (lo = mid)
        hi - lo < 1e-15 && break
    end
    x = exp(0.5*(lo + hi))
    abs(chord(x) - te_ds) <= 1e-9*te_ds ||
        throw(ArgumentError("te_ds=$te_ds met only to $(abs(chord(x) - te_ds)/te_ds) on the $side side"))
    tv(x)
end

"""
    _wake_distances(dste, NIW, wake_length; near_ds=0.0, near_len=0.0, near_growth=1.05, growth_max=1.2) -> Vector{Float64}

Cumulative distances along the channel wake cut, d[1] = 0 .. d[NIW+1] = wake_length (docs/_records/2026-10-02_blunt_te/
wake/wake_near.jl, `wake_distances`, verbatim). near_ds = 0: the single geometric distribution (first interval dste)
that the channel branch has always used, statement for statement. near_ds > 0, the two-segment near wake: intervals start
at dste and grow by near_growth per cell up to the cap near_ds until the cumulative length reaches near_len (n1 cells);
the remaining NIW - n1 cells grow geometrically from the last of them to the exact wake length. Refused, naming the wake
count that would work where one exists: n1 >= NIW, or a far-segment growth above growth_max.
"""
function _wake_distances(dste::Float64, NIW::Int, wake_length::Float64; near_ds::Float64=0.0, near_len::Float64=0.0,
                         near_growth::Float64=1.05, growth_max::Float64=1.2)
    if near_ds == 0.0
        rw = _solve_growth2(dste, NIW, wake_length)
        dw = zeros(NIW+1)
        for k in 1:NIW; dw[k+1] = dw[k] + dste*rw^(k-1); end
        dw .*= wake_length/max(dw[end], 1e-30)
        dw[end] = wake_length
        return dw
    end
    isfinite(near_ds) && near_ds > 0 && isfinite(near_len) && near_len > 0 && isfinite(near_growth) && near_growth >= 1 ||
        throw(ArgumentError("near wake: wake_near_ds and wake_near_len must be positive and wake_near_growth >= 1"))
    near_len < wake_length || throw(ArgumentError("near wake: wake_near_len $near_len is not shorter than the wake $wake_length"))
    dw = zeros(NIW+1); ds = dste; k = 0
    while dw[k+1] < near_len
        k += 1
        k < NIW || throw(ArgumentError(
            "near wake: $NIW wake cells do not reach wake_near_len=$near_len at spacing <= $near_ds; raise n_wake"))
        dw[k+1] = dw[k] + ds
        ds = min(ds*near_growth, near_ds)
    end
    n1 = k; ds_last = dw[n1+1] - dw[n1]
    rest = wake_length - dw[n1+1]; n2 = NIW - n1
    r2 = _solve_growth2(ds_last, n2, rest)
    if r2 > growth_max
        need = n1 + ceil(Int, log(1 + rest*(growth_max - 1)/ds_last)/log(growth_max))
        throw(ArgumentError("near wake: the far segment would grow by $(round(r2, digits=4)) per cell (limit $growth_max); " *
                            "n_wake must be at least $need"))
    end
    for j in 1:n2; dw[n1+1+j] = dw[n1+j] + ds_last*r2^(j-1); end
    # the far segment is scaled to end exactly at the wake length (as the single segment is scaled)
    far = dw[n1+2:end] .- dw[n1+1]
    far .*= rest/max(far[end], 1e-30)
    dw[n1+2:end] .= dw[n1+1] .+ far
    dw[end] = wake_length
    dw
end

"""
    _te_near_wake_cap_guard(dw, near_ds, near_len, n_base, te_ds, gap)

G14 (added after the verification, docs/_records/2026-10-02_blunt_te/stage1/verify_range/README.md F1), on the wake
distances `dw` that `_wake_distances` built with `near_ds > 0`. Its near segment starts at the trailing-edge interval and
caps only the intervals after it, so a cap below that first interval used to build silently with the cap exceeded (2.0x
at n_base 2 with h/8, 3.93x on a sharp edge without te_ds). Refused when any interval starting within `near_len` exceeds
`near_ds` by more than 1e-9 relative, naming the realised first and largest intervals and what sets the first: the base
interval gap/(2 n_base) with a base; without one te_ds, or the default distribution's trailing-edge interval.
"""
function _te_near_wake_cap_guard(dw::Vector{Float64}, near_ds::Float64, near_len::Float64, n_base::Int, te_ds::Float64,
                                 gap::Float64; strip::Bool=false)
    w1 = dw[2] - dw[1]; wmax = 0.0
    for k in 1:length(dw)-1
        dw[k] < near_len || break
        wmax = max(wmax, dw[k+1] - dw[k])
    end
    wmax <= near_ds*(1 + 1e-9) && return nothing
    # (strip topology, plan section 6.4: the legs start at the corner interval te_ds, not at a base interval)
    src = strip ? "te_ds=$te_ds, the corner interval at which the strip's legs start (lower te_ds within G7', or raise wake_near_ds)" :
          n_base > 0 ? "the base interval gap/(2 n_base) = $(gap/(2n_base)) (raise n_base within G8, or wake_near_ds)" :
          te_ds > 0 ? "te_ds=$te_ds (lower te_ds or raise wake_near_ds)" :
          "the default distribution's trailing-edge interval, te_ds = 0 (set te_ds <= wake_near_ds, or raise wake_near_ds)"
    throw(ArgumentError("wake_near_ds=$near_ds is exceeded within wake_near_len=$near_len: the realised first wake interval " *
        "is $w1 and the largest $wmax. The wake starts at the trailing-edge interval, which must not exceed the cap: " *
        "here $src"))
end

_unit2(v) = (v[1]/hypot(v[1], v[2]), v[2]/hypot(v[1], v[2]))
_rot90(t) = (-t[2], t[1])                                      # outward normal of a clockwise-wrap tangent
"Midpoint-split foil node list: [m..lower corner] + surface interior + [upper corner..m] (uniform base halves)."
function _blunt_foil_nodes(surf_s::Vector{Tuple{Float64,Float64}}, n_base::Int)
    CL = surf_s[1]; CU = surf_s[end]
    mpt = (0.5*(CL[1] + CU[1]), 0.5*(CL[2] + CU[2]))
    lob = [(mpt[1] + (CL[1]-mpt[1])*(k/n_base), mpt[2] + (CL[2]-mpt[2])*(k/n_base)) for k in 0:n_base]
    lob[1] = mpt; lob[end] = CL
    upb = [(CU[1] + (mpt[1]-CU[1])*(k/n_base), CU[2] + (mpt[2]-CU[2])*(k/n_base)) for k in 0:n_base]
    upb[1] = CU; upb[end] = mpt
    vcat(lob, surf_s[2:end-1], upb)
end

_wrap(a) = a - 2pi*round(a/(2pi))                              # angle difference to (-pi, pi]
"Wall-distance coordinate of the TE window: 0 on the base, arc length from the base into the wake and onto the surface."
_te_dist(sig, sig_c) = sig_c >= 0 ? (sig <= 0 ? -sig : sig >= sig_c ? sig - sig_c : 0.0) :
                                    (sig >= 0 ? sig : sig <= sig_c ? sig_c - sig : 0.0)
"""
    _te_curves(ib, nb, sides, dn1, bd, c, win; bd_te=bd) -> Dict(i => (s, x, y, chi))

One side of the wall is a contiguous index range (lower: lower wake outlet .. m .. lower corner .. LE node; upper: LE node
.. upper corner .. m .. upper wake outlet) with arc-length coordinate sig (0 at m, increasing with i) and arc-length share
w_q of each wall node. Column p of a side, at depth s along it, points along the angle
    theta_p(s) = theta0_p + chi_p * sum_q G_q wrap(theta0_q - theta0_p) / sum_q G_q,  G_q = w_q exp(-((sig_q - sig_p)/(c s))^2/2),
the wall angles theta0 filtered at a width c s that grows with depth, so a column turns where its neighbours' wall
directions differ within about c s of it: the corner fan widens with depth and the base bundle opens corner side first.
chi_p = 1 within `win` of the base (wall distance _te_dist), cosine taper to 0 at 2 win; chi = 0 columns are not returned
and keep the straight anchored rays. Each returned column is integrated to its anchor depth bd + chi*max(0, bd_te - bd)
with straight sub-steps (the first exactly dn1 long, then x1.03, at most 1/100 of the anchor depth), so the first node
lies at distance dn1 from the wall exactly. The two sides share the LE node, which lies far outside the window (chi = 0)
for any window much shorter than the chord. (P2's measured-and-rejected homing toward the outer target is not ported.)
"""
function _te_curves(ib, nb, sides, dn1, bd, c, win; bd_te::Float64=bd)
    out = Dict{Int,NamedTuple{(:s, :x, :y, :chi),Tuple{Vector{Float64},Vector{Float64},Vector{Float64},Float64}}}()
    for (ir, im, ic) in sides                                  # index range, junction index (m), corner index
        n = length(ir)
        sig = zeros(n); th = zeros(n); w = zeros(n)
        for (p, i) in enumerate(ir)
            p > 1 && (sig[p] = sig[p-1] + hypot(ib[i][1]-ib[i-1][1], ib[i][2]-ib[i-1][2]))
            th[p] = atan(nb[i][2], nb[i][1])
        end
        pm = im - first(ir) + 1; sig .-= sig[pm]               # 0 at m
        sig_c = sig[ic - first(ir) + 1]
        for p in 1:n
            w[p] = 0.5*((p > 1 ? sig[p]-sig[p-1] : 0.0) + (p < n ? sig[p+1]-sig[p] : 0.0))
        end
        for p in 1:n
            d = _te_dist(sig[p], sig_c)
            chi = d <= win ? 1.0 : d >= 2win ? 0.0 : 0.5*(1.0 + cos(pi*(d - win)/win))
            chi == 0.0 && continue
            i = ir[p]
            thp(s) = begin
                l = c*s; num = 0.0; den = 0.0
                q = p
                while q >= 1 && sig[p] - sig[q] <= 5l           # walk outward both ways while inside 5 widths
                    g = w[q]*exp(-0.5*((sig[q]-sig[p])/l)^2); num += g*_wrap(th[q]-th[p]); den += g; q -= 1
                end
                q = p + 1
                while q <= n && sig[q] - sig[p] <= 5l
                    g = w[q]*exp(-0.5*((sig[q]-sig[p])/l)^2); num += g*_wrap(th[q]-th[p]); den += g; q += 1
                end
                th[p] + (den > 0 ? chi*num/den : 0.0)
            end
            ss = [0.0]; xs = [ib[i][1]]; ys = [ib[i][2]]
            ds = dn1
            bdi = bd + chi*max(0.0, bd_te - bd)               # this column's anchor depth (bd unless the TE anchor is deeper)
            while ss[end] < bdi
                st = min(ds, bdi - ss[end])
                a = thp(ss[end] + 0.5st)
                push!(xs, xs[end] + st*cos(a)); push!(ys, ys[end] + st*sin(a)); push!(ss, ss[end] + st)
                ds = min(1.03ds, bdi/100)
            end
            ss[end] = bdi
            out[i] = (s=ss, x=xs, y=ys, chi=chi)
        end
    end
    out
end
"""
    _base_dir_maxmin(es, tb, nbase, nsurf) -> unit vector

The direction v, scanned in 1/2000 steps of the corner turn strictly between the base normal nbase and the corner's surface
normal nsurf, that maximises min(|v x es|, |v x tb|, cos(angle(v, nbase)), cos(angle(v, nsurf))): the wall-corner sines of
the wake cell (cut direction es) and the base cell (base direction tb from m toward the corner) at m, and of the base and
surface cells at the corner (corner angles 90 deg + the ray's tilt).
"""
function _base_dir_maxmin(es, tb, nbase, nsurf)
    ab = atan(nbase[2], nbase[1]); d = _wrap(atan(nsurf[2], nsurf[1]) - ab)
    best = -Inf; bv = nbase
    for k in 1:1999
        a = ab + d*k/2000; v = (cos(a), sin(a))
        s = min(abs(v[1]*es[2] - v[2]*es[1]), abs(v[1]*tb[2] - v[2]*tb[1]), cos(d*k/2000), cos(d - d*k/2000))
        s > best && (best = s; bv = v)
    end
    bv
end
"Point at arc length s along a _te_curves polyline (linear interpolation between its sub-step nodes)."
function _curve_at(cv, s)
    k = clamp(searchsortedlast(cv.s, s), 1, length(cv.s)-1)
    f = (s - cv.s[k])/(cv.s[k+1] - cv.s[k])
    (cv.x[k] + f*(cv.x[k+1]-cv.x[k]), cv.y[k] + f*(cv.y[k+1]-cv.y[k]))
end

"""
    _te_base_guards(gap, n_base, te_ds, dn1; topology=:aplus)

G6-G8 (DECISION.md 4.2) on the resampled corner-to-corner gap: an open edge, te_ds within [1, 2] base intervals, base
interval >= dn1. G7 and G8 carry the same 1e-9 relative slack (G8's was added after the verification,
docs/_records/2026-10-02_blunt_te/stage1/verify_range/README.md F4: a base interval equal to dn1 in decimal was refused by
rounding). G8 names the largest n_base its own test admits or, when none does, the requirement dn1 <= gap/2. With
te_ds > 0 the guards run before the te_ds solve (F8) and again after the resample, on the same gap.

Strip topology (`topology = :strip`, plan section 6.4): the same tests on the same number, which is now the strip's edge
cross interval gap/(2 n_base) (the corner cell's cross spacing and the legs' ramp target): G7' names it as such, and G8'
names the largest n_base admitted or, when none at or above the strip's minimum of 2 does, the requirement dn1 <= gap/4.
"""
function _te_base_guards(gap::Float64, n_base::Int, te_ds::Float64, dn1::Float64; topology::Symbol=:aplus)
    strip_mode = topology === :strip
    gap > 1e-8 || throw(ArgumentError("n_base > 0 needs an open trailing edge (resampled TE gap=$gap)"))
    if strip_mode
        gap/(2n_base)*(1 - 1e-9) <= te_ds <= gap/n_base*(1 + 1e-9) || throw(ArgumentError(
            "te_topology=:strip needs te_ds between the strip's edge cross interval $(gap/(2n_base)) (= gap/(2 n_base): the " *
            "corner cell's cross spacing and the legs' ramp target; plan section 6.4, G7') and twice it, " *
            "e.g. te_ds = $(1.5*gap/(2n_base)) (got $te_ds)"))
    else
        gap/(2n_base)*(1 - 1e-9) <= te_ds <= gap/n_base*(1 + 1e-9) || throw(ArgumentError(
            "n_base > 0 needs te_ds between the base interval $(gap/(2n_base)) and twice it (plan section 3; measured corner " *
            "ratios 1.0-2.0), e.g. te_ds = $(1.5*gap/(2n_base)) (got $te_ds)"))
    end
    if gap/(2n_base) < dn1*(1 - 1e-9)
        nok(n) = gap/(2n) >= dn1*(1 - 1e-9)                    # G8's own test, for n cells per base half
        nmax = floor(Int, gap/(2dn1))
        while nmax >= 1 && !nok(nmax); nmax -= 1; end
        while nok(nmax + 1); nmax += 1; end
        strip_mode && throw(ArgumentError("the strip's edge cross interval $(gap/(2n_base)) (= gap/(2 n_base)) is below " *
            "dn1=$dn1 (plan section 6.4, G8'); " * (nmax >= 2 ? "use n_base <= $nmax" :
            "no strip can meet dn1 at this gap: te_topology=:strip needs n_base >= 2, i.e. dn1 <= gap/4 = $(gap/4)")))
        throw(ArgumentError("the base interval $(gap/(2n_base)) is below dn1=$dn1 (plan section 3); " * (nmax >= 1 ?
            "use n_base <= $nmax" : "no base can meet dn1 at this gap: n_base >= 1 needs dn1 <= gap/2 = $(gap/2)")))
    end
    nothing
end

"""
    _te_ds_corner_gap(af, s_all, s_le, s_tot) -> gap

The corner-to-corner gap of the te_ds resample, before its solve. The resample's end nodes sit at t = 0 and t = 1 exactly
(`_vinokur` sets both), so this is the resampled gap `te_gap_base` bit for bit, and G6-G8 can run before the te_ds solve
(verify_range/README.md F8: a tiny te_ds reached the solve's refusal G11 first, which named the wrong cause).
"""
function _te_ds_corner_gap(af, s_all, s_le::Float64, s_tot::Float64)
    lo = _cr_eval(af.x, af.y, s_all, 0.0*s_le)                    # surf_s[1], the lower corner (tl[1] = 0.0)
    up = _cr_eval(af.x, af.y, s_all, s_le + 1.0*(s_tot - s_le))   # surf_s[end], the upper corner (tu[end] = 1.0)
    hypot(lo[1]-up[1], lo[2]-up[2])
end

"""
    _te_base_dirs(surf_s, surf, n_base, es) -> (b_lo, b_up)

One uniform wall direction per base half (P2 lines 612-624, base_dir = :maxmin): `_base_dir_maxmin` between the base
normal and each corner's surface normal, with es the wake-cut direction. They replace the one-sided trailing-edge normals
from which the wake normals blend into the channel normal.
"""
function _te_base_dirs(surf_s, surf, n_base::Int, es)
    NIF_s = length(surf_s) - 1; NIF = length(surf) - 1
    n_surf_lo = _rot90(_unit2((surf_s[2][1]-surf_s[1][1], surf_s[2][2]-surf_s[1][2])))   # lower surface's last segment
    n_surf_up = _rot90(_unit2((surf_s[NIF_s+1][1]-surf_s[NIF_s][1], surf_s[NIF_s+1][2]-surf_s[NIF_s][2])))
    n_basev = _rot90(_unit2((surf_s[1][1]-surf_s[end][1], surf_s[1][2]-surf_s[end][2])))  # base, upper -> lower corner
    mpt = surf[1]; CL = surf[n_base+1]; CU = surf[NIF+1-n_base]
    b_lo = _base_dir_maxmin(es, _unit2((CL[1]-mpt[1], CL[2]-mpt[2])), n_basev, n_surf_lo)
    b_up = _base_dir_maxmin(es, _unit2((CU[1]-mpt[1], CU[2]-mpt[2])), n_basev, n_surf_up)
    (b_lo, b_up)
end

"""
    _te_wall_normals!(nb, surf, surf_s, NIW, n_base, b_lo, b_up)

Wall directions of the junction, base and fan nodes (P2 lines 656-682, ray_mode :bisector, fan_cells 2): every node of a
base half, m and the corner included, takes that half's direction; the surface nodes within _TE_FAN_CELLS first surface
intervals of a corner blend from it into their own normal with a cosine. The upper junction node is then set by the upper
wake loop (to b_up, at wake distance 0).
"""
function _te_wall_normals!(nb, surf, surf_s, NIW::Int, n_base::Int, b_lo, b_up)
    NIF_s = length(surf_s) - 1; NIF = length(surf) - 1
    nb[NIW+1] = b_lo                                           # lower junction node m
    nb0 = copy(nb)                                             # the foil loop's central-difference normals
    for k in 1:n_base+1; nb[NIW+k] = b_lo; end                 # lower base: m .. lower corner
    for k in NIF+1-n_base:NIF+1; nb[NIW+k] = b_up; end         # upper base: upper corner .. m
    Lc_lo = _TE_FAN_CELLS*hypot(surf_s[2][1]-surf_s[1][1], surf_s[2][2]-surf_s[1][2])
    Lc_up = _TE_FAN_CELLS*hypot(surf_s[NIF_s+1][1]-surf_s[NIF_s][1], surf_s[NIF_s+1][2]-surf_s[NIF_s][2])
    sarc = 0.0
    for k in n_base+2:NIF-n_base                               # lower surface, walking away from the lower corner
        sarc += hypot(surf[k][1]-surf[k-1][1], surf[k][2]-surf[k-1][2])
        sarc >= Lc_lo && break
        w = 0.5*(1.0 + cos(pi*sarc/Lc_lo))
        nb[NIW+k] = _unit2((w*b_lo[1] + (1.0-w)*nb0[NIW+k][1], w*b_lo[2] + (1.0-w)*nb0[NIW+k][2]))
    end
    sarc = 0.0
    for k in NIF-n_base:-1:n_base+2                            # upper surface, walking away from the upper corner
        sarc += hypot(surf[k][1]-surf[k+1][1], surf[k][2]-surf[k+1][2])
        sarc >= Lc_up && break
        w = 0.5*(1.0 + cos(pi*sarc/Lc_up))
        nb[NIW+k] = _unit2((w*b_up[1] + (1.0-w)*nb0[NIW+k][1], w*b_up[2] + (1.0-w)*nb0[NIW+k][2]))
    end
    nb
end

"""
    _te_outer_downstream(NI, NIW, NIF_s, n_base, nlo, nup, upstream_length, h, wake_length, xi_te, xi_in, to_physical)
        -> (ob, outer_face_kind)

The channel's outer boundary with a base (P2 lines 699-729, outer_base = :downstream). The surface columns keep exactly
the sharp mesh's outer targets (side counts nlo, nup). Each base half's n_base targets are inserted on its own channel
wall DOWNSTREAM of the trailing-edge projection xi_te, at spacing ds_b = _TE_OUTER_DS ds_join, where the base bundle
points; the outer wake legs start at xi_te + n_base ds_b with first spacing ds_join. The node shared by the wake and base
outer runs is computed by one expression from both loops, so the outer boundary is continuous and i-monotone.
"""
function _te_outer_downstream(NI, NIW, NIF_s, n_base, nlo, nup, upstream_length, h, wake_length, xi_te, xi_in, to_physical)
    NIF = NIF_s + 2n_base
    nlw = clamp(round(Int, nlo*upstream_length/(upstream_length+h)), 1, nlo-1)
    nli = nlo - nlw
    nui = clamp(round(Int, nup*h/(h+upstream_length)), 1, nup-1)
    nuw = nup - nui
    ds_join = 0.5*(upstream_length/nlw + upstream_length/nuw)
    ds_b = _TE_OUTER_DS*ds_join
    dlt = n_base*ds_b
    if !(dlt < 0.5*wake_length)                                # the cap on n_base (verify_range/README.md F5)
        nmax = floor(Int, 0.5*wake_length/ds_b)                # the largest n_base this test admits
        while nmax >= 1 && !(nmax*ds_b < 0.5*wake_length); nmax -= 1; end
        while (nmax + 1)*ds_b < 0.5*wake_length; nmax += 1; end
        throw(ArgumentError("n_base > 0: the base columns' outer run $(dlt) (n_base x the outer wall spacing ds_join " *
            "$(ds_b)) must be shorter than half the wake length, $(0.5*wake_length) (wake $(wake_length)); " *
            (nmax >= 1 ? "use n_base <= $nmax, or more foil points (ds_join falls with n_foil)" :
                         "use more foil points (ds_join falls with n_foil)")))
    end
    rwo = _solve_growth2(ds_join, NIW, wake_length - dlt)
    swo = zeros(NIW+1)
    for k in 1:NIW; swo[k+1] = swo[k] + ds_join*rwo^(k-1); end
    swo .*= (wake_length - dlt)/max(swo[end], 1e-30)
    swo .+= dlt; swo[1] = dlt; swo[end] = wake_length
    ob = Vector{Tuple{Float64,Float64}}(undef, NI+1)
    for k in 1:NIW+1; ob[k] = to_physical(xi_te+swo[NIW+2-k], -h); end
    for q in 0:n_base; ob[NIW+1+q] = to_physical(xi_te + (dlt - q*ds_b), -h); end
    for q in 0:nlo
        xi, eta = q <= nlw ? (xi_te - q*upstream_length/nlw, -h) : (xi_in, -h + (q-nlw)*h/nli)
        ob[NIW+n_base+q+1] = to_physical(xi, eta)
    end
    for q in 1:nup
        xi, eta = q <= nui ? (xi_in, q*h/nui) : (xi_in + (q-nui)*upstream_length/nuw, h)
        ob[NIW+n_base+nlo+q+1] = to_physical(xi, eta)
    end
    for q in 1:n_base; ob[NIW+n_base+NIF_s+1+q] = to_physical(xi_te + (dlt - (n_base-q)*ds_b), h); end
    for k in 1:NIW+1; ob[NIW+NIF+k] = to_physical(xi_te+swo[k], h); end
    outer_face_kind = Vector{UInt8}(undef, NI)
    outer_face_kind[1:NIW+n_base+nlw] .= OUTER_LOWER_WALL
    outer_face_kind[NIW+n_base+nlw+1:NIW+n_base+nlo+nui] .= OUTER_INLET
    outer_face_kind[NIW+n_base+nlo+nui+1:NI] .= OUTER_UPPER_WALL
    (ob, outer_face_kind)
end

"""
    _te_curved_columns!(X, Y, growth, ib, nb, ob, NI, NJ, NIW, NIF, n_base, nlo, dn1, bd, gap)
        -> (curves, femask, wsoft, bd_te)

Rebuild the columns within the trailing-edge window as curved columns (P2 lines 791-809 and 831-854; every other column
keeps the straight anchored ray already in X, Y). A curved column follows its `_te_curves` polyline to its anchor depth
(bd + chi max(0, bd_te - bd), bd_te = max(bd, 1.5 (5 h + 5 h)), h the corner-to-corner gap), then the straight segment
to its outer target, with the generator's geometric dn1 schedule. Returned with the polylines: the Winslow-frozen core
mask (nodes within chi 5 h of construction arc length) and the screening weight of the ramp above it (chi falling to 0
over 5 h), both zero outside the curved columns.
"""
function _te_curved_columns!(X, Y, growth, ib, nb, ob, NI::Int, NJ::Int, NIW::Int, NIF::Int, n_base::Int, nlo::Int,
                             dn1::Float64, bd::Float64, gap::Float64)
    i_le = NIW + n_base + nlo + 1                              # LE node: the lower and upper sides meet here
    Dfz = _TE_FREEZE_H*gap                                     # frozen-core depth at chi = 1
    bd_te = max(bd, _TE_BD_RATIO*(Dfz + _TE_SOFT_H*gap))       # anchor depth at chi = 1 (te_base_construction's anchor_depth)
    curves = _te_curves(ib, nb, ((1:i_le, NIW+1, NIW+n_base+1), (i_le:NI+1, NIW+NIF+1, NIW+NIF+1-n_base)), dn1, bd,
                        _TE_CURVE_C, _TE_CURVE_WIN; bd_te=bd_te)
    femask = falses(NI+1, NJ+1)
    Rsoft = _TE_SOFT_H*gap
    wsoft = zeros(NI+1, NJ+1)
    for i in 1:NI+1
        cv = get(curves, i, nothing)
        cv === nothing && continue
        xi, yi = ib[i]; xo, yo = ob[i]
        bdi = cv.s[end]                                        # this column's anchor depth
        ax, ay = cv.x[end], cv.y[end]                          # anchor: the curved column at arc length bdi
        L2 = hypot(xo-ax, yo-ay)
        Lpath = bdi + L2
        r = _solve_growth2(dn1, NJ, Lpath); growth[i] = r
        acc = 0.0
        X[i,1] = xi; Y[i,1] = yi
        for j in 1:NJ
            acc += dn1 * r^(j-1)
            if acc <= bdi
                X[i,j+1], Y[i,j+1] = _curve_at(cv, acc)
            else
                f = min((acc - bdi)/L2, 1.0)
                X[i,j+1] = ax + f*(xo-ax); Y[i,j+1] = ay + f*(yo-ay)
            end
            femask[i,j+1] = acc <= cv.chi*Dfz
            Dc = cv.chi*Dfz
            wsoft[i,j+1] = acc <= Dc ? cv.chi : acc >= Dc + Rsoft ? 0.0 : cv.chi*0.5*(1.0 + cos(pi*(acc - Dc)/Rsoft))
        end
        X[i,NJ+1] = xo; Y[i,NJ+1] = yo                         # exact outer landing
        femask[i,NJ+1] = false
        wsoft[i,NJ+1] = 0.0
    end
    (curves=curves, femask=femask, wsoft=wsoft, bd_te=bd_te)
end

"""
    _te_anchor_clearance(curves, ch) -> (clearance, i)

The smallest distance (chords) from a curved column's anchor to the nearer wall of channel `ch`, negative beyond the
wall, and the column (absolute i) where it occurs. The anchor depth grows with the gap, max(0.10, 15 h), so a thick edge in
a narrow channel drives the trailing-edge anchors into the walls.
"""
function _te_anchor_clearance(curves, ch::ChannelSpec)
    esx, esy = ch.streamwise; enx, eny = -esy, esx; px, py = ch.ref; h = 0.5*ch.height
    cmin = Inf; at = 0
    for i in sort!(collect(keys(curves)))
        cv = curves[i]
        c = h - abs((cv.x[end]-px)*enx + (cv.y[end]-py)*eny)
        c < cmin && (cmin = c; at = i)
    end
    (cmin, at)
end

"G13 (the integration's anchor guard): refuse a blunt mesh whose trailing-edge anchors come within _TE_MIN_WALL of a channel wall."
function _te_anchor_guard(curves, ch::ChannelSpec, bd_te::Float64, NIW::Int)
    clr, i = _te_anchor_clearance(curves, ch)
    clr >= _TE_MIN_WALL || throw(ArgumentError("n_base > 0: a trailing-edge anchor lies $clr c inside a channel wall " *
        "(negative: beyond it) at column i-NIW=$(i - NIW); the anchor depth is $bd_te c = max(0.10, 15 x gap) and the blunt " *
        "construction needs at least 0.10 c (measured: every build with an anchor at or beyond a wall failed) -- use a " *
        "thinner trailing edge or a wider channel"))
    nothing
end

"The frozen-block repair's shared-spacing placement along the curved columns (P2 lines 877-886): node j at arc length j dn1."
function _te_frozen_placement!(X, Y, curves, nbl::Int, dn1::Float64)
    for (i, cv) in curves, j in 1:nbl
        X[i,j+1], Y[i,j+1] = _curve_at(cv, j*dn1)
    end
    nothing
end

"""
    _te_winslow_sweep(X, Y, NI, NJ, bl_freeze, femask, wsoft) -> closure

The Winslow sweep of a blunt mesh (P2 lines 901-924): the generator's update everywhere wsoft == 0, nodes of the frozen
core (femask) skipped, and above the core the screened update X += 0.3((1-ws)(x_winslow - X) + ws(X0 - X)) toward the
construction X0 (copied here, after any frozen-block repair). Every captured variable is assigned once (a reassigned
captured variable is boxed: P1 measured a 50x slowdown).
"""
function _te_winslow_sweep(X, Y, NI::Int, NJ::Int, bl_freeze::Int, femask, wsoft)
    X0s = copy(X); Y0s = copy(Y)
    () -> begin
        for j in bl_freeze+2:NJ, i in 2:NI
            femask[i,j] && continue                            # the frozen core of a curved column
            xxi = 0.5*(X[i+1,j]-X[i-1,j]); yxi = 0.5*(Y[i+1,j]-Y[i-1,j])
            xet = 0.5*(X[i,j+1]-X[i,j-1]); yet = 0.5*(Y[i,j+1]-Y[i,j-1])
            a = xet^2 + yet^2; g = xxi^2 + yxi^2; bb = xxi*xet + yxi*yet
            den = 2*(a + g); den < 1e-14 && continue
            xnew = (a*(X[i+1,j]+X[i-1,j]) + g*(X[i,j+1]+X[i,j-1]) -
                    0.5*bb*(X[i+1,j+1]-X[i-1,j+1]-X[i+1,j-1]+X[i-1,j-1])) / den
            ynew = (a*(Y[i+1,j]+Y[i-1,j]) + g*(Y[i,j+1]+Y[i,j-1]) -
                    0.5*bb*(Y[i+1,j+1]-Y[i-1,j+1]-Y[i+1,j-1]+Y[i-1,j-1])) / den
            if wsoft[i,j] == 0.0
                X[i,j] = 0.7*X[i,j] + 0.3*xnew; Y[i,j] = 0.7*Y[i,j] + 0.3*ynew
            else                                               # the screened update toward the construction
                ws = wsoft[i,j]
                X[i,j] += 0.3*((1.0-ws)*(xnew - X[i,j]) + ws*(X0s[i,j] - X[i,j]))
                Y[i,j] += 0.3*((1.0-ws)*(ynew - Y[i,j]) + ws*(Y0s[i,j] - Y[i,j]))
            end
        end
        nothing
    end
end

"P1-P4 (DECISION.md 4.4): the checks of every channel mesh built with a trailing-edge option, after the build."
function _te_post_checks(X, Y, NI::Int, NJ::Int, NIW::Int, NIF::Int)
    wpe = maximum(hypot(X[k,1]-X[NI+2-k,1], Y[k,1]-Y[NI+2-k,1]) for k in 1:NIW+1)
    wpe == 0.0 || error("cmesh (trailing-edge options): wake partners not coincident (max $wpe)")
    sx = 0.0; sy = 0.0
    for i in NIW+1:NIW+NIF; sx += -(Y[i+1,1]-Y[i,1]); sy += X[i+1,1]-X[i,1]; end
    hypot(sx, sy) <= 1e-12 || error("cmesh (trailing-edge options): closed-loop face sum $(hypot(sx, sy)) exceeds 1e-12")
    nbad, worst, where = _convexity_defects(X, Y, NI, 1:NJ)
    nbad == 0 || error("cmesh (trailing-edge options): $nbad non-convex cells (worst corner Jacobian $worst at (i,j)=$where)")
    smin, sat = corner_sine_min(X, Y, 1:NI, 1:NJ)
    # P4's text states only what was measured on refused meshes (it named "n_wake 64, or smooth_iters 800" until the
    # verification found that wrong for the cases it measured: verify_range/README.md F3)
    smin >= _TE_MIN_SINE || throw(ArgumentError("cmesh: minimum corner sine $smin at (i,j)=$sat is below the 0.2 floor of the " *
        "trailing-edge options. Measured on refused meshes (docs/_records/2026-10-02_blunt_te/stage1/verify_range/README.md " *
        "F2-F3; DECISION.md 2.6(c)): smooth_iters 800 lifted every one tried except the near wake in the Leroux channel " *
        "(H/c 1.28), which nothing tried rescued, and 1600 broke a near-wake mesh that 80-800 built; fewer wake cells " *
        "lifted some plain-wake ones and not others (stage1/verify_guardfix.md V6) and more cells lowered them, as they " *
        "lowered a WP3L near wake, but raised the Leroux near wake (still below 0.2 at 300 cells)"))
    nothing
end

# ---- strip topology (5 Oct 2026): the legs' ramp, the strip block and its post checks ---------------------------------
"""
    _leg_first_spacing(dn1, growth_max, edge, NIW, NIF) -> dn1_i (length 2 NIW + NIF + 1)

The strip topology's per-column first wall-normal spacing (plan section 6.3): dn1 on every surface column, the two corner
columns included (station 1 of each leg), then min(max(edge, dn1), dn1 growth_max^(s-1)) at wake station s >= 2 on both
legs, `edge` being the strip's edge cross interval gap/(2 n_base). The lower leg's station s is column NIW+2-s, the upper's
column NIW+NIF+s, so the two legs carry the same ramp. With te_topology = :aplus the generator uses fill(dn1, NI+1)
instead: the same Float64 in the same operations at the five sites that read it.

The ramp only raises. Its target is max(edge, dn1), not edge: G8' admits an edge interval down to dn1 (1 - 1e-9), the
slack G8 carries so that a base interval equal to dn1 in decimal is not refused by rounding (verify_range/README.md F4),
and with edge at or below dn1 the legs keep dn1 (no ramp station; the receipt's leg_ramp_stations reads 0 and
leg_row1_ratio 1). The edge alone as the target set every wake station's first spacing one ulp below dn1 there, which
_column_schedule refuses as an internal inconsistency while the A-plus twin built (stage-1 review gate D1,
docs/_records/2026-10-05_strip_topology/stage1/REVIEW_GATE.md section 9). With edge >= dn1, max(edge, dn1) is edge itself,
so every strip mesh that built before the fix is bitwise unchanged.
"""
function _leg_first_spacing(dn1::Float64, growth_max::Float64, edge::Float64, NIW::Int, NIF::Int)
    d = fill(dn1, 2NIW + NIF + 1)
    target = max(edge, dn1)                                    # the ramp only raises (D1): edge itself whenever edge >= dn1
    for s in 2:NIW+1
        v = min(target, dn1*growth_max^(s-1))
        d[NIW+2-s] = v; d[NIW+NIF+s] = v
    end
    d
end

"""
    _column_schedule(dn1_i, dn1, r, NJ, L) -> s (length NJ+1)

The cumulative wall-normal distances s[1] = 0 .. s[NJ+1] ~ L of a column's NJ+1 nodes. With dn1_i == dn1 (every column
of a sharp or A-plus mesh, and the surface and corner columns of a strip mesh) it is the generator's geometric schedule,
s[j+1] = s[j] + dn1 r^(j-1), accumulated as the column loop and the reparameterisation always accumulated it (the same
Float64 in the same operations; r is the column's ratio from _solve_growth2(dn1, NJ, L)). On a strip mesh's leg column
with the ramp's larger first spacing dn1_i (plan section 6.3): the first interval is dn1_i, the following intervals
shrink by the column's own ratio, dn1_i r^-(j-1), until the first row m at which the standard interval dn1 r^(j-1) is at
least that (the fastest return to the standard schedule the growth admission allows, since the adjacent-interval ratio
is 1/r there), and from row m on the standard intervals scaled by one constant c = (L - s[m])/(L - s_std[m]) so that the
column still ends at L; the ratio at the junction lies within [c/r, c r]. A first spacing below dn1 is refused as an
internal inconsistency: the generator's ramp only raises it (_leg_first_spacing's target max(edge, dn1), stage-1 review
gate D1), so no generator input reaches that refusal.

Why not a geometric schedule from dn1_i (reading dn1_i at the schedule sites, as plan section 6.3 is written): measured
on BB8-S, it lifts every row of a ramp column by the ratio of its first spacing to dn1 (the j = 8 heights grow from
4.5e-4 to 3.1e-3 c over the 12 ramp columns, and at j = 66 the corner column sits at 0.086 c and the far ramp columns
above 0.2 c, 0.009 c away), so the j-lines across the ramp tilt to 80-85 degrees: corner sine 0.011 (P4) and a folded
cell under the reparameterisation. A uniform piece of dn1_i up to the standard schedule (measured too) tilts them by
(j-1)(dn1_s+1 - dn1_s) per station over tens of rows: sine 0.22 on TA-S and below the floor on a 4% c edge. The shrink
confines the mismatch between neighbouring ramp columns to about 0.2 dn1_s/(1 - 1/r), under their station spacing, and
every row above m stays where the sharp construction puts it (record docs/_records/2026-10-05_strip_topology/stage1/README.md).
"""
function _column_schedule(dn1_i::Float64, dn1::Float64, r::Float64, NJ::Int, L::Float64)
    s = zeros(NJ+1)
    acc = 0.0
    for j in 1:NJ
        acc += dn1 * r^(j-1)
        s[j+1] = acc
    end
    dn1_i == dn1 && return s
    dn1_i > dn1 || throw(ArgumentError("a column's first spacing $dn1_i is below dn1=$dn1 (the legs' ramp only raises it)"))
    r > 1 || throw(ArgumentError("the legs' ramp needs a growing wall-normal schedule (column ratio $r): the column is too " *
        "short for its $NJ cells of dn1=$dn1"))
    m = 0
    for j in 2:NJ
        if dn1 * r^(j-1) >= dn1_i * r^(-(j-1)); m = j; break; end
    end
    m >= 2 || throw(ArgumentError("the legs' ramp from $dn1_i does not meet the column's schedule (ratio $r) within its $NJ " *
        "cells: raise n_normal, or lower the strip's edge interval with more base cells (n_base)"))
    hs = s[m]                                                  # the standard height of row m, before it is overwritten
    acc = 0.0
    for j in 1:m-1
        acc += dn1_i * r^(-(j-1))
        s[j+1] = acc
    end
    L > acc || throw(ArgumentError("the legs' ramp ($(m-1) cells from $dn1_i) is longer than the column ($L): raise " *
        "n_normal or lower the strip's edge interval with more base cells (n_base)"))
    c = (L - acc)/(L - hs)
    for j in m:NJ
        acc += c*(dn1 * r^(j-1))
        s[j+1] = acc
    end
    s
end

"""
    _strip_block(X, Y, NIW, NIF, NS) -> (Xs, Ys, t)

The strip's nodes (plan section 6.2, item 6), from the C block's final row j = 1: lower[k] = node (NIW+2-k, 1) on the lower
leg and upper[k] = node (NIW+NIF+k, 1) on the upper leg, k = 1..NIW+1 (k = 1 the corners, k = NIW+1 the outlet). Column k
of the strip is the linear interpolation between them at the uniform cross parameter t_j = (j-1)/NS, j = 1..NS+1, written as
lower + t (upper - lower): rows 1 and NS+1 are ASSIGNED the leg nodes, so the edge identities P1' hold bitwise by
construction; column 1 is the base and column NIW+1 lies on the outlet plane to rounding; and when the two legs share a
coordinate at a station (alpha 0: the same x), every interior node of that column shares it bitwise (identity (c) of plan
section 7: a rectangular strip). Every cell is a trapezoid: its two streamwise edges are parallel (both along the
channel direction), and its two cross edges differ by the change, across the cell, of the legs' station difference
dU - dL divided by NS (the legs are rescaled to lengths differing by h sin(tilt); the difference is 0 over a near wake,
where both legs carry the same intervals, and grows to h sin(tilt) in the far wake: 1.55% of the cross edge at most on
BB8-S, in its last column). The column's skew against the channel normal falls from the base's tilt at column 1 to 0 at the
outlet column, so the corner sines are at least the cosine of that tilt; at alpha 0 on a base normal to the chord the
cells are exact rectangles.
"""
function _strip_block(X, Y, NIW::Int, NIF::Int, NS::Int)
    Xs = zeros(NIW+1, NS+1); Ys = zeros(NIW+1, NS+1)
    t = [(j-1)/NS for j in 1:NS+1]
    for k in 1:NIW+1
        xl, yl = X[NIW+2-k, 1], Y[NIW+2-k, 1]
        xu, yu = X[NIW+NIF+k, 1], Y[NIW+NIF+k, 1]
        Xs[k, 1] = xl; Ys[k, 1] = yl; Xs[k, NS+1] = xu; Ys[k, NS+1] = yu
        for j in 2:NS
            Xs[k, j] = xl + t[j]*(xu - xl); Ys[k, j] = yl + t[j]*(yu - yl)
        end
    end
    (Xs, Ys, t)
end

"P1' as a number: the largest distance between a strip edge node and its leg node (0 on every delivered strip mesh)."
function _strip_edge_error(X, Y, Xs, Ys, NIW::Int, NIF::Int, NS::Int)
    e = 0.0
    for k in 1:NIW+1
        e = max(e, hypot(Xs[k,1]-X[NIW+2-k,1], Ys[k,1]-Y[NIW+2-k,1]),
                   hypot(Xs[k,NS+1]-X[NIW+NIF+k,1], Ys[k,NS+1]-Y[NIW+NIF+k,1]))
    end
    e
end

"""
    _te_post_checks_strip(X, Y, Xs, Ys, NI, NJ, NIW, NIF, NS, channel)

The post-build checks of a strip mesh (plan section 6.4): P1' the two edge identities Xs[k,1] == X[NIW+2-k,1] and
Xs[k,NS+1] == X[NIW+NIF+k,1] (and Y), bitwise, replacing P1; P2' the closed-loop face sum of the foil faces plus the
strip's column-1 faces (the base; the foil range alone sums to minus the base vector) <= 1e-12; P3 strict convexity and
P4 the 0.2 corner-sine floor on both blocks; P5 column 1 on the base segment and column NIW+1 on the outlet plane to
1e-12 relative, and every node between its column's two leg nodes (the transfinite interpolation is exact).
"""
function _te_post_checks_strip(X, Y, Xs, Ys, NI::Int, NJ::Int, NIW::Int, NIF::Int, NS::Int, ch::ChannelSpec)
    for k in 1:NIW+1
        (Xs[k,1] == X[NIW+2-k,1] && Ys[k,1] == Y[NIW+2-k,1]) ||
            error("cmesh (strip topology): the strip's lower edge node $k is not the lower leg's node $(NIW+2-k) bitwise (P1')")
        (Xs[k,NS+1] == X[NIW+NIF+k,1] && Ys[k,NS+1] == Y[NIW+NIF+k,1]) ||
            error("cmesh (strip topology): the strip's upper edge node $k is not the upper leg's node $(NIW+NIF+k) bitwise (P1')")
    end
    sx = 0.0; sy = 0.0
    for i in NIW+1:NIW+NIF; sx += -(Y[i+1,1]-Y[i,1]); sy += X[i+1,1]-X[i,1]; end          # the surface's j-faces (+j into the fluid)
    for j in 1:NS; sx += Ys[1,j+1]-Ys[1,j]; sy += -(Xs[1,j+1]-Xs[1,j]); end              # the strip's column-1 i-faces (+i, downstream)
    hypot(sx, sy) <= 1e-12 || error("cmesh (strip topology): closed-loop face sum of the foil faces plus the base " *
                                    "$(hypot(sx, sy)) exceeds 1e-12 (P2')")
    nbad, worst, where = _convexity_defects(X, Y, NI, 1:NJ)
    nbad == 0 || error("cmesh (strip topology): $nbad non-convex cells in the C block (worst corner Jacobian $worst at (i,j)=$where)")
    nbad, worst, where = _convexity_defects(Xs, Ys, NIW, 1:NS)
    nbad == 0 || error("cmesh (strip topology): $nbad non-convex cells in the strip (worst corner Jacobian $worst at (i,j)=$where)")
    smin, sat = corner_sine_min(X, Y, 1:NI, 1:NJ)
    smin >= _TE_MIN_SINE || throw(ArgumentError("cmesh: minimum corner sine $smin at (i,j)=$sat of the C block is below the " *
        "0.2 floor of the trailing-edge options (strip topology: the C block is the sharp construction with the legs leaving " *
        "the corners; the levers measured on the sharp control are the wake count and the sweeps, " *
        "docs/_records/2026-10-02_blunt_te/stage1/verify_range/README.md F2-F3)"))
    smin_s, sat_s = corner_sine_min(Xs, Ys, 1:NIW, 1:NS)
    smin_s >= _TE_MIN_SINE || throw(ArgumentError("cmesh: minimum corner sine $smin_s at strip cell (i,j)=$sat_s is below the " *
        "0.2 floor (the strip's cells are trapezoids skewed by the base's tilt against the channel normal; a value this " *
        "low means a leg is not parallel to the channel or the base is nearly streamwise)"))
    # P5: column 1 on the base segment, column NIW+1 on the outlet plane, every node between its two leg nodes
    esx, esy = ch.streamwise; px, py = ch.ref
    xi(x, y) = ch.xi_ref + (x-px)*esx + (y-py)*esy
    scale = max(1.0, ch.height, abs(ch.xi_in), abs(ch.xi_out))
    xl, yl, xu, yu = Xs[1,1], Ys[1,1], Xs[1,NS+1], Ys[1,NS+1]
    gap = hypot(xu-xl, yu-yl)
    for j in 1:NS+1
        off = abs((xu-xl)*(Ys[1,j]-yl) - (yu-yl)*(Xs[1,j]-xl))/gap
        off <= 1e-12*gap || error("cmesh (strip topology): strip column 1 node $j lies $off c off the base segment (P5)")
        dxo = abs(xi(Xs[NIW+1,j], Ys[NIW+1,j]) - ch.xi_out)
        dxo <= 1e-12*scale || error("cmesh (strip topology): strip column $(NIW+1) node $j misses the outlet plane xi=$(ch.xi_out) by $dxo (P5)")
    end
    for k in 1:NIW+1, j in 2:NS
        ax, ay = Xs[k,1], Ys[k,1]; bx, by = Xs[k,NS+1], Ys[k,NS+1]; L = hypot(bx-ax, by-ay)
        off = abs((bx-ax)*(Ys[k,j]-ay) - (by-ay)*(Xs[k,j]-ax))/L
        tt = ((Xs[k,j]-ax)*(bx-ax) + (Ys[k,j]-ay)*(by-ay))/(L*L)
        (off <= 1e-12*L && -1e-12 <= tt <= 1 + 1e-12) ||
            error("cmesh (strip topology): strip node ($k,$j) is not between its two leg nodes (P5): off the segment by $off c, parameter $tt")
    end
    nothing
end

"""
    generate_cmesh(af; n_foil, n_wake, n_normal, dn1, growth_max, R_ff, L_wake, cluster_le, cluster_te, smooth_iters,
                   bl_freeze, channel, smooth_block, smooth_max_extra, le_ds,
                   n_base, te_ds, wake_near_ds, wake_near_len, wake_near_growth, te_topology) -> CMesh or ChannelMesh

With `channel=nothing` (the default), the legacy far-field C-grid path is unchanged. Passing a
`ChannelSpec` replaces the circular far field by a freestream-aligned rectangular U-boundary and
aligns the coincident wake cut with the same streamwise direction.

Trailing-edge options (2 Oct 2026; every one inert at its default, see the file header), all on the channel branch:
`n_base` cells per half of a blunt trailing-edge base (needs an open edge, `te_ds` within 1-2 base intervals, base
interval >= `dn1` (both to 1e-9 relative), `smooth_iters >= 80`, and its trailing-edge anchors at least 0.10 c inside
the channel walls), `te_ds` the surface interval at the trailing edge (corners; with `n_base = 0` at least `dn1`, G15),
and the near wake `wake_near_ds`, `wake_near_len`, `wake_near_growth` (finite and >= 1, checked even with the near wake
off), whose cap must not be below the trailing-edge interval at which the wake starts (G14). A mesh built with any of
them is checked after the build: wake partners coincident exactly, closed-loop face sum <= 1e-12, every cell strictly
convex, smallest corner sine >= 0.2. The free-air branch refuses an open trailing edge.

Measured limits (docs/_records/2026-10-02_blunt_te/stage1/verify_range/README.md):
- The outer boundary caps `n_base`: the base columns' outer targets run n_base x ds_join downstream of the edge (ds_join,
  the outer wall spacing there, falls with `n_foil`), and the run must be shorter than half the wake length, or the build
  is refused naming the largest `n_base` admitted. Measured (F5): in the NACA channel (H 20, inlet -20 c, outlet 10.5 c)
  at 256 points n_base 19 builds and 20 is refused (the 1.40% c cut at alpha 3), and n_base 24 is refused for every
  section and incidence tried; in the WP3L channel n_base 64 is refused at 256 points.
- The near wake cannot be built in the Leroux channel (H/c 1.28): 0 of 37 builds at the minimum wake counts (gaps
  0.12-1.5% c, n_base 4/8/16, 256 and 512 points, alpha 4 and 8), 35 refused by the 0.2 corner-sine floor (sines
  0.007-0.133) and 2 by the reparameterisation's convexity error. Up to 300 wake cells and 400-1600 sweeps did not rescue
  it; the sharp control with the same near wake fails the same way, while the blunt mesh without it builds (F2).

Strip topology (5 Oct 2026, docs/2026-10-05_STRIP_TOPOLOGY_PLAN.md section 6): `te_topology = :strip` (default `:aplus`,
the construction above) needs a channel and `n_base >= 2` (G-S1), an open edge (G6), `te_ds` within [1, 2] strip edge
intervals gap/(2 n_base) (G7') and that interval at least `dn1` (G8'); `smooth_iters >= 80` is not required. The C block is
the sharp construction with `NIF = n_foil`, its two legs leaving the two corners with their own station vectors (G12 and
G14 run per leg); the strip is the linear transfinite interpolation between the legs (`ChannelMesh.strip`), and the legs'
first spacing ramps from `dn1` to the strip's edge interval, or keeps `dn1` where that interval is at or below it inside
G8''s slack (section 6.3). Checked after the build: the strip's edges equal the legs bitwise (P1'), the foil plus base
face sum <= 1e-12 (P2'), both blocks strictly convex with corner sines >= 0.2 (P3, P4), column 1 on the base and column
NIW+1 on the outlet plane (P5).
"""
function generate_cmesh(af::AirfoilPts; n_foil::Int=256, n_wake::Int=64, n_normal::Int=96,
                        dn1::Float64=4.0e-6, growth_max::Float64=1.25, R_ff::Float64=30.0,
                        L_wake::Float64=25.0, cluster_le::Float64=0.35, cluster_te::Float64=0.30,
                        smooth_iters::Int=60, bl_freeze::Int=12,
                        channel::Union{Nothing,ChannelSpec}=nothing,
                        smooth_block::Int=40, smooth_max_extra::Int=4000, le_ds::Float64=0.0,
                        n_base::Int=0, te_ds::Float64=0.0, wake_near_ds::Float64=0.0, wake_near_len::Float64=0.0,
                        wake_near_growth::Float64=1.05, te_topology::Symbol=:aplus)
    # Trailing-edge options, entry guards (DECISION.md 4.2, G1-G5; the growth check and G15 were added after the
    # verification, docs/_records/2026-10-02_blunt_te/stage1/verify_range/README.md F6 and F9); every one passes at the
    # defaults.
    n_base >= 0 || throw(ArgumentError("n_base must be >= 0 (got $n_base)"))
    isfinite(te_ds) && 0 <= te_ds < 0.1 || throw(ArgumentError("te_ds must lie in [0, 0.1) (got $te_ds)"))
    isfinite(wake_near_ds) && wake_near_ds >= 0 ||
        throw(ArgumentError("wake_near_ds must be finite and >= 0 (got $wake_near_ds)"))
    wake_near_ds == 0 && wake_near_len != 0 && throw(ArgumentError("wake_near_len needs wake_near_ds > 0"))
    isfinite(wake_near_growth) && wake_near_growth >= 1 ||     # checked even with the near wake off (F6)
        throw(ArgumentError("wake_near_growth must be finite and >= 1 (got $wake_near_growth)"))
    n_base > 0 && isnothing(channel) &&
        throw(ArgumentError("n_base > 0 (blunt trailing edge) is implemented on the channel branch only"))
    wake_near_ds > 0 && isnothing(channel) &&
        throw(ArgumentError("wake_near_ds > 0 (near wake) is implemented on the channel branch only"))
    te_ds > 0 && isnothing(channel) &&                         # (the integration task's G4c: unchecked in free air)
        throw(ArgumentError("te_ds > 0 (trailing-edge spacing) is implemented on the channel branch only"))
    # G-S1 (strip topology, 5 Oct 2026, plan sections 6.1 and 6.4): the knob's two values, the channel, and n_base >= 2 (the
    # strip is 2 n_base cells across; a leg's three ghost rows must not reach the other leg, 2 n_base >= 3).
    te_topology in (:aplus, :strip) || throw(ArgumentError("te_topology must be :aplus or :strip (got $te_topology)"))
    strip_mode = te_topology === :strip
    strip_mode && isnothing(channel) &&
        throw(ArgumentError("te_topology=:strip (the strip trailing-edge topology) is implemented on the channel branch only"))
    strip_mode && n_base < 2 && throw(ArgumentError("te_topology=:strip needs n_base >= 2 (the strip is 2 n_base cells " *
        "across, and a leg's three ghost rows must not reach the other leg: 2 n_base >= 3); got n_base=$n_base"))
    # (the A-plus construction's sweep floor; the strip's C block is the sharp construction and does not need it)
    n_base > 0 && !strip_mode && smooth_iters < _TE_MIN_SWEEPS && throw(ArgumentError("n_base > 0 needs smooth_iters >= 80 (the blunt " *
        "construction was measured at 80-400 sweeps; got $smooth_iters)"))
    # G15 (F9): the sharp control's trailing-edge interval is at least dn1 (1e-9 relative, as G7 and G8); with n_base > 0,
    # G7 and G8 together already hold te_ds >= the base interval >= dn1, and keep their own named causes.
    n_base == 0 && 0 < te_ds < dn1*(1 - 1e-9) && throw(ArgumentError("te_ds=$te_ds is below dn1=$dn1: the trailing-edge " *
        "interval must be at least the first wall spacing (plan section 3; a blunt edge meets it through G7 and G8). " *
        "Measured at dn1 1e-4: te_ds 1e-6 to 3e-5 failed late, at the convexity check or the 0.2 floor " *
        "(docs/_records/2026-10-02_blunt_te/stage1/verify_range/README.md F9)"))
    NIF_s, NIW, NJ = n_foil, n_wake, n_normal                  # NIF_s: surface cells
    NIF = strip_mode ? NIF_s : NIF_s + 2n_base                 # foil-range cells: the surface and two base halves (strip: no fold)
    NI = 2NIW + NIF
    s_all = _arclen(af.x, af.y)
    ile = argmin(af.x)                                         # LE index in the ordered wrap
    s_le, s_tot = s_all[ile], s_all[end]

    # -- surface resample: NIF_s cells split across lower/upper in proportion to arc length -------------
    b = clamp(0.5*(cluster_le + cluster_te)*2, 0.0, 0.95)      # cosine blend weight from the clustering knobs
    nlo = round(Int, NIF_s * s_le/s_tot); nlo = clamp(nlo, 8, NIF_s-8); nup = NIF_s - nlo
    surf_s = Vector{Tuple{Float64,Float64}}(undef, NIF_s+1)
    if te_ds > 0
        # With a base, G6-G8 first, on the gap this resample will have, so a refusal of the base is not reported as the
        # te_ds solve's refusal G11 (F8); they run again on te_gap_base below, the same value.
        n_base > 0 && _te_base_guards(_te_ds_corner_gap(af, s_all, s_le, s_tot), n_base, te_ds, dn1; topology=te_topology)
        # Trailing-edge spacing (2 Oct 2026): each side is the two-sided Vinokur stretching whose trailing-edge interval is
        # te_ds, met exactly as the chord between the resampled nodes (_vinokur_te_chord), and whose leading-edge interval
        # is le_ds or, without it, the default distribution's own leading-edge interval.
        le_lo = le_ds > 0 ? le_ds/s_le : 1 - _cluster((nlo-1)/nlo, b)
        le_up = le_ds > 0 ? le_ds/(s_tot - s_le) : _cluster(1/nup, b)
        tl = _vinokur_te_chord(af, s_all, 0.0, s_le, nlo, te_ds, le_lo, :lower)
        tu = _vinokur_te_chord(af, s_all, s_le, s_tot - s_le, nup, te_ds, le_up, :upper)
        for k in 0:nlo; surf_s[k+1] = _cr_eval(af.x, af.y, s_all, tl[k+1]*s_le); end
        for k in 1:nup; surf_s[nlo+1+k] = _cr_eval(af.x, af.y, s_all, s_le + tu[k+1]*(s_tot-s_le)); end
    elseif le_ds > 0
        # Independent leading-edge spacing (speed-up plan W5, 30 Sep 2026): each side is a two-sided Vinokur stretching
        # whose LE interval is le_ds (arc length, chord units) and whose TE interval is the default distribution's, so the
        # trailing edge and the wake junction are those of the default mesh; the tangential resolution moves from mid-chord
        # to the leading edge, where the V2 ladder found the error.
        te_lo = _cluster(1/nlo, b); te_up = 1 - _cluster((nup-1)/nup, b)
        tl = _vinokur(nlo, te_lo, le_ds/s_le)                   # lower: TE (t = 0) -> LE (t = 1)
        tu = _vinokur(nup, le_ds/(s_tot - s_le), te_up)         # upper: LE -> TE
        for k in 0:nlo; surf_s[k+1] = _cr_eval(af.x, af.y, s_all, tl[k+1]*s_le); end
        for k in 1:nup; surf_s[nlo+1+k] = _cr_eval(af.x, af.y, s_all, s_le + tu[k+1]*(s_tot-s_le)); end
    else
    for k in 0:nlo                                             # TE-lower -> LE
        t = _cluster(k/nlo, b)
        surf_s[k+1] = _cr_eval(af.x, af.y, s_all, t*s_le)
    end
    for k in 1:nup                                             # LE -> TE-upper
        t = _cluster(k/nup, b)
        surf_s[nlo+1+k] = _cr_eval(af.x, af.y, s_all, s_le + t*(s_tot-s_le))
    end
    end
    # Trailing-edge base (n_base > 0): the corner-to-corner gap of the resample, G6-G8, and the foil node list of the
    # midpoint split. With n_base = 0 the foil node list IS the resample (the same array).
    te_gap_base = hypot(surf_s[1][1]-surf_s[end][1], surf_s[1][2]-surf_s[end][2])
    n_base > 0 && _te_base_guards(te_gap_base, n_base, te_ds, dn1; topology=te_topology)
    # (strip topology: no fold, the foil node list is the open resample, surf[1] the lower corner CL and surf[end] the
    # upper corner CU; the trailing-edge reference below is then the base midpoint, the outer wake targets' reference)
    surf = n_base > 0 && !strip_mode ? _blunt_foil_nodes(surf_s, n_base) : surf_s
    xte = 0.5*(surf[1][1] + surf[end][1]); yte = 0.5*(surf[1][2] + surf[end][2])   # (sharp TE: identical)

    if isnothing(channel)
    # G10 (2 Oct 2026): this branch's wake pair is not coincident for an open edge (the upper wake loop overwrites only
    # its own junction node); a blunt section is meshed on the channel branch with n_base > 0.
    te_gap_base <= 1e-8 || throw(ArgumentError("free-air C-grid requires a sharp trailing edge (resampled TE gap=" *
        "$te_gap_base); a blunt edge is meshed on the channel branch with n_base > 0"))
    # -- wake x-distribution: first spacing = local TE surface spacing, geometric to L_wake -------------
    dste = max(hypot(surf[2][1]-surf[1][1], surf[2][2]-surf[1][2]), 1e-6)
    rw = _solve_growth2(dste, NIW, L_wake)
    xw = zeros(NIW+1); xw[1] = xte
    for k in 1:NIW; xw[k+1] = xw[k] + dste*rw^(k-1); end
    xw[end] = xte + L_wake                                     # exact endpoint
    ywk(x) = yte * max(0.0, 1.0 - (x - xte)/L_wake)            # wake line: TE y decaying linearly to 0

    # -- inner boundary + unit normals (n̂ = rotate(tangent, +90°): outward for the clockwise wrap) ------
    ib  = Vector{Tuple{Float64,Float64}}(undef, NI+1)
    nb  = Vector{Tuple{Float64,Float64}}(undef, NI+1)
    for k in 1:NIW+1                                           # lower wake: outlet -> TE (x decreasing)
        x = xw[NIW+2-k]; ib[k] = (x, ywk(x)); nb[k] = (0.0, -1.0)
    end
    for k in 1:NIF+1
        ib[NIW+k] = surf[k]
        km, kp = max(k-1,1), min(k+1,NIF+1)
        tx = surf[kp][1]-surf[km][1]; ty = surf[kp][2]-surf[km][2]; tn = hypot(tx,ty)
        nb[NIW+k] = (-ty/tn, tx/tn)
    end
    nb[NIW+1]      = (0.0, -1.0)                               # TE junction nodes take the wake normal
    nb[NIW+NIF+1]  = (0.0,  1.0)
    for k in 1:NIW+1                                           # upper wake: TE -> outlet
        x = xw[k]; ib[NIW+NIF+k] = (x, ywk(x)); nb[NIW+NIF+k] = (0.0, 1.0)
    end

    # -- outer C boundary --------------------------------------------------------------------------------
    # The outer distribution is DECOUPLED from the inner: wake outer points use their own one-sided
    # stretch whose first spacing MATCHES the semicircle arc spacing (pi*R/NIF) at the junction —
    # otherwise adjacent rays at the wake<->semicircle junction point to outer locations 1000x apart
    # (inner wake is TE-clustered ~1e-4c vs semicircle ~0.4c) and the blended mid-field folds
    # (measured: 819 folds at i=junction±30, j>=63 — outer half only — before this fix).
    ob = Vector{Tuple{Float64,Float64}}(undef, NI+1)
    ds_semi = pi*R_ff/NIF                                      # semicircle spacing at the junction
    rwo = _solve_growth2(min(ds_semi, L_wake/NIW), NIW, L_wake)
    xwo = zeros(NIW+1); xwo[1] = xte
    for k in 1:NIW; xwo[k+1] = xwo[k] + min(ds_semi, L_wake/NIW)*rwo^(k-1); end
    xwo .*= (xte + L_wake - xte)/max(xwo[end]-xte, 1e-30)      # normalize span exactly to the outlet
    xwo .+= xte .- xwo[1]
    xwo[end] = xte + L_wake
    for k in 1:NIW+1; ob[k] = (xwo[NIW+2-k], -R_ff); end       # lower leg outer: outlet -> TE junction
    for k in 1:NIF+1                                           # semicircle centered at the TE
        sn = (k-1)/NIF                                         # 0 at TE-lower, 1 at TE-upper
        th = -pi/2 - pi*sn
        ob[NIW+k] = (xte + R_ff*cos(th), R_ff*sin(th))
    end
    for k in 1:NIW+1; ob[NIW+NIF+k] = (xwo[k], R_ff); end      # upper leg outer: TE junction -> outlet
    else
        # -- Leroux-style rectangular channel -----------------------------------------------------------
        ch = channel::ChannelSpec
        esx, esy = ch.streamwise
        enx, eny = -esy, esx
        px, py = ch.ref
        to_channel(x, y) = (ch.xi_ref + (x-px)*esx + (y-py)*esy,
                            (x-px)*enx + (y-py)*eny)
        to_physical(xi, eta) = (px + (xi-ch.xi_ref)*esx + eta*enx,
                                py + (xi-ch.xi_ref)*esy + eta*eny)

        pte = (xte, yte)
        # (G9: with n_base > 0 the foil node list starts and ends at the base midpoint m, so this gap is exactly 0 and the
        # refusal applies only to an open edge meshed with n_base = 0, as before. Strip topology: the open gap is the
        # strip's width, required by G6, so G9 is skipped.)
        te_gap = hypot(surf[1][1]-surf[end][1], surf[1][2]-surf[end][2])
        strip_mode || te_gap <= 1e-8 || throw(ArgumentError(
            "channel C-grid requires a sharp trailing edge (resampled TE gap=$te_gap)"))
        xi_te, _ = to_channel(pte...)
        wake_length = ch.xi_out - xi_te
        upstream_length = xi_te - ch.xi_in
        wake_length > 0 || throw(ArgumentError(
            "channel outlet xi=$(ch.xi_out) must be downstream of TE xi=$xi_te"))
        upstream_length > 0 || throw(ArgumentError(
            "channel inlet xi=$(ch.xi_in) must be upstream of TE xi=$xi_te"))
        h = 0.5*ch.height
        foil_clearance = minimum(h - abs(to_channel(p...)[2]) for p in surf)
        foil_clearance > 0 || throw(ArgumentError(
            "foil crosses channel wall (minimum normal clearance=$foil_clearance)"))

        # Build one wake vector and assign its tuples to both sides. This includes both TE junctions,
        # so node i and node NI+2-i on the wake cut are bitwise coincident. (_wake_distances with wake_near_ds = 0 is the
        # single geometric distribution this branch has always built, statement for statement; wake_near_ds > 0 is the
        # two-segment near wake, 2 Oct 2026.)
        dste = max(hypot(surf[2][1]-surf[1][1], surf[2][2]-surf[1][2]), 1e-6)
        if strip_mode
            # Strip topology (plan section 6.2, items 4-5): one station vector per leg, from its own corner along the
            # channel direction to the outlet plane; dste is the corner interval (te_ds). The corners' streamwise
            # coordinates differ by the base's streamwise extent (h sin(alpha) for a base normal to the chord), so the two
            # wake lengths differ and _wake_distances rescales each leg to its own length by its existing rule. The
            # longer leg is built first: a G12 refusal then names the larger wake count (its need grows with the length).
            xi_cl = to_channel(surf[1]...)[1]; xi_cu = to_channel(surf[end]...)[1]
            wl_lo = ch.xi_out - xi_cl; wl_up = ch.xi_out - xi_cu
            leg_stations(w) = _wake_distances(dste, NIW, w; near_ds=wake_near_ds, near_len=wake_near_len,
                                              near_growth=wake_near_growth, growth_max=growth_max)
            if wl_lo >= wl_up
                dL = leg_stations(wl_lo); dU = leg_stations(wl_up)
            else
                dU = leg_stations(wl_up); dL = leg_stations(wl_lo)
            end
            if wake_near_ds > 0                                # G14 on both legs
                _te_near_wake_cap_guard(dL, wake_near_ds, wake_near_len, n_base, te_ds, te_gap_base; strip=true)
                _te_near_wake_cap_guard(dU, wake_near_ds, wake_near_len, n_base, te_ds, te_gap_base; strip=true)
            end
        else
        dw = _wake_distances(dste, NIW, wake_length; near_ds=wake_near_ds, near_len=wake_near_len,
                             near_growth=wake_near_growth, growth_max=growth_max)
        wake_near_ds > 0 && _te_near_wake_cap_guard(dw, wake_near_ds, wake_near_len, n_base, te_ds, te_gap_base)   # G14
        wake = [(xte + d*esx, yte + d*esy) for d in dw]
        end

        ib = Vector{Tuple{Float64,Float64}}(undef, NI+1)
        nb = Vector{Tuple{Float64,Float64}}(undef, NI+1)
        # ★ TE-JUNCTION NORMAL REPAIR (2026-09-24). The wake-leg extrusion normals were the constant
        # channel normal, and the two TE junction nodes were FORCED to it, while the adjacent foil
        # nodes carry local surface normals: at this cambered section and incidence the adjacent-column
        # extrusion kink was 16.9 deg AT THE WALL on the upper junction (measured,
        # S0903_RUNS/te_junction_mesh_probe_20260924.jl), inside the frozen BL block where Winslow
        # smoothing cannot repair it. That kink is the documented root of the refinement fold, the
        # n_foil=1024 TE squeeze, the two invalid channel meshes and the stiff-liquid wedge cell
        # (docs/2026-09-23_STIFF_LIQUID_WEDGE_FINDINGS.md). Repair: the wake cut GEOMETRY is untouched
        # (ib below is the previous construction bit for bit, preserving the i <-> NI+2-i node
        # coincidence on the cut); only the extrusion DIRECTION field changes. Each junction node keeps
        # its one-sided TE surface normal (exactly what the foil loop computes at k=1 / k=NIF+1), and
        # the wake-node normals rotate smoothly (C1 cosine in wake arclength) from that into the exact
        # channel normal by a fixed quarter of the wake length -- identically +-en beyond it, so the
        # outlet columns i=1 and i=NI+1 keep their exact channel-normal construction and the
        # end-geometry checks see what they always saw. Free-air meshes are untouched (this is the
        # channel branch; the free-air branch above keeps its original path).
        tlo = (surf[2][1]-surf[1][1], surf[2][2]-surf[1][2]); tlon = hypot(tlo...)
        n_lo = (-tlo[2]/tlon, tlo[1]/tlon)                     # lower-surface TE normal (points down)
        tup = (surf[NIF+1][1]-surf[NIF][1], surf[NIF+1][2]-surf[NIF][2]); tupn = hypot(tup...)
        n_up = (-tup[2]/tupn, tup[1]/tupn)                     # upper-surface TE normal (points up)
        if n_base > 0 && !strip_mode                           # blunt edge (A-plus): one direction per base half, from which the wake blends
            n_lo, n_up = _te_base_dirs(surf_s, surf, n_base, (esx, esy))
        end                                                    # (strip: the one-sided surface normals at the corners, as a sharp edge)
        wake_blend_span = 0.25*wake_length
        # The production wakes are ~5.8 chords, so the rotation spreads over >= 1.4 chords (14x the 0.10-chord
        # anchor depth). A short-wake ChannelSpec compresses it toward the anchor scale, where adjacent-column
        # rays can cross inside the frozen block again -- warn rather than silently rely on the self-heal.
        wake_blend_span < 0.2 && @warn "channel cmesh: wake normal blend span $(wake_blend_span) c is short " *
            "against the 0.10 c anchor depth; junction rays may converge inside the frozen block (short wake?)"
        blendw(d) = d >= wake_blend_span ? 0.0 : 0.5*(1.0 + cos(pi*d/wake_blend_span))
        wake_normal(d, nte, sx, sy) = begin
            w = blendw(d)
            w == 0.0 && return (sx, sy)                        # exact channel normal beyond the blend
            w == 1.0 && return nte                             # exact TE surface normal at the junction node
            bx = w*nte[1] + (1.0-w)*sx; by = w*nte[2] + (1.0-w)*sy; bn = hypot(bx, by)
            (bx/bn, by/bn)
        end
        if strip_mode
            for k in 1:NIW+1                                   # lower leg: outlet -> lower corner CL, its own stations
                d = dL[NIW+2-k]
                ib[k] = (surf[1][1] + d*esx, surf[1][2] + d*esy); nb[k] = wake_normal(d, n_lo, -enx, -eny)
            end
        else
        for k in 1:NIW+1
            ib[k] = wake[NIW+2-k]; nb[k] = wake_normal(dw[NIW+2-k], n_lo, -enx, -eny)
        end
        end
        for k in 1:NIF+1
            ib[NIW+k] = surf[k]
            km, kp = max(k-1,1), min(k+1,NIF+1)
            tx = surf[kp][1]-surf[km][1]; ty = surf[kp][2]-surf[km][2]; tn = hypot(tx,ty)
            nb[NIW+k] = (-ty/tn, tx/tn)
        end
        n_base > 0 && !strip_mode && _te_wall_normals!(nb, surf, surf_s, NIW, n_base, n_lo, n_up)   # blunt edge: base, junction, corner fan
        if strip_mode
            for k in 1:NIW+1                                   # upper leg: upper corner CU -> outlet, its own stations
                d = dU[k]
                ib[NIW+NIF+k] = (surf[end][1] + d*esx, surf[end][2] + d*esy); nb[NIW+NIF+k] = wake_normal(d, n_up, enx, eny)
            end
            ib[NIW+1] = surf[1]; ib[NIW+NIF+1] = surf[end]     # the junction nodes are the corners themselves (bitwise)
        else
        for k in 1:NIW+1
            ib[NIW+NIF+k] = wake[k]; nb[NIW+NIF+k] = wake_normal(dw[k], n_up, enx, eny)
        end
        ib[NIW+1] = wake[1]; ib[NIW+NIF+1] = wake[1]
        end

        if n_base > 0 && !strip_mode                           # blunt edge (A-plus): the base columns' outer targets downstream
            ob, outer_face_kind = _te_outer_downstream(NI, NIW, NIF_s, n_base, nlo, nup, upstream_length, h, wake_length,
                                                       xi_te, ch.xi_in, to_physical)
        else
        # The NIF central outer faces traverse TE-lower -> lower inlet corner -> inlet centre
        # (the LE column) -> upper inlet corner -> TE-upper. Segment counts are proportional to
        # physical length and every corner is an explicit node, never a diagonal boundary face.
        nlw = clamp(round(Int, nlo*upstream_length/(upstream_length+h)), 1, nlo-1)
        nli = nlo - nlw
        nui = clamp(round(Int, nup*h/(h+upstream_length)), 1, nup-1)
        nuw = nup - nui

        # Decouple outer wake-wall spacing from the tiny inner TE spacing, matching the mean adjacent
        # wall spacing at the junction (the same anti-braiding principle as the legacy far field).
        ds_join = 0.5*(upstream_length/nlw + upstream_length/nuw)
        rwo = _solve_growth2(ds_join, NIW, wake_length)
        swo = zeros(NIW+1)
        for k in 1:NIW; swo[k+1] = swo[k] + ds_join*rwo^(k-1); end
        swo .*= wake_length/max(swo[end], 1e-30)
        swo[end] = wake_length

        ob = Vector{Tuple{Float64,Float64}}(undef, NI+1)
        for k in 1:NIW+1
            ob[k] = to_physical(xi_te+swo[NIW+2-k], -h)
        end
        for q in 0:nlo
            xi, eta = if q <= nlw
                (xi_te - q*upstream_length/nlw, -h)
            else
                (ch.xi_in, -h + (q-nlw)*h/nli)
            end
            ob[NIW+q+1] = to_physical(xi, eta)
        end
        for q in 1:nup
            xi, eta = if q <= nui
                (ch.xi_in, q*h/nui)
            else
                (ch.xi_in + (q-nui)*upstream_length/nuw, h)
            end
            ob[NIW+nlo+q+1] = to_physical(xi, eta)
        end
        for k in 1:NIW+1
            ob[NIW+NIF+k] = to_physical(xi_te+swo[k], h)
        end

        outer_face_kind = Vector{UInt8}(undef, NI)
        outer_face_kind[1:NIW+nlw] .= OUTER_LOWER_WALL
        outer_face_kind[NIW+nlw+1:NIW+nlo+nui] .= OUTER_INLET
        outer_face_kind[NIW+nlo+nui+1:NI] .= OUTER_UPPER_WALL
        end
    end

    # -- anchored two-segment rays with per-column geometric wall-normal distribution --------------------
    # Each column = wall-orthogonal segment of length bd (the BL anchor) + a STRAIGHT segment to the
    # outer target. Straight segments between two i-monotone curves cannot cross — this replaced a
    # smoothstep normal->target blend whose S-shaped rays (normal tilt then target veer, up to ~90°
    # apart on the aft surfaces) braided through their neighbors (measured 770-819 folds, outer half).
    bd = 0.10                                                  # anchor distance (chords): orthogonal BL depth
    # The per-column first wall-normal spacing (strip topology, plan section 6.3): the legs' ramp from dn1 at the corners
    # to the strip's edge interval gap/(2 n_base). On every other path fill(dn1, NI+1): the sites below that read dn1_i[i]
    # (this loop's schedule, the frozen-block check and repair, the reparameterisation's schedule) then take the same
    # Float64 through the same operations, bitwise (_column_schedule's standard branch is the old accumulation).
    dn1_i = strip_mode ? _leg_first_spacing(dn1, growth_max, te_gap_base/(2n_base), NIW, NIF) : fill(dn1, NI+1)
    X = zeros(NI+1, NJ+1); Y = zeros(NI+1, NJ+1)
    growth = zeros(NI+1)
    for i in 1:NI+1
        xi, yi = ib[i]; xo, yo = ob[i]
        ax = xi + bd*nb[i][1]; ay = yi + bd*nb[i][2]           # BL anchor
        L2 = hypot(xo-ax, yo-ay)
        Lpath = bd + L2
        r = _solve_growth2(dn1, NJ, Lpath); growth[i] = r
        sched = _column_schedule(dn1_i[i], dn1, r, NJ, Lpath)
        X[i,1] = xi; Y[i,1] = yi
        for j in 1:NJ
            acc = sched[j+1]
            if acc <= bd
                X[i,j+1] = xi + acc*nb[i][1]; Y[i,j+1] = yi + acc*nb[i][2]
            else
                f = min((acc - bd)/L2, 1.0)
                X[i,j+1] = ax + f*(xo-ax); Y[i,j+1] = ay + f*(yo-ay)
            end
        end
        X[i,NJ+1] = xo; Y[i,NJ+1] = yo                         # exact outer landing
    end
    # Blunt edge (n_base > 0): the columns within the trailing-edge window are rebuilt as curved columns; every other
    # column, and every column of a sharp mesh, is the straight anchored ray above (columns are independent).
    # (Strip topology: no curved columns, no anchor guard, no frozen placement, no screened sweep: tecols === nothing.)
    tecols = n_base > 0 && !strip_mode ? _te_curved_columns!(X, Y, growth, ib, nb, ob, NI, NJ, NIW, NIF, n_base, nlo, dn1, bd,
                                                             te_gap_base) : nothing
    tecols === nothing || _te_anchor_guard(tecols.curves, channel::ChannelSpec, tecols.bd_te, NIW)   # G13

    # A very distant rectangular wall makes the per-column geometric ratios differ most strongly at
    # the TE/wake junction.  If that variation overturns a cell inside the deliberately frozen BL
    # block, Winslow smoothing cannot touch it.  Repair the spacing, not the physical boundaries:
    # use one shared r=1 near-wall schedule through the frozen block, then let the ordinary TFI and
    # Winslow region absorb the transition.  This first repair is geometry-triggered and channel-only;
    # all legacy free-air meshes remain on their original byte-for-byte path.
    frozen_spacing_repair = false
    if !isnothing(channel) && bl_freeze > 0
        nbl = min(bl_freeze, NJ)
        cell_area(i, j) = begin
            dx1 = X[i+1,j+1]-X[i,j]; dy1 = Y[i+1,j+1]-Y[i,j]
            dx2 = X[i,j+1]-X[i+1,j]; dy2 = Y[i,j+1]-Y[i+1,j]
            0.5*(dx1*dy2 - dy1*dx2)
        end
        frozen_fold = any(cell_area(i,j) <= 0 for j in 1:nbl for i in 1:NI)
        if frozen_fold
            nbl*maximum(dn1_i) <= bd || throw(ArgumentError(  # (the largest per-column first spacing: dn1 unless the strip's ramp)
                "channel frozen-BL untangling needs bl_freeze*dn1 <= $bd " *
                "(got $(nbl*maximum(dn1_i))); reduce dn1 or bl_freeze"))
            for i in 1:NI+1, j in 1:nbl
                d = j*dn1_i[i]
                X[i,j+1] = ib[i][1] + d*nb[i][1]
                Y[i,j+1] = ib[i][2] + d*nb[i][2]
            end
            tecols === nothing || _te_frozen_placement!(X, Y, tecols.curves, nbl, dn1)   # blunt edge: along the curves
            any(cell_area(i,j) <= 0 for j in 1:nbl for i in 1:NI) &&
                error("channel shared-spacing repair did not untangle the frozen BL block")
            frozen_spacing_repair = true
            @warn "channel cmesh: repaired a frozen-block fold with shared r=1 spacing over the first $nbl wall-normal cells"
        end
    end
    ngrow = count(>(growth_max), growth)
    ngrow > 0 && @warn "cmesh: $ngrow/$(NI+1) columns exceed growth_max=$growth_max (max used $(round(maximum(growth),digits=3))) — increase mesh.n_normal or relax the target"

    # -- Winslow elliptic smoothing (interior, j > bl_freeze; boundaries + BL block frozen) --------------
    # (blunt edge, n_base > 0: _te_winslow_sweep, this update plus the frozen core and the screened ramp near the base)
    winslow_sweep! = tecols === nothing ? (() -> begin
        for j in bl_freeze+2:NJ, i in 2:NI
            xxi = 0.5*(X[i+1,j]-X[i-1,j]); yxi = 0.5*(Y[i+1,j]-Y[i-1,j])
            xet = 0.5*(X[i,j+1]-X[i,j-1]); yet = 0.5*(Y[i,j+1]-Y[i,j-1])
            a = xet^2 + yet^2; g = xxi^2 + yxi^2; bb = xxi*xet + yxi*yet
            den = 2*(a + g); den < 1e-14 && continue
            xnew = (a*(X[i+1,j]+X[i-1,j]) + g*(X[i,j+1]+X[i,j-1]) -
                    0.5*bb*(X[i+1,j+1]-X[i-1,j+1]-X[i+1,j-1]+X[i-1,j-1])) / den
            ynew = (a*(Y[i+1,j]+Y[i-1,j]) + g*(Y[i,j+1]+Y[i,j-1]) -
                    0.5*bb*(Y[i+1,j+1]-Y[i-1,j+1]-Y[i+1,j-1]+Y[i-1,j-1])) / den
            X[i,j] = 0.7*X[i,j] + 0.3*xnew; Y[i,j] = 0.7*Y[i,j] + 0.3*ynew
        end
        nothing
    end) : _te_winslow_sweep(X, Y, NI, NJ, bl_freeze, tecols.femask, tecols.wsoft)
    for _ in 1:smooth_iters
        winslow_sweep!()
    end

    # -- Defect-triggered continuation (rebuild, 22 Sep 2026; junction root cause REPAIRED 24 Sep) ----------
    # UNTIL 24 Sep 2026 the ray construction forced the channel wake normal on the two trailing-edge junction
    # nodes while their neighbours kept the true surface normal: a ~17-degree kink, whose crossing rays this
    # block was built to untangle (measured then: n_foil 448 needed ~120 extra sweeps, 768 ~240, 1024 ~320).
    # The TE-JUNCTION NORMAL REPAIR above removed that forcing (junction kink 16.9 -> 0.008 deg at the wall,
    # docs/2026-09-24_TE_JUNCTION_REPAIR.md), so the junction no longer generates the load this block absorbs.
    # The continuation STAYS as the generator's general convexity backstop: it keeps smoothing, in blocks,
    # until every cell is strictly convex, and errors only if a cap is reached. A mesh already valid after
    # smooth_iters sweeps takes no extra sweep and is bitwise what it was.
    let (nbad0, worst0, where0) = _convexity_defects(X, Y, NI, 1:NJ), extra = 0
        nbad, worst, where = nbad0, worst0, where0
        if nbad > 0 && where[2] <= bl_freeze + 1
            error("cmesh: $nbad non-convex cells inside the frozen boundary-layer block (worst corner Jacobian " *
                  "$(worst) at (i,j)=$(where)); Winslow smoothing cannot reach j <= bl_freeze+1 -- reduce dn1 or bl_freeze")
        end
        while nbad > 0 && extra < smooth_max_extra
            for _ in 1:smooth_block; winslow_sweep!(); end
            extra += smooth_block
            nbad, worst, where = _convexity_defects(X, Y, NI, 1:NJ)
        end
        if nbad0 > 0
            @warn "cmesh: $smooth_iters Winslow sweeps left $nbad0 non-convex cells (worst corner Jacobian $(worst0) at (i,j)=$(where0)); " *
                  "$extra extra sweeps " * (nbad == 0 ? "untangled them" : "did NOT: $nbad remain, worst $(worst) at $(where)")
        end
        nbad > 0 && error("cmesh: $nbad cells remain non-convex after $(smooth_iters + extra) Winslow sweeps (worst corner " *
                          "Jacobian $(worst) at (i,j)=$(where)); the junction normal taper is already the default (24 Sep 2026), so " *
                          "look elsewhere first -- raise smooth_max_extra or n_normal, widen wake_blend_span, or reduce dn1")
    end

    # Nominal `growth` is not an admission metric: the frozen/smoothed interface measured 2.01x in
    # the H/c=1.28 mesh even though its largest construction ratio was only 1.079, and the initially
    # untangled H/c=50 mesh measured 4.80x.  Measure the final coordinates independently and
    # reparameterize every offending channel column by its own arc length.  Column curves and both
    # physical boundaries stay fixed while the dn1-to-outer schedule is restored.  A frozen-block
    # repair also forces this pass, even if subsequent smoothing happens to reduce the measured ratio.
    grading_tol = 512eps(Float64)*max(1.0,abs(growth_max))
    realised_before = isnothing(channel) ? nothing : _radial_spacing_stats(X, Y, NI, NJ)
    needs_reparameterization = frozen_spacing_repair ||
        (!isnothing(realised_before) &&
         realised_before.ratio_symmetric_max > growth_max + grading_tol)
    if needs_reparameterization
        if !frozen_spacing_repair
            @warn "channel cmesh: realised adjacent spacing ratio $(realised_before.ratio_symmetric_max) exceeds growth_max=$growth_max; reparameterizing columns by arc length"
        end
        Xcurve = copy(X); Ycurve = copy(Y)
        sold = zeros(NJ+1); snew = similar(sold)
        # ★ CHANNEL REPARAMETERISATION SMOOTHING (2026-08-05, TOPO_CHREPAR; 0 = bitwise legacy).
        # DEFECT: each column was reparameterised INDEPENDENTLY -- its own arc length, its own solved
        # growth ratio r. Neighbouring columns therefore place their j-th node at different relative
        # arc positions, and where the column curves are not parallel that shear can INVERT a cell.
        # MEASURED (tools/channel_mesh_resolution_study.jl): at n_normal=128 the narrow channels
        # (H/c=2.5 and 1.28) FAIL TO BUILD -- "arc-length reparameterization reintroduced a folded
        # cell" -- so the confined configuration CANNOT BE GRID-REFINED, and no grid-convergence study
        # of it is possible. Doubling NJ makes snew far more uniform (r -> 1) while sold stays
        # strongly graded from the smoothing, so nodes travel further and the column-to-column
        # mismatch grows.
        # FIX: solve every column's r FIRST, smooth r across i, then build. dn1 and each column's
        # total arc length are still honoured exactly; only the GRADING between neighbours is made
        # consistent, which is precisely the shear that folds the cell.
        chrepar = parse(Int, get(ENV, "TOPO_CHREPAR", "0"))
        rs = zeros(NI+1); send = zeros(NI+1)
        for i in 1:NI+1
            acc = 0.0
            for j in 1:NJ
                acc += hypot(Xcurve[i,j+1]-Xcurve[i,j], Ycurve[i,j+1]-Ycurve[i,j])
            end
            send[i] = acc; rs[i] = _solve_growth2(dn1, NJ, acc)
        end
        if chrepar >= 1
            for _ in 1:chrepar                       # Jacobi passes; ends held (physical boundaries)
                rprev = copy(rs)
                for i in 2:NI
                    rs[i] = 0.25*rprev[i-1] + 0.5*rprev[i] + 0.25*rprev[i+1]
                end
            end
        end
        for i in 1:NI+1
            sold[1] = 0.0
            for j in 1:NJ
                sold[j+1] = sold[j] + hypot(Xcurve[i,j+1]-Xcurve[i,j],
                                             Ycurve[i,j+1]-Ycurve[i,j])
            end
            r = rs[i]; growth[i] = r
            # (the column's schedule along its own arc length: the geometric one, accumulated as before, or the strip
            # leg's ramp schedule when its first spacing dn1_i[i] exceeds dn1)
            snew .= _column_schedule(dn1_i[i], dn1, r, NJ, sold[end])
            # renormalise so the outer point lands exactly on the physical boundary even when r was
            # smoothed away from this column's own solved value (legacy path: r is exact, scale == 1)
            sc = snew[end] > 0 ? sold[end]/snew[end] : 1.0
            chrepar >= 1 && (for j in 2:NJ; snew[j] *= sc; end)
            snew[end] = sold[end]
            for j in 2:NJ
                k = clamp(searchsortedlast(sold, snew[j]), 1, NJ)
                f = (snew[j]-sold[k])/max(sold[k+1]-sold[k], 1e-30)
                X[i,j] = (1-f)*Xcurve[i,k] + f*Xcurve[i,k+1]
                Y[i,j] = (1-f)*Ycurve[i,k] + f*Ycurve[i,k+1]
            end
        end
        let (nbad, worst, where) = _convexity_defects(X, Y, NI, 1:NJ)
            # (rebuild, 22 Sep 2026) the old test here was signed area > 0 and its message blamed the
            # reparameterisation; the fold it reported was already present before this pass, in the cells the
            # ray construction crossed. The test is now the solver's (strict convexity) and the message says so.
            nbad > 0 && error("cmesh: $nbad cells are not strictly convex after the channel arc-length reparameterisation " *
                              "(worst corner Jacobian $(worst) at (i,j)=$(where)); the mesh was convex before this pass, so " *
                              "the column redistribution created the defect -- report this configuration")
        end
        realised = _radial_spacing_stats(X, Y, NI, NJ)
        realised.ratio_symmetric_max <= growth_max + grading_tol || error(
            "channel spacing repair exceeds requested growth_max=$growth_max: " *
            "realised adjacent ratio=$(realised.ratio_symmetric_max) at " *
            "$(realised.ratio_symmetric_location)")
    end

    # Trailing-edge options: the post-build checks of every channel mesh built with one (DECISION.md 4.4, P1-P4). Strip
    # topology: the strip block from the C block's final row j = 1, then P1'-P5 (plan section 6.4) on both blocks.
    local stripblock = nothing
    if strip_mode
        NS = 2n_base
        Xs, Ys, ts = _strip_block(X, Y, NIW, NIF, NS)
        _te_post_checks_strip(X, Y, Xs, Ys, NI, NJ, NIW, NIF, NS, channel::ChannelSpec)
        stripblock = StripBlock(NIW, NS, n_base, Xs, Ys, ts, dn1_i)
    else
        !isnothing(channel) && (n_base > 0 || te_ds > 0 || wake_near_ds > 0) && _te_post_checks(X, Y, NI, NJ, NIW, NIF)
    end

    if isnothing(channel)
        CMesh(NI, NJ, NIW, NIF, X, Y, dn1, growth)
    else
        ChannelMesh(NI, NJ, NIW, NIF, X, Y, dn1, growth,
                    channel::ChannelSpec, outer_face_kind, stripblock)
    end
end

# ---- quality report ------------------------------------------------------------------------------------
function mesh_quality(m::Union{CMesh,ChannelMesh}; reynolds::Float64=6.5e6)
    NI, NJ, X, Y = m.NI, m.NJ, m.X, m.Y
    nfold = 0; amin = Inf; armax = 0.0; orthmin = 90.0
    for j in 1:NJ, i in 1:NI
        ax = X[i+1,j]-X[i,j];   ay = Y[i+1,j]-Y[i,j]           # quad area via shoelace (cross of diagonals /2)
        dx1 = X[i+1,j+1]-X[i,j]; dy1 = Y[i+1,j+1]-Y[i,j]
        dx2 = X[i,j+1]-X[i+1,j]; dy2 = Y[i,j+1]-Y[i+1,j]
        area = 0.5*(dx1*dy2 - dy1*dx2)
        area <= 0 && (nfold += 1)
        amin = min(amin, area)
        if j == 1
            ds = hypot(ax, ay)
            dn = hypot(X[i,2]-X[i,1], Y[i,2]-Y[i,1])
            armax = max(armax, ds/max(dn,1e-30))
            # wall orthogonality: angle between the normal ray and the surface tangent
            rx = X[i,2]-X[i,1]; ry = Y[i,2]-Y[i,1]
            ct = abs(ax*rx + ay*ry) / max(hypot(ax,ay)*hypot(rx,ry), 1e-30)
            orthmin = min(orthmin, 90.0 - rad2deg(asin(clamp(ct,0,1))))
        end
    end
    cf = 0.026 * reynolds^(-1/7)                               # flat-plate estimate for the y+ conversion
    yplus = m.dn1 * reynolds * sqrt(cf/2)
    spacing = _radial_spacing_stats(X, Y, NI, NJ)
    # ★ WAKE-CUT COINCIDENCE (2026-08-07). The header contract at the top of this file states that
    # node (i,1) and node (NI+2-i,1) are the SAME PHYSICAL POINT for i in 1..NIW+1, and the solver
    # relies on it at ~8 sites (cmesh_solver_ka.jl:463/:606/:1719, cmesh_solver.jl:237/:869,
    # cmesh_span3d.jl:271/:946, cmesh_cav_tem.jl:918, cmesh_cav_tem3d.jl:213) to exchange ghosts
    # across the C-mesh branch cut.
    # ⛔ IT WAS CHECKED ONLY ON THE CHANNEL PATH (channel_quality). The FREE-AIR path -- which is what
    # every cavitation battery runs -- never verified it. Measured while scoping the blunt trailing
    # edge: a finite-TE foil silently drives this to 4.000e-3 (against 6.78e-21 sharp), ASYMMETRICALLY,
    # because the upper-wake loop overwrites its junction node while the lower one is never touched.
    # No error, no warning, on every free-air tool. Computed here so BOTH paths report and gate it.
    # Strip topology (5 Oct 2026): the legs are the strip's width apart, so the partner distance is not the invariant;
    # the edge coincidence P1' is (0 on every delivered strip mesh, and admit_outer_geometry requires 0 of this field), and
    # the strip's cells join the fold count and the smallest area.
    stripb = m isa ChannelMesh ? m.strip : nothing
    local wake_pair_err
    if stripb === nothing
        wake_pair_err = m.NIW >= 1 ?
            maximum(hypot(X[k,1]-X[NI+2-k,1], Y[k,1]-Y[NI+2-k,1]) for k in 1:m.NIW+1) : 0.0
    else
        wake_pair_err = _strip_edge_error(X, Y, stripb.X, stripb.Y, m.NIW, m.NIF, stripb.NJ)
        for j in 1:stripb.NJ, i in 1:stripb.NI
            dx1 = stripb.X[i+1,j+1]-stripb.X[i,j]; dy1 = stripb.Y[i+1,j+1]-stripb.Y[i,j]
            dx2 = stripb.X[i,j+1]-stripb.X[i+1,j]; dy2 = stripb.Y[i,j+1]-stripb.Y[i+1,j]
            area = 0.5*(dx1*dy2 - dy1*dx2)
            area <= 0 && (nfold += 1)
            amin = min(amin, area)
        end
    end
    (nfold=nfold, area_min=amin, ar_max=armax, orth_min_deg=orthmin, wake_pair_err=wake_pair_err,
     dn1=m.dn1, yplus_est=yplus, growth_max_used=maximum(m.growth),
     spacing_ratio_max=spacing.ratio_max,
     spacing_ratio_symmetric_max=spacing.ratio_symmetric_max,
     spacing_ratio_location=spacing.ratio_location,
     spacing_ratio_symmetric_location=spacing.ratio_symmetric_location)
end

"""Geometry-only admission report for a channel mesh (GCL is reported after `build_metrics`)."""
channel_quality(::CMesh) = (enabled=false,)
function channel_quality(m::ChannelMesh)
    NI, NJ, NIW, NIF = m.NI, m.NJ, m.NIW, m.NIF
    length(m.outer_face_kind) == NI || throw(ArgumentError(
        "ChannelMesh outer_face_kind has length $(length(m.outer_face_kind)); expected $NI"))
    ch = m.channel
    esx, esy = ch.streamwise
    enx, eny = -esy, esx
    px, py = ch.ref
    coord(x, y) = (ch.xi_ref + (x-px)*esx + (y-py)*esy,
                   (x-px)*enx + (y-py)*eny)
    h = 0.5*ch.height
    scale = max(1.0, ch.height, abs(ch.xi_in), abs(ch.xi_out))
    tol = 2048eps(Float64)*scale

    lower_eta_err = 0.0; upper_eta_err = 0.0; inlet_xi_err = 0.0
    wall_parallel_err = 0.0; inlet_parallel_err = 0.0
    monotonicity_violations = 0
    ds_min = Inf; ds_max = 0.0
    for i in 1:NI
        x1, y1 = m.X[i,NJ+1], m.Y[i,NJ+1]
        x2, y2 = m.X[i+1,NJ+1], m.Y[i+1,NJ+1]
        xi1, eta1 = coord(x1, y1); xi2, eta2 = coord(x2, y2)
        tx, ty = x2-x1, y2-y1
        ds = hypot(tx, ty); ds_min = min(ds_min, ds); ds_max = max(ds_max, ds)
        kind = m.outer_face_kind[i]
        if kind == OUTER_LOWER_WALL
            lower_eta_err = max(lower_eta_err, abs(eta1+h), abs(eta2+h))
            wall_parallel_err = max(wall_parallel_err, abs(tx*enx + ty*eny)/max(ds,1e-30))
            xi2-xi1 > tol && (monotonicity_violations += 1)
        elseif kind == OUTER_INLET
            inlet_xi_err = max(inlet_xi_err, abs(xi1-ch.xi_in), abs(xi2-ch.xi_in))
            inlet_parallel_err = max(inlet_parallel_err, abs(tx*esx + ty*esy)/max(ds,1e-30))
            eta2-eta1 < -tol && (monotonicity_violations += 1)
        elseif kind == OUTER_UPPER_WALL
            upper_eta_err = max(upper_eta_err, abs(eta1-h), abs(eta2-h))
            wall_parallel_err = max(wall_parallel_err, abs(tx*enx + ty*eny)/max(ds,1e-30))
            xi2-xi1 < -tol && (monotonicity_violations += 1)
        else
            throw(ArgumentError("ChannelMesh outer face $i has invalid kind $kind"))
        end
    end

    xis = Vector{Float64}(undef, NI+1); etas = similar(xis)
    for i in 1:NI+1
        xis[i], etas[i] = coord(m.X[i,NJ+1], m.Y[i,NJ+1])
    end
    lower_corner_err = minimum(hypot(xis[i]-ch.xi_in, etas[i]+h) for i in eachindex(xis))
    upper_corner_err = minimum(hypot(xis[i]-ch.xi_in, etas[i]-h) for i in eachindex(xis))
    lower_corner_count = count(i -> hypot(xis[i]-ch.xi_in, etas[i]+h) <= tol, eachindex(xis))
    upper_corner_count = count(i -> hypot(xis[i]-ch.xi_in, etas[i]-h) <= tol, eachindex(xis))

    # (strip topology: the edge-coincidence error P1' in place of the partner distance, as in mesh_quality)
    wake_pair_err = m.strip === nothing ? maximum(hypot(m.X[k,1]-m.X[NI+2-k,1],
                                                        m.Y[k,1]-m.Y[NI+2-k,1]) for k in 1:NIW+1) :
                    _strip_edge_error(m.X, m.Y, m.strip.X, m.strip.Y, NIW, NIF, m.strip.NJ)
    wake_alignment_err = 0.0
    for k in 1:NIW
        tx = m.X[k+1,1]-m.X[k,1]; ty = m.Y[k+1,1]-m.Y[k,1]
        wake_alignment_err = max(wake_alignment_err,
            abs(tx*esy-ty*esx)/max(hypot(tx,ty),1e-30))
    end
    outlet_xi_1 = coord(m.X[1,1], m.Y[1,1])[1]
    outlet_xi_2 = coord(m.X[NI+1,1], m.Y[NI+1,1])[1]
    wake_outlet_xi_err = max(abs(outlet_xi_1-ch.xi_out), abs(outlet_xi_2-ch.xi_out))
    outer_outlet_xi_err = max(abs(xis[1]-ch.xi_out), abs(xis[end]-ch.xi_out))
    outlet_parallel_err = 0.0
    for i in (1, NI+1)
        tx = m.X[i,NJ+1]-m.X[i,1]; ty = m.Y[i,NJ+1]-m.Y[i,1]
        outlet_parallel_err = max(outlet_parallel_err,
            abs(tx*esx+ty*esy)/max(hypot(tx,ty),1e-30))
    end
    min_foil_wall_clearance = minimum(begin
        _, eta = coord(m.X[i,1], m.Y[i,1]); h-abs(eta)
    end for i in NIW+1:NIW+NIF+1)

    mq = mesh_quality(m)
    tags = m.outer_face_kind
    (enabled=true,
     tag_counts=(farfield=count(==(OUTER_FARFIELD),tags), inlet=count(==(OUTER_INLET),tags),
                 lower=count(==(OUTER_LOWER_WALL),tags), upper=count(==(OUTER_UPPER_WALL),tags)),
     height_target=ch.height, min_foil_wall_clearance=min_foil_wall_clearance,
     lower_wall_eta_err=lower_eta_err, upper_wall_eta_err=upper_eta_err,
     inlet_xi_err=inlet_xi_err, outlet_xi_err=outer_outlet_xi_err,
     lower_corner_err=lower_corner_err, upper_corner_err=upper_corner_err,
     lower_corner_count=lower_corner_count, upper_corner_count=upper_corner_count,
     wall_parallel_err=wall_parallel_err, inlet_parallel_err=inlet_parallel_err,
     outlet_parallel_err=outlet_parallel_err,
     outer_monotonicity_violations=monotonicity_violations,
     outer_ds_min=ds_min, outer_ds_max=ds_max,
     wake_pair_err=wake_pair_err, wake_alignment_err=wake_alignment_err,
     wake_outlet_xi_err=wake_outlet_xi_err,
     spacing_ratio_max=mq.spacing_ratio_max,
     spacing_ratio_symmetric_max=mq.spacing_ratio_symmetric_max,
     spacing_ratio_location=mq.spacing_ratio_location,
     spacing_ratio_symmetric_location=mq.spacing_ratio_symmetric_location,
     nfold=mq.nfold, area_min=mq.area_min, growth_max_used=mq.growth_max_used)
end

function save_cmesh(m::CMesh, path::AbstractString)
    open(path, "w") do io
        write(io, Int32(m.NI), Int32(m.NJ), Int32(m.NIW), Int32(m.NIF))
        write(io, Float64(m.dn1)); write(io, m.X); write(io, m.Y)
    end
end

# plotting lives in the DRIVER (tools/cmesh_airfoil.jl) — the module stays dependency-light.

end # module
