# =====================================================================================================================
# mesh_remap.jl -- work item W4 (22 Oct 2026): the opt-in wall-resolved remap of a free-air C-mesh's columns
# (PB_MESH_REMAP=common; default none, which reaches nothing in this file). Extended by W4b (22 Oct 2026, after W4's
# review, docs/_records/2026-10-13_pressure_based/air/w4_mesh/review/REVIEW_W4.md): the graded, widening strip of a finite
# trailing-edge base (item 7, the review's F1), an opt-in field of RemapSpec; with it off every statement W4 ran runs as
# before, so W4's meshes rebuild bit for bit. (The review's F4, the band of 40-48 degree faces about 0.1 c off the wall, is
# item 6: measured and accepted, not cured here.) Corrected and extended at the merge of W4/W4b into version 2's main line
# (22 Oct 2026, after W4b's review, docs/_records/2026-10-13_pressure_based/air/w4b_mesh/review/REVIEW_W4B.md): item 5 gives
# where the realised nodes depart most (its F2), item 6 the probe's numbers (F5) and the rotated outer domain's band (F1), and
# item 7(d) the opt-in capped base of the graded strip (F3; strip_cap, default 0: W4b's construction bit for bit).
#
# THE DEFECT IT CURES (W2, docs/_records/2026-10-13_pressure_based/air/w2_air/README.md, sizing section). With a first wall
# spacing dn1 of 1e-6 to 3e-6 c, version 1's generator keeps dn1 on every wake-cut column down to the outlet, where the
# streamwise interval is 2-3 c: aspect ratio 0.65e6-2e6. Each column also carries its own geometric ratio (solved from its own
# length), so the cells' taper changes from row to row by ln(r_i+1/r_i) ~ 1e-4, and in a cell 1e6 times longer than high
# that turns the centroid line through 70-85 degrees (theta) with skew/d 6-13. (In a trapezoid of width w and side heights
# a, b the centroid sits w (b - a)/(6 (a + b)) off the middle, so between rows j and j+1 the centroid line leans by about
# w Delta(b/a)/(12 h): any row-to-row change of the taper is amplified by the aspect ratio.)
#
# THE CONSTRUCTION (after the stall-layer remap of the app's C+H mesh, apps/airfoil_web/stall_layer_mesh.jl,
# _stall_layer_remap_common and _pchip, read and adapted, not included). Only node positions ALONG the parent's grid columns
# move: the tangential distribution, row 0 (the wall, the cut, the strip's legs; item 7 excepted), the outer boundary and
# every column curve are the parent's. Row index j counts nodes from the wall (j = 0) to the outer boundary (j = NJn).
# Items 1-4 define TARGET arc-length distances along each column; item 5 says how nodes are placed at them and how far the
# realised nodes depart from them.
#   1. ONE REFERENCE ROW DISTRIBUTION D(j), shared by every column up to the law's extent y_law: the law dd/dxi = F(d) in the
#      continuous row coordinate xi, F(d) = ln r (d + h1ref/(r - 1)) (an exact geometric distribution of ratio r from h1ref;
#      h1ref the foil columns' median first spacing, r by default their median realised ratio over the kept rows, so on the
#      foil the law is the parent's own distribution). With the STALL LAYER, F = softmin_6(F, H): H = h_p (uniform cells of
#      height h_p) up to the plateau's end y_p, then growing again at ln r per row past y_p (a softplus over 2 h_p), so the plateau is
#      the same in every column (the app's 2.5e-3 c to 0.2 c). D(j) at integer j by RK4 (32 substeps per row); xi_o the
#      fractional row where D reaches y_law.
#   2. THE FOIL COLUMNS (and the trailing-edge junction columns): rows 0..n_keep are the parent's nodes bit for bit (the wall
#      spacing and the generator's frozen boundary-layer rows); from n_keep to n_blend a smoothstep weight hands the
#      parent's distances over to D; D from n_blend to xi_o. So the rows do not step between neighbouring narrow columns.
#   3. THE WAKE COLUMNS: a first spacing that grows smoothly with the distance s along the cut from the trailing edge,
#          h1(s) = h0 + softmin_4( sqrt(h0^2 + (kappa s)^2) - h0 , h_cap - h0 ),   softmin_q(a, b) = (a^-q + b^-q)^(-1/q),
#      h0 the junction column's first spacing, kappa = w_out/(AR_far s_out) (w_out the leg's last cut interval and s_out its
#      length: along a geometric far wake the first row's aspect ratio is then about AR_far) and h_cap = w_out/AR_outlet (the
#      outlet's aspect ratio; with the stall layer at most stall_wake_frac h_p, 0.4 by default: a wake column whose cells
#      approach h_p cannot catch the plateau up, so it merges only past the plateau's end, and along the stretch of wake
#      where the merge row moves from inside the plateau to past it (measured: row 96 to 157 while h1 goes from 0.39 to 0.68
#      h_p, five rows per station) neighbouring columns sit in different phases at the same row, which in cells of aspect
#      ratio ~300 gave theta 78 degrees; the cap keeps every merge inside the plateau). Each wake column then MERGES into D at the same row index: its cells grow at the rate
#          lambda(j) = lambda_ref(j) + g [ (1 - T_A(j)) (ln q_slow - lambda_ref(j)) + T_A(j)(1 - T_B(j)) (ln q_fast - lambda_ref(j)) ],
#      lambda_ref the reference's own rate, T_A, T_B smoothsteps of width w centred at the switch rows j_A, j_B, and
#      g = tanh(Lambda0/lg), Lambda0 = ln(h1/h1ref): a slow phase (cells growing at q_slow, by default uniform) while the
#      reference's cells catch up with and pass the wake's, a fast phase (q_fast) that brings the wake's cells back to the
#      reference's, then the reference itself. j_A and j_B are solved (nested bisection) so that at the merge the cell size
#      AND the distance equal the reference's: from there the wake column's rows ARE D's rows. g -> 0 with Lambda0, so the
#      profile tends to D continuously at the trailing edge; every rate is a convex combination of lambda_ref, ln q_slow and
#      ln q_fast, so no TARGET cell shrinks and no TARGET ratio exceeds max(r, q_fast) (the realised nodes: item 5). Why a
#      merge and not a distance law: a column whose
#      first cell is 1000 times the foil's reaches a given height about 60 rows earlier; if that lag is kept the rows fan out
#      of the trailing edge like rays (measured on the first draft: theta 81 degrees at x 1.26 c, row 97), whereas a merge
#      confines the difference to heights below about j_m h1 (proportional to s), where the rows' slope is about j_m kappa.
#   4. FROM xi_o, EVERY COLUMN'S OWN OUTER SECTION to its end S_i at row NJn: spacing R(t) = F(y_law) exp(Lambda(t)),
#      t = xi - xi_o, whose growth rate moves smoothly (cubic smoothstep over n_t = min(8, (NJn - xi_o)/2) rows) from the
#      law's own rate at y_law to the constant ln rho_i solved (bisection) so that the column ends exactly at S_i.
#   5. PLACEMENT. Nodes are placed on the parent's column curve at those TARGET arc-length distances by monotone cubic
#      Hermite interpolation (Fritsch-Carlson PCHIP) of x(s) and y(s); row 0, the foil's kept rows and the outer node are
#      copied exactly, and a coordinate constant along the parent's column stays exactly constant (a branch that the
#      production meshes never take: their outlet columns' x varies by 1 ulp, and remapped it is x_out +- 7e-15). PCHIP of
#      x(s), y(s) on the parent polyline's arc length is not unit speed where the parent's column turns (item 6), so the
#      REALISED nodes lie on a cubic through the parent's nodes, not on its edges, and their intervals depart from the targets
#      (W4's review, F6: up to 0.5% (A2) and 2.7% (A3) of an interval; 1,162 and 16,958 realised intervals shrink, minimum
#      ratios 0.9996 and 0.977; nodes up to 0.18 of the tangential spacing off the parent's polyline). On the C+H meshes the
#      largest departures (W4b's review, F2: 3.3% on CH at x/c 0.976, lower; 9.2% on its 16-degree variant CH16 at x/c 0.983,
#      upper, 0.1 c above the suction-side trailing edge) sit at the TRAILING-EDGE columns' ONE-NODE kink of 21-36 degrees at
#      the generator's bd = 0.10 c anchor, where PCHIP cuts the corner: they follow that kink's sharpness, not the column's
#      total turn (item 6), whose band departs by at most 0.6% although its turn (49-66 degrees) is larger. The guarantees of
#      items 1-4 ("no cell shrinks", "every ratio within ...") are guarantees of the targets; the suite tests the realised
#      nodes against the measured bounds.
#   6. THE BAND OF 40-48 DEGREE FACES ABOUT 0.1 c OFF THE WALL (the review's F4; measured by W4b, accepted). Version 1's
#      generator builds every column as a wall-orthogonal segment of bd = 0.10 c followed by a straight ray to its outer node
#      (cmesh_gen.jl, the anchored two-segment rays). The remap keeps the column curves and puts the rows at equal arc
#      distance along them, so below the turn the rows are offset curves of the wall, orthogonal to the columns, and above it
#      they meet the rays at the rays' angle to the wall normal: at x/c 0.14 on A2's lower surface the column turns by 45.0
#      degrees in total between 0.02 and 0.5 c (toward its outer node on the inlet; 45.5 the largest over A2's foil columns),
#      and theta goes from 1.0 to 41.3 degrees across the turn, then decays as the fan of rays opens (W4b record,
#      probe_f4_20261022.out; numbers corrected at the merge, W4b's review F5). It is a property of the ray field with a
#      bitwise outer boundary, not of the corner's sharpness: replacing each column's turn by a smooth curve (a quintic
#      Hermite with the parent's ends) raised the worst face (48.81 degrees smoothed over 0.03-0.40 c, 52.35 over 0.05-1.00 c)
#      or folded cells (0.05-3.00 c: 16 cells, neighbouring columns smoothed independently cross). It stays within the
#      65-degree gate (G2a).
#      THE ROTATED OUTER DOMAIN (PB_FARFIELD_ANGLE != 0; W4b's CH16 at 16 degrees; W4b's review, F1). Rotating the outer
#      boundary tilts the suction side's columns further toward their outer nodes (65.5 degrees of total turn at x/c 0.41,
#      upper, against 49.3 on CH's lower side), so the band rises to 62.95 degrees, and EVERY face above 50 degrees (16,529)
#      and above 60 (4,764) lies on the UPPER surface 0.11-1.1 c off the wall over x/c 0.02-0.95, with the smallest cell
#      angle (26.98) 0.15 c above x/c 0.31: inside the 0.2 c stall layer, where the separated shear layer lies at 16 degrees.
#      SUCH A MESH (CH16) IS NOT TO BE USED FOR A STALL OR POST-STALL READING ALONE: compare it against the unrotated mesh (CH)
#      at the same alpha, or cure the column field first (the outer nodes redistributed, or orthogonal-trajectory columns:
#      generator work); audit every angle built with the location of its high-angle faces.
#   7. THE GRADED STRIP (W4b, strip = :graded; the review's F1). Version 1's strip topology puts behind a blunt base of width
#      gap a second block, the strip, of NS = 2 n_base cells across (uniform, gap/NS) bounded by two straight legs, row 0 of the
#      C block's wake columns; the legs' first spacing ramps from dn1 at the corners to gap/NS by a factor growth (1.2) per
#      station. At wall resolution that gave the review a 212-fold ramp within 28 stations (merge rows jumping 49 -> 81, theta
#      66 degrees at x 1.033 c), an interface volume ratio of 195 at the base corners, and a strip whose cells keep gap/NS
#      across to the outlet (aspect ratio 2,241-4,210, a fixed-width strip cannot do better: NS cells of at least w_out/1000
#      need a width of NS w_out/1000). With strip = :graded:
#      (a) the legs' first spacing is item 3's law with h0 = e_c (strip_edge, 0: the corner columns' own first spacing, dn1),
#          on both legs; the C block's leg columns merge into D as wake columns do;
#      (b) the strip's cross distribution FOLLOWS the legs: at station k (k = 1 the base) its two edge cells equal the two legs'
#          first cells e_lo(k), e_up(k), and the NS cells are two geometric halves growing from the edges to one shared middle
#          cell (solved so that they fill the station's length); at the base this is the base graded toward the corners
#          (edge e_c, the corner cells' volume ratio across the interface about 1 instead of 195);
#      (c) the strip WIDENS downstream so that its cells can grow with the legs' first cells: the station's length is
#          W(k)^4 = W0(k)^4 + (c_w NS)^4 (max(e_lo,e_up)^4 - max(e_base_lo,e_base_up)^4). The base contribution is
#          subtracted, so W(1)=W0(1) without an exceptional jump; added width starts quadratically downstream. The factor
#          c_w=1.02 retains the far-wake margin. Reached by moving each leg node outward ALONG ITS C COLUMN by the same distance
#          delta(k) (solved by bisection; delta(1) = 0, the base and the corners exact); the strip's station k is the segment
#          between the two new leg nodes (the strip's edge nodes ARE the C block's new row-0 nodes, copied, so the mesh stays
#          conforming), and the leg column's rows are delta(k) plus its merge profile, then its own outer section;
#      (d) THE CAPPED BASE (the merge of W4/W4b, strip_cap != 0, opt-in; W4b's review F3, and version 1's own review of a
#          graded strip, docs/_records/2026-10-15_mesh_quality/review/REVIEW.md R2, which asked for the centre cell to be
#          capped for the base C_pb). (b)'s halves meet in a middle cell that at wall resolution is several times the parent's
#          uniform base interval (CH: 1.84e-4 c against 4.2e-5, 8 cells over the base's middle half instead of the parent's
#          30). With a cap c (strip_cap > 0 in chords; -1: the parent's base interval gap/NS) the cells from each edge are
#              h_j = e s(e q^j, m)/s(e, m),   s(a, b) = (a^-2 + b^-2)^(-1/2) (a softmin of order 2),
#          so h_0 = e exactly, the cells grow at q (q_lo, q_up: the uncapped base's ratios of (b), so G2k is unchanged)
#          near the edge and level off below (m^2 + e^2)^(1/2) toward the middle, the switch spread over about six rows;
#          the strip's NS becomes the fewest 2 n (n cells per half) that fill the base with no cell above c (CH with the
#          parent's interval: 46 per half, NS 92, the multigrid's coarsest line 2 NJ + NS = 524 against the limit 700);
#          EVERY station takes that NS, with the same ratios from its two edges and ONE m(k) shared by both halves, solved
#          (bisection) to fill the station (`capped_sided`), and (c)'s widening uses that NS. The edge cells stay the legs'
#          first cells; the C block's foil columns, its outer row and the provenance hashes are unchanged, its leg columns
#          move out by the larger delta. WHY THE SMOOTH SWITCH (measured; ../meshint/ records): a HARD switch min(e q^j, m)
#          was built first. At the row where a cell turns from geometric to plateau the station-to-station growth rate
#          jumps (the geometric cells grow with the legs' first cells, up to 3.5% per station; the plateau does not), and in
#          the strip's cells of aspect ratio ~300-500 that turned a face by 40.6 degrees at x 1.30 c (CH's strip: 4.4). The
#          order-2 switch spreads the jump over the switch's rows: the strip's worst face 9.6 degrees, none above 10 (order
#          8: 15.5; 4: 11.9; 1: 7.2 with NS 100). Tag PBMeshRemapCommonV3; strip_cap = 0 (the default) never reaches it.
#      so at the outlet both blocks carry cells of about h_cap = w_out/AR_outlet (or the stall cap) across the wake, aspect
#      ratio about AR_outlet. The corner gate G2l of version 1 (te_ds against the base's first interval) is not met by
#      construction at wall resolution: the base's first interval follows the legs' first cells (dn1), not the foil's
#      trailing-edge interval (the record says so with numbers).
# The row count NJn is a knob (0: the parent's); the pressure multigrid needs 2 NJn (+ NS with a strip, the capped NS with
# item 7(d)) <= MG_COARSE_INVERSE_MAX (700) at L0.
#
# THE STRIP TOPOLOGY, strip = :keep (W4's default): the same code remaps the C block only; the strip block and the interfaces
# (the C legs' row 0) are untouched, so the mesh stays conforming, and the legs' first spacing is the parent's (the leg ramp),
# merged into D like a wake column's. A wake growth of the legs' first spacing is REFUSED there: the strip's cells beside the
# interface are gap/(2 n_base) high whatever the C block does, so growing the legs' first cells would put a volume-ratio jump
# across the interface; the strip's own outlet cells keep their aspect ratio. At production size this path is not fit for
# use (the review's F1: G2a 66 degrees on its S3 mesh); strip = :graded is.
#
# Pure host geometry, once per mesh, Float64, deterministic (no threads). Nothing here is reached when PB_MESH_REMAP=none.
# =====================================================================================================================

"The remap's version tag (in its receipt line) when no W4b option is on: W4's construction, bit for bit."
const MESH_REMAP_VERSION = "PBMeshRemapCommonV1"
"The tag when W4b's graded strip is on (strip = :graded); the receipt then also gives the strip's numbers and hashes."
const MESH_REMAP_VERSION_B = "PBMeshRemapCommonV2"
"The tag when the graded strip's base is capped (strip_cap != 0, header item 7(d)); the receipt then also gives the cap and the strip's NS."
const MESH_REMAP_VERSION_C = "PBMeshRemapCommonV3"
# Base-compatible graded-strip widening; removes the forced station-1/2 jump.
const MESH_REMAP_VERSION_D = "PBMeshRemapCommonV4"

"""
    RemapSpec(; rows=0, n_keep=6, n_blend=24, ratio=0.0, y_law=5.0, wake_ar=500.0, wake_ar_outlet=750.0, stall_h=0.0,
              stall_y=0.0, stall_wake_frac=0.4, q_slow=1.0, q_fast=0.0, merge_w=6.0, merge_lg=1.0,
              strip=:keep, strip_edge=0.0, strip_cw=1.02, strip_cap=0.0)

The remap's settings (the header gives the construction): `rows` the remapped mesh's NJ (0: the parent's); `n_keep` the
foil columns' rows kept bit for bit; `n_blend` the row by which the parent's rows have handed over to D; `ratio` the law's
ratio (0: the foil's median realised ratio over the kept rows); `y_law` the shared law's extent (chords), beyond which each
column has its own outer section; `wake_ar` the far wake's first-row aspect-ratio target AR_far (0: the wake columns keep the
parent's first spacing); `wake_ar_outlet` the outlet's (sets the cap); `stall_h`, `stall_y` the stall layer's plateau
(0: none) and `stall_wake_frac` the wake's cap with it, a fraction of the plateau's height (header, item 3); `q_slow`,
`q_fast`, `merge_w`, `merge_lg` the wake merge's slow and fast growth ratios (q_fast 0: max(1.145, 1 + 1.2 (r - 1)), above
the law's ratio r and, for the usual r = 1.12, below version 1's 1.15 wall-band growth limit), its smoothstep width (rows) and
the amplitude scale of g = tanh(Lambda0/lg).
W4b (header, item 7): `strip` :keep (W4's: the strip block untouched) or :graded (the legs' first spacing grown by the wake
law from `strip_edge`, 0 the corner columns' first spacing; the strip's cross distribution following the legs, graded toward
the corners at the base and widened downstream with the margin `strip_cw`).
The merge of W4/W4b (header, item 7(d); W4b's review F3): `strip_cap` 0 (the default, W4b's graded strip bit for bit), > 0 a
cap in chords on the graded base's cross cells, or -1 the parent's uniform base interval (gap/NS); the strip's NS then grows
to the fewest cells that fill the base at the cap, and every station uses it.
THE ROTATED OUTER DOMAIN (header, item 6; W4b's review F1): a mesh built with PB_FARFIELD_ANGLE != 0 (W4b's CH16) puts its
worst faces in the suction-side stall layer; it is not to be used for a stall or post-stall reading alone.
"""
Base.@kwdef struct RemapSpec
    rows::Int = 0
    n_keep::Int = 6
    n_blend::Int = 24
    ratio::Float64 = 0.0
    y_law::Float64 = 5.0
    wake_ar::Float64 = 500.0
    wake_ar_outlet::Float64 = 750.0
    stall_h::Float64 = 0.0
    stall_y::Float64 = 0.0
    stall_wake_frac::Float64 = 0.4
    q_slow::Float64 = 1.0
    q_fast::Float64 = 0.0
    merge_w::Float64 = 6.0
    merge_lg::Float64 = 1.0
    strip::Symbol = :keep
    strip_edge::Float64 = 0.0
    strip_cw::Float64 = 1.02
    strip_cap::Float64 = 0.0
end

"Whether a spec uses W4b's option (the receipt's tag and fields)."
_remap_b(spec::RemapSpec) = spec.strip !== :keep

"""
    remap_spec(vals) -> RemapSpec or nothing

The remap the knobs ask for: `nothing` with PB_MESH_REMAP=none (the default; the other remap knobs are still parsed, so a
malformed value is refused, but nothing reads them). Refused outside free air (the remap is built and audited for
PB_DOMAIN=freeair, W4). PB_MESH_REMAP_STRIP_CAP (the merge of W4/W4b) is refused without PB_MESH_REMAP_STRIP=graded.
With PB_FARFIELD_ANGLE != 0 the remapped mesh's worst faces lie in the suction-side stall layer (W4b's CH16; header item 6,
W4b's review F1): not for a stall or post-stall reading alone.
"""
function remap_spec(vals::AbstractDict)
    get(vals, "PB_MESH_REMAP", :none) === :none && return nothing
    get(vals, "PB_DOMAIN", :tunnel) === :freeair || throw(ArgumentError("PB_MESH_REMAP=common is built and audited for free " *
        "air only (PB_DOMAIN=freeair; work item W4); the tunnel's meshes are not remapped"))
    k = vals["PB_MESH_REMAP_KEEP"]; l = vals["PB_MESH_REMAP_LAW"]; w = vals["PB_MESH_REMAP_WAKE"]; s = vals["PB_MESH_REMAP_STALL"]
    g = get(vals, "PB_MESH_REMAP_STRIP", nothing)
    gc = get(vals, "PB_MESH_REMAP_STRIP_CAP", nothing)
    gc === nothing || g !== nothing || throw(ArgumentError("PB_MESH_REMAP_STRIP_CAP caps the graded strip's base: it needs " *
        "PB_MESH_REMAP_STRIP=graded (got the strip untouched)"))
    RemapSpec(; rows=l.rows, n_keep=k.n_keep, n_blend=k.n_blend, ratio=l.ratio, y_law=l.y_law > 0 ? l.y_law : 5.0,
              wake_ar=w.ar, wake_ar_outlet=w.ar_outlet, stall_h=s === nothing ? 0.0 : s.h, stall_y=s === nothing ? 0.0 : s.y,
              stall_wake_frac=s === nothing ? 0.4 : s.f,
              strip=g === nothing ? :keep : :graded, strip_edge=g === nothing ? 0.0 : g.edge,
              strip_cap=gc === nothing ? 0.0 : gc.cap)
end

# ---- small pieces ------------------------------------------------------------------------------------------------------
_softmin(a::Float64, b::Float64, q::Real) = (a^(-q) + b^(-q))^(-1/q)
_qmedian(v) = (w = sort(collect(Float64, v)); n = length(w); isodd(n) ? w[(n+1)÷2] : 0.5*(w[n÷2] + w[n÷2+1]))
_smoothstep(t::Real) = (u = clamp(Float64(t), 0.0, 1.0); u*u*(3 - 2u))

"""
    pchip_eval(s, v, q) -> Vector

Monotone cubic Hermite interpolation (Fritsch-Carlson, the harmonic-mean slopes of apps/airfoil_web/stall_layer_mesh.jl's
_pchip) of v(s) at the points q; `s` strictly increasing, q within [s[1], s[end]]. Exact at the knots; monotone data give a
monotone interpolant; constant and linear data are reproduced to round-off (the remap keeps a constant column exact itself).
"""
function pchip_eval(s::Vector{Float64}, v::Vector{Float64}, q::Vector{Float64})
    n = length(s); h = diff(s); d = diff(v) ./ h
    m = zeros(n)
    m[1] = d[1]; m[n] = d[n-1]
    for k in 2:n-1
        m[k] = d[k-1]*d[k] <= 0 ? 0.0 :
            (w1 = 2h[k] + h[k-1]; w2 = h[k] + 2h[k-1]; (w1 + w2)/(w1/d[k-1] + w2/d[k]))
    end
    out = similar(q)
    for (t, x) in enumerate(q)
        k = clamp(searchsortedlast(s, x), 1, n - 1)
        hk = h[k]; u = (x - s[k])/hk
        h00 = (1 + 2u)*(1 - u)^2; h10 = u*(1 - u)^2; h01 = u^2*(3 - 2u); h11 = u^2*(u - 1)
        out[t] = h00*v[k] + h10*hk*m[k] + h01*v[k+1] + h11*hk*m[k+1]
    end
    out
end

"""
    ColumnPath

One parent column as its node polyline (x, y) and cumulative arc length s (s[1] = 0, accumulated as W4 accumulates it), on
which the remap places nodes by PCHIP (header, item 5).
"""
struct ColumnPath
    s::Vector{Float64}; x::Vector{Float64}; y::Vector{Float64}
end
"The points at arc lengths `q` along the path (header, item 5: W4's placement, statement for statement)."
function path_points(p::ColumnPath, q::Vector{Float64})
    xn = all(==(p.x[1]), p.x) ? fill(p.x[1], length(q)) : pchip_eval(p.s, p.x, q)
    yn = all(==(p.y[1]), p.y) ? fill(p.y[1], length(q)) : pchip_eval(p.s, p.y, q)
    xn, yn
end

"The cumulative arc length of the polyline (x, y), accumulated as W4 accumulates a parent column's."
function _arclength(x::Vector{Float64}, y::Vector{Float64})
    s = zeros(length(x))
    for j in 1:length(x)-1; s[j+1] = s[j] + hypot(x[j+1] - x[j], y[j+1] - y[j]); end
    s
end

"The parent's column paths of `b`."
function column_paths(b::CBlocks)
    [(xs = collect(Float64, b.X[i, :]); ys = collect(Float64, b.Y[i, :]); ColumnPath(_arclength(xs, ys), xs, ys)) for i in 1:b.NI+1]
end

"""
    two_sided(L, n, e1, e2) -> (t, q1, q2, m)

The strip's cross distribution of header item 7(b): n cells across a station of length L, the first n1 = n div 2 growing
geometrically from e1 (the lower edge) at q1, the last n - n1 from e2 (the upper edge, counted downward) at q2, the two halves
meeting in one shared middle cell size m. Returns the node fractions t (t[1] = 0, t[n+1] = 1, from the lower edge), the two
ratios and m. Refuses a station too short for halves that do not shrink toward the middle (L < n1 e1 + n2 e2 at m = max(e1, e2)).
"""
function two_sided(L::Float64, n::Int, e1::Float64, e2::Float64)
    n >= 2 && L > 0 && e1 > 0 && e2 > 0 || throw(ArgumentError("two_sided: need n >= 2 and positive L, e1, e2"))
    n1 = n ÷ 2; n2 = n - n1
    G(e, lq, k) = lq == 0 ? e*k : e*expm1(k*lq)/expm1(lq)
    lq(e, k, m) = k <= 1 ? 0.0 : log(m/e)/(k - 1)
    tot(m) = G(e1, lq(e1, n1, m), n1) + G(e2, lq(e2, n2, m), n2)
    lo = log(max(e1, e2))
    tot(exp(lo)) <= L*(1 + 1e-12) || throw(ArgumentError("two_sided: a station of length $L cannot hold $n cells growing from " *
        "its edges ($e1, $e2): it must be at least $(tot(exp(lo)))"))
    hi = lo + 1.0
    while tot(exp(hi)) < L; hi += 1.0; hi > lo + 200 && error("two_sided: no middle cell"); end
    for _ in 1:200
        mid = 0.5*(lo + hi); tot(exp(mid)) < L ? (lo = mid) : (hi = mid)
        hi - lo <= 1e-15*max(1.0, abs(hi)) && break
    end
    m = exp(0.5*(lo + hi)); l1 = lq(e1, n1, m); l2 = lq(e2, n2, m)
    pos = zeros(n + 1)
    for k in 1:n1; pos[k+1] = G(e1, l1, k); end                          # the lower half, from the lower edge
    for k in 1:n2-1; pos[n+1-k] = L - G(e2, l2, k); end                  # the upper half, from the upper edge
    pos[n+1] = L
    t = pos ./ L; t[1] = 0.0; t[n+1] = 1.0
    t, exp(l1), exp(l2), m
end

"The order of the capped strip's smooth switch (header item 7(d)): h_j = e s(e q^j, m)/s(e, m), s the softmin of this order."
const STRIP_CAP_SOFTMIN_P = 2.0
"The capped strip's cell j (0 at the edge) from an edge cell e at the ratio q with the switch parameter m (header item 7(d))."
_capcell(e::Float64, q::Float64, j::Int, m::Float64) = e*_softmin(e*q^j, m, STRIP_CAP_SOFTMIN_P)/_softmin(e, m, STRIP_CAP_SOFTMIN_P)

"""
    capped_sided(L, n, e1, e2, q1, q2) -> (t, m, hmax)

The capped strip's cross distribution of header item 7(d) (the merge of W4/W4b; W4b's review, F3): n cells across a
station of length L, the first n1 = n div 2 from the lower edge of sizes h_j = e1 s(e1 q1^j, m)/s(e1, m), j = 0..n1-1, the
last n2 = n - n1 from the upper edge (counted downward) alike with e2, q2; s the softmin of order STRIP_CAP_SOFTMIN_P; ONE m
shared by both halves, solved by bisection (in ln m) so that they fill L. The edge cells are exactly e1 and e2; each half
grows toward the middle, no adjacent ratio above its q, no cell above (m^p + e^p)^(1/p). Returns the node fractions t
(t[1] = 0, t[n+1] = 1, from the lower edge), m and the largest cell. Refuses a station too short for halves that do not
shrink toward the middle (L at most n1 e1 + n2 e2) and one too long for the ratios (L at least the geometric halves' fill).
"""
function capped_sided(L::Float64, n::Int, e1::Float64, e2::Float64, q1::Float64, q2::Float64)
    n >= 2 && L > 0 && e1 > 0 && e2 > 0 && q1 >= 1 && q2 >= 1 ||
        throw(ArgumentError("capped_sided: need n >= 2, positive L, e1, e2 and ratios >= 1"))
    n1 = n ÷ 2; n2 = n - n1
    tot(m) = (a = 0.0; for j in 0:n1-1; a += _capcell(e1, q1, j, m); end; for j in 0:n2-1; a += _capcell(e2, q2, j, m); end; a)
    geo = (q1 == 1 ? n1*e1 : e1*expm1(n1*log(q1))/(q1 - 1)) + (q2 == 1 ? n2*e2 : e2*expm1(n2*log(q2))/(q2 - 1))
    L > n1*e1 + n2*e2 || throw(ArgumentError("capped_sided: a station of length $L cannot hold $n cells growing from its " *
        "edges ($e1, $e2): it must exceed $(n1*e1 + n2*e2)"))
    L < geo || throw(ArgumentError("capped_sided: a station of length $L is longer than $n cells growing from its edges " *
        "($e1, $e2) at the ratios ($q1, $q2) can fill ($geo): raise the cells or the cap"))
    lo = log(min(e1, e2)) - 60.0; hi = log(max(e1*q1^(n1 - 1), e2*q2^(n2 - 1))) + 60.0
    for _ in 1:400
        mid = 0.5*(lo + hi); tot(exp(mid)) < L ? (lo = mid) : (hi = mid)
        hi - lo <= 4eps(max(abs(lo), abs(hi))) && break
    end
    m = exp(0.5*(lo + hi))
    pos = zeros(n + 1)
    acc = 0.0
    for k in 1:n1; acc += _capcell(e1, q1, k - 1, m); pos[k+1] = acc; end           # the lower half, from the lower edge
    acc = 0.0
    for k in 1:n2-1; acc += _capcell(e2, q2, k - 1, m); pos[n+1-k] = L - acc; end   # the upper half, from the upper edge
    pos[n+1] = L
    issorted(pos) && all(>(0), diff(pos)) || error("capped_sided: a non-positive cell (L $L, n $n, m $m)")
    t = pos ./ L; t[1] = 0.0; t[n+1] = 1.0
    t, m, maximum(diff(pos))
end

"""
    strip_cap_plan(Wb, ns, e1, e2, cap) -> (ns_cap, q1, q2, cap_value, m_uncapped)

Header item 7(d): the capped base's strip count. The uncapped base (`two_sided` with the parent's `ns`) gives the ratios q1, q2
and its middle cell; the cap c is `cap` (> 0, chords) or, for cap = -1, the parent's uniform base interval Wb/ns. With the
switch parameter m_c = (c^p - max(e1, e2)^p)^(1/p) no cell of either half exceeds c; ns_cap = 2 n, n the fewest cells per half
that fill the base at m_c (the base's own m then lies below m_c). Refuses a cap that would not act (at or above the uncapped
middle cell) or that the corner cells reach.
"""
function strip_cap_plan(Wb::Float64, ns::Int, e1::Float64, e2::Float64, cap::Float64)
    cap == -1 || cap > 0 || throw(ArgumentError("mesh remap: strip_cap must be > 0 (chords) or -1 (the parent's base interval)"))
    _, q1, q2, mu = two_sided(Wb, ns, e1, e2)
    c = cap > 0 ? cap : Wb/ns
    c < mu || throw(ArgumentError("mesh remap: the strip's cap $c is not below the uncapped base's middle cell $mu: it would " *
        "not act (lower the cap, or leave PB_MESH_REMAP_STRIP_CAP empty)"))
    c > max(e1, e2) || throw(ArgumentError("mesh remap: the strip's cap $c is not above the base's corner cells ($e1, $e2)"))
    pp = STRIP_CAP_SOFTMIN_P; mc = (c^pp - max(e1, e2)^pp)^(1/pp)
    fillc(e, q, k) = (a = 0.0; for j in 0:k-1; a += _capcell(e, q, j, mc); end; a)
    n = ns ÷ 2
    while fillc(e1, q1, n) + fillc(e2, q2, n) <= Wb
        n += 1; n > 100_000 && error("mesh remap: no capped strip count")
    end
    (2n, q1, q2, c, mu)
end

"The reference law's cell height F(d) (dd/dxi; header, item 1)."
struct _Law
    h1ref::Float64; r::Float64; lr::Float64; hp::Float64; yp::Float64
end
@inline function _law_rate(L::_Law, d::Float64)
    F = L.lr*(d + L.h1ref/(L.r - 1))
    L.hp > 0 || return F
    wh = 2L.hp; x = (d - L.yp)/wh
    H = L.hp + L.lr*wh*(x > 30 ? x : log1p(exp(x)))          # h_p, then growing at ln r per row past y_p (softplus)
    _softmin(F, H, 6)
end

"""The law's rows: d at xi = 0, 1, 2, ... below y_law (RK4 on dd/dxi = F, 32 substeps per row) and the fractional row xi_o
at which d = y_law (the last partial substep inverted, Simpson on dxi/dd = 1/F)."""
function _law_rows(L::_Law, y_law::Float64; nsub::Int=32, maxrows::Int=100_000)
    D = [0.0]; d = 0.0; h = 1.0/nsub
    for row in 0:maxrows
        for k in 1:nsub
            k1 = _law_rate(L, d); k2 = _law_rate(L, d + 0.5h*k1); k3 = _law_rate(L, d + 0.5h*k2); k4 = _law_rate(L, d + h*k3)
            dn = d + h*(k1 + 2k2 + 2k3 + k4)/6
            if dn >= y_law
                g = 0.0; dd = (y_law - d)/8; x = d
                for _ in 1:8
                    a1 = 1/_law_rate(L, x); a2 = 1/_law_rate(L, x + 0.5dd); a4 = 1/_law_rate(L, x + dd)
                    g += dd*(a1 + 4a2 + a4)/6; x += dd
                end
                return D, row + (k - 1)*h + g
            end
            d = dn
        end
        push!(D, d)
    end
    error("mesh remap: the law did not reach y_law = $y_law within $maxrows rows")
end

"""The outer section's distance from y_law after t rows: the integral of R(t) = F_o exp(Lambda(t)),
Lambda(t) = Lr t + (lam_o - Lr) Phi(t), Phi(t) = n_t (u - u^3 + u^4/2), u = t/n_t <= 1 (n_t/2 beyond): the growth rate moves
from lam_o to Lr by a cubic smoothstep over n_t rows (Simpson over [0, min(t, n_t)], closed form beyond)."""
function _outer_dist(t::Float64, Lr::Float64, Fo::Float64, lam::Float64, nt::Float64; m::Int=256)
    Phi(x) = (u = x/nt; u <= 1 ? nt*(u - u^3 + u^4/2) : nt/2)
    R(x) = Fo*exp(Lr*x + (lam - Lr)*Phi(x))
    a = min(t, nt)
    hs = a/m; acc = R(0.0) + R(a)
    for k in 1:m-1; acc += (isodd(k) ? 4 : 2)*R(k*hs); end
    I = acc*hs/3
    if t > nt
        c = Fo*exp((lam - Lr)*nt/2 + Lr*nt)            # R(nt)
        I += abs(Lr) < 1e-14 ? c*(t - nt) : c*expm1(Lr*(t - nt))/Lr
    end
    I
end

"""
    merge_profile(Dref, h1; q_slow=1.0, q_fast=1.145, w=6.0, lg=1.0) -> (d, info)

A wake column's node distances on the rows of the reference distribution `Dref` (nodes 0..J): first cell h1, the slow and
fast phases of the header (item 3), then Dref's rows exactly from the merge row on. `info` = (Lambda0, jA, jB, merge row).
Returns Dref itself when |ln(h1/Dref's first cell)| <= 1e-12. Refuses (ArgumentError) a first cell smaller than the
reference's (not reached by the remap: the wake's first spacing only grows) and a merge that does not fit in J rows.
"""
function merge_profile(Dref::Vector{Float64}, h1::Float64; q_slow::Float64=1.0, q_fast::Float64=1.145, w::Float64=6.0,
                       lg::Float64=1.0)
    J = length(Dref) - 1
    dr = diff(Dref); ldr = log.(dr); lr = diff(ldr)            # lr[j+1]: the reference's rate from cell j to cell j+1
    L0 = log(h1) - ldr[1]
    abs(L0) <= 1e-12 && return copy(Dref), (Lambda0=L0, jA=NaN, jB=NaN, merge=0)
    L0 > 0 || throw(ArgumentError("merge_profile: a first cell below the reference's ($h1 < $(dr[1])) is not built"))
    q_slow < q_fast || throw(ArgumentError("merge_profile: q_slow must be below q_fast"))
    g = tanh(L0/lg); lA = log(q_slow); lB = log(q_fast)
    maximum(lr) < lB || throw(ArgumentError("merge_profile: q_fast $(q_fast) must exceed the reference's largest ratio $(exp(maximum(lr)))"))
    jmax = J - 1 - w/2 - 1e-9                                   # the latest switch whose transition ends by node J - 1
    function run(jA, jB)
        je = ceil(Int, jB + w/2)
        ln = log(h1); d = 0.0
        for j in 0:je-1
            d += exp(ln)
            TA = _smoothstep((j - jA)/w + 0.5); TB = _smoothstep((j - jB)/w + 0.5)
            r = lr[j+1]
            ln += r + g*((1 - TA)*(lA - r) + TA*(1 - TB)*(lB - r))
        end
        (ln - ldr[je+1], d - Dref[je+1], je)
    end
    Lend(jA, jB) = run(jA, jB)[1]
    bis(f, lo, hi) = (for _ in 1:200; mid = 0.5*(lo + hi); f(mid) ? (hi = mid) : (lo = mid); hi - lo <= 1e-11 && break; end; 0.5*(lo + hi))
    # the shortest slow phase that brings the cell size down to the reference's (no fast phase)
    lo = -w/2
    tag = "a first cell $(round(h1, sigdigits=4)) ($(round(exp(L0), sigdigits=4)) times the reference's)"
    Lend(jmax, jmax) <= 0 || throw(ArgumentError("merge_profile: $tag cannot fall to the reference's cell size within $J " *
        "rows: raise the law's extent (y_law) or the rows"))
    jA0 = bis(x -> Lend(x, x) <= 0, lo, jmax)
    # for a slow phase ending at jA, the fast phase ending at jB(jA) that restores the cell size; E its distance excess
    function jB_of(jA)
        Lend(jA, jmax) >= 0 || return NaN
        bis(x -> Lend(jA, x) >= 0, jA, jmax)
    end
    Eof(jA) = (jB = jB_of(jA); isnan(jB) ? NaN : run(jA, jB)[2])
    e0 = Eof(jA0)
    (isnan(e0) || e0 < 0) && throw(ArgumentError("merge_profile: no merge from the shortest slow phase (excess $e0)"))
    hi = jA0
    while true
        hi += 1.0
        hi > jmax && throw(ArgumentError("merge_profile: $tag does not merge within $J rows: raise the law's extent (y_law) " *
            "or the rows"))
        e = Eof(hi)
        isnan(e) && throw(ArgumentError("merge_profile: $tag does not merge within $J rows (the fast phase runs out of rows): " *
            "raise the law's extent (y_law) or the rows"))
        e < 0 && break
    end
    jA = bis(x -> (e = Eof(x); !isnan(e) && e < 0), hi - 1.0, hi)
    jB = jB_of(jA)
    _, _, je = run(jA, jB)
    d = copy(Dref)
    ln = log(h1); acc = 0.0
    for j in 0:je-1
        acc += exp(ln); d[j+2] = acc
        TA = _smoothstep((j - jA)/w + 0.5); TB = _smoothstep((j - jB)/w + 0.5)
        r = lr[j+1]
        ln += r + g*((1 - TA)*(lA - r) + TA*(1 - TB)*(lB - r))
    end
    d[je+1] = Dref[je+1]                                         # the merge node is the reference's (residual ~1e-11 rows)
    d, (Lambda0=L0, jA=jA, jB=jB, merge=je)
end

# Graded-strip widening starts at the exact physical base, with zero initial
# slope under the quadratic wake law. Subtracting the base edge contribution
# avoids the previous nonzero offset followed by a forced station-1 exception.
# The factored fourth-power difference avoids cancellation near the base.
function _graded_strip_width(W0::Float64, edge::Float64, edge0::Float64, NS::Int, cw::Float64)
    edge <= edge0 && return W0
    extra = (cw*NS)^4 * (edge-edge0)*(edge+edge0)*(edge^2+edge0^2)
    sqrt(sqrt(W0^4 + extra))
end

"""
    remap_target_distances(b::CBlocks, spec::RemapSpec; paths=column_paths(b)) -> (D, info)

The arc-length distances (NI+1) x (NJn+1) along the parent's columns at which every C column's new nodes sit (header, items
1-4 and 7; with strip = :graded a leg column's D[i, 1] is its offset delta, its new row 0), and the construction's numbers.
Refuses (ArgumentError) a subdivided mesh, inconsistent settings, a wake column that cannot merge before y_law, an outer
section that would shrink toward the boundary, and a non-monotone column.
"""
function remap_target_distances(b::CBlocks, spec::RemapSpec; paths::Vector{ColumnPath}=column_paths(b))
    b.level == 0 || throw(ArgumentError("mesh remap: remap the generated mesh (level 0), then subdivide (got level $(b.level))"))
    NI, NJ, NIW, NIF = b.NI, b.NJ, b.NIW, b.NIF
    X, Y = b.X, b.Y
    NJn = spec.rows == 0 ? NJ : spec.rows
    nk, nbl = spec.n_keep, spec.n_blend
    2 <= nk < nbl <= NJ || throw(ArgumentError("mesh remap: need 2 <= n_keep < n_blend <= the parent's rows ($NJ); got $nk, $nbl"))
    spec.y_law > 0 || throw(ArgumentError("mesh remap: y_law must be > 0"))
    spec.strip in (:keep, :graded) || throw(ArgumentError("mesh remap: strip must be :keep or :graded (got $(spec.strip))"))
    stall = spec.stall_h > 0
    stall && !(0 < spec.stall_h < spec.stall_y < spec.y_law) &&
        throw(ArgumentError("mesh remap: the stall layer needs 0 < h < y < y_law (got h $(spec.stall_h), y $(spec.stall_y), " *
                            "y_law $(spec.y_law))"))
    strip = b.NS > 0
    graded = spec.strip === :graded
    graded && !strip && throw(ArgumentError("mesh remap: strip = :graded needs a strip mesh (a blunt base with " *
        "PB_MESH_TE_TOPOLOGY=strip)"))
    graded && spec.wake_ar <= 0 && throw(ArgumentError("mesh remap: the graded strip grows the legs' first spacing by the " *
        "wake law: PB_MESH_REMAP_WAKE must be ar_far,ar_outlet > 0 (got 0,0)"))
    graded && !(spec.strip_cw >= 1) && throw(ArgumentError("mesh remap: strip_cw must be >= 1"))
    capped = spec.strip_cap != 0
    capped && !graded && throw(ArgumentError("mesh remap: strip_cap caps the graded strip's base: it needs strip = :graded"))
    strip && !graded && spec.wake_ar > 0 && throw(ArgumentError("mesh remap: a wake growth of the first spacing is refused on the strip " *
        "topology (the strip's cells beside the interface keep gap/(2 n_base); see mesh_remap.jl): set PB_MESH_REMAP_WAKE=0,0, " *
        "or grade the strip (PB_MESH_REMAP_STRIP=graded)"))
    length(paths) == NI + 1 || throw(ArgumentError("mesh remap: $(length(paths)) column paths for $(NI + 1) columns"))
    S = zeros(NI + 1, NJ + 1)
    for i in 1:NI+1, j in 1:NJ
        S[i, j+1] = S[i, j] + hypot(X[i, j+1] - X[i, j], Y[i, j+1] - Y[i, j])
    end
    foil = NIW+1:NIW+NIF+1
    lower = 1:NIW; upper = NIW+NIF+2:NI+1
    h1par = S[:, 2]
    h1ref = _qmedian(h1par[foil])
    r = spec.ratio > 0 ? spec.ratio :
        _qmedian([((S[i, nk+1] - S[i, nk])/S[i, 2])^(1/(nk - 1)) for i in foil])
    r > 1 || throw(ArgumentError("mesh remap: the law's ratio must exceed 1 (got $r)"))
    qf = spec.q_fast > 0 ? spec.q_fast : max(1.145, 1 + 1.2*(r - 1))
    # the wake's first spacing, from the distance along the cut
    h1 = copy(h1par); sw = zeros(NI + 1)
    for i in reverse(lower); sw[i] = sw[i+1] + hypot(X[i+1, 1] - X[i, 1], Y[i+1, 1] - Y[i, 1]); end
    for i in upper; sw[i] = sw[i-1] + hypot(X[i, 1] - X[i-1, 1], Y[i, 1] - Y[i-1, 1]); end
    kap = (0.0, 0.0); cap = (0.0, 0.0)
    if spec.wake_ar > 0
        legs = ((lower, NIW + 1, hypot(X[2, 1] - X[1, 1], Y[2, 1] - Y[1, 1]), sw[1]),
                (upper, NIW + NIF + 1, hypot(X[NI+1, 1] - X[NI, 1], Y[NI+1, 1] - Y[NI, 1]), sw[NI+1]))
        kk = Float64[]; cc = Float64[]
        for (rng, ij, wout, sout) in legs
            h0 = graded && spec.strip_edge > 0 ? spec.strip_edge : h1par[ij]
            graded && h0 < h1par[ij]*(1 - 1e-12) && throw(ArgumentError("mesh remap: the strip's corner interval " *
                "$(spec.strip_edge) is below the corner column's first spacing $(h1par[ij])"))
            k_ = wout/(spec.wake_ar*sout)
            c_ = wout/spec.wake_ar_outlet
            stall && (c_ = min(c_, spec.stall_wake_frac*spec.stall_h))   # with the stall layer: below the plateau's (item 3)
            c_ > h0 || throw(ArgumentError("mesh remap: the wake's cap $c_ is not above the trailing-edge first spacing $h0 " *
                "(lower the outlet's aspect-ratio target, PB_MESH_REMAP_WAKE's second field, or set PB_MESH_REMAP_WAKE=0,0)"))
            for i in rng
                a = sqrt(h0^2 + (k_*sw[i])^2) - h0
                h1[i] = a > 0 ? h0 + _softmin(a, c_ - h0, 4) : h0
            end
            push!(kk, k_); push!(cc, c_)
        end
        kap = (kk[1], kk[2]); cap = (cc[1], cc[2])
    end
    # 7(c). the graded strip's widening: each leg node moves out along its C column by delta (both legs alike) so that the
    # station between them is W(k) long (bisection on the realised station length; delta(1) = 0: the base and the corners)
    delta = zeros(NI + 1); Wk = Float64[]; W0k = Float64[]
    ns_strip = b.NS; q_cap = (NaN, NaN); cap_value = NaN; m_uncapped = NaN
    if graded
        NIs, NS = b.NIs, b.NS
        NIs == NIW || throw(ArgumentError("mesh remap: the strip has $NIs stations for $NIW wake stations"))
        e_c = (spec.strip_edge > 0 ? spec.strip_edge : h1par[NIW+1], spec.strip_edge > 0 ? spec.strip_edge : h1par[NIW+NIF+1])
        if capped               # 7(d): the capped base's NS, the base's ratios and the cap (NS stays the parent's otherwise)
            Wb = hypot(X[NIW+NIF+1, 1] - X[NIW+1, 1], Y[NIW+NIF+1, 1] - Y[NIW+1, 1])
            ns_strip, qa, qb, cap_value, m_uncapped = strip_cap_plan(Wb, NS, e_c[1], e_c[2], spec.strip_cap)
            q_cap = (qa, qb); NS = ns_strip
        end
        for k in 1:NIW+1
            il = NIW + 2 - k; iu = NIW + NIF + k
            pl = paths[il]; pu = paths[iu]
            W0 = hypot(X[iu, 1] - X[il, 1], Y[iu, 1] - Y[il, 1])
            el = k == 1 ? e_c[1] : h1[il]; eu = k == 1 ? e_c[2] : h1[iu]
            W = _graded_strip_width(W0, max(el, eu), max(e_c...), NS, spec.strip_cw)
            push!(W0k, W0); push!(Wk, W)
            k == 1 && continue
            len(dl) = (a = path_points(pl, [dl]); c = path_points(pu, [dl]); hypot(c[1][1] - a[1][1], c[2][1] - a[2][1]))
            if W - W0 > 1e-14*W0
                lo, hi = 0.0, W
                len(hi) >= W || throw(ArgumentError("mesh remap: the strip's station $k cannot widen to $W along its C columns"))
                for _ in 1:200
                    mid = 0.5*(lo + hi); len(mid) < W ? (lo = mid) : (hi = mid)
                    hi - lo <= 4eps(W) && break
                end
                delta[il] = delta[iu] = hi
            end
        end
    end
    # 1. the reference rows, to the law's extent
    L = _Law(h1ref, r, log(r), stall ? spec.stall_h : 0.0, stall ? spec.stall_y : 0.0)
    Dl, xo = _law_rows(L, spec.y_law)
    nl = length(Dl) - 1                                       # reference rows 0..nl (nl = floor(xo))
    nl >= nbl + 1 || throw(ArgumentError("mesh remap: the law reaches y_law at row $(round(xo, digits=2)), before the blend " *
        "ends (row $nbl): lower n_blend or raise y_law"))
    T = NJn - xo
    T >= 4 || throw(ArgumentError("mesh remap: the law uses $(round(xo, digits=2)) of the $NJn rows, leaving " *
        "$(round(T, digits=2)) for the outer section: raise the rows (PB_MESH_REMAP_LAW) or lower y_law"))
    nt = min(8.0, T/2)
    Fo = _law_rate(L, spec.y_law); dl = 1e-7*spec.y_law
    lam = (log(_law_rate(L, spec.y_law + dl)) - log(_law_rate(L, spec.y_law - dl)))/(2dl)*Fo   # d ln F/dxi = F'(d)
    D = zeros(NI + 1, NJn + 1)
    rho = zeros(NI + 1); rmax_out = 0.0; merge_rows = zeros(Int, NI + 1); jAB = fill((NaN, NaN), NI + 1)
    for i in 1:NI+1
        Si = paths[i].s[end]
        di = delta[i]
        Si - di > spec.y_law || throw(ArgumentError("mesh remap: column $i (length $Si) is not longer than y_law $(spec.y_law)"))
        # 4. the column's own outer section: solve its rate so that it ends at S_i
        need = Si - di - spec.y_law
        lo, hi = -0.5, 1.0
        while _outer_dist(T, hi, Fo, lam, nt) < need; hi *= 2; hi > 50 && error("mesh remap: no outer ratio for column $i"); end
        _outer_dist(T, lo, Fo, lam, nt) > need && throw(ArgumentError("mesh remap: column $i's outer section would shrink by " *
            "more than exp(-0.5) per row to end at its boundary ($Si): fewer rows"))
        for _ in 1:200
            mid = 0.5*(lo + hi)
            _outer_dist(T, mid, Fo, lam, nt) < need ? (lo = mid) : (hi = mid)
            hi - lo <= 4eps(max(abs(lo), abs(hi), 1.0)) && break
        end
        Lr = 0.5*(lo + hi)
        Lr >= 0 || throw(ArgumentError("mesh remap: column $i's outer section shrinks toward its boundary (ratio $(exp(Lr))): " *
            "lower the rows (PB_MESH_REMAP_LAW) or raise y_law"))
        rho[i] = exp(Lr); rmax_out = max(rmax_out, exp(max(Lr, lam)))
        d = zeros(NJn + 1)
        # 2-3. the shared rows: the foil's blend, the wake's merge
        if i in foil
            for j in 0:nl; d[j+1] = Dl[j+1]; end
            for j in 0:nbl
                wgt = 1 - _smoothstep((j - nk)/(nbl - nk))
                d[j+1] = j <= nk ? S[i, j+1] : wgt*S[i, j+1] + (1 - wgt)*Dl[j+1]
            end
        else
            dm, mi = merge_profile(Dl, h1[i]; q_slow=spec.q_slow, q_fast=qf, w=spec.merge_w, lg=spec.merge_lg)
            if di == 0
                for j in 0:nl; d[j+1] = dm[j+1]; end
            else
                for j in 0:nl; d[j+1] = di + dm[j+1]; end
            end
            merge_rows[i] = mi.merge; jAB[i] = (mi.jA, mi.jB)
        end
        if di == 0
            for j in nl+1:NJn-1; d[j+1] = spec.y_law + _outer_dist(j - xo, Lr, Fo, lam, nt); end
        else
            for j in nl+1:NJn-1; d[j+1] = di + spec.y_law + _outer_dist(j - xo, Lr, Fo, lam, nt); end
        end
        d[NJn+1] = Si
        st = diff(d)
        minimum(st) > 0 || throw(ArgumentError("mesh remap: column $i is not monotone (row $(argmin(st)))"))
        D[i, :] = d
    end
    gr = [D[i, j+2] - D[i, j+1] for i in 1:NI+1, j in 0:NJn-1]
    rr = [max(gr[i, j+1]/gr[i, j], gr[i, j]/gr[i, j+1]) for i in 1:NI+1, j in 1:NJn-1]
    wk = vcat(collect(lower), collect(upper))
    info = (version=graded ? MESH_REMAP_VERSION_D : _remap_b(spec) ? MESH_REMAP_VERSION_B : MESH_REMAP_VERSION, rows=NJn,
            parent_rows=NJ, n_keep=nk,
            n_blend=nbl, h1ref=h1ref, ratio=r,
            y_law=spec.y_law, xi_o=xo, law_rows=nl, wake_ar=spec.wake_ar, wake_ar_outlet=spec.wake_ar_outlet, wake_slope=kap,
            wake_cap=cap, stall_h=spec.stall_h, stall_y=spec.stall_y, q_slow=spec.q_slow, q_fast=qf,
            outer_ratio=extrema(rho), outer_rate_max=rmax_out, target_ratio_max=maximum(rr),
            merge_row_max=maximum(merge_rows), h1=h1, s_wake=sw, merge_rows=merge_rows, jAB=jAB,
            h1_wake_max=isempty(wk) ? 0.0 : maximum(h1[wk]),
            strip=spec.strip, strip_edge=spec.strip_edge,
            delta=delta, strip_W=Wk, strip_W0=W0k,
            strip_cap=cap_value, strip_ns=ns_strip, strip_ns_parent=b.NS, strip_q_cap=q_cap, strip_m_uncapped=m_uncapped)
    D, info
end

"""
    remap_columns(b::CBlocks, spec::RemapSpec) -> (blocks, info, D)

The remapped mesh (header): the C block's nodes moved along the parent's columns to the distances `D`
(`remap_target_distances`), NJ replaced by the remapped row count; with strip = :graded the strip block's nodes re-placed on
the stations between the legs' new row-0 nodes (header item 7); the outer kinds and the generator's hashes (`blocks.hashes`,
provenance) unchanged; the label gains "+remap". `info` is the construction's record (with strip = :graded also the strip's
base ratios and its first and last stations' widths).
"""
function remap_columns(b::CBlocks, spec::RemapSpec)
    paths = column_paths(b)
    D, info = remap_target_distances(b, spec; paths)
    NI, NJ, NIW, NIF = b.NI, b.NJ, b.NIW, b.NIF
    NJn = size(D, 2) - 1
    foil = NIW+1:NIW+NIF+1
    Xn = zeros(NI + 1, NJn + 1); Yn = zeros(NI + 1, NJn + 1)
    for i in 1:NI+1
        xs = collect(Float64, b.X[i, :]); ys = collect(Float64, b.Y[i, :]); q = D[i, :]
        xn, yn = path_points(paths[i], q)
        if D[i, 1] == 0
            xn[1] = xs[1]; yn[1] = ys[1]
        end
        xn[end] = xs[end]; yn[end] = ys[end]
        if i in foil
            for j in 1:spec.n_keep+1; xn[j] = xs[j]; yn[j] = ys[j]; end
        end
        Xn[i, :] = xn; Yn[i, :] = yn
    end
    Xs, Ys = b.Xs, b.Ys
    sinfo = (q_base=(NaN, NaN), m_base=NaN, q_out=(NaN, NaN), W_base=NaN, W_out=NaN, e_out=(NaN, NaN))
    NSn = b.NS
    if spec.strip === :graded
        NIs, NS = b.NIs, info.strip_ns           # the parent's NS, or the capped base's (header item 7(d))
        NSn = NS; capped = spec.strip_cap != 0
        Xs = zeros(NIs + 1, NS + 1); Ys = zeros(NIs + 1, NS + 1)
        for k in 1:NIs+1
            il = NIW + 2 - k; iu = NIW + NIF + k
            xl, yl = Xn[il, 1], Yn[il, 1]; xu, yu = Xn[iu, 1], Yn[iu, 1]
            Lk = hypot(xu - xl, yu - yl)
            el = k == 1 ? (spec.strip_edge > 0 ? spec.strip_edge : D[NIW+1, 2] - D[NIW+1, 1]) : D[il, 2] - D[il, 1]
            eu = k == 1 ? (spec.strip_edge > 0 ? spec.strip_edge : D[NIW+NIF+1, 2] - D[NIW+NIF+1, 1]) : D[iu, 2] - D[iu, 1]
            if capped
                t, _, m = capped_sided(Lk, NS, el, eu, info.strip_q_cap...); q1, q2 = info.strip_q_cap   # m: the largest cell
            else
                t, q1, q2, m = two_sided(Lk, NS, el, eu)
            end
            Xs[k, 1] = xl; Ys[k, 1] = yl; Xs[k, NS+1] = xu; Ys[k, NS+1] = yu
            for j in 2:NS
                Xs[k, j] = xl + t[j]*(xu - xl); Ys[k, j] = yl + t[j]*(yu - yl)
            end
            k == 1 && (sinfo = merge(sinfo, (q_base=(q1, q2), m_base=m, W_base=Lk)))
            k == NIs + 1 && (sinfo = merge(sinfo, (q_out=(q1, q2), W_out=Lk, e_out=(el, eu))))
            capped && k == NIs + 1 && (sinfo = merge(sinfo, (m_out=m,)))
        end
    end
    nb = CBlocks(string(b.label, "+remap"), NI, NJn, NIW, NIF, Xn, Yn, copy(b.kind), b.NIs, NSn, Xs, Ys, 0, b.hashes)
    (blocks=nb, info=merge(info, (strip_info=sinfo,)), D=D)
end

"The remap's receipt line (the driver prints it after version 1's MESH COORDINATE RECEIPT when PB_MESH_REMAP=common)."
function remap_receipt_line(r)
    a = r.info; h = node_hashes(r.blocks)
    s = @sprintf("PB MESH REMAP: version=%s rows=%d parent_rows=%d keep=%d blend=%d h1ref=%.6e ratio=%.6f y_law=%.6g xi_o=%.3f wake_ar=%.6g wake_ar_outlet=%.6g wake_slope=%.6e,%.6e wake_cap=%.6e,%.6e h1_wake_max=%.6e merge_row_max=%d q_slow=%.4g q_fast=%.4g stall_h=%.6g stall_y=%.6g outer_ratio=%.5f-%.5f target_ratio_max=%.5f x_sha256=%s y_sha256=%s",
             a.version, a.rows, a.parent_rows, a.n_keep, a.n_blend, a.h1ref, a.ratio, a.y_law, a.xi_o, a.wake_ar, a.wake_ar_outlet,
             a.wake_slope..., a.wake_cap..., a.h1_wake_max, a.merge_row_max, a.q_slow, a.q_fast, a.stall_h, a.stall_y,
             a.outer_ratio..., a.target_ratio_max, h[1], h[2])
    a.version == MESH_REMAP_VERSION && return s
    si = a.strip_info
    s2 = s * @sprintf(" strip=%s strip_edge=%.6g", a.strip, a.strip_edge) *
        (a.strip === :graded ? @sprintf(" strip_q_base=%.5f,%.5f strip_W_base=%.6e strip_W_out=%.6e strip_e_out=%.6e,%.6e delta_max=%.6e strip_x_sha256=%s strip_y_sha256=%s",
                                        si.q_base..., si.W_base, si.W_out, si.e_out..., maximum(a.delta), h[3], h[4]) : "")
    isfinite(a.strip_cap) || return s2
    s2 * @sprintf(" strip_cap=%.6e strip_ns=%d strip_ns_parent=%d strip_m_uncapped=%.6e strip_base_max=%.6e strip_out_max=%.6e",
                  a.strip_cap, a.strip_ns, a.strip_ns_parent, a.strip_m_uncapped, si.m_base, si.m_out)
end
