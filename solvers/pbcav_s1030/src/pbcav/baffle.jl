# =====================================================================================================================
# baffle.jl -- interior walls ("baffles") on existing faces of a C-mesh: the Gurney plate (24 Oct 2026; branch pbcav-baffle,
# records docs/_records/2026-10-13_pressure_based/air/baffle/). OPT-IN: PB_BAFFLE empty (the default) reaches nothing here,
# and every mesh and case is bit for bit the one before.
#
# THE USER'S DIRECTION (24 Oct 2026): "the mesh of a finite thickness trailing edge seems perfectly suitable to model the gurney
# flap by just 'disconnecting' some cells following the red line" -- the C block's grid line from the trailing-edge corner away
# from the wall. The Gurney plate (the user's definition of 23 Oct): zero thickness, exactly sharp, perpendicular to the chord at
# the trailing edge, of height h; on one side, or symmetric (h/2 above and h/2 below); the airfoil untouched.
#
# THE MECHANISM (mesh_faces.jl, `BaffleCut`, `_cut_faces!`): the internal faces of the plate become two boundary faces each, one
# per side, tagged BT_BAFFLE_FRONT (the side facing upstream: the foil column's cells) and BT_BAFFLE_BACK (the wake leg's cells),
# both no-slip walls (bc.jl's BC_NOSLIP). Everything the solver does at a wall then holds on both sides by construction, with
# nothing new in any assembler or kernel:
#   * no convective or pressure-correction flux through the plate (a velocity face's flux is u_b . S_b = 0, fixed);
#   * the momentum's no-slip two-point viscous term and its deferred non-orthogonal part, as on the foil;
#   * the pressure equation's Neumann face on both sides (no pressure value; the least-squares pressure stencil leaves it out);
#   * no gradient stencil, line or face interpolation reaches across the plate: the plate's cells are no longer neighbours (the
#     velocity's least-squares stencil takes the wall value 0 instead). The review of 27 Oct 2026 measured it on every work
#     array of a coupled outer iteration (181 arrays: 0 change at the front cells for any perturbation behind the plate);
#   * SST: the wall distance (to the condition set's no-slip faces) includes the plate's segments; k = 0 and omega = omega_wall
#     (W-a, from each face's own wall-normal distance) on both sides; the cells beside the plate are wall-adjacent cells (W6's
#     omega correction off on their faces, the bound's budget as at any wall cell);
#   * the i-lines crossing the plate are split there (mesh_faces.jl), so the line solves (Thomas, PCR) treat the plate as the
#     two lines' ends; the j-lines run along the plate and are unchanged, and so is the multigrid's fold through the wake;
#   * the multigrid (linsolve_mg.jl) is built from the cut operator. ITS COARSE LEVELS DO COUPLE ACROSS THE PLATE (corrected
#     27 Oct 2026, the review's F7): an F-line cell beside the plate has no partner in the C-line across it, so Schaffer's
#     interpolation gives it the virtual position directly across, weighted by the line solve's response to the couplings above
#     the tip; P, and R A P, then couple coarse cells across the plate below the tip. It is a preconditioner matter only (the
#     converged answer is the cut operator's; FCG 14 iterations with the plate against 13 without);
#   * the loads: both sides of every plate are wall faces of `LoadsWork` (pressure and friction), and the surface output lists
#     them (`surface.csv` with tags 13 front / 14 back; `baffle.csv` with plate, side and row).
#
# THE GEOMETRIC RULE (`baffle_plan`, PB_BAFFLE=gurney:side=...,h=...). On a C-mesh (sharp or C+H), plate 1 (lower) lies on node
# column NIW+1 from the lower trailing-edge corner, plate 2 (upper) on node column NIW+NIF+1 from the upper corner. The plate's
# height is measured along the chord's normal (the chord from the leading edge, the wall-row node farthest from the trailing
# edge's midpoint, to that midpoint), and the plate ends at the line's node NEAREST that height (h, or h/2 each when symmetric;
# the outer boundary excluded). Two optional steps, each recorded, make it the user's plate exactly:
#   * STRAIGHTEN (on by default when the line deviates from the chord's normal by more than 0.5 degree; measured on the free-air
#     C+H demo mesh: 8.0-9.7 degrees, the generator's corner column bisecting the corner): the line's nodes from the corner to
#     the tip are moved onto the normal;
#   * SNAP EXACT (the default): the tip at the requested height exactly (snap=nearest keeps the node's height).
#   THE LINE'S NEW NODE DISTANCES (27 Oct 2026, the review's F6: the first form scaled every node by h/s_K, which changed the
#   trailing edge's first-cell heights by up to 7% and kinked the growth ratio at the tip row): node k of the line goes to the
#   arc distance sigma_k = s_k + delta S(x_k), s the arc length along the line as the mesh has it, delta the tip's change, x_k =
#   (s_k - s_a)/(s_K - s_a) and S the smoothstep x^2 (3 - 2x): the first two cells keep their lengths (s_a = s_3; from the corner
#   when the plate has fewer than 4 faces), the stretch rises smoothly along the plate, and its slope is zero at the tip, where
#   the displacement above (below) continues with zero slope too.
#   The moved line's displacement D_k (k the row) is carried to its neighbourhood smoothly: above the tip it decays to zero over
#   an arc length L (a cosine), and every other C column i takes w_i D_k at the same row, w_i = 0.5 (1 + cos(pi d_i/L)) within
#   the distance d_i < L along the wall row (the foil on one side, the wake leg on the other), else 0. L = pi max|D|/(2 shear)
#   bounds the relative tangential shear the movement adds to any cell by `shear` (default 0.1). The wall row, the strip (H
#   block), the outer boundary and the outlets do not move (refused if the window would reach them); every cell is checked convex.
#   With both steps off (straighten=off, snap=nearest) no node moves: the plate is the grid line as the mesh has it.
#   A plate of height 0 makes no cut: the mesh is the one without a baffle, bit for bit.
# THE PLATE'S QUALITY GATE (27 Oct 2026, the review's F3; limits registered before the range was measured): on the blocks the
#   faces are built on, the plate's faces' boundary non-orthogonality (<= BAFFLE_THETA_B_MAX: a tall plate's top faces sit in the
#   corner column's thin cells far from the wall), the tip face's length over h (<= BAFFLE_TIP_FRACTION_MAX: a plate of a few
#   faces), and the largest change of a moved column's first-cell height (<= BAFFLE_FIRST_CELL_MAX). A plate outside them is
#   refused, the message naming the range of h the gate admits on that mesh (`baffle_h_range`). The receipt carries the three
#   values, the wall-normal distance range of the plate's faces and the growth-ratio change along the plate's line.
# THE START: see freeair.jl (the ramp start, PB_START=ramp) and the records' README section 3: the plain projection start is
#   singular at the plate's tip and the base corners.
# KernelAbstractions issue 652: no kernel here (this file is host setup and host output only).
# =====================================================================================================================

"""
    BaffleSpec(; kind=:gurney, side=:lower, h, snap=:exact, straighten=0.5, shear=0.1, check=true)

One interior-wall request (PB_BAFFLE; knobs.jl `_parse_baffle`): `side` :lower, :upper or :symmetric (h/2 on each corner),
`h` the plate height in chords (0 makes no plate), `snap` :exact or :nearest, `straighten` the largest deviation from the
chord's normal (degrees) left as it is (Inf: never straightened), `shear` the largest relative shear the node movement adds.
`check` (not a knob: always true from PB_BAFFLE) applies the plate's quality gate; false only for diagnostics and tests.
"""
struct BaffleSpec
    kind::Symbol; side::Symbol; h::Float64; snap::Symbol; straighten::Float64; shear::Float64; check::Bool
    function BaffleSpec(kind::Symbol, side::Symbol, h::Real, snap::Symbol, straighten::Real, shear::Real, check::Bool=true)
        kind === :gurney || throw(ArgumentError("BaffleSpec: kind must be :gurney (got :$kind)"))
        side in (:lower, :upper, :symmetric) || throw(ArgumentError("BaffleSpec: side must be :lower, :upper or :symmetric"))
        isfinite(h) && 0 <= h <= 0.2 || throw(ArgumentError("BaffleSpec: h must lie in [0, 0.2] chords (got $h)"))
        snap in (:exact, :nearest) || throw(ArgumentError("BaffleSpec: snap must be :exact or :nearest"))
        straighten >= 0 && !isnan(straighten) || throw(ArgumentError("BaffleSpec: straighten must be >= 0 degrees (Inf: off)"))
        0 < shear <= 0.5 || throw(ArgumentError("BaffleSpec: shear must lie in (0, 0.5]"))
        new(kind, side, Float64(h), snap, Float64(straighten), Float64(shear), check)
    end
end
BaffleSpec(; kind::Symbol=:gurney, side::Symbol=:lower, h::Real, snap::Symbol=:exact, straighten::Real=0.5, shear::Real=0.1,
           check::Bool=true) = BaffleSpec(kind, side, h, snap, straighten, shear, check)

"The knob's spec string of a BaffleSpec (the receipt's form; `_parse_baffle` reads it back to the same spec)."
spec_string(s::BaffleSpec) = string("gurney:side=", s.side, ",h=", s.h, ",snap=", s.snap, ",straighten=",
                                    isinf(s.straighten) ? "off" : string(s.straighten), ",shear=", s.shear)

"""
    baffle_spec(knobs) -> BaffleSpec or nothing

The request of PB_BAFFLE (read_knobs's parsed value), or nothing when it is empty (the default). The quality gate is on.
"""
function baffle_spec(vals::AbstractDict)
    v = get(vals, "PB_BAFFLE", nothing)
    v === nothing && return nothing
    BaffleSpec(; kind=v.kind, side=v.side, h=v.h, snap=v.snap, straighten=v.straighten, shear=v.shear)
end

"""The plate's quality limits (the review's F3, 27 Oct 2026; registered before the range of h was measured on any mesh):
the plate's faces' boundary non-orthogonality (degrees), the tip face's length over h, and the change of a moved column's
first-cell height (relative). Recorded in PB BAFFLE; a plate outside any of them is refused (`baffle_plan`)."""
const BAFFLE_THETA_B_MAX = 30.0
const BAFFLE_TIP_FRACTION_MAX = 0.20
const BAFFLE_FIRST_CELL_MAX = 0.05

"""
    BafflePlan

A placed baffle: the `spec`, the `blocks` the faces are built on (the given blocks themselves when no node moved, else a copy
with the moved C-block nodes; label "+baffle"; their `hashes` stay the GENERATED mesh's, and the moved nodes' hashes are
`node_hashes(plan.blocks)`, the receipt's x/y_sha256), the `cuts` (one BaffleCut per plate; none for h = 0), one record per plate
(`plates`: side, column, faces, the requested, grid and realised heights, the deviations from the chord's normal before and
after, whether it was straightened and snapped, the largest displacement and the window, and the quality: theta_b_max and its
row, d1_min, d1_max, tip_fraction, growth_kink), the number of nodes moved and the largest displacement, the smallest corner
angle of the cells beside moved nodes before and after, the largest first-cell height change of a moved column, whether the
quality gate was applied, and the chord used. `build_faces(plan.blocks; baffle=plan)` builds the mesh.
"""
struct BafflePlan
    spec::BaffleSpec
    blocks::CBlocks
    cuts::Vector{BaffleCut}
    plates::Vector{NamedTuple}
    moved::Int
    disp_max::Float64
    angle_min_before::Float64
    angle_min_after::Float64
    first_cell_change::Float64
    checked::Bool
    chord::NTuple{4,Float64}       # leading edge (x, y), unit chord direction (cx, cy)
    refine::Any                    # nothing (the default, PB_BAFFLE_REFINE empty) or the refinement's record (`baffle_refine`)
end
BafflePlan(spec, blocks, cuts, plates, moved, disp_max, amb, ama, fcc, checked, chord) =
    BafflePlan(spec, blocks, cuts, plates, moved, disp_max, amb, ama, fcc, checked, chord, nothing)

"The smallest corner angle (degrees) of the C cells (i, j) in `cells`."
function _min_corner_angle(X, Y, cells)
    a = Inf
    for (i, j) in cells
        p = ((X[i, j], Y[i, j]), (X[i+1, j], Y[i+1, j]), (X[i+1, j+1], Y[i+1, j+1]), (X[i, j+1], Y[i, j+1]))
        for k in 1:4
            o = p[k]; u = p[mod1(k - 1, 4)]; v = p[mod1(k + 1, 4)]
            e1 = (u[1] - o[1], u[2] - o[2]); e2 = (v[1] - o[1], v[2] - o[2])
            a = min(a, acosd(clamp((e1[1]*e2[1] + e1[2]*e2[2])/(hypot(e1...)*hypot(e2...)), -1.0, 1.0)))
        end
    end
    a
end

"The smallest corner cross product over the C block's cells (positive: every cell strictly convex and counter-clockwise)."
function _min_corner_cross(X, Y)
    NI, NJ = size(X) .- 1; m = Inf; at = (0, 0)
    for i in 1:NI, j in 1:NJ
        p = ((X[i, j], Y[i, j]), (X[i+1, j], Y[i+1, j]), (X[i+1, j+1], Y[i+1, j+1]), (X[i, j+1], Y[i, j+1]))
        for k in 1:4
            a = p[k]; b = p[mod1(k + 1, 4)]; e = p[mod1(k + 2, 4)]
            cr = (b[1] - a[1])*(e[2] - b[2]) - (b[2] - a[2])*(e[1] - b[1])
            cr < m && (m = cr; at = (i, j))
        end
    end
    m, at
end

"The angle (degrees) between the segment (dx, dy) and the unit vector (nx, ny)."
_seg_dev(dx, dy, nx, ny) = acosd(clamp((dx*nx + dy*ny)/hypot(dx, dy), -1.0, 1.0))

"The smoothstep x^2 (3 - 2x) on [0, 1] (0 below, 1 above)."
_smoothstep(x) = x <= 0 ? 0.0 : x >= 1 ? 1.0 : x*x*(3 - 2x)

"The true (shoelace) centroid of C cell (i, j), as mesh_faces.jl's cell kernel forms it (relative to the vertex average)."
function _cell_centroid(X, Y, i, j)
    xs = (X[i, j], X[i+1, j], X[i+1, j+1], X[i, j+1]); ys = (Y[i, j], Y[i+1, j], Y[i+1, j+1], Y[i, j+1])
    mx = 0.25*(xs[1] + xs[2] + xs[4] + xs[3]); my = 0.25*(ys[1] + ys[2] + ys[4] + ys[3])
    a2 = 0.0; cx = 0.0; cy = 0.0
    for k in 1:4
        u1 = xs[k] - mx; w1 = ys[k] - my; u2 = xs[mod1(k + 1, 4)] - mx; w2 = ys[mod1(k + 1, 4)] - my
        cr = u1*w2 - u2*w1; a2 += cr; cx += (u1 + u2)*cr; cy += (w1 + w2)*cr
    end
    (mx + cx/(3a2), my + cy/(3a2))
end

"""The plate's quality on blocks (X, Y), for the faces of node column i0, rows 1..K-1, both sides: the largest boundary
non-orthogonality theta_b (the angle between the face's outward area vector and the owner's centroid-to-face vector: the boundary
faces' metric of mesh_faces.jl) and its row, the wall-normal distance range d1 = S_b.d_b/|S_b|, and the tip face's length over h."""
function _plate_quality(X, Y, i0::Integer, K::Integer, h::Real)
    tb = 0.0; tbrow = 0; d1lo = Inf; d1hi = 0.0
    for j in 1:K-1
        xa = X[i0, j]; ya = Y[i0, j]; xb = X[i0, j+1]; yb = Y[i0, j+1]
        sx = yb - ya; sy = -(xb - xa); sm = hypot(sx, sy); xf = 0.5*(xa + xb); yf = 0.5*(ya + yb)
        for (ic, sg) in ((i0 - 1, 1.0), (i0, -1.0))           # the cell before the line (S points out of it), the cell after
            cx, cy = _cell_centroid(X, Y, ic, j)
            dx = xf - cx; dy = yf - cy; sd = sg*(sx*dx + sy*dy)
            t = acosd(clamp(sd/(sm*hypot(dx, dy)), -1.0, 1.0))
            t > tb && (tb = t; tbrow = j)
            d1 = sd/sm; d1lo = min(d1lo, d1); d1hi = max(d1hi, d1)
        end
    end
    (theta_b_max=tb, theta_b_row=tbrow, d1_min=d1lo, d1_max=d1hi,
     tip_fraction=hypot(X[i0, K] - X[i0, K-1], Y[i0, K] - Y[i0, K-1])/h)
end

"""
    baffle_plan(b::CBlocks, spec::BaffleSpec) -> BafflePlan

The Gurney plate(s) of `spec` on the C-mesh `b` by the geometric rule of this file's header: the corner column(s), the node
nearest the height, and (when asked) the straightening and the exact snap with their smooth neighbourhood; then the plate's
quality (the gate of the header, applied when spec.check). Host setup, once per mesh. Refuses a plate reaching within two rows
of the outer boundary, heights that do not increase along the line (the line does not leave the wall), a window or blend that
would reach the outer boundary or an outlet column, a non-convex cell after the movement, and (spec.check) a plate outside the
quality gate, naming the range of h the gate admits on this mesh.
"""
function baffle_plan(b::CBlocks, spec::BaffleSpec)
    NI, NJ, NIW, NIF = b.NI, b.NJ, b.NIW, b.NIF
    X0, Y0 = b.X, b.Y
    il = NIW + 1; iu = NIW + NIF + 1
    xm = 0.5*(X0[il, 1] + X0[iu, 1]); ym = 0.5*(Y0[il, 1] + Y0[iu, 1])
    ile = il - 1 + argmax([hypot(X0[i, 1] - xm, Y0[i, 1] - ym) for i in il:iu])
    ch = hypot(xm - X0[ile, 1], ym - Y0[ile, 1]); cx = (xm - X0[ile, 1])/ch; cy = (ym - Y0[ile, 1])/ch
    hs = spec.side === :lower ? (spec.h, 0.0) : spec.side === :upper ? (0.0, spec.h) : (spec.h/2, spec.h/2)
    DX = zeros(NI + 1, NJ + 1); DY = zeros(NI + 1, NJ + 1)
    cuts = BaffleCut[]; plates = NamedTuple[]
    for (p, i0, hp, lab) in ((1, il, hs[1], :lower), (2, iu, hs[2], :upper))
        hp > 0 || continue
        nx, ny = p == 1 ? (cy, -cx) : (-cy, cx)                   # the chord's normal, away from the foil
        H = [(X0[i0, k] - X0[i0, 1])*nx + (Y0[i0, k] - Y0[i0, 1])*ny for k in 1:NJ+1]
        s = zeros(NJ + 1); for k in 2:NJ+1; s[k] = s[k-1] + hypot(X0[i0, k] - X0[i0, k-1], Y0[i0, k] - Y0[i0, k-1]); end
        K = 1 + argmin([abs(H[k] - hp) for k in 2:NJ-1])           # the tip node (2 .. NJ-1: two rows kept below the boundary)
        all(>(0), diff(H[1:K+1])) || throw(ArgumentError("baffle_plan: plate $p ($lab): the heights along node column $i0 do not increase from the corner to row $(K + 1) " *
            "(the line does not leave the wall along the chord's normal)"))
        devg = maximum(_seg_dev(X0[i0, k] - X0[i0, k-1], Y0[i0, k] - Y0[i0, k-1], nx, ny) for k in 2:K)
        straight = devg > spec.straighten
        ht = spec.snap === :exact ? hp : H[K]
        snapped = spec.snap === :exact && H[K] != hp
        dmax = 0.0; L = 0.0; nwin = 0
        if straight || snapped
            # the line's new arc distances (header): the first two cells kept, a smoothstep stretch to the tip, zero slope there
            ka = K >= 5 ? 3 : 1; sa = s[ka]
            sK = straight ? ht : ht*s[K]/H[K]                      # the tip's distance along the (straightened) line
            dl = sK - s[K]
            Dx = zeros(NJ + 1); Dy = zeros(NJ + 1)
            for k in 2:K
                sg = k <= ka ? s[k] : s[k] + dl*_smoothstep((s[k] - sa)/(s[K] - sa))
                qx, qy = straight ? (X0[i0, 1] + nx*sg, Y0[i0, 1] + ny*sg) :
                                    (X0[i0, 1] + (X0[i0, k] - X0[i0, 1])*sg/s[k], Y0[i0, 1] + (Y0[i0, k] - Y0[i0, 1])*sg/s[k])
                Dx[k] = qx - X0[i0, k]; Dy[k] = qy - Y0[i0, k]
            end
            dmax = maximum(hypot(Dx[k], Dy[k]) for k in 2:K)
            L = pi*dmax/(2*spec.shear)
            for k in K+1:NJ+1                                      # the blend above the tip, in arc length along the line
                r = (s[k] - s[K])/L; be = r < 1 ? 0.5*(1 + cos(pi*r)) : 0.0
                Dx[k] = be*Dx[K]; Dy[k] = be*Dy[K]
            end
            (Dx[NJ+1] == 0 && Dy[NJ+1] == 0) || throw(ArgumentError("baffle_plan: plate $p ($lab): the blend above the tip ($(round(L, sigdigits=4)) c) would reach the outer boundary"))
            # the window along the wall row (row 1), both ways from the corner
            d = zeros(NI + 1)
            for i in i0-1:-1:1; d[i] = d[i+1] + hypot(X0[i+1, 1] - X0[i, 1], Y0[i+1, 1] - Y0[i, 1]); end
            for i in i0+1:NI+1; d[i] = d[i-1] + hypot(X0[i, 1] - X0[i-1, 1], Y0[i, 1] - Y0[i-1, 1]); end
            for i in 1:NI+1
                d[i] < L || continue
                (i == 1 || i == NI + 1) && throw(ArgumentError("baffle_plan: plate $p ($lab): the window ($(round(L, sigdigits=4)) c) would reach an outlet column"))
                w = 0.5*(1 + cos(pi*d[i]/L)); nwin += 1
                for k in 2:NJ+1; DX[i, k] += w*Dx[k]; DY[i, k] += w*Dy[k]; end
            end
        end
        push!(cuts, BaffleCut(; dir=:i, line=i0, lo=1, hi=K - 1, plate=p, front=p == 1 ? :plus : :minus))
        push!(plates, (plate=p, side=lab, column=i0, faces=K - 1, tip_row=K, h=hp, h_grid=H[K], s_grid=s[K], dev_grid=devg,
                       straightened=straight, snapped=snapped, disp_max=dmax, window=L, window_columns=nwin, nx=nx, ny=ny))
    end
    # Match neighbouring row heights at the plate. A column-only straightening
    # leaves the neighbouring centroids offset along a cambered foil's plate;
    # as TE cells become thin this creates large boundary non-orthogonality.
    # Moving interior nodes only along the plate normal aligns the adjacent
    # row lines, with a smooth compact window. The section, base interfaces,
    # plate line, exact tip, outer boundary and wall spacings remain checked.
    if spec.snap === :exact
        for q in plates
            i0=q.column;K=q.tip_row;nx=q.nx;ny=q.ny
            width=max(3q.h,0.02)
            d=zeros(NI+1)
            for i in i0-1:-1:1;d[i]=d[i+1]+hypot(X0[i+1,1]-X0[i,1],Y0[i+1,1]-Y0[i,1]);end
            for i in i0+1:NI+1;d[i]=d[i-1]+hypot(X0[i,1]-X0[i-1,1],Y0[i,1]-Y0[i-1,1]);end
            xroot=X0[i0,1];yroot=Y0[i0,1]
            for j in 2:NJ
                height=(X0[i0,j]+DX[i0,j]-xroot)*nx+(Y0[i0,j]+DY[i0,j]-yroot)*ny
                fade=(height<=q.h ? 1.0 : 1-_smoothstep((height-q.h)/max(q.h,0.01)))*_smoothstep((j-2)/4)
                fade>0 || continue
                for i in 2:NI
                    (i==i0 || d[i]>=width) && continue
                    weight=0.5*(1+cos(pi*d[i]/width))*fade
                    here=(X0[i,j]+DX[i,j]-X0[i,1])*nx+(Y0[i,j]+DY[i,j]-Y0[i,1])*ny
                    delta=weight*(height-here)
                    DX[i,j]+=delta*nx;DY[i,j]+=delta*ny
                end
            end
        end
    end
    moved = count(k -> DX[k] != 0 || DY[k] != 0, eachindex(DX))
    fcc = 0.0
    if moved == 0
        blocks = b; amb = ama = NaN
    else
        all(==(0), view(DX, :, 1)) && all(==(0), view(DY, :, 1)) || error("baffle_plan: a wall-row node moved (internal error)")
        X = copy(X0); Y = copy(Y0)                               # only the moved nodes change (an unmoved -0.0 stays -0.0)
        for k in eachindex(DX); (DX[k] != 0 || DY[k] != 0) && (X[k] += DX[k]; Y[k] += DY[k]); end
        near = Set{Tuple{Int,Int}}()
        for i in 1:NI+1, k in 1:NJ+1
            (DX[i, k] != 0 || DY[i, k] != 0) || continue
            for (a, c) in ((i - 1, k - 1), (i, k - 1), (i - 1, k), (i, k)); (1 <= a <= NI && 1 <= c <= NJ) && push!(near, (a, c)); end
        end
        amb = _min_corner_angle(X0, Y0, near); ama = _min_corner_angle(X, Y, near)
        mc, at = _min_corner_cross(X, Y)
        mc > 0 || throw(ArgumentError("baffle_plan: the node movement leaves C cell $(at) non-convex (corner cross product $(mc)); lower the shear " *
            "(PB_BAFFLE ...,shear=) or set straighten=off / snap=nearest"))
        # the first-cell height of every column whose row-2 node moved (the wall row is fixed)
        for i in 1:NI+1
            (DX[i, 2] != 0 || DY[i, 2] != 0) || continue
            h0 = hypot(X0[i, 2] - X0[i, 1], Y0[i, 2] - Y0[i, 1]); h1 = hypot(X[i, 2] - X[i, 1], Y[i, 2] - Y[i, 1])
            fcc = max(fcc, abs(h1/h0 - 1))
        end
        blocks = CBlocks(b.label * "+baffle", NI, NJ, NIW, NIF, X, Y, b.kind, b.NIs, b.NS, b.Xs, b.Ys, b.level, b.hashes)
    end
    # the realised plates on the blocks the faces are built on, and their quality
    Xb, Yb = blocks.X, blocks.Y
    seglen(XX, YY, i, k) = hypot(XX[i, k+1] - XX[i, k], YY[i, k+1] - YY[i, k])
    plates = map(plates) do q
        hr = (Xb[q.column, q.tip_row] - Xb[q.column, 1])*q.nx + (Yb[q.column, q.tip_row] - Yb[q.column, 1])*q.ny
        pq = _plate_quality(Xb, Yb, q.column, q.tip_row, hr)
        gk = maximum(abs((seglen(Xb, Yb, q.column, k)/seglen(Xb, Yb, q.column, k-1))/
                         (seglen(X0, Y0, q.column, k)/seglen(X0, Y0, q.column, k-1)) - 1) for k in 2:min(NJ, q.tip_row + 10))
        merge(q, (h_real=hr,
                  dev_real=maximum(_seg_dev(Xb[q.column, k] - Xb[q.column, k-1], Yb[q.column, k] - Yb[q.column, k-1], q.nx, q.ny)
                                   for k in 2:q.tip_row),
                  tip=(Xb[q.column, q.tip_row], Yb[q.column, q.tip_row]), root=(Xb[q.column, 1], Yb[q.column, 1])), pq,
              (growth_kink=gk,))
    end
    plan = BafflePlan(spec, blocks, cuts, plates, moved, moved == 0 ? 0.0 : maximum(hypot.(DX, DY)), amb, ama, fcc, spec.check,
                      (X0[ile, 1], Y0[ile, 1], cx, cy))
    if spec.check
        why = baffle_gate_failures(plan)
        if !isempty(why)
            r = baffle_h_range(b, spec)
            rng = isempty(r.admitted) ? "no height on this mesh's corner line(s) passes the gate" :
                  "the gate admits h from $(round(r.lo, sigdigits=3)) to $(round(r.hi, sigdigits=3)) c on this mesh" *
                  (r.contiguous ? "" : " (not every height between: $(length(r.admitted)) of $(r.scanned) scanned heights pass)")
            throw(ArgumentError("PB_BAFFLE: the plate of h = $(spec.h) c ($(spec.side)) fails the plate's quality gate on this mesh: " *
                join(why, "; ") * "; $rng (records air/baffle/README.md: the gate and the range on the app's demo mesh)"))
        end
    end
    plan
end

"The gate's failures of a plan (empty when every plate passes; the limits BAFFLE_THETA_B_MAX, BAFFLE_TIP_FRACTION_MAX, BAFFLE_FIRST_CELL_MAX)."
function baffle_gate_failures(plan::BafflePlan)
    why = String[]
    for q in plan.plates
        q.theta_b_max > BAFFLE_THETA_B_MAX && push!(why, "plate $(q.side): the faces' boundary non-orthogonality reaches " *
            "$(round(q.theta_b_max, digits=1)) deg at row $(q.theta_b_row) (limit $(BAFFLE_THETA_B_MAX); a tall plate's top faces sit in thin cells far from the wall)")
        q.tip_fraction > BAFFLE_TIP_FRACTION_MAX && push!(why, "plate $(q.side): the tip face is $(round(q.tip_fraction, digits=3)) " *
            "of h (limit $(BAFFLE_TIP_FRACTION_MAX); $(q.faces) faces: too few to resolve the plate)")
    end
    plan.first_cell_change > BAFFLE_FIRST_CELL_MAX && push!(why, "the trailing edge's first-cell height changes by " *
        "$(round(100plan.first_cell_change, digits=2))% (limit $(100BAFFLE_FIRST_CELL_MAX)%)")
    why
end

"""
    baffle_h_range(b, spec) -> (admitted, lo, hi, contiguous, scanned)

The heights the quality gate admits for `spec`'s side, snap, straightening and shear on the C-mesh `b`: every node of the corner
line(s) is tried as the tip (its height, twice it when symmetric), with the gate off, and kept when the plan passes it. `lo`,
`hi` the smallest and largest admitted; `contiguous` whether every scanned height between them passes. Host setup; for the
gate's refusal message and the records.
"""
function baffle_h_range(b::CBlocks, spec::BaffleSpec)
    i0 = spec.side === :upper ? b.NIW + b.NIF + 1 : b.NIW + 1
    p0 = baffle_plan(b, BaffleSpec(; kind=spec.kind, side=spec.side === :upper ? :upper : :lower, h=1e-9, snap=:nearest,
                                   straighten=Inf, shear=spec.shear, check=false))
    q = only(p0.plates); nx, ny = q.nx, q.ny
    hk = [(b.X[i0, k] - b.X[i0, 1])*nx + (b.Y[i0, k] - b.Y[i0, 1])*ny for k in 2:b.NJ-1]
    hk = [h*(spec.side === :symmetric ? 2 : 1) for h in hk if h*(spec.side === :symmetric ? 2 : 1) <= 0.2]
    ok = Bool[]
    for h in hk
        pass = try
            pl = baffle_plan(b, BaffleSpec(; kind=spec.kind, side=spec.side, h, snap=spec.snap, straighten=spec.straighten,
                                           shear=spec.shear, check=false))
            isempty(baffle_gate_failures(pl))
        catch e
            e isa ArgumentError || rethrow()
            false
        end
        push!(ok, pass)
    end
    adm = hk[ok]
    isempty(adm) && return (admitted=adm, lo=NaN, hi=NaN, contiguous=false, scanned=length(hk))
    a = findfirst(ok); z = findlast(ok)
    (admitted=adm, lo=hk[a], hi=hk[z], contiguous=all(ok[a:z]), scanned=length(hk))
end

"""
    apply_baffle(blocks::CBlocks, knobs) -> (blocks, plan)

The driver's step (and the app's): with PB_BAFFLE empty, `(blocks, nothing)` (nothing changes); otherwise the plan of
`baffle_plan` (the quality gate on) and the blocks to build on (`plan.blocks`). Build with `build_faces(blocks; baffle=plan)`.
THE MESH'S PROVENANCE: the returned blocks' `hashes` are the generated mesh's (as for every CBlocks); the nodes the faces are
built on hash to `node_hashes(blocks)`, which PB BAFFLE prints as x_sha256 and y_sha256: a caller recording the mesh records those.
"""
function apply_baffle(b::CBlocks, vals::AbstractDict)
    spec = baffle_spec(vals)
    spec === nothing && return (b, nothing)
    plan = baffle_plan(b, spec)
    rs = refine_spec(vals)
    rs === nothing || (plan = baffle_refine(plan, rs))
    (plan.blocks, plan)
end

"""
    baffle_receipt_line(plan) -> String

PB BAFFLE: the request, the chord, and per plate its corner column, faces, the requested, grid (the nearest node on the line as
the mesh had it) and realised heights, the deviations from the chord's normal before and after, whether it was straightened
and snapped, the largest node displacement and the window, and its quality (theta_b_max and its row, the wall-normal distance
range d1 of its faces, the tip face over h, the growth-ratio change along its line); then the nodes moved, the smallest corner
angle near them before and after, the largest first-cell height change of a moved column, the gate's result, and the hashes of
the C block's nodes the faces are built on (the moved nodes: the mesh's provenance; the generator's stay in MESH COORDINATE
RECEIPT and PB MESH REMAP).
"""
function baffle_receipt_line(plan::BafflePlan)
    s = plan.spec; c = plan.chord
    io = IOBuffer()
    @printf(io, "PB BAFFLE: version=PBBaffleGurneyV3 spec=%s chord_le=%.9g,%.9g chord_dir=%.9g,%.9g plates=%d", spec_string(s),
            c[1], c[2], c[3], c[4], length(plan.plates))
    for q in plan.plates
        @printf(io, " | plate%d=%s column=%d faces=%d h=%.9g h_grid=%.9g h_real=%.9g dev_grid_max=%.4f dev_real_max=%.3e straightened=%s snapped=%s disp_max=%.4e window=%.4e window_columns=%d tip=%.9g,%.9g theta_b_max=%.3f theta_b_row=%d d1=%.4e..%.4e tip_fraction=%.4f growth_kink=%.4f",
                q.plate, q.side, q.column, q.faces, q.h, q.h_grid, q.h_real, q.dev_grid, q.dev_real, q.straightened ? "yes" : "no",
                q.snapped ? "yes" : "no", q.disp_max, q.window, q.window_columns, q.tip[1], q.tip[2], q.theta_b_max, q.theta_b_row,
                q.d1_min, q.d1_max, q.tip_fraction, q.growth_kink)
    end
    isempty(plan.plates) && print(io, " (h = 0: no plate, no face cut; the mesh is the one without a baffle)")
    @printf(io, " | nodes_moved=%d disp_max=%.4e", plan.moved, plan.disp_max)
    plan.moved > 0 && @printf(io, " angle_min_near_moved=%.3f->%.3f", plan.angle_min_before, plan.angle_min_after)
    @printf(io, " first_cell_change=%.4f gate=%s", plan.first_cell_change,
            !plan.checked ? "unchecked" : isempty(baffle_gate_failures(plan)) ? "pass" : "FAIL")
    @printf(io, "(theta_b<=%g,tip_fraction<=%g,first_cell<=%g)", BAFFLE_THETA_B_MAX, BAFFLE_TIP_FRACTION_MAX, BAFFLE_FIRST_CELL_MAX)
    h = node_hashes(plan.blocks)
    print(io, " x_sha256=", h[1], " y_sha256=", h[2])
    String(take!(io))
end

# ---- the outputs: the plate faces and the plates' share of the loads -----------------------------------------------------
"""
    baffle_table(sd, mesh) -> NamedTuple of host vectors

The baffle rows of a surface distribution (`surface_distribution`, freeair.jl): per baffle face its boundary face index, plate
(1 lower, 2 upper on a C-mesh: the sign of bsta), side ("front": the side facing upstream, tag 13; "back": tag 14), row along
the plate from the root, centroid, length, C_p, C_f and the viscous force over q (cfx, cfy), and the face's area vector (out of
the fluid, into the plate). Ordered by plate, side (front first) and row.
"""
function baffle_table(sd, m::PBMesh)
    bst = _host(m.bsta); bSx = _host(m.bSx); bSy = _host(m.bSy)
    q = findall(t -> t == Int(BT_BAFFLE_FRONT) || t == Int(BT_BAFFLE_BACK), sd.tag)
    plate = [bst[sd.face[k]] > 0 ? 1 : 2 for k in q]; side = [sd.tag[k] == Int(BT_BAFFLE_FRONT) ? "front" : "back" for k in q]
    row = [abs(bst[sd.face[k]]) for k in q]
    o = sortperm(collect(zip(plate, side .== "back", row)))
    q = q[o]; plate = plate[o]; side = side[o]; row = row[o]
    (face=sd.face[q], plate=plate, side=side, row=Int.(row), x=sd.x[q], y=sd.y[q], ds=sd.ds[q], Cp=sd.Cp[q], Cf=sd.Cf[q],
     cfx=sd.cfx[q], cfy=sd.cfy[q], Sx=bSx[sd.face[q]], Sy=bSy[sd.face[q]])
end

"Write a `baffle_table` as a CSV: face, plate, side, row, x, y, ds, Cp, Cf, cfx, cfy."
function write_baffle_csv(path::AbstractString, t)
    open(path, "w") do io
        println(io, "face,plate,side,row,x,y,ds,Cp,Cf,cfx,cfy")
        for k in eachindex(t.x)
            println(io, join((t.face[k], t.plate[k] == 1 ? "lower" : "upper", t.side[k], t.row[k], t.x[k], t.y[k], t.ds[k],
                              t.Cp[k], t.Cf[k], t.cfx[k], t.cfy[k]), ","))
        end
    end
    path
end

"""
    baffle_loads(sd, mesh, fs) -> Vector of NamedTuple

Each plate's share of the loads (both sides; from a surface distribution after `airfoil_loads!`, so the same face pressures and
viscous forces as C_L and C_D): its pressure and friction parts of C_L and C_D (wind axes), and the length-weighted mean C_p
of its front and back faces. On a plate the two sides' p_inf terms cancel, so the pressure part is sum C_p S_b / c.
"""
function baffle_loads(sd, m::PBMesh, fs::FreeStream)
    t = baffle_table(sd, m)
    ca = cosd(fs.alpha); sa = sind(fs.alpha); c = fs.chord
    out = NamedTuple[]
    for p in sort(unique(t.plate))
        q = findall(==(p), t.plate)
        fpx = sum(t.Cp[k]*t.Sx[k] for k in q); fpy = sum(t.Cp[k]*t.Sy[k] for k in q)       # pressure force over q
        fvx = sum(t.cfx[k]*t.ds[k] for k in q); fvy = sum(t.cfy[k]*t.ds[k] for k in q)       # viscous force over q
        lift(x, y) = (-x*sa + y*ca)/c; drag(x, y) = (x*ca + y*sa)/c
        mean_cp(sname) = (r = [k for k in q if t.side[k] == sname]; isempty(r) ? NaN : sum(t.Cp[k]*t.ds[k] for k in r)/sum(t.ds[k] for k in r))
        push!(out, (plate=p, side=p == 1 ? :lower : :upper, faces=length(q), dCL=lift(fpx + fvx, fpy + fvy), dCD=drag(fpx + fvx, fpy + fvy),
                    dCLp=lift(fpx, fpy), dCDp=drag(fpx, fpy), dCLf=lift(fvx, fvy), dCDf=drag(fvx, fvy),
                    Cp_front=mean_cp("front"), Cp_back=mean_cp("back")))
    end
    out
end

"One line of the plates' loads (the driver prints it after a free-air run on a mesh with a baffle)."
function baffle_loads_line(sd, m::PBMesh, fs::FreeStream)
    io = IOBuffer(); print(io, "PB BAFFLE LOADS:")
    for r in baffle_loads(sd, m, fs)
        @printf(io, " plate%d=%s faces=%d dCL=%.6f dCD=%.6f dCDp=%.6f dCDf=%.6f Cp_front_mean=%.5f Cp_back_mean=%.5f;", r.plate,
                r.side, r.faces, r.dCL, r.dCD, r.dCDp, r.dCDf, r.Cp_front, r.Cp_back)
    end
    String(take!(io))
end

"""
    freeair_baffle_mesh(call, spec; level=0, backend=CPU(), label="pbcav", remap=nothing) -> (gen, blocks, mesh, plan)

`freeair_mesh` with a baffle: the generator, the remap, the subdivision and the far-field tags as there, then `baffle_plan`
on those blocks and the faces built on the plan's blocks (on the CPU, moved to `backend`).
"""
function freeair_baffle_mesh(call::CMeshCall, spec::BaffleSpec; level::Integer=0, backend=CPU(), label::AbstractString="pbcav",
                             remap=nothing)
    gen = generate_v1(call)
    b0 = cblocks(gen; label)
    remap === nothing || (b0 = remap_columns(b0, remap).blocks)
    b = freeair_blocks(subdivide(b0, level))
    plan = baffle_plan(b, spec)
    (gen, plan.blocks, to_backend(backend, build_faces(plan.blocks; backend=CPU(), baffle=plan)), plan)
end

# =====================================================================================================================
# FREE-EDGE REFINEMENT of the Gurney plate (30 Oct 2026; branch pbcav-flapref; PB_BAFFLE_REFINE, OPT-IN: empty, the default,
# reaches nothing below and every baffle mesh is bit for bit the one before).
#
# THE DEFECT (measured on the app's demo mesh, NACA 0012 native base, h/c 0.01 lower plate: 24 faces on the plate). The plate's
# faces grow from 3e-5 c at the root to 1.54e-3 c at the TIP, the plate's free edge (tip_fraction 0.154), and the grid rows
# through and above the tip, which carry the shear layer that leaves the tip, stay 1.5e-3 to 3e-3 c apart a long way downstream:
# the layer is resolved by one or two cells. The plate's flat sides are already resolved across (the columns fan into the plate,
# d1 7e-5 to 1.3e-4 c).
#
# THE CONSTRUCTION. Rows are added around the tip row K of the C block, in every column, by a monotone re-sampling of each
# column's own curve (PCHIP of the column's nodes over the row index, as mesh_remap.jl places nodes): column i's new node s
# sits at the old fractional index t_i(s), s(t) = 1 + integral of the density f_i, the number of new nodes per old row.
#   * AT THE TIP the density is q (the cells are 1/q of the mesh's own), and it falls back to 1 (the mesh's own cells) as a
#     CONSTANT GROWTH RATIO rho of the new cells: with a = D0/f, D0 the column's own cell length (log-linear in the row index),
#     d ln f / du = sigma lambda0(t) - ln(rho) f, lambda0 = d ln D0 / dt, u the index distance from the tip (sigma = +1 above
#     it, -1 below it, along the plate), integrated by RK4 until f reaches 1. A density bump instead (the first idea) is steepest
#     where the mesh already grows by 1.17 per row and gave transition ratios of 1.46 at q 3; the constant-ratio form needs
#     2-3 rows above and 1-2 below the tip for q 3 at ratio 1.25.
#   * THE TIP IS EXACT: the plate column's two ratios (above and below the tip) are solved by bisection so that the rows
#     added on each side are INTEGERS (e_d below, e_u above), so that the tip node, an old node, is a new node: the plate
#     keeps its exact height, its straightness and its tip.
#   * WHERE: the strength w_i in [0, 1] multiplies (f - 1): 1 on the plate column and for `wake` chords along the wake leg
#     downstream of it (default 30 h, at least 0.15 c) and `foil` plate heights along the foil upstream (default 2), then a cosine
#     fade over `wake_taper` (default three times the wake length) and `foil_taper` (default 4 h); 0 elsewhere. The fade is in
#     the column index, so the rows (grid lines) stay smooth. The row through the tip row K is the wake's own shear layer's
#     grid line (the wake columns' rows merge into the reference distribution at the same row index).
#   * EVERY COLUMN HAS THE SAME NUMBER OF NEW ROWS (a structured block): a column the refinement fades out of receives its share
#     of E = e_d + e_u as an extra density over a common zone around the tip row: sin^2 over REFINE_SHARE_DOWN rows below and
#     REFINE_SHARE_UP above (at least the plate column's own transition), and, far down the wake, a wider band that closes
#     REFINE_FAR_UP rows above the tip; the extra density is E - w_i E_i times the zone's unit-integral shape. Every column then
#     shifts the rows above the zone by exactly E, so the outer rows of neighbouring columns stay aligned. (The first form spread
#     the share over all the outer rows: the index shift then differed between neighbouring columns where the rows are 0.1 to 1 c
#     tall and the columns 0.2 c wide, and the cells sheared to 18-28 degrees. Rows around the tip are 0.004 to 0.05 c tall, so
#     the compact zone and the wide band, which end below the tall rows, shear nothing.)
#   * ROOT / RECIRCULATION TRANSFER: a zero-integral pair of smooth density lobes removes rows from the inherited clustering
#     next to the plate root and the finite-TE wake leg and places them across the middle and upper plate height. This transfer
#     fades to zero by 0.3 c from the plate on the demo mesh, before long wake columns can become skewed. Three further
#     rows are added through the middle and upper plate height, following the same wake. The tip stays
#     an old node. Rows 1..keep (default 3) of every C column, the wall and wake legs (row 1), and the outer boundary are the
#     plan's nodes bit for bit. The foil retains its first three wall cells. The finite-TE H block is coarsened from 26 to 20
#     cross rows on the app's demo mesh: its interface curves and first cross cells at both edges stay exact.
# THE ADMISSION (each a check on the refined blocks against the plan's): every cell strictly convex; the smallest corner angle
#   at most REFINE_ANGLE_DROP degrees below the plan's; the
#   largest ratio of consecutive cells along a column at most max(REFINE_RN_MAX, 1.02 x the plan's); the largest cell aspect ratio
#   at most REFINE_AR_FACTOR times the plan's; the plate's quality gate; face non-orthogonality at most 65 degrees or five
#   degrees above the native baffle mesh, whichever is greater. A mesh that fails is
#   not used: the request is retried at a weaker q (1 + 0.7, 0.5, 0.35 of q - 1), and when none passes the plan is returned
#   UNREFINED with the reasons in its record (strict=yes refuses instead). Host setup, once per mesh.
# =====================================================================================================================
const REFINE_TABLE = 64                    # table points per row index in the density integrals
const REFINE_CAP_ROWS = 16                 # a transition longer than this is refused
const REFINE_SHARE_DOWN = 6.0              # rows below / above the tip over which a column the refinement fades out of takes its share
const REFINE_SHARE_UP = 10.0               # of the added rows (at least E below and 2 E above, and the plate column's own transition + 1)
const REFINE_RN_MAX = 1.50
const REFINE_FAR_UP = 25.0                 # the far wake's share is spread over the rows from `keep` to this many rows above the tip
const REFINE_AR_FACTOR = 1.5
const REFINE_ANGLE_DROP = 2.0
const REFINE_FACE_THETA_MAX = 65.0         # diagnostic non-orthogonality target; allow up to five degrees over a native mesh
const REFINE_FACE_THETA_RISE = 5.0
const REFINE_ROOT_TRANSFER = 0.30          # transfer row density from the root/TE wake leg to the middle of the plate height
const REFINE_MID_ROWS = 3                 # additional rows across the flap-height recirculation band
const REFINE_STRIP_ROWS = 20              # cap cross rows in the finite-TE H block; keep its two edge spacings

"""
    FlapRefineSpec(; q=3, ratio=1.25, wake=0, wake_taper=0, foil=2, foil_taper=4, keep=3, strict=false)

The request of PB_BAFFLE_REFINE (knobs.jl `_parse_baffle_refine`): the refinement factor `q` at the tip, the growth `ratio` of
the transition, the distances (`wake` and `wake_taper` in chords, 0 = automatic; `foil` and `foil_taper` in plate heights) over
which the refinement is at full strength and then fades, the `keep` rows next to the wall never touched, and `strict`.
"""
struct FlapRefineSpec
    q::Float64; ratio::Float64; wake::Float64; wake_taper::Float64; foil::Float64; foil_taper::Float64; keep::Int; strict::Bool
    function FlapRefineSpec(q::Real, ratio::Real, wake::Real, wake_taper::Real, foil::Real, foil_taper::Real, keep::Integer, strict::Bool)
        1 < q <= 8 || throw(ArgumentError("FlapRefineSpec: q must lie in (1, 8] (got $q)"))
        1.05 <= ratio <= 1.6 || throw(ArgumentError("FlapRefineSpec: ratio must lie in [1.05, 1.6] (got $ratio)"))
        wake >= 0 && wake_taper >= 0 && foil >= 0 && foil_taper > 0 || throw(ArgumentError("FlapRefineSpec: distances must be non-negative (taper lengths positive)"))
        3 <= keep <= 40 || throw(ArgumentError("FlapRefineSpec: keep must lie in [3, 40] rows"))
        new(Float64(q), Float64(ratio), Float64(wake), Float64(wake_taper), Float64(foil), Float64(foil_taper), Int(keep), strict)
    end
end
FlapRefineSpec(; q::Real=3.0, ratio::Real=1.25, wake::Real=0.0, wake_taper::Real=0.0, foil::Real=2.0, foil_taper::Real=4.0,
               keep::Integer=3, strict::Bool=false) = FlapRefineSpec(q, ratio, wake, wake_taper, foil, foil_taper, keep, strict)

"The knob's spec string of a FlapRefineSpec (the receipt's form; `_parse_baffle_refine` reads it back to the same spec)."
refine_spec_string(s::FlapRefineSpec) = string("tip:q=", s.q, ",ratio=", s.ratio, ",wake=", s.wake == 0 ? "auto" : s.wake,
    ",wake_taper=", s.wake_taper == 0 ? "auto" : s.wake_taper, ",foil=", s.foil, ",foil_taper=", s.foil_taper, ",keep=", s.keep,
    ",strict=", s.strict ? "yes" : "no")

"""
    refine_spec(knobs) -> FlapRefineSpec or nothing

The request of PB_BAFFLE_REFINE (read_knobs's parsed value), or nothing when it is empty (the default).
"""
function refine_spec(vals::AbstractDict)
    v = get(vals, "PB_BAFFLE_REFINE", nothing)
    v === nothing && return nothing
    FlapRefineSpec(; q=v.q, ratio=v.ratio, wake=v.wake, wake_taper=v.wake_taper, foil=v.foil, foil_taper=v.foil_taper, keep=v.keep, strict=v.strict)
end

"A column's log cell lengths ln L_k (k = 1 .. NJ), the profile the transition's ODE reads."
_refine_profile(X, Y, i) = [log(hypot(X[i, k+1] - X[i, k], Y[i, k+1] - Y[i, k])) for k in 1:size(X, 2)-1]

"d ln L / dt of the log-linear profile (knots at the interval midpoints k + 1/2): constant between midpoints, 0 outside them."
function _lam0(p::Vector{Float64}, t::Float64)
    n = length(p)
    (t < 1.5 || t >= n + 0.5) && return 0.0
    k = clamp(floor(Int, t - 0.5), 1, n - 1)
    p[k+1] - p[k]
end

"""
    _half_zone(p, K, q, rho, sigma, keep) -> (f, E, ok, reason)

One side of the transition from the tip row K: the density f at the table points u = 0, 1/T, 2/T, ... of the index distance from
the tip (sigma = +1 above the tip, -1 below), f(0) = q, d ln f / du = sigma lambda0(K + sigma u) - ln(rho) f by RK4, ending where f
reaches 1, and E the integral of f - 1 (trapezoid on the table): the rows this side adds. Not ok when f does not return to 1
within REFINE_CAP_ROWS rows or the zone would reach row `keep` below or the outer rows above.
"""
function _half_zone(p::Vector{Float64}, K::Int, q::Float64, rho::Float64, sigma::Int, keep::Int)
    NJ = length(p); T = REFINE_TABLE; du = 1.0/T; lr = log(rho)
    F(t, y) = sigma*_lam0(p, t) - lr*exp(y)
    f = Float64[q]; E = 0.0; y = log(q)
    for n in 1:REFINE_CAP_ROWS*T
        t = K + sigma*(n - 1)*du
        tn = t + sigma*du
        sigma < 0 && tn < keep + 1.0 && return (f, E, false, "the transition below the tip would reach row $(keep + 1) (rows 1..$keep are kept)")
        sigma > 0 && tn > NJ - 2.0 && return (f, E, false, "the transition above the tip would reach the outer rows")
        k1 = F(t, y); k2 = F(t + sigma*du/2, y + du/2*k1); k3 = F(t + sigma*du/2, y + du/2*k2); k4 = F(tn, y + du*k3)
        yn = y + du/6*(k1 + 2k2 + 2k3 + k4)
        fp = f[end]
        if yn <= 0
            E += 0.5*(fp - 1)*du; push!(f, 1.0)
            return (f, E, true, "")
        end
        fn = exp(yn); E += 0.5*((fp - 1) + (fn - 1))*du; push!(f, fn); y = yn
    end
    (f, E, false, "the density does not return to the mesh's own within $(REFINE_CAP_ROWS) rows (ratio too close to the mesh's own growth)")
end

"""
    _solve_ratio(p, K, q, sigma, keep, target) -> (rho, ok, reason)

The growth ratio at which one side of the transition adds exactly `target` rows (E is continuous and decreasing in the ratio;
a transition that does not close counts as infinite), by bisection in ln(rho).
"""
function _solve_ratio(p::Vector{Float64}, K::Int, q::Float64, sigma::Int, keep::Int, target::Float64)
    Eof(r) = (z = _half_zone(p, K, q, r, sigma, keep); z[3] ? z[2] : Inf)
    lo = log(1.02); hi = log(8.0)
    Eof(exp(hi)) <= target || return (NaN, false, "no growth ratio up to 8 adds so few as $(target) rows $(sigma > 0 ? "above" : "below") the tip")
    for _ in 1:100
        mid = 0.5*(lo + hi)
        Eof(exp(mid)) > target ? (lo = mid) : (hi = mid)
    end
    (exp(0.5*(lo + hi)), true, "")
end

"The cosine fade: 1 up to `full`, 0 from full + taper."
_refine_fade(d, full, taper) = d <= full ? 1.0 : d >= full + taper ? 0.0 : 0.5*(1 + cos(pi*(d - full)/taper))

"""A zero-integral row-density transfer below a plate tip. The lower half of the available span loses density and the
upper half gains the same amount. Its two sin² lobes close smoothly at the kept wall rows, their midpoint, and the tip;
the exact tip index and the total row count therefore remain unchanged. Its weight fades to zero near the flap so longer
wake columns do not shear."""
function _refine_root_transfer(t::Float64, keep::Int, K::Int)
    a = keep + 1.0; z = K - 1.0
    z - a >= 4 || return 0.0
    (a < t < z) || return 0.0
    u = (t - a)/(z - a)
    REFINE_ROOT_TRANSFER * (u < 0.5 ? -1.0 : 1.0) * sinpi(2u)^2
end

"""
    _refine_weights(b, plan, spec) -> (w, plate_of)

The strength of the refinement in each C column and the plate (1 lower, 2 upper; 0 none) whose tip row centres it: the fade of
each plate along the wall row (the foil on one side of the plate's column, the wake leg on the other), the larger of the two
plates' where they meet.
"""
function _refine_weights(b::CBlocks, plan::BafflePlan, spec::FlapRefineSpec)
    NI = b.NI; X = b.X; Y = b.Y
    seg = [hypot(X[i+1, 1] - X[i, 1], Y[i+1, 1] - Y[i, 1]) for i in 1:NI]
    w = zeros(NI + 1); owner = zeros(Int, NI + 1)
    for q in plan.plates
        i0 = q.column; hr = q.h_real
        wf = spec.wake > 0 ? spec.wake : clamp(30hr, 0.09, 0.30); wt = spec.wake_taper > 0 ? spec.wake_taper : wf*clamp(hr/0.01, 0.8, 1.0)
        ff = spec.foil*hr; ft = spec.foil_taper*hr
        wake_up = q.plate == 2           # the upper plate's wake is the columns above its own; the lower plate's, those below
        d = 0.0
        for i in i0:NI+1                 # toward the higher column index
            i > i0 && (d += seg[i-1])
            wi = wake_up ? _refine_fade(d, wf, wt) : _refine_fade(d, ff, ft)
            wi > w[i] && (w[i] = wi; owner[i] = q.plate)
        end
        d = 0.0
        for i in i0:-1:1                 # toward the lower column index
            i < i0 && (d += seg[i])
            wi = wake_up ? _refine_fade(d, ff, ft) : _refine_fade(d, wf, wt)
            wi > w[i] && (w[i] = wi; owner[i] = q.plate)
        end
    end
    w, owner
end

"The statistics the admission compares: the smallest corner cross product, the smallest corner angle, the largest ratio of consecutive cells along a column and the largest cell aspect ratio (longest over shortest edge)."
function _refine_stats(X, Y)
    NI, NJ = size(X) .- 1
    cm, at = _min_corner_cross(X, Y)
    am = _min_corner_angle(X, Y, Iterators.product(1:NI, 1:NJ))
    rn = 1.0; ar = 1.0
    for i in 1:NI+1
        prev = hypot(X[i, 2] - X[i, 1], Y[i, 2] - Y[i, 1])
        for j in 2:NJ
            cur = hypot(X[i, j+1] - X[i, j], Y[i, j+1] - Y[i, j]); rn = max(rn, cur/prev, prev/cur); prev = cur
        end
    end
    for i in 1:NI, j in 1:NJ
        l1 = hypot(X[i+1, j] - X[i, j], Y[i+1, j] - Y[i, j]); l2 = hypot(X[i+1, j+1] - X[i+1, j], Y[i+1, j+1] - Y[i+1, j])
        l3 = hypot(X[i+1, j+1] - X[i, j+1], Y[i+1, j+1] - Y[i, j+1]); l4 = hypot(X[i, j+1] - X[i, j], Y[i, j+1] - Y[i, j])
        ar = max(ar, max(l1, l2, l3, l4)/min(l1, l2, l3, l4))
    end
    (cross_min=cm, cross_at=at, angle_min=am, rn_max=rn, ar_max=ar)
end

"""
    _refine_try(plan, spec, q) -> NamedTuple

One refinement at the tip factor `q`: the integer row counts and ratios from the plate columns, every column's re-sampling, the
refined blocks and plan, and the admission. `ok` false (with `reason`) when a transition does not close or a check fails.
"""
function _refine_try(plan::BafflePlan, spec::FlapRefineSpec, q::Float64)
    b = plan.blocks; NI, NJ = b.NI, b.NJ; X0 = b.X; Y0 = b.Y; keep = spec.keep
    fail(r) = (ok=false, reason=r)
    isempty(plan.plates) && return fail("the plan has no plate")
    # 1. the integer rows each plate adds below and above its tip, from the first plate at the requested ratio, and each plate's ratios
    ed = 0; eu = 0; rho_d = Dict{Int,Float64}(); rho_u = Dict{Int,Float64}()
    for (n, pl) in enumerate(plan.plates)
        p = _refine_profile(X0, Y0, pl.column)
        if n == 1
            zd = _half_zone(p, pl.tip_row, q, spec.ratio, -1, keep); zu = _half_zone(p, pl.tip_row, q, spec.ratio, +1, keep)
            zd[3] || return fail("plate $(pl.plate) ($(pl.side)): $(zd[4])"); zu[3] || return fail("plate $(pl.plate) ($(pl.side)): $(zu[4])")
            ed = max(1, ceil(Int, zd[2] - 1e-9)); eu = max(1, ceil(Int, zu[2] - 1e-9))     # rounded UP: a ratio no steeper than the request's
        end
        rd = _solve_ratio(p, pl.tip_row, q, -1, keep, Float64(ed)); rd[2] || return fail("plate $(pl.plate) ($(pl.side)): $(rd[3])")
        ru = _solve_ratio(p, pl.tip_row, q, +1, keep, Float64(eu)); ru[2] || return fail("plate $(pl.plate) ($(pl.side)): $(ru[3])")
        rho_d[pl.plate] = rd[1]; rho_u[pl.plate] = ru[1]
    end
    ed += REFINE_MID_ROWS                 # these rows lie below the tip, across the recirculation band
    E = ed + eu; NJ2 = NJ + E
    # 2. the strengths, the table shared by every column, and each plate's zone: the rows its transition covers at the plate's own
    #    column. A column the refinement fades out of puts its share of the E rows UNIFORMLY into the SAME zone (a plateau with
    #    one-row smooth edges), so that every column shifts the rows above the zone by exactly E and the outer rows of neighbouring
    #    columns stay aligned (a spread over the outer rows sheared the cells there: the first attempt, 18 degrees).
    w, owner = _refine_weights(b, plan, spec)
    T = REFINE_TABLE; du = 1.0/T; M = NJ*T + 1
    tg = [1.0 + (m - 1)*du for m in 1:M]
    zones = Dict{Int,NamedTuple}()
    for pl in plan.plates
        p = _refine_profile(X0, Y0, pl.column)
        zd = _half_zone(p, pl.tip_row, q, rho_d[pl.plate], -1, keep); zu = _half_zone(p, pl.tip_row, q, rho_u[pl.plate], +1, keep)
        ze0 = pl.tip_row - (length(zd[1]) - 1)*du; ze1 = pl.tip_row + (length(zu[1]) - 1)*du    # the plate column's own transition
        sdn = max(REFINE_SHARE_DOWN, 1.0*E); sup = max(REFINE_SHARE_UP, 2.0*E)      # a band that widens with the rows to share: its edges' ratio stays small
        z0 = max(keep + 1.0, min(ze0 - 1.0, pl.tip_row - sdn)); z1 = min(NJ - 2.0, max(ze1 + 1.0, pl.tip_row + sup))
        hz = [(z0 < t < z1) ? sinpi((t - z0)/(z1 - z0))^2 : 0.0 for t in tg]                       # a smooth bump: no ratio jump at its edges
        Hz = sum(0.5*(hz[m] + hz[m+1])*du for m in 1:M-1)
        zg1 = min(NJ - 2.0, pl.tip_row + max(REFINE_FAR_UP, sup + 9.0))
        gz = [(keep + 1.0 < t < zg1) ? sinpi((t - keep - 1.0)/(zg1 - keep - 1.0))^2 : 0.0 for t in tg]   # the far wake's share: a wide bump, closed well below the tall outer rows
        Gz = sum(0.5*(gz[m] + gz[m+1])*du for m in 1:M-1)
        tr = [_refine_root_transfer(t, keep, pl.tip_row) for t in tg]
        m0 = max(keep + 2.0, pl.tip_row - 7.0); m1 = pl.tip_row - 1.0
        m1 - m0 >= 2 || return fail("plate $(pl.plate): insufficient rows across the middle of the plate")
        mid = [(m0 < t < m1) ? sinpi((t - m0)/(m1 - m0))^2 : 0.0 for t in tg]
        midint = sum(0.5*(mid[m] + mid[m+1])*du for m in 1:M-1)
        zones[pl.plate] = (z0=z0, z1=z1, h=hz./Hz, g=gz./Gz, transfer=tr, mid=mid.*(REFINE_MID_ROWS/midint))
    end
    # sigma_i: how far a wake column has left the plate's neighbourhood. Where the columns are long compared with the height of the
    # rows around the tip (the far wake: 0.04 to 0.3 c long, the rows 0.004 to 0.05 c) the share of the added rows is spread over a
    # wide band (a change of the rows' density of E/40, so the outlet's aspect ratio rises by about 15 %); the index shift that
    # differs between neighbouring columns there shears nothing. The band closes REFINE_FAR_UP rows above the tip: spread over all
    # the outer rows (the first form) it left a shift mismatch where the rows are 0.1 to 1 c tall and sheared them to 28 degrees.
    # Near the plate and on the foil the columns are narrow and the rows tall and the share stays in the compact zone.
    sig = zeros(NI + 1)
    let d = 0.0, wf = max(30*maximum(pl.h_real for pl in plan.plates), 0.15)
        d0 = 0.05; d1 = max(0.25, wf)
        for i in b.NIW:-1:1; d += hypot(X0[i+1, 1] - X0[i, 1], Y0[i+1, 1] - Y0[i, 1]); sig[i] = _smoothstep((d - d0)/(d1 - d0)); end
        d = 0.0
        for i in b.NIW+b.NIF+2:NI+1; d += hypot(X0[i, 1] - X0[i-1, 1], Y0[i, 1] - Y0[i-1, 1]); sig[i] = _smoothstep((d - d0)/(d1 - d0)); end
    end
    # the plate a column belongs to: its owner where the refinement acts, else the nearest plate's
    zone_of = [owner[i] > 0 ? owner[i] : plan.plates[argmin([abs(i - pl.column) for pl in plan.plates])].plate for i in 1:NI+1]
    X2 = Matrix{Float64}(undef, NI + 1, NJ2 + 1); Y2 = Matrix{Float64}(undef, NI + 1, NJ2 + 1)
    knots = collect(1.0:NJ + 1)
    plate_of = Dict(pl.plate => pl for pl in plan.plates)
    sv = Vector{Float64}(undef, M)
    for i in 1:NI+1
        wi = w[i]; fr = ones(M); Eref = 0.0; zn = zones[zone_of[i]]
        pltr = plate_of[zone_of[i]]
        dtr = hypot(X0[i,1] - X0[pltr.column,1], Y0[i,1] - Y0[pltr.column,1])
        trband = clamp(15pltr.h_real, 0.05, 0.15)
        trw = wi*_refine_fade(dtr, trband, trband)
        if wi > 0
            pl = plate_of[owner[i]]; K = pl.tip_row; p = _refine_profile(X0, Y0, i)
            zd = _half_zone(p, K, q, rho_d[pl.plate], -1, keep); zu = _half_zone(p, K, q, rho_u[pl.plate], +1, keep)
            zd[3] || return fail("column $i (strength $(round(wi, digits=3))): $(zd[4])"); zu[3] || return fail("column $i (strength $(round(wi, digits=3))): $(zu[4])")
            gK = (K - 1)*T + 1
            for (n, f) in enumerate(zd[1]); fr[gK - (n - 1)] = f; end
            for (n, f) in enumerate(zu[1]); fr[gK + (n - 1)] = f; end
            fr .+= zn.mid
            Eref = zd[2] + zu[2] + REFINE_MID_ROWS
        end
        c = E - wi*Eref
        sv[1] = 1.0
        for m in 1:M-1
            transfer1 = trw*zn.transfer[m]; transfer2 = trw*zn.transfer[m+1]
            f1 = 1 + wi*(fr[m] - 1) + transfer1 + c*((1 - sig[i])*zn.h[m] + sig[i]*zn.g[m]); f2 = 1 + wi*(fr[m+1] - 1) + transfer2 + c*((1 - sig[i])*zn.h[m+1] + sig[i]*zn.g[m+1])
            (f1 > 0.2 && f2 > 0.2) || return fail("column $i: the row density falls below 0.2 of the mesh's own")
            sv[m+1] = sv[m] + 0.5*(f1 + f2)*du
        end
        abs(sv[M] - (NJ2 + 1)) < 1e-9 || error("_refine_try: column $i closes at $(sv[M]) instead of $(NJ2 + 1) (internal error)")
        tq = Vector{Float64}(undef, NJ2 + 1)
        for s in 1:NJ2+1
            if s == 1; tq[s] = 1.0
            elseif s == NJ2 + 1; tq[s] = Float64(NJ + 1)
            else
                m = clamp(searchsortedlast(sv, Float64(s)), 1, M - 1)
                tq[s] = tg[m] + (s - sv[m])/(sv[m+1] - sv[m])*du
                abs(tq[s] - round(tq[s])) < 1e-9 && (tq[s] = round(tq[s]))
            end
        end
        xs = collect(Float64, view(X0, i, :)); ys = collect(Float64, view(Y0, i, :))
        xn = all(==(xs[1]), xs) ? fill(xs[1], length(tq)) : pchip_eval(knots, xs, tq)
        yn = all(==(ys[1]), ys) ? fill(ys[1], length(tq)) : pchip_eval(knots, ys, tq)
        for s in 1:NJ2+1
            if tq[s] == round(tq[s])               # an old node: copied, not re-evaluated
                X2[i, s] = xs[Int(tq[s])]; Y2[i, s] = ys[Int(tq[s])]
            else
                X2[i, s] = xn[s]; Y2[i, s] = yn[s]
            end
        end
    end
    # 3. coarsen the separate finite-TE H block across its gap. Its lower and upper interface curves and the first
    #    cross cell at each edge remain the original nodes; the graded interior uses fewer rows at every station.
    NS2 = min(b.NS, REFINE_STRIP_ROWS)
    Xs2 = b.Xs; Ys2 = b.Ys
    if NS2 < b.NS
        Xs2 = Matrix{Float64}(undef, b.NIs+1, NS2+1); Ys2 = similar(Xs2)
        for st in 1:b.NIs+1
            xl, yl = b.Xs[st,1], b.Ys[st,1]; xu, yu = b.Xs[st,end], b.Ys[st,end]
            L = hypot(xu-xl, yu-yl)
            el = hypot(b.Xs[st,2]-xl, b.Ys[st,2]-yl)
            eup = hypot(xu-b.Xs[st,end-1], yu-b.Ys[st,end-1])
            t = try
                two_sided(L, NS2, el, eup)[1]
            catch err
                err isa ArgumentError || rethrow()
                return fail("the finite-TE H block cannot use $NS2 cross rows at station $st: $(sprint(showerror, err))")
            end
            for j in 1:NS2+1
                Xs2[st,j] = xl + t[j]*(xu-xl); Ys2[st,j] = yl + t[j]*(yu-yl)
            end
            Xs2[st,1] = xl; Ys2[st,1] = yl; Xs2[st,end] = xu; Ys2[st,end] = yu
            Xs2[st,2] = b.Xs[st,2]; Ys2[st,2] = b.Ys[st,2]
            Xs2[st,end-1] = b.Xs[st,end-1]; Ys2[st,end-1] = b.Ys[st,end-1]
        end
    end
    blocks = CBlocks(b.label * "+refine", NI, NJ2, b.NIW, b.NIF, X2, Y2, b.kind, b.NIs, NS2, Xs2, Ys2, b.level, b.hashes)
    seglen(XX, YY, i, k) = hypot(XX[i, k+1] - XX[i, k], YY[i, k+1] - YY[i, k])
    plates2 = map(plan.plates) do pl
        K2 = pl.tip_row + ed
        hr = (X2[pl.column, K2] - X2[pl.column, 1])*pl.nx + (Y2[pl.column, K2] - Y2[pl.column, 1])*pl.ny
        pq = _plate_quality(X2, Y2, pl.column, K2, hr)
        gk = maximum(abs((seglen(X2, Y2, pl.column, k)/seglen(X2, Y2, pl.column, k-1))/(seglen(X2, Y2, pl.column, k-1)/seglen(X2, Y2, pl.column, max(k-2, 1))) - 1)
                     for k in 3:min(NJ2, K2 + 10))
        merge(pl, (faces=K2 - 1, tip_row=K2, h_real=hr,
                   dev_real=maximum(_seg_dev(X2[pl.column, k] - X2[pl.column, k-1], Y2[pl.column, k] - Y2[pl.column, k-1], pl.nx, pl.ny) for k in 2:K2),
                   tip=(X2[pl.column, K2], Y2[pl.column, K2]), root=(X2[pl.column, 1], Y2[pl.column, 1])), pq, (growth_kink=gk,))
    end
    cuts2 = [BaffleCut(; dir=:i, line=pl.column, lo=1, hi=pl.tip_row - 1, plate=pl.plate, front=pl.plate == 1 ? :plus : :minus) for pl in plates2]
    st0 = _refine_stats(X0, Y0); st1 = _refine_stats(X2, Y2)
    hs0 = b.NS > 0 ? _refine_stats(b.Xs, b.Ys) : nothing
    hs1 = NS2 > 0 ? _refine_stats(Xs2, Ys2) : nothing
    rec = (spec=spec, q_used=q, rows_added=E, rows_below=ed, rows_above=eu, NJ_before=NJ, NJ_after=NJ2,
           ratio_below=Dict(k => v for (k, v) in rho_d), ratio_above=Dict(k => v for (k, v) in rho_u),
           columns_refined=count(>(0), w), columns_full=count(>=(1), w), stats_before=st0, stats_after=st1,
           strip_ns_before=b.NS, strip_ns_after=NS2, strip_stats_before=hs0, strip_stats_after=hs1,
           tip_face_before=[seglen(X0, Y0, pl.column, pl.tip_row - 1) for pl in plan.plates],
           tip_face_after=[seglen(X2, Y2, pl.column, pl.tip_row - 1) for pl in plates2],
           admitted=true, attempts=NamedTuple[], reason="")
    plan2 = BafflePlan(plan.spec, blocks, cuts2, plates2, plan.moved, plan.disp_max, plan.angle_min_before, plan.angle_min_after,
                       plan.first_cell_change, plan.checked, plan.chord, rec)
    why = String[]
    st1.cross_min > 0 || push!(why, "cell $(st1.cross_at) is not strictly convex (corner cross product $(st1.cross_min))")
    st1.angle_min >= st0.angle_min - REFINE_ANGLE_DROP || push!(why, "the smallest corner angle falls from $(round(st0.angle_min, digits=2)) to $(round(st1.angle_min, digits=2)) degrees (limit: $(REFINE_ANGLE_DROP) degrees)")
    st1.rn_max <= max(REFINE_RN_MAX, 1.02*st0.rn_max) || push!(why, "the largest ratio of consecutive cells along a column rises from $(round(st0.rn_max, digits=3)) to $(round(st1.rn_max, digits=3)) (limit $(round(max(REFINE_RN_MAX, 1.02*st0.rn_max), digits=3)))")
    st1.ar_max <= REFINE_AR_FACTOR*st0.ar_max || push!(why, "the largest cell aspect ratio rises from $(round(st0.ar_max, digits=1)) to $(round(st1.ar_max, digits=1)) (limit $(REFINE_AR_FACTOR) times)")
    if NS2 < b.NS
        hs1.cross_min > 0 || push!(why, "finite-TE H cell $(hs1.cross_at) is not strictly convex")
        hs1.angle_min >= hs0.angle_min - REFINE_ANGLE_DROP || push!(why, "the finite-TE H block's smallest corner angle falls from $(round(hs0.angle_min, digits=2)) to $(round(hs1.angle_min, digits=2)) degrees")
        hs1.rn_max <= max(REFINE_RN_MAX, 1.02*hs0.rn_max) || push!(why, "the finite-TE H block's consecutive cell-size ratio rises from $(round(hs0.rn_max, digits=3)) to $(round(hs1.rn_max, digits=3))")
        hs1.ar_max <= REFINE_AR_FACTOR*hs0.ar_max || push!(why, "the finite-TE H block's largest aspect ratio rises from $(round(hs0.ar_max, digits=1)) to $(round(hs1.ar_max, digits=1))")
    end
    plan.checked && append!(why, baffle_gate_failures(plan2))
    if isempty(why)
        th0 = maximum(build_faces(b; backend=CPU(), baffle=plan).theta)
        th1 = maximum(build_faces(blocks; backend=CPU(), baffle=plan2).theta)
        limit = max(REFINE_FACE_THETA_MAX, th0 + REFINE_FACE_THETA_RISE)
        th1 <= limit || push!(why, "the largest face non-orthogonality rises from $(round(th0, digits=2)) to $(round(th1, digits=2)) degrees (limit $(round(limit, digits=2)))")
    end
    isempty(why) || return (ok=false, reason=join(why, "; "), plan=plan2)
    (ok=true, reason="", plan=plan2)
end

"""
    baffle_refine(plan::BafflePlan, spec::FlapRefineSpec) -> BafflePlan

The plan's plate(s) with the rows around each tip and along the shear layer refined (header). Tries the requested q and then weaker
ones (1 + 0.7, 0.5 and 0.35 of q - 1, not below 1.2); the first refined mesh that passes the admission is the result, with its
record in `plan.refine` (the attempts, the rows added, the statistics before and after). When none passes: the plan itself with
`refine` = a record with admitted = false and the reasons, or an ArgumentError when spec.strict.
"""
function baffle_refine(plan::BafflePlan, spec::FlapRefineSpec)
    with_record(p, rec) = BafflePlan(p.spec, p.blocks, p.cuts, p.plates, p.moved, p.disp_max, p.angle_min_before, p.angle_min_after,
                                     p.first_cell_change, p.checked, p.chord, rec)
    isempty(plan.plates) && return with_record(plan, (spec=spec, admitted=false, reason="h = 0: no plate to refine", attempts=NamedTuple[]))
    qs = unique([spec.q; [1 + (spec.q - 1)*f for f in (0.7, 0.5, 0.35)]]); filter!(>=(1.2), qs)
    attempts = NamedTuple[]
    for q in qs
        r = _refine_try(plan, spec, q)
        push!(attempts, (q=q, ok=r.ok, reason=r.reason))
        r.ok && return with_record(r.plan, merge(r.plan.refine, (attempts=attempts,)))
    end
    msg = "PB_BAFFLE_REFINE: the refined mesh fails its admission at every q tried (" * join(["q=$(round(a.q, digits=2)): $(a.reason)" for a in attempts], " | ") * ")"
    spec.strict && throw(ArgumentError(msg))
    with_record(plan, (spec=spec, admitted=false, reason=msg, attempts=attempts))
end

"""
    baffle_refine_receipt_line(plan) -> String

PB BAFFLE REFINE: the request, whether the refinement was admitted (and at which q, after how many attempts), the rows added below
and above each tip (the growth ratios), the number of columns refined, each plate's tip face before and after, the statistics
(smallest corner angle, largest ratio of consecutive cells, largest aspect ratio) before and after, and the new row count; or the
reasons it was not applied. Empty when the plan was made without PB_BAFFLE_REFINE.
"""
function baffle_refine_receipt_line(plan::BafflePlan)
    r = plan.refine
    r === nothing && return ""
    io = IOBuffer()
    @printf(io, "PB BAFFLE REFINE: version=PBBaffleRefineV2 spec=%s", refine_spec_string(r.spec))
    if !r.admitted
        print(io, " admitted=no (the mesh is the plan's, unrefined) reason=", replace(r.reason, r"\s+" => " "))
        return String(take!(io))
    end
    @printf(io, " admitted=yes q_used=%.4g attempts=%d rows_added=%d (below %d, above %d) NJ=%d->%d strip_NS=%d->%d columns_refined=%d columns_full=%d",
            r.q_used, length(r.attempts), r.rows_added, r.rows_below, r.rows_above, r.NJ_before, r.NJ_after, r.strip_ns_before, r.strip_ns_after, r.columns_refined, r.columns_full)
    for (k, pl) in enumerate(plan.plates)
        @printf(io, " | plate%d=%s tip_row=%d tip_face=%.4e->%.4e ratio_below=%.4f ratio_above=%.4f", pl.plate, pl.side, pl.tip_row,
                r.tip_face_before[k], r.tip_face_after[k], r.ratio_below[pl.plate], r.ratio_above[pl.plate])
    end
    s0 = r.stats_before; s1 = r.stats_after
    @printf(io, " | angle_min=%.3f->%.3f rn_max=%.4f->%.4f AR_max=%.1f->%.1f", s0.angle_min, s1.angle_min, s0.rn_max, s1.rn_max, s0.ar_max, s1.ar_max)
    String(take!(io))
end
