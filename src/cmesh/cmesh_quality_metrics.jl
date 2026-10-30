# =====================================================================================================================
# cmesh_quality_metrics.jl -- the mesh-quality metrics and gates of CMeshQualityV1 (15 Oct 2026).
#
# Included by src/cmesh/cmesh_gen.jl (inside module CMeshGen, as the submodule CMeshGen.CMeshQualityMetrics), so the
# generator can gate its own builds (the generator in src/ cannot include a tools/ library: review R7 of
# memory/project_docs/_records/2026-10-15_mesh_quality/review/REVIEW.md). The metric code is the audit's library verbatim
# (memory/project_docs/_records/2026-10-15_mesh_quality/mesh_quality_lib_20261015.jl.txt, commit 4abaea14; its 56 self-tests are carried
# by tools/test_cmesh_mesh_quality.jl), with Statistics replaced by a local median so the generator stays dependency free.
# The summary and the gates are new (implementation record memory/project_docs/_records/2026-10-15_mesh_quality/implement/).
#
# Pure geometry: Float64 from the node arrays, nothing read from a solver. Nothing here is reachable from a mesh built
# without the quality option (generate_cmesh calls it only when `quality !== nothing`); tools may call it on any mesh.
#
# DEFINITIONS (the audit's):
#   Cell (i, j) of a block has the nodes n1 = (i,j), n2 = (i+1,j), n3 = (i+1,j+1), n4 = (i,j+1).
#   area            the polygon (shoelace) area.
#   centroid        the TRUE AREA CENTROID (version 2's); `xv, yv` the vertex average (version 1's).
#   AR_edge         longest / shortest of the four edge lengths.
#   AR_principal    max/min of the two principal lengths L_i = |mid(n2,n3) - mid(n1,n4)|, L_j = |mid(n4,n3) - mid(n1,n2)|.
#   AR_sv           ratio of the singular values of [L_i L_j] (sees shear).
#   min/max angle   the smallest / largest interior corner angle (degrees).
#   scaled Jacobian the smallest corner sine (the generator's corner_sine_min per cell).
#   J ratio         the smallest over the largest of the four corner Jacobians (the taper).
#   Internal face between owner P and neighbour N, nodes a -> b (version 2's orders): S = (y_b - y_a, -(x_b - x_a)),
#   d = x_N - x_P between TRUE centroids;
#   theta           non-orthogonality, the angle between S and d (degrees); theta_v the same with vertex averages (v1).
#   skew            |x_f - x_f'| / |d| (version 2's skew/d), x_f the face midpoint, x_f' = x_P + (1 - w) d,
#                   w = S.(x_N - x_f)/S.d; skew_OF = skew |d| / max(0.2 |d|, |S|/2) (OpenFOAM's normalisation).
#   VR              adjacent-cell area ratio max(A_P/A_N, A_N/A_P).
#   Boundary face:  theta_b, the angle between S (out of the cell) and x_f - x_P.
# =====================================================================================================================
module CMeshQualityMetrics

export quad_metrics, face_metrics, bface_theta, block_cells, FaceRec, mesh_faces, cell_face_max, quality_summary,
       quality_gates, QUALITY_GATE_LIMITS, gate_failures, qmedian

# ---- one cell ---------------------------------------------------------------------------------------------------------
"Geometry and shape of the quadrilateral (x1,y1)..(x4,y4) in the order n1, n2, n3, n4."
function quad_metrics(x1, y1, x2, y2, x3, y3, x4, y4)
    mx = 0.25*(x1 + x2 + x4 + x3); my = 0.25*(y1 + y2 + y4 + y3)
    u1 = x1 - mx; u2 = x2 - mx; u3 = x3 - mx; u4 = x4 - mx
    w1 = y1 - my; w2 = y2 - my; w3 = y3 - my; w4 = y4 - my
    k1 = u1*w2 - u2*w1; k2 = u2*w3 - u3*w2; k3 = u3*w4 - u4*w3; k4 = u4*w1 - u1*w4
    A2 = k1 + k2 + k3 + k4
    cx = (u1 + u2)*k1 + (u2 + u3)*k2 + (u3 + u4)*k3 + (u4 + u1)*k4
    cy = (w1 + w2)*k1 + (w2 + w3)*k2 + (w3 + w4)*k3 + (w4 + w1)*k4
    area = 0.5*A2
    xc = mx + cx/(3*A2); yc = my + cy/(3*A2)
    l1 = hypot(x2-x1, y2-y1); l2 = hypot(x3-x2, y3-y2); l3 = hypot(x4-x3, y4-y3); l4 = hypot(x1-x4, y1-y4)
    ar_edge = max(l1, l2, l3, l4)/min(l1, l2, l3, l4)
    lix = 0.5*(x2 + x3 - x1 - x4); liy = 0.5*(y2 + y3 - y1 - y4)
    ljx = 0.5*(x3 + x4 - x1 - x2); ljy = 0.5*(y3 + y4 - y1 - y2)
    Li = hypot(lix, liy); Lj = hypot(ljx, ljy)
    ar_p = max(Li, Lj)/min(Li, Lj)
    s = lix^2 + liy^2 + ljx^2 + ljy^2; dt = abs(lix*ljy - liy*ljx)
    disc = sqrt(max(s^2 - 4dt^2, 0.0))
    s1 = sqrt(0.5*(s + disc)); s2 = sqrt(max(0.5*(s - disc), 0.0))
    ar_sv = s2 > 0 ? s1/s2 : Inf
    c1 = (x2-x1)*(y4-y1) - (y2-y1)*(x4-x1)
    c2 = (x3-x2)*(y1-y2) - (y3-y2)*(x1-x2)
    c3 = (x4-x3)*(y2-y3) - (y4-y3)*(x2-x3)
    c4 = (x1-x4)*(y3-y4) - (y1-y4)*(x3-x4)
    s_1 = c1/(l1*l4); s_2 = c2/(l2*l1); s_3 = c3/(l3*l2); s_4 = c4/(l4*l3)
    sj = min(s_1, s_2, s_3, s_4)
    cmin = min(c1, c2, c3, c4); cmax = max(c1, c2, c3, c4)
    jratio = cmin/cmax
    ang(ax, ay, bx, by) = acosd(clamp((ax*bx + ay*by)/(hypot(ax, ay)*hypot(bx, by)), -1.0, 1.0))
    a1 = ang(x2-x1, y2-y1, x4-x1, y4-y1); a2 = ang(x3-x2, y3-y2, x1-x2, y1-y2)
    a3 = ang(x4-x3, y4-y3, x2-x3, y2-y3); a4 = ang(x1-x4, y1-y4, x3-x4, y3-y4)
    a1 = c1 < 0 ? 360 - a1 : a1; a2 = c2 < 0 ? 360 - a2 : a2; a3 = c3 < 0 ? 360 - a3 : a3; a4 = c4 < 0 ? 360 - a4 : a4
    (area=area, xc=xc, yc=yc, xv=mx, yv=my, ar_edge=ar_edge, ar_p=ar_p, ar_sv=ar_sv, Li=Li, Lj=Lj,
     minang=min(a1, a2, a3, a4), maxang=max(a1, a2, a3, a4), sj=sj, jratio=jratio, cmin=cmin)
end

"Non-orthogonality (true and vertex-average centres), skew and length of the face a -> b between P and N."
function face_metrics(xP, yP, xN, yN, xa, ya, xb, yb, xPv, yPv, xNv, yNv)
    sx = yb - ya; sy = -(xb - xa); sm = hypot(sx, sy)
    fx = 0.5*(xa + xb); fy = 0.5*(ya + yb)
    dx = xN - xP; dy = yN - yP; dl = hypot(dx, dy)
    sd = sx*dx + sy*dy
    theta = acosd(clamp(sd/(sm*dl), -1.0, 1.0))
    w = (sx*(xN - fx) + sy*(yN - fy))/sd
    px = xP + (1 - w)*dx; py = yP + (1 - w)*dy
    skew = hypot(fx - px, fy - py)/dl
    dvx = xNv - xPv; dvy = yNv - yPv
    theta_v = acosd(clamp((sx*dvx + sy*dvy)/(sm*hypot(dvx, dvy)), -1.0, 1.0))
    wv = (sx*(xNv - fx) + sy*(yNv - fy))/(sx*dvx + sy*dvy)
    skew_v = hypot(fx - (xPv + (1 - wv)*dvx), fy - (yPv + (1 - wv)*dvy))/hypot(dvx, dvy)
    (theta=theta, theta_v=theta_v, skew=skew, sd=sd, smag=sm, dlen=dl, xf=fx, yf=fy, skew_v=skew_v)
end

"Boundary-face non-orthogonality: S (out of the cell) against x_f - x_P."
function bface_theta(xP, yP, xa, ya, xb, yb)
    sx = yb - ya; sy = -(xb - xa)
    fx = 0.5*(xa + xb); fy = 0.5*(ya + yb)
    dx = fx - xP; dy = fy - yP
    sd = sx*dx + sy*dy
    (theta=acosd(clamp(sd/(hypot(sx, sy)*hypot(dx, dy)), -1.0, 1.0)), sd=sd, xf=fx, yf=fy)
end

# ---- a block of cells ------------------------------------------------------------------------------------------------
const CELL_FIELDS = (:area, :xc, :yc, :xv, :yv, :ar_edge, :ar_p, :ar_sv, :Li, :Lj, :minang, :maxang, :sj, :jratio, :cmin)

"Every cell metric of the node block (X, Y) as a NamedTuple of NI x NJ matrices."
function block_cells(X::AbstractMatrix, Y::AbstractMatrix)
    NI, NJ = size(X) .- 1
    out = Dict{Symbol,Matrix{Float64}}(f => zeros(NI, NJ) for f in CELL_FIELDS)
    for j in 1:NJ, i in 1:NI
        q = quad_metrics(X[i,j], Y[i,j], X[i+1,j], Y[i+1,j], X[i+1,j+1], Y[i+1,j+1], X[i,j+1], Y[i,j+1])
        for f in CELL_FIELDS; out[f][i, j] = getfield(q, f); end
    end
    (; (f => out[f] for f in CELL_FIELDS)...)
end

"A face record: kind (:Ci, :Cj, :cut, :ifl, :ifu, :Si, :Sj), owner (block, i, j), neighbour (block, i, j), metrics."
struct FaceRec
    kind::Symbol
    Pb::Int; Pi::Int; Pj::Int
    Nb::Int; Ni::Int; Nj::Int
    theta::Float64; theta_v::Float64; skew::Float64; vr::Float64
    sd::Float64; smag::Float64; dlen::Float64; xf::Float64; yf::Float64
    skew_v::Float64
end

"""
    mesh_faces(blocks, cells; NIW, NIF, strip) -> (faces::Vector{FaceRec}, boundary records)

`blocks` = [(X, Y)] for the C block, plus the strip's (Xs, Ys) when `strip`; `cells` their block_cells. Faces in version
2's order: C i faces (j outer, i inner), C j faces, then the cut (sharp) or the interfaces (station order, lower then
upper), then strip i faces and strip j faces. Boundary records `(kind, block, i, j, theta_b, S.d, x_f, y_f)`, kinds
:foil, :outer, :outlet (C block) and :base, :soutlet (strip).
"""
function mesh_faces(blocks, cells; NIW::Int, NIF::Int, strip::Bool)
    X, Y = blocks[1]
    NI, NJ = size(X) .- 1
    faces = FaceRec[]
    cen(b, i, j) = (cells[b].xc[i,j], cells[b].yc[i,j], cells[b].xv[i,j], cells[b].yv[i,j], cells[b].area[i,j])
    nodeC(i, j) = (X[i,j], Y[i,j])
    function add!(kind, Pb, Pi, Pj, Nb, Ni, Nj, a, b)
        xP, yP, xPv, yPv, AP = cen(Pb, Pi, Pj); xN, yN, xNv, yNv, AN = cen(Nb, Ni, Nj)
        m = face_metrics(xP, yP, xN, yN, a[1], a[2], b[1], b[2], xPv, yPv, xNv, yNv)
        push!(faces, FaceRec(kind, Pb, Pi, Pj, Nb, Ni, Nj, m.theta, m.theta_v, m.skew, max(AP/AN, AN/AP),
                             m.sd, m.smag, m.dlen, m.xf, m.yf, m.skew_v))
    end
    for j in 1:NJ, i in 2:NI; add!(:Ci, 1, i-1, j, 1, i, j, nodeC(i, j), nodeC(i, j+1)); end
    for j in 2:NJ, i in 1:NI; add!(:Cj, 1, i, j-1, 1, i, j, nodeC(i+1, j), nodeC(i, j)); end
    if !strip
        for i in 1:NIW; add!(:cut, 1, NI+1-i, 1, 1, i, 1, nodeC(i+1, 1), nodeC(i, 1)); end
    else
        Xs, Ys = blocks[2]
        NIs, NS = size(Xs) .- 1
        nodeS(st, js) = (Xs[st, js], Ys[st, js])
        for st in 1:NIW
            add!(:ifl, 1, NIW+1-st, 1, 2, st, 1, nodeS(st+1, 1), nodeS(st, 1))
            add!(:ifu, 2, st, NS, 1, NIW+NIF+st, 1, nodeS(st+1, NS+1), nodeS(st, NS+1))
        end
        for js in 1:NS, st in 2:NIs; add!(:Si, 2, st-1, js, 2, st, js, nodeS(st, js), nodeS(st, js+1)); end
        for js in 2:NS, st in 1:NIs; add!(:Sj, 2, st, js-1, 2, st, js, nodeS(st+1, js), nodeS(st, js)); end
    end
    bnd = Tuple{Symbol,Int,Int,Int,Float64,Float64,Float64,Float64}[]
    addb!(kind, b, i, j, a, c) = begin
        r = bface_theta(cells[b].xc[i,j], cells[b].yc[i,j], a[1], a[2], c[1], c[2])
        push!(bnd, (kind, b, i, j, r.theta, r.sd, r.xf, r.yf))
    end
    for i in NIW+1:NIW+NIF; addb!(:foil, 1, i, 1, nodeC(i, 1), nodeC(i+1, 1)); end
    for i in 1:NI; addb!(:outer, 1, i, NJ, nodeC(i+1, NJ+1), nodeC(i, NJ+1)); end
    for j in 1:NJ
        addb!(:outlet, 1, 1, j, nodeC(1, j+1), nodeC(1, j))
        addb!(:outlet, 1, NI, j, nodeC(NI+1, j), nodeC(NI+1, j+1))
    end
    if strip
        Xs, Ys = blocks[2]; NIs, NS = size(Xs) .- 1
        nodeS2(st, js) = (Xs[st, js], Ys[st, js])
        for js in 1:NS; addb!(:base, 2, 1, js, nodeS2(1, js+1), nodeS2(1, js)); end
        for js in 1:NS; addb!(:soutlet, 2, NIs, js, nodeS2(NIs+1, js), nodeS2(NIs+1, js+1)); end
    end
    faces, bnd
end

"Per-cell maxima of the face metrics: theta, theta_v, skew, VR_i, VR_j (cut and interfaces count as j)."
function cell_face_max(faces::Vector{FaceRec}, dims)
    th = [fill(0.0, d) for d in dims]; thv = [fill(0.0, d) for d in dims]; sk = [fill(0.0, d) for d in dims]
    vri = [fill(1.0, d) for d in dims]; vrj = [fill(1.0, d) for d in dims]
    for f in faces
        for (b, i, j) in ((f.Pb, f.Pi, f.Pj), (f.Nb, f.Ni, f.Nj))
            th[b][i,j] = max(th[b][i,j], f.theta); thv[b][i,j] = max(thv[b][i,j], f.theta_v)
            sk[b][i,j] = max(sk[b][i,j], f.skew)
            if f.kind in (:Ci, :Si)
                vri[b][i,j] = max(vri[b][i,j], f.vr)
            else
                vrj[b][i,j] = max(vrj[b][i,j], f.vr)
            end
        end
    end
    (theta=th, theta_v=thv, skew=sk, vr_i=vri, vr_j=vrj)
end

"The median of a vector (the average of the two middle values for an even count; NaN when empty)."
function qmedian(v)
    w = sort!(filter(isfinite, collect(Float64, v)))
    n = length(w); n == 0 && return NaN
    isodd(n) ? w[(n+1)÷2] : 0.5*(w[n÷2] + w[n÷2+1])
end

# ---- the mesh summary (CMeshQualityV1's measured quantities) ---------------------------------------------------------
"""
    quality_summary(X, Y, NIW, NIF; strip=nothing, band_j=40, nb=3) -> NamedTuple

The realised quality of a C-mesh (node arrays X, Y; `strip` a block with node arrays `X, Y`, or nothing). Every gate of
CMeshQualityV1 reads it. Bands: "j <= band_j" the boundary-layer and wake band of the C block; "within nb cells of the
outlet": C columns 1..nb and NI-nb+1..NI and the strip's last nb stations; "within nb cells of the tunnel walls": C rows
NJ-nb+1..NJ.

Fields: cells; ar_max (both blocks), ar_offband (C block, j > band_j), ar_out3, ar_wall3, ar_strip, jr_hi (smallest J ratio
over C cells with AR > 1000; NaN when none); theta_max, theta_p50, n60, n70, n80, theta_out3 (faces with a cell within nb of
the outlet); theta_v_max, ntv30, ntv45; skew_max (version 2's skew/d, all internal faces), skew_cut (the cut faces of a
sharp mesh, 0 on a strip mesh), skew_of_max (OpenFOAM's); vr_max (every internal face), vr_junction (the strip interfaces;
1 without a strip), vr_nonjunction (every other face); minang, sj_min; rn_band / rn_else (the C block's wall-normal node
interval ratios, max of r and 1/r, j <= band_j and beyond); rt_foil, rt_cut (the tangential ratios of row j = 1 over the
foil and along the wake legs), rt_te (the trailing-edge junction ratio: the last foil interval against the first wake
interval, both corners, max of r and 1/r); thb_outlet_max (C and strip outlet faces), thb_wall_p50 / thb_wall_max (the C
block's outer faces); len_min (the shortest principal cell length).
Strip only (NaN without a strip; finalisation of 15 Oct 2026, generator review F1 and quality review F7): rt_base (the base
wall's adjacent cross-interval ratio, max of r and 1/r: the strip's station 1), rt_corner_max / rt_corner_min (at each base
corner the foil's last interval, te_ds, over the base's first interval: G7's ratio on the realised mesh, both corners),
theta_junction / skew_junction (the largest non-orthogonality and skew/d over the strip interface faces).
"""
function quality_summary(X::AbstractMatrix, Y::AbstractMatrix, NIW::Int, NIF::Int; strip=nothing, band_j::Int=40, nb::Int=3)
    has_strip = strip !== nothing
    blocks = has_strip ? [(X, Y), (strip.X, strip.Y)] : [(X, Y)]
    cells = [block_cells(b...) for b in blocks]
    faces, bnd = mesh_faces(blocks, cells; NIW=NIW, NIF=NIF, strip=has_strip)
    NI, NJ = size(X) .- 1
    C = cells[1]; arC = C.ar_p
    ncell = sum(length(c.area) for c in cells)
    ar_str = has_strip ? maximum(cells[2].ar_p) : NaN
    ar_out3 = max(maximum(arC[1:nb, :]), maximum(arC[NI-nb+1:NI, :]), has_strip ? maximum(cells[2].ar_p[end-nb+1:end, :]) : 0.0)
    ar_wall3 = maximum(arC[:, NJ-nb+1:NJ])
    ar_off = band_j < NJ ? maximum(arC[:, band_j+1:end]) : NaN
    hi = [C.jratio[k] for k in eachindex(arC) if arC[k] > 1000]
    jr_hi = isempty(hi) ? NaN : minimum(hi)
    th = [f.theta for f in faces]; thv = [f.theta_v for f in faces]
    near_out(b, i) = b == 1 ? (i <= nb || i >= NI - nb + 1) : (i >= size(strip.X, 1) - nb)
    th_out = maximum(f.theta for f in faces if near_out(f.Pb, f.Pi) || near_out(f.Nb, f.Ni))
    sk = [f.skew for f in faces]
    skof = [f.skew*f.dlen/max(0.2*f.dlen, 0.5*f.smag) for f in faces]
    skcut = maximum((f.skew for f in faces if f.kind === :cut); init=0.0)
    vr_all = maximum(f.vr for f in faces)
    vr_junc = maximum((f.vr for f in faces if f.kind in (:ifl, :ifu)); init=1.0)
    vr_non = maximum((f.vr for f in faces if !(f.kind in (:ifl, :ifu))); init=1.0)
    minang = minimum(minimum(c.minang) for c in cells); sjmin = minimum(minimum(c.sj) for c in cells)
    rnb = 1.0; rne = 1.0
    for i in 1:NI+1, j in 1:NJ-1
        d1 = hypot(X[i,j+1]-X[i,j], Y[i,j+1]-Y[i,j]); d2 = hypot(X[i,j+2]-X[i,j+1], Y[i,j+2]-Y[i,j+1])
        r = max(d2/d1, d1/d2)
        j <= band_j ? (rnb = max(rnb, r)) : (rne = max(rne, r))
    end
    seg = [hypot(X[i+1,1]-X[i,1], Y[i+1,1]-Y[i,1]) for i in 1:NI]
    rtr(rng) = isempty(rng) ? 1.0 : maximum(max(seg[i+1]/seg[i], seg[i]/seg[i+1]) for i in rng)
    rt_foil = rtr(NIW+1:NIW+NIF-1)
    rt_cut = max(rtr(1:NIW-1), rtr(NIW+NIF+1:NI-1))
    rt_te = max(max(seg[NIW]/seg[NIW+1], seg[NIW+1]/seg[NIW]), max(seg[NIW+NIF]/seg[NIW+NIF+1], seg[NIW+NIF+1]/seg[NIW+NIF]))
    tb_wall = [b[5] for b in bnd if b[1] === :outer]
    tb_out = [b[5] for b in bnd if b[1] in (:outlet, :soutlet)]
    len_min = minimum(min(minimum(c.Li), minimum(c.Lj)) for c in cells)
    rt_base = NaN; rc_max = NaN; rc_min = NaN; th_junc = NaN; sk_junc = NaN
    if has_strip
        SX, SY = strip.X, strip.Y; NSb = size(SX, 2) - 1
        db = [hypot(SX[1,j+1]-SX[1,j], SY[1,j+1]-SY[1,j]) for j in 1:NSb]
        rt_base = NSb < 2 ? 1.0 : maximum(max(db[j+1]/db[j], db[j]/db[j+1]) for j in 1:NSb-1)
        rcl = seg[NIW+1]/db[1]; rcu = seg[NIW+NIF]/db[NSb]       # lower corner: foil interval NIW+1; upper: NIW+NIF
        rc_max = max(rcl, rcu); rc_min = min(rcl, rcu)
        th_junc = maximum((f.theta for f in faces if f.kind in (:ifl, :ifu)); init=0.0)
        sk_junc = maximum((f.skew for f in faces if f.kind in (:ifl, :ifu)); init=0.0)
    end
    (cells=ncell, ar_max=max(maximum(arC), has_strip ? ar_str : 0.0), ar_offband=ar_off, ar_out3=ar_out3, ar_wall3=ar_wall3,
     ar_strip=ar_str, jr_hi=jr_hi,
     theta_max=maximum(th), theta_p50=qmedian(th), n60=count(>(60), th), n70=count(>(70), th), n80=count(>(80), th),
     theta_out3=th_out, theta_v_max=maximum(thv), ntv30=count(>(30), thv), ntv45=count(>(45), thv),
     skew_max=maximum(sk), skew_cut=skcut, skew_of_max=maximum(skof),
     vr_max=vr_all, vr_junction=vr_junc, vr_nonjunction=vr_non, minang=minang, sj_min=sjmin,
     rn_band=rnb, rn_else=rne, rt_foil=rt_foil, rt_cut=rt_cut, rt_te=rt_te,
     thb_outlet_max=maximum(tb_out), thb_wall_p50=qmedian(tb_wall), thb_wall_max=maximum(tb_wall), len_min=len_min,
     rt_base=rt_base, rt_corner_max=rc_max, rt_corner_min=rc_min, theta_junction=th_junc, skew_junction=sk_junc)
end
quality_summary(m; kw...) = quality_summary(m.X, m.Y, m.NIW, m.NIF; strip=hasproperty(m, :strip) ? m.strip : nothing, kw...)

# ---- the gates ---------------------------------------------------------------------------------------------------------
"""
The CMeshQualityV1 gate limits (G2, implementation record memory/project_docs/_records/2026-10-15_mesh_quality/implement/README.md,
section 4, where each limit's source and its margin against the adopted variants are stated). Every gate refuses.

    G2a theta_max            <= 65 deg   (version 2's hard limit: OpenFOAM's maxNonOrtho; an external standard)
    G2b theta_out3           <= 20 deg   cavitating presets (strip, sharp)
    G2c ar_out3              <= 1000     cavitating presets; ar_wall3 <= 1000 and jr_hi >= 0.95 wherever AR > 1000 (any)
    G2c-a ar_max             <= 1000     cavitating presets: every cell of both blocks (quality review F2, 15 Oct 2026)
    G2d vr_nonjunction       <= 2.0      every internal face but the strip interfaces; vr_junction <= 2.0 only with Q-D
    G2e rn_band <= 1.15, rn_else <= 1.2
    G2f rt_foil, rt_cut, rt_te <= 1.2
    G2g skew_of_max          <= 4        (OpenFOAM's maxInternalSkewness)
    G2h thb_outlet_max       <= 15 deg
    G2i minang               >= 20 deg
    G2j skew_max (skew/d)    <= 2.0      cavitating presets (version 2's skew/d; review R5)
    G2k rt_base              <= 1.2      strip meshes: the base wall's adjacent ratio, the foil's G2f limit (generator review F1)
    G2l rt_corner_max <= 2.0, G2l' rt_corner_min >= 1.0
                                         strip meshes: G7's corner ratio te_ds / first base interval on the realised mesh,
                                         both corners (generator review F1; a uniform strip meets it through G7')
"""
const QUALITY_GATE_LIMITS = (theta_max=65.0, theta_out3=20.0, ar_out3=1000.0, jr_hi=0.95, vr=2.0, rn_band=1.15,
                             rn_else=1.2, rt=1.2, skew_of_max=4.0, thb_outlet_max=15.0, minang=20.0, skew_max=2.0,
                             ar_max=1000.0, rt_base=1.2, rt_corner_max=2.0, rt_corner_min=1.0)

"""
    quality_gates(s; cavitating, strip_grade) -> Vector of (gate, quantity, value, limit, pass)

Evaluate the G2 gates on the summary `s` (quality_summary). `cavitating`: the strip and sharp presets (G2b, G2c's outlet
band, G2c-a, G2j); `strip_grade`: Q-D on (G2d' then gates the strip interfaces). G2k, G2l and G2l' are evaluated when the
summary carries a strip (a finite `rt_base`).
"""
function quality_gates(s; cavitating::Bool, strip_grade::Bool=false)
    L = QUALITY_GATE_LIMITS
    g = Tuple{String,String,Float64,Float64,Bool}[]
    add!(name, q, v, lim, ok) = push!(g, (name, q, Float64(v), Float64(lim), ok))
    add!("G2a", "theta_max", s.theta_max, L.theta_max, s.theta_max <= L.theta_max)
    cavitating && add!("G2b", "theta_out3", s.theta_out3, L.theta_out3, s.theta_out3 <= L.theta_out3)
    cavitating && add!("G2c", "ar_out3", s.ar_out3, L.ar_out3, s.ar_out3 <= L.ar_out3)
    cavitating && add!("G2c-a", "ar_max", s.ar_max, L.ar_max, s.ar_max <= L.ar_max)
    add!("G2c-w", "ar_wall3", s.ar_wall3, L.ar_out3, s.ar_wall3 <= L.ar_out3)
    add!("G2c'", "jr_hi", isnan(s.jr_hi) ? 1.0 : s.jr_hi, L.jr_hi, isnan(s.jr_hi) || s.jr_hi >= L.jr_hi)
    add!("G2d", "vr_nonjunction", s.vr_nonjunction, L.vr, s.vr_nonjunction <= L.vr)
    strip_grade && add!("G2d'", "vr_junction", s.vr_junction, L.vr, s.vr_junction <= L.vr)
    add!("G2e", "rn_band", s.rn_band, L.rn_band, s.rn_band <= L.rn_band)
    add!("G2e'", "rn_else", s.rn_else, L.rn_else, s.rn_else <= L.rn_else)
    add!("G2f", "rt_foil", s.rt_foil, L.rt, s.rt_foil <= L.rt)
    add!("G2f'", "rt_cut", s.rt_cut, L.rt, s.rt_cut <= L.rt)
    add!("G2f''", "rt_te", s.rt_te, L.rt, s.rt_te <= L.rt)
    add!("G2g", "skew_of_max", s.skew_of_max, L.skew_of_max, s.skew_of_max <= L.skew_of_max)
    add!("G2h", "thb_outlet_max", s.thb_outlet_max, L.thb_outlet_max, s.thb_outlet_max <= L.thb_outlet_max)
    add!("G2i", "minang", s.minang, L.minang, s.minang >= L.minang)
    cavitating && add!("G2j", "skew_max", s.skew_max, L.skew_max, s.skew_max <= L.skew_max)
    if hasproperty(s, :rt_base) && isfinite(s.rt_base)
        add!("G2k", "rt_base", s.rt_base, L.rt_base, s.rt_base <= L.rt_base)
        add!("G2l", "rt_corner_max", s.rt_corner_max, L.rt_corner_max, s.rt_corner_max <= L.rt_corner_max)
        add!("G2l'", "rt_corner_min", s.rt_corner_min, L.rt_corner_min, s.rt_corner_min >= L.rt_corner_min)
    end
    g
end

"The failed gates of `quality_gates`, as one line each (empty when every gate passes)."
gate_failures(g) = [string(n, " ", q, "=", round(v; sigdigits=6), (n in ("G2i", "G2c'", "G2l'") ? " < " : " > "), lim) for (n, q, v, lim, ok) in g if !ok]

end # module CMeshQualityMetrics
