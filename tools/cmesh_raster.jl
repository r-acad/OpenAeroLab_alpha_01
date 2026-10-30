# ============================================================================================
# cmesh_raster.jl -- SHARED curvilinear-C-mesh -> pixel rasteriser for field rendering.
#
# EXTRACTED VERBATIM from tools/cmesh_case9_movie.jl (2026-08-02) so the cavitating movie tool and the
# transonic one cannot drift apart. This campaign has had TWO defects caused by exactly that -- both
# EOS bugs came from 2D and 3D holding separate copies of one formula -- so a second copy of a
# 70-line geometric kernel is not something to create casually.
#
# WHAT IT DOES, and why a plain heatmap is wrong here. A C-mesh is CURVILINEAR: `heatmap(x, y, f)`
# assumes a rectilinear grid and would shear the field. Instead, per output pixel:
#   1. find the containing CELL by bounding-box fill (`build_pixmap`);
#   2. recover fractional quad coordinates (a,b) by inverse-bilinear Newton on the four nodes;
#   3. bilinearly blend the FOUR SURROUNDING CELL CENTRES, weights precomputed once.
# Step 3 matters: nearest-cell rendering makes block edges read as spurious wave fronts, which was
# user-reported on the transonic side.
#
# USAGE
#     include(joinpath(ROOT, "tools", "cmesh_raster.jl"))
#     R = build_raster(mesh, met, WIN, PX)     # once, after the mesh exists
#     img = raster_gather(R, cellfield)        # per frame; cellfield is (NI*NJ,) or (NI,NJ)
#     heatmap(R.XR, R.YR, img; ...)
# ============================================================================================

struct CMeshRaster
    PIX::Matrix{Int32}
    I4::Array{Int32,3}
    W4::Array{Float32,3}
    PX::Int
    PY::Int
    WIN::NTuple{4,Float64}
    XR::AbstractRange
    YR::AbstractRange
end

"""
    build_raster(mesh, met, WIN, PX) -> CMeshRaster

`WIN = (x0, x1, y0, y1)` in chord units; `PX` is the horizontal pixel count (vertical follows the
aspect ratio). Cost is O(PX*PY) with a 4-iteration Newton per pixel — a second or two, once.
"""
function build_raster(mesh, met, WIN::NTuple{4,Float64}, PX::Int)
    x0, x1, y0, y1 = WIN
    PY = round(Int, PX*(y1-y0)/(x1-x0))
    X, Y = mesh.X, mesh.Y
    NIm, NJm = met.NI, met.NJ

    # --- 1. containing cell per pixel, by bounding-box fill (later j overwrite earlier => outermost wins)
    PIX = zeros(Int32, PY, PX)
    for j in NJm:-1:1, i in 1:NIm
        xs = (X[i,j], X[i+1,j], X[i,j+1], X[i+1,j+1]); ys = (Y[i,j], Y[i+1,j], Y[i,j+1], Y[i+1,j+1])
        xmin, xmax = extrema(xs); ymin, ymax = extrema(ys)
        (xmax < x0 || xmin > x1 || ymax < y0 || ymin > y1) && continue
        pa = max(1, floor(Int,(xmin-x0)/(x1-x0)*PX)+1); pb = min(PX, ceil(Int,(xmax-x0)/(x1-x0)*PX))
        qa = max(1, floor(Int,(ymin-y0)/(y1-y0)*PY)+1); qb = min(PY, ceil(Int,(ymax-y0)/(y1-y0)*PY))
        v = Int32(i + (j-1)*NIm)
        for qq in qa:qb, pp in pa:pb; PIX[qq,pp] = v; end
    end

    # --- 2/3. inverse-bilinear -> weights over the four surrounding CELL CENTRES
    I4 = zeros(Int32, PY, PX, 4); W4 = zeros(Float32, PY, PX, 4)
    for p in 1:PX, q in 1:PY
        c = PIX[q,p]; c == 0 && continue
        i = Int(mod1(c, NIm)); j = Int((c-1) ÷ NIm) + 1
        _raster_weights!(I4, W4, X, Y, NIm, NJm, x0, x1, y0, y1, PX, PY, q, p, i, j)
    end
    CMeshRaster(PIX, I4, W4, PX, PY, WIN,
                range(x0, x1, length=PX), range(y0, y1, length=PY))
end

# Steps 2/3 for one pixel (q, p) claimed by cell (i, j): the inverse-bilinear quad coordinates of the pixel centre, then the
# weights over the four surrounding cell centres. build_raster's own loop body, moved here unchanged (7 Oct 2026) so that the
# exact containment pass below (raster_inside!) computes a reassigned pixel's weights by the same arithmetic.
function _raster_weights!(I4, W4, X, Y, NIm, NJm, x0, x1, y0, y1, PX, PY, q, p, i, j)
    xp = x0 + (p-0.5)*(x1-x0)/PX; yp = y0 + (q-0.5)*(y1-y0)/PY
    x00=X[i,j]; y00=Y[i,j]; x10=X[i+1,j]; y10=Y[i+1,j]
    x01=X[i,j+1]; y01=Y[i,j+1]; x11=X[i+1,j+1]; y11=Y[i+1,j+1]
    a = 0.5; b = 0.5
    for _ in 1:4
        fx = (1-a)*(1-b)*x00 + a*(1-b)*x10 + (1-a)*b*x01 + a*b*x11 - xp
        fy = (1-a)*(1-b)*y00 + a*(1-b)*y10 + (1-a)*b*y01 + a*b*y11 - yp
        dxa = -(1-b)*x00 + (1-b)*x10 - b*x01 + b*x11; dxb = -(1-a)*x00 - a*x10 + (1-a)*x01 + a*x11
        dya = -(1-b)*y00 + (1-b)*y10 - b*y01 + b*y11; dyb = -(1-a)*y00 - a*y10 + (1-a)*y01 + a*y11
        det = dxa*dyb - dxb*dya; abs(det) < 1e-30 && break
        a -= ( dyb*fx - dxb*fy)/det; b -= (-dya*fx + dxa*fy)/det
        a = clamp(a, 0.0, 1.0); b = clamp(b, 0.0, 1.0)
    end
    s = a - 0.5; t = b - 0.5
    i2 = clamp(i + (s >= 0 ? 1 : -1), 1, NIm); j2 = clamp(j + (t >= 0 ? 1 : -1), 1, NJm)
    ws = abs(s); wt = abs(t)
    I4[q,p,1] = Int32(i  + (j -1)*NIm); W4[q,p,1] = Float32((1-ws)*(1-wt))
    I4[q,p,2] = Int32(i2 + (j -1)*NIm); W4[q,p,2] = Float32(ws*(1-wt))
    I4[q,p,3] = Int32(i  + (j2-1)*NIm); W4[q,p,3] = Float32((1-ws)*wt)
    I4[q,p,4] = Int32(i2 + (j2-1)*NIm); W4[q,p,4] = Float32(ws*wt)
    nothing
end

# Is the point (px, py) inside the quad of cell (i, j), (i,j)-(i+1,j)-(i+1,j+1)-(i,j+1)? The signs of its four edge cross
# products all >= 0 or all <= 0 (either orientation; the boundary inclusive). Exact for a convex quad, which every cell of an
# admitted mesh is (the generator's convexity guard).
_raster_cross(ax, ay, bx, by, px, py) = (bx - ax)*(py - ay) - (by - ay)*(px - ax)
function _raster_inquad(X, Y, i, j, px, py)
    s1 = _raster_cross(X[i,j], Y[i,j], X[i+1,j], Y[i+1,j], px, py)
    s2 = _raster_cross(X[i+1,j], Y[i+1,j], X[i+1,j+1], Y[i+1,j+1], px, py)
    s3 = _raster_cross(X[i+1,j+1], Y[i+1,j+1], X[i,j+1], Y[i,j+1], px, py)
    s4 = _raster_cross(X[i,j+1], Y[i,j+1], X[i,j], Y[i,j], px, py)
    (s1 >= 0 && s2 >= 0 && s3 >= 0 && s4 >= 0) || (s1 <= 0 && s2 <= 0 && s3 <= 0 && s4 <= 0)
end

"""
    raster_inside!(R, mesh, met) -> Matrix{Bool}   (PY x PX)

The exact containment pass of a raster (the strip trailing-edge topology, 7 Oct 2026: the stage-3 review gate's O1,
memory/project_docs/_records/2026-10-05_strip_topology/stage3/REVIEW_GATE.md section 10). `build_raster` claims a pixel by the cells'
axis-aligned bounding boxes, the later cell winning, so a pixel whose centre lies inside one cell of an inclined block can
be claimed by a neighbour and drawn with that neighbour's clamped edge value. This pass tests every pixel centre in each
cell's bounding box for containment in the cell's quad (`_raster_inquad`), gives each contained pixel to its containing cell
(a centre on an edge two cells share goes to the later one in build_raster's order) with its four weights recomputed there by
build_raster's own arithmetic (`_raster_weights!`), and returns the mask of those pixels. Every other pixel keeps its
bounding-box claim. `R` must be the raster of the same mesh, window and pixel count. Done once, after build_raster: the
driver does it for the strip block's raster and gives the strip every pixel whose centre lies inside one of its cells
(`raster_gather_composite(...; inside2)`).
"""
function raster_inside!(R::CMeshRaster, mesh, met)
    X, Y = mesh.X, mesh.Y
    NIm, NJm = met.NI, met.NJ
    size(X) == size(Y) == (NIm+1, NJm+1) || throw(ArgumentError("raster_inside!: the mesh's nodes are not (NI+1) x (NJ+1)"))
    x0, x1, y0, y1 = R.WIN; PX, PY = R.PX, R.PY
    inside = fill(false, PY, PX)
    for j in NJm:-1:1, i in 1:NIm
        xs = (X[i,j], X[i+1,j], X[i,j+1], X[i+1,j+1]); ys = (Y[i,j], Y[i+1,j], Y[i,j+1], Y[i+1,j+1])
        xmin, xmax = extrema(xs); ymin, ymax = extrema(ys)
        (xmax < x0 || xmin > x1 || ymax < y0 || ymin > y1) && continue
        pa = max(1, floor(Int,(xmin-x0)/(x1-x0)*PX)+1); pb = min(PX, ceil(Int,(xmax-x0)/(x1-x0)*PX))
        qa = max(1, floor(Int,(ymin-y0)/(y1-y0)*PY)+1); qb = min(PY, ceil(Int,(ymax-y0)/(y1-y0)*PY))
        for qq in qa:qb, pp in pa:pb
            _raster_inquad(X, Y, i, j, x0 + (pp-0.5)*(x1-x0)/PX, y0 + (qq-0.5)*(y1-y0)/PY) || continue
            R.PIX[qq,pp] = Int32(i + (j-1)*NIm); inside[qq,pp] = true
            _raster_weights!(R.I4, R.W4, X, Y, NIm, NJm, x0, x1, y0, y1, PX, PY, qq, pp, i, j)
        end
    end
    inside
end

"""
    raster_gather(R, cellf) -> Matrix{Float64}   (PY x PX, NaN outside the mesh)

`cellf` may be `(NI,NJ)` or a flat `(NI*NJ,)` vector; indices are the same linear order the raster
was built with. NaN outside the mesh so the caller can render it as background.
"""
function raster_gather(R::CMeshRaster, cellf::AbstractArray{<:Real})
    f = vec(cellf)
    img = fill(NaN, R.PY, R.PX)
    Threads.@threads for p in 1:R.PX
        for q in 1:R.PY
            R.PIX[q,p] == 0 && continue
            @inbounds img[q,p] = R.W4[q,p,1]*f[R.I4[q,p,1]] + R.W4[q,p,2]*f[R.I4[q,p,2]] +
                                 R.W4[q,p,3]*f[R.I4[q,p,3]] + R.W4[q,p,4]*f[R.I4[q,p,4]]
        end
    end
    img
end

"""
    raster_gather_composite(R, cellf, R2, cellf2; inside2=nothing) -> Matrix{Float64}   (PY x PX, NaN outside both meshes)

One image from two blocks (the strip trailing-edge topology, 5 Oct 2026: the C block and the strip behind the base,
memory/project_docs/2026-10-05_STRIP_TOPOLOGY_PLAN.md section 4.7): `raster_gather(R, cellf)`, with every pixel that the first block's raster
leaves empty (`R.PIX == 0`) and the second block's covers taken from `raster_gather(R2, cellf2)`. Without `inside2` the first
block wins where both claim a pixel (its bounding-box fill can reach a pixel of the strip beside a leg's first row; its value
there is the leg cell's edge value: the stage-3 review gate's O1). With `inside2`, the mask `raster_inside!(R2, ...)`
returned for the second block's raster (7 Oct 2026), the second block also wins every pixel whose centre lies inside one of
its cells, so the strip is drawn from the strip wherever it is. Both rasters must have been built over the same window and
pixel count; each field is in its own raster's cell order.
"""
function raster_gather_composite(R::CMeshRaster, cellf::AbstractArray{<:Real}, R2::CMeshRaster, cellf2::AbstractArray{<:Real};
                                 inside2=nothing)
    (R.PX, R.PY, R.WIN) == (R2.PX, R2.PY, R2.WIN) ||
        throw(ArgumentError("raster_gather_composite: the two rasters differ in window or pixel count"))
    inside2 === nothing || (inside2 isa AbstractMatrix{Bool} && size(inside2) == size(R2.PIX)) ||
        throw(ArgumentError("raster_gather_composite: inside2 must be the second raster's containment mask (raster_inside!), PY x PX"))
    img = raster_gather(R, cellf)
    img2 = raster_gather(R2, cellf2)
    for p in 1:R.PX, q in 1:R.PY
        if (R.PIX[q,p] == 0 && R2.PIX[q,p] != 0) || (inside2 !== nothing && inside2[q,p])
            img[q,p] = img2[q,p]
        end
    end
    img
end
