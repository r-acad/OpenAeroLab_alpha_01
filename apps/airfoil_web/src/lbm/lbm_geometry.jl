# =====================================================================================================================
# lbm_geometry.jl -- geometry of the "LBM 3D (LUDWIG)" backend (27 Oct 2026). Pure Julia (Base, Printf, LinearAlgebra);
# no solver code. The LBM solver (src/main.jl, a pinned snapshot) is driven through the files this module writes:
#
#   1. the 2D section (chord units, x along the chord, y = the app's thickness direction) as one simple polygon: the host
#      airfoil, with an optional THICK Gurney plate at the trailing edge and an optional THICK spoiler hinged on the upper
#      surface, joined to the airfoil without gap or overlap (the polygon is the union, built by splicing the plate faces
#      into the contour at exact corner points; it is checked to be simple);
#   2. its extrusion along the span (the solver's Y axis) into a closed, watertight binary STL: the extrusion runs past
#      both periodic faces of the domain by `ext` (6 lattice cells), so the end caps never touch a lattice cell; inside the
#      domain the side walls are split at stations every `station_cells` cells and at both periodic faces exactly, so every
#      side triangle lies wholly inside or wholly outside the domain (the solver integrates surface forces per triangle and
#      gives a triangle outside the domain zero force);
#   3. a replica of the solver's single-level domain arithmetic (solver/physics_scaling.jl compute_domain_from_mesh and
#      geometry/geometry.jl rotate_mesh, on the Float32 vertices the solver reads), giving the lattice dimensions, the
#      placement of the body and the mid-span obstacle mask by the solver's own rule (geometry/domain_generation.jl
#      voxelize_blocks!: a cell is solid when its 1.5 dx box touches the surface, or when it is enclosed by those cells).
#
# Axes. STL/solver: X streamwise, Y span, Z up. The app's 2D section (x, y) maps to (X, Z). LUDWIG's basic.alpha
# rotates the STL about +Y with the matrix [cos a 0 -sin a; 0 1 0; sin a 0 cos a] about the STL's bounding-box centre:
# a positive basic.alpha raises the trailing edge (verified 27 Oct 2026), i.e. it is a NEGATIVE angle of attack for flow
# along +X with lift along +Z. The app's angle of attack alpha (nose up positive) is therefore written as
# basic.alpha = -alpha.
# =====================================================================================================================
module AirfoilLBMGeometry

using Printf
using LinearAlgebra

export lbm_section, lbm_extrude, write_stl_binary, read_stl_binary, lbm_lattice, stl_check_closed

const P2 = NTuple{2,Float64}

# ---------------------------------------------------------------------------------------------------- small helpers
_pt(x, y) = (Float64(x), Float64(y))
_sub(a, b) = (a[1] - b[1], a[2] - b[2])
_add(a, b) = (a[1] + b[1], a[2] + b[2])
_mul(a, s) = (a[1]*s, a[2]*s)
_norm(a) = hypot(a[1], a[2])
_unit(a) = (n = _norm(a); n > 0 || throw(ArgumentError("degenerate direction")); (a[1]/n, a[2]/n))
_cross(a, b) = a[1]*b[2] - a[2]*b[1]
_rot90(a) = (-a[2], a[1])                       # counter-clockwise quarter turn
_lerp(a, b, t) = (a[1] + t*(b[1] - a[1]), a[2] + t*(b[2] - a[2]))

"Signed area of a closed polygon (positive = counter-clockwise)."
function polygon_area(p::AbstractVector{P2})
    s = 0.0
    n = length(p)
    for i in 1:n
        a = p[i]; b = p[mod1(i + 1, n)]
        s += a[1]*b[2] - b[1]*a[2]
    end
    0.5s
end

"Proper or touching intersection of segments ab and cd (endpoints excluded when `strict`)."
function _segments_cross(a, b, c, d)
    d1 = _cross(_sub(b, a), _sub(c, a)); d2 = _cross(_sub(b, a), _sub(d, a))
    d3 = _cross(_sub(d, c), _sub(a, c)); d4 = _cross(_sub(d, c), _sub(b, c))
    (d1*d2 < 0) && (d3*d4 < 0)
end

"""The first self-intersection of a closed polygon (two non-adjacent edges crossing), or nothing. O(n^2), with an x-sorted
sweep prefilter; used once per build on at most a few thousand edges."""
function polygon_self_intersection(p::AbstractVector{P2})
    n = length(p)
    lo = [min(p[i][1], p[mod1(i + 1, n)][1]) for i in 1:n]
    hi = [max(p[i][1], p[mod1(i + 1, n)][1]) for i in 1:n]
    order = sortperm(lo)
    active = Int[]
    for i in order
        filter!(j -> hi[j] >= lo[i], active)
        a = p[i]; b = p[mod1(i + 1, n)]
        for j in active
            (abs(i - j) <= 1 || abs(i - j) == n - 1) && continue
            c = p[j]; d = p[mod1(j + 1, n)]
            _segments_cross(a, b, c, d) && return (i, j)
        end
        push!(active, i)
    end
    nothing
end

"Resample the polyline `pts` (exact end points) at nearly uniform arc length, at most `ds` apart."
function _resample(pts::AbstractVector{P2}, ds::Float64)
    length(pts) >= 2 || throw(ArgumentError("a contour piece needs two points"))
    s = zeros(length(pts))
    for k in 2:length(pts)
        s[k] = s[k-1] + _norm(_sub(pts[k], pts[k-1]))
    end
    L = s[end]
    L > 0 || throw(ArgumentError("a contour piece has zero length"))
    m = max(1, ceil(Int, L/ds - 1e-9))
    out = P2[pts[1]]
    k = 1
    for q in 1:m-1
        t = q*L/m
        while s[k+1] < t; k += 1; end
        w = (t - s[k])/(s[k+1] - s[k])
        push!(out, _lerp(pts[k], pts[k+1], w))
    end
    push!(out, pts[end])
    out
end

# ---------------------------------------------------------------------------------------------------- the 2D section
"""Index k and weight w with the point at x = xc on the lower surface (wrap order: index 1 is the lower trailing edge, the
surface runs toward the leading edge at `ile`): x[k] >= xc >= x[k+1], scanning from the trailing edge."""
function _lower_crossing(x, y, ile, xc)
    for k in 1:ile-1
        if x[k] >= xc >= x[k+1]
            w = x[k] == x[k+1] ? 0.0 : (x[k] - xc)/(x[k] - x[k+1])
            return k, w
        end
    end
    throw(ArgumentError(@sprintf("no lower-surface point at x/c = %.5f", xc)))
end
"The same on the upper surface (index n is the upper trailing edge), scanning from the trailing edge: x[k-1] <= xc <= x[k]."
function _upper_crossing(x, y, ile, xc)
    n = length(x)
    for k in n:-1:ile+1
        if x[k-1] <= xc <= x[k]
            w = x[k] == x[k-1] ? 0.0 : (x[k] - xc)/(x[k] - x[k-1])
            return k, w
        end
    end
    throw(ArgumentError(@sprintf("no upper-surface point at x/c = %.5f", xc)))
end

"""Validated spoiler settings from the request's flow map, or nothing when off. Keys: lbm_spoiler (Bool),
lbm_spoiler_hinge (x/c of the hinge on the upper surface, 0.2-0.95, default 0.7), lbm_spoiler_length (l/c, 0.01-0.3,
default 0.1), lbm_spoiler_deflection (deg relative to the local upper-surface tangent, 10-90, default 30)."""
function spoiler_settings(flow::AbstractDict)
    get(flow, "lbm_spoiler", false) === true || return nothing
    num(k, d, lo, hi) = begin
        v = get(flow, k, d)
        (v isa Real && !(v isa Bool) && isfinite(v)) || throw(ArgumentError("$k must be a finite number"))
        lo <= v <= hi || throw(ArgumentError("$k must lie in [$lo, $hi] (got $v)"))
        Float64(v)
    end
    (hinge=num("lbm_spoiler_hinge", 0.7, 0.2, 0.95), length=num("lbm_spoiler_length", 0.1, 0.01, 0.3),
     deflection=num("lbm_spoiler_deflection", 30.0, 10.0, 90.0))
end

"""    lbm_section(x, y; ds, gurney=nothing, spoiler=nothing, thickness) -> (polygon, metadata)

The section polygon (counter-clockwise, chord units, (x, z) with z the app's y) of the host contour `x, y` (wrap order:
lower trailing edge -> leading edge -> upper trailing edge; a finite trailing edge is closed by its straight base), with:
- `gurney` = (height=h, mode="lower"|"upper"|"symmetric"): a plate of thickness `thickness` perpendicular to the chord at
  the trailing edge: x in [x_TE - t, x_TE] (its back face on the base line x = x_TE), from the trailing-edge point(s) out
  to h (h/2 each side for symmetric). The two trailing-edge points must share x (gurney.jl trims a cambered section);
- `spoiler` = (hinge, length, deflection): a flat plate of thickness `thickness` hinged on the upper surface at x/c =
  hinge: its rear face runs from the hinge point H along D (the local tangent turned out of the surface by the deflection)
  for `length`; its front face is offset by the thickness on the upstream side and runs down to the surface, which it
  meets at Q, upstream of H. The cove between the rear face and the surface stays fluid.
Every corner (trailing-edge points, plate roots and tips, hinge, Q) is an exact polygon vertex; the smooth parts and the
straight faces are resampled at most `ds` apart."""
function lbm_section(x0::AbstractVector{<:Real}, y0::AbstractVector{<:Real}; ds::Real, thickness::Real,
                     gurney=nothing, spoiler=nothing)
    x = Float64.(collect(x0)); y = Float64.(collect(y0)); n = length(x)
    n == length(y) >= 21 || throw(ArgumentError("the section needs at least 21 contour points"))
    all(isfinite, x) && all(isfinite, y) || throw(ArgumentError("airfoil coordinates must be finite"))
    ds = Float64(ds); t = Float64(thickness)
    ds > 0 && t > 0 || throw(ArgumentError("ds and the plate thickness must be positive"))
    ile = argmin(x)
    3 <= ile <= n - 2 || throw(ArgumentError("the leading edge (minimum x) must lie inside the contour"))
    lte = (x[1], y[1]); ute = (x[n], y[n])
    meta = Dict{String,Any}("lower_te"=>[lte...], "upper_te"=>[ute...], "te_gap"=>_norm(_sub(ute, lte)),
                            "plate_thickness"=>t, "ds"=>ds)
    # The airfoil part between its two ends: start/stop indices and exact end points (replaced by plate roots).
    k0, p0 = 1, lte                # first airfoil point (wrap order) and the index of the next source point
    k1, p1 = n, ute                # last airfoil point and the index of the previous source point
    head = P2[]                    # straight pieces before the airfoil part (lower plate: B -> F -> R)
    tail = P2[]                    # after it (upper plate: R -> F -> B)
    base_from = ute; base_to = lte # the base: from the last point back to the first (straight)
    if gurney !== nothing
        h = Float64(gurney.height); mode = String(gurney.mode)
        abs(x[1] - x[n]) <= 1e-6 || throw(ArgumentError(@sprintf(
            "the Gurney plate stands on one trailing-edge station; this section's trailing-edge points are %.2e c apart in x", abs(x[1] - x[n]))))
        xte = 0.5(x[1] + x[n])
        hl, hu = mode == "lower" ? (h, 0.0) : mode == "upper" ? (0.0, h) : mode == "symmetric" ? (0.5h, 0.5h) :
                 throw(ArgumentError("gurney mode must be lower, upper or symmetric"))
        xr = xte - t
        xr > x[ile] + 0.5 || throw(ArgumentError("the Gurney plate thickness is too large for this chord"))
        if hl > 0
            k, w = _lower_crossing(x, y, ile, xr)
            R = (xr, y[k] + w*(y[k+1] - y[k]))
            B = (xte, y[1] - hl); F = (xr, y[1] - hl)
            R[2] > F[2] || throw(ArgumentError("the lower Gurney plate root lies below its tip"))
            head = P2[B, F, R]; k0 = k + 1; p0 = R; base_to = B
            meta["lower_plate"] = Dict("tip_back"=>[B...], "tip_front"=>[F...], "root"=>[R...], "height"=>hl)
        end
        if hu > 0
            k, w = _upper_crossing(x, y, ile, xr)
            R = (xr, y[k] + w*(y[k-1] - y[k]))
            F = (xr, y[n] + hu); B = (xte, y[n] + hu)
            R[2] < F[2] || throw(ArgumentError("the upper Gurney plate root lies above its tip"))
            tail = P2[R, F, B]; k1 = k - 1; p1 = R; base_from = B
            meta["upper_plate"] = Dict("tip_back"=>[B...], "tip_front"=>[F...], "root"=>[R...], "height"=>hu)
        end
        meta["gurney"] = Dict("height"=>h, "mode"=>mode, "x_te"=>xte, "thickness"=>t,
            "construction"=>"plate x in [x_TE - t, x_TE] (back face on the base line x = x_TE), perpendicular to the chord, exact corners")
    end
    # The airfoil part as one dense polyline (exact ends), then the spoiler splice.
    airfoil = P2[p0; [(x[k], y[k]) for k in k0:k1]; p1]
    # drop duplicated ends (a root may coincide with a source point)
    unique_ends!(v) = (length(v) >= 2 && v[1] == v[2] && deleteat!(v, 2);
                       length(v) >= 2 && v[end] == v[end-1] && deleteat!(v, length(v) - 1); v)
    unique_ends!(airfoil)
    pieces = Vector{Vector{P2}}()
    smooth = Bool[]
    for k in 1:length(head)-1
        push!(pieces, P2[head[k], head[k+1]]); push!(smooth, false)
    end
    if spoiler !== nothing
        xh = Float64(spoiler.hinge); l = Float64(spoiler.length); dl = deg2rad(Float64(spoiler.deflection))
        # the hinge H on the upper surface, in the dense airfoil polyline (the upper surface runs toward the trailing edge)
        ia = findfirst(p -> p == (x[ile], y[ile]), airfoil)
        ia === nothing && (ia = argmin(first.(airfoil)))
        kh = 0
        for k in ia:length(airfoil)-1
            if airfoil[k][1] <= xh <= airfoil[k+1][1]
                kh = k; break
            end
        end
        kh > 0 || throw(ArgumentError(@sprintf("the spoiler hinge x/c = %.3f is not on the upper surface", xh)))
        w = (xh - airfoil[kh][1])/(airfoil[kh+1][1] - airfoil[kh][1])
        H = _lerp(airfoil[kh], airfoil[kh+1], w)
        T = _unit(_sub(airfoil[min(kh + 2, end)], airfoil[max(kh - 1, ia)]))   # local tangent, toward the trailing edge
        N = _rot90(T)                                                            # outward (up)
        N[2] > 0 || throw(ArgumentError("the spoiler hinge must be on the upper surface"))
        D = _add(_mul(T, cos(dl)), _mul(N, sin(dl)))                             # the panel direction from the hinge
        P = _rot90(D)                                                            # toward the panel's upstream face
        A = _add(_add(H, _mul(P, t)), _mul(D, l))                               # front (upstream) tip corner
        Bt = _add(H, _mul(D, l))                                                 # rear tip corner
        # Q: the front face line H + t P + s D (s < 0) meets the upper surface upstream of H
        O = _add(H, _mul(P, t))
        Q = nothing; kq = 0
        for k in kh:-1:ia
            a = airfoil[k]; b = airfoil[k+1]
            den = _cross(D, _sub(b, a))
            abs(den) < 1e-300 && continue
            # O + s D = a + u (b - a)
            s = _cross(_sub(a, O), _sub(b, a))/den
            u = _cross(_sub(a, O), D)/den
            if s <= 0 && -1e-12 <= u <= 1 + 1e-12 && (k < kh || u <= w)
                Q = _lerp(a, b, clamp(u, 0.0, 1.0)); kq = k; break
            end
        end
        Q === nothing && throw(ArgumentError("the spoiler's front face does not meet the upper surface upstream of the hinge; reduce its thickness or increase the deflection"))
        front = P2[airfoil[1:kq]; Q]
        back = P2[H; airfoil[kh+1:end]]
        unique_ends!(front); unique_ends!(back)
        push!(pieces, front); push!(smooth, true)
        push!(pieces, P2[Q, A]); push!(smooth, false)
        push!(pieces, P2[A, Bt]); push!(smooth, false)
        push!(pieces, P2[Bt, H]); push!(smooth, false)
        push!(pieces, back); push!(smooth, true)
        meta["spoiler"] = Dict("hinge"=>[H...], "front_root"=>[Q...], "front_tip"=>[A...], "rear_tip"=>[Bt...],
            "hinge_x"=>xh, "length"=>l, "deflection_deg"=>Float64(spoiler.deflection), "thickness"=>t,
            "tangent"=>[T...], "direction"=>[D...],
            "front_root_offset"=>_norm(_sub(Q, H)),
            "construction"=>"rear face from the hinge H on the upper surface along the tangent turned out by the deflection; front face offset t upstream and run down to the surface at Q; the cove under the panel is fluid")
    else
        push!(pieces, airfoil); push!(smooth, true)
    end
    for k in 1:length(tail)-1
        push!(pieces, P2[tail[k], tail[k+1]]); push!(smooth, false)
    end
    # the base (straight): from the last point back to the first (through the trailing-edge points)
    if _norm(_sub(base_from, base_to)) > 1e-12
        push!(pieces, P2[base_from, base_to]); push!(smooth, false)
    end
    # resample each piece; the joints are exact vertices
    wrap = P2[]
    corners = P2[]
    for (pc, sm) in zip(pieces, smooth)
        r = _resample(pc, ds)
        push!(corners, r[1])
        isempty(wrap) ? append!(wrap, r) : append!(wrap, r[2:end])
    end
    wrap[end] == wrap[1] && pop!(wrap)          # closed: the base ends on the first point
    poly = reverse(wrap)                          # the wrap order is clockwise; the polygon is counter-clockwise
    A2 = polygon_area(poly)
    A2 > 0 || throw(ArgumentError("the section polygon is not counter-clockwise (area $(A2))"))
    hit = polygon_self_intersection(poly)
    hit === nothing || throw(ArgumentError(@sprintf("the section with its plates intersects itself near (%.4f, %.4f): change the Gurney or spoiler settings",
                                                     poly[hit[1]][1], poly[hit[1]][2])))
    meta["points"] = length(poly); meta["area"] = A2
    meta["corners"] = [[c...] for c in corners]
    xs = first.(poly); zs = last.(poly)
    meta["bbox"] = [minimum(xs), maximum(xs), minimum(zs), maximum(zs)]
    poly, meta
end

# ---------------------------------------------------------------------------------------------------- triangulation
"Ear-clipping triangulation of a simple counter-clockwise polygon: index triples, counter-clockwise."
function earclip(p::AbstractVector{P2})
    n = length(p)
    idx = collect(1:n)
    tris = NTuple{3,Int}[]
    inside(a, b, c, q) = _cross(_sub(b, a), _sub(q, a)) >= 0 && _cross(_sub(c, b), _sub(q, b)) >= 0 &&
                         _cross(_sub(a, c), _sub(q, c)) >= 0
    guard = 0
    while length(idx) > 3
        m = length(idx); clipped = false
        for k in 1:m
            i0 = idx[mod1(k - 1, m)]; i1 = idx[k]; i2 = idx[mod1(k + 1, m)]
            a, b, c = p[i0], p[i1], p[i2]
            _cross(_sub(b, a), _sub(c, b)) > 0 || continue          # convex corner
            ok = true
            for j in idx
                (j == i0 || j == i1 || j == i2) && continue
                if inside(a, b, c, p[j]) ; ok = false; break; end
            end
            ok || continue
            push!(tris, (i0, i1, i2)); deleteat!(idx, k); clipped = true
            break
        end
        clipped || throw(ArgumentError("ear clipping failed: the section polygon is not simple"))
        (guard += 1) > 10n && throw(ArgumentError("ear clipping did not terminate"))
    end
    push!(tris, (idx[1], idx[2], idx[3]))
    tris
end

# ---------------------------------------------------------------------------------------------------- extrusion and STL
"""    lbm_extrude(poly; span, ext, dy_station) -> (vertices, triangles, metadata)

Extrude the counter-clockwise (x, z) polygon along Y from -ext to span + ext: stations at -ext, 0, every `dy_station`
(rounded to divide the span), span and span + ext; end caps at -ext and span + ext (ear clipping). Vertices are
(X, Y, Z) = (x, y, z) in chord units; triangles are index triples with outward normals."""
function lbm_extrude(poly::AbstractVector{P2}; span::Real, ext::Real, dy_station::Real)
    span = Float64(span); ext = Float64(ext)
    nseg = max(1, round(Int, span/Float64(dy_station)))
    ys = [-ext; [span*k/nseg for k in 0:nseg]; span + ext]
    np = length(poly); ns = length(ys)
    V = NTuple{3,Float64}[]
    for y in ys, (px, pz) in poly
        push!(V, (px, y, pz))
    end
    vid(i, s) = (s - 1)*np + i
    T = NTuple{3,Int}[]
    for s in 1:ns-1, i in 1:np
        j = mod1(i + 1, np)
        a = vid(i, s); b = vid(j, s); c = vid(j, s + 1); d = vid(i, s + 1)
        push!(T, (a, c, b)); push!(T, (a, d, c))       # outward for a counter-clockwise (x, z) polygon
    end
    caps = earclip(poly)
    for (i0, i1, i2) in caps
        push!(T, (vid(i0, 1), vid(i1, 1), vid(i2, 1)))       # y = -ext: normal -Y
        push!(T, (vid(i0, ns), vid(i2, ns), vid(i1, ns)))    # y = span + ext: normal +Y
    end
    V, T, Dict{String,Any}("stations"=>ys, "station_count"=>ns, "side_triangles"=>2np*(ns - 1), "cap_triangles"=>2length(caps),
                           "triangles"=>length(T), "vertices"=>length(V), "span"=>span, "extension"=>ext)
end

"Closedness: every edge used exactly twice, once in each direction (a watertight, consistently oriented surface)."
function stl_check_closed(T::AbstractVector{NTuple{3,Int}})
    count = Dict{Tuple{Int,Int},Int}()
    for (a, b, c) in T, e in ((a, b), (b, c), (c, a))
        count[e] = get(count, e, 0) + 1
    end
    bad = 0
    for (e, k) in count
        (k == 1 && get(count, (e[2], e[1]), 0) == 1) || (bad += 1)
    end
    (closed=bad == 0, edges=length(count), bad_edges=bad)
end

"""Write a binary STL (80-byte header, Float32 facet normals and vertices). Returns the Float32-rounded vertex triples
exactly as a reader sees them."""
function write_stl_binary(path::AbstractString, V, T; header::AbstractString="Airfoil Lab LBM 3D section")
    open(path, "w") do io
        h = Vector{UInt8}(codeunits(first(header, 80)))
        write(io, vcat(h, zeros(UInt8, 80 - length(h))))
        write(io, UInt32(length(T)))
        for (a, b, c) in T
            p1 = V[a]; p2 = V[b]; p3 = V[c]
            u = p2 .- p1; w = p3 .- p1
            nrm = (u[2]*w[3] - u[3]*w[2], u[3]*w[1] - u[1]*w[3], u[1]*w[2] - u[2]*w[1])
            L = sqrt(sum(abs2, nrm)); L > 0 && (nrm = nrm ./ L)
            for v in (nrm, p1, p2, p3), q in v
                write(io, Float32(q))
            end
            write(io, UInt16(0))
        end
    end
    path
end

"Read a binary STL as the solver does (Float32 vertices to Float64): a vector of triangles ((x,y,z) x 3)."
function read_stl_binary(path::AbstractString)
    open(path, "r") do io
        skip(io, 80)
        n = read(io, UInt32)
        tris = Vector{NTuple{3,NTuple{3,Float64}}}(undef, n)
        for i in 1:n
            skip(io, 12)
            v = ntuple(_ -> ntuple(_ -> Float64(read(io, Float32)), 3), 3)
            skip(io, 2)
            tris[i] = v
        end
        tris
    end
end

# ---------------------------------------------------------------------------------------------------- the lattice
"""    lbm_lattice(tris; alpha_lbm_deg, cells_per_chord, span_cells, upstream, downstream, height, levels=1) -> NamedTuple

The solver's domain for the STL triangles `tris` (as read): rotation by basic.alpha = `alpha_lbm_deg` about the
bounding-box centre (geometry.jl rotate_mesh; Rz(0)*Ry), then physics_scaling.jl compute_domain_from_mesh with
reference_length_for_meshing = 1 (the chord), surface_resolution = `cells_per_chord` (the FINEST level), num_levels =
`levels`, the span override written as (span_cells - 0.5) dx_finest, and the domain extents upstream/downstream/height
in chords. Every dimension is rounded up to whole 8-cell blocks of the COARSEST level (dx_coarse = dx 2^(levels-1))
exactly as the solver does. With levels > 1 (the solver's periodic-Y AMR, periodic_y_amr): the level count must
survive the solver's healthy-pyramid clamp (x and z: at least `min_coarse_blocks` coarsest blocks; the periodic Y: one)
and the span must be a whole number of coarsest blocks; both are refused here with the solver's reasons. nx, ny, nz,
dx and cells are in FINEST cells (the uniform-equivalent lattice; the mid-span mask is on it); ncx, ncy, ncz and
blocks are the coarsest level's."""
function lbm_lattice(tris; alpha_lbm_deg::Real, cells_per_chord::Integer, span_cells::Integer,
                     upstream::Real, downstream::Real, height::Real, block_size::Integer=8, levels::Integer=1,
                     min_coarse_blocks::Integer=4)
    # rotation about the bounding-box centre (rotate_mesh: origin = (min + max)/2 of the loaded vertices)
    mn = [Inf, Inf, Inf]; mx = [-Inf, -Inf, -Inf]
    for t in tris, p in t, d in 1:3
        mn[d] = min(mn[d], p[d]); mx[d] = max(mx[d], p[d])
    end
    origin = (mn .+ mx) ./ 2
    a = deg2rad(Float64(alpha_lbm_deg)); ca, sa = cos(a), sin(a)
    R = [ca 0.0 -sa; 0.0 1.0 0.0; sa 0.0 ca]
    rot(p) = (q = R*([p[1], p[2], p[3]] .- origin) .+ origin; (q[1], q[2], q[3]))
    rmn = [Inf, Inf, Inf]; rmx = [-Inf, -Inf, -Inf]
    if alpha_lbm_deg != 0
        for t in tris, p in t
            q = rot(p)
            for d in 1:3
                rmn[d] = min(rmn[d], q[d]); rmx[d] = max(rmx[d], q[d])
            end
        end
    else
        rmn .= mn; rmx .= mx
    end
    mesh_center = (rmn .+ rmx) ./ 2
    mesh_extent = rmx .- rmn
    ref = 1.0
    dx = ref/cells_per_chord
    span_override = (span_cells - 0.5)*dx
    domain_x = ref*(upstream + downstream) + mesh_extent[1]
    domain_y = span_override
    domain_z = mesh_extent[3] + 2*ref*height
    bs = block_size
    L = Int(levels)
    if L > 1
        # physics_scaling.jl compute_max_levels_for_domain (soft cap), periodic-Y AMR form: x/z with min_coarse_blocks
        # on the requested extents, the periodic Y with one block on the span as the solver rounds it (span_cells finest
        # cells; the integration check's fix of 27 Oct: the unfixed solver read the raw span_override, so a span of one
        # coarsest block lost a level)
        cap(size, mb) = (r = size/(dx*mb*bs); r < 1.0 ? 1 : Int(floor(1 + log2(r + 1e-9))))
        maxL = min(cap(min(domain_x, domain_z), min_coarse_blocks), cap(span_cells*dx, 1))
        L <= maxL || throw(ArgumentError("$(L) refinement levels need a larger domain or span: the solver would reduce them to $(maxL) (each axis needs whole blocks of the coarsest level, $(bs*2^(L-1)) finest cells; x and z at least $(min_coarse_blocks) of them)."))
    end
    f = 2^(L - 1)
    dxc = dx*f
    ncx = max(bs, Int(ceil(ceil(domain_x/dxc)/bs)*bs))
    ncy = max(bs, Int(ceil(ceil(domain_y/dxc)/bs)*bs))
    ncz = max(bs, Int(ceil(ceil(domain_z/dxc)/bs)*bs))
    Lx, Ly, Lz = ncx*dxc, ncy*dxc, ncz*dxc
    if L > 1 && abs(Ly - span_override) >= dx
        q = bs*f
        throw(ArgumentError("the span ($(span_cells) finest cells) is not a whole number of coarsest blocks ($(q) finest cells with $(L) levels): the solver refuses it; use $(q*max(1, fld(span_cells, q))) or $(q*(fld(span_cells, q) + 1)) cells."))
    end
    nxc, nyc, nzc = ncx*f, ncy*f, ncz*f
    offset = (ref*upstream - rmn[1], Ly/2 - mesh_center[2], Lz/2 - mesh_center[3])
    (dx=dx, nx=nxc, ny=nyc, nz=nzc, bx=ncx ÷ bs, by=ncy ÷ bs, bz=ncz ÷ bs, Lx=Lx, Ly=Ly, Lz=Lz,
     cells=nxc*nyc*nzc, blocks=(ncx ÷ bs)*(ncy ÷ bs)*(ncz ÷ bs), offset=offset, origin=Tuple(origin),
     rotation=R, rotated_min=Tuple(rmn), rotated_max=Tuple(rmx), mesh_center=Tuple(mesh_center),
     mesh_extent=Tuple(mesh_extent), span_override=span_override, domain_x_raw=domain_x, domain_z_raw=domain_z,
     upstream=Float64(upstream), downstream=Float64(downstream), height=Float64(height),
     levels=L, dx_coarse=dxc, ncx=ncx, ncy=ncy, ncz=ncz, coarse_cells=ncx*ncy*ncz, block_m=bs*dxc)
end

"""The largest finest-level shell (in finest 8-cell blocks) that keeps the refined region out of the solver's farfield
wall buffer (domain_topology.jl: WALL_BUFFER_BLOCKS = 4 finest blocks at each x and z face of an axis with at least 9
blocks; the standoff only warns "[Standoff] ... fall in the farfield wall buffer" and adds the blocks anyway, so a
refined level would touch the wall). The body's footprint is its cells (the 1.5 dx box rule) in finest blocks; the shell
dilates it isotropically. Returns (max_shell, limiting face)."""
function lbm_max_shell_blocks(pp::AbstractVector, lat; wall_buffer::Integer=4)
    dx = lat.dx; bsz = 8dx
    xs = first.(pp); zs = last.(pp)
    blk(v) = floor(Int, v/bsz) + 1
    bx0 = blk(minimum(xs) - 0.75dx); bx1 = blk(maximum(xs) + 0.75dx)
    bz0 = blk(minimum(zs) - 0.75dx); bz1 = blk(maximum(zs) + 0.75dx)
    bxm = lat.nx ÷ 8; bzm = lat.nz ÷ 8
    room = Tuple{Int,String}[]
    # a block b is in the buffer when b <= wall_buffer or b >= b_max - (wall_buffer - 1)
    bxm >= 2wall_buffer + 1 && append!(room, [(bx0 - wall_buffer - 1, "inlet"), (bxm - wall_buffer - bx1, "outlet")])
    bzm >= 2wall_buffer + 1 && append!(room, [(bz0 - wall_buffer - 1, "bottom wall"), (bzm - wall_buffer - bz1, "top wall")])
    isempty(room) && return (typemax(Int), "none")
    m = minimum(first.(room))
    (m, room[findfirst(r -> first(r) == m, room)][2])
end

"The section polygon rotated and placed in domain coordinates (X, Z), with the lattice `lat` of lbm_lattice."
function placed_polygon(poly::AbstractVector{P2}, lat)
    R = lat.rotation; o = lat.origin; off = lat.offset
    out = Vector{P2}(undef, length(poly))
    for (k, p) in enumerate(poly)
        q = R*[p[1] - o[1], 0.0, p[2] - o[3]]
        out[k] = (q[1] + o[1] + off[1], q[3] + o[3] + off[3])
    end
    out
end

"Does the axis-aligned square (centre c, half-width hw) touch the segment ab? (separating-axis test in 2D)"
function _square_touches_segment(c, hw, a, b)
    # bounding boxes
    (max(a[1], b[1]) < c[1] - hw || min(a[1], b[1]) > c[1] + hw ||
     max(a[2], b[2]) < c[2] - hw || min(a[2], b[2]) > c[2] + hw) && return false
    # the segment's normal axis
    nx, nz = b[2] - a[2], -(b[1] - a[1])
    d0 = nx*(a[1] - c[1]) + nz*(a[2] - c[2])
    r = hw*(abs(nx) + abs(nz))
    abs(d0) <= r
end

"Point in polygon (crossing number)."
function _inside(poly, q)
    c = false; n = length(poly)
    for i in 1:n
        a = poly[i]; b = poly[mod1(i + 1, n)]
        if (a[2] > q[2]) != (b[2] > q[2])
            xi = a[1] + (q[2] - a[2])*(b[1] - a[1])/(b[2] - a[2])
            xi > q[1] && (c = !c)
        end
    end
    c
end

"""The solver's obstacle mask on one span-normal (XZ) plane, in the window of cells around the placed polygon `pp`
(domain coordinates): a cell is solid when its 1.5 dx box touches the surface (voxelize_blocks!, box half-width
0.75 dx) or when its centre is inside the section (the flood fill leaves enclosed cells solid). Returns the window's
first cell indices (1-based), its size and the mask as a vector, i fastest."""
function obstacle_window(pp::AbstractVector{P2}, lat; margin_cells::Integer=6)
    dx = lat.dx
    xs = first.(pp); zs = last.(pp)
    i0 = max(1, floor(Int, minimum(xs)/dx) + 1 - margin_cells); i1 = min(lat.nx, ceil(Int, maximum(xs)/dx) + margin_cells)
    k0 = max(1, floor(Int, minimum(zs)/dx) + 1 - margin_cells); k1 = min(lat.nz, ceil(Int, maximum(zs)/dx) + margin_cells)
    ni = i1 - i0 + 1; nk = k1 - k0 + 1
    mask = falses(ni*nk)
    shell = falses(ni*nk)
    hw = 0.75dx
    n = length(pp)
    for e in 1:n
        a = pp[e]; b = pp[mod1(e + 1, n)]
        ia = max(i0, floor(Int, (min(a[1], b[1]) - hw)/dx - 0.5) + 1); ib = min(i1, ceil(Int, (max(a[1], b[1]) + hw)/dx + 0.5))
        ka = max(k0, floor(Int, (min(a[2], b[2]) - hw)/dx - 0.5) + 1); kb = min(k1, ceil(Int, (max(a[2], b[2]) + hw)/dx + 0.5))
        for k in ka:kb, i in ia:ib
            c = ((i - 0.5)*dx, (k - 0.5)*dx)
            _square_touches_segment(c, hw, a, b) && (shell[(k - k0)*ni + (i - i0) + 1] = true)
        end
    end
    for k in k0:k1, i in i0:i1
        q = (k - k0)*ni + (i - i0) + 1
        mask[q] = shell[q] || _inside(pp, ((i - 0.5)*dx, (k - 0.5)*dx))
    end
    (i0=i0, k0=k0, ni=ni, nk=nk, mask=mask, solid=count(mask), shell=count(shell))
end

end # module AirfoilLBMGeometry
