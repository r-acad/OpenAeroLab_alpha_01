# Surface geometry: STL reading (binary and ASCII), a triangle bounding-volume hierarchy, and the queries the mesh
# generator needs: nearest distance to the surface, first crossing of a segment (the Bouzidi q of a cut link),
# axis-ray parity (inside/outside), and the nearest triangle (wall normal and triangle id). Plain tuples of Float64
# are used for vectors; the queries take a caller-owned stack so that threads do not allocate.

const V3 = NTuple{3,Float64}
const Tri = NTuple{3,V3}

@inline vsub(a::V3, b::V3) = (a[1] - b[1], a[2] - b[2], a[3] - b[3])
@inline vadd(a::V3, b::V3) = (a[1] + b[1], a[2] + b[2], a[3] + b[3])
@inline vmul(a::V3, s::Real) = (a[1] * s, a[2] * s, a[3] * s)
@inline vdot(a::V3, b::V3) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
@inline vcross(a::V3, b::V3) = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
@inline vnorm2(a::V3) = vdot(a, a)
@inline vnorm(a::V3) = sqrt(vnorm2(a))
@inline vmin(a::V3, b::V3) = (min(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
@inline vmax(a::V3, b::V3) = (max(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

"Read a binary or ASCII STL; triangles are translated by `offset`. Degenerate triangles are dropped and counted."
function read_stl(path::AbstractString; offset::V3=(0.0, 0.0, 0.0))
    bytes = read(path)
    ascii = length(bytes) >= 6 && String(bytes[1:5]) == "solid" && !(length(bytes) >= 84 && 84 + 50 * Int(reinterpret(UInt32, bytes[81:84])[1]) == length(bytes))
    tri = Tri[]
    if ascii
        v = V3[]
        for line in eachline(IOBuffer(bytes))
            s = split(strip(line))
            if length(s) == 4 && s[1] == "vertex"
                push!(v, vadd((parse(Float64, s[2]), parse(Float64, s[3]), parse(Float64, s[4])), offset))
                if length(v) == 3
                    push!(tri, (v[1], v[2], v[3])); empty!(v)
                end
            end
        end
    else
        length(bytes) >= 84 || error("STL $path is too short")
        n = Int(reinterpret(UInt32, bytes[81:84])[1])
        84 + 50n == length(bytes) || error("STL $path: $(n) triangles announced, $(length(bytes)) bytes present")
        sizehint!(tri, n)
        for t in 0:n-1
            o = 84 + 50t
            f = reinterpret(Float32, bytes[o+13:o+48])
            push!(tri, (vadd((Float64(f[1]), Float64(f[2]), Float64(f[3])), offset),
                        vadd((Float64(f[4]), Float64(f[5]), Float64(f[6])), offset),
                        vadd((Float64(f[7]), Float64(f[8]), Float64(f[9])), offset)))
        end
    end
    kept = Tri[]; dropped = 0
    for t in tri
        if vnorm2(vcross(vsub(t[2], t[1]), vsub(t[3], t[1]))) > 0
            push!(kept, t)
        else
            dropped += 1
        end
    end
    kept, dropped
end

"Axis-aligned bounds of a triangle list."
function bounds(tri::Vector{Tri})
    lo = (Inf, Inf, Inf); hi = (-Inf, -Inf, -Inf)
    for t in tri, v in t
        lo = vmin(lo, v); hi = vmax(hi, v)
    end
    lo, hi
end

"Triangle normal (unit) and area."
@inline function tri_normal_area(t::Tri)
    n = vcross(vsub(t[2], t[1]), vsub(t[3], t[1])); l = vnorm(n)
    vmul(n, 1 / l), 0.5l
end

# ---------------------------------------------------------------- BVH
struct BVH
    lo::Vector{V3}; hi::Vector{V3}
    left::Vector{Int32}; right::Vector{Int32}
    first::Vector{Int32}; count::Vector{Int32}
    order::Vector{Int32}
    tri::Vector{Tri}
end

function tri_box(t::Tri)
    vmin(vmin(t[1], t[2]), t[3]), vmax(vmax(t[1], t[2]), t[3])
end

"Build a median-split BVH with leaves of at most `leaf` triangles."
function build_bvh(tri::Vector{Tri}; leaf::Int=8)
    n = length(tri); n > 0 || error("no triangles")
    cen = [vmul(vadd(vadd(t[1], t[2]), t[3]), 1 / 3) for t in tri]
    order = Int32.(1:n)
    lo = V3[]; hi = V3[]; left = Int32[]; right = Int32[]; first = Int32[]; count = Int32[]
    function node!(a, b)
        push!(lo, (Inf, Inf, Inf)); push!(hi, (-Inf, -Inf, -Inf)); push!(left, 0); push!(right, 0); push!(first, 0); push!(count, 0)
        id = length(lo)
        blo = (Inf, Inf, Inf); bhi = (-Inf, -Inf, -Inf); clo = (Inf, Inf, Inf); chi = (-Inf, -Inf, -Inf)
        for k in a:b
            tlo, thi = tri_box(tri[order[k]]); c = cen[order[k]]
            blo = vmin(blo, tlo); bhi = vmax(bhi, thi); clo = vmin(clo, c); chi = vmax(chi, c)
        end
        lo[id] = blo; hi[id] = bhi
        if b - a + 1 <= leaf
            first[id] = a; count[id] = b - a + 1
            return id
        end
        ext = vsub(chi, clo); axis = ext[1] >= ext[2] && ext[1] >= ext[3] ? 1 : ext[2] >= ext[3] ? 2 : 3
        mid = (a + b) ÷ 2
        # the lower half must hold the smaller centroids: `partialsort!(v, k)` guarantees only the element at k
        # (Julia 1.13), so the range form is used
        partialsort!(view(order, a:b), 1:(mid - a + 1); by=i -> cen[i][axis])
        l = node!(a, mid); r = node!(mid + 1, b)
        left[id] = l; right[id] = r
        id
    end
    node!(1, n)
    BVH(lo, hi, left, right, first, count, order, tri)
end

"A traversal stack sized for any BVH of this package (depth is at most about 2 log2 n + leaf)."
new_stack() = Vector{Int32}(undef, 4096)

@inline function box_dist2(p::V3, lo::V3, hi::V3)
    d = 0.0
    @inbounds for a in 1:3
        t = p[a] < lo[a] ? lo[a] - p[a] : p[a] > hi[a] ? p[a] - hi[a] : 0.0
        d += t * t
    end
    d
end

"Squared distance from p to the triangle (a, b, c) and the closest point (Ericson, Real-Time Collision Detection 5.1.5)."
@inline function tri_closest(p::V3, a::V3, b::V3, c::V3)
    ab = vsub(b, a); ac = vsub(c, a); ap = vsub(p, a)
    d1 = vdot(ab, ap); d2 = vdot(ac, ap)
    (d1 <= 0 && d2 <= 0) && return a
    bp = vsub(p, b); d3 = vdot(ab, bp); d4 = vdot(ac, bp)
    (d3 >= 0 && d4 <= d3) && return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 && d1 >= 0 && d3 <= 0
        return vadd(a, vmul(ab, d1 / (d1 - d3)))
    end
    cp = vsub(p, c); d5 = vdot(ab, cp); d6 = vdot(ac, cp)
    (d6 >= 0 && d5 <= d6) && return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 && d2 >= 0 && d6 <= 0
        return vadd(a, vmul(ac, d2 / (d2 - d6)))
    end
    va = d3 * d6 - d5 * d4
    if va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0
        return vadd(b, vmul(vsub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))))
    end
    den = 1 / (va + vb + vc); v = vb * den; w = vc * den
    vadd(a, vadd(vmul(ab, v), vmul(ac, w)))
end

"""
Nearest surface point to p: returns (distance, triangle index, closest point). `bound` prunes the search; when
nothing is closer than `bound`, the triangle index is 0 and the distance is `bound`.
"""
function nearest(bvh::BVH, p::V3, stack::Vector{Int32}; bound::Float64=Inf)
    best = bound * bound; besttri = Int32(0); bestq = p
    sp = 1; stack[1] = 1
    @inbounds while sp > 0
        id = stack[sp]; sp -= 1
        box_dist2(p, bvh.lo[id], bvh.hi[id]) >= best && continue
        if bvh.left[id] == 0
            for k in bvh.first[id]:(bvh.first[id] + bvh.count[id] - 1)
                ti = bvh.order[k]; t = bvh.tri[ti]
                q = tri_closest(p, t[1], t[2], t[3]); d = vnorm2(vsub(p, q))
                if d < best
                    best = d; besttri = ti; bestq = q
                end
            end
        else
            l = bvh.left[id]; r = bvh.right[id]
            dl = box_dist2(p, bvh.lo[l], bvh.hi[l]); dr = box_dist2(p, bvh.lo[r], bvh.hi[r])
            if dl < dr
                dr < best && (sp += 1; stack[sp] = r); dl < best && (sp += 1; stack[sp] = l)
            else
                dl < best && (sp += 1; stack[sp] = l); dr < best && (sp += 1; stack[sp] = r)
            end
        end
    end
    sqrt(best), Int(besttri), bestq
end
nearest_distance(bvh::BVH, p::V3, stack::Vector{Int32}; bound::Float64=Inf) = nearest(bvh, p, stack; bound=bound)[1]

@inline function segment_box_overlap(a::V3, b::V3, lo::V3, hi::V3)
    # slab test of the segment a -> b against the box
    tmin = 0.0; tmax = 1.0
    @inbounds for k in 1:3
        d = b[k] - a[k]
        if abs(d) < 1e-300
            (a[k] < lo[k] || a[k] > hi[k]) && return false
        else
            t1 = (lo[k] - a[k]) / d; t2 = (hi[k] - a[k]) / d
            t1 > t2 && ((t1, t2) = (t2, t1))
            tmin = max(tmin, t1); tmax = min(tmax, t2)
            tmin > tmax && return false
        end
    end
    true
end

"""
First crossing of the segment a -> b with the surface: returns (q, triangle index) with q in (0, 1] the parametric
position of the first hit, or (1.0, 0) when the segment does not cross. Check the triangle index to distinguish
an endpoint hit from no hit. Hits exactly at a = 0 count as crossings
at q = 0 only if `include_start` is set (used by the leak check); ordinary cut links use q > 0.
"""
function first_crossing(bvh::BVH, a::V3, b::V3, stack::Vector{Int32}; include_start::Bool=false)
    dir = vsub(b, a)
    best = Inf; besttri = Int32(0)
    sp = 1; stack[1] = 1
    @inbounds while sp > 0
        id = stack[sp]; sp -= 1
        segment_box_overlap(a, b, bvh.lo[id], bvh.hi[id]) || continue
        if bvh.left[id] == 0
            for k in bvh.first[id]:(bvh.first[id] + bvh.count[id] - 1)
                ti = bvh.order[k]; t = bvh.tri[ti]
                e1 = vsub(t[2], t[1]); e2 = vsub(t[3], t[1]); h = vcross(dir, e2); det = vdot(e1, h)
                abs(det) < 1e-300 && continue
                inv = 1 / det; s = vsub(a, t[1]); u = inv * vdot(s, h)
                (u < -1e-12 || u > 1 + 1e-12) && continue
                qv = vcross(s, e1); v = inv * vdot(dir, qv)
                (v < -1e-12 || u + v > 1 + 1e-12) && continue
                tt = inv * vdot(e2, qv)
                lowest = include_start ? -1e-12 : 1e-12
                if tt >= lowest && tt <= 1 + 1e-12 && tt < best
                    best = max(tt, 0.0); besttri = ti
                end
            end
        else
            sp += 1; stack[sp] = bvh.left[id]; sp += 1; stack[sp] = bvh.right[id]
        end
    end
    min(best, 1.0), Int(besttri)
end

"Number of surface crossings of an axis ray from p."
function ray_crossings(bvh::BVH, p::V3, axis::Int, stack::Vector{Int32}, ::Val{D}) where D
    D in (-1,1) || throw(ArgumentError("ray direction must be -1 or +1"))
    n = 0; sp = 1; stack[1] = 1
    b1 = axis == 1 ? 2 : 1; b2 = axis == 3 ? 2 : 3
    dir = axis == 1 ? (Float64(D), 0.0, 0.0) : axis == 2 ? (0.0, Float64(D), 0.0) : (0.0, 0.0, Float64(D))
    @inbounds while sp > 0
        id = stack[sp]; sp -= 1
        lo = bvh.lo[id]; hi = bvh.hi[id]
        ((D==1 ? hi[axis] < p[axis] : lo[axis] > p[axis]) || p[b1] < lo[b1] || p[b1] > hi[b1] || p[b2] < lo[b2] || p[b2] > hi[b2]) && continue
        if bvh.left[id] == 0
            for k in bvh.first[id]:(bvh.first[id] + bvh.count[id] - 1)
                t = bvh.tri[bvh.order[k]]
                e1 = vsub(t[2], t[1]); e2 = vsub(t[3], t[1]); h = vcross(dir, e2); det = vdot(e1, h)
                abs(det) < 1e-300 && continue
                inv = 1 / det; s = vsub(p, t[1]); u = inv * vdot(s, h)
                (u < 0 || u > 1) && continue
                q = vcross(s, e1); v = inv * vdot(dir, q)
                (v < 0 || u + v > 1) && continue
                tt = inv * vdot(e2, q)
                tt > 1e-12 && (n += 1)
            end
        else
            sp += 1; stack[sp] = bvh.left[id]; sp += 1; stack[sp] = bvh.right[id]
        end
    end
    n
end
ray_crossings(bvh::BVH,p::V3,axis::Int,stack::Vector{Int32}) = ray_crossings(bvh,p,axis,stack,Val(1))

"Inside test by majority of the three axis-ray parities (robust to a single edge double count)."
function is_inside(bvh::BVH, p::V3, stack::Vector{Int32})
    (isodd(ray_crossings(bvh, p, 1, stack)) + isodd(ray_crossings(bvh, p, 2, stack)) + isodd(ray_crossings(bvh, p, 3, stack))) >= 2
end

"""Conservative parity-based cavity sealing for open or nonmanifold surfaces.

Return 1 for a sealing seed, 2 when a one-sided parity projection was rejected,
or 0 for an exterior point. Near the actual surface retain the existing majority
rule to seal small openings. Farther away require odd crossings in *both*
directions on at least two axes. This avoids turning the long projection of an
open panel into a detached solid. It does not make an open STL watertight.
"""
function cavity_seal_decision(bvh::BVH,p::V3,stack::Vector{Int32},near_surface::Bool)
    positives=ntuple(axis->isodd(ray_crossings(bvh,p,axis,stack)),3)
    count(identity,positives)>=2 || return UInt8(0)
    near_surface && return UInt8(1)
    pairs=0
    for axis in 1:3
        if positives[axis] && isodd(ray_crossings(bvh,p,axis,stack,Val(-1)))
            pairs+=1
            pairs>=2 && return UInt8(1)
        end
    end
    UInt8(2)
end
