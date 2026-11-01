"""Bounded, deterministic triangle subdivision shared by case preflight and native geometry."""
module SurfaceSubdivision

export triangle_max_edge, visit_subtriangles, subdivision_count

@inline _edge2(a,b) = sum((a[i]-b[i])^2 for i in 1:3)
@inline _midpoint(a,b) = ntuple(i -> a[i] + (b[i]-a[i])/2, 3)
triangle_max_edge(t) = sqrt(max(_edge2(t[1],t[2]),_edge2(t[2],t[3]),_edge2(t[3],t[1])))

"""
Visit longest-edge children without allocating a complete child array. Return
`limit + 1` immediately if the output would exceed `limit`; the extra child is
not emitted. Child winding and the original planar geometry are retained.
"""
function visit_subtriangles(emit, tri::T, max_edge::Real; limit::Integer=50_000_000) where T
    isfinite(max_edge) && max_edge > 0 || throw(ArgumentError("Maximum triangle edge must be positive and finite"))
    limit >= 0 || throw(ArgumentError("Triangle limit must be nonnegative"))
    threshold=Float64(max_edge)^2
    pending=T[tri]; count=0
    while !isempty(pending)
        t=pop!(pending)
        a,b,c=t
        ab=_edge2(a,b);bc=_edge2(b,c);ca=_edge2(c,a)
        longest=max(ab,bc,ca)
        isfinite(longest) || throw(ArgumentError("Triangle coordinates are too large or nonfinite"))
        if longest <= threshold
            count+=1
            count > limit && return count
            emit(t)
        else
            if ab >= bc && ab >= ca
                m=_midpoint(a,b)
                (m==a || m==b) && throw(ArgumentError("Maximum triangle edge is smaller than coordinate precision"))
                push!(pending,(a,m,c),(m,b,c))
            elseif bc >= ca
                m=_midpoint(b,c)
                (m==b || m==c) && throw(ArgumentError("Maximum triangle edge is smaller than coordinate precision"))
                push!(pending,(b,m,a),(m,c,a))
            else
                m=_midpoint(c,a)
                (m==c || m==a) && throw(ArgumentError("Maximum triangle edge is smaller than coordinate precision"))
                push!(pending,(c,m,b),(m,a,b))
            end
        end
    end
    count
end

"Exact child count, capped at `limit + 1`, without allocating the refined mesh."
function subdivision_count(triangles,max_edge::Real;limit::Integer=50_000_000)
    count=0
    for tri in triangles
        count += visit_subtriangles(_ -> nothing,tri,max_edge;limit=limit-count)
        count > limit && return limit+1
    end
    count
end

end
