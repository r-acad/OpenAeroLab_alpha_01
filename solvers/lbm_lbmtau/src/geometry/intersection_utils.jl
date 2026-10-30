using StaticArrays

"""
Shared triangle/AABB helpers used by both voxelization and AMR topology selection.

- `triangle_intersects_aabb` performs an exact SAT-style overlap test.
- `triangle_bounds_with_offset` computes translated triangle vertices and bounds once.
- `block_index_bounds_from_aabb` converts a physical AABB into block-index ranges.
"""

@inline function triangle_intersects_aabb(center::SVector{3,Float64},
                                          box_half_size::SVector{3,Float64},
                                          v1::SVector{3,Float64},
                                          v2::SVector{3,Float64},
                                          v3::SVector{3,Float64})
    tol = 1.001
    h = box_half_size * tol
    t1 = v1 - center
    t2 = v2 - center
    t3 = v3 - center

    if min(t1[1], t2[1], t3[1]) > h[1] || max(t1[1], t2[1], t3[1]) < -h[1]; return false; end
    if min(t1[2], t2[2], t3[2]) > h[2] || max(t1[2], t2[2], t3[2]) < -h[2]; return false; end
    if min(t1[3], t2[3], t3[3]) > h[3] || max(t1[3], t2[3], t3[3]) < -h[3]; return false; end

    f = (t2 - t1, t3 - t2, t1 - t3)
    u = (SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0))

    for i in 1:3
        for j in 1:3
            axis = cross(u[i], f[j])
            if dot(axis, axis) < 1e-10
                continue
            end
            p1 = dot(t1, axis)
            p2 = dot(t2, axis)
            p3 = dot(t3, axis)
            r = h[1] * abs(axis[1]) + h[2] * abs(axis[2]) + h[3] * abs(axis[3])
            if min(p1, min(p2, p3)) > r || max(p1, max(p2, p3)) < -r
                return false
            end
        end
    end
    return true
end

@inline function triangle_bounds_with_offset(tri, mesh_offset::SVector{3,Float64})
    v1 = SVector(tri[1]) + mesh_offset
    v2 = SVector(tri[2]) + mesh_offset
    v3 = SVector(tri[3]) + mesh_offset
    t_min = min.(v1, min.(v2, v3))
    t_max = max.(v1, max.(v2, v3))
    return v1, v2, v3, t_min, t_max
end

@inline function block_index_bounds_from_aabb(t_min::SVector{3,Float64},
                                              t_max::SVector{3,Float64},
                                              block_span::Float64,
                                              bx_max::Int, by_max::Int, bz_max::Int;
                                              margin::Float64=0.0)
    inv_block_span = 1.0 / block_span

    min_bx = floor(Int, (t_min[1] - margin) * inv_block_span) + 1
    min_by = floor(Int, (t_min[2] - margin) * inv_block_span) + 1
    min_bz = floor(Int, (t_min[3] - margin) * inv_block_span) + 1

    max_bx = floor(Int, (t_max[1] + margin) * inv_block_span) + 1
    max_by = floor(Int, (t_max[2] + margin) * inv_block_span) + 1
    max_bz = floor(Int, (t_max[3] + margin) * inv_block_span) + 1

    min_bx = max(1, min_bx); max_bx = min(max_bx, bx_max)
    min_by = max(1, min_by); max_by = min(max_by, by_max)
    min_bz = max(1, min_bz); max_bz = min(max_bz, bz_max)

    return min_bx, max_bx, min_by, max_by, min_bz, max_bz
end
