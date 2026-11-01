// # FILE: .\src\bouzidi_math.jl
using StaticArrays
using LinearAlgebra

"""
BOUZIDI_MATH.JL - Geometric Intersection Logic
"""

@inline function ray_triangle_intersection(origin::SVector{3,Float64}, 
                                           dir::SVector{3,Float64},
                                           v1::SVector{3,Float64}, 
                                           v2::SVector{3,Float64}, 
                                           v3::SVector{3,Float64})
    EPSILON = 1e-9
    
    edge1 = v2 - v1
    edge2 = v3 - v1
    h = cross(dir, edge2)
    a = dot(edge1, h)
    
    if abs(a) < EPSILON
        return (false, 0.0)
    end
    
    f = 1.0 / a
    s = origin - v1
    u = f * dot(s, h)
    
    # Two-sided Moller-Trumbore: the determinant sign does not cull a face.
    # Include shared edges/corners within floating-point roundoff so reversing
    # face winding cannot drop a grazing wall link on one side of an open sheet.
    barycentric_tolerance = 64 * eps(Float64)
    if u < -barycentric_tolerance || u > 1.0 + barycentric_tolerance
        return (false, 0.0)
    end
    
    q_vec = cross(s, edge1)
    v = f * dot(dir, q_vec)
    
    if v < -barycentric_tolerance || u + v > 1.0 + barycentric_tolerance
        return (false, 0.0)
    end
    
    t = f * dot(edge2, q_vec)
    
    if t > EPSILON
        return (true, t)
    else
        return (false, 0.0)
    end
end

"""
    compute_q_for_cell - Compute Q-values and identify intersecting Triangles
    Returns: (q_values, triangle_indices)
"""
function compute_q_for_cell(cell_center::SVector{3,Float64},
                            dx::Float64,
                            triangles::Vector,
                            mesh_offset::SVector{3,Float64},
                            cx::Vector{Int32},
                            cy::Vector{Int32},
                            cz::Vector{Int32},
                            original_indices::Vector{Int})
    
    n_dirs = length(cx)
    q_values = zeros(Float64, n_dirs)
    tri_indices = zeros(Int32, n_dirs)

    for k in 1:n_dirs
        dir = SVector(Float64(cx[k]), Float64(cy[k]), Float64(cz[k]))
        
        if cx[k] == 0 && cy[k] == 0 && cz[k] == 0
            q_values[k] = 0.0
            continue
        end
        
        dir_norm = dir / norm(dir)
        min_t = Inf
        best_tri_idx = -1
        
        for (i, tri) in enumerate(triangles)
            v1 = SVector{3,Float64}(tri[1]) + mesh_offset
            v2 = SVector{3,Float64}(tri[2]) + mesh_offset
            v3 = SVector{3,Float64}(tri[3]) + mesh_offset
            
            hit, t = ray_triangle_intersection(cell_center, dir_norm, v1, v2, v3)
            
            if hit && t < min_t
                min_t = t
                best_tri_idx = original_indices[i]
            end
        end
        
        if min_t < Inf
            # SYMMETRY-PLANE FIX (2026-06-04, primary, STL path): reject wall links
            # whose triangle hit dips to/below the y=0 mirror plane (the open
            # half-shell STL still has surface near the plane that produces spurious
            # plane-crossing links). Lets the free-slip y-mirror BC own those
            # directions instead of the Bouzidi no-slip overwrite. Same rationale as
            # the analytic-sphere path. Only under a true mirror plane.
            # Threshold 0.5*dx (not ~0): a plane fluid cell center sits at y=0.5*dx,
            # so near-plane (equator) wall links hit at small POSITIVE hit_y while
            # their cy-flipped mirror partners hit at hit_y<=0 and are rejected —
            # the BROKEN cy-PAIR that injects net y-momentum every step. Rejecting
            # the whole half-cell band keeps both halves of each near-plane pair out
            # of the no-slip Bouzidi overwrite, handing them to the free-slip mirror.
            hit_y = cell_center[2] + min_t * dir_norm[2]
            sym_plane = (@isdefined(HALF_Y_DOMAIN) && HALF_Y_DOMAIN) &&
                        !(@isdefined(GROUND_ANALYSIS) && GROUND_ANALYSIS)
            if sym_plane && hit_y <= 0.5 * dx
                continue
            end
            c_magnitude = norm(dir)
            q = min_t / (dx * c_magnitude)

            if q > 0.0 && q <= 1.0
                q_values[k] = q
                tri_indices[k] = Int32(best_tri_idx)
            end
        end
    end

    return q_values, tri_indices
end

@inline function ray_sphere_first_hit(origin::SVector{3,Float64},
                                      dir_unit::SVector{3,Float64},
                                      center::SVector{3,Float64},
                                      radius::Float64)
    oc = origin - center
    b = dot(oc, dir_unit)
    c = dot(oc, oc) - radius * radius
    disc = b * b - c
    if disc < 0.0
        return Inf
    end

    sqrt_disc = sqrt(disc)
    t1 = -b - sqrt_disc
    t2 = -b + sqrt_disc
    eps = 1.0e-10 * max(radius, 1.0)
    if t1 > eps
        return t1
    elseif t2 > eps
        return t2
    else
        return Inf
    end
end

"""
    compute_q_for_sphere_cell

Analytic line-sphere q-values for validation cases with spherical geometry.
This avoids STL ray-miss / voxel-mask inconsistencies while retaining the same
Bouzidi link convention as the general q-map path.
"""
function compute_q_for_sphere_cell(cell_center::SVector{3,Float64},
                                   dx::Float64,
                                   sphere_center::SVector{3,Float64},
                                   sphere_radius::Float64,
                                   cx::Vector{Int32},
                                   cy::Vector{Int32},
                                   cz::Vector{Int32})
    n_dirs = length(cx)
    q_values = zeros(Float64, n_dirs)
    tri_indices = zeros(Int32, n_dirs)

    for k in 1:n_dirs
        dir = SVector(Float64(cx[k]), Float64(cy[k]), Float64(cz[k]))
        c_mag = norm(dir)
        if c_mag <= 0.0
            continue
        end

        t = ray_sphere_first_hit(cell_center, dir / c_mag, sphere_center, sphere_radius)
        if isfinite(t)
            # SYMMETRY-PLANE FIX (2026-06-04, primary): for a body sitting ON the
            # y=0 mirror plane, the analytic q-map ray-casts the FULL sphere, so
            # links pointing DOWN toward the (non-existent) lower hemisphere get a
            # spurious finite hit at y≤0. The Bouzidi correction would then reflect
            # them with no-slip bounce-back (opp_arr) and OVERWRITE the specular
            # y-mirror BC → net y-momentum injection (the localized max|u_y| spike
            # the [SymPlane] probe measured). Threshold 0.5*dx (plane cell center is
            # at y=0.5*dx): reject the whole near-plane half-cell band so BOTH halves
            # of each near-equator cy-pair stay out of the no-slip Bouzidi overwrite
            # and are handled by the free-slip y-mirror BC. A bare ~0 threshold leaves
            # the broken-pair injection (the +cy survivor) that grows over time.
            # Only under a true mirror plane (HALF_Y_DOMAIN && !GROUND_ANALYSIS).
            hit_y = cell_center[2] + t * (dir[2] / c_mag)
            sym_plane = (@isdefined(HALF_Y_DOMAIN) && HALF_Y_DOMAIN) &&
                        !(@isdefined(GROUND_ANALYSIS) && GROUND_ANALYSIS)
            if sym_plane && hit_y <= 0.5 * dx
                continue
            end
            q = t / (dx * c_mag)
            if q > 0.0 && q <= 1.0
                q_values[k] = q
            end
        end
    end

    return q_values, tri_indices
end
