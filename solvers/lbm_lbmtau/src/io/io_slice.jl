using Printf
using Plots
isdefined(@__MODULE__, :configure_gr_windows_path!) || include(joinpath(@__DIR__,"..","..","..","..","tools","install","gr_runtime.jl"))
configure_gr_windows_path!(Plots)

@inline function _coord_to_cell_index(coord_phys::Float64, dx::Float64, n_cells::Int)
    # Clamp only at the domain edge. A previous `coord - 0.01*dx` bias nudged
    # exact boundary hits toward the negative side, which made slice ownership
    # and nearest-fallback sampling subtly favor lower-index cells.
    max_coord = max(0.0, prevfloat(Float64(n_cells) * dx))
    coord_clamped = clamp(coord_phys, 0.0, max_coord)
    return clamp(floor(Int, coord_clamped / dx) + 1, 1, n_cells)
end

@inline _cell_center_coord(global_idx::Int, dx::Float64) = (global_idx - 0.5) * dx

@inline function _coord_to_cell_index_unclamped(coord_phys::Float64, dx::Float64)
    return floor(Int, coord_phys / dx) + 1
end

@inline function _interp_cell_indices(coord_phys::Float64, dx::Float64, n_cells::Int)
    pos = clamp(coord_phys / dx + 0.5, 1.0, Float64(n_cells))
    i0 = clamp(floor(Int, pos), 1, n_cells)
    i1 = min(i0 + 1, n_cells)
    w = Float32(clamp(pos - i0, 0.0, 1.0))
    return i0, i1, w
end

@inline function _interp_cell_indices_unclamped(coord_phys::Float64, dx::Float64)
    pos = coord_phys / dx + 0.5
    i0 = floor(Int, pos)
    i1 = i0 + 1
    w = Float32(clamp(pos - i0, 0.0, 1.0))
    return i0, i1, w
end

@inline function _sample_level_cell_from_pointer(data, block_pointer::AbstractArray{Int32,3},
                                                 plane::Symbol,
                                                 primary_idx::Int, secondary_idx::Int, slice_idx::Int;
                                                 owner_only::Bool=false)
    bs = BLOCK_SIZE

    if primary_idx < 1 || secondary_idx < 1 || slice_idx < 1
        return (Int32(0), NaN32, NaN32, NaN32, NaN32, NaN32, NaN32, NaN32)
    end

    if plane == :xz
        gx = primary_idx
        gy = slice_idx
        gz = secondary_idx
    elseif plane == :xy
        gx = primary_idx
        gy = secondary_idx
        gz = slice_idx
    elseif plane == :yz
        gx = slice_idx
        gy = primary_idx
        gz = secondary_idx
    else
        error("Unsupported slice plane: $plane")
    end

    bx = (gx - 1) ÷ bs + 1
    by = (gy - 1) ÷ bs + 1
    bz = (gz - 1) ÷ bs + 1

    if bx < 1 || by < 1 || bz < 1 ||
       bx > size(block_pointer, 1) ||
       by > size(block_pointer, 2) ||
       bz > size(block_pointer, 3)
        return (Int32(0), NaN32, NaN32, NaN32, NaN32, NaN32, NaN32, NaN32)
    end

    b_idx = block_pointer[bx, by, bz]
    b_idx == 0 && return (Int32(0), NaN32, NaN32, NaN32, NaN32, NaN32, NaN32, NaN32)

    lx = (gx - 1) % bs + 1
    ly = (gy - 1) % bs + 1
    lz = (gz - 1) % bs + 1

    if owner_only && data.amr_covered !== nothing && data.amr_covered[lx, ly, lz, b_idx]
        return (Int32(0), NaN32, NaN32, NaN32, NaN32, NaN32, NaN32, NaN32)
    end

    if data.obs[lx, ly, lz, b_idx]
        return (Int32(1), NaN32, NaN32, NaN32, NaN32, NaN32, NaN32, NaN32)
    end

    rho_val = data.rho[lx, ly, lz, b_idx]
    vx_val = data.vel[lx, ly, lz, b_idx, 1]
    vy_val = data.vel[lx, ly, lz, b_idx, 2]
    vz_val = data.vel[lx, ly, lz, b_idx, 3]
    temp_val = data.temperature !== nothing ? data.temperature[lx, ly, lz, b_idx] : NaN32
    pinq_val = data.pi_neq_mag !== nothing ? data.pi_neq_mag[lx, ly, lz, b_idx] : NaN32
    shift_val = data.block_shift !== nothing ? Float32(data.block_shift[b_idx]) : 0.0f0

    if !(isfinite(rho_val) && isfinite(vx_val))
        return (Int32(0), NaN32, NaN32, NaN32, NaN32, NaN32, NaN32, NaN32)
    end

    return (Int32(2), rho_val, vx_val, vy_val, vz_val, temp_val, pinq_val, shift_val)
end

@inline function _sample_level_cell(data, plane::Symbol, primary_idx::Int, secondary_idx::Int, slice_idx::Int)
    return _sample_level_cell_from_pointer(data, data.sample_block_pointer,
                                           plane, primary_idx, secondary_idx, slice_idx)
end

# Sample the signed-distance-to-wall field at a single cell on a given level.
# Unlike _sample_level_cell, this does NOT short-circuit on obstacle cells —
# wall_dist is defined everywhere (negative inside the body, zero at the wall,
# positive in fluid) and the whole point of visualizing it is to see the
# sub-cell zero-crossing that the solver actually uses via Bouzidi q-maps.
# Returns NaN32 if the cell falls outside the level's block coverage or the
# level has no wall_dist array.
@inline function _sample_wall_dist_cell(data, plane::Symbol,
                                        primary_idx::Int, secondary_idx::Int, slice_idx::Int)
    data.wall_dist === nothing && return NaN32
    bs = BLOCK_SIZE
    if primary_idx < 1 || secondary_idx < 1 || slice_idx < 1
        return NaN32
    end
    if plane == :xz
        gx, gy, gz = primary_idx, slice_idx, secondary_idx
    elseif plane == :xy
        gx, gy, gz = primary_idx, secondary_idx, slice_idx
    elseif plane == :yz
        gx, gy, gz = slice_idx, primary_idx, secondary_idx
    else
        error("Unsupported slice plane: $plane")
    end
    bx = (gx - 1) ÷ bs + 1
    by = (gy - 1) ÷ bs + 1
    bz = (gz - 1) ÷ bs + 1
    bp = data.sample_block_pointer
    if bx < 1 || by < 1 || bz < 1 ||
       bx > size(bp, 1) || by > size(bp, 2) || bz > size(bp, 3)
        return NaN32
    end
    b_idx = bp[bx, by, bz]
    b_idx == 0 && return NaN32
    lx = (gx - 1) % bs + 1
    ly = (gy - 1) % bs + 1
    lz = (gz - 1) % bs + 1
    return data.wall_dist[lx, ly, lz, b_idx]
end

@inline function _sample_level_cell_owned(data, plane::Symbol,
                                          primary_idx::Int, secondary_idx::Int, slice_idx::Int)
    return _sample_level_cell_from_pointer(data, data.owner_block_pointer,
                                           plane, primary_idx, secondary_idx, slice_idx;
                                           owner_only=true)
end

@inline function _level_owns_point(data, plane::Symbol,
                                   x_phys::Float64, secondary_phys::Float64, slice_coord::Float64)
    bs = BLOCK_SIZE

    slice_idx = _coord_to_cell_index_unclamped(slice_coord, data.dx)
    primary_idx = _coord_to_cell_index_unclamped(x_phys, data.dx)
    secondary_idx = _coord_to_cell_index_unclamped(secondary_phys, data.dx)
    status = _sample_level_cell_owned(data, plane, primary_idx, secondary_idx, slice_idx)[1]
    return status != Int32(0)
end

@inline function _level_covers_point(data, plane::Symbol,
                                     x_phys::Float64, secondary_phys::Float64, slice_coord::Float64)
    slice_idx = _coord_to_cell_index_unclamped(slice_coord, data.dx)
    primary_idx = _coord_to_cell_index_unclamped(x_phys, data.dx)
    secondary_idx = _coord_to_cell_index_unclamped(secondary_phys, data.dx)
    status = _sample_level_cell(data, plane, primary_idx, secondary_idx, slice_idx)[1]
    return status != Int32(0)
end

"""
    _sample_level_slice(...)

Trilinear-in-3D sample of a level's macroscopic field at the slice plane.
Wrapper around `_sample_level_slice_bilinear_layer`: calls it at the two
slice-normal cell layers bracketing `slice_coord` and linearly blends the
results.

Why this matters: each AMR level has its own `dx`. If the slice-normal axis
is snap-to-cell, level L1 (dx=0.089 m) and level L4 (dx=0.0112 m) sample at
different actual physical Y positions when the user requests Y=0.0056 (L1
center 0.0445, L4 center 0.0056). When the renderer composites pixels from
different levels into one image, an AMR boundary becomes a step in actual
sampled Y — a fake rectangle outline that has nothing to do with the solver
state. Doing trilinear here makes every level sample the same physical
slice plane.
"""
@inline function _sample_level_slice(data, plane::Symbol,
                                     x_phys::Float64, secondary_phys::Float64, slice_coord::Float64;
                                     annotate_owned_frontier::Bool=false)
    iy0, iy1, wy_frac = _interp_cell_indices_unclamped(slice_coord, data.dx)

    s0 = _sample_level_slice_bilinear_layer(data, plane, x_phys, secondary_phys, iy0;
                                            annotate_owned_frontier=annotate_owned_frontier)
    s1 = _sample_level_slice_bilinear_layer(data, plane, x_phys, secondary_phys, iy1;
                                            annotate_owned_frontier=annotate_owned_frontier)

    s0_ok = s0[1] == Int32(2)
    s1_ok = s1[1] == Int32(2)

    if s0_ok && s1_ok
        w0 = 1.0f0 - wy_frac
        w1 = wy_frac
        # Mode priority: prefer 2 (clean bilinear) over 4 (frontier-tagged) over
        # 5 (touches domain edge); 3 (nearest-cell fallback) cannot be safely
        # blended in the slice-normal direction so degrade to it if either
        # layer is mode 3.
        mode_out =
            (s0[8] == Int32(3) || s1[8] == Int32(3)) ? Int32(3) :
            (s0[8] == Int32(5) || s1[8] == Int32(5)) ? Int32(5) :
            (s0[8] == Int32(4) || s1[8] == Int32(4)) ? Int32(4) : Int32(2)
        return (
            Int32(2),
            w0 * s0[2] + w1 * s1[2],
            w0 * s0[3] + w1 * s1[3],
            w0 * s0[4] + w1 * s1[4],
            w0 * s0[5] + w1 * s1[5],
            w0 * s0[6] + w1 * s1[6],
            w0 * s0[7] + w1 * s1[7],
            mode_out,
            w0 * s0[9] + w1 * s1[9],
        )
    end

    # Only one layer is fluid (or both are obstacle/no-data): defer to whichever
    # layer actually has data, leaving the existing edge-case fallback logic in
    # `_sample_level_slice_bilinear_layer` to decide the mode tag.
    return s0_ok ? s0 : s1
end

@inline function _sample_level_slice_bilinear_layer(data, plane::Symbol,
                                                    x_phys::Float64, secondary_phys::Float64, slice_idx::Int;
                                                    annotate_owned_frontier::Bool=false)
    bs = BLOCK_SIZE
    n_primary = plane == :yz ? data.grid_dim_y * bs : data.grid_dim_x * bs
    n_secondary = plane == :xy ? data.grid_dim_y * bs : data.grid_dim_z * bs
    n_slice = plane == :xz ? data.grid_dim_y * bs :
              plane == :xy ? data.grid_dim_z * bs :
              plane == :yz ? data.grid_dim_x * bs :
              error("Unsupported slice plane: $plane")

    ix0, ix1, wx = _interp_cell_indices_unclamped(x_phys, data.dx)
    is0, is1, ws = _interp_cell_indices_unclamped(secondary_phys, data.dx)

    s00 = _sample_level_cell(data, plane, ix0, is0, slice_idx)
    s10 = _sample_level_cell(data, plane, ix1, is0, slice_idx)
    s01 = _sample_level_cell(data, plane, ix0, is1, slice_idx)
    s11 = _sample_level_cell(data, plane, ix1, is1, slice_idx)

    w00 = (1.0f0 - wx) * (1.0f0 - ws)
    w10 = wx * (1.0f0 - ws)
    w01 = (1.0f0 - wx) * ws
    w11 = wx * ws

    samples = ((s00, w00), (s10, w10), (s01, w01), (s11, w11))
    fluid_count = 0
    weight_sum = 0.0f0
    rho_acc = 0.0f0
    vx_acc = 0.0f0
    vy_acc = 0.0f0
    vz_acc = 0.0f0
    temp_acc = 0.0f0
    pinq_acc = 0.0f0
    shift_acc = 0.0f0
    any_obstacle = false

    for (sample, weight) in samples
        if sample[1] == Int32(2)
            fluid_count += 1
            weight_sum += weight
            rho_acc += weight * sample[2]
            vx_acc += weight * sample[3]
            vy_acc += weight * sample[4]
            vz_acc += weight * sample[5]
            temp_acc += weight * sample[6]
            pinq_acc += weight * sample[7]
            shift_acc += weight * sample[8]
        elseif sample[1] == Int32(1)
            any_obstacle = true
        end
    end

    # Full 4-cell bilinear — preferred path. Status code 2 in position 1
    # *and* in position 8 (the "sample mode" marker read by the caller).
    #
    # If the owning finest level reaches this bilinear sample only by using
    # covered frontier support, keep the smooth bilinear value but tag it as
    # mode 4 for diagnostics. The previous "strict owned only" downgrade to
    # nearest-cell sampling printed the whole AMR box into schlieren.
    if fluid_count == 4
        sample_mode = Int32(2)
        if annotate_owned_frontier
            # Frontier tag: set mode 4 if ANY of the 3×3 cells around the pixel's
            # primary/secondary position is not owned by this level. The bilinear
            # stencil itself is [ix0..ix1] × [is0..is1] = [i,i+1]×[j,j+1], which
            # is one-sided (always reaches in +primary/+secondary direction).
            # Using just those 4 cells makes the frontier visible only on the
            # "lower" face of each AMR box (where the stencil crosses into the
            # not-owned parent region) and invisible on the "upper" face (where
            # the stencil reaches further into the owned region and never sees
            # the not-owned block below). Extending to the full 3×3 neighborhood
            # restores top/bottom and left/right symmetry.
            any_not_owned = false
            for pi in (ix0 - 1):(ix1 + 1), pj in (is0 - 1):(is1 + 1)
                s = _sample_level_cell_owned(data, plane, pi, pj, slice_idx)
                if s[1] != Int32(2)
                    any_not_owned = true
                    break
                end
            end
            if any_not_owned
                sample_mode = Int32(4)
            end
        end
        inv_w = 1.0f0 / max(weight_sum, 1.0f-6)
        return (
            Int32(2),
            rho_acc * inv_w,
            vx_acc * inv_w,
            vy_acc * inv_w,
            vz_acc * inv_w,
            temp_acc * inv_w,
            pinq_acc * inv_w,
            sample_mode,
            shift_acc * inv_w,
        )
    end

    touches_domain = ix0 < 1 || ix1 > n_primary || is0 < 1 || is1 > n_secondary

    # Absolute domain edge: keep the interpolated sample continuous by
    # renormalizing the available in-domain weights instead of snapping to the
    # nearest cell. The old nearest-neighbor fallback leaves a one-pixel
    # perimeter ring, and differentiating that ring in the schlieren export
    # prints the same "comb teeth" pattern we previously chased at AMR
    # interfaces.
    if touches_domain && fluid_count > 0
        inv_w = 1.0f0 / max(weight_sum, 1.0f-6)
        return (
            Int32(2),
            rho_acc * inv_w,
            vx_acc * inv_w,
            vy_acc * inv_w,
            vz_acc * inv_w,
            temp_acc * inv_w,
            pinq_acc * inv_w,
            Int32(5),
            shift_acc * inv_w,
        )
    end

    # Partial interior stencil (fluid_count ∈ {0..3}): the bilinear footprint
    # really is incomplete on this level, so fall back to the nearest in-level
    # fluid sample. We only use this when the sampled level itself cannot
    # produce a continuous bilinear value; covered-frontier support is already
    # handled above via mode 4.
    nearest_i = _coord_to_cell_index(x_phys, data.dx, n_primary)
    nearest_s = _coord_to_cell_index(secondary_phys, data.dx, n_secondary)
    nearest = _sample_level_cell(data, plane, nearest_i, nearest_s, slice_idx)
    if nearest[1] == Int32(2)
        return (
            nearest[1],
            nearest[2],
            nearest[3],
            nearest[4],
            nearest[5],
            nearest[6],
            nearest[7],
            Int32(3),
            nearest[8],
        )
    end
    return (
        nearest[1],
        nearest[2],
        nearest[3],
        nearest[4],
        nearest[5],
        nearest[6],
        nearest[7],
        nearest[1],
        nearest[8],
    )
end

function _plane_has_obstacle(data, valid_blocks, plane::Symbol, slice_idx::Int)
    bs = BLOCK_SIZE

    if plane == :xz
        by_slice = (slice_idx - 1) ÷ bs + 1
        ly = (slice_idx - 1) % bs + 1

        for (bx, by, bz) in valid_blocks
            by == by_slice || continue
            b_idx = data.block_pointer[bx, by, bz]
            b_idx == 0 && continue

            for lz in 1:bs, lx in 1:bs
                data.obs[lx, ly, lz, b_idx] && return true
            end
        end

        return false
    elseif plane == :xy
        bz_slice = (slice_idx - 1) ÷ bs + 1
        lz = (slice_idx - 1) % bs + 1

        for (bx, by, bz) in valid_blocks
            bz == bz_slice || continue
            b_idx = data.block_pointer[bx, by, bz]
            b_idx == 0 && continue

            for ly in 1:bs, lx in 1:bs
                data.obs[lx, ly, lz, b_idx] && return true
            end
        end

        return false
    elseif plane == :yz
        bx_slice = div(slice_idx - 1, bs) + 1
        lx = (slice_idx - 1) % bs + 1

        for (bx, by, bz) in valid_blocks
            bx == bx_slice || continue
            b_idx = data.block_pointer[bx, by, bz]
            b_idx == 0 && continue

            for lz in 1:bs, ly in 1:bs
                data.obs[lx, ly, lz, b_idx] && return true
            end
        end

        return false
    end

    error("Unsupported slice plane: $plane")
end

function _pick_slice_plane(level_data, valid_blocks;
                           plane_override::Union{Nothing,Symbol}=nothing,
                           y_override::Union{Nothing,Float64}=nothing,
                           x_override::Union{Nothing,Float64}=nothing,
                           z_override::Union{Nothing,Float64}=nothing,   # lbm-io-opts (27 Oct 2026): explicit XY plane
                           normal_clamp::Symbol=:body)                   # lbm-io-opts: :domain for explicit planes
    params = get_domain_params()
    fine = level_data[end]
    coarsest = level_data[1]

    placed_center = (
        params.mesh_center[1] + params.mesh_offset[1],
        params.mesh_center[2] + params.mesh_offset[2],
        params.mesh_center[3] + params.mesh_offset[3],
    )
    placed_min = (
        params.mesh_min[1] + params.mesh_offset[1],
        params.mesh_min[2] + params.mesh_offset[2],
        params.mesh_min[3] + params.mesh_offset[3],
    )
    placed_max = (
        params.mesh_max[1] + params.mesh_offset[1],
        params.mesh_max[2] + params.mesh_offset[2],
        params.mesh_max[3] + params.mesh_offset[3],
    )

    requested_plane = plane_override === nothing ? (GROUND_ANALYSIS ? :xy : :xz) : plane_override

    if requested_plane == :xy
        plane = :xy
        plane_label = "XY"
        normal_label = "Z"
        axis_labels = ("X", "Y")
        target_coord = z_override !== nothing ? z_override : placed_center[3]
        coord_lo = placed_min[3]
        coord_hi = placed_max[3]
        n_cells_fine = fine.grid_dim_z * BLOCK_SIZE
    elseif requested_plane == :xz
        plane = :xz
        plane_label = "XZ"
        normal_label = "Y"
        axis_labels = ("X", "Z")
        # y_override lets caller diagnose at a custom Y (e.g. away from wing BL).
        # For HALF_Y_DOMAIN (symmetric_analysis), default Y target was
        # `placed_min[2] + fine.dx * 0.5` — the FINEST level's first cell center.
        # That's representable only on the finest level; coarser levels have
        # their first cell center at `coarsest.dx * 0.5` (e.g. 0.045 m at
        # SR=224 vs 0.006 m at L4). The slice compositor then renders L4 at
        # Y_fine, L3 at 2*Y_fine, L2 at 4*Y_fine, L1 at 8*Y_fine and prints a
        # step at each AMR boundary where the wing's swept cross-section
        # places different geometry. Default to the COARSEST cell center so
        # all levels sample at the same physical Y.
        target_coord = y_override !== nothing ? y_override :
                       (HALF_Y_DOMAIN ? (placed_min[2] + coarsest.dx * 0.5) : placed_center[2])
        coord_lo = placed_min[2]
        coord_hi = placed_max[2]
        n_cells_fine = fine.grid_dim_y * BLOCK_SIZE
    elseif requested_plane == :yz
        plane = :yz
        plane_label = "YZ"
        normal_label = "X"
        axis_labels = ("Y", "Z")
        target_coord = x_override !== nothing ? x_override : placed_center[1]
        coord_lo = placed_min[1]
        coord_hi = placed_max[1]
        n_cells_fine = fine.grid_dim_x * BLOCK_SIZE
    else
        error("Unsupported slice plane: $requested_plane")
    end

    lo_coord = clamp(coord_lo + fine.dx * 1.0e-3, 0.0, n_cells_fine * fine.dx)
    hi_coord = clamp(coord_hi - fine.dx * 1.0e-3, 0.0, n_cells_fine * fine.dx)
    hi_coord = max(lo_coord, hi_coord)

    lo_idx = _coord_to_cell_index(lo_coord, fine.dx, n_cells_fine)
    hi_idx = _coord_to_cell_index(hi_coord, fine.dx, n_cells_fine)
    # lbm-io-opts (27 Oct 2026): an explicit plane of advanced.slice.planes (normal_clamp=:domain)
    # may lie anywhere in the domain (e.g. a wake plane); the legacy default keeps the body clamp.
    if normal_clamp === :domain && ((plane == :xz && y_override !== nothing) ||
                                    (plane == :yz && x_override !== nothing) ||
                                    (plane == :xy && z_override !== nothing))
        lo_idx = 1
        hi_idx = n_cells_fine
    end
    target_idx = clamp(_coord_to_cell_index(target_coord, fine.dx, n_cells_fine), lo_idx, hi_idx)

    actual_idx = target_idx
    intersects_geometry = _plane_has_obstacle(fine, valid_blocks[end], plane, actual_idx)
    used_fallback = false

    explicit_normal_override =
        (plane == :xz && y_override !== nothing) ||
        (plane == :yz && x_override !== nothing) ||
        (plane == :xy && z_override !== nothing)

    # Only seek obstacle-intersecting plane if no explicit normal-coordinate
    # override was requested. With override the user specifically wants to see
    # that plane even if it misses geometry.
    if !intersects_geometry && !explicit_normal_override
        max_offset = max(actual_idx - lo_idx, hi_idx - actual_idx)
        for offset in 1:max_offset
            lower_idx = actual_idx - offset
            if lower_idx >= lo_idx && _plane_has_obstacle(fine, valid_blocks[end], plane, lower_idx)
                actual_idx = lower_idx
                intersects_geometry = true
                used_fallback = true
                break
            end

            upper_idx = actual_idx + offset
            if upper_idx <= hi_idx && _plane_has_obstacle(fine, valid_blocks[end], plane, upper_idx)
                actual_idx = upper_idx
                intersects_geometry = true
                used_fallback = true
                break
            end
        end
    end

    return (
        plane=plane,
        plane_label=plane_label,
        normal_label=normal_label,
        axis_labels=axis_labels,
        target_coord=target_coord,
        coord=_cell_center_coord(actual_idx, fine.dx),
        slice_index=actual_idx,
        intersects_geometry=intersects_geometry,
        used_fallback=used_fallback,
    )
end

"""
    extract_slice_data(t_step, grids, backend)

Extract a geometry-aware 2D slice from the multi-level grid hierarchy.

- `ground` mode: exports an `XY` slice at the STL centerline in `Z`
- `free-flight` and `mirror` symmetry: exports an `XZ` slice through the body in `Y`

If the requested plane misses the geometry, the exporter nudges to the nearest
plane inside the STL bounds that intersects obstacle voxels.
Returns `(img_w, img_h, rho_2d, vel_2d, obs_2d, temp_2d, pinq_mag_2d, owner_level_2d, sample_mode_2d, local_shift_2d, wall_dist_2d, slice_meta)`.
"""
function extract_slice_data(t_step::Int, grids, backend;
                            cx_gpu=nothing, cy_gpu=nothing, cz_gpu=nothing, w_gpu=nothing,
                            plane_override::Union{Nothing,Symbol}=nothing,
                            y_override::Union{Nothing,Float64}=nothing,
                            x_override::Union{Nothing,Float64}=nothing,
                            z_override::Union{Nothing,Float64}=nothing,   # lbm-io-opts (27 Oct 2026)
                            normal_clamp::Symbol=:body)
    # Debug: LUDWIG_SLICE_MAX_LEVEL env var caps the finest level the slicer renders.
    # Set to 2 to hide Level 3 data and show what the parent (L2) has in that region.
    # Useful for diagnosing whether a deficit is in the finest level or inherited from parent.
    max_level_env = get(ENV, "LUDWIG_SLICE_MAX_LEVEL", "")
    max_level_cap = isempty(max_level_env) ? length(grids) :
                    min(length(grids), max(1, tryparse(Int, max_level_env) === nothing ? length(grids) : parse(Int, max_level_env)))
    n_levels = max_level_cap
    if max_level_cap < length(grids)
        @info "[Slice] DEBUG: capped at level $max_level_cap of $(length(grids))"
    end
    bs = BLOCK_SIZE

    compute_pineq = COMPUTE_PI_NEQ && cx_gpu !== nothing && cy_gpu !== nothing &&
                    cz_gpu !== nothing && w_gpu !== nothing
    shift_x_f = Float32(LATTICE_SHIFT_X)
    shift_y_f = Float32(LATTICE_SHIFT_Y)
    shift_z_f = Float32(LATTICE_SHIFT_Z)

    level_data = Vector{NamedTuple}(undef, n_levels)
    ddf_active = DDF_ENABLED
    for lvl in 1:n_levels
        level = grids[lvl]
        rho_cpu = Array(level.rho)
        vel_cpu = lvl == 1 ? (iseven(t_step) ? Array(level.vel_temp) : Array(level.vel)) : Array(level.vel)
        obs_cpu = Array(level.obstacle)
        amr_covered_cpu = length(level.amr_covered) > 1 ? Array(level.amr_covered) : nothing
        temp_cpu = (ddf_active && has_thermal(level)) ? Array(level.temperature) : nothing
        # Wall signed-distance field — continuous (sub-cell smooth) SDF computed
        # from STL triangles at meshing time. Negative inside the body, zero at
        # the actual wall, positive in fluid. Used by the :wall_dist diagnostic
        # slice to show the geometry as the solver numerically sees it (the
        # zero contour is sub-cell-accurate, while the obstacle field is only
        # cell-resolved).
        wall_dist_cpu = (level.wall_dist !== nothing && length(level.wall_dist) > 1) ?
                        Array(level.wall_dist) : nothing

        pinq_mag_cpu = nothing
        # ET (Esoteric-Pull) single-buffer path: level.f is PERMUTED (Esoteric layout)
        # (ET/AA-pattern single-buffer path removed 2026-06-24 with the D3Q19 retirement.)
        if compute_pineq
            f_current = (lvl == 1 && iseven(t_step)) ? level.f_temp : level.f
            pi_neq_gpu = compute_pi_neq_for_level!(level, f_current,
                                                   cx_gpu, cy_gpu, cz_gpu, w_gpu, backend;
                                                   shift_x=shift_x_f, shift_y=shift_y_f, shift_z=shift_z_f)
            pi_neq_cpu = Array(pi_neq_gpu)
            tau_lvl = Float32(level.tau)
            denom = tau_lvl - 1.0f0
            pre_scale = (abs(denom) < 1.0f-3) ? 1.0f0 : (tau_lvl / denom)
            abs_scale = abs(pre_scale)
            n_blocks_lvl = size(pi_neq_cpu, 4)
            pinq_mag_cpu = Array{Float32}(undef, bs, bs, bs, n_blocks_lvl)
            @inbounds for b in 1:n_blocks_lvl, lz in 1:bs, ly in 1:bs, lx in 1:bs
                pxx = pi_neq_cpu[lx, ly, lz, b, 1]
                pyy = pi_neq_cpu[lx, ly, lz, b, 2]
                pzz = pi_neq_cpu[lx, ly, lz, b, 3]
                pxy = pi_neq_cpu[lx, ly, lz, b, 4]
                pxz = pi_neq_cpu[lx, ly, lz, b, 5]
                pyz = pi_neq_cpu[lx, ly, lz, b, 6]
                mag_post = sqrt(pxx*pxx + pyy*pyy + pzz*pzz +
                                2.0f0*(pxy*pxy + pxz*pxz + pyz*pyz))
                pinq_mag_cpu[lx, ly, lz, b] = abs_scale * mag_post
            end
        end

        level_data[lvl] = (
            rho=rho_cpu,
            vel=vel_cpu,
            obs=obs_cpu,
            wall_dist=wall_dist_cpu,
            amr_covered=amr_covered_cpu,
            temperature=temp_cpu,
            pi_neq_mag=pinq_mag_cpu,
            block_shift=Array(level.block_shift),
            grid_dim_x=level.grid_dim_x,
            grid_dim_y=level.grid_dim_y,
            grid_dim_z=level.grid_dim_z,
            block_pointer=Array(level.block_pointer),
            # Keep all active blocks available for sampling so coarser levels can
            # still serve as bilinear donor fields near fine-level frontiers.
            sample_block_pointer=Array(level.block_pointer),
            owner_block_pointer=zeros(Int32, size(level.block_pointer)),
            dx=level.dx,
            level_id=Int32(lvl),
        )
    end

    level_block_sets = [Set{Tuple{Int,Int,Int}}() for _ in 1:n_levels]
    for lvl in 1:n_levels
        for block in grids[lvl].active_block_coords
            push!(level_block_sets[lvl], block)
        end
    end

    valid_blocks = [Set{Tuple{Int,Int,Int}}() for _ in 1:n_levels]
    for lvl in 1:n_levels
        next_set = lvl < n_levels ? level_block_sets[lvl + 1] : nothing
        for (bx, by, bz) in level_block_sets[lvl]
            should_use = true
            if next_set !== nothing
                children = 0
                for dbz in 0:1, dby in 0:1, dbx in 0:1
                    if (2 * bx - 1 + dbx, 2 * by - 1 + dby, 2 * bz - 1 + dbz) in next_set
                        children += 1
                    end
                end
                should_use = children != 8
            end
            should_use && push!(valid_blocks[lvl], (bx, by, bz))
        end
    end

    for lvl in 1:n_levels
        owner_ptr = zeros(Int32, size(level_data[lvl].block_pointer))
        for (bx, by, bz) in valid_blocks[lvl]
            owner_ptr[bx, by, bz] = level_data[lvl].block_pointer[bx, by, bz]
        end
        level_data[lvl] = merge(level_data[lvl], (owner_block_pointer=owner_ptr,))
    end

    slice_meta = _pick_slice_plane(level_data, valid_blocks;
                                   plane_override=plane_override,
                                   y_override=y_override,
                                   x_override=x_override,
                                   z_override=z_override,
                                   normal_clamp=normal_clamp)

    coarsest = level_data[1]
    dx_coarse = coarsest.dx
    dx_fine = level_data[n_levels].dx

    domain_x = coarsest.grid_dim_x * bs * dx_coarse
    domain_y = coarsest.grid_dim_y * bs * dx_coarse
    domain_z = coarsest.grid_dim_z * bs * dx_coarse
    domain_primary = slice_meta.plane == :yz ? domain_y : domain_x
    domain_secondary = slice_meta.plane == :xy ? domain_y : domain_z

    full_w = round(Int, domain_primary / dx_fine)
    full_h = round(Int, domain_secondary / dx_fine)

    max_px = 1024
    downsample = max(1, max(full_w, full_h) ÷ max_px)
    img_w = full_w ÷ downsample
    img_h = full_h ÷ downsample
    px_size = dx_fine * downsample

    rho_2d = fill(NaN32, img_w, img_h)
    vel_2d = fill(NaN32, img_w, img_h, 3)
    temp_2d = fill(NaN32, img_w, img_h)
    pinq_mag_2d = fill(NaN32, img_w, img_h)
    obs_2d = falses(img_w, img_h)
    owner_level_2d = fill(0.0f0, img_w, img_h)
    sample_mode_2d = fill(0.0f0, img_w, img_h)
    local_shift_2d = fill(NaN32, img_w, img_h)

    for py in 1:img_h, px in 1:img_w
        x_phys = (px - 0.5) * px_size
        secondary_phys = (py - 0.5) * px_size

        finest_sample = nothing
        coarser_sample = nothing
        finest_level_id = Int32(0)
        owner_set = false
        # Nearest-neighbor fallback, used only if *no* level produces a
        # full bilinear sample at this pixel. Accepting a nearest sample
        # directly from the first level that offers one produces a visible
        # 1-cell-wide staircase band at every refinement-box edge (the
        # "comb teeth"), because the nearest cell at the frontier is a
        # halo/ghost cell that disagrees with the interior bilinear by
        # O(dx) in each direction. Preferring coarser bilinear over finer
        # nearest eliminates the band.
        fallback_sample = nothing
        fallback_level_id = Int32(0)

        for lvl in n_levels:-1:1
            owns_point = _level_owns_point(level_data[lvl], slice_meta.plane,
                                           x_phys, secondary_phys, slice_meta.coord)
            covers_point = owns_point || _level_covers_point(level_data[lvl], slice_meta.plane,
                                                             x_phys, secondary_phys, slice_meta.coord)
            if !covers_point
                continue
            end
            if !owner_set
                owner_level_2d[px, py] = Float32(level_data[lvl].level_id)
                owner_set = true
            end
            sample = _sample_level_slice(level_data[lvl], slice_meta.plane,
                                         x_phys, secondary_phys, slice_meta.coord;
                                         annotate_owned_frontier=owns_point)
            status = sample[1]
            mode = sample[8]

            if status == Int32(0)
                continue
            elseif status == Int32(1)
                # A coarser donor level may mark a cell as obstacle even when a finer
                # level resolves fluid there. Only let the owning level decide whether
                # this slice pixel is truly obstacle.
                if owns_point
                    obs_2d[px, py] = true
                    sample_mode_2d[px, py] = 1.0f0
                    rho_2d[px, py] = NaN32
                    vel_2d[px, py, 1] = NaN32
                    vel_2d[px, py, 2] = NaN32
                    vel_2d[px, py, 3] = NaN32
                    temp_2d[px, py] = NaN32
                    pinq_mag_2d[px, py] = NaN32
                    local_shift_2d[px, py] = NaN32
                    break
                end
                continue
            elseif mode == Int32(3)
                # Nearest-neighbor on this level; keep it as a fallback
                # only from the owning level, while still searching covered
                # coarser levels for a smoother bilinear donor field.
                if owns_point && fallback_sample === nothing
                    fallback_sample = sample
                    fallback_level_id = level_data[lvl].level_id
                end
                continue
            else
                # Full bilinear (mode == Int32(2))
                if finest_sample === nothing
                    finest_sample = sample
                    finest_level_id = level_data[lvl].level_id
                else
                    coarser_sample = sample
                    break
                end
            end
        end

        # No full-bilinear level owned this pixel — fall back to the
        # finest nearest sample we saw, so boundary pixels still render.
        if finest_sample === nothing && fallback_sample !== nothing && !obs_2d[px, py]
            finest_sample = fallback_sample
            finest_level_id = fallback_level_id
        end

        if finest_sample !== nothing && !obs_2d[px, py]
            # Render each pixel from exactly one level (the finest that
            # can provide it). The previous 0.85 · L_fine + 0.15 · L_coarser
            # cosmetic blend was dropped because the pair of levels used
            # in the blend changes discontinuously across AMR interfaces
            # (L_n+L_{n-1} inside a refinement box, L_{n-1}+L_{n-2} just
            # outside) — and that formula change itself produces a
            # ~1e-4 density step that the targeted interface smoother
            # below then has to chase. Measured in step-200 CSVs: the
            # blend contributed ~35 % of the visible seam magnitude.
            # Removing it exposes the genuine physics/prolongation
            # residual (~0.65 × 1e-4 = 6.5e-5) for the next fix.
            obs_2d[px, py] = false
            sample_mode_2d[px, py] = Float32(finest_sample[8])
            rho_2d[px, py] = finest_sample[2]
            vel_2d[px, py, 1] = finest_sample[3]
            vel_2d[px, py, 2] = finest_sample[4]
            vel_2d[px, py, 3] = finest_sample[5]
            temp_2d[px, py] = finest_sample[6]
            pinq_mag_2d[px, py] = finest_sample[7]
            local_shift_2d[px, py] = finest_sample[9]
        end
    end

    # Sample-mode is a diagnostic overlay. Build a centered AMR frontier band
    # from the final per-pixel owner map rather than from the bilinear stencil's
    # one-sided donor footprint. This keeps +x/-x and +z/-z faces visually and
    # logically symmetric: mode 4 means "near an AMR owner transition", while
    # mode 5 is reserved for true physical/domain edges.
    amr_frontier_2d = falses(img_w, img_h)
    for py in 1:img_h, px in 1:img_w
        owner_here = owner_level_2d[px, py]
        owner_here == 0.0f0 && continue

        if px < img_w
            owner_r = owner_level_2d[px + 1, py]
            if owner_r != 0.0f0 && owner_r != owner_here
                amr_frontier_2d[px, py] = true
                amr_frontier_2d[px + 1, py] = true
            end
        end
        if py < img_h
            owner_u = owner_level_2d[px, py + 1]
            if owner_u != 0.0f0 && owner_u != owner_here
                amr_frontier_2d[px, py] = true
                amr_frontier_2d[px, py + 1] = true
            end
        end
    end

    # Dilate the owner-transition band by one pixel on both sides. The band is
    # diagnostic-only; broadening it removes the misleading impression that only
    # the lower/right face needed reduced support.
    amr_frontier_dilated = copy(amr_frontier_2d)
    for py in 1:img_h, px in 1:img_w
        amr_frontier_2d[px, py] || continue
        for dj in -1:1, di in -1:1
            qx = px + di
            qy = py + dj
            if qx >= 1 && qx <= img_w && qy >= 1 && qy <= img_h &&
               owner_level_2d[qx, qy] != 0.0f0
                amr_frontier_dilated[qx, qy] = true
            end
        end
    end

    for py in 1:img_h, px in 1:img_w
        obs_2d[px, py] && continue
        if px == 1 || px == img_w || py == 1 || py == img_h
            sample_mode_2d[px, py] = max(sample_mode_2d[px, py], 5.0f0)
            continue
        end

        touches_obstacle =
            obs_2d[px - 1, py] || obs_2d[px + 1, py] ||
            obs_2d[px, py - 1] || obs_2d[px, py + 1] ||
            obs_2d[px - 1, py - 1] || obs_2d[px + 1, py - 1] ||
            obs_2d[px - 1, py + 1] || obs_2d[px + 1, py + 1]
        if touches_obstacle
            sample_mode_2d[px, py] = max(sample_mode_2d[px, py], 5.0f0)
        elseif amr_frontier_dilated[px, py]
            sample_mode_2d[px, py] = 4.0f0
        elseif sample_mode_2d[px, py] > 4.0f0
            sample_mode_2d[px, py] = 4.0f0
        end
    end

    # Per-level pixel statistics for diagnostic (before cosmetic smoothing)
    for lvl_id in 1:n_levels
        n_pix = 0
        u_max_lvl = 0.0f0
        u_sum_lvl = 0.0
        for gv in 1:img_h, gu in 1:img_w
            if owner_level_2d[gu, gv] == Float32(lvl_id) && isfinite(vel_2d[gu, gv, 1])
                u_mag = sqrt(vel_2d[gu, gv, 1]^2 + vel_2d[gu, gv, 2]^2 + vel_2d[gu, gv, 3]^2)
                u_max_lvl = max(u_max_lvl, u_mag)
                u_sum_lvl += u_mag
                n_pix += 1
            end
        end
        if n_pix > 0
            @printf("[Slice] L%d pixels: n=%d  u_max=%.4f  u_mean=%.4f\n",
                    lvl_id, n_pix, u_max_lvl, u_sum_lvl / n_pix)
        end
    end
    flush(stdout)

    # ── AMR interface smoothing pass ──────────────────────────────────
    # Detect pixels near level boundaries and apply distance-weighted
    # blending to soften the seams that appear in derivative fields
    # (schlieren, pi_neq) and temperature.
    if SLICE_COSMETIC_FILTERING
        blend_radius = max(1, round(Int, 3.0 / downsample))
        _amr_interface_smooth!(rho_2d, vel_2d, temp_2d, pinq_mag_2d,
                               obs_2d, owner_level_2d, img_w, img_h, blend_radius)
    end

    # ── Signed-distance-to-wall field (numerical geometry as the solver sees it) ──
    # Sampled per pixel from the OWNER level, INCLUDING obstacle pixels (we
    # want the SDF interior of the body too — its zero contour is the wall).
    # Cell-snap sampling; the sub-cell accuracy lives in the values themselves.
    wall_dist_2d = fill(NaN32, img_w, img_h)
    for py in 1:img_h, px in 1:img_w
        owner_lvl = Int(owner_level_2d[px, py])
        owner_lvl == 0 && continue
        owner_lvl > length(level_data) && continue
        data_lvl = level_data[owner_lvl]
        data_lvl.wall_dist === nothing && continue
        x_phys = (px - 0.5) * px_size
        secondary_phys = (py - 0.5) * px_size
        primary_idx = _coord_to_cell_index_unclamped(x_phys, data_lvl.dx)
        secondary_idx = _coord_to_cell_index_unclamped(secondary_phys, data_lvl.dx)
        slice_idx = _coord_to_cell_index_unclamped(slice_meta.coord, data_lvl.dx)
        val = _sample_wall_dist_cell(data_lvl, slice_meta.plane,
                                     primary_idx, secondary_idx, slice_idx)
        isfinite(val) && (wall_dist_2d[px, py] = val)
    end

    return (img_w, img_h, rho_2d, vel_2d, obs_2d, temp_2d, pinq_mag_2d, owner_level_2d, sample_mode_2d, local_shift_2d, wall_dist_2d, slice_meta)
end

"""
    _amr_interface_smooth!(rho, vel, temp, pinq, obs, level_map, w, h, radius)

Distance-weighted blending around AMR level transitions.
For each pixel within `radius` of a level boundary (where the owning level
changes), blend the value with the mean of its neighbors, weighted by
distance from the interface. This eliminates the single-pixel hard seam.
"""
function _amr_interface_smooth!(rho::Matrix{Float32}, vel::Array{Float32,3},
                                temp::Matrix{Float32}, pinq::Matrix{Float32},
                                obs::BitMatrix, level_map::Matrix{Float32},
                                w::Int, h::Int, radius::Int)
    # Build interface distance map (0 = on interface, radius = far away)
    is_boundary = falses(w, h)
    @inbounds for py in 2:h-1, px in 2:w-1
        obs[px, py] && continue
        lv = level_map[px, py]
        lv == 0.0f0 && continue
        if (level_map[px-1, py] != lv && level_map[px-1, py] != 0.0f0) ||
           (level_map[px+1, py] != lv && level_map[px+1, py] != 0.0f0) ||
           (level_map[px, py-1] != lv && level_map[px, py-1] != 0.0f0) ||
           (level_map[px, py+1] != lv && level_map[px, py+1] != 0.0f0)
            is_boundary[px, py] = true
        end
    end

    # For each field pixel near a boundary, blend with local average
    buf_rho = copy(rho)
    buf_vel = copy(vel)
    buf_temp = copy(temp)
    buf_pinq = copy(pinq)

    r = radius
    @inbounds for py in (1+r):(h-r), px in (1+r):(w-r)
        obs[px, py] && continue
        !isfinite(rho[px, py]) && continue

        # Check if this pixel is within radius of a boundary
        near_boundary = false
        for dy in -r:r, dx in -r:r
            if is_boundary[px+dx, py+dy]
                near_boundary = true
                break
            end
        end
        !near_boundary && continue

        # Distance to nearest boundary pixel
        min_dist_sq = r * r + 1
        for dy in -r:r, dx in -r:r
            if is_boundary[px+dx, py+dy]
                d2 = dx*dx + dy*dy
                d2 < min_dist_sq && (min_dist_sq = d2)
            end
        end
        dist = sqrt(Float32(min_dist_sq))
        # Blending weight: 1.0 at boundary, 0.0 at edge of radius
        alpha = clamp(1.0f0 - dist / Float32(r + 1), 0.0f0, 0.6f0)
        alpha < 0.01f0 && continue

        # Compute local weighted average (3×3 box with distance weighting)
        w_sum = 0.0f0
        r_acc = 0.0f0; vx_a = 0.0f0; vy_a = 0.0f0; vz_a = 0.0f0
        t_acc = 0.0f0; p_acc = 0.0f0
        for dy in -1:1, dx in -1:1
            nx, ny = px + dx, py + dy
            (obs[nx, ny] || !isfinite(rho[nx, ny])) && continue
            wt = (dx == 0 && dy == 0) ? 2.0f0 : 1.0f0
            w_sum += wt
            r_acc += wt * rho[nx, ny]
            vx_a += wt * vel[nx, ny, 1]
            vy_a += wt * vel[nx, ny, 2]
            vz_a += wt * vel[nx, ny, 3]
            isfinite(temp[nx, ny]) && (t_acc += wt * temp[nx, ny])
            isfinite(pinq[nx, ny]) && (p_acc += wt * pinq[nx, ny])
        end
        w_sum < 1.0f-6 && continue
        inv_w = 1.0f0 / w_sum

        buf_rho[px, py] = (1.0f0 - alpha) * rho[px, py] + alpha * r_acc * inv_w
        buf_vel[px, py, 1] = (1.0f0 - alpha) * vel[px, py, 1] + alpha * vx_a * inv_w
        buf_vel[px, py, 2] = (1.0f0 - alpha) * vel[px, py, 2] + alpha * vy_a * inv_w
        buf_vel[px, py, 3] = (1.0f0 - alpha) * vel[px, py, 3] + alpha * vz_a * inv_w
        isfinite(temp[px, py]) && (buf_temp[px, py] = (1.0f0 - alpha) * temp[px, py] + alpha * t_acc * inv_w)
        isfinite(pinq[px, py]) && (buf_pinq[px, py] = (1.0f0 - alpha) * pinq[px, py] + alpha * p_acc * inv_w)
    end

    copyto!(rho, buf_rho)
    copyto!(vel, buf_vel)
    copyto!(temp, buf_temp)
    copyto!(pinq, buf_pinq)
end

"""
    _gaussian_smooth_3x3!(field, obs, w, h)

In-place 3×3 Gaussian smooth (σ≈0.7) on a 2D Float32 field.
Skips obstacle and NaN cells; boundary pixels untouched.
Kernel: [1 2 1; 2 4 2; 1 2 1] / 16.
"""
function _gaussian_smooth_3x3!(field::AbstractMatrix{Float32}, obs::BitMatrix, w::Int, h::Int)
    buf = copy(field)
    @inbounds for py in 2:h-1, px in 2:w-1
        obs[px, py] && continue
        !isfinite(field[px, py]) && continue

        w_sum = 0.0f0
        acc = 0.0f0
        for (dy, dx, wt) in ((-1,-1,1.0f0), (-1,0,2.0f0), (-1,1,1.0f0),
                              ( 0,-1,2.0f0), ( 0,0,4.0f0), ( 0,1,2.0f0),
                              ( 1,-1,1.0f0), ( 1,0,2.0f0), ( 1,1,1.0f0))
            nx, ny = px + dx, py + dy
            if !obs[nx, ny] && isfinite(field[nx, ny])
                w_sum += wt
                acc += wt * field[nx, ny]
            end
        end
        w_sum > 0.0f0 && (buf[px, py] = acc / w_sum)
    end
    copyto!(field, buf)
end

"""
    _make_heatmap(field_2d, obs_2d, title_str, clims; cmap=:jet, x_label="X", y_label="Z")

Create a Plots heatmap with obstacle masking (black).
Returns the plot object.
"""
function _make_heatmap(field_2d::Matrix{Float32}, obs_2d::BitMatrix,
                       title_str::String, clims::Tuple{Float64,Float64};
                       cmap=:jet, x_label::String="X", y_label::String="Z")
    # Keep the rendered rows in the same vertical order as the sampled slice.
    # The earlier row reversal mirrored positive-AoA aircraft vertically in
    # XZ/YZ PNGs; leaving row order intact flips the image around its
    # horizontal axis without changing the sampled physics arrays or CSVs.
    data = permutedims(field_2d, (2, 1))
    obs_t = permutedims(obs_2d, (2, 1))

    masked = copy(data)
    masked[obs_t] .= NaN32

    img_h, img_w = size(masked)

    p = heatmap(1:img_w, 1:img_h, masked;
                clims=clims,
                color=cmap,
                aspect_ratio=:equal,
                title=title_str,
                xlabel=x_label, ylabel=y_label,
                xlims=(1, img_w), ylims=(1, img_h),
                colorbar=true,
                size=(max(800, img_w * 3), max(600, img_h * 3)),
                background_color=:black,
                dpi=150)
    return p
end

function _display_asinh_map(field_2d::Matrix{Float32}, obs_2d::BitMatrix,
                            scale_ref::Float64; strength::Float32=24.0f0)
    mapped = similar(field_2d)
    denom = asinh(strength)
    inv_scale = 1.0f0 / Float32(max(scale_ref, 1.0e-12))

    @inbounds for gv in 1:size(field_2d, 2), gu in 1:size(field_2d, 1)
        if obs_2d[gu, gv] || !isfinite(field_2d[gu, gv])
            mapped[gu, gv] = NaN32
        else
            raw = max(field_2d[gu, gv], 0.0f0)
            mapped[gu, gv] = asinh(strength * raw * inv_scale) / denom
        end
    end

    return mapped
end

function _mark_invalid_nonobstacle!(mapped::Matrix{Float32}, obs_2d::BitMatrix,
                                    sentinel::Float32)
    @inbounds for gv in 1:size(mapped, 2), gu in 1:size(mapped, 1)
        if !obs_2d[gu, gv] && !isfinite(mapped[gu, gv])
            mapped[gu, gv] = sentinel
        end
    end
    return mapped
end

function _display_signed_asinh_map(field_2d::Matrix{Float32}, obs_2d::BitMatrix,
                                   scale_ref::Float64; strength::Float32=16.0f0)
    mapped = similar(field_2d)
    denom = asinh(strength)
    inv_scale = 1.0f0 / Float32(max(scale_ref, 1.0e-12))

    @inbounds for gv in 1:size(field_2d, 2), gu in 1:size(field_2d, 1)
        if obs_2d[gu, gv] || !isfinite(field_2d[gu, gv])
            mapped[gu, gv] = NaN32
        else
            raw = clamp(field_2d[gu, gv] * inv_scale, -1.0f0, 1.0f0)
            mapped[gu, gv] = asinh(strength * raw) / denom
        end
    end

    return mapped
end

"""
    export_slice_image(t_step, grids, output_dir, backend; field=:all)

Export geometry-aware 2D slice PNGs using Plots.jl.

Fields:
- :mach — Mach number |u|/cs
- :density — density ρ
- :velocity — velocity magnitude |u|
- :vorticity — slice-normal vorticity magnitude
- :temperature — temperature T (when DDF enabled)
- :pressure — pressure p = ρ·cs²·T
- :schlieren — numerical schlieren |∇ρ|
- :pi_neq — |Π_neq| non-equilibrium stress magnitude
- :sample_mode — 2=in-level interp, 3=nearest fallback, 4=frontier-assisted bilinear, 5=edge reduced stencil
- :amr_debug — AMR edge mask plus rho/velocity jump and owner-local high-pass diagnostics
- :all — exports all fields (default)
"""
# One-shot CSV dump of the slice density + AMR ownership for diagnosis
# of the visible 1-pixel seam at refinement-box edges. Forward differences
# in ρ per axis are pre-computed so post-processing can sort by |Δρ| to
# surface the largest seams. Called twice per slice export (once with
# SLICE_COSMETIC_FILTERING on, once off) to separate the cosmetic-blend
# contribution from the physics/prolongation residual.
function _dump_amr_seam_csv(output_dir::String, t_step::Int,
                            w::Int, h::Int,
                            rho::AbstractMatrix{<:AbstractFloat},
                            owner::AbstractMatrix{<:AbstractFloat},
                            mode::AbstractMatrix{<:AbstractFloat},
                            obs::BitMatrix,
                            px_size::Float64; tag::String="smoothed")
    fname = joinpath(output_dir,
                     @sprintf("amr_seam_%06d_%s.csv", t_step, tag))
    open(fname, "w") do f
        println(f, "px,py,x_phys,sec_phys,rho,owner_level,sample_mode,is_obstacle,drho_x_left,drho_x_right,drho_y_down,drho_y_up,owner_step_left,owner_step_right,owner_step_down,owner_step_up,mode_step_left,mode_step_right,mode_step_down,mode_step_up")
        for py in 1:h, px in 1:w
            # Pixel-center physical coordinates (primary = x, secondary = y or z).
            x_phys = (Float64(px) - 0.5) * px_size
            sec_phys = (Float64(py) - 0.5) * px_size
            rho_v = Float64(rho[px, py])
            own_v = Float64(owner[px, py])
            mod_v = Float64(mode[px, py])
            isobs = obs[px, py] ? 1 : 0
            drho_xl = (px > 1) ? rho_v - Float64(rho[px-1, py]) : NaN
            drho_xr = (px < w) ? Float64(rho[px+1, py]) - rho_v : NaN
            drho_yd = (py > 1) ? rho_v - Float64(rho[px, py-1]) : NaN
            drho_yu = (py < h) ? Float64(rho[px, py+1]) - rho_v : NaN
            own_xl = (px > 1) ? own_v - Float64(owner[px-1, py]) : NaN
            own_xr = (px < w) ? Float64(owner[px+1, py]) - own_v : NaN
            own_yd = (py > 1) ? own_v - Float64(owner[px, py-1]) : NaN
            own_yu = (py < h) ? Float64(owner[px, py+1]) - own_v : NaN
            mod_xl = (px > 1) ? mod_v - Float64(mode[px-1, py]) : NaN
            mod_xr = (px < w) ? Float64(mode[px+1, py]) - mod_v : NaN
            mod_yd = (py > 1) ? mod_v - Float64(mode[px, py-1]) : NaN
            mod_yu = (py < h) ? Float64(mode[px, py+1]) - mod_v : NaN
            @printf(f, "%d,%d,%.6f,%.6f,%.9e,%.2f,%.2f,%d,%.9e,%.9e,%.9e,%.9e,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                    px, py, x_phys, sec_phys, rho_v, own_v, mod_v, isobs,
                    drho_xl, drho_xr, drho_yd, drho_yu,
                    own_xl, own_xr, own_yd, own_yu,
                    mod_xl, mod_xr, mod_yd, mod_yu)
        end
    end
    @printf("[Slice-diag] Wrote %s\n", fname)
end

function _dump_amr_seam_summary(output_dir::String, t_step::Int,
                                w::Int, h::Int,
                                rho::AbstractMatrix{<:AbstractFloat},
                                owner::AbstractMatrix{<:AbstractFloat},
                                mode::AbstractMatrix{<:AbstractFloat},
                                obs::BitMatrix; tag::String="smoothed")
    x_owner_sum = 0.0
    x_mode_sum = 0.0
    x_plain_sum = 0.0
    y_owner_sum = 0.0
    y_mode_sum = 0.0
    y_plain_sum = 0.0
    x_owner_count = 0
    x_mode_count = 0
    x_plain_count = 0
    y_owner_count = 0
    y_mode_count = 0
    y_plain_count = 0

    x_count = zeros(Int, w)
    x_jump_sum = zeros(Float64, w)
    x_owner_hits = zeros(Int, w)
    x_mode_hits = zeros(Int, w)
    y_count = zeros(Int, h)
    y_jump_sum = zeros(Float64, h)
    y_owner_hits = zeros(Int, h)
    y_mode_hits = zeros(Int, h)

    @inbounds for py in 1:h, px in 1:w
        obs[px, py] && continue
        rho_v = Float64(rho[px, py])
        own_v = Float64(owner[px, py])
        mod_v = Float64(mode[px, py])

        if px > 1 && px < w
            drho_xl = rho_v - Float64(rho[px-1, py])
            drho_xr = Float64(rho[px+1, py]) - rho_v
            if isfinite(drho_xl) && isfinite(drho_xr)
                jump_x = 0.5 * (abs(drho_xl) + abs(drho_xr))
                owner_x = abs(own_v - Float64(owner[px-1, py])) > 1.0e-6 ||
                          abs(Float64(owner[px+1, py]) - own_v) > 1.0e-6
                mode_x = abs(mod_v - Float64(mode[px-1, py])) > 1.0e-6 ||
                         abs(Float64(mode[px+1, py]) - mod_v) > 1.0e-6

                x_count[px] += 1
                x_jump_sum[px] += jump_x
                owner_x && (x_owner_hits[px] += 1)
                mode_x && (x_mode_hits[px] += 1)

                if owner_x
                    x_owner_sum += jump_x
                    x_owner_count += 1
                end
                if mode_x
                    x_mode_sum += jump_x
                    x_mode_count += 1
                end
                if !owner_x && !mode_x
                    x_plain_sum += jump_x
                    x_plain_count += 1
                end
            end
        end

        if py > 1 && py < h
            drho_yd = rho_v - Float64(rho[px, py-1])
            drho_yu = Float64(rho[px, py+1]) - rho_v
            if isfinite(drho_yd) && isfinite(drho_yu)
                jump_y = 0.5 * (abs(drho_yd) + abs(drho_yu))
                owner_y = abs(own_v - Float64(owner[px, py-1])) > 1.0e-6 ||
                          abs(Float64(owner[px, py+1]) - own_v) > 1.0e-6
                mode_y = abs(mod_v - Float64(mode[px, py-1])) > 1.0e-6 ||
                         abs(Float64(mode[px, py+1]) - mod_v) > 1.0e-6

                y_count[py] += 1
                y_jump_sum[py] += jump_y
                owner_y && (y_owner_hits[py] += 1)
                mode_y && (y_mode_hits[py] += 1)

                if owner_y
                    y_owner_sum += jump_y
                    y_owner_count += 1
                end
                if mode_y
                    y_mode_sum += jump_y
                    y_mode_count += 1
                end
                if !owner_y && !mode_y
                    y_plain_sum += jump_y
                    y_plain_count += 1
                end
            end
        end
    end

    mean_or_zero(sum_v, count_v) = count_v > 0 ? sum_v / count_v : 0.0

    summary_name = joinpath(output_dir,
                            @sprintf("amr_seam_%06d_summary_%s.csv", t_step, tag))
    open(summary_name, "w") do f
        println(f, "axis,bucket,count,mean_abs_jump")
        @printf(f, "x,owner,%d,%.9e\n", x_owner_count, mean_or_zero(x_owner_sum, x_owner_count))
        @printf(f, "x,mode,%d,%.9e\n", x_mode_count, mean_or_zero(x_mode_sum, x_mode_count))
        @printf(f, "x,plain,%d,%.9e\n", x_plain_count, mean_or_zero(x_plain_sum, x_plain_count))
        @printf(f, "y,owner,%d,%.9e\n", y_owner_count, mean_or_zero(y_owner_sum, y_owner_count))
        @printf(f, "y,mode,%d,%.9e\n", y_mode_count, mean_or_zero(y_mode_sum, y_mode_count))
        @printf(f, "y,plain,%d,%.9e\n", y_plain_count, mean_or_zero(y_plain_sum, y_plain_count))
    end

    profile_name = joinpath(output_dir,
                            @sprintf("amr_seam_%06d_profile_%s.csv", t_step, tag))
    open(profile_name, "w") do f
        println(f, "axis,index,count,mean_abs_jump,owner_frac,mode_frac")
        for px in 1:w
            if x_count[px] == 0
                continue
            end
            @printf(f, "x,%d,%d,%.9e,%.9e,%.9e\n",
                    px, x_count[px], x_jump_sum[px] / x_count[px],
                    x_owner_hits[px] / x_count[px],
                    x_mode_hits[px] / x_count[px])
        end
        for py in 1:h
            if y_count[py] == 0
                continue
            end
            @printf(f, "y,%d,%d,%.9e,%.9e,%.9e\n",
                    py, y_count[py], y_jump_sum[py] / y_count[py],
                    y_owner_hits[py] / y_count[py],
                    y_mode_hits[py] / y_count[py])
        end
    end

    @printf("[Slice-diag] Wrote %s\n", summary_name)
    @printf("[Slice-diag] Wrote %s\n", profile_name)
    @printf("[Slice-diag] step %d %s: x(owner/plain)=%.3e / %.3e, y(owner/plain)=%.3e / %.3e\n",
            t_step, tag,
            mean_or_zero(x_owner_sum, x_owner_count), mean_or_zero(x_plain_sum, x_plain_count),
            mean_or_zero(y_owner_sum, y_owner_count), mean_or_zero(y_plain_sum, y_plain_count))
end

function export_slice_image(t_step::Int, grids, output_dir::String, backend;
                            field::Symbol=:all,
                            cx_gpu=nothing, cy_gpu=nothing, cz_gpu=nothing, w_gpu=nothing,
                            plane_override::Union{Nothing,Symbol}=nothing,
                            y_override::Union{Nothing,Float64}=nothing,
                            x_override::Union{Nothing,Float64}=nothing,
                            z_override::Union{Nothing,Float64}=nothing,              # lbm-io-opts (27 Oct 2026)
                            normal_clamp::Symbol=:body,                              # lbm-io-opts: :domain for explicit planes
                            fields::Union{Nothing,AbstractVector{Symbol}}=nothing,   # lbm-io-opts: advanced.slice.fields
                            fname_suffix::String="")
    # Per-step subdirectory: every slice (PNG + AMR-seam CSV) for this
    # iteration goes into <output_dir>/step_NNNNNN/. Created lazily on first
    # write of each step. Makes browsing a long run much easier — one folder
    # per timestep snapshot — and keeps the SLICES/ root from accumulating
    # thousands of files at high slice_output_freq.
    output_dir = joinpath(output_dir, @sprintf("step_%06d", t_step))
    isdir(output_dir) || mkpath(output_dir)

    img_w, img_h, rho_2d, vel_2d, obs_2d, temp_2d, pinq_mag_2d, owner_level_2d, sample_mode_2d, local_shift_2d, wall_dist_2d, slice_meta =
        extract_slice_data(t_step, grids, backend;
                           cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu,
                           plane_override=plane_override,
                           y_override=y_override,
                           x_override=x_override,
                           z_override=z_override,
                           normal_clamp=normal_clamp)

    # ── AMR-seam diagnostic dump (Phase 3 / Step 1) ──────────────────
    # Once early in the run (when density has enough contrast to expose
    # interface steps but flow is still mid-ramp and smooth), dump two
    # paired CSVs — one with the cosmetic smoother applied (what the
    # PNG renders), one re-extracted with SLICE_COSMETIC_FILTERING
    # forced off (what the raw bilinear-per-level stitch looks like).
    # The difference between the two isolates the cosmetic-blend
    # contribution (cause 1) from the residual physics+prolongation
    # contribution (causes 2/3).
    # Default: emit seam diagnostics at three early checkpoint steps. The
    # free-stream preservation CI gate (see PREPOST/freestream_gate_check.jl
    # and case 013) sets AMR_SEAM_DIAGNOSTIC_ALL_STEPS=true to emit every
    # slice step so the gate has a dense time series to assert against.
    # 2026-06-22 (user request): seam *.csv diagnostics are a DEBUG-only artifact.
    # In production slice mode (SLICE_FIELD_SET != :all) they are never written.
    if SLICE_FIELD_SET == :all &&
       (t_step == 100 || t_step == 200 || t_step == 400 ||
        (AMR_SEAM_DIAGNOSTIC_ALL_STEPS && t_step > 0))
        p = get_domain_params()
        domain_primary = slice_meta.plane == :yz ? p.domain_size[2] : p.domain_size[1]
        px_size = domain_primary / Float64(img_w)
        suffix_tag = isempty(fname_suffix) ? "" : replace(fname_suffix, "_" => "")
        smoothed_tag = isempty(suffix_tag) ? "smoothed" : "smoothed_" * suffix_tag
        raw_tag = isempty(suffix_tag) ? "raw" : "raw_" * suffix_tag
        _dump_amr_seam_csv(output_dir, t_step, img_w, img_h,
                           rho_2d, owner_level_2d, sample_mode_2d, obs_2d,
                           px_size; tag=smoothed_tag)
        _dump_amr_seam_summary(output_dir, t_step, img_w, img_h,
                               rho_2d, owner_level_2d, sample_mode_2d, obs_2d;
                               tag=smoothed_tag)
        # Re-extract with cosmetic filtering off — temporarily override
        # the global so we see the raw per-level stitched density.
        _saved = SLICE_COSMETIC_FILTERING
        global SLICE_COSMETIC_FILTERING = false
        try
            raw_w, raw_h, rho_raw, _, obs_raw, _, _, owner_raw, mode_raw, _, _, _ =
                extract_slice_data(t_step, grids, backend;
                                   cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu,
                                   plane_override=plane_override,
                                   y_override=y_override,
                                   x_override=x_override,
                                   z_override=z_override,
                                   normal_clamp=normal_clamp)
            _dump_amr_seam_csv(output_dir, t_step, raw_w, raw_h,
                               rho_raw, owner_raw, mode_raw, obs_raw,
                               px_size; tag=raw_tag)
            _dump_amr_seam_summary(output_dir, t_step, raw_w, raw_h,
                                   rho_raw, owner_raw, mode_raw, obs_raw;
                                   tag=raw_tag)
        finally
            global SLICE_COSMETIC_FILTERING = _saved
        end
    end

    if slice_meta.used_fallback
        @printf("[Slice] %s center plane at %s=%.6f m missed the geometry; shifted to %.6f m to intersect obstacle cells\n",
                slice_meta.plane_label, slice_meta.normal_label, slice_meta.target_coord, slice_meta.coord)
    else
        @printf("[Slice] Using %s slice through STL center: %s=%.6f m\n",
                slice_meta.plane_label, slice_meta.normal_label, slice_meta.coord)
    end
    if !slice_meta.intersects_geometry
        @printf("[Slice] Warning: no obstacle cells detected on the selected %s slice\n", slice_meta.plane_label)
    end

    vel_mag = fill(NaN32, img_w, img_h)
    mach_2d = fill(NaN32, img_w, img_h)
    schlieren_2d = fill(NaN32, img_w, img_h)
    schlieren_support_2d = fill(0.0f0, img_w, img_h)
    vorticity_2d = fill(NaN32, img_w, img_h)
    pressure_2d = fill(NaN32, img_w, img_h)
    pressure_rel_2d = fill(NaN32, img_w, img_h)
    cp_2d = fill(NaN32, img_w, img_h)
    pressure_coeff = EOS_THERMODYNAMIC ?
        Float32((DDF_GAMMA - 1.0) * DDF_CV) :
        Float32(CS2_PHYSICS)
    t_inf = (DDF_ENABLED || EOS_THERMODYNAMIC) ? Float32(DDF_T_INLET) : 1.0f0
    p_inf_ref = max(Float32(IC_UNIFORM_RHO) * pressure_coeff * t_inf, 1.0f-12)
    q_inf_ref = max(0.5f0 * Float32(IC_UNIFORM_RHO) * Float32(U_TARGET) * Float32(U_TARGET), 1.0f-12)
    u_ref_slice = abs(Float64(U_INLET_CURRENT[])) > 0.0 ?
                  abs(Float64(U_INLET_CURRENT[])) :
                  abs(Float64(U_TARGET))
    for gv in 1:img_h, gu in 1:img_w
        if !obs_2d[gu, gv] && isfinite(rho_2d[gu, gv]) && isfinite(vel_2d[gu, gv, 1])
            vx = vel_2d[gu, gv, 1]
            vy = vel_2d[gu, gv, 2]
            vz = vel_2d[gu, gv, 3]
            vm = sqrt(vx * vx + vy * vy + vz * vz)
            vel_mag[gu, gv] = vm
            t_val = isfinite(temp_2d[gu, gv]) ? temp_2d[gu, gv] : 1.0f0
            if !isfinite(temp_2d[gu, gv]) && EOS_THERMODYNAMIC && LATTICE_TYPE == :d3q39
                t_val = Float32(diagnostic_effective_temperature(
                    rho_2d[gu, gv], vx, vy, vz, NaN, 0.0, u_ref_slice))
                temp_2d[gu, gv] = t_val
            end
            mach_2d[gu, gv] = vm / sqrt(max(Float32(CS2_PHYSICS) * t_val, 1.0f-12))
            # SOTA A3: when EOS_THERMODYNAMIC is on, use the true ideal-gas
            # form p = (γ-1)·ρ·cv·T. Otherwise use legacy linearized
            # p = ρ·cs²·T. The lattice's cs² is a discrete-acoustic constant;
            # the thermodynamic p depends on γ and cv. For air (γ=1.4, cv=2.5)
            # the two differ by ~50 % even at T=1.
            p_val = rho_2d[gu, gv] * pressure_coeff * t_val
            pressure_2d[gu, gv] = p_val
            pressure_rel_2d[gu, gv] = p_val / p_inf_ref
            cp_2d[gu, gv] = (p_val - p_inf_ref) / q_inf_ref
        end
    end

    params_for_vorticity = get_domain_params()
    slice_primary_size =
        slice_meta.plane == :yz ? params_for_vorticity.domain_size[2] :
        params_for_vorticity.domain_size[1]
    slice_secondary_size =
        slice_meta.plane == :xy ? params_for_vorticity.domain_size[2] :
        params_for_vorticity.domain_size[3]
    slice_du = Float32(slice_primary_size / max(Float64(img_w), 1.0))
    slice_dv = Float32(slice_secondary_size / max(Float64(img_h), 1.0))

    derivative_margin = 2

    function _inside_derivative_frame(gu0::Int, gv0::Int)
        return gu0 > derivative_margin && gu0 <= img_w - derivative_margin &&
               gv0 > derivative_margin && gv0 <= img_h - derivative_margin
    end

    function _same_mode_for_derivative(gu0::Int, gv0::Int, gu1::Int, gv1::Int)
        mode0 = sample_mode_2d[gu0, gv0]
        mode1 = sample_mode_2d[gu1, gv1]
        mode0 > 0.0f0 && mode1 > 0.0f0 || return false

        # Mode 4 is a diagnostic AMR-frontier tag applied after sampling; the
        # underlying value is still continuous bilinear data. Mode 5 is reduced
        # support at physical/domain edges, but is also a valid fluid sample.
        # Do not let those visualization tags create black holes in derivative
        # fields. Only nearest fallback (mode 3) remains a distinct class.
        if mode0 == 3.0f0 || mode1 == 3.0f0
            return mode0 == mode1
        end
        return true
    end

    function _same_owner_rho_ok(gu0::Int, gv0::Int, gu1::Int, gv1::Int)
        owner = owner_level_2d[gu0, gv0]
        return owner > 0.0f0 &&
               owner_level_2d[gu1, gv1] == owner &&
               _same_mode_for_derivative(gu0, gv0, gu1, gv1) &&
               !obs_2d[gu1, gv1] &&
               isfinite(rho_2d[gu1, gv1])
    end

    function _same_owner_vel_ok(gu0::Int, gv0::Int, gu1::Int, gv1::Int, comp::Int)
        owner = owner_level_2d[gu0, gv0]
        return owner > 0.0f0 &&
               owner_level_2d[gu1, gv1] == owner &&
               _same_mode_for_derivative(gu0, gv0, gu1, gv1) &&
               !obs_2d[gu1, gv1] &&
               isfinite(vel_2d[gu1, gv1, comp])
    end

    for gv in 2:img_h-1, gu in 2:img_w-1
        derivative_center_ok =
            _inside_derivative_frame(gu, gv) &&
            sample_mode_2d[gu, gv] > 0.0f0

        if derivative_center_ok && !obs_2d[gu, gv] && isfinite(rho_2d[gu, gv])
            # Schlieren is a derivative diagnostic. Differentiating across a
            # stitched AMR owner change samples values whose two sides come
            # from levels coupled by an interpolator that has a small but
            # non-zero residual — central diff there amplifies the residual
            # into fake gradient lobes. Refuse the stencil instead and leave
            # NaN; the dark ring around each AMR rectangle is the slice
            # honestly reporting "no derivative supported here".
            left_ok = _same_owner_rho_ok(gu, gv, gu - 1, gv)
            right_ok = _same_owner_rho_ok(gu, gv, gu + 1, gv)
            down_ok = _same_owner_rho_ok(gu, gv, gu, gv - 1)
            up_ok = _same_owner_rho_ok(gu, gv, gu, gv + 1)

            if left_ok && right_ok && down_ok && up_ok
                drho_du = 0.5f0 * (rho_2d[gu + 1, gv] - rho_2d[gu - 1, gv])
                drho_dv = 0.5f0 * (rho_2d[gu, gv + 1] - rho_2d[gu, gv - 1])
                schlieren_2d[gu, gv] = sqrt(drho_du * drho_du + drho_dv * drho_dv)
                schlieren_support_2d[gu, gv] = 1.0f0
            end
        end

        if derivative_center_ok && !obs_2d[gu, gv] && owner_level_2d[gu, gv] > 0.0f0 && isfinite(vel_2d[gu, gv, 1])
            # Vorticity is a derivative diagnostic; same reasoning as schlieren
            # — refusing the stencil across an AMR owner boundary keeps the
            # tiny inter-level velocity residual from being amplified into
            # fake vortex lobes parked at the AMR rectangles.
            left_x_ok = _same_owner_vel_ok(gu, gv, gu - 1, gv, 1)
            right_x_ok = _same_owner_vel_ok(gu, gv, gu + 1, gv, 1)
            down_x_ok = _same_owner_vel_ok(gu, gv, gu, gv - 1, 1)
            up_x_ok = _same_owner_vel_ok(gu, gv, gu, gv + 1, 1)

            if slice_meta.plane == :xy && isfinite(vel_2d[gu, gv, 2])
                left_y_ok = _same_owner_vel_ok(gu, gv, gu - 1, gv, 2)
                right_y_ok = _same_owner_vel_ok(gu, gv, gu + 1, gv, 2)

                if left_y_ok && right_y_ok && down_x_ok && up_x_ok
                    duy_dx = 0.5f0 * (vel_2d[gu + 1, gv, 2] - vel_2d[gu - 1, gv, 2]) / slice_du
                    dux_dy = 0.5f0 * (vel_2d[gu, gv + 1, 1] - vel_2d[gu, gv - 1, 1]) / slice_dv
                    vorticity_2d[gu, gv] = duy_dx - dux_dy
                end
            elseif slice_meta.plane == :xz && isfinite(vel_2d[gu, gv, 3])
                left_z_ok = _same_owner_vel_ok(gu, gv, gu - 1, gv, 3)
                right_z_ok = _same_owner_vel_ok(gu, gv, gu + 1, gv, 3)

                if down_x_ok && up_x_ok && left_z_ok && right_z_ok
                    dux_dz = 0.5f0 * (vel_2d[gu, gv + 1, 1] - vel_2d[gu, gv - 1, 1]) / slice_dv
                    duz_dx = 0.5f0 * (vel_2d[gu + 1, gv, 3] - vel_2d[gu - 1, gv, 3]) / slice_du
                    vorticity_2d[gu, gv] = dux_dz - duz_dx
                end
            elseif slice_meta.plane == :yz &&
                   isfinite(vel_2d[gu, gv, 2]) && isfinite(vel_2d[gu, gv, 3])
                left_z_ok = _same_owner_vel_ok(gu, gv, gu - 1, gv, 3)
                right_z_ok = _same_owner_vel_ok(gu, gv, gu + 1, gv, 3)
                down_y_ok = _same_owner_vel_ok(gu, gv, gu, gv - 1, 2)
                up_y_ok = _same_owner_vel_ok(gu, gv, gu, gv + 1, 2)

                if left_z_ok && right_z_ok && down_y_ok && up_y_ok
                    duz_dy = 0.5f0 * (vel_2d[gu + 1, gv, 3] - vel_2d[gu - 1, gv, 3]) / slice_du
                    duy_dz = 0.5f0 * (vel_2d[gu, gv + 1, 2] - vel_2d[gu, gv - 1, 2]) / slice_dv
                    vorticity_2d[gu, gv] = duz_dy - duy_dz
                end
            end
        end
    end

    amr_edge_2d = fill(0.0f0, img_w, img_h)
    amr_rho_jump_2d = fill(NaN32, img_w, img_h)
    amr_u_jump_2d = fill(NaN32, img_w, img_h)
    amr_x_u_jump_2d = fill(NaN32, img_w, img_h)
    amr_y_ux_jump_2d = fill(NaN32, img_w, img_h)
    velocity_highpass_2d = fill(NaN32, img_w, img_h)

    for gv in 1:img_h, gu in 1:img_w
        if obs_2d[gu, gv] || owner_level_2d[gu, gv] <= 0.0f0
            continue
        end

        owner = owner_level_2d[gu, gv]
        rho_jump = 0.0f0
        u_jump = 0.0f0
        x_u_jump = 0.0f0
        y_ux_jump = 0.0f0
        edge_hit = false

        for dir_id in 1:4
            ou = dir_id == 1 ? -1 : (dir_id == 2 ? 1 : 0)
            ov = dir_id == 3 ? -1 : (dir_id == 4 ? 1 : 0)
            nu = gu + ou
            nv = gv + ov
            if nu < 1 || nu > img_w || nv < 1 || nv > img_h ||
               obs_2d[nu, nv] || owner_level_2d[nu, nv] <= 0.0f0 ||
               owner_level_2d[nu, nv] == owner
                continue
            end

            edge_hit = true
            if isfinite(rho_2d[gu, gv]) && isfinite(rho_2d[nu, nv])
                rho_jump = max(rho_jump, abs(rho_2d[nu, nv] - rho_2d[gu, gv]))
            end
            if isfinite(vel_2d[gu, gv, 1]) && isfinite(vel_2d[nu, nv, 1]) &&
               isfinite(vel_2d[gu, gv, 2]) && isfinite(vel_2d[nu, nv, 2]) &&
               isfinite(vel_2d[gu, gv, 3]) && isfinite(vel_2d[nu, nv, 3])
                dvx = vel_2d[nu, nv, 1] - vel_2d[gu, gv, 1]
                dvy = vel_2d[nu, nv, 2] - vel_2d[gu, gv, 2]
                dvz = vel_2d[nu, nv, 3] - vel_2d[gu, gv, 3]
                du_norm = sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
                u_jump = max(u_jump, du_norm)
                if ou != 0
                    x_u_jump = max(x_u_jump, du_norm)
                end
                if ov != 0
                    y_ux_jump = max(y_ux_jump, abs(dvx))
                end
            end
        end

        if edge_hit
            amr_edge_2d[gu, gv] = 1.0f0
            amr_rho_jump_2d[gu, gv] = rho_jump
            amr_u_jump_2d[gu, gv] = u_jump
            amr_x_u_jump_2d[gu, gv] = x_u_jump
            amr_y_ux_jump_2d[gu, gv] = y_ux_jump
        end
    end

    for gv in 2:img_h-1, gu in 2:img_w-1
        if obs_2d[gu, gv] || owner_level_2d[gu, gv] <= 0.0f0 || !isfinite(vel_mag[gu, gv])
            continue
        end
        if !_inside_derivative_frame(gu, gv)
            continue
        end
        owner = owner_level_2d[gu, gv]
        sum_vm = 0.0f0
        count_vm = 0
        for dir_id in 1:4
            ou = dir_id == 1 ? -1 : (dir_id == 2 ? 1 : 0)
            ov = dir_id == 3 ? -1 : (dir_id == 4 ? 1 : 0)
            nu = gu + ou
            nv = gv + ov
            if owner_level_2d[nu, nv] == owner &&
               _same_mode_for_derivative(gu, gv, nu, nv) &&
               !obs_2d[nu, nv] && isfinite(vel_mag[nu, nv])
                sum_vm += vel_mag[nu, nv]
                count_vm += 1
            end
        end
        # A one-sided stencil at an AMR owner boundary draws the refinement
        # rectangle even for a smooth field. Keep this diagnostic strictly
        # interior to same-owner/same-sample patches; use explicit AMR jump
        # images for boundary continuity.
        if count_vm == 4
            velocity_highpass_2d[gu, gv] = abs(vel_mag[gu, gv] - sum_vm / Float32(count_vm))
        end
    end

    function _percentile_clims(field_2d, obs_2d, lo_pct::Float64, hi_pct::Float64;
                               margin::Float64=0.05)
        vals = Float64[]
        for gv in 1:size(field_2d, 2), gu in 1:size(field_2d, 1)
            if !obs_2d[gu, gv] && isfinite(field_2d[gu, gv])
                push!(vals, Float64(field_2d[gu, gv]))
            end
        end
        if isempty(vals)
            return (0.0, 1.0)
        end
        sort!(vals)
        n = length(vals)
        lo_idx = max(1, round(Int, lo_pct * n))
        hi_idx = min(n, round(Int, hi_pct * n))
        lo_val = vals[lo_idx]
        hi_val = vals[hi_idx]
        span = hi_val - lo_val
        if span < 1e-10
            span = max(abs(lo_val) * 0.1, 1e-6)
        end
        return (lo_val - margin * span, hi_val + margin * span)
    end

    function _finite_max(field_2d, obs_2d)
        vmax = 0.0
        for gv in 1:size(field_2d, 2), gu in 1:size(field_2d, 1)
            if !obs_2d[gu, gv] && isfinite(field_2d[gu, gv])
                vmax = max(vmax, Float64(field_2d[gu, gv]))
            end
        end
        return vmax
    end

    function _finite_abs_max(field_2d, obs_2d)
        vmax = 0.0
        for gv in 1:size(field_2d, 2), gu in 1:size(field_2d, 1)
            if !obs_2d[gu, gv] && isfinite(field_2d[gu, gv])
                vmax = max(vmax, abs(Float64(field_2d[gu, gv])))
            end
        end
        return vmax
    end

    slice_desc = @sprintf("%s slice at %s = %.4f m", slice_meta.plane_label, slice_meta.normal_label, slice_meta.coord)
    # AMR jump diagnostics are computed along image-plane axes. Name those axes
    # physically in titles so XZ/YZ slices do not masquerade as x/y neighbors.
    plane_u_axis = slice_meta.plane == :yz ? "y" : "x"
    plane_v_axis = slice_meta.plane == :xy ? "y" : "z"
    # :level and :sample_mode are geometric and do not change across
    # timesteps. They are emitted only when field=:geometric (a one-shot
    # startup call); :all skips them so per-iteration slice dumps stay lean.
    if field == :all
        fields_to_export = Symbol[:mach, :density, :velocity, :vorticity, :temperature, :schlieren,
                                  :schlieren_calibrated,
                                  :pressure, :pressure_rel, :cp, :pi_neq,
                                  :velocity_highpass, :velocity_highpass_calibrated, :local_shift]
    elseif field == :production
        # 2026-06-22 (user spec): the exact production inspection set — velocity,
        # velocity high-pass, vorticity, pressure, density, Mach, schlieren. (The
        # seam *.csv diagnostics are gated OFF in production; see _dump_amr_seam gate.)
        fields_to_export = Symbol[:velocity, :velocity_highpass, :vorticity,
                                  :pressure, :density, :mach, :schlieren, :temperature]
        # :temperature added to the production set 2026-06-29 (user request). Self-guards
        # via `!has_any_temp && continue` below, so it is a no-op for non-DDF cases.
    elseif field == :geometric
        fields_to_export = Symbol[:level, :sample_mode, :local_shift, :amr_edge, :wall_dist, :schlieren_support]
    elseif field == :amr_debug
        fields_to_export = Symbol[:amr_edge, :amr_rho_jump, :amr_u_jump, :amr_x_u_jump, :amr_y_ux_jump, :velocity_highpass]
    else
        fields_to_export = [field]
    end
    fields === nothing || (fields_to_export = collect(Symbol, fields))   # lbm-io-opts (27 Oct 2026): advanced.slice.fields

    for f in fields_to_export
        if f == :mach
            data_2d = mach_2d
            lo, hi = _percentile_clims(mach_2d, obs_2d, 0.01, 0.99; margin=0.05)
            clims = (max(0.0, lo), hi)
            # ANTI-ILLUSION (2026-06-27): annotate the TRUE Mach range (auto-scaled colorbar hides it).
            _mlo = Inf; _mhi = -Inf
            @inbounds for gv in 1:size(mach_2d, 2), gu in 1:size(mach_2d, 1)
                if !obs_2d[gu, gv] && isfinite(mach_2d[gu, gv])
                    _v = Float64(mach_2d[gu, gv])
                    _v < _mlo && (_mlo = _v); _v > _mhi && (_mhi = _v)
                end
            end
            title_str = isfinite(_mlo) ?
                @sprintf("Mach number ∈ [%.3f, %.3f] — %s — step %d", _mlo, _mhi, slice_desc, t_step) :
                @sprintf("Mach number — %s — step %d", slice_desc, t_step)
            fname = @sprintf("%s/slice_mach_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :density
            data_2d = rho_2d
            clims = _percentile_clims(rho_2d, obs_2d, 0.01, 0.99; margin=0.05)
            # ANTI-ILLUSION (2026-06-27): the colorbar AUTO-SCALES to the data percentile range,
            # so a weak (few-percent) density field renders identically to a textbook strong shock.
            # This fooled a multi-day shock-strength investigation. Annotate the TRUE absolute ρ
            # range in the title so the amplitude can never be hidden by auto-scaling.
            _rlo = Inf; _rhi = -Inf
            @inbounds for gv in 1:size(rho_2d, 2), gu in 1:size(rho_2d, 1)
                if !obs_2d[gu, gv] && isfinite(rho_2d[gu, gv])
                    _v = Float64(rho_2d[gu, gv])
                    _v < _rlo && (_rlo = _v); _v > _rhi && (_rhi = _v)
                end
            end
            title_str = isfinite(_rlo) ?
                @sprintf("Density ρ ∈ [%.3f, %.3f]  Δρ=%.3f — %s — step %d",
                         _rlo, _rhi, _rhi - _rlo, slice_desc, t_step) :
                @sprintf("Density ρ — %s — step %d", slice_desc, t_step)
            fname = @sprintf("%s/slice_density_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :temperature
            has_any_temp = any(isfinite, temp_2d)
            !has_any_temp && continue
            data_2d = temp_2d
            clims = _percentile_clims(temp_2d, obs_2d, 0.01, 0.99; margin=0.05)
            title_str = @sprintf("Temperature T — %s — step %d", slice_desc, t_step)
            fname = @sprintf("%s/slice_temperature_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :schlieren
            hi_s = max(_finite_max(schlieren_2d, obs_2d), 1e-8)
            data_2d = _display_asinh_map(schlieren_2d, obs_2d, hi_s; strength=24.0f0)
            _mark_invalid_nonobstacle!(data_2d, obs_2d, 0.0f0)
            clims = (0.0, 1.0)
            title_str = @sprintf("Numerical Schlieren |∇ρ| (asinh contrast, no spatial smoothing) — %s — step %d", slice_desc, t_step)
            title_str = @sprintf("Numerical Schlieren |grad rho| (owner-local asinh contrast) -- %s -- step %d", slice_desc, t_step)
            fname = @sprintf("%s/slice_schlieren_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :schlieren_calibrated
            # ── Calibrated numerical schlieren ───────────────────────────────────
            # The asinh(strength=24) schlieren above amplifies any |grad rho|
            # variation 5-10x and is normalized to the SLICE-GLOBAL max, so a
            # single outlier pixel (e.g. a 5e-3 box-edge spike) redefines the whole
            # colorbar and lights up every faint feature near peak — it cannot tell
            # a fixed AMR seam from a still-broken one. (The "owner-local" in the
            # asinh title is a misnomer; that render is slice-global.)
            #
            # This companion renders |grad rho| on a LINEAR ABSOLUTE scale with a
            # physics-tied ceiling:
            #   ceiling = max(0.5 % of rho_ref ≈ 5e-3, p99 of |grad rho|)
            # so the colorbar reads the TRUE per-cell density-gradient amplitude.
            # The AMR box-edge seam (measured ~2.7e-4 mean, ~5e-3 max on rho~1) is
            # then judged on absolute amplitude, not on an outlier-driven transfer
            # function. Use THIS for "is the seam real / fixed?"; use the asinh
            # variant for "where is ANY density-gradient structure?".
            physical_threshold = 0.005f0   # 0.5 % of rho_ref=1 — the measured box-edge peak scale
            obs_p99 = _percentile_clims(schlieren_2d, obs_2d, 0.01, 0.99; margin=0.0)[2]
            color_ceiling = Float64(max(physical_threshold, Float32(obs_p99), 1.0f-8))
            data_2d = schlieren_2d
            _mark_invalid_nonobstacle!(data_2d, obs_2d, 0.0f0)
            clims = (0.0, color_ceiling)
            title_str = @sprintf("Numerical Schlieren |grad rho| (linear, calibrated) [ceil=%.1e, 0.5%%rho=%.1e] -- %s -- step %d",
                                 color_ceiling, physical_threshold, slice_desc, t_step)
            fname = @sprintf("%s/slice_schlieren_calibrated_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :schlieren_support
            data_2d = schlieren_support_2d
            clims = (0.0, 1.0)
            title_str = @sprintf("Schlieren derivative support mask (1=valid) -- %s -- step %d", slice_desc, t_step)
            fname = @sprintf("%s/slice_schlieren_support_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :vorticity
            hi_w = max(_finite_abs_max(vorticity_2d, obs_2d), 1e-8)
            data_2d = _display_signed_asinh_map(vorticity_2d, obs_2d, hi_w; strength=16.0f0)
            clims = (-1.0, 1.0)
            vort_comp = slice_meta.plane == :xy ? "omega_z" :
                        slice_meta.plane == :yz ? "omega_x" : "omega_y"
            title_str = @sprintf("Slice-normal vorticity %s (signed asinh contrast) -- %s -- step %d",
                                 vort_comp, slice_desc, t_step)
            fname = @sprintf("%s/slice_vorticity_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :pressure
            data_2d = pressure_2d
            clims = _percentile_clims(pressure_2d, obs_2d, 0.01, 0.99; margin=0.05)
            p_desc = EOS_THERMODYNAMIC ? "Pressure p = (gamma-1)*rho*cv*T" : "Pressure p = rho*cs^2*T"
            title_str = @sprintf("%s — %s — step %d", p_desc, slice_desc, t_step)
            fname = @sprintf("%s/slice_pressure_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :pressure_rel
            data_2d = pressure_rel_2d
            clims = (0.85, 1.15)
            title_str = @sprintf("Pressure ratio p/p_inf (fixed scale) — %s — step %d", slice_desc, t_step)
            fname = @sprintf("%s/slice_pressure_rel_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :cp
            data_2d = cp_2d
            clims = (-3.0, 2.0)
            title_str = @sprintf("Pressure coefficient Cp (fixed scale) — %s — step %d", slice_desc, t_step)
            fname = @sprintf("%s/slice_cp_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :pi_neq
            if !any(isfinite, pinq_mag_2d)
                continue
            end
            data_2d = pinq_mag_2d
            clims = (0.0, max(_finite_max(pinq_mag_2d, obs_2d), 1e-8))
            title_str = @sprintf("|Π_neq| (raw linear scale) — %s — step %d", slice_desc, t_step)
            fname = @sprintf("%s/slice_pi_neq_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :level
            data_2d = owner_level_2d
            clims = (1.0, max(1.0, Float64(length(grids))))
            title_str = @sprintf("AMR owning level â€” %s â€” step %d", slice_desc, t_step)
            fname = @sprintf("%s/slice_level_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :sample_mode
            data_2d = sample_mode_2d
            clims = (0.0, 5.0)
            title_str = @sprintf("Slice sample mode (2=interp, 3=nearest, 4=blend, 5=edge) â€” %s â€” step %d", slice_desc, t_step)
            fname = @sprintf("%s/slice_sample_mode_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :local_shift
            data_2d = local_shift_2d
            clims = (0.0, max(1.0, Float64(LOCAL_LATTICE_SHIFT_U_MAX)))
            title_str = @sprintf("Local lattice shift candidate Ux -- %s -- step %d", slice_desc, t_step)
            fname = @sprintf("%s/slice_local_shift_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :amr_edge
            data_2d = amr_edge_2d
            clims = (0.0, 1.0)
            title_str = @sprintf("AMR owner-edge mask -- %s -- step %d", slice_desc, t_step)
            fname = @sprintf("%s/slice_amr_edge_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :wall_dist
            # Near-wall band: distance (m) from each fluid cell to its nearest
            # obstacle-cell neighbour. Computed in domain_generation.jl only
            # for fluid cells with at least one obstacle neighbour in the 3×3×3
            # stencil — values are ~dx, sqrt(2)·dx, or sqrt(3)·dx; cells outside
            # the band (and obstacle cells) carry the sentinel 100.0f0 which we
            # filter to NaN for display. This band IS the set of cells where
            # Bouzidi BC operates: comparing it to slice_obstacle / slice_level
            # shows where the numerical wall actually lives on each AMR level.
            wd_filtered = copy(wall_dist_2d)
            @inbounds for i in eachindex(wd_filtered)
                v = wd_filtered[i]
                if !isfinite(v) || v >= 50.0f0
                    wd_filtered[i] = NaN32
                end
            end
            data_2d = wd_filtered
            wd_max = _finite_max(wd_filtered, falses(img_w, img_h))
            wd_lim = Float64(max(wd_max, eps(Float32)))
            clims = (0.0, wd_lim)
            title_str = @sprintf("Near-wall band — distance to nearest obstacle cell (Bouzidi-active fluid cells) -- %s -- step %d", slice_desc, t_step)
            fname = @sprintf("%s/slice_wall_dist_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :amr_rho_jump
            hi_j = max(_finite_max(amr_rho_jump_2d, obs_2d), 1e-10)
            data_2d = _display_asinh_map(amr_rho_jump_2d, obs_2d, hi_j; strength=24.0f0)
            clims = (0.0, 1.0)
            title_str = @sprintf("AMR rho jump across owner changes (asinh) -- %s -- step %d", slice_desc, t_step)
            fname = @sprintf("%s/slice_amr_rho_jump_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :amr_u_jump
            hi_j = max(_finite_max(amr_u_jump_2d, obs_2d), 1e-10)
            data_2d = _display_asinh_map(amr_u_jump_2d, obs_2d, hi_j; strength=24.0f0)
            clims = (0.0, 1.0)
            title_str = @sprintf("AMR |du| jump across owner changes (asinh) -- %s -- step %d", slice_desc, t_step)
            fname = @sprintf("%s/slice_amr_u_jump_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :amr_x_u_jump
            hi_j = max(_finite_max(amr_x_u_jump_2d, obs_2d), 1e-10)
            data_2d = _display_asinh_map(amr_x_u_jump_2d, obs_2d, hi_j; strength=24.0f0)
            clims = (0.0, 1.0)
            title_str = @sprintf("AMR |du| jump on %s-neighbor owner changes (asinh) -- %s -- step %d", plane_u_axis, slice_desc, t_step)
            fname = @sprintf("%s/slice_amr_x_u_jump_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :amr_y_ux_jump
            hi_j = max(_finite_max(amr_y_ux_jump_2d, obs_2d), 1e-10)
            data_2d = _display_asinh_map(amr_y_ux_jump_2d, obs_2d, hi_j; strength=24.0f0)
            clims = (0.0, 1.0)
            title_str = @sprintf("AMR |dux| jump on %s-neighbor owner changes (asinh) -- %s -- step %d", plane_v_axis, slice_desc, t_step)
            fname = @sprintf("%s/slice_amr_y_ux_jump_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :velocity_highpass
            hi_j = max(_finite_max(velocity_highpass_2d, obs_2d), 1e-10)
            data_2d = _display_asinh_map(velocity_highpass_2d, obs_2d, hi_j; strength=24.0f0)
            clims = (0.0, 1.0)
            title_str = @sprintf("Owner-local velocity high-pass (asinh) -- %s -- step %d", slice_desc, t_step)
            fname = @sprintf("%s/slice_velocity_highpass_%06d%s.png", output_dir, t_step, fname_suffix)
        elseif f == :velocity_highpass_calibrated
            # ── Calibrated owner-local velocity high-pass ────────────────────────
            # Renders the high-pass on a LINEAR absolute scale, NOT asinh-amplified.
            # The asinh strength=24 variant above maps any cell-by-cell variation
            # — including ~1e-6 numerical noise floor — into prominent visual
            # structure. That can mislead the reader into chasing sub-physical-
            # amplitude features (e.g. tip-vortex acoustic emission at ~1e-4 of
            # U_inlet that the lattice represents at Nyquist scale).
            #
            # This calibrated rendering uses:
            #   - LINEAR scaling (no asinh amplification)
            #   - color ceiling = max(0.5 % of U_inlet, observed max in field):
            #     reproducible threshold tied to physics, not to noise floor.
            #   - colorbar shows ABSOLUTE high-pass amplitude (lattice units)
            # So a value at the 1e-6 noise floor renders as 0.02 % of the bar
            # (essentially black), while real BL/shock structure at 1e-3+ stays
            # visible. The viewer reads amplitude directly from the colorbar.
            #
            # Use this rendering for "is there a real flow feature here?"; use
            # the asinh variant for "where is the lattice's highest noise level?"
            physical_threshold = 0.005f0 * U_TARGET   # 0.5 % of inlet U
            observed_max = max(_finite_max(velocity_highpass_2d, obs_2d), 1e-10)
            color_ceiling = Float64(max(physical_threshold, observed_max))
            data_2d = velocity_highpass_2d
            clims = (0.0, color_ceiling)
            title_str = @sprintf("Owner-local velocity high-pass (linear, calibrated) [ceil=%.1e, 0.5%%·U_inlet=%.1e] -- %s -- step %d",
                                 color_ceiling, physical_threshold, slice_desc, t_step)
            fname = @sprintf("%s/slice_velocity_highpass_calibrated_%06d%s.png", output_dir, t_step, fname_suffix)
        else
            data_2d = vel_mag
            lo, hi = _percentile_clims(vel_mag, obs_2d, 0.01, 0.99; margin=0.05)
            clims = (max(0.0, lo), hi)
            title_str = @sprintf("Velocity |u| — %s — step %d", slice_desc, t_step)
            fname = @sprintf("%s/slice_velocity_%06d%s.png", output_dir, t_step, fname_suffix)
        end

        cmap_use = f == :vorticity ? cgrad([:blue, :black, :red]) :
                   ((f == :level || f == :sample_mode || f == :local_shift || f == :amr_edge || f == :schlieren_support) ? :viridis :
                   ((f == :schlieren || f == :schlieren_calibrated || f == :amr_rho_jump || f == :amr_u_jump ||
                      f == :amr_x_u_jump || f == :amr_y_ux_jump || f == :velocity_highpass ||
                      f == :velocity_highpass_calibrated || f == :wall_dist) ? :inferno :
                   ((f == :pi_neq) ? :grays : :jet)))
        p = _make_heatmap(data_2d, obs_2d, title_str, clims;
                          cmap=cmap_use,
                          x_label=slice_meta.axis_labels[1],
                          y_label=slice_meta.axis_labels[2])
        savefig(p, fname)
        @printf("[Slice] Exported %s: %s (%d×%d px)\n", String(f), fname, img_w, img_h)
    end

    closeall()
end
