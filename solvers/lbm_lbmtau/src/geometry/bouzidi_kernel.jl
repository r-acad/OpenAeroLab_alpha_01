// # FILE: .\src\bouzidi_kernel.jl
using KernelAbstractions
using CUDA

@inline function _shifted_wall_eq_index(frame_x::Int32, frame_count::Int32, k::Int32)
    if frame_x <= Int32(0) || frame_count <= Int32(1)
        return k
    end
    frame_x = min(frame_x, frame_count - Int32(1))
    return frame_x * Int32(39) + k
end

@inline function _shifted_wall_eq_value(rho_f::Float32,
                                        wall_eq_arr,
                                        cx_arr,
                                        w_arr,
                                        frame_count::Int32,
                                        frame_x::Int32,
                                        k::Int32,
                                        correction_scale::Float32,
                                        cap_rel::Float32,
                                        cap_abs::Float32)
    base = rho_f * Float32(w_arr[k])
    if correction_scale <= 0.0f0
        return base
    end
    if frame_x <= Int32(0)
        return base
    end
    target = rho_f * Float32(wall_eq_arr[_shifted_wall_eq_index(frame_x, frame_count, k)])
    delta = correction_scale * (target - base)
    cap = max(cap_abs, cap_rel * max(max(abs(base), abs(target)), 1.0f-12))
    delta = clamp(delta, -cap, cap)
    return max(base + delta, 1.0f-12)
end

@inline function _apply_shifted_wall_correction(raw::Float32,
                                                rho_f::Float32,
                                                wall_eq_arr,
                                                cx_arr,
                                                w_arr,
                                                frame_count::Int32,
                                                frame_x::Int32,
                                                k_out::Int32,
                                                correction_scale::Float32,
                                                cap_rel::Float32,
                                                cap_abs::Float32)
    if correction_scale <= 0.0f0
        return max(raw, 1.0f-12)
    end
    if frame_x <= Int32(0)
        return max(raw, 1.0f-12)
    end
    # Shifted lattices make even the base rest population a streaming link.
    # Relax the returned population toward the stationary-wall equilibrium for
    # its actual output direction, with a per-step cap to avoid Z031-style kicks.
    target = rho_f * Float32(wall_eq_arr[_shifted_wall_eq_index(frame_x, frame_count, k_out)])
    delta = correction_scale * (target - raw)
    scale = max(max(abs(raw), abs(target)), 1.0f-12)
    cap = max(cap_abs, cap_rel * scale)
    delta = clamp(delta, -cap, cap)
    return max(raw + delta, 1.0f-12)
end

@inline function _limit_wall_population(raw::Float32,
                                        rho_f::Float32,
                                        wall_eq_arr,
                                        cx_arr,
                                        w_arr,
                                        frame_count::Int32,
                                        frame_x::Int32,
                                        k_out::Int32,
                                        shifted_correction_scale::Float32,
                                        shifted_cap_rel::Float32,
                                        shifted_cap_abs::Float32,
                                        limiter_rel_cap::Float32,
                                        limiter_abs_cap::Float32)
    if limiter_rel_cap <= 0.0f0 && limiter_abs_cap <= 0.0f0
        return max(raw, 1.0f-12)
    end
    target = _shifted_wall_eq_value(
        rho_f, wall_eq_arr, cx_arr, w_arr, frame_count, frame_x, k_out,
        shifted_correction_scale, shifted_cap_rel, shifted_cap_abs)
    scale = max(abs(target), 1.0f-12)
    cap = max(limiter_abs_cap, limiter_rel_cap * scale)
    return max(clamp(raw, target - cap, target + cap), 1.0f-12)
end

@inline function _wall_effective_c(cx_arr, cy_arr, cz_arr,
                                   k::Int32,
                                   array_shift_x::Int32, array_shift_y::Int32, array_shift_z::Int32,
                                   frame_shift_x::Int32, frame_shift_y::Int32, frame_shift_z::Int32)
    return (Int32(cx_arr[k]) - array_shift_x + frame_shift_x,
            Int32(cy_arr[k]) - array_shift_y + frame_shift_y,
            Int32(cz_arr[k]) - array_shift_z + frame_shift_z)
end

@inline function _wall_q_frame_offset(frame_shift_x::Int32,
                                      q_map_frame_count::Int32,
                                      n_dirs::Int32)
    if q_map_frame_count <= Int32(1)
        return Int32(0)
    end
    frame_x = clamp(frame_shift_x, Int32(0), q_map_frame_count - Int32(1))
    return frame_x * n_dirs
end

@inline function _shifted_wall_exact_reflect_index(cx_arr, cy_arr, cz_arr,
                                                   k::Int32, n_dirs::Int32,
                                                   array_shift_x::Int32, array_shift_y::Int32, array_shift_z::Int32,
                                                   frame_shift_x::Int32, frame_shift_y::Int32, frame_shift_z::Int32)
    # Wall reflection is defined in the lab/streaming velocity, not in the
    # unshifted base velocity.  For shifted D3Q39 this differs from OPP: the
    # reflected partner satisfies c'_j = -c'_k when that velocity exists.
    ckx, cky, ckz = _wall_effective_c(cx_arr, cy_arr, cz_arr, k,
                                      array_shift_x, array_shift_y, array_shift_z,
                                      frame_shift_x, frame_shift_y, frame_shift_z)
    tx = -ckx
    ty = -cky
    tz = -ckz
    for j in Int32(1):n_dirs
        cjx, cjy, cjz = _wall_effective_c(cx_arr, cy_arr, cz_arr, j,
                                          array_shift_x, array_shift_y, array_shift_z,
                                          frame_shift_x, frame_shift_y, frame_shift_z)
        if cjx == tx && cjy == ty && cjz == tz
            return j
        end
    end
    return Int32(0)
end

@inline function _shifted_wall_output_index(cx_arr, cy_arr, cz_arr, opp_arr,
                                            k::Int32, n_dirs::Int32,
                                            array_shift_x::Int32, array_shift_y::Int32, array_shift_z::Int32,
                                            frame_shift_x::Int32, frame_shift_y::Int32, frame_shift_z::Int32)
    exact = _shifted_wall_exact_reflect_index(cx_arr, cy_arr, cz_arr, k, n_dirs,
                                              array_shift_x, array_shift_y, array_shift_z,
                                              frame_shift_x, frame_shift_y, frame_shift_z)
    return exact > Int32(0) ? exact : opp_arr[k]
end

"""
BOUZIDI_KERNEL.JL - GPU Correction Kernel

The Bouzidi interpolation scheme:
- For q < 0.5:  f_opp(x_f, t+1) = 2q * f_k(x_f, t*) + (1-2q) * f_k(x_ff, t*)
- For q >= 0.5: f_opp(x_f, t+1) = (1/2q) * f_k(x_f, t*) + (2q-1)/(2q) * f_opp(x_f, t*)
"""

@kernel function bouzidi_correction_kernel_fixed!(
    f_out,
    f_post_collision,
    obstacle,
    q_map,
    cell_block,
    cell_x,
    cell_y,
    cell_z,
    neighbor_table,
    block_size::Int32,
    cx_arr, cy_arr, cz_arr,
    w_arr,                       # pass193: lattice weights (used by wall_eq fallback)
    opp_arr,
    block_shift_arr,
    q_min_threshold::Float32,
    q_map_frame_count::Int32,
    n_dirs::Int32,
    cell_offset::Int32,    # 0 for single-GPU; >0 shifts cell_idx for multi-GPU partitioning
    wall_eq_arr,
    wall_eq_frame_count::Int32,
    shifted_wall_correction_scale::Float32,
    shifted_wall_correction_cap_rel::Float32,
    shifted_wall_correction_cap_abs::Float32,
    wall_pop_limiter_rel_cap::Float32,
    wall_pop_limiter_abs_cap::Float32,
    local_shift_active::Int32,
    # Pass193: q<0.5 behind-cell fallback when x_ff is solid / OOB / no-neighbor.
    #   0 = legacy: f_ff = f_k → silently collapses to halfway BB. Pre-pass193 default.
    #   1 = wall_eq: f_ff = w_k * rho_f → continuum no-slip equilibrium at q→0.
    # See PASS193_DIAGNOSIS.md for the motivation.
    ff_fallback_mode::Int32
)
    cell_lin = @index(Global, Linear)   # 2026-10-27: plain `x = @index(...)` -- the KernelAbstractions CPU backend splices the work-item index only into that form (else MethodError on the CPU)
    cell_idx = cell_lin + cell_offset

    @inbounds begin
        b_idx = cell_block[cell_idx]
        x = cell_x[cell_idx]
        y = cell_y[cell_idx]
        z = cell_z[cell_idx]

        # Lazy local density for the wall_eq fallback. Computed at most once
        # per boundary cell, only if the fallback actually fires.
        rho_f = 0.0f0
        rho_f_valid = false
        array_shift_x_i = Int32(cx_arr[Int32(1)])
        array_shift_y_i = Int32(cy_arr[Int32(1)])
        array_shift_z_i = Int32(cz_arr[Int32(1)])
        frame_shift_x_i = local_shift_active == Int32(1) ? block_shift_arr[b_idx] : array_shift_x_i
        frame_shift_y_i = array_shift_y_i
        frame_shift_z_i = array_shift_z_i
        q_frame_offset = _wall_q_frame_offset(frame_shift_x_i, q_map_frame_count, n_dirs)

        for k in 1:n_dirs
            q = Float32(q_map[x, y, z, b_idx, q_frame_offset + Int32(k)])

            if q > q_min_threshold && q <= 1.0f0
                out_k = _shifted_wall_output_index(cx_arr, cy_arr, cz_arr, opp_arr, Int32(k), n_dirs,
                                                   array_shift_x_i, array_shift_y_i, array_shift_z_i,
                                                   frame_shift_x_i, frame_shift_y_i, frame_shift_z_i)
                exact_out = _shifted_wall_exact_reflect_index(cx_arr, cy_arr, cz_arr, Int32(k), n_dirs,
                                                              array_shift_x_i, array_shift_y_i, array_shift_z_i,
                                                              frame_shift_x_i, frame_shift_y_i, frame_shift_z_i)
                if exact_out == Int32(0)
                    exact_src_for_out = _shifted_wall_exact_reflect_index(cx_arr, cy_arr, cz_arr, out_k, n_dirs,
                                                                          array_shift_x_i, array_shift_y_i, array_shift_z_i,
                                                                          frame_shift_x_i, frame_shift_y_i, frame_shift_z_i)
                    if exact_src_for_out > Int32(0)
                        q_src = Float32(q_map[x, y, z, b_idx, q_frame_offset + exact_src_for_out])
                        if q_src > q_min_threshold && q_src <= 1.0f0
                            continue
                        end
                    end
                end

                # f_k at fluid node x_f (post-collision)
                f_k = f_post_collision[x, y, z, b_idx, k]

                if q < 0.5f0

                    # This is x_ff: the cell "behind" x_f
                    ncx, ncy, ncz = _wall_effective_c(cx_arr, cy_arr, cz_arr, out_k,
                                                      array_shift_x_i, array_shift_y_i, array_shift_z_i,
                                                      frame_shift_x_i, frame_shift_y_i, frame_shift_z_i)

                    nx = x + ncx
                    ny = y + ncy
                    nz = z + ncz

                    f_ff = f_k
                    have_ff_support = false

                    if nx >= 1 && nx <= block_size && ny >= 1 && ny <= block_size && nz >= 1 && nz <= block_size
                        if !obstacle[nx, ny, nz, b_idx]
                            f_ff = f_post_collision[nx, ny, nz, b_idx, k]
                            have_ff_support = true
                        end
                    else
                        nb_off_x = (nx < 1) ? Int32(-1) : (nx > block_size ? Int32(1) : Int32(0))
                        nb_off_y = (ny < 1) ? Int32(-1) : (ny > block_size ? Int32(1) : Int32(0))
                        nb_off_z = (nz < 1) ? Int32(-1) : (nz > block_size ? Int32(1) : Int32(0))

                        dir_idx = (nb_off_x + Int32(1)) + (nb_off_y + Int32(1)) * Int32(3) + (nb_off_z + Int32(1)) * Int32(9) + Int32(1)
                        nb_block = neighbor_table[b_idx, dir_idx]

                        if nb_block > 0
                            nnx = nx < 1 ? nx + block_size : (nx > block_size ? nx - block_size : nx)
                            nny = ny < 1 ? ny + block_size : (ny > block_size ? ny - block_size : ny)
                            nnz = nz < 1 ? nz + block_size : (nz > block_size ? nz - block_size : nz)
                            if !obstacle[nnx, nny, nnz, nb_block]
                                f_ff = f_post_collision[nnx, nny, nnz, nb_block, k]
                                have_ff_support = true
                            end
                        end
                    end

                    # Pass193: when x_ff support is missing, replace the silent
                    # f_ff = f_k collapse (halfway BB) with the local no-slip
                    # equilibrium projection f_ff = w_k * rho_f. This removes
                    # the cell-pair-asymmetric injection mechanism diagnosed in
                    # PASS192_DIAGNOSIS.md (third-channel macroscopic Nyquist).
                    if (!have_ff_support) && ff_fallback_mode == Int32(1)
                        if !rho_f_valid
                            s = 0.0f0
                            for kk in 1:n_dirs
                                s += f_post_collision[x, y, z, b_idx, kk]
                            end
                            rho_f = s
                            rho_f_valid = true
                        end
                        f_ff = _shifted_wall_eq_value(
                            rho_f, wall_eq_arr, cx_arr, w_arr, wall_eq_frame_count,
                            frame_shift_x_i,
                            Int32(k), shifted_wall_correction_scale,
                            shifted_wall_correction_cap_rel, shifted_wall_correction_cap_abs)
                    elseif (!have_ff_support) && ff_fallback_mode == Int32(2)
                        f_ff = f_post_collision[x, y, z, b_idx, out_k]
                    end

                    # Interpolation for q < 0.5
                    coeff1 = 2.0f0 * q
                    raw = coeff1 * f_k + (1.0f0 - coeff1) * f_ff
                    if shifted_wall_correction_scale > 0.0f0
                        if !rho_f_valid
                            s = 0.0f0
                            for kk in 1:n_dirs
                                s += f_post_collision[x, y, z, b_idx, kk]
                            end
                            rho_f = s
                            rho_f_valid = true
                        end
                        raw = _apply_shifted_wall_correction(
                            raw, rho_f, wall_eq_arr, cx_arr, w_arr, wall_eq_frame_count,
                            frame_shift_x_i,
                            Int32(out_k), shifted_wall_correction_scale,
                            shifted_wall_correction_cap_rel, shifted_wall_correction_cap_abs)
                    end
                    if wall_pop_limiter_rel_cap > 0.0f0 || wall_pop_limiter_abs_cap > 0.0f0
                        if !rho_f_valid
                            s = 0.0f0
                            for kk in 1:n_dirs
                                s += f_post_collision[x, y, z, b_idx, kk]
                            end
                            rho_f = s
                            rho_f_valid = true
                        end
                        raw = _limit_wall_population(
                            raw, rho_f, wall_eq_arr, cx_arr, w_arr, wall_eq_frame_count,
                            frame_shift_x_i, Int32(out_k),
                            shifted_wall_correction_scale,
                            shifted_wall_correction_cap_rel,
                            shifted_wall_correction_cap_abs,
                            wall_pop_limiter_rel_cap,
                            wall_pop_limiter_abs_cap)
                    end
                    f_out[x, y, z, b_idx, out_k] = raw

                else

                    # Interpolation for q >= 0.5
                    f_opp_post = f_post_collision[x, y, z, b_idx, out_k]

                    inv_2q = 1.0f0 / (2.0f0 * q)
                    coeff2 = (2.0f0 * q - 1.0f0) * inv_2q

                    raw = inv_2q * f_k + coeff2 * f_opp_post
                    if shifted_wall_correction_scale > 0.0f0
                        if !rho_f_valid
                            s = 0.0f0
                            for kk in 1:n_dirs
                                s += f_post_collision[x, y, z, b_idx, kk]
                            end
                            rho_f = s
                            rho_f_valid = true
                        end
                        raw = _apply_shifted_wall_correction(
                            raw, rho_f, wall_eq_arr, cx_arr, w_arr, wall_eq_frame_count,
                            frame_shift_x_i,
                            Int32(out_k), shifted_wall_correction_scale,
                            shifted_wall_correction_cap_rel, shifted_wall_correction_cap_abs)
                    end
                    if wall_pop_limiter_rel_cap > 0.0f0 || wall_pop_limiter_abs_cap > 0.0f0
                        if !rho_f_valid
                            s = 0.0f0
                            for kk in 1:n_dirs
                                s += f_post_collision[x, y, z, b_idx, kk]
                            end
                            rho_f = s
                            rho_f_valid = true
                        end
                        raw = _limit_wall_population(
                            raw, rho_f, wall_eq_arr, cx_arr, w_arr, wall_eq_frame_count,
                            frame_shift_x_i, Int32(out_k),
                            shifted_wall_correction_scale,
                            shifted_wall_correction_cap_rel,
                            shifted_wall_correction_cap_abs,
                            wall_pop_limiter_rel_cap,
                            wall_pop_limiter_abs_cap)
                    end
                    f_out[x, y, z, b_idx, out_k] = raw
                end
            end
        end
    end
end

"""
    apply_bouzidi_correction!(f_out, f_post_collision, ...)

Apply Bouzidi boundary correction. Call AFTER the main stream-collide kernel.
"""
function apply_bouzidi_correction!(f_out, f_post_collision, obstacle,
                                   q_map, cell_block, cell_x, cell_y, cell_z, n_boundary_cells,
                                   neighbor_table,
                                   block_size::Int,
                                   cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu,
                                    q_min_threshold::Float32,
                                    backend;
                                    cell_offset::Int32 = Int32(0),
                                    ndrange_cells::Int = n_boundary_cells,
                                    wall_eq_arr = w_gpu,
                                    wall_eq_frame_count::Int32 = Int32(1),
                                    shifted_wall_correction_scale::Float32 = 0.0f0,
                                    shifted_wall_correction_cap_rel::Float32 = 2.0f0,
                                    shifted_wall_correction_cap_abs::Float32 = 0.0f0,
                                    wall_pop_limiter_rel_cap::Float32 = 0.0f0,
                                    wall_pop_limiter_abs_cap::Float32 = 0.0f0,
                                    local_shift_active::Int32 = Int32(0),
                                    block_shift_arr = cell_block,
                                    ff_fallback_mode::Int32 = Int32(0))

    if ndrange_cells == 0
        return
    end

    kernel! = bouzidi_correction_kernel_fixed!(backend)
    q_map_frame_count = max(Int32(1), Int32(size(q_map, 5) ÷ Int(N_DIRS)))
    kernel!(f_out, f_post_collision, obstacle,
            q_map,
            cell_block,
            cell_x,
            cell_y,
            cell_z,
            neighbor_table,
            Int32(block_size),
            cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu,
            block_shift_arr,
            q_min_threshold,
            q_map_frame_count,
            N_DIRS,
            cell_offset,
            wall_eq_arr,
            wall_eq_frame_count,
            shifted_wall_correction_scale,
            shifted_wall_correction_cap_rel,
            shifted_wall_correction_cap_abs,
            wall_pop_limiter_rel_cap,
            wall_pop_limiter_abs_cap,
            local_shift_active,
            ff_fallback_mode,
            ndrange=(ndrange_cells,))
end

@kernel function bouzidi_missed_link_fix_kernel!(
    f_out,
    f_post_collision,
    obstacle,
    q_map,
    cell_block,
    cell_x,
    cell_y,
    cell_z,
    neighbor_table,
    block_size::Int32,
    cx_arr, cy_arr, cz_arr,
    w_arr,
    opp_arr,
    block_shift_arr,
    q_min_threshold::Float32,
    q_map_frame_count::Int32,
    n_dirs::Int32,
    cell_offset::Int32,
    wall_eq_arr,
    wall_eq_frame_count::Int32,
    shifted_wall_correction_scale::Float32,
    shifted_wall_correction_cap_rel::Float32,
    shifted_wall_correction_cap_abs::Float32,
    local_shift_active::Int32,
    missed_link_fix_mode::Int32,
    missed_link_eq_blend::Float32,
    missed_link_delta_rel_cap::Float32,
    missed_link_delta_abs_cap::Float32
)
    cell_lin = @index(Global, Linear)   # 2026-10-27: plain `x = @index(...)` -- the KernelAbstractions CPU backend splices the work-item index only into that form (else MethodError on the CPU)
    cell_idx = cell_lin + cell_offset

    @inbounds begin
        b_idx = cell_block[cell_idx]
        x = cell_x[cell_idx]
        y = cell_y[cell_idx]
        z = cell_z[cell_idx]

        rho_f = 0.0f0
        rho_f_valid = false
        array_shift_x_i = Int32(cx_arr[Int32(1)])
        array_shift_y_i = Int32(cy_arr[Int32(1)])
        array_shift_z_i = Int32(cz_arr[Int32(1)])
        frame_shift_x_i = local_shift_active == Int32(1) ? block_shift_arr[b_idx] : array_shift_x_i
        frame_shift_y_i = array_shift_y_i
        frame_shift_z_i = array_shift_z_i
        q_frame_offset = _wall_q_frame_offset(frame_shift_x_i, q_map_frame_count, n_dirs)

        for k in 1:n_dirs
            q = Float32(q_map[x, y, z, b_idx, q_frame_offset + Int32(k)])
            q <= q_min_threshold || q > 1.0f0 || continue

            # If any active wall link reflects into population k, Bouzidi
            # already wrote it. In a shifted lattice this is not generally the
            # unshifted OPP_k, so scan the small direction set in the same
            # shifted-wall output map used by the main wall kernel.
            covered_by_wall_link = false
            for r in 1:n_dirs
                q_r = Float32(q_map[x, y, z, b_idx, q_frame_offset + Int32(r)])
                if q_r > q_min_threshold && q_r <= 1.0f0
                    out_r = _shifted_wall_output_index(cx_arr, cy_arr, cz_arr, opp_arr, Int32(r), n_dirs,
                                                       array_shift_x_i, array_shift_y_i, array_shift_z_i,
                                                       frame_shift_x_i, frame_shift_y_i, frame_shift_z_i)
                    exact_r = _shifted_wall_exact_reflect_index(cx_arr, cy_arr, cz_arr, Int32(r), n_dirs,
                                                                array_shift_x_i, array_shift_y_i, array_shift_z_i,
                                                                frame_shift_x_i, frame_shift_y_i, frame_shift_z_i)
                    if exact_r == Int32(0)
                        exact_src_for_out = _shifted_wall_exact_reflect_index(cx_arr, cy_arr, cz_arr, out_r, n_dirs,
                                                                              array_shift_x_i, array_shift_y_i, array_shift_z_i,
                                                                              frame_shift_x_i, frame_shift_y_i, frame_shift_z_i)
                        if exact_src_for_out > Int32(0)
                            q_src = Float32(q_map[x, y, z, b_idx, q_frame_offset + exact_src_for_out])
                            if q_src > q_min_threshold && q_src <= 1.0f0
                                continue
                            end
                        end
                    end
                    if out_r == Int32(k)
                        covered_by_wall_link = true
                    end
                end
            end
            if covered_by_wall_link
                continue
            end
            opp_k = _shifted_wall_output_index(cx_arr, cy_arr, cz_arr, opp_arr, Int32(k), n_dirs,
                                               array_shift_x_i, array_shift_y_i, array_shift_z_i,
                                               frame_shift_x_i, frame_shift_y_i, frame_shift_z_i)

            ckx, cky, ckz = _wall_effective_c(cx_arr, cy_arr, cz_arr, Int32(k),
                                              array_shift_x_i, array_shift_y_i, array_shift_z_i,
                                              frame_shift_x_i, frame_shift_y_i, frame_shift_z_i)
            nx = x - ckx
            ny = y - cky
            nz = z - ckz

            upstream_solid = false
            if nx >= 1 && nx <= block_size && ny >= 1 && ny <= block_size && nz >= 1 && nz <= block_size
                upstream_solid = obstacle[nx, ny, nz, b_idx]
            else
                nb_off_x = (nx < 1) ? Int32(-1) : (nx > block_size ? Int32(1) : Int32(0))
                nb_off_y = (ny < 1) ? Int32(-1) : (ny > block_size ? Int32(1) : Int32(0))
                nb_off_z = (nz < 1) ? Int32(-1) : (nz > block_size ? Int32(1) : Int32(0))
                dir_idx = (nb_off_x + Int32(1)) + (nb_off_y + Int32(1)) * Int32(3) + (nb_off_z + Int32(1)) * Int32(9) + Int32(1)
                nb_block = neighbor_table[b_idx, dir_idx]

                if nb_block > 0
                    nnx = nx < 1 ? nx + block_size : (nx > block_size ? nx - block_size : nx)
                    nny = ny < 1 ? ny + block_size : (ny > block_size ? ny - block_size : ny)
                    nnz = nz < 1 ? nz + block_size : (nz > block_size ? nz - block_size : nz)
                    upstream_solid = obstacle[nnx, nny, nnz, nb_block]
                end
            end

            if upstream_solid
                if missed_link_fix_mode == Int32(1) || missed_link_fix_mode == Int32(3) || missed_link_fix_mode == Int32(4)
                    if !rho_f_valid
                        s = 0.0f0
                        for kk in 1:n_dirs
                            s += f_post_collision[x, y, z, b_idx, kk]
                        end
                        rho_f = s
                        rho_f_valid = true
                    end
                    wall_eq = _shifted_wall_eq_value(
                        rho_f, wall_eq_arr, cx_arr, w_arr, wall_eq_frame_count,
                        frame_shift_x_i,
                        Int32(k), shifted_wall_correction_scale,
                        shifted_wall_correction_cap_rel, shifted_wall_correction_cap_abs)
                    if missed_link_fix_mode == Int32(1)
                        f_out[x, y, z, b_idx, k] = wall_eq
                    else
                        bb = f_post_collision[x, y, z, b_idx, opp_k]
                        target = missed_link_eq_blend * wall_eq + (1.0f0 - missed_link_eq_blend) * bb
                        if missed_link_fix_mode == Int32(4)
                            base = f_out[x, y, z, b_idx, k]
                            scale = max(max(abs(base), abs(target)), 1.0f-12)
                            cap = max(missed_link_delta_abs_cap, missed_link_delta_rel_cap * scale)
                            delta = clamp(target - base, -cap, cap)
                            f_out[x, y, z, b_idx, k] = max(base + delta, 1.0f-12)
                        else
                            f_out[x, y, z, b_idx, k] = target
                        end
                    end
                elseif missed_link_fix_mode == Int32(2)
                    f_out[x, y, z, b_idx, k] = f_post_collision[x, y, z, b_idx, opp_k]
                end
            end
        end
    end
end

function apply_bouzidi_missed_link_fix!(f_out, f_post_collision, obstacle,
                                        q_map, cell_block, cell_x, cell_y, cell_z, n_boundary_cells,
                                        neighbor_table,
                                        block_size::Int,
                                        cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu,
                                        q_min_threshold::Float32,
                                        backend;
                                        cell_offset::Int32 = Int32(0),
                                        ndrange_cells::Int = n_boundary_cells,
                                        wall_eq_arr = w_gpu,
                                        wall_eq_frame_count::Int32 = Int32(1),
                                        shifted_wall_correction_scale::Float32 = 0.0f0,
                                        shifted_wall_correction_cap_rel::Float32 = 2.0f0,
                                        shifted_wall_correction_cap_abs::Float32 = 0.0f0,
                                        local_shift_active::Int32 = Int32(0),
                                        block_shift_arr = cell_block,
                                        missed_link_fix_mode::Int32 = Int32(0),
                                        missed_link_eq_blend::Float32 = 0.35f0,
                                        missed_link_delta_rel_cap::Float32 = 0.08f0,
                                        missed_link_delta_abs_cap::Float32 = 2.0f-4)
    if ndrange_cells == 0
        return
    end

    kernel! = bouzidi_missed_link_fix_kernel!(backend)
    q_map_frame_count = max(Int32(1), Int32(size(q_map, 5) ÷ Int(N_DIRS)))
    kernel!(f_out, f_post_collision, obstacle,
            q_map,
            cell_block,
            cell_x,
            cell_y,
            cell_z,
            neighbor_table,
            Int32(block_size),
            cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu,
            block_shift_arr,
            q_min_threshold,
            q_map_frame_count,
            N_DIRS,
            cell_offset,
            wall_eq_arr,
            wall_eq_frame_count,
            shifted_wall_correction_scale,
            shifted_wall_correction_cap_rel,
            shifted_wall_correction_cap_abs,
            local_shift_active,
            missed_link_fix_mode,
            missed_link_eq_blend,
            missed_link_delta_rel_cap,
            missed_link_delta_abs_cap,
            ndrange=(ndrange_cells,))
end

@kernel function thermal_adiabatic_wall_reflection_kernel!(
    g_out,
    g_post_collision,
    q_map,
    cell_block,
    cell_x,
    cell_y,
    cell_z,
    opp_arr,
    q_min_threshold::Float32,
    n_dirs::Int32,
    cell_offset::Int32
)
    cell_lin = @index(Global, Linear)   # 2026-10-27: plain `x = @index(...)` -- the KernelAbstractions CPU backend splices the work-item index only into that form (else MethodError on the CPU)
    cell_idx = cell_lin + cell_offset

    @inbounds begin
        b_idx = cell_block[cell_idx]
        x = cell_x[cell_idx]
        y = cell_y[cell_idx]
        z = cell_z[cell_idx]

        for k in 1:n_dirs
            q = Float32(q_map[x, y, z, b_idx, k])

            if q > q_min_threshold && q <= 1.0f0
                opp_k = opp_arr[k]
                g_reflected = g_post_collision[x, y, z, b_idx, k]
                if g_reflected == g_reflected
                    g_out[x, y, z, b_idx, opp_k] = max(g_reflected, Float32(1e-10))
                end
            end
        end
    end
end

"""
    apply_thermal_adiabatic_bouzidi_correction!(g_out, g_post_collision, ...)

Apply zero-normal-heat-flux reflection on Bouzidi wall links for the thermal DDF.
The momentum BFL interpolation is intentionally not reused here: for an adiabatic
wall the thermal condition is no flux, so reflecting the outgoing thermal
population is the conservative and non-dispersive choice.
"""
function apply_thermal_adiabatic_bouzidi_correction!(g_out, g_post_collision,
                                                     q_map, cell_block, cell_x, cell_y, cell_z, n_boundary_cells,
                                                     opp_gpu,
                                                     q_min_threshold::Float32,
                                                     n_dirs::Int32,
                                                     backend;
                                                     cell_offset::Int32 = Int32(0),
                                                     ndrange_cells::Int = n_boundary_cells)

    if ndrange_cells == 0
        return
    end

    kernel! = thermal_adiabatic_wall_reflection_kernel!(backend)
    kernel!(g_out, g_post_collision,
            q_map,
            cell_block,
            cell_x,
            cell_y,
            cell_z,
            opp_gpu,
            q_min_threshold,
            n_dirs,
            cell_offset,
            ndrange=(ndrange_cells,))
end
