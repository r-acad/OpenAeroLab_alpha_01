#
# B7 — Gold-standard curved-wall BC ladder
# ==========================================
#
# Two siblings to the legacy linear Bouzidi (BFL 2001 linear variant) in
# bouzidi_kernel.jl. Both keep the same q-map, the same boundary-cell
# table, and the same fallback machinery; they differ only in the
# population-reconstruction formula at each wall-cut link.
#
# Tier A — YMLS-quadratic (Yu-Mei-Luo-Shyy 2003 / BFL-quadratic 2001)
# -------------------------------------------------------------------
# 3-point interpolation, uniformly 2nd-order in space, smooth across q=1/2:
#
#   q < 1/2:
#     f_ī(x_f, t+1) =  q·(1+2q)         · f̃_i(x_f)
#                    + (1+2q)·(1-2q)    · f̃_i(x_ff)
#                    - q·(1-2q)         · f̃_i(x_fff)
#
#   q ≥ 1/2:
#     f_ī(x_f, t+1) =  1/(q·(2q+1))     · f̃_i(x_f)
#                    + (2q-1)/q         · f̃_ī(x_f)
#                    - (2q-1)/(2q+1)    · f̃_ī(x_ff)
#
# where ī = opp(i), x_ff = x_f + c_ī (one cell deeper into the fluid),
# x_fff = x_f + 2·c_ī. f̃ is the post-collision distribution.
#
# Reference: Bouzidi, Firdaouss, Lallemand 2001, Phys. Fluids 13, 3452;
# Yu, Mei, Luo, Shyy 2003, Prog. Aerosp. Sci. 39, 329 (the "unified
# treatment" that combines BFL-quadratic with moving-wall and force-
# evaluation extensions).
#
# Tier B — Marson 2021 enriched bounce-back
# ------------------------------------------
# Adds isotropy-correcting consistency conditions to YMLS that remove
# the rotational asymmetry classical BFL/YMLS exhibits on 3D curved
# walls. The enrichment is a per-direction additive term computed from
# the local wall geometry (q gradients across the 3D direction set) and
# enforces that the recovered stress is invariant under domain rotation.
#
# Reference: Marson, Thorimbert, Chopard, Ginzburg, Latt 2021,
# "Enhanced single-node lattice Boltzmann boundary condition for fluid
# flows", Phys. Rev. E 103, 053308.
#
# In this implementation the enrichment is structured as a SCAFFOLD on
# top of the YMLS-quadratic kernel: when `bc_scheme = marson_enriched`,
# the YMLS reconstruction is computed AND a Marson correction term is
# added. The correction term currently uses the simplified single-cell
# form (Marson 2021 §3.2 Eq. 21) which captures the leading rotational-
# bias correction; the full multi-cell consistency closure (their §3.3)
# is a follow-up.
#
# Fallback behaviour
# ------------------
# Both kernels gracefully degrade to the legacy linear formula when the
# 3-point stencil isn't available (x_fff or x_ff is solid / OOB / has no
# neighbor block). This preserves correctness at thin-wall configurations
# where the higher-order stencil would otherwise read into the obstacle.
# The `ff_fallback_mode` parameter (wall_eq / mirror / silent) controls
# the bottom-floor fallback in turn.

using KernelAbstractions
using CUDA

# ─────────────────────────────────────────────────────────────────────────
# Shared helper: read f at a cross-block neighbor position.
# Returns (value, ok_flag). ok_flag=true if the neighbor cell exists AND
# is fluid (not obstacle). ok_flag=false signals "use fallback".
# ─────────────────────────────────────────────────────────────────────────
@inline function _read_f_neighbor_fluid(f_post, obstacle, neighbor_table,
                                         x, y, z, b_idx,
                                         nx, ny, nz,
                                         block_size, k)
    if nx >= 1 && nx <= block_size && ny >= 1 && ny <= block_size && nz >= 1 && nz <= block_size
        if !obstacle[nx, ny, nz, b_idx]
            return (f_post[nx, ny, nz, b_idx, k], true)
        else
            return (Float32(0.0), false)
        end
    end
    nb_off_x = (nx < 1) ? Int32(-1) : (nx > block_size ? Int32(1) : Int32(0))
    nb_off_y = (ny < 1) ? Int32(-1) : (ny > block_size ? Int32(1) : Int32(0))
    nb_off_z = (nz < 1) ? Int32(-1) : (nz > block_size ? Int32(1) : Int32(0))
    dir_idx = (nb_off_x + Int32(1)) + (nb_off_y + Int32(1)) * Int32(3) +
              (nb_off_z + Int32(1)) * Int32(9) + Int32(1)
    nb_block = neighbor_table[b_idx, dir_idx]
    if nb_block > 0
        nnx = nx < 1 ? nx + block_size : (nx > block_size ? nx - block_size : nx)
        nny = ny < 1 ? ny + block_size : (ny > block_size ? ny - block_size : ny)
        nnz = nz < 1 ? nz + block_size : (nz > block_size ? nz - block_size : nz)
        if !obstacle[nnx, nny, nnz, nb_block]
            return (f_post[nnx, nny, nnz, nb_block, k], true)
        end
    end
    return (Float32(0.0), false)
end

# ─────────────────────────────────────────────────────────────────────────
# YMLS-quadratic kernel (B7 Tier A)
# ─────────────────────────────────────────────────────────────────────────
@kernel function bouzidi_ymls_correction_kernel_fixed!(
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
    ff_fallback_mode::Int32,
    wall_eq_arr,
    wall_eq_frame_count::Int32,
    shifted_wall_correction_scale::Float32,
    shifted_wall_correction_cap_rel::Float32,
    shifted_wall_correction_cap_abs::Float32,
    wall_pop_limiter_rel_cap::Float32,
    wall_pop_limiter_abs_cap::Float32,
    local_shift_active::Int32,
    # Tier B Marson enrichment flag: 0 = pure YMLS, 1 = + Marson correction
    marson_enriched::Int32
)
    cell_lin = @index(Global, Linear)   # 2026-10-27: plain `x = @index(...)` -- the KernelAbstractions CPU backend splices the work-item index only into that form (else MethodError on the CPU)
    cell_idx = cell_lin + cell_offset

    @inbounds begin
        b_idx = cell_block[cell_idx]
        x = cell_x[cell_idx]
        y = cell_y[cell_idx]
        z = cell_z[cell_idx]

        # Lazy local density for the wall_eq fallback.
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
            (q <= q_min_threshold || q > 1.0f0) && continue

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

            # f_k at x_f (post-collision, direction toward wall)
            f_k = f_post_collision[x, y, z, b_idx, k]

            # First neighbour: x_ff = x_f + c_out_k (one cell deeper into fluid)
            ncx, ncy, ncz = _wall_effective_c(cx_arr, cy_arr, cz_arr, out_k,
                                              array_shift_x_i, array_shift_y_i, array_shift_z_i,
                                              frame_shift_x_i, frame_shift_y_i, frame_shift_z_i)
            nx1 = x + ncx
            ny1 = y + ncy
            nz1 = z + ncz

            # Second neighbour: x_fff = x_f + 2·c_out_k
            nx2 = x + Int32(2) * ncx
            ny2 = y + Int32(2) * ncy
            nz2 = z + Int32(2) * ncz

            if q < 0.5f0
                # ── q < 1/2: 3-point upstream extrapolation on f_k ──
                #   f_ī(x_f, t+1) = q(1+2q)·f̃_k(x_f) + (1+2q)(1-2q)·f̃_k(x_ff) - q(1-2q)·f̃_k(x_fff)
                (f_ff, have_ff) = _read_f_neighbor_fluid(
                    f_post_collision, obstacle, neighbor_table,
                    x, y, z, b_idx, nx1, ny1, nz1, block_size, k)
                (f_fff, have_fff) = _read_f_neighbor_fluid(
                    f_post_collision, obstacle, neighbor_table,
                    x, y, z, b_idx, nx2, ny2, nz2, block_size, k)

                # Bottom-floor fallback for x_ff when neither neighbour is fluid.
                if !have_ff
                    if ff_fallback_mode == Int32(1)
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
                    elseif ff_fallback_mode == Int32(2)
                        f_ff = f_post_collision[x, y, z, b_idx, out_k]
                    else
                        f_ff = f_k
                    end
                end

                if have_fff
                    # Full 3-point quadratic
                    one_plus_2q = 1.0f0 + 2.0f0 * q
                    one_minus_2q = 1.0f0 - 2.0f0 * q
                    a1 = q * one_plus_2q
                    a2 = one_plus_2q * one_minus_2q
                    a3 = -q * one_minus_2q
                    f_out[x, y, z, b_idx, out_k] = a1 * f_k + a2 * f_ff + a3 * f_fff
                else
                    # Degrade to legacy linear Bouzidi (2-point) when x_fff missing.
                    # This preserves correctness on thin walls / sharp edges
                    # where the third upstream cell is solid.
                    coeff1 = 2.0f0 * q
                    f_out[x, y, z, b_idx, out_k] = coeff1 * f_k + (1.0f0 - coeff1) * f_ff
                end
            else
                # ── q ≥ 1/2: 3-point form on f_opp ──
                #   f_ī(x_f, t+1) = 1/(q(2q+1))·f̃_k(x_f) + (2q-1)/q·f̃_ī(x_f) - (2q-1)/(2q+1)·f̃_ī(x_ff)
                f_opp_post = f_post_collision[x, y, z, b_idx, out_k]
                (f_opp_ff, have_opp_ff) = _read_f_neighbor_fluid(
                    f_post_collision, obstacle, neighbor_table,
                    x, y, z, b_idx, nx1, ny1, nz1, block_size, out_k)

                two_q = 2.0f0 * q
                two_q_minus_1 = two_q - 1.0f0
                two_q_plus_1 = two_q + 1.0f0
                if have_opp_ff
                    a1 = 1.0f0 / (q * two_q_plus_1)
                    a2 = two_q_minus_1 / q
                    a3 = -two_q_minus_1 / two_q_plus_1
                    f_out[x, y, z, b_idx, out_k] = a1 * f_k + a2 * f_opp_post + a3 * f_opp_ff
                else
                    # Degrade to legacy linear Bouzidi (2-point) when x_ff missing.
                    inv_2q = 1.0f0 / two_q
                    coeff2 = two_q_minus_1 * inv_2q
                    f_out[x, y, z, b_idx, out_k] = inv_2q * f_k + coeff2 * f_opp_post
                end
            end

            # ── Tier B: Marson 2021 enriched bounce-back ──
            # Additive isotropy-correcting term. Uses the local q at the
            # OPPOSITE wall-cut direction (when present) to estimate the
            # wall-normal asymmetry of the q-stencil and compensate the
            # rotational bias. This is the simplified single-cell form
            # (Marson 2021 §3.2 Eq. 21) — the full multi-cell consistency
            # closure (their §3.3) is a follow-up.
            #
            # When `marson_enriched == 0` this branch is skipped → kernel
            # behaves exactly as YMLS-quadratic.
            if marson_enriched == Int32(1)
                # Asymmetry probe: q in direction k vs q in the reflected
                # output direction. In a shifted lattice this differs from the
                # unshifted OPP direction.
                # On a perfectly symmetric stencil these match (the wall is
                # equidistant from x_f in both directions). For curved walls
                # at 3D corners the q's differ; the difference is the
                # rotational bias the enrichment compensates for.
                q_opp = Float32(q_map[x, y, z, b_idx, q_frame_offset + out_k])
                if q_opp > q_min_threshold && q_opp <= 1.0f0
                    # Per-link wall-equilibrium reference: f_eq projected on
                    # the no-slip wall condition (u_w = 0). For an isothermal
                    # adiabatic wall this is just w_k · ρ.
                    if !rho_f_valid
                        s = 0.0f0
                        for kk in 1:n_dirs
                            s += f_post_collision[x, y, z, b_idx, kk]
                        end
                        rho_f = s
                        rho_f_valid = true
                    end
                    f_wall_eq = _shifted_wall_eq_value(
                        rho_f, wall_eq_arr, cx_arr, w_arr, wall_eq_frame_count,
                        frame_shift_x_i,
                        Int32(out_k), shifted_wall_correction_scale,
                        shifted_wall_correction_cap_rel, shifted_wall_correction_cap_abs)
                    # Asymmetry-weighted correction. The factor 1/(q + q_opp)
                    # normalises by the total stencil span; the (q - q_opp)
                    # numerator gives the wall-normal asymmetry. The pre-
                    # factor 0.5 keeps the correction in the leading-order
                    # range Marson 2021 derives. Bounded to ±10% of f_wall_eq
                    # so a noisy q-pair can't blow up the wall stress.
                    denom = max(q + q_opp, 1.0f-3)
                    asym = (q - q_opp) / denom
                    corr = 0.5f0 * asym * (f_wall_eq - f_out[x, y, z, b_idx, out_k])
                    cap = 0.1f0 * abs(f_wall_eq) + 1.0f-9
                    corr = clamp(corr, -cap, cap)
                    f_out[x, y, z, b_idx, out_k] = f_out[x, y, z, b_idx, out_k] + corr
                end
            end

            if shifted_wall_correction_scale > 0.0f0
                if !rho_f_valid
                    s = 0.0f0
                    for kk in 1:n_dirs
                        s += f_post_collision[x, y, z, b_idx, kk]
                    end
                    rho_f = s
                    rho_f_valid = true
                end
                f_out[x, y, z, b_idx, out_k] = _apply_shifted_wall_correction(
                    f_out[x, y, z, b_idx, out_k],
                    rho_f, wall_eq_arr, cx_arr, w_arr, wall_eq_frame_count,
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
                f_out[x, y, z, b_idx, out_k] = _limit_wall_population(
                    f_out[x, y, z, b_idx, out_k],
                    rho_f, wall_eq_arr, cx_arr, w_arr, wall_eq_frame_count,
                    frame_shift_x_i, Int32(out_k),
                    shifted_wall_correction_scale,
                    shifted_wall_correction_cap_rel,
                    shifted_wall_correction_cap_abs,
                    wall_pop_limiter_rel_cap,
                    wall_pop_limiter_abs_cap)
            end
        end
    end
end

"""
    apply_bouzidi_ymls_correction!(...)

Drive the YMLS-quadratic boundary kernel (B7 Tier A) and, when
`marson_enriched=Int32(1)`, the Marson 2021 enrichment on top of it
(B7 Tier B). Call this in place of `apply_bouzidi_correction!` when
the `BC_SCHEME` selects `ymls_quadratic` or `marson_enriched`.
"""
function apply_bouzidi_ymls_correction!(f_out, f_post_collision, obstacle,
                                        q_map, cell_block, cell_x, cell_y, cell_z, n_boundary_cells,
                                        neighbor_table,
                                        block_size::Int,
                                        cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu,
                                        q_min_threshold::Float32,
                                        backend;
                                        cell_offset::Int32 = Int32(0),
                                        ndrange_cells::Int = n_boundary_cells,
                                        ff_fallback_mode::Int32 = Int32(0),
                                        wall_eq_arr = w_gpu,
                                        wall_eq_frame_count::Int32 = Int32(1),
                                        shifted_wall_correction_scale::Float32 = 0.0f0,
                                        shifted_wall_correction_cap_rel::Float32 = 2.0f0,
                                        shifted_wall_correction_cap_abs::Float32 = 0.0f0,
                                        wall_pop_limiter_rel_cap::Float32 = 0.0f0,
                                        wall_pop_limiter_abs_cap::Float32 = 0.0f0,
                                        local_shift_active::Int32 = Int32(0),
                                        block_shift_arr = cell_block,
                                        marson_enriched::Int32 = Int32(0))

    if ndrange_cells == 0
        return
    end

    kernel! = bouzidi_ymls_correction_kernel_fixed!(backend)
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
            ff_fallback_mode,
            wall_eq_arr,
            wall_eq_frame_count,
            shifted_wall_correction_scale,
            shifted_wall_correction_cap_rel,
            shifted_wall_correction_cap_abs,
            wall_pop_limiter_rel_cap,
            wall_pop_limiter_abs_cap,
            local_shift_active,
            marson_enriched,
            ndrange=(ndrange_cells,))
end
