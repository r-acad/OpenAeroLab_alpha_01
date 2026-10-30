# ==============================================================================
# THERMAL T-FILTER — Post-Recovery 6-Neighbor Laplacian (Option B)
# ==============================================================================
# Mitigates visible AMR-interface banding in the temperature field by applying
# a small laplacian smoothing AFTER the thermal kernel recovers T from E_kin.
#
#   T'[i,j,k] = (1-α)·T[i,j,k] + (α/6)·Σ_neighbors T
#
# The mismatch that drives the artifacts is a small (~1%) jump in E_kin at
# coarse-fine interfaces — highly visible because the T range is narrow
# (e.g., 0.997-1.004 for Ma~1.5). At α≈0.05 the filter kills the interface
# jumps with negligible impact on bulk wake physics. For obstacle neighbors
# the center value is used (zero contribution to the laplacian stencil).
#
# Two-pass launcher: the current T field is copied into a scratch buffer, then
# the kernel reads from scratch and writes the filtered result back into the
# canonical `level.temperature` array.
# ==============================================================================

using KernelAbstractions

@kernel function filter_temperature_kernel!(
    T_out, T_in, g_current,
    rho_arr, vel_arr, kinetic_energy_arr,
    obstacle, neighbor_table,
    w_arr,
    alpha::Float32, block_size::Int32,
    n_dirs_thermal::Int32,
    Cv::Float32,
    shift_x::Float32, shift_y::Float32, shift_z::Float32,
    repair_g_energy::Int32
)
    x, y, z, b_idx = @index(Global, NTuple)

    @inbounds begin
        if obstacle[x, y, z, b_idx]
            T_out[x, y, z, b_idx] = T_in[x, y, z, b_idx]
        else
            T_c = T_in[x, y, z, b_idx]
            # 6-neighbor fluid-aware sampling: obstacles return center → no contribution.
            T_xp = get_scalar_neighbor_fluid(T_in, obstacle, x, y, z, b_idx,  1,  0,  0, block_size, neighbor_table)
            T_xm = get_scalar_neighbor_fluid(T_in, obstacle, x, y, z, b_idx, -1,  0,  0, block_size, neighbor_table)
            T_yp = get_scalar_neighbor_fluid(T_in, obstacle, x, y, z, b_idx,  0,  1,  0, block_size, neighbor_table)
            T_ym = get_scalar_neighbor_fluid(T_in, obstacle, x, y, z, b_idx,  0, -1,  0, block_size, neighbor_table)
            T_zp = get_scalar_neighbor_fluid(T_in, obstacle, x, y, z, b_idx,  0,  0,  1, block_size, neighbor_table)
            T_zm = get_scalar_neighbor_fluid(T_in, obstacle, x, y, z, b_idx,  0,  0, -1, block_size, neighbor_table)

            T_avg = (T_xp + T_xm + T_yp + T_ym + T_zp + T_zm) * (1.0f0/6.0f0)
            T_filtered = (1.0f0 - alpha) * T_c + alpha * T_avg
            T_out[x, y, z, b_idx] = T_filtered

            if repair_g_energy == Int32(1)
                rho_local = max(rho_arr[x, y, z, b_idx], 0.01f0)
                ux_rel = vel_arr[x, y, z, b_idx, 1] - shift_x
                uy_rel = vel_arr[x, y, z, b_idx, 2] - shift_y
                uz_rel = vel_arr[x, y, z, b_idx, 3] - shift_z
                u_sq = ux_rel*ux_rel + uy_rel*uy_rel + uz_rel*uz_rel
                E_kin = kinetic_energy_arr[x, y, z, b_idx]

                if T_filtered == T_filtered && E_kin == E_kin && rho_local == rho_local
                    g_sum = 0.0f0
                    w_sum = 0.0f0
                    for k in Int32(1):n_dirs_thermal
                        g_sum += g_current[x, y, z, b_idx, k]
                        w_sum += w_arr[k]
                    end

                    if w_sum > 1.0f-8
                        target_g_sum = rho_local * Cv * T_filtered -
                                       0.5f0 * (E_kin - rho_local * u_sq)
                        correction = (target_g_sum - g_sum) / w_sum

                        for k in Int32(1):n_dirs_thermal
                            g_new = g_current[x, y, z, b_idx, k] + correction * w_arr[k]
                            g_current[x, y, z, b_idx, k] = max(g_new, Float32(1e-10))
                        end
                    end
                end
            end
        end
    end
end

"""
    apply_t_filter!(level, alpha, backend)

Apply a 6-neighbor laplacian filter to `level.temperature` in place.
Reads from `level.temperature_filter_scratch` (which is refreshed here via
copyto!) and writes the filtered values back into `level.temperature`.

No-op when THERMAL_FILTER_ENABLED is false, level has no thermal field,
the scratch buffer is a stub, or α ≤ 0.
"""
function apply_t_filter!(level, alpha::Float32, backend;
                         g_current=nothing, vel_current=nothing, w_gpu=nothing)
    n_blocks = length(level.active_block_coords)
    if n_blocks == 0 || alpha <= 0.0f0
        return
    end
    if !has_thermal(level) || !has_thermal_filter(level)
        return
    end

    # Snapshot current T into scratch, then Jacobi-filter back into level.temperature.
    copyto!(level.temperature_filter_scratch, level.temperature)

    kernel! = filter_temperature_kernel!(backend)
    repair_g_energy = (g_current !== nothing &&
                       vel_current !== nothing &&
                       w_gpu !== nothing &&
                       DDF_ENABLED &&
                       LATTICE_TYPE == :d3q39 &&
                       length(g_current) > 1) ? Int32(1) : Int32(0)
    kernel!(
        level.temperature, level.temperature_filter_scratch,
        repair_g_energy == Int32(1) ? g_current : level.g,
        level.rho,
        repair_g_energy == Int32(1) ? vel_current : level.vel,
        level.kinetic_energy,
        level.obstacle, level.neighbor_table,
        repair_g_energy == Int32(1) ? w_gpu : level.g,
        alpha, Int32(BLOCK_SIZE),
        N_DIRS_THERMAL,
        Float32(DDF_CV),
        Float32(LATTICE_SHIFT_X),
        Float32(LATTICE_SHIFT_Y),
        Float32(LATTICE_SHIFT_Z),
        repair_g_energy,
        ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    )
    KernelAbstractions.synchronize(backend)
end
