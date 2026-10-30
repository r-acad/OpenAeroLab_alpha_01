@inline function thermal_interface_dir_index(dx::Int, dy::Int, dz::Int)
    return (dx + 1) + (dy + 1) * 3 + (dz + 1) * 9 + 1
end

@inline _fine_to_parent_interp_coord_host(fine_g::Int) = Float32(fine_g) * 0.5f0 + 0.25f0

@inline function thermal_interface_targets(T_local::Float32, rho_local_raw::Float32,
                                           ux::Float32, uy::Float32, uz::Float32,
                                           shift_x::Float32, shift_y::Float32, shift_z::Float32)
    if LATTICE_TYPE == :d3q39
        coeff_g = Float32(DDF_CV - 1.5 * Float64(CS2_LATTICE))
        rho_local = max(rho_local_raw, 0.01f0)
        ux_rel = ux - shift_x
        uy_rel = uy - shift_y
        uz_rel = uz - shift_z
        K_target = coeff_g * T_local * rho_local
        qx_target = coeff_g * T_local * rho_local * ux_rel
        qy_target = coeff_g * T_local * rho_local * uy_rel
        qz_target = coeff_g * T_local * rho_local * uz_rel
        return K_target, qx_target, qy_target, qz_target
    end

    K_target = T_local
    qx_target = T_local * ux
    qy_target = T_local * uy
    qz_target = T_local * uz
    return K_target, qx_target, qy_target, qz_target
end

@inline function sample_parent_scalar_at_fine_coord_host(parent_field, parent_ptr,
                                                         parent_dim_x::Int, parent_dim_y::Int, parent_dim_z::Int,
                                                         fine_gx::Int, fine_gy::Int, fine_gz::Int,
                                                         block_size::Int,
                                                         default_value::Float32)
    px_cont = _fine_to_parent_interp_coord_host(fine_gx)
    py_cont = _fine_to_parent_interp_coord_host(fine_gy)
    pz_cont = _fine_to_parent_interp_coord_host(fine_gz)

    px0_raw = floor(Int, px_cont)
    py0_raw = floor(Int, py_cont)
    pz0_raw = floor(Int, pz_cont)
    px1 = px0_raw + 1
    py1 = py0_raw + 1
    pz1 = pz0_raw + 1

    wx = px_cont - Float32(px0_raw)
    wy = py_cont - Float32(py0_raw)
    wz = pz_cont - Float32(pz0_raw)

    px0 = max(1, px0_raw)
    py0 = max(1, py0_raw)
    pz0 = max(1, pz0_raw)

    @inline function get_parent_scalar(pgx::Int, pgy::Int, pgz::Int)
        pbx = (pgx - 1) ÷ block_size + 1
        pby = (pgy - 1) ÷ block_size + 1
        pbz = (pgz - 1) ÷ block_size + 1

        if pbx >= 1 && pbx <= parent_dim_x &&
           pby >= 1 && pby <= parent_dim_y &&
           pbz >= 1 && pbz <= parent_dim_z
            pb_idx = parent_ptr[pbx, pby, pbz]
            if pb_idx > 0
                plx = (pgx - 1) % block_size + 1
                ply = (pgy - 1) % block_size + 1
                plz = (pgz - 1) % block_size + 1
                return (Float32(parent_field[plx, ply, plz, pb_idx]), true)
            end
        end

        return (default_value, false)
    end

    d000 = get_parent_scalar(px0, py0, pz0)
    d100 = get_parent_scalar(px1, py0, pz0)
    d010 = get_parent_scalar(px0, py1, pz0)
    d110 = get_parent_scalar(px1, py1, pz0)
    d001 = get_parent_scalar(px0, py0, pz1)
    d101 = get_parent_scalar(px1, py0, pz1)
    d011 = get_parent_scalar(px0, py1, pz1)
    d111 = get_parent_scalar(px1, py1, pz1)

    v000 = d000
    v100 = d100[2] ? d100 : v000
    v010 = d010[2] ? d010 : v000
    v110 = d110[2] ? d110 : v000
    v001 = d001[2] ? d001 : v000
    v101 = d101[2] ? d101 : v000
    v011 = d011[2] ? d011 : v000
    v111 = d111[2] ? d111 : v000

    c00 = v000[1] * (1.0f0 - wx) + v100[1] * wx
    c01 = v001[1] * (1.0f0 - wx) + v101[1] * wx
    c10 = v010[1] * (1.0f0 - wx) + v110[1] * wx
    c11 = v011[1] * (1.0f0 - wx) + v111[1] * wx
    c0 = c00 * (1.0f0 - wy) + c10 * wy
    c1 = c01 * (1.0f0 - wy) + c11 * wy
    return c0 * (1.0f0 - wz) + c1 * wz
end

@inline function thermal_interface_uses_lagrange4_host(parent_dim_x::Int, parent_dim_y::Int, parent_dim_z::Int,
                                                       parent_ptr,
                                                       fine_gx::Int, fine_gy::Int, fine_gz::Int,
                                                       block_size::Int)
    px_cont = _fine_to_parent_interp_coord_host(fine_gx)
    py_cont = _fine_to_parent_interp_coord_host(fine_gy)
    pz_cont = _fine_to_parent_interp_coord_host(fine_gz)

    px0 = max(1, floor(Int, px_cont))
    py0 = max(1, floor(Int, py_cont))
    pz0 = max(1, floor(Int, pz_cont))

    parent_nx = parent_dim_x * block_size
    parent_ny = parent_dim_y * block_size
    parent_nz = parent_dim_z * block_size

    if !(px0 - 1 >= 1 && px0 + 2 <= parent_nx &&
         py0 - 1 >= 1 && py0 + 2 <= parent_ny &&
         pz0 - 1 >= 1 && pz0 + 2 <= parent_nz)
        return false
    end

    for pgz in (pz0 - 1):(pz0 + 2), pgy in (py0 - 1):(py0 + 2), pgx in (px0 - 1):(px0 + 2)
        pbx = (pgx - 1) ÷ block_size + 1
        pby = (pgy - 1) ÷ block_size + 1
        pbz = (pgz - 1) ÷ block_size + 1
        if pbx < 1 || pbx > parent_dim_x || pby < 1 || pby > parent_dim_y || pbz < 1 || pbz > parent_dim_z
            return false
        end
        if parent_ptr[pbx, pby, pbz] <= 0
            return false
        end
    end

    return true
end

@inline function thermal_interface_valid_trilinear_corner_count_host(parent_dim_x::Int, parent_dim_y::Int, parent_dim_z::Int,
                                                                     parent_ptr,
                                                                     fine_gx::Int, fine_gy::Int, fine_gz::Int,
                                                                     block_size::Int)
    px_cont = _fine_to_parent_interp_coord_host(fine_gx)
    py_cont = _fine_to_parent_interp_coord_host(fine_gy)
    pz_cont = _fine_to_parent_interp_coord_host(fine_gz)

    px0 = max(1, floor(Int, px_cont))
    py0 = max(1, floor(Int, py_cont))
    pz0 = max(1, floor(Int, pz_cont))
    px1 = px0 + 1
    py1 = py0 + 1
    pz1 = pz0 + 1

    count_valid = 0
    for (pgx, pgy, pgz) in ((px0, py0, pz0), (px1, py0, pz0), (px0, py1, pz0), (px1, py1, pz0),
                            (px0, py0, pz1), (px1, py0, pz1), (px0, py1, pz1), (px1, py1, pz1))
        pbx = (pgx - 1) ÷ block_size + 1
        pby = (pgy - 1) ÷ block_size + 1
        pbz = (pgz - 1) ÷ block_size + 1
        if pbx >= 1 && pbx <= parent_dim_x &&
           pby >= 1 && pby <= parent_dim_y &&
           pbz >= 1 && pbz <= parent_dim_z &&
           parent_ptr[pbx, pby, pbz] > 0
            count_valid += 1
        end
    end

    return count_valid
end

@inline function thermal_interface_tensor_support_host(parent_dim_x::Int, parent_dim_y::Int, parent_dim_z::Int,
                                                       parent_ptr,
                                                       fine_gx::Int, fine_gy::Int, fine_gz::Int,
                                                       block_size::Int,
                                                       use_x4::Bool, use_y4::Bool, use_z4::Bool)
    px_cont = _fine_to_parent_interp_coord_host(fine_gx)
    py_cont = _fine_to_parent_interp_coord_host(fine_gy)
    pz_cont = _fine_to_parent_interp_coord_host(fine_gz)

    px0 = max(1, floor(Int, px_cont))
    py0 = max(1, floor(Int, py_cont))
    pz0 = max(1, floor(Int, pz_cont))
    px1 = px0 + 1
    py1 = py0 + 1
    pz1 = pz0 + 1

    parent_nx = parent_dim_x * block_size
    parent_ny = parent_dim_y * block_size
    parent_nz = parent_dim_z * block_size

    if use_x4 && !(px0 - 1 >= 1 && px0 + 2 <= parent_nx)
        return false
    end
    if use_y4 && !(py0 - 1 >= 1 && py0 + 2 <= parent_ny)
        return false
    end
    if use_z4 && !(pz0 - 1 >= 1 && pz0 + 2 <= parent_nz)
        return false
    end

    ixmax = use_x4 ? 3 : 1
    iymax = use_y4 ? 3 : 1
    izmax = use_z4 ? 3 : 1
    for iz in 0:izmax
        pgz = use_z4 ? (pz0 - 1 + iz) : (iz == 0 ? pz0 : pz1)
        for iy in 0:iymax
            pgy = use_y4 ? (py0 - 1 + iy) : (iy == 0 ? py0 : py1)
            for ix in 0:ixmax
                pgx = use_x4 ? (px0 - 1 + ix) : (ix == 0 ? px0 : px1)
                pbx = (pgx - 1) ÷ block_size + 1
                pby = (pgy - 1) ÷ block_size + 1
                pbz = (pgz - 1) ÷ block_size + 1
                if pbx < 1 || pbx > parent_dim_x || pby < 1 || pby > parent_dim_y || pbz < 1 || pbz > parent_dim_z
                    return false
                end
                if parent_ptr[pbx, pby, pbz] <= 0
                    return false
                end
            end
        end
    end

    return true
end

function export_thermal_interface_diagnostics(step::Int, grids::AbstractVector{<:BlockLevel}, output_dir::String)
    if !THERMAL_INTERFACE_DIAGNOSTICS || !DDF_ENABLED || length(grids) < 2
        return nothing
    end

    summary_path = joinpath(output_dir, "thermal_interface_summary.csv")
    stencil_summary_path = joinpath(output_dir, "thermal_interface_stencil_summary.csv")
    fallback_support_path = joinpath(output_dir, "thermal_interface_fallback_support.csv")
    fallback_pattern_path = joinpath(output_dir, "thermal_interface_fallback_pattern.csv")
    if !isfile(summary_path)
        open(summary_path, "w") do io
            println(io, "step,level,n_interface_cells,mean_abs_delta_T,max_abs_delta_T,mean_delta_T,mean_abs_K,max_abs_K,mean_K,mean_abs_q,max_abs_q,mean_qx,mean_qy,mean_qz,mean_abs_K_energy,max_abs_K_energy,mean_abs_K_pop_minus_energy")
        end
    end
    if !isfile(stencil_summary_path)
        open(stencil_summary_path, "w") do io
            println(io, "step,level,n_interface_cells,n_lagrange4,n_trilinear,mean_abs_delta_T_lagrange4,mean_abs_delta_T_trilinear,mean_abs_K_lagrange4,mean_abs_K_trilinear,mean_abs_q_lagrange4,mean_abs_q_trilinear,max_abs_K_lagrange4,max_abs_K_trilinear,mean_abs_K_energy_lagrange4,mean_abs_K_energy_trilinear,mean_abs_K_pop_minus_energy_lagrange4,mean_abs_K_pop_minus_energy_trilinear")
        end
    end
    if !isfile(fallback_support_path)
        open(fallback_support_path, "w") do io
            println(io, "step,level,valid_corners,n_cells,mean_abs_delta_T,mean_abs_K,mean_abs_q,max_abs_K,mean_abs_K_energy,mean_abs_K_pop_minus_energy")
        end
    end
    if !isfile(fallback_pattern_path)
        open(fallback_pattern_path, "w") do io
            println(io, "step,level,pattern,n_cells,mean_abs_delta_T,mean_abs_K,mean_abs_q,max_abs_K,mean_abs_K_energy,mean_abs_K_pop_minus_energy")
        end
    end

    shift_x = Float32(LATTICE_SHIFT_X)
    shift_y = Float32(LATTICE_SHIFT_Y)
    shift_z = Float32(LATTICE_SHIFT_Z)
    n_dirs_thermal = Int(N_DIRS_THERMAL)
    block_size = Int32(BLOCK_SIZE)
    dir_xm = thermal_interface_dir_index(-1, 0, 0)
    dir_xp = thermal_interface_dir_index(1, 0, 0)
    dir_ym = thermal_interface_dir_index(0, -1, 0)
    dir_yp = thermal_interface_dir_index(0, 1, 0)
    dir_zm = thermal_interface_dir_index(0, 0, -1)
    dir_zp = thermal_interface_dir_index(0, 0, 1)

    open(summary_path, "a") do io
        open(stencil_summary_path, "a") do stencil_io
        open(fallback_support_path, "a") do support_io
        open(fallback_pattern_path, "a") do pattern_io
        for level_idx in 2:length(grids)
            level = grids[level_idx]
            parent = grids[level_idx - 1]
            if !has_thermal(level) || isempty(level.active_block_coords)
                continue
            end

            neighbor_table = Array(level.neighbor_table)
            parent_temperature = Array(parent.temperature)
            parent_ptr = Array(parent.block_pointer)

            parent_dim_x = Int(parent.grid_dim_x)
            parent_dim_y = Int(parent.grid_dim_y)
            parent_dim_z = Int(parent.grid_dim_z)

            n_interface_cells = 0
            sum_abs_delta_T = 0.0
            max_abs_delta_T = 0.0
            sum_delta_T = 0.0
            sum_abs_K = 0.0
            max_abs_K = 0.0
            sum_K = 0.0
            sum_abs_q = 0.0
            max_abs_q = 0.0
            sum_qx = 0.0
            sum_qy = 0.0
            sum_qz = 0.0
            sum_abs_K_energy = 0.0
            max_abs_K_energy = 0.0
            sum_abs_K_pop_minus_energy = 0.0
            n_lagrange4 = 0
            n_trilinear = 0
            sum_abs_delta_T_lagrange4 = 0.0
            sum_abs_delta_T_trilinear = 0.0
            sum_abs_K_lagrange4 = 0.0
            sum_abs_K_trilinear = 0.0
            sum_abs_q_lagrange4 = 0.0
            sum_abs_q_trilinear = 0.0
            max_abs_K_lagrange4 = 0.0
            max_abs_K_trilinear = 0.0
            sum_abs_K_energy_lagrange4 = 0.0
            sum_abs_K_energy_trilinear = 0.0
            sum_abs_K_pop_minus_energy_lagrange4 = 0.0
            sum_abs_K_pop_minus_energy_trilinear = 0.0
            fallback_counts = zeros(Int, 9)
            fallback_sum_abs_delta_T = zeros(Float64, 9)
            fallback_sum_abs_K = zeros(Float64, 9)
            fallback_sum_abs_q = zeros(Float64, 9)
            fallback_max_abs_K = zeros(Float64, 9)
            fallback_sum_abs_K_energy = zeros(Float64, 9)
            fallback_sum_abs_K_pop_minus_energy = zeros(Float64, 9)
            fallback_pattern_counts = zeros(Int, 7)
            fallback_pattern_sum_abs_delta_T = zeros(Float64, 7)
            fallback_pattern_sum_abs_K = zeros(Float64, 7)
            fallback_pattern_sum_abs_q = zeros(Float64, 7)
            fallback_pattern_max_abs_K = zeros(Float64, 7)
            fallback_pattern_sum_abs_K_energy = zeros(Float64, 7)
            fallback_pattern_sum_abs_K_pop_minus_energy = zeros(Float64, 7)

            interface_blocks = Int[]
            for b in eachindex(level.active_block_coords)
                bx, by, bz = level.active_block_coords[b]
                has_xm = neighbor_table[b, dir_xm] > 0
                has_xp = neighbor_table[b, dir_xp] > 0
                has_ym = neighbor_table[b, dir_ym] > 0
                has_yp = neighbor_table[b, dir_yp] > 0
                has_zm = neighbor_table[b, dir_zm] > 0
                has_zp = neighbor_table[b, dir_zp] > 0

                if has_xm && has_xp && has_ym && has_yp && has_zm && has_zp
                    continue
                end

                push!(interface_blocks, b)
            end

            for b in interface_blocks
                bx, by, bz = level.active_block_coords[b]
                has_xm = neighbor_table[b, dir_xm] > 0
                has_xp = neighbor_table[b, dir_xp] > 0
                has_ym = neighbor_table[b, dir_ym] > 0
                has_yp = neighbor_table[b, dir_yp] > 0
                has_zm = neighbor_table[b, dir_zm] > 0
                has_zp = neighbor_table[b, dir_zp] > 0

                temperature_block = Array(view(level.temperature, :, :, :, b))
                rho_block = Array(view(level.rho, :, :, :, b))
                vel_block = Array(view(level.vel, :, :, :, b, :))
                g_block = Array(view(level.g, :, :, :, b, :))
                kinetic_energy_block = Array(view(level.kinetic_energy, :, :, :, b))
                obstacle_block = Array(view(level.obstacle, :, :, :, b))

                for z in 1:BLOCK_SIZE, y in 1:BLOCK_SIZE, x in 1:BLOCK_SIZE
                    if obstacle_block[x, y, z]
                        continue
                    end

                    interface_cell =
                        (x == 1 && !has_xm) || (x == BLOCK_SIZE && !has_xp) ||
                        (y == 1 && !has_ym) || (y == BLOCK_SIZE && !has_yp) ||
                        (z == 1 && !has_zm) || (z == BLOCK_SIZE && !has_zp)
                    if !interface_cell
                        continue
                    end

                    gx = (bx - 1) * Int(BLOCK_SIZE) + x
                    gy = (by - 1) * Int(BLOCK_SIZE) + y
                    gz = (bz - 1) * Int(BLOCK_SIZE) + z

                    T_local = Float32(temperature_block[x, y, z])
                    T_parent = sample_parent_scalar_at_fine_coord_host(
                        parent_temperature, parent_ptr,
                        parent_dim_x, parent_dim_y, parent_dim_z,
                        gx, gy, gz,
                        Int(BLOCK_SIZE),
                        T_local
                    )
                    delta_T = Float64(T_local - T_parent)

                    rho_local = Float32(rho_block[x, y, z])
                    ux = Float32(vel_block[x, y, z, 1])
                    uy = Float32(vel_block[x, y, z, 2])
                    uz = Float32(vel_block[x, y, z, 3])
                    Ekin_local = Float32(kinetic_energy_block[x, y, z])

                    K_target, qx_target, qy_target, qz_target = thermal_interface_targets(
                        T_local, rho_local, ux, uy, uz, shift_x, shift_y, shift_z
                    )

                    g_sum = 0.0f0
                    qx = 0.0f0
                    qy = 0.0f0
                    qz = 0.0f0
                    for k in 1:n_dirs_thermal
                        gk = Float32(g_block[x, y, z, k])
                        cx = Float32(C_X[k]) - shift_x
                        cy = Float32(C_Y[k]) - shift_y
                        cz = Float32(C_Z[k]) - shift_z
                        g_sum += gk
                        qx += cx * gk
                        qy += cy * gk
                        qz += cz * gk
                    end

                    K_res = Float64(g_sum - K_target)
                    u_sq = Float64(ux * ux + uy * uy + uz * uz)
                    K_energy = Float64(rho_local) * Float64(DDF_CV) * Float64(T_local) -
                               0.5 * (Float64(Ekin_local) - Float64(rho_local) * u_sq)
                    K_energy_res = K_energy - Float64(K_target)
                    K_pop_minus_energy = Float64(g_sum) - K_energy
                    qx_res = Float64(qx - qx_target)
                    qy_res = Float64(qy - qy_target)
                    qz_res = Float64(qz - qz_target)
                    q_abs = sqrt(qx_res * qx_res + qy_res * qy_res + qz_res * qz_res)
                    uses_lagrange4 = thermal_interface_uses_lagrange4_host(
                        parent_dim_x, parent_dim_y, parent_dim_z, parent_ptr,
                        gx, gy, gz, Int(BLOCK_SIZE)
                    )

                    n_interface_cells += 1
                    sum_abs_delta_T += abs(delta_T)
                    max_abs_delta_T = max(max_abs_delta_T, abs(delta_T))
                    sum_delta_T += delta_T
                    sum_abs_K += abs(K_res)
                    max_abs_K = max(max_abs_K, abs(K_res))
                    sum_K += K_res
                    sum_abs_q += q_abs
                    max_abs_q = max(max_abs_q, q_abs)
                    sum_qx += qx_res
                    sum_qy += qy_res
                    sum_qz += qz_res
                    sum_abs_K_energy += abs(K_energy_res)
                    max_abs_K_energy = max(max_abs_K_energy, abs(K_energy_res))
                    sum_abs_K_pop_minus_energy += abs(K_pop_minus_energy)
                    if uses_lagrange4
                        n_lagrange4 += 1
                        sum_abs_delta_T_lagrange4 += abs(delta_T)
                        sum_abs_K_lagrange4 += abs(K_res)
                        sum_abs_q_lagrange4 += q_abs
                        max_abs_K_lagrange4 = max(max_abs_K_lagrange4, abs(K_res))
                        sum_abs_K_energy_lagrange4 += abs(K_energy_res)
                        sum_abs_K_pop_minus_energy_lagrange4 += abs(K_pop_minus_energy)
                    else
                        n_trilinear += 1
                        sum_abs_delta_T_trilinear += abs(delta_T)
                        sum_abs_K_trilinear += abs(K_res)
                        sum_abs_q_trilinear += q_abs
                        max_abs_K_trilinear = max(max_abs_K_trilinear, abs(K_res))
                        sum_abs_K_energy_trilinear += abs(K_energy_res)
                        sum_abs_K_pop_minus_energy_trilinear += abs(K_pop_minus_energy)
                        valid_corners = thermal_interface_valid_trilinear_corner_count_host(
                            parent_dim_x, parent_dim_y, parent_dim_z, parent_ptr,
                            gx, gy, gz, Int(BLOCK_SIZE)
                        )
                        bucket = clamp(valid_corners, 0, 8) + 1
                        fallback_counts[bucket] += 1
                        fallback_sum_abs_delta_T[bucket] += abs(delta_T)
                        fallback_sum_abs_K[bucket] += abs(K_res)
                        fallback_sum_abs_q[bucket] += q_abs
                        fallback_max_abs_K[bucket] = max(fallback_max_abs_K[bucket], abs(K_res))
                        fallback_sum_abs_K_energy[bucket] += abs(K_energy_res)
                        fallback_sum_abs_K_pop_minus_energy[bucket] += abs(K_pop_minus_energy)

                        pattern_idx =
                            thermal_interface_tensor_support_host(parent_dim_x, parent_dim_y, parent_dim_z, parent_ptr, gx, gy, gz, Int(BLOCK_SIZE), true, true, false) ? 1 :
                            thermal_interface_tensor_support_host(parent_dim_x, parent_dim_y, parent_dim_z, parent_ptr, gx, gy, gz, Int(BLOCK_SIZE), true, false, true) ? 2 :
                            thermal_interface_tensor_support_host(parent_dim_x, parent_dim_y, parent_dim_z, parent_ptr, gx, gy, gz, Int(BLOCK_SIZE), false, true, true) ? 3 :
                            thermal_interface_tensor_support_host(parent_dim_x, parent_dim_y, parent_dim_z, parent_ptr, gx, gy, gz, Int(BLOCK_SIZE), true, false, false) ? 4 :
                            thermal_interface_tensor_support_host(parent_dim_x, parent_dim_y, parent_dim_z, parent_ptr, gx, gy, gz, Int(BLOCK_SIZE), false, true, false) ? 5 :
                            thermal_interface_tensor_support_host(parent_dim_x, parent_dim_y, parent_dim_z, parent_ptr, gx, gy, gz, Int(BLOCK_SIZE), false, false, true) ? 6 : 7
                        fallback_pattern_counts[pattern_idx] += 1
                        fallback_pattern_sum_abs_delta_T[pattern_idx] += abs(delta_T)
                        fallback_pattern_sum_abs_K[pattern_idx] += abs(K_res)
                        fallback_pattern_sum_abs_q[pattern_idx] += q_abs
                        fallback_pattern_max_abs_K[pattern_idx] = max(fallback_pattern_max_abs_K[pattern_idx], abs(K_res))
                        fallback_pattern_sum_abs_K_energy[pattern_idx] += abs(K_energy_res)
                        fallback_pattern_sum_abs_K_pop_minus_energy[pattern_idx] += abs(K_pop_minus_energy)
                    end
                end
            end

            if n_interface_cells == 0
                continue
            end

            inv_n = 1.0 / n_interface_cells
            @printf(io, "%d,%d,%d,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                    step, level_idx, n_interface_cells,
                    sum_abs_delta_T * inv_n, max_abs_delta_T, sum_delta_T * inv_n,
                    sum_abs_K * inv_n, max_abs_K, sum_K * inv_n,
                    sum_abs_q * inv_n, max_abs_q,
                    sum_qx * inv_n, sum_qy * inv_n, sum_qz * inv_n,
                    sum_abs_K_energy * inv_n, max_abs_K_energy,
                    sum_abs_K_pop_minus_energy * inv_n)
            inv_l4 = n_lagrange4 > 0 ? 1.0 / n_lagrange4 : 0.0
            inv_tri = n_trilinear > 0 ? 1.0 / n_trilinear : 0.0
            @printf(stencil_io, "%d,%d,%d,%d,%d,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                    step, level_idx, n_interface_cells, n_lagrange4, n_trilinear,
                    sum_abs_delta_T_lagrange4 * inv_l4,
                    sum_abs_delta_T_trilinear * inv_tri,
                    sum_abs_K_lagrange4 * inv_l4,
                    sum_abs_K_trilinear * inv_tri,
                    sum_abs_q_lagrange4 * inv_l4,
                    sum_abs_q_trilinear * inv_tri,
                    max_abs_K_lagrange4,
                    max_abs_K_trilinear,
                    sum_abs_K_energy_lagrange4 * inv_l4,
                    sum_abs_K_energy_trilinear * inv_tri,
                    sum_abs_K_pop_minus_energy_lagrange4 * inv_l4,
                    sum_abs_K_pop_minus_energy_trilinear * inv_tri)
            for valid_corners in 0:8
                bucket = valid_corners + 1
                if fallback_counts[bucket] == 0
                    continue
                end
                inv_bucket = 1.0 / fallback_counts[bucket]
                @printf(support_io, "%d,%d,%d,%d,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                        step, level_idx, valid_corners, fallback_counts[bucket],
                        fallback_sum_abs_delta_T[bucket] * inv_bucket,
                        fallback_sum_abs_K[bucket] * inv_bucket,
                        fallback_sum_abs_q[bucket] * inv_bucket,
                        fallback_max_abs_K[bucket],
                        fallback_sum_abs_K_energy[bucket] * inv_bucket,
                        fallback_sum_abs_K_pop_minus_energy[bucket] * inv_bucket)
            end
            pattern_labels = ("442", "424", "244", "422", "242", "224", "222")
            for pattern_idx in 1:length(pattern_labels)
                if fallback_pattern_counts[pattern_idx] == 0
                    continue
                end
                inv_pattern = 1.0 / fallback_pattern_counts[pattern_idx]
                @printf(pattern_io, "%d,%d,%s,%d,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                        step, level_idx, pattern_labels[pattern_idx], fallback_pattern_counts[pattern_idx],
                        fallback_pattern_sum_abs_delta_T[pattern_idx] * inv_pattern,
                        fallback_pattern_sum_abs_K[pattern_idx] * inv_pattern,
                        fallback_pattern_sum_abs_q[pattern_idx] * inv_pattern,
                        fallback_pattern_max_abs_K[pattern_idx],
                        fallback_pattern_sum_abs_K_energy[pattern_idx] * inv_pattern,
                        fallback_pattern_sum_abs_K_pop_minus_energy[pattern_idx] * inv_pattern)
            end
        end
        end
        end
        end
    end

    return nothing
end
