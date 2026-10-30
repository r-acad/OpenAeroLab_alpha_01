using KernelAbstractions
using CUDA
using Printf

@inline _fine_to_parent_interp_coord_diag(fine_g::Int) = Float32(fine_g) * 0.5f0 + 0.25f0

@inline function _parent_interp_bracket_diag(p_cont::Float32, parent_n::Int)
    if parent_n <= 1
        return 1, 1, 0.0f0
    end
    i0 = clamp(floor(Int, p_cont), 1, parent_n - 1)
    i1 = i0 + 1
    return i0, i1, p_cont - Float32(i0)
end

@kernel function vorticity_kernel!(w_mag, vel, nx, ny, nz)
    i, j, k, b = @index(Global, NTuple)
    
    if i > 1 && i < nx && j > 1 && j < ny && k > 1 && k < nz
        @inbounds begin
            duz_dy = 0.5f0 * (vel[i, j+1, k, b, 3] - vel[i, j-1, k, b, 3])
            duy_dz = 0.5f0 * (vel[i, j, k+1, b, 2] - vel[i, j, k-1, b, 2])
            wx = duz_dy - duy_dz

            dux_dz = 0.5f0 * (vel[i, j, k+1, b, 1] - vel[i, j, k-1, b, 1])
            duz_dx = 0.5f0 * (vel[i+1, j, k, b, 3] - vel[i-1, j, k, b, 3])
            wy = dux_dz - duz_dx

            duy_dx = 0.5f0 * (vel[i+1, j, k, b, 2] - vel[i-1, j, k, b, 2])
            dux_dy = 0.5f0 * (vel[i, j+1, k, b, 1] - vel[i, j-1, k, b, 1])
            wz = duy_dx - dux_dy

            w_mag[i, j, k, b] = sqrt(wx*wx + wy*wy + wz*wz)
        end
    else
        @inbounds w_mag[i, j, k, b] = 0.0f0
    end
end

function compute_vorticity(level::BlockLevel)
    backend = get_backend(level.vel)
    n_blocks = length(level.active_block_coords)
    w_mag = KernelAbstractions.zeros(backend, Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    
    if n_blocks > 0
        kernel! = vorticity_kernel!(backend)
        kernel!(w_mag, level.vel, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, 
                ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks))
    end
    
    return w_mag
end

function compute_flow_stats(level::BlockLevel)
    backend = get_backend(level.rho)
    
    if backend isa CUDABackend
        rho_flat = reshape(level.rho, :)
        vel_flat = reshape(level.vel, :, 3)
        obs_flat = reshape(level.obstacle, :)
        
        valid_mask = .!obs_flat
        n_fluid = sum(valid_mask)

        if n_fluid > 0
            rho_fluid = rho_flat .* valid_mask
            rho_mean = sum(rho_fluid) / n_fluid
            rho_min = minimum(rho_flat[obs_flat .== false])
            rho_max = maximum(rho_flat[obs_flat .== false])

            v2 = sum(abs2, vel_flat, dims=2)
            v_mag = sqrt.(vec(v2))
            v_max = maximum(v_mag .* valid_mask)
            u_mean = sum(v_mag .* valid_mask) / n_fluid

            # ACTIVE-only extrema (exclude AMR-covered cells, which are overlapped
            # by a finer level and take the unclamped is_covered macro path). If
            # rho_max_active ≪ rho_max, the spike lives on COVERED cells (the
            # realizability-clamp bypass) rather than active fluid. Diagnostic only.
            rho_max_active = rho_max
            v_max_active = v_max
            try
                cov_flat = reshape(level.amr_covered, :)
                act_mask = valid_mask .& .!cov_flat
                if sum(act_mask) > 0
                    rho_max_active = maximum(rho_flat[act_mask])
                    v_max_active = maximum((v_mag .* act_mask)[act_mask])
                end
            catch
            end
            
            ke = 0.5f0 * sum(rho_flat .* vec(v2) .* valid_mask)

            # Lattice 2nd-moment energy E_kin = Σf|c|² (the value the Frapolli
            # T-recovery actually consumes; stored in level.kinetic_energy by the
            # flow kernel). For a freestream at T=1, u=0.735: E_kin ≈ 2.54; at the
            # drifted T=2.24 it is ≈6.26. Diagnostic only.
            ekin_lattice_mean = 0.0f0
            gsum_mean = 0.0f0
            try
                if has_thermal(level)
                    ek_flat = reshape(level.kinetic_energy, :)
                    ekin_lattice_mean = sum(ek_flat .* valid_mask) / n_fluid
                    # Direct mean Σ_k g[k] (the thermal energy the recovery consumes as 2·Σg).
                    # For a freestream at T=1: Σg = coeff_g·T·ρ ≈ 1.0. If ≫1.0 -> real g excess.
                    ng = size(level.g, 5)
                    gacc = reshape(sum(@view(level.g[:,:,:,:,1:ng]), dims=5), :)
                    gsum_mean = sum(gacc .* valid_mask) / n_fluid
                end
            catch
            end

            # Thermal DDF statistics (Phase 2)
            T_min_val = 1.0f0
            T_max_val = 1.0f0
            T_mean_val = 1.0f0
            if DDF_ENABLED && has_thermal(level)
                temp_flat = reshape(level.temperature, :)
                T_min_val = minimum(temp_flat[obs_flat .== false])
                T_max_val = maximum(temp_flat[obs_flat .== false])
                T_mean_val = sum(temp_flat .* valid_mask) / n_fluid
            end

            return (
                n_fluid = n_fluid,
                rho_mean = rho_mean,
                rho_min = rho_min,
                rho_max = rho_max,
                rho_max_active = rho_max_active,
                v_max = v_max,
                v_max_active = v_max_active,
                u_mean = u_mean,
                kinetic_energy = ke,
                ekin_lattice_mean = ekin_lattice_mean,
                gsum_mean = gsum_mean,
                T_min = T_min_val,
                T_max = T_max_val,
                T_mean = T_mean_val
            )
        end
    end

    return (n_fluid=0, rho_mean=1.0, rho_min=1.0, rho_max=1.0, rho_max_active=1.0, v_max=0.0, v_max_active=0.0,
            u_mean=0.0, kinetic_energy=0.0, ekin_lattice_mean=0.0, gsum_mean=0.0, T_min=1.0, T_max=1.0, T_mean=1.0)
end

function diagnostic_effective_temperature(
    rho::Real,
    ux::Real,
    uy::Real,
    uz::Real,
    stored_T::Real,
    sponge::Real,
    u_ref::Real,
)
    if isfinite(Float64(stored_T))
        return Float64(stored_T)
    end
    if EOS_THERMODYNAMIC && LATTICE_TYPE == :d3q39
        u_eff = Float32(abs(Float64(u_ref)))
        if !(u_eff > 0.0f0)
            u_eff = Float32(abs(Float64(U_TARGET)))
        end
        return Float64(transonic_algebraic_temperature_d3q39(
            Float32(rho), Float32(ux), Float32(uy), Float32(uz),
            u_eff,
            Float32(CS2_PHYSICS),
            Float32(DDF_T_INLET),
            Float32((DDF_GAMMA - 1.0) * DDF_CV),
            Int32(0),
            Float32(sponge),
        ))
    end
    return 1.0
end

@inline diagnostic_pressure_coeff() =
    EOS_THERMODYNAMIC ? Float64((DDF_GAMMA - 1.0) * DDF_CV) :
                        Float64(CS2_PHYSICS)

# ────────────────────────────────────────────────────────────────────
# AMR interface diagnostics
# ────────────────────────────────────────────────────────────────────
# Compare density / velocity statistics of coarse cells that border a
# refined region against cells in the coarse interior of the same level.
# A well-coupled AMR interface should produce similar distributions on
# both subsets. Persistent mean/stddev gaps point at bad coarse→fine
# prolongation (over-dissipative or over-injecting non-equilibrium) or
# a restriction that's leaking aliased fine-grid content upscale.
#
# Cheap CPU implementation: copies ρ, vel, obstacle, block_pointer once
# per diagnostic tick — only runs when AMR_INTERFACE_DIAGNOSTICS is on.
function compute_amr_interface_stats(parent_level::BlockLevel, child_level::BlockLevel)
    rho_cpu = Array(parent_level.rho)
    vel_cpu = Array(parent_level.vel)
    obs_cpu = Array(parent_level.obstacle)
    parent_map_x_cpu = Array(parent_level.map_x)
    parent_map_y_cpu = Array(parent_level.map_y)
    parent_map_z_cpu = Array(parent_level.map_z)
    child_ptr_cpu = Array(child_level.block_pointer)

    cdx = size(child_ptr_cpu, 1)
    cdy = size(child_ptr_cpu, 2)
    cdz = size(child_ptr_cpu, 3)
    bs = BLOCK_SIZE

    @inline function has_children_at(gx::Int, gy::Int, gz::Int)
        # A coarse cell at (gx,gy,gz) covers the 2×2×2 child cells at
        # (2gx−1 .. 2gx, …). All 8 must belong to active child blocks for
        # the parent cell to be considered "refined" / covered by children.
        for sx in 0:1, sy in 0:1, sz in 0:1
            fgx = 2*gx - 1 + sx
            fgy = 2*gy - 1 + sy
            fgz = 2*gz - 1 + sz
            if fgx < 1 || fgy < 1 || fgz < 1; return false; end
            cbx = (fgx - 1) ÷ bs + 1
            cby = (fgy - 1) ÷ bs + 1
            cbz = (fgz - 1) ÷ bs + 1
            if cbx > cdx || cby > cdy || cbz > cdz; return false; end
            if child_ptr_cpu[cbx, cby, cbz] == 0; return false; end
        end
        return true
    end

    n_blocks = size(rho_cpu, 4)

    n_if = 0
    rho_sum_if = 0.0
    rho_sq_if  = 0.0
    rho_min_if = Inf
    rho_max_if = -Inf
    u_sum_if = 0.0
    u_max_if = 0.0

    n_in = 0
    rho_sum_in = 0.0
    rho_sq_in  = 0.0
    rho_min_in = Inf
    rho_max_in = -Inf
    u_sum_in = 0.0
    u_max_in = 0.0

    @inbounds for pb in 1:n_blocks
        pbx = Int(parent_map_x_cpu[pb])
        pby = Int(parent_map_y_cpu[pb])
        pbz = Int(parent_map_z_cpu[pb])
        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            obs_cpu[lx, ly, lz, pb] && continue
            gx = (pbx - 1) * bs + lx
            gy = (pby - 1) * bs + ly
            gz = (pbz - 1) * bs + lz

            # Skip cells that are themselves covered by children — those
            # aren't part of the "coarse active" set, they're overwritten
            # by restriction.
            has_children_at(gx, gy, gz) && continue

            rho = Float64(rho_cpu[lx, ly, lz, pb])
            ux = Float64(vel_cpu[lx, ly, lz, pb, 1])
            uy = Float64(vel_cpu[lx, ly, lz, pb, 2])
            uz = Float64(vel_cpu[lx, ly, lz, pb, 3])
            umag = sqrt(ux*ux + uy*uy + uz*uz)

            # Interface: at least one of the 6 face-neighbors is refined.
            is_interface = has_children_at(gx+1, gy, gz) ||
                           has_children_at(gx-1, gy, gz) ||
                           has_children_at(gx, gy+1, gz) ||
                           has_children_at(gx, gy-1, gz) ||
                           has_children_at(gx, gy, gz+1) ||
                           has_children_at(gx, gy, gz-1)

            if is_interface
                n_if += 1
                rho_sum_if += rho
                rho_sq_if  += rho * rho
                rho_min_if = min(rho_min_if, rho)
                rho_max_if = max(rho_max_if, rho)
                u_sum_if += umag
                u_max_if = max(u_max_if, umag)
            else
                n_in += 1
                rho_sum_in += rho
                rho_sq_in  += rho * rho
                rho_min_in = min(rho_min_in, rho)
                rho_max_in = max(rho_max_in, rho)
                u_sum_in += umag
                u_max_in = max(u_max_in, umag)
            end
        end
    end

    rho_mean_if = n_if > 0 ? rho_sum_if / n_if : 0.0
    rho_mean_in = n_in > 0 ? rho_sum_in / n_in : 0.0
    rho_std_if = n_if > 0 ? sqrt(max(0.0, rho_sq_if / n_if - rho_mean_if^2)) : 0.0
    rho_std_in = n_in > 0 ? sqrt(max(0.0, rho_sq_in / n_in - rho_mean_in^2)) : 0.0
    u_mean_if = n_if > 0 ? u_sum_if / n_if : 0.0
    u_mean_in = n_in > 0 ? u_sum_in / n_in : 0.0

    return (
        n_interface = n_if,
        n_interior = n_in,
        rho_mean_if = rho_mean_if,
        rho_mean_in = rho_mean_in,
        rho_std_if = rho_std_if,
        rho_std_in = rho_std_in,
        rho_min_if = n_if > 0 ? rho_min_if : 1.0,
        rho_max_if = n_if > 0 ? rho_max_if : 1.0,
        rho_min_in = n_in > 0 ? rho_min_in : 1.0,
        rho_max_in = n_in > 0 ? rho_max_in : 1.0,
        u_mean_if = u_mean_if,
        u_mean_in = u_mean_in,
        u_max_if = u_max_if,
        u_max_in = u_max_in,
    )
end

function compute_amr_shell_face_jump_stats(parent_level::BlockLevel, child_level::BlockLevel)
    rho_cpu = Array(parent_level.rho)
    vel_cpu = Array(parent_level.vel)
    obs_cpu = Array(parent_level.obstacle)
    parent_map_x_cpu = Array(parent_level.map_x)
    parent_map_y_cpu = Array(parent_level.map_y)
    parent_map_z_cpu = Array(parent_level.map_z)
    child_ptr_cpu = Array(child_level.block_pointer)
    child_rho_cpu = Array(child_level.rho)
    child_vel_cpu = Array(child_level.vel)

    cdx = size(child_ptr_cpu, 1)
    cdy = size(child_ptr_cpu, 2)
    cdz = size(child_ptr_cpu, 3)
    bs = BLOCK_SIZE

    @inline function child_cell_state(fgx::Int, fgy::Int, fgz::Int)
        if fgx < 1 || fgy < 1 || fgz < 1
            return (0.0, 0.0, 0.0, 0.0, false)
        end
        cbx = div(fgx - 1, bs) + 1
        cby = div(fgy - 1, bs) + 1
        cbz = div(fgz - 1, bs) + 1
        if cbx > cdx || cby > cdy || cbz > cdz
            return (0.0, 0.0, 0.0, 0.0, false)
        end
        cb_idx = child_ptr_cpu[cbx, cby, cbz]
        if cb_idx == 0
            return (0.0, 0.0, 0.0, 0.0, false)
        end
        clx = (fgx - 1) % bs + 1
        cly = (fgy - 1) % bs + 1
        clz = (fgz - 1) % bs + 1
        return (
            Float64(child_rho_cpu[clx, cly, clz, cb_idx]),
            Float64(child_vel_cpu[clx, cly, clz, cb_idx, 1]),
            Float64(child_vel_cpu[clx, cly, clz, cb_idx, 2]),
            Float64(child_vel_cpu[clx, cly, clz, cb_idx, 3]),
            true
        )
    end

    @inline function has_children_at(gx::Int, gy::Int, gz::Int)
        for sx in 0:1, sy in 0:1, sz in 0:1
            fgx = 2 * gx - 1 + sx
            fgy = 2 * gy - 1 + sy
            fgz = 2 * gz - 1 + sz
            _, _, _, _, valid = child_cell_state(fgx, fgy, fgz)
            valid || return false
        end
        return true
    end

    @inline function shell_face_target_current(face::Int, gx::Int, gy::Int, gz::Int)
        mother_fx0 = 0
        mother_fy0 = 0
        mother_fz0 = 0
        refined_neighbor = false

        if face == 1
            refined_neighbor = has_children_at(gx + 1, gy, gz)
            mother_fx0 = 2 * gx + 1
            mother_fy0 = 2 * gy - 1
            mother_fz0 = 2 * gz - 1
        elseif face == 2
            refined_neighbor = has_children_at(gx - 1, gy, gz)
            mother_fx0 = 2 * gx - 2
            mother_fy0 = 2 * gy - 1
            mother_fz0 = 2 * gz - 1
        elseif face == 3
            refined_neighbor = has_children_at(gx, gy + 1, gz)
            mother_fx0 = 2 * gx - 1
            mother_fy0 = 2 * gy + 1
            mother_fz0 = 2 * gz - 1
        elseif face == 4
            refined_neighbor = has_children_at(gx, gy - 1, gz)
            mother_fx0 = 2 * gx - 1
            mother_fy0 = 2 * gy - 2
            mother_fz0 = 2 * gz - 1
        elseif face == 5
            refined_neighbor = has_children_at(gx, gy, gz + 1)
            mother_fx0 = 2 * gx - 1
            mother_fy0 = 2 * gy - 1
            mother_fz0 = 2 * gz + 1
        else
            refined_neighbor = has_children_at(gx, gy, gz - 1)
            mother_fx0 = 2 * gx - 1
            mother_fy0 = 2 * gy - 1
            mother_fz0 = 2 * gz - 2
        end

        refined_neighbor || return (0.0, 0.0, 0.0, 0.0, false)

        sx_first = 0
        sx_last = 1
        sy_first = 0
        sy_last = 1
        sz_first = 0
        sz_last = 1
        if face <= 2
            sx_last = 0
        elseif face <= 4
            sy_last = 0
        else
            sz_last = 0
        end

        rho_sum = 0.0
        ux_sum = 0.0
        uy_sum = 0.0
        uz_sum = 0.0
        n_face = 0
        for sx in sx_first:sx_last, sy in sy_first:sy_last, sz in sz_first:sz_last
            rho_v, ux_v, uy_v, uz_v, valid = child_cell_state(
                mother_fx0 + sx, mother_fy0 + sy, mother_fz0 + sz
            )
            valid || return (0.0, 0.0, 0.0, 0.0, false)
            rho_sum += rho_v
            ux_sum += ux_v
            uy_sum += uy_v
            uz_sum += uz_v
            n_face += 1
        end

        inv_face = 1.0 / max(n_face, 1)
        return (inv_face * rho_sum, inv_face * ux_sum, inv_face * uy_sum, inv_face * uz_sum, true)
    end

    n_faces = 0
    abs_drho_sum = 0.0
    abs_du_sum = 0.0
    max_abs_drho = 0.0
    max_abs_du = 0.0
    # Pass190: L² (RMS) accumulators — paper-comparable jump norms over
    # coarse↔fine shell faces. Mean-abs (L¹) is dominated by the many small
    # jumps; RMS is dominated by the few large ones, which is what Astoul
    # 2020 and Lagrava 2012 report when claiming "second-order accuracy at
    # interface". Observation-only, no physics behavior change.
    sq_drho_sum = 0.0
    sq_du_sum = 0.0

    n_faces_x = 0
    n_faces_y = 0
    n_faces_z = 0
    abs_drho_sum_x = 0.0
    abs_drho_sum_y = 0.0
    abs_drho_sum_z = 0.0
    abs_du_sum_x = 0.0
    abs_du_sum_y = 0.0
    abs_du_sum_z = 0.0
    n_faces_x_pos = 0
    n_faces_x_neg = 0
    n_faces_y_pos = 0
    n_faces_y_neg = 0
    n_faces_z_pos = 0
    n_faces_z_neg = 0
    abs_drho_sum_x_pos = 0.0
    abs_drho_sum_x_neg = 0.0
    abs_drho_sum_y_pos = 0.0
    abs_drho_sum_y_neg = 0.0
    abs_drho_sum_z_pos = 0.0
    abs_drho_sum_z_neg = 0.0
    abs_du_sum_x_pos = 0.0
    abs_du_sum_x_neg = 0.0
    abs_du_sum_y_pos = 0.0
    abs_du_sum_y_neg = 0.0
    abs_du_sum_z_pos = 0.0
    abs_du_sum_z_neg = 0.0
    drho_sum_x_pos = 0.0
    drho_sum_x_neg = 0.0
    drho_sum_y_pos = 0.0
    drho_sum_y_neg = 0.0
    drho_sum_z_pos = 0.0
    drho_sum_z_neg = 0.0
    dun_sum_x_pos = 0.0
    dun_sum_x_neg = 0.0
    dun_sum_y_pos = 0.0
    dun_sum_y_neg = 0.0
    dun_sum_z_pos = 0.0
    dun_sum_z_neg = 0.0
    abs_dmass_flux_sum = 0.0
    sq_dmass_flux_sum = 0.0
    dmass_flux_sum = 0.0
    abs_dmom_flux_sum = 0.0
    sq_dmom_flux_sum = 0.0
    dmom_flux_sum = 0.0
    abs_dmass_flux_sum_x = 0.0
    abs_dmass_flux_sum_y = 0.0
    abs_dmass_flux_sum_z = 0.0
    abs_dmom_flux_sum_x = 0.0
    abs_dmom_flux_sum_y = 0.0
    abs_dmom_flux_sum_z = 0.0
    dmass_flux_sum_x_pos = 0.0
    dmass_flux_sum_x_neg = 0.0
    dmass_flux_sum_y_pos = 0.0
    dmass_flux_sum_y_neg = 0.0
    dmass_flux_sum_z_pos = 0.0
    dmass_flux_sum_z_neg = 0.0
    dmom_flux_sum_x_pos = 0.0
    dmom_flux_sum_x_neg = 0.0
    dmom_flux_sum_y_pos = 0.0
    dmom_flux_sum_y_neg = 0.0
    dmom_flux_sum_z_pos = 0.0
    dmom_flux_sum_z_neg = 0.0
    n_faces_single = 0
    n_faces_multi = 0
    abs_drho_sum_single = 0.0
    abs_drho_sum_multi = 0.0
    abs_du_sum_single = 0.0
    abs_du_sum_multi = 0.0
    abs_dmass_flux_sum_single = 0.0
    abs_dmass_flux_sum_multi = 0.0
    abs_dmom_flux_sum_single = 0.0
    abs_dmom_flux_sum_multi = 0.0

    n_blocks = size(rho_cpu, 4)
    @inbounds for pb in 1:n_blocks
        pbx = Int(parent_map_x_cpu[pb])
        pby = Int(parent_map_y_cpu[pb])
        pbz = Int(parent_map_z_cpu[pb])
        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            obs_cpu[lx, ly, lz, pb] && continue
            gx = (pbx - 1) * bs + lx
            gy = (pby - 1) * bs + ly
            gz = (pbz - 1) * bs + lz

            has_children_at(gx, gy, gz) && continue

            rho_coarse = Float64(rho_cpu[lx, ly, lz, pb])
            ux_coarse = Float64(vel_cpu[lx, ly, lz, pb, 1])
            uy_coarse = Float64(vel_cpu[lx, ly, lz, pb, 2])
            uz_coarse = Float64(vel_cpu[lx, ly, lz, pb, 3])

            shell_face_count = 0
            for face_probe in 1:6
                _, _, _, _, valid_probe = shell_face_target_current(face_probe, gx, gy, gz)
                shell_face_count += valid_probe ? 1 : 0
            end

            for face in 1:6
                rho_face, ux_face, uy_face, uz_face, valid = shell_face_target_current(face, gx, gy, gz)
                valid || continue

                drho = rho_coarse - rho_face
                dux = ux_coarse - ux_face
                duy = uy_coarse - uy_face
                duz = uz_coarse - uz_face
                abs_drho = abs(drho)
                abs_du = sqrt(dux * dux + duy * duy + duz * duz)
                du_normal = face <= 2 ? dux : (face <= 4 ? duy : duz)
                normal_sign = (face == 1 || face == 3 || face == 5) ? 1.0 : -1.0
                un_coarse = normal_sign * (face <= 2 ? ux_coarse : (face <= 4 ? uy_coarse : uz_coarse))
                un_face = normal_sign * (face <= 2 ? ux_face : (face <= 4 ? uy_face : uz_face))
                dmass_flux = rho_coarse * un_coarse - rho_face * un_face
                p_coarse = Float64(CS2_PHYSICS) * rho_coarse
                p_face = Float64(CS2_PHYSICS) * rho_face
                dmom_flux = (rho_coarse * un_coarse * un_coarse + p_coarse) -
                            (rho_face * un_face * un_face + p_face)
                abs_dmass_flux = abs(dmass_flux)
                abs_dmom_flux = abs(dmom_flux)

                n_faces += 1
                abs_drho_sum += abs_drho
                abs_du_sum += abs_du
                sq_drho_sum += abs_drho * abs_drho
                sq_du_sum += abs_du * abs_du
                abs_dmass_flux_sum += abs_dmass_flux
                sq_dmass_flux_sum += dmass_flux * dmass_flux
                dmass_flux_sum += dmass_flux
                abs_dmom_flux_sum += abs_dmom_flux
                sq_dmom_flux_sum += dmom_flux * dmom_flux
                dmom_flux_sum += dmom_flux
                max_abs_drho = max(max_abs_drho, abs_drho)
                max_abs_du = max(max_abs_du, abs_du)

                if shell_face_count == 1
                    n_faces_single += 1
                    abs_drho_sum_single += abs_drho
                    abs_du_sum_single += abs_du
                    abs_dmass_flux_sum_single += abs_dmass_flux
                    abs_dmom_flux_sum_single += abs_dmom_flux
                else
                    n_faces_multi += 1
                    abs_drho_sum_multi += abs_drho
                    abs_du_sum_multi += abs_du
                    abs_dmass_flux_sum_multi += abs_dmass_flux
                    abs_dmom_flux_sum_multi += abs_dmom_flux
                end

                if face <= 2
                    n_faces_x += 1
                    abs_drho_sum_x += abs_drho
                    abs_du_sum_x += abs_du
                    abs_dmass_flux_sum_x += abs_dmass_flux
                    abs_dmom_flux_sum_x += abs_dmom_flux
                    if face == 1
                        n_faces_x_pos += 1
                        abs_drho_sum_x_pos += abs_drho
                        abs_du_sum_x_pos += abs_du
                        drho_sum_x_pos += drho
                        dun_sum_x_pos += du_normal
                        dmass_flux_sum_x_pos += dmass_flux
                        dmom_flux_sum_x_pos += dmom_flux
                    else
                        n_faces_x_neg += 1
                        abs_drho_sum_x_neg += abs_drho
                        abs_du_sum_x_neg += abs_du
                        drho_sum_x_neg += drho
                        dun_sum_x_neg += du_normal
                        dmass_flux_sum_x_neg += dmass_flux
                        dmom_flux_sum_x_neg += dmom_flux
                    end
                elseif face <= 4
                    n_faces_y += 1
                    abs_drho_sum_y += abs_drho
                    abs_du_sum_y += abs_du
                    abs_dmass_flux_sum_y += abs_dmass_flux
                    abs_dmom_flux_sum_y += abs_dmom_flux
                    if face == 3
                        n_faces_y_pos += 1
                        abs_drho_sum_y_pos += abs_drho
                        abs_du_sum_y_pos += abs_du
                        drho_sum_y_pos += drho
                        dun_sum_y_pos += du_normal
                        dmass_flux_sum_y_pos += dmass_flux
                        dmom_flux_sum_y_pos += dmom_flux
                    else
                        n_faces_y_neg += 1
                        abs_drho_sum_y_neg += abs_drho
                        abs_du_sum_y_neg += abs_du
                        drho_sum_y_neg += drho
                        dun_sum_y_neg += du_normal
                        dmass_flux_sum_y_neg += dmass_flux
                        dmom_flux_sum_y_neg += dmom_flux
                    end
                else
                    n_faces_z += 1
                    abs_drho_sum_z += abs_drho
                    abs_du_sum_z += abs_du
                    abs_dmass_flux_sum_z += abs_dmass_flux
                    abs_dmom_flux_sum_z += abs_dmom_flux
                    if face == 5
                        n_faces_z_pos += 1
                        abs_drho_sum_z_pos += abs_drho
                        abs_du_sum_z_pos += abs_du
                        drho_sum_z_pos += drho
                        dun_sum_z_pos += du_normal
                        dmass_flux_sum_z_pos += dmass_flux
                        dmom_flux_sum_z_pos += dmom_flux
                    else
                        n_faces_z_neg += 1
                        abs_drho_sum_z_neg += abs_drho
                        abs_du_sum_z_neg += abs_du
                        drho_sum_z_neg += drho
                        dun_sum_z_neg += du_normal
                        dmass_flux_sum_z_neg += dmass_flux
                        dmom_flux_sum_z_neg += dmom_flux
                    end
                end
            end
        end
    end

    return (
        n_faces = n_faces,
        mean_abs_drho = n_faces > 0 ? abs_drho_sum / n_faces : 0.0,
        max_abs_drho = max_abs_drho,
        mean_abs_du = n_faces > 0 ? abs_du_sum / n_faces : 0.0,
        max_abs_du = max_abs_du,
        rms_drho = n_faces > 0 ? sqrt(sq_drho_sum / n_faces) : 0.0,
        rms_du = n_faces > 0 ? sqrt(sq_du_sum / n_faces) : 0.0,
        n_faces_x = n_faces_x,
        n_faces_y = n_faces_y,
        n_faces_z = n_faces_z,
        mean_abs_drho_x = n_faces_x > 0 ? abs_drho_sum_x / n_faces_x : 0.0,
        mean_abs_drho_y = n_faces_y > 0 ? abs_drho_sum_y / n_faces_y : 0.0,
        mean_abs_drho_z = n_faces_z > 0 ? abs_drho_sum_z / n_faces_z : 0.0,
        mean_abs_du_x = n_faces_x > 0 ? abs_du_sum_x / n_faces_x : 0.0,
        mean_abs_du_y = n_faces_y > 0 ? abs_du_sum_y / n_faces_y : 0.0,
        mean_abs_du_z = n_faces_z > 0 ? abs_du_sum_z / n_faces_z : 0.0,
        mean_abs_drho_x_pos = n_faces_x_pos > 0 ? abs_drho_sum_x_pos / n_faces_x_pos : 0.0,
        mean_abs_drho_x_neg = n_faces_x_neg > 0 ? abs_drho_sum_x_neg / n_faces_x_neg : 0.0,
        mean_abs_drho_y_pos = n_faces_y_pos > 0 ? abs_drho_sum_y_pos / n_faces_y_pos : 0.0,
        mean_abs_drho_y_neg = n_faces_y_neg > 0 ? abs_drho_sum_y_neg / n_faces_y_neg : 0.0,
        mean_abs_drho_z_pos = n_faces_z_pos > 0 ? abs_drho_sum_z_pos / n_faces_z_pos : 0.0,
        mean_abs_drho_z_neg = n_faces_z_neg > 0 ? abs_drho_sum_z_neg / n_faces_z_neg : 0.0,
        mean_abs_du_x_pos = n_faces_x_pos > 0 ? abs_du_sum_x_pos / n_faces_x_pos : 0.0,
        mean_abs_du_x_neg = n_faces_x_neg > 0 ? abs_du_sum_x_neg / n_faces_x_neg : 0.0,
        mean_abs_du_y_pos = n_faces_y_pos > 0 ? abs_du_sum_y_pos / n_faces_y_pos : 0.0,
        mean_abs_du_y_neg = n_faces_y_neg > 0 ? abs_du_sum_y_neg / n_faces_y_neg : 0.0,
        mean_abs_du_z_pos = n_faces_z_pos > 0 ? abs_du_sum_z_pos / n_faces_z_pos : 0.0,
        mean_abs_du_z_neg = n_faces_z_neg > 0 ? abs_du_sum_z_neg / n_faces_z_neg : 0.0,
        mean_drho_x_pos = n_faces_x_pos > 0 ? drho_sum_x_pos / n_faces_x_pos : 0.0,
        mean_drho_x_neg = n_faces_x_neg > 0 ? drho_sum_x_neg / n_faces_x_neg : 0.0,
        mean_drho_y_pos = n_faces_y_pos > 0 ? drho_sum_y_pos / n_faces_y_pos : 0.0,
        mean_drho_y_neg = n_faces_y_neg > 0 ? drho_sum_y_neg / n_faces_y_neg : 0.0,
        mean_drho_z_pos = n_faces_z_pos > 0 ? drho_sum_z_pos / n_faces_z_pos : 0.0,
        mean_drho_z_neg = n_faces_z_neg > 0 ? drho_sum_z_neg / n_faces_z_neg : 0.0,
        mean_dun_x_pos = n_faces_x_pos > 0 ? dun_sum_x_pos / n_faces_x_pos : 0.0,
        mean_dun_x_neg = n_faces_x_neg > 0 ? dun_sum_x_neg / n_faces_x_neg : 0.0,
        mean_dun_y_pos = n_faces_y_pos > 0 ? dun_sum_y_pos / n_faces_y_pos : 0.0,
        mean_dun_y_neg = n_faces_y_neg > 0 ? dun_sum_y_neg / n_faces_y_neg : 0.0,
        mean_dun_z_pos = n_faces_z_pos > 0 ? dun_sum_z_pos / n_faces_z_pos : 0.0,
        mean_dun_z_neg = n_faces_z_neg > 0 ? dun_sum_z_neg / n_faces_z_neg : 0.0,
        mean_abs_dmass_flux = n_faces > 0 ? abs_dmass_flux_sum / n_faces : 0.0,
        rms_dmass_flux = n_faces > 0 ? sqrt(sq_dmass_flux_sum / n_faces) : 0.0,
        mean_dmass_flux = n_faces > 0 ? dmass_flux_sum / n_faces : 0.0,
        mean_abs_dmom_flux = n_faces > 0 ? abs_dmom_flux_sum / n_faces : 0.0,
        rms_dmom_flux = n_faces > 0 ? sqrt(sq_dmom_flux_sum / n_faces) : 0.0,
        mean_dmom_flux = n_faces > 0 ? dmom_flux_sum / n_faces : 0.0,
        mean_abs_dmass_flux_x = n_faces_x > 0 ? abs_dmass_flux_sum_x / n_faces_x : 0.0,
        mean_abs_dmass_flux_y = n_faces_y > 0 ? abs_dmass_flux_sum_y / n_faces_y : 0.0,
        mean_abs_dmass_flux_z = n_faces_z > 0 ? abs_dmass_flux_sum_z / n_faces_z : 0.0,
        mean_abs_dmom_flux_x = n_faces_x > 0 ? abs_dmom_flux_sum_x / n_faces_x : 0.0,
        mean_abs_dmom_flux_y = n_faces_y > 0 ? abs_dmom_flux_sum_y / n_faces_y : 0.0,
        mean_abs_dmom_flux_z = n_faces_z > 0 ? abs_dmom_flux_sum_z / n_faces_z : 0.0,
        mean_dmass_flux_x_pos = n_faces_x_pos > 0 ? dmass_flux_sum_x_pos / n_faces_x_pos : 0.0,
        mean_dmass_flux_x_neg = n_faces_x_neg > 0 ? dmass_flux_sum_x_neg / n_faces_x_neg : 0.0,
        mean_dmass_flux_y_pos = n_faces_y_pos > 0 ? dmass_flux_sum_y_pos / n_faces_y_pos : 0.0,
        mean_dmass_flux_y_neg = n_faces_y_neg > 0 ? dmass_flux_sum_y_neg / n_faces_y_neg : 0.0,
        mean_dmass_flux_z_pos = n_faces_z_pos > 0 ? dmass_flux_sum_z_pos / n_faces_z_pos : 0.0,
        mean_dmass_flux_z_neg = n_faces_z_neg > 0 ? dmass_flux_sum_z_neg / n_faces_z_neg : 0.0,
        mean_dmom_flux_x_pos = n_faces_x_pos > 0 ? dmom_flux_sum_x_pos / n_faces_x_pos : 0.0,
        mean_dmom_flux_x_neg = n_faces_x_neg > 0 ? dmom_flux_sum_x_neg / n_faces_x_neg : 0.0,
        mean_dmom_flux_y_pos = n_faces_y_pos > 0 ? dmom_flux_sum_y_pos / n_faces_y_pos : 0.0,
        mean_dmom_flux_y_neg = n_faces_y_neg > 0 ? dmom_flux_sum_y_neg / n_faces_y_neg : 0.0,
        mean_dmom_flux_z_pos = n_faces_z_pos > 0 ? dmom_flux_sum_z_pos / n_faces_z_pos : 0.0,
        mean_dmom_flux_z_neg = n_faces_z_neg > 0 ? dmom_flux_sum_z_neg / n_faces_z_neg : 0.0,
        n_faces_single = n_faces_single,
        n_faces_multi = n_faces_multi,
        mean_abs_drho_single = n_faces_single > 0 ? abs_drho_sum_single / n_faces_single : 0.0,
        mean_abs_drho_multi = n_faces_multi > 0 ? abs_drho_sum_multi / n_faces_multi : 0.0,
        mean_abs_du_single = n_faces_single > 0 ? abs_du_sum_single / n_faces_single : 0.0,
        mean_abs_du_multi = n_faces_multi > 0 ? abs_du_sum_multi / n_faces_multi : 0.0,
        mean_abs_dmass_flux_single = n_faces_single > 0 ? abs_dmass_flux_sum_single / n_faces_single : 0.0,
        mean_abs_dmass_flux_multi = n_faces_multi > 0 ? abs_dmass_flux_sum_multi / n_faces_multi : 0.0,
        mean_abs_dmom_flux_single = n_faces_single > 0 ? abs_dmom_flux_sum_single / n_faces_single : 0.0,
        mean_abs_dmom_flux_multi = n_faces_multi > 0 ? abs_dmom_flux_sum_multi / n_faces_multi : 0.0,
    )
end

# ────────────────────────────────────────────────────────────────────
# Pass190: Cross-shell Π_neq Frobenius jump
# ────────────────────────────────────────────────────────────────────
# Companion to compute_amr_shell_face_jump_stats. For every parent-cell
# face that abuts a refined neighbor, compare the parent Π_neq tensor to
# the face-averaged child Π_neq, and aggregate ||ΔΠ||_F over all such
# faces. Frobenius norm with the standard off-diagonal weight of 2:
#   ||ΔΠ||_F² = Δπxx² + Δπyy² + Δπzz² + 2(Δπxy² + Δπxz² + Δπyz²)
# This is the stress-tensor analogue of the existing ρ/u jump and is
# what Astoul 2020 and Schornbaum waLBerla use to declare "smooth stress
# transfer at refinement interfaces". Returns zero stats when either
# level lacks a Π_neq carrier (e.g. BGK without STRESS_TENSOR_TRANSFER).
# Observation-only — no physics behavior change.
function write_amr_face_register_csv_header(filepath::String)
    open(filepath, "w") do io
        println(io,
            "step,parent_level,child_level,face_id,parent_block,parent_gx,parent_gy,parent_gz," *
            "axis,normal_sign,shell_face_count,n_child_face_cells," *
            "rho_parent,ux_parent,uy_parent,uz_parent,un_parent," *
            "rho_child_face,ux_child_face,uy_child_face,uz_child_face,un_child_face," *
            "drho,du,du_normal,dmass_flux,dmom_flux,abs_dmass_flux,abs_dmom_flux," *
            "has_time_snapshot,rho_parent_old,ux_parent_old,uy_parent_old,uz_parent_old,un_parent_old," *
            "rho_child_face_old,ux_child_face_old,uy_child_face_old,uz_child_face_old,un_child_face_old," *
            "dmass_flux_old,dmom_flux_old,dmass_flux_trap,dmom_flux_trap," *
            "abs_dmass_flux_trap,abs_dmom_flux_trap")
    end
end

# Read-only per-face conservative ledger for AMR interfaces.
#
# The aggregate AMR diagnostics tell us the size of the jump. This register
# tells us which exact coarse/fine faces carry the signed residual. It is the
# data structure we need before applying any refluxing: first prove the face
# accounting, then turn the ledger into a correction.
function append_amr_face_register_rows!(
    io,
    step::Int,
    lvl::Int,
    parent_level::BlockLevel,
    child_level::BlockLevel,
)
    rho_cpu = Array(parent_level.rho)
    vel_cpu = Array(parent_level.vel)
    obs_cpu = Array(parent_level.obstacle)
    parent_map_x_cpu = Array(parent_level.map_x)
    parent_map_y_cpu = Array(parent_level.map_y)
    parent_map_z_cpu = Array(parent_level.map_z)
    child_ptr_cpu = Array(child_level.block_pointer)
    child_rho_cpu = Array(child_level.rho)
    child_vel_cpu = Array(child_level.vel)
    parent_has_time = length(parent_level.rho_old) > 1 && length(parent_level.vel_old) > 1
    child_has_time = length(child_level.rho_old) > 1 && length(child_level.vel_old) > 1
    parent_rho_old_cpu = parent_has_time ? Array(parent_level.rho_old) : nothing
    parent_vel_old_cpu = parent_has_time ? Array(parent_level.vel_old) : nothing
    child_rho_old_cpu = child_has_time ? Array(child_level.rho_old) : nothing
    child_vel_old_cpu = child_has_time ? Array(child_level.vel_old) : nothing
    has_time_snapshot = parent_has_time && child_has_time

    cdx = size(child_ptr_cpu, 1)
    cdy = size(child_ptr_cpu, 2)
    cdz = size(child_ptr_cpu, 3)
    bs = BLOCK_SIZE
    cs2 = Float64(CS2_PHYSICS)
    rows_written = 0

    @inline function child_cell_state(fgx::Int, fgy::Int, fgz::Int)
        if fgx < 1 || fgy < 1 || fgz < 1
            return (0.0, 0.0, 0.0, 0.0, false)
        end
        cbx = div(fgx - 1, bs) + 1
        cby = div(fgy - 1, bs) + 1
        cbz = div(fgz - 1, bs) + 1
        if cbx > cdx || cby > cdy || cbz > cdz
            return (0.0, 0.0, 0.0, 0.0, false)
        end
        cb_idx = child_ptr_cpu[cbx, cby, cbz]
        cb_idx == 0 && return (0.0, 0.0, 0.0, 0.0, false)
        clx = (fgx - 1) % bs + 1
        cly = (fgy - 1) % bs + 1
        clz = (fgz - 1) % bs + 1
        return (
            Float64(child_rho_cpu[clx, cly, clz, cb_idx]),
            Float64(child_vel_cpu[clx, cly, clz, cb_idx, 1]),
            Float64(child_vel_cpu[clx, cly, clz, cb_idx, 2]),
            Float64(child_vel_cpu[clx, cly, clz, cb_idx, 3]),
            true,
        )
    end

    @inline function child_cell_state_old(fgx::Int, fgy::Int, fgz::Int)
        if fgx < 1 || fgy < 1 || fgz < 1
            return (0.0, 0.0, 0.0, 0.0, false)
        end
        cbx = div(fgx - 1, bs) + 1
        cby = div(fgy - 1, bs) + 1
        cbz = div(fgz - 1, bs) + 1
        if cbx > cdx || cby > cdy || cbz > cdz
            return (0.0, 0.0, 0.0, 0.0, false)
        end
        cb_idx = child_ptr_cpu[cbx, cby, cbz]
        cb_idx == 0 && return (0.0, 0.0, 0.0, 0.0, false)
        clx = (fgx - 1) % bs + 1
        cly = (fgy - 1) % bs + 1
        clz = (fgz - 1) % bs + 1
        return (
            Float64(child_rho_old_cpu[clx, cly, clz, cb_idx]),
            Float64(child_vel_old_cpu[clx, cly, clz, cb_idx, 1]),
            Float64(child_vel_old_cpu[clx, cly, clz, cb_idx, 2]),
            Float64(child_vel_old_cpu[clx, cly, clz, cb_idx, 3]),
            true,
        )
    end

    @inline function has_children_at(gx::Int, gy::Int, gz::Int)
        for sx in 0:1, sy in 0:1, sz in 0:1
            fgx = 2 * gx - 1 + sx
            fgy = 2 * gy - 1 + sy
            fgz = 2 * gz - 1 + sz
            _, _, _, _, valid = child_cell_state(fgx, fgy, fgz)
            valid || return false
        end
        return true
    end

    @inline function shell_face_target(face::Int, gx::Int, gy::Int, gz::Int, use_old::Bool=false)
        mother_fx0 = 0
        mother_fy0 = 0
        mother_fz0 = 0
        refined_neighbor = false

        if face == 1
            refined_neighbor = has_children_at(gx + 1, gy, gz)
            mother_fx0 = 2 * gx + 1
            mother_fy0 = 2 * gy - 1
            mother_fz0 = 2 * gz - 1
        elseif face == 2
            refined_neighbor = has_children_at(gx - 1, gy, gz)
            mother_fx0 = 2 * gx - 2
            mother_fy0 = 2 * gy - 1
            mother_fz0 = 2 * gz - 1
        elseif face == 3
            refined_neighbor = has_children_at(gx, gy + 1, gz)
            mother_fx0 = 2 * gx - 1
            mother_fy0 = 2 * gy + 1
            mother_fz0 = 2 * gz - 1
        elseif face == 4
            refined_neighbor = has_children_at(gx, gy - 1, gz)
            mother_fx0 = 2 * gx - 1
            mother_fy0 = 2 * gy - 2
            mother_fz0 = 2 * gz - 1
        elseif face == 5
            refined_neighbor = has_children_at(gx, gy, gz + 1)
            mother_fx0 = 2 * gx - 1
            mother_fy0 = 2 * gy - 1
            mother_fz0 = 2 * gz + 1
        else
            refined_neighbor = has_children_at(gx, gy, gz - 1)
            mother_fx0 = 2 * gx - 1
            mother_fy0 = 2 * gy - 1
            mother_fz0 = 2 * gz - 2
        end

        refined_neighbor || return (0.0, 0.0, 0.0, 0.0, 0, false)

        sx_first = 0
        sx_last = 1
        sy_first = 0
        sy_last = 1
        sz_first = 0
        sz_last = 1
        if face <= 2
            sx_last = 0
        elseif face <= 4
            sy_last = 0
        else
            sz_last = 0
        end

        rho_sum = 0.0
        ux_sum = 0.0
        uy_sum = 0.0
        uz_sum = 0.0
        n_face = 0
        for sx in sx_first:sx_last, sy in sy_first:sy_last, sz in sz_first:sz_last
            rho_v, ux_v, uy_v, uz_v, valid = (use_old && has_time_snapshot) ?
                child_cell_state_old(mother_fx0 + sx, mother_fy0 + sy, mother_fz0 + sz) :
                child_cell_state(mother_fx0 + sx, mother_fy0 + sy, mother_fz0 + sz)
            valid || return (0.0, 0.0, 0.0, 0.0, 0, false)
            rho_sum += rho_v
            ux_sum += ux_v
            uy_sum += uy_v
            uz_sum += uz_v
            n_face += 1
        end

        inv_face = 1.0 / max(n_face, 1)
        return (
            inv_face * rho_sum,
            inv_face * ux_sum,
            inv_face * uy_sum,
            inv_face * uz_sum,
            n_face,
            true,
        )
    end

    n_blocks = size(rho_cpu, 4)
    @inbounds for pb in 1:n_blocks
        pbx = Int(parent_map_x_cpu[pb])
        pby = Int(parent_map_y_cpu[pb])
        pbz = Int(parent_map_z_cpu[pb])
        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            obs_cpu[lx, ly, lz, pb] && continue
            gx = (pbx - 1) * bs + lx
            gy = (pby - 1) * bs + ly
            gz = (pbz - 1) * bs + lz
            has_children_at(gx, gy, gz) && continue

            rho_parent = Float64(rho_cpu[lx, ly, lz, pb])
            ux_parent = Float64(vel_cpu[lx, ly, lz, pb, 1])
            uy_parent = Float64(vel_cpu[lx, ly, lz, pb, 2])
            uz_parent = Float64(vel_cpu[lx, ly, lz, pb, 3])

            shell_face_count = 0
            for face_probe in 1:6
                _, _, _, _, _, valid_probe = shell_face_target(face_probe, gx, gy, gz)
                shell_face_count += valid_probe ? 1 : 0
            end
            shell_face_count == 0 && continue

            for face in 1:6
                rho_face, ux_face, uy_face, uz_face, n_child_face_cells, valid =
                    shell_face_target(face, gx, gy, gz)
                valid || continue

                axis = face <= 2 ? 1 : (face <= 4 ? 2 : 3)
                axis_name = axis == 1 ? "x" : (axis == 2 ? "y" : "z")
                normal_sign = (face == 1 || face == 3 || face == 5) ? 1 : -1
                normal_scale = Float64(normal_sign)
                u_parent_axis = axis == 1 ? ux_parent : (axis == 2 ? uy_parent : uz_parent)
                u_face_axis = axis == 1 ? ux_face : (axis == 2 ? uy_face : uz_face)
                un_parent = normal_scale * u_parent_axis
                un_face = normal_scale * u_face_axis

                dux = ux_parent - ux_face
                duy = uy_parent - uy_face
                duz = uz_parent - uz_face
                drho = rho_parent - rho_face
                du = sqrt(dux * dux + duy * duy + duz * duz)
                du_normal = u_parent_axis - u_face_axis
                dmass_flux = rho_parent * un_parent - rho_face * un_face
                dmom_flux = (rho_parent * un_parent * un_parent + cs2 * rho_parent) -
                            (rho_face * un_face * un_face + cs2 * rho_face)
                if has_time_snapshot
                    rho_parent_old = Float64(parent_rho_old_cpu[lx, ly, lz, pb])
                    ux_parent_old = Float64(parent_vel_old_cpu[lx, ly, lz, pb, 1])
                    uy_parent_old = Float64(parent_vel_old_cpu[lx, ly, lz, pb, 2])
                    uz_parent_old = Float64(parent_vel_old_cpu[lx, ly, lz, pb, 3])
                else
                    rho_parent_old = rho_parent
                    ux_parent_old = ux_parent
                    uy_parent_old = uy_parent
                    uz_parent_old = uz_parent
                end
                rho_face_old, ux_face_old, uy_face_old, uz_face_old, _, valid_old =
                    shell_face_target(face, gx, gy, gz, true)
                if !valid_old
                    rho_face_old = rho_face
                    ux_face_old = ux_face
                    uy_face_old = uy_face
                    uz_face_old = uz_face
                end
                u_parent_axis_old = axis == 1 ? ux_parent_old : (axis == 2 ? uy_parent_old : uz_parent_old)
                u_face_axis_old = axis == 1 ? ux_face_old : (axis == 2 ? uy_face_old : uz_face_old)
                un_parent_old = normal_scale * u_parent_axis_old
                un_face_old = normal_scale * u_face_axis_old
                dmass_flux_old = rho_parent_old * un_parent_old - rho_face_old * un_face_old
                dmom_flux_old = (rho_parent_old * un_parent_old * un_parent_old + cs2 * rho_parent_old) -
                                (rho_face_old * un_face_old * un_face_old + cs2 * rho_face_old)
                time_snapshot_flag = (has_time_snapshot && valid_old) ? 1 : 0
                dmass_flux_trap = time_snapshot_flag == 1 ? 0.5 * (dmass_flux + dmass_flux_old) : dmass_flux
                dmom_flux_trap = time_snapshot_flag == 1 ? 0.5 * (dmom_flux + dmom_flux_old) : dmom_flux
                face_id = @sprintf("L%d_%d_%d_%d_%s_%+d", lvl, gx, gy, gz, axis_name, normal_sign)

                @printf(io,
                    "%d,%d,%d,%s,%d,%d,%d,%d,%s,%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",
                    step, lvl, lvl + 1, face_id, pb, gx, gy, gz, axis_name, normal_sign,
                    shell_face_count, n_child_face_cells,
                    rho_parent, ux_parent, uy_parent, uz_parent, un_parent,
                    rho_face, ux_face, uy_face, uz_face, un_face,
                    drho, du, du_normal, dmass_flux, dmom_flux, abs(dmass_flux), abs(dmom_flux),
                    time_snapshot_flag,
                    rho_parent_old, ux_parent_old, uy_parent_old, uz_parent_old, un_parent_old,
                    rho_face_old, ux_face_old, uy_face_old, uz_face_old, un_face_old,
                    dmass_flux_old, dmom_flux_old, dmass_flux_trap, dmom_flux_trap,
                    abs(dmass_flux_trap), abs(dmom_flux_trap))
                rows_written += 1
            end
        end
    end

    return rows_written
end

function write_amr_face_register(grids; csv_path::Union{Nothing,String}=nothing, step::Int=-1)
    csv_path === nothing && return
    n = length(grids)
    n < 2 && return
    total_rows = 0
    open(csv_path, "a") do io
        for lvl in 1:(n - 1)
            parent_level = grids[lvl]
            child_level = grids[lvl + 1]
            length(parent_level.active_block_coords) == 0 && continue
            length(child_level.active_block_coords) == 0 && continue
            total_rows += append_amr_face_register_rows!(io, step, lvl, parent_level, child_level)
        end
    end
    @printf("          [AMR-FACE] register rows=%d -> %s\n", total_rows, csv_path)
end

function write_amr_population_stream_register_csv_header(filepath::String)
    open(filepath, "w") do io
        println(io,
            "step,parent_level,child_level,k,cx,cy,cz,active_cells,cross_count," *
            "coarse_stream,fine_stream,stream_delta,abs_stream_delta,rel_stream_delta," *
            "mean_coarse_stream,mean_fine_stream")
    end
end

function write_amr_population_stream_register(
    grids,
    cx_arr,
    cy_arr,
    cz_arr;
    csv_path::Union{Nothing,String}=nothing,
    step::Int=-1
)
    csv_path === nothing && return
    n = length(grids)
    n < 2 && return
    cx_cpu = Array(cx_arr)
    cy_cpu = Array(cy_arr)
    cz_cpu = Array(cz_arr)
    total_rows = 0
    open(csv_path, "a") do io
        for lvl in 1:(n - 1)
            parent_level = grids[lvl]
            child_level = grids[lvl + 1]
            length(parent_level.active_block_coords) == 0 && continue
            length(child_level.active_block_coords) == 0 && continue
            size(parent_level.amr_face_budget, 5) >=
                Int(AMR_POP_STREAM_REGISTER_BASE +
                    AMR_POP_STREAM_REGISTER_STRIDE * AMR_POP_STREAM_REGISTER_MAX_DIRS) ||
                continue
            budget_cpu = Array(parent_level.amr_face_budget)
            n_dirs = min(
                Int(AMR_POP_STREAM_REGISTER_MAX_DIRS),
                length(cx_cpu),
                div(size(budget_cpu, 5) - Int(AMR_POP_STREAM_REGISTER_BASE),
                    Int(AMR_POP_STREAM_REGISTER_STRIDE))
            )
            for k in 1:n_dirs
                coarse_slot = Int(AMR_POP_STREAM_REGISTER_BASE) +
                              (k - 1) * Int(AMR_POP_STREAM_REGISTER_STRIDE) + 1
                fine_slot = coarse_slot + 1
                count_slot = coarse_slot + 2
                coarse_sum = 0.0
                fine_sum = 0.0
                count_sum = 0.0
                active_cells = 0
                coarse_view = @view budget_cpu[:, :, :, :, coarse_slot]
                fine_view = @view budget_cpu[:, :, :, :, fine_slot]
                count_view = @view budget_cpu[:, :, :, :, count_slot]
                for (coarse_v, fine_v, count_v) in zip(coarse_view, fine_view, count_view)
                    coarse_f = Float64(coarse_v)
                    fine_f = Float64(fine_v)
                    count_f = Float64(count_v)
                    coarse_sum += coarse_f
                    fine_sum += fine_f
                    count_sum += count_f
                    if count_f != 0.0 || coarse_f != 0.0 || fine_f != 0.0
                        active_cells += 1
                    end
                end
                active_cells == 0 && continue
                delta = fine_sum - coarse_sum
                denom = max(abs(coarse_sum) + abs(fine_sum), 1.0e-30)
                rel_delta = abs(delta) / denom
                mean_coarse = count_sum > 0.0 ? coarse_sum / count_sum : 0.0
                mean_fine = count_sum > 0.0 ? fine_sum / count_sum : 0.0
                @printf(io,
                    "%d,%d,%d,%d,%d,%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",
                    step, lvl, lvl + 1, k,
                    Int(cx_cpu[k]), Int(cy_cpu[k]), Int(cz_cpu[k]),
                    active_cells, count_sum,
                    coarse_sum, fine_sum, delta, abs(delta), rel_delta,
                    mean_coarse, mean_fine)
                total_rows += 1
            end
        end
    end
    @printf("          [AMR-POP] stream-register rows=%d -> %s\n", total_rows, csv_path)
end

function write_amr_population_stream_face_register_csv_header(filepath::String)
    open(filepath, "w") do io
        println(io,
            "step,parent_level,child_level,face_id,axis,normal_sign,k,cx,cy,cz," *
            "active_cells,cross_count,coarse_stream,fine_stream,stream_delta," *
            "abs_stream_delta,rel_stream_delta,mean_coarse_stream,mean_fine_stream")
    end
end

function amr_population_face_axis_sign(face_id::Int)
    if face_id == 1
        return ("x", 1)
    elseif face_id == 2
        return ("x", -1)
    elseif face_id == 3
        return ("y", 1)
    elseif face_id == 4
        return ("y", -1)
    elseif face_id == 5
        return ("z", 1)
    end
    return ("z", -1)
end

function write_amr_population_stream_face_register(
    grids,
    cx_arr,
    cy_arr,
    cz_arr;
    csv_path::Union{Nothing,String}=nothing,
    step::Int=-1
)
    csv_path === nothing && return
    n = length(grids)
    n < 2 && return
    cx_cpu = Array(cx_arr)
    cy_cpu = Array(cy_arr)
    cz_cpu = Array(cz_arr)
    total_rows = 0
    open(csv_path, "a") do io
        for lvl in 1:(n - 1)
            parent_level = grids[lvl]
            child_level = grids[lvl + 1]
            length(parent_level.active_block_coords) == 0 && continue
            length(child_level.active_block_coords) == 0 && continue
            size(parent_level.amr_face_budget, 5) >=
                Int(AMR_POP_STREAM_FACE_BASE + AMR_POP_STREAM_FACE_COMPONENTS) ||
                continue
            budget_cpu = Array(parent_level.amr_face_budget)
            n_dirs = min(Int(AMR_POP_STREAM_REGISTER_MAX_DIRS), length(cx_cpu), length(cy_cpu), length(cz_cpu))
            for face_id in 1:Int(AMR_POP_STREAM_FACE_COUNT)
                axis, normal_sign = amr_population_face_axis_sign(face_id)
                for k in 1:n_dirs
                    coarse_slot = Int(AMR_POP_STREAM_FACE_BASE) +
                                  ((face_id - 1) * Int(AMR_POP_STREAM_REGISTER_MAX_DIRS) +
                                   (k - 1)) * Int(AMR_POP_STREAM_REGISTER_STRIDE) + 1
                    fine_slot = coarse_slot + 1
                    count_slot = coarse_slot + 2
                    coarse_sum = 0.0
                    fine_sum = 0.0
                    count_sum = 0.0
                    active_cells = 0
                    coarse_view = @view budget_cpu[:, :, :, :, coarse_slot]
                    fine_view = @view budget_cpu[:, :, :, :, fine_slot]
                    count_view = @view budget_cpu[:, :, :, :, count_slot]
                    for (coarse_v, fine_v, count_v) in zip(coarse_view, fine_view, count_view)
                        coarse_f = Float64(coarse_v)
                        fine_f = Float64(fine_v)
                        count_f = Float64(count_v)
                        coarse_sum += coarse_f
                        fine_sum += fine_f
                        count_sum += count_f
                        if count_f != 0.0 || coarse_f != 0.0 || fine_f != 0.0
                            active_cells += 1
                        end
                    end
                    active_cells == 0 && continue
                    delta = fine_sum - coarse_sum
                    denom = max(abs(coarse_sum) + abs(fine_sum), 1.0e-30)
                    rel_delta = abs(delta) / denom
                    mean_coarse = count_sum > 0.0 ? coarse_sum / count_sum : 0.0
                    mean_fine = count_sum > 0.0 ? fine_sum / count_sum : 0.0
                    @printf(io,
                        "%d,%d,%d,%d,%s,%d,%d,%d,%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",
                        step, lvl, lvl + 1, face_id, axis, normal_sign, k,
                        Int(cx_cpu[k]), Int(cy_cpu[k]), Int(cz_cpu[k]),
                        active_cells, count_sum,
                        coarse_sum, fine_sum, delta, abs(delta), rel_delta,
                        mean_coarse, mean_fine)
                    total_rows += 1
                end
            end
        end
    end
    @printf("          [AMR-POP-FACE] stream-register rows=%d -> %s\n", total_rows, csv_path)
end

# Runtime, subcycle-aware face budget accumulator.
#
# This is deliberately read-only and guarded by
# AMR_RUNTIME_FACE_BUDGET_DIAGNOSTICS. Unlike the snapshot trapezoid in the
# face register, this collects the parent-side face flux once per parent step
# and the child-side face flux after every fine substep, weighted by each
# level's own dt. This is the conservative budget a future reflux correction
# should close.
mutable struct AMRRuntimeFaceBudgetRow
    parent_level::Int
    child_level::Int
    face_id::String
    parent_block::Int
    parent_lx::Int
    parent_ly::Int
    parent_lz::Int
    parent_gx::Int
    parent_gy::Int
    parent_gz::Int
    axis::String
    normal_sign::Int
    shell_face_count::Int
    n_child_face_cells::Int
    n_parent_samples::Int
    n_child_samples::Int
    parent_time::Float64
    child_time::Float64
    parent_mass_flux_int::Float64
    child_mass_flux_int::Float64
    parent_mom_flux_int::Float64
    child_mom_flux_int::Float64
end

const AMR_RUNTIME_FACE_BUDGET = Ref(Dict{String, AMRRuntimeFaceBudgetRow}())

function reset_amr_runtime_face_budget!()
    empty!(AMR_RUNTIME_FACE_BUDGET[])
    return nothing
end

function write_amr_runtime_face_budget_csv_header(filepath::String)
    open(filepath, "w") do io
        println(io,
            "step,parent_level,child_level,face_id,parent_block,parent_lx,parent_ly,parent_lz," *
            "parent_gx,parent_gy,parent_gz,axis,normal_sign," *
            "shell_face_count,n_child_face_cells,n_parent_samples,n_child_samples," *
            "parent_time,child_time,parent_mass_flux_int,child_mass_flux_int,dmass_flux_int," *
            "parent_mom_flux_int,child_mom_flux_int,dmom_flux_int," *
            "dmass_flux_rate,dmom_flux_rate,abs_dmass_flux_rate,abs_dmom_flux_rate")
    end
end

@inline function _amr_runtime_budget_row!(
    parent_level_idx::Int,
    face_id::String,
    parent_block::Int,
    parent_lx::Int,
    parent_ly::Int,
    parent_lz::Int,
    parent_gx::Int,
    parent_gy::Int,
    parent_gz::Int,
    axis_name::String,
    normal_sign::Int,
    shell_face_count::Int,
    n_child_face_cells::Int,
)
    rows = AMR_RUNTIME_FACE_BUDGET[]
    return get!(rows, face_id) do
        AMRRuntimeFaceBudgetRow(
            parent_level_idx,
            parent_level_idx + 1,
            face_id,
            parent_block,
            parent_lx,
            parent_ly,
            parent_lz,
            parent_gx,
            parent_gy,
            parent_gz,
            axis_name,
            normal_sign,
            shell_face_count,
            n_child_face_cells,
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    end
end

function accumulate_amr_runtime_face_budget!(
    parent_level::BlockLevel,
    child_level::BlockLevel,
    parent_level_idx::Int;
    side::Symbol,
    dt_weight::Float64,
    parent_vel_current=nothing,
    child_vel_current=nothing,
)
    side === :parent || side === :child ||
        error("AMR runtime face budget side must be :parent or :child")
    dt_weight > 0.0 || return 0

    parent_rho_cpu = Array(parent_level.rho)
    parent_vel_cpu = Array(parent_vel_current === nothing ? parent_level.vel : parent_vel_current)
    parent_obs_cpu = Array(parent_level.obstacle)
    parent_map_x_cpu = Array(parent_level.map_x)
    parent_map_y_cpu = Array(parent_level.map_y)
    parent_map_z_cpu = Array(parent_level.map_z)

    child_ptr_cpu = Array(child_level.block_pointer)
    child_obs_cpu = Array(child_level.obstacle)
    child_rho_cpu = Array(child_level.rho)
    child_vel_cpu = Array(child_vel_current === nothing ? child_level.vel : child_vel_current)

    cdx = size(child_ptr_cpu, 1)
    cdy = size(child_ptr_cpu, 2)
    cdz = size(child_ptr_cpu, 3)
    bs = BLOCK_SIZE
    cs2 = Float64(CS2_PHYSICS)
    rows_touched = 0

    @inline function child_cell_state(fgx::Int, fgy::Int, fgz::Int)
        if fgx < 1 || fgy < 1 || fgz < 1
            return (0.0, 0.0, 0.0, 0.0, false)
        end
        cbx = div(fgx - 1, bs) + 1
        cby = div(fgy - 1, bs) + 1
        cbz = div(fgz - 1, bs) + 1
        if cbx > cdx || cby > cdy || cbz > cdz
            return (0.0, 0.0, 0.0, 0.0, false)
        end
        cb_idx = child_ptr_cpu[cbx, cby, cbz]
        cb_idx == 0 && return (0.0, 0.0, 0.0, 0.0, false)
        clx = (fgx - 1) % bs + 1
        cly = (fgy - 1) % bs + 1
        clz = (fgz - 1) % bs + 1
        child_obs_cpu[clx, cly, clz, cb_idx] &&
            return (0.0, 0.0, 0.0, 0.0, false)
        return (
            Float64(child_rho_cpu[clx, cly, clz, cb_idx]),
            Float64(child_vel_cpu[clx, cly, clz, cb_idx, 1]),
            Float64(child_vel_cpu[clx, cly, clz, cb_idx, 2]),
            Float64(child_vel_cpu[clx, cly, clz, cb_idx, 3]),
            true,
        )
    end

    @inline function has_children_at(gx::Int, gy::Int, gz::Int)
        for sx in 0:1, sy in 0:1, sz in 0:1
            _, _, _, _, valid = child_cell_state(
                2 * gx - 1 + sx,
                2 * gy - 1 + sy,
                2 * gz - 1 + sz,
            )
            valid || return false
        end
        return true
    end

    @inline function shell_face_target(face::Int, gx::Int, gy::Int, gz::Int)
        mother_fx0 = 0
        mother_fy0 = 0
        mother_fz0 = 0
        refined_neighbor = false

        if face == 1
            refined_neighbor = has_children_at(gx + 1, gy, gz)
            mother_fx0 = 2 * gx + 1
            mother_fy0 = 2 * gy - 1
            mother_fz0 = 2 * gz - 1
        elseif face == 2
            refined_neighbor = has_children_at(gx - 1, gy, gz)
            mother_fx0 = 2 * gx - 2
            mother_fy0 = 2 * gy - 1
            mother_fz0 = 2 * gz - 1
        elseif face == 3
            refined_neighbor = has_children_at(gx, gy + 1, gz)
            mother_fx0 = 2 * gx - 1
            mother_fy0 = 2 * gy + 1
            mother_fz0 = 2 * gz - 1
        elseif face == 4
            refined_neighbor = has_children_at(gx, gy - 1, gz)
            mother_fx0 = 2 * gx - 1
            mother_fy0 = 2 * gy - 2
            mother_fz0 = 2 * gz - 1
        elseif face == 5
            refined_neighbor = has_children_at(gx, gy, gz + 1)
            mother_fx0 = 2 * gx - 1
            mother_fy0 = 2 * gy - 1
            mother_fz0 = 2 * gz + 1
        else
            refined_neighbor = has_children_at(gx, gy, gz - 1)
            mother_fx0 = 2 * gx - 1
            mother_fy0 = 2 * gy - 1
            mother_fz0 = 2 * gz - 2
        end
        refined_neighbor || return (0.0, 0.0, 0.0, 0.0, 0, false)

        sx_last = face <= 2 ? 0 : 1
        sy_last = (face == 3 || face == 4) ? 0 : 1
        sz_last = face >= 5 ? 0 : 1
        rho_sum = 0.0
        ux_sum = 0.0
        uy_sum = 0.0
        uz_sum = 0.0
        n_face = 0
        for sx in 0:sx_last, sy in 0:sy_last, sz in 0:sz_last
            rho_v, ux_v, uy_v, uz_v, valid = child_cell_state(
                mother_fx0 + sx,
                mother_fy0 + sy,
                mother_fz0 + sz,
            )
            valid || return (0.0, 0.0, 0.0, 0.0, 0, false)
            rho_sum += rho_v
            ux_sum += ux_v
            uy_sum += uy_v
            uz_sum += uz_v
            n_face += 1
        end
        inv_face = 1.0 / max(n_face, 1)
        return (
            inv_face * rho_sum,
            inv_face * ux_sum,
            inv_face * uy_sum,
            inv_face * uz_sum,
            n_face,
            true,
        )
    end

    n_blocks = size(parent_rho_cpu, 4)
    @inbounds for pb in 1:n_blocks
        pbx = Int(parent_map_x_cpu[pb])
        pby = Int(parent_map_y_cpu[pb])
        pbz = Int(parent_map_z_cpu[pb])
        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            parent_obs_cpu[lx, ly, lz, pb] && continue
            gx = (pbx - 1) * bs + lx
            gy = (pby - 1) * bs + ly
            gz = (pbz - 1) * bs + lz
            has_children_at(gx, gy, gz) && continue

            rho_parent = Float64(parent_rho_cpu[lx, ly, lz, pb])
            ux_parent = Float64(parent_vel_cpu[lx, ly, lz, pb, 1])
            uy_parent = Float64(parent_vel_cpu[lx, ly, lz, pb, 2])
            uz_parent = Float64(parent_vel_cpu[lx, ly, lz, pb, 3])

            shell_face_count = 0
            for face_probe in 1:6
                _, _, _, _, _, valid_probe = shell_face_target(face_probe, gx, gy, gz)
                shell_face_count += valid_probe ? 1 : 0
            end
            shell_face_count == 0 && continue

            for face in 1:6
                rho_face, ux_face, uy_face, uz_face, n_child_face_cells, valid =
                    shell_face_target(face, gx, gy, gz)
                valid || continue

                axis = face <= 2 ? 1 : (face <= 4 ? 2 : 3)
                axis_name = axis == 1 ? "x" : (axis == 2 ? "y" : "z")
                normal_sign = (face == 1 || face == 3 || face == 5) ? 1 : -1
                normal_scale = Float64(normal_sign)
                u_parent_axis = axis == 1 ? ux_parent : (axis == 2 ? uy_parent : uz_parent)
                u_face_axis = axis == 1 ? ux_face : (axis == 2 ? uy_face : uz_face)
                un_parent = normal_scale * u_parent_axis
                un_face = normal_scale * u_face_axis
                parent_mass = rho_parent * un_parent
                child_mass = rho_face * un_face
                parent_mom = rho_parent * un_parent * un_parent + cs2 * rho_parent
                child_mom = rho_face * un_face * un_face + cs2 * rho_face
                face_id = @sprintf("L%d_%d_%d_%d_%s_%+d", parent_level_idx, gx, gy, gz, axis_name, normal_sign)
                row = _amr_runtime_budget_row!(
                    parent_level_idx,
                    face_id,
                    pb,
                    lx,
                    ly,
                    lz,
                    gx,
                    gy,
                    gz,
                    axis_name,
                    normal_sign,
                    shell_face_count,
                    n_child_face_cells,
                )
                if side === :parent
                    row.n_parent_samples += 1
                    row.parent_time += dt_weight
                    row.parent_mass_flux_int += dt_weight * parent_mass
                    row.parent_mom_flux_int += dt_weight * parent_mom
                else
                    row.n_child_samples += 1
                    row.child_time += dt_weight
                    row.child_mass_flux_int += dt_weight * child_mass
                    row.child_mom_flux_int += dt_weight * child_mom
                end
                rows_touched += 1
            end
        end
    end

    return rows_touched
end

function apply_amr_runtime_face_budget_reflux!(
    parent_level::BlockLevel,
    child_level::BlockLevel,
    parent_level_idx::Int,
    parent_f_current,
    parent_vel_current,
    cx_arr,
    cy_arr,
    cz_arr,
    w_arr;
    alpha::Float32=0.10f0,
    max_parent_level::Int=1,
    x_only::Bool=true,
)
    alpha <= 0.0f0 && return 0
    parent_level_idx > max_parent_level && return 0

    rows = AMR_RUNTIME_FACE_BUDGET[]
    isempty(rows) && return 0

    rho_cpu = Array(parent_level.rho)
    vel_cpu = Array(parent_vel_current)
    f_cpu = Array(parent_f_current)
    cx_cpu = Array(cx_arr)
    cy_cpu = Array(cy_arr)
    cz_cpu = Array(cz_arr)
    w_cpu = Array(w_arr)

    cs2 = Float64(CS2_PHYSICS)
    compressible_flag = Int32(COMPRESSIBLE_MODE ? 1 : 0)
    shift_x = Float32(LATTICE_SHIFT_X)
    shift_y = Float32(LATTICE_SHIFT_Y)
    shift_z = Float32(LATTICE_SHIFT_Z)
    n_dirs = min(size(f_cpu, 5), length(cx_cpu), length(w_cpu))
    applied = 0

    for row in values(rows)
        row.parent_level == parent_level_idx || continue
        row.shell_face_count == 1 || continue
        row.n_parent_samples > 0 || continue
        row.n_child_samples > 0 || continue
        row.parent_time > 0.0 || continue
        row.child_time > 0.0 || continue
        x_only && row.axis != "x" && continue

        lx = row.parent_lx
        ly = row.parent_ly
        lz = row.parent_lz
        pb = row.parent_block
        (1 <= lx <= size(rho_cpu, 1) && 1 <= ly <= size(rho_cpu, 2) &&
         1 <= lz <= size(rho_cpu, 3) && 1 <= pb <= size(rho_cpu, 4)) || continue

        axis = row.axis == "x" ? 1 : (row.axis == "y" ? 2 : 3)
        normal_sign = Float64(row.normal_sign)
        rho = max(Float64(rho_cpu[lx, ly, lz, pb]), 0.01)
        ux = Float64(vel_cpu[lx, ly, lz, pb, 1])
        uy = Float64(vel_cpu[lx, ly, lz, pb, 2])
        uz = Float64(vel_cpu[lx, ly, lz, pb, 3])
        u_axis = axis == 1 ? ux : (axis == 2 ? uy : uz)
        un = normal_sign * u_axis

        parent_mass_rate = row.parent_mass_flux_int / row.parent_time
        child_mass_rate = row.child_mass_flux_int / row.child_time
        parent_mom_rate = row.parent_mom_flux_int / row.parent_time
        child_mom_rate = row.child_mom_flux_int / row.child_time
        dmass_rate = parent_mass_rate - child_mass_rate
        dmom_rate = parent_mom_rate - child_mom_rate

        u_ref = max(abs(u_axis), 1.0e-3)
        mass_cap = rho * max(Float64(AMR_SHELL_FLUX_DU_CAP_ABS),
                             Float64(AMR_SHELL_FLUX_DU_CAP_REL) * u_ref)
        mom_scale = max(abs(parent_mom_rate), abs(child_mom_rate), cs2 * rho)
        mom_cap = max(Float64(AMR_SHELL_MOMFLUX_STRESS_CAP_ABS),
                      Float64(AMR_SHELL_MOMFLUX_STRESS_CAP_REL) * mom_scale)

        target_dmass = -Float64(alpha) * clamp(dmass_rate, -mass_cap, mass_cap)
        target_dmom = -Float64(alpha) * clamp(dmom_rate, -mom_cap, mom_cap)

        qfac = un * un + cs2
        det = rho * (un * un - cs2)
        abs(det) > 1.0e-14 || continue
        drho = (target_dmass * 2.0 * rho * un - rho * target_dmom) / det
        dun = (un * target_dmom - qfac * target_dmass) / det

        drho_cap = max(Float64(AMR_SHELL_DRHO_CAP_X_ABS),
                       Float64(AMR_SHELL_DRHO_CAP_X_REL) * rho)
        du_cap = max(Float64(AMR_SHELL_DU_CAP_X_ABS),
                     Float64(AMR_SHELL_DU_CAP_X_REL) * u_ref)
        drho = clamp(drho, -drho_cap, drho_cap)
        dun = clamp(dun, -du_cap, du_cap)

        rho_new = max(rho + drho, 0.01)
        u_axis_new = u_axis + normal_sign * dun
        ux_new = axis == 1 ? u_axis_new : ux
        uy_new = axis == 2 ? u_axis_new : uy
        uz_new = axis == 3 ? u_axis_new : uz

        rho32 = Float32(rho)
        rho_new32 = Float32(rho_new)
        ux32 = Float32(ux)
        uy32 = Float32(uy)
        uz32 = Float32(uz)
        ux_new32 = Float32(ux_new)
        uy_new32 = Float32(uy_new)
        uz_new32 = Float32(uz_new)
        @inbounds for k in 1:n_dirs
            cxk = Float32(cx_cpu[k]) - shift_x
            cyk = Float32(cy_cpu[k]) - shift_y
            czk = Float32(cz_cpu[k]) - shift_z
            wk = Float32(w_cpu[k])
            feq_old = calculate_equilibrium_auto(
                rho32, ux32, uy32, uz32, wk, cxk, cyk, czk, compressible_flag
            )
            feq_new = calculate_equilibrium_auto(
                rho_new32, ux_new32, uy_new32, uz_new32, wk, cxk, cyk, czk, compressible_flag
            )
            f_cpu[lx, ly, lz, pb, k] = max(f_cpu[lx, ly, lz, pb, k] + (feq_new - feq_old), 1.0f-10)
        end

        rho_cpu[lx, ly, lz, pb] = rho_new32
        vel_cpu[lx, ly, lz, pb, 1] = ux_new32
        vel_cpu[lx, ly, lz, pb, 2] = uy_new32
        vel_cpu[lx, ly, lz, pb, 3] = uz_new32
        applied += 1
    end

    applied == 0 && return 0
    copyto!(parent_level.rho, rho_cpu)
    copyto!(parent_vel_current, vel_cpu)
    copyto!(parent_f_current, f_cpu)
    @printf("          [AMR-BUDGET] runtime reflux applied=%d L%d->L%d alpha=%.3f\n",
            applied, parent_level_idx, parent_level_idx + 1, Float64(alpha))
    return applied
end

function write_amr_runtime_face_budget(; csv_path::Union{Nothing,String}=nothing, step::Int=-1)
    csv_path === nothing && return
    rows = AMR_RUNTIME_FACE_BUDGET[]
    open(csv_path, "a") do io
        for row in values(rows)
            dmass_int = row.parent_mass_flux_int - row.child_mass_flux_int
            dmom_int = row.parent_mom_flux_int - row.child_mom_flux_int
            denom = max(max(row.parent_time, row.child_time), 1.0e-30)
            dmass_rate = dmass_int / denom
            dmom_rate = dmom_int / denom
            @printf(io,
                "%d,%d,%d,%s,%d,%d,%d,%d,%d,%d,%d,%s,%d,%d,%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",
                step,
                row.parent_level,
                row.child_level,
                row.face_id,
                row.parent_block,
                row.parent_lx,
                row.parent_ly,
                row.parent_lz,
                row.parent_gx,
                row.parent_gy,
                row.parent_gz,
                row.axis,
                row.normal_sign,
                row.shell_face_count,
                row.n_child_face_cells,
                row.n_parent_samples,
                row.n_child_samples,
                row.parent_time,
                row.child_time,
                row.parent_mass_flux_int,
                row.child_mass_flux_int,
                dmass_int,
                row.parent_mom_flux_int,
                row.child_mom_flux_int,
                dmom_int,
                dmass_rate,
                dmom_rate,
                abs(dmass_rate),
                abs(dmom_rate))
        end
    end
    @printf("          [AMR-BUDGET] runtime face rows=%d -> %s\n", length(rows), csv_path)
end

function compute_amr_shell_face_pi_jump_stats(parent_level::BlockLevel, child_level::BlockLevel)
    parent_has_pi = has_pi_neq(parent_level)
    child_has_pi = has_pi_neq(child_level)
    if !(parent_has_pi && child_has_pi)
        return (
            n_faces = 0,
            mean_abs_dpi = 0.0,
            rms_dpi = 0.0,
            max_abs_dpi = 0.0,
            mean_abs_dpi_x = 0.0,
            mean_abs_dpi_y = 0.0,
            mean_abs_dpi_z = 0.0,
            mean_pi_coarse = 0.0,
        )
    end

    obs_cpu = Array(parent_level.obstacle)
    parent_pi_cpu = Array(parent_level.pi_neq)
    parent_map_x_cpu = Array(parent_level.map_x)
    parent_map_y_cpu = Array(parent_level.map_y)
    parent_map_z_cpu = Array(parent_level.map_z)
    child_ptr_cpu = Array(child_level.block_pointer)
    child_pi_cpu = Array(child_level.pi_neq)

    cdx = size(child_ptr_cpu, 1)
    cdy = size(child_ptr_cpu, 2)
    cdz = size(child_ptr_cpu, 3)
    bs = BLOCK_SIZE

    @inline function child_pi_at(fgx::Int, fgy::Int, fgz::Int)
        if fgx < 1 || fgy < 1 || fgz < 1
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false)
        end
        cbx = (fgx - 1) ÷ bs + 1
        cby = (fgy - 1) ÷ bs + 1
        cbz = (fgz - 1) ÷ bs + 1
        if cbx > cdx || cby > cdy || cbz > cdz
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false)
        end
        cb_idx = child_ptr_cpu[cbx, cby, cbz]
        cb_idx == 0 && return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false)
        clx = (fgx - 1) % bs + 1
        cly = (fgy - 1) % bs + 1
        clz = (fgz - 1) % bs + 1
        return (
            Float64(child_pi_cpu[clx, cly, clz, cb_idx, 1]),
            Float64(child_pi_cpu[clx, cly, clz, cb_idx, 2]),
            Float64(child_pi_cpu[clx, cly, clz, cb_idx, 3]),
            Float64(child_pi_cpu[clx, cly, clz, cb_idx, 4]),
            Float64(child_pi_cpu[clx, cly, clz, cb_idx, 5]),
            Float64(child_pi_cpu[clx, cly, clz, cb_idx, 6]),
            true,
        )
    end

    @inline function has_children_at(gx::Int, gy::Int, gz::Int)
        for sx in 0:1, sy in 0:1, sz in 0:1
            fgx = 2 * gx - 1 + sx
            fgy = 2 * gy - 1 + sy
            fgz = 2 * gz - 1 + sz
            _, _, _, _, _, _, valid = child_pi_at(fgx, fgy, fgz)
            valid || return false
        end
        return true
    end

    @inline function shell_face_pi_target(face::Int, gx::Int, gy::Int, gz::Int)
        # Mirror of shell_face_target in compute_amr_shell_face_jump_stats.
        # The 4 child cells on the face of a parent neighbor are selected by
        # collapsing one of (sx, sy, sz) depending on which face we're on.
        mother_fx0 = 0
        mother_fy0 = 0
        mother_fz0 = 0
        refined_neighbor = false

        if face == 1
            refined_neighbor = has_children_at(gx + 1, gy, gz)
            mother_fx0 = 2 * gx + 1
            mother_fy0 = 2 * gy - 1
            mother_fz0 = 2 * gz - 1
        elseif face == 2
            refined_neighbor = has_children_at(gx - 1, gy, gz)
            mother_fx0 = 2 * gx - 2
            mother_fy0 = 2 * gy - 1
            mother_fz0 = 2 * gz - 1
        elseif face == 3
            refined_neighbor = has_children_at(gx, gy + 1, gz)
            mother_fx0 = 2 * gx - 1
            mother_fy0 = 2 * gy + 1
            mother_fz0 = 2 * gz - 1
        elseif face == 4
            refined_neighbor = has_children_at(gx, gy - 1, gz)
            mother_fx0 = 2 * gx - 1
            mother_fy0 = 2 * gy - 2
            mother_fz0 = 2 * gz - 1
        elseif face == 5
            refined_neighbor = has_children_at(gx, gy, gz + 1)
            mother_fx0 = 2 * gx - 1
            mother_fy0 = 2 * gy - 1
            mother_fz0 = 2 * gz + 1
        else
            refined_neighbor = has_children_at(gx, gy, gz - 1)
            mother_fx0 = 2 * gx - 1
            mother_fy0 = 2 * gy - 1
            mother_fz0 = 2 * gz - 2
        end

        refined_neighbor || return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false)

        sx_first = 0; sx_last = 1
        sy_first = 0; sy_last = 1
        sz_first = 0; sz_last = 1
        if face <= 2
            sx_last = 0
        elseif face <= 4
            sy_last = 0
        else
            sz_last = 0
        end

        pxx_sum = 0.0; pyy_sum = 0.0; pzz_sum = 0.0
        pxy_sum = 0.0; pxz_sum = 0.0; pyz_sum = 0.0
        n_face = 0
        for sx in sx_first:sx_last, sy in sy_first:sy_last, sz in sz_first:sz_last
            pxx, pyy, pzz, pxy, pxz, pyz, valid = child_pi_at(
                mother_fx0 + sx, mother_fy0 + sy, mother_fz0 + sz
            )
            valid || return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false)
            pxx_sum += pxx; pyy_sum += pyy; pzz_sum += pzz
            pxy_sum += pxy; pxz_sum += pxz; pyz_sum += pyz
            n_face += 1
        end

        inv_face = 1.0 / max(n_face, 1)
        return (inv_face * pxx_sum, inv_face * pyy_sum, inv_face * pzz_sum,
                inv_face * pxy_sum, inv_face * pxz_sum, inv_face * pyz_sum, true)
    end

    n_faces = 0
    abs_sum = 0.0
    sq_sum = 0.0
    max_abs = 0.0
    pi_mag_coarse_sum = 0.0

    n_faces_x = 0; n_faces_y = 0; n_faces_z = 0
    abs_sum_x = 0.0; abs_sum_y = 0.0; abs_sum_z = 0.0

    n_blocks = size(parent_pi_cpu, 4)
    @inbounds for pb in 1:n_blocks
        pbx = Int(parent_map_x_cpu[pb])
        pby = Int(parent_map_y_cpu[pb])
        pbz = Int(parent_map_z_cpu[pb])
        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            obs_cpu[lx, ly, lz, pb] && continue
            gx = (pbx - 1) * bs + lx
            gy = (pby - 1) * bs + ly
            gz = (pbz - 1) * bs + lz
            has_children_at(gx, gy, gz) && continue

            pxx_c = Float64(parent_pi_cpu[lx, ly, lz, pb, 1])
            pyy_c = Float64(parent_pi_cpu[lx, ly, lz, pb, 2])
            pzz_c = Float64(parent_pi_cpu[lx, ly, lz, pb, 3])
            pxy_c = Float64(parent_pi_cpu[lx, ly, lz, pb, 4])
            pxz_c = Float64(parent_pi_cpu[lx, ly, lz, pb, 5])
            pyz_c = Float64(parent_pi_cpu[lx, ly, lz, pb, 6])
            pi_mag_coarse = sqrt(pxx_c^2 + pyy_c^2 + pzz_c^2 +
                                 2.0 * (pxy_c^2 + pxz_c^2 + pyz_c^2))

            for face in 1:6
                pxx_f, pyy_f, pzz_f, pxy_f, pxz_f, pyz_f, valid =
                    shell_face_pi_target(face, gx, gy, gz)
                valid || continue

                dpxx = pxx_c - pxx_f
                dpyy = pyy_c - pyy_f
                dpzz = pzz_c - pzz_f
                dpxy = pxy_c - pxy_f
                dpxz = pxz_c - pxz_f
                dpyz = pyz_c - pyz_f
                dpi2 = dpxx^2 + dpyy^2 + dpzz^2 +
                       2.0 * (dpxy^2 + dpxz^2 + dpyz^2)
                dpi  = sqrt(dpi2)

                n_faces += 1
                abs_sum += dpi
                sq_sum += dpi2
                max_abs = max(max_abs, dpi)
                pi_mag_coarse_sum += pi_mag_coarse

                if face <= 2
                    n_faces_x += 1
                    abs_sum_x += dpi
                elseif face <= 4
                    n_faces_y += 1
                    abs_sum_y += dpi
                else
                    n_faces_z += 1
                    abs_sum_z += dpi
                end
            end
        end
    end

    return (
        n_faces = n_faces,
        mean_abs_dpi = n_faces > 0 ? abs_sum / n_faces : 0.0,
        rms_dpi = n_faces > 0 ? sqrt(sq_sum / n_faces) : 0.0,
        max_abs_dpi = max_abs,
        mean_abs_dpi_x = n_faces_x > 0 ? abs_sum_x / n_faces_x : 0.0,
        mean_abs_dpi_y = n_faces_y > 0 ? abs_sum_y / n_faces_y : 0.0,
        mean_abs_dpi_z = n_faces_z > 0 ? abs_sum_z / n_faces_z : 0.0,
        mean_pi_coarse = n_faces > 0 ? pi_mag_coarse_sum / n_faces : 0.0,
    )
end

# Conservative fine-to-coarse audit.
#
# For every parent cell fully covered by 8 child cells, compare the parent
# conserved moments against the volume average of those children:
#
#   drho = rho_parent - <rho_child>
#   dj   = rho_parent*u_parent - <rho_child*u_child>
#
# This is not a face-continuity metric. It directly tests whether restriction
# has made the parent state conservative with respect to the fine state that
# replaces it in the active AMR domain.
function compute_amr_restriction_conservation_stats(parent_level::BlockLevel, child_level::BlockLevel)
    rho_cpu = Array(parent_level.rho)
    vel_cpu = Array(parent_level.vel)
    obs_cpu = Array(parent_level.obstacle)
    parent_map_x_cpu = Array(parent_level.map_x)
    parent_map_y_cpu = Array(parent_level.map_y)
    parent_map_z_cpu = Array(parent_level.map_z)

    child_ptr_cpu = Array(child_level.block_pointer)
    child_obs_cpu = Array(child_level.obstacle)
    child_rho_cpu = Array(child_level.rho)
    child_vel_cpu = Array(child_level.vel)

    cdx = size(child_ptr_cpu, 1)
    cdy = size(child_ptr_cpu, 2)
    cdz = size(child_ptr_cpu, 3)
    bs = BLOCK_SIZE

    @inline function child_state(fgx::Int, fgy::Int, fgz::Int)
        if fgx < 1 || fgy < 1 || fgz < 1
            return (0.0, 0.0, 0.0, 0.0, false)
        end
        cbx = (fgx - 1) ÷ bs + 1
        cby = (fgy - 1) ÷ bs + 1
        cbz = (fgz - 1) ÷ bs + 1
        if cbx > cdx || cby > cdy || cbz > cdz
            return (0.0, 0.0, 0.0, 0.0, false)
        end
        cb_idx = child_ptr_cpu[cbx, cby, cbz]
        cb_idx == 0 && return (0.0, 0.0, 0.0, 0.0, false)
        clx = (fgx - 1) % bs + 1
        cly = (fgy - 1) % bs + 1
        clz = (fgz - 1) % bs + 1
        child_obs_cpu[clx, cly, clz, cb_idx] && return (0.0, 0.0, 0.0, 0.0, false)
        rho = Float64(child_rho_cpu[clx, cly, clz, cb_idx])
        ux = Float64(child_vel_cpu[clx, cly, clz, cb_idx, 1])
        uy = Float64(child_vel_cpu[clx, cly, clz, cb_idx, 2])
        uz = Float64(child_vel_cpu[clx, cly, clz, cb_idx, 3])
        return (rho, ux, uy, uz, true)
    end

    n_cells = 0
    abs_drho_sum = 0.0
    sq_drho_sum = 0.0
    max_abs_drho = 0.0
    drho_sum = 0.0

    abs_dj_sum = 0.0
    sq_dj_sum = 0.0
    max_abs_dj = 0.0
    djx_sum = 0.0
    djy_sum = 0.0
    djz_sum = 0.0

    n_blocks = size(rho_cpu, 4)
    @inbounds for pb in 1:n_blocks
        pbx = Int(parent_map_x_cpu[pb])
        pby = Int(parent_map_y_cpu[pb])
        pbz = Int(parent_map_z_cpu[pb])
        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            obs_cpu[lx, ly, lz, pb] && continue
            gx = (pbx - 1) * bs + lx
            gy = (pby - 1) * bs + ly
            gz = (pbz - 1) * bs + lz

            rho_child_sum = 0.0
            jx_child_sum = 0.0
            jy_child_sum = 0.0
            jz_child_sum = 0.0
            valid_children = 0
            all_valid = true
            for sx in 0:1, sy in 0:1, sz in 0:1
                fgx = 2 * gx - 1 + sx
                fgy = 2 * gy - 1 + sy
                fgz = 2 * gz - 1 + sz
                rho_c, ux_c, uy_c, uz_c, valid = child_state(fgx, fgy, fgz)
                if !valid
                    all_valid = false
                    continue
                end
                rho_child_sum += rho_c
                jx_child_sum += rho_c * ux_c
                jy_child_sum += rho_c * uy_c
                jz_child_sum += rho_c * uz_c
                valid_children += 1
            end
            (all_valid && valid_children == 8) || continue

            inv_children = 1.0 / 8.0
            rho_child = rho_child_sum * inv_children
            jx_child = jx_child_sum * inv_children
            jy_child = jy_child_sum * inv_children
            jz_child = jz_child_sum * inv_children

            rho_parent = Float64(rho_cpu[lx, ly, lz, pb])
            ux_parent = Float64(vel_cpu[lx, ly, lz, pb, 1])
            uy_parent = Float64(vel_cpu[lx, ly, lz, pb, 2])
            uz_parent = Float64(vel_cpu[lx, ly, lz, pb, 3])
            jx_parent = rho_parent * ux_parent
            jy_parent = rho_parent * uy_parent
            jz_parent = rho_parent * uz_parent

            drho = rho_parent - rho_child
            djx = jx_parent - jx_child
            djy = jy_parent - jy_child
            djz = jz_parent - jz_child
            abs_drho = abs(drho)
            abs_dj = sqrt(djx * djx + djy * djy + djz * djz)

            n_cells += 1
            drho_sum += drho
            abs_drho_sum += abs_drho
            sq_drho_sum += drho * drho
            max_abs_drho = max(max_abs_drho, abs_drho)
            djx_sum += djx
            djy_sum += djy
            djz_sum += djz
            abs_dj_sum += abs_dj
            sq_dj_sum += abs_dj * abs_dj
            max_abs_dj = max(max_abs_dj, abs_dj)
        end
    end

    return (
        n_cells = n_cells,
        mean_abs_drho = n_cells > 0 ? abs_drho_sum / n_cells : 0.0,
        rms_drho = n_cells > 0 ? sqrt(sq_drho_sum / n_cells) : 0.0,
        max_abs_drho = max_abs_drho,
        mean_drho = n_cells > 0 ? drho_sum / n_cells : 0.0,
        mean_abs_dj = n_cells > 0 ? abs_dj_sum / n_cells : 0.0,
        rms_dj = n_cells > 0 ? sqrt(sq_dj_sum / n_cells) : 0.0,
        max_abs_dj = max_abs_dj,
        mean_djx = n_cells > 0 ? djx_sum / n_cells : 0.0,
        mean_djy = n_cells > 0 ? djy_sum / n_cells : 0.0,
        mean_djz = n_cells > 0 ? djz_sum / n_cells : 0.0,
    )
end

function compute_amr_partial_frontier_stats(parent_level::BlockLevel, child_level::BlockLevel)
    rho_cpu = Array(parent_level.rho)
    vel_cpu = Array(parent_level.vel)
    obs_cpu = Array(parent_level.obstacle)
    parent_map_x_cpu = Array(parent_level.map_x)
    parent_map_y_cpu = Array(parent_level.map_y)
    parent_map_z_cpu = Array(parent_level.map_z)
    child_ptr_cpu = Array(child_level.block_pointer)
    child_rho_cpu = Array(child_level.rho)
    child_vel_cpu = Array(child_level.vel)

    bs = BLOCK_SIZE
    child_dim_x = size(child_ptr_cpu, 1)
    child_dim_y = size(child_ptr_cpu, 2)
    child_dim_z = size(child_ptr_cpu, 3)

    @inline function child_cell_state(fgx::Int, fgy::Int, fgz::Int)
        if fgx < 1 || fgy < 1 || fgz < 1
            return (0.0, 0.0, 0.0, 0.0, false)
        end
        cbx = (fgx - 1) ÷ bs + 1
        cby = (fgy - 1) ÷ bs + 1
        cbz = (fgz - 1) ÷ bs + 1
        if cbx > child_dim_x || cby > child_dim_y || cbz > child_dim_z
            return (0.0, 0.0, 0.0, 0.0, false)
        end
        cb_idx = child_ptr_cpu[cbx, cby, cbz]
        cb_idx == 0 && return (0.0, 0.0, 0.0, 0.0, false)
        clx = (fgx - 1) % bs + 1
        cly = (fgy - 1) % bs + 1
        clz = (fgz - 1) % bs + 1
        return (
            Float64(child_rho_cpu[clx, cly, clz, cb_idx]),
            Float64(child_vel_cpu[clx, cly, clz, cb_idx, 1]),
            Float64(child_vel_cpu[clx, cly, clz, cb_idx, 2]),
            Float64(child_vel_cpu[clx, cly, clz, cb_idx, 3]),
            true
        )
    end

    n_cells = 0
    abs_drho_sum = 0.0
    abs_du_sum = 0.0
    max_abs_drho = 0.0
    max_abs_du = 0.0

    n_cells_x = 0
    n_cells_y = 0
    n_cells_z = 0
    abs_drho_sum_x = 0.0
    abs_drho_sum_y = 0.0
    abs_drho_sum_z = 0.0
    n_cells_x_neg = 0
    n_cells_x_pos = 0
    n_cells_y_neg = 0
    n_cells_y_pos = 0
    n_cells_z_neg = 0
    n_cells_z_pos = 0
    abs_drho_sum_x_neg = 0.0
    abs_drho_sum_x_pos = 0.0
    abs_drho_sum_y_neg = 0.0
    abs_drho_sum_y_pos = 0.0
    abs_drho_sum_z_neg = 0.0
    abs_drho_sum_z_pos = 0.0

    n_blocks = size(rho_cpu, 4)
    @inbounds for pb in 1:n_blocks
        pbx = Int(parent_map_x_cpu[pb])
        pby = Int(parent_map_y_cpu[pb])
        pbz = Int(parent_map_z_cpu[pb])
        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            obs_cpu[lx, ly, lz, pb] && continue

            gx = (pbx - 1) * bs + lx
            gy = (pby - 1) * bs + ly
            gz = (pbz - 1) * bs + lz

            rho_sum = 0.0
            ux_sum = 0.0
            uy_sum = 0.0
            uz_sum = 0.0
            valid_children = 0

            has_x_neg = false
            has_x_pos = false
            has_y_neg = false
            has_y_pos = false
            has_z_neg = false
            has_z_pos = false

            for sx in 0:1, sy in 0:1, sz in 0:1
                fgx = 2 * gx - 1 + sx
                fgy = 2 * gy - 1 + sy
                fgz = 2 * gz - 1 + sz
                rho_v, ux_v, uy_v, uz_v, valid = child_cell_state(fgx, fgy, fgz)
                valid || continue
                valid_children += 1
                rho_sum += rho_v
                ux_sum += ux_v
                uy_sum += uy_v
                uz_sum += uz_v

                has_x_neg |= (sx == 0)
                has_x_pos |= (sx == 1)
                has_y_neg |= (sy == 0)
                has_y_pos |= (sy == 1)
                has_z_neg |= (sz == 0)
                has_z_pos |= (sz == 1)
            end

            (valid_children == 0 || valid_children == 8) && continue

            inv_valid = 1.0 / valid_children
            rho_child = rho_sum * inv_valid
            ux_child = ux_sum * inv_valid
            uy_child = uy_sum * inv_valid
            uz_child = uz_sum * inv_valid

            rho_coarse = Float64(rho_cpu[lx, ly, lz, pb])
            ux_coarse = Float64(vel_cpu[lx, ly, lz, pb, 1])
            uy_coarse = Float64(vel_cpu[lx, ly, lz, pb, 2])
            uz_coarse = Float64(vel_cpu[lx, ly, lz, pb, 3])

            drho = rho_coarse - rho_child
            dux = ux_coarse - ux_child
            duy = uy_coarse - uy_child
            duz = uz_coarse - uz_child
            abs_drho = abs(drho)
            abs_du = sqrt(dux * dux + duy * duy + duz * duz)

            n_cells += 1
            abs_drho_sum += abs_drho
            abs_du_sum += abs_du
            max_abs_drho = max(max_abs_drho, abs_drho)
            max_abs_du = max(max_abs_du, abs_du)

            if xor(has_x_neg, has_x_pos)
                n_cells_x += 1
                abs_drho_sum_x += abs_drho
            end
            if xor(has_y_neg, has_y_pos)
                n_cells_y += 1
                abs_drho_sum_y += abs_drho
            end
            if xor(has_z_neg, has_z_pos)
                n_cells_z += 1
                abs_drho_sum_z += abs_drho
            end
            if !has_x_neg
                n_cells_x_neg += 1
                abs_drho_sum_x_neg += abs_drho
            end
            if !has_x_pos
                n_cells_x_pos += 1
                abs_drho_sum_x_pos += abs_drho
            end
            if !has_y_neg
                n_cells_y_neg += 1
                abs_drho_sum_y_neg += abs_drho
            end
            if !has_y_pos
                n_cells_y_pos += 1
                abs_drho_sum_y_pos += abs_drho
            end
            if !has_z_neg
                n_cells_z_neg += 1
                abs_drho_sum_z_neg += abs_drho
            end
            if !has_z_pos
                n_cells_z_pos += 1
                abs_drho_sum_z_pos += abs_drho
            end
        end
    end

    return (
        n_cells = n_cells,
        mean_abs_drho = n_cells > 0 ? abs_drho_sum / n_cells : 0.0,
        max_abs_drho = max_abs_drho,
        mean_abs_du = n_cells > 0 ? abs_du_sum / n_cells : 0.0,
        max_abs_du = max_abs_du,
        n_cells_x = n_cells_x,
        n_cells_y = n_cells_y,
        n_cells_z = n_cells_z,
        mean_abs_drho_x = n_cells_x > 0 ? abs_drho_sum_x / n_cells_x : 0.0,
        mean_abs_drho_y = n_cells_y > 0 ? abs_drho_sum_y / n_cells_y : 0.0,
        mean_abs_drho_z = n_cells_z > 0 ? abs_drho_sum_z / n_cells_z : 0.0,
        mean_abs_drho_x_neg = n_cells_x_neg > 0 ? abs_drho_sum_x_neg / n_cells_x_neg : 0.0,
        mean_abs_drho_x_pos = n_cells_x_pos > 0 ? abs_drho_sum_x_pos / n_cells_x_pos : 0.0,
        mean_abs_drho_y_neg = n_cells_y_neg > 0 ? abs_drho_sum_y_neg / n_cells_y_neg : 0.0,
        mean_abs_drho_y_pos = n_cells_y_pos > 0 ? abs_drho_sum_y_pos / n_cells_y_pos : 0.0,
        mean_abs_drho_z_neg = n_cells_z_neg > 0 ? abs_drho_sum_z_neg / n_cells_z_neg : 0.0,
        mean_abs_drho_z_pos = n_cells_z_pos > 0 ? abs_drho_sum_z_pos / n_cells_z_pos : 0.0,
    )
end

function compute_amr_prolongation_topology_stats(parent_level::BlockLevel, child_level::BlockLevel)
    child_obs_cpu = Array(child_level.obstacle)
    child_ptr_cpu = Array(child_level.block_pointer)
    child_map_x_cpu = Array(child_level.map_x)
    child_map_y_cpu = Array(child_level.map_y)
    child_map_z_cpu = Array(child_level.map_z)
    parent_ptr_cpu = Array(parent_level.block_pointer)

    bs = BLOCK_SIZE
    parent_dim_x = size(parent_ptr_cpu, 1)
    parent_dim_y = size(parent_ptr_cpu, 2)
    parent_dim_z = size(parent_ptr_cpu, 3)
    child_dim_x = size(child_ptr_cpu, 1)
    child_dim_y = size(child_ptr_cpu, 2)
    child_dim_z = size(child_ptr_cpu, 3)
    parent_nx = parent_dim_x * bs
    parent_ny = parent_dim_y * bs
    parent_nz = parent_dim_z * bs
    child_nx = child_dim_x * bs
    child_ny = child_dim_y * bs
    child_nz = child_dim_z * bs
    n_dirs = LATTICE_TYPE == :d3q39 ? 39 : 27

    @inline function parent_sample_valid(pgx::Int, pgy::Int, pgz::Int)
        pbx = (pgx - 1) ÷ bs + 1
        pby = (pgy - 1) ÷ bs + 1
        pbz = (pgz - 1) ÷ bs + 1
        if pbx < 1 || pbx > parent_dim_x ||
           pby < 1 || pby > parent_dim_y ||
           pbz < 1 || pbz > parent_dim_z
            return false
        end
        return parent_ptr_cpu[pbx, pby, pbz] > 0
    end

    @inline function child_block_active(bx::Int, by::Int, bz::Int)
        if bx < 1 || bx > child_dim_x ||
           by < 1 || by > child_dim_y ||
           bz < 1 || bz > child_dim_z
            return false
        end
        return child_ptr_cpu[bx, by, bz] > 0
    end

    n_frontier_cells = 0
    n_parent_pulls = 0
    n_incomplete_cubic = 0
    n_anchor_invalid = 0
    n_incomplete_x = 0
    n_incomplete_y = 0
    n_incomplete_z = 0
    n_incomplete_x_neg = 0
    n_incomplete_x_pos = 0
    n_incomplete_y_neg = 0
    n_incomplete_y_pos = 0
    n_incomplete_z_neg = 0
    n_incomplete_z_pos = 0

    n_blocks = length(child_level.active_block_coords)
    @inbounds for cb in 1:n_blocks
        cbx = Int(child_map_x_cpu[cb])
        cby = Int(child_map_y_cpu[cb])
        cbz = Int(child_map_z_cpu[cb])

        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            child_obs_cpu[lx, ly, lz, cb] && continue

            cell_has_parent_pull = false
            gx = (cbx - 1) * bs + lx
            gy = (cby - 1) * bs + ly
            gz = (cbz - 1) * bs + lz

            for k in 1:n_dirs
                sx = lx - Int(C_X[k])
                sy = ly - Int(C_Y[k])
                sz = lz - Int(C_Z[k])

                if sx >= 1 && sx <= bs && sy >= 1 && sy <= bs && sz >= 1 && sz <= bs
                    continue
                end

                nb_off_x = sx < 1 ? -1 : (sx > bs ? 1 : 0)
                nb_off_y = sy < 1 ? -1 : (sy > bs ? 1 : 0)
                nb_off_z = sz < 1 ? -1 : (sz > bs ? 1 : 0)
                if child_block_active(cbx + nb_off_x, cby + nb_off_y, cbz + nb_off_z)
                    continue
                end

                src_gx = gx - Int(C_X[k])
                src_gy = gy - Int(C_Y[k])
                src_gz = gz - Int(C_Z[k])
                if src_gx < 1 || src_gx > child_nx ||
                   src_gy < 1 || src_gy > child_ny ||
                   src_gz < 1 || src_gz > child_nz
                    continue
                end

                cell_has_parent_pull = true
                n_parent_pulls += 1

                px_cont = _fine_to_parent_interp_coord_diag(src_gx)
                py_cont = _fine_to_parent_interp_coord_diag(src_gy)
                pz_cont = _fine_to_parent_interp_coord_diag(src_gz)
                px0, _, _ = _parent_interp_bracket_diag(px_cont, parent_nx)
                py0, _, _ = _parent_interp_bracket_diag(py_cont, parent_ny)
                pz0, _, _ = _parent_interp_bracket_diag(pz_cont, parent_nz)

                if !parent_sample_valid(px0, py0, pz0)
                    n_anchor_invalid += 1
                end

                incomplete = false
                bad_x = false
                bad_y = false
                bad_z = false
                bad_x_neg = false
                bad_x_pos = false
                bad_y_neg = false
                bad_y_pos = false
                bad_z_neg = false
                bad_z_pos = false
                for ix in -1:2, iy in -1:2, iz in -1:2
                    if !parent_sample_valid(px0 + ix, py0 + iy, pz0 + iz)
                        incomplete = true
                        bad_x |= (ix == -1 || ix == 2)
                        bad_y |= (iy == -1 || iy == 2)
                        bad_z |= (iz == -1 || iz == 2)
                        bad_x_neg |= (ix == -1)
                        bad_x_pos |= (ix == 2)
                        bad_y_neg |= (iy == -1)
                        bad_y_pos |= (iy == 2)
                        bad_z_neg |= (iz == -1)
                        bad_z_pos |= (iz == 2)
                    end
                end
                if incomplete
                    n_incomplete_cubic += 1
                    n_incomplete_x += bad_x ? 1 : 0
                    n_incomplete_y += bad_y ? 1 : 0
                    n_incomplete_z += bad_z ? 1 : 0
                    n_incomplete_x_neg += bad_x_neg ? 1 : 0
                    n_incomplete_x_pos += bad_x_pos ? 1 : 0
                    n_incomplete_y_neg += bad_y_neg ? 1 : 0
                    n_incomplete_y_pos += bad_y_pos ? 1 : 0
                    n_incomplete_z_neg += bad_z_neg ? 1 : 0
                    n_incomplete_z_pos += bad_z_pos ? 1 : 0
                end
            end

            if cell_has_parent_pull
                n_frontier_cells += 1
            end
        end
    end

    # Fallback/topological audit: some AMR paths fill directional halos through
    # macro/equilibrium reconstruction before this CPU-side q-direction detector
    # can infer a parent pull.  If that precise detector sees nothing, still
    # audit the actual coarse/fine frontier faces directly.  The counts then
    # represent face ghost samples rather than individual q-direction pulls,
    # but the cubic-completeness signal is the one we need for AMR development.
    if n_parent_pulls == 0
        frontier_cells = Set{Tuple{Int,Int,Int}}()

        @inline function audit_face_sample(src_gx::Int, src_gy::Int, src_gz::Int)
            if src_gx < 1 || src_gx > child_nx ||
               src_gy < 1 || src_gy > child_ny ||
               src_gz < 1 || src_gz > child_nz
                return
            end
            push!(frontier_cells, (src_gx, src_gy, src_gz))
            n_parent_pulls += 1

            px_cont = _fine_to_parent_interp_coord_diag(src_gx)
            py_cont = _fine_to_parent_interp_coord_diag(src_gy)
            pz_cont = _fine_to_parent_interp_coord_diag(src_gz)
            px0, _, _ = _parent_interp_bracket_diag(px_cont, parent_nx)
            py0, _, _ = _parent_interp_bracket_diag(py_cont, parent_ny)
            pz0, _, _ = _parent_interp_bracket_diag(pz_cont, parent_nz)

            if !parent_sample_valid(px0, py0, pz0)
                n_anchor_invalid += 1
            end

            incomplete = false
            bad_x = false; bad_y = false; bad_z = false
            bad_x_neg = false; bad_x_pos = false
            bad_y_neg = false; bad_y_pos = false
            bad_z_neg = false; bad_z_pos = false
            for ix in -1:2, iy in -1:2, iz in -1:2
                if !parent_sample_valid(px0 + ix, py0 + iy, pz0 + iz)
                    incomplete = true
                    bad_x |= (ix == -1 || ix == 2)
                    bad_y |= (iy == -1 || iy == 2)
                    bad_z |= (iz == -1 || iz == 2)
                    bad_x_neg |= (ix == -1)
                    bad_x_pos |= (ix == 2)
                    bad_y_neg |= (iy == -1)
                    bad_y_pos |= (iy == 2)
                    bad_z_neg |= (iz == -1)
                    bad_z_pos |= (iz == 2)
                end
            end
            if incomplete
                n_incomplete_cubic += 1
                n_incomplete_x += bad_x ? 1 : 0
                n_incomplete_y += bad_y ? 1 : 0
                n_incomplete_z += bad_z ? 1 : 0
                n_incomplete_x_neg += bad_x_neg ? 1 : 0
                n_incomplete_x_pos += bad_x_pos ? 1 : 0
                n_incomplete_y_neg += bad_y_neg ? 1 : 0
                n_incomplete_y_pos += bad_y_pos ? 1 : 0
                n_incomplete_z_neg += bad_z_neg ? 1 : 0
                n_incomplete_z_pos += bad_z_pos ? 1 : 0
            end
        end

        @inbounds for cb in 1:n_blocks
            cbx = Int(child_map_x_cpu[cb])
            cby = Int(child_map_y_cpu[cb])
            cbz = Int(child_map_z_cpu[cb])

            x0 = (cbx - 1) * bs
            y0 = (cby - 1) * bs
            z0 = (cbz - 1) * bs

            if !child_block_active(cbx - 1, cby, cbz)
                for lz in 1:bs, ly in 1:bs
                    audit_face_sample(x0, y0 + ly, z0 + lz)
                end
            end
            if !child_block_active(cbx + 1, cby, cbz)
                for lz in 1:bs, ly in 1:bs
                    audit_face_sample(x0 + bs + 1, y0 + ly, z0 + lz)
                end
            end
            if !child_block_active(cbx, cby - 1, cbz)
                for lz in 1:bs, lx in 1:bs
                    audit_face_sample(x0 + lx, y0, z0 + lz)
                end
            end
            if !child_block_active(cbx, cby + 1, cbz)
                for lz in 1:bs, lx in 1:bs
                    audit_face_sample(x0 + lx, y0 + bs + 1, z0 + lz)
                end
            end
            if !child_block_active(cbx, cby, cbz - 1)
                for ly in 1:bs, lx in 1:bs
                    audit_face_sample(x0 + lx, y0 + ly, z0)
                end
            end
            if !child_block_active(cbx, cby, cbz + 1)
                for ly in 1:bs, lx in 1:bs
                    audit_face_sample(x0 + lx, y0 + ly, z0 + bs + 1)
                end
            end
        end

        n_frontier_cells = length(frontier_cells)
    end

    return (
        n_frontier_cells = n_frontier_cells,
        n_parent_pulls = n_parent_pulls,
        n_incomplete_cubic = n_incomplete_cubic,
        n_anchor_invalid = n_anchor_invalid,
        n_incomplete_x = n_incomplete_x,
        n_incomplete_y = n_incomplete_y,
        n_incomplete_z = n_incomplete_z,
        n_incomplete_x_neg = n_incomplete_x_neg,
        n_incomplete_x_pos = n_incomplete_x_pos,
        n_incomplete_y_neg = n_incomplete_y_neg,
        n_incomplete_y_pos = n_incomplete_y_pos,
        n_incomplete_z_neg = n_incomplete_z_neg,
        n_incomplete_z_pos = n_incomplete_z_pos
    )
end

# ────────────────────────────────────────────────────────────────────
# Mass conservation diagnostic (Phase 1.4 paper-quality)
# ────────────────────────────────────────────────────────────────────
# Sums ρ·V over every ACTIVE non-obstacle non-covered cell across all levels.
# "Non-covered" = not replaced by a finer-level block. Physical volume per cell
# scales with (dx_level)³ so the sum is a true integral not a count. For a
# closed domain the total should change only by inlet/outlet flux + sponge; a
# systematic drift indicates a bug in the restriction / prolongation operators.
#
# Implementation: coarse level iterates over active blocks, skipping cells
# whose 2×2×2 child footprint is fully refined. Fine levels include all their
# active non-obstacle cells. Returns the physical-units total mass (kg if
# rho is in kg/m³, but we treat ρ as lattice-units ρ · V_phys per level; the
# scale is arbitrary — what matters is the *drift over time*).
function compute_total_active_mass(grids)
    n = length(grids)
    total = 0.0
    @inbounds for lvl in 1:n
        level = grids[lvl]
        length(level.active_block_coords) == 0 && continue
        rho_cpu = Array(level.rho)
        obs_cpu = Array(level.obstacle)
        mapx    = Array(level.map_x)
        mapy    = Array(level.map_y)
        mapz    = Array(level.map_z)
        bs  = BLOCK_SIZE
        vol_cell = level.dx^3   # physical cell volume for this level

        # If a finer level exists, build its block pointer for "is this
        # coarse cell covered?" check. Else every cell counts.
        has_finer = lvl < n && length(grids[lvl+1].active_block_coords) > 0
        child_ptr = has_finer ? Array(grids[lvl+1].block_pointer) : nothing
        cdx = has_finer ? size(child_ptr, 1) : 0
        cdy = has_finer ? size(child_ptr, 2) : 0
        cdz = has_finer ? size(child_ptr, 3) : 0

        @inline function covered_by_children(gx::Int, gy::Int, gz::Int)
            has_finer || return false
            for sx in 0:1, sy in 0:1, sz in 0:1
                fgx = 2*gx - 1 + sx
                fgy = 2*gy - 1 + sy
                fgz = 2*gz - 1 + sz
                (fgx < 1 || fgy < 1 || fgz < 1) && return false
                cbx = (fgx - 1) ÷ bs + 1
                cby = (fgy - 1) ÷ bs + 1
                cbz = (fgz - 1) ÷ bs + 1
                (cbx > cdx || cby > cdy || cbz > cdz) && return false
                child_ptr[cbx, cby, cbz] == 0 && return false
            end
            true
        end

        n_blocks = length(level.active_block_coords)
        level_sum = 0.0
        for pb in 1:n_blocks
            pbx = Int(mapx[pb]); pby = Int(mapy[pb]); pbz = Int(mapz[pb])
            for lz in 1:bs, ly in 1:bs, lx in 1:bs
                obs_cpu[lx, ly, lz, pb] && continue
                gx = (pbx - 1)*bs + lx
                gy = (pby - 1)*bs + ly
                gz = (pbz - 1)*bs + lz
                covered_by_children(gx, gy, gz) && continue
                level_sum += Float64(rho_cpu[lx, ly, lz, pb])
            end
        end
        total += level_sum * vol_cell
    end
    return total
end

# Stateful drift tracker (one module-level reference avoids threading a state
# struct through main.jl). First call seeds the baseline; subsequent calls
# report drift. Designed to be ~cheap enough to call at every DIAG_FREQ.
const _MASS_BASELINE = Ref{Float64}(0.0)
const _MASS_FIRST    = Ref{Bool}(true)

function print_mass_conservation(grids, step::Int)
    mass = compute_total_active_mass(grids)
    if _MASS_FIRST[]
        _MASS_BASELINE[] = mass
        _MASS_FIRST[]    = false
        @printf("          [Mass] step=%d  total=%.6e  (baseline)\n", step, mass)
        return
    end
    drift = mass - _MASS_BASELINE[]
    rel   = _MASS_BASELINE[] > 1e-12 ? drift / _MASS_BASELINE[] : 0.0
    @printf("          [Mass] step=%d  total=%.6e  Δ=%+.3e  (%.3e%% of baseline)\n",
            step, mass, drift, 100.0 * rel)
end

function _diag_percentile!(vals::Vector{Float64}, pct::Float64)
    isempty(vals) && return 0.0
    sort!(vals)
    idx = clamp(ceil(Int, pct * length(vals)), 1, length(vals))
    return vals[idx]
end

function _diag_mean(vals::Vector{Float64})
    isempty(vals) && return 0.0
    s = 0.0
    @inbounds for v in vals
        s += v
    end
    return s / length(vals)
end

"""
    compute_level_native_highpass_stats(level, child_level=nothing; band_depth=3)

Measure the raw velocity-magnitude high-pass on a native 3D AMR level before
2D slice compositing. The high-pass is `abs(|u| - mean_6(|u_neighbor|))` using
a complete same-level, non-obstacle, non-covered six-neighbor stencil.

The categories separate likely causes of rectangular slice artifacts:
- `fine_boundary`: cells on a fine level within `band_depth` cells of an
  inactive same-level block face.
- `coarse_shell`: parent-level cells within `band_depth` cells of child
  coverage.
- `interior`: neither of the above.
"""
function compute_level_native_highpass_stats(level::BlockLevel,
                                             child_level::Union{Nothing,BlockLevel}=nothing;
                                             band_depth::Int=3)
    vel_cpu = Array(level.vel)
    obs_cpu = Array(level.obstacle)
    ptr_cpu = Array(level.block_pointer)
    map_x_cpu = Array(level.map_x)
    map_y_cpu = Array(level.map_y)
    map_z_cpu = Array(level.map_z)
    # s-mode Nyquist proxy: |Π_neq|. h-mode contributes zero to Π_neq by
    # mirror parity (c_α c_β is even in c, h is odd), so a Nyquist mode in
    # |Π_neq| isolates the mirror-symmetric stress content. Falls back to a
    # zero stub when pi_neq isn't allocated (e.g. BGK without stress transfer)
    # — the pi_* columns then report 0 for all regions.
    has_pi = has_pi_neq(level)
    pi_cpu = has_pi ? Array(level.pi_neq) : Array{Float32}(undef, 0, 0, 0, 0, 0)

    child_ptr_cpu = child_level === nothing ? nothing : Array(child_level.block_pointer)
    bs = BLOCK_SIZE
    dim_x = size(ptr_cpu, 1)
    dim_y = size(ptr_cpu, 2)
    dim_z = size(ptr_cpu, 3)
    child_dim_x = child_ptr_cpu === nothing ? 0 : size(child_ptr_cpu, 1)
    child_dim_y = child_ptr_cpu === nothing ? 0 : size(child_ptr_cpu, 2)
    child_dim_z = child_ptr_cpu === nothing ? 0 : size(child_ptr_cpu, 3)

    @inline function block_idx_at(gx::Int, gy::Int, gz::Int)
        (gx < 1 || gy < 1 || gz < 1) && return 0
        bx = div(gx - 1, bs) + 1
        by = div(gy - 1, bs) + 1
        bz = div(gz - 1, bs) + 1
        (bx < 1 || bx > dim_x || by < 1 || by > dim_y || bz < 1 || bz > dim_z) && return 0
        return Int(ptr_cpu[bx, by, bz])
    end

    @inline function active_block_at(gx::Int, gy::Int, gz::Int)
        return block_idx_at(gx, gy, gz) > 0
    end

    @inline function cell_vel_at(gx::Int, gy::Int, gz::Int)
        b = block_idx_at(gx, gy, gz)
        b == 0 && return (0.0, 0.0, 0.0, false)
        lx = (gx - 1) % bs + 1
        ly = (gy - 1) % bs + 1
        lz = (gz - 1) % bs + 1
        obs_cpu[lx, ly, lz, b] && return (0.0, 0.0, 0.0, false)
        return (
            Float64(vel_cpu[lx, ly, lz, b, 1]),
            Float64(vel_cpu[lx, ly, lz, b, 2]),
            Float64(vel_cpu[lx, ly, lz, b, 3]),
            true
        )
    end

    @inline function cell_pi_mag_at(gx::Int, gy::Int, gz::Int)
        !has_pi && return (0.0, false)
        b = block_idx_at(gx, gy, gz)
        b == 0 && return (0.0, false)
        lx = (gx - 1) % bs + 1
        ly = (gy - 1) % bs + 1
        lz = (gz - 1) % bs + 1
        obs_cpu[lx, ly, lz, b] && return (0.0, false)
        pxx = Float64(pi_cpu[lx, ly, lz, b, 1])
        pyy = Float64(pi_cpu[lx, ly, lz, b, 2])
        pzz = Float64(pi_cpu[lx, ly, lz, b, 3])
        pxy = Float64(pi_cpu[lx, ly, lz, b, 4])
        pxz = Float64(pi_cpu[lx, ly, lz, b, 5])
        pyz = Float64(pi_cpu[lx, ly, lz, b, 6])
        # Frobenius norm of the symmetric stress tensor.
        mag = sqrt(pxx * pxx + pyy * pyy + pzz * pzz +
                   2.0 * (pxy * pxy + pxz * pxz + pyz * pyz))
        return (mag, true)
    end

    @inline function covered_by_children(gx::Int, gy::Int, gz::Int)
        child_ptr_cpu === nothing && return false
        for sx in 0:1, sy in 0:1, sz in 0:1
            fgx = 2 * gx - 1 + sx
            fgy = 2 * gy - 1 + sy
            fgz = 2 * gz - 1 + sz
            (fgx < 1 || fgy < 1 || fgz < 1) && return false
            cbx = div(fgx - 1, bs) + 1
            cby = div(fgy - 1, bs) + 1
            cbz = div(fgz - 1, bs) + 1
            (cbx < 1 || cbx > child_dim_x ||
             cby < 1 || cby > child_dim_y ||
             cbz < 1 || cbz > child_dim_z) && return false
            child_ptr_cpu[cbx, cby, cbz] == 0 && return false
        end
        return true
    end

    @inline function near_fine_boundary(gx::Int, gy::Int, gz::Int)
        level.level_id <= 1 && return false
        for d in 1:band_depth
            (!active_block_at(gx - d, gy, gz) ||
             !active_block_at(gx + d, gy, gz) ||
             !active_block_at(gx, gy - d, gz) ||
             !active_block_at(gx, gy + d, gz) ||
             !active_block_at(gx, gy, gz - d) ||
             !active_block_at(gx, gy, gz + d)) && return true
        end
        return false
    end

    @inline function near_coarse_shell(gx::Int, gy::Int, gz::Int)
        child_ptr_cpu === nothing && return false
        covered_by_children(gx, gy, gz) && return false
        for d in 1:band_depth
            (covered_by_children(gx - d, gy, gz) ||
             covered_by_children(gx + d, gy, gz) ||
             covered_by_children(gx, gy - d, gz) ||
             covered_by_children(gx, gy + d, gz) ||
             covered_by_children(gx, gy, gz - d) ||
             covered_by_children(gx, gy, gz + d)) && return true
        end
        return false
    end

    @inline function coarse_shell_face_mask(gx::Int, gy::Int, gz::Int)
        child_ptr_cpu === nothing && return 0
        covered_by_children(gx, gy, gz) && return 0
        mask = 0
        for d in 1:band_depth
            covered_by_children(gx - d, gy, gz) && (mask |= 0x01) # x-
            covered_by_children(gx + d, gy, gz) && (mask |= 0x02) # x+
            covered_by_children(gx, gy - d, gz) && (mask |= 0x04) # y-
            covered_by_children(gx, gy + d, gz) && (mask |= 0x08) # y+
            covered_by_children(gx, gy, gz - d) && (mask |= 0x10) # z-
            covered_by_children(gx, gy, gz + d) && (mask |= 0x20) # z+
        end
        return mask
    end

    all_vals = Float64[]
    fine_vals = Float64[]
    coarse_vals = Float64[]
    interior_vals = Float64[]
    coarse_xneg_vals = Float64[]
    coarse_xpos_vals = Float64[]
    coarse_yneg_vals = Float64[]
    coarse_ypos_vals = Float64[]
    coarse_zneg_vals = Float64[]
    coarse_zpos_vals = Float64[]
    # s-mode (|Π_neq|) Nyquist highpass, in lockstep with the |u| stencil.
    pi_all_vals = Float64[]
    pi_fine_vals = Float64[]
    pi_coarse_vals = Float64[]
    pi_interior_vals = Float64[]

    n_blocks = size(vel_cpu, 4)
    @inbounds for b in 1:n_blocks
        bx = Int(map_x_cpu[b])
        by = Int(map_y_cpu[b])
        bz = Int(map_z_cpu[b])
        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            obs_cpu[lx, ly, lz, b] && continue
            gx = (bx - 1) * bs + lx
            gy = (by - 1) * bs + ly
            gz = (bz - 1) * bs + lz
            covered_by_children(gx, gy, gz) && continue

            ux = Float64(vel_cpu[lx, ly, lz, b, 1])
            uy = Float64(vel_cpu[lx, ly, lz, b, 2])
            uz = Float64(vel_cpu[lx, ly, lz, b, 3])
            u_mag = sqrt(ux * ux + uy * uy + uz * uz)

            sum_mag = 0.0
            complete = true
            for dir in 1:6
                ox = dir == 1 ? -1 : (dir == 2 ? 1 : 0)
                oy = dir == 3 ? -1 : (dir == 4 ? 1 : 0)
                oz = dir == 5 ? -1 : (dir == 6 ? 1 : 0)
                ngx = gx + ox
                ngy = gy + oy
                ngz = gz + oz
                if covered_by_children(ngx, ngy, ngz)
                    complete = false
                    break
                end
                nux, nuy, nuz, valid = cell_vel_at(ngx, ngy, ngz)
                if !valid
                    complete = false
                    break
                end
                sum_mag += sqrt(nux * nux + nuy * nuy + nuz * nuz)
            end
            complete || continue

            hp = abs(u_mag - sum_mag / 6.0)
            push!(all_vals, hp)

            fine_band = near_fine_boundary(gx, gy, gz)
            coarse_mask = coarse_shell_face_mask(gx, gy, gz)
            coarse_band = coarse_mask != 0
            if fine_band
                push!(fine_vals, hp)
            end
            if coarse_band
                push!(coarse_vals, hp)
                (coarse_mask & 0x01) != 0 && push!(coarse_xneg_vals, hp)
                (coarse_mask & 0x02) != 0 && push!(coarse_xpos_vals, hp)
                (coarse_mask & 0x04) != 0 && push!(coarse_yneg_vals, hp)
                (coarse_mask & 0x08) != 0 && push!(coarse_ypos_vals, hp)
                (coarse_mask & 0x10) != 0 && push!(coarse_zneg_vals, hp)
                (coarse_mask & 0x20) != 0 && push!(coarse_zpos_vals, hp)
            end
            if !fine_band && !coarse_band
                push!(interior_vals, hp)
            end

            # s-mode Nyquist proxy: same 6-neighbor highpass on |Π_neq|.
            # Only the same-level, complete-stencil cells already validated
            # above are eligible. h-mode contributes 0 by parity, so a
            # Nyquist mode in |Π_neq| isolates the BGK-marginal s-component
            # that τ_min ≈ 0.5 fails to damp.
            if has_pi
                pi_self, pi_self_valid = cell_pi_mag_at(gx, gy, gz)
                if pi_self_valid
                    pi_sum = 0.0
                    pi_complete = true
                    for dir in 1:6
                        ox = dir == 1 ? -1 : (dir == 2 ? 1 : 0)
                        oy = dir == 3 ? -1 : (dir == 4 ? 1 : 0)
                        oz = dir == 5 ? -1 : (dir == 6 ? 1 : 0)
                        pn, pn_valid = cell_pi_mag_at(gx + ox, gy + oy, gz + oz)
                        if !pn_valid
                            pi_complete = false
                            break
                        end
                        pi_sum += pn
                    end
                    if pi_complete
                        pi_hp = abs(pi_self - pi_sum / 6.0)
                        push!(pi_all_vals, pi_hp)
                        if fine_band
                            push!(pi_fine_vals, pi_hp)
                        end
                        if coarse_band
                            push!(pi_coarse_vals, pi_hp)
                        end
                        if !fine_band && !coarse_band
                            push!(pi_interior_vals, pi_hp)
                        end
                    end
                end
            end
        end
    end

    all_p95 = _diag_percentile!(all_vals, 0.95)
    fine_p95 = _diag_percentile!(fine_vals, 0.95)
    coarse_p95 = _diag_percentile!(coarse_vals, 0.95)
    interior_p95 = _diag_percentile!(interior_vals, 0.95)

    all_mean = _diag_mean(all_vals)
    fine_mean = _diag_mean(fine_vals)
    coarse_mean = _diag_mean(coarse_vals)
    interior_mean = _diag_mean(interior_vals)

    pi_all_p95 = _diag_percentile!(pi_all_vals, 0.95)
    pi_fine_p95 = _diag_percentile!(pi_fine_vals, 0.95)
    pi_coarse_p95 = _diag_percentile!(pi_coarse_vals, 0.95)
    pi_interior_p95 = _diag_percentile!(pi_interior_vals, 0.95)

    pi_all_mean = _diag_mean(pi_all_vals)
    pi_fine_mean = _diag_mean(pi_fine_vals)
    pi_coarse_mean = _diag_mean(pi_coarse_vals)
    pi_interior_mean = _diag_mean(pi_interior_vals)

    return (
        n_all = length(all_vals),
        mean_all = all_mean,
        p95_all = all_p95,
        max_all = isempty(all_vals) ? 0.0 : maximum(all_vals),
        n_fine_boundary = length(fine_vals),
        mean_fine_boundary = fine_mean,
        p95_fine_boundary = fine_p95,
        max_fine_boundary = isempty(fine_vals) ? 0.0 : maximum(fine_vals),
        n_coarse_shell = length(coarse_vals),
        mean_coarse_shell = coarse_mean,
        p95_coarse_shell = coarse_p95,
        max_coarse_shell = isempty(coarse_vals) ? 0.0 : maximum(coarse_vals),
        n_coarse_xneg = length(coarse_xneg_vals),
        mean_coarse_xneg = _diag_mean(coarse_xneg_vals),
        p95_coarse_xneg = _diag_percentile!(coarse_xneg_vals, 0.95),
        n_coarse_xpos = length(coarse_xpos_vals),
        mean_coarse_xpos = _diag_mean(coarse_xpos_vals),
        p95_coarse_xpos = _diag_percentile!(coarse_xpos_vals, 0.95),
        n_coarse_yneg = length(coarse_yneg_vals),
        mean_coarse_yneg = _diag_mean(coarse_yneg_vals),
        p95_coarse_yneg = _diag_percentile!(coarse_yneg_vals, 0.95),
        n_coarse_ypos = length(coarse_ypos_vals),
        mean_coarse_ypos = _diag_mean(coarse_ypos_vals),
        p95_coarse_ypos = _diag_percentile!(coarse_ypos_vals, 0.95),
        n_coarse_zneg = length(coarse_zneg_vals),
        mean_coarse_zneg = _diag_mean(coarse_zneg_vals),
        p95_coarse_zneg = _diag_percentile!(coarse_zneg_vals, 0.95),
        n_coarse_zpos = length(coarse_zpos_vals),
        mean_coarse_zpos = _diag_mean(coarse_zpos_vals),
        p95_coarse_zpos = _diag_percentile!(coarse_zpos_vals, 0.95),
        n_interior = length(interior_vals),
        mean_interior = interior_mean,
        p95_interior = interior_p95,
        max_interior = isempty(interior_vals) ? 0.0 : maximum(interior_vals),
        fine_to_interior_mean = interior_mean > 1e-30 ? fine_mean / interior_mean : NaN,
        coarse_to_interior_mean = interior_mean > 1e-30 ? coarse_mean / interior_mean : NaN,
        fine_to_interior_p95 = interior_p95 > 1e-30 ? fine_p95 / interior_p95 : NaN,
        coarse_to_interior_p95 = interior_p95 > 1e-30 ? coarse_p95 / interior_p95 : NaN,
        # s-mode (|Π_neq|) Nyquist content. n_pi_* count is the eligible
        # subset of the |u| highpass population (cells with the full Π_neq
        # 6-neighbor stencil + pi_neq allocated on this level). Stays 0 on
        # levels with no Π_neq carrier.
        n_pi_all = length(pi_all_vals),
        pi_mean_all = pi_all_mean,
        pi_p95_all = pi_all_p95,
        pi_max_all = isempty(pi_all_vals) ? 0.0 : maximum(pi_all_vals),
        n_pi_fine_boundary = length(pi_fine_vals),
        pi_mean_fine_boundary = pi_fine_mean,
        pi_p95_fine_boundary = pi_fine_p95,
        pi_max_fine_boundary = isempty(pi_fine_vals) ? 0.0 : maximum(pi_fine_vals),
        n_pi_coarse_shell = length(pi_coarse_vals),
        pi_mean_coarse_shell = pi_coarse_mean,
        pi_p95_coarse_shell = pi_coarse_p95,
        pi_max_coarse_shell = isempty(pi_coarse_vals) ? 0.0 : maximum(pi_coarse_vals),
        n_pi_interior = length(pi_interior_vals),
        pi_mean_interior = pi_interior_mean,
        pi_p95_interior = pi_interior_p95,
        pi_max_interior = isempty(pi_interior_vals) ? 0.0 : maximum(pi_interior_vals),
    )
end

function write_amr_highpass_csv_header(filepath::String)
    open(filepath, "w") do io
        println(io,
            "step,level,n_all,mean_all,p95_all,max_all," *
            "n_fine_boundary,mean_fine_boundary,p95_fine_boundary,max_fine_boundary," *
            "n_coarse_shell,mean_coarse_shell,p95_coarse_shell,max_coarse_shell," *
            "n_coarse_xneg,mean_coarse_xneg,p95_coarse_xneg," *
            "n_coarse_xpos,mean_coarse_xpos,p95_coarse_xpos," *
            "n_coarse_yneg,mean_coarse_yneg,p95_coarse_yneg," *
            "n_coarse_ypos,mean_coarse_ypos,p95_coarse_ypos," *
            "n_coarse_zneg,mean_coarse_zneg,p95_coarse_zneg," *
            "n_coarse_zpos,mean_coarse_zpos,p95_coarse_zpos," *
            "n_interior,mean_interior,p95_interior,max_interior," *
            "fine_to_interior_mean,coarse_to_interior_mean," *
            "fine_to_interior_p95,coarse_to_interior_p95," *
            # s-mode (|Π_neq|) Nyquist content. Zero on levels with no
            # pi_neq carrier (e.g. BGK without stress transfer).
            "n_pi_all,pi_mean_all,pi_p95_all,pi_max_all," *
            "n_pi_fine_boundary,pi_mean_fine_boundary,pi_p95_fine_boundary,pi_max_fine_boundary," *
            "n_pi_coarse_shell,pi_mean_coarse_shell,pi_p95_coarse_shell,pi_max_coarse_shell," *
            "n_pi_interior,pi_mean_interior,pi_p95_interior,pi_max_interior")
    end
end

function _append_amr_highpass_csv_row(filepath::String, step::Int, lvl::Int, h)
    open(filepath, "a") do io
        @printf(io,
            "%d,%d,%d,%.9e,%.9e,%.9e,%d,%.9e,%.9e,%.9e,%d,%.9e,%.9e,%.9e,%d,%.9e,%.9e,%d,%.9e,%.9e,%d,%.9e,%.9e,%d,%.9e,%.9e,%d,%.9e,%.9e,%d,%.9e,%.9e,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%d,%.9e,%.9e,%.9e,%d,%.9e,%.9e,%.9e,%d,%.9e,%.9e,%.9e,%d,%.9e,%.9e,%.9e\n",
            step, lvl,
            h.n_all, h.mean_all, h.p95_all, h.max_all,
            h.n_fine_boundary, h.mean_fine_boundary, h.p95_fine_boundary, h.max_fine_boundary,
            h.n_coarse_shell, h.mean_coarse_shell, h.p95_coarse_shell, h.max_coarse_shell,
            h.n_coarse_xneg, h.mean_coarse_xneg, h.p95_coarse_xneg,
            h.n_coarse_xpos, h.mean_coarse_xpos, h.p95_coarse_xpos,
            h.n_coarse_yneg, h.mean_coarse_yneg, h.p95_coarse_yneg,
            h.n_coarse_ypos, h.mean_coarse_ypos, h.p95_coarse_ypos,
            h.n_coarse_zneg, h.mean_coarse_zneg, h.p95_coarse_zneg,
            h.n_coarse_zpos, h.mean_coarse_zpos, h.p95_coarse_zpos,
            h.n_interior, h.mean_interior, h.p95_interior, h.max_interior,
            h.fine_to_interior_mean, h.coarse_to_interior_mean,
            h.fine_to_interior_p95, h.coarse_to_interior_p95,
            h.n_pi_all, h.pi_mean_all, h.pi_p95_all, h.pi_max_all,
            h.n_pi_fine_boundary, h.pi_mean_fine_boundary, h.pi_p95_fine_boundary, h.pi_max_fine_boundary,
            h.n_pi_coarse_shell, h.pi_mean_coarse_shell, h.pi_p95_coarse_shell, h.pi_max_coarse_shell,
            h.n_pi_interior, h.pi_mean_interior, h.pi_p95_interior, h.pi_max_interior)
    end
end

function print_amr_highpass_stats(grids; csv_path::Union{Nothing,String}=nothing, step::Int=-1)
    n = length(grids)
    for lvl in 1:n
        level = grids[lvl]
        length(level.active_block_coords) == 0 && continue
        child_level = lvl < n ? grids[lvl + 1] : nothing
        h = compute_level_native_highpass_stats(level, child_level; band_depth=3)
        h.n_all == 0 && continue
        @printf("          [AMR-HP] L%d native |u|-HP mean=%.2e p95=%.2e max=%.2e",
                lvl, h.mean_all, h.p95_all, h.max_all)
        if h.n_fine_boundary > 0
            @printf("  fine-boundary mean=%.2e p95=%.2e ratio=%.2f/%.2f",
                    h.mean_fine_boundary, h.p95_fine_boundary,
                    h.fine_to_interior_mean, h.fine_to_interior_p95)
        end
        if h.n_coarse_shell > 0
            @printf("  coarse-shell mean=%.2e p95=%.2e ratio=%.2f/%.2f",
                    h.mean_coarse_shell, h.p95_coarse_shell,
                    h.coarse_to_interior_mean, h.coarse_to_interior_p95)
        end
        @printf("  interior mean=%.2e p95=%.2e\n", h.mean_interior, h.p95_interior)
        if h.n_coarse_shell > 0
            @printf("          [AMR-HP-FACE] L%d coarse |u|-HP mean x-/x+=%.2e/%.2e  y-/y+=%.2e/%.2e  z-/z+=%.2e/%.2e\n",
                    lvl,
                    h.mean_coarse_xneg, h.mean_coarse_xpos,
                    h.mean_coarse_yneg, h.mean_coarse_ypos,
                    h.mean_coarse_zneg, h.mean_coarse_zpos)
        end
        if h.n_pi_all > 0
            # s-mode (|Π_neq|) Nyquist. Same units throughout the column;
            # interior is the diagnostically interesting region for the
            # τ→0.5 marginal-decay mechanism.
            @printf("          [AMR-HP] L%d |Π_neq|-HP interior mean=%.2e p95=%.2e max=%.2e (n=%d)\n",
                    lvl, h.pi_mean_interior, h.pi_p95_interior,
                    h.pi_max_interior, h.n_pi_interior)
        end
        if csv_path !== nothing
            _append_amr_highpass_csv_row(csv_path, step, lvl, h)
        end
    end
end

"""
    compute_level_block_modal_stats(level, child_level=nothing)

Pass 260 — per-block modal decomposition for AMR levels.

For each active block on `level` whose 8×8×8 cell cube is **not** fully
covered by `child_level` (so its cells actually carry the level's evolution),
project `|u| − mean(|u|)_block` onto two orthogonal spatial bases:

- `block_standing`: separable sin(π·(l−0.5)/BS) on each axis — peaks at the
  block centre, vanishes on the block faces (wavelength = one block).
- `cell_nyquist`: `(−1)^(lx+ly+lz)` — the 1-cell checkerboard.

Per-block scalars (block-mean centred, divided by `BS³` for amplitude):

    standing_proj = (1/BS³) · Σ W_sin(l) · (|u_l| − ⟨|u|⟩_block)
    nyquist_proj  = (1/BS³) · Σ (−1)^|l| · (|u_l| − ⟨|u|⟩_block)
    interior_minus_halo = ⟨|u|⟩_inner − ⟨|u|⟩_halo

`inner` = `lx,ly,lz ∈ 2..BS−1` (6×6×6 when BS=8), `halo` = the one-cell
outer shell of the block. Obstacle and child-covered cells are excluded;
blocks with fewer than 16 valid inner cells or 32 valid halo cells are
skipped to keep the means well-defined.

A pass-259 L3 instability with block-scale wavelength shows
`block_standing_*` growing while `cell_nyquist_*` stays at floor — that
disambiguation is the whole reason this diagnostic exists.
"""
function compute_level_block_modal_stats(level::BlockLevel,
                                         child_level::Union{Nothing,BlockLevel}=nothing)
    vel_cpu = Array(level.vel)
    obs_cpu = Array(level.obstacle)
    ptr_cpu = Array(level.block_pointer)
    map_x_cpu = Array(level.map_x)
    map_y_cpu = Array(level.map_y)
    map_z_cpu = Array(level.map_z)
    child_ptr_cpu = child_level === nothing ? nothing : Array(child_level.block_pointer)

    bs = BLOCK_SIZE
    bs_f = Float64(bs)
    cells_per_block = bs * bs * bs

    child_dim_x = child_ptr_cpu === nothing ? 0 : size(child_ptr_cpu, 1)
    child_dim_y = child_ptr_cpu === nothing ? 0 : size(child_ptr_cpu, 2)
    child_dim_z = child_ptr_cpu === nothing ? 0 : size(child_ptr_cpu, 3)

    @inline function block_fully_covered(bx::Int, by::Int, bz::Int)
        child_ptr_cpu === nothing && return false
        for dbz in 0:1, dby in 0:1, dbx in 0:1
            cbx = 2 * bx - 1 + dbx
            cby = 2 * by - 1 + dby
            cbz = 2 * bz - 1 + dbz
            (cbx < 1 || cbx > child_dim_x ||
             cby < 1 || cby > child_dim_y ||
             cbz < 1 || cbz > child_dim_z) && return false
            child_ptr_cpu[cbx, cby, cbz] == 0 && return false
        end
        return true
    end

    # Precompute the separable sin basis on the local axis (BS samples).
    # W_axis[l] = sin(π · (l − 0.5) / BS)  for l = 1..BS.
    W_axis = Vector{Float64}(undef, bs)
    @inbounds for l in 1:bs
        W_axis[l] = sin(pi * (Float64(l) - 0.5) / bs_f)
    end

    # Aggregators over qualifying blocks.
    standing_abs_vals  = Float64[]
    nyquist_abs_vals   = Float64[]
    # Pass 261: per-axis 1D Nyquist projections — needed because a vertical-X
    # stripe pattern projects to ZERO against the fully-3D (−1)^(lx+ly+lz)
    # basis (the (−1)^(ly+lz) sum over a block is zero). The pass-259 image
    # showed X stripes, so the relevant signal lives in `nyquist_x` here.
    nyquist_x_abs_vals = Float64[]
    nyquist_y_abs_vals = Float64[]
    nyquist_z_abs_vals = Float64[]
    interior_minus_halo_vals = Float64[]
    block_mean_dev_vals = Float64[]      # |⟨|u|⟩_block − ⟨|u|⟩_level|
    n_blocks_active = 0

    # First pass: level-wide mean |u| over valid cells (obstacle/covered excluded).
    # Used to compute per-block "block-mean drift" relative to the level mean —
    # complements the block-internal projections, which are mean-centred per block.
    level_sum_u = 0.0
    level_n_u = 0
    n_blocks_total = size(vel_cpu, 4)
    @inbounds for b in 1:n_blocks_total
        bx = Int(map_x_cpu[b])
        by = Int(map_y_cpu[b])
        bz = Int(map_z_cpu[b])
        block_fully_covered(bx, by, bz) && continue
        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            obs_cpu[lx, ly, lz, b] && continue
            ux = Float64(vel_cpu[lx, ly, lz, b, 1])
            uy = Float64(vel_cpu[lx, ly, lz, b, 2])
            uz = Float64(vel_cpu[lx, ly, lz, b, 3])
            level_sum_u += sqrt(ux*ux + uy*uy + uz*uz)
            level_n_u += 1
        end
    end
    level_mean_u = level_n_u > 0 ? level_sum_u / level_n_u : 0.0

    # Second pass: per-block stats.
    @inbounds for b in 1:n_blocks_total
        bx = Int(map_x_cpu[b])
        by = Int(map_y_cpu[b])
        bz = Int(map_z_cpu[b])
        block_fully_covered(bx, by, bz) && continue

        sum_u = 0.0
        n_valid = 0
        sum_inner = 0.0
        n_inner = 0
        sum_halo = 0.0
        n_halo = 0
        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            obs_cpu[lx, ly, lz, b] && continue
            ux = Float64(vel_cpu[lx, ly, lz, b, 1])
            uy = Float64(vel_cpu[lx, ly, lz, b, 2])
            uz = Float64(vel_cpu[lx, ly, lz, b, 3])
            u_mag = sqrt(ux*ux + uy*uy + uz*uz)
            sum_u += u_mag
            n_valid += 1
            on_face = (lx == 1) | (lx == bs) | (ly == 1) | (ly == bs) | (lz == 1) | (lz == bs)
            if on_face
                sum_halo += u_mag
                n_halo += 1
            else
                sum_inner += u_mag
                n_inner += 1
            end
        end

        # Need enough cells in each region for the means to be meaningful.
        (n_inner < 16 || n_halo < 32) && continue
        block_mean = sum_u / n_valid
        inner_mean = sum_inner / n_inner
        halo_mean = sum_halo / n_halo

        # Projections onto block-mean-centred residual. Using the block mean
        # (not the level mean) isolates the *intra-block* mode structure
        # from the slowly-varying level-scale gradient.
        standing_acc = 0.0
        nyquist_acc = 0.0
        nyquist_x_acc = 0.0
        nyquist_y_acc = 0.0
        nyquist_z_acc = 0.0
        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            obs_cpu[lx, ly, lz, b] && continue
            ux = Float64(vel_cpu[lx, ly, lz, b, 1])
            uy = Float64(vel_cpu[lx, ly, lz, b, 2])
            uz = Float64(vel_cpu[lx, ly, lz, b, 3])
            u_mag = sqrt(ux*ux + uy*uy + uz*uz)
            resid = u_mag - block_mean
            w_sin = W_axis[lx] * W_axis[ly] * W_axis[lz]
            w_nyq = ((lx + ly + lz) & 1) == 0 ? 1.0 : -1.0
            w_nyq_x = (lx & 1) == 1 ? 1.0 : -1.0
            w_nyq_y = (ly & 1) == 1 ? 1.0 : -1.0
            w_nyq_z = (lz & 1) == 1 ? 1.0 : -1.0
            standing_acc += w_sin * resid
            nyquist_acc += w_nyq * resid
            nyquist_x_acc += w_nyq_x * resid
            nyquist_y_acc += w_nyq_y * resid
            nyquist_z_acc += w_nyq_z * resid
        end
        push!(standing_abs_vals, abs(standing_acc) / Float64(cells_per_block))
        push!(nyquist_abs_vals,  abs(nyquist_acc)  / Float64(cells_per_block))
        push!(nyquist_x_abs_vals, abs(nyquist_x_acc) / Float64(cells_per_block))
        push!(nyquist_y_abs_vals, abs(nyquist_y_acc) / Float64(cells_per_block))
        push!(nyquist_z_abs_vals, abs(nyquist_z_acc) / Float64(cells_per_block))
        push!(interior_minus_halo_vals, inner_mean - halo_mean)
        push!(block_mean_dev_vals, abs(block_mean - level_mean_u))
        n_blocks_active += 1
    end

    @inline _rms(v) = isempty(v) ? 0.0 : sqrt(sum(x*x for x in v) / length(v))
    @inline _maxabs(v) = isempty(v) ? 0.0 : maximum(abs, v)
    @inline _meanabs(v) = isempty(v) ? 0.0 : sum(abs, v) / length(v)

    return (
        n_blocks = n_blocks_active,
        level_mean_u = level_mean_u,
        standing_mean = _meanabs(standing_abs_vals),
        standing_rms  = _rms(standing_abs_vals),
        standing_max  = _maxabs(standing_abs_vals),
        nyquist_mean  = _meanabs(nyquist_abs_vals),
        nyquist_rms   = _rms(nyquist_abs_vals),
        nyquist_max   = _maxabs(nyquist_abs_vals),
        nyquist_x_mean = _meanabs(nyquist_x_abs_vals),
        nyquist_x_max  = _maxabs(nyquist_x_abs_vals),
        nyquist_y_mean = _meanabs(nyquist_y_abs_vals),
        nyquist_y_max  = _maxabs(nyquist_y_abs_vals),
        nyquist_z_mean = _meanabs(nyquist_z_abs_vals),
        nyquist_z_max  = _maxabs(nyquist_z_abs_vals),
        inner_minus_halo_mean = isempty(interior_minus_halo_vals) ?
                                0.0 : sum(interior_minus_halo_vals) / length(interior_minus_halo_vals),
        inner_minus_halo_meanabs = _meanabs(interior_minus_halo_vals),
        inner_minus_halo_rms = _rms(interior_minus_halo_vals),
        inner_minus_halo_max = _maxabs(interior_minus_halo_vals),
        block_mean_dev_meanabs = _meanabs(block_mean_dev_vals),
        block_mean_dev_max = _maxabs(block_mean_dev_vals),
    )
end

function write_amr_block_modal_csv_header(filepath::String)
    open(filepath, "w") do io
        println(io,
            "step,level,n_blocks,level_mean_u," *
            "standing_mean,standing_rms,standing_max," *
            "nyquist_mean,nyquist_rms,nyquist_max," *
            "nyquist_x_mean,nyquist_x_max," *
            "nyquist_y_mean,nyquist_y_max," *
            "nyquist_z_mean,nyquist_z_max," *
            "inner_minus_halo_mean,inner_minus_halo_meanabs,inner_minus_halo_rms,inner_minus_halo_max," *
            "block_mean_dev_meanabs,block_mean_dev_max")
    end
    # Pass 263: reset the dedup set whenever a fresh header is written
    # (new run / new filepath). Without this, a Julia session that re-runs
    # the same case would silently drop all rows.
    _BLOCK_MODAL_WRITTEN[filepath] = Set{Tuple{Int,Int}}()
end

# Pass 263: dedup guard. Pass-262 runs produced two byte-identical CSV rows
# per (step, level) — `print_amr_block_modal_stats` reaches `_append…` via
# two call paths per qualifying batch. The pattern is (56,1)(56,2)(56,3)
# repeated, so consecutive-write tracking misses the second sequence. Track
# the full set of written (step, level) pairs per filepath and skip exact
# repeats. Set is re-init'd when a fresh header is written.
const _BLOCK_MODAL_WRITTEN = Dict{String,Set{Tuple{Int,Int}}}()

function _append_amr_block_modal_csv_row(filepath::String, step::Int, lvl::Int, h)
    seen = get!(_BLOCK_MODAL_WRITTEN, filepath) do
        Set{Tuple{Int,Int}}()
    end
    key = (step, lvl)
    if key in seen
        return  # already wrote this (step, level) — skip duplicate call path
    end
    push!(seen, key)
    open(filepath, "a") do io
        @printf(io,
            "%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",
            step, lvl,
            h.n_blocks, h.level_mean_u,
            h.standing_mean, h.standing_rms, h.standing_max,
            h.nyquist_mean, h.nyquist_rms, h.nyquist_max,
            h.nyquist_x_mean, h.nyquist_x_max,
            h.nyquist_y_mean, h.nyquist_y_max,
            h.nyquist_z_mean, h.nyquist_z_max,
            h.inner_minus_halo_mean, h.inner_minus_halo_meanabs,
            h.inner_minus_halo_rms, h.inner_minus_halo_max,
            h.block_mean_dev_meanabs, h.block_mean_dev_max)
    end
end

function print_amr_block_modal_stats(grids; csv_path::Union{Nothing,String}=nothing, step::Int=-1)
    n = length(grids)
    for lvl in 1:n
        level = grids[lvl]
        length(level.active_block_coords) == 0 && continue
        child_level = lvl < n ? grids[lvl + 1] : nothing
        h = compute_level_block_modal_stats(level, child_level)
        h.n_blocks == 0 && continue
        # Pass 261: report per-axis Nyquist (X/Y/Z) — the dominant axis is the
        # one to target with a damper. Pure-X stripes show nyq_x ≫ nyq_y, nyq_z.
        @printf("          [AMR-MODE] L%d n=%d  standing(mean/max)=%.2e/%.2e  nyquist(mean/max)=%.2e/%.2e  nyq_x/y/z mean=%.2e/%.2e/%.2e  inner-halo(mean/max)=%+.2e/%.2e\n",
                lvl, h.n_blocks,
                h.standing_mean, h.standing_max,
                h.nyquist_mean,  h.nyquist_max,
                h.nyquist_x_mean, h.nyquist_y_mean, h.nyquist_z_mean,
                h.inner_minus_halo_mean, h.inner_minus_halo_max)
        if csv_path !== nothing
            _append_amr_block_modal_csv_row(csv_path, step, lvl, h)
        end
    end
end

function write_amr_interface_csv_header(filepath::String)
    open(filepath, "w") do io
        println(io,
            "step,parent_level,child_level,n_interface,n_interior," *
            "drho_mean,rho_std_ratio,du_mean,u_max_if," *
            "face_count,face_mean_abs_drho,face_max_abs_drho,face_mean_abs_du," *
            "face_mean_abs_drho_x,face_mean_abs_drho_y,face_mean_abs_drho_z," *
            "face_mean_abs_du_x,face_mean_abs_du_y,face_mean_abs_du_z," *
            "face_mean_abs_drho_x_pos,face_mean_abs_drho_x_neg," *
            "face_mean_abs_drho_y_pos,face_mean_abs_drho_y_neg," *
            "face_mean_abs_drho_z_pos,face_mean_abs_drho_z_neg," *
            "face_mean_abs_du_x_pos,face_mean_abs_du_x_neg," *
            "face_mean_abs_du_y_pos,face_mean_abs_du_y_neg," *
            "face_mean_abs_du_z_pos,face_mean_abs_du_z_neg," *
            "face_mean_drho_x_pos,face_mean_drho_x_neg," *
            "face_mean_drho_y_pos,face_mean_drho_y_neg," *
            "face_mean_drho_z_pos,face_mean_drho_z_neg," *
            "face_mean_dun_x_pos,face_mean_dun_x_neg," *
            "face_mean_dun_y_pos,face_mean_dun_y_neg," *
            "face_mean_dun_z_pos,face_mean_dun_z_neg," *
            "face_rms_drho,face_rms_du," *
            "pi_face_count,pi_mean_abs_dpi,pi_rms_dpi,pi_max_abs_dpi," *
            "pi_mean_abs_dpi_x,pi_mean_abs_dpi_y,pi_mean_abs_dpi_z,pi_mean_pi_coarse," *
            "restrict_n_cells,restrict_mean_abs_drho,restrict_rms_drho,restrict_max_abs_drho,restrict_mean_drho," *
            "restrict_mean_abs_dj,restrict_rms_dj,restrict_max_abs_dj,restrict_mean_djx,restrict_mean_djy,restrict_mean_djz," *
            "partial_count,partial_mean_abs_drho,partial_max_abs_drho,partial_mean_abs_du," *
            "prolong_frontier_cells,prolong_parent_pulls,prolong_incomplete_cubic,prolong_anchor_invalid," *
            "face_mean_abs_dmass_flux,face_rms_dmass_flux,face_mean_dmass_flux," *
            "face_mean_abs_dmom_flux,face_rms_dmom_flux,face_mean_dmom_flux," *
            "face_mean_abs_dmass_flux_x,face_mean_abs_dmass_flux_y,face_mean_abs_dmass_flux_z," *
            "face_mean_abs_dmom_flux_x,face_mean_abs_dmom_flux_y,face_mean_abs_dmom_flux_z," *
            "face_mean_dmass_flux_x_pos,face_mean_dmass_flux_x_neg," *
            "face_mean_dmass_flux_y_pos,face_mean_dmass_flux_y_neg," *
            "face_mean_dmass_flux_z_pos,face_mean_dmass_flux_z_neg," *
            "face_mean_dmom_flux_x_pos,face_mean_dmom_flux_x_neg," *
            "face_mean_dmom_flux_y_pos,face_mean_dmom_flux_y_neg," *
            "face_mean_dmom_flux_z_pos,face_mean_dmom_flux_z_neg," *
            "face_count_single,face_count_multi," *
            "face_mean_abs_drho_single,face_mean_abs_drho_multi," *
            "face_mean_abs_du_single,face_mean_abs_du_multi," *
            "face_mean_abs_dmass_flux_single,face_mean_abs_dmass_flux_multi," *
            "face_mean_abs_dmom_flux_single,face_mean_abs_dmom_flux_multi")
    end
end

function _append_amr_interface_csv_row(filepath::String, step::Int, lvl::Int, s, j, pij, rc, pf, p)
    rho_std_ratio = s.rho_std_in > 1e-12 ? s.rho_std_if / s.rho_std_in : NaN
    open(filepath, "a") do io
        @printf(io,
            "%d,%d,%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%d,%.9e,%.9e,%.9e,%d,%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",
            step, lvl, lvl + 1, s.n_interface, s.n_interior,
            s.rho_mean_if - s.rho_mean_in,
            rho_std_ratio,
            s.u_mean_if - s.u_mean_in,
            s.u_max_if,
            j.n_faces,
            j.mean_abs_drho,
            j.max_abs_drho,
            j.mean_abs_du,
            j.mean_abs_drho_x,
            j.mean_abs_drho_y,
            j.mean_abs_drho_z,
            j.mean_abs_du_x,
            j.mean_abs_du_y,
            j.mean_abs_du_z,
            j.mean_abs_drho_x_pos,
            j.mean_abs_drho_x_neg,
            j.mean_abs_drho_y_pos,
            j.mean_abs_drho_y_neg,
            j.mean_abs_drho_z_pos,
            j.mean_abs_drho_z_neg,
            j.mean_abs_du_x_pos,
            j.mean_abs_du_x_neg,
            j.mean_abs_du_y_pos,
            j.mean_abs_du_y_neg,
            j.mean_abs_du_z_pos,
            j.mean_abs_du_z_neg,
            j.mean_drho_x_pos,
            j.mean_drho_x_neg,
            j.mean_drho_y_pos,
            j.mean_drho_y_neg,
            j.mean_drho_z_pos,
            j.mean_drho_z_neg,
            j.mean_dun_x_pos,
            j.mean_dun_x_neg,
            j.mean_dun_y_pos,
            j.mean_dun_y_neg,
            j.mean_dun_z_pos,
            j.mean_dun_z_neg,
            j.rms_drho,
            j.rms_du,
            pij.n_faces,
            pij.mean_abs_dpi,
            pij.rms_dpi,
            pij.max_abs_dpi,
            pij.mean_abs_dpi_x,
            pij.mean_abs_dpi_y,
            pij.mean_abs_dpi_z,
            pij.mean_pi_coarse,
            rc.n_cells,
            rc.mean_abs_drho,
            rc.rms_drho,
            rc.max_abs_drho,
            rc.mean_drho,
            rc.mean_abs_dj,
            rc.rms_dj,
            rc.max_abs_dj,
            rc.mean_djx,
            rc.mean_djy,
            rc.mean_djz,
            pf.n_cells,
            pf.mean_abs_drho,
            pf.max_abs_drho,
            pf.mean_abs_du,
            p.n_frontier_cells,
            p.n_parent_pulls,
            p.n_incomplete_cubic,
            p.n_anchor_invalid,
            j.mean_abs_dmass_flux,
            j.rms_dmass_flux,
            j.mean_dmass_flux,
            j.mean_abs_dmom_flux,
            j.rms_dmom_flux,
            j.mean_dmom_flux,
            j.mean_abs_dmass_flux_x,
            j.mean_abs_dmass_flux_y,
            j.mean_abs_dmass_flux_z,
            j.mean_abs_dmom_flux_x,
            j.mean_abs_dmom_flux_y,
            j.mean_abs_dmom_flux_z,
            j.mean_dmass_flux_x_pos,
            j.mean_dmass_flux_x_neg,
            j.mean_dmass_flux_y_pos,
            j.mean_dmass_flux_y_neg,
            j.mean_dmass_flux_z_pos,
            j.mean_dmass_flux_z_neg,
            j.mean_dmom_flux_x_pos,
            j.mean_dmom_flux_x_neg,
            j.mean_dmom_flux_y_pos,
            j.mean_dmom_flux_y_neg,
            j.mean_dmom_flux_z_pos,
            j.mean_dmom_flux_z_neg,
            j.n_faces_single,
            j.n_faces_multi,
            j.mean_abs_drho_single,
            j.mean_abs_drho_multi,
            j.mean_abs_du_single,
            j.mean_abs_du_multi,
            j.mean_abs_dmass_flux_single,
            j.mean_abs_dmass_flux_multi,
            j.mean_abs_dmom_flux_single,
            j.mean_abs_dmom_flux_multi)
    end
end

# ──────────────────────────────────────────────────────────────────────────────
# SYMMETRY-PLANE PROBE (2026-06-04) — the missing instrument that actually PROVES
# a y=0 mirror plane is free-slip. A free-slip plane is DEFINED by u_y = 0 at y=0
# (the normal velocity is odd under the mirror). Every other diagnostic on disk is
# rho (even under the mirror → blind to the symmetry defect) or |u| magnitude
# (weakly sensitive). This walks every level's y-min boundary fluid cells (global
# y-index 1 ⇒ block by==1, local ly==1), reads the SIGNED normal velocity u_y, and
# reports max/mean |u_y| normalized by u_inlet. A correct mirror BC ⇒ |u_y|/u_in
# ≈ machine-noise (≲1e-4); a broken plane (inlet/outlet preempting the mirror, the
# pre-2026-06-04 bug) injects a non-zero normal velocity that this catches directly.
#
# Returns (max_abs_uy_norm, mean_abs_uy_norm, n_plane_cells). Emits a console line
# and, when csv_path given, appends a row. u_inlet in LATTICE units.
function print_symmetry_plane_residual(grids, u_inlet::Real;
                                       step::Int=-1,
                                       csv_path::Union{Nothing,String}=nothing)
    u_ref = abs(Float64(u_inlet)) > 1e-12 ? abs(Float64(u_inlet)) : 1.0
    n = length(grids)
    glob_max = 0.0
    glob_sum = 0.0
    glob_cnt = 0
    worst_level = 0
    for lvl in 1:n
        level = grids[lvl]
        length(level.active_block_coords) == 0 && continue
        vel_cpu = Array(level.vel)            # GPU→CPU pullback
        obs_cpu = Array(level.obstacle)
        bs = BLOCK_SIZE
        lvl_max = 0.0; lvl_sum = 0.0; lvl_cnt = 0
        @inbounds for (b_idx, (_bx, by, _bz)) in enumerate(level.active_block_coords)
            by == 1 || continue              # only blocks touching the y=0 plane
            ly = 1                           # local y-index of the plane layer
            for lz in 1:bs, lx in 1:bs
                obs_cpu[lx, ly, lz, b_idx] && continue   # skip solid cells
                uy = abs(Float64(vel_cpu[lx, ly, lz, b_idx, 2])) / u_ref
                lvl_sum += uy; lvl_cnt += 1
                uy > lvl_max && (lvl_max = uy)
            end
        end
        lvl_cnt == 0 && continue
        glob_sum += lvl_sum; glob_cnt += lvl_cnt
        if lvl_max > glob_max
            glob_max = lvl_max; worst_level = lvl
        end
    end
    mean_uy = glob_cnt > 0 ? glob_sum / glob_cnt : 0.0
    @printf("          [SymPlane] y=0 free-slip residual: max|u_y|/u_in=%.3e (L%d)  mean=%.3e  cells=%d  %s\n",
            glob_max, worst_level, mean_uy, glob_cnt,
            glob_max <= 1.0e-4 ? "PASS(≤1e-4)" : (glob_max <= 1.0e-3 ? "WARN(≤1e-3)" : "FAIL(>1e-3)"))
    if csv_path !== nothing
        write_header = !isfile(csv_path)
        open(csv_path, "a") do io
            write_header && println(io, "step,max_abs_uy_norm,mean_abs_uy_norm,worst_level,n_plane_cells")
            @printf(io, "%d,%.6e,%.6e,%d,%d\n", step, glob_max, mean_uy, worst_level, glob_cnt)
        end
    end
    return (glob_max, mean_uy, glob_cnt)
end

function should_emit_full_amr_diagnostics(step::Int)
    # Perf opt 2026-05-28 — flipped default semantics. Was "always print
    # verbose AMR diagnostics when AMR_FULL_DIAGNOSTICS_FREQ <= 0", which
    # adds ~0.5-1 sec per batch of CPU-side AMR-registry computation. Now
    # the default (freq <= 0) emits verbose only at milestone steps
    # (0, 10, 100, 1000, 10000) — matches what users actually need. Users
    # who want it every step can set `amr_full_diagnostics_freq: 1`.
    freq = AMR_FULL_DIAGNOSTICS_FREQ
    # Milestone steps — emitted regardless of freq setting
    step <= 0 && return true
    step == 10 && return true
    step == 100 && return true
    step == 1000 && return true
    step == 10000 && return true
    # Periodic emit only when freq > 0
    freq > 0 && return step % freq == 0
    return false
end

function print_amr_interface_stats(grids; csv_path::Union{Nothing,String}=nothing, step::Int=-1, full::Bool=true)
    n = length(grids)
    n < 2 && return
    for lvl in 1:(n - 1)
        parent_level = grids[lvl]
        child_level = grids[lvl + 1]
        length(parent_level.active_block_coords) == 0 && continue
        length(child_level.active_block_coords) == 0 && continue
        s = compute_amr_interface_stats(parent_level, child_level)
        s.n_interface == 0 && continue
        d_rho = s.rho_mean_if - s.rho_mean_in
        d_u = s.u_mean_if - s.u_mean_in
        @printf("          [AMR] L%d↔L%d  n_if=%d n_in=%d  Δρ̄=%+.2e  σρ_if/σρ_in=%.2f  Δū=%+.2e  umax_if=%.4f\n",
                lvl, lvl + 1, s.n_interface, s.n_interior,
                d_rho,
                s.rho_std_in > 1e-12 ? s.rho_std_if / s.rho_std_in : NaN,
                d_u,
                s.u_max_if)
        j = compute_amr_shell_face_jump_stats(parent_level, child_level)
        if !full
            if j.n_faces > 0
                @printf("                compact face jump: n=%d  mean|drho|=%.2e  rms|drho|=%.2e  mean|du|=%.2e  rms|du|=%.2e  net mass/mom flux=%+.2e/%+.2e\n",
                        j.n_faces,
                        j.mean_abs_drho,
                        j.rms_drho,
                        j.mean_abs_du,
                        j.rms_du,
                        j.mean_dmass_flux,
                        j.mean_dmom_flux)
            end
            continue
        end
        if j.n_faces > 0
            @printf("                raw face jump: n=%d  mean|Δρ|=%.2e  rms|Δρ|=%.2e  max|Δρ|=%.2e  mean|Δu|=%.2e  rms|Δu|=%.2e  x/y/z mean|Δρ|=%.2e/%.2e/%.2e\n",
                    j.n_faces,
                    j.mean_abs_drho,
                    j.rms_drho,
                    j.max_abs_drho,
                    j.mean_abs_du,
                    j.rms_du,
                    j.mean_abs_drho_x,
                    j.mean_abs_drho_y,
                    j.mean_abs_drho_z)
            @printf("                face sign mean|drho|: +x/-x=%.2e/%.2e  +y/-y=%.2e/%.2e  +z/-z=%.2e/%.2e\n",
                    j.mean_abs_drho_x_pos,
                    j.mean_abs_drho_x_neg,
                    j.mean_abs_drho_y_pos,
                    j.mean_abs_drho_y_neg,
                    j.mean_abs_drho_z_pos,
                    j.mean_abs_drho_z_neg)
            @printf("                face sign mean|du|:   +x/-x=%.2e/%.2e  +y/-y=%.2e/%.2e  +z/-z=%.2e/%.2e\n",
                    j.mean_abs_du_x_pos,
                    j.mean_abs_du_x_neg,
                    j.mean_abs_du_y_pos,
                    j.mean_abs_du_y_neg,
                    j.mean_abs_du_z_pos,
                    j.mean_abs_du_z_neg)
            @printf("                face signed mean drho: +x/-x=%+.2e/%+.2e  +y/-y=%+.2e/%+.2e  +z/-z=%+.2e/%+.2e\n",
                    j.mean_drho_x_pos,
                    j.mean_drho_x_neg,
                    j.mean_drho_y_pos,
                    j.mean_drho_y_neg,
                    j.mean_drho_z_pos,
                    j.mean_drho_z_neg)
            @printf("                face signed mean dun:  +x/-x=%+.2e/%+.2e  +y/-y=%+.2e/%+.2e  +z/-z=%+.2e/%+.2e\n",
                    j.mean_dun_x_pos,
                    j.mean_dun_x_neg,
                    j.mean_dun_y_pos,
                    j.mean_dun_y_neg,
                    j.mean_dun_z_pos,
                    j.mean_dun_z_neg)
            @printf("                face flux jump: mean|d(rho*u_n)|=%.2e rms=%.2e net=%+.2e  x/y/z=%.2e/%.2e/%.2e\n",
                    j.mean_abs_dmass_flux,
                    j.rms_dmass_flux,
                    j.mean_dmass_flux,
                    j.mean_abs_dmass_flux_x,
                    j.mean_abs_dmass_flux_y,
                    j.mean_abs_dmass_flux_z)
            @printf("                normal momentum flux jump: mean=%.2e rms=%.2e net=%+.2e  x/y/z=%.2e/%.2e/%.2e\n",
                    j.mean_abs_dmom_flux,
                    j.rms_dmom_flux,
                    j.mean_dmom_flux,
                    j.mean_abs_dmom_flux_x,
                    j.mean_abs_dmom_flux_y,
                    j.mean_abs_dmom_flux_z)
            @printf("                signed flux budget: mass x+/x-=%+.2e/%+.2e y+/y-=%+.2e/%+.2e z+/z-=%+.2e/%+.2e  mom x+/x-=%+.2e/%+.2e y+/y-=%+.2e/%+.2e z+/z-=%+.2e/%+.2e\n",
                    j.mean_dmass_flux_x_pos,
                    j.mean_dmass_flux_x_neg,
                    j.mean_dmass_flux_y_pos,
                    j.mean_dmass_flux_y_neg,
                    j.mean_dmass_flux_z_pos,
                    j.mean_dmass_flux_z_neg,
                    j.mean_dmom_flux_x_pos,
                    j.mean_dmom_flux_x_neg,
                    j.mean_dmom_flux_y_pos,
                    j.mean_dmom_flux_y_neg,
                    j.mean_dmom_flux_z_pos,
                    j.mean_dmom_flux_z_neg)
            @printf("                shell multiplicity: single n=%d drho/du/flux/mom=%.2e/%.2e/%.2e/%.2e  multi n=%d %.2e/%.2e/%.2e/%.2e\n",
                    j.n_faces_single,
                    j.mean_abs_drho_single,
                    j.mean_abs_du_single,
                    j.mean_abs_dmass_flux_single,
                    j.mean_abs_dmom_flux_single,
                    j.n_faces_multi,
                    j.mean_abs_drho_multi,
                    j.mean_abs_du_multi,
                    j.mean_abs_dmass_flux_multi,
                    j.mean_abs_dmom_flux_multi)
        end
        pij = compute_amr_shell_face_pi_jump_stats(parent_level, child_level)
        if pij.n_faces > 0
            rel = pij.mean_pi_coarse > 1e-30 ? pij.mean_abs_dpi / pij.mean_pi_coarse : NaN
            @printf("                stress face jump |ΔΠ|_F: n=%d  mean=%.2e  rms=%.2e  max=%.2e  x/y/z=%.2e/%.2e/%.2e  rel=%.2e\n",
                    pij.n_faces,
                    pij.mean_abs_dpi,
                    pij.rms_dpi,
                    pij.max_abs_dpi,
                    pij.mean_abs_dpi_x,
                    pij.mean_abs_dpi_y,
                    pij.mean_abs_dpi_z,
                    rel)
        end
        rc = compute_amr_restriction_conservation_stats(parent_level, child_level)
        if rc.n_cells > 0
            @printf("                restriction conservation: n=%d  mean|drho|=%.2e  rms=%.2e  max=%.2e  mean|dj|=%.2e  rms=%.2e  net drho/djx/djy/djz=%+.2e/%+.2e/%+.2e/%+.2e\n",
                    rc.n_cells,
                    rc.mean_abs_drho,
                    rc.rms_drho,
                    rc.max_abs_drho,
                    rc.mean_abs_dj,
                    rc.rms_dj,
                    rc.mean_drho,
                    rc.mean_djx,
                    rc.mean_djy,
                    rc.mean_djz)
        end
        pf = compute_amr_partial_frontier_stats(parent_level, child_level)
        if pf.n_cells > 0
            @printf("                partial frontier: n=%d  mean|Δρ|=%.2e  max|Δρ|=%.2e  mean|Δu|=%.2e  x/y/z mean|Δρ|=%.2e/%.2e/%.2e\n",
                    pf.n_cells,
                    pf.mean_abs_drho,
                    pf.max_abs_drho,
                    pf.mean_abs_du,
                    pf.mean_abs_drho_x,
                    pf.mean_abs_drho_y,
                    pf.mean_abs_drho_z)
            @printf("                missing-side mean|drho|: x-/x+=%.2e/%.2e  y-/y+=%.2e/%.2e  z-/z+=%.2e/%.2e\n",
                    pf.mean_abs_drho_x_neg,
                    pf.mean_abs_drho_x_pos,
                    pf.mean_abs_drho_y_neg,
                    pf.mean_abs_drho_y_pos,
                    pf.mean_abs_drho_z_neg,
                    pf.mean_abs_drho_z_pos)
        end
        p = compute_amr_prolongation_topology_stats(parent_level, child_level)
        if p.n_parent_pulls > 0
            @printf("                prolong topology: cells=%d  pulls=%d  incomplete_cubic=%d (%.1f%%)  anchor_invalid=%d  x/y/z hits=%d/%d/%d\n",
                    p.n_frontier_cells,
                    p.n_parent_pulls,
                    p.n_incomplete_cubic,
                    100.0 * p.n_incomplete_cubic / max(p.n_parent_pulls, 1),
                    p.n_anchor_invalid,
                    p.n_incomplete_x,
                    p.n_incomplete_y,
                    p.n_incomplete_z)
            @printf("                prolong sign hits: x-/x+=%d/%d  y-/y+=%d/%d  z-/z+=%d/%d\n",
                    p.n_incomplete_x_neg,
                    p.n_incomplete_x_pos,
                    p.n_incomplete_y_neg,
                    p.n_incomplete_y_pos,
                    p.n_incomplete_z_neg,
                    p.n_incomplete_z_pos)
        end
        registry = build_amr_interface_registry(parent_level, child_level)
        reg = summarize_amr_interface_registry(registry)
        if reg.n_entities > 0
            @printf("                interface registry: entities=%d  face/edge/corner=%d/%d/%d  face x+/x-=%d/%d y+/y-=%d/%d z+/z-=%d/%d\n",
                    reg.n_entities,
                    reg.n_faces,
                    reg.n_edges,
                    reg.n_corners,
                    reg.n_face_x_pos,
                    reg.n_face_x_neg,
                    reg.n_face_y_pos,
                    reg.n_face_y_neg,
                    reg.n_face_z_pos,
                    reg.n_face_z_neg)
            @printf("                registry donor support: parent_inside_missing=%d  exterior_missing=%d  tangent_incomplete=%d  cubic_block_incomplete=%d  fill tangent/cubic=%.3f/%.3f\n",
                    reg.n_parent_inside_missing,
                    reg.n_parent_exterior_missing,
                    reg.n_tangent_incomplete,
                    reg.n_cubic_incomplete,
                    reg.mean_tangent_fill,
                    reg.mean_cubic_fill)
        end
        decomp_buffers = build_amr_decomposed_transfer_buffers(registry, parent_level, child_level)
        decomp = summarize_amr_decomposed_transfer_buffers(registry, decomp_buffers)
        if decomp.n_pairs > 0
            @printf("                decomp transfer buffer: pairs=%d face/edge/corner=%d/%d/%d  mean|drho|=%.2e  mean|dj|=%.2e  mean|dT|=%.2e  mean|dPi|=%.2e\n",
                    decomp.n_pairs,
                    decomp.n_face_pairs,
                    decomp.n_edge_pairs,
                    decomp.n_corner_pairs,
                    decomp.mean_abs_drho,
                    decomp.mean_abs_dj,
                    decomp.mean_abs_dT,
                    decomp.mean_abs_dpi)
            @printf("                decomp topology split: |drho| f/e/c=%.2e/%.2e/%.2e  |dj| f/e/c=%.2e/%.2e/%.2e  |dPi| f/e/c=%.2e/%.2e/%.2e\n",
                    decomp.mean_abs_drho_face,
                    decomp.mean_abs_drho_edge,
                    decomp.mean_abs_drho_corner,
                    decomp.mean_abs_dj_face,
                    decomp.mean_abs_dj_edge,
                    decomp.mean_abs_dj_corner,
                    decomp.mean_abs_dpi_face,
                    decomp.mean_abs_dpi_edge,
                    decomp.mean_abs_dpi_corner)
        end
        staged_buffers = build_amr_staged_transfer_buffers(registry, parent_level, child_level, decomp_buffers)
        staged = summarize_amr_staged_transfer_buffers(registry, staged_buffers)
        if staged.n_pairs > 0
            @printf("                staged transfer: pairs=%d temporal=%d pi_temporal=%d  child|t1 drho/dj/dT/dPi=%.2e/%.2e/%.2e/%.2e  child|t12=%.2e/%.2e/%.2e/%.2e\n",
                    staged.n_pairs,
                    staged.temporal_available ? 1 : 0,
                    staged.pi_temporal_available ? 1 : 0,
                    staged.mean_abs_child_t1_drho,
                    staged.mean_abs_child_t1_dj,
                    staged.mean_abs_child_t1_dT,
                    staged.mean_abs_child_t1_dpi,
                    staged.mean_abs_child_t12_drho,
                    staged.mean_abs_child_t12_dj,
                    staged.mean_abs_child_t12_dT,
                    staged.mean_abs_child_t12_dpi)
            @printf("                staged temporal parent: |t1-t0| drho/dj/dT=%.2e/%.2e/%.2e  t12 |drho| f/e/c=%.2e/%.2e/%.2e\n",
                    staged.mean_abs_parent_temporal_drho,
                    staged.mean_abs_parent_temporal_dj,
                    staged.mean_abs_parent_temporal_dT,
                    staged.mean_abs_child_t12_drho_face,
                    staged.mean_abs_child_t12_drho_edge,
                    staged.mean_abs_child_t12_drho_corner)
        end
        recomposition = summarize_amr_recomposition_preview(registry, staged_buffers; state=:t12)
        if recomposition.n_pairs > 0
            @printf("                recomposition preview(%s): pairs=%d  mean|drho/dj/dPi|=%.2e/%.2e/%.2e  max=%.2e/%.2e/%.2e  min_pop=%.2e floored=%d\n",
                    String(recomposition.state),
                    recomposition.n_pairs,
                    recomposition.mean_abs_drho,
                    recomposition.mean_abs_dj,
                    recomposition.mean_abs_dpi,
                    recomposition.max_abs_drho,
                    recomposition.max_abs_dj,
                    recomposition.max_abs_dpi,
                    recomposition.min_population,
                    recomposition.n_floored_populations)
            @printf("                recomposition topology: |dPi| f/e/c=%.2e/%.2e/%.2e\n",
                    recomposition.mean_abs_dpi_face,
                    recomposition.mean_abs_dpi_edge,
                    recomposition.mean_abs_dpi_corner)
        end
        live_preview = summarize_amr_live_preview_comparison(
            registry, staged_buffers, parent_level, child_level; state=:t12)
        stage_selector = summarize_amr_live_stage_selector(
            registry, staged_buffers, parent_level, child_level;
            t12_summary=live_preview)
        blend_sweep = summarize_amr_live_stage_blend_sweep(
            registry, staged_buffers, parent_level, child_level)
        blend_best = summarize_amr_live_stage_blend_best(blend_sweep)
        if live_preview.n_child_pairs > 0
            @printf("                live preview compare(%s): child pairs=%d  |df| mean/rms/max=%.2e/%.2e/%.2e  moments |drho/dj/dPi|=%.2e/%.2e/%.2e\n",
                    String(live_preview.state),
                    live_preview.n_child_pairs,
                    live_preview.mean_abs_df_child,
                    live_preview.mean_rms_df_child,
                    live_preview.max_abs_df_child,
                    live_preview.mean_abs_drho_child,
                    live_preview.mean_abs_dj_child,
                    live_preview.mean_abs_dpi_child)
            @printf("                live preview topology: |df| f/e/c=%.2e/%.2e/%.2e  face |djn|=%.2e  parent |df/drho/dj|=%.2e/%.2e/%.2e\n",
                    live_preview.mean_abs_df_face,
                    live_preview.mean_abs_df_edge,
                    live_preview.mean_abs_df_corner,
                    live_preview.mean_abs_face_normal_dj,
                    live_preview.mean_abs_df_parent,
                    live_preview.mean_abs_drho_parent,
                    live_preview.mean_abs_dj_parent)
        end
        if stage_selector.n_child_pairs > 0
            @printf("                live stage selector: preferred df/moment/face=%s/%s/%s  t1/t12 |df|=%.2e/%.2e  moment=%.2e/%.2e  face|djn|=%.2e/%.2e\n",
                    String(stage_selector.preferred_df),
                    String(stage_selector.preferred_moment),
                    String(stage_selector.preferred_face_normal),
                    stage_selector.t1_mean_abs_df_child,
                    stage_selector.t12_mean_abs_df_child,
                    stage_selector.t1_moment_score,
                    stage_selector.t12_moment_score,
                    stage_selector.t1_mean_abs_face_normal_dj,
                    stage_selector.t12_mean_abs_face_normal_dj)
        end
        if blend_best.n_weights > 0
            @printf("                live blend sweep: best w_t1 df/moment/face=%.2f/%.2f/%.2f  values=%.2e/%.2e/%.2e\n",
                    blend_best.best_df_weight,
                    blend_best.best_moment_weight,
                    blend_best.best_face_weight,
                    blend_best.best_df,
                    blend_best.best_moment,
                    blend_best.best_face_normal)
        end
        if csv_path !== nothing
            _append_amr_interface_csv_row(csv_path, step, lvl, s, j, pij, rc, pf, p)
            append_amr_interface_registry_csv_row(
                amr_interface_registry_csv_path(csv_path),
                step,
                lvl,
                lvl + 1,
                reg
            )
            append_amr_decomposed_transfer_csv_row(
                amr_decomposed_transfer_csv_path(csv_path),
                step,
                lvl,
                lvl + 1,
                decomp
            )
            append_amr_staged_transfer_csv_row(
                amr_staged_transfer_csv_path(csv_path),
                step,
                lvl,
                lvl + 1,
                staged
            )
            append_amr_recomposition_preview_csv_row(
                amr_recomposition_preview_csv_path(csv_path),
                step,
                lvl,
                lvl + 1,
                recomposition
            )
            append_amr_live_preview_compare_csv_row(
                amr_live_preview_compare_csv_path(csv_path),
                step,
                lvl,
                lvl + 1,
                live_preview
            )
            append_amr_live_stage_selector_csv_row(
                amr_live_stage_selector_csv_path(csv_path),
                step,
                lvl,
                lvl + 1,
                stage_selector
            )
            append_amr_live_stage_blend_sweep_csv_rows(
                amr_live_stage_blend_sweep_csv_path(csv_path),
                step,
                lvl,
                lvl + 1,
                blend_sweep
            )
        end
    end
end

function write_transonic_wake_csv_header(filepath::String)
    cols = [
        "step", "active", "level", "u_inf",
        "obs_x_min", "obs_x_max", "obs_y_min", "obs_y_max", "obs_z_min", "obs_z_max",
        "wake_x_min", "wake_x_max",
        "n_core", "n_shear", "n_base",
        "wake_deficit", "core_ux_mean", "core_umag_mean",
        "shear_umag_max", "shear_overspeed_max", "shear_overspeed_frac",
        "wake_crossflow_rms", "wake_upper_lower_asym",
        "base_pressure_rel_mean",
        "shift_total_blocks", "shift0_blocks", "shift1_blocks", "shift_other_blocks",
        "shift_faces", "shift_faces_x", "shift_faces_y", "shift_faces_z",
    ]
    open(filepath, "w") do io
        println(io, join(cols, ","))
    end
end

function _transonic_shift_summary(grids)
    total_blocks = 0
    shift0_blocks = 0
    shift1_blocks = 0
    shift_other_blocks = 0
    shift_faces = 0
    shift_faces_x = 0
    shift_faces_y = 0
    shift_faces_z = 0

    for level in grids
        if !hasproperty(level, :block_shift) || length(level.active_block_coords) == 0
            continue
        end
        shifts = Array(level.block_shift)
        mapx = Array(level.map_x)
        mapy = Array(level.map_y)
        mapz = Array(level.map_z)
        n_blocks = length(level.active_block_coords)
        total_blocks += n_blocks

        coord_to_block = Dict{NTuple{3,Int},Int}()
        for b in 1:n_blocks
            sh = Int(shifts[b])
            if sh == 0
                shift0_blocks += 1
            elseif sh == 1
                shift1_blocks += 1
            else
                shift_other_blocks += 1
            end
            coord_to_block[(Int(mapx[b]), Int(mapy[b]), Int(mapz[b]))] = b
        end

        for (coord, b) in coord_to_block
            sh = shifts[b]
            nb = get(coord_to_block, (coord[1] + 1, coord[2], coord[3]), 0)
            if nb > 0 && shifts[nb] != sh
                shift_faces += 1
                shift_faces_x += 1
            end
            nb = get(coord_to_block, (coord[1], coord[2] + 1, coord[3]), 0)
            if nb > 0 && shifts[nb] != sh
                shift_faces += 1
                shift_faces_y += 1
            end
            nb = get(coord_to_block, (coord[1], coord[2], coord[3] + 1), 0)
            if nb > 0 && shifts[nb] != sh
                shift_faces += 1
                shift_faces_z += 1
            end
        end
    end

    return (
        total_blocks=total_blocks,
        shift0_blocks=shift0_blocks,
        shift1_blocks=shift1_blocks,
        shift_other_blocks=shift_other_blocks,
        shift_faces=shift_faces,
        shift_faces_x=shift_faces_x,
        shift_faces_y=shift_faces_y,
        shift_faces_z=shift_faces_z,
    )
end

function _find_transonic_wake_obstacle_bbox(grids)
    for lvl_idx in reverse(eachindex(grids))
        level = grids[lvl_idx]
        length(level.active_block_coords) == 0 && continue
        obs = Array(level.obstacle)
        any(obs) || continue
        mapx = Array(level.map_x)
        mapy = Array(level.map_y)
        mapz = Array(level.map_z)
        bs = BLOCK_SIZE
        x_min = typemax(Int); x_max = typemin(Int)
        y_min = typemax(Int); y_max = typemin(Int)
        z_min = typemax(Int); z_max = typemin(Int)
        n_blocks = length(level.active_block_coords)
        for b in 1:n_blocks
            bx = Int(mapx[b]); by = Int(mapy[b]); bz = Int(mapz[b])
            for lz in 1:bs, ly in 1:bs, lx in 1:bs
                obs[lx, ly, lz, b] || continue
                gx = (bx - 1) * bs + lx
                gy = (by - 1) * bs + ly
                gz = (bz - 1) * bs + lz
                x_min = min(x_min, gx); x_max = max(x_max, gx)
                y_min = min(y_min, gy); y_max = max(y_max, gy)
                z_min = min(z_min, gz); z_max = max(z_max, gz)
            end
        end
        if x_min <= x_max
            return lvl_idx, (x_min=x_min, x_max=x_max,
                             y_min=y_min, y_max=y_max,
                             z_min=z_min, z_max=z_max)
        end
    end
    return 0, nothing
end

function compute_transonic_wake_diagnostics(
    grids;
    u_inf::Real,
    x_min_cells::Real=0.0,
    max_x_diameters::Real=3.0,
    overspeed_warn_frac::Real=0.03,
)
    shift_stats = _transonic_shift_summary(grids)
    lvl_idx, bbox = _find_transonic_wake_obstacle_bbox(grids)
    u_ref = Float64(u_inf)
    u_ref = u_ref > 1e-8 ? u_ref : Float64(U_TARGET)

    if lvl_idx == 0 || bbox === nothing
        return merge((
            active=false, level=0, u_inf=u_ref,
            obs_x_min=0, obs_x_max=0, obs_y_min=0, obs_y_max=0, obs_z_min=0, obs_z_max=0,
            wake_x_min=0.0, wake_x_max=0.0,
            n_core=0, n_shear=0, n_base=0,
            wake_deficit=0.0, core_ux_mean=0.0, core_umag_mean=0.0,
            shear_umag_max=0.0, shear_overspeed_max=0.0, shear_overspeed_frac=0.0,
            wake_crossflow_rms=0.0, wake_upper_lower_asym=0.0,
            base_pressure_rel_mean=0.0,
        ), shift_stats)
    end

    level = grids[lvl_idx]
    rho = Array(level.rho)
    vel = Array(level.vel)
    obs = Array(level.obstacle)
    sponge = hasproperty(level, :sponge) ? Array(level.sponge) : nothing
    mapx = Array(level.map_x)
    mapy = Array(level.map_y)
    mapz = Array(level.map_z)
    has_T = DDF_ENABLED && has_thermal(level)
    temp = has_T ? Array(level.temperature) : nothing

    bs = BLOCK_SIZE
    diam_y = max(1.0, Float64(bbox.y_max - bbox.y_min + 1))
    diam_z = max(1.0, Float64(bbox.z_max - bbox.z_min + 1))
    y_center = 0.5 * Float64(bbox.y_min + bbox.y_max)
    z_center = 0.5 * Float64(bbox.z_min + bbox.z_max)

    wake_x_min = Float64(x_min_cells) > 0.0 ? Float64(x_min_cells) :
                 max(Float64(bbox.x_max + 1), Float64(LOCAL_LATTICE_SHIFT_ZERO_ALLOWED_X_MIN_CELLS))
    wake_x_min <= 0.0 && (wake_x_min = Float64(bbox.x_max + 1))
    wake_x_max = Float64(bbox.x_max) + max(Float64(max_x_diameters), 0.25) * diam_z

    core_half_z = max(1.5, 0.20 * diam_z)
    mid_half_y = max(1.5, 0.42 * diam_y)
    shear_lo_z = 0.34 * diam_z
    shear_hi_z = 0.72 * diam_z
    wake_band_z = max(shear_hi_z, core_half_z)
    base_x_max = Float64(bbox.x_max) + max(2.0, 0.35 * diam_z)

    pressure_coeff = diagnostic_pressure_coeff()
    p_inf = max(Float64(IC_UNIFORM_RHO) * pressure_coeff *
                ((DDF_ENABLED || EOS_THERMODYNAMIC) ? Float64(DDF_T_INLET) : 1.0), 1e-12)

    n_core = 0; n_shear = 0; n_base = 0
    core_ux_sum = 0.0
    core_umag_sum = 0.0
    shear_umag_max = 0.0
    shear_overspeed_count = 0
    crossflow_sq_sum = 0.0
    n_crossflow = 0
    upper_ux_sum = 0.0; lower_ux_sum = 0.0
    n_upper = 0; n_lower = 0
    base_p_rel_sum = 0.0
    n_blocks = length(level.active_block_coords)

    for b in 1:n_blocks
        bx = Int(mapx[b]); by = Int(mapy[b]); bz = Int(mapz[b])
        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            obs[lx, ly, lz, b] && continue
            gx = (bx - 1) * bs + lx
            gy = (by - 1) * bs + ly
            gz = (bz - 1) * bs + lz
            gx_f = Float64(gx)
            in_wake_x = gx_f >= wake_x_min && gx_f <= wake_x_max
            in_wake_x || continue

            dy = abs(Float64(gy) - y_center)
            dz = Float64(gz) - z_center
            adz = abs(dz)
            in_midspan = dy <= mid_half_y
            in_midspan || continue

            ux = Float64(vel[lx, ly, lz, b, 1])
            uy = Float64(vel[lx, ly, lz, b, 2])
            uz = Float64(vel[lx, ly, lz, b, 3])
            umag = sqrt(ux*ux + uy*uy + uz*uz)

            if adz <= wake_band_z
                crossflow_sq_sum += uy*uy + uz*uz
                n_crossflow += 1
            end

            if adz <= core_half_z
                n_core += 1
                core_ux_sum += ux
                core_umag_sum += umag
            end

            if shear_lo_z <= adz <= shear_hi_z
                n_shear += 1
                shear_umag_max = max(shear_umag_max, umag)
                umag > u_ref * (1.0 + Float64(overspeed_warn_frac)) &&
                    (shear_overspeed_count += 1)
                if dz > 0.0
                    upper_ux_sum += ux
                    n_upper += 1
                elseif dz < 0.0
                    lower_ux_sum += ux
                    n_lower += 1
                end
            end

            if gx_f <= base_x_max && adz <= core_half_z
                sp = sponge === nothing ? 0.0 : Float64(sponge[lx, ly, lz, b])
                T = has_T ? Float64(temp[lx, ly, lz, b]) :
                    diagnostic_effective_temperature(
                        rho[lx, ly, lz, b], ux, uy, uz, NaN, sp, u_ref)
                p_rel = Float64(rho[lx, ly, lz, b]) * pressure_coeff * T / p_inf
                base_p_rel_sum += p_rel
                n_base += 1
            end
        end
    end

    core_ux_mean = n_core > 0 ? core_ux_sum / n_core : 0.0
    core_umag_mean = n_core > 0 ? core_umag_sum / n_core : 0.0
    wake_deficit = n_core > 0 && u_ref > 1e-8 ? 1.0 - core_ux_mean / u_ref : 0.0
    shear_overspeed_max = n_shear > 0 && u_ref > 1e-8 ? max(shear_umag_max / u_ref - 1.0, 0.0) : 0.0
    shear_overspeed_frac = n_shear > 0 ? shear_overspeed_count / n_shear : 0.0
    wake_crossflow_rms = n_crossflow > 0 ? sqrt(crossflow_sq_sum / n_crossflow) : 0.0
    upper_mean = n_upper > 0 ? upper_ux_sum / n_upper : 0.0
    lower_mean = n_lower > 0 ? lower_ux_sum / n_lower : 0.0
    wake_upper_lower_asym = (n_upper > 0 && n_lower > 0 && u_ref > 1e-8) ?
                            abs(upper_mean - lower_mean) / u_ref : 0.0
    base_pressure_rel_mean = n_base > 0 ? base_p_rel_sum / n_base : 0.0

    return merge((
        active=true,
        level=lvl_idx,
        u_inf=u_ref,
        obs_x_min=bbox.x_min, obs_x_max=bbox.x_max,
        obs_y_min=bbox.y_min, obs_y_max=bbox.y_max,
        obs_z_min=bbox.z_min, obs_z_max=bbox.z_max,
        wake_x_min=wake_x_min, wake_x_max=wake_x_max,
        n_core=n_core, n_shear=n_shear, n_base=n_base,
        wake_deficit=wake_deficit,
        core_ux_mean=core_ux_mean,
        core_umag_mean=core_umag_mean,
        shear_umag_max=shear_umag_max,
        shear_overspeed_max=shear_overspeed_max,
        shear_overspeed_frac=shear_overspeed_frac,
        wake_crossflow_rms=wake_crossflow_rms,
        wake_upper_lower_asym=wake_upper_lower_asym,
        base_pressure_rel_mean=base_pressure_rel_mean,
    ), shift_stats)
end

function _append_transonic_wake_csv_row(filepath::String, step::Int, s)
    open(filepath, "a") do io
        @printf(io,
                "%d,%d,%d,%.9e,%d,%d,%d,%d,%d,%d,%.9e,%.9e,%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%d,%d,%d,%d,%d,%d,%d,%d\n",
                step, s.active ? 1 : 0, s.level, s.u_inf,
                s.obs_x_min, s.obs_x_max, s.obs_y_min, s.obs_y_max, s.obs_z_min, s.obs_z_max,
                s.wake_x_min, s.wake_x_max,
                s.n_core, s.n_shear, s.n_base,
                s.wake_deficit, s.core_ux_mean, s.core_umag_mean,
                s.shear_umag_max, s.shear_overspeed_max, s.shear_overspeed_frac,
                s.wake_crossflow_rms, s.wake_upper_lower_asym,
                s.base_pressure_rel_mean,
                s.total_blocks, s.shift0_blocks, s.shift1_blocks, s.shift_other_blocks,
                s.shift_faces, s.shift_faces_x, s.shift_faces_y, s.shift_faces_z)
    end
end

function print_transonic_wake_diagnostics(grids, step::Int, u_inf::Real; csv_path::Union{Nothing,String}=nothing)
    s = compute_transonic_wake_diagnostics(
        grids;
        u_inf=u_inf,
        x_min_cells=TRANSONIC_WAKE_DIAG_X_MIN_CELLS,
        max_x_diameters=TRANSONIC_WAKE_DIAG_MAX_X_DIAMETERS,
        overspeed_warn_frac=TRANSONIC_WAKE_OVERSPEED_WARN_FRAC,
    )
    if s.active
        @printf("          [Wake] step=%d  deficit=%+.3f  core_ux=%.4f  shear_over(max/frac)=%.3f/%.3f  cross_rms=%.4f  asym=%.3f  pbase/pinf=%.3f  shift0/1/other=%d/%d/%d faces=%d\n",
                step, s.wake_deficit, s.core_ux_mean,
                s.shear_overspeed_max, s.shear_overspeed_frac,
                s.wake_crossflow_rms, s.wake_upper_lower_asym,
                s.base_pressure_rel_mean,
                s.shift0_blocks, s.shift1_blocks, s.shift_other_blocks,
                s.shift_faces)
    else
        @printf("          [Wake] step=%d  no obstacle wake bbox available  shift0/1/other=%d/%d/%d faces=%d\n",
                step, s.shift0_blocks, s.shift1_blocks, s.shift_other_blocks, s.shift_faces)
    end
    csv_path !== nothing && _append_transonic_wake_csv_row(csv_path, step, s)
    return s
end

function check_stability(level::BlockLevel, step::Int)
    stats = compute_flow_stats(level)
    
    warnings = String[]
    
    if stats.v_max > 0.3
        push!(warnings, @sprintf("High velocity: %.4f (Ma > 0.5)", stats.v_max))
    end
    
    if stats.rho_min < 0.5
        push!(warnings, @sprintf("Low density: %.4f", stats.rho_min))
    end
    
    if stats.rho_max > 1.5
        push!(warnings, @sprintf("High density: %.4f", stats.rho_max))
    end
    
    if !isempty(warnings)
        println("[WARNING] Step $step stability issues:")
        for w in warnings
            println("  - $w")
        end
        return false
    end

    return true
end

const _STABILITY_ABORT_MASS_BASELINE = Ref{Float64}(0.0)
const _STABILITY_ABORT_FIRST = Ref{Bool}(true)

function reset_stability_abort_baseline!()
    _STABILITY_ABORT_MASS_BASELINE[] = 0.0
    _STABILITY_ABORT_FIRST[] = true
    return nothing
end

function _stability_abort_global_flow_stats(grids)
    rho_min = Inf
    rho_max = -Inf
    u_max = 0.0
    n_fluid_total = 0

    for level in grids
        length(level.active_block_coords) == 0 && continue
        s = compute_flow_stats(level)
        n_fluid = Int(s.n_fluid)
        n_fluid <= 0 && continue
        rho_min = min(rho_min, Float64(s.rho_min))
        rho_max = max(rho_max, Float64(s.rho_max))
        u_max = max(u_max, Float64(s.v_max))
        n_fluid_total += n_fluid
    end

    if n_fluid_total == 0
        return (n_fluid=0, rho_min=1.0, rho_max=1.0, u_max=0.0)
    end
    return (n_fluid=n_fluid_total, rho_min=rho_min, rho_max=rho_max, u_max=u_max)
end

function _stability_abort_extrema_records(grids)
    rho_min_val = Inf
    rho_max_val = -Inf
    u_max_val = -Inf

    rho_min_loc = nothing
    rho_max_loc = nothing
    u_max_loc = nothing

    for (lvl_idx, level) in enumerate(grids)
        length(level.active_block_coords) == 0 && continue

        rho = Array(level.rho)
        vel = Array(level.vel)
        obs = Array(level.obstacle)
        amr_cov = hasproperty(level, :amr_covered) ? Array(level.amr_covered) : nothing
        wall_dist = hasproperty(level, :wall_dist) ? Array(level.wall_dist) : nothing
        map_x = Array(level.map_x)
        map_y = Array(level.map_y)
        map_z = Array(level.map_z)
        shifts = hasproperty(level, :block_shift) ? Array(level.block_shift) : Int32[]
        wall_guard = hasproperty(level, :local_shift_wall_guard) ? Array(level.local_shift_wall_guard) : Int32[]
        total_x = Int(level.grid_dim_x) * BLOCK_SIZE
        total_y = Int(level.grid_dim_y) * BLOCK_SIZE
        total_z = Int(level.grid_dim_z) * BLOCK_SIZE

        nx, ny, nz, nb = size(rho)
        for b in 1:nb
            bx = Int(map_x[b])
            by = Int(map_y[b])
            bz = Int(map_z[b])
            sh = !isempty(shifts) && b <= length(shifts) ? Int(shifts[b]) : 0
            for lz in 1:nz, ly in 1:ny, lx in 1:nx
                obs[lx, ly, lz, b] && continue

                rv = Float64(rho[lx, ly, lz, b])
                ux = Float64(vel[lx, ly, lz, b, 1])
                uy = Float64(vel[lx, ly, lz, b, 2])
                uz = Float64(vel[lx, ly, lz, b, 3])
                umag = sqrt(ux*ux + uy*uy + uz*uz)
                gx = (bx - 1) * BLOCK_SIZE + lx
                gy = (by - 1) * BLOCK_SIZE + ly
                gz = (bz - 1) * BLOCK_SIZE + lz
                wd = (wall_dist !== nothing && size(wall_dist, 4) >= b) ?
                     Float64(wall_dist[lx, ly, lz, b]) : NaN
                amr = (amr_cov !== nothing && size(amr_cov, 4) >= b) ?
                      Bool(amr_cov[lx, ly, lz, b]) : false
                owns_wall_links = !isempty(wall_guard) && b <= length(wall_guard) &&
                                  wall_guard[b] > Int32(0)
                face_dx = min(gx - 1, total_x - gx)
                face_dy = min(gy - 1, total_y - gy)
                face_dz = min(gz - 1, total_z - gz)
                loc = (lvl=lvl_idx, b=b, lx=lx, ly=ly, lz=lz,
                       gx=gx, gy=gy, gz=gz, shift=sh,
                       rho=rv, ux=ux, uy=uy, uz=uz, umag=umag,
                       wall_dist=wd, owns_wall_links=owns_wall_links,
                       amr=amr,
                       face_dx=face_dx, face_dy=face_dy, face_dz=face_dz)

                if rv < rho_min_val
                    rho_min_val = rv
                    rho_min_loc = loc
                end
                if rv > rho_max_val
                    rho_max_val = rv
                    rho_max_loc = loc
                end
                if umag > u_max_val
                    u_max_val = umag
                    u_max_loc = loc
                end
            end
        end
    end

    return (rho_min=rho_min_loc, rho_max=rho_max_loc, u_max=u_max_loc)
end

function _stability_extrema_q_link_count(grids, loc, q_map_cache)
    loc === nothing && return 0
    1 <= loc.lvl <= length(grids) || return 0
    level = grids[loc.lvl]
    if !(hasproperty(level, :bouzidi_enabled) && level.bouzidi_enabled &&
         hasproperty(level, :bouzidi_q_map))
        return 0
    end
    q_map = get!(q_map_cache, loc.lvl) do
        Array(level.bouzidi_q_map)
    end
    n_dirs = Int(N_DIRS)
    q_dirs_total = size(q_map, 5)
    q_frame_count = (q_dirs_total >= n_dirs && n_dirs > 0) ? max(div(q_dirs_total, n_dirs), 1) : 1
    if size(q_map, 1) < loc.lx || size(q_map, 2) < loc.ly ||
       size(q_map, 3) < loc.lz || size(q_map, 4) < loc.b
        return 0
    end
    q_offset = q_frame_count > 1 ? clamp(loc.shift, 0, q_frame_count - 1) * n_dirs : 0
    count = 0
    for kk in 1:n_dirs
        qv = Float32(q_map[loc.lx, loc.ly, loc.lz, loc.b, q_offset + kk])
        if qv > 0.0f0 && qv <= 1.0f0
            count += 1
        end
    end
    return count
end

function _format_stability_extremum(grids, label::String, loc, q_map_cache)
    loc === nothing && return label * "@none"
    return @sprintf("%s@L%d b=%d local=(%d,%d,%d) global=(%d,%d,%d) shift=%d rho=%.6g u=(%.6g,%.6g,%.6g)|u|=%.6g wall_dist=%.4g wall_block=%d qlinks=%d amr=%d face_dist=(%d,%d,%d)",
                    label, loc.lvl, loc.b, loc.lx, loc.ly, loc.lz,
                    loc.gx, loc.gy, loc.gz, loc.shift, loc.rho,
                    loc.ux, loc.uy, loc.uz, loc.umag,
                    loc.wall_dist, loc.owns_wall_links ? 1 : 0,
                    _stability_extrema_q_link_count(grids, loc, q_map_cache),
                    loc.amr ? 1 : 0,
                    loc.face_dx, loc.face_dy, loc.face_dz)
end

function _stability_abort_extrema_locations(grids)
    rec = _stability_abort_extrema_records(grids)
    q_map_cache = Dict{Int,Any}()
    return _format_stability_extremum(grids, "rho_min", rec.rho_min, q_map_cache) * "; " *
           _format_stability_extremum(grids, "rho_max", rec.rho_max, q_map_cache) * "; " *
           _format_stability_extremum(grids, "u_max", rec.u_max, q_map_cache)
end

function write_stability_extrema_csv_header(filepath::String)
    open(filepath, "w") do io
        println(io, "step,label,level,block,lx,ly,lz,gx,gy,gz,shift,rho,ux,uy,uz,umag,wall_dist,wall_block,qlinks,amr,face_dx,face_dy,face_dz")
    end
end

function _append_stability_extremum_csv(io, step::Int, label::String, loc, qlinks::Int)
    if loc === nothing
        @printf(io, "%d,%s,0,0,0,0,0,0,0,0,0,NaN,NaN,NaN,NaN,NaN,NaN,0,0,0,0,0,0\n",
                step, label)
    else
        @printf(io,
                "%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%d,%d,%d,%d,%d,%d\n",
                step, label, loc.lvl, loc.b, loc.lx, loc.ly, loc.lz,
                loc.gx, loc.gy, loc.gz, loc.shift,
                loc.rho, loc.ux, loc.uy, loc.uz, loc.umag, loc.wall_dist,
                loc.owns_wall_links ? 1 : 0, qlinks, loc.amr ? 1 : 0,
                loc.face_dx, loc.face_dy, loc.face_dz)
    end
end

function print_stability_extrema_diagnostics(grids, step::Int; csv_path::Union{Nothing,String}=nothing)
    rec = _stability_abort_extrema_records(grids)
    q_map_cache = Dict{Int,Any}()
    if csv_path !== nothing
        open(csv_path, "a") do io
            _append_stability_extremum_csv(io, step, "rho_min", rec.rho_min,
                                           _stability_extrema_q_link_count(grids, rec.rho_min, q_map_cache))
            _append_stability_extremum_csv(io, step, "rho_max", rec.rho_max,
                                           _stability_extrema_q_link_count(grids, rec.rho_max, q_map_cache))
            _append_stability_extremum_csv(io, step, "u_max", rec.u_max,
                                           _stability_extrema_q_link_count(grids, rec.u_max, q_map_cache))
        end
    end
    if (rec.u_max !== nothing && rec.u_max.umag > 0.98) ||
       (rec.rho_min !== nothing && rec.rho_min.rho < 0.45)
        println("          [Extrema] step=$step  " *
                _format_stability_extremum(grids, "rho_min", rec.rho_min, q_map_cache) * "; " *
                _format_stability_extremum(grids, "u_max", rec.u_max, q_map_cache))
    end
    return rec
end

function evaluate_stability_abort(grids, step::Int; wake_stats=nothing)
    if !STABILITY_ABORT_ENABLED
        return (abort=false, message="", mass=NaN, mass_rel=0.0)
    end

    reasons = String[]
    flow = _stability_abort_global_flow_stats(grids)
    if flow.n_fluid <= 0
        push!(reasons, "no active fluid cells")
    end
    if !isfinite(flow.rho_min) || !isfinite(flow.rho_max) || !isfinite(flow.u_max)
        push!(reasons, @sprintf("non-finite flow stats rho_min=%.6e rho_max=%.6e u_max=%.6e",
                                flow.rho_min, flow.rho_max, flow.u_max))
    end
    if STABILITY_ABORT_RHO_MIN > 0.0f0 && flow.rho_min < Float64(STABILITY_ABORT_RHO_MIN)
        push!(reasons, @sprintf("rho_min %.6f < %.6f", flow.rho_min, STABILITY_ABORT_RHO_MIN))
    end
    if STABILITY_ABORT_RHO_MAX > 0.0f0 && flow.rho_max > Float64(STABILITY_ABORT_RHO_MAX)
        push!(reasons, @sprintf("rho_max %.6f > %.6f", flow.rho_max, STABILITY_ABORT_RHO_MAX))
    end
    if STABILITY_ABORT_U_MAX > 0.0f0 && flow.u_max > Float64(STABILITY_ABORT_U_MAX)
        push!(reasons, @sprintf("u_max %.6f > %.6f", flow.u_max, STABILITY_ABORT_U_MAX))
    end

    mass = compute_total_active_mass(grids)
    mass_rel = 0.0
    if _STABILITY_ABORT_FIRST[]
        _STABILITY_ABORT_MASS_BASELINE[] = mass
        _STABILITY_ABORT_FIRST[] = false
    elseif _STABILITY_ABORT_MASS_BASELINE[] > 1e-12
        mass_rel = (mass - _STABILITY_ABORT_MASS_BASELINE[]) / _STABILITY_ABORT_MASS_BASELINE[]
        if STABILITY_ABORT_MASS_REL > 0.0f0 && abs(mass_rel) > Float64(STABILITY_ABORT_MASS_REL)
            push!(reasons, @sprintf("mass drift %.6e exceeds %.6e", mass_rel, STABILITY_ABORT_MASS_REL))
        end
    end
    if !isfinite(mass)
        push!(reasons, @sprintf("non-finite active mass %.6e", mass))
    end

    if wake_stats !== nothing && hasproperty(wake_stats, :active) && wake_stats.active
        shear_over = Float64(wake_stats.shear_overspeed_max)
        shear_frac = Float64(wake_stats.shear_overspeed_frac)
        pbase = Float64(wake_stats.base_pressure_rel_mean)
        if STABILITY_ABORT_WAKE_SHEAR_OVERSPEED > 0.0f0 &&
           shear_over > Float64(STABILITY_ABORT_WAKE_SHEAR_OVERSPEED)
            push!(reasons, @sprintf("wake shear overspeed %.6f > %.6f",
                                    shear_over, STABILITY_ABORT_WAKE_SHEAR_OVERSPEED))
        end
        if STABILITY_ABORT_WAKE_SHEAR_FRAC > 0.0f0 &&
           shear_frac > Float64(STABILITY_ABORT_WAKE_SHEAR_FRAC)
            push!(reasons, @sprintf("wake shear overspeed fraction %.6f > %.6f",
                                    shear_frac, STABILITY_ABORT_WAKE_SHEAR_FRAC))
        end
        if STABILITY_ABORT_BASE_PRESSURE_REL_MIN > 0.0f0 &&
           pbase < Float64(STABILITY_ABORT_BASE_PRESSURE_REL_MIN)
            push!(reasons, @sprintf("base pressure %.6f < %.6f",
                                    pbase, STABILITY_ABORT_BASE_PRESSURE_REL_MIN))
        end
        if STABILITY_ABORT_BASE_PRESSURE_REL_MAX > 0.0f0 &&
           pbase > Float64(STABILITY_ABORT_BASE_PRESSURE_REL_MAX)
            push!(reasons, @sprintf("base pressure %.6f > %.6f",
                                    pbase, STABILITY_ABORT_BASE_PRESSURE_REL_MAX))
        end
    end

    if !isempty(reasons)
        push!(reasons, _stability_abort_extrema_locations(grids))
    end

    msg = join(reasons, "; ")
    return (abort=!isempty(reasons), message=msg, mass=mass, mass_rel=mass_rel)
end

# ============================================================================
# Pass 190 — Compressible audit diagnostic (read-only, default-off)
# ============================================================================
# Integrates the conservation moments of the Frapolli two-population energy
# model over the active domain and emits a per-step CSV. Behavior is
# unchanged — this only reads state.
#
# Quantities (per `DIAG_FREQ` step):
#  - Total mass         M = Σ ρ · dV_level                           (drift vs t=0 baseline)
#  - Total momentum     J = Σ ρ u · dV_level   (vector, 3 components)
#  - Total energy       E = Σ (ρ·Cv·T + ½·ρ|u|²) · dV_level          (Frapolli total)
#  - Internal / kinetic split, KE fraction
#  - Per-level: |u|_max, |u|_mean, Ma_max, Ma_mean, frac(Ma>1.0), frac(Ma>1.2)
#  - Per-level: T_min, T_max, T_mean
#  - Global: T_min_negative_count   (positivity sentry on T recovery)
#
# Speed of sound on the lattice: cs² = 2/3 for D3Q39, 1/3 for D3Q27.
# Local sound speed in the Frapolli stack: a² = cs²·T → Ma = |u| / √(cs²·T).
# When the thermal DDF is off but the algebraic transonic pressure closure is
# active, the audit reuses the same effective T as the flow equilibrium. Purely
# athermal cases still fall back to T=1.
#
# Gate: COMPRESSIBLE_AUDIT_DIAGNOSTICS (advanced.diagnostics.compressible_audit).
# Default off. When off, nothing in this section ever runs.
# ============================================================================

# Stateful baselines (module-level refs, same pattern as _MASS_BASELINE above).
const _COMP_AUDIT_BASELINE_M  = Ref{Float64}(0.0)
const _COMP_AUDIT_BASELINE_E  = Ref{Float64}(0.0)
const _COMP_AUDIT_BASELINE_J  = Ref{NTuple{3,Float64}}((0.0, 0.0, 0.0))
const _COMP_AUDIT_FIRST       = Ref{Bool}(true)

# Per-level Mach / temperature / shock-cell summary, one row.
#
# Pass 197a adds the shock-cell counts at the end. The thresholds 0.15 and
# 0.30 for normalized density gradient |∇ρ|/ρ · dx are calibrated against
# the Ducros-style shock sensor used in DG / FV-RKDG. Cells with normalized
# gradient above 0.15 are classified "mild" shock candidates (rarefaction
# heads, contact discontinuities); above 0.30 they are "strong" shock
# candidates (true compressive jumps). The numbers are purely diagnostic
# in Pass 197a — they will gate the bulk-viscosity injection in the
# eventual Pass 197b only when COMP_AUDIT_SHOCK_THRESH is wired into that
# kernel, which has not happened yet.
struct CompAuditLevelStats
    n_cells::Int
    u_max::Float64
    u_mean::Float64
    ma_max::Float64
    ma_mean::Float64
    n_supersonic::Int       # cells with Ma > 1.0
    n_strong_shock::Int     # cells with Ma > 1.2
    T_min::Float64
    T_max::Float64
    T_mean::Float64
    n_T_negative::Int       # cells with T <= 0.5 — Frapolli recovery sentry
    n_shock_mild::Int       # Pass 197a: cells with |∇ρ|/ρ·dx > 0.15
    n_shock_strong::Int     # Pass 197a: cells with |∇ρ|/ρ·dx > 0.30
    drho_norm_max::Float64  # Pass 197a: max of |∇ρ|/ρ·dx on this level
end

const _COMP_AUDIT_SHOCK_MILD_THRESH   = 0.15
const _COMP_AUDIT_SHOCK_STRONG_THRESH = 0.30

# Total accumulator across all levels (volume-weighted).
struct CompAuditTotals
    mass::Float64
    jx::Float64
    jy::Float64
    jz::Float64
    energy_total::Float64
    energy_internal::Float64
    energy_kinetic::Float64
    n_cells_total::Int
end

# Compute everything in one CPU pass over each level. Reads ρ, u, T (when DDF
# is on), obstacle mask, and the covered-by-children mask. Volume-weights by
# level.dx^3. Cells covered by an active finer child are skipped to avoid
# double-counting the same physical volume.
function compute_compressible_audit(grids)
    n = length(grids)
    cs2_lat = LATTICE_TYPE == :d3q39 ? (2.0 / 3.0) : (1.0 / 3.0)
    cv      = Float64(DDF_CV)
    ddf_on  = DDF_ENABLED
    u_ref_audit = abs(Float64(U_INLET_CURRENT[])) > 0.0 ?
                  abs(Float64(U_INLET_CURRENT[])) :
                  abs(Float64(U_TARGET))

    per_level = Vector{CompAuditLevelStats}(undef, n)

    # Domain-wide accumulators (per-level summaries are filled inside the loop).
    total_mass = 0.0
    total_jx = 0.0; total_jy = 0.0; total_jz = 0.0
    total_E = 0.0; total_Eint = 0.0; total_Ekin = 0.0
    total_cells = 0

    @inbounds for lvl in 1:n
        level = grids[lvl]
        if length(level.active_block_coords) == 0
            per_level[lvl] = CompAuditLevelStats(0, 0.0, 0.0, 0.0, 0.0, 0, 0,
                                                 1.0, 1.0, 1.0, 0,
                                                 0, 0, 0.0)
            continue
        end

        rho_cpu = Array(level.rho)
        vel_cpu = Array(level.vel)
        obs_cpu = Array(level.obstacle)
        mapx    = Array(level.map_x)
        mapy    = Array(level.map_y)
        mapz    = Array(level.map_z)

        has_T = ddf_on && has_thermal(level)
        uses_effective_T = has_T || (EOS_THERMODYNAMIC && LATTICE_TYPE == :d3q39)
        temp_cpu = has_T ? Array(level.temperature) : nothing
        sponge_cpu = hasproperty(level, :sponge) ? Array(level.sponge) : nothing

        bs       = BLOCK_SIZE
        vol_cell = level.dx^3

        # Covered-by-children predicate (same convention as compute_total_active_mass).
        has_finer = lvl < n && length(grids[lvl+1].active_block_coords) > 0
        child_ptr = has_finer ? Array(grids[lvl+1].block_pointer) : nothing
        cdx = has_finer ? size(child_ptr, 1) : 0
        cdy = has_finer ? size(child_ptr, 2) : 0
        cdz = has_finer ? size(child_ptr, 3) : 0

        @inline function covered_by_children(gx::Int, gy::Int, gz::Int)
            has_finer || return false
            for sx in 0:1, sy in 0:1, sz in 0:1
                fgx = 2*gx - 1 + sx
                fgy = 2*gy - 1 + sy
                fgz = 2*gz - 1 + sz
                (fgx < 1 || fgy < 1 || fgz < 1) && return false
                cbx = (fgx - 1) ÷ bs + 1
                cby = (fgy - 1) ÷ bs + 1
                cbz = (fgz - 1) ÷ bs + 1
                (cbx > cdx || cby > cdy || cbz > cdz) && return false
                child_ptr[cbx, cby, cbz] == 0 && return false
            end
            true
        end

        n_cells_lvl = 0
        u_sum_lvl = 0.0
        u_max_lvl = 0.0
        ma_sum_lvl = 0.0
        ma_max_lvl = 0.0
        n_supersonic = 0
        n_strong_shock = 0
        T_min_lvl = Inf
        T_max_lvl = -Inf
        T_sum_lvl = 0.0
        n_T_neg = 0

        # Pass 197a: shock-cell sensor accumulators.
        # `|∇ρ|/ρ · dx` is dimensionless: dimensional ∇ρ ~ Δρ/dx and the
        # dx factor in the numerator cancels the per-cell dx in the
        # gradient stencil, so the level-independent threshold compares
        # gradient steepness in lattice units across all AMR levels.
        n_shock_mild   = 0
        n_shock_strong = 0
        drho_norm_max  = 0.0

        n_blocks = length(level.active_block_coords)
        for pb in 1:n_blocks
            pbx = Int(mapx[pb]); pby = Int(mapy[pb]); pbz = Int(mapz[pb])
            for lz in 1:bs, ly in 1:bs, lx in 1:bs
                obs_cpu[lx, ly, lz, pb] && continue
                gx = (pbx - 1)*bs + lx
                gy = (pby - 1)*bs + ly
                gz = (pbz - 1)*bs + lz
                covered_by_children(gx, gy, gz) && continue

                rho = Float64(rho_cpu[lx, ly, lz, pb])
                ux  = Float64(vel_cpu[lx, ly, lz, pb, 1])
                uy  = Float64(vel_cpu[lx, ly, lz, pb, 2])
                uz  = Float64(vel_cpu[lx, ly, lz, pb, 3])
                usq = ux*ux + uy*uy + uz*uz
                umag = sqrt(usq)
                sp = sponge_cpu === nothing ? 0.0 : Float64(sponge_cpu[lx, ly, lz, pb])
                T = has_T ? Float64(temp_cpu[lx, ly, lz, pb]) :
                    diagnostic_effective_temperature(rho, ux, uy, uz, NaN, sp, u_ref_audit)

                # Sound speed in lattice units: a = sqrt(cs²·T) for Frapolli.
                # T=1 ⇒ a = sqrt(cs²) (the lattice baseline).
                a2 = cs2_lat * max(T, 1e-6)
                ma = umag / sqrt(a2)

                # Energy: E = ρ·cv·T + ½·ρ|u|². When DDF is off (T=1, cv=2.5
                # by default), the "internal" component is a constant offset
                # — its drift is then identically zero, which the user
                # should expect.
                e_int = rho * cv * T
                e_kin = 0.5 * rho * usq
                e_tot = e_int + e_kin

                total_mass += rho * vol_cell
                total_jx   += rho * ux * vol_cell
                total_jy   += rho * uy * vol_cell
                total_jz   += rho * uz * vol_cell
                total_E    += e_tot * vol_cell
                total_Eint += e_int * vol_cell
                total_Ekin += e_kin * vol_cell
                total_cells += 1

                n_cells_lvl += 1
                u_sum_lvl += umag
                u_max_lvl = max(u_max_lvl, umag)
                ma_sum_lvl += ma
                ma_max_lvl = max(ma_max_lvl, ma)
                ma > 1.0 && (n_supersonic   += 1)
                ma > 1.2 && (n_strong_shock += 1)
                if uses_effective_T
                    T_min_lvl = min(T_min_lvl, T)
                    T_max_lvl = max(T_max_lvl, T)
                    T_sum_lvl += T
                    T <= 0.5 && (n_T_neg += 1)
                end

                # Pass 197a: same-block central-difference shock sensor.
                # Skip cells at block boundaries (block-edge halos require
                # neighbor_table indexing; one-sided differences inside the
                # block are accurate enough for a count statistic). Skip
                # also if the same-block neighbor is an obstacle.
                if 2 <= lx <= bs-1 && 2 <= ly <= bs-1 && 2 <= lz <= bs-1
                    if !(obs_cpu[lx+1, ly, lz, pb] || obs_cpu[lx-1, ly, lz, pb] ||
                         obs_cpu[lx, ly+1, lz, pb] || obs_cpu[lx, ly-1, lz, pb] ||
                         obs_cpu[lx, ly, lz+1, pb] || obs_cpu[lx, ly, lz-1, pb])
                        drho_dx = 0.5 * (Float64(rho_cpu[lx+1, ly, lz, pb]) -
                                          Float64(rho_cpu[lx-1, ly, lz, pb]))
                        drho_dy = 0.5 * (Float64(rho_cpu[lx, ly+1, lz, pb]) -
                                          Float64(rho_cpu[lx, ly-1, lz, pb]))
                        drho_dz = 0.5 * (Float64(rho_cpu[lx, ly, lz+1, pb]) -
                                          Float64(rho_cpu[lx, ly, lz-1, pb]))
                        grad_mag = sqrt(drho_dx*drho_dx + drho_dy*drho_dy + drho_dz*drho_dz)
                        drho_norm = grad_mag / max(rho, 1e-6)
                        drho_norm > drho_norm_max && (drho_norm_max = drho_norm)
                        if drho_norm > _COMP_AUDIT_SHOCK_MILD_THRESH
                            n_shock_mild += 1
                            drho_norm > _COMP_AUDIT_SHOCK_STRONG_THRESH && (n_shock_strong += 1)
                        end
                    end
                end
            end
        end

        if uses_effective_T && n_cells_lvl == 0
            T_min_lvl = 1.0; T_max_lvl = 1.0
        elseif !uses_effective_T
            T_min_lvl = 1.0; T_max_lvl = 1.0
            T_sum_lvl = Float64(n_cells_lvl)
        end

        per_level[lvl] = CompAuditLevelStats(
            n_cells_lvl,
            u_max_lvl,
            n_cells_lvl > 0 ? u_sum_lvl / n_cells_lvl : 0.0,
            ma_max_lvl,
            n_cells_lvl > 0 ? ma_sum_lvl / n_cells_lvl : 0.0,
            n_supersonic,
            n_strong_shock,
            T_min_lvl, T_max_lvl,
            n_cells_lvl > 0 ? T_sum_lvl / n_cells_lvl : 1.0,
            n_T_neg,
            n_shock_mild, n_shock_strong, drho_norm_max,
        )
    end

    totals = CompAuditTotals(total_mass, total_jx, total_jy, total_jz,
                              total_E, total_Eint, total_Ekin, total_cells)
    return totals, per_level
end


# Standard 11-column header plus a level-detail block (one set of columns per
# level, indexed L1..LN). N is fixed at the first call by the runtime
# resolution; the level set never changes mid-run on a given case.
function write_compressible_audit_csv_header(filepath::String, n_levels::Int)
    cols = String[
        "step",
        "total_mass", "mass_drift_abs", "mass_drift_rel",
        "jx_total", "jy_total", "jz_total",
        "jx_drift_abs", "jy_drift_abs", "jz_drift_abs",
        "energy_total", "energy_internal", "energy_kinetic",
        "energy_drift_abs", "energy_drift_rel", "kinetic_fraction",
    ]
    for lvl in 1:n_levels
        suf = "_L$(lvl)"
        push!(cols, "n_cells"*suf)
        push!(cols, "u_max"*suf)
        push!(cols, "u_mean"*suf)
        push!(cols, "ma_max"*suf)
        push!(cols, "ma_mean"*suf)
        push!(cols, "n_supersonic"*suf)
        push!(cols, "n_strong_shock"*suf)
        push!(cols, "T_min"*suf)
        push!(cols, "T_max"*suf)
        push!(cols, "T_mean"*suf)
        push!(cols, "n_T_negative"*suf)
        # Pass 197a appended columns
        push!(cols, "n_shock_mild"*suf)
        push!(cols, "n_shock_strong"*suf)
        push!(cols, "drho_norm_max"*suf)
    end
    open(filepath, "w") do io
        println(io, join(cols, ","))
    end
end

function _append_compressible_audit_csv_row(filepath::String, step::Int,
                                             totals::CompAuditTotals,
                                             per_level::Vector{CompAuditLevelStats},
                                             baseline_M::Float64, baseline_E::Float64,
                                             baseline_J::NTuple{3,Float64})
    M = totals.mass
    E = totals.energy_total
    Jx, Jy, Jz = totals.jx, totals.jy, totals.jz
    dM = M - baseline_M
    dE = E - baseline_E
    dJx = Jx - baseline_J[1]
    dJy = Jy - baseline_J[2]
    dJz = Jz - baseline_J[3]
    rel_M = baseline_M > 1e-12 ? dM / baseline_M : 0.0
    rel_E = abs(baseline_E) > 1e-12 ? dE / baseline_E : 0.0
    ke_frac = E > 1e-12 ? totals.energy_kinetic / E : 0.0
    open(filepath, "a") do io
        @printf(io, "%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e",
                step,
                M, dM, rel_M,
                Jx, Jy, Jz,
                dJx, dJy, dJz,
                E, totals.energy_internal, totals.energy_kinetic,
                dE, rel_E, ke_frac)
        for s in per_level
            @printf(io, ",%d,%.9e,%.9e,%.9e,%.9e,%d,%d,%.9e,%.9e,%.9e,%d,%d,%d,%.9e",
                    s.n_cells, s.u_max, s.u_mean, s.ma_max, s.ma_mean,
                    s.n_supersonic, s.n_strong_shock,
                    s.T_min, s.T_max, s.T_mean, s.n_T_negative,
                    s.n_shock_mild, s.n_shock_strong, s.drho_norm_max)
        end
        println(io)
    end
end

# Public entrypoint, called once per DIAG_FREQ step. First call seeds the
# baseline. Console line is intentionally one line so it doesn't drown the
# existing per-level stats block.
function print_compressible_audit(grids, step::Int; csv_path::Union{Nothing,String}=nothing)
    totals, per_level = compute_compressible_audit(grids)

    if _COMP_AUDIT_FIRST[]
        _COMP_AUDIT_BASELINE_M[] = totals.mass
        _COMP_AUDIT_BASELINE_E[] = totals.energy_total
        _COMP_AUDIT_BASELINE_J[] = (totals.jx, totals.jy, totals.jz)
        _COMP_AUDIT_FIRST[] = false
        @printf("          [CompAudit] step=%d  baseline  M=%.6e  E=%.6e  J=(%.3e,%.3e,%.3e)  cells=%d\n",
                step, totals.mass, totals.energy_total,
                totals.jx, totals.jy, totals.jz, totals.n_cells_total)
    else
        dM  = totals.mass - _COMP_AUDIT_BASELINE_M[]
        dE  = totals.energy_total - _COMP_AUDIT_BASELINE_E[]
        relM = _COMP_AUDIT_BASELINE_M[] > 1e-12 ? dM / _COMP_AUDIT_BASELINE_M[] : 0.0
        relE = abs(_COMP_AUDIT_BASELINE_E[]) > 1e-12 ? dE / _COMP_AUDIT_BASELINE_E[] : 0.0
        # Compose the per-level Ma_max digest (most-likely-to-spike level wins).
        ma_global = 0.0
        T_min_global = Inf
        n_super_global = 0
        n_T_neg_global = 0
        n_shock_mild_global = 0
        n_shock_strong_global = 0
        for s in per_level
            ma_global = max(ma_global, s.ma_max)
            T_min_global = min(T_min_global, s.T_min)
            n_super_global += s.n_supersonic
            n_T_neg_global += s.n_T_negative
            n_shock_mild_global   += s.n_shock_mild
            n_shock_strong_global += s.n_shock_strong
        end
        T_min_str = isfinite(T_min_global) ? @sprintf("%.4f", T_min_global) : "  n/a"
        @printf("          [CompAudit] step=%d  ΔM=%+.3e%%  ΔE=%+.3e%%  Ma_max=%.3f  T_min=%s  n_super=%d  n_T_neg=%d  shock(mild/strong)=%d/%d\n",
                step, 100.0 * relM, 100.0 * relE, ma_global, T_min_str,
                n_super_global, n_T_neg_global,
                n_shock_mild_global, n_shock_strong_global)
    end

    if csv_path !== nothing
        _append_compressible_audit_csv_row(csv_path, step, totals, per_level,
                                            _COMP_AUDIT_BASELINE_M[],
                                            _COMP_AUDIT_BASELINE_E[],
                                            _COMP_AUDIT_BASELINE_J[])
    end
    return totals, per_level
end

# ==============================================================================
# RAW-FIELD z-MIRROR RESIDUAL PROBE (2026-06-11, LUDWIG_MIRROR_PROBE=1)
# ------------------------------------------------------------------------------
# Measures the solver's z-parity directly on the GPU-resident rho/vel arrays —
# deliberately BYPASSING the slice pipeline, whose one-sided bilinear pixel
# stencil is documented (io_slice.jl ~L297) to render interface effects
# "visible only on the lower face, invisible on the upper face": every slice
# PNG and every seam/jump CSV inherits that measurement asymmetry, so none of
# them can prove or disprove a solver parity bug.
#
# For a setup that is exactly z-mirror-symmetric (Z200/Z201 probes: no
# obstacle, z-mirrored marker boxes), the field must satisfy
#   ux(z) = ux(z'),  uy(z) = uy(z'),  uz(z) = -uz(z'),  rho(z) = rho(z')
# with z' = nz_global+1-z. Reports per level: max/mean |Δux|, max |uz+uz'|,
# max |Δρ|, and the global z-cell of the max-|Δux| pair (localizes any
# violation). FP-noise-level residuals ⇒ solver parity-clean ⇒ the visual
# asymmetry is the renderer. Structural residuals ⇒ solver bug at the printed
# location. Obstacle cells are not excluded (use on marker-only cases).
# ==============================================================================
const _MIRROR_PROBE = get(ENV, "LUDWIG_MIRROR_PROBE", "0") == "1"
# LUDWIG_MIRROR_PROBE_FREQ=N: ALSO probe at every batch end whose step % N == 0
# (independent of the slice tick — with gpu async_depth: 1 this is per STEP).
# Used to catch the FIRST parity-violating write and identify the kernel.
const _MIRROR_PROBE_FREQ = parse(Int, get(ENV, "LUDWIG_MIRROR_PROBE_FREQ", "0"))

function dump_z_mirror_residual(grids, domain_nz_l1::Integer, step::Integer;
                                top_n::Int=3)
    println("    [MirrorProbe] step $step - raw-field z-mirror residual (slice pipeline bypassed)")
    for (lvl, level) in enumerate(grids)
        n_blocks = length(level.active_block_coords)
        n_blocks == 0 && continue
        bs = Int(BLOCK_SIZE)
        nzg = Int(domain_nz_l1) * (1 << (lvl - 1))
        rho = Array(level.rho)
        vel = Array(level.vel)
        bp = Array(level.block_pointer)
        coords = level.active_block_coords
        bzmax = size(bp, 3)
        max_duz = 0.0; max_drho = 0.0
        sum_dux = 0.0; npairs = 0
        # top-N |Δux| violators: (dux, gx, gy, gz, ux_lo_side, ux_hi_side)
        top = Vector{NTuple{6, Float64}}()
        @inbounds for b in 1:n_blocks
            bx, by, bz = coords[b]
            for lz in 1:bs, ly in 1:bs, lx in 1:bs
                gz = (Int(bz) - 1) * bs + lz
                gzm = nzg + 1 - gz
                gzm <= gz && continue              # each mirror pair once
                bzm = (gzm - 1) ÷ bs + 1
                bzm <= bzmax || continue
                bm = Int(bp[Int(bx), Int(by), bzm])
                bm > 0 || continue
                lzm = (gzm - 1) % bs + 1
                ux_lo = Float64(vel[lx, ly, lz, b, 1])
                ux_hi = Float64(vel[lx, ly, lzm, bm, 1])
                dux  = abs(ux_lo - ux_hi)
                duz  = abs(Float64(vel[lx, ly, lz, b, 3]) + Float64(vel[lx, ly, lzm, bm, 3]))
                drho = abs(Float64(rho[lx, ly, lz, b])    - Float64(rho[lx, ly, lzm, bm]))
                if length(top) < top_n || dux > top[end][1]
                    gx = (Int(bx) - 1) * bs + lx
                    gy = (Int(by) - 1) * bs + ly
                    push!(top, (dux, Float64(gx), Float64(gy), Float64(gz), ux_lo, ux_hi))
                    sort!(top; by=first, rev=true)
                    length(top) > top_n && pop!(top)
                end
                max_duz  = max(max_duz, duz)
                max_drho = max(max_drho, drho)
                sum_dux += dux
                npairs += 1
            end
        end
        npairs == 0 && continue
        t1 = top[1]
        @printf("      L%d: pairs=%d  max|dux|=%.3e @(gx=%d,gy=%d,gz=%d)  mean|dux|=%.3e  max|uz+uz'|=%.3e  max|drho|=%.3e\n",
                lvl, npairs, t1[1], Int(t1[2]), Int(t1[3]), Int(t1[4]),
                sum_dux / npairs, max_duz, max_drho)
        for (rank, t) in enumerate(top)
            t[1] <= 0.0 && break
            @printf("          #%d |dux|=%.3e @(%d,%d,%d)  ux(z_lo)=%+.6e  ux(z_hi)=%+.6e\n",
                    rank, t[1], Int(t[2]), Int(t[3]), Int(t[4]), t[5], t[6])
        end
    end
    return nothing
end
