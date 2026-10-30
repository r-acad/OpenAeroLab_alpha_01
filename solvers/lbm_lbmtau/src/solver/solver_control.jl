using KernelAbstractions
using CUDA
using Adapt

# (ET-AMR / Esoteric-Twist restriction surface removed 2026-06-24 with the D3Q19 retirement.)

@inline function should_snapshot_parent_state(level::BlockLevel, use_temporal_interp::Bool)
    return (use_temporal_interp && has_temporal_storage(level)) ||
           (INTERFACE_MACRO_TEMPORAL_INTERPOLATION && has_macro_temporal_storage(level))
end

@inline function should_snapshot_level_state(level::BlockLevel, has_children::Bool, use_temporal_interp::Bool)
    return (has_children && should_snapshot_parent_state(level, use_temporal_interp)) ||
           (AMR_SHELL_COUPLED_REFLUX && has_macro_temporal_storage(level))
end

@inline function amr_temporal_bias_scale()
    bias = INTERFACE_TEMPORAL_WEIGHT_BIAS
    bias == 0.0f0 && return 0.0f0

    target = abs(U_TARGET)
    progress = target > 0.0f0 ? clamp(abs(U_INLET_CURRENT[]) / target, 0.0f0, 1.0f0) : 1.0f0

    rise_start = INTERFACE_TEMPORAL_WEIGHT_BIAS_RISE_U_START
    rise_end = INTERFACE_TEMPORAL_WEIGHT_BIAS_RISE_U_END
    rise_scale = 1.0f0
    if rise_end > rise_start
        if progress <= rise_start
            rise_scale = 0.0f0
        elseif progress < rise_end
            s = (progress - rise_start) / (rise_end - rise_start)
            rise_scale = s * s * (3.0f0 - 2.0f0 * s)
        end
    end

    decay_start = INTERFACE_TEMPORAL_WEIGHT_BIAS_DECAY_U_START
    decay_end = INTERFACE_TEMPORAL_WEIGHT_BIAS_DECAY_U_END
    decay_scale = 1.0f0
    if decay_end > decay_start
        if progress >= decay_end
            decay_scale = 0.0f0
        elseif progress > decay_start
            s = (progress - decay_start) / (decay_end - decay_start)
            decay_smooth = s * s * (3.0f0 - 2.0f0 * s)
            decay_scale = 1.0f0 - decay_smooth
        end
    end

    return rise_scale * decay_scale
end

@inline amr_child_temporal_weight(base_weight::Float32) =
    clamp(base_weight + INTERFACE_TEMPORAL_WEIGHT_BIAS * amr_temporal_bias_scale(), 0.0f0, 1.0f0)

function recursive_step!(grids, current_lvl::Int, t_sub::Int,
                         parent_f, parent_rho, parent_vel, parent_ptr,
                         parent_f_old, parent_rho_old, parent_vel_old,
                         parent_tau::Float32, u_vel::Float32,
                         cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                         domain_nx::Int, domain_ny::Int, domain_nz::Int,
                         wall_model_active::Bool, c_wale_val::Float32, nu_sgs_bg::Float32,
                         inlet_turbulence::Float32, use_temporal_interp::Bool, sponge_blend_dist::Bool)

    if current_lvl > length(grids); return; end

    level = grids[current_lvl]

    # ── ESOTERIC PULL (AA-pattern) L1 path ──────────────────────────────────
    # When LUDWIG_D3Q19_AA_PATTERN is on (D3Q19 only), L1 runs the in-place ET
    # stream-collide on the single `f` buffer (no f_temp). Default OFF ⇒ this
    # branch is never taken and the ping-pong path below is byte-identical.
    #
    # SINGLE-LEVEL (length(grids)==1): the ET step is the whole timestep — early
    # return (the original Phase-1 path, validated).
    #
    # MULTI-LEVEL L1 (STAGE 2, 2026-06-18): the ET step replaces the ping-pong
    # collision, but L1 still has CHILDREN that must prolongate from L1's macros +
    # Π_neq and (later) restrict back. So instead of returning, we run the ET step,
    # set f_out/vel_out = level.f/level.vel (the single buffer is both in and out),
    # and FALL THROUGH to the existing children-recursion tail — skipping ONLY the
    # ping-pong buffer-select + perform_timestep_v2! (the `did_et_step` guard). The
    # ET step writes level.pi_neq (fused-Π-under-ET) so the children's halo reads it.
    # (ET/AA-pattern single-buffer L1 step removed 2026-06-24 with the D3Q19 retirement;
    #  the conventional ping-pong path below is now unconditional.)
    if iseven(t_sub)
        f_in, f_out = level.f, level.f_temp
        vel_in, vel_out = level.vel, level.vel_temp
    else
        f_in, f_out = level.f_temp, level.f
        vel_in, vel_out = level.vel_temp, level.vel
    end

    # Thermal DDF ping-pong (same parity as flow)
    local g_in_t, g_out_t
    ddf_active = DDF_ENABLED && has_thermal(level)
    if ddf_active
        if iseven(t_sub)
            g_in_t, g_out_t = level.g, level.g_temp
        else
            g_in_t, g_out_t = level.g_temp, level.g
        end
    end

    has_children = current_lvl < length(grids)

    if should_snapshot_level_state(level, has_children, use_temporal_interp) || LOCAL_LATTICE_SHIFT_ENABLED
        copy_to_old!(level, f_in, vel_in; g_current = ddf_active ? g_in_t : nothing)
    end

    # FUSED-Π (2026-06-13): the fused path writes the NEW level.pi_neq DURING
    # perform_timestep_v2!, so the pi_neq_old ← pi_neq rotation (which the
    # legacy refresh did BEFORE its recompute) must happen HERE, while pi_neq
    # still holds the PREVIOUS step's value. Non-leaf only (pi_neq_old is
    # consumed solely by the children's halo staging).
    if _d3q19_fuse_pineq_active() && has_children && has_pi_neq(level)
        copy_pineq_scratch_only!(level, get_backend(f_in))
    end

    perform_timestep_v2!(level,
                         parent_f, parent_rho, parent_vel, parent_ptr,
                         parent_f_old, parent_rho_old, parent_vel_old,
                         parent_tau,
                         f_out, f_in, vel_out, vel_in,
                         level.f_post_collision, u_vel,
                         cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                         domain_nx, domain_ny, domain_nz,
                         wall_model_active, c_wale_val, nu_sgs_bg,
                         t_sub, inlet_turbulence, 0.0f0, use_temporal_interp, sponge_blend_dist;
                         g_out = ddf_active ? g_out_t : nothing,
                         g_in = ddf_active ? g_in_t : nothing,
                         g_post_collision_arr = ddf_active ? level.g_post_collision : nothing,
                         parent_g = nothing,
                         parent_temperature = nothing,
                         parent_g_old = nothing,
                         parent_temperature_old = nothing,
                         parent_pi_neq = nothing,
                         parent_vel_thermal = nothing,
                         parent_vel_old_thermal = nothing,
                         parent_tau_g = 0.5f0)

    # Refresh persistent Π_neq field from post-collision populations. Always runs
    # when the field is allocated — whether for variable ν_sgs or for use as the
    # coarse↔fine stress carrier. Zero-cost when the field is a stub (has_pi_neq
    # returns false because length(pi_neq) == 6).
    # 2026-06-13: a LEAF level's scratch copy is skipped — pi_neq_old is read
    # only by the children's halo staging (bitwise: dead store elision).
    # FUSED-Π (2026-06-13): when fused, the collision kernels + SVV epilogue
    # already wrote level.pi_neq for every cell (collision = obstacle + owned
    # fluid; SVV = SVV-active fluid), and the scratch rotation already ran
    # PRE-timestep (above) — so the entire post-step refresh is skipped.
    if has_pi_neq(level) && !_d3q19_fuse_pineq_active()
        refresh_pi_neq!(level, f_out, cx_gpu, cy_gpu, cz_gpu, w_gpu, get_backend(f_out);
                        shift_x=Float32(LATTICE_SHIFT_X),
                        shift_y=Float32(LATTICE_SHIFT_Y),
                        shift_z=Float32(LATTICE_SHIFT_Z),
                        copy_scratch=has_children)
    end
    # C2/C4 quintic B-spline prolongation (2026-06-04): compute the parent's
    # B-spline macro coefficients ONCE per coarse step (here, after the parent
    # macros level.rho / vel_out are final and before children prolongate). The
    # children's fine halo eval then reads these coeffs via the compact 6-tap W
    # for a C4-continuous + exactly-2:1-conservative coarse→fine macro interp.
    # Amortized over all child substeps. Stub-gated (has_spline_coeff) → no-op when
    # the feature is off. Only meaningful when this level has children.
    if has_children && has_spline_coeff(level) && high_order_continuity_active(u_vel)
        # SCHEDULE GATE (2026-06-23): skip the B-spline prefilter while the high-
        # order path is inactive (e.g. :steady_state mode during the ramp) — saves
        # the prefilter cost during cheap-C0 spin-up. Gate on the current ramp
        # velocity u_vel via high_order_continuity_active.
        if _PHASE_TIMING; KernelAbstractions.synchronize(get_backend(f_out)); _ts0 = time(); end
        compute_spline_coeffs!(level, vel_out, get_backend(f_out))
        if _PHASE_TIMING; KernelAbstractions.synchronize(get_backend(f_out)); _phase_op_add!(_T_OP_SPLINE, time() - _ts0); end
    end

    # LEVER 1b (2026-06-20) — fill the CHILD level's prolonged-halo macro+Π cache
    # ONCE per coarse step, from THIS (parent) level's just-computed state. The
    # child's substeps then read the cache + scalar-lerp instead of re-running the
    # (D3Q19 LEVER-1b halo-hoist cache fill removed 2026-06-24 with the D3Q19 retirement.)
    # NOTE: the Π_neq filter is NOT applied globally here — that caused a
    # smoothing cascade across timesteps (collision reads smoothed Π →
    # produces slightly smoothed f → next refresh writes smoothed Π → filter
    # smooths AGAIN → ...), dissipating the local |Π_neq| variations that
    # variable_sigma / variable_nu_sgs use to differentiate shock/BL from
    # freestream. The filter is instead applied only AT the decimation point
    # (just before `restrict_fine_to_parent!` on the child level) — see the
    # filter_pi_neq! call inside the RESTRICT_FINE_TO_PARENT block below.
    # This matches Lagrava 2012 §3.3: filter f_neq AT the fine→coarse transfer,
    # not globally in the fine grid.

    # Option B: post-recovery T laplacian filter (AMR-interface banding fix).
    # No-op when THERMAL_FILTER_ENABLED=false (scratch buffer is a stub).
    if THERMAL_FILTER_ENABLED && ddf_active && has_thermal_filter(level)
        apply_t_filter!(level, THERMAL_FILTER_ALPHA, get_backend(f_out);
                        g_current=g_out_t, vel_current=vel_out, w_gpu=w_gpu)
    end

    if AMR_RUNTIME_FACE_BUDGET_DIAGNOSTICS || AMR_RUNTIME_FACE_BUDGET_REFLUX
        if current_lvl > 1
            accumulate_amr_runtime_face_budget!(
                grids[current_lvl - 1], level, current_lvl - 1;
                side = :child,
                dt_weight = Float64(level.dt),
                child_vel_current = vel_out,
            )
        end
        if has_children
            accumulate_amr_runtime_face_budget!(
                level, grids[current_lvl + 1], current_lvl;
                side = :parent,
                dt_weight = Float64(level.dt),
                parent_vel_current = vel_out,
            )
        end
    end

    if AMR_GPU_FACE_BUDGET_REFLUX || AMR_SHELL_PAIRED_REFLUX_BUDGET_MEAN_PRESERVING
        if current_lvl > 1
            accumulate_amr_gpu_x_face_budget!(
                grids[current_lvl - 1], level;
                side = :child,
                dt_weight = Float32(level.dt),
                child_vel_current = vel_out,
            )
        end
        if has_children
            accumulate_amr_gpu_x_face_budget!(
                level, grids[current_lvl + 1];
                side = :parent,
                dt_weight = Float32(level.dt),
                parent_vel_current = vel_out,
            )
        end
    end

    if AMR_POPULATION_STREAM_REGISTER
        if current_lvl > 1
            accumulate_amr_population_stream_register!(
                grids[current_lvl - 1], level;
                side = :child,
                dt_weight = Float32(level.dt),
                child_f_current = f_out,
                cx_arr = cx_gpu,
                cy_arr = cy_gpu,
                cz_arr = cz_gpu,
            )
        end
        if has_children
            accumulate_amr_population_stream_register!(
                level, grids[current_lvl + 1];
                side = :parent,
                dt_weight = Float32(level.dt),
                parent_f_current = f_out,
                cx_arr = cx_gpu,
                cy_arr = cy_gpu,
                cz_arr = cz_gpu,
            )
        end
    end

    if has_children
        recursive_step_temporal!(grids, current_lvl + 1, 2*t_sub,
                                 f_out, level.rho, vel_out, level.block_pointer,
                                 level.f_old, level.rho_old, level.vel_old,
                                 level.tau, amr_child_temporal_weight(0.0f0), u_vel,
                                 cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                                 domain_nx, domain_ny, domain_nz,
                                 wall_model_active, c_wale_val, nu_sgs_bg,
                                 inlet_turbulence, use_temporal_interp, sponge_blend_dist;
                                 parent_g_data = ddf_active ? g_out_t : nothing,
                                 parent_temperature_data = ddf_active ? level.temperature : nothing,
                                 parent_g_old_data = ddf_active ? level.g_old : nothing,
                                 parent_temperature_old_data = ddf_active ? level.temperature_old : nothing,
                                 parent_vel_for_thermal = ddf_active ? vel_out : nothing,
                                 parent_vel_old_for_thermal = ddf_active ? level.vel_old : nothing,
                                 parent_kinetic_energy_data = ddf_active ? level.kinetic_energy : nothing,
                                 parent_tau_g_val = ddf_active ? level.tau_g : 0.5f0,
                                 # Pass180: pass this level's `_older` snapshots as parent_*_older for the child
                                 parent_rho_older = length(level.rho_older) > 1 ? level.rho_older : nothing,
                                 parent_vel_older = length(level.vel_older) > 1 ? level.vel_older : nothing,
                                 parent_temperature_older = (ddf_active && length(level.temperature_older) > 1) ? level.temperature_older : nothing,
                                 # B1 P3: cubic Hermite temporal oldest snapshots
                                 parent_rho_oldest = length(level.rho_oldest) > 1 ? level.rho_oldest : nothing,
                                 parent_vel_oldest = length(level.vel_oldest) > 1 ? level.vel_oldest : nothing,
                                 parent_temperature_oldest = (ddf_active && length(level.temperature_oldest) > 1) ? level.temperature_oldest : nothing,
                                 # B1 P3 RK4-equivalent: quintic Lagrange temporal oldest2 snapshots
                                 parent_rho_oldest2 = length(level.rho_oldest2) > 1 ? level.rho_oldest2 : nothing,
                                 parent_vel_oldest2 = length(level.vel_oldest2) > 1 ? level.vel_oldest2 : nothing,
                                 parent_temperature_oldest2 = (ddf_active && length(level.temperature_oldest2) > 1) ? level.temperature_oldest2 : nothing)

        recursive_step_temporal!(grids, current_lvl + 1, 2*t_sub + 1,
                                 f_out, level.rho, vel_out, level.block_pointer,
                                 level.f_old, level.rho_old, level.vel_old,
                                 level.tau, amr_child_temporal_weight(0.5f0), u_vel,
                                 cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                                 domain_nx, domain_ny, domain_nz,
                                 wall_model_active, c_wale_val, nu_sgs_bg,
                                 inlet_turbulence, use_temporal_interp, sponge_blend_dist;
                                 parent_g_data = ddf_active ? g_out_t : nothing,
                                 parent_temperature_data = ddf_active ? level.temperature : nothing,
                                 parent_g_old_data = ddf_active ? level.g_old : nothing,
                                 parent_temperature_old_data = ddf_active ? level.temperature_old : nothing,
                                 parent_vel_for_thermal = ddf_active ? vel_out : nothing,
                                 parent_vel_old_for_thermal = ddf_active ? level.vel_old : nothing,
                                 parent_kinetic_energy_data = ddf_active ? level.kinetic_energy : nothing,
                                 parent_tau_g_val = ddf_active ? level.tau_g : 0.5f0,
                                 # Pass180: pass this level's `_older` snapshots as parent_*_older for the child
                                 parent_rho_older = length(level.rho_older) > 1 ? level.rho_older : nothing,
                                 parent_vel_older = length(level.vel_older) > 1 ? level.vel_older : nothing,
                                 parent_temperature_older = (ddf_active && length(level.temperature_older) > 1) ? level.temperature_older : nothing,
                                 # B1 P3: cubic Hermite temporal oldest snapshots
                                 parent_rho_oldest = length(level.rho_oldest) > 1 ? level.rho_oldest : nothing,
                                 parent_vel_oldest = length(level.vel_oldest) > 1 ? level.vel_oldest : nothing,
                                 parent_temperature_oldest = (ddf_active && length(level.temperature_oldest) > 1) ? level.temperature_oldest : nothing,
                                 # B1 P3 RK4-equivalent: quintic Lagrange temporal oldest2 snapshots
                                 parent_rho_oldest2 = length(level.rho_oldest2) > 1 ? level.rho_oldest2 : nothing,
                                 parent_vel_oldest2 = length(level.vel_oldest2) > 1 ? level.vel_oldest2 : nothing,
                                 parent_temperature_oldest2 = (ddf_active && length(level.temperature_oldest2) > 1) ? level.temperature_oldest2 : nothing)

        child_level = grids[current_lvl + 1]
        if AMR_RUNTIME_FACE_BUDGET_REFLUX
            apply_amr_runtime_face_budget_reflux!(
                level, child_level, current_lvl, f_out, vel_out,
                cx_gpu, cy_gpu, cz_gpu, w_gpu;
                alpha = AMR_RUNTIME_FACE_BUDGET_REFLUX_ALPHA,
                max_parent_level = AMR_RUNTIME_FACE_BUDGET_REFLUX_MAX_PARENT_LEVEL,
                x_only = AMR_RUNTIME_FACE_BUDGET_REFLUX_X_ONLY,
            )
        end
        if AMR_POPULATION_STREAM_REGISTER_APPLY
            apply_amr_population_stream_register_reflux!(
                level,
                current_lvl,
                f_out,
                vel_out,
                cx_gpu,
                cy_gpu,
                cz_gpu,
                w_gpu;
                alpha = AMR_POPULATION_STREAM_REGISTER_APPLY_ALPHA,
                max_parent_level = AMR_POPULATION_STREAM_REGISTER_APPLY_MAX_PARENT_LEVEL,
            )
            if has_pi_neq(level)
                refresh_pi_neq!(level, f_out, cx_gpu, cy_gpu, cz_gpu, w_gpu, get_backend(f_out);
                                shift_x=Float32(LATTICE_SHIFT_X),
                                shift_y=Float32(LATTICE_SHIFT_Y),
                                shift_z=Float32(LATTICE_SHIFT_Z))
            end
        end
        if AMR_POPULATION_STREAM_REGISTER_FACE_APPLY
            apply_amr_population_stream_xface_register_reflux!(
                level,
                current_lvl,
                f_out,
                vel_out,
                cx_gpu,
                cy_gpu,
                cz_gpu;
                alpha = AMR_POPULATION_STREAM_REGISTER_FACE_APPLY_ALPHA,
                max_parent_level = AMR_POPULATION_STREAM_REGISTER_FACE_APPLY_MAX_PARENT_LEVEL,
            )
            if has_pi_neq(level)
                refresh_pi_neq!(level, f_out, cx_gpu, cy_gpu, cz_gpu, w_gpu, get_backend(f_out);
                                shift_x=Float32(LATTICE_SHIFT_X),
                                shift_y=Float32(LATTICE_SHIFT_Y),
                                shift_z=Float32(LATTICE_SHIFT_Z))
            end
        end
        if AMR_POPULATION_STREAM_REGISTER_XFACE_HIGHPASS_APPLY
            apply_amr_population_stream_xface_highpass_register_reflux!(
                level,
                current_lvl,
                f_out,
                vel_out,
                cx_gpu,
                cy_gpu,
                cz_gpu;
                alpha = AMR_POPULATION_STREAM_REGISTER_XFACE_HIGHPASS_APPLY_ALPHA,
                max_parent_level = AMR_POPULATION_STREAM_REGISTER_XFACE_HIGHPASS_APPLY_MAX_PARENT_LEVEL,
            )
            if has_pi_neq(level)
                refresh_pi_neq!(level, f_out, cx_gpu, cy_gpu, cz_gpu, w_gpu, get_backend(f_out);
                                shift_x=Float32(LATTICE_SHIFT_X),
                                shift_y=Float32(LATTICE_SHIFT_Y),
                                shift_z=Float32(LATTICE_SHIFT_Z))
            end
        end
        if AMR_POPULATION_STREAM_REGISTER_XFACE_GLOBAL_APPLY
            apply_amr_population_stream_xface_global_register_reflux!(
                level,
                current_lvl,
                f_out,
                vel_out,
                cx_gpu,
                cy_gpu,
                cz_gpu;
                alpha = AMR_POPULATION_STREAM_REGISTER_XFACE_GLOBAL_APPLY_ALPHA,
                max_parent_level = AMR_POPULATION_STREAM_REGISTER_XFACE_GLOBAL_APPLY_MAX_PARENT_LEVEL,
            )
            if has_pi_neq(level)
                refresh_pi_neq!(level, f_out, cx_gpu, cy_gpu, cz_gpu, w_gpu, get_backend(f_out);
                                shift_x=Float32(LATTICE_SHIFT_X),
                                shift_y=Float32(LATTICE_SHIFT_Y),
                                shift_z=Float32(LATTICE_SHIFT_Z))
            end
        end

        # Fine→coarse restriction (conventional). (ET-aware restriction removed
        # 2026-06-24 with the D3Q19 retirement.)
        if RESTRICT_FINE_TO_PARENT
            if AMR_RESTRICTION_PI_NEQ_FILTER
                filter_pi_neq!(child_level, get_backend(f_out))
                filter_pi_neq!(child_level, get_backend(f_out))
            end
            restrict_fine_to_parent!(level, f_out, vel_out, child_level,
                                      cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu;
                                      parent_g_current = ddf_active ? g_out_t : nothing,
                                      child_f_current = child_level.f,
                                      child_g_current = (DDF_ENABLED && has_thermal(child_level)) ? child_level.g : nothing,
                                      current_step = t_sub)
        end
    end
end

function recursive_step_temporal!(grids, current_lvl::Int, t_sub::Int,
                                  parent_f, parent_rho, parent_vel, parent_ptr,
                                  parent_f_old, parent_rho_old, parent_vel_old,
                                  parent_tau::Float32, temporal_weight::Float32, u_vel::Float32,
                                  cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                                  domain_nx::Int, domain_ny::Int, domain_nz::Int,
                                  wall_model_active::Bool, c_wale_val::Float32, nu_sgs_bg::Float32,
                                  inlet_turbulence::Float32, use_temporal_interp::Bool, sponge_blend_dist::Bool;
                                  # Thermal DDF parent data (Phase 2/4b)
                                  parent_g_data=nothing, parent_temperature_data=nothing,
                                  parent_g_old_data=nothing, parent_temperature_old_data=nothing,
                                  parent_vel_for_thermal=nothing, parent_vel_old_for_thermal=nothing,
                                  parent_kinetic_energy_data=nothing,
                                  parent_tau_g_val::Float32=0.5f0,
                                  # Pass180 (Tier A item A2 — quadratic temporal):
                                  # Older parent snapshots (1 parent step before _old).
                                  parent_rho_older=nothing, parent_vel_older=nothing,
                                  parent_temperature_older=nothing,
                                  # B1 P3 (2026-05-28 — cubic Hermite temporal):
                                  # Oldest parent snapshots (2 parent steps before _old).
                                  parent_rho_oldest=nothing, parent_vel_oldest=nothing,
                                  parent_temperature_oldest=nothing,
                                  # B1 P3 RK4-equivalent (quintic Lagrange temporal):
                                  # Oldest2 parent snapshots (3 parent steps before _old).
                                  parent_rho_oldest2=nothing, parent_vel_oldest2=nothing,
                                  parent_temperature_oldest2=nothing)

    if current_lvl > length(grids); return; end

    level = grids[current_lvl]

    # ── ESOTERIC PULL refined-level path (STAGE 3, 2026-06-18) ─────────────────
    # (ET/AA-pattern single-buffer path removed 2026-06-24 with the D3Q19 retirement;
    #  conventional ping-pong buffer-select.)
    if iseven(t_sub)
        f_in, f_out = level.f, level.f_temp
        vel_in, vel_out = level.vel, level.vel_temp
    else
        f_in, f_out = level.f_temp, level.f
        vel_in, vel_out = level.vel_temp, level.vel
    end

    # Thermal DDF ping-pong
    local g_in_t, g_out_t
    ddf_active = DDF_ENABLED && has_thermal(level)
    if ddf_active
        if iseven(t_sub)
            g_in_t, g_out_t = level.g, level.g_temp
        else
            g_in_t, g_out_t = level.g_temp, level.g
        end
    end

    has_children = current_lvl < length(grids)

    if should_snapshot_level_state(level, has_children, use_temporal_interp) || LOCAL_LATTICE_SHIFT_ENABLED
        copy_to_old!(level, f_in, vel_in; g_current = ddf_active ? g_in_t : nothing)
    end

    # FUSED-Π (2026-06-13): the fused path writes the NEW level.pi_neq DURING
    # perform_timestep_v2!, so the pi_neq_old ← pi_neq rotation (which the
    # legacy refresh did BEFORE its recompute) must happen HERE, while pi_neq
    # still holds the PREVIOUS step's value. Non-leaf only (pi_neq_old is
    # consumed solely by the children's halo staging).
    if _d3q19_fuse_pineq_active() && has_children && has_pi_neq(level)
        copy_pineq_scratch_only!(level, get_backend(f_in))
    end

    perform_timestep_v2!(level,
                         parent_f, parent_rho, parent_vel, parent_ptr,
                         parent_f_old, parent_rho_old, parent_vel_old,
                         parent_tau,
                         f_out, f_in, vel_out, vel_in,
                         level.f_post_collision, u_vel,
                         cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                         domain_nx, domain_ny, domain_nz,
                         wall_model_active, c_wale_val, nu_sgs_bg,
                         t_sub, inlet_turbulence, temporal_weight, use_temporal_interp, sponge_blend_dist;
                         g_out = ddf_active ? g_out_t : nothing,
                         g_in = ddf_active ? g_in_t : nothing,
                         g_post_collision_arr = ddf_active ? level.g_post_collision : nothing,
                         parent_g = parent_g_data,
                         parent_temperature = parent_temperature_data,
                         parent_g_old = parent_g_old_data,
                         parent_temperature_old = parent_temperature_old_data,
                         parent_pi_neq = parent_ptr !== nothing ? grids[current_lvl - 1].pi_neq : nothing,
                         parent_pi_neq_old = parent_ptr !== nothing ? grids[current_lvl - 1].pi_neq_scratch : nothing,
                         # C2/C4 B-spline: parent level's precomputed spline coeffs.
                         parent_rho_spline_coeff = (parent_ptr !== nothing && has_spline_coeff(grids[current_lvl - 1])) ? grids[current_lvl - 1].rho_spline_coeff : nothing,
                         parent_vel_spline_coeff = (parent_ptr !== nothing && has_spline_coeff(grids[current_lvl - 1])) ? grids[current_lvl - 1].vel_spline_coeff : nothing,
                         parent_vel_thermal = parent_vel_for_thermal,
                         parent_vel_old_thermal = parent_vel_old_for_thermal,
                         parent_kinetic_energy = parent_kinetic_energy_data,
                         parent_tau_g = parent_tau_g_val,
                         # Pass180: forward older parent snapshots for quadratic temporal interp
                         parent_rho_older = parent_rho_older,
                         parent_vel_older = parent_vel_older,
                         parent_temperature_older = parent_temperature_older,
                         # B1 P3: forward oldest parent snapshots for cubic Hermite temporal interp
                         parent_rho_oldest = parent_rho_oldest,
                         parent_vel_oldest = parent_vel_oldest,
                         parent_temperature_oldest = parent_temperature_oldest,
                         # B1 P3 RK4-equivalent: forward oldest2 snapshots for quintic Lagrange temporal interp
                         parent_rho_oldest2 = parent_rho_oldest2,
                         parent_vel_oldest2 = parent_vel_oldest2,
                         parent_temperature_oldest2 = parent_temperature_oldest2)

    # Refresh persistent Π_neq (same rule as recursive_step! — always when allocated).
    # Filter is NOT applied globally here (see detailed note in recursive_step!).
    # 2026-06-13: leaf scratch-copy skip; fused-Π skip (see recursive_step!).
    if has_pi_neq(level) && !_d3q19_fuse_pineq_active()
        refresh_pi_neq!(level, f_out, cx_gpu, cy_gpu, cz_gpu, w_gpu, get_backend(f_out);
                        shift_x=Float32(LATTICE_SHIFT_X),
                        shift_y=Float32(LATTICE_SHIFT_Y),
                        shift_z=Float32(LATTICE_SHIFT_Z),
                        copy_scratch=has_children)
    end
    # (ET mass/halo debug probes removed 2026-06-24 with the D3Q19 retirement.)
    # C2/C4 quintic B-spline prolongation (2026-06-04) — see recursive_step!.
    # Compute parent B-spline macro coefficients once per coarse step (this level
    # below L1), consumed by child prolongation. Stub-gated (no-op when off).
    if has_children && has_spline_coeff(level) && high_order_continuity_active(u_vel)
        # SCHEDULE GATE (2026-06-23): skip the B-spline prefilter while the high-
        # order path is inactive (e.g. :steady_state mode during the ramp) — saves
        # the prefilter cost during cheap-C0 spin-up. Gate on the current ramp
        # velocity u_vel via high_order_continuity_active.
        if _PHASE_TIMING; KernelAbstractions.synchronize(get_backend(f_out)); _ts0 = time(); end
        compute_spline_coeffs!(level, vel_out, get_backend(f_out))
        if _PHASE_TIMING; KernelAbstractions.synchronize(get_backend(f_out)); _phase_op_add!(_T_OP_SPLINE, time() - _ts0); end
    end

    # LEVER 1b (2026-06-20) — fill the CHILD level's prolonged-halo macro+Π cache
    # ONCE per coarse step, from THIS (parent) level's just-computed state. The
    # child's substeps then read the cache + scalar-lerp instead of re-running the
    # (D3Q19 LEVER-1b halo-hoist cache fill removed 2026-06-24 with the D3Q19 retirement.)

    # Option B: post-recovery T laplacian filter (same rule as recursive_step!).
    if THERMAL_FILTER_ENABLED && ddf_active && has_thermal_filter(level)
        apply_t_filter!(level, THERMAL_FILTER_ALPHA, get_backend(f_out);
                        g_current=g_out_t, vel_current=vel_out, w_gpu=w_gpu)
    end

    if AMR_RUNTIME_FACE_BUDGET_DIAGNOSTICS || AMR_RUNTIME_FACE_BUDGET_REFLUX
        if current_lvl > 1
            accumulate_amr_runtime_face_budget!(
                grids[current_lvl - 1], level, current_lvl - 1;
                side = :child,
                dt_weight = Float64(level.dt),
                child_vel_current = vel_out,
            )
        end
        if has_children
            accumulate_amr_runtime_face_budget!(
                level, grids[current_lvl + 1], current_lvl;
                side = :parent,
                dt_weight = Float64(level.dt),
                parent_vel_current = vel_out,
            )
        end
    end

    if AMR_GPU_FACE_BUDGET_REFLUX || AMR_SHELL_PAIRED_REFLUX_BUDGET_MEAN_PRESERVING
        if current_lvl > 1
            accumulate_amr_gpu_x_face_budget!(
                grids[current_lvl - 1], level;
                side = :child,
                dt_weight = Float32(level.dt),
                child_vel_current = vel_out,
            )
        end
        if has_children
            accumulate_amr_gpu_x_face_budget!(
                level, grids[current_lvl + 1];
                side = :parent,
                dt_weight = Float32(level.dt),
                parent_vel_current = vel_out,
            )
        end
    end

    if AMR_POPULATION_STREAM_REGISTER
        if current_lvl > 1
            accumulate_amr_population_stream_register!(
                grids[current_lvl - 1], level;
                side = :child,
                dt_weight = Float32(level.dt),
                child_f_current = f_out,
                cx_arr = cx_gpu,
                cy_arr = cy_gpu,
                cz_arr = cz_gpu,
            )
        end
        if has_children
            accumulate_amr_population_stream_register!(
                level, grids[current_lvl + 1];
                side = :parent,
                dt_weight = Float32(level.dt),
                parent_f_current = f_out,
                cx_arr = cx_gpu,
                cy_arr = cy_gpu,
                cz_arr = cz_gpu,
            )
        end
    end

    if has_children
        recursive_step_temporal!(grids, current_lvl + 1, 2*t_sub,
                                 f_out, level.rho, vel_out, level.block_pointer,
                                 level.f_old, level.rho_old, level.vel_old,
                                 level.tau, amr_child_temporal_weight(0.0f0), u_vel,
                                 cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                                 domain_nx, domain_ny, domain_nz,
                                 wall_model_active, c_wale_val, nu_sgs_bg,
                                 inlet_turbulence, use_temporal_interp, sponge_blend_dist;
                                 parent_g_data = ddf_active ? g_out_t : nothing,
                                 parent_temperature_data = ddf_active ? level.temperature : nothing,
                                 parent_g_old_data = ddf_active ? level.g_old : nothing,
                                 parent_temperature_old_data = ddf_active ? level.temperature_old : nothing,
                                 parent_vel_for_thermal = ddf_active ? vel_out : nothing,
                                 parent_vel_old_for_thermal = ddf_active ? level.vel_old : nothing,
                                 parent_kinetic_energy_data = ddf_active ? level.kinetic_energy : nothing,
                                 parent_tau_g_val = ddf_active ? level.tau_g : 0.5f0,
                                 # Pass180: pass this level's `_older` snapshots as parent_*_older for the child
                                 parent_rho_older = length(level.rho_older) > 1 ? level.rho_older : nothing,
                                 parent_vel_older = length(level.vel_older) > 1 ? level.vel_older : nothing,
                                 parent_temperature_older = (ddf_active && length(level.temperature_older) > 1) ? level.temperature_older : nothing,
                                 # B1 P3: cubic Hermite temporal oldest snapshots
                                 parent_rho_oldest = length(level.rho_oldest) > 1 ? level.rho_oldest : nothing,
                                 parent_vel_oldest = length(level.vel_oldest) > 1 ? level.vel_oldest : nothing,
                                 parent_temperature_oldest = (ddf_active && length(level.temperature_oldest) > 1) ? level.temperature_oldest : nothing,
                                 # B1 P3 RK4-equivalent: quintic Lagrange temporal oldest2 snapshots
                                 parent_rho_oldest2 = length(level.rho_oldest2) > 1 ? level.rho_oldest2 : nothing,
                                 parent_vel_oldest2 = length(level.vel_oldest2) > 1 ? level.vel_oldest2 : nothing,
                                 parent_temperature_oldest2 = (ddf_active && length(level.temperature_oldest2) > 1) ? level.temperature_oldest2 : nothing)

        recursive_step_temporal!(grids, current_lvl + 1, 2*t_sub + 1,
                                 f_out, level.rho, vel_out, level.block_pointer,
                                 level.f_old, level.rho_old, level.vel_old,
                                 level.tau, amr_child_temporal_weight(0.5f0), u_vel,
                                 cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                                 domain_nx, domain_ny, domain_nz,
                                 wall_model_active, c_wale_val, nu_sgs_bg,
                                 inlet_turbulence, use_temporal_interp, sponge_blend_dist;
                                 parent_g_data = ddf_active ? g_out_t : nothing,
                                 parent_temperature_data = ddf_active ? level.temperature : nothing,
                                 parent_g_old_data = ddf_active ? level.g_old : nothing,
                                 parent_temperature_old_data = ddf_active ? level.temperature_old : nothing,
                                 parent_vel_for_thermal = ddf_active ? vel_out : nothing,
                                 parent_vel_old_for_thermal = ddf_active ? level.vel_old : nothing,
                                 parent_kinetic_energy_data = ddf_active ? level.kinetic_energy : nothing,
                                 parent_tau_g_val = ddf_active ? level.tau_g : 0.5f0,
                                 # Pass180: pass this level's `_older` snapshots as parent_*_older for the child
                                 parent_rho_older = length(level.rho_older) > 1 ? level.rho_older : nothing,
                                 parent_vel_older = length(level.vel_older) > 1 ? level.vel_older : nothing,
                                 parent_temperature_older = (ddf_active && length(level.temperature_older) > 1) ? level.temperature_older : nothing,
                                 # B1 P3: cubic Hermite temporal oldest snapshots
                                 parent_rho_oldest = length(level.rho_oldest) > 1 ? level.rho_oldest : nothing,
                                 parent_vel_oldest = length(level.vel_oldest) > 1 ? level.vel_oldest : nothing,
                                 parent_temperature_oldest = (ddf_active && length(level.temperature_oldest) > 1) ? level.temperature_oldest : nothing,
                                 # B1 P3 RK4-equivalent: quintic Lagrange temporal oldest2 snapshots
                                 parent_rho_oldest2 = length(level.rho_oldest2) > 1 ? level.rho_oldest2 : nothing,
                                 parent_vel_oldest2 = length(level.vel_oldest2) > 1 ? level.vel_oldest2 : nothing,
                                 parent_temperature_oldest2 = (ddf_active && length(level.temperature_oldest2) > 1) ? level.temperature_oldest2 : nothing)

        child_level = grids[current_lvl + 1]
        if AMR_RUNTIME_FACE_BUDGET_REFLUX
            apply_amr_runtime_face_budget_reflux!(
                level, child_level, current_lvl, f_out, vel_out,
                cx_gpu, cy_gpu, cz_gpu, w_gpu;
                alpha = AMR_RUNTIME_FACE_BUDGET_REFLUX_ALPHA,
                max_parent_level = AMR_RUNTIME_FACE_BUDGET_REFLUX_MAX_PARENT_LEVEL,
                x_only = AMR_RUNTIME_FACE_BUDGET_REFLUX_X_ONLY,
            )
        end

        # Fine→coarse restriction (conventional). (ET-aware restriction removed
        # 2026-06-24 with the D3Q19 retirement.)
        if RESTRICT_FINE_TO_PARENT
            # Decimation f_neq filter — gated on AMR_RESTRICTION_PI_NEQ_FILTER.
            if AMR_POPULATION_STREAM_REGISTER_APPLY
                apply_amr_population_stream_register_reflux!(
                    level,
                    current_lvl,
                    f_out,
                    vel_out,
                    cx_gpu,
                    cy_gpu,
                    cz_gpu,
                    w_gpu;
                    alpha = AMR_POPULATION_STREAM_REGISTER_APPLY_ALPHA,
                    max_parent_level = AMR_POPULATION_STREAM_REGISTER_APPLY_MAX_PARENT_LEVEL,
                )
                if has_pi_neq(level)
                    refresh_pi_neq!(level, f_out, cx_gpu, cy_gpu, cz_gpu, w_gpu, get_backend(f_out);
                                    shift_x=Float32(LATTICE_SHIFT_X),
                                    shift_y=Float32(LATTICE_SHIFT_Y),
                                    shift_z=Float32(LATTICE_SHIFT_Z))
                end
            end
            if AMR_POPULATION_STREAM_REGISTER_FACE_APPLY
                apply_amr_population_stream_xface_register_reflux!(
                    level,
                    current_lvl,
                    f_out,
                    vel_out,
                    cx_gpu,
                    cy_gpu,
                    cz_gpu;
                    alpha = AMR_POPULATION_STREAM_REGISTER_FACE_APPLY_ALPHA,
                    max_parent_level = AMR_POPULATION_STREAM_REGISTER_FACE_APPLY_MAX_PARENT_LEVEL,
                )
                if has_pi_neq(level)
                    refresh_pi_neq!(level, f_out, cx_gpu, cy_gpu, cz_gpu, w_gpu, get_backend(f_out);
                                    shift_x=Float32(LATTICE_SHIFT_X),
                                    shift_y=Float32(LATTICE_SHIFT_Y),
                                    shift_z=Float32(LATTICE_SHIFT_Z))
                end
            end
            if AMR_POPULATION_STREAM_REGISTER_XFACE_HIGHPASS_APPLY
                apply_amr_population_stream_xface_highpass_register_reflux!(
                    level,
                    current_lvl,
                    f_out,
                    vel_out,
                    cx_gpu,
                    cy_gpu,
                    cz_gpu;
                    alpha = AMR_POPULATION_STREAM_REGISTER_XFACE_HIGHPASS_APPLY_ALPHA,
                    max_parent_level = AMR_POPULATION_STREAM_REGISTER_XFACE_HIGHPASS_APPLY_MAX_PARENT_LEVEL,
                )
                if has_pi_neq(level)
                    refresh_pi_neq!(level, f_out, cx_gpu, cy_gpu, cz_gpu, w_gpu, get_backend(f_out);
                                    shift_x=Float32(LATTICE_SHIFT_X),
                                    shift_y=Float32(LATTICE_SHIFT_Y),
                                    shift_z=Float32(LATTICE_SHIFT_Z))
                end
            end
            if AMR_POPULATION_STREAM_REGISTER_XFACE_GLOBAL_APPLY
                apply_amr_population_stream_xface_global_register_reflux!(
                    level,
                    current_lvl,
                    f_out,
                    vel_out,
                    cx_gpu,
                    cy_gpu,
                    cz_gpu;
                    alpha = AMR_POPULATION_STREAM_REGISTER_XFACE_GLOBAL_APPLY_ALPHA,
                    max_parent_level = AMR_POPULATION_STREAM_REGISTER_XFACE_GLOBAL_APPLY_MAX_PARENT_LEVEL,
                )
                if has_pi_neq(level)
                    refresh_pi_neq!(level, f_out, cx_gpu, cy_gpu, cz_gpu, w_gpu, get_backend(f_out);
                                    shift_x=Float32(LATTICE_SHIFT_X),
                                    shift_y=Float32(LATTICE_SHIFT_Y),
                                    shift_z=Float32(LATTICE_SHIFT_Z))
                end
            end
            # Default OFF per Dorschner 2016 (entropic LBM doesn't need it) and
            # Schukmann 2025 (the filter INCREASES vc-ov OASPL by 6.5 dB).
            if AMR_RESTRICTION_PI_NEQ_FILTER
                filter_pi_neq!(child_level, get_backend(f_out))
                filter_pi_neq!(child_level, get_backend(f_out))
            end
            restrict_fine_to_parent!(level, f_out, vel_out, child_level,
                                      cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu;
                                      parent_g_current = ddf_active ? g_out_t : nothing,
                                      child_f_current = child_level.f,
                                      child_g_current = (DDF_ENABLED && has_thermal(child_level)) ? child_level.g : nothing,
                                      current_step = t_sub)
        end
    end
end

function execute_timestep_batch!(grids, t_start::Int, batch_size::Int, u_curr::Float32,
                                 cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                                 domain_nx::Int, domain_ny::Int, domain_nz::Int,
                                 wall_model_active::Bool, c_wale_val::Float32, nu_sgs_bg::Float32,
                                 inlet_turbulence::Float32, use_temporal_interp::Bool, sponge_blend_dist::Bool)
    backend = get_backend(grids[1].rho)

    for t_offset in 0:(batch_size-1)
        t = t_start + t_offset
        if AMR_RUNTIME_FACE_BUDGET_DIAGNOSTICS || AMR_RUNTIME_FACE_BUDGET_REFLUX
            reset_amr_runtime_face_budget!()
        end
        if AMR_GPU_FACE_BUDGET_REFLUX ||
           AMR_SHELL_PAIRED_REFLUX_BUDGET_MEAN_PRESERVING ||
           AMR_POPULATION_STREAM_REGISTER
            reset_amr_gpu_face_budgets!(grids)
        end
        recursive_step!(grids, 1, t,
                        nothing, nothing, nothing, nothing,
                        nothing, nothing, nothing,
                        Float32(0.5), u_curr,
                        cx_gpu, cy_gpu, cz_gpu, w_gpu, opp_gpu, mirror_y_gpu, mirror_z_gpu,
                        domain_nx, domain_ny, domain_nz,
                        wall_model_active, c_wale_val, nu_sgs_bg,
                        inlet_turbulence, use_temporal_interp, sponge_blend_dist)
    end

    KernelAbstractions.synchronize(backend)
end

# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-GPU: Strategy C — Replicated Coarse + Partitioned Fine
# ═══════════════════════════════════════════════════════════════════════════════

function materialize_current_state_for_io!(grids, step::Int)
    # The timestepper ping-pongs f/vel. Diagnostics, slices, and summaries read
    # the canonical arrays. At an even global step only level 1's current state is
    # in temp storage; refined levels finish their final subcycled step on the
    # canonical buffers. The next odd level-1 step still reads f_temp/vel_temp, so
    # this is observational and does not advance or rewind the solver state.
    iseven(step) || return
    isempty(grids) && return
    # ★ ET GATE (2026-06-18): the Esoteric-Pull (AA-pattern) path is SINGLE-BUFFER —
    # it streams in place on level.f and NEVER writes level.f_temp (which is left
    # frozen at the one-shot initial encode = the rest IC). The ping-pong IO
    # (ET/AA-pattern single-buffer L1 skip removed 2026-06-24 with the D3Q19 retirement.)
    level = grids[1]
    if length(level.active_block_coords) > 0
        copyto!(level.f, level.f_temp)
        copyto!(level.vel, level.vel_temp)
        if DDF_ENABLED && has_thermal(level)
            copyto!(level.g, level.g_temp)
        end
    end
    KernelAbstractions.synchronize(get_backend(grids[1].rho))
end

"""
    run_level_on_gpu!(gpu_idx, mgpu, level_idx, t_sub, parent_args...; kwargs...)

Run perform_timestep_v2! for a single level on a specific GPU, with optional
block_offset and ndrange_blocks for partitioned execution.
"""
function run_level_on_gpu!(gpu_idx::Int, mgpu::MultiGPUState, level_idx::Int,
                           t_sub::Int,
                           parent_f, parent_rho, parent_vel, parent_ptr,
                           parent_f_old, parent_rho_old, parent_vel_old,
                           parent_tau::Float32, u_vel::Float32,
                           domain_nx::Int, domain_ny::Int, domain_nz::Int,
                           wall_model_active::Bool, c_wale_val::Float32, nu_sgs_bg::Float32,
                           inlet_turbulence::Float32, temporal_weight::Float32,
                           use_temporal_interp::Bool, sponge_blend_dist::Bool;
                           block_offset::Int32=Int32(0), ndrange_blocks::Int=0,
                           parent_g=nothing, parent_temperature=nothing,
                           parent_g_old=nothing, parent_temperature_old=nothing,
                           parent_vel_thermal=nothing, parent_vel_old_thermal=nothing,
                           parent_tau_g::Float32=0.5f0)

    CUDA.device!(mgpu.device_ids[gpu_idx])
    level = mgpu.grids[gpu_idx][level_idx]
    cx, cy, cz, w, opp, my, mz = mgpu.lattice_arrays[gpu_idx]

    if iseven(t_sub)
        f_in, f_out = level.f, level.f_temp
        vel_in, vel_out = level.vel, level.vel_temp
    else
        f_in, f_out = level.f_temp, level.f
        vel_in, vel_out = level.vel_temp, level.vel
    end

    ddf_active = DDF_ENABLED && has_thermal(level)
    local g_in_t, g_out_t
    if ddf_active
        if iseven(t_sub)
            g_in_t, g_out_t = level.g, level.g_temp
        else
            g_in_t, g_out_t = level.g_temp, level.g
        end
    end

    perform_timestep_v2!(level,
                         parent_f, parent_rho, parent_vel, parent_ptr,
                         parent_f_old, parent_rho_old, parent_vel_old,
                         parent_tau,
                         f_out, f_in, vel_out, vel_in,
                         level.f_post_collision, u_vel,
                         cx, cy, cz, w, opp, my, mz,
                         domain_nx, domain_ny, domain_nz,
                         wall_model_active, c_wale_val, nu_sgs_bg,
                         t_sub, inlet_turbulence, temporal_weight, use_temporal_interp, sponge_blend_dist;
                         g_out = ddf_active ? g_out_t : nothing,
                         g_in = ddf_active ? g_in_t : nothing,
                         g_post_collision_arr = ddf_active ? level.g_post_collision : nothing,
                         parent_g = parent_g,
                         parent_temperature = parent_temperature,
                         parent_g_old = parent_g_old,
                         parent_temperature_old = parent_temperature_old,
                         parent_pi_neq = level.level_id > 1 ? mgpu.grids[gpu_idx][level_idx - 1].pi_neq : nothing,
                         parent_vel_thermal = parent_vel_thermal,
                         parent_vel_old_thermal = parent_vel_old_thermal,
                         parent_tau_g = parent_tau_g,
                         block_offset = block_offset,
                         ndrange_blocks = ndrange_blocks)
end

"""
Multi-GPU recursive step: dispatches to both GPUs per level.
Coarse levels (≤ split_level) run redundantly on both GPUs.
Fine levels (> split_level) are block-partitioned, then merged.
"""
function recursive_step_multigpu!(mgpu::MultiGPUState, current_lvl::Int, t_sub::Int,
                                   parent_data_per_gpu,  # Vector of tuples per GPU
                                   parent_tau::Float32, u_vel::Float32,
                                   domain_nx::Int, domain_ny::Int, domain_nz::Int,
                                   wall_model_active::Bool, c_wale_val::Float32, nu_sgs_bg::Float32,
                                   inlet_turbulence::Float32, use_temporal_interp::Bool, sponge_blend_dist::Bool)

    n_levels = length(mgpu.grids[1])
    if current_lvl > n_levels; return; end

    level_1 = mgpu.grids[1][current_lvl]
    n_blocks = length(level_1.active_block_coords)
    if n_blocks == 0; return; end

    is_partitioned = current_lvl > mgpu.split_level
    has_children = current_lvl < n_levels

    # Save old state for temporal interpolation on both GPUs
    for gpu_idx in 1:2
        level = mgpu.grids[gpu_idx][current_lvl]
        ddf_active = DDF_ENABLED && has_thermal(level)
        if should_snapshot_level_state(level, has_children, use_temporal_interp) || LOCAL_LATTICE_SHIFT_ENABLED
            CUDA.device!(mgpu.device_ids[gpu_idx])
            if iseven(t_sub)
                f_in = level.f; vel_in = level.vel
                g_in_t = ddf_active ? level.g : nothing
            else
                f_in = level.f_temp; vel_in = level.vel_temp
                g_in_t = ddf_active ? level.g_temp : nothing
            end
            copy_to_old!(level, f_in, vel_in; g_current = g_in_t)
        end
    end

    if is_partitioned
        # PARTITIONED: split blocks across GPUs
        n_half = n_blocks ÷ 2

        # GPU 0: blocks 1..n_half
        pf0, prho0, pvel0, pptr0, pf_old0, prho_old0, pvel_old0 = parent_data_per_gpu[1]
        pg0, pT0, pg_old0, pT_old0, pvel_t0, pvel_old_t0, ptau_g0 = parent_data_per_gpu[1][8:14]
        run_level_on_gpu!(1, mgpu, current_lvl, t_sub,
                          pf0, prho0, pvel0, pptr0, pf_old0, prho_old0, pvel_old0,
                          parent_tau, u_vel,
                          domain_nx, domain_ny, domain_nz,
                          wall_model_active, c_wale_val, nu_sgs_bg,
                          inlet_turbulence, 0.0f0, use_temporal_interp, sponge_blend_dist;
                          block_offset=Int32(0), ndrange_blocks=n_half,
                          parent_g=pg0, parent_temperature=pT0,
                          parent_g_old=pg_old0, parent_temperature_old=pT_old0,
                          parent_vel_thermal=pvel_t0, parent_vel_old_thermal=pvel_old_t0,
                          parent_tau_g=ptau_g0)

        # GPU 1: blocks n_half+1..n_blocks
        pf1, prho1, pvel1, pptr1, pf_old1, prho_old1, pvel_old1 = parent_data_per_gpu[2]
        pg1, pT1, pg_old1, pT_old1, pvel_t1, pvel_old_t1, ptau_g1 = parent_data_per_gpu[2][8:14]
        run_level_on_gpu!(2, mgpu, current_lvl, t_sub,
                          pf1, prho1, pvel1, pptr1, pf_old1, prho_old1, pvel_old1,
                          parent_tau, u_vel,
                          domain_nx, domain_ny, domain_nz,
                          wall_model_active, c_wale_val, nu_sgs_bg,
                          inlet_turbulence, 0.0f0, use_temporal_interp, sponge_blend_dist;
                          block_offset=Int32(n_half), ndrange_blocks=n_blocks - n_half,
                          parent_g=pg1, parent_temperature=pT1,
                          parent_g_old=pg_old1, parent_temperature_old=pT_old1,
                          parent_vel_thermal=pvel_t1, parent_vel_old_thermal=pvel_old_t1,
                          parent_tau_g=ptau_g1)

        sync_all_gpus!(mgpu)
        merge_level_arrays!(mgpu, current_lvl, t_sub)
        sync_all_gpus!(mgpu)
    else
        # REPLICATED: run full kernel on both GPUs independently
        for gpu_idx in 1:2
            pf, prho, pvel, pptr, pf_old, prho_old, pvel_old = parent_data_per_gpu[gpu_idx]
            pg, pT, pg_old, pT_old, pvel_t, pvel_old_t, ptau_g = parent_data_per_gpu[gpu_idx][8:14]
            run_level_on_gpu!(gpu_idx, mgpu, current_lvl, t_sub,
                              pf, prho, pvel, pptr, pf_old, prho_old, pvel_old,
                              parent_tau, u_vel,
                              domain_nx, domain_ny, domain_nz,
                              wall_model_active, c_wale_val, nu_sgs_bg,
                              inlet_turbulence, 0.0f0, use_temporal_interp, sponge_blend_dist;
                              parent_g=pg, parent_temperature=pT,
                              parent_g_old=pg_old, parent_temperature_old=pT_old,
                              parent_vel_thermal=pvel_t, parent_vel_old_thermal=pvel_old_t,
                              parent_tau_g=ptau_g)
        end
        sync_all_gpus!(mgpu)
    end

    # Recurse into children
    if has_children
        for sub_step in 0:1
            child_t = 2 * t_sub + sub_step
            tw = amr_child_temporal_weight(sub_step == 0 ? 0.0f0 : 0.5f0)

            # Build parent data for child level from each GPU's current level
            child_parent_data = Vector{Any}(undef, 2)
            for gpu_idx in 1:2
                level = mgpu.grids[gpu_idx][current_lvl]
                ddf_active = DDF_ENABLED && has_thermal(level)
                if iseven(t_sub)
                    f_out = level.f_temp; vel_out = level.vel_temp
                    g_out_t = ddf_active ? level.g_temp : nothing
                else
                    f_out = level.f; vel_out = level.vel
                    g_out_t = ddf_active ? level.g : nothing
                end
                child_parent_data[gpu_idx] = (
                    f_out, level.rho, vel_out, level.block_pointer,
                    level.f_old, level.rho_old, level.vel_old,
                    # thermal parent data
                    ddf_active ? g_out_t : nothing,
                    ddf_active ? level.temperature : nothing,
                    ddf_active ? level.g_old : nothing,
                    ddf_active ? level.temperature_old : nothing,
                    ddf_active ? vel_out : nothing,
                    ddf_active ? level.vel_old : nothing,
                    ddf_active ? level.tau_g : 0.5f0
                )
            end

            recursive_step_temporal_multigpu!(mgpu, current_lvl + 1, child_t,
                                              child_parent_data,
                                              level_1.tau, tw, u_vel,
                                              domain_nx, domain_ny, domain_nz,
                                              wall_model_active, c_wale_val, nu_sgs_bg,
                                              inlet_turbulence, use_temporal_interp, sponge_blend_dist)
        end

        if RESTRICT_FINE_TO_PARENT
            for gpu_idx in 1:2
                level = mgpu.grids[gpu_idx][current_lvl]
                child_level = mgpu.grids[gpu_idx][current_lvl + 1]
                ddf_active = DDF_ENABLED && has_thermal(level)
                if iseven(t_sub)
                    f_out = level.f_temp
                    vel_out = level.vel_temp
                    g_out_t = ddf_active ? level.g_temp : nothing
                else
                    f_out = level.f
                    vel_out = level.vel
                    g_out_t = ddf_active ? level.g : nothing
                end
                cx_loc, cy_loc, cz_loc, w_loc, opp_loc, _, _ = mgpu.lattice_arrays[gpu_idx]
                # Decimation f_neq filter — gated on AMR_RESTRICTION_PI_NEQ_FILTER
                # (Dorschner 2016 + Schukmann 2025: filter not needed for entropic
                # and actively hurts vc-ov by ~6.5 dB OASPL).
                if AMR_RESTRICTION_PI_NEQ_FILTER
                    filter_pi_neq!(child_level, get_backend(f_out))
                    filter_pi_neq!(child_level, get_backend(f_out))
                end
                restrict_fine_to_parent!(level, f_out, vel_out, child_level,
                                          cx_loc, cy_loc, cz_loc, w_loc, opp_loc;
                                          parent_g_current = g_out_t,
                                          child_f_current = child_level.f,
                                          child_g_current = (DDF_ENABLED && has_thermal(child_level)) ? child_level.g : nothing,
                                          current_step = t_sub)
            end
            sync_all_gpus!(mgpu)
        end
    end
end

"""
Multi-GPU recursive step with temporal interpolation weight.
"""
function recursive_step_temporal_multigpu!(mgpu::MultiGPUState, current_lvl::Int, t_sub::Int,
                                           parent_data_per_gpu,
                                           parent_tau::Float32, temporal_weight::Float32, u_vel::Float32,
                                           domain_nx::Int, domain_ny::Int, domain_nz::Int,
                                           wall_model_active::Bool, c_wale_val::Float32, nu_sgs_bg::Float32,
                                           inlet_turbulence::Float32, use_temporal_interp::Bool, sponge_blend_dist::Bool)

    n_levels = length(mgpu.grids[1])
    if current_lvl > n_levels; return; end

    level_1 = mgpu.grids[1][current_lvl]
    n_blocks = length(level_1.active_block_coords)
    if n_blocks == 0; return; end

    is_partitioned = current_lvl > mgpu.split_level
    has_children = current_lvl < n_levels

    # Save old state on both GPUs
    for gpu_idx in 1:2
        level = mgpu.grids[gpu_idx][current_lvl]
        ddf_active = DDF_ENABLED && has_thermal(level)
        if should_snapshot_level_state(level, has_children, use_temporal_interp) || LOCAL_LATTICE_SHIFT_ENABLED
            CUDA.device!(mgpu.device_ids[gpu_idx])
            if iseven(t_sub)
                f_in = level.f; vel_in = level.vel
                g_in_t = ddf_active ? level.g : nothing
            else
                f_in = level.f_temp; vel_in = level.vel_temp
                g_in_t = ddf_active ? level.g_temp : nothing
            end
            copy_to_old!(level, f_in, vel_in; g_current = g_in_t)
        end
    end

    if is_partitioned
        n_half = n_blocks ÷ 2

        pf0, prho0, pvel0, pptr0, pf_old0, prho_old0, pvel_old0 = parent_data_per_gpu[1]
        pg0, pT0, pg_old0, pT_old0, pvel_t0, pvel_old_t0, ptau_g0 = parent_data_per_gpu[1][8:14]
        run_level_on_gpu!(1, mgpu, current_lvl, t_sub,
                          pf0, prho0, pvel0, pptr0, pf_old0, prho_old0, pvel_old0,
                          parent_tau, u_vel,
                          domain_nx, domain_ny, domain_nz,
                          wall_model_active, c_wale_val, nu_sgs_bg,
                          inlet_turbulence, temporal_weight, use_temporal_interp, sponge_blend_dist;
                          block_offset=Int32(0), ndrange_blocks=n_half,
                          parent_g=pg0, parent_temperature=pT0,
                          parent_g_old=pg_old0, parent_temperature_old=pT_old0,
                          parent_vel_thermal=pvel_t0, parent_vel_old_thermal=pvel_old_t0,
                          parent_tau_g=ptau_g0)

        pf1, prho1, pvel1, pptr1, pf_old1, prho_old1, pvel_old1 = parent_data_per_gpu[2]
        pg1, pT1, pg_old1, pT_old1, pvel_t1, pvel_old_t1, ptau_g1 = parent_data_per_gpu[2][8:14]
        run_level_on_gpu!(2, mgpu, current_lvl, t_sub,
                          pf1, prho1, pvel1, pptr1, pf_old1, prho_old1, pvel_old1,
                          parent_tau, u_vel,
                          domain_nx, domain_ny, domain_nz,
                          wall_model_active, c_wale_val, nu_sgs_bg,
                          inlet_turbulence, temporal_weight, use_temporal_interp, sponge_blend_dist;
                          block_offset=Int32(n_half), ndrange_blocks=n_blocks - n_half,
                          parent_g=pg1, parent_temperature=pT1,
                          parent_g_old=pg_old1, parent_temperature_old=pT_old1,
                          parent_vel_thermal=pvel_t1, parent_vel_old_thermal=pvel_old_t1,
                          parent_tau_g=ptau_g1)

        sync_all_gpus!(mgpu)
        merge_level_arrays!(mgpu, current_lvl, t_sub)
        sync_all_gpus!(mgpu)
    else
        for gpu_idx in 1:2
            pf, prho, pvel, pptr, pf_old, prho_old, pvel_old = parent_data_per_gpu[gpu_idx]
            pg, pT, pg_old, pT_old, pvel_t, pvel_old_t, ptau_g = parent_data_per_gpu[gpu_idx][8:14]
            run_level_on_gpu!(gpu_idx, mgpu, current_lvl, t_sub,
                              pf, prho, pvel, pptr, pf_old, prho_old, pvel_old,
                              parent_tau, u_vel,
                              domain_nx, domain_ny, domain_nz,
                              wall_model_active, c_wale_val, nu_sgs_bg,
                              inlet_turbulence, temporal_weight, use_temporal_interp, sponge_blend_dist;
                              parent_g=pg, parent_temperature=pT,
                              parent_g_old=pg_old, parent_temperature_old=pT_old,
                              parent_vel_thermal=pvel_t, parent_vel_old_thermal=pvel_old_t,
                              parent_tau_g=ptau_g)
        end
        sync_all_gpus!(mgpu)
    end

    if has_children
        for sub_step in 0:1
            child_t = 2 * t_sub + sub_step
            tw = amr_child_temporal_weight(sub_step == 0 ? 0.0f0 : 0.5f0)

            child_parent_data = Vector{Any}(undef, 2)
            for gpu_idx in 1:2
                level = mgpu.grids[gpu_idx][current_lvl]
                ddf_active = DDF_ENABLED && has_thermal(level)
                if iseven(t_sub)
                    f_out = level.f_temp; vel_out = level.vel_temp
                    g_out_t = ddf_active ? level.g_temp : nothing
                else
                    f_out = level.f; vel_out = level.vel
                    g_out_t = ddf_active ? level.g : nothing
                end
                child_parent_data[gpu_idx] = (
                    f_out, level.rho, vel_out, level.block_pointer,
                    level.f_old, level.rho_old, level.vel_old,
                    ddf_active ? g_out_t : nothing,
                    ddf_active ? level.temperature : nothing,
                    ddf_active ? level.g_old : nothing,
                    ddf_active ? level.temperature_old : nothing,
                    ddf_active ? vel_out : nothing,
                    ddf_active ? level.vel_old : nothing,
                    ddf_active ? level.tau_g : 0.5f0
                )
            end

            recursive_step_temporal_multigpu!(mgpu, current_lvl + 1, child_t,
                                              child_parent_data,
                                              level_1.tau, tw, u_vel,
                                              domain_nx, domain_ny, domain_nz,
                                              wall_model_active, c_wale_val, nu_sgs_bg,
                                              inlet_turbulence, use_temporal_interp, sponge_blend_dist)
        end

        if RESTRICT_FINE_TO_PARENT
            for gpu_idx in 1:2
                level = mgpu.grids[gpu_idx][current_lvl]
                child_level = mgpu.grids[gpu_idx][current_lvl + 1]
                ddf_active = DDF_ENABLED && has_thermal(level)
                if iseven(t_sub)
                    f_out = level.f_temp
                    vel_out = level.vel_temp
                    g_out_t = ddf_active ? level.g_temp : nothing
                else
                    f_out = level.f
                    vel_out = level.vel
                    g_out_t = ddf_active ? level.g : nothing
                end
                cx_loc, cy_loc, cz_loc, w_loc, opp_loc, _, _ = mgpu.lattice_arrays[gpu_idx]
                # Decimation f_neq filter — gated on AMR_RESTRICTION_PI_NEQ_FILTER
                # (Dorschner 2016 + Schukmann 2025: filter not needed for entropic
                # and actively hurts vc-ov by ~6.5 dB OASPL).
                if AMR_RESTRICTION_PI_NEQ_FILTER
                    filter_pi_neq!(child_level, get_backend(f_out))
                    filter_pi_neq!(child_level, get_backend(f_out))
                end
                restrict_fine_to_parent!(level, f_out, vel_out, child_level,
                                          cx_loc, cy_loc, cz_loc, w_loc, opp_loc;
                                          parent_g_current = g_out_t,
                                          child_f_current = child_level.f,
                                          child_g_current = (DDF_ENABLED && has_thermal(child_level)) ? child_level.g : nothing,
                                          current_step = t_sub)
            end
            sync_all_gpus!(mgpu)
        end
    end
end

"""
Multi-GPU batch execution: replaces execute_timestep_batch! when multi-GPU is active.
"""
function execute_timestep_batch_multigpu!(mgpu::MultiGPUState, t_start::Int, batch_size::Int, u_curr::Float32,
                                          domain_nx::Int, domain_ny::Int, domain_nz::Int,
                                          wall_model_active::Bool, c_wale_val::Float32, nu_sgs_bg::Float32,
                                          inlet_turbulence::Float32, use_temporal_interp::Bool, sponge_blend_dist::Bool)

    for t_offset in 0:(batch_size-1)
        t = t_start + t_offset

        # Build L1 parent data (nothing for coarsest level) for each GPU
        l1_parent_data = Vector{Any}(undef, 2)
        for gpu_idx in 1:2
            l1_parent_data[gpu_idx] = (
                nothing, nothing, nothing, nothing,
                nothing, nothing, nothing,
                nothing, nothing, nothing, nothing, nothing, nothing, 0.5f0
            )
        end

        recursive_step_multigpu!(mgpu, 1, t,
                                  l1_parent_data,
                                  Float32(0.5), u_curr,
                                  domain_nx, domain_ny, domain_nz,
                                  wall_model_active, c_wale_val, nu_sgs_bg,
                                  inlet_turbulence, use_temporal_interp, sponge_blend_dist)
    end

    sync_all_gpus!(mgpu)
end
