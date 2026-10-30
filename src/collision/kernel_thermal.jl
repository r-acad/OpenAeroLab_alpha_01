# ==============================================================================
# THERMAL DDF KERNEL — Frapolli Two-Population Model (Phase 4b)
# ==============================================================================
# Implements the Frapolli/Chikatamarla/Karlin (Phys Rev E 2015-2020) thermal
# model where g distributions use the SAME lattice as f (D3Q39 or D3Q27).
#
# Key physics:
#   g_eq_i = (C_v - D*cs²/2) * T * f_eq_i
#   T = (E_kin - rho*u^2 + 2*sum(g)) / (2*rho*C_v)
#
# The coefficient (Cv - D*cs²/2) accounts for the lattice's cs² in the energy
# decomposition: f carries D*cs²/2 of internal energy per unit temperature,
# g carries the remainder (Cv - D*cs²/2).
# For D3Q39 (cs²=2/3): Cv - 3*(2/3)/2 = 2.5 - 1.0 = 1.5
# For D3Q27 (cs²=1/3): Cv - 3*(1/3)/2 = 2.5 - 0.5 = 2.0 (but D3Q27 uses legacy model)
#
# Temperature is recovered from energy conservation — no explicit source terms
# needed. Viscous heating and compression work emerge automatically from kinetic
# theory, producing correct Rankine-Hugoniot shock relations.
#
# When flow uses D3Q27 (incompressible mode), falls back to the simpler
# advection-diffusion model: g_eq = w_k * T * (1 + 3*c.u), T = sum(g).
# ==============================================================================

using KernelAbstractions


@kernel function stream_collide_thermal!(
    g_out, g_in, g_post_collision,
    temperature_out, rho_arr, vel_arr,
    obstacle, sponge_arr, wall_dist_arr,
    neighbor_table,
    active_coords_x, active_coords_y, active_coords_z,
    parent_g, parent_temperature, parent_rho, parent_vel, parent_kinetic_energy, parent_ptr,
    parent_g_old, parent_temperature_old, parent_rho_old, parent_vel_old,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    tau_g::Float32,
    tau_g_parent::Float32,
    tau_flow::Float32,
    tau_flow_parent::Float32,
    is_level_1::Int32,
    ymin_wall_bc::Int32,
    nx_global::Int32, ny_global::Int32, nz_global::Int32,
    u_inlet::Float32,
    T_inlet::Float32,
    T_wall::Float32,
    wall_bc_adiabatic::Int32,   # 1=adiabatic (bounce-back), 0=isothermal (anti-bounce-back)
    n_blocks::Int32,
    block_size::Int32,
    cx_arr, cy_arr, cz_arr, w_arr, opp_arr,
    mirror_y_arr, mirror_z_arr,
    store_post_collision::Int32,
    temporal_weight::Float32,
    use_temporal_interp::Int32,
    use_macro_temporal_interp::Int32,
    sponge_blend_distributions::Int32,
    # Phase 4b: SGS thermal diffusivity via WALE + turbulent Prandtl number
    c_wale::Float32,
    nu_sgs_background::Float32,
    amr_sgs_scale::Float32,
    Pr_t::Float32,             # turbulent Prandtl number (kappa_sgs = nu_eddy / Pr_t)
    # Phase 4b: Frapolli model parameters
    n_dirs_thermal::Int32,     # number of thermal directions (27 or 39)
    kinetic_energy_arr,        # E_kin = sum(f*|c_unshifted|^2) from flow kernel
    Cv::Float32,               # C_v (2.5 for air). 0.0 = legacy advection-diffusion mode
    cs2_thermal::Float32,      # cs^2 for thermal lattice (1/3 D3Q27, 2/3 D3Q39)
    shift_x::Float32,          # Galilean shift x (0 for unshifted/D3Q27)
    shift_y::Float32,          # Galilean shift y
    shift_z::Float32,          # Galilean shift z
    f_eq_inlet_arr,            # 39-element numerical freestream equilibrium for D3Q39
    # ── Multi-GPU block offset ──
    block_offset::Int32,       # 0 for single-GPU; >0 shifts b_idx for multi-GPU partitioning
    # ── Phase B4: adaptive sigma_even (h-damping of even g-modes) from |Π_neq| ──
    # When variable_sigma_th_i==1, sigma_even varies per cell:
    #   sigma_even = sigma_ev_min + (sigma_ev_max − sigma_ev_min)·quadratic_blend(|Π_neq^pre|/pi_ref)
    # Smooth regions → sigma_ev_min (pure mirror, kills noise)
    # Physics regions → sigma_ev_max (partial mirror, preserves energy content)
    pi_neq_in,                   # BS×BS×BS×n_blocks×6 (stub when disabled)
    variable_sigma_th_i::Int32,  # 0 = use constant sigma_even=0; 1 = per-cell
    sigma_ev_min::Float32,       # Min damping permissiveness (freestream)
    sigma_ev_max::Float32,       # Max damping permissiveness (shocks/wakes)
    sigma_th_pi_ref::Float32,    # |Π_neq^pre| normalization threshold
    # AMR feather-band thermal cleanup. Same cell-local weight used by the
    # D3Q39 flow kernel: 1 at a refinement frontier, decaying to 0 inward.
    feather_weight_in,
    feather_nu_max::Float32,
    feather_active_i::Int32,
    # ── SOTA #1 (2026-05-29): total-energy source terms ──
    # When energy_source_active==1 (Frapolli/D3Q39 only), the per-cell
    # compressible energy sources are injected into Σg:
    #   S_E = 2·(−p·∇·u + Φ),   p = (γ−1)·ρ·Cv·T,   Φ = Π_neq:∇u
    # distributed as 0.5·S_E·w_k per direction so that 2·Σg gains S_E and the
    # Frapolli closure yields ρCv·ΔT = −p·∇·u + Φ per step. eos_gamma supplies
    # γ for the thermodynamic pressure.
    energy_source_active::Int32,
    eos_gamma::Float32,
    # AUDIT FIX (2026-06-03): the effective pressure coefficient the FLOW kernel
    # uses for its equilibrium, i.e. p = ρ·eos_R_flow·T. The pressure-work term
    # −p·∇·u in the energy source MUST use this same coefficient so the energy
    # budget is consistent with the momentum field. Previously the source used
    # the thermodynamic (γ−1)·Cv unconditionally, while the default flow EOS is
    # the linearized cs² (2/3 for D3Q39) — a 1.0-vs-0.667 mismatch that fed the
    # wrong compression work into the energy equation by default. Caller passes
    # (γ−1)·Cv when EOS_THERMODYNAMIC is on, else cs2_thermal.
    eos_R_flow::Float32,
    # AUDIT FIX (2026-06-03): explicit T-recovery upper bound. >0 overrides the
    # auto 2×-stagnation cap (which clips high-T Riemann states like Lax L T=7.93).
    # 0 = use the auto formula (aero unchanged).
    t_admissible_max_override::Float32,
    # INTENSIVE-T ADVECTION FIX (2026-06-29): 1 = rescale streamed g by ρ_local/ρ_upstream
    # so the thermal advects the intensive temperature, not the extensive energy-density
    # ρT (fixes spurious heating at sharp ρ gradients). 0 = legacy (byte-identical).
    thermal_intensive_i::Int32
)
    x, y, z, b_local = @index(Global, NTuple)
    b_idx = b_local + block_offset

    if b_idx <= n_blocks
        @inbounds begin
            bx = active_coords_x[b_idx]
            by = active_coords_y[b_idx]
            bz = active_coords_z[b_idx]
            gx = (bx - Int32(1)) * block_size + Int32(x)
            gy = (by - Int32(1)) * block_size + Int32(y)
            gz = (bz - Int32(1)) * block_size + Int32(z)

            is_obs = obstacle[x, y, z, b_idx]

            # Read velocity from flow kernel output (already computed)
            ux = vel_arr[x, y, z, b_idx, 1]
            uy = vel_arr[x, y, z, b_idx, 2]
            uz = vel_arr[x, y, z, b_idx, 3]

            # Frapolli mode: Cv > 0 means we use energy-based T recovery
            # Legacy mode: Cv = 0 means T = sum(g) (advection-diffusion)
            is_frapolli = Cv > 0.1f0
            T_admissible_max = T_inlet
            if is_frapolli
                gamma_thermal = 1.0f0 + 1.0f0 / Cv
                Ma_inlet_sq = (u_inlet * u_inlet) / max(cs2_thermal * T_inlet, 1.0f-6)
                # Allow 2× the isentropic stagnation temperature as upper bound.
                # Normal shocks, numerical overshoots, and compression heating can
                # produce transient T > T_stagnation. The previous 1.05× margin was
                # too tight for supersonic flows and clipped physical temperature
                # states, causing energy loss → wrong Rankine-Hugoniot jumps.
                T_admissible_max = max(
                    T_inlet,
                    T_inlet * (1.0f0 + 0.5f0 * (gamma_thermal - 1.0f0) * Ma_inlet_sq) * 2.0f0
                )
                # AUDIT FIX (2026-06-03): explicit override for high-T Riemann
                # states (Lax left T=7.93). When >0, use it directly so the auto
                # 2×-stagnation cap doesn't clip the physical per-side temperature.
                if t_admissible_max_override > 0.0f0
                    T_admissible_max = t_admissible_max_override
                end
            end

            # ══════════════════════════════════════════════════════════
            # STEP 1: Pull-streaming — gather g distributions
            # ══════════════════════════════════════════════════════════
            g_sum = 0.0f0
            coeff_g_bc = is_frapolli ? (Cv - 1.5f0 * eos_R_flow) : 0.0f0   # AUDIT 2026-06-03: eos_R-consistent g variance
            # INTENSIVE-T ADVECTION FIX (2026-06-29): the Frapolli g carries the energy
            # DENSITY ρ·T (g_eq = coeff_g·T·ρ·…), so streaming it across a DENSITY gradient
            # advects ρ_upstream·T while the recovery normalises by ρ_local → spurious heat
            # at any sharp ρ jump (over-density edge, shock, AMR-edge). Root-caused +
            # validated offline (tools/diag_thermal_1d.jl): a ρ 1.8↔0.2 step drifts T to 5.8
            # WITHOUT this and holds T=1.0000 WITH it. Fix: rescale each streamed g by
            # ρ_local/ρ_upstream so the advected quantity is the INTENSIVE temperature.
            # Byte-identical where ρ is uniform (ratio=1). Gated (default off) pending
            # shock-tube regression. Frapolli only.
            rho_local_th = (thermal_intensive_i == Int32(1) && is_frapolli) ?
                           max(rho_arr[x, y, z, b_idx], 0.01f0) : 1.0f0

            for k in Int32(1):n_dirs_thermal
                cx = cx_arr[k]
                cy = cy_arr[k]
                cz = cz_arr[k]

                sx = Int32(x) - cx
                sy = Int32(y) - cy
                sz = Int32(z) - cz

                val = 0.0f0

                if sx >= Int32(1) && sx <= block_size && sy >= Int32(1) && sy <= block_size && sz >= Int32(1) && sz <= block_size
                    val = g_in[sx, sy, sz, b_idx, k]
                    if thermal_intensive_i == Int32(1) && is_frapolli
                        val *= rho_local_th / max(rho_arr[sx, sy, sz, b_idx], 0.01f0)
                    end
                else
                    nb_off_x = sx < Int32(1) ? Int32(-1) : (sx > block_size ? Int32(1) : Int32(0))
                    nb_off_y = sy < Int32(1) ? Int32(-1) : (sy > block_size ? Int32(1) : Int32(0))
                    nb_off_z = sz < Int32(1) ? Int32(-1) : (sz > block_size ? Int32(1) : Int32(0))
                    dir_idx = (nb_off_x + Int32(1)) + (nb_off_y + Int32(1))*Int32(3) + (nb_off_z + Int32(1))*Int32(9) + Int32(1)
                    nb_idx_global = neighbor_table[b_idx, dir_idx]

                    if nb_idx_global > Int32(0)
                        nsx = sx < Int32(1) ? sx + block_size : (sx > block_size ? sx - block_size : sx)
                        nsy = sy < Int32(1) ? sy + block_size : (sy > block_size ? sy - block_size : sy)
                        nsz = sz < Int32(1) ? sz + block_size : (sz > block_size ? sz - block_size : sz)
                        val = g_in[nsx, nsy, nsz, nb_idx_global, k]
                        if thermal_intensive_i == Int32(1) && is_frapolli
                            val *= rho_local_th / max(rho_arr[nsx, nsy, nsz, nb_idx_global], 0.01f0)
                        end
                    else
                        src_gx = gx - cx
                        # Periodic-Y AMR: wrap the source across the Y seam (identity when off).
                        src_gy = _pwrap_y(gy - cy, ny_global)
                        src_gz = gz - cz

                        is_inlet  = src_gx < Int32(1)
                        is_outlet = src_gx > nx_global
                        is_y_min  = src_gy < Int32(1)
                        is_y_max  = src_gy > ny_global
                        is_z_min  = src_gz < Int32(1)
                        is_z_max  = src_gz > nz_global

                        at_y_boundary = is_y_min || is_y_max
                        at_z_boundary = is_z_min || is_z_max

                        if is_inlet
                            if is_frapolli
                                val = coeff_g_bc * T_inlet * f_eq_inlet_arr[k]
                            else
                                val = calculate_thermal_equilibrium(T_inlet, u_inlet, 0.0f0, 0.0f0,
                                          w_arr[k], Float32(cx), Float32(cy), Float32(cz))
                            end
                        elseif is_outlet && at_y_boundary && at_z_boundary
                            k_mirror_y = mirror_y_arr[k]
                            k_mirror_z = mirror_z_arr[k]
                            k_mirror_yz = mirror_y_arr[k_mirror_z]
                            val = 0.25f0 * (
                                g_in[x, y, z, b_idx, k] +
                                g_in[x, y, z, b_idx, k_mirror_y] +
                                g_in[x, y, z, b_idx, k_mirror_z] +
                                g_in[x, y, z, b_idx, k_mirror_yz]
                            )
                        elseif is_outlet && at_y_boundary
                            k_mirror_y = mirror_y_arr[k]
                            val = 0.5f0 * (g_in[x, y, z, b_idx, k] +
                                           g_in[x, y, z, b_idx, k_mirror_y])
                        elseif is_outlet && at_z_boundary
                            k_mirror_z = mirror_z_arr[k]
                            val = 0.5f0 * (g_in[x, y, z, b_idx, k] +
                                           g_in[x, y, z, b_idx, k_mirror_z])
                        elseif is_outlet
                            # Supersonic outlet: zero-gradient carry-over from the last
                            # interior state. Imposing inlet thermal equilibrium here
                            # feeds non-physical waves back into the domain.
                            # (2026-06-29: TESTED imposing freestream g here for the subsonic
                            # case — byte-identical T, because the outlet sponge already
                            # clamps this region toward T_inlet. So the outlet is NOT the
                            # T-drift source; the source is the interior thermal advection.)
                            val = g_in[x, y, z, b_idx, k]
                        elseif is_y_min && ymin_wall_bc == Int32(1)
                            if wall_bc_adiabatic == Int32(1)
                                val = g_in[x, y, z, b_idx, opp_arr[k]]
                            else
                                if is_frapolli
                                    coeff_g_w = Cv - 1.5f0 * eos_R_flow   # AUDIT 2026-06-03: eos_R-consistent g variance
                                    g_eq_wall = coeff_g_w * T_wall * w_arr[k]
                                else
                                    g_eq_wall = calculate_thermal_equilibrium(T_wall, 0.0f0, 0.0f0, 0.0f0,
                                                      w_arr[k], Float32(cx), Float32(cy), Float32(cz))
                                end
                                g_opp = g_in[x, y, z, b_idx, opp_arr[k]]
                                val = 2.0f0 * g_eq_wall - g_opp
                            end
                        elseif at_y_boundary
                            val = g_in[x, y, z, b_idx, mirror_y_arr[k]]
                        elseif at_z_boundary
                            val = g_in[x, y, z, b_idx, mirror_z_arr[k]]
                        elseif is_level_1 == Int32(0)
                            # Coarse-to-fine interpolation for thermal field
                            val = interpolate_thermal_with_rescaling(
                                parent_g, parent_temperature, parent_rho, parent_vel, parent_kinetic_energy,
                                parent_g_old, parent_temperature_old, parent_rho_old, parent_vel_old,
                                parent_ptr,
                                parent_dim_x, parent_dim_y, parent_dim_z,
                                src_gx, src_gy, src_gz,
                                k, opp_arr[k], block_size, w_arr[k],
                                Float32(cx), Float32(cy), Float32(cz),
                                w_arr[opp_arr[k]], Float32(cx_arr[opp_arr[k]]), Float32(cy_arr[opp_arr[k]]), Float32(cz_arr[opp_arr[k]]),
                                tau_g_parent, tau_g,
                                tau_flow_parent, tau_flow,
                                temporal_weight,
                                use_temporal_interp,
                                use_macro_temporal_interp,
                                # Phase 4b: Frapolli interpolation params
                                is_frapolli ? Int32(1) : Int32(0),
                                Cv, cs2_thermal, shift_x, shift_y, shift_z,
                                cx_arr, cy_arr, cz_arr, w_arr,
                                eos_R_flow   # SOTA A3: EOS-consistent g variance across AMR
                            )
                        else
                            if is_frapolli
                                val = coeff_g_bc * T_inlet * f_eq_inlet_arr[k]
                            else
                                val = w_arr[k] * T_inlet
                            end
                        end
                    end
                end

                g_sum += val

                # ── Obstacle handling ──
                if is_obs
                    if wall_bc_adiabatic == Int32(1)
                        # Adiabatic: bounce-back (reflects g, preserves temperature)
                        g_out[x, y, z, b_idx, k] = g_in[x, y, z, b_idx, opp_arr[k]]
                    else
                        # Isothermal: anti-bounce-back to impose T_wall
                        if is_frapolli
                            coeff_g_w = Cv - 1.5f0 * eos_R_flow   # AUDIT 2026-06-03: eos_R-consistent g variance
                            # At wall, u=0 → f_eq = rho * w_k, so g_eq = coeff * T_wall * rho * w_k
                            # Use rho=1 at wall (approximate)
                            g_eq_wall = coeff_g_w * T_wall * w_arr[k]
                            g_opp = g_in[x, y, z, b_idx, opp_arr[k]]
                            g_out[x, y, z, b_idx, k] = 2.0f0 * g_eq_wall - g_opp
                        else
                            g_eq_wall = calculate_thermal_equilibrium(T_wall, 0.0f0, 0.0f0, 0.0f0,
                                            w_arr[k], Float32(cx), Float32(cy), Float32(cz))
                            g_opp = g_in[x, y, z, b_idx, opp_arr[k]]
                            g_out[x, y, z, b_idx, k] = 2.0f0 * g_eq_wall - g_opp
                        end
                    end
                    if store_post_collision == Int32(1)
                        g_post_collision[x, y, z, b_idx, k] = g_out[x, y, z, b_idx, k]
                    end
                else
                    # Store streamed value temporarily (will be overwritten by collision below)
                    g_out[x, y, z, b_idx, k] = val
                end
            end

            # ══════════════════════════════════════════════════════════
            # STEP 2: Compute temperature + sponge + collision (non-obstacle)
            # ══════════════════════════════════════════════════════════
            if is_obs
                # Solid cells do not carry a meaningful Frapolli temperature state.
                # Writing g_sum here contaminates the flow-side temperature filter
                # near immersed walls, because in the Frapolli model sum(g) != T.
                temperature_out[x, y, z, b_idx] = wall_bc_adiabatic == Int32(1) ? T_inlet : T_wall
            else
                # ── Temperature recovery (Frapolli + temporal smoothing) ──
                if is_frapolli
                    # Full Frapolli formula: g_sum smooths E_kin noise (g acts
                    # as low-pass filter on energy). Pure mirror (σ=0) ensures
                    # g_sum ≈ equilibrium (no checkerboard). Strong temporal
                    # smoothing (0.5/0.5) damps block-boundary banding.
                    rho_local_raw = rho_arr[x, y, z, b_idx]
                    rho_local = max(rho_local_raw, 0.01f0)
                    ux_rel = ux - shift_x
                    uy_rel = uy - shift_y
                    uz_rel = uz - shift_z
                    u_sq = ux_rel*ux_rel + uy_rel*uy_rel + uz_rel*uz_rel
                    E_kin = kinetic_energy_arr[x, y, z, b_idx]
                    if !(rho_local_raw == rho_local_raw) || !(E_kin == E_kin) || !(g_sum == g_sum) || rho_local_raw < 0.01f0
                        T_local = T_inlet
                    else
                        # AUDIT FIX (2026-06-03, Derivation A): convention-consistent
                        # Frapolli closure. The flow E_kin is built with variance
                        # eos_R_flow (thermodynamic R when eos_thermodynamic on):
                        #   E_kin − ρ|u|² = D·ρ·cs² + D·ρ·eos_R·(T−1),  D=3
                        # while g carries (Cv − D·eos_R/2)·ρ·T (coeff_g below). The
                        # legacy closure assumed the lattice-cs² variance, so the
                        # mismatch D·ρ·(eos_R−cs²)·(T−1) made it a contraction whose
                        # fixed point is T=1 (pinning the Sod right state to ~1 not
                        # 0.8). Add the offset D·ρ·(cs²−eos_R) → T_rec ≡ T for any T.
                        # When eos_R_flow==cs2_thermal (linearized/aero default) the
                        # offset is exactly 0 → byte-identical to the prior code.
                        eos_offset = 3.0f0 * rho_local * (cs2_thermal - eos_R_flow)
                        T_local = (E_kin - rho_local * u_sq + 2.0f0 * g_sum - eos_offset) /
                                  (2.0f0 * rho_local * Cv)
                        T_local = clamp(T_local, 0.1f0, T_admissible_max)
                    end
                else
                    # Legacy advection-diffusion: T = sum(g_k)
                    T_local = max(g_sum, 0.01f0)
                end

                # Sponge blending for temperature — required for compressible
                # stability; without it T drifts in the sponge and p = ρ·cs²·T
                # feedback produces unbounded pressure waves.
                sp = sponge_arr[x, y, z, b_idx]
                if sp > 0.0f0
                    T_local = T_local * (1.0f0 - sp) + T_inlet * sp
                    ux = ux * (1.0f0 - sp) + u_inlet * sp
                    uy = uy * (1.0f0 - sp)
                    uz = uz * (1.0f0 - sp)
                end
                if is_frapolli
                    T_local = clamp(T_local, 0.1f0, T_admissible_max)
                    # Strong temporal smoothing: damps block-boundary banding.
                    # At steady state: T_smoothed = T_frapolli (unbiased).
                    # Response time: ~5 timesteps (fast enough for shocks at 30 cells/D).
                    T_prev = temperature_out[x, y, z, b_idx]
                    if T_prev == T_prev && T_prev > 0.05f0
                        T_local = 0.5f0 * T_local + 0.5f0 * T_prev
                    end
                end

                temperature_out[x, y, z, b_idx] = T_local

                # ── WALE SGS thermal diffusivity ──
                # Compute velocity gradients and WALE eddy viscosity,
                # then derive turbulent thermal diffusivity: kappa_sgs = nu_eddy / Pr_t
                kappa_mol = (tau_g - 0.5f0) * cs2_thermal

                g11, g12, g13, g21, g22, g23, g31, g32, g33 = compute_velocity_gradients_amr_fluid(
                    vel_arr, obstacle, x, y, z, b_idx, block_size, neighbor_table,
                    is_level_1, parent_vel, parent_vel_old, parent_ptr,
                    parent_dim_x, parent_dim_y, parent_dim_z,
                    gx, gy, gz, temporal_weight, use_temporal_interp
                )

                gsq11 = g11*g11 + g12*g21 + g13*g31
                gsq12 = g11*g12 + g12*g22 + g13*g32
                gsq13 = g11*g13 + g12*g23 + g13*g33
                gsq21 = g21*g11 + g22*g21 + g23*g31
                gsq22 = g21*g12 + g22*g22 + g23*g32
                gsq23 = g21*g13 + g22*g23 + g23*g33
                gsq31 = g31*g11 + g32*g21 + g33*g31
                gsq32 = g31*g12 + g32*g22 + g33*g32
                gsq33 = g31*g13 + g32*g23 + g33*g33

                tr_gsq  = gsq11 + gsq22 + gsq33
                tr_term = tr_gsq / 3.0f0

                Sd11 = gsq11 - tr_term
                Sd22 = gsq22 - tr_term
                Sd33 = gsq33 - tr_term
                Sd12 = 0.5f0 * (gsq12 + gsq21)
                Sd13 = 0.5f0 * (gsq13 + gsq31)
                Sd23 = 0.5f0 * (gsq23 + gsq32)

                S12 = 0.5f0 * (g12 + g21)
                S13 = 0.5f0 * (g13 + g31)
                S23 = 0.5f0 * (g23 + g32)

                OP1 = Sd11*Sd11 + Sd22*Sd22 + Sd33*Sd33 + 2.0f0*(Sd12*Sd12 + Sd13*Sd13 + Sd23*Sd23)
                OP2 = g11*g11 + g22*g22 + g33*g33 + 2.0f0*(S12*S12 + S13*S13 + S23*S23)

                nu_sgs_background_local = nu_sgs_background * amr_sgs_scale
                nu_eddy = 0.0f0
                if OP1 > 1.0f-12
                    OP1_32 = OP1 * sqrt(OP1)
                    OP2_52 = OP2 * OP2 * sqrt(max(OP2, 1.0f-12))
                    denom = OP2_52 + OP1 * sqrt(sqrt(max(OP1, 1.0f-12)))
                    if denom > 1.0f-12
                        nu_eddy = (c_wale * c_wale) * OP1_32 / denom
                    end
                end
                nu_eddy = max(nu_eddy, nu_sgs_background_local)

                # Turbulent thermal diffusivity: kappa_sgs = nu_eddy / Pr_t
                kappa_sgs = Pr_t > 0.01f0 ? nu_eddy / Pr_t : 0.0f0
                kappa_eff = kappa_mol + kappa_sgs

                fw_thermal = 0.0f0
                if feather_active_i == Int32(1)
                    fw_thermal = feather_weight_in[x, y, z, b_idx]
                    if fw_thermal > 0.0f0
                        # Pass185 REVERT: thermal feather kept at zero. The pass184
                        # attempt to activate it (kappa_feather = 0.2 · ν_max · fw / Pr_t)
                        # never actually compiled — Pr_t inside the kernel was the
                        # likely culprit. Even more importantly, the broader pass183
                        # feather aggression (coarse-side + all-levels + wider depth)
                        # introduced visible horizontal stripes in the velocity
                        # highpass slice. Reverted to the historical setting.
                        kappa_feather = 0.0f0
                        kappa_eff = max(kappa_eff, kappa_mol + kappa_feather)
                    end
                end
                tau_g_eff = 0.5f0 + kappa_eff / cs2_thermal

                # BGK collision for thermal distributions
                omega_g = 1.0f0 / max(tau_g_eff, 0.500001f0)

                # Frapolli coefficient: g_eq = (Cv - D*cs²/2) * T * f_eq
                # D=3, so D*cs²/2 = 1.5*cs²  (NOT D/2 = 1.5)
                coeff_g = is_frapolli ? (Cv - 1.5f0 * eos_R_flow) : 0.0f0   # AUDIT 2026-06-03: eos_R-consistent g variance

                # Relative velocity for D3Q39 equilibrium (shifted lattice)
                ux_rel = ux - shift_x
                uy_rel = uy - shift_y
                uz_rel = uz - shift_z
                rho_eq = 1.0f0
                lambda_eq = (
                    0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
                    0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0
                )
                mass_corr_eq = 1.0f0
                if is_frapolli
                    rho_eq = max(rho_arr[x, y, z, b_idx], 0.01f0)
                    lambda_eq = solve_13mom_lambdas_d3q39(
                        rho_eq, ux_rel, uy_rel, uz_rel,
                        cx_arr, cy_arr, cz_arr, w_arr,
                        cs2_thermal, Int32(60), T_local,
                        shift_x, shift_y, shift_z
                    )
                    mass_corr_eq = compute_13mom_mass_correction_d3q39(
                        rho_eq, lambda_eq,
                        cx_arr, cy_arr, cz_arr, w_arr,
                        cs2_thermal, shift_x, shift_y, shift_z
                    )
                end

                # ── SOTA #1: total-energy source terms (Frapolli only) ──
                # The passive g-relaxation above conserves Σg toward g_eq but
                # omits the two source terms of the compressible energy equation.
                # We add them here, distributed across directions as S_E_half·w_k
                # inside the collision loop, so 2·Σg gains S_E and the closure
                # T=(E_kin−ρ|u|²+2Σg)/(2ρCv) yields ρCv·ΔT = −p·∇·u + Φ per step.
                #   ∇·u   = g11 + g22 + g33               (from gradient tensor)
                #   p     = (γ−1)·ρ·Cv·T                  (thermodynamic EOS)
                #   Φ     = 2ρν(S:S − ⅓(∇·u)²) ≥ 0        (deviatoric dissipation)
                # Per-step bounded to ±25% of the local internal energy. A real
                # shock spreads its heating over the several cells it occupies, so
                # the physical per-step contribution is well under this cap; the
                # bound only catches noise/under-resolved-gradient transients that
                # would otherwise drive T negative or blow up in a single step.
                S_E_half = 0.0f0
                if energy_source_active == Int32(1) && is_frapolli
                    div_u = g11 + g22 + g33
                    # AUDIT FIX (2026-06-03): use the SAME pressure the flow
                    # equilibrium uses (p = ρ·eos_R_flow·T), not the unconditional
                    # thermodynamic (γ−1)·Cv·T. eos_R_flow = (γ−1)·Cv when the
                    # thermodynamic EOS is active, else cs²_thermal (linearized).
                    # This keeps the compression work −p·∇·u in the energy budget
                    # consistent with the momentum field's pressure.
                    p_thermo = eos_R_flow * rho_eq * T_local
                    pressure_work = -p_thermo * div_u
                    # Viscous heating Φ = σ_ij ∂_j u_i, with the deviatoric
                    # Newtonian stress σ_ij = 2ρν(S_ij − ⅓δ_ij∇·u). Computed
                    # directly from the velocity-gradient strain rate (not from
                    # pi_neq, whose stored value is post-collision-scaled and
                    # τ-ambiguous per kernel_pi_neq.jl docstring). Closed form:
                    #   Φ = 2ρν·(S_ij S_ij − ⅓(∇·u)²) ≥ 0  (entropy-consistent)
                    # ν_eff = molecular + eddy = (kappa_mol·Pr-free) → use the
                    # flow viscosity (τ_flow), not thermal κ. ν = cs²·(τ_flow−0.5).
                    nu_phys = cs2_thermal * (tau_flow - 0.5f0) + nu_eddy
                    Sxx = g11; Syy = g22; Szz = g33
                    # S12,S13,S23 already computed above for WALE.
                    S_ddot = Sxx*Sxx + Syy*Syy + Szz*Szz +
                             2.0f0*(S12*S12 + S13*S13 + S23*S23)
                    visc_heat = 2.0f0 * rho_eq * nu_phys *
                                (S_ddot - (div_u*div_u) / 3.0f0)
                    visc_heat = max(visc_heat, 0.0f0)   # dissipation is ≥ 0
                    S_E = 2.0f0 * (pressure_work + visc_heat)
                    e_int = rho_eq * Cv * T_local        # internal energy density
                    # S_E enters as ρCv·ΔT = 0.5·S_E, so bounding ΔT to ±25% of T
                    # means |S_E| ≤ 0.5·e_int (since 0.5·S_E ≤ 0.25·e_int).
                    s_e_cap = 0.5f0 * e_int
                    S_E = clamp(S_E, -s_e_cap, s_e_cap)
                    S_E_half = 0.5f0 * S_E
                end

                # ── Partial-mirror BGK collision for thermal distributions ──
                # Pure mirror (σ=0) discards all even modes → T stays constant.
                # Pure BGK (σ=1) preserves all modes → checkerboard instability.
                # Partial mirror (σ≈0.1) keeps physical energy content while
                # strongly damping ghost modes that cause even-odd decoupling.
                #   g* = g_eq + (1-ω_g) · [h_g + σ · s_g]
                # where h_g = ½(g_neq - g_neq_opp) (odd, heat flux)
                #       s_g = ½(g_neq + g_neq_opp) (even, energy + ghosts)
                sigma_even = 0.0f0  # Default: pure mirror for even modes.
                thermal_physics_blend = 0.0f0
                # Phase B4: adaptive sigma_even from local |Π_neq^pre| (quadratic blend).
                # Low in smooth regions (kill even modes → clean), higher in physics
                # regions (preserve energy/thermal content). Same convention as flow
                # kernel's h-damping but with OPPOSITE sense: here LOW sigma_even=more
                # damping, HIGH=less. So we INCREASE sigma_even with |Π_neq|.
                if variable_sigma_th_i == Int32(1)
                    pxx_t = pi_neq_in[x, y, z, b_idx, 1]
                    pyy_t = pi_neq_in[x, y, z, b_idx, 2]
                    pzz_t = pi_neq_in[x, y, z, b_idx, 3]
                    pxy_t = pi_neq_in[x, y, z, b_idx, 4]
                    pxz_t = pi_neq_in[x, y, z, b_idx, 5]
                    pyz_t = pi_neq_in[x, y, z, b_idx, 6]
                    pi_mag_post_t = sqrt(pxx_t*pxx_t + pyy_t*pyy_t + pzz_t*pzz_t +
                                         2.0f0*(pxy_t*pxy_t + pxz_t*pxz_t + pyz_t*pyz_t))
                    # τ-correction post→pre using flow tau (tau_flow)
                    denom_tau_t = tau_flow - 1.0f0
                    tau_corr_t = (abs(denom_tau_t) < 1.0f-3) ? 1.0f0 : (tau_flow / denom_tau_t)
                    pi_mag_pre_t = abs(tau_corr_t) * pi_mag_post_t
                    # Quadratic blend, INCREASING with |Π_neq| (opposite of flow sigma)
                    t_blend_t = min(pi_mag_pre_t / max(sigma_th_pi_ref, 1.0f-10), 1.0f0)
                    blend_t = t_blend_t * t_blend_t
                    thermal_physics_blend = blend_t
                    sigma_even = sigma_ev_min + (sigma_ev_max - sigma_ev_min) * blend_t
                end
                # Do not make the even-mode damping a function of AMR geometry.
                # The old frontier clamp was a rectangular thermal operator: it
                # reduced ringing locally, but the rectangle itself appeared in
                # temperature and then in schlieren. Let the physics-dependent
                # sigma and wall treatment decide the mirror strength.
                if wall_dist_arr[x, y, z, b_idx] < 99.0f0
                    # Wall-adjacent thermal populations are the source of the
                    # row-wise ringing seen in the temperature slices. Enforce a
                    # pure mirror there: keep the odd heat-flux mode, discard
                    # even thermal ghost/energy oscillations. Interior shock and
                    # wake cells still use the adaptive sigma above.
                    sigma_even = 0.0f0
                end

                thermal_odd_damp = 0.0f0
                if is_frapolli
                    if is_level_1 == Int32(0)
                        # The persistent diagonal checkerboard is carried mostly by
                        # the odd heat-flux/parity component. Leave physical/shock
                        # cells almost untouched, but bleed the smooth refined-grid
                        # odd mode a little every substep so it cannot accumulate.
                        thermal_odd_damp = 0.04f0 * (1.0f0 - thermal_physics_blend)
                    end
                    # Avoid AMR-geometry odd-mode damping. A uniform refined-level
                    # cleanup is acceptable; a frontier-only one prints AMR boxes.
                    if wall_dist_arr[x, y, z, b_idx] < 99.0f0
                        thermal_odd_damp = max(thermal_odd_damp, 0.08f0)
                    end
                end
                thermal_odd_scale = 1.0f0 - clamp(thermal_odd_damp, 0.0f0, 0.45f0)

                # Pass 1: Compute streamed values and equilibrium for all directions
                g_str = MVector{39, Float32}(undef)
                g_eq_arr = MVector{39, Float32}(undef)
                for k in Int32(1):n_dirs_thermal
                    g_val = g_out[x, y, z, b_idx, k]

                    # Sponge distribution blending
                    if sp > 0.0f0 && sponge_blend_distributions == Int32(1)
                        if is_frapolli
                            geq_target = coeff_g * T_inlet * f_eq_inlet_arr[k]
                        else
                            geq_target = calculate_thermal_equilibrium(T_inlet, u_inlet, 0.0f0, 0.0f0,
                                             w_arr[k], Float32(cx_arr[k]), Float32(cy_arr[k]), Float32(cz_arr[k]))
                        end
                        g_val = g_val * (1.0f0 - sp) + geq_target * sp
                    end
                    g_str[k] = g_val

                    # Compute equilibrium
                    if is_frapolli
                        cx_eq = Float32(cx_arr[k]) - shift_x
                        cy_eq = Float32(cy_arr[k]) - shift_y
                        cz_eq = Float32(cz_arr[k]) - shift_z
                        f_eq_k = mass_corr_eq * eval_13mom_feq_single(
                            rho_eq, lambda_eq, cx_eq, cy_eq, cz_eq, w_arr[k], cs2_thermal
                        )
                        g_eq_arr[k] = coeff_g * T_local * f_eq_k
                    else
                        g_eq_arr[k] = calculate_thermal_equilibrium(T_local, ux, uy, uz,
                                    w_arr[k], Float32(cx_arr[k]), Float32(cy_arr[k]), Float32(cz_arr[k]))
                    end
                end

                # Pass 2: Mirror decomposition + partial even-mode collision
                for k in Int32(1):n_dirs_thermal
                    kb = opp_arr[k]
                    g_neq_k  = g_str[k]  - g_eq_arr[k]
                    g_neq_kb = g_str[kb] - g_eq_arr[kb]

                    # Mirror decomposition:
                    # h_g = ½(g_neq - g_neq_opp) — odd (heat flux, physical)
                    # s_g = ½(g_neq + g_neq_opp) — even (energy + ghost modes)
                    h_g_k = 0.5f0 * (g_neq_k - g_neq_kb)
                    s_g_k = 0.5f0 * (g_neq_k + g_neq_kb)

                    # Partial-mirror collision. The even mode is mirror-filtered
                    # by sigma_even; the odd heat-flux mode gets an additional
                    # AMR-aware cleanup because the observed diagonal checkerboard
                    # is an odd/parity mode that pure mirror collision preserves.
                    g_coll = g_eq_arr[k] + (1.0f0 - omega_g) *
                             (thermal_odd_scale * h_g_k + sigma_even * s_g_k)

                    # SOTA #1: inject the per-direction energy source. Sum over k
                    # of S_E_half·w_k = S_E_half (weights sum to 1) → 2·Σg gains
                    # S_E. Zero when energy_source_active==0 (S_E_half stays 0).
                    g_coll += S_E_half * w_arr[k]

                    # Positivity enforcement
                    g_coll = max(g_coll, Float32(1e-10))

                    if store_post_collision == Int32(1)
                        g_post_collision[x, y, z, b_idx, k] = g_coll
                    end
                    g_out[x, y, z, b_idx, k] = g_coll
                end
            end
        end
    end
end
