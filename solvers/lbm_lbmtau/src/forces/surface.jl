using Printf
using KernelAbstractions
using Atomix


mutable struct ForceAccumulator
    sum_Fx_p::Float64;  sum_Fy_p::Float64;  sum_Fz_p::Float64
    sum_Fx_v::Float64;  sum_Fy_v::Float64;  sum_Fz_v::Float64
    sum_Mx::Float64;    sum_My::Float64;    sum_Mz::Float64

    min_Cd::Float64;    max_Cd::Float64
    min_Cl::Float64;    max_Cl::Float64
    min_Cs::Float64;    max_Cs::Float64

    rho_ref::Float64
    u_ref::Float64
    area_ref::Float64
    chord_ref::Float64

    n_samples::Int
end

function ForceAccumulator(fd)
    ForceAccumulator(
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        Inf, -Inf,
        Inf, -Inf,
        Inf, -Inf,
        fd.rho_ref, fd.u_ref, fd.area_ref, fd.chord_ref,
        0
    )
end

function accumulate_forces!(acc::ForceAccumulator, fd)
    acc.sum_Fx_p += fd.Fx_pressure
    acc.sum_Fy_p += fd.Fy_pressure
    acc.sum_Fz_p += fd.Fz_pressure
    acc.sum_Fx_v += fd.Fx_viscous
    acc.sum_Fy_v += fd.Fy_viscous
    acc.sum_Fz_v += fd.Fz_viscous
    acc.sum_Mx   += fd.Mx
    acc.sum_My   += fd.My
    acc.sum_Mz   += fd.Mz

    acc.min_Cd = min(acc.min_Cd, fd.Cd)
    acc.max_Cd = max(acc.max_Cd, fd.Cd)
    acc.min_Cl = min(acc.min_Cl, fd.Cl)
    acc.max_Cl = max(acc.max_Cl, fd.Cl)
    acc.min_Cs = min(acc.min_Cs, fd.Cs)
    acc.max_Cs = max(acc.max_Cs, fd.Cs)

    acc.n_samples += 1
    return nothing
end

function coefficient_divergence_metrics(
    acc::ForceAccumulator,
    avg_Cd::Float64,
    avg_Cl::Float64,
    avg_Cs::Float64;
    ratio_limit::Float64=100.0,
    peak_floor::Float64=1.0,
)
    peak_Cd = max(abs(acc.min_Cd), abs(acc.max_Cd))
    peak_Cl = max(abs(acc.min_Cl), abs(acc.max_Cl))
    peak_Cs = max(abs(acc.min_Cs), abs(acc.max_Cs))
    ratio_Cd = peak_Cd / max(abs(avg_Cd), 1.0e-12)
    ratio_Cl = peak_Cl / max(abs(avg_Cl), 1.0e-12)
    ratio_Cs = peak_Cs / max(abs(avg_Cs), 1.0e-12)
    finite = all(isfinite, (peak_Cd, peak_Cl, peak_Cs, ratio_Cd, ratio_Cl, ratio_Cs))
    flagged = !finite ||
              (peak_Cd > peak_floor && ratio_Cd > ratio_limit) ||
              (peak_Cl > peak_floor && ratio_Cl > ratio_limit) ||
              (peak_Cs > peak_floor && ratio_Cs > ratio_limit)
    return (
        flagged = flagged,
        peak_Cd = peak_Cd,
        peak_Cl = peak_Cl,
        peak_Cs = peak_Cs,
        ratio_Cd = ratio_Cd,
        ratio_Cl = ratio_Cl,
        ratio_Cs = ratio_Cs,
    )
end

function print_averaged_summary(acc::ForceAccumulator; re_number::Float64=0.0,
                                 ma_physical::Float64=0.0, ma_lattice::Float64=0.0,
                                 compressibility_correction::Bool=false)
    n = acc.n_samples
    if n == 0
        println("[Forces] No samples accumulated (all steps during ramp?)")
        return
    end

    inv_n = 1.0 / n

    avg_Fx_p = acc.sum_Fx_p * inv_n
    avg_Fy_p = acc.sum_Fy_p * inv_n
    avg_Fz_p = acc.sum_Fz_p * inv_n
    avg_Fx_v = acc.sum_Fx_v * inv_n
    avg_Fy_v = acc.sum_Fy_v * inv_n
    avg_Fz_v = acc.sum_Fz_v * inv_n
    avg_Mx   = acc.sum_Mx * inv_n
    avg_My   = acc.sum_My * inv_n
    avg_Mz   = acc.sum_Mz * inv_n

    avg_Fx = avg_Fx_p + avg_Fx_v
    avg_Fy = avg_Fy_p + avg_Fy_v
    avg_Fz = avg_Fz_p + avg_Fz_v

    q_inf = 0.5 * acc.rho_ref * acc.u_ref^2
    F_ref = q_inf * acc.area_ref
    M_ref = F_ref * acc.chord_ref

    avg_Cd  = F_ref > 1e-10 ? avg_Fx / F_ref : 0.0
    avg_Cl  = F_ref > 1e-10 ? avg_Fz / F_ref : 0.0
    avg_Cs  = F_ref > 1e-10 ? avg_Fy / F_ref : 0.0
    avg_Cdp = F_ref > 1e-10 ? avg_Fx_p / F_ref : 0.0
    avg_Cdv = F_ref > 1e-10 ? avg_Fx_v / F_ref : 0.0
    avg_Cmx = M_ref > 1e-10 ? avg_Mx / M_ref : 0.0
    avg_Cmy = M_ref > 1e-10 ? avg_My / M_ref : 0.0
    avg_Cmz = M_ref > 1e-10 ? avg_Mz / M_ref : 0.0

    pct_pressure = abs(avg_Fx) > 1e-10 ? 100.0 * abs(avg_Fx_p) / (abs(avg_Fx_p) + abs(avg_Fx_v)) : 0.0

    println()
    println("============================================================")
    println("          TIME-AVERAGED AERODYNAMIC FORCES")
    @printf("          (%d samples after ramp)\n", n)
    if compressibility_correction
        println("          (Compressible solver: O(u⁴) equilibrium + DDF)")
    end
    println("============================================================")
    println()
    println("Reference Values:")
    @printf("  Re     = %.2e\n", re_number)
    @printf("  Ma     = %.4f (physical), %.4f (lattice)\n", ma_physical, ma_lattice)
    @printf("  ρ_ref  = %.4f kg/m³\n", acc.rho_ref)
    @printf("  U_ref  = %.4f m/s\n", acc.u_ref)
    @printf("  A_ref  = %.4f m²\n", acc.area_ref)
    @printf("  L_ref  = %.4f m\n", acc.chord_ref)
    @printf("  q_∞    = %.4f Pa\n", q_inf)
    println()
    println("Averaged Forces [N]:")
    @printf("  Fx (drag)  = %+.4e  (pressure: %+.4e, viscous: %+.4e)\n", avg_Fx, avg_Fx_p, avg_Fx_v)
    @printf("  Fy (side)  = %+.4e  (pressure: %+.4e, viscous: %+.4e)\n", avg_Fy, avg_Fy_p, avg_Fy_v)
    @printf("  Fz (lift)  = %+.4e  (pressure: %+.4e, viscous: %+.4e)\n", avg_Fz, avg_Fz_p, avg_Fz_v)
    println()
    println("Averaged Moments [N·m]:")
    @printf("  Mx (roll)  = %+.4e\n", avg_Mx)
    @printf("  My (pitch) = %+.4e\n", avg_My)
    @printf("  Mz (yaw)   = %+.4e\n", avg_Mz)
    println()
    println("Averaged Coefficients:")
    @printf("  Cd  = %+.6f  (Cdp = %+.6f, Cdv = %+.6f)\n", avg_Cd, avg_Cdp, avg_Cdv)
    @printf("  Cl  = %+.6f\n", avg_Cl)
    @printf("  Cs  = %+.6f\n", avg_Cs)
    @printf("  Cmy = %+.6f\n", avg_Cmy)
    println()
    println("Coefficient Ranges (instantaneous min / max):")
    @printf("  Cd  ∈ [%+.6f, %+.6f]\n", acc.min_Cd, acc.max_Cd)
    @printf("  Cl  ∈ [%+.6f, %+.6f]\n", acc.min_Cl, acc.max_Cl)
    @printf("  Cs  ∈ [%+.6f, %+.6f]\n", acc.min_Cs, acc.max_Cs)
    println()
    divergence = coefficient_divergence_metrics(acc, avg_Cd, avg_Cl, avg_Cs)
    if divergence.flagged
        @printf("[Forces] WARNING: coefficient history looks divergent. Peaks Cd/Cl/Cs=%.3e/%.3e/%.3e, peak/mean=%.1f/%.1f/%.1f\n",
                divergence.peak_Cd, divergence.peak_Cl, divergence.peak_Cs,
                divergence.ratio_Cd, divergence.ratio_Cl, divergence.ratio_Cs)
        println()
    end
    @printf("Drag breakdown: %.1f%% pressure, %.1f%% viscous\n", pct_pressure, 100.0 - pct_pressure)
    println("============================================================")
end



@inline function compute_stress_from_cell(
    rho_val::Float32,
    ux::Float32, uy::Float32, uz::Float32,
    nx::Float32, ny::Float32, nz::Float32,
    dist_to_wall::Float32,
    tau_molecular::Float32,
    pressure_scale::Float32,
    stress_scale::Float32,
    c_wale::Float32,
    nu_sgs_bg::Float32,
    cs2_val::Float32,
    T_val::Float32=1.0f0   # Phase 4: temperature for coupled EOS p = ρ·cs²·T
)
    # Guard against unphysical density (uninitialized/interior cells)
    if !isfinite(rho_val) || rho_val < 0.01f0 || rho_val > 10.0f0
        return (0.0f0, 0.0f0, 0.0f0, 0.0f0)
    end
    if !isfinite(ux) || !isfinite(uy) || !isfinite(uz)
        return (0.0f0, 0.0f0, 0.0f0, 0.0f0)
    end
    # Guard against unphysical velocity magnitude
    u_mag_sq = ux*ux + uy*uy + uz*uz
    if u_mag_sq > 100.0f0  # |u| > 10 lattice units
        return (0.0f0, 0.0f0, 0.0f0, 0.0f0)
    end

    # Phase 4: Coupled EOS — p = ρ·cs²·T, gauge pressure = (ρ·T - ρ_ref·T_ref)·cs²
    # When T=1.0 (default, athermal), this reduces to (ρ-1)·cs² (unchanged).
    p_gauge_lat = (rho_val * T_val - 1.0f0) * cs2_val
    p_phys = p_gauge_lat * pressure_scale

    u_dot_n = ux * nx + uy * ny + uz * nz
    ut_x = ux - u_dot_n * nx
    ut_y = uy - u_dot_n * ny
    ut_z = uz - u_dot_n * nz

    u_tan_mag = sqrt(ut_x * ut_x + ut_y * ut_y + ut_z * ut_z)

    nu_mol = (tau_molecular - 0.5f0) * cs2_val

    # Estimate SGS viscosity from local shear rate: |S| ≈ u_tan / d_wall
    # WALE model: nu_sgs = (Cw·Δ)² · |S|  (simplified; Δ = 1 in lattice units)
    #
    # ARTIFACT FIX (2026-06-24): the crude near-wall strain estimate S ≈ u_tan/d_wall
    # is SINGULAR as d_wall→0. Combined with the τ ∝ nu_eff·u_tan/d_wall step below,
    # the SGS contribution scales as (u_tan/d_wall)² — quadratic in 1/d_wall. Where a
    # few triangles snap their nearest good probe to the minimum distance (cell-snap
    # scatter), this self-amplifies into isolated shear spikes (the surface "hot spots"
    # against a ~0 background). A real WALE/mixing-length closure is bounded by the
    # resolved strain invariants; the surrogate here is not, so we cap it. Two bounds:
    #   (a) floor d_wall at 1.0 cell (a sub-cell probe distance is not resolvable on
    #       this grid — the gradient over <1 dx is below the discretization scale), and
    #   (b) cap nu_sgs at a multiple of nu_mol so the eddy viscosity cannot dwarf the
    #       molecular value at a near-singular probe (Smagorinsky-style realizability).
    # In the freestream (u_tan→0 or large d_wall) both bounds are inactive ⇒ no change
    # to the converged Cdv there; they only clip the unphysical near-wall spikes.
    nu_sgs = 0.0f0
    d_wall_eff = max(dist_to_wall, 1.0f0)
    if c_wale > 0.0f0 && u_tan_mag > 1.0f-10
        S_mag = u_tan_mag / d_wall_eff
        nu_sgs = (c_wale * c_wale) * S_mag
    end
    nu_sgs = max(nu_sgs, nu_sgs_bg)
    # Realizability cap: eddy viscosity bounded relative to molecular (NU_SGS_MAX_RATIO·nu_mol).
    nu_sgs = min(nu_sgs, 20.0f0 * nu_mol)
    nu_eff = nu_mol + nu_sgs

    tau_x = 0.0f0
    tau_y = 0.0f0
    tau_z = 0.0f0

    if u_tan_mag > 1.0f-10 && dist_to_wall > 0.01f0
        # Use the SAME floored distance as the SGS estimate above. The wall-law
        # gradient u_tan/d_wall is meaningless below ~1 cell on this grid, and an
        # unfloored 1/dist_to_wall here is the other half of the spike mechanism.
        tau_lat_mag = rho_val * nu_eff * u_tan_mag / d_wall_eff
        tau_phys_mag = tau_lat_mag * stress_scale

        tau_x = (ut_x / u_tan_mag) * tau_phys_mag
        tau_y = (ut_y / u_tan_mag) * tau_phys_mag
        tau_z = (ut_z / u_tan_mag) * tau_phys_mag
    end

    # Final safety: ensure all outputs are finite
    if !isfinite(p_phys) || !isfinite(tau_x) || !isfinite(tau_y) || !isfinite(tau_z)
        return (0.0f0, 0.0f0, 0.0f0, 0.0f0)
    end

    return (p_phys, tau_x, tau_y, tau_z)
end

@inline function get_cell_at_position(
    gx::Int32, gy::Int32, gz::Int32,
    block_ptr, 
    block_size::Int32,
    dim_x::Int32, dim_y::Int32, dim_z::Int32
)
    # Periodic-Y AMR: probe stencils near the seam read the wrapped cell (the
    # triangle itself must lie in the period — see map_stresses_kernel!).
    gy = _pwrap_y(gy, dim_y * block_size)
    if gx < Int32(1) || gy < Int32(1) || gz < Int32(1)
        return (Int32(0), Int32(0), Int32(0), Int32(0), false)
    end

    bs = block_size
    bx = (gx - Int32(1)) ÷ bs + Int32(1)
    by = (gy - Int32(1)) ÷ bs + Int32(1)
    bz = (gz - Int32(1)) ÷ bs + Int32(1)
    
    if bx < Int32(1) || bx > dim_x || by < Int32(1) || by > dim_y || bz < Int32(1) || bz > dim_z
        return (Int32(0), Int32(0), Int32(0), Int32(0), false)
    end
    
    b_idx = block_ptr[bx, by, bz]
    if b_idx <= Int32(0)
        return (Int32(0), Int32(0), Int32(0), Int32(0), false)
    end
    
    lx = ((gx - Int32(1)) % bs) + Int32(1)
    ly = ((gy - Int32(1)) % bs) + Int32(1)
    lz = ((gz - Int32(1)) % bs) + Int32(1)
    
    return (b_idx, lx, ly, lz, true)
end


# Trilinearly sample the 6-component symmetric Π_neq tensor at a probe point.
# Component order: 1=xx, 2=yy, 3=zz, 4=xy, 5=xz, 6=yz.
# Returns (pxx, pyy, pzz, pxy, pxz, pyz, ok, w_total). Skips obstacle cells.
@inline function trilinear_sample_pi_neq(
    pi_neq_arr, obstacle_arr, block_ptr,
    px::Float32, py::Float32, pz::Float32,
    dx::Float32, block_size::Int32,
    dim_x::Int32, dim_y::Int32, dim_z::Int32
)
    inv_dx = 1.0f0 / dx
    fx = px * inv_dx - 0.5f0
    fy = py * inv_dx - 0.5f0
    fz = pz * inv_dx - 0.5f0

    ix0 = floor(Int32, fx) + Int32(1)
    iy0 = floor(Int32, fy) + Int32(1)
    iz0 = floor(Int32, fz) + Int32(1)

    wx = clamp(fx - Float32(ix0 - Int32(1)), 0.0f0, 1.0f0)
    wy = clamp(fy - Float32(iy0 - Int32(1)), 0.0f0, 1.0f0)
    wz = clamp(fz - Float32(iz0 - Int32(1)), 0.0f0, 1.0f0)

    w0x = 1.0f0 - wx; w1x = wx
    w0y = 1.0f0 - wy; w1y = wy
    w0z = 1.0f0 - wz; w1z = wz

    pxx = 0.0f0; pyy = 0.0f0; pzz = 0.0f0
    pxy = 0.0f0; pxz = 0.0f0; pyz = 0.0f0
    w_acc = 0.0f0

    # 8 trilinear corners inlined manually — captures don't compose well in GPU kernels.
    # Corner (0,0,0)
    let gx=ix0, gy=iy0, gz=iz0, wk=w0x*w0y*w0z
        b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
        if valid && !obstacle_arr[lx, ly, lz, b_idx]
            pxx += wk * pi_neq_arr[lx, ly, lz, b_idx, 1]
            pyy += wk * pi_neq_arr[lx, ly, lz, b_idx, 2]
            pzz += wk * pi_neq_arr[lx, ly, lz, b_idx, 3]
            pxy += wk * pi_neq_arr[lx, ly, lz, b_idx, 4]
            pxz += wk * pi_neq_arr[lx, ly, lz, b_idx, 5]
            pyz += wk * pi_neq_arr[lx, ly, lz, b_idx, 6]
            w_acc += wk
        end
    end
    let gx=ix0+Int32(1), gy=iy0, gz=iz0, wk=w1x*w0y*w0z
        b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
        if valid && !obstacle_arr[lx, ly, lz, b_idx]
            pxx += wk * pi_neq_arr[lx, ly, lz, b_idx, 1]
            pyy += wk * pi_neq_arr[lx, ly, lz, b_idx, 2]
            pzz += wk * pi_neq_arr[lx, ly, lz, b_idx, 3]
            pxy += wk * pi_neq_arr[lx, ly, lz, b_idx, 4]
            pxz += wk * pi_neq_arr[lx, ly, lz, b_idx, 5]
            pyz += wk * pi_neq_arr[lx, ly, lz, b_idx, 6]
            w_acc += wk
        end
    end
    let gx=ix0, gy=iy0+Int32(1), gz=iz0, wk=w0x*w1y*w0z
        b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
        if valid && !obstacle_arr[lx, ly, lz, b_idx]
            pxx += wk * pi_neq_arr[lx, ly, lz, b_idx, 1]
            pyy += wk * pi_neq_arr[lx, ly, lz, b_idx, 2]
            pzz += wk * pi_neq_arr[lx, ly, lz, b_idx, 3]
            pxy += wk * pi_neq_arr[lx, ly, lz, b_idx, 4]
            pxz += wk * pi_neq_arr[lx, ly, lz, b_idx, 5]
            pyz += wk * pi_neq_arr[lx, ly, lz, b_idx, 6]
            w_acc += wk
        end
    end
    let gx=ix0+Int32(1), gy=iy0+Int32(1), gz=iz0, wk=w1x*w1y*w0z
        b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
        if valid && !obstacle_arr[lx, ly, lz, b_idx]
            pxx += wk * pi_neq_arr[lx, ly, lz, b_idx, 1]
            pyy += wk * pi_neq_arr[lx, ly, lz, b_idx, 2]
            pzz += wk * pi_neq_arr[lx, ly, lz, b_idx, 3]
            pxy += wk * pi_neq_arr[lx, ly, lz, b_idx, 4]
            pxz += wk * pi_neq_arr[lx, ly, lz, b_idx, 5]
            pyz += wk * pi_neq_arr[lx, ly, lz, b_idx, 6]
            w_acc += wk
        end
    end
    let gx=ix0, gy=iy0, gz=iz0+Int32(1), wk=w0x*w0y*w1z
        b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
        if valid && !obstacle_arr[lx, ly, lz, b_idx]
            pxx += wk * pi_neq_arr[lx, ly, lz, b_idx, 1]
            pyy += wk * pi_neq_arr[lx, ly, lz, b_idx, 2]
            pzz += wk * pi_neq_arr[lx, ly, lz, b_idx, 3]
            pxy += wk * pi_neq_arr[lx, ly, lz, b_idx, 4]
            pxz += wk * pi_neq_arr[lx, ly, lz, b_idx, 5]
            pyz += wk * pi_neq_arr[lx, ly, lz, b_idx, 6]
            w_acc += wk
        end
    end
    let gx=ix0+Int32(1), gy=iy0, gz=iz0+Int32(1), wk=w1x*w0y*w1z
        b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
        if valid && !obstacle_arr[lx, ly, lz, b_idx]
            pxx += wk * pi_neq_arr[lx, ly, lz, b_idx, 1]
            pyy += wk * pi_neq_arr[lx, ly, lz, b_idx, 2]
            pzz += wk * pi_neq_arr[lx, ly, lz, b_idx, 3]
            pxy += wk * pi_neq_arr[lx, ly, lz, b_idx, 4]
            pxz += wk * pi_neq_arr[lx, ly, lz, b_idx, 5]
            pyz += wk * pi_neq_arr[lx, ly, lz, b_idx, 6]
            w_acc += wk
        end
    end
    let gx=ix0, gy=iy0+Int32(1), gz=iz0+Int32(1), wk=w0x*w1y*w1z
        b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
        if valid && !obstacle_arr[lx, ly, lz, b_idx]
            pxx += wk * pi_neq_arr[lx, ly, lz, b_idx, 1]
            pyy += wk * pi_neq_arr[lx, ly, lz, b_idx, 2]
            pzz += wk * pi_neq_arr[lx, ly, lz, b_idx, 3]
            pxy += wk * pi_neq_arr[lx, ly, lz, b_idx, 4]
            pxz += wk * pi_neq_arr[lx, ly, lz, b_idx, 5]
            pyz += wk * pi_neq_arr[lx, ly, lz, b_idx, 6]
            w_acc += wk
        end
    end
    let gx=ix0+Int32(1), gy=iy0+Int32(1), gz=iz0+Int32(1), wk=w1x*w1y*w1z
        b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
        if valid && !obstacle_arr[lx, ly, lz, b_idx]
            pxx += wk * pi_neq_arr[lx, ly, lz, b_idx, 1]
            pyy += wk * pi_neq_arr[lx, ly, lz, b_idx, 2]
            pzz += wk * pi_neq_arr[lx, ly, lz, b_idx, 3]
            pxy += wk * pi_neq_arr[lx, ly, lz, b_idx, 4]
            pxz += wk * pi_neq_arr[lx, ly, lz, b_idx, 5]
            pyz += wk * pi_neq_arr[lx, ly, lz, b_idx, 6]
            w_acc += wk
        end
    end

    if w_acc > 0.01f0
        inv_w = 1.0f0 / w_acc
        return (pxx * inv_w, pyy * inv_w, pzz * inv_w,
                pxy * inv_w, pxz * inv_w, pyz * inv_w, true, w_acc)
    end
    return (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, false, 0.0f0)
end


# Compute wall traction from sampled post-collision Π_neq:
#   σ_viscous,ij = −(τ−0.5)/(τ−1) · Π_neq^post,ij        (Chapman-Enskog + τ-correction)
#   pressure     = (ρT − 1)·cs² (gauge, same as Prandtl path)
#   t_i          = σ_ij · n_j (viscous traction vector from fluid on wall along outward normal)
# Returns (p_phys, tx_phys, ty_phys, tz_phys). pressure_scale and stress_scale convert lattice to SI.
@inline function compute_stress_from_pi_neq(
    rho_val::Float32, T_val::Float32,
    nx::Float32, ny::Float32, nz::Float32,
    pxx::Float32, pyy::Float32, pzz::Float32,
    pxy::Float32, pxz::Float32, pyz::Float32,
    tau_post::Float32,
    pressure_scale::Float32, stress_scale::Float32,
    cs2_val::Float32
)
    if !isfinite(rho_val) || rho_val < 0.01f0 || rho_val > 10.0f0
        return (0.0f0, 0.0f0, 0.0f0, 0.0f0)
    end

    # Gauge pressure (matches Prandtl path exactly)
    p_gauge_lat = (rho_val * T_val - 1.0f0) * cs2_val
    p_phys = p_gauge_lat * pressure_scale

    # τ-correction factor: σ_viscous,ij = −f_corr · Π_neq^post,ij
    # f_corr = (τ−0.5)/(τ−1). Guard against τ≈1 singularity (inviscid-ish) and τ<=0.5.
    denom = tau_post - 1.0f0
    # When |denom| is tiny, we're at τ≈1 where post≈pre is degenerate for this formula.
    # Fall back to the simpler identity: σ = −(1 − ω/2) · Π_neq^pre ≈ −0.5·Π_neq^pre ≈ −0.5·Π_neq^post
    # which at τ=1 gives ω=1, so f_corr should be 0.5. Switch smoothly:
    f_corr = (abs(denom) < 1.0f-3) ? 0.5f0 : ((tau_post - 0.5f0) / denom)

    # Use DEVIATORIC (traceless) Π_neq for viscous traction. The trace of Π_neq
    # contains bulk-viscosity effects from ∇·u which dominates across shocks in
    # compressible flow. That bulk pressure is already accounted for via the
    # EOS pressure term p = (ρT−1)·cs². Including the trace would double-count
    # and inflate viscous drag at supersonic speeds.
    trace_over_3 = (pxx + pyy + pzz) * (1.0f0/3.0f0)
    pxx_d = pxx - trace_over_3
    pyy_d = pyy - trace_over_3
    pzz_d = pzz - trace_over_3
    # Off-diagonal components are already traceless (symmetric Π_αβ, α≠β)

    # Viscous traction: t_i = σ_ij · n_j (using deviatoric Π)
    tx_lat = -f_corr * (pxx_d*nx + pxy*ny + pxz*nz)
    ty_lat = -f_corr * (pxy*nx + pyy_d*ny + pyz*nz)
    tz_lat = -f_corr * (pxz*nx + pyz*ny + pzz_d*nz)

    tx_phys = tx_lat * stress_scale
    ty_phys = ty_lat * stress_scale
    tz_phys = tz_lat * stress_scale

    if !isfinite(p_phys) || !isfinite(tx_phys) || !isfinite(ty_phys) || !isfinite(tz_phys)
        return (0.0f0, 0.0f0, 0.0f0, 0.0f0)
    end

    return (p_phys, tx_phys, ty_phys, tz_phys)
end


@inline function trilinear_sample_flow(
    rho_arr, vel_arr, obstacle_arr, block_ptr,
    px::Float32, py::Float32, pz::Float32,
    dx::Float32, block_size::Int32,
    dim_x::Int32, dim_y::Int32, dim_z::Int32
)
    inv_dx = 1.0f0 / dx
    fx = px * inv_dx - 0.5f0
    fy = py * inv_dx - 0.5f0
    fz = pz * inv_dx - 0.5f0

    ix0 = floor(Int32, fx) + Int32(1)
    iy0 = floor(Int32, fy) + Int32(1)
    iz0 = floor(Int32, fz) + Int32(1)

    wx = clamp(fx - Float32(ix0 - Int32(1)), 0.0f0, 1.0f0)
    wy = clamp(fy - Float32(iy0 - Int32(1)), 0.0f0, 1.0f0)
    wz = clamp(fz - Float32(iz0 - Int32(1)), 0.0f0, 1.0f0)

    w0x = 1.0f0 - wx;  w1x = wx
    w0y = 1.0f0 - wy;  w1y = wy
    w0z = 1.0f0 - wz;  w1z = wz

    rho_acc = 0.0f0
    ux_acc  = 0.0f0
    uy_acc  = 0.0f0
    uz_acc  = 0.0f0
    w_acc   = 0.0f0

    gx = ix0;              gy = iy0;              gz = iz0
    w_k = w0x * w0y * w0z
    b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
    if valid && !obstacle_arr[lx, ly, lz, b_idx]
        rho_acc += w_k * rho_arr[lx, ly, lz, b_idx]
        ux_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 1]
        uy_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 2]
        uz_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 3]
        w_acc   += w_k
    end

    gx = ix0 + Int32(1);  gy = iy0;              gz = iz0
    w_k = w1x * w0y * w0z
    b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
    if valid && !obstacle_arr[lx, ly, lz, b_idx]
        rho_acc += w_k * rho_arr[lx, ly, lz, b_idx]
        ux_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 1]
        uy_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 2]
        uz_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 3]
        w_acc   += w_k
    end

    gx = ix0;              gy = iy0 + Int32(1);  gz = iz0
    w_k = w0x * w1y * w0z
    b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
    if valid && !obstacle_arr[lx, ly, lz, b_idx]
        rho_acc += w_k * rho_arr[lx, ly, lz, b_idx]
        ux_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 1]
        uy_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 2]
        uz_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 3]
        w_acc   += w_k
    end

    gx = ix0 + Int32(1);  gy = iy0 + Int32(1);  gz = iz0
    w_k = w1x * w1y * w0z
    b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
    if valid && !obstacle_arr[lx, ly, lz, b_idx]
        rho_acc += w_k * rho_arr[lx, ly, lz, b_idx]
        ux_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 1]
        uy_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 2]
        uz_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 3]
        w_acc   += w_k
    end

    gx = ix0;              gy = iy0;              gz = iz0 + Int32(1)
    w_k = w0x * w0y * w1z
    b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
    if valid && !obstacle_arr[lx, ly, lz, b_idx]
        rho_acc += w_k * rho_arr[lx, ly, lz, b_idx]
        ux_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 1]
        uy_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 2]
        uz_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 3]
        w_acc   += w_k
    end

    gx = ix0 + Int32(1);  gy = iy0;              gz = iz0 + Int32(1)
    w_k = w1x * w0y * w1z
    b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
    if valid && !obstacle_arr[lx, ly, lz, b_idx]
        rho_acc += w_k * rho_arr[lx, ly, lz, b_idx]
        ux_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 1]
        uy_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 2]
        uz_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 3]
        w_acc   += w_k
    end

    gx = ix0;              gy = iy0 + Int32(1);  gz = iz0 + Int32(1)
    w_k = w0x * w1y * w1z
    b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
    if valid && !obstacle_arr[lx, ly, lz, b_idx]
        rho_acc += w_k * rho_arr[lx, ly, lz, b_idx]
        ux_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 1]
        uy_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 2]
        uz_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 3]
        w_acc   += w_k
    end

    gx = ix0 + Int32(1);  gy = iy0 + Int32(1);  gz = iz0 + Int32(1)
    w_k = w1x * w1y * w1z
    b_idx, lx, ly, lz, valid = get_cell_at_position(gx, gy, gz, block_ptr, block_size, dim_x, dim_y, dim_z)
    if valid && !obstacle_arr[lx, ly, lz, b_idx]
        rho_acc += w_k * rho_arr[lx, ly, lz, b_idx]
        ux_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 1]
        uy_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 2]
        uz_acc  += w_k * vel_arr[lx, ly, lz, b_idx, 3]
        w_acc   += w_k
    end

    if w_acc > 0.01f0
        inv_w = 1.0f0 / w_acc
        r_out = rho_acc * inv_w
        u_out = ux_acc * inv_w
        v_out = uy_acc * inv_w
        w_out = uz_acc * inv_w
        # Reject if interpolated values are non-finite
        if !isfinite(r_out) || !isfinite(u_out) || !isfinite(v_out) || !isfinite(w_out)
            return (1.0f0, 0.0f0, 0.0f0, 0.0f0, false, 0.0f0)
        end
        return (r_out, u_out, v_out, w_out, true, w_acc)
    end
    return (1.0f0, 0.0f0, 0.0f0, 0.0f0, false, 0.0f0)
end



@kernel function map_stresses_kernel!(
    p_map, sx_map, sy_map, sz_map,
    wall_dist_map,                   # Phase 1.7: physical distance from triangle to probe (m)
    cx_tri, cy_tri, cz_tri,
    nx_tri, ny_tri, nz_tri,
    rho, vel, obstacle, block_ptr,
    block_size::Int32, dx::Float32,
    dim_x::Int32, dim_y::Int32, dim_z::Int32,
    mesh_offset_x::Float32, mesh_offset_y::Float32, mesh_offset_z::Float32,
    pressure_scale::Float32, stress_scale::Float32,
    tau_molecular::Float32,
    search_radius::Int32,
    c_wale::Float32,
    nu_sgs_bg::Float32,
    cs2_val::Float32,
    temperature_arr,                # Phase 4: temperature field (or dummy = rho when no thermal)
    thermal_coupling::Int32,        # Phase 4: 0=isothermal, 1=coupled (p = ρ·cs²·T)
    pi_neq_arr,                     # Phase A4: Π_neq tensor field (dummy 1x1x1x1x6 when unused)
    force_method_i::Int32           # 0=prandtl (1D wall-law), 1=stress_tensor (σ·n from Π_neq)
)
    i = @index(Global)

    @inbounds begin
        tx = cx_tri[i] + mesh_offset_x
        ty = cy_tri[i] + mesh_offset_y
        tz = cz_tri[i] + mesh_offset_z

        # Periodic-Y AMR: cell lookups wrap across the Y seam, so a triangle whose
        # centroid lies OUTSIDE the period (an overspan copy of the extruded body)
        # must be excluded explicitly — single-level runs got that for free from
        # the out-of-domain lookups. Constant-false when the gate is off.
        tri_outside_period = _PERIODIC_Y_AMR &&
            !(ty >= 0.0f0 && ty < Float32(dim_y * block_size) * dx)

        n_x = nx_tri[i]
        n_y = ny_tri[i]
        n_z = nz_tri[i]

        best_rho       = 1.0f0
        best_ux        = 0.0f0
        best_uy        = 0.0f0
        best_uz        = 0.0f0
        best_T         = 1.0f0    # Phase 4: temperature at probe point
        best_wall_dist = Float32(search_radius + 1)
        best_quality   = 0.0f0
        found_fluid    = false

        # Accumulators for SHEPARD-WEIGHTED PRESSURE probe (antidote to cell-snap
        # striation on the surface). Pressure depends only on ρ and T, which vary
        # slowly across the near-wall layer — averaging 3–4 probes along the outward
        # normal removes the O(dx) step discontinuities that appear when adjacent
        # triangles snap to different interpolation stencils. Shear stays on the
        # closest-good probe (best_rho, best_u, best_wall_dist) because it depends
        # explicitly on the wall distance via the velocity gradient estimate.
        p_rho_accum = 0.0f0
        p_T_accum   = 0.0f0
        p_w_accum   = 0.0f0

        # Multi-sample probe: scan ALL distances, keep CLOSEST with acceptable quality.
        # Strategy: accept any probe with w_q >= min_quality, preferring closer ones.
        # If no probe meets threshold, fall back to best-quality probe at any distance.
        # Lower threshold (0.25 vs old 0.5) captures near-wall cells in shock regions
        # where the trilinear stencil is partially in solid — still directionally accurate.
        min_quality = 0.25f0
        pressure_probe_max = min(Int32(4), search_radius)  # only average the first few probes

        # Pass 1: find closest probe with quality >= min_quality
        for probe_i in Int32(1):search_radius
            d_cells = Float32(probe_i) + 0.5f0

            probe_x = tx + n_x * d_cells * dx
            probe_y = ty + n_y * d_cells * dx
            probe_z = tz + n_z * d_cells * dx

            r, ux, uy, uz, ok, w_q = trilinear_sample_flow(
                rho, vel, obstacle, block_ptr,
                probe_x, probe_y, probe_z,
                dx, block_size, dim_x, dim_y, dim_z
            )

            # Phase 4: temperature at same probe
            T_here = 1.0f0
            T_ok_here = false
            if ok && thermal_coupling == Int32(1)
                T_r, _, _, _, T_ok, _ = trilinear_sample_flow(
                    temperature_arr, vel, obstacle, block_ptr,
                    probe_x, probe_y, probe_z,
                    dx, block_size, dim_x, dim_y, dim_z
                )
                if T_ok
                    T_here = T_r
                    T_ok_here = true
                end
            end

            # Accumulate PRESSURE probes within the first few near-wall distances.
            # Weights are the interpolation-quality metric; distant probes naturally
            # drop out via the inner multi-scale logic below.
            if ok && w_q >= min_quality && probe_i <= pressure_probe_max
                p_rho_accum += w_q * r
                p_T_accum   += w_q * (T_ok_here ? T_here : 1.0f0)
                p_w_accum   += w_q
            end

            if ok && w_q >= min_quality
                # Good quality at this distance — LOCK this as the "shear probe".
                # Do NOT break: we keep looping up to pressure_probe_max so the
                # Shepard sum (above) accumulates several probes for a smooth
                # pressure field. best_* is written only on the first good probe
                # (`!found_fluid` guard below), so subsequent probes contribute
                # to the pressure average but not to the shear probe.
                if !found_fluid
                    best_rho       = r
                    best_ux        = ux
                    best_uy        = uy
                    best_uz        = uz
                    best_wall_dist = d_cells
                    best_quality   = w_q
                    found_fluid    = true
                    if T_ok_here; best_T = T_here; end
                end
                # Exit once we've accumulated enough pressure samples.
                if probe_i >= pressure_probe_max
                    break
                end
            elseif ok && w_q > best_quality && !found_fluid
                # Below threshold but non-zero — keep as fallback for shear.
                # Guarded by `!found_fluid` so we don't overwrite a previously-
                # locked good probe with a later weaker one.
                best_rho       = r
                best_ux        = ux
                best_uy        = uy
                best_uz        = uz
                best_wall_dist = d_cells
                best_quality   = w_q
                found_fluid    = true
                if T_ok_here; best_T = T_here; end
            end
        end

        # Finalize Shepard-averaged pressure inputs. If enough probes accumulated,
        # use the averaged ρ/T for the pressure EOS (smooth surface field). Otherwise
        # fall through to the single best_rho/best_T values (staircased but at least
        # defined).
        pressure_rho = p_w_accum > 0.5f0 ? p_rho_accum / p_w_accum : best_rho
        pressure_T   = p_w_accum > 0.5f0 ? p_T_accum   / p_w_accum : best_T

        if !found_fluid
            gx_f = tx / dx
            gy_f = ty / dx
            gz_f = tz / dx
            g_x = floor(Int32, gx_f) + Int32(1)
            g_y = floor(Int32, gy_f) + Int32(1)
            g_z = floor(Int32, gz_f) + Int32(1)

            best_dist_sq = Float32(1e10)

            for radius in Int32(1):search_radius
                if found_fluid
                    break
                end
                for dz in -radius:radius
                    for dy in -radius:radius
                        for ddx in -radius:radius
                            at_shell = (abs(ddx) == radius) || (abs(dy) == radius) || (abs(dz) == radius)
                            if !at_shell
                                continue
                            end

                            check_gx = g_x + Int32(ddx)
                            check_gy = g_y + Int32(dy)
                            check_gz = g_z + Int32(dz)

                            b_idx, lx, ly, lz, valid = get_cell_at_position(
                                check_gx, check_gy, check_gz,
                                block_ptr, block_size, dim_x, dim_y, dim_z
                            )

                            if valid && !obstacle[lx, ly, lz, b_idx]
                                # Validate cell data before accepting
                                cell_rho = rho[lx, ly, lz, b_idx]
                                if !isfinite(cell_rho) || cell_rho < 0.01f0 || cell_rho > 100.0f0
                                    continue
                                end

                                cell_cx = (Float32(check_gx) - 0.5f0) * dx
                                cell_cy = (Float32(check_gy) - 0.5f0) * dx
                                cell_cz = (Float32(check_gz) - 0.5f0) * dx

                                disp_x = cell_cx - tx
                                disp_y = cell_cy - ty
                                disp_z = cell_cz - tz

                                dot_n = disp_x * n_x + disp_y * n_y + disp_z * n_z
                                if dot_n <= 0.0f0
                                    continue
                                end

                                dist_sq = disp_x^2 + disp_y^2 + disp_z^2
                                if dist_sq < best_dist_sq
                                    best_dist_sq = dist_sq
                                    best_rho = cell_rho
                                    best_ux  = vel[lx, ly, lz, b_idx, 1]
                                    best_uy  = vel[lx, ly, lz, b_idx, 2]
                                    best_uz  = vel[lx, ly, lz, b_idx, 3]
                                    best_wall_dist = dot_n / dx
                                    # Phase 4: read temperature at same cell
                                    if thermal_coupling == Int32(1)
                                        cell_T = temperature_arr[lx, ly, lz, b_idx]
                                        if isfinite(cell_T) && cell_T > 0.01f0
                                            best_T = cell_T
                                        end
                                    end
                                    found_fluid = true
                                end
                            end
                        end
                    end
                end
            end
        end

        p_val = 0.0f0
        tau_x = 0.0f0
        tau_y = 0.0f0
        tau_z = 0.0f0

        if found_fluid
            wall_dist = max(best_wall_dist, 0.5f0)

            if force_method_i == Int32(1)
                # Stress-tensor path: sample Π_neq and project σ_ij n_j where
                # σ = −(τ−0.5)/(τ−1)·Π_neq^post.
                #
                # ARTIFACT FIX (2026-06-24): a SINGLE Π_neq sample at wall_dist makes
                # the surface shear inherit the per-triangle probe-distance scatter
                # (adjacent triangles snap to different trilinear stencils → O(dx)
                # cell-snap striation, the staircased look). The pressure path already
                # smooths this with a Shepard (quality-weighted) average over the first
                # few near-wall probes; the shear path did not. Mirror that here:
                # accumulate the Π_neq tensor over probe distances 1.5,2.5,..,pmax+0.5
                # cells along the outward normal, weighted by interpolation quality.
                # Π_neq varies slowly across the thin near-wall band, so the average is
                # a faithful wall value with the cell-snap noise removed. Falls back to
                # the single closest-good probe if fewer than ~1 quality-unit accrues.
                pxx_acc = 0.0f0; pyy_acc = 0.0f0; pzz_acc = 0.0f0
                pxy_acc = 0.0f0; pxz_acc = 0.0f0; pyz_acc = 0.0f0
                pi_w_acc = 0.0f0
                pi_nprobes = min(Int32(4), search_radius)
                for pj in Int32(1):pi_nprobes
                    d_pj = Float32(pj) + 0.5f0
                    qpx = tx + n_x * d_pj * dx
                    qpy = ty + n_y * d_pj * dx
                    qpz = tz + n_z * d_pj * dx
                    pxx_j, pyy_j, pzz_j, pxy_j, pxz_j, pyz_j, ok_j, wq_j =
                        trilinear_sample_pi_neq(
                            pi_neq_arr, obstacle, block_ptr,
                            qpx, qpy, qpz,
                            dx, block_size, dim_x, dim_y, dim_z
                        )
                    if ok_j && wq_j >= 0.25f0
                        pxx_acc += wq_j * pxx_j; pyy_acc += wq_j * pyy_j; pzz_acc += wq_j * pzz_j
                        pxy_acc += wq_j * pxy_j; pxz_acc += wq_j * pxz_j; pyz_acc += wq_j * pyz_j
                        pi_w_acc += wq_j
                    end
                end
                # GPU-safe: declare-then-conditionally-overwrite (no `local`/branch-
                # only assignment, which breaks KernelAbstractions lowering). Default
                # to the single closest-good probe; overwrite with the Shepard average
                # when enough quality probes accrued.
                probe_x = tx + n_x * wall_dist * dx
                probe_y = ty + n_y * wall_dist * dx
                probe_z = tz + n_z * wall_dist * dx
                pxx_s, pyy_s, pzz_s, pxy_s, pxz_s, pyz_s, pi_ok, _ =
                    trilinear_sample_pi_neq(
                        pi_neq_arr, obstacle, block_ptr,
                        probe_x, probe_y, probe_z,
                        dx, block_size, dim_x, dim_y, dim_z
                    )
                if pi_w_acc > 0.5f0
                    inv_w = 1.0f0 / pi_w_acc
                    pxx_s = pxx_acc*inv_w; pyy_s = pyy_acc*inv_w; pzz_s = pzz_acc*inv_w
                    pxy_s = pxy_acc*inv_w; pxz_s = pxz_acc*inv_w; pyz_s = pyz_acc*inv_w
                    pi_ok = true
                end
                if pi_ok
                    # AUDIT FIX (2026-06-03): the stored Π_neq is POST-collision,
                    # relaxed with the SGS-AUGMENTED tau_eff, not the molecular τ.
                    # The (τ−0.5)/(τ−1) correction must therefore use tau_eff, or
                    # the recovered viscous traction is mis-scaled exactly in the
                    # BL/wake where SGS is active (biasing friction drag Cdv).
                    # Reconstruct the local effective viscosity with the SAME
                    # WALE-style near-wall estimate the Prandtl path uses, so both
                    # force methods are consistent. In freestream (nu_sgs→0) this
                    # reduces to tau_molecular, i.e. no change.
                    # ARTIFACT FIX (2026-06-24): same bounded WALE surrogate as the
                    # Prandtl path (compute_stress_from_cell). The reconstructed
                    # tau_eff_fp multiplies the Π_neq traction via f_corr, so an
                    # unbounded (utan/wall_dist) SGS term re-injects the same
                    # near-wall spike into what should be a clean σ·n projection.
                    nu_mol_fp = (tau_molecular - 0.5f0) * cs2_val
                    udn = best_ux*n_x + best_uy*n_y + best_uz*n_z
                    utx = best_ux - udn*n_x; uty = best_uy - udn*n_y; utz = best_uz - udn*n_z
                    utan = sqrt(utx*utx + uty*uty + utz*utz)
                    wall_dist_fp = max(wall_dist, 1.0f0)
                    nu_sgs_fp = 0.0f0
                    if c_wale > 0.0f0 && utan > 1.0f-10
                        nu_sgs_fp = (c_wale * c_wale) * (utan / wall_dist_fp)
                    end
                    nu_sgs_fp = max(nu_sgs_fp, nu_sgs_bg)
                    nu_sgs_fp = min(nu_sgs_fp, 20.0f0 * nu_mol_fp)
                    tau_eff_fp = 0.5f0 + (nu_mol_fp + nu_sgs_fp) / cs2_val
                    p_val, tau_x, tau_y, tau_z = compute_stress_from_pi_neq(
                        best_rho, best_T,
                        n_x, n_y, n_z,
                        pxx_s, pyy_s, pzz_s, pxy_s, pxz_s, pyz_s,
                        tau_eff_fp,
                        pressure_scale, stress_scale, cs2_val
                    )
                else
                    # Π_neq sample invalid (all corners in solid) — fall back to Prandtl
                    p_val, tau_x, tau_y, tau_z = compute_stress_from_cell(
                        best_rho, best_ux, best_uy, best_uz,
                        n_x, n_y, n_z,
                        wall_dist, tau_molecular,
                        pressure_scale, stress_scale,
                        c_wale, nu_sgs_bg, cs2_val,
                        best_T
                    )
                end
            else
                # Default: 1D Prandtl wall-law estimate
                p_val, tau_x, tau_y, tau_z = compute_stress_from_cell(
                    best_rho, best_ux, best_uy, best_uz,
                    n_x, n_y, n_z,
                    wall_dist, tau_molecular,
                    pressure_scale, stress_scale,
                    c_wale, nu_sgs_bg, cs2_val,
                    best_T   # Phase 4: temperature for coupled EOS
                )
            end

            # Override p_val with the Shepard-averaged pressure built from multiple
            # probes along the outward normal. ρ and T are slowly varying across the
            # near-wall layer, so averaging 2–4 near-wall probes weighted by their
            # interpolation-quality factor smooths out O(dx) cell-snap striations
            # that single-probe sampling leaves on the surface. Shear stays on the
            # closest-good-probe values (best_rho, wall_dist, best_u) since it
            # depends on the near-wall gradient.
            if isfinite(pressure_rho) && pressure_rho >= 0.01f0 && pressure_rho <= 10.0f0 &&
               isfinite(pressure_T)   && pressure_T   >= 0.1f0  && pressure_T   <= 10.0f0
                p_val = (pressure_rho * pressure_T - 1.0f0) * cs2_val * pressure_scale
            end
        end

        if tri_outside_period   # periodic-Y AMR overspan copy: zero force (single-level semantics)
            p_val = 0.0f0; tau_x = 0.0f0; tau_y = 0.0f0; tau_z = 0.0f0
            found_fluid = false
        end

        p_map[i] = p_val
        sx_map[i] = tau_x
        sy_map[i] = tau_y
        sz_map[i] = tau_z
        # Physical wall distance of the probe cell (used for y⁺ computation).
        # best_wall_dist is in *cells*; multiplying by dx gives meters.
        wall_dist_map[i] = found_fluid ? best_wall_dist * dx : 0.0f0
    end
end



@kernel function map_minus_side_only_kernel!(
    p_map_minus, sx_map_minus, sy_map_minus, sz_map_minus,
    cx_tri, cy_tri, cz_tri,
    nx_tri, ny_tri, nz_tri,
    rho, vel, obstacle, block_ptr,
    block_size::Int32, dx::Float32,
    dim_x::Int32, dim_y::Int32, dim_z::Int32,
    mesh_offset_x::Float32, mesh_offset_y::Float32, mesh_offset_z::Float32,
    pressure_scale::Float32, stress_scale::Float32,
    tau_molecular::Float32,
    search_radius::Int32,
    c_wale::Float32,
    nu_sgs_bg::Float32,
    cs2_val::Float32,
    temperature_arr,                # Phase 4: temperature field
    thermal_coupling::Int32         # Phase 4: 0=isothermal, 1=coupled
)
    i = @index(Global)

    @inbounds begin
        tx = cx_tri[i] + mesh_offset_x
        ty = cy_tri[i] + mesh_offset_y
        tz = cz_tri[i] + mesh_offset_z

        n_x = nx_tri[i]
        n_y = ny_tri[i]
        n_z = nz_tri[i]

        # --- Probe only the MINUS side (opposite to outward normal) ---
        minus_rho       = 1.0f0
        minus_ux        = 0.0f0
        minus_uy        = 0.0f0
        minus_uz        = 0.0f0
        minus_T         = 1.0f0    # Phase 4
        minus_wall_dist = Float32(search_radius + 1)
        minus_quality   = 0.0f0
        found_minus     = false

        min_quality = 0.25f0

        # Ray-march along -normal: scan all distances, keep closest with acceptable quality
        for probe_i in Int32(1):search_radius
            d_cells = Float32(probe_i) + 0.5f0

            probe_x = tx - n_x * d_cells * dx
            probe_y = ty - n_y * d_cells * dx
            probe_z = tz - n_z * d_cells * dx

            r, ux, uy, uz, ok, w_q = trilinear_sample_flow(
                rho, vel, obstacle, block_ptr,
                probe_x, probe_y, probe_z,
                dx, block_size, dim_x, dim_y, dim_z
            )

            if ok && w_q >= min_quality
                minus_rho       = r
                minus_ux        = ux
                minus_uy        = uy
                minus_uz        = uz
                minus_wall_dist = d_cells
                minus_quality   = w_q
                found_minus     = true
                if thermal_coupling == Int32(1)
                    T_r, _, _, _, T_ok, _ = trilinear_sample_flow(
                        temperature_arr, vel, obstacle, block_ptr,
                        probe_x, probe_y, probe_z,
                        dx, block_size, dim_x, dim_y, dim_z
                    )
                    if T_ok; minus_T = T_r; end
                end
                break
            elseif ok && w_q > minus_quality
                minus_rho       = r
                minus_ux        = ux
                minus_uy        = uy
                minus_uz        = uz
                minus_wall_dist = d_cells
                minus_quality   = w_q
                found_minus     = true
                if thermal_coupling == Int32(1)
                    T_r, _, _, _, T_ok, _ = trilinear_sample_flow(
                        temperature_arr, vel, obstacle, block_ptr,
                        probe_x, probe_y, probe_z,
                        dx, block_size, dim_x, dim_y, dim_z
                    )
                    if T_ok; minus_T = T_r; end
                end
            end
        end

        # Brute-force fallback: find nearest fluid cell on the minus side
        if !found_minus
            gx_f = tx / dx
            gy_f = ty / dx
            gz_f = tz / dx
            g_x = floor(Int32, gx_f) + Int32(1)
            g_y = floor(Int32, gy_f) + Int32(1)
            g_z = floor(Int32, gz_f) + Int32(1)

            best_dist_sq = Float32(1e10)

            for radius in Int32(1):search_radius
                if found_minus; break; end
                for dz in -radius:radius
                    for dy in -radius:radius
                        for ddx in -radius:radius
                            at_shell = (abs(ddx) == radius) || (abs(dy) == radius) || (abs(dz) == radius)
                            if !at_shell; continue; end

                            check_gx = g_x + Int32(ddx)
                            check_gy = g_y + Int32(dy)
                            check_gz = g_z + Int32(dz)

                            b_idx, lx, ly, lz, valid = get_cell_at_position(
                                check_gx, check_gy, check_gz,
                                block_ptr, block_size, dim_x, dim_y, dim_z
                            )

                            if valid && !obstacle[lx, ly, lz, b_idx]
                                # Validate cell data before accepting
                                cell_rho = rho[lx, ly, lz, b_idx]
                                if !isfinite(cell_rho) || cell_rho < 0.01f0 || cell_rho > 100.0f0
                                    continue
                                end

                                cell_cx = (Float32(check_gx) - 0.5f0) * dx
                                cell_cy = (Float32(check_gy) - 0.5f0) * dx
                                cell_cz = (Float32(check_gz) - 0.5f0) * dx

                                disp_x = cell_cx - tx
                                disp_y = cell_cy - ty
                                disp_z = cell_cz - tz

                                # Only accept cells on the MINUS side (dot < 0)
                                dot_n = disp_x * n_x + disp_y * n_y + disp_z * n_z
                                if dot_n >= 0.0f0; continue; end

                                dist_sq = disp_x^2 + disp_y^2 + disp_z^2
                                if dist_sq < best_dist_sq
                                    best_dist_sq = dist_sq
                                    minus_rho = cell_rho
                                    minus_ux  = vel[lx, ly, lz, b_idx, 1]
                                    minus_uy  = vel[lx, ly, lz, b_idx, 2]
                                    minus_uz  = vel[lx, ly, lz, b_idx, 3]
                                    minus_wall_dist = abs(dot_n) / dx
                                    if thermal_coupling == Int32(1)
                                        cell_T = temperature_arr[lx, ly, lz, b_idx]
                                        if isfinite(cell_T) && cell_T > 0.01f0
                                            minus_T = cell_T
                                        end
                                    end
                                    found_minus = true
                                end
                            end
                        end
                    end
                end
            end
        end

        # Compute stress from minus-side fluid (zero if no fluid found — correct for closed bodies)
        p_minus  = 0.0f0
        tx_minus = 0.0f0
        ty_minus = 0.0f0
        tz_minus = 0.0f0

        # Periodic-Y AMR: overspan triangles (centroid outside the Y period) get no
        # force (see map_stresses_kernel!). Constant-false when the gate is off.
        if _PERIODIC_Y_AMR && !(ty >= 0.0f0 && ty < Float32(dim_y * block_size) * dx)
            found_minus = false
        end
        if found_minus
            wall_dist_m = max(minus_wall_dist, 0.5f0)
            p_minus, tx_minus, ty_minus, tz_minus = compute_stress_from_cell(
                minus_rho, minus_ux, minus_uy, minus_uz,
                -n_x, -n_y, -n_z,
                wall_dist_m, tau_molecular,
                pressure_scale, stress_scale,
                c_wale, nu_sgs_bg, cs2_val,
                minus_T   # Phase 4: temperature for coupled EOS
            )
        end

        p_map_minus[i]  = p_minus
        sx_map_minus[i] = tx_minus
        sy_map_minus[i] = ty_minus
        sz_map_minus[i] = tz_minus
    end
end



@kernel function compute_net_triangle_forces_kernel!(
    net_Fx, net_Fy, net_Fz,
    p_map_plus, sx_map_plus, sy_map_plus, sz_map_plus,
    p_map_minus, sx_map_minus, sy_map_minus, sz_map_minus,
    nx_tri, ny_tri, nz_tri,
    areas
)
    i = @index(Global)

    @inbounds begin
        p_net = p_map_plus[i] - p_map_minus[i]

        sx_net = sx_map_plus[i] - sx_map_minus[i]
        sy_net = sy_map_plus[i] - sy_map_minus[i]
        sz_net = sz_map_plus[i] - sz_map_minus[i]

        nx = nx_tri[i]
        ny = ny_tri[i]
        nz = nz_tri[i]
        A  = areas[i]

        Fp_x = -p_net * nx * A
        Fp_y = -p_net * ny * A
        Fp_z = -p_net * nz * A

        Fv_x = sx_net * A
        Fv_y = sy_net * A
        Fv_z = sz_net * A

        net_Fx[i] = Fp_x + Fv_x
        net_Fy[i] = Fp_y + Fv_y
        net_Fz[i] = Fp_z + Fv_z
    end
end



@kernel function integrate_forces_kernel!(
    Fx_p_acc, Fy_p_acc, Fz_p_acc,
    Fx_v_acc, Fy_v_acc, Fz_v_acc,
    Mx_acc, My_acc, Mz_acc,
    p_map, sx_map, sy_map, sz_map,
    cx_tri, cy_tri, cz_tri,
    nx_tri, ny_tri, nz_tri,
    areas,
    offset_x::Float32, offset_y::Float32, offset_z::Float32,
    ref_x::Float32, ref_y::Float32, ref_z::Float32
)
    i = @index(Global)

    @inbounds begin
        p = p_map[i]
        tau_x = sx_map[i]
        tau_y = sy_map[i]
        tau_z = sz_map[i]

        # Skip triangles with Inf/NaN pressure (prevents poisoning the sum)
        if isfinite(p) && isfinite(tau_x) && isfinite(tau_y) && isfinite(tau_z)
            nx = nx_tri[i]
            ny = ny_tri[i]
            nz = nz_tri[i]
            A = areas[i]

            cx = cx_tri[i] + offset_x
            cy = cy_tri[i] + offset_y
            cz = cz_tri[i] + offset_z

            dFp_x = -p * nx * A
            dFp_y = -p * ny * A
            dFp_z = -p * nz * A

            dFv_x = tau_x * A
            dFv_y = tau_y * A
            dFv_z = tau_z * A

            dFx = dFp_x + dFv_x
            dFy = dFp_y + dFv_y
            dFz = dFp_z + dFv_z

            rx = cx - ref_x
            ry = cy - ref_y
            rz = cz - ref_z

            dMx = ry * dFz - rz * dFy
            dMy = rz * dFx - rx * dFz
            dMz = rx * dFy - ry * dFx

            Atomix.@atomic Fx_p_acc[1] += dFp_x
            Atomix.@atomic Fy_p_acc[1] += dFp_y
            Atomix.@atomic Fz_p_acc[1] += dFp_z

            Atomix.@atomic Fx_v_acc[1] += dFv_x
            Atomix.@atomic Fy_v_acc[1] += dFv_y
            Atomix.@atomic Fz_v_acc[1] += dFv_z

            Atomix.@atomic Mx_acc[1] += dMx
            Atomix.@atomic My_acc[1] += dMy
            Atomix.@atomic Mz_acc[1] += dMz
        end
    end
end



function map_surface_stresses!(force_data::ForceData, level, gpu_mesh,
                               backend, params;
                               search_radius::Int=8,
                               cx_gpu=nothing, cy_gpu=nothing, cz_gpu=nothing, w_gpu=nothing)

    mesh_offset = params.mesh_offset
    tau_mol = level.tau

    velocity_scale = params.velocity_scale
    rho_phys = params.rho_physical

    pressure_scale = Float32(rho_phys * velocity_scale * velocity_scale)
    stress_scale = pressure_scale

    kernel! = map_stresses_kernel!(backend)
    # Phase 4: pass temperature array for coupled EOS p = ρ·cs²·T
    ddf_active = DDF_ENABLED && has_thermal(level)
    temp_arr = ddf_active ? level.temperature : level.rho  # dummy=rho when no thermal
    thermal_flag = (ddf_active && LATTICE_TYPE == :d3q39) ? Int32(1) : Int32(0)

    # Phase A4: stress-tensor force extraction
    # Compute Π_neq transiently when FORCE_METHOD == :stress_tensor AND lattice
    # arrays were provided. Falls back to Prandtl if lattice arrays missing.
    use_stress_tensor = (FORCE_METHOD == :stress_tensor) && cx_gpu !== nothing &&
                        cy_gpu !== nothing && cz_gpu !== nothing && w_gpu !== nothing
    local pi_neq_arr
    if use_stress_tensor
        # Force_level is always finest level → level.f is always current (levels 2+
        # end every coarse step on odd sub-step, which writes to level.f).
        shift_x_f = Float32(LATTICE_SHIFT_X)
        shift_y_f = Float32(LATTICE_SHIFT_Y)
        shift_z_f = Float32(LATTICE_SHIFT_Z)
        pi_neq_arr = compute_pi_neq_for_level!(level, level.f,
                                               cx_gpu, cy_gpu, cz_gpu, w_gpu, backend;
                                               shift_x=shift_x_f, shift_y=shift_y_f, shift_z=shift_z_f)
        force_method_i = Int32(1)
    else
        # Dummy 1×1×1×1×6 stub — kernel never reads it when force_method_i=0
        pi_neq_arr = KernelAbstractions.zeros(backend, Float32, 1, 1, 1, 1, 6)
        force_method_i = Int32(0)
    end

    kernel!(
        force_data.pressure_map,
        force_data.shear_x_map, force_data.shear_y_map, force_data.shear_z_map,
        force_data.wall_dist_map,
        gpu_mesh.centers_x, gpu_mesh.centers_y, gpu_mesh.centers_z,
        gpu_mesh.normals_x, gpu_mesh.normals_y, gpu_mesh.normals_z,
        level.rho, level.vel, level.obstacle, level.block_pointer,
        Int32(BLOCK_SIZE), Float32(level.dx),
        Int32(level.grid_dim_x), Int32(level.grid_dim_y), Int32(level.grid_dim_z),
        Float32(mesh_offset[1]), Float32(mesh_offset[2]), Float32(mesh_offset[3]),
        pressure_scale, stress_scale,
        Float32(tau_mol),
        Int32(search_radius),
        Float32(C_SMAGO),
        Float32(NU_SGS_BACKGROUND),
        Float32(CS2_PHYSICS),
        temp_arr,
        thermal_flag,
        pi_neq_arr,
        force_method_i,
        ndrange=(gpu_mesh.n_triangles,)
    )

    KernelAbstractions.synchronize(backend)

    if !force_data.diagnostics_printed
        p_cpu = Array(force_data.pressure_map)
        sx_cpu = Array(force_data.shear_x_map)
        sy_cpu = Array(force_data.shear_y_map)
        sz_cpu = Array(force_data.shear_z_map)

        n_total = gpu_mesh.n_triangles
        n_nonzero_p = count(x -> abs(x) > 1e-10, p_cpu)
        n_nonzero_s = count(x -> abs(x) > 1e-10, sx_cpu)

        # Compute total |shear| per triangle for viscous coverage
        shear_mag = sqrt.(sx_cpu.^2 .+ sy_cpu.^2 .+ sz_cpu.^2)
        n_nonzero_shear = count(x -> x > 1e-10, shear_mag)

        println("[Surface] Mapped $(n_total) triangles (nearest-cell probing, quality threshold=0.25):")
        println("          Mesh offset: ($(round(mesh_offset[1], digits=3)), $(round(mesh_offset[2], digits=3)), $(round(mesh_offset[3], digits=3)))")
        println("          Grid spacing dx = $(round(level.dx, digits=6)) m, search_radius = $(search_radius)")
        println("          Pressure scale = $(round(pressure_scale, digits=2)) Pa")
        println("          Coverage: pressure=$(n_nonzero_p)/$(n_total) ($(round(100*n_nonzero_p/n_total, digits=1))%), shear=$(n_nonzero_shear)/$(n_total) ($(round(100*n_nonzero_shear/n_total, digits=1))%)")

        if n_nonzero_p > 0
            p_nz = filter(x -> abs(x) > 1e-10, p_cpu)
            n_inf = count(x -> !isfinite(x), p_nz)
            if n_inf > 0
                @printf("          WARNING: %d triangles have Inf/NaN pressure (filtered from stats)\n", n_inf)
            end
            p_finite = filter(isfinite, p_nz)
            if length(p_finite) > 0
                p_abs = abs.(p_finite)
                p_mean = sum(p_abs) / length(p_abs)
                p_std = sqrt(sum((p_abs .- p_mean).^2) / length(p_abs))
                @printf("          Pressure range: [%.3e, %.3e] Pa\n", minimum(p_finite), maximum(p_finite))
                @printf("          Pressure |p|: mean=%.3e, std=%.3e, CoV=%.1f%%\n",
                        p_mean, p_std, 100.0 * p_std / max(p_mean, 1e-20))
            end
        end

        if n_nonzero_shear > 0
            s_nz = filter(x -> x > 1e-10, shear_mag)
            s_finite = filter(isfinite, s_nz)
            if length(s_finite) > 0
                @printf("          Shear |τ|: mean=%.3e, max=%.3e Pa\n",
                        sum(s_finite)/length(s_finite), maximum(s_finite))
            end
        end

        force_data.diagnostics_printed = true
    end
end

function integrate_surface_forces!(force_data::ForceData, gpu_mesh, backend, params)
    n_tri = gpu_mesh.n_triangles

    reset_forces!(force_data)

    Fx_p_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Fy_p_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Fz_p_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Fx_v_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Fy_v_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Fz_v_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Mx_acc = KernelAbstractions.zeros(backend, Float32, 1)
    My_acc = KernelAbstractions.zeros(backend, Float32, 1)
    Mz_acc = KernelAbstractions.zeros(backend, Float32, 1)

    mc = force_data.moment_center
    mesh_offset = params.mesh_offset

    kernel! = integrate_forces_kernel!(backend)
    kernel!(
        Fx_p_acc, Fy_p_acc, Fz_p_acc,
        Fx_v_acc, Fy_v_acc, Fz_v_acc,
        Mx_acc, My_acc, Mz_acc,
        force_data.pressure_map,
        force_data.shear_x_map, force_data.shear_y_map, force_data.shear_z_map,
        gpu_mesh.centers_x, gpu_mesh.centers_y, gpu_mesh.centers_z,
        gpu_mesh.normals_x, gpu_mesh.normals_y, gpu_mesh.normals_z,
        gpu_mesh.areas,
        Float32(mesh_offset[1]), Float32(mesh_offset[2]), Float32(mesh_offset[3]),
        Float32(mc[1]), Float32(mc[2]), Float32(mc[3]),
        ndrange=(n_tri,)
    )

    KernelAbstractions.synchronize(backend)

    Fx_p = Float64(Array(Fx_p_acc)[1])
    Fy_p = Float64(Array(Fy_p_acc)[1])
    Fz_p = Float64(Array(Fz_p_acc)[1])
    Fx_v = Float64(Array(Fx_v_acc)[1])
    Fy_v = Float64(Array(Fy_v_acc)[1])
    Fz_v = Float64(Array(Fz_v_acc)[1])
    Mx = Float64(Array(Mx_acc)[1])
    My = Float64(Array(My_acc)[1])
    Mz = Float64(Array(Mz_acc)[1])

    if force_data.symmetric
        Fx_p *= 2.0; Fz_p *= 2.0
        Fx_v *= 2.0; Fz_v *= 2.0
        My *= 2.0
        Fy_p = 0.0; Fy_v = 0.0
        Mx = 0.0; Mz = 0.0
    end

    force_data.Fx_pressure = Fx_p
    force_data.Fy_pressure = Fy_p
    force_data.Fz_pressure = Fz_p
    force_data.Fx_viscous = Fx_v
    force_data.Fy_viscous = Fy_v
    force_data.Fz_viscous = Fz_v

    force_data.Fx = Fx_p + Fx_v
    force_data.Fy = Fy_p + Fy_v
    force_data.Fz = Fz_p + Fz_v

    force_data.Mx = Mx
    force_data.My = My
    force_data.Mz = Mz

    q_inf = 0.5 * force_data.rho_ref * force_data.u_ref^2
    F_ref = q_inf * force_data.area_ref
    M_ref = F_ref * force_data.chord_ref

    if F_ref > 1e-10
        force_data.Cd = force_data.Fx / F_ref
        force_data.Cl = force_data.Fz / F_ref
        force_data.Cs = force_data.Fy / F_ref
    end

    if M_ref > 1e-10
        force_data.Cmx = force_data.Mx / M_ref
        force_data.Cmy = force_data.My / M_ref
        force_data.Cmz = force_data.Mz / M_ref
    end
end

"""
Accumulate running (time-averaged) means of the instantaneous surface Cp and
pressure maps, and of the scalar Cd / Cl / Cs / Cm coefficients. Called by the
main loop at the same cadence as `compute_aerodynamics!` (once per diagnostic
tick is fine). Uses the numerically stable Welford update
    x̄_n = x̄_{n-1} + (x_n − x̄_{n-1}) / n
which needs no accumulator for sum-of-squares and avoids catastrophic
cancellation for large n.

The averages restart from zero on a fresh solver start. For a case that already
ramped up, ignore the averages until post-ramp steady-state is reached — this
is a property of the *reader*, not the writer.
"""
function accumulate_force_averages!(force_data::ForceData, _backend=nothing)
    force_data.avg_n += 1
    n = Float64(force_data.avg_n)
    inv_n = Float32(1.0 / n)

    q_inf = Float32(0.5 * force_data.rho_ref * force_data.u_ref^2)
    inv_q = q_inf > Float32(1.0e-10) ? Float32(1) / q_inf : Float32(0)

    # GPU-side running means of per-triangle pressure and Cp (outer side).
    # x̄_n = x̄_{n-1} + (x_n − x̄_{n-1}) / n  →  broadcast over triangles.
    p_map = force_data.pressure_map
    p_avg = force_data.pressure_map_avg
    cp_avg = force_data.cp_map_avg

    @. p_avg = p_avg + (p_map - p_avg) * inv_n
    @. cp_avg = cp_avg + (p_map * inv_q - cp_avg) * inv_n

    # CPU scalars.
    w = 1.0 / n
    force_data.Cd_avg  += (force_data.Cd  - force_data.Cd_avg)  * w
    force_data.Cl_avg  += (force_data.Cl  - force_data.Cl_avg)  * w
    force_data.Cs_avg  += (force_data.Cs  - force_data.Cs_avg)  * w
    force_data.Cmx_avg += (force_data.Cmx - force_data.Cmx_avg) * w
    force_data.Cmy_avg += (force_data.Cmy - force_data.Cmy_avg) * w
    force_data.Cmz_avg += (force_data.Cmz - force_data.Cmz_avg) * w
end

function compute_aerodynamics!(force_data::ForceData, level, gpu_mesh, backend, params;
                               search_radius::Int=8,
                               cx_gpu=nothing, cy_gpu=nothing, cz_gpu=nothing, w_gpu=nothing)
    map_surface_stresses!(force_data, level, gpu_mesh, backend, params;
                          search_radius=search_radius,
                          cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu)

    integrate_surface_forces!(force_data, gpu_mesh, backend, params)
end



function map_minus_side_stresses!(force_data::ForceData, level, gpu_mesh,
                                  backend, params; search_radius::Int=8)

    mesh_offset = params.mesh_offset
    tau_mol = level.tau

    velocity_scale = params.velocity_scale
    rho_phys = params.rho_physical

    pressure_scale = Float32(rho_phys * velocity_scale * velocity_scale)
    stress_scale = pressure_scale

    # Phase 4: pass temperature array for coupled EOS
    ddf_active = DDF_ENABLED && has_thermal(level)
    temp_arr = ddf_active ? level.temperature : level.rho
    thermal_flag = (ddf_active && LATTICE_TYPE == :d3q39) ? Int32(1) : Int32(0)

    kernel! = map_minus_side_only_kernel!(backend)
    kernel!(
        force_data.pressure_map_minus,
        force_data.shear_x_map_minus, force_data.shear_y_map_minus, force_data.shear_z_map_minus,
        gpu_mesh.centers_x, gpu_mesh.centers_y, gpu_mesh.centers_z,
        gpu_mesh.normals_x, gpu_mesh.normals_y, gpu_mesh.normals_z,
        level.rho, level.vel, level.obstacle, level.block_pointer,
        Int32(BLOCK_SIZE), Float32(level.dx),
        Int32(level.grid_dim_x), Int32(level.grid_dim_y), Int32(level.grid_dim_z),
        Float32(mesh_offset[1]), Float32(mesh_offset[2]), Float32(mesh_offset[3]),
        pressure_scale, stress_scale,
        Float32(tau_mol),
        Int32(search_radius),
        Float32(C_SMAGO),
        Float32(NU_SGS_BACKGROUND),
        Float32(CS2_PHYSICS),
        temp_arr,
        thermal_flag,
        ndrange=(gpu_mesh.n_triangles,)
    )

    KernelAbstractions.synchronize(backend)
end

function compute_net_triangle_forces!(force_data::ForceData, gpu_mesh, backend)
    n_tri = gpu_mesh.n_triangles

    net_Fx_gpu = KernelAbstractions.zeros(backend, Float32, n_tri)
    net_Fy_gpu = KernelAbstractions.zeros(backend, Float32, n_tri)
    net_Fz_gpu = KernelAbstractions.zeros(backend, Float32, n_tri)

    kernel! = compute_net_triangle_forces_kernel!(backend)
    kernel!(
        net_Fx_gpu, net_Fy_gpu, net_Fz_gpu,
        force_data.pressure_map,
        force_data.shear_x_map, force_data.shear_y_map, force_data.shear_z_map,
        force_data.pressure_map_minus,
        force_data.shear_x_map_minus, force_data.shear_y_map_minus, force_data.shear_z_map_minus,
        gpu_mesh.normals_x, gpu_mesh.normals_y, gpu_mesh.normals_z,
        gpu_mesh.areas,
        ndrange=(n_tri,)
    )

    KernelAbstractions.synchronize(backend)

    return (Array(net_Fx_gpu), Array(net_Fy_gpu), Array(net_Fz_gpu))
end

function compute_net_aerodynamics!(force_data::ForceData, level, gpu_mesh, backend, params;
                                   search_radius::Int=8, closed_body::Bool=true,
                                   cx_gpu=nothing, cy_gpu=nothing, cz_gpu=nothing, w_gpu=nothing)
    # PLUS SIDE: reuse the proven one-sided mapping kernel (map_stresses_kernel!)
    # This populates force_data.pressure_map, shear_x/y/z_map with robust plus-side values
    map_surface_stresses!(force_data, level, gpu_mesh, backend, params;
                          search_radius=search_radius,
                          cx_gpu=cx_gpu, cy_gpu=cy_gpu, cz_gpu=cz_gpu, w_gpu=w_gpu)

    if closed_body
        # For closed bodies (cube, car, aircraft fuselage, etc.):
        # No fluid exists inside the body, so minus-side contribution is exactly zero.
        # Net force = plus-side force directly.
        # This avoids spurious results from unmasked interior cells in block-structured grids.

        # Zero out minus maps (in case they held stale data)
        fill!(force_data.pressure_map_minus, 0.0f0)
        fill!(force_data.shear_x_map_minus, 0.0f0)
        fill!(force_data.shear_y_map_minus, 0.0f0)
        fill!(force_data.shear_z_map_minus, 0.0f0)

        println("[NetForce] Closed-body mode: skipping minus-side probing (net = plus-side)")

    else
        # MINUS SIDE: for thin/open surfaces where fluid exists on both sides
        map_minus_side_stresses!(force_data, level, gpu_mesh, backend, params;
                                 search_radius=search_radius)

        # --- Diagnostic: check minus-side coverage ---
        p_minus_cpu = Array(force_data.pressure_map_minus)
        n_total = gpu_mesh.n_triangles
        n_minus = count(x -> abs(x) > 1e-10, p_minus_cpu)
        println("[NetForce] Minus-side coverage: $(n_minus)/$(n_total) ($(round(100*n_minus/n_total, digits=1))%)")
        if n_minus > 0
            pm_nz = filter(x -> abs(x) > 1e-10, p_minus_cpu)
            @printf("[NetForce] Minus-side pressure range: [%.3e, %.3e] Pa\n", minimum(pm_nz), maximum(pm_nz))
        end
    end

    # Compute net force per triangle: F_net = F_plus - F_minus
    net_Fx, net_Fy, net_Fz = compute_net_triangle_forces!(force_data, gpu_mesh, backend)

    # --- Diagnostic: net force coverage ---
    p_plus_cpu = Array(force_data.pressure_map)
    n_total = gpu_mesh.n_triangles
    n_plus = count(x -> abs(x) > 1e-10, p_plus_cpu)
    net_mag = sqrt.(net_Fx.^2 .+ net_Fy.^2 .+ net_Fz.^2)
    n_nonzero_net = count(x -> x > 1e-10, net_mag)
    println("[NetForce] Plus-side  coverage: $(n_plus)/$(n_total) ($(round(100*n_plus/n_total, digits=1))%)")
    println("[NetForce] Net force coverage: $(n_nonzero_net)/$(n_total) ($(round(100*n_nonzero_net/n_total, digits=1))%)")
    if n_nonzero_net > 0
        nz_mag = filter(x -> x > 1e-10, net_mag)
        @printf("[NetForce] Net |F| range: [%.3e, %.3e] N\n", minimum(nz_mag), maximum(nz_mag))
    end
    # ---

    return (net_Fx, net_Fy, net_Fz)
end
