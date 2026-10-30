# ==============================================================================
# Grad-reconstruction (regularized) wall BC — 2026-06-05
# ==============================================================================
# A geometry- and axis-AGNOSTIC wall boundary condition that REPLACES link-wise
# Bouzidi at curved walls. Link-wise Bouzidi (BFL/YMLS/Marson) reconstructs each
# population independently from a per-link wall-distance q; on a coarse curved
# surface this is ill-conditioned at axis-tangent points (q→0/1 grazing links →
# "hot spots") and excites a cell-scale odd-even (Nyquist) velocity mode at the
# shoulders (the "ring") that no downstream damping eliminates and finer
# resolution does NOT cure (verified: SR 96→192 still rings). Both pathologies
# are artifacts of the per-LINK degrees of freedom.
#
# The Grad / regularized wall (Latt-Chopard-Malaspinas PRE 77 056703 2008;
# Chikatamarla-Karlin 2013) rebuilds ALL populations of a wall-adjacent fluid
# cell from a single MACROSCOPIC moment set (ρ, u_wall, Π_neq) — there is NO
# per-link q, NO surface normal, NO lattice-axis singling-out, so neither the
# ring nor the hot spots can form. It is exactly the D3Q19 RR (order-2 Hermite)
# projection already in stream_collide_d3q19_rr!, but anchored at the no-slip
# wall velocity u_wall instead of the local velocity:
#
#   f_k = f_eq_k(ρ, u_wall) + (1−ω)·w_k·(1/2cs⁴)·(Q_k : Π_neq)
#   Q_αβ = c_α c_β − cs²δ_αβ ,  cs²=1/3 ⇒ 1/2cs⁴ = 4.5
#
# Π_neq is recomputed locally from the post-collision populations about
# (ρ, u_wall) so the wall carries the correct viscous shear stress (set
# grad_pineq_scale<1 to relax toward a more diffusive wall).
#
# Applied over the EXISTING Bouzidi boundary-cell list (bouzidi_cell_*), which by
# construction is exactly the set of fluid cells with ≥1 wall-crossing link — so
# no new geometry pass and no in-kernel obstacle scan are needed. Runs as a
# separate post-collision kernel (like apply_bouzidi_correction!) so the already
# register-starved collision kernel is untouched. It OVERWRITES f_out at wall
# cells, so it naturally supersedes the interior RR and SVV writes there — no
# explicit disabling of those is needed.
#
# GPU-JIT-safe: reads only threaded array/scalar params (no module-global reads
# in the kernel), no MVector, uses the GPU-proven calculate_equilibrium_auto.
# ==============================================================================

using KernelAbstractions

@kernel function grad_wall_correction_kernel!(
    f_out, f_post_collision, rho_out, vel_out,
    wall_dist_arr, q_map,         # q_map: 5D bouzidi per-link sub-cell distance (x,y,z,b,k); may be nothing-typed only if mode!=2
    cell_block, cell_x, cell_y, cell_z,
    block_size::Int32,
    cx_arr, cy_arr, cz_arr, w_arr,
    cs2_val::Float32, cs4_val::Float32,
    n_dirs::Int32, cell_offset::Int32,
    omega_wall::Float32,          # 1/τ — relaxation factor on f_neq_reg (post-collision consistency)
    grad_pineq_scale::Float32,    # [0,1] keep-stress knob (1 = full Grad shear, 0 = diffusive/eq wall)
    grad_vel_mode::Int32,         # 0 = hard u_wall=0 ; 1 = wall_dist (cell-center) blend ; 2 = q-map sub-cell blend (de-staircases the ring)
    grad_blend_dx::Float32,       # dx in lattice units for the blend (=1)
    write_post_collision::Int32   # 1 = also write f_post_collision (so forces see the wall stress)
)
    cell_lin = @index(Global, Linear)   # 2026-10-27: plain `x = @index(...)` -- the KernelAbstractions CPU backend splices the work-item index only into that form (else MethodError on the CPU)
    cell_idx = cell_lin + cell_offset
    @inbounds begin
        b_idx = cell_block[cell_idx]
        x = cell_x[cell_idx]; y = cell_y[cell_idx]; z = cell_z[cell_idx]

        rho  = rho_out[x, y, z, b_idx]
        ux_l = vel_out[x, y, z, b_idx, 1]
        uy_l = vel_out[x, y, z, b_idx, 2]
        uz_l = vel_out[x, y, z, b_idx, 3]

        # ── Wall velocity target (no-slip, static body) ──
        ux_w = 0.0f0; uy_w = 0.0f0; uz_w = 0.0f0
        if grad_vel_mode == Int32(1)
            dist = wall_dist_arr[x, y, z, b_idx]
            # s∈[0,1]: 1 = on the wall (full no-slip), 0 = ≥1 cell away (keep u).
            s = dist > 0.0f0 ? clamp(1.0f0 - dist / grad_blend_dx, 0.0f0, 1.0f0) : 1.0f0
            ux_w = (1.0f0 - s) * ux_l
            uy_w = (1.0f0 - s) * uy_l
            uz_w = (1.0f0 - s) * uz_l
        elseif grad_vel_mode == Int32(2)
            # ── q-map sub-cell proximity blend — THE de-staircasing fix ─────────
            # The Bouzidi q on each wall-crossing link is the TRUE sub-cell distance
            # (∈[0,1]) from this fluid cell center to the STL surface along c_k.
            # q_cell = min over wall-crossing links = closest surface approach.
            # A cell hugging the surface (q→0) → s→1 = full no-slip; a cell whose
            # surface is a near-full cell away (q→1) → s→0 = keep velocity. Because
            # q varies CONTINUOUSLY around the curved shoulder (unlike the binary
            # "is this cell in the staircased shell" mask), the no-slip enforcement
            # becomes a smooth function of real geometry → no cell-scale ring.
            q_cell = 2.0f0   # sentinel > 1: no valid wall link found yet
            for k in Int32(1):n_dirs
                qk = Float32(q_map[x, y, z, b_idx, k])
                # valid wall-crossing link: q ∈ (0,1]. (q<0 / q≥big = no crossing.)
                if qk > 0.0f0 && qk <= 1.0f0 && qk < q_cell
                    q_cell = qk
                end
            end
            # If no valid link (shouldn't happen on the bouzidi list), fall back to
            # hard no-slip (s=1) to stay safe.
            s = q_cell <= 1.0f0 ? clamp(1.0f0 - q_cell, 0.0f0, 1.0f0) : 1.0f0
            ux_w = (1.0f0 - s) * ux_l
            uy_w = (1.0f0 - s) * uy_l
            uz_w = (1.0f0 - s) * uz_l
        end

        # ── Recompute Π_neq from post-collision pops about (ρ, u_wall) ──
        Pi_xx = 0.0f0; Pi_yy = 0.0f0; Pi_zz = 0.0f0
        Pi_xy = 0.0f0; Pi_yz = 0.0f0; Pi_zx = 0.0f0
        for k in Int32(1):n_dirs
            cxf = Float32(cx_arr[k]); cyf = Float32(cy_arr[k]); czf = Float32(cz_arr[k])
            feq = calculate_equilibrium_auto(rho, ux_w, uy_w, uz_w, w_arr[k], cxf, cyf, czf, Int32(0))
            fneq = f_post_collision[x, y, z, b_idx, k] - feq
            Pi_xx += fneq * cxf * cxf
            Pi_yy += fneq * cyf * cyf
            Pi_zz += fneq * czf * czf
            Pi_xy += fneq * cxf * cyf
            Pi_yz += fneq * cyf * czf
            Pi_zx += fneq * czf * cxf
        end
        Pi_xx *= grad_pineq_scale; Pi_yy *= grad_pineq_scale; Pi_zz *= grad_pineq_scale
        Pi_xy *= grad_pineq_scale; Pi_yz *= grad_pineq_scale; Pi_zx *= grad_pineq_scale

        # ── Reconstruct ALL populations: f_eq(ρ,u_wall) + (1−ω)·RR(Π_neq) ──
        inv_2cs4 = 0.5f0 / cs4_val          # = 4.5 for cs²=1/3
        one_m_w  = 1.0f0 - omega_wall
        for k in Int32(1):n_dirs
            cxf = Float32(cx_arr[k]); cyf = Float32(cy_arr[k]); czf = Float32(cz_arr[k]); wk = w_arr[k]
            feq = calculate_equilibrium_auto(rho, ux_w, uy_w, uz_w, wk, cxf, cyf, czf, Int32(0))
            Qxx = cxf*cxf - cs2_val; Qyy = cyf*cyf - cs2_val; Qzz = czf*czf - cs2_val
            fneq_reg = wk * inv_2cs4 * (
                Pi_xx*Qxx + Pi_yy*Qyy + Pi_zz*Qzz +
                2.0f0*(Pi_xy*cxf*cyf + Pi_yz*cyf*czf + Pi_zx*czf*cxf)
            )
            fc = feq + one_m_w * fneq_reg
            fc = max(fc, 1.0f-12)           # positivity floor (O(1e-12) mass, negligible)
            f_out[x, y, z, b_idx, k] = fc
            if write_post_collision == Int32(1)
                f_post_collision[x, y, z, b_idx, k] = fc
            end
        end

        # Macro velocity → wall target so diagnostics/forces and the next stream
        # see consistent no-slip. (ρ is unchanged: Σ f_eq = ρ, Σ f_neq_reg = 0.)
        vel_out[x, y, z, b_idx, 1] = ux_w
        vel_out[x, y, z, b_idx, 2] = uy_w
        vel_out[x, y, z, b_idx, 3] = uz_w
    end
end

# Host launcher — cloned from apply_bouzidi_correction!'s pattern, plus the macro
# arrays (rho_out/vel_out/wall_dist) the Grad reconstruction needs.
function apply_grad_wall_correction!(
    f_out, f_post_collision, rho_out, vel_out, wall_dist_arr, q_map,
    cell_block, cell_x, cell_y, cell_z, n_boundary_cells,
    block_size::Int,
    cx_gpu, cy_gpu, cz_gpu, w_gpu,
    cs2_val::Float32, cs4_val::Float32, n_dirs::Int,
    backend;
    omega_wall::Float32 = 1.0f0,
    grad_pineq_scale::Float32 = 1.0f0,
    grad_vel_mode::Int32 = Int32(0),
    grad_blend_dx::Float32 = 1.0f0,
    write_post_collision::Int32 = Int32(1),
    cell_offset::Int = 0
)
    n_boundary_cells == 0 && return nothing
    # Mode 2 (q-blend) needs a real 5D q_map. If it is absent, demote to hard
    # no-slip (mode 0) so the kernel never indexes a non-5D placeholder — the GPU
    # compiler specializes the q_map[...] index on its array type, so we MUST pass
    # a genuine 5D array whenever mode 2 could run. Caller guarantees q_map !== nothing
    # when BC_SCHEME==:grad (the bouzidi q-map is always built for that path).
    eff_mode = (grad_vel_mode == Int32(2) && q_map === nothing) ? Int32(0) : grad_vel_mode
    kernel! = grad_wall_correction_kernel!(backend)
    kernel!(
        f_out, f_post_collision, rho_out, vel_out, wall_dist_arr, q_map,
        cell_block, cell_x, cell_y, cell_z,
        Int32(block_size),
        cx_gpu, cy_gpu, cz_gpu, w_gpu,
        cs2_val, cs4_val,
        Int32(n_dirs), Int32(cell_offset),
        omega_wall, grad_pineq_scale, eff_mode, grad_blend_dx, write_post_collision;
        ndrange = (n_boundary_cells,)
    )
    return nothing
end
