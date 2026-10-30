# ==============================================================================
# DYNAMIC SUBGRID-SCALE (SGS) — Germano-Lilly procedure (SOTA B2)
# ==============================================================================
# Implements the dynamic SGS coefficient procedure of:
#   Germano, Piomelli, Moin & Cabot 1991, Phys. Fluids A 3, 1760
#     "A dynamic subgrid-scale eddy viscosity model"
#   Lilly 1992, Phys. Fluids A 4, 633
#     "A proposed modification of the Germano subgrid-scale closure method"
#
# Replaces the fixed `c_pi: 0.4` hand-tuned coefficient in variable_nu_sgs with
# a per-cell self-calibrating value computed from the local strain-rate field
# via the Germano identity. The procedure:
#   1. Test-filter velocity field at scale 2Δ:  ũ_α = ⟨u_α⟩_2Δ
#   2. Compute Leonard stress (resolved stress between Δ and 2Δ):
#        L_αβ = (u_α u_β)~ − ũ_α · ũ_β
#   3. Compute modeled stress at scale Δ: m_αβ = −2·Δ²·|S|·S_αβ
#   4. Compute modeled stress at scale 2Δ: M_αβ = −2·Δ²_test·|S̃|·S̃_αβ
#   5. Germano identity:  L_αβ^d = L_αβ − (1/3)·L_γγ·δ_αβ = c_d·(M_αβ − m̃_αβ)
#   6. Least-squares (Lilly):  c_d = ½·⟨L_αβ M_αβ⟩ / ⟨M_αβ M_αβ⟩
#   7. Local averaging or Lagrangian averaging stabilizes c_d.
#      This implementation uses LOCAL (per-cell, no averaging) for GPU friendliness.
#   8. Clamp c_d to [0, 0.25] (Cs ∈ [0, 0.5]) to prevent anti-damping.
#
# Simplifications vs. the canonical formulation:
#  (a) 7-point box test filter (1 center + 6 orthogonal neighbors) instead of a
#      higher-order Gaussian. Test filter scale ratio Δ_test/Δ = 7^(1/3) ≈ 1.91.
#  (b) The m̃_αβ term in step 5 (test-filter of the grid-scale modeled stress)
#      is OMITTED — taken as ≈ 0 in the simplified Germano formulation. This is
#      equivalent to using only the test-scale modeled stress as M.
#  (c) No spatial averaging (Lilly 1992) or Lagrangian averaging (Meneveau-Lund-
#      Cabot 1996). Pure local. For VRAM-constrained GPU runs this trades
#      noise for memory/compute. Refinement: add 5³-cell averaging buffer.
#
# These simplifications are documented in published dynamic-Smagorinsky LBM
# work (e.g., Premnath, Pattison & Banerjee 2009 PRE 79, 026703 use a similar
# local-only formulation). The c_d this routine returns is a defensible Cs².
#
# The returned value REPLACES the hand-tuned `c_pi_stress` in the existing
# `variable_nu_sgs` path. Conversion to the LUDWIG c_pi parameterization:
#   The LUDWIG formula is  ν_sgs = c_pi · |Π_neq|/(ρ·cs²).
#   The Smagorinsky equivalent  ν_sgs = Cs²·Δ²·|S|  with |Π_neq| ≈ 2·ν·ρ·|S|
#   (Chapman-Enskog leading order) gives  c_pi ≈ Cs²·Δ²·cs²/(2·ν).
#   This routine returns Cs² directly; the caller (entropic kernel) converts.
# ==============================================================================

using StaticArrays

"""
    get_velocity_at(vel_in, x, y, z, b_idx, dx, dy, dz, block_size, neighbor_table, obstacle)

Read the 3-component velocity at cell offset (dx, dy, dz) from the current
cell. Handles cross-block lookups via `neighbor_table` for offsets that exit
the current block. When the neighbor is an obstacle or out of domain, returns
the center velocity (Neumann fallback — preserves the test-filter mean from
exiting the fluid domain).
"""
@inline function get_velocity_at(
    vel_in, obstacle,
    x, y, z, b_idx,
    dx_off, dy_off, dz_off,
    block_size, neighbor_table,
    ux_center, uy_center, uz_center
)
    @inbounds begin
        nx = x + dx_off
        ny = y + dy_off
        nz = z + dz_off

        nb_off_x = nx < Int32(1) ? Int32(-1) : (nx > block_size ? Int32(1) : Int32(0))
        nb_off_y = ny < Int32(1) ? Int32(-1) : (ny > block_size ? Int32(1) : Int32(0))
        nb_off_z = nz < Int32(1) ? Int32(-1) : (nz > block_size ? Int32(1) : Int32(0))

        if nb_off_x == Int32(0) && nb_off_y == Int32(0) && nb_off_z == Int32(0)
            if obstacle[nx, ny, nz, b_idx]
                return (ux_center, uy_center, uz_center)
            end
            return (vel_in[nx, ny, nz, b_idx, 1],
                    vel_in[nx, ny, nz, b_idx, 2],
                    vel_in[nx, ny, nz, b_idx, 3])
        end

        # Cross-block via neighbor_table
        dir_idx = (nb_off_x + Int32(1)) + (nb_off_y + Int32(1)) * Int32(3) +
                  (nb_off_z + Int32(1)) * Int32(9) + Int32(1)
        nb_b_idx = neighbor_table[b_idx, dir_idx]

        if nb_b_idx == Int32(0)
            return (ux_center, uy_center, uz_center)
        end

        nx_local = nx < Int32(1) ? nx + block_size :
                   nx > block_size ? nx - block_size : nx
        ny_local = ny < Int32(1) ? ny + block_size :
                   ny > block_size ? ny - block_size : ny
        nz_local = nz < Int32(1) ? nz + block_size :
                   nz > block_size ? nz - block_size : nz

        if obstacle[nx_local, ny_local, nz_local, nb_b_idx]
            return (ux_center, uy_center, uz_center)
        end
        return (vel_in[nx_local, ny_local, nz_local, nb_b_idx, 1],
                vel_in[nx_local, ny_local, nz_local, nb_b_idx, 2],
                vel_in[nx_local, ny_local, nz_local, nb_b_idx, 3])
    end
end

"""
    compute_dynamic_cs2_inline(vel_in, obstacle, neighbor_table,
                               x, y, z, b_idx, block_size,
                               ux_center, uy_center, uz_center,
                               g11, g22, g33, S12, S13, S23,
                               dx_cell)

Returns the Germano-Lilly dynamic Smagorinsky coefficient `Cs² ∈ [0, 0.25]`
for the cell at (x, y, z, b_idx). Reads 6 orthogonal-neighbor velocities,
constructs the 7-point test filter, computes Leonard and modeled stresses,
and applies the Lilly least-squares closure.

The strain rate components `(g11, g22, g33, S12, S13, S23)` are computed by
the caller (entropic kernel already has them from its WALE path).
`dx_cell` is the level's physical cell size.
"""
@inline function compute_dynamic_cs2_inline(
    vel_in, obstacle, neighbor_table,
    x, y, z, b_idx, block_size,
    ux_center::Float32, uy_center::Float32, uz_center::Float32,
    g11::Float32, g22::Float32, g33::Float32,
    S12::Float32, S13::Float32, S23::Float32,
    dx_cell::Float32
)
    @inbounds begin
        # ── STEP 1: Read 6 orthogonal-neighbor velocities ────────────────
        ux_xp, uy_xp, uz_xp = get_velocity_at(vel_in, obstacle,
            x, y, z, b_idx, Int32( 1), Int32(0), Int32(0),
            block_size, neighbor_table, ux_center, uy_center, uz_center)
        ux_xm, uy_xm, uz_xm = get_velocity_at(vel_in, obstacle,
            x, y, z, b_idx, Int32(-1), Int32(0), Int32(0),
            block_size, neighbor_table, ux_center, uy_center, uz_center)
        ux_yp, uy_yp, uz_yp = get_velocity_at(vel_in, obstacle,
            x, y, z, b_idx, Int32(0), Int32( 1), Int32(0),
            block_size, neighbor_table, ux_center, uy_center, uz_center)
        ux_ym, uy_ym, uz_ym = get_velocity_at(vel_in, obstacle,
            x, y, z, b_idx, Int32(0), Int32(-1), Int32(0),
            block_size, neighbor_table, ux_center, uy_center, uz_center)
        ux_zp, uy_zp, uz_zp = get_velocity_at(vel_in, obstacle,
            x, y, z, b_idx, Int32(0), Int32(0), Int32( 1),
            block_size, neighbor_table, ux_center, uy_center, uz_center)
        ux_zm, uy_zm, uz_zm = get_velocity_at(vel_in, obstacle,
            x, y, z, b_idx, Int32(0), Int32(0), Int32(-1),
            block_size, neighbor_table, ux_center, uy_center, uz_center)

        # ── STEP 2: Test-filtered velocity (7-point box filter) ──────────
        # u_tilde_α = (u_α(center) + Σ_6 u_α(neighbors)) / 7
        inv7 = 1.0f0 / 7.0f0
        ut_x = inv7 * (ux_center + ux_xp + ux_xm + ux_yp + ux_ym + ux_zp + ux_zm)
        ut_y = inv7 * (uy_center + uy_xp + uy_xm + uy_yp + uy_ym + uy_zp + uy_zm)
        ut_z = inv7 * (uz_center + uz_xp + uz_xm + uz_yp + uz_ym + uz_zp + uz_zm)

        # ── STEP 3: Test-filtered velocity products ((u_α·u_β)~) ─────────
        # 7-point average of the per-cell product u_α·u_β
        uxux_t = inv7 * (ux_center*ux_center + ux_xp*ux_xp + ux_xm*ux_xm +
                         ux_yp*ux_yp + ux_ym*ux_ym + ux_zp*ux_zp + ux_zm*ux_zm)
        uyuy_t = inv7 * (uy_center*uy_center + uy_xp*uy_xp + uy_xm*uy_xm +
                         uy_yp*uy_yp + uy_ym*uy_ym + uy_zp*uy_zp + uy_zm*uy_zm)
        uzuz_t = inv7 * (uz_center*uz_center + uz_xp*uz_xp + uz_xm*uz_xm +
                         uz_yp*uz_yp + uz_ym*uz_ym + uz_zp*uz_zp + uz_zm*uz_zm)
        uxuy_t = inv7 * (ux_center*uy_center + ux_xp*uy_xp + ux_xm*uy_xm +
                         ux_yp*uy_yp + ux_ym*uy_ym + ux_zp*uy_zp + ux_zm*uy_zm)
        uxuz_t = inv7 * (ux_center*uz_center + ux_xp*uz_xp + ux_xm*uz_xm +
                         ux_yp*uz_yp + ux_ym*uz_ym + ux_zp*uz_zp + ux_zm*uz_zm)
        uyuz_t = inv7 * (uy_center*uz_center + uy_xp*uz_xp + uy_xm*uz_xm +
                         uy_yp*uz_yp + uy_ym*uz_ym + uy_zp*uz_zp + uy_zm*uz_zm)

        # ── STEP 4: Leonard stress ──────────────────────────────────────
        # L_αβ = (u_α·u_β)~ − ũ_α · ũ_β  (the resolved stress between Δ and 2Δ)
        Lxx = uxux_t - ut_x * ut_x
        Lyy = uyuy_t - ut_y * ut_y
        Lzz = uzuz_t - ut_z * ut_z
        Lxy = uxuy_t - ut_x * ut_y
        Lxz = uxuz_t - ut_x * ut_z
        Lyz = uyuz_t - ut_y * ut_z

        # Deviatoric part (subtract 1/3 trace from the diagonals)
        L_trace = (Lxx + Lyy + Lzz) * (1.0f0 / 3.0f0)
        Lxx_d = Lxx - L_trace
        Lyy_d = Lyy - L_trace
        Lzz_d = Lzz - L_trace
        # Off-diagonals are already deviatoric (zero contribution from trace δ_αβ)

        # ── STEP 5: Strain-rate magnitude |S| at the CENTER (grid scale Δ) ──
        # The caller passes pre-computed S_ij components (already used by WALE).
        # |S|² = 2·S_αβ·S_αβ = 2·(g11² + g22² + g33²) + 4·(S12² + S13² + S23²)
        # (because S_xy = S_yx etc; the factor 2 accounts for symmetric off-diag)
        S_mag_sq = 2.0f0 * (g11*g11 + g22*g22 + g33*g33) +
                   4.0f0 * (S12*S12 + S13*S13 + S23*S23)
        S_mag = sqrt(max(S_mag_sq, 1.0f-30))

        # ── STEP 6: Modeled stress at TEST scale (2Δ) ──────────────────
        # M_αβ = −2·(Δ_test)²·|S̃|·S̃_αβ
        # Simplification: we approximate S̃_αβ ≈ S_αβ (grid-scale strain rate).
        # The classical Germano subtracts filter(M_grid) — here we use only the
        # test-scale piece. The (Δ_test/Δ)² = (7^(1/3))² ≈ 3.66 ratio is the
        # scale separation. M's overall sign is irrelevant for the L:M/M:M
        # ratio, so we drop the −2 prefactor.
        scale_test = 2.0f0 * dx_cell * dx_cell        # 2·Δ² piece
        # 7^(2/3) ≈ 3.659306 — the (Δ_test/Δ)² ratio for a 7-point box filter
        M_coeff = scale_test * 3.659306f0 * S_mag

        Mxx = M_coeff * g11
        Myy = M_coeff * g22
        Mzz = M_coeff * g33
        Mxy = M_coeff * S12
        Mxz = M_coeff * S13
        Myz = M_coeff * S23

        # M's deviatoric part — same idea
        M_trace = (Mxx + Myy + Mzz) * (1.0f0 / 3.0f0)
        Mxx_d = Mxx - M_trace
        Myy_d = Myy - M_trace
        Mzz_d = Mzz - M_trace

        # ── STEP 7: Lilly least-squares contraction ─────────────────────
        # c_d = ½ · ⟨L_αβ · M_αβ⟩ / ⟨M_αβ · M_αβ⟩
        # Contractions of symmetric tensors:
        #   diagonal terms 1×, off-diagonals 2× (counts both (αβ) and (βα))
        LM = Lxx_d * Mxx_d + Lyy_d * Myy_d + Lzz_d * Mzz_d +
             2.0f0 * (Lxy * Mxy + Lxz * Mxz + Lyz * Myz)
        MM = Mxx_d * Mxx_d + Myy_d * Myy_d + Mzz_d * Mzz_d +
             2.0f0 * (Mxy * Mxy + Mxz * Mxz + Myz * Myz)

        c_d = 0.5f0 * LM / (MM + 1.0f-20)

        # ── STEP 8: Clamp to [0, 0.25] (Smagorinsky stability range) ────
        # Cs² ∈ [0, 0.25] → Cs ∈ [0, 0.5]. Negative c_d would amplify
        # turbulent fluctuations instead of damping them ("anti-damping"),
        # which is physically wrong; clamp to 0. Above 0.25 risks over-
        # diffusion. The standard Smagorinsky constant is Cs² ≈ 0.026 in
        # the inertial range, so the upper limit 0.25 is generous.
        return clamp(c_d, 0.0f0, 0.25f0)
    end
end
