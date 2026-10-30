# ==============================================================================
# RECURSIVE REGULARIZATION (RR-LBM) — Hermite projection through order 3
# ==============================================================================
# Pure-projection regularization per:
#   Latt & Chopard 2006, Math. Comput. Simul. 72 (regularized LBM, original)
#   Coreixas 2017, PRE 96, 033306 (recursive-regularized LBM)
#   Coreixas & Latt 2019, Comput. Math. Appl. 79 (stability of Hermite-based LBM)
#   Schukmann & Hass 2025, Fluids 10 (HRR-LBM with WALE for aeroacoustics)
#
# The D3Q39 lattice has 39 degrees of freedom and exactly supports Hermite
# moments through order N=4 in 3D. The physical (resolvable) subspace has
# dimension 1 + 3 + 6 + 10 + 15 = 35 (mass, momentum, stress, heat-flux,
# hyperflux). The remaining 4-dimensional ghost subspace contains
# non-hydrodynamic eigenmodes including the (-1)^(y+z) "checkerboard" mode
# that escaped all per-direction stabilizers in cases 001-006.
#
# This routine applies the orthogonal projection f -> f^reg defined by:
#
#   f^reg_k = f_eq_k + w_k * Σ_{n=2..N_max} (1/(n!·cs^(2n))) · H^(n)(c_k) : a^(n),neq
#
# where a^(n),neq = Σ_k (f_k - f_eq_k) H^(n)(c_k) are the projection of the
# non-equilibrium part onto the n-th order Hermite tensor.
#
# By construction:
#   * Mass and momentum are exactly conserved (H^(n) is orthogonal to H^(0,1)
#     under the lattice weight measure).
#   * Stress (a^(2)) and heat-flux (a^(3)) are preserved exactly when the
#     pre-projection populations carry only physical content.
#   * Any non-hydrodynamic mode (≥ order 4 + ghost) is annihilated.
#
# Implementation: N_max = 4 (orders 2 + 3 + 4 — SOTA roadmap A5 closed 2026-05-27).
# This completes the projection onto the full 35-dimensional physical Hermite
# subspace that D3Q39 can resolve:
#    1 (mass) + 3 (momentum) + 6 (stress) + 10 (heat-flux Q) + 15 (hyperflux R) = 35
# The remaining 4 dimensions out of D3Q39's 39 are the **ghost subspace** —
# population patterns with zero macroscopic moments at all representable orders.
# By construction, projecting onto the 35-D physical subspace and reconstructing
# **annihilates all 4 ghost modes exactly**. The earlier order-3-only version
# left the R-tensor moments (15 dimensions of physical content) UN-regularized,
# meaning hyperflux-mode aliased noise could re-emerge between successive
# regularization applications. Order 4 closes that gap.
#
# Cost per cell, full order-4 implementation:
#   - Order 2: 6 moments + 6 reconstruction terms × 39 dirs ≈ 470 ops
#   - Order 3: 10 moments + 10 reconstruction terms × 39 dirs ≈ 780 ops
#   - Order 4: 15 moments + 15 reconstruction terms × 39 dirs ≈ 1170 ops
#   - Total ≈ 2420 ops per cell per call. About 25-40 % overhead on the
#     entropic collision step. Acceptable per the user's "irrespective of cost"
#     directive (high-impact paper quality).
# ==============================================================================

using StaticArrays

"""
    regularize_d3q39_inline!(f_stored, f_eq, cx_arr, cy_arr, cz_arr, w_arr, n_d, cs2_val)

Project f_stored onto the order-3 Hermite subspace and reconstruct in-place.

`f_stored` and `f_eq` are local MVector{39, Float32} living on the GPU
register stack (or shared memory). `cx_arr`, `cy_arr`, `cz_arr`, `w_arr`
are the lattice velocity and weight tables (Galilean-SHIFTED if a shift is
active). `cs2_val` is the lattice speed of sound squared (2/3 for D3Q39).

`shift_x/y/z` are the Galilean shift components (0 for the unshifted lattice).

AUDIT FIX (2026-06-03): The Hermite projection's mass/momentum conservation
guarantee — `Σ_k w_k H^(n)(c_k) = 0` and `Σ_k w_k c_k H^(n≥1)(c_k) = 0` —
holds ONLY for the velocity set the weights were derived for, where
`Σ w_k c_k = 0`. With a Galilean shift the stored `cx_arr` is `c' = c + U`,
so `Σ w_k c'_k = U ≠ 0` and the H^(n)(c') are no longer orthogonal Hermite
polynomials of the weight → the projection injects mass/momentum. Previously
this routine used the shifted `cx_arr[k]` directly, breaking conservation in
every supersonic shifted run (the SVV add-on already un-shifts correctly; RR
was the lone outlier). We now form all Hermite tensors in the LAB frame
`c_lab = c' − U`, matching `f_eq` (which is centered on `u_rel = u − U` in the
same frame). For shift=0 this is byte-identical to the previous behavior.

The reconstruction discards population content outside the Hermite-N
subspace, which is what kills the Y-Z checkerboard ghost eigenmode.
"""
@inline function regularize_d3q39_inline!(
    f_stored::MVector{39, Float32},
    f_eq::MVector{39, Float32},
    cx_arr, cy_arr, cz_arr, w_arr,
    n_d::Int32,
    cs2_val::Float32,
    shift_x::Float32=0.0f0, shift_y::Float32=0.0f0, shift_z::Float32=0.0f0
)
    # ── STEP 1: Compute non-equilibrium moments a^(2) and a^(3) ──────────
    #
    # a^(2)_αβ = Σ_k (f_k − f_eq_k) · H^(2)_αβ(c_k)
    #          = Σ_k f_neq_k · (c_α c_β − cs² δ_αβ)
    #
    # a^(3)_αβγ = Σ_k (f_k − f_eq_k) · H^(3)_αβγ(c_k)
    #
    # All tensors are fully symmetric; store only unique components.
    @inbounds begin
        # Pre-compute cs² powers (used in Hermite tensor formulas)
        cs4 = cs2_val * cs2_val
        three_cs2 = 3.0f0 * cs2_val
        six_cs2 = 6.0f0 * cs2_val
        three_cs4 = 3.0f0 * cs4

        # ── Order 2 (6 components, symmetric) ──
        a2_xx = 0.0f0; a2_yy = 0.0f0; a2_zz = 0.0f0
        a2_xy = 0.0f0; a2_xz = 0.0f0; a2_yz = 0.0f0

        # ── Order 3 (10 components, fully symmetric) ──
        a3_xxx = 0.0f0; a3_yyy = 0.0f0; a3_zzz = 0.0f0
        a3_xxy = 0.0f0; a3_xyy = 0.0f0
        a3_xxz = 0.0f0; a3_xzz = 0.0f0
        a3_yyz = 0.0f0; a3_yzz = 0.0f0
        a3_xyz = 0.0f0

        # ── Order 4 (15 components, fully symmetric R-tensor) ──
        a4_xxxx = 0.0f0; a4_yyyy = 0.0f0; a4_zzzz = 0.0f0      # aaaa (3)
        a4_xxxy = 0.0f0; a4_xxxz = 0.0f0                       # aaab (6)
        a4_xyyy = 0.0f0; a4_yyyz = 0.0f0
        a4_xzzz = 0.0f0; a4_yzzz = 0.0f0
        a4_xxyy = 0.0f0; a4_xxzz = 0.0f0; a4_yyzz = 0.0f0      # aabb (3)
        a4_xxyz = 0.0f0; a4_xyyz = 0.0f0; a4_xyzz = 0.0f0      # aabc (3)

        # ── STEP 1: Compute non-equilibrium moments a^(2), a^(3), a^(4) ──
        # AUDIT FIX (2026-06-03): use LAB-frame velocities c_lab = c' − U so the
        # Hermite tensors are orthogonal to mass/momentum under the weight (see
        # docstring). For shift=0 this is identical to the previous code.
        for k in Int32(1):n_d
            fk = f_stored[k] - f_eq[k]
            cx_k = Float32(cx_arr[k]) - shift_x
            cy_k = Float32(cy_arr[k]) - shift_y
            cz_k = Float32(cz_arr[k]) - shift_z

            # Powers and pairs cached once per k
            cxcx = cx_k * cx_k
            cycy = cy_k * cy_k
            czcz = cz_k * cz_k
            cxcy = cx_k * cy_k
            cxcz = cx_k * cz_k
            cycz = cy_k * cz_k

            # ── Order 2 (H^(2)_αβ = c_α c_β − cs² δ_αβ) ──
            a2_xx += fk * (cxcx - cs2_val)
            a2_yy += fk * (cycy - cs2_val)
            a2_zz += fk * (czcz - cs2_val)
            a2_xy += fk * cxcy
            a2_xz += fk * cxcz
            a2_yz += fk * cycz

            # ── Order 3 (H^(3)_αβγ = c_α c_β c_γ − cs² (c_α δ_βγ + c_β δ_γα + c_γ δ_αβ)) ──
            # aaa diagonal:    H^(3)_xxx = c_x^3 − 3 cs² c_x
            # aab "near-diag": H^(3)_xxy = c_x^2 c_y − cs² c_y
            # abc all-distinct: H^(3)_xyz = c_x c_y c_z
            a3_xxx += fk * (cxcx * cx_k - three_cs2 * cx_k)
            a3_yyy += fk * (cycy * cy_k - three_cs2 * cy_k)
            a3_zzz += fk * (czcz * cz_k - three_cs2 * cz_k)
            a3_xxy += fk * (cxcx * cy_k - cs2_val * cy_k)
            a3_xyy += fk * (cycy * cx_k - cs2_val * cx_k)
            a3_xxz += fk * (cxcx * cz_k - cs2_val * cz_k)
            a3_xzz += fk * (czcz * cx_k - cs2_val * cx_k)
            a3_yyz += fk * (cycy * cz_k - cs2_val * cz_k)
            a3_yzz += fk * (czcz * cy_k - cs2_val * cy_k)
            a3_xyz += fk * (cxcy * cz_k)

            # ── Order 4 (H^(4)_αβγδ — R-tensor, 15 unique components) ──
            # General formula:
            #   H^(4)_αβγδ = c_α c_β c_γ c_δ
            #              − cs²·(6 permutations of c_α c_β δ_γδ)
            #              + cs⁴·(3 permutations of δ_αβ δ_γδ)
            #
            # Worked out per-pattern below (only non-zero δ-products survive):
            #   aaaa (e.g. xxxx): H^(4) = c^4 − 6cs²·c² + 3cs⁴
            #   aaab (e.g. xxxy): H^(4) = c_a^3·c_b − 3cs²·c_a·c_b
            #   aabb (e.g. xxyy): H^(4) = c_a²·c_b² − cs²·(c_a² + c_b²) + cs⁴
            #   aabc (e.g. xxyz): H^(4) = c_a²·c_b·c_c − cs²·c_b·c_c
            cx4 = cxcx * cxcx
            cy4 = cycy * cycy
            cz4 = czcz * czcz
            a4_xxxx += fk * (cx4 - six_cs2 * cxcx + three_cs4)
            a4_yyyy += fk * (cy4 - six_cs2 * cycy + three_cs4)
            a4_zzzz += fk * (cz4 - six_cs2 * czcz + three_cs4)

            # aaab (6 components): H^(4)_aaab = c_a^3·c_b − 3·cs²·c_a·c_b
            a4_xxxy += fk * (cxcx * cxcy - three_cs2 * cxcy)
            a4_xxxz += fk * (cxcx * cxcz - three_cs2 * cxcz)
            a4_xyyy += fk * (cycy * cxcy - three_cs2 * cxcy)
            a4_yyyz += fk * (cycy * cycz - three_cs2 * cycz)
            a4_xzzz += fk * (czcz * cxcz - three_cs2 * cxcz)
            a4_yzzz += fk * (czcz * cycz - three_cs2 * cycz)

            # aabb (3 components): H^(4)_aabb = c_a²·c_b² − cs²(c_a² + c_b²) + cs⁴
            a4_xxyy += fk * (cxcx * cycy - cs2_val * (cxcx + cycy) + cs4)
            a4_xxzz += fk * (cxcx * czcz - cs2_val * (cxcx + czcz) + cs4)
            a4_yyzz += fk * (cycy * czcz - cs2_val * (cycy + czcz) + cs4)

            # aabc (3 components): H^(4)_aabc = c_a²·c_b·c_c − cs²·c_b·c_c
            a4_xxyz += fk * (cxcx * cycz - cs2_val * cycz)
            a4_xyyz += fk * (cycy * cxcz - cs2_val * cxcz)
            a4_xyzz += fk * (czcz * cxcy - cs2_val * cxcy)
        end

        # ── STEP 2: Reconstruct populations from the projected moments ────
        #
        # f^reg_k = f_eq_k + w_k · [ (1/(2 cs^4)) H^(2):a^(2)
        #                          + (1/(6 cs^6)) H^(3):a^(3)
        #                          + (1/(24 cs^8)) H^(4):a^(4) ]
        #
        # For fully-symmetric tensors the contraction has multiplicities:
        #   H^(2): a_αβ           : 1 (aa), 2 (ab)
        #   H^(3): a_αβγ          : 1 (aaa), 3 (aab), 6 (abc)
        #   H^(4): a_αβγδ         : 1 (aaaa), 4 (aaab), 6 (aabb), 12 (aabc)
        inv_2cs4  = 0.5f0 / cs4
        inv_6cs6  = 1.0f0 / (6.0f0 * cs4 * cs2_val)
        inv_24cs8 = 1.0f0 / (24.0f0 * cs4 * cs4)

        for k in Int32(1):n_d
            # AUDIT FIX (2026-06-03): LAB-frame velocities (see STEP 1). The
            # reconstruction H^(n) basis must match the one used to form a^(n).
            cx_k = Float32(cx_arr[k]) - shift_x
            cy_k = Float32(cy_arr[k]) - shift_y
            cz_k = Float32(cz_arr[k]) - shift_z
            w_k  = Float32(w_arr[k])

            cxcx = cx_k * cx_k
            cycy = cy_k * cy_k
            czcz = cz_k * cz_k
            cxcy = cx_k * cy_k
            cxcz = cx_k * cz_k
            cycz = cy_k * cz_k
            cx4 = cxcx * cxcx
            cy4 = cycy * cycy
            cz4 = czcz * czcz

            # H^(2):a^(2) contraction
            H2a2 = a2_xx * (cxcx - cs2_val) +
                   a2_yy * (cycy - cs2_val) +
                   a2_zz * (czcz - cs2_val) +
                   2.0f0 * (a2_xy * cxcy + a2_xz * cxcz + a2_yz * cycz)

            # H^(3):a^(3) contraction (with symmetric-tensor multiplicities)
            H3a3 = a3_xxx * (cxcx * cx_k - three_cs2 * cx_k) +
                   a3_yyy * (cycy * cy_k - three_cs2 * cy_k) +
                   a3_zzz * (czcz * cz_k - three_cs2 * cz_k) +
                   3.0f0 * (
                       a3_xxy * (cxcx * cy_k - cs2_val * cy_k) +
                       a3_xyy * (cycy * cx_k - cs2_val * cx_k) +
                       a3_xxz * (cxcx * cz_k - cs2_val * cz_k) +
                       a3_xzz * (czcz * cx_k - cs2_val * cx_k) +
                       a3_yyz * (cycy * cz_k - cs2_val * cz_k) +
                       a3_yzz * (czcz * cy_k - cs2_val * cy_k)
                   ) +
                   6.0f0 * a3_xyz * (cxcy * cz_k)

            # H^(4):a^(4) contraction (R-tensor, with symmetric-tensor multiplicities)
            # aaaa: mult 1; aaab: mult 4; aabb: mult 6; aabc: mult 12
            H4_xxxx = cx4 - six_cs2 * cxcx + three_cs4
            H4_yyyy = cy4 - six_cs2 * cycy + three_cs4
            H4_zzzz = cz4 - six_cs2 * czcz + three_cs4

            H4_xxxy = cxcx * cxcy - three_cs2 * cxcy
            H4_xxxz = cxcx * cxcz - three_cs2 * cxcz
            H4_xyyy = cycy * cxcy - three_cs2 * cxcy
            H4_yyyz = cycy * cycz - three_cs2 * cycz
            H4_xzzz = czcz * cxcz - three_cs2 * cxcz
            H4_yzzz = czcz * cycz - three_cs2 * cycz

            H4_xxyy = cxcx * cycy - cs2_val * (cxcx + cycy) + cs4
            H4_xxzz = cxcx * czcz - cs2_val * (cxcx + czcz) + cs4
            H4_yyzz = cycy * czcz - cs2_val * (cycy + czcz) + cs4

            H4_xxyz = cxcx * cycz - cs2_val * cycz
            H4_xyyz = cycy * cxcz - cs2_val * cxcz
            H4_xyzz = czcz * cxcy - cs2_val * cxcy

            H4a4 = (a4_xxxx * H4_xxxx + a4_yyyy * H4_yyyy + a4_zzzz * H4_zzzz) +
                   4.0f0 * (a4_xxxy * H4_xxxy + a4_xxxz * H4_xxxz +
                            a4_xyyy * H4_xyyy + a4_yyyz * H4_yyyz +
                            a4_xzzz * H4_xzzz + a4_yzzz * H4_yzzz) +
                   6.0f0 * (a4_xxyy * H4_xxyy + a4_xxzz * H4_xxzz + a4_yyzz * H4_yyzz) +
                   12.0f0 * (a4_xxyz * H4_xxyz + a4_xyyz * H4_xyyz + a4_xyzz * H4_xyzz)

            f_stored[k] = f_eq[k] +
                          w_k * (H2a2 * inv_2cs4 + H3a3 * inv_6cs6 + H4a4 * inv_24cs8)
        end
    end
    return nothing
end
