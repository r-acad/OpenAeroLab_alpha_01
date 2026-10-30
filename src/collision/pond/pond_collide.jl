# ==============================================================================
# Particles-on-Demand — per-cell co-moving collision   [path A, Phase 2]
# ==============================================================================
# In the local co-moving frame (u_ref ≈ u) the equilibrium is the rest Maxwellian
# f_eq_i = ρ w_i — EXACT, no Taylor/Mach truncation (P1 Eq.17). Collision relaxes the
# non-equilibrium moments toward equilibrium with SEPARATE per-order rates:
#   ω_s  shear   (off-diagonal stress + deviatoric diagonal)  → ν   = cs²(1/ω_s − 1/2)
#   ω_b  bulk    (trace of the stress)                        → ν_b = cs²(1/ω_b − 1/2)
#   ω_h  heat    (3rd-order energy/heat flux)                 → κ   (Prandtl number)
# (LELBM Noh-Lee arXiv:2508.06911, Eqs. 29a–f.)
#
# A Knudsen limiter (LELBM §II.D / App. D) is the shock safety valve: where the
# non-equilibrium stress ratio |Π_neq|/p is excessive (under-resolved shock), the
# effective ω is pushed toward 1 — relax that cell MORE toward the continuum
# equilibrium. This is graceful and moment-aware, the opposite of the wholesale 0.35×
# damping that wrecked the per-block transition. Off by default (knudsen_limit ≤ 0).
#
# Realized through the existing entropic 13-moment machinery (so the reconstructed
# populations are positive and reproduce the relaxed moments exactly).
# ==============================================================================

"""
    pond_collide(f, T, ω_s, ω_b, ω_h, cx,cy,cz,w, cs2, eos_R, max_iter;
                 knudsen_limit=0) -> (f*::MVector{39}, converged::Bool)

Per-cell co-moving collision. Conserves mass + momentum exactly; relaxes shear/bulk
stress and heat flux at ω_s/ω_b/ω_h toward the (density-only) equilibrium.
"""
@inline function pond_collide(
    f::MVector{39, Float32}, T::Float32,
    omega_s::Float32, omega_b::Float32, omega_h::Float32,
    cx_arr, cy_arr, cz_arr, w_arr,
    cs2::Float32, eos_R::Float32, max_iter::Int32,
    knudsen_limit::Float32 = 0.0f0
)
    # Moments in the co-moving frame (peculiar velocity = c_i ⇒ shift 0).
    M = compute_13_moments_of_populations_d3q39(
        f, cx_arr, cy_arr, cz_arr, cs2, 0.0f0, 0.0f0, 0.0f0)
    rho = max(M[1], 1.0f-6)
    inv = 1.0f0 / rho
    ux = M[2] * inv; uy = M[3] * inv; uz = M[4] * inv   # residual peculiar velocity

    # Equilibrium moments at the actual (ρ, peculiar u, T): the convective part of the
    # stress (ρ u u) is equilibrium and must NOT be relaxed; only the true non-eq is.
    Meq = compute_target_moments_d3q39(rho, ux, uy, uz, T, cs2, eos_R)

    # Non-equilibrium moments (stress 5..10, heat flux 11..13).
    n5  = M[5]  - Meq[5];  n6  = M[6]  - Meq[6];  n7  = M[7]  - Meq[7]
    n8  = M[8]  - Meq[8];  n9  = M[9]  - Meq[9];  n10 = M[10] - Meq[10]
    n11 = M[11] - Meq[11]; n12 = M[12] - Meq[12]; n13 = M[13] - Meq[13]

    # Knudsen limiter: push ω→1 where the non-eq stress is large vs the pressure p=ρ·R·T.
    os = omega_s; ob = omega_b; oh = omega_h
    if knudsen_limit > 0.0f0
        p_ref = max(rho * eos_R * T, 1.0f-6)
        neq_mag = sqrt(n5*n5 + n6*n6 + n7*n7 +
                       2.0f0*(n8*n8 + n9*n9 + n10*n10)) / p_ref
        # smoothstep from 0 (at knudsen_limit) to 1 (at 2·knudsen_limit)
        x = clamp((neq_mag - knudsen_limit) / knudsen_limit, 0.0f0, 1.0f0)
        boost = x * x * (3.0f0 - 2.0f0 * x)
        os = os + (1.0f0 - os) * boost
        ob = ob + (1.0f0 - ob) * boost
        oh = oh + (1.0f0 - oh) * boost
    end

    # Diagonal stress: split into bulk (trace, ω_b) and deviatoric (ω_s).
    tr   = n5 + n6 + n7
    trd3 = tr * Float32(1/3)
    d5 = n5 - trd3; d6 = n6 - trd3; d7 = n7 - trd3
    tr_relaxed = (1.0f0 - ob) * tr
    n5s = tr_relaxed * Float32(1/3) + (1.0f0 - os) * d5
    n6s = tr_relaxed * Float32(1/3) + (1.0f0 - os) * d6
    n7s = tr_relaxed * Float32(1/3) + (1.0f0 - os) * d7
    # Off-diagonal stress (shear) and heat flux.
    n8s  = (1.0f0 - os) * n8;  n9s  = (1.0f0 - os) * n9;  n10s = (1.0f0 - os) * n10
    n11s = (1.0f0 - oh) * n11; n12s = (1.0f0 - oh) * n12; n13s = (1.0f0 - oh) * n13

    # Post-collision target moments: equilibrium + relaxed non-eq; mass+momentum conserved.
    mt1 = M[1]; mt2 = M[2]; mt3 = M[3]; mt4 = M[4]
    mt5  = Meq[5]  + n5s;  mt6  = Meq[6]  + n6s;  mt7  = Meq[7]  + n7s
    mt8  = Meq[8]  + n8s;  mt9  = Meq[9]  + n9s;  mt10 = Meq[10] + n10s
    mt11 = Meq[11] + n11s; mt12 = Meq[12] + n12s; mt13 = Meq[13] + n13s

    lam = solve_13mom_lambdas_explicit_d3q39(
        rho, mt1, mt2, mt3, mt4, mt5, mt6, mt7, mt8, mt9, mt10, mt11, mt12, mt13,
        cx_arr, cy_arr, cz_arr, w_arr, cs2, max_iter, 0.0f0, 0.0f0, 0.0f0, 1.0f0)
    converged = lam[14] > 0.5f0

    fstar = MVector{39, Float32}(undef)
    @inbounds for k in Int32(1):Int32(39)
        fstar[k] = eval_13mom_feq_single(
            rho, lam, Float32(cx_arr[k]), Float32(cy_arr[k]), Float32(cz_arr[k]),
            Float32(w_arr[k]), cs2)
    end
    return (fstar, converged)
end
