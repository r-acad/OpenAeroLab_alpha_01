using KernelAbstractions
using StaticArrays

# ==============================================================================
# ENTROPIC LBM (KBC) COLLISION OPERATOR — D3Q39
# ==============================================================================
# Implements the KBC (Karlin-Bösch-Chikatamarla) entropic multi-relaxation-time
# model on the D3Q39 Variant B lattice (Frapolli/Chikatamarla/Karlin, 2015).
#
# This is the D3Q39 counterpart to kernel_entropic.jl (D3Q27). Key differences:
#   - 39 directions instead of 27
#   - cs² = 2/3 instead of 1/3
#   - 13-moment Newton-Raphson numerical equilibrium instead of polynomial
#   - Max |c| = 3 (vs 1 for D3Q27) — streaming reaches 3 cells
#
# The mirror decomposition, α solvers, and KBC collision formula are identical
# in structure to the D3Q27 version, just operating over 39 directions.
#
# Pull-scheme streaming handles |c| ≤ 3 transparently because:
#   - Within a block: sx = x - cx is always within [-3, block_size+3]
#   - BLOCK_SIZE = 8 >> 3, so multi-hop reaches at most the nearest neighbor block
#   - The neighbor_table already encodes block offsets for {-1,0,+1}³
#   - For |c|=2 or |c|=3 along one axis, the offset is still within ±1 blocks
#
# IMPORTANT: The neighbor table indexing uses (off_x+1) + (off_y+1)*3 + (off_z+1)*9
# where off ∈ {-1,0,+1}. For |c|≤3 with BLOCK_SIZE=8:
#   c=2: sx = x-2 can be < 1 at x=1,2 → off = -1 ✓
#   c=3: sx = x-3 can be < 1 at x=1,2,3 → off = -1 ✓
#   c=-3: sx = x+3 can be > 8 at x=6,7,8 → off = +1 ✓
#
# The wrap-around: nsx = sx + block_size (or sx - block_size) handles the exact
# local coordinate within the neighbor block.
# ==============================================================================

# ── Population floor + GPU-safe log ──────────────────────────────────────────
# RESTORED 2026-06-25: these were previously defined in kernel_entropic.jl (the
# D3Q27 entropic kernel), which was DELETED in the 2026-06-24 cleanup that retired
# the D3Q19/D3Q27-entropic paths. The D3Q39 supersonic kernel still references them
# in ~30 places, so their removal broke the entire D3Q39 path with a GPU-JIT
# InvalidIRError ("use of an undefined name: F_FLOOR / safe_log") at the first
# timestep. Re-defined here so the D3Q39 entropic kernel is self-contained.
#   F_FLOOR  — positive floor on populations (KBC entropy needs f_k > 0 for log).
#   safe_log — log guarded against non-positive argument (returns a large negative
#              sentinel instead of NaN/Inf, keeping the H-functional finite on GPU).
const F_FLOOR = Float32(1e-15)

@inline function safe_log(x::Float32)
    return x > 0.0f0 ? log(x) : -88.0f0   # log(1e-38) ≈ -87.3; -88 is a finite floor
end

# ──────────────────────────────────────────────────────────────────────
# D3Q39 entropy and α solvers — identical algorithm, 39 directions
# ──────────────────────────────────────────────────────────────────────

@inline function entropy_H_39(f::MVector{39, Float32}, w_arr, n::Int32)
    H = 0.0f0
    for k in Int32(1):n
        fk = max(f[k], F_FLOOR)
        H += fk * safe_log(fk / w_arr[k])
    end
    return H
end

@inline function entropy_at_alpha_39(
    f_eq::MVector{39, Float32},
    s::MVector{39, Float32},
    h::MVector{39, Float32},
    w_arr,
    beta::Float32,
    alpha::Float32,
    n::Int32
)
    coeff_s = 1.0f0 - 2.0f0 * beta
    coeff_h = 1.0f0 - 2.0f0 * beta * alpha
    H = 0.0f0
    for k in Int32(1):n
        fk = f_eq[k] + coeff_s * s[k] + coeff_h * h[k]
        fk = max(fk, F_FLOOR)
        H += fk * safe_log(fk / w_arr[k])
    end
    return H
end

# ── FUSED s/h fast path (2026-05-30 register-pressure reduction) ──────────
# On the fast path (HRR off AND variable_σ off), the mirror-decomposition
# stress (s) and higher-order (h) components are PURE functions of the
# non-equilibrium Δf = f_stored − f_eq and its mirror:
#   s[k] = ½(Δf_k + Δf_k̄),  h[k] = ½(Δf_k − Δf_k̄),   k̄ = opp(k)
# So they need NOT be materialized into two MVector{39} arrays (78 floats of
# register pressure). These fused siblings recompute s_k/h_k per-k from
# f_stored/f_eq (both already live) — pure arithmetic, NO exp (unlike a lazy
# f_eq would need). They reproduce entropy_at_alpha_39 / its derivative /
# h_norm EXACTLY for the no-HRR-no-σ case. The general path (HRR or σ on)
# still materializes s_arr/h_arr because those modify s/h post-decomposition.
# h_scale folds the variable_σ damping (h_arr[k] *= h_scale) into the fused
# recompute so the fast path also covers the variable_sigma-on case. h_scale=1
# is the no-damping default. (HRR is NOT foldable — it adds an s_fd term from
# Π_neq — so the fused path is gated on !HRR_HYBRID_VAL only.)
@inline function _mirror_sh_inline(f_stored, f_eq, opp_arr, k::Int32, h_scale::Float32)
    kb = opp_arr[k]
    df_k  = f_stored[k]  - f_eq[k]
    df_kb = f_stored[kb] - f_eq[kb]
    s_k = 0.5f0 * (df_k + df_kb)
    # FP grouping MUST match the non-fused path EXACTLY: there h is built as
    # h = 0.5*(df_k-df_kb) and THEN damped h *= h_scale, i.e. (0.5*(..))*h_scale.
    # Writing it as h_scale*0.5*(..) regroups the multiply and changes the last
    # ULP — which, with strong σ-damping (h_scale~0.03 → tiny h_norm_sq near the
    # α-solver's 1e-20 activation threshold), can FLIP the branch and diverge.
    h_k = (0.5f0 * (df_k - df_kb)) * h_scale
    return s_k, h_k
end

@inline function entropy_at_alpha_39_fused(
    f_stored::MVector{39, Float32},
    f_eq::MVector{39, Float32},
    opp_arr,
    w_arr,
    beta::Float32,
    alpha::Float32,
    h_scale::Float32,
    n::Int32
)
    coeff_s = 1.0f0 - 2.0f0 * beta
    coeff_h = 1.0f0 - 2.0f0 * beta * alpha
    H = 0.0f0
    for k in Int32(1):n
        s_k, h_k = _mirror_sh_inline(f_stored, f_eq, opp_arr, k, h_scale)
        fk = f_eq[k] + coeff_s * s_k + coeff_h * h_k
        fk = max(fk, F_FLOOR)
        H += fk * safe_log(fk / w_arr[k])
    end
    return H
end

@inline function transonic_algebraic_temperature_d3q39(
    rho::Float32,
    ux::Float32,
    uy::Float32,
    uz::Float32,
    u_inlet::Float32,
    cs2_val::Float32,
    T_inlet::Float32,
    eos_R_flow::Float32,
    thermal_coupling::Int32,
    sponge::Float32
)::Float32
    # Thermal DDF is not yet block-frame aware for local_lattice_shift. When
    # eos_thermodynamic is enabled without DDF feedback, give the D3Q39
    # equilibrium a local pressure response instead of forcing transonic
    # pockets to survive through density collapse and hard velocity guards.
    if thermal_coupling != Int32(0) || eos_R_flow <= 0.0f0 || u_inlet <= 0.0f0
        return max(T_inlet, 0.1f0)
    end
    if !(rho == rho) || !(ux == ux) || !(uy == uy) || !(uz == uz)
        return max(T_inlet, 0.1f0)
    end

    gamma = 1.4f0
    gm1_half = 0.5f0 * (gamma - 1.0f0)
    T_ref = max(T_inlet, 0.1f0)
    a2_ref = max(cs2_val * T_ref, 1.0f-6)
    ma_inf_sq = (u_inlet * u_inlet) / a2_ref
    u_sq = ux*ux + uy*uy + uz*uz
    ma_loc_sq = min(u_sq / a2_ref, 6.25f0)
    ma_loc = sqrt(max(ma_loc_sq, 0.0f0))

    T0_ratio = 1.0f0 + gm1_half * ma_inf_sq
    T_iso = T_ref * T0_ratio / max(1.0f0 + gm1_half * ma_loc_sq, 1.0f-6)
    T_iso = clamp(T_iso, 0.55f0 * T_ref, 1.45f0 * T_ref)

    p_rel_iso = exp((gamma / (gamma - 1.0f0)) *
                    log(max(T_iso / T_ref, 1.0f-6)))
    rho_safe = max(rho, 0.05f0)
    T_pressure = T_ref * p_rel_iso / rho_safe
    T_pressure = clamp(T_pressure, 0.55f0 * T_ref, 1.75f0 * T_ref)

    t_mach = clamp((ma_loc - 0.35f0) / 0.35f0, 0.0f0, 1.0f0)
    mach_w = t_mach * t_mach * (3.0f0 - 2.0f0 * t_mach)
    t_rho_lo = clamp((0.86f0 - rho) / 0.34f0, 0.0f0, 1.0f0)
    rho_lo_w = t_rho_lo * t_rho_lo * (3.0f0 - 2.0f0 * t_rho_lo)
    t_rho_hi = clamp((rho - 1.28f0) / 0.72f0, 0.0f0, 1.0f0)
    rho_hi_w = t_rho_hi * t_rho_hi * (3.0f0 - 2.0f0 * t_rho_hi)
    rho_w = max(rho_lo_w, rho_hi_w)
    closure_relax = min(0.86f0, 0.18f0 + 0.56f0 * mach_w + 0.24f0 * rho_w)

    T_eff = T_ref + closure_relax * (T_pressure - T_ref)
    sp = clamp(sponge, 0.0f0, 1.0f0)
    T_eff = T_eff * (1.0f0 - sp) + T_ref * sp
    return clamp(T_eff, 0.50f0 * T_ref, 1.80f0 * T_ref)
end

@inline function entropy_and_deriv_at_alpha_39_fused(
    f_stored::MVector{39, Float32},
    f_eq::MVector{39, Float32},
    opp_arr,
    w_arr,
    beta::Float32,
    alpha::Float32,
    h_scale::Float32,
    n::Int32
)
    coeff_s = 1.0f0 - 2.0f0 * beta
    coeff_h = 1.0f0 - 2.0f0 * beta * alpha
    H  = 0.0f0
    dH = 0.0f0
    for k in Int32(1):n
        s_k, h_k = _mirror_sh_inline(f_stored, f_eq, opp_arr, k, h_scale)
        fk = f_eq[k] + coeff_s * s_k + coeff_h * h_k
        fk = max(fk, F_FLOOR)
        log_ratio = safe_log(fk / w_arr[k])
        H  += fk * log_ratio
        dH += (-2.0f0 * beta * h_k) * (1.0f0 + log_ratio)
    end
    return H, dH
end

@inline function entropy_and_deriv_at_alpha_39(
    f_eq::MVector{39, Float32},
    s::MVector{39, Float32},
    h::MVector{39, Float32},
    w_arr,
    beta::Float32,
    alpha::Float32,
    n::Int32
)
    coeff_s = 1.0f0 - 2.0f0 * beta
    coeff_h = 1.0f0 - 2.0f0 * beta * alpha
    H  = 0.0f0
    dH = 0.0f0
    for k in Int32(1):n
        fk = f_eq[k] + coeff_s * s[k] + coeff_h * h[k]
        fk = max(fk, F_FLOOR)
        log_ratio = safe_log(fk / w_arr[k])
        H  += fk * log_ratio
        dH += (-2.0f0 * beta * h[k]) * (1.0f0 + log_ratio)
    end
    return H, dH
end

@inline function solve_alpha_bisection_39(
    f_eq::MVector{39, Float32},
    s::MVector{39, Float32},
    h::MVector{39, Float32},
    w_arr,
    beta::Float32,
    H_pre::Float32,
    max_iter::Int32,
    n::Int32
)
    inv_2beta = 0.5f0 / max(beta, 0.01f0)
    alpha_lo = max(inv_2beta - 0.1f0, 0.1f0)
    alpha_hi = max(inv_2beta + 3.0f0, 4.0f0)

    H_lo = entropy_at_alpha_39(f_eq, s, h, w_arr, beta, alpha_lo, n)
    H_hi = entropy_at_alpha_39(f_eq, s, h, w_arr, beta, alpha_hi, n)

    if !((H_lo - H_pre) * (H_hi - H_pre) < 0.0f0)
        return inv_2beta
    end

    for _ in Int32(1):max_iter
        alpha_mid = 0.5f0 * (alpha_lo + alpha_hi)
        H_mid = entropy_at_alpha_39(f_eq, s, h, w_arr, beta, alpha_mid, n)
        if (H_mid - H_pre) * (H_lo - H_pre) > 0.0f0
            alpha_lo = alpha_mid
            H_lo = H_mid
        else
            alpha_hi = alpha_mid
            H_hi = H_mid
        end
        # Early exit: bracket converged.
        # 2026-05-28 (Tier 1 Item 5): threshold loosened 0.001→0.005 (5× looser).
        # Alpha is O(1.5-3); a 0.005 bracket = ~0.2% relative error in alpha,
        # well within the H-theorem stabilizer's intrinsic precision floor. The
        # bisection solver is chosen for GPU-safety (no divergence), not max
        # precision — looser bound is consistent with its design philosophy.
        # Average iter count drops ~10→~7 on smooth cells: ~30% bisection
        # speedup, dominates L1 (freestream).
        if (alpha_hi - alpha_lo) < 0.005f0 * (inv_2beta + 1.0f-6)
            break
        end
    end
    return 0.5f0 * (alpha_lo + alpha_hi)
end

@inline function solve_alpha_newton_39(
    f_eq::MVector{39, Float32},
    s::MVector{39, Float32},
    h::MVector{39, Float32},
    w_arr,
    beta::Float32,
    H_pre::Float32,
    max_iter::Int32,
    tol::Float32,
    n::Int32
)
    alpha = 2.0f0
    inv_2beta = 0.5f0 / max(beta, 0.01f0)

    for iter in Int32(1):max_iter
        H_val, dH = entropy_and_deriv_at_alpha_39(f_eq, s, h, w_arr, beta, alpha, n)
        residual = H_val - H_pre
        if abs(residual) < tol
            break
        end
        if abs(dH) < 1.0f-12
            break
        end
        delta = residual / dH
        alpha -= delta
        alpha = clamp(alpha, max(inv_2beta - 0.1f0, 0.1f0), 6.0f0)
        # Early exit: step size negligible
        if abs(delta) < 1.0f-5
            break
        end
    end
    if !(alpha > 0.0f0)
        alpha = inv_2beta
    end
    return alpha
end

@inline function solve_alpha_approximate_39(
    f_eq::MVector{39, Float32},
    s::MVector{39, Float32},
    h::MVector{39, Float32},
    w_arr,
    beta::Float32,
    n::Int32
)
    h_norm_sq = 0.0f0
    f_norm_sq = 0.0f0
    for k in Int32(1):n
        h_norm_sq += h[k] * h[k]
        f_norm_sq += f_eq[k] * f_eq[k]
    end
    ratio = h_norm_sq / max(f_norm_sq, 1.0f-15)
    inv_2beta = 0.5f0 / max(beta, 0.01f0)
    alpha = 2.0f0 - 0.5f0 * ratio
    return clamp(alpha, inv_2beta, 4.0f0)
end

# ──────────────────────────────────────────────────────────────────────
# Three-regime (Taylor) alpha solver for D3Q39 — see D3Q27 version for
# full documentation.  Identical algorithm, different MVector size.
# ──────────────────────────────────────────────────────────────────────
@inline function solve_alpha_taylor_39(
    f_eq::MVector{39, Float32},
    s::MVector{39, Float32},
    h::MVector{39, Float32},
    w_arr,
    beta::Float32,
    H_pre::Float32,
    max_iter::Int32,
    n::Int32
)
    dev = 0.0f0
    for k in Int32(1):n
        fneq_k = s[k] + h[k]
        f_k = max(f_eq[k] + fneq_k, F_FLOOR)
        t = abs(fneq_k) / f_k
        dev = max(dev, t)
        dev > 0.01f0 && break
    end

    inv_2beta = 0.5f0 / max(beta, 0.01f0)

    # Regime 1: Trivial
    if dev < 1.0f-6
        return 2.0f0
    end

    # Regime 2: Taylor series (closed-form, no log, no iteration)
    if dev < 0.01f0
        a1 = 0.0f0
        a2 = 0.0f0
        a3 = 0.0f0
        a4 = 0.0f0
        for k in Int32(1):n
            t = s[k] + h[k]
            f_k = max(f_eq[k] + t, F_FLOOR)
            inv = 1.0f0 / f_k
            p = t * t * inv
            a1 += p
            p *= t * inv
            a2 += p
            p *= t * inv
            a3 += p
            p *= t * inv
            a4 += p
        end
        a1 *= 0.5f0
        a2 *= -1.0f0 / 6.0f0
        a3 *= 1.0f0 / 12.0f0
        a4 *= -0.05f0

        if abs(a1) > 1.0f-20
            inv_a1 = 1.0f0 / a1
            alpha = 2.0f0 - 4.0f0 * a2 * inv_a1 +
                    16.0f0 * a2 * a2 * inv_a1 * inv_a1 - 8.0f0 * a3 * inv_a1 +
                    80.0f0 * a2 * a3 * inv_a1 * inv_a1 - 80.0f0 * a2 * a2 * a2 * inv_a1 * inv_a1 * inv_a1 -
                    16.0f0 * a4 * inv_a1
            return clamp(alpha, inv_2beta, 6.0f0)
        else
            return 2.0f0
        end
    end

    # Regime 3: Large deviation — fall back to bisection
    return solve_alpha_bisection_39(f_eq, s, h, w_arr, beta, H_pre, max_iter, n)
end

# ── FUSED α-solvers (2026-05-30) — take f_stored/f_eq/opp_arr instead of the
# materialized s/h arrays, recomputing s_k/h_k per-k via _mirror_sh_inline.
# Algebraically IDENTICAL to the non-fused versions; used only on the no-HRR-
# no-σ fast path so the kernel can skip allocating s_arr+h_arr (78 floats).
@inline function solve_alpha_bisection_39_fused(
    f_stored::MVector{39, Float32}, f_eq::MVector{39, Float32}, opp_arr,
    w_arr, beta::Float32, H_pre::Float32, max_iter::Int32, h_scale::Float32, n::Int32
)
    inv_2beta = 0.5f0 / max(beta, 0.01f0)
    alpha_lo = max(inv_2beta - 0.1f0, 0.1f0)
    alpha_hi = max(inv_2beta + 3.0f0, 4.0f0)
    H_lo = entropy_at_alpha_39_fused(f_stored, f_eq, opp_arr, w_arr, beta, alpha_lo, h_scale, n)
    H_hi = entropy_at_alpha_39_fused(f_stored, f_eq, opp_arr, w_arr, beta, alpha_hi, h_scale, n)
    if !((H_lo - H_pre) * (H_hi - H_pre) < 0.0f0)
        return inv_2beta
    end
    for _ in Int32(1):max_iter
        alpha_mid = 0.5f0 * (alpha_lo + alpha_hi)
        H_mid = entropy_at_alpha_39_fused(f_stored, f_eq, opp_arr, w_arr, beta, alpha_mid, h_scale, n)
        if (H_mid - H_pre) * (H_lo - H_pre) > 0.0f0
            alpha_lo = alpha_mid; H_lo = H_mid
        else
            alpha_hi = alpha_mid; H_hi = H_mid
        end
        if (alpha_hi - alpha_lo) < 0.005f0 * (inv_2beta + 1.0f-6)
            break
        end
    end
    return 0.5f0 * (alpha_lo + alpha_hi)
end

@inline function solve_alpha_newton_39_fused(
    f_stored::MVector{39, Float32}, f_eq::MVector{39, Float32}, opp_arr,
    w_arr, beta::Float32, H_pre::Float32, max_iter::Int32, tol::Float32, h_scale::Float32, n::Int32
)
    alpha = 2.0f0
    inv_2beta = 0.5f0 / max(beta, 0.01f0)
    for iter in Int32(1):max_iter
        H_val, dH = entropy_and_deriv_at_alpha_39_fused(f_stored, f_eq, opp_arr, w_arr, beta, alpha, h_scale, n)
        residual = H_val - H_pre
        if abs(residual) < tol; break; end
        if abs(dH) < 1.0f-12; break; end
        delta = residual / dH
        alpha -= delta
        alpha = clamp(alpha, max(inv_2beta - 0.1f0, 0.1f0), 6.0f0)
        if abs(delta) < 1.0f-5; break; end
    end
    if !(alpha > 0.0f0); alpha = inv_2beta; end
    return alpha
end

@inline function solve_alpha_approximate_39_fused(
    f_stored::MVector{39, Float32}, f_eq::MVector{39, Float32}, opp_arr,
    w_arr, beta::Float32, h_scale::Float32, n::Int32
)
    h_norm_sq = 0.0f0
    f_norm_sq = 0.0f0
    for k in Int32(1):n
        _, h_k = _mirror_sh_inline(f_stored, f_eq, opp_arr, k, h_scale)
        h_norm_sq += h_k * h_k
        f_norm_sq += f_eq[k] * f_eq[k]
    end
    ratio = h_norm_sq / max(f_norm_sq, 1.0f-15)
    inv_2beta = 0.5f0 / max(beta, 0.01f0)
    alpha = 2.0f0 - 0.5f0 * ratio
    return clamp(alpha, inv_2beta, 4.0f0)
end

@inline function solve_alpha_taylor_39_fused(
    f_stored::MVector{39, Float32}, f_eq::MVector{39, Float32}, opp_arr,
    w_arr, beta::Float32, H_pre::Float32, max_iter::Int32, h_scale::Float32, n::Int32
)
    dev = 0.0f0
    for k in Int32(1):n
        s_k, h_k = _mirror_sh_inline(f_stored, f_eq, opp_arr, k, h_scale)
        fneq_k = s_k + h_k
        f_k = max(f_eq[k] + fneq_k, F_FLOOR)
        t = abs(fneq_k) / f_k
        dev = max(dev, t)
        dev > 0.01f0 && break
    end
    inv_2beta = 0.5f0 / max(beta, 0.01f0)
    if dev < 1.0f-6
        return 2.0f0
    end
    if dev < 0.01f0
        a1 = 0.0f0; a2 = 0.0f0; a3 = 0.0f0; a4 = 0.0f0
        for k in Int32(1):n
            s_k, h_k = _mirror_sh_inline(f_stored, f_eq, opp_arr, k, h_scale)
            t = s_k + h_k
            f_k = max(f_eq[k] + t, F_FLOOR)
            inv = 1.0f0 / f_k
            p = t * t * inv; a1 += p
            p *= t * inv; a2 += p
            p *= t * inv; a3 += p
            p *= t * inv; a4 += p
        end
        a1 *= 0.5f0; a2 *= -1.0f0 / 6.0f0; a3 *= 1.0f0 / 12.0f0; a4 *= -0.05f0
        if abs(a1) > 1.0f-20
            inv_a1 = 1.0f0 / a1
            alpha = 2.0f0 - 4.0f0 * a2 * inv_a1 +
                    16.0f0 * a2 * a2 * inv_a1 * inv_a1 - 8.0f0 * a3 * inv_a1 +
                    80.0f0 * a2 * a3 * inv_a1 * inv_a1 - 80.0f0 * a2 * a2 * a2 * inv_a1 * inv_a1 * inv_a1 -
                    16.0f0 * a4 * inv_a1
            return clamp(alpha, inv_2beta, 6.0f0)
        else
            return 2.0f0
        end
    end
    return solve_alpha_bisection_39_fused(f_stored, f_eq, opp_arr, w_arr, beta, H_pre, max_iter, h_scale, n)
end


# ==============================================================================
# MAIN KERNEL: stream_collide_entropic_d3q39!
# ==============================================================================
@kernel function stream_collide_entropic_d3q39!(
    f_out, f_in, f_post_collision,
    rho_out, vel_out, vel_in, rho_in_prev,
    obstacle, sponge_arr, wall_dist_arr, amr_covered_in,
    neighbor_table,
    active_coords_x, active_coords_y, active_coords_z,
    parent_f, parent_rho, parent_vel, parent_ptr,
    parent_f_old, parent_rho_old, parent_vel_old,
    parent_temperature, parent_temperature_old, parent_pi_neq,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    tau_molecular::Float32,
    tau_parent::Float32,
    c_wale::Float32,
    nu_sgs_background::Float32,
    amr_sgs_scale::Float32,
    cs2_val::Float32, cs4_val::Float32,
    ::Val{IS_L1_VAL},
    ymin_wall_bc::Int32,
    _zwall_bc::Int32,        # advanced.boundary.z_walls (lbm-zwalls): uniform-signature stub, unused here (config refuses z_walls on D3Q39)
    _level_visc_physical::Int32,  # advanced.refinement.level_viscosity (lbm-taufix): stub, unused here (config refuses "physical" on D3Q39)
    nx_global::Int32, ny_global::Int32, nz_global::Int32,
    u_inlet::Float32,
    n_blocks::Int32,
    block_size::Int32,
    cx_arr, cy_arr, cz_arr, w_arr, opp_arr,
    mirror_y_arr, mirror_z_arr,
    # 2026-05-30: Val-promoted — dead-codes the ~85-line Van Driest/Larsson
    # wall-model block [~1586] per-case when the wall model is off (045/047).
    ::Val{WALL_MODEL_VAL},
    wall_model_max_distance::Float32,
    time_step_seed::Int32,
    store_post_collision::Int32,
    inlet_turbulence::Float32,
    temporal_weight::Float32,
    use_temporal_interp::Int32,
    use_macro_temporal_interp::Int32,
    use_parent_pi_neq::Int32,
    interface_entropic_h_scale::Float32,
    sponge_blend_distributions::Int32,
    # ── Cumulant params (unused, kept for uniform interface) ──
    _omega_bulk_param::Float32,
    _omega_3_param::Float32,
    _omega_4_param::Float32,
    _adaptive_omega4::Int32,    # Reused: 0=bisection, 1=newton, 2=approximate
    _lambda_param::Float32,
    _limiter_type::Int32,
    # ── Compressibility correction (unused for D3Q39 — always numerical eq) ──
    compressible_corr::Int32,
    # ── Phase 4: two-way thermal coupling ──
    temperature_arr,           # temperature field from DDF (or dummy = rho when no thermal)
    T_inlet::Float32,          # freestream temperature (1.0 for athermal)
    thermal_coupling::Int32,   # 0 = isothermal (T ignored), 1 = coupled (p = ρ·cs²·T)
    feedback_relaxation::Float32,
    shock_tube_thermal::Int32, # AUDIT 2026-06-03: 1 ⇒ ungate Mach-limited thermal feedback (Riemann/shock-tube), keep per-side T
    # ── Phase 4b: Frapolli energy transport — kinetic energy output ──
    kinetic_energy_out,        # [8,8,8,n_blocks] array: stores Σf·|c_unshifted|² per cell
    Cv_thermal::Float32,       # C_v (2.5 for air). 0.0 = no energy output (skip E_kin store)
    # ── Phase 5: Pre-computed inlet/outlet equilibrium ──
    f_eq_inlet_arr,            # 39-element array/table: numerical equilibrium at inlet velocity
    block_shift_arr,           # per-block local shifted-lattice x-frame (D3Q39 local mode)
    local_shift_active_i::Int32,
    transverse_freestream_boundary::Int32,
    transverse_freestream_blend::Float32,
    outlet_freestream_boundary::Int32,
    outlet_boundary_mode::Int32,
    outlet_freestream_blend::Float32,
    inlet_density_mode::Int32,
    # Non-reflecting inlet blend (2026-05-29): 1.0 = legacy fully-reflective
    # (pure f_eq imposition); <1.0 blends in the interior-extrapolated population
    # so acoustic waves transmit through the inlet instead of reflecting.
    inlet_neq_blend::Float32,
    # ── Multi-GPU block offset ──
    block_offset::Int32,       # 0 for single-GPU; >0 shifts b_idx for multi-GPU partitioning
    # ── Phase B: variable ν_sgs from local Π_neq ──
    pi_neq_in,                 # BS×BS×BS×n_blocks×6 (stub 1×1×1×1×6 when disabled)
    variable_nu_sgs_i::Int32,  # 0 = use constant nu_sgs_background; 1 = per-cell from Π_neq
    c_pi_stress::Float32,      # coupling constant for |Π_neq^pre|/(ρ·cs²)
    nu_sgs_floor::Float32,     # minimum ν_sgs in smooth regions (replaces nu_sgs_background)
    # ── Phase B4: adaptive sigma (h-damping) from |Π_neq| ──
    variable_sigma_i::Int32,   # 0 = use constant lambda_p; 1 = per-cell from Π_neq
    sigma_min_p::Float32,      # minimum damping (applied where |Π_neq| >> pi_ref)
    sigma_max_p::Float32,      # maximum damping (applied in smooth regions)
    sigma_pi_ref_p::Float32,   # |Π_neq^pre| normalization threshold
    # ── Phase C1: Xu-Sagaut sponge (post-collision additive forcing) ──
    sponge_xu_sagaut_i::Int32, # 0 = use pre-collision blending (legacy); 1 = post-collision Xu-Sagaut
    # ── Tier A-2: Knudsen-adaptive τ (Latt 2020 Eq. 2.28) ──
    knudsen_adaptive_i::Int32,  # 0 = off; 1 = apply α(ε) threshold on τ using Π_neq as ε proxy
    # ── AMR feather-band extra dissipation ──
    # Per-cell Float32 in [0,1], 1.0 at halo-adjacent face, 0.0 in interior.
    # Multiplied by feather_nu_max to add an extra viscous contribution that
    # damps the O(1e-4) density seam injected by the prolongation-streaming
    # mix at AMR boundaries. Stub (1×1×1×1) when FEATHER_DEPTH = 0.
    feather_weight_in,
    feather_nu_max::Float32,
    feather_active_i::Int32,
    # Separate switch for AMR interface h-mode cleanup. This may use the same
    # geometric weights as feathering, but it must not imply equilibrium-only
    # prolongation or added viscosity when feather_nu_max=0.
    amr_regularization_active_i::Int32,
    # ── Recursive Regularization (RR-LBM) — Hermite projection (order 2+3) ──
    # Latt-Chopard 2006, Coreixas 2017, Coreixas-Latt 2019, Schukmann 2025.
    # When active, the post-collision populations are projected onto the
    # physically representable Hermite subspace (mass + momentum + stress +
    # heat-flux). The 4-dimensional ghost subspace of D3Q39 (which contains
    # the (-1)^(y+z) checkerboard eigenmode) is annihilated exactly. See
    # src/collision/kernel_regularization_d3q39.jl for the full derivation.
    # Gated per-level by the caller in timestep.jl; pass Int32(1) to apply on
    # this level, Int32(0) to skip.
    ::Val{RR_LBM_VAL},   # 2026-05-30 Val-promoted (RR-LBM Hermite projection dead-codes when off)
    # ── Spectral Vanishing Viscosity (SVV) — Y-Z bi-Laplacian (order 4) ──
    # Tadmor 1989 / Karamanos-Karniadakis 2000 / Pasquetti 2006.
    # Addresses spatial high-wavenumber modes (e.g., (-1)^(y+z) checkerboard)
    # that have NONZERO macroscopic moments and therefore CANNOT be killed by
    # Hermite projection. The bi-Laplacian eigenvalue scales as k⁴ — 47×
    # selectivity between Nyquist (k=π) and 4-cell wavelength (k=π/4).
    # See src/collision/kernel_svv_yz_d3q39.jl for the derivation.
    # epsilon == 0 → filter disabled (no-op). Per-level gating by caller.
    svv_yz_epsilon::Float32,
    # SOTA #5: streamwise-X SVV epsilon (gated to coarse levels by caller; 0=off).
    svv_x_epsilon::Float32,
    # ── SOTA A3: two-way EOS coupling (thermodynamic ideal gas) ─────────
    # Linearized form (legacy): p = ρ·cs²·T. Thermodynamic form: p = (γ-1)·ρ·cv·T.
    # When `eos_R_flow < 0` (sentinel), the 13-moment solver falls back to cs²
    # (legacy). When set to (γ-1)·cv from the caller, the variance targets in
    # the Newton solver use the true thermodynamic R. Required for any
    # defensible supersonic claim. See SOTA_ROADMAP P2 / A3.
    eos_R_flow::Float32,
    # ── SOTA B2: Dynamic SGS (Germano-Lilly) ──────────────────────────
    # When dynamic_sgs_active == 1 AND variable_nu_sgs_i == 1, the fixed
    # hand-tuned `c_pi_stress` is replaced by a per-cell coefficient computed
    # via the Germano (1991) identity with Lilly (1992) least-squares closure.
    # See src/collision/kernel_dynamic_sgs.jl for the derivation. Cost:
    # ~6 neighbor velocity reads + ~80 FLOPs per cell. Removes per-case manual
    # c_pi tuning; coefficient auto-zeros in freestream and rises to ~0.04-0.10
    # in BL/wake. Backward compatible: pass Int32(0) for legacy behavior.
    # 2026-05-30: promoted Int32→Val{DYNAMIC_SGS_VAL} — dead-codes the ~26-line
    # Germano-Lilly block [~2099] per-case when off (same as shock_sensor).
    ::Val{DYNAMIC_SGS_VAL},
    # ── SOTA B6: Compressible wall model (Van Driest transformation) ──
    # When wall_model_compressible_i == 1 AND wall_model_active == 1, the
    # equilibrium wall model is upgraded to use the Van Driest (1956) velocity
    # transformation, scaled by the local density ratio ρ/ρ_w. Required for
    # compressible BLs at high Re — without it the log-law residual scales
    # with the BL temperature ratio (~10% error at Ma=0.84, growing rapidly
    # for hotter/colder walls). The physical wall stress is then computed
    # with ρ_w (estimated from ideal gas at constant p): τ_w = ρ_w · u_τ_VD².
    # Sets the structural foundation for Larsson 2010 NEQ-WM (the pressure-
    # gradient correction is a documented future extension).
    wall_model_compressible_i::Int32,
    wall_T_wall::Float32,
    # ── SOTA B6-ext: Larsson 2010 NEQ-WM pressure-gradient correction ──
    # When wall_model_larsson_pgr_i == 1 AND wall_model_active == 1, the
    # log-law residual u_VD/u_τ gets the pressure-gradient term
    # `+ (y / (ρ_w · u_τ²)) · ∂p/∂x_stream` added before the u_τ Picard
    # iteration. Streamwise ∂p/∂x is computed from local ρ neighbors via
    # `compute_streamwise_dpdx_inline` (see kernel_larsson_pgr.jl). Cost:
    # 6 extra rho_out reads per active wall cell + ~30 FLOPs. Backward
    # compatible: pass Int32(0) for legacy Van-Driest-only behavior.
    wall_model_larsson_pgr_i::Int32,
    # ── HRR-hybrid stress reconstruction (Coreixas 2017, Jacob-Malaspinas-Sagaut 2018,
    # Astoul 2020, Schukmann 2025) ──
    # When active, the projection-based 2nd-order non-equilibrium coefficient is
    # blended with a finite-difference strain-rate version. σ=1 → pure projection
    # (legacy entropic, no change). σ=0.98 → recommended for transonic per Schukmann
    # 2025 §3.1. Suppresses non-hydrodynamic 2nd-order modes that alias at AMR
    # interfaces and emit spurious acoustic waves.
    ::Val{HRR_HYBRID_VAL},   # 2026-05-30 Val-promoted (~44-line HRR block dead-codes when off)
    hrr_sigma::Float32,
    amr_halo_fullcell_recompose_i::Int32,
    amr_halo_fullcell_macro_alpha::Float32,
    amr_halo_fullcell_pop_alpha::Float32,
    amr_prolongation_monotone_velocity_i::Int32,
    amr_halo_temporal_macro_closure_i::Int32,
    amr_halo_eq_projection_i::Int32,
    amr_halo_constrained_completion_i::Int32,
    # Pass180 (Tier A item A2 — quadratic temporal interpolation at AMR halos).
    # Older parent snapshots (state 1 parent step before parent_*_old). When
    # use_quadratic_temporal=1, the AMR halo interpolation uses 3-point quadratic
    # Lagrange through {*_older, *_old, *_new} for the macro fields. Default 0 /
    # stub arrays → linear-in-time blend (existing behavior, backward-compatible).
    parent_rho_older,
    parent_vel_older,
    parent_temperature_older,
    use_quadratic_temporal::Int32,
    # B1 P3 (2026-05-28) — Cubic Hermite (C²) temporal interpolation.
    # Oldest parent snapshots (state 2 parent steps before parent_*_old).
    # When use_cubic_temporal=1, the AMR halo interpolation upgrades from
    # quadratic to a 4-point cubic Lagrange in time through
    # {*_oldest, *_older, *_old, *_new} at {-2dt_c, -dt_c, 0, +dt_c}.
    # Falls back to quadratic / linear when 0 or stub.
    parent_rho_oldest,
    parent_vel_oldest,
    parent_temperature_oldest,
    ::Val{USE_CUBIC_VAL},
    # B1 P3 RK4-equivalent (2026-05-28) — quintic Lagrange (C³) temporal interp.
    # The 5th anchor `*_oldest2` lives at t = -3·dt_c. When USE_QUINTIC_VAL,
    # the AMR halo macro interpolation upgrades from cubic to a 5-point Lagrange
    # in time through {*_oldest2, *_oldest, *_older, *_old, *_new}. Falls back to
    # cubic / quadratic / linear when off or stub (the first 4 parent steps).
    parent_rho_oldest2,
    parent_vel_oldest2,
    parent_temperature_oldest2,
    ::Val{USE_QUINTIC_VAL},
    # Tier 1 Item 4 LIVE (2026-05-28) — SgsAndFeaturesPack. Bundles 11 SGS-
    # related scalars (6 Float32 thresholds + 5 Int32 feature flags) into a
    # single 44-byte isbits struct. Locals derived at body top shadow the
    # existing per-scalar args (which remain in the signature for source
    # compatibility but are now unused — LLVM dead-arg elim removes them
    # from PTX). See kernel_param_packs.jl for the rollout plan.
    sgs_cfg::SgsAndFeaturesPack,
    # AUDIT FIX (2026-06-03) — configurable macro realizability clamp bounds.
    macro_bounds::MacroBoundsPack,
    # Tier B4 (2026-05-28) — Newton solver warm-start cache.
    # Shape (BS, BS, BS, n_blocks, 13). Read at the main Newton call site
    # (STEP 3a) as init_lambda; written back after Newton converges. Cells
    # in steady state get λ already near-converged, Newton finishes in 1-3
    # iters instead of 5-10. Initialized to NaN sentinel — Newton falls back
    # to polynomial init on the very first step.
    lambda_cache_arr,
    # ── SOTA #3 robustness (2026-05-29): Ducros shock-sensor dissipation ──
    # "Pass 197b" — wires the diagnostic-only Ducros sensor into the collision.
    # Per cell, s = |∇ρ|/ρ·dx (6-neighbor central diff). A smoothstep over
    # [thresh_lo, thresh_hi] gives boost∈[0,1] that adds boost·shock_nu_max to
    # the local ν_sgs floor — bulk-viscosity-like damping that suppresses
    # Gibbs/carbuncle at strong compressive jumps, zero in smooth flow. Required
    # for stable Ma≳1.6. shock_sensor_active==0 → no-op (boost stays 0).
    # 2026-05-30: promoted Int32→Val{SHOCK_SENSOR_VAL} so the ~50-line Ducros
    # block [~2023] is DEAD-CODE-ELIMINATED per-case when the sensor is off.
    ::Val{SHOCK_SENSOR_VAL},
    shock_nu_max::Float32,
    shock_thresh_lo::Float32,
    shock_thresh_hi::Float32,
    # ── Level-independent lattice-ν floor (2026-05-29) — see body comment.
    # Suppresses L1 streamwise/checkerboard lattice modes without breaking the
    # uniform-physical-ν AMR scaling. Applied via max() unscaled by level. 0=off.
    nu_sgs_lattice_floor::Float32,
    adaptive_ma_thresh::Float32,
    _parent_rho_spline_coeff, _parent_vel_spline_coeff, _use_bspline_macro_i::Int32,  # C2/C4 B-spline prolongation (forwarded into interpolate_entropic_mirror_d3q39 since 2026-06-27)
    # Gendre 2017 (2026-06-28): Val-gated AMR-interface normal-gradient correction.
    # The ~20-line transition-cell correction block DEAD-CODE-ELIMINATES when off
    # ⇒ true byte-identical (zero added register pressure) for cases that don't opt in.
    ::Val{GENDRE_VAL}
) where {IS_L1_VAL, USE_CUBIC_VAL, USE_QUINTIC_VAL, SHOCK_SENSOR_VAL, DYNAMIC_SGS_VAL, WALL_MODEL_VAL, HRR_HYBRID_VAL, RR_LBM_VAL, GENDRE_VAL}
    # ── Val-specialization derivations (C4 win 2026-05-28) ─────────────────
    # These five flags were runtime `::Int32` kernel params. Promoting them to
    # `::Val{T}` type-parameters lets Julia compile a separate kernel variant
    # per (IS_L1, USE_CUBIC, USE_QUINTIC, USE_QUARTIC, USE_FP64_HALO) tuple,
    # so the compiler can dead-code the irrelevant branches at IR-gen time.
    # The runtime Int32 locals are kept for backward-compatible pass-through
    # to @inline helpers (LLVM constant-folds them through inlining).
    is_level_1 = IS_L1_VAL ? Int32(1) : Int32(0)
    use_cubic_temporal = USE_CUBIC_VAL ? Int32(1) : Int32(0)
    use_quintic_temporal = USE_QUINTIC_VAL ? Int32(1) : Int32(0)
    shock_sensor_active = SHOCK_SENSOR_VAL ? Int32(1) : Int32(0)  # 2026-05-30 Val-promoted
    dynamic_sgs_active = DYNAMIC_SGS_VAL ? Int32(1) : Int32(0)     # 2026-05-30 Val-promoted
    wall_model_active = WALL_MODEL_VAL ? Int32(1) : Int32(0)       # 2026-05-30 Val-promoted
    hrr_hybrid_active = HRR_HYBRID_VAL ? Int32(1) : Int32(0)       # 2026-05-30 Val-promoted
    rr_lbm_active_i = RR_LBM_VAL ? Int32(1) : Int32(0)            # 2026-05-30 Val-promoted
    # GENDRE_VAL (2026-06-28) is consumed directly in the write loop (DCE when off).
    # FUSED s/h fast path (2026-05-30): when HRR is off, the mirror-decomp s/h
    # are pure functions of Δf=f_stored−f_eq (variable_σ's scalar h-damp is
    # folded via h_scale), so we skip materializing s_arr/h_arr (78 floats) and
    # recompute per-k in the α-solver + KBC. Compile-time Val ⇒ the unused branch
    # (and its MVector{39} allocations) is dead-code-eliminated.
    # FUSED s/h fast path (2026-05-30): when HRR is off, skip materializing
    # s_arr/h_arr (78 floats) — recompute s/h per-k from f_stored/f_eq in the
    # α-solver + KBC. Cuts L1 spill 1072→592 B (−45%). The σ h-damp is folded as
    # the scalar h_scale_fused. BUG FOUND + FIXED: the fused KBC reads f_stored[k]
    # AND f_stored[opp(k)] inside the write loop, so f_post must NOT alias f_stored
    # on this path (it does on the non-fused path) — see the f_post allocation in
    # STEP 8. Verified byte-identical on the variable_sigma-ON config (045b).
    USE_FUSED_SH = !HRR_HYBRID_VAL

    # ── SgsAndFeaturesPack live derivation (A1 win 2026-05-28) ────────────
    # Locals shadow the per-scalar kernel args (still in signature for source
    # compat). Body references these locals → original args become unused →
    # LLVM dead-arg elim drops them from PTX, freeing 11 arg-shadow registers.
    c_pi_stress = sgs_cfg.c_pi_stress
    nu_sgs_floor = sgs_cfg.nu_sgs_floor
    sigma_min_p = sgs_cfg.sigma_min_p
    sigma_max_p = sgs_cfg.sigma_max_p
    sigma_pi_ref_p = sgs_cfg.sigma_pi_ref_p
    feather_nu_max = sgs_cfg.feather_nu_max
    variable_nu_sgs_i = sgs_cfg.variable_nu_sgs_i
    variable_sigma_i = sgs_cfg.variable_sigma_i
    sponge_xu_sagaut_i = sgs_cfg.sponge_xu_sagaut_i
    knudsen_adaptive_i = sgs_cfg.knudsen_adaptive_i
    feather_active_i = sgs_cfg.feather_active_i

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
            is_covered = amr_covered_in[x, y, z, b_idx] && !is_obs

            n_d = Int32(39)

            # ── Galilean shift (Phase 3.2) ──
            # Extract shift from rest direction (index 1): c_rest = (0,0,0) unshifted,
            # so cx_arr[1] = shift_x, cy_arr[1] = shift_y, cz_arr[1] = shift_z.
            # When shift=0, these are all zero and all u_rel computations reduce to u.
            shift_x = Float32(cx_arr[Int32(1)])
            shift_y = Float32(cy_arr[Int32(1)])
            shift_z = Float32(cz_arr[Int32(1)])
            array_shift_x_i = Int32(round(shift_x))
            array_shift_y_i = Int32(round(shift_y))
            array_shift_z_i = Int32(round(shift_z))
            local_shift_active = local_shift_active_i == Int32(1)
            local_frame_code = local_shift_active ? block_shift_arr[b_idx] :
                                 local_shift_encode3_d3q39(array_shift_x_i, array_shift_y_i, array_shift_z_i)
            frame_shift_x_i = local_shift_active ? local_shift_decode_x_d3q39(local_frame_code) : array_shift_x_i
            frame_shift_y_i = local_shift_active ? local_shift_decode_y_d3q39(local_frame_code) : array_shift_y_i
            frame_shift_z_i = local_shift_active ? local_shift_decode_z_d3q39(local_frame_code) : array_shift_z_i
            frame_shift_x = Float32(frame_shift_x_i)
            frame_shift_y = Float32(frame_shift_y_i)
            frame_shift_z = Float32(frame_shift_z_i)
            f_eq_inlet_offset = (local_shift_active && frame_shift_y_i == Int32(0) &&
                                 frame_shift_z_i == Int32(0) && frame_shift_x_i >= Int32(0)) ?
                                frame_shift_x_i * n_d : Int32(0)

            # ══════════════════════════════════════════════════════════
            # STEP 1: Pull-streaming — gather 39 distributions
            # ══════════════════════════════════════════════════════════
            # For D3Q39, velocities can be |c| ≤ 3 (unshifted), |c'| ≤ 3+|U| (shifted).
            # The pull scheme: f_stored[k] = f_in(x - c'_k)
            # Multi-hop streaming is handled by the same neighbor table
            # because BLOCK_SIZE=8 >> max|c'| means at most 1 block hop.

            rho = 0.0f0
            jx  = 0.0f0
            jy  = 0.0f0
            jz  = 0.0f0
            E_kin_shifted = 0.0f0   # Phase 4b: Σf·|c'_shifted|² for Frapolli T recovery
            # Inlet-conservation flag: set when this cell pulls from outside the
            # DOMAIN inlet (the is_inlet branch fires). Distinguishes the true
            # domain-inlet face from a fine-level AMR edge (which is handled by the
            # halo branch, not the domain-boundary branch). See the mass+momentum
            # pin applied after the streaming loop.
            cell_saw_inlet = false
            f_stored = MVector{39, Float32}(undef)
            parent_halo_mask = MVector{39, Int32}(undef)
            for kk in Int32(1):n_d
                parent_halo_mask[kk] = Int32(0)
            end
            parent_halo_links = Int32(0)
            parent_halo_x_neg = Int32(0)
            parent_halo_x_pos = Int32(0)
            parent_halo_y_neg = Int32(0)
            parent_halo_y_pos = Int32(0)
            parent_halo_z_neg = Int32(0)
            parent_halo_z_pos = Int32(0)
            parent_halo_y_neg_pure = Int32(0)
            parent_halo_y_pos_pure = Int32(0)
            parent_halo_z_neg_pure = Int32(0)
            parent_halo_z_pos_pure = Int32(0)

            # Inlet/outlet BCs use fast polynomial O(u²) equilibrium.
            # Only the main collision step uses the 13-moment Newton-Raphson numerical
            # equilibrium. This keeps the kernel size manageable for GPU JIT compilation.

            for k in Int32(1):n_d
                val = 0.0f0

                if is_covered
                    val = f_in[x, y, z, b_idx, k]
                else
                    cx_base = Int32(cx_arr[k]) - array_shift_x_i
                    cy_base = Int32(cy_arr[k]) - array_shift_y_i
                    cz_base = Int32(cz_arr[k]) - array_shift_z_i
                    cx = cx_base + frame_shift_x_i
                    cy = cy_base + frame_shift_y_i
                    cz = cz_base + frame_shift_z_i

                    sx = Int32(x) - cx
                    sy = Int32(y) - cy
                    sz = Int32(z) - cz

                    if sx >= Int32(1) && sx <= block_size && sy >= Int32(1) && sy <= block_size && sz >= Int32(1) && sz <= block_size
                        val = f_in[sx, sy, sz, b_idx, k]
                    else
                    # Multi-hop: compute which neighbor block to read from.
                    # For |c| ≤ 3, off ∈ {-1,0,+1} because BLOCK_SIZE=8.
                    nb_off_x = sx < Int32(1) ? Int32(-1) : (sx > block_size ? Int32(1) : Int32(0))
                    nb_off_y = sy < Int32(1) ? Int32(-1) : (sy > block_size ? Int32(1) : Int32(0))
                    nb_off_z = sz < Int32(1) ? Int32(-1) : (sz > block_size ? Int32(1) : Int32(0))
                    dir_idx = (nb_off_x + Int32(1)) + (nb_off_y + Int32(1))*Int32(3) + (nb_off_z + Int32(1))*Int32(9) + Int32(1)
                    nb_idx_global = neighbor_table[b_idx, dir_idx]

                    if nb_idx_global > Int32(0)
                        # Wrap local coordinate into neighbor block
                        nsx = sx < Int32(1) ? sx + block_size : (sx > block_size ? sx - block_size : sx)
                        nsy = sy < Int32(1) ? sy + block_size : (sy > block_size ? sy - block_size : sy)
                        nsz = sz < Int32(1) ? sz + block_size : (sz > block_size ? sz - block_size : sz)
                        if local_shift_active
                            src_code = block_shift_arr[nb_idx_global]
                            src_shift_x_i = local_shift_decode_x_d3q39(src_code)
                            src_shift_y_i = local_shift_decode_y_d3q39(src_code)
                            src_shift_z_i = local_shift_decode_z_d3q39(src_code)
                            if src_shift_x_i == frame_shift_x_i &&
                               src_shift_y_i == frame_shift_y_i &&
                               src_shift_z_i == frame_shift_z_i
                                val = f_in[nsx, nsy, nsz, nb_idx_global, k]
                            else
                                transverse_frame_change =
                                    src_shift_y_i != frame_shift_y_i ||
                                    src_shift_z_i != frame_shift_z_i
                                if transverse_frame_change
                                    rho_src = rho_in_prev[nsx, nsy, nsz, nb_idx_global]
                                    ux_src = vel_in[nsx, nsy, nsz, nb_idx_global, Int32(1)]
                                    uy_src = vel_in[nsx, nsy, nsz, nb_idx_global, Int32(2)]
                                    uz_src = vel_in[nsx, nsy, nsz, nb_idx_global, Int32(3)]
                                    val = local_shift_regularized_reconstruct3_d3q39(
                                        f_in,
                                        nsx, nsy, nsz, nb_idx_global,
                                        rho_src, ux_src, uy_src, uz_src,
                                        Float32(src_shift_x_i), Float32(src_shift_y_i), Float32(src_shift_z_i),
                                        frame_shift_x, frame_shift_y, frame_shift_z,
                                        k, cx_arr, cy_arr, cz_arr, w_arr,
                                        cs2_val, T_inlet)
                                else
                                    km = local_shift_remap_d3q39(
                                        cx_arr, cy_arr, cz_arr, k,
                                        frame_shift_x_i - src_shift_x_i,
                                        frame_shift_y_i - src_shift_y_i,
                                        frame_shift_z_i - src_shift_z_i)
                                    if km > Int32(0)
                                        val = f_in[nsx, nsy, nsz, nb_idx_global, km]
                                    else
                                        rho_src = rho_in_prev[nsx, nsy, nsz, nb_idx_global]
                                        ux_src = vel_in[nsx, nsy, nsz, nb_idx_global, Int32(1)]
                                        uy_src = vel_in[nsx, nsy, nsz, nb_idx_global, Int32(2)]
                                        uz_src = vel_in[nsx, nsy, nsz, nb_idx_global, Int32(3)]
                                        val = local_shift_regularized_reconstruct3_d3q39(
                                            f_in,
                                            nsx, nsy, nsz, nb_idx_global,
                                            rho_src, ux_src, uy_src, uz_src,
                                            Float32(src_shift_x_i), Float32(src_shift_y_i), Float32(src_shift_z_i),
                                            frame_shift_x, frame_shift_y, frame_shift_z,
                                            k, cx_arr, cy_arr, cz_arr, w_arr,
                                            cs2_val, T_inlet)
                                    end
                                end
                            end
                        else
                            val = f_in[nsx, nsy, nsz, nb_idx_global, k]
                        end
                    else
                        # Domain boundary
                        src_gx = gx - cx
                        # Periodic-Y AMR: a source across the Y seam is the wrapped
                        # cell (prolongated from the wrapped parent); identity when off.
                        src_gy = _pwrap_y(gy - cy, ny_global)
                        src_gz = gz - cz

                        is_inlet  = src_gx < Int32(1)
                        is_outlet = src_gx > nx_global
                        is_y_min  = src_gy < Int32(1)
                        is_y_max  = src_gy > ny_global
                        is_z_min  = src_gz < Int32(1)
                        is_z_max  = src_gz > nz_global

                        # Mark the cell as a true domain-inlet cell (any incoming
                        # direction pulls from upstream of the domain). Used to pin
                        # the inlet face to the exact freestream mass+momentum after
                        # the streaming loop completes.
                        cell_saw_inlet = cell_saw_inlet || is_inlet

                        at_y_boundary = is_y_min || is_y_max
                        at_z_boundary = is_z_min || is_z_max

                        if transverse_freestream_boundary == Int32(1) && (at_y_boundary || at_z_boundary)
                            # beta=1 preserves the legacy hard farfield clamp; beta=0 gives
                            # zero-gradient carry-over.  The carry-over base is less
                            # injective than a mirror ghost for multi-speed D3Q39 acoustics.
                            beta_t = transverse_freestream_blend
                            donor = f_in[x, y, z, b_idx, k]
                            val = (1.0f0 - beta_t) * donor + beta_t * f_eq_inlet_arr[f_eq_inlet_offset + k]
                        elseif (outlet_boundary_mode == Int32(1) ||
                                (outlet_boundary_mode == Int32(0) && outlet_freestream_boundary == Int32(1))) && is_outlet
                            beta = outlet_freestream_blend
                            val = (1.0f0 - beta) * f_in[x, y, z, b_idx, k] + beta * f_eq_inlet_arr[f_eq_inlet_offset + k]
                        elseif outlet_boundary_mode == Int32(3) && is_outlet
                            donor = f_in[x, y, z, b_idx, k]
                            if x > Int32(1)
                                donor = f_in[x - Int32(1), y, z, b_idx, k]
                            else
                                # One-cell interior extrapolation across a block edge.
                                # Direction (-x,0,0) maps to neighbor-table index 13.
                                nb_xm = neighbor_table[b_idx, Int32(13)]
                                if nb_xm > Int32(0)
                                    donor = f_in[block_size, y, z, nb_xm, k]
                                end
                            end
                            beta = outlet_freestream_blend
                            val = (1.0f0 - beta) * donor + beta * f_eq_inlet_arr[f_eq_inlet_offset + k]
                        elseif is_inlet && at_y_boundary && at_z_boundary
                            # Inlet corner: subordinate the x-inlet prescription to the
                            # slip walls, mirroring the outlet-corner treatment below.
                            # Pure freestream equilibrium here resets the corner
                            # non-equilibrium every step and prints a bright corner
                            # artifact into schlieren that can seed weak reflections.
                            k_mirror_z = mirror_z_arr[k]
                            val = f_in[x, y, z, b_idx, mirror_y_arr[k_mirror_z]]
                        elseif is_inlet && at_y_boundary
                            # Inlet+y-wall edge: preserve the slip wall rather than
                            # overwriting the edge state with pure inlet equilibrium.
                            val = f_in[x, y, z, b_idx, mirror_y_arr[k]]
                        elseif is_inlet && at_z_boundary
                            # Inlet+z-wall edge: same rationale as the y-wall edge.
                            val = f_in[x, y, z, b_idx, mirror_z_arr[k]]
                        elseif is_inlet
                            # Interior inlet: impose the ramped inlet velocity, but
                            # optionally carry the local boundary density. The local
                            # mode avoids resetting missing populations to rho=1
                            # during acceleration, which otherwise pumps a global
                            # compressible mass transient into obstacle-free AMR tests.
                            #
                            # 2026-05-29 bug fix — "local" mode previously read the
                            # bare local density, which is a positive feedback loop
                            # against any density pile-up from upstream stagnation
                            # waves (e.g., bow-shock reflection from the sphere).
                            # Case 030 saw the inlet ρ build to 1.32 over 200 steps
                            # — a 32% mass-pump that ultimately seeded the L1↔L2
                            # AMR explosion. The fix: anchor the local read toward
                            # the freestream target ρ=1 with a 90/10 blend, so
                            # accumulated stagnation can still relieve through the
                            # inlet but the inlet never reinforces a runaway pile-up.
                            rho_inlet_bc = 1.0f0
                            if inlet_density_mode == Int32(1)
                                rho_local = rho_out[x, y, z, b_idx]
                                if !(rho_local == rho_local) || rho_local < 0.01f0
                                    rho_local = 1.0f0
                                end
                                # Soft anchor: 90% local + 10% freestream target.
                                # Equivalent to a relaxation rate of 0.1/step toward ρ=1.
                                rho_inlet_bc = 0.9f0 * rho_local + 0.1f0 * 1.0f0
                                # Bound to prevent runaway transients. Was [0.8,1.2]
                                # (tuned for ρ≈1 aero cases) but that floors low-density
                                # left states in Riemann/shock-tube tests (Lax ρ_L=0.445).
                                # Widened to [0.1,2.0]: still kills runaway pile-ups,
                                # admits genuine low/high-density boundary states.
                                rho_inlet_bc = clamp(rho_inlet_bc, 0.1f0, 2.0f0)
                            end
                            # ── Non-reflecting inlet (2026-05-29) ──────────────
                            # Hard-imposing f = rho·f_eq_inlet on the incoming
                            # directions every step makes the inlet ACOUSTICALLY
                            # REFLECTIVE: any pressure/acoustic wave reaching the
                            # inlet from the interior is fully discarded and
                            # reflected, seeding a streamwise standing wave (the
                            # observed vertical density banding). Fix: blend the
                            # freestream-equilibrium target with the interior-
                            # extrapolated population (1 cell in +x), so outgoing
                            # acoustic content partially TRANSMITS through the inlet
                            # instead of reflecting. inlet_neq_blend=1 → legacy
                            # fully-reflective (pure f_eq); <1 → non-reflecting.
                            target_in = rho_inlet_bc * f_eq_inlet_arr[f_eq_inlet_offset + k]
                            if inlet_turbulence > 0.0f0
                                # D3Q39 inlet uses a precomputed equilibrium table, so
                                # apply a small linearized equilibrium perturbation here.
                                # Cross-stream components are essential for breaking the
                                # perfectly symmetric cylinder wake.
                                amp = inlet_turbulence * max(abs(u_inlet), 1.0f-6)
                                dux_t = 0.25f0 * amp * gradient_noise(gx, gy, gz, time_step_seed + Int32(1234))
                                duy_t = 0.50f0 * amp * gradient_noise(gx, gy, gz, time_step_seed + Int32(5678))
                                duz_t = amp * gradient_noise(gx, gy, gz, time_step_seed + Int32(9012))
                                cx_rel_in = Float32(cx_arr[k]) - frame_shift_x
                                cy_rel_in = Float32(cy_arr[k]) - frame_shift_y
                                cz_rel_in = Float32(cz_arr[k]) - frame_shift_z
                                target_in += Float32(w_arr[k]) * rho_inlet_bc *
                                             (cx_rel_in * dux_t + cy_rel_in * duy_t + cz_rel_in * duz_t) /
                                             max(cs2_val, 1.0f-6)
                                target_in = max(target_in, 1.0f-12)
                            end
                            if inlet_neq_blend < 1.0f0
                                # interior donor at +x (zero-gradient carry of the
                                # outgoing wave). +x neighbor dir_idx = 15.
                                donor_in = f_in[x, y, z, b_idx, k]
                                if x < block_size
                                    donor_in = f_in[x + Int32(1), y, z, b_idx, k]
                                else
                                    nb_xp = neighbor_table[b_idx, Int32(15)]
                                    if nb_xp > Int32(0)
                                        donor_in = f_in[Int32(1), y, z, nb_xp, k]
                                    end
                                end
                                val = inlet_neq_blend * target_in +
                                      (1.0f0 - inlet_neq_blend) * donor_in
                            else
                                val = target_in
                            end
                        elseif is_outlet && at_y_boundary && at_z_boundary
                            # Outlet corner cell: subordinate outlet to the slip walls.
                            # Earlier 4-way mirror blend caused a density pile-up
                            # (+1% hotspot) at the top-right corner because averaging
                            # f_k with its transverse mirrors does not preserve
                            # mass locally — the blended state has a different ρ
                            # than either constituent. Use composite wall mirror
                            # (reflect in both transverse walls) so the corner
                            # respects the slip boundaries without mass gain.
                            k_mirror_z = mirror_z_arr[k]
                            val = f_in[x, y, z, b_idx, mirror_y_arr[k_mirror_z]]
                        elseif is_outlet && at_y_boundary
                            # Slip wall dominates at outlet+y-wall edge
                            val = f_in[x, y, z, b_idx, mirror_y_arr[k]]
                        elseif is_outlet && at_z_boundary
                            # Slip wall dominates at outlet+z-wall edge
                            val = f_in[x, y, z, b_idx, mirror_z_arr[k]]
                        elseif is_outlet
                            # Supersonic outlet: carry over the outgoing state instead of
                            # imposing the inlet equilibrium, which reflects compression waves.
                            val = f_in[x, y, z, b_idx, k]
                        elseif is_y_min && ymin_wall_bc == Int32(1)
                            val = f_in[x, y, z, b_idx, opp_arr[k]]
                        elseif at_y_boundary
                            val = f_in[x, y, z, b_idx, mirror_y_arr[k]]
                        elseif at_z_boundary
                            val = f_in[x, y, z, b_idx, mirror_z_arr[k]]
                        elseif is_level_1 == Int32(0)
                            parent_halo_links += Int32(1)
                            parent_halo_mask[k] = Int32(1)
                            parent_halo_x_neg += nb_off_x < Int32(0) ? Int32(1) : Int32(0)
                            parent_halo_x_pos += nb_off_x > Int32(0) ? Int32(1) : Int32(0)
                            parent_halo_y_neg += nb_off_y < Int32(0) ? Int32(1) : Int32(0)
                            parent_halo_y_pos += nb_off_y > Int32(0) ? Int32(1) : Int32(0)
                            parent_halo_z_neg += nb_off_z < Int32(0) ? Int32(1) : Int32(0)
                            parent_halo_z_pos += nb_off_z > Int32(0) ? Int32(1) : Int32(0)
                            if nb_off_x == Int32(0)
                                parent_halo_y_neg_pure += nb_off_y < Int32(0) ? Int32(1) : Int32(0)
                                parent_halo_y_pos_pure += nb_off_y > Int32(0) ? Int32(1) : Int32(0)
                                parent_halo_z_neg_pure += nb_off_z < Int32(0) ? Int32(1) : Int32(0)
                                parent_halo_z_pos_pure += nb_off_z > Int32(0) ? Int32(1) : Int32(0)
                            end
                            # Mirror-decomposed interpolation from parent
                            # In the AMR feather band, keep only the interpolated
                            # conserved macros. Coarse-grid Pi_neq/h content is a
                            # persistent source of comb/checkerboard frontier modes.
                            kb = opp_arr[k]

                            val = interpolate_entropic_mirror_d3q39(
                                parent_f, parent_rho, parent_vel,
                                parent_f_old, parent_rho_old, parent_vel_old,
                                parent_ptr,
                                parent_dim_x, parent_dim_y, parent_dim_z,
                                src_gx, src_gy, src_gz,
                                k, kb, block_size,
                                w_arr[k],  Float32(cx),         Float32(cy),         Float32(cz),
                                w_arr[kb], Float32(cx_arr[kb]), Float32(cy_arr[kb]), Float32(cz_arr[kb]),
                                tau_parent, tau_molecular,
                                temporal_weight,
                                use_temporal_interp,
                                use_macro_temporal_interp,
                                cs2_val,
                                cx_arr, cy_arr, cz_arr, w_arr,
                                shift_x, shift_y, shift_z,
                                # AMR-EOS FIX (2026-06-25): was hard-coded Int32(0),
                                # which forced the coarse→fine halo prolongation to
                                # reconstruct fine-boundary populations at T=T_inlet=1
                                # REGARDLESS of the (shock-heated) coarse-cell
                                # temperature. The collision kernel builds its
                                # equilibrium with the LOCAL T whenever
                                # thermal_coupling==1 (p=(γ-1)ρ·cv·T via the 13-mom
                                # solver), so the prolongation at T=1 produced a
                                # coherent bulk pressure DEFICIT (≈T_coarse×) along the
                                # whole refinement-box boundary — the rectangular
                                # box-edge step / refined-vs-coarse offset. Gating on
                                # the SAME thermal_coupling flag the collision uses
                                # makes prolongation and collision share the identical
                                # EOS across the dx jump. At T≈1 (isothermal / D3Q27)
                                # this is byte-identical to the legacy Int32(0).
                                thermal_coupling,
                                parent_temperature,
                                parent_temperature_old,
                                use_parent_pi_neq,
                                parent_pi_neq,
                                interface_entropic_h_scale,
                                T_inlet,
                                amr_prolongation_monotone_velocity_i,
                                # ★ STRUCTURAL FIX A (2026-06-25, verified D3Q27-vs-D3Q39 diff):
                                # near_wall_eq_only Int32(1)→Int32(0). The hard-wired Int32(1)
                                # made the D3Q39 halo return EQUILIBRIUM-ONLY (discarding the
                                # Π_neq deviatoric stress + momentum flux), where the WORKING
                                # D3Q27 path (interpolate_with_rescaling) carries feq + rescaled
                                # real f_neq. A Ma1.5 shock is encoded almost entirely as
                                # non-equilibrium STRESS; a stress-free halo = a zero-momentum-
                                # flux ring = a stagnation/slip-WALL the supersonic flow deflects
                                # around (the high-Mach perimeter ring). Int32(0) carries the
                                # τ-rescaled stress across the seam, matching D3Q27 so the halo
                                # is a smooth flow continuation, not a barrier. D3Q39's max|c|=3
                                # makes this far more impactful than on D3Q27 (max|c|=1).
                                # MUST be paired with AMR_HALO_EQ_PROJECTION off (Fix B).
                                Int32(0),
                                # Pass180: quadratic temporal interpolation through
                                # {parent_*_older, parent_*_old, parent_*_new} when
                                # use_quadratic_temporal=1 AND macro_temporal=1. The
                                # interpolate function gates internally on both.
                                parent_rho_older,
                                parent_vel_older,
                                parent_temperature_older,
                                use_quadratic_temporal,
                                # B1 P3 (2026-05-28): cubic Hermite (C²) temporal
                                # interpolation through {oldest, older, old, new}
                                # when use_cubic_temporal=1. Falls back to quadratic
                                # / linear inside the interpolator when 0 or stub.
                                parent_rho_oldest,
                                parent_vel_oldest,
                                parent_temperature_oldest,
                                use_cubic_temporal,
                                # B1 P3 RK4-equivalent (2026-05-28): quintic Lagrange
                                # (C³) temporal interpolation through the 5 anchors
                                # {oldest2, oldest, older, old, new} when
                                # use_quintic_temporal=1. Falls back to cubic /
                                # quadratic / linear inside the interpolator when 0
                                # or stub (the first 4 parent-step warm-up window).
                                parent_rho_oldest2,
                                parent_vel_oldest2,
                                parent_temperature_oldest2,
                                use_quintic_temporal,
                                # SOTA A3 — thermodynamic EOS R. Pass the SAME
                                # eos_R_flow the collision solve uses so the
                                # prolongation f_eq is identical across the AMR
                                # interface (no spurious p/ρ step at T≠1).
                                eos_R_flow,
                                # C2/C4 QUINTIC B-SPLINE prolongation (2026-06-27):
                                # forward the parent spline coeffs + gate the kernel
                                # already receives (the _parent_*_spline_coeff /
                                # _use_bspline_macro_i stubs). When enabled AND the
                                # symmetric ±3 parent stencil is present, the macro
                                # (ρ,u) is evaluated from these coeffs via the
                                # @noinline interpolate_macro_bspline (C4 + exact 2:1
                                # conservation); on a ragged box edge it falls back to
                                # the cubic-Lagrange macros. use_bspline=0 (default,
                                # no flag) ⇒ byte-identical to the legacy cubic path.
                                _parent_rho_spline_coeff,
                                _parent_vel_spline_coeff,
                                _use_bspline_macro_i
                            )
                        else
                            val = w_arr[k]
                        end
                    end
                    end
                end

                f_stored[k] = val
                rho += val
                cx_phys_k = Float32(Int32(cx_arr[k]) - array_shift_x_i + frame_shift_x_i)
                cy_phys_k = Float32(Int32(cy_arr[k]) - array_shift_y_i + frame_shift_y_i)
                cz_phys_k = Float32(Int32(cz_arr[k]) - array_shift_z_i + frame_shift_z_i)
                jx += val * cx_phys_k
                jy += val * cy_phys_k
                jz += val * cz_phys_k
                # Phase 4b: accumulate Σf·|c'|² (shifted frame) for Frapolli T recovery
                E_kin_shifted += val * (cx_phys_k*cx_phys_k +
                                        cy_phys_k*cy_phys_k +
                                        cz_phys_k*cz_phys_k)
            end

            # ══════════════════════════════════════════════════════════
            # INLET CONSERVATION (2026-06-28): pin the domain-inlet FACE to the
            # exact freestream mass + momentum.
            #
            # The pull-scheme inlet imposes f_eq_inlet only on the incoming
            # (upstream-sourced) directions; the outgoing/transverse directions
            # stream from the interior. The assembled inlet cell's Σf / Σf·c
            # therefore drift away from (ρ_∞, ρ_∞·u_∞) — at Ma 0.9 this produced
            # the rho→2.8, velocity-clamped inlet streaks (which subsonically
            # propagate upstream and seed the AMR-box artifact). A mass + momentum
            # (4-moment) correction pins the CONSERVED hydrodynamic moments exactly
            # to the freestream while leaving the higher (acoustic) moments free to
            # transmit — i.e. characteristic-consistent for the subsonic inlet.
            #
            # The target is taken from f_eq_inlet_arr itself (built for ρ=1, u_inlet
            # in the current frame), so it is exact for shifted and unshifted
            # lattices alike. Applied only on the first cell column (gx==1) of the
            # true domain-inlet face; cell_saw_inlet excludes fine-level AMR edges.
            if cell_saw_inlet && gx == Int32(1) && !is_obs && !is_covered &&
               inlet_density_mode == Int32(0)
                # Gated to inlet_density "freestream" (mode 0): the exact mass+momentum
                # pin is the correct enforcement of a HARD freestream inlet (supersonic,
                # or a deliberately-clamped subsonic farfield). For "local" (mode 1) the
                # inlet is intentionally SOFT/non-reflecting (it carries the local
                # boundary density), and hard-pinning it would make it acoustically
                # reflective — the source of the subsonic inlet-region streaks.
                trho_in = 0.0f0; tjx_in = 0.0f0; tjy_in = 0.0f0; tjz_in = 0.0f0
                @inbounds for ki in Int32(1):n_d
                    fe_in = f_eq_inlet_arr[f_eq_inlet_offset + ki]
                    cxp_in = Float32(Int32(cx_arr[ki]) - array_shift_x_i + frame_shift_x_i)
                    cyp_in = Float32(Int32(cy_arr[ki]) - array_shift_y_i + frame_shift_y_i)
                    czp_in = Float32(Int32(cz_arr[ki]) - array_shift_z_i + frame_shift_z_i)
                    trho_in += fe_in
                    tjx_in += fe_in * cxp_in
                    tjy_in += fe_in * cyp_in
                    tjz_in += fe_in * czp_in
                end
                # The 4-moment solver forms Σf·(c − shift); pass shift so that
                # (c − shift) == c_phys == (c − array_shift + frame_shift).
                sx_in = Float32(array_shift_x_i - frame_shift_x_i)
                sy_in = Float32(array_shift_y_i - frame_shift_y_i)
                sz_in = Float32(array_shift_z_i - frame_shift_z_i)
                apply_4momentum_correction_d3q39!(
                    f_stored, trho_in, tjx_in, tjy_in, tjz_in,
                    cx_arr, cy_arr, cz_arr, w_arr,
                    sx_in, sy_in, sz_in
                )
                # Recompute macros from the corrected populations.
                rho = 0.0f0; jx = 0.0f0; jy = 0.0f0; jz = 0.0f0; E_kin_shifted = 0.0f0
                @inbounds for ki in Int32(1):n_d
                    v_in = f_stored[ki]
                    cxp_in = Float32(Int32(cx_arr[ki]) - array_shift_x_i + frame_shift_x_i)
                    cyp_in = Float32(Int32(cy_arr[ki]) - array_shift_y_i + frame_shift_y_i)
                    czp_in = Float32(Int32(cz_arr[ki]) - array_shift_z_i + frame_shift_z_i)
                    rho += v_in
                    jx += v_in * cxp_in
                    jy += v_in * cyp_in
                    jz += v_in * czp_in
                    E_kin_shifted += v_in * (cxp_in*cxp_in + cyp_in*cyp_in + czp_in*czp_in)
                end
            end

            # ══════════════════════════════════════════════════════════
            if false &&
               amr_halo_constrained_completion_i == Int32(1) &&
               is_level_1 == Int32(0) &&
               parent_halo_links >= Int32(4) &&
               !is_obs && !is_covered
                rho_target_cc, ux_target_cc, uy_target_cc, uz_target_cc, parent_cc_valid =
                    sample_parent_macro_for_fine_cell(
                        parent_rho, parent_vel,
                        parent_rho_old, parent_vel_old,
                        parent_ptr,
                        parent_dim_x, parent_dim_y, parent_dim_z,
                        gx, gy, gz,
                        block_size,
                        temporal_weight,
                        use_temporal_interp
                    )
                if parent_cc_valid &&
                   rho_target_cc == rho_target_cc &&
                   ux_target_cc == ux_target_cc &&
                   uy_target_cc == uy_target_cc &&
                   uz_target_cc == uz_target_cc &&
                   rho_target_cc > 0.01f0
                    rho_target_cc = max(rho_target_cc, 0.01f0)
                    lambda_parent_cc = solve_13mom_lambdas_d3q39(
                        rho_target_cc,
                        ux_target_cc - shift_x,
                        uy_target_cc - shift_y,
                        uz_target_cc - shift_z,
                        cx_arr, cy_arr, cz_arr, w_arr,
                        cs2_val, Int32(80), T_inlet,
                        shift_x, shift_y, shift_z,
                        100.0f0,
                        eos_R_flow          # SOTA A3 — thermodynamic R
                    )
                    mass_corr_parent_cc = compute_13mom_mass_correction_d3q39(
                        rho_target_cc, lambda_parent_cc,
                        cx_arr, cy_arr, cz_arr, w_arr,
                        cs2_val, shift_x, shift_y, shift_z
                    )

                    m00_cc = 0.0f0
                    m01_cc = 0.0f0
                    m02_cc = 0.0f0
                    m03_cc = 0.0f0
                    m11_cc = 0.0f0
                    m12_cc = 0.0f0
                    m13_cc = 0.0f0
                    m22_cc = 0.0f0
                    m23_cc = 0.0f0
                    m33_cc = 0.0f0
                    rho_current_subset_cc = 0.0f0
                    jx_current_subset_cc = 0.0f0
                    jy_current_subset_cc = 0.0f0
                    jz_current_subset_cc = 0.0f0
                    rho_target_subset_cc = 0.0f0
                    jx_target_subset_cc = 0.0f0
                    jy_target_subset_cc = 0.0f0
                    jz_target_subset_cc = 0.0f0
                    active_cc = Int32(0)
                    for kk in Int32(1):n_d
                        if parent_halo_mask[kk] == Int32(1)
                            wk_cc = max(Float32(w_arr[kk]), 1.0f-12)
                            cx_cc = Float32(cx_arr[kk])
                            cy_cc = Float32(cy_arr[kk])
                            cz_cc = Float32(cz_arr[kk])
                            f_current_cc = f_stored[kk]
                            f_target_cc = mass_corr_parent_cc * eval_13mom_feq_single(
                                rho_target_cc,
                                lambda_parent_cc,
                                cx_cc - shift_x,
                                cy_cc - shift_y,
                                cz_cc - shift_z,
                                wk_cc,
                                cs2_val
                            )

                            rho_current_subset_cc += f_current_cc
                            jx_current_subset_cc += f_current_cc * cx_cc
                            jy_current_subset_cc += f_current_cc * cy_cc
                            jz_current_subset_cc += f_current_cc * cz_cc
                            rho_target_subset_cc += f_target_cc
                            jx_target_subset_cc += f_target_cc * cx_cc
                            jy_target_subset_cc += f_target_cc * cy_cc
                            jz_target_subset_cc += f_target_cc * cz_cc

                            m00_cc += wk_cc
                            m01_cc += wk_cc * cx_cc
                            m02_cc += wk_cc * cy_cc
                            m03_cc += wk_cc * cz_cc
                            m11_cc += wk_cc * cx_cc * cx_cc
                            m12_cc += wk_cc * cx_cc * cy_cc
                            m13_cc += wk_cc * cx_cc * cz_cc
                            m22_cc += wk_cc * cy_cc * cy_cc
                            m23_cc += wk_cc * cy_cc * cz_cc
                            m33_cc += wk_cc * cz_cc * cz_cc
                            active_cc += Int32(1)
                        end
                    end

                    if active_cc >= Int32(4)
                        a0_cc, ax_cc, ay_cc, az_cc, ok_cc =
                            solve_shell_moment4(
                                m00_cc, m01_cc, m02_cc, m03_cc,
                                m11_cc, m12_cc, m13_cc,
                                m22_cc, m23_cc,
                                m33_cc,
                                rho_target_subset_cc - rho_current_subset_cc,
                                jx_target_subset_cc - jx_current_subset_cc,
                                jy_target_subset_cc - jy_current_subset_cc,
                                jz_target_subset_cc - jz_current_subset_cc
                            )
                        if ok_cc
                            pos_limiter_cc = 1.0f0
                            for kk in Int32(1):n_d
                                if parent_halo_mask[kk] == Int32(1)
                                    wk_cc = max(Float32(w_arr[kk]), 1.0f-12)
                                    cx_cc = Float32(cx_arr[kk])
                                    cy_cc = Float32(cy_arr[kk])
                                    cz_cc = Float32(cz_arr[kk])
                                    df_cc = wk_cc * (
                                        a0_cc +
                                        ax_cc * cx_cc +
                                        ay_cc * cy_cc +
                                        az_cc * cz_cc
                                    )
                                    if df_cc < 0.0f0
                                        pos_limiter_cc = min(
                                            pos_limiter_cc,
                                            max((f_stored[kk] - F_FLOOR) / (-df_cc), 0.0f0)
                                        )
                                    end
                                end
                            end
                            pos_limiter_cc = min(pos_limiter_cc, 1.0f0)

                            if pos_limiter_cc > 0.0f0
                                for kk in Int32(1):n_d
                                    if parent_halo_mask[kk] == Int32(1)
                                        wk_cc = max(Float32(w_arr[kk]), 1.0f-12)
                                        cx_cc = Float32(cx_arr[kk])
                                        cy_cc = Float32(cy_arr[kk])
                                        cz_cc = Float32(cz_arr[kk])
                                        df_cc = pos_limiter_cc * wk_cc * (
                                            a0_cc +
                                            ax_cc * cx_cc +
                                            ay_cc * cy_cc +
                                            az_cc * cz_cc
                                        )
                                        f_stored[kk] = max(f_stored[kk] + df_cc, F_FLOOR)
                                    end
                                end

                                rho = 0.0f0
                                jx = 0.0f0
                                jy = 0.0f0
                                jz = 0.0f0
                                E_kin_shifted = 0.0f0
                                for kk in Int32(1):n_d
                                    fval_cc = f_stored[kk]
                                    cx_cc = Float32(cx_arr[kk])
                                    cy_cc = Float32(cy_arr[kk])
                                    cz_cc = Float32(cz_arr[kk])
                                    rho += fval_cc
                                    jx += fval_cc * cx_cc
                                    jy += fval_cc * cy_cc
                                    jz += fval_cc * cz_cc
                                    E_kin_shifted += fval_cc * (
                                        cx_cc * cx_cc +
                                        cy_cc * cy_cc +
                                        cz_cc * cz_cc
                                    )
                                end
                            end
                        end
                    end
                end
            end

            # Obstacle: simple bounce-back
            # ══════════════════════════════════════════════════════════
            if is_covered
                rho_raw = rho
                rho = max(rho_raw, 0.01f0)
                if !(rho_raw == rho_raw)
                    rho = max(rho_out[x, y, z, b_idx], 0.01f0)
                end

                inv_rho = 1.0f0 / rho
                ux = jx * inv_rho
                uy = jy * inv_rho
                uz = jz * inv_rho
                if !(ux == ux) || !(uy == uy) || !(uz == uz)
                    ux = vel_in[x, y, z, b_idx, 1]
                    uy = vel_in[x, y, z, b_idx, 2]
                    uz = vel_in[x, y, z, b_idx, 3]
                end

                # ── Realizability bound on COVERED cells (2026-06-28) ──
                # The active-cell collision path projects (ρ,u) back onto the D3Q39
                # positive manifold (the u_cap_macro / rho_abs clamp, ~line 2085).
                # The is_covered macro path had NO such bound, so covered cells (dead,
                # overlapped by the finer level) drifted to unphysical ρ→3.1 / |u_rel|→1.08
                # past the manifold (instrumented 2026-06-28: L1 active ρ_max 2.77 / |u| 1.00
                # vs all-cell 3.13 / 1.08 — the excess was entirely on covered cells).
                # Harmless to the current solution (covered cells feed neither the visible
                # field nor the finer-level prolongation, which reads NON-covered parents),
                # but a robustness hazard if AMR topology changes and a covered cell becomes
                # active. Apply the SAME hard manifold bounds the active path uses (ρ≤2.80,
                # |u_rel|≤1.0), then keep momentum consistent. Matches the active clamp for a
                # real shock too (active cells under a shock get the identical bound).
                rho = min(rho, 2.80f0)
                inv_rho = 1.0f0 / rho
                u_rel_x = ux - frame_shift_x
                u_rel_y = uy - frame_shift_y
                u_rel_z = uz - frame_shift_z
                u_rel_mag = sqrt(u_rel_x*u_rel_x + u_rel_y*u_rel_y + u_rel_z*u_rel_z)
                if u_rel_mag > 1.0f0
                    sc = 1.0f0 / u_rel_mag
                    ux = frame_shift_x + u_rel_x * sc
                    uy = frame_shift_y + u_rel_y * sc
                    uz = frame_shift_z + u_rel_z * sc
                end
                jx = rho * ux
                jy = rho * uy
                jz = rho * uz

                vel_out[x, y, z, b_idx, 1] = ux
                vel_out[x, y, z, b_idx, 2] = uy
                vel_out[x, y, z, b_idx, 3] = uz
                rho_out[x, y, z, b_idx] = rho

                if Cv_thermal > 0.0f0
                    U_dot_j = frame_shift_x * jx + frame_shift_y * jy + frame_shift_z * jz
                    U_sq = frame_shift_x*frame_shift_x + frame_shift_y*frame_shift_y + frame_shift_z*frame_shift_z
                    E_kin_unshifted = E_kin_shifted - 2.0f0 * U_dot_j + U_sq * rho
                    kinetic_energy_out[x, y, z, b_idx] = E_kin_unshifted
                end

                for k in Int32(1):n_d
                    f_out[x, y, z, b_idx, k] = f_stored[k]
                    if store_post_collision == Int32(1)
                        f_post_collision[x, y, z, b_idx, k] = f_stored[k]
                    end
                end
            elseif is_obs
                vel_out[x, y, z, b_idx, 1] = 0.0f0
                vel_out[x, y, z, b_idx, 2] = 0.0f0
                vel_out[x, y, z, b_idx, 3] = 0.0f0
                rho_out[x, y, z, b_idx] = 1.0f0

                for k in Int32(1):n_d
                    f_coll = f_stored[opp_arr[k]]
                    f_out[x, y, z, b_idx, k] = f_coll
                    if store_post_collision == Int32(1)
                        f_post_collision[x, y, z, b_idx, k] = f_coll
                    end
                end
            else
                # ══════════════════════════════════════════════════════
                # STEP 2: Macroscopic quantities + sponge + wall model
                # ══════════════════════════════════════════════════════
                rho_raw = rho
                rho = max(rho_raw, 0.01f0)
                inv_rho = 1.0f0 / rho
                # Lab-frame velocity from shifted-lattice moments:
                # jx = Σ f·c'_x = ρ·u_lab (shifted c' gives lab-frame momentum)
                # Do NOT subtract shift here — ux_rel = ux - shift is done later
                # for the equilibrium only. Subtracting here would double-shift.
                ux = jx * inv_rho
                uy = jy * inv_rho
                uz = jz * inv_rho

                # ── NaN / unphysical density guard ──
                # Reset to freestream equilibrium ONLY for truly catastrophic states:
                # NaN or density < 0.01. Physical supersonic flows can have low density
                # from isentropic expansion (ρ_min ≈ 0.24 at Ma=1.1, 0.09 at Ma=2.5).
                # The previous threshold of 0.3 was clipping physical states and
                # causing mass-conservation violations that triggered cascading instability.
                is_nan = !(rho_raw == rho_raw) || !(ux == ux) || !(uy == uy) || !(uz == uz)
                is_unphysical = rho_raw < 0.01f0
                if is_nan || is_unphysical
                    rho = 1.0f0
                    ux  = u_inlet
                    uy  = 0.0f0
                    uz  = 0.0f0
                    inv_rho = 1.0f0
                    jx = u_inlet  # ρ*u for E_kin unshifting
                    jy = 0.0f0
                    jz = 0.0f0
                    # Phase 5: Use pre-computed numerical inlet equilibrium
                    E_kin_shifted = 0.0f0
                    for kk in Int32(1):n_d
                        f_stored[kk] = f_eq_inlet_arr[f_eq_inlet_offset + kk]
                        cxx_reset = Float32(Int32(cx_arr[kk]) - array_shift_x_i + frame_shift_x_i)
                        cyy_reset = Float32(Int32(cy_arr[kk]) - array_shift_y_i + frame_shift_y_i)
                        czz_reset = Float32(Int32(cz_arr[kk]) - array_shift_z_i + frame_shift_z_i)
                        E_kin_shifted += f_eq_inlet_arr[f_eq_inlet_offset + kk] *
                                         (cxx_reset*cxx_reset + cyy_reset*cyy_reset + czz_reset*czz_reset)
                    end
                end

                # ── Phase 4: Read temperature for two-way coupling ──
                # When thermal_coupling=1, T_local enters equilibrium via p = ρ·cs²·T
                # When thermal_coupling=0 or T≈1, this is bit-identical to isothermal behavior
                sp = sponge_arr[x, y, z, b_idx]

                T_local = 1.0f0
                T_feedback_max = T_inlet
                if thermal_coupling == Int32(1)
                    if Cv_thermal > 0.1f0
                        gamma_thermal = 1.0f0 + 1.0f0 / Cv_thermal
                        Ma_inlet_sq = (u_inlet * u_inlet) / max(cs2_val * T_inlet, 1.0f-6)
                        # Allow 2× the isentropic stagnation temperature as upper bound.
                        # Must match the thermal kernel's bound to avoid inconsistency.
                        T_feedback_max = max(
                            T_inlet,
                            T_inlet * (1.0f0 + 0.5f0 * (gamma_thermal - 1.0f0) * Ma_inlet_sq) * 2.0f0
                        )
                    end
                    T_center = temperature_arr[x, y, z, b_idx]
                    T_e = get_scalar_neighbor_fluid_amr(
                        temperature_arr, obstacle, x, y, z, b_idx, 1, 0, 0,
                        block_size, neighbor_table,
                        is_level_1,
                        parent_temperature, parent_temperature_old, parent_ptr,
                        parent_dim_x, parent_dim_y, parent_dim_z,
                        gx, gy, gz,
                        temporal_weight, use_temporal_interp
                    )
                    T_w = get_scalar_neighbor_fluid_amr(
                        temperature_arr, obstacle, x, y, z, b_idx, -1, 0, 0,
                        block_size, neighbor_table,
                        is_level_1,
                        parent_temperature, parent_temperature_old, parent_ptr,
                        parent_dim_x, parent_dim_y, parent_dim_z,
                        gx, gy, gz,
                        temporal_weight, use_temporal_interp
                    )
                    T_n = get_scalar_neighbor_fluid_amr(
                        temperature_arr, obstacle, x, y, z, b_idx, 0, 1, 0,
                        block_size, neighbor_table,
                        is_level_1,
                        parent_temperature, parent_temperature_old, parent_ptr,
                        parent_dim_x, parent_dim_y, parent_dim_z,
                        gx, gy, gz,
                        temporal_weight, use_temporal_interp
                    )
                    T_s = get_scalar_neighbor_fluid_amr(
                        temperature_arr, obstacle, x, y, z, b_idx, 0, -1, 0,
                        block_size, neighbor_table,
                        is_level_1,
                        parent_temperature, parent_temperature_old, parent_ptr,
                        parent_dim_x, parent_dim_y, parent_dim_z,
                        gx, gy, gz,
                        temporal_weight, use_temporal_interp
                    )
                    T_t = get_scalar_neighbor_fluid_amr(
                        temperature_arr, obstacle, x, y, z, b_idx, 0, 0, 1,
                        block_size, neighbor_table,
                        is_level_1,
                        parent_temperature, parent_temperature_old, parent_ptr,
                        parent_dim_x, parent_dim_y, parent_dim_z,
                        gx, gy, gz,
                        temporal_weight, use_temporal_interp
                    )
                    T_b = get_scalar_neighbor_fluid_amr(
                        temperature_arr, obstacle, x, y, z, b_idx, 0, 0, -1,
                        block_size, neighbor_table,
                        is_level_1,
                        parent_temperature, parent_temperature_old, parent_ptr,
                        parent_dim_x, parent_dim_y, parent_dim_z,
                        gx, gy, gz,
                        temporal_weight, use_temporal_interp
                    )
                    shifted_lattice = abs(frame_shift_x) + abs(frame_shift_y) + abs(frame_shift_z) > 1.0f-6
                    T_mean6 = (T_e + T_w + T_n + T_s + T_t + T_b) * (1.0f0 / 6.0f0)
                    if shifted_lattice
                        T_local = 0.5f0 * T_center + 0.5f0 * T_mean6
                    else
                        Tmin_nb = min(min(T_e, T_w), min(T_n, min(T_s, min(T_t, T_b))))
                        Tmax_nb = max(max(T_e, T_w), max(T_n, max(T_s, max(T_t, T_b))))
                        T_center_clipped = clamp(T_center, Tmin_nb, Tmax_nb)
                        T_local = 0.25f0 * T_center_clipped + 0.75f0 * T_mean6
                    end
                    # NaN and safety guard
                    if !(T_local == T_local) || T_local < 0.1f0
                        T_local = T_inlet
                    end
                    if is_nan
                        T_local = T_inlet
                    end
                    # AUDIT FIX (2026-06-03 Bug B part 2b): Riemann/shock-tube cases
                    # are near-stationary between waves (Ma≈0), so the Mach gate below
                    # would zero the feedback and let the per-side IC temperature drift
                    # back to T_inlet. When shock_tube_thermal==1, couple fully at all
                    # Mach and do NOT cap T at T_feedback_max (a shock tube can have
                    # T>1, e.g. Lax left T=7.9). Default path (else) is verbatim the
                    # prior code → all aero cases byte-identical.
                    if shock_tube_thermal == Int32(1)
                        # AUDIT FIX (2026-06-03): for a Riemann/shock-tube case the
                        # recovered T IS the physical per-side temperature. The legacy
                        # relaxation toward T_inlet has fixed point T_inlet=1.0, which
                        # pulls the Sod right state (0.8) UP toward 1.0. With the IC
                        # energy seeds now eos_R-consistent (so the closure is exact
                        # and the left state holds T=1 exactly), full coupling is
                        # stable AND lets the right state hold 0.8 — keep the recovered
                        # T as-is (no pull to T_inlet, no T_feedback_max cap so shock
                        # tubes with T>1 like Lax are admitted). (Earlier full-removal
                        # diverged ONLY because the IC over-seeded energy; that is now
                        # fixed at the three IC seed sites in main.jl.)
                        T_local = T_local
                    else
                        T_local = min(T_local, T_feedback_max)
                        # Scale feedback by local Mach, but keep the transonic regime
                        # thermodynamically coupled. The earlier Ma=1.2 gate made the
                        # Ma≈0.84 cylinder effectively athermal exactly where shoulder
                        # expansions and weak shocks need p=rho*cs²*T consistency.
                        # Keep very-low-Mach startup quiet, then ramp to full coupling
                        # before the freestream enters the transonic band.
                        u_mag_sq = ux*ux + uy*uy + uz*uz
                        Ma_local_sq = u_mag_sq / max(cs2_val * T_inlet, 1.0f-6)
                        Ma_local = sqrt(Ma_local_sq)
                        # Linear ramp: 0 below Ma=0.30, 1.0 at Ma=0.70.
                        feedback_scale = clamp((Ma_local - 0.30f0) / 0.40f0, 0.0f0, 1.0f0)
                        effective_relaxation = feedback_relaxation * feedback_scale
                        T_local = T_inlet + effective_relaxation * (T_local - T_inlet)
                    end
                end

                if sp > 0.0f0 && sponge_xu_sagaut_i == Int32(0)
                    rho_target = 1.0f0
                    ux_target  = u_inlet

                    rho = rho * (1.0f0 - sp) + rho_target * sp
                    ux  = ux  * (1.0f0 - sp) + ux_target  * sp
                    uy  = uy  * (1.0f0 - sp)
                    uz  = uz  * (1.0f0 - sp)

                    # Phase 4: blend temperature toward freestream in sponge zone.
                    # Required for compressible stability — without this, T drifts in the
                    # sponge, feeding back through p = ρ·cs²·T and generating unbounded
                    # pressure waves. The "cold stripe" cosmetic artifact is the lesser
                    # evil vs. solver divergence.
                    if thermal_coupling == Int32(1)
                        T_local = T_local * (1.0f0 - sp) + T_inlet * sp
                    end

                    if sponge_blend_distributions == Int32(1)
                        # Blend distributions toward inlet equilibrium (freestream conditions).
                        # f_eq_inlet_arr is pre-computed by NR on CPU at (ρ=1, u=u_inlet),
                        # which is the target state the sponge should drive toward.
                        for kk in Int32(1):n_d
                            f_eq_sponge = f_eq_inlet_arr[f_eq_inlet_offset + kk]
                            f_stored[kk] = f_stored[kk] * (1.0f0 - sp) + f_eq_sponge * sp
                        end

                        # CRITICAL: Recompute macroscopic quantities and E_kin from
                        # blended distributions. Without this, E_kin (computed from
                        # pre-sponge f) is inconsistent with rho/u (blended macroscopic).
                        # The mismatch T = (E_kin_old - rho_new*u_new² + 2Σg) / (2*rho*Cv)
                        # produces severe temperature artifacts in the sponge zone.
                        rho = 0.0f0
                        jx  = 0.0f0
                        jy  = 0.0f0
                        jz  = 0.0f0
                        E_kin_shifted = 0.0f0
                        for kk in Int32(1):n_d
                            fval = f_stored[kk]
                            rho += fval
                            cxx_sp = Float32(Int32(cx_arr[kk]) - array_shift_x_i + frame_shift_x_i)
                            cyy_sp = Float32(Int32(cy_arr[kk]) - array_shift_y_i + frame_shift_y_i)
                            czz_sp = Float32(Int32(cz_arr[kk]) - array_shift_z_i + frame_shift_z_i)
                            jx  += fval * cxx_sp
                            jy  += fval * cyy_sp
                            jz  += fval * czz_sp
                            E_kin_shifted += fval * (cxx_sp*cxx_sp + cyy_sp*cyy_sp + czz_sp*czz_sp)
                        end
                        rho = max(rho, 0.01f0)
                        inv_rho = 1.0f0 / rho
                        # Lab-frame velocity (no shift subtraction — same as primary)
                        ux = jx * inv_rho
                        uy = jy * inv_rho
                        uz = jz * inv_rho
                    end
                end

                # ── Wall model ──
                amr_eq_project = 0.0f0
                amr_mismatch_weight = 0.0f0
                if parent_halo_links > Int32(0)
                    # Parent-halo links are a directional subset of the cell, so
                    # they leave the recovered f with a small non-hydrodynamic
                    # mixed-grid residue. Clean it with a whole-cell projection
                    # toward the same rho/u equilibrium. That projection preserves
                    # conserved moments; the rejected pairwise h cleanup changed
                    # pair momentum and brightened the signed velocity rail.
                    halo_frac = Float32(parent_halo_links) / Float32(n_d)
                    # The visible "y" AMR-jump artifact on XZ slices is a
                    # physical z-neighbor ux jump. Treat transverse y/z pulls
                    # separately from x-facing pulls: pass125-128 showed that a
                    # soft x projection plus stronger signed y/z projection is
                    # the least invasive compromise seen in the A/B runs.
                    x_halo_count = parent_halo_x_neg + parent_halo_x_pos
                    y_halo_count_pure = parent_halo_y_neg_pure + parent_halo_y_pos_pure
                    z_halo_count_pure = parent_halo_z_neg_pure + parent_halo_z_pos_pure
                    # Diagonal D3Q39 links on an x-facing AMR frontier can also
                    # cross a y/z block face. Counting those as signed transverse
                    # halo links over-cleans the top/bottom corners of the x face
                    # and shows up as an apparent z-asymmetry. Keep the stronger
                    # transverse projection for pure y/z halo pulls only; x-corner
                    # links stay on the softer x-face path.
                    yz_halo_count = y_halo_count_pure + z_halo_count_pure
                    signed_halo_count = max(
                        abs(parent_halo_y_pos_pure - parent_halo_y_neg_pure),
                        abs(parent_halo_z_pos_pure - parent_halo_z_neg_pure)
                    )
                    x_halo_frac = Float32(x_halo_count) / Float32(n_d)
                    yz_halo_frac = Float32(yz_halo_count) / Float32(n_d)
                    signed_halo_frac = Float32(signed_halo_count) / Float32(n_d)
                    amr_x_project = min(0.10f0, 0.18f0 * x_halo_frac)
                    amr_yz_project = min(
                        0.55f0,
                        0.85f0 * yz_halo_frac + 0.40f0 * signed_halo_frac
                    )
                    if amr_halo_eq_projection_i == Int32(1)
                        amr_eq_project = max(amr_x_project, amr_yz_project)
                    end

                    if amr_halo_temporal_macro_closure_i == Int32(1)
                    # Keep conserved-macro closure at pass122 strength. The
                    # sign-resolved term is for h-mode/equilibrium cleanup; in
                    # pass123, also raising the macro closure amplified the
                    # x-face density split.
                    alpha_macro = min(0.14f0, 0.35f0 * halo_frac)

                    rho_prev_halo = max(rho_out[x, y, z, b_idx], 0.01f0)
                    ux_prev_halo = vel_in[x, y, z, b_idx, 1]
                    uy_prev_halo = vel_in[x, y, z, b_idx, 2]
                    uz_prev_halo = vel_in[x, y, z, b_idx, 3]

                    drho_halo = rho_prev_halo - rho
                    drho_cap = max(1.0f-5, 3.5f-4 * max(rho, 0.01f0))
                    rho = max(rho + alpha_macro * clamp(drho_halo, -drho_cap, drho_cap), 0.01f0)
                    inv_rho = 1.0f0 / rho

                    dux_halo = ux_prev_halo - ux
                    duy_halo = uy_prev_halo - uy
                    duz_halo = uz_prev_halo - uz
                    du_mag_halo = sqrt(dux_halo * dux_halo +
                                       duy_halo * duy_halo +
                                       duz_halo * duz_halo)
                    u_ref_halo = max(
                        max(
                            sqrt(ux * ux + uy * uy + uz * uz),
                            sqrt(ux_prev_halo * ux_prev_halo +
                                 uy_prev_halo * uy_prev_halo +
                                 uz_prev_halo * uz_prev_halo)
                        ),
                        0.1f0
                    )
                    du_cap_halo = max(1.0f-4, 2.0f-3 * u_ref_halo)
                    du_limiter_halo = du_mag_halo > du_cap_halo ?
                        du_cap_halo / max(du_mag_halo, 1.0f-12) : 1.0f0
                    ux += alpha_macro * du_limiter_halo * dux_halo
                    uy += alpha_macro * du_limiter_halo * duy_halo
                    uz += alpha_macro * du_limiter_halo * duz_halo

                    jx = rho * ux
                    jy = rho * uy
                    jz = rho * uz
                    end
                end

                if amr_halo_fullcell_recompose_i == Int32(1) &&
                   parent_halo_links > Int32(0) &&
                   is_level_1 == Int32(0)
                    rho_parent_fc, ux_parent_fc, uy_parent_fc, uz_parent_fc, parent_fc_valid =
                        sample_parent_macro_for_fine_cell(
                            parent_rho, parent_vel,
                            parent_rho_old, parent_vel_old,
                            parent_ptr,
                            parent_dim_x, parent_dim_y, parent_dim_z,
                            gx, gy, gz,
                            block_size,
                            temporal_weight,
                            use_temporal_interp
                        )
                    if parent_fc_valid
                        macro_alpha_fc = clamp(amr_halo_fullcell_macro_alpha, 0.0f0, 1.0f0)
                        pop_alpha_fc = clamp(amr_halo_fullcell_pop_alpha, 0.0f0, 0.95f0)

                        rho = max((1.0f0 - macro_alpha_fc) * rho + macro_alpha_fc * rho_parent_fc, 0.01f0)
                        ux = (1.0f0 - macro_alpha_fc) * ux + macro_alpha_fc * ux_parent_fc
                        uy = (1.0f0 - macro_alpha_fc) * uy + macro_alpha_fc * uy_parent_fc
                        uz = (1.0f0 - macro_alpha_fc) * uz + macro_alpha_fc * uz_parent_fc
                        inv_rho = 1.0f0 / rho
                        jx = rho * ux
                        jy = rho * uy
                        jz = rho * uz

                        # Reuse the existing whole-population projection below.
                        # The important distinction from the old partial cleanup is
                        # that the equilibrium is built from one coherent parent
                        # macro state, not from the mixed fine/parent streamed sum.
                        if amr_halo_eq_projection_i == Int32(1)
                            amr_eq_project = max(amr_eq_project, pop_alpha_fc)
                        end
                    end
                end

                if feather_active_i == Int32(1) || amr_regularization_active_i == Int32(1)
                    fw_u = feather_weight_in[x, y, z, b_idx]
                    if fw_u > 0.0f0
                        wall_dist_local = wall_dist_arr[x, y, z, b_idx]
                        far_from_wall = wall_dist_local > 6.0f0
                        wall_scale = far_from_wall ? 1.0f0 : 0.50f0
                        # Do not force fine-side macros toward parent values just
                        # because the cell is in the AMR feather band. That
                        # geometry-conditioned pull turns the refinement box into
                        # a body force and prints vorticity/schlieren rectangles.
                        # We still measure the parent/fine mismatch below so any
                        # optional dissipation can react to an actual mismatch,
                        # but the conservative distribution transfer remains the
                        # only source of macro coupling.

                        if is_level_1 == Int32(0)
                            rho_parent, ux_parent, uy_parent, uz_parent, parent_valid =
                                sample_parent_macro_for_fine_cell(
                                    parent_rho, parent_vel,
                                    parent_rho_old, parent_vel_old,
                                    parent_ptr,
                                    parent_dim_x, parent_dim_y, parent_dim_z,
                                    gx, gy, gz,
                                    block_size,
                                    temporal_weight,
                                    use_temporal_interp
                                )
                            if parent_valid
                                drho_parent = rho_parent - rho

                                du_parent_x = ux_parent - ux
                                du_parent_y = uy_parent - uy
                                du_parent_z = uz_parent - uz
                                du_parent_mag = sqrt(
                                    du_parent_x * du_parent_x +
                                    du_parent_y * du_parent_y +
                                    du_parent_z * du_parent_z
                                )
                                u_parent_ref = max(
                                    max(
                                        sqrt(ux * ux + uy * uy + uz * uz),
                                        sqrt(ux_parent * ux_parent + uy_parent * uy_parent + uz_parent * uz_parent)
                                    ),
                                    0.1f0
                                )
                                drho_rel_parent = abs(drho_parent) / max(rho, 0.01f0)
                                du_rel_parent = du_parent_mag / u_parent_ref
                                amr_mismatch_weight = max(
                                    amr_mismatch_weight,
                                    wall_scale * clamp(max(90.0f0 * drho_rel_parent, 18.0f0 * du_rel_parent), 0.0f0, 1.0f0)
                                )
                            end
                        end
                        jx = rho * ux
                        jy = rho * uy
                        jz = rho * uz
                    end
                end

                Fx_wall = 0.0f0
                Fy_wall = 0.0f0
                Fz_wall = 0.0f0

                if wall_model_active == Int32(1)
                    dist_wall = wall_dist_arr[x, y, z, b_idx]
                    if dist_wall > 0.0f0 && dist_wall < wall_model_max_distance
                        u_mag  = sqrt(ux*ux + uy*uy + uz*uz)
                        nu_visc = (tau_molecular - 0.5f0) * cs2_val  # nu = (tau - 0.5) * cs²

                        # ── SOTA B6: Compressible wall model (Van Driest) ──
                        # For compressible BLs, transform the velocity by the
                        # density profile before applying the log-law:
                        #   u_VD = u · √(ρ_local / ρ_wall)
                        # Then τ_w = ρ_wall · u_τ². Approximate ρ_wall using
                        # ideal-gas constant-pressure: ρ_w ≈ ρ · T/T_w.
                        # When wall_model_compressible_i == 0, vd_factor = 1
                        # and rho_w_eff = rho — recovers the legacy incompressible
                        # equilibrium wall model exactly.
                        T_loc = (wall_model_compressible_i == Int32(1)) ?
                                temperature_arr[x, y, z, b_idx] : 1.0f0
                        if !isfinite(T_loc) || T_loc < 0.1f0
                            T_loc = 1.0f0
                        end
                        rho_w_eff = (wall_model_compressible_i == Int32(1)) ?
                                    (rho * T_loc / max(wall_T_wall, 1.0f-3)) : rho
                        vd_factor = (wall_model_compressible_i == Int32(1)) ?
                                    sqrt(max(wall_T_wall / T_loc, 1.0f-6)) : 1.0f0
                        u_VD = u_mag * vd_factor

                        if u_VD > 1.0f-6 && nu_visc > 1.0f-10
                            # Initial guess via Spalding-like power law on u_VD
                            u_tau = u_VD * (nu_visc / (dist_wall * u_VD + 1.0f-10))^(1.0f0/7.0f0) * (2.0f0 * 8.3f0)^(-1.0f0/7.0f0)
                            u_tau = max(u_tau, 1.0f-6)

                            # ── SOTA B6-ext: Larsson 2010 PG residual ──
                            # u_plus_law_pg = (1/κ)·ln(y⁺) + B + (y/(ρ_w·u_τ²))·∂p/∂x_stream
                            # Computed once on the Spalding-seeded u_τ; the Picard
                            # update below then converges in one step in the
                            # equilibrium limit (∂p/∂x → 0) and within ~1% of
                            # the converged Larsson solution otherwise.
                            dpdx_stream = 0.0f0
                            if wall_model_larsson_pgr_i == Int32(1)
                                dpdx_stream = compute_streamwise_dpdx_inline(
                                    rho_out, x, y, z, b_idx,
                                    ux, uy, uz, u_mag,
                                    eos_R_flow, cs2_val,
                                    block_size, neighbor_table)
                            end

                            # Log-law iteration in Van-Driest-transformed velocity
                            y_p = u_tau * dist_wall / nu_visc
                            if y_p > 11.81f0
                                u_plus_law = (1.0f0 / KAPPA) * log(y_p) + 5.2f0
                                if wall_model_larsson_pgr_i == Int32(1)
                                    pg_corr = dist_wall * dpdx_stream /
                                              (max(rho_w_eff, 1.0f-3) *
                                               max(u_tau * u_tau, 1.0f-10))
                                    # Schur-bound: keep |Δu⁺| ≤ 0.5·u⁺ (Hickel 2012 §3.2)
                                    pg_cap = 0.5f0 * u_plus_law
                                    pg_corr = clamp(pg_corr, -pg_cap, pg_cap)
                                    u_plus_law = u_plus_law + pg_corr
                                end
                                if u_plus_law > 0.1f0
                                    u_tau = u_tau * ((u_VD / u_tau) / u_plus_law)
                                    u_tau = max(u_tau, 1.0f-6)
                                end
                            end

                            # Physical wall stress: use WALL density (Van Driest convention)
                            # When compressible mode is off, rho_w_eff == rho → legacy behavior.
                            tau_wall = rho_w_eff * u_tau * u_tau
                            tau_res  = rho * nu_visc * (u_mag / dist_wall)

                            if tau_wall > tau_res
                                force_mag = (tau_wall - tau_res) / dist_wall
                                # Bound the wall-model half-kick. At high Re the
                                # algebraic equilibrium wall model can otherwise
                                # inject an O(1) velocity change in one lattice
                                # step on cut cells with very small wall distance,
                                # which overwhelms Bouzidi and the AMR shell.
                                cap_du = min(0.02f0, 0.12f0 * u_mag)
                                force_mag = min(force_mag, 2.0f0 * rho * cap_du)
                                Fx_wall = -force_mag * ux / u_mag
                                Fy_wall = -force_mag * uy / u_mag
                                Fz_wall = -force_mag * uz / u_mag
                            end
                        end
                    end
                end

                # Transonic macroscopic realizability projection. The D3Q39
                # entropy/equilibrium solve only has a positive population
                # manifold for a bounded velocity and density range. If a curved
                # shoulder pocket leaves that manifold, merely clipping f_eq is
                # inconsistent: diagnostics and the next-stream macro state still
                # see the impossible rho/u. Project only pathological pockets and
                # let the residual guards below rebuild f close to the projected
                # equilibrium.
                macro_realizability_limited = false
                begin
                    u_macro_mag = sqrt(ux*ux + uy*uy + uz*uz)
                    u_trans_macro = sqrt(uy*uy + uz*uz)
                    # SHIFTED-LATTICE FIX (2026-06-27): the speed realizability cap (u_cap_macro,
                    # capped at 1.0) bounds the RELATIVE (shifted-frame) velocity — the lattice's
                    # positive-population manifold depends on |u - U_shift|, NOT |u_lab|. The Galilean
                    # shift exists so u_lab can be supersonic (1.2247) while u_rel stays small (0.2247).
                    # Checking u_macro_mag (lab) flagged the supersonic freestream as pathological and
                    # threw it down to u_cap~1.0 EVERY step → uniform collapse to the subsonic branch
                    # (the long-hunted freestream "decay"). Check/scale the RELATIVE magnitude instead.
                    # For shift=0, uxr==ux ⇒ byte-identical to the legacy path.
                    # LOCAL-SHIFT FIX (2026-06-27): use the per-block FRAME shift, not the global
                    # shift_x (=cx_arr[1]=array_shift, which is 0 in local-shift mode). In local mode a
                    # fast freestream block has frame_shift_x=1 but shift_x=0, so `ux - shift_x` = lab
                    # velocity 1.2247 → flagged pathological → throttled → freestream collapse (same
                    # signature as the original global-mode bug). frame_shift_x == shift_x in global/
                    # unshifted mode ⇒ byte-identical there; only the local-shift path changes.
                    uxr_mac = ux - frame_shift_x; uyr_mac = uy - frame_shift_y; uzr_mac = uz - frame_shift_z
                    u_macro_mag_rel = sqrt(uxr_mac*uxr_mac + uyr_mac*uyr_mac + uzr_mac*uzr_mac)
                    u_ref_macro = max(max(u_inlet, 0.1f0), 0.1f0)
                    u_cap_macro = min(1.00f0, max(0.80f0, 1.35f0 * u_ref_macro + 0.15f0))
                    u_trans_cap_macro = min(0.92f0, max(0.55f0, 1.15f0 * u_ref_macro + 0.12f0))
                    wall_dist_macro = wall_dist_arr[x, y, z, b_idx]
                    far_bulk_macro = (wall_dist_macro != wall_dist_macro || wall_dist_macro > 16.0f0) && sp < 0.03f0
                    if far_bulk_macro
                        # NOTE (2026-06-28): raising this far-bulk cap 0.78→0.95 was TESTED as a
                        # candidate fix for the Ma0.9 AMR box stagnation and REFUTED — the box
                        # u_mean stayed ~0.32 / rho ~1.70 (the cap firing at L2 u_max=0.78 was a
                        # SYMPTOM of the seam, not the momentum sink), and it raised the transient
                        # rho_max to ~4.06. Reverted. The real sink is a spurious mass/energy
                        # injection in the AMR coupling (box pressure exceeds freestream stagnation).
                        u_cap_macro = min(u_cap_macro, min(0.78f0, max(0.48f0, 1.12f0 * u_ref_macro + 0.08f0)))
                        u_trans_cap_macro = min(u_trans_cap_macro, min(0.46f0, max(0.18f0, 0.68f0 * u_ref_macro + 0.04f0)))
                    end
                    speed_pathological = u_macro_mag_rel > u_cap_macro   # SHIFTED-LATTICE FIX: relative magnitude
                    trans_pathological = u_trans_macro > u_trans_cap_macro
                    # AUDIT FIX (2026-06-03): configurable bounds (legacy
                    # 0.42/2.10/0.35/2.40 by default). Canonical low-density
                    # cases (Sod ρ_R=0.125) set a lower floor so the clamp does
                    # not destroy the physical low-density state.
                    rho_floor_macro = far_bulk_macro ? macro_bounds.rho_floor_far : macro_bounds.rho_floor_near
                    rho_ceil_macro = far_bulk_macro ? macro_bounds.rho_ceil_far : macro_bounds.rho_ceil_near
                    # AUDIT FIX (2026-06-03): only apply the aggressive isentropic
                    # EOS floor (which raises the floor to [0.46,0.58]) when the
                    # CONFIGURED floor is in the aero range. A low configured floor
                    # (e.g. 0.05 for Sod/Lax/Riemann) signals legitimate low-density
                    # physics → skip the EOS floor so ρ_R=0.125 survives.
                    if thermal_coupling == Int32(0) && eos_R_flow > 0.0f0 && far_bulk_macro &&
                       macro_bounds.rho_floor_far >= 0.30f0
                        # Algebraic EOS consistency guard. The fixed 0.42 far-bulk
                        # floor is only a positivity fallback; in a Ma~0.6-0.9
                        # shoulder/wake-edge pocket the isentropic density implied
                        # by the local speed is much higher. Use that estimate only
                        # when a low-density, high-speed pocket is about to leave
                        # the D3Q39 realizability basin. This is intentionally
                        # narrower than changing the effective T everywhere.
                        gamma_macro = 1.4f0
                        gm1_half_macro = 0.5f0 * (gamma_macro - 1.0f0)
                        T_ref_macro = max(T_inlet, 0.1f0)
                        a2_ref_macro = max(cs2_val * T_ref_macro, 1.0f-6)
                        ma_inf_sq_macro = (u_inlet * u_inlet) / a2_ref_macro
                        ma_loc_sq_macro = min((u_macro_mag * u_macro_mag) / a2_ref_macro, 6.25f0)
                        T_ratio_macro =
                            (1.0f0 + gm1_half_macro * ma_inf_sq_macro) /
                            max(1.0f0 + gm1_half_macro * ma_loc_sq_macro, 1.0f-6)
                        rho_iso_macro = exp((1.0f0 / (gamma_macro - 1.0f0)) *
                                            log(max(T_ratio_macro, 1.0f-6)))
                        rho_floor_eos = clamp(0.58f0 * rho_iso_macro, 0.46f0, 0.58f0)
                        if rho < rho_floor_eos &&
                           (u_macro_mag > 0.44f0 || u_trans_macro > 0.20f0)
                            rho_floor_macro = max(rho_floor_macro, rho_floor_eos)
                        end
                    end
                    # AUDIT FIX (2026-06-03): the absolute hard bounds (0.30/2.80) and
                    # the speed-pathological band (0.45/2.20) were hard-coded and fatal
                    # to legitimately-low-density flow (Sod ρ_R=0.125). Keep the EXACT
                    # legacy literals when the configured floor is the aero default
                    # (≥0.30 ⇒ byte-identical for all aero cases), and only RELAX the
                    # lower bounds below the configured floor when a low floor signals
                    # a canonical low-density case. Upper bounds track the ceiling the
                    # same way.
                    low_density_case = macro_bounds.rho_floor_far < 0.30f0
                    rho_abs_lo = low_density_case ? min(0.30f0, 0.5f0 * macro_bounds.rho_floor_far) : 0.30f0
                    rho_speed_lo = low_density_case ? min(0.45f0, macro_bounds.rho_floor_far) : 0.45f0
                    rho_abs_hi = macro_bounds.rho_ceil_far > 2.10f0 ? max(2.80f0, 1.33f0 * macro_bounds.rho_ceil_far) : 2.80f0
                    rho_speed_hi = macro_bounds.rho_ceil_far > 2.10f0 ? max(2.20f0, 1.05f0 * macro_bounds.rho_ceil_far) : 2.20f0
                    density_pathological =
                        rho < rho_abs_lo || rho > rho_abs_hi ||
                        (far_bulk_macro && (rho < rho_floor_macro || rho > rho_ceil_macro)) ||
                        ((rho < rho_speed_lo || rho > rho_speed_hi) &&
                         (speed_pathological || trans_pathological))

                    if density_pathological
                        rho = clamp(rho, rho_floor_macro, rho_ceil_macro)
                        inv_rho = 1.0f0 / max(rho, 0.01f0)
                        macro_realizability_limited = true
                    end

                    if trans_pathological && u_trans_macro > 1.0f-8
                        trans_scale = u_trans_cap_macro / max(u_trans_macro, 1.0f-8)
                        uy *= trans_scale
                        uz *= trans_scale
                        u_macro_mag = sqrt(ux*ux + uy*uy + uz*uz)
                        uyr_mac = uy - frame_shift_y; uzr_mac = uz - frame_shift_z   # LOCAL-SHIFT FIX: per-block frame
                        u_macro_mag_rel = sqrt(uxr_mac*uxr_mac + uyr_mac*uyr_mac + uzr_mac*uzr_mac)
                        macro_realizability_limited = true
                    end

                    if speed_pathological && u_macro_mag_rel > 1.0f-8
                        # SHIFTED-LATTICE FIX: scale the RELATIVE velocity to the cap, then add the
                        # shift back, so the lab streamwise velocity is throttled only by the relative
                        # excess. For shift=0 this reduces to ux *= speed_scale (byte-identical).
                        speed_scale = u_cap_macro / max(u_macro_mag_rel, 1.0f-8)
                        ux = frame_shift_x + uxr_mac * speed_scale   # LOCAL-SHIFT FIX: add back the per-block frame
                        uy = frame_shift_y + uyr_mac * speed_scale
                        uz = frame_shift_z + uzr_mac * speed_scale
                        macro_realizability_limited = true
                    end

                    if macro_realizability_limited
                        jx = rho * ux
                        jy = rho * uy
                        jz = rho * uz
                    end
                end

                if thermal_coupling == Int32(0) && eos_R_flow > 0.0f0
                    T_local = transonic_algebraic_temperature_d3q39(
                        rho, ux, uy, uz,
                        u_inlet, cs2_val, T_inlet,
                        eos_R_flow, thermal_coupling, sp
                    )
                end

                ux_eq = ux + 0.5f0 * Fx_wall * inv_rho
                uy_eq = uy + 0.5f0 * Fy_wall * inv_rho
                uz_eq = uz + 0.5f0 * Fz_wall * inv_rho

                # Shifted lattice: relative velocity for equilibrium (u_rel = u - U)
                # When shift=0, these equal ux_eq, uy_eq, uz_eq (no change).
                ux_rel = ux_eq - frame_shift_x
                uy_rel = uy_eq - frame_shift_y
                uz_rel = uz_eq - frame_shift_z
                eq_velocity_limited = false
                u_rel_eq_mag = sqrt(ux_rel*ux_rel + uy_rel*uy_rel + uz_rel*uz_rel)
                u_rel_cap = 0.82f0
                if u_rel_eq_mag > u_rel_cap
                    rel_scale = u_rel_cap / max(u_rel_eq_mag, 1.0f-6)
                    ux_rel *= rel_scale
                    uy_rel *= rel_scale
                    uz_rel *= rel_scale
                    ux_eq = ux_rel + frame_shift_x
                    uy_eq = uy_rel + frame_shift_y
                    uz_eq = uz_rel + frame_shift_z
                    eq_velocity_limited = true
                end

                vel_out[x, y, z, b_idx, 1] = ux
                vel_out[x, y, z, b_idx, 2] = uy
                vel_out[x, y, z, b_idx, 3] = uz
                rho_out[x, y, z, b_idx] = rho

                # ── Phase 4b: Unshift E_kin and store for Frapolli T recovery ──
                # E_kin_unshifted = Σf·|c_unshifted|² = Σf·|c'-U|²
                #   = Σf·|c'|² - 2U·Σ(f·c') + |U|²·Σf
                #   = E_kin_shifted - 2U·j_shifted + |U|²·ρ
                # where j_shifted = (jx,jy,jz) = ρ·u_lab (in shifted frame moment)
                if Cv_thermal > 0.0f0
                    U_dot_j = frame_shift_x * jx + frame_shift_y * jy + frame_shift_z * jz
                    U_sq = frame_shift_x*frame_shift_x + frame_shift_y*frame_shift_y + frame_shift_z*frame_shift_z
                    E_kin_unshifted = E_kin_shifted - 2.0f0 * U_dot_j + U_sq * rho
                    kinetic_energy_out[x, y, z, b_idx] = E_kin_unshifted
                end

                # ══════════════════════════════════════════════════════
                # STEP 3a: Compute f_eq EARLY (Phase 1.1 paper-quality)
                # ══════════════════════════════════════════════════════
                # Latt 2020 Knudsen-adaptive τ needs ε = ‖f − f_eq‖ / ‖f_eq‖ as
                # its input — that requires f_eq available BEFORE τ_eff. We
                # compute it here so the nu_eddy block below can use the real
                # Latt ε, and the later collision STEP 5 onward re-uses this
                # same f_eq (no duplication). This replaces the old Π_neq-based
                # Knudsen-τ proxy with the canonical Latt formulation.
                #
                # Implementation is identical to the later "STEP 4" — the 13-moment
                # Newton-Raphson run once per cell per timestep, amortised across
                # positivity clamp, Knudsen-τ, mirror decomposition, and collision.
                # C4 perf (2026-05-28): max_iter 200 → 30. The 3-group coordinate
                # descent with OMEGA=1.5 over-relaxation converges in 5-15 sweeps
                # for typical (ρ, u, T) cells; the early-exit TOL_SQ check at the
                # bottom of each sweep terminates well-conditioned cells fast.
                # 200 was a paranoid worst-case bound — on 2.5M cells per step
                # those non-converging laggards dominated wall-time at ~700 GFLOPs
                # per step (15-30% of measured 15s/step). Reducing to 30 leaves
                # plenty of headroom for slow-converging cells while bounding the
                # worst case to ~6.7× faster. Frapolli/Karlin papers use 5-10.
                # B4 (2026-05-28) — Newton warm-start.
                # 2026-05-29 SMART B4: warm-start re-enabled. The previous
                # diagnostic confirmed the stale warm-start at transient wave
                # fronts caused the L1↔L2 stripe. The fix is NOT to disable B4
                # (that degraded perf to ~57s/step) but a residual gate inside
                # solve_13mom_lambdas_d3q39: the warm-start is validated against
                # the target moments once per cell; if too stale it falls back
                # to the polynomial init. Steady cells keep the B4 perf win;
                # wave-front cells get the known-good polynomial basin.
                lambda_warm = (lambda_cache_arr[x, y, z, b_idx, Int32(1)],
                               lambda_cache_arr[x, y, z, b_idx, Int32(2)],
                               lambda_cache_arr[x, y, z, b_idx, Int32(3)],
                               lambda_cache_arr[x, y, z, b_idx, Int32(4)],
                               lambda_cache_arr[x, y, z, b_idx, Int32(5)],
                               lambda_cache_arr[x, y, z, b_idx, Int32(6)],
                               lambda_cache_arr[x, y, z, b_idx, Int32(7)],
                               lambda_cache_arr[x, y, z, b_idx, Int32(8)],
                               lambda_cache_arr[x, y, z, b_idx, Int32(9)],
                               lambda_cache_arr[x, y, z, b_idx, Int32(10)],
                               lambda_cache_arr[x, y, z, b_idx, Int32(11)],
                               lambda_cache_arr[x, y, z, b_idx, Int32(12)],
                               lambda_cache_arr[x, y, z, b_idx, Int32(13)])
                # ADAPTIVE OPERATOR (2026-05-30): the in-kernel branch version was
                # tested and REVERTED — warp divergence killed the speedup (mixed
                # smooth/hard cells in one warp → no saving) and the poly path drifted
                # (missing the Newton mass-correction). The real win needs a SEPARATE
                # smooth-cell kernel (no divergence, high occupancy). adaptive_ma_thresh
                # plumbing kept (currently unused). See project_next_perf_work memory.
                lambda_eq = solve_13mom_lambdas_d3q39(
                    rho, ux_rel, uy_rel, uz_rel,
                    cx_arr, cy_arr, cz_arr, w_arr,
                    cs2_val, Int32(200), T_local,
                    shift_x, shift_y, shift_z,
                    1.0f0,                  # tol_multiplier
                    eos_R_flow,             # SOTA A3 — thermodynamic R
                    lambda_warm             # B4 — warm-start init
                )
                # B4 — write back the converged λ for next step's warm-start.
                lambda_cache_arr[x, y, z, b_idx, Int32(1)]  = lambda_eq[1]
                lambda_cache_arr[x, y, z, b_idx, Int32(2)]  = lambda_eq[2]
                lambda_cache_arr[x, y, z, b_idx, Int32(3)]  = lambda_eq[3]
                lambda_cache_arr[x, y, z, b_idx, Int32(4)]  = lambda_eq[4]
                lambda_cache_arr[x, y, z, b_idx, Int32(5)]  = lambda_eq[5]
                lambda_cache_arr[x, y, z, b_idx, Int32(6)]  = lambda_eq[6]
                lambda_cache_arr[x, y, z, b_idx, Int32(7)]  = lambda_eq[7]
                lambda_cache_arr[x, y, z, b_idx, Int32(8)]  = lambda_eq[8]
                lambda_cache_arr[x, y, z, b_idx, Int32(9)]  = lambda_eq[9]
                lambda_cache_arr[x, y, z, b_idx, Int32(10)] = lambda_eq[10]
                lambda_cache_arr[x, y, z, b_idx, Int32(11)] = lambda_eq[11]
                lambda_cache_arr[x, y, z, b_idx, Int32(12)] = lambda_eq[12]
                lambda_cache_arr[x, y, z, b_idx, Int32(13)] = lambda_eq[13]
                mass_corr_eq = compute_13mom_mass_correction_d3q39(
                    rho, lambda_eq,
                    cx_arr, cy_arr, cz_arr, w_arr,
                    cs2_val, shift_x, shift_y, shift_z
                )
                f_eq = MVector{39, Float32}(undef)
                for k in Int32(1):n_d
                    cx_eq = Float32(cx_arr[k]) - shift_x
                    cy_eq = Float32(cy_arr[k]) - shift_y
                    cz_eq = Float32(cz_arr[k]) - shift_z
                    f_eq[k] = mass_corr_eq * eval_13mom_feq_single(
                        rho, lambda_eq, cx_eq, cy_eq, cz_eq, w_arr[k], cs2_val
                    )
                end

                if amr_halo_constrained_completion_i == Int32(1) &&
                   parent_halo_links > Int32(0) &&
                   is_level_1 == Int32(0) &&
                   !is_obs && !is_covered
                    # Regularized AMR boundary closure. A fine boundary cell is
                    # assembled from local fine streams plus parent-supplied
                    # ghost links; leaving that mixed population untouched keeps
                    # high-order kinetic content that has no single-grid
                    # Chapman-Enskog interpretation. Reconstruct the whole cell
                    # from the coherent conserved state plus the projected
                    # second-order non-equilibrium stress. This is the LBM AMR
                    # analogue of moment-consistent prolongation/restriction:
                    # mass and momentum are preserved, stress is retained, and
                    # non-hydrodynamic grid-parity modes are removed at the
                    # place where they enter the refined level.
                    rho_target_amr = rho
                    jx_target_amr = jx
                    jy_target_amr = jy
                    jz_target_amr = jz

                    pxx_amr = 0.0f0
                    pyy_amr = 0.0f0
                    pzz_amr = 0.0f0
                    pxy_amr = 0.0f0
                    pxz_amr = 0.0f0
                    pyz_amr = 0.0f0

                    # Use the local post-streaming non-equilibrium tensor as
                    # the hydrodynamic stress carrier. A finite-difference HRR
                    # stress at a coarse/fine cell samples a composite stencil
                    # with mixed phase and resolution; in the AMR-only uniform
                    # flow case that aliases the transfer error into vertical
                    # bands. The regularized LBM transfer used by Palabos/OpenLB
                    # style coupling is moment-space: keep rho and rho*u exactly,
                    # project the measured second moment, discard higher kinetic
                    # content.
                    for kk in Int32(1):n_d
                        df_amr = f_stored[kk] - f_eq[kk]
                        cxr = Float32(cx_arr[kk]) - shift_x
                        cyr = Float32(cy_arr[kk]) - shift_y
                        czr = Float32(cz_arr[kk]) - shift_z
                        pxx_amr += cxr * cxr * df_amr
                        pyy_amr += cyr * cyr * df_amr
                        pzz_amr += czr * czr * df_amr
                        pxy_amr += cxr * cyr * df_amr
                        pxz_amr += cxr * czr * df_amr
                        pyz_amr += cyr * czr * df_amr
                    end

                    stress_theta_amr = 1.0f0
                    for kk in Int32(1):n_d
                        cxr = Float32(cx_arr[kk]) - shift_x
                        cyr = Float32(cy_arr[kk]) - shift_y
                        czr = Float32(cz_arr[kk]) - shift_z
                        s_amr = project_pi_neq_stress_mode(
                            Float32(w_arr[kk]),
                            cxr, cyr, czr,
                            cs2_val,
                            pxx_amr, pyy_amr, pzz_amr,
                            pxy_amr, pxz_amr, pyz_amr
                        )
                        if f_eq[kk] + s_amr < F_FLOOR
                            stress_theta_amr = min(
                                stress_theta_amr,
                                max((f_eq[kk] - F_FLOOR) / max(-s_amr, 1.0f-20), 0.0f0)
                            )
                        end
                    end
                    stress_theta_amr = clamp(stress_theta_amr, 0.0f0, 1.0f0)

                    for kk in Int32(1):n_d
                        cxr = Float32(cx_arr[kk]) - shift_x
                        cyr = Float32(cy_arr[kk]) - shift_y
                        czr = Float32(cz_arr[kk]) - shift_z
                        s_amr = project_pi_neq_stress_mode(
                            Float32(w_arr[kk]),
                            cxr, cyr, czr,
                            cs2_val,
                            pxx_amr, pyy_amr, pzz_amr,
                            pxy_amr, pxz_amr, pyz_amr
                        )
                        f_stored[kk] = max(f_eq[kk] + stress_theta_amr * s_amr, F_FLOOR)
                    end

                    # The stress projection has zero zeroth and first moments on
                    # the lattice quadrature. A single scalar positivity limiter
                    # therefore keeps the reconstructed cell conservative without
                    # an additional affine correction sweep.
                    rho = rho_target_amr
                    jx = jx_target_amr
                    jy = jy_target_amr
                    jz = jz_target_amr
                    inv_rho = 1.0f0 / max(rho, 0.01f0)
                    ux = jx * inv_rho
                    uy = jy * inv_rho
                    uz = jz * inv_rho
                    E_kin_shifted = 0.0f0
                    for kk in Int32(1):n_d
                        cxx = Float32(Int32(cx_arr[kk]) - array_shift_x_i + frame_shift_x_i)
                        cyy = Float32(Int32(cy_arr[kk]) - array_shift_y_i + frame_shift_y_i)
                        czz = Float32(Int32(cz_arr[kk]) - array_shift_z_i + frame_shift_z_i)
                        E_kin_shifted += f_stored[kk] * (cxx*cxx + cyy*cyy + czz*czz)
                    end
                end

                if amr_eq_project > 0.0f0
                    keep_f = 1.0f0 - clamp(amr_eq_project, 0.0f0, 0.95f0)
                    proj_f = 1.0f0 - keep_f

                    rho_proj = 0.0f0
                    jx_proj = 0.0f0
                    jy_proj = 0.0f0
                    jz_proj = 0.0f0
                    E_kin_shifted_proj = 0.0f0

                    for kk in Int32(1):n_d
                        f_blend = keep_f * f_stored[kk] + proj_f * f_eq[kk]
                        f_stored[kk] = max(f_blend, F_FLOOR)

                        cxx = Float32(Int32(cx_arr[kk]) - array_shift_x_i + frame_shift_x_i)
                        cyy = Float32(Int32(cy_arr[kk]) - array_shift_y_i + frame_shift_y_i)
                        czz = Float32(Int32(cz_arr[kk]) - array_shift_z_i + frame_shift_z_i)

                        rho_proj += f_stored[kk]
                        jx_proj += f_stored[kk] * cxx
                        jy_proj += f_stored[kk] * cyy
                        jz_proj += f_stored[kk] * czz
                        E_kin_shifted_proj += f_stored[kk] * (cxx*cxx + cyy*cyy + czz*czz)
                    end

                    rho_proj = 0.0f0
                    jx_proj = 0.0f0
                    jy_proj = 0.0f0
                    jz_proj = 0.0f0
                    E_kin_shifted_proj = 0.0f0
                    for kk in Int32(1):n_d
                        cxx = Float32(Int32(cx_arr[kk]) - array_shift_x_i + frame_shift_x_i)
                        cyy = Float32(Int32(cy_arr[kk]) - array_shift_y_i + frame_shift_y_i)
                        czz = Float32(Int32(cz_arr[kk]) - array_shift_z_i + frame_shift_z_i)
                        rho_proj += f_stored[kk]
                        jx_proj += f_stored[kk] * cxx
                        jy_proj += f_stored[kk] * cyy
                        jz_proj += f_stored[kk] * czz
                        E_kin_shifted_proj += f_stored[kk] * (cxx*cxx + cyy*cyy + czz*czz)
                    end

                    rho = max(rho_proj, 0.01f0)
                    inv_rho = 1.0f0 / rho
                    jx = jx_proj
                    jy = jy_proj
                    jz = jz_proj
                    ux = jx * inv_rho
                    uy = jy * inv_rho
                    uz = jz * inv_rho
                    E_kin_shifted = E_kin_shifted_proj

                    ux_eq = ux + 0.5f0 * Fx_wall * inv_rho
                    uy_eq = uy + 0.5f0 * Fy_wall * inv_rho
                    uz_eq = uz + 0.5f0 * Fz_wall * inv_rho
                    ux_rel = ux_eq - frame_shift_x
                    uy_rel = uy_eq - frame_shift_y
                    uz_rel = uz_eq - frame_shift_z

                    # C4 perf (2026-05-28): see note at STEP 3a — max_iter 200 → 30
                    lambda_eq = solve_13mom_lambdas_d3q39(
                        rho, ux_rel, uy_rel, uz_rel,
                        cx_arr, cy_arr, cz_arr, w_arr,
                        cs2_val, Int32(200), T_local,
                        shift_x, shift_y, shift_z,
                        1.0f0,              # tol_multiplier
                        eos_R_flow          # SOTA A3 — thermodynamic R
                    )
                    mass_corr_eq = compute_13mom_mass_correction_d3q39(
                        rho, lambda_eq,
                        cx_arr, cy_arr, cz_arr, w_arr,
                        cs2_val, shift_x, shift_y, shift_z
                    )
                    for kk in Int32(1):n_d
                        cx_eq = Float32(cx_arr[kk]) - shift_x
                        cy_eq = Float32(cy_arr[kk]) - shift_y
                        cz_eq = Float32(cz_arr[kk]) - shift_z
                        f_eq[kk] = mass_corr_eq * eval_13mom_feq_single(
                            rho, lambda_eq, cx_eq, cy_eq, cz_eq, w_arr[kk], cs2_val
                        )
                    end

                    vel_out[x, y, z, b_idx, 1] = ux
                    vel_out[x, y, z, b_idx, 2] = uy
                    vel_out[x, y, z, b_idx, 3] = uz
                    rho_out[x, y, z, b_idx] = rho

                    if Cv_thermal > 0.0f0
                        U_dot_j = frame_shift_x * jx + frame_shift_y * jy + frame_shift_z * jz
                        U_sq = frame_shift_x*frame_shift_x + frame_shift_y*frame_shift_y + frame_shift_z*frame_shift_z
                        kinetic_energy_out[x, y, z, b_idx] =
                            E_kin_shifted - 2.0f0 * U_dot_j + U_sq * rho
                    end
                end

                # STEP 3: WALE SGS → effective relaxation rate β
                # ══════════════════════════════════════════════════════
                g11, g12, g13, g21, g22, g23, g31, g32, g33 = compute_velocity_gradients_amr_fluid(
                    vel_in, obstacle, x, y, z, b_idx, block_size, neighbor_table,
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

                # Keep additive AMR stabilizers level-consistent in lattice units.
                # A constant lattice ν on fine levels imprints visible plateaus when
                # the interface band is wide, so scale the viscous feather with the
                # same level factor as the background/floor terms. AMR-specific
                # high-order cleanup is handled below through h-mode damping.
                nu_sgs_background_local = nu_sgs_background * amr_sgs_scale
                nu_sgs_floor_local = nu_sgs_floor * amr_sgs_scale
                feather_nu_max_local = feather_nu_max * amr_sgs_scale

                # ── SOTA #3: Ducros shock-sensor dissipation (Pass 197b) ──
                # Normalized density-gradient sensor; smoothstep boost raises the
                # ν_sgs floor only at strong compressive jumps. Smooth flow → 0.
                shock_boost = 0.0f0
                if shock_sensor_active == Int32(1)
                    rho_xp = get_scalar_neighbor_fluid_amr(
                        rho_out, obstacle, x, y, z, b_idx, 1, 0, 0,
                        block_size, neighbor_table, is_level_1,
                        parent_rho, parent_rho_old, parent_ptr,
                        parent_dim_x, parent_dim_y, parent_dim_z,
                        gx, gy, gz, temporal_weight, use_temporal_interp)
                    rho_xm = get_scalar_neighbor_fluid_amr(
                        rho_out, obstacle, x, y, z, b_idx, -1, 0, 0,
                        block_size, neighbor_table, is_level_1,
                        parent_rho, parent_rho_old, parent_ptr,
                        parent_dim_x, parent_dim_y, parent_dim_z,
                        gx, gy, gz, temporal_weight, use_temporal_interp)
                    rho_yp = get_scalar_neighbor_fluid_amr(
                        rho_out, obstacle, x, y, z, b_idx, 0, 1, 0,
                        block_size, neighbor_table, is_level_1,
                        parent_rho, parent_rho_old, parent_ptr,
                        parent_dim_x, parent_dim_y, parent_dim_z,
                        gx, gy, gz, temporal_weight, use_temporal_interp)
                    rho_ym = get_scalar_neighbor_fluid_amr(
                        rho_out, obstacle, x, y, z, b_idx, 0, -1, 0,
                        block_size, neighbor_table, is_level_1,
                        parent_rho, parent_rho_old, parent_ptr,
                        parent_dim_x, parent_dim_y, parent_dim_z,
                        gx, gy, gz, temporal_weight, use_temporal_interp)
                    rho_zp = get_scalar_neighbor_fluid_amr(
                        rho_out, obstacle, x, y, z, b_idx, 0, 0, 1,
                        block_size, neighbor_table, is_level_1,
                        parent_rho, parent_rho_old, parent_ptr,
                        parent_dim_x, parent_dim_y, parent_dim_z,
                        gx, gy, gz, temporal_weight, use_temporal_interp)
                    rho_zm = get_scalar_neighbor_fluid_amr(
                        rho_out, obstacle, x, y, z, b_idx, 0, 0, -1,
                        block_size, neighbor_table, is_level_1,
                        parent_rho, parent_rho_old, parent_ptr,
                        parent_dim_x, parent_dim_y, parent_dim_z,
                        gx, gy, gz, temporal_weight, use_temporal_interp)
                    drdx = 0.5f0 * (rho_xp - rho_xm)
                    drdy = 0.5f0 * (rho_yp - rho_ym)
                    drdz = 0.5f0 * (rho_zp - rho_zm)
                    grad_rho = sqrt(drdx*drdx + drdy*drdy + drdz*drdz)
                    s_shock = grad_rho / max(rho, 1.0f-6)   # dx = 1 (lattice)
                    # smoothstep(s; lo, hi)
                    t_sh = clamp((s_shock - shock_thresh_lo) /
                                 max(shock_thresh_hi - shock_thresh_lo, 1.0f-6),
                                 0.0f0, 1.0f0)
                    boost = t_sh * t_sh * (3.0f0 - 2.0f0 * t_sh)
                    shock_boost = boost
                    nu_sgs_floor_local = max(nu_sgs_floor_local,
                                             boost * shock_nu_max * amr_sgs_scale)
                end

                nu_eddy = 0.0f0
                if OP1 > 1.0f-12
                    OP1_32 = OP1 * sqrt(OP1)
                    OP2_52 = OP2 * OP2 * sqrt(max(OP2, 1.0f-12))
                    denom = OP2_52 + OP1 * sqrt(sqrt(max(OP1, 1.0f-12)))
                    if denom > 1.0f-12
                        nu_eddy = (c_wale * c_wale) * OP1_32 / denom
                    end
                end

                # Phase B: variable ν_sgs from local Π_neq (per-cell adaptive dissipation).
                # When enabled, reads Π_neq^post from pi_neq_in (populated by refresh_pi_neq!
                # after PREVIOUS timestep), applies τ-correction, and computes Smagorinsky-like
                # proxy ν_cell = C_π·|Π_neq^pre|/(ρ·cs²).  Takes max against WALE and floor.
                if variable_nu_sgs_i == Int32(1)
                    # ── SOTA B2: Dynamic Smagorinsky branch ──────────────
                    # When dynamic_sgs_active is on, compute c_d via the Germano-
                    # Lilly procedure inline (per-cell) and use the standard
                    # Smagorinsky form ν_sgs = c_d · Δ² · |S|. In lattice units
                    # Δ = 1, so this reduces to ν_sgs = c_d · |S|. Replaces the
                    # hand-tuned `c_pi_stress` formulation with a self-calibrating
                    # coefficient that auto-zeros in freestream.
                    if dynamic_sgs_active == Int32(1)
                        c_d_dyn = compute_dynamic_cs2_inline(
                            vel_in, obstacle, neighbor_table,
                            x, y, z, b_idx, block_size,
                            ux, uy, uz,
                            g11, g22, g33, S12, S13, S23,
                            1.0f0          # Δ = 1 in lattice units
                        )
                        S_mag_loc = sqrt(max(2.0f0 * (g11*g11 + g22*g22 + g33*g33) +
                                             4.0f0 * (S12*S12 + S13*S13 + S23*S23),
                                             1.0f-30))
                        # Smagorinsky-like ν_sgs = Cs²·Δ²·|S| (Δ = 1 lattice unit)
                        # AMR uniform-ν_phys fix v2 (2026-05-28): apply amr_sgs_scale
                        # to the stress-based contribution too. v1 only scaled the
                        # floor (nu_sgs_floor_local) but the stress-based term
                        # `c_d_dyn · |S|` is computed per-cell with no level scaling.
                        # In high-stress regions the stress term dominates → uniform
                        # lattice ν_sgs → physical ν scales with dx → L1 still has 8×
                        # the dissipation of L4 → visible velocity step at L1↔L2.
                        # Multiplying by amr_sgs_scale = (1/2)^(NUM_LEVELS - level_id)
                        # makes the stress contribution scale 1/dx in lattice units,
                        # so physical ν_phys = ν_lat · dx · cs is level-uniform.
                        nu_sgs_stress = c_d_dyn * S_mag_loc * amr_sgs_scale
                        # Safety clamp: cap adaptive ν_sgs at a physical maximum (prevents runaway)
                        nu_sgs_stress = min(nu_sgs_stress, 0.1f0)
                        nu_eddy = max(nu_eddy, nu_sgs_stress, nu_sgs_floor_local)
                    else
                        # ── Legacy variable_nu_sgs path (Π_neq-based) ─────────
                        pxx_pn = pi_neq_in[x, y, z, b_idx, 1]
                        pyy_pn = pi_neq_in[x, y, z, b_idx, 2]
                        pzz_pn = pi_neq_in[x, y, z, b_idx, 3]
                        pxy_pn = pi_neq_in[x, y, z, b_idx, 4]
                        pxz_pn = pi_neq_in[x, y, z, b_idx, 5]
                        pyz_pn = pi_neq_in[x, y, z, b_idx, 6]
                        pi_mag_post = sqrt(pxx_pn*pxx_pn + pyy_pn*pyy_pn + pzz_pn*pzz_pn +
                                           2.0f0*(pxy_pn*pxy_pn + pxz_pn*pxz_pn + pyz_pn*pyz_pn))
                        # τ-correction for post→pre: |Π^pre| = |τ/(τ−1)|·|Π^post|
                        denom_tau = tau_molecular - 1.0f0
                        tau_corr = (abs(denom_tau) < 1.0f-3) ? 1.0f0 : (tau_molecular / denom_tau)
                        pi_mag_pre = abs(tau_corr) * pi_mag_post
                        # Guard ρ against uninitialized values
                        rho_cell = rho_out[x, y, z, b_idx]
                        if !isfinite(rho_cell) || rho_cell < 0.01f0
                            rho_cell = 1.0f0
                        end
                        # AMR uniform-ν_phys fix v2 (2026-05-28): apply amr_sgs_scale
                        # to the stress-based term. See dynamic Germano branch above
                        # for full rationale. Required because the stress contribution
                        # `c_pi_stress · |Π_neq| / (ρ·cs²)` is per-cell with no level
                        # scaling, which leaves a residual velocity step at L1↔L2 in
                        # high-stress regions (inlet pulse front, near-AMR-boundary).
                        nu_sgs_stress = c_pi_stress * pi_mag_pre / (rho_cell * cs2_val + 1.0f-10) * amr_sgs_scale
                        # Safety clamp: cap adaptive ν_sgs at a physical maximum (prevents runaway)
                        nu_sgs_stress = min(nu_sgs_stress, 0.1f0)
                        nu_eddy = max(nu_eddy, nu_sgs_stress, nu_sgs_floor_local)
                    end
                else
                    nu_eddy = max(nu_eddy, nu_sgs_background_local)
                end

                # Local transonic shoulder-pocket limiter. The cylinder failure in
                # long-span Z096 was not a streamwise-frame issue: it was a lateral
                # shoulder pocket where |u_z| went supersonic while the x-only local
                # lattice selector quite correctly stayed in Ux=0. Add dissipation
                # only when transverse/local Mach is paired with density split or a
                # density-gradient sensor, so the wake is not globally overdamped.
                mach_pocket_boost = 0.0f0
                begin
                    cs_local = sqrt(max(cs2_val * max(T_local, 0.2f0), 1.0f-6))
                    inv_cs_local = 1.0f0 / cs_local
                    u_mag_pocket = sqrt(ux*ux + uy*uy + uz*uz)
                    u_perp_pocket = sqrt(uy*uy + uz*uz)
                    mach_cell = u_mag_pocket * inv_cs_local
                    mach_perp = u_perp_pocket * inv_cs_local

                    t_mach = clamp((mach_cell - 0.50f0) / 0.25f0, 0.0f0, 1.0f0)
                    mach_boost = t_mach * t_mach * (3.0f0 - 2.0f0 * t_mach)
                    t_perp = clamp((mach_perp - 0.32f0) / 0.25f0, 0.0f0, 1.0f0)
                    perp_boost = t_perp * t_perp * (3.0f0 - 2.0f0 * t_perp)
                    t_rho_lo = clamp((0.90f0 - rho) / 0.26f0, 0.0f0, 1.0f0)
                    rho_lo_boost = t_rho_lo * t_rho_lo * (3.0f0 - 2.0f0 * t_rho_lo)
                    t_rho_hi = clamp((rho - 1.30f0) / 0.70f0, 0.0f0, 1.0f0)
                    rho_hi_boost = t_rho_hi * t_rho_hi * (3.0f0 - 2.0f0 * t_rho_hi)
                    rho_split_boost = max(rho_lo_boost, rho_hi_boost)

                    transverse_only_boost = perp_boost * mach_boost
                    mach_pocket_boost = max(
                        max(perp_boost * max(rho_split_boost, shock_boost),
                            mach_boost * rho_split_boost),
                        max(shock_boost * mach_boost, transverse_only_boost)
                    )
                    if mach_pocket_boost > 0.0f0
                        pocket_nu_max = max(5.00f0 * shock_nu_max, 0.24f0)
                        nu_eddy = max(nu_eddy, mach_pocket_boost * pocket_nu_max * amr_sgs_scale)
                    end
                end

                # ── Level-independent lattice-ν floor (2026-05-29) ──
                # The amr_sgs_scale machinery deliberately gives COARSE levels a
                # SMALL lattice ν (to make PHYSICAL ν uniform → clean AMR coupling).
                # But a streamwise/checkerboard lattice mode is a LATTICE-scale
                # instability governed by lattice ν, so L1 (smallest lattice ν) is
                # the least damped against it — observed in case 030 as growing
                # vertical velocity/Mach banding in the L1 freestream. This floor
                # guarantees a MINIMUM lattice ν on EVERY level, applied WITHOUT
                # amr_sgs_scale, so it lifts the under-damped coarse levels while
                # staying inert on fine levels where nu_eddy already exceeds it.
                # nu_sgs_lattice_floor == 0 → no-op (backward compatible).
                nu_eddy = max(nu_eddy, nu_sgs_lattice_floor)

                # ── Knudsen-adaptive τ — Latt, Coreixas, Beny, Parmigiani 2020 Eq. 2.28 ──
                # FULL (not proxy) form using the population-level Knudsen number:
                #     ε = Σ_k |f_k − f_eq_k| / Σ_k |f_eq_k|      (L1 relative residual)
                # Captures all 39-direction non-equilibrium, not just the 2nd moment —
                # this is what Latt 2020 actually publishes. f_eq is already computed
                # above (STEP 3a). α(ε) piecewise-constant table: smooth flow (ε<0.01)
                # gets α=1 (no extra dissipation); boundary-layer / shock regions
                # (ε≥1) get α=1/τ, capping τ_eff at 1 (max stability margin). The
                # dissipation increment is layered via max() with existing WALE +
                # variable_nu_sgs so the user can validate Knudsen-alone by disabling
                # those independently.
                knudsen_eps_local = 0.0f0
                if knudsen_adaptive_i == Int32(1)
                    sum_df  = 0.0f0
                    sum_feq = 0.0f0
                    for k in Int32(1):n_d
                        sum_df  += abs(f_stored[k] - f_eq[k])
                        sum_feq += abs(f_eq[k])
                    end
                    eps_kn = sum_feq > 1.0f-10 ? sum_df / sum_feq : 0.0f0
                    knudsen_eps_local = eps_kn
                    alpha_kn = eps_kn < 0.01f0 ? 1.0f0 :
                               eps_kn < 0.1f0  ? 1.05f0 :
                               eps_kn < 1.0f0  ? 1.35f0 :
                                                 (tau_molecular > 1.0f-3 ? 1.0f0 / tau_molecular : 2.0f0)
                    if alpha_kn > 1.0f0
                        # Δτ = (α−1)·τ → Δν = Δτ·cs² (layered via max with existing stack)
                        nu_knudsen = (alpha_kn - 1.0f0) * tau_molecular * cs2_val
                        nu_eddy = max(nu_eddy, nu_knudsen)
                    end
                end

                # ── AMR feather-band extra dissipation ────────────────
                # Adds fw·feather_nu_max to ν_eddy in the N-cell-thick band
                # adjacent to any inactive-block face of this fine level.
                # Damps the O(1e-4) density seam that the prolongation-
                # streaming mix injects at the halo, preventing it from
                # propagating inward into the flow region of interest.
                # Layered via max() so it never reduces dissipation when
                # WALE / variable_ν_sgs / Knudsen-τ are already larger.
                if feather_active_i == Int32(1)
                    fw = feather_weight_in[x, y, z, b_idx]
                    if fw > 0.0f0
                        fw_eff = fw * amr_mismatch_weight
                        if fw_eff > 0.0f0
                            nu_eddy = max(nu_eddy, fw_eff * feather_nu_max_local)
                        end
                    end
                end

                # tau = nu / cs² + 0.5  (generalized for cs² = 2/3)
                nu_mol  = (tau_molecular - 0.5f0) * cs2_val
                nu_eff  = nu_mol + nu_eddy
                tau_eff = 0.5f0 + nu_eff / cs2_val
                omega   = 1.0f0 / max(tau_eff, 0.500001f0)
                beta = 0.5f0 * omega

                # ══════════════════════════════════════════════════════
                # STEP 4: f_eq already computed in STEP 3a above (moved up for
                # the Knudsen-τ refactor). `lambda_eq`, `mass_corr_eq`, `f_eq`
                # are in scope and reused through the rest of the collision.
                # ══════════════════════════════════════════════════════

                # ══════════════════════════════════════════════════════
                # STEP 4b: Clamp f_stored for positivity before entropy
                # ══════════════════════════════════════════════════════
                any_negative = false
                for k in Int32(1):n_d
                    if f_stored[k] < F_FLOOR
                        any_negative = true
                    end
                end
                if any_negative
                    worst_ratio = 1.0f0
                    for k in Int32(1):n_d
                        fneq_k = f_stored[k] - f_eq[k]
                        if fneq_k < 0.0f0 && f_eq[k] > F_FLOOR
                            alpha_k = (f_eq[k] - F_FLOOR) / (-fneq_k + F_FLOOR)
                            worst_ratio = min(worst_ratio, alpha_k)
                        end
                    end
                    worst_ratio = clamp(worst_ratio, 0.0f0, 1.0f0)
                    for k in Int32(1):n_d
                        f_stored[k] = f_eq[k] + worst_ratio * (f_stored[k] - f_eq[k])
                        f_stored[k] = max(f_stored[k], F_FLOOR)
                    end
                end

                # Local realizability guard for transonic rarefaction/shear pockets.
                # It damps only the non-conserved residual f-feq, and only when
                # density, density-gradient, or speed sensors indicate a dangerous
                # cell. This catches the rho_min/rho_max split without imposing a
                # global wake viscosity floor.
                u_guard_mag = sqrt(ux*ux + uy*uy + uz*uz)
                cs_guard = sqrt(max(cs2_val * max(T_local, 0.2f0), 1.0f-6))
                mach_guard = u_guard_mag / cs_guard
                trans_guard_mag = sqrt(uy*uy + uz*uz)
                mach_trans_guard = trans_guard_mag / cs_guard
                t_mach_guard = clamp((mach_guard - 0.62f0) / 0.22f0, 0.0f0, 1.0f0)
                mach_guard_boost = t_mach_guard * t_mach_guard * (3.0f0 - 2.0f0 * t_mach_guard)
                t_trans_guard = clamp((mach_trans_guard - 0.38f0) / 0.22f0, 0.0f0, 1.0f0)
                trans_guard_boost = t_trans_guard * t_trans_guard * (3.0f0 - 2.0f0 * t_trans_guard)
                rho_low_guard = clamp((0.88f0 - rho) / 0.28f0, 0.0f0, 1.0f0)
                rho_high_guard = clamp((rho - 1.35f0) / 0.65f0, 0.0f0, 1.0f0)
                u_ref_guard = max(max(u_inlet, 0.1f0), 0.54f0)
                u_spike_guard = max(
                    clamp((u_guard_mag - u_ref_guard) / 0.24f0, 0.0f0, 1.0f0),
                    0.85f0 * max(mach_guard_boost, trans_guard_boost)
                )
                neq_guard = clamp((knudsen_eps_local - 0.12f0) / 0.35f0, 0.0f0, 1.0f0)
                pocket_neq_guard = max(max(shock_boost, neq_guard), max(mach_guard_boost, trans_guard_boost))
                guard_strength = max(
                    max(shock_boost * max(neq_guard, mach_guard_boost), rho_low_guard * pocket_neq_guard),
                    max(rho_high_guard, u_spike_guard * pocket_neq_guard)
                )
                guard_strength = max(guard_strength, mach_pocket_boost)
                if eq_velocity_limited || macro_realizability_limited
                    guard_strength = max(guard_strength, 1.0f0)
                end
                if guard_strength > 0.0f0
                    realizability_eps = knudsen_eps_local
                    if realizability_eps <= 1.0f-6
                        sum_df_guard = 0.0f0
                        sum_feq_guard = 0.0f0
                        for k in Int32(1):n_d
                            sum_df_guard += abs(f_stored[k] - f_eq[k])
                            sum_feq_guard += abs(f_eq[k])
                        end
                        realizability_eps = sum_feq_guard > 1.0f-10 ? sum_df_guard / sum_feq_guard : 0.0f0
                    end
                    eps_cap = max(0.025f0, 0.18f0 - 0.155f0 * guard_strength)
                    if realizability_eps > eps_cap
                        theta_guard = clamp(eps_cap / max(realizability_eps, 1.0f-12), 0.0f0, 1.0f0)
                        for k in Int32(1):n_d
                            f_stored[k] = f_eq[k] + theta_guard * (f_stored[k] - f_eq[k])
                            f_stored[k] = max(f_stored[k], F_FLOOR)
                        end
                    end
                end

                # ══════════════════════════════════════════════════════
                # STEP 5: Mirror decomposition  Δf = s + h
                # ══════════════════════════════════════════════════════
                s_arr = MVector{39, Float32}(undef)
                h_arr = MVector{39, Float32}(undef)
                for k in Int32(1):n_d
                    df_k  = f_stored[k] - f_eq[k]
                    df_kb = f_stored[opp_arr[k]] - f_eq[opp_arr[k]]
                    s_arr[k] = 0.5f0 * (df_k + df_kb)
                    h_arr[k] = 0.5f0 * (df_k - df_kb)
                end

                # ══════════════════════════════════════════════════════
                # STEP 5b: HRR-style hybridization of the stress mode
                # ══════════════════════════════════════════════════════
                # Per Coreixas 2017 (PRE 96, 033306), Jacob-Malaspinas-Sagaut 2018
                # (J. Turbulence 19), Astoul 2020 (JCP 418), Schukmann 2025 (Fluids
                # 10, 31): the projection-based 2nd-order non-equilibrium coefficient
                # contains aliased non-hydrodynamic content that AMR interfaces
                # amplify, emitting spurious acoustic waves.
                #
                # The HRR cure blends a_1^(2),PR (projection from f_neq, current path)
                # with a_1^(2),FD (finite-difference Chapman-Enskog continuum stress):
                #     a_1^(2)_αβ = σ·a_1^(2),PR_αβ + (1-σ)·a_1^(2),FD_αβ
                # where a_1^(2),FD_αβ = -ρ·c_s²·(2τ-1)·S_αβ and S = (∇u + ∇uᵀ)/2.
                # The strain rate tensor uses the velocity gradients g_αβ already
                # computed above for WALE (line 1027-1032) — zero extra memory traffic.
                #
                # σ=1 → pure projection (legacy entropic). σ=0.98 → recommended for
                # transonic per Schukmann 2025 §3.1. σ=0.999 → low-Re per DAGA 2024.
                if hrr_hybrid_active == Int32(1) && hrr_sigma < 0.999f0
                    # Strain rate from already-computed velocity gradients
                    Sxx_hrr = g11
                    Syy_hrr = g22
                    Szz_hrr = g33
                    # S12, S13, S23 already computed at lines 1063-1065 (above)
                    Sxy_hrr = S12
                    Sxz_hrr = S13
                    Syz_hrr = S23

                    # FD-based stress moment via Chapman-Enskog: Π^FD = -ρ·c_s²·(2τ-1)·S
                    # tau_eff is in scope from STEP 3 (after WALE/Knudsen/SGS adjustments)
                    tau_minus_half = max(tau_eff - 0.5f0, 0.0f0)
                    coeff_pi_fd = -2.0f0 * rho * cs2_val * tau_minus_half
                    pi_xx_fd = coeff_pi_fd * Sxx_hrr
                    pi_yy_fd = coeff_pi_fd * Syy_hrr
                    pi_zz_fd = coeff_pi_fd * Szz_hrr
                    pi_xy_fd = coeff_pi_fd * Sxy_hrr
                    pi_xz_fd = coeff_pi_fd * Sxz_hrr
                    pi_yz_fd = coeff_pi_fd * Syz_hrr

                    # Project Π^FD onto each direction's 2nd-order Hermite tensor
                    # to produce the FD-based stress mode s_FD_k. The result is
                    # symmetric in k ↔ k̄ by construction (Q_αβ is even in c_k),
                    # so it lives entirely in the s subspace and does not pollute h.
                    inv_2cs4 = 0.5f0 / (cs2_val * cs2_val)
                    one_minus_sigma = 1.0f0 - hrr_sigma
                    @inbounds for k in Int32(1):n_d
                        cxk = Float32(cx_arr[k]) - shift_x
                        cyk = Float32(cy_arr[k]) - shift_y
                        czk = Float32(cz_arr[k]) - shift_z
                        wk  = Float32(w_arr[k])
                        Qxx = cxk * cxk - cs2_val
                        Qyy = cyk * cyk - cs2_val
                        Qzz = czk * czk - cs2_val
                        Qxy = cxk * cyk
                        Qxz = cxk * czk
                        Qyz = cyk * czk
                        contraction = Qxx * pi_xx_fd + Qyy * pi_yy_fd + Qzz * pi_zz_fd +
                                      2.0f0 * (Qxy * pi_xy_fd + Qxz * pi_xz_fd + Qyz * pi_yz_fd)
                        s_fd_k = wk * contraction * inv_2cs4
                        s_arr[k] = hrr_sigma * s_arr[k] + one_minus_sigma * s_fd_k
                    end
                end

                # ── Sigma regularization: damp higher-order h-component ──
                # _lambda_param is reused as sigma ∈ [0,1]:
                #   sigma=0 → pure KBC (no damping, standard entropic)
                #   sigma=1 → full regularization (discard h, only stress remains)
                # Applied before α solver so entropy is computed on dampened h.
                # This selectively removes spurious high-frequency content from
                # polynomial equilibrium errors at supersonic speeds without
                # destroying the stress tensor (which sets viscosity correctly).
                # Phase B4: adaptive sigma (h-damping) from local |Π_neq|.
                # Smooth regions (small |Π_neq|): sigma → sigma_max (kills checkerboard)
                # Physics regions (shocks, BL): sigma → sigma_min (preserves structure)
                sigma = _lambda_param
                if variable_sigma_i == Int32(1)
                    # Read Π_neq^post already loaded by variable_nu_sgs path (or re-read)
                    pxx_sg = pi_neq_in[x, y, z, b_idx, 1]
                    pyy_sg = pi_neq_in[x, y, z, b_idx, 2]
                    pzz_sg = pi_neq_in[x, y, z, b_idx, 3]
                    pxy_sg = pi_neq_in[x, y, z, b_idx, 4]
                    pxz_sg = pi_neq_in[x, y, z, b_idx, 5]
                    pyz_sg = pi_neq_in[x, y, z, b_idx, 6]
                    pi_mag_post_sg = sqrt(pxx_sg*pxx_sg + pyy_sg*pyy_sg + pzz_sg*pzz_sg +
                                          2.0f0*(pxy_sg*pxy_sg + pxz_sg*pxz_sg + pyz_sg*pyz_sg))
                    # τ-correction to pre-collision magnitude
                    denom_tau_sg = tau_molecular - 1.0f0
                    tau_corr_sg = (abs(denom_tau_sg) < 1.0f-3) ? 1.0f0 : (tau_molecular / denom_tau_sg)
                    pi_mag_pre_sg = abs(tau_corr_sg) * pi_mag_post_sg
                    # Quadratic blend: keeps sigma HIGH unless |Π|/pi_ref is large.
                    # Linear blend gave too little damping in moderate-|Π| regions (wakes,
                    # post-shock) where noise still accumulates. Quadratic concentrates
                    # the transition at strong shocks only.
                    #   σ = σ_max − (σ_max−σ_min)·min((|Π|/pi_ref)², 1)
                    t_blend = min(pi_mag_pre_sg / max(sigma_pi_ref_p, 1.0f-10), 1.0f0)
                    blend = t_blend * t_blend
                    sigma = sigma_max_p - (sigma_max_p - sigma_min_p) * blend
                end
                if amr_regularization_active_i == Int32(1)
                    fw_sigma = feather_weight_in[x, y, z, b_idx]
                    if fw_sigma > 0.0f0
                        # AMR bands need high-order cleanup more than extra stress
                        # viscosity. Suppress h modes near refinement frontiers while
                        # leaving the hydrodynamic stress mode to τ/WALE/Knudsen.
                        fw_sigma_eff = fw_sigma * amr_mismatch_weight
                        if fw_sigma_eff > 0.0f0
                            sigma = max(sigma, min(0.995f0, 0.98f0 + 0.015f0 * fw_sigma_eff))
                        end
                    end
                end
                if mach_pocket_boost > 0.0f0
                    sigma = max(sigma, min(0.995f0, 0.80f0 + 0.195f0 * mach_pocket_boost))
                end
                # h_scale_fused carries the σ h-damp scalar for the fused path
                # (where h_arr is not materialized). 1.0 = no damping.
                h_scale_fused = (sigma > 0.0f0) ? (1.0f0 - sigma) : 1.0f0
                if !USE_FUSED_SH && sigma > 0.0f0
                    for k in Int32(1):n_d
                        h_arr[k] *= h_scale_fused
                    end
                end

                # ══════════════════════════════════════════════════════
                # STEP 6: Solve for entropic stabilizer α
                # ══════════════════════════════════════════════════════
                h_norm_sq = 0.0f0
                if USE_FUSED_SH
                    for k in Int32(1):n_d
                        _, h_k = _mirror_sh_inline(f_stored, f_eq, opp_arr, k, h_scale_fused)
                        h_norm_sq += h_k * h_k
                    end
                else
                    for k in Int32(1):n_d
                        h_norm_sq += h_arr[k] * h_arr[k]
                    end
                end

                inv_2beta = 0.5f0 / max(beta, 0.01f0)
                alpha = 2.0f0

                if h_norm_sq > 1.0f-20
                    H_pre = entropy_H_39(f_stored, w_arr, n_d)

                    solver_type = _adaptive_omega4

                    if USE_FUSED_SH
                        if solver_type == Int32(1)
                            alpha = solve_alpha_newton_39_fused(f_stored, f_eq, opp_arr, w_arr, beta, H_pre,
                                        Int32(8), 1.0f-5, h_scale_fused, n_d)
                        elseif solver_type == Int32(2)
                            alpha = solve_alpha_approximate_39_fused(f_stored, f_eq, opp_arr, w_arr, beta, h_scale_fused, n_d)
                        elseif solver_type == Int32(3)
                            alpha = solve_alpha_taylor_39_fused(f_stored, f_eq, opp_arr, w_arr, beta, H_pre,
                                        Int32(10), h_scale_fused, n_d)
                        else
                            alpha = solve_alpha_bisection_39_fused(f_stored, f_eq, opp_arr, w_arr, beta, H_pre,
                                        Int32(10), h_scale_fused, n_d)
                        end
                    else
                        if solver_type == Int32(1)
                            alpha = solve_alpha_newton_39(f_eq, s_arr, h_arr, w_arr, beta, H_pre,
                                        Int32(8), 1.0f-5, n_d)
                        elseif solver_type == Int32(2)
                            alpha = solve_alpha_approximate_39(f_eq, s_arr, h_arr, w_arr, beta, n_d)
                        elseif solver_type == Int32(3)
                            alpha = solve_alpha_taylor_39(f_eq, s_arr, h_arr, w_arr, beta, H_pre,
                                        Int32(10), n_d)
                        else
                            alpha = solve_alpha_bisection_39(f_eq, s_arr, h_arr, w_arr, beta, H_pre,
                                        Int32(10), n_d)
                        end
                    end

                    if !(alpha > 0.0f0)
                        alpha = inv_2beta
                    end
                    alpha = clamp(alpha, inv_2beta, 6.0f0)
                end

                # ══════════════════════════════════════════════════════
                # STEP 7: KBC collision
                #   f* = f_eq + (1 - 2β)·s + (1 - 2β·α)·h
                # ══════════════════════════════════════════════════════
                coeff_s = 1.0f0 - 2.0f0 * beta
                coeff_h = 1.0f0 - 2.0f0 * beta * alpha

                # ══════════════════════════════════════════════════════
                # STEP 8: Guo forcing + positivity check + write output
                # ══════════════════════════════════════════════════════
                inv_cs2 = 1.0f0 / cs2_val

                # 2026-05-30 register-pressure fix: REUSE f_stored's buffer for the
                # post-collision populations instead of allocating a separate
                # MVector{39} f_post. f_stored's last read on the NON-FUSED path is
                # the α-solver's entropy_H_39(f_stored,...) at ~line 2407; it is dead
                # from here, so the alias is safe (saves 39 floats).
                # ★ On the FUSED path f_post must NOT alias f_stored: the fused KBC
                # recomputes s_k/h_k from f_stored[k] AND f_stored[opp(k)] inside the
                # write loop, so writing f_post[k]=f_stored[k] would corrupt the
                # opp(k) read on later iterations. Use a separate buffer there. Net
                # still saves 39 (fused drops s+h=78, spends f_post=39 → −39 vs mono).
                f_post = USE_FUSED_SH ? MVector{39, Float32}(undef) : f_stored
                f_min_val = 1.0f0
                for k in Int32(1):n_d
                    # Shifted lattice: Guo forcing uses unshifted c with relative velocity
                    cx_f = Float32(cx_arr[k]) - shift_x
                    cy_f = Float32(cy_arr[k]) - shift_y
                    cz_f = Float32(cz_arr[k]) - shift_z
                    w_k  = w_arr[k]

                    if USE_FUSED_SH
                        s_k_f, h_k_f = _mirror_sh_inline(f_stored, f_eq, opp_arr, k, h_scale_fused)
                        f_coll_k = f_eq[k] + coeff_s * s_k_f + coeff_h * h_k_f
                    else
                        f_coll_k = f_eq[k] + coeff_s * s_arr[k] + coeff_h * h_arr[k]
                    end

                    # Guo forcing with variable cs² (unshifted c, relative velocity)
                    cu = cx_f*ux_rel + cy_f*uy_rel + cz_f*uz_rel
                    force_term = w_k * inv_cs2 * (
                        (cx_f - ux_rel + inv_cs2*cu*cx_f) * Fx_wall +
                        (cy_f - uy_rel + inv_cs2*cu*cy_f) * Fy_wall +
                        (cz_f - uz_rel + inv_cs2*cu*cz_f) * Fz_wall
                    )

                    f_post[k] = f_coll_k + (1.0f0 - 0.5f0*omega) * force_term
                    f_min_val = min(f_min_val, f_post[k])
                end

                # ── Positivity enforcement ──
                if f_min_val < 0.0f0
                    blend = 1.0f0
                    for k in Int32(1):n_d
                        feq_k = f_eq[k]
                        fneq_k = f_post[k] - feq_k
                        if fneq_k < 0.0f0 && feq_k > F_FLOOR
                            blend_k = feq_k / (feq_k - fneq_k + F_FLOOR)
                            blend = min(blend, blend_k)
                        end
                    end
                    blend = max(blend, 0.0f0)
                    for k in Int32(1):n_d
                        f_post[k] = f_eq[k] + blend * (f_post[k] - f_eq[k])
                    end
                end

                if guard_strength > 0.0f0
                    sum_post_df = 0.0f0
                    sum_post_feq = 0.0f0
                    for k in Int32(1):n_d
                        sum_post_df += abs(f_post[k] - f_eq[k])
                        sum_post_feq += abs(f_eq[k])
                    end
                    eps_post = sum_post_feq > 1.0f-10 ? sum_post_df / sum_post_feq : 0.0f0
                    eps_post_cap = max(0.025f0, 0.22f0 - 0.195f0 * guard_strength)
                    if eps_post > eps_post_cap
                        theta_post = clamp(eps_post_cap / max(eps_post, 1.0f-12), 0.0f0, 1.0f0)
                        for k in Int32(1):n_d
                            f_post[k] = f_eq[k] + theta_post * (f_post[k] - f_eq[k])
                            f_post[k] = max(f_post[k], F_FLOOR)
                        end
                    end
                end

                # ── Recursive Regularization (RR-LBM) — Hermite projection ──
                # Project the post-collision populations onto the order-3
                # Hermite subspace (mass + momentum + stress + heat-flux).
                # This annihilates the ghost eigenmodes — including the
                # (-1)^(y+z) checkerboard mode visible in the wake of cases
                # 001-006 — by construction. Mass, momentum, stress, and
                # heat-flux are preserved exactly (Hermite orthogonality).
                # See kernel_regularization_d3q39.jl for derivation.
                if rr_lbm_active_i == Int32(1)
                    # AUDIT FIX (2026-06-03): pass the Galilean shift so the
                    # Hermite projection works in the lab frame and preserves
                    # mass/momentum under a supersonic shift (matches the SVV
                    # call below). For shift=0 the result is unchanged.
                    regularize_d3q39_inline!(
                        f_post, f_eq, cx_arr, cy_arr, cz_arr, w_arr, n_d, cs2_val,
                        shift_x, shift_y, shift_z
                    )
                end

                # ── Spectral Vanishing Viscosity (SVV) — Y-Z bi-Laplacian ──
                # Damps spatial high-k modes (the visible Y-Z checkerboard)
                # without touching low-k physical scales. Conservation correction
                # restores ρ and ρu exactly per cell. See
                # kernel_svv_yz_d3q39.jl for derivation and theory.
                if svv_yz_epsilon > 0.0f0
                    apply_svv_yz_d3q39_inline!(
                        f_post, f_in, obstacle, neighbor_table,
                        x, y, z, b_idx, block_size,
                        svv_yz_epsilon, n_d,
                        w_arr, cx_arr, cy_arr, cz_arr, cs2_val,
                        shift_x, shift_y, shift_z
                    )
                end

                # SOTA #5 (2026-05-29): streamwise-X SVV. Gated by the caller to
                # the coarsest level(s) only (svv_x_epsilon > 0 there, 0 elsewhere)
                # to kill the growing L1 streamwise velocity/Mach banding without
                # adding streamwise dissipation on fine levels where the real
                # physics lives. Same bi-Laplacian + exact mass/momentum
                # conservation correction as the Y-Z variant.
                if svv_x_epsilon > 0.0f0
                    apply_svv_x_d3q39_inline!(
                        f_post, f_in, obstacle, neighbor_table,
                        x, y, z, b_idx, block_size,
                        svv_x_epsilon, n_d,
                        w_arr, cx_arr, cy_arr, cz_arr, cs2_val,
                        shift_x, shift_y, shift_z
                    )
                end

                # ── AUDIT FIX (2026-06-03): post-RR/SVV positivity re-enforcement ──
                # RR-LBM and SVV run AFTER the entropic positivity guard above,
                # and their Hermite/bi-Laplacian projections can overshoot a
                # population slightly negative. The entropic operator's whole
                # stability argument is post-collision positivity, so re-apply the
                # SAME mass-and-momentum-CONSERVING uniform blend toward f_eq used
                # at the main guard (scaling f−f_eq uniformly preserves all moments
                # of f_eq). Only runs when an add-on was active and actually
                # produced a negative — a no-op otherwise (the common path).
                if rr_lbm_active_i == Int32(1) || svv_yz_epsilon > 0.0f0 || svv_x_epsilon > 0.0f0
                    f_min_post = f_post[1]
                    for k in Int32(2):n_d
                        f_min_post = min(f_min_post, f_post[k])
                    end
                    if f_min_post < 0.0f0
                        blend_pp = 1.0f0
                        for k in Int32(1):n_d
                            feq_k = f_eq[k]
                            fneq_k = f_post[k] - feq_k
                            if fneq_k < 0.0f0 && feq_k > F_FLOOR
                                blend_k = feq_k / (feq_k - fneq_k + F_FLOOR)
                                blend_pp = min(blend_pp, blend_k)
                            end
                        end
                        blend_pp = max(blend_pp, 0.0f0)
                        for k in Int32(1):n_d
                            f_post[k] = f_eq[k] + blend_pp * (f_post[k] - f_eq[k])
                        end
                    end
                end

                # Write output (with optional Xu-Sagaut sponge correction)
                # Phase C1: when sponge_xu_sagaut enabled, apply post-collision
                # additive forcing:  f_out[k] = f_coll + sp · (f_eq_ref[k] − f_eq_local[k])
                # f_eq_inlet_arr is the pre-computed reference equilibrium (const inlet state).
                apply_xu_sagaut = (sponge_xu_sagaut_i == Int32(1)) && (sp > 0.0f0)
                # ── Gendre 2017 AMR normal-gradient correction (transition cells) ──
                # Restore the advective gradient NORMAL to the refinement interface
                # that standard stream-and-collide drops at the coarse↔fine seam
                # (Eq. 21; leading-order ΔFα = −c⊥α·∂⊥fα, see kernel_gendre_*.jl).
                # Only on AMR-halo cells (parent_halo_links>0 ⇒ never L1), and only
                # when a single dominant interface normal exists (planar face; edge/
                # corner cells fall back to no correction — Gendre §IV.B). DCE'd off.
                gendre_axis = Int32(0)
                if GENDRE_VAL && parent_halo_links > Int32(0) && !is_obs && !is_covered
                    gendre_axis = _gendre_pick_normal(
                        parent_halo_x_neg, parent_halo_x_pos,
                        parent_halo_y_neg, parent_halo_y_pos,
                        parent_halo_z_neg, parent_halo_z_pos)
                end
                for k in Int32(1):n_d
                    f_coll = f_post[k]
                    if !(f_coll == f_coll) || f_coll > 1.0f3
                        f_coll = f_eq[k]
                    end
                    if apply_xu_sagaut
                        f_coll += sp * (f_eq_inlet_arr[f_eq_inlet_offset + k] - f_eq[k])
                    end
                    if GENDRE_VAL && gendre_axis != Int32(0)
                        c_perp = gendre_axis == Int32(1) ?
                                     Float32(Int32(cx_arr[k]) - array_shift_x_i + frame_shift_x_i) :
                                 gendre_axis == Int32(2) ?
                                     Float32(Int32(cy_arr[k]) - array_shift_y_i + frame_shift_y_i) :
                                     Float32(Int32(cz_arr[k]) - array_shift_z_i + frame_shift_z_i)
                        dF = gendre_normal_correction_d3q39(
                                f_in, neighbor_table,
                                Int32(x), Int32(y), Int32(z), Int32(b_idx), Int32(block_size),
                                k, c_perp, gendre_axis, f_eq[k], 0.5f0)
                        f_coll += dF
                        if f_coll < F_FLOOR
                            f_coll = F_FLOOR
                        end
                    end
                    if store_post_collision == Int32(1)
                        f_post_collision[x, y, z, b_idx, k] = f_coll
                    end
                    f_out[x, y, z, b_idx, k] = f_coll
                end
            end
        end
    end
end
