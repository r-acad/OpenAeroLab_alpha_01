# ==============================================================================
# Gendre 2017 — directional-splitting NORMAL-GRADIENT correction at AMR interfaces
#   Gendre, Ricot, Fritz, Sagaut, Phys. Rev. E 96, 023311 (2017)
# ==============================================================================
#
# THE PROBLEM (Gendre §III.A). At a coarse↔fine transition node the standard
# LBM stream-and-collide omits the gradient of f NORMAL to the interface: the
# characteristic is carried by the IN-PLANE velocity component c∥, so the
# perpendicular advection  −c⊥ ∂⊥ f  is silently dropped. The result is the
# weak SPURIOUS ACOUSTIC fluctuation that emerges when a vortical/acoustic
# structure crosses the refinement box — worse at higher subsonic Mach. This is
# exactly the structural coarse↔fine SEAM diagnosed in this solver at Ma 0.9
# (inner-halo ~0.1, robust to ALL collision-level damping: σ-tuning, SVV,
# feather, feedback — because none of them restore the missing normal advection).
#
# THE FIX (Gendre Eqs. 19–30). Re-introduce the dropped term as a finite-
# difference correction Fα = c⊥α ∂⊥ fα (Eq. 21), via a change of variable
# hα = gα + (Δt/2)Fα (Eqs. 24–30) that wraps the collision as a black box. This
# is OPERATOR-INDEPENDENT — Gendre §III.B Eq. 34 proves the split term is
# unchanged for any collision operator (BGK, MRT, … and therefore the entropic
# KBC here): it touches NEITHER the α-solver, the mirror s/h decomposition, NOR
# the 13-mom Newton equilibrium. It is a pure pre/post-collision additive
# population correction.
#
# LEADING-ORDER FORM USED HERE (v1). The full Eq. 27 streamed correction is
#     [ (1/(2τ) − 1)·Fα  −  (1/(2τ))·Fα^eq ].
# In the smooth near-equilibrium flow at a clean refinement interface f ≈ f^eq,
# so ∂⊥f ≈ ∂⊥f^eq ⇒ Fα ≈ Fα^eq, and the bracket collapses to exactly  −Fα,
# INDEPENDENT of τ. This v1 implements ΔFα = −Fα = −c⊥α·∂⊥fα. Properties:
#   • EXACTLY ZERO on a uniform freestream (∂⊥f = 0) ⇒ the freestream-AMR gate
#     (case 013) stays byte-identical — the cheapest correctness check: if the
#     gate residual CHANGES on uniform flow, the FD addressing is wrong.
#   • τ-independent ⇒ no coupling to the per-cell entropic α / variable ν.
#   • The separate Fα^eq consistency term (the smaller, non-equilibrium-gradient
#     correction) is the documented v2 follow-up; it needs the neighbour f^eq
#     (two extra 13-mom evals) and is dropped here to keep the kernel JIT-small.
#
# JIT SAFETY. The entropic D3Q39 kernel sits at the @inline size threshold
# (documented InvalidIRError incidents). The correction therefore lives in the
# @noinline `gendre_normal_correction_d3q39` below; the kernel call site is a
# thin per-direction loop. Mirrors the proven `interpolate_macro_bspline`
# (kernel_quintic_prolongation.jl) and the high-order-temporal pattern.
# ==============================================================================


# Read f_in[..., k] at the cell one step (`sgn` = ±1) along `normal_axis` (1=x,
# 2=y, 3=z). Within-block: direct. Cross-block: the SAME neighbor_table address
# arithmetic the streaming loop uses (kernel_entropic_d3q39.jl ~line 911-917,
# unshifted path). Returns (value, ok). ok=false ⇒ the normal neighbour block is
# absent (ragged box edge / coarse side) ⇒ caller degrades to no correction
# (Gendre §IV.B "plane of neighbours not found" fallback).
#
# NOTE: the simple unshifted read is used (matches the `else val = f_in[nsx,…]`
# branch). In LOCAL-lattice-shift mode a cross-block neighbour may live in a
# different Galilean frame; v1 of the Gendre feature is intended for the
# global/unshifted subsonic AMR path (the Ma 0.9 freestream/wing regime), so the
# frame-remap is not reproduced here. Enabling it in a local-shift case would
# need the frame handling from the streaming loop.
@inline function _gendre_read_normal_neighbor(
        f_in, neighbor_table,
        x::Int32, y::Int32, z::Int32, b_idx::Int32, block_size::Int32,
        k::Int32, normal_axis::Int32, sgn::Int32)
    ox = normal_axis == Int32(1) ? sgn : Int32(0)
    oy = normal_axis == Int32(2) ? sgn : Int32(0)
    oz = normal_axis == Int32(3) ? sgn : Int32(0)
    tx = x + ox
    ty = y + oy
    tz = z + oz
    if tx >= Int32(1) && tx <= block_size &&
       ty >= Int32(1) && ty <= block_size &&
       tz >= Int32(1) && tz <= block_size
        return f_in[tx, ty, tz, b_idx, k], true
    end
    nb_off_x = tx < Int32(1) ? Int32(-1) : (tx > block_size ? Int32(1) : Int32(0))
    nb_off_y = ty < Int32(1) ? Int32(-1) : (ty > block_size ? Int32(1) : Int32(0))
    nb_off_z = tz < Int32(1) ? Int32(-1) : (tz > block_size ? Int32(1) : Int32(0))
    dir_idx = (nb_off_x + Int32(1)) + (nb_off_y + Int32(1)) * Int32(3) +
              (nb_off_z + Int32(1)) * Int32(9) + Int32(1)
    nb_idx = neighbor_table[b_idx, dir_idx]
    if nb_idx > Int32(0)
        nsx = tx < Int32(1) ? tx + block_size : (tx > block_size ? tx - block_size : tx)
        nsy = ty < Int32(1) ? ty + block_size : (ty > block_size ? ty - block_size : ty)
        nsz = tz < Int32(1) ? tz + block_size : (tz > block_size ? tz - block_size : tz)
        return f_in[nsx, nsy, nsz, nb_idx, k], true
    end
    return 0.0f0, false
end


# Pick the interface NORMAL axis from the parent-halo directional tallies. The
# face normal is the axis whose parent-streamed links are ONE-SIDED (all +n or
# all −n): on a +x box face every parent-sourced direction has nb_off_x=+1 while
# the transverse y/z links are symmetric (both ± present, since the face spans
# them). So the normal = the axis of MAXIMUM |pos − neg| asymmetry. Corner/edge
# cells (no single dominant axis, or none) ⇒ return 0 (no correction). A centered
# FD needs only the axis, not the sign.
@inline function _gendre_pick_normal(
        hx_neg::Int32, hx_pos::Int32,
        hy_neg::Int32, hy_pos::Int32,
        hz_neg::Int32, hz_pos::Int32)
    ax = hx_pos > hx_neg ? hx_pos - hx_neg : hx_neg - hx_pos
    ay = hy_pos > hy_neg ? hy_pos - hy_neg : hy_neg - hy_pos
    az = hz_pos > hz_neg ? hz_pos - hz_neg : hz_neg - hz_pos
    if ax >= ay && ax >= az
        return ax > Int32(0) ? Int32(1) : Int32(0)
    elseif ay >= ax && ay >= az
        return ay > Int32(0) ? Int32(2) : Int32(0)
    else
        return az > Int32(0) ? Int32(3) : Int32(0)
    end
end


# The Gendre leading-order normal-gradient correction ΔFα for direction k:
#   ΔFα = −Fα = −c⊥α · ∂⊥fα ,   ∂⊥fα ≈ ½(f_in[+⊥] − f_in[−⊥])  (2nd-order centered FD).
# Returns 0 when c⊥=0 (no perpendicular component ⇒ nothing dropped), or when
# either normal neighbour is absent (ragged edge ⇒ standard-LBM fallback).
# `f_eq_k` bounds the correction to ±`clamp_frac`·f_eq_k (a sharp-gradient
# safety net; never fires on smooth flow where ΔF ≪ f_eq).
#
# @noinline is LOAD-BEARING (not perf): the entropic D3Q39 kernel sits at the
# Julia @inline size threshold; inlining this body (FD + neighbour read + clamp)
# into the giant kernel pushes it over, after which type inference degrades to
# dynamic dispatch and EVERY op in the kernel throws InvalidIRError (the
# documented "jl_f_throw_methoderror" failure, stacktrace misleadingly pointing
# at this line). Keeping it @noinline isolates the body. Verified 2026-06-28.
@noinline function gendre_normal_correction_d3q39(
        f_in, neighbor_table,
        x::Int32, y::Int32, z::Int32, b_idx::Int32, block_size::Int32,
        k::Int32, c_perp::Float32, normal_axis::Int32,
        f_eq_k::Float32, clamp_frac::Float32)
    c_perp == 0.0f0 && return 0.0f0
    fp, okp = _gendre_read_normal_neighbor(f_in, neighbor_table, x, y, z, b_idx, block_size, k, normal_axis, Int32(1))
    fm, okm = _gendre_read_normal_neighbor(f_in, neighbor_table, x, y, z, b_idx, block_size, k, normal_axis, Int32(-1))
    # NOTE (v1.5 tested + REVERTED 2026-06-28): a one-sided 2nd-order FD from the
    # fine interior was tried for the refined-region OUTER face (where the outward
    # normal-neighbour is at the coarser parent level ⇒ centered stencil incomplete).
    # It REGRESSED: the L1-L2 seam is a genuine DISCONTINUITY, so a one-sided
    # fine-side stencil reads the step as a huge gradient and −c⊥·∂⊥f OVERSHOOTS
    # there (peak Mach 1.027→1.137, corner patches more saturated). The correct
    # outer-face treatment needs the real coarse "ghost" value (cross-level CENTERED
    # FD reading parent_f, grid-ratio-aware) — Gendre's actual §IV ghost-node form —
    # which is the documented v2. For now: bail on an incomplete stencil (correct +
    # safe; only the INNER AMR faces L2-L3/L3-L4 are corrected, where it helps).
    (okp && okm) || return 0.0f0
    dfd = 0.5f0 * (fp - fm)          # 2nd-order centered FD, Δx = 1 (Gendre §III.A)
    dF = -c_perp * dfd               # ΔFα = −Fα  (leading-order, τ-independent)
    lim = clamp_frac * f_eq_k        # f_eq_k ≥ 0 (it is w_k·exp(...) ≥ 0)
    return dF >  lim ?  lim :
           dF < -lim ? -lim : dF
end
