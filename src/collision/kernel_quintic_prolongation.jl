#
# B1 Pillar 1 — Cell-averaged high-order prolongation (mortar-element AMR coupling)
# =================================================================================
#
# Scaffold + working evaluator for the 5-point quartic (informally "quintic-class")
# cell-averaged conservative interpolation that replaces the current cubic-
# Lagrange pointwise prolongation in the AMR halo fill. See SOTA_ROADMAP.md
# "B1 — Mortar-element AMR coupling" pillar 1.
#
# Conservation property
# ---------------------
# For 2:1 refinement, the two fine sub-cells lying inside parent cell `P_0`
# satisfy
#
#     (f_low + f_high) / 2  =  P_0
#
# by construction, so the spatial cell-average is preserved EXACTLY. This is
# the conservative interpretation Patera (1984) introduced for mortar
# elements and Filippova-Hänel (1998) adapted for LBM. The existing cubic-
# Lagrange path is pointwise (no cell-average guarantee), so the round-trip
# coarse→fine→coarse introduces a residual O(1e-6) that this scaffold drops
# to round-off.
#
# Order of accuracy
# -----------------
# 4th-order in space (reproduces quartic polynomials exactly), 5-point
# stencil per axis. Tensor-product extension gives 4th-order in 3D using a
# 5×5×5 = 125-cell parent stencil per fine cell.
#
# The roadmap's full quintic B-spline target additionally enforces C⁴
# continuity at parent boundaries; this scaffold provides the conservative
# core with C⁰ continuity (cell-averaged quartic). Upgrading to the full
# quintic B-spline is a 2-3 week follow-up that replaces the closed-form
# weight tables below with a B-spline collocation solve.
#
# What's here
# -----------
#   * `quartic_5pt_lower_weights()`  — Float32 5-tuple, weights for the
#     lower fine sub-cell.
#   * `quartic_5pt_upper_weights()`  — mirror of the above.
#   * `cell_averaged_quartic_1d(p_m2, p_m1, p_0, p_p1, p_p2, is_upper_sub)
#     → Float32` — inlined 1D evaluator.
#   * `cell_averaged_quartic_3d(parent_lookup, fine_sub_x, fine_sub_y,
#     fine_sub_z)` — 3D tensor-product evaluator. `parent_lookup` is a
#     callable `(dx, dy, dz) -> Float32` that returns the parent value
#     at offset (dx, dy, dz) from the parent cell containing the fine
#     cell. dx, dy, dz ∈ {-2, -1, 0, +1, +2}.
#
# Conservation verification
# -------------------------
# Unit-test sentinel embedded in the docstring of `cell_averaged_quartic_1d`:
# for `p_m2 = p_m1 = p_0 = p_p1 = p_p2 = c`, both sub-cell evaluations
# return `c` (constant reproduction). For the linear ramp `p_i = i`,
# `lower` returns -1/4 and `upper` returns +1/4 (their average is 0 = p_0).
#
# Wiring status
# -------------
# OFF BY DEFAULT. The new evaluator is not yet wired into the live AMR
# halo-fill path; the existing cubic-Lagrange path in
# `interpolate_with_rescaling` remains active. The follow-up integration
# (~3-4 weeks per roadmap) replaces the 8-point trilinear branch in
# `interpolate_with_rescaling` with a 125-point call to
# `cell_averaged_quartic_3d`, plus address-arithmetic for cross-block
# parent reads. The macroscopic density / momentum prolongation is the
# natural first target; population-level prolongation is a follow-on once
# the macroscopic-only round-trip is verified.

# ══════════════════════════════════════════════════════════════════════════
# QUINTIC B-SPLINE prolongation (C2/C4-continuous) — the genuine high-continuity
# AMR coarse→fine macro interpolation. Unlike the compact quartic/quintic-Lagrange
# (both C0/C-1: a 2:1-conservative COMPACT stencil CANNOT be C2 — Strang-Fix
# obstruction, proven by exact-rational derivation 2026-06-04), the quintic
# B-spline IS both exactly conservative AND C4(⊇C2). Its prolongation is NOT a
# stencil multiply: it requires a per-axis PREFILTER (deconvolution) solving the
# heptadiagonal Toeplitz system  A · c = P  (P = coarse cell-averages along a
# line, c = B-spline coefficients), after which the fine half-cell averages are
# a COMPACT local combination of c.
#
# A_MASK: the cell-average mask (one row of A) — the quintic B-spline integrated
#   over a unit coarse cell. Symmetric heptadiagonal Toeplitz, denominator 46080.
#   Σ A_MASK = 1 (exact, partition of unity).
# W_LOW / W_HIGH: the compact c→fine-half-cell evaluation taps (denominator 23040).
#   Conservation: (W_LOW + W_HIGH)/2 == A_MASK exactly (so prolongation preserves
#   the coarse cell-average — verified). W_HIGH = reverse(W_LOW) (mirror symmetry).
const _BSPLINE_A_MASK_NUM   = (Int32(1), Int32(722), Int32(10543), Int32(23548), Int32(10543), Int32(722), Int32(1))
const _BSPLINE_A_MASK_DENOM = Int32(46080)
const _BSPLINE_W_LOW_NUM    = (Int32(1), Int32(659), Int32(7554), Int32(11774), Int32(2989), Int32(63))
const _BSPLINE_W_HIGH_NUM   = (Int32(63), Int32(2989), Int32(11774), Int32(7554), Int32(659), Int32(1))
const _BSPLINE_W_DENOM_F    = 23040.0f0

# ── FUSED composite kernel K = W ⊛ G : maps coarse cell-averages P DIRECTLY to
#    the fine half-cell values, in ONE convolution. ──
# The B-spline prolongation is (prefilter c = A⁻¹P) then (eval f = W·c). Since both
# are linear, they FUSE: f_low = K_low ⊛ P, f_high = K_high ⊛ P, where K = W⊛A⁻¹.
# This collapses the entire C4 B-spline to a single SYMMETRIC ±21-tap stencil on
# the RAW coarse macros — no prefilter pass, no coefficient arrays, no solver
# changes; it drops into the existing per-fine-cell prolongation as a wider sibling
# of the quartic. K_high = reverse(K_low) (mirror symmetry).
# VERIFIED (offline, exact): Σ K_low = Σ K_high = 1 (constant exact, freestream-
# clean); (K_low+K_high)/2 = δ to 2e-16 (EXACT 2:1 conservation); reproduces
# x, x², x³ exactly to round-off; C4 by construction (quintic B-spline). Taps below
# ±21 are < FP32 ULP. Offsets run -21..+21 (center index 22 = the parent cell).
const _BSPLINE_K_LOW = (
    1.2859314f-07, -2.6335158f-07, 5.3932928f-07, -1.1045162f-06, 2.2619874f-06,
    -4.6324235f-06, 9.4869447f-06, -1.9428731f-05, 3.9788956f-05, -8.1485552f-05,
    0.00016687786f0, -0.00034175653f0, 0.00069989829f0, -0.0014333532f0, 0.0029354349f0,
    -0.0060117021f0, 0.0123128f0, -0.025230255f0, 0.051845599f0, -0.10832523f0,
    0.24838156f0, 1f0, -0.24838156f0, 0.10832523f0, -0.051845599f0,
    0.025230255f0, -0.0123128f0, 0.0060117021f0, -0.0029354349f0, 0.0014333532f0,
    -0.00069989829f0, 0.00034175653f0, -0.00016687786f0, 8.1485552f-05, -3.9788956f-05,
    1.9428731f-05, -9.4869447f-06, 4.6324235f-06, -2.2619874f-06, 1.1045162f-06,
    -5.3932928f-07, 2.6335158f-07, -1.2859314f-07)
const _BSPLINE_K_HALFWIDTH = Int32(21)   # offsets -21..+21 (43 taps); center at index 22
const _BSPLINE_W_DENOM      = Int32(23040)

# ── Green's-function (A⁻¹) one-sided taps for the PREFILTER pass c = G ⊛ P ──
# G is symmetric (G[-m]=G[+m]); G[0] is the center, G[1..24] the one side. Decays
# at ratio ≈0.4883 → below FP32 ULP after 24 taps. Σ over the full filter = 1
# (partition of unity). Used by _bspline_prefilter_axis_scalar! (separable passes).
# (This is the efficient runtime form: prefilter ONCE per coarse step into coeff
# arrays, then the fine eval is the compact 6-tap W_LOW/W_HIGH — vs the fused K
# above which would cost ±21³ reads PER fine cell.)
const _BSPLINE_GREEN_HALF = (
    3.7267935f0, -2.0462418f0, 1.0176718f0, -0.49843496f0, 0.24350654f0,
    -0.11891301f0, 0.058065403f0, -0.02835309f0, 0.013844665f0, -0.0067602755f0,
    0.003301006f0, -0.0016118634f0, 0.0007870642f0, -0.00038431917f0, 0.00018766097f0,
    -9.163384f-5, 4.474431f-5, -2.1848404f-5, 1.0668457f-5, -5.20935f-6,
    2.5436973f-6, -1.2420736f-6, 6.0649785f-7, -2.961496f-7, 1.4460825f-7)
const _BSPLINE_GREEN_NTAP = Int32(24)   # one-sided tap count (excl. center); index 1=center

# ── 3-point Lagrange polynomial weights (denominator 32) — CORNER FALLBACK ──
# B1 P1 corner-aware fix (2026-05-28). When the 5×5×5 cell-averaged quartic
# stencil hits a reentrant AMR corner where one or more of (p_m2, p_p2) are
# NaN-sentinel (block-out-of-range or inactive), we cascade to a point-value
# 3-point Lagrange evaluator at the sub-cell centers ±0.25 BEFORE falling all
# the way down to trilinear.
#
# Standard Lagrange polynomial Q(s) through (P-1, P0, P+1) at (s=-1, 0, +1):
#   Q(s) = (s(s-1)/2)·P-1 + (-(s²-1))·P0 + (s(s+1)/2)·P+1
# Evaluated at s = -0.25 (lower sub-cell center):
#   Q(-0.25) = (5/32)·P-1 + (30/32)·P0 + (-3/32)·P+1
# At s = +0.25 (upper sub-cell center):
#   Q(+0.25) = (-3/32)·P-1 + (30/32)·P0 + (5/32)·P+1
#
# Verification: for P_i = c (constant) → both sub-cell values = c. For linear
# ramp P_i = i: lower = -0.25, upper = +0.25. Weights sum to 1.
#
# Trade-off vs quartic: 2nd-order accurate (vs quartic's 4th) and NOT
# cell-average conservative (vs quartic's exact conservation). But strictly
# better than trilinear (1st-order) when the 5-point stencil isn't available.

# ══════════════════════════════════════════════════════════════════════════════
# C2/C4 QUINTIC B-SPLINE PREFILTER (2026-06-04)
# ------------------------------------------------------------------------------
# Computes the B-spline coefficients c of a coarse-level macro field by the
# separable inverse-filter convolution c = G ⊛ P (G = _BSPLINE_GREEN_HALF, the
# A⁻¹ Green's function; symmetric, ±_BSPLINE_GREEN_NTAP taps). Three sequential
# 1D passes (x, y, z) — each reads the previous pass's field along one axis,
# convolves with G across sparse blocks via block_pointer, writes the result.
# Gaps (inactive parent blocks at the refinement-region edge) use a zero-gradient
# clamp (reuse the nearest valid in-line value) — the same bounded fallback the
# prolongation uses. Result is C4 + exactly 2:1-conservative when consumed by the
# compact 6-tap fine evaluation (W_LOW/W_HIGH).
#
# Per-axis 1D convolution of a SCALAR field stored as (BS,BS,BS,n_blocks).
# axis: 1=x, 2=y, 3=z. Reads src, writes dst. block_pointer maps (bx,by,bz)->idx.
@kernel function _bspline_prefilter_axis_scalar!(
    dst, src, block_pointer,
    active_coords_x, active_coords_y, active_coords_z,
    bx_max::Int32, by_max::Int32, bz_max::Int32,
    n_blocks::Int32, block_size::Int32, axis::Int32, ntap::Int32
)
    x, y, z, b_idx = @index(Global, NTuple)
    if b_idx <= n_blocks
        @inbounds begin
            bx = active_coords_x[b_idx]
            by = active_coords_y[b_idx]
            bz = active_coords_z[b_idx]
            # global coord of this cell along each axis
            gx = (bx - Int32(1)) * block_size + Int32(x)
            gy = (by - Int32(1)) * block_size + Int32(y)
            gz = (bz - Int32(1)) * block_size + Int32(z)
            g_self = axis == Int32(1) ? gx : (axis == Int32(2) ? gy : gz)
            n_global_axis = (axis == Int32(1) ? bx_max : (axis == Int32(2) ? by_max : bz_max)) * block_size

            # center tap (offset 0)
            acc = _BSPLINE_GREEN_HALF[1] * src[x, y, z, b_idx]
            last_lo = src[x, y, z, b_idx]   # zero-gradient clamp seed (toward −)
            last_hi = src[x, y, z, b_idx]   # toward +
            for m in Int32(1):ntap
                gtap = _BSPLINE_GREEN_HALF[m + Int32(1)]
                # − side
                vlo = _bspline_read_axis(src, block_pointer, gx, gy, gz, axis, -m,
                                         bx_max, by_max, bz_max, block_size, g_self, n_global_axis)
                vlo = (vlo == vlo) ? vlo : last_lo   # NaN sentinel → clamp
                last_lo = vlo
                # + side
                vhi = _bspline_read_axis(src, block_pointer, gx, gy, gz, axis, m,
                                         bx_max, by_max, bz_max, block_size, g_self, n_global_axis)
                vhi = (vhi == vhi) ? vhi : last_hi
                last_hi = vhi
                acc += gtap * (vlo + vhi)
            end
            dst[x, y, z, b_idx] = acc
        end
    end
end

# Read a scalar at global-axis offset `off` along `axis` from this cell; returns
# NaN32 if the target block is inactive/out-of-range (caller clamps).
@inline function _bspline_read_axis(src, block_pointer, gx::Int32, gy::Int32, gz::Int32,
                                    axis::Int32, off::Int32,
                                    bx_max::Int32, by_max::Int32, bz_max::Int32,
                                    block_size::Int32, g_self::Int32, n_global_axis::Int32)
    @inbounds begin
        tgx = axis == Int32(1) ? gx + off : gx
        tgy = axis == Int32(2) ? gy + off : gy
        tgz = axis == Int32(3) ? gz + off : gz
        # bounds on the active axis (others are within-block by construction)
        tg = g_self + off
        if _PERIODIC_Y_AMR
            # periodic-Y AMR: the Y prefilter line wraps across the seam (the
            # periodic B-spline prefilter = circulant inverse, image-summed taps)
            tgy = _pwrap_y(tgy, by_max * block_size)
            axis == Int32(2) && (tg = tgy)
        end
        if tg < Int32(1) || tg > n_global_axis
            return NaN32
        end
        pbx = (tgx - Int32(1)) ÷ block_size + Int32(1)
        pby = (tgy - Int32(1)) ÷ block_size + Int32(1)
        pbz = (tgz - Int32(1)) ÷ block_size + Int32(1)
        if pbx < Int32(1) || pbx > bx_max || pby < Int32(1) || pby > by_max || pbz < Int32(1) || pbz > bz_max
            return NaN32
        end
        pb = block_pointer[pbx, pby, pbz]
        if pb <= Int32(0)
            return NaN32
        end
        plx = (tgx - Int32(1)) % block_size + Int32(1)
        ply = (tgy - Int32(1)) % block_size + Int32(1)
        plz = (tgz - Int32(1)) % block_size + Int32(1)
        return src[plx, ply, plz, pb]
    end
end

# ── FUSED 4-FIELD PREFILTER PASS (2026-06-11, 100-MLUPS item 2) ──────────────
# The scalar prefilter runs 12 launches per compute_spline_coeffs! call (ρ + 3
# velocity components × 3 axis passes), and EVERY tap of every field repeats
# the identical per-tap address work (axis bounds, block coords, block_pointer,
# wrap). spline_prefilter measured 17% of the production batch (Z155 ledger).
# This fused kernel does one axis pass for ALL FOUR fields at once: the tap
# address is resolved ONCE and 4 values are gathered → 12 → 3 launches and ~4×
# less address ALU. Tap validity is purely ADDRESS-based (NaN only arises from
# out-of-axis/out-of-grid/inactive-block — never from data), so the per-field
# zero-gradient clamp chain (last_lo/last_hi) sees exactly the same
# valid/invalid pattern per tap as the scalar kernels → BITWISE-identical
# per-field results by construction (same Green taps, same accumulation order).
# Gate: LUDWIG_D3Q19_SPLINE_FUSED (default ON since 2026-06-11: Z156 GPU A/B
# bitwise-identical seam CSVs AND spline_prefilter 3.53s → 2.06s = 1.7× — see
# CASE_LOGS/z156_splinefused_on.log vs z156_svvtile_off.log; "0" reverts to
# the per-field scalar passes).
const _D3Q19_SPLINE_FUSED = get(ENV, "LUDWIG_D3Q19_SPLINE_FUSED", "1") != "0"

@kernel function _bspline_prefilter_axis_fused4!(
    dst_rho, src_rho, dst_vel, src_vel, block_pointer,
    active_coords_x, active_coords_y, active_coords_z,
    bx_max::Int32, by_max::Int32, bz_max::Int32,
    n_blocks::Int32, block_size::Int32, axis::Int32, ntap::Int32
)
    x, y, z, b_idx = @index(Global, NTuple)
    if b_idx <= n_blocks
        @inbounds begin
            bx = active_coords_x[b_idx]
            by = active_coords_y[b_idx]
            bz = active_coords_z[b_idx]
            gx = (bx - Int32(1)) * block_size + Int32(x)
            gy = (by - Int32(1)) * block_size + Int32(y)
            gz = (bz - Int32(1)) * block_size + Int32(z)
            g_self = axis == Int32(1) ? gx : (axis == Int32(2) ? gy : gz)
            n_global_axis = (axis == Int32(1) ? bx_max : (axis == Int32(2) ? by_max : bz_max)) * block_size

            # center tap (offset 0) — 4 fields
            c_r = src_rho[x, y, z, b_idx]
            c_u = src_vel[x, y, z, b_idx, 1]
            c_v = src_vel[x, y, z, b_idx, 2]
            c_w = src_vel[x, y, z, b_idx, 3]
            g0 = _BSPLINE_GREEN_HALF[1]
            acc_r = g0 * c_r; acc_u = g0 * c_u; acc_v = g0 * c_v; acc_w = g0 * c_w
            llo_r = c_r; llo_u = c_u; llo_v = c_v; llo_w = c_w   # zero-gradient clamp seeds
            lhi_r = c_r; lhi_u = c_u; lhi_v = c_v; lhi_w = c_w
            for m in Int32(1):ntap
                gtap = _BSPLINE_GREEN_HALF[m + Int32(1)]
                # ── − side: tap ADDRESS resolved once for all 4 fields ──
                ok_lo = false
                pb_lo = Int32(0); lx_lo = Int32(1); ly_lo = Int32(1); lz_lo = Int32(1)
                begin
                    tgx = axis == Int32(1) ? gx - m : gx
                    tgy = axis == Int32(2) ? gy - m : gy
                    tgz = axis == Int32(3) ? gz - m : gz
                    tg = g_self - m
                    if _PERIODIC_Y_AMR   # periodic-Y AMR: wrap the Y prefilter line
                        tgy = _pwrap_y(tgy, by_max * block_size)
                        axis == Int32(2) && (tg = tgy)
                    end
                    if !(tg < Int32(1) || tg > n_global_axis)
                        pbx = (tgx - Int32(1)) ÷ block_size + Int32(1)
                        pby = (tgy - Int32(1)) ÷ block_size + Int32(1)
                        pbz = (tgz - Int32(1)) ÷ block_size + Int32(1)
                        if !(pbx < Int32(1) || pbx > bx_max || pby < Int32(1) || pby > by_max ||
                             pbz < Int32(1) || pbz > bz_max)
                            pb_lo = block_pointer[pbx, pby, pbz]
                            if pb_lo > Int32(0)
                                lx_lo = (tgx - Int32(1)) % block_size + Int32(1)
                                ly_lo = (tgy - Int32(1)) % block_size + Int32(1)
                                lz_lo = (tgz - Int32(1)) % block_size + Int32(1)
                                ok_lo = true
                            end
                        end
                    end
                end
                # ── + side ──
                ok_hi = false
                pb_hi = Int32(0); lx_hi = Int32(1); ly_hi = Int32(1); lz_hi = Int32(1)
                begin
                    tgx = axis == Int32(1) ? gx + m : gx
                    tgy = axis == Int32(2) ? gy + m : gy
                    tgz = axis == Int32(3) ? gz + m : gz
                    tg = g_self + m
                    if _PERIODIC_Y_AMR   # periodic-Y AMR: wrap the Y prefilter line
                        tgy = _pwrap_y(tgy, by_max * block_size)
                        axis == Int32(2) && (tg = tgy)
                    end
                    if !(tg < Int32(1) || tg > n_global_axis)
                        pbx = (tgx - Int32(1)) ÷ block_size + Int32(1)
                        pby = (tgy - Int32(1)) ÷ block_size + Int32(1)
                        pbz = (tgz - Int32(1)) ÷ block_size + Int32(1)
                        if !(pbx < Int32(1) || pbx > bx_max || pby < Int32(1) || pby > by_max ||
                             pbz < Int32(1) || pbz > bz_max)
                            pb_hi = block_pointer[pbx, pby, pbz]
                            if pb_hi > Int32(0)
                                lx_hi = (tgx - Int32(1)) % block_size + Int32(1)
                                ly_hi = (tgy - Int32(1)) % block_size + Int32(1)
                                lz_hi = (tgz - Int32(1)) % block_size + Int32(1)
                                ok_hi = true
                            end
                        end
                    end
                end
                # ── gather + accumulate, EXACT scalar-kernel expression order:
                #    vlo then vhi per field, acc += gtap * (vlo + vhi) ──
                vlo_r = ok_lo ? src_rho[lx_lo, ly_lo, lz_lo, pb_lo]    : llo_r; llo_r = vlo_r
                vhi_r = ok_hi ? src_rho[lx_hi, ly_hi, lz_hi, pb_hi]    : lhi_r; lhi_r = vhi_r
                acc_r += gtap * (vlo_r + vhi_r)
                vlo_u = ok_lo ? src_vel[lx_lo, ly_lo, lz_lo, pb_lo, 1] : llo_u; llo_u = vlo_u
                vhi_u = ok_hi ? src_vel[lx_hi, ly_hi, lz_hi, pb_hi, 1] : lhi_u; lhi_u = vhi_u
                acc_u += gtap * (vlo_u + vhi_u)
                vlo_v = ok_lo ? src_vel[lx_lo, ly_lo, lz_lo, pb_lo, 2] : llo_v; llo_v = vlo_v
                vhi_v = ok_hi ? src_vel[lx_hi, ly_hi, lz_hi, pb_hi, 2] : lhi_v; lhi_v = vhi_v
                acc_v += gtap * (vlo_v + vhi_v)
                vlo_w = ok_lo ? src_vel[lx_lo, ly_lo, lz_lo, pb_lo, 3] : llo_w; llo_w = vlo_w
                vhi_w = ok_hi ? src_vel[lx_hi, ly_hi, lz_hi, pb_hi, 3] : lhi_w; lhi_w = vhi_w
                acc_w += gtap * (vlo_w + vhi_w)
            end
            dst_rho[x, y, z, b_idx]    = acc_r
            dst_vel[x, y, z, b_idx, 1] = acc_u
            dst_vel[x, y, z, b_idx, 2] = acc_v
            dst_vel[x, y, z, b_idx, 3] = acc_w
        end
    end
end

# ── LINE-TILE FUSED PREFILTER (2026-06-12, MLUPS phase 1 lever 3) ────────────
# The fused4 kernel above still gathers 48 tap addresses × 4 fields PER CELL
# (~196 global reads/cell). All 8 cells of a block row along the pass axis
# share the same 56-cell tap line (8 + 2·24), so a workgroup loads the
# 56 × 16-lines × 4-fields tile (14.3 KB @localmem) ONCE — ~7× fewer global
# reads and the per-tap address math collapses from per-(cell,tap) to
# per-line-cell. The per-cell scan below is the fused4 code VERBATIM with the
# gather source swapped to the tile (same Green taps, same clamp chain, same
# accumulation order). Tap validity stays address-based: invalid line cells
# hold NaN in the tile (field-1 NaN ⇒ all four NaN).
# Workgroups: axes 1,2 → (8,8,2,1); axis 3 → (8,2,8,1) (the pass axis needs
# the full 8-cell local extent). ntap must equal 24 (static tile width 56) —
# the driver guards this. GATES PASSED 2026-06-12: CPU bitwise vs fused4 all
# 3 axes (tools/test_spline_line_tile.jl); GPU BITWISE at Z160 (steps 30+60,
# XZ+YZ) AND Z197 production (step 80). Timing (Z197 SR=235): spline_prefilter
# 4.0-4.3 → 1.31 s/batch (3.1×). DEFAULT ON; "0" reverts to flat fused4.
const _D3Q19_SPLINE_LINE_TILE = get(ENV, "LUDWIG_D3Q19_SPLINE_LINE_TILE", "1") != "0"

@kernel function _bspline_prefilter_axis_fused4_tile!(
    dst_rho, src_rho, dst_vel, src_vel, block_pointer,
    active_coords_x, active_coords_y, active_coords_z,
    bx_max::Int32, by_max::Int32, bz_max::Int32,
    n_blocks::Int32, block_size::Int32, axis::Int32, ntap::Int32
)
    x, y, z, b_local = @index(Global, NTuple)
    lx, ly, lz, _lb = @index(Local, NTuple)
    tile = @localmem Float32 (56, 16, 4)

    # ── Cooperative load: 16 lines × 56 cells × 4 fields, addresses via the
    # same _bspline_read_axis logic (NaN for out-of-axis/grid/inactive).
    # All derived locals are recomputed after the barrier (KA CPU rule).
    b_idx = Int32(b_local)
    if b_idx <= n_blocks
        @inbounds begin
            bx = active_coords_x[b_idx]
            by = active_coords_y[b_idx]
            bz = active_coords_z[b_idx]
            # local→(pos along axis, line_id) decomposition + this thread's lid
            lid = Int32(lx) + (Int32(ly) - Int32(1)) * Int32(8) +
                  (Int32(lz) - Int32(1)) * (axis == Int32(3) ? Int32(16) : Int32(64))
            # block-global origin of the slab the workgroup covers (the two
            # non-axis coordinates of line L are derived per entry below)
            gx0 = (bx - Int32(1)) * block_size
            gy0 = (by - Int32(1)) * block_size
            gz0 = (bz - Int32(1)) * block_size
            # DOMAIN-GLOBAL origin of the partial (2-wide) local slab dim:
            # block origin + (within-block index of local 0). @index(Global)
            # here is the within-block cell index (ndrange dims 1-3 = 8).
            zslab0 = gz0 + Int32(z) - Int32(lz)  # domain-global z of local lz=0 (axes 1,2)
            yslab0 = gy0 + Int32(y) - Int32(ly)  # domain-global y of local ly=0 (axis 3)
            n_global_axis = (axis == Int32(1) ? bx_max : (axis == Int32(2) ? by_max : bz_max)) * block_size
            t = lid
            while t <= Int32(896)                # 56 · 16 entries
                line = (t - Int32(1)) ÷ Int32(56) + Int32(1)   # 1..16
                j    = (t - Int32(1)) % Int32(56) + Int32(1)   # 1..56 along the axis
                la = (line - Int32(1)) % Int32(8) + Int32(1)   # 1..8  (first cross dim)
                lb = (line - Int32(1)) ÷ Int32(8) + Int32(1)   # 1..2  (second cross dim)
                # base cell (pos=1) of this line + axis offset (j-25 ∈ -24..31)
                off = j - Int32(25)
                if axis == Int32(1)
                    cgx = gx0 + Int32(1); cgy = gy0 + la; cgz = zslab0 + lb
                    v = _bspline_read_axis(src_rho, block_pointer, cgx, cgy, cgz, axis, off,
                                           bx_max, by_max, bz_max, block_size, cgx, n_global_axis)
                elseif axis == Int32(2)
                    cgx = gx0 + la; cgy = gy0 + Int32(1); cgz = zslab0 + lb
                    v = _bspline_read_axis(src_rho, block_pointer, cgx, cgy, cgz, axis, off,
                                           bx_max, by_max, bz_max, block_size, cgy, n_global_axis)
                else
                    cgx = gx0 + la; cgy = yslab0 + lb; cgz = gz0 + Int32(1)
                    v = _bspline_read_axis(src_rho, block_pointer, cgx, cgy, cgz, axis, off,
                                           bx_max, by_max, bz_max, block_size, cgz, n_global_axis)
                end
                tile[j, line, 1] = v
                if v == v
                    # resolve once more for the vel fields at the same address
                    tgx = axis == Int32(1) ? cgx + off : cgx
                    tgy = _pwrap_y(axis == Int32(2) ? cgy + off : cgy, by_max * block_size)   # periodic-Y AMR (identity when off)
                    tgz = axis == Int32(3) ? cgz + off : cgz
                    pb = block_pointer[(tgx - Int32(1)) ÷ block_size + Int32(1),
                                       (tgy - Int32(1)) ÷ block_size + Int32(1),
                                       (tgz - Int32(1)) ÷ block_size + Int32(1)]
                    plx = (tgx - Int32(1)) % block_size + Int32(1)
                    ply = (tgy - Int32(1)) % block_size + Int32(1)
                    plz = (tgz - Int32(1)) % block_size + Int32(1)
                    tile[j, line, 2] = src_vel[plx, ply, plz, pb, 1]
                    tile[j, line, 3] = src_vel[plx, ply, plz, pb, 2]
                    tile[j, line, 4] = src_vel[plx, ply, plz, pb, 3]
                else
                    tile[j, line, 2] = NaN32
                    tile[j, line, 3] = NaN32
                    tile[j, line, 4] = NaN32
                end
                t += Int32(128)
            end
        end
    end
    @synchronize

    # ── Per-cell scan: fused4 VERBATIM, gather from the tile ──
    b_idx = Int32(b_local)
    if b_idx <= n_blocks
        @inbounds begin
            pos  = axis == Int32(1) ? Int32(lx) : (axis == Int32(2) ? Int32(ly) : Int32(lz))
            line = axis == Int32(1) ? (Int32(ly) + (Int32(lz) - Int32(1)) * Int32(8)) :
                   axis == Int32(2) ? (Int32(lx) + (Int32(lz) - Int32(1)) * Int32(8)) :
                                      (Int32(lx) + (Int32(ly) - Int32(1)) * Int32(8))
            jc = pos + Int32(24)                 # center index in the 56-line

            c_r = tile[jc, line, 1]
            c_u = tile[jc, line, 2]
            c_v = tile[jc, line, 3]
            c_w = tile[jc, line, 4]
            g0 = _BSPLINE_GREEN_HALF[1]
            acc_r = g0 * c_r; acc_u = g0 * c_u; acc_v = g0 * c_v; acc_w = g0 * c_w
            llo_r = c_r; llo_u = c_u; llo_v = c_v; llo_w = c_w
            lhi_r = c_r; lhi_u = c_u; lhi_v = c_v; lhi_w = c_w
            for m in Int32(1):ntap
                gtap = _BSPLINE_GREEN_HALF[m + Int32(1)]
                tlo = tile[jc - m, line, 1]
                thi = tile[jc + m, line, 1]
                ok_lo = tlo == tlo
                ok_hi = thi == thi
                vlo_r = ok_lo ? tlo                     : llo_r; llo_r = vlo_r
                vhi_r = ok_hi ? thi                     : lhi_r; lhi_r = vhi_r
                acc_r += gtap * (vlo_r + vhi_r)
                vlo_u = ok_lo ? tile[jc - m, line, 2]   : llo_u; llo_u = vlo_u
                vhi_u = ok_hi ? tile[jc + m, line, 2]   : lhi_u; lhi_u = vhi_u
                acc_u += gtap * (vlo_u + vhi_u)
                vlo_v = ok_lo ? tile[jc - m, line, 3]   : llo_v; llo_v = vlo_v
                vhi_v = ok_hi ? tile[jc + m, line, 3]   : lhi_v; lhi_v = vhi_v
                acc_v += gtap * (vlo_v + vhi_v)
                vlo_w = ok_lo ? tile[jc - m, line, 4]   : llo_w; llo_w = vlo_w
                vhi_w = ok_hi ? tile[jc + m, line, 4]   : lhi_w; lhi_w = vhi_w
                acc_w += gtap * (vlo_w + vhi_w)
            end
            dst_rho[x, y, z, b_idx]    = acc_r
            dst_vel[x, y, z, b_idx, 1] = acc_u
            dst_vel[x, y, z, b_idx, 2] = acc_v
            dst_vel[x, y, z, b_idx, 3] = acc_w
        end
    end
end

# (D3Q19 LEVER-1b fill_halo_hoist_cache! driver removed 2026-06-24 with the D3Q19 retirement.)

# Driver: compute B-spline coefficients for a level's rho + 3 velocity components.
# Ping-pongs coeff↔scratch through the 3 separable axis passes. After this,
# level.rho_spline_coeff / level.vel_spline_coeff hold the coefficients.
function compute_spline_coeffs!(level, vel_src, backend)
    n_blocks = Int32(length(level.active_block_coords))
    (n_blocks == Int32(0) || !has_spline_coeff(level)) && return
    bs = Int32(BLOCK_SIZE)
    bx_max = Int32(size(level.block_pointer, 1))
    by_max = Int32(size(level.block_pointer, 2))
    bz_max = Int32(size(level.block_pointer, 3))
    ntap = _BSPLINE_GREEN_NTAP
    nd = (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, Int(n_blocks))

    if _D3Q19_SPLINE_FUSED
        if _D3Q19_SPLINE_LINE_TILE && ntap == Int32(24)
            # LINE-TILE path (lever 3): same 3 fused launches, gathers from a
            # cooperative 56×16×4 @localmem line tile. Workgroup must give the
            # pass axis its full 8-cell local extent: (8,8,2) for x/y, (8,2,8)
            # for z. ntap==24 is required by the static tile width (56).
            kxy = _bspline_prefilter_axis_fused4_tile!(backend, (8, 8, 2, 1))
            kz  = _bspline_prefilter_axis_fused4_tile!(backend, (8, 2, 8, 1))
            kxy(level.rho_spline_coeff, level.rho,
                level.vel_spline_coeff, vel_src, level.block_pointer,
                level.map_x, level.map_y, level.map_z, bx_max, by_max, bz_max,
                n_blocks, bs, Int32(1), ntap; ndrange=nd)
            kxy(level.rho_spline_scratch, level.rho_spline_coeff,
                level.vel_spline_scratch, level.vel_spline_coeff, level.block_pointer,
                level.map_x, level.map_y, level.map_z, bx_max, by_max, bz_max,
                n_blocks, bs, Int32(2), ntap; ndrange=nd)
            kz(level.rho_spline_coeff, level.rho_spline_scratch,
               level.vel_spline_coeff, level.vel_spline_scratch, level.block_pointer,
               level.map_x, level.map_y, level.map_z, bx_max, by_max, bz_max,
               n_blocks, bs, Int32(3), ntap; ndrange=nd)
            return
        end
        # FUSED path: 3 launches (one per axis), 4 fields each — same
        # final-lands-in-coeff src≠dst ordering as the scalar path below.
        kernf = _bspline_prefilter_axis_fused4!(backend)
        kernf(level.rho_spline_coeff, level.rho,
              level.vel_spline_coeff, vel_src, level.block_pointer,
              level.map_x, level.map_y, level.map_z, bx_max, by_max, bz_max,
              n_blocks, bs, Int32(1), ntap; ndrange=nd)
        kernf(level.rho_spline_scratch, level.rho_spline_coeff,
              level.vel_spline_scratch, level.vel_spline_coeff, level.block_pointer,
              level.map_x, level.map_y, level.map_z, bx_max, by_max, bz_max,
              n_blocks, bs, Int32(2), ntap; ndrange=nd)
        kernf(level.rho_spline_coeff, level.rho_spline_scratch,
              level.vel_spline_coeff, level.vel_spline_scratch, level.block_pointer,
              level.map_x, level.map_y, level.map_z, bx_max, by_max, bz_max,
              n_blocks, bs, Int32(3), ntap; ndrange=nd)
        return
    end

    kern = _bspline_prefilter_axis_scalar!(backend)

    # ρ: 3 separable passes. PERF (2026-06-04): ordered so the FINAL pass lands
    # directly in coeff — eliminates the copyto! + the host synchronize that the
    # old `…→scratch then copy` form needed. Each pass has distinct src≠dst:
    # pass x: rho → coeff ; pass y: coeff → scratch ; pass z: scratch → coeff.
    # KernelAbstractions auto-orders these dependent same-backend kernels, so NO
    # explicit host sync is required between them (a single sync happens later at
    # the natural batch boundary). Byte-identical result (same 3-pass convolution).
    kern(level.rho_spline_coeff, level.rho, level.block_pointer,
         level.map_x, level.map_y, level.map_z, bx_max, by_max, bz_max,
         n_blocks, bs, Int32(1), ntap; ndrange=nd)
    kern(level.rho_spline_scratch, level.rho_spline_coeff, level.block_pointer,
         level.map_x, level.map_y, level.map_z, bx_max, by_max, bz_max,
         n_blocks, bs, Int32(2), ntap; ndrange=nd)
    kern(level.rho_spline_coeff, level.rho_spline_scratch, level.block_pointer,
         level.map_x, level.map_y, level.map_z, bx_max, by_max, bz_max,
         n_blocks, bs, Int32(3), ntap; ndrange=nd)

    # velocity components: convolve each of the 3 components. Same final-lands-in-
    # coeff ordering — no copyto!, no per-component sync.
    for comp in 1:3
        vc = @view level.vel_spline_coeff[:, :, :, :, comp]
        vs = @view level.vel_spline_scratch[:, :, :, :, comp]
        vsrc_c = @view vel_src[:, :, :, :, comp]
        kern(vc, vsrc_c, level.block_pointer,
             level.map_x, level.map_y, level.map_z, bx_max, by_max, bz_max,
             n_blocks, bs, Int32(1), ntap; ndrange=nd)
        kern(vs, vc, level.block_pointer,
             level.map_x, level.map_y, level.map_z, bx_max, by_max, bz_max,
             n_blocks, bs, Int32(2), ntap; ndrange=nd)
        kern(vc, vs, level.block_pointer,
             level.map_x, level.map_y, level.map_z, bx_max, by_max, bz_max,
             n_blocks, bs, Int32(3), ntap; ndrange=nd)
    end
    return
end

# ══════════════════════════════════════════════════════════════════════════════
# C2/C4 B-SPLINE FINE EVALUATION (2026-06-04)
# ------------------------------------------------------------------------------
# Given the parent's precomputed B-spline coefficient arrays (rho_spline_coeff,
# vel_spline_coeff — filled by compute_spline_coeffs!), evaluate the fine cell's
# macro values as the compact 6-tap product f = Σ_ijk W_a(i)W_b(j)W_c(k)·c, where
# (a,b,c) ∈ {low,high} per axis by the fine sub-position. C4-continuous + exactly
# 2:1-conservative. Reads coeffs at parent GLOBAL cells via parent_ptr (like
# get_macro). Returns (rho,ux,uy,uz, ok); ok=false if any tap hits an inactive
# parent block (caller falls back).
#
# Read one coeff scalar at parent global cell (pgx,pgy,pgz); NaN32 if inactive.
@inline function _bspline_coeff_read(coeff_arr, parent_ptr, pgx::Int32, pgy::Int32, pgz::Int32,
                                     parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
                                     block_size::Int32)
    @inbounds begin
        pgy = _pwrap_y(pgy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
        # 2026-10-27: ÷ truncates toward zero, so a global index -6..0 (below the domain's
        # lower x/y/z face) mapped to block 1 with a local index <= 0: an out-of-bounds read.
        (pgx < Int32(1) || pgy < Int32(1) || pgz < Int32(1)) && return NaN32
        pbx = (pgx - Int32(1)) ÷ block_size + Int32(1)
        pby = (pgy - Int32(1)) ÷ block_size + Int32(1)
        pbz = (pgz - Int32(1)) ÷ block_size + Int32(1)
        (pbx < Int32(1) || pbx > parent_dim_x || pby < Int32(1) || pby > parent_dim_y ||
         pbz < Int32(1) || pbz > parent_dim_z) && return NaN32
        pb = parent_ptr[pbx, pby, pbz]
        pb <= Int32(0) && return NaN32
        plx = (pgx - Int32(1)) % block_size + Int32(1)
        ply = (pgy - Int32(1)) % block_size + Int32(1)
        plz = (pgz - Int32(1)) % block_size + Int32(1)
        return coeff_arr[plx, ply, plz, pb]
    end
end

# Per-axis: parent center cell p0 and the 6 W-taps (low/high) for a fine global coord.
@inline function _bspline_axis_setup(fine_g::Int32)
    p0 = (fine_g + Int32(1)) ÷ Int32(2)          # parent global cell containing this fine cell
    sub_low = isodd(fine_g)                        # true → lower sub-cell (use W_LOW), else W_HIGH
    return p0, sub_low
end

@inline function _bspline_symmetric_support_ready(parent_ptr,
                                                 p0x::Int32, p0y::Int32, p0z::Int32,
                                                 parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
                                                 block_size::Int32)
    # 2026-10-27: `div` truncates toward zero, so a tap at global index -6..0 (below the
    # domain's lower x/y/z face) counted as block 1 and the support passed: the
    # coefficient read then indexed a local cell <= 0 (out of bounds). Lower bound explicit.
    @inbounds for iz in Int32(-3):Int32(3)
        pgz = p0z + iz
        pgz >= Int32(1) || return false
        pbz = div(pgz - Int32(1), block_size) + Int32(1)
        (pbz >= Int32(1) && pbz <= parent_dim_z) || return false
        for iy in Int32(-3):Int32(3)
            pgy = _pwrap_y(p0y + iy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
            pgy >= Int32(1) || return false
            pby = div(pgy - Int32(1), block_size) + Int32(1)
            (pby >= Int32(1) && pby <= parent_dim_y) || return false
            for ix in Int32(-3):Int32(3)
                pgx = p0x + ix
                pgx >= Int32(1) || return false
                pbx = div(pgx - Int32(1), block_size) + Int32(1)
                (pbx >= Int32(1) && pbx <= parent_dim_x) || return false
                parent_ptr[pbx, pby, pbz] > Int32(0) || return false
            end
        end
    end
    return true
end

# Evaluate a single scalar coeff field at the fine cell via the 6×6×6 W tensor.
# w_off_a etc. encode the per-axis offset start (-3 for low, -2 for high).
@inline function _bspline_eval_scalar(coeff_arr, parent_ptr,
                                      p0x::Int32, p0y::Int32, p0z::Int32,
                                      lowx::Bool, lowy::Bool, lowz::Bool,
                                      parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
                                      block_size::Int32)
    @inbounds begin
        offx0 = lowx ? Int32(-3) : Int32(-2)
        offy0 = lowy ? Int32(-3) : Int32(-2)
        offz0 = lowz ? Int32(-3) : Int32(-2)
        acc = 0.0f0
        for iz in Int32(0):Int32(5)
            wz = (lowz ? _BSPLINE_W_LOW_NUM[iz+Int32(1)] : _BSPLINE_W_HIGH_NUM[iz+Int32(1)])
            pgz = p0z + offz0 + iz
            for iy in Int32(0):Int32(5)
                wy = (lowy ? _BSPLINE_W_LOW_NUM[iy+Int32(1)] : _BSPLINE_W_HIGH_NUM[iy+Int32(1)])
                pgy = _pwrap_y(p0y + offy0 + iy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
                wyz = Float32(wy) * Float32(wz)
                for ix in Int32(0):Int32(5)
                    wx = (lowx ? _BSPLINE_W_LOW_NUM[ix+Int32(1)] : _BSPLINE_W_HIGH_NUM[ix+Int32(1)])
                    pgx = p0x + offx0 + ix
                    v = _bspline_coeff_read(coeff_arr, parent_ptr, pgx, pgy, pgz,
                                            parent_dim_x, parent_dim_y, parent_dim_z, block_size)
                    (v == v) || return (NaN32, false)   # inactive tap → bail
                    acc += Float32(wx) * wyz * v
                end
            end
        end
        return (acc / (_BSPLINE_W_DENOM_F * _BSPLINE_W_DENOM_F * _BSPLINE_W_DENOM_F), true)
    end
end

# High-level: evaluate (rho,ux,uy,uz) for a fine cell from parent spline coeffs.
@noinline function interpolate_macro_bspline(
    rho_coeff, vel_coeff, parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    fine_gx::Int32, fine_gy::Int32, fine_gz::Int32, block_size::Int32)
    @inbounds begin
        p0x, lowx = _bspline_axis_setup(fine_gx)
        p0y, lowy = _bspline_axis_setup(fine_gy)
        p0z, lowz = _bspline_axis_setup(fine_gz)
        # Sparse AMR corners used to make low/high fine subcells take different
        # paths because their 6-tap windows are offset by one parent cell. Require
        # symmetric support around p0 so corner cells either all use B-spline or
        # all fall back to the conservative partial-stencil prolongation.
        _bspline_symmetric_support_ready(parent_ptr, p0x, p0y, p0z,
                                         parent_dim_x, parent_dim_y, parent_dim_z,
                                         block_size) || return (1.0f0, 0.0f0, 0.0f0, 0.0f0, false)
        r, okr = _bspline_eval_scalar(rho_coeff, parent_ptr, p0x,p0y,p0z, lowx,lowy,lowz,
                                      parent_dim_x,parent_dim_y,parent_dim_z, block_size)
        okr || return (1.0f0, 0.0f0, 0.0f0, 0.0f0, false)
        # velocity: 3 components from vel_coeff[:,:,:,:,c]
        vx, okx = _bspline_eval_vel(vel_coeff, parent_ptr, p0x,p0y,p0z, lowx,lowy,lowz,
                                    parent_dim_x,parent_dim_y,parent_dim_z, block_size, Int32(1))
        vy, oky = _bspline_eval_vel(vel_coeff, parent_ptr, p0x,p0y,p0z, lowx,lowy,lowz,
                                    parent_dim_x,parent_dim_y,parent_dim_z, block_size, Int32(2))
        vz, okz = _bspline_eval_vel(vel_coeff, parent_ptr, p0x,p0y,p0z, lowx,lowy,lowz,
                                    parent_dim_x,parent_dim_y,parent_dim_z, block_size, Int32(3))
        (okx && oky && okz) || return (1.0f0, 0.0f0, 0.0f0, 0.0f0, false)
        return (r, vx, vy, vz, true)
    end
end

# velocity-component variant (reads vel_coeff[...,comp]).
@inline function _bspline_eval_vel(vel_coeff, parent_ptr,
                                   p0x::Int32, p0y::Int32, p0z::Int32,
                                   lowx::Bool, lowy::Bool, lowz::Bool,
                                   parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
                                   block_size::Int32, comp::Int32)
    @inbounds begin
        offx0 = lowx ? Int32(-3) : Int32(-2)
        offy0 = lowy ? Int32(-3) : Int32(-2)
        offz0 = lowz ? Int32(-3) : Int32(-2)
        acc = 0.0f0
        for iz in Int32(0):Int32(5)
            wz = (lowz ? _BSPLINE_W_LOW_NUM[iz+Int32(1)] : _BSPLINE_W_HIGH_NUM[iz+Int32(1)])
            pgz = p0z + offz0 + iz
            for iy in Int32(0):Int32(5)
                wy = (lowy ? _BSPLINE_W_LOW_NUM[iy+Int32(1)] : _BSPLINE_W_HIGH_NUM[iy+Int32(1)])
                pgy = _pwrap_y(p0y + offy0 + iy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
                wyz = Float32(wy) * Float32(wz)
                for ix in Int32(0):Int32(5)
                    wx = (lowx ? _BSPLINE_W_LOW_NUM[ix+Int32(1)] : _BSPLINE_W_HIGH_NUM[ix+Int32(1)])
                    pgx = p0x + offx0 + ix
                    (pgx < Int32(1) || pgy < Int32(1) || pgz < Int32(1)) && return (NaN32, false)   # 2026-10-27: ÷ truncates toward zero
                    pbx = (pgx - Int32(1)) ÷ block_size + Int32(1)
                    pby = (pgy - Int32(1)) ÷ block_size + Int32(1)
                    pbz = (pgz - Int32(1)) ÷ block_size + Int32(1)
                    if pbx < Int32(1) || pbx > parent_dim_x || pby < Int32(1) || pby > parent_dim_y || pbz < Int32(1) || pbz > parent_dim_z
                        return (NaN32, false)
                    end
                    pb = parent_ptr[pbx, pby, pbz]
                    pb <= Int32(0) && return (NaN32, false)
                    plx = (pgx - Int32(1)) % block_size + Int32(1)
                    ply = (pgy - Int32(1)) % block_size + Int32(1)
                    plz = (pgz - Int32(1)) % block_size + Int32(1)
                    acc += Float32(wx) * wyz * vel_coeff[plx, ply, plz, pb, comp]
                end
            end
        end
        return (acc / (_BSPLINE_W_DENOM_F * _BSPLINE_W_DENOM_F * _BSPLINE_W_DENOM_F), true)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# COLLAPSED C4 B-SPLINE (2026-06-20) — single fused ±21-tap stencil on RAW macros
# ------------------------------------------------------------------------------
# The prefilter+6-tap pair above (compute_spline_coeffs! → interpolate_macro_bspline)
# is the *two-stage* form of the C4 B-spline prolongation: a once-per-coarse-step
# heptadiagonal deconvolution into coefficient arrays, then a compact 6-tap eval.
# Because BOTH stages are linear, they FUSE into a single convolution kernel
# K = W ⊛ A⁻¹ (the _BSPLINE_K_LOW / _BSPLINE_K_HIGH constants at the top of this
# file, offline-verified EXACT: Σ K = 1, (K_low+K_high)/2 = δ to 2e-16, reproduces
# x/x²/x³ to round-off, C4 by construction). Applying K directly to the RAW parent
# macros:
#   • produces the SAME C4 fine-cell value as prefilter+6-tap (to FP32 summation
#     order), so it is true-C4 on EVERY substep — no tw<0.99 trilinear downgrade;
#   • needs NO prefilter pass (deletes compute_spline_coeffs!, ~7.9 s/batch); and
#   • needs NO coefficient arrays (rho/vel_spline_coeff + scratch → reclaimable
#     VRAM), which compounds with the ET single-buffer VRAM win.
#
# COST: the naive separable tensor form is 43³ raw reads per fine cell — far too
# much PER SUBSTEP. This per-cell evaluator exists to (a) PROVE GPU-JIT-safety +
# C4 correctness on the thin halo shell, and (b) be the kernel that the
# once-per-coarse-step HOIST (Lever 1b) lifts out of the substep loop so the 43³
# convolution runs ONCE per coarse step into a cached halo field, after which each
# substep is a scalar temporal lerp. Off by default ⇒ byte-identical.
#
# The 1D K stencil is applied as 3 separable reductions (z, then y, then x). Per
# axis the fine sub-cell parity (isodd(fine_g) → lower sub-cell) selects K_LOW;
# the upper sub-cell uses K_HIGH = reverse(K_LOW). The 1D tap offset relative to
# the parent cell p0 (the parent cell CONTAINING the fine cell) runs -21..+21 with
# the center tap (index 22, value 1.0) landing on p0.

# Per-axis parent center cell p0 and lower/upper sub-cell flag for the collapsed
# stencil. p0 = the parent global cell that contains fine cell `fine_g` (2:1).
@inline function _collapsed_axis_setup(fine_g::Int32)
    p0 = (fine_g + Int32(1)) ÷ Int32(2)
    low = isodd(fine_g)                 # true → lower sub-cell → K_LOW; else K_HIGH
    return p0, low
end

# Read the i-th tap weight (i ∈ 1..43, center = 22) for the lower/upper sub-cell.
# Upper sub-cell mirrors: K_HIGH[i] = K_LOW[44 - i].
@inline function _collapsed_tap(i::Int32, low::Bool)
    @inbounds return low ? _BSPLINE_K_LOW[i] : _BSPLINE_K_LOW[Int32(44) - i]
end

# Evaluate ONE raw macro field at the fine cell via the separable 43-tap K stencil.
# `read_raw(pgx,pgy,pgz)` returns the (temporal-blended) raw parent macro at parent
# GLOBAL cell (pgx,pgy,pgz), or NaN32 if that parent cell is inactive/out-of-range.
# Returns (value, ok). ok=false (NaN) on any tap miss → caller falls back.
#
# Separable 3-pass: reduce over x (innermost), then y, then z, with early-NaN
# bail. The center tap carries the dominant weight (1.0); outer taps are < FP32
# ULP beyond ±21 and are truncated by the stencil width (_BSPLINE_K_HALFWIDTH).
@inline function _collapsed_eval_field(read_raw,
                                       p0x::Int32, p0y::Int32, p0z::Int32,
                                       lowx::Bool, lowy::Bool, lowz::Bool)
    hw = _BSPLINE_K_HALFWIDTH
    acc = 0.0f0
    @inbounds for jz in -hw:hw
        wz = _collapsed_tap(jz + hw + Int32(1), lowz)
        pgz = p0z + jz
        accyz = 0.0f0
        for jy in -hw:hw
            wy = _collapsed_tap(jy + hw + Int32(1), lowy)
            pgy = p0y + jy
            accx = 0.0f0
            for jx in -hw:hw
                wx = _collapsed_tap(jx + hw + Int32(1), lowx)
                pgx = p0x + jx
                v = read_raw(pgx, pgy, pgz)
                (v == v) || return (NaN32, false)   # inactive tap → bail to fallback
                accx += wx * v
            end
            accyz += wy * accx
        end
        acc += wz * accyz
    end
    return (acc, true)
end

# Whether the FULL ±21 stencil support is in-range and active for parent cell p0.
# The collapsed stencil has NO partial-stencil fallback of its own (unlike the
# trilinear path) — if any tap in the 43³ window is inactive we return ok=false so
# the caller (sample_parent_macro_collapsed) drops to the existing trilinear/
# B-spline path. In practice the outer taps beyond ~±6 are < FP32 ULP, so a
# tighter "effective support" check would suffice, but the conservative full-width
# check keeps the result bit-faithful to the offline-verified stencil.
@inline function _collapsed_support_ready(parent_ptr,
                                          p0x::Int32, p0y::Int32, p0z::Int32,
                                          parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
                                          block_size::Int32)
    hw = _BSPLINE_K_HALFWIDTH
    @inbounds for jz in -hw:hw
        pgz = p0z + jz
        pgz >= Int32(1) || return false   # 2026-10-27: explicit lower bound (div truncates toward zero)
        pbz = div(pgz - Int32(1), block_size) + Int32(1)
        (pbz >= Int32(1) && pbz <= parent_dim_z) || return false
        for jy in -hw:hw
            pgy = _pwrap_y(p0y + jy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
            pgy >= Int32(1) || return false
            pby = div(pgy - Int32(1), block_size) + Int32(1)
            (pby >= Int32(1) && pby <= parent_dim_y) || return false
            for jx in -hw:hw
                pgx = p0x + jx
                pgx >= Int32(1) || return false
                pbx = div(pgx - Int32(1), block_size) + Int32(1)
                (pbx >= Int32(1) && pbx <= parent_dim_x) || return false
                parent_ptr[pbx, pby, pbz] > Int32(0) || return false
            end
        end
    end
    return true
end
