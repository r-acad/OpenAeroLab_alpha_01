using KernelAbstractions

# AMR-SEAM FIX 1A (2026-06-16): post-collision-consistent f_neq rescaling across
# the AMR interface. The parent/child stored f_neq is POST-collision (carries an
# extra (1-ω) factor), so the (τ-0.5)-ratio transfer alone over-injects stress at
# the interface (the box-edge seam, worst at the τ-clamped finest level). The
# correction multiplies the prolongation scale by (1-ω_coarse)/(1-ω_fine) and the
# restriction ratio by the reciprocal. Default ON; LUDWIG_D3Q19_AMR_FNEQ_POSTCOLL=0
# reverts. Freestream byte-identical (f_neq≡0 ⇒ scale irrelevant).
const _AMR_FNEQ_POSTCOLL =
    get(ENV, "LUDWIG_D3Q19_AMR_FNEQ_POSTCOLL", "1") != "0"

# AMR-SEAM TIER 1 (2026-06-16): cubic-Lagrange f_neq carrier for D3Q19 prolongation.
# The D3Q19 bspline-halo path interpolates the non-equilibrium f_neq TRILINEARLY (C0),
# while D3Q27/D3Q39 use cubic-Lagrange (C1). Lagrava 2012 §3.7: trilinear on f_neq
# injects an O(1/N) PRESSURE JUMP at the refinement interface and breaks LBM's global
# 2nd-order accuracy — this is the visible box-edge seam fraction at physical viscosity.
# When set, the 8-corner trilinear f_neq blend is replaced by a 4×4×4 cubic-Lagrange
# sweep over the SAME per-corner f_neq accessor (_d3q19_parent_corner_neq_quintic),
# matching the D3Q27 carrier order. Default OFF ⇒ byte-identical (trilinear).
# LUDWIG_D3Q19_AMR_FNEQ_CUBIC=1 enables.
const _D3Q19_AMR_FNEQ_CUBIC =
    get(ENV, "LUDWIG_D3Q19_AMR_FNEQ_CUBIC", "0") == "1"

# AMR-CONTINUITY C2 (2026-06-17): cubic-Lagrange stress-tensor (Pi_neq) carrier.
# The PRODUCTION D3Q19 staged kernel reconstructs fine f from (rho, u, Pi_neq), and
# Pi_neq is interpolated coarse→fine by sample_parent_pi_neq_fast with C0 TRILINEAR
# weights — this is the TRUE full-distribution continuity gap (macro rho/u are already
# C4 B-spline; the stress is the one C0 piece). The earlier cubic *f_neq* carrier
# (_D3Q19_AMR_FNEQ_CUBIC) had no effect because the production path uses Pi_neq, not
# raw f_neq. When set, the 6 Pi components are interpolated with the SAME 4×4×4
# cubic-Lagrange sweep used for the macros (C1, matching D3Q27/D3Q39 carrier order),
# raising the reconstructed distribution from C0 to C1+ at the interface. Default OFF
# ⇒ byte-identical (trilinear). LUDWIG_D3Q19_PI_NEQ_CUBIC=1 enables.
const _D3Q19_PI_NEQ_CUBIC =
    get(ENV, "LUDWIG_D3Q19_PI_NEQ_CUBIC", "0") == "1"

# Coarse→fine (prolongation) reciprocal correction. tau_coarse/tau_fine are full τ.
@inline function _amr_fneq_postcoll_prolong_scale(scale::Float32, tau_coarse::Float32, tau_fine::Float32)
    if _AMR_FNEQ_POSTCOLL
        omega_c = 1.0f0 / max(tau_coarse, 0.500001f0)
        omega_f = 1.0f0 / max(tau_fine, 0.500001f0)
        one_m_wc = 1.0f0 - omega_c
        one_m_wf = 1.0f0 - omega_f
        # Prolongation factor (1-ω_fine)/(1-ω_coarse) — the RECIPROCAL of the
        # restriction-side (1-ω_parent)/(1-ω_child)=(1-ω_coarse)/(1-ω_fine), so the
        # round-trip R·P=1 (freestream + any round-tripped field byte-preserved).
        if abs(one_m_wc) > 1.0f-6
            return clamp(scale * (one_m_wf / one_m_wc), 0.01f0, 100.0f0)
        end
    end
    return scale
end


# ────────────────────────────────────────────────────────────────────
# Cubic midpoint weights (Palabos-style):
#   For a fine cell at the midpoint between parent cells i and i+1,
#   the cubic Lagrange weights over 4 parent cells {i-1, i, i+1, i+2} are:
#     {-1/16, 9/16, 9/16, -1/16}
#   This gives C1 continuity (matching value AND derivative) at the midpoint.
# ────��───────────────────────────────────────────────────────────────
@inline function cubic_interp_1d(vm1::Float32, v0::Float32, v1::Float32, v2::Float32, w::Float32)
    # General cubic Lagrange at fractional position w in [0,1] between v0 and v1:
    #   L(w) = vm1*(-w*(w-1)*(w-2)/6) + v0*((w+1)*(w-1)*(w-2)/2)
    #        + v1*(-w*(w+1)*(w-2)/2)  + v2*(w*(w+1)*(w-1)/6)
    # At w=0.5 (midpoint), this reduces to: -1/16*vm1 + 9/16*v0 + 9/16*v1 - 1/16*v2
    a = -w * (w - 1.0f0) * (w - 2.0f0) / 6.0f0
    b = (w + 1.0f0) * (w - 1.0f0) * (w - 2.0f0) / 2.0f0
    c = -w * (w + 1.0f0) * (w - 2.0f0) / 2.0f0
    d = w * (w + 1.0f0) * (w - 1.0f0) / 6.0f0
    return a * vm1 + b * v0 + c * v1 + d * v2
end

# ────────────────────────────────────────────────────────────────────
# C3 TEMPORAL (2026-06-04): Quintic Lagrange interpolation in TIME across
# AMR halo prolongation, for the D3Q19 path. Mirrors the D3Q39 B1 P3
# quintic-Lagrange chain. 5 historical anchors at times {-3,-2,-1,0,+1}·dt_c
# evaluated at the fine substep tw∈[0,1]:
#   L_{-3}(t)=  (t+2)(t+1)t(t-1)/24    [w_oldest2]
#   L_{-2}(t)= -(t+3)(t+1)t(t-1)/6     [w_oldest]
#   L_{-1}(t)=  (t+3)(t+2)t(t-1)/4     [w_older]
#   L_{ 0}(t)= -(t+3)(t+2)(t+1)(t-1)/6 [w_old]
#   L_{+1}(t)=  (t+3)(t+2)(t+1)t/24    [w_new]
# Returns the 5 Lagrange weights (sum≡1). C³ continuity at the parent steps.
# Soft-clamp of the blended result to [old,new]±10%|old-new| is applied at the
# call site to bound transient overshoot while preserving smooth-regime C³.
@inline function _quintic_lagrange_time_weights(t::Float32)
    tm1 = t - 1.0f0; tp1 = t + 1.0f0; tp2 = t + 2.0f0; tp3 = t + 3.0f0
    w_o2  =  (tp2 * tp1 * t   * tm1) / 24.0f0   # oldest2  (t=-3)
    w_o1  = -(tp3 * tp1 * t   * tm1) /  6.0f0   # oldest   (t=-2)
    w_or  =  (tp3 * tp2 * t   * tm1) /  4.0f0   # older    (t=-1)
    w_ol  = -(tp3 * tp2 * tp1 * tm1) /  6.0f0   # old      (t= 0)
    w_nw  =  (tp3 * tp2 * tp1 * t  ) / 24.0f0   # new      (t=+1)
    return (w_o2, w_o1, w_or, w_ol, w_nw)
end

# C3 TEMPORAL soft-clamp helper: bound a quintic-blended value to the
# [old,new] interval widened by 10% of |old-new|, so the negative Lagrange
# lobes cannot extrapolate beyond a bounded transient overshoot. In smooth
# regions old≈new ⇒ the band is tight; in ramps it collapses spurious
# extrapolation back to the bracketing pair.
#
# ACOUSTICS FIX (2026-06-04, numerical-proof workflow): the band half-width
# delta has an ABSOLUTE FLOOR of 1e-6 — `max(hi-lo, 1e-6)`. Without it, when
# old==new (a near-flat field carrying a SMALL acoustic wave — the exact
# "perfect subsonic acoustics" target) delta collapses to 0 and the quintic
# is clamped to a single point, DESTROYING the temporal curvature the C3
# quintic exists to preserve (measured ~380× worse than the unclamped quintic
# on a sinusoid sampled at old==new). The floor gives ±1e-6 of breathing room:
# a no-op everywhere the band already has width (ramp, gradients), rescuing
# only the degenerate old==new acoustic case. This MATCHES the D3Q39 path
# (`_soft_clamp_q5`/`_soft_clamp`), which already floors delta — the D3Q19
# variant had drifted without it.
@inline function _quintic_soft_clamp(v::Float32, old::Float32, new::Float32)
    lo = min(old, new); hi = max(old, new)
    delta = 0.1f0 * max(hi - lo, 1.0f-6)
    return clamp(v, lo - delta, hi + delta)
end

# Fine-cell centers live at x_f = (i_f - 0.5) * dx_f with dx_f = dx_c / 2.
# Expressed in the parent cell-center index space, that becomes:
#   x_f / dx_c + 0.5 = 0.5 * i_f + 0.25
# The previous map omitted the +0.5 parent-center shift and sampled half a
# coarse cell too far toward the negative side.
@inline _fine_to_parent_interp_coord(fine_g::Int32) = Float32(fine_g) * 0.5f0 + 0.25f0

# ── MIRROR-PLANE REFLECTED PULL (2026-06-11 z-PARITY FIX) ────────────────────
# At a free-slip mirror plane, the specular BC for a pull whose source lies
# beyond the plane is f_k(src) = f_{mirror(k)}(reflect(src)). The legacy code
# approximated this with the SAME-CELL population f_{mirror(k)}(cell) — exact
# only for pure plane-normal pulls from cells ON the plane. For COMBINED
# offsets (|c_tangential| ≠ 0, i.e. plane-adjacent cells at AMR box edges and
# domain corners) it reads the wrong cell, closing a one-step feedback loop
# with the cell's own boundary-filled populations. Measured consequence (Z201/
# Z203 MirrorProbe): FP-ULP mirror noise is transiently amplified ~1e4× at
# box-edge∩symmetry-plane cells (see HANDOVER_2026-06-10c §6). The fix:
# reflect the source position and direction, then fetch same-level — or let
# the caller parent-fill the REFLECTED in-domain pull when no same-level
# source exists. Gated by LUDWIG_D3Q19_MIRROR_REFLECT_FIX (timestep.jl).
#
# Reflection planes sit on cell faces: y_min between cells 0|1 (g' = 1 − g),
# y_max between ny|ny+1 (g' = 2·ny + 1 − g); |c| ≤ 1 ⇒ only g = 0 / n+1 occur
# and the reflected BLOCK-LOCAL coordinate is always back in range 1..bs (the
# crossing cell lives in the domain-edge block row).
@inline function _d3q19_mirror_reflected_target(
    k::Int32,
    sy::Int32, sz::Int32,
    src_gy::Int32, src_gz::Int32,
    is_y_min::Bool, is_y_max::Bool, is_z_min::Bool, is_z_max::Bool,
    block_size::Int32, ny_global::Int32, nz_global::Int32,
    mirror_y_arr, mirror_z_arr,
)
    @inbounds begin
        k_r = k
        r_sy = sy; r_gy = src_gy
        r_sz = sz; r_gz = src_gz
        if is_y_min || is_y_max
            k_r = mirror_y_arr[k_r]
            if is_y_min
                r_gy = Int32(1) - src_gy
                r_sy = Int32(1) - sy
            else
                r_gy = Int32(2) * ny_global + Int32(1) - src_gy
                r_sy = Int32(2) * block_size + Int32(1) - sy
            end
        end
        if is_z_min || is_z_max
            k_r = mirror_z_arr[k_r]
            if is_z_min
                r_gz = Int32(1) - src_gz
                r_sz = Int32(1) - sz
            else
                r_gz = Int32(2) * nz_global + Int32(1) - src_gz
                r_sz = Int32(2) * block_size + Int32(1) - sz
            end
        end
        return k_r, r_sy, r_sz, r_gy, r_gz
    end
end

# Same-level fetch of f_{k_r} at the reflected source. After reflection only
# the x local coordinate can still be out of block (±1 hop). Returns
# (found, value); !found ⇒ the reflected source is outside the active set
# (AMR box edge) and the caller must parent-fill k_r at the reflected globals.
@inline function _d3q19_mirror_reflected_fetch(
    f_in, neighbor_table,
    b_idx::Int32, k_r::Int32,
    sx::Int32, r_sy::Int32, r_sz::Int32,
    block_size::Int32,
)
    @inbounds begin
        if sx >= Int32(1) && sx <= block_size
            return true, f_in[sx, r_sy, r_sz, b_idx, k_r]
        end
        nb_off_x = sx < Int32(1) ? Int32(-1) : Int32(1)
        dir_idx = (nb_off_x + Int32(1)) + Int32(3) + Int32(9) + Int32(1)
        nb = neighbor_table[b_idx, dir_idx]
        if nb > Int32(0)
            nsx = sx < Int32(1) ? sx + block_size : sx - block_size
            return true, f_in[nsx, r_sy, r_sz, nb, k_r]
        end
        return false, 0.0f0
    end
end

# ── EQUILIBRIUM-BLEND DAMPING AT plane∩box-edge CELLS (2026-06-11, round 7) ──
# Does the same-level source at BLOCK-LOCAL (sx, sy, sz) of block b_idx exist
# in the active set? Unlike `_d3q19_mirror_reflected_fetch` above (which
# assumes only x can be off-block after reflection — true only when every
# non-x off-block offset was a domain crossing), this handles off-block
# offsets on ALL THREE axes via the 27-direction neighbor table. Needed for
# COMBINED plane∩box-edge pulls, where the tangential offset can be off-block
# WITHOUT being a domain crossing (the reflected source then sits outside the
# refinement box → the wrong-cell same-cell specular is the live treatment).
# Used as the per-pull "is this a dangerous wrong-cell specular?" trigger of
# the equilibrium-blend lever (HANDOVER_2026-06-10c §6 round 6 next-fix (a)).
@inline function _d3q19_samelevel_src_active(
    neighbor_table,
    b_idx::Int32,
    sx::Int32, sy::Int32, sz::Int32,
    block_size::Int32,
)
    @inbounds begin
        nb_off_x = sx < Int32(1) ? Int32(-1) : (sx > block_size ? Int32(1) : Int32(0))
        nb_off_y = sy < Int32(1) ? Int32(-1) : (sy > block_size ? Int32(1) : Int32(0))
        nb_off_z = sz < Int32(1) ? Int32(-1) : (sz > block_size ? Int32(1) : Int32(0))
        if nb_off_x == Int32(0) && nb_off_y == Int32(0) && nb_off_z == Int32(0)
            return true
        end
        dir_idx = (nb_off_x + Int32(1)) + (nb_off_y + Int32(1)) * Int32(3) +
                  (nb_off_z + Int32(1)) * Int32(9) + Int32(1)
        return neighbor_table[b_idx, dir_idx] > Int32(0)
    end
end

# Number of AXIS directions (±x, ±y, ±z) whose unit-offset source cell is a
# BOX VOID: inside the domain but outside the same-level active set. Flat
# box-face ring cells have exactly 1; box corner/edge ring cells have ≥ 2.
# Used by the eqblend `corner` scope (round 7b): the parity-amplification
# locus is the box-edge∩plane CORNER cells, so the damping can skip the flat
# face rows (whose feq fill produced the visible upstream-edge line).
@inline function _d3q19_void_axis_count(
    neighbor_table,
    b_idx::Int32,
    ix::Int32, iy::Int32, iz::Int32,
    gx::Int32, gy::Int32, gz::Int32,
    block_size::Int32,
    nx_global::Int32, ny_global::Int32, nz_global::Int32,
)
    cnt = Int32(0)
    @inbounds for d in Int32(1):Int32(6)
        dx = d == Int32(1) ? Int32(1) : d == Int32(2) ? Int32(-1) : Int32(0)
        dy = d == Int32(3) ? Int32(1) : d == Int32(4) ? Int32(-1) : Int32(0)
        dz = d == Int32(5) ? Int32(1) : d == Int32(6) ? Int32(-1) : Int32(0)
        sgx = gx + dx
        sgy = gy + dy
        sgz = gz + dz
        in_domain = sgx >= Int32(1) && sgx <= nx_global &&
                    sgy >= Int32(1) && sgy <= ny_global &&
                    sgz >= Int32(1) && sgz <= nz_global
        if in_domain &&
           !_d3q19_samelevel_src_active(neighbor_table, b_idx,
                                        ix + dx, iy + dy, iz + dz, block_size)
            cnt += Int32(1)
        end
    end
    return cnt
end

# Prev-step macro fetch at BLOCK-LOCAL (nx_loc, ny_loc, nz_loc) of block b_idx,
# with off-block offsets on all three axes via the 27-direction neighbor table.
# Returns (ok, rho, ux, uy, uz); !ok ⇒ the neighbor block is missing or the
# cell is an obstacle. Used by the eqblend `localgrad` target (round 7b): the
# opposite-side same-level neighbor supplies the one-sided gradient for the
# parent-free extrapolated fill ρ_src ≈ 2ρ_cell − ρ_opp.
@inline function _d3q19_fetch_macro_neighbor(
    rho_arr, vel_arr, obstacle, neighbor_table,
    b_idx::Int32,
    nx_loc::Int32, ny_loc::Int32, nz_loc::Int32,
    block_size::Int32,
)
    @inbounds begin
        nb_off_x = nx_loc < Int32(1) ? Int32(-1) : (nx_loc > block_size ? Int32(1) : Int32(0))
        nb_off_y = ny_loc < Int32(1) ? Int32(-1) : (ny_loc > block_size ? Int32(1) : Int32(0))
        nb_off_z = nz_loc < Int32(1) ? Int32(-1) : (nz_loc > block_size ? Int32(1) : Int32(0))
        nb = b_idx
        if !(nb_off_x == Int32(0) && nb_off_y == Int32(0) && nb_off_z == Int32(0))
            dir_idx = (nb_off_x + Int32(1)) + (nb_off_y + Int32(1)) * Int32(3) +
                      (nb_off_z + Int32(1)) * Int32(9) + Int32(1)
            nb = neighbor_table[b_idx, dir_idx]
            if nb <= Int32(0)
                return false, 1.0f0, 0.0f0, 0.0f0, 0.0f0
            end
        end
        lx = nx_loc < Int32(1) ? nx_loc + block_size : (nx_loc > block_size ? nx_loc - block_size : nx_loc)
        ly = ny_loc < Int32(1) ? ny_loc + block_size : (ny_loc > block_size ? ny_loc - block_size : ny_loc)
        lz = nz_loc < Int32(1) ? nz_loc + block_size : (nz_loc > block_size ? nz_loc - block_size : nz_loc)
        if obstacle[lx, ly, lz, nb]
            return false, 1.0f0, 0.0f0, 0.0f0, 0.0f0
        end
        r = rho_arr[lx, ly, lz, nb]
        ux = vel_arr[lx, ly, lz, nb, 1]
        uy = vel_arr[lx, ly, lz, nb, 2]
        uz = vel_arr[lx, ly, lz, nb, 3]
        ok = (r == r) && r >= 0.01f0 && (ux == ux) && (uy == uy) && (uz == uz)
        return ok, r, ux, uy, uz
    end
end

@inline function _parent_interp_bracket(p_cont::Float32, parent_n::Int32)
    if parent_n <= Int32(1)
        return Int32(1), Int32(1), 0.0f0
    end
    # Clamp the continuous parent coordinate before computing the bracket.
    # Without this, fine cells adjacent to a physical wall use wx/wy/wz < 0
    # or > 1 after the index clamp, i.e. an unintended extrapolation. That
    # shows up as wall-corner rails in temperature/schlieren slices.
    p_clamped = clamp(p_cont, 1.0f0, Float32(parent_n))
    i0 = clamp(floor(Int32, p_clamped), Int32(1), parent_n - Int32(1))
    i1 = i0 + Int32(1)
    return i0, i1, clamp(p_clamped - Float32(i0), 0.0f0, 1.0f0)
end

# ── MIRROR-GHOST PARENT INTERPOLATION (2026-06-12, round 8) ──────────────────
# The y_min/y_max/z_min/z_max transverse boundaries are FREE-SLIP MIRROR planes,
# yet every AMR coarse→fine macro interpolation degrades to a one-sided/clamped
# special case at the plane-adjacent fine rows (gy=1,2 / gz=1,2 / top rows):
# the cubic 4×4×4 stencil is invalid there (±2 margin), the bracket clamps to
# wy=0 (evaluation AT parent row 1 — a half-parent-cell offset, uy not driven
# to 0 at the plane), and the B-spline support check always fails. The plane
# rows therefore get the WORST interpolation in the domain — and they are
# exactly the cells of the z-parity amplification and the corner artifacts
# (HANDOVER_2026-06-10c §6). This gate supplies SPECULAR GHOSTS instead:
# value(ghost row) = value(reflected in-domain row) with the plane-normal
# velocity component NEGATED (ρ/u_tangential even, u_normal odd) — for a
# symmetry plane this is EXACT, and every stencil can run at full quality.
# Plane convention: the mirror sits on the cell FACE below parent row 1 /
# above row n (parent-index coordinate 0.5 / n+0.5): reflect(g) = 1−g (low),
# 2n+1−g (high).
# Gated COMPILE-TIME (no ABI change; branches DCE'd when off — bitwise legacy).
# Only valid when the transverse boundaries are mirrors (symmetric_analysis
# free-slip y_min; NOT for ground/no-slip y_min cases — leave OFF there).
const _AMR_MIRROR_GHOST_INTERP = get(ENV, "LUDWIG_AMR_MIRROR_GHOST", "0") == "1"

@inline function _mirror_reflect_parent_axis(pg::Int32, parent_n::Int32)
    if pg < Int32(1)
        return Int32(1) - pg, -1.0f0
    elseif pg > parent_n
        return Int32(2) * parent_n + Int32(1) - pg, -1.0f0
    end
    return pg, 1.0f0
end

# Ghost-capable bracket: i0 ∈ [0, parent_n] so the pair can straddle the mirror
# plane (rows 0 / n+1 are specular ghosts resolved by _mirror_reflect_parent_axis).
@inline function _parent_interp_bracket_mirror(p_cont::Float32, parent_n::Int32)
    if parent_n <= Int32(1)
        return Int32(1), Int32(1), 0.0f0
    end
    i0 = clamp(floor(Int32, p_cont), Int32(0), parent_n)
    return i0, i0 + Int32(1), clamp(p_cont - Float32(i0), 0.0f0, 1.0f0)
end

# @noinline (2026-06-04): this gradient-preserving partial-stencil macro
# interpolation is large (3 separable extrapolation passes). Keeping it OUT of
# line is REQUIRED — inlining it into interpolate_with_rescaling (itself inlined
# into the D3Q19 kernel) blew the @inline size threshold → the whole kernel lost
# type-stability → GPU InvalidIRError (every *,+,max became a dynamic dispatch).
# As a separate compiled unit it is type-stable and GPU-JIT-safe.
@noinline function sample_parent_macro_for_fine_cell(
    parent_rho_new, parent_vel_new,
    parent_rho_old, parent_vel_old,
    parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    fine_gx::Int32, fine_gy::Int32, fine_gz::Int32,
    block_size::Int32,
    temporal_weight::Float32,
    use_temporal_interp::Int32
)
    px_cont = _fine_to_parent_interp_coord(fine_gx)
    py_cont = _fine_to_parent_interp_coord(fine_gy)
    pz_cont = _fine_to_parent_interp_coord(fine_gz)

    parent_nx = parent_dim_x * block_size
    parent_ny = parent_dim_y * block_size
    parent_nz = parent_dim_z * block_size
    px0, px1, wx = _parent_interp_bracket(px_cont, parent_nx)
    # MIRROR-GHOST (round 8): ghost-capable y/z brackets — at the plane rows the
    # pair straddles the mirror (row 0 / n+1 ghosts) instead of clamping wy=0.
    # PERIODIC-Y AMR: unclamped Y bracket; the lookups below wrap across the seam.
    py0, py1, wy = _PERIODIC_Y_AMR ? _pbracket_y(py_cont) :
        _AMR_MIRROR_GHOST_INTERP ?
        _parent_interp_bracket_mirror(py_cont, parent_ny) :
        _parent_interp_bracket(py_cont, parent_ny)
    pz0, pz1, wz = _AMR_MIRROR_GHOST_INTERP ?
        _parent_interp_bracket_mirror(pz_cont, parent_nz) :
        _parent_interp_bracket(pz_cont, parent_nz)

    @inline function get_macro(pgx::Int32, pgy_req::Int32, pgz_req::Int32)
        # MIRROR-GHOST (round 8): resolve ghost rows by specular reflection
        # (ρ/u_tangential even, u_normal odd). DCE'd when the const is off.
        pgy = pgy_req
        pgz = pgz_req
        sy = 1.0f0
        sz = 1.0f0
        if _AMR_MIRROR_GHOST_INTERP
            pgy, sy = _mirror_reflect_parent_axis(pgy_req, parent_ny)
            pgz, sz = _mirror_reflect_parent_axis(pgz_req, parent_nz)
        end
        pgy = _pwrap_y(pgy, parent_ny)   # periodic-Y AMR (identity when off)
        pbx = div(pgx - Int32(1), block_size) + Int32(1)
        pby = div(pgy - Int32(1), block_size) + Int32(1)
        pbz = div(pgz - Int32(1), block_size) + Int32(1)
        if pbx >= Int32(1) && pbx <= parent_dim_x &&
           pby >= Int32(1) && pby <= parent_dim_y &&
           pbz >= Int32(1) && pbz <= parent_dim_z
            pb_idx = parent_ptr[pbx, pby, pbz]
            if pb_idx > Int32(0)
                plx = (pgx - Int32(1)) % block_size + Int32(1)
                ply = (pgy - Int32(1)) % block_size + Int32(1)
                plz = (pgz - Int32(1)) % block_size + Int32(1)
                rho_v = parent_rho_new[plx, ply, plz, pb_idx]
                ux_v = parent_vel_new[plx, ply, plz, pb_idx, 1]
                uy_v = parent_vel_new[plx, ply, plz, pb_idx, 2]
                uz_v = parent_vel_new[plx, ply, plz, pb_idx, 3]
                if use_temporal_interp == Int32(1) && temporal_weight < 0.99f0 &&
                   parent_rho_old !== nothing && parent_vel_old !== nothing
                    tw = temporal_weight
                    rho_v = parent_rho_old[plx, ply, plz, pb_idx] * (1.0f0 - tw) + rho_v * tw
                    ux_v = parent_vel_old[plx, ply, plz, pb_idx, 1] * (1.0f0 - tw) + ux_v * tw
                    uy_v = parent_vel_old[plx, ply, plz, pb_idx, 2] * (1.0f0 - tw) + uy_v * tw
                    uz_v = parent_vel_old[plx, ply, plz, pb_idx, 3] * (1.0f0 - tw) + uz_v * tw
                end
                if _AMR_MIRROR_GHOST_INTERP
                    uy_v *= sy
                    uz_v *= sz
                end
                return (rho_v, ux_v, uy_v, uz_v, true)
            end
        end
        return (1.0f0, 0.0f0, 0.0f0, 0.0f0, false)
    end

    d000 = get_macro(px0, py0, pz0)
    d100 = get_macro(px1, py0, pz0)
    d010 = get_macro(px0, py1, pz0)
    d110 = get_macro(px1, py1, pz0)
    d001 = get_macro(px0, py0, pz1)
    d101 = get_macro(px1, py0, pz1)
    d011 = get_macro(px0, py1, pz1)
    d111 = get_macro(px1, py1, pz1)

    w000 = (1.0f0 - wx) * (1.0f0 - wy) * (1.0f0 - wz)
    w100 = wx * (1.0f0 - wy) * (1.0f0 - wz)
    w010 = (1.0f0 - wx) * wy * (1.0f0 - wz)
    w110 = wx * wy * (1.0f0 - wz)
    w001 = (1.0f0 - wx) * (1.0f0 - wy) * wz
    w101 = wx * (1.0f0 - wy) * wz
    w011 = (1.0f0 - wx) * wy * wz
    w111 = wx * wy * wz

    # ── Gradient-preserving partial-stencil fill (2026-05-31) ───────────────
    # BUG (confirmed by offline unit test + the isentropic-vortex AMR gate):
    # when the 8-corner stencil straddles the refined-box edge, some corners are
    # invalid. The legacy path accumulated only the valid corners and divided by
    # their weight sum (wsum < 1) — a renormalization that does NOT reproduce a
    # linear field. For a gradient field (a vortex, a bow shock) that biases the
    # interpolant toward the valid side → a systematic velocity/density offset at
    # the AMR-box edge → energy injection (vortex peak grew +60% across the AMR
    # interface while the single-level operator preserved it to viscous decay).
    #
    # FIX: reconstruct each INVALID corner by axis-mirror linear extrapolation
    # from valid corners (3 separable 1D passes: x, then y, then z). For a linear
    # field this reproduces the missing corner EXACTLY, so the full-weight
    # trilinear sum (Σw = 1, no renormalization) is gradient-exact at the edge.
    # If a whole axis-pair is invalid we fall back to copying the valid partner
    # (zero-gradient — the best available), and only if a corner has no valid
    # in-stencil source does it stay flagged invalid (caller falls through).

    # All 8 corners valid → exact full trilinear (Σw = 1). Fast path.
    if d000[5] && d100[5] && d010[5] && d110[5] && d001[5] && d101[5] && d011[5] && d111[5]
        rho_acc = w000*d000[1] + w100*d100[1] + w010*d010[1] + w110*d110[1] +
                  w001*d001[1] + w101*d101[1] + w011*d011[1] + w111*d111[1]
        ux_acc  = w000*d000[2] + w100*d100[2] + w010*d010[2] + w110*d110[2] +
                  w001*d001[2] + w101*d101[2] + w011*d011[2] + w111*d111[2]
        uy_acc  = w000*d000[3] + w100*d100[3] + w010*d010[3] + w110*d110[3] +
                  w001*d001[3] + w101*d101[3] + w011*d011[3] + w111*d111[3]
        uz_acc  = w000*d000[4] + w100*d100[4] + w010*d010[4] + w110*d110[4] +
                  w001*d001[4] + w101*d101[4] + w011*d011[4] + w111*d111[4]
        return (rho_acc, ux_acc, uy_acc, uz_acc, true)
    end

    # Partial stencil: fill invalid corners by LINEAR EXTRAPOLATION from the valid
    # same-line partner + a 3rd parent cell one step further (missing = 2·near −
    # far). For a linear field this reconstructs the missing corner EXACTLY (verified
    # offline: err 1e-16 vs the legacy renormalization's 6.8e-1), so the full-weight
    # trilinear sum (Σw = 1, no renormalization) is gradient-exact at the box edge.
    # The far cell is fetched via get_macro at px0-1 / px1+1 etc.; if it too is
    # invalid we fall back to a zero-gradient copy (best available). 3 separable
    # passes (x,y,z). Plain scalar locals only — GPU-JIT-safe.
    # Corner order: 0=000 1=100 2=010 3=110 4=001 5=101 6=011 7=111 (bit0=x,1=y,2=z)
    r0=d000[1]; x0=d000[2]; y0=d000[3]; z0=d000[4]; v0=d000[5]
    r1=d100[1]; x1=d100[2]; y1=d100[3]; z1=d100[4]; v1=d100[5]
    r2=d010[1]; x2=d010[2]; y2=d010[3]; z2=d010[4]; v2=d010[5]
    r3=d110[1]; x3=d110[2]; y3=d110[3]; z3=d110[4]; v3=d110[5]
    r4=d001[1]; x4=d001[2]; y4=d001[3]; z4=d001[4]; v4=d001[5]
    r5=d101[1]; x5=d101[2]; y5=d101[3]; z5=d101[4]; v5=d101[5]
    r6=d011[1]; x6=d011[2]; y6=d011[3]; z6=d011[4]; v6=d011[5]
    r7=d111[1]; x7=d111[2]; y7=d111[3]; z7=d111[4]; v7=d111[5]

    if !(v0||v1||v2||v3||v4||v5||v6||v7)
        return (1.0f0, 0.0f0, 0.0f0, 0.0f0, false)
    end

    # far parent indices (one step beyond the bracket, for extrapolation)
    pxm = px0 - Int32(1); pxp = px1 + Int32(1)
    pym = py0 - Int32(1); pyp = py1 + Int32(1)
    pzm = pz0 - Int32(1); pzp = pz1 + Int32(1)

    # X pass — same-line pairs (near0=x0, near1=x1): reconstruct the invalid side
    # as 2·near − far; if far invalid, copy near (zero-gradient fallback).
    # line (0,1) at (py0,pz0); (2,3) at (py1,pz0); (4,5) at (py0,pz1); (6,7) at (py1,pz1)
    if v0 != v1
        if v0  # x1 missing: far = pxm at (py0,pz0)
            fd=get_macro(pxm,py0,pz0); r1=fd[5] ? 2f0*r0-fd[1] : r0; x1=fd[5] ? 2f0*x0-fd[2] : x0; y1=fd[5] ? 2f0*y0-fd[3] : y0; z1=fd[5] ? 2f0*z0-fd[4] : z0; v1=true
        else   # x0 missing: far = pxp at (py0,pz0)
            fd=get_macro(pxp,py0,pz0); r0=fd[5] ? 2f0*r1-fd[1] : r1; x0=fd[5] ? 2f0*x1-fd[2] : x1; y0=fd[5] ? 2f0*y1-fd[3] : y1; z0=fd[5] ? 2f0*z1-fd[4] : z1; v0=true
        end
    end
    if v2 != v3
        if v2; fd=get_macro(pxm,py1,pz0); r3=fd[5] ? 2f0*r2-fd[1] : r2; x3=fd[5] ? 2f0*x2-fd[2] : x2; y3=fd[5] ? 2f0*y2-fd[3] : y2; z3=fd[5] ? 2f0*z2-fd[4] : z2; v3=true
        else;  fd=get_macro(pxp,py1,pz0); r2=fd[5] ? 2f0*r3-fd[1] : r3; x2=fd[5] ? 2f0*x3-fd[2] : x3; y2=fd[5] ? 2f0*y3-fd[3] : y3; z2=fd[5] ? 2f0*z3-fd[4] : z3; v2=true; end
    end
    if v4 != v5
        if v4; fd=get_macro(pxm,py0,pz1); r5=fd[5] ? 2f0*r4-fd[1] : r4; x5=fd[5] ? 2f0*x4-fd[2] : x4; y5=fd[5] ? 2f0*y4-fd[3] : y4; z5=fd[5] ? 2f0*z4-fd[4] : z4; v5=true
        else;  fd=get_macro(pxp,py0,pz1); r4=fd[5] ? 2f0*r5-fd[1] : r5; x4=fd[5] ? 2f0*x5-fd[2] : x5; y4=fd[5] ? 2f0*y5-fd[3] : y5; z4=fd[5] ? 2f0*z5-fd[4] : z5; v4=true; end
    end
    if v6 != v7
        if v6; fd=get_macro(pxm,py1,pz1); r7=fd[5] ? 2f0*r6-fd[1] : r6; x7=fd[5] ? 2f0*x6-fd[2] : x6; y7=fd[5] ? 2f0*y6-fd[3] : y6; z7=fd[5] ? 2f0*z6-fd[4] : z6; v7=true
        else;  fd=get_macro(pxp,py1,pz1); r6=fd[5] ? 2f0*r7-fd[1] : r7; x6=fd[5] ? 2f0*x7-fd[2] : x7; y6=fd[5] ? 2f0*y7-fd[3] : y7; z6=fd[5] ? 2f0*z7-fd[4] : z7; v6=true; end
    end
    # Y pass — pairs (0,2)(1,3)(4,6)(5,7), far = pym/pyp
    if v0 != v2
        if v0; fd=get_macro(px0,pym,pz0); r2=fd[5] ? 2f0*r0-fd[1] : r0; x2=fd[5] ? 2f0*x0-fd[2] : x0; y2=fd[5] ? 2f0*y0-fd[3] : y0; z2=fd[5] ? 2f0*z0-fd[4] : z0; v2=true
        else;  fd=get_macro(px0,pyp,pz0); r0=fd[5] ? 2f0*r2-fd[1] : r2; x0=fd[5] ? 2f0*x2-fd[2] : x2; y0=fd[5] ? 2f0*y2-fd[3] : y2; z0=fd[5] ? 2f0*z2-fd[4] : z2; v0=true; end
    end
    if v1 != v3
        if v1; fd=get_macro(px1,pym,pz0); r3=fd[5] ? 2f0*r1-fd[1] : r1; x3=fd[5] ? 2f0*x1-fd[2] : x1; y3=fd[5] ? 2f0*y1-fd[3] : y1; z3=fd[5] ? 2f0*z1-fd[4] : z1; v3=true
        else;  fd=get_macro(px1,pyp,pz0); r1=fd[5] ? 2f0*r3-fd[1] : r3; x1=fd[5] ? 2f0*x3-fd[2] : x3; y1=fd[5] ? 2f0*y3-fd[3] : y3; z1=fd[5] ? 2f0*z3-fd[4] : z3; v1=true; end
    end
    if v4 != v6
        if v4; fd=get_macro(px0,pym,pz1); r6=fd[5] ? 2f0*r4-fd[1] : r4; x6=fd[5] ? 2f0*x4-fd[2] : x4; y6=fd[5] ? 2f0*y4-fd[3] : y4; z6=fd[5] ? 2f0*z4-fd[4] : z4; v6=true
        else;  fd=get_macro(px0,pyp,pz1); r4=fd[5] ? 2f0*r6-fd[1] : r6; x4=fd[5] ? 2f0*x6-fd[2] : x6; y4=fd[5] ? 2f0*y6-fd[3] : y6; z4=fd[5] ? 2f0*z6-fd[4] : z6; v4=true; end
    end
    if v5 != v7
        if v5; fd=get_macro(px1,pym,pz1); r7=fd[5] ? 2f0*r5-fd[1] : r5; x7=fd[5] ? 2f0*x5-fd[2] : x5; y7=fd[5] ? 2f0*y5-fd[3] : y5; z7=fd[5] ? 2f0*z5-fd[4] : z5; v7=true
        else;  fd=get_macro(px1,pyp,pz1); r5=fd[5] ? 2f0*r7-fd[1] : r7; x5=fd[5] ? 2f0*x7-fd[2] : x7; y5=fd[5] ? 2f0*y7-fd[3] : y7; z5=fd[5] ? 2f0*z7-fd[4] : z7; v5=true; end
    end
    # Z pass — pairs (0,4)(1,5)(2,6)(3,7), far = pzm/pzp
    if v0 != v4
        if v0; fd=get_macro(px0,py0,pzm); r4=fd[5] ? 2f0*r0-fd[1] : r0; x4=fd[5] ? 2f0*x0-fd[2] : x0; y4=fd[5] ? 2f0*y0-fd[3] : y0; z4=fd[5] ? 2f0*z0-fd[4] : z0; v4=true
        else;  fd=get_macro(px0,py0,pzp); r0=fd[5] ? 2f0*r4-fd[1] : r4; x0=fd[5] ? 2f0*x4-fd[2] : x4; y0=fd[5] ? 2f0*y4-fd[3] : y4; z0=fd[5] ? 2f0*z4-fd[4] : z4; v0=true; end
    end
    if v1 != v5
        if v1; fd=get_macro(px1,py0,pzm); r5=fd[5] ? 2f0*r1-fd[1] : r1; x5=fd[5] ? 2f0*x1-fd[2] : x1; y5=fd[5] ? 2f0*y1-fd[3] : y1; z5=fd[5] ? 2f0*z1-fd[4] : z1; v5=true
        else;  fd=get_macro(px1,py0,pzp); r1=fd[5] ? 2f0*r5-fd[1] : r5; x1=fd[5] ? 2f0*x5-fd[2] : x5; y1=fd[5] ? 2f0*y5-fd[3] : y5; z1=fd[5] ? 2f0*z5-fd[4] : z5; v1=true; end
    end
    if v2 != v6
        if v2; fd=get_macro(px0,py1,pzm); r6=fd[5] ? 2f0*r2-fd[1] : r2; x6=fd[5] ? 2f0*x2-fd[2] : x2; y6=fd[5] ? 2f0*y2-fd[3] : y2; z6=fd[5] ? 2f0*z2-fd[4] : z2; v6=true
        else;  fd=get_macro(px0,py1,pzp); r2=fd[5] ? 2f0*r6-fd[1] : r6; x2=fd[5] ? 2f0*x6-fd[2] : x6; y2=fd[5] ? 2f0*y6-fd[3] : y6; z2=fd[5] ? 2f0*z6-fd[4] : z6; v2=true; end
    end
    if v3 != v7
        if v3; fd=get_macro(px1,py1,pzm); r7=fd[5] ? 2f0*r3-fd[1] : r3; x7=fd[5] ? 2f0*x3-fd[2] : x3; y7=fd[5] ? 2f0*y3-fd[3] : y3; z7=fd[5] ? 2f0*z3-fd[4] : z3; v7=true
        else;  fd=get_macro(px1,py1,pzp); r3=fd[5] ? 2f0*r7-fd[1] : r7; x3=fd[5] ? 2f0*x7-fd[2] : x7; y3=fd[5] ? 2f0*y7-fd[3] : y7; z3=fd[5] ? 2f0*z7-fd[4] : z7; v3=true; end
    end

    rho_acc = w000*r0 + w100*r1 + w010*r2 + w110*r3 + w001*r4 + w101*r5 + w011*r6 + w111*r7
    ux_acc  = w000*x0 + w100*x1 + w010*x2 + w110*x3 + w001*x4 + w101*x5 + w011*x6 + w111*x7
    uy_acc  = w000*y0 + w100*y1 + w010*y2 + w110*y3 + w001*y4 + w101*y5 + w011*y6 + w111*y7
    uz_acc  = w000*z0 + w100*z1 + w010*z2 + w110*z3 + w001*z4 + w101*z5 + w011*z6 + w111*z7
    return (rho_acc, ux_acc, uy_acc, uz_acc, true)
end

@noinline function sample_parent_pi_neq_for_fine_cell(
    parent_pi_neq,
    parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    fine_gx::Int32, fine_gy::Int32, fine_gz::Int32,
    block_size::Int32
)
    px_cont = _fine_to_parent_interp_coord(fine_gx)
    py_cont = _fine_to_parent_interp_coord(fine_gy)
    pz_cont = _fine_to_parent_interp_coord(fine_gz)

    parent_nx = parent_dim_x * block_size
    parent_ny = parent_dim_y * block_size
    parent_nz = parent_dim_z * block_size
    px0, px1, wx = _parent_interp_bracket(px_cont, parent_nx)
    py0, py1, wy = _PERIODIC_Y_AMR ? _pbracket_y(py_cont) : _parent_interp_bracket(py_cont, parent_ny)
    pz0, pz1, wz = _parent_interp_bracket(pz_cont, parent_nz)

    @inline function get_pi(pgx::Int32, pgy::Int32, pgz::Int32)
        pbx = (pgx - Int32(1)) ÷ block_size + Int32(1)
        pgy = _pwrap_y(pgy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
        pby = (pgy - Int32(1)) ÷ block_size + Int32(1)
        pbz = (pgz - Int32(1)) ÷ block_size + Int32(1)
        if pbx >= Int32(1) && pbx <= parent_dim_x &&
           pby >= Int32(1) && pby <= parent_dim_y &&
           pbz >= Int32(1) && pbz <= parent_dim_z
            pb_idx = parent_ptr[pbx, pby, pbz]
            if pb_idx > Int32(0)
                plx = (pgx - Int32(1)) % block_size + Int32(1)
                ply = (pgy - Int32(1)) % block_size + Int32(1)
                plz = (pgz - Int32(1)) % block_size + Int32(1)
                return (
                    parent_pi_neq[plx, ply, plz, pb_idx, 1],
                    parent_pi_neq[plx, ply, plz, pb_idx, 2],
                    parent_pi_neq[plx, ply, plz, pb_idx, 3],
                    parent_pi_neq[plx, ply, plz, pb_idx, 4],
                    parent_pi_neq[plx, ply, plz, pb_idx, 5],
                    parent_pi_neq[plx, ply, plz, pb_idx, 6],
                    true
                )
            end
        end
        return (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, false)
    end

    if _D3Q19_PI_NEQ_CUBIC
        # ── C2 stress carrier: 4×4×4 cubic-Lagrange over the 6 Pi_neq components ──
        # Raises the production stress transfer from C0 trilinear to C1+, matching the
        # macro carrier order, so the reconstructed fine distribution is C1+ at the
        # interface (the real full-distribution continuity lever — the production staged
        # kernel reconstructs f from rho,u,Pi_neq, so Pi_neq's order IS the gap). Needs
        # the FULL 4×4×4 stencil valid; on ANY missing tap (box edge) it falls through
        # to trilinear below. Cubic weights identical to the macro/f_neq cubic sweep.
        sxx = 0.0f0; syy = 0.0f0; szz = 0.0f0; sxy = 0.0f0; sxz = 0.0f0; syz = 0.0f0
        all_ok = true
        for ix in Int32(-1):Int32(2)
            pcx = px0 + ix
            wxv = ix == Int32(-1) ? (-wx*(wx-1.0f0)*(wx-2.0f0)/6.0f0) :
                  ix == Int32(0)  ? ((wx+1.0f0)*(wx-1.0f0)*(wx-2.0f0)/2.0f0) :
                  ix == Int32(1)  ? (-wx*(wx+1.0f0)*(wx-2.0f0)/2.0f0) :
                                    (wx*(wx+1.0f0)*(wx-1.0f0)/6.0f0)
            for iy in Int32(-1):Int32(2)
                pcy = py0 + iy
                wyv = iy == Int32(-1) ? (-wy*(wy-1.0f0)*(wy-2.0f0)/6.0f0) :
                      iy == Int32(0)  ? ((wy+1.0f0)*(wy-1.0f0)*(wy-2.0f0)/2.0f0) :
                      iy == Int32(1)  ? (-wy*(wy+1.0f0)*(wy-2.0f0)/2.0f0) :
                                        (wy*(wy+1.0f0)*(wy-1.0f0)/6.0f0)
                for iz in Int32(-1):Int32(2)
                    pcz = pz0 + iz
                    wzv = iz == Int32(-1) ? (-wz*(wz-1.0f0)*(wz-2.0f0)/6.0f0) :
                          iz == Int32(0)  ? ((wz+1.0f0)*(wz-1.0f0)*(wz-2.0f0)/2.0f0) :
                          iz == Int32(1)  ? (-wz*(wz+1.0f0)*(wz-2.0f0)/2.0f0) :
                                            (wz*(wz+1.0f0)*(wz-1.0f0)/6.0f0)
                    dpic = get_pi(pcx, pcy, pcz)
                    if !dpic[7]
                        all_ok = false
                    else
                        w3c = wxv * wyv * wzv
                        sxx += w3c * dpic[1]; syy += w3c * dpic[2]; szz += w3c * dpic[3]
                        sxy += w3c * dpic[4]; sxz += w3c * dpic[5]; syz += w3c * dpic[6]
                    end
                end
            end
        end
        if all_ok
            return (sxx, syy, szz, sxy, sxz, syz, true)
        end
        # else: fall through to trilinear (box-edge graceful degradation)
    end

    w000 = (1.0f0 - wx) * (1.0f0 - wy) * (1.0f0 - wz)
    w100 = wx * (1.0f0 - wy) * (1.0f0 - wz)
    w010 = (1.0f0 - wx) * wy * (1.0f0 - wz)
    w110 = wx * wy * (1.0f0 - wz)
    w001 = (1.0f0 - wx) * (1.0f0 - wy) * wz
    w101 = wx * (1.0f0 - wy) * wz
    w011 = (1.0f0 - wx) * wy * wz
    w111 = wx * wy * wz

    d000 = get_pi(px0, py0, pz0)
    d100 = get_pi(px1, py0, pz0)
    d010 = get_pi(px0, py1, pz0)
    d110 = get_pi(px1, py1, pz0)
    d001 = get_pi(px0, py0, pz1)
    d101 = get_pi(px1, py0, pz1)
    d011 = get_pi(px0, py1, pz1)
    d111 = get_pi(px1, py1, pz1)

    if d000[7] && d100[7] && d010[7] && d110[7] && d001[7] && d101[7] && d011[7] && d111[7]
        pxx = w000*d000[1] + w100*d100[1] + w010*d010[1] + w110*d110[1] +
              w001*d001[1] + w101*d101[1] + w011*d011[1] + w111*d111[1]
        pyy = w000*d000[2] + w100*d100[2] + w010*d010[2] + w110*d110[2] +
              w001*d001[2] + w101*d101[2] + w011*d011[2] + w111*d111[2]
        pzz = w000*d000[3] + w100*d100[3] + w010*d010[3] + w110*d110[3] +
              w001*d001[3] + w101*d101[3] + w011*d011[3] + w111*d111[3]
        pxy = w000*d000[4] + w100*d100[4] + w010*d010[4] + w110*d110[4] +
              w001*d001[4] + w101*d101[4] + w011*d011[4] + w111*d111[4]
        pxz = w000*d000[5] + w100*d100[5] + w010*d010[5] + w110*d110[5] +
              w001*d001[5] + w101*d101[5] + w011*d011[5] + w111*d111[5]
        pyz = w000*d000[6] + w100*d100[6] + w010*d010[6] + w110*d110[6] +
              w001*d001[6] + w101*d101[6] + w011*d011[6] + w111*d111[6]
        return (pxx, pyy, pzz, pxy, pxz, pyz, true)
    end

    pxx0=d000[1]; pyy0=d000[2]; pzz0=d000[3]; pxy0=d000[4]; pxz0=d000[5]; pyz0=d000[6]; v0=d000[7]
    pxx1=d100[1]; pyy1=d100[2]; pzz1=d100[3]; pxy1=d100[4]; pxz1=d100[5]; pyz1=d100[6]; v1=d100[7]
    pxx2=d010[1]; pyy2=d010[2]; pzz2=d010[3]; pxy2=d010[4]; pxz2=d010[5]; pyz2=d010[6]; v2=d010[7]
    pxx3=d110[1]; pyy3=d110[2]; pzz3=d110[3]; pxy3=d110[4]; pxz3=d110[5]; pyz3=d110[6]; v3=d110[7]
    pxx4=d001[1]; pyy4=d001[2]; pzz4=d001[3]; pxy4=d001[4]; pxz4=d001[5]; pyz4=d001[6]; v4=d001[7]
    pxx5=d101[1]; pyy5=d101[2]; pzz5=d101[3]; pxy5=d101[4]; pxz5=d101[5]; pyz5=d101[6]; v5=d101[7]
    pxx6=d011[1]; pyy6=d011[2]; pzz6=d011[3]; pxy6=d011[4]; pxz6=d011[5]; pyz6=d011[6]; v6=d011[7]
    pxx7=d111[1]; pyy7=d111[2]; pzz7=d111[3]; pxy7=d111[4]; pxz7=d111[5]; pyz7=d111[6]; v7=d111[7]

    if !(v0||v1||v2||v3||v4||v5||v6||v7)
        return (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, false)
    end

    # PI/stress must complete partial stencils exactly like macro interpolation.
    # Renormalizing only valid corners samples stress from the populated side of
    # sparse AMR corners, while rho/u use a full-weight completed stencil.
    pxm = px0 - Int32(1); pxp = px1 + Int32(1)
    pym = py0 - Int32(1); pyp = py1 + Int32(1)
    pzm = pz0 - Int32(1); pzp = pz1 + Int32(1)

    # X pass: pairs (0,1), (2,3), (4,5), (6,7).
    if v0 != v1
        if v0
            fd=get_pi(pxm,py0,pz0)
            pxx1=fd[7] ? 2f0*pxx0-fd[1] : pxx0; pyy1=fd[7] ? 2f0*pyy0-fd[2] : pyy0; pzz1=fd[7] ? 2f0*pzz0-fd[3] : pzz0
            pxy1=fd[7] ? 2f0*pxy0-fd[4] : pxy0; pxz1=fd[7] ? 2f0*pxz0-fd[5] : pxz0; pyz1=fd[7] ? 2f0*pyz0-fd[6] : pyz0; v1=true
        else
            fd=get_pi(pxp,py0,pz0)
            pxx0=fd[7] ? 2f0*pxx1-fd[1] : pxx1; pyy0=fd[7] ? 2f0*pyy1-fd[2] : pyy1; pzz0=fd[7] ? 2f0*pzz1-fd[3] : pzz1
            pxy0=fd[7] ? 2f0*pxy1-fd[4] : pxy1; pxz0=fd[7] ? 2f0*pxz1-fd[5] : pxz1; pyz0=fd[7] ? 2f0*pyz1-fd[6] : pyz1; v0=true
        end
    end
    if v2 != v3
        if v2
            fd=get_pi(pxm,py1,pz0)
            pxx3=fd[7] ? 2f0*pxx2-fd[1] : pxx2; pyy3=fd[7] ? 2f0*pyy2-fd[2] : pyy2; pzz3=fd[7] ? 2f0*pzz2-fd[3] : pzz2
            pxy3=fd[7] ? 2f0*pxy2-fd[4] : pxy2; pxz3=fd[7] ? 2f0*pxz2-fd[5] : pxz2; pyz3=fd[7] ? 2f0*pyz2-fd[6] : pyz2; v3=true
        else
            fd=get_pi(pxp,py1,pz0)
            pxx2=fd[7] ? 2f0*pxx3-fd[1] : pxx3; pyy2=fd[7] ? 2f0*pyy3-fd[2] : pyy3; pzz2=fd[7] ? 2f0*pzz3-fd[3] : pzz3
            pxy2=fd[7] ? 2f0*pxy3-fd[4] : pxy3; pxz2=fd[7] ? 2f0*pxz3-fd[5] : pxz3; pyz2=fd[7] ? 2f0*pyz3-fd[6] : pyz3; v2=true
        end
    end
    if v4 != v5
        if v4
            fd=get_pi(pxm,py0,pz1)
            pxx5=fd[7] ? 2f0*pxx4-fd[1] : pxx4; pyy5=fd[7] ? 2f0*pyy4-fd[2] : pyy4; pzz5=fd[7] ? 2f0*pzz4-fd[3] : pzz4
            pxy5=fd[7] ? 2f0*pxy4-fd[4] : pxy4; pxz5=fd[7] ? 2f0*pxz4-fd[5] : pxz4; pyz5=fd[7] ? 2f0*pyz4-fd[6] : pyz4; v5=true
        else
            fd=get_pi(pxp,py0,pz1)
            pxx4=fd[7] ? 2f0*pxx5-fd[1] : pxx5; pyy4=fd[7] ? 2f0*pyy5-fd[2] : pyy5; pzz4=fd[7] ? 2f0*pzz5-fd[3] : pzz5
            pxy4=fd[7] ? 2f0*pxy5-fd[4] : pxy5; pxz4=fd[7] ? 2f0*pxz5-fd[5] : pxz5; pyz4=fd[7] ? 2f0*pyz5-fd[6] : pyz5; v4=true
        end
    end
    if v6 != v7
        if v6
            fd=get_pi(pxm,py1,pz1)
            pxx7=fd[7] ? 2f0*pxx6-fd[1] : pxx6; pyy7=fd[7] ? 2f0*pyy6-fd[2] : pyy6; pzz7=fd[7] ? 2f0*pzz6-fd[3] : pzz6
            pxy7=fd[7] ? 2f0*pxy6-fd[4] : pxy6; pxz7=fd[7] ? 2f0*pxz6-fd[5] : pxz6; pyz7=fd[7] ? 2f0*pyz6-fd[6] : pyz6; v7=true
        else
            fd=get_pi(pxp,py1,pz1)
            pxx6=fd[7] ? 2f0*pxx7-fd[1] : pxx7; pyy6=fd[7] ? 2f0*pyy7-fd[2] : pyy7; pzz6=fd[7] ? 2f0*pzz7-fd[3] : pzz7
            pxy6=fd[7] ? 2f0*pxy7-fd[4] : pxy7; pxz6=fd[7] ? 2f0*pxz7-fd[5] : pxz7; pyz6=fd[7] ? 2f0*pyz7-fd[6] : pyz7; v6=true
        end
    end

    # Y pass: pairs (0,2), (1,3), (4,6), (5,7).
    if v0 != v2
        if v0
            fd=get_pi(px0,pym,pz0)
            pxx2=fd[7] ? 2f0*pxx0-fd[1] : pxx0; pyy2=fd[7] ? 2f0*pyy0-fd[2] : pyy0; pzz2=fd[7] ? 2f0*pzz0-fd[3] : pzz0
            pxy2=fd[7] ? 2f0*pxy0-fd[4] : pxy0; pxz2=fd[7] ? 2f0*pxz0-fd[5] : pxz0; pyz2=fd[7] ? 2f0*pyz0-fd[6] : pyz0; v2=true
        else
            fd=get_pi(px0,pyp,pz0)
            pxx0=fd[7] ? 2f0*pxx2-fd[1] : pxx2; pyy0=fd[7] ? 2f0*pyy2-fd[2] : pyy2; pzz0=fd[7] ? 2f0*pzz2-fd[3] : pzz2
            pxy0=fd[7] ? 2f0*pxy2-fd[4] : pxy2; pxz0=fd[7] ? 2f0*pxz2-fd[5] : pxz2; pyz0=fd[7] ? 2f0*pyz2-fd[6] : pyz2; v0=true
        end
    end
    if v1 != v3
        if v1
            fd=get_pi(px1,pym,pz0)
            pxx3=fd[7] ? 2f0*pxx1-fd[1] : pxx1; pyy3=fd[7] ? 2f0*pyy1-fd[2] : pyy1; pzz3=fd[7] ? 2f0*pzz1-fd[3] : pzz1
            pxy3=fd[7] ? 2f0*pxy1-fd[4] : pxy1; pxz3=fd[7] ? 2f0*pxz1-fd[5] : pxz1; pyz3=fd[7] ? 2f0*pyz1-fd[6] : pyz1; v3=true
        else
            fd=get_pi(px1,pyp,pz0)
            pxx1=fd[7] ? 2f0*pxx3-fd[1] : pxx3; pyy1=fd[7] ? 2f0*pyy3-fd[2] : pyy3; pzz1=fd[7] ? 2f0*pzz3-fd[3] : pzz3
            pxy1=fd[7] ? 2f0*pxy3-fd[4] : pxy3; pxz1=fd[7] ? 2f0*pxz3-fd[5] : pxz3; pyz1=fd[7] ? 2f0*pyz3-fd[6] : pyz3; v1=true
        end
    end
    if v4 != v6
        if v4
            fd=get_pi(px0,pym,pz1)
            pxx6=fd[7] ? 2f0*pxx4-fd[1] : pxx4; pyy6=fd[7] ? 2f0*pyy4-fd[2] : pyy4; pzz6=fd[7] ? 2f0*pzz4-fd[3] : pzz4
            pxy6=fd[7] ? 2f0*pxy4-fd[4] : pxy4; pxz6=fd[7] ? 2f0*pxz4-fd[5] : pxz4; pyz6=fd[7] ? 2f0*pyz4-fd[6] : pyz4; v6=true
        else
            fd=get_pi(px0,pyp,pz1)
            pxx4=fd[7] ? 2f0*pxx6-fd[1] : pxx6; pyy4=fd[7] ? 2f0*pyy6-fd[2] : pyy6; pzz4=fd[7] ? 2f0*pzz6-fd[3] : pzz6
            pxy4=fd[7] ? 2f0*pxy6-fd[4] : pxy6; pxz4=fd[7] ? 2f0*pxz6-fd[5] : pxz6; pyz4=fd[7] ? 2f0*pyz6-fd[6] : pyz6; v4=true
        end
    end
    if v5 != v7
        if v5
            fd=get_pi(px1,pym,pz1)
            pxx7=fd[7] ? 2f0*pxx5-fd[1] : pxx5; pyy7=fd[7] ? 2f0*pyy5-fd[2] : pyy5; pzz7=fd[7] ? 2f0*pzz5-fd[3] : pzz5
            pxy7=fd[7] ? 2f0*pxy5-fd[4] : pxy5; pxz7=fd[7] ? 2f0*pxz5-fd[5] : pxz5; pyz7=fd[7] ? 2f0*pyz5-fd[6] : pyz5; v7=true
        else
            fd=get_pi(px1,pyp,pz1)
            pxx5=fd[7] ? 2f0*pxx7-fd[1] : pxx7; pyy5=fd[7] ? 2f0*pyy7-fd[2] : pyy7; pzz5=fd[7] ? 2f0*pzz7-fd[3] : pzz7
            pxy5=fd[7] ? 2f0*pxy7-fd[4] : pxy7; pxz5=fd[7] ? 2f0*pxz7-fd[5] : pxz7; pyz5=fd[7] ? 2f0*pyz7-fd[6] : pyz7; v5=true
        end
    end

    # Z pass: pairs (0,4), (1,5), (2,6), (3,7).
    if v0 != v4
        if v0
            fd=get_pi(px0,py0,pzm)
            pxx4=fd[7] ? 2f0*pxx0-fd[1] : pxx0; pyy4=fd[7] ? 2f0*pyy0-fd[2] : pyy0; pzz4=fd[7] ? 2f0*pzz0-fd[3] : pzz0
            pxy4=fd[7] ? 2f0*pxy0-fd[4] : pxy0; pxz4=fd[7] ? 2f0*pxz0-fd[5] : pxz0; pyz4=fd[7] ? 2f0*pyz0-fd[6] : pyz0; v4=true
        else
            fd=get_pi(px0,py0,pzp)
            pxx0=fd[7] ? 2f0*pxx4-fd[1] : pxx4; pyy0=fd[7] ? 2f0*pyy4-fd[2] : pyy4; pzz0=fd[7] ? 2f0*pzz4-fd[3] : pzz4
            pxy0=fd[7] ? 2f0*pxy4-fd[4] : pxy4; pxz0=fd[7] ? 2f0*pxz4-fd[5] : pxz4; pyz0=fd[7] ? 2f0*pyz4-fd[6] : pyz4; v0=true
        end
    end
    if v1 != v5
        if v1
            fd=get_pi(px1,py0,pzm)
            pxx5=fd[7] ? 2f0*pxx1-fd[1] : pxx1; pyy5=fd[7] ? 2f0*pyy1-fd[2] : pyy1; pzz5=fd[7] ? 2f0*pzz1-fd[3] : pzz1
            pxy5=fd[7] ? 2f0*pxy1-fd[4] : pxy1; pxz5=fd[7] ? 2f0*pxz1-fd[5] : pxz1; pyz5=fd[7] ? 2f0*pyz1-fd[6] : pyz1; v5=true
        else
            fd=get_pi(px1,py0,pzp)
            pxx1=fd[7] ? 2f0*pxx5-fd[1] : pxx5; pyy1=fd[7] ? 2f0*pyy5-fd[2] : pyy5; pzz1=fd[7] ? 2f0*pzz5-fd[3] : pzz5
            pxy1=fd[7] ? 2f0*pxy5-fd[4] : pxy5; pxz1=fd[7] ? 2f0*pxz5-fd[5] : pxz5; pyz1=fd[7] ? 2f0*pyz5-fd[6] : pyz5; v1=true
        end
    end
    if v2 != v6
        if v2
            fd=get_pi(px0,py1,pzm)
            pxx6=fd[7] ? 2f0*pxx2-fd[1] : pxx2; pyy6=fd[7] ? 2f0*pyy2-fd[2] : pyy2; pzz6=fd[7] ? 2f0*pzz2-fd[3] : pzz2
            pxy6=fd[7] ? 2f0*pxy2-fd[4] : pxy2; pxz6=fd[7] ? 2f0*pxz2-fd[5] : pxz2; pyz6=fd[7] ? 2f0*pyz2-fd[6] : pyz2; v6=true
        else
            fd=get_pi(px0,py1,pzp)
            pxx2=fd[7] ? 2f0*pxx6-fd[1] : pxx6; pyy2=fd[7] ? 2f0*pyy6-fd[2] : pyy6; pzz2=fd[7] ? 2f0*pzz6-fd[3] : pzz6
            pxy2=fd[7] ? 2f0*pxy6-fd[4] : pxy6; pxz2=fd[7] ? 2f0*pxz6-fd[5] : pxz6; pyz2=fd[7] ? 2f0*pyz6-fd[6] : pyz6; v2=true
        end
    end
    if v3 != v7
        if v3
            fd=get_pi(px1,py1,pzm)
            pxx7=fd[7] ? 2f0*pxx3-fd[1] : pxx3; pyy7=fd[7] ? 2f0*pyy3-fd[2] : pyy3; pzz7=fd[7] ? 2f0*pzz3-fd[3] : pzz3
            pxy7=fd[7] ? 2f0*pxy3-fd[4] : pxy3; pxz7=fd[7] ? 2f0*pxz3-fd[5] : pxz3; pyz7=fd[7] ? 2f0*pyz3-fd[6] : pyz3; v7=true
        else
            fd=get_pi(px1,py1,pzp)
            pxx3=fd[7] ? 2f0*pxx7-fd[1] : pxx7; pyy3=fd[7] ? 2f0*pyy7-fd[2] : pyy7; pzz3=fd[7] ? 2f0*pzz7-fd[3] : pzz7
            pxy3=fd[7] ? 2f0*pxy7-fd[4] : pxy7; pxz3=fd[7] ? 2f0*pxz7-fd[5] : pxz7; pyz3=fd[7] ? 2f0*pyz7-fd[6] : pyz7; v3=true
        end
    end

    pxx = w000*pxx0 + w100*pxx1 + w010*pxx2 + w110*pxx3 + w001*pxx4 + w101*pxx5 + w011*pxx6 + w111*pxx7
    pyy = w000*pyy0 + w100*pyy1 + w010*pyy2 + w110*pyy3 + w001*pyy4 + w101*pyy5 + w011*pyy6 + w111*pyy7
    pzz = w000*pzz0 + w100*pzz1 + w010*pzz2 + w110*pzz3 + w001*pzz4 + w101*pzz5 + w011*pzz6 + w111*pzz7
    pxy = w000*pxy0 + w100*pxy1 + w010*pxy2 + w110*pxy3 + w001*pxy4 + w101*pxy5 + w011*pxy6 + w111*pxy7
    pxz = w000*pxz0 + w100*pxz1 + w010*pxz2 + w110*pxz3 + w001*pxz4 + w101*pxz5 + w011*pxz6 + w111*pxz7
    pyz = w000*pyz0 + w100*pyz1 + w010*pyz2 + w110*pyz3 + w001*pyz4 + w101*pyz5 + w011*pyz6 + w111*pyz7
    return (pxx, pyy, pzz, pxy, pxz, pyz, true)
end

# @noinline (2026-06-04, profiling workflow): this ~390-line function (3 inner
# closures, a 4×4×4 cubic stencil loop, the B-spline branch, and the C3 quintic
# 5-anchor temporal branch) is called ONLY on the AMR (L>1) path
# (kernel_d3q19_rr.jl:283). Inlining it into the stream-collide kernel blew the
# register frame — the sibling D3Q39 kernel was measured at 254 regs/thread with
# 3120 B/thread SPILL to local (VRAM-backed) memory and ~1 block/SM occupancy,
# producing the 100%-util/29W (latency-bound, ALUs-stalled) signature. As a
# separate compiled unit its register footprint leaves the kernel's live frame,
# raising occupancy. Direct precedent: sample_parent_macro_for_fine_cell above
# was made @noinline for exactly this reason. Byte-identical (only the inline
# boundary moves). NOTE: the flow launch must NOT force always_inline=true
# (CUDABackend() default is false → @noinline honored).
@noinline function interpolate_with_rescaling(
    parent_f_new, parent_rho_new, parent_vel_new,
    parent_f_old, parent_rho_old, parent_vel_old,
    parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    fine_gx::Int32, fine_gy::Int32, fine_gz::Int32,
    k::Int32, block_size::Int32,
    w_k::Float32, cx::Float32, cy::Float32, cz::Float32,
    tau_coarse::Float32, tau_fine::Float32,
    temporal_weight::Float32,
    use_temporal_interp::Int32,
    compressible_corr::Int32 = Int32(0),
    entropic_mode::Int32 = Int32(0),
    # AMR-CONTINUITY Stage-2 (2026-06-03): when 1, the trilinear macro fallback
    # (used at every refinement-box outer ring where the cubic 4×4×4 stencil
    # pokes outside the parent's active region) reconstructs invalid corners by
    # gradient-preserving linear extrapolation (missing = 2·near − far) instead
    # of substituting the central v000 corner. v000-substitution is an order
    # change vs the cubic interior → a DC per-level offset on gradient fields
    # (the residual transverse AMR imprint). Byte-identical on uniform fields
    # (extrapolating a constant returns the constant). DEFAULT 0 (legacy v000)
    # so D3Q27 cumulant / regularized-BGK call sites are unchanged; only the
    # D3Q19 kernel passes 1.
    use_extrap_fallback::Int32 = Int32(0),
    # C2/C4 QUINTIC B-SPLINE prolongation (2026-06-04): when use_bspline_macro==1,
    # the macro fields (rho,u) are evaluated from the parent's PRECOMPUTED B-spline
    # coefficients (rho_spline_coeff/vel_spline_coeff, filled by the per-coarse-step
    # prefilter) via the compact 6-tap @noinline interpolate_macro_bspline — giving
    # C4 continuity + exact 2:1 conservation. Falls through to cubic/extrap on any
    # inactive tap. DEFAULT off + stub arrays (nothing) so non-D3Q19 callers are
    # unchanged. f_neq carrier (trilin(7)) is unchanged — only the macro changes.
    rho_spline_coeff = nothing,
    vel_spline_coeff = nothing,
    use_bspline_macro::Int32 = Int32(0),
    # C3 TEMPORAL (2026-06-04): higher-order time anchors for the D3Q19 path.
    # When use_quintic_temporal==1 AND all four extra anchor arrays are non-nothing,
    # the linear 2-anchor (old,new) time blend is replaced by a 5-anchor quintic
    # Lagrange blend (C³ in time) over {oldest2,oldest,older,old,new}, soft-clamped
    # to [old,new]±10%. Falls back to linear when any anchor array is nothing
    # (chain not yet populated — first 4 coarse steps) or the flag is 0. DEFAULT
    # off + nothing so non-D3Q19 callers are byte-identical.
    parent_rho_older = nothing,   parent_vel_older = nothing,
    parent_rho_oldest = nothing,  parent_vel_oldest = nothing,
    parent_rho_oldest2 = nothing, parent_vel_oldest2 = nothing,
    use_quintic_temporal::Int32 = Int32(0),
    # advanced.refinement.level_viscosity (lbm-taufix, 28 Oct 2026): 1 = "physical" level taus. The f_neq carrier is
    # then rescaled by the pre-collision ratio (tau dt)_fine/(tau dt)_coarse = tau_fine/(2 tau_coarse) (dt_c = 2 dt_f)
    # instead of the lattice-viscosity ratio (tau_f-1/2)/(tau_c-1/2), which equals 1/2 only under the legacy level
    # rule. DEFAULT 0 ⇒ every existing call is unchanged.
    level_visc_physical::Int32 = Int32(0)
)

    px_cont = _fine_to_parent_interp_coord(fine_gx)
    py_cont = _fine_to_parent_interp_coord(fine_gy)
    pz_cont = _fine_to_parent_interp_coord(fine_gz)

    parent_nx = parent_dim_x * block_size
    parent_ny = parent_dim_y * block_size
    parent_nz = parent_dim_z * block_size
    px0, px1, wx = _parent_interp_bracket(px_cont, parent_nx)
    py0, py1, wy = _PERIODIC_Y_AMR ? _pbracket_y(py_cont) : _parent_interp_bracket(py_cont, parent_ny)
    pz0, pz1, wz = _parent_interp_bracket(pz_cont, parent_nz)
    px0_cubic = px0
    py0_cubic = py0
    pz0_cubic = pz0
    px1_cubic = px1
    py1_cubic = py1
    pz1_cubic = pz1
    # MIRROR-GHOST (round 8): separate ghost-capable y/z brackets for the MACRO
    # path only (cubic eval/probe + extrap fallback). The trilinear f/neq-carrier
    # keeps the legacy clamped brackets (no population mirroring needed). When
    # the const is off these are identical to the legacy brackets (DCE'd).
    mpy0 = py0
    mwy = wy
    mpz0 = pz0
    mwz = wz
    if _AMR_MIRROR_GHOST_INTERP
        mpy0, _, mwy = _parent_interp_bracket_mirror(py_cont, parent_ny)
        mpz0, _, mwz = _parent_interp_bracket_mirror(pz_cont, parent_nz)
    end


    @inline function get_blended(pgx::Int32, pgy::Int32, pgz::Int32)
        pbx = (pgx - Int32(1)) ÷ block_size + Int32(1)
        pgy = _pwrap_y(pgy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
        pby = (pgy - Int32(1)) ÷ block_size + Int32(1)
        pbz = (pgz - Int32(1)) ÷ block_size + Int32(1)

        if pbx >= Int32(1) && pbx <= parent_dim_x &&
           pby >= Int32(1) && pby <= parent_dim_y &&
           pbz >= Int32(1) && pbz <= parent_dim_z
            pb_idx = parent_ptr[pbx, pby, pbz]
            if pb_idx > Int32(0)
                plx = (pgx - Int32(1)) % block_size + Int32(1)
                ply = (pgy - Int32(1)) % block_size + Int32(1)
                plz = (pgz - Int32(1)) % block_size + Int32(1)

                f_new = parent_f_new[plx, ply, plz, pb_idx, k]
                rho_new = parent_rho_new[plx, ply, plz, pb_idx]
                ux_new = parent_vel_new[plx, ply, plz, pb_idx, 1]
                uy_new = parent_vel_new[plx, ply, plz, pb_idx, 2]
                uz_new = parent_vel_new[plx, ply, plz, pb_idx, 3]

                # `parent_*_old !== nothing` guard (2026-06-09): lets the D3Q19 halo
                # kernel pass the `nothing` singleton for the _old arrays when temporal
                # interpolation is OFF (drops 3 live array pointers from the register-
                # starved halo kernel). Nothing-typed → statically false → DCE'd before
                # type-checking the indexing. Real arrays → statically true → identical
                # to the prior `use_temporal_interp == 1` path (byte-identical for the
                # D3Q27/D3Q39/monolith callers that always pass real _old arrays).
                if use_temporal_interp == Int32(1) && temporal_weight < 0.99f0 &&
                   parent_f_old !== nothing && parent_rho_old !== nothing && parent_vel_old !== nothing
                    f_old = parent_f_old[plx, ply, plz, pb_idx, k]
                    rho_old = parent_rho_old[plx, ply, plz, pb_idx]
                    ux_old = parent_vel_old[plx, ply, plz, pb_idx, 1]
                    uy_old = parent_vel_old[plx, ply, plz, pb_idx, 2]
                    uz_old = parent_vel_old[plx, ply, plz, pb_idx, 3]

                    tw = temporal_weight
                    # C3 TEMPORAL: quintic Lagrange (5-anchor) blend when enabled and
                    # the full anchor chain is populated; else linear (2-anchor).
                    if use_quintic_temporal == Int32(1) &&
                       parent_rho_older !== nothing && parent_rho_oldest !== nothing &&
                       parent_rho_oldest2 !== nothing
                        wq = _quintic_lagrange_time_weights(tw)
                        # NOTE: f anchors only go back 1 step (old);
                        # higher-order f anchors are not stored. For the f_neq carrier we
                        # keep the LINEAR f blend (f_neq is tiny; its temporal order is not
                        # the visible lever) and apply quintic ONLY to the macro (rho,u),
                        # which drive the equilibrium that dominates the seam.
                        f_b   = f_old*(1.0f0-tw) + f_new*tw
                        ro_or = parent_rho_older[plx, ply, plz, pb_idx]
                        ro_o1 = parent_rho_oldest[plx, ply, plz, pb_idx]
                        ro_o2 = parent_rho_oldest2[plx, ply, plz, pb_idx]
                        rho_b = _quintic_soft_clamp(
                            wq[1]*ro_o2 + wq[2]*ro_o1 + wq[3]*ro_or + wq[4]*rho_old + wq[5]*rho_new,
                            rho_old, rho_new)
                        ux_or = parent_vel_older[plx, ply, plz, pb_idx, 1]
                        ux_o1 = parent_vel_oldest[plx, ply, plz, pb_idx, 1]
                        ux_o2 = parent_vel_oldest2[plx, ply, plz, pb_idx, 1]
                        ux_b  = _quintic_soft_clamp(
                            wq[1]*ux_o2 + wq[2]*ux_o1 + wq[3]*ux_or + wq[4]*ux_old + wq[5]*ux_new,
                            ux_old, ux_new)
                        uy_or = parent_vel_older[plx, ply, plz, pb_idx, 2]
                        uy_o1 = parent_vel_oldest[plx, ply, plz, pb_idx, 2]
                        uy_o2 = parent_vel_oldest2[plx, ply, plz, pb_idx, 2]
                        uy_b  = _quintic_soft_clamp(
                            wq[1]*uy_o2 + wq[2]*uy_o1 + wq[3]*uy_or + wq[4]*uy_old + wq[5]*uy_new,
                            uy_old, uy_new)
                        uz_or = parent_vel_older[plx, ply, plz, pb_idx, 3]
                        uz_o1 = parent_vel_oldest[plx, ply, plz, pb_idx, 3]
                        uz_o2 = parent_vel_oldest2[plx, ply, plz, pb_idx, 3]
                        uz_b  = _quintic_soft_clamp(
                            wq[1]*uz_o2 + wq[2]*uz_o1 + wq[3]*uz_or + wq[4]*uz_old + wq[5]*uz_new,
                            uz_old, uz_new)
                    else
                        f_b   = f_old*(1.0f0-tw) + f_new*tw
                        rho_b = rho_old*(1.0f0-tw) + rho_new*tw
                        ux_b  = ux_old*(1.0f0-tw) + ux_new*tw
                        uy_b  = uy_old*(1.0f0-tw) + uy_new*tw
                        uz_b  = uz_old*(1.0f0-tw) + uz_new*tw
                    end
                    # AMR-continuity fix (2026-06-03): per-corner TRUE non-equilibrium.
                    # The reconstruction below interpolates this (small) neq trilinearly
                    # and adds it to the CUBIC equilibrium — instead of (trilinear f −
                    # cubic feq), which injected the trilinear-vs-cubic equilibrium
                    # interpolation-order gap as spurious neq → the per-level AMR shade
                    # step. In uniform freestream feq_corner==f_b ⇒ fneq=0 (byte-identical).
                    feqc = calculate_equilibrium_auto(rho_b, ux_b, uy_b, uz_b, w_k, cx, cy, cz, compressible_corr)
                    return (f_b, rho_b, ux_b, uy_b, uz_b, true, f_b - feqc)
                end
                feqc = calculate_equilibrium_auto(rho_new, ux_new, uy_new, uz_new, w_k, cx, cy, cz, compressible_corr)
                return (f_new, rho_new, ux_new, uy_new, uz_new, true, f_new - feqc)
            end
        end
        return (w_k, 1.0f0, 0.0f0, 0.0f0, 0.0f0, false, 0.0f0)
    end

    # ── Fetch macroscopic field from a parent cell (rho, u only) ──
    @inline function get_macro(pgx::Int32, pgy_req::Int32, pgz_req::Int32)
        # MIRROR-GHOST (round 8): resolve ghost rows by specular reflection
        # (ρ/u_tangential even, u_normal odd). DCE'd when the const is off.
        pgy = pgy_req
        pgz = pgz_req
        sy = 1.0f0
        sz = 1.0f0
        if _AMR_MIRROR_GHOST_INTERP
            pgy, sy = _mirror_reflect_parent_axis(pgy_req, parent_ny)
            pgz, sz = _mirror_reflect_parent_axis(pgz_req, parent_nz)
        end
        pbx = (pgx - Int32(1)) ÷ block_size + Int32(1)
        pgy = _pwrap_y(pgy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
        pby = (pgy - Int32(1)) ÷ block_size + Int32(1)
        pbz = (pgz - Int32(1)) ÷ block_size + Int32(1)
        if pbx >= Int32(1) && pbx <= parent_dim_x &&
           pby >= Int32(1) && pby <= parent_dim_y &&
           pbz >= Int32(1) && pbz <= parent_dim_z
            pb_idx = parent_ptr[pbx, pby, pbz]
            if pb_idx > Int32(0)
                plx = (pgx - Int32(1)) % block_size + Int32(1)
                ply = (pgy - Int32(1)) % block_size + Int32(1)
                plz = (pgz - Int32(1)) % block_size + Int32(1)
                rho_v = parent_rho_new[plx, ply, plz, pb_idx]
                ux_v = parent_vel_new[plx, ply, plz, pb_idx, 1]
                uy_v = parent_vel_new[plx, ply, plz, pb_idx, 2]
                uz_v = parent_vel_new[plx, ply, plz, pb_idx, 3]
                # `parent_*_old !== nothing` guard (2026-06-09) — see get_blended above.
                if use_temporal_interp == Int32(1) && temporal_weight < 0.99f0 &&
                   parent_rho_old !== nothing && parent_vel_old !== nothing
                    tw = temporal_weight
                    rho_old_m = parent_rho_old[plx, ply, plz, pb_idx]
                    ux_old_m  = parent_vel_old[plx, ply, plz, pb_idx, 1]
                    uy_old_m  = parent_vel_old[plx, ply, plz, pb_idx, 2]
                    uz_old_m  = parent_vel_old[plx, ply, plz, pb_idx, 3]
                    if use_quintic_temporal == Int32(1) &&
                       parent_rho_older !== nothing && parent_rho_oldest !== nothing &&
                       parent_rho_oldest2 !== nothing
                        wq = _quintic_lagrange_time_weights(tw)
                        rho_v = _quintic_soft_clamp(
                            wq[1]*parent_rho_oldest2[plx,ply,plz,pb_idx] +
                            wq[2]*parent_rho_oldest[plx,ply,plz,pb_idx] +
                            wq[3]*parent_rho_older[plx,ply,plz,pb_idx] +
                            wq[4]*rho_old_m + wq[5]*rho_v, rho_old_m, rho_v)
                        ux_v = _quintic_soft_clamp(
                            wq[1]*parent_vel_oldest2[plx,ply,plz,pb_idx,1] +
                            wq[2]*parent_vel_oldest[plx,ply,plz,pb_idx,1] +
                            wq[3]*parent_vel_older[plx,ply,plz,pb_idx,1] +
                            wq[4]*ux_old_m + wq[5]*ux_v, ux_old_m, ux_v)
                        uy_v = _quintic_soft_clamp(
                            wq[1]*parent_vel_oldest2[plx,ply,plz,pb_idx,2] +
                            wq[2]*parent_vel_oldest[plx,ply,plz,pb_idx,2] +
                            wq[3]*parent_vel_older[plx,ply,plz,pb_idx,2] +
                            wq[4]*uy_old_m + wq[5]*uy_v, uy_old_m, uy_v)
                        uz_v = _quintic_soft_clamp(
                            wq[1]*parent_vel_oldest2[plx,ply,plz,pb_idx,3] +
                            wq[2]*parent_vel_oldest[plx,ply,plz,pb_idx,3] +
                            wq[3]*parent_vel_older[plx,ply,plz,pb_idx,3] +
                            wq[4]*uz_old_m + wq[5]*uz_v, uz_old_m, uz_v)
                    else
                        rho_v = rho_old_m * (1.0f0 - tw) + rho_v * tw
                        ux_v  = ux_old_m  * (1.0f0 - tw) + ux_v  * tw
                        uy_v  = uy_old_m  * (1.0f0 - tw) + uy_v  * tw
                        uz_v  = uz_old_m  * (1.0f0 - tw) + uz_v  * tw
                    end
                end
                if _AMR_MIRROR_GHOST_INTERP
                    uy_v *= sy
                    uz_v *= sz
                end
                return (rho_v, ux_v, uy_v, uz_v, true)
            end
        end
        return (1.0f0, 0.0f0, 0.0f0, 0.0f0, false)
    end

    # ── Cubic interpolation of macroscopic fields (rho, u) ──
    # Uses the Palabos-style cubic Lagrange stencil for C1 continuity at
    # refinement interfaces.  Reduces AMR banding artifacts by providing
    # a smoother equilibrium at the coarse-fine boundary.
    # The distribution f_k is still trilinearly interpolated (cost-effective)
    # because f_neq is rescaled via the cubic-interpolated equilibrium anyway.
    # CORNER-QUAD FIX (2026-06-09): SYMMETRIC cubic-validity margin.
    # The 4-point cubic stencil {px0-1, px0, px0+1, px0+2} reaches 1 parent cell
    # toward − and 2 toward +. The previous validity test mirrored that asymmetry
    # (px0-1>=1 on −, px0+2<=parent_n on +), so on the +X/+Y/+Z box faces the cubic
    # fell back to extrapolation ONE PARENT CELL EARLIER than on the − faces — a
    # directional bias on the + faces only that compounds where two + faces meet
    # (and the mirror − corner) into the "isolated quads at two OPPOSITE corners"
    # seen in density/schlieren (present with the monolithic kernel too, since it
    # is in the shared prolongation). A SYMMETRIC 2-cell margin on BOTH sides makes
    # the cubic→extrap transition identical on + and − faces, removing the bias.
    # MIRROR-GHOST (round 8): with specular ghosts the cubic y/z stencil is
    # always completable across the mirror planes (every requested ghost row
    # reflects to an in-domain row for parent_n ≥ 3); only the box-interior
    # validity probe (cubic_all_valid) can still reject. The x axis (inlet/
    # outlet, not a mirror) keeps the legacy symmetric margin.
    use_cubic =
        px0 - Int32(2) >= Int32(1) && px0 + Int32(2) <= parent_nx &&
        # PERIODIC-Y AMR: the periodic Y axis never truncates the stencil (the
        # lookups wrap); only the active-parent validity probe can reject.
        (_PERIODIC_Y_AMR ? parent_ny >= Int32(4) :
         _AMR_MIRROR_GHOST_INTERP ? parent_ny >= Int32(3) :
            (py0 - Int32(2) >= Int32(1) && py0 + Int32(2) <= parent_ny)) &&
        (_AMR_MIRROR_GHOST_INTERP ? parent_nz >= Int32(3) :
            (pz0 - Int32(2) >= Int32(1) && pz0 + Int32(2) <= parent_nz))

    # ── Trilinear interpolation of f_k (always) ──
    d000 = get_blended(px0, py0, pz0)
    d100 = get_blended(px1, py0, pz0)
    d010 = get_blended(px0, py1, pz0)
    d110 = get_blended(px1, py1, pz0)
    d001 = get_blended(px0, py0, pz1)
    d101 = get_blended(px1, py0, pz1)
    d011 = get_blended(px0, py1, pz1)
    d111 = get_blended(px1, py1, pz1)

    valid_anchor =
        d000[6] ? d000 :
        d100[6] ? d100 :
        d010[6] ? d010 :
        d110[6] ? d110 :
        d001[6] ? d001 :
        d101[6] ? d101 :
        d011[6] ? d011 : d111

    v000 = d000[6] ? d000 : valid_anchor
    v100 = d100[6] ? d100 : valid_anchor
    v010 = d010[6] ? d010 : valid_anchor
    v110 = d110[6] ? d110 : valid_anchor
    v001 = d001[6] ? d001 : valid_anchor
    v101 = d101[6] ? d101 : valid_anchor
    v011 = d011[6] ? d011 : valid_anchor
    v111 = d111[6] ? d111 : valid_anchor

    @inline function trilin(i)
        c00 = v000[i]*(1.0f0-wx) + v100[i]*wx
        c01 = v001[i]*(1.0f0-wx) + v101[i]*wx
        c10 = v010[i]*(1.0f0-wx) + v110[i]*wx
        c11 = v011[i]*(1.0f0-wx) + v111[i]*wx
        c0 = c00*(1.0f0-wy) + c10*wy
        c1 = c01*(1.0f0-wy) + c11*wy
        return c0*(1.0f0-wz) + c1*wz
    end

    # (trilin(1) — full-population interpolation — is no longer used: the AMR-
    # continuity fix below reconstructs from cubic feq + trilinear true-neq instead.)

    # Validate cubic stencil: all 4×4×4 parent cells must lie in active parent
    # blocks, otherwise the cubic weighted sum would inject the (rho=1, u=0)
    # fallback from get_macro with non-zero weight — and because the stencil
    # reaches two cells toward +primary and only one toward −primary (it is
    # {px0-1, px0, px0+1, px0+2}), that injection is directional: it biases
    # the prolongated ρ/u on the +X/+Y/+Z faces of the AMR box only. That
    # manifested downstream as a dark-then-bright schlieren dipole at every
    # fine/halo outer face on one side but not the mirror side.
    cubic_all_valid = use_cubic
    if use_cubic
        for ix in Int32(-1):Int32(2), iy in Int32(-1):Int32(2), iz in Int32(-1):Int32(2)
            m = get_macro(px0 + ix, mpy0 + iy, mpz0 + iz)
            if !m[5]
                cubic_all_valid = false
                break
            end
        end
    end

    # ── C2/C4 quintic B-spline macro evaluation (top priority when enabled) ──
    # Reads the parent's precomputed B-spline coefficients via the compact 6-tap
    # @noinline interpolate_macro_bspline. On success (all taps active) it yields
    # C4-continuous + exactly-conservative macros; on any inactive tap it returns
    # ok=false and we fall through to the cubic→extrap cascade below unchanged.
    _bspline_ok = false
    local rho_int, ux_int, uy_int, uz_int
    if use_bspline_macro == Int32(1) && rho_spline_coeff !== nothing && vel_spline_coeff !== nothing
        rb, uxb, uyb, uzb, okb = interpolate_macro_bspline(
            rho_spline_coeff, vel_spline_coeff, parent_ptr,
            parent_dim_x, parent_dim_y, parent_dim_z,
            fine_gx, fine_gy, fine_gz, block_size)
        if okb
            rho_int = rb; ux_int = uxb; uy_int = uyb; uz_int = uzb
            _bspline_ok = true
        end
    end

    if _bspline_ok
        # already set above
    elseif cubic_all_valid
        # Cubic Lagrange interpolation of macroscopic fields over 4×4×4 stencil.
        rho_int = 0.0f0
        ux_int = 0.0f0
        uy_int = 0.0f0
        uz_int = 0.0f0
        for ix in Int32(-1):Int32(2)
            px = px0 + ix
            for iy in Int32(-1):Int32(2)
                py = mpy0 + iy
                for iz in Int32(-1):Int32(2)
                    pz = mpz0 + iz
                    m = get_macro(px, py, pz)
                    r, uxv, uyv, uzv, _ = m
                    # Separable cubic weights (y/z use the MIRROR brackets)
                    wxv = ix == Int32(-1) ? (-wx*(wx-1.0f0)*(wx-2.0f0)/6.0f0) :
                          ix == Int32(0)  ? ((wx+1.0f0)*(wx-1.0f0)*(wx-2.0f0)/2.0f0) :
                          ix == Int32(1)  ? (-wx*(wx+1.0f0)*(wx-2.0f0)/2.0f0) :
                                            (wx*(wx+1.0f0)*(wx-1.0f0)/6.0f0)
                    wyv = iy == Int32(-1) ? (-mwy*(mwy-1.0f0)*(mwy-2.0f0)/6.0f0) :
                          iy == Int32(0)  ? ((mwy+1.0f0)*(mwy-1.0f0)*(mwy-2.0f0)/2.0f0) :
                          iy == Int32(1)  ? (-mwy*(mwy+1.0f0)*(mwy-2.0f0)/2.0f0) :
                                            (mwy*(mwy+1.0f0)*(mwy-1.0f0)/6.0f0)
                    wzv = iz == Int32(-1) ? (-mwz*(mwz-1.0f0)*(mwz-2.0f0)/6.0f0) :
                          iz == Int32(0)  ? ((mwz+1.0f0)*(mwz-1.0f0)*(mwz-2.0f0)/2.0f0) :
                          iz == Int32(1)  ? (-mwz*(mwz+1.0f0)*(mwz-2.0f0)/2.0f0) :
                                            (mwz*(mwz+1.0f0)*(mwz-1.0f0)/6.0f0)
                    w3 = wxv * wyv * wzv
                    rho_int += w3 * r
                    ux_int += w3 * uxv
                    uy_int += w3 * uyv
                    uz_int += w3 * uzv
                end
            end
        end
    elseif use_extrap_fallback == Int32(1)
        # AMR-CONTINUITY Stage-2 (JIT-safe redo 2026-06-04): gradient-preserving
        # partial-stencil macro fallback via the @noinline standalone helper
        # (NOT inline — inlining its ~80-line body here blew the @inline threshold
        # and caused the GPU InvalidIRError). The helper reconstructs invalid box-
        # edge corners by linear extrapolation (2·near−far, gradient-exact) instead
        # of v000-substitution, removing the per-level DC offset on gradient fields.
        rho_int, ux_int, uy_int, uz_int, _extrap_ok = sample_parent_macro_for_fine_cell(
            parent_rho_new, parent_vel_new, parent_rho_old, parent_vel_old, parent_ptr,
            parent_dim_x, parent_dim_y, parent_dim_z,
            fine_gx, fine_gy, fine_gz, block_size,
            temporal_weight, use_temporal_interp)
        if !_extrap_ok
            rho_int = trilin(2); ux_int = trilin(3); uy_int = trilin(4); uz_int = trilin(5)
        end
    else
        # Legacy trilinear fallback: invalid corners substituted with the central
        # valid sample (v000). Order change vs cubic interior on gradient fields,
        # but symmetric and stable. Used by all non-D3Q19 lattices (default).
        rho_int = trilin(2)
        ux_int = trilin(3)
        uy_int = trilin(4)
        uz_int = trilin(5)
    end


    feq_int = calculate_equilibrium_auto(rho_int, ux_int, uy_int, uz_int, w_k, cx, cy, cz, compressible_corr)
    # AMR-continuity fix (2026-06-03/26): use the TRUE per-corner non-equilibrium
    # carrier, f_neq = f - feq, rather than mixing trilinear f with higher-order
    # equilibrium macros. The 2026-06-26 update below also raises that carrier from
    # trilinear to cubic-Lagrange for D3Q27, matching the macro/order used at seams.
    # Keep the non-equilibrium carrier at the same spatial order as the macro
    # equilibrium. A trilinear f_neq carrier leaves a C0 stress piece at the AMR
    # seam, which feeds back into density/pressure as a visible blockage.
    f_neq = 0.0f0
    if use_cubic
        f_neq_anchor = valid_anchor[7]
        for ix in Int32(-1):Int32(2)
            px = px0 + ix
            wxv = ix == Int32(-1) ? (-wx*(wx-1.0f0)*(wx-2.0f0)/6.0f0) :
                  ix == Int32(0)  ? ((wx+1.0f0)*(wx-1.0f0)*(wx-2.0f0)/2.0f0) :
                  ix == Int32(1)  ? (-wx*(wx+1.0f0)*(wx-2.0f0)/2.0f0) :
                                    (wx*(wx+1.0f0)*(wx-1.0f0)/6.0f0)
            for iy in Int32(-1):Int32(2)
                py = py0 + iy
                wyv = iy == Int32(-1) ? (-wy*(wy-1.0f0)*(wy-2.0f0)/6.0f0) :
                      iy == Int32(0)  ? ((wy+1.0f0)*(wy-1.0f0)*(wy-2.0f0)/2.0f0) :
                      iy == Int32(1)  ? (-wy*(wy+1.0f0)*(wy-2.0f0)/2.0f0) :
                                        (wy*(wy+1.0f0)*(wy-1.0f0)/6.0f0)
                for iz in Int32(-1):Int32(2)
                    pz = pz0 + iz
                    wzv = iz == Int32(-1) ? (-wz*(wz-1.0f0)*(wz-2.0f0)/6.0f0) :
                          iz == Int32(0)  ? ((wz+1.0f0)*(wz-1.0f0)*(wz-2.0f0)/2.0f0) :
                          iz == Int32(1)  ? (-wz*(wz+1.0f0)*(wz-2.0f0)/2.0f0) :
                                            (wz*(wz+1.0f0)*(wz-1.0f0)/6.0f0)
                    d = get_blended(px, py, pz)
                    f_neq += wxv * wyv * wzv * (d[6] ? d[7] : f_neq_anchor)
                end
            end
        end
    else
        f_neq = trilin(7)
    end

    tau_c = tau_coarse - 0.5f0
    tau_f = tau_fine - 0.5f0
    scale = tau_c > 1.0f-6 ? clamp(tau_f / tau_c, 0.01f0, 100.0f0) : 1.0f0
    if level_visc_physical == Int32(1)
        # pre-collision non-equilibrium ∝ tau·dt·S_phys: fine/coarse = tau_f dt_f/(tau_c dt_c) = tau_f/(2 tau_c)
        scale = clamp(tau_fine / (2.0f0 * max(tau_coarse, 0.500001f0)), 0.01f0, 100.0f0)
    end
    scale = _amr_fneq_postcoll_prolong_scale(scale, tau_coarse, tau_fine)

    # For the entropic (KBC) operator, the standard f_neq rescaling is incorrect
    # because the non-equilibrium part contains both a stress component (s, symmetric)
    # and a higher-order component (h, antisymmetric). The s part scales correctly
    # with tau_f/tau_c, but the h part relaxes at a cell-dependent rate 2*beta*alpha
    # which is unknown at the interface. Rescaling h with the viscous rate introduces
    # an antisymmetric error that produces checkerboard artifacts.
    #
    # entropic_mode == 1: equilibrium-only interpolation (VRAM-constrained fallback).
    #   Zeroes both s and h.  Mildly dissipative at interfaces but zero memory cost.
    #   The fine level's H-theorem solver reconstructs the non-equilibrium in ~1 step.
    #
    # For the full mirror-decomposed interpolation (preserves stress), use
    # interpolate_entropic_mirror() instead — see below.
    if entropic_mode == Int32(1)
        scale = 0.0f0
    end

    return feq_int + f_neq * scale
end

@inline function _d3q19_bspline_macro_ready(
    parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    fine_gx::Int32, fine_gy::Int32, fine_gz::Int32,
    block_size::Int32,
)
    p0x, _ = _bspline_axis_setup(fine_gx)
    p0y, _ = _bspline_axis_setup(fine_gy)
    p0z, _ = _bspline_axis_setup(fine_gz)
    return _bspline_symmetric_support_ready(parent_ptr, p0x, p0y, p0z,
                                            parent_dim_x, parent_dim_y, parent_dim_z,
                                            block_size)
end

@inline function _d3q19_bspline_halo_ready(
    ix::Int32, iy::Int32, iz::Int32,
    gx::Int32, gy::Int32, gz::Int32,
    b_idx::Int32,
    block_size::Int32,
    neighbor_table,
    parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    cx_arr, cy_arr, cz_arr,
)
    @inbounds for k in Int32(1):Int32(19)
        cx = cx_arr[k]
        cy = cy_arr[k]
        cz = cz_arr[k]
        sx = ix - cx
        sy = iy - cy
        sz = iz - cz
        if sx < Int32(1) || sx > block_size ||
           sy < Int32(1) || sy > block_size ||
           sz < Int32(1) || sz > block_size
            nb_off_x = sx < Int32(1) ? Int32(-1) : (sx > block_size ? Int32(1) : Int32(0))
            nb_off_y = sy < Int32(1) ? Int32(-1) : (sy > block_size ? Int32(1) : Int32(0))
            nb_off_z = sz < Int32(1) ? Int32(-1) : (sz > block_size ? Int32(1) : Int32(0))
            dir_idx = (nb_off_x + Int32(1)) +
                      (nb_off_y + Int32(1)) * Int32(3) +
                      (nb_off_z + Int32(1)) * Int32(9) + Int32(1)
            if neighbor_table[b_idx, dir_idx] <= Int32(0)
                _d3q19_bspline_macro_ready(
                    parent_ptr,
                    parent_dim_x, parent_dim_y, parent_dim_z,
                    gx - cx, gy - cy, gz - cz,
                    block_size,
                ) || return false
            end
        end
    end
    return true
end

@inline function _d3q19_parent_corner_neq_quintic(
    parent_f_new, parent_rho_new, parent_vel_new,
    parent_f_old, parent_rho_old, parent_vel_old,
    parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    pgx::Int32, pgy::Int32, pgz::Int32,
    k::Int32, block_size::Int32,
    w_k::Float32, cx::Float32, cy::Float32, cz::Float32,
    temporal_weight::Float32,
    use_temporal_interp::Int32,
    parent_rho_older, parent_vel_older,
    parent_rho_oldest, parent_vel_oldest,
    parent_rho_oldest2, parent_vel_oldest2,
    ::Val{USE_QUINTIC_TEMPORAL_VAL},
) where {USE_QUINTIC_TEMPORAL_VAL}
    pbx = div(pgx - Int32(1), block_size) + Int32(1)
    pgy = _pwrap_y(pgy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
    pby = div(pgy - Int32(1), block_size) + Int32(1)
    pbz = div(pgz - Int32(1), block_size) + Int32(1)
    if pbx < Int32(1) || pbx > parent_dim_x ||
       pby < Int32(1) || pby > parent_dim_y ||
       pbz < Int32(1) || pbz > parent_dim_z
        return (0.0f0, false)
    end
    @inbounds begin
        pb_idx = parent_ptr[pbx, pby, pbz]
        pb_idx > Int32(0) || return (0.0f0, false)

        plx = mod(pgx - Int32(1), block_size) + Int32(1)
        ply = mod(pgy - Int32(1), block_size) + Int32(1)
        plz = mod(pgz - Int32(1), block_size) + Int32(1)

        f_new = parent_f_new[plx, ply, plz, pb_idx, k]
        rho_new = parent_rho_new[plx, ply, plz, pb_idx]
        ux_new = parent_vel_new[plx, ply, plz, pb_idx, 1]
        uy_new = parent_vel_new[plx, ply, plz, pb_idx, 2]
        uz_new = parent_vel_new[plx, ply, plz, pb_idx, 3]

        f_b = f_new
        rho_b = rho_new
        ux_b = ux_new
        uy_b = uy_new
        uz_b = uz_new

        # HALO-KERNEL REGISTER REDUCTION (2026-06-09): the `parent_*_old !== nothing`
        # guard lets the D3Q19 halo kernel pass the `nothing` singleton for the _old
        # arrays when temporal interpolation is OFF — dropping 3 live array pointers
        # from the register-starved halo kernel. With parent_f_old::Nothing the guard
        # is statically false → the whole branch is DCE'd before the indexing is
        # type-checked. With real arrays it is statically true → identical to the
        # prior `use_temporal_interp == 1` behaviour (byte-identical for all callers
        # that pass real _old arrays, incl. D3Q27/D3Q39).
        if use_temporal_interp == Int32(1) && temporal_weight < 0.99f0 &&
           parent_f_old !== nothing && parent_rho_old !== nothing && parent_vel_old !== nothing
            tw = temporal_weight
            f_old = parent_f_old[plx, ply, plz, pb_idx, k]
            rho_old = parent_rho_old[plx, ply, plz, pb_idx]
            ux_old = parent_vel_old[plx, ply, plz, pb_idx, 1]
            uy_old = parent_vel_old[plx, ply, plz, pb_idx, 2]
            uz_old = parent_vel_old[plx, ply, plz, pb_idx, 3]
            f_b = f_old * (1.0f0 - tw) + f_new * tw
            if USE_QUINTIC_TEMPORAL_VAL
                wq = _quintic_lagrange_time_weights(tw)
                rho_b = _quintic_soft_clamp(
                    wq[1] * parent_rho_oldest2[plx, ply, plz, pb_idx] +
                    wq[2] * parent_rho_oldest[plx, ply, plz, pb_idx] +
                    wq[3] * parent_rho_older[plx, ply, plz, pb_idx] +
                    wq[4] * rho_old + wq[5] * rho_new,
                    rho_old, rho_new,
                )
                ux_b = _quintic_soft_clamp(
                    wq[1] * parent_vel_oldest2[plx, ply, plz, pb_idx, 1] +
                    wq[2] * parent_vel_oldest[plx, ply, plz, pb_idx, 1] +
                    wq[3] * parent_vel_older[plx, ply, plz, pb_idx, 1] +
                    wq[4] * ux_old + wq[5] * ux_new,
                    ux_old, ux_new,
                )
                uy_b = _quintic_soft_clamp(
                    wq[1] * parent_vel_oldest2[plx, ply, plz, pb_idx, 2] +
                    wq[2] * parent_vel_oldest[plx, ply, plz, pb_idx, 2] +
                    wq[3] * parent_vel_older[plx, ply, plz, pb_idx, 2] +
                    wq[4] * uy_old + wq[5] * uy_new,
                    uy_old, uy_new,
                )
                uz_b = _quintic_soft_clamp(
                    wq[1] * parent_vel_oldest2[plx, ply, plz, pb_idx, 3] +
                    wq[2] * parent_vel_oldest[plx, ply, plz, pb_idx, 3] +
                    wq[3] * parent_vel_older[plx, ply, plz, pb_idx, 3] +
                    wq[4] * uz_old + wq[5] * uz_new,
                    uz_old, uz_new,
                )
            else
                rho_b = rho_old * (1.0f0 - tw) + rho_new * tw
                ux_b = ux_old * (1.0f0 - tw) + ux_new * tw
                uy_b = uy_old * (1.0f0 - tw) + uy_new * tw
                uz_b = uz_old * (1.0f0 - tw) + uz_new * tw
            end
        end

        feqc = calculate_equilibrium_auto(rho_b, ux_b, uy_b, uz_b, w_k, cx, cy, cz, Int32(0))
        return (f_b - feqc, true)
    end
end

# ── FAST-PATH-ONLY Π SAMPLER (2026-06-12, phase 2b register diet) ────────────
# The full sampler below carries a partial-stencil fallback (8×7 corner tuples
# + 3 extrapolation passes) that dominates the REGISTER footprint of every
# kernel that inlines/calls it — yet for the staged-halo path it is provably
# DEAD: the mark kernel admits only bspline-READY cells (±3-parent support for
# all 19 src positions), which strictly contains this sampler's 2×2×2 corner
# stencil ⇒ the all-valid branch always executes. This variant compiles ONLY
# that branch, with the corner reads, weights, accumulation expressions and
# temporal blend copied VERBATIM (bitwise-identical on every executed path).
# If a corner is somehow invalid it returns ok=false (caller's NaN failure
# path) instead of extrapolating — unreachable for marked cells.
@noinline function sample_parent_pi_neq_fast(
    parent_pi_neq,
    parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    fine_gx::Int32, fine_gy::Int32, fine_gz::Int32,
    block_size::Int32
)
    px_cont = _fine_to_parent_interp_coord(fine_gx)
    py_cont = _fine_to_parent_interp_coord(fine_gy)
    pz_cont = _fine_to_parent_interp_coord(fine_gz)

    parent_nx = parent_dim_x * block_size
    parent_ny = parent_dim_y * block_size
    parent_nz = parent_dim_z * block_size
    px0, px1, wx = _parent_interp_bracket(px_cont, parent_nx)
    py0, py1, wy = _PERIODIC_Y_AMR ? _pbracket_y(py_cont) : _parent_interp_bracket(py_cont, parent_ny)
    pz0, pz1, wz = _parent_interp_bracket(pz_cont, parent_nz)

    @inline function get_pi(pgx::Int32, pgy::Int32, pgz::Int32)
        pbx = (pgx - Int32(1)) ÷ block_size + Int32(1)
        pgy = _pwrap_y(pgy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
        pby = (pgy - Int32(1)) ÷ block_size + Int32(1)
        pbz = (pgz - Int32(1)) ÷ block_size + Int32(1)
        if pbx >= Int32(1) && pbx <= parent_dim_x &&
           pby >= Int32(1) && pby <= parent_dim_y &&
           pbz >= Int32(1) && pbz <= parent_dim_z
            pb_idx = parent_ptr[pbx, pby, pbz]
            if pb_idx > Int32(0)
                plx = (pgx - Int32(1)) % block_size + Int32(1)
                ply = (pgy - Int32(1)) % block_size + Int32(1)
                plz = (pgz - Int32(1)) % block_size + Int32(1)
                return (
                    parent_pi_neq[plx, ply, plz, pb_idx, 1],
                    parent_pi_neq[plx, ply, plz, pb_idx, 2],
                    parent_pi_neq[plx, ply, plz, pb_idx, 3],
                    parent_pi_neq[plx, ply, plz, pb_idx, 4],
                    parent_pi_neq[plx, ply, plz, pb_idx, 5],
                    parent_pi_neq[plx, ply, plz, pb_idx, 6],
                    true
                )
            end
        end
        return (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, false)
    end

    if _D3Q19_PI_NEQ_CUBIC
        # ── C2 stress carrier: 4×4×4 cubic-Lagrange over the 6 Pi_neq components ──
        # Raises the production stress transfer from C0 trilinear to C1+, matching the
        # macro carrier order, so the reconstructed fine distribution is C1+ at the
        # interface (the real full-distribution continuity lever — the production staged
        # kernel reconstructs f from rho,u,Pi_neq, so Pi_neq's order IS the gap). Needs
        # the FULL 4×4×4 stencil valid; on ANY missing tap (box edge) it falls through
        # to trilinear below. Cubic weights identical to the macro/f_neq cubic sweep.
        sxx = 0.0f0; syy = 0.0f0; szz = 0.0f0; sxy = 0.0f0; sxz = 0.0f0; syz = 0.0f0
        all_ok = true
        for ix in Int32(-1):Int32(2)
            pcx = px0 + ix
            wxv = ix == Int32(-1) ? (-wx*(wx-1.0f0)*(wx-2.0f0)/6.0f0) :
                  ix == Int32(0)  ? ((wx+1.0f0)*(wx-1.0f0)*(wx-2.0f0)/2.0f0) :
                  ix == Int32(1)  ? (-wx*(wx+1.0f0)*(wx-2.0f0)/2.0f0) :
                                    (wx*(wx+1.0f0)*(wx-1.0f0)/6.0f0)
            for iy in Int32(-1):Int32(2)
                pcy = py0 + iy
                wyv = iy == Int32(-1) ? (-wy*(wy-1.0f0)*(wy-2.0f0)/6.0f0) :
                      iy == Int32(0)  ? ((wy+1.0f0)*(wy-1.0f0)*(wy-2.0f0)/2.0f0) :
                      iy == Int32(1)  ? (-wy*(wy+1.0f0)*(wy-2.0f0)/2.0f0) :
                                        (wy*(wy+1.0f0)*(wy-1.0f0)/6.0f0)
                for iz in Int32(-1):Int32(2)
                    pcz = pz0 + iz
                    wzv = iz == Int32(-1) ? (-wz*(wz-1.0f0)*(wz-2.0f0)/6.0f0) :
                          iz == Int32(0)  ? ((wz+1.0f0)*(wz-1.0f0)*(wz-2.0f0)/2.0f0) :
                          iz == Int32(1)  ? (-wz*(wz+1.0f0)*(wz-2.0f0)/2.0f0) :
                                            (wz*(wz+1.0f0)*(wz-1.0f0)/6.0f0)
                    dpic = get_pi(pcx, pcy, pcz)
                    if !dpic[7]
                        all_ok = false
                    else
                        w3c = wxv * wyv * wzv
                        sxx += w3c * dpic[1]; syy += w3c * dpic[2]; szz += w3c * dpic[3]
                        sxy += w3c * dpic[4]; sxz += w3c * dpic[5]; syz += w3c * dpic[6]
                    end
                end
            end
        end
        if all_ok
            return (sxx, syy, szz, sxy, sxz, syz, true)
        end
        # else: fall through to trilinear (box-edge graceful degradation)
    end

    w000 = (1.0f0 - wx) * (1.0f0 - wy) * (1.0f0 - wz)
    w100 = wx * (1.0f0 - wy) * (1.0f0 - wz)
    w010 = (1.0f0 - wx) * wy * (1.0f0 - wz)
    w110 = wx * wy * (1.0f0 - wz)
    w001 = (1.0f0 - wx) * (1.0f0 - wy) * wz
    w101 = wx * (1.0f0 - wy) * wz
    w011 = (1.0f0 - wx) * wy * wz
    w111 = wx * wy * wz

    d000 = get_pi(px0, py0, pz0)
    d100 = get_pi(px1, py0, pz0)
    d010 = get_pi(px0, py1, pz0)
    d110 = get_pi(px1, py1, pz0)
    d001 = get_pi(px0, py0, pz1)
    d101 = get_pi(px1, py0, pz1)
    d011 = get_pi(px0, py1, pz1)
    d111 = get_pi(px1, py1, pz1)

    if d000[7] && d100[7] && d010[7] && d110[7] && d001[7] && d101[7] && d011[7] && d111[7]
        pxx = w000*d000[1] + w100*d100[1] + w010*d010[1] + w110*d110[1] +
              w001*d001[1] + w101*d101[1] + w011*d011[1] + w111*d111[1]
        pyy = w000*d000[2] + w100*d100[2] + w010*d010[2] + w110*d110[2] +
              w001*d001[2] + w101*d101[2] + w011*d011[2] + w111*d111[2]
        pzz = w000*d000[3] + w100*d100[3] + w010*d010[3] + w110*d110[3] +
              w001*d001[3] + w101*d101[3] + w011*d011[3] + w111*d111[3]
        pxy = w000*d000[4] + w100*d100[4] + w010*d010[4] + w110*d110[4] +
              w001*d001[4] + w101*d101[4] + w011*d011[4] + w111*d111[4]
        pxz = w000*d000[5] + w100*d100[5] + w010*d010[5] + w110*d110[5] +
              w001*d001[5] + w101*d101[5] + w011*d011[5] + w111*d111[5]
        pyz = w000*d000[6] + w100*d100[6] + w010*d010[6] + w110*d110[6] +
              w001*d001[6] + w101*d101[6] + w011*d011[6] + w111*d111[6]
        return (pxx, pyy, pzz, pxy, pxz, pyz, true)
    end
    return (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, false)
end

@noinline function sample_parent_pi_neq_fast_temporal(
    parent_pi_neq_new,
    parent_pi_neq_old,
    parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    fine_gx::Int32, fine_gy::Int32, fine_gz::Int32,
    block_size::Int32,
    temporal_weight::Float32,
    use_temporal_interp::Int32,
    use_parent_pi_old::Int32,
)
    pxx_n, pyy_n, pzz_n, pxy_n, pxz_n, pyz_n, ok_n =
        sample_parent_pi_neq_fast(
            parent_pi_neq_new, parent_ptr,
            parent_dim_x, parent_dim_y, parent_dim_z,
            fine_gx, fine_gy, fine_gz, block_size)
    if !(ok_n)
        return (pxx_n, pyy_n, pzz_n, pxy_n, pxz_n, pyz_n, false)
    end
    if use_temporal_interp == Int32(1) &&
       use_parent_pi_old == Int32(1) &&
       temporal_weight < 0.99f0
        pxx_o, pyy_o, pzz_o, pxy_o, pxz_o, pyz_o, ok_o =
            sample_parent_pi_neq_fast(
                parent_pi_neq_old, parent_ptr,
                parent_dim_x, parent_dim_y, parent_dim_z,
                fine_gx, fine_gy, fine_gz, block_size)
        if ok_o
            tw = clamp(temporal_weight, 0.0f0, 1.0f0)
            om = 1.0f0 - tw
            return (
                om * pxx_o + tw * pxx_n,
                om * pyy_o + tw * pyy_n,
                om * pzz_o + tw * pzz_n,
                om * pxy_o + tw * pxy_n,
                om * pxz_o + tw * pxz_n,
                om * pyz_o + tw * pyz_n,
                true,
            )
        end
    end
    return (pxx_n, pyy_n, pzz_n, pxy_n, pxz_n, pyz_n, true)
end

@noinline function sample_parent_pi_neq_for_fine_cell_temporal(
    parent_pi_neq_new,
    parent_pi_neq_old,
    parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    fine_gx::Int32, fine_gy::Int32, fine_gz::Int32,
    block_size::Int32,
    temporal_weight::Float32,
    use_temporal_interp::Int32,
    use_parent_pi_old::Int32,
)
    pxx_n, pyy_n, pzz_n, pxy_n, pxz_n, pyz_n, ok_n =
        sample_parent_pi_neq_for_fine_cell(
            parent_pi_neq_new,
            parent_ptr,
            parent_dim_x, parent_dim_y, parent_dim_z,
            fine_gx, fine_gy, fine_gz,
            block_size,
        )
    if !(ok_n)
        return (pxx_n, pyy_n, pzz_n, pxy_n, pxz_n, pyz_n, false)
    end

    if use_temporal_interp == Int32(1) &&
       use_parent_pi_old == Int32(1) &&
       temporal_weight < 0.99f0
        pxx_o, pyy_o, pzz_o, pxy_o, pxz_o, pyz_o, ok_o =
            sample_parent_pi_neq_for_fine_cell(
                parent_pi_neq_old,
                parent_ptr,
                parent_dim_x, parent_dim_y, parent_dim_z,
                fine_gx, fine_gy, fine_gz,
                block_size,
            )
        if ok_o
            tw = clamp(temporal_weight, 0.0f0, 1.0f0)
            om = 1.0f0 - tw
            return (
                om * pxx_o + tw * pxx_n,
                om * pyy_o + tw * pyy_n,
                om * pzz_o + tw * pzz_n,
                om * pxy_o + tw * pxy_n,
                om * pxz_o + tw * pxz_n,
                om * pyz_o + tw * pyz_n,
                true,
            )
        end
    end

    return (pxx_n, pyy_n, pzz_n, pxy_n, pxz_n, pyz_n, true)
end

# BOX-EDGE SEAM A/B (2026-06-09e): the tw<0.99 branch below downgrades the halo
# macro interpolation from the C4 B-spline to C0 trilinear(+temporal) — "time-
# consistent" — while the monolithic kernel's generic path keeps the (new-time)
# B-spline at ALL substep weights (physics_interpolation.jl ~L854). The K1-input
# split-vs-monolith probe (CASE_LOGS/seam_k1input_Z156.log) showed this method
# gap IS the dominant split-vs-monolith divergence: L2 max|Δρ| ~1.5e-4 during
# the ramp transient (vs ULP-class on quiescent L3/L4 — the earlier "2 ULP
# round-off" diagnosis sampled pre-transient steps). The trilinear bracket
# switch is a C0 interpolation-error kink along the refinement-box edge; the
# monolith — which never takes this branch — clears the seam. Set
# LUDWIG_D3Q19_BSPLINE_MACRO_TIME_CONSISTENT=0 to use the B-spline at all tw
# (monolith-aligned, new-time macros). Default "1" = legacy, byte-identical.
const _D3Q19_BSPLINE_MACRO_TIME_CONSISTENT =
    get(ENV, "LUDWIG_D3Q19_BSPLINE_MACRO_TIME_CONSISTENT", "1") == "1"

# 2026-06-16 — D3Q19 B-spline macro bounds-preservation clamp. The quintic
# B-spline macro prolongation has large negative outer lobes that OVERSHOOT at
# AMR box-edge gradients, injecting the (-1)^(x+z) Nyquist checkerboard that is
# the dominant D3Q19 AMR instability seed unmasked at physical viscosity
# (Z208/Z231/Z234). When ON, each interpolated fine-cell macro (rho, u) is
# clamped to the [min,max] envelope of its 8 bracketing parent corners (discrete
# maximum principle) before the equilibrium reconstruction — no-op in smooth
# regions (B-spline accuracy preserved), bounded at gradients (overshoot killed),
# NO global viscosity. Set LUDWIG_D3Q19_BSPLINE_MONOTONE=1 to enable. Default
# "0" = legacy unbounded B-spline, byte-identical.
const _D3Q19_BSPLINE_MONOTONE =
    get(ENV, "LUDWIG_D3Q19_BSPLINE_MONOTONE", "0") == "1"

# 2026-06-20 — COLLAPSED C4 B-SPLINE macro prolongation (100-MLUPS lever 1). The
# C4 B-spline above is normally a two-stage op: a once-per-coarse-step prefilter
# (compute_spline_coeffs!, ~7.9 s/batch + coefficient arrays) feeding a compact
# 6-tap eval. When ON, the macro funnel uses instead the FUSED ±21-tap stencil
# (_BSPLINE_K_LOW, kernel_quintic_prolongation.jl) applied DIRECTLY to the raw
# parent macros — the SAME C4 result (to FP32 summation order) with NO prefilter
# and NO coefficient arrays, and true-C4 on EVERY substep (no tw<0.99 trilinear
# downgrade). This is the per-cell PROOF path (43³ raw reads/cell — expensive per
# substep; the once-per-coarse-step hoist amortizes it). Requires the funnel to
# carry the raw parent macros (parent_rho_new/_old, parent_vel_new/_old), which it
# already receives. Default "0" = legacy prefilter+6-tap path, byte-identical.
const _D3Q19_BSPLINE_COLLAPSED =
    get(ENV, "LUDWIG_D3Q19_BSPLINE_COLLAPSED", "0") == "1"

# ── Seam-lever provenance (2026-06-16) ──────────────────────────────────────
# The D3Q19 AMR-seam features are COMPILE-TIME env-gated consts that bake into
# the GPU kernels at module load — there is no per-case YAML record of which
# were active. The audit (wss7lq8pn) flagged that this makes a posthoc A/B
# ambiguous ("was the flag actually on?"). Print the resolved state ONCE so every
# run self-documents. Print-only ⇒ byte-identical to numerics.
let _seam_levers = (
        ("BSPLINE_MACRO_TIME_CONSISTENT", _D3Q19_BSPLINE_MACRO_TIME_CONSISTENT),
        ("AMR_FNEQ_POSTCOLL",             _AMR_FNEQ_POSTCOLL),
        ("AMR_FNEQ_CUBIC",                _D3Q19_AMR_FNEQ_CUBIC),
        ("BSPLINE_MONOTONE",              _D3Q19_BSPLINE_MONOTONE),
        ("BSPLINE_COLLAPSED",             _D3Q19_BSPLINE_COLLAPSED),
    )
    println("[SeamLevers] D3Q19 AMR-interface gates (compile-time): " *
            join(["$n=$(v ? "ON" : "off")" for (n, v) in _seam_levers], "  "))
end

# ── COLLAPSED C4 B-SPLINE macro sampler (2026-06-20, lever 1) ────────────────
# Evaluate (rho, ux, uy, uz) at the fine cell via the fused ±21-tap K stencil on
# the RAW temporal-blended parent macros — no spline-coeff arrays, true C4 at all
# tw. Reads exactly the parent cells `get_macro` would (same blend, same NaN-on-
# inactive semantics). Returns (rho,ux,uy,uz, ok); ok=false if the ±21 support is
# incomplete OR any tap reads an inactive parent → caller falls back to the
# trilinear/coeff-B-spline path. `field` selector: 0=rho, 1=ux, 2=uy, 3=uz.
@inline function _collapsed_read_raw_field(
    parent_rho_new, parent_vel_new, parent_rho_old, parent_vel_old,
    parent_ptr, parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    block_size::Int32, temporal_weight::Float32, use_temporal_interp::Int32,
    field::Int32, pgx::Int32, pgy::Int32, pgz::Int32,
)
    @inbounds begin
        pbx = div(pgx - Int32(1), block_size) + Int32(1)
        pgy = _pwrap_y(pgy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
        pby = div(pgy - Int32(1), block_size) + Int32(1)
        pbz = div(pgz - Int32(1), block_size) + Int32(1)
        (pbx >= Int32(1) && pbx <= parent_dim_x &&
         pby >= Int32(1) && pby <= parent_dim_y &&
         pbz >= Int32(1) && pbz <= parent_dim_z) || return NaN32
        pb = parent_ptr[pbx, pby, pbz]
        pb > Int32(0) || return NaN32
        plx = (pgx - Int32(1)) % block_size + Int32(1)
        ply = (pgy - Int32(1)) % block_size + Int32(1)
        plz = (pgz - Int32(1)) % block_size + Int32(1)
        vn = field == Int32(0) ? parent_rho_new[plx, ply, plz, pb] :
                                 parent_vel_new[plx, ply, plz, pb, field]
        if use_temporal_interp == Int32(1) && temporal_weight < 0.99f0 &&
           parent_rho_old !== nothing && parent_vel_old !== nothing
            tw = temporal_weight
            vo = field == Int32(0) ? parent_rho_old[plx, ply, plz, pb] :
                                     parent_vel_old[plx, ply, plz, pb, field]
            return vo * (1.0f0 - tw) + vn * tw
        end
        return vn
    end
end

@noinline function sample_parent_macro_collapsed(
    parent_rho_new, parent_vel_new,
    parent_rho_old, parent_vel_old,
    parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    fine_gx::Int32, fine_gy::Int32, fine_gz::Int32,
    block_size::Int32,
    temporal_weight::Float32,
    use_temporal_interp::Int32,
)
    p0x, lowx = _collapsed_axis_setup(fine_gx)
    p0y, lowy = _collapsed_axis_setup(fine_gy)
    p0z, lowz = _collapsed_axis_setup(fine_gz)
    # Require the full ±21 support active so the result is bit-faithful to the
    # offline-verified stencil (no partial-stencil renormalization here — the
    # caller falls back to the trilinear path at ragged AMR-box edges).
    _collapsed_support_ready(parent_ptr, p0x, p0y, p0z,
                             parent_dim_x, parent_dim_y, parent_dim_z, block_size) ||
        return (1.0f0, 0.0f0, 0.0f0, 0.0f0, false)
    @inline read_rho(pgx, pgy, pgz) = _collapsed_read_raw_field(
        parent_rho_new, parent_vel_new, parent_rho_old, parent_vel_old,
        parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z,
        block_size, temporal_weight, use_temporal_interp, Int32(0), pgx, pgy, pgz)
    @inline read_ux(pgx, pgy, pgz) = _collapsed_read_raw_field(
        parent_rho_new, parent_vel_new, parent_rho_old, parent_vel_old,
        parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z,
        block_size, temporal_weight, use_temporal_interp, Int32(1), pgx, pgy, pgz)
    @inline read_uy(pgx, pgy, pgz) = _collapsed_read_raw_field(
        parent_rho_new, parent_vel_new, parent_rho_old, parent_vel_old,
        parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z,
        block_size, temporal_weight, use_temporal_interp, Int32(2), pgx, pgy, pgz)
    @inline read_uz(pgx, pgy, pgz) = _collapsed_read_raw_field(
        parent_rho_new, parent_vel_new, parent_rho_old, parent_vel_old,
        parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z,
        block_size, temporal_weight, use_temporal_interp, Int32(3), pgx, pgy, pgz)
    r, okr  = _collapsed_eval_field(read_rho, p0x, p0y, p0z, lowx, lowy, lowz)
    okr || return (1.0f0, 0.0f0, 0.0f0, 0.0f0, false)
    vx, okx = _collapsed_eval_field(read_ux, p0x, p0y, p0z, lowx, lowy, lowz)
    vy, oky = _collapsed_eval_field(read_uy, p0x, p0y, p0z, lowx, lowy, lowz)
    vz, okz = _collapsed_eval_field(read_uz, p0x, p0y, p0z, lowx, lowy, lowz)
    (okx && oky && okz) || return (1.0f0, 0.0f0, 0.0f0, 0.0f0, false)
    return (r, vx, vy, vz, true)
end

@noinline function sample_parent_macro_bspline_time_consistent(
    parent_rho_new, parent_vel_new,
    parent_rho_old, parent_vel_old,
    rho_spline_coeff, vel_spline_coeff,
    parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    fine_gx::Int32, fine_gy::Int32, fine_gz::Int32,
    block_size::Int32,
    temporal_weight::Float32,
    use_temporal_interp::Int32,
)
    # COLLAPSED C4 (lever 1): fused ±21-tap K stencil on RAW macros — true C4 at
    # all tw, no spline coeffs. Takes precedence over both the trilinear downgrade
    # and the coeff-B-spline. Falls through to the legacy paths when the ±21
    # support is incomplete (ragged AMR-box edge). Default off ⇒ byte-identical.
    if _D3Q19_BSPLINE_COLLAPSED
        rho_c, ux_c, uy_c, uz_c, okc = sample_parent_macro_collapsed(
            parent_rho_new, parent_vel_new,
            parent_rho_old, parent_vel_old,
            parent_ptr,
            parent_dim_x, parent_dim_y, parent_dim_z,
            fine_gx, fine_gy, fine_gz,
            block_size,
            temporal_weight,
            use_temporal_interp,
        )
        if okc
            if _D3Q19_BSPLINE_MONOTONE
                rho_c, ux_c, uy_c, uz_c = _clamp_macro_to_parent_corners(
                    rho_c, ux_c, uy_c, uz_c,
                    parent_rho_new, parent_vel_new, parent_rho_old, parent_vel_old,
                    parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z,
                    fine_gx, fine_gy, fine_gz, block_size,
                    temporal_weight, use_temporal_interp,
                )
            end
            return (rho_c, ux_c, uy_c, uz_c, true)
        end
        # support incomplete → fall through to legacy paths below
    end
    # `parent_*_old !== nothing` guard (2026-06-09): _old may be the `nothing`
    # singleton when the D3Q19 halo kernel runs with temporal interp OFF.
    if _D3Q19_BSPLINE_MACRO_TIME_CONSISTENT &&
       use_temporal_interp == Int32(1) && temporal_weight < 0.99f0 &&
       parent_rho_old !== nothing && parent_vel_old !== nothing
        return sample_parent_macro_for_fine_cell(
            parent_rho_new, parent_vel_new,
            parent_rho_old, parent_vel_old,
            parent_ptr,
            parent_dim_x, parent_dim_y, parent_dim_z,
            fine_gx, fine_gy, fine_gz,
            block_size,
            temporal_weight,
            use_temporal_interp,
        )
    end
    rho_b, ux_b, uy_b, uz_b, okb = interpolate_macro_bspline(
        rho_spline_coeff, vel_spline_coeff, parent_ptr,
        parent_dim_x, parent_dim_y, parent_dim_z,
        fine_gx, fine_gy, fine_gz, block_size,
    )
    # ── B-spline macro bounds-preservation clamp (CHOKEPOINT, 2026-06-17) ──
    # Applied HERE (the single function every D3Q19 macro-prolongation caller
    # funnels through — both the per-direction halo wrapper AND the staged
    # pi_neq path that calls this DIRECTLY) so the clamp can never be bypassed.
    # An earlier placement inside interpolate_with_rescaling_d3q19_bspline_halo
    # ONLY missed the staged pi_neq path (kernel_d3q19_rr_amr_halo_stage.jl:563/695)
    # → Z249 was byte-identical to baseline (clamp never ran on the seam cells).
    # Clamps each interpolated macro to the [min,max] envelope of the 8 bracketing
    # parent corners (discrete maximum principle): no-op in smooth regions
    # (freestream byte-identical), bounds ONLY the negative-lobe overshoot at
    # box-edge gradients. Default off ⇒ byte-identical.
    if _D3Q19_BSPLINE_MONOTONE && okb
        rho_b, ux_b, uy_b, uz_b = _clamp_macro_to_parent_corners(
            rho_b, ux_b, uy_b, uz_b,
            parent_rho_new, parent_vel_new, parent_rho_old, parent_vel_old,
            parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z,
            fine_gx, fine_gy, fine_gz, block_size,
            temporal_weight, use_temporal_interp,
        )
    end
    return (rho_b, ux_b, uy_b, uz_b, okb)
end

# 8-corner discrete-maximum-principle clamp for the B-spline macro prolongation.
# Bounds rho/u to the [min,max] of the 8 parent cells bracketing the fine cell.
@inline function _clamp_macro_to_parent_corners(
    rho_int::Float32, ux_int::Float32, uy_int::Float32, uz_int::Float32,
    parent_rho_new, parent_vel_new, parent_rho_old, parent_vel_old,
    parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    fine_gx::Int32, fine_gy::Int32, fine_gz::Int32,
    block_size::Int32,
    temporal_weight::Float32, use_temporal_interp::Int32,
)
    px_cont = _fine_to_parent_interp_coord(fine_gx)
    py_cont = _fine_to_parent_interp_coord(fine_gy)
    pz_cont = _fine_to_parent_interp_coord(fine_gz)
    parent_nx = parent_dim_x * block_size
    parent_ny = parent_dim_y * block_size
    parent_nz = parent_dim_z * block_size
    px0, px1, _ = _parent_interp_bracket(px_cont, parent_nx)
    py0, py1, _ = _PERIODIC_Y_AMR ? _pbracket_y(py_cont) : _parent_interp_bracket(py_cont, parent_ny)
    pz0, pz1, _ = _parent_interp_bracket(pz_cont, parent_nz)

    @inline function _corner(pgx::Int32, pgy::Int32, pgz::Int32)
        pbx = div(pgx - Int32(1), block_size) + Int32(1)
        pgy = _pwrap_y(pgy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
        pby = div(pgy - Int32(1), block_size) + Int32(1)
        pbz = div(pgz - Int32(1), block_size) + Int32(1)
        if pbx < Int32(1) || pbx > parent_dim_x ||
           pby < Int32(1) || pby > parent_dim_y ||
           pbz < Int32(1) || pbz > parent_dim_z
            return (0.0f0, 0.0f0, 0.0f0, 0.0f0, false)
        end
        pb_idx = parent_ptr[pbx, pby, pbz]
        pb_idx <= Int32(0) && return (0.0f0, 0.0f0, 0.0f0, 0.0f0, false)
        plx = (pgx - Int32(1)) % block_size + Int32(1)
        ply = (pgy - Int32(1)) % block_size + Int32(1)
        plz = (pgz - Int32(1)) % block_size + Int32(1)
        @inbounds begin
            rv = parent_rho_new[plx, ply, plz, pb_idx]
            xv = parent_vel_new[plx, ply, plz, pb_idx, 1]
            yv = parent_vel_new[plx, ply, plz, pb_idx, 2]
            zv = parent_vel_new[plx, ply, plz, pb_idx, 3]
            if use_temporal_interp == Int32(1) && temporal_weight < 0.99f0 &&
               parent_rho_old !== nothing && parent_vel_old !== nothing
                tw = temporal_weight; om = 1.0f0 - tw
                rv = parent_rho_old[plx, ply, plz, pb_idx] * om + rv * tw
                xv = parent_vel_old[plx, ply, plz, pb_idx, 1] * om + xv * tw
                yv = parent_vel_old[plx, ply, plz, pb_idx, 2] * om + yv * tw
                zv = parent_vel_old[plx, ply, plz, pb_idx, 3] * om + zv * tw
            end
        end
        return (rv, xv, yv, zv, true)
    end

    rmn = rho_int; rmx = rho_int; xmn = ux_int; xmx = ux_int
    ymn = uy_int; ymx = uy_int; zmn = uz_int; zmx = uz_int
    seeded = false
    for cz in (pz0, pz1), cy in (py0, py1), cx2 in (px0, px1)
        rv, xv, yv, zv, okc = _corner(cx2, cy, cz)
        if okc
            if !seeded
                rmn = rv; rmx = rv; xmn = xv; xmx = xv
                ymn = yv; ymx = yv; zmn = zv; zmx = zv
                seeded = true
            else
                rmn = min(rmn, rv); rmx = max(rmx, rv)
                xmn = min(xmn, xv); xmx = max(xmx, xv)
                ymn = min(ymn, yv); ymx = max(ymx, yv)
                zmn = min(zmn, zv); zmx = max(zmx, zv)
            end
        end
    end
    if seeded
        return (clamp(rho_int, rmn, rmx), clamp(ux_int, xmn, xmx),
                clamp(uy_int, ymn, ymx), clamp(uz_int, zmn, zmx))
    end
    return (rho_int, ux_int, uy_int, uz_int)
end

@noinline function interpolate_with_rescaling_d3q19_bspline_halo(
    parent_f_new, parent_rho_new, parent_vel_new,
    parent_f_old, parent_rho_old, parent_vel_old,
    parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    fine_gx::Int32, fine_gy::Int32, fine_gz::Int32,
    k::Int32, block_size::Int32,
    w_k::Float32, cx::Float32, cy::Float32, cz::Float32,
    tau_coarse::Float32, tau_fine::Float32,
    temporal_weight::Float32,
    use_temporal_interp::Int32,
    rho_spline_coeff, vel_spline_coeff,
    parent_rho_older, parent_vel_older,
    parent_rho_oldest, parent_vel_oldest,
    parent_rho_oldest2, parent_vel_oldest2,
    ::Val{USE_QUINTIC_TEMPORAL_VAL},
) where {USE_QUINTIC_TEMPORAL_VAL}
    px_cont = _fine_to_parent_interp_coord(fine_gx)
    py_cont = _fine_to_parent_interp_coord(fine_gy)
    pz_cont = _fine_to_parent_interp_coord(fine_gz)
    parent_nx = parent_dim_x * block_size
    parent_ny = parent_dim_y * block_size
    parent_nz = parent_dim_z * block_size
    px0, px1, wx = _parent_interp_bracket(px_cont, parent_nx)
    py0, py1, wy = _PERIODIC_Y_AMR ? _pbracket_y(py_cont) : _parent_interp_bracket(py_cont, parent_ny)
    pz0, pz1, wz = _parent_interp_bracket(pz_cont, parent_nz)

    n000, ok000 = _d3q19_parent_corner_neq_quintic(parent_f_new, parent_rho_new, parent_vel_new, parent_f_old, parent_rho_old, parent_vel_old, parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z, px0, py0, pz0, k, block_size, w_k, cx, cy, cz, temporal_weight, use_temporal_interp, parent_rho_older, parent_vel_older, parent_rho_oldest, parent_vel_oldest, parent_rho_oldest2, parent_vel_oldest2, Val(USE_QUINTIC_TEMPORAL_VAL))
    n100, ok100 = _d3q19_parent_corner_neq_quintic(parent_f_new, parent_rho_new, parent_vel_new, parent_f_old, parent_rho_old, parent_vel_old, parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z, px1, py0, pz0, k, block_size, w_k, cx, cy, cz, temporal_weight, use_temporal_interp, parent_rho_older, parent_vel_older, parent_rho_oldest, parent_vel_oldest, parent_rho_oldest2, parent_vel_oldest2, Val(USE_QUINTIC_TEMPORAL_VAL))
    n010, ok010 = _d3q19_parent_corner_neq_quintic(parent_f_new, parent_rho_new, parent_vel_new, parent_f_old, parent_rho_old, parent_vel_old, parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z, px0, py1, pz0, k, block_size, w_k, cx, cy, cz, temporal_weight, use_temporal_interp, parent_rho_older, parent_vel_older, parent_rho_oldest, parent_vel_oldest, parent_rho_oldest2, parent_vel_oldest2, Val(USE_QUINTIC_TEMPORAL_VAL))
    n110, ok110 = _d3q19_parent_corner_neq_quintic(parent_f_new, parent_rho_new, parent_vel_new, parent_f_old, parent_rho_old, parent_vel_old, parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z, px1, py1, pz0, k, block_size, w_k, cx, cy, cz, temporal_weight, use_temporal_interp, parent_rho_older, parent_vel_older, parent_rho_oldest, parent_vel_oldest, parent_rho_oldest2, parent_vel_oldest2, Val(USE_QUINTIC_TEMPORAL_VAL))
    n001, ok001 = _d3q19_parent_corner_neq_quintic(parent_f_new, parent_rho_new, parent_vel_new, parent_f_old, parent_rho_old, parent_vel_old, parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z, px0, py0, pz1, k, block_size, w_k, cx, cy, cz, temporal_weight, use_temporal_interp, parent_rho_older, parent_vel_older, parent_rho_oldest, parent_vel_oldest, parent_rho_oldest2, parent_vel_oldest2, Val(USE_QUINTIC_TEMPORAL_VAL))
    n101, ok101 = _d3q19_parent_corner_neq_quintic(parent_f_new, parent_rho_new, parent_vel_new, parent_f_old, parent_rho_old, parent_vel_old, parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z, px1, py0, pz1, k, block_size, w_k, cx, cy, cz, temporal_weight, use_temporal_interp, parent_rho_older, parent_vel_older, parent_rho_oldest, parent_vel_oldest, parent_rho_oldest2, parent_vel_oldest2, Val(USE_QUINTIC_TEMPORAL_VAL))
    n011, ok011 = _d3q19_parent_corner_neq_quintic(parent_f_new, parent_rho_new, parent_vel_new, parent_f_old, parent_rho_old, parent_vel_old, parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z, px0, py1, pz1, k, block_size, w_k, cx, cy, cz, temporal_weight, use_temporal_interp, parent_rho_older, parent_vel_older, parent_rho_oldest, parent_vel_oldest, parent_rho_oldest2, parent_vel_oldest2, Val(USE_QUINTIC_TEMPORAL_VAL))
    n111, ok111 = _d3q19_parent_corner_neq_quintic(parent_f_new, parent_rho_new, parent_vel_new, parent_f_old, parent_rho_old, parent_vel_old, parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z, px1, py1, pz1, k, block_size, w_k, cx, cy, cz, temporal_weight, use_temporal_interp, parent_rho_older, parent_vel_older, parent_rho_oldest, parent_vel_oldest, parent_rho_oldest2, parent_vel_oldest2, Val(USE_QUINTIC_TEMPORAL_VAL))

    any_corner_ok =
        ok000 || ok100 || ok010 || ok110 ||
        ok001 || ok101 || ok011 || ok111
    any_corner_ok || return (0.0f0, false)
    neq_anchor =
        ok000 ? n000 :
        ok100 ? n100 :
        ok010 ? n010 :
        ok110 ? n110 :
        ok001 ? n001 :
        ok101 ? n101 :
        ok011 ? n011 : n111

    v000 = ok000 ? n000 : neq_anchor
    v100 = ok100 ? n100 : neq_anchor
    v010 = ok010 ? n010 : neq_anchor
    v110 = ok110 ? n110 : neq_anchor
    v001 = ok001 ? n001 : neq_anchor
    v101 = ok101 ? n101 : neq_anchor
    v011 = ok011 ? n011 : neq_anchor
    v111 = ok111 ? n111 : neq_anchor

    if _D3Q19_AMR_FNEQ_CUBIC
        # ── TIER 1: cubic-Lagrange f_neq carrier (C1, matches D3Q27/D3Q39) ──
        # 4×4×4 cubic-Lagrange sweep over the SAME per-corner f_neq accessor used
        # for the trilinear corners above (_d3q19_parent_corner_neq_quintic), reusing
        # the D3Q27 weight formulas verbatim. Replaces the O(1/N)-pressure-jump
        # trilinear blend (Lagrava 2012 §3.7). The 8 trilinear corners (v000..v111)
        # become the central 2×2×2 sub-block of the 4×4×4 stencil. Cells that fall
        # off the parent grid return ok=false ⇒ contribute neq_anchor (the same
        # graceful-degradation anchor the trilinear path uses for missing corners),
        # so the cubic sweep never reads garbage at the domain/box boundary.
        f_neq = 0.0f0
        for ix in Int32(-1):Int32(2)
            px = px0 + ix
            wxv = ix == Int32(-1) ? (-wx*(wx-1.0f0)*(wx-2.0f0)/6.0f0) :
                  ix == Int32(0)  ? ((wx+1.0f0)*(wx-1.0f0)*(wx-2.0f0)/2.0f0) :
                  ix == Int32(1)  ? (-wx*(wx+1.0f0)*(wx-2.0f0)/2.0f0) :
                                    (wx*(wx+1.0f0)*(wx-1.0f0)/6.0f0)
            for iy in Int32(-1):Int32(2)
                py = py0 + iy
                wyv = iy == Int32(-1) ? (-wy*(wy-1.0f0)*(wy-2.0f0)/6.0f0) :
                      iy == Int32(0)  ? ((wy+1.0f0)*(wy-1.0f0)*(wy-2.0f0)/2.0f0) :
                      iy == Int32(1)  ? (-wy*(wy+1.0f0)*(wy-2.0f0)/2.0f0) :
                                        (wy*(wy+1.0f0)*(wy-1.0f0)/6.0f0)
                for iz in Int32(-1):Int32(2)
                    pz = pz0 + iz
                    wzv = iz == Int32(-1) ? (-wz*(wz-1.0f0)*(wz-2.0f0)/6.0f0) :
                          iz == Int32(0)  ? ((wz+1.0f0)*(wz-1.0f0)*(wz-2.0f0)/2.0f0) :
                          iz == Int32(1)  ? (-wz*(wz+1.0f0)*(wz-2.0f0)/2.0f0) :
                                            (wz*(wz+1.0f0)*(wz-1.0f0)/6.0f0)
                    np, okp = _d3q19_parent_corner_neq_quintic(parent_f_new, parent_rho_new, parent_vel_new, parent_f_old, parent_rho_old, parent_vel_old, parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z, px, py, pz, k, block_size, w_k, cx, cy, cz, temporal_weight, use_temporal_interp, parent_rho_older, parent_vel_older, parent_rho_oldest, parent_vel_oldest, parent_rho_oldest2, parent_vel_oldest2, Val(USE_QUINTIC_TEMPORAL_VAL))
                    nv = okp ? np : neq_anchor
                    f_neq += wxv * wyv * wzv * nv
                end
            end
        end
    else
        c00 = v000 * (1.0f0 - wx) + v100 * wx
        c01 = v001 * (1.0f0 - wx) + v101 * wx
        c10 = v010 * (1.0f0 - wx) + v110 * wx
        c11 = v011 * (1.0f0 - wx) + v111 * wx
        c0 = c00 * (1.0f0 - wy) + c10 * wy
        c1 = c01 * (1.0f0 - wy) + c11 * wy
        f_neq = c0 * (1.0f0 - wz) + c1 * wz
    end

    rho_int, ux_int, uy_int, uz_int, okb = sample_parent_macro_bspline_time_consistent(
        parent_rho_new, parent_vel_new,
        parent_rho_old, parent_vel_old,
        rho_spline_coeff, vel_spline_coeff,
        parent_ptr,
        parent_dim_x, parent_dim_y, parent_dim_z,
        fine_gx, fine_gy, fine_gz,
        block_size,
        temporal_weight,
        use_temporal_interp,
    )
    okb || return (0.0f0, false)

    # B-spline macro bounds-preservation clamp now lives at the CHOKEPOINT inside
    # sample_parent_macro_bspline_time_consistent (above) so BOTH this per-direction
    # wrapper AND the staged pi_neq path get it — rho_int/u_int are already clamped
    # here when _D3Q19_BSPLINE_MONOTONE is on.

    feq_int = calculate_equilibrium_auto(rho_int, ux_int, uy_int, uz_int, w_k, cx, cy, cz, Int32(0))
    tau_c = tau_coarse - 0.5f0
    tau_f = tau_fine - 0.5f0
    scale = tau_c > 1.0f-6 ? clamp(tau_f / tau_c, 0.01f0, 100.0f0) : 1.0f0
    # AMR-SEAM FIX 1A reciprocal (2026-06-16): the parent f_neq is POST-collision,
    # so the coarse→fine transfer needs the (1−ω_coarse)/(1−ω_fine) factor (the
    # reciprocal of the restriction-side correction in amr_coupling.jl). Closes the
    # round-trip so neither side is left with a residual stress bias. Freestream
    # byte-identical (f_neq≡0). Gated by the same flag (default ON).
    scale = _amr_fneq_postcoll_prolong_scale(scale, tau_coarse, tau_fine)
    return (feq_int + f_neq * scale, true)
end

# ────────────────────────────────────────────────────────────────────
# Entropic mirror-decomposed interpolation (non-VRAM-constrained)
# ────────────────────────────────────────────────────────────────────
# This function interpolates BOTH direction k and its opposite k̄ from
# the parent grid, then decomposes f_neq into:
#   s_k = ½(f_neq_k + f_neq_k̄)  — mirror-symmetric (viscous stress)
#   h_k = ½(f_neq_k − f_neq_k̄)  — mirror-antisymmetric (higher-order)
#
# Only the symmetric part s is rescaled by tau_f/tau_c and kept.
# The antisymmetric part h is discarded because:
#   1. It relaxes at rate 2βα (unknown, cell-dependent) → wrong rescaling
#   2. It changes sign between k and k̄ → causes checkerboard if incorrectly scaled
#   3. The fine level's H-theorem solver reconstructs it in one collision step
#
# This is more accurate than equilibrium-only interpolation (entropic_mode=1)
# because it preserves the viscous stress tensor across refinement boundaries.
# Cost: 2× the parent memory reads per interpolated cell (reads both k and k̄).
# ────────────────────────────────────────────────────────────────────
@inline function interpolate_entropic_mirror(
    parent_f_new, parent_rho_new, parent_vel_new,
    parent_f_old, parent_rho_old, parent_vel_old,
    parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    fine_gx::Int32, fine_gy::Int32, fine_gz::Int32,
    k::Int32, k_opp::Int32, block_size::Int32,
    w_k::Float32, cx::Float32, cy::Float32, cz::Float32,
    w_kb::Float32, cx_b::Float32, cy_b::Float32, cz_b::Float32,
    tau_coarse::Float32, tau_fine::Float32,
    temporal_weight::Float32,
    use_temporal_interp::Int32,
    compressible_corr::Int32
)
    # ── Common parent coordinate mapping (shared for both k and k̄) ──
    px_cont = _fine_to_parent_interp_coord(fine_gx)
    py_cont = _fine_to_parent_interp_coord(fine_gy)
    pz_cont = _fine_to_parent_interp_coord(fine_gz)

    parent_nx = parent_dim_x * block_size
    parent_ny = parent_dim_y * block_size
    parent_nz = parent_dim_z * block_size
    px0, px1, wx = _parent_interp_bracket(px_cont, parent_nx)
    py0, py1, wy = _PERIODIC_Y_AMR ? _pbracket_y(py_cont) : _parent_interp_bracket(py_cont, parent_ny)
    pz0, pz1, wz = _parent_interp_bracket(pz_cont, parent_nz)

    # ── Fetch and blend for a given parent global position ──
    # Returns (f_k, f_kb, rho, ux, uy, uz, valid)
    @inline function get_blended_pair(pgx::Int32, pgy::Int32, pgz::Int32)
        pbx = (pgx - Int32(1)) ÷ block_size + Int32(1)
        pgy = _pwrap_y(pgy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
        pby = (pgy - Int32(1)) ÷ block_size + Int32(1)
        pbz = (pgz - Int32(1)) ÷ block_size + Int32(1)

        if pbx >= Int32(1) && pbx <= parent_dim_x &&
           pby >= Int32(1) && pby <= parent_dim_y &&
           pbz >= Int32(1) && pbz <= parent_dim_z
            pb_idx = parent_ptr[pbx, pby, pbz]
            if pb_idx > Int32(0)
                plx = (pgx - Int32(1)) % block_size + Int32(1)
                ply = (pgy - Int32(1)) % block_size + Int32(1)
                plz = (pgz - Int32(1)) % block_size + Int32(1)

                fk_new  = parent_f_new[plx, ply, plz, pb_idx, k]
                fkb_new = parent_f_new[plx, ply, plz, pb_idx, k_opp]
                rho_new = parent_rho_new[plx, ply, plz, pb_idx]
                ux_new  = parent_vel_new[plx, ply, plz, pb_idx, 1]
                uy_new  = parent_vel_new[plx, ply, plz, pb_idx, 2]
                uz_new  = parent_vel_new[plx, ply, plz, pb_idx, 3]

                if use_temporal_interp == Int32(1) && temporal_weight < 0.99f0
                    fk_old  = parent_f_old[plx, ply, plz, pb_idx, k]
                    fkb_old = parent_f_old[plx, ply, plz, pb_idx, k_opp]
                    rho_old = parent_rho_old[plx, ply, plz, pb_idx]
                    ux_old  = parent_vel_old[plx, ply, plz, pb_idx, 1]
                    uy_old  = parent_vel_old[plx, ply, plz, pb_idx, 2]
                    uz_old  = parent_vel_old[plx, ply, plz, pb_idx, 3]

                    tw = temporal_weight
                    return (fk_old*(1.0f0-tw) + fk_new*tw,
                            fkb_old*(1.0f0-tw) + fkb_new*tw,
                            rho_old*(1.0f0-tw) + rho_new*tw,
                            ux_old*(1.0f0-tw) + ux_new*tw,
                            uy_old*(1.0f0-tw) + uy_new*tw,
                            uz_old*(1.0f0-tw) + uz_new*tw, true)
                end
                return (fk_new, fkb_new, rho_new, ux_new, uy_new, uz_new, true)
            end
        end
        return (w_k, w_kb, 1.0f0, 0.0f0, 0.0f0, 0.0f0, false)
    end

    # ── Fetch macroscopic (rho, u) only from a parent cell ──
    @inline function get_macro_m(pgx::Int32, pgy::Int32, pgz::Int32)
        pbx = (pgx - Int32(1)) ÷ block_size + Int32(1)
        pgy = _pwrap_y(pgy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
        pby = (pgy - Int32(1)) ÷ block_size + Int32(1)
        pbz = (pgz - Int32(1)) ÷ block_size + Int32(1)
        if pbx >= Int32(1) && pbx <= parent_dim_x &&
           pby >= Int32(1) && pby <= parent_dim_y &&
           pbz >= Int32(1) && pbz <= parent_dim_z
            pb_idx = parent_ptr[pbx, pby, pbz]
            if pb_idx > Int32(0)
                plx = (pgx - Int32(1)) % block_size + Int32(1)
                ply = (pgy - Int32(1)) % block_size + Int32(1)
                plz = (pgz - Int32(1)) % block_size + Int32(1)
                rho_v = parent_rho_new[plx, ply, plz, pb_idx]
                ux_v = parent_vel_new[plx, ply, plz, pb_idx, 1]
                uy_v = parent_vel_new[plx, ply, plz, pb_idx, 2]
                uz_v = parent_vel_new[plx, ply, plz, pb_idx, 3]
                if use_temporal_interp == Int32(1) && temporal_weight < 0.99f0
                    tw = temporal_weight
                    rho_v = parent_rho_old[plx, ply, plz, pb_idx] * (1.0f0 - tw) + rho_v * tw
                    ux_v = parent_vel_old[plx, ply, plz, pb_idx, 1] * (1.0f0 - tw) + ux_v * tw
                    uy_v = parent_vel_old[plx, ply, plz, pb_idx, 2] * (1.0f0 - tw) + uy_v * tw
                    uz_v = parent_vel_old[plx, ply, plz, pb_idx, 3] * (1.0f0 - tw) + uz_v * tw
                end
                return (rho_v, ux_v, uy_v, uz_v, true)
            end
        end
        return (1.0f0, 0.0f0, 0.0f0, 0.0f0, false)
    end

    # CORNER-QUAD FIX (2026-06-09): SYMMETRIC cubic-validity margin.
    # The 4-point cubic stencil {px0-1, px0, px0+1, px0+2} reaches 1 parent cell
    # toward − and 2 toward +. The previous validity test mirrored that asymmetry
    # (px0-1>=1 on −, px0+2<=parent_n on +), so on the +X/+Y/+Z box faces the cubic
    # fell back to extrapolation ONE PARENT CELL EARLIER than on the − faces — a
    # directional bias on the + faces only that compounds where two + faces meet
    # (and the mirror − corner) into the "isolated quads at two OPPOSITE corners"
    # seen in density/schlieren (present with the monolithic kernel too, since it
    # is in the shared prolongation). A SYMMETRIC 2-cell margin on BOTH sides makes
    # the cubic→extrap transition identical on + and − faces, removing the bias.
    use_cubic =
        px0 - Int32(2) >= Int32(1) && px0 + Int32(2) <= parent_nx &&
        (_PERIODIC_Y_AMR ? parent_ny >= Int32(4) :   # periodic-Y AMR: lookups wrap
         (py0 - Int32(2) >= Int32(1) && py0 + Int32(2) <= parent_ny)) &&
        pz0 - Int32(2) >= Int32(1) && pz0 + Int32(2) <= parent_nz

    # ── Trilinear interpolation over 8 parent corners ──
    d000 = get_blended_pair(px0, py0, pz0)
    d100 = get_blended_pair(px1, py0, pz0)
    d010 = get_blended_pair(px0, py1, pz0)
    d110 = get_blended_pair(px1, py1, pz0)
    d001 = get_blended_pair(px0, py0, pz1)
    d101 = get_blended_pair(px1, py0, pz1)
    d011 = get_blended_pair(px0, py1, pz1)
    d111 = get_blended_pair(px1, py1, pz1)

    valid_pair_anchor =
        d000[7] ? d000 :
        d100[7] ? d100 :
        d010[7] ? d010 :
        d110[7] ? d110 :
        d001[7] ? d001 :
        d101[7] ? d101 :
        d011[7] ? d011 : d111

    v000 = d000[7] ? d000 : valid_pair_anchor
    v100 = d100[7] ? d100 : valid_pair_anchor
    v010 = d010[7] ? d010 : valid_pair_anchor
    v110 = d110[7] ? d110 : valid_pair_anchor
    v001 = d001[7] ? d001 : valid_pair_anchor
    v101 = d101[7] ? d101 : valid_pair_anchor
    v011 = d011[7] ? d011 : valid_pair_anchor
    v111 = d111[7] ? d111 : valid_pair_anchor

    @inline function trilin_m(i)
        c00 = v000[i]*(1.0f0-wx) + v100[i]*wx
        c01 = v001[i]*(1.0f0-wx) + v101[i]*wx
        c10 = v010[i]*(1.0f0-wx) + v110[i]*wx
        c11 = v011[i]*(1.0f0-wx) + v111[i]*wx
        c0 = c00*(1.0f0-wy) + c10*wy
        c1 = c01*(1.0f0-wy) + c11*wy
        return c0*(1.0f0-wz) + c1*wz
    end

    f_k_int  = trilin_m(1)   # interpolated f for direction k
    f_kb_int = trilin_m(2)   # interpolated f for direction k̄

    if use_cubic
        # Cubic Lagrange interpolation over the 4×4×4 parent stencil.
        # Applied to BOTH the macroscopic fields (ρ, u) and the raw populations
        # (f_k, f_k̄). Lagrava 2012 §3.7 shows that using linear/trilinear on f_neq
        # produces an O(1/N) pressure jump at the refinement interface and breaks
        # the global second-order accuracy of LBM; cubic is required for f_neq as
        # well as for ρ, u. We fetch everything from the blended-pair accessor in
        # a single cubic sweep so the 64 parent reads are shared between the two
        # interpolations (no extra memory traffic vs. the prior cubic-on-macros path).
        rho_int = 0.0f0
        ux_int = 0.0f0
        uy_int = 0.0f0
        uz_int = 0.0f0
        f_k_cubic = 0.0f0
        f_kb_cubic = 0.0f0
        for ix in Int32(-1):Int32(2)
            px = px0 + ix
            wxv = ix == Int32(-1) ? (-wx*(wx-1.0f0)*(wx-2.0f0)/6.0f0) :
                  ix == Int32(0)  ? ((wx+1.0f0)*(wx-1.0f0)*(wx-2.0f0)/2.0f0) :
                  ix == Int32(1)  ? (-wx*(wx+1.0f0)*(wx-2.0f0)/2.0f0) :
                                    (wx*(wx+1.0f0)*(wx-1.0f0)/6.0f0)
            for iy in Int32(-1):Int32(2)
                py = py0 + iy
                wyv = iy == Int32(-1) ? (-wy*(wy-1.0f0)*(wy-2.0f0)/6.0f0) :
                      iy == Int32(0)  ? ((wy+1.0f0)*(wy-1.0f0)*(wy-2.0f0)/2.0f0) :
                      iy == Int32(1)  ? (-wy*(wy+1.0f0)*(wy-2.0f0)/2.0f0) :
                                        (wy*(wy+1.0f0)*(wy-1.0f0)/6.0f0)
                for iz in Int32(-1):Int32(2)
                    pz = pz0 + iz
                    wzv = iz == Int32(-1) ? (-wz*(wz-1.0f0)*(wz-2.0f0)/6.0f0) :
                          iz == Int32(0)  ? ((wz+1.0f0)*(wz-1.0f0)*(wz-2.0f0)/2.0f0) :
                          iz == Int32(1)  ? (-wz*(wz+1.0f0)*(wz-2.0f0)/2.0f0) :
                                            (wz*(wz+1.0f0)*(wz-1.0f0)/6.0f0)
                    fk_p, fkb_p, r, uxv, uyv, uzv, _ = get_blended_pair(px, py, pz)
                    w3 = wxv * wyv * wzv
                    rho_int += w3 * r
                    ux_int += w3 * uxv
                    uy_int += w3 * uyv
                    uz_int += w3 * uzv
                    f_k_cubic  += w3 * fk_p
                    f_kb_cubic += w3 * fkb_p
                end
            end
        end
        f_k_int  = f_k_cubic
        f_kb_int = f_kb_cubic
    else
        rho_int = trilin_m(3)
        ux_int  = trilin_m(4)
        uy_int  = trilin_m(5)
        uz_int  = trilin_m(6)
    end

    # ── Equilibrium for both directions ──
    feq_k  = calculate_equilibrium_auto(rho_int, ux_int, uy_int, uz_int,
                 w_k, cx, cy, cz, compressible_corr)
    feq_kb = calculate_equilibrium_auto(rho_int, ux_int, uy_int, uz_int,
                 w_kb, cx_b, cy_b, cz_b, compressible_corr)

    # ── Non-equilibrium mirror decomposition ──
    # For entropic KBC, only the mirror-symmetric stress component should be
    # rescaled across levels. Rescaling the antisymmetric higher-order part with
    # the viscous factor injects a checkerboard mode that the fine level then
    # amplifies in the supersonic refined zone.
    f_neq_k  = f_k_int  - feq_k
    f_neq_kb = f_kb_int - feq_kb
    s_k = 0.5f0 * (f_neq_k + f_neq_kb)

    tau_c = tau_coarse - 0.5f0
    tau_f = tau_fine - 0.5f0
    scale = tau_c > 1.0f-6 ? clamp(tau_f / tau_c, 0.01f0, 100.0f0) : 1.0f0

    return max(feq_k + scale * s_k, Float32(1e-10))
end


# ────────────────────────────────────────────────────────────────────
# D3Q39 entropic mirror-decomposed interpolation
# ────────────────────────────────────────────────────────────────────
# Same algorithm as D3Q27 version above, but uses the polynomial
# D3Q39 equilibrium (calculate_equilibrium_d3q39) with cs² = 2/3.
# The mirror decomposition into s (symmetric/stress) and h (antisymmetric)
# is identical — only the equilibrium computation differs.
#
# Arguments are the same as interpolate_entropic_mirror(), with the addition
# of cs2 (speed of sound squared for D3Q39).
# ────────────────────────────────────────────────────────────────────
@inline function project_pi_neq_stress_mode(
    wk::Float32,
    cx::Float32, cy::Float32, cz::Float32,
    cs2::Float32,
    pxx::Float32, pyy::Float32, pzz::Float32,
    pxy::Float32, pxz::Float32, pyz::Float32
)
    # Regularized AMR transfers the hydrodynamic viscous stress, not the
    # isotropic second-order/energy residue.  In the athermal D3Q39 path there
    # is no thermal distribution carrying that trace consistently, so retaining
    # it aliases AMR transfer error into pressure/Mach bands.  Use the
    # deviatoric stress tensor for the population projection.
    tr = (pxx + pyy + pzz) * (1.0f0 / 3.0f0)
    pxx_d = pxx - tr
    pyy_d = pyy - tr
    pzz_d = pzz - tr
    qxx = cx * cx - cs2
    qyy = cy * cy - cs2
    qzz = cz * cz - cs2
    qxy = cx * cy
    qxz = cx * cz
    qyz = cy * cz
    contraction = qxx * pxx_d + qyy * pyy_d + qzz * pzz_d +
                  2.0f0 * (qxy * pxy + qxz * pxz + qyz * pyz)
    return wk * contraction / (2.0f0 * cs2 * cs2)
end

@inline function reconstruct_entropic_interface_population(
    feq_k::Float32,
    feq_kb::Float32,
    s_k::Float32,
    h_k::Float32,
    scale_s::Float32,
    scale_h_target::Float32,
    eps_pop::Float32
)
    base_k = feq_k + scale_s * s_k
    base_kb = feq_kb + scale_s * s_k

    if scale_h_target <= 0.0f0 || abs(h_k) <= eps_pop
        return max(base_k, eps_pop)
    end

    h_abs = abs(h_k)
    max_h_scale = scale_h_target

    if h_k > 0.0f0
        max_h_scale = min(max_h_scale, max(0.0f0, (base_kb - eps_pop) / max(h_abs, eps_pop)))
    else
        max_h_scale = min(max_h_scale, max(0.0f0, (base_k - eps_pop) / max(h_abs, eps_pop)))
    end

    return max(base_k + max_h_scale * h_k, eps_pop)
end


# ── Far-cell macro reader (current-time `_new` only) for the GP-fill helpers ──
# The gradient-preserving box-edge extrapolation needs ONE extra parent cell
# beyond the bracket to form `missing = 2·near − far`. The far cell supplies a
# GRADIENT direction only; using the current-time `_new` arrays (no temporal
# blend) is a sound 2nd-order-negligible simplification that keeps the @noinline
# helper signature small. Returns (rho, ux, uy, uz, T, valid).
@inline function _read_parent_macro_new_39(
    parent_rho_new, parent_vel_new, parent_temperature_new,
    parent_ptr, parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    block_size::Int32, has_T::Bool, T_default::Float32,
    pgx::Int32, pgy::Int32, pgz::Int32)
    pbx = (pgx - Int32(1)) ÷ block_size + Int32(1)
    pgy = _pwrap_y(pgy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
    pby = (pgy - Int32(1)) ÷ block_size + Int32(1)
    pbz = (pgz - Int32(1)) ÷ block_size + Int32(1)
    if pbx >= Int32(1) && pbx <= parent_dim_x &&
       pby >= Int32(1) && pby <= parent_dim_y &&
       pbz >= Int32(1) && pbz <= parent_dim_z
        pb_idx = parent_ptr[pbx, pby, pbz]
        if pb_idx > Int32(0)
            plx = (pgx - Int32(1)) % block_size + Int32(1)
            ply = (pgy - Int32(1)) % block_size + Int32(1)
            plz = (pgz - Int32(1)) % block_size + Int32(1)
            r = parent_rho_new[plx, ply, plz, pb_idx]
            ux = parent_vel_new[plx, ply, plz, pb_idx, 1]
            uy = parent_vel_new[plx, ply, plz, pb_idx, 2]
            uz = parent_vel_new[plx, ply, plz, pb_idx, 3]
            T = has_T ? parent_temperature_new[plx, ply, plz, pb_idx] : T_default
            return (r, ux, uy, uz, T, true)
        end
    end
    return (1.0f0, 0.0f0, 0.0f0, 0.0f0, T_default, false)
end

# ── @noinline gradient-preserving MACRO box-edge fill ──────────────────────
# Extracted from interpolate_entropic_mirror_d3q39 (2026-06-26) so the heavy
# 12-branch extrapolation logic compiles as its OWN unit and does NOT bloat the
# @inline interpolator's IR (which is inlined into the 39-direction entropic
# kernel loop → the cold-compile blew up to >18 min before this split, matching
# the documented 2026-05-28 @inline-threshold incident). Takes the 8 pre-fetched
# corner tuples (d000..d111, layout [3]=rho [4]=ux [5]=uy [6]=uz [7]=T [8]=valid)
# + weights + bracket indices + parent `_new` arrays for far reads. Returns
# (rho, ux, uy, uz, T) — caller floors rho.
@noinline function _gp_macro_fill_39(
    d000, d100, d010, d110, d001, d101, d011, d111,
    w000::Float32, w100::Float32, w010::Float32, w110::Float32,
    w001::Float32, w101::Float32, w011::Float32, w111::Float32,
    px0::Int32, px1::Int32, py0::Int32, py1::Int32, pz0::Int32, pz1::Int32,
    parent_rho_new, parent_vel_new, parent_temperature_new,
    parent_ptr, parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    block_size::Int32, has_T::Bool, T_default::Float32)
    v0=d000[8]; v1=d100[8]; v2=d010[8]; v3=d110[8]
    v4=d001[8]; v5=d101[8]; v6=d011[8]; v7=d111[8]
    if v0 && v1 && v2 && v3 && v4 && v5 && v6 && v7
        rho_a = w000*d000[3]+w100*d100[3]+w010*d010[3]+w110*d110[3]+w001*d001[3]+w101*d101[3]+w011*d011[3]+w111*d111[3]
        ux_a  = w000*d000[4]+w100*d100[4]+w010*d010[4]+w110*d110[4]+w001*d001[4]+w101*d101[4]+w011*d011[4]+w111*d111[4]
        uy_a  = w000*d000[5]+w100*d100[5]+w010*d010[5]+w110*d110[5]+w001*d001[5]+w101*d101[5]+w011*d011[5]+w111*d111[5]
        uz_a  = w000*d000[6]+w100*d100[6]+w010*d010[6]+w110*d110[6]+w001*d001[6]+w101*d101[6]+w011*d011[6]+w111*d111[6]
        T_a   = w000*d000[7]+w100*d100[7]+w010*d010[7]+w110*d110[7]+w001*d001[7]+w101*d101[7]+w011*d011[7]+w111*d111[7]
        return (rho_a, ux_a, uy_a, uz_a, T_a)
    elseif !(v0||v1||v2||v3||v4||v5||v6||v7)
        return (d000[3], d000[4], d000[5], d000[6], d000[7])
    end
    r0=d000[3];x0=d000[4];y0=d000[5];z0=d000[6];t0=d000[7]
    r1=d100[3];x1=d100[4];y1=d100[5];z1=d100[6];t1=d100[7]
    r2=d010[3];x2=d010[4];y2=d010[5];z2=d010[6];t2=d010[7]
    r3=d110[3];x3=d110[4];y3=d110[5];z3=d110[6];t3=d110[7]
    r4=d001[3];x4=d001[4];y4=d001[5];z4=d001[6];t4=d001[7]
    r5=d101[3];x5=d101[4];y5=d101[5];z5=d101[6];t5=d101[7]
    r6=d011[3];x6=d011[4];y6=d011[5];z6=d011[6];t6=d011[7]
    r7=d111[3];x7=d111[4];y7=d111[5];z7=d111[6];t7=d111[7]
    pxm = px0 - Int32(1); pxp = px1 + Int32(1)
    pym = py0 - Int32(1); pyp = py1 + Int32(1)
    pzm = pz0 - Int32(1); pzp = pz1 + Int32(1)
    rd(a,b,c) = _read_parent_macro_new_39(parent_rho_new, parent_vel_new, parent_temperature_new,
        parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z, block_size, has_T, T_default, a, b, c)
    # X pass — pairs (0,1)(2,3)(4,5)(6,7); far = pxm/pxp.
    if v0 != v1
        if v0; fd=rd(pxm,py0,pz0); r1=fd[6] ? 2f0*r0-fd[1] : r0; x1=fd[6] ? 2f0*x0-fd[2] : x0; y1=fd[6] ? 2f0*y0-fd[3] : y0; z1=fd[6] ? 2f0*z0-fd[4] : z0; t1=fd[6] ? 2f0*t0-fd[5] : t0; v1=true
        else;  fd=rd(pxp,py0,pz0); r0=fd[6] ? 2f0*r1-fd[1] : r1; x0=fd[6] ? 2f0*x1-fd[2] : x1; y0=fd[6] ? 2f0*y1-fd[3] : y1; z0=fd[6] ? 2f0*z1-fd[4] : z1; t0=fd[6] ? 2f0*t1-fd[5] : t1; v0=true; end
    end
    if v2 != v3
        if v2; fd=rd(pxm,py1,pz0); r3=fd[6] ? 2f0*r2-fd[1] : r2; x3=fd[6] ? 2f0*x2-fd[2] : x2; y3=fd[6] ? 2f0*y2-fd[3] : y2; z3=fd[6] ? 2f0*z2-fd[4] : z2; t3=fd[6] ? 2f0*t2-fd[5] : t2; v3=true
        else;  fd=rd(pxp,py1,pz0); r2=fd[6] ? 2f0*r3-fd[1] : r3; x2=fd[6] ? 2f0*x3-fd[2] : x3; y2=fd[6] ? 2f0*y3-fd[3] : y3; z2=fd[6] ? 2f0*z3-fd[4] : z3; t2=fd[6] ? 2f0*t3-fd[5] : t3; v2=true; end
    end
    if v4 != v5
        if v4; fd=rd(pxm,py0,pz1); r5=fd[6] ? 2f0*r4-fd[1] : r4; x5=fd[6] ? 2f0*x4-fd[2] : x4; y5=fd[6] ? 2f0*y4-fd[3] : y4; z5=fd[6] ? 2f0*z4-fd[4] : z4; t5=fd[6] ? 2f0*t4-fd[5] : t4; v5=true
        else;  fd=rd(pxp,py0,pz1); r4=fd[6] ? 2f0*r5-fd[1] : r5; x4=fd[6] ? 2f0*x5-fd[2] : x5; y4=fd[6] ? 2f0*y5-fd[3] : y5; z4=fd[6] ? 2f0*z5-fd[4] : z5; t4=fd[6] ? 2f0*t5-fd[5] : t5; v4=true; end
    end
    if v6 != v7
        if v6; fd=rd(pxm,py1,pz1); r7=fd[6] ? 2f0*r6-fd[1] : r6; x7=fd[6] ? 2f0*x6-fd[2] : x6; y7=fd[6] ? 2f0*y6-fd[3] : y6; z7=fd[6] ? 2f0*z6-fd[4] : z6; t7=fd[6] ? 2f0*t6-fd[5] : t6; v7=true
        else;  fd=rd(pxp,py1,pz1); r6=fd[6] ? 2f0*r7-fd[1] : r7; x6=fd[6] ? 2f0*x7-fd[2] : x7; y6=fd[6] ? 2f0*y7-fd[3] : y7; z6=fd[6] ? 2f0*z7-fd[4] : z7; t6=fd[6] ? 2f0*t7-fd[5] : t7; v6=true; end
    end
    # Y pass — pairs (0,2)(1,3)(4,6)(5,7); far = pym/pyp.
    if v0 != v2
        if v0; fd=rd(px0,pym,pz0); r2=fd[6] ? 2f0*r0-fd[1] : r0; x2=fd[6] ? 2f0*x0-fd[2] : x0; y2=fd[6] ? 2f0*y0-fd[3] : y0; z2=fd[6] ? 2f0*z0-fd[4] : z0; t2=fd[6] ? 2f0*t0-fd[5] : t0; v2=true
        else;  fd=rd(px0,pyp,pz0); r0=fd[6] ? 2f0*r2-fd[1] : r2; x0=fd[6] ? 2f0*x2-fd[2] : x2; y0=fd[6] ? 2f0*y2-fd[3] : y2; z0=fd[6] ? 2f0*z2-fd[4] : z2; t0=fd[6] ? 2f0*t2-fd[5] : t2; v0=true; end
    end
    if v1 != v3
        if v1; fd=rd(px1,pym,pz0); r3=fd[6] ? 2f0*r1-fd[1] : r1; x3=fd[6] ? 2f0*x1-fd[2] : x1; y3=fd[6] ? 2f0*y1-fd[3] : y1; z3=fd[6] ? 2f0*z1-fd[4] : z1; t3=fd[6] ? 2f0*t1-fd[5] : t1; v3=true
        else;  fd=rd(px1,pyp,pz0); r1=fd[6] ? 2f0*r3-fd[1] : r3; x1=fd[6] ? 2f0*x3-fd[2] : x3; y1=fd[6] ? 2f0*y3-fd[3] : y3; z1=fd[6] ? 2f0*z3-fd[4] : z3; t1=fd[6] ? 2f0*t3-fd[5] : t3; v1=true; end
    end
    if v4 != v6
        if v4; fd=rd(px0,pym,pz1); r6=fd[6] ? 2f0*r4-fd[1] : r4; x6=fd[6] ? 2f0*x4-fd[2] : x4; y6=fd[6] ? 2f0*y4-fd[3] : y4; z6=fd[6] ? 2f0*z4-fd[4] : z4; t6=fd[6] ? 2f0*t4-fd[5] : t4; v6=true
        else;  fd=rd(px0,pyp,pz1); r4=fd[6] ? 2f0*r6-fd[1] : r6; x4=fd[6] ? 2f0*x6-fd[2] : x6; y4=fd[6] ? 2f0*y6-fd[3] : y6; z4=fd[6] ? 2f0*z6-fd[4] : z6; t4=fd[6] ? 2f0*t6-fd[5] : t6; v4=true; end
    end
    if v5 != v7
        if v5; fd=rd(px1,pym,pz1); r7=fd[6] ? 2f0*r5-fd[1] : r5; x7=fd[6] ? 2f0*x5-fd[2] : x5; y7=fd[6] ? 2f0*y5-fd[3] : y5; z7=fd[6] ? 2f0*z5-fd[4] : z5; t7=fd[6] ? 2f0*t5-fd[5] : t5; v7=true
        else;  fd=rd(px1,pyp,pz1); r5=fd[6] ? 2f0*r7-fd[1] : r7; x5=fd[6] ? 2f0*x7-fd[2] : x7; y5=fd[6] ? 2f0*y7-fd[3] : y7; z5=fd[6] ? 2f0*z7-fd[4] : z7; t5=fd[6] ? 2f0*t7-fd[5] : t7; v5=true; end
    end
    # Z pass — pairs (0,4)(1,5)(2,6)(3,7); far = pzm/pzp.
    if v0 != v4
        if v0; fd=rd(px0,py0,pzm); r4=fd[6] ? 2f0*r0-fd[1] : r0; x4=fd[6] ? 2f0*x0-fd[2] : x0; y4=fd[6] ? 2f0*y0-fd[3] : y0; z4=fd[6] ? 2f0*z0-fd[4] : z0; t4=fd[6] ? 2f0*t0-fd[5] : t0; v4=true
        else;  fd=rd(px0,py0,pzp); r0=fd[6] ? 2f0*r4-fd[1] : r4; x0=fd[6] ? 2f0*x4-fd[2] : x4; y0=fd[6] ? 2f0*y4-fd[3] : y4; z0=fd[6] ? 2f0*z4-fd[4] : z4; t0=fd[6] ? 2f0*t4-fd[5] : t4; v0=true; end
    end
    if v1 != v5
        if v1; fd=rd(px1,py0,pzm); r5=fd[6] ? 2f0*r1-fd[1] : r1; x5=fd[6] ? 2f0*x1-fd[2] : x1; y5=fd[6] ? 2f0*y1-fd[3] : y1; z5=fd[6] ? 2f0*z1-fd[4] : z1; t5=fd[6] ? 2f0*t1-fd[5] : t1; v5=true
        else;  fd=rd(px1,py0,pzp); r1=fd[6] ? 2f0*r5-fd[1] : r5; x1=fd[6] ? 2f0*x5-fd[2] : x5; y1=fd[6] ? 2f0*y5-fd[3] : y5; z1=fd[6] ? 2f0*z5-fd[4] : z5; t1=fd[6] ? 2f0*t5-fd[5] : t5; v1=true; end
    end
    if v2 != v6
        if v2; fd=rd(px0,py1,pzm); r6=fd[6] ? 2f0*r2-fd[1] : r2; x6=fd[6] ? 2f0*x2-fd[2] : x2; y6=fd[6] ? 2f0*y2-fd[3] : y2; z6=fd[6] ? 2f0*z2-fd[4] : z2; t6=fd[6] ? 2f0*t2-fd[5] : t2; v6=true
        else;  fd=rd(px0,py1,pzp); r2=fd[6] ? 2f0*r6-fd[1] : r6; x2=fd[6] ? 2f0*x6-fd[2] : x6; y2=fd[6] ? 2f0*y6-fd[3] : y6; z2=fd[6] ? 2f0*z6-fd[4] : z6; t2=fd[6] ? 2f0*t6-fd[5] : t6; v2=true; end
    end
    if v3 != v7
        if v3; fd=rd(px1,py1,pzm); r7=fd[6] ? 2f0*r3-fd[1] : r3; x7=fd[6] ? 2f0*x3-fd[2] : x3; y7=fd[6] ? 2f0*y3-fd[3] : y3; z7=fd[6] ? 2f0*z3-fd[4] : z3; t7=fd[6] ? 2f0*t3-fd[5] : t3; v7=true
        else;  fd=rd(px1,py1,pzp); r3=fd[6] ? 2f0*r7-fd[1] : r7; x3=fd[6] ? 2f0*x7-fd[2] : x7; y3=fd[6] ? 2f0*y7-fd[3] : y7; z3=fd[6] ? 2f0*z7-fd[4] : z7; t3=fd[6] ? 2f0*t7-fd[5] : t7; v3=true; end
    end
    rho_a = w000*r0+w100*r1+w010*r2+w110*r3+w001*r4+w101*r5+w011*r6+w111*r7
    ux_a  = w000*x0+w100*x1+w010*x2+w110*x3+w001*x4+w101*x5+w011*x6+w111*x7
    uy_a  = w000*y0+w100*y1+w010*y2+w110*y3+w001*y4+w101*y5+w011*y6+w111*y7
    uz_a  = w000*z0+w100*z1+w010*z2+w110*z3+w001*z4+w101*z5+w011*z6+w111*z7
    T_a   = w000*t0+w100*t1+w010*t2+w110*t3+w001*t4+w101*t5+w011*t6+w111*t7
    return (rho_a, ux_a, uy_a, uz_a, T_a)
end

# Far-cell Pi_neq reader (current-time) for the GP stress fill.
@inline function _read_parent_pi_new_39(
    parent_pi_neq, parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    block_size::Int32, pgx::Int32, pgy::Int32, pgz::Int32)
    pbx = div(pgx - Int32(1), block_size) + Int32(1)
    pgy = _pwrap_y(pgy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
    pby = div(pgy - Int32(1), block_size) + Int32(1)
    pbz = div(pgz - Int32(1), block_size) + Int32(1)
    if pbx >= Int32(1) && pbx <= parent_dim_x &&
       pby >= Int32(1) && pby <= parent_dim_y &&
       pbz >= Int32(1) && pbz <= parent_dim_z
        pb_idx = parent_ptr[pbx, pby, pbz]
        if pb_idx > Int32(0)
            plx = (pgx - Int32(1)) % block_size + Int32(1)
            ply = (pgy - Int32(1)) % block_size + Int32(1)
            plz = (pgz - Int32(1)) % block_size + Int32(1)
            return (parent_pi_neq[plx, ply, plz, pb_idx, 1],
                    parent_pi_neq[plx, ply, plz, pb_idx, 2],
                    parent_pi_neq[plx, ply, plz, pb_idx, 3],
                    parent_pi_neq[plx, ply, plz, pb_idx, 4],
                    parent_pi_neq[plx, ply, plz, pb_idx, 5],
                    parent_pi_neq[plx, ply, plz, pb_idx, 6], true)
        end
    end
    return (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, false)
end

# @noinline gradient-preserving Pi_neq STRESS box-edge fill (sibling of macro).
# p-corner layout [1..6]=(pxx,pyy,pzz,pxy,pxz,pyz), [7]=valid. Far reads via
# _read_parent_pi_new_39. @noinline so it compiles separately (cold-compile sanity).
@noinline function _gp_stress_fill_39(
    p000, p100, p010, p110, p001, p101, p011, p111,
    w000::Float32, w100::Float32, w010::Float32, w110::Float32,
    w001::Float32, w101::Float32, w011::Float32, w111::Float32,
    px0::Int32, px1::Int32, py0::Int32, py1::Int32, pz0::Int32, pz1::Int32,
    parent_pi_neq, parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32, block_size::Int32)
    v0=p000[7]; v1=p100[7]; v2=p010[7]; v3=p110[7]
    v4=p001[7]; v5=p101[7]; v6=p011[7]; v7=p111[7]
    if v0 && v1 && v2 && v3 && v4 && v5 && v6 && v7
        return (w000*p000[1]+w100*p100[1]+w010*p010[1]+w110*p110[1]+w001*p001[1]+w101*p101[1]+w011*p011[1]+w111*p111[1],
                w000*p000[2]+w100*p100[2]+w010*p010[2]+w110*p110[2]+w001*p001[2]+w101*p101[2]+w011*p011[2]+w111*p111[2],
                w000*p000[3]+w100*p100[3]+w010*p010[3]+w110*p110[3]+w001*p001[3]+w101*p101[3]+w011*p011[3]+w111*p111[3],
                w000*p000[4]+w100*p100[4]+w010*p010[4]+w110*p110[4]+w001*p001[4]+w101*p101[4]+w011*p011[4]+w111*p111[4],
                w000*p000[5]+w100*p100[5]+w010*p010[5]+w110*p110[5]+w001*p001[5]+w101*p101[5]+w011*p011[5]+w111*p111[5],
                w000*p000[6]+w100*p100[6]+w010*p010[6]+w110*p110[6]+w001*p001[6]+w101*p101[6]+w011*p011[6]+w111*p111[6])
    elseif !(v0||v1||v2||v3||v4||v5||v6||v7)
        return (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
    end
    xx0=p000[1];yy0=p000[2];zz0=p000[3];xy0=p000[4];xz0=p000[5];yz0=p000[6]
    xx1=p100[1];yy1=p100[2];zz1=p100[3];xy1=p100[4];xz1=p100[5];yz1=p100[6]
    xx2=p010[1];yy2=p010[2];zz2=p010[3];xy2=p010[4];xz2=p010[5];yz2=p010[6]
    xx3=p110[1];yy3=p110[2];zz3=p110[3];xy3=p110[4];xz3=p110[5];yz3=p110[6]
    xx4=p001[1];yy4=p001[2];zz4=p001[3];xy4=p001[4];xz4=p001[5];yz4=p001[6]
    xx5=p101[1];yy5=p101[2];zz5=p101[3];xy5=p101[4];xz5=p101[5];yz5=p101[6]
    xx6=p011[1];yy6=p011[2];zz6=p011[3];xy6=p011[4];xz6=p011[5];yz6=p011[6]
    xx7=p111[1];yy7=p111[2];zz7=p111[3];xy7=p111[4];xz7=p111[5];yz7=p111[6]
    pxm = px0 - Int32(1); pxp = px1 + Int32(1)
    pym = py0 - Int32(1); pyp = py1 + Int32(1)
    pzm = pz0 - Int32(1); pzp = pz1 + Int32(1)
    rd(a,b,c) = _read_parent_pi_new_39(parent_pi_neq, parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z, block_size, a, b, c)
    # X pass
    if v0 != v1
        if v0; fd=rd(pxm,py0,pz0); xx1=fd[7] ? 2f0*xx0-fd[1] : xx0; yy1=fd[7] ? 2f0*yy0-fd[2] : yy0; zz1=fd[7] ? 2f0*zz0-fd[3] : zz0; xy1=fd[7] ? 2f0*xy0-fd[4] : xy0; xz1=fd[7] ? 2f0*xz0-fd[5] : xz0; yz1=fd[7] ? 2f0*yz0-fd[6] : yz0; v1=true
        else;  fd=rd(pxp,py0,pz0); xx0=fd[7] ? 2f0*xx1-fd[1] : xx1; yy0=fd[7] ? 2f0*yy1-fd[2] : yy1; zz0=fd[7] ? 2f0*zz1-fd[3] : zz1; xy0=fd[7] ? 2f0*xy1-fd[4] : xy1; xz0=fd[7] ? 2f0*xz1-fd[5] : xz1; yz0=fd[7] ? 2f0*yz1-fd[6] : yz1; v0=true; end
    end
    if v2 != v3
        if v2; fd=rd(pxm,py1,pz0); xx3=fd[7] ? 2f0*xx2-fd[1] : xx2; yy3=fd[7] ? 2f0*yy2-fd[2] : yy2; zz3=fd[7] ? 2f0*zz2-fd[3] : zz2; xy3=fd[7] ? 2f0*xy2-fd[4] : xy2; xz3=fd[7] ? 2f0*xz2-fd[5] : xz2; yz3=fd[7] ? 2f0*yz2-fd[6] : yz2; v3=true
        else;  fd=rd(pxp,py1,pz0); xx2=fd[7] ? 2f0*xx3-fd[1] : xx3; yy2=fd[7] ? 2f0*yy3-fd[2] : yy3; zz2=fd[7] ? 2f0*zz3-fd[3] : zz3; xy2=fd[7] ? 2f0*xy3-fd[4] : xy3; xz2=fd[7] ? 2f0*xz3-fd[5] : xz3; yz2=fd[7] ? 2f0*yz3-fd[6] : yz3; v2=true; end
    end
    if v4 != v5
        if v4; fd=rd(pxm,py0,pz1); xx5=fd[7] ? 2f0*xx4-fd[1] : xx4; yy5=fd[7] ? 2f0*yy4-fd[2] : yy4; zz5=fd[7] ? 2f0*zz4-fd[3] : zz4; xy5=fd[7] ? 2f0*xy4-fd[4] : xy4; xz5=fd[7] ? 2f0*xz4-fd[5] : xz4; yz5=fd[7] ? 2f0*yz4-fd[6] : yz4; v5=true
        else;  fd=rd(pxp,py0,pz1); xx4=fd[7] ? 2f0*xx5-fd[1] : xx5; yy4=fd[7] ? 2f0*yy5-fd[2] : yy5; zz4=fd[7] ? 2f0*zz5-fd[3] : zz5; xy4=fd[7] ? 2f0*xy5-fd[4] : xy5; xz4=fd[7] ? 2f0*xz5-fd[5] : xz5; yz4=fd[7] ? 2f0*yz5-fd[6] : yz5; v4=true; end
    end
    if v6 != v7
        if v6; fd=rd(pxm,py1,pz1); xx7=fd[7] ? 2f0*xx6-fd[1] : xx6; yy7=fd[7] ? 2f0*yy6-fd[2] : yy6; zz7=fd[7] ? 2f0*zz6-fd[3] : zz6; xy7=fd[7] ? 2f0*xy6-fd[4] : xy6; xz7=fd[7] ? 2f0*xz6-fd[5] : xz6; yz7=fd[7] ? 2f0*yz6-fd[6] : yz6; v7=true
        else;  fd=rd(pxp,py1,pz1); xx6=fd[7] ? 2f0*xx7-fd[1] : xx7; yy6=fd[7] ? 2f0*yy7-fd[2] : yy7; zz6=fd[7] ? 2f0*zz7-fd[3] : zz7; xy6=fd[7] ? 2f0*xy7-fd[4] : xy7; xz6=fd[7] ? 2f0*xz7-fd[5] : xz7; yz6=fd[7] ? 2f0*yz7-fd[6] : yz7; v6=true; end
    end
    # Y pass
    if v0 != v2
        if v0; fd=rd(px0,pym,pz0); xx2=fd[7] ? 2f0*xx0-fd[1] : xx0; yy2=fd[7] ? 2f0*yy0-fd[2] : yy0; zz2=fd[7] ? 2f0*zz0-fd[3] : zz0; xy2=fd[7] ? 2f0*xy0-fd[4] : xy0; xz2=fd[7] ? 2f0*xz0-fd[5] : xz0; yz2=fd[7] ? 2f0*yz0-fd[6] : yz0; v2=true
        else;  fd=rd(px0,pyp,pz0); xx0=fd[7] ? 2f0*xx2-fd[1] : xx2; yy0=fd[7] ? 2f0*yy2-fd[2] : yy2; zz0=fd[7] ? 2f0*zz2-fd[3] : zz2; xy0=fd[7] ? 2f0*xy2-fd[4] : xy2; xz0=fd[7] ? 2f0*xz2-fd[5] : xz2; yz0=fd[7] ? 2f0*yz2-fd[6] : yz2; v0=true; end
    end
    if v1 != v3
        if v1; fd=rd(px1,pym,pz0); xx3=fd[7] ? 2f0*xx1-fd[1] : xx1; yy3=fd[7] ? 2f0*yy1-fd[2] : yy1; zz3=fd[7] ? 2f0*zz1-fd[3] : zz1; xy3=fd[7] ? 2f0*xy1-fd[4] : xy1; xz3=fd[7] ? 2f0*xz1-fd[5] : xz1; yz3=fd[7] ? 2f0*yz1-fd[6] : yz1; v3=true
        else;  fd=rd(px1,pyp,pz0); xx1=fd[7] ? 2f0*xx3-fd[1] : xx3; yy1=fd[7] ? 2f0*yy3-fd[2] : yy3; zz1=fd[7] ? 2f0*zz3-fd[3] : zz3; xy1=fd[7] ? 2f0*xy3-fd[4] : xy3; xz1=fd[7] ? 2f0*xz3-fd[5] : xz3; yz1=fd[7] ? 2f0*yz3-fd[6] : yz3; v1=true; end
    end
    if v4 != v6
        if v4; fd=rd(px0,pym,pz1); xx6=fd[7] ? 2f0*xx4-fd[1] : xx4; yy6=fd[7] ? 2f0*yy4-fd[2] : yy4; zz6=fd[7] ? 2f0*zz4-fd[3] : zz4; xy6=fd[7] ? 2f0*xy4-fd[4] : xy4; xz6=fd[7] ? 2f0*xz4-fd[5] : xz4; yz6=fd[7] ? 2f0*yz4-fd[6] : yz4; v6=true
        else;  fd=rd(px0,pyp,pz1); xx4=fd[7] ? 2f0*xx6-fd[1] : xx6; yy4=fd[7] ? 2f0*yy6-fd[2] : yy6; zz4=fd[7] ? 2f0*zz6-fd[3] : zz6; xy4=fd[7] ? 2f0*xy6-fd[4] : xy6; xz4=fd[7] ? 2f0*xz6-fd[5] : xz6; yz4=fd[7] ? 2f0*yz6-fd[6] : yz6; v4=true; end
    end
    if v5 != v7
        if v5; fd=rd(px1,pym,pz1); xx7=fd[7] ? 2f0*xx5-fd[1] : xx5; yy7=fd[7] ? 2f0*yy5-fd[2] : yy5; zz7=fd[7] ? 2f0*zz5-fd[3] : zz5; xy7=fd[7] ? 2f0*xy5-fd[4] : xy5; xz7=fd[7] ? 2f0*xz5-fd[5] : xz5; yz7=fd[7] ? 2f0*yz5-fd[6] : yz5; v7=true
        else;  fd=rd(px1,pyp,pz1); xx5=fd[7] ? 2f0*xx7-fd[1] : xx7; yy5=fd[7] ? 2f0*yy7-fd[2] : yy7; zz5=fd[7] ? 2f0*zz7-fd[3] : zz7; xy5=fd[7] ? 2f0*xy7-fd[4] : xy7; xz5=fd[7] ? 2f0*xz7-fd[5] : xz7; yz5=fd[7] ? 2f0*yz7-fd[6] : yz7; v5=true; end
    end
    # Z pass
    if v0 != v4
        if v0; fd=rd(px0,py0,pzm); xx4=fd[7] ? 2f0*xx0-fd[1] : xx0; yy4=fd[7] ? 2f0*yy0-fd[2] : yy0; zz4=fd[7] ? 2f0*zz0-fd[3] : zz0; xy4=fd[7] ? 2f0*xy0-fd[4] : xy0; xz4=fd[7] ? 2f0*xz0-fd[5] : xz0; yz4=fd[7] ? 2f0*yz0-fd[6] : yz0; v4=true
        else;  fd=rd(px0,py0,pzp); xx0=fd[7] ? 2f0*xx4-fd[1] : xx4; yy0=fd[7] ? 2f0*yy4-fd[2] : yy4; zz0=fd[7] ? 2f0*zz4-fd[3] : zz4; xy0=fd[7] ? 2f0*xy4-fd[4] : xy4; xz0=fd[7] ? 2f0*xz4-fd[5] : xz4; yz0=fd[7] ? 2f0*yz4-fd[6] : yz4; v0=true; end
    end
    if v1 != v5
        if v1; fd=rd(px1,py0,pzm); xx5=fd[7] ? 2f0*xx1-fd[1] : xx1; yy5=fd[7] ? 2f0*yy1-fd[2] : yy1; zz5=fd[7] ? 2f0*zz1-fd[3] : zz1; xy5=fd[7] ? 2f0*xy1-fd[4] : xy1; xz5=fd[7] ? 2f0*xz1-fd[5] : xz1; yz5=fd[7] ? 2f0*yz1-fd[6] : yz1; v5=true
        else;  fd=rd(px1,py0,pzp); xx1=fd[7] ? 2f0*xx5-fd[1] : xx5; yy1=fd[7] ? 2f0*yy5-fd[2] : yy5; zz1=fd[7] ? 2f0*zz5-fd[3] : zz5; xy1=fd[7] ? 2f0*xy5-fd[4] : xy5; xz1=fd[7] ? 2f0*xz5-fd[5] : xz5; yz1=fd[7] ? 2f0*yz5-fd[6] : yz5; v1=true; end
    end
    if v2 != v6
        if v2; fd=rd(px0,py1,pzm); xx6=fd[7] ? 2f0*xx2-fd[1] : xx2; yy6=fd[7] ? 2f0*yy2-fd[2] : yy2; zz6=fd[7] ? 2f0*zz2-fd[3] : zz2; xy6=fd[7] ? 2f0*xy2-fd[4] : xy2; xz6=fd[7] ? 2f0*xz2-fd[5] : xz2; yz6=fd[7] ? 2f0*yz2-fd[6] : yz2; v6=true
        else;  fd=rd(px0,py1,pzp); xx2=fd[7] ? 2f0*xx6-fd[1] : xx6; yy2=fd[7] ? 2f0*yy6-fd[2] : yy6; zz2=fd[7] ? 2f0*zz6-fd[3] : zz6; xy2=fd[7] ? 2f0*xy6-fd[4] : xy6; xz2=fd[7] ? 2f0*xz6-fd[5] : xz6; yz2=fd[7] ? 2f0*yz6-fd[6] : yz6; v2=true; end
    end
    if v3 != v7
        if v3; fd=rd(px1,py1,pzm); xx7=fd[7] ? 2f0*xx3-fd[1] : xx3; yy7=fd[7] ? 2f0*yy3-fd[2] : yy3; zz7=fd[7] ? 2f0*zz3-fd[3] : zz3; xy7=fd[7] ? 2f0*xy3-fd[4] : xy3; xz7=fd[7] ? 2f0*xz3-fd[5] : xz3; yz7=fd[7] ? 2f0*yz3-fd[6] : yz3; v7=true
        else;  fd=rd(px1,py1,pzp); xx3=fd[7] ? 2f0*xx7-fd[1] : xx7; yy3=fd[7] ? 2f0*yy7-fd[2] : yy7; zz3=fd[7] ? 2f0*zz7-fd[3] : zz7; xy3=fd[7] ? 2f0*xy7-fd[4] : xy7; xz3=fd[7] ? 2f0*xz7-fd[5] : xz7; yz3=fd[7] ? 2f0*yz7-fd[6] : yz7; v3=true; end
    end
    pxx_a = w000*xx0+w100*xx1+w010*xx2+w110*xx3+w001*xx4+w101*xx5+w011*xx6+w111*xx7
    pyy_a = w000*yy0+w100*yy1+w010*yy2+w110*yy3+w001*yy4+w101*yy5+w011*yy6+w111*yy7
    pzz_a = w000*zz0+w100*zz1+w010*zz2+w110*zz3+w001*zz4+w101*zz5+w011*zz6+w111*zz7
    pxy_a = w000*xy0+w100*xy1+w010*xy2+w110*xy3+w001*xy4+w101*xy5+w011*xy6+w111*xy7
    pxz_a = w000*xz0+w100*xz1+w010*xz2+w110*xz3+w001*xz4+w101*xz5+w011*xz6+w111*xz7
    pyz_a = w000*yz0+w100*yz1+w010*yz2+w110*yz3+w001*yz4+w101*yz5+w011*yz6+w111*yz7
    return (pxx_a, pyy_a, pzz_a, pxy_a, pxz_a, pyz_a)
end


@inline function interpolate_entropic_mirror_d3q39(
    parent_f_new, parent_rho_new, parent_vel_new,
    parent_f_old, parent_rho_old, parent_vel_old,
    parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    fine_gx::Int32, fine_gy::Int32, fine_gz::Int32,
    k::Int32, k_opp::Int32, block_size::Int32,
    w_k::Float32, cx::Float32, cy::Float32, cz::Float32,
    w_kb::Float32, cx_b::Float32, cy_b::Float32, cz_b::Float32,
    tau_coarse::Float32, tau_fine::Float32,
    temporal_weight::Float32,
    use_temporal_interp::Int32,
    use_macro_temporal_interp::Int32,
    cs2::Float32,
    # Full lattice arrays for 13-moment equilibrium solver
    cx_arr_full, cy_arr_full, cz_arr_full, w_arr_full,
    # Shifted lattice (Phase 3.2): shift vector for Galilean transform.
    # When shift=(0,0,0), reduces to standard behavior.
    shift_x::Float32=0.0f0, shift_y::Float32=0.0f0, shift_z::Float32=0.0f0,
    # Phase 4: parent temperature arrays for two-way thermal coupling
    use_parent_temperature::Int32=Int32(0),
    parent_temperature_new=nothing, parent_temperature_old=nothing,
    use_parent_pi_neq::Int32=Int32(0),
    parent_pi_neq=nothing,
    interface_h_scale::Float32=0.0f0,
    T_default::Float32=1.0f0,
    monotone_velocity::Int32=Int32(0),
    near_wall_eq_only::Int32=Int32(0),
    # Pass176 (Tier A item A2 — quadratic temporal interpolation).
    # Older snapshots of the parent macros: state at t=-dt_c (one parent step
    # ago). Combined with parent_rho_old (t=0) and parent_rho_new (t=+dt_c),
    # these give 3 anchor points for quadratic Lagrange temporal interpolation
    # at fine sub-cycle times. When nothing (default), the macro_temporal
    # branch falls back to standard linear blending — backward-compatible.
    parent_rho_older=nothing, parent_vel_older=nothing,
    parent_temperature_older=nothing,
    use_quadratic_temporal::Int32=Int32(0),
    # B1 P3 (2026-05-28) — cubic Hermite (C²) temporal interpolation.
    # The fourth anchor `parent_*_oldest` lives at t = -2·dt_c. Combined
    # with the three quadratic anchors {older, old, new} at {-dt_c, 0,
    # +dt_c} we form a 4-point cubic Lagrange polynomial through times
    # {-2, -1, 0, 1} (in units of dt_c) and evaluate at t = tw·dt_c.
    # When nothing or stub, falls back to the existing quadratic / linear
    # paths — backward-compatible. Gated by `use_cubic_temporal` flag.
    parent_rho_oldest=nothing, parent_vel_oldest=nothing,
    parent_temperature_oldest=nothing,
    use_cubic_temporal::Int32=Int32(0),
    # B1 P3 RK4-equivalent (2026-05-28) — quintic Lagrange (C³) temporal interp.
    # The fifth anchor `parent_*_oldest2` lives at t = -3·dt_c. Combined with the
    # four cubic anchors {oldest, older, old, new} at {-2,-1,0,+1} we form a
    # 5-point Lagrange polynomial through times {-3,-2,-1,0,+1} (units of dt_c).
    # When nothing/stub, falls back to cubic → quadratic → linear (backward-compat).
    # Gated by `use_quintic_temporal`.
    parent_rho_oldest2=nothing, parent_vel_oldest2=nothing,
    parent_temperature_oldest2=nothing,
    use_quintic_temporal::Int32=Int32(0),
    # SOTA A3 — thermodynamic EOS R = (γ-1)·cv. MUST match the value the
    # collision kernel uses for `eos_R_flow` so the prolongation f_eq solve
    # uses the IDENTICAL equilibrium across the AMR interface. Sentinel -1
    # falls back to cs² (legacy linearized EOS); at T=1 the eos_R term
    # vanishes, so freestream/non-thermodynamic cases are byte-identical.
    eos_R_flow::Float32=Float32(-1.0f0),
    # C2/C4 QUINTIC B-SPLINE prolongation (2026-06-27): when use_bspline_macro==1
    # AND the symmetric ±3 parent stencil is present, the macro fields (ρ,u) are
    # evaluated from the parent's PRECOMPUTED B-spline coefficients (rho/vel
    # _spline_coeff, filled once per coarse step by compute_spline_coeffs!) via the
    # compact 6-tap @noinline interpolate_macro_bspline — C4 continuity + exact 2:1
    # conservation. On a ragged AMR box edge (stencil incomplete) the helper returns
    # ok=false and we keep the cubic-Lagrange macros computed below (byte-identical
    # fallback). The populations are still reconstructed from these macros via the
    # SAME downstream 13-mom Newton solver, and the f_neq/Π carrier is unchanged.
    # DEFAULT off + nothing stubs ⇒ all existing callers are byte-identical.
    rho_spline_coeff=nothing, vel_spline_coeff=nothing,
    use_bspline_macro::Int32=Int32(0)
)
    px_cont = _fine_to_parent_interp_coord(fine_gx)
    py_cont = _fine_to_parent_interp_coord(fine_gy)
    pz_cont = _fine_to_parent_interp_coord(fine_gz)

    has_T = use_parent_temperature == Int32(1)
    has_pi = use_parent_pi_neq == Int32(1)
    parent_nx = parent_dim_x * block_size
    parent_ny = parent_dim_y * block_size
    parent_nz = parent_dim_z * block_size
    px0, px1, wx = _parent_interp_bracket(px_cont, parent_nx)
    py0, py1, wy = _PERIODIC_Y_AMR ? _pbracket_y(py_cont) : _parent_interp_bracket(py_cont, parent_ny)
    pz0, pz1, wz = _parent_interp_bracket(pz_cont, parent_nz)

    use_macro_temporal = use_macro_temporal_interp == Int32(1) &&
                         temporal_weight < 0.99f0
    # Pass154: when macro-temporal interpolation is requested, prefer it over
    # raw population time blending. Population blending can carry a parent
    # half-step basis mismatch into child halos; the macro path blends rho/u/T
    # and reconstructs populations from the local non-equilibrium transfer.
    use_full_temporal = use_temporal_interp == Int32(1) &&
                        temporal_weight < 0.99f0 &&
                        !use_macro_temporal

    # Returns (f_k, f_kb, rho, ux, uy, uz, T, valid) — 8-tuple
    @inline function get_parent_pair_direct_39(pgx::Int32, pgy::Int32, pgz::Int32)
        pbx = (pgx - Int32(1)) ÷ block_size + Int32(1)
        pgy = _pwrap_y(pgy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
        pby = (pgy - Int32(1)) ÷ block_size + Int32(1)
        pbz = (pgz - Int32(1)) ÷ block_size + Int32(1)

        if pbx >= Int32(1) && pbx <= parent_dim_x &&
           pby >= Int32(1) && pby <= parent_dim_y &&
           pbz >= Int32(1) && pbz <= parent_dim_z
            pb_idx = parent_ptr[pbx, pby, pbz]
            if pb_idx > Int32(0)
                plx = (pgx - Int32(1)) % block_size + Int32(1)
                ply = (pgy - Int32(1)) % block_size + Int32(1)
                plz = (pgz - Int32(1)) % block_size + Int32(1)

                fk_new  = parent_f_new[plx, ply, plz, pb_idx, k]
                fkb_new = parent_f_new[plx, ply, plz, pb_idx, k_opp]
                rho_new = parent_rho_new[plx, ply, plz, pb_idx]
                ux_new  = parent_vel_new[plx, ply, plz, pb_idx, 1]
                uy_new  = parent_vel_new[plx, ply, plz, pb_idx, 2]
                uz_new  = parent_vel_new[plx, ply, plz, pb_idx, 3]
                T_new   = has_T ? parent_temperature_new[plx, ply, plz, pb_idx] : T_default

                if use_full_temporal
                    fk_old  = parent_f_old[plx, ply, plz, pb_idx, k]
                    fkb_old = parent_f_old[plx, ply, plz, pb_idx, k_opp]
                    rho_old = parent_rho_old[plx, ply, plz, pb_idx]
                    ux_old  = parent_vel_old[plx, ply, plz, pb_idx, 1]
                    uy_old  = parent_vel_old[plx, ply, plz, pb_idx, 2]
                    uz_old  = parent_vel_old[plx, ply, plz, pb_idx, 3]
                    T_old   = has_T ? parent_temperature_old[plx, ply, plz, pb_idx] : T_default

                    tw = temporal_weight
                    return (fk_old*(1.0f0-tw) + fk_new*tw,
                            fkb_old*(1.0f0-tw) + fkb_new*tw,
                            rho_old*(1.0f0-tw) + rho_new*tw,
                            ux_old*(1.0f0-tw) + ux_new*tw,
                            uy_old*(1.0f0-tw) + uy_new*tw,
                            uz_old*(1.0f0-tw) + uz_new*tw,
                            T_old*(1.0f0-tw) + T_new*tw, true)
                end
                if use_macro_temporal
                    rho_old = parent_rho_old[plx, ply, plz, pb_idx]
                    ux_old  = parent_vel_old[plx, ply, plz, pb_idx, 1]
                    uy_old  = parent_vel_old[plx, ply, plz, pb_idx, 2]
                    uz_old  = parent_vel_old[plx, ply, plz, pb_idx, 3]
                    T_old   = has_T ? parent_temperature_old[plx, ply, plz, pb_idx] : T_default
                    tw = temporal_weight
                    # Pass176: quadratic Lagrange in time through 3 anchors:
                    #   parent_*_older at t = -dt_c (1 parent step ago)
                    #   parent_*_old   at t =  0    (start of current step)
                    #   parent_*_new   at t = +dt_c (end of current step)
                    # Evaluate at t = tw·dt_c (fine substep time, tw ∈ [0,1]).
                    # Weights: w_older = tw(tw-1)/2, w_old = 1−tw², w_new = (tw+1)tw/2
                    # At tw=0.5: {-0.125, 0.75, 0.375} vs linear {0, 0.5, 0.5}.
                    # The difference is proportional to the 2nd time derivative
                    # of the parent state (zero for steady flow). Raises temporal
                    # continuity at substep boundaries from C⁰ to C¹.
                    @inline function _soft_clamp(q::Float32, v_old::Float32, v_new::Float32)
                        v_min = min(v_old, v_new)
                        v_max = max(v_old, v_new)
                        delta = 0.1f0 * max(v_max - v_min, 1.0f-6)
                        return clamp(q, v_min - delta, v_max + delta)
                    end

                    # B1 P3 (2026-05-28) — Cubic Hermite (C²) temporal blend.
                    # 4-point Lagrange polynomial through {oldest, older, old, new}
                    # at times {-2, -1, 0, +1} (units of dt_c) evaluated at τ=tw.
                    # Closed-form basis (derived from the standard Lagrange
                    # construction; sums to 1 by partition-of-unity):
                    #   L_{-2}(τ) = -(τ-1)·τ·(τ+1) / 6
                    #   L_{-1}(τ) =  (τ-1)·τ·(τ+2) / 2
                    #   L_{0}(τ)  = -(τ-1)·(τ+1)·(τ+2) / 2
                    #   L_{+1}(τ) =  τ·(τ+1)·(τ+2) / 6
                    # At τ=0.5 these reduce to {0.0625, -0.3125, 0.9375, 0.3125}.
                    # The same soft-clamp from the quadratic path is applied,
                    # bounded relative to [old, new], to keep the extrapolation
                    # overshoot tame in transients (the cubic correction term
                    # is proportional to ∂³/∂t³ of the parent state, which is
                    # large during ramp curvature phases).
                    # B1 P3 RK4-equivalent (2026-05-28) — quintic Lagrange (C³)
                    # temporal blend through 5 anchors {oldest2,oldest,older,old,new}
                    # at times {-3,-2,-1,0,+1} (units of dt_c), evaluated at τ=tw.
                    # Weights from _quintic_lagrange_time_weights (sum≡1); the same
                    # soft-clamp as the cubic path bounds the negative Lagrange lobes.
                    # Falls through to cubic → quadratic → linear when any anchor is
                    # nothing/stub (the first 4 parent steps of the warm-up chain).
                    if use_quintic_temporal == Int32(1) &&
                       parent_rho_oldest2 !== nothing &&
                       parent_rho_oldest !== nothing &&
                       parent_rho_older !== nothing
                        wq5 = _quintic_lagrange_time_weights(tw)
                        ro_o2 = parent_rho_oldest2[plx, ply, plz, pb_idx]
                        ro_o1 = parent_rho_oldest[plx, ply, plz, pb_idx]
                        ro_or = parent_rho_older[plx, ply, plz, pb_idx]
                        ux_o2 = parent_vel_oldest2[plx, ply, plz, pb_idx, 1]
                        ux_o1 = parent_vel_oldest[plx, ply, plz, pb_idx, 1]
                        ux_or = parent_vel_older[plx, ply, plz, pb_idx, 1]
                        uy_o2 = parent_vel_oldest2[plx, ply, plz, pb_idx, 2]
                        uy_o1 = parent_vel_oldest[plx, ply, plz, pb_idx, 2]
                        uy_or = parent_vel_older[plx, ply, plz, pb_idx, 2]
                        uz_o2 = parent_vel_oldest2[plx, ply, plz, pb_idx, 3]
                        uz_o1 = parent_vel_oldest[plx, ply, plz, pb_idx, 3]
                        uz_or = parent_vel_older[plx, ply, plz, pb_idx, 3]
                        T_o2 = (has_T && parent_temperature_oldest2 !== nothing) ?
                               parent_temperature_oldest2[plx, ply, plz, pb_idx] : T_default
                        T_o1 = (has_T && parent_temperature_oldest !== nothing) ?
                               parent_temperature_oldest[plx, ply, plz, pb_idx] : T_default
                        T_or = (has_T && parent_temperature_older !== nothing) ?
                               parent_temperature_older[plx, ply, plz, pb_idx] : T_default
                        rho_q5 = wq5[1]*ro_o2 + wq5[2]*ro_o1 + wq5[3]*ro_or + wq5[4]*rho_old + wq5[5]*rho_new
                        ux_q5  = wq5[1]*ux_o2 + wq5[2]*ux_o1 + wq5[3]*ux_or + wq5[4]*ux_old + wq5[5]*ux_new
                        uy_q5  = wq5[1]*uy_o2 + wq5[2]*uy_o1 + wq5[3]*uy_or + wq5[4]*uy_old + wq5[5]*uy_new
                        uz_q5  = wq5[1]*uz_o2 + wq5[2]*uz_o1 + wq5[3]*uz_or + wq5[4]*uz_old + wq5[5]*uz_new
                        T_q5   = wq5[1]*T_o2  + wq5[2]*T_o1  + wq5[3]*T_or  + wq5[4]*T_old  + wq5[5]*T_new
                        return (fk_new,
                                fkb_new,
                                max(_quintic_soft_clamp(rho_q5, rho_old, rho_new), 0.01f0),
                                _quintic_soft_clamp(ux_q5, ux_old, ux_new),
                                _quintic_soft_clamp(uy_q5, uy_old, uy_new),
                                _quintic_soft_clamp(uz_q5, uz_old, uz_new),
                                _quintic_soft_clamp(T_q5,  T_old,  T_new), true)
                    end

                    if use_cubic_temporal == Int32(1) &&
                       parent_rho_oldest !== nothing &&
                       parent_rho_older !== nothing
                        rho_oldest_v = parent_rho_oldest[plx, ply, plz, pb_idx]
                        ux_oldest_v  = parent_vel_oldest[plx, ply, plz, pb_idx, 1]
                        uy_oldest_v  = parent_vel_oldest[plx, ply, plz, pb_idx, 2]
                        uz_oldest_v  = parent_vel_oldest[plx, ply, plz, pb_idx, 3]
                        T_oldest_v   = (has_T && parent_temperature_oldest !== nothing) ?
                                       parent_temperature_oldest[plx, ply, plz, pb_idx] : T_default
                        rho_older_v = parent_rho_older[plx, ply, plz, pb_idx]
                        ux_older_v  = parent_vel_older[plx, ply, plz, pb_idx, 1]
                        uy_older_v  = parent_vel_older[plx, ply, plz, pb_idx, 2]
                        uz_older_v  = parent_vel_older[plx, ply, plz, pb_idx, 3]
                        T_older_v   = (has_T && parent_temperature_older !== nothing) ?
                                      parent_temperature_older[plx, ply, plz, pb_idx] : T_default
                        τ = tw
                        τ_minus1 = τ - 1.0f0
                        τ_plus1  = τ + 1.0f0
                        τ_plus2  = τ + 2.0f0
                        w_oldest = -(τ_minus1 * τ * τ_plus1) * (1.0f0 / 6.0f0)
                        w_older  =  (τ_minus1 * τ * τ_plus2) * 0.5f0
                        w_old    = -(τ_minus1 * τ_plus1 * τ_plus2) * 0.5f0
                        w_new    =  (τ * τ_plus1 * τ_plus2) * (1.0f0 / 6.0f0)
                        rho_c = w_oldest*rho_oldest_v + w_older*rho_older_v + w_old*rho_old + w_new*rho_new
                        ux_c  = w_oldest*ux_oldest_v  + w_older*ux_older_v  + w_old*ux_old  + w_new*ux_new
                        uy_c  = w_oldest*uy_oldest_v  + w_older*uy_older_v  + w_old*uy_old  + w_new*uy_new
                        uz_c  = w_oldest*uz_oldest_v  + w_older*uz_older_v  + w_old*uz_old  + w_new*uz_new
                        T_c   = w_oldest*T_oldest_v   + w_older*T_older_v   + w_old*T_old   + w_new*T_new
                        return (fk_new,
                                fkb_new,
                                max(_soft_clamp(rho_c, rho_old, rho_new), 0.01f0),
                                _soft_clamp(ux_c, ux_old, ux_new),
                                _soft_clamp(uy_c, uy_old, uy_new),
                                _soft_clamp(uz_c, uz_old, uz_new),
                                _soft_clamp(T_c,  T_old,  T_new), true)
                    end

                    if use_quadratic_temporal == Int32(1) && parent_rho_older !== nothing
                        rho_older_v = parent_rho_older[plx, ply, plz, pb_idx]
                        ux_older_v  = parent_vel_older[plx, ply, plz, pb_idx, 1]
                        uy_older_v  = parent_vel_older[plx, ply, plz, pb_idx, 2]
                        uz_older_v  = parent_vel_older[plx, ply, plz, pb_idx, 3]
                        T_older_v   = (has_T && parent_temperature_older !== nothing) ?
                                      parent_temperature_older[plx, ply, plz, pb_idx] : T_default
                        # Quadratic Lagrange weights: w_older negative (-0.125 at tw=0.5).
                        # This is mathematically interpolation but BEHAVIORALLY extrapolation
                        # when |∂²/∂t²| is large (e.g., during smoothstep ramp's curvature
                        # phase). Pass182: apply a per-field SOFT CLAMP to limit quadratic
                        # result to [min(old,new) − δ, max(old,new) + δ] with δ = 10% of
                        # |old−new|. This bounds the extrapolation overshoot while
                        # preserving full C¹ continuity in smooth temporal regions where
                        # the quadratic value naturally lies inside the bounded range.
                        # No step-gate needed — the clamp self-disables in transients.
                        w_older = tw * (tw - 1.0f0) * 0.5f0
                        w_old   = 1.0f0 - tw * tw
                        w_new   = (tw + 1.0f0) * tw * 0.5f0
                        rho_q = w_older*rho_older_v + w_old*rho_old + w_new*rho_new
                        ux_q  = w_older*ux_older_v  + w_old*ux_old  + w_new*ux_new
                        uy_q  = w_older*uy_older_v  + w_old*uy_old  + w_new*uy_new
                        uz_q  = w_older*uz_older_v  + w_old*uz_old  + w_new*uz_new
                        T_q   = w_older*T_older_v   + w_old*T_old   + w_new*T_new
                        return (fk_new,
                                fkb_new,
                                max(_soft_clamp(rho_q, rho_old, rho_new), 0.01f0),
                                _soft_clamp(ux_q, ux_old, ux_new),
                                _soft_clamp(uy_q, uy_old, uy_new),
                                _soft_clamp(uz_q, uz_old, uz_new),
                                _soft_clamp(T_q,  T_old,  T_new), true)
                    end
                    # Linear fallback (no older array provided).
                    return (fk_new,
                            fkb_new,
                            rho_old*(1.0f0-tw) + rho_new*tw,
                            ux_old*(1.0f0-tw) + ux_new*tw,
                            uy_old*(1.0f0-tw) + uy_new*tw,
                            uz_old*(1.0f0-tw) + uz_new*tw,
                            T_old*(1.0f0-tw) + T_new*tw, true)
                end
                return (fk_new, fkb_new, rho_new, ux_new, uy_new, uz_new, T_new, true)
            end
        end
        return (w_k, w_kb, 1.0f0, 0.0f0, 0.0f0, 0.0f0, T_default, false)
    end

    @inline function get_blended_pair_39(pgx::Int32, pgy::Int32, pgz::Int32)
        fk_v, fkb_v, rho_v, ux_v, uy_v, uz_v, T_v, valid =
            get_parent_pair_direct_39(pgx, pgy, pgz)
        if valid
            return (fk_v, fkb_v, rho_v, ux_v, uy_v, uz_v, T_v, true)
        end

        # When the direct coarse donor is missing, keep the fallback centered on
        # the *nearest* valid parent shell only. Averaging over all surviving
        # context neighbors is symmetric only if the missing stencil is itself
        # symmetric; on truncated AMR frontiers it can lean toward the populated
        # side and produce upper/lower edge bias in the fine-level imprint.
        best_dist = Int32(4)
        @inbounds for dx in Int32(-1):Int32(1), dy in Int32(-1):Int32(1), dz in Int32(-1):Int32(1)
            if dx == Int32(0) && dy == Int32(0) && dz == Int32(0)
                continue
            end
            _, _, _, _, _, _, _, valid_n =
                get_parent_pair_direct_39(pgx + dx, pgy + dy, pgz + dz)
            if valid_n
                dist = abs(dx) + abs(dy) + abs(dz)
                best_dist = min(best_dist, dist)
            end
        end

        best_dist == Int32(4) && return (fk_v, fkb_v, rho_v, ux_v, uy_v, uz_v, T_v, false)

        fk_acc = 0.0f0
        fkb_acc = 0.0f0
        rho_acc = 0.0f0
        ux_acc = 0.0f0
        uy_acc = 0.0f0
        uz_acc = 0.0f0
        T_acc = 0.0f0
        donor_count = Int32(0)
        @inbounds for dx in Int32(-1):Int32(1), dy in Int32(-1):Int32(1), dz in Int32(-1):Int32(1)
            if dx == Int32(0) && dy == Int32(0) && dz == Int32(0)
                continue
            end
            dist = abs(dx) + abs(dy) + abs(dz)
            dist == best_dist || continue
            fk_n, fkb_n, rho_n, ux_n, uy_n, uz_n, T_n, valid_n =
                get_parent_pair_direct_39(pgx + dx, pgy + dy, pgz + dz)
            if valid_n
                fk_acc += fk_n
                fkb_acc += fkb_n
                rho_acc += rho_n
                ux_acc += ux_n
                uy_acc += uy_n
                uz_acc += uz_n
                T_acc += T_n
                donor_count += Int32(1)
            end
        end
        if donor_count > Int32(0)
            inv_w = 1.0f0 / Float32(donor_count)
            return (
                fk_acc * inv_w,
                fkb_acc * inv_w,
                rho_acc * inv_w,
                ux_acc * inv_w,
                uy_acc * inv_w,
                uz_acc * inv_w,
                T_acc * inv_w,
                true
            )
        end
        return (fk_v, fkb_v, rho_v, ux_v, uy_v, uz_v, T_v, false)
    end

    # Fetch (rho, ux, uy, uz, T) only from a parent cell
    @inline function get_macro_m_39(pgx::Int32, pgy::Int32, pgz::Int32)
        pbx = (pgx - Int32(1)) ÷ block_size + Int32(1)
        pgy = _pwrap_y(pgy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
        pby = (pgy - Int32(1)) ÷ block_size + Int32(1)
        pbz = (pgz - Int32(1)) ÷ block_size + Int32(1)
        if pbx >= Int32(1) && pbx <= parent_dim_x &&
           pby >= Int32(1) && pby <= parent_dim_y &&
           pbz >= Int32(1) && pbz <= parent_dim_z
            pb_idx = parent_ptr[pbx, pby, pbz]
            if pb_idx > Int32(0)
                plx = (pgx - Int32(1)) % block_size + Int32(1)
                ply = (pgy - Int32(1)) % block_size + Int32(1)
                plz = (pgz - Int32(1)) % block_size + Int32(1)
                rho_v = parent_rho_new[plx, ply, plz, pb_idx]
                ux_v  = parent_vel_new[plx, ply, plz, pb_idx, 1]
                uy_v  = parent_vel_new[plx, ply, plz, pb_idx, 2]
                uz_v  = parent_vel_new[plx, ply, plz, pb_idx, 3]
                T_v   = has_T ? parent_temperature_new[plx, ply, plz, pb_idx] : T_default
                if use_macro_temporal
                    tw = temporal_weight
                    rho_v = parent_rho_old[plx, ply, plz, pb_idx] * (1.0f0 - tw) + rho_v * tw
                    ux_v  = parent_vel_old[plx, ply, plz, pb_idx, 1] * (1.0f0 - tw) + ux_v * tw
                    uy_v  = parent_vel_old[plx, ply, plz, pb_idx, 2] * (1.0f0 - tw) + uy_v * tw
                    uz_v  = parent_vel_old[plx, ply, plz, pb_idx, 3] * (1.0f0 - tw) + uz_v * tw
                    if has_T
                        T_v = parent_temperature_old[plx, ply, plz, pb_idx] * (1.0f0 - tw) + T_v * tw
                    end
                end
                return (rho_v, ux_v, uy_v, uz_v, T_v, true)
            end
        end
        return (1.0f0, 0.0f0, 0.0f0, 0.0f0, T_default, false)
    end

    @inline function get_pi_direct_39(pgx::Int32, pgy::Int32, pgz::Int32)
        if !has_pi
            return (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, false)
        end

        pbx = div(pgx - Int32(1), block_size) + Int32(1)
        pgy = _pwrap_y(pgy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
        pby = div(pgy - Int32(1), block_size) + Int32(1)
        pbz = div(pgz - Int32(1), block_size) + Int32(1)
        if pbx >= Int32(1) && pbx <= parent_dim_x &&
           pby >= Int32(1) && pby <= parent_dim_y &&
           pbz >= Int32(1) && pbz <= parent_dim_z
            pb_idx = parent_ptr[pbx, pby, pbz]
            if pb_idx > Int32(0)
                plx = (pgx - Int32(1)) % block_size + Int32(1)
                ply = (pgy - Int32(1)) % block_size + Int32(1)
                plz = (pgz - Int32(1)) % block_size + Int32(1)
                return (
                    parent_pi_neq[plx, ply, plz, pb_idx, 1],
                    parent_pi_neq[plx, ply, plz, pb_idx, 2],
                    parent_pi_neq[plx, ply, plz, pb_idx, 3],
                    parent_pi_neq[plx, ply, plz, pb_idx, 4],
                    parent_pi_neq[plx, ply, plz, pb_idx, 5],
                    parent_pi_neq[plx, ply, plz, pb_idx, 6],
                    true
                )
            end
        end
        return (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, false)
    end

    @inline function get_pi_m_39(pgx::Int32, pgy::Int32, pgz::Int32)
        pxx_v, pyy_v, pzz_v, pxy_v, pxz_v, pyz_v, valid =
            get_pi_direct_39(pgx, pgy, pgz)
        if valid
            return (pxx_v, pyy_v, pzz_v, pxy_v, pxz_v, pyz_v, true)
        end

        best_dist = Int32(4)
        @inbounds for dx in Int32(-1):Int32(1), dy in Int32(-1):Int32(1), dz in Int32(-1):Int32(1)
            if dx == Int32(0) && dy == Int32(0) && dz == Int32(0)
                continue
            end
            _, _, _, _, _, _, valid_n =
                get_pi_direct_39(pgx + dx, pgy + dy, pgz + dz)
            if valid_n
                dist = abs(dx) + abs(dy) + abs(dz)
                best_dist = min(best_dist, dist)
            end
        end

        best_dist == Int32(4) && return (pxx_v, pyy_v, pzz_v, pxy_v, pxz_v, pyz_v, false)

        pxx_acc = 0.0f0
        pyy_acc = 0.0f0
        pzz_acc = 0.0f0
        pxy_acc = 0.0f0
        pxz_acc = 0.0f0
        pyz_acc = 0.0f0
        donor_count = Int32(0)
        @inbounds for dx in Int32(-1):Int32(1), dy in Int32(-1):Int32(1), dz in Int32(-1):Int32(1)
            if dx == Int32(0) && dy == Int32(0) && dz == Int32(0)
                continue
            end
            dist = abs(dx) + abs(dy) + abs(dz)
            dist == best_dist || continue
            pxx_n, pyy_n, pzz_n, pxy_n, pxz_n, pyz_n, valid_n =
                get_pi_direct_39(pgx + dx, pgy + dy, pgz + dz)
            if valid_n
                pxx_acc += pxx_n
                pyy_acc += pyy_n
                pzz_acc += pzz_n
                pxy_acc += pxy_n
                pxz_acc += pxz_n
                pyz_acc += pyz_n
                donor_count += Int32(1)
            end
        end
        if donor_count > Int32(0)
            inv_w = 1.0f0 / Float32(donor_count)
            return (
                pxx_acc * inv_w,
                pyy_acc * inv_w,
                pzz_acc * inv_w,
                pxy_acc * inv_w,
                pxz_acc * inv_w,
                pyz_acc * inv_w,
                true
            )
        end
        return (pxx_v, pyy_v, pzz_v, pxy_v, pxz_v, pyz_v, false)
    end

    # Cubic Lagrange is always attempted — the accumulator now tracks the
    # per-stencil weight mass of valid parents (`cubic_w_sum`) and
    # normalizes at the end. Previously a boolean `use_cubic` guard cut
    # the cubic path entirely whenever the 4×4×4 stencil extended past
    # the parent grid (px0−1 < 1 or px0+2 > parent_nx), dropping to
    # trilinear for that cell. Trilinear has an O(1) pressure jump at
    # coarse–fine interfaces (Lagrava 2012 §3.7), so the fallback showed
    # up as stripes at the transverse domain walls and at the upstream
    # corners of every refinement level. Weighted renormalization keeps
    # the cubic accuracy everywhere except the pathological case where
    # the entire stencil is invalid (wsum → 0), in which case trilinear
    # is still a safe last resort.

    d000 = get_blended_pair_39(px0, py0, pz0)
    d100 = get_blended_pair_39(px1, py0, pz0)
    d010 = get_blended_pair_39(px0, py1, pz0)
    d110 = get_blended_pair_39(px1, py1, pz0)
    d001 = get_blended_pair_39(px0, py0, pz1)
    d101 = get_blended_pair_39(px1, py0, pz1)
    d011 = get_blended_pair_39(px0, py1, pz1)
    d111 = get_blended_pair_39(px1, py1, pz1)
    p000 = get_pi_m_39(px0, py0, pz0)
    p100 = get_pi_m_39(px1, py0, pz0)
    p010 = get_pi_m_39(px0, py1, pz0)
    p110 = get_pi_m_39(px1, py1, pz0)
    p001 = get_pi_m_39(px0, py0, pz1)
    p101 = get_pi_m_39(px1, py0, pz1)
    p011 = get_pi_m_39(px0, py1, pz1)
    p111 = get_pi_m_39(px1, py1, pz1)

    w000 = (1.0f0 - wx) * (1.0f0 - wy) * (1.0f0 - wz)
    w100 = wx * (1.0f0 - wy) * (1.0f0 - wz)
    w010 = (1.0f0 - wx) * wy * (1.0f0 - wz)
    w110 = wx * wy * (1.0f0 - wz)
    w001 = (1.0f0 - wx) * (1.0f0 - wy) * wz
    w101 = wx * (1.0f0 - wy) * wz
    w011 = (1.0f0 - wx) * wy * wz
    w111 = wx * wy * wz

    if near_wall_eq_only == Int32(1) &&
       d000[8] && d100[8] && d010[8] && d110[8] &&
       d001[8] && d101[8] && d011[8] && d111[8]
        rho_const = 0.125f0 * (d000[3] + d100[3] + d010[3] + d110[3] +
                               d001[3] + d101[3] + d011[3] + d111[3])
        ux_const = 0.125f0 * (d000[4] + d100[4] + d010[4] + d110[4] +
                              d001[4] + d101[4] + d011[4] + d111[4])
        uy_const = 0.125f0 * (d000[5] + d100[5] + d010[5] + d110[5] +
                              d001[5] + d101[5] + d011[5] + d111[5])
        uz_const = 0.125f0 * (d000[6] + d100[6] + d010[6] + d110[6] +
                              d001[6] + d101[6] + d011[6] + d111[6])
        T_const = 0.125f0 * (d000[7] + d100[7] + d010[7] + d110[7] +
                             d001[7] + d101[7] + d011[7] + d111[7])

        max_const_dev = 0.0f0
        max_const_dev = max(max_const_dev, abs(d000[3] - rho_const)); max_const_dev = max(max_const_dev, abs(d100[3] - rho_const))
        max_const_dev = max(max_const_dev, abs(d010[3] - rho_const)); max_const_dev = max(max_const_dev, abs(d110[3] - rho_const))
        max_const_dev = max(max_const_dev, abs(d001[3] - rho_const)); max_const_dev = max(max_const_dev, abs(d101[3] - rho_const))
        max_const_dev = max(max_const_dev, abs(d011[3] - rho_const)); max_const_dev = max(max_const_dev, abs(d111[3] - rho_const))
        max_const_dev = max(max_const_dev, abs(d000[4] - ux_const));  max_const_dev = max(max_const_dev, abs(d100[4] - ux_const))
        max_const_dev = max(max_const_dev, abs(d010[4] - ux_const));  max_const_dev = max(max_const_dev, abs(d110[4] - ux_const))
        max_const_dev = max(max_const_dev, abs(d001[4] - ux_const));  max_const_dev = max(max_const_dev, abs(d101[4] - ux_const))
        max_const_dev = max(max_const_dev, abs(d011[4] - ux_const));  max_const_dev = max(max_const_dev, abs(d111[4] - ux_const))
        max_const_dev = max(max_const_dev, abs(d000[5] - uy_const));  max_const_dev = max(max_const_dev, abs(d100[5] - uy_const))
        max_const_dev = max(max_const_dev, abs(d010[5] - uy_const));  max_const_dev = max(max_const_dev, abs(d110[5] - uy_const))
        max_const_dev = max(max_const_dev, abs(d001[5] - uy_const));  max_const_dev = max(max_const_dev, abs(d101[5] - uy_const))
        max_const_dev = max(max_const_dev, abs(d011[5] - uy_const));  max_const_dev = max(max_const_dev, abs(d111[5] - uy_const))
        max_const_dev = max(max_const_dev, abs(d000[6] - uz_const));  max_const_dev = max(max_const_dev, abs(d100[6] - uz_const))
        max_const_dev = max(max_const_dev, abs(d010[6] - uz_const));  max_const_dev = max(max_const_dev, abs(d110[6] - uz_const))
        max_const_dev = max(max_const_dev, abs(d001[6] - uz_const));  max_const_dev = max(max_const_dev, abs(d101[6] - uz_const))
        max_const_dev = max(max_const_dev, abs(d011[6] - uz_const));  max_const_dev = max(max_const_dev, abs(d111[6] - uz_const))
        max_const_dev = max(max_const_dev, abs(d000[7] - T_const));   max_const_dev = max(max_const_dev, abs(d100[7] - T_const))
        max_const_dev = max(max_const_dev, abs(d010[7] - T_const));   max_const_dev = max(max_const_dev, abs(d110[7] - T_const))
        max_const_dev = max(max_const_dev, abs(d001[7] - T_const));   max_const_dev = max(max_const_dev, abs(d101[7] - T_const))
        max_const_dev = max(max_const_dev, abs(d011[7] - T_const));   max_const_dev = max(max_const_dev, abs(d111[7] - T_const))

        if max_const_dev <= 2.0f-6
            lambda_const = solve_13mom_lambdas_d3q39(
                max(rho_const, 0.01f0),
                ux_const - shift_x,
                uy_const - shift_y,
                uz_const - shift_z,
                cx_arr_full, cy_arr_full, cz_arr_full, w_arr_full,
                cs2, Int32(80), T_const, shift_x, shift_y, shift_z,
                100.0f0
            )
            mass_corr_const = compute_13mom_mass_correction_d3q39(
                max(rho_const, 0.01f0), lambda_const,
                cx_arr_full, cy_arr_full, cz_arr_full, w_arr_full,
                cs2, shift_x, shift_y, shift_z
            )
            return max(
                mass_corr_const * eval_13mom_feq_single(
                    max(rho_const, 0.01f0),
                    lambda_const,
                    cx - shift_x,
                    cy - shift_y,
                    cz - shift_z,
                    w_k,
                    cs2
                ),
                Float32(1e-10)
            )
        end
    end

    @inline function trilin_39(i)
        accum = 0.0f0
        wsum = 0.0f0
        if d000[8]; accum += w000 * d000[i]; wsum += w000; end
        if d100[8]; accum += w100 * d100[i]; wsum += w100; end
        if d010[8]; accum += w010 * d010[i]; wsum += w010; end
        if d110[8]; accum += w110 * d110[i]; wsum += w110; end
        if d001[8]; accum += w001 * d001[i]; wsum += w001; end
        if d101[8]; accum += w101 * d101[i]; wsum += w101; end
        if d011[8]; accum += w011 * d011[i]; wsum += w011; end
        if d111[8]; accum += w111 * d111[i]; wsum += w111; end
        if wsum > Float32(1e-12)
            return accum / wsum
        end
        return d000[i]
    end

    @inline function trilin_pi_39(i)
        accum = 0.0f0
        wsum = 0.0f0
        if p000[7]; accum += w000 * p000[i]; wsum += w000; end
        if p100[7]; accum += w100 * p100[i]; wsum += w100; end
        if p010[7]; accum += w010 * p010[i]; wsum += w010; end
        if p110[7]; accum += w110 * p110[i]; wsum += w110; end
        if p001[7]; accum += w001 * p001[i]; wsum += w001; end
        if p101[7]; accum += w101 * p101[i]; wsum += w101; end
        if p011[7]; accum += w011 * p011[i]; wsum += w011; end
        if p111[7]; accum += w111 * p111[i]; wsum += w111; end
        if wsum > Float32(1e-12)
            return accum / wsum
        end
        return 0.0f0
    end

    f_k_int  = trilin_39(1)
    f_kb_int = trilin_39(2)

    # ── Gradient-preserving partial-stencil MACRO fill (D3Q39 port, 2026-06-26) ──
    # ROOT CAUSE of the D3Q39 AMR "density wall" (schlieren shows the refinement-box
    # edges as the brightest |∇ρ| feature, brighter than the bow shock — the flow is
    # blocked at the box edges). The legacy `trilin_39` RENORMALIZES over the valid
    # corners (`accum / wsum`, wsum < 1 at the box edge) — which does NOT reproduce a
    # linear field. For ANY gradient (the supersonic compression ahead of the body,
    # the inlet→outlet pressure drop) that biases the halo macros toward the interior
    # side → a systematic ρ/u step at the box edge = the density wall.
    #
    # This is EXACTLY the bug that was fixed for the D3Q27 path on 2026-05-31
    # (`sample_parent_macro_for_fine_cell`, lines ~460-579) but never ported to the
    # D3Q39 entropic path. FIX (identical algorithm): reconstruct each INVALID corner
    # by axis-mirror linear extrapolation from valid corners (3 separable 1D passes:
    # x, y, z; `missing = 2·near − far`). For a linear field this reproduces the
    # missing corner EXACTLY, so the FULL-WEIGHT trilinear sum (Σw = 1, NO
    # renormalization) is gradient-exact at the box edge → the seam vanishes.
    # Applied to the 5 macros (ρ, ux, uy, uz, T = corner indices 3,4,5,6,7); the f_k
    # populations keep `trilin_39` since they are reconstructed from these macros via
    # the 13-mom solver downstream. Plain scalar locals only — GPU-JIT-safe.
    rho_gp, ux_trilin, uy_trilin, uz_trilin, T_trilin = _gp_macro_fill_39(
        d000, d100, d010, d110, d001, d101, d011, d111,
        w000, w100, w010, w110, w001, w101, w011, w111,
        px0, px1, py0, py1, pz0, pz1,
        parent_rho_new, parent_vel_new,
        has_T ? parent_temperature_new : parent_rho_new,   # substitute a real array when !has_T (avoid passing nothing to @noinline)
        parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z,
        block_size, has_T, T_default)
    rho_trilin = max(rho_gp, 0.01f0)

    # ── Gradient-preserving partial-stencil STRESS (Π_neq) fill (2026-06-26) ──
    # SIBLING of the density-wall bug, found by adversarial verification: the
    # stress helper `trilin_pi_39` ALSO renormalizes over valid corners only
    # (`accum / wsum`, wsum < 1 at the box edge) — the identical bias, now on the
    # Π_neq stress tensor that builds the s_k stress carrier (project_pi_neq_stress_mode).
    # A signed-tensor gradient distorted at the edge = wrong viscous/shear stress in
    # the halo. Same fix as the macros: reconstruct invalid corners by axis-mirror
    # linear extrapolation (3 separable passes; `missing = 2·near − far`), full-weight
    # sum (Σw=1, NO renormalization) → gradient-exact. p-corner layout from
    # get_pi_m_39: [1..6] = (pxx,pyy,pzz,pxy,pxz,pyz), [7] = valid. The far cell uses
    # get_pi_m_39 (which itself donor-averages on a fully-invalid cell). Only computed
    # when Π is tracked; if not, these are inert. GPU-JIT-safe scalar code.
    # GP stress fill via @noinline helper (extracted 2026-06-26 to keep the
    # entropic-kernel cold-compile from blowing past the @inline threshold).
    pxx_trilin, pyy_trilin, pzz_trilin, pxy_trilin, pxz_trilin, pyz_trilin = has_pi ?
        _gp_stress_fill_39(
            p000, p100, p010, p110, p001, p101, p011, p111,
            w000, w100, w010, w110, w001, w101, w011, w111,
            px0, px1, py0, py1, pz0, pz1,
            parent_pi_neq, parent_ptr,
            parent_dim_x, parent_dim_y, parent_dim_z, block_size) :
        (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)

    cubic_all_valid_39 =
        px0 - Int32(1) >= Int32(1) && px0 + Int32(2) <= parent_nx &&
        (_PERIODIC_Y_AMR ? parent_ny >= Int32(4) :   # periodic-Y AMR: lookups wrap
         (py0 - Int32(1) >= Int32(1) && py0 + Int32(2) <= parent_ny)) &&
        pz0 - Int32(1) >= Int32(1) && pz0 + Int32(2) <= parent_nz
    cubic_pi_all_valid_39 = cubic_all_valid_39
    if cubic_all_valid_39
        for ix in Int32(-1):Int32(2), iy in Int32(-1):Int32(2), iz in Int32(-1):Int32(2)
            _, _, _, _, _, _, _, valid = get_parent_pair_direct_39(px0 + ix, py0 + iy, pz0 + iz)
            if !valid
                cubic_all_valid_39 = false
                cubic_pi_all_valid_39 = false
                break
            end
            if has_pi
                _, _, _, _, _, _, pi_valid = get_pi_direct_39(px0 + ix, py0 + iy, pz0 + iz)
                if !pi_valid
                    cubic_pi_all_valid_39 = false
                end
            end
        end
    end

    have_cubic_macros = false
    have_cubic_pi = false
    pxx_cubic = 0.0f0
    pyy_cubic = 0.0f0
    pzz_cubic = 0.0f0
    pxy_cubic = 0.0f0
    pxz_cubic = 0.0f0
    pyz_cubic = 0.0f0
    # Cubic Lagrange interpolation over the 4×4×4 parent stencil.
    # Applied in a SINGLE sweep to: ρ, u, T (macros), f_k, f_k̄ (populations),
    # and Π_neq (stress tensor, 6 components) — so the 64 parent reads are
    # shared across all three. Lagrava 2012 §3.7 shows that trilinear/linear
    # interpolation of f_neq-equivalent quantities breaks global 2nd-order
    # LBM accuracy and produces an O(1/N) pressure jump at the refinement
    # interface; cubic is mandatory wherever at least part of the stencil is
    # inside the parent domain.
    #
    # Stencil cells that fall outside the parent grid or inside an inactive
    # block are substituted by the *center* parent (px0, py0, pz0), which is
    # guaranteed valid (px0/py0/pz0 are clamped to ≥ 1 above and every fine
    # cell has an active parent by `ensure_complete_parent_coverage!`).
    # Effect: the 1-D cubic weights continue to sum to exactly 1.0 (no
    # renormalization), so the interpolant is amplitude-preserving and
    # approaches the anchor value smoothly at the stencil edge.
    #
    # The earlier scheme excluded invalid cells from the sum and divided by
    # the remaining `cubic_w_sum`. Because cubic Lagrange weights include
    # negative lobes (w_{-1}, w_{+2} < 0 for wx ∈ (0,1)), dropping those
    # lobes makes Σw > 1 — so the renormalized interpolant undershoots the
    # true value at rate (Σw − 1). This bias repeated every 8 fine cells
    # (one coarse-cell spacing) and produced a visible comb pattern at the
    # top/bottom edges of every refinement box in x-direction slices.
    anchor_fk  = d000[1]
    anchor_fkb = d000[2]
    anchor_r   = d000[3]
    anchor_ux  = d000[4]
    anchor_uy  = d000[5]
    anchor_uz  = d000[6]
    anchor_T   = d000[7]
    anchor_valid = d000[8]
    anchor_pxx = p000[1]
    anchor_pyy = p000[2]
    anchor_pzz = p000[3]
    anchor_pxy = p000[4]
    anchor_pxz = p000[5]
    anchor_pyz = p000[6]
    anchor_pi_valid = p000[7]
    rho_cubic = 0.0f0
    ux_cubic = 0.0f0
    uy_cubic = 0.0f0
    uz_cubic = 0.0f0
    T_cubic = 0.0f0
    f_k_cubic = 0.0f0
    f_kb_cubic = 0.0f0
    cubic_any_valid = false
    cubic_pi_any_valid = false
    cubic_truncated_x = false
    for ix in Int32(-1):Int32(2)
        px = px0 + ix
        wxv = ix == Int32(-1) ? (-wx*(wx-1.0f0)*(wx-2.0f0)/6.0f0) :
              ix == Int32(0)  ? ((wx+1.0f0)*(wx-1.0f0)*(wx-2.0f0)/2.0f0) :
              ix == Int32(1)  ? (-wx*(wx+1.0f0)*(wx-2.0f0)/2.0f0) :
                                (wx*(wx+1.0f0)*(wx-1.0f0)/6.0f0)
        for iy in Int32(-1):Int32(2)
            py = py0 + iy
            wyv = iy == Int32(-1) ? (-wy*(wy-1.0f0)*(wy-2.0f0)/6.0f0) :
                  iy == Int32(0)  ? ((wy+1.0f0)*(wy-1.0f0)*(wy-2.0f0)/2.0f0) :
                  iy == Int32(1)  ? (-wy*(wy+1.0f0)*(wy-2.0f0)/2.0f0) :
                                    (wy*(wy+1.0f0)*(wy-1.0f0)/6.0f0)
            for iz in Int32(-1):Int32(2)
                pz = pz0 + iz
                wzv = iz == Int32(-1) ? (-wz*(wz-1.0f0)*(wz-2.0f0)/6.0f0) :
                      iz == Int32(0)  ? ((wz+1.0f0)*(wz-1.0f0)*(wz-2.0f0)/2.0f0) :
                      iz == Int32(1)  ? (-wz*(wz+1.0f0)*(wz-2.0f0)/2.0f0) :
                                        (wz*(wz+1.0f0)*(wz-1.0f0)/6.0f0)
                w3 = wxv * wyv * wzv
                fk_p, fkb_p, r, uxv, uyv, uzv, Tv, valid =
                    get_parent_pair_direct_39(px, py, pz)
                if !valid
                    if px < px0 || px > px1
                        cubic_truncated_x = true
                    end
                    # Higher-order cubic lobes should not borrow shell-averaged
                    # donors from outside the active parent patch. On truncated
                    # AMR x-faces that prints a one-coarse-cell cadence into the
                    # fine halo. Collapse the missing lobe back onto the nearest
                    # donor inside the local 2x2x2 parent cube instead; this
                    # preserves the cubic weight sum and degrades smoothly to a
                    # lower-order extension exactly on the truncated axis.
                    cpx = px < px0 ? px0 : (px > px1 ? px1 : px)
                    cpy = py < py0 ? py0 : (py > py1 ? py1 : py)
                    cpz = pz < pz0 ? pz0 : (pz > pz1 ? pz1 : pz)
                    fk_p, fkb_p, r, uxv, uyv, uzv, Tv, valid =
                        get_parent_pair_direct_39(cpx, cpy, cpz)
                    if !valid
                        fk_p, fkb_p, r, uxv, uyv, uzv, Tv, valid =
                            get_blended_pair_39(cpx, cpy, cpz)
                    end
                end
                if !valid
                    fk_p  = anchor_fk
                    fkb_p = anchor_fkb
                    r     = anchor_r
                    uxv   = anchor_ux
                    uyv   = anchor_uy
                    uzv   = anchor_uz
                    Tv    = anchor_T
                else
                    cubic_any_valid = true
                end
                rho_cubic += w3 * r
                ux_cubic  += w3 * uxv
                uy_cubic  += w3 * uyv
                uz_cubic  += w3 * uzv
                T_cubic   += w3 * Tv
                f_k_cubic  += w3 * fk_p
                f_kb_cubic += w3 * fkb_p
                if has_pi
                    pxx_p, pyy_p, pzz_p, pxy_p, pxz_p, pyz_p, pi_valid =
                        get_pi_direct_39(px, py, pz)
                    if !pi_valid
                        if px < px0 || px > px1
                            cubic_truncated_x = true
                        end
                        cpx = px < px0 ? px0 : (px > px1 ? px1 : px)
                        cpy = py < py0 ? py0 : (py > py1 ? py1 : py)
                        cpz = pz < pz0 ? pz0 : (pz > pz1 ? pz1 : pz)
                        pxx_p, pyy_p, pzz_p, pxy_p, pxz_p, pyz_p, pi_valid =
                            get_pi_direct_39(cpx, cpy, cpz)
                        if !pi_valid
                            pxx_p, pyy_p, pzz_p, pxy_p, pxz_p, pyz_p, pi_valid =
                                get_pi_m_39(cpx, cpy, cpz)
                        end
                    end
                    if !pi_valid
                        pxx_p = anchor_pxx
                        pyy_p = anchor_pyy
                        pzz_p = anchor_pzz
                        pxy_p = anchor_pxy
                        pxz_p = anchor_pxz
                        pyz_p = anchor_pyz
                    else
                        cubic_pi_any_valid = true
                    end
                    pxx_cubic += w3 * pxx_p
                    pyy_cubic += w3 * pyy_p
                    pzz_cubic += w3 * pzz_p
                    pxy_cubic += w3 * pxy_p
                    pxz_cubic += w3 * pxz_p
                    pyz_cubic += w3 * pyz_p
                end
            end
        end
    end
    # Weights sum to exactly 1 by construction, so no division is needed.
    # Cubic output is accepted below only when every direct parent donor exists;
    # otherwise the already-computed trilinear fallback is used.
    # Only use the D3Q39 cubic path when the entire parent stencil is direct
    # and active.  Truncated/anchor-extended cubic looked smooth in scalar
    # fields but left a derivative-level AMR footprint in vorticity/schlieren.
    # The domain builder now adds a parent donor halo around refined patches,
    # so valid AMR interfaces keep cubic support without one-sided substitutes.
    accept_cubic_macros = cubic_all_valid_39 && cubic_any_valid && anchor_valid
    accept_cubic_pi = has_pi && cubic_pi_all_valid_39 && cubic_pi_any_valid && anchor_pi_valid

    # Pass157: extend pass155's 4x2x4 support fallback to every single
    # truncated axis. The current YZ/XZ images show the strongest residual on
    # top/bottom AMR rails, i.e. the complementary 4x4x2 support case. Keep
    # cubic interpolation in the two fully-supported axes and use linear only
    # in the axis whose 4-point parent stencil is unavailable.
    #
    # Pass174g (SOTA roadmap Tier 3 / item 14, sub-task a): replaced cubic
    # Lagrange with Catmull-Rom cubic Hermite. Same 4-point stencil cost,
    # same call signature, but C¹ globally at parent cell boundaries instead
    # of C⁰. The Catmull-Rom slope at every parent cell center is the
    # central-difference (v[i+1] - v[i-1])/2, which is identical whether
    # computed from the left or right sliding stencil — so derivatives match
    # across stencil shifts (the source of the previous C⁰ break).
    #
    # Hermite basis at fractional position q ∈ [0,1] in the [v0, v1] interval:
    #   h00(q) = 2q³ − 3q² + 1     (value at v0)
    #   h10(q) = q³ − 2q² + q      (slope at v0)
    #   h01(q) = −2q³ + 3q²        (value at v1)
    #   h11(q) = q³ − q²           (slope at v1)
    # With Catmull-Rom slopes m0 = (v1−v(-1))/2, m1 = (v2−v0)/2, the per-axis
    # weight on each of the 4 stencil cells is:
    #   w(-1) = −h10/2          = (−q³ + 2q² − q) / 2
    #   w( 0) = h00 − h11/2     = (3q³ − 5q² + 2) / 2
    #   w( 1) = h01 + h10/2     = (−3q³ + 4q² + q) / 2
    #   w( 2) = h11/2           = (q³ − q²) / 2
    # At q=0.5 these reduce to {−1/16, 9/16, 9/16, −1/16}, identical to
    # cubic Lagrange at the midpoint (so old "Palabos midpoint" tests
    # remain bit-identical). At q=0.25 / 0.75 they differ: Hermite gives
    # MORE weight on the nearer point and LESS on the far points, which is
    # exactly what makes the result C¹ at parent cell boundaries.
    @inline function _lagrange4_axis_weight(q::Float32, offset::Int32)
        q2 = q * q
        q3 = q2 * q
        return offset == Int32(-1) ? ((-q3 + 2.0f0*q2 - q) * 0.5f0) :
               offset == Int32(0)  ? ((3.0f0*q3 - 5.0f0*q2 + 2.0f0) * 0.5f0) :
               offset == Int32(1)  ? ((-3.0f0*q3 + 4.0f0*q2 + q) * 0.5f0) :
                                      ((q3 - q2) * 0.5f0)
    end

    @inline function _tensor_axis_bounds_ok(linear_axis::Int32)
        x_ok = linear_axis == Int32(1) ?
               (px0 >= Int32(1) && px1 <= parent_nx) :
               (px0 - Int32(1) >= Int32(1) && px0 + Int32(2) <= parent_nx)
        y_ok = _PERIODIC_Y_AMR ? (parent_ny >= Int32(4)) :   # periodic-Y AMR: lookups wrap
               linear_axis == Int32(2) ?
               (py0 >= Int32(1) && py1 <= parent_ny) :
               (py0 - Int32(1) >= Int32(1) && py0 + Int32(2) <= parent_ny)
        z_ok = linear_axis == Int32(3) ?
               (pz0 >= Int32(1) && pz1 <= parent_nz) :
               (pz0 - Int32(1) >= Int32(1) && pz0 + Int32(2) <= parent_nz)
        return x_ok && y_ok && z_ok
    end

    @inline function _tensor_offset_active(offset::Int32, linear_axis::Int32, axis::Int32)
        return axis == linear_axis ? (offset == Int32(0) || offset == Int32(1)) : true
    end

    @inline function _tensor_axis_weight(q::Float32, offset::Int32,
                                         linear_axis::Int32, axis::Int32)
        return axis == linear_axis ?
               (offset == Int32(0) ? (1.0f0 - q) : q) :
               _lagrange4_axis_weight(q, offset)
    end

    @inline function _tensor_direct_valid(linear_axis::Int32)
        _tensor_axis_bounds_ok(linear_axis) || return false
        @inbounds for ix in Int32(-1):Int32(2), iy in Int32(-1):Int32(2), iz in Int32(-1):Int32(2)
            _tensor_offset_active(ix, linear_axis, Int32(1)) || continue
            _tensor_offset_active(iy, linear_axis, Int32(2)) || continue
            _tensor_offset_active(iz, linear_axis, Int32(3)) || continue
            _, _, _, _, _, _, _, valid =
                get_parent_pair_direct_39(px0 + ix, py0 + iy, pz0 + iz)
            valid || return false
        end
        return true
    end

    @inline function _tensor_pi_direct_valid(linear_axis::Int32)
        has_pi || return false
        _tensor_axis_bounds_ok(linear_axis) || return false
        @inbounds for ix in Int32(-1):Int32(2), iy in Int32(-1):Int32(2), iz in Int32(-1):Int32(2)
            _tensor_offset_active(ix, linear_axis, Int32(1)) || continue
            _tensor_offset_active(iy, linear_axis, Int32(2)) || continue
            _tensor_offset_active(iz, linear_axis, Int32(3)) || continue
            _, _, _, _, _, _, pi_valid =
                get_pi_direct_39(px0 + ix, py0 + iy, pz0 + iz)
            pi_valid || return false
        end
        return true
    end

    tensor_macro_axis_count = Int32(0)
    tensor_pi_axis_count = Int32(0)
    rho_tensor_sum = 0.0f0
    ux_tensor_sum = 0.0f0
    uy_tensor_sum = 0.0f0
    uz_tensor_sum = 0.0f0
    T_tensor_sum = 0.0f0
    fk_tensor_sum = 0.0f0
    fkb_tensor_sum = 0.0f0
    pxx_tensor_sum = 0.0f0
    pyy_tensor_sum = 0.0f0
    pzz_tensor_sum = 0.0f0
    pxy_tensor_sum = 0.0f0
    pxz_tensor_sum = 0.0f0
    pyz_tensor_sum = 0.0f0

    # Pass178 REVERTED: the "symmetric corner stencil" change (`if false` to
    # skip the tensor-product axis-degraded fallback and fall directly to
    # pure trilinear) was a REGRESSION. It eliminated the original corner
    # lumps but introduced WORSE continuous halo bands along every AMR edge,
    # because trading C¹-in-2-axes (the old edge behavior) for C⁰-everywhere
    # at boundaries dragged ALL boundary cells down instead of pulling the
    # corners up. Restored to the original `!accept_cubic_macros` gate so
    # edges retain the cubic-in-2-axes fallback. Corner lumps remain — the
    # true fix is the Tier 3 mortar coupling (SOTA_ROADMAP item 14).
    if !accept_cubic_macros
        @inbounds for tensor_linear_axis in Int32(1):Int32(3)
            _tensor_direct_valid(tensor_linear_axis) || continue
            rho_tensor = 0.0f0
            ux_tensor = 0.0f0
            uy_tensor = 0.0f0
            uz_tensor = 0.0f0
            T_tensor = 0.0f0
            fk_tensor = 0.0f0
            fkb_tensor = 0.0f0
            pxx_tensor = 0.0f0
            pyy_tensor = 0.0f0
            pzz_tensor = 0.0f0
            pxy_tensor = 0.0f0
            pxz_tensor = 0.0f0
            pyz_tensor = 0.0f0
            tensor_any_valid = false
            tensor_pi_any_valid = false
            tensor_axis_pi_valid = _tensor_pi_direct_valid(tensor_linear_axis)

            @inbounds for ix in Int32(-1):Int32(2)
                _tensor_offset_active(ix, tensor_linear_axis, Int32(1)) || continue
                px = px0 + ix
                wxv = _tensor_axis_weight(wx, ix, tensor_linear_axis, Int32(1))
                for iy in Int32(-1):Int32(2)
                    _tensor_offset_active(iy, tensor_linear_axis, Int32(2)) || continue
                    py = py0 + iy
                    wyv = _tensor_axis_weight(wy, iy, tensor_linear_axis, Int32(2))
                    for iz in Int32(-1):Int32(2)
                        _tensor_offset_active(iz, tensor_linear_axis, Int32(3)) || continue
                        pz = pz0 + iz
                        wzv = _tensor_axis_weight(wz, iz, tensor_linear_axis, Int32(3))
                        w3 = wxv * wyv * wzv
                        fk_p, fkb_p, r, uxv, uyv, uzv, Tv, valid =
                            get_parent_pair_direct_39(px, py, pz)
                        if valid
                            tensor_any_valid = true
                            rho_tensor += w3 * r
                            ux_tensor += w3 * uxv
                            uy_tensor += w3 * uyv
                            uz_tensor += w3 * uzv
                            T_tensor += w3 * Tv
                            fk_tensor += w3 * fk_p
                            fkb_tensor += w3 * fkb_p
                        end
                        if tensor_axis_pi_valid
                            pxx_p, pyy_p, pzz_p, pxy_p, pxz_p, pyz_p, pi_valid =
                                get_pi_direct_39(px, py, pz)
                            if pi_valid
                                tensor_pi_any_valid = true
                                pxx_tensor += w3 * pxx_p
                                pyy_tensor += w3 * pyy_p
                                pzz_tensor += w3 * pzz_p
                                pxy_tensor += w3 * pxy_p
                                pxz_tensor += w3 * pxz_p
                                pyz_tensor += w3 * pyz_p
                            end
                        end
                    end
                end
            end

            if tensor_any_valid
                tensor_macro_axis_count += Int32(1)
                rho_tensor_sum += rho_tensor
                ux_tensor_sum += ux_tensor
                uy_tensor_sum += uy_tensor
                uz_tensor_sum += uz_tensor
                T_tensor_sum += T_tensor
                fk_tensor_sum += fk_tensor
                fkb_tensor_sum += fkb_tensor
            end
            if tensor_axis_pi_valid && tensor_pi_any_valid
                tensor_pi_axis_count += Int32(1)
                pxx_tensor_sum += pxx_tensor
                pyy_tensor_sum += pyy_tensor
                pzz_tensor_sum += pzz_tensor
                pxy_tensor_sum += pxy_tensor
                pxz_tensor_sum += pxz_tensor
                pyz_tensor_sum += pyz_tensor
            end
        end
    end

    if tensor_macro_axis_count > Int32(0) && anchor_valid
        inv_tensor_axes = 1.0f0 / Float32(tensor_macro_axis_count)
        rho_cubic = rho_tensor_sum * inv_tensor_axes
        ux_cubic = ux_tensor_sum * inv_tensor_axes
        uy_cubic = uy_tensor_sum * inv_tensor_axes
        uz_cubic = uz_tensor_sum * inv_tensor_axes
        T_cubic = T_tensor_sum * inv_tensor_axes
        f_k_cubic = fk_tensor_sum * inv_tensor_axes
        f_kb_cubic = fkb_tensor_sum * inv_tensor_axes
        cubic_any_valid = true
        accept_cubic_macros = true
    end
    if tensor_pi_axis_count > Int32(0) && anchor_pi_valid
        inv_tensor_pi_axes = 1.0f0 / Float32(tensor_pi_axis_count)
        pxx_cubic = pxx_tensor_sum * inv_tensor_pi_axes
        pyy_cubic = pyy_tensor_sum * inv_tensor_pi_axes
        pzz_cubic = pzz_tensor_sum * inv_tensor_pi_axes
        pxy_cubic = pxy_tensor_sum * inv_tensor_pi_axes
        pxz_cubic = pxz_tensor_sum * inv_tensor_pi_axes
        pyz_cubic = pyz_tensor_sum * inv_tensor_pi_axes
        cubic_pi_any_valid = true
        accept_cubic_pi = true
    end

    if accept_cubic_macros
        rho_int  = max(rho_cubic, 0.01f0)
        ux_int   = ux_cubic
        uy_int   = uy_cubic
        uz_int   = uz_cubic
        T_int    = T_cubic
        f_k_int  = f_k_cubic
        f_kb_int = f_kb_cubic
        have_cubic_macros = true
    else
        # ISOLATION-REBUILD BASELINE (2026-06-25): reverted to the as-found anchor
        # trilinear box-edge fallback. The gradient-preserving extrapolation port
        # (sample_parent_macro_for_fine_cell) saturated the refined box and is
        # reverted; it will be re-tested in isolation if the baseline warrants it.
        rho_int = rho_trilin
        ux_int  = ux_trilin
        uy_int  = uy_trilin
        uz_int  = uz_trilin
        T_int   = T_trilin
    end
    if accept_cubic_pi
        have_cubic_pi = true
    end

    # ── C2/C4 quintic B-spline MACRO override (2026-06-27, D3Q39 port) ──────
    # When enabled and the symmetric ±3 parent stencil is present, replace the
    # (cubic/trilinear) macro (ρ,u) with the C4-continuous + exactly-2:1-
    # conservative B-spline evaluation read from the parent's precomputed
    # coefficients (filled once per coarse step by compute_spline_coeffs!). This
    # is the SAME @noinline interpolate_macro_bspline used by the D3Q27 path
    # (interpolate_with_rescaling), so the macro is lattice-identical across the
    # AMR seam. ONLY the macro (ρ,u) changes — T_int, f_k_int/f_kb_int, and the
    # Π_neq / population non-equilibrium carrier are untouched (the populations
    # are reconstructed from these macros by the downstream 13-mom Newton solve).
    # On any inactive tap the helper returns ok=false and we keep the cubic macros
    # (the ragged-edge fallback). `have_cubic_macros` is cleared on success so the
    # monotone clamp below (a 2×2×2 trilinear-box limiter) does NOT distort the
    # conservative C4 result — mirrors the D3Q27 path, which leaves B-spline output
    # unclamped. use_bspline_macro=0 (default) ⇒ byte-identical to the legacy path.
    if use_bspline_macro == Int32(1) &&
       rho_spline_coeff !== nothing && vel_spline_coeff !== nothing
        rb, uxb, uyb, uzb, okb = interpolate_macro_bspline(
            rho_spline_coeff, vel_spline_coeff, parent_ptr,
            parent_dim_x, parent_dim_y, parent_dim_z,
            fine_gx, fine_gy, fine_gz, block_size)
        if okb
            rho_int = max(rb, 0.01f0)
            ux_int  = uxb
            uy_int  = uyb
            uz_int  = uzb
            have_cubic_macros = false
        end
    end

    # ── Monotone bounds preservation (local maximum principle) ──────────
    # Cubic Lagrange has negative outer lobes (weights −1/16) that produce
    # a small over/undershoot on any non-monotone parent profile. For a
    # transonic Ma=0.84 freestream the amplitude is O(1e-4), but flow-aligned
    # convection (Pe ≈ 80 along x) carries that seam across ~80 fine cells
    # before viscosity can smear it — producing the visible vertical band
    # at every refinement-box upstream face. The fix is to eliminate the
    # overshoot at its source rather than dampen it after the fact.
    #
    # Clamp every interpolated quantity to [min, max] across the 8 parents
    # of the local 2x2x2 inner cube. This is the standard finite-volume
    # "local bounds preserving" monotone-cubic result: the fine value is
    # always a convex combination of its 8 closest parents, so it cannot
    # overshoot. In smooth regions where the cubic already lies within
    # these bounds the clamp is a no-op and 3rd-order accuracy is preserved.
    # At non-monotone regions the clamp tightens the interpolant toward
    # trilinear, which is the correct behaviour (cubic isn't appropriate
    # there anyway).
    #
    # This supersedes the previous x-only "truncated-stencil fallback" and
    # "streamwise monotone-profile limiter": both were partial fixes that
    # only acted in x; bounds preservation acts in all directions simultaneously
    # and is symmetric in +x/-x/+y/-y/+z/-z. It does NOT touch Π_neq (stress
    # has no monotonicity property — it's a signed derivative tensor).
    if have_cubic_macros
        # Seed min/max with the anchor value, then fold in the other 7 cells
        # if they are valid. d000 is guaranteed valid by anchor_valid above.
        @inline function _mm(vmin, vmax, d, idx)
            if d[8]
                v = d[idx]
                return (min(vmin, v), max(vmax, v))
            end
            return (vmin, vmax)
        end

        # ρ bounds
        rho_min = d000[3]; rho_max = d000[3]
        rho_min, rho_max = _mm(rho_min, rho_max, d100, 3)
        rho_min, rho_max = _mm(rho_min, rho_max, d010, 3)
        rho_min, rho_max = _mm(rho_min, rho_max, d110, 3)
        rho_min, rho_max = _mm(rho_min, rho_max, d001, 3)
        rho_min, rho_max = _mm(rho_min, rho_max, d101, 3)
        rho_min, rho_max = _mm(rho_min, rho_max, d011, 3)
        rho_min, rho_max = _mm(rho_min, rho_max, d111, 3)
        rho_int = max(clamp(rho_int, rho_min, rho_max), 0.01f0)

        if monotone_velocity == Int32(1)
            ux_min = d000[4]; ux_max = d000[4]
            ux_min, ux_max = _mm(ux_min, ux_max, d100, 4)
            ux_min, ux_max = _mm(ux_min, ux_max, d010, 4)
            ux_min, ux_max = _mm(ux_min, ux_max, d110, 4)
            ux_min, ux_max = _mm(ux_min, ux_max, d001, 4)
            ux_min, ux_max = _mm(ux_min, ux_max, d101, 4)
            ux_min, ux_max = _mm(ux_min, ux_max, d011, 4)
            ux_min, ux_max = _mm(ux_min, ux_max, d111, 4)
            ux_int = clamp(ux_int, ux_min, ux_max)

            uy_min = d000[5]; uy_max = d000[5]
            uy_min, uy_max = _mm(uy_min, uy_max, d100, 5)
            uy_min, uy_max = _mm(uy_min, uy_max, d010, 5)
            uy_min, uy_max = _mm(uy_min, uy_max, d110, 5)
            uy_min, uy_max = _mm(uy_min, uy_max, d001, 5)
            uy_min, uy_max = _mm(uy_min, uy_max, d101, 5)
            uy_min, uy_max = _mm(uy_min, uy_max, d011, 5)
            uy_min, uy_max = _mm(uy_min, uy_max, d111, 5)
            uy_int = clamp(uy_int, uy_min, uy_max)

            uz_min = d000[6]; uz_max = d000[6]
            uz_min, uz_max = _mm(uz_min, uz_max, d100, 6)
            uz_min, uz_max = _mm(uz_min, uz_max, d010, 6)
            uz_min, uz_max = _mm(uz_min, uz_max, d110, 6)
            uz_min, uz_max = _mm(uz_min, uz_max, d001, 6)
            uz_min, uz_max = _mm(uz_min, uz_max, d101, 6)
            uz_min, uz_max = _mm(uz_min, uz_max, d011, 6)
            uz_min, uz_max = _mm(uz_min, uz_max, d111, 6)
            uz_int = clamp(uz_int, uz_min, uz_max)
        end

        # Optional monotone velocity reconstruction: bounds each component by
        # the enclosing parent values. This is a maximum-principle limiter
        # for parent-to-child halo reconstruction; it is disabled by default
        # — a fine halo cell sampled at a velocity extremum would lose cubic
        # and enabled only for AMR checkerboard isolation cases.
        # ρ still clamped (positivity required); T and f_k/f_kb still clamped.
        # The same block still clamps rho, T, and paired populations.
        # field smoothness (O(dx³·∂⁴u/∂x⁴), <1% of u_max in the worst-case
        # wing wake at SR=384). The collision step's α solver and variable_sigma
        # It does not alter Pi_neq or add artificial damping.

        # T bounds (index 7)
        T_min = d000[7]; T_max = d000[7]
        T_min, T_max = _mm(T_min, T_max, d100, 7)
        T_min, T_max = _mm(T_min, T_max, d010, 7)
        T_min, T_max = _mm(T_min, T_max, d110, 7)
        T_min, T_max = _mm(T_min, T_max, d001, 7)
        T_min, T_max = _mm(T_min, T_max, d101, 7)
        T_min, T_max = _mm(T_min, T_max, d011, 7)
        T_min, T_max = _mm(T_min, T_max, d111, 7)
        T_int = clamp(T_int, T_min, T_max)

        # f_k, f_k̄ bounds (indices 1, 2)
        fk_min = d000[1]; fk_max = d000[1]
        fk_min, fk_max = _mm(fk_min, fk_max, d100, 1)
        fk_min, fk_max = _mm(fk_min, fk_max, d010, 1)
        fk_min, fk_max = _mm(fk_min, fk_max, d110, 1)
        fk_min, fk_max = _mm(fk_min, fk_max, d001, 1)
        fk_min, fk_max = _mm(fk_min, fk_max, d101, 1)
        fk_min, fk_max = _mm(fk_min, fk_max, d011, 1)
        fk_min, fk_max = _mm(fk_min, fk_max, d111, 1)
        f_k_int = clamp(f_k_int, fk_min, fk_max)

        fkb_min = d000[2]; fkb_max = d000[2]
        fkb_min, fkb_max = _mm(fkb_min, fkb_max, d100, 2)
        fkb_min, fkb_max = _mm(fkb_min, fkb_max, d010, 2)
        fkb_min, fkb_max = _mm(fkb_min, fkb_max, d110, 2)
        fkb_min, fkb_max = _mm(fkb_min, fkb_max, d001, 2)
        fkb_min, fkb_max = _mm(fkb_min, fkb_max, d101, 2)
        fkb_min, fkb_max = _mm(fkb_min, fkb_max, d011, 2)
        fkb_min, fkb_max = _mm(fkb_min, fkb_max, d111, 2)
        f_kb_int = clamp(f_kb_int, fkb_min, fkb_max)
    end

    # Use the same 13-moment equilibrium path as the collision kernel so the
    # interface interpolation stays consistent with the entropic update.
    ux_rel = ux_int - shift_x
    uy_rel = uy_int - shift_y
    uz_rel = uz_int - shift_z
    # Relaxed Newton tolerance (100× looser than the collision kernel).
    # Tested vs. 1× (collision-kernel default) in the Step-200
    # diagnostic: the tight tolerance slightly WORSENED the seam
    # (8.8e-5 → 9.2e-5), so f_eq accuracy is not the bottleneck.
    # Keeping the fast tolerance for GPU throughput.
    lambda_eq = solve_13mom_lambdas_d3q39(
        rho_int, ux_rel, uy_rel, uz_rel,
        cx_arr_full, cy_arr_full, cz_arr_full, w_arr_full,
        cs2, Int32(80), T_int, shift_x, shift_y, shift_z,
        100.0f0, eos_R_flow          # SOTA A3 — match collision-kernel EOS R
    )
    mass_corr_eq = compute_13mom_mass_correction_d3q39(
        rho_int, lambda_eq,
        cx_arr_full, cy_arr_full, cz_arr_full, w_arr_full,
        cs2, shift_x, shift_y, shift_z
    )

    feq_k = mass_corr_eq * eval_13mom_feq_single(
        rho_int, lambda_eq,
        cx - shift_x, cy - shift_y, cz - shift_z,
        w_k, cs2
    )
    feq_kb = mass_corr_eq * eval_13mom_feq_single(
        rho_int, lambda_eq,
        cx_b - shift_x, cy_b - shift_y, cz_b - shift_z,
        w_kb, cs2
    )

    if near_wall_eq_only == Int32(1)
        return max(feq_k, Float32(1e-10))
    end

    # ── Phase 1.2 — Moment-based temporal interpolation (Lagrava 2012 §3.5) ──
    # When temporal_interpolation is on, ρ, u, T are time-blended via
    # get_blended_pair_39 (returns values at t + temporal_weight·δt_c). The
    # fine-cell populations are then reconstructed from those time-blended
    # moments + the current Π_neq stress projection — NOT by blending raw
    # populations, which would inject non-equilibrium inconsistent with the
    # interpolated macros (Lagrava's explicit warning, §3.5 last paragraph).
    #
    # This replaces the earlier early-return at `use_macro_temporal` (which
    # dropped the stress entirely, leaving pure equilibrium) and the
    # `!use_full_temporal` gate on the Π path. Whenever Π_neq is available,
    # the projection is the right reconstruction regardless of temporal mode.
    # The small remaining O(δt) inaccuracy in Π_neq (not time-blended — we
    # use current Π, not pi_neq_old) is bounded and symmetric; vs. the
    # O(1) penalty of mixing populations across timesteps, this is a huge
    # net win.
    # ISOLATION-REBUILD BASELINE (2026-06-25): reverted to the as-found Π-projection
    # stress carrier (deviatoric, when Π_neq tracked) with the population-mirror
    # fallback. The population-mirror port (always real f_k_int−feq_k) is reverted to
    # re-test in isolation.
    s_k = if has_pi
        if have_cubic_pi
            pxx_int = pxx_cubic
            pyy_int = pyy_cubic
            pzz_int = pzz_cubic
            pxy_int = pxy_cubic
            pxz_int = pxz_cubic
            pyz_int = pyz_cubic
        else
            # 2026-06-26: use the gradient-preserving stress fill (computed above)
            # instead of the renormalizing trilin_pi_39 — removes the box-edge
            # stress bias that mirrors the density wall (adversarially verified).
            pxx_int = pxx_trilin
            pyy_int = pyy_trilin
            pzz_int = pzz_trilin
            pxy_int = pxy_trilin
            pxz_int = pxz_trilin
            pyz_int = pyz_trilin
        end
        project_pi_neq_stress_mode(
            w_k,
            cx - shift_x, cy - shift_y, cz - shift_z,
            cs2,
            pxx_int, pyy_int, pzz_int,
            pxy_int, pxz_int, pyz_int
        )
    else
        f_neq_k  = f_k_int  - feq_k
        f_neq_kb = f_kb_int - feq_kb
        0.5f0 * (f_neq_k + f_neq_kb)
    end
    h_k = 0.5f0 * ((f_k_int - feq_k) - (f_kb_int - feq_kb))

    tau_c = tau_coarse - 0.5f0
    tau_f = tau_fine - 0.5f0
    scale = tau_c > 1.0f-6 ? clamp(tau_f / tau_c, 0.01f0, 100.0f0) : 1.0f0

    eps_pop = Float32(1e-10)
    if s_k < 0.0f0
        denom = max(-s_k, eps_pop)
        theta_k = (feq_k - eps_pop) / denom
        theta_kb = (feq_kb - eps_pop) / denom
        positivity_theta = max(0.0f0, min(theta_k, theta_kb))
        scale = min(scale, positivity_theta)
    end

    # Regularized AMR coupling: when Pi_neq is available the interface
    # transfers conserved moments plus second-order stress only. The mirror
    # h-mode is non-hydrodynamic; carrying it across refinement levels seeds
    # the level-local ringing seen in uniform-flow AMR tests.
    h_scale = has_pi ? 0.0f0 : clamp(interface_h_scale, 0.0f0, 1.0f0)
    return reconstruct_entropic_interface_population(
        feq_k,
        feq_kb,
        s_k,
        h_k,
        scale,
        h_scale,
        eps_pop
    )
end


# ────────────────────────────────────────────────────────────────────
# Thermal DDF interpolation (Phase 2/4b Frapolli)
# Same trilinear + non-equilibrium rescaling as flow, but uses the
# thermal equilibrium.
# Phase 4b: supports Frapolli D3Q39 model (frapolli_mode=1) where
#   g_eq = (Cv - D*cs²/2) * T * f_eq_D3Q39
# Legacy D3Q27 (frapolli_mode=0): g_eq = w_k * T * (1 + 3*c·u)
# ────────────────────────────────────────────────────────────────────
@inline function frapolli_thermal_targets(
    T::Float32, rho_raw::Float32,
    ux::Float32, uy::Float32, uz::Float32,
    Cv::Float32, cs2_thermal::Float32,
    shift_x::Float32, shift_y::Float32, shift_z::Float32,
    eos_R_flow::Float32 = -1.0f0   # SOTA A3: thermodynamic R; ≤0 → legacy cs²
)
    # EOS-CONSISTENT g VARIANCE (2026-06-25): match the flow/thermal collision
    # kernels' coeff_g = Cv − 1.5·eos_R_flow. Sentinel eos_R_flow≤0 ⇒ cs²_thermal
    # fallback keeps the legacy linearized path byte-identical.
    R_g = eos_R_flow > 0.0f0 ? eos_R_flow : cs2_thermal
    coeff_g = Cv - 1.5f0 * R_g
    rho = max(rho_raw, 0.01f0)
    ux_rel = ux - shift_x
    uy_rel = uy - shift_y
    uz_rel = uz - shift_z
    K_eq = coeff_g * T * rho
    qx_eq = K_eq * ux_rel
    qy_eq = K_eq * uy_rel
    qz_eq = K_eq * uz_rel
    return K_eq, qx_eq, qy_eq, qz_eq
end

@inline function frapolli_noneq_moments_from_populations(
    parent_g,
    plx::Int32, ply::Int32, plz::Int32, pb_idx::Int32,
    T::Float32, rho::Float32,
    ux::Float32, uy::Float32, uz::Float32,
    Cv::Float32, cs2_thermal::Float32,
    shift_x::Float32, shift_y::Float32, shift_z::Float32,
    cx_arr_full, cy_arr_full, cz_arr_full,
    eos_R_flow::Float32 = -1.0f0   # SOTA A3: thermodynamic R; ≤0 → legacy cs²
)
    g_sum = 0.0f0
    qx = 0.0f0
    qy = 0.0f0
    qz = 0.0f0
    for i in Int32(1):Int32(39)
        gi = parent_g[plx, ply, plz, pb_idx, i]
        cx_i = Float32(cx_arr_full[i]) - shift_x
        cy_i = Float32(cy_arr_full[i]) - shift_y
        cz_i = Float32(cz_arr_full[i]) - shift_z
        g_sum += gi
        qx += cx_i * gi
        qy += cy_i * gi
        qz += cz_i * gi
    end

    K_eq, qx_eq, qy_eq, qz_eq = frapolli_thermal_targets(
        T, rho, ux, uy, uz,
        Cv, cs2_thermal,
        shift_x, shift_y, shift_z,
        eos_R_flow
    )
    return g_sum - K_eq, qx - qx_eq, qy - qy_eq, qz - qz_eq
end

@inline function rescale_frapolli_kq_mode(
    geq_k::Float32, g_k::Float32, w_k::Float32,
    cx_k::Float32, cy_k::Float32, cz_k::Float32,
    T::Float32,
    ux::Float32, uy::Float32, uz::Float32,
    shift_x::Float32, shift_y::Float32, shift_z::Float32,
    K_neq::Float32, qx_neq::Float32, qy_neq::Float32, qz_neq::Float32,
    scale_f::Float32, scale_g::Float32
)
    _ = (g_k, ux, uy, uz, shift_x, shift_y, shift_z, K_neq, scale_f)

    # Pass106 showed that transferring the scalar K residual and the raw
    # population residual can amplify long-wavelength thermal columns at the
    # AMR boundary. Keep the equilibrium energy from the interpolated macro
    # state and transfer only a small odd heat-flux correction.
    q_transfer_strength = 0.25f0
    max_correction_fraction = 0.12f0
    T_safe = max(T, 0.05f0)
    scale_q = min(scale_g, 1.0f0) * q_transfer_strength
    q_corr = scale_q * w_k * (cx_k*qx_neq + cy_k*qy_neq + cz_k*qz_neq) / T_safe
    corr_cap = max_correction_fraction * max(abs(geq_k), Float32(1e-10))
    q_corr = clamp(q_corr, -corr_cap, corr_cap)
    return max(geq_k + q_corr, Float32(1e-10))
end

@inline function interpolate_thermal_with_rescaling(
    parent_g_new, parent_temperature_new, parent_rho_new, parent_vel_new, parent_kinetic_energy_new,
    parent_g_old, parent_temperature_old, parent_rho_old, parent_vel_old,
    parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    fine_gx::Int32, fine_gy::Int32, fine_gz::Int32,
    k::Int32, k_opp::Int32, block_size::Int32,
    w_k::Float32, cx::Float32, cy::Float32, cz::Float32,
    w_kb::Float32, cx_b::Float32, cy_b::Float32, cz_b::Float32,
    tau_g_coarse::Float32, tau_g_fine::Float32,
    tau_flow_coarse::Float32, tau_flow_fine::Float32,
    temporal_weight::Float32,
    use_temporal_interp::Int32,
    use_macro_temporal_interp::Int32,
    frapolli_mode::Int32 = Int32(0),
    Cv::Float32 = 0.0f0,
    cs2_thermal::Float32 = Float32(1.0 / 3.0),
    shift_x::Float32 = 0.0f0,
    shift_y::Float32 = 0.0f0,
    shift_z::Float32 = 0.0f0,
    cx_arr_full=nothing, cy_arr_full=nothing, cz_arr_full=nothing, w_arr_full=nothing,
    eos_R_flow::Float32 = -1.0f0   # SOTA A3: thermodynamic R; ≤0 → legacy cs²
)
    px_cont = _fine_to_parent_interp_coord(fine_gx)
    py_cont = _fine_to_parent_interp_coord(fine_gy)
    pz_cont = _fine_to_parent_interp_coord(fine_gz)

    parent_nx = parent_dim_x * block_size
    parent_ny = parent_dim_y * block_size
    parent_nz = parent_dim_z * block_size
    px0, px1, wx = _parent_interp_bracket(px_cont, parent_nx)
    py0, py1, wy = _PERIODIC_Y_AMR ? _pbracket_y(py_cont) : _parent_interp_bracket(py_cont, parent_ny)
    pz0, pz1, wz = _parent_interp_bracket(pz_cont, parent_nz)

    shifted_lattice = abs(shift_x) + abs(shift_y) + abs(shift_z) > 1.0f-6
    use_macro_temporal = use_macro_temporal_interp == Int32(1) &&
                         temporal_weight < 0.99f0
    # Match flow prolongation: time-smooth thermal macros and reconstruct the
    # thermal populations instead of blending raw parent populations directly.
    use_full_temporal = use_temporal_interp == Int32(1) &&
                        temporal_weight < 0.99f0 &&
                        !use_macro_temporal
    @inline function get_blended_thermal_pair(pgx::Int32, pgy::Int32, pgz::Int32)
        pbx = (pgx - Int32(1)) ÷ block_size + Int32(1)
        pgy = _pwrap_y(pgy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
        pby = (pgy - Int32(1)) ÷ block_size + Int32(1)
        pbz = (pgz - Int32(1)) ÷ block_size + Int32(1)

        if pbx >= Int32(1) && pbx <= parent_dim_x &&
           pby >= Int32(1) && pby <= parent_dim_y &&
           pbz >= Int32(1) && pbz <= parent_dim_z
            pb_idx = parent_ptr[pbx, pby, pbz]
            if pb_idx > Int32(0)
                plx = (pgx - Int32(1)) % block_size + Int32(1)
                ply = (pgy - Int32(1)) % block_size + Int32(1)
                plz = (pgz - Int32(1)) % block_size + Int32(1)

                g_new = parent_g_new[plx, ply, plz, pb_idx, k]
                gk_new = parent_g_new[plx, ply, plz, pb_idx, k_opp]
                T_new = parent_temperature_new[plx, ply, plz, pb_idx]
                rho_new = parent_rho_new[plx, ply, plz, pb_idx]
                ux_new = parent_vel_new[plx, ply, plz, pb_idx, 1]
                uy_new = parent_vel_new[plx, ply, plz, pb_idx, 2]
                uz_new = parent_vel_new[plx, ply, plz, pb_idx, 3]
                Ekin_new = parent_kinetic_energy_new[plx, ply, plz, pb_idx]
                # Carry Frapolli thermal non-equilibrium diagnostics in moment
                # space. The reconstruction below keeps K equilibrium-only and
                # transfers a capped q correction; pass106 showed that moving
                # raw g_neq/K through this interface imprints thermal columns.
                need_kq_split = frapolli_mode == Int32(1)
                if use_full_temporal
                    g_old = parent_g_old[plx, ply, plz, pb_idx, k]
                    gk_old = parent_g_old[plx, ply, plz, pb_idx, k_opp]
                    T_old = parent_temperature_old[plx, ply, plz, pb_idx]
                    rho_old = parent_rho_old[plx, ply, plz, pb_idx]
                    ux_old = parent_vel_old[plx, ply, plz, pb_idx, 1]
                    uy_old = parent_vel_old[plx, ply, plz, pb_idx, 2]
                    uz_old = parent_vel_old[plx, ply, plz, pb_idx, 3]
                    tw = temporal_weight
                    if need_kq_split
                        K_new, qx_new, qy_new, qz_new = frapolli_noneq_moments_from_populations(
                            parent_g_new,
                            plx, ply, plz, pb_idx,
                            T_new, rho_new, ux_new, uy_new, uz_new,
                            Cv, cs2_thermal, shift_x, shift_y, shift_z,
                            cx_arr_full, cy_arr_full, cz_arr_full,
                            eos_R_flow
                        )
                        K_old, qx_old, qy_old, qz_old = frapolli_noneq_moments_from_populations(
                            parent_g_old,
                            plx, ply, plz, pb_idx,
                            T_old, rho_old, ux_old, uy_old, uz_old,
                            Cv, cs2_thermal, shift_x, shift_y, shift_z,
                            cx_arr_full, cy_arr_full, cz_arr_full,
                            eos_R_flow
                        )
                        return (g_old * (1.0f0 - tw) + g_new * tw,
                                gk_old * (1.0f0 - tw) + gk_new * tw,
                                T_old * (1.0f0 - tw) + T_new * tw,
                                rho_old * (1.0f0 - tw) + rho_new * tw,
                                ux_old * (1.0f0 - tw) + ux_new * tw,
                                uy_old * (1.0f0 - tw) + uy_new * tw,
                                uz_old * (1.0f0 - tw) + uz_new * tw,
                                Ekin_new,
                                K_old * (1.0f0 - tw) + K_new * tw,
                                qx_old * (1.0f0 - tw) + qx_new * tw,
                                qy_old * (1.0f0 - tw) + qy_new * tw,
                                qz_old * (1.0f0 - tw) + qz_new * tw,
                                true)
                    end
                    return (g_old * (1.0f0 - tw) + g_new * tw,
                            gk_old * (1.0f0 - tw) + gk_new * tw,
                            T_old * (1.0f0 - tw) + T_new * tw,
                            rho_old * (1.0f0 - tw) + rho_new * tw,
                            ux_old * (1.0f0 - tw) + ux_new * tw,
                            uy_old * (1.0f0 - tw) + uy_new * tw,
                            uz_old * (1.0f0 - tw) + uz_new * tw,
                            Ekin_new, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
                            true)
                end
                if use_macro_temporal
                    T_old = parent_temperature_old[plx, ply, plz, pb_idx]
                    rho_old = parent_rho_old[plx, ply, plz, pb_idx]
                    ux_old = parent_vel_old[plx, ply, plz, pb_idx, 1]
                    uy_old = parent_vel_old[plx, ply, plz, pb_idx, 2]
                    uz_old = parent_vel_old[plx, ply, plz, pb_idx, 3]
                    tw = temporal_weight
                    if need_kq_split
                        K_new, qx_new, qy_new, qz_new = frapolli_noneq_moments_from_populations(
                            parent_g_new,
                            plx, ply, plz, pb_idx,
                            T_new, rho_new, ux_new, uy_new, uz_new,
                            Cv, cs2_thermal, shift_x, shift_y, shift_z,
                            cx_arr_full, cy_arr_full, cz_arr_full,
                            eos_R_flow
                        )
                        K_old, qx_old, qy_old, qz_old = frapolli_noneq_moments_from_populations(
                            parent_g_old,
                            plx, ply, plz, pb_idx,
                            T_old, rho_old, ux_old, uy_old, uz_old,
                            Cv, cs2_thermal, shift_x, shift_y, shift_z,
                            cx_arr_full, cy_arr_full, cz_arr_full,
                            eos_R_flow
                        )
                        return (g_new, gk_new,
                                T_old * (1.0f0 - tw) + T_new * tw,
                                rho_old * (1.0f0 - tw) + rho_new * tw,
                                ux_old * (1.0f0 - tw) + ux_new * tw,
                                uy_old * (1.0f0 - tw) + uy_new * tw,
                                uz_old * (1.0f0 - tw) + uz_new * tw,
                                Ekin_new,
                                K_old * (1.0f0 - tw) + K_new * tw,
                                qx_old * (1.0f0 - tw) + qx_new * tw,
                                qy_old * (1.0f0 - tw) + qy_new * tw,
                                qz_old * (1.0f0 - tw) + qz_new * tw,
                                true)
                    end
                    return (g_new, gk_new,
                            T_old * (1.0f0 - tw) + T_new * tw,
                            rho_old * (1.0f0 - tw) + rho_new * tw,
                            ux_old * (1.0f0 - tw) + ux_new * tw,
                            uy_old * (1.0f0 - tw) + uy_new * tw,
                            uz_old * (1.0f0 - tw) + uz_new * tw,
                            Ekin_new, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
                            true)
                end
                if need_kq_split
                    K_new, qx_new, qy_new, qz_new = frapolli_noneq_moments_from_populations(
                        parent_g_new,
                        plx, ply, plz, pb_idx,
                        T_new, rho_new, ux_new, uy_new, uz_new,
                        Cv, cs2_thermal, shift_x, shift_y, shift_z,
                        cx_arr_full, cy_arr_full, cz_arr_full,
                        eos_R_flow
                    )
                    return (g_new, gk_new, T_new, rho_new, ux_new, uy_new, uz_new, Ekin_new,
                            K_new, qx_new, qy_new, qz_new, true)
                end

                return (g_new, gk_new, T_new, rho_new, ux_new, uy_new, uz_new, Ekin_new,
                        0.0f0, 0.0f0, 0.0f0, 0.0f0, true)
            end
        end

        return (w_k, w_kb, 1.0f0, 1.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
                0.0f0, 0.0f0, 0.0f0, 0.0f0, false)
    end

    @inline function direct_parent_thermal_valid(pgx::Int32, pgy::Int32, pgz::Int32)
        pbx = (pgx - Int32(1)) ÷ block_size + Int32(1)
        pgy = _pwrap_y(pgy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
        pby = (pgy - Int32(1)) ÷ block_size + Int32(1)
        pbz = (pgz - Int32(1)) ÷ block_size + Int32(1)

        if pbx >= Int32(1) && pbx <= parent_dim_x &&
           pby >= Int32(1) && pby <= parent_dim_y &&
           pbz >= Int32(1) && pbz <= parent_dim_z
            return parent_ptr[pbx, pby, pbz] > Int32(0)
        end

        return false
    end

    # Pass174g: replaced cubic Lagrange with Catmull-Rom cubic Hermite for C¹
    # continuity at parent cell boundaries. See the long comment on
    # _lagrange4_axis_weight above for the derivation. Function name kept as
    # `lagrange4_weights` so call sites and any external references don't
    # break. Returns 4 axis weights for stencil cells {i0-1, i0, i0+1, i0+2}.
    @inline function lagrange4_weights(x::Float32, i0::Int32)
        q = x - Float32(i0)             # fractional position in [0, 1]
        q2 = q * q
        q3 = q2 * q
        w0 = (-q3 + 2.0f0*q2 - q) * 0.5f0          # cell i0-1
        w1 = (3.0f0*q3 - 5.0f0*q2 + 2.0f0) * 0.5f0 # cell i0
        w2 = (-3.0f0*q3 + 4.0f0*q2 + q) * 0.5f0    # cell i0+1
        w3 = (q3 - q2) * 0.5f0                      # cell i0+2
        return (w0, w1, w2, w3)
    end

    @inline function thermal_geq_from_macro(T_int::Float32, rho_raw::Float32,
                                            ux_int::Float32, uy_int::Float32, uz_int::Float32)
        if frapolli_mode == Int32(1)
            rho_i = max(rho_raw, 0.01f0)
            ux_rel_i = ux_int - shift_x
            uy_rel_i = uy_int - shift_y
            uz_rel_i = uz_int - shift_z
            lambda_i = solve_13mom_lambdas_d3q39(
                rho_i, ux_rel_i, uy_rel_i, uz_rel_i,
                cx_arr_full, cy_arr_full, cz_arr_full, w_arr_full,
                cs2_thermal, Int32(200), T_int, shift_x, shift_y, shift_z,
                1.0f0, eos_R_flow
            )
            mass_corr_i = compute_13mom_mass_correction_d3q39(
                rho_i, lambda_i,
                cx_arr_full, cy_arr_full, cz_arr_full, w_arr_full,
                cs2_thermal, shift_x, shift_y, shift_z
            )
            f_eq_i = mass_corr_i * eval_13mom_feq_single(
                rho_i, lambda_i,
                cx - shift_x, cy - shift_y, cz - shift_z,
                w_k, cs2_thermal
            )
            R_g_i = eos_R_flow > 0.0f0 ? eos_R_flow : cs2_thermal
            coeff_g_i = Cv - 1.5f0 * R_g_i
            return max(coeff_g_i * T_int * f_eq_i, Float32(1e-10))
        end

        return max(calculate_thermal_equilibrium(T_int, ux_int, uy_int, uz_int,
                                                 w_k, cx, cy, cz), Float32(1e-10))
    end

    use_lagrange4 =
        px0 - Int32(1) >= Int32(1) && px0 + Int32(2) <= parent_nx &&
        (_PERIODIC_Y_AMR ? parent_ny >= Int32(4) :   # periodic-Y AMR: lookups wrap
         (py0 - Int32(1) >= Int32(1) && py0 + Int32(2) <= parent_ny)) &&
        pz0 - Int32(1) >= Int32(1) && pz0 + Int32(2) <= parent_nz

    lagrange4_all_direct_valid = use_lagrange4
    if lagrange4_all_direct_valid
        for iz in Int32(0):Int32(3), iy in Int32(0):Int32(3), ix in Int32(0):Int32(3)
            pgx = px0 - Int32(1) + ix
            pgy = py0 - Int32(1) + iy
            pgz = pz0 - Int32(1) + iz
            if !direct_parent_thermal_valid(pgx, pgy, pgz)
                lagrange4_all_direct_valid = false
            end
        end
    end

    if lagrange4_all_direct_valid
        wxs = lagrange4_weights(px_cont, px0)
        wys = lagrange4_weights(py_cont, py0)
        wzs = lagrange4_weights(pz_cont, pz0)

        g_int = 0.0f0
        gk_int = 0.0f0
        T_int = 0.0f0
        rho_int = 0.0f0
        ux_int = 0.0f0
        uy_int = 0.0f0
        uz_int = 0.0f0
        Ekin_int = 0.0f0
        Kneq_int = 0.0f0
        qxneq_int = 0.0f0
        qyneq_int = 0.0f0
        qzneq_int = 0.0f0
        all_valid = true

        for iz in Int32(0):Int32(3)
            pz = pz0 - Int32(1) + iz
            wzv = iz == Int32(0) ? wzs[1] : (iz == Int32(1) ? wzs[2] : (iz == Int32(2) ? wzs[3] : wzs[4]))
            for iy in Int32(0):Int32(3)
                py = py0 - Int32(1) + iy
                wyv = iy == Int32(0) ? wys[1] : (iy == Int32(1) ? wys[2] : (iy == Int32(2) ? wys[3] : wys[4]))
                for ix in Int32(0):Int32(3)
                    px = px0 - Int32(1) + ix
                    wxv = ix == Int32(0) ? wxs[1] : (ix == Int32(1) ? wxs[2] : (ix == Int32(2) ? wxs[3] : wxs[4]))
                    d = get_blended_thermal_pair(px, py, pz)
                    if d[13]
                        wxyz = wxv * wyv * wzv
                        g_int += wxyz * d[1]
                        gk_int += wxyz * d[2]
                        T_int += wxyz * d[3]
                        rho_int += wxyz * d[4]
                        ux_int += wxyz * d[5]
                        uy_int += wxyz * d[6]
                        uz_int += wxyz * d[7]
                        Ekin_int += wxyz * d[8]
                        Kneq_int += wxyz * d[9]
                        qxneq_int += wxyz * d[10]
                        qyneq_int += wxyz * d[11]
                        qzneq_int += wxyz * d[12]
                    else
                        all_valid = false
                    end
                end
            end
        end

        if all_valid
            if frapolli_mode == Int32(1)
                # EOS-CONSISTENT g VARIANCE (2026-06-25): mirror the flow/thermal
                # collision kernels' coeff_g = Cv − 1.5·eos_R_flow, and feed the
                # SAME eos_R into the 13-mom λ solve so the interpolated thermal
                # f_eq carrier matches the collision EOS. Sentinel eos_R_flow≤0 ⇒
                # cs²_thermal fallback keeps the legacy linearized path identical
                # (the explicit 1.0f0 tol_multiplier reproduces the prior default).
                R_g = eos_R_flow > 0.0f0 ? eos_R_flow : cs2_thermal
                coeff_g = Cv - 1.5f0 * R_g
                ux_rel = ux_int - shift_x
                uy_rel = uy_int - shift_y
                uz_rel = uz_int - shift_z
                rho_int = max(rho_int, 0.01f0)
                lambda_eq = solve_13mom_lambdas_d3q39(
                    rho_int, ux_rel, uy_rel, uz_rel,
                    cx_arr_full, cy_arr_full, cz_arr_full, w_arr_full,
                    cs2_thermal, Int32(200), T_int, shift_x, shift_y, shift_z,
                    1.0f0, eos_R_flow
                )
                mass_corr_eq = compute_13mom_mass_correction_d3q39(
                    rho_int, lambda_eq,
                    cx_arr_full, cy_arr_full, cz_arr_full, w_arr_full,
                    cs2_thermal, shift_x, shift_y, shift_z
                )
                f_eq_int = mass_corr_eq * eval_13mom_feq_single(
                    rho_int, lambda_eq,
                    cx - shift_x, cy - shift_y, cz - shift_z,
                    w_k, cs2_thermal
                )
                geq_int = coeff_g * T_int * f_eq_int
                f_eq_int_opp = mass_corr_eq * eval_13mom_feq_single(
                    rho_int, lambda_eq,
                    cx_b - shift_x, cy_b - shift_y, cz_b - shift_z,
                    w_kb, cs2_thermal
                )
                geq_int_opp = coeff_g * T_int * f_eq_int_opp
            else
                geq_int = calculate_thermal_equilibrium(T_int, ux_int, uy_int, uz_int, w_k, cx, cy, cz)
                geq_int_opp = calculate_thermal_equilibrium(T_int, ux_int, uy_int, uz_int, w_kb, cx_b, cy_b, cz_b)
            end

            g_neq = g_int - geq_int
            gk_neq = gk_int - geq_int_opp
            tau_gc = tau_g_coarse - 0.5f0
            tau_gf = tau_g_fine - 0.5f0
            scale_g = tau_gc > 1.0f-6 ? clamp(tau_gf / tau_gc, 0.01f0, 100.0f0) : 1.0f0
            tau_fc = tau_flow_coarse - 0.5f0
            tau_ff = tau_flow_fine - 0.5f0
            scale_f = tau_fc > 1.0f-6 ? clamp(tau_ff / tau_fc, 0.01f0, 100.0f0) : 1.0f0
            if shifted_lattice
                if frapolli_mode == Int32(1)
                    return rescale_frapolli_kq_mode(
                        geq_int, g_int, w_k,
                        cx - shift_x, cy - shift_y, cz - shift_z,
                        T_int, ux_int, uy_int, uz_int,
                        shift_x, shift_y, shift_z,
                        Kneq_int, qxneq_int, qyneq_int, qzneq_int,
                        scale_f, scale_g
                    )
                end
                odd_pair = 0.5f0 * (g_neq - gk_neq)
                odd_scaled = limit_pair_odd_mode(odd_pair * scale_g, geq_int, geq_int_opp, 0.8f0)
                return max(geq_int + odd_scaled, Float32(1e-10))
            end

            if frapolli_mode == Int32(1)
                return rescale_frapolli_kq_mode(
                    geq_int, g_int, w_k,
                    cx - shift_x, cy - shift_y, cz - shift_z,
                    T_int, ux_int, uy_int, uz_int,
                    shift_x, shift_y, shift_z,
                    Kneq_int, qxneq_int, qyneq_int, qzneq_int,
                    scale_f, scale_g
                )
            end

            odd_pair = 0.5f0 * (g_neq - gk_neq)
            odd_scaled = limit_pair_odd_mode(odd_pair * scale_g, geq_int, geq_int_opp, 0.8f0)
            return max(geq_int + odd_scaled, Float32(1e-10))
        end
    end

    # Match the flow path: if exactly one parent-stencil direction is support
    # limited, keep cubic in the two supported axes and linear only in the
    # limited axis. When more than one such tensor stencil is valid, average the
    # candidates so the fallback has no x/y/z precedence.
    # Pass174g: Catmull-Rom cubic Hermite (C¹ at parent cell boundaries),
    # matching the entropic D3Q39 flow path _lagrange4_axis_weight above.
    # Same 4-point stencil, same cost; C⁰ → C¹ globally. Keeps the function
    # name for call-site compatibility.
    @inline function _thermal_lagrange4_axis_weight(q::Float32, offset::Int32)
        q2 = q * q
        q3 = q2 * q
        return offset == Int32(-1) ? ((-q3 + 2.0f0*q2 - q) * 0.5f0) :
               offset == Int32(0)  ? ((3.0f0*q3 - 5.0f0*q2 + 2.0f0) * 0.5f0) :
               offset == Int32(1)  ? ((-3.0f0*q3 + 4.0f0*q2 + q) * 0.5f0) :
                                      ((q3 - q2) * 0.5f0)
    end

    @inline function _thermal_tensor_axis_bounds_ok(linear_axis::Int32)
        x_ok = linear_axis == Int32(1) ?
               (px0 >= Int32(1) && px1 <= parent_nx) :
               (px0 - Int32(1) >= Int32(1) && px0 + Int32(2) <= parent_nx)
        y_ok = _PERIODIC_Y_AMR ? (parent_ny >= Int32(4)) :   # periodic-Y AMR: lookups wrap
               linear_axis == Int32(2) ?
               (py0 >= Int32(1) && py1 <= parent_ny) :
               (py0 - Int32(1) >= Int32(1) && py0 + Int32(2) <= parent_ny)
        z_ok = linear_axis == Int32(3) ?
               (pz0 >= Int32(1) && pz1 <= parent_nz) :
               (pz0 - Int32(1) >= Int32(1) && pz0 + Int32(2) <= parent_nz)
        return x_ok && y_ok && z_ok
    end

    @inline function _thermal_tensor_offset_active(offset::Int32, linear_axis::Int32, axis::Int32)
        return axis == linear_axis ? (offset == Int32(0) || offset == Int32(1)) : true
    end

    @inline function _thermal_tensor_axis_weight(q::Float32, offset::Int32,
                                                 linear_axis::Int32, axis::Int32)
        return axis == linear_axis ?
               (offset == Int32(0) ? (1.0f0 - q) : q) :
               _thermal_lagrange4_axis_weight(q, offset)
    end

    thermal_tensor_axis_count = Int32(0)
    g_tensor_sum = 0.0f0
    gk_tensor_sum = 0.0f0
    T_tensor_sum = 0.0f0
    rho_tensor_sum = 0.0f0
    ux_tensor_sum = 0.0f0
    uy_tensor_sum = 0.0f0
    uz_tensor_sum = 0.0f0
    Ekin_tensor_sum = 0.0f0
    Kneq_tensor_sum = 0.0f0
    qxneq_tensor_sum = 0.0f0
    qyneq_tensor_sum = 0.0f0
    qzneq_tensor_sum = 0.0f0

    if !lagrange4_all_direct_valid
        @inbounds for linear_axis in Int32(1):Int32(3)
            _thermal_tensor_axis_bounds_ok(linear_axis) || continue
            axis_valid = true
            g_axis = 0.0f0
            gk_axis = 0.0f0
            T_axis = 0.0f0
            rho_axis = 0.0f0
            ux_axis = 0.0f0
            uy_axis = 0.0f0
            uz_axis = 0.0f0
            Ekin_axis = 0.0f0
            Kneq_axis = 0.0f0
            qxneq_axis = 0.0f0
            qyneq_axis = 0.0f0
            qzneq_axis = 0.0f0

            for ix in Int32(-1):Int32(2)
                _thermal_tensor_offset_active(ix, linear_axis, Int32(1)) || continue
                px = px0 + ix
                wxv = _thermal_tensor_axis_weight(wx, ix, linear_axis, Int32(1))
                for iy in Int32(-1):Int32(2)
                    _thermal_tensor_offset_active(iy, linear_axis, Int32(2)) || continue
                    py = py0 + iy
                    wyv = _thermal_tensor_axis_weight(wy, iy, linear_axis, Int32(2))
                    for iz in Int32(-1):Int32(2)
                        _thermal_tensor_offset_active(iz, linear_axis, Int32(3)) || continue
                        pz = pz0 + iz
                        wzv = _thermal_tensor_axis_weight(wz, iz, linear_axis, Int32(3))
                        d = get_blended_thermal_pair(px, py, pz)
                        if d[13]
                            wxyz = wxv * wyv * wzv
                            g_axis += wxyz * d[1]
                            gk_axis += wxyz * d[2]
                            T_axis += wxyz * d[3]
                            rho_axis += wxyz * d[4]
                            ux_axis += wxyz * d[5]
                            uy_axis += wxyz * d[6]
                            uz_axis += wxyz * d[7]
                            Ekin_axis += wxyz * d[8]
                            Kneq_axis += wxyz * d[9]
                            qxneq_axis += wxyz * d[10]
                            qyneq_axis += wxyz * d[11]
                            qzneq_axis += wxyz * d[12]
                        else
                            axis_valid = false
                        end
                    end
                end
            end

            if axis_valid
                thermal_tensor_axis_count += Int32(1)
                g_tensor_sum += g_axis
                gk_tensor_sum += gk_axis
                T_tensor_sum += T_axis
                rho_tensor_sum += rho_axis
                ux_tensor_sum += ux_axis
                uy_tensor_sum += uy_axis
                uz_tensor_sum += uz_axis
                Ekin_tensor_sum += Ekin_axis
                Kneq_tensor_sum += Kneq_axis
                qxneq_tensor_sum += qxneq_axis
                qyneq_tensor_sum += qyneq_axis
                qzneq_tensor_sum += qzneq_axis
            end
        end
    end

    if thermal_tensor_axis_count > Int32(0)
        inv_tensor_axes = 1.0f0 / Float32(thermal_tensor_axis_count)
        g_int = g_tensor_sum * inv_tensor_axes
        gk_int = gk_tensor_sum * inv_tensor_axes
        T_int = T_tensor_sum * inv_tensor_axes
        rho_int = rho_tensor_sum * inv_tensor_axes
        ux_int = ux_tensor_sum * inv_tensor_axes
        uy_int = uy_tensor_sum * inv_tensor_axes
        uz_int = uz_tensor_sum * inv_tensor_axes
        Ekin_int = Ekin_tensor_sum * inv_tensor_axes
        Kneq_int = Kneq_tensor_sum * inv_tensor_axes
        qxneq_int = qxneq_tensor_sum * inv_tensor_axes
        qyneq_int = qyneq_tensor_sum * inv_tensor_axes
        qzneq_int = qzneq_tensor_sum * inv_tensor_axes

        if frapolli_mode == Int32(1)
            R_g = eos_R_flow > 0.0f0 ? eos_R_flow : cs2_thermal
            coeff_g = Cv - 1.5f0 * R_g
            ux_rel = ux_int - shift_x
            uy_rel = uy_int - shift_y
            uz_rel = uz_int - shift_z
            rho_int = max(rho_int, 0.01f0)
            lambda_eq = solve_13mom_lambdas_d3q39(
                rho_int, ux_rel, uy_rel, uz_rel,
                cx_arr_full, cy_arr_full, cz_arr_full, w_arr_full,
                cs2_thermal, Int32(200), T_int, shift_x, shift_y, shift_z,
                1.0f0, eos_R_flow
            )
            mass_corr_eq = compute_13mom_mass_correction_d3q39(
                rho_int, lambda_eq,
                cx_arr_full, cy_arr_full, cz_arr_full, w_arr_full,
                cs2_thermal, shift_x, shift_y, shift_z
            )
            f_eq_int = mass_corr_eq * eval_13mom_feq_single(
                rho_int, lambda_eq,
                cx - shift_x, cy - shift_y, cz - shift_z,
                w_k, cs2_thermal
            )
            geq_int = coeff_g * T_int * f_eq_int
            tau_gc = tau_g_coarse - 0.5f0
            tau_gf = tau_g_fine - 0.5f0
            scale_g = tau_gc > 1.0f-6 ? clamp(tau_gf / tau_gc, 0.01f0, 100.0f0) : 1.0f0
            tau_fc = tau_flow_coarse - 0.5f0
            tau_ff = tau_flow_fine - 0.5f0
            scale_f = tau_fc > 1.0f-6 ? clamp(tau_ff / tau_fc, 0.01f0, 100.0f0) : 1.0f0
            return rescale_frapolli_kq_mode(
                geq_int, g_int, w_k,
                cx - shift_x, cy - shift_y, cz - shift_z,
                T_int, ux_int, uy_int, uz_int,
                shift_x, shift_y, shift_z,
                Kneq_int, qxneq_int, qyneq_int, qzneq_int,
                scale_f, scale_g
            )
        end

        geq_int = calculate_thermal_equilibrium(T_int, ux_int, uy_int, uz_int, w_k, cx, cy, cz)
        geq_opp = calculate_thermal_equilibrium(T_int, ux_int, uy_int, uz_int, w_kb, cx_b, cy_b, cz_b)
        if use_macro_temporal
            return max(geq_int, Float32(1e-10))
        end
        tau_gc = tau_g_coarse - 0.5f0
        tau_gf = tau_g_fine - 0.5f0
        scale_g = tau_gc > 1.0f-6 ? clamp(tau_gf / tau_gc, 0.01f0, 100.0f0) : 1.0f0
        odd_pair = 0.5f0 * ((g_int - geq_int) - (gk_int - geq_opp))
        odd_scaled = limit_pair_odd_mode(odd_pair * scale_g, geq_int, geq_opp, 0.8f0)
        return max(geq_int + odd_scaled, Float32(1e-10))
    end

    d000 = get_blended_thermal_pair(px0, py0, pz0)
    d100 = get_blended_thermal_pair(px1, py0, pz0)
    d010 = get_blended_thermal_pair(px0, py1, pz0)
    d110 = get_blended_thermal_pair(px1, py1, pz0)
    d001 = get_blended_thermal_pair(px0, py0, pz1)
    d101 = get_blended_thermal_pair(px1, py0, pz1)
    d011 = get_blended_thermal_pair(px0, py1, pz1)
    d111 = get_blended_thermal_pair(px1, py1, pz1)

    w000 = (1.0f0 - wx) * (1.0f0 - wy) * (1.0f0 - wz)
    w100 = wx * (1.0f0 - wy) * (1.0f0 - wz)
    w010 = (1.0f0 - wx) * wy * (1.0f0 - wz)
    w110 = wx * wy * (1.0f0 - wz)
    w001 = (1.0f0 - wx) * (1.0f0 - wy) * wz
    w101 = wx * (1.0f0 - wy) * wz
    w011 = (1.0f0 - wx) * wy * wz
    w111 = wx * wy * wz

    @inline function trilin_t(i)
        accum = 0.0f0
        wsum = 0.0f0
        if d000[13]; accum += w000 * d000[i]; wsum += w000; end
        if d100[13]; accum += w100 * d100[i]; wsum += w100; end
        if d010[13]; accum += w010 * d010[i]; wsum += w010; end
        if d110[13]; accum += w110 * d110[i]; wsum += w110; end
        if d001[13]; accum += w001 * d001[i]; wsum += w001; end
        if d101[13]; accum += w101 * d101[i]; wsum += w101; end
        if d011[13]; accum += w011 * d011[i]; wsum += w011; end
        if d111[13]; accum += w111 * d111[i]; wsum += w111; end
        if wsum > Float32(1e-12)
            return accum / wsum
        end
        return d000[i]
    end

    g_int = trilin_t(1)
    gk_int = trilin_t(2)
    T_int = trilin_t(3)
    rho_int = trilin_t(4)
    ux_int = trilin_t(5)
    uy_int = trilin_t(6)
    uz_int = trilin_t(7)
    Ekin_int = trilin_t(8)
    Kneq_int = trilin_t(9)
    qxneq_int = trilin_t(10)
    qyneq_int = trilin_t(11)
    qzneq_int = trilin_t(12)

    if frapolli_mode == Int32(1)
        R_g = eos_R_flow > 0.0f0 ? eos_R_flow : cs2_thermal
        coeff_g = Cv - 1.5f0 * R_g
        ux_rel = ux_int - shift_x
        uy_rel = uy_int - shift_y
        uz_rel = uz_int - shift_z
        rho_int = max(rho_int, 0.01f0)
        lambda_eq = solve_13mom_lambdas_d3q39(
            rho_int, ux_rel, uy_rel, uz_rel,
            cx_arr_full, cy_arr_full, cz_arr_full, w_arr_full,
            cs2_thermal, Int32(200), T_int, shift_x, shift_y, shift_z,
            1.0f0, eos_R_flow
        )
        mass_corr_eq = compute_13mom_mass_correction_d3q39(
            rho_int, lambda_eq,
            cx_arr_full, cy_arr_full, cz_arr_full, w_arr_full,
            cs2_thermal, shift_x, shift_y, shift_z
        )
        f_eq_int = mass_corr_eq * eval_13mom_feq_single(
            rho_int, lambda_eq,
            cx - shift_x, cy - shift_y, cz - shift_z,
            w_k, cs2_thermal
        )
        geq_int = coeff_g * T_int * f_eq_int
        f_eq_int_opp = mass_corr_eq * eval_13mom_feq_single(
            rho_int, lambda_eq,
            cx_b - shift_x, cy_b - shift_y, cz_b - shift_z,
            w_kb, cs2_thermal
        )
        geq_int_opp = coeff_g * T_int * f_eq_int_opp
    else
        geq_int = calculate_thermal_equilibrium(T_int, ux_int, uy_int, uz_int, w_k, cx, cy, cz)
        geq_int_opp = calculate_thermal_equilibrium(T_int, ux_int, uy_int, uz_int, w_kb, cx_b, cy_b, cz_b)
    end

    # Same rationale as flow interpolation above: with macro-only temporal
    # blending, the thermal populations are not time-interpolated consistently
    # with T/rho/u. Use the equilibrium reconstructed from the smoothed
    # macroscopic state to avoid interface heat-flux ringing.
    if use_macro_temporal
        return max(geq_int, Float32(1e-10))
    end

    g_neq = g_int - geq_int
    gk_neq = gk_int - geq_int_opp
    tau_gc = tau_g_coarse - 0.5f0
    tau_gf = tau_g_fine - 0.5f0
    scale_g = tau_gc > 1.0f-6 ? clamp(tau_gf / tau_gc, 0.01f0, 100.0f0) : 1.0f0
    tau_fc = tau_flow_coarse - 0.5f0
    tau_ff = tau_flow_fine - 0.5f0
    scale_f = tau_fc > 1.0f-6 ? clamp(tau_ff / tau_fc, 0.01f0, 100.0f0) : 1.0f0
    if shifted_lattice
        if frapolli_mode == Int32(1)
            return rescale_frapolli_kq_mode(
                geq_int, g_int, w_k,
                cx - shift_x, cy - shift_y, cz - shift_z,
                T_int, ux_int, uy_int, uz_int,
                shift_x, shift_y, shift_z,
                Kneq_int, qxneq_int, qyneq_int, qzneq_int,
                scale_f, scale_g
            )
        end
        odd_pair = 0.5f0 * (g_neq - gk_neq)
        odd_scaled = limit_pair_odd_mode(odd_pair * scale_g, geq_int, geq_int_opp, 0.8f0)
        return max(geq_int + odd_scaled, Float32(1e-10))
    end

    if frapolli_mode == Int32(1)
        return rescale_frapolli_kq_mode(
            geq_int, g_int, w_k,
            cx - shift_x, cy - shift_y, cz - shift_z,
            T_int, ux_int, uy_int, uz_int,
            shift_x, shift_y, shift_z,
            Kneq_int, qxneq_int, qyneq_int, qzneq_int,
            scale_f, scale_g
        )
    end

    odd_pair = 0.5f0 * (g_neq - gk_neq)
    odd_scaled = limit_pair_odd_mode(odd_pair * scale_g, geq_int, geq_int_opp, 0.8f0)
    return max(geq_int + odd_scaled, Float32(1e-10))
end
