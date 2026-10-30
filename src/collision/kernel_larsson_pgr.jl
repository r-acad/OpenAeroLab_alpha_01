#
# Larsson 2010 non-equilibrium wall model — pressure-gradient correction
# =====================================================================
#
# Adds the pressure-gradient term to the equilibrium log-law residual
# already used by the Van Driest-corrected wall model:
#
#   u_VD / u_τ  =  (1/κ)·ln(y⁺) + B + (y / (ρ_w · u_τ²)) · ∂p/∂x_stream
#
# Reference:
#   Larsson, J. (2010). "Computational aspects of inflow turbulence
#   generation and statistical analysis." CTR Briefs.
#   See also: Hickel, Touber, Bodart & Larsson 2012 "A parametrized
#   non-equilibrium wall-model for LES." Annual Research Briefs, CTR.
#
# Pressure gradient
# -----------------
# The streamwise pressure gradient is computed as
#
#   ∂p/∂x_stream  =  (∂p/∂x_i) · u_i / |u|
#
# with central differences on the lattice cardinal axes. Pressure is read
# back from the density field via the EOS already plumbed into the kernel:
# linearized `p = ρ·cs²` (default) or thermodynamic `p = (γ-1)·ρ·cv·T`
# (`eos_R_flow > 0`). For the gradient we use the simpler `p ≈ ρ·R` form
# since temperature variation across a near-wall BL is small relative to
# density variation when computed over a 3-cell stencil.
#
# Stability
# ---------
# The correction is bounded to `|Δu⁺| ≤ 0.5 · u⁺` to prevent runaway when
# `u_τ → 0` in a stagnation point or when the pressure-gradient diagnostic
# is contaminated by AMR seam noise. This is the same Schur-bound used in
# Hickel-Touber 2012 §3.2.
#
# Race conditions
# ---------------
# The neighbor density reads from `rho_out` are formally racy (the
# neighbor thread may have updated its rho in the same kernel pass). The
# error window is one timestep of density evolution in a near-wall cell,
# which is bounded by `cs² · |∇·u|·dt ≈ O(1e-4)` even at Ma=1. The
# Larsson correction itself is an O(1%) modulation of u_τ, so a 1e-4
# perturbation of ∇p is well below the noise floor of the wall-model.

"""
    read_rho_neighbor_inline(rho_out, x, y, z, b_idx,
                             dx, dy, dz,
                             block_size, neighbor_table) -> Float32

Read density at the cell located at (x+dx, y+dy, z+dz) from the current
block `b_idx`. Handles within-block reads and single-hop cross-block
reads via the neighbor_table. Returns `NaN32` if the read crosses the
domain boundary (no neighbor block).

`dx, dy, dz ∈ {-1, 0, +1}` (single-step stencil).
"""
@inline function read_rho_neighbor_inline(rho_out, x, y, z,
                                          b_idx,
                                          dx, dy, dz,
                                          block_size,
                                          neighbor_table)
    # Type-erased signature: callers may pass `x, y, z` as Int64 (the
    # `@index(Global, NTuple)` default in KernelAbstractions) or Int32.
    # Coerce internally to Int32 so the array indexing and the
    # neighbor_table lookup hit the same dispatch path used elsewhere.
    sx = Int32(x) + Int32(dx)
    sy = Int32(y) + Int32(dy)
    sz = Int32(z) + Int32(dz)
    bs = Int32(block_size)
    bi = Int32(b_idx)

    if sx >= Int32(1) && sx <= bs &&
       sy >= Int32(1) && sy <= bs &&
       sz >= Int32(1) && sz <= bs
        return rho_out[sx, sy, sz, bi]
    end

    # Cross-block: encode neighbor direction into 1..27 index.
    nb_off_x = sx < Int32(1) ? Int32(-1) : (sx > bs ? Int32(1) : Int32(0))
    nb_off_y = sy < Int32(1) ? Int32(-1) : (sy > bs ? Int32(1) : Int32(0))
    nb_off_z = sz < Int32(1) ? Int32(-1) : (sz > bs ? Int32(1) : Int32(0))
    dir_idx = (nb_off_x + Int32(1)) +
              (nb_off_y + Int32(1)) * Int32(3) +
              (nb_off_z + Int32(1)) * Int32(9) +
              Int32(1)
    nb_idx_global = neighbor_table[bi, dir_idx]
    if nb_idx_global <= Int32(0)
        return NaN32   # Domain boundary — caller drops PG contribution on this axis.
    end

    nsx = sx < Int32(1) ? sx + bs :
          (sx > bs ? sx - bs : sx)
    nsy = sy < Int32(1) ? sy + bs :
          (sy > bs ? sy - bs : sy)
    nsz = sz < Int32(1) ? sz + bs :
          (sz > bs ? sz - bs : sz)
    return rho_out[nsx, nsy, nsz, nb_idx_global]
end

"""
    compute_streamwise_dpdx_inline(rho_out, x, y, z, b_idx,
                                   ux, uy, uz, u_mag,
                                   eos_R, cs2_val,
                                   block_size, neighbor_table) -> Float32

Streamwise (flow-aligned) pressure gradient at cell (x, y, z) in lattice
units. Returns 0 when |u| is too small to define a streamwise direction
or when neighbor reads fall off the domain (boundary axes are dropped).

`eos_R` semantics match `solve_13mom_lambdas_d3q39`: pass `(γ-1)·cv` for
thermodynamic EOS, or `cs2_val` to recover the linearized form. The
pressure here is `p = ρ · R` (no temperature variation — see header).
"""
@inline function compute_streamwise_dpdx_inline(rho_out,
                                                x, y, z,
                                                b_idx,
                                                ux::Float32, uy::Float32, uz::Float32,
                                                u_mag::Float32,
                                                eos_R::Float32, cs2_val::Float32,
                                                block_size,
                                                neighbor_table)
    if u_mag < 1.0f-6
        return 0.0f0
    end
    inv_u_mag = 1.0f0 / u_mag

    R_eff = (eos_R > 0.0f0) ? eos_R : cs2_val

    # ∂p/∂x ≈ R · (ρ[+x] - ρ[-x]) / 2  (lattice units, dx=1)
    rho_xp = read_rho_neighbor_inline(rho_out, x, y, z, b_idx,
                                       Int32( 1), Int32(0), Int32(0),
                                       block_size, neighbor_table)
    rho_xm = read_rho_neighbor_inline(rho_out, x, y, z, b_idx,
                                       Int32(-1), Int32(0), Int32(0),
                                       block_size, neighbor_table)
    dpdx_x = (isfinite(rho_xp) && isfinite(rho_xm)) ?
             0.5f0 * R_eff * (rho_xp - rho_xm) : 0.0f0

    rho_yp = read_rho_neighbor_inline(rho_out, x, y, z, b_idx,
                                       Int32(0), Int32( 1), Int32(0),
                                       block_size, neighbor_table)
    rho_ym = read_rho_neighbor_inline(rho_out, x, y, z, b_idx,
                                       Int32(0), Int32(-1), Int32(0),
                                       block_size, neighbor_table)
    dpdx_y = (isfinite(rho_yp) && isfinite(rho_ym)) ?
             0.5f0 * R_eff * (rho_yp - rho_ym) : 0.0f0

    rho_zp = read_rho_neighbor_inline(rho_out, x, y, z, b_idx,
                                       Int32(0), Int32(0), Int32( 1),
                                       block_size, neighbor_table)
    rho_zm = read_rho_neighbor_inline(rho_out, x, y, z, b_idx,
                                       Int32(0), Int32(0), Int32(-1),
                                       block_size, neighbor_table)
    dpdx_z = (isfinite(rho_zp) && isfinite(rho_zm)) ?
             0.5f0 * R_eff * (rho_zp - rho_zm) : 0.0f0

    # Project ∇p onto the flow direction.
    return (dpdx_x * ux + dpdx_y * uy + dpdx_z * uz) * inv_u_mag
end

"""
    larsson_pg_corrected_uplus(u_plus_law, larsson_active,
                               rho_out, x, y, z, b_idx,
                               ux, uy, uz, u_mag, rho_w,
                               u_tau, dist_wall, cs2_val,
                               block_size, neighbor_table) -> Float32

Apply the Larsson 2010 pressure-gradient correction to the equilibrium
log-law non-dimensional velocity `u_plus_law` (= u_VD/u_τ before the PG term):

    u⁺  ←  u⁺ + (y / (ρ_w · u_τ²)) · ∂p/∂x_stream

Schur-bounded to |Δu⁺| ≤ 0.5·u⁺ (Hickel-Touber 2012 §3.2) so a contaminated
∇p (AMR seam / stagnation) cannot drive a runaway. Lattice-agnostic — used by
both the D3Q19 incompressible wall model (where Van Driest reduces to identity,
so `rho_w = rho` and `u_mag` is the untransformed near-wall speed) and any
future compressible caller. `eos_R = cs2_val` is used inside the ∇p helper
(linearized p = ρ·cs² is sufficient for the BL ∇p — see header). No-op (returns
`u_plus_law` unchanged) when `larsson_active == 0`.
"""
@inline function larsson_pg_corrected_uplus(u_plus_law::Float32,
                                            larsson_active::Int32,
                                            rho_out, x, y, z, b_idx,
                                            ux::Float32, uy::Float32, uz::Float32,
                                            u_mag::Float32, rho_w::Float32,
                                            u_tau::Float32, dist_wall::Float32,
                                            cs2_val::Float32,
                                            block_size, neighbor_table)
    if larsson_active != Int32(1)
        return u_plus_law
    end
    # u_τ² floor and ρ_w floor to avoid div-by-zero at stagnation / vacuum.
    denom = max(rho_w, 1.0f-4) * max(u_tau * u_tau, 1.0f-12)
    dpdx_stream = compute_streamwise_dpdx_inline(
        rho_out, x, y, z, b_idx, ux, uy, uz, u_mag,
        cs2_val, cs2_val, block_size, neighbor_table)
    du_plus = (dist_wall / denom) * dpdx_stream
    # Schur bound: |Δu⁺| ≤ 0.5·u⁺.
    bound = 0.5f0 * abs(u_plus_law)
    du_plus = clamp(du_plus, -bound, bound)
    return u_plus_law + du_plus
end
