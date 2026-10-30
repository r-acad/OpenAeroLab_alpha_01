# ==============================================================================
# SPECTRAL VANISHING VISCOSITY (SVV) — Y-Z plane bi-Laplacian for D3Q39
# ==============================================================================
# Pure spatial high-wavenumber damping per:
#   Tadmor 1989, SIAM J. Numer. Anal. 26, 30 (SVV theory, original)
#   Karamanos & Karniadakis 2000, J. Comput. Phys. 163, 22 (SVV for LES)
#   Pasquetti 2006, J. Sci. Comput. 27, 365 (SVV cutoff design)
#   Lallemand & Luo 2000, Phys. Rev. E 61, 6546 (LBM generalized hydrodynamics —
#     theoretical justification for damping high-k ghost modes)
#
# This routine addresses spatial Nyquist modes (e.g., the (-1)^(y+z) checkerboard
# in the wake of ONERA M6 cases) that have NONZERO macroscopic Hermite moments
# and therefore CANNOT be annihilated by Hermite projection / RR-LBM. They live
# IN the physical 35-dimensional subspace, just at the highest representable
# wavenumber. RR-LBM and SVV are complementary:
#   - RR-LBM (kernel_regularization_d3q39.jl) kills direction-space ghosts
#     (population patterns with zero macroscopic moments).
#   - SVV (this file) kills SPATIAL high-wavenumber modes that have real
#     macroscopic content but are aliased / under-resolved.
#
# Theory: Karamanos-Karniadakis 2000 §2 specifies SVV via three properties:
#   (1) viscosity amplitude vanishing with mode number (ε → 0 as N → ∞),
#   (2) viscosity-free spectrum at low, energetic modes (k < M),
#   (3) appropriate kernel for high wavenumbers (k ≥ M).
# In real-space (collocated) LBM, this is realized as a HYPERVISCOSITY operator:
# the n-th order Laplacian. The k-eigenvalue scales as k^(2n) — sharper cutoff
# for higher n. We use n=2 (bi-Laplacian, ∇⁴): k⁴ scaling, 47× selectivity
# between Nyquist (k=π) and 4-cell wavelength (k=π/4).
#
# Discrete bi-Laplacian 5-point stencil:
#   ∂⁴f/∂y⁴ ≈ f(y-2) − 4·f(y-1) + 6·f(y) − 4·f(y+1) + f(y+2)
# Y-Z plane: total = ∂⁴/∂y⁴ + ∂⁴/∂z⁴. X-axis (streamwise) is left untouched.
#
# k-space eigenvalues of the 5-point bi-Laplacian (per axis): 4·(cos(k)−1)²
#   k=π     (Nyquist):       16    ← target
#   k=π/2   (4 cells):         4
#   k=π/4   (8 cells):         0.343
#   k=π/8   (16 cells):        0.023
# Y-Z combined: 2× these for the (-1)^(y+z) eigenmode → 32 at Nyquist.
#
# Conservation: bi-Laplacian on each population is independently symmetric.
# Globally, Σ_k f^reg = Σ_k f via integration-by-parts of the bi-Laplacian.
# Locally, a 4-moment correction (Lagrava 2012 §3.5 style) is applied AFTER
# the filter to restore mass and momentum EXACTLY per cell. The correction
# uses the equilibrium projection: `f_corrected[k] = f_filt[k] − w_k·(Δρ +
# c_k·Δj/cs²)`. This zeroes the changes to Σf and Σc·f per cell while
# preserving the higher-moment content (stress, heat-flux, …) that the filter
# was designed to damp.
#
# Implementation note: neighbor reads use f_in (pre-collision) for the 8 Y-Z
# neighbors. This introduces a 1-step temporal lag in the filter — acceptable
# for small ε since the spatial pattern (the checkerboard) evolves slowly
# compared to the collision step.
# ==============================================================================

using StaticArrays

"""
    get_pop_neighbor_yz(f_in, obstacle, x, y, z, b_idx, dy, dz, k, block_size, neighbor_table)

Fetch population `k` from cell (x, y+dy, z+dz). Handles cross-block lookups via
`neighbor_table` for offsets that exit the current block. Supports |dy|, |dz| ≤ 2
(BLOCK_SIZE = 8 ensures at most one block hop).

When the neighbor block is outside the domain (neighbor_table returns 0), falls
back to the center value — equivalent to a Neumann (zero-flux) boundary for
the bi-Laplacian operator at domain edges.
"""
@inline function get_pop_neighbor_yz(
    f_in,
    obstacle,
    x, y, z, b_idx,
    dy, dz, k,
    block_size, neighbor_table
)
    @inbounds begin
        ny = y + dy
        nz = z + dz

        nb_off_y = ny < Int32(1) ? Int32(-1) : (ny > block_size ? Int32(1) : Int32(0))
        nb_off_z = nz < Int32(1) ? Int32(-1) : (nz > block_size ? Int32(1) : Int32(0))

        if nb_off_y == Int32(0) && nb_off_z == Int32(0)
            # Same block (most common case for interior cells).
            # Obstacle-aware Neumann fallback: if the neighbor is solid, use the
            # center value so the bi-Laplacian sees a zero-flux wall.
            if obstacle[x, ny, nz, b_idx]
                return f_in[x, y, z, b_idx, k]
            end
            return f_in[x, ny, nz, b_idx, k]
        end

        # Cross-block: lookup neighbor block via neighbor_table.
        # Index convention: dir_idx = (off_x+1) + (off_y+1)*3 + (off_z+1)*9 + 1
        # with off_x = 0 (Y-Z filter only): dir_idx = 1 + (off_y+1)*3 + (off_z+1)*9 + 1
        dir_idx = Int32(1) + (nb_off_y + Int32(1)) * Int32(3) + (nb_off_z + Int32(1)) * Int32(9) + Int32(1)
        nb_b_idx = neighbor_table[b_idx, dir_idx]

        if nb_b_idx == Int32(0)
            # Out of domain — Neumann fallback (use center value).
            return f_in[x, y, z, b_idx, k]
        end

        # Wrap local indices into the neighbor block.
        ny_local = ny < Int32(1)        ? ny + block_size :
                   ny > block_size      ? ny - block_size : ny
        nz_local = nz < Int32(1)        ? nz + block_size :
                   nz > block_size      ? nz - block_size : nz

        # Obstacle-aware Neumann fallback for cross-block neighbors too.
        if obstacle[x, ny_local, nz_local, nb_b_idx]
            return f_in[x, y, z, b_idx, k]
        end
        return f_in[x, ny_local, nz_local, nb_b_idx, k]
    end
end

"""
    apply_svv_yz_d3q39_inline!(f_post, f_in, obstacle, neighbor_table,
                                x, y, z, b_idx, block_size,
                                epsilon, n_d, w_arr, cx_arr, cy_arr, cz_arr,
                                cs2_val, shift_x, shift_y, shift_z)

Apply Y-Z bi-Laplacian SVV to `f_post` (39-element MVector) in place, using
pre-collision populations in `f_in` for the 8 Y-Z neighbors per direction.

Step 1: for each direction k, compute the discrete 5-point bi-Laplacian in Y
and in Z and update `f_post[k] -= epsilon · (biLap_y + biLap_z)`.

Step 2: compute the cell-integrated changes to mass (Δρ) and momentum (Δj) and
apply a 4-moment correction `f_post[k] -= w_k · (Δρ + (c_k − shift) · Δj / cs²)`
so that the post-filter populations have EXACTLY the same ρ and ρu as before
the filter. Higher moments (stress, heat-flux) retain the SVV damping.

The neighbor offsets used are (0, ±1, 0), (0, ±2, 0), (0, 0, ±1), (0, 0, ±2)
— X-axis (streamwise) is intentionally NOT touched.

No-op when `epsilon == 0`.
"""
@inline function apply_svv_yz_d3q39_inline!(
    f_post::MVector{39, Float32},
    f_in,
    obstacle, neighbor_table,
    x, y, z, b_idx, block_size,
    epsilon::Float32,
    n_d,
    w_arr, cx_arr, cy_arr, cz_arr,
    cs2_val::Float32,
    shift_x::Float32, shift_y::Float32, shift_z::Float32
)
    @inbounds begin
        if epsilon == 0.0f0
            return nothing
        end

        # ── STEP 1: Apply bi-Laplacian to each direction ────────────────────
        delta_rho = 0.0f0
        delta_jx  = 0.0f0
        delta_jy  = 0.0f0
        delta_jz  = 0.0f0

        for k in Int32(1):n_d
            f_center = f_post[k]

            # Y-axis 5-point stencil: f(y-2) − 4f(y-1) + 6f(y) − 4f(y+1) + f(y+2)
            f_ym2 = get_pop_neighbor_yz(f_in, obstacle, x, y, z, b_idx, Int32(-2), Int32(0), k, block_size, neighbor_table)
            f_ym1 = get_pop_neighbor_yz(f_in, obstacle, x, y, z, b_idx, Int32(-1), Int32(0), k, block_size, neighbor_table)
            f_yp1 = get_pop_neighbor_yz(f_in, obstacle, x, y, z, b_idx, Int32( 1), Int32(0), k, block_size, neighbor_table)
            f_yp2 = get_pop_neighbor_yz(f_in, obstacle, x, y, z, b_idx, Int32( 2), Int32(0), k, block_size, neighbor_table)
            biLap_y = f_ym2 - 4.0f0 * f_ym1 + 6.0f0 * f_center - 4.0f0 * f_yp1 + f_yp2

            # Z-axis 5-point stencil
            f_zm2 = get_pop_neighbor_yz(f_in, obstacle, x, y, z, b_idx, Int32(0), Int32(-2), k, block_size, neighbor_table)
            f_zm1 = get_pop_neighbor_yz(f_in, obstacle, x, y, z, b_idx, Int32(0), Int32(-1), k, block_size, neighbor_table)
            f_zp1 = get_pop_neighbor_yz(f_in, obstacle, x, y, z, b_idx, Int32(0), Int32( 1), k, block_size, neighbor_table)
            f_zp2 = get_pop_neighbor_yz(f_in, obstacle, x, y, z, b_idx, Int32(0), Int32( 2), k, block_size, neighbor_table)
            biLap_z = f_zm2 - 4.0f0 * f_zm1 + 6.0f0 * f_center - 4.0f0 * f_zp1 + f_zp2

            # SVV update — damps high-k modes by ε·k⁴ (per axis).
            # Sign convention: bi-Laplacian eigenvalue is POSITIVE for oscillatory
            # modes (e.g., +16 at Nyquist). To DAMP, subtract: f_new = f − ε·biLap.
            f_filtered = f_center - epsilon * (biLap_y + biLap_z)

            # Track cell-integrated changes for the conservation correction.
            # Use unshifted velocities for the macroscopic moments (Galilean shift
            # is an internal lattice transformation; macros are computed in the
            # lab frame after un-shifting).
            diff = f_filtered - f_center
            delta_rho += diff
            delta_jx  += diff * (Float32(cx_arr[k]) - shift_x)
            delta_jy  += diff * (Float32(cy_arr[k]) - shift_y)
            delta_jz  += diff * (Float32(cz_arr[k]) - shift_z)

            f_post[k] = f_filtered
        end

        # ── STEP 2: 4-moment conservation correction ──────────────────────
        # Restore exact local mass and momentum (Lagrava 2012 §3.5 style):
        #   f_corrected[k] = f_filtered[k] − w_k · (Δρ + (c_k − shift) · Δj / cs²)
        # The correction lives in the equilibrium subspace (mass + momentum)
        # by construction, so the SVV-damped higher moments (stress, heat-flux)
        # are preserved exactly.
        inv_cs2 = 1.0f0 / cs2_val
        for k in Int32(1):n_d
            w_k = Float32(w_arr[k])
            cx_lab = Float32(cx_arr[k]) - shift_x
            cy_lab = Float32(cy_arr[k]) - shift_y
            cz_lab = Float32(cz_arr[k]) - shift_z
            cu_delta = cx_lab * delta_jx + cy_lab * delta_jy + cz_lab * delta_jz
            correction = w_k * (delta_rho + inv_cs2 * cu_delta)
            f_post[k] -= correction
        end
    end
    return nothing
end

# ==============================================================================
# STREAMWISE-X SVV (SOTA #5, 2026-05-29) — companion to the Y-Z SVV above.
# ==============================================================================
# The Y-Z SVV intentionally leaves the streamwise (X) axis untouched, because in
# the wake the under-resolved K-H content is transverse. But case 030 exposed a
# DIFFERENT mode: a streamwise (X-periodic) velocity/Mach banding on the L1
# (coarsest) grid, growing in time. It is a real spatial high-k mode in X — the
# Y-Z filter cannot see it. This routine is the X-axis bi-Laplacian analog,
# meant to be gated to the COARSEST level(s) only (where the lattice ν is
# smallest under the uniform-physical-ν amr_sgs_scale and the mode is least
# damped). Same k⁴ selectivity, same exact mass+momentum conservation
# correction. No-op when epsilon == 0.
@inline function get_pop_neighbor_x(
    f_in, obstacle, x, y, z, b_idx, dx, k, block_size, neighbor_table
)
    @inbounds begin
        nx = x + dx
        nb_off_x = nx < Int32(1) ? Int32(-1) : (nx > block_size ? Int32(1) : Int32(0))
        if nb_off_x == Int32(0)
            if obstacle[nx, y, z, b_idx]
                return f_in[x, y, z, b_idx, k]
            end
            return f_in[nx, y, z, b_idx, k]
        end
        # Cross-block in X: dir_idx = (off_x+1) + (off_y+1)*3 + (off_z+1)*9 + 1
        # with off_y = off_z = 0.
        dir_idx = (nb_off_x + Int32(1)) + Int32(3) + Int32(9) + Int32(1)
        nb_b_idx = neighbor_table[b_idx, dir_idx]
        if nb_b_idx == Int32(0)
            return f_in[x, y, z, b_idx, k]
        end
        nx_local = nx < Int32(1) ? nx + block_size :
                   nx > block_size ? nx - block_size : nx
        if obstacle[nx_local, y, z, nb_b_idx]
            return f_in[x, y, z, b_idx, k]
        end
        return f_in[nx_local, y, z, nb_b_idx, k]
    end
end

@inline function apply_svv_x_d3q39_inline!(
    f_post::MVector{39, Float32},
    f_in,
    obstacle, neighbor_table,
    x, y, z, b_idx, block_size,
    epsilon::Float32,
    n_d,
    w_arr, cx_arr, cy_arr, cz_arr,
    cs2_val::Float32,
    shift_x::Float32, shift_y::Float32, shift_z::Float32
)
    @inbounds begin
        if epsilon == 0.0f0
            return nothing
        end
        delta_rho = 0.0f0; delta_jx = 0.0f0; delta_jy = 0.0f0; delta_jz = 0.0f0
        for k in Int32(1):n_d
            f_center = f_post[k]
            f_xm2 = get_pop_neighbor_x(f_in, obstacle, x, y, z, b_idx, Int32(-2), k, block_size, neighbor_table)
            f_xm1 = get_pop_neighbor_x(f_in, obstacle, x, y, z, b_idx, Int32(-1), k, block_size, neighbor_table)
            f_xp1 = get_pop_neighbor_x(f_in, obstacle, x, y, z, b_idx, Int32( 1), k, block_size, neighbor_table)
            f_xp2 = get_pop_neighbor_x(f_in, obstacle, x, y, z, b_idx, Int32( 2), k, block_size, neighbor_table)
            biLap_x = f_xm2 - 4.0f0 * f_xm1 + 6.0f0 * f_center - 4.0f0 * f_xp1 + f_xp2
            f_filtered = f_center - epsilon * biLap_x
            diff = f_filtered - f_center
            delta_rho += diff
            delta_jx  += diff * (Float32(cx_arr[k]) - shift_x)
            delta_jy  += diff * (Float32(cy_arr[k]) - shift_y)
            delta_jz  += diff * (Float32(cz_arr[k]) - shift_z)
            f_post[k] = f_filtered
        end
        inv_cs2 = 1.0f0 / cs2_val
        for k in Int32(1):n_d
            w_k = Float32(w_arr[k])
            cx_lab = Float32(cx_arr[k]) - shift_x
            cy_lab = Float32(cy_arr[k]) - shift_y
            cz_lab = Float32(cz_arr[k]) - shift_z
            cu_delta = cx_lab * delta_jx + cy_lab * delta_jy + cz_lab * delta_jz
            f_post[k] -= w_k * (delta_rho + inv_cs2 * cu_delta)
        end
    end
    return nothing
end

# ==============================================================================
# D3Q19 SVV (2026-06-05) — MVector{19} sibling of the D3Q39 routines above.
# ==============================================================================
# The D3Q19 RR-BGK kernel has NO high-wavenumber dissipation that can see a
# cell-scale velocity Nyquist ring: WALE uses a central difference (identically
# zero for a (-1)^i checkerboard) and RR order-2 PRESERVES the mode (it carries
# real macroscopic velocity, so it is not a ghost mode). This routine is the
# purpose-built sink: a k⁴-selective bi-Laplacian that damps the cell-scale ring
# ~6%/step/axis at Nyquist while losing <0.15%/step at well-resolved k≤π/4.
#
# Filters ALL THREE axes (X via eps_x, Y+Z via eps_yz) in a SINGLE conservation
# pass — the D3Q19 sphere ring sits in the XZ plane at the shoulders, so X must
# be filtered (a Y-Z-only filter, the D3Q39 wake default, leaves the X-stripes).
# D3Q19 has NO Galilean shift, so the lab-frame velocities are the bare c_k.
# The 4-moment correction restores mass + 3·momentum EXACTLY per cell regardless
# of how many axes are filtered (only the damped stress/heat-flux high-k content
# is removed — the intent). No-op when both epsilons are 0. Neighbor fetches
# reuse the lattice-generic get_pop_neighbor_x / get_pop_neighbor_yz helpers.
@inline function apply_svv_d3q19_inline!(
    f_post::MVector{N, Float32},   # 2026-06-23: generic N (19 for D3Q19, 27 for D3Q27
                                   # cumulant). The k⁴ bi-Laplacian + 4-moment
                                   # conservation are direction-agnostic; only the
                                   # buffer size was lattice-specific. Loops use n_d.
    f_in,
    obstacle, neighbor_table,
    x, y, z, b_idx, block_size,
    eps_x::Float32, eps_yz::Float32,
    n_d,
    w_arr, cx_arr, cy_arr, cz_arr,
    cs2_val::Float32
) where {N}
    @inbounds begin
        if eps_x == 0.0f0 && eps_yz == 0.0f0
            return nothing
        end
        delta_rho = 0.0f0; delta_jx = 0.0f0; delta_jy = 0.0f0; delta_jz = 0.0f0
        for k in Int32(1):n_d
            f_center = f_post[k]
            biLap = 0.0f0
            if eps_x > 0.0f0
                f_xm2 = get_pop_neighbor_x(f_in, obstacle, x, y, z, b_idx, Int32(-2), k, block_size, neighbor_table)
                f_xm1 = get_pop_neighbor_x(f_in, obstacle, x, y, z, b_idx, Int32(-1), k, block_size, neighbor_table)
                f_xp1 = get_pop_neighbor_x(f_in, obstacle, x, y, z, b_idx, Int32( 1), k, block_size, neighbor_table)
                f_xp2 = get_pop_neighbor_x(f_in, obstacle, x, y, z, b_idx, Int32( 2), k, block_size, neighbor_table)
                biLap += eps_x * (f_xm2 - 4.0f0*f_xm1 + 6.0f0*f_center - 4.0f0*f_xp1 + f_xp2)
            end
            if eps_yz > 0.0f0
                f_ym2 = get_pop_neighbor_yz(f_in, obstacle, x, y, z, b_idx, Int32(-2), Int32(0), k, block_size, neighbor_table)
                f_ym1 = get_pop_neighbor_yz(f_in, obstacle, x, y, z, b_idx, Int32(-1), Int32(0), k, block_size, neighbor_table)
                f_yp1 = get_pop_neighbor_yz(f_in, obstacle, x, y, z, b_idx, Int32( 1), Int32(0), k, block_size, neighbor_table)
                f_yp2 = get_pop_neighbor_yz(f_in, obstacle, x, y, z, b_idx, Int32( 2), Int32(0), k, block_size, neighbor_table)
                biLap += eps_yz * (f_ym2 - 4.0f0*f_ym1 + 6.0f0*f_center - 4.0f0*f_yp1 + f_yp2)
                f_zm2 = get_pop_neighbor_yz(f_in, obstacle, x, y, z, b_idx, Int32(0), Int32(-2), k, block_size, neighbor_table)
                f_zm1 = get_pop_neighbor_yz(f_in, obstacle, x, y, z, b_idx, Int32(0), Int32(-1), k, block_size, neighbor_table)
                f_zp1 = get_pop_neighbor_yz(f_in, obstacle, x, y, z, b_idx, Int32(0), Int32( 1), k, block_size, neighbor_table)
                f_zp2 = get_pop_neighbor_yz(f_in, obstacle, x, y, z, b_idx, Int32(0), Int32( 2), k, block_size, neighbor_table)
                biLap += eps_yz * (f_zm2 - 4.0f0*f_zm1 + 6.0f0*f_center - 4.0f0*f_zp1 + f_zp2)
            end
            f_filtered = f_center - biLap
            diff = f_filtered - f_center
            delta_rho += diff
            delta_jx  += diff * Float32(cx_arr[k])   # shift == 0 for D3Q19
            delta_jy  += diff * Float32(cy_arr[k])
            delta_jz  += diff * Float32(cz_arr[k])
            f_post[k] = f_filtered
        end
        # 4-moment conservation correction (shift=0): restores ρ and ρu exactly.
        inv_cs2 = 1.0f0 / cs2_val
        for k in Int32(1):n_d
            w_k = Float32(w_arr[k])
            cu_delta = Float32(cx_arr[k])*delta_jx + Float32(cy_arr[k])*delta_jy + Float32(cz_arr[k])*delta_jz
            f_post[k] -= w_k * (delta_rho + inv_cs2 * cu_delta)
        end
    end
    return nothing
end

# ==============================================================================
# D3Q19 SVV as a SEPARATE post-collision kernel (2026-06-05, OCCUPANCY FIX)
# ==============================================================================
# Running SVV INLINE in stream_collide_d3q19_rr! forced the f_stored MVector{19}
# to stay register-live to the end of the kernel and added ~16 neighbor/accum
# temporaries → pushed the AMR variant to the 255-reg cap / 16.7% occupancy. This
# standalone kernel applies the SAME k⁴ bi-Laplacian SVV (via apply_svv_d3q19_inline!)
# as a post-collision pass — exactly like apply_grad_wall_correction! /
# apply_bouzidi_correction! — so the main collision kernel sheds all the SVV
# register pressure. Center pops are loaded from f_out (the just-written post-
# collision field); neighbor pops are read from f_in (1-step lag, race-free: f_in
# is read-only this step, f_out is the separate buffer). Per-cell 4-moment
# conservation is unchanged, so continuity/seam behaviour is identical to the
# inline version. No-op when both epsilons are 0; skips solid cells.
@kernel function apply_svv_d3q19_postcollision!(
    f_out, f_in, f_post_collision, obstacle, neighbor_table,
    block_size::Int32,
    cx_arr, cy_arr, cz_arr, w_arr,
    cs2_val::Float32, n_dirs::Int32,
    eps_x::Float32, eps_yz::Float32,
    write_post_collision::Int32
)
    x, y, z, b_idx = @index(Global, NTuple)
    @inbounds begin
        if !(eps_x == 0.0f0 && eps_yz == 0.0f0) && !obstacle[x, y, z, b_idx]
            # 2026-06-23: sized for D3Q27 (27 dirs); D3Q19 uses [1:19]. Loops bound
            # by n_dirs so the extra slots are dead for D3Q19 ⇒ byte-identical.
            f_loc = MVector{27, Float32}(undef)
            for k in Int32(1):n_dirs
                f_loc[k] = f_out[x, y, z, b_idx, k]
            end
            apply_svv_d3q19_inline!(
                f_loc, f_in, obstacle, neighbor_table,
                x, y, z, b_idx, block_size,
                eps_x, eps_yz, n_dirs,
                w_arr, cx_arr, cy_arr, cz_arr, cs2_val
            )
            for k in Int32(1):n_dirs
                fc = f_loc[k] < 1.0f-12 ? 1.0f-12 : f_loc[k]
                f_out[x, y, z, b_idx, k] = fc
                if write_post_collision == Int32(1)
                    f_post_collision[x, y, z, b_idx, k] = fc
                end
            end
        end
    end
end

# ==============================================================================
# D3Q19 SVV @localmem TILE (2026-06-10, 100-MLUPS item 1 — SVV was 23% of the
# production batch, the single heaviest non-flow op)
# ==============================================================================
# Same math as apply_svv_d3q19_postcollision! (bitwise-identical by
# construction), but the 228 per-cell f_in tap reads (19 dirs × 12 axial taps,
# each with its own block-pointer/neighbor_table/obstacle address logic) become
# a cooperative 3-phase tile:
#   P0  one thread-block per LBM block precomputes a (bs+4)³ tile of FLAT f_in
#       base indices (k=1) — the cross-block/obstacle/domain-edge logic runs
#       ONCE per tile cell instead of per (cell, dir, tap). Invalid cells
#       (missing neighbor block, solid, or the never-read diagonal halo
#       regions) get base 0.
#   P1  per direction k: gather the f_in tile (base + (k−1)·k_stride), sync,
#       each thread evaluates the SAME biLap expression with NaN→own-center
#       substitution (= the legacy get_pop_neighbor_* "use center value"
#       fallback, which substitutes the READER's f_in center), sync.
#   P2  the unchanged 4-moment correction + positivity clamp + write-back.
# Global f_in reads: 19 × ~1280 (cross region) vs 19 × 12 × 512 dense → ~4.8×
# fewer, plus all per-tap integer address math hoisted out of the k×tap loops.
# Workgroup is the full (bs,bs,bs) block; localmem = (bs+4)³ × (Int32 + Float32)
# = 13.5 KB at bs=8 (same budget class as the shared-bspline tile).
# Gate: LUDWIG_D3Q19_SVV_TILE (default ON since 2026-06-11: Z156 GPU A/B
# bitwise-identical seam CSVs at steps 30+60 AND svv phase 3.7s → 1.42s =
# 2.6× — see CASE_LOGS/svvtile_ab + z156_svvtile_{off,on}.log; "0" reverts
# to the legacy per-tap kernel).
const _D3Q19_SVV_TILE = get(ENV, "LUDWIG_D3Q19_SVV_TILE", "1") != "0"

@kernel function apply_svv_d3q19_postcollision_tile!(
    f_out, f_in, f_post_collision, obstacle, neighbor_table,
    block_size::Int32,
    cx_arr, cy_arr, cz_arr, w_arr,
    cs2_val::Float32, n_dirs::Int32,
    eps_x::Float32, eps_yz::Float32,
    write_post_collision::Int32,
    n_blocks_i::Int32,
    # FUSED-Π (2026-06-13): SVV is the LAST f-writer on every SVV-active
    # level, so it is the correct place to compute level.pi_neq (the post-
    # step refresh is otherwise a separate 19-read dense pass). SVV conserves
    # ρ+momentum exactly, so post-SVV ρ/u == the collision-stored rho_macro/
    # vel_macro — the epilogue reads those (bitwise match to refresh's
    # vel_arr read) and the final fc. Val(false) ⇒ dead, stub args.
    rho_macro, vel_macro, pi_neq_out,
    ::Val{FUSE_PINEQ}
) where {FUSE_PINEQ}
    # (bs+4)³ tile; coordinates: tile t ∈ 1..bs+4 ↔ block-local l = t − 2
    # (l ∈ −1..bs+2; the block interior is t ∈ 3..bs+2).
    tile_base = @localmem Int32 (12, 12, 12)
    tile_f    = @localmem Float32 (12, 12, 12)
    # Per-thread state that must SURVIVE @synchronize (the KA CPU backend
    # splits the kernel at barriers — plain locals do not cross them; on GPU
    # @private lowers to registers, same as locals).
    f_priv = @private Float32 (19)   # filtered populations
    acc    = @private Float32 (4)    # Δρ, Δjx, Δjy, Δjz

    # @index variables are re-materialized by KA in every barrier segment (CPU
    # backend) / live in registers (GPU) — only DERIVED locals must be
    # recomputed per segment below.
    x, y, z, _b_local = @index(Local, NTuple)
    _g1, _g2, _g3, b_idx = @index(Group, NTuple)

    # ── P0: base-index tile (runs the neighbor/obstacle logic once per cell) ──
    @inbounds begin
        n_tile = Int32(12)
        n_tile3 = n_tile * n_tile * n_tile
        nthreads = block_size * block_size * block_size
        t = Int32(x) + (Int32(y) - Int32(1)) * block_size +
            (Int32(z) - Int32(1)) * block_size * block_size          # tid ∈ 1..bs³
        while t <= n_tile3
            tz = (t - Int32(1)) ÷ (n_tile * n_tile) + Int32(1)
            rem2 = (t - Int32(1)) % (n_tile * n_tile)
            ty = rem2 ÷ n_tile + Int32(1)
            tx = rem2 % n_tile + Int32(1)
            lx = tx - Int32(2); ly = ty - Int32(2); lz = tz - Int32(2)
            hx = (lx < Int32(1)) | (lx > block_size)
            hy = (ly < Int32(1)) | (ly > block_size)
            hz = (lz < Int32(1)) | (lz > block_size)
            n_halo = Int32(hx) + Int32(hy) + Int32(hz)
            base = Int32(0)
            if n_halo == Int32(0)
                # interior cell of THIS block
                if !obstacle[lx, ly, lz, b_idx]
                    base = lx + (ly - Int32(1)) * block_size +
                           (lz - Int32(1)) * block_size * block_size +
                           (Int32(b_idx) - Int32(1)) * block_size * block_size * block_size
                end
            elseif n_halo == Int32(1)
                # axial halo cell — at most one block hop (|l−edge| ≤ 2 < bs)
                nb_off_x = lx < Int32(1) ? Int32(-1) : (lx > block_size ? Int32(1) : Int32(0))
                nb_off_y = ly < Int32(1) ? Int32(-1) : (ly > block_size ? Int32(1) : Int32(0))
                nb_off_z = lz < Int32(1) ? Int32(-1) : (lz > block_size ? Int32(1) : Int32(0))
                dir_idx = (nb_off_x + Int32(1)) + (nb_off_y + Int32(1)) * Int32(3) +
                          (nb_off_z + Int32(1)) * Int32(9) + Int32(1)
                nb = neighbor_table[b_idx, dir_idx]
                if nb > Int32(0)
                    wx = lx < Int32(1) ? lx + block_size : (lx > block_size ? lx - block_size : lx)
                    wy = ly < Int32(1) ? ly + block_size : (ly > block_size ? ly - block_size : ly)
                    wz = lz < Int32(1) ? lz + block_size : (lz > block_size ? lz - block_size : lz)
                    if !obstacle[wx, wy, wz, nb]
                        base = wx + (wy - Int32(1)) * block_size +
                               (wz - Int32(1)) * block_size * block_size +
                               (nb - Int32(1)) * block_size * block_size * block_size
                    end
                end
            end
            # n_halo ≥ 2: diagonal halo region — never read by the axial taps.
            tile_base[tx, ty, tz] = base
            t += nthreads
        end
        # SVV lever 1 (2026-06-13): compile-time 19 bounds — this kernel is
        # D3Q19-only (launcher guards block_size==8; n_dirs is always 19).
        # Runtime n_dirs bounds + dynamic f_priv[k] indexing lower the
        # @private arrays to CUDA local memory; constant bounds let LLVM
        # unroll and promote them to registers. Bitwise (same arithmetic).
        for k in Int32(1):Int32(19)
            f_priv[k] = f_out[x, y, z, b_idx, k]
        end
        acc[1] = 0.0f0; acc[2] = 0.0f0; acc[3] = 0.0f0; acc[4] = 0.0f0
    end
    @synchronize

    for k in Int32(1):Int32(19)
        # gather this direction's f_in tile
        @inbounds begin
            n_tile = Int32(12)
            n_tile3 = n_tile * n_tile * n_tile
            nthreads = block_size * block_size * block_size
            k_stride = block_size * block_size * block_size * n_blocks_i
            t = Int32(x) + (Int32(y) - Int32(1)) * block_size +
                (Int32(z) - Int32(1)) * block_size * block_size
            while t <= n_tile3
                tz = (t - Int32(1)) ÷ (n_tile * n_tile) + Int32(1)
                rem2 = (t - Int32(1)) % (n_tile * n_tile)
                ty = rem2 ÷ n_tile + Int32(1)
                tx = rem2 % n_tile + Int32(1)
                base = tile_base[tx, ty, tz]
                tile_f[tx, ty, tz] = base > Int32(0) ?
                    f_in[base + (k - Int32(1)) * k_stride] : NaN32
                t += nthreads
            end
        end
        @synchronize

        @inbounds begin
            if !obstacle[x, y, z, b_idx] && !(eps_x == 0.0f0 && eps_yz == 0.0f0)
                tcx = Int32(x) + Int32(2); tcy = Int32(y) + Int32(2); tcz = Int32(z) + Int32(2)
                f_center = f_priv[k]
                c_in = tile_f[tcx, tcy, tcz]   # reader's own f_in center (valid: non-solid interior)
                biLap = 0.0f0
                if eps_x > 0.0f0
                    f_xm2 = tile_f[tcx - Int32(2), tcy, tcz]; f_xm2 = f_xm2 == f_xm2 ? f_xm2 : c_in
                    f_xm1 = tile_f[tcx - Int32(1), tcy, tcz]; f_xm1 = f_xm1 == f_xm1 ? f_xm1 : c_in
                    f_xp1 = tile_f[tcx + Int32(1), tcy, tcz]; f_xp1 = f_xp1 == f_xp1 ? f_xp1 : c_in
                    f_xp2 = tile_f[tcx + Int32(2), tcy, tcz]; f_xp2 = f_xp2 == f_xp2 ? f_xp2 : c_in
                    biLap += eps_x * (f_xm2 - 4.0f0*f_xm1 + 6.0f0*f_center - 4.0f0*f_xp1 + f_xp2)
                end
                if eps_yz > 0.0f0
                    f_ym2 = tile_f[tcx, tcy - Int32(2), tcz]; f_ym2 = f_ym2 == f_ym2 ? f_ym2 : c_in
                    f_ym1 = tile_f[tcx, tcy - Int32(1), tcz]; f_ym1 = f_ym1 == f_ym1 ? f_ym1 : c_in
                    f_yp1 = tile_f[tcx, tcy + Int32(1), tcz]; f_yp1 = f_yp1 == f_yp1 ? f_yp1 : c_in
                    f_yp2 = tile_f[tcx, tcy + Int32(2), tcz]; f_yp2 = f_yp2 == f_yp2 ? f_yp2 : c_in
                    biLap += eps_yz * (f_ym2 - 4.0f0*f_ym1 + 6.0f0*f_center - 4.0f0*f_yp1 + f_yp2)
                    f_zm2 = tile_f[tcx, tcy, tcz - Int32(2)]; f_zm2 = f_zm2 == f_zm2 ? f_zm2 : c_in
                    f_zm1 = tile_f[tcx, tcy, tcz - Int32(1)]; f_zm1 = f_zm1 == f_zm1 ? f_zm1 : c_in
                    f_zp1 = tile_f[tcx, tcy, tcz + Int32(1)]; f_zp1 = f_zp1 == f_zp1 ? f_zp1 : c_in
                    f_zp2 = tile_f[tcx, tcy, tcz + Int32(2)]; f_zp2 = f_zp2 == f_zp2 ? f_zp2 : c_in
                    biLap += eps_yz * (f_zm2 - 4.0f0*f_zm1 + 6.0f0*f_center - 4.0f0*f_zp1 + f_zp2)
                end
                f_filtered = f_center - biLap
                diff = f_filtered - f_center
                acc[1] += diff
                acc[2] += diff * Float32(cx_arr[k])
                acc[3] += diff * Float32(cy_arr[k])
                acc[4] += diff * Float32(cz_arr[k])
                f_priv[k] = f_filtered
            end
        end
        @synchronize   # tile_f is overwritten by the next direction's gather
    end

    @inbounds begin
        if !obstacle[x, y, z, b_idx] && !(eps_x == 0.0f0 && eps_yz == 0.0f0)
            inv_cs2 = 1.0f0 / cs2_val
            delta_rho = acc[1]; delta_jx = acc[2]; delta_jy = acc[3]; delta_jz = acc[4]
            for k in Int32(1):Int32(19)
                w_k = Float32(w_arr[k])
                cu_delta = Float32(cx_arr[k])*delta_jx + Float32(cy_arr[k])*delta_jy + Float32(cz_arr[k])*delta_jz
                f_priv[k] -= w_k * (delta_rho + inv_cs2 * cu_delta)
            end
            # FUSED-Π: read the collision-stored macros (= post-SVV ρ/u, since
            # SVV conserves both; bitwise match to the refresh's vel_arr read)
            # and replicate compute_pi_neq_kernel!'s D3Q19 arithmetic verbatim.
            fp_rho = FUSE_PINEQ ? rho_macro[x, y, z, b_idx] : 0.0f0
            fp_ux  = FUSE_PINEQ ? vel_macro[x, y, z, b_idx, 1] : 0.0f0
            fp_uy  = FUSE_PINEQ ? vel_macro[x, y, z, b_idx, 2] : 0.0f0
            fp_uz  = FUSE_PINEQ ? vel_macro[x, y, z, b_idx, 3] : 0.0f0
            # refresh skip mask: obstacles excluded above; mirror the rest.
            fp_skip = FUSE_PINEQ && ((!isfinite(fp_rho)) | (fp_rho < 0.01f0) |
                      (fp_rho > 10.0f0) | (!isfinite(fp_ux)) | (!isfinite(fp_uy)) |
                      (!isfinite(fp_uz)))
            fp_rho_s = fp_skip ? 0.0f0 : fp_rho
            fp_ux_s  = fp_skip ? 0.0f0 : fp_ux
            fp_uy_s  = fp_skip ? 0.0f0 : fp_uy
            fp_uz_s  = fp_skip ? 0.0f0 : fp_uz
            fp_pxx = 0.0f0; fp_pyy = 0.0f0; fp_pzz = 0.0f0
            fp_pxy = 0.0f0; fp_pxz = 0.0f0; fp_pyz = 0.0f0
            for k in Int32(1):Int32(19)
                fc = f_priv[k] < 1.0f-12 ? 1.0f-12 : f_priv[k]
                f_out[x, y, z, b_idx, k] = fc
                if write_post_collision == Int32(1)
                    f_post_collision[x, y, z, b_idx, k] = fc
                end
                if FUSE_PINEQ
                    cx_f = Float32(cx_arr[k]); cy_f = Float32(cy_arr[k]); cz_f = Float32(cz_arr[k])
                    w_k = Float32(w_arr[k])
                    fp_cu  = cx_f*fp_ux_s + cy_f*fp_uy_s + cz_f*fp_uz_s
                    fp_usq = fp_ux_s*fp_ux_s + fp_uy_s*fp_uy_s + fp_uz_s*fp_uz_s
                    fp_feq = fp_rho_s * w_k *
                             (1.0f0 + 3.0f0*fp_cu + 4.5f0*fp_cu*fp_cu - 1.5f0*fp_usq)
                    fp_fk = fp_skip ? 0.0f0 : fc
                    fp_fneq = fp_fk - fp_feq
                    fp_pxx += cx_f*cx_f * fp_fneq
                    fp_pyy += cy_f*cy_f * fp_fneq
                    fp_pzz += cz_f*cz_f * fp_fneq
                    fp_pxy += cx_f*cy_f * fp_fneq
                    fp_pxz += cx_f*cz_f * fp_fneq
                    fp_pyz += cy_f*cz_f * fp_fneq
                end
            end
            if FUSE_PINEQ
                fp_mask = fp_skip ? 0.0f0 : 1.0f0
                pi_neq_out[x, y, z, b_idx, 1] = fp_pxx * fp_mask
                pi_neq_out[x, y, z, b_idx, 2] = fp_pyy * fp_mask
                pi_neq_out[x, y, z, b_idx, 3] = fp_pzz * fp_mask
                pi_neq_out[x, y, z, b_idx, 4] = fp_pxy * fp_mask
                pi_neq_out[x, y, z, b_idx, 5] = fp_pxz * fp_mask
                pi_neq_out[x, y, z, b_idx, 6] = fp_pyz * fp_mask
            end
        end
    end
end

# Host launcher for the standalone D3Q19 SVV post-collision pass.
# FUSED-Π (2026-06-13): pass rho_macro/vel_macro/pi_neq_out + fuse_pineq.
# When fuse_pineq, the tile kernel also writes level.pi_neq for the cells it
# owns (every SVV-active fluid cell), replacing the post-step refresh there.
# nothing/false ⇒ the Val(false) path DCEs the fused block (stub-safe args).
function apply_svv_d3q19_postcollision_launch!(
    f_out, f_in, f_post_collision, obstacle, neighbor_table,
    block_size::Int, n_blocks::Int,
    cx_gpu, cy_gpu, cz_gpu, w_gpu,
    cs2_val::Float32, n_dirs::Int,
    eps_x::Float32, eps_yz::Float32,
    write_post_collision::Int32,
    backend;
    rho_macro=nothing, vel_macro=nothing, pi_neq_out=nothing,
    fuse_pineq::Bool=false,
)
    (eps_x == 0.0f0 && eps_yz == 0.0f0) && return nothing
    n_blocks == 0 && return nothing
    _fp_rho = rho_macro === nothing ? f_out : rho_macro
    _fp_vel = vel_macro === nothing ? f_out : vel_macro
    _fp_pi  = pi_neq_out === nothing ? f_out : pi_neq_out
    if _D3Q19_SVV_TILE && block_size == 8 && n_dirs == 19
        # 2026-06-23: the localmem TILE kernel is hard-coded for the 19-direction
        # D3Q19 layout — gate it on n_dirs==19 so D3Q27 (n_dirs=27) falls through to
        # the plain MVector-generic kernel below. D3Q19 path unchanged.
        # Static full-block workgroup (one thread-block per LBM block) — same
        # instantiation pattern as the shared-bspline tile kernel.
        kernel_tile! = apply_svv_d3q19_postcollision_tile!(backend, (8, 8, 8, 1))
        kernel_tile!(
            f_out, f_in, f_post_collision, obstacle, neighbor_table,
            Int32(block_size),
            cx_gpu, cy_gpu, cz_gpu, w_gpu,
            cs2_val, Int32(n_dirs),
            eps_x, eps_yz, write_post_collision,
            Int32(n_blocks),
            _fp_rho, _fp_vel, _fp_pi, Val(fuse_pineq);
            ndrange = (block_size, block_size, block_size, n_blocks)
        )
        return nothing
    end
    kernel! = apply_svv_d3q19_postcollision!(backend)
    kernel!(
        f_out, f_in, f_post_collision, obstacle, neighbor_table,
        Int32(block_size),
        cx_gpu, cy_gpu, cz_gpu, w_gpu,
        cs2_val, Int32(n_dirs),
        eps_x, eps_yz, write_post_collision;
        ndrange = (block_size, block_size, block_size, n_blocks)
    )
    return nothing
end
