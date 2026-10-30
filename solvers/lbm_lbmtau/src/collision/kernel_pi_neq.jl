"""
KERNEL_PI_NEQ.JL — Non-Equilibrium Stress Tensor Diagnostic

Computes Π_neq,αβ = Σ_k c_kα c_kβ (f_k − f_eq_k) at every fluid cell.
This is the second-order moment of the non-equilibrium distribution and
drives viscous stress (σ = −(1−ω/2)·Π_neq via Chapman-Enskog).

Physical interpretation:
- |Π_neq| is large where strain rate is large (boundary layers, shock fronts, wakes)
- Its trace captures bulk dissipation (compression/expansion)
- Its deviatoric part captures shear stress

Diagnostic use:
- Visualization of boundary-layer / shock structure
- Precursor infrastructure for Palabos-style variable-ω (Item 2 of port queue)
- Precursor for stress-tensor-based Bouzidi force integration (Item 3)

Equilibrium form:
- D3Q27: polynomial O(u²) (or O(u⁴) if COMPRESSIBILITY_CORRECTION=true) — matches
  what the cumulant/BGK collision kernel actually subtracts
- D3Q39: polynomial O(u⁴) at cs²=2/3. Note: the entropic collision uses numerical
  13-moment NR equilibrium. Polynomial O(u⁴) differs by O(Ma⁶); adequate for
  qualitative visualization but NOT for force computation (Step A4).

Galilean shift (D3Q39 shifted lattice):
- cx_gpu contains SHIFTED velocities c'_k = c_k + U
- We unshift inside the kernel to compute Π_neq in the standard (rest-frame) form:
  Π_neq uses unshifted c_k and u_rel = u − U. This is frame-invariant because
  mass and momentum are exactly conserved by f_eq.
"""

using KernelAbstractions

# ──────────────────────────────────────────────────────────────────────────────
# Generic Π_neq kernel — works for D3Q27 and D3Q39
# Output layout: pi_neq[lx, ly, lz, b_idx, comp] where comp ∈ 1..6:
#   1=xx, 2=yy, 3=zz, 4=xy, 5=xz, 6=yz (symmetric 3×3 tensor)
# ──────────────────────────────────────────────────────────────────────────────
@kernel function compute_pi_neq_kernel!(
    pi_neq_out,                   # (BS,BS,BS,n_blocks,6)
    f_in,                         # (BS,BS,BS,n_blocks,n_dirs)
    rho_arr, vel_arr, obstacle_arr,
    cx_gpu, cy_gpu, cz_gpu, w_gpu,
    n_dirs::Int32,
    cs2::Float32,
    lattice_is_d3q39::Int32,
    comp_corr::Int32,
    shift_x::Float32, shift_y::Float32, shift_z::Float32,
    block_shift_arr,
    local_shift_active_i::Int32,
    # 100-MLUPS item 2b (2026-06-12): covered-block banding for the
    # restrict-side refresh. `nothing` ⇒ dense legacy (4th ndrange dim =
    # block index); a device Int32 list ⇒ banded (remapped below).
    covered_list
)
    lx, ly, lz, b_raw = @index(Global, NTuple)
    b_idx = covered_list === nothing ? Int32(b_raw) :
            @inbounds(covered_list[b_raw])

    is_obs = obstacle_arr[lx, ly, lz, b_idx]
    rho = rho_arr[lx, ly, lz, b_idx]
    ux  = vel_arr[lx, ly, lz, b_idx, 1]
    uy  = vel_arr[lx, ly, lz, b_idx, 2]
    uz  = vel_arr[lx, ly, lz, b_idx, 3]

    # Guard: zero output for obstacles or non-finite/unphysical macroscopic state.
    # Sanitize inputs so the accumulator loop produces 0 rather than NaN
    # (avoids NaN·0 = NaN when applying the final mask).
    skip = is_obs | (!isfinite(rho)) | (rho < 0.01f0) | (rho > 10.0f0) |
           (!isfinite(ux)) | (!isfinite(uy)) | (!isfinite(uz))
    rho_s = skip ? 0.0f0 : rho
    ux_s  = skip ? 0.0f0 : ux
    uy_s  = skip ? 0.0f0 : uy
    uz_s  = skip ? 0.0f0 : uz

    frame_shift_x = (local_shift_active_i == Int32(1) && lattice_is_d3q39 == Int32(1)) ?
                    Float32(block_shift_arr[b_idx]) : shift_x

    # Rest-frame velocity for equilibrium (identical to collision kernel convention).
    # In local shifted-lattice mode the velocity arrays remain unshifted; only the
    # block-local frame changes the relative velocity.
    ux_r = ux_s - frame_shift_x
    uy_r = uy_s - shift_y
    uz_r = uz_s - shift_z
    # Use zero when skip (f_neq computed as 0 - 0 = 0)

    # For D3Q39: solve for 13-moment Newton-Raphson equilibrium ONCE per cell.
    # This matches what the entropic collision kernel actually uses, eliminating
    # polynomial-equilibrium error at blunt-body stagnation (where u→0, ρ spikes).
    # Returns NTuple{13, Float32} of Lagrange multipliers.
    # Relaxed tolerance (10×) acceptable for diagnostic Π_neq computation.
    lambda_nr = if lattice_is_d3q39 == Int32(1)
        solve_13mom_lambdas_d3q39(
            max(rho_s, 1.0f-4), ux_r, uy_r, uz_r,
            cx_gpu, cy_gpu, cz_gpu, w_gpu,
            cs2, Int32(50), 1.0f0,
            shift_x, shift_y, shift_z,
            10.0f0  # tol_multiplier
        )
    else
        # Dummy NTuple for D3Q27 path (not used)
        (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0,
         0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)
    end
    # Mass correction normalizes Σf_eq = ρ (the NR solution may have tiny drift)
    mass_corr_nr = if lattice_is_d3q39 == Int32(1)
        compute_13mom_mass_correction_d3q39(
            max(rho_s, 1.0f-4), lambda_nr,
            cx_gpu, cy_gpu, cz_gpu, w_gpu,
            cs2, shift_x, shift_y, shift_z
        )
    else
        1.0f0
    end
    rho_eq = rho_s

    pxx = 0.0f0; pyy = 0.0f0; pzz = 0.0f0
    pxy = 0.0f0; pxz = 0.0f0; pyz = 0.0f0

    @inbounds for k in Int32(1):n_dirs
        # Unshift velocities (cx_gpu holds c' = c + U for shifted D3Q39)
        cxk = Float32(cx_gpu[k]) - shift_x
        cyk = Float32(cy_gpu[k]) - shift_y
        czk = Float32(cz_gpu[k]) - shift_z
        wk  = w_gpu[k]

        # Polynomial equilibrium (matches the collision kernel it approximates)
        cu  = cxk*ux_r + cyk*uy_r + czk*uz_r
        usq = ux_r*ux_r + uy_r*uy_r + uz_r*uz_r

        if lattice_is_d3q39 == Int32(1)
            # D3Q39: numerical 13-moment equilibrium (matches entropic collision kernel).
            # Uses pre-computed lambda_nr from solve_13mom_lambdas_d3q39 above.
            feq = mass_corr_nr * eval_13mom_feq_single(
                rho_s, lambda_nr, cxk, cyk, czk, wk, cs2
            )
        else
            # D3Q27 at cs² = 1/3
            if comp_corr == Int32(1)
                # O(u⁴) Hermite expansion
                cu2  = cu * cu
                cu3  = cu2 * cu
                cu4  = cu2 * cu2
                usq2 = usq * usq
                val = 1.0f0 + 3.0f0*cu + 4.5f0*cu2 - 1.5f0*usq
                val += 4.5f0*cu3 - 4.5f0*cu*usq
                val += 3.375f0*cu4 - 6.75f0*cu2*usq + 1.125f0*usq2
                feq = rho_eq * wk * val
            else
                # Standard O(u²)
                feq = rho_eq * wk * (1.0f0 + 3.0f0*cu + 4.5f0*cu*cu - 1.5f0*usq)
            end
        end

        # Sanitize f read for obstacle/unphysical cells to prevent NaN·0 = NaN
        f_k_raw = f_in[lx, ly, lz, b_idx, k]
        f_k_safe = skip ? 0.0f0 : f_k_raw
        f_neq_k = f_k_safe - feq

        pxx += cxk*cxk * f_neq_k
        pyy += cyk*cyk * f_neq_k
        pzz += czk*czk * f_neq_k
        pxy += cxk*cyk * f_neq_k
        pxz += cxk*czk * f_neq_k
        pyz += cyk*czk * f_neq_k
    end

    # Zero output for obstacles / unphysical cells; otherwise store accumulated Π_neq
    mask = skip ? 0.0f0 : 1.0f0
    @inbounds pi_neq_out[lx, ly, lz, b_idx, 1] = pxx * mask
    @inbounds pi_neq_out[lx, ly, lz, b_idx, 2] = pyy * mask
    @inbounds pi_neq_out[lx, ly, lz, b_idx, 3] = pzz * mask
    @inbounds pi_neq_out[lx, ly, lz, b_idx, 4] = pxy * mask
    @inbounds pi_neq_out[lx, ly, lz, b_idx, 5] = pxz * mask
    @inbounds pi_neq_out[lx, ly, lz, b_idx, 6] = pyz * mask
end


"""
    compute_pi_neq_for_level!(level, f_current, cx_gpu, cy_gpu, cz_gpu, w_gpu, backend)

Allocate a transient 6-component Π_neq buffer and fill it from the passed
f_current buffer (caller selects the correct ping-pong buffer for level).

Returns the GPU array of shape (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 6).
Caller is responsible for the lifetime of the returned array.

NOTE: Π_neq is computed from POST-collision populations (stored in level.f or
level.f_temp), so it is scaled by (1 − ω) relative to the pre-collision Π_neq
that drives Chapman-Enskog stress. For a visual diagnostic this scaling is
uniform within each AMR level. When this is later consumed for forces or
variable-ω, the (1−ω) factor MUST be corrected.
"""
# refresh_pi_neq!(level, f_current, cx_gpu, cy_gpu, cz_gpu, w_gpu, backend; ...)
#
# In-place variant: writes Π_neq into the pre-allocated `level.pi_neq` field
# (requires VARIABLE_NU_SGS=true so the field is not a stub). No allocation.
# Called from solver_control.jl after each collision step.
function refresh_pi_neq!(level, f_current,
                          cx_gpu, cy_gpu, cz_gpu, w_gpu, backend;
                          shift_x::Float32=0.0f0,
                          shift_y::Float32=0.0f0,
                          shift_z::Float32=0.0f0,
                          # 100-MLUPS item 2b: device Int32 block list ⇒ refresh
                          # only those blocks (restrict-side: only covered
                          # parents change). nothing ⇒ dense legacy.
                          covered_band=nothing,
                          # 2026-06-13: pi_neq_scratch (= the staging's
                          # parent_pi_neq_old) is read ONLY by this level's
                          # CHILDREN's halo machinery — a LEAF level's scratch
                          # copy is dead. Callers pass has_children.
                          copy_scratch::Bool=true)
    n_blocks = length(level.active_block_coords)
    if n_blocks == 0 || !has_pi_neq(level)
        return
    end
    band_n = covered_band === nothing ? n_blocks : length(covered_band)

    # OpTiming (2026-06-12): the post-level-step refresh (63×/L1-step dense)
    # was invisible to PhaseTiming — wire the dead _T_OP_PINEQ bucket. Only
    # the dense path is timed here; the banded restrict-side call is already
    # inside the restrict bracket.
    _tpq = 0.0
    if _PHASE_TIMING && covered_band === nothing
        KernelAbstractions.synchronize(backend); _tpq = time()
    end

    # Preserve previous PI for AMR temporal halo reconstruction. Reuse the
    # existing scratch buffer to avoid adding another full tensor to VRAM.
    if copy_scratch && length(level.pi_neq_scratch) == length(level.pi_neq)
        if covered_band === nothing
            copy_kernel! = copy_pi_neq_kernel!(backend)
            copy_kernel!(
                level.pi_neq_scratch, level.pi_neq,
                ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
            )
        else
            copy_banded! = copy_pi_neq_banded_kernel!(backend)
            copy_banded!(
                level.pi_neq_scratch, level.pi_neq, covered_band,
                ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, band_n)
            )
        end
    end

    lattice_is_d3q39 = (LATTICE_TYPE == :d3q39) ? Int32(1) : Int32(0)
    comp_corr        = COMPRESSIBILITY_CORRECTION ? Int32(1) : Int32(0)
    cs2_use = Float32(CS2_PHYSICS)

    kernel! = compute_pi_neq_kernel!(backend)
    kernel!(
        level.pi_neq, f_current,
        level.rho, level.vel, level.obstacle,
        cx_gpu, cy_gpu, cz_gpu, w_gpu,
        N_DIRS, cs2_use, lattice_is_d3q39, comp_corr,
        shift_x, shift_y, shift_z,
        level.block_shift,
        (LOCAL_LATTICE_SHIFT_ENABLED && lattice_is_d3q39 == Int32(1)) ? Int32(1) : Int32(0),
        covered_band,
        ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, band_n)
    )
    # PERF (2026-06-04, deep-sync workflow): removed host synchronize — this fires
    # ~94×/step (63 post-level-step + 31 restrict-side), the single largest source
    # of host barriers. level.pi_neq is consumed ONLY on the GPU by the next
    # collision kernel (variable_sigma / variable_nu_sgs read pi_neq_in) and the
    # restriction stress carrier; no host read of pi_neq exists in the hot path.
    # KernelAbstractions auto-orders that dependent same-backend kernel. The
    # gated filter_pi_neq! path keeps its own syncs (real pi_neq↔scratch race).
    # Byte-identical on the single-GPU same-backend path.
    if _PHASE_TIMING && covered_band === nothing
        KernelAbstractions.synchronize(backend); _T_OP_PINEQ[] += time() - _tpq
    end
end


# ──────────────────────────────────────────────────────────────────────────────
# Lagrava q-direction filter on Π_neq (Paper-quality item 5).
#
# Lagrava 2012 Eq. 33: a low-pass box filter is applied to f_neq on the FINE
# grid immediately before fine→coarse decimation. Unfiltered, fine-scale content
# that the coarse grid cannot represent (sub-grid turbulence, numerical noise)
# aliases upscale at the restriction and destabilises high-Re AMR. Lagrava
# prescribes averaging over the q lattice-direction neighbours.
#
# Here we apply the equivalent on the MOMENT carrier Π_neq (the 6-component
# symmetric stress tensor), which is what LUDWIG now transfers across levels
# (item 1). A 3×3×3 spatial box filter is close to Lagrava's q-direction
# average and is much easier to implement with block-local access. ρ, u, T are
# NOT filtered (Lagrava §3.3 explicitly: filtering macros over-dissipates).
#
# Block-local: crossing block boundaries falls back to the cell's own value
# (safe no-op that keeps the filter well-defined at the outer edge of a block).
# For interior cells all 27 neighbours are in-block. The effect at block
# interiors is the full 27-point box filter; at block edges the filter has
# reduced support (weighted by the valid-neighbour count) — this is the same
# pattern as valid-cell-weighted trilinear we use in prolongation.
# ──────────────────────────────────────────────────────────────────────────────
@kernel function filter_pi_neq_kernel!(
    pi_neq_out,                    # (BS,BS,BS,n_blocks,6) — filtered output
    pi_neq_in,                     # (BS,BS,BS,n_blocks,6) — raw input
    obstacle_arr,
    block_pointer,
    map_x,
    map_y,
    map_z,
    dim_x::Int32,
    dim_y::Int32,
    dim_z::Int32,
    block_size::Int32
)
    lx, ly, lz, b_idx = @index(Global, NTuple)
    bs = block_size

    is_obs = obstacle_arr[lx, ly, lz, b_idx]
    gx = (map_x[b_idx] - Int32(1)) * bs + lx
    gy = (map_y[b_idx] - Int32(1)) * bs + ly
    gz = (map_z[b_idx] - Int32(1)) * bs + lz
    nx_tot = dim_x * bs
    ny_tot = dim_y * bs
    nz_tot = dim_z * bs

    # 3×3×3 box filter over the level-wide active cells.
    #
    # Important AMR-detail: when a neighbour falls outside the active level
    # (or into an obstacle), we do NOT drop that sample from the average.
    # Dropping it turns the filter into a one-sided stencil near refinement-box
    # faces, which biases the filtered Π_neq exactly where the fine→coarse
    # transfer is most sensitive and shows up as faint box-aligned density
    # bands in schlieren. Instead, missing samples fall back to the center-cell
    # value so the stencil keeps a constant support and approaches the boundary
    # in a bias-free way.
    @inbounds for comp in Int32(1):Int32(6)
        if is_obs
            pi_neq_out[lx, ly, lz, b_idx, comp] = pi_neq_in[lx, ly, lz, b_idx, comp]
        else
            accum = 0.0f0
            center_val = pi_neq_in[lx, ly, lz, b_idx, comp]
            for dz in Int32(-1):Int32(1)
                ngz = gz + dz
                for dy in Int32(-1):Int32(1)
                    ngy = _pwrap_y(gy + dy, ny_tot)   # periodic-Y AMR (identity when off)
                    for dx in Int32(-1):Int32(1)
                        ngx = gx + dx

                        sample_val = center_val
                        if ngx >= Int32(1) && ngx <= nx_tot &&
                           ngy >= Int32(1) && ngy <= ny_tot &&
                           ngz >= Int32(1) && ngz <= nz_tot
                            nbx = div(ngx - Int32(1), bs) + Int32(1)
                            nby = div(ngy - Int32(1), bs) + Int32(1)
                            nbz = div(ngz - Int32(1), bs) + Int32(1)
                            nb_idx = block_pointer[nbx, nby, nbz]
                            if nb_idx > Int32(0)
                                nlx = mod(ngx - Int32(1), bs) + Int32(1)
                                nly = mod(ngy - Int32(1), bs) + Int32(1)
                                nlz = mod(ngz - Int32(1), bs) + Int32(1)
                                if !obstacle_arr[nlx, nly, nlz, nb_idx]
                                    sample_val = pi_neq_in[nlx, nly, nlz, nb_idx, comp]
                                end
                            end
                        end

                        accum += sample_val
                    end
                end
            end
            pi_neq_out[lx, ly, lz, b_idx, comp] =
                accum / 27.0f0
        end
    end
end

@kernel function copy_pi_neq_kernel!(dst, src)
    lx, ly, lz, b_idx = @index(Global, NTuple)
    @inbounds for comp in Int32(1):Int32(6)
        dst[lx, ly, lz, b_idx, comp] = src[lx, ly, lz, b_idx, comp]
    end
end

# FUSED-Π (2026-06-13): module-level query of the fuse switch + a scratch-
# only rotation helper. The switch const lives in timestep.jl (included
# earlier in the dependency chain); guard with @isdefined for load order.
#
# 2026-06-24 (post-D3Q19 retirement): the ONLY remaining producer of a fused
# level.pi_neq is the SVV epilogue (no collision kernel writes pi_neq inline now —
# neither cumulant nor entropic_d3q39). So fused-Π is only truly "active" (i.e. the
# post-step refresh_pi_neq! may be skipped because pi_neq was already written) when
# SVV runs. With SVV off — the common D3Q27 variable_nu_sgs case — this returns false
# so the refresh runs and pi_neq is fresh for the next step's adaptive viscosity.
@inline function _d3q19_fuse_pineq_active()
    _fuse = (@isdefined _D3Q19_FUSE_PINEQ) && _D3Q19_FUSE_PINEQ
    _svv  = ((@isdefined SVV_YZ_ENABLED) && SVV_YZ_ENABLED) ||
            ((@isdefined SVV_X_ENABLED)  && SVV_X_ENABLED)
    return _fuse && _svv
end

# Rotate pi_neq_old ← pi_neq WITHOUT recomputing pi_neq (the fused kernels do
# that). Used by the fused path's pre-timestep rotation. No-op when the
# scratch buffer isn't allocated to full size.
function copy_pineq_scratch_only!(level, backend)
    n_blocks = length(level.active_block_coords)
    (n_blocks == 0 || !has_pi_neq(level)) && return
    if length(level.pi_neq_scratch) == length(level.pi_neq)
        copy_kernel! = copy_pi_neq_kernel!(backend)
        copy_kernel!(
            level.pi_neq_scratch, level.pi_neq,
            ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
        )
    end
    return
end

# Banded sibling for the restrict-side refresh (100-MLUPS item 2b): copies
# only the listed blocks. The shared dense kernel above stays untouched.
@kernel function copy_pi_neq_banded_kernel!(dst, src, covered_list)
    lx, ly, lz, b_raw = @index(Global, NTuple)
    @inbounds begin
        b_idx = covered_list[b_raw]
        for comp in Int32(1):Int32(6)
            dst[lx, ly, lz, b_idx, comp] = src[lx, ly, lz, b_idx, comp]
        end
    end
end

"""
Apply the Lagrava q-direction low-pass filter to `level.pi_neq` in place.

Writes to `level.pi_neq_scratch` first to avoid read/write races, then copies
scratch back to pi_neq. Called after refresh_pi_neq! and before any cross-level
restriction pulls from this level's Π_neq. No-op when the field is a stub.
"""
function filter_pi_neq!(level, backend)
    n_blocks = length(level.active_block_coords)
    (n_blocks == 0 || !has_pi_neq(level)) && return
    # Stub scratch check: same size as pi_neq when both are allocated
    if length(level.pi_neq_scratch) != length(level.pi_neq); return; end

    filter_kernel! = filter_pi_neq_kernel!(backend)
    filter_kernel!(
        level.pi_neq_scratch, level.pi_neq,
        level.obstacle,
        level.block_pointer,
        level.map_x,
        level.map_y,
        level.map_z,
        Int32(size(level.block_pointer, 1)),
        Int32(size(level.block_pointer, 2)),
        Int32(size(level.block_pointer, 3)),
        Int32(BLOCK_SIZE),
        ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    )
    KernelAbstractions.synchronize(backend)

    copy_kernel! = copy_pi_neq_kernel!(backend)
    copy_kernel!(
        level.pi_neq, level.pi_neq_scratch,
        ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    )
    KernelAbstractions.synchronize(backend)
end


function compute_pi_neq_for_level!(level, f_current,
                                   cx_gpu, cy_gpu, cz_gpu, w_gpu, backend;
                                   shift_x::Float32=0.0f0,
                                   shift_y::Float32=0.0f0,
                                   shift_z::Float32=0.0f0)
    n_blocks = length(level.active_block_coords)
    if n_blocks == 0
        return KernelAbstractions.zeros(backend, Float32, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, 0, 6)
    end

    pi_neq = KernelAbstractions.zeros(backend, Float32,
                                      BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks, 6)

    lattice_is_d3q39 = (LATTICE_TYPE == :d3q39) ? Int32(1) : Int32(0)
    comp_corr        = COMPRESSIBILITY_CORRECTION ? Int32(1) : Int32(0)

    # For D3Q39 the polynomial equilibrium must use cs²=2/3; for D3Q27 use 1/3.
    # CS2_PHYSICS is set to match the lattice at startup, so it is correct here.
    cs2_use = Float32(CS2_PHYSICS)

    kernel! = compute_pi_neq_kernel!(backend)
    kernel!(
        pi_neq, f_current,
        level.rho, level.vel, level.obstacle,
        cx_gpu, cy_gpu, cz_gpu, w_gpu,
        N_DIRS, cs2_use, lattice_is_d3q39, comp_corr,
        shift_x, shift_y, shift_z,
        level.block_shift,
        (LOCAL_LATTICE_SHIFT_ENABLED && lattice_is_d3q39 == Int32(1)) ? Int32(1) : Int32(0),
        nothing,
        ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    )
    KernelAbstractions.synchronize(backend)
    return pi_neq
end
