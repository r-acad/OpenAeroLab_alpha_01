# ==============================================================================
# NSCBC OUTFLOW BOUNDARY — Poinsot-Lele characteristic BC for LBM (SOTA A4)
# ==============================================================================
# Implements the Navier-Stokes Characteristic Boundary Condition for the +x
# outlet face of LUDWIG's L1 domain. Adapted to LBM per:
#   Poinsot & Lele 1992, JCP 101, 104     (original NSCBC for compressible NS)
#   Rudy & Strikwerda 1980, JCP 36, 55    (LODI relaxation constant σ ≈ 0.27)
#   Heubes, Bartel & Ehrhardt 2014, JCAM 262, 51
#     "Characteristic boundary conditions in the lattice Boltzmann method"
#   Jin, Seo & Yoo 2015, JCP 290, 232
#     "Two-dimensional characteristic boundary conditions for open boundaries
#      in the lattice Boltzmann methods"
#
# Why NSCBC (vs. the existing sponge outlet):
#   The sponge outlet works by adding a relaxation term that drags the solution
#   toward freestream. It damps incoming disturbances BUT ALSO reflects acoustic
#   energy back into the domain and requires a thick (8-20% chord) zone that
#   wastes useful cells. Every open-source LBM solver (OpenLB, Palabos,
#   waLBerla) uses sponges; none ships a true characteristic outlet. This is a
#   SOTA gap across the field.
#
# Algorithm (subsonic + supersonic-aware):
#   For each face cell at x = x_max of L1:
#     1. Read interior state (one cell upstream) and face state.
#     2. Compute streamwise gradients via 1st-order one-sided FD.
#     3. Compute local sound speed c, Mach number M = u_x/c.
#     4. Compute characteristic amplitudes L_1..L_5:
#          L_1 = (u-c)·[∂p/∂x − ρc·∂u/∂x]    incoming acoustic
#          L_2 = u·[c²·∂ρ/∂x − ∂p/∂x]         entropy (advected)
#          L_3 = u·∂v/∂x                       tangential v
#          L_4 = u·∂w/∂x                       tangential w
#          L_5 = (u+c)·[∂p/∂x + ρc·∂u/∂x]    outgoing acoustic
#     5. SUBSONIC outflow (0 < u < c): L_1 is INCOMING; all others OUTGOING.
#        Replace L_1 with the LODI relaxation target:
#          L_1_target = K · (p − p_∞)
#        with K = σ·(1−M²)·c/L_ref, σ ≈ 0.27 (Rudy-Strikwerda).
#     6. SUPERSONIC outflow (u ≥ c): ALL characteristics outgoing → pure
#        extrapolation, no replacement of L_1.
#     7. Update face state via Euler step:
#          dρ/dt = -(1/c²)·[L_2 + 0.5·(L_5 + L_1)]
#          du/dt = -(L_5 − L_1)/(2·ρ·c)
#          dv/dt = -L_3
#          dw/dt = -L_4
#          dp/dt = -0.5·(L_5 + L_1)
#     8. New T from EOS: T_new = p_new / (ρ_new · R_eos)
#     9. Recompute f at the face from the equilibrium of the new state.
#        Optionally preserve the non-equilibrium part from the interior cell
#        (blended into f_face) so viscous stress information is not lost.
#
# Configuration:
#   advanced:
#     boundary:
#       outlet_nscbc:
#         enabled: true
#         p_target: 1.0             # lattice pressure target (= ρ_ref · cs²)
#         sigma_relax: 0.27         # Rudy-Strikwerda relaxation constant
#         neq_blend: 0.5            # blend factor for preserving interior f_neq
# ==============================================================================

using KernelAbstractions

"""
    apply_nscbc_outlet_kernel!(f_out, f_in, rho_arr, vel_arr, temp_arr,
                                obstacle, cx_arr, cy_arr, cz_arr, w_arr,
                                cs2_val, eos_R, p_target, K_relax,
                                neq_blend, n_dirs, x_face_global)

GPU kernel applied to cells at the +x outlet face of L1. Reads interior (face−1)
and face state, computes characteristic amplitudes, applies LODI relaxation
for incoming characteristic (subsonic), and reconstructs face populations
from the updated equilibrium state plus an interior-preserved f_neq blend.

Pre-conditions:
  - This kernel runs AFTER the L1 collision step has written f_in to f_out
    (we read from the already-updated f_out).
  - The face cells are those at local x = block_size in blocks with
    bx = grid_dim_x_L1 (the last block in the +x direction).
  - Currently subsonic outflow handled correctly; supersonic outflow is
    detected and triggers pure-extrapolation (no L_1 replacement).
"""
@kernel function apply_nscbc_outlet_kernel!(
    f_out, f_in,
    rho_arr, vel_arr, temp_arr,
    obstacle,                              # neighbor_table omitted: face is always in-block (x-1 inside)
    cx_arr, cy_arr, cz_arr, w_arr,
    cs2_val::Float32,
    eos_R::Float32,
    gamma_eos::Float32,                    # 2026-05-28: ratio of specific heats for true c = √(γ·R·T)
    p_target::Float32,
    K_relax::Float32,
    neq_blend::Float32,
    n_dirs::Int32,
    block_size::Int32,
    map_y, map_z,                          # 2026-05-31: block coords for lateral-edge (corner) detection
    grid_dim_y::Int32, grid_dim_z::Int32
)
    # 4D launch: (x_local, y, z, b_idx) over face blocks only.
    # The caller is responsible for filtering to outlet face blocks
    # (bx = grid_dim_x_L1). Within each face block, only cells at
    # x_local = block_size are processed (the rest of the block is
    # interior and skipped via an early return).
    x, y, z, b_idx = @index(Global, NTuple)

    @inbounds begin
        # KernelAbstractions disallows `return` inside @kernel functions.
        # Wrap the entire work in an if-block; cells that don't meet the
        # face-cell condition simply do no work.
        x_int = x - Int32(1)
        do_work = (x == block_size) &&
                  (x_int >= Int32(1)) &&
                  !obstacle[x, y, z, b_idx] &&
                  !obstacle[x_int, y, z, b_idx]

        if do_work
        # ── Corner detection (2026-05-31) ──────────────────────────────
        # An outlet face cell that ALSO sits on a lateral (y or z) domain
        # boundary is a wall-junction "corner". There the slip wall already
        # governs the tangential velocity, so the NSCBC tangential
        # characteristics (L_3 = u·dv/dx, L_4 = u·dw/dx) are computed from a
        # one-sided FD whose interior cell is itself wall-constrained — the
        # gradient is spurious and, injected every step, seeds the corner
        # hot-spot that detonates transonic bluff-body runs (Ma=0.84) by
        # ~step 150. Fix: at a corner, SUBORDINATE to the wall — zero the
        # tangential characteristics and freeze v,w at their interior values
        # (zero-gradient), keeping only the NORMAL (x) acoustic relaxation
        # that the outlet legitimately owns. Mirrors the inlet-corner
        # treatment in the flow kernel.
        gy_glob = (map_y[b_idx] - Int32(1)) * block_size + y
        gz_glob = (map_z[b_idx] - Int32(1)) * block_size + z
        ny_glob = grid_dim_y * block_size
        nz_glob = grid_dim_z * block_size
        is_corner = (gy_glob == Int32(1)) || (gy_glob == ny_glob) ||
                    (gz_glob == Int32(1)) || (gz_glob == nz_glob)

        # ── STEP 1: Read face and interior states ───────────────────────
        rho_face = rho_arr[x, y, z, b_idx]
        ux_face  = vel_arr[x, y, z, b_idx, 1]
        uy_face  = vel_arr[x, y, z, b_idx, 2]
        uz_face  = vel_arr[x, y, z, b_idx, 3]
        T_face   = temp_arr[x, y, z, b_idx]
        if !isfinite(T_face) || T_face < 0.1f0
            T_face = 1.0f0
        end

        rho_int = rho_arr[x_int, y, z, b_idx]
        ux_int  = vel_arr[x_int, y, z, b_idx, 1]
        uy_int  = vel_arr[x_int, y, z, b_idx, 2]
        uz_int  = vel_arr[x_int, y, z, b_idx, 3]
        T_int   = temp_arr[x_int, y, z, b_idx]
        if !isfinite(T_int) || T_int < 0.1f0
            T_int = T_face
        end

        # ── STEP 2: Compute pressure, sound speed, gradients ────────────
        # EOS: p = ρ · R_eos · T (when eos_R > 0 it's thermodynamic; else
        # falls back to cs² · T for the legacy linearized form).
        R_use = (eos_R > 0.0f0) ? eos_R : cs2_val
        p_face = rho_face * R_use * T_face
        p_int  = rho_int  * R_use * T_int

        # Local sound speed c = sqrt(γ·p/ρ) = sqrt(γ·R·T). For ideal gas
        # with R = (γ-1)·cv from the configured thermal block. Without the γ
        # factor (legacy path), `c` was under-estimated by sqrt(γ) ≈ 1.18 at
        # γ=1.4, which under-estimated the LODI relaxation rate by the same
        # factor and drove the visible outflow density drift observed in
        # case 015 (ρ_outlet ≈ 0.97 vs target 1.0).
        gamma_use = (gamma_eos > 0.0f0) ? gamma_eos : 1.0f0
        c_sound = sqrt(max(gamma_use * R_use * T_face, 1.0f-12))

        # One-sided FD gradients (in lattice units, dx = 1)
        drho_dx = rho_face - rho_int
        du_dx   = ux_face  - ux_int
        dv_dx   = uy_face  - uy_int
        dw_dx   = uz_face  - uz_int
        dp_dx   = p_face   - p_int

        # ── STEP 3: Compute characteristic amplitudes L_1..L_5 ──────────
        u_n = ux_face                              # normal velocity (+x is normal)
        Mach_local = u_n / max(c_sound, 1.0f-6)

        # L_1 = (u-c)·[dp/dx - ρ·c·du/dx]   incoming acoustic (for u>0)
        L_1 = (u_n - c_sound) * (dp_dx - rho_face * c_sound * du_dx)
        # L_2 = u · [c²·dρ/dx - dp/dx]      entropy / advected
        L_2 = u_n * (c_sound * c_sound * drho_dx - dp_dx)
        # L_3 = u · dv/dx                   tangential
        L_3 = u_n * dv_dx
        # L_4 = u · dw/dx                   tangential
        L_4 = u_n * dw_dx
        # Corner: subordinate to the lateral wall — drop the (spurious)
        # tangential characteristics so the corner cell only relaxes its
        # NORMAL acoustic mode. v,w are frozen at interior values below.
        if is_corner
            L_3 = 0.0f0
            L_4 = 0.0f0
        end
        # L_5 = (u+c) · [dp/dx + ρ·c·du/dx] outgoing acoustic
        L_5 = (u_n + c_sound) * (dp_dx + rho_face * c_sound * du_dx)

        # ── STEP 4: Apply LODI relaxation for incoming char (subsonic) ──
        # SUBSONIC outflow (0 < u_n < c): L_1 is INCOMING (its
        # characteristic speed u-c < 0). For supersonic outflow (u_n ≥ c)
        # all characteristics outgoing → no replacement (L_1 keeps its
        # computed value, which is what extrapolation gives).
        if Mach_local < 1.0f0 && u_n > 0.0f0
            # Rudy-Strikwerda LODI: L_1 = σ·(1-M²)·c/L_ref · (p - p_∞).
            # 2026-05-29 bug fix — the per-cell `(1-M²)` factor was missing
            # from the caller-supplied K_relax (caller only computes the
            # freestream-evaluated σ·c_ref/L_ref). At Ma_local=0.84 the
            # missing factor over-drove relaxation by a factor of ~3.4,
            # creating high-frequency standing waves visible upstream of
            # the outlet that fed corruption into the L1↔L2 AMR coupling
            # and ultimately blew the run up by step ~280 in case 030.
            # Mach floor at 0.05 prevents K → 0 in nearly-stagnant cells;
            # clamp to 0.95 keeps relaxation bounded near the sonic line.
            Mach_clamped = clamp(Mach_local, 0.05f0, 0.95f0)
            mach_correction = (1.0f0 - Mach_clamped * Mach_clamped)
            L_1 = K_relax * mach_correction * (p_face - p_target)
        end

        # ── STEP 5: Time derivatives of conservative variables ──────────
        # d_1 controls density; standard Poinsot-Lele decomposition.
        d_1 = (1.0f0 / (c_sound * c_sound)) * (L_2 + 0.5f0 * (L_5 + L_1))
        d_2 = 0.5f0 * (L_5 + L_1)
        d_3 = (L_5 - L_1) / (2.0f0 * rho_face * c_sound + 1.0f-12)
        # tangential char amplitudes directly give the v, w time derivatives

        # ── STEP 6: Euler update (dt = 1 in lattice units) ─────────────
        # 2026-05-29 — Per-step magnitude cap on Δρ and Δu. The LODI Euler
        # step is unconditionally stable only when |d_1|, |d_2|, |d_3| are
        # small relative to the face state. For a transient where the
        # upstream density gradient is large, |d_1| can exceed ρ_face/2 in a
        # single step — driving ρ_new negative or beyond ten times target.
        #
        # 2026-05-29 (v2) — The original 5% cap was TOO TIGHT. It prevented the
        # single-step blow-up, but it also throttled the outlet's steady relief
        # of an inlet-driven mass pile-up: at Ma=0.84 the inlet flux raised the
        # upstream density faster than a 5%/step outlet could bleed it, so the
        # domain density gradient GREW monotonically (4.5% at step 50 → 13% at
        # step 100 in case 030's density slice) — a mass pump heading for
        # divergence. Raising the density cap to 20%/step lets the LODI
        # relaxation (which is conservation-correct — it drives p→p_target)
        # actually keep up with the inlet flux, while still bounding the
        # worst-case single-step excursion well away from the ρ_new sign flip.
        # The velocity/pressure caps stay tighter (acoustic CFL) at 10%.
        # Corner cells get a tighter density cap (10% vs 20%) — they sit on the
        # steepest gradients and have the least lateral relief, so they are the
        # most prone to a single-step excursion.
        rho_cap_frac = is_corner ? 0.10f0 : 0.20f0
        d_1_capped = clamp(d_1, -rho_cap_frac * rho_face, rho_cap_frac * rho_face)
        d_2_capped = clamp(d_2, -0.10f0 * p_face,   0.10f0 * p_face)
        d_3_capped = clamp(d_3, -0.10f0 * c_sound,  0.10f0 * c_sound)
        L_3_capped = clamp(L_3, -0.10f0 * c_sound,  0.10f0 * c_sound)
        L_4_capped = clamp(L_4, -0.10f0 * c_sound,  0.10f0 * c_sound)

        rho_new = rho_face - d_1_capped
        u_new   = ux_face  - d_3_capped
        # Tangential velocities: normal NSCBC update away from the wall;
        # at a corner, freeze to the interior value (zero-gradient) so the
        # slip wall — not a spurious face gradient — sets the tangential flow.
        v_new   = is_corner ? uy_int : (uy_face - L_3_capped)
        w_new   = is_corner ? uz_int : (uz_face - L_4_capped)
        p_new   = p_face   - d_2_capped

        # Safety: clamp ρ, p, T to physical ranges (prevents collapse from
        # an over-aggressive K_relax or a transient supersonic spike).
        rho_new = clamp(rho_new, 0.1f0, 10.0f0)
        p_new   = max(p_new, 1.0f-6)

        # Recover T from updated state: T = p / (ρ · R)
        T_new = p_new / (rho_new * R_use + 1.0f-12)
        T_new = clamp(T_new, 0.1f0, 10.0f0)

        # Write updated macros back so downstream diagnostics see the
        # new state. Other levels read from these arrays for halo fills.
        rho_arr[x, y, z, b_idx]    = rho_new
        vel_arr[x, y, z, b_idx, 1] = u_new
        vel_arr[x, y, z, b_idx, 2] = v_new
        vel_arr[x, y, z, b_idx, 3] = w_new
        temp_arr[x, y, z, b_idx]   = T_new

        # ── STEP 7: Reconstruct face populations from updated state ─────
        # f_face = f_eq(ρ_new, u_new, T_new)  +  neq_blend · (f_interior − f_eq_interior)
        # The blend preserves a fraction of the interior non-equilibrium
        # content (viscous stress) so the face doesn't drop to pure
        # equilibrium (which would imply no shear and cause a discontinuity
        # in shear at the BC). Default blend factor 0.5 = moderate.
        for k in Int32(1):n_dirs
            cx_k = Float32(cx_arr[k])
            cy_k = Float32(cy_arr[k])
            cz_k = Float32(cz_arr[k])
            w_k  = Float32(w_arr[k])

            # Polynomial O(u²) equilibrium with the EOS-aware variance.
            # f_eq_k = ρ·w_k·[1 + c·u/(R·T) + (c·u)²/(2(R·T)²) − u²/(2·R·T)]
            inv_RT = 1.0f0 / max(R_use * T_new, 1.0f-12)
            cu_new = cx_k * u_new + cy_k * v_new + cz_k * w_new
            usq_new = u_new * u_new + v_new * v_new + w_new * w_new
            f_eq_new = rho_new * w_k * (1.0f0 + cu_new * inv_RT +
                                         0.5f0 * cu_new * cu_new * inv_RT * inv_RT -
                                         0.5f0 * usq_new * inv_RT)
            f_eq_new = max(f_eq_new, 1.0f-15)

            # Interior reference (one cell upstream)
            inv_RT_int = 1.0f0 / max(R_use * T_int, 1.0f-12)
            cu_int = cx_k * ux_int + cy_k * uy_int + cz_k * uz_int
            usq_int = ux_int * ux_int + uy_int * uy_int + uz_int * uz_int
            f_eq_int = rho_int * w_k * (1.0f0 + cu_int * inv_RT_int +
                                         0.5f0 * cu_int * cu_int * inv_RT_int * inv_RT_int -
                                         0.5f0 * usq_int * inv_RT_int)
            f_eq_int = max(f_eq_int, 1.0f-15)

            f_neq_int = f_in[x_int, y, z, b_idx, k] - f_eq_int

            # Final face population: equilibrium at new state + preserved
            # interior non-equilibrium content (blended).
            f_out[x, y, z, b_idx, k] = f_eq_new + neq_blend * f_neq_int
        end
        end  # close `if do_work`
    end
end

"""
    apply_nscbc_outlet!(level, backend; p_target, sigma_relax, neq_blend, eos_R)

Driver for the NSCBC outlet kernel. Launches over the L1 face blocks (those
with bx = grid_dim_x). Only the cells with local x = BLOCK_SIZE are processed
(see in-kernel early return).

This routine should be called once per parent step, AFTER the L1 collision
has updated f_out and the macros (ρ, u, T) for the cells along the outlet
face. The kernel reads the post-collision macros, computes the LODI update,
and overwrites the face populations + macros with the NSCBC-consistent values.

Identifies face blocks by their bx coordinate; falls back to no-op if no
blocks are at the +x face (e.g. multi-GPU partition without the outlet).
"""
function apply_nscbc_outlet!(level, backend,
                              cx_arr, cy_arr, cz_arr, w_arr;
                              p_target::Float32 = 1.0f0,
                              sigma_relax::Float32 = 0.27f0,
                              neq_blend::Float32 = 0.5f0,
                              eos_R::Float32 = Float32(-1.0f0),
                              gamma_eos::Float32 = Float32(-1.0f0))
    n_blocks = length(level.active_block_coords)
    n_blocks == 0 && return

    # Identify the maximum bx among active L1 blocks — the +x face.
    bx_max = Int32(0)
    for coord in level.active_block_coords
        bx = Int32(coord[1])
        if bx > bx_max
            bx_max = bx
        end
    end
    bx_max == Int32(0) && return

    # Reference length L_ref for K_relax normalization — physical domain
    # length in x in lattice units. The kernel filters internally to face
    # cells (x == block_size in face blocks), so we accept the early-return
    # overhead on non-face blocks for simplicity. A face-block-only launch
    # would build a face_block_list; v1 keeps it simple.
    L_ref_phys = Float32(level.grid_dim_x * BLOCK_SIZE)
    # 2026-05-28 bug fix — reference sound speed used in K_relax now uses
    # the thermodynamic c = √(γ·R·T_ref) when eos_R/gamma are supplied
    # (T_ref = 1 in lattice units). Previously this defaulted to cs_lattice
    # = √(2/3), which under-estimated K_relax by √γ ≈ 1.18 for air and led
    # to slow outflow pressure relaxation (visible as the ρ_outlet ≈ 0.97
    # drift in case 015 after 500 steps).
    R_for_cref = eos_R > 0.0f0 ? eos_R : Float32(2.0/3.0)
    gamma_for_cref = gamma_eos > 0.0f0 ? gamma_eos : 1.0f0
    c_ref = sqrt(gamma_for_cref * R_for_cref)
    K_relax = sigma_relax * c_ref / max(L_ref_phys, 1.0f0)

    kernel! = apply_nscbc_outlet_kernel!(backend)
    kernel!(
        level.f, level.f,                     # in-place: f_out == f_in
        level.rho, level.vel, level.temperature,
        level.obstacle,                       # neighbor_table not needed (face in-block)
        cx_arr, cy_arr, cz_arr, w_arr,
        Float32(CS2_PHYSICS),
        eos_R,
        gamma_for_cref,                       # new: γ for true sound speed
        p_target,
        K_relax,
        neq_blend,
        Int32(N_DIRS),
        Int32(BLOCK_SIZE),
        level.map_y, level.map_z,
        Int32(level.grid_dim_y), Int32(level.grid_dim_z),
        ndrange = (BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks)
    )
    KernelAbstractions.synchronize(backend)
end
