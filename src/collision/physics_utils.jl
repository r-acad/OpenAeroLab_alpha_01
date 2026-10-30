using KernelAbstractions



@inline function gpu_hash(x::Int32)::UInt32
    h = reinterpret(UInt32, x)
    h = (h ⊻ (h >> 16)) * UInt32(0x85ebca6b)
    h = (h ⊻ (h >> 13)) * UInt32(0xc2b2ae35)
    return h ⊻ (h >> 16)
end

@inline function gradient_noise(gx::Int32, gy::Int32, gz::Int32, seed::Int32)::Float32
    combined = gx * Int32(374761393) + gy * Int32(668265263) + gz * Int32(1274126177) + seed
    h = gpu_hash(combined)
    return (Float32(h & UInt32(0xFFFF)) / 32768.0f0) - 1.0f0
end

@inline _fine_to_parent_interp_coord_scalar(fine_g::Int32) = Float32(fine_g) * 0.5f0 + 0.25f0

@inline function _parent_interp_bracket_scalar(p_cont::Float32, parent_n::Int32)
    if parent_n <= Int32(1)
        return Int32(1), Int32(1), 0.0f0
    end
    i0 = clamp(floor(Int32, p_cont), Int32(1), parent_n - Int32(1))
    i1 = i0 + Int32(1)
    return i0, i1, p_cont - Float32(i0)
end


@inline function calculate_equilibrium(rho::Float32, ux::Float32, uy::Float32, uz::Float32,
                                       w_k::Float32, cx::Float32, cy::Float32, cz::Float32)
    cu = cx*ux + cy*uy + cz*uz
    usq = ux*ux + uy*uy + uz*uz
    return rho * w_k * (1.0f0 + 3.0f0*cu + 4.5f0*cu*cu - 1.5f0*usq)
end

# Compressibility-corrected equilibrium including 3rd and 4th order Hermite terms.
# On the D3Q27 lattice these terms are representable and extend the valid Mach
# number range from ~0.3 to ~0.6–0.9.
#
# Using the Hermite expansion (Coreixas et al. 2017, Shan et al. 2006):
# f_eq = ρ w_k [ 1 + (c·u)/cs² + ((c·u)² − cs²u²)/(2cs⁴)
#                + ((c·u)³ − 3cs²(c·u)u²)/(6cs⁶)
#                + ((c·u)⁴ − 6cs²(c·u)²u² + 3cs⁴u⁴)/(24cs⁸) ]
#
# Expanded with cs² = 1/3:
#   2nd order:  +3(c·u)  +4.5(c·u)²  −1.5u²
#   3rd order:  +4.5(c·u)³  −4.5(c·u)u²
#   4th order:  +3.375(c·u)⁴  −6.75(c·u)²u²  +1.125u⁴
# GRADIENT/VELOCITY LIMITER for the O(u⁴) compressibility correction (2026-06-23).
# The 3rd+4th-order Hermite terms (which give the Ma>0.3 accuracy) scale as u³,u⁴ and
# BLOW UP where the local velocity spikes — e.g. at a sharp leading edge that is
# severely under-resolved (ONERA M6 at Re=2.3e7, τ→0.5: the cumulant collision is
# near-inviscid and the LE velocity overshoots, the O(u⁴) terms amplify it, and the
# near-body finest level explodes). This limiter smoothly blends the high-order terms
# toward 0 (= the stable O(u²) equilibrium) as |u|² rises past a safe band, so full
# O(u⁴) accuracy is kept in the smooth bulk (where |u|≈u_lattice) and only the
# velocity SPIKES that cause the blow-up fall back to O(u²). Gated by
# _COMPRESSIBLE_HIORDER_LIMIT (env LUDWIG_OU4_LIMITER); default OFF ⇒ byte-identical.
# Band: φ=1 for |u|²≤_OU4_LIM_LO², ramps (smoothstep) to φ=0 at |u|²≥_OU4_LIM_HI².
const _COMPRESSIBLE_HIORDER_LIMIT = get(ENV, "LUDWIG_OU4_LIMITER", "0") != "0"
const _OU4_LIM_LO = Float32(parse(Float64, get(ENV, "LUDWIG_OU4_LIMITER_LO", "0.20")))
const _OU4_LIM_HI = Float32(parse(Float64, get(ENV, "LUDWIG_OU4_LIMITER_HI", "0.40")))

@inline function calculate_equilibrium_compressible(rho::Float32, ux::Float32, uy::Float32, uz::Float32,
                                                     w_k::Float32, cx::Float32, cy::Float32, cz::Float32)
    cu  = cx*ux + cy*uy + cz*uz
    usq = ux*ux + uy*uy + uz*uz
    cu2 = cu * cu
    cu3 = cu2 * cu
    cu4 = cu2 * cu2
    usq2 = usq * usq

    # Standard 2nd-order terms (always — these are the stable O(u²) base)
    val = 1.0f0 + 3.0f0*cu + 4.5f0*cu2 - 1.5f0*usq

    # 3rd + 4th-order Hermite (the O(u⁴) compressibility correction).
    ho = (4.5f0*cu3 - 4.5f0*cu*usq) +
         (3.375f0*cu4 - 6.75f0*cu2*usq + 1.125f0*usq2)

    if _COMPRESSIBLE_HIORDER_LIMIT
        # Smoothstep blend factor φ(|u|): 1 below LO², 0 above HI².
        lo2 = _OU4_LIM_LO * _OU4_LIM_LO
        hi2 = _OU4_LIM_HI * _OU4_LIM_HI
        t = (usq - lo2) / (hi2 - lo2)
        t = t < 0.0f0 ? 0.0f0 : (t > 1.0f0 ? 1.0f0 : t)
        phi = 1.0f0 - t*t*(3.0f0 - 2.0f0*t)   # smoothstep, =1 at t=0, =0 at t=1
        val += phi * ho
    else
        val += ho
    end

    return rho * w_k * val
end

# Dispatch: pick compressible or standard equilibrium based on flag
@inline function calculate_equilibrium_auto(rho::Float32, ux::Float32, uy::Float32, uz::Float32,
                                             w_k::Float32, cx::Float32, cy::Float32, cz::Float32,
                                             compressible::Int32)
    if compressible == Int32(1)
        return calculate_equilibrium_compressible(rho, ux, uy, uz, w_k, cx, cy, cz)
    else
        return calculate_equilibrium(rho, ux, uy, uz, w_k, cx, cy, cz)
    end
end


# Thermal equilibrium for the energy/temperature DDF (Phase 2).
# Advection-diffusion model: g_eq_k = w_k * T * (1 + 3*(c_k . u))
# Recovers: dT/dt + u.grad(T) = kappa * laplacian(T)
@inline function calculate_thermal_equilibrium(T::Float32, ux::Float32, uy::Float32, uz::Float32,
                                                w_k::Float32, cx::Float32, cy::Float32, cz::Float32)
    cu = cx*ux + cy*uy + cz*uz
    return w_k * T * (1.0f0 + 3.0f0*cu)
end


# ==============================================================================
# D3Q39 NUMERICAL EQUILIBRIUM — 13-Moment Newton-Raphson Solver
# ==============================================================================
# Implements the entropic equilibrium for the D3Q39 multi-speed lattice
# (Frapolli, Chikatamarla, Karlin, Phys. Rev. E 2015).
#
# The equilibrium has the form:
#   f_eq_i = ρ · w_i · exp(Σ_{M=1}^{13} λ_M · a_M(c_i))
#
# where a_M(c_i) are 13 Hermite polynomial basis functions evaluated at
# lattice velocity c_i, and λ_M are Lagrange multipliers determined by
# enforcing 13 moment constraints from the Maxwell-Boltzmann distribution:
#
#   Constraint 1:      mass           ρ
#   Constraints 2-4:   momentum       ρ·u_x, ρ·u_y, ρ·u_z
#   Constraints 5-10:  stress tensor  ρ(cs²+u_α·u_β) for αβ = xx,yy,zz,xy,xz,yz
#   Constraints 11-13: energy flux    ρ·u_α((D+2)cs²/2 + u²/2) for α = x,y,z
#                      = ρ·u_α(2.5·cs² + 0.5·u²)     [D=3]
#
# The Newton-Raphson system uses MVector{13}/MMatrix{13,13} from StaticArrays
# for GPU-compatible stack allocation (no heap). The 13×13 linear system is
# solved via Gaussian elimination with partial pivoting.
#
# Initial guess: derived from the 2nd-order polynomial equilibrium.
# Convergence: typically 3-5 iterations for Ma < 2.
# Fallback: if Newton diverges (>max_iter), returns polynomial equilibrium.
# ==============================================================================

# 13 basis functions a_M(c) for the entropic equilibrium:
#   M=1:  1                    (mass)
#   M=2:  c_x                  (x-momentum)
#   M=3:  c_y                  (y-momentum)
#   M=4:  c_z                  (z-momentum)
#   M=5:  c_x² - cs²           (xx stress, traceless)
#   M=6:  c_y² - cs²           (yy stress, traceless)
#   M=7:  c_z² - cs²           (zz stress, traceless)
#   M=8:  c_x·c_y              (xy stress)
#   M=9:  c_x·c_z              (xz stress)
#   M=10: c_y·c_z              (yz stress)
#   M=11: c_x(|c|² - (D+2)cs²) (x energy flux)
#   M=12: c_y(|c|² - (D+2)cs²) (y energy flux)
#   M=13: c_z(|c|² - (D+2)cs²) (z energy flux)

@inline function compute_basis_d3q39(cx_k::Float32, cy_k::Float32, cz_k::Float32, cs2::Float32)
    # Returns MVector{13, Float32} of basis functions for direction k
    a = MVector{13, Float32}(undef)
    csq = cx_k*cx_k + cy_k*cy_k + cz_k*cz_k  # |c|²
    dplus2_cs2 = 5.0f0 * cs2   # (D+2)·cs² = 5·cs² for D=3

    a[1]  = 1.0f0                          # mass
    a[2]  = cx_k                           # x-momentum
    a[3]  = cy_k                           # y-momentum
    a[4]  = cz_k                           # z-momentum
    a[5]  = cx_k*cx_k - cs2                # xx stress (traceless)
    a[6]  = cy_k*cy_k - cs2                # yy stress (traceless)
    a[7]  = cz_k*cz_k - cs2                # zz stress (traceless)
    a[8]  = cx_k*cy_k                      # xy stress
    a[9]  = cx_k*cz_k                      # xz stress
    a[10] = cy_k*cz_k                      # yz stress
    a[11] = cx_k * (csq - dplus2_cs2)      # x energy flux
    a[12] = cy_k * (csq - dplus2_cs2)      # y energy flux
    a[13] = cz_k * (csq - dplus2_cs2)      # z energy flux

    return a
end

# ──────────────────────────────────────────────────────────────────────
# Frapolli DDF thermal-equilibrium g-coefficient (EOS-consistent)
# ──────────────────────────────────────────────────────────────────────
# The Frapolli two-population thermal model uses g_eq_k = (Cv − D·R/2)·T·f_eq_k
# where R is the SAME EOS pressure coefficient the flow/thermal collision
# kernels use. With eos_thermodynamic ON, R = (γ−1)·cv (= eos_R_flow); with it
# OFF (or D3Q27), eos_R_flow arrives as the −1 sentinel and the coefficient
# falls back to the lattice cs²_thermal (legacy linearized-EOS behavior).
# Centralizing the expression here prevents the recurring AMR-coupling bug
# class where a coeff_g site is left on the hard-coded linearized cs² form,
# producing an EOS-inconsistent g variance across refinement interfaces.
# The `> 0` guard makes this byte-identical to the legacy path when
# eos_R_flow ≤ 0, so non-thermodynamic / D3Q27 cases are unaffected.
@inline function frapolli_g_coeff(cv_thermal::Float32, eos_R_flow::Float32, cs2_thermal::Float32)
    R_g = eos_R_flow > 0.0f0 ? eos_R_flow : cs2_thermal
    return cv_thermal - 1.5f0 * R_g
end

# Compute target moments from Maxwell-Boltzmann for given (rho, u):
#   m[1]    = ρ
#   m[2-4]  = ρ·u_α
#   m[5-7]  = ρ·u_α² (traceless: ρ·u_α² since we subtracted cs² in basis)
#   m[8-10] = ρ·u_α·u_β
#   m[11-13] = ρ·u_α · (u² / 2)  (from energy flux after basis transform)
#
# The target moment for basis function a_M is: Σ_i f_eq_i · a_M(c_i)
# For the Maxwell-Boltzmann, with our basis choice:
#   <a_5> = <c_x² - cs²> = ρ·(u_x² + cs²·(T-1))
#   <a_11> = <c_x(|c|² - 5cs²)> = ρ·u_x·(u² + 5·cs²·(T-1))
# Phase 4: T parameter for thermal EOS. When T=1 (default), reduces to athermal.
@inline function compute_target_moments_d3q39(rho::Float32, ux::Float32, uy::Float32, uz::Float32,
                                               T::Float32=1.0f0, _cs2::Float32=Float32(2/3),
                                               eos_R::Float32=Float32(2/3))
    # `_cs2` retained for API/positional-arg stability across existing callers;
    # the variance coefficient is now `eos_R` (see docstring). For D3Q39 the
    # numerical default cs² = 2/3 = default eos_R, so legacy callers that pass
    # only cs2 still get the linearized behavior.
    # SOTA A3 (2026-05-27): `eos_R` is the THERMODYNAMIC pressure coefficient
    # used for the T-deviation contribution to the variance: p = ρ·R·T.
    # - LEGACY (linearized EOS): pass `eos_R = cs2` (default) — recovers the
    #   prior behavior where the thermal contribution implicitly assumes
    #   R = cs² (i.e. (γ-1)·cv = cs², which for cs²=2/3 and γ=1.4 means
    #   cv = 5/3 implicitly, regardless of the config `thermal.cv` value).
    # - TRUE (thermodynamic EOS): pass `eos_R = (γ-1)·cv` — the proper ideal
    #   gas pressure: p = (γ-1)·ρ·cv·T. The lattice's discrete cs² remains
    #   the variance basis (used in the polynomial init guess at line ~456),
    #   but the moment targets use the actual thermodynamic R.
    # The split is deliberate: cs² is a property of the lattice (acoustic
    # speed at lattice scale), R is a property of the fluid (EOS). Conflating
    # them is what makes the linearized form wrong at large |T-1|.
    m = MVector{13, Float32}(undef)
    usq = ux*ux + uy*uy + uz*uz
    dT = T - 1.0f0                         # deviation from isothermal

    m[1]  = rho                            # mass
    m[2]  = rho * ux                       # x-momentum
    m[3]  = rho * uy                       # y-momentum
    m[4]  = rho * uz                       # z-momentum
    # Stress: <c_α² - cs²> = ρ·(u_α² + R·(T-1))
    m[5]  = rho * (ux * ux + eos_R * dT)  # <c_x² - cs²>
    m[6]  = rho * (uy * uy + eos_R * dT)  # <c_y² - cs²>
    m[7]  = rho * (uz * uz + eos_R * dT)  # <c_z² - cs²>
    m[8]  = rho * ux * uy                  # <c_x c_y> = ρ u_x u_y (unchanged by T)
    m[9]  = rho * ux * uz                  # <c_x c_z> = ρ u_x u_z
    m[10] = rho * uy * uz                  # <c_y c_z> = ρ u_y u_z
    # Energy flux: <c_α(|c|²-5cs²)> = ρ·u_α·(u² + 5·R·(T-1))
    local eflux_corr = usq + 5.0f0 * eos_R * dT
    m[11] = rho * ux * eflux_corr          # <c_x(|c|²-5cs²)>
    m[12] = rho * uy * eflux_corr          # <c_y(|c|²-5cs²)>
    m[13] = rho * uz * eflux_corr          # <c_z(|c|²-5cs²)>

    return m
end

# ──────────────────────────────────────────────────────────────────────
# Main D3Q39 numerical equilibrium function
# ──────────────────────────────────────────────────────────────────────
# Computes f_eq[1:39] for given (rho, ux, uy, uz) using 3-group
# over-relaxed coordinate descent on 13 Lagrange multipliers.
#
# The entropic form is: f_eq_i = rho * w_i * exp(Σ_M λ_M * a_M(c_i))
#
# 13 moments matched: mass(1) + momentum(3) + full stress tensor(6)
# + energy flux(3) = 13.
#
# 3-group structure (Jacobi within groups, Gauss-Seidel between):
#   Group A: mass + momentum (moments 1-4)
#   Group B: stress tensor (moments 5-10)
#   Group C: energy flux (moments 11-13)
#
# "Recompute" approach: each group recomputes f_eq from current λ via exp(),
# eliminating separate apply loops. Only 3 inner loops + 1 output loop = 4 total.
# This avoids MVector{39} storage during iteration and reduces LLVM loop count.
# With ω=1.5, converges in 40-100 sweeps depending on flow conditions.
# ──────────────────────────────────────────────────────────────────────

@noinline function calculate_equilibrium_numerical_d3q39!(
    f_eq::MVector{39, Float32},
    rho::Float32, ux::Float32, uy::Float32, uz::Float32,
    cx_arr, cy_arr, cz_arr, w_arr,
    cs2::Float32,
    max_iter::Int32,
    T::Float32=1.0f0,
    shift_x::Float32=0.0f0, shift_y::Float32=0.0f0, shift_z::Float32=0.0f0,
    eos_R::Float32=Float32(-1.0f0)        # SOTA A3: -1 sentinel → fall back to cs2 (legacy)
)
    n = Int32(39)
    F_FLOOR_EQ = Float32(1e-15)
    OMEGA = Float32(1.5)
    # Machine-epsilon-based tolerance: 13 moments × 39 dirs accumulation error
    TOL_SQ = Float32(507) * eps(Float32) * eps(Float32) * rho * rho

    # SOTA A3: eos_R < 0 sentinel → use cs2 (legacy linearized EOS).
    # Explicit eos_R = (γ-1)·cv activates thermodynamic EOS.
    eos_R_eff = (eos_R < 0.0f0) ? cs2 : eos_R

    usq = ux * ux + uy * uy + uz * uz
    dT = T - 1.0f0
    dplus2_cs2 = 5.0f0 * cs2

    # Target moments from Maxwell-Boltzmann (eos_R_eff parameterizes the
    # thermodynamic R that contributes to the variance; defaults to cs2)
    mt1  = rho
    mt2  = rho * ux
    mt3  = rho * uy
    mt4  = rho * uz
    mt5  = rho * (ux * ux + eos_R_eff * dT)
    mt6  = rho * (uy * uy + eos_R_eff * dT)
    mt7  = rho * (uz * uz + eos_R_eff * dT)
    mt8  = rho * ux * uy
    mt9  = rho * ux * uz
    mt10 = rho * uy * uz
    # AUDIT FIX (2026-06-03): energy-flux variance must use the same
    # thermodynamic R as the stress moments (mt5-mt7) above. Previously this
    # line used `cs2`, making this in-place solver disagree with both its own
    # stress targets and the NTuple variant `solve_13mom_lambdas_d3q39` (which
    # uses eos_R_eff). With the legacy sentinel (eos_R<0 → eos_R_eff=cs2) the
    # value is unchanged; the fix only matters when eos_thermodynamic is active
    # at T≠1, where the prior code injected an inconsistent energy flux.
    eflux_corr = usq + 5.0f0 * eos_R_eff * dT
    mt11 = rho * ux * eflux_corr
    mt12 = rho * uy * eflux_corr
    mt13 = rho * uz * eflux_corr

    # Initial guess for λ from polynomial equilibrium
    inv_cs2 = 1.0f0 / cs2
    inv_cs4 = inv_cs2 * inv_cs2

    la1  = -0.5f0 * usq * inv_cs2
    la2  = ux * inv_cs2
    la3  = uy * inv_cs2
    la4  = uz * inv_cs2
    la5  = 0.5f0 * inv_cs4 * ux * ux
    la6  = 0.5f0 * inv_cs4 * uy * uy
    la7  = 0.5f0 * inv_cs4 * uz * uz
    la8  = inv_cs4 * ux * uy
    la9  = inv_cs4 * ux * uz
    la10 = inv_cs4 * uy * uz
    la11 = Float32(0)
    la12 = Float32(0)
    la13 = Float32(0)

    # 3-group over-relaxed coordinate descent (recompute approach)
    # Each group recomputes f_eq from current λ — no stored f_eq between loops
    for sweep in Int32(1):max_iter

        # ── Group A: mass + momentum (moments 1-4) ──
        d1 = Float32(0); d2 = Float32(0); d3 = Float32(0); d4 = Float32(0)
        j1 = Float32(0); j2 = Float32(0); j3 = Float32(0); j4 = Float32(0)
        for i in Int32(1):n
            cx_i = Float32(cx_arr[i]) - shift_x
            cy_i = Float32(cy_arr[i]) - shift_y
            cz_i = Float32(cz_arr[i]) - shift_z
            csq_i = cx_i*cx_i + cy_i*cy_i + cz_i*cz_i
            efl = csq_i - dplus2_cs2
            lda = la1 + la2*cx_i + la3*cy_i + la4*cz_i +
                  la5*(cx_i*cx_i - cs2) + la6*(cy_i*cy_i - cs2) +
                  la7*(cz_i*cz_i - cs2) +
                  la8*cx_i*cy_i + la9*cx_i*cz_i + la10*cy_i*cz_i +
                  la11*cx_i*efl + la12*cy_i*efl + la13*cz_i*efl
            fi = max(rho * w_arr[i] * exp(clamp(lda, -20.0f0, 20.0f0)), F_FLOOR_EQ)
            d1 += fi;       j1 += fi
            d2 += fi*cx_i;  j2 += fi*cx_i*cx_i
            d3 += fi*cy_i;  j3 += fi*cy_i*cy_i
            d4 += fi*cz_i;  j4 += fi*cz_i*cz_i
        end
        la1 += j1 > F_FLOOR_EQ ? clamp(OMEGA * (mt1 - d1) / j1, -2.0f0, 2.0f0) : 0.0f0
        la2 += j2 > F_FLOOR_EQ ? clamp(OMEGA * (mt2 - d2) / j2, -2.0f0, 2.0f0) : 0.0f0
        la3 += j3 > F_FLOOR_EQ ? clamp(OMEGA * (mt3 - d3) / j3, -2.0f0, 2.0f0) : 0.0f0
        la4 += j4 > F_FLOOR_EQ ? clamp(OMEGA * (mt4 - d4) / j4, -2.0f0, 2.0f0) : 0.0f0

        # ── Group B: stress tensor (moments 5-10) ──
        d5 = Float32(0); d6 = Float32(0); d7 = Float32(0)
        d8 = Float32(0); d9 = Float32(0); d10 = Float32(0)
        j5 = Float32(0); j6 = Float32(0); j7 = Float32(0)
        j8 = Float32(0); j9 = Float32(0); j10 = Float32(0)
        for i in Int32(1):n
            cx_i = Float32(cx_arr[i]) - shift_x
            cy_i = Float32(cy_arr[i]) - shift_y
            cz_i = Float32(cz_arr[i]) - shift_z
            csq_i = cx_i*cx_i + cy_i*cy_i + cz_i*cz_i
            efl = csq_i - dplus2_cs2
            lda = la1 + la2*cx_i + la3*cy_i + la4*cz_i +
                  la5*(cx_i*cx_i - cs2) + la6*(cy_i*cy_i - cs2) +
                  la7*(cz_i*cz_i - cs2) +
                  la8*cx_i*cy_i + la9*cx_i*cz_i + la10*cy_i*cz_i +
                  la11*cx_i*efl + la12*cy_i*efl + la13*cz_i*efl
            fi = max(rho * w_arr[i] * exp(clamp(lda, -20.0f0, 20.0f0)), F_FLOOR_EQ)
            a5 = cx_i*cx_i - cs2; a6 = cy_i*cy_i - cs2; a7 = cz_i*cz_i - cs2
            a8 = cx_i*cy_i;       a9 = cx_i*cz_i;       a10 = cy_i*cz_i
            d5 += fi*a5;  j5 += fi*a5*a5
            d6 += fi*a6;  j6 += fi*a6*a6
            d7 += fi*a7;  j7 += fi*a7*a7
            d8 += fi*a8;  j8 += fi*a8*a8
            d9 += fi*a9;  j9 += fi*a9*a9
            d10+= fi*a10; j10+= fi*a10*a10
        end
        la5  += j5 > F_FLOOR_EQ ? clamp(OMEGA * (mt5  - d5) / j5,  -2.0f0, 2.0f0) : 0.0f0
        la6  += j6 > F_FLOOR_EQ ? clamp(OMEGA * (mt6  - d6) / j6,  -2.0f0, 2.0f0) : 0.0f0
        la7  += j7 > F_FLOOR_EQ ? clamp(OMEGA * (mt7  - d7) / j7,  -2.0f0, 2.0f0) : 0.0f0
        la8  += j8 > F_FLOOR_EQ ? clamp(OMEGA * (mt8  - d8) / j8,  -2.0f0, 2.0f0) : 0.0f0
        la9  += j9 > F_FLOOR_EQ ? clamp(OMEGA * (mt9  - d9) / j9,  -2.0f0, 2.0f0) : 0.0f0
        la10 += j10> F_FLOOR_EQ ? clamp(OMEGA * (mt10 -d10) / j10, -2.0f0, 2.0f0) : 0.0f0

        # ── Group C: energy flux (moments 11-13) ──
        d11 = Float32(0); d12 = Float32(0); d13 = Float32(0)
        j11 = Float32(0); j12 = Float32(0); j13 = Float32(0)
        for i in Int32(1):n
            cx_i = Float32(cx_arr[i]) - shift_x
            cy_i = Float32(cy_arr[i]) - shift_y
            cz_i = Float32(cz_arr[i]) - shift_z
            csq_i = cx_i*cx_i + cy_i*cy_i + cz_i*cz_i
            efl = csq_i - dplus2_cs2
            lda = la1 + la2*cx_i + la3*cy_i + la4*cz_i +
                  la5*(cx_i*cx_i - cs2) + la6*(cy_i*cy_i - cs2) +
                  la7*(cz_i*cz_i - cs2) +
                  la8*cx_i*cy_i + la9*cx_i*cz_i + la10*cy_i*cz_i +
                  la11*cx_i*efl + la12*cy_i*efl + la13*cz_i*efl
            fi = max(rho * w_arr[i] * exp(clamp(lda, -20.0f0, 20.0f0)), F_FLOOR_EQ)
            a11 = cx_i*efl; a12 = cy_i*efl; a13 = cz_i*efl
            d11 += fi*a11; j11 += fi*a11*a11
            d12 += fi*a12; j12 += fi*a12*a12
            d13 += fi*a13; j13 += fi*a13*a13
        end
        la11 += j11> F_FLOOR_EQ ? clamp(OMEGA * (mt11-d11)/ j11,-2.0f0, 2.0f0) : 0.0f0
        la12 += j12> F_FLOOR_EQ ? clamp(OMEGA * (mt12-d12)/ j12,-2.0f0, 2.0f0) : 0.0f0
        la13 += j13> F_FLOOR_EQ ? clamp(OMEGA * (mt13-d13)/ j13,-2.0f0, 2.0f0) : 0.0f0

        # Convergence check (machine-epsilon tolerance)
        r1=mt1-d1; r2=mt2-d2; r3=mt3-d3; r4=mt4-d4
        r5=mt5-d5; r6=mt6-d6; r7=mt7-d7
        r8=mt8-d8; r9=mt9-d9; r10=mt10-d10
        r11=mt11-d11; r12=mt12-d12; r13=mt13-d13
        res_sq = r1*r1+r2*r2+r3*r3+r4*r4+r5*r5+r6*r6+r7*r7+
                 r8*r8+r9*r9+r10*r10+r11*r11+r12*r12+r13*r13
        if res_sq < TOL_SQ
            break
        end
    end

    # Final output: write converged f_eq
    for i in Int32(1):n
        cx_i = Float32(cx_arr[i]) - shift_x
        cy_i = Float32(cy_arr[i]) - shift_y
        cz_i = Float32(cz_arr[i]) - shift_z
        csq_i = cx_i*cx_i + cy_i*cy_i + cz_i*cz_i
        efl = csq_i - dplus2_cs2
        lda = la1 + la2*cx_i + la3*cy_i + la4*cz_i +
              la5*(cx_i*cx_i - cs2) + la6*(cy_i*cy_i - cs2) +
              la7*(cz_i*cz_i - cs2) +
              la8*cx_i*cy_i + la9*cx_i*cz_i + la10*cy_i*cz_i +
              la11*cx_i*efl + la12*cy_i*efl + la13*cz_i*efl
        f_eq[i] = max(rho * w_arr[i] * exp(clamp(lda, -20.0f0, 20.0f0)), F_FLOOR_EQ)
    end

    return true
end

# ── 13-moment solver-only: returns λ[1:13] without storing full f_eq ──
# Used by interpolation kernel which only needs f_eq for 2 specific directions.
# Same 3-group recompute approach as main solver: 3 inner loops, no stored f_eq.
@noinline function solve_13mom_lambdas_d3q39!(
    lambda::MVector{13, Float32},
    rho::Float32, ux::Float32, uy::Float32, uz::Float32,
    cx_arr, cy_arr, cz_arr, w_arr,
    cs2::Float32,
    max_iter::Int32,
    T::Float32=1.0f0,
    shift_x::Float32=0.0f0, shift_y::Float32=0.0f0, shift_z::Float32=0.0f0,
    eos_R::Float32=Float32(-1.0f0)        # SOTA A3: thermodynamic R; -1 → legacy cs2
)
    lambda_nt = solve_13mom_lambdas_d3q39(
        rho, ux, uy, uz,
        cx_arr, cy_arr, cz_arr, w_arr,
        cs2, max_iter, T,
        shift_x, shift_y, shift_z,
        1.0f0,                             # tol_multiplier
        eos_R                              # SOTA A3 pass-through
    )
    @inbounds for i in 1:13
        lambda[i] = lambda_nt[i]
    end
    return nothing
end

# ── NTuple-returning variant for GPU: returns NTuple{13, Float32} in registers ──
# MVector across @noinline boundary causes GPU heap allocation (OOM).
# NTuple is immutable → passed entirely in registers → no heap allocation.
# Same 3-group recompute approach as solve_13mom_lambdas_d3q39!.
@noinline function solve_13mom_lambdas_d3q39(
    rho::Float32, ux::Float32, uy::Float32, uz::Float32,
    cx_arr, cy_arr, cz_arr, w_arr,
    cs2::Float32,
    max_iter::Int32,
    T::Float32=1.0f0,
    shift_x::Float32=0.0f0, shift_y::Float32=0.0f0, shift_z::Float32=0.0f0,
    tol_multiplier::Float32=1.0f0,
    eos_R::Float32=Float32(-1.0f0),         # SOTA A3: see compute_target_moments_d3q39 docstring
    # Tier B4 (2026-05-28) — warm-start λ from previous timestep's solve.
    # NaN32 in the first slot = sentinel "no warm-start, use polynomial init".
    # When the caller passes the previous step's λ (read from the per-cell
    # lambda_cache), Newton starts near the solution and converges in 1-3
    # iterations instead of 5-10 in steady-state cells. Saves ~50-70% of the
    # solver cost in freestream-dominated regions.
    init_lambda::NTuple{13, Float32} = (NaN32, NaN32, NaN32, NaN32, NaN32, NaN32, NaN32, NaN32, NaN32, NaN32, NaN32, NaN32, NaN32)
)::NTuple{13, Float32}
    n = Int32(39)
    F_FLOOR_EQ = Float32(1e-15)
    OMEGA = Float32(1.5)
    # tol_multiplier allows caller to relax tolerance (e.g., 100× for interpolation)
    TOL_SQ = Float32(507) * eps(Float32) * eps(Float32) * rho * rho * tol_multiplier

    # SOTA A3: when caller passes eos_R < 0 (default sentinel), fall back to
    # cs2 (legacy linearized behavior). Explicit eos_R = (γ-1)·cv activates
    # thermodynamic EOS. The polynomial init guess below still uses cs² for
    # the lattice-Hermite expansion (that's lattice geometry, not EOS).
    eos_R_eff = (eos_R < 0.0f0) ? cs2 : eos_R

    usq = ux * ux + uy * uy + uz * uz
    dT = T - 1.0f0
    dplus2_cs2 = 5.0f0 * cs2

    # Target moments from Maxwell-Boltzmann (eos_R_eff parameterizes the
    # thermodynamic R that contributes to the variance; defaults to cs2)
    mt1  = rho
    mt2  = rho * ux
    mt3  = rho * uy
    mt4  = rho * uz
    mt5  = rho * (ux * ux + eos_R_eff * dT)
    mt6  = rho * (uy * uy + eos_R_eff * dT)
    mt7  = rho * (uz * uz + eos_R_eff * dT)
    mt8  = rho * ux * uy
    mt9  = rho * ux * uz
    mt10 = rho * uy * uz
    eflux_corr = usq + 5.0f0 * eos_R_eff * dT
    mt11 = rho * ux * eflux_corr
    mt12 = rho * uy * eflux_corr
    mt13 = rho * uz * eflux_corr

    # Initial λ guess.
    # The polynomial (cold) init is ALWAYS computed — it is the fallback target
    # for the smart-B4 residual gate below. It costs only a handful of FLOPs.
    inv_cs2 = 1.0f0 / cs2
    inv_cs4 = inv_cs2 * inv_cs2
    pla1  = -0.5f0 * usq * inv_cs2
    pla2  = ux * inv_cs2
    pla3  = uy * inv_cs2
    pla4  = uz * inv_cs2
    pla5  = 0.5f0 * inv_cs4 * ux * ux
    pla6  = 0.5f0 * inv_cs4 * uy * uy
    pla7  = 0.5f0 * inv_cs4 * uz * uz
    pla8  = inv_cs4 * ux * uy
    pla9  = inv_cs4 * ux * uz
    pla10 = inv_cs4 * uy * uz
    pla11 = Float32(0)
    pla12 = Float32(0)
    pla13 = Float32(0)

    # Tier B4: if init_lambda is finite (non-NaN sentinel), provisionally use it
    # as warm-start from the previous timestep's converged λ. Otherwise the
    # polynomial init is used directly. The smart-B4 residual gate below
    # re-validates the warm-start AFTER the Hermite basis is hoisted and reverts
    # to the polynomial init if the warm-start is too stale (wave-front cells).
    use_warm = !isnan(init_lambda[1])
    if use_warm
        la1  = init_lambda[1]
        la2  = init_lambda[2]
        la3  = init_lambda[3]
        la4  = init_lambda[4]
        la5  = init_lambda[5]
        la6  = init_lambda[6]
        la7  = init_lambda[7]
        la8  = init_lambda[8]
        la9  = init_lambda[9]
        la10 = init_lambda[10]
        la11 = init_lambda[11]
        la12 = init_lambda[12]
        la13 = init_lambda[13]
    else
        la1  = pla1;  la2  = pla2;  la3  = pla3;  la4  = pla4
        la5  = pla5;  la6  = pla6;  la7  = pla7;  la8  = pla8
        la9  = pla9;  la10 = pla10; la11 = pla11; la12 = pla12; la13 = pla13
    end

    # ── Register-pressure fix (2026-05-30): RECOMPUTE the Hermite basis inline ──
    # The C4 "Tier 1" opt (2026-05-28) hoisted the per-direction basis into 12×
    # MVector{39} caches (cx_e..rw_e) to save ~31.5K FLOPs/cell. But a register
    # probe (LUDWIG_KERNEL_PROBE) measured the D3Q39 entropic kernel at 254
    # regs/thread with 3120 B/thread SPILL and ~1 block/SM occupancy — and those
    # 429 cached floats ARE the spill. On a register-starved kernel the spill
    # traffic + lost occupancy cost more than the cheap recompute, so we revert
    # to recomputing the basis on demand via this @inline helper. The arithmetic
    # is byte-identical to the cached version (same ops, same order) → results
    # unchanged; only the register footprint drops.
    # MEASURED 2026-05-30 (probe + case 045): registers 254→242, spill
    # 3120→1072 B/thread (−66%), step-56 M/E/J byte-identical to the cached
    # version. Still occupancy-limited at 1 block/SM (242 > 168) → the bigger
    # feature-split is the remaining win; this is the safe first step.
    @inline _basis(i) = begin
        cxi = Float32(cx_arr[i]) - shift_x
        cyi = Float32(cy_arr[i]) - shift_y
        czi = Float32(cz_arr[i]) - shift_z
        csqi = cxi*cxi + cyi*cyi + czi*czi
        efli = csqi - dplus2_cs2
        # (cx, cy, cz, h5, h6, h7, h8, h9, h10, h11, h12, h13, rw)
        (cxi, cyi, czi,
         cxi*cxi - cs2, cyi*cyi - cs2, czi*czi - cs2,
         cxi*cyi, cxi*czi, cyi*czi,
         cxi*efli, cyi*efli, czi*efli,
         rho * Float32(w_arr[i]))
    end

    # ── Smart B4 (2026-05-29): warm-start staleness gate ────────────────────
    # The warm-start λ from the previous step is excellent in steady-state cells
    # (Newton converges in 1-3 sweeps) but can be far from the truth at transient
    # wave fronts — where the macros (ρ, u, T) jump between steps. A stale guess
    # there can leave Newton only partially converged after max_iter, producing
    # the wrong f_eq → the L1↔L2 velocity stripe seen in case 030.
    #
    # Fix: evaluate the residual of the 13 moments at the warm-start λ ONCE
    # (one 39-direction pass, ≈ one extra sweep's cost, paid only on warm cells).
    # If it exceeds GATE_FACTOR × TOL_SQ, the warm-start is too stale → discard
    # it and restart from the polynomial init (always a known-good basin). Steady
    # cells pass the gate cheaply and keep the B4 perf win; only wave-front cells
    # pay the polynomial-init convergence cost — exactly the B4-disabled behavior,
    # but localized to the cells that need it.
    #
    # GATE_FACTOR = 100 → accept warm-starts whose per-moment residual is within
    # ~10× of the convergence tolerance. Tunable in [25, 1000]; 100 is the
    # handover's recommended starting point.
    if use_warm
        GATE_FACTOR = Float32(100)
        gd1 = Float32(0); gd2 = Float32(0); gd3 = Float32(0); gd4 = Float32(0)
        gd5 = Float32(0); gd6 = Float32(0); gd7 = Float32(0)
        gd8 = Float32(0); gd9 = Float32(0); gd10 = Float32(0)
        gd11 = Float32(0); gd12 = Float32(0); gd13 = Float32(0)
        @inbounds for i in Int32(1):n
            cx_i, cy_i, cz_i, a5, a6, a7, a8, a9, a10, a11, a12, a13, rw_i = _basis(i)
            lda = la1 + la2*cx_i + la3*cy_i + la4*cz_i +
                  la5*a5 + la6*a6 + la7*a7 +
                  la8*a8 + la9*a9 + la10*a10 +
                  la11*a11 + la12*a12 + la13*a13
            fi = max(rw_i * exp(clamp(lda, -20.0f0, 20.0f0)), F_FLOOR_EQ)
            gd1 += fi
            gd2 += fi*cx_i;  gd3 += fi*cy_i;  gd4 += fi*cz_i
            gd5 += fi*a5;    gd6 += fi*a6;    gd7 += fi*a7
            gd8 += fi*a8;    gd9 += fi*a9;    gd10 += fi*a10
            gd11 += fi*a11;  gd12 += fi*a12;  gd13 += fi*a13
        end
        gr1 = mt1 - gd1; gr2 = mt2 - gd2; gr3 = mt3 - gd3; gr4 = mt4 - gd4
        gr5 = mt5 - gd5; gr6 = mt6 - gd6; gr7 = mt7 - gd7
        gr8 = mt8 - gd8; gr9 = mt9 - gd9; gr10 = mt10 - gd10
        gr11 = mt11 - gd11; gr12 = mt12 - gd12; gr13 = mt13 - gd13
        gres_sq = gr1*gr1 + gr2*gr2 + gr3*gr3 + gr4*gr4 + gr5*gr5 + gr6*gr6 +
                  gr7*gr7 + gr8*gr8 + gr9*gr9 + gr10*gr10 +
                  gr11*gr11 + gr12*gr12 + gr13*gr13
        if gres_sq > GATE_FACTOR * TOL_SQ
            # Warm-start too stale — revert to the polynomial init basin.
            la1  = pla1;  la2  = pla2;  la3  = pla3;  la4  = pla4
            la5  = pla5;  la6  = pla6;  la7  = pla7;  la8  = pla8
            la9  = pla9;  la10 = pla10; la11 = pla11; la12 = pla12; la13 = pla13
        end
    end

    for sweep in Int32(1):max_iter
        d1 = Float32(0); d2 = Float32(0); d3 = Float32(0); d4 = Float32(0)
        j1 = Float32(0); j2 = Float32(0); j3 = Float32(0); j4 = Float32(0)
        @inbounds for i in Int32(1):n
            cx_i, cy_i, cz_i, b5, b6, b7, b8, b9, b10, b11, b12, b13, rw_i = _basis(i)
            lda = la1 + la2*cx_i + la3*cy_i + la4*cz_i +
                  la5*b5 + la6*b6 + la7*b7 +
                  la8*b8 + la9*b9 + la10*b10 +
                  la11*b11 + la12*b12 + la13*b13
            fi = max(rw_i * exp(clamp(lda, -20.0f0, 20.0f0)), F_FLOOR_EQ)
            d1 += fi;       j1 += fi
            d2 += fi*cx_i;  j2 += fi*cx_i*cx_i
            d3 += fi*cy_i;  j3 += fi*cy_i*cy_i
            d4 += fi*cz_i;  j4 += fi*cz_i*cz_i
        end
        la1 += j1 > F_FLOOR_EQ ? clamp(OMEGA * (mt1 - d1) / j1, -2.0f0, 2.0f0) : 0.0f0
        la2 += j2 > F_FLOOR_EQ ? clamp(OMEGA * (mt2 - d2) / j2, -2.0f0, 2.0f0) : 0.0f0
        la3 += j3 > F_FLOOR_EQ ? clamp(OMEGA * (mt3 - d3) / j3, -2.0f0, 2.0f0) : 0.0f0
        la4 += j4 > F_FLOOR_EQ ? clamp(OMEGA * (mt4 - d4) / j4, -2.0f0, 2.0f0) : 0.0f0

        d5 = Float32(0); d6 = Float32(0); d7 = Float32(0)
        d8 = Float32(0); d9 = Float32(0); d10 = Float32(0)
        j5 = Float32(0); j6 = Float32(0); j7 = Float32(0)
        j8 = Float32(0); j9 = Float32(0); j10 = Float32(0)
        @inbounds for i in Int32(1):n
            cx_i, cy_i, cz_i, a5, a6, a7, a8, a9, a10, b11, b12, b13, rw_i = _basis(i)
            lda = la1 + la2*cx_i + la3*cy_i + la4*cz_i +
                  la5*a5 + la6*a6 + la7*a7 +
                  la8*a8 + la9*a9 + la10*a10 +
                  la11*b11 + la12*b12 + la13*b13
            fi = max(rw_i * exp(clamp(lda, -20.0f0, 20.0f0)), F_FLOOR_EQ)
            d5 += fi*a5;  j5 += fi*a5*a5
            d6 += fi*a6;  j6 += fi*a6*a6
            d7 += fi*a7;  j7 += fi*a7*a7
            d8 += fi*a8;  j8 += fi*a8*a8
            d9 += fi*a9;  j9 += fi*a9*a9
            d10 += fi*a10; j10 += fi*a10*a10
        end
        la5  += j5 > F_FLOOR_EQ ? clamp(OMEGA * (mt5  - d5) / j5,  -2.0f0, 2.0f0) : 0.0f0
        la6  += j6 > F_FLOOR_EQ ? clamp(OMEGA * (mt6  - d6) / j6,  -2.0f0, 2.0f0) : 0.0f0
        la7  += j7 > F_FLOOR_EQ ? clamp(OMEGA * (mt7  - d7) / j7,  -2.0f0, 2.0f0) : 0.0f0
        la8  += j8 > F_FLOOR_EQ ? clamp(OMEGA * (mt8  - d8) / j8,  -2.0f0, 2.0f0) : 0.0f0
        la9  += j9 > F_FLOOR_EQ ? clamp(OMEGA * (mt9  - d9) / j9,  -2.0f0, 2.0f0) : 0.0f0
        la10 += j10 > F_FLOOR_EQ ? clamp(OMEGA * (mt10 - d10) / j10, -2.0f0, 2.0f0) : 0.0f0

        d11 = Float32(0); d12 = Float32(0); d13 = Float32(0)
        j11 = Float32(0); j12 = Float32(0); j13 = Float32(0)
        @inbounds for i in Int32(1):n
            cx_i, cy_i, cz_i, b5, b6, b7, b8, b9, b10, a11, a12, a13, rw_i = _basis(i)
            lda = la1 + la2*cx_i + la3*cy_i + la4*cz_i +
                  la5*b5 + la6*b6 + la7*b7 +
                  la8*b8 + la9*b9 + la10*b10 +
                  la11*a11 + la12*a12 + la13*a13
            fi = max(rw_i * exp(clamp(lda, -20.0f0, 20.0f0)), F_FLOOR_EQ)
            d11 += fi*a11; j11 += fi*a11*a11
            d12 += fi*a12; j12 += fi*a12*a12
            d13 += fi*a13; j13 += fi*a13*a13
        end
        la11 += j11 > F_FLOOR_EQ ? clamp(OMEGA * (mt11 - d11) / j11, -2.0f0, 2.0f0) : 0.0f0
        la12 += j12 > F_FLOOR_EQ ? clamp(OMEGA * (mt12 - d12) / j12, -2.0f0, 2.0f0) : 0.0f0
        la13 += j13 > F_FLOOR_EQ ? clamp(OMEGA * (mt13 - d13) / j13, -2.0f0, 2.0f0) : 0.0f0

        r1 = mt1 - d1; r2 = mt2 - d2; r3 = mt3 - d3; r4 = mt4 - d4
        r5 = mt5 - d5; r6 = mt6 - d6; r7 = mt7 - d7
        r8 = mt8 - d8; r9 = mt9 - d9; r10 = mt10 - d10
        r11 = mt11 - d11; r12 = mt12 - d12; r13 = mt13 - d13
        res_sq = r1*r1 + r2*r2 + r3*r3 + r4*r4 + r5*r5 + r6*r6 + r7*r7 +
                 r8*r8 + r9*r9 + r10*r10 + r11*r11 + r12*r12 + r13*r13
        if res_sq < TOL_SQ
            break
        end
    end

    return (la1, la2, la3, la4, la5, la6, la7, la8, la9, la10, la11, la12, la13)
end

# ── Single-direction evaluation from precomputed 13-moment λ parameters ──
# cx_k, cy_k, cz_k should be UNSHIFTED lattice velocities (already shifted by caller).
# Accepts both NTuple{13, Float32} and MVector{13, Float32} for lambda.
@inline function eval_13mom_feq_single(
    rho::Float32, lambda,
    cx_k::Float32, cy_k::Float32, cz_k::Float32,
    w_k::Float32, cs2::Float32
)
    csq_k = cx_k*cx_k + cy_k*cy_k + cz_k*cz_k
    dplus2_cs2 = 5.0f0 * cs2

    a1  = 1.0f0
    a2  = cx_k
    a3  = cy_k
    a4  = cz_k
    a5  = cx_k*cx_k - cs2
    a6  = cy_k*cy_k - cs2
    a7  = cz_k*cz_k - cs2
    a8  = cx_k*cy_k
    a9  = cx_k*cz_k
    a10 = cy_k*cz_k
    a11 = cx_k * (csq_k - dplus2_cs2)
    a12 = cy_k * (csq_k - dplus2_cs2)
    a13 = cz_k * (csq_k - dplus2_cs2)

    lda = lambda[1]*a1 + lambda[2]*a2 + lambda[3]*a3 + lambda[4]*a4 +
          lambda[5]*a5 + lambda[6]*a6 + lambda[7]*a7 +
          lambda[8]*a8 + lambda[9]*a9 + lambda[10]*a10 +
          lambda[11]*a11 + lambda[12]*a12 + lambda[13]*a13

    fi = rho * w_k * exp(clamp(lda, -20.0f0, 20.0f0))
    return max(fi, Float32(1e-15))
end

@inline function compute_13mom_mass_correction_d3q39(
    rho::Float32, lambda,
    cx_arr, cy_arr, cz_arr, w_arr,
    cs2::Float32,
    shift_x::Float32=0.0f0, shift_y::Float32=0.0f0, shift_z::Float32=0.0f0
)
    feq_sum = 0.0f0
    for i in Int32(1):Int32(39)
        feq_sum += eval_13mom_feq_single(
            rho, lambda,
            Float32(cx_arr[i]) - shift_x,
            Float32(cy_arr[i]) - shift_y,
            Float32(cz_arr[i]) - shift_z,
            w_arr[i], cs2
        )
    end
    return feq_sum > Float32(1e-15) ? rho / feq_sum : 1.0f0
end

@inline function mass_correct_feq39!(f_eq::MVector{39, Float32}, rho::Float32)
    feq_sum = 0.0f0
    for i in Int32(1):Int32(39)
        feq_sum += f_eq[i]
    end
    if feq_sum > Float32(1e-15)
        corr = rho / feq_sum
        for i in Int32(1):Int32(39)
            f_eq[i] *= corr
        end
    end
    return nothing
end

# Pass 265 — exact mass + momentum correction for a 13-moment numerical
# equilibrium. The Newton solver in `solve_13mom_lambdas_d3q39` converges
# to ~1e-5 tolerance on all 13 moments. For AMR restriction we need the
# four conserved moments (mass, three momentum components) to match the
# target macros to ROUND-OFF, so the parent-side populations are exactly
# what the restriction protocol requires.
#
# Adds δf_eq_k = w_k · (a0 + a1·(c_x − shift_x) + a2·(c_y − shift_y) + a3·(c_z − shift_z))
# with (a0,a1,a2,a3) solving the 4×4 weighted Gram system. This basis is
# orthogonal to the 2nd-order Hermite stress basis used in
# `project_second_order_stress_mode`, so the subsequent stress
# reconstruction is unaffected.
#
# Frame note: the correction is computed in the shifted frame (matches what
# `solve_13mom_lambdas_d3q39` and the kernel collision use). `target_jx_rel`
# should be `rho · u_rel`, not `rho · u_lab`. For shift=0 they coincide.
#
# Returns (ε_rho_before, ε_jx_before, ε_jy_before, ε_jz_before) so callers
# can log the residual that was just closed.
@inline function apply_4momentum_correction_d3q39!(
    f_eq::MVector{39, Float32},
    target_rho::Float32,
    target_jx_rel::Float32, target_jy_rel::Float32, target_jz_rel::Float32,
    cx_arr, cy_arr, cz_arr, w_arr,
    shift_x::Float32=0.0f0, shift_y::Float32=0.0f0, shift_z::Float32=0.0f0
)
    rho_eq = 0.0f0
    jx_eq = 0.0f0; jy_eq = 0.0f0; jz_eq = 0.0f0
    @inbounds for k in Int32(1):Int32(39)
        cxk = Float32(cx_arr[k]) - shift_x
        cyk = Float32(cy_arr[k]) - shift_y
        czk = Float32(cz_arr[k]) - shift_z
        rho_eq += f_eq[k]
        jx_eq += f_eq[k] * cxk
        jy_eq += f_eq[k] * cyk
        jz_eq += f_eq[k] * czk
    end

    eps_rho = target_rho - rho_eq
    eps_jx = target_jx_rel - jx_eq
    eps_jy = target_jy_rel - jy_eq
    eps_jz = target_jz_rel - jz_eq

    # Early exit: already at round-off for Float32 (~1e-7 relative).
    if abs(eps_rho) < 1.0f-9 && abs(eps_jx) < 1.0f-9 &&
       abs(eps_jy) < 1.0f-9 && abs(eps_jz) < 1.0f-9
        return (eps_rho, eps_jx, eps_jy, eps_jz)
    end

    # Weighted Gram matrix M_ij = Σ w_k · ψ_i(c_k − shift) · ψ_j(c_k − shift)
    # with ψ = {1, c_x − shift_x, c_y − shift_y, c_z − shift_z}.
    m00 = 0.0f0; m01 = 0.0f0; m02 = 0.0f0; m03 = 0.0f0
    m11 = 0.0f0; m12 = 0.0f0; m13 = 0.0f0
    m22 = 0.0f0; m23 = 0.0f0; m33 = 0.0f0
    @inbounds for k in Int32(1):Int32(39)
        wk = Float32(w_arr[k])
        cxk = Float32(cx_arr[k]) - shift_x
        cyk = Float32(cy_arr[k]) - shift_y
        czk = Float32(cz_arr[k]) - shift_z
        m00 += wk
        m01 += wk * cxk
        m02 += wk * cyk
        m03 += wk * czk
        m11 += wk * cxk * cxk
        m12 += wk * cxk * cyk
        m13 += wk * cxk * czk
        m22 += wk * cyk * cyk
        m23 += wk * cyk * czk
        m33 += wk * czk * czk
    end

    (a0, a1, a2, a3, ok) = solve_shell_moment4(
        m00, m01, m02, m03,
        m11, m12, m13,
        m22, m23,
        m33,
        eps_rho, eps_jx, eps_jy, eps_jz
    )

    if !ok
        return (eps_rho, eps_jx, eps_jy, eps_jz)
    end

    @inbounds for k in Int32(1):Int32(39)
        wk = Float32(w_arr[k])
        cxk = Float32(cx_arr[k]) - shift_x
        cyk = Float32(cy_arr[k]) - shift_y
        czk = Float32(cz_arr[k]) - shift_z
        f_eq[k] += wk * (a0 + a1 * cxk + a2 * cyk + a3 * czk)
    end

    return (eps_rho, eps_jx, eps_jy, eps_jz)
end

# ==============================================================================
# 5-MOMENT CONSERVATION CORRECTION FOR D3Q39 — mass + 3·momentum + kinetic energy
# ==============================================================================
# SOTA Roadmap Active Workstream P1: conservative ρE across AMR.
#
# Extends `apply_4momentum_correction_d3q39!` (which handles mass + momentum)
# by also enforcing Σ f · |c|² = E_kin_target. This is what's needed to make
# the AMR fine→coarse restriction strictly conservative in total energy.
#
# Mathematical structure: closed-form (no matrix solve) for D3Q39 in the
# orthogonalized basis {1, c_x, c_y, c_z, |c|² − ⟨|c|²⟩}, exploiting:
#   ⟨w⟩         = 1           (normalized weights)
#   ⟨w·c_α⟩     = 0           (lattice isotropy)
#   ⟨w·c_α·c_β⟩ = cs² δ_αβ    (isotropy, cs² = 2/3 for D3Q39)
#   ⟨w·|c|²⟩    = 3·cs² = 2   (D3Q39)
#   ⟨w·|c|⁴⟩    = 20/3        (D3Q39 — computed once from shell weights)
#
# The 5-moment-target system reduces (using lattice isotropy) to:
#   a0          = Δρ
#   a_α         = 1.5 · Δj_α      for α = x, y, z
#   a4          = 0.375 · (ΔE_kin − 2·Δρ)
#
# Correction applied per population:
#   δf[k] = w_k · (a0 + a1·c_x + a2·c_y + a3·c_z + a4·(|c_k|² − 2))
#
# By construction this is moment-orthogonal to the 2nd-order traceless stress
# basis (Π_αβ − ⅓·trace), so the shear stress carrier is preserved exactly.
# The bulk-pressure (trace) part of stress IS coupled to the energy correction
# (necessary — pressure and energy share the same trace mode in 3D Hermite).
#
# Frame note: matches the existing 4-moment correction convention. Targets
# must be in the SHIFTED frame (j_rel = j_lab − ρ·shift; E_kin_rel computed
# from shifted velocities), to be consistent with `solve_13mom_lambdas_d3q39`
# and the rest of the collision kernel.
#
# Returns (ε_rho, ε_jx, ε_jy, ε_jz, ε_Ekin) for diagnostics.
# ==============================================================================
@inline function apply_5momentum_correction_d3q39!(
    f_eq::MVector{39, Float32},
    target_rho::Float32,
    target_jx_rel::Float32, target_jy_rel::Float32, target_jz_rel::Float32,
    target_Ekin_rel::Float32,
    cx_arr, cy_arr, cz_arr, w_arr,
    shift_x::Float32=0.0f0, shift_y::Float32=0.0f0, shift_z::Float32=0.0f0
)
    # ── Compute current moments of f_eq in the shifted frame ──────────────
    rho_eq  = 0.0f0
    jx_eq   = 0.0f0
    jy_eq   = 0.0f0
    jz_eq   = 0.0f0
    Ekin_eq = 0.0f0
    @inbounds for k in Int32(1):Int32(39)
        cxk = Float32(cx_arr[k]) - shift_x
        cyk = Float32(cy_arr[k]) - shift_y
        czk = Float32(cz_arr[k]) - shift_z
        c2k = cxk * cxk + cyk * cyk + czk * czk
        fk  = f_eq[k]
        rho_eq  += fk
        jx_eq   += fk * cxk
        jy_eq   += fk * cyk
        jz_eq   += fk * czk
        Ekin_eq += fk * c2k
    end

    eps_rho  = target_rho       - rho_eq
    eps_jx   = target_jx_rel    - jx_eq
    eps_jy   = target_jy_rel    - jy_eq
    eps_jz   = target_jz_rel    - jz_eq
    eps_Ekin = target_Ekin_rel  - Ekin_eq

    # Early exit if already at Float32 round-off.
    if abs(eps_rho)  < 1.0f-9 && abs(eps_jx)   < 1.0f-9 &&
       abs(eps_jy)   < 1.0f-9 && abs(eps_jz)   < 1.0f-9 &&
       abs(eps_Ekin) < 1.0f-9
        return (eps_rho, eps_jx, eps_jy, eps_jz, eps_Ekin)
    end

    # ── Closed-form coefficients (D3Q39, orthogonalized basis) ────────────
    # cs² = 2/3, ⟨|c|²⟩ = 2, ⟨e5²⟩ = ⟨|c|⁴⟩ − 9·cs⁴ = 20/3 − 4 = 8/3
    # Moment-target system reduces to:
    #   a0 = Δρ
    #   a_α = Δj_α / cs²        = 1.5 · Δj_α
    #   a4·(8/3) + 2·a0 = ΔE_kin  ⇒  a4 = (ΔE_kin − 2·a0) / (8/3)
    cs2_mean = 2.0f0   # ⟨|c|²⟩ for D3Q39 = 3·cs² = 2
    a0 = eps_rho
    a1 = 1.5f0 * eps_jx
    a2 = 1.5f0 * eps_jy
    a3 = 1.5f0 * eps_jz
    a4 = 0.375f0 * (eps_Ekin - 2.0f0 * a0)   # 0.375 = 3/8 = 1/(8/3)

    # ── Apply per-population correction in the shifted frame ──────────────
    # δf[k] = w_k · (a0 + a1·c_x + a2·c_y + a3·c_z + a4·(|c|² − 2))
    @inbounds for k in Int32(1):Int32(39)
        wk  = Float32(w_arr[k])
        cxk = Float32(cx_arr[k]) - shift_x
        cyk = Float32(cy_arr[k]) - shift_y
        czk = Float32(cz_arr[k]) - shift_z
        c2k = cxk * cxk + cyk * cyk + czk * czk
        f_eq[k] += wk * (a0 + a1 * cxk + a2 * cyk + a3 * czk + a4 * (c2k - cs2_mean))
    end

    return (eps_rho, eps_jx, eps_jy, eps_jz, eps_Ekin)
end

# Convenience wrapper that allocates and returns f_eq
@inline function calculate_equilibrium_numerical_d3q39(
    rho::Float32, ux::Float32, uy::Float32, uz::Float32,
    cx_arr, cy_arr, cz_arr, w_arr,
    cs2::Float32,
    max_iter::Int32
)
    f_eq = MVector{39, Float32}(undef)
    calculate_equilibrium_numerical_d3q39!(f_eq, rho, ux, uy, uz,
                                            cx_arr, cy_arr, cz_arr, w_arr,
                                            cs2, max_iter)
    return f_eq
end

# ──────────────────────────────────────────────────────────────────────
# B1 Pillar 2 — 13-moment-exact restriction (mortar-element AMR)
# ──────────────────────────────────────────────────────────────────────
#
# Solve for λ such that the resulting f_eq matches 13 EXPLICIT moment
# targets exactly. Same 3-group coordinate descent as the MB-target
# variant `solve_13mom_lambdas_d3q39`, but with mt1..mt13 supplied by
# the caller rather than derived from Maxwell-Boltzmann.
#
# Used by the AMR fine→coarse restriction path to drive all 13 moments
# (mass + momentum + 6 stress + 3 energy flux) to round-off exactness,
# replacing the legacy 4- or 5-moment correction. See `SOTA_ROADMAP.md`
# active workstream "B1 — Mortar-element AMR coupling" pillar 2.
#
# Reference: Frapolli, Chikatamarla, Karlin 2021 (entropic LBM with
# conservative AMR for compressible flow).
@noinline function solve_13mom_lambdas_explicit_d3q39(
    rho::Float32,
    mt1::Float32, mt2::Float32, mt3::Float32, mt4::Float32,
    mt5::Float32, mt6::Float32, mt7::Float32,
    mt8::Float32, mt9::Float32, mt10::Float32,
    mt11::Float32, mt12::Float32, mt13::Float32,
    cx_arr, cy_arr, cz_arr, w_arr,
    cs2::Float32,
    max_iter::Int32,
    shift_x::Float32=0.0f0, shift_y::Float32=0.0f0, shift_z::Float32=0.0f0,
    tol_multiplier::Float32=1.0f0
)::NTuple{14, Float32}
    # GOLD-STANDARD CONVERGENCE FLAG (2026-06-25): the 14th tuple slot encodes
    # convergence as a Float32 sentinel (1.0 = converged & finite, 0.0 = not).
    # A homogeneous NTuple{14,Float32} is GPU-tuple-safe (no mixed Bool packing,
    # no heap). Callers that don't care about convergence simply ignore [14].
    n = Int32(39)
    F_FLOOR_EQ = Float32(1e-15)
    OMEGA = Float32(1.5)
    TOL_SQ = Float32(507) * eps(Float32) * eps(Float32) * rho * rho * tol_multiplier

    dplus2_cs2 = 5.0f0 * cs2

    # Polynomial init guess derived from the supplied mass + momentum + diagonal stress.
    inv_rho = 1.0f0 / max(rho, F_FLOOR_EQ)
    ux = mt2 * inv_rho
    uy = mt3 * inv_rho
    uz = mt4 * inv_rho
    usq = ux * ux + uy * uy + uz * uz
    inv_cs2 = 1.0f0 / cs2
    inv_cs4 = inv_cs2 * inv_cs2

    la1  = -0.5f0 * usq * inv_cs2
    la2  = ux * inv_cs2
    la3  = uy * inv_cs2
    la4  = uz * inv_cs2
    la5  = 0.5f0 * inv_cs4 * ux * ux
    la6  = 0.5f0 * inv_cs4 * uy * uy
    la7  = 0.5f0 * inv_cs4 * uz * uz
    la8  = inv_cs4 * ux * uy
    la9  = inv_cs4 * ux * uz
    la10 = inv_cs4 * uy * uz
    la11 = Float32(0)
    la12 = Float32(0)
    la13 = Float32(0)

    # Persist the last residual across the loop so we can report convergence.
    # Initialised huge so a 0-iteration run (max_iter=0) reports NOT converged.
    res_sq = Float32(Inf)

    for sweep in Int32(1):max_iter
        # ── Group A: mass + momentum (moments 1-4) ──
        d1 = Float32(0); d2 = Float32(0); d3 = Float32(0); d4 = Float32(0)
        j1 = Float32(0); j2 = Float32(0); j3 = Float32(0); j4 = Float32(0)
        for i in Int32(1):n
            cx_i = Float32(cx_arr[i]) - shift_x
            cy_i = Float32(cy_arr[i]) - shift_y
            cz_i = Float32(cz_arr[i]) - shift_z
            csq_i = cx_i*cx_i + cy_i*cy_i + cz_i*cz_i
            efl = csq_i - dplus2_cs2
            lda = la1 + la2*cx_i + la3*cy_i + la4*cz_i +
                  la5*(cx_i*cx_i - cs2) + la6*(cy_i*cy_i - cs2) +
                  la7*(cz_i*cz_i - cs2) +
                  la8*cx_i*cy_i + la9*cx_i*cz_i + la10*cy_i*cz_i +
                  la11*cx_i*efl + la12*cy_i*efl + la13*cz_i*efl
            fi = max(rho * w_arr[i] * exp(clamp(lda, -20.0f0, 20.0f0)), F_FLOOR_EQ)
            d1 += fi;       j1 += fi
            d2 += fi*cx_i;  j2 += fi*cx_i*cx_i
            d3 += fi*cy_i;  j3 += fi*cy_i*cy_i
            d4 += fi*cz_i;  j4 += fi*cz_i*cz_i
        end
        la1 += j1 > F_FLOOR_EQ ? clamp(OMEGA * (mt1 - d1) / j1, -2.0f0, 2.0f0) : 0.0f0
        la2 += j2 > F_FLOOR_EQ ? clamp(OMEGA * (mt2 - d2) / j2, -2.0f0, 2.0f0) : 0.0f0
        la3 += j3 > F_FLOOR_EQ ? clamp(OMEGA * (mt3 - d3) / j3, -2.0f0, 2.0f0) : 0.0f0
        la4 += j4 > F_FLOOR_EQ ? clamp(OMEGA * (mt4 - d4) / j4, -2.0f0, 2.0f0) : 0.0f0

        # ── Group B: stress tensor (moments 5-10) ──
        d5 = Float32(0); d6 = Float32(0); d7 = Float32(0)
        d8 = Float32(0); d9 = Float32(0); d10 = Float32(0)
        j5 = Float32(0); j6 = Float32(0); j7 = Float32(0)
        j8 = Float32(0); j9 = Float32(0); j10 = Float32(0)
        for i in Int32(1):n
            cx_i = Float32(cx_arr[i]) - shift_x
            cy_i = Float32(cy_arr[i]) - shift_y
            cz_i = Float32(cz_arr[i]) - shift_z
            csq_i = cx_i*cx_i + cy_i*cy_i + cz_i*cz_i
            efl = csq_i - dplus2_cs2
            lda = la1 + la2*cx_i + la3*cy_i + la4*cz_i +
                  la5*(cx_i*cx_i - cs2) + la6*(cy_i*cy_i - cs2) +
                  la7*(cz_i*cz_i - cs2) +
                  la8*cx_i*cy_i + la9*cx_i*cz_i + la10*cy_i*cz_i +
                  la11*cx_i*efl + la12*cy_i*efl + la13*cz_i*efl
            fi = max(rho * w_arr[i] * exp(clamp(lda, -20.0f0, 20.0f0)), F_FLOOR_EQ)
            a5 = cx_i*cx_i - cs2; a6 = cy_i*cy_i - cs2; a7 = cz_i*cz_i - cs2
            a8 = cx_i*cy_i;       a9 = cx_i*cz_i;       a10 = cy_i*cz_i
            d5 += fi*a5;  j5 += fi*a5*a5
            d6 += fi*a6;  j6 += fi*a6*a6
            d7 += fi*a7;  j7 += fi*a7*a7
            d8 += fi*a8;  j8 += fi*a8*a8
            d9 += fi*a9;  j9 += fi*a9*a9
            d10 += fi*a10; j10 += fi*a10*a10
        end
        la5  += j5  > F_FLOOR_EQ ? clamp(OMEGA * (mt5  - d5)  / j5,  -2.0f0, 2.0f0) : 0.0f0
        la6  += j6  > F_FLOOR_EQ ? clamp(OMEGA * (mt6  - d6)  / j6,  -2.0f0, 2.0f0) : 0.0f0
        la7  += j7  > F_FLOOR_EQ ? clamp(OMEGA * (mt7  - d7)  / j7,  -2.0f0, 2.0f0) : 0.0f0
        la8  += j8  > F_FLOOR_EQ ? clamp(OMEGA * (mt8  - d8)  / j8,  -2.0f0, 2.0f0) : 0.0f0
        la9  += j9  > F_FLOOR_EQ ? clamp(OMEGA * (mt9  - d9)  / j9,  -2.0f0, 2.0f0) : 0.0f0
        la10 += j10 > F_FLOOR_EQ ? clamp(OMEGA * (mt10 - d10) / j10, -2.0f0, 2.0f0) : 0.0f0

        # ── Group C: energy flux (moments 11-13) ──
        d11 = Float32(0); d12 = Float32(0); d13 = Float32(0)
        j11 = Float32(0); j12 = Float32(0); j13 = Float32(0)
        for i in Int32(1):n
            cx_i = Float32(cx_arr[i]) - shift_x
            cy_i = Float32(cy_arr[i]) - shift_y
            cz_i = Float32(cz_arr[i]) - shift_z
            csq_i = cx_i*cx_i + cy_i*cy_i + cz_i*cz_i
            efl = csq_i - dplus2_cs2
            lda = la1 + la2*cx_i + la3*cy_i + la4*cz_i +
                  la5*(cx_i*cx_i - cs2) + la6*(cy_i*cy_i - cs2) +
                  la7*(cz_i*cz_i - cs2) +
                  la8*cx_i*cy_i + la9*cx_i*cz_i + la10*cy_i*cz_i +
                  la11*cx_i*efl + la12*cy_i*efl + la13*cz_i*efl
            fi = max(rho * w_arr[i] * exp(clamp(lda, -20.0f0, 20.0f0)), F_FLOOR_EQ)
            a11 = cx_i*efl; a12 = cy_i*efl; a13 = cz_i*efl
            d11 += fi*a11; j11 += fi*a11*a11
            d12 += fi*a12; j12 += fi*a12*a12
            d13 += fi*a13; j13 += fi*a13*a13
        end
        la11 += j11 > F_FLOOR_EQ ? clamp(OMEGA * (mt11 - d11) / j11, -2.0f0, 2.0f0) : 0.0f0
        la12 += j12 > F_FLOOR_EQ ? clamp(OMEGA * (mt12 - d12) / j12, -2.0f0, 2.0f0) : 0.0f0
        la13 += j13 > F_FLOOR_EQ ? clamp(OMEGA * (mt13 - d13) / j13, -2.0f0, 2.0f0) : 0.0f0

        r1 = mt1 - d1; r2 = mt2 - d2; r3 = mt3 - d3; r4 = mt4 - d4
        r5 = mt5 - d5; r6 = mt6 - d6; r7 = mt7 - d7
        r8 = mt8 - d8; r9 = mt9 - d9; r10 = mt10 - d10
        r11 = mt11 - d11; r12 = mt12 - d12; r13 = mt13 - d13
        res_sq = r1*r1 + r2*r2 + r3*r3 + r4*r4 + r5*r5 + r6*r6 + r7*r7 +
                 r8*r8 + r9*r9 + r10*r10 + r11*r11 + r12*r12 + r13*r13
        if res_sq < TOL_SQ
            break
        end
    end

    # Convergence sentinel (14th slot): 1.0 iff the residual met TOL_SQ AND all
    # λ are finite. NaN/Inf λ (degenerate cell) ⇒ NOT converged. Pure Float32.
    all_finite = isfinite(la1) & isfinite(la2) & isfinite(la3) & isfinite(la4) &
                 isfinite(la5) & isfinite(la6) & isfinite(la7) & isfinite(la8) &
                 isfinite(la9) & isfinite(la10) & isfinite(la11) & isfinite(la12) &
                 isfinite(la13)
    converged_f = ((res_sq < TOL_SQ) & all_finite) ? 1.0f0 : 0.0f0

    return (la1, la2, la3, la4, la5, la6, la7, la8, la9, la10, la11, la12, la13,
            converged_f)
end

# ──────────────────────────────────────────────────────────────────────
# B1 Pillar 2 — Compute 13 moments of a population vector
# ──────────────────────────────────────────────────────────────────────
"""
Compute the 13 Hermite moments of a 39-population vector in the SHIFTED
frame (c_k − U). The basis matches `compute_target_moments_d3q39`.
Returned as an NTuple to avoid heap allocation under @kernel.
"""
@inline function compute_13_moments_of_populations_d3q39(
    f::MVector{39, Float32},
    cx_arr, cy_arr, cz_arr,
    cs2::Float32,
    shift_x::Float32=0.0f0, shift_y::Float32=0.0f0, shift_z::Float32=0.0f0
)::NTuple{13, Float32}
    m1 = Float32(0); m2 = Float32(0); m3 = Float32(0); m4 = Float32(0)
    m5 = Float32(0); m6 = Float32(0); m7 = Float32(0)
    m8 = Float32(0); m9 = Float32(0); m10 = Float32(0)
    m11 = Float32(0); m12 = Float32(0); m13 = Float32(0)
    dplus2_cs2 = 5.0f0 * cs2
    @inbounds for k in Int32(1):Int32(39)
        cxk = Float32(cx_arr[k]) - shift_x
        cyk = Float32(cy_arr[k]) - shift_y
        czk = Float32(cz_arr[k]) - shift_z
        c2k = cxk*cxk + cyk*cyk + czk*czk
        efl = c2k - dplus2_cs2
        fk  = f[k]
        m1  += fk
        m2  += fk * cxk
        m3  += fk * cyk
        m4  += fk * czk
        m5  += fk * (cxk*cxk - cs2)
        m6  += fk * (cyk*cyk - cs2)
        m7  += fk * (czk*czk - cs2)
        m8  += fk * cxk * cyk
        m9  += fk * cxk * czk
        m10 += fk * cyk * czk
        m11 += fk * cxk * efl
        m12 += fk * cyk * efl
        m13 += fk * czk * efl
    end
    return (m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13)
end

"""
    ok = apply_13moment_restriction_d3q39!(f_out, f_avg, tau_ratio_neq,
                                            cx_arr, cy_arr, cz_arr, w_arr,
                                            cs2, eos_R, shift..., max_iter,
                                            T_parent=1.0)

Conservative 13-moment-exact restriction. Computes the 13 raw moments
M_raw of `f_avg` (child-averaged populations), splits them into the MB
equilibrium part `M_eq(ρ, j, T_parent)` and the non-equilibrium deviation,
applies the LBM τ-rescale `M_target = M_eq + tau_ratio_neq · (M_raw - M_eq)`
(off-diagonal stress + energy flux rescale bounded to a realizable factor),
then solves for the parent λ that matches all 13 target moments exactly
via `solve_13mom_lambdas_explicit_d3q39`. The result is written into
`f_out` evaluated at all 39 directions.

This is the B1 Pillar 2 contribution: replaces the legacy 4-or-5-moment
correction on f_eq + projected stress carrier with a single closed-form
solve that conserves all 13 Hermite moments to round-off exactly. No
secondary stress-carrier projection is needed afterwards.

`T_parent` (default 1.0) is the parent's reference temperature for the
M_eq split — threading the actual T keeps the thermal pressure in the
equilibrium reference instead of mislabeling it as non-equilibrium content.
T_parent=1.0 reproduces the prior athermal behavior byte-identically.

RETURNS `ok::Bool` — `true` when the solve converged, all populations are
finite & non-negative, and the realized higher moments stayed within a
bounded factor of the targets. `false` signals a non-realizable solve
(off-axis oblique-shock+shear cells): the caller MUST NOT commit `f_out`
and should fall back to the guaranteed-realizable feq+projected-stress
reconstruction for that cell. This preserves strict 13-moment conservation
where the solve succeeds and degrades gracefully where it fails.

Strict conservation: mass, 3·momentum, full stress tensor (6 components),
and the 3 energy-flux moments are conserved on each fine→coarse hop
to Float32 round-off (~1e-7) in the converged (ok==true) cells.
"""
@inline function apply_13moment_restriction_d3q39!(
    f_out::MVector{39, Float32},
    f_avg::MVector{39, Float32},
    tau_ratio_neq::Float32,
    cx_arr, cy_arr, cz_arr, w_arr,
    cs2::Float32,
    eos_R::Float32,
    shift_x::Float32, shift_y::Float32, shift_z::Float32,
    max_iter::Int32,
    T_parent::Float32=1.0f0
)::Bool
    # 1. Raw moments of averaged children, in the SHIFTED frame.
    M_raw = compute_13_moments_of_populations_d3q39(
        f_avg, cx_arr, cy_arr, cz_arr, cs2, shift_x, shift_y, shift_z)

    # 2. Parent ρ, u from raw moments (shifted-frame relative velocity).
    rho_p = max(M_raw[1], 0.01f0)
    inv_rho = 1.0f0 / rho_p
    ux_p = M_raw[2] * inv_rho
    uy_p = M_raw[3] * inv_rho
    uz_p = M_raw[4] * inv_rho

    eos_R_eff = (eos_R > 0.0f0) ? eos_R : cs2

    # 3. Maxwell-Boltzmann equilibrium moments at the parent's ACTUAL state.
    #    GOLD-STANDARD FIX (2026-06-25): the equilibrium reference must use the
    #    parent's real temperature, NOT a hard-coded T=1. With T=1 the thermal
    #    pressure ρ·R·(T−1) in the diagonal stress (M_eq[5..7]) and the thermal
    #    part of the energy flux (M_eq[11..13]) were misclassified as
    #    NON-equilibrium content and then amplified by tau_ratio_neq — driving
    #    the off-axis spike. Threading T_parent puts that thermal pressure in
    #    M_eq where it belongs, so only the genuine deviatoric/heat-flux f_neq
    #    is rescaled. T_parent defaults to 1.0 → byte-identical for athermal.
    T_eq = clamp(T_parent, 0.1f0, 10.0f0)
    M_eq = compute_target_moments_d3q39(rho_p, ux_p, uy_p, uz_p,
                                         T_eq, cs2, eos_R_eff)

    # 4. τ-rescale the non-equilibrium portion. Mass + momentum (1-4) are
    #    invariants (M_raw[i] ≡ M_eq[i] by construction). Stress (5-10) and
    #    energy flux (11-13) carry the non-equilibrium part that scales as
    #    f_neq ~ τ · ∂u/∂x; the coarse-cell f_neq is therefore the
    #    fine-averaged f_neq rescaled by τ_c / τ_f (in the τ-0.5 sense).
    #
    #    REALIZABILITY CLAMP (2026-06-25): bound the rescale factor applied to
    #    the OFF-DIAGONAL stress (mt8/9/10) and the ENERGY-FLUX (mt11/12/13) to
    #    ≤ 5. At off-axis oblique-shock+shear cells these moments are large and
    #    a tau_ratio_neq>1 over-amplification is what pushes the solve into the
    #    non-realizable, spiked-distribution regime. The diagonal stress
    #    (mt5/6/7) keeps the full factor (it is anchored by the positive
    #    thermal pressure now in M_eq), so bulk physics is unchanged in the
    #    common tau_ratio_neq≤1 production case where the clamp is inactive.
    tau_offdiag = clamp(tau_ratio_neq, -5.0f0, 5.0f0)
    mt1  = M_raw[1]
    mt2  = M_raw[2]
    mt3  = M_raw[3]
    mt4  = M_raw[4]
    mt5  = M_eq[5]  + tau_ratio_neq * (M_raw[5]  - M_eq[5])
    mt6  = M_eq[6]  + tau_ratio_neq * (M_raw[6]  - M_eq[6])
    mt7  = M_eq[7]  + tau_ratio_neq * (M_raw[7]  - M_eq[7])
    mt8  = M_eq[8]  + tau_offdiag * (M_raw[8]  - M_eq[8])
    mt9  = M_eq[9]  + tau_offdiag * (M_raw[9]  - M_eq[9])
    mt10 = M_eq[10] + tau_offdiag * (M_raw[10] - M_eq[10])
    mt11 = M_eq[11] + tau_offdiag * (M_raw[11] - M_eq[11])
    mt12 = M_eq[12] + tau_offdiag * (M_raw[12] - M_eq[12])
    mt13 = M_eq[13] + tau_offdiag * (M_raw[13] - M_eq[13])

    # 5. Solve for parent λ matching the 13 explicit targets. The 14th tuple
    #    slot reports convergence (1.0 = converged & finite).
    lambda = solve_13mom_lambdas_explicit_d3q39(
        rho_p,
        mt1, mt2, mt3, mt4, mt5, mt6, mt7,
        mt8, mt9, mt10, mt11, mt12, mt13,
        cx_arr, cy_arr, cz_arr, w_arr,
        cs2, max_iter, shift_x, shift_y, shift_z, 1.0f0)
    converged = lambda[14] > 0.5f0

    # 6. Evaluate parent f at all 39 directions, tracking realizability.
    all_finite_nonneg = true
    @inbounds for k in Int32(1):Int32(39)
        cxk_eq = Float32(cx_arr[k]) - shift_x
        cyk_eq = Float32(cy_arr[k]) - shift_y
        czk_eq = Float32(cz_arr[k]) - shift_z
        wk_eq  = Float32(w_arr[k])
        fk = eval_13mom_feq_single(
            rho_p, lambda, cxk_eq, cyk_eq, czk_eq, wk_eq, cs2)
        f_out[k] = fk
        # eval_13mom_feq_single floors at 1e-15 (>0), so the non-negativity is
        # structural; we still guard finiteness in case rho_p·exp overflows.
        all_finite_nonneg &= (isfinite(fk) & (fk >= 0.0f0))
    end

    # 7. GOLD-STANDARD VALIDATION (2026-06-25): recompute the realized 13
    #    moments of f_out and confirm the stress (5-10) + energy-flux (11-13)
    #    moments did not BLOW UP relative to the targets. A spiked, positive-
    #    but-inconsistent distribution (the off-axis failure mode) realizes
    #    |M| ≫ |target|. Accept only when each realized higher moment stays
    #    within ~10× the target magnitude (+ an absolute ρ-scaled floor so a
    #    near-zero target isn't a false trip).
    M_chk = compute_13_moments_of_populations_d3q39(
        f_out, cx_arr, cy_arr, cz_arr, cs2, shift_x, shift_y, shift_z)
    abs_floor = 1.0f-3 * rho_p
    # Explicit per-moment bound (fully unrolled — all tuple indices are
    # compile-time literals; no closures, no runtime tuple/array indexing →
    # GPU-codegen-safe). A realized higher moment is "blown up" if it exceeds
    # 10× its target magnitude plus a ρ-scaled absolute floor (so a near-zero
    # target can't false-trip). `f5 = 10·|mtX| + abs_floor` per moment.
    moments_bounded =
        (abs(M_chk[5])  <= 10.0f0*abs(mt5)  + abs_floor) &
        (abs(M_chk[6])  <= 10.0f0*abs(mt6)  + abs_floor) &
        (abs(M_chk[7])  <= 10.0f0*abs(mt7)  + abs_floor) &
        (abs(M_chk[8])  <= 10.0f0*abs(mt8)  + abs_floor) &
        (abs(M_chk[9])  <= 10.0f0*abs(mt9)  + abs_floor) &
        (abs(M_chk[10]) <= 10.0f0*abs(mt10) + abs_floor) &
        (abs(M_chk[11]) <= 10.0f0*abs(mt11) + abs_floor) &
        (abs(M_chk[12]) <= 10.0f0*abs(mt12) + abs_floor) &
        (abs(M_chk[13]) <= 10.0f0*abs(mt13) + abs_floor)

    ok = converged & all_finite_nonneg & moments_bounded
    return ok
end

# Simple polynomial equilibrium for D3Q39 (used as fallback and for BCs)
# Uses the standard 2nd-order Hermite expansion with cs² = 2/3
# O(u²) polynomial equilibrium for D3Q39.
# Phase 4: T parameter — effective sound speed becomes cs²·T. Default T=1 for athermal.
@inline function calculate_equilibrium_d3q39(rho::Float32, ux::Float32, uy::Float32, uz::Float32,
                                              w_k::Float32, cx::Float32, cy::Float32, cz::Float32,
                                              cs2::Float32, T::Float32=1.0f0)
    cs2T = cs2 * T
    inv_cs2 = 1.0f0 / cs2T
    inv_2cs4 = 0.5f0 * inv_cs2 * inv_cs2
    cu = cx*ux + cy*uy + cz*uz
    usq = ux*ux + uy*uy + uz*uz
    return rho * w_k * (1.0f0 + cu*inv_cs2 + cu*cu*inv_2cs4 - usq*inv_cs2*0.5f0)
end

# O(u⁴) polynomial equilibrium for D3Q39 — extended accuracy for compressible flows.
# Includes 3rd and 4th order Hermite terms. Compiles instantly (no iterative solver).
# Phase 4: T parameter — effective sound speed becomes cs²·T (thermal EOS: p = ρ·cs²·T).
# When T=1.0 (default), this is mathematically identical to the athermal version.
@inline function calculate_equilibrium_d3q39_4th(rho::Float32, ux::Float32, uy::Float32, uz::Float32,
                                                  w_k::Float32, cx::Float32, cy::Float32, cz::Float32,
                                                  cs2::Float32, T::Float32=1.0f0)
    cs2T = cs2 * T
    inv_cs2 = 1.0f0 / cs2T
    inv_cs4 = inv_cs2 * inv_cs2
    cu = cx*ux + cy*uy + cz*uz
    usq = ux*ux + uy*uy + uz*uz

    # O(u²) terms (standard)
    t2 = cu * inv_cs2 + 0.5f0 * cu * cu * inv_cs4 - 0.5f0 * usq * inv_cs2

    # O(u³) terms
    inv_cs6 = inv_cs4 * inv_cs2
    t3 = cu * cu * cu * inv_cs6 / 6.0f0 - cu * usq * inv_cs4 * 0.5f0

    # O(u⁴) terms
    inv_cs8 = inv_cs4 * inv_cs4
    cu2 = cu * cu
    t4 = cu2 * cu2 * inv_cs8 / 24.0f0 - cu2 * usq * inv_cs6 / 4.0f0 + usq * usq * inv_cs4 / 8.0f0

    return rho * w_k * (1.0f0 + t2 + t3 + t4)
end

@inline function local_shift_remap_x_d3q39(cx_arr, cy_arr, cz_arr, k::Int32, delta_x::Int32)
    return local_shift_remap_d3q39(cx_arr, cy_arr, cz_arr, k, delta_x, Int32(0), Int32(0))
end

@inline function local_shift_encode3_d3q39(sx::Int32, sy::Int32, sz::Int32)
    return sx + Int32(10) * sy + Int32(100) * sz
end

@inline function local_shift_decode_x_d3q39(code::Int32)
    sz = div(code, Int32(100))
    rem_z = code - Int32(100) * sz
    sy = div(rem_z, Int32(10))
    return rem_z - Int32(10) * sy
end

@inline function local_shift_decode_y_d3q39(code::Int32)
    sz = div(code, Int32(100))
    rem_z = code - Int32(100) * sz
    return div(rem_z, Int32(10))
end

@inline function local_shift_decode_z_d3q39(code::Int32)
    return div(code, Int32(100))
end

@inline function local_shift_remap_d3q39(cx_arr, cy_arr, cz_arr,
                                        k::Int32,
                                        delta_x::Int32,
                                        delta_y::Int32,
                                        delta_z::Int32)
    target_cx = Int32(cx_arr[k]) + delta_x
    target_cy = Int32(cy_arr[k])
    target_cy += delta_y
    target_cz = Int32(cz_arr[k]) + delta_z
    for j in Int32(1):Int32(39)
        if Int32(cx_arr[j]) == target_cx &&
           Int32(cy_arr[j]) == target_cy &&
           Int32(cz_arr[j]) == target_cz
            return j
        end
    end
    return Int32(0)
end

@inline function local_shift_fallback_feq3_d3q39(rho::Float32,
                                                 ux_lab::Float32, uy_lab::Float32, uz_lab::Float32,
                                                 frame_shift_x::Float32,
                                                 frame_shift_y::Float32,
                                                 frame_shift_z::Float32,
                                                 k::Int32,
                                                 cx_arr, cy_arr, cz_arr, w_arr,
                                                 cs2::Float32, T::Float32)
    array_shift_x = Float32(cx_arr[Int32(1)])
    array_shift_y = Float32(cy_arr[Int32(1)])
    array_shift_z = Float32(cz_arr[Int32(1)])
    ux_rel = ux_lab - frame_shift_x
    uy_rel = uy_lab - frame_shift_y
    uz_rel = uz_lab - frame_shift_z
    feq = calculate_equilibrium_d3q39_4th(
        max(rho, 0.01f0), ux_rel, uy_rel, uz_rel,
        Float32(w_arr[k]),
        Float32(cx_arr[k]) - array_shift_x,
        Float32(cy_arr[k]) - array_shift_y,
        Float32(cz_arr[k]) - array_shift_z,
        cs2, T
    )
    return max(feq, 1.0f-10)
end

@inline function local_shift_fallback_feq_d3q39(rho::Float32,
                                                ux_lab::Float32, uy_lab::Float32, uz_lab::Float32,
                                                frame_shift_x::Float32,
                                                k::Int32,
                                                cx_arr, cy_arr, cz_arr, w_arr,
                                                cs2::Float32, T::Float32)
    return local_shift_fallback_feq3_d3q39(
        rho, ux_lab, uy_lab, uz_lab,
        frame_shift_x, 0.0f0, 0.0f0,
        k, cx_arr, cy_arr, cz_arr, w_arr, cs2, T)
end

@inline function local_shift_regularized_reconstruct3_d3q39(f_arr,
                                                            sx::Int32, sy::Int32, sz::Int32, sb::Int32,
                                                            fallback_rho::Float32,
                                                            fallback_ux::Float32,
                                                            fallback_uy::Float32,
                                                            fallback_uz::Float32,
                                                            src_frame_x::Float32,
                                                            src_frame_y::Float32,
                                                            src_frame_z::Float32,
                                                            dst_frame_x::Float32,
                                                            dst_frame_y::Float32,
                                                            dst_frame_z::Float32,
                                                            k::Int32,
                                                            cx_arr, cy_arr, cz_arr, w_arr,
                                                            cs2::Float32, T::Float32)
    array_shift_x = Float32(cx_arr[Int32(1)])
    array_shift_y = Float32(cy_arr[Int32(1)])
    array_shift_z = Float32(cz_arr[Int32(1)])
    rho = 0.0f0
    jx = 0.0f0
    jy = 0.0f0
    jz = 0.0f0
    for j in Int32(1):Int32(39)
        fj = max(f_arr[sx, sy, sz, sb, j], 1.0f-10)
        qx = Float32(cx_arr[j]) - array_shift_x
        qy = Float32(cy_arr[j]) - array_shift_y
        qz = Float32(cz_arr[j]) - array_shift_z
        rho += fj
        jx += fj * (qx + src_frame_x)
        jy += fj * (qy + src_frame_y)
        jz += fj * (qz + src_frame_z)
    end

    if !(rho == rho) || rho < 0.01f0
        rho = max(fallback_rho, 0.01f0)
        jx = rho * fallback_ux
        jy = rho * fallback_uy
        jz = rho * fallback_uz
    end
    inv_rho = 1.0f0 / max(rho, 0.01f0)
    ux_lab = jx * inv_rho
    uy_lab = jy * inv_rho
    uz_lab = jz * inv_rho
    if !(ux_lab == ux_lab) || !(uy_lab == uy_lab) || !(uz_lab == uz_lab)
        ux_lab = fallback_ux
        uy_lab = fallback_uy
        uz_lab = fallback_uz
    end

    pi_xx = 0.0f0
    pi_yy = 0.0f0
    pi_zz = 0.0f0
    pi_xy = 0.0f0
    pi_xz = 0.0f0
    pi_yz = 0.0f0
    for j in Int32(1):Int32(39)
        fj = max(f_arr[sx, sy, sz, sb, j], 1.0f-10)
        feq_j = local_shift_fallback_feq3_d3q39(
            rho, ux_lab, uy_lab, uz_lab,
            src_frame_x, src_frame_y, src_frame_z,
            j, cx_arr, cy_arr, cz_arr, w_arr, cs2, T)
        fneq = fj - feq_j
        qx = Float32(cx_arr[j]) - array_shift_x
        qy = Float32(cy_arr[j]) - array_shift_y
        qz = Float32(cz_arr[j]) - array_shift_z
        pi_xx += fneq * (qx*qx - cs2)
        pi_yy += fneq * (qy*qy - cs2)
        pi_zz += fneq * (qz*qz - cs2)
        pi_xy += fneq * (qx*qy)
        pi_xz += fneq * (qx*qz)
        pi_yz += fneq * (qy*qz)
    end

    feq_k = local_shift_fallback_feq3_d3q39(
        rho, ux_lab, uy_lab, uz_lab,
        dst_frame_x, dst_frame_y, dst_frame_z,
        k, cx_arr, cy_arr, cz_arr, w_arr, cs2, T)
    qxk = Float32(cx_arr[k]) - array_shift_x
    qyk = Float32(cy_arr[k]) - array_shift_y
    qzk = Float32(cz_arr[k]) - array_shift_z
    inv_2cs4 = 0.5f0 / max(cs2 * cs2, 1.0f-12)
    fneq_k = Float32(w_arr[k]) * inv_2cs4 * (
        pi_xx * (qxk*qxk - cs2) +
        pi_yy * (qyk*qyk - cs2) +
        pi_zz * (qzk*qzk - cs2) +
        2.0f0 * pi_xy * (qxk*qyk) +
        2.0f0 * pi_xz * (qxk*qzk) +
        2.0f0 * pi_yz * (qyk*qzk)
    )

    # PonD gauge-transition correction (2026-06-29): a frame transition must carry the FULL
    # non-equilibrium stress across the gauge (Dorschner-Bösch-Karlin PRL 121, 130602 2018;
    # the moments are Galilean/gauge-invariant — P1 Eq.3). The legacy code damped this Grad
    # reconstruction to 0.35× (0.08× transverse) and hard-clamped it, discarding ~65% of the
    # stress non-equilibrium → the block-interface speckle (the stress is what carries the
    # viscous/shock physics). Carry it FULL (scale 1.0); keep only a realizability bound so the
    # reconstructed population stays positive (a Knudsen-style safety valve, not a wholesale
    # damp). Transverse frame jumps are still attenuated (the 13-mom transverse remap is the
    # exact path there); streamwise U=0↔U=1 — the case here — is now un-damped.
    transverse_frame_jump = (abs(src_frame_y - dst_frame_y) + abs(src_frame_z - dst_frame_z)) > 0.5f0
    fneq_scale = transverse_frame_jump ? 0.25f0 : 1.0f0
    # realizability bound: keep f_k = feq_k + fneq_k strictly positive, but do not throttle.
    fneq_rel_low = transverse_frame_jump ? 0.30f0 : 0.98f0
    fneq_rel_high = transverse_frame_jump ? 0.50f0 : 3.0f0
    fneq_k = clamp(fneq_scale * fneq_k, -fneq_rel_low * feq_k, fneq_rel_high * feq_k)
    return max(feq_k + fneq_k, 1.0f-10)
end

@inline function local_shift_regularized_reconstruct_x_d3q39(f_arr,
                                                             sx::Int32, sy::Int32, sz::Int32, sb::Int32,
                                                             fallback_rho::Float32,
                                                             fallback_ux::Float32,
                                                             fallback_uy::Float32,
                                                             fallback_uz::Float32,
                                                             src_frame_x::Float32,
                                                             dst_frame_x::Float32,
                                                             k::Int32,
                                                             cx_arr, cy_arr, cz_arr, w_arr,
                                                             cs2::Float32, T::Float32)
    array_shift_x = Float32(cx_arr[Int32(1)])
    array_shift_y = Float32(cy_arr[Int32(1)])
    array_shift_z = Float32(cz_arr[Int32(1)])
    rho = 0.0f0
    jx = 0.0f0
    jy = 0.0f0
    jz = 0.0f0
    for j in Int32(1):Int32(39)
        fj = max(f_arr[sx, sy, sz, sb, j], 1.0f-10)
        qx = Float32(cx_arr[j]) - array_shift_x
        qy = Float32(cy_arr[j]) - array_shift_y
        qz = Float32(cz_arr[j]) - array_shift_z
        rho += fj
        jx += fj * (qx + src_frame_x)
        jy += fj * qy
        jz += fj * qz
    end

    if !(rho == rho) || rho < 0.01f0
        rho = max(fallback_rho, 0.01f0)
        jx = rho * fallback_ux
        jy = rho * fallback_uy
        jz = rho * fallback_uz
    end
    inv_rho = 1.0f0 / max(rho, 0.01f0)
    ux_lab = jx * inv_rho
    uy_lab = jy * inv_rho
    uz_lab = jz * inv_rho
    if !(ux_lab == ux_lab) || !(uy_lab == uy_lab) || !(uz_lab == uz_lab)
        ux_lab = fallback_ux
        uy_lab = fallback_uy
        uz_lab = fallback_uz
    end

    pi_xx = 0.0f0
    pi_yy = 0.0f0
    pi_zz = 0.0f0
    pi_xy = 0.0f0
    pi_xz = 0.0f0
    pi_yz = 0.0f0
    for j in Int32(1):Int32(39)
        fj = max(f_arr[sx, sy, sz, sb, j], 1.0f-10)
        feq_j = local_shift_fallback_feq_d3q39(
            rho, ux_lab, uy_lab, uz_lab, src_frame_x,
            j, cx_arr, cy_arr, cz_arr, w_arr, cs2, T)
        fneq = fj - feq_j
        qx = Float32(cx_arr[j]) - array_shift_x
        qy = Float32(cy_arr[j]) - array_shift_y
        qz = Float32(cz_arr[j]) - array_shift_z
        pi_xx += fneq * (qx*qx - cs2)
        pi_yy += fneq * (qy*qy - cs2)
        pi_zz += fneq * (qz*qz - cs2)
        pi_xy += fneq * (qx*qy)
        pi_xz += fneq * (qx*qz)
        pi_yz += fneq * (qy*qz)
    end

    feq_k = local_shift_fallback_feq_d3q39(
        rho, ux_lab, uy_lab, uz_lab, dst_frame_x,
        k, cx_arr, cy_arr, cz_arr, w_arr, cs2, T)
    qxk = Float32(cx_arr[k]) - array_shift_x
    qyk = Float32(cy_arr[k]) - array_shift_y
    qzk = Float32(cz_arr[k]) - array_shift_z
    inv_2cs4 = 0.5f0 / max(cs2 * cs2, 1.0f-12)
    fneq_k = Float32(w_arr[k]) * inv_2cs4 * (
        pi_xx * (qxk*qxk - cs2) +
        pi_yy * (qyk*qyk - cs2) +
        pi_zz * (qzk*qzk - cs2) +
        2.0f0 * pi_xy * (qxk*qyk) +
        2.0f0 * pi_xz * (qxk*qzk) +
        2.0f0 * pi_yz * (qyk*qzk)
    )

    # Frame interfaces should preserve the resolved viscous/shear stress well
    # enough not to behave like a thin equilibrium sponge. Keep the projection
    # bounded for positivity, but do not collapse it almost entirely to feq.
    fneq_k = clamp(0.35f0 * fneq_k, -0.30f0 * feq_k, 0.70f0 * feq_k)
    return max(feq_k + fneq_k, 1.0f-10)
end


@inline function get_scalar_neighbor(field_in, x, y, z, b_idx, dx, dy, dz, block_size, neighbor_table)
    nx = Int32(x) + Int32(dx)
    ny = Int32(y) + Int32(dy)
    nz = Int32(z) + Int32(dz)
    ibs = Int32(block_size)
    ib = Int32(b_idx)

    if nx >= Int32(1) && nx <= ibs && ny >= Int32(1) && ny <= ibs && nz >= Int32(1) && nz <= ibs
        return field_in[nx, ny, nz, ib]
    end

    off_x = nx < Int32(1) ? Int32(-1) : (nx > ibs ? Int32(1) : Int32(0))
    off_y = ny < Int32(1) ? Int32(-1) : (ny > ibs ? Int32(1) : Int32(0))
    off_z = nz < Int32(1) ? Int32(-1) : (nz > ibs ? Int32(1) : Int32(0))
    dir_idx = (off_x + Int32(1)) + (off_y + Int32(1))*Int32(3) + (off_z + Int32(1))*Int32(9) + Int32(1)
    nb_idx = neighbor_table[ib, dir_idx]

    if nb_idx > Int32(0)
        nnx = nx < Int32(1) ? nx + ibs : (nx > ibs ? nx - ibs : nx)
        nny = ny < Int32(1) ? ny + ibs : (ny > ibs ? ny - ibs : ny)
        nnz = nz < Int32(1) ? nz + ibs : (nz > ibs ? nz - ibs : nz)
        return field_in[nnx, nny, nnz, nb_idx]
    end

    return field_in[x, y, z, ib]
end

@inline function get_scalar_neighbor_fluid(field_in, obstacle, x, y, z, b_idx, dx, dy, dz, block_size, neighbor_table)
    nx = Int32(x) + Int32(dx)
    ny = Int32(y) + Int32(dy)
    nz = Int32(z) + Int32(dz)
    ibs = Int32(block_size)
    ib = Int32(b_idx)

    if nx >= Int32(1) && nx <= ibs && ny >= Int32(1) && ny <= ibs && nz >= Int32(1) && nz <= ibs
        return obstacle[nx, ny, nz, ib] ? field_in[x, y, z, ib] : field_in[nx, ny, nz, ib]
    end

    off_x = nx < Int32(1) ? Int32(-1) : (nx > ibs ? Int32(1) : Int32(0))
    off_y = ny < Int32(1) ? Int32(-1) : (ny > ibs ? Int32(1) : Int32(0))
    off_z = nz < Int32(1) ? Int32(-1) : (nz > ibs ? Int32(1) : Int32(0))
    dir_idx = (off_x + Int32(1)) + (off_y + Int32(1))*Int32(3) + (off_z + Int32(1))*Int32(9) + Int32(1)
    nb_idx = neighbor_table[ib, dir_idx]

    if nb_idx > Int32(0)
        nnx = nx < Int32(1) ? nx + ibs : (nx > ibs ? nx - ibs : nx)
        nny = ny < Int32(1) ? ny + ibs : (ny > ibs ? ny - ibs : ny)
        nnz = nz < Int32(1) ? nz + ibs : (nz > ibs ? nz - ibs : nz)
        return obstacle[nnx, nny, nnz, nb_idx] ? field_in[x, y, z, ib] : field_in[nnx, nny, nnz, nb_idx]
    end

    return field_in[x, y, z, ib]
end

@inline function sample_parent_scalar_at_fine_coord(
    parent_field_new, parent_field_old, parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    fine_gx::Int32, fine_gy::Int32, fine_gz::Int32,
    block_size::Int32,
    temporal_weight::Float32,
    use_temporal_interp::Int32,
    default_value::Float32
)
    px_cont = _fine_to_parent_interp_coord_scalar(fine_gx)
    py_cont = _fine_to_parent_interp_coord_scalar(fine_gy)
    pz_cont = _fine_to_parent_interp_coord_scalar(fine_gz)

    parent_nx = parent_dim_x * block_size
    parent_ny = parent_dim_y * block_size
    parent_nz = parent_dim_z * block_size
    px0, px1, wx = _parent_interp_bracket_scalar(px_cont, parent_nx)
    py0, py1, wy = _PERIODIC_Y_AMR ? _pbracket_y(py_cont) : _parent_interp_bracket_scalar(py_cont, parent_ny)
    pz0, pz1, wz = _parent_interp_bracket_scalar(pz_cont, parent_nz)

    @inline function get_parent_scalar(pgx::Int32, pgy::Int32, pgz::Int32)
        pgy = _pwrap_y(pgy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
        pbx = (pgx - Int32(1)) ÷ block_size + Int32(1)
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

                val_new = parent_field_new[plx, ply, plz, pb_idx]
                if use_temporal_interp == Int32(1) && temporal_weight < 0.99f0
                    val_old = parent_field_old[plx, ply, plz, pb_idx]
                    tw = temporal_weight
                    return (val_old * (1.0f0 - tw) + val_new * tw, true)
                end
                return (val_new, true)
            end
        end

        return (default_value, false)
    end

    d000 = get_parent_scalar(px0, py0, pz0)
    d100 = get_parent_scalar(px1, py0, pz0)
    d010 = get_parent_scalar(px0, py1, pz0)
    d110 = get_parent_scalar(px1, py1, pz0)
    d001 = get_parent_scalar(px0, py0, pz1)
    d101 = get_parent_scalar(px1, py0, pz1)
    d011 = get_parent_scalar(px0, py1, pz1)
    d111 = get_parent_scalar(px1, py1, pz1)

    w000 = (1.0f0 - wx) * (1.0f0 - wy) * (1.0f0 - wz)
    w100 = wx * (1.0f0 - wy) * (1.0f0 - wz)
    w010 = (1.0f0 - wx) * wy * (1.0f0 - wz)
    w110 = wx * wy * (1.0f0 - wz)
    w001 = (1.0f0 - wx) * (1.0f0 - wy) * wz
    w101 = wx * (1.0f0 - wy) * wz
    w011 = (1.0f0 - wx) * wy * wz
    w111 = wx * wy * wz

    accum = 0.0f0
    wsum = 0.0f0
    if d000[2]; accum += w000 * d000[1]; wsum += w000; end
    if d100[2]; accum += w100 * d100[1]; wsum += w100; end
    if d010[2]; accum += w010 * d010[1]; wsum += w010; end
    if d110[2]; accum += w110 * d110[1]; wsum += w110; end
    if d001[2]; accum += w001 * d001[1]; wsum += w001; end
    if d101[2]; accum += w101 * d101[1]; wsum += w101; end
    if d011[2]; accum += w011 * d011[1]; wsum += w011; end
    if d111[2]; accum += w111 * d111[1]; wsum += w111; end

    val = if wsum > Float32(1e-12)
        accum / wsum
    else
        default_value
    end

    return val == val ? val : default_value
end

@inline function get_scalar_neighbor_fluid_amr(
    field_in, obstacle,
    x, y, z, b_idx,
    dx, dy, dz,
    block_size, neighbor_table,
    is_level_1::Int32,
    parent_field_new, parent_field_old, parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    gx::Int32, gy::Int32, gz::Int32,
    temporal_weight::Float32,
    use_temporal_interp::Int32
)
    nx = Int32(x) + Int32(dx)
    ny = Int32(y) + Int32(dy)
    nz = Int32(z) + Int32(dz)
    ibs = Int32(block_size)
    ib = Int32(b_idx)

    if nx >= Int32(1) && nx <= ibs && ny >= Int32(1) && ny <= ibs && nz >= Int32(1) && nz <= ibs
        return obstacle[nx, ny, nz, ib] ? field_in[x, y, z, ib] : field_in[nx, ny, nz, ib]
    end

    off_x = nx < Int32(1) ? Int32(-1) : (nx > ibs ? Int32(1) : Int32(0))
    off_y = ny < Int32(1) ? Int32(-1) : (ny > ibs ? Int32(1) : Int32(0))
    off_z = nz < Int32(1) ? Int32(-1) : (nz > ibs ? Int32(1) : Int32(0))
    dir_idx = (off_x + Int32(1)) + (off_y + Int32(1))*Int32(3) + (off_z + Int32(1))*Int32(9) + Int32(1)
    nb_idx = neighbor_table[ib, dir_idx]

    if nb_idx > Int32(0)
        nnx = nx < Int32(1) ? nx + ibs : (nx > ibs ? nx - ibs : nx)
        nny = ny < Int32(1) ? ny + ibs : (ny > ibs ? ny - ibs : ny)
        nnz = nz < Int32(1) ? nz + ibs : (nz > ibs ? nz - ibs : nz)
        return obstacle[nnx, nny, nnz, nb_idx] ? field_in[x, y, z, ib] : field_in[nnx, nny, nnz, nb_idx]
    end

    center_val = field_in[x, y, z, ib]
    # This helper is used by limiter/feedback operators, not by streaming.
    # Sampling the parent here turns a small coarse/fine interpolation mismatch
    # into a level-shaped source term. Keep derivative-like stencils level-local;
    # the population streaming path remains the only cross-level transfer.
    return center_val
end

@inline function get_scalar_neighbor_stats_26_amr(
    field_in, obstacle,
    x, y, z, b_idx,
    block_size, neighbor_table,
    is_level_1::Int32,
    parent_field_new, parent_field_old, parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    gx::Int32, gy::Int32, gz::Int32,
    temporal_weight::Float32,
    use_temporal_interp::Int32
)
    sum_w = 0.0f0
    sum_T = 0.0f0
    Tmin = Float32(Inf)
    Tmax = -Float32(Inf)

    for dz in -1:1
        for dy in -1:1
            for dx in -1:1
                if dx == 0 && dy == 0 && dz == 0
                    continue
                end
                val = get_scalar_neighbor_fluid_amr(
                    field_in, obstacle,
                    x, y, z, b_idx, dx, dy, dz,
                    block_size, neighbor_table,
                    is_level_1,
                    parent_field_new, parent_field_old, parent_ptr,
                    parent_dim_x, parent_dim_y, parent_dim_z,
                    gx, gy, gz,
                    temporal_weight, use_temporal_interp
                )
                manhattan = abs(dx) + abs(dy) + abs(dz)
                w = manhattan == 1 ? 4.0f0 : (manhattan == 2 ? 2.0f0 : 1.0f0)
                sum_w += w
                sum_T += w * val
                Tmin = min(Tmin, val)
                Tmax = max(Tmax, val)
            end
        end
    end

    return sum_T / max(sum_w, 1.0f0), Tmin, Tmax
end

@inline function limit_pair_odd_mode(q::Float32, geq_k::Float32, geq_opp::Float32, factor::Float32)
    geq_pair_min = max(min(geq_k, geq_opp), Float32(1e-10))
    q_bound = max(factor, 0.0f0) * geq_pair_min
    return clamp(q, -q_bound, q_bound)
end

@inline function get_velocity_neighbor(vel_in, x, y, z, b_idx, dx, dy, dz, block_size, neighbor_table)
    nx = Int32(x) + Int32(dx)
    ny = Int32(y) + Int32(dy)
    nz = Int32(z) + Int32(dz)
    ibs = Int32(block_size)
    ib = Int32(b_idx)
    
    if nx >= Int32(1) && nx <= ibs && ny >= Int32(1) && ny <= ibs && nz >= Int32(1) && nz <= ibs
        return vel_in[nx, ny, nz, ib, 1], vel_in[nx, ny, nz, ib, 2], vel_in[nx, ny, nz, ib, 3]
    end
    
    off_x = nx < Int32(1) ? Int32(-1) : (nx > ibs ? Int32(1) : Int32(0))
    off_y = ny < Int32(1) ? Int32(-1) : (ny > ibs ? Int32(1) : Int32(0))
    off_z = nz < Int32(1) ? Int32(-1) : (nz > ibs ? Int32(1) : Int32(0))
    dir_idx = (off_x + Int32(1)) + (off_y + Int32(1))*Int32(3) + (off_z + Int32(1))*Int32(9) + Int32(1)
    nb_idx = neighbor_table[ib, dir_idx]
    
    if nb_idx > Int32(0)
        nnx = nx < Int32(1) ? nx + ibs : (nx > ibs ? nx - ibs : nx)
        nny = ny < Int32(1) ? ny + ibs : (ny > ibs ? ny - ibs : ny)
        nnz = nz < Int32(1) ? nz + ibs : (nz > ibs ? nz - ibs : nz)
        return vel_in[nnx, nny, nnz, nb_idx, 1], vel_in[nnx, nny, nnz, nb_idx, 2], vel_in[nnx, nny, nnz, nb_idx, 3]
    end
    
    return vel_in[x, y, z, ib, 1], vel_in[x, y, z, ib, 2], vel_in[x, y, z, ib, 3]
end

@inline function get_velocity_neighbor_fluid(
    vel_in, obstacle,
    x, y, z, b_idx,
    dx, dy, dz,
    block_size, neighbor_table,
    sym_y_plane::Int32 = Int32(0)   # 1 = free-slip y-plane: mirror-ghost off-domain y-steps
)
    nx = Int32(x) + Int32(dx)
    ny = Int32(y) + Int32(dy)
    nz = Int32(z) + Int32(dz)
    ibs = Int32(block_size)
    ib = Int32(b_idx)

    ux0 = vel_in[x, y, z, ib, 1]
    uy0 = vel_in[x, y, z, ib, 2]
    uz0 = vel_in[x, y, z, ib, 3]

    if nx >= Int32(1) && nx <= ibs && ny >= Int32(1) && ny <= ibs && nz >= Int32(1) && nz <= ibs
        if obstacle[nx, ny, nz, ib]
            return ux0, uy0, uz0
        end
        return vel_in[nx, ny, nz, ib, 1], vel_in[nx, ny, nz, ib, 2], vel_in[nx, ny, nz, ib, 3]
    end

    off_x = nx < Int32(1) ? Int32(-1) : (nx > ibs ? Int32(1) : Int32(0))
    off_y = ny < Int32(1) ? Int32(-1) : (ny > ibs ? Int32(1) : Int32(0))
    off_z = nz < Int32(1) ? Int32(-1) : (nz > ibs ? Int32(1) : Int32(0))
    dir_idx = (off_x + Int32(1)) + (off_y + Int32(1))*Int32(3) + (off_z + Int32(1))*Int32(9) + Int32(1)
    nb_idx = neighbor_table[ib, dir_idx]

    if nb_idx > Int32(0)
        nnx = nx < Int32(1) ? nx + ibs : (nx > ibs ? nx - ibs : nx)
        nny = ny < Int32(1) ? ny + ibs : (ny > ibs ? ny - ibs : ny)
        nnz = nz < Int32(1) ? nz + ibs : (nz > ibs ? nz - ibs : nz)
        if obstacle[nnx, nny, nnz, nb_idx]
            return ux0, uy0, uz0
        end
        return vel_in[nnx, nny, nnz, nb_idx, 1], vel_in[nnx, nny, nnz, nb_idx, 2], vel_in[nnx, nny, nnz, nb_idx, 3]
    end

    # SYMMETRY-PLANE FIX (2026-06-04): off-domain step across a free-slip y-plane.
    # Return the MIRROR GHOST (ux→+ux, u_y→−u_y, uz→+uz) so the WALE strain tensor is
    # symmetric across the plane. The legacy zero-gradient fallback (own velocity) gave
    # an asymmetric eddy viscosity at plane cells that re-injected u_y EVERY step — the
    # systematic per-step source behind the GROWING symmetry residual (max 7e-3→0.28).
    # The first-order ghost of a cell sitting on the plane is the cell itself with u_y
    # flipped (cell reflected across the plane at distance 1 ≡ itself, normal flipped);
    # the single −uy0 form is correct for both the y=0 and y=max planes (u_y is the
    # wall-normal component for both). GPU-safe: gated on the passed Int32 param, no
    # module-global read inside the kernel. Default 0 ⇒ other operators byte-identical.
    if sym_y_plane == Int32(1) && off_y != Int32(0)
        return ux0, -uy0, uz0
    end
    return ux0, uy0, uz0
end

@inline function get_parent_velocity_sample(
    parent_vel_new, parent_vel_old, parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    pgx::Int32, pgy::Int32, pgz::Int32,
    block_size::Int32,
    temporal_weight::Float32,
    use_temporal_interp::Int32,
    default_ux::Float32, default_uy::Float32, default_uz::Float32
)
    pgy = _pwrap_y(pgy, parent_dim_y * block_size)   # periodic-Y AMR (identity when off)
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

            ux_v = parent_vel_new[plx, ply, plz, pb_idx, 1]
            uy_v = parent_vel_new[plx, ply, plz, pb_idx, 2]
            uz_v = parent_vel_new[plx, ply, plz, pb_idx, 3]
            if use_temporal_interp == Int32(1) && temporal_weight < 0.99f0
                tw = temporal_weight
                ux_v = parent_vel_old[plx, ply, plz, pb_idx, 1] * (1.0f0 - tw) + ux_v * tw
                uy_v = parent_vel_old[plx, ply, plz, pb_idx, 2] * (1.0f0 - tw) + uy_v * tw
                uz_v = parent_vel_old[plx, ply, plz, pb_idx, 3] * (1.0f0 - tw) + uz_v * tw
            end
            return ux_v, uy_v, uz_v, 1.0f0
        end
    end

    return default_ux, default_uy, default_uz, 0.0f0
end

@inline function sample_parent_velocity_at_fine_coord(
    parent_vel_new, parent_vel_old, parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    fine_gx::Int32, fine_gy::Int32, fine_gz::Int32,
    block_size::Int32,
    temporal_weight::Float32,
    use_temporal_interp::Int32,
    default_ux::Float32, default_uy::Float32, default_uz::Float32
)
    px_cont = _fine_to_parent_interp_coord_scalar(fine_gx)
    py_cont = _fine_to_parent_interp_coord_scalar(fine_gy)
    pz_cont = _fine_to_parent_interp_coord_scalar(fine_gz)

    parent_nx = parent_dim_x * block_size
    parent_ny = parent_dim_y * block_size
    parent_nz = parent_dim_z * block_size
    px0, px1, wx = _parent_interp_bracket_scalar(px_cont, parent_nx)
    py0, py1, wy = _PERIODIC_Y_AMR ? _pbracket_y(py_cont) : _parent_interp_bracket_scalar(py_cont, parent_ny)
    pz0, pz1, wz = _parent_interp_bracket_scalar(pz_cont, parent_nz)

    ux000, uy000, uz000, v000 = get_parent_velocity_sample(parent_vel_new, parent_vel_old, parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z, px0, py0, pz0, block_size, temporal_weight, use_temporal_interp, default_ux, default_uy, default_uz)
    ux100, uy100, uz100, v100 = get_parent_velocity_sample(parent_vel_new, parent_vel_old, parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z, px1, py0, pz0, block_size, temporal_weight, use_temporal_interp, default_ux, default_uy, default_uz)
    ux010, uy010, uz010, v010 = get_parent_velocity_sample(parent_vel_new, parent_vel_old, parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z, px0, py1, pz0, block_size, temporal_weight, use_temporal_interp, default_ux, default_uy, default_uz)
    ux110, uy110, uz110, v110 = get_parent_velocity_sample(parent_vel_new, parent_vel_old, parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z, px1, py1, pz0, block_size, temporal_weight, use_temporal_interp, default_ux, default_uy, default_uz)
    ux001, uy001, uz001, v001 = get_parent_velocity_sample(parent_vel_new, parent_vel_old, parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z, px0, py0, pz1, block_size, temporal_weight, use_temporal_interp, default_ux, default_uy, default_uz)
    ux101, uy101, uz101, v101 = get_parent_velocity_sample(parent_vel_new, parent_vel_old, parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z, px1, py0, pz1, block_size, temporal_weight, use_temporal_interp, default_ux, default_uy, default_uz)
    ux011, uy011, uz011, v011 = get_parent_velocity_sample(parent_vel_new, parent_vel_old, parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z, px0, py1, pz1, block_size, temporal_weight, use_temporal_interp, default_ux, default_uy, default_uz)
    ux111, uy111, uz111, v111 = get_parent_velocity_sample(parent_vel_new, parent_vel_old, parent_ptr, parent_dim_x, parent_dim_y, parent_dim_z, px1, py1, pz1, block_size, temporal_weight, use_temporal_interp, default_ux, default_uy, default_uz)

    w000 = (1.0f0 - wx) * (1.0f0 - wy) * (1.0f0 - wz) * v000
    w100 = wx * (1.0f0 - wy) * (1.0f0 - wz) * v100
    w010 = (1.0f0 - wx) * wy * (1.0f0 - wz) * v010
    w110 = wx * wy * (1.0f0 - wz) * v110
    w001 = (1.0f0 - wx) * (1.0f0 - wy) * wz * v001
    w101 = wx * (1.0f0 - wy) * wz * v101
    w011 = (1.0f0 - wx) * wy * wz * v011
    w111 = wx * wy * wz * v111

    wsum = w000 + w100 + w010 + w110 + w001 + w101 + w011 + w111
    if wsum > Float32(1e-12)
        inv_w = 1.0f0 / wsum
        ux_v = (w000*ux000 + w100*ux100 + w010*ux010 + w110*ux110 +
                w001*ux001 + w101*ux101 + w011*ux011 + w111*ux111) * inv_w
        uy_v = (w000*uy000 + w100*uy100 + w010*uy010 + w110*uy110 +
                w001*uy001 + w101*uy101 + w011*uy011 + w111*uy111) * inv_w
        uz_v = (w000*uz000 + w100*uz100 + w010*uz010 + w110*uz110 +
                w001*uz001 + w101*uz101 + w011*uz011 + w111*uz111) * inv_w
        if ux_v == ux_v && uy_v == uy_v && uz_v == uz_v
            return ux_v, uy_v, uz_v
        end
    end

    return default_ux, default_uy, default_uz
end

@inline function sample_parent_velocity_at_fine_coord(
    parent_vel_new::Nothing, parent_vel_old, parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    fine_gx::Int32, fine_gy::Int32, fine_gz::Int32,
    block_size::Int32,
    temporal_weight::Float32,
    use_temporal_interp::Int32,
    default_ux::Float32, default_uy::Float32, default_uz::Float32
)
    return default_ux, default_uy, default_uz
end

@inline function get_velocity_neighbor_amr(
    vel_in,
    x, y, z, b_idx,
    dx, dy, dz,
    block_size, neighbor_table,
    is_level_1::Int32,
    parent_vel_new, parent_vel_old, parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    gx::Int32, gy::Int32, gz::Int32,
    temporal_weight::Float32,
    use_temporal_interp::Int32
)
    nx = Int32(x) + Int32(dx)
    ny = Int32(y) + Int32(dy)
    nz = Int32(z) + Int32(dz)
    ibs = Int32(block_size)
    ib = Int32(b_idx)

    if nx >= Int32(1) && nx <= ibs && ny >= Int32(1) && ny <= ibs && nz >= Int32(1) && nz <= ibs
        return vel_in[nx, ny, nz, ib, 1], vel_in[nx, ny, nz, ib, 2], vel_in[nx, ny, nz, ib, 3]
    end

    off_x = nx < Int32(1) ? Int32(-1) : (nx > ibs ? Int32(1) : Int32(0))
    off_y = ny < Int32(1) ? Int32(-1) : (ny > ibs ? Int32(1) : Int32(0))
    off_z = nz < Int32(1) ? Int32(-1) : (nz > ibs ? Int32(1) : Int32(0))
    dir_idx = (off_x + Int32(1)) + (off_y + Int32(1))*Int32(3) + (off_z + Int32(1))*Int32(9) + Int32(1)
    nb_idx = neighbor_table[ib, dir_idx]

    if nb_idx > Int32(0)
        nnx = nx < Int32(1) ? nx + ibs : (nx > ibs ? nx - ibs : nx)
        nny = ny < Int32(1) ? ny + ibs : (ny > ibs ? ny - ibs : ny)
        nnz = nz < Int32(1) ? nz + ibs : (nz > ibs ? nz - ibs : nz)
        return vel_in[nnx, nny, nnz, nb_idx, 1], vel_in[nnx, nny, nnz, nb_idx, 2], vel_in[nnx, nny, nnz, nb_idx, 3]
    end

    ux0 = vel_in[x, y, z, ib, 1]
    uy0 = vel_in[x, y, z, ib, 2]
    uz0 = vel_in[x, y, z, ib, 3]
    # WALE/thermal-SGS gradients are local closure estimates. If their stencil
    # crosses an AMR owner boundary, parent interpolation noise is interpreted as
    # resolved shear and prints vorticity at the refinement face. Use a zero-
    # normal-gradient closure for the missing same-level side; true cross-level
    # physics is still carried by the distribution interpolation in streaming.
    return ux0, uy0, uz0
end

@inline function get_velocity_neighbor_amr(
    vel_in,
    x, y, z, b_idx,
    dx, dy, dz,
    block_size, neighbor_table,
    is_level_1::Int32,
    parent_vel_new::Nothing, parent_vel_old, parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    gx::Int32, gy::Int32, gz::Int32,
    temporal_weight::Float32,
    use_temporal_interp::Int32
)
    return get_velocity_neighbor(vel_in, x, y, z, b_idx, dx, dy, dz, block_size, neighbor_table)
end

@inline function compute_velocity_gradients(vel_in, x, y, z, b_idx, block_size, neighbor_table)
    ux_E, uy_E, uz_E = get_velocity_neighbor(vel_in, x, y, z, b_idx, 1, 0, 0, block_size, neighbor_table)
    ux_W, uy_W, uz_W = get_velocity_neighbor(vel_in, x, y, z, b_idx, -1, 0, 0, block_size, neighbor_table)
    ux_N, uy_N, uz_N = get_velocity_neighbor(vel_in, x, y, z, b_idx, 0, 1, 0, block_size, neighbor_table)
    ux_S, uy_S, uz_S = get_velocity_neighbor(vel_in, x, y, z, b_idx, 0, -1, 0, block_size, neighbor_table)
    ux_T, uy_T, uz_T = get_velocity_neighbor(vel_in, x, y, z, b_idx, 0, 0, 1, block_size, neighbor_table)
    ux_B, uy_B, uz_B = get_velocity_neighbor(vel_in, x, y, z, b_idx, 0, 0, -1, block_size, neighbor_table)
    
    return (0.5f0*(ux_E-ux_W), 0.5f0*(ux_N-ux_S), 0.5f0*(ux_T-ux_B),
            0.5f0*(uy_E-uy_W), 0.5f0*(uy_N-uy_S), 0.5f0*(uy_T-uy_B),
            0.5f0*(uz_E-uz_W), 0.5f0*(uz_N-uz_S), 0.5f0*(uz_T-uz_B))
end

@inline function compute_velocity_gradients_fluid(
    vel_in, obstacle,
    x, y, z, b_idx,
    block_size, neighbor_table,
    sym_y_plane::Int32 = Int32(0)   # forwarded to the neighbor fetch; mirror-ghosts off-domain y-steps at a free-slip plane
)
    ux_E, uy_E, uz_E = get_velocity_neighbor_fluid(vel_in, obstacle, x, y, z, b_idx, 1, 0, 0, block_size, neighbor_table, sym_y_plane)
    ux_W, uy_W, uz_W = get_velocity_neighbor_fluid(vel_in, obstacle, x, y, z, b_idx, -1, 0, 0, block_size, neighbor_table, sym_y_plane)
    ux_N, uy_N, uz_N = get_velocity_neighbor_fluid(vel_in, obstacle, x, y, z, b_idx, 0, 1, 0, block_size, neighbor_table, sym_y_plane)
    ux_S, uy_S, uz_S = get_velocity_neighbor_fluid(vel_in, obstacle, x, y, z, b_idx, 0, -1, 0, block_size, neighbor_table, sym_y_plane)
    ux_T, uy_T, uz_T = get_velocity_neighbor_fluid(vel_in, obstacle, x, y, z, b_idx, 0, 0, 1, block_size, neighbor_table, sym_y_plane)
    ux_B, uy_B, uz_B = get_velocity_neighbor_fluid(vel_in, obstacle, x, y, z, b_idx, 0, 0, -1, block_size, neighbor_table, sym_y_plane)

    return (0.5f0*(ux_E-ux_W), 0.5f0*(ux_N-ux_S), 0.5f0*(ux_T-ux_B),
            0.5f0*(uy_E-uy_W), 0.5f0*(uy_N-uy_S), 0.5f0*(uy_T-uy_B),
            0.5f0*(uz_E-uz_W), 0.5f0*(uz_N-uz_S), 0.5f0*(uz_T-uz_B))
end

@inline function compute_velocity_gradients_amr(
    vel_in,
    x, y, z, b_idx,
    block_size, neighbor_table,
    is_level_1::Int32,
    parent_vel_new::Nothing, parent_vel_old, parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    gx::Int32, gy::Int32, gz::Int32,
    temporal_weight::Float32,
    use_temporal_interp::Int32
)
    return compute_velocity_gradients(vel_in, x, y, z, b_idx, block_size, neighbor_table)
end

@inline function compute_velocity_gradients_amr(
    vel_in,
    x, y, z, b_idx,
    block_size, neighbor_table,
    is_level_1::Int32,
    parent_vel_new, parent_vel_old, parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    gx::Int32, gy::Int32, gz::Int32,
    temporal_weight::Float32,
    use_temporal_interp::Int32
)
    ux_E, uy_E, uz_E = get_velocity_neighbor_amr(
        vel_in, x, y, z, b_idx, 1, 0, 0, block_size, neighbor_table,
        is_level_1, parent_vel_new, parent_vel_old, parent_ptr,
        parent_dim_x, parent_dim_y, parent_dim_z, gx, gy, gz,
        temporal_weight, use_temporal_interp
    )
    ux_W, uy_W, uz_W = get_velocity_neighbor_amr(
        vel_in, x, y, z, b_idx, -1, 0, 0, block_size, neighbor_table,
        is_level_1, parent_vel_new, parent_vel_old, parent_ptr,
        parent_dim_x, parent_dim_y, parent_dim_z, gx, gy, gz,
        temporal_weight, use_temporal_interp
    )
    ux_N, uy_N, uz_N = get_velocity_neighbor_amr(
        vel_in, x, y, z, b_idx, 0, 1, 0, block_size, neighbor_table,
        is_level_1, parent_vel_new, parent_vel_old, parent_ptr,
        parent_dim_x, parent_dim_y, parent_dim_z, gx, gy, gz,
        temporal_weight, use_temporal_interp
    )
    ux_S, uy_S, uz_S = get_velocity_neighbor_amr(
        vel_in, x, y, z, b_idx, 0, -1, 0, block_size, neighbor_table,
        is_level_1, parent_vel_new, parent_vel_old, parent_ptr,
        parent_dim_x, parent_dim_y, parent_dim_z, gx, gy, gz,
        temporal_weight, use_temporal_interp
    )
    ux_T, uy_T, uz_T = get_velocity_neighbor_amr(
        vel_in, x, y, z, b_idx, 0, 0, 1, block_size, neighbor_table,
        is_level_1, parent_vel_new, parent_vel_old, parent_ptr,
        parent_dim_x, parent_dim_y, parent_dim_z, gx, gy, gz,
        temporal_weight, use_temporal_interp
    )
    ux_B, uy_B, uz_B = get_velocity_neighbor_amr(
        vel_in, x, y, z, b_idx, 0, 0, -1, block_size, neighbor_table,
        is_level_1, parent_vel_new, parent_vel_old, parent_ptr,
        parent_dim_x, parent_dim_y, parent_dim_z, gx, gy, gz,
        temporal_weight, use_temporal_interp
    )

    return (0.5f0*(ux_E-ux_W), 0.5f0*(ux_N-ux_S), 0.5f0*(ux_T-ux_B),
            0.5f0*(uy_E-uy_W), 0.5f0*(uy_N-uy_S), 0.5f0*(uy_T-uy_B),
            0.5f0*(uz_E-uz_W), 0.5f0*(uz_N-uz_S), 0.5f0*(uz_T-uz_B))
end

@inline function compute_velocity_gradients_amr_fluid(
    vel_in, obstacle,
    x, y, z, b_idx,
    block_size, neighbor_table,
    is_level_1::Int32,
    parent_vel_new, parent_vel_old, parent_ptr,
    parent_dim_x::Int32, parent_dim_y::Int32, parent_dim_z::Int32,
    gx::Int32, gy::Int32, gz::Int32,
    temporal_weight::Float32,
    use_temporal_interp::Int32
)
    # Closure/SGS gradients must be local fluid derivatives. Missing same-level
    # sides and solid neighbors use zero normal gradient; cross-level physics is
    # still carried by the population transfer, not by derivative stencils.
    return compute_velocity_gradients_fluid(
        vel_in, obstacle, x, y, z, b_idx, block_size, neighbor_table
    )
end
