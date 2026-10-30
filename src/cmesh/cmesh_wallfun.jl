# ============================================================================================
# EQUILIBRIUM LOG-LAW WALL FUNCTION for the cmesh KFVS solver  (2026-08-02, plan v3 S3i)
#
# WHY. cmesh has `wallmode` 0/1/2 = transparent / slip / no-slip and NO WALL FUNCTION AT ALL. The
# no-slip branch takes the wall shear from the resolved gradient, `tau = mu_eff u_1/d_1`, which is
# correct ONLY if the mesh resolves the viscous sublayer. Measured error of that assumption:
#
#     y+     u+_true   tau_noslip/tau_true
#      0.8     0.800        1.000     <- the VALIDATED clean-foil protocol: exact, no model needed
#      8.0     7.215        0.902     <- the cavitation mesh: 10% low, in the layer that makes the jet
#     30.0    12.787        0.426
#
# and EVERY published cavitation shedder runs wall functions at y+ 10-100 (Frikha, Leroux, Reboud,
# Coutier-Delgosha). Neither of this campaign's meshes is a literature-standard configuration.
#
# ★★ AND IT EXPLAINS THE FOUR DIVERGENCES. The campaign tried wall functions four times, all diverged,
# and filed the leg as BLOCKED -- concluding wall functions do not work here. They were tried at
# y+ ~ 8, where the log law is not approximate but INVALID:
#
#     y+= 8.0  log-law u+=10.272  true u+= 7.215  error +42.4%   -> infers u_tau 30% low, tau_w ~51% low
#     y+=30.0  log-law u+=13.496  true u+=12.787  error  +5.5%
#
# A wall model under-predicting tau_w by half lets the boundary layer separate spuriously, and in a
# cavitating flow that drives the void field hard. Divergence is the EXPECTED outcome. The correct
# conclusion was never "wall functions are blocked"; it was "a log law cannot live in the buffer
# layer". ⇒ THIS MODEL MUST BE RUN AT y+ 30-100. Running it on the present dn1 = 2e-4 mesh would
# reproduce those four failures exactly, and that is a property of the mesh, not of this code.
#
# WHAT IS AND IS NOT HERE. Incompressible equilibrium log law + the Larsson pressure-gradient term.
# The VAN DRIEST compressible transformation is deliberately NOT included: it needs
# `vd = sqrt(T_wall/T_loc)`, and in cmesh `T` is the PRESSURE PEDESTAL, railed to a constant 1.2 in
# every cell (measured: tools/measure_pedestal_rail.jl, 5 of 5 states railed by 3.4x-294x). Building a
# Van Driest factor out of a pressure ratio would be dimensionally meaningless and, worse, plausible-
# looking. It is blocked behind the two-population energy equation (S3g).
# ⇒ And that is the MATCHED choice anyway: Frikha's solver, which sheds at our exact alpha and Re, is
# INCOMPRESSIBLE with classical incompressible boundary conditions. An incompressible wall function is
# what the replication argument actually calls for.
# ============================================================================================

const WF_KAPPA = 0.41f0
const WF_B     = 5.2f0

"""
    wf_utau(u_t, d1, nu, dpdx, rho, larsson) -> u_tau

Equilibrium log-law friction velocity from the first-cell tangential speed `u_t` at wall distance
`d1`, by a Spalding-seeded power law followed by ONE log-law Picard update — the same two-step scheme
the sibling D3Q39 solver uses, which converges in one step in the equilibrium limit.

`larsson > 0` adds the Larsson 2010 pressure-gradient correction to the log-law residual,

    u+ = ln(y+)/kappa + B + (y / (rho u_tau^2)) dp/dx_stream

Schur-bounded to |du+| <= 0.5 u+ (Hickel-Touber-Bodart-Larsson 2012 SS3.2) so a stagnation point
cannot drive it away. Equilibrium wall models are accurate where dp/dx ~ 0; over a cavity CLOSURE,
which is exactly where the re-entrant jet is generated, the pressure gradient is the dominant term.

Below y+ = 11.81 the viscous sublayer is returned instead of the log law -- not as a fallback but
because the log law is simply not valid there.
"""
@inline function wf_utau(u_t::Float32, d1::Float32, nu::Float32,
                         dpdx::Float32, rho::Float32, larsson::Float32)
    (u_t <= 1.0f-8 || nu <= 0.0f0 || d1 <= 0.0f0) && return 0.0f0
    # Spalding-like power-law seed
    ut = u_t*(nu/(d1*u_t + 1.0f-10))^(1.0f0/7.0f0)*(2.0f0*8.3f0)^(-1.0f0/7.0f0)
    ut = ut < 1.0f-8 ? 1.0f-8 : ut
    yp = ut*d1/nu
    if yp > 11.81f0
        # ⚠ FIVE Picard sweeps, not one. The sibling D3Q39 model does ONE, and transliterating that
        # here left u_tau 14-20% LOW across the whole valid range y+ 30-300 (measured: -19.5% at
        # y+ 30, -14.4% at y+ 300) -- the Spalding power-law seed is simply not close enough for a
        # single update to land. Five sweeps converge to <0.1%. This costs nothing (it is scalar
        # arithmetic on data already in registers) and a 15% wall-shear error would have been
        # indistinguishable from the 10% buffer-layer error the wall function exists to remove.
        for _ in 1:5
            yp = ut*d1/nu
            yp <= 11.81f0 && break
            up = log(yp)/WF_KAPPA + WF_B
            if larsson > 0.0f0
                pg = larsson*d1*dpdx/(max(rho, 1.0f-6)*max(ut*ut, 1.0f-12))
                cap = 0.5f0*up
                pg = pg < -cap ? -cap : (pg > cap ? cap : pg)
                up += pg
            end
            up <= 0.1f0 && break
            ut = u_t/up
            ut = ut < 1.0f-8 ? 1.0f-8 : ut
        end
    else
        # viscous sublayer: u+ = y+ exactly, i.e. tau = mu u/d, the resolved-gradient answer
        ut = sqrt(nu*u_t/d1)
    end
    return ut
end

"""
    wf_yplus(u_tau, d1, nu) -> y+

Reported per run so the mesh's legality for a wall function is VISIBLE rather than assumed. A log law
needs y+ >~ 30; y+ ~ 8 is the buffer layer and is where the campaign's four wall-function attempts
diverged.
"""
@inline wf_yplus(u_tau::Float32, d1::Float32, nu::Float32) = u_tau*d1/max(nu, 1.0f-30)

# ============================================================================================
# ALL-y+ WALL FUNCTION FOR THE PRODUCTION SPECIES PATH  (26 Sep 2026)
#
# Selected by `s.wallfun == 2` (0 = wall-resolved SST, the default and bitwise the September path; the
# legacy log law above, `s.wallfun == 1`, stays refused on the species path). Why it exists and what it
# answers: memory/project_docs/2026-09-26_PAPERS_REVIEW_AND_WAY_FORWARD.md (D1) and docs/2026-09-26_WALL_FUNCTION_SPECIES_PATH.md.
# Every published RANS success on the Caltech NACA 66(MOD) case used a log-layer wall treatment, and the
# 26 Sep WP3L/WP3H reading found the wall-resolved SST wall cell letting k run away above first-cell y+ ~7.
#
# LAW. Reichardt's profile, valid from the wall through the buffer layer into the log layer, so there is no
# switch between laws (the legacy function switches at y+ 11.81, and was run -- four times -- at y+ ~8):
#
#     u+(y+) = ln(1 + kappa y+)/kappa + C [1 - exp(-y+/chi) - (y+/chi) exp(-y+/3)],  kappa 0.41, C 7.8, chi 11
#
# u+ -> y+ at the wall (the resolved-gradient answer) and u+ -> ln(y+)/kappa + 5.6 in the log layer.
# The friction velocity is found from the first-cell TANGENTIAL speed U at the cell-centre distance y by
# solving  y+ u+(y+) = U y / nu  for y+ (Newton; the left side is increasing and convex, so Newton converges
# from any start after at most one overshoot), then u_tau = y+ nu / y.
#
# TURBULENCE (Menter's automatic wall treatment, the SST's own): the first cell's omega is set to
#     omega = sqrt(omega_vis^2 + omega_log^2),  omega_vis = 6 nu/(beta1 y^2),  omega_log = u_tau/(sqrt(beta*) kappa y)
# -- the sublayer value the resolved treatment already produces at low y+ (measured 26 Sep: 0.96-1.04 of it at
# WP3L), and the log-layer value asymptotically. P/D approaches 1 for the pure logarithmic
# asymptote; the finite-y+ blend with S = (u_tau^2/nu) du+/dy+ does not enforce local equilibrium.
# The transported k balance also includes advection and diffusion. The k wall flux is zero.
# ============================================================================================

const WF_R_KAPPA = 0.41
const WF_R_C     = 7.8
const WF_R_CHI   = 11.0
const WF_BETA1   = 0.075
const WF_SQRT_BETASTAR = 0.3

"""
    wf_reichardt(yp) -> u+

Reichardt's all-y+ velocity profile, in the precision of `yp`.
"""
@inline function wf_reichardt(yp::T) where {T<:AbstractFloat}
    k = T(WF_R_KAPPA); C = T(WF_R_C); chi = T(WF_R_CHI)
    log1p(k*yp)/k + C*(one(T) - exp(-yp/chi) - (yp/chi)*exp(-yp/T(3)))
end

"""
    wf_reichardt_d(yp) -> du+/dy+
"""
@inline function wf_reichardt_d(yp::T) where {T<:AbstractFloat}
    k = T(WF_R_KAPPA); C = T(WF_R_C); chi = T(WF_R_CHI)
    e3 = exp(-yp/T(3))
    one(T)/(one(T) + k*yp) + C*(exp(-yp/chi)/chi - e3/chi + yp*e3/(T(3)*chi))
end

"""
    wf_reichardt_yplus(rey) -> y+

Solves `y+ u+(y+) = rey` (rey = U y / nu >= 0) for y+ >= 0 by Newton. The function is increasing and convex
in y+, so Newton converges from any positive start after at most one overshoot; twelve iterations from a
sublayer/log-law seed reach Float32 and Float64 round-off over rey 0 to 1e7 (tested in
tools/test_cmesh_cav_species.jl). Returns 0 for rey <= 0 or a nonfinite input.
"""
@inline function wf_reichardt_yplus(rey::T) where {T<:AbstractFloat}
    (isfinite(rey) && rey > zero(T)) || return zero(T)
    # seed: sublayer y+ = sqrt(rey) below rey ~ 130, a log-law estimate above
    yp = rey <= T(130) ? sqrt(rey) : rey/(log(rey)/T(WF_R_KAPPA) + T(5.6) - log(log(rey)/T(WF_R_KAPPA) + T(5.6))/T(WF_R_KAPPA))
    for _ in 1:12
        f = wf_reichardt(yp); df = wf_reichardt_d(yp)
        gp = f + yp*df
        gp > zero(T) || break
        nyp = yp - (yp*f - rey)/gp
        yp = nyp > zero(T) ? nyp : T(0.5)*yp
    end
    yp
end

"""
    wf_species_state(U, y, nu) -> (u_tau, y+, dU/dy)

The species wall function's state at a wall cell: tangential speed `U` at cell-centre distance `y`,
kinematic viscosity `nu`. `dU/dy` is the wall-law strain `(u_tau^2/nu) du+/dy+`, which replaces the
ghost-based Green-Gauss strain in the first cell's SST coefficients.
"""
@inline function wf_species_state(U::T, y::T, nu::T) where {T<:AbstractFloat}
    (U > zero(T) && y > zero(T) && nu > zero(T)) || return (zero(T), zero(T), zero(T))
    yp = wf_reichardt_yplus(U*y/nu)
    ut = yp*nu/y
    (ut, yp, ut*ut/nu*wf_reichardt_d(yp))
end

"""
    wf_species_omega(u_tau, y, nu) -> omega

Menter's automatic-wall-treatment omega for the first cell: sqrt(omega_vis^2 + omega_log^2).
"""
@inline function wf_species_omega(ut::T, y::T, nu::T) where {T<:AbstractFloat}
    wv = T(6)*nu/(T(WF_BETA1)*y*y)
    wl = ut/(T(WF_SQRT_BETASTAR)*T(WF_R_KAPPA)*y)
    sqrt(wv*wv + wl*wl)
end

"""
    wf_species_clock_factor(yp) -> factor

Linearised stiffness of the wall shear `rho u_tau(U)^2` in units of the laminar conductance `rho nu / y`:
`d(tau_w)/dU = rho 2 u_tau / (u+ + y+ du+/dy+)`, i.e. `2 y+ / (u+ + y+ du+/dy+)`. It is 1 in the sublayer
and ~3.7 / ~6.6 at y+ 30 / 60; the viscous clock uses max(4/3, factor) so it never charges less than the
no-slip Stokes wall it replaces.
"""
@inline function wf_species_clock_factor(yp::T) where {T<:AbstractFloat}
    yp > T(1.0e-6) || return one(T)
    T(2)*yp/(wf_reichardt(yp) + yp*wf_reichardt_d(yp))
end
