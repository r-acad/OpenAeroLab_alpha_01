# ============================================================================================
# RAYLEIGH-PLESSET BUBBLE DYNAMICS + SCHNERR-SAUER NUCLEI  (2026-08-01)
#
# The physical feature this campaign has been missing, and the reason it matters is NOT the one the
# implementation plan gave. That plan argued "a first-order relaxation cannot oscillate, so a slaved
# vapour field cannot be cut" -- and that argument is REFUTED (Frikha 2008, on disk: a BAROTROPIC
# model, tau = 0 exactly, sheds at our exact Re and alpha; see
# memory/project_docs/2026-08-01_FRIKHA_REFUTES_THE_LAG_PREMISE.md). Inertia is NOT necessary for shedding.
#
# It is implemented here anyway, and the honest justification is:
#   (a) it is REAL PHYSICS that every SOTA cavitation code carries (Kubota 1992, Schnerr-Sauer 2001,
#       Singhal 2002, and every CFX/Fluent/OpenFOAM cavitation model in production use), and we do not
#       have it. "Not necessary for shedding" is not the same as "not part of the physics";
#   (b) it brings SURFACE TENSION and NON-CONDENSABLE GAS CONTENT with it as first-class terms, which
#       are what actually make a bubble an OSCILLATOR (see the Minnaert note below) rather than a
#       runaway;
#   (c) it introduces a NUCLEI FIELD, i.e. an inception threshold and spatial selectivity, which the
#       pressure-driven Merkle source has never had.
#
# ------------------------------------------------------------------------------------------------
# THE MODEL
#
# SCHNERR-SAUER geometry. A fixed number density n0 of nuclei per unit LIQUID volume, each of radius
# R, gives a void fraction
#       V = (4/3) pi R^3 n0                  (vapour volume per unit liquid volume)
#       alpha = V / (1 + V)                  =>   R = ( (3/(4 pi n0)) * alpha/(1-alpha) )^(1/3)
# and the exact differential relation used for the source term:
#       dalpha/dR = 4 pi R^2 n0 / (1+V)^2 = 4 pi R^2 n0 (1-alpha)^2
#
# RAYLEIGH-PLESSET, in full, with every term SOTA codes carry:
#       R Rddot + (3/2) Rdot^2 = (p_B - p)/rho_l  -  4 nu_l Rdot / R  -  2 S / (rho_l R)
#       p_B = p_v + p_g0 (R0/R)^(3 kappa)
# i.e. inertia (the Rddot and the (3/2)Rdot^2 that Schnerr-Sauer THROWS AWAY), viscous damping,
# surface tension, and a polytropic non-condensable gas content.
#
# ★ WHY THE GAS TERM IS NOT OPTIONAL. Linearise about an equilibrium radius: a pure-vapour bubble has
# NO restoring force (p_v is constant, so the RHS has no R-dependence except the destabilising
# surface-tension term) -- it is a runaway, not an oscillator. The restoring force comes from the gas
# compressing as R falls, giving the Minnaert frequency
#       omega_0^2 = ( 3 kappa p_g0 - 2S/R0 ) / (rho_l R0^2)
# So "give the vapour inertia" without gas content buys an inertial DELAY but still no natural
# frequency. Codes that report R-P as an oscillator all carry p_g0. `pg0 = 0` is permitted here and
# gives the pure-vapour inertial model, but it is not the oscillator and must not be described as one.
#
# ------------------------------------------------------------------------------------------------
# HOW IT ENTERS THE EXISTING SOLVER -- and why this is NOT a restructure.
#
# The alpha source is  dalpha/dt|_src = (dalpha/dR) * Rdot = 4 pi R^2 n0 (1-alpha)^2 Rdot,  which
# factors ONTO THE EXISTING POINT-IMPLICIT STRUCTURE exactly. `_k_tem_sweep!` assembles
#       num = a0 + dtl*(E - adv),   den = 1 + dtl*(E + D)
# from a source  E*(1-alpha) - D*alpha. Choosing
#       E_rp = 4 pi R^2 n0 (1-alpha)   * max( Rdot, 0)
#       D_rp = 4 pi R^2 n0 (1-alpha)^2 * max(-Rdot, 0) / max(alpha, eps)
# reproduces  E_rp*(1-a) - D_rp*a  ==  4 pi R^2 n0 (1-a)^2 Rdot  for either sign, so growth lands on
# the (1-alpha) branch and collapse on the alpha branch exactly as Merkle's E and D do. The entire
# num/den machinery, the BDF2 dual-time terms, the advection, the clamp and the residual are reused
# UNCHANGED. Only the two coefficients change.
#
# ⚠ ONE REAL CONSEQUENCE, stated because it will show up in the convergence behaviour. Under Merkle
# the dominant diagonal term is the pressure feedback jacp = -dE/dalpha ~ 4e3 (that is what makes the
# void iteration contract at all -- see memory/project_docs/2026-07-31_THE_VOID_ITERATION_AMPLIFIES.md). Under R-P the
# pressure does NOT act on the alpha equation directly; it acts on the Rdot equation. So the alpha
# equation's pressure-feedback Jacobian is genuinely ZERO here, and the stiffness has MOVED to the Rdot
# ODE (which is why that one is given its own point-implicit treatment below). Expect the void
# residual's convergence character to change. That is physics, not a bug -- but MEASURE it, do not
# assume it, and do not compare `ares` across the two models as if it meant the same thing.
# ============================================================================================

const RP_FOURPI = 12.566370614359172f0     # 4*pi
const RP_EPS    = 1.0f-30
# ★ PHYSICAL upper bound on the bubble radius, in chords. Not a numerical guard: a single bubble
# larger than a quarter chord is not a bubble, it is the cavity. Added after an audit found R
# floored below but UNBOUNDED above, which lets a small n_active send R -> inf through every 1/R
# term (surface tension, the gas law, the cloud ratio A/R) SILENTLY rather than by erroring.
const RP_RMAX   = 0.25f0

"""
    rp_R_of_alpha(a, n0, R0)

Schnerr-Sauer bubble radius from the void fraction, floored at the nucleus radius `R0`.

`alpha/(1-alpha)` is the vapour volume per unit LIQUID volume, so this inverts
`alpha = V/(1+V)`, `V = (4/3) pi R^3 n0`. The floor matters: at `alpha = 0` the formula gives `R = 0`
and every `1/R` term in Rayleigh-Plesset would divide by zero. Physically the floor IS the model --
below the nucleus radius there is no bubble, there is a nucleus.
"""
@inline function rp_R_of_alpha(a::Float32, n0::Float32, R0::Float32)
    ac = a < 0.0f0 ? 0.0f0 : (a > 0.999999f0 ? 0.999999f0 : a)
    V  = ac/(1.0f0 - ac)                                   # vapour volume per unit liquid volume
    # ⚠ GPU SAFETY: return early at V == 0 rather than calling cbrt(0). The early return is also the
    # physically correct branch (no vapour => the nucleus radius), so this is not a numerical dodge.
    V <= 0.0f0 && return R0
    x = 3.0f0*V/(RP_FOURPI*(n0 < RP_EPS ? RP_EPS : n0))
    R = exp(log(x)*0.3333333f0)                            # cube root via exp/log: portable to every
                                                           # backend, and x > 0 is guaranteed above
    # ⚠ UPPER CAP, added after auditing today's epsilon floors. R was floored at R0 but UNBOUNDED
    # above, and `rp_poly_nactive` can legitimately return a very small n_active -- at which point
    # x = 3V/(4 pi n0) explodes and R with it. R then appears in 1/R terms (surface tension 2S/R, the
    # gas law (R0/R)^3k) and in the cloud ratio A/R, so an unbounded R silently rewrites the physics
    # rather than erroring. RP_RMAX = 0.25 chord is the same scale the cloud radius is clamped to: a
    # single bubble larger than a quarter chord is not a bubble, it is the cavity.
    return R < R0 ? R0 : (R > RP_RMAX ? RP_RMAX : R)
end

"""
    rp_alpha_of_R(R, n0)

The forward Schnerr-Sauer relation `alpha = V/(1+V)`, `V = (4/3) pi R^3 n0`. Exact inverse of
`rp_R_of_alpha` above the nucleus floor; provided for the unit tests and for diagnostics.
"""
@inline function rp_alpha_of_R(R::Float32, n0::Float32)
    V = (RP_FOURPI/3.0f0)*R*R*R*n0
    return V/(1.0f0 + V)
end

"""
    rp_dalpha_dR(R, a, n0)

`dalpha/dR = 4 pi R^2 n0 (1-alpha)^2`. Written in terms of `alpha` rather than `V` so it stays exact
against whatever `alpha` the transport equation actually carries, instead of against the `alpha` the
radius would imply.
"""
@inline function rp_dalpha_dR(R::Float32, a::Float32, n0::Float32)
    om = 1.0f0 - a
    return RP_FOURPI*R*R*n0*om*om
end

"""
    rp_ED(R, a, rdot, n0)  ->  (E_rp, D_rp)

Map the Rayleigh-Plesset radial velocity onto the growth/collapse coefficient pair the existing
point-implicit sweep consumes, such that `E*(1-a) - D*a` equals `(dalpha/dR)*rdot` for either sign.

Growth lands on the `(1-alpha)` branch (it saturates as alpha -> 1, i.e. no liquid left to vaporise)
and collapse on the `alpha` branch (it saturates as alpha -> 0, i.e. no vapour left to condense),
which is the same saturation structure Merkle has and the reason this substitution is safe.
"""
@inline function rp_ED(R::Float32, a::Float32, rdot::Float32, n0::Float32, R0::Float32)
    dadR = rp_dalpha_dR(R, a, n0)
    om   = 1.0f0 - a
    if rdot >= 0.0f0
        # E*(1-a) = dadR*rdot   =>   E = dadR*rdot/(1-a) = 4 pi R^2 n0 (1-a) rdot
        return (RP_FOURPI*R*R*n0*om*rdot, 0.0f0)
    else
        # ⛔⛔ THE BUG THAT KILLED THE FIRST ALL-ON RUN (amax went to 0.000 in the warm-up).
        # D*a = dadR*|rdot| means D = dadR*|rdot|/a, which DIVERGES as a -> 0. With a floor of 1e-6 it
        # reached D = 1.9e6, giving den = 1 + dtl*D = 11.7 -- every cell that momentarily saw p > p_v
        # had its vapour crushed 11.7x PER SWEEP and could never recover. The whole cavity was
        # annihilated during the warm-up.
        #
        # THE PHYSICS IT GOT WRONG: in a NUCLEI model alpha cannot reach zero. There is always the
        # resting nuclei population at R = R0, i.e. alpha_seed = alpha(R0) -- and condensation CANNOT
        # collapse a bubble smaller than the nucleus that seeded it. So:
        #   * D is exactly ZERO once R has returned to R0 (nothing left to condense), and
        #   * the divisor is floored at alpha_seed, not at an arbitrary 1e-6.
        # Merkle never hit this because its D = kdest*|dpv| carries no 1/alpha at all.
        R <= R0*1.000001f0 && return (0.0f0, 0.0f0)
        aseed = rp_alpha_of_R(R0, n0)
        return (0.0f0, dadR*(-rdot)/max(a, aseed))
    end
end

"""
    rp_rhs(R, rdot, p, p_v, rho_l, nu_l, S, pg0, R0, kappa)  ->  (A, B, Q)

Split the Rayleigh-Plesset acceleration into the pieces a point-implicit update needs:

    Rddot = A - B*rdot - Q
      A = [ (p_B - p)/rho_l - 2S/(rho_l R) ] / R     explicit forcing (pressure + surface tension)
      B = 4 nu_l / R^2                               linear viscous damping, >= 0, safe on the diagonal
      Q = (3/2) rdot^2 / R                           the quadratic inertia term, >= 0, ALWAYS SUBTRACTED
      p_B = p_v + pg0 (R0/R)^(3 kappa)

⛔ SIGN TRAP, caught by T4 of tools/test_rayleigh_plesset.jl and recorded so it is not reintroduced.
The first version of this function returned the quadratic term as a pseudo-damping coefficient
`C = (3/2)|rdot|/R` intended for the implicit diagonal, giving `Rddot = A - B*rdot - C*rdot`. That is
correct for GROWTH and WRONG FOR COLLAPSE: `-C*rdot = -(3/2)|rdot|*rdot/R` flips to `+(3/2)rdot^2/R`
when `rdot < 0`, i.e. it DECELERATES a collapsing bubble. The Rayleigh collapse came out 1.86x too
slow. In the true equation `R Rddot + (3/2) Rdot^2 = RHS` the quadratic sits on the LHS as a SQUARE,
so it is subtracted for either sign. It is returned here as a bare non-negative `Q` precisely so no
caller can re-introduce a sign by "linearising" it.

HOW A POINT-IMPLICIT CALLER SHOULD USE IT. `B` is genuinely dissipative and belongs on the diagonal.
`Q` is dissipative only while the bubble GROWS; during collapse it is a destabilising source (that is
the physics -- Rayleigh collapse accelerates without bound), so putting it on the diagonal there would
make the diagonal smaller than 1. Split it on the sign of `rdot`:
    Cpos = max( (3/2)*rdot/R , 0 )          # == Q/rdot when rdot > 0, else 0
    den  = 1 + dt*(B + Cpos)
    num  = rdot + dt*(A - (Q - Cpos*rdot))  # the bracket is 0 for growth, Q for collapse
which is fixed-point-preserving for either sign and never reduces the diagonal.

⚠ `A` is the only term carrying the pressure, which is what moves the stiffness out of the alpha
equation and into this one.
"""
@inline function rp_rhs(R::Float32, rdot::Float32, p::Float32, p_v::Float32,
                        rho_l::Float32, nu_l::Float32, S::Float32,
                        pg0::Float32, R0::Float32, kappa::Float32)
    Rs = R < RP_EPS ? RP_EPS : R
    # polytropic non-condensable gas content: p_g = pg0 (R0/R)^(3 kappa)
    pg = pg0 > 0.0f0 ? pg0*exp(-3.0f0*kappa*log(Rs/R0)) : 0.0f0
    pB = p_v + pg
    A  = ((pB - p)/rho_l - 2.0f0*S/(rho_l*Rs))/Rs
    B  = 4.0f0*nu_l/(Rs*Rs)
    Q  = 1.5f0*rdot*rdot/Rs                  # >= 0 ALWAYS; subtracted for either sign of rdot
    return (A, B, Q)
end

"""
    rp_rdot_quasistatic(R, p, p_v, rho_l, S, pg0, R0, kappa)

The INERTIA-FREE bubble-wall velocity — Schnerr–Sauer's standard production form:

    Rdot = sign(dp) * sqrt( 2|dp| / (3 rho_l) ),   dp = p_B - p,  p_B = p_v + pg0(R0/R)^(3k) - 2S/R

Obtained by dropping `R Rddot` from Rayleigh–Plesset and solving the remaining
`(3/2) Rdot^2 = dp/rho_l` for `Rdot`. Positive dp grows the bubble, negative collapses it.

★ WHY THIS IS THE CORRECT FORM HERE, not a fallback. At the benchmark point with 1 um nuclei the
Minnaert period is 1.02e-05 CTU against a 0.35 CTU cavity transit — the bubble responds 3.4e4 times
faster than the flow, so it is QUASI-STATIC and its inertia cannot influence anything on the flow's
timescale (memory/project_docs/2026-08-01_RP_IS_QUASI_STATIC_A_PRIORI.md). Integrating the full second-order system
at the flow's timestep is not merely wasteful, it is UNSTABLE: the restoring force is explicit, and an
explicit oscillator stepped at 1e4x its own period diverges. That is exactly how the first rp=1 smoke
test NaN'd.

Surface tension and the gas term are retained — they set the equilibrium radius and hence where the
driving changes sign, which matters even without inertia.
"""
@inline function rp_rdot_quasistatic(R::Float32, p::Float32, p_v::Float32, rho_l::Float32,
                                     S::Float32, pg0::Float32, R0::Float32, kappa::Float32)
    Rs = R < RP_EPS ? RP_EPS : R
    pg = pg0 > 0.0f0 ? pg0*exp(-3.0f0*kappa*log(Rs/R0)) : 0.0f0
    dp = (p_v + pg - 2.0f0*S/Rs) - p
    m  = sqrt(2.0f0*abs(dp)/(3.0f0*max(rho_l, RP_EPS)))
    return dp >= 0.0f0 ? m : -m
end

"""
    rp_poly_nactive(n0, dpv, S, Rmin, Rmax, expo, poly) -> n_eff

**POLYDISPERSE NUCLEI** (S3c). Schnerr–Sauer assumes every nucleus has the SAME radius `R0`. Real
nuclei follow a size spectrum, conventionally a power law `n(R) dR ~ R^-expo dR` on `[Rmin, Rmax]`
with `expo ≈ 3–4` (Billet; Brennen ch.1).

**Why it is not cosmetic: it makes inception PROGRESSIVE and SPATIALLY SELECTIVE.** A nucleus is
unstable only once the under-pressure exceeds its Laplace barrier, i.e. above the critical radius

    R_crit = 2S / (p_v - p)

so as the suction deepens, LARGER nuclei activate first and the active population grows continuously.
A monodisperse model instead switches its whole population on at one threshold. With a power law the
active fraction has a closed form:

    N(>R_crit)/N_total = (R_crit^(1-e) - Rmax^(1-e)) / (Rmin^(1-e) - Rmax^(1-e))

Returns `n0` unchanged when `poly <= 0` (bitwise legacy) or when the flow is above `p_v` (no
activation, so the population is whatever it already was — this function only gates INCEPTION, it does
not extinguish existing vapour).
"""
@inline function rp_poly_nactive(n0::Float32, dpv::Float32, S::Float32,
                                 Rmin::Float32, Rmax::Float32, expo::Float32, poly::Float32)
    poly <= 0.0f0 && return n0
    dpv <= 0.0f0  && return n0                      # not under-pressured: nothing to activate
    S    <= 0.0f0 && return n0                      # no surface tension => no Laplace barrier
    Rc = 2.0f0*S/dpv
    Rc <= Rmin && return n0                         # everything is already unstable
    Rc >= Rmax && return n0*1.0f-6                  # nothing is: a floor, not zero (keeps 1/n finite)
    e  = 1.0f0 - expo
    fr = (exp(e*log(Rc)) - exp(e*log(Rmax))) / (exp(e*log(Rmin)) - exp(e*log(Rmax)))
    # The 1e-6 floor keeps n_active > 0 so 1/n stays finite in rp_R_of_alpha. It is SAFE only
    # because R is now capped at RP_RMAX -- before that cap, a near-zero n_active sent R to
    # infinity. Documented rather than left as a bare magic number.
    fr = fr < 1.0f-6 ? 1.0f-6 : (fr > 1.0f0 ? 1.0f0 : fr)
    return n0*(1.0f0 + poly*(fr - 1.0f0))           # poly=1 => full effect, poly=0 => n0 exactly
end

"""
    rp_hk_limit(rate, dpv, p_v, rho_v, hk) -> limited_rate

**HERTZ–KNUDSEN INTERFACE KINETICS** (S3d). Merkle and Rayleigh–Plesset both assume the interface
sits AT saturation, i.e. that phase change is limited only by heat/mass transport in the bulk. Kinetic
theory says otherwise: the molecular flux across the interface is bounded by

    mdot_max = (2 sigma_a / (2 - sigma_a)) * sqrt(M / (2 pi R_g T)) * dp

with `sigma_a` the accommodation coefficient (0.01–1 for water; the literature spread is two orders,
which is itself why this term is usually omitted). The ceiling matters at LOW pressure, where the bulk
model would otherwise predict an unbounded evaporation rate.

Implemented as a smooth saturating limiter on the phase-change coefficient rather than a hard clip:

    rate' = rate / (1 + rate/rate_max)

so it is inert when `rate << rate_max` and asymptotes to `rate_max` above it — no kink for the
point-implicit iteration to straddle, which the campaign has been bitten by before (the `src_smooth`
hard-switch band). `hk <= 0` returns the rate unchanged, bitwise.
"""
@inline function rp_hk_limit(rate::Float32, a::Float32, R::Float32,
                             p_v::Float32, rho_v::Float32, hk::Float32)
    hk <= 0.0f0 && return rate
    # ⚠⚠ DIMENSIONAL FIX (2026-08-02). The first version of this function read
    #        rmax = hk*sqrt(p_v/rho_v)
    # and the comment claimed that was "in the same units as `rate` (1/time)". IT IS NOT.
    # sqrt(p/rho) is a VELOCITY [L/T]; `rate` is a reciprocal time [1/T], since
    #        E = 4 pi R^2 n0 (1-a) Rdot   ->   L^2 * L^-3 * L/T = 1/T.
    # Comparing them is meaningless, and it went unnoticed because in chord-normalised units both
    # land at O(1), so the limiter produced plausible-looking numbers. The per-term sweep measured
    # it cutting the cavity 8x (amax 0.997 -> 0.127) -- a large, confident, DIMENSIONALLY INVALID
    # effect. This is exactly the failure mode the campaign keeps hitting: a term that is active and
    # explicable-sounding but wrong, which is far more dangerous than a term that is inert.
    #
    # The kinetic ceiling is properly a MASS FLUX PER UNIT INTERFACE AREA. Converting it to a
    # volumetric rate requires the interfacial area density of the bubble population,
    #        a_iv = 3 alpha / R    [1/L]      (surface area of n bubbles per unit volume)
    # so that
    #        rate_max = hk * a_iv * v_thermal = [1/L]*[L/T] = [1/T]   <- commensurate with `rate`.
    # v_thermal = sqrt(p_v/rho_v) remains the vapour thermal speed to within an O(1) constant, which
    # is absorbed into the accommodation coefficient `hk` (itself uncertain across two orders).
    vth  = sqrt(max(p_v, RP_EPS)/max(rho_v, RP_EPS))          # [L/T]
    aiv  = 3.0f0*max(a, 0.0f0)/max(R, RP_EPS)                 # [1/L]
    rmax = hk*aiv*vth                                          # [1/T]
    return rate/(1.0f0 + rate/max(rmax, RP_EPS))
end

"""
    rp_wall_nuclei(n0, jrel, wallnuc, wdecay) -> n_eff

**HETEROGENEOUS WALL NUCLEATION** (S3e). Cavitation inception in a real flow is overwhelmingly
HETEROGENEOUS — it happens at wall crevices and surface sites, not in the bulk. Our nuclei field is
uniform, so inception has no reason to prefer the surface.

`jrel` is the wall-normal index normalised by the decay length; the enhancement is
`n0 * (1 + wallnuc * exp(-jrel/wdecay))`, i.e. a surface excess that decays into the bulk.
`wallnuc = 0` returns `n0` exactly.

⚠ Deliberately crude: a proper model keys on surface roughness and a site-density distribution, which
we have no data for on this foil. This gives the CORRECT QUALITATIVE STRUCTURE (inception biased to
the wall) with one knob, and its arbitrariness is why it must be swept rather than trusted at a value.
"""
@inline function rp_wall_nuclei(n0::Float32, jrel::Float32, wallnuc::Float32, wdecay::Float32)
    wallnuc <= 0.0f0 && return n0
    return n0*(1.0f0 + wallnuc*exp(-jrel/max(wdecay, 1.0f-6)))
end

"""
    rp_tke_source(a, rdot, R, n0, btke) -> k_production

**BUBBLE-INDUCED TURBULENCE MODULATION** (S3f). Reboud's `f(rho)` correction is a one-way, ad-hoc
reduction of the eddy viscosity by mixture density. It carries no mechanism: growing and collapsing
bubbles do WORK on the surrounding liquid and inject turbulent kinetic energy.

Volumetric production, following the bubble-induced-turbulence closures used in bubbly-flow RANS
(Troshko & Hassan class): the interfacial work rate per unit volume scales as the bubble surface area
density times the wall velocity cubed,

    P_b = btke * (3 alpha / R) * |Rdot|^3

`btke = 0` returns zero exactly, so the SST source is bitwise unchanged.

⚠ Returns the production term only; the caller must add it to the k equation. It is NOT wired into
the SST solve yet — doing so needs the k-source plumbing in cmesh_solver_ka.jl, which is a separate
edit. Exposed and unit-testable now so the physics is not lost.
"""
@inline function rp_tke_source(a::Float32, rdot::Float32, R::Float32, btke::Float32)
    btke <= 0.0f0 && return 0.0f0
    ar = abs(rdot)
    return btke*(3.0f0*a/max(R, RP_EPS))*ar*ar*ar
end

"""
    rp_cloud_beta(a, R, gradA, cloudint) -> beta

Brennen's **cloud interaction parameter** `beta = alpha (1-alpha) (A/R)^2`, with the cloud radius `A`
estimated LOCALLY as the void-fraction gradient length `A ~ alpha / |grad alpha|`.

★ WHY THIS IS THE ONE ABSENT PHYSICS ITEM WHOSE TIMESCALE IS NEAR THE FLOW'S. Our Rayleigh-Plesset is
strictly SINGLE-BUBBLE, and a single 1 um bubble rings at 3.44 MHz -- five orders above the shedding
frequency, hence quasi-static and irrelevant (memory/project_docs/2026-08-01_RP_IS_QUASI_STATIC_A_PRIORI.md). But a
CLOUD of bubbles oscillates far more slowly than its members, because each bubble must displace the
surrounding bubbly medium rather than pure liquid (d'Agostino & Brennen 1989):

    omega_cloud = omega_0 / sqrt(1 + beta)

With plausible cloud radii that lands at **0.026-0.26 CTU** against a 3.2 CTU shedding period --
closing 3-4 of the 5 orders. It does not reach the flow timescale, but it is the only item on the
missing-physics list that gets within two orders of it.

★★ AND IT DOES SOMETHING A UNIFORM LAG CANNOT. The 2026-08-01 lag ladder swept `Cprod` over 100x and
changed the phase-change timescale EVERYWHERE BY THE SAME FACTOR; it moved cavity magnitude and
produced no oscillation at any rung. This term is different in kind: `beta` scales with the local void
fraction and cluster size, so the DENSE CAVITY CORE responds more slowly than the DILUTE EDGES. That
is a spatially-varying lag, which the uniform ladder could not produce and therefore did not test.
Whether it matters is open — but "the uniform-lag ladder was flat" is not evidence against it.

`A` is clamped into [R, 0.25 chord]: below R the cloud is a single bubble (beta -> 0 as it should),
and above a quarter chord the gradient length is measuring the domain, not a cloud.
`cloudint` scales the whole effect; 0 disables it exactly (`beta = 0`, `1/sqrt(1+beta) = 1`).
"""
@inline function rp_cloud_beta(a::Float32, R::Float32, gradA::Float32, cloudint::Float32)
    cloudint <= 0.0f0 && return 0.0f0
    (a <= 1.0f-6 || a >= 0.999999f0) && return 0.0f0     # no cluster in either pure phase
    A = a/max(gradA, 1.0f-12)
    A = A < R ? R : (A > 0.25f0 ? 0.25f0 : A)
    r = A/max(R, RP_EPS)
    return cloudint*a*(1.0f0 - a)*r*r
end

"""
    rp_nsub(dtl, R0, rho_l, pg0, S, kappa, nmax) -> Int32

How many sub-steps the INERTIAL form needs to resolve its own oscillation at pseudo-step `dtl`,
capped at `nmax`. Returns 0 when the oscillator has no natural frequency (pg0 too small), i.e. when
there is nothing to resolve.

The caller uses this to degrade gracefully: sub-cycle when `nsub <= nmax`, and fall back to the
quasi-static limit when it does not — which is the physically correct choice, since being unable to
resolve an oscillation and it being irrelevant are the same statement here.
"""
@inline function rp_nsub(dtl::Float32, R0::Float32, rho_l::Float32, pg0::Float32,
                         S::Float32, kappa::Float32, nmax::Int32)
    w2 = rp_minnaert_omega2(R0, rho_l, pg0, S, kappa)
    w2 <= 0.0f0 && return Int32(0)
    T = 6.2831853f0/sqrt(w2)
    n = ceil(dtl/(0.1f0*T))            # >= 10 sub-steps per period
    n < 1.0f0 && return Int32(1)
    n > Float32(nmax) && return Int32(-1)   # -1 => cannot resolve; caller uses quasi-static
    return Int32(n)
end

"""
    rp_minnaert_omega2(R0, rho_l, pg0, S, kappa)

The linearised natural frequency squared of a gas-loaded bubble,
`omega_0^2 = (3 kappa p_g0 - 2S/R0) / (rho_l R0^2)`.

Exposed because it is the ONE quantity that says whether the model as configured is an oscillator at
all: `omega_0^2 <= 0` means the restoring force is negative and there is no natural frequency, only
runaway growth. Tools should print it, and the unit test asserts the integrator reproduces it.
"""
@inline function rp_minnaert_omega2(R0::Float32, rho_l::Float32, pg0::Float32,
                                    S::Float32, kappa::Float32)
    return (3.0f0*kappa*pg0 - 2.0f0*S/max(R0, RP_EPS))/(rho_l*max(R0*R0, RP_EPS))
end

"""
    rp_timescale_report(R0, rho_l, pg0, S, kappa, U) -> (T_minnaert_CTU, ratio_to_transit)

The Minnaert period expressed in CHORD TRANSIT UNITS, and its ratio to the ~0.35 CTU cavity transit.

★ WHY THIS IS PRINTED RATHER THAN LEFT TO BE REDISCOVERED. At the benchmark point with standard 1 um
Schnerr-Sauer nuclei the Minnaert period is 1.03e-05 CTU against a 3.2 CTU shedding period -- the
bubble rings at 3.44 MHz while the shedding it is supposed to drive happens at 11 Hz, a factor of
3.1e5. R-P is QUASI-STATIC here by five orders of magnitude: it cannot set the shedding timescale and
cannot resonate with it, which is precisely why Schnerr-Sauer's inertia-free truncation is the
standard production form.

So a run with rp=1 that behaves like the Merkle baseline is CONFIRMING this arithmetic, not failing.
Printing the ratio at build time means nobody has to rediscover that after spending a run on it.
Full derivation, including why COLLECTIVE (cloud) dynamics closes 3-4 of those 5 orders and single-
bubble dynamics does not: memory/project_docs/2026-08-01_RP_IS_QUASI_STATIC_A_PRIORI.md
"""
@inline function rp_timescale_report(R0::Float32, rho_l::Float32, pg0::Float32,
                                     S::Float32, kappa::Float32, U::Float64)
    w2 = rp_minnaert_omega2(R0, rho_l, pg0, S, kappa)
    w2 <= 0.0f0 && return (Inf, Inf)           # no natural frequency: runaway, not oscillator
    T_lat = 2pi/sqrt(Float64(w2))              # lattice time
    T_ctu = T_lat*U                            # CTU = 1/U on a unit-chord mesh
    (T_ctu, 0.35/T_ctu)                        # ratio to the ~0.35 CTU cavity transit
end
