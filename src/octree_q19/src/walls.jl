# Wall functions (DESIGN.md section 7.1): the laws of the wall, their inversion for u_tau, and the conversions
# between a wall cell's lattice units and wall units. Every quantity carries its unit in the name; the unit tests
# of gate G0.7 check each conversion against hand values. The population reconstruction that applies the model
# to a wall cell lives in the step (phase 4).

const KAPPA = 0.41
const LOG_B = 5.2

"Reichardt's law u+(y+), valid from the viscous sublayer to the log layer (Reichardt 1951)."
@inline function uplus_reichardt(yp::T) where {T<:AbstractFloat}
    # expm1 avoids losing significant digits in 1-exp(-y+/11) close to a wall.
    a = yp / T(11); b = yp / T(3)
    T(1 / KAPPA) * log1p(T(KAPPA) * yp) + T(7.8) * (-expm1(-a) - a * exp(-b))
end
"du+/dy+ of Reichardt's law."
@inline function duplus_reichardt(yp::T) where {T<:AbstractFloat}
    one(T) / (one(T) + T(KAPPA) * yp) + T(7.8) * (exp(-yp / T(11)) / T(11) - exp(-yp / T(3)) / T(11) + (yp / T(33)) * exp(-yp / T(3)))
end

"Exponent remainder after the cubic term, evaluated without cancellation near zero."
@inline function spalding_remainder3(x::T) where {T<:AbstractFloat}
    abs(x) < T(.1) ? x^4 * (T(1/24) + x * (T(1/120) + x * (T(1/720) + x * (T(1/5040) + x * (T(1/40320) + x * (T(1/362880) + x*T(1/3628800))))))) :
        expm1(x) - x - x*x/T(2) - x*x*x/T(6)
end
"Derivative of the cubic exponential remainder."
@inline function spalding_remainder2(x::T) where {T<:AbstractFloat}
    abs(x) < T(.1) ? x^3 * (T(1/6) + x * (T(1/24) + x * (T(1/120) + x * (T(1/720) + x * (T(1/5040) + x * (T(1/40320) + x*T(1/362880))))))) :
        expm1(x) - x - x*x/T(2)
end

"Spalding's law in its explicit form y+(u+) (Spalding 1961)."
@inline function yplus_spalding(up::T) where {T<:AbstractFloat}
    ku = T(KAPPA) * up
    up + T(exp(-KAPPA * LOG_B)) * spalding_remainder3(ku)
end
"u+(y+) of Spalding's law by Newton iteration on y+(u+)."
function uplus_spalding(yp::T) where {T<:AbstractFloat}
    up = yp < T(11) ? yp : T(1 / KAPPA) * log(max(yp, one(T))) + T(LOG_B)
    for _ in 1:30
        f = yplus_spalding(up) - yp
        ku = T(KAPPA) * up
        df = one(T) + T(exp(-KAPPA * LOG_B)) * T(KAPPA) * spalding_remainder2(ku)
        step = f / df
        up -= step
        abs(step) < T(1e-7) * max(one(T), abs(up)) && break
    end
    up
end
"du+/dy+ for Spalding's law, obtained by implicit differentiation."
@inline function duplus_spalding(yp::T) where {T<:AbstractFloat}
    ku = T(KAPPA) * uplus_spalding(yp)
    inv(one(T) + T(exp(-KAPPA * LOG_B)) * T(KAPPA) * spalding_remainder2(ku))
end

"Werner-Wengle: the power-law profile u+ = y+ (y+ < 11.81), else 8.3 y+^(1/7)."
@inline uplus_werner_wengle(yp::T) where {T<:AbstractFloat} = yp < T(11.81) ? yp : T(8.3) * yp^T(1 / 7)

"u+ of the chosen law."
@inline function uplus(law::Symbol, yp::T) where {T<:AbstractFloat}
    isfinite(yp) && yp >= zero(T) || throw(ArgumentError("y+ must be finite and nonnegative"))
    law == :reichardt ? uplus_reichardt(yp) : law == :spalding ? uplus_spalding(yp) :
    law == :werner_wengle ? uplus_werner_wengle(yp) : throw(ArgumentError("Unknown wall law: $law"))
end

"""
Friction velocity from the tangential speed `u` (lattice units) at wall distance `y` (lattice units) and the
kinematic viscosity `nu` (lattice units), by Newton iteration on u = u_tau u+(y u_tau / nu), warm-started from
`guess` (0 = the viscous estimate). Returns u_tau in lattice units (0 for u <= 0).
"""
function u_tau_from_law(law::Symbol, u::T, y::T, nu::T; guess::T=zero(T)) where {T<:AbstractFloat}
    law in (:reichardt, :spalding, :werner_wengle) || throw(ArgumentError("Unknown wall law: $law"))
    all(isfinite, (u, y, nu, guess)) || throw(ArgumentError("Wall-law inputs must be finite"))
    y > zero(T) && nu > zero(T) || throw(ArgumentError("Wall distance and kinematic viscosity must be positive"))
    u <= zero(T) && return zero(T)
    ut = guess > zero(T) ? guess : sqrt(u * nu / y)          # the viscous-sublayer estimate
    isfinite(ut) || error("Wall-law inversion cannot represent the initial friction velocity")
    ut = max(ut, floatmin(T))
    for _ in 1:40
        yp = y * ut / nu
        isfinite(yp) || error("Wall-law inversion cannot represent y+")
        f = ut * uplus(law, yp) - u
        # d/du_tau [u_tau u+(y u_tau/nu)] = u+ + y+ du+/dy+
        dup = law == :reichardt ? duplus_reichardt(yp) :
              law == :werner_wengle ? (yp < T(11.81) ? one(T) : T(8.3 / 7) * yp^T(-6 / 7)) :
              duplus_spalding(yp)
        df = uplus(law, yp) + yp * dup
        step = f / df
        isfinite(step) || error("Wall-law inversion failed to find a finite Newton step")
        ut -= step
        # The old fixed 1e-12 floor could exceed the true solution by orders
        # of magnitude for near-stagnant flow. Use the type's representable
        # scale and verify the reconstructed velocity after iteration.
        ut = max(ut, floatmin(T))
        abs(step) < max(T(1e-8), T(4) * eps(T)) * ut && break
    end
    modeled = ut * uplus(law, y * ut / nu)
    tolerance = max(T(1e-7), T(16) * eps(T))
    isfinite(modeled) && abs(modeled / u - one(T)) <= tolerance ||
        error("Wall-law inversion did not converge to the requested tangential speed")
    ut
end

"Wall shear stress (lattice units of mass per cell per time^2) from u_tau and the density."
@inline tau_wall(rho::T, u_tau::T) where T = rho * u_tau * u_tau

"""
Wall-model velocity gradient at distance `y`: du_t/dn from the law's derivative, u_tau^2 / (nu (1 + ...)) in the
sublayer and u_tau / (kappa y) in the log layer, through du+/dy+ of the law.
"""
@inline function wall_gradient(law::Symbol, u_tau::T, y::T, nu::T) where {T<:AbstractFloat}
    law in (:reichardt, :spalding, :werner_wengle) || throw(ArgumentError("Unknown wall law: $law"))
    all(isfinite, (u_tau, y, nu)) || throw(ArgumentError("Wall-law inputs must be finite"))
    u_tau >= zero(T) && y > zero(T) && nu > zero(T) || throw(ArgumentError("Friction velocity must be nonnegative; wall distance and kinematic viscosity must be positive"))
    yp = y * u_tau / nu
    isfinite(yp) || error("Wall-law gradient cannot represent y+")
    dup = law == :reichardt ? duplus_reichardt(yp) :
          law == :werner_wengle ? (yp < T(11.81) ? one(T) : T(8.3 / 7) * yp^T(-6 / 7)) :
          duplus_spalding(yp)
    gradient = u_tau * u_tau / nu * dup
    isfinite(gradient) || error("Wall-law gradient cannot be represented")
    gradient
end

# ---------------------------------------------------------------- units
"y+ of a wall distance given in metres, with u_tau and nu in SI."
@inline yplus_si(y_m::Real, u_tau_m_s::Real, nu_m2_s::Real) = y_m * u_tau_m_s / nu_m2_s
"Lattice velocity of level l to metres per second: the velocity scale is the same on every level."
@inline lattice_to_si_velocity(u_lat::Real, velocity_scale::Real) = u_lat * velocity_scale
"Lattice viscosity of level l (nu_l) to SI: nu_l dx_l^2 / dt_l."
@inline lattice_to_si_viscosity(nu_lat::Real, dx_l::Real, dt_l::Real) = nu_lat * dx_l^2 / dt_l
"Wall distance in cells of level l from metres."
@inline metres_to_cells(y_m::Real, dx_l::Real) = y_m / dx_l
