# Wall functions (DESIGN.md section 7.1): the laws of the wall, their inversion for u_tau, and the conversions
# between a wall cell's lattice units and wall units. Every quantity carries its unit in the name; the unit tests
# of gate G0.7 check each conversion against hand values. The population reconstruction that applies the model
# to a wall cell lives in the step (phase 4).

const KAPPA = 0.41
const LOG_B = 5.2

"Reichardt's law u+(y+), valid from the viscous sublayer to the log layer (Reichardt 1951)."
@inline function uplus_reichardt(yp::T) where {T<:AbstractFloat}
    T(1 / KAPPA) * log1p(T(KAPPA) * yp) + T(7.8) * (one(T) - exp(-yp / T(11)) - (yp / T(11)) * exp(-yp / T(3)))
end
"du+/dy+ of Reichardt's law."
@inline function duplus_reichardt(yp::T) where {T<:AbstractFloat}
    one(T) / (one(T) + T(KAPPA) * yp) + T(7.8) * (exp(-yp / T(11)) / T(11) - exp(-yp / T(3)) / T(11) + (yp / T(33)) * exp(-yp / T(3)))
end

"Spalding's law in its explicit form y+(u+) (Spalding 1961)."
@inline function yplus_spalding(up::T) where {T<:AbstractFloat}
    ku = T(KAPPA) * up
    up + T(exp(-KAPPA * LOG_B)) * (exp(ku) - one(T) - ku - ku * ku / T(2) - ku * ku * ku / T(6))
end
"u+(y+) of Spalding's law by Newton iteration on y+(u+)."
function uplus_spalding(yp::T) where {T<:AbstractFloat}
    up = yp < T(11) ? yp : T(1 / KAPPA) * log(max(yp, one(T))) + T(LOG_B)
    for _ in 1:30
        f = yplus_spalding(up) - yp
        ku = T(KAPPA) * up
        df = one(T) + T(exp(-KAPPA * LOG_B)) * T(KAPPA) * (exp(ku) - one(T) - ku - ku * ku / T(2))
        step = f / df
        up -= step
        abs(step) < T(1e-7) * max(one(T), abs(up)) && break
    end
    up
end

"Werner-Wengle: the power-law profile u+ = y+ (y+ < 11.81), else 8.3 y+^(1/7)."
@inline uplus_werner_wengle(yp::T) where {T<:AbstractFloat} = yp < T(11.81) ? yp : T(8.3) * yp^T(1 / 7)

"u+ of the chosen law."
@inline function uplus(law::Symbol, yp::T) where {T<:AbstractFloat}
    law == :reichardt ? uplus_reichardt(yp) : law == :spalding ? uplus_spalding(yp) : uplus_werner_wengle(yp)
end

"""
Friction velocity from the tangential speed `u` (lattice units) at wall distance `y` (lattice units) and the
kinematic viscosity `nu` (lattice units), by Newton iteration on u = u_tau u+(y u_tau / nu), warm-started from
`guess` (0 = the viscous estimate). Returns u_tau in lattice units (0 for u <= 0).
"""
function u_tau_from_law(law::Symbol, u::T, y::T, nu::T; guess::T=zero(T)) where {T<:AbstractFloat}
    u <= zero(T) && return zero(T)
    ut = guess > zero(T) ? guess : sqrt(u * nu / y)          # the viscous-sublayer estimate
    for _ in 1:40
        yp = y * ut / nu
        f = ut * uplus(law, yp) - u
        # d/du_tau [u_tau u+(y u_tau/nu)] = u+ + y+ du+/dy+
        dup = law == :reichardt ? duplus_reichardt(yp) :
              law == :werner_wengle ? (yp < T(11.81) ? one(T) : T(8.3 / 7) * yp^T(-6 / 7)) :
              (uplus_spalding(yp * T(1.001)) - uplus_spalding(yp)) / (yp * T(0.001) + T(1e-12))
        df = uplus(law, yp) + yp * dup
        step = f / df
        ut -= step
        ut = max(ut, T(1e-12))
        abs(step) < T(1e-8) * ut && break
    end
    ut
end

"Wall shear stress (lattice units of mass per cell per time^2) from u_tau and the density."
@inline tau_wall(rho::T, u_tau::T) where T = rho * u_tau * u_tau

"""
Wall-model velocity gradient at distance `y`: du_t/dn from the law's derivative, u_tau^2 / (nu (1 + ...)) in the
sublayer and u_tau / (kappa y) in the log layer, through du+/dy+ of the law.
"""
@inline function wall_gradient(law::Symbol, u_tau::T, y::T, nu::T) where {T<:AbstractFloat}
    yp = y * u_tau / nu
    dup = law == :reichardt ? duplus_reichardt(yp) :
          law == :werner_wengle ? (yp < T(11.81) ? one(T) : T(8.3 / 7) * yp^T(-6 / 7)) :
          (uplus_spalding(yp * T(1.001)) - uplus_spalding(yp)) / (yp * T(0.001) + T(1e-12))
    u_tau * u_tau / nu * dup
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
