# =====================================================================================================================
# phase_laws.jl -- the phase laws of plan 3.2 (stage P1; ψ = 0 limits in P0b). Interfaces only in this commit.
#
#   liquid   Tait: rho_l = rho_l0 [(p + B)/(p0 + B)]^(1/n), B = 3.0665e8 Pa, n = 7.15; PB_LIQUID=incompressible gives psi_l = 0
#   vapour   isothermal: rho_v = rho_v,sat max(p, 0.01 p_v)/p_v; floor hits counted; PB_VAPOUR=incompressible
#   gas      a fixed gas mass fraction carried by the liquid (zero until a measured value exists: decision D4)
#   v1 fluid version 1's _rhomix_vl (src/cmesh/cmesh_eos_mix.jl:11-20), copied with a parity test, bridge arms only
# Every law is a scalar point function (Float64, GPU-callable, no allocation) of the pressure on the dynamic scale.
# =====================================================================================================================

"""
    PhaseParams

The two phases' constants in solver units (chord, U, rho_l,sat): liquid reference density `rho_l0` at `p0`, Tait `B` and
exponent `n_tait`; vapour saturation density `rho_v_sat` at `p_v`; the vapour floor fraction `pv_floor` (0.01);
`liquid_compressible`, `vapour_compressible` (false: the incompressible limits of design A, stage P0).
"""
Base.@kwdef struct PhaseParams
    rho_l0::Float64 = 1.0
    p0::Float64 = 0.0
    B::Float64 = 0.0
    n_tait::Float64 = 7.15
    rho_v_sat::Float64 = 1/43218
    p_v::Float64 = 0.0
    pv_floor::Float64 = 0.01
    liquid_compressible::Bool = false
    vapour_compressible::Bool = false
end

"Liquid density and its pressure derivative (rho_l, d rho_l / dp) at p. Stage P1."
liquid_density(p, par::PhaseParams) = _nyi(:liquid_density, "stage P1")
"Vapour density and its pressure derivative (rho_v, d rho_v / dp) at p, with the floor-hit flag. Stage P1."
vapour_density(p, par::PhaseParams) = _nyi(:vapour_density, "stage P1")
"Frozen (Wood) mixture compressibility alpha_l psi_l/rho_l + alpha_v psi_v/rho_v at (p, alpha_v). Stage P1."
wood_compressibility(p, alpha_v, par::PhaseParams) = _nyi(:wood_compressibility, "stage P1")
"The closure Theta = (rho - q)/rho_l(p) + q/rho_v(p) [+ gas] - 1 of plan 3.1. Stage P5a."
closure_defect(rho, q, p, par::PhaseParams) = _nyi(:closure_defect, "stage P5a")
"Version 1's mixture law _rhomix_vl, copied with a bitwise parity test (the C1x / W1 bridge only). Stage P4."
rhomix_v1(args...) = _nyi(:rhomix_v1, "stage P4")
