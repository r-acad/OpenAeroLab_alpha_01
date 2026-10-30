# =====================================================================================================================
# turbulence.jl -- SST-2003m point functions, the wall omega options, the trip and Reboud's correction (plan 3.5;
# stage P2, pulled forward as work item W1 of plan Amendment 12, section 32; 21 Oct 2026).
#
# WHAT IS BUILT (W1): the SST-2003m point functions copied from version 1 (src/cmesh/cmesh_species_sst.jl, read-only):
#   * `sst_terms`        the cell's terms: F1, F2, the blended constants, nu_t with the SST limiter (a1 k / max(a1 w, S F2)),
#                        the k production with its limiter min(nu_t S^2, 10 beta* k w), the omega production
#                        gamma min(S^2, 10 beta* w D / a1) and the cross diffusion 2 (1 - F1) sigma_w2 / w grad k . grad w;
#   * `sst_coefficients` version 1's `_sst_coefficients` (nu_t, sigma_k, sigma_w, the net sources, the clock rate) formed from
#                        `sst_terms` in version 1's operation order;
#   * `sst_wall_omega_v1` option W-a, version 1's `_sst_wall_omega`, 60 nu / (beta1 d1^2) (PBConfig.wall_omega = :v1);
#   * `sst_freestream`   the inflow k and omega from the turbulence intensity and the eddy-viscosity ratio (version 1's
#                        `set_sst_ka!`: k = 1.5 (Tu U)^2, omega = max(k / (ratio nu), 1e-6)).
# Every function is generic in the floating type with each constant written T(c): called with Float32 arguments it is
# version 1's function bit for bit (tools/test_pbcav.jl, testset "W1", evaluates version 1's own definitions, extracted
# from its source text, against these; every Float32(c) equals the c f0 literal version 1 writes); called with Float64
# arguments (the solver) it is the same formula in double precision.
# SST-2003m (version 1's header; https://tmbwg.github.io/turbmodels/sst.html): Menter, Kuntz and Langtry 2003 with the
# strain-rate production and without the isotropic rho k stress; the CD floor 1e-10 (divided by rho: the kinematic form);
# arg1 and arg2 capped at 4 before tanh (tanh(4^4), tanh(4^2) are 1 in Float32), as version 1 does.
# STILL STUBS (loud): option W-b (Menter's automatic wall treatment, stage P2's wall-omega gate), the trip, Reboud V3 and V2
# (later stages: plan 3.5, 32).
# =====================================================================================================================

"SST-2003m constants (Menter, Kuntz and Langtry 2003), as version 1 writes them: beta*, a1, sigma_k1/2, sigma_w1/2,
beta1/2, gamma1/2 (gamma1 = 5/9, gamma2 = 0.44)."
const SST_BETA_STAR = 0.09
const SST_A1 = 0.31
const SST_BETA1 = 0.075

"""
    sst_terms(k, w, d, nu, rho, dudx, dudy, dvdx, dvdy, gkw, fp=1) -> NamedTuple

The SST-2003m terms of one cell (kinematic: per unit density) at k, omega (w), the wall distance d, the laminar kinematic
viscosity nu, the density rho (it enters the CD floor only), the velocity gradient and gkw = grad k . grad omega:
`nut` (a1 k / max(a1 w, S F2)), `sigk`, `sigw` (blended), `P` (min(fp nut S^2, 10 beta* k w)), `Pw` (gamma min(S^2,
(10 beta*/a1) w D)), `beta` (blended), `cross` (2 (1 - F1) sigma_w2 gkw / w), `F1`, `F2`, `S2` (2 S_ij S_ij). Version 1's
operation order (src/cmesh/cmesh_species_sst.jl, `_sst_coefficients`), so the shared terms are its bits.
"""
@inline function sst_terms(k::T, w::T, d::T, nu::T, rho::T, dudx::T, dudy::T, dvdx::T, dvdy::T, gkw::T,
                           fp::T=one(T)) where {T<:AbstractFloat}
    Sxy = T(0.5)*(dudy + dvdx)
    S2 = T(2)*(dudx*dudx + dvdy*dvdy + T(2)*Sxy*Sxy); S = sqrt(S2)
    CD = max(T(2)*T(0.856)/w*gkw, T(1e-10)/rho); sqk = sqrt(k)
    arg1 = min(min(max(sqk/(T(0.09)*w*d), T(500)*nu/(d*d*w)), T(4)*T(0.856)*k/(CD*d*d)), T(4))
    arg2 = min(max(T(2)*sqk/(T(0.09)*w*d), T(500)*nu/(d*d*w)), T(4))
    F1 = tanh(arg1^4); F2 = tanh(arg2^2)
    D = max(T(0.31)*w, S*F2); nut = T(0.31)*k/D
    P = min(fp*(nut*S2), T(10)*T(0.09)*k*w)
    sigk = F1*T(0.85) + (one(T) - F1); sigw = F1*T(0.5) + (one(T) - F1)*T(0.856)
    beta = F1*T(0.075) + (one(T) - F1)*T(0.0828); gam = F1*(T(5)/T(9)) + (one(T) - F1)*T(0.44)
    Pw = gam*min(S2, (T(10)*T(0.09)/T(0.31))*w*D); cross = T(2)*(one(T) - F1)*T(0.856)/w*gkw
    (nut=nut, sigk=sigk, sigw=sigw, P=P, Pw=Pw, beta=beta, cross=cross, F1=F1, F2=F2, S2=S2)
end

"""
    sst_coefficients(k, w, d, nu, rho, dudx, dudy, dvdx, dvdy, gkw, fp=1) -> (nut, sigk, sigw, sk, sw, rate)

Version 1's `_sst_coefficients` (src/cmesh/cmesh_species_sst.jl): `sst_terms` and the net sources sk = P - beta* w k,
sw = Pw - beta w^2 + cross and version 1's clock rate, in its operation order (bitwise with Float32 arguments).
"""
@inline function sst_coefficients(k::T, w::T, d::T, nu::T, rho::T, dudx::T, dudy::T, dvdx::T, dvdy::T, gkw::T,
                                  fp::T=one(T)) where {T<:AbstractFloat}
    t = sst_terms(k, w, d, nu, rho, dudx, dudy, dvdx, dvdy, gkw, fp)
    sk = t.P - T(0.09)*w*k; sw = t.Pw - t.beta*w*w + t.cross
    rate = max(t.P/k + T(0.09)*w, (t.Pw + abs(t.cross))/w + t.beta*w)
    (t.nut, t.sigk, t.sigw, sk, sw, rate)
end
sst_coefficients(args...) = throw(ArgumentError("sst_coefficients(k, w, d, nu, rho, dudx, dudy, dvdx, dvdy, gkw[, fp]): " *
                                                "ten (or eleven) arguments of one floating type (got $(length(args)))"))

"Wall omega, option W-a (PBConfig.wall_omega = :v1): version 1's `_sst_wall_omega`, 60 nu/(beta1 d1^2), d1 the distance
from the wall face to the first cell's centre (bitwise with Float32 arguments)."
@inline sst_wall_omega_v1(nu::T, d1::T) where {T<:AbstractFloat} = T(60)*nu/(T(0.075)*d1*d1)
sst_wall_omega_v1(nu, d1) = sst_wall_omega_v1(promote(float(nu), float(d1))...)

"The inflow k of version 1's `set_sst_ka!` (src/cmesh/cmesh_solver_ka.jl), in Float64: 1.5 (tu U)^2."
sst_freestream_k(U::Real, tu::Real) = 1.5*(Float64(tu)*Float64(U))^2
"The inflow omega of version 1's `set_sst_ka!`, in Float64: max(k/(mut_ratio nu), 1e-6) (nu kinematic). Version 1 forms it
from its Float32-rounded k and stores Float32 of the result."
sst_freestream_omega(k::Real, mut_ratio::Real, nu::Real) = max(Float64(k)/(Float64(mut_ratio)*Float64(nu)), 1e-6)

"""
    sst_freestream(U, tu, mut_ratio, nu) -> (k, w)

The inflow k and omega (sst_freestream_k, sst_freestream_omega) in Float64, with the ranges version 1's driver admits
(tu in (0, 0.2), mut_ratio in (0, 1e4]).
"""
function sst_freestream(U::Real, tu::Real, mut_ratio::Real, nu::Real)
    (isfinite(tu) && 0 < tu < 0.2) || throw(ArgumentError("the turbulence intensity must lie in (0, 0.2) (got $tu)"))
    (isfinite(mut_ratio) && 0 < mut_ratio <= 1e4) || throw(ArgumentError("the eddy-viscosity ratio must lie in (0, 1e4] (got $mut_ratio)"))
    (isfinite(nu) && nu > 0) || throw(ArgumentError("sst_freestream needs nu > 0 (got $nu)"))
    k = sst_freestream_k(U, tu)
    (k, sst_freestream_omega(k, mut_ratio, nu))
end

"Wall omega, option W-b: Menter's automatic blend of 6 nu/(beta1 y^2) and u_tau/(sqrt(beta*) kappa y). Stage P2."
sst_wall_omega_auto(nu, y, u_tau) = _nyi(:sst_wall_omega_auto, "stage P2")
"The trip's k source (version 1's _sst_trip_k). Stage P2."
sst_trip_k(args...) = _nyi(:sst_trip_k, "stage P2")
"Reboud's eddy-viscosity factor of the void fraction alpha_v, exponent n (form V3, the default). Stage P2."
reboud_void(alpha_v, ratio, n) = _nyi(:reboud_void, "stage P2")
"Reboud's factor of the composition (form V2, PB_REBOUD=v2). Stage P2."
reboud_ratio(alpha, ratio, n) = _nyi(:reboud_ratio, "stage P2")
