# =====================================================================================================================
# source.jl -- mass transfer behind one interface (plan 3.4; stage P5a). Interfaces only in this commit.
#
# Schnerr-Sauer by default (OpenFOAM's form, constants and regularisation, as on cav-models 6ab74d56; its _ss_terms is
# copied with a bitwise parity test), Merkle and Zwart-Gerber-Belamri behind the same coefficient interface. Each model
# returns the rate Gamma and its pressure derivative dGamma/dp, the latter regularised at 0.01 p_v, entering the pressure
# equation as the non-negative diagonal -(1/rho_v - 1/rho_l) dGamma/dp (plan 4.5).
# =====================================================================================================================

"A mass-transfer model: `source_terms(model, p, alpha_v, rho, par)` returns (Gamma, dGamma_dp)."
abstract type MassTransferModel end
"Schnerr-Sauer (default): nucleus density `n0` (in c^-3) and nucleus diameter `d_nuc` (in c)."
Base.@kwdef struct SchnerrSauer <: MassTransferModel
    n0::Float64 = 5.66e10
    d_nuc::Float64 = 1.31e-5
    c_cond::Float64 = 1.0
    c_vap::Float64 = 1.0
end
"Merkle: version 1's rates (production, destruction)."
Base.@kwdef struct Merkle <: MassTransferModel
    c_prod::Float64 = 300.0
    c_dest::Float64 = 5.0
end
"Zwart-Gerber-Belamri."
Base.@kwdef struct ZwartGerberBelamri <: MassTransferModel
    r_b::Float64 = 1e-6
    alpha_nuc::Float64 = 5e-4
    f_vap::Float64 = 50.0
    f_cond::Float64 = 0.01
end

"(Gamma, dGamma_dp) of `model` at pressure p, void fraction alpha_v, mixture density rho. Stage P5a."
source_terms(model::MassTransferModel, p, alpha_v, rho, par) = _nyi(:source_terms, "stage P5a")
