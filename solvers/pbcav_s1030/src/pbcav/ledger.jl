# =====================================================================================================================
# ledger.jl -- the conservation ledgers of plan section 8, measured every committed step with Float64 two-pass sums.
# Stage P0b (psi = 0, single phase): the mixture mass (inventory, inward boundary flux, the MMS mass source, the error),
# the closure defect Theta (max and volume-weighted), the continuity-error term's integral, the non-positive D count, and
# the rejections by reason. The vapour rows stay zero until stage P5a.
#
# THE MASS ERROR is measured, not assumed: e_n = M_n - M_{n-1} - dt (inflow_n + source_n), M = sum rho V, inflow =
# -rho_l sum_b F_b (inward positive), each a deterministic two-pass sum. Flux-form transport makes e_n round-off; the
# ledger reports max_n |e_n|/M_n (`step_err_max`) and the cumulative error.
# =====================================================================================================================

"""
    PBLedger

Running Float64 sums, one field per ledger row of plan section 8 (vapour rows zero before stage P5a).
"""
mutable struct PBLedger
    nsteps::Int
    mass0::Float64
    mass_inventory::Float64; mass_inflow::Float64; mass_source::Float64; mass_error::Float64; step_err_max::Float64
    vapour_inventory::Float64; vapour_inflow::Float64; vapour_generated::Float64; vapour_condensed::Float64
    vapour_error::Float64
    theta_max::Float64; theta_volume::Float64
    continuity_term::Float64
    nonpositive_D::Int
    rejections::Dict{Symbol,Int}
    last_rejection::Any          # the record of the last rejected attempt (pimple.jl, rejection_record; review_b1, R7), or nothing
    # the SST floors (W1, 21 Oct 2026; zero on the laminar path): committed steps' floored cells (cell-solves) and the added
    # rho k V, rho omega V (assemble_komega.jl, komega_solve!)
    turb_floor_k_cells::Int; turb_floor_w_cells::Int
    turb_floor_k_added::Float64; turb_floor_w_added::Float64
    # W1b (review N3): the committed steps' k-omega line solves continued past their first budget (assemble_komega.jl)
    turb_retries::Int
end
PBLedger() = PBLedger(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, Dict{Symbol,Int}(), nothing,
                      0, 0, 0.0, 0.0, 0)

@kernel function _k_mul3!(out, a, b)
    i = @index(Global)
    @inbounds out[i] = a[i]*b[i]
end
@kernel function _k_mms_mass!(out, vol, xc, yc, mms, t::Float64)
    c = @index(Global)
    @inbounds out[c] = vol[c]*mms_source(mms, EQ_MASS, xc[c], yc[c], t)
end

"The mass inventory sum rho V of time level L (two-pass)."
function mass_inventory(m::PBMesh, st::PBState, L::Integer, scratch; backend=CPU())
    _launch!(_k_mul3!, backend, m.nc, scratch, lev(st.rho, L), m.vol)
    pb_sum(scratch; backend)
end

"Start the ledger on the state's level 2 (the committed state)."
function ledger_init!(led::PBLedger, m::PBMesh, st::PBState, w, cfg)
    led.mass0 = mass_inventory(m, st, 2, w.scr; backend=cfg.backend); led.mass_inventory = led.mass0
    led
end

"""
    ledger_update!(led, mesh, st, w, cfg)

After a committed step (level 2 is the new state, level 3 the previous one): the mass rows, Theta, the continuity term
and the non-positive D count of the step (w.cont_step, w.nonpos_step), and the step's relative mass error.
"""
function ledger_update!(led::PBLedger, m::PBMesh, st::PBState, w, cfg)
    be = cfg.backend
    Mn = mass_inventory(m, st, 2, w.scr; backend=be)
    infl = -cfg.rho_l*pb_sum(lev(st.Fb, 2); backend=be)*st.dt_prev
    _launch!(_k_mms_mass!, be, m.nc, w.scr, m.vol, m.xc, m.yc, cfg.mms, st.t)
    src = cfg.rho_l*pb_sum(w.scr; backend=be)*st.dt_prev
    e = Mn - led.mass_inventory - infl - src
    led.nsteps += 1
    led.mass_inflow += infl; led.mass_source += src
    led.mass_error = Mn - led.mass0 - led.mass_inflow - led.mass_source
    led.step_err_max = max(led.step_err_max, abs(e)/abs(Mn))
    led.mass_inventory = Mn
    led.theta_max = pb_maxabs(w.theta; backend=be)
    _launch!(_k_mul3!, be, m.nc, w.scr, w.theta, m.vol)
    led.theta_volume = pb_sum(w.scr; backend=be)/pb_sum(m.vol; backend=be)
    led.continuity_term += w.cont_step
    led.nonpositive_D += w.nonpos_step
    if hasproperty(w, :turb) && w.turb !== nothing          # W1: the committed step's SST floors
        T = w.turb
        led.turb_floor_k_cells += T.nfloor_k; led.turb_floor_w_cells += T.nfloor_w
        led.turb_floor_k_added += T.added_k; led.turb_floor_w_added += T.added_w
        led.turb_retries += T.retries
    end
    led
end
