# =====================================================================================================================
# assemble_phase.jl -- the phase transport of plan 4.4: flux-form transport of rho - q and q with the final mass flux, an
# implicit upwind M-matrix predictor, Zalesak's limited correction rerun with the final flux, Patankar's source split.
# Interfaces only in this commit (stage P5a; the psi = 0 limit of P0b transports nothing).
# =====================================================================================================================

"The implicit upwind phase predictor (an M-matrix: column sums V/dt > 0). Stage P5a."
assemble_phase_predictor!(sys, mesh, state, config) = _nyi(:assemble_phase_predictor!, "stage P5a")
"Zalesak's flux-corrected second-order step, bounded by the predictor's local extrema. Stage P5a."
zalesak_limit!(flux, mesh, state, config) = _nyi(:zalesak_limit!, "stage P5a")
