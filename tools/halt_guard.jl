# ============================================================================================
# halt_guard.jl -- refuse to start a solve while ./SOLVER_HALT exists.
#
# ⛔ WHY THIS IS ON THE TOOL AND NOT ONLY ON preflight.sh. Measured 2026-08-08: of the shell scripts
# that launch a solve, **11 do not call tools/preflight.sh at all** (battery2/3/4/5, battery11,
# battery18_p3, battery20_arecon, gate_battery, measure_global_dt, termsweep_v2, termsweep_v3). A halt
# enforced only in preflight would therefore have been bypassed by two thirds of the launch paths --
# and by the most common path of all, which is a human running the tool directly:
#
#     julia --project=src -O3 -t auto tools/cmesh_cav_movie.jl 1.25 12.0 640000 240
#
# That is the same failure shape the audit was called to fix: a control that covers the route people
# document rather than the routes they use. Putting the check in the ENTRY POINT covers every route.
#
# It is also placed at the very TOP of each entry point, before the includes, so it costs nothing and
# fires before any compilation.
#
# TO LIFT: delete ./SOLVER_HALT. Read that file first -- it says what has to be true first.
# ============================================================================================
let halt = joinpath(normpath(joinpath(@__DIR__, "..")), "SOLVER_HALT")
    if isfile(halt)
        printstyled("\n⛔ SOLVER HALTED -- refusing to start a solve.\n\n"; color=:red, bold=true)
        for ln in eachline(halt); println("   ", ln); end
        println()
        error("SOLVER_HALT is present; this tool will not run. Delete it deliberately, not to unblock a run.")
    end
end
