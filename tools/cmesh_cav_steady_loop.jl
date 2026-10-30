# cmesh_cav_steady_loop.jl -- the steady pseudo-time loop of tools/cmesh_cav_movie.jl (speed-up plan W2, 30 Sep 2026;
# memory/project_docs/2026-09-30_SPEEDUP_PLAN.md). The driver includes this file ONLY when TOPO_SPECIES_STEADY=1, after the build and
# in place of the physical-time march, and exits after it. It iterates step_species_steady_ka!
# (src/cmesh/cmesh_species_steady.jl): local time steps, a relaxation to the steady state of the same discrete
# equations the physical-time step integrates, for STEADY WETTED targets only.
# NOTHING HERE IS PHYSICAL TIME. The history is indexed by iteration, the checkpoints carry
# mode="steady_pseudo_time" and zero physical time, and no movie frame is written.
#
# Knobs, all read here:
#   TOPO_STEADY_ITERS   maximum number of iterations (required)
#   TOPO_STEADY_CHECK   iterations between monitor rows (default 500)
#   TOPO_STEADY_CKPT    monitor rows between checkpoints (default 20; the initial and final states are always written)
#   TOPO_STEADY_MARGIN  safety factor on every local cap (default 0.8)
#   TOPO_STEADY_STOP_ROWS, TOPO_STEADY_STOP_TOL
#                       optional early stop, set together: stop when the relative spread (max - min)/|mean| of BOTH
#                       Cl and Cd over the last STOP_ROWS monitor rows is below STOP_TOL (default 0 = never)
#   TOPO_STEADY_LINE    1 = the wall-normal line relaxation (src/cmesh/cmesh_species_steady_line.jl); 2 = the
#                       two-direction approximate factorisation (cmesh_species_steady_adi.jl, local step TOPO_STEADY_ADI_CFL
#                       x the i-face limit, default 2); 3 = the j lines with the block operator of the first-order face
#                       Jacobians (cmesh_species_block.jl); default 0
#   TOPO_STEADY_END_FACTOR  the j-line relaxation's outlet-end step factor (first and last 3 columns), default 1 = off
#   TOPO_STEADY_WALL_S  optional wall-clock budget of the loop in seconds, checked at monitor rows: the loop stops
#                       and writes its final state instead of being killed by an owner's cap (default 0 = none)
# Outputs in OUTD: <tag>_steady.csv (monitor rows), <tag>_steady_initial.jls, <tag>_steady_<iteration>.jls and
# <tag>_steady_final.jls, plus three PNG renders of the final state in the frame directory.

const STEADY_ITERS = parse(Int, get(ENV, "TOPO_STEADY_ITERS", "0"))
const STEADY_CHECK = parse(Int, get(ENV, "TOPO_STEADY_CHECK", "500"))
const STEADY_CKPT = parse(Int, get(ENV, "TOPO_STEADY_CKPT", "20"))
const STEADY_MARGIN_V = parse(Float32, get(ENV, "TOPO_STEADY_MARGIN", "0.8"))
const STEADY_STOP_ROWS = parse(Int, get(ENV, "TOPO_STEADY_STOP_ROWS", "0"))
const STEADY_STOP_TOL = parse(Float64, get(ENV, "TOPO_STEADY_STOP_TOL", "0"))
const STEADY_WALL_S = parse(Float64, get(ENV, "TOPO_STEADY_WALL_S", "0"))
const STEADY_LINE_RAW = get(ENV, "TOPO_STEADY_LINE", "0")
STEADY_LINE_RAW in ("0", "1", "2", "3") || error("TOPO_STEADY_LINE must be 0, 1, 2 or 3")
const STEADY_ADI_CFL = parse(Float32, get(ENV, "TOPO_STEADY_ADI_CFL", "2"))
isfinite(STEADY_ADI_CFL) && 0 < STEADY_ADI_CFL <= 1000 || error("TOPO_STEADY_ADI_CFL must lie in (0, 1000]")
CMeshCavSpecies.ADI_CFL[] = STEADY_ADI_CFL
const STEADY_END_FACTOR = parse(Float32, get(ENV, "TOPO_STEADY_END_FACTOR", "1"))
isfinite(STEADY_END_FACTOR) && 0 < STEADY_END_FACTOR <= 1 || error("TOPO_STEADY_END_FACTOR must lie in (0, 1]")
CMeshCavSpecies.LINE_END_FACTOR[] = STEADY_END_FACTOR
isfinite(STEADY_WALL_S) && STEADY_WALL_S >= 0 || error("TOPO_STEADY_WALL_S must be finite and nonnegative")
STEADY_ITERS > 0 || error("TOPO_SPECIES_STEADY=1 requires TOPO_STEADY_ITERS > 0")
STEADY_CHECK > 0 && STEADY_CKPT > 0 || error("TOPO_STEADY_CHECK and TOPO_STEADY_CKPT must be positive")
isfinite(STEADY_MARGIN_V) && 0 < STEADY_MARGIN_V <= 1 || error("TOPO_STEADY_MARGIN must lie in (0, 1]")
STEADY_STOP_ROWS >= 0 && isfinite(STEADY_STOP_TOL) && STEADY_STOP_TOL >= 0 &&
    (STEADY_STOP_ROWS == 0) == (STEADY_STOP_TOL == 0) ||
    error("TOPO_STEADY_STOP_ROWS and TOPO_STEADY_STOP_TOL must both be zero or both positive")
CMeshCavSpecies.STEADY_MARGIN[] = STEADY_MARGIN_V
const steady_ws = build_species_steady(species; line=STEADY_LINE_RAW == "1", adi=STEADY_LINE_RAW == "2",
                                      block=STEADY_LINE_RAW == "3")
const steady_start = (steps=species.steps, elapsed=species.elapsed)

println("STEADY PSEUDO-TIME MODE (TOPO_SPECIES_STEADY=1): local time steps, a relaxation to the steady state. ",
        "This is NOT physical time: iterations are not time, and nothing below is a transient.")
@printf("  physical clock held at steps=%d elapsed=%.17g; cfl=%.4g margin=%.3g max_iterations=%d check_every=%d checkpoint_every=%d rows; early stop %s\n",
        steady_start.steps, steady_start.elapsed, CFLV, STEADY_MARGIN_V, STEADY_ITERS, STEADY_CHECK, STEADY_CKPT,
        STEADY_STOP_ROWS == 0 ? "off" : @sprintf("when Cl and Cd each vary by less than %.3g (relative) over %d rows", STEADY_STOP_TOL, STEADY_STOP_ROWS))
STEADY_WALL_S > 0 && @printf("  wall-clock budget of the loop: %.0f s (checked at monitor rows)\n", STEADY_WALL_S)
println("  iteration: ", STEADY_LINE_RAW == "1" ?
    "wall-normal line relaxation (local step from the i faces; implicit tridiagonal lines in j, through the wake cut)" :
    STEADY_LINE_RAW == "3" ?
    "wall-normal line relaxation with the block operator (the first-order j-face Jacobians, 4x4 per face; local step from the i faces)" :
    STEADY_LINE_RAW == "2" ?
    "approximate factorisation (i lines, then j lines through the wake cut; local step ADI_CFL=$(STEADY_ADI_CFL) x the i-face limit)" :
    "explicit local time stepping (every cell at its own caps)")
flush(stdout)

function steady_checkpoint(path)
    # A checkpoint publishes dtf as the physical clock's last applied step (the restart validator requires one uniform
    # value). The local steps are per-iteration scratch that every attempt recomputes before reading, so they are
    # replaced here, after the monitor row has used them.
    fill!(s.dtf, s.last_global_dt)
    metadata = (driver="cmesh_cav_movie.jl", mode="steady_pseudo_time", tag=tag, output_directory=OUTD,
        parent_checkpoint=RESTART_PATH, requested_ctu=0.0, scored_elapsed=0.0, scored_steps=0,
        total_rejected_attempts=species_retry_count[], pseudo_iterations=steady_ws.iterations,
        pseudo_attempts=steady_ws.attempts, pseudo_rejections=steady_ws.rejections, cfl=CFLV,
        margin=Float64(STEADY_MARGIN_V))
    FIXED_OUTER_PROVENANCE === nothing || (metadata = merge(metadata, (fixed_outer=FIXED_OUTER_PROVENANCE,)))
    MESH_TE_SETTINGS === nothing || (metadata = merge(metadata, (te_base=MESH_TE_SETTINGS,)))   # trailing-edge mesh, 2 Oct
    SST_OMEGA_INIT == "wall" && (metadata = merge(metadata, (sst_omega_start="wall",)))
    write_cmesh_species_restart(path, species; metadata)
end
# Wall Cp at the driver's ten stations, and the field's lowest Cp with its cell (i, j): the margin to the saturation
# pressure is what the steady mode's source refusal depends on (Cp_v = (p_v - p_ref)/q), so every row reports it.
function steady_pressure_rows()
    P = Array(species.pressure)
    cps = [(Float64(P[i+g, 1+g]) - p_ref)/Q_INF for i in cpx_stations]
    k = argmin([Float64(P[i+g, j+g]) for i in 1:NI, j in 1:NJ])
    (cps, (Float64(P[k[1]+g, k[2]+g]) - p_ref)/Q_INF, k[1], k[2])
end
const STEADY_CPV = (Float64(tem.p_v) - p_ref)/Q_INF

const steady_csv = joinpath(OUTD, tag * "_steady.csv")
open(steady_csv, "w") do io
    println(io, "iteration,wall_s,res_rho,res_mx,res_my,res_q,res_k,res_w,Cl,Cd,dt_min,dt_max,attempts,rejections,last_scale,cp_min,i_min,j_min,",
            join(["cp_x" * @sprintf("%.2f", x) for x in CPX_XS], ","))
end
steady_checkpoint(joinpath(OUTD, tag * "_steady_initial.jls"))
steady_rows = Tuple{Int,Float64,Float64}[]
steady_t0 = time()
steady_stop_reason = "iteration budget reached ($(STEADY_ITERS))"
println("  iteration   res_rho     res_mx      res_my      res_k       res_w       Cl          Cd          dt_min      dt_max     rej   wall_s")
for it in 1:STEADY_ITERS
    step_species_steady_ka!(species, steady_ws, Float32(CFLV))
    (it == 1 || it % STEADY_CHECK == 0 || it == STEADY_ITERS) || continue
    r = species_steady_residuals(species)
    Cl, Cd = movie_loads()
    isfinite(Cl) && isfinite(Cd) || error("nonfinite loads at steady iteration $it")
    cps, cpmin, imin, jmin = steady_pressure_rows()
    wall = time() - steady_t0
    open(steady_csv, "a") do io
        println(io, join((it, @sprintf("%.3f", wall), r.rho, r.mx, r.my, r.q, r.k, r.w, Float64(Cl), Float64(Cd),
                          r.dt_min, r.dt_max, steady_ws.attempts, steady_ws.rejections, steady_ws.last_scale, cpmin, imin, jmin), ","), ",",
                join([@sprintf("%.6f", v) for v in cps], ","))
    end
    push!(steady_rows, (it, Float64(Cl), Float64(Cd)))
    lr = steady_ws.last_rejection
    @printf("  %9d  %.3e  %.3e  %.3e  %.3e  %.3e  %+.7f  %.8f  %.3e  %.3e  %4d  %8.1f  Cp_min %+.3f at (%d,%d), Cp_v %+.2f  last rejection %s\n",
            it, r.rho, r.mx, r.my, r.k, r.w, Float64(Cl), Float64(Cd), r.dt_min, r.dt_max, steady_ws.rejections, wall,
            cpmin, imin, jmin, STEADY_CPV, lr === nothing ? "-" : string(lr.reason, "@stage", lr.stage))
    flush(stdout)
    length(steady_rows) % STEADY_CKPT == 0 && steady_checkpoint(joinpath(OUTD, tag * @sprintf("_steady_%08d.jls", it)))
    if STEADY_WALL_S > 0 && wall > STEADY_WALL_S
        global steady_stop_reason = @sprintf("wall-clock budget reached (%.0f s > %.0f s)", wall, STEADY_WALL_S)
        break
    end
    if STEADY_STOP_ROWS > 0 && length(steady_rows) >= STEADY_STOP_ROWS
        last = steady_rows[end-STEADY_STOP_ROWS+1:end]
        spread(v) = (maximum(v) - minimum(v))/abs(sum(v)/length(v))
        dl, dd = spread([x[2] for x in last]), spread([x[3] for x in last])
        if dl < STEADY_STOP_TOL && dd < STEADY_STOP_TOL
            global steady_stop_reason = @sprintf("loads stationary: relative spread over the last %d rows Cl %.3g, Cd %.3g < %.3g",
                                                 STEADY_STOP_ROWS, dl, dd, STEADY_STOP_TOL)
            break
        end
    end
end
species.steps == steady_start.steps && species.elapsed == steady_start.elapsed ||
    error("the steady mode advanced the physical clock -- refusing to write its final state")
steady_checkpoint(joinpath(OUTD, tag * "_steady_final.jls"))
let (Cl, Cd) = movie_loads()
    @printf("STEADY RECEIPT: iterations=%d attempts=%d rejections=%d wall_s=%.1f stop=\"%s\" final Cl=%.7f Cd=%.8f; physical clock unchanged (steps=%d elapsed=%.17g)\n",
            steady_ws.iterations, steady_ws.attempts, steady_ws.rejections, time() - steady_t0, steady_stop_reason,
            Float64(Cl), Float64(Cd), species.steps, species.elapsed)
end
channel_diagnostics(s, tem, met, U; label="final steady pseudo-time state", mixture_pressure=species.pressure)

# The final state, rendered once with the movie's own raster and colour conventions (no video: there is no time axis).
let fields = cmesh_species_fields(species), A = fields.beta, PF = fields.pressure, UX = Array(s.ux)
    fa = [Float64(A[i+g, j+g]) for i in 1:NI, j in 1:NJ]
    fu = [Float64(UX[i+g, j+g])/U for i in 1:NI, j in 1:NJ]
    fp = [(Float64(PF[i+g, j+g]) - p_ref)/q for i in 1:NI, j in 1:NJ]
    for (nm, cl, cmap, ttl) in FIELDS
        fld = nm == "alpha" ? fa : (nm == "ux" ? fu : fp)
        hm = heatmap(R.XR, R.YR, raster_gather(R, fld); aspect_ratio=1, c=cmap, clims=cl,
                     background_color=:black, framestyle=:none, axis=false, ticks=false,
                     colorbar=true, legend=false, size=(R.PX, R.PY), nan_color=:black,
                     xlims=(WIN[1], WIN[2]), ylims=(WIN[3], WIN[4]), title=ttl * "  (steady pseudo-time, final)",
                     titlefontsize=9, titlefontcolor=:white)
        savefig(hm, joinpath(FRD, "steady_final_" * nm * ".png"))
    end
end
println("STEADY MODE DONE: ", steady_csv); flush(stdout)
