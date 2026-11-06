# Command line, case dispatch and the time loop.
#   julia --project=src/octree_q19 src/octree_q19/main.jl --case <dir> [--build-only] [--resume]
# The case directory holds config.yaml and the STL; outputs go to <dir>/RESULTS as for the existing solver.

"Advance the state by one coarse step: the recursive step over all levels; forces in coarse-lattice units when collected."
function advance!(st::State; collect::Bool=false)
    step_recursive!(st, 1; collect=collect)
    st.step += 1
    collect && reduce_forces!(st)
    st
end

"""
Uniform mass control (the config's `octree_q19.mass_control`, off by default; for closed and periodic cases): the
stored populations and densities of every slot that carries a state, in both buffers, are scaled so that the
mass over the fluid leaves returns to `mass0`. The transfers between levels are cubic in space but not
flux-conserving (READINGS_G2.md, G2.5): a flow crossing the interfaces leaves a one-time gain of order 1e-4; an
inlet/outlet case loses it through the outlet and needs no control.
"""
function mass_control!(st::State{T}) where T
    m = total_mass(st)
    (isfinite(m) && m > 0) || return st
    factor = T(st.mass0 / m)
    for (l, Lv) in enumerate(st.mesh.levels)
        S = st.levels[l]
        @inbounds for o in 1:Lv.nocts, k in 1:8
            kd = Lv.kind[k, o]
            (kd == K_FLUID || kd == K_GHOST || kd == K_COVERED) || continue
            for i in 1:Q
                w = T(WT[i])
                S.f[k, i, o] = (S.f[k, i, o] + w) * factor - w; S.g[k, i, o] = (S.g[k, i, o] + w) * factor - w
            end
            S.rho[k, o] *= factor; S.rhob[k, o] *= factor
        end
    end
    st
end

"Newtons per unit of coarse-lattice momentum per coarse step: rho_0 dx_1^3 (dx_1 / dt_1) / dt_1."
function force_factor(st::State)
    sc = st.scaling; L = nlevels(st.mesh)
    dx1 = sc.dx_finest * 2.0^(L - 1); dt1 = sc.dt_finest * 2.0^(L - 1)
    st.cfg.flow.density * dx1^3 * sc.velocity_scale / dt1
end
"The last collected force in newtons; symmetry outputs are the mirrored full-model load."
function forces_newton(st::State)
    F = st.forces .* force_factor(st)
    st.cfg.mesh.faces[3] == :symmetry ? [2F[1], 0.0, 2F[3]] : F
end

"""
The mesh of a case: `mesh.oct` beside the config when its hashes match the case, else the sibling mesh
directory's (the app's `native/CASES/mesh` next to `run`), else a fresh build (written beside the config).
"""
function case_mesh(cfg::CaseConfig)
    for candidate in (joinpath(cfg.dir, "mesh.oct"), joinpath(dirname(cfg.dir), "mesh", "mesh.oct"))
        isfile(candidate) || continue
        m = try read_mesh(candidate) catch e; println("[Run] $(candidate): $(sprint(showerror, e)); rebuilding"); nothing end
        m === nothing && continue
        current_symmetry = !(:symmetry in cfg.mesh.faces) || get(m.counts,"symmetry_boundary_refinement",0)>=1
        if !current_symmetry
            println("[Run] mesh $(candidate) predates refinement at symmetry faces; rebuilding")
            continue
        end
        if m.config_sha == cfg.config_sha && m.stl_sha == cfg.stl_sha
            println("[Run] mesh $(candidate) matches the case (config and STL hashes)")
            return m
        end
        println("[Run] mesh $(candidate) belongs to another configuration; rebuilding")
    end
    build_only_case(cfg)
end

"""
Run a case: the time loop over coarse steps with the inlet ramp, forces every `forces_every` steps, histories,
field planes, surface and volume outputs, periodic checkpoints, and a final summary. `resume` restarts from the
latest checkpoint of RESULTS/checkpoints.
"""
function run_case(cfg::CaseConfig; resume::Bool=false)
    t_start = time()
    POSITIVITY_LIMITED_CELLS[] = 0
    out = joinpath(cfg.dir, "RESULTS"); mkpath(out)
    println("[Run] $(SOLVER_NAME): $(cfg.name), $(cfg.outputs.steps) coarse steps, precision $(cfg.precision), threads $(Threads.nthreads())")
    mesh = case_mesh(cfg)
    sc = lattice_scaling(cfg)
    L = nlevels(mesh)
    W = sum(p["fluid"] * 2^(p["level"] - 1) for p in mesh.counts["per_level"])
    println(@sprintf("[Run] %d levels, %d fluid cells, work %.4g cell updates per coarse step; dt_finest %.4g s, u_lattice %.3f, tau %s, Re %.4g",
            L, mesh.counts["fluid_total"], W, sc.dt_finest, sc.u_lattice, string(round.(sc.tau; digits=6)), sc.reynolds))
    flush(stdout)
    st = State(mesh, cfg, cfg.precision)
    at_rest = cfg.outputs.ramp_steps > 0
    initialise_uniform!(st, 1.0, at_rest ? 0.0 : sc.u_lattice, 0.0, 0.0)
    h = open_histories(out; append=resume)
    if resume
        s = resume_checkpoint!(st, out, h)
        s === nothing ? println("[Run] no checkpoint to resume; starting from step 0") : println("[Run] resumed at step $s")
        s === nothing || println("[Run] repaired $(repair_state_positivity!(st)) pre-existing non-positive checkpoint tuples")
    end
    sampler = field_sampler(st)
    steps = cfg.outputs.steps; o = cfg.outputs
    tri_factor = force_factor(st)
    F = zeros(3); M = zeros(3); coeffs = zeros(12)
    t_last = time(); step_last = st.step; last_ckpt = time()
    write_provenance(out, cfg, mesh.counts["reports"] isa AbstractDict ? get(mesh.counts, "mesh_sha", "") : "")
    if o.startup_slices && !resume && st.step==0
        println("[InitialFields] Writing step 0 initialized flow planes; these are initialization data, not converged results.")
        export_planes(mesh,cfg,sc,0,joinpath(out,"SLICES");with_fields=true,sampler=sampler)
        flush(stdout)
    end
    while st.step < steps
        ramp_inlet!(st)
        step = st.step + 1
        collect = step==1 || step % o.forces_every == 0 || step == steps
        advance!(st; collect=collect)
        rmin, rmax, umax = state_extrema(st)
        (isfinite(rmin) && isfinite(rmax) && isfinite(umax) && rmin > 0.2 && umax < 0.5) ||
            error("the state is leaving the stable range at step $step (rho_min $rmin, rho_max $rmax, |u|max $umax); stopping")
        if collect
            F .= forces_newton(st); tri_F = st.tri_force .* tri_factor; M .= moments_newton(st, tri_F)
            coeffs .= coefficients(st, F, M)
            write_forces!(h, st, F, M, coeffs)
        end
        cfg.mass_control && step % o.diagnostics_every == 0 && mass_control!(st)
        if step==1 || step % o.diagnostics_every == 0 || step == steps
            now = time(); dt = now - t_last; n = step - step_last
            mlups = n * W / max(dt, 1e-9) / 1e6; ips = n / max(dt, 1e-9)
            write_convergence!(h, st, (coeffs[1], coeffs[2], coeffs[3]), mlups, ips)
            println(@sprintf("[Step] DONE step %d of %d: %.2f MLUPS, %.3f steps/s, rho %.4f..%.4f, |u|max %.4f, Cd %.4f Cl %.4f Cs %.4f, elapsed %.0f s, positivity_limited %d",
                    step, steps, mlups, ips, rmin, rmax, umax, coeffs[1], coeffs[2], coeffs[3], now - h.t_wall0, POSITIVITY_LIMITED_CELLS[]))
            flush(stdout)
            t_last = now; step_last = step
        end
        if step % o.slice_every == 0 || step == steps
            export_planes(mesh, cfg, sc, step, joinpath(out, "SLICES"); with_fields=true, sampler=sampler)
            collect && write_surface!(st, out, step, st.tri_force .* tri_factor)
        end
        if (o.volume_vtk && step % o.output_every == 0) || (o.volume_vtk && step == steps)
            write_volume!(st, out, step)
        end
        if o.checkpoint_minutes > 0 && (time() - last_ckpt) / 60 >= o.checkpoint_minutes
            write_checkpoint!(st, out, h; keep=o.checkpoint_keep); last_ckpt = time()
            println("[Checkpoint] step $step written"); flush(stdout)
        end
    end
    o.checkpoint_minutes > 0 && write_checkpoint!(st, out, h; keep=o.checkpoint_keep)
    close_histories(h)
    summary = Dict{String,Any}("case" => cfg.name, "solver" => SOLVER_NAME, "status" => "completed_steps", "steps" => st.step,
        "convergence_assessed"=>false,"converged"=>false,"force_split_available"=>false,
        "force_note"=>"Full-model totals; symmetry mirrored. Pressure/viscous force decomposition is unavailable; legacy _p/_v columns must not be interpreted as a physical split.",
        "wall_seconds" => time() - t_start, "levels" => L, "fluid_cells" => mesh.counts["fluid_total"], "work_per_step" => W,
        "Cd" => coeffs[1], "Cl" => coeffs[2], "Cs" => coeffs[3], "Cd_mean" => h.n_avg > 0 ? h.sums[1] / h.n_avg : 0.0,
        "Cl_mean" => h.n_avg > 0 ? h.sums[2] / h.n_avg : 0.0, "written_utc" => Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS") * "Z")
    open(joinpath(out, "octree_run_summary.toml"), "w") do io; TOML.print(io, summary) end
    println(@sprintf("[Run] completed %d steps in %.0f s", st.step, time() - t_start)); flush(stdout)
    st
end

"Inlet ramp of the existing solver's default profile: a smooth half-cosine from 0 to the target over ramp_steps."
function ramp_inlet!(st::State{T}) where T
    n = st.cfg.outputs.ramp_steps
    st.u_inlet = n <= 0 || st.step >= n ? st.u_target : st.u_target * T(0.5 * (1 - cos(pi * st.step / n)))
    st
end

function main(args::Vector{String})
    case = ""; build_only = nothing; resume = false
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--case"
            i += 1; i <= length(args) || error("--case needs a directory"); case = args[i]
        elseif a == "--build-only"
            build_only = true
        elseif a == "--resume"
            resume = true
        elseif a == "--help" || a == "-h"
            println("usage: main.jl --case <dir> [--build-only] [--resume]"); return 0
        else
            error("unknown argument $a")
        end
        i += 1
    end
    isempty(case) && error("--case <dir> is required")
    isdir(case) || error("no such case directory: $case")
    cfg = load_case(case; build_only=build_only)
    if cfg.build_only
        build_only_case(cfg)
        return 0
    end
    run_case(cfg; resume=resume)
    0
end
