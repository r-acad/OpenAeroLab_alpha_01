# ==============================================================================
# Particles-on-Demand — production-case runner  [makes `lattice: pond` run end-to-end]
# ==============================================================================
# A self-contained entry that main.jl dispatches to when POND_ENABLED. It runs the
# VALIDATED single-level uniform-grid PonD conservative kinetic-FV driver
# (pond_fv3d_driver.jl) on a grid + immersed body declared by the case config, writing
# a density mid-plane slice + a drag/stability summary to the case RESULTS dir. It
# deliberately BYPASSES the LBM block/AMR machinery (setup_multilevel_domain, the
# stream-collide recursion) — the PonD FV evolves conserved moments, structurally
# distinct from stream-collide — so this path is low-risk and cannot perturb existing
# LBM cases. Cross-block/AMR execution on the production block domain is the next
# integration (the validated substrate is pond_fv3d_blocks.jl).
#
# Config (under advanced.numerics.pond): nx,ny,nz, dx, sphere_radius, sphere_cx/cy/cz,
# steps, ramp_steps, eta, wband. Mach comes from the case's physical Mach (REGIME_MA).
# ==============================================================================

using KernelAbstractions, Printf
import CUDA
try
    @eval using Plots
catch
end

function _pondcfg(keys...; default)
    safe_get(CFG, "advanced", "numerics", "pond", keys...; default=default)
end

function run_pond_fv_case!(runtime::CaseIDContract.RuntimeSelection,
                           claimed_output::CaseIDContract.ClaimedOutput;
                           root_dir::AbstractString=dirname(dirname(runtime.case.case_dir)))
    CaseIDContract.verify_runtime_inputs(root_dir, runtime)
    CaseIDContract.verify_claimed_output(runtime, claimed_output)
    resdir = claimed_output.target.path
    outcome_start_datetime = Dates.now()
    outcome_start_time = time()

    backend = CUDA.functional() ? CUDA.CUDABackend() : CPU()
    dev(x) = CUDA.functional() ? CUDA.CuArray(x) : copy(x)

    # lattice tables (D3Q39 velocity set; cs²=2/3) — POND maps to :d3q39 internally
    initialize_lattice!(:pond)
    cs2 = Float32(CS2_LATTICE); cs = sqrt(cs2)

    Ma   = Float32(REGIME_MA > 0 ? REGIME_MA : _pondcfg("mach"; default=1.5))
    Uinf = Ma * cs
    nx = Int(_pondcfg("nx"; default=192)); ny = Int(_pondcfg("ny"; default=112)); nz = Int(_pondcfg("nz"; default=112))
    dx = Float32(_pondcfg("dx"; default=1.0))
    R  = Float32(_pondcfg("sphere_radius"; default=12.0))
    cx = Float32(_pondcfg("sphere_cx"; default=nx*0.29)); cy = Float32(_pondcfg("sphere_cy"; default=ny*dx/2)); cz = Float32(_pondcfg("sphere_cz"; default=nz*dx/2))
    steps = Int(_pondcfg("steps"; default=4000)); ramp = Int(_pondcfg("ramp_steps"; default=800))
    eta = Float32(_pondcfg("eta"; default=1.0e-3)); wband = Float32(_pondcfg("wband"; default=1.0))
    dt  = Float32(0.6 / (3.0 + Float64(Uinf)))   # CFL: max signal speed = max|c|+U = 3+U
    # Reynolds → τ (DUGKS viscous flux). reynolds≤0 ⇒ τ=0 ⇒ inviscid (Re≈350 numerical, over-damped).
    # τ from the 1D-calibrated ν(τ) table (tools/test_pond_dugks1d.jl); ν = U·(2R)/Re.
    reynolds = Float64(_pondcfg("reynolds"; default=0.0))
    tau = 0.0f0
    if reynolds > 0 && R > 0
        νt = Float64(Uinf)*Float64(2R)/reynolds
        cal = [(0.0,0.0),(0.010,0.0054),(0.020,0.0115),(0.040,0.0261),(0.080,0.060)]
        for i in 1:length(cal)-1
            if νt >= cal[i][1] && νt <= cal[i+1][1]
                tau = Float32(cal[i][2] + (νt-cal[i][1])/(cal[i+1][1]-cal[i][1])*(cal[i+1][2]-cal[i][2])); break
            end
        end
        tau == 0.0f0 && (tau = Float32(0.6*νt))
        @printf("  Reynolds=%.0f → ν=%.4f → τ=%.4f (DUGKS viscous flux)\n", reynolds, νt, tau)
    end

    println("="^70)
    @printf("  PonD per-cell kinetic-FV case run | Ma=%.3f U∞=%.4f cs=%.4f dt=%.4f\n", Ma, Uinf, cs, dt)
    @printf("  grid %d×%d×%d  sphere R=%.1f @ (%.1f,%.1f,%.1f)  steps=%d ramp=%d eta=%.1e\n",
            nx,ny,nz, R, cx,cy,cz, steps, ramp, eta)
    println("  backend = ", CUDA.functional() ? "CUDA" : "CPU", "   → ", resdir)
    println("="^70)

    χ = R > 0 ? pond_sphere_chi(nx,ny,nz, cx,cy,cz, R, dx; wband=wband) : zeros(Float32,nx,ny,nz)
    d = pond_fv3d_driver(backend, dev, nx,ny,nz; dx=dx, dt=dt, cs2=cs2,
                         rho_inf=1.0f0, U_inf=Uinf, chi_host=χ, eta=eta, tau=tau,
                         ramp_steps=ramp, ng=2, cxv=C_X, cyv=C_Y, czv=C_Z, wv=W)

    qS = 0.5*Float64(1.0)*Float64(Uinf)^2*(pi*R^2)
    # time-average the drag over the last quarter of the run (the control-volume drag assumes
    # steady state; a single end-of-run snapshot of a slightly-unsteady wake is unreliable).
    avg_from = R > 0 ? round(Int, 0.75*steps) : steps + 1
    cv_samples = Float64[]; pen_samples = Float64[]
    nanstep = 0
    for t in 1:steps
        Uin = pond_fv3d_advance!(d, t)
        if t % 250 == 0
            ρh = Array(d.rho)
            if !all(isfinite, ρh); nanstep = t; break; end
            @printf("  step %5d  Uin=%.4f  ρ∈[%.3f,%.3f]\n", t, Uin, minimum(ρh), maximum(ρh))
        end
        if R > 0 && t >= avg_from && t % 50 == 0
            push!(cv_samples, pond_drag_controlvolume(d; inset=8)/qS)
            push!(pen_samples, pond_body_force(d)[1]/qS)
        end
    end

    summary = joinpath(resdir, "pond_summary.txt")
    if nanstep > 0
        CaseIDContract.verify_claimed_output(runtime, claimed_output)
        open(summary,"w") do io; @printf(io, "PonD case FAILED — NaN at step %d (Ma=%.3f)\n", nanstep, Ma); end
        @printf("  NaN at step %d — see %s\n", nanstep, summary)
        return BatchOutcomeContract.pond_case_outcome(false;
            case_name=runtime.case.case_id,
            error_msg="NaN detected at PonD step $nanstep",
            re_number=reynolds, ma_physical=Ma,
            total_cells=nx * ny * nz, wall_time=time() - outcome_start_time,
            steps=nanstep, start_time=outcome_start_datetime, end_time=Dates.now(),
            output_dir=resdir)
    end

    ρh = Array(d.rho); jc = round(Int,cy/dx); kc = round(Int,cz/dx)
    if !all(isfinite, ρh)
        CaseIDContract.verify_claimed_output(runtime, claimed_output)
        open(summary, "w") do io
            println(io, "PonD case FAILED — nonfinite final density state")
        end
        println("  PonD FAILED — nonfinite final density state — see ", summary)
        return BatchOutcomeContract.pond_case_outcome(false;
            case_name=runtime.case.case_id,
            error_msg="nonfinite final PonD density state",
            re_number=reynolds, ma_physical=Ma,
            total_cells=nx * ny * nz, wall_time=time() - outcome_start_time,
            steps=steps, start_time=outcome_start_datetime, end_time=Dates.now(),
            output_dir=resdir)
    end
    nose_i = max(round(Int,(cx-R)/dx), 1)
    ρ_stag = R > 0 ? maximum(ρh[max(nose_i-8,1):nose_i, jc, kc]) : maximum(ρh)
    mean(v) = isempty(v) ? 0.0 : sum(v)/length(v)
    Cd_cv  = isempty(cv_samples)  ? (R>0 ? pond_drag_controlvolume(d;inset=8)/qS : 0.0) : mean(cv_samples)
    Cd_pen = isempty(pen_samples) ? (R>0 ? pond_body_force(d)[1]/qS : 0.0)             : mean(pen_samples)
    if !all(isfinite, (Float64(ρ_stag), Float64(Cd_cv), Float64(Cd_pen)))
        CaseIDContract.verify_claimed_output(runtime, claimed_output)
        open(summary, "w") do io
            println(io, "PonD case FAILED — nonfinite final reported metric")
        end
        println("  PonD FAILED — nonfinite final reported metric — see ", summary)
        return BatchOutcomeContract.pond_case_outcome(false;
            case_name=runtime.case.case_id,
            error_msg="nonfinite final PonD reported metric",
            re_number=reynolds, ma_physical=Ma,
            total_cells=nx * ny * nz, wall_time=time() - outcome_start_time,
            steps=steps, start_time=outcome_start_datetime, end_time=Dates.now(),
            output_dir=resdir)
    end
    outcome = BatchOutcomeContract.pond_case_outcome(true;
        case_name=runtime.case.case_id,
        re_number=reynolds, ma_physical=Ma,
        total_cells=nx * ny * nz, wall_time=time() - outcome_start_time,
        steps=steps, start_time=outcome_start_datetime, end_time=Dates.now(),
        output_dir=resdir, cd_cv=Cd_cv, n_samples=length(cv_samples))
    CaseIDContract.verify_claimed_output(runtime, claimed_output)
    open(summary,"w") do io
        @printf(io, "PonD per-cell kinetic-FV — case %s\n", runtime.case.case_id)
        @printf(io, "Ma=%.3f  U_inf=%.4f  cs=%.4f  grid=%dx%dx%d  R=%.1f  steps=%d\n", Ma,Uinf,cs,nx,ny,nz,R,steps)
        @printf(io, "stable: yes   rho in [%.3f, %.3f]\n", minimum(ρh), maximum(ρh))
        @printf(io, "nose stagnation rho_stag = %.3f  (rise %.1f%%)\n", ρ_stag, 100*(ρ_stag-1))
        @printf(io, "Cd (control-volume, physical, time-avg last 25%%) = %.3f  [n=%d]\n", Cd_cv, length(cv_samples))
        @printf(io, "Cd (penalization-sink, time-avg last 25%%)        = %.3f\n", Cd_pen)
        @printf(io, "NOTE: isothermal EOS ⇒ Cd above the adiabatic-air ~0.9-1.0 ref (Cp_stag~1.56); see handover.\n")
    end
    @printf("  STABLE. ρ_stag=%.3f  Cd_cv=%.3f  Cd_pen=%.3f (time-avg) → %s\n", ρ_stag, Cd_cv, Cd_pen, summary)

    # presentation-quality mid-plane (z-centre) field slices: density, Mach, |u|, pressure
    if isdefined(Main, :Plots) || isdefined(@__MODULE__, :Plots)
        try
            uxh = Array(d.ux); uyh = Array(d.uy); uzh = Array(d.uz)
            umag = sqrt.(uxh.^2 .+ uyh.^2 .+ uzh.^2)
            mach = umag ./ cs                       # local Mach (isothermal: a = cs everywhere)
            pres = ρh .* cs2                          # isothermal pressure p = ρ cs²
            tag = "PonD $(runtime.case.case_id) Ma=$(round(Ma,digits=2))"
            for (field, data, name) in (("density", ρh, "ρ"), ("mach", mach, "Mach"),
                                        ("umag", umag, "|u|"), ("pressure", pres, "p"))
                sl = data[:, :, kc]'
                p = Plots.heatmap(sl, aspect_ratio=1, c=:jet, title="$tag — $name (z-midplane)",
                                  xlabel="x", ylabel="y")
                CaseIDContract.verify_claimed_output(runtime, claimed_output)
                fn = joinpath(resdir, "pond_$(field)_midplane.png"); Plots.savefig(p, fn)
                println("  $field slice → ", fn)
            end
        catch e
            @warn "PonD slice render skipped" exception=e
        end
    end
    return outcome
end
