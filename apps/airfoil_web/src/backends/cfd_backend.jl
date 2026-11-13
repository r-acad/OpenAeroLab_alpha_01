module CFDBackend

using Printf
using TOML
using SHA
if !isdefined(parentmodule(@__MODULE__), :AirfoilJobControl)
    Base.include(parentmodule(@__MODULE__), joinpath(@__DIR__, "..", "job_control.jl"))
end
using ..AirfoilJobControl
include(joinpath(@__DIR__, "rans_acceleration.jl"))
using .RANSAcceleration
include(joinpath(@__DIR__, "cfd_result_qualification.jl"))
using .CFDResultQualification
include(joinpath(@__DIR__, "..", "workers", "implicit_methods.jl"))
include(joinpath(@__DIR__, "rans_convergence.jl"))
include(joinpath(@__DIR__, "rans_checkpoint.jl"))
include(joinpath(@__DIR__, "..", "protocol.jl"))
using .AirfoilWire
include(joinpath(@__DIR__, "..", "full_field_artifact.jl"))
using .FullFieldArtifact
include(joinpath(@__DIR__, "..", "diagnostics", "convergence.jl"))
using .CFDConvergence
include(joinpath(@__DIR__, "..", "diagnostics", "mass_diagnostics.jl"))
using .CFDMassDiagnostics
include(joinpath(@__DIR__, "..", "diagnostics", "equation_residuals.jl"))
using .EquationResidualProbe
include(joinpath(@__DIR__, "..", "diagnostics", "time_statistics.jl"))
using .CFDTimeStatistics

const ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
const SRC = joinpath(ROOT, "src")
const LBM_F = Float32  # Required by src/collision/lattice.jl in the isolated CFD worker.
const IS_WORKER = abspath(PROGRAM_FILE) == abspath(@__FILE__) &&
                  !isempty(ARGS) && ARGS[1] in ("--transonic-worker","--species-worker","--ch-transonic-worker")
include(joinpath(SRC, "cmesh", "cmesh_gen.jl"))
include(joinpath(SRC, "cmesh", "cmesh_metrics.jl"))
include(joinpath(@__DIR__, "..", "mesh", "transonic_mesh.jl"))
include(joinpath(@__DIR__, "..", "mesh", "stall_layer_mesh.jl"))
include(joinpath(@__DIR__, "..", "mesh", "ch_mesh.jl"))
# Gurney flap (23 Oct 2026; a baffle in the low-speed CFD since 28 Oct 2026): the pure geometry module, the regional base audit and the backend hooks; inert when off.
include(joinpath(@__DIR__, "..", "mesh", "gurney.jl"))
include(joinpath(@__DIR__, "..", "mesh", "gurney_mesh_audit.jl"))   # the regional base audit
include(joinpath(@__DIR__, "gurney_backend.jl"))

# Load solver methods before calling the worker. Julia 1.12 world-age rules do not permit
# defining these modules inside _transonic_worker and immediately invoking their methods.
if IS_WORKER
    include(joinpath(SRC, "collision", "lattice.jl"))
    include(joinpath(SRC, "compressible", "cmesh_solver.jl"))
    include(joinpath(SRC, "compressible", "cmesh_solver_ka.jl"))
    if ARGS[1] == "--species-worker"
        include(joinpath(SRC,"cavitation","cmesh_cav_tem.jl"))
        include(joinpath(SRC,"cavitation","cmesh_cav_species.jl"))
    end
    @eval using KernelAbstractions
    TOML.parsefile(ARGS[2])["gpu"] && (@eval import CUDA)
end

_get(d::AbstractDict, key::AbstractString, default=nothing) =
    haskey(d, key) ? d[key] : (haskey(d, Symbol(key)) ? d[Symbol(key)] : default)

function _number(d::AbstractDict, key::AbstractString, default; lo=-Inf, hi=Inf)
    raw = _get(d, key, default)
    x = raw isa AbstractString ? parse(Float64, raw) : Float64(raw)
    isfinite(x) && lo <= x <= hi || throw(ArgumentError("$key must be finite and in [$lo, $hi]"))
    x
end

function _integer(d::AbstractDict, key::AbstractString, default; lo=1, hi=typemax(Int))
    x = _number(d, key, default; lo=lo, hi=hi)
    isinteger(x) || throw(ArgumentError("$key must be an integer"))
    Int(x)
end

function _boolean(d::AbstractDict, key::AbstractString, default=false)
    x = _get(d, key, default)
    x isa Bool ? x : lowercase(string(x)) in ("1", "true", "yes")
end

# Validation-only: inspect the same raw continuity operator at selected
# intermediate states of one RK/SFD/SST step. Copies are sparse and remain on
# the host; the normal worker never enters this path.
function _mass_stage_state(s, vol_host; refresh::Bool=true)
    refresh && CMeshSolverKA.refresh_raw_residual_ka!(s; wallmode=Int32(2))
    dc = Array(s.dc_r)
    scaled = abs.(Float64.(dc) ./ vol_host)
    idx = argmax(scaled)
    (maximum=scaled[idx], rms=sqrt(sum(abs2,scaled)/length(scaled)),
     i=idx[1], j=idx[2], signed=Float64(dc[idx])/vol_host[idx])
end

function _sfd_mass_ledger(before_rho, after_rho, dtf_host, vol_host,
                          g::Integer, rho_inf::Real, speed_inf::Real, chord::Real,
                          peak_i::Integer, peak_j::Integer)
    ni,nj=size(vol_host)
    sumsquare=0.0; maximum_source=0.0; integral_source=0.0
    peak_source=0.0
    for j in 1:nj, i in 1:ni
        source=(Float64(after_rho[i+g,j+g])-Float64(before_rho[i+g,j+g])) /
               Float64(dtf_host[i,j])
        sumsquare+=source*source
        maximum_source=max(maximum_source,abs(source))
        integral_source+=source*Float64(vol_host[i,j])
        if i==peak_i && j==peak_j; peak_source=source; end
    end
    scale=Float64(rho_inf)*Float64(speed_inf)*Float64(chord)
    (maximum=maximum_source,rms=sqrt(sumsquare/(ni*nj)),
     integral=integral_source/scale,at_peak=peak_source)
end

function _deliver_log(log::Function, line::AbstractString)
    try
        log(String(line))
    catch err
        @warn "CFD log callback failed" exception=(err, catch_backtrace())
    end
end

function _alpha_values(case::AbstractDict)
    flow = _get(case, "flow", Dict())
    sweep = _get(case, "sweep", Dict())
    flow isa AbstractDict && sweep isa AbstractDict || throw(ArgumentError("flow and sweep must be objects"))
    if !_boolean(sweep, "enabled")
        return [_number(flow, "alpha_deg", 0.0; lo=-30, hi=30)]
    end
    a = _number(sweep, "start", 0.0; lo=-30, hi=30)
    b = _number(sweep, "stop", 10.0; lo=-30, hi=30)
    h = _number(sweep, "step", 1.0; lo=-60, hi=60)
    h != 0 && (b-a)*h >= 0 || throw(ArgumentError("alpha sweep step must advance toward stop"))
    n = floor(Int, (b-a)/h + 1e-10) + 1
    1 <= n <= 61 || throw(ArgumentError("alpha sweep must contain 1–61 points"))
    [a + (i-1)*h for i in 1:n]
end

function _geometry(case::AbstractDict)
    foil = _get(case, "airfoil", Dict())
    foil isa AbstractDict || throw(ArgumentError("airfoil must be an object"))
    path = _get(foil, "path", nothing)
    path isa AbstractString && isfile(path) || throw(ArgumentError("airfoil.path must name a prepared .dat file"))
    abspath(path)
end

"""Create and record a gas-CFD-only sharp trailing edge for the free-air C-grid.

The original section is never overwritten. The endpoint shift tapers as ξ^10 from the leading
edge to each trailing-edge corner, so the two corners meet at their midpoint. Any coefficient
comparison against an unmodified XFoil section must carry this geometry difference.
"""
function _sharp_te_geometry(path::AbstractString, dir::AbstractString)
    af = CMeshGen.read_airfoil_dat(path)
    gap = hypot(af.x[1]-af.x[end], af.y[1]-af.y[end])
    if gap <= 1e-8
        return String(path), Dict{String,Any}("source_te_gap"=>gap,
            "cfd_te_gap"=>gap, "transform"=>"none", "path"=>String(path))
    end
    ile = argmin(af.x)
    2 < ile < length(af.x)-1 || throw(ArgumentError("airfoil leading-edge position is invalid"))
    xm = 0.5*(af.x[1]+af.x[end]); ym = 0.5*(af.y[1]+af.y[end])
    xx = copy(af.x); yy = copy(af.y)
    for k in eachindex(xx)
        te = k <= ile ? 1 : length(xx)
        denom = af.x[te]-af.x[ile]
        denom > 0 || throw(ArgumentError("airfoil has invalid chord direction"))
        weight = clamp((af.x[k]-af.x[ile])/denom, 0.0, 1.0)^10
        xx[k] += weight*(xm-af.x[te])
        yy[k] += weight*(ym-af.y[te])
    end
    derived = joinpath(dir,"airfoil_cfd_sharp_te.dat")
    open(derived,"w") do io
        println(io,"CFD sharp-TE derivative of ",basename(path))
        for k in reverse(eachindex(xx))
            @printf(io,"%.17g %.17g\n",xx[k],yy[k])
        end
    end
    checked = CMeshGen.read_airfoil_dat(derived)
    new_gap = hypot(checked.x[1]-checked.x[end],checked.y[1]-checked.y[end])
    new_gap <= 1e-8 || error("sharp-TE derivative failed to close the edge: $new_gap")
    derived, Dict{String,Any}("source_te_gap"=>gap, "cfd_te_gap"=>new_gap,
        "transform"=>"aft taper ξ^10 to trailing-edge midpoint", "path"=>derived,
        "source_path"=>String(path))
end

function _log_process(cmd::Cmd, log_path::AbstractString, progress::Function,
                      log::Function, backend::String, alpha::Float64;
                      physical_seconds_per_ctu::Union{Nothing,Real}=nothing,
                      residual_kind::Union{Nothing,AbstractString}=nothing)
    AirfoilJobControl.check_cancelled()
    open(log_path, "w") do io
        output = Pipe()
        proc = run(pipeline(AirfoilJobControl.process_command(cmd); stdout=output, stderr=output); wait=false)
        owned = AirfoilJobControl.track!(proc)
        reader = nothing
        wpid = getpid(proc)
        try
        close(output.in)  # Only the child writes; EOF must reach the reader.
        gpu_wait = Ref{Union{Nothing,String}}(nothing)   # the low-speed worker waiting for the shared GPU lock (gpu_lock.jl)
        reader = @async begin
            last_iteration_event = nothing
            last_physical_clock = nothing
            for line in eachline(output)
                println(io, line)
                flush(io)
                # The worker emits a compact machine-readable record as soon as each
                # history row is flushed. Send it through the existing job progress
                # channel so charts can update before the point finishes.
                values = split(line, ',')
                if backend in ("transonic","cavitation","lowspeed") && length(values) == 2 && values[1] == "MESH_READY"
                    # The worker writes this before GPU setup or time marching.
                    preview_path = joinpath(dirname(log_path), "mesh_preview.msgpack")
                    if values[2] == basename(preview_path) && isfile(preview_path)
                        mesh = AirfoilWire.decode(read(preview_path), :msgpack)
                        mesh["artifact"] = joinpath(dirname(log_path), "mesh_full.msgpack")
                        mesh["preview_artifact"] = preview_path
                        progress(Dict("event"=>"mesh_ready", "backend"=>backend,
                                      "alpha_deg"=>alpha, "mesh"=>mesh))
                        description=get(mesh,"kind","")=="structured_blocks" ?
                            "$(get(mesh,"total_cells",0)) cells across $(length(mesh["blocks"])) connected blocks" :
                            "$(mesh["full_ni"]-1) × $(mesh["full_nj"]-1) cells"
                        _deliver_log(log, "CFD mesh ready for inspection at alpha=$alpha deg: $description.")
                    end
                elseif backend in ("lowspeed","transonic") && length(values) == 3 && values[1] == "FIELD_SNAPSHOT" &&
                       occursin(r"^live_field_[0-9]+\.msgpack$", values[2]) && tryparse(Int, values[3]) !== nothing
                    # Committed CFD field, announced only after a complete, checked write.
                    artifact=joinpath(dirname(log_path),values[2])
                    if !isfile(artifact) || filesize(artifact)==0
                        _deliver_log(log,"SNAPSHOT_WARNING: ignored missing or empty field artifact $(values[2]).")
                        continue
                    end
                    progress(Dict("event"=>"field_snapshot", "backend"=>backend, "alpha_deg"=>alpha,
                                  "artifact"=>artifact, "iteration"=>parse(Int, values[3])))
                elseif backend in ("transonic","cavitation") && length(values)==5 && values[1]=="PHYSICAL_TIME"
                    n=tryparse(Int,values[2])
                    t=tryparse(Float64,values[3]);duration=tryparse(Float64,values[4]);dt=tryparse(Float64,values[5])
                    if n!==nothing && all(x->x!==nothing && isfinite(x),(t,duration,dt)) && t>=0 && duration>0 && dt>0
                        last_physical_clock=Dict{String,Any}("iteration"=>n,"t_ctu"=>t,
                            "requested_duration_ctu"=>duration,"applied_dt_ctu"=>dt)
                        if physical_seconds_per_ctu!==nothing
                            last_physical_clock["t_seconds"]=t*physical_seconds_per_ctu
                            last_physical_clock["requested_duration_seconds"]=duration*physical_seconds_per_ctu
                            last_physical_clock["dt_seconds"]=dt*physical_seconds_per_ctu
                        end
                        # Either worker output order is supported. If the force
                        # row arrived first, publish the same iteration enriched
                        # with its clock. Consumers merge by solver iteration.
                        if last_iteration_event!==nothing && last_iteration_event["iteration"]==n
                            progress(merge(last_iteration_event,last_physical_clock))
                        end
                    end
                    _deliver_log(log,line)
                elseif backend in ("transonic","cavitation","lowspeed") && length(values) in (8, 9, 10) && values[1] == "ITERATION"
                    n = tryparse(Int, values[2])
                    total = tryparse(Int, values[3])
                    r = tryparse(Float64, values[4])
                    rraw = tryparse(Float64, values[5])
                    rl2 = tryparse(Float64, values[6])
                    cl = tryparse(Float64, values[7])
                    cd = tryparse(Float64, values[8])
                    mass_net = length(values) >= 9 ? tryparse(Float64, values[9]) : nothing
                    cm = length(values) == 10 ? tryparse(Float64,values[10]) : nothing
                    if n !== nothing && total !== nothing &&
                       all(x -> x !== nothing && isfinite(x), (r, rraw, rl2, cl, cd)) &&
                       (length(values) == 8 || (mass_net !== nothing && isfinite(mass_net))) &&
                       (length(values) < 10 || (cm !== nothing && isfinite(cm)))
                        event = Dict{String,Any}("event"=>"iteration", "backend"=>backend,
                                      "alpha_deg"=>alpha, "iteration"=>n,
                                      "total_iterations"=>total,
                                      "residual"=>r, "residual_scaled"=>r,
                                      "residual_linf"=>rraw, "residual_l2"=>rl2,
                                      "cl"=>cl, "cd"=>cd)
                        residual_kind===nothing || (event["residual_kind"]=residual_kind)
                        mass_net === nothing || (event["mass_net_frac"] = mass_net)
                        cm === nothing || (event["cm"] = cm)
                        if last_physical_clock!==nothing && last_physical_clock["iteration"]==n
                            merge!(event,last_physical_clock)
                        end
                        last_iteration_event=event
                        progress(event)
                    end
                elseif backend == "lowspeed" && (startswith(line, "GPU_WAIT,") || startswith(line, "GPU_ACQUIRED,"))
            # The opt-in GPU run: the worker waits for the shared GPU lock and says who holds it (gpu_lock.jl).
            owner = String(line[findfirst(',', line)+1:end])
            if startswith(line, "GPU_WAIT,")
                gpu_wait[] == owner || _deliver_log(log, "Waiting for the GPU (held by $(owner)); the worker retries every 60 s and never breaks the lock.")
                gpu_wait[] = owner
                progress(Dict("event"=>"gpu_wait", "backend"=>backend, "alpha_deg"=>alpha, "owner"=>owner))
            else
                gpu_wait[] = nothing
                _deliver_log(log, "GPU lock taken: $(owner).")
                progress(Dict("event"=>"gpu_acquired", "backend"=>backend, "alpha_deg"=>alpha, "owner"=>owner))
            end
        else
            _deliver_log(log, line)
        end
    end
end
progress(Dict("event"=>"started", "backend"=>backend, "alpha_deg"=>alpha, "pid"=>wpid))
        started = time(); heartbeat = started
        while process_running(proc)
            sleep(1)
            if time()-heartbeat >= 10
                ev = Dict{String,Any}("event"=>"running", "backend"=>backend, "alpha_deg"=>alpha,
                              "elapsed_seconds"=>round(Int, time()-started))
                gpu_wait[] === nothing || (ev["gpu_wait"] = gpu_wait[])
                progress(ev)
                heartbeat = time()
            end
        end
        wait(proc)
        AirfoilJobControl.finish!(owned)
        wait(reader)  # Include final buffered lines before reporting completion.
        success(proc) || AirfoilJobControl.check_cancelled()
        success(proc)
        finally
            AirfoilJobControl.finish!(owned)
            wait(proc)
            reader === nothing || (try wait(reader) catch end)
            backend == "lowspeed" && AirfoilGPULock.release_if_pid!(wpid) &&
                _deliver_log(log, "Released the GPU lock that worker pid $(wpid) left behind.")
        end
    end
end

"The first actual Julia failure, without its potentially enormous type stack."
function _solver_failure_summary(path::AbstractString;max_chars::Int=800)
    isfile(path) || return "solver log is unavailable"
    for line in eachline(path)
        startswith(strip(line),"ERROR:") || continue
        summary=replace(strip(line),r"^ERROR:\s*(?:LoadError:\s*)?"=>"")
        return length(summary)>max_chars ? first(summary,max_chars)*"…" : summary
    end
    "worker exited unsuccessfully; inspect its retained log"
end

function _csv_rows(path::AbstractString)
    isfile(path) || throw(ArgumentError("solver did not write $(basename(path))"))
    lines = readlines(path)
    length(lines) >= 2 || throw(ArgumentError("solver wrote an empty $(basename(path))"))
    header = split(chomp(lines[1]), ',')
    [Dict(String(k)=>String(v) for (k,v) in zip(header, split(chomp(line), ',')))
        for line in lines[2:end] if !isempty(strip(line))]
end

function _float(s)
    x = tryparse(Float64, string(s))
    x === nothing ? nothing : (isfinite(x) ? x : nothing)
end

# XFOIL/AGARD convention: positive aerodynamic Cm is nose-up, opposite
# the mathematical positive-CCW z-torque x*Fy-y*Fx on the body.
@inline _aerodynamic_cm(xwall, ywall, fx, fy, q) =
    -((xwall-0.25)*fy-ywall*fx)/q

_strict_aplus_profile(mode)=mode in ("aplus_uniform_quality","aplus_uniform_fixed")
function _transonic_mesh_angle(mode,flow,default_angle)
    value=_number(flow,"cfd_farfield_angle_deg",mode=="aplus_uniform_fixed" ? 0. : default_angle;lo=-90,hi=90)
    mode=="aplus_uniform_fixed"&&value!=0 && throw(ArgumentError("The fixed A+ mesh requires contour angle zero; flow alpha remains independent"))
    value
end

function _transonic_point(case, path, flow, alpha, dir, progress, log, warmdir=nothing;
                          species_backend::Bool=false, mesh_anchor=nothing)
    if _auto_ch_selected(case,path,flow)
        species_backend && throw(ArgumentError("The conservative Species/cavitation app driver currently uses Legacy A-plus. C+H Run is available for the Compressible backend; select that backend for the new mesh."))
        return _ch_transonic_point(case,path,flow,alpha,dir,progress,log,warmdir)
    end
    streamline_enabled(flow) && throw(ArgumentError(STREAMLINE_CH_ONLY))
    mesh_mode = String(_get(flow, "cfd_mesh", "cavitation_aplus_farfield"))
    mesh_mode in ("cavitation_aplus_farfield", "aplus_nested_refined", "aplus_uniform_quality", "aplus_uniform_fixed", "legacy_sharp_cmesh") ||
        throw(ArgumentError("Unknown CFD mesh family"))
    mesh_mode == "legacy_sharp_cmesh" && _number(flow,"cfd_te_thickness",0.0) > 0 &&
        throw(ArgumentError("finite trailing-edge thickness requires an A-plus mesh"))
    cfd_path, geometry = if mesh_mode == "legacy_sharp_cmesh"
        _sharp_te_geometry(path, dir)
    else
        af = CMeshGen.read_airfoil_dat(path)
        gap = hypot(af.x[1]-af.x[end], af.y[1]-af.y[end])
        String(path), Dict{String,Any}(
            "source_te_gap"=>gap,"cfd_te_gap"=>gap,
            "transform"=>"none", "path"=>String(path))
    end
    geometry["mesh_mode"] = mesh_mode
    resolution = String(_get(flow, "cfd_resolution", "standard"))
    mesh_sizes = get(Dict("standard"=>(256,64,112),
                          "fine"=>(384,96,144),
                          "very_fine"=>(512,128,176)), resolution, nothing)
    mesh_sizes === nothing && throw(ArgumentError("cfd_resolution must be standard, fine, or very_fine"))
    _strict_aplus_profile(mesh_mode) && (mesh_sizes=(384,64,112))
    cfg = Dict{String,Any}(
        "geometry"=>cfd_path, "source_geometry"=>String(path),
        "mesh_mode"=>mesh_mode, "outdir"=>dir,
        "wake_cut_viscous_distance_model"=>"paired_cell_center_distance",
        "solver_ka_source_sha256"=>bytes2hex(sha256(read(joinpath(
            SRC, "compressible", "cmesh_solver_ka.jl")))),
        "mach"=>_number(flow, "mach", 0.73; lo=0.02, hi=0.98),
        "reynolds"=>_number(flow, "reynolds", 6.5e6; lo=1e3, hi=1e9),
        "alpha_deg"=>alpha,
        "time_mode"=>begin
            mode=String(_get(flow,"cfd_time_mode","steady"))
            mode in ("steady","physical") || throw(ArgumentError("cfd_time_mode must be steady or physical"))
            mode
        end,
        "duration_ctu"=>_number(flow,"cfd_duration_ctu",100.0;lo=0.001,hi=10000),
        "burnin_ctu"=>_number(flow,"cfd_burnin_ctu",50.0;lo=0.0,hi=10000),
        "block_ctu"=>_number(flow,"cfd_block_ctu",5.0;lo=0.001,hi=1000),
        "physical_time_refined"=>_boolean(flow,"cfd_physical_time_refined",false),
        "history_stride"=>_integer(flow,"cfd_history_stride",400;lo=1,hi=1000000),
        "display_every"=>_integer(flow,"cfd_display_every",0;lo=0,hi=1000000),
        "snapshot_interval_seconds"=>_number(flow,"snapshot_interval_seconds",600.;lo=0,hi=86400),
        "force_sample_dt_ctu"=>_number(flow,"cfd_force_sample_dt_ctu",0.1;lo=0.001,hi=1.0),
        # Validation-only outer-frame orientation; freestream remains at alpha_deg.
        "cfd_farfield_angle_deg"=>_transonic_mesh_angle(mesh_mode,flow,
                                            mesh_anchor === nothing ? alpha : mesh_anchor),
        # The earlier web defaults (20k steps, 96 wall-normal cells, bare reconstruction)
        # were startup transients. The clean-foil validation in the repository uses a
        # wall-resolved mesh, WENO/full viscous stress, and >=400k steps with SFD at 150k.
        # 500k gives a longer post-SFD force history at low Mach.
        "steps"=>_integer(flow, "cfd_steps", 500_000; lo=1, hi=2_000_000),
        "gpu"=>_boolean(flow, "cfd_gpu", true),
        # Correct the Float32 three-stage weight sum in the GPU air solver.
        # CPU runs have no KA stage weights, so their default remains false.
        "rk_weight_closure"=>_boolean(flow, "cfd_rk_weight_closure",
                                       _boolean(flow, "cfd_gpu", true) &&
                                       String(_get(flow,"cfd_time_mode","steady"))=="steady"),
        # Match the SST state ghost with its reflected wall/paired wake center.
        # Explicit false retains the legacy metric for controlled reproduction.
        "sst_wall_metric_correction"=>_boolean(flow,
            "cfd_sst_wall_metric_correction", true),
        # Validation opt-in: synchronize final flow ghosts before SST.
        "refresh_flow_bc_before_sst"=>_boolean(flow,
            "cfd_refresh_flow_bc_before_sst", false),
        # Validation opt-in: prescribe the incoming acoustic characteristic
        # at the two downstream C-grid i ends.
        "characteristic_i_outlet"=>_boolean(flow,
            "cfd_characteristic_i_outlet", false),
        "sst_nonorth_correction"=>_boolean(flow,
            "cfd_sst_nonorth_correction", false),
        "auto_stop"=>_boolean(flow, "cfd_auto_stop", true),
        "sfd_release"=>_boolean(flow,"cfd_sfd_release",true),
        "mass_ceiling"=>_number(flow, "cfd_mass_ceiling", 0.1; lo=1e-6, hi=1000.0),
        # Provisional signed global continuity gate, normalized by rho_inf*U_inf*chord.
        "mass_net_limit"=>_number(flow, "cfd_mass_net_limit", 0.001;
                                    lo=1e-8, hi=1.0),
        # Opt-in diagnostic only; does not alter marching or early-stop gates.
        "equation_probe"=>_boolean(flow, "cfd_equation_probe", false),
        # GPU filters use Float64 by default so slow filter updates are retained.
        # Flow variables remain Float32; the legacy filter is still selectable.
        "sfd_storage_type"=>begin
            storage = lowercase(String(_get(flow, "cfd_sfd_storage", _boolean(flow,"cfd_gpu",true) ? "float64" : "float32")))
            storage in ("float32", "float64") ||
                throw(ArgumentError("cfd_sfd_storage must be float32 or float64"))
            storage
        end,
        "viscosity_model"=>begin
            model = lowercase(String(_get(flow, "cfd_viscosity_model", "constant_kinematic")))
            model in ("constant_kinematic", "sutherland", "constant_dynamic") ||
                throw(ArgumentError("cfd_viscosity_model must be constant_kinematic, sutherland, or constant_dynamic"))
            model
        end,
        "freestream_temperature_k"=>_number(flow,
            "cfd_freestream_temperature_k", 288.15; lo=100.0, hi=2000.0),
        "resolution"=>resolution,
        "n_foil"=>_integer(flow, "cfd_n_foil", mesh_sizes[1]; lo=32, hi=2048),
        "n_wake"=>_integer(flow, "cfd_n_wake", mesh_sizes[2]; lo=8, hi=512),
        "n_normal"=>_integer(flow, "cfd_n_normal", mesh_sizes[3]; lo=16, hi=512),
        "n_base"=>_integer(flow, "cfd_n_base", _strict_aplus_profile(mesh_mode) ? 2 : 4; lo=1, hi=8),
        "cfd_te_thickness"=>_number(flow, "cfd_te_thickness", 0.0; lo=0.0, hi=0.05),
        # Hidden validation knob; growth=1 follows the archived generator path.
        "wake_first_spacing_growth"=>_number(flow,
            "cfd_wake_first_spacing_growth", 1.0; lo=1.0, hi=2.0),
        "wake_first_spacing_max"=>_number(flow,
            "cfd_wake_first_spacing_max", 0.01; lo=0.0, hi=0.1),
        "wake_first_spacing_delay_cells"=>_integer(flow,
            "cfd_wake_first_spacing_delay_cells", 0; lo=0, hi=511),
        "cfl"=>_number(flow, "cfd_cfl", 0.45; lo=0.01, hi=0.7),
        "sfd_start"=>_integer(flow, "cfd_sfd_start", 150_000; lo=1, hi=2_000_000),
        "trip_upper"=>_number(flow, "trip_upper", species_backend ? 0.05 : 0.03; lo=0.0, hi=1.0),
        "trip_lower"=>_number(flow, "trip_lower", 0.05; lo=0.0, hi=1.0))
    source = _get(case,"airfoil",Dict())
    if _get(source,"kind","") == "naca4" && mesh_mode != "legacy_sharp_cmesh"
        code = String(_get(source,"id",""))
        occursin(r"^[0-9]{4}$",code) || throw(ArgumentError("invalid NACA four-digit code"))
        cfg["naca4_code"] = parse(Int,code)
    end
    # Retain the older hidden IRS-J knob. The shared setting selects the same
    # steady-only implementation and is reported in the result receipt.
    accel = RANSAcceleration.policy(_boolean(flow,"rans_acceleration",false),
        species_backend ? :species_air : :compressible;
        physical=species_backend ?
            !(String(_get(flow,"cav_fluid","water"))=="air" && String(_get(flow,"cav_air_mode","steady"))=="steady") :
            cfg["time_mode"]=="physical")
    legacy_irsj = !species_backend && _boolean(flow,"cfd_irsj",false)
    cfg["rans_acceleration"] = accel.enabled || legacy_irsj
    cfg["rans_acceleration_method"] = legacy_irsj ? "j-line implicit residual smoothing" : accel.method
    (legacy_irsj || accel.irsj) && (cfg["irsj"] = true)
    accel.species_line && (cfg["species_steady_line"] = true)
    # Hidden mesh knob (21 Oct 2026): the stall layer (stall_layer_mesh.jl); absence leaves the mesh intact.
    _boolean(flow,"cfd_stall_layer",false) && (cfg["stall_layer"] = true)
    _get(flow,"cfd_le_ds_scale",nothing) === nothing ||
        (cfg["le_ds_scale"] = _number(flow,"cfd_le_ds_scale",1.0;lo=0.1,hi=1.0))
    # Hidden validation knob (21 Oct 2026, stall validation): fully turbulent SST. The trip
    # intermittency ramps from 0 at the trip station to 1 over 0.02 c (build_metrics), so a
    # trip at x = 0 still leaves the nose partly laminar; stations 0.05 c ahead of the
    # leading edge make it 1 on the whole foil. Absence leaves the trip inputs untouched.
    if _boolean(flow,"cfd_fully_turbulent",false)
        cfg["trip_upper"] = -0.05; cfg["trip_lower"] = -0.05; cfg["fully_turbulent"] = true
    end
    # Hidden validation probe; absence leaves the input and solver path intact.
    if _get(flow,"cfd_mass_probe_stride",nothing) !== nothing
        cfg["mass_probe_stride"] = _integer(flow,"cfd_mass_probe_stride",400;
                                             lo=1,hi=100_000)
        cfg["mass_probe_start"] = _integer(flow,"cfd_mass_probe_start",
                                            min(cfg["sfd_start"],cfg["steps"]);
                                            lo=1,hi=cfg["steps"])
    end
    if species_backend
        cfg["cav_fluid"] = String(_get(flow,"cav_fluid","water"))
        cfg["cav_domain"] = String(_get(flow,"cav_domain","tunnel"))
        cfg["sigma"] = _number(flow,"sigma",1.25;lo=0.01,hi=20)
        cfg["cav_channel_height"] = _number(flow,"cav_channel_height",1.28;lo=0.2,hi=100)
        cfg["cav_air_mode"] = String(_get(flow,"cav_air_mode","steady"))
        cfg["cav_duration_ctu"] = _number(flow,"cav_duration_ctu",10.0;lo=0.001,hi=1000)
        _get(flow,"cfd_burnin_ctu",nothing)===nothing && (cfg["burnin_ctu"]=0.5cfg["cav_duration_ctu"])
        cfg["cav_history_stride"] = _integer(flow,"cav_history_stride",200;lo=1,hi=1000000)
        merge!(cfg,_species_model_settings(flow;alpha=alpha))
    end
    mesh_anchor === nothing || (cfg["polar_mesh_anchor_deg"] = cfg["cfd_farfield_angle_deg"])
    if !species_backend && warmdir !== nothing
        warm = joinpath(warmdir, "state_checkpoint.jls")
        isfile(warm) && (cfg["warm_start"] = warm)
    end
    species_backend || RANSAcceleration.configure_compressible!(cfg,flow)
    cfg_path = joinpath(dir, "input.toml")
    open(cfg_path, "w") do io; TOML.print(io, cfg); end
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    worker_flag = species_backend ? "--species-worker" : "--transonic-worker"
    backend_name = species_backend ? "cavitation" : "transonic"
    cmd = `$julia --project=$SRC --startup-file=no -t auto $(@__FILE__) $worker_flag $cfg_path`
    ok = _log_process(cmd, joinpath(dir, "solver.log"), progress, log, backend_name, alpha)
    ok || throw(ErrorException("CMESH2D worker failed: $(_solver_failure_summary(joinpath(dir,"solver.log"))); see solver.log"))
    species_backend && return _collect_species_output(dir,alpha)
    meta = TOML.parsefile(joinpath(dir, "result.toml"))
    mesh_preview = AirfoilWire.decode(read(joinpath(dir, "mesh_preview.msgpack")), :msgpack)
    mesh_preview["artifact"] = joinpath(dir, "mesh_full.msgpack")
    mesh_preview["preview_artifact"] = joinpath(dir, "mesh_preview.msgpack")
    haskey(meta, "geometry") && merge!(geometry, meta["geometry"])
    cp = Dict{String,Any}("upper"=>Any[], "lower"=>Any[])
    for row in _csv_rows(joinpath(dir, meta["cp_csv"]))
        push!(cp[row["surface"]], [_float(row["x"]), _float(row["cp"])])
    end
    rows = _csv_rows(joinpath(dir, meta["field_csv"]))
    cols = ["x", "y", "mach", "cp", "pressure", "density", "u", "v", "temperature", "vorticity", "nut"]
    samples = Dict(k=>Float64[] for k in cols)
    for row in rows, k in cols
        value = _float(row[k])
        value === nothing && throw(ArgumentError("nonfinite $k in CFD field output"))
        push!(samples[k], value)
    end
    raw_history = _csv_rows(joinpath(dir, meta["iteration_csv"]))
    physical_history = Dict{Int,Any}()
    if haskey(meta,"physical_force_csv") && isfile(joinpath(dir,meta["physical_force_csv"]))
        for row in _csv_rows(joinpath(dir,meta["physical_force_csv"]))
            physical_history[parse(Int,row["iteration"])]=row
        end
    end
    history_stride = max(1, ceil(Int, length(raw_history)/1200))
    history_indices = unique(vcat(collect(1:history_stride:length(raw_history)), length(raw_history)))
    iteration_history = map(history_indices) do k
        row = Dict{String,Any}(
            "iteration"=>parse(Int, raw_history[k]["iteration"]),
            "residual"=>_float(raw_history[k]["residual_scaled"]),
            "residual_linf"=>_float(raw_history[k]["residual_linf"]),
            "residual_l2"=>_float(raw_history[k]["residual_l2"]),
            "cl"=>_float(raw_history[k]["cl"]),
            "cd"=>_float(raw_history[k]["cd"]))
        mass_net = _float(get(raw_history[k], "mass_net_frac", ""))
        mass_net === nothing || (row["mass_net_frac"] = mass_net)
        physical=get(physical_history,row["iteration"],nothing)
        if physical!==nothing
            for key in ("t_ctu","cm","applied_dt_ctu")
                value=_float(get(physical,key,""))
                value===nothing || (row[key]=value)
            end
        end
        row
    end
    numerically_stationary = get(meta, "numerically_stationary", false)
    physical_sanity = get(meta, "physical_sanity", false)
    guard = get(meta, "convergence_guard", Dict{String,Any}())
    guard_status = get(guard, "status", "unavailable")
    physical_time=get(meta,"time_mode","steady")=="physical"
    time_statistics=get(meta,"time_statistics",Dict{String,Any}())
    physical_status=!get(time_statistics,"sampling_adequate",false) ? "insufficient_sampling" :
        (get(time_statistics,"physical_mean_eligible",false) ? "completed" : "unqualified_physical_mean")
    point=Dict{String,Any}(
        "backend"=>"transonic", "alpha_deg"=>alpha,
        "status"=>physical_time ? physical_status :
            (guard_status == "stalled_stop" ? "stopped_without_convergence" :
                (guard_status == "early_stop_eligible" ? "completed" : "not_converged")),
        "convergence"=>physical_time ? "physical time; sampling status: $(get(time_statistics,"status","insufficient_sampling")); coefficient fields retain the instantaneous final state" : guard_status == "early_stop_eligible" ?
            "continuity and CL/CD history passed the conservative stop guard; momentum, energy, turbulence, mesh independence, and physical validation remain unverified" :
            "convergence guard: $guard_status; $(get(guard,"reason","inspect the iteration history"))",
        "validation"=>"experimental generic airfoil path",
        "geometry"=>geometry,
        "mesh"=>mesh_preview,
        "cl"=>meta["cl"], "cd"=>meta["cd"], "cm"=>meta["cm"],
        "iteration_history"=>iteration_history,
        "cp"=>cp, "fields"=>Dict("kind"=>"samples", "samples"=>samples,
                                 "sample_shape"=>[meta["sampled_ni"], meta["sampled_nj"]],
                                 "sample_stride"=>meta["sample_stride"],
                                 "artifact"=>joinpath(dir, meta["field_csv"]),
                                 "full_artifact"=>joinpath(dir, meta["field_full_msgpack"]),
                                 "full_cell_count"=>meta["full_field_cells"],
                                 "mesh_sha256"=>meta["full_field_mesh_sha256"],
                                 "full_iteration"=>meta["steps"]),
        "artifacts"=>Dict("cp_csv"=>joinpath(dir, meta["cp_csv"]),
                          "mesh_full"=>joinpath(dir, "mesh_full.msgpack"),
                          "mesh_preview"=>joinpath(dir, "mesh_preview.msgpack"),
                          "field_csv"=>joinpath(dir, meta["field_csv"]),
                          "field_full"=>joinpath(dir, meta["field_full_msgpack"]),
                          "iteration_csv"=>joinpath(dir, meta["iteration_csv"]),
                          "implicit_history_csv"=>get(meta,"implicit_method",false) ? joinpath(dir,"implicit_history.csv") : nothing,
                          "steady_assessment_csv"=>get(meta,"implicit_method",false) ? joinpath(dir,"steady_assessment_history.csv") : nothing,
                          "physical_force_csv"=>haskey(meta,"physical_force_csv") ? joinpath(dir,meta["physical_force_csv"]) : nothing,
                          "log"=>joinpath(dir, "solver.log")),
        "diagnostics"=>Dict("steps"=>meta["steps"], "max_steps"=>get(meta,"max_steps",meta["steps"]),
                            "rans_method"=>get(meta,"rans_method","standard"),
                            "implicit_method"=>get(meta,"implicit_method",false),
                            "rans_acceleration_method"=>get(meta,"rans_acceleration_method","standard"),
                            "implicit_maturity"=>get(meta,"implicit_maturity",""),
                            "implicit_numerical_controls"=>get(meta,"implicit_numerical_controls",Dict{String,Any}()),
                            "implicit_raw_sst_residuals"=>get(meta,"implicit_raw_sst_residuals",Dict{String,Any}()),
                            "implicit_accepted_steps"=>get(meta,"implicit_accepted_steps",0),
                            "implicit_stop_reason"=>get(meta,"implicit_stop_reason",""),
                            "steady_assessment"=>get(meta,"steady_assessment",Dict()),
                            "warm_start_source"=>get(meta,"warm_start_source",""),
                            "time_mode"=>get(meta,"time_mode","steady"),"clock"=>get(meta,"clock","steady pseudo-time"),
                            "elapsed_ctu"=>get(meta,"elapsed_ctu",nothing),
                            "requested_duration_ctu"=>get(meta,"requested_duration_ctu",nothing),
                            "physical_duration_reached"=>get(meta,"physical_duration_reached",false),
                            "field_time_scope"=>get(meta,"field_time_scope","final steady iterate"),
                            "time_statistics"=>time_statistics,
                            "convergence_guard"=>guard,
                            "residual_linf"=>meta["residual_linf"],
                            "residual_scaled"=>meta["residual_scaled"],
                            "residual_l2"=>meta["residual_l2"],
                            "mass_net_frac"=>get(meta,"mass_net_frac",nothing),
                            "mass_net_scope"=>get(meta,"mass_net_scope",nothing),
                            "residual_ratio"=>meta["residual_ratio"],
                            "residual_reference_iteration"=>meta["residual_reference_iteration"],
                            "residual_scope"=>meta["residual_scope"],
                            "mass_residual_peak_cell"=>meta["mass_residual_peak_cell"],
                            "mass_residual_fraction_near_peak"=>meta["mass_residual_fraction_near_peak"],
                            "mass_balance"=>get(meta,"mass_balance",Dict{String,Any}()),
                            "mesh_cells"=>meta["mesh_cells"],
                            "load_drift"=>get(meta,"load_drift",nothing),
                            "prior_load_drift"=>get(meta,"prior_load_drift",nothing),
                            "drag_drift"=>get(meta,"drag_drift",nothing),
                            "prior_drag_drift"=>get(meta,"prior_drag_drift",nothing),
                            "cl_mean_shift"=>get(meta,"cl_mean_shift",nothing),
                            "cd_mean_shift"=>get(meta,"cd_mean_shift",nothing),
                            "numerically_stationary"=>numerically_stationary,
                            "stationarity_scope"=>meta["stationarity_scope"],
                            "stationarity_criteria"=>Dict(
                                "cl_span_limit"=>meta["stationarity_cl_span_limit"],
                                "cd_span_limit"=>meta["stationarity_cd_span_limit"],
                                "cl_mean_limit"=>meta["stationarity_cl_mean_limit"],
                                "cd_mean_limit"=>meta["stationarity_cd_mean_limit"],
                                "window_iterations"=>meta["stationarity_window_iterations"]),
                            "physical_sanity"=>physical_sanity,
                            "physical_sanity_scope"=>meta["physical_sanity_scope"],
                            "cl_pressure"=>meta["cl_pressure"],
                            "cl_friction"=>meta["cl_friction"],
                            "cd_pressure"=>meta["cd_pressure"],
                            "cd_friction"=>meta["cd_friction"],
                            "cl_pressure_upper"=>get(meta,"cl_pressure_upper",nothing),
                            "cl_pressure_lower"=>get(meta,"cl_pressure_lower",nothing),
                            "cl_pressure_base"=>get(meta,"cl_pressure_base",nothing),
                            "cd_pressure_upper"=>get(meta,"cd_pressure_upper",nothing),
                            "cd_pressure_lower"=>get(meta,"cd_pressure_lower",nothing),
                            "cd_pressure_base"=>get(meta,"cd_pressure_base",nothing),
                            "cm_pressure_upper"=>get(meta,"cm_pressure_upper",nothing),
                            "cm_pressure_lower"=>get(meta,"cm_pressure_lower",nothing),
                            "cm_pressure_base"=>get(meta,"cm_pressure_base",nothing),
                            "pressure_region_scope"=>get(meta,"pressure_region_scope",nothing),
                            "wall_normal_vector_closure"=>get(meta,"wall_normal_vector_closure",nothing),
                            "gpu"=>meta["gpu"], "reconstruction"=>meta["reconstruction"],
                            "rk_weight_closure"=>get(meta,"rk_weight_closure",false),
                            "sst_wall_metric_correction"=>get(meta,
                                "sst_wall_metric_correction",false),
                            "refresh_flow_bc_before_sst"=>get(meta,
                                "refresh_flow_bc_before_sst",false),
                            "characteristic_i_outlet"=>get(meta,
                                "characteristic_i_outlet",false),
                            "sst_nonorth_correction"=>get(meta,
                                "sst_nonorth_correction",false),
                            "full_viscous_stress"=>meta["full_viscous_stress"],
                            "sfd_engaged"=>meta["sfd_engaged"],
                            "sfd_storage_type"=>get(meta,"sfd_storage_type","float32"),
                            "residual_sampling_state"=>get(meta,"residual_sampling_state","unspecified"),
                            "viscosity_model"=>get(meta,"viscosity_model","constant_kinematic"),
                            "freestream_temperature_k"=>get(meta,"freestream_temperature_k",nothing),
                            "sfd_start"=>get(meta,"sfd_start",150_000),
                            "estimated_yplus"=>meta["estimated_yplus"],
                            "mesh_mode"=>meta["mesh_mode"],
                            "wake_cut_viscous_distance_model"=>get(meta,
                                "wake_cut_viscous_distance_model", "historical_unspecified"),
                            "solver_ka_source_sha256"=>get(meta,
                                "solver_ka_source_sha256", nothing),
                            "mesh_topology"=>meta["geometry"]["mesh_topology"],
                            "outer_boundary_condition"=>meta["geometry"]["outer_boundary_condition"],
                            "n_base"=>meta["geometry"]["n_base"],
                            "requested_n_base"=>get(meta["geometry"], "requested_n_base", 0),
                            "source_te_gap"=>meta["geometry"]["source_te_gap"],
                            "cfd_te_gap"=>meta["geometry"]["cfd_te_gap"],
                            "trip_upper"=>meta["trip_upper"],
                            "trip_lower"=>meta["trip_lower"]),
        "notes"=>["Compressible CMESH2D with SST and a free-air C-grid. The generic mesh and solution have no case-specific validation.",
                    geometry["transform"] == "none" ? "The CFD geometry preserves the source trailing edge." :
                        "CFD geometry change: $(geometry["transform"]). XFoil uses the source section; account for the recorded geometry difference when comparing results.",
                    meta["mesh_mode"] != "legacy_sharp_cmesh" ?
                        "The finite trailing-edge base is included in force integrals; Cp curves show the upper and lower surfaces only." :
                        "This explicit legacy mode uses the sharp-edge C-grid.",
                    "A completed iteration count is not a convergence certificate. The far-field correction is subsonic.",
                    physical_sanity ? "Loose force bounds passed; compare Cp and loads with XFoil/reference data." :
                                      "Loose force bounds failed; do not use these coefficients as a polar datum."])
    qualify_result_point(point)
end

"""Build the same external-flow mesh as the CFD worker, without loading a solver.

The web preview runs this host-side geometry work in a separate task. It never
allocates CUDA state or advances the flow equations.
"""
function preview_mesh(path::AbstractString, flow::AbstractDict, outdir::AbstractString;
                      source::AbstractDict=Dict(),solver_backend="transonic")
    # Gurney flap (28 Oct 2026): a baffle, which only the low-speed solver has; refused here with the reason (gurney_backend.jl).
    gurney_enabled(flow) && gurney_check_request(flow,[solver_backend])
    resolution = String(_get(flow,"cfd_resolution","standard"))
    sizes = get(Dict("standard"=>(256,64,112), "fine"=>(384,96,144),
                     "very_fine"=>(512,128,176)), resolution, nothing)
    sizes === nothing && throw(ArgumentError("unknown CFD mesh resolution"))
    re = _number(flow,"reynolds",300_000.0;lo=1e3,hi=1e9)
    alpha = _number(flow,"alpha_deg",0.0;lo=-90,hi=90)
    cf = 0.026*re^(-1/7)
    dn1 = 0.7/(re*sqrt(cf/2))
    mode = String(_get(flow,"cfd_mesh","cavitation_aplus_farfield"))
    _strict_aplus_profile(mode) && (sizes=(384,64,112))
    mode == "legacy_sharp_cmesh" && _number(flow,"cfd_te_thickness",0.0) > 0 &&
        throw(ArgumentError("finite trailing-edge thickness requires an A-plus mesh"))
    equation_naca = _get(source,"kind","") == "naca4" && mode != "legacy_sharp_cmesh"
    af = equation_naca ? CMeshGen.naca4_points(parse(Int,String(_get(source,"id","")))) :
                         CMeshGen.read_airfoil_dat(path)
    topology=String(_get(flow,"cfd_te_topology","aplus"))
    topology in ("auto","aplus") || throw(ArgumentError("unknown trailing-edge topology"))
    conditioned=CMeshGen.condition_airfoil(af;trailing_edge_thickness=_number(flow,"cfd_te_thickness",0.;lo=0,hi=.05))
    finite=hypot(conditioned.airfoil.x[1]-conditioned.airfoil.x[end],conditioned.airfoil.y[1]-conditioned.airfoil.y[end])>1e-8
    if topology=="auto" && finite
        mode=="legacy_sharp_cmesh" && throw(ArgumentError("Auto C+H preserves the native finite trailing edge; choose the shared mesh family, or explicitly choose Legacy A-plus before requesting a sharp derivative"))
        built=_build_ch_mesh(af,flow;resolution,solver_backend)
        # streamline alignment (opt-in, development): the same morph the C+H worker applies at Run (ch_streamline.jl)
        st=streamline_worker_cfg(source,flow,alpha,outdir)
        aligned=st===nothing ? nothing : _ch_streamline_apply(built,st)
        aligned===nothing || (built=(;mesh=aligned.mesh,metadata=aligned.metadata))
        payload=_write_ch_mesh_artifacts(built.mesh,built.metadata,outdir,resolution;streamlines=aligned===nothing ? nothing : aligned.streamlines)
        payload["artifact"]=joinpath(outdir,"mesh_full.msgpack")
        payload["preview_artifact"]=joinpath(outdir,"mesh_preview.msgpack")
        return (;mesh=payload,metadata=built.metadata)
    end
    streamline_enabled(flow) && throw(ArgumentError(STREAMLINE_CH_ONLY))
    n_foil = _integer(flow,"cfd_n_foil",sizes[1];lo=32,hi=2048)
    n_wake = _integer(flow,"cfd_n_wake",sizes[2];lo=8,hi=512)
    n_normal = _integer(flow,"cfd_n_normal",sizes[3];lo=16,hi=512)
    if mode == "legacy_sharp_cmesh"
        sharp_path, metadata = _sharp_te_geometry(path,outdir)
        af = CMeshGen.read_airfoil_dat(sharp_path)
        sharp = _sharp_nose_mesh(af;n_foil,n_wake,n_normal,dn1,
            le_ds_scale=_number(flow,"cfd_le_ds_scale",1.0;lo=0.1,hi=1.0))
        mesh = sharp.mesh
        metadata["n_base"] = 0
        metadata["wall_dn1"] = dn1
        metadata["surface_distribution"] = "curvature"
        metadata["leading_edge_target_spacing"] = sharp.le_ds
        metadata["leading_edge_max_curvature"] = sharp.le_kappa
        metadata["quality_report"] = _transonic_quality_report(mesh)
    else
        mode in ("cavitation_aplus_farfield","aplus_nested_refined","aplus_uniform_quality","aplus_uniform_fixed") ||
            throw(ArgumentError("unknown CFD mesh family"))
        built = _build_transonic_mesh(af;n_foil,n_wake,n_normal,dn1,
            alpha_deg=alpha,
            trip_upper=_number(flow,"trip_upper",0.03;lo=0,hi=1),
            trip_lower=_number(flow,"trip_lower",0.05;lo=0,hi=1),
            n_base=_integer(flow,"cfd_n_base",_strict_aplus_profile(mode) ? 2 : 4;lo=1,hi=8),
            cfd_te_thickness=_number(flow,"cfd_te_thickness",0.0;lo=0,hi=0.05),
            mesh_refinement=_strict_aplus_profile(mode) ? Symbol(mode) : mode == "aplus_nested_refined" ? :aplus_nested_refined : :none,
            normal_collar=!_strict_aplus_profile(mode),shock_layer=!_strict_aplus_profile(mode),
            wake_first_spacing_growth=_number(flow,"cfd_wake_first_spacing_growth",1.0;lo=1,hi=2),
            wake_first_spacing_max=_number(flow,"cfd_wake_first_spacing_max",0.01;lo=0,hi=0.1),
            wake_first_spacing_delay_cells=_integer(flow,"cfd_wake_first_spacing_delay_cells",0;lo=0,hi=511),
            farfield_angle_deg=_transonic_mesh_angle(mode,flow,alpha),
            stall_layer=_boolean(flow,"cfd_stall_layer",false),
            le_ds_scale=_number(flow,"cfd_le_ds_scale",1.0;lo=0.1,hi=1.0))
        mesh = built.mesh
        metadata = built.metadata
    end
    metadata["geometry_support"] = equation_naca ?
        "NACA four-digit equations evaluated at 4097 cosine-spaced support points" :
        "C2 interpolation of source coordinates"
    payload = _write_mesh_artifacts(mesh,metadata,outdir,resolution)
    payload["artifact"] = joinpath(outdir,"mesh_full.msgpack")
    payload["preview_artifact"] = joinpath(outdir,"mesh_preview.msgpack")
    (; mesh=payload, metadata)
end

function _cavitation_point(case, path, flow, alpha, dir, progress, log)
    material, domain = cavitation_configuration(flow)
    gurney_enabled(flow) && gurney_check_request(flow, ["cavitation"])   # refused: the tunnel driver builds its own mesh
    domain=="underwater" && return _underwater_point(case,path,flow,alpha,dir,progress,log)
    if material == "air" || domain == "free_fluid"
        return _transonic_point(case,path,flow,alpha,dir,progress,log;species_backend=true)
    end
    _get(flow, "sigma", nothing) === nothing && throw(ArgumentError("cavitation requires flow.sigma"))
    sigma = _number(flow, "sigma", nothing; lo=0.01, hi=20.0)
    mach = _number(flow, "mach", 0.2; lo=0.02, hi=0.3)
    reynolds = _number(flow, "reynolds", 8e5; lo=1e3, hi=1e9)
    nwarm = _integer(flow, "cav_warm_steps", 300_000; lo=0, hi=10_000_000)
    duration = _number(flow, "cav_duration_ctu", 10.0; lo=0.001, hi=1_000.0)
    frames = _integer(flow, "cav_frames", 30; lo=2, hi=1000)
    plan = _production_tunnel_settings(path,flow,alpha)
    height = plan.channel.height
    cav_out = joinpath(dir, "cavitation_output")  # Driver claims this path exclusively.
    ispath(cav_out) && throw(ArgumentError("cavitation output directory already exists: $cav_out"))
    solver_path=path
    if plan.te>0
        solver_path=joinpath(dir,"production_tunnel_geometry.dat")
        open(solver_path,"w") do io
            println(io,"Airfoil Lab explicit finite trailing-edge geometry")
            for (x,y) in zip(plan.af.x,plan.af.y);println(io,x,' ',y);end
        end
    end
    env = _production_cavitation_environment(plan,solver_path,flow,alpha,cav_out,frames,reynolds,mach)
    realization=Dict{String,Any}("channel"=>_species_channel_metadata(plan.channel),
        "model_settings"=>plan.model,"mesh_settings"=>plan.mesh,"initialization"=>plan.model["cav_init"],
        "driver_environment"=>env,"source_geometry_sha256"=>bytes2hex(sha256(read(path))),
        "solver_geometry_sha256"=>bytes2hex(sha256(read(solver_path))),
        "production_driver_sha256"=>bytes2hex(sha256(read(joinpath(ROOT,"tools","cmesh_cav_movie.jl")))),
        "scope"=>"Validated environment passed to the production driver; computed geometry and model banners are retained separately in solver.log",
        "warm_steps"=>nwarm,"physical_duration_ctu"=>duration,"frames"=>frames,
        "cfd_steps_controls_production_driver"=>false,"parity_qualified"=>false)
    open(joinpath(dir,"cavitation_driver_input.toml"),"w") do io;TOML.print(io,realization);end
    _deliver_log(log,"Production tunnel: H/c=$(height), xi=$(plan.channel.xi_in)..$(plan.channel.xi_out), channel angle=$(alpha) deg; $(plan.model["cav_init"]) start; $(plan.model["cav_nut_density"]), Cprod=$(plan.model["cav_cprod"]), Cdest=$(plan.model["cav_cdest"]), trip=$(plan.model["cav_trip_x"]).")
    _deliver_log(log,"Production tunnel limits: $nwarm warm-up steps followed by $duration physical CTU and $frames frames; the transonic iteration cap does not control this driver.")
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    cmd = `$julia --project=$SRC --startup-file=no -O3 -t auto $(joinpath(ROOT, "tools", "cmesh_cav_movie.jl")) $sigma $duration $nwarm $frames`
    inherited = Dict{String,String}(k=>v for (k,v) in ENV
        if !startswith(k,"TOPO_") && !startswith(k,"CAV_MOVIE_"))
    ok = _log_process(setenv(cmd, merge(inherited, env)), joinpath(dir, "solver.log"), progress,
                      log, "cavitation", alpha)
    ok || throw(ErrorException("production cavitation driver failed: $(_solver_failure_summary(joinpath(dir,"solver.log"))); see solver.log"))
    _collect_cavitation_output(cav_out, dir, alpha, sigma, mach, reynolds, nwarm, duration, frames, height;realization)
end

function _collect_cavitation_output(cav_out, dir, alpha, sigma, mach, reynolds, nwarm, duration, frames, height;realization=Dict{String,Any}())
    csvs = filter(f -> startswith(basename(f), "cavmov_") && endswith(f, ".csv") &&
                       !occursin(r"_(inventory|interface|wallcp)\.csv$", f),
                  readdir(cav_out; join=true))
    length(csvs) == 1 || throw(ErrorException("expected one cavitation history CSV, found $(length(csvs))"))
    rows = _csv_rows(only(csvs))
    isempty(rows) && throw(ErrorException("cavitation history has no frames"))
    start = max(1, cld(length(rows), 2))
    tail = rows[start:end]
    clvals = Float64[_float(r["Cl"]) for r in tail]
    cdvals = Float64[_float(r["Cd"]) for r in tail]
    cp = Dict{String,Any}("kind"=>"upper surface stations", "upper"=>Any[])
    wallcp = replace(only(csvs), ".csv"=>"_wallcp.csv")
    if isfile(wallcp)
        for (k,v) in last(_csv_rows(wallcp))
            startswith(k, "cp_x") || continue
            x = tryparse(Float64, k[5:end]); c = _float(v)
            x === nothing || c === nothing || push!(cp["upper"], [x,c])
        end
        sort!(cp["upper"]; by=first)
    end
    frame_dirs = filter(isdir, readdir(cav_out; join=true))
    frames_dir = only(filter(f -> startswith(basename(f), "CAVMOV_FRAMES_"), frame_dirs))
    images = Dict{String,String}()
    for name in ("alpha", "ux", "cp", "makie")
        matches = sort(filter(f -> startswith(basename(f), name*"_") && endswith(f, ".png"),
                              readdir(frames_dir; join=true)))
        isempty(matches) || (images[name] = last(matches))
    end
    videos = Dict{String,String}()
    for name in ("alpha", "ux", "cp")
        matches = filter(f -> endswith(f, "_"*name*".mp4"), readdir(cav_out; join=true))
        isempty(matches) || (videos[name] = only(matches))
    end
    times = Dict(k=>Float64[] for k in ("t_ctu", "Cl", "Cd", "Vvap", "detached", "nclouds"))
    for row in rows, k in keys(times)
        value = _float(row[k]); value === nothing && throw(ArgumentError("nonfinite cavitation $k history"))
        push!(times[k], value)
    end
    realized_geometry=Dict{String,Any}();realized_operating=Dict{String,Any}();model_banners=String[]
    for line in eachline(joinpath(dir,"solver.log"))
        if occursin("REALISED GEOMETRY:",line)
            for m in eachmatch(r"\b(n_foil|n_normal|dn1|growth|smooth|channel_H|xi_in|xi_out|angle)=([+\-0-9.eE]+)",line)
                realized_geometry[m.captures[1]]=parse(Float64,m.captures[2])
            end
        end
        if occursin("CONSERVATIVE OPERATING POINT:",line) || occursin("REALISED OUTLET:",line)
            for m in eachmatch(r"\b(sigma_actual|saturation_density_ratio|p_out|sigma_exit)=([+\-0-9.eE]+)",line)
                realized_operating[m.captures[1]]=parse(Float64,m.captures[2])
            end
        end
        any(key->occursin(key,line),("SPECIES NUT DENSITY:","SPECIES SOURCE MODEL:","SPECIES SOURCE WEIGHT:","SPECIES TRIP:","CONSERVATIVE MODEL:","SST DIFFUSION CORRECTION:","CONSERVATIVE OPERATING POINT:","REALISED OUTLET:")) && push!(model_banners,line)
    end
    Dict{String,Any}(
        "backend"=>"cavitation", "alpha_deg"=>alpha, "status"=>"completed",
        "convergence"=>"time-accurate segment; no steady convergence claim",
        "validation"=>"geometry and operating-point dependent; inspect solver receipt",
        "cl"=>sum(clvals)/length(clvals), "cd"=>sum(cdvals)/length(cdvals),
        "cl_last"=>_float(rows[end]["Cl"]), "cd_last"=>_float(rows[end]["Cd"]),
        "coefficient_method"=>"mean of latter half of recorded frames",
        "cp"=>cp,
        "fields"=>Dict("kind"=>"rendered frames", "images"=>images, "videos"=>videos,
                         "history"=>times),
        "diagnostics"=>Dict("sigma"=>sigma, "mach"=>mach, "reynolds"=>reynolds,
                            "warm_steps"=>nwarm, "duration_ctu"=>duration,
                            "frames"=>frames, "channel_height"=>height,"driver_input"=>realization,
                            "realized_geometry"=>realized_geometry,"realized_operating_point"=>realized_operating,"realized_model_banners"=>model_banners),
        "artifacts"=>Dict("history_csv"=>only(csvs), "wallcp_csv"=>wallcp,
                          "driver_input"=>joinpath(dir,"cavitation_driver_input.toml"),"log"=>joinpath(dir, "solver.log")),
        "notes"=>["Conservative Species cavitation solver in a pressure-outlet channel.",
                    "Flow-field images show transported vapor fraction, streamwise velocity and pressure coefficient; the Makie panel adds density, Mach and vorticity when rendering succeeds."])
end

"""Run each requested incidence with the actual CMESH2D or conservative Species driver.

The caller owns a fresh job directory. A failed point retains its log and does not acquire invented
coefficients. The function is synchronous; call it in a Task for a responsive web server.
"""
function run_case(case::AbstractDict, backend::AbstractString, outdir::AbstractString;
                  progress::Function=(_)->nothing, log::Function=(_)->nothing,initial_warmdir=nothing)
    backend in ("transonic", "cavitation", "lowspeed", "lbm") || throw(ArgumentError("unsupported CFD backend: $backend"))
    initial_warmdir===nothing || backend=="lowspeed" || throw(ArgumentError("Checked initial flow is supported only by low-speed CFD"))
    path = _geometry(case)
    flow = _get(case, "flow", Dict())
    flow isa AbstractDict || throw(ArgumentError("flow must be an object"))
    alphas = _alpha_values(case)
    mkpath(outdir)
    points = Dict{String,Any}[]
    warmdir = initial_warmdir
    for (i, alpha) in enumerate(alphas)
        AirfoilJobControl.check_cancelled()
        dir = joinpath(outdir, backend, @sprintf("%03d_alpha_%+.3f", i, alpha))
        mkpath(dir)
        _deliver_log(log, @sprintf("Starting %s point %d/%d at alpha %.3f deg",
                                   backend, i, length(alphas), alpha))
        try
            point = backend == "lbm" ? _lbm_point(case,path,flow,alpha,dir,progress,log) :
                    backend == "lowspeed" ? _lowspeed_point(case,path,flow,alpha,dir,progress,log,warmdir;
                        warm_required=i==1 && initial_warmdir!==nothing) :
                    backend == "transonic" ? _transonic_point(case,path,flow,alpha,dir,progress,log,warmdir;
                        mesh_anchor=length(alphas)>1 ? first(alphas) : nothing) :
                                               _cavitation_point(case,path,flow,alpha,dir,progress,log)
            if i==1 && initial_warmdir!==nothing
                offered=abspath(joinpath(initial_warmdir,"state_checkpoint.jls"))
                used=get(get(point,"diagnostics",Dict()),"warm_start_source","")
                used isa AbstractString && !isempty(used) && normpath(abspath(used))==normpath(offered) ||
                    throw(ArgumentError("The worker refused the required checked initial flow; inspect solver.log for its compatibility reason. A cold fallback is not accepted."))
            end
            push!(points, point)
            backend in ("lowspeed", "transonic") && isfile(joinpath(dir, "state_checkpoint.jls")) && (warmdir = dir)
            progress(Dict("event"=>"point_complete", "backend"=>backend, "alpha_deg"=>alpha,
                          "point"=>points[end]))
            _deliver_log(log, @sprintf("Completed %s point %d/%d at alpha %.3f deg",
                                       backend, i, length(alphas), alpha))
        catch err
            AirfoilJobControl.check_cancelled()
            i==1 && initial_warmdir!==nothing && rethrow()
            push!(points, Dict{String,Any}("backend"=>backend, "alpha_deg"=>alpha,
                                          "status"=>"failed", "error"=>sprint(showerror, err),
                                          "artifacts"=>Dict("log"=>joinpath(dir,"solver.log"))))
            progress(Dict("event"=>"point_failed", "backend"=>backend, "alpha_deg"=>alpha,
                          "error"=>sprint(showerror, err), "point"=>points[end]))
            _deliver_log(log, @sprintf("Failed %s point %d/%d at alpha %.3f deg: %s",
                                       backend, i, length(alphas), alpha, sprint(showerror, err)))
        end
    end
    n_ok = count(p -> p["status"] == "completed", points)
    n_experimental = count(p -> p["status"] == "experimental_physical_state", points)
    Dict{String,Any}("backend"=>String(backend), "status"=>
        n_ok == length(points) ? "completed" : (n_ok + n_experimental == 0 ? "failed" : "partial"),
        "points"=>points)
end

"""Serialize grid nodes in j-major, i-fastest order for an inspectable mesh view.

The preview keeps the first sixteen wall-normal rows and topology junctions,
while the full artifact preserves every grid node. Neither changes the solve.
"""
function _mesh_payload(mesh, i_indices::Vector{Int}, j_indices::Vector{Int},
                       geometry_metadata::AbstractDict, resolution::AbstractString)
    ni = length(i_indices); nj = length(j_indices)
    Dict{String,Any}(
        "kind"=>"structured", "ni"=>ni, "nj"=>nj,
        "full_ni"=>mesh.NI+1, "full_nj"=>mesh.NJ+1,
        "i_indices"=>i_indices, "j_indices"=>j_indices,
        "stride_i"=>ni>1 ? maximum(diff(i_indices)) : 1,
        "stride_j"=>nj>1 ? maximum(diff(j_indices)) : 1,
        "x"=>[Float64(mesh.X[i,j]) for j in j_indices for i in i_indices],
        "y"=>[Float64(mesh.Y[i,j]) for j in j_indices for i in i_indices],
        "wall_i_start"=>mesh.NIW+1,
        "wall_i_end"=>mesh.NIW+mesh.NIF+1,
        "foil_surface_cells"=>mesh.NIF-2*get(geometry_metadata,"n_base",0),
        "wake_cells_per_side"=>mesh.NIW,
        "normal_cells"=>mesh.NJ,
        "base_faces_per_side"=>get(geometry_metadata,"n_base",0),
        "wall_dn1"=>get(geometry_metadata,"wall_dn1",mesh.dn1),
        # Carry the exact wall independently of the volume-grid display stride.
        # This prevents a coarse preview from drawing chords through the nose.
        "wall_x"=>Float64.(mesh.X[mesh.NIW+1:mesh.NIW+mesh.NIF+1,1]),
        "wall_y"=>Float64.(mesh.Y[mesh.NIW+1:mesh.NIW+mesh.NIF+1,1]),
        "quality"=>get(geometry_metadata,"quality_report",Dict{String,Any}()),
        "resolution"=>String(resolution),
        "node_order"=>"j-major, i-fastest")
end

function _write_mesh_artifacts(mesh, geometry_metadata::AbstractDict,
                               outdir::AbstractString, resolution::AbstractString)
    ni = mesh.NI+1; nj = mesh.NJ+1
    # Keep every tangential station. Decimating i used to make the displayed
    # nose and finite trailing-edge base disagree with the actual mesh wall.
    i_step = 1
    j_step = max(1, cld(nj, 81))
    critical_i = [1, ni, mesh.NIW+1, mesh.NIW+mesh.NIF+1,
                  argmin(@view mesh.X[:,1])]
    preview_i = sort!(unique(vcat(collect(1:i_step:ni), critical_i)))
    preview_j = sort!(unique(vcat(collect(1:min(16,nj)),
                           collect(1:j_step:nj), [nj])))
    preview = _mesh_payload(mesh, preview_i, preview_j, geometry_metadata, resolution)
    full = _mesh_payload(mesh, collect(1:ni), collect(1:nj),
                         geometry_metadata, resolution)
    write(joinpath(outdir,"mesh_full.msgpack"), AirfoilWire.encode(full, :msgpack))
    write(joinpath(outdir,"mesh_preview.msgpack"), AirfoilWire.encode(preview, :msgpack))
    preview
end

function _write_transonic_live_field(s, outdir::AbstractString, step::Int)
    ni=Int(s.NI);nj=Int(s.NJ);g=Int(s.g)
    xc=Array(s.xc);yc=Array(s.yc);rho=Array(s.rho);u=Array(s.ux);v=Array(s.uy);temp=Array(s.T);nut=Array(s.nut)
    gamma=(5+Float64(s.drot))/(3+Float64(s.drot));speed=hypot(Float64(s.uxi),Float64(s.uyi))
    q=.5Float64(s.rho_inf)*speed^2;pinf=(2/3)*Float64(s.rho_inf)*Float64(s.T_inf)
    function vort(i,j)
        im=max(1,i-1);ip=min(ni,i+1);jm=max(1,j-1);jp=min(nj,j+1)
        xi=xc[ip,j]-xc[im,j];yi=yc[ip,j]-yc[im,j];xj=xc[i,jp]-xc[i,jm];yj=yc[i,jp]-yc[i,jm]
        ui=Float64(u[ip+g,j+g]-u[im+g,j+g]);uj=Float64(u[i+g,jp+g]-u[i+g,jm+g])
        vi=Float64(v[ip+g,j+g]-v[im+g,j+g]);vj=Float64(v[i+g,jp+g]-v[i+g,jm+g]);jac=xi*yj-xj*yi
        abs(jac)<1e-20 ? 0. : ((vi*yj-vj*yi)-(-ui*xj+uj*xi))/jac
    end
    filename=@sprintf("live_field_%06d.msgpack",step)
    FullFieldArtifact.write_full_field((i,j)->begin
        pressure=Float64(rho[i+g,j+g])*(2/3)*Float64(temp[i+g,j+g])
        (xc[i,j],yc[i,j],hypot(u[i+g,j+g],v[i+g,j+g])/sqrt(gamma*(2/3)*temp[i+g,j+g]),
         (pressure-pinf)/q,pressure,rho[i+g,j+g],u[i+g,j+g],v[i+g,j+g],temp[i+g,j+g],vort(i,j),nut[i+g,j+g])
    end,joinpath(outdir,filename),ni,nj,joinpath(outdir,"mesh_full.msgpack"),step)
    println("FIELD_SNAPSHOT,",filename,",",step);flush(stdout)
end

function _transonic_worker(cfg_path::AbstractString)
    ENV["GKSwstype"] = "100"
    cfg = TOML.parsefile(cfg_path)
    time_mode = String(get(cfg,"time_mode","steady"))
    time_mode in ("steady","physical") || throw(ArgumentError("time_mode must be steady or physical"))
    physical = time_mode == "physical"
    gpu = Bool(cfg["gpu"])
    method=String(get(cfg,"rans_method",get(cfg,"irsj",false) ? "irsj" : "standard"))
    method in RANSAcceleration.METHODS || throw(ArgumentError("Unknown RANS method: $method"))
    implicit=RANSAcceleration.implicit_method(method)
    implicit && physical && throw(ArgumentError("Implicit research methods require steady mode"))
    # Physical mode uses the same KA integrator on CPU and GPU. The older
    # plain CPU solver remains the unchanged default steady CPU path.
    ka = gpu || physical || implicit
    duration_ctu = Float64(get(cfg,"duration_ctu",100.0))
    burnin_ctu = Float64(get(cfg,"burnin_ctu",50.0))
    block_ctu = Float64(get(cfg,"block_ctu",5.0))
    physical_history_stride = Int(get(cfg,"history_stride",400))
    force_sample_dt_ctu = Float64(get(cfg,"force_sample_dt_ctu",0.1))
    if physical
        isfinite(duration_ctu) && duration_ctu>0 && isfinite(burnin_ctu) &&
            0<=burnin_ctu<duration_ctu && isfinite(block_ctu) && block_ctu>0 ||
            throw(ArgumentError("physical duration, burn-in and block duration are invalid"))
        physical_history_stride>0 && isfinite(force_sample_dt_ctu) && force_sample_dt_ctu>0 ||
            throw(ArgumentError("physical force sampling settings are invalid"))
        !Bool(get(cfg,"rk_weight_closure",false)) ||
            throw(ArgumentError("physical mode uses rk_incremental; rk_weight_closure must be false"))
        !Bool(get(cfg,"sst_nonorth_correction",false)) ||
            throw(ArgumentError("physical mode refuses nonorthogonal SST because it clips its own timestep"))
        Int(get(cfg,"mass_probe_stride",0))==0 ||
            throw(ArgumentError("the SFD mass-stage probe is unavailable in physical mode"))
    end
    cx, cy, cz, w, _, _ = build_d3q39_static()
    af = haskey(cfg,"naca4_code") ? CMeshGen.naca4_points(cfg["naca4_code"]) :
                                    CMeshGen.read_airfoil_dat(cfg["geometry"])
    re = cfg["reynolds"]
    cf = 0.026*re^(-1/7)
    # Estimate first-cell y+ from u_tau/U ~= sqrt(Cf/2). Use y+ ~= 0.7;
    # the existing validated NACA0015 polar used y+ ~= 0.8. The previous
    # y+ ~= 1.1 mesh was marginal for wall-resolved SST at low Reynolds number.
    dn1 = 0.7/(re*sqrt(cf/2))
    mesh_mode = get(cfg, "mesh_mode", "legacy_sharp_cmesh")
    if mesh_mode in ("cavitation_aplus_farfield", "aplus_nested_refined", "aplus_uniform_quality", "aplus_uniform_fixed")
        built = _build_transonic_mesh(af;
            n_foil=cfg["n_foil"], n_wake=cfg["n_wake"],
            n_normal=cfg["n_normal"], dn1=dn1,
            alpha_deg=cfg["alpha_deg"],
            trip_upper=cfg["trip_upper"], trip_lower=cfg["trip_lower"],
            n_base=get(cfg, "n_base", 4),
            cfd_te_thickness=get(cfg,"cfd_te_thickness",0.0),
            mesh_refinement=_strict_aplus_profile(mesh_mode) ? Symbol(mesh_mode) :
                mesh_mode == "aplus_nested_refined" ? :aplus_nested_refined : :none,
            normal_collar=!_strict_aplus_profile(mesh_mode),shock_layer=!_strict_aplus_profile(mesh_mode),
            wake_first_spacing_growth=get(cfg,"wake_first_spacing_growth",1.0),
            wake_first_spacing_max=get(cfg,"wake_first_spacing_max",0.01),
            wake_first_spacing_delay_cells=get(cfg,"wake_first_spacing_delay_cells",0),
            farfield_angle_deg=get(cfg, "cfd_farfield_angle_deg", mesh_mode=="aplus_uniform_fixed" ? 0. : cfg["alpha_deg"]),
            stall_layer=Bool(get(cfg, "stall_layer", false)),
            le_ds_scale=Float64(get(cfg, "le_ds_scale", 1.0)))
        mesh = built.mesh; met = built.metrics
        geometry_metadata = built.metadata
        foil_indices = built.foil_indices
        surface_indices = built.surface_indices
        estimated_yplus = re*Float64(geometry_metadata["wall_dn1"])*sqrt(cf/2)
    elseif mesh_mode == "legacy_sharp_cmesh"
        sharp = _sharp_nose_mesh(af; n_foil=cfg["n_foil"], n_wake=cfg["n_wake"],
            n_normal=cfg["n_normal"], dn1=dn1,
            le_ds_scale=Float64(get(cfg,"le_ds_scale",1.0)))
        mesh = sharp.mesh
        quality = CMeshGen.mesh_quality(mesh; reynolds=re)
        quality.nfold == 0 && quality.area_min > 0 ||
            error("CMESH2D mesh gate failed: folds=$(quality.nfold), min area=$(quality.area_min)")
        met = CMeshMetrics.build_metrics(mesh;
            trip_upper=cfg["trip_upper"], trip_lower=cfg["trip_lower"])
        foil_indices = met.NIW+1:met.NIW+met.NIF
        surface_indices = foil_indices
        source_af = CMeshGen.read_airfoil_dat(get(cfg,"source_geometry",cfg["geometry"]))
        source_gap = hypot(source_af.x[1]-source_af.x[end],source_af.y[1]-source_af.y[end])
        cfd_gap = hypot(af.x[1]-af.x[end],af.y[1]-af.y[end])
        geometry_metadata = Dict{String,Any}(
            "source_te_gap"=>source_gap,"cfd_te_gap"=>cfd_gap,
            "transform"=>source_gap>1e-8 ? "aft taper to sharp trailing edge" : "none",
            "mesh_family"=>"CMeshGen legacy sharp-edge C-grid",
            "mesh_topology"=>"sharp", "n_base"=>0,
            "surface_distribution"=>"curvature",
            "leading_edge_target_spacing"=>sharp.le_ds,
            "leading_edge_max_curvature"=>sharp.le_kappa,
            "wall_dn1"=>dn1, "outer_boundary_condition"=>"characteristic farfield",
            "quality_report"=>_transonic_quality_report(mesh))
        estimated_yplus = re*dn1*sqrt(cf/2)
    else
        error("unknown transonic mesh mode $mesh_mode")
    end
    # Fail impossible export dimensions before allocating GPU state or marching.
    FullFieldArtifact.field_layout(mesh.NI,mesh.NJ)
    geometry_metadata["geometry_support"] = haskey(cfg,"naca4_code") ?
        "NACA four-digit equations evaluated at 4097 cosine-spaced support points" :
        "C2 interpolation of source coordinates"
    @printf("CMESH2D mesh generated: mode=%s, %d foil, %d wake, %d normal cells; dn1=%.3e, estimated y+=%.2f, base faces/side=%d\n",
            mesh_mode, length(surface_indices), mesh.NIW, mesh.NJ,
            geometry_metadata["wall_dn1"], estimated_yplus, geometry_metadata["n_base"])
    flush(stdout)
    _write_mesh_artifacts(mesh, geometry_metadata, cfg["outdir"],
                          get(cfg,"resolution","custom"))
    println("MESH_READY,mesh_preview.msgpack")
    flush(stdout)
    equation_probe = implicit || Bool(get(cfg,"equation_probe",false))
    equation_probe && !ka &&
        throw(ArgumentError("equation residual probe requires the KA worker"))
    rk_weight_closure = Bool(get(cfg,"rk_weight_closure",false))
    sst_wall_metric_correction = Bool(get(cfg,"sst_wall_metric_correction",true))
    sst_nonorth_correction = Bool(get(cfg,"sst_nonorth_correction",false))
    sst_nonorth_correction && !ka &&
        throw(ArgumentError("sst_nonorth_correction currently requires the KA GPU worker"))
    refresh_flow_bc_before_sst = Bool(get(cfg,"refresh_flow_bc_before_sst",false))
    refresh_flow_bc_before_sst && !ka &&
        throw(ArgumentError("refresh_flow_bc_before_sst currently requires the KA GPU worker"))
    characteristic_i_outlet = Bool(get(cfg,"characteristic_i_outlet",false))
    characteristic_i_outlet && !ka &&
        throw(ArgumentError("characteristic_i_outlet currently requires the KA GPU worker"))
    rk_weight_closure && !ka &&
        throw(ArgumentError("rk_weight_closure currently requires the KA GPU worker"))
    mass_probe_stride = Int(get(cfg,"mass_probe_stride",0))
    mass_probe_start = Int(get(cfg,"mass_probe_start",min(cfg["sfd_start"],cfg["steps"])))
    mass_probe_stride >= 0 && 1 <= mass_probe_start <= cfg["steps"] ||
        throw(ArgumentError("invalid mass-stage probe stride or start"))
    mass_probe_stride == 0 || ka ||
        throw(ArgumentError("mass-stage probe requires the KA GPU worker"))
    sfd_storage = get(cfg,"sfd_storage_type","float32")
    sfd_storage in ("float32","float64") ||
        throw(ArgumentError("worker sfd_storage_type must be float32 or float64"))
    ka || sfd_storage == "float32" ||
        throw(ArgumentError("Float64 SFD filter storage currently requires the KernelAbstractions GPU worker"))
    viscosity_model = get(cfg,"viscosity_model","constant_kinematic")
    viscosity_model in ("constant_kinematic","sutherland","constant_dynamic") ||
        throw(ArgumentError("worker viscosity_model must be constant_kinematic, sutherland, or constant_dynamic"))
    ka || viscosity_model == "constant_kinematic" ||
        throw(ArgumentError("dynamic air viscosity currently requires the KernelAbstractions GPU worker"))
    if gpu
        CUDA.functional() || error("CUDA was requested but is unavailable")
    end
    backend = gpu ? CUDA.CUDABackend() : KernelAbstractions.CPU()
    s = ka ? CMeshSolverKA.build_sol_ka(backend, met; drot=2.0f0, cxv=cx, cyv=cy, czv=cz, wv=w,
                                        sfd_storage_type=sfd_storage == "float64" ? Float64 : Float32) :
              CMeshSolver.build_sol(backend, met; drot=2.0f0, cxv=cx, cyv=cy, czv=cz, wv=w)
    println("CMESH2D solver initialized on ", gpu ? "GPU" : "CPU")
    println("CMESH2D SFD filter storage: ", sfd_storage)
    flush(stdout)
    s.wall_flux = true
    if ka
        s.characteristic_i_outlet = Int32(characteristic_i_outlet)
        CMeshSolverKA.GLOBAL_DT[] = physical ? 1 : 0
    end
    if physical || implicit
        s.sfd=Int32(0);s.irsj=Int32(0);s.irs_ac=Int32(0);s.acou=Int32(0)
        s.sponge_sigma=0.0f0
        println(physical ? "CMESH2D physical clock: one global explicit timestep; no SFD, residual smoothing, acoustic filter or sponge" :
            "CMESH2D experimental $method: raw undamped equations, explicit startup followed by the selected implicit method")
    end
    s.recon = Int32(1)    # WENO5-Z with sensor blend; production Case 9 path.
    s.fullvisc = Int32(1) # Full stress tensor; required by the validated clean-foil polar.
    if !implicit && Bool(get(cfg,"irsj",false))
        # The solver's existing IRS-J: the pseudo-time step follows the streamwise restriction and
        # per-column tridiagonal solves absorb the wall-normal stiffness. It smooths the residual,
        # so a converged steady state is unchanged; physical time keeps its global explicit step.
        physical && throw(ArgumentError("cfd_irsj applies to steady pseudo-time only"))
        s.irsj = Int32(1)
        println("CMESH2D j-line implicit residual smoothing (IRS-J): on (steady pseudo-time)")
    end
    @printf("CMESH2D numerics: WENO5-Z, full viscous stress, CFL=%.3f, clock=%s, SFD=%s\n",
            cfg["cfl"], time_mode, physical ? "off" : "at n=$(cfg["sfd_start"])")
    flush(stdout)
    if ka
        CMeshSolverKA.set_freestream_ka!(s, cfg["mach"], cfg["alpha_deg"]; pg_correction=!physical)
    else
        CMeshSolver.set_freestream!(s, cfg["mach"], cfg["alpha_deg"]; pg_correction=true)
    end
    speed = hypot(Float64(s.uxi), Float64(s.uyi))
    chord = maximum(af.x)-minimum(af.x)
    mass_flux_scale = Float64(s.rho_inf)*speed*(maximum(af.x)-minimum(af.x))
    isfinite(mass_flux_scale) && mass_flux_scale > 0 ||
        error("freestream mass-flux scale must be finite and positive")
    equation_scales = equation_probe ? EquationResidualProbe.reference_scales(
        rho_inf=s.rho_inf, ux_inf=s.uxi, uy_inf=s.uyi,
        temp_inf=s.T_inf, eint=s.eint,
        chord=maximum(af.x)-minimum(af.x)) : nothing
    s.nu_lam = Float32(speed/re)
    if viscosity_model == "sutherland"
        CMeshSolverKA.set_air_sutherland_ka!(s,cfg["freestream_temperature_k"])
        @printf("CMESH2D viscosity: Sutherland air, T_inf=%.2f K, S=110.4 K\n",
                cfg["freestream_temperature_k"])
    elseif viscosity_model == "constant_dynamic"
        CMeshSolverKA.set_air_constant_dynamic_ka!(s)
        println("CMESH2D viscosity: constant molecular dynamic; no Kelvin temperature used")
    else
        println("CMESH2D viscosity: constant kinematic (legacy)")
    end
    flush(stdout)
    ka ? CMeshSolverKA.set_sst_ka!(s; tu_inf=0.001, mu_t_ratio=0.1) :
          CMeshSolver.set_sst!(s; tu_inf=0.001, mu_t_ratio=0.1)
    warm_source = physical ? "" : _rans_load_restart!(cfg, joinpath(cfg["outdir"], "mesh_full.msgpack"),
        String(mesh_mode), [(name="C", solver=s)])
    if !isempty(warm_source)
        ka ? CMeshSolverKA.applyBC_ka!(s, Int32(2)) : CMeshSolver.applyBC!(s; wall=:noslip)
    end
    implicit_runtime=implicit ? _implicit_runtime(CMeshSolverKA,s,cfg) : nothing
    implicit_warmup=implicit && isempty(warm_source) ? Int(get(cfg,"implicit_warmup_steps",1000)) : 0
    implicit_stopped=false;implicit_accepted=0;implicit_failure=""
    implicit_started=time()
    implicit_io=implicit ? open(joinpath(cfg["outdir"],"implicit_history.csv"),"w") : nothing
    if implicit
        println(get(cfg,"implicit_maturity","Research method: full aerodynamic convergence and acceleration remain unverified."))
        println("Implicit warm-up: $implicit_warmup explicit iterations; output fields remain available if the nonlinear solve stops.")
        println(implicit_io,"iteration,wall_seconds,accepted,step_kind,flow_merit_before,flow_merit_after,pseudo_cfl,linear_iterations,linear_relative,attempts,recovery_steps,recovery_total");flush(implicit_io)
    end
    loads_now() = ka ? CMeshSolverKA.loads_ka(s) : CMeshSolver.loads(s)
    function moment_now()
        g=s.g;indices=collect(foil_indices);ir=(first(indices)+g):(last(indices)+g)
        rw,uw,vw,tw=map(a->Array(view(a,ir,g+1)),(s.rho,s.ux,s.uy,s.T))
        sx=ka ? s.Sjx1_h : s.Sjx[:,1];sy=ka ? s.Sjy1_h : s.Sjy[:,1]
        vv=ka ? s.vol1_h : s.vol[:,1]
        q=.5Float64(s.rho_inf)*speed^2;moment=0.0
        for (k,i) in enumerate(indices)
            sxi,syi=Float64(sx[i]),Float64(sy[i]);sf=hypot(sxi,syi)
            d1=.5Float64(vv[i])/sf+1e-30
            mu=(viscosity_model!="constant_kinematic" ? Float64(s.rho_inf)*Float64(s.nu_lam)*
                Float64(CMeshSolverKA._mufac(tw[k],s.suth)) : Float64(rw[k])*Float64(s.nu_lam))*sf/d1
            pressure=Float64(rw[k])*(2/3)*Float64(tw[k])
            if s.fullvisc==Int32(1)
                wall_module=ka ? CMeshSolverKA : CMeshSolver
                viscosity_model=="constant_kinematic" && (mu *= Float64(wall_module._mufac(tw[k],s.suth)))
                tx,ty=wall_module._wall_stokes_traction(Float64(uw[k]),Float64(vw[k]),sxi/sf,syi/sf,mu)
                fx=-pressure*sxi+tx;fy=-pressure*syi+ty
            else
                fx=-pressure*sxi+mu*Float64(uw[k]);fy=-pressure*syi+mu*Float64(vw[k])
            end
            moment+=_aerodynamic_cm(.5(mesh.X[i,1]+mesh.X[i+1,1]),
                .5(mesh.Y[i,1]+mesh.Y[i+1,1]),fx,fy,q)
        end
        moment
    end
    function residual_now()
        if ka
            scaled_max, scaled_l2, _, _, _ = CMeshSolverKA.residual_report_ka(s)
            return (Float64(CMeshSolverKA.residual_linf_ka(s)),
                    Float64(scaled_max), Float64(scaled_l2))
        end
        raw = maximum(abs, s.dc_r)
        scaled = abs.(s.dc_r) ./ s.vol
        (Float64(raw), Float64(maximum(scaled)),
         sqrt(sum(abs2, scaled)/length(scaled)))
    end
    history = NamedTuple{(:iteration,:residual_linf,:residual_scaled,:residual_l2,:cl,:cd,:cm,:mass_net_frac),
                         Tuple{Int,Float64,Float64,Float64,Float64,Float64,Float64,
                               Union{Nothing,Float64}}}[]
    historyfile = joinpath(cfg["outdir"], "iteration_history.csv")
    history_io = open(historyfile, "w")
    println(history_io, "iteration,residual_linf,residual_scaled,residual_l2,cl,cd,cm,mass_net_frac")
    flush(history_io)
    physical_history=Dict{String,Any}[]
    physical_io=physical ? open(joinpath(cfg["outdir"],"physical_force_history.csv"),"w") : nothing
    elapsed_ctu=0.0;next_force_sample_ctu=force_sample_dt_ctu
    dt_ctu_min=Inf;dt_ctu_max=0.0;physical_duration_reached=false
    if physical
        cl0,cd0=loads_now();cm0=moment_now()
        push!(physical_history,Dict{String,Any}("iteration"=>0,"t_ctu"=>0.0,"cl"=>cl0,"cd"=>cd0,"cm"=>cm0))
        println(physical_io,"iteration,t_ctu,applied_dt_ctu,cl,cd,cm")
        @printf(physical_io,"0,0,0,%.12g,%.12g,%.12g\n",cl0,cd0,cm0);flush(physical_io)
    end
    equation_file = joinpath(cfg["outdir"],"equation_residual_history.csv")
    equation_io = equation_probe ? open(equation_file,"w") : nothing
    equation_vol = equation_probe ? Array(s.vol) : nothing
    implicit_rows=Dict{String,Any}[]
    implicit_metrics_io=implicit ? open(joinpath(cfg["outdir"],"steady_assessment_history.csv"),"w") : nothing
    implicit_metric_names=_implicit_metric_names()
    implicit && (println(implicit_metrics_io,join(implicit_metric_names,','));flush(implicit_metrics_io))
    function implicit_assessment()
        RANSConvergence.assess(implicit_rows;
            residual_limits=Dict(k=>.02 for k in ("mass_l2","momentum_x_l2","momentum_y_l2","energy_l2")),
            update_limits=Dict("turb_k_update_defect"=>1e-4,"turb_omega_update_defect"=>1e-4),
            config=RANSConvergence.Config(window_samples=20,mass_limit=mass_net_limit),
            phase_start=implicit_warmup+1,equation_coverage_complete=false,
            scope="Four convection-scaled flow RMS errors, safe-clock turbulence update defects, raw SST transport residuals, prescribed-wall omega constraint, mass and force history are recorded. Qualified raw-SST stopping limits are not yet configured; the existing experimental limits remain provisional.")
    end
    if equation_io !== nothing
        println(equation_io,EquationResidualProbe.history_header())
        flush(equation_io)
    end
    mass_probe_file = joinpath(cfg["outdir"],"mass_stage_probe.csv")
    mass_probe_io = mass_probe_stride > 0 ? open(mass_probe_file,"w") : nothing
    probe_vol = mass_probe_stride > 0 ? Float64.(Array(s.vol)) : nothing
    if mass_probe_io !== nothing
        println(mass_probe_io,
            "iteration,post_rk_max,post_rk_l2,post_rk_i,post_rk_j," *
            "post_sfd_max,post_sfd_l2,post_sfd_i,post_sfd_j," *
            "post_monitor_max,post_monitor_l2,post_monitor_i,post_monitor_j," *
            "sfd_source_max,sfd_source_l2,sfd_source_integral_over_freestream_flux," *
            "sfd_source_at_post_monitor_peak,post_monitor_signed_residual_at_peak," *
            "source_to_raw_at_peak")
        flush(mass_probe_io)
    end
    mass_net_limit = Float64(get(cfg,"mass_net_limit",0.001))
    isfinite(mass_net_limit) && mass_net_limit > 0 ||
        throw(ArgumentError("worker mass_net_limit must be finite and positive"))
    diagnostic_stride=implicit ? min(20,max(1,Int(get(cfg,"display_every",20)))) : max(1,Int(get(cfg,"display_every",400)))
    implicit_window=20diagnostic_stride
    monitor = CFDConvergence.Monitor(implicit ? CFDConvergence.Config(
        sfd_start=implicit_warmup,max_steps=cfg["steps"],window_iterations=implicit_window,
        min_post_sfd=2implicit_window,min_samples_per_window=12,
        confirmation_iterations=10diagnostic_stride,confirmation_checks=8,
        residual_scaled_limit=cfg["mass_ceiling"],mass_net_limit=mass_net_limit,
        stop_on_stall=false) : CFDConvergence.Config(
        sfd_start=physical ? 0 : cfg["sfd_start"], max_steps=cfg["steps"],
        residual_scaled_limit=cfg["mass_ceiling"],
        mass_net_limit=mass_net_limit,stop_on_stall=!physical && cfg["auto_stop"]))
    last_status_time = time()
    release_start=0
    release_enabled=!physical && !implicit && ka && get(cfg,"sfd_release",true)
    profile_stages=gpu && Bool(get(cfg,"profile_stages",false))
    profile_flow_seconds=0.0;profile_sst_seconds=0.0;profile_samples=0
    try
        snapshots=FullFieldArtifact.FieldSnapshots.Schedule(get(cfg,"snapshot_interval_seconds",600.))
        println("Field snapshots: every ",snapshots.interval/60," wall-clock minutes (0 = disabled); final fields are saved separately.")
        for n in 1:cfg["steps"]
            profile_flow_start=profile_stages ? time() : 0.0
            probe_now = mass_probe_stride > 0 &&
                        (n == cfg["steps"] ||
                         (n >= mass_probe_start &&
                          (n-mass_probe_start) % mass_probe_stride == 0))
            if !physical && !implicit && n == cfg["sfd_start"]
                ka ? CMeshSolverKA.sfd_seed_ka!(s) : CMeshSolver.sfd_seed!(s)
                s.sfd = Int32(1)
                s.sfd_chi = 0.005f0
                s.sfd_alpha = 5.0f-5
                @printf("CMESH2D selective frequency damping engaged at n=%d\n", n)
                flush(stdout)
            end
            if release_start>0
                fraction=clamp((n-release_start)/20000,0.0,1.0)
                s.sfd_chi=Float32(0.005*(1-fraction)^2)
                if fraction>=1 && s.sfd==1
                    s.sfd=Int32(0)
                    println("CMESH2D SFD released at n=$n; evaluating the undamped equations with fresh convergence windows")
                    flush(stdout)
                end
            end
            if physical
                remaining_dt=(duration_ctu-elapsed_ctu)*chord/speed
                dt_cap=Float32(remaining_dt)
                Float64(dt_cap)>remaining_dt && (dt_cap=prevfloat(dt_cap))
                isfinite(dt_cap) && dt_cap>0 || error("physical timestep cap cannot advance the clock")
                CMeshSolverKA.step_ka!(s,Float32(cfg["cfl"]);wallmode=Int32(2),
                    max_dt=dt_cap,rk_incremental=true)
                # SST reads the final committed flow at the same applied dt.
                CMeshSolverKA.applyBC_ka!(s,Int32(2))
                if profile_stages
                    CUDA.synchronize()
                    profile_flow_seconds+=time()-profile_flow_start
                end
                profile_sst_start=profile_stages ? time() : 0.0
                CMeshSolverKA.sst_step_ka!(s;corrected_wall_metric=sst_wall_metric_correction)
                if profile_stages
                    CUDA.synchronize()
                    profile_sst_seconds+=time()-profile_sst_start
                    profile_samples+=1
                end
                dt_ctu=Float64(s.last_global_dt)*speed/chord
                isfinite(dt_ctu) && dt_ctu>0 && elapsed_ctu+dt_ctu>elapsed_ctu ||
                    error("the accepted physical timestep did not advance the clock")
                elapsed_ctu+=dt_ctu
                dt_ctu_min=min(dt_ctu_min,dt_ctu);dt_ctu_max=max(dt_ctu_max,dt_ctu)
                physical_duration_reached=elapsed_ctu>=duration_ctu*(1-1e-7)
                if n==1
                    minimum(s.dtf)==maximum(s.dtf)==s.last_global_dt ||
                        error("physical mode did not apply one uniform global timestep")
                end
            elseif implicit && n>implicit_warmup
                update=_implicit_step!(CMeshSolverKA,s,implicit_runtime,cfg)
                _implicit_log!(implicit_io,n,update,time()-implicit_started)
                if update.accepted
                    implicit_accepted+=1
                else
                    implicit_stopped=true
                    implicit_failure="The experimental $method step was rejected after retries. The last accepted state and all diagnostic results are saved; choose a different method or continue from a saved study."
                    println(implicit_failure);flush(stdout)
                end
            elseif ka
                if probe_now
                    probe_rk = Ref{Any}(nothing)
                    probe_sfd = Ref{Any}(nothing)
                    probe_before = Ref{Any}(nothing)
                    probe_after = Ref{Any}(nothing)
                    probe_dtf = Ref{Any}(nothing)
                    after_rk = sol -> begin
                        probe_rk[] = _mass_stage_state(sol,probe_vol)
                        probe_before[] = Array(sol.rho)
                        probe_dtf[] = Array(sol.dtf)
                    end
                    after_sfd = sol -> begin
                        probe_after[] = Array(sol.rho)
                        probe_sfd[] = _mass_stage_state(sol,probe_vol)
                    end
                    CMeshSolverKA.step_ka!(s, Float32(cfg["cfl"]); wallmode=Int32(2),
                                           rk_weight_closure=rk_weight_closure,
                                           probe_after_rk=after_rk,
                                           probe_after_sfd=after_sfd)
                else
                    CMeshSolverKA.step_ka!(s, Float32(cfg["cfl"]); wallmode=Int32(2),
                                           rk_weight_closure=rk_weight_closure)
                end
                # step_ka! leaves legacy-mode flow ghosts at the pre-final-RK/SFD state.
                # The SST stencil below reads those ghosts at j=1 and the outer edge.
                if refresh_flow_bc_before_sst
                    CMeshSolverKA.applyBC_ka!(s,Int32(2))
                end
                if profile_stages
                    CUDA.synchronize()
                    profile_flow_seconds+=time()-profile_flow_start
                end
                profile_sst_start=profile_stages ? time() : 0.0
                CMeshSolverKA.sst_step_ka!(s;
                    corrected_wall_metric=sst_wall_metric_correction,
                    nonorth_diffusion=sst_nonorth_correction)
                if profile_stages
                    CUDA.synchronize()
                    profile_sst_seconds+=time()-profile_sst_start
                    profile_samples+=1
                end
            else
                CMeshSolver.step!(s, Float32(cfg["cfl"]); wall=:noslip)
                CMeshSolver.sst_step!(s;
                    corrected_wall_metric=sst_wall_metric_correction)
            end
            if !physical && n % 400 == 0
                cl, _ = loads_now()
                s.gam = 0.9f0*s.gam + 0.1f0*Float32(0.5*speed*cl)
            end
            display_every=Int(get(cfg,"display_every",0))
            FullFieldArtifact.FieldSnapshots.save_if_due!(snapshots) do
                _write_transonic_live_field(s,String(cfg["outdir"]),n)
            end
            history_due=physical ? (n==1 || n%physical_history_stride==0 ||
                elapsed_ctu>=next_force_sample_ctu || physical_duration_reached || n==cfg["steps"]) :
                (n==1 || n%(implicit ? diagnostic_stride : display_every>0 ? display_every : 400)==0 || n==cfg["steps"] || probe_now || implicit_stopped)
            if history_due
                # step_ka! leaves raw fluxes/divergence from its last RK stage,
                # before the final primitive update and SFD. Sample the actual
                # stored state for live diagnostics, stopping, and mass balance.
                ka && CMeshSolverKA.refresh_raw_residual_ka!(s; wallmode=Int32(2))
                cl, cd = loads_now()
                cm_sample=moment_now()
                raw, scaled, l2 = residual_now()
                # The GPU path has just refreshed the final post-step raw
                # continuity divergence. Accumulate signed cell divergence in
                # Float64, before taking an absolute value in the stop guard.
                # The plain CPU path lacks that refresh and reports unavailable.
                mass_dc_sample = ka ? Array(s.dc_r) : nothing
                mass_net = ka ? sum(Float64,mass_dc_sample)/mass_flux_scale : nothing
                if equation_io !== nothing
                    equation_report = EquationResidualProbe.assess((
                        mass=mass_dc_sample, mom_x=Array(s.dc_mx),
                        mom_y=Array(s.dc_my), energy=Array(s.dc_e)),
                        equation_vol,equation_scales)
                    abs(equation_report.mass.signed_flux_fraction-mass_net) <= 1e-6 ||
                        error("equation probe disagrees with continuity mass net")
                    println(equation_io,EquationResidualProbe.history_row(n,equation_report))
                    flush(equation_io)
                end
                if probe_now
                    # This is the same state as the iteration history: after
                    # SST and any n%400 far-field circulation feedback.
                    post_monitor = _mass_stage_state(s,probe_vol;refresh=false)
                    source = _sfd_mass_ledger(probe_before[],probe_after[],
                        probe_dtf[],probe_vol,s.g,s.rho_inf,speed,
                        maximum(af.x)-minimum(af.x),post_monitor.i,post_monitor.j)
                    ratio = post_monitor.signed == 0.0 ? NaN :
                            source.at_peak/post_monitor.signed
                    rk = probe_rk[]; sf = probe_sfd[]
                    @printf(mass_probe_io,
                        "%d,%.12g,%.12g,%d,%d,%.12g,%.12g,%d,%d,%.12g,%.12g,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                        n,rk.maximum,rk.rms,rk.i,rk.j,
                        sf.maximum,sf.rms,sf.i,sf.j,
                        post_monitor.maximum,post_monitor.rms,post_monitor.i,post_monitor.j,
                        source.maximum,source.rms,source.integral,
                        source.at_peak,post_monitor.signed,ratio)
                    flush(mass_probe_io)
                end
                all(isfinite, (cl, cd, raw, scaled, l2)) &&
                    (mass_net === nothing || isfinite(mass_net)) ||
                    error("CMESH2D developed a nonfinite load or residual at iteration $n")
                push!(history, (iteration=n, residual_linf=raw, residual_scaled=scaled,
                                residual_l2=l2, cl=cl, cd=cd, cm=cm_sample,
                                mass_net_frac=mass_net))
                @printf(history_io, "%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,",
                        n, raw, scaled, l2, cl, cd,cm_sample)
                if mass_net === nothing
                    println(history_io)
                else
                    @printf(history_io, "%.12g\n", mass_net)
                end
                flush(history_io)
                if implicit
                    metrics=CMeshSolverKA.steady_airfoil_diagnostics_ka(s;
                        corrected_wall_metric=sst_wall_metric_correction,nonorth_diffusion=sst_nonorth_correction)
                    values=_implicit_metric_values(n,metrics,mass_net,cl,cd,cm_sample)
                    push!(implicit_rows,Dict{String,Any}(zip(implicit_metric_names,values)))
                    println(implicit_metrics_io,join(values,','));flush(implicit_metrics_io)
                end
                if physical
                    cm_sample=moment_now()
                    isfinite(cm_sample) || error("physical moment history became nonfinite")
                    push!(physical_history,Dict{String,Any}("iteration"=>n,"t_ctu"=>elapsed_ctu,
                        "cl"=>cl,"cd"=>cd,"cm"=>cm_sample))
                    @printf(physical_io,"%d,%.16g,%.16g,%.12g,%.12g,%.12g\n",
                        n,elapsed_ctu,Float64(s.last_global_dt)*speed/chord,cl,cd,cm_sample)
                    flush(physical_io)
                    next_force_sample_ctu=elapsed_ctu+force_sample_dt_ctu
                    if n==1 || physical_duration_reached || time()-last_status_time>=5
                        @printf("PHYSICAL_TIME,%d,%.12g,%.12g,%.12g\n",n,elapsed_ctu,duration_ctu,
                            Float64(s.last_global_dt)*speed/chord);flush(stdout)
                    end
                end
                if mass_net === nothing
                    @printf("ITERATION,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                            n, cfg["steps"], scaled, raw, l2, cl, cd)
                else
                    @printf("ITERATION,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                            n, cfg["steps"], scaled, raw, l2, cl, cd, mass_net)
                end
                flush(stdout)
                if n == 1 || n % 2000 == 0 || time() - last_status_time >= 5
                    if mass_net === nothing
                        @printf("n=%d/%d Cl=%+.5f Cd=%+.6f mass_Rscaled=%.3e mass_net=unavailable\n",
                                n, cfg["steps"], cl, cd, scaled)
                    else
                        @printf("n=%d/%d Cl=%+.5f Cd=%+.6f mass_Rscaled=%.3e mass_net=%+.3e\n",
                                n, cfg["steps"], cl, cd, scaled, mass_net)
                    end
                    flush(stdout)
                    last_status_time = time()
                end
                prior_guard_status = monitor.last_status
                decision = CFDConvergence.observe!(monitor, history[end])
                implicit_stopped && break
                if release_enabled && release_start==0 && n>=cfg["sfd_start"]+100000 &&
                   get(decision.metrics,"forces_stationary",0.0)==1.0
                    release_start=n
                    println("CMESH2D stationary loads at n=$n: tapering SFD over 20000 iterations, then checking undamped continuity")
                    monitor=CFDConvergence.Monitor(CFDConvergence.Config(
                        sfd_start=n+20000,max_steps=cfg["steps"],residual_scaled_limit=cfg["mass_ceiling"],
                        mass_net_limit=mass_net_limit,stop_on_stall=cfg["auto_stop"]))
                    decision=CFDConvergence.observe!(monitor,history[end])
                end
                if decision.status != prior_guard_status
                    @printf("CMESH2D stop guard [%s] at n=%d: %s\n",decision.status,n,decision.reason)
                    flush(stdout)
                end
                decision.abort && error("CMESH2D stopped on sustained divergence at n=$n: $(decision.reason)")
                checked_implicit=implicit && n>implicit_warmup ? implicit_assessment() : nothing
                implicit_checks_pass=checked_implicit!==nothing && checked_implicit["criteria_passed"]
                if !physical && cfg["auto_stop"] && (implicit ? implicit_checks_pass : decision.stop) && (!release_enabled || (release_start>0 && s.sfd==0))
                    stop_status=implicit ? "monitored_checks_passed" : decision.status
                    stop_reason=implicit ? checked_implicit["reason"] : decision.reason
                    @printf("CMESH2D smart stop [%s] at n=%d/%d: %s\n", stop_status, n, cfg["steps"], stop_reason)
                    flush(stdout)
                    break
                end
                physical && physical_duration_reached && break
            end
        end
    finally
        close(history_io)
        equation_io === nothing || close(equation_io)
        mass_probe_io === nothing || close(mass_probe_io)
        physical_io === nothing || close(physical_io)
        implicit_io === nothing || close(implicit_io)
        implicit_metrics_io === nothing || close(implicit_metrics_io)
    end
    cl, cd = loads_now()
    isfinite(cl) && isfinite(cd) || error("CMESH2D returned nonfinite loads")
    last_record = last(history)
    actual_steps = last_record.iteration
    guard_report = CFDConvergence.convergence_report(monitor)
    final_assessment=implicit ? implicit_assessment() : RANSConvergence.assess(history;
        residual_limits=Dict("residual_scaled"=>cfg["mass_ceiling"],"residual_l2"=>.02),
        config=RANSConvergence.Config(mass_limit=mass_net_limit),
        phase_start=release_start>0 ? release_start+20000 : 0,equation_coverage_complete=false,
        scope="Continuity and measured load stability. Momentum, energy and turbulence require separate evidence.")
    guard_report["damping_release_start"]=release_start
    guard_report["undamped_final_state"]=s.sfd==0
    if release_enabled && s.sfd!=0 && guard_report["early_stop_eligible"]
        guard_report["early_stop_eligible"]=false
        guard_report["status"]="damping_not_released"
        guard_report["reason"]="Filtered-state gates passed, but the iteration budget ended before undamped confirmation"
    end
    guard_report["auto_stop_enabled"] = !physical && cfg["auto_stop"]
    guard_report["stopped_early"] = actual_steps < cfg["steps"]
    guard_report["termination"] = physical ?
        (physical_duration_reached ? "physical_duration_reached" : "iteration_limit_before_physical_duration") :
        actual_steps < cfg["steps"] ?
            (guard_report["status"] == "stalled_stop" ? "stopped_without_convergence" : "convergence_guard_passed") : "iteration_limit"
    if implicit_stopped
        guard_report["early_stop_eligible"]=false
        guard_report["status"]="stalled_stop"
        guard_report["termination"]="implicit_step_rejected"
        guard_report["reason"]=implicit_failure
    elseif implicit
        guard_report["early_stop_eligible"]=false
        guard_report["status"]=final_assessment["criteria_passed"] ? "monitored_checks_passed" : final_assessment["status"]
        guard_report["reason"]=final_assessment["reason"]
        guard_report["termination"]=final_assessment["criteria_passed"] ? "monitored_checks_passed" : "iteration_limit"
    end
    if physical
        guard_report["steady_guard_status"]=guard_report["status"]
        guard_report["status"]=physical_duration_reached ? "physical_time_complete" : "physical_time_incomplete"
        guard_report["reason"]="physical duration and force statistics determine sampling status; the steady guard cannot stop this march"
        guard_report["scope"]="instantaneous continuity diagnostics; no steady convergence or physical stall validation claim"
    end
    residual = last_record.residual_linf
    if ka
        _, _, mass_peak_fraction, mass_peak_i, mass_peak_j = CMeshSolverKA.residual_report_ka(s)
    else
        scaled_mass = abs.(s.dc_r) ./ s.vol
        mass_peak = argmax(scaled_mass)
        mass_peak_i, mass_peak_j = mass_peak[1], mass_peak[2]
        mass_peak_fraction = count(>=(0.99*maximum(scaled_mass)), scaled_mass)/length(scaled_mass)
    end
    mass_dc=Array(s.dc_r); mass_vol=Array(s.vol)
    mass_xc=Array(s.xc); mass_yc=Array(s.yc)
    mass_balance = CFDMassDiagnostics.nonfatal_diagnostic("continuity mass balance") do
        CFDMassDiagnostics.assess_mass_balance(
            mass_dc,mass_vol,Array(s.Fi_r),Array(s.Fj_r),
            mass_xc,mass_yc;
            niw=met.NIW,nif=met.NIF,nbase=get(geometry_metadata,"n_base",0),
            rho_inf=s.rho_inf,speed_inf=speed,chord=maximum(af.x)-minimum(af.x))
    end
    if ka
        haskey(mass_balance,"diagnostic_error") && error(
            "final continuity mass-balance diagnostic failed: $(mass_balance["diagnostic_error"])")
        final_net = mass_balance["cell_divergence_total_over_freestream_flux"]
        net_difference = abs((last_record.mass_net_frac::Float64)-final_net)
        net_tolerance = max(1e-6, 10*abs(mass_balance["telescoping_error_over_freestream_flux"]))
        net_difference <= net_tolerance || error(
            "final signed mass history disagrees with cell-divergence diagnostic by $net_difference (limit $net_tolerance)")
    end
    if !haskey(mass_balance,"diagnostic_error")
        @printf("CMESH2D mass diagnostic: peak region=%s cell=(%d,%d), boundary net/(rho U c)=%.3e, wake-cut net/(rho U c)=%.3e, telescoping error=%.3e\n",
                mass_balance["peak"]["region"],mass_balance["peak"]["i"],
                mass_balance["peak"]["j"],
                mass_balance["boundary_total_over_freestream_flux"],
                mass_balance["boundary_signed_flux_over_freestream_flux"]["inner_wake_lower"]+
                mass_balance["boundary_signed_flux_over_freestream_flux"]["inner_wake_upper"],
                mass_balance["telescoping_error_over_freestream_flux"])
        incoming_ratio = mass_balance["outer_boundary_net_over_incoming_flux"]
        if !isnothing(incoming_ratio)
            @printf("CMESH2D numerical outer-face net / incoming flux = %.3e (domain-size dependent; chord-scaled convergence check unchanged)\n", incoming_ratio)
        end
        flush(stdout)
    end
    g = s.g; ni = met.NI; nj = met.NJ
    rho = Array(s.rho); ux = Array(s.ux); uy = Array(s.uy); temp = Array(s.T); nut = Array(s.nut)
    xc = mass_xc; yc = mass_yc
    mass_balance["peak_dynamics"] = CFDMassDiagnostics.nonfatal_diagnostic(
            "peak timestep/SFD") do
        CFDMassDiagnostics.assess_peak_dynamics(
            mass_peak_i,mass_peak_j,
            (;rho,ux,uy,temp,nut,vol=mass_vol,dtf=Array(s.dtf),
              sb_r=Array(s.sb_r),sb_u=Array(s.sb_u),sb_v=Array(s.sb_v),sb_T=Array(s.sb_T),
              six=Array(s.Six),siy=Array(s.Siy),sjx=Array(s.Sjx),sjy=Array(s.Sjy));
            g=s.g,cfl=cfg["cfl"],nu_lam=s.nu_lam,sst=s.sst,
            viscosity_model=viscosity_model,rho_inf=s.rho_inf,
            sutherland_ratio=s.suth,
            irsj=s.irsj == Int32(1),sfd=s.sfd == Int32(1),
            sfd_alpha=s.sfd_alpha,sfd_chi=s.sfd_chi)
    end
    peak_dt=mass_balance["peak_dynamics"]
    if !haskey(peak_dt,"diagnostic_error")
        @printf("CMESH2D peak timestep: dtf=%.3e acoustic_est=%.3e viscous_est=%.3e limiter=%s\n",
                peak_dt["dtf_actual"],peak_dt["dt_acoustic_estimate"],
                peak_dt["dt_viscous_estimate"],peak_dt["dt_limiter_estimate"])
        flush(stdout)
    end
    all(isfinite, rho) && all(isfinite, ux) && all(isfinite, uy) && all(isfinite, temp) ||
        error("CMESH2D returned a nonfinite flow field")
    minimum(temp[g+1:g+ni,g+1:g+nj]) > 0 || error("CMESH2D returned nonpositive temperature")
    q = 0.5*Float64(s.rho_inf)*speed^2
    p_inf = (2/3)*Float64(s.rho_inf)*Float64(s.T_inf)
    pressure(i,j) = Float64(rho[i+g,j+g])*(2/3)*Float64(temp[i+g,j+g])
    cp(i,j) = (pressure(i,j)-p_inf)/q
    gamma = (5.0+Float64(s.drot))/(3.0+Float64(s.drot))
    mach(i,j) = hypot(Float64(ux[i+g,j+g]), Float64(uy[i+g,j+g])) /
                sqrt(gamma*(2/3)*Float64(temp[i+g,j+g]))
    foil = collect(foil_indices)       # Includes the finite-TE base in wall forces.
    surface = collect(surface_indices) # Excludes the base from Cp upper/lower curves.
    ile = surface[argmin(xc[surface,1])]
    cpfile = joinpath(cfg["outdir"], "surface_cp.csv")
    open(cpfile,"w") do io
        println(io,"surface,x,cp")
        for i in surface
            println(io, i > ile ? "upper" : "lower", ",", xc[i,1], ",", cp(i,1))
        end
    end
    # Same pressure and near-wall shear terms as loads()/loads_ka(), integrated around x/c=0.25.
    cm = 0.0
    fx_pressure = 0.0; fy_pressure = 0.0
    fx_friction = 0.0; fy_friction = 0.0
    # Excess-pressure contributions by wall region expose base drag without
    # attributing a large, cancelling ambient-pressure force to either side.
    fx_pressure_region = zeros(3); fy_pressure_region = zeros(3)
    cm_pressure_region = zeros(3)
    wall_sjx_sum = 0.0; wall_sjy_sum = 0.0
    sjx = ka ? Array(s.Sjx1_h) : s.Sjx[:,1]
    sjy = ka ? Array(s.Sjy1_h) : s.Sjy[:,1]
    vol1 = ka ? Array(s.vol1_h) : s.vol[:,1]
    for i in foil
        fxp = -pressure(i,1)*Float64(sjx[i]); fyp = -pressure(i,1)*Float64(sjy[i])
        wall_sjx_sum += Float64(sjx[i]); wall_sjy_sum += Float64(sjy[i])
        region = i < first(surface) || i > last(surface) ? 3 : (i > ile ? 1 : 2)
        p_excess = pressure(i,1) - p_inf
        fx_pressure_region[region] -= p_excess*Float64(sjx[i])
        fy_pressure_region[region] -= p_excess*Float64(sjy[i])
        sf = hypot(Float64(sjx[i]), Float64(sjy[i]))
        d1 = 0.5*Float64(vol1[i])/sf + 1e-30
        muw = (viscosity_model != "constant_kinematic" ?
            Float64(s.rho_inf)*Float64(s.nu_lam)*
            Float64(CMeshSolverKA._mufac(temp[i+g,g+1],s.suth)) :
            Float64(rho[i+g,g+1])*Float64(s.nu_lam))*sf/d1
        if s.fullvisc==Int32(1)
            wall_module=ka ? CMeshSolverKA : CMeshSolver
            viscosity_model=="constant_kinematic" && (muw *= Float64(wall_module._mufac(temp[i+g,g+1],s.suth)))
            fxf,fyf=wall_module._wall_stokes_traction(Float64(ux[i+g,g+1]),Float64(uy[i+g,g+1]),
                Float64(sjx[i])/sf,Float64(sjy[i])/sf,muw)
        else
            fxf = muw*Float64(ux[i+g,g+1]); fyf = muw*Float64(uy[i+g,g+1])
        end
        fx_pressure += fxp; fy_pressure += fyp
        fx_friction += fxf; fy_friction += fyf
        fx = fxp + fxf; fy = fyp + fyf
        xwall = 0.5*(mesh.X[i,1]+mesh.X[i+1,1])
        ywall = 0.5*(mesh.Y[i,1]+mesh.Y[i+1,1])
        cm += _aerodynamic_cm(xwall, ywall, fx, fy, q)
        cm_pressure_region[region] += _aerodynamic_cm(
            xwall, ywall, -p_excess*Float64(sjx[i]),
            -p_excess*Float64(sjy[i]), q)
    end
    ca = Float64(s.ca); sa = Float64(s.sa)
    cl_pressure = (fy_pressure*ca-fx_pressure*sa)/q
    cd_pressure = (fx_pressure*ca+fy_pressure*sa)/q
    cl_friction = (fy_friction*ca-fx_friction*sa)/q
    cd_friction = (fx_friction*ca+fy_friction*sa)/q
    cl_pressure_region = [(fy_pressure_region[k]*ca-fx_pressure_region[k]*sa)/q
                          for k in 1:3]
    cd_pressure_region = [(fx_pressure_region[k]*ca+fy_pressure_region[k]*sa)/q
                          for k in 1:3]
    pressure_region_closure = hypot(wall_sjx_sum,wall_sjy_sum)
    pressure_region_tolerance = max(1e-5, 2*p_inf*pressure_region_closure/q)
    pressure_region_closure <= 1e-5 ||
        error("CMESH2D foil/base wall normals do not close")
    abs(sum(cl_pressure_region)-cl_pressure) <= pressure_region_tolerance &&
    abs(sum(cd_pressure_region)-cd_pressure) <= pressure_region_tolerance ||
        error("CMESH2D pressure-region forces disagree with the total pressure force")
    abs(cl_pressure+cl_friction-cl) < 1e-3 && abs(cd_pressure+cd_friction-cd) < 1e-3 ||
        error("CMESH2D wall-load decomposition disagrees with loads_ka/loads")
    @printf("CMESH2D force split: Cl_pressure=%+.5f Cl_friction=%+.5f Cd_pressure=%+.6f Cd_friction=%+.6f\n",
            cl_pressure, cl_friction, cd_pressure, cd_friction)
    @printf("CMESH2D excess-pressure drag: upper=%+.6f lower=%+.6f base=%+.6f; wall-normal closure=%.3e\n",
            cd_pressure_region[1], cd_pressure_region[2], cd_pressure_region[3],
            pressure_region_closure)
    flush(stdout)
    function vorticity(i,j)
        im=max(1,i-1); ip=min(ni,i+1); jm=max(1,j-1); jp=min(nj,j+1)
        x_i=xc[ip,j]-xc[im,j]; y_i=yc[ip,j]-yc[im,j]
        x_j=xc[i,jp]-xc[i,jm]; y_j=yc[i,jp]-yc[i,jm]
        u_i=Float64(ux[ip+g,j+g]-ux[im+g,j+g]); u_j=Float64(ux[i+g,jp+g]-ux[i+g,jm+g])
        v_i=Float64(uy[ip+g,j+g]-uy[im+g,j+g]); v_j=Float64(uy[i+g,jp+g]-uy[i+g,jm+g])
        jac=x_i*y_j-x_j*y_i
        abs(jac)<1e-20 ? 0.0 : ((v_i*y_j-v_j*y_i)-(-u_i*x_j+u_j*x_i))/jac
    end
    # Export enough cell centers for a useful near-foil view. Keep the final
    # row/column so the plotted far-field and outlet do not end prematurely.
    stride=max(1,ceil(Int,sqrt(ni*nj/18000)))
    sample_i = sort!(unique(vcat(collect(1:stride:ni), [ni])))
    sample_j = sort!(unique(vcat(collect(1:stride:nj), [nj])))
    sampled_ni = length(sample_i)
    sampled_nj = length(sample_j)
    fieldfile=joinpath(cfg["outdir"],"field_samples.csv")
    open(fieldfile,"w") do io
        println(io,"x,y,mach,cp,pressure,density,u,v,temperature,vorticity,nut")
        for j in sample_j, i in sample_i
            println(io,join((xc[i,j],yc[i,j],mach(i,j),cp(i,j),pressure(i,j),rho[i+g,j+g],
                             ux[i+g,j+g],uy[i+g,j+g],temp[i+g,j+g],vorticity(i,j),nut[i+g,j+g]),','))
        end
    end
    full_field = FullFieldArtifact.write_full_field(
        (i,j)->(xc[i,j], yc[i,j], mach(i,j), cp(i,j), pressure(i,j),
                rho[i+g,j+g], ux[i+g,j+g], uy[i+g,j+g], temp[i+g,j+g],
                vorticity(i,j), nut[i+g,j+g]),
        joinpath(cfg["outdir"], "field_full.msgpack"), ni, nj,
        joinpath(cfg["outdir"], "mesh_full.msgpack"), actual_steps)
    @printf("CMESH2D full field saved: %d cells, %.1f MiB, final n=%d\n",
            full_field["cell_count"], full_field["bytes"]/2.0^20, actual_steps)
    flush(stdout)
    # Evaluate two 50k-iteration windows after the damping transient. A short
    # flat segment alone can be a temporarily frozen, biased flow state.
    tail = history[max(1, end-124):end]
    prior = length(history) >= 250 ? history[end-249:end-125] : typeof(history)()
    cl_span = maximum(h.cl for h in tail) - minimum(h.cl for h in tail)
    cd_span = maximum(h.cd for h in tail) - minimum(h.cd for h in tail)
    cl_mean_shift = isempty(prior) ? Inf : abs(sum(h.cl for h in tail)/length(tail) -
                                             sum(h.cl for h in prior)/length(prior))
    cd_mean_shift = isempty(prior) ? Inf : abs(sum(h.cd for h in tail)/length(tail) -
                                             sum(h.cd for h in prior)/length(prior))
    sfd_index = findfirst(h -> h.iteration >= cfg["sfd_start"], history)
    reference_record = sfd_index === nothing ? first(history) : history[sfd_index]
    residual_ratio = last_record.residual_scaled /
                     max(reference_record.residual_scaled, eps(Float64))
    post_sfd = actual_steps >= cfg["sfd_start"] + 100_000
    # These limits correspond to 0.0025 lift-coefficient units and three drag
    # counts of drift between successive 50k windows. Both windows must also
    # be internally quiet. This is a force-history check, not an equation or
    # model validation certificate.
    prior_cl_span = isempty(prior) ? Inf : maximum(h.cl for h in prior)-minimum(h.cl for h in prior)
    prior_cd_span = isempty(prior) ? Inf : maximum(h.cd for h in prior)-minimum(h.cd for h in prior)
    cl_span_limit = 0.005; cd_span_limit = 0.001
    cl_mean_limit = 0.0025; cd_mean_limit = 0.0003
    numerically_stationary = !physical && post_sfd && length(prior) == 125 &&
                             cl_span <= cl_span_limit && cd_span <= cd_span_limit &&
                             prior_cl_span <= cl_span_limit && prior_cd_span <= cd_span_limit &&
                             cl_mean_shift <= cl_mean_limit && cd_mean_shift <= cd_mean_limit
    implicit && (numerically_stationary=final_assessment["gates"]["loads_stable"])
    physical_sanity = cd >= 0.0 && cd <= 1.0 && abs(cl) <= 3.0
    if implicit
        @printf("CMESH2D implicit final: Cl=%+.5f Cd=%+.6f mass_Rscaled=%.3e stationary=%s basic_force_sanity=%s; %s\n",
            cl,cd,last_record.residual_scaled,string(numerically_stationary),string(physical_sanity),final_assessment["reason"])
    else
        @printf("CMESH2D final: Cl=%+.5f Cd=%+.6f mass_Rscaled=%.3e post-SFD ratio=%.3e Cl_span=%.3e Cd_span=%.3e stationary=%s basic_force_sanity=%s\n",
            cl, cd, last_record.residual_scaled, residual_ratio,
            cl_span, cd_span, string(numerically_stationary), string(physical_sanity))
    end
    flush(stdout)
    time_stats=physical ? CFDTimeStatistics.time_statistics(physical_history;
        burn_in_ctu=burnin_ctu,block_duration_ctu=block_ctu,
        requested_duration_ctu=duration_ctu,
        time_step_refined=Bool(get(cfg,"physical_time_refined",false))) :
        Dict{String,Any}("available"=>false,"reason"=>"steady pseudo-time has no physical averaging clock",
            "physical_mean_eligible"=>false)
    if physical
        @printf("CMESH2D physical final: t=%.12g/%.12g CTU, %d steps, sampling=%s, final field is instantaneous\n",
            elapsed_ctu,duration_ctu,actual_steps,time_stats["status"]);flush(stdout)
    end
    meta = Dict{String,Any}("cl"=>cl,"cd"=>cd,"cm"=>cm,"steps"=>actual_steps,
        "rans_method"=>method,"implicit_method"=>implicit,"implicit_accepted_steps"=>implicit_accepted,
        "implicit_warmup_steps"=>implicit_warmup,"implicit_stop_reason"=>implicit_failure,
        "implicit_history_csv"=>implicit ? "implicit_history.csv" : "",
        "implicit_numerical_controls"=>implicit ? Dict(k=>v for (k,v) in cfg if startswith(k,"implicit_")) : Dict{String,Any}(),
        "steady_assessment"=>final_assessment,"implicit_maturity"=>get(cfg,"implicit_maturity",""),
        "implicit_raw_sst_residuals"=>Dict("available"=>implicit && s.sst && s.keps==0.0f0 && _implicit_raw_sst_available(implicit_rows),
            "csv_columns"=>collect(_IMPLICIT_RAW_SST_METRICS),"qualified_stopping_limits"=>false,
            "omega_prescribed_cells"=>s.keps==0.0f0 ? min(2,s.NJ)*s.NIF : 0,
            "scope"=>"Unscaled primitive transport residuals before clipping; omega transport excludes the exact two prescribed wall rows, which have a separate relative constraint defect. The unmasked omega PDE is diagnostic only."),
        "rans_acceleration"=>Bool(get(cfg,"rans_acceleration",false)),
        "rans_acceleration_method"=>String(get(cfg,"rans_acceleration_method","standard")),
        "irsj"=>s.irsj==Int32(1),
        "stage_profile"=>Dict("enabled"=>profile_stages,"samples"=>profile_samples,
            "flow_seconds"=>profile_flow_seconds,"sst_seconds"=>profile_sst_seconds,
            "scope"=>"CUDA synchronized flow and SST stages; opt-in timing changes synchronization and is diagnostic only"),
        "warm_start_source"=>warm_source,
        "polar_mesh_anchor_deg"=>get(cfg,"polar_mesh_anchor_deg",cfg["cfd_farfield_angle_deg"]),
        "time_mode"=>time_mode,"clock"=>physical ? "physical time" : "steady pseudo-time",
        "elapsed_ctu"=>elapsed_ctu,"requested_duration_ctu"=>duration_ctu,
        "burnin_ctu"=>burnin_ctu,"block_ctu"=>block_ctu,"time_statistics"=>time_stats,
        "physical_duration_reached"=>physical_duration_reached,
        "physical_time_refined"=>Bool(get(cfg,"physical_time_refined",false)),
        "physical_clock_scope"=>physical ? "sum of actual accepted last_global_dt for every step, multiplied by Uinf/chord" : "no physical clock",
        "coefficient_scope"=>"final instantaneous coefficients; qualified physical means are separate time_statistics entries",
        "field_time_scope"=>"instantaneous final state",
        "time_integration_scope"=>physical ? "global explicit SSPRK flow with split SST; coupled temporal order and timestep refinement require separate evidence" : "local pseudo-time steady iteration",
        "physical_dt_ctu_min"=>physical ? dt_ctu_min : 0.0,
        "physical_dt_ctu_max"=>physical ? dt_ctu_max : 0.0,
        "global_time_step"=>physical,"rk_incremental"=>physical,
        "max_steps"=>cfg["steps"], "convergence_guard"=>guard_report,
        "cl_pressure"=>cl_pressure,"cl_friction"=>cl_friction,
        "cd_pressure"=>cd_pressure,"cd_friction"=>cd_friction,
        "cl_pressure_upper"=>cl_pressure_region[1],
        "cl_pressure_lower"=>cl_pressure_region[2],
        "cl_pressure_base"=>cl_pressure_region[3],
        "cd_pressure_upper"=>cd_pressure_region[1],
        "cd_pressure_lower"=>cd_pressure_region[2],
        "cd_pressure_base"=>cd_pressure_region[3],
        "cm_pressure_upper"=>cm_pressure_region[1],
        "cm_pressure_lower"=>cm_pressure_region[2],
        "cm_pressure_base"=>cm_pressure_region[3],
        "pressure_region_scope"=>"excess wall pressure relative to freestream; lower, upper and finite-TE base",
        "wall_normal_vector_closure"=>pressure_region_closure,
        "residual_linf"=>Float64(residual),
        "residual_scaled"=>last_record.residual_scaled,
        "residual_l2"=>last_record.residual_l2,
        "mass_net_scope"=>"signed sum of final post-step continuity divergence / (rho_inf*U_inf*chord); unavailable on plain CPU worker",
        "residual_ratio"=>residual_ratio,
        "residual_reference_iteration"=>reference_record.iteration,
        "mesh_cells"=>ni*nj, "sample_stride"=>stride,
        "sampled_ni"=>sampled_ni,"sampled_nj"=>sampled_nj,
        "gpu"=>gpu,"reconstruction"=>Int(s.recon),
        "rk_weight_closure"=>rk_weight_closure,
        "sst_wall_metric_correction"=>sst_wall_metric_correction,
        "sst_nonorth_correction"=>sst_nonorth_correction,
        "refresh_flow_bc_before_sst"=>refresh_flow_bc_before_sst,
        "characteristic_i_outlet"=>characteristic_i_outlet,
        "full_viscous_stress"=>s.fullvisc == Int32(1),
        "sfd_engaged"=>s.sfd==1,
        "sfd_storage_type"=>sfd_storage,
        "residual_sampling_state"=>ka ? "post_step_final_primitive" : "last_rk_stage",
        "viscosity_model"=>viscosity_model,
        "freestream_temperature_k"=>cfg["freestream_temperature_k"],
        "freestream_temperature_used_for_viscosity"=>viscosity_model == "sutherland",
        "sutherland_constant_k"=>viscosity_model == "sutherland" ? 110.4 : 0.0,
        "sfd_start"=>physical ? 0 : cfg["sfd_start"],
        "estimated_yplus"=>estimated_yplus,
        "mesh_mode"=>mesh_mode,
        "wake_cut_viscous_distance_model"=>get(cfg,
            "wake_cut_viscous_distance_model", "historical_unspecified"),
        "solver_ka_source_sha256"=>get(cfg,"solver_ka_source_sha256", ""),
        "geometry"=>geometry_metadata,
        "trip_upper"=>cfg["trip_upper"], "trip_lower"=>cfg["trip_lower"],
        "numerically_stationary"=>numerically_stationary,
        "stationarity_scope"=>physical ? "physical force sampling and block means reported in time_statistics; no steady convergence claim" : "CL/CD force history in two trailing 50k-iteration windows; not equation convergence",
        "stationarity_window_iterations"=>50_000,
        "stationarity_cl_span_limit"=>cl_span_limit,
        "stationarity_cd_span_limit"=>cd_span_limit,
        "stationarity_cl_mean_limit"=>cl_mean_limit,
        "stationarity_cd_mean_limit"=>cd_mean_limit,
        "physical_sanity"=>physical_sanity,
        "residual_scope"=>"continuity mass equation only",
        "mass_residual_peak_cell"=>[mass_peak_i,mass_peak_j],
        "mass_residual_fraction_near_peak"=>mass_peak_fraction,
        "mass_balance"=>mass_balance,
        "physical_sanity_scope"=>"loose force bounds only; not experimental validation",
        "load_drift"=>cl_span,"drag_drift"=>cd_span,
        "prior_load_drift"=>isfinite(prior_cl_span) ? prior_cl_span : -1.0,
        "prior_drag_drift"=>isfinite(prior_cd_span) ? prior_cd_span : -1.0,
        "cl_mean_shift"=>isfinite(cl_mean_shift) ? cl_mean_shift : -1.0,
        "cd_mean_shift"=>isfinite(cd_mean_shift) ? cd_mean_shift : -1.0,
        "cp_csv"=>basename(cpfile),"field_csv"=>basename(fieldfile),
        "field_full_msgpack"=>full_field["filename"],
        "full_field_cells"=>full_field["cell_count"],
        "full_field_mesh_sha256"=>full_field["mesh_sha256"],
        "full_field_bytes"=>full_field["bytes"],
        "iteration_csv"=>basename(historyfile))
    last_record.mass_net_frac === nothing ||
        (meta["mass_net_frac"] = last_record.mass_net_frac)
    physical && (meta["physical_force_csv"]="physical_force_history.csv")
    if equation_io !== nothing
        meta["equation_residual_csv"] = basename(equation_file)
        meta["equation_residual_scope"] = "raw final-state conservative face divergence for mass, x/y momentum and energy; convective nondimensionalization; no SST equation residual"
    end
    if mass_probe_io !== nothing
        meta["mass_stage_probe_csv"] = basename(mass_probe_file)
        meta["mass_stage_probe_scope"] = "raw post-RK, post-SFD and monitored post-step continuity; SFD source is delta-rho/local-pseudo-dt"
    end
    open(joinpath(cfg["outdir"],"result.toml"),"w") do io; TOML.print(io,meta); end
    physical || _rans_save_restart(cfg, joinpath(cfg["outdir"], "mesh_full.msgpack"),
        String(mesh_mode), [(name="C", solver=s)])
    println("CMESH2D worker complete")
end

include(joinpath(@__DIR__,"species_backend.jl"))
include(joinpath(@__DIR__,"underwater_backend.jl"))
include(joinpath(@__DIR__,"ch_transonic_backend.jl"))
include(joinpath(@__DIR__,"..","workers","ch_transonic_worker.jl"))
include(joinpath(@__DIR__,"..","gpu_lock.jl"))           # the shared GPU lock (the low-speed GPU option; 24 Oct 2026)
include(joinpath(@__DIR__,"..","lowspeed","lowspeed_backend.jl"))
include(joinpath(@__DIR__,"..","lowspeed","lowspeed_ch_backend.jl"))
include(joinpath(@__DIR__,"streamline_backend.jl"))
include(joinpath(@__DIR__,"..","mesh","ch_streamline.jl"))
include(joinpath(@__DIR__,"..","lbm","lbm_geometry.jl"))
include(joinpath(@__DIR__,"..","lbm","lbm_case.jl"))
include(joinpath(@__DIR__,"..","lbm","lbm_render.jl"))
include(joinpath(@__DIR__,"..","lbm","lbm_backend.jl"))

end # module CFDBackend

if abspath(PROGRAM_FILE) == abspath(@__FILE__) && !isempty(ARGS)
    ARGS[1] == "--transonic-worker" && CFDBackend._transonic_worker(ARGS[2])
    ARGS[1] == "--species-worker" && CFDBackend._species_worker(ARGS[2])
    ARGS[1] == "--ch-transonic-worker" && CFDBackend._ch_transonic_worker(ARGS[2])
end
