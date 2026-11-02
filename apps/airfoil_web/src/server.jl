module AirfoilWeb

using HTTP
using UUIDs
using Dates
using Printf
using SHA
using JSON3

include("protocol.jl")
using .AirfoilWire
include(joinpath("backends", "jx_backend.jl"))
using .AirfoilJXBackend
const CavitationReferences = AirfoilJXBackend.AirfoilCavitationCases
include(joinpath("backends", "xfoil_backend.jl"))
include(joinpath("references", "reference_data.jl"))
using .AirfoilReferenceData
include(joinpath("backends", "cfd_backend.jl"))
using .CFDBackend
include(joinpath("references", "nearby_references.jl"))
using .AirfoilNearbyReferences
# Streamline-aligned meshes (development, 24 Oct 2026): the JXfoil paths are computed here (CFDBackend's workers never
# load JXfoil) and handed to the solver worker as a hashed file (streamline_backend.jl).
include(joinpath("mesh", "streamline_alignment.jl"))
using .AirfoilStreamlineAlignment
const STREAMLINE_CACHE = Dict{Any,Any}()
const STREAMLINE_LOCK = ReentrantLock()
"""The JXfoil wake path and stagnation streamline of the CLEAN source section at alpha (cached per section, alpha, Re,
Ncrit, mode and fallback), written as `streamline_paths.toml` in `dir`. A refused viscous path throws its reason."""
function streamline_paths_for(source, flow, alpha::Float64, dir::AbstractString; mode::AbstractString="viscous", fallback::Bool=false)
    kind = String(get(source, "kind", "")); id = String(get(source, "id", ""))
    re = Float64(get(flow, "reynolds", 3e6)); ncrit = Float64(get(flow, "ncrit", 9.0))
    key = (kind, id, bytes2hex(sha256(encode(get(source,"coordinates",nothing),:msgpack))), alpha, re, ncrit, String(mode), fallback)
    d = lock(() -> get(STREAMLINE_CACHE, key, nothing), STREAMLINE_LOCK)
    if d === nothing
        foil = AirfoilJXBackend.resolve_airfoil(source).airfoil
        d = extract_streamlines(foil, alpha; reynolds=re, ncrit, mode=Symbol(mode), allow_inviscid_fallback=fallback)
        lock(() -> (STREAMLINE_CACHE[key] = d), STREAMLINE_LOCK)
    end
    mkpath(dir); f = joinpath(dir, "streamline_paths.toml")
    write_streamline_file(f, d)
    Dict{String,Any}("path"=>f, "sha256"=>d["sha256"], "source"=>d["source"], "source_label"=>d["source_label"])
end
CFDBackend.STREAMLINE_PROVIDER[] = streamline_paths_for

export serve, app_handler

const APP_DIR = normpath(joinpath(@__DIR__, ".."))
const ROOT = normpath(joinpath(APP_DIR, "..", ".."))
const PRIVATE_ROOT = normpath(get(ENV, "LUDWIG_PRIVATE_ROOT",
    joinpath(ROOT, "..", "02_PRIVATE_PROJECT_DEVELOPMENT")))
const STATIC_DIR = joinpath(APP_DIR, "static")
const SOLVER_GUIDE_DIR = joinpath(ROOT, "apps", "ludwig_web", "static")
const STATIC_FILES = Dict(
    "index.html" => "index.html", "favicon.svg" => "favicon.svg",
    "styles/app.css" => "styles/app.css", "styles/ergonomics.css" => "styles/ergonomics.css",
    "styles/output_controls.css"=>"styles/output_controls.css", "scripts/output_controls.js"=>"scripts/output_controls.js",
    "scripts/studies.js"=>"scripts/studies.js", "scripts/benchmarks.js"=>"scripts/benchmarks.js",
    "styles/studies.css"=>"styles/studies.css",
    "styles/plots.css"=>"styles/plots.css", "scripts/plot_controls.js"=>"scripts/plot_controls.js",
    "scripts/app.js" => "scripts/app.js", "scripts/lbm.js" => "scripts/lbm.js",
    "scripts/campaign.js" => "scripts/campaign.js", "scripts/babylon_view.js" => "scripts/babylon_view.js",
    "vendor/babylon.js" => "vendor/babylon.js",
    # Old asset URLs remain available to open tabs and local integrations.
    "app.css" => "styles/app.css", "app.js" => "scripts/app.js",
    "lbm.js" => "scripts/lbm.js", "campaign.js" => "scripts/campaign.js",
    "babylon_view.js" => "scripts/babylon_view.js",
)
const RUN_DIR = joinpath(PRIVATE_ROOT, "results", "airfoil_runs")
const PREVIEW_DIR = joinpath(PRIVATE_ROOT, "TEMP_DELETEABLE", "airfoil_previews")
const LEGACY_RUN_DIR = joinpath(PRIVATE_ROOT, "apps", "airfoil_web", "runs")
const JOBS = Dict{String,Dict{String,Any}}()
const JOB_LOGS = Dict{String,Vector{Dict{String,Any}}}()
const JOB_LOG_SEQ = Dict{String,Int}()
const JOB_HISTORY = Dict{String,Vector{Dict{String,Any}}}()
const JOB_HISTORY_SEQ = Dict{String,Int}()
const JOB_LOCK = ReentrantLock()
const QUEUE = Channel{String}(32)
const WORKER_STARTED = Ref(false)
const MAX_JOBS = 32
const MAX_LOG_LINES = 5000
const LOG_PAGE_SIZE = 500
const MAX_HISTORY_SAMPLES = 10000
const HISTORY_PAGE_SIZE = 500
const PREVIEW_CACHE = Dict{String,Dict{String,Any}}()
const PREVIEW_LOCK = ReentrantLock()
const PREVIEW_MESH_LOCK = ReentrantLock()
const MAX_PREVIEWS = 12
const MIME = Dict(
    ".html"=>"text/html; charset=utf-8", ".js"=>"application/javascript; charset=utf-8",
    ".css"=>"text/css; charset=utf-8", ".svg"=>"image/svg+xml", ".png"=>"image/png",
    ".jpg"=>"image/jpeg", ".jpeg"=>"image/jpeg", ".mp4"=>"video/mp4",
    ".csv"=>"text/csv; charset=utf-8", ".txt"=>"text/plain; charset=utf-8",
    ".log"=>"text/plain; charset=utf-8", ".msgpack"=>"application/msgpack",
    ".vtk"=>"text/plain; charset=utf-8", ".vtu"=>"application/xml", ".vtp"=>"application/xml",
    ".json"=>"application/json; charset=utf-8", ".pdf"=>"application/pdf",
)
include("output_options.jl")
include(joinpath(ROOT,"apps","common","NativeDialogs.jl"))
include(joinpath(ROOT,"apps","common","StaticAssets.jl"))
include(joinpath("api", "campaign_status.jl"))

_header(req::HTTP.Request, name::String) = get(Dict(lowercase(String(k))=>String(v) for (k,v) in req.headers), lowercase(name), "")
_fmt(req; default=:msgpack) = try
    wire_format(_header(req, "Accept"))
catch
    default
end
function _response(code::Integer, payload, format::Symbol)
    HTTP.Response(code, ["Content-Type"=>media_type(format), "Cache-Control"=>"no-store"], encode(_clean(payload), format))
end
_error(code, msg, format) = _response(code, Dict("error"=>string(msg), "ok"=>false), format)

function _clean(x)
    if x isa AbstractDict || x isa NamedTuple
        return Dict(string(k)=>_clean(v) for (k,v) in pairs(x))
    elseif x isa AbstractVector && !(x isa AbstractVector{UInt8})
        return [_clean(v) for v in x]
    elseif x isa Tuple
        return [_clean(v) for v in x]
    elseif x isa AbstractFloat
        return isfinite(x) ? Float64(x) : nothing
    elseif x isa Symbol
        return String(x)
    elseif x isa Integer || x isa Bool || x isa AbstractString || x === nothing || x isa AbstractVector{UInt8}
        return x
    end
    string(x)
end

function _num(x, label; positive=false, nonnegative=false)
    x isa Real && !(x isa Bool) || throw(ArgumentError("$label must be a number"))
    y = Float64(x)
    isfinite(y) || throw(ArgumentError("$label must be finite"))
    positive && y <= 0 && throw(ArgumentError("$label must be positive"))
    nonnegative && y < 0 && throw(ArgumentError("$label must be nonnegative"))
    y
end
function _validated_case(data; geometry_only::Bool=false)
    data isa AbstractDict || throw(ArgumentError("request must be a map"))
    foil = get(data, "airfoil", nothing)
    foil isa AbstractDict || throw(ArgumentError("airfoil must be a map"))
    kind = String(get(foil, "kind", ""))
    kind in ("naca4", "uiuc", "rae2822", "oat15a", "cavitation_reference") ||
        throw(ArgumentError("airfoil.kind must be naca4, uiuc, rae2822, oat15a, or cavitation_reference"))
    id = strip(String(get(foil, "id", "")))
    1 <= ncodeunits(id) <= 128 || throw(ArgumentError("airfoil.id is required and must be <=128 bytes"))
    preset = kind == "cavitation_reference" ? CavitationReferences.preset(id) : nothing
    flow = get(data, "flow", preset === nothing ? nothing : Dict{String,Any}())
    sweep = get(data, "sweep", preset === nothing ? Dict{String,Any}() : preset["sweep"])
    backends = get(data, "backends", preset === nothing ? nothing : preset["backends"])
    flow isa AbstractDict || throw(ArgumentError("flow must be a map"))
    sweep isa AbstractDict || throw(ArgumentError("sweep must be a map"))
    backends isa AbstractVector || throw(ArgumentError("backends must be an array"))
    preset === nothing || (flow = merge(preset["flow"], flow))
    kind == "naca4" && !occursin(r"^[0-9]{4}$", id) && throw(ArgumentError("NACA airfoil must be four digits"))
    kind in ("rae2822", "oat15a") && lowercase(id) != kind &&
        throw(ArgumentError("airfoil.id must be $kind for this source"))
    backend_names = unique(String.(backends))
    !isempty(backend_names) && length(backend_names) <= 4 || throw(ArgumentError("select at least one backend"))
    all(b->b in ("jxfoil", "xfoil", "transonic", "cavitation", "lowspeed", "lbm"), backend_names) || throw(ArgumentError("unknown backend"))
    # LBM 3D (LUDWIG, 27 Oct 2026): runs alone (its own geometry, lattice, GPU queue and outputs)
    "lbm" in backend_names && length(backend_names) > 1 && throw(ArgumentError("The LBM 3D (LUDWIG) backend runs alone; select it without other backends."))
    mach = _num(get(flow, "mach", 0.0), "mach"; nonnegative=true)
    mach <= 5 || throw(ArgumentError("mach must be <= 5"))
    "xfoil" in backend_names && mach>=1 && throw(ArgumentError("XFOIL executable supports subsonic Mach below1; shocks and supersonic flow require CFD"))
    reynolds = _num(get(flow, "reynolds", 100000.0), "reynolds"; positive=true)
    ncrit = _num(get(flow, "ncrit", 9.0), "ncrit"; positive=true)
    sigma = _num(get(flow, "sigma", 1.25), "sigma"; positive=true)
    flow_out = Dict{String,Any}(String(k)=>v for (k,v) in pairs(flow))
    outputs = _output_preferences(data)
    flow_out["snapshot_interval_seconds"]=60*outputs["snapshot_interval_minutes"]
    display_every = get(data,"display_every",get(flow,"display_every",nothing))
    if display_every !== nothing
        interval=_num(display_every,"display_every";positive=true)
        isinteger(interval) && interval<=1_000_000 || throw(ArgumentError("display_every must be an integer in [1,1000000]"))
        display_every=Int(interval)
        for key in ("display_every","lbm_display_every","cfd_history_stride","cfd_display_every","cav_history_stride","cav_fs_frame_stride")
            flow_out[key]=display_every
        end
    end
    if haskey(data,"target_vram_percent")
        percent=_num(data["target_vram_percent"],"target_vram_percent";positive=true)
        percent<=100 || throw(ArgumentError("target_vram_percent must not exceed 100"))
        flow_out["lbm_target_vram_percent"]=percent
    end
    flow_out["output_preferences"]=outputs
    flow_out["mach"] = mach; flow_out["reynolds"] = reynolds
    flow_out["ncrit"] = ncrit; flow_out["sigma"] = sigma
    for key in ("trip_upper","trip_lower")
        haskey(flow_out,key)||continue
        location=_num(flow_out[key],key;nonnegative=true)
        location<=1||throw(ArgumentError("$key must be a chord fraction in[0,1]"))
        flow_out[key]=location
    end
    material=lowercase(String(get(flow_out,"cav_fluid","water")))
    domain=lowercase(String(get(flow_out,"cav_domain","tunnel")))
    material in ("water","air") || throw(ArgumentError("cav_fluid must be water or air"))
    domain in ("tunnel","free_fluid","underwater") || throw(ArgumentError("invalid cav_domain"))
    flow_out["cav_fluid"]=material;flow_out["cav_domain"]=domain
    topology=String(get(flow_out,"cfd_te_topology","auto"))
    topology in ("auto","aplus") || throw(ArgumentError("cfd_te_topology must be auto or aplus"))
    flow_out["cfd_te_topology"]=topology
    if "cavitation" in backend_names && domain=="underwater"
        material=="water" || throw(ArgumentError("underwater requires water"))
        backend_names==["cavitation"] || throw(ArgumentError("underwater uses SI speed and a distinct physical model; select the cavitation backend alone"))
        CFDBackend._underwater_config("",flow_out,"";source=foil)
    end
    if "cavitation" in backend_names && material=="air" &&
       (haskey(flow_out,"trip_upper")||haskey(flow_out,"trip_lower"))
        get(flow_out,"trip_upper",0.03)==get(flow_out,"trip_lower",0.05)||throw(ArgumentError(
            "Species air supports only a symmetric SST k-floor trip; natural XFOIL transition and independent upper/lower trips are not implemented"))
    end
    flow_out["cav_depth_chords"]=_num(get(flow_out,"cav_depth_chords",1.0),"cav_depth_chords";positive=true)
    flow_out["cav_chord_m"]=_num(get(flow_out,"cav_chord_m",1.0),"cav_chord_m";positive=true)
    "cavitation" in backend_names && CFDBackend.cavitation_configuration(flow_out)
    # Low-speed CFD (PBCav, version 2): incompressible, so Mach is not one of its inputs; its own settings are checked here.
    "lowspeed" in backend_names && CFDBackend.lowspeed_configuration(flow_out)
    "lbm" in backend_names && CFDBackend.lbm_configuration(flow_out)
    if haskey(flow_out, "xfoil_geometry")
        String(flow_out["xfoil_geometry"]) in ("same_as_cfd", "native") ||
            throw(ArgumentError("xfoil_geometry must be same_as_cfd or native"))
        flow_out["xfoil_geometry"] = String(flow_out["xfoil_geometry"])
    end
    alphas = Float64[]
    if get(sweep, "enabled", false) == true
        start = _num(get(sweep, "start", nothing), "sweep.start")
        stop = _num(get(sweep, "stop", nothing), "sweep.stop")
        step = _num(get(sweep, "step", nothing), "sweep.step")
        step != 0 && sign(step) == sign(stop-start) || throw(ArgumentError("sweep step must point toward stop"))
        n = floor(Int, (stop-start)/step + 1e-9) + 1
        1 <= n <= ("lbm" in backend_names ? 12 : any(b->b in ("transonic","cavitation","lowspeed"), backend_names) ? 31 : 121) ||
            throw(ArgumentError("sweep has too many points for the selected backend"))
        alphas = [start+(i-1)*step for i in 1:n]
        abs(alphas[end]-stop) <= 1e-7 || throw(ArgumentError("sweep step must land on stop"))
    else
        alphas = [_num(get(flow, "alpha_deg", nothing), "alpha_deg")]
    end
    all(abs(a)<=90 for a in alphas) || throw(ArgumentError("alpha must lie within ±90 degrees"))
    flow_out["alpha_deg"] = alphas[1]
    for key in ("cfd_steps", "cfd_n_base", "cav_warm_steps", "cav_frames", "cfd_history_stride")
        haskey(flow_out,key) || continue
        val = _num(flow_out[key], key; nonnegative=true)
        val == round(val) || throw(ArgumentError("$key must be an integer"))
        flow_out[key] = Int(val)
    end
    if haskey(flow_out,"cfd_n_base")
        1 <= flow_out["cfd_n_base"] <= 8 ||
            throw(ArgumentError("cfd_n_base must be between 1 and 8 faces per base half"))
    end
    for key in ("cav_duration_ctu", "cav_channel_height")
        haskey(flow_out,key) && (flow_out[key] = _num(flow_out[key], key; positive=true))
    end
    haskey(flow_out,"cfd_gpu") && (flow_out["cfd_gpu"] isa Bool || throw(ArgumentError("cfd_gpu must be boolean")))
    haskey(flow_out,"cfd_rk_weight_closure") &&
        (flow_out["cfd_rk_weight_closure"] isa Bool ||
         throw(ArgumentError("cfd_rk_weight_closure must be boolean")))
    haskey(flow_out,"cfd_sst_wall_metric_correction") &&
        (flow_out["cfd_sst_wall_metric_correction"] isa Bool ||
         throw(ArgumentError("cfd_sst_wall_metric_correction must be boolean")))
    haskey(flow_out,"cfd_auto_stop") && (flow_out["cfd_auto_stop"] isa Bool || throw(ArgumentError("cfd_auto_stop must be boolean")))
    if haskey(flow_out,"cfd_time_mode")
        String(flow_out["cfd_time_mode"]) in ("steady","physical") || throw(ArgumentError("cfd_time_mode must be steady or physical"))
    end
    for key in ("cfd_duration_ctu","cfd_block_ctu")
        haskey(flow_out,key) && (flow_out[key]=_num(flow_out[key],key;positive=true))
    end
    if haskey(flow_out,"cfd_history_stride")
        1<=flow_out["cfd_history_stride"]<=1000000 || throw(ArgumentError("cfd_history_stride must be in [1, 1000000]"))
    end
    if haskey(flow_out,"cfd_force_sample_dt_ctu")
        interval=_num(flow_out["cfd_force_sample_dt_ctu"],"cfd_force_sample_dt_ctu";positive=true)
        0.001<=interval<=1.0 || throw(ArgumentError("cfd_force_sample_dt_ctu must be in [0.001, 1]"))
        flow_out["cfd_force_sample_dt_ctu"]=interval
    end
    haskey(flow_out,"cfd_burnin_ctu") && (flow_out["cfd_burnin_ctu"]=_num(flow_out["cfd_burnin_ctu"],"cfd_burnin_ctu";nonnegative=true))
    haskey(flow_out,"cfd_physical_time_refined") && !(flow_out["cfd_physical_time_refined"] isa Bool) &&
        throw(ArgumentError("cfd_physical_time_refined must be boolean"))
    if haskey(flow_out,"cfd_te_thickness")
        thickness = _num(flow_out["cfd_te_thickness"],"cfd_te_thickness";nonnegative=true)
        thickness <= 0.05 || throw(ArgumentError("cfd_te_thickness must be <= 0.05 chord"))
        flow_out["cfd_te_thickness"] = thickness
    end
    haskey(flow_out,"cfd_mass_ceiling") && (flow_out["cfd_mass_ceiling"] = _num(flow_out["cfd_mass_ceiling"], "cfd_mass_ceiling"; positive=true))
    haskey(flow_out,"cfd_mass_ceiling") && flow_out["cfd_mass_ceiling"] > 1000 && throw(ArgumentError("cfd_mass_ceiling must not exceed 1000"))
    if haskey(flow_out, "cfd_mesh")
        flow_out["cfd_mesh"] isa AbstractString || throw(ArgumentError("cfd_mesh must be a string"))
        mesh = String(flow_out["cfd_mesh"])
        mesh in ("cavitation_aplus_farfield", "aplus_nested_refined", "legacy_sharp_cmesh") ||
            throw(ArgumentError("cfd_mesh must be cavitation_aplus_farfield, aplus_nested_refined, or legacy_sharp_cmesh"))
        flow_out["cfd_mesh"] = mesh
        mesh == "legacy_sharp_cmesh" && get(flow_out,"cfd_te_thickness",0.0) > 0 &&
            throw(ArgumentError("finite trailing-edge thickness requires an A-plus mesh"))
    end
    if haskey(flow_out, "cfd_sfd_storage")
        storage = String(flow_out["cfd_sfd_storage"])
        storage in ("float32", "float64") ||
            throw(ArgumentError("cfd_sfd_storage must be float32 or float64"))
        flow_out["cfd_sfd_storage"] = storage
    end
    if haskey(flow_out, "cfd_viscosity_model")
        model = String(flow_out["cfd_viscosity_model"])
        model in ("constant_kinematic", "sutherland", "constant_dynamic") ||
            throw(ArgumentError("cfd_viscosity_model must be constant_kinematic, sutherland, or constant_dynamic"))
        flow_out["cfd_viscosity_model"] = model
    end
    if haskey(flow_out, "cfd_freestream_temperature_k")
        temperature = _num(flow_out["cfd_freestream_temperature_k"],
                           "cfd_freestream_temperature_k"; positive=true)
        100.0 <= temperature <= 2000.0 ||
            throw(ArgumentError("cfd_freestream_temperature_k must be in [100, 2000] K"))
        flow_out["cfd_freestream_temperature_k"] = temperature
    end
    if haskey(flow_out, "cfd_resolution")
        flow_out["cfd_resolution"] isa AbstractString ||
            throw(ArgumentError("cfd_resolution must be a string"))
        resolution = String(flow_out["cfd_resolution"])
        resolution in ("standard", "fine", "very_fine") ||
            throw(ArgumentError("cfd_resolution must be standard, fine, or very_fine"))
        flow_out["cfd_resolution"] = resolution
    end
    # Gurney flap (23 Oct 2026): validated and normalized; every unsupported combination is refused with its reason.
    # Absent gurney_height leaves the request exactly as before.
    gurney = CFDBackend.gurney_check_request(flow_out, backend_names; mesh=!geometry_only)   # a geometry-only preview is not a mesh
    # Streamline alignment (development, opt-in): absent keys leave the request exactly as before.
    geometry_only || CFDBackend.streamline_check_request(flow_out, backend_names)
    if gurney !== nothing
        flow_out["gurney_height"] = gurney.height; flow_out["gurney_mode"] = gurney.mode
    end
    Dict{String,Any}(
        "airfoil"=>merge(Dict{String,Any}("kind"=>kind, "id"=>id), kind == "uiuc" && haskey(foil,"coordinates") ?
            Dict("coordinates"=>AirfoilJXBackend.study_coordinates(foil["coordinates"])) : Dict()),
        "flow"=>flow_out,
        "sweep"=>Dict("enabled"=>length(alphas)>1, "start"=>alphas[1], "stop"=>alphas[end], "step"=>length(alphas)>1 ? alphas[2]-alphas[1] : 0.0),
        "alphas"=>alphas,
        "backends"=>backend_names,
        "outputs"=>outputs, "display_every"=>display_every,
    )
end

function _put_job(id::String, changes::AbstractDict)
    lock(JOB_LOCK) do
        haskey(JOBS, id) || return
        merge!(JOBS[id], changes)
        JOBS[id]["updated_at"] = string(now(UTC)) * "Z"
    end
end
function _get_job(id::String)
    lock(JOB_LOCK) do
        haskey(JOBS,id) ? deepcopy(JOBS[id]) : nothing
    end
end
function _append_log(id::String, source::String, message)
    lines = split(replace(string(message), '\r'=>'\n'), '\n')
    for raw in lines
        isempty(raw) && continue
        line = String(raw)
        at = string(now(UTC)) * "Z"
        lock(JOB_LOCK) do
            haskey(JOB_LOGS, id) || return
            seq = JOB_LOG_SEQ[id] + 1
            JOB_LOG_SEQ[id] = seq
            push!(JOB_LOGS[id], Dict{String,Any}(
                "seq"=>seq, "at"=>at, "source"=>source, "line"=>line))
            if length(JOB_LOGS[id]) > MAX_LOG_LINES
                deleteat!(JOB_LOGS[id], 1:(length(JOB_LOGS[id])-MAX_LOG_LINES))
            end
            try
                dir = _safe_job_dir(id)
                mkpath(dir)
                open(joinpath(dir, "job.log"), "a") do io
                    println(io, "[$at] [$source] $line")
                end
            catch err
                @warn "Could not write job log" id exception=err
            end
        end
    end
end
function _get_logs(id::String, after::Int)
    after >= 0 || throw(ArgumentError("after must be nonnegative"))
    lock(JOB_LOCK) do
        haskey(JOB_LOGS, id) || return nothing
        lines = JOB_LOGS[id]
        first_seq = isempty(lines) ? JOB_LOG_SEQ[id] + 1 : lines[1]["seq"]
        dropped = max(0, first_seq - after - 1)
        start = findfirst(entry -> entry["seq"] > after, lines)
        page = isnothing(start) ? Dict{String,Any}[] :
            deepcopy(lines[start:min(start+LOG_PAGE_SIZE-1,length(lines))])
        next_seq = isempty(page) ? after : page[end]["seq"]
        return Dict{String,Any}(
            "entries"=>page, "next_seq"=>next_seq, "dropped"=>dropped,
            "status"=>JOBS[id]["status"],
            "log_url"=>"/api/jobs/$id/files/job.log",
        )
    end
end
function _append_history(id::String, backend::String, event)
    lock(JOB_LOCK) do
        haskey(JOB_HISTORY, id) || return
        seq = JOB_HISTORY_SEQ[id] + 1
        JOB_HISTORY_SEQ[id] = seq
        entry = Dict{String,Any}(
            "seq"=>seq, "backend"=>backend,
            "alpha_deg"=>get(event,"alpha_deg",nothing),
            "iteration"=>get(event,"iteration",nothing),
            "total_iterations"=>get(event,"total_iterations",nothing),
            "residual"=>get(event,"residual",get(event,"residual_scaled",get(event,"residual_linf",nothing))),
            "residual_linf"=>get(event,"residual_linf",nothing),
            "residual_scaled"=>get(event,"residual_scaled",nothing),
            "residual_l2"=>get(event,"residual_l2",nothing),
            "residual_kind"=>get(event,"residual_kind",nothing),
            "at"=>string(now(UTC))*"Z",
            "mass_net_frac"=>get(event,"mass_net_frac",nothing),
            "t_ctu"=>get(event,"t_ctu",nothing),
            "requested_duration_ctu"=>get(event,"requested_duration_ctu",nothing),
            "applied_dt_ctu"=>get(event,"applied_dt_ctu",nothing),
            "t_seconds"=>get(event,"t_seconds",nothing),
            "requested_duration_seconds"=>get(event,"requested_duration_seconds",nothing),
            "dt_seconds"=>get(event,"dt_seconds",nothing),
            "cl"=>get(event,"cl",nothing), "cd"=>get(event,"cd",nothing),"cm"=>get(event,"cm",nothing),
        )
        push!(JOB_HISTORY[id], _clean(entry))
        if length(JOB_HISTORY[id]) > MAX_HISTORY_SAMPLES
            deleteat!(JOB_HISTORY[id], 1:(length(JOB_HISTORY[id])-MAX_HISTORY_SAMPLES))
        end
    end
end
function _get_history(id::String, after::Int)
    after >= 0 || throw(ArgumentError("after must be nonnegative"))
    lock(JOB_LOCK) do
        haskey(JOB_HISTORY, id) || return nothing
        values = JOB_HISTORY[id]
        first_seq = isempty(values) ? JOB_HISTORY_SEQ[id] + 1 : values[1]["seq"]
        dropped = max(0, first_seq-after-1)
        start = findfirst(entry -> entry["seq"] > after, values)
        page = isnothing(start) ? Dict{String,Any}[] :
            deepcopy(values[start:min(start+HISTORY_PAGE_SIZE-1,length(values))])
        next_seq = isempty(page) ? after : page[end]["seq"]
        return Dict{String,Any}(
            "entries"=>page, "next_seq"=>next_seq, "dropped"=>dropped,
            "latest_seq"=>JOB_HISTORY_SEQ[id],
            "status"=>JOBS[id]["status"],
        )
    end
end

function _preview_request(data)
    data isa AbstractDict || throw(ArgumentError("request must be a map"))
    mesh = get(data,"mesh",false)
    mesh isa Bool || throw(ArgumentError("mesh must be boolean"))
    flow = get(data,"flow",Dict{String,Any}())
    flow isa AbstractDict || throw(ArgumentError("flow must be a map"))
    foil = get(data,"airfoil",nothing)
    preset = foil isa AbstractDict && get(foil,"kind","")=="cavitation_reference" ?
        CavitationReferences.preset(String(get(foil,"id",""))) : nothing
    defaults = preset === nothing ? Dict{String,Any}("mach"=>0.11,"reynolds"=>300_000.0,"alpha_deg"=>0.0) : preset["flow"]
    flow = merge(defaults,flow)
    case = _validated_case(Dict{String,Any}("airfoil"=>foil,
        "flow"=>flow,"sweep"=>Dict("enabled"=>false),"backends"=>get(data,"backends",preset === nothing ? ["transonic"] : preset["backends"]));geometry_only=!mesh)
    case, mesh
end

function _make_preview(case, want_mesh::Bool)
    # Cache and meshing have separate locks: selecting another airfoil must
    # display its geometry immediately even while an older mesh is building.
    cache_input = want_mesh ? Dict("case"=>case,"mesh"=>true) :
        Dict("airfoil"=>case["airfoil"],"cfd_te_thickness"=>get(case["flow"],"cfd_te_thickness",0.0),
             "mesh"=>false)
    # The Gurney flap changes the geometry: its settings join the geometry-only key (absent when the flap is off).
    gurney_on = CFDBackend.gurney_enabled(case["flow"])
    if gurney_on && !want_mesh
        cache_input["gurney"] = Dict(k=>get(case["flow"],k,nothing) for k in ("gurney_height","gurney_mode"))
    end
    # LBM 3D: its section (thick plates, spoiler) joins the geometry-only key; absent for every other backend
    lbm_on = case["backends"] == ["lbm"]
    lbm_on && !want_mesh && (cache_input["lbm"] = Dict(k=>v for (k,v) in case["flow"] if startswith(String(k),"lbm_") || k in ("gurney_height","gurney_mode")))
    key = bytes2hex(sha256(encode(cache_input,:msgpack)))
    cached = lock(PREVIEW_LOCK) do
        get(PREVIEW_CACHE,key,nothing)
    end
    if cached !== nothing
        # Geometry-only cache keys intentionally omit the flow. Reference
        # setup metadata must still describe the user's current editable input.
        return case["airfoil"]["kind"]=="cavitation_reference" ?
            merge(cached,Dict("reference_case"=>_clean(_reference_setup(case)))) : cached
    end
        resolved = AirfoilJXBackend.resolve_airfoil(case["airfoil"])
        foil = resolved.airfoil
        id = string(uuid4())
        outdir = joinpath(PREVIEW_DIR,id)
        mkpath(outdir)
        path = joinpath(outdir,"airfoil.dat")
        _save_geometry(foil,path)
        equation_naca = case["airfoil"]["kind"] == "naca4"
        gurney = gurney_on ? CFDBackend.gurney_section_file(case["airfoil"],path,case["flow"],outdir) : nothing
        af = gurney !== nothing ? gurney.af :
             equation_naca ? CFDBackend.CMeshGen.naca4_points(parse(Int,case["airfoil"]["id"])) :
                             CFDBackend.CMeshGen.read_airfoil_dat(path)
        conditioned = CFDBackend.CMeshGen.condition_airfoil(af;
            trailing_edge_thickness=get(case["flow"],"cfd_te_thickness",0.0))
        conditioned.metadata["geometry_support"] = equation_naca ?
            "NACA four-digit equations evaluated at 4097 cosine-spaced support points" :
            "C2 interpolation of source coordinates"
        if gurney !== nothing
            conditioned.metadata["gurney"] = gurney.metadata
            conditioned.metadata["transform"] = CFDBackend.gurney_point_note(gurney.metadata)
            conditioned.metadata["geometry_support"] = "Gurney flap host section (the airfoil, unchanged; the plate is drawn on it; in the low-speed CFD it is a baffle, a wall in the section's own C+H mesh): $(gurney.metadata["support"]), natural C2 reconstruction"
        end
        geometry = Dict{String,Any}("name"=>gurney === nothing ? foil.name : "$(foil.name) · Gurney flap","x"=>conditioned.airfoil.x,
            "y"=>conditioned.airfoil.y,"curvature"=>conditioned.curvature,
            "source_url"=>get(resolved.metadata,"source_url",nothing),
            "attribution"=>get(resolved.metadata,"attribution",nothing))
        payload = Dict{String,Any}("geometry"=>geometry,
            "geometry_metadata"=>conditioned.metadata,"mesh"=>nothing)
        gurney === nothing || (payload["gurney"] = gurney.metadata)
        lbm_on && !want_mesh && (payload["lbm_section"] = try
                CFDBackend.lbm_section_preview(path,case["flow"];source=case["airfoil"])
            catch err
                Dict{String,Any}("error"=>sprint(showerror,err))
            end)
        if case["airfoil"]["kind"]=="cavitation_reference"
            payload["reference_case"]=_reference_setup(case)
            payload["geometry_metadata"]["source_geometry_sha256"]=resolved.metadata["source_geometry_sha256"]
        end
        if want_mesh
            built = lock(PREVIEW_MESH_LOCK) do
                if lbm_on
                    # LBM 3D: the section, its STL and the lattice (pure geometry, no solver process)
                    CFDBackend.preview_lbm_mesh(path,case["flow"],outdir;source=case["airfoil"])
                elseif case["backends"]==["cavitation"]
                    CFDBackend.preview_species_mesh(path,case["flow"],outdir;source=case["airfoil"])
                elseif "lowspeed" in case["backends"] && all(b->b in ("lowspeed","xfoil"),case["backends"])
                    # The low-speed CFD's own mesh (PBCav worker, mesh only): the same nodes as its Run.
                    CFDBackend.preview_lowspeed_mesh(path,case["flow"],outdir;source=case["airfoil"])
                else
                    CFDBackend.preview_mesh(path,case["flow"],outdir;source=case["airfoil"])
                end
            end
            if lbm_on
                lbm = built.lbm
                lbm["stl"]["file"] = "/api/previews/$id/files/lbm_section.stl"
                lbm["config"] = "/api/previews/$id/files/config.yaml"
                payload["lbm"] = lbm
                payload["geometry_metadata"] = merge(payload["geometry_metadata"], built.metadata)
            else
            mesh = built.mesh
            # Preserve smooth source coordinates/curvature for the geometry
            # chart. Mesh views draw the exact independent wall_x/wall_y.
            mesh["artifact"] = "/api/previews/$id/files/mesh_full.msgpack"
            mesh["preview_artifact"] = "/api/previews/$id/files/mesh_preview.msgpack"
            payload["mesh"] = mesh
            payload["geometry_metadata"] = built.metadata
            gurney === nothing || haskey(payload["geometry_metadata"],"gurney") || (payload["geometry_metadata"]["gurney"] = gurney.metadata)
            end
        end
        if case["airfoil"]["kind"]=="cavitation_reference"
            payload["geometry_metadata"]["source_geometry_sha256"]=resolved.metadata["source_geometry_sha256"]
        end
    lock(PREVIEW_LOCK) do
        length(PREVIEW_CACHE) >= MAX_PREVIEWS && delete!(PREVIEW_CACHE,first(keys(PREVIEW_CACHE)))
        PREVIEW_CACHE[key] = _clean(payload)
        PREVIEW_CACHE[key]
    end
end

"Nominal reference conditions and explicit user changes; no validation certificate."
function _reference_setup(case)
    get(case["airfoil"],"kind","")=="cavitation_reference" || return nothing
    preset=CavitationReferences.preset(case["airfoil"]["id"])
    changes=Dict{String,Any}(k=>Dict("preset"=>v,"actual"=>get(case["flow"],k,nothing))
        for (k,v) in preset["flow"] if get(case["flow"],k,nothing)!=v)
    backend_match=case["backends"]==preset["backends"]
    sweep_match=!case["sweep"]["enabled"] && case["alphas"]==[preset["flow"]["alpha_deg"]]
    merge(preset,Dict("modified_flow"=>changes,"actual_backends"=>case["backends"],
        "actual_sweep"=>case["sweep"],"matches_preset"=>isempty(changes)&&backend_match&&sweep_match,
        "qualification"=>"Reference setup preset only; current solver and mesh are not an archived validation reproduction"))
end
function _safe_job_dir(id::String)
    occursin(r"^[0-9a-f-]{36}$", id) || throw(ArgumentError("invalid job id"))
    current=joinpath(RUN_DIR,id)
    preview=joinpath(PREVIEW_DIR,id)
    isdir(preview) && !isdir(current) && return preview
    historical=joinpath(LEGACY_RUN_DIR,id)
    isdir(current) || !isdir(historical) ? current : historical
end
function _save_geometry(foil, path)
    open(path, "w") do io
        println(io, foil.name)
        for (x,y) in zip(foil.x, foil.y)
            println(io, x, " ", y)
        end
    end
end
function _point_list(output, backend)
    if output isa AbstractDict && haskey(output, "points")
        return output["points"]
    elseif output isa AbstractDict && haskey(output, "results")
        return output["results"]
    elseif output isa AbstractVector
        return output
    elseif output isa AbstractDict
        return [output]
    end
    throw(ArgumentError("backend $backend returned no point records"))
end
function _asset_urls(x, id::String, outdir::String)
    if x isa AbstractString
        s = String(x)
        if isfile(s)
            rel = relpath(abspath(s), abspath(outdir))
            startswith(rel, "..") || return "/api/jobs/$id/files/" * replace(rel, '\\'=>'/')
        elseif isfile(joinpath(outdir, s))
            return "/api/jobs/$id/files/" * replace(s, '\\'=>'/')
        end
        return s
    elseif x isa AbstractDict || x isa NamedTuple
        return Dict(string(k)=>_asset_urls(v,id,outdir) for (k,v) in pairs(x))
    elseif x isa AbstractVector && !(x isa AbstractVector{UInt8})
        return [_asset_urls(v,id,outdir) for v in x]
    end
    x
end
function _point_outcome(results)
    n = length(results)
    experimental=count(p->get(p,"status","")=="experimental_physical_state",results)
    experimental>0 && return ("partial","Experimental physical states saved; startup coefficients are unqualified")
    completed = count(p -> lowercase(string(get(p,"status","completed"))) in
                             ("completed","converged","ok"), results)
    n > 0 && completed == n && return ("completed", "Analysis complete")
    # A successful diagnostic run can save real, inspectable CFD fields while
    # failing convergence. Keep that outcome separate from a worker exception.
    saved_diagnostics=count(results) do p
        get(p,"status","") in ("not_converged","stopped_without_convergence") || return false
        fields=get(p,"fields",nothing)
        fields isa AbstractDict && get(fields,"full_artifact",nothing) isa AbstractString || return false
        all(k->get(p,k,nothing) isa Real && isfinite(p[k]),("cl","cd","cm"))
    end
    completed==0 && saved_diagnostics>0 && return ("partial","CFD diagnostic results saved; convergence was not reached. Inspect the fields and residual history.")
    stopped = count(p -> get(p,"status","") == "stopped_without_convergence",results)
    stopped == n && n > 0 &&
        return ("failed","Stopped without convergence; diagnostic results saved")
    unqualified=count(p->get(p,"status","") in ("not_converged","insufficient_sampling","unqualified_physical_mean","invalid_physical_state"),results)
    completed==0 && unqualified>0 && return ("failed","No qualified polar points; unconverged or insufficiently sampled diagnostic results saved")
    completed == 0 && return ("failed", "All operating points failed")
    unqualified>0 && return ("partial","Analysis finished; some points lack convergence or physical sampling qualification, with diagnostic results saved")
    stopped > 0 && return ("partial","Analysis finished; some points stopped without convergence, with diagnostic results saved")
    ("partial", "Analysis finished with partial results")
end
function _run_job(id::String)
    job = _get_job(id)
    job === nothing && return
    case = job["case"]
    outdir = _safe_job_dir(id)
    mkpath(outdir)
    _put_job(id, Dict("status"=>"running", "message"=>"Preparing airfoil geometry", "progress"=>0.02))
    _append_log(id, "server", "Preparing $(case["airfoil"]["kind"]) $(case["airfoil"]["id"]) at Mach $(case["flow"]["mach"]), Re $(case["flow"]["reynolds"]); angles $(join(case["alphas"], ", ")) deg.")
    try
        resolved = AirfoilJXBackend.resolve_airfoil(case["airfoil"])
        foil = resolved.airfoil
        _append_log(id, "server", "Geometry loaded: $(foil.name), $(length(foil.x)) points.")
        geom = Dict{String,Any}("name"=>foil.name, "x"=>foil.x, "y"=>foil.y,
            "source_url"=>get(resolved.metadata,"source_url",nothing),
            "attribution"=>get(resolved.metadata,"attribution",nothing))
        geopath = joinpath(outdir, "airfoil.dat")
        _save_geometry(foil, geopath)
        case["airfoil"]["path"] = geopath
        results = Any[]
        meshes = Dict{String,Any}()
        live_fields = Dict{String,Any}()
        saved_snapshots = Any[]
        lbm_live = Dict{String,Any}(); lbm_lattices = Dict{String,Any}()   # LBM 3D: running mid-plane frames and lattices
        modified_te = get(case["flow"],"cfd_te_thickness",0.0) > 0
        gurney_on = CFDBackend.gurney_enabled(case["flow"])
        reference = (modified_te || gurney_on) ? nothing : AirfoilReferenceData.reference_for_case(case)
        modified_te && _append_log(id,"geometry",
            "CFD trailing-edge thickness explicitly set to $(case["flow"]["cfd_te_thickness"]) chord. Published native-geometry reference overlays are disabled; XFoil retains the source geometry.")
        if gurney_on
            gsec = CFDBackend.gurney_section(case["airfoil"], geopath, case["flow"])
            _append_log(id,"geometry","$(CFDBackend.gurney_point_note(gsec.metadata)). The plate is a baffle (a zero-thickness wall) in the low-speed CFD's C+H mesh, whose quality gate may refuse a height; XFOIL/JXfoil run the clean section as the $(CFDBackend.GURNEY_PANEL_LABEL). Published native-geometry reference overlays are disabled.")
        end
        reference === nothing || _append_log(id,"reference",
            "Matched $(reference["name"]) experimental Cp at the published flow conditions.")
        backends = case["backends"]
        # With the low-speed CFD in the job, XFOIL/JXfoil run the CFD's own section (sharp/closed trailing edge) unless the
        # request says xfoil_geometry = "native". Every job without the low-speed CFD is unchanged.
        paired = nothing
        if gurney_on
            # The Gurney flap: XFOIL and JXfoil cannot model the plate; they run the clean source section, labelled as the
            # no-flap reference (record_point! below), and no geometry is paired.
        elseif "lowspeed" in backends && any(b->b in ("xfoil","jxfoil"), backends) &&
           !CFDBackend._lowspeed_ch_selected(case["flow"]) &&
           String(get(case["flow"],"xfoil_geometry","same_as_cfd")) == "same_as_cfd"
            try
                paired = CFDBackend.lowspeed_paired_geometry(case, geopath, outdir)
            catch err
                _append_log(id, "geometry", "XFOIL/JXfoil keep the native source section: the low-speed CFD section is unavailable ($(sprint(showerror, err))).")
            end
        end
        function publish_results!()
            polar = [Dict{String,Any}("backend"=>get(p,"backend",""), "alpha_deg"=>get(p,"alpha_deg",nothing),
                "cl"=>get(p,"cl",nothing), "cd"=>get(p,"cd",nothing), "cm"=>get(p,"cm",nothing),
                "status"=>get(p,"status","unknown")) for p in results]
            payload = Dict{String,Any}("geometry"=>geom,"results"=>results,
                                       "polar"=>polar,"meshes"=>meshes,
                                       "reference"=>reference)
            case["airfoil"]["kind"]=="cavitation_reference" && (payload["reference_case"]=_reference_setup(case))
            payload["saved_snapshots"]=saved_snapshots
            # Latest committed fields of running CFD points.
            isempty(live_fields) || (payload["live_fields"]=live_fields)
            isempty(lbm_live) || (payload["lbm_live"]=lbm_live)
            isempty(lbm_lattices) || (payload["lbm_lattices"]=lbm_lattices)
            _put_job(id, Dict("result"=>_clean(payload)))
        end
        function record_point!(backend, p; publish=false)
            q = Dict{String,Any}(string(k)=>v for (k,v) in pairs(p))
            q["backend"] = backend
            get(q,"alpha_deg",nothing) isa Real && delete!(live_fields, "$(backend):$(@sprintf("%.6f",Float64(q["alpha_deg"])))")
            if backend == "jxfoil" || (backend=="xfoil" && !haskey(q,"native_geometry"))
                q["native_geometry"] = Dict{String,Any}("name"=>foil.name,
                    "x"=>copy(foil.x),"y"=>copy(foil.y),"role"=>"native source geometry")
            end
            if paired !== nothing && backend in ("xfoil","jxfoil")
                gurney_pair = get(paired,"pairing","") == "gurney_section"
                # The low-speed CFD's own section (like-for-like comparison, 23 Oct 2026), or the Gurney flap section
                q["native_geometry"] = Dict{String,Any}("name"=>paired["name"],"x"=>copy(paired["x"]),"y"=>copy(paired["y"]),
                    "role"=>gurney_pair ? "the Gurney flap section (same as the CFD)" : "same closed-trailing-edge section as the low-speed CFD")
                q["geometry_pairing"] = gurney_pair ? "gurney_section" : "same_as_lowspeed_cfd"
                q["paired_geometry_sha256"] = paired["sha256"]
                gurney_pair && (q["gurney"] = paired["gurney"])
                q["notes"] = vcat(get(q,"notes",String[]) isa AbstractVector ? get(q,"notes",String[]) : String[],
                    ["$(backend=="xfoil" ? "XFOIL" : "JXfoil") geometry: $(paired["label"]); file SHA-256 $(paired["sha256"])."],
                    gurney_pair ? [CFDBackend.gurney_point_note(paired["gurney"]) * ".", CFDBackend.GURNEY_PANEL_CAVEAT] : String[])
                q["message"] = string(get(q,"message",""), gurney_pair ?
                    " $(backend=="xfoil" ? "XFOIL" : "JXfoil") geometry: the Gurney flap section (approximate for a panel/boundary-layer code)." :
                    " $(backend=="xfoil" ? "XFOIL" : "JXfoil") geometry: same closed-trailing-edge section as the low-speed CFD.")
            end
            if gurney_on && backend in ("xfoil","jxfoil")
                q["comparison_role"] = CFDBackend.GURNEY_PANEL_LABEL
                q["series_label"] = "$(backend=="xfoil" ? "XFOIL" : "JXfoil") · $(CFDBackend.GURNEY_PANEL_LABEL)"
                haskey(q,"native_geometry") && (q["native_geometry"]["role"] = "clean section: $(CFDBackend.GURNEY_PANEL_LABEL)")
                q["notes"] = vcat(get(q,"notes",String[]) isa AbstractVector ? get(q,"notes",String[]) : String[], [CFDBackend.GURNEY_PANEL_CAVEAT])
                q["message"] = string(get(q,"message",""), " $(backend=="xfoil" ? "XFOIL" : "JXfoil"): $(CFDBackend.GURNEY_PANEL_LABEL).")
            end
            if backend == "jxfoil" && "transonic" in backends && case["flow"]["mach"] > 1e-6
                q["requested_mach"] = case["flow"]["mach"]
                q["analysis_mach"] = 0.0
                q["comparison_role"] = "incompressible reference"
                q["message"] = string(get(q,"message",""), " JXfoil ran at Mach 0 as an incompressible reference; CFD uses the requested Mach.")
            end
            if backend=="jxfoil" && (haskey(case["flow"],"trip_upper")||haskey(case["flow"],"trip_lower"))
                q["message"]=string(get(q,"message","")," Prescribed trip controls apply to the other selected solvers; this JXfoil clone retains its native transition model.")
            end
            haskey(q,"cm") || (q["cm"] = get(q, "cm_c4", nothing))
            haskey(q,"fields") || (q["fields"] = Dict{String,Any}())
            haskey(q,"status") || (q["status"] = "completed")
            alpha = get(q, "alpha_deg", "?")
            if backend == "transonic" && alpha isa Real && get(q,"mesh",nothing) isa AbstractDict
                meshes[@sprintf("%.6f",Float64(alpha))] = _asset_urls(q["mesh"],id,outdir)
            end
            index = findfirst(r -> get(r,"backend",nothing) == backend &&
                                   get(r,"alpha_deg",nothing) == alpha, results)
            if index === nothing
                push!(results, _asset_urls(q,id,outdir))
            else
                results[index] = _asset_urls(q,id,outdir)
            end
            index === nothing && _append_log(id, backend,
                "alpha=$alpha deg: $(q["status"]); CL=$(get(q,"cl",nothing)), CD=$(get(q,"cd",nothing)), CM=$(get(q,"cm",nothing)).")
            publish && (publish_results!(); _persist_job(id))
        end
        publish_results!()
        for (ib, backend) in enumerate(backends)
            base = (ib-1)/length(backends)
            span = 1/length(backends)
            _put_job(id, Dict("message"=>"Running $backend", "progress"=>base))
            _append_log(id, "server", "Starting $backend for $(length(case["alphas"])) angle(s).")
            if paired !== nothing && backend in ("xfoil","jxfoil")
                _append_log(id, backend, "$(backend=="xfoil" ? "XFOIL" : "JXfoil") geometry: $(paired["label"]) ($(basename(paired["path"])), $(length(paired["x"])) points, SHA-256 $(paired["sha256"])).")
            end
            output = if backend == "jxfoil"
                # In a two-backend job, JXfoil supplies an explicitly labeled
                # incompressible reference while CFD uses the requested Mach.
                reference_only = "transonic" in backends && case["flow"]["mach"] > 1e-6
                AirfoilJXBackend.solve_sweep(paired === nothing ? foil : AirfoilJXBackend.JXfoil.Airfoil(paired["name"], paired["x"], paired["y"]), case["alphas"];
                    reynolds=case["flow"]["reynolds"], ncrit=case["flow"]["ncrit"],
                    mach=reference_only ? 0.0 : case["flow"]["mach"])
            elseif backend=="xfoil"
                AirfoilXfoilBackend.run_case(case,foil,outdir;
                    geometry_override=paired === nothing ? nothing :
                        (geometry=(name=paired["name"],x=paired["x"],y=paired["y"]),
                         method=get(paired,"pairing","")=="gurney_section" ?
                            "the Gurney flap section file $(basename(paired["path"])) ($(length(paired["x"])) of its $(paired["file_points"]) points); actual executable repanels to240 computational panels" :
                            "the low-speed CFD's own section file $(basename(paired["path"])) ($(length(paired["x"])) points); actual executable repanels to240 computational panels",
                         scope=get(paired,"pairing","")=="gurney_section" ?
                            "XFOIL geometry: $(paired["label"]); the CFD's Gurney section (solver computational panelings differ). $(CFDBackend.GURNEY_PANEL_CAVEAT)" :
                            "XFOIL geometry: $(paired["label"]); like-for-like section comparison (solver computational panelings differ)",
                         pairing=get(paired,"pairing","")=="gurney_section" ? "gurney_section" : "same_as_lowspeed_cfd", paired_sha256=paired["sha256"]),
                    log=msg->_append_log(id,backend,msg),progress=ev->begin
                        event=get(ev,"event","");angle=get(ev,"alpha_deg",nothing)
                        idx=something(findfirst(==(angle),case["alphas"]),1)
                        if event in ("point_complete","point_failed")
                            record_point!(backend,ev["point"];publish=true)
                        end
                        event=="running" && _append_log(id,backend,"Actual XFOIL still running at alpha=$angle deg after$(ev["elapsed_seconds"])s")
                        fraction=(idx-1+(event in ("point_complete","point_failed") ? 1 : 0))/length(case["alphas"])
                        _put_job(id,Dict("message"=>"Actual XFOIL at alpha=$angle deg: $(replace(event,'_'=>' '))","progress"=>base+span*fraction))
                    end)
            else
                CFDBackend.run_case(case, backend, outdir; log=msg->_append_log(id, backend, msg), progress=ev->begin
                    event = string(get(ev, "event", "running"))
                    angle = get(ev, "alpha_deg", nothing)
                    idx = findfirst(==(angle), case["alphas"])
                    idx === nothing && (idx = 1)
                    nangle = length(case["alphas"])
                    if event == "iteration"
                        interval=get(case,"display_every",nothing);iteration=get(ev,"iteration",nothing)
                        if interval isa Integer && iteration isa Integer && iteration>1 && iteration%interval!=0 && iteration!=get(ev,"total_iterations",nothing)
                            return nothing
                        end
                        _append_history(id, backend, ev)
                    elseif event == "mesh_ready" && get(ev,"mesh",nothing) isa AbstractDict &&
                           angle isa Real
                        meshes[@sprintf("%.6f",Float64(angle))] =
                            _asset_urls(ev["mesh"],id,outdir)
                        publish_results!()
                    elseif event in ("point_complete", "point_failed") &&
                           get(ev,"point",nothing) isa AbstractDict
                        record_point!(backend, ev["point"]; publish=true)
                    elseif event == "field_snapshot" && angle isa Real
                        entry = Dict{String,Any}(
                            "backend"=>backend, "alpha_deg"=>angle, "iteration"=>get(ev,"iteration",nothing),
                            "saved_at"=>string(now(UTC))*"Z",
                            "artifact"=>_asset_urls(ev["artifact"],id,outdir))
                        live_fields["$(backend):$(@sprintf("%.6f",Float64(angle)))"] = entry
                        push!(saved_snapshots,entry)
                        publish_results!()
                        _persist_job(id)
                        return nothing          # the job message and progress stay those of the iterations
                    elseif event == "lbm_lattice" && angle isa Real && get(ev,"lbm",nothing) isa AbstractDict
                        lbm_lattices[@sprintf("%.6f",Float64(angle))] = _asset_urls(ev["lbm"],id,outdir)
                        publish_results!()
                    elseif event == "lbm_frame" && angle isa Real
                        live = Dict{String,Any}("alpha_deg"=>angle, "iteration"=>get(ev,"iteration",nothing))
                        for k in ("velocity", "vorticity", "numeric")   # pictures ("lbmapp") or numeric fields ("lbmio")
                            haskey(ev, k) && (live[k] = _asset_urls(ev[k],id,outdir))
                        end
                        lbm_live[@sprintf("%.6f",Float64(angle))] = live
                        publish_results!()
                        return nothing
                    end
                    msg = if event == "iteration"
                        "$backend at α=$(angle)°: iteration $(get(ev,"iteration","?"))/$(get(ev,"total_iterations","?")), CL=$(get(ev,"cl","?")), CD=$(get(ev,"cd","?"))"
                    elseif event == "gpu_wait" || (event == "running" && haskey(ev, "gpu_wait"))
                        # the opt-in low-speed GPU run queues for the shared GPU lock (gpu_lock.jl)
                        "$backend at α=$(angle)°: waiting for the GPU (held by $(get(ev, "owner", get(ev, "gpu_wait", "another owner"))))"
                    elseif event == "gpu_acquired"
                        "$backend at α=$(angle)°: GPU lock taken; starting on the GPU"
                    elseif event == "running"
                        "Running $backend at α=$(angle)° ($(get(ev,"elapsed_seconds",0)) s)"
                    elseif event == "point_failed"
                        "$backend at α=$(angle)° failed: $(get(ev,"error","see solver log"))"
                    else
                        "$backend at α=$(angle)°: $(replace(event,'_'=>' '))"
                    end
                    if event == "started"
                        _append_log(id, backend, "Worker process started for alpha=$angle deg (PID $(get(ev,"pid","?"))).")
                    elseif event == "running"
                        _append_log(id, backend, "Still running at alpha=$angle deg after $(get(ev,"elapsed_seconds",0)) s.")
                    end
                    f = if event in ("point_complete", "point_failed")
                        idx/nangle
                    elseif event == "iteration" && get(ev,"total_iterations",0) isa Real &&
                            get(ev,"total_iterations",0) > 0 && get(ev,"iteration",0) isa Real
                        (idx-1 + clamp(ev["iteration"]/ev["total_iterations"],0,1))/nangle
                    else
                        (idx-1)/nangle
                    end
                    _put_job(id, Dict("message"=>msg, "progress"=>base + span*f))
                end)
            end
            for p in _point_list(output, backend)
                p isa AbstractDict || continue
                record_point!(backend, p)
            end
            publish_results!()
            _append_log(id, "server", "Finished $backend.")
        end
        _safe_finalize_outputs!(id,case,get(_get_job(id),"result",Dict{String,Any}()),outdir)
        status, message = _point_outcome(results)
        _put_job(id, Dict("status"=>status, "message"=>message, "progress"=>1.0))
        _append_log(id, "server", "$message: $(length(results)) point record(s).")
    catch err
        _safe_finalize_outputs!(id,case,get(_get_job(id),"result",Dict{String,Any}()),outdir)
        _put_job(id, Dict("status"=>"failed", "message"=>"Analysis failed", "error"=>sprint(showerror,err)))
        _append_log(id, "server", "Analysis failed: $(sprint(showerror, err))")
    end
    _persist_job(id)
end

# ---- saved job records (22 Oct 2026): a finished job's public record survives a server restart ----------------------
# Each job's folder runs/<id>/ receives job_record.msgpack (the public record, its request and the save time) after every
# completed point and at the end, and job_summary.msgpack (a few fields for the Recent results list). The live job table
# stays in memory; a job that is not in it is served read-only from its saved record.
const JOB_RECORD_FILE = "job_record.msgpack"
const JOB_SUMMARY_FILE = "job_summary.msgpack"
_job_request(case) = case isa AbstractDict ? Dict{String,Any}(
    "airfoil"=>Dict{String,Any}(k=>v for (k,v) in get(case,"airfoil",Dict()) if k in ("kind","id","coordinates")),
    "flow"=>get(case,"flow",Dict{String,Any}()),"sweep"=>get(case,"sweep",Dict{String,Any}()),
    "backends"=>get(case,"backends",String[]),"alphas"=>get(case,"alphas",Float64[]),
    "outputs"=>get(case,"outputs",Dict{String,Any}()),"display_every"=>get(case,"display_every",nothing)) : nothing
function _job_summary(job)
    request = _job_request(get(job,"case",nothing))
    results = get(get(job,"result",Dict()),"results",Any[])
    results isa AbstractVector || (results = Any[])
    flow = request === nothing ? Dict{String,Any}() : request["flow"]
    Dict{String,Any}("id"=>job["id"],"status"=>get(job,"status",""),"message"=>get(job,"message",""),
        "created_at"=>get(job,"created_at",nothing),"updated_at"=>get(job,"updated_at",nothing),
        "airfoil"=>request === nothing ? nothing : request["airfoil"],"backends"=>request === nothing ? String[] : request["backends"],
        "alphas"=>request === nothing ? Float64[] : request["alphas"],"reynolds"=>get(flow,"reynolds",nothing),
        "mach"=>get(flow,"mach",nothing),"points"=>length(results),
        "completed_points"=>count(p->p isa AbstractDict && get(p,"status","") in ("completed","converged","ok"),results))
end
function _persist_job(id::String)
    job = _get_job(id)
    job === nothing && return
    try
        dir = _safe_job_dir(id); mkpath(dir)
        record = Dict{String,Any}(k=>v for (k,v) in job if k != "case")
        record["request"] = _job_request(get(job,"case",nothing))
        record["saved_at"] = string(now(UTC))*"Z"; record["record_version"] = 1
        for (file, value) in ((JOB_RECORD_FILE, record), (JOB_SUMMARY_FILE, merge(_job_summary(job), Dict("saved_at"=>record["saved_at"]))))
            CFDBackend.FullFieldArtifact.FieldSnapshots.atomic_write(joinpath(dir,file),encode(_clean(value),:msgpack);reserve_bytes=0)
        end
    catch err
        @warn "Could not save the job record" id exception=err
    end
end
"A saved job record (or nothing). A record saved while its job was running, whose job is no longer live, is marked interrupted."
function _saved_job(id::String)
    path = joinpath(_safe_job_dir(id), JOB_RECORD_FILE)
    isfile(path) || return nothing
    record = decode(read(path), :msgpack)
    record isa AbstractDict || return nothing
    record["restored"] = true
    if get(record,"status","") in ("queued","running")
        record["status"] = "partial"
        record["message"] = "The server stopped while this job was running; the points saved before the stop are shown."
    end
    record
end
function _recent_jobs(limit::Int=20)
    entries = Tuple{Float64,String}[]
    for parent in (RUN_DIR,LEGACY_RUN_DIR)
        isdir(parent) || continue
        for name in readdir(parent)
            occursin(r"^[0-9a-f-]{36}$", name) || continue
            path = joinpath(parent, name, JOB_SUMMARY_FILE)
            isfile(path) && push!(entries, (mtime(path), path))
        end
    end
    sort!(entries; by=first, rev=true)
    items = Any[]
    for (_, path) in entries[1:min(limit, length(entries))]
        try
            summary = decode(read(path), :msgpack)
            # A summary marked "hidden" (set by hand, e.g. a superseded demonstration job) stays on disk and opens by its
            # URL, but is not listed.
            get(summary, "hidden", false) === true && continue
            live = _get_job(String(summary["id"]))
            if live !== nothing
                summary["status"] = get(live,"status",summary["status"]); summary["message"] = get(live,"message",summary["message"])
                summary["live"] = true
            elseif get(summary,"status","") in ("queued","running")
                summary["status"] = "interrupted"
            end
            push!(items, summary)
        catch err
            @warn "Could not read a saved job summary" path exception=err
        end
    end
    items
end
"The log of a job that is not live: the tail of its saved job.log, as the live endpoint's entries."
function _saved_logs(id::String, after::Int, status)
    path = joinpath(_safe_job_dir(id), "job.log")
    lines = isfile(path) ? readlines(path) : String[]
    lines = lines[max(1, end-1999):end]
    entries = Dict{String,Any}[]
    for (seq, raw) in enumerate(lines)
        seq > after || continue
        m = match(r"^\[([^\]]*)\] \[([^\]]*)\] (.*)$", raw)
        push!(entries, Dict{String,Any}("seq"=>seq, "at"=>m === nothing ? "" : m[1], "source"=>m === nothing ? "saved" : m[2],
                                        "line"=>m === nothing ? raw : m[3]))
        length(entries) >= LOG_PAGE_SIZE && break
    end
    Dict{String,Any}("entries"=>entries, "next_seq"=>isempty(entries) ? after : entries[end]["seq"], "dropped"=>0,
                     "status"=>status, "log_url"=>"/api/jobs/$id/files/job.log")
end
function _worker()
    for id in QUEUE
        _run_job(id)
    end
end
function _start_worker()
    WORKER_STARTED[] && return
    WORKER_STARTED[] = true
    Threads.@spawn _worker()
end

function _static_response(path::String)
    filename = path == "/" ? "index.html" : replace(path, r"^/static/" => "", count=1)
    startswith(filename, "/") && (filename = filename[2:end])
    asset = get(STATIC_FILES, filename, nothing)
    asset === nothing && return HTTP.Response(404, "Not found")
    full = joinpath(STATIC_DIR, asset)
    isfile(full) || return HTTP.Response(404, "Not found")
    HTTP.Response(200, ["Content-Type"=>get(MIME, lowercase(splitext(full)[2]),"application/octet-stream"),
        "Cache-Control"=>"no-cache"], read(full))
end

function _solver_guide_response(filename::String, mime::String)
    full = joinpath(SOLVER_GUIDE_DIR, filename)
    isfile(full) || return HTTP.Response(404, "Not found")
    HTTP.Response(200, ["Content-Type"=>mime, "Cache-Control"=>"no-cache"], read(full))
end
function _asset_response(id::String, rel::String; range::AbstractString="")
    dir = _safe_job_dir(id)
    full = abspath(normpath(joinpath(dir, replace(rel, '/'=>'\\'))))
    root = abspath(dir)
    startswith(lowercase(full), lowercase(root * "\\")) || return HTTP.Response(403, "Forbidden")
    isfile(full) || return HTTP.Response(404, "Not found")
    # Byte ranges for the LBM 3D videos (27 Oct 2026): browsers seek an MP4 with Range requests. Every other file unchanged.
    m = endswith(lowercase(full), ".mp4") ? match(r"^bytes=(\d*)-(\d*)$", strip(range)) : nothing
    if m !== nothing && !(isempty(m.captures[1]) && isempty(m.captures[2]))
        n = filesize(full)
        a = isempty(m.captures[1]) ? max(0, n - parse(Int, m.captures[2])) : parse(Int, m.captures[1])
        b = (isempty(m.captures[1]) || isempty(m.captures[2])) ? n - 1 : min(n - 1, parse(Int, m.captures[2]))
        a <= b || return HTTP.Response(416, ["Content-Range"=>"bytes */$n"], UInt8[])
        data = open(io -> (seek(io, a); read(io, b - a + 1)), full)
        return HTTP.Response(206, ["Content-Type"=>"video/mp4", "Accept-Ranges"=>"bytes", "Content-Range"=>"bytes $a-$b/$n",
            "Cache-Control"=>"no-store"], data)
    end
    HTTP.Response(200, ["Content-Type"=>get(MIME,lowercase(splitext(full)[2]),"application/octet-stream"),
        "Cache-Control"=>"no-store"], read(full))
end

include(joinpath("api", "studies.jl"))

function app_handler(req::HTTP.Request)
    uri = HTTP.URI(req.target)
    path = String(uri.path); method = String(req.method)
    startswith(path,"/common-static/") && method == "GET" && return WorkbenchStatic.response(path)
    fmt = _fmt(req)
    try
        if startswith(path,"/api/airfoil/studies/")
            return _airfoil_study_route(req,path)
        elseif method == "GET" && path == "/api/airfoil/benchmarks"
            return _study_json(200,Dict("items"=>AirfoilReferenceData.benchmark_catalog()))
        elseif method == "POST" && path == "/api/select-folder"
            payload=NativeDialogs.choose_folder(req,joinpath(PRIVATE_ROOT,"TEMP_DELETEABLE","native_dialogs"))
            return HTTP.Response(200,["Content-Type"=>"application/json; charset=utf-8","Cache-Control"=>"no-store"],JSON3.write(payload))
        elseif method == "GET" && path == "/api/output-folders"
            payload=_output_folders(String(get(HTTP.queryparams(uri),"path","")))
            return HTTP.Response(200,["Content-Type"=>"application/json; charset=utf-8","Cache-Control"=>"no-store"],JSON3.write(payload))
        elseif method == "GET" && path == "/api/health"
            return _response(200, Dict("ok"=>true,"api_revision"=>20,"protocols"=>["msgpack","bson"],
                "backends"=>["jxfoil","xfoil","transonic","cavitation","lowspeed","lbm"],"native_solver_platform_supported"=>!Sys.isapple(),"xfoil"=>AirfoilXfoilBackend.capabilities(),
                "lbm"=>CFDBackend.lbm_capabilities(), "output_capabilities"=>_output_capabilities(),
                "cavitation"=>CFDBackend.cavitation_capabilities(),"lowspeed"=>CFDBackend.lowspeed_capabilities(),
                "gurney"=>CFDBackend.gurney_capabilities(),
                "streamline"=>Dict("label"=>CFDBackend.STREAMLINE_DEV_LABEL, "backends"=>["lowspeed","transonic"], "admission"=>["strict","demo"], "modes"=>["viscous","inviscid"])), fmt)
        elseif method == "GET" && path == "/api/campaign"
            return _response(200,_campaign_cached_status(),fmt)
        elseif method == "GET" && startswith(path,"/api/campaign/files/")
            return _campaign_report_asset(String(path[length("/api/campaign/files/")+1:end]))
        elseif method == "GET" && path == "/api/airfoils"
            q = String(get(HTTP.queryparams(uri), "q", ""))
            items = AirfoilJXBackend.airfoil_catalog(; query=q, limit=2000)
            return _response(200, Dict("items"=>items,"count"=>length(items)), fmt)
        elseif method == "GET" && path == "/api/cavitation-references"
            items=CavitationReferences.catalog()
            return _response(200,Dict("items"=>items,"count"=>length(items)),fmt)
        elseif method == "POST" && path == "/api/preview"
            ctype = wire_format(_header(req,"Content-Type"))
            fmt = _fmt(req;default=ctype)
            length(req.body) <= AirfoilWire.MAX_BYTES || return _error(413,"payload too large",fmt)
            case, want_mesh = _preview_request(decode(Vector{UInt8}(req.body),ctype))
            # Waiting yields the HTTP task; CPU mesh generation can run on a
            # worker thread while the application serves diagnostics requests.
            pending = Threads.@spawn try
                _make_preview(case,want_mesh)
            catch err
                err
            end
            preview = fetch(pending)
            preview isa Exception && throw(preview)
            return _response(200,preview,fmt)
        elseif method == "GET" && startswith(path,"/api/previews/")
            parts = split(path,'/';keepempty=false)
            length(parts) >= 5 && parts[4] == "files" || return _error(404,"Not found",fmt)
            return _asset_response(String(parts[3]),join(parts[5:end],"/"))
        elseif method == "POST" && path == "/api/jobs"
            ctype = wire_format(_header(req,"Content-Type"))
            fmt = _fmt(req; default=ctype)
            length(req.body) <= AirfoilWire.MAX_BYTES || return _error(413,"payload too large",fmt)
            case = _validated_case(decode(Vector{UInt8}(req.body), ctype))
            id = string(uuid4())
            job = Dict{String,Any}("id"=>id,"status"=>"queued","message"=>"Queued", "progress"=>0.0,
                "created_at"=>string(now(UTC))*"Z", "case"=>case,"output_settings"=>case["outputs"])
            lock(JOB_LOCK) do
                length(JOBS) < MAX_JOBS || throw(ArgumentError("job capacity reached; restart after downloading results"))
                JOBS[id] = job
                JOB_LOGS[id] = Dict{String,Any}[]
                JOB_LOG_SEQ[id] = 0
                JOB_HISTORY[id] = Dict{String,Any}[]
                JOB_HISTORY_SEQ[id] = 0
            end
            _append_log(id, "server", "Job queued with backend(s): $(join(case["backends"], ", ")).")
            _persist_job(id)
            _start_worker(); put!(QUEUE,id)
            return _response(202, Dict("id"=>id,"status"=>"queued","message"=>"Queued"),fmt)
        elseif method == "GET" && startswith(path,"/api/jobs/")
            parts = split(path,'/'; keepempty=false)
            length(parts) >= 3 || return _error(404,"Not found",fmt)
            id = String(parts[3])
            # Saved binary/CSV fields survive a Julia restart even though the
            # live job table is in memory. UUID and path confinement are still
            # enforced by _asset_response and _safe_job_dir.
            if length(parts) >= 5 && parts[4] == "files"
                return _asset_response(id, join(parts[5:end],"/"); range=_header(req,"Range"))
            end
            job = _get_job(id)
            if job === nothing
                # Not live (for example after a server restart): serve the saved record read-only.
                occursin(r"^[0-9a-f-]{36}$", id) || return _error(404,"Unknown job",fmt)
                saved = _saved_job(id)
                saved === nothing && return _error(404,"Unknown job",fmt)
                if length(parts) == 4 && parts[4] == "logs"
                    after = parse(Int, String(get(HTTP.queryparams(uri), "after", "0")))
                    return _response(200, _saved_logs(id, after, saved["status"]), fmt)
                elseif length(parts) == 4 && parts[4] == "history"
                    return _response(200, Dict{String,Any}("entries"=>Any[], "next_seq"=>0, "dropped"=>0, "latest_seq"=>0,
                                                           "status"=>saved["status"]), fmt)
                elseif length(parts) == 3
                    nearby = AirfoilNearbyReferences.nearby_experiments(get(saved,"request",nothing))
                    isempty(nearby) || (saved["nearby_experiments"] = nearby)
                    return _response(200, saved, fmt)
                end
                return _error(404,"Not found",fmt)
            end
            if length(parts) == 4 && parts[4] == "logs"
                after = parse(Int, String(get(HTTP.queryparams(uri), "after", "0")))
                return _response(200, _get_logs(id, after), fmt)
            elseif length(parts) == 4 && parts[4] == "history"
                after = parse(Int, String(get(HTTP.queryparams(uri), "after", "0")))
                return _response(200, _get_history(id, after), fmt)
            elseif length(parts) == 3
                public = Dict(k=>v for (k,v) in job if k != "case")
                public["request"] = _job_request(get(job,"case",nothing))
                # Wind-tunnel lift at nearby conditions (NACA 0012 with the low-speed CFD only; never a matched reference)
                nearby = AirfoilNearbyReferences.nearby_experiments(public["request"])
                isempty(nearby) || (public["nearby_experiments"] = nearby)
                return _response(200, public, fmt)
            end
            return _error(404,"Not found",fmt)
        elseif method == "GET" && path == "/api/recent"
            items = _recent_jobs(20)
            return _response(200, Dict("items"=>items, "count"=>length(items)), fmt)
        elseif method == "GET" && path in ("/solvers", "/solvers/")
            return _solver_guide_response("solver_catalog.html", "text/html; charset=utf-8")
        elseif method == "GET" && path == "/workbench-static/solver_catalog.css"
            return _solver_guide_response("solver_catalog.css", "text/css; charset=utf-8")
        elseif method == "GET" && (path == "/" || startswith(path,"/static/") || path in ("/index.html","/app.js","/app.css","/favicon.svg"))
            return _static_response(path)
        end
        _error(404,"Not found",fmt)
    catch err
        code = err isa ArgumentError || err isa EOFError ? 400 : 500
        if path in ("/api/output-folders","/api/select-folder")
            return HTTP.Response(code,["Content-Type"=>"application/json; charset=utf-8","Cache-Control"=>"no-store"],JSON3.write(Dict("ok"=>false,"error"=>sprint(showerror,err))))
        end
        _error(code, sprint(showerror,err),fmt)
    end
end

function serve(; host::String="127.0.0.1", port::Int=8058,
    on_listening::Function=()->nothing)
    host in ("127.0.0.1","localhost","::1") || throw(ArgumentError("app binds to loopback only"))
    mkpath(RUN_DIR)
    _start_worker()
    server = HTTP.serve!(app_handler, host, port; verbose=false)
    println("Airfoil Web at http://$host:$port")
    if get(ENV,"AIRFOIL_WEB_WARM_PREVIEW","1") != "0"
        println("Preparing geometry preview in Julia...")
        flush(stdout)
        try
            preview_case, _ = _preview_request(Dict("airfoil"=>Dict("kind"=>"naca4","id"=>"2412")))
            _make_preview(preview_case,false)
            println("Geometry preview ready.")
            flush(stdout)
        catch err
            @warn "Geometry preview warmup failed; selections can be retried in the web app" exception=err
        end
    end
    try
        on_listening()
    catch err
        @warn "The browser could not be opened automatically; the web app is still running" exception=err
    end
    wait(server)
end

end
