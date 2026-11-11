# App orchestration of the "lowspeed" backend: version 2's pressure-based incompressible solver (PBCav) in free air.
# Included inside CFDBackend. The HTTP process loads no PBCav code: each point runs lowspeed_worker.jl in a separate
# Julia process from a manifest-verified vendored PBCav snapshot, or a historical worktree.
# S1030_FLAPREF is based on a43013d1 with the local changes recorded in its manifest. PB_BAFFLE_REFINE adds tip and
# flap-height wake rows and thins the finite-TE H block only when requested with a Gurney baffle. S1028_APPBAF at af9beda1
# and S1024_A2_JL113 at b7a7b29b remain available for historical tests.

isdefined(@__MODULE__, :RANSConvergence) ||
    include(joinpath(@__DIR__, "..", "backends", "rans_convergence.jl"))

"The PBCav snapshots and their upstream base commits, with the manifest of each. The vendored snapshot also has separately
recorded local changes. The pin
in force is AIRFOIL_LOWSPEED_PIN (default s1030). AIRFOIL_LOWSPEED_SNAPSHOT overrides the path of the snapshot for a test (no
manifest check); the commit is then read from that worktree and recorded."
const LOWSPEED_VENDOR_SNAPSHOT = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "solvers", "pbcav_s1030"))
const LOWSPEED_LEGACY_ROOT = normpath(get(ENV, "LUDWIG_LEGACY_SNAPSHOT_ROOT",
    joinpath(@__DIR__, "..", "..", "..", "..", "..", "02_PRIVATE_PROJECT_DEVELOPMENT", "legacy_snapshots")))
const LOWSPEED_PINS = Dict(
    # 30 Oct 2026: opt-in flap tip, root and wake redistribution (PB_BAFFLE_REFINE); all other knobs keep their defaults
    "s1030"=>(snapshot=LOWSPEED_VENDOR_SNAPSHOT, commit="a43013d1", manifest="lowspeed_snapshot_manifest_s1030.toml"),
    # 28 Oct 2026: the baffle snapshot, pbcav af9beda1 (interior walls, PB_BAFFLE; the ramp start; PB_TURB_MUT_BASE)
    "s1028"=>(snapshot=joinpath(LOWSPEED_LEGACY_ROOT, "S1028_APPBAF"), commit="af9beda1", manifest="lowspeed_snapshot_manifest.toml"),
    # 24 Oct 2026, the previous pin (no baffle support): reachable for a test only, with its own manifest and the same checks
    "s1024"=>(snapshot=joinpath(LOWSPEED_LEGACY_ROOT, "S1024_A2_JL113"), commit="b7a7b29b", manifest="lowspeed_snapshot_manifest_s1024_b7a7b29b.toml"))
"""The pin in force: AIRFOIL_LOWSPEED_PIN (s1030 by default; s1028 and s1024 for historical tests). Unknown names are refused."""
function lowspeed_pin()
    name = get(ENV, "AIRFOIL_LOWSPEED_PIN", "s1030")
    haskey(LOWSPEED_PINS, name) || throw(ArgumentError("AIRFOIL_LOWSPEED_PIN must be one of $(join(sort(collect(keys(LOWSPEED_PINS))), ", ")) (got $name)"))
    LOWSPEED_PINS[name]
end
const LOWSPEED_SNAPSHOT_DEFAULT = LOWSPEED_PINS["s1030"].snapshot
const LOWSPEED_COMMIT_PINNED = LOWSPEED_PINS["s1030"].commit
"""The merged head's W6b settings and the measured CPU launch mode, named explicitly in every low-speed point.
The GPU override keeps its registered fixed launch mode."""
const LOWSPEED_W6B_KNOBS = ("PB_CPL_AU"=>"0.8", "PB_CPL_VISC_NONORTH"=>"jacobian", "PB_TURB_OMEGA_BOUND"=>"0.2", "PB_CPU_LAUNCH"=>"balanced")
lowspeed_snapshot() = normpath(get(ENV, "AIRFOIL_LOWSPEED_SNAPSHOT", lowspeed_pin().snapshot))
"""The GPU option (opt-in request key lowspeed_gpu, 24 Oct 2026; the CPU stays the default): the snapshot's CUDA backend with
W7's adopted speed settings for CUDA free air (PB_SPEED=auto: PB_SYNC=reads, PB_GRAPHS=on, PB_CPL_GS=cgs2,
PB_LINE_SOLVER=pcr, PB_DOT=tree, PB_CPU_LAUNCH=fixed; the preconditioner and k-omega stay Float64 -- the Float32 forms are
gated by the snapshot, knobs.jl SPEED_GATED_F32). The worker takes the shared GPU lock (gpu_lock.jl,
the private GPU lock) before the march and waits while another owner holds it."""
const LOWSPEED_GPU_KNOBS = ("PB_BACKEND"=>"cuda", "PB_SPEED"=>"auto", "PB_CPU_LAUNCH"=>"fixed")
"The request's GPU choice: lowspeed_gpu must be a boolean when present (absent: the CPU)."
function lowspeed_gpu(flow::AbstractDict)
    haskey(flow, "lowspeed_gpu") || return false
    flow["lowspeed_gpu"] isa Bool || throw(ArgumentError("lowspeed_gpu must be boolean"))
    flow["lowspeed_gpu"]
end
"The hardware line of a point's log: the CPU threads, or the GPU after its lock."
lowspeed_hardware(ls) = get(ls, "gpu", false) ?
    @sprintf("on the GPU (CUDA; PB_SPEED=auto, W7's adopted CUDA free-air settings; it queues for the shared GPU lock), %d CPU threads", LOWSPEED_THREADS) :
    @sprintf("on %d CPU threads", LOWSPEED_THREADS)

"Mesh presets: PB_MESH = n_foil,n_normal,dn1,growth,smooth,n_wake (sharp-edged free-air C-mesh, version 1's generator)."
const LOWSPEED_MESHES = Dict(
    "demo"=>(mesh="192,72,3e-5,1.20,80,64", le_ds=0.0,
              label="Demo (fast): 192 foil x 72 normal cells + 2 x 64 wake, 23,040 cells, first wall spacing 3e-5 c"),
    "fine"=>(mesh="384,128,7e-6,1.18,80,96", le_ds=2e-4,
              label="Fine stall study: 384 foil x 128 remapped normal cells + 2 x 96 wake, first wall spacing 7e-6 c, leading-edge interval 2e-4 c"),
    # API-only smoke mesh for the app's own tests; coarse wall (y+ >> 1 at high Re): not a result mesh.
    "smoke"=>(mesh="64,40,1e-3,1.30,40,16", le_ds=0.0,
               label="Smoke test only: 64 foil x 40 normal cells + 2 x 16 wake, 3,840 cells, first wall spacing 1e-3 c"))
const LOWSPEED_THREADS = parse(Int, get(ENV, "AIRFOIL_LOWSPEED_THREADS", "8"))   # 8; a lower value only on request (shared machine)
const LOWSPEED_DEFAULTS = Dict{String,Any}("lowspeed_resolution"=>"demo", "lowspeed_tu"=>0.01, "lowspeed_mut_ratio"=>10.0,
    "lowspeed_wall_cap_s"=>3600.0, "lowspeed_max_steps"=>2000, "lowspeed_live_every"=>25,
    "lowspeed_steady_tol"=>1e-6, "lowspeed_turb_steady_tol"=>1e-7)

"""Validated low-speed settings from a request's flow map (ArgumentError on a bad value). Mach is not an input.
A Gurney flap (gurney_height) or the hidden key lowspeed_topology = "ch_native" selects the C+H path
(lowspeed_ch_configuration); every other request is the sharp-trailing-edge path, unchanged."""
function lowspeed_configuration(flow::AbstractDict)
    _lowspeed_ch_selected(flow) && return lowspeed_ch_configuration(flow)
    for key in ("lowspeed_start", "lowspeed_turb_mut_base")     # 28 Oct 2026: hidden C+H controls (the Gurney baffle's start and mu_t base)
        haskey(flow, key) && throw(ArgumentError("$key applies to the low-speed CFD's C+H path only (a Gurney flap, or lowspeed_topology = ch_native)"))
    end
    res = String(_get(flow, "lowspeed_resolution", LOWSPEED_DEFAULTS["lowspeed_resolution"]))
    haskey(LOWSPEED_MESHES, res) || throw(ArgumentError("lowspeed_resolution must be demo or fine (or smoke for tests)"))
    re = _number(flow, "reynolds", 3e6; lo=1e4, hi=1e8)
    tu = _number(flow, "lowspeed_tu", LOWSPEED_DEFAULTS["lowspeed_tu"]; lo=1e-6, hi=0.19)
    mr = _number(flow, "lowspeed_mut_ratio", LOWSPEED_DEFAULTS["lowspeed_mut_ratio"]; lo=1e-4, hi=1e4)
    steady_tol = _number(flow, "lowspeed_steady_tol", LOWSPEED_DEFAULTS["lowspeed_steady_tol"]; lo=1e-9, hi=1e-3)
    turb_steady_tol = _number(flow, "lowspeed_turb_steady_tol", LOWSPEED_DEFAULTS["lowspeed_turb_steady_tol"]; lo=1e-9, hi=1e-3)
    cap = _number(flow, "lowspeed_wall_cap_s", LOWSPEED_DEFAULTS["lowspeed_wall_cap_s"]; lo=30.0, hi=7200.0)
    ms = _integer(flow, "lowspeed_max_steps", LOWSPEED_DEFAULTS["lowspeed_max_steps"]; lo=1, hi=20000)
    _number(flow, "cfd_te_thickness", 0.0; lo=0.0, hi=0.05) > 0 && throw(ArgumentError(
        "the low-speed CFD uses a sharp-trailing-edge free-air C-mesh: clear the CFD trailing-edge thickness override (cfd_te_thickness) to run it"))
    live = _integer(flow, "lowspeed_live_every", LOWSPEED_DEFAULTS["lowspeed_live_every"]; lo=0, hi=1000000)
    accel = RANSAcceleration.policy(_boolean(flow,"rans_acceleration",false),:lowspeed)
    Dict{String,Any}("resolution"=>res, "mesh"=>LOWSPEED_MESHES[res].mesh, "le_ds"=>LOWSPEED_MESHES[res].le_ds,
        "snapshot_interval_seconds"=>_number(flow,"snapshot_interval_seconds",600.;lo=0,hi=86400),
        "mesh_label"=>LOWSPEED_MESHES[res].label,
        "reynolds"=>re, "tu"=>tu, "mut_ratio"=>mr, "steady_tol"=>steady_tol, "turb_steady_tol"=>turb_steady_tol,
        "wall_cap"=>cap, "max_steps"=>ms, "live_every"=>live, "gpu"=>lowspeed_gpu(flow),
        "rans_acceleration"=>accel.enabled, "rans_acceleration_method"=>accel.method, "pseudo_dt"=>accel.pseudo_dt)
end

lowspeed_capabilities() = Dict{String,Any}("label"=>"Low-speed CFD (pressure-based, new)",
    "solver"=>"PBCav (version 2, in development, not yet validated)",
    "commit"=>haskey(ENV, "AIRFOIL_LOWSPEED_SNAPSHOT") ? "external snapshot" : lowspeed_pin().commit,
    "revision"=>haskey(ENV, "AIRFOIL_LOWSPEED_SNAPSHOT") ? "external snapshot; content is not fingerprint-checked" :
        lowspeed_snapshot() == LOWSPEED_VENDOR_SNAPSHOT ?
            get(TOML.parsefile(joinpath(@__DIR__, "..", "..", "manifests", lowspeed_pin().manifest)), "revision", lowspeed_pin().commit) : lowspeed_pin().commit,
    "snapshot"=>lowspeed_snapshot(), "available"=>isfile(joinpath(lowspeed_snapshot(), "src", "pbcav", "PBCav.jl")),
    "turbulence"=>"fully turbulent k-omega SST (SST-2003m)", "threads"=>LOWSPEED_THREADS, "gpu"=>false,
    "gpu_option"=>Dict{String,Any}("request_key"=>"lowspeed_gpu", "default"=>false, "knobs"=>Dict(LOWSPEED_GPU_KNOBS),
        "lock"=>AirfoilGPULock.LOCK_DIR, "note"=>"opt-in; the worker waits for the shared GPU lock and never breaks it"),
    "mach"=>"ignored (incompressible)", "resolutions"=>Dict(k=>v.label for (k, v) in LOWSPEED_MESHES),
    "defaults"=>LOWSPEED_DEFAULTS)

function _lowspeed_snapshot_mismatches(snap::AbstractString, manifest::AbstractDict; vendored::Bool)
    [f for (f, h) in manifest["sha256"] if !(vendored &&
        (startswith(f, "src/pbcav/test/") || startswith(basename(f), "test_"))) &&
        (!isfile(joinpath(snap, f)) || bytes2hex(open(sha256, joinpath(snap, f))) != h)]
end

"""The snapshot's HEAD (full hash) and, unless the path was overridden for a test, the checks of the pin: HEAD at the pinned
commit and every solver file of the pin's manifest (sha256) unmodified. `override` is true when the path came from an
environment override (AIRFOIL_LOWSPEED_SNAPSHOT / AIRFOIL_LOWSPEED_CH_SNAPSHOT): then the commit is read and recorded, nothing
is compared."""
function _lowspeed_snapshot_verify(snap::AbstractString, override::Bool, what::AbstractString)
    pin = lowspeed_pin()
    isfile(joinpath(snap, "src", "pbcav", "PBCav.jl")) || throw(ArgumentError(
        "the $what snapshot is missing: $snap (create it with git worktree add --detach $snap $(pin.commit))"))
    # HEAD from the worktree's own metadata (no git executable needed): .git is "gitdir: <dir>", <dir>/HEAD the hash
    vendored = normpath(snap) == LOWSPEED_VENDOR_SNAPSHOT
    gitfile = joinpath(snap, ".git")
    head = if vendored
        String(TOML.parsefile(joinpath(@__DIR__, "..", "..", "manifests", pin.manifest))["commit"])
    elseif isfile(gitfile)
        gd = strip(replace(read(gitfile, String), "gitdir:" => ""))
        strip(read(joinpath(gd, "HEAD"), String))
    else
        strip(read(`git -C $snap rev-parse HEAD`, String))
    end
    revision = String(head)
    if !override
        startswith(head, pin.commit) || throw(ArgumentError("the $what snapshot $snap is at $head, not the pinned $(pin.commit)"))
        # every solver file must still have the content recorded when the snapshot was pinned (the pin's manifest)
        m = TOML.parsefile(joinpath(@__DIR__, "..", "..", "manifests", pin.manifest))
        m["commit"] == head || throw(ArgumentError("the snapshot manifest $(pin.manifest) is for $(m["commit"]), the snapshot is at $head"))
        changed = _lowspeed_snapshot_mismatches(snap, m; vendored)
        isempty(changed) || throw(ArgumentError("the $what snapshot $snap has modified solver files ($(join(sort(changed), ", "))); refusing to run"))
        # The vendored tree contains documented changes on top of the original
        # upstream commit. Preserve both in saved run provenance.
        vendored && (revision = String(get(m, "revision", head)))
    end
    snap, revision
end
"The snapshot's commit (full hash) and whether its tracked files are unmodified; refuses a snapshot whose HEAD is not the pin."
_lowspeed_snapshot_check() = _lowspeed_snapshot_verify(lowspeed_snapshot(), haskey(ENV, "AIRFOIL_LOWSPEED_SNAPSHOT"), "low-speed solver")

"NACA four-digit section with the CLOSED-trailing-edge thickness coefficient (-0.1036), unit chord, Selig order."
function _naca4_closed_te_dat(code::AbstractString, path::AbstractString; n::Int=601)
    occursin(r"^[0-9]{4}$", code) || throw(ArgumentError("invalid NACA four-digit code"))
    m = parse(Int, code[1:1])/100; p = parse(Int, code[2:2])/10; t = parse(Int, code[3:4])/100
    t > 0 || throw(ArgumentError("NACA $code has zero thickness"))
    beta = range(0, pi; length=n); xs = @. 0.5*(1 - cos(beta))
    yt(x) = 5t*(0.2969*sqrt(x) - 0.1260x - 0.3516x^2 + 0.2843x^3 - 0.1036x^4)
    yc(x) = m == 0 ? 0.0 : x < p ? m/p^2*(2p*x - x^2) : m/(1 - p)^2*((1 - 2p) + 2p*x - x^2)
    dyc(x) = m == 0 ? 0.0 : x < p ? 2m/p^2*(p - x) : 2m/(1 - p)^2*(p - x)
    up = [(x - yt(x)*sin(atan(dyc(x))), yc(x) + yt(x)*cos(atan(dyc(x)))) for x in xs]
    lo = [(x + yt(x)*sin(atan(dyc(x))), yc(x) - yt(x)*cos(atan(dyc(x)))) for x in xs]
    open(path, "w") do io
        println(io, "NACA $code closed-trailing-edge form (thickness coefficient -0.1036), unit chord, $(2n-1) points")
        for (x, y) in reverse(up); @printf(io, "%.12f %.12f\n", x, y); end
        for (x, y) in lo[2:end]; @printf(io, "%.12f %.12f\n", x, y); end
    end
    path
end

"""The CFD geometry the free-air C-mesh needs (a sharp trailing edge) and its recorded transform. Never silent:
NACA 0012 uses the NASA TMR sharp section file in the snapshot; other NACA four-digit sections use the closed-TE equation;
a coordinate file with a gap above 1e-8 c and at most 0.01 c uses the app's sharp-TE transform; a larger gap is refused."""
function _lowspeed_geometry(case, path::AbstractString, dir::AbstractString, snap::AbstractString)
    source = _get(case, "airfoil", Dict())
    kind = String(_get(source, "kind", "")); id = String(_get(source, "id", ""))
    if kind == "naca4" && id == "0012"
        f = joinpath(snap, "tools", "naca0012_tmr_sharp_app1201.dat")
        return f, Dict{String,Any}("transform"=>"NASA TMR NACA 0012 sharp section (the 4-digit equation extended to x = 1.00893 and scaled to unit chord, t/c 0.118972; 1,201 points)",
            "change"=>"CFD geometry change: trailing edge closed (NASA TMR NACA 0012 sharp section instead of the open-TE 4-digit equation)",
            "path"=>f, "source_te_gap"=>0.00252, "cfd_te_gap"=>0.0, "mesh_topology"=>"sharp_C")
    elseif kind == "naca4"
        f = _naca4_closed_te_dat(id, joinpath(dir, "airfoil_cfd_closed_te.dat"))
        return f, Dict{String,Any}("transform"=>"NACA $id closed-trailing-edge thickness form (coefficient -0.1036 instead of -0.1015)",
            "change"=>"CFD geometry change: trailing edge closed (NACA closed-TE thickness coefficient -0.1036)",
            "path"=>f, "source_te_gap"=>0.021*parse(Int, id[3:4])/100, "cfd_te_gap"=>0.0, "mesh_topology"=>"sharp_C")
    end
    af = CMeshGen.read_airfoil_dat(path)
    gap = hypot(af.x[1] - af.x[end], af.y[1] - af.y[end])
    gap <= 0.01 || throw(ArgumentError(@sprintf("the low-speed CFD free-air mesh needs a sharp trailing edge; this section's trailing-edge gap is %.4f c (above the 0.01 c the app will close). Refused rather than changing the geometry substantially; use XFOIL or the compressible CFD (C+H) for this section.", gap)))
    if gap <= 1e-8
        f = joinpath(dir, "airfoil_cfd.dat")
        open(f, "w") do io
            println(io, "CFD section (source points, Selig order) of ", basename(path))
            for q in reverse(eachindex(af.x)); @printf(io, "%.17g %.17g\n", af.x[q], af.y[q]); end
        end
        return f, Dict{String,Any}("transform"=>"none", "change"=>"", "path"=>f, "source_te_gap"=>gap, "cfd_te_gap"=>gap,
                                   "mesh_topology"=>"sharp_C")
    end
    f, meta = _sharp_te_geometry(path, dir)
    meta["change"] = @sprintf("CFD geometry change: trailing edge closed (%s; source gap %.5f c)", meta["transform"], gap)
    meta["mesh_topology"] = "sharp_C"
    f, meta
end

"""The section the low-speed CFD runs, for XFOIL/JXfoil in the same job (23 Oct 2026: XFOIL on the native blunt NACA 0012
gives CL 0.567 at alpha 5, on the closed/sharp section 0.542; comparing across the two sections was not like for like).
Returns a Dict (path, sha256, name, x, y in Selig order, transform, label) or throws the CFD's own refusal."""
function lowspeed_paired_geometry(case, path::AbstractString, outdir::AbstractString)
    snap, _ = _lowspeed_snapshot_check()
    dir = mkpath(joinpath(outdir, "paired_geometry"))
    f, meta = _lowspeed_geometry(case, path, dir, snap)
    x = Float64[]; y = Float64[]; name = ""
    for line in eachline(f)
        v = split(strip(line))
        xy = length(v) >= 2 ? tryparse.(Float64, v[1:2]) : (nothing, nothing)
        if any(isnothing, xy)
            isempty(name) && !isempty(strip(line)) && (name = String(strip(line)))
            continue
        end
        push!(x, xy[1]); push!(y, xy[2])
    end
    length(x) >= 20 || throw(ArgumentError("the low-speed CFD section file has too few points"))
    x[1] > 0.5 || throw(ArgumentError("the low-speed CFD section file is not in Selig order"))
    # XFOIL's spline buffer overflows above a few hundred input points (SPLIND: array overflow at 1,201): keep every s-th
    # point of the same file, with both trailing-edge points and the leading-edge point, at most 401 points.
    n = length(x); s = max(1, cld(n - 1, 400)); ile = argmin(x)
    keep = sort!(unique(vcat(collect(1:s:n), [1, n, ile])))
    Dict{String,Any}("path"=>f, "sha256"=>bytes2hex(open(sha256, f)), "name"=>isempty(name) ? basename(f) : name,
        "x"=>x[keep], "y"=>y[keep], "file_points"=>n, "stride"=>s, "transform"=>meta["transform"],
        "label"=>"same closed-trailing-edge section as the low-speed CFD ($(meta["transform"]))" *
                 (s > 1 ? "; every $(s) of its $(n) points ($(length(keep)) points) for the panel codes" : ""))
end

"The worker's input for one point (or, with mesh_only, for Create mesh: the same knobs, so the same mesh node for node)."
function _lowspeed_input(snap, head, foil, ls, alpha, dir; mesh_only::Bool=false, warm_start=nothing)
    cfg = Dict{String,Any}("snapshot"=>snap, "commit"=>head, "foil"=>replace(String(foil), '\\'=>'/'),
        "mesh"=>ls["mesh"], "le_ds"=>ls["le_ds"], "resolution"=>ls["resolution"], "tu"=>ls["tu"], "mut_ratio"=>ls["mut_ratio"],
        "dt"=>ls["pseudo_dt"], "steady_tol"=>ls["steady_tol"], "turb_steady_tol"=>ls["turb_steady_tol"], "max_steps"=>ls["max_steps"], "wall_cap"=>ls["wall_cap"],
        "rans_acceleration"=>ls["rans_acceleration"], "rans_acceleration_method"=>ls["rans_acceleration_method"],
        "n_outer_max"=>40, "vortex_every"=>10, "vortex_relax"=>0.5, "reynolds"=>ls["reynolds"], "alpha"=>Float64(alpha),
        "outdir"=>replace(String(dir), '\\'=>'/'), "live_every"=>ls["live_every"],
        "snapshot_interval_seconds"=>get(ls,"snapshot_interval_seconds",600.),
        "knobs"=>Dict{String,Any}(k=>v for (k, v) in LOWSPEED_W6B_KNOBS))
    get(ls, "gpu", false) && merge!(cfg["knobs"], Dict{String,Any}(LOWSPEED_GPU_KNOBS))   # the GPU (opt-in); absent: unchanged
    if ls["resolution"] == "fine"
        merge!(cfg["knobs"], Dict{String,Any}(
            "PB_MESH_REMAP"=>"common", "PB_MESH_REMAP_LAW"=>"128,1.12,5"))
    end
    warm_start === nothing || (cfg["warm_start"] = String(warm_start))
    mesh_only && (cfg["mesh_only"] = true)
    cfg
end
"The worker script (AIRFOIL_LOWSPEED_WORKER overrides it for a development test only)."
lowspeed_worker_path() = get(ENV, "AIRFOIL_LOWSPEED_WORKER", normpath(joinpath(@__DIR__, "..", "workers", "lowspeed_worker.jl")))
_lowspeed_cmd(snap, cfg_path) = Cmd(`$(joinpath(Sys.BINDIR, Base.julia_exename())) --startup-file=no --threads=$LOWSPEED_THREADS --project=$(joinpath(snap, "src", "pbcav")) $(lowspeed_worker_path()) $cfg_path`;
    env=filter(kv -> !startswith(first(kv), "PB_") && !any(p -> startswith(first(kv), p), ("TOPO_", "TEM_", "CAV_MOVIE_")), copy(ENV)))

"""Create mesh for the low-speed backend: the worker builds the run's own mesh (PBCav's generator call and audit) and stops
before marching; about a minute (Julia start and mesh generation). Returns (mesh, metadata) like preview_mesh."""
function preview_lowspeed_mesh(path::AbstractString, flow::AbstractDict, outdir::AbstractString; source::AbstractDict=Dict())
    _lowspeed_ch_selected(flow) && return preview_lowspeed_ch_mesh(path, flow, outdir; source)   # Gurney flap / native base: C+H
    snap, head = _lowspeed_snapshot_check()
    ls = lowspeed_configuration(flow)
    foil, geometry = _lowspeed_geometry(Dict("airfoil"=>source), path, outdir, snap)
    cfg_path = joinpath(outdir, "lowspeed_mesh_input.toml")
    open(io -> TOML.print(io, _lowspeed_input(snap, head, foil, ls, _number(flow, "alpha_deg", 0.0; lo=-30, hi=30), outdir; mesh_only=true)),
         cfg_path, "w")
    logpath = joinpath(outdir, "lowspeed_mesh.log")
    ok = success(pipeline(_lowspeed_cmd(snap, cfg_path); stdout=logpath, stderr=logpath))
    ok && isfile(joinpath(outdir, "mesh_preview.msgpack")) ||
        throw(ErrorException("low-speed CFD mesh generation failed: $(_solver_failure_summary(logpath))"))
    mesh = AirfoilWire.decode(read(joinpath(outdir, "mesh_preview.msgpack")), :msgpack)
    meta = TOML.parsefile(joinpath(outdir, "mesh_result.toml"))
    metadata = merge(Dict{String,Any}(string(k)=>v for (k, v) in geometry), Dict{String,Any}(
        "geometry_support"=>"low-speed CFD (PBCav $(head[1:min(8,end)])) free-air sharp-TE C-mesh: $(ls["mesh_label"])",
        "mesh_cells"=>meta["mesh_cells"], "mesh_x_sha256"=>meta["mesh_x_sha256"], "mesh_y_sha256"=>meta["mesh_y_sha256"],
        "mesh_audit"=>meta["mesh_audit_line"], "mesh_gate_failures"=>meta["mesh_gate_failures"],
        "cfd_te_gap"=>geometry["cfd_te_gap"], "geometry_change"=>geometry["change"]))
    (mesh=mesh, metadata=metadata)
end

function _lowspeed_point(case, path, flow, alpha, dir, progress, log, warmdir=nothing)
    _lowspeed_ch_selected(flow) && return _lowspeed_ch_point(case, path, flow, alpha, dir, progress, log, warmdir)   # C+H path
    snap, head = _lowspeed_snapshot_check()
    ls = lowspeed_configuration(flow)
    foil, geometry = _lowspeed_geometry(case, path, dir, snap)
    isempty(geometry["change"]) || _deliver_log(log, geometry["change"] * ".")
    warm = warmdir === nothing ? nothing : joinpath(warmdir, "state_checkpoint.jls")
    cfg = _lowspeed_input(snap, head, foil, ls, alpha, dir;
                          warm_start=warm !== nothing && isfile(warm) ? warm : nothing)
    cfg_path = joinpath(dir, "input.toml")
    open(io -> TOML.print(io, cfg), cfg_path, "w")
    _deliver_log(log, @sprintf("Starting low-speed CFD (PBCav %s, pressure-based incompressible, fully turbulent SST, Tu %.3g%%, mu_t/mu %.3g) %s; %s; wall cap %.0f s.",
                               head[1:min(8, end)], 100ls["tu"], ls["mut_ratio"], lowspeed_hardware(ls), ls["mesh_label"], ls["wall_cap"]))
    cmd = _lowspeed_cmd(snap, cfg_path)
    started = time()
    ok = _log_process(cmd, joinpath(dir, "solver.log"), progress, log, "lowspeed", Float64(alpha);
                      residual_kind="PBCav steady residual (momentum, rho U^2/c per unit volume)")
    ok && isfile(joinpath(dir, "result.toml")) ||
        throw(ErrorException("low-speed CFD worker failed: $(_solver_failure_summary(joinpath(dir,"solver.log"))); see solver.log"))
    _collect_lowspeed_output(dir, Float64(alpha), geometry, ls, time() - started)
end

"Assess PBCav's measured steady equations and load history without replacing its native stopping rule."
function _lowspeed_convergence_assessment(meta, raw; ramp_time=0.)
    number(x) = x isa Real ? Float64(x) : something(tryparse(Float64, string(x)), NaN)
    mapping = ("iteration"=>"step", "cl"=>"CL", "cd"=>"CD", "cm"=>"CM",
        "momentum"=>"steady", "k"=>"steady_k", "omega"=>"steady_w",
        "circulation"=>"Gamma_defect", "mass_net_frac"=>"flux_net")
    history = [Dict{String,Any}(key=>get(row,column,nothing) for (key,column) in mapping)
        for row in raw if ramp_time<=0 || number(get(row,"t",NaN))>=ramp_time]
    # Final metadata is the actual saved field, which need not coincide with a
    # periodically logged row. Recheck it instead of trusting a stale history.
    if !isempty(history)
        for (key, column) in ("cl"=>"cl", "cd"=>"cd", "cm"=>"cm", "momentum"=>"steady_residual",
                "k"=>"steady_k", "omega"=>"steady_w", "circulation"=>"gamma_defect", "mass_net_frac"=>"flux_net")
            history[end][key] = get(meta,column,nothing)
        end
    end
    laminar = occursin(r"\bturbulence=laminar\b", String(get(meta,"config_line","")))
    limits = Dict{String,Float64}("circulation"=>1e-6)
    missing_limits = String[]
    for (key, column) in ("momentum"=>"steady_tol", "k"=>"turb_steady_tol", "omega"=>"turb_steady_tol")
        laminar && key in ("k","omega") && continue
        limit=number(get(meta,column,NaN))
        isfinite(limit) && limit>0 ? (limits[key]=limit) : push!(missing_limits,column)
    end
    outcome=String(get(meta,"outcome","UNKNOWN"))
    report=RANSConvergence.assess(history; residual_limits=limits,
        config=RANSConvergence.Config(drag_min=-1e-5,drag_max=20.,lift_max=20.),
        solver_steady_passed=outcome=="STEADY" && isempty(missing_limits),
        physical_state_valid=all(key->isfinite(number(get(meta,key,NaN))),("cl","cd","cm")),
        failed=!(outcome in ("STEADY","REACHED","INCOMPLETE")),
        scope=(laminar ? "PBCav momentum steady residual" : "PBCav momentum and SST steady residuals") *
            ", circulation defect and signed mass balance; mesh independence and experimental validation are separate")
    report["solver_outcome"]=outcome
    report["mass_normalization"]="PBCav free-air rho_inf=U_inf=chord=1; flux_net/(rho_inf*U_inf*chord)"
    report["native_stop_required"]=true
    report["ramp_samples_excluded"]=length(raw)-length(history)
    report["missing_limits"]=unique(missing_limits)
    if !isempty(missing_limits)
        report["criteria_passed"]=false
        append!(report["missing"],["limit:"*key for key in unique(missing_limits)])
    end
    report["reason"]=replace(report["reason"], "complete equation convergence evidence is unavailable"=>
        "the native PBCav steady stop has not been reached or its stop limits are unavailable")
    report["criteria_passed"] && !report["converged"] && (report["reason"]=
        "Monitored residual, mass-balance and load checks passed; the native PBCav steady stop has not been reached")
    report
end

function _lowspeed_convergence_message(meta, assessment)
    titles=Dict("converged"=>"Converged", "loads_stable_flow_unsettled"=>"Loads changed little in the recent window; flow not yet converged",
        "incomplete"=>"Analysis incomplete", "failed"=>"Analysis failed")
    outcome=String(get(meta,"outcome","UNKNOWN")); steps=get(meta,"steps",0)
    ending=outcome=="INCOMPLETE" ? "Wall-time limit reached at step $steps." :
        outcome=="REACHED" ? "Requested iteration/time limit reached at step $steps." : "Solver outcome $outcome at step $steps."
    refusal=String(get(meta,"refusal",""));isempty(refusal) || (ending*=" "*refusal)
    stability_caveat=assessment["status"]=="loads_stable_flow_unsettled" ?
        " Recent load stability does not establish the final CL/CD or mesh independence." : ""
    "$(titles[assessment["status"]]). $(assessment["reason"]). $ending$stability_caveat " *
        (assessment["converged"] ? "Mesh independence and experimental validation are separate." :
         "Saved fields and coefficients remain available for inspection.")
end

function _collect_lowspeed_output(dir, alpha, geometry, ls, elapsed)
    meta = TOML.parsefile(joinpath(dir, "result.toml"))
    fnum(x) = x isa Real ? Float64(x) : something(tryparse(Float64, string(x)), NaN)
    mesh = AirfoilWire.decode(read(joinpath(dir, "mesh_preview.msgpack")), :msgpack)
    mesh["artifact"] = joinpath(dir, "mesh_full.msgpack"); mesh["preview_artifact"] = joinpath(dir, "mesh_preview.msgpack")
    cp = Dict{String,Any}("upper"=>Any[], "lower"=>Any[])
    for row in _csv_rows(joinpath(dir, "surface_cp.csv"))
        push!(cp[row["surface"]], [_float(row["x"]), _float(row["cp"])])
    end
    cols = collect(FullFieldArtifact.COLUMNS)
    samples = Dict(k=>Float64[] for k in cols)
    for row in _csv_rows(joinpath(dir, "field_samples.csv")), k in cols
        v = _float(row[k]); v === nothing && throw(ArgumentError("nonfinite $k in the low-speed field output"))
        push!(samples[k], v)
    end
    raw = _csv_rows(joinpath(dir, "loads_history.csv"))
    stride = max(1, ceil(Int, length(raw)/1200))
    idx = isempty(raw) ? Int[] : unique(vcat(collect(1:stride:length(raw)), length(raw)))
    history = [Dict{String,Any}("iteration"=>parse(Int, raw[k]["step"]), "residual"=>_float(raw[k]["steady"]),
        "residual_linf"=>_float(raw[k]["steady"]), "cl"=>_float(raw[k]["CL"]), "cd"=>_float(raw[k]["CD"]),
        "cm"=>_float(raw[k]["CM"]), "steady_k"=>_float(raw[k]["steady_k"]), "steady_w"=>_float(raw[k]["steady_w"]),
        "t_pseudo"=>_float(raw[k]["t"])) for k in idx]
    outcome = String(meta["outcome"])
    assessment = _lowspeed_convergence_assessment(meta,raw)
    steady = assessment["converged"]
    status = steady ? "completed" : assessment["status"]=="failed" ? "failed" : "not_converged"
    convergence = _lowspeed_convergence_message(meta,assessment)
    notes = String[
        "Low-speed CFD: version 2's incompressible pressure-based solver (PBCav, commit $(meta["commit"])), IN DEVELOPMENT and NOT YET VALIDATED.",
        "Fully turbulent k-omega SST (SST-2003m; no transition model): freestream Tu $(round(100*ls["tu"]; sigdigits=3))%, mu_t/mu $(ls["mut_ratio"]). XFOIL comparisons should use fixed transition near the leading edge.",
        "Free air: far-field velocity with a point-vortex correction, pressure outlet; Mach is ignored (incompressible). Adaptive backward-Euler pseudo-time (requested maximum dt $(ls["pseudo_dt"]) c/U) to the solver's own steady stop." *
            (ls["rans_acceleration"] ? " Larger pseudo-time ceiling is experimental; compare convergence and wall time before relying on a speedup." : ""),
        isempty(String(get(meta,"warm_start_source",""))) ? "Kutta potential-flow start." :
            "Started from the previous polar angle's checked full solver state; source $(meta["warm_start_source"]).",
        "Mesh: $(ls["mesh_label"]). PBCav mesh audit reference gates: $(isempty(meta["mesh_gate_failures"]) ? "all pass" : join(meta["mesh_gate_failures"], "; ")). Grid independence is not established.",
        isempty(geometry["change"]) ? "The CFD geometry is the source section (sharp trailing edge)." : geometry["change"] * ". XFOIL uses the source section; account for the difference when comparing.",
        "Fields: the 'mach' column holds the speed |U|/U_inf (incompressible); density and temperature are 1; nut is mu_t/mu; vorticity from the solver's least-squares gradients."]
    fields = Dict{String,Any}("kind"=>"samples", "samples"=>samples, "sample_shape"=>[meta["sampled_ni"], meta["sampled_nj"]],
        "sample_stride"=>meta["sample_stride"], "artifact"=>joinpath(dir, "field_samples.csv"),
        "full_artifact"=>joinpath(dir, "field_full.msgpack"), "full_cell_count"=>meta["full_field_cells"],
        "mesh_sha256"=>meta["full_field_mesh_sha256"], "full_iteration"=>meta["full_field_iteration"],
        "speed_column"=>"mach", "hidden_quantities"=>["density", "temperature"],
        "quantity_labels"=>Dict("mach"=>"Speed |U|/U∞", "cp"=>"Pressure coefficient Cp", "pressure"=>"Pressure (p - p∞)/(ρU∞²)",
                                "u"=>"Velocity u/U∞", "v"=>"Velocity v/U∞", "vorticity"=>"Vorticity ωc/U∞", "nut"=>"Eddy viscosity ratio μt/μ"),
        "quantity_units"=>Dict("mach"=>"|U|/U∞", "cp"=>"dimensionless", "pressure"=>"ρU∞²", "u"=>"U∞", "v"=>"U∞",
                               "vorticity"=>"U∞/c", "nut"=>"μt/μ", "density"=>"ρ∞", "temperature"=>"T∞"))
    Dict{String,Any}(
        "backend"=>"lowspeed", "alpha_deg"=>alpha, "status"=>status, "convergence"=>convergence,
        "convergence_status"=>assessment["status"],
        "validation"=>"version 2 pressure-based solver in development; not yet validated",
        "geometry"=>geometry, "mesh"=>mesh, "cl"=>fnum(meta["cl"]), "cd"=>fnum(meta["cd"]), "cm"=>fnum(meta["cm"]),
        "iteration_history"=>history, "cp"=>cp, "fields"=>fields, "notes"=>notes,
        "artifacts"=>Dict("cp_csv"=>joinpath(dir, "surface_cp.csv"), "surface_csv"=>joinpath(dir, "surface.csv"),
            "loads_history_csv"=>joinpath(dir, "loads_history.csv"), "mesh_full"=>joinpath(dir, "mesh_full.msgpack"),
            "mesh_preview"=>joinpath(dir, "mesh_preview.msgpack"), "field_csv"=>joinpath(dir, "field_samples.csv"),
            "field_full"=>joinpath(dir, "field_full.msgpack"), "result_toml"=>joinpath(dir, "result.toml"),
            "state_checkpoint"=>isfile(joinpath(dir,"state_checkpoint.jls")) ? joinpath(dir,"state_checkpoint.jls") : nothing,
            "input_toml"=>joinpath(dir, "input.toml"), "log"=>joinpath(dir, "solver.log")),
        "diagnostics"=>Dict{String,Any}("solver"=>meta["solver"], "commit"=>meta["commit"], "snapshot"=>meta["snapshot"],
            "solver_outcome"=>outcome, "steps"=>meta["steps"], "max_steps"=>meta["max_steps"], "wall_cap_s"=>fnum(meta["wall_cap"]),
            "warm_start_source"=>get(meta,"warm_start_source",""),
            "march_wall_seconds"=>fnum(meta["march_wall_seconds"]), "point_wall_seconds"=>elapsed, "threads"=>meta["threads"],
            "time_mode"=>"steady", "clock"=>"steady adaptive pseudo-time (backward Euler, maximum dt $(ls["pseudo_dt"]) c/U)",
            "rans_acceleration"=>ls["rans_acceleration"], "rans_acceleration_method"=>ls["rans_acceleration_method"],
            "requested_pseudo_dt"=>ls["pseudo_dt"],
            "convergence_guard"=>Dict("status"=>steady ? "solver_steady_stop" : assessment["status"], "reason"=>convergence),
            "convergence_assessment"=>assessment,
            "steady_assessment"=>assessment,
            "residual_scaled"=>fnum(meta["steady_residual"]), "residual_kind"=>"PBCav steady residual (momentum)",
            "steady_residual"=>fnum(meta["steady_residual"]), "steady_tol"=>fnum(meta["steady_tol"]),
            "steady_k"=>fnum(meta["steady_k"]), "steady_w"=>fnum(meta["steady_w"]), "turb_steady_tol"=>fnum(meta["turb_steady_tol"]),
            "cl_pressure"=>fnum(meta["cl_pressure"]), "cl_friction"=>fnum(meta["cl_friction"]),
            "cd_pressure"=>fnum(meta["cd_pressure"]), "cd_friction"=>fnum(meta["cd_friction"]),
            "circulation"=>fnum(meta["gamma"]), "circulation_defect"=>fnum(meta["gamma_defect"]),
            "mass_net_flux"=>fnum(meta["flux_net"]), "mass_step_err_max"=>fnum(meta["mass_step_err_max"]),
            "rejections"=>meta["rejections"], "mut_ratio_max"=>fnum(meta["mut_ratio_max"]),
            "yplus_median"=>fnum(meta["yplus_median"]), "yplus_max"=>fnum(meta["yplus_max"]),
            "omega_bound"=>fnum(get(meta, "omega_bound", NaN)), "omega_bound_tail"=>get(meta, "omega_bound_tail", ""),
            "reynolds"=>fnum(meta["reynolds"]), "mach"=>0.0, "freestream_tu"=>fnum(meta["tu"]), "freestream_mut_ratio"=>fnum(meta["mut_ratio"]),
            "turbulence"=>"fully turbulent k-omega SST-2003m", "mesh"=>meta["mesh"], "mesh_resolution"=>ls["resolution"],
            "mesh_cells"=>meta["mesh_cells"], "mesh_gates"=>meta["mesh_gates"], "mesh_gate_failures"=>meta["mesh_gate_failures"],
            "mesh_audit"=>meta["mesh_audit_line"], "mesh_topology"=>"sharp_C", "foil_file"=>meta["foil"], "foil_sha256"=>meta["foil_sha256"],
            "geometry_transform"=>geometry["transform"], "configuration"=>meta["config_line"]))
end
