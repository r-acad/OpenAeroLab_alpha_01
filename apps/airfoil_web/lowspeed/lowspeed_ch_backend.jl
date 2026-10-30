# =====================================================================================================================
# The low-speed CFD's C+H path (23 Oct 2026, the Gurney flap; the flap as a BAFFLE since 28 Oct 2026). Included inside
# CFDBackend after lowspeed_backend.jl. PBCav's conforming C + H ("strip") free-air mesh (the strip block in the free-air path,
# the column remap with a graded strip, and W6b) runs with lowspeed_ch_worker.jl, from the SAME pinned, read-only snapshot and
# manifest as the sharp-trailing-edge cases (lowspeed_backend.jl: the pin table; S1030_FLAPREF, a43013d1, since 30 Oct 2026).
# AIRFOIL_LOWSPEED_CH_SNAPSHOT still overrides the path for a test.
#
# THE GURNEY FLAP IS A BAFFLE (the user's rulings of 27 Oct 2026; design GURNEY_BAFFLE_DESIGN.md, the solver's records
# memory/project_docs/_records/2026-10-13_pressure_based/air/baffle/ in the snapshot): a zero-thickness plate normal to the chord at the trailing
# edge, exactly sharp, symmetric = h/2 up and h/2 down; a wall INSIDE the section's own finite-trailing-edge C+H mesh along the C
# block's grid line from the trailing-edge corner. The mesh is the one of the native-base control (lowspeed_topology = ch_native):
# the app does not reshape it and inserts nothing; PBCav's PB_BAFFLE straightens that grid line, snaps its tip to h and cuts the
# faces. The knobs: PB_BAFFLE=gurney:side=lower|upper|symmetric,h=<h/c>, PB_START=ramp with PB_START_RAMP=2,0.05 (the projected
# start is singular at the tip), PB_TURB_MUT_BASE=committed (A2's march fix). The solver's quality gate refuses a plate outside
# the mesh's admitted range of h; its message (with the range) reaches the user unchanged.
# =====================================================================================================================
const LOWSPEED_CH_SNAPSHOT_DEFAULT = LOWSPEED_SNAPSHOT_DEFAULT
const LOWSPEED_CH_COMMIT_PINNED = LOWSPEED_COMMIT_PINNED
lowspeed_ch_snapshot() = normpath(get(ENV, "AIRFOIL_LOWSPEED_CH_SNAPSHOT", lowspeed_pin().snapshot))
"C+H mesh presets: the demo's C-block sizes (PB_MESH), with the base cells set per base height by _lowspeed_ch_knobs."
const LOWSPEED_CH_MESHES = Dict(
    "demo"=>(mesh="192,72,3e-5,1.20,80,64", dn1=3e-5,
             label="Demo (fast) C+H: 192 foil x 72 normal cells + 2 x 64 wake legs, H block 64 stations x (2 x base cells), first wall spacing 3e-5 c, column remap with the graded strip"),
    "smoke"=>(mesh="64,40,1e-3,1.30,40,16", dn1=1e-3,
              label="Smoke test only (C+H): 64 foil x 40 normal cells + 2 x 16 wake, first wall spacing 1e-3 c"))
"""The Gurney flap's start (PB_START=ramp, PB_START_RAMP): the far field is accelerated from r0 U to U over t = T c/U, because the
projected potential flow is singular at the plate's tip and at the base corners (baffle records, section 3). The values are the
solver's defaults, named explicitly so that a later change of its defaults cannot change the app silently. The loads before
t = T include the accelerating domain's uniform pressure gradient and are not meaningful."""
const LOWSPEED_CH_RAMP = "2,0.05"
const LOWSPEED_CH_RAMP_T = 2.0
"The base's cross cells grow from the wall spacing at both legs by at most this ratio (the C+H air gate G2k's 1.2)."
const LOWSPEED_CH_BASE_RATIO = 1.2

_lowspeed_ch_selected(flow::AbstractDict) = gurney_enabled(flow) || String(_get(flow, "lowspeed_topology", "sharp")) == "ch_native"

"""Validated settings of the C+H path: the sharp path's inputs plus the topology. lowspeed_topology may be "sharp" (the
default) or "ch_native" (the native finite trailing edge on C+H, a like-for-like control for a Gurney flap)."""
function lowspeed_ch_configuration(flow::AbstractDict)
    topo = String(_get(flow, "lowspeed_topology", "sharp"))
    topo in ("sharp", "ch_native") || throw(ArgumentError("lowspeed_topology must be sharp or ch_native"))
    gurney_enabled(flow) && topo == "ch_native" && throw(ArgumentError("lowspeed_topology=ch_native is the native-base control: remove the Gurney flap or the key"))
    res = String(_get(flow, "lowspeed_resolution", LOWSPEED_DEFAULTS["lowspeed_resolution"]))
    haskey(LOWSPEED_CH_MESHES, res) || throw(ArgumentError("lowspeed_resolution must be demo (or smoke for tests)"))
    _number(flow, "cfd_te_thickness", 0.0; lo=0.0, hi=0.05) > 0 && throw(ArgumentError(gurney_enabled(flow) ?
        "The Gurney flap and the CFD trailing-edge thickness override both set the trailing edge: clear the CFD trailing-edge thickness (the flap is a baffle on the section's own finite trailing edge, which it keeps)." :
        "the low-speed CFD's C+H path takes the native trailing edge: clear the CFD trailing-edge thickness override"))
    g = gurney_request(flow)
    # the hidden start and mu_t-base controls (C+H path only): defaults are the Gurney flap's (ramp, committed) with a flap and the
    # control's (projection, extrapolated: the solver's defaults, nothing named) without one
    start = String(_get(flow, "lowspeed_start", g === nothing ? "projection" : "ramp"))
    start in ("projection", "ramp") || throw(ArgumentError("lowspeed_start must be projection or ramp"))
    mutb = String(_get(flow, "lowspeed_turb_mut_base", g === nothing ? "extrapolated" : "committed"))
    mutb in ("extrapolated", "committed") || throw(ArgumentError("lowspeed_turb_mut_base must be extrapolated or committed"))
    cfgd = Dict{String,Any}("topology"=>"ch", "ch_source"=>g !== nothing ? "gurney" : "native", "resolution"=>res,
        "mesh"=>LOWSPEED_CH_MESHES[res].mesh, "dn1"=>LOWSPEED_CH_MESHES[res].dn1, "mesh_label"=>LOWSPEED_CH_MESHES[res].label,
        "reynolds"=>_number(flow, "reynolds", 3e6; lo=1e4, hi=1e8),
        "tu"=>_number(flow, "lowspeed_tu", LOWSPEED_DEFAULTS["lowspeed_tu"]; lo=1e-6, hi=0.19),
        "mut_ratio"=>_number(flow, "lowspeed_mut_ratio", LOWSPEED_DEFAULTS["lowspeed_mut_ratio"]; lo=1e-4, hi=1e4),
        "wall_cap"=>_number(flow, "lowspeed_wall_cap_s", 7200.0; lo=30.0, hi=14400.0),
        "max_steps"=>_integer(flow, "lowspeed_max_steps", LOWSPEED_DEFAULTS["lowspeed_max_steps"]; lo=1, hi=20000),
        "live_every"=>0, "gpu"=>lowspeed_gpu(flow), "start"=>start, "mut_base"=>mutb)
    g === nothing || (cfgd["baffle"] = Dict{String,Any}("side"=>g.mode, "h"=>g.height))
    # 30 Oct 2026: the refinement of the plate's free edge and its shear layer (PBCav's PB_BAFFLE_REFINE); named in the
    # configuration only when asked for, so that every request without it is configured exactly as before
    g === nothing || (lv = gurney_refine_level(flow); lv == "off" || (cfgd["baffle"]["refine"] = lv))
    cfgd
end

"""The C+H snapshot's commit, after the same checks as the sharp snapshot (HEAD at the pin, every solver file unmodified).
With `baffle=true` (a Gurney flap) the snapshot must also have the interior-wall support (src/pbcav/baffle.jl): the previous
pin (AIRFOIL_LOWSPEED_PIN=s1024) cannot run the flap and says so rather than failing in its worker."""
function _lowspeed_ch_snapshot_check(; baffle::Bool=false, refine::Bool=false)
    snap, head = _lowspeed_snapshot_verify(lowspeed_ch_snapshot(), haskey(ENV, "AIRFOIL_LOWSPEED_CH_SNAPSHOT"), "low-speed C+H solver")
    baffle && !isfile(joinpath(snap, "src", "pbcav", "baffle.jl")) && throw(ArgumentError(
        "The Gurney flap is a baffle (an interior wall of PBCav's C+H mesh), and the low-speed solver snapshot $snap (commit $(head[1:min(8, end)])) has no interior-wall support (src/pbcav/baffle.jl). Use the pinned snapshot $(LOWSPEED_PINS["s1028"].snapshot) (commit $(LOWSPEED_PINS["s1028"].commit))."))
    refine && !occursin("PB_BAFFLE_REFINE", read(joinpath(snap, "src", "pbcav", "knobs.jl"), String)) && throw(ArgumentError(
        "The Gurney flap's mesh refinement (gurney_refine standard or fine) is PBCav's PB_BAFFLE_REFINE, and the low-speed solver snapshot $snap (commit $(head[1:min(8, end)])) does not have it. Use the pinned snapshot $(LOWSPEED_PINS["s1030"].snapshot) (commit $(LOWSPEED_PINS["s1030"].commit)), or send gurney_refine off."))
    snap, head
end

"""The section file of the C+H path and its geometry record: the section's NATIVE finite-trailing-edge contour (NACA four-digit:
the equation contour at 4097 points, the CFD meshes' support; coordinate files: the source points), in Selig order. It is the
same file, hence the same C+H mesh, with and without a Gurney flap: the flap is a baffle in that mesh (header), the contour is not
changed. With a flap the record also carries the plate's drawing record (gurney.jl: tips, trailing-edge points; for a cambered
section the preview's record is on the trimmed host section, which the CFD does not use). Returns (path, geometry::Dict,
base_height)."""
function _lowspeed_ch_geometry(case, path::AbstractString, dir::AbstractString, flow::AbstractDict)
    source = _get(case, "airfoil", Dict())
    kind = String(_get(source, "kind", ""))
    af = kind == "naca4" ? CMeshGen.naca4_points(parse(Int, String(_get(source, "id", "")))) : CMeshGen.read_airfoil_dat(path)
    gap = hypot(af.x[1] - af.x[end], af.y[1] - af.y[end])
    gap > 1e-6 || throw(ArgumentError(@sprintf("%s needs a finite trailing edge; this section's gap is %.2e c",
        gurney_enabled(flow) ? "the Gurney flap (a baffle on the section's finite-trailing-edge C+H mesh)" : "lowspeed_topology=ch_native", gap)))
    f = joinpath(mkpath(dir), "airfoil_cfd_native_base.dat")
    open(f, "w") do io
        println(io, "CFD section (native finite trailing edge, Selig order) of ", kind == "naca4" ? "NACA $(_get(source, "id", "")) (equations, 4097 points)" : basename(path))
        for q in reverse(eachindex(af.x)); @printf(io, "%.17g %.17g\n", af.x[q], af.y[q]); end
    end
    back = CMeshGen.read_airfoil_dat(f)
    back.x == af.x && back.y == af.y || error("the native C+H section file does not read back bit for bit")
    geometry = Dict{String,Any}("transform"=>"none (native finite trailing edge on C+H)", "change"=>"", "path"=>f, "source_te_gap"=>gap,
        "cfd_te_gap"=>gap, "mesh_topology"=>"conformal_C_H_v1")
    gs = gurney_section(source, path, flow)
    if gs !== nothing
        g = gs.metadata
        g["cfd_section"] = "native"; g["baffle"] = true
        g["note"] = gurney_baffle_note(g)
        geometry["transform"] = g["note"]; geometry["change"] = g["note"]; geometry["gurney"] = g
    end
    f, geometry, gap
end

"""The PB_* knobs of a C+H point, every one named. Base cells per half: the fewest that let the graded base grow from the
wall spacing at both legs by at most 1.2 (n = ceil(log(1 + 0.2 (h/2)/dn1)/log 1.2), at least 4), with the trailing-edge
interval 1.5 h/(2n), inside the generator's G7' range [h/(2n), h/n]; G8' (h/(2n) >= dn1) is checked. `gap` is the section's
NATIVE trailing-edge gap (never the plate's: the flap is a baffle in the native mesh). A Gurney flap adds PB_BAFFLE, the ramp
start and PB_TURB_MUT_BASE (the header); the control adds them only when its hidden keys ask."""
function _lowspeed_ch_knobs(ls::AbstractDict, gap::Real)
    q = LOWSPEED_CH_BASE_RATIO; dn1 = ls["dn1"]
    nb = max(4, ceil(Int, log(1 + (q - 1)*(gap/2)/dn1)/log(q)))
    nb <= 256 || throw(ArgumentError(@sprintf("the C+H base of %.4f c needs %d cells per half, above the 256 the app admits", gap, nb)))
    gap/(2nb) >= dn1 || throw(ArgumentError(@sprintf("the C+H base of %.4f c is too short for %d cells per half at the wall spacing %.1e c (G8')", gap, nb, dn1)))
    kn = Dict{String,String}("PB_MESH"=>ls["mesh"], "PB_MESH_LE_DS"=>"0", "PB_MESH_TE_BASE"=>string(nb),
        "PB_MESH_TE_DS"=>@sprintf("%.6e", 1.5gap/(2nb)), "PB_MESH_TE_TOPOLOGY"=>"strip",
        "PB_MESH_REMAP"=>"common", "PB_MESH_REMAP_STRIP"=>"graded",
        "PB_TURBULENCE"=>"sst", "PB_SST_TU_INF"=>string(ls["tu"]), "PB_SST_MUT_RATIO"=>string(ls["mut_ratio"]),
        "PB_DT"=>"0.25", "PB_STEADY_TOL"=>"1e-6", "PB_TURB_STEADY_TOL"=>"1e-7", "PB_MAX_STEPS"=>string(ls["max_steps"]),
        "PB_WALL_CAP"=>string(ls["wall_cap"]), "PB_N_OUTER_MAX"=>"40", "PB_VORTEX"=>"point", "PB_VORTEX_EVERY"=>"10",
        "PB_VORTEX_RELAX"=>"0.5",
        # W6b's settings, named explicitly (their values are the defaults of d62a577f and of b7a7b29b alike; the sharp path
        # names the same three, LOWSPEED_W6B_KNOBS): the coupled step's SIMPLEC alpha_u (0.8), the coupled Jacobian with its
        # viscous non-orthogonal part, and the omega non-orthogonal bound.
        "PB_CPL_AU"=>"0.8", "PB_CPL_VISC_NONORTH"=>"jacobian", "PB_TURB_OMEGA_BOUND"=>"0.2")
    # the ramp start (the Gurney flap's; the control only when asked) and the mu_t base (named whenever it is not the default,
    # and always with a flap, so that a point's receipt shows it)
    get(ls, "start", "projection") == "ramp" && merge!(kn, Dict{String,String}("PB_START"=>"ramp", "PB_START_RAMP"=>LOWSPEED_CH_RAMP))
    (haskey(ls, "baffle") || get(ls, "mut_base", "extrapolated") != "extrapolated") &&
        (kn["PB_TURB_MUT_BASE"] = String(get(ls, "mut_base", "extrapolated")))
    # the Gurney flap: a baffle on the native mesh (snap=exact, straighten=0.5 deg, shear=0.1: the solver's defaults, which give
    # the user's plate exactly: perpendicular to the chord, tip at h)
    haskey(ls, "baffle") && (kn["PB_BAFFLE"] = "gurney:side=" * String(ls["baffle"]["side"]) * ",h=" * string(Float64(ls["baffle"]["h"])))
    haskey(ls, "baffle") && haskey(ls["baffle"], "refine") && (kn["PB_BAFFLE_REFINE"] = GURNEY_REFINE_KNOBS[String(ls["baffle"]["refine"])])
    get(ls, "gpu", false) && merge!(kn, Dict{String,String}(LOWSPEED_GPU_KNOBS))   # the GPU (opt-in; lowspeed_backend.jl)
    kn
end

function _lowspeed_ch_input(snap, head, foil, ls, knobs, alpha, dir; mesh_only::Bool=false, streamline=nothing)
    cfg = Dict{String,Any}("snapshot"=>snap, "commit"=>head, "foil"=>replace(String(foil), '\\'=>'/'), "knobs"=>knobs,
        "resolution"=>ls["resolution"], "reynolds"=>ls["reynolds"], "alpha"=>Float64(alpha),
        "outdir"=>replace(String(dir), '\\'=>'/'))
    mesh_only && (cfg["mesh_only"] = true)
    # streamline alignment (opt-in, development): the hashed JXfoil path file and the morph (streamline_backend.jl)
    streamline === nothing || (cfg["streamline"] = streamline)
    cfg
end
lowspeed_ch_worker_path() = get(ENV, "AIRFOIL_LOWSPEED_CH_WORKER", normpath(joinpath(@__DIR__, "..", "lowspeed_ch_worker.jl")))
_lowspeed_ch_cmd(snap, cfg_path) = Cmd(`$(joinpath(Sys.BINDIR, Base.julia_exename())) --startup-file=no --threads=$LOWSPEED_THREADS --project=$(joinpath(snap, "src", "pbcav")) $(lowspeed_ch_worker_path()) $cfg_path`;
    env=filter(kv -> !startswith(first(kv), "PB_") && !any(p -> startswith(first(kv), p), ("TOPO_", "TEM_", "CAV_MOVIE_")), copy(ENV)))

"""The app's C+H air gates (CMeshQualityV1, cavitating=false, strip_grade=true, as Create mesh of the compressible C+H) and
the regional base audit, measured on the PBCav mesh's own nodes (mesh_full.msgpack). Geometry only."""
function _lowspeed_ch_app_audit(mesh_full_path::AbstractString)
    full = AirfoilWire.decode(read(mesh_full_path), :msgpack)
    c = only(b for b in full["blocks"] if b["id"] == "C"); h = only(b for b in full["blocks"] if b["id"] == "H")
    X = reshape(Float64.(c["x"]), c["ni"], c["nj"]); Y = reshape(Float64.(c["y"]), c["ni"], c["nj"])
    Xs = reshape(Float64.(h["x"]), h["ni"], h["nj"]); Ys = reshape(Float64.(h["y"]), h["ni"], h["nj"])
    NIW = Int(full["wake_cells_per_side"]); NIF = Int(full["foil_surface_cells"]); strip = (X=Xs, Y=Ys)
    Q = CMeshGen.CMeshQualityMetrics
    s = Q.quality_summary(X, Y, NIW, NIF; strip=strip)
    gates = Q.quality_gates(s; cavitating=false, strip_grade=true)
    fin(x) = x isa Real && !isfinite(x) ? nothing : x
    Dict{String,Any}("air_reference_gates"=>Dict{String,Any}("passed"=>all(g[5] for g in gates), "failed"=>Q.gate_failures(gates),
            "gates"=>[Dict{String,Any}("id"=>g[1], "quantity"=>g[2], "value"=>fin(g[3]), "limit"=>g[4], "passed"=>g[5]) for g in gates],
            "scope"=>"the app's C+H air gates (CMeshQualityV1, as the compressible C+H Create mesh), measured on the PBCav nodes; geometry only"),
        "global"=>Dict{String,Any}("maximum_face_nonorthogonality_deg"=>s.theta_max, "minimum_corner_angle_deg"=>s.minang,
            "maximum_aspect_ratio"=>s.ar_max, "maximum_neighbour_volume_ratio"=>s.vr_max, "maximum_skew_over_distance"=>s.skew_max,
            "base_cross_ratio"=>fin(s.rt_base), "corner_ratio_max"=>fin(s.rt_corner_max), "corner_ratio_min"=>fin(s.rt_corner_min)),
        "regional"=>gurney_regional_audit(X, Y, NIW, NIF, strip),
        # the baffle's record (the worker's; taken out again by _lowspeed_ch_mesh_metadata into the mesh metadata)
        "baffle_record"=>get(get(full, "geometry_metadata", Dict()), "gurney_baffle", nothing),
        "streamline_record"=>get(get(full, "geometry_metadata", Dict()), "streamline_alignment", nothing))
end

function _lowspeed_ch_mesh_metadata(geometry, head, ls, knobs, meta, app_audit)
    baffle = pop!(app_audit, "baffle_record", nothing)
    aligned = pop!(app_audit, "streamline_record", nothing)
    md = merge(Dict{String,Any}(string(k)=>v for (k, v) in geometry), Dict{String,Any}(
        "geometry_support"=>"low-speed CFD (PBCav $(head[1:min(8,end)])) free-air C+H mesh: $(ls["mesh_label"])",
        "mesh_family"=>"PBCav conforming C + finite-base H (strip), column remap with the graded strip",
        "mesh_cells"=>meta["mesh_cells"], "mesh_x_sha256"=>meta["mesh_x_sha256"], "mesh_y_sha256"=>meta["mesh_y_sha256"],
        "strip_x_sha256"=>meta["strip_x_sha256"], "strip_y_sha256"=>meta["strip_y_sha256"],
        "mesh_audit"=>meta["mesh_audit_line"], "mesh_gate_failures"=>meta["mesh_gate_failures"],
        "geometry_change"=>geometry["change"], "pbcav_knobs"=>knobs, "base_cells_per_half"=>parse(Int, knobs["PB_MESH_TE_BASE"]),
        "app_mesh_audit"=>app_audit, "snapshot"=>lowspeed_ch_snapshot(), "commit"=>head))
    haskey(geometry, "gurney") && (md["gurney_mesh_audit"] = app_audit["regional"])
    if baffle !== nothing
        md["gurney_baffle"] = baffle
        md["mesh_family"] *= " + Gurney baffle (an interior wall on the C block's grid line from the trailing-edge corner)"
        rf = get(baffle, "refine", nothing)
        rf !== nothing && get(rf, "admitted", false) === true && (md["mesh_family"] *= @sprintf(" + free-edge refinement (%d rows added around the plate's tip, cells 1/%.3g of the section's own at the tip)", rf["rows_added"], rf["q_used"]))
        # the receipt hashes name the nodes the faces are built on (PB BAFFLE x_sha256 / y_sha256), not the generator's
        md["mesh_hash_note"] = "mesh_x_sha256 / mesh_y_sha256 are the PB BAFFLE receipt's x_sha256 / y_sha256: the hashes of the C-block nodes the faces are built on (after the plate's node movement)"
    end
    aligned === nothing || (md["streamline_alignment"] = aligned; md["mesh_family"] *= " + streamline-aligned (JXfoil, development)")
    md
end

"""A failed worker's reason as the user should see it: the baffle gate's own refusal (the worker writes baffle_refusal.toml and
the GURNEY_BAFFLE_REFUSED line) as an ArgumentError, the message unchanged but for a plain lead-in; any other failure as an
ErrorException with the log's ERROR line."""
function _lowspeed_ch_failure(dir::AbstractString, logpath::AbstractString, what::AbstractString)
    rf = joinpath(dir, "baffle_refusal.toml")
    if isfile(rf)
        r = TOML.parsefile(rf)
        throw(ArgumentError("The Gurney flap was refused by the solver's plate quality gate (the flap is a baffle on this section's C+H mesh); " *
            "choose a height inside the admitted range: " * String(r["message"])))
    end
    throw(ErrorException("$what: $(_solver_failure_summary(logpath))"))
end

"Create mesh on the C+H path: the C+H worker builds the run's own mesh and stops before marching."
function preview_lowspeed_ch_mesh(path::AbstractString, flow::AbstractDict, outdir::AbstractString; source::AbstractDict=Dict())
    ls = lowspeed_ch_configuration(flow)
    snap, head = _lowspeed_ch_snapshot_check(baffle=haskey(ls, "baffle"), refine=haskey(get(ls, "baffle", Dict()), "refine"))
    foil, geometry, gap = _lowspeed_ch_geometry(Dict("airfoil"=>source, "flow"=>flow), path, outdir, flow)
    knobs = _lowspeed_ch_knobs(ls, gap)
    cfg_path = joinpath(outdir, "lowspeed_mesh_input.toml")
    alpha = _number(flow, "alpha_deg", 0.0; lo=-30, hi=30)
    streamline = streamline_worker_cfg(source, flow, alpha, outdir; gurney=get(geometry, "gurney", nothing))
    open(io -> TOML.print(io, _lowspeed_ch_input(snap, head, foil, ls, knobs, alpha, outdir; mesh_only=true, streamline)),
         cfg_path, "w")
    logpath = joinpath(outdir, "lowspeed_mesh.log")
    ok = success(pipeline(_lowspeed_ch_cmd(snap, cfg_path); stdout=logpath, stderr=logpath))
    ok && isfile(joinpath(outdir, "mesh_preview.msgpack")) || _lowspeed_ch_failure(outdir, logpath, "low-speed CFD C+H mesh generation failed")
    mesh = AirfoilWire.decode(read(joinpath(outdir, "mesh_preview.msgpack")), :msgpack)
    meta = TOML.parsefile(joinpath(outdir, "mesh_result.toml"))
    metadata = _lowspeed_ch_mesh_metadata(geometry, head, ls, knobs, meta, _lowspeed_ch_app_audit(joinpath(outdir, "mesh_full.msgpack")))
    mesh["geometry_metadata"] = metadata
    (mesh=mesh, metadata=metadata)
end

function _lowspeed_ch_point(case, path, flow, alpha, dir, progress, log)
    ls = lowspeed_ch_configuration(flow)
    snap, head = _lowspeed_ch_snapshot_check(baffle=haskey(ls, "baffle"), refine=haskey(get(ls, "baffle", Dict()), "refine"))
    foil, geometry, gap = _lowspeed_ch_geometry(case, path, dir, flow)
    knobs = _lowspeed_ch_knobs(ls, gap)
    isempty(geometry["change"]) || _deliver_log(log, geometry["change"] * ".")
    streamline = streamline_worker_cfg(_get(case, "airfoil", Dict()), flow, alpha, dir; gurney=get(geometry, "gurney", nothing))
    streamline === nothing || _deliver_log(log, "$(STREAMLINE_DEV_LABEL): $(streamline["path_label"]), admission $(streamline["admission"]); the worker morphs its own mesh with the hashed path file.")
    cfg = _lowspeed_ch_input(snap, head, foil, ls, knobs, alpha, dir; streamline)
    cfg_path = joinpath(dir, "input.toml")
    open(io -> TOML.print(io, cfg), cfg_path, "w")
    startword = ls["start"] == "ramp" ? "ramp start (far field accelerated over t = $(LOWSPEED_CH_RAMP_T) c/U)" : "projection start"
    _deliver_log(log, @sprintf("Starting low-speed CFD on C+H (PBCav %s from %s, pressure-based incompressible, fully turbulent SST, Tu %.3g%%, mu_t/mu %.3g, %s%s) %s; %s; %s base cells per half; wall cap %.0f s.",
                               head[1:min(8, end)], snap, 100ls["tu"], ls["mut_ratio"], startword,
                               haskey(ls, "baffle") ? @sprintf(", Gurney baffle %s h/c %.4g%s, PB_TURB_MUT_BASE=%s", ls["baffle"]["side"], ls["baffle"]["h"],
                                                              haskey(ls["baffle"], "refine") ? ", tip refinement " * ls["baffle"]["refine"] : "", ls["mut_base"]) : "",
                               lowspeed_hardware(ls), ls["mesh_label"], knobs["PB_MESH_TE_BASE"], ls["wall_cap"]))
    started = time()
    ok = _log_process(_lowspeed_ch_cmd(snap, cfg_path), joinpath(dir, "solver.log"), progress, log, "lowspeed", Float64(alpha);
                      residual_kind="PBCav steady residual (momentum, rho U^2/c per unit volume)")
    ok && isfile(joinpath(dir, "result.toml")) || _lowspeed_ch_failure(dir, joinpath(dir, "solver.log"), "low-speed CFD C+H worker failed")
    _collect_lowspeed_ch_output(dir, Float64(alpha), geometry, ls, knobs, head, time() - started)
end

"""The Gurney flap's unsteady caveat, shown wherever a Gurney result is shown (the lead's wording, 27 Oct 2026; the source is the
URANS development point validation/gurney_urans_20261027/README.md, run on the EARLIER reshaped-mesh plate, so it is a statement
about the physics of this flap, not about this mesh)."""
const GURNEY_UNSTEADY_CAVEAT = "Steady RANS caveat: a Gurney flap sheds (the trailing-edge wake behind a bluff plate is unsteady, St about 0.15 on the plate height), and this steady pseudo-time march suppresses the shedding: the plate's back-face suction is a bound, not a time average, and the plate's drag increment is a lower bound in that sense. Development evidence (apps/airfoil_web/validation/gurney_urans_20261027/README.md, a 2 %c flap on NACA 0012 at Re 3e6, alpha 4, run time-accurately at small dt on the earlier reshaped-mesh plate): the wake sheds, and the time-mean C_D rose to 22 % above the steady value (a lower bound on the saturated shedding). Development, not validated."
const GURNEY_RUN_CAVEATS = "One mesh: grid independence is not established; before quoting a Gurney C_D, run one finer mesh level (PB_MESH_LEVEL=1, the plate re-placed by the same rule) to the steady stop. Near stall, steady branches can depend on the start: run one high-alpha point from both starts (the ramp and the projection) and compare."

"""The ramp start's record from the loads history (dt column) and the result: the first and smallest dt, the steps below the base
dt, and the rejections; and, for a flap, the statement that the loads before t = T are not meaningful. Stated limitation of the
start, not a defect of the plate."""
function _gurney_start_record(raw, meta, knobs)
    dts = [_float(r["dt"]) for r in raw]; steps = [parse(Int, r["step"]) for r in raw]; ts = [_float(r["t"]) for r in raw]
    pbdt = something(tryparse(Float64, String(get(knobs, "PB_DT", "0.25"))), 0.25)
    mode = String(get(meta, "start_mode", "projection")); T = Float64(get(meta, "ramp_T", 0.0))
    k = isempty(dts) ? 0 : argmin(dts)
    below = [i for i in eachindex(dts) if dts[i] < pbdt*(1 - 1e-9)]
    Dict{String,Any}("start"=>mode, "ramp_T"=>T, "ramp_r0"=>Float64(get(meta, "ramp_r0", 1.0)), "turb_mut_base"=>String(get(meta, "turb_mut_base", "extrapolated")),
        "dt_base"=>pbdt, "dt_first"=>isempty(dts) ? nothing : dts[1], "dt_min"=>k == 0 ? nothing : dts[k], "dt_min_step"=>k == 0 ? nothing : steps[k],
        "steps_below_base_dt"=>length(below), "last_step_below_base_dt"=>isempty(below) ? nothing : steps[last(below)],
        "accepted_steps"=>length(dts), "steps_in_ramp"=>count(t -> t !== nothing && t < T, ts), "rejections"=>String(get(meta, "rejections", "")))
end

function _collect_lowspeed_ch_output(dir, alpha, geometry, ls, knobs, head, elapsed)
    meta = TOML.parsefile(joinpath(dir, "result.toml"))
    fnum(x) = x isa Real ? Float64(x) : something(tryparse(Float64, string(x)), NaN)
    mesh = AirfoilWire.decode(read(joinpath(dir, "mesh_preview.msgpack")), :msgpack)
    mesh["artifact"] = joinpath(dir, "mesh_full.msgpack"); mesh["preview_artifact"] = joinpath(dir, "mesh_preview.msgpack")
    app_audit = _lowspeed_ch_app_audit(joinpath(dir, "mesh_full.msgpack"))
    mesh["geometry_metadata"] = _lowspeed_ch_mesh_metadata(geometry, head, ls, knobs, meta, app_audit)
    cp = Dict{String,Any}("upper"=>Any[], "lower"=>Any[])
    for row in _csv_rows(joinpath(dir, "surface_cp.csv"))
        push!(cp[row["surface"]], [_float(row["x"]), _float(row["cp"])])
    end
    base_cp = [[_float(r["y"]), _float(r["cp"])] for r in _csv_rows(joinpath(dir, "base_cp.csv"))]
    # the baffle's faces (a flap only): per plate and side (front = upstream, foil-side cells; back = downstream, wake-side
    # cells) the [y, Cp] pairs, drawn at the trailing-edge station as the base's are
    baffle = get(meta, "baffle", nothing)
    baffle_cp = nothing
    if baffle !== nothing
        baffle_cp = Dict{String,Any}()
        for r in _csv_rows(joinpath(dir, "baffle.csv"))
            push!(get!(get!(baffle_cp, String(r["plate"]), Dict{String,Any}()), String(r["side"]), Any[]), [_float(r["y"]), _float(r["Cp"])])
        end
    end
    fields = AirfoilWire.decode(read(joinpath(dir, "fields_blocks.msgpack")), :msgpack)
    get(fields, "kind", nothing) == "structured_blocks" && [b["id"] for b in fields["blocks"]] == ["C", "H"] ||
        error("the low-speed C+H worker did not return C and H block fields")
    fields["mesh_sha256"] == bytes2hex(open(sha256, joinpath(dir, "mesh_full.msgpack"))) || error("low-speed C+H field/mesh provenance differs")
    for b in fields["blocks"]; b["full_artifact"] = joinpath(dir, b["full_artifact"]); end
    fields["full_artifact"] = joinpath(dir, "fields_blocks_full.msgpack"); fields["artifact"] = joinpath(dir, "fields_blocks.msgpack")
    fields["full_cell_count"] = meta["full_field_cells"]; fields["full_iteration"] = meta["full_field_iteration"]
    fields["speed_column"] = "mach"; fields["hidden_quantities"] = ["density", "temperature"]
    fields["quantity_labels"] = Dict("mach"=>"Speed |U|/U∞", "cp"=>"Pressure coefficient Cp", "pressure"=>"Pressure (p - p∞)/(ρU∞²)",
                                     "u"=>"Velocity u/U∞", "v"=>"Velocity v/U∞", "vorticity"=>"Vorticity ωc/U∞", "nut"=>"Eddy viscosity ratio μt/μ")
    fields["quantity_units"] = Dict("mach"=>"|U|/U∞", "cp"=>"dimensionless", "pressure"=>"ρU∞²", "u"=>"U∞", "v"=>"U∞",
                                    "vorticity"=>"U∞/c", "nut"=>"μt/μ", "density"=>"ρ∞", "temperature"=>"T∞")
    raw = _csv_rows(joinpath(dir, "loads_history.csv"))
    stride = max(1, ceil(Int, length(raw)/1200))
    idx = isempty(raw) ? Int[] : unique(vcat(collect(1:stride:length(raw)), length(raw)))
    rampT = Float64(get(meta, "ramp_T", 0.0))
    history = [Dict{String,Any}("iteration"=>parse(Int, raw[k]["step"]), "residual"=>_float(raw[k]["steady"]),
        "residual_linf"=>_float(raw[k]["steady"]), "cl"=>_float(raw[k]["CL"]), "cd"=>_float(raw[k]["CD"]),
        "cm"=>_float(raw[k]["CM"]), "steady_k"=>_float(raw[k]["steady_k"]), "steady_w"=>_float(raw[k]["steady_w"]),
        "t_pseudo"=>_float(raw[k]["t"]), "in_ramp"=>(rampT > 0 && something(_float(raw[k]["t"]), Inf) < rampT)) for k in idx]
    outcome = String(meta["outcome"]); steady = outcome == "STEADY"
    convergence = steady ?
        @sprintf("solver steady stop reached at step %d: momentum steady residual %.2e <= %.0e, k and omega residuals <= %.0e, point-vortex circulation settled. Mesh independence and validation are not established.",
                 meta["steps"], fnum(meta["steady_residual"]), fnum(meta["steady_tol"]), fnum(meta["turb_steady_tol"])) :
        outcome == "INCOMPLETE" ? @sprintf("wall cap of %.0f s reached at step %d before the steady stop (momentum steady residual %.2e, stop at %.0e); diagnostic only.",
                                           fnum(meta["wall_cap"]), meta["steps"], fnum(meta["steady_residual"]), fnum(meta["steady_tol"])) :
        @sprintf("solver outcome %s at step %d without the steady stop (momentum steady residual %.2e); diagnostic only.", outcome,
                 meta["steps"], fnum(meta["steady_residual"]))
    gates = app_audit["air_reference_gates"]
    gstart = haskey(geometry, "gurney") ? _gurney_start_record(raw, meta, knobs) : nothing
    notes = String[
        "Low-speed CFD: version 2's incompressible pressure-based solver (PBCav, commit $(meta["commit"]), snapshot $(meta["snapshot"])), IN DEVELOPMENT and NOT YET VALIDATED.",
        "Finite trailing edge on PBCav's conforming C+H free-air mesh (C block around the foil, H block behind the straight base; the base is a no-slip wall and its pressure and friction are in the loads). The sharp-trailing-edge and the C+H cases run the same PBCav commit ($(head[1:min(8, end)]), the baffle snapshot on Julia 1.13.2).",
        "Fully turbulent k-omega SST (SST-2003m; no transition model): freestream Tu $(round(100*ls["tu"]; sigdigits=3))%, mu_t/mu $(ls["mut_ratio"]). " *
            (ls["start"] == "ramp" ? @sprintf("Ramp start (PB_START=ramp): the far field is accelerated from %.3g U to U over t = %.3g c/U (the projected start is singular at the plate's tip and the base corners); the loads before t = %.3g include the accelerating domain's uniform pressure gradient and are not meaningful. ", fnum(get(meta, "ramp_r0", 1.0)), rampT, rampT) :
             "Plain projection start (PBCav's Kutta start refuses a mesh with a base); ") * "The point vortex relaxes the circulation from zero.",
        "Free air: far-field velocity with a point-vortex correction, pressure outlet; Mach is ignored (incompressible). Steady pseudo-time (backward Euler, dt 0.25 c/U) to the solver's own steady stop.",
        "Mesh: $(ls["mesh_label"]); $(knobs["PB_MESH_TE_BASE"]) base cells per half. PBCav audit gates failed: $(isempty(meta["mesh_gate_failures"]) ? "none" : join(meta["mesh_gate_failures"], "; ")); the app's C+H air gates: $(gates["passed"] ? "all pass" : "failed " * join(gates["failed"], "; ")). Grid independence is not established.",
        isempty(geometry["change"]) ? "The CFD geometry is the source section with its native finite trailing edge." : geometry["change"] * ".",
        "Fields: the 'mach' column holds the speed |U|/U_inf (incompressible); density and temperature are 1; nut is mu_t/mu; no live field snapshots on the C+H path (the final field is shown)."]
    if baffle !== nothing      # the caveats come right after the first note, the plate's record at the end
        insert!(notes, 2, GURNEY_UNSTEADY_CAVEAT); insert!(notes, 3, GURNEY_RUN_CAVEATS)
        append!(notes, _gurney_baffle_notes(meta, baffle, ls))
    end
    append!(notes, streamline_point_notes(get(mesh["geometry_metadata"], "streamline_alignment", nothing)))
    Dict{String,Any}(
        "backend"=>"lowspeed", "alpha_deg"=>alpha, "status"=>steady ? "completed" : "not_converged", "convergence"=>convergence,
        "validation"=>baffle === nothing ? "version 2 pressure-based solver in development; not yet validated; app C+H path" :
                      "version 2 pressure-based solver in development; not yet validated; app C+H path with the Gurney flap as a baffle (development, one mesh, steady RANS)",
        "geometry"=>geometry, "mesh"=>mesh, "cl"=>fnum(meta["cl"]), "cd"=>fnum(meta["cd"]), "cm"=>fnum(meta["cm"]),
        "iteration_history"=>history, "cp"=>cp, "base_cp"=>base_cp, "fields"=>fields, "notes"=>notes,
        "artifacts"=>merge(Dict("cp_csv"=>joinpath(dir, "surface_cp.csv"), "base_cp_csv"=>joinpath(dir, "base_cp.csv"),
            "surface_csv"=>joinpath(dir, "surface.csv"), "loads_history_csv"=>joinpath(dir, "loads_history.csv"),
            "mesh_full"=>joinpath(dir, "mesh_full.msgpack"), "mesh_preview"=>joinpath(dir, "mesh_preview.msgpack"),
            "field_preview"=>joinpath(dir, "fields_blocks.msgpack"), "field_full"=>joinpath(dir, "fields_blocks_full.msgpack"),
            "result_toml"=>joinpath(dir, "result.toml"), "input_toml"=>joinpath(dir, "input.toml"), "log"=>joinpath(dir, "solver.log"),
            "section"=>geometry["path"]), baffle === nothing ? Dict{String,Any}() : Dict("baffle_csv"=>joinpath(dir, "baffle.csv"))),
        "diagnostics"=>merge(Dict{String,Any}("solver"=>meta["solver"], "commit"=>meta["commit"], "snapshot"=>meta["snapshot"],
            "solver_outcome"=>outcome, "steps"=>meta["steps"], "max_steps"=>meta["max_steps"], "wall_cap_s"=>fnum(meta["wall_cap"]),
            "march_wall_seconds"=>fnum(meta["march_wall_seconds"]), "point_wall_seconds"=>elapsed, "threads"=>meta["threads"],
            "time_mode"=>"steady", "clock"=>"steady pseudo-time (backward Euler, dt 0.25 c/U)",
            "convergence_guard"=>Dict("status"=>steady ? "solver_steady_stop" : "not_converged", "reason"=>convergence),
            "residual_scaled"=>fnum(meta["steady_residual"]), "residual_kind"=>"PBCav steady residual (momentum)",
            "steady_residual"=>fnum(meta["steady_residual"]), "steady_tol"=>fnum(meta["steady_tol"]),
            "steady_k"=>fnum(meta["steady_k"]), "steady_w"=>fnum(meta["steady_w"]), "turb_steady_tol"=>fnum(meta["turb_steady_tol"]),
            "cl_pressure"=>fnum(meta["cl_pressure"]), "cl_friction"=>fnum(meta["cl_friction"]),
            "cd_pressure"=>fnum(meta["cd_pressure"]), "cd_friction"=>fnum(meta["cd_friction"]), "base_cp_mean"=>fnum(meta["base_cp_mean"]),
            "circulation"=>fnum(meta["gamma"]), "circulation_defect"=>fnum(meta["gamma_defect"]), "start"=>meta["start"],
            "mass_net_flux"=>fnum(meta["flux_net"]), "mass_step_err_max"=>fnum(meta["mass_step_err_max"]),
            "rejections"=>meta["rejections"], "mut_ratio_max"=>fnum(meta["mut_ratio_max"]),
            "yplus_median"=>fnum(meta["yplus_median"]), "yplus_max"=>fnum(meta["yplus_max"]),
            "omega_bound"=>fnum(meta["omega_bound"]), "omega_bound_tail"=>meta["omega_bound_tail"],
            "reynolds"=>fnum(meta["reynolds"]), "mach"=>0.0, "freestream_tu"=>fnum(meta["tu"]), "freestream_mut_ratio"=>fnum(meta["mut_ratio"]),
            "turbulence"=>"fully turbulent k-omega SST-2003m", "mesh"=>knobs["PB_MESH"], "mesh_resolution"=>ls["resolution"],
            "mesh_cells"=>meta["mesh_cells"], "mesh_gates"=>meta["mesh_gates"], "mesh_gate_failures"=>meta["mesh_gate_failures"],
            "mesh_audit"=>meta["mesh_audit_line"], "mesh_topology"=>"conformal_C_H_v1", "app_mesh_audit"=>app_audit,
            "mesh_x_sha256"=>meta["mesh_x_sha256"], "mesh_y_sha256"=>meta["mesh_y_sha256"],
            "pbcav_knobs"=>knobs, "foil_file"=>meta["foil"], "foil_sha256"=>meta["foil_sha256"],
            "geometry_transform"=>geometry["transform"], "configuration"=>meta["config_line"], "knob_receipt"=>meta["knob_receipt"],
            "turb_mut_base"=>String(get(meta, "turb_mut_base", "extrapolated"))),
            baffle === nothing ? Dict{String,Any}() : Dict{String,Any}(
                "gurney_baffle"=>baffle, "gurney_baffle_loads"=>get(meta, "baffle_loads", Any[]),
                "gurney_start"=>gstart, "gurney_unsteady_caveat"=>GURNEY_UNSTEADY_CAVEAT)),
        "baffle"=>baffle, "baffle_cp"=>baffle_cp)
end

"""The result notes of a Gurney (baffle) point: the model and the plate as realised, the gate and the admitted range, the mesh
receipt (the PB BAFFLE line's x_sha256 / y_sha256) and the plate's share of the loads. (The caveats are placed first by the caller.)"""
function _gurney_baffle_notes(meta, baffle, ls)
    f(x) = x isa Real ? Float64(x) : something(tryparse(Float64, string(x)), NaN)
    notes = String[]
    for q in baffle["plates"]
        push!(notes, @sprintf("Gurney flap, a baffle: PBCav's interior wall (commit %s) on this section's own finite-trailing-edge C+H mesh; the mesh is not reshaped. Plate %d (%s): %d faces along the C block's grid line from the trailing-edge corner, %s onto the chord's normal (it leaned %.2f deg from it) and snapped so that its tip is exactly h = %.6g c from the corner (the mesh's nearest node was %.6g c; the requested %s h/c %.6g%s); %d nodes moved by at most %.2e c. Both sides of the plate are no-slip walls; their pressure and friction are in the loads.",
            String(meta["commit"])[1:min(8, end)], q["plate"], q["side"], q["faces"], q["straightened"] ? "straightened" : "left as it is", f(q["dev_grid_deg"]),
            f(q["h_real"]), f(q["h_grid"]), String(baffle["side"]) == "symmetric" ? "total" : "plate", f(baffle["h"]),
            String(baffle["side"]) == "symmetric" ? " (h/2 on each corner)" : "", baffle["nodes_moved"], f(baffle["disp_max"])))
    end
    lim = baffle["gate_limits"]; rg = baffle["range"]; q1 = first(baffle["plates"])
    push!(notes, @sprintf("Plate quality gate: %s (largest boundary non-orthogonality of the plate's faces %.1f deg, limit %g; tip face %.3f of h, limit %g; largest first-cell height change of a moved column %.2f %%, limit %g %%). %s",
        String(baffle["gate"]), maximum(f(q["theta_b_max_deg"]) for q in baffle["plates"]), f(lim["theta_b_max_deg"]),
        maximum(f(q["tip_fraction"]) for q in baffle["plates"]), f(lim["tip_fraction_max"]), 100f(baffle["first_cell_change"]), 100f(lim["first_cell_change_max"]),
        haskey(rg, "lo") ? @sprintf("On this mesh the gate admits h/c from %.4g to %.4g%s (%d of %d heights of the corner line%s).", f(rg["lo"]), f(rg["hi"]),
            String(baffle["side"]) == "symmetric" ? " (total, h/2 on each corner)" : "", rg["admitted"], rg["scanned"], get(rg, "contiguous", true) ? "" : ", not contiguous") :
            "No height of this mesh's corner line passes the gate."))
    rf = get(baffle, "refine", nothing)
    if rf !== nothing
        if get(rf, "admitted", false) === true
            s0 = rf["stats_before"]; s1 = rf["stats_after"]; tf0 = f(first(rf["tip_face_before"])); tf1 = f(first(rf["tip_face_after"]))
            push!(notes, @sprintf("Flap and wake refinement (PBCav PB_BAFFLE_REFINE %s): %d rows added (%d below, %d above the tip; C-block normal rows %d -> %d), H-block gap rows %d -> %d. Tip factor 1/%.3g, with cells growing back at a controlled ratio. Row density moves from the plate root and finite-TE wake-leg extension into the middle and upper flap height, following the recirculation region downstream. The H block has fewer cross rows behind the finite trailing edge. Plate tip face %.3g -> %.3g c. The first three C-block wall rows, wall and wake-leg nodes, outer boundary, H-block interfaces and H-block first edge cells retain their unrefined nodes; the plate keeps its exact height. Checks (unrefined -> refined): smallest C-cell corner angle %.2f -> %.2f deg, largest ratio of consecutive C cells along a column %.3f -> %.3f, largest C-cell aspect ratio %.0f -> %.0f. A refined and an unrefined run are two meshes: grid independence is not established.",
                String(rf["spec"]), Int(rf["rows_added"]), Int(rf["rows_below"]), Int(rf["rows_above"]), Int(rf["nj_before"]), Int(rf["nj_after"]), Int(rf["strip_ns_before"]), Int(rf["strip_ns_after"]), f(rf["q_used"]), tf0, tf1,
                f(s0["angle_min"]), f(s1["angle_min"]), f(s0["rn_max"]), f(s1["rn_max"]), f(s0["ar_max"]), f(s1["ar_max"])))
        else
            push!(notes, "Free-edge refinement (PBCav PB_BAFFLE_REFINE " * String(rf["spec"]) * ") was requested and NOT APPLIED: " * String(get(rf, "reason", "no reason recorded")) * " The run uses the section's own C+H mesh.")
        end
    end
    push!(notes, "Mesh receipt (the PB BAFFLE line's x_sha256 / y_sha256, the C-block nodes the faces are built on, after the plate's node movement and optional refinement): x $(baffle["x_sha256"]), y $(baffle["y_sha256"]). The mesh uses the section's native C+H host, the baffle's cut and, when requested, redistributed C-block rows; Create mesh and Run build the same nodes.")
    loads = get(meta, "baffle_loads", Any[])
    isempty(loads) || push!(notes, "The plate's share of the loads (pressure on both of its sides, wind axes): " *
        join([@sprintf("plate %d (%s) dC_L %.5f, dC_D %.5f (pressure %.5f, friction %.1e); mean C_p on its front (upstream) face %.3f, back (downstream) face %.3f",
            r["plate"], r["side"], f(r["dCL"]), f(r["dCD"]), f(r["dCDp"]), f(r["dCDf"]), f(r["Cp_front_mean"]), f(r["Cp_back_mean"])) for r in loads], "; ") * ".")
    push!(notes, @sprintf("Start and turbulence settings of the flap: PB_START=%s%s, PB_TURB_MUT_BASE=%s (committed: the step's frozen mu_t is formed at the committed level, the same steady fixed point; A2's march fix for the neutral travelling mode under the stress limiter).",
        String(get(meta, "start_mode", "ramp")), get(meta, "start_mode", "ramp") == "ramp" ? @sprintf(" (T %.3g c/U, r0 %.3g)", f(get(meta, "ramp_T", 0.0)), f(get(meta, "ramp_r0", 1.0))) : "",
        String(get(meta, "turb_mut_base", "extrapolated"))))
    notes
end
