# =====================================================================================================================
# lowspeed_ch_worker.jl -- the Airfoil Lab worker of the "lowspeed" backend on a FINITE BASE (the Gurney flap, 23 Oct 2026;
# the flap as a BAFFLE since 28 Oct 2026): PBCav (version 2) in free air on its conforming C + H ("strip") mesh, run from a
# PINNED, READ-ONLY snapshot worktree.
#
#   julia --startup-file=no --threads=8 --project=<snapshot>/src/pbcav lowspeed_ch_worker.jl <input.toml>
#
# The sharp-trailing-edge worker (lowspeed_worker.jl, the same snapshot) is separate; this one is used only when the request
# carries a Gurney flap (or the hidden native-base C+H control). It reproduces the snapshot driver's free-air run
# (tools/pbcav_driver.jl with PB_DOMAIN=freeair, PB_STOP=run: mesh_call, remap_spec, generate_v1, cblocks, remap_columns,
# subdivide, freeair_blocks, apply_baffle, build_faces on the CPU, then freeair.jl `freeair_drive!`) statement for statement,
# and adds only the app's output: an ITERATION line per step through freeair_run!'s own on_step hook (called after the
# driver's omega-bound census hook), MESH_READY before marching, and the final state in the app's two-block formats
# (mesh_full/mesh_preview.msgpack with kind "structured_blocks", fields_blocks(.._full).msgpack with one structured_cells
# field per block, surface_cp.csv for the foil faces, base_cp.csv for the base faces, baffle.csv for the plate's faces, the
# solver's loads_history.csv and surface.csv, and result.toml). No live field snapshots are written (the app's live viewer is
# single-block). The snapshot is never written to. The CPU is the default backend; the GPU (opt-in, PB_BACKEND=cuda) is used
# only after taking the shared GPU lock (gpu_lock.jl).
#
# THE GURNEY FLAP IS A BAFFLE (the user's rulings of 27 Oct 2026): a zero-thickness plate normal to the chord at the trailing
# edge, exactly sharp, a wall INSIDE the section's normal finite-trailing-edge C+H mesh along the C block's grid line from the
# trailing-edge corner. This worker does not reshape the mesh and inserts nothing into the CBlocks: with PB_BAFFLE in the
# input's knobs (gurney:side=lower|upper|symmetric,h=<h/c>), PBCav's own `apply_baffle` straightens that grid line onto the
# chord's normal, snaps the tip to h exactly (a smooth movement of the neighbouring nodes) and cuts the faces
# (`build_faces(...; baffle)`). The gate of src/pbcav/baffle.jl refuses a plate outside the mesh's range of h: the refusal is
# written (baffle_refusal.toml, the GURNEY_BAFFLE_REFUSED line) and the worker exits 3, for the app to show.
# =====================================================================================================================
using TOML
const CFG = TOML.parsefile(ARGS[1])
const SNAP = normpath(CFG["snapshot"])
include(joinpath(SNAP, "tools", "halt_guard.jl"))           # the snapshot's SOLVER_HALT guard (refuses if present)
filter!(p -> p != "@v#.#", LOAD_PATH)                       # as the driver: never resolve from the global environment
let ap = Base.active_project()
    ap !== nothing && normpath(ap) == normpath(joinpath(SNAP, "src", "pbcav", "Project.toml")) ||
        error("lowspeed_ch_worker.jl runs in the snapshot's PBCav environment only (active project: $(ap))")
end
include(joinpath(SNAP, "src", "pbcav", "PBCav.jl")); using .PBCav
using KernelAbstractions
using Printf
using SHA
using Serialization
include(joinpath(@__DIR__, "..", "protocol.jl")); using .AirfoilWire
include(joinpath(@__DIR__, "..", "full_field_artifact.jl")); using .FullFieldArtifact

const OUT = CFG["outdir"]
mkpath(OUT)
const T_START = time()
println("Low-speed CFD C+H worker: PBCav ", PBCav.PBCAV_VERSION, " from ", SNAP, " (commit ", CFG["commit"], "), Julia ",
        VERSION, ", threads ", Threads.nthreads(), ", ", get(CFG["knobs"], "PB_BACKEND", "cpu") == "cuda" ? "CUDA backend requested (after the GPU lock)" : "CPU backend")
flush(stdout)

# ---- the knobs: every PB_* value the app sets is named here (read_knobs refuses anything unregistered) ----------------
const ENVK = Dict{String,String}(String(k)=>String(v) for (k, v) in CFG["knobs"])
for (k, v) in ("PB_DOMAIN"=>"freeair", "PB_STOP"=>"run", "PB_OUT_DIR"=>OUT,
               "PB_RE"=>string(CFG["reynolds"]), "PB_ALPHA"=>string(CFG["alpha"]), "PB_FOIL"=>CFG["foil"])
    ENVK[k] = v
end
haskey(ENVK, "PB_START") || (ENVK["PB_START"] = "projection")   # the plain projection start unless the app asked for the ramp
const KNOBS, KNOB_SOURCE = PBCav.read_knobs(ENVK)
println(PBCav.knob_receipt(KNOBS, KNOB_SOURCE, ENVK))
# GPU (opt-in, 24 Oct 2026; the request's lowspeed_gpu): PB_BACKEND=cuda in the input's knobs, with PB_SPEED=auto (W7's
# adopted CUDA free-air row: PB_SYNC=reads, PB_GRAPHS=on, PB_CPL_GS=cgs2, PB_LINE_SOLVER=pcr, PB_DOT=tree, the
# preconditioner and k-omega in Float64; the Float32 forms stay gated by the snapshot). The mesh is built on the CPU and
# published first; the GPU lock (gpu_lock.jl) is taken only before the march, never in Create mesh (mesh_only).
const GPU_REQUESTED = KNOBS["PB_BACKEND"] === :cuda
const SPEED_AUTO = GPU_REQUESTED ? PBCav.apply_speed_auto!(KNOBS, KNOB_SOURCE) : String[]
PBCav.pb_cpu_launch!(KNOBS["PB_CPU_LAUNCH"])
GPU_REQUESTED && KNOBS["PB_SPEED"] === :auto &&
    println("PB SPEED: auto, cuda freeair: ", isempty(SPEED_AUTO) ? "nothing changed" : join(SPEED_AUTO, " "))

# ---- the mesh (the driver's construction, line for line) -------------------------------------------------------------
const CALL = PBCav.mesh_call(KNOBS, KNOB_SOURCE)
const REMAP = PBCav.remap_spec(KNOBS)
const GENMESH = PBCav.generate_v1(CALL)
println(PBCav.mesh_receipt_line(GENMESH, PBCav.foil_path(CALL)))
const BLOCKSG = PBCav.cblocks(GENMESH; label="pbcav")
const BLOCKSR = REMAP === nothing ? BLOCKSG :
    let r = PBCav.remap_columns(BLOCKSG, REMAP); println(PBCav.remap_receipt_line(r)); r.blocks; end
# ---- streamline alignment (app layer, opt-in: absent from an unaligned input, which is then unchanged bit for bit): the
# app's mesh_streamline_morph.jl moves the nodes of the host mesh onto the JXfoil wake path and stagnation streamline the app
# computed (a hashed path file), with the requested admission rule judged by the app's C+H air gates (CMeshQualityV1, read
# from the checkout's src/cmesh/cmesh_quality_metrics.jl). Pure Julia; the snapshot is not touched. The baffle (below) is
# placed AFTER it, on the aligned, subdivided blocks, exactly as it is placed on the unaligned ones ----------------------
const STREAM = get(CFG, "streamline", nothing)
STREAM === nothing || (include(String(STREAM["morph_module"])); include(String(STREAM["quality_module"])))
const ALIGN = STREAM === nothing ? nothing : let b = BLOCKSR
    MSM = Main.MeshStreamlineMorph; QM = Main.CMeshQualityMetrics
    paths = MSM.read_streamline_paths(String(STREAM["paths"]); expected=String(STREAM["paths_sha256"]))
    params = MSM.params_from(get(STREAM, "params", Dict{String,Any}()))
    quality(X, Y, Xs, Ys) = (s = QM.quality_summary(X, Y, b.NIW, b.NIF; strip=(X=Xs, Y=Ys));
        Dict{String,Any}("gates"=>[Dict{String,Any}("gate"=>g[1], "value"=>g[3], "limit"=>g[4], "pass"=>g[5])
                                   for g in QM.quality_gates(s; cavitating=false, strip_grade=true)]))
    X2, Y2, Xs2, Ys2, rep = MSM.admit_morph(b.X, b.Y, b.NIW, b.NIF, b.Xs, b.Ys, paths; quality, params,
                                            rule=Symbol(String(STREAM["admission"])))
    adm = rep["admission"]
    wake = MSM.aligned_wake_path(b.X, b.Y, b.NIW, b.NIF, b.Xs, b.Ys, paths["wake"]; params)
    println(@sprintf("STREAMLINE ALIGNMENT: %s path (sha256 %s), %s admission: %s, TE amplitude %.3f, LE amplitude %.3f, %d attempts; gates worse: %s; nodes sha256 %s",
        String(paths["source"]), paths["sha256"][1:12], adm["rule"], adm["stage"], adm["te_amplitude"], adm["le_amplitude"],
        length(adm["attempts"]), isempty(adm["gates_worse"]) ? "none" : join(adm["gates_worse"], ", "), rep["morphed_sha256"][1:12]))
    (blocks=PBCav.CBlocks(b.label * "+aligned", b.NI, b.NJ, b.NIW, b.NIF, X2, Y2, b.kind, b.NIs, b.NS, Xs2, Ys2, b.level,
                          (PBCav.h64(X2), PBCav.h64(Y2), PBCav.h64(Xs2), PBCav.h64(Ys2))),
     report=rep, paths=paths, wake=wake)
end
const BLOCKSA = ALIGN === nothing ? BLOCKSR : ALIGN.blocks
const BLOCKS0 = PBCav.subdivide(BLOCKSA, KNOBS["PB_MESH_LEVEL"])
const BLOCKS1 = PBCav.freeair_blocks(BLOCKS0)

# ---- the Gurney baffle (opt-in: PB_BAFFLE empty, the default, changes nothing and prints nothing) --------------------------
# The driver's step: apply_baffle on the free-air blocks, its receipt PB BAFFLE, then build_faces(...; baffle). The gate of
# baffle.jl may refuse the plate (its range of h depends on the mesh): the message is the gate's own, with the admitted range.
const HAS_BAFFLE = isdefined(PBCav, :apply_baffle)       # false on the previous pin (S1024_A2_JL113): the no-flap cases still run there, for a test
const BSPEC = HAS_BAFFLE ? PBCav.baffle_spec(KNOBS) : nothing
"The gate's admitted range of h on this mesh, as a TOML-friendly table (lo and hi are absent when no height passes)."
function range_record(r)
    d = Dict{String,Any}("admitted"=>length(r.admitted), "scanned"=>r.scanned, "contiguous"=>r.contiguous)
    isempty(r.admitted) || (d["lo"] = r.lo; d["hi"] = r.hi)
    d
end
function refuse_baffle(e::ArgumentError)
    msg = String(e.msg)
    println("GURNEY_BAFFLE_REFUSED,", replace(msg, r"\s+" => " "))
    rec = Dict{String,Any}("message"=>msg, "spec"=>PBCav.spec_string(BSPEC), "side"=>String(BSPEC.side), "h"=>BSPEC.h,
                           "snapshot_commit"=>CFG["commit"])
    try; rec["range"] = range_record(PBCav.baffle_h_range(BLOCKS1, BSPEC)); catch; end
    open(io -> TOML.print(io, rec), joinpath(OUT, "baffle_refusal.toml"), "w")
    flush(stdout)
    exit(3)
end
const BAFFLE = !HAS_BAFFLE ? (BLOCKS1, nothing) : try
    PBCav.apply_baffle(BLOCKS1, KNOBS)
catch e
    e isa ArgumentError || rethrow()
    refuse_baffle(e)
end
const BLOCKS = BAFFLE[1]; const PLAN = BAFFLE[2]
PLAN === nothing || println(PBCav.baffle_receipt_line(PLAN))
# the refinement of the plate's free edge (PB_BAFFLE_REFINE, 30 Oct 2026; absent from the previous pins and empty by default): its receipt
const HAS_REFINE = isdefined(PBCav, :baffle_refine_receipt_line)
PLAN === nothing || !HAS_REFINE || (let l = PBCav.baffle_refine_receipt_line(PLAN); isempty(l) || println(l); end)
const H_RANGE = BSPEC === nothing ? nothing : PBCav.baffle_h_range(BLOCKS1, BSPEC)
const MESHC = PLAN === nothing ? PBCav.build_faces(BLOCKS; backend=CPU()) : PBCav.build_faces(BLOCKS; backend=CPU(), baffle=PLAN)   # the CPU's faces; moved to the backend before the march
println(PBCav.face_census_line(MESHC))
const AUDIT = PBCav.mesh_audit(MESHC)
const GATES = PBCav.mesh_audit_gates(AUDIT)
println(PBCav.mesh_audit_line(AUDIT))
flush(stdout)
BLOCKS.NS > 0 || error("lowspeed_ch_worker.jl runs C + H (strip) meshes only; this mesh has no strip block")
const NI = BLOCKS.NI; const NJ = BLOCKS.NJ; const NIS = BLOCKS.NIs; const NS = BLOCKS.NS
MESHC.nc == NI*NJ + NIS*NS || error("unexpected cell count $(MESHC.nc) for C $(NI) x $(NJ) + H $(NIS) x $(NS)")

# ---- the plate's record: the receipt, the realised plates and the mesh hashes (the nodes the faces are built on) ---------
"The refinement's record (the mesh metadata's gurney_baffle.refine): nothing without PB_BAFFLE_REFINE; with it the request, whether it
was admitted (and the reason when not), the rows added, the plate's tip face before and after, the quality statistics before and
after and every attempt."
function refine_record()
    (PLAN === nothing || !HAS_REFINE || PLAN.refine === nothing) && return nothing
    r = PLAN.refine
    st(x) = Dict{String,Any}(string(k)=>v for (k, v) in pairs(x) if k != :cross_at)
    att = [Dict{String,Any}("q"=>a.q, "ok"=>a.ok, "reason"=>a.reason) for a in r.attempts]
    d = Dict{String,Any}("version"=>"PBBaffleRefineV2", "spec"=>PBCav.refine_spec_string(r.spec), "admitted"=>r.admitted,
        "receipt"=>PBCav.baffle_refine_receipt_line(PLAN), "attempts"=>att)
    r.admitted ? merge!(d, Dict{String,Any}("q_used"=>r.q_used, "rows_added"=>r.rows_added, "rows_below"=>r.rows_below,
        "rows_above"=>r.rows_above, "nj_before"=>r.NJ_before, "nj_after"=>r.NJ_after, "columns_refined"=>r.columns_refined,
        "columns_full"=>r.columns_full, "tip_face_before"=>collect(Float64, r.tip_face_before), "tip_face_after"=>collect(Float64, r.tip_face_after),
        "stats_before"=>st(r.stats_before), "stats_after"=>st(r.stats_after),
        "strip_ns_before"=>r.strip_ns_before, "strip_ns_after"=>r.strip_ns_after,
        "strip_stats_before"=>r.strip_stats_before === nothing ? nothing : st(r.strip_stats_before),
        "strip_stats_after"=>r.strip_stats_after === nothing ? nothing : st(r.strip_stats_after))) : (d["reason"] = r.reason)
    d
end

"The baffle's record for the mesh metadata and result.toml: the receipt line (its x_sha256 / y_sha256 are the mesh's
provenance: PBCav.node_hashes of the MOVED C nodes, not BLOCKS.hashes), the plates' realised values, the gate and its range."
function baffle_record()
    PLAN === nothing && return nothing
    hx, hy = PBCav.node_hashes(PLAN.blocks)
    fail = PBCav.baffle_gate_failures(PLAN)
    plates = [Dict{String,Any}("plate"=>q.plate, "side"=>String(q.side), "column"=>q.column, "faces"=>q.faces, "tip_row"=>q.tip_row,
        "h_requested"=>q.h, "h_grid"=>q.h_grid, "h_real"=>q.h_real, "dev_grid_deg"=>q.dev_grid, "dev_real_deg"=>q.dev_real,
        "straightened"=>q.straightened, "snapped"=>q.snapped, "disp_max"=>q.disp_max, "window"=>q.window,
        "window_columns"=>q.window_columns, "tip"=>[q.tip[1], q.tip[2]], "root"=>[q.root[1], q.root[2]],
        "normal"=>[q.nx, q.ny], "theta_b_max_deg"=>q.theta_b_max, "theta_b_row"=>q.theta_b_row, "d1_min"=>q.d1_min,
        "d1_max"=>q.d1_max, "tip_fraction"=>q.tip_fraction, "growth_kink"=>q.growth_kink) for q in PLAN.plates]
    rec = Dict{String,Any}("version"=>"PBBaffleGurneyV2", "spec"=>PBCav.spec_string(BSPEC), "side"=>String(BSPEC.side), "h"=>BSPEC.h,
        "receipt"=>PBCav.baffle_receipt_line(PLAN), "x_sha256"=>hx, "y_sha256"=>hy,
        "gate"=>isempty(fail) ? "pass" : "FAIL", "gate_failures"=>fail,
        "gate_limits"=>Dict{String,Any}("theta_b_max_deg"=>PBCav.BAFFLE_THETA_B_MAX, "tip_fraction_max"=>PBCav.BAFFLE_TIP_FRACTION_MAX,
                                        "first_cell_change_max"=>PBCav.BAFFLE_FIRST_CELL_MAX),
        "nodes_moved"=>PLAN.moved, "disp_max"=>PLAN.disp_max, "first_cell_change"=>PLAN.first_cell_change,
        "angle_min_before"=>PLAN.angle_min_before, "angle_min_after"=>PLAN.angle_min_after,
        "chord_le"=>[PLAN.chord[1], PLAN.chord[2]], "chord_dir"=>[PLAN.chord[3], PLAN.chord[4]],
        "range"=>range_record(H_RANGE), "plates"=>plates)
    rr = refine_record()
    rr === nothing || (rec["refine"] = rr)
    rec
end
const BAFFLE_RECORD = baffle_record()
"The plates to draw: per plate the node polyline of the grid line from the corner (root) to the tip, on the moved nodes."
baffle_polylines() = PLAN === nothing ? Any[] : [Dict{String,Any}("plate"=>q.plate, "side"=>String(q.side),
    "x"=>[Float64(BLOCKS.X[q.column, j]) for j in 1:q.tip_row], "y"=>[Float64(BLOCKS.Y[q.column, j]) for j in 1:q.tip_row],
    "faces"=>q.faces, "h_real"=>q.h_real, "h_requested"=>q.h, "tip"=>[q.tip[1], q.tip[2]], "root"=>[q.root[1], q.root[2]])
    for q in PLAN.plates]
const MESH_HASHES = PLAN === nothing ? (BLOCKS.hashes[1], BLOCKS.hashes[2]) : (BAFFLE_RECORD["x_sha256"], BAFFLE_RECORD["y_sha256"])

# ---- the app's two-block mesh artifacts (the format of ch_mesh.jl _write_ch_mesh_artifacts) ---------------------------
const GATE_FAILURES = [@sprintf("%s %.4g > %.4g", n, v, lim) for (n, v, lim, p) in GATES if !p]
const AUDIT_DICT = Dict{String,Any}(string(k)=>(v isa NamedTuple ? Dict(string(a)=>b for (a, b) in pairs(v)) :
                                                 v isa Tuple ? collect(v) : v) for (k, v) in pairs(AUDIT))
const GATE_LIST = [Dict{String,Any}("gate"=>n, "value"=>v, "limit"=>lim, "pass"=>p) for (n, v, lim, p) in GATES]
safe(x::AbstractDict) = Dict{String,Any}(string(k)=>safe(v) for (k, v) in pairs(x))
safe(x::AbstractVector) = [safe(v) for v in x]
safe(x::Tuple) = [safe(v) for v in x]
safe(x::AbstractFloat) = isfinite(x) ? x : nothing
safe(x::Symbol) = String(x)
safe(x) = x
function block_payload(X, Y, id, full)
    ni, nj = size(X)
    ii = full ? collect(1:ni) : sort!(unique(vcat(collect(1:2:ni), [ni])))
    jj = full ? collect(1:nj) : sort!(unique(vcat(collect(1:min(8, nj)), collect(1:2:nj), [nj])))
    Dict{String,Any}("id"=>id, "kind"=>"structured", "ni"=>length(ii), "nj"=>length(jj), "full_ni"=>ni, "full_nj"=>nj,
        "i_indices"=>ii, "j_indices"=>jj, "x"=>[Float64(X[i, j]) for j in jj for i in ii], "y"=>[Float64(Y[i, j]) for j in jj for i in ii],
        "stride_i"=>length(ii) > 1 ? maximum(diff(ii)) : 1, "stride_j"=>length(jj) > 1 ? maximum(diff(jj)) : 1, "node_order"=>"j-major, i-fastest")
end
"The alignment's record for the mesh metadata (the morph report without its long per-attempt arrays duplicated)."
function streamline_record()
    rep = ALIGN.report; p = ALIGN.paths
    Dict{String,Any}("aligned"=>true, "tag"=>rep["tag"], "path_source"=>p["source"], "path_label"=>p["source_label"],
        "path_sha256"=>p["sha256"], "alpha_deg"=>p["alpha_deg"], "reynolds"=>p["reynolds"], "fallback_reason"=>get(p, "fallback_reason", ""),
        "cl_viscous"=>get(get(p, "viscous", Dict()), "cl", nothing), "cl_inviscid"=>p["cl_inviscid"], "cl_field"=>p["cl_field"],
        "section_note"=>String(get(STREAM, "section_note", "")), "admission"=>rep["admission"], "params"=>rep["params_used"],
        "te"=>rep["te"], "le"=>rep["le"], "source_nodes_sha256"=>rep["source_sha256"], "morphed_nodes_sha256"=>rep["morphed_sha256"],
        "foil_first_spacing_max_rel_change"=>rep["foil_first_spacing_max_rel_change"],
        "wake_first_spacing_max_rel_change"=>rep["wake_first_spacing_max_rel_change"])
end
"The streamlines to draw over the mesh: the wake path the mesh follows (moved to the mesh's trailing-edge point) to 8 c of
arc and the stagnation streamline to 3 c, at most 800 points each."
function streamline_overlay()
    thin(x, y, s, L) = (k = findlast(<=(L), s); idx = unique(vcat(collect(1:max(1, cld(k, 800)):k), k)); Dict("x"=>Float64.(x[idx]), "y"=>Float64.(y[idx])))
    w = ALIGN.wake; sw = zeros(length(w.x)); for q in 2:length(w.x); sw[q] = sw[q-1] + hypot(w.x[q] - w.x[q-1], w.y[q] - w.y[q-1]); end
    st = ALIGN.paths["stagnation_streamline"]
    Dict{String,Any}("wake"=>thin(w.x, w.y, sw, 8.0), "stagnation"=>thin(st["x"], st["y"], st["s"], 3.0),
        "stagnation_point"=>[Float64(ALIGN.paths["stagnation"]["x"]), Float64(ALIGN.paths["stagnation"]["y"])],
        "source"=>ALIGN.paths["source"], "label"=>ALIGN.paths["source_label"], "alpha_deg"=>ALIGN.paths["alpha_deg"],
        "admission"=>ALIGN.report["admission"]["rule"], "note"=>String(get(STREAM, "section_note", "")))
end
function mesh_payload(full::Bool)
    X = BLOCKS.X; Y = BLOCKS.Y; NIW = BLOCKS.NIW; NIF = BLOCKS.NIF
    blocks = [block_payload(X, Y, "C", full), block_payload(BLOCKS.Xs, BLOCKS.Ys, "H", full)]
    d = Dict{String,Any}("kind"=>"structured_blocks", "ni"=>blocks[1]["ni"], "nj"=>blocks[1]["nj"], "full_ni"=>NI+1, "full_nj"=>NJ+1,
        "blocks"=>blocks, "resolution"=>String(CFG["resolution"]),
        "wall_x"=>Float64.(X[NIW+1:NIW+NIF+1, 1]), "wall_y"=>Float64.(Y[NIW+1:NIW+NIF+1, 1]),
        "topology"=>"conformal_C_H_v1", "mesh_topology"=>"conformal_C_H_v1",
        "quality"=>safe(Dict{String,Any}("pbcav_mesh_audit"=>AUDIT_DICT, "pbcav_gates"=>GATE_LIST)),
        "quality_passed"=>isempty(GATE_FAILURES), "quality_status"=>isempty(GATE_FAILURES) ? "passed_pbcav_audit" : "pbcav_audit_gate_failures_recorded",
        "quality_failures"=>GATE_FAILURES, "solver_supported"=>true,
        "solver_support_message"=>"Low-speed CFD (PBCav $(CFG["commit"][1:min(8, end)])) on its free-air C+H mesh: conforming C block around the foil, H block behind the finite base" *
            (PLAN === nothing ? "" : "; the Gurney flap is a baffle (an interior wall) on the C block's grid line from the trailing-edge corner" *
                (BAFFLE_RECORD !== nothing && haskey(BAFFLE_RECORD, "refine") ? (BAFFLE_RECORD["refine"]["admitted"] ?
                    @sprintf("; the plate's free edge and its shear layer are refined (%d rows added, cells 1/%.3g of the section's own at the tip)", BAFFLE_RECORD["refine"]["rows_added"], BAFFLE_RECORD["refine"]["q_used"]) :
                    "; the requested refinement of the plate's free edge was NOT applied (see the baffle record's refine.reason)") : "")),
        "geometry_metadata"=>merge(Dict{String,Any}("qualification"=>"PBCav mesh audit against version 1's G2 reference limits (not registered for this family)"),
            PLAN === nothing ? Dict{String,Any}() : Dict{String,Any}("gurney_baffle"=>safe(BAFFLE_RECORD)),
            ALIGN === nothing ? Dict{String,Any}() : Dict{String,Any}("streamline_alignment"=>safe(streamline_record()))),
        "total_cells"=>NI*NJ + NIS*NS, "normal_cells"=>NJ, "wake_cells_per_side"=>NIW, "foil_surface_cells"=>NIF,
        "base_faces_per_side"=>NS ÷ 2, "base_faces"=>NS, "wall_dn1"=>CALL.dn1, "preview"=>!full, "full_resolution_available"=>true,
        "interfaces"=>[Dict("from"=>"C", "to"=>"H", "side"=>"lower", "conforming"=>true),
                       Dict("from"=>"C", "to"=>"H", "side"=>"upper", "conforming"=>true)],
        "solver"=>"PBCav $(CFG["commit"])")
    PLAN === nothing || (d["baffles"] = baffle_polylines())      # the plate(s) to draw over the mesh and the field
    d
end
aligned_payload(full) = (d = mesh_payload(full); ALIGN === nothing || (d["streamlines"] = streamline_overlay()); d)
write(joinpath(OUT, "mesh_preview.msgpack"), AirfoilWire.encode(aligned_payload(false), :msgpack))
write(joinpath(OUT, "mesh_full.msgpack"), AirfoilWire.encode(aligned_payload(true), :msgpack))
println("MESH_READY,mesh_preview.msgpack")
const MESH_RESULT = Dict{String,Any}("commit"=>CFG["commit"], "mesh_cells"=>MESHC.nc, "ni"=>NI, "nj"=>NJ, "nis"=>NIS, "ns"=>NS,
    "mesh_x_sha256"=>MESH_HASHES[1], "mesh_y_sha256"=>MESH_HASHES[2],
    "strip_x_sha256"=>bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(BLOCKS.Xs))))),
    "strip_y_sha256"=>bytes2hex(sha256(reinterpret(UInt8, vec(Float64.(BLOCKS.Ys))))),
    "mesh_full_sha256"=>bytes2hex(open(sha256, joinpath(OUT, "mesh_full.msgpack"))),
    "mesh_gate_failures"=>GATE_FAILURES, "mesh_audit_line"=>PBCav.mesh_audit_line(AUDIT),
    "foil"=>CFG["foil"], "foil_sha256"=>bytes2hex(sha256(read(CFG["foil"]))))
BAFFLE_RECORD === nothing || (MESH_RESULT["baffle"] = BAFFLE_RECORD)
if ALIGN !== nothing          # the alignment's receipts (absent from an unaligned point)
    MESH_RESULT["streamline_aligned"] = true
    MESH_RESULT["streamline_path_sha256"] = ALIGN.paths["sha256"]
    MESH_RESULT["streamline_path_source"] = String(ALIGN.paths["source"])
    MESH_RESULT["streamline_admission"] = ALIGN.report["admission"]["rule"]
    MESH_RESULT["streamline_admission_stage"] = ALIGN.report["admission"]["stage"]
    MESH_RESULT["streamline_te_amplitude"] = ALIGN.report["admission"]["te_amplitude"]
    MESH_RESULT["streamline_le_amplitude"] = ALIGN.report["admission"]["le_amplitude"]
    MESH_RESULT["streamline_gates_worse"] = String.(ALIGN.report["admission"]["gates_worse"])
    MESH_RESULT["streamline_source_nodes_sha256"] = ALIGN.report["source_sha256"]
    MESH_RESULT["streamline_morphed_nodes_sha256"] = ALIGN.report["morphed_sha256"]
end
if get(CFG, "mesh_only", false) === true          # Create mesh in the app: the run's own mesh, its receipts and audit; no march
    open(io -> TOML.print(io, MESH_RESULT), joinpath(OUT, "mesh_result.toml"), "w")
    println("Low-speed CFD C+H mesh only: ", MESHC.nc, " cells; no march")
    flush(stdout)
    exit(0)
end
@printf("Low-speed CFD C+H mesh: C %d x %d + H %d x %d cells (%d), %d foil faces, %d base faces, first wall spacing %.3g c; PBCav audit gates: %s\n",
        NI, NJ, NIS, NS, MESHC.nc, BLOCKS.NIF, NS, CALL.dn1, isempty(GATE_FAILURES) ? "all pass" : "FAIL " * join(GATE_FAILURES, "; "))
flush(stdout)

# ---- the backend: the CPU, or (opt-in) the GPU after its shared private lock -----------
include(joinpath(@__DIR__, "..", "gpu_lock.jl"))
const GPU_LOCK = GPU_REQUESTED ? AirfoilGPULock.acquire!("output $(OUT)") : nothing
GPU_REQUESTED && AirfoilGPULock.gate_wait(() -> PBCav.gpu_gate("the app's low-speed worker (PB_BACKEND=cuda)"))   # waits while a julia process is on the GPU
GPU_REQUESTED && @eval using CUDA
const BACKEND = GPU_REQUESTED ?
    (CUDA.functional() || error("PB_BACKEND=cuda: CUDA.jl reports no functional GPU (CUDA.functional() is false)"); CUDA.CUDABackend()) : CPU()
GPU_REQUESTED && PBCav.pb_graph_backend!(f -> CUDA.capture(f), CUDA.instantiate, CUDA.launch)
GPU_REQUESTED && println("PB BACKEND: cuda, ", CUDA.name(CUDA.device()), "; CUDA.jl ", pkgversion(CUDA), ", runtime ",
                         CUDA.runtime_version(), "; the mesh is built on the CPU and moved")
const MESH = PBCav.to_backend(BACKEND, MESHC)
flush(stdout)

# ---- the run: freeair_drive! (freeair.jl) with the app's ITERATION callback ---------------------------------------------
# The start: the plain projection start (default), or the RAMP start (PB_START=ramp; the Gurney flap's: the freestream is
# accelerated smoothly because the projected potential flow is singular at the plate's tip and the base corners, records
# air/baffle/README.md section 3). Kutta's start refuses a mesh with a base or a baffle and is not offered.
const k = KNOBS
const fs = PBCav.FreeStream(; alpha=k["PB_ALPHA"])
const bc = PBCav.freeair_bcset(MESH, fs; backend=BACKEND)
const cfg = PBCav.freeair_config(k; backend=BACKEND)
const SST = cfg.turbulence === :sst
println(PBCav.freeair_config_line(cfg, fs, k))
const RAMP = get(k, "PB_START", :projection) === :ramp ? k["PB_START_RAMP"] : nothing
const st = PBCav.allocate_state(MESH; backend=BACKEND)
const Ux, Uy = PBCav.uinf(fs)
const w = PBCav.pbwork(MESH, bc, cfg)
const FOIL_SHA = bytes2hex(sha256(read(CFG["foil"])))
function checked_warm_state!()
    path = get(CFG, "warm_start", nothing)
    path === nothing && return nothing
    try
        saved = open(deserialize, path)
        saved["topology"] == "conformal_C_H_v1" || error("topology differs")
        saved["commit"] == CFG["commit"] || error("solver commit differs")
        saved["mesh_hashes"] == collect(MESH_HASHES) || error("C-block mesh differs")
        saved["strip_hashes"] == [MESH_RESULT["strip_x_sha256"], MESH_RESULT["strip_y_sha256"]] || error("H-block mesh differs")
        saved["foil_sha256"] == FOIL_SHA || error("foil differs")
        saved["baffle"] == get(ENVK, "PB_BAFFLE", "") || error("baffle differs")
        saved["reynolds"] == k["PB_RE"] || error("Reynolds number differs")
        saved["tu"] == k["PB_SST_TU_INF"] || error("turbulence intensity differs")
        saved["mut_ratio"] == k["PB_SST_MUT_RATIO"] || error("eddy viscosity differs")
        abs(saved["alpha_deg"] - k["PB_ALPHA"]) <= 2.5 || error("incidence jump exceeds 2.5 degrees")
        all(isfinite, (saved["gamma"], saved["dt"], saved["dt_prev"])) && saved["dt_prev"] > 0 || error("invalid restart scalars")
        for f in PBCav.STATE_FIELDS
            data = saved[string(f)]
            size(data) == size(getfield(st, f)) && all(isfinite, data) || error("invalid state in $f")
        end
        for f in PBCav.STATE_FIELDS; copyto!(getfield(st, f), saved[string(f)]); end
        st.t = 0.0; st.nstep = 0
        jump = abs(saved["alpha_deg"] - k["PB_ALPHA"])
        st.dt = min(cfg.dt, max(1e-4, saved["dt"]), jump >= 1.5 ? 2.5e-4 : cfg.dt)
        st.dt_prev = saved["dt_prev"]
        PBCav.set_farfield!(bc, MESH, fs, saved["gamma"]; backend=BACKEND)
        @printf("PB FREEAIR START: checked C+H warm state from alpha %.3f deg, dt %.5g, Gamma %.6g\n", saved["alpha_deg"], st.dt, saved["gamma"])
        return (source=String(path), gamma=saved["gamma"])
    catch err
        println("PB FREEAIR START: C+H warm state refused (", sprint(showerror, err), "); using configured fresh start")
        return nothing
    end
end
const WARM = checked_warm_state!()
const RUNKW = WARM !== nothing || RAMP === nothing ? (;) : (; ramp=RAMP)
if WARM === nothing && RAMP === nothing
    PBCav.init_state!(st, MESH, (x, y) -> (Ux, Uy, fs.p_inf); backend=BACKEND)
    SST && PBCav.init_turbulence!(st, MESH, bc, cfg; work=w)
end
const ps = WARM !== nothing ? nothing : RAMP === nothing ? PBCav.project_start!(st, MESH, bc, cfg; work=w) : PBCav.ramp_start!(st, MESH, bc, cfg, fs, RAMP; work=w)
const START_TAIL = RAMP === nothing ? "" :
    @sprintf(", start ramp: the far field from %.4g U to U over t = %.4g c/U (r = r0 + (1 - r0)(1 - cos(pi t/T))/2)", RAMP.r0, RAMP.T)
WARM === nothing && @printf("PB FREEAIR START: projection FCG %d, rel %.2e, max |div F|/V %.3e -> %.3e%s\n", ps.fcg, ps.rel, ps.div_before, ps.div_after, START_TAIL)
flush(stdout)
const pv = k["PB_VORTEX"] === :point ? PBCav.PointVortex(; relax=k["PB_VORTEX_RELAX"], every=k["PB_VORTEX_EVERY"], G=WARM === nothing ? 0.0 : WARM.gamma) : nothing
const MAXS = k["PB_MAX_STEPS"]
const T_MARCH = time()
const BON = PBCav.omega_bound_on(cfg)
const bio = BON ? open(joinpath(OUT, "omega_bound_history.csv"), "w") : nothing
bio === nothing || println(bio, PBCav.OMEGA_BOUND_HISTORY_HEADER)
const BSTEPS = Ref(0); const BLAST = Ref(0)
function app_step(s, r, led)
    if BON                                           # the driver's census hook, first (it only reads)
        n = 0; c = nothing
        if PBCav.pb_maxabs(w.turb.wcut; backend=BACKEND) > 0
            c = PBCav.omega_bound_census(w, MESH, cfg; ncells=0); n = c.n; BSTEPS[] += 1; BLAST[] = r.step
        end
        bio === nothing || (println(bio, join((r.step, r.t, n, n > 0 ? c.ratio : 0.0, n > 0 ? c.at_b : 0,
                                               n > 0 ? c.at_i : 0, n > 0 ? c.at_j : 0), ",")); flush(bio))
    end
    sk = r.steady_k; sw = r.steady_w
    tr = isfinite(sk) && isfinite(sw) ? max(sk, sw) : r.steady
    @printf("ITERATION,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n", r.step, MAXS, r.steady, r.steady, tr,
            r.CL, r.CD, r.flux_net, r.CM)
    if r.step % 25 == 0
        @printf("PB progress: step %d of at most %d, wall %.0f s, CL %.5f, CD %.6f, CM %.5f, steady residual %.3e (stop at %.1e), k/omega %.3e/%.3e (stop at %.1e), circulation defect %.2e\n",
                r.step, MAXS, time() - T_MARCH, r.CL, r.CD, r.CM, r.steady, k["PB_STEADY_TOL"], sk, sw, k["PB_TURB_STEADY_TOL"], r.Gamma_defect)
    end
    flush(stdout)
end
const hio = open(joinpath(OUT, "loads_history.csv"), "w")
const R = try
    PBCav.freeair_run!(st, MESH, bc, cfg, fs; wall_cap=k["PB_WALL_CAP"], vortex=pv, max_steps=MAXS,
                       steady_tol=k["PB_STEADY_TOL"], turb_steady_tol=k["PB_TURB_STEADY_TOL"], history=hio, work=w,
                       on_step=app_step, RUNKW...)
finally
    close(hio)
    bio === nothing || close(bio)
end
if R.outcome != :NOT_REACHED
    checkpoint = Dict{String,Any}(
        "topology"=>"conformal_C_H_v1", "commit"=>CFG["commit"],
        "mesh_hashes"=>collect(MESH_HASHES),
        "strip_hashes"=>[MESH_RESULT["strip_x_sha256"], MESH_RESULT["strip_y_sha256"]],
        "foil_sha256"=>FOIL_SHA, "baffle"=>get(ENVK, "PB_BAFFLE", ""),
        "reynolds"=>k["PB_RE"], "tu"=>k["PB_SST_TU_INF"],
        "mut_ratio"=>k["PB_SST_MUT_RATIO"], "alpha_deg"=>k["PB_ALPHA"],
        "gamma"=>pv === nothing ? 0.0 : pv.G, "dt"=>st.dt, "dt_prev"=>st.dt_prev)
    for f in PBCav.STATE_FIELDS; checkpoint[string(f)] = Array(getfield(st, f)); end
    tmp = joinpath(OUT, "state_checkpoint.jls.tmp")
    open(io -> serialize(io, checkpoint), tmp, "w")
    mv(tmp, joinpath(OUT, "state_checkpoint.jls"); force=true)
end
const SD = PBCav.surface_distribution(R.lw, MESH, fs)
PBCav.write_surface_csv(joinpath(OUT, "surface.csv"), SD)
# the baffle's faces (only on a mesh that has one): baffle.csv with plate, side and row, and each plate's share of the loads
const BLOADS = PLAN === nothing ? NamedTuple[] : PBCav.baffle_loads(SD, MESH, fs)
if PLAN !== nothing
    PBCav.write_baffle_csv(joinpath(OUT, "baffle.csv"), PBCav.baffle_table(SD, MESH))
    println(PBCav.baffle_loads_line(SD, MESH, fs))
end
const L = R.loads; const H = R.hist
const WS = PBCav.wall_shear(st, MESH, bc, cfg; L=2, form=:assembled, work=w); const YP = sort(WS.yplus)
const FC = PBCav.floor_census(st, MESH, cfg)
const MUT_MAX = PBCav.pb_maxabs(PBCav.lev(st.mut, 2); backend=BACKEND)/cfg.mu
# the final state's fields, read BEFORE the omega-bound census (omega_bound_census! overwrites level 1 with level 2)
host(x) = Array(x)
const U = host(PBCav.lev(st.ux, 1)); const V = host(PBCav.lev(st.uy, 1)); const P = host(PBCav.lev(st.p, 1))
const MUT = host(PBCav.lev(st.mut, 2)); const XC = host(MESH.xc); const YC = host(MESH.yc)
const VORT = host(R.lw.gvx) .- host(R.lw.guy)
const BCS = SST ? PBCav.omega_bound_census!(w, MESH, st, bc, cfg) : nothing
const OMEGA_TAIL = !SST ? "" : BON ?
    @sprintf(" omega_bound=%.3g omega_bound_steps=%d omega_bound_last_step=%d omega_bound_final_cells=%d omega_bound_final_max_cut_over_Q=%.3e omega_bound_final_at=%s",
             cfg.turb_omega_bound, BSTEPS[], BLAST[], BCS.n, BCS.ratio, BCS.at) :
    (cfg.turb_omega_bound > 0 ? @sprintf(" omega_bound=unsplit(%.3g)", cfg.turb_omega_bound) : " omega_bound=0")
@printf("PB FREEAIR RESULT: outcome=%s t=%.6g steps=%d wall=%.1f CL=%.6f CD=%.6f CDp=%.6f CDf=%.6f CM=%.6f Gamma=%.6f Gamma_defect=%.3e flux_net=%.3e mass_step_err_max=%.3e rejections=%s turbulence=sst steady_k=%.3e steady_w=%.3e mut_ratio_max=%.4g yplus_median=%.4g yplus_max=%.4g floor_k=%d floor_w=%d turb_retries=%d floor_cells_final_k=%d floor_cells_final_w=%d floor_w_at=%s%s\n",
        R.outcome, R.t, R.nstep, R.wall, L.CL, L.CD, L.CDp, L.CDf, L.CM, pv === nothing ? 0.0 : pv.G, isempty(H) ? NaN : H[end].Gamma_defect,
        R.fluxes.net, R.ledger.step_err_max, string(R.tally), isempty(H) ? NaN : H[end].steady_k, isempty(H) ? NaN : H[end].steady_w,
        MUT_MAX, isempty(YP) ? NaN : YP[cld(length(YP), 2)], isempty(YP) ? NaN : YP[end], R.ledger.turb_floor_k_cells,
        R.ledger.turb_floor_w_cells, R.ledger.turb_retries, FC.nk, FC.nw, FC.w_at, OMEGA_TAIL)
isempty(R.refusal) || println("PB FREEAIR REFUSAL: ", R.refusal)
flush(stdout)

# ---- the final state in the app's two-block formats ------------------------------------------------------------------
const QINF = PBCav.qinf(fs); const MU = cfg.mu
const NSTEP = max(R.nstep, 1)
const MESH_PATH = joinpath(OUT, "mesh_full.msgpack"); const MESH_SHA = bytes2hex(open(sha256, MESH_PATH))
rowc(c) = (XC[c], YC[c], hypot(U[c], V[c])/fs.U, (P[c] - fs.p_inf)/QINF, P[c], 1.0, U[c], V[c], 1.0, VORT[c], MUT[c]/MU)
blocks = Any[]; fullblocks = Any[]; total = 0
for (id, blk, bni, bnj) in (("C", 0, NI, NJ), ("H", 1, NIS, NS))
    at(i, j) = rowc(PBCav.cell_index(MESH, blk, i, j))
    stride = max(1, ceil(Int, sqrt(bni*bnj/6000)))
    ii = sort!(unique(vcat(collect(1:stride:bni), [bni]))); jj = sort!(unique(vcat(collect(1:stride:bnj), [bnj])))
    samples = Dict(name=>Float64[] for name in FullFieldArtifact.COLUMNS)
    for j in jj, i in ii
        for (name, value) in zip(FullFieldArtifact.COLUMNS, at(i, j))
            isfinite(value) || error("nonfinite $name in block $id at ($i, $j)")
            push!(samples[name], value)
        end
    end
    filename = "field_" * id * "_full.msgpack"
    full = FullFieldArtifact.write_full_field(at, joinpath(OUT, filename), bni, bnj, MESH_PATH, NSTEP)
    global total += full["cell_count"]
    push!(blocks, Dict{String,Any}("id"=>id, "kind"=>"structured_cells", "ni"=>length(ii), "nj"=>length(jj),
        "samples"=>samples, "sample_shape"=>[length(ii), length(jj)], "sample_stride"=>stride,
        "i_indices"=>ii, "j_indices"=>jj, "full_ni"=>bni, "full_nj"=>bnj, "iteration"=>NSTEP, "mesh_sha256"=>MESH_SHA,
        "cell_order"=>"j-major, i-fastest", "full_artifact"=>filename))
    push!(fullblocks, Dict("id"=>id, "artifact"=>filename, "ni"=>bni, "nj"=>bnj))
end
const COMMON = Dict{String,Any}("kind"=>"structured_blocks", "cell_order"=>"j-major, i-fastest", "mesh_sha256"=>MESH_SHA,
    "iteration"=>NSTEP, "total_cells"=>total, "coordinate_units"=>"chord")
write(joinpath(OUT, "fields_blocks.msgpack"), AirfoilWire.encode(merge(COMMON, Dict("blocks"=>blocks, "full_artifact"=>"fields_blocks_full.msgpack")), :msgpack))
write(joinpath(OUT, "fields_blocks_full.msgpack"), AirfoilWire.encode(merge(COMMON, Dict("blocks"=>fullblocks)), :msgpack))
# surface Cp: foil faces (lower trailing edge -> leading edge -> upper trailing edge), the base faces and, separately, the
# baffle's faces (baffle.csv; the plate's two sides are not part of the foil's curve)
const NONFOIL = HAS_BAFFLE ? (Int(PBCav.BT_BASE), Int(PBCav.BT_BAFFLE_FRONT), Int(PBCav.BT_BAFFLE_BACK)) : (Int(PBCav.BT_BASE),)
const FOILQ = findall(t -> !(t in NONFOIL), SD.tag); const BASEQ = findall(t -> t == Int(PBCav.BT_BASE), SD.tag)
const ILE = FOILQ[argmin(SD.x[FOILQ])]
open(joinpath(OUT, "surface_cp.csv"), "w") do io
    println(io, "surface,x,y,cp,cf")
    for q in FOILQ
        println(io, q > ILE ? "upper" : "lower", ",", SD.x[q], ",", SD.y[q], ",", SD.Cp[q], ",", SD.Cf[q])
    end
end
open(joinpath(OUT, "base_cp.csv"), "w") do io
    println(io, "x,y,cp,cf")
    for q in sort(BASEQ; by=q -> SD.y[q])
        println(io, SD.x[q], ",", SD.y[q], ",", SD.Cp[q], ",", SD.Cf[q])
    end
end
const CPB = isempty(BASEQ) ? NaN : sum(SD.Cp[q]*SD.ds[q] for q in BASEQ)/sum(SD.ds[q] for q in BASEQ)
const START_TEXT = RAMP === nothing ? "projection (plain; Kutta refuses a base)" :
    @sprintf("ramp (the far field accelerated from %.4g U to U over t = %.4g c/U; Kutta refuses a base and a baffle)", RAMP.r0, RAMP.T)
result = merge(MESH_RESULT, Dict{String,Any}(
    "solver"=>"PBCav $(PBCav.PBCAV_VERSION)", "snapshot"=>SNAP, "threads"=>Threads.nthreads(), "topology"=>"conformal_C_H_v1",
    "rans_acceleration"=>Bool(get(CFG,"rans_acceleration",false)),
    "rans_acceleration_method"=>String(get(CFG,"rans_acceleration_method","standard")),
    "requested_pseudo_dt"=>parse(Float64,String(CFG["knobs"]["PB_DT"])),
    "outcome"=>string(R.outcome), "exit_code"=>PBCav.freeair_exit_code(R.outcome), "steps"=>R.nstep, "max_steps"=>MAXS,
    "t_ctu"=>R.t, "march_wall_seconds"=>R.wall, "worker_wall_seconds"=>time() - T_START,
    "cl"=>L.CL, "cd"=>L.CD, "cm"=>L.CM, "cd_pressure"=>L.CDp, "cd_friction"=>L.CDf, "cl_pressure"=>L.CLp, "cl_friction"=>L.CLf,
    "base_cp_mean"=>CPB, "base_faces"=>length(BASEQ),
    "gamma"=>pv === nothing ? 0.0 : pv.G, "gamma_kutta_start"=>0.0, "start"=>WARM === nothing ? START_TEXT : "checked warm state",
    "warm_start_source"=>WARM === nothing ? "" : WARM.source,
    "start_mode"=>WARM !== nothing ? "warm" : RAMP === nothing ? "projection" : "ramp",
    "ramp_T"=>WARM !== nothing || RAMP === nothing ? 0.0 : RAMP.T,
    "ramp_r0"=>WARM !== nothing || RAMP === nothing ? 1.0 : RAMP.r0,
    "turb_mut_base"=>hasproperty(cfg, :turb_mut_base) ? string(cfg.turb_mut_base) : "extrapolated",
    "gamma_defect"=>isempty(H) ? NaN : H[end].Gamma_defect,
    "steady_residual"=>isempty(H) ? NaN : H[end].steady, "steady_k"=>isempty(H) ? NaN : H[end].steady_k,
    "steady_w"=>isempty(H) ? NaN : H[end].steady_w, "steady_tol"=>k["PB_STEADY_TOL"], "turb_steady_tol"=>k["PB_TURB_STEADY_TOL"],
    "flux_net"=>R.fluxes.net, "mass_step_err_max"=>R.ledger.step_err_max, "rejections"=>string(R.tally),
    "refusal"=>R.refusal, "mut_ratio_max"=>MUT_MAX, "yplus_median"=>isempty(YP) ? NaN : YP[cld(length(YP), 2)],
    "yplus_max"=>isempty(YP) ? NaN : YP[end], "floor_cells_final_k"=>FC.nk, "floor_cells_final_w"=>FC.nw,
    "omega_bound"=>cfg.turb_omega_bound, "omega_bound_steps"=>BSTEPS[], "omega_bound_tail"=>OMEGA_TAIL,
    "reynolds"=>k["PB_RE"], "alpha_deg"=>k["PB_ALPHA"], "tu"=>k["PB_SST_TU_INF"], "mut_ratio"=>k["PB_SST_MUT_RATIO"],
    "dt"=>k["PB_DT"], "wall_cap"=>k["PB_WALL_CAP"], "n_outer_max"=>k["PB_N_OUTER_MAX"], "mesh"=>k["PB_MESH"] isa AbstractString ? k["PB_MESH"] : string(CFG["knobs"]["PB_MESH"]),
    "n_foil"=>BLOCKS.NIF, "n_wake"=>BLOCKS.NIW, "knob_receipt"=>PBCav.knob_receipt(KNOBS, KNOB_SOURCE, ENVK),
    "mesh_gates"=>Dict(n=>p for (n, _, _, p) in GATES), "config_line"=>PBCav.freeair_config_line(cfg, fs, k),
    "full_field_cells"=>total, "full_field_mesh_sha256"=>MESH_SHA, "full_field_iteration"=>NSTEP))
if PLAN !== nothing      # each plate's share of the loads (PB BAFFLE LOADS) and its two sides' mean Cp
    result["baffle_loads"] = [Dict{String,Any}("plate"=>r.plate, "side"=>String(r.side), "faces"=>r.faces, "dCL"=>r.dCL, "dCD"=>r.dCD,
        "dCLp"=>r.dCLp, "dCDp"=>r.dCDp, "dCLf"=>r.dCLf, "dCDf"=>r.dCDf, "Cp_front_mean"=>r.Cp_front, "Cp_back_mean"=>r.Cp_back) for r in BLOADS]
end
for (key, v) in collect(result)
    v isa AbstractFloat && !isfinite(v) && (result[key] = string(v))
    v isa Symbol && (result[key] = String(v))
end
open(io -> TOML.print(io, result), joinpath(OUT, "result.toml"), "w")
@printf("Low-speed CFD C+H worker finished: %s after %d steps, %.0f s march, %.0f s total\n", R.outcome, R.nstep, R.wall, time() - T_START)
flush(stdout)
exit(0)   # the outcome is in result.toml; the backend sets the point status from it
