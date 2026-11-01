# =====================================================================================================================
# lowspeed_worker.jl -- the Airfoil Lab worker of the "lowspeed" backend: version 2's pressure-based incompressible solver
# (PBCav, branch pbcav) in free air, run from a PINNED, READ-ONLY snapshot worktree of that branch.
#
#   julia --startup-file=no --threads=8 --project=<snapshot>/src/pbcav lowspeed_worker.jl <input.toml>
#
# It reproduces the free-air run of the snapshot's driver (tools/pbcav_driver.jl with PB_DOMAIN=freeair, PB_STOP=run,
# PB_START=kutta; freeair.jl `freeair_drive!`), statement for statement, with two additions that do not change the arithmetic:
#   * an `on_step` callback (freeair_run!'s own hook) that prints the app's machine-readable ITERATION line every step,
#     after the driver's own omega-bound census hook (W6b; it only reads; omega_bound_history.csv as the driver writes it);
# Updated 24 Oct 2026 for the merged pbcav head (snapshot S1024_A2_JL113, b7a7b29b; Julia 1.13.2): the driver's mesh path
# (remap_spec, faces built on the CPU and moved with to_backend), W6b's census hook and result tail, and the W6b knobs
# the backend names in the input's [knobs] table (PB_CPL_AU, PB_CPL_VISC_NONORTH, PB_TURB_OMEGA_BOUND).
#   * after the march, the final state is exported in the app's artifact formats (mesh_full/mesh_preview.msgpack with
#     MESH_READY printed before marching, field_full.msgpack and field_samples.csv in "structured_cells" j-major order,
#     surface_cp.csv, the solver's own loads_history.csv and surface.csv, and result.toml).
# The snapshot is never written to. The backend is KernelAbstractions' CPU() by default; the GPU (opt-in, PB_BACKEND=cuda
# in the input's knobs) is used only after taking the shared GPU lock (gpu_lock.jl).
# =====================================================================================================================
using TOML
const CFG = TOML.parsefile(ARGS[1])
const SNAP = normpath(CFG["snapshot"])
include(joinpath(SNAP, "tools", "halt_guard.jl"))           # the snapshot's SOLVER_HALT guard (refuses if present)
filter!(p -> p != "@v#.#", LOAD_PATH)                       # as the driver: never resolve from the global environment
let ap = Base.active_project()
    ap !== nothing && normpath(ap) == normpath(joinpath(SNAP, "src", "pbcav", "Project.toml")) ||
        error("lowspeed_worker.jl runs in the snapshot's PBCav environment only (active project: $(ap))")
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
println("Low-speed CFD worker: PBCav ", PBCav.PBCAV_VERSION, " from ", SNAP, " (commit ", CFG["commit"], "), Julia ",
        VERSION, ", threads ", Threads.nthreads(), ", ", get(get(CFG, "knobs", Dict()), "PB_BACKEND", "cpu") == "cuda" ? "CUDA backend requested (after the GPU lock)" : "CPU backend")
flush(stdout)

# ---- the knobs: the driver's free-air knobs as an environment Dict (read_knobs refuses anything unregistered) --------
const ENVK = Dict{String,String}(
    "PB_DOMAIN"=>"freeair", "PB_STOP"=>"run", "PB_FOIL"=>CFG["foil"], "PB_MESH"=>CFG["mesh"],
    "PB_MESH_LE_DS"=>string(get(CFG,"le_ds",0.0)), "PB_MESH_TE_BASE"=>"0", "PB_TURBULENCE"=>"sst",
    "PB_SST_TU_INF"=>string(CFG["tu"]), "PB_SST_MUT_RATIO"=>string(CFG["mut_ratio"]),
    "PB_DT"=>string(CFG["dt"]), "PB_STEADY_TOL"=>string(CFG["steady_tol"]), "PB_TURB_STEADY_TOL"=>string(CFG["turb_steady_tol"]),
    "PB_MAX_STEPS"=>string(CFG["max_steps"]), "PB_WALL_CAP"=>string(CFG["wall_cap"]), "PB_N_OUTER_MAX"=>string(CFG["n_outer_max"]),
    "PB_VORTEX"=>"point", "PB_VORTEX_EVERY"=>string(CFG["vortex_every"]), "PB_VORTEX_RELAX"=>string(CFG["vortex_relax"]),
    "PB_START"=>"kutta", "PB_OUT_DIR"=>OUT, "PB_RE"=>string(CFG["reynolds"]), "PB_ALPHA"=>string(CFG["alpha"]))
for (k, v) in get(CFG, "knobs", Dict{String,Any}())     # the backend's named settings (LOWSPEED_W6B_KNOBS); never a driver knob above
    haskey(ENVK, String(k)) && error("the input's [knobs] table may not override $(k)")
    ENVK[String(k)] = String(v)
end
const KNOBS, KNOB_SOURCE = PBCav.read_knobs(ENVK)
println(PBCav.knob_receipt(KNOBS, KNOB_SOURCE, ENVK))
# GPU (opt-in, 24 Oct 2026; the request's lowspeed_gpu): PB_BACKEND=cuda in the input's knobs, with PB_SPEED=auto (W7's
# adopted CUDA free-air row: PB_SYNC=reads, PB_GRAPHS=on, PB_CPL_GS=cgs2, PB_LINE_SOLVER=pcr, PB_DOT=tree, the
# preconditioner and k-omega in Float64; the Float32 forms stay gated by the snapshot). The mesh is built on the CPU and
# published first; the GPU lock (gpu_lock.jl) is taken only before the march, never in Create mesh (mesh_only).
const GPU_REQUESTED = KNOBS["PB_BACKEND"] === :cuda
const SPEED_AUTO = GPU_REQUESTED ? PBCav.apply_speed_auto!(KNOBS, KNOB_SOURCE) : String[]
GPU_REQUESTED && PBCav.pb_cpu_launch!(KNOBS["PB_CPU_LAUNCH"])
GPU_REQUESTED && KNOBS["PB_SPEED"] === :auto &&
    println("PB SPEED: auto, cuda freeair: ", isempty(SPEED_AUTO) ? "nothing changed" : join(SPEED_AUTO, " "))

# ---- the mesh (the driver's construction, line for line) -------------------------------------------------------------
const CALL = PBCav.mesh_call(KNOBS, KNOB_SOURCE)
const REMAP = PBCav.remap_spec(KNOBS)          # nothing for the app's knobs (PB_MESH_REMAP unset); the driver's line kept
const GENMESH = PBCav.generate_v1(CALL)
println(PBCav.mesh_receipt_line(GENMESH, PBCav.foil_path(CALL)))
const BLOCKSG = PBCav.cblocks(GENMESH; label="pbcav")
const BLOCKSR = REMAP === nothing ? BLOCKSG :
    let r = PBCav.remap_columns(BLOCKSG, REMAP); println(PBCav.remap_receipt_line(r)); r.blocks; end
const BLOCKS0 = PBCav.subdivide(BLOCKSR, KNOBS["PB_MESH_LEVEL"])
const BLOCKS = PBCav.freeair_blocks(BLOCKS0)
const MESHC = PBCav.build_faces(BLOCKS; backend=CPU())     # the CPU's faces; moved to the backend before the march
println(PBCav.face_census_line(MESHC))
const AUDIT = PBCav.mesh_audit(MESHC)
const GATES = PBCav.mesh_audit_gates(AUDIT)
println(PBCav.mesh_audit_line(AUDIT))
flush(stdout)
BLOCKS.NS == 0 || error("the low-speed backend runs sharp-trailing-edge C-meshes only (this mesh has a strip block)")
const NI = BLOCKS.NI; const NJ = BLOCKS.NJ
MESHC.nc == NI*NJ || error("unexpected cell count $(MESHC.nc) for a $(NI) x $(NJ) C block")

# ---- the app's mesh artifacts (the format of cfd_backend.jl _mesh_payload / _write_mesh_artifacts) ------------------
const GATE_FAILURES = [@sprintf("%s %.4g > %.4g", n, v, lim) for (n, v, lim, p) in GATES if !p]
const AUDIT_DICT = Dict{String,Any}(string(k)=>(v isa NamedTuple ? Dict(string(a)=>b for (a, b) in pairs(v)) :
                                                 v isa Tuple ? collect(v) : v) for (k, v) in pairs(AUDIT))
const GATE_LIST = [Dict{String,Any}("gate"=>n, "value"=>v, "limit"=>lim, "pass"=>p) for (n, v, lim, p) in GATES]
function mesh_payload(ii::Vector{Int}, jj::Vector{Int})
    X = BLOCKS.X; Y = BLOCKS.Y; NIW = BLOCKS.NIW; NIF = BLOCKS.NIF
    Dict{String,Any}(
        "kind"=>"structured", "ni"=>length(ii), "nj"=>length(jj), "full_ni"=>NI+1, "full_nj"=>NJ+1,
        "i_indices"=>ii, "j_indices"=>jj,
        "stride_i"=>length(ii) > 1 ? maximum(diff(ii)) : 1, "stride_j"=>length(jj) > 1 ? maximum(diff(jj)) : 1,
        "x"=>[X[i, j] for j in jj for i in ii], "y"=>[Y[i, j] for j in jj for i in ii],
        "wall_i_start"=>NIW+1, "wall_i_end"=>NIW+NIF+1, "foil_surface_cells"=>NIF, "wake_cells_per_side"=>NIW,
        "normal_cells"=>NJ, "base_faces_per_side"=>0, "wall_dn1"=>CALL.dn1,
        "wall_x"=>X[NIW+1:NIW+NIF+1, 1], "wall_y"=>Y[NIW+1:NIW+NIF+1, 1],
        "mesh_topology"=>"sharp_C", "quality_passed"=>isempty(GATE_FAILURES), "quality_failures"=>GATE_FAILURES,
        "quality"=>Dict{String,Any}("pbcav_mesh_audit"=>AUDIT_DICT, "pbcav_gates"=>GATE_LIST),
        "geometry_metadata"=>Dict{String,Any}("qualification"=>"PBCav mesh audit against version 1's G2 reference limits (not registered for this family)"),
        "resolution"=>String(CFG["resolution"]), "node_order"=>"j-major, i-fastest", "solver"=>"PBCav $(CFG["commit"])")
end
let ni = NI+1, nj = NJ+1
    # A bounded preview for the job response (every second node, the first eight rows, the wall/LE/TE stations); the
    # viewer replaces it with mesh_full.msgpack (every node) in the background, and the exact wall is carried separately.
    pi_ = sort!(unique(vcat(collect(1:2:ni), [1, ni, BLOCKS.NIW+1, BLOCKS.NIW+BLOCKS.NIF+1, argmin(@view BLOCKS.X[:, 1])])))
    pj = sort!(unique(vcat(collect(1:min(8, nj)), collect(1:2:nj), [nj])))
    write(joinpath(OUT, "mesh_full.msgpack"), AirfoilWire.encode(mesh_payload(collect(1:ni), collect(1:nj)), :msgpack))
    write(joinpath(OUT, "mesh_preview.msgpack"), AirfoilWire.encode(mesh_payload(pi_, pj), :msgpack))
end
println("MESH_READY,mesh_preview.msgpack")
if get(CFG, "mesh_only", false) === true          # Create mesh in the app: the run's own mesh, its receipts and audit; no march
    open(io -> TOML.print(io, Dict{String,Any}("commit"=>CFG["commit"], "mesh"=>CFG["mesh"], "mesh_cells"=>MESHC.nc,
        "ni"=>NI, "nj"=>NJ, "mesh_x_sha256"=>BLOCKS.hashes[1], "mesh_y_sha256"=>BLOCKS.hashes[2],
        "mesh_gate_failures"=>GATE_FAILURES, "mesh_audit_line"=>PBCav.mesh_audit_line(AUDIT))), joinpath(OUT, "mesh_result.toml"), "w")
    println("Low-speed CFD mesh only: ", MESHC.nc, " cells; no march")
    flush(stdout)
    exit(0)
end
@printf("Low-speed CFD mesh: %d x %d C-mesh cells (%d), %d foil faces, %d wake cells per side, first wall spacing %.3g c; PBCav audit gates: %s\n",
        NI, NJ, MESHC.nc, BLOCKS.NIF, BLOCKS.NIW, CALL.dn1, isempty(GATE_FAILURES) ? "all pass" : "FAIL " * join(GATE_FAILURES, "; "))
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

# ---- the run: freeair_drive! (freeair.jl) with the app's ITERATION callback ------------------------------------------
const k = KNOBS
const fs = PBCav.FreeStream(; alpha=k["PB_ALPHA"])
const bc = PBCav.freeair_bcset(MESH, fs; backend=BACKEND)
const cfg = PBCav.freeair_config(k; backend=BACKEND)
println(PBCav.freeair_config_line(cfg, fs, k))
const st = PBCav.allocate_state(MESH; backend=BACKEND)
const w = PBCav.pbwork(MESH, bc, cfg)
const FOIL_SHA = bytes2hex(sha256(read(CFG["foil"])))
function start_state!()
    path = get(CFG, "warm_start", nothing)
    if path !== nothing
        try
            saved = open(deserialize, path)
            saved["commit"] == CFG["commit"] || error("solver commit differs")
            saved["mesh_hashes"] == collect(BLOCKS.hashes) || error("mesh differs")
            saved["foil_sha256"] == FOIL_SHA || error("foil differs")
            saved["reynolds"] == k["PB_RE"] || error("Reynolds number differs")
            saved["tu"] == k["PB_SST_TU_INF"] || error("turbulence intensity differs")
            saved["mut_ratio"] == k["PB_SST_MUT_RATIO"] || error("freestream eddy viscosity differs")
            abs(saved["alpha_deg"] - k["PB_ALPHA"]) <= 2.5 || error("incidence change exceeds 2.5 degrees")
            all(isfinite, (saved["gamma"], saved["dt"], saved["dt_prev"])) || error("nonfinite restart scalar")
            saved["dt_prev"] > 0 || error("invalid previous time step")
            for f in PBCav.STATE_FIELDS
                data = saved[string(f)]
                size(data) == size(getfield(st, f)) || error("state size differs for $f")
                all(isfinite, data) || error("nonfinite state in $f")
            end
            for f in PBCav.STATE_FIELDS
                copyto!(getfield(st, f), saved[string(f)])
            end
            st.t = 0.0; st.nstep = 0
            # A two-degree incidence jump changed the far-field BC enough that a
            # 0.25 c/U first attempt required ten pressure-solve rejections before
            # the accepted step reached 2.44e-4 c/U (12 deg after 10 deg, Fine
            # NACA 0012, Re=6M). Start near that measured safe step; the march's
            # normal regrowth restores its chosen step without failed solves.
            incidence_jump = abs(saved["alpha_deg"] - k["PB_ALPHA"])
            first_step_cap = incidence_jump >= 1.5 ? 2.5e-4 : cfg.dt
            st.dt = min(cfg.dt, max(1e-4, saved["dt"]), first_step_cap)
            st.dt_prev = saved["dt_prev"]
            G = saved["gamma"]
            PBCav.set_farfield!(bc, MESH, fs, G; backend=BACKEND)
            @printf("PB FREEAIR START: checked warm state from alpha %.3f deg, dt %.5g, Gamma %.6g\n", saved["alpha_deg"], st.dt, G)
            return G, NaN, NaN, String(path)
        catch err
            println("PB FREEAIR START: warm state refused (", sprint(showerror, err), "); using Kutta start")
        end
    end
    G, J0, JG = PBCav.kutta_circulation(MESH, bc, cfg, fs; work=w)
    ps = PBCav.vortex_projection_start!(st, MESH, bc, cfg, fs, G; work=w)
    @printf("PB FREEAIR START: projection FCG %d, rel %.2e, max |div F|/V %.3e -> %.3e, start kutta: G_K %.6f (potential C_L %.5f), trailing-edge velocity jump %.4e -> %.4e\n",
            ps.fcg, ps.rel, ps.div_before, ps.div_after, G, 2G/(fs.U*fs.chord), J0, JG)
    G, J0, JG, ""
end
const G0, J0, JG, WARM_SOURCE = start_state!()
flush(stdout)
const pv = PBCav.PointVortex(; relax=k["PB_VORTEX_RELAX"], every=k["PB_VORTEX_EVERY"], G=G0)
const MAXS = k["PB_MAX_STEPS"]
const T_MARCH = time()
# Live flow-field snapshots (the app's P4; input key live_every > 0): every live_every committed steps the committed state is
# written as live_field_<step>.msgpack in the full-field format and announced as FIELD_SNAPSHOT,<file>,<step>. The gradients
# for the vorticity go into a separate LoadsWork, as the loads' own every step, so the march is unchanged.
const LIVE_EVERY = Int(get(CFG, "live_every", 0))
const LIVE_LW = LIVE_EVERY > 0 ? PBCav.LoadsWork(MESH; backend=BACKEND) : nothing
const LIVE_FILES = String[]
function write_live(s, step)
    PBCav.loads_gradients!(LIVE_LW, s, MESH, bc, w, cfg)
    u = Array(PBCav.lev(s.ux, 1)); v = Array(PBCav.lev(s.uy, 1)); p = Array(PBCav.lev(s.p, 1)); mt = Array(PBCav.lev(s.mut, 2))
    vo = Array(LIVE_LW.gvx) .- Array(LIVE_LW.guy); xc = Array(MESH.xc); yc = Array(MESH.yc); q = PBCav.qinf(fs)
    name = @sprintf("live_field_%06d.msgpack", step)
    FullFieldArtifact.write_full_field((i, j) -> (c = PBCav.cell_index(MESH, 0, i, j);
        (xc[c], yc[c], hypot(u[c], v[c])/fs.U, (p[c] - fs.p_inf)/q, p[c], 1.0, u[c], v[c], 1.0, vo[c], mt[c]/cfg.mu)),
        joinpath(OUT, name), NI, NJ, joinpath(OUT, "mesh_full.msgpack"), max(step, 1))
    push!(LIVE_FILES, name)
    while length(LIVE_FILES) > 3                          # keep the newest three (the viewer may still be reading one)
        old = popfirst!(LIVE_FILES)
        try rm(joinpath(OUT, old); force=true) catch end
    end
    println("FIELD_SNAPSHOT,", name, ",", step)
end
# W6b (the driver's freeair_drive!): where turb_omega_bound acted, per committed step, in omega_bound_history.csv
const BON = PBCav.omega_bound_on(cfg)
const bio = BON ? open(joinpath(OUT, "omega_bound_history.csv"), "w") : nothing
bio === nothing || println(bio, PBCav.OMEGA_BOUND_HISTORY_HEADER)
const REJECTION_KEYS = PBCav.REJECTION_REASONS
const rejection_io = open(joinpath(OUT, "rejection_history.csv"), "w")
println(rejection_io, join(("step", "t", "dt", "total_rejections", string.(REJECTION_KEYS)...), ","))
const BSTEPS = Ref(0); const BLAST = Ref(0)
function app_step(s, r, led)
    counts = (get(led.rejections, key, 0) for key in REJECTION_KEYS)
    println(rejection_io, join((r.step, r.t, r.dt, sum(values(led.rejections)), counts...), ","))
    flush(rejection_io)
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
    if LIVE_EVERY > 0 && (r.step == 5 || r.step % LIVE_EVERY == 0)
        try
            write_live(s, r.step)
        catch err
            println("Live field snapshot skipped at step ", r.step, ": ", sprint(showerror, err))
        end
    end
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
                       on_step=app_step)
finally
    close(hio)
    bio === nothing || close(bio)
    close(rejection_io)
end
# Keep a complete, mesh-checked restart for the next polar angle and for a longer
# continuation of an incomplete point. Write atomically so a killed worker cannot
# pass a partial state to the next point.
if R.outcome != :NOT_REACHED
    checkpoint = Dict{String,Any}(
        "commit"=>CFG["commit"], "mesh_hashes"=>collect(BLOCKS.hashes),
        "foil_sha256"=>FOIL_SHA, "reynolds"=>k["PB_RE"],
        "tu"=>k["PB_SST_TU_INF"], "mut_ratio"=>k["PB_SST_MUT_RATIO"],
        "alpha_deg"=>k["PB_ALPHA"], "gamma"=>pv.G,
        "dt"=>st.dt, "dt_prev"=>st.dt_prev)
    for f in PBCav.STATE_FIELDS
        checkpoint[string(f)] = Array(getfield(st, f))
    end
    tmp = joinpath(OUT, "state_checkpoint.jls.tmp")
    open(io -> serialize(io, checkpoint), tmp, "w")
    mv(tmp, joinpath(OUT, "state_checkpoint.jls"); force=true)
end
const SD = PBCav.surface_distribution(R.lw, MESH, fs)
PBCav.write_surface_csv(joinpath(OUT, "surface.csv"), SD)
const L = R.loads; const H = R.hist
const WS = PBCav.wall_shear(st, MESH, bc, cfg; L=2, form=:assembled, work=w); const YP = sort(WS.yplus)
const FC = PBCav.floor_census(st, MESH, cfg)
const MUT_MAX = PBCav.pb_maxabs(PBCav.lev(st.mut, 2); backend=BACKEND)/cfg.mu
# the final state's fields, read BEFORE the omega-bound census (omega_bound_census! overwrites level 1 with level 2)
host(x) = Array(x)
const U = host(PBCav.lev(st.ux, 1)); const V = host(PBCav.lev(st.uy, 1)); const P = host(PBCav.lev(st.p, 1))
const MUT = host(PBCav.lev(st.mut, 2)); const XC = host(MESH.xc); const YC = host(MESH.yc)
const VORT = host(R.lw.gvx) .- host(R.lw.guy)               # dv/dx - du/dy, the loads' least-squares gradients of level 1
const BCS = PBCav.omega_bound_census!(w, MESH, st, bc, cfg)     # the driver's census at the committed state (SST only here)
const OMEGA_TAIL = BON ?
    @sprintf(" omega_bound=%.3g omega_bound_steps=%d omega_bound_last_step=%d omega_bound_final_cells=%d omega_bound_final_max_cut_over_Q=%.3e omega_bound_final_at=%s",
             cfg.turb_omega_bound, BSTEPS[], BLAST[], BCS.n, BCS.ratio, BCS.at) :
    (cfg.turb_omega_bound > 0 ? @sprintf(" omega_bound=unsplit(%.3g)", cfg.turb_omega_bound) : " omega_bound=0")
@printf("PB FREEAIR RESULT: outcome=%s t=%.6g steps=%d wall=%.1f CL=%.6f CD=%.6f CDp=%.6f CDf=%.6f CM=%.6f Gamma=%.6f Gamma_defect=%.3e flux_net=%.3e mass_step_err_max=%.3e rejections=%s turbulence=sst steady_k=%.3e steady_w=%.3e mut_ratio_max=%.4g yplus_median=%.4g yplus_max=%.4g floor_k=%d floor_w=%d turb_retries=%d floor_cells_final_k=%d floor_cells_final_w=%d floor_w_at=%s%s\n",
        R.outcome, R.t, R.nstep, R.wall, L.CL, L.CD, L.CDp, L.CDf, L.CM, pv.G, isempty(H) ? NaN : H[end].Gamma_defect,
        R.fluxes.net, R.ledger.step_err_max, string(R.tally), isempty(H) ? NaN : H[end].steady_k, isempty(H) ? NaN : H[end].steady_w,
        MUT_MAX, isempty(YP) ? NaN : YP[cld(length(YP), 2)], isempty(YP) ? NaN : YP[end], R.ledger.turb_floor_k_cells,
        R.ledger.turb_floor_w_cells, R.ledger.turb_retries, FC.nk, FC.nw, FC.w_at, OMEGA_TAIL)
BON && @printf("PB FREEAIR OMEGA BOUND: at the committed state (steady assembly, bound %.3g): %d cells cut, max cut/Q %.3e at %s; cells %s; acted in %d of %d committed steps (last %d)\n",
               cfg.turb_omega_bound, BCS.n, BCS.ratio, BCS.at, BCS.cells, BSTEPS[], R.nstep, BLAST[])
isempty(R.refusal) || println("PB FREEAIR REFUSAL: ", R.refusal)
flush(stdout)

# ---- the final state in the app's formats ----------------------------------------------------------------------------
const QINF = PBCav.qinf(fs); const MU = cfg.mu
cellidx(i, j) = PBCav.cell_index(MESH, 0, i, j)
row(i, j) = (c = cellidx(i, j); (XC[c], YC[c], hypot(U[c], V[c])/fs.U, (P[c] - fs.p_inf)/QINF, P[c], 1.0, U[c], V[c], 1.0,
                                 VORT[c], MUT[c]/MU))
const NSTEP = max(R.nstep, 1)
const FULL = FullFieldArtifact.write_full_field(row, joinpath(OUT, "field_full.msgpack"), NI, NJ,
                                                joinpath(OUT, "mesh_full.msgpack"), NSTEP)
const STRIDE = max(1, ceil(Int, sqrt(NI*NJ/3000)))      # a bounded preview; field_full.msgpack keeps every cell
const SI = sort!(unique(vcat(collect(1:STRIDE:NI), [NI]))); const SJ = sort!(unique(vcat(collect(1:STRIDE:NJ), [NJ])))
open(joinpath(OUT, "field_samples.csv"), "w") do io
    println(io, join(FullFieldArtifact.COLUMNS, ","))
    for j in SJ, i in SI
        println(io, join(row(i, j), ","))
    end
end
# surface Cp: the wall faces run from the lower trailing edge round the leading edge to the upper trailing edge
const ILE = argmin(SD.x)
open(joinpath(OUT, "surface_cp.csv"), "w") do io
    println(io, "surface,x,y,cp,cf")
    for q in eachindex(SD.x)
        println(io, q > ILE ? "upper" : "lower", ",", SD.x[q], ",", SD.y[q], ",", SD.Cp[q], ",", SD.Cf[q])
    end
end
const CLp = L.CLp; const CLf = L.CLf
result = Dict{String,Any}(
    "solver"=>"PBCav $(PBCav.PBCAV_VERSION)", "commit"=>CFG["commit"], "snapshot"=>SNAP, "threads"=>Threads.nthreads(),
    "outcome"=>string(R.outcome), "exit_code"=>PBCav.freeair_exit_code(R.outcome), "steps"=>R.nstep, "max_steps"=>MAXS,
    "t_ctu"=>R.t, "march_wall_seconds"=>R.wall, "worker_wall_seconds"=>time() - T_START,
    "cl"=>L.CL, "cd"=>L.CD, "cm"=>L.CM, "cd_pressure"=>L.CDp, "cd_friction"=>L.CDf, "cl_pressure"=>CLp, "cl_friction"=>CLf,
    "gamma"=>pv.G, "gamma_start"=>G0, "gamma_kutta_start"=>isempty(WARM_SOURCE) ? G0 : NaN,
    "warm_start_source"=>WARM_SOURCE,
    "gamma_defect"=>isempty(H) ? NaN : H[end].Gamma_defect,
    "steady_residual"=>isempty(H) ? NaN : H[end].steady, "steady_k"=>isempty(H) ? NaN : H[end].steady_k,
    "steady_w"=>isempty(H) ? NaN : H[end].steady_w, "steady_tol"=>k["PB_STEADY_TOL"], "turb_steady_tol"=>k["PB_TURB_STEADY_TOL"],
    "flux_net"=>R.fluxes.net, "mass_step_err_max"=>R.ledger.step_err_max, "rejections"=>string(R.tally),
    "refusal"=>R.refusal, "mut_ratio_max"=>MUT_MAX, "yplus_median"=>isempty(YP) ? NaN : YP[cld(length(YP), 2)],
    "yplus_max"=>isempty(YP) ? NaN : YP[end], "floor_cells_final_k"=>FC.nk, "floor_cells_final_w"=>FC.nw,
    "omega_bound"=>cfg.turb_omega_bound, "omega_bound_steps"=>BSTEPS[], "omega_bound_tail"=>OMEGA_TAIL,
    "knob_receipt"=>PBCav.knob_receipt(KNOBS, KNOB_SOURCE, ENVK),
    "reynolds"=>k["PB_RE"], "alpha_deg"=>k["PB_ALPHA"], "tu"=>k["PB_SST_TU_INF"], "mut_ratio"=>k["PB_SST_MUT_RATIO"],
    "dt"=>k["PB_DT"], "wall_cap"=>k["PB_WALL_CAP"], "n_outer_max"=>k["PB_N_OUTER_MAX"], "mesh"=>CFG["mesh"],
    "mesh_cells"=>MESH.nc, "ni"=>NI, "nj"=>NJ, "n_foil"=>BLOCKS.NIF, "n_wake"=>BLOCKS.NIW,
    "mesh_x_sha256"=>BLOCKS.hashes[1], "mesh_y_sha256"=>BLOCKS.hashes[2],
    "foil"=>CFG["foil"], "foil_sha256"=>bytes2hex(sha256(read(CFG["foil"]))),
    "mesh_gates"=>Dict(n=>p for (n, _, _, p) in GATES), "mesh_gate_failures"=>GATE_FAILURES,
    "mesh_audit_line"=>PBCav.mesh_audit_line(AUDIT), "config_line"=>PBCav.freeair_config_line(cfg, fs, k),
    "full_field_cells"=>FULL["cell_count"], "full_field_mesh_sha256"=>FULL["mesh_sha256"], "full_field_iteration"=>NSTEP,
    "sample_stride"=>STRIDE, "sampled_ni"=>length(SI), "sampled_nj"=>length(SJ))
for (key, v) in collect(result)
    v isa AbstractFloat && !isfinite(v) && (result[key] = string(v))
end
open(io -> TOML.print(io, result), joinpath(OUT, "result.toml"), "w")
for f in LIVE_FILES                                   # the final field replaces the live snapshots
    try rm(joinpath(OUT, f); force=true) catch end
end
@printf("Low-speed CFD worker finished: %s after %d steps, %.0f s march, %.0f s total\n", R.outcome, R.nstep, R.wall, time() - T_START)
flush(stdout)
exit(0)   # the outcome is in result.toml; the backend sets the point status from it
