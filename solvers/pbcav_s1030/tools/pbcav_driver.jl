include(joinpath(@__DIR__, "halt_guard.jl"))
# =====================================================================================================================
# pbcav_driver.jl -- the version-2 solver driver (the pressure-based real-water cavitation solver, src/pbcav/).
# Plan: docs/2026-10-13_PRESSURE_BASED_SOLVER_PLAN.md section 6 (amended bc79814f). Stage P0b, 14 Oct 2026.
#
# Usage (one thread, version 2's own environment):
#   julia --startup-file=no --threads=1 --project=src/pbcav tools/pbcav_driver.jl
#
# KNOBS. Only PB_* variables, each registered in src/pbcav/knobs.jl (PB_KNOBS) with its default; an unregistered PB_*
# variable, or any version-1 knob (TOPO_*, TEM_*, CAV_MOVIE_*), is refused before anything is built. The defaults are N60R's
# mesh. The registered knobs today:
#   PB_STOP=mesh|run         mesh: build the mesh, print the receipts, stop (exit 0); run: march (not built yet: refuses)
#   PB_BACKEND=cpu|cuda      the KernelAbstractions backend. cuda (stage P3, W3 of plan Amendment 12, 21 Oct 2026): this file
#                            loads CUDA.jl only then (a CPU run never loads it or touches a GPU), refuses when nvidia-smi
#                            shows a julia process on the GPU or one the system cannot identify (one GPU owner at a time; PBCav.gpu_gate,
#                            review of W3 F11), builds the mesh on the CPU (tunnel and free air: review of W3, F6)
#                            and moves it with PBCav.to_backend, so every receipt below is the CPU's bit for bit
#   PB_FOIL, PB_MESH, PB_MESH_LE_DS, PB_MESH_TE_BASE, PB_MESH_TE_DS, PB_MESH_WAKE_NEAR, PB_MESH_TE_TOPOLOGY,
#   PB_CHANNEL_H, PB_CHANNEL_XMIN, PB_CHANNEL_XMAX, PB_ALPHA   version 1's TOPO_ mesh knobs, same meaning
#   PB_MESH_LEVEL=k          k bilinear subdivisions of the generated mesh (the nested levels of plan 10.1)
#   PB_TURBULENCE=laminar|sst, PB_SST_TU_INF, PB_SST_MUT_RATIO, PB_WALL_OMEGA=v1|auto   the turbulence model (W1, 21 Oct
#                            2026; turbulence_settings gives PBConfig's keywords; read by the free-air run, below)
#   PB_DOMAIN=tunnel|freeair free air (W2, plan section 32, 21 Oct 2026; opt-in): version 1's generator channel of
#                            PB_FARFIELD_H/XMIN/XMAX at PB_FARFIELD_ANGLE with every outer face far field and the downstream end
#                            a pressure outlet; PB_ALPHA is then the freestream's incidence. PB_STOP=mesh also prints the mesh
#                            audit (PB MESH AUDIT); PB_STOP=run runs the free-air case (PB_RE, PB_DT, PB_MAX_STEPS,
#                            PB_STEADY_TOL, PB_WALL_CAP, PB_N_OUTER_MAX, PB_VORTEX, PB_VORTEX_RELAX, PB_VORTEX_EVERY, PB_OUT_DIR,
#                            PB_TURB_STEADY_TOL and the turbulence knobs above) and prints PB FREEAIR CONFIG (the configuration
#                            that ran), PB FREEAIR START, PB FREEAIR RESULT and PB FREEAIR EXIT. The tunnel's PB_STOP=run still
#                            refuses (stage P3). In free air the tunnel's PB_MESH_TE_DS and PB_MESH_WAKE_NEAR defaults are not
#                            applied (set them to cluster the trailing edge; knobs.jl, mesh_call)
#   PB_SYNC, PB_LINE_SOLVER, PB_PREC_PRECISION, PB_KW_PRECISION, PB_CPL_GS, PB_GRAPHS, PB_DOT, PB_CPU_LAUNCH, PB_SPEED   W7's speed settings (plan
#                            Amendment 13, 22 Oct 2026; every default the path before W7; knobs.jl, speed_settings). On
#                            CUDA, while its row is PROVISIONAL, PB_SPEED=auto and every non-default numerical speed knob
#                            are refused unless PB_SPEED_DEV=1 (a development run; printed in the receipts)
#   PB_TURB_COUPLING, PB_TURB_TOL, PB_TURB_STEP_TOL, PB_TURB_NONORTH, PB_TURB_OUTER_RAMP   the SST solve's settings (the
#                            W1+W2 integration, 22 Oct 2026), PBConfig's defaults
#   PB_TURB_MUT, PB_TURB_OMEGA_NONORTH, PB_TURB_OUTER_TOL, PB_START=projection|kutta   W6 (22 Oct 2026; listed here by W6b,
#                            the review of W6, R12): the momentum's mu_t within a step (frozen), omega's non-orthogonal form
#                            (wall_off), :outer's own acceptance, and the free-air start (kutta: the lifting potential flow)
#   PB_TURB_OMEGA_BOUND, PB_CPL_AU, PB_CPL_VISC_NONORTH   W6b (22 Oct 2026; the review of W6, R1 and R2): the bound on omega's
#                            net non-orthogonal correction (0.2 of the cell's steady budget), SIMPLEC's alpha_u in the coupled
#                            step's preconditioner only (0.8 PBConfig's default; 0.99 A2's measured value, registered per mesh),
#                            and the free-air run's coupled J with (jacobian) or without (deferred) the momentum's viscous
#                            non-orthogonal part. Free air only: the tunnel refuses them (read_knobs; the review of W6b, F12).
#                            Under SST with the bound on, the free-air run prints the bound's census at the committed state
#                            (PB FREEAIR RESULT's omega_bound_* fields, PB FREEAIR OMEGA BOUND) and writes its per-step history
#                            (PB_OUT_DIR/omega_bound_history.csv; the review of W6b, F1)
#   PB_MESH_REMAP=none|common  W4 (22 Oct 2026; opt-in, free air only): the generated mesh's C columns remapped before any
#                            subdivision (PB_MESH_REMAP_KEEP, _LAW, _WAKE, _STALL, _STRIP; src/pbcav/mesh_remap.jl); prints PB MESH
#                            REMAP (the construction and the remapped node hashes) after version 1's receipt, which stays the
#                            generator's. none (the default) prints nothing new and builds the generator's mesh bit for bit. The
#                            remap is host work on the generator's node arrays, before build_faces, so on PB_BACKEND=cuda the
#                            remapped mesh is built on the CPU and moved like any other (the merge of W4/W4b, 22 Oct 2026)
#   PB_BAFFLE=gurney:side=lower|upper|symmetric,h=<h/c>[,snap=exact|nearest][,straighten=<deg>|off][,shear=<s>]   interior
#                            walls (24 Oct 2026; opt-in, src/pbcav/baffle.jl): a zero-thickness Gurney plate on the C column from
#                            the trailing-edge corner, its faces cut into walls on both sides; placed after the far-field tags
#                            and before the faces are built, it prints PB BAFFLE (the plate, its realised height and straightness,
#                            the nodes moved and their hashes) and, after a free-air run, PB BAFFLE LOADS and PB_OUT_DIR/baffle.csv.
#                            Empty (the default) changes nothing and prints nothing new (PB KNOBS lists it, as every knob)
#   PB_BAFFLE_REFINE=tip:q=...  refinement of the plate's free edge and its shear layer (30 Oct 2026; opt-in, src/pbcav/baffle.jl;
#                            needs PB_BAFFLE): rows added around the tip row and along the grid row through it downstream, the
#                            tip cells 1/q of the mesh's own; admitted by convexity, angle, ratio and aspect-ratio checks, else
#                            the unrefined mesh with the reasons printed (PB BAFFLE REFINE). Empty (the default): nothing
#   PB_START=ramp, PB_START_RAMP=T,r0   the ramp start (24 Oct 2026; opt-in, src/pbcav/freeair.jl): the freestream accelerated
#                            smoothly from r0 U to U over t = T from the projected potential flow at r0 U, instead of the
#                            impulsive (projection) start, whose potential flow is singular at the base corners and a plate's tip
#
# EXIT CODES (the W1+W2 integration; review of W2, R3; the 14 Oct 2026 rule that a non-zero exit is a FAIL): 0 for a mesh stop
# and for a free-air run that is STEADY or REACHED; 2 when the run was refused (NOT_REACHED); 3 when it reached its wall cap
# (INCOMPLETE); 1 for an error (an uncaught exception, e.g. a refused configuration).
#
# WHAT IT PRINTS, in order: the knob receipt (PB KNOBS), version 1's MESH COORDINATE RECEIPT of the generated mesh
# (byte for byte the version-1 driver's line, so receipt binders and CSV readers run unchanged), the nested level's
# receipt (PB MESH LEVEL) when PB_MESH_LEVEL > 0, and the face census (PB FACE CENSUS).
#
# ISOLATION (plan 6.2). The halt guard is this file's first expression (tools/audit_solver_halt.jl checks it). The driver
# refuses any environment other than src/pbcav's, and drops the user's global environment (@v#.#) from the load path,
# which would otherwise resolve CUDA 5.11.0 / KernelAbstractions 0.9.41 [audit EC]. Version 1's driver cannot reach this
# file or src/pbcav/ (tools/check_code_map.jl verifies it).
# =====================================================================================================================
filter!(p -> p != "@v#.#", LOAD_PATH)
const ROOT = normpath(joinpath(@__DIR__, ".."))
const PB_PROJECT = joinpath(ROOT, "src", "pbcav", "Project.toml")
let ap = Base.active_project()
    ap !== nothing && normpath(ap) == normpath(PB_PROJECT) ||
        error("tools/pbcav_driver.jl runs in version 2's environment only: julia --project=src/pbcav " *
              "(the active project is $(ap === nothing ? "none" : ap))")
end
isempty(ARGS) || error("tools/pbcav_driver.jl takes no arguments; its settings are PB_* knobs (got $(ARGS))")

include(joinpath(ROOT, "src", "pbcav", "PBCav.jl")); using .PBCav
using KernelAbstractions

const KNOBS, KNOB_SOURCE = PBCav.read_knobs(ENV)
const SPEED_AUTO = PBCav.apply_speed_auto!(KNOBS, KNOB_SOURCE)   # W7: PB_SPEED=auto (off, the default: nothing changes)
PBCav.pb_cpu_launch!(KNOBS["PB_CPU_LAUNCH"])     # W7: the CPU's workgroup shape (fixed, the default: the path before W7)
println("PBCav ", PBCav.PBCAV_VERSION, ", Julia ", VERSION, ", threads ", Threads.nthreads())
println(PBCav.knob_receipt(KNOBS, KNOB_SOURCE, ENV))
KNOBS["PB_SPEED"] === :auto && println("PB SPEED: auto, ", KNOBS["PB_BACKEND"], " ", KNOBS["PB_DOMAIN"], ": ", isempty(SPEED_AUTO) ? "nothing changed" : join(SPEED_AUTO, " "),
                                        haskey(PBCav.PB_SPEED_PROVISIONAL, (KNOBS["PB_BACKEND"], :air)) ? "; " * PBCav.PB_SPEED_PROVISIONAL[(KNOBS["PB_BACKEND"], :air)] : "")
KNOBS["PB_SPEED_DEV"] && println("PB SPEED DEV: PB_SPEED_DEV=1, the PROVISIONAL gate acknowledged: a development run, not a registered reading (W7, second review R1)")
# PB_BACKEND=cuda (W3): the GPU's owners checked BEFORE CUDA.jl is loaded (PBCav.gpu_gate: a julia owner, or a GPU process the
# system cannot identify, refuses; review of W3, F11); then CUDA.jl, its backend and a functional device
KNOBS["PB_BACKEND"] === :cuda && PBCav.gpu_gate("PB_BACKEND=cuda")
KNOBS["PB_BACKEND"] === :cuda && @eval using CUDA
const BACKEND = KNOBS["PB_BACKEND"] === :cpu ? CPU() :
    (CUDA.functional() || error("PB_BACKEND=cuda: CUDA.jl reports no functional GPU (CUDA.functional() is false)"); CUDA.CUDABackend())
KNOBS["PB_BACKEND"] === :cuda && PBCav.pb_graph_backend!(f -> CUDA.capture(f), CUDA.instantiate, CUDA.launch)   # W7 (graphs)
KNOBS["PB_BACKEND"] === :cuda && println("PB BACKEND: cuda, ", CUDA.name(CUDA.device()), "; CUDA.jl ", pkgversion(CUDA),
                                         ", runtime ", CUDA.runtime_version(), "; the mesh is built on the CPU and moved")

const CALL = PBCav.mesh_call(KNOBS, KNOB_SOURCE)
const REMAP = PBCav.remap_spec(KNOBS)      # W4: nothing unless PB_MESH_REMAP=common (free air only; src/pbcav/mesh_remap.jl)
const GENMESH = PBCav.generate_v1(CALL)
println(PBCav.mesh_receipt_line(GENMESH, PBCav.foil_path(CALL)))
const BLOCKSG = PBCav.cblocks(GENMESH; label="pbcav")
const BLOCKSR = REMAP === nothing ? BLOCKSG :
    let r = PBCav.remap_columns(BLOCKSG, REMAP); println(PBCav.remap_receipt_line(r)); r.blocks; end
const BLOCKS0 = PBCav.subdivide(BLOCKSR, KNOBS["PB_MESH_LEVEL"])
BLOCKS0.level > 0 && println(PBCav.mesh_level_line(BLOCKS0))
# free air (W2, plan section 32; PB_DOMAIN=freeair, opt-in): the outer faces re-tagged far field (src/pbcav/freeair.jl)
const FREEAIR = KNOBS["PB_DOMAIN"] === :freeair
const BLOCKS1 = FREEAIR ? PBCav.freeair_blocks(BLOCKS0) : BLOCKS0
# interior walls (24 Oct 2026; PB_BAFFLE, opt-in, src/pbcav/baffle.jl): the Gurney plate placed on the blocks (host work), its
# receipt PB BAFFLE; PB_BAFFLE empty (the default) changes nothing and prints nothing
const BLOCKS, BAFFLE_PLAN = PBCav.apply_baffle(BLOCKS1, KNOBS)
BAFFLE_PLAN === nothing || println(PBCav.baffle_receipt_line(BAFFLE_PLAN))
BAFFLE_PLAN === nothing || (l = PBCav.baffle_refine_receipt_line(BAFFLE_PLAN); isempty(l) || println(l))   # PB_BAFFLE_REFINE (30 Oct 2026): empty, the default, prints nothing
# the faces built on the CPU and moved (W3): on PB_BACKEND=cuda the metrics are the CPU's bit for bit, tunnel and free air alike
const MESH = PBCav.to_backend(BACKEND, PBCav.build_faces(BLOCKS; backend=CPU(), baffle=BAFFLE_PLAN))
println(PBCav.face_census_line(MESH))
FREEAIR && println(PBCav.mesh_audit_line(PBCav.mesh_audit(MESH)))

if KNOBS["PB_STOP"] === :mesh
    println("PB_STOP=mesh: the mesh is built; stopping before the state (exit 0)")
    exit(0)
end
if FREEAIR                     # the free-air run (src/pbcav/freeair.jl, freeair_drive!; laminar or SST by PB_TURBULENCE)
    let r = PBCav.freeair_drive!(MESH, KNOBS; backend=BACKEND), code = PBCav.freeair_exit_code(r.outcome)
        println("PB FREEAIR EXIT: ", code, " (outcome ", r.outcome, ")")
        exit(code)
    end
end

const STATE = PBCav.build_pbcav_state(MESH, KNOBS)
PBCav.pbcav_march!(STATE, MESH, KNOBS)
