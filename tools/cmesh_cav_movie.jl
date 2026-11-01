# ============================================================================================
# CAVITATING MOVIE ON THE HONEST CLOCK  (2026-08-02, plan v3 item S1+S2)
#
# Usage:
#   julia --project=src -O3 -t auto tools/cmesh_cav_movie.jl [sigma=1.25] [T_ctu=10] [nwarm=300000] [nframes=300]
#
# Env: every TOPO_* knob the explicit march takes, plus
#   CAV_MOVIE_TG=1     push the finished mp4s + summary panel to Telegram
#   CAV_MOVIE_WIN="x0,x1,y0,y1"   render window in chord units (default -0.15,1.35,-0.30,0.45)
#   CAV_MOVIE_PX=1100
#   CAV_MOVIE_OUT_DIR=path   claim a new, dedicated output directory (must not already exist)
#   TOPO_SPECIES_RECON=1   reconstruct interior conservative faces (requires TOPO_SPECIES=1)
#   CAV_MOVIE_RESTART=path   restore a conservative checkpoint into a new scored segment; skip warm-up
#
# ------------------------------------------------------------------------------------------------
# The global-dt flow march is coupled to TEM and SST. Time is accumulated from the
# actual timestep applied by each step and capped at requested frame times. A spatially
# uniform timestep alone does not prove the elapsed physical time or coupled temporal order.
# Topology and field movies complement scalar diagnostics; none alone establishes validation.
#
# WHAT IS RENDERED, and why each one:
#   alpha   the cavity itself -- extent, density, whether anything detaches
#   ux      SIGNED streamwise velocity on a diverging map centred at 0. The re-entrant jet is then a
#           COLOUR, not an inference from a scalar. This is the field the whole campaign is about.
#   Cp      pressure coefficient; the plateau at -sigma and the closure recovery
#   alpha(s,t)  the space-time map along a wall-following line;
#           upstream interface motion can be compared with reversed flow and topology.
#
# ★ AND IT CARRIES TOPOLOGY (plan v3 S2). `analyse_topology` from tools/topology_core.jl -- the SAME
# shared, unit-tested function the arbiter's Gate B uses -- runs per frame, so for the first time the
# honest clock reports component count, detached area and LE attachment.
# ============================================================================================

# ⛔ HALT GUARD (2026-08-08) -- refuses to start a solve while ./SOLVER_HALT exists. See tools/halt_guard.jl.
include(joinpath(dirname(@__DIR__), "tools", "halt_guard.jl"))

ENV["GKSwstype"] = "100"
const LBM_F = Float32; const F = Float32
const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "collision", "lattice.jl")); cx, cy, cz, w, _, _ = build_d3q39_static()
include(joinpath(ROOT, "src", "cmesh", "cmesh_gen.jl"));       using .CMeshGen
include(joinpath(ROOT, "tools", "cmesh_fixed_outer_mesh.jl")); using .CMeshFixedOuterMesh
include(joinpath(ROOT, "src", "cmesh", "cmesh_metrics.jl"));   using .CMeshMetrics
include(joinpath(ROOT, "src", "compressible", "cmesh_solver_ka.jl")); using .CMeshSolverKA
include(joinpath(ROOT, "src", "cavitation", "cmesh_cav_tem.jl"));   using .CMeshCavTEM
include(joinpath(ROOT, "src", "cavitation", "cmesh_cav_species.jl")); using .CMeshCavSpecies
include(joinpath(ROOT, "tools", "cmesh_species_diagnostics.jl"))
include(joinpath(ROOT, "tools", "topology_core.jl"))
include(joinpath(ROOT, "tools", "jet_connectivity_core.jl")); using .JetConnectivityCore
include(joinpath(ROOT, "tools", "realised_state.jl"))
include(joinpath(ROOT, "tools", "channel_diagnostics.jl"))
include(joinpath(ROOT, "tools", "panel_init.jl")); using .PanelInit
include(joinpath(ROOT, "tools", "cmesh_te_base.jl")); using .CMeshTEBase
include(joinpath(ROOT, "tools", "cmesh_mesh_quality_receipt.jl")); using .CMeshMeshQualityReceipt
include(joinpath(ROOT, "tools", "physical_clock.jl")); using .PhysicalTimeAccounting
include(joinpath(ROOT, "tools", "cmesh_raster.jl"))
using KernelAbstractions, Printf, Plots, SHA; gr()
import CUDA
# Makie field renders (density / Mach / Cp / vorticity / transported alpha / u_x) on the physical
# C-mesh, every MAKIE_EVERY frames, from the live state (standing directive 2026-07-18: every solver
# report carries Makie field renders, not only the Plots summaries). CAV_MOVIE_MAKIE=0 disables.
const MAKIE = get(ENV, "CAV_MOVIE_MAKIE", "1") == "1"
const MAKIE_EVERY = parse(Int, get(ENV, "CAV_MOVIE_MAKIE_EVERY", "10"))
MAKIE_EVERY >= 1 || error("CAV_MOVIE_MAKIE_EVERY must be >= 1")
if MAKIE
    include(joinpath(ROOT, "tools", "render", "cmesh_makie.jl")); using .CMeshMakie
end

const SIGMA  = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 1.25
const TCTU   = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 10.0
const NWARM  = length(ARGS) >= 3 ? parse(Int,     ARGS[3]) : 300000
const NFRAME = length(ARGS) >= 4 ? parse(Int,     ARGS[4]) : 300
isfinite(TCTU) && TCTU > 0 && NFRAME > 0 && NWARM >= 0 ||
    error("duration and frame count must be positive; warm-up count must be nonnegative")

const MESHV = parse.(Float64, split(get(ENV, "TOPO_MESH", "256,96,2.0e-4,1.20,80"), ","))
const MESH  = (Int(MESHV[1]), Int(MESHV[2]), MESHV[3])
# (rebuild, 22 Sep 2026) the wake column count was hard-coded at 64 and unreachable; NI = n_foil + 2 n_wake.
const NWAKE = length(MESHV) >= 6 ? Int(MESHV[6]) : 64
NWAKE >= 8 || error("TOPO_MESH sixth field n_wake must be >= 8 (got $NWAKE)")
const GROWTH = length(MESHV) >= 4 ? MESHV[4] : 1.3
const SMOOTH = length(MESHV) >= 5 ? Int(MESHV[5]) : 40
# Independent leading-edge surface spacing (speed-up plan W5, 30 Sep 2026): TOPO_MESH_LE_DS > 0 gives each side of the
# foil a two-sided stretching with this LE interval (chord units) and the default distribution's TE interval
# (cmesh_gen.jl, keyword le_ds). Default 0: the existing distribution, bitwise.
const MESH_LE_DS = parse(Float64, get(ENV, "TOPO_MESH_LE_DS", "0"))
isfinite(MESH_LE_DS) && 0 <= MESH_LE_DS < 0.1 || error("TOPO_MESH_LE_DS must lie in [0, 0.1) (got $MESH_LE_DS)")
# The app opts into one smooth geometry/curvature station rule for preview and
# Run. Unset controls retain the historical driver and archived meshes exactly.
const MESH_SURFACE_EVALUATOR = Symbol(get(ENV,"TOPO_MESH_SURFACE_EVALUATOR","legacy_catmull_rom"))
const MESH_SURFACE_DISTRIBUTION = Symbol(get(ENV,"TOPO_MESH_SURFACE_DISTRIBUTION","legacy"))
MESH_SURFACE_EVALUATOR in (:legacy_catmull_rom,:natural_c2) || error("invalid TOPO_MESH_SURFACE_EVALUATOR")
MESH_SURFACE_DISTRIBUTION in (:legacy,:curvature) || error("invalid TOPO_MESH_SURFACE_DISTRIBUTION")
MESH_SURFACE_DISTRIBUTION===:curvature && MESH_SURFACE_EVALUATOR!==:natural_c2 &&
    error("curvature surface distribution requires natural_c2 geometry")
# Experimental geometric control: explicitly provide all three settings or none.
# TOPO_MESH keeps the target wall spacing; the reference spacing only builds the
# unchanged outer mesh before a versioned inner remap. Default remains inactive.
const FIXED_OUTER_SETTINGS = fixed_outer_settings(ENV, MESH[3], MESH[2])
MESH_LE_DS > 0 && FIXED_OUTER_SETTINGS !== nothing && error("TOPO_MESH_LE_DS is not supported with the fixed-outer mesh control (its reference mesh keeps the default distribution)")
# Finite trailing-edge thickness (2 Oct 2026, memory/project_docs/2026-10-01_BLUNT_TRAILING_EDGE_PLAN.md; integration record
# memory/project_docs/_records/2026-10-02_blunt_te/). Every knob defaults off, which is the existing mesh, bitwise:
#   TOPO_MESH_TE_BASE    cells per half of the trailing-edge base (midpoint split, NIF = n_foil + 2 n_base); the section
#                        file must have an open edge (the generator refuses an open edge with 0, as before)
#   TOPO_MESH_TE_DS      the surface interval at the trailing-edge corners (sharp edge: at the trailing edge), chord units
#   TOPO_MESH_WAKE_NEAR  "ds,len[,growth]": the near wake's streamwise spacing held at or below ds over len (two segments)
# The base region is built by the generator's one blunt construction (CMeshGen.TE_BASE_CONSTRUCTION, named in the receipt).
# The settings they produce are the generator keywords, printed in the REALISED TRAILING-EDGE BASE receipt, stored in the
# checkpoint metadata (te_base) and bound by the wall instruments (tools/cmesh_wall_diagnostics.jl).
const MESH_TE_BASE = parse(Int, get(ENV, "TOPO_MESH_TE_BASE", "0"))
MESH_TE_BASE >= 0 || error("TOPO_MESH_TE_BASE must be >= 0 (got $MESH_TE_BASE)")
const MESH_TE_DS = parse(Float64, get(ENV, "TOPO_MESH_TE_DS", "0"))
isfinite(MESH_TE_DS) && 0 <= MESH_TE_DS < 0.1 || error("TOPO_MESH_TE_DS must lie in [0, 0.1) (got $MESH_TE_DS)")
const MESH_WAKE_NEAR = let raw = get(ENV, "TOPO_MESH_WAKE_NEAR", "")
    if isempty(raw)
        nothing
    else
        v = parse.(Float64, split(raw, ","))
        length(v) in (2, 3) && all(isfinite, v) && v[1] > 0 && v[2] > 0 && (length(v) == 2 || v[3] >= 1) ||
            error("TOPO_MESH_WAKE_NEAR must be ds,len[,growth] with ds > 0, len > 0, growth >= 1 (got $raw)")
        (wake_near_ds=v[1], wake_near_len=v[2], wake_near_growth=length(v) == 3 ? v[3] : 1.05)
    end
end
# Trailing-edge topology (5 Oct 2026, memory/project_docs/2026-10-05_STRIP_TOPOLOGY_PLAN.md; records memory/project_docs/_records/2026-10-05_strip_topology/):
#   TOPO_MESH_TE_TOPOLOGY  aplus (the default; also when unset: the midpoint-split blunt construction above, bitwise) or
#                          strip: the base's 2 n_base cells extruded downstream as a second solver block (the strip, an H
#                          block between the two wake legs; CMeshGen.TE_STRIP_CONSTRUCTION) and the sharp C-mesh wrapped
#                          around foil plus strip (NIF = n_foil). Needs TOPO_MESH_TE_BASE >= 2, and, as every
#                          trailing-edge knob, TOPO_SPECIES=1 and a channel. It refuses the steady mode, kinetic faces, the
#                          all-y+ wall function (either knob) and the Makie renders (plan section 3.3; below); the ledger runs
#                          over both blocks since version 1.5 (plan section 4.8, 7 Oct 2026) and the line-implicit step
#                          carries the wake line through the strip since stage 4 (plan section 5, 7 Oct 2026). It
#                          enters MESH_TE_SETTINGS as te_topology=:strip, so the receipt, the
#                          tag (_strip), the checkpoint metadata, the restart guard and the wall binder see it; with aplus
#                          nothing is merged and every A-plus tag, receipt and checkpoint is the one before the strip.
const MESH_TE_TOPOLOGY_RAW = get(ENV, "TOPO_MESH_TE_TOPOLOGY", "aplus")
MESH_TE_TOPOLOGY_RAW in ("aplus", "strip") ||
    error("TOPO_MESH_TE_TOPOLOGY must be aplus or strip (unset: aplus; got \"$MESH_TE_TOPOLOGY_RAW\")")
const MESH_TE_STRIP = MESH_TE_TOPOLOGY_RAW == "strip"
MESH_TE_STRIP && MESH_TE_BASE < 2 &&
    error("TOPO_MESH_TE_TOPOLOGY=strip requires TOPO_MESH_TE_BASE >= 2 (the strip is the base's 2 n_base cells extruded downstream; got $MESH_TE_BASE)")
# The foil range's base offset of the surface instruments: n_base on the A-plus mesh (its foil range holds the two
# half-bases), 0 with the strip (the foil range is the surface; the base is the strip block's column 1).
const BASE_IN_FOIL = MESH_TE_STRIP ? 0 : MESH_TE_BASE
# The generator keywords these knobs set, in a fixed order; nothing when every knob is at its default (the exact
# legacy call, receipt, tag and metadata).
const MESH_TE_SETTINGS = let nt = (; n_base=MESH_TE_BASE)
    MESH_TE_DS > 0 && (nt = merge(nt, (te_ds=MESH_TE_DS,)))
    MESH_WAKE_NEAR === nothing || (nt = merge(nt, MESH_WAKE_NEAR))
    MESH_TE_STRIP && (nt = merge(nt, (te_topology=:strip,)))
    MESH_TE_BASE == 0 && length(nt) == 1 ? nothing : nt
end
# (the sweep floor is the A-plus construction's; the strip's C block is the sharp construction, and strip meshes were built
# at 0 to 400 sweeps in the stage-1 range and stress sets, memory/project_docs/_records/2026-10-05_strip_topology/stage1/README.md)
MESH_TE_BASE > 0 && !MESH_TE_STRIP && SMOOTH < 80 &&
    error("TOPO_MESH_TE_BASE > 0 needs the fifth TOPO_MESH field (smoothing sweeps) at 80 or more (got $SMOOTH); the blunt construction was measured at 80-400")
MESH_TE_SETTINGS !== nothing && FIXED_OUTER_SETTINGS !== nothing &&
    error("TOPO_MESH_TE_BASE / TOPO_MESH_TE_DS / TOPO_MESH_WAKE_NEAR are not supported with the fixed-outer mesh control")
const FOIL   = get(ENV, "TOPO_FOIL", "tools/naca66mod312_a08.dat")
# Explicit rotated rectangular tunnel. Zero is the exact legacy C-shaped characteristic far field.
const CHANNEL_H    = parse(Float64, get(ENV, "TOPO_CHANNEL_H", "0"))
const CHANNEL_XMIN = parse(Float64, get(ENV, "TOPO_CHANNEL_XMIN", "-1.3333333333333333"))
const CHANNEL_XMAX = parse(Float64, get(ENV, "TOPO_CHANNEL_XMAX", "5.0"))
MESH_TE_SETTINGS !== nothing && CHANNEL_H <= 0 &&
    error("TOPO_MESH_TE_BASE / TOPO_MESH_TE_DS / TOPO_MESH_WAKE_NEAR require TOPO_CHANNEL_H > 0")
# Mesh quality V1 (15 Oct 2026, memory/project_docs/_records/2026-10-15_mesh_quality/implement/README.md): TOPO_MESH_QUALITY=v1 builds the mesh
# with the generator's CMeshQualityV1 option (Q-A the far-wake interval cap, Q-B the sharp wake cut's relaxation, Q-C the
# wall targets of the wake columns, Q-D the graded strip by TOPO_MESH_Q_STRIP only; then the G2 gates refuse a mesh that
# misses them). Unset: nothing is passed and every call, receipt, tag and checkpoint is the one before. Overrides
# TOPO_MESH_Q_PRESET/_WAKE/_RELAX/_OUTER/_STRIP (tools/cmesh_mesh_quality_receipt.jl) are refused without it. The option
# prints a MESH QUALITY RECEIPT after the coordinate receipt, names its tag CMeshQualityV1/<params> in the checkpoint metadata
# (mesh_quality, bound on restart) and adds _mq1h<8 hex of the tag> to the output tag. The version-1 release validation set
# stays on its registered meshes (the lead's decision of 15 Oct, review R1): nothing registered sets this knob.
const MESH_QUALITY = mesh_quality_env_option(ENV, CMeshGen; strip=MESH_TE_STRIP, n_base=MESH_TE_BASE, channel=CHANNEL_H > 0)
const MESH_QUALITY_TAG_STR = MESH_QUALITY === nothing ? nothing : CMeshGen.mesh_quality_tag(MESH_QUALITY)
MESH_QUALITY !== nothing && FIXED_OUTER_SETTINGS !== nothing &&
    error("TOPO_MESH_QUALITY=v1 is not supported with the fixed-outer mesh control")
# Opt-in static exit pressure: "ambient" uses the realised TEM ambient pressure;
# a number is an absolute pressure in solver units. Both warm-up and march require cav=4.
const OUTLET_PRESSURE_SET = haskey(ENV, "TOPO_OUTLET_PRESSURE")
const OUTLET_PRESSURE_RAW = get(ENV, "TOPO_OUTLET_PRESSURE", "")
OUTLET_PRESSURE_SET && isempty(OUTLET_PRESSURE_RAW) && error("TOPO_OUTLET_PRESSURE may not be empty")
const OUTLET_PRESSURE = OUTLET_PRESSURE_SET && OUTLET_PRESSURE_RAW != "ambient" ?
    parse(Float64, OUTLET_PRESSURE_RAW) : 0.0
OUTLET_PRESSURE_SET && CHANNEL_H <= 0 && error("TOPO_OUTLET_PRESSURE requires TOPO_CHANNEL_H > 0")
OUTLET_PRESSURE_SET && OUTLET_PRESSURE_RAW != "ambient" &&
    !(isfinite(OUTLET_PRESSURE) && OUTLET_PRESSURE > 0) && error("TOPO_OUTLET_PRESSURE must be ambient or a positive finite pressure")
# WATER DEFAULTS (23 Sep 2026, user directive: "The water stiffness and density mismatches have been lingering
# for months now. Fix them right now."). Both values are computed from the matched experiment -- Caltech HSWT
# 1142 at 39 ft/s = 11.887 m/s, water at the tunnel state (c = 1490 m/s, within 1%) -- by
# S0903_RUNS/water_calibration_probe_20260923.jl, which reads U from the code's own freestream setup:
#   TOPO_A2L = (c/U * U_lat)^2 / TEM_CS2 = (125.34 * 0.193218)^2 / (2/3) = 879.8   (c_liq/U 125.3; was 5.0)
#   TOPO_RATIO = 43197, water's own saturation density ratio (memory/project_docs/2026-09-21_DENSITY_RATIO_REACH_MEASUREMENT.md)
# The acoustic clock predicts about 21x more steps at full stiffness at the same mesh; the water ladder of
# 23 Sep measures it rather than trusting it. The old soft values remain available explicitly and every
# historical arm's ARM.txt records what it ran. The vapour-pressure scale eps_pv (0.1 against a physical 0.050
# at sigma 0.842) is the remaining member of this family and is NOT changed here: it sets the tension margin,
# and halving that deserves its own measured segment, not a rider on this commit.
const A2LF   = parse(Float64, get(ENV, "TOPO_A2L",   "879.8"))
const RATIOV = parse(Float64, get(ENV, "TOPO_RATIO", "43197.0"))
const ARECONV = parse(Int, get(ENV, "TOPO_ARECON", "0"))   # Tier 0: alpha face reconstruction
const CPRODV = parse(Float64, get(ENV, "TOPO_CPROD", "100.0"))
const CDESTV = parse(Float64, get(ENV, "TOPO_CDEST", "5.0"))
const MACH   = parse(Float64, get(ENV, "TOPO_MACH",  "0.2"))
const RPV    = parse(Float64, get(ENV, "TOPO_RP",       "0"))
const RP_S   = parse(Float64, get(ENV, "TOPO_RP_S",     "0"))
const RP_PG0 = parse(Float64, get(ENV, "TOPO_RP_PG0",   "0"))
const RP_KAP = parse(Float64, get(ENV, "TOPO_RP_KAPPA", "1.4"))
const TVPV   = parse(Float64, get(ENV, "TOPO_TVP",      "0"))
const SLIPV  = parse(Float64, get(ENV, "TOPO_SLIP",     "0"))
const SLIPTAU= parse(Float64, get(ENV, "TOPO_SLIPTAU",  "0"))
# ★★★ COLLECTIVE / CLOUD BUBBLE DYNAMICS (d'Agostino-Brennen). The ONLY absent physics item whose
# timescale (0.026-0.26 CTU) is near the flow's, and the only lag term that varies IN SPACE -- dense
# cores slow, dilute edges do not. The uniform Cprod ladder could not produce that and did not test it.
const CLOUDV = parse(Float64, get(ENV, "TOPO_CLOUD",    "0"))
# S3c/S3d/S3e. ⚠ TOPO_POLY needs a spectrum reaching ~100 um: the 1 um Schnerr-Sauer standard
# nucleus is SUB-CRITICAL here by 9.3x (R_crit = 9.28 um at our suction peak), so a
# monodisperse 1 um population cannot be activated at all.
const POLYV    = parse(Float64, get(ENV, "TOPO_POLY",    "0"))
const HKV      = parse(Float64, get(ENV, "TOPO_HK",      "0"))
const WALLNUCV = parse(Float64, get(ENV, "TOPO_WALLNUC", "0"))
const BTKEV    = parse(Float64, get(ENV, "TOPO_BTKE",    "0"))   # S3f bubble-induced TKE into the k eq
const PVTURBV  = parse(Float64, get(ENV, "TOPO_PVTURB",  "0"))
const PVVORTV   = parse(Float64, get(ENV, "TOPO_PVVORT",  "0.0"))   # S3k: Le & Tran C_v (0 = bitwise off)   # S3k Singhal turbulent vapour threshold
const TFREEV   = parse(Float64, get(ENV, "TOPO_TFREE",   "0"))   # S3g step 1: keep the MARCHED T
# ★★★ THE PEDESTAL FLOOR. Tmin has been 1.2 for the whole campaign and measure_pedestal_rail.jl found
# T railed AT it in 5 of 5 sampled states -- so T is not a temperature, it is a constant, and Tmin is
# a clamp on a PRESSURE. tfree alone could not fix this: it stops _k_tem_commit! OVERWRITING T, but
# the clamp below still catches the marched value, which is why latent heat measured EXACTLY inert
# even with tfree=1. Unrailing is a SECOND, separate step. Default 1.2 = bitwise legacy.
const TMINV    = parse(Float64, get(ENV, "TOPO_TMIN",    "1.2"))
const EMIXV    = parse(Float64, get(ENV, "TOPO_EMIX",    "0"))   # S3g step 2: mixture cv + latent heat
const KEPSV    = parse(Float64, get(ENV, "TOPO_KEPS",    "0"))   # S3j k-eps RNG (Frikha closure)
const WALLFUNV = parse(Float64, get(ENV, "TOPO_WALLFUN", "0"))  # S3i log-law wall function (needs y+ 30-100)
const WFPGV    = parse(Float64, get(ENV, "TOPO_WF_PG",   "0"))   # Larsson pressure-gradient correction
const CVVV     = parse(Float64, get(ENV, "TOPO_CVV",     "-1"))  # vapour cv; <0 => same as liquid
const LATENTV  = parse(Float64, get(ENV, "TOPO_LATENT",  "0"))   # latent heat of vaporisation
const EPSPV    = parse(Float64, get(ENV, "TOPO_EPSPV",   "0.1"))  # vapour-pressure scale p_v = eps_pv*sigma*q; 0.1 legacy, 0.0394 physical at sigma 0.842
const LQV      = parse(Float64, get(ENV, "TOPO_LQ",      "0"))   # TIER-3 dissolved-gas liquid compliance (volume fraction at local p); wedge remedy 2026-09-24
# The lq the TEM is built with (10 Oct 2026): TOPO_LQ, or the closed-form conversion of TOPO_SPECIES_FREE_GAS (set below, after the
# carrier and an lq = 0 TEM exist; the two knobs are exclusive).
const LQ_EFFECTIVE = Ref(LQV)
# ★ ANGLE OF ATTACK IS NOW A KNOB (2026-08-07). It was hardcoded at 8.0 in all three tools, so
# EVERY cavitating run of this campaign -- all twelve 'frozen' configurations and every battery --
# was at one angle. alpha is the only macroscopic parameter never varied here.
# It is applied as FREESTREAM incidence (set_freestream_ka!), not by rotating the mesh, and the foil
# file is at zero incidence (the 'a08' in its name is the NACA a=0.8 MEAN LINE, not 8 degrees), so
# this is a clean one-parameter change.
# ⚠ OFF-BENCHMARK ABOVE 8 deg: Leroux and Frikha are both at 8 deg and Le's pattern map stops there,
#   so alpha != 8 is EXPLORATORY, not validation. Label it so.
# A same-incidence single-phase control is needed to separate cavity dynamics from base-flow
# variation. Archived five-CTU controls drifted and did not establish a steady base flow.
const ALPHA = parse(Float64, get(ENV, "TOPO_ALPHA", "8.0"))
# TOPO_RE (2026-09-14): chord Reynolds number of the march. The default string parses to exactly the archived
# constant 0.8e6, so every existing recipe, fixture and receipt is unchanged; the laminar validation cases of the
# ladder (tier 1b, NACA0012 at Re 1000) set it explicitly.
const RE = parse(Float64, get(ENV, "TOPO_RE", "8.0e5"))
isfinite(RE) && RE > 0 || error("TOPO_RE must be a positive finite Reynolds number (got $RE)")
# TOPO_INIT (2026-09-14, validation ladder tier 2b): "uniform" (default, every existing recipe bitwise unchanged) starts
# from the uniform freestream around the blocking foil, a water-hammer start in a compressible liquid; "panel" starts from
# the confined potential flow of tools/panel_init.jl (velocity and Bernoulli density about the ambient), so the impulsive
# no-slip start makes vorticity but no compression wave. Requires a channel, TOPO_SPECIES=1 and no restart.
const INIT_RAW = get(ENV, "TOPO_INIT", "uniform")
INIT_RAW in ("uniform", "panel") || error("TOPO_INIT must be uniform or panel (got $INIT_RAW)")
const INIT_PANEL = INIT_RAW == "panel"
# TOPO_CFL (2026-09-12): explicit march CFL on the operational clock. Default 0.15 is the
# archived production value (bitwise unchanged). The pre-registered margin study
# memory/project_docs/2026-09-12_SPECIES_CFL_MARGIN_STUDY.md measured 0.30 and 0.45 within 1e-5 of 0.15 on the
# production checkpoint; the species step itself admits (0, 0.5].
const CFLV = parse(Float64, get(ENV, "TOPO_CFL", "0.30"))  # (rebuild, 22 Sep 2026) 0.45 was adopted on the production
# mesh (x0.77 wall) and WITHDRAWN the same day by arm F768: on the 768 mesh with a growing cavity it rejects 38% of
# attempts in a frame and is 1.46x SLOWER than 0.30, which rejects none from the same state. Paired test:
# rebuild_bench_20260922/stormA_cfl045 vs stormA_cfl030, docs/2026-09-22_REBUILD_ARM_F768_AMENDMENT.md.
# Operational CFL ceiling (speed-up plan W1a, 30 Sep 2026): TOPO_SPECIES_CFL_MAX, default 0.5 (bitwise the previous
# refusal); at most 1.0. A value above 0.5 is admitted for a configuration only by a registered measurement.
const CFL_MAX_V = parse(Float64, get(ENV, "TOPO_SPECIES_CFL_MAX", "0.5"))
isfinite(CFL_MAX_V) && 0.5 <= CFL_MAX_V <= 1.0 || error("TOPO_SPECIES_CFL_MAX must be in [0.5, 1.0] (got $CFL_MAX_V)")
isfinite(CFLV) && 0.0 < CFLV <= CFL_MAX_V || error("TOPO_CFL must be in (0, $CFL_MAX_V] (got $CFLV)")
# ★★ CARRIER FLUX LEVEL (2026-09-03). KSol.cav: 1 = PDP pexc split, 2 = +Layer C (the archived
# B29-B32 production value), 3 = +negcomp floor, 4 = LOW-MACH MIXTURE FLUX (AUSM+-up on rho, u,
# p_mix, a_m; replaces the KFVS carrier flux, the centred pexc force and Layer C at every face).
# Measured on CPU with the production functions: the KFVS carrier pumps liquid mass across a
# stationary liquid/vapour contact at 1.63x the freestream convective flux, identically for minmod
# and WENO5-Z; the mixture flux holds the same contact exactly. Default 2 = bitwise legacy.
const CAVLV = parse(Int, get(ENV, "TOPO_CAV", "2"))
CAVLV in (1, 2, 3, 4) || error("TOPO_CAV must be 1, 2, 3 or 4 (got $CAVLV)")
# TOPO_CAV_WARM: the level used for the LTS WARM-UP (default = TOPO_CAV). Warming up on the legacy
# level 2 and switching to 4 for the scored march gives the mixture flux the identical developed
# state as the archived control, so the first frames are a clean flux A/B on one initial condition.
const CAVWARM = parse(Int, get(ENV, "TOPO_CAV_WARM", string(CAVLV)))
CAVWARM in (1, 2, 3, 4) || error("TOPO_CAV_WARM must be 1, 2, 3 or 4 (got $CAVWARM)")
OUTLET_PRESSURE_SET && !(CAVLV == 4 && CAVWARM == 4) &&
    error("TOPO_OUTLET_PRESSURE requires TOPO_CAV=4 and TOPO_CAV_WARM=4")
# ★★ ALPHA CLOCK IN THE SCORED MARCH (2026-09-03). tem.omega is the under-relaxation of the
# segregated flow<->alpha Picard loop, needed in the LTS warm-up. It was ALSO left at 0.2 during the
# explicit global-dt march, where _k_tem_commit! keeps only that fraction of the alpha
# increment. omega=1 applies the full TEM step. This repairs the relaxed alpha clock;
# stability and the order of the segregated flow/TEM/SST update still need timestep studies.
# Default 1.0; use 0.2 only when deliberately reproducing the historical relaxed-alpha path.
const MARCH_OMEGA = parse(Float64, get(ENV, "TOPO_MARCH_OMEGA", "1.0"))
0.0 < MARCH_OMEGA <= 1.0 || error("TOPO_MARCH_OMEGA must lie in (0, 1] (got $MARCH_OMEGA)")
# Optional conservative increment-form RK blending for the physical march only.
# Keep the archived LTS warm-up and default numerical path unchanged.
const MARCH_RK_INCREMENTAL_RAW = get(ENV, "TOPO_RK_INCREMENTAL", "0")
MARCH_RK_INCREMENTAL_RAW in ("0", "1") || error("TOPO_RK_INCREMENTAL must be 0 or 1")
const MARCH_RK_INCREMENTAL = MARCH_RK_INCREMENTAL_RAW == "1"
# Opt-in transport within this driver; use the same physical stages for warm-up.
const SPECIES_RAW = get(ENV, "TOPO_SPECIES", "0")
SPECIES_RAW in ("0","1") || error("TOPO_SPECIES must be 0 or 1")
const CONSERVATIVE = SPECIES_RAW == "1"
FIXED_OUTER_SETTINGS !== nothing && !(CONSERVATIVE && CHANNEL_H > 0) &&
    error("fixed-outer mesh control requires TOPO_SPECIES=1 and a channel")
const SPECIES_RECON_RAW = get(ENV,"TOPO_SPECIES_RECON","0")
SPECIES_RECON_RAW in ("0","1") || error("TOPO_SPECIES_RECON must be 0 or 1")
const SPECIES_RECON = SPECIES_RECON_RAW == "1"
SPECIES_RECON && !CONSERVATIVE && error("TOPO_SPECIES_RECON requires TOPO_SPECIES=1")
# Conservative face family (2026-09-10): "ausm" keeps the existing AUSM+-up split; "kfvs"
# selects the common-gauge kinetic face with half-stream vapor transport and kinetic clock.
const SPECIES_FLUX = get(ENV,"TOPO_SPECIES_FLUX","ausm")
SPECIES_FLUX in ("ausm","kfvs") || error("TOPO_SPECIES_FLUX must be ausm or kfvs (got $SPECIES_FLUX)")
const SPECIES_KFVS = SPECIES_FLUX == "kfvs"
SPECIES_KFVS && !CONSERVATIVE && error("TOPO_SPECIES_FLUX=kfvs requires TOPO_SPECIES=1")
# Admission mode (2026-09-12): "immediate" reads every status/clock on the host as before;
# "deferred" reads flags and accounting once per attempt and classifies numerical
# rejections directly; uncertain failures retain immediate diagnostic replay.
const SPECIES_ADMISSION_RAW = get(ENV,"TOPO_SPECIES_ADMISSION","immediate")
SPECIES_ADMISSION_RAW in ("immediate","deferred") || error("TOPO_SPECIES_ADMISSION must be immediate or deferred (got $SPECIES_ADMISSION_RAW)")
const SPECIES_ADMISSION = Symbol(SPECIES_ADMISSION_RAW)
SPECIES_ADMISSION==:deferred && !CONSERVATIVE && error("TOPO_SPECIES_ADMISSION=deferred requires TOPO_SPECIES=1")
const SPECIES_GRAPH_RAW = get(ENV,"TOPO_SPECIES_GRAPH","0")
SPECIES_GRAPH_RAW in ("0","1") || error("TOPO_SPECIES_GRAPH must be 0 or 1")
const SPECIES_GRAPH = SPECIES_GRAPH_RAW=="1"
SPECIES_GRAPH && !(CONSERVATIVE && SPECIES_ADMISSION==:deferred) && error("TOPO_SPECIES_GRAPH requires conservative deferred admission")
# Opt-in execution choice; default baseline unchanged and physical configuration preserved.
const SPECIES_REDUCTION_TILE = CMeshCavSpecies._species_reduction_setting(ENV,CONSERVATIVE,SPECIES_ADMISSION)
const SPECIES_REPLAY = get(ENV,"TOPO_SPECIES_REPLAY","fallback")
SPECIES_REPLAY in ("fallback","always") || error("TOPO_SPECIES_REPLAY must be fallback or always")
CMeshCavSpecies.REPLAY_REJECTIONS[]=SPECIES_REPLAY=="always"
if SPECIES_GRAPH
    include(joinpath(ROOT,"src/cmesh/cmesh_species_cuda_graph.jl"))
end
# Conservation ledger (rebuild, 22 Sep 2026): off by default. It is a diagnostic of a structurally
# conserved march; deferred admission packs it into the existing host read.
const SPECIES_LEDGER = get(ENV,"TOPO_SPECIES_LEDGER","0") == "1"
CMeshCavSpecies.LEDGER_ENABLED[] = SPECIES_LEDGER
const SPECIES_CLOCK_MARGIN = parse(Float32, get(ENV,"TOPO_SPECIES_CLOCK_MARGIN","0.9"))
0.5f0 <= SPECIES_CLOCK_MARGIN <= 1.0f0 || error("TOPO_SPECIES_CLOCK_MARGIN must be in [0.5, 1.0] (got $SPECIES_CLOCK_MARGIN)")
CMeshCavSpecies.CLOCK_MARGIN[] = SPECIES_CLOCK_MARGIN
CONSERVATIVE && println("  SPECIES CLOCK MARGIN: ", SPECIES_CLOCK_MARGIN, " x the reconstructed face cap (TOPO_SPECIES_CLOCK_MARGIN; 0.9 was the unmeasured hard-coded value)")
CMeshCavSpecies.CFL_MAX[] = Float32(CFL_MAX_V)
CONSERVATIVE && CFL_MAX_V != 0.5 && println("  SPECIES CFL CEILING: ", CFL_MAX_V, " (TOPO_SPECIES_CFL_MAX; 0.5 is the default and the previous hard refusal)")
CONSERVATIVE && println("  SPECIES LEDGER: ", SPECIES_LEDGER ? "on (per-step boundary/source accounting; TOPO_SPECIES_LEDGER=1)" :
                                                              "off (inventory mass_error/vapor_error columns are NaN; TOPO_SPECIES_LEDGER=0)")
# Steady pseudo-time mode (speed-up plan W2, 30 Sep 2026): TOPO_SPECIES_STEADY=1 replaces the physical-time march by
# local-time-step iterations toward the steady state of a WETTED target (tools/cmesh_cav_steady_loop.jl, with its own
# TOPO_STEADY_* knobs; src/cmesh/cmesh_species_steady.jl). Default 0: this driver is unchanged. It is not physical
# time, and it is refused with the ledger, kinetic faces, graph execution, a warm-up or a nonzero phase-change source.
const STEADY_RAW = get(ENV,"TOPO_SPECIES_STEADY","0")
STEADY_RAW in ("0","1") || error("TOPO_SPECIES_STEADY must be 0 or 1 (got $STEADY_RAW)")
const STEADY = STEADY_RAW=="1"
STEADY && !CONSERVATIVE && error("TOPO_SPECIES_STEADY=1 requires TOPO_SPECIES=1")
STEADY && SPECIES_LEDGER && error("TOPO_SPECIES_STEADY=1 requires TOPO_SPECIES_LEDGER=0: the ledger's accounting weights one physical dt")
STEADY && SPECIES_KFVS && error("TOPO_SPECIES_STEADY=1 supports the AUSM+-up faces only")
STEADY && SPECIES_GRAPH && error("TOPO_SPECIES_STEADY=1 requires TOPO_SPECIES_GRAPH=0: the graph captures the physical-time attempt")
STEADY && NWARM != 0 && error("TOPO_SPECIES_STEADY=1 requires a zero warm-up (third argument): the warm-up is a physical-time march")
STEADY && println("  STEADY PSEUDO-TIME MODE: local time steps toward a steady wetted state (TOPO_SPECIES_STEADY=1); ",
                  "the CTU and frame arguments are not used as physical times")
# Line-implicit physical-time step (speed-up plan W3a, 30 Sep 2026): TOPO_SPECIES_IMEX=1 marches physical time with
# each SSPRK3 stage linearly implicit along the wall-normal j lines (src/cmesh/cmesh_species_imex.jl); the global dt
# comes from the i-direction clocks. Default 0: this driver is unchanged. An opt-in integrator whose accuracy against
# the explicit step is a registered bridge; refused with the ledger, kinetic faces, graph execution, immediate
# admission, a warm-up and the steady mode.
const IMEX_RAW = get(ENV,"TOPO_SPECIES_IMEX","0")
IMEX_RAW in ("0","1") || error("TOPO_SPECIES_IMEX must be 0 or 1 (got $IMEX_RAW)")
const IMEX = IMEX_RAW=="1"
IMEX && !CONSERVATIVE && error("TOPO_SPECIES_IMEX=1 requires TOPO_SPECIES=1")
IMEX && STEADY && error("TOPO_SPECIES_IMEX=1 and TOPO_SPECIES_STEADY=1 are exclusive")
IMEX && SPECIES_LEDGER && error("TOPO_SPECIES_IMEX=1 requires TOPO_SPECIES_LEDGER=0: the ledger's accounting assumes explicit stage fluxes")
IMEX && SPECIES_KFVS && error("TOPO_SPECIES_IMEX=1 supports the AUSM+-up faces only")
IMEX && SPECIES_GRAPH && error("TOPO_SPECIES_IMEX=1 requires TOPO_SPECIES_GRAPH=0: the graph captures the explicit attempt")
IMEX && SPECIES_ADMISSION != :deferred && error("TOPO_SPECIES_IMEX=1 requires TOPO_SPECIES_ADMISSION=deferred")
IMEX && NWARM != 0 && error("TOPO_SPECIES_IMEX=1 requires a zero warm-up (third argument)")
IMEX && println("  LINE-IMPLICIT PHYSICAL-TIME STEP (TOPO_SPECIES_IMEX=1): SSPRK3 stages linearly implicit along the j lines, ",
                "global dt from the i-direction clocks (not the default integrator)")
# W3a second stage: TOPO_SPECIES_IMEX_SOURCE=implicit linearises the phase-change source into the line systems and
# removes its clock from the global dt; explicit (default) keeps the source explicit with its own clock.
const IMEX_SOURCE_RAW = get(ENV,"TOPO_SPECIES_IMEX_SOURCE","explicit")
IMEX_SOURCE_RAW in ("explicit","implicit") || error("TOPO_SPECIES_IMEX_SOURCE must be explicit or implicit (got $IMEX_SOURCE_RAW)")
!IMEX && IMEX_SOURCE_RAW=="implicit" && error("TOPO_SPECIES_IMEX_SOURCE=implicit requires TOPO_SPECIES_IMEX=1")
IMEX && println("  LINE-IMPLICIT PHASE SOURCE: ", IMEX_SOURCE_RAW,
                IMEX_SOURCE_RAW=="implicit" ? " (its Jacobian in the q row of the line blocks; no source clock in dt)" : " (its own explicit clock)")
# SCOREABILITY WARNINGS (23 Sep 2026). Three rebuild commits each loosened or removed something a reading
# instrument still requires, and nothing in the driver said so: the ledger became opt-in (mass_error goes NaN),
# the per-cell phase-volume throw was loosened to 1e-2, and the single-precision EOS became the default
# (interior closure 5e-8 -> 1.4e-6, measured). Every reader of this campaign still demands
# mass_error/mass <= 1e-9 AND interior closure <= 64*eps(Float32). An arm can therefore march to completion and
# be refused at reading time; arm F768b spent 9.95 hours doing exactly that. These warnings change no number,
# stop no run, and are the cheapest thing that converts a silent loss into a line someone can act on.
const SCORED_MASS_RESIDUAL_LIMIT = 1e-9              # the readers' MASS_RESIDUAL_LIMIT, verbatim
const SCORED_CLOSURE_LIMIT = 64 * eps(Float32)       # the readers' CLOSURE_LIMIT, verbatim
const SCORED_CLOSURE_WARNED = Ref(false)
CONSERVATIVE && !SPECIES_LEDGER && @warn string(
    "UNSCOREABLE ARM: TOPO_SPECIES_LEDGER=0 writes NaN into the inventory mass_error column, and every reading ",
    "instrument requires maximum(abs.(mass_error)./mass) <= ", SCORED_MASS_RESIDUAL_LIMIT, ", which NaN fails. ",
    "This run will read INCOMPLETE whatever its physics. Set TOPO_SPECIES_LEDGER=1 for any arm to be scored; ",
    "measured cost about 4% of the step.")
# EOS precision (2026-09-12, speed-first on FP32 GPUs): "double" keeps the September Float64 EOS
# arithmetic; "single" evaluates the EOS in Float32 on the Float64 state (tag _eos32, bound in checkpoints).
const SPECIES_EOS_RAW = get(ENV,"TOPO_SPECIES_EOS","single") # rebuild default (22 Sep 2026); "double" keeps the Float64 EOS
SPECIES_EOS_RAW in ("double","single") || error("TOPO_SPECIES_EOS must be double or single (got $SPECIES_EOS_RAW)")
const SPECIES_EOS = Symbol(SPECIES_EOS_RAW)
SPECIES_EOS==:single && !CONSERVATIVE && error("TOPO_SPECIES_EOS=single requires TOPO_SPECIES=1")
# Phase-change source weighting (2026-09-19): alpha = the stored reference composition (default,
# bitwise the September path); beta = the represented vapour volume fraction. Bound in the species
# configuration fingerprint as :CMeshSpeciesBetaWeightedSourceV1, so a restart cannot mix them.
const SPECIES_SOURCE_WEIGHT_RAW = get(ENV,"TOPO_SPECIES_SOURCE_WEIGHT","alpha")
SPECIES_SOURCE_WEIGHT_RAW in ("alpha","beta") || error("TOPO_SPECIES_SOURCE_WEIGHT must be alpha or beta (got $SPECIES_SOURCE_WEIGHT_RAW)")
const SPECIES_SOURCE_WEIGHT = Symbol(SPECIES_SOURCE_WEIGHT_RAW)
SPECIES_SOURCE_WEIGHT==:beta && !CONSERVATIVE && error("TOPO_SPECIES_SOURCE_WEIGHT=beta requires TOPO_SPECIES=1")
# Reboud correction of the eddy viscosity (2026-09-20; on by default since 10 Oct 2026, the void-fraction form since the review of
# 13 Oct, memory/project_docs/2026-10-10_REBOUD_DEFAULT_AND_SCHNERR_SAUER_DESIGN.md sections 1 and 6): TOPO_SPECIES_NUT_DENSITY = none (no
# correction, the September path: it reproduces every archived arm), reboud_v3 (the void-fraction form V3: Reboud's function of the
# cell's represented vapour volume fraction at its own pressure, bound as CMeshSpeciesReboudNutV3/n=<n>), reboud_v2 (the
# composition form of 10 Oct, CMeshSpeciesReboudNutV2/n=<n>) or reboud_v1 (the density form of 20 Sep, arm T25's,
# CMeshSpeciesReboudNutV1/n=<n>), exponent TOPO_REBOUD_N (default 10); TOPO_SPECIES_REBOUD_PRODUCTION=1 (opt-in, 13 Oct) forms the
# SST k production from the corrected nu_t (CMeshSpeciesReboudProductionV1; default 0: from the uncorrected nu_t, the 20 Sep
# convention). A species run that leaves TOPO_SPECIES_NUT_DENSITY unset is REFUSED unless it declares
# TOPO_SPECIES_DEFAULTS=2026-10-10: every launcher written before 10 Oct relies on the old default by omission. The rules are pure
# functions, unit-tested by tools/test_species_model_knobs.jl.
const SPECIES_NUT_SETTING = try
    CMeshCavSpecies._species_nut_density_setting(ENV, CONSERVATIVE)
catch err
    err isa ArgumentError ? error(err.msg) : rethrow()
end
const SPECIES_NUT_DENSITY = SPECIES_NUT_SETTING.density
const SPECIES_REBOUD_N = SPECIES_NUT_SETTING.reboud_n
const SPECIES_REBOUD_PRODUCTION = try
    CMeshCavSpecies._species_reboud_production_setting(ENV, CONSERVATIVE, SPECIES_NUT_DENSITY)
catch err
    err isa ArgumentError ? error(err.msg) : rethrow()
end
const SPECIES_REBOUD_PRODUCTION_TEXT = SPECIES_REBOUD_PRODUCTION ?
    "; SST k production from the corrected nu_t (TOPO_SPECIES_REBOUD_PRODUCTION=1; tag CMeshSpeciesReboudProductionV1)" :
    "; SST k production from the uncorrected nu_t (the 20 Sep convention; TOPO_SPECIES_REBOUD_PRODUCTION=1 forms it from the corrected one)"
CONSERVATIVE && println("  SPECIES NUT DENSITY: ", SPECIES_NUT_DENSITY,
    SPECIES_NUT_DENSITY==:reboud_v3 ? "  (void-fraction form V3, n=$(SPECIES_REBOUD_N); tag CMeshSpeciesReboudNutV3; set by " *
                                   (SPECIES_NUT_SETTING.set_by===:epoch ? "the 2026-10-10 defaults epoch, TOPO_SPECIES_DEFAULTS" : "the knob TOPO_SPECIES_NUT_DENSITY") *
                                   SPECIES_REBOUD_PRODUCTION_TEXT * ")" :
    SPECIES_NUT_DENSITY==:reboud_v2 ? "  (composition form V2 of 10 Oct, n=$(SPECIES_REBOUD_N); tag CMeshSpeciesReboudNutV2" * SPECIES_REBOUD_PRODUCTION_TEXT * ")" :
    SPECIES_NUT_DENSITY==:reboud_v1 ? "  (density form of 20 Sep, n=$(SPECIES_REBOUD_N); tag CMeshSpeciesReboudNutV1" * SPECIES_REBOUD_PRODUCTION_TEXT * ")" :
                                      "  (no correction; September path)")
# Phase-change source model (10 Oct 2026, design note section 2): TOPO_SPECIES_SOURCE_MODEL = merkle (the default: the
# TOPO_CPROD/TOPO_CDEST rates, bitwise) or schnerr_sauer (the OpenFOAM v11 form with its nucleus seed: TOPO_SPECIES_SS_N0 [m^-3,
# default 1.6e13], TOPO_SPECIES_SS_DNUC [m, default 2e-6] and TOPO_SPECIES_SS_CHORD [m, required: the physical chord], converted to
# chord units; refused with TOPO_SPECIES=0, TOPO_SPECIES_SOURCE_WEIGHT=beta, TOPO_SPECIES_IMEX=1 and an explicit TOPO_CPROD or
# TOPO_CDEST). Bound in the species fingerprint as CMeshSpeciesSchnerrSauerV2/n0=<N0>/d=<D> (V2, 13 Oct 2026: the trace level).
const SPECIES_SOURCE_SETTING = try
    CMeshCavSpecies._species_source_model_setting(ENV, CONSERVATIVE)
catch err
    err isa ArgumentError ? error(err.msg) : rethrow()
end
const SPECIES_SOURCE_MODEL = SPECIES_SOURCE_SETTING.model
const SPECIES_SS = SPECIES_SOURCE_MODEL===:schnerr_sauer
# Free gas carried by the liquid (10 Oct 2026, design note section 3): TOPO_SPECIES_FREE_GAS = the free-gas volume fraction of the
# incoming liquid at the free-stream static pressure (default 0: off, bitwise), converted to the TEM's lq in closed form below;
# exclusive with a nonzero TOPO_LQ; [0, 0.05). Bound in the species fingerprint as CMeshSpeciesFreeGasV1/lq=<lq>.
const SPECIES_FREE_GAS = try
    CMeshCavSpecies._species_free_gas_setting(ENV, CONSERVATIVE)
catch err
    err isa ArgumentError ? error(err.msg) : rethrow()
end
# All-y+ wall function on the species path (26 Sep 2026): none (default: the wall-resolved SST wall, bitwise the
# September path) or reichardt (s.wallfun = 2: Reichardt wall shear at every y+, SST omega pinned to Menter's blend
# in the wall cell, zero k wall flux; memory/project_docs/2026-09-26_WALL_FUNCTION_SPECIES_PATH.md). Meant for a mesh whose first
# cell sits in the log layer (y+ ~30-60, set through TOPO_MESH's dn1); the per-frame y+ line reports it.
const SPECIES_WALL_FUNCTION_RAW = get(ENV,"TOPO_SPECIES_WALL_FUNCTION","none")
SPECIES_WALL_FUNCTION_RAW in ("none","reichardt") || error("TOPO_SPECIES_WALL_FUNCTION must be none or reichardt (got $SPECIES_WALL_FUNCTION_RAW)")
const SPECIES_WALL_FUNCTION = Symbol(SPECIES_WALL_FUNCTION_RAW)
SPECIES_WALL_FUNCTION==:reichardt && !CONSERVATIVE && error("TOPO_SPECIES_WALL_FUNCTION=reichardt requires TOPO_SPECIES=1")
SPECIES_WALL_FUNCTION==:reichardt && WALLFUNV != 0.0 && error("TOPO_SPECIES_WALL_FUNCTION=reichardt excludes the legacy TOPO_WALLFUN")
CONSERVATIVE && println("  SPECIES WALL FUNCTION: ", SPECIES_WALL_FUNCTION, SPECIES_WALL_FUNCTION==:reichardt ?
    "  (all-y+ Reichardt wall shear; SST omega pinned to Menter's blend in the wall cell; zero k wall flux; s.wallfun=2; tag CMeshSpeciesWallFunctionReichardtV1)" :
    "  (wall-resolved SST wall; September path)")
CONSERVATIVE && println("  SPECIES SOURCE WEIGHT: ", SPECIES_SOURCE_WEIGHT, SPECIES_SOURCE_WEIGHT==:beta ? "  (represented vapour volume fraction; tag CMeshSpeciesBetaWeightedSourceV1)" : "  (reference composition; September path)")
const SST_RAW = get(ENV, "TOPO_SST", "1")
SST_RAW in ("0","1") || error("TOPO_SST must be 0 or 1")
const SST_ENABLED = SST_RAW == "1"
# Prescribed boundary-layer trip on the species path (30 Sep 2026): 0 (default: off, bitwise the September path) or
# the wall station x/c from which the SST k is raised to an inner-layer turbulent level over a strip of
# TOPO_SPECIES_TRIP_WIDTH (default 0.005). The Caltech foil was tripped by glass spheres over its first 1.5% of chord;
# bound in the species fingerprint as CMeshSpeciesTripV1 (memory/project_docs/2026-09-30_LEADING_EDGE_FIX.md).
const SPECIES_TRIP_X = parse(Float64, get(ENV, "TOPO_SPECIES_TRIP", "0"))
const SPECIES_TRIP_WIDTH = parse(Float64, get(ENV, "TOPO_SPECIES_TRIP_WIDTH", "0.005"))
SPECIES_TRIP_X != 0 && !CONSERVATIVE && error("TOPO_SPECIES_TRIP requires TOPO_SPECIES=1")
SPECIES_TRIP_X != 0 && !SST_ENABLED && error("TOPO_SPECIES_TRIP requires TOPO_SST=1")
SPECIES_TRIP_X != 0 && SPECIES_WALL_FUNCTION==:reichardt && error("TOPO_SPECIES_TRIP is not defined with the wall function")
# Limited SST non-orthogonal diffusion correction (2 Oct 2026, opt-in; blunt trailing edge,
# memory/project_docs/_records/2026-10-02_blunt_te/sst_clock/README.md): lambda in [0, 1] bounds each face's non-orthogonal correction of
# the k and omega diffusion by lambda times its two-point flux, and the positivity clock by (1 + lambda) times the two-point
# coefficient. Removes the reject-retry storm of the skewed junction cell beside a blunt base. Default -1: the full
# correction, bitwise. Bound in the species configuration as CMeshSpeciesSSTCorrectionLimitV2 (3 Oct 2026: the clock's
# lambda allowance is charged on the corrected faces only; V1, which also charged the uncorrected wake-cut and wall faces,
# is refused on restart; memory/project_docs/_records/2026-10-02_blunt_te/sst_clock/FIX.md).
const SPECIES_SST_CLIMIT = parse(Float64, get(ENV, "TOPO_SPECIES_SST_CORRECTION_LIMIT", "-1"))
SPECIES_SST_CLIMIT < 0 || (isfinite(SPECIES_SST_CLIMIT) && SPECIES_SST_CLIMIT <= 1) ||
    error("TOPO_SPECIES_SST_CORRECTION_LIMIT must lie in [0, 1] (negative: off; got $SPECIES_SST_CLIMIT)")
SPECIES_SST_CLIMIT >= 0 && !(CONSERVATIVE && SST_ENABLED && SPECIES_RECON) &&
    error("TOPO_SPECIES_SST_CORRECTION_LIMIT requires TOPO_SPECIES=1, TOPO_SST=1 and TOPO_SPECIES_RECON=1")
# Stage-1 clock margin (3 Oct 2026, opt-in; memory/project_docs/_records/2026-10-02_blunt_te/sst_clock/BB8_STORM.md and FIX.md): m in (0, 1]
# caps the step's proposal at m x the stage-1 phase-source, viscous and SST clocks, as TOPO_SPECIES_CLOCK_MARGIN already
# does for the face cap. Without it a binding clock that falls by more than 16 eps(Float32) within one step fails its
# stage-2 check at every first attempt and the step halves (arm BB8's storm). A time-step setting: no flux or clock
# changes. Unset (the default): off, bitwise. Bound in the species configuration as CMeshSpeciesStageClockMarginV1/m=<m>.
# The line-implicit step and the steady mode refuse it (they form their steps from other clocks and margins).
const SPECIES_STAGE_MARGIN_SET = haskey(ENV, "TOPO_SPECIES_STAGE_CLOCK_MARGIN")
SPECIES_STAGE_MARGIN_SET && isempty(strip(ENV["TOPO_SPECIES_STAGE_CLOCK_MARGIN"])) &&
    error("TOPO_SPECIES_STAGE_CLOCK_MARGIN may not be empty (unset it for the default, off)")
const SPECIES_STAGE_MARGIN = SPECIES_STAGE_MARGIN_SET ? parse(Float64, ENV["TOPO_SPECIES_STAGE_CLOCK_MARGIN"]) : -1.0
SPECIES_STAGE_MARGIN_SET && !(isfinite(Float32(SPECIES_STAGE_MARGIN)) && 0.5f0 <= Float32(SPECIES_STAGE_MARGIN) <= 1.0f0) &&
    error("TOPO_SPECIES_STAGE_CLOCK_MARGIN must lie in [0.5, 1] (unset: off; got $SPECIES_STAGE_MARGIN)")
SPECIES_STAGE_MARGIN_SET && !CONSERVATIVE && error("TOPO_SPECIES_STAGE_CLOCK_MARGIN requires TOPO_SPECIES=1")
SPECIES_STAGE_MARGIN_SET && IMEX &&
    error("TOPO_SPECIES_STAGE_CLOCK_MARGIN is not supported with TOPO_SPECIES_IMEX=1: the line-implicit step forms its proposal from its i-direction clocks and its own growth cap, and refuses the margin")
SPECIES_STAGE_MARGIN_SET && STEADY &&
    error("TOPO_SPECIES_STAGE_CLOCK_MARGIN is not supported with TOPO_SPECIES_STEADY=1: the steady mode's local steps already carry STEADY_MARGIN on every cap")
# The reconstruction's phase-bound retreat (15 Oct 2026, opt-in; option H of the V1-N60 storm review,
# memory/project_docs/_records/2026-10-15_recon_h/DESIGN.md): TOPO_SPECIES_RECON_RETREAT = halving (the default, bitwise: when a face's phase
# densities leave 0 <= q <= rho at the limiter's theta, one prevfloat step and then theta halves, a jump of the face states and of
# the face cap) or continuous (the largest common phase theta meeting every face's phase bounds, in closed form). Bound in the
# species configuration as CMeshSpeciesContinuousPhaseRetreatV1, so a restart cannot switch it. Refused without TOPO_SPECIES=1 and
# TOPO_SPECIES_RECON=1. The rule is a pure function, unit-tested by tools/test_species_model_knobs.jl.
const SPECIES_RECON_RETREAT = try
    CMeshCavSpecies._species_recon_retreat_setting(ENV, CONSERVATIVE, SPECIES_RECON)
catch err
    err isa ArgumentError ? error(err.msg) : rethrow()
end
# Freestream SST turbulence (1 Oct 2026, the NACA 0012 code-to-code check against NASA TMR,
# memory/project_docs/2026-10-01_NACA0012_CHECK_PREREGISTRATION.md): TOPO_SST_TU_INF (turbulence intensity) and TOPO_SST_MUT_RATIO
# (eddy-viscosity ratio) are the inflow values given to set_sst_ka!. The defaults are the literals the driver has always
# passed (0.001, 0.1), parsed from the same strings, so the default march is bitwise unchanged. A non-default pair is in
# the run tag and refused with a restart (the checkpoint does not carry it).
const SST_TU_INF = parse(Float64, get(ENV, "TOPO_SST_TU_INF", "0.001"))
const SST_MUT_RATIO = parse(Float64, get(ENV, "TOPO_SST_MUT_RATIO", "0.1"))
isfinite(SST_TU_INF) && 0 < SST_TU_INF < 0.2 || error("TOPO_SST_TU_INF must lie in (0, 0.2) (got $SST_TU_INF)")
isfinite(SST_MUT_RATIO) && 0 < SST_MUT_RATIO <= 1.0e4 || error("TOPO_SST_MUT_RATIO must lie in (0, 1e4] (got $SST_MUT_RATIO)")
const SST_FREESTREAM_DEFAULT = SST_TU_INF == 0.001 && SST_MUT_RATIO == 0.1
!SST_FREESTREAM_DEFAULT && !SST_ENABLED && error("TOPO_SST_TU_INF / TOPO_SST_MUT_RATIO require TOPO_SST=1")
CONSERVATIVE && !(CAVLV==4 && CAVWARM==4 && OUTLET_PRESSURE_SET && MARCH_OMEGA==1.0) &&
    error("TOPO_SPECIES=1 requires cav=4 for both phases, a channel pressure outlet and omega=1")
const RESTART_RAW = get(ENV,"CAV_MOVIE_RESTART","")
const RESTART_SET = !isempty(RESTART_RAW)
RESTART_SET && !CONSERVATIVE && error("CAV_MOVIE_RESTART requires TOPO_SPECIES=1")
# A blunt section with the uniform start (2 Oct 2026): measured on the stage-0 blunt meshes at sigma 6, the impulsive
# start evacuates the base cells (p down to the EOS floor, vapour fraction 0.20, still 0.16 after 2000 steps), which the
# same start on the sharp section does not. The blunt panel start is the measured one; the uniform start is not refused.
# A restart builds no start field (its TOPO_INIT=uniform is the restart convention), so it is not warned.
MESH_TE_BASE > 0 && !INIT_PANEL && !RESTART_SET &&
    println("⚠ BLUNT TRAILING EDGE WITH THE UNIFORM START: measured to put vapour at the base at sigma 6 on the stage-0 meshes; TOPO_INIT=panel is the measured start")
CONSERVATIVE && isempty(get(ENV,"CAV_MOVIE_OUT_DIR","")) &&
    error("a conservative run requires a new CAV_MOVIE_OUT_DIR for its checkpoints and scored segment")
const RESTART_PATH = RESTART_SET ? abspath(expanduser(RESTART_RAW)) : ""
RESTART_SET && !isfile(RESTART_PATH) && error("restart file does not exist: $RESTART_PATH")
RESTART_SET && !SST_FREESTREAM_DEFAULT && error("TOPO_SST_TU_INF / TOPO_SST_MUT_RATIO are not supported with CAV_MOVIE_RESTART")
# SST omega start from the wall distance (2 Oct 2026, blunt trailing edge; memory/project_docs/_records/2026-10-02_blunt_te/sst_clock/):
# "wall" raises the initial omega to Menter's near-wall solution, omega = max(w_inf, 6 nu/(beta1 d^2)), beta1 = 0.075,
# with nu_t = k/omega, before the species state is built. On a blunt mesh the first wake cell beside the base midpoint
# otherwise starts with a 400-fold omega jump to its wall-adjacent neighbour; on that skewed cell the explicit
# non-orthogonal diffusion correction collapsed the conservative SST clock (a reject-retry storm from the first steps,
# measured at dn1 5e-5). It changes the initial condition only. Default "freestream": the existing start, bitwise.
const SST_OMEGA_INIT = get(ENV, "TOPO_SST_OMEGA_INIT", "freestream")
SST_OMEGA_INIT in ("freestream", "wall") || error("TOPO_SST_OMEGA_INIT must be freestream or wall (got $SST_OMEGA_INIT)")
SST_OMEGA_INIT == "wall" && !(SST_ENABLED && CONSERVATIVE) && error("TOPO_SST_OMEGA_INIT=wall requires TOPO_SST=1 and TOPO_SPECIES=1")
SST_OMEGA_INIT == "wall" && KEPSV != 0 && error("TOPO_SST_OMEGA_INIT=wall is an omega start; k-eps reinterprets omega as epsilon")
SST_OMEGA_INIT == "wall" && RESTART_SET && error("TOPO_SST_OMEGA_INIT=wall is a start field; a restart's state is authoritative")
MESH_TE_SETTINGS !== nothing && !CONSERVATIVE && error("the trailing-edge mesh knobs require TOPO_SPECIES=1 (the production path)")
MESH_QUALITY !== nothing && !CONSERVATIVE && error("TOPO_MESH_QUALITY=v1 requires TOPO_SPECIES=1 (the production path)")
# What the strip topology refuses at the driver (plan section 3.3), before the output directory is claimed. The legacy path
# (TOPO_SPECIES=0), the fixed-outer control and a run without a channel are refused above with every trailing-edge knob; the
# solver refuses kinetic faces, the wall function and the steady mode again with a strip bound (cmesh_species_strip.jl,
# cmesh_species_steady.jl). The conservation ledger, refused in version 1, runs over both blocks since version 1.5 (plan
# section 4.8, 7 Oct 2026), so TOPO_SPECIES_LEDGER=1 is admitted with the strip. The line-implicit step, refused here until
# stage 4, carries the wake line through the strip since then (plan section 5, 7 Oct 2026; cmesh_species_imex.jl and
# cmesh_species_strip.jl), so TOPO_SPECIES_IMEX=1 is admitted with the strip; its own refusals above (the ledger, kinetic faces,
# graph execution, immediate admission, a warm-up, the steady mode, the stage clock margin) hold on every mesh, the strip's too.
# The legacy TOPO_WALLFUN sets the same solver scalar as TOPO_SPECIES_WALL_FUNCTION=reichardt when it rounds to 2 in Float32
# (s.wallfun = 2, the all-y+ wall function): refused here by name too (7 Oct 2026, the stage-3 review gate's O2), where the
# solver's strip admission would otherwise refuse it at build, after the output directory is claimed, under the other knob's name.
if MESH_TE_STRIP
    STEADY && error("TOPO_SPECIES_STEADY=1 does not support the strip topology (TOPO_MESH_TE_TOPOLOGY=strip, CMeshSpeciesStripTopologyV1) in version 1: steady pseudo-time mode is refused in all its variants; use the physical-time step")
    SPECIES_KFVS && error("TOPO_SPECIES_FLUX=kfvs does not support the strip topology (TOPO_MESH_TE_TOPOLOGY=strip, CMeshSpeciesStripTopologyV1) in version 1: the base wall would need the kinetic half-mass zeros on an i face; the strip runs the AUSM+-up faces")
    SPECIES_WALL_FUNCTION==:reichardt && error("TOPO_SPECIES_WALL_FUNCTION=reichardt does not support the strip topology (TOPO_MESH_TE_TOPOLOGY=strip, CMeshSpeciesStripTopologyV1) in version 1: the base cells would need the wall-law strain and the omega pin on an i face")
    Float32(WALLFUNV)==2.0f0 && error("TOPO_WALLFUN=$(WALLFUNV) does not support the strip topology (TOPO_MESH_TE_TOPOLOGY=strip, CMeshSpeciesStripTopologyV1) in version 1: it sets s.wallfun = 2, the all-y+ wall function (TOPO_SPECIES_WALL_FUNCTION=reichardt's solver scalar), and the base cells would need the wall-law strain and the omega pin on an i face")
    MAKIE && error("CAV_MOVIE_MAKIE=1 does not support the strip topology (TOPO_MESH_TE_TOPOLOGY=strip, CMeshSpeciesStripTopologyV1): the Makie pixmap renders one block; set CAV_MOVIE_MAKIE=0 (the Plots frames render both blocks)")
end
# High-rate base pressure probe (2 Oct 2026, blunt trailing edge plan section 5.6): every N accepted steps, the
# face-length-weighted mean C_p over the base faces' first cells, written to base/<tag>_baseprobe.csv. The 0.1 CTU frame
# cadence would alias any base shedding (period of order 0.1 CTU at f(h + 2 delta*)/U = 0.2). One small device read per
# probe. Default 0: off.
const BASE_PROBE_EVERY = parse(Int, get(ENV, "TOPO_BASE_PROBE_EVERY", "0"))
BASE_PROBE_EVERY >= 0 || error("TOPO_BASE_PROBE_EVERY must be >= 0 (got $BASE_PROBE_EVERY)")
BASE_PROBE_EVERY > 0 && MESH_TE_BASE == 0 && error("TOPO_BASE_PROBE_EVERY requires TOPO_MESH_TE_BASE > 0")
# The steady loop (TOPO_SPECIES_STEADY=1) exits before the march, where the probe is sampled and the base sidecar is
# written: a steady blunt run carries no base reading, so a probe request there would be silently ignored.
BASE_PROBE_EVERY > 0 && STEADY &&
    error("TOPO_BASE_PROBE_EVERY is a physical-time probe; not supported with TOPO_SPECIES_STEADY=1")
MESH_TE_BASE > 0 && STEADY &&
    println("⚠ STEADY MODE ON A BLUNT TRAILING EDGE: the steady loop writes no base sidecar (base/<tag>_base.csv); read the base from its checkpoints")
const WARM_STEPS = RESTART_SET ? 0 : NWARM
const THETA = 0.10; const THETA_DENSE = 0.50; const MIN_AREA = 20
# ★★ CLOUD-FILTER SENSITIVITY LADDER. First pair is PRODUCTION (Gate B); the rest relax the two
# filters independently and together. tools/test_shedding_detector.jl showed those filters are a HARD
# BINARY CLIFF -- 4 of 4 events at peak alpha 0.50, ZERO at 0.49, no warning -- so a single-pair
# negative cannot distinguish "steady" from "shedding, but dilute".
const FILTERS = [(0.50, 20), (0.25, 20), (0.10, 20), (0.50, 5), (0.10, 5)]
const TG = get(ENV, "CAV_MOVIE_TG", "0") == "1"
const WIN = let s = get(ENV, "CAV_MOVIE_WIN", "-0.15,1.35,-0.30,0.45")
    v = parse.(Float64, split(s, ",")); (v[1], v[2], v[3], v[4]) end
const PXW = parse(Int, get(ENV, "CAV_MOVIE_PX", "1100"))

# ---- tag: every physics knob in the name (filename discipline) ---------------------------------
tag = @sprintf("cavmov_sig%.2f_m%dx%dd%.0e_T%.0f", SIGMA, MESH[1], MESH[2], MESH[3], TCTU)
isempty(FOIL)     || (tag *= "_f" * splitext(basename(FOIL))[1][1:min(12,length(splitext(basename(FOIL))[1]))])
GROWTH != 1.3     && (tag *= @sprintf("_g%g", GROWTH))
MESH_LE_DS > 0    && (tag *= @sprintf("_le%.3g", MESH_LE_DS))
# Trailing-edge mesh knobs (2 Oct 2026): named only when set, so default tags are unchanged. The foil fragment is 12
# characters and cannot tell a blunt section file from its sharp companion; _nb does. The wake column count is named
# whenever a trailing-edge knob is set, because the near-wake ladders change it.
MESH_TE_BASE > 0  && (tag *= @sprintf("_nb%d", MESH_TE_BASE))
MESH_TE_DS > 0    && (tag *= @sprintf("_te%.3g", MESH_TE_DS))
MESH_WAKE_NEAR === nothing || (tag *= @sprintf("_wkn%.3gx%.3g", MESH_WAKE_NEAR.wake_near_ds, MESH_WAKE_NEAR.wake_near_len))
# the near wake's far growth, named only when it differs from the 1.05 default (so arms that differ only in it differ)
MESH_WAKE_NEAR === nothing || MESH_WAKE_NEAR.wake_near_growth == 1.05 ||
    (tag *= @sprintf("g%.3g", MESH_WAKE_NEAR.wake_near_growth))
MESH_TE_SETTINGS !== nothing && (tag *= @sprintf("_nwk%d", NWAKE))
MESH_TE_STRIP && (tag *= "_strip")           # the strip topology (5 Oct 2026), after _nwk so every other tag is unchanged
# the mesh quality option (15 Oct 2026): _mq1h<first 8 hex of sha256 of its tag>, so meshes differing in any parameter differ;
# the wake count is named too when no trailing-edge knob already named it (Q-A sets the wake count it needs)
MESH_QUALITY === nothing || (tag *= (MESH_TE_SETTINGS === nothing ? @sprintf("_nwk%d", NWAKE) : "") * "_mq1h" *
                                    bytes2hex(sha256(MESH_QUALITY_TAG_STR))[1:8])
SST_OMEGA_INIT == "wall" && (tag *= "_wwall")
SPECIES_SST_CLIMIT >= 0 && (tag *= @sprintf("_sstcl%g", SPECIES_SST_CLIMIT))
SPECIES_STAGE_MARGIN_SET && (tag *= @sprintf("_scm%g", SPECIES_STAGE_MARGIN))
SPECIES_RECON_RETREAT === :continuous && (tag *= "_rrc")   # the continuous phase-bound retreat (15 Oct 2026); halving adds nothing
SST_FREESTREAM_DEFAULT || (tag *= @sprintf("_tu%.3g_mutr%.3g", SST_TU_INF, SST_MUT_RATIO))
SMOOTH != 40      && (tag *= @sprintf("_sm%d", SMOOTH))
RATIOV != 100.0   && (tag *= @sprintf("_rt%g", RATIOV));  CPRODV != 100.0 && (tag *= @sprintf("_Cp%g", CPRODV))
CDESTV != 5.0     && (tag *= @sprintf("_Cd%g", CDESTV));  A2LF   != 1.4   && (tag *= @sprintf("_a2l%g", A2LF))
LQV    != 0.0     && (tag *= @sprintf("_lq%g", LQV))
EPSPV  != 0.1     && (tag *= @sprintf("_epv%g", EPSPV))
MACH   != 0.2     && (tag *= @sprintf("_M%g", MACH));     RPV    != 0.0   && (tag *= @sprintf("_rp%g", RPV))
# The archived records carry NO tag for cav=2 / omega=0.2; every other value is named so a new record
# can never collide with (or overwrite the frame directory of) an archived one.
CAVLV != 2          && (tag *= @sprintf("_cav%d", CAVLV))
CAVWARM != CAVLV    && (tag *= @sprintf("_cw%d", CAVWARM))
MARCH_OMEGA != 0.2  && (tag *= @sprintf("_om%g", MARCH_OMEGA))
MARCH_RK_INCREMENTAL && (tag *= "_rki1")
CONSERVATIVE && (tag *= "_species1_exactref")
SPECIES_RECON && (tag *= "_muscl")
SPECIES_KFVS && (tag *= "_kfvs")
SPECIES_EOS==:single && (tag *= "_eos32")
SPECIES_SOURCE_WEIGHT==:beta && (tag *= "_srcbeta")
# (10 Oct 2026) the Schnerr-Sauer source _ssn<n0>d<d>c<chord>, the free gas _fg<phi>, then the Reboud treatment: _rbv<n> the
# void-fraction form (the default, 13 Oct), _rbc<n> the composition form of 10 Oct, _reboud<n> the density form of 20 Sep (T25's
# name), nothing with none (every archived name); a trailing p with the production-consistent form (13 Oct)
tag *= CMeshCavSpecies._species_model_tag_suffix(SPECIES_NUT_SETTING, SPECIES_SOURCE_SETTING, SPECIES_FREE_GAS;
                                                 production=SPECIES_REBOUD_PRODUCTION)
SPECIES_WALL_FUNCTION==:reichardt && (tag *= "_wfr")
SPECIES_TRIP_X != 0 && (tag *= @sprintf("_trip%.3g", SPECIES_TRIP_X) * (SPECIES_TRIP_WIDTH != 0.005 ? @sprintf("w%.3g", SPECIES_TRIP_WIDTH) : ""))
FIXED_OUTER_SETTINGS !== nothing &&
    (tag *= @sprintf("_fo%dr%gj%d", FIXED_OUTER_SETTINGS.version,
                    FIXED_OUTER_SETTINGS.reference_dn1, FIXED_OUTER_SETTINGS.transition_j))
RESTART_SET && (tag *= "_restart")
SST_ENABLED || (tag *= "_laminar")
TVPV   != 0.0     && (tag *= @sprintf("_tvp%g", TVPV));   SLIPV  != 0.0   && (tag *= @sprintf("_sl%g", SLIPV))
CLOUDV != 0.0     && (tag *= @sprintf("_cld%g", CLOUDV))
POLYV    != 0.0 && (tag *= @sprintf("_ply%g", POLYV))
HKV      != 0.0 && (tag *= @sprintf("_hk%g", HKV))
WALLNUCV != 0.0 && (tag *= @sprintf("_wn%g", WALLNUCV))
KEPSV    != 0.0 && (tag *= @sprintf("_keps%g", KEPSV))
WALLFUNV != 0.0 && (tag *= @sprintf("_wf%g", WALLFUNV))
WFPGV    != 0.0 && (tag *= @sprintf("_wfpg%g", WFPGV))
CHANNEL_H > 0 && (tag *= @sprintf("_chH%g_xi%gto%g", CHANNEL_H, CHANNEL_XMIN, CHANNEL_XMAX))
OUTLET_PRESSURE_SET && (tag *= OUTLET_PRESSURE_RAW == "ambient" ? "_outpambient" : @sprintf("_outp%.8g", OUTLET_PRESSURE))
INIT_PANEL && (tag *= "_initpanel")
STEADY && (tag *= "_steady")
IMEX && (tag *= IMEX_SOURCE_RAW=="implicit" ? "_imexsrc" : "_imex")

const CAV_MOVIE_OUT_DIR_SET = haskey(ENV, "CAV_MOVIE_OUT_DIR")
const CAV_MOVIE_OUT_DIR_RAW = get(ENV, "CAV_MOVIE_OUT_DIR", "")

"True when `child` is `parent` itself or is lexically below it."
function movie_path_within(child::AbstractString, parent::AbstractString)
    rel = relpath(abspath(child), abspath(parent))
    rel == "." || (!isabspath(rel) && first(splitpath(rel)) != "..")
end

"Normalise and validate an explicitly requested one-shot movie output directory."
function movie_override_out_dir(raw::AbstractString, root::AbstractString)
    requested = strip(raw)
    isempty(requested) && error("CAV_MOVIE_OUT_DIR was set but is empty")

    # Relative overrides are repository-local. An absolute value is an explicit opt-in to an
    # external destination (for example a scratch volume), and is still subject to the exclusive
    # creation and existing-parent rules below.
    out = if isabspath(requested)
        normpath(requested)
    else
        candidate = normpath(joinpath(root, requested))
        movie_path_within(candidate, root) ||
            error("relative CAV_MOVIE_OUT_DIR must remain inside the repository: $requested")
        candidate
    end

    dirname(out) == out && error("CAV_MOVIE_OUT_DIR may not be a filesystem root: $out")
    parent = dirname(out)
    isdir(parent) || error("CAV_MOVIE_OUT_DIR parent must already exist: $parent")
    if !isabspath(requested)
        movie_path_within(realpath(parent), realpath(root)) ||
            error("relative CAV_MOVIE_OUT_DIR resolves outside the repository: $requested")
    end
    out
end

const OUTD = CAV_MOVIE_OUT_DIR_SET ?
    movie_override_out_dir(CAV_MOVIE_OUT_DIR_RAW, ROOT) :
    joinpath(ROOT, "CASES", "CMESH_RAE2822", "RESULTS")
const FRD  = joinpath(OUTD, "CAVMOV_FRAMES_" * tag)
if CAV_MOVIE_OUT_DIR_SET
    # Override mode claims a fresh directory before any solver build. `mkdir`, unlike `mkpath`, is
    # an exclusive claim and closes the ordinary check/create race by failing if another process
    # wins it. A later failure deliberately leaves the partial candidate in place as evidence; this
    # branch promises neither rollback nor protection from hostile concurrent writes inside OUTD.
    (ispath(OUTD) || islink(OUTD)) &&
        error("CAV_MOVIE_OUT_DIR already exists; refusing to overwrite it: $OUTD")
    mkdir(OUTD)
    mkdir(FRD)
else
    # Preserve the historical default for existing callers: reuse RESULTS and replace only this
    # tag's frame directory. No recursive deletion is reachable from override mode.
    mkpath(OUTD)
    rm(FRD; force=true, recursive=true)
    mkpath(FRD)
end

# ---- build ------------------------------------------------------------------------------------
# ⚠ USE THE SHARED LOADER. My first version here was a naive two-column parser; it read the file
# without error and produced a point ORDERING the mesh generator could not extrude, giving
# "build_metrics: 152 non-positive cell volumes" -- a mesh failure masquerading as a solver failure.
# `read_airfoil_dat` (src/cmesh/cmesh_gen.jl) is what every other tool uses and handles the
# Selig/dual conventions. Same lesson as the EOS: do not re-implement, call the one that exists.
foilpts = isempty(FOIL) ? error("TOPO_FOIL must be set for the benchmark") : read_airfoil_dat(FOIL)
channel = CHANNEL_H > 0 ? ChannelSpec(; height=CHANNEL_H, xi_in=CHANNEL_XMIN,
                                      xi_out=CHANNEL_XMAX, xi_ref=0.5,
                                      ref=(0.5, 0.0), angle_deg=ALPHA) : nothing
mesh = generate_cmesh(foilpts; n_foil=MESH[1], n_wake=NWAKE, n_normal=MESH[2],
                      dn1=FIXED_OUTER_SETTINGS === nothing ? MESH[3] : FIXED_OUTER_SETTINGS.reference_dn1,
                      growth_max=GROWTH, R_ff=25.0, L_wake=25.0, smooth_iters=SMOOTH, bl_freeze=6, le_ds=MESH_LE_DS,
                      surface_evaluator=MESH_SURFACE_EVALUATOR,surface_distribution=MESH_SURFACE_DISTRIBUTION,
                      channel=channel, (MESH_TE_SETTINGS === nothing ? (;) : MESH_TE_SETTINGS)...,
                      (MESH_QUALITY === nothing ? (;) : (quality=MESH_QUALITY,))...)
const FIXED_OUTER_PROVENANCE = if FIXED_OUTER_SETTINGS === nothing
    nothing
else
    reference_mesh = mesh
    mesh = fixed_outer_mesh(reference_mesh; target_dn1=MESH[3],
        transition_j=FIXED_OUTER_SETTINGS.transition_j, growth_limit=GROWTH,
        version=FIXED_OUTER_SETTINGS.version)
    fixed_outer_provenance(reference_mesh, mesh; transition_j=FIXED_OUTER_SETTINGS.transition_j,
        growth_limit=GROWTH, version=FIXED_OUTER_SETTINGS.version)
end
FIXED_OUTER_PROVENANCE === nothing ||
    println("FIXED OUTER MESH RECEIPT: ", join((string(k,"=",v) for (k,v) in pairs(FIXED_OUTER_PROVENANCE))," "))
# The strip topology (5 Oct 2026, plan section 4.7): the strip block's node arrays are hashed as the C block's and printed
# between them and the foil's (strip_x_sha256, strip_y_sha256: the wall binder's spelling); every other mesh prints the
# line it printed before (the inserted text is empty).
const STRIP_MESH_RECEIPT = MESH_TE_STRIP ?
    string(" strip_x_sha256=", bytes2hex(sha256(reinterpret(UInt8, vec(mesh.strip.X)))),
           " strip_y_sha256=", bytes2hex(sha256(reinterpret(UInt8, vec(mesh.strip.Y))))) : ""
println("MESH COORDINATE RECEIPT: x_sha256=", bytes2hex(sha256(reinterpret(UInt8, vec(mesh.X)))),
        " y_sha256=", bytes2hex(sha256(reinterpret(UInt8, vec(mesh.Y)))), STRIP_MESH_RECEIPT,
        " foil_sha256=", bytes2hex(sha256(read(FOIL))))
println("SURFACE STATION RECEIPT: evaluator=",MESH_SURFACE_EVALUATOR,
        " distribution=",MESH_SURFACE_DISTRIBUTION,
        " version=",MESH_SURFACE_DISTRIBUTION===:curvature ? "curvature_v1" : "legacy",
        " le_ds=",MESH_LE_DS)
# The mesh quality receipt (15 Oct 2026), only with TOPO_MESH_QUALITY=v1: the canonical parameters (the readers rebuild the
# mesh from them) and the generator's own measurement of the delivered mesh (CMeshGen.LAST_QUALITY_BUILD: the G2 quantities,
# the pre-reparameterisation ratio, the cell count and, with the strip, the base faces' range).
if MESH_QUALITY !== nothing
    let b = CMeshGen.LAST_QUALITY_BUILD[], sm = b.summary
        b.tag == MESH_QUALITY_TAG_STR || error("the generator's last quality build is not this run's ($(b.tag))")
        base = mesh.strip === nothing ? (base_ds_min=nothing, base_ds_max=nothing) :
            let d = [hypot(mesh.strip.X[1,j+1]-mesh.strip.X[1,j], mesh.strip.Y[1,j+1]-mesh.strip.Y[1,j]) for j in 1:mesh.strip.NJ]
                (base_ds_min=minimum(d), base_ds_max=maximum(d))
            end
        println(mesh_quality_receipt_line(MESH_QUALITY_TAG_STR, (n_wake=mesh.NIW, n_base=MESH_TE_BASE, cells=sm.cells,
            gates=all(g[5] for g in b.gates) ? "pass" : "FAIL", gate_set=MESH_QUALITY.preset === :air ? "air" : "cavitating",
            pre_repar_ratio=b.pre_repar_ratio,
            theta_max=sm.theta_max, theta_out3=sm.theta_out3, n60=sm.n60, n70=sm.n70, theta_v_max=sm.theta_v_max,
            ar_max=sm.ar_max, ar_out3=sm.ar_out3, ar_wall3=sm.ar_wall3, ar_offband=sm.ar_offband, ar_strip=isnan(sm.ar_strip) ? nothing : sm.ar_strip,
            vr_max=sm.vr_max, vr_junction=sm.vr_junction, vr_nonjunction=sm.vr_nonjunction, skew_max=sm.skew_max, skew_cut=sm.skew_cut,
            skew_of_max=sm.skew_of_max, rn_band=sm.rn_band, rn_else=sm.rn_else, rt_foil=sm.rt_foil, rt_cut=sm.rt_cut, rt_te=sm.rt_te,
            thb_outlet_max=sm.thb_outlet_max, thb_wall_p50=sm.thb_wall_p50, min_angle=sm.minang, len_min=sm.len_min, base...,
            (mesh.strip === nothing ? (;) : (rt_base=sm.rt_base, rt_corner_max=sm.rt_corner_max, rt_corner_min=sm.rt_corner_min,
                                            theta_junction=sm.theta_junction, skew_junction=sm.skew_junction))...)))
    end
end
# The object overload is mandatory: the matrix overload intentionally creates a legacy all-zero
# mask and would silently turn a requested tunnel back into a characteristic far field.
met  = build_metrics(mesh)
# The strip block's metrics (outer tags all OUTER_STRIP_INTERFACE, NIF = 0, the wall distance to the base); the channel
# admission then gates and prints the strip's GCL, folds and corner sine too (REALISED STRIP QUALITY). nothing otherwise.
met_strip = MESH_TE_STRIP ? build_strip_metrics(mesh) : nothing
admit_outer_geometry(mesh, met; label="cav movie", growth_limit=GROWTH, strip_met=met_strip)
# CAV_MOVIE_BACKEND=cpu (30 Sep 2026): the KernelAbstractions CPU backend, for CPU tests of the driver on small meshes
# (e.g. the steady mode's); default cuda, unchanged. Graph execution requires cuda.
const BACKEND_RAW = get(ENV, "CAV_MOVIE_BACKEND", "cuda")
BACKEND_RAW in ("cuda", "cpu") || error("CAV_MOVIE_BACKEND must be cuda or cpu (got $BACKEND_RAW)")
BACKEND_RAW == "cpu" && SPECIES_GRAPH && error("TOPO_SPECIES_GRAPH=1 requires CAV_MOVIE_BACKEND=cuda")
backend = BACKEND_RAW == "cpu" ? KernelAbstractions.CPU() : CUDA.CUDABackend()
s = build_sol_ka(backend, met; drot=2.0f0, cxv=cx, cyv=cy, czv=cz, wv=w)
s.wall_flux = true; s.recon = Int32(1); s.fullvisc = Int32(1)
set_freestream_ka!(s, MACH, ALPHA; pg_correction=true)
U = sqrt(Float64(s.uxi)^2 + Float64(s.uyi)^2)
# ★ REALISED INCIDENCE BANNER (2026-08-07). Added because a user watching the rendered video asked
# "are you sure this is 8 degrees? it doesn't look like it" -- AND THE RECORD COULD NOT ANSWER.
# This tool never printed the angle: it was asserted by a constant, which is the exact pattern that
# produced DEFECT A (a probe that ran a NACA0015 for weeks while every script exported the NACA66).
# The campaign rule is that a realised state must be COMPUTED FROM THE STATE, never asserted.
# ⚠ WHY THE VIDEO LOOKS LEVEL, and it is not a bug: generate_cmesh() takes NO angle, so the foil sits
# at ZERO geometric incidence in the mesh and is rendered horizontal. The incidence is imposed by
# ROTATING THE FREESTREAM (uxi, uyi below), which is equivalent for the circular R_ff=25c far field.
# The angle is therefore invisible in the geometry and lives only in the flow direction.
# (The rotated-tunnel case is different and DOES rotate the mesh -- that is what ChannelSpec's
# angle_deg is for -- because a rectangular tunnel is not rotationally symmetric.)
@printf("★ REALISED INCIDENCE: requested alpha=%.3f deg -> realised atan2(uy,ux)=%.6f deg  [uxi=%.6f uyi=%.6f |U|=%.6f, ca=%.6f sa=%.6f]\n",
        ALPHA, atand(Float64(s.uyi), Float64(s.uxi)),
        Float64(s.uxi), Float64(s.uyi), U, Float64(s.ca), Float64(s.sa))
@printf("★ REALISED GEOMETRIC INCIDENCE OF THE MESH: 0.000 deg (foil unrotated; free-air far field). Expect a LEVEL foil in every rendered frame -- the %.1f deg is in the FLOW, not the shape.\n", ALPHA)
abs(atand(Float64(s.uyi), Float64(s.uxi)) - ALPHA) < 1e-4 ||
    error("realised freestream incidence does not match the requested alpha")
s.nu_lam = Float32(U/RE)
SST_ENABLED && set_sst_ka!(s; tu_inf=SST_TU_INF, mu_t_ratio=SST_MUT_RATIO)
SST_ENABLED && !SST_FREESTREAM_DEFAULT && @printf("★ SST FREESTREAM (non-default): tu_inf=%.6g mu_t_ratio=%.6g -> k_inf=%.6g w_inf=%.6g (inflow values)\n",
    SST_TU_INF, SST_MUT_RATIO, Float64(s.k_inf), Float64(s.w_inf))
s.cav = Int32(CAVWARM)                                   # warm-up level; the scored march sets TOPO_CAV
# ★ S3g STEP 2 (mixture cv + latent heat). emix = 0 => yvf never written/read and every energy
# expression collapses to the legacy single-cv form BITWISE. eint_v defaults to the liquid cv so
# that turning emix on with no cv split isolates the LATENT HEAT term alone -- the two effects are
# separable knobs on purpose, because a combined change could not be attributed.
s.emix = Float32(EMIXV); s.latent = Float32(LATENTV); s.keps = Float32(KEPSV)
s.wallfun = SPECIES_WALL_FUNCTION==:reichardt ? 2.0f0 : Float32(WALLFUNV); s.wf_larsson = Float32(WFPGV)
# ★ k-eps reinterprets `tom` as epsilon, so every initial/boundary value built as an OMEGA has to be
# converted. Skipping this made the first k-eps runs LAMINAR (nu_t ~ 1e-16) while looking healthy.
set_keps_ka!(s)
s.eint_v = Float32(CVVV > 0 ? CVVV : Float64(s.eint))
# The TEM of a carrier: the C block's, and with the strip topology the strip's too (5 Oct 2026), so both blocks carry one
# physics from one constructor call (the configuration census of tools/audit_knobs.jl keeps a single call site).
movie_tem(q) = build_tem(q; sigma=SIGMA, Cprod=CPRODV, Cdest=CDESTV, ratio=RATIOV, vfloor=false, eos=1,
                exact_reference=CONSERVATIVE,
                eps_pv=EPSPV, wallis=1, Tmin=Float32(TMINV), rate_ref_pv=true, arecon=ARECONV, vaploc=1.0, a2l_frac=A2LF, lq=LQ_EFFECTIVE[],
                rp=RPV, rp_S=RP_S, rp_pg0=RP_PG0, rp_kappa=RP_KAP,
                tvp=TVPV, slip=SLIPV, slip_tau=SLIPTAU, cloudint=CLOUDV, poly=POLYV, hk=HKV, wallnuc=WALLNUCV, btke=BTKEV, pvturb=PVTURBV, pvvort=PVVORTV, tfree=TFREEV, chord_m=0.150)
# TOPO_SPECIES_FREE_GAS (10 Oct 2026, design note section 3.2): the closed-form conversion of the free-stream gas fraction to lq reads
# the realised Float32 constants (p_v, a2l, psm) of an lq = 0 TEM of this carrier, exactly as build_tem's exact-reference inversion
# forms them; the TEM is then built with that lq, and its own inversion is checked against the closed form below.
const FREE_GAS_CONVERSION = SPECIES_FREE_GAS > 0 ?
    CMeshCavSpecies.free_gas_lq(SPECIES_FREE_GAS, s, movie_tem(s), SIGMA) : nothing
FREE_GAS_CONVERSION === nothing || (LQ_EFFECTIVE[] = FREE_GAS_CONVERSION.lq)
tem = movie_tem(s)
tem.omega = 0.2f0; tem.src_smooth = 0.0f0; tem.src_jac = Int32(3)
if CONSERVATIVE
    tem.omega=1.0f0;tem.src_jac=Int32(0)
    CMeshSolverKA.GLOBAL_DT[]=1
end
p_ref, a2m = tem_ambient(tem, s)
set_farfield_mixture!(s; p_ref=p_ref, a2m=a2m, Tmin=tem.Tmin, Tmax=tem.Tmax)
if CONSERVATIVE
    @printf("CONSERVATIVE OPERATING POINT: exact_reference=true sigma_actual=%.9g saturation_density_ratio=%.9g\n",
            (Float64(p_ref)-Float64(tem.p_v))/(0.5*Float64(s.rho_inf)*U^2),
            Float64(tem.rho_l_sat)/Float64(tem.rho_v_sat))
end
# The realised phase-change source and free gas (10 Oct 2026), computed from the built TEM and carrier, never from the knobs alone.
if CONSERVATIVE && !SPECIES_SS
    println("  SPECIES SOURCE MODEL: merkle (the TEM's rates kprod/kdest from TOPO_CPROD/TOPO_CDEST; September path)")
elseif CONSERVATIVE
    # ep: the EOS tuple the source kernel reads (s.outlet_eos, which set_pressure_outlet! fills from these TEM fields further down;
    # at this point it still holds the carrier's zeros, so it is formed here from the TEM, in the same order)
    let src = SPECIES_SOURCE_SETTING, (_, ssp) = CMeshCavSpecies._source_model_setting(:schnerr_sauer, src.n0, src.dnuc, :alpha),
        pv = Float64(tem.p_v), rl = Float64(tem.rho_l_sat), rv = Float64(tem.rho_v_sat), psm = Float64(tem.psm),
        ep = Float64.((tem.p_v, tem.a2l, tem.rho_l_sat, tem.rho_v_sat, tem.psm, tem.lq, tem.am_floor_frac)),
        c3 = ssp[3], an = ssp[4], R(a) = 3/(c3*cbrt((1-a)/(a+an))),
        seed = CMeshCavSpecies._ss_terms(0.0, 0.0, pv, rl, rv, rl, 0.0, c3, an)[1],
        rtr = CMeshCavTEM.CMeshEOSMix._rhomix_vl(Float64(p_ref), 1e-6, pv, Float64(tem.a2l), rl, rv, psm, Float64(tem.lq)),
        (gtr, ctr) = CMeshCavSpecies._phase_source_ss(rtr, rtr*CMeshCavSpecies._Y(1e-6, rl, rv), ep, (c3, an))
        # (13 Oct 2026) where n0 and d_nuc come from: the knobs, or the defaults, which are OpenFOAM's tutorial calibration
        # constants (its SchnerrSauer class documentation), not a measurement of any tunnel's nuclei
        prov(isdef, knob) = isdef ? "default: OpenFOAM's tutorial constant, not a measurement" : "set by $knob"
        @printf("  SPECIES SOURCE MODEL: schnerr_sauer (OpenFOAM v11 form, Cv = Cc = 1, 0.01 p_v regularisation, nucleus seed, trace level q/r <= %.0e above p_v; tag CMeshSpeciesSchnerrSauerV2/n0=%s/d=%s): n0=%.6g m^-3 (%s) d_nuc=%.6g m (%s) chord=%.6g m -> N0=%.6e per chord^3 D=%.6e chord alpha_nuc=%.6e R(0)=%.6e R(0.5)=%.6e chord (%.3g um); seed rate at p -> 0 S(0)=%.6g per solver time = %.6g per CTU; trace-condensation clock at p_ref (a = 1e-6) %.6g solver time = %.6g CTU (it does not depend on the amount of vapour above the trace level)\n",
                CMeshCavSpecies.SS_TRACE_Y, CMeshCavSpecies._tagnum(Float32(ssp[1])), CMeshCavSpecies._tagnum(Float32(ssp[2])),
                src.n0_m3, prov(src.n0_default, "TOPO_SPECIES_SS_N0"), src.dnuc_m, prov(src.dnuc_default, "TOPO_SPECIES_SS_DNUC"),
                src.chord, ssp[1], ssp[2], an, R(0.0), R(0.5), 1e6*R(0.5)*src.chord, seed, seed/U, ctr, ctr*U)
        isfinite(ctr) && ctr > 0 && gtr < 0 || error("the Schnerr-Sauer banner's trace-condensation state did not condense (gamma $gtr, cap $ctr)")
    end
end
if CONSERVATIVE && FREE_GAS_CONVERSION !== nothing
    let fg = FREE_GAS_CONVERSION, pv = Float64(tem.p_v), rl = Float64(tem.rho_l_sat), a2l = Float64(tem.a2l), lq = Float64(tem.lq),
        x = fg.x, Vph(p) = 1 - (p - pv)/(rl*a2l) + lq*(pv/p - 1), L0 = 1/(rl*a2l), cr(p) = sqrt(L0/(L0 + lq*pv/p^2))
        # the TEM's own exact-reference inversion against the closed form (Float32 storage of lq and rho_l_sat)
        abs(rl/fg.rho_l_sat - 1) <= 1e-6 ||
            error("TOPO_SPECIES_FREE_GAS: the TEM's rho_l_sat $rl differs from the closed form's $(fg.rho_l_sat) beyond Float32 storage")
        phi_ref = lq*pv/x/Vph(x)
        abs(phi_ref/SPECIES_FREE_GAS - 1) <= 1e-6 ||
            error("TOPO_SPECIES_FREE_GAS: the realised gas fraction at p_ref $phi_ref differs from the request $SPECIES_FREE_GAS")
        @printf("  SPECIES FREE GAS: phi_ref=%g requested -> lq=%.9g (TEM, Float32), rho_l_sat=%.9g (closed form %.9g); realised gas fraction of the liquid at p_ref %.6e and at p_v %.6e; liquid sound speed / gas-free at p_ref %.6f, at p_v %.6f; tag CMeshSpeciesFreeGasV1/lq=%s (free gas carried by the liquid: isothermal, attached to the liquid mass, none accumulates in the cavity; no surface tension, so no Blake threshold; not seen by the void-fraction Reboud form or the beta readers, which count it with the liquid; below phi_ref about 2e-5 the admission's sound-speed floor still refuses deep tension)\n",
                SPECIES_FREE_GAS, lq, rl, fg.rho_l_sat, phi_ref, lq/Vph(pv), cr(x), cr(pv), CMeshCavSpecies._tagnum(tem.lq))
    end
end
"""The strip block's carrier and TEM (strip topology, plan sections 3.2 and 4.5): a KSol of the strip's metrics configured
statement for statement as the C block's carrier above (wall flux, reconstruction and full viscosity, the freestream, the
viscosity, SST and its freestream turbulence, the warm-up flux level, the model switches, k-eps, the vapour cv), the C
block's physics (`movie_tem`) and TEM settings, and its own ambient mixture state. The pair's admission
(build_species_ka's strip admission) refuses the pair if any scalar of the carrier or of the TEM differs from the C
block's, so a drift between this function and the C block's lines cannot run."""
function movie_strip_carrier(m)
    q = build_sol_ka(backend, m; drot=2.0f0, cxv=cx, cyv=cy, czv=cz, wv=w)
    q.wall_flux = true; q.recon = Int32(1); q.fullvisc = Int32(1)
    set_freestream_ka!(q, MACH, ALPHA; pg_correction=true)
    q.nu_lam = Float32(U/RE)
    SST_ENABLED && set_sst_ka!(q; tu_inf=SST_TU_INF, mu_t_ratio=SST_MUT_RATIO)
    q.cav = Int32(CAVWARM)
    q.emix = Float32(EMIXV); q.latent = Float32(LATENTV); q.keps = Float32(KEPSV)
    q.wallfun = SPECIES_WALL_FUNCTION==:reichardt ? 2.0f0 : Float32(WALLFUNV); q.wf_larsson = Float32(WFPGV)
    set_keps_ka!(q)
    q.eint_v = Float32(CVVV > 0 ? CVVV : Float64(q.eint))
    t = movie_tem(q)
    t.omega = tem.omega; t.src_smooth = tem.src_smooth; t.src_jac = tem.src_jac
    pr, a2 = tem_ambient(t, q)
    set_farfield_mixture!(q; p_ref=pr, a2m=a2, Tmin=t.Tmin, Tmax=t.Tmax)
    (q, t)
end
s_strip, tem_strip = MESH_TE_STRIP ? movie_strip_carrier(met_strip) : (nothing, nothing)
MESH_TE_STRIP && @printf("STRIP BLOCK: carrier %d x %d cells behind the base (n_base %d per half; outer tags OUTER_STRIP_INTERFACE), configured as the C block's; ambient p_ref=%.17g (C block %.17g)\n",
                         s_strip.NI, s_strip.NJ, MESH_TE_BASE, Float64(tem_ambient(tem_strip, s_strip)[1]), Float64(p_ref))
# ★ TOPO_INIT=panel (2026-09-14, validation ladder tier 2b). Replace the uniform initial field by the confined potential
# flow before the EOS projection: velocity from the panel solution with the wall image system, density from the
# Bernoulli pressure about the ambient through the local EOS inverse (liquid branch; cells that would fall below the
# saturation pressure are held there and counted). The ghosts are refilled from the new interior with the solver's own
# boundary law, since the species builder checks every ghost against its donor.
"""The panel start's interior fill of one carrier (TOPO_INIT=panel; with the strip topology, both blocks, plan section 4.5):
the velocity of the potential field `pf` (evaluated at this carrier's centroids) and the density of the Bernoulli
pressure about the ambient `pr` through the liquid-branch EOS inverse of the carrier's TEM `tq`, over the interior only
(the ghosts are left to the caller); a cell below the saturation pressure is held there and counted. Returns
`(clamped cells, p_min, max |u|/U)`."""
function movie_panel_fill!(q, tq, pf, pr, qq)
    g = q.g
    r = Array(q.rho); ux = Array(q.ux); uy = Array(q.uy); nclamp = 0; pmin = Inf; umax = 0.0
    pfloor = Float64(tq.p_v) + Float64(tq.psm)      # liquid branch only: no vapour in the initial field
    for j in 1:q.NJ, i in 1:q.NI
        pb = pr + qq*(1.0 - (pf.u[i, j]^2 + pf.v[i, j]^2))
        pmin = min(pmin, pb); pb < pfloor && (nclamp += 1; pb = pfloor)
        r[i+g, j+g] = CMeshCavTEM.CMeshEOSMix._rhomix_vl(Float32(pb), 0.0f0, tq.p_v, tq.a2l, tq.rho_l_sat, tq.rho_v_sat, tq.psm, tq.lq)
        ux[i+g, j+g] = Float32(U*pf.u[i, j]); uy[i+g, j+g] = Float32(U*pf.v[i, j])
        umax = max(umax, hypot(pf.u[i, j], pf.v[i, j]))
    end
    all(isfinite, r) && all(>(0.0f0), r) || error("panel initial field: nonpositive or nonfinite density")
    copyto!(q.rho, r); copyto!(q.ux, ux); copyto!(q.uy, uy)
    (nclamp, pmin, umax)
end
if INIT_PANEL
    RESTART_SET && error("TOPO_INIT=panel is incompatible with CAV_MOVIE_RESTART: the checkpoint state is authoritative")
    mesh isa ChannelMesh || error("TOPO_INIT=panel requires TOPO_CHANNEL_H > 0 (the wall image system)")
    CONSERVATIVE || error("TOPO_INIT=panel requires TOPO_SPECIES=1")
    # A blunt section (TOPO_MESH_TE_BASE > 0) is closed through its base midpoint (PanelInit.blunt_panel_loop, 2 Oct
    # 2026): a solid base of n_base panels per half (the mesh's own base resolution; 2 x 8 panels on an n_base 16 mesh
    # was measured to reach -6.14 q with a vapour trace) with the Kutta node at the midpoint. The solver refuses an open
    # loop (closing it with one base panel puts the Kutta node at the lower corner: C_L 2.79 against 0.64).
    MESH_TE_BASE > 0 && @printf("PANEL BLUNT TRAILING EDGE: loop closed through the base midpoint (%.7f, %.7f) by 2 x %d base panels; Kutta node at the midpoint\n",
                               0.5*(foilpts.x[1] + foilpts.x[end]), 0.5*(foilpts.y[1] + foilpts.y[end]), MESH_TE_BASE)
    let sol = MESH_TE_BASE > 0 ?
            PanelInit.solve_confined_panels(PanelInit.blunt_panel_loop(foilpts.x, foilpts.y; n_half=MESH_TE_BASE)...; alpha_deg=ALPHA, H=CHANNEL_H/2, M=60) :
            PanelInit.solve_confined_panels(foilpts.x, foilpts.y; alpha_deg=ALPHA, H=CHANNEL_H/2, M=60),
        pf = PanelInit.potential_field(sol, met.xc, met.yc), qq = 0.5*Float64(s.rho_inf)*U^2
        nclamp, pmin, umax = movie_panel_fill!(s, tem, pf, p_ref, qq)
        CMeshSolverKA.applyBC_ka!(s, Int32(2))
        @printf("PANEL INITIAL FIELD: alpha=%.3f H=%.4f images=%d panels=%d Cl_confined=%.4f Gamma=%.4f wall_residual=%.2e umax/U=%.3f p_min/q=%.3f clamped_cells=%d
",
                ALPHA, CHANNEL_H/2, 60, length(sol.S), sol.Cl, sol.Gamma, PanelInit.wall_residual(sol), umax, (pmin-p_ref)/qq, nclamp)
        if MESH_TE_STRIP
            # The strip block (plan section 4.5): the same closed-loop solution evaluated at the strip's centroids fills its
            # interior; its ghosts are filled by the pair's build sequence (applyBC_ka! is never called on the strip
            # carrier). The tension refusal counts both blocks.
            pfs = PanelInit.potential_field(sol, met_strip.xc, met_strip.yc)
            nclamp_s, pmin_s, umax_s = movie_panel_fill!(s_strip, tem_strip, pfs, p_ref, qq)
            @printf("PANEL INITIAL FIELD (strip block): cells=%dx%d umax/U=%.3f p_min/q=%.3f clamped_cells=%d (the same panel solution at the strip's centroids)\n",
                    s_strip.NI, s_strip.NJ, umax_s, (pmin_s-p_ref)/qq, nclamp_s)
            nclamp + nclamp_s > 0 &&
                error("strip panel start asks for tension in $nclamp C block and $nclamp_s strip cells (p_min/q $((min(pmin, pmin_s)-p_ref)/qq)); refused")
        end
        # A blunt start that asks for tension is refused (2 Oct 2026): the clamp would hide it while the base corners
        # evacuate in the first steps (measured on the stage-0 meshes).
        MESH_TE_BASE > 0 && nclamp > 0 && error("blunt panel start asks for tension in $nclamp cells (p_min/q $((pmin-p_ref)/qq)); refused")
    end
end
if OUTLET_PRESSURE_SET
    # Freshly built dtf is zero: project the initial pure-liquid state onto the
    # mixture EOS without advancing alpha. The characteristic outlet needs its
    # physical pressure/sound speed before the very first flow stage.
    if CONSERVATIVE
        tem_project!(s,tem;defer_pressure_outlet=true)
    else
        tem_step!(s, tem)
    end
    set_pressure_outlet!(s, tem; p_out=OUTLET_PRESSURE_RAW == "ambient" ? p_ref : OUTLET_PRESSURE)
    @printf("  REALISED OUTLET: fixed static pressure p_out=%.9g sigma_exit=%.9g; reflected acoustics; cav=%d\n",
            s.outlet_p, (Float64(s.outlet_p)-Float64(tem.p_v))/(0.5*Float64(s.rho_inf)*U^2), Int(s.cav))
    if MESH_TE_STRIP
        # The strip carrier (plan section 3.2): its own EOS projection and the same exit pressure on its i = NI+1 side,
        # the strip's only outlet (its i = 1 side is the base wall; the pair's trio carries the sides mask 2)
        tem_project!(s_strip, tem_strip; defer_pressure_outlet=true)
        set_pressure_outlet!(s_strip, tem_strip; p_out=OUTLET_PRESSURE_RAW == "ambient" ? p_ref : OUTLET_PRESSURE)
        @printf("  REALISED OUTLET (strip block): fixed static pressure p_out=%.9g on its downstream side only (C block %.9g)\n",
                s_strip.outlet_p, s.outlet_p)
    end
end
# TOPO_SST_OMEGA_INIT=wall (2 Oct 2026): the omega start from the wall distance, after the start field and before the
# species state is built from it (see the knob's comment). The wall distance is the metric's own (minimum over every
# wall segment, the base faces included). With the strip topology both blocks, each from its own wall distance (the
# strip's to the base, the C block's to the surface and the base segment; plan section 4.5).
"The wall-distance omega start of one carrier `q` from its metrics' `m.wdist`; returns (cells raised, largest omega)."
function movie_omega_wall!(q, m)
    g = q.g; tom = Array(q.tom); tk = Array(q.tk); nut = Array(q.nut); nu = Float64(q.nu_lam); nraise = 0; wmax = 0.0
    for j in 1:q.NJ, i in 1:q.NI
        ww = 6.0*nu/(0.075*Float64(m.wdist[i, j])^2)
        if ww > tom[i+g, j+g]
            tom[i+g, j+g] = Float32(ww); nut[i+g, j+g] = tk[i+g, j+g]/tom[i+g, j+g]; nraise += 1
        end
        wmax = max(wmax, Float64(tom[i+g, j+g]))
    end
    copyto!(q.tom, tom); copyto!(q.nut, nut)
    (nraise, wmax)
end
if SST_OMEGA_INIT == "wall"
    let (nraise, wmax) = movie_omega_wall!(s, met)
        @printf("SST OMEGA START: wall (omega = max(w_inf, 6 nu/(0.075 d^2)), nu_t = k/omega) raised in %d of %d cells; max omega %.4g (w_inf %.4g)\n",
                nraise, s.NI*s.NJ, wmax, Float64(s.w_inf))
    end
    if MESH_TE_STRIP
        let (nraise, wmax) = movie_omega_wall!(s_strip, met_strip)
            @printf("SST OMEGA START (strip block): wall, from the strip's distance to the base, raised in %d of %d cells; max omega %.4g\n",
                    nraise, s_strip.NI*s_strip.NJ, wmax)
        end
    end
end
# Bind the channel-only discrete ambient reference to the fully configured solver.  The seed uses
# scratch fields, so it neither primes TEM nor changes the physical F/W/C initial condition.
mesh isa ChannelMesh && seed_channel_reference_ka!(s)
const restart_metadata = Ref{Any}(nothing)
# The strip topology (plan sections 3.1 and 4.7): the strip's carrier and TEM bound to the C block's cache, so the build
# runs the pair's sequence and the restart restores a CMeshSpeciesRestart/3 file into both; nothing otherwise (the
# keyword's default: every other configuration builds and restores as before).
const STRIP_CARRIERS = MESH_TE_STRIP ? (solver=s_strip, tem=tem_strip) : nothing
species = if CONSERVATIVE
    CMeshSolverKA._mixture_pressure_bc!(s,similar(s.rho);conservative_sst=s.sst)
    cache=if RESTART_SET
        restored=restore_cmesh_species_restart(s,tem,RESTART_PATH;wallmode=Int32(2),reconstruction=SPECIES_RECON,kinetic=SPECIES_KFVS,
                                               eos_precision=SPECIES_EOS,source_weight=SPECIES_SOURCE_WEIGHT,nut_density=SPECIES_NUT_DENSITY,reboud_n=SPECIES_REBOUD_N,
                                               reboud_production=SPECIES_REBOUD_PRODUCTION,source_model=SPECIES_SOURCE_MODEL,ss_n0=SPECIES_SOURCE_SETTING.n0,ss_dnuc=SPECIES_SOURCE_SETTING.dnuc,
                                               trip_x=SPECIES_TRIP_X,trip_width=SPECIES_TRIP_WIDTH,reduction_tile=SPECIES_REDUCTION_TILE,
                                               sst_correction_limit=SPECIES_SST_CLIMIT,stage_clock_margin=SPECIES_STAGE_MARGIN,recon_retreat=SPECIES_RECON_RETREAT,strip=STRIP_CARRIERS)
        fixed_outer_restart_guard(restored.metadata,FIXED_OUTER_PROVENANCE)
        te_base_restart_guard(restored.metadata,MESH_TE_SETTINGS)    # trailing-edge mesh settings (2 Oct 2026)
        mesh_quality_restart_guard(restored.metadata,MESH_QUALITY_TAG_STR)  # the mesh quality option (15 Oct 2026)
        restart_metadata[]=restored.metadata
        @printf("CONSERVATIVE RESTART: file=%s stored_steps=%d stored_elapsed_lattice=%.17g new_scored_segment=true warmup_skipped=true\n",
                RESTART_PATH,restored.cache.steps,restored.cache.elapsed)
        restored.cache
    else
        # The builder publishes pressure from its authoritative conserved state.
        build_species_ka(s,tem;wallmode=Int32(2),reconstruction=SPECIES_RECON,kinetic=SPECIES_KFVS,eos_precision=SPECIES_EOS,source_weight=SPECIES_SOURCE_WEIGHT,nut_density=SPECIES_NUT_DENSITY,reboud_n=SPECIES_REBOUD_N,
                         reboud_production=SPECIES_REBOUD_PRODUCTION,source_model=SPECIES_SOURCE_MODEL,ss_n0=SPECIES_SOURCE_SETTING.n0,ss_dnuc=SPECIES_SOURCE_SETTING.dnuc,
                         trip_x=SPECIES_TRIP_X,trip_width=SPECIES_TRIP_WIDTH,reduction_tile=SPECIES_REDUCTION_TILE,
                         sst_correction_limit=SPECIES_SST_CLIMIT,stage_clock_margin=SPECIES_STAGE_MARGIN,recon_retreat=SPECIES_RECON_RETREAT,strip=STRIP_CARRIERS)
    end
    cache
else
    nothing
end
MESH_TE_STRIP && @printf("SPECIES STRIP PAIR: C block %dx%d (NIW %d, NIF %d), strip %dx%d; configuration tag %s; pair geometry %s; checkpoint %s\n",
                         s.NI, s.NJ, s.NIW, s.NIF, s_strip.NI, s_strip.NJ, string(species.configuration[end]),
                         species.geometry_sha256[1:16], CMeshCavSpecies.STRIP_CHECKPOINT_FORMAT)
# The section's loads: with the strip topology the C block's surface (loads_ka) plus the strip's base (loads_strip_ka, its
# column-1 faces in loads_ka's convention, plan section 4.1), so the main CSV's Cl and Cd keep their meaning.
movie_loads() = CONSERVATIVE ? (MESH_TE_STRIP ?
    loads_ka(s;mixture_pressure=species.pressure,wallmode=Int32(2)) .+ CMeshCavSpecies.loads_strip_ka(species.strip) :
    loads_ka(s;mixture_pressure=species.pressure,wallmode=Int32(2))) : loads_ka(s)
const species_executor = SPECIES_GRAPH ? CMeshSpeciesCUDAGraph.SpeciesGraphExecutor(species) : nothing
CONSERVATIVE && println("  SPECIES EXECUTION: graph=",SPECIES_GRAPH," rejection_replay=",SPECIES_REPLAY," reduction_tile=",SPECIES_REDUCTION_TILE)
# First-cell y+ under the all-y+ wall function (26 Sep 2026), from the same law the flux kernel applies: printed with
# the frame line so the mesh's fitness for a log-layer wall is visible in every log. Returns min, median, max and the
# fraction of wall faces below y+ 30.
function wall_yplus_summary(UX, UY)
    yp = Float64[]
    for i in s.NIW+1+BASE_IN_FOIL:s.NIW+s.NIF-BASE_IN_FOIL        # surface faces (the trailing-edge base excluded)
        Sf = hypot(Float64(s.Sjx1_h[i]), Float64(s.Sjy1_h[i])); d1 = 0.5*Float64(s.vol1_h[i])/Sf + 1e-30
        nx = Float64(s.Sjx1_h[i])/Sf; ny = Float64(s.Sjy1_h[i])/Sf
        u = Float64(UX[i+s.g, s.g+1]); v = Float64(UY[i+s.g, s.g+1]); un = u*nx + v*ny
        _, y, _ = CMeshSolverKA.wf_species_state(hypot(u - un*nx, v - un*ny), d1, Float64(s.nu_lam))
        push!(yp, y)
    end
    sort!(yp)
    (yp[1], yp[cld(length(yp), 2)], yp[end], count(<(30.0), yp)/length(yp))
end
const species_retry_count = Ref(let m=restart_metadata[]
    n=m isa NamedTuple && hasproperty(m,:total_rejected_attempts) ? m.total_rejected_attempts : 0
    n isa Int && n>=0 || error("invalid restart retry count")
    n
end)
const species_rejection_reasons = Dict{String,Int}()   # this process's tally only (not carried by a restart)
function movie_checkpoint(path;scored_elapsed=0.0,scored_steps=0)
    metadata=(driver="cmesh_cav_movie.jl",tag=tag,
        output_directory=OUTD,parent_checkpoint=RESTART_PATH,requested_ctu=TCTU,
        scored_elapsed=scored_elapsed,scored_steps=scored_steps,total_rejected_attempts=species_retry_count[])
    FIXED_OUTER_PROVENANCE === nothing || (metadata=merge(metadata,(fixed_outer=FIXED_OUTER_PROVENANCE,)))
    IMEX && (metadata=merge(metadata,(integrator=IMEX_SOURCE_RAW=="implicit" ? "line_imex_w3a_implicit_source" : "line_imex_w3a",)))
    MESH_TE_SETTINGS === nothing || (metadata=merge(metadata,(te_base=MESH_TE_SETTINGS,)))
    MESH_QUALITY === nothing || (metadata=merge(metadata,(mesh_quality=MESH_QUALITY_TAG_STR,)))
    SST_OMEGA_INIT == "wall" && (metadata=merge(metadata,(sst_omega_start="wall",)))
    write_cmesh_species_restart(path,species;metadata)
end
const imex_ws = IMEX ? build_species_imex(species;source=Symbol(IMEX_SOURCE_RAW)) : nothing
function movie_species_step!(max_dt::Float32=Inf32)
    if IMEX
        step_species_imex_ka!(species,imex_ws,Float32(CFLV);max_dt)
    else
    step_species_ka!(species,Float32(CFLV);max_dt,admission=SPECIES_ADMISSION,executor=species_executor)
    end
    species_retry_count[]+=species.last_retries
    # Rejection reasons (1 Oct 2026, the NACA 0012 NA10 storm): each accepted step that needed retries carries the reason
    # of its last rejection; the tally is printed beside the progress line, and only when a rejection has occurred, so a
    # run without rejections prints exactly what it printed before.
    if species.last_retries>0 && species.last_rejection!==nothing
        r=species.last_rejection
        key=string(r.reason,"/stage",r.stage)
        species_rejection_reasons[key]=get(species_rejection_reasons,key,0)+species.last_retries
    end
    if species.steps==1 || species.steps%1000==0
        @printf("CONSERVATIVE PROGRESS: accepted_steps_total=%d elapsed_solver_ctu=%.9g rejected_attempts_total=%d last_dt=%.9g\n",
                species.steps,species.elapsed*U,species_retry_count[],s.last_global_dt)
        isempty(species_rejection_reasons) ||
            println("CONSERVATIVE REJECTION REASONS (last rejection of each retried step, weighted by its retries): ",
                    join([string(k,"=",species_rejection_reasons[k]) for k in sort(collect(keys(species_rejection_reasons)))], " "))
        flush(stdout)
    end
end
q = 0.5*Float64(s.rho_inf)*U^2

@printf("CAVITATING MOVIE  sigma=%.2f T=%.1f CTU mesh=%dx%d dn1=%.1e warm=%d frames=%d\n",
        SIGMA, TCTU, MESH[1], MESH[2], MESH[3], WARM_STEPS, NFRAME)
if mesh isa ChannelMesh
    @printf("  REALISED GEOMETRY: foil=%s n_foil=%d n_normal=%d dn1=%.2e growth=%.3f smooth=%d channel_H=%.8g xi_in=%.8g xi_out=%.8g angle=%.4f outer[inlet/lower/upper]=%d/%d/%d\n",
            FOIL, MESH[1], MESH[2], MESH[3], GROWTH, SMOOTH, mesh.channel.height,
            mesh.channel.xi_in, mesh.channel.xi_out, ALPHA,
            count(==(OUTER_INLET), mesh.outer_face_kind),
            count(==(OUTER_LOWER_WALL), mesh.outer_face_kind),
            count(==(OUTER_UPPER_WALL), mesh.outer_face_kind))
    # The independent leading-edge spacing (W5) is printed only when set, so default banners are unchanged; the wall
    # instruments rebuild the mesh from this line (cmesh_wall_diagnostics.jl, wall_mesh_options).
    MESH_LE_DS > 0 && @printf("  REALISED LEADING-EDGE SPACING: le_ds=%.6g (arc length, chord units)\n", MESH_LE_DS)
    # Trailing-edge mesh settings (2 Oct 2026), printed only when a trailing-edge knob is set: the generator keywords
    # (exact, round-trip) before " | ", then the realised gap, corners and spacings. The wall instruments rebuild the
    # mesh from the settings half; REALISED GEOMETRY's n_foil stays the surface count (NIF = n_foil + 2 n_base).
    # Readings (stage-1 DECISION.md 4.6): with a base, the generator's construction, its version and gap-dependent depths
    # (CMeshGen.te_base_construction), so a later construction change reads as a named mismatch, not a bare hash failure;
    # and the smallest corner sine over all cells and over the trailing-edge windows (30 columns either side of each
    # junction and corner). With only te_ds or the near wake set, the smallest corner sine alone.
    # Strip topology (5 Oct 2026, plan section 4.7): the settings carry te_topology=strip; the readings are the strip's
    # (te_base_realised with the strip block: the base from the strip's column 1, the strip's cell count, smallest corner
    # sine and skew, the legs' first-row ramp, the base tilt), the C block's smallest corner sine over all cells and over the
    # windows 30 columns either side of each corner (the junctions are the corners), and the construction named for the
    # topology (strip_v1; no depths).
    if MESH_TE_SETTINGS !== nothing
        let r = MESH_TE_STRIP ? te_base_realised(mesh.X, mesh.Y, mesh.NIW, mesh.NIF, MESH_TE_BASE; strip=mesh.strip) :
                                te_base_realised(mesh.X, mesh.Y, mesh.NIW, mesh.NIF, MESH_TE_BASE),
            (smin, sat) = CMeshGen.corner_sine_min(mesh.X, mesh.Y, 1:mesh.NI, 1:mesh.NJ),
            q = (min_sine=round(smin; sigdigits=7), min_sine_at=(sat[1] - mesh.NIW, sat[2]))
            if MESH_TE_STRIP
                lo = max(1, mesh.NIW - 29):min(mesh.NI, mesh.NIW + 30)
                hi = max(1, mesh.NIW + mesh.NIF - 29):min(mesh.NI, mesh.NIW + mesh.NIF + 30)
                ste = min(CMeshGen.corner_sine_min(mesh.X, mesh.Y, lo, 1:mesh.NJ)[1],
                          CMeshGen.corner_sine_min(mesh.X, mesh.Y, hi, 1:mesh.NJ)[1])
                c = CMeshGen.te_base_construction(r.gap; topology=:strip)
                q = merge(q, (min_sine_te=round(ste; sigdigits=7), construction=c.construction))
            elseif MESH_TE_BASE > 0
                lo = max(1, mesh.NIW - 29):min(mesh.NI, mesh.NIW + MESH_TE_BASE + 30)
                hi = max(1, mesh.NIW + mesh.NIF - MESH_TE_BASE - 29):min(mesh.NI, mesh.NIW + mesh.NIF + 30)
                ste = min(CMeshGen.corner_sine_min(mesh.X, mesh.Y, lo, 1:mesh.NJ)[1],
                          CMeshGen.corner_sine_min(mesh.X, mesh.Y, hi, 1:mesh.NJ)[1])
                c = CMeshGen.te_base_construction(r.gap)
                q = merge(q, (min_sine_te=round(ste; sigdigits=7), construction=c.construction,
                              anchor_depth=round(c.anchor_depth; sigdigits=7), freeze_depth=round(c.freeze_depth; sigdigits=7),
                              ramp_depth=round(c.ramp_depth; sigdigits=7)))
            end
            println("  ", te_base_receipt_line(MESH_TE_SETTINGS, merge(r, q)))
        end
    end
    MESH_TE_SETTINGS === nothing || MESH_TE_STRIP || mesh.NIF == MESH[1] + 2MESH_TE_BASE ||
        error("realised foil range $(mesh.NIF) differs from n_foil + 2 n_base = $(MESH[1] + 2MESH_TE_BASE)")
    # the strip topology: the foil range is the surface alone (the base is the strip block's column 1)
    MESH_TE_STRIP && !(mesh.NIF == MESH[1] && mesh.strip.NI == mesh.NIW && mesh.strip.NJ == 2MESH_TE_BASE) &&
        error("realised strip-topology mesh differs from its settings: foil range $(mesh.NIF) (n_foil $(MESH[1])), strip $(mesh.strip.NI) x $(mesh.strip.NJ) (n_wake $(mesh.NIW) x 2 n_base $(2MESH_TE_BASE))")
else
    @printf("  REALISED GEOMETRY: foil=%s n_foil=%d n_normal=%d dn1=%.2e growth=%.3f smooth=%d R_ff=25\n",
            FOIL, MESH[1], MESH[2], MESH[3], GROWTH, SMOOTH)
end
if STEADY
    println("  STEADY PSEUDO-TIME MODE: local time steps (every cell on its own clock), a relaxation to the steady state.")
    println("  This run is NOT a physical-time march; nothing it writes is a transient or has a time axis.")
elseif IMEX
    println("  LINE-IMPLICIT global-dt RK3 (W3a): each stage linearly implicit along the j lines (first-order j-face Jacobians,")
    println("  frozen per step); explicit in i. Physical time, but NOT the explicit integrator: its accuracy is a registered bridge.")
else
println("  EXPLICIT global-dt RK3 intended: no dual time, no inner loop, no BDF2. Whether it IS")
println("  physical time is checked by the applied-step clock receipt and frame times below; coupled temporal order remains unverified.")
end
print_realised_state(tem, s, U; label="cav movie", extra=realised_config_pairs(tem, s))
CONSERVATIVE && SPECIES_SS &&
    println("            tau_evap/tau_cond above are the Merkle coefficient reference: NOT USED by the Schnerr-Sauer source (TOPO_SPECIES_SOURCE_MODEL=schnerr_sauer; see SPECIES SOURCE MODEL)")
cav_label(lv) = lv >= 4 ? "LOW-MACH MIXTURE FLUX: AUSM+-up on rho,u,p_mix,a_m" :
                lv == 3 ? "KFVS carrier + pexc + Layer C + negcomp floor" :
                lv == 2 ? "KFVS carrier + pexc + Layer C (archived production)" : "KFVS carrier + pexc"
@printf("            MACH=%.4g   warm-up cav=%d (%s)   warm-up omega=%.3f\n", MACH, Int(s.cav), cav_label(Int(s.cav)), Float64(tem.omega))
@printf("            explicit march CFL=%.4g on the operational clock (TOPO_CFL; archived production value 0.15)\n", CFLV)
@printf("            scored march: cav=%d (%s)   omega=%.3f\n", CAVLV, cav_label(CAVLV), MARCH_OMEGA); flush(stdout)
@printf("            turbulence closure=%s (keps=%.3g)   wall treatment=%s (wallfun=%.3g)   Larsson-PG=%s\n",
        !s.sst ? "laminar" : Float64(s.keps) > 0 ? "k-eps RNG" : "SST", Float64(s.keps),
        s.wallfun == 2.0f0 ? "all-y+ Reichardt wall function (species)" :
        Float64(s.wallfun) > 0 ? "log-law wall function" : "wall-resolved/no wall function", Float64(s.wallfun),
        Float64(s.wf_larsson) > 0 ? "ON" : "OFF")
println("  TRANSPORT: ",CONSERVATIVE ?
    "conservative rho*Y; physical SSPRK stages; " *
    (SPECIES_RECON ? "reconstructed interior flux; " : "PC spatial flux; ") *
    (SPECIES_KFVS ? "KINETIC KFVS faces (common gauge T*=1, half-stream vapor transport, kinetic clock); " :
                    "AUSM+-up faces (net-donor vapor transport); ") *
    "direct EOS pressure; resolved explicit phase change; " *
    (SPECIES_EOS==:single ? "single-precision EOS arithmetic on the Float64 state; " : "double-precision EOS arithmetic; ") *
    (SPECIES_ADMISSION==:deferred ? "deferred admission (device flags and accounting, one host read per attempt, classified retries)" :
                                    "immediate admission") : "legacy segregated flow/TEM/SST")
# The Reboud clause reports the REALISED knob. It used to be the constant "no Reboud correction", so a
# Reboud run wrote both its enabling line and this denial into one log (see T25's movie.log, 20 Sep).
CONSERVATIVE && println("  CONSERVATIVE MODEL: ",s.sst ? "fully turbulent SST-2003m; " *
    (SPECIES_TRIP_X != 0 ? @sprintf("prescribed trip ACTIVE (k raised to the inner-layer turbulent level at wall x/c %.4g-%.4g; tag CMeshSpeciesTripV1); ",
                                   SPECIES_TRIP_X, SPECIES_TRIP_X + SPECIES_TRIP_WIDTH) : "prescribed trip inactive; ") *
    (SPECIES_NUT_DENSITY==:reboud_v3 ?
        "Reboud nu_t correction ON, void-fraction form (n=$(SPECIES_REBOUD_N); tag CMeshSpeciesReboudNutV3)" :
     SPECIES_NUT_DENSITY==:reboud_v2 ?
        "Reboud nu_t correction ON, composition form (10 Oct) (n=$(SPECIES_REBOUD_N); tag CMeshSpeciesReboudNutV2)" :
     SPECIES_NUT_DENSITY==:reboud_v1 ?
        "Reboud nu_t correction ON, density form (20 Sep) (n=$(SPECIES_REBOUD_N); tag CMeshSpeciesReboudNutV1)" :
        "Reboud nu_t correction OFF (TOPO_SPECIES_NUT_DENSITY=none; nu_t bitwise the September path)") *
    (SPECIES_NUT_DENSITY!==:none ? (SPECIES_REBOUD_PRODUCTION ? ", SST k production from the corrected nu_t" :
                                                                ", SST k production from the uncorrected nu_t") : "") : "laminar",
    "; output beta is physical vapor volume, TEM alpha is reference composition")
if CONSERVATIVE && SPECIES_TRIP_X != 0
    trip_cols = CMeshCavSpecies.trip_columns(s, species.trip)
    println("  SPECIES TRIP: foil columns in the strip (lower, upper) = ", trip_cols, "; k raised in cells with y+ <= ",
        CMeshCavSpecies.TRIP_YPLUS_MAX, "; the added k is ledgered with the k floor (reported in the k source)")
    # the Caltech foil was tripped on both surfaces: a strip that misses one is a configuration error, not a choice
    minimum(trip_cols) >= 1 || error("TOPO_SPECIES_TRIP strip holds no foil column on one surface; widen TOPO_SPECIES_TRIP_WIDTH")
end
CONSERVATIVE && SPECIES_SST_CLIMIT >= 0 &&
    @printf("  SST DIFFUSION CORRECTION: limited, lambda=%.4g (the non-orthogonal correction of the k/omega diffusion bounded by lambda x the two-point flux; clock (1+lambda) x two-point on the corrected faces, two-point alone on the uncorrected ones; tag CMeshSpeciesSSTCorrectionLimitV2)\n",
            SPECIES_SST_CLIMIT)
CONSERVATIVE && SPECIES_STAGE_MARGIN_SET &&
    @printf("  SPECIES STAGE CLOCK MARGIN: %.4g x the stage-1 phase-source, viscous and SST clocks in the step's proposal (TOPO_SPECIES_STAGE_CLOCK_MARGIN; the face cap keeps TOPO_SPECIES_CLOCK_MARGIN; tag CMeshSpeciesStageClockMarginV1/m=%s)\n",
            SPECIES_STAGE_MARGIN, string(Float32(SPECIES_STAGE_MARGIN)))
# The continuous phase-bound retreat (15 Oct 2026): printed from the built cache, so the line states what the solver binds.
if CONSERVATIVE && SPECIES_RECON_RETREAT === :continuous
    CMeshCavSpecies._recon_retreat(species) == Int32(1) && CMeshCavSpecies.RECON_RETREAT_TAG in species.configuration ||
        error("TOPO_SPECIES_RECON_RETREAT=continuous was not bound by the species cache")
    println("  SPECIES RECONSTRUCTION RETREAT: continuous (TOPO_SPECIES_RECON_RETREAT; where a face's phase densities leave 0 <= q <= rho at the limiter's theta, ",
            "and the prevfloat step does not restore them, the largest common phase theta meeting every face's bounds, in closed form, comes before the halving; tag ",
            CMeshCavSpecies.RECON_RETREAT_TAG, ")")
end
CONSERVATIVE && println("  VAPOR VOLUME: main CSV Vvap integrates beta only where beta >= ",THETA,
    "; inventory CSV vapor_volume_total integrates beta over every interior cell")
flush(stdout)

g = s.g; NI, NJ, NIW, NIF = s.NI, s.NJ, met.NIW, met.NIF
xw = [Float64(Array(s.xc)[NIW+ii, 1]) for ii in 1:NIF]
ii_le = argmin(xw)
# A symmetric section (2026-09-14, NACA0012 of the validation ladder) puts the nose exactly between two wall cells
# of equal x; the diagnostics need a strictly increasing strip from the leading edge, so the index advances past
# a leading tie. A cambered section has an increasing successor at argmin and is unaffected.
while ii_le < NIF && xw[ii_le + 1] <= xw[ii_le]; global ii_le += 1; end
# Surface view of the foil range (2 Oct 2026, blunt trailing edge): the wall-row instruments below (the LE-to-TE strip,
# the interface and wall-C_p stations, Gate-B topology and the interior jet) walk the surface only; the trailing-edge
# base faces are read on their own (the base/<tag>_base.csv sidecar). At n_base = 0 the view is the full range: bitwise.
# With the strip topology the foil range is the surface already (BASE_IN_FOIL = 0) and the partition appends the strip's
# column-1 faces after it (lower half j_s = 1..n_base, upper half n_base+1..2 n_base; te_base_partition(...; strip=true)).
const NIW_S, NIF_S = surface_view(NIW, NIF, MESH_TE_BASE; strip=MESH_TE_STRIP)
const II_LE_S = ii_le - BASE_IN_FOIL
BASE_IN_FOIL < ii_le <= NIF - BASE_IN_FOIL || error("the leading-edge cell $ii_le lies on the trailing-edge base")
const TE_PART = te_base_partition(NIF, MESH_TE_BASE; strip=MESH_TE_STRIP)
R = build_raster(mesh, met, WIN, PXW)
@printf("raster %d x %d pixels over window (%.2f,%.2f)x(%.2f,%.2f)\n", R.PX, R.PY, WIN...); flush(stdout)
# The strip block's raster over the same window (strip topology, plan section 4.7): every frame is one image of both blocks
# (raster_gather_composite). The strip's raster gets an exact containment pass once (raster_inside!; 7 Oct 2026, the stage-3
# review gate's O1): every pixel whose centre lies inside one of the strip's cells is drawn from that cell, including those
# the C block's bounding-box fill also claims (the inclined leg cells beside the strip reach into it), and the strip's
# bounding-box fill keeps the pixels the C block's raster leaves empty, as before.
const R_STRIP = MESH_TE_STRIP ? build_raster(mesh.strip, met_strip, WIN, PXW) : nothing
const IN_STRIP = MESH_TE_STRIP ? raster_inside!(R_STRIP, mesh.strip, met_strip) : nothing
MESH_TE_STRIP && @printf("raster (strip block): %d pixel centres of the window inside the strip's cells, every one drawn from the strip (%d of them in the C block's bounding-box fill too); %d more pixels of the strip's bounding-box fill where the C block's raster has none, drawn from the strip\n",
                         count(IN_STRIP), count(k -> IN_STRIP[k] && R.PIX[k] != 0, eachindex(R.PIX)),
                         count(k -> !IN_STRIP[k] && R.PIX[k] == 0 && R_STRIP.PIX[k] != 0, eachindex(R.PIX)))
# Makie pixel map, built once (same window as the Plots frames). Cp is normalised on the realised
# far-field p_mix (= p_ref from tem_ambient) and the realised |U|, both computed from the state above.
const PM = MAKIE ? build_pixmap(mesh.X, mesh.Y, Int(met.NI), Int(met.NJ); win=WIN, PX=PXW) : nothing
MAKIE && println("makie: field renders every $(MAKIE_EVERY) frames (Cp on p_ref=$(p_ref), U=$(U))")

# ---- PHASE 1: warm-up on LTS (a relaxation to a developed cavity; time-accuracy is irrelevant here)
# ⛔⛔⛔ TIME-ACCURACY REPAIR 2026-08-03. THIS TOOL NEVER SET `GLOBAL_DT` AND NEVER HAD, while
# printing (line ~163) "EXPLICIT global-dt RK3: no dual time, no inner loop, no BDF2 -- this IS
# physical time." `GLOBAL_DT` defaults to 0 (cmesh_solver_ka.jl:20) = LOCAL TIME STEPPING, and a
# tree-wide grep found it ASSIGNED IN EXACTLY TWO PLACES, both in cmesh_explicit_march.jl (:195,
# :224). So every "movie" record this campaign scored was marched on LTS -- every cell on its own
# timestep -- and then had its time axis labelled with min(dtf)*U as if that were a uniform dt.
#
# WHY THIS IS NOT A COSMETIC BUG: LTS is a STEADY-STATE CONVERGENCE ACCELERATOR. It cannot carry a
# physical transient; driving a solution monotonically to a fixed point is precisely what it is
# built to do. S5 -- the Frikha replication -- was scored FROZEN with the reading "V(t) is a
# MONOTONE RELAXATION to a fixed point, not a damped oscillation ... there is no oscillator."
# That is the defining behaviour of the integrator it was run on, not a property of the physics.
# ⇒ EVERY movie-based dynamics verdict is VOID and must be re-run. The banner asserting physical
# time is what made this invisible: the campaign's "print the realised state" rule was satisfied
# by a print that was false.
CMeshSolverKA.GLOBAL_DT[] = CONSERVATIVE ? 1 : 0
for n in 1:WARM_STEPS
    if CONSERVATIVE
        movie_species_step!()
    else
        step_ka!(s, Float32(CFLV); wallmode=Int32(2)); tem_step!(s, tem; dtp=0.0f0)
        sst_step_ka!(s; reboud_n=10.0f0, reboud_rhov=Float32(tem.rho_v_sat))
    end
end
KernelAbstractions.synchronize(backend)
if CONSERVATIVE
    @printf("CONSERVATIVE WARMUP CLOCK RECEIPT: steps=%d elapsed_lattice=%.17g elapsed_ctu=%.17g scored_clock_origin=after_warmup\n",
            species.steps,species.elapsed,species.elapsed*U)
end
if WARM_STEPS > 0 && CONSERVATIVE
    println("WARMUP RESIDUAL RECEIPT: iterations=$WARM_STEPS stage=NONE available=false (the species march does not store the carrier divergence; rebuild 22 Sep 2026)")
elseif WARM_STEPS > 0
    # dc_r is the last flow RK stage's stored integrated mass residual. TEM and
    # SST have since advanced; this read-only diagnostic is not a freshly
    # recomputed residual of the final coupled state or a convergence certificate.
    wr_linf, wr_rms, wr_frac, wr_i, wr_j = CMeshSolverKA.residual_report_ka(s)
    @printf("WARMUP RESIDUAL RECEIPT: iterations=%d stage=LAST_FLOW_RK3 stored_mass_over_volume_linf=%.9g stored_mass_over_volume_cell_rms=%.9g fraction_ge_0p99max=%.9g imax=%d jmax=%d recomputed_final_coupled=false\n",
            WARM_STEPS, wr_linf, wr_rms, wr_frac, wr_i, wr_j)
else
    println("WARMUP RESIDUAL RECEIPT: iterations=0 stage=NONE available=false")
end
let (Cl, Cd) = movie_loads()
    @printf("after %s warm-up (%d iters): Cl=%.4f Cd=%.5f amax=%.3f\n",
            CONSERVATIVE ? "global physical" : "LTS", WARM_STEPS, Float64(Cl), Float64(Cd), Float64(maximum(Array(tem.alpha))))
    # ★ TIER 0 BOUNDEDNESS RECEIPT (2026-08-05). alpha is bounded in [0,1]. A higher-order face
    # reconstruction can overshoot and the existing clamp would HIDE it, producing an interface that
    # looks sharper only because it is unphysical. That is the one way the Tier 0 fix silently breaks
    # the model, and nothing reported it before. Must stay at round-off.
    let a = Array(tem.alpha), ov = 0.0f0, un = 0.0f0
        for x in a
            x > 1.0f0 && (ov = max(ov, x - 1.0f0))
            x < 0.0f0 && (un = min(un, x))
        end
        @printf("            alpha bounds: max over 1 = %.3e   min under 0 = %.3e   (arecon=%d)\n",
                Float64(ov), Float64(un), ARECONV)
        (ov > 1.0f-5 || un < -1.0f-5) && error("alpha left [0,1] by more than 1e-5 under arecon=" *
            string(ARECONV) * " -- the reconstruction is UNBOUNDED and the clamp was hiding it.")
    end
    # ★ TIER 4 GROUNDWORK — MEASURE THE CLAMP, DO NOT ARGUE ABOUT IT (2026-08-05).
    # The energy equation is destroyed by T = clamp(p/(rho*cs2), Tmin, Tmax): temperature is half a
    # pressure representation, and latent heat has measured EXACTLY inert three separate times because
    # the clamp erases evaporative cooling. That is asserted throughout the campaign record and has
    # never been MEASURED on a production state. This receipt does it from the tool side, needing no
    # solver edit: recompute the unclamped ratio and report how much of the field the clamp is moving.
    # Tier 4 (two-population energy) cannot be scoped honestly until this number is known -- if the
    # clamp is inactive on most of the field, the energy work is lower priority than the record claims.
    let Ta = Array(s.T), ra = Array(s.rho), pe = Array(s.pexc), cs2 = Float64(2/3),
        nlo = 0, nhi = 0, ntot = 0, worst = 0.0
        for i in eachindex(Ta)
            r = Float64(ra[i]); r <= 0 && continue
            praw = r*cs2*Float64(Ta[i]) + Float64(pe[i])      # the mixture pressure actually carried
            traw = praw/(r*cs2)                                # what T would be with no clamp
            ntot += 1
            traw < Float64(tem.Tmin) && (nlo += 1)
            traw > Float64(tem.Tmax) && (nhi += 1)
            d = traw < Float64(tem.Tmin) ? Float64(tem.Tmin) - traw :
                (traw > Float64(tem.Tmax) ? traw - Float64(tem.Tmax) : 0.0)
            d > worst && (worst = d)
        end
        @printf("            T-CLAMP ACTIVITY: %.2f%% of cells at the Tmin=%.2f rail, %.2f%% at Tmax=%.2f, worst excursion %.4g\n",
                100*nlo/max(ntot,1), Float64(tem.Tmin), 100*nhi/max(ntot,1), Float64(tem.Tmax), worst)
        # The realised-state banner above was printed BEFORE the warm-up (field T = T_inf = 1 there);
        # the speeds that matter are the DEVELOPED field's, so print them again from this state.
        let g0 = Int(s.g), sub = @view(Ta[(g0+1):(g0+s.NI), (g0+1):(g0+s.NJ)]), Tm = Float64(sum(sub))/length(sub)
            @printf("            POST-WARMUP reference scales at the field mean T=%.4f: c_carrier/U=%.2f (3*sqrt(T)/U)  c_hydro/U=%.2f (sqrt(gamma*cs2*T)/U)  c_thermal/U=%.2f (sqrt(cs2*T)/U, carrier thermal reference)\n",
                    Tm, 3.0*sqrt(Tm)/U, sqrt(1.4*cs2*Tm)/U, sqrt(cs2*Tm)/U)
        end
        (nlo + nhi) == 0 && println("            (clamp INACTIVE on this state -- latent-heat inertness is NOT the clamp here)")
    end
    # A closure label is only a requested configuration. The realised eddy viscosity proves whether
    # that closure actually engaged; a prior k-eps run silently stayed laminar after epsilon was
    # initialised as omega. Keep this next to the scored record so the run is self-auditing.
    let nutc = Array(s.nut), nl = Float64(s.nu_lam), g0 = s.g
        sub = @view nutc[(g0+1):(g0+s.NI), (g0+1):(g0+s.NJ)]
        fw = @view nutc[(g0+s.NIW+1+BASE_IN_FOIL):(g0+s.NIW+s.NIF-BASE_IN_FOIL), (g0+1):(g0+min(8, s.NJ))]   # surface columns
        mx = maximum(sub); mn = sum(sub)/length(sub); fwmx = maximum(fw)
        @printf("            nu_t/nu: max=%.4g mean=%.4g foil-near-wall max=%.4g  (nu=%.3e, closure=%s)\n",
                mx/nl, mn/nl, fwmx/nl, nl, Float64(s.keps) > 0 ? "k-eps RNG" : "SST")
        (mx/nl < 1.0) && println("            ⚠⚠ nu_t/nu < 1 everywhere on this state; eddy viscosity is below laminar viscosity.")
    end
    # The strip block (strip topology, 5 Oct 2026): the lines above cover the C block; this one gives the strip's share of the
    # loads (its base, included in Cl and Cd above; on the absolute pressure, as loads_ka integrates the surface, so it closes
    # the section with the surface's share and is no force by itself: the base pressure drag is the base sidecar's gauge
    # reading), its alpha excursions (bounded as the C block's, the same refusal), its T-clamp activity and its eddy
    # viscosity, over its own NIW x NS cells.
    if MESH_TE_STRIP
        let a = Array(tem_strip.alpha), ov = 0.0f0, un = 0.0f0, (Cls, Cds) = CMeshCavSpecies.loads_strip_ka(species.strip),
            Ta = Array(s_strip.T), ra = Array(s_strip.rho), pe = Array(s_strip.pexc), cs2 = Float64(2/3), nlo = 0, nhi = 0, ntot = 0,
            nutc = Array(s_strip.nut), nl = Float64(s_strip.nu_lam), g0 = s_strip.g
            for x in a
                x > 1.0f0 && (ov = max(ov, x - 1.0f0))
                x < 0.0f0 && (un = min(un, x))
            end
            for i in eachindex(Ta)
                r = Float64(ra[i]); r <= 0 && continue
                traw = (r*cs2*Float64(Ta[i]) + Float64(pe[i]))/(r*cs2)
                ntot += 1
                traw < Float64(tem_strip.Tmin) && (nlo += 1)
                traw > Float64(tem_strip.Tmax) && (nhi += 1)
            end
            sub = @view nutc[(g0+1):(g0+s_strip.NI), (g0+1):(g0+s_strip.NJ)]
            @printf("            STRIP BLOCK: base share of the loads Cl=%.6f Cd=%.7f (loads_strip_ka on absolute pressure, included in Cl/Cd above; with the surface's share it closes the section); amax=%.3f alpha over 1 = %.3e under 0 = %.3e; T-CLAMP %.2f%% at Tmin, %.2f%% at Tmax; nu_t/nu max=%.4g mean=%.4g (%d x %d cells)\n",
                    Cls, Cds, Float64(maximum(a)), Float64(ov), Float64(un), 100*nlo/max(ntot,1), 100*nhi/max(ntot,1),
                    maximum(sub)/nl, (sum(sub)/length(sub))/nl, s_strip.NI, s_strip.NJ)
            (ov > 1.0f-5 || un < -1.0f-5) && error("alpha left [0,1] by more than 1e-5 in the strip block under arecon=" *
                string(ARECONV) * " -- the reconstruction is UNBOUNDED and the clamp was hiding it.")
        end
    end
    flush(stdout)
end
channel_diagnostics(s, tem, met, U; label=CONSERVATIVE ? "developed global-step state" : "developed LTS state",
                    mixture_pressure=CONSERVATIVE ? species.pressure : nothing)

# ---- PHASE 2: the scored march, ONE GLOBAL CLOCK (mirrors cmesh_explicit_march.jl:223-227)
CMeshSolverKA.GLOBAL_DT[] = 1
# ★★ ALPHA CLOCK (2026-09-03). The warm-up's Picard under-relaxation must not carry into the
# explicit march: omega scales the committed point-implicit void increment. Its numerical
# relaxation effect is not a measurement of the nonlinear physical response time.
tem.omega = Float32(MARCH_OMEGA)
@printf("★ ALPHA CLOCK: tem.omega=%.3f for the scored march (archived B29-B32 used 0.2 of the full point-implicit alpha increment)\n",
        Float64(tem.omega)); flush(stdout)
s.cav = Int32(CAVLV)
# the strip block's carrier and TEM follow the C block's (the species path requires omega 1 and cav 4 in both phases, so
# these are the values they hold; written so that the pair's scalars stay equal if that rule ever changes)
MESH_TE_STRIP && (tem_strip.omega = Float32(MARCH_OMEGA); s_strip.cav = Int32(CAVLV))
println("MARCH RK RECEIPT: incremental=", CONSERVATIVE || MARCH_RK_INCREMENTAL,
        " warmup_incremental=",CONSERVATIVE,
        CONSERVATIVE ? "; conserved state Float64; primitive caches Float32" : "; primitive recovery and Float32 storage remain")
print_realised_state(tem, s, U; label="cav movie marched configuration", extra=realised_config_pairs(tem, s))
@printf("★ CARRIER FLUX for the scored march: cav=%d (%s)\n", Int(s.cav), cav_label(Int(s.cav))); flush(stdout)
# ★★ CHANNEL REFERENCE RESEED (2026-09-05). `s.cav` is hashed into the channel residual-reference
# signature (cmesh_solver_ka.jl:311, checked in _validate_channel_ka! at :1658), so switching the
# carrier for the scored march leaves the seeded reference STALE and step_ka! throws
# "...is stale after a solver configuration change; reseed it". Without this line a confined run
# could only be made to work by warming up on the march carrier (TOPO_CAV_WARM == TOPO_CAV), and a
# cav=4 LTS warm-up DIVERGES on the production mesh (S0906 free-air control, 2026-09-05: NaN at
# 640k iterations). Reseeding here re-derives the discrete residual of the AMBIENT state under the
# march configuration -- it reads no developed field, so it cannot launder the solution -- and lets
# a confined run use the same warm-up protocol as every free-air arm in the archive.
if mesh isa ChannelMesh && !CONSERVATIVE
    seed_channel_reference_ka!(s)
    println("★ CHANNEL REFERENCE RESEEDED for the scored march carrier (cav=$(Int(s.cav)))"); flush(stdout)
end
if CONSERVATIVE
    # Preserve the restored/last accepted timestep. The Species step selects
    # its actual face/source/viscous clock; an estimate here is not an applied step.
    STEADY || @printf("CONSERVATIVE GLOBAL CLOCK: stored_steps=%d last_accepted_dt=%.9g; adaptive physical clock, %d equally spaced output targets\n",
            species.steps,s.last_global_dt,NFRAME)
else
    prepare_dt_ka!(s, Float32(CFLV))  # measure dt without an unpaired/unrecorded flow update
    KernelAbstractions.synchronize(backend)
    let dtf = Array(s.dtf), lo = minimum(dtf), hi = maximum(dtf)
        @printf("★ GLOBAL CLOCK ENGAGED: dtf spread hi/lo = %.6f (must be 1.000000 under global dt)\n", hi/lo)
        # ⛔ NaN GUARD (2026-08-05) -- see cmesh_explicit_march.jl. `NaN > x` is false, so a corrupt
        # state printed "GLOBAL CLOCK ENGAGED: dtf spread hi/lo = NaN" and the movie emitted a verdict.
        (isnan(hi/lo) || !isfinite(hi/lo)) && error("GLOBAL_DT=1 produced a NON-FINITE dtf spread " *
            "(hi=$hi lo=$lo) -- the mesh or state is corrupt. REFUSING to emit a dynamics verdict.")
        (hi/lo > 1.0 + 1e-6) && error("GLOBAL_DT=1 did not collapse dtf (hi/lo=$(hi/lo)) -- the march " *
                                      "would NOT be time-accurate; refusing to produce a dynamics verdict.")
    end

    dt_lat = Float64(minimum(Array(s.dtf))); dt_ctu = dt_lat*U
    @printf("initial global dt = %.4e lattice = %.4e CTU; adaptive physical clock, %d equally spaced output targets\n",
            dt_lat, dt_ctu, NFRAME)
end
flush(stdout)

# ---- the three rendered fields -----------------------------------------------------------------
# alpha: sequential.  ux: DIVERGING about 0 so reversal reads as colour.  Cp: diverging about -sigma.
const FIELDS = [("alpha", (0.0, 1.0),          :turbo,   CONSERVATIVE ? "physical vapor volume fraction" : "void fraction alpha"),
                ("ux",    (-1.0, 1.0),         :balance, "u_x / U   (BLUE = REVERSED)"),
                ("cp",    (-2.5, 1.0),         :RdBu,    "Cp")]

hist = NamedTuple[]; stmap = Vector{Float64}[]; stux = Vector{Float64}[]

# ★★ PER-FRAME JET DIAGNOSTICS ON THE HONEST CLOCK (2026-08-07).
#
# WHY. memory/project_docs/2026-08-07_E1b_JET_THICKNESS_IS_MEASURABLE_BUT_NOT_MESH_ROBUST.md found that at 512x192
# the wall-row reversed flow is DISCONNECTED -- present at x/c 0.051 and 0.100, absent for seven
# consecutive stations across the cavity, present again in the wake. A re-entrant jet is BY
# DEFINITION a layer travelling upstream FROM CLOSURE. A reversed patch at the leading edge with
# nothing joining it to closure is not one.
# That matters because this campaign's most-cited positive evidence is "a STRONG re-entrant jet
# (-0.530U reaching x/c = 0.049 -- FURTHER UPSTREAM than Leroux)" and x/c = 0.049 is exactly where
# that disconnected LE feature sits. If it is leading-edge separation rather than the terminus of a
# jet that crossed the cavity, then "a strong jet that cuts nothing" has a simpler explanation: no
# jet is arriving to cut.
# The steady probe cannot settle it -- its states are LTS endpoints and B27d showed the march has no
# fixed point. This tool records a historical wall-strip proximity proxy on the honest clock; it
# does not by itself establish that the vapour-run endpoint is the physical cavity closure.
#
# ⚠ FUNCTION-SCOPED DELIBERATELY. A `for` body at top level is a Julia SOFT scope and an accumulator
# assigned outside it becomes a fresh local on first assignment -- the trap that hit
# probe_cl_swing_stationarity.jl (2026-07-28), score_cavmovie_attachment_aware.jl (2026-08-06, where
# it crashed on any SHEDDING record) and probe_cavity_pressure.jl's jet family (2026-08-07).
# (The one-line docstring "contiguous runs of `mask`" that stood here documented a local `runs`
# helper that the 2026-08-11 refactor moved into tools/jet_connectivity_core.jl; left behind, it
# attached itself to the docstring below and made this file UNLOADABLE -- "cannot document the
# following expression" -- from that commit until the 2026-09-03 smoke test found it.)
"""
★ INTERIOR (LIFTED-JET) CONNECTIVITY -- the one caveat that could overturn the wall-row finding.

`jet_diag` below reads j = 1 only. B30 and B31 measured, over 551 frames at two cavitation numbers,
that NO frame has a wall reversed layer joining the first wall-vapour-run endpoint to the leading
edge. But a re-entrant jet
that LIFTS OFF the surface would be invisible to a wall-row metric, exactly as it is to every other
wall-row measure in this tree -- so "no jet on the wall" is not "no jet".

This attempts the missing measurement with a 4-connected flood fill over the suction-side foil block
(all j, not just j=1). A candidate component must approach that endpoint within 10% of the wall-cavity
length before its upstream reach is tested. B32 was halted before producing a CSV, so this remains a
mutation-tested instrument, not a validated lifted-jet result.

Costs one flood fill over NIF x NJ per frame -- negligible against the render, and it reuses `fu`,
which is already built for the u_x panel. Returns:
  n2d      number of DISCONNECTED reversed components in the interior
  x0_2d    upstream-most x/c reached by the component seeded near the wall-run endpoint
  conn2d   1 if that component reaches within 20% of the cavity length of the cavity start
  jmax2d   the highest wall-normal index that component occupies (how far it has lifted)
"""
function jet_diag_2d(fu, xc, NIW, ii_le, NIF, NJ, has_wall_cavity, cav_x1, x_start)
    interior_jet_diagnostics(fu, xc, NIW, ii_le, NIF, NJ,
                             has_wall_cavity, cav_x1, x_start)
end

"""
Per-frame wall-row jet diagnostics. `al`/`ux_wall`/`xs` are alpha, global x-velocity u_x/U and x/c along the wall
from LE to TE. Returns the first wall-vapour-run endpoint (legacy column `cav_x1`), reversed-flow
extent, number of DISCONNECTED reversed regions, and whether the nearest one reaches toward the LE.
"""
function jet_diag(al, ux_wall, xs, theta)
    wall_jet_diagnostics(al, ux_wall, xs, theta)
end
stline = [(NIW+ii, 1) for ii in ii_le:NIF-BASE_IN_FOIL]          # wall row from LE to TE (surface faces)
stx    = [xw[ii] for ii in ii_le:NIF-BASE_IN_FOIL]

# ★★ INTERFACE INSTRUMENT (2026-09-03). The primary metric of the pre-registered cav-4 experiment is
# the liquid/vapour interface WIDTH IN CELLS along wall-normal lines -- the quantity the KFVS contact
# leak controls and the one every wall-row scorer was blind to. Until today it could only be read by
# inverting the alpha PNG colours (critic's png_interface.py: 12-24 cells on B30). This computes it
# FROM THE STATE, per frame, at fixed suction-side stations, and writes a sidecar CSV beside the record
# (a separate file so the versioned 25-column record contract is untouched).
#   e_cells / e_c : contiguous alpha >= 0.5 layer from the wall (cells / wall-normal chord distance)
#   w_cells / w_c : first outward 0.9 -> 0.1 crossing region (cells / chord)  [the interface width]
#   a_wall, a_max, j_amax : wall alpha, line maximum and its cell index (lifted vapour => j_amax > e_cells)
const IFACE_XS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
const IFACE_ATH = 0.5
ycH0 = Array(s.yc)
iface_stations = let cand = [i for i in NIW+1+BASE_IN_FOIL:NIW+NIF-BASE_IN_FOIL if Float64(ycH0[i,1]) > 0]   # suction surface
    isempty(cand) && error("interface instrument: no suction-side wall cells found (yc > 0)")
    [cand[argmin(abs.([Float64(xcH0_i) for xcH0_i in Array(s.xc)[cand, 1]] .- xs))] for xs in IFACE_XS]
end
# wall-FACE midpoints of the stations (same origin convention as tools/probe_cavity_pressure.jl)
iface_wall = [(0.5*(Float64(mesh.X[i,1]) + Float64(mesh.X[i+1,1])),
               0.5*(Float64(mesh.Y[i,1]) + Float64(mesh.Y[i+1,1]))) for i in iface_stations]
# ★ WALL-PRESSURE TRANSDUCERS (2026-09-04). Leroux/Coutier-Delgosha/Astolfi read the shedding frequency
# from WALL-PRESSURE SPECTRA at ten suction-side stations x/c 0.05..0.90 (C31 = x/c 0.20 is their phase
# reference, C61 = 0.50) -- memory/project_docs/2026-09-04_LEROUX_CONVENTION_MATCH.md a9-a12. The like-for-like f is
# therefore the spectrum of the wall Cp at those stations, not an event count. Cp uses the physical
# p_mix = rho*cs2*T + pexc in the first cell off the wall, p_ref from tem_ambient, q = 1/2 rho_inf U^2.
const CPX_XS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
cpx_stations = let cand = [i for i in NIW+1+BASE_IN_FOIL:NIW+NIF-BASE_IN_FOIL if Float64(ycH0[i,1]) > 0], xs_all = Float64.(Array(s.xc)[cand, 1])
    [cand[argmin(abs.(xs_all .- xs))] for xs in CPX_XS]
end
const Q_INF = 0.5*Float64(s.rho_inf)*U^2
cphist = NamedTuple[]
# ★ TRAILING-EDGE BASE (2 Oct 2026, blunt trailing edge plan sections 5.2 and 5.6). Per frame: the face-length-weighted
# gauge C_p of the base faces' first cells (both halves, and each half), and the base's pressure drag and lift from GAUGE
# pressure in loads_ka's sign convention (the CSV's Cl/Cd already include the base: this is its part). The base friction
# is left to the wall instruments. Written to base/<tag>_base.csv only when n_base > 0; the main CSV is unchanged.
# With the strip topology (plan section 4.7) the base is the strip block's column 1: its faces j_s = 1..2 n_base (the
# partition's lower half j_s = 1..n_base beside the lower corner first, then the upper half), their i-face vectors (the
# strip's host mirrors Six1_h, Siy1_h, which point into the fluid as the C block's j = 1 normals do, so the force has
# loads_ka's sign) and the strip's published pressure of its column-1 cells. The sidecar's schema and its columns' meaning
# are unchanged.
const BASE_FACES = MESH_TE_STRIP ? [k - NIF for k in TE_PART.base] : [NIW + k for k in TE_PART.base]
const BASE_SX = MESH_TE_STRIP ? species.strip.Six1_h : s.Sjx1_h
const BASE_SY = MESH_TE_STRIP ? species.strip.Siy1_h : s.Sjy1_h
const BASE_LEN = [hypot(Float64(BASE_SX[i]), Float64(BASE_SY[i])) for i in BASE_FACES]
basehist = NamedTuple[]
baseprobe = Tuple{Float64,Float64}[]
base_cp_mean(cp, w) = sum(cp .* w)/sum(w)
"The published pressure of base face `k`'s first cell (a BASE_FACES entry): the C block's row-1 cell, or the strip's column-1 cell."
base_p(P, k) = MESH_TE_STRIP ? Float64(P[1+g, k+g]) : Float64(P[k+g, 1+g])
function base_reading(P, t)
    nb = MESH_TE_BASE
    cp = [(base_p(P, i) - p_ref)/Q_INF for i in BASE_FACES]
    Fx = 0.0; Fy = 0.0
    for i in BASE_FACES
        pg = base_p(P, i) - p_ref
        Fx -= pg*Float64(BASE_SX[i]); Fy -= pg*Float64(BASE_SY[i])
    end
    (t=t, cpb=base_cp_mean(cp, BASE_LEN), cpb_lower=base_cp_mean(cp[1:nb], BASE_LEN[1:nb]),
     cpb_upper=base_cp_mean(cp[nb+1:end], BASE_LEN[nb+1:end]),
     cd_base_pressure=(Fx*Float64(s.ca) + Fy*Float64(s.sa))/Q_INF, cl_base_pressure=(Fy*Float64(s.ca) - Fx*Float64(s.sa))/Q_INF)
end
"""High-rate base probe (TOPO_BASE_PROBE_EVERY): two contiguous first-row slices of the published pressure, one small read;
with the strip topology the strip's column-1 cells, one strided read (the pattern of loads_strip_ka)."""
function sample_base_probe!(t_ctu)
    if MESH_TE_STRIP
        col = Array(view(species.strip.pressure, g+1, g+1:g+2MESH_TE_BASE))
        cp = (Float64.(col) .- p_ref) ./ Q_INF
    else
        P = species.pressure
        lo = Array(view(P, g+NIW+1:g+NIW+MESH_TE_BASE, g+1))
        up = Array(view(P, g+NIW+NIF-MESH_TE_BASE+1:g+NIW+NIF, g+1))
        cp = (vcat(Float64.(lo), Float64.(up)) .- p_ref) ./ Q_INF
    end
    push!(baseprobe, (t_ctu, base_cp_mean(cp, BASE_LEN)))
end
function iface_profile(A, xcH, ycH, i, wx, wy, g, NJ)
    a = [Float64(A[i+g, j+g]) for j in 1:NJ]
    d = [sqrt((Float64(xcH[i,j]) - wx)^2 + (Float64(ycH[i,j]) - wy)^2) for j in 1:NJ]   # from the wall face
    e_cells = 0
    while e_cells < NJ && a[e_cells+1] >= IFACE_ATH; e_cells += 1; end
    e_c = e_cells == 0 ? 0.0 : d[e_cells]
    # first outward 0.9 -> 0.1 crossing: start where a first drops below 0.9 (from the wall), end at a < 0.1
    j0 = findfirst(<(0.9), a); w_cells = 0; w_c = 0.0
    if j0 !== nothing
        j1 = j0
        while j1 <= NJ && a[j1] > 0.1; j1 += 1; end
        w_cells = j1 - j0
        # chord width of the same cells, face to face (cell faces at the midpoints between centres)
        w_c = w_cells == 0 ? 0.0 :
              (j1 <= NJ ? 0.5*(d[j1] + d[j1-1]) - (j0 > 1 ? 0.5*(d[j0] + d[j0-1]) : 0.0) : NaN)
    end
    jm = argmax(a)
    (a_wall=a[1], a_max=a[jm], j_amax=jm, e_cells=e_cells, e_c=e_c, w_cells=w_cells, w_c=w_c)
end
ihist = NamedTuple[]
t0 = time()
physical_clock = PhysicalClock()
"""The pair's interior inventory (strip topology, plan section 4.7): the C block's and the strip block's mass and species
sums added (`fields`, `fields_strip`: the same frame's host fields of each block, or nothing to compute them)."""
function movie_pair_inventory(fields=nothing, fields_strip=nothing)
    a = cmesh_species_inventory(species; fields)
    b = cmesh_species_strip_inventory(species; fields=fields_strip)
    (mass=a.mass + b.mass, species=a.species + b.species)
end
species_initial = CONSERVATIVE ? (MESH_TE_STRIP ? movie_pair_inventory() : cmesh_species_inventory(species)) : nothing
species_march_retry_start = species_retry_count[]
species_boundary_totals = zeros(Float64,3) # inward carrier, inward vapor, phase generation
species_history = NamedTuple[]
function write_histories!(;announce=false)
    # ---- CSV --------------------------------------------------------------------------------------
    csv = joinpath(OUTD, tag * ".csv")
    open(csv, "w") do io
        # ⚠ The jet columns are APPENDED after the filter block so every existing consumer -- the two
        # scorers, battery12/13 parsers, spectrum_cavmovie.jl -- keeps its column positions. New readers
        # find them by NAME from the header, which is what score_cavmovie.jl already does.
        #
        # ⚠ Cd IS APPENDED LAST (2026-09-06, schema v3), NOT placed next to Cl, for the same reason. It
        # was computed every frame by loads_ka above and then thrown away, so no archived record has a
        # drag time series. Inserting it beside Cl would renumber the v1 projection ("the first twenty
        # lexical fields") and invalidate all eleven pinned records in memory/project_docs/_fixtures/cavmovie/ together
        # with their scorer fingerprints. Appended, every v2 record stays exactly as valid as it was.
        # See tools/cavmovie_record_contract.jl for what this is FOR: not the Leroux comparison, which
        # states no numeric drag anywhere in its text, but the stall confound recorded at line 125.
        println(io, "t_ctu,Cl,Vvap,ncav,ncomp,detached,nclouds,ledx,attached_area," *
                    join(["ncl_a$(td)_n$(ma)" for (td, ma) in FILTERS], ",") *
                    ",cav_x1,jet_x0,jet_x1,jet_nruns,jet_conn,us_min,n2d,x0_2d,conn2d,jmax2d," *
                    "has_wall_cavity,Cd")
        for h in hist
            @printf(io, "%.5f,%.6f,%.6e,%d,%d,%d,%d,%.4f,%d",
                    h.t, h.Cl, h.V, h.ncav, h.ncomp, h.detached, h.nclouds, h.ledx, h.atta)
            for v in h.scan; @printf(io, ",%d", v); end
            @printf(io, ",%.4f,%.4f,%.4f,%d,%d,%+.4f,%d,%.4f,%d,%d,%d,%.6f",
                     h.cav_x1, h.jet_x0, h.jet_x1, h.jet_nruns, h.jet_conn, h.us_min,
                     h.n2d, h.x0_2d, h.conn2d, h.jmax2d, Int(h.has_wall_cavity), h.Cd)
            println(io)
        end
    end
    announce && println("  wrote ", csv)
    if CONSERVATIVE
        inventory_csv=joinpath(OUTD,tag*"_inventory.csv")
        open(inventory_csv,"w") do io
            println(io,"t_ctu,mass,vapor_mass,inward_mass,inward_vapor,generated_vapor,mass_error,vapor_error,phase_volume_closure,vapor_volume_total,accepted_steps,rejected_attempts,ghost_closure")
            for h in species_history
                println(io,join((h.t,h.mass,h.species,h.mass_boundary,h.species_boundary,h.species_source,h.mass_error,h.species_error,h.volume_closure,h.vapor_volume_total,h.accepted_steps,h.rejected_attempts,h.ghost_closure),','))
            end
        end
        announce && println("  wrote ",inventory_csv,SPECIES_LEDGER ?
            " (physical phase volume; conservative mass and source inventories)" :
            " (physical phase volume; boundary/source columns and mass_error/vapor_error are NaN -- TOPO_SPECIES_LEDGER=0)")
    end

    # ---- ★ INTERFACE sidecar (2026-09-03): the pre-registered primary metric, computed from the state ----
    iface_csv = joinpath(OUTD, tag * "_interface.csv")
    open(iface_csv, "w") do io
        println(io, "t_ctu,x_over_c,i_wall,a_wall,a_max,j_amax,e_cells,e_over_c,w_cells,w_over_c")
        for h in ihist
            @printf(io, "%.5f,%.3f,%d,%.4f,%.4f,%d,%d,%.5f,%d,%.5f\n",
                    h.t, h.x, h.i, h.a_wall, h.a_max, h.j_amax, h.e_cells, h.e_c, h.w_cells, h.w_c)
        end
    end
    announce && println("  wrote ", iface_csv)
    wallcp_csv = joinpath(OUTD, tag * "_wallcp.csv")
    open(wallcp_csv, "w") do io
        println(io, "t_ctu," * join(["cp_x" * @sprintf("%.2f", x) for x in CPX_XS], ","))
        for h in cphist
            println(io, @sprintf("%.5f,", h.t) * join([@sprintf("%.6f", v) for v in h.cp], ","))
        end
    end
    announce && println("  wrote ", wallcp_csv, "  (suction-side wall Cp at Leroux's ten transducer stations, cells ",
            join(string.(cpx_stations), " "), ")")
    if MESH_TE_BASE > 0
        # In a subdirectory (2 Oct 2026): the readers find the main CSV as the one cavmov_*.csv that is not a known
        # sidecar, so a new sidecar beside it would make that lookup ambiguous; base/ keeps every existing reader unchanged.
        mkpath(joinpath(OUTD, "base"))
        base_csv = joinpath(OUTD, "base", tag * "_base.csv")
        open(base_csv, "w") do io
            println(io, "t_ctu,cpb,cpb_lower,cpb_upper,cd_base_pressure,cl_base_pressure")
            for h in basehist
                @printf(io, "%.5f,%.6f,%.6f,%.6f,%.7f,%.7f\n", h.t, h.cpb, h.cpb_lower, h.cpb_upper,
                        h.cd_base_pressure, h.cl_base_pressure)
            end
        end
        announce && println("  wrote ", base_csv, "  (trailing-edge base: gauge C_p of the first cells; base pressure drag and lift)")
        if BASE_PROBE_EVERY > 0
            probe_csv = joinpath(OUTD, "base", tag * "_baseprobe.csv")
            open(probe_csv, "w") do io
                println(io, "t_ctu,cpb")
                for (tt, c) in baseprobe; @printf(io, "%.7f,%.6f\n", tt, c); end
            end
            announce && println("  wrote ", probe_csv, "  (base C_p every ", BASE_PROBE_EVERY, " accepted steps)")
        end
    end
end

"""A frame's admissibility check of one block's interior (the cells' alpha or beta `A`, density, temperature, excess
pressure and velocity, ghost-padded host arrays): every value finite, density and temperature positive, `A` in [0, 1].
`label` names the block in the refusal (empty for the C block: the message the driver has always printed)."""
function movie_check_fields(A, RHO, TT, PX_, UX, UY, ni, nj, kf, t, label)
    for j in 1:nj, i in 1:ni
        ii, jj = i+g, j+g
        all(isfinite, (A[ii,jj], RHO[ii,jj], TT[ii,jj], PX_[ii,jj], UX[ii,jj], UY[ii,jj])) ||
            error("nonfinite field at cell ($i,$j)$label, frame $kf, t=$t CTU")
        RHO[ii,jj] > 0 && TT[ii,jj] > 0 && 0 <= A[ii,jj] <= 1 ||
            error("inadmissible density/temperature/alpha at cell ($i,$j)$label, frame $kf, t=$t CTU")
    end
    nothing
end
"""The strip block's share of a frame (strip topology, plan section 4.7), from its host fields `f`
(`cmesh_species_strip_fields`): the admissibility check of its interior (the C block's, the strip block named), its rendered
fields in its raster's cell order (beta, u_x/U, C_p on the C block's p_ref and q) and its contribution to the main CSV's
thresholded vapour volume and cell count."""
function movie_strip_frame(f, kf, t)
    ss = s_strip
    RHO_ = Array(ss.rho); TT_ = Array(ss.T); PXs = Array(ss.pexc); UX_ = Array(ss.ux); UY_ = Array(ss.uy); vs = Array(ss.vol)
    movie_check_fields(f.beta, RHO_, TT_, PXs, UX_, UY_, ss.NI, ss.NJ, kf, t, " in the strip block")
    fa = Array{Float64}(undef, ss.NI, ss.NJ); fu = similar(fa); fp = similar(fa); V = 0.0; nc = 0
    for j in 1:ss.NJ, i in 1:ss.NI
        fa[i,j] = Float64(f.beta[i+g, j+g])
        fu[i,j] = Float64(UX_[i+g, j+g])/U
        fp[i,j] = (Float64(f.pressure[i+g, j+g]) - p_ref)/q
        fa[i,j] >= THETA && (V += fa[i,j]*Float64(vs[i,j]); nc += 1)
    end
    (; fa, fu, fp, V, nc)
end
if STEADY
    # Steady pseudo-time mode (W2): replaces the physical-time march below entirely; the driver ends with it.
    include(joinpath(@__DIR__, "cmesh_cav_steady_loop.jl"))
    exit(0)
end
CONSERVATIVE && movie_checkpoint(joinpath(OUTD,tag*"_initial.jls"))

for kf in 1:NFRAME
    target = (TCTU*kf/NFRAME)/U
    while !reached_time(physical_clock, target)
        if CONSERVATIVE
            movie_species_step!(remaining_step(physical_clock,target))
            species_boundary_totals .+= (species.last_mass_boundary,species.last_species_boundary,species.last_species_source)
        else
            step_ka!(s, Float32(CFLV); wallmode=Int32(2), max_dt=remaining_step(physical_clock, target),
                     rk_incremental=MARCH_RK_INCREMENTAL)
            tem_step!(s, tem; dtp=0.0f0)
            sst_step_ka!(s; reboud_n=10.0f0, reboud_rhov=Float32(tem.rho_v_sat))
        end
        record_step!(physical_clock, s.last_global_dt)
        BASE_PROBE_EVERY > 0 && physical_clock.steps % BASE_PROBE_EVERY == 0 && sample_base_probe!(physical_clock.elapsed*U)
    end
    KernelAbstractions.synchronize(backend)
    t = physical_clock.elapsed*U
    A = Array(tem.alpha); RHO = Array(s.rho); TT = Array(s.T); PX_ = Array(s.pexc)
    volH = Array(s.vol)
    fields_s = nothing    # the strip block's host fields (strip topology); nothing otherwise
    PF = if CONSERVATIVE
        fields=cmesh_species_fields(species)
        A=fields.beta
        # With the strip topology the inventory row covers both blocks: the two interior inventories and vapour volumes
        # added, the closures the larger of the two blocks' (plan section 4.7). With TOPO_SPECIES_LEDGER=1 the ledger columns
        # are the pair's (version 1.5, plan 4.8: the published boundary is both blocks' outer faces), so mass_error closes
        # over both blocks.
        MESH_TE_STRIP && (fields_s = cmesh_species_strip_fields(species))
        totals = MESH_TE_STRIP ? movie_pair_inventory(fields, fields_s) : cmesh_species_inventory(species;fields)
        vvt = sum(fields.beta[i+g,j+g]*Float64(volH[i,j]) for j in 1:NJ,i in 1:NI)
        vcl = maximum(abs.(@view(fields.phase_volume_sum[g+1:g+NI,g+1:g+NJ]) .- 1.0)) # interior: the conserved state
        gcl = maximum(abs.(fields.phase_volume_sum .- 1.0))                             # all cells: the boundary copies dominate
        if MESH_TE_STRIP
            let ns = s_strip.NI, js = s_strip.NJ, vs = Array(s_strip.vol)
                vvt += sum(fields_s.beta[i+g,j+g]*Float64(vs[i,j]) for j in 1:js,i in 1:ns)
                vcl = max(vcl, maximum(abs.(@view(fields_s.phase_volume_sum[g+1:g+ns,g+1:g+js]) .- 1.0)))
                gcl = max(gcl, maximum(abs.(fields_s.phase_volume_sum .- 1.0)))
            end
        end
        push!(species_history,(t=t,totals...,mass_boundary=species_boundary_totals[1],
            species_boundary=species_boundary_totals[2],species_source=species_boundary_totals[3],
            mass_error=totals.mass-species_initial.mass-species_boundary_totals[1],
            species_error=totals.species-species_initial.species-species_boundary_totals[2]-species_boundary_totals[3],
            vapor_volume_total=vvt,
            volume_closure=vcl,
            ghost_closure=gcl,
            accepted_steps=physical_clock.steps,rejected_attempts=species_retry_count[]-species_march_retry_start))
        let h = species_history[end]
            if !SCORED_CLOSURE_WARNED[] && h.volume_closure > SCORED_CLOSURE_LIMIT
                SCORED_CLOSURE_WARNED[] = true
                @warn string("UNSCOREABLE ARM: interior phase-volume closure ", h.volume_closure,
                    " exceeds the reading instruments' CLOSURE_LIMIT ", SCORED_CLOSURE_LIMIT,
                    " (64*eps(Float32)) at t=", t, " CTU (ghost closure ", h.ghost_closure, "). The driver's own ",
                    "per-cell throw is 1e-2 and will NOT stop this run, but every reader will refuse the arm. ",
                    # (14 Oct 2026, review 2 of cav-models) the double EOS does not cure every excursion: the start-up
                    # excursion of near-vacuum cells at the guard floor behind the base reaches 1.0e-5 to 1.7e-5 at about
                    # 0.006 CTU under the double EOS (marches F0, F1, M1, R1), set by the Float32 pressure cache's rounding
                    SPECIES_EOS === :double ?
                    "This run already uses the double EOS: an excursion of near-vacuum cells at the guard floor (about 0.005-0.008 CTU after a uniform start) is set by the Float32 pressure cache and is not cured by it; see memory/project_docs/_records/2026-10-10_models/review2_tolerances/README.md." :
                    "TOPO_SPECIES_EOS=double restores the Float64 EOS closure (5e-8 measured) at about 1.2x the step cost, but not the guard-floor excursion of near-vacuum cells after a uniform start (memory/project_docs/_records/2026-10-10_models/review2_tolerances/README.md).")
            end
        end
        fields.pressure
    else
        nothing
    end
    UX = Array(s.ux); UY = Array(s.uy); xcH = Array(s.xc)
    (Cl, Cd) = movie_loads()
    isfinite(Cl) && isfinite(Cd) || error("nonfinite loads at frame $kf, t=$t CTU")
    # Check the actual interior before rendering/topology can omit invalid cells (with the strip topology both blocks).
    movie_check_fields(A, RHO, TT, PX_, UX, UY, NI, NJ, kf, t, "")
    # the strip block's checks and its rendered fields (strip topology; nothing otherwise)
    sf = MESH_TE_STRIP ? movie_strip_frame(fields_s, kf, t) : nothing

    # --- fields on the interior grid, in the raster's linear order ---
    fa = Array{Float64}(undef, NI, NJ); fu = similar(fa); fp = similar(fa)
    for j in 1:NJ, i in 1:NI
        fa[i,j] = Float64(A[i+g, j+g])
        fu[i,j] = Float64(UX[i+g, j+g])/U
        pmix    = PF===nothing ? Float64(RHO[i+g,j+g])*Float64(TEM_CS2)*Float64(TT[i+g,j+g]) + Float64(PX_[i+g,j+g]) : Float64(PF[i+g,j+g])
        fp[i,j] = (pmix - p_ref)/q
    end

    # --- ★ TOPOLOGY on the honest clock, via the SAME shared function Gate B uses ---
    (ncomp, maxarea, detached, ledx, totala, nclouds, atta) =
        analyse_topology(A, xcH, g, NIW_S, NIF_S, NJ, THETA, II_LE_S, THETA_DENSE, MIN_AREA)
    # --- ★★ MULTI-THRESHOLD SCAN (plan v3 S0b). tools/test_shedding_detector.jl showed Gate B's cloud
    # filters are a HARD CLIFF: at peak alpha 0.50 it counts 4 of 4 events, at 0.49 it counts ZERO,
    # with no warning and no graceful degradation. So a production negative means only "no cloud above
    # 0.50 peak and 20 cells" -- NOT "no shedding", which is the stronger claim this campaign has been
    # drawing from it. Scanning several filter pairs every frame makes that distinction VISIBLE
    # instead of assumed: if the permissive pairs fire while production does not, we were blind.
    # Cost is ~5 flood fills over 24.6k cells per frame -- negligible against the render.
    scan = [(td, ma, analyse_topology(A, xcH, g, NIW_S, NIF_S, NJ, THETA, II_LE_S, td, Int32(ma))[6])
            for (td, ma) in FILTERS]
    V = 0.0; nc = 0
    for j in 1:NJ, i in 1:NI
        a = Float64(A[i+g,j+g]); a >= THETA && (V += a*Float64(volH[i,j]); nc += 1)
    end
    # the thresholded vapour volume of the whole section: with the strip topology the strip's cells behind the base too
    MESH_TE_STRIP && (V += sf.V; nc += sf.nc)
    al = [Float64(A[ii+g, jj+g]) for (ii,jj) in stline]
    ux_wall = [Float64(UX[ii+g, jj+g])/U for (ii,jj) in stline]
    push!(stmap, al); push!(stux, ux_wall)
    xsw = [Float64(xcH[ii, jj]) for (ii,jj) in stline]      # x/c along the same wall row
    jd  = jet_diag(al, ux_wall, xsw, THETA)
    jd2 = jet_diag_2d(fu, xcH, NIW_S, II_LE_S, NIF_S, NJ,
                      jd.has_wall_cavity, jd.cav_x1, xsw[1])
    push!(hist, (t=t, Cl=Float64(Cl), Cd=Float64(Cd), V=V, ncav=nc, ncomp=ncomp, detached=detached,
                 nclouds=nclouds, ledx=ledx, atta=atta, scan=[x[3] for x in scan],
                 has_wall_cavity=jd.has_wall_cavity,
                 cav_x1=jd.cav_x1, jet_x0=jd.jet_x0, jet_x1=jd.jet_x1,
                 jet_nruns=jd.jet_nruns, jet_conn=jd.jet_conn, us_min=jd.us_min,
                 n2d=jd2.n2d, x0_2d=jd2.x0_2d, conn2d=jd2.conn2d, jmax2d=jd2.jmax2d))
    let ycH = Array(s.yc)
        for (xs, i, (wx, wy)) in zip(IFACE_XS, iface_stations, iface_wall)
            pr = iface_profile(A, xcH, ycH, i, wx, wy, g, NJ)
            push!(ihist, (t=t, x=xs, i=i, pr...))
        end
    end
    push!(cphist, (t=t, cp=[((PF===nothing ? Float64(RHO[i+g, 1+g])*(2/3)*Float64(TT[i+g, 1+g]) + Float64(PX_[i+g, 1+g]) : Float64(PF[i+g,1+g])) - p_ref)/Q_INF
                            for i in cpx_stations]))
    MESH_TE_BASE > 0 && push!(basehist, base_reading(MESH_TE_STRIP ? fields_s.pressure : PF, t))
    CONSERVATIVE && movie_checkpoint(joinpath(OUTD,tag*@sprintf("_state_%05d.jls",kf));
        scored_elapsed=physical_clock.elapsed,scored_steps=physical_clock.steps)
    CONSERVATIVE && write_histories!()

    for (nm, cl, cmap, ttl) in FIELDS
        fld = nm == "alpha" ? fa : (nm == "ux" ? fu : fp)
        # with the strip topology one image of both blocks (the strip's pixels: every pixel centre inside a strip cell, and its
        # bounding-box fill where the C block's raster has none)
        img = MESH_TE_STRIP ? raster_gather_composite(R, fld, R_STRIP, nm == "alpha" ? sf.fa : (nm == "ux" ? sf.fu : sf.fp);
                                                      inside2=IN_STRIP) :
                              raster_gather(R, fld)
        hm = heatmap(R.XR, R.YR, img; aspect_ratio=1, c=cmap, clims=cl,
                     background_color=:black, framestyle=:none, axis=false, ticks=false,
                     colorbar=true, legend=false, size=(R.PX, R.PY), nan_color=:black,
                     xlims=(WIN[1],WIN[2]), ylims=(WIN[3],WIN[4]), title=ttl, titlefontsize=9,
                     titlefontcolor=:white)
        annotate!(hm, WIN[1]+0.03*(WIN[2]-WIN[1]), WIN[4]-0.06*(WIN[4]-WIN[3]),
                  text(@sprintf("t=%.3f CTU   clouds=%d  detached=%d cells", t, nclouds, detached),
                       :white, :left, 9))
        savefig(hm, joinpath(FRD, @sprintf("%s_%05d.png", nm, kf)))
    end
    if MAKIE && (kf == 1 || kf == NFRAME || kf % MAKIE_EVERY == 0)
        try
            render_cmesh_2d(s, mesh, met;
                title=@sprintf("Cavitation sigma=%.2f   t=%.3f CTU   Cl=%.4f   V=%.3e   clouds=%d detached=%d",
                               SIGMA,t,Float64(Cl),V,nclouds,detached),
                out=joinpath(FRD, @sprintf("makie_%05d.png", kf)), win=WIN, PX=PXW, pm=PM,
                alpha=CONSERVATIVE ? A : tem.alpha, alpha_is_beta=CONSERVATIVE,
                pexc=true, pressure=PF, sound_speed=CONSERVATIVE ? s.am : nothing, Uref=U, pref=p_ref)
        catch e
            @warn "makie render failed (frame $kf)" e
        end
    end
    if kf % 20 == 0
        CONSERVATIVE && @printf("  SPECIES execution: fast_rejections=%d diagnostic_replays=%d graph_captures=%d\n",
            species.fast_rejections,species.diagnostic_replays,species_executor===nothing ? 0 : species_executor.captures)
        @printf("  frame %4d/%4d  t=%7.3f  Cl=%+.4f V=%.3e ncomp=%d clouds=%d det=%d  [%.0fs]\n",
                kf, NFRAME, t, Float64(Cl), V, ncomp, nclouds, detached, time()-t0); flush(stdout)
        if SPECIES_WALL_FUNCTION == :reichardt
            ymin, ymed, ymax, flow = wall_yplus_summary(UX, UY)
            @printf("  WALL y+ (all-y+ wall function): min %.2f  median %.2f  max %.2f  below 30: %.0f%%\n",
                    ymin, ymed, ymax, 100flow); flush(stdout)
        end
    end
end

KernelAbstractions.synchronize(backend)
@printf("PHYSICAL CLOCK RECEIPT: steps=%d elapsed=%.12g CTU dt_lattice=[%.9g,%.9g] requested=%.12g CTU\n",
        physical_clock.steps, physical_clock.elapsed*U, physical_clock.dt_min,
        physical_clock.dt_max, TCTU)
channel_diagnostics(s, tem, met, U; label="final honest-clock state",
                    mixture_pressure=CONSERVATIVE ? species.pressure : nothing)

# ---- assemble videos ---------------------------------------------------------------------------
println("assembling videos..."); flush(stdout)
mp4s = String[]
for (nm, _, _, _) in FIELDS
    out = joinpath(OUTD, tag * "_" * nm * ".mp4")
    try
        Plots.FFMPEG.exe("-y", "-framerate", "25", "-i", joinpath(FRD, nm * "_%05d.png"),
                         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out)
        push!(mp4s, out); println("  wrote ", out)
    catch e
        @warn "ffmpeg failed for $nm" e
        CAV_MOVIE_OUT_DIR_SET && rethrow()
    end
end

# ---- space-time map + history panel ------------------------------------------------------------
panel = joinpath(OUTD, tag * "_panel.png")
let ts = [h.t for h in hist]
    M = hcat(stmap...)          # stations x frames
    Mu = hcat(stux...)
    p1 = heatmap(ts, stx, M;  c=:turbo,   clims=(0,1),   xlabel="t [CTU]", ylabel="x/c",
                 title=CONSERVATIVE ? "beta(s,t) -- current wall-line vapor volume fraction" :
                                     "alpha(s,t) -- wall-line interface motion; compare with velocity and topology",
                 titlefontsize=8, colorbar=true)
    p2 = heatmap(ts, stx, Mu; c=:balance, clims=(-1,1),  xlabel="t [CTU]", ylabel="x/c",
                 title="u_x(s,t)/U  -- BLUE = reversed x velocity", titlefontsize=8)
    p3 = plot(ts, [h.V for h in hist], xlabel="t [CTU]", ylabel=CONSERVATIVE ? "V(beta >= 0.1)" : "V_vapour", lw=1.2, legend=false,
              title=CONSERVATIVE ? "thresholded vapor volume (beta >= 0.1)" : "vapour volume", titlefontsize=8)
    p4 = plot(ts, [h.detached for h in hist], xlabel="t [CTU]", ylabel="detached cells", lw=1.2,
              legend=false, title="DETACHED AREA (topology, Gate-B definition)", titlefontsize=8)
    plot!(twinx(p4), ts, [h.nclouds for h in hist], color=:red, lw=1.0, legend=false, ylabel="n clouds")
    savefig(plot(p1, p2, p3, p4; layout=(4,1), size=(1250, 1500)), panel)
    println("  wrote ", panel)
end

write_histories!(;announce=true)

let last = filter(h -> h.t == ihist[end].t, ihist)
    println("  INTERFACE at the last frame (0.9->0.1 width in wall-normal cells; archive B30: 12-24):")
    for h in last
        @printf("    x/c %.2f : a_wall=%.3f a_max=%.3f@j=%d  e=%d cells (%.4f c)  width=%d cells (%.4f c)\n",
                h.x, h.a_wall, h.a_max, h.j_amax, h.e_cells, h.e_c, h.w_cells, h.w_c)
    end
end

# ---- ★ verdict from TOPOLOGY on the honest clock -- the thing no tool could report before -------
let nev = 0, prev = false
    for h in hist
        d = h.nclouds >= 1
        d && !prev && (nev += 1)
        prev = d
    end
    dmax = maximum(h.detached for h in hist); dfrac = count(h -> h.nclouds >= 1, hist)/length(hist)
    println()
    println("="^90)
    @printf("SAMPLED TOPOLOGY: %d cloud-detection onsets, duty %.1f%%, max detached %d cells\n",
            nev, 100*dfrac, dmax)
    println("  These counts depend on the sampled interval and detection thresholds; they do not establish shedding.")
    println()
    println("  CLOUD-FILTER SENSITIVITY -- events at each (peak alpha, min area) pair:")
    for (k, (td, ma)) in enumerate(FILTERS)
        e = 0; pr = false
        for h in hist
            d = h.scan[k] >= 1
            d && !pr && (e += 1)
            pr = d
        end
        @printf("    peak alpha >= %.2f, area >= %2d cells :  %d events%s\n", td, ma, e,
                k == 1 ? "   <-- PRODUCTION (Gate B)" : "")
    end
    println("  Differences between filter pairs indicate detection sensitivity in this sampled interval.")
    println("="^90)
end

# ---- Telegram ----------------------------------------------------------------------------------
if TG
    py = Sys.iswindows() ? "python" : "python3"
    send(p, cap, kind) = try
        run(`$py $(joinpath(ROOT,"tools","tg_send.py")) $p $cap $kind`)
    catch e; @warn "telegram send failed" p e; end
    send(panel, "CAV $tag -- space-time alpha(s,t), u_x(s,t), V(t), detached area", "photo")
    for m in mp4s; send(m, "CAV $tag -- " * basename(m), "doc"); end
    makie_frames = MAKIE ? sort(filter(f -> startswith(f, "makie_") && endswith(f, ".png"), readdir(FRD))) : String[]
    for f in unique([first(makie_frames, 1); last(makie_frames, 1)])   # first and last Makie renders
        send(joinpath(FRD, f), "CAV $tag -- Makie field render $(f): rho, Mach, Cp(p_mix), vorticity, alpha (transported), u_x/U", "photo")
    end
    println("telegram: sent panel + $(length(mp4s)) videos + $(length(unique([first(makie_frames, 1); last(makie_frames, 1)]))) makie renders")
end
@printf("DONE in %.1f min\n", (time()-t0)/60)
