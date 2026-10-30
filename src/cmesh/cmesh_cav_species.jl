# ---------------------------------------------------------------------------------------------
# PATH STATUS: **PRODUCTION**. This is the solver every current cavitating result runs on
# (TOPO_SPECIES=1). Its step is step_species_ka!; its phase change is _phase_source in this file.
# Map and dispatch: memory/project_docs/CAVITATION_CODE_MAP.md (checked by tools/check_code_map.jl).
# ---------------------------------------------------------------------------------------------
# Opt-in conservative species transport with resolved phase change. Load the existing solver and
# TEM modules first; this module does not instantiate duplicate copies of either.
module CMeshCavSpecies
using KernelAbstractions
using SHA
using Printf: @sprintf   # the movie-tag suffixes of the 10 Oct 2026 settings (_species_model_tag_suffix)
import ..CMeshSolverKA
import ..CMeshCavTEM
const KA = KernelAbstractions
export SpeciesKA, build_species_ka, step_species_ka!, species_checkpoint, SpeciesRetryExhausted
export SpeciesSteadyKA, build_species_steady, step_species_steady_ka!, species_steady_residuals
export SpeciesImexKA, build_species_imex, step_species_imex_ka!
# Deferred admission (2026-09-12): status/clock reductions into device slots (see
# _step_species_deferred! below). The immediate mode remains the reference path.
const _slot_max! = CMeshSolverKA.CMeshPressureOutlet._slot_max!
const _slot_min! = CMeshSolverKA.CMeshPressureOutlet._slot_min!
const ADMISSION_SLOTS = 64
const DEFAULT_ADMISSION = Ref{Symbol}(:immediate)
# EOS arithmetic precision on the Float64 conserved state (2026-09-12, speed-first on
# FP32 GPUs): :double is the September precision path; :single forms the liquid density
# excess over saturation in Float64 and evaluates the rest of the EOS in Float32.
# Bound in the configuration tag :CMeshSpeciesSingleEOSV1; checkpoints of one
# precision are refused by the other.
const DEFAULT_EOS_PRECISION = Ref{Symbol}(:single) # rebuild, 22 Sep 2026: measured x0.84 wall, closure 5e-8 -> 1.4e-6; :double stays available
# Conservation ledger (rebuild, 22 Sep 2026). The per-stage boundary/source accounting is a DIAGNOSTIC:
# conservation itself is structural -- the flux-difference update telescopes exactly -- and the ledger only
# measures it, at six launches and one host read per accepted step. Off by default. The driver enables it
# with TOPO_SPECIES_LEDGER=1; the test suite enables it for its conservation checks. When off, the published
# boundary/source scalars are NaN so no consumer can mistake an unmeasured quantity for zero.
const LEDGER_ENABLED = Ref{Bool}(false)
# Reconstructed-face clock margin (rebuild, 22 Sep 2026): the admitted step on a reconstructed arm was
# 0.9 x the mixture face cap, a factor nobody had measured. It is now a knob (TOPO_SPECIES_CLOCK_MARGIN),
# default unchanged at 0.9 until a segment at 1.0 has been read. The host admission and the device dt fill
# must use the same value, so both read this Ref.
const CLOCK_MARGIN = Ref{Float32}(0.9f0)
# Operational CFL ceiling (speed-up plan W1a, 30 Sep 2026; memory/project_docs/2026-09-30_SPEEDUP_PLAN.md). The step refused
# cfl > 0.5 with no derivation recorded in the source; the ceiling is now a knob (TOPO_SPECIES_CFL_MAX in the
# driver), default 0.5 and therefore bitwise the September path. A larger value is admitted for a configuration
# only by a registered accuracy/stability measurement (the W1a registration); the step's own admission still
# rejects any inadmissible stage.
const CFL_MAX = Ref{Float32}(0.5f0)
# Keep full immediate replay available for diagnostic comparison. Routine,
# classified numerical rejections use the recorded deferred slots directly.
const REPLAY_REJECTIONS = Ref{Bool}(false)
# Phase-change source weighting (2026-09-19): :alpha weights the Merkle rates by the stored
# REFERENCE composition a (the September path, bitwise unchanged); :beta weights them by the
# REPRESENTED vapour volume fraction beta = q v_v / ((r-q) v_l + q v_v) formed from the same
# phase specific volumes the source already computes. memory/project_docs/2026-09-08_COMPOSITION_SEMANTICS_AUDIT.md
# records that a and 1-a are reference-volume weights, not current volumes, and that OpenFOAM's
# Merkle multiplies current phase volumes; memory/project_docs/2026-09-19_E1E2_CAVITY_CORE_READINGS.md measured
# a up to 5.9x beta at cavity closure, where the condensation sink acts. Only the weights of S
# and of the source clock's feedback term change; S remains an alpha-rate and the conversion to
# a partial-density rate is untouched. Bound in the configuration tag
# :CMeshSpeciesBetaWeightedSourceV1; checkpoints of one weighting are refused by the other.
const DEFAULT_SOURCE_WEIGHT = Ref{Symbol}(:alpha)
# Reboud correction of the eddy viscosity (2026-09-20; on by default since 10 Oct 2026, in its void-fraction form since the review
# of 13 Oct 2026, memory/project_docs/2026-10-10_REBOUD_DEFAULT_AND_SCHNERR_SAUER_DESIGN.md sections 1 and 6): :reboud_v3 (the default) is the
# VOID-FRACTION form V3, Reboud's function of the cell's represented vapour volume fraction at its own pressure (_reboud_void;
# exactly 1 wherever no vapour is held, at any pressure); :reboud_v2 the composition form of 10 Oct (_reboud_ratio of the stored
# reference composition, opt-in); :reboud_v1 the density form of 20 Sep (_reboud_factor of the mixture density, arm T25's); :none
# no correction, bitwise the September path and every archived arm. Exponent DEFAULT_REBOUD_N; see cmesh_species_sst.jl. Bound in
# the configuration fingerprint with its exponent (:CMeshSpeciesReboudNutV3 / n, V2 / n, V1 / n; none: no tag), so a restart
# never silently switches the turbulence treatment; _configuration_mismatch names the setting that restores a file of another
# treatment. The SST k production uses the uncorrected nu_t unless `reboud_production=true` (13 Oct 2026, opt-in, tag
# :CMeshSpeciesReboudProductionV1): see _sst_coefficients.
const DEFAULT_NUT_DENSITY = Ref{Symbol}(:reboud_v3)
# The defaults epoch of the driver (10 Oct 2026, _species_nut_density_setting): a species run must name the Reboud treatment
# or declare TOPO_SPECIES_DEFAULTS=<this>, because the default changed under every launcher written before it. Its defaults were
# fixed at the review of 13 Oct 2026, before any merge: the void-fraction form, the production from the uncorrected nu_t.
const DEFAULTS_EPOCH = "2026-10-10"
# Phase-change source model (10 Oct 2026, memory/project_docs/2026-10-10_REBOUD_DEFAULT_AND_SCHNERR_SAUER_DESIGN.md section 2): :merkle (the
# default, bitwise the September path: the TEM's kprod/kdest rates, _phase_source) or :schnerr_sauer (the OpenFOAM v11 form
# with its nucleus seed, _phase_source_ss, parameters ss_n0 [nuclei per chord^3] and ss_dnuc [chords], no module default).
# Bound in the configuration fingerprint as :CMeshSpeciesSchnerrSauerV2/n0=<N0>/d=<D> (V2 since 13 Oct 2026: the trace level
# SS_TRACE_Y; V1, 613dabb, had none).
const DEFAULT_SOURCE_MODEL = Ref{Symbol}(:merkle)
# Schnerr-Sauer trace level (13 Oct 2026, the review of 13 Oct; design note section 6): at or above saturation, a cell whose vapour
# mass fraction q/r is at most this is treated as vapour-free: no source and no clock, exactly as a cell with q = 0. The decision
# is made on the authoritative Float64 state in both EOS precisions, so the single and the double EOS take the same cells; the
# level lies 8 decades inside Float32's normal range, so above it both precisions form the source and its clock in normal numbers.
# Without it the 3/R_nuc condensation clock of a trace (it does not depend on the amount) bound the step down to any vapour mass the
# double EOS could hold, while the single EOS returned (0, Inf) once q left Float32's range. It does not change the clock of
# traces above it; the exact decay of the condensing trace is the follow-up (design note section 5).
const SS_TRACE_Y = 1.0e-30
# Species all-y+ wall function (2026-09-26): selected by the solver scalar `s.wallfun == 2` (0 = the wall-
# resolved SST wall, the default and bitwise the September path; the legacy log law, 1, stays refused here).
# The flux kernel, loads, viscous clock and SST wall treatment all read it through this one predicate; the
# solver scalars are part of `_configuration`, and the tag below names it, so checkpoints cannot mix walls.
# docs/2026-09-26_WALL_FUNCTION_SPECIES_PATH.md.
_wall_function(s)=s.wallfun==2.0f0 ? Int32(1) : Int32(0)
const DEFAULT_REBOUD_N = Ref{Float32}(10.0f0)
# The reconstruction's phase-bound retreat (15 Oct 2026, option H of the V1-N60 storm review; memory/project_docs/_records/2026-10-15_recon_h/
# DESIGN.md): :halving (the default, bitwise the path before: one prevfloat step, then theta halves) or :continuous (the largest
# common phase theta that meets every face's phase bounds, in closed form, _phase_bound_theta). Acts only where the face phase
# bounds fail at the Barth-Jespersen theta. Bound in the configuration fingerprint as :CMeshSpeciesContinuousPhaseRetreatV1,
# after the reconstruction's limiter tag; :halving adds no tag, so every archived checkpoint restores under the default.
const DEFAULT_RECON_RETREAT = Ref{Symbol}(:halving)
const RECON_RETREAT_TAG = :CMeshSpeciesContinuousPhaseRetreatV1
"The cache code of a reconstruction retreat (0 halving, 1 continuous); :continuous requires the reconstruction it acts in."
function _recon_retreat_setting(reconstruction::Bool,retreat::Symbol)
    retreat in (:halving,:continuous) || throw(ArgumentError("species recon_retreat must be :halving or :continuous (got :$retreat)"))
    retreat===:halving && return Int32(0)
    reconstruction || throw(ArgumentError(
        "species recon_retreat=:continuous is the reconstruction's phase-bound retreat; it requires reconstruction=true (driver TOPO_SPECIES_RECON=1)"))
    Int32(1)
end
# The retreat a builder asks for: an explicit `retreat`, or (nothing, the keyword's default) the module default where a
# reconstruction exists and :halving where none does, so a battery run under DEFAULT_RECON_RETREAT[] = :continuous still builds
# its unreconstructed caches; an explicit :continuous without a reconstruction is refused by _recon_retreat_setting.
_recon_retreat_choice(reconstruction::Bool,retreat::Union{Nothing,Symbol})=
    retreat===nothing ? (reconstruction ? DEFAULT_RECON_RETREAT[] : :halving) : retreat
# The retreat of a built cache (or strip block): its reconstruction workspace's code, 0 without one.
_recon_retreat(c)=c.reconstruction===nothing ? Int32(0) : c.reconstruction.retreat
_recon_retreat_symbol(code::Int32)=code==Int32(1) ? :continuous : :halving
# Prescribed boundary-layer trip (2026-09-30, opt-in): the Caltech foil was tripped by 0.10 mm glass spheres over the
# first 1.5% of chord on both surfaces, while the SST boundary layer transitions naturally at a station that moves with
# the leading-edge resolution. `trip_x > 0` raises the SST k, at each stage commit, to an inner-layer turbulent level
# in the cells of the foil columns whose wall station x/c lies in [trip_x, trip_x + trip_width] and whose y+ is at most
# TRIP_YPLUS_MAX; the added k is ledgered with the k floor. (0, 0) is off and bitwise the default path. Bound in the
# configuration fingerprint (:CMeshSpeciesTripV1/x/w). docs/2026-09-30_LEADING_EDGE_FIX.md.
const TRIP_YPLUS_MAX = 30.0f0
const DEFAULT_TRIP_WIDTH = 0.005
"Foil columns (lower, upper) whose wall-cell centre lies in the trip strip [x, x + w] (host arithmetic)."
function trip_columns(s,trip::NTuple{2,Float32})
    xw=Array(s.xc)[:,1];NIW,NIF=Int(s.NIW),Int(s.NIF)
    cols=NIW+1:NIW+NIF;ile=cols[argmin(xw[cols])]
    inside(i)=trip[1]<=xw[i]<=trip[1]+trip[2]
    (count(inside,NIW+1:ile-1),count(inside,ile:NIW+NIF))
end
# Limited SST non-orthogonal diffusion correction (2026-10-02, opt-in; blunt trailing edge,
# memory/project_docs/_records/2026-10-02_blunt_te/sst_clock/README.md). On a skewed cell next to a steep omega jump (the wake cell beside a
# blunt base midpoint, next to the wall-adjacent base cell) the explicit non-orthogonal part of the interior SST diffusion
# is a large outflow against the cell's small content, and the positivity clock collapses (a reject-retry storm, measured).
# lambda = sst_correction_limit in [0, 1] bounds each face correction by lambda times its two-point flux (the "limited
# corrected" Laplacian); the clock then bounds each face's total diffusive outflow by (1 + lambda) times its two-point
# coefficient on the faces that carry a correction and by 1 times it on the others (V2, 3 Oct 2026: V1 charged the
# allowance on every face, the uncorrected wake-cut and wall faces included; memory/project_docs/_records/2026-10-02_blunt_te/sst_clock/FIX.md).
# Negative (the default): the full correction, bitwise the existing path.
function _sst_climit_setting(s,reconstruction::Bool,limit::Real)
    limit<0 && return -1.0f0
    isfinite(limit) && 0<=limit<=1 || throw(ArgumentError("species sst_correction_limit must lie in [0, 1] (negative = off)"))
    s.sst && reconstruction || throw(ArgumentError("species sst_correction_limit acts on the reconstructed SST diffusion; it requires SST and reconstruction"))
    Float32(limit)
end
# Stage-1 clock margin (3 Oct 2026, opt-in; memory/project_docs/_records/2026-10-02_blunt_te/sst_clock/BB8_STORM.md and FIX.md). The step's
# proposal is min(CFL field, max_dt, CLOCK_MARGIN x the reconstructed face cap, source, viscous, SST clocks), all evaluated
# on the state the step starts from, and stages 2 and 3 re-check the face, source, viscous and SST clocks on their stage
# states with a relative tolerance of 16 eps(Float32) and no margin. A binding source, viscous or SST clock that falls
# by more than that tolerance within one step therefore fails its stage-2 check at every first attempt, and the retry
# is exactly half the step (BB8 at 0.9 CTU: 4.0-4.5x the clean cost per CTU). stage_clock_margin = m in (0, 1] gives
# those three clocks the margin the face cap already has: the proposal takes m x each of them. It is a time-step
# setting -- no flux, source or clock definition changes -- bound in the configuration as
# CMeshSpeciesStageClockMarginV1/m=<m>. The same stage-1 slots already hold those clocks, so it adds no clock
# evaluation and no launch. Negative (the default): off, and the proposal is bitwise the existing one.
function _stage_margin_setting(margin::Real)
    margin<0 && return -1.0f0
    # validated after the Float32 conversion the cache stores, and bounded below like TOPO_SPECIES_CLOCK_MARGIN, so a
    # tiny margin can neither underflow to 0 (which every consumer reads as off) nor propose a vanishing step
    m=Float32(margin)
    isfinite(m) && 0.5f0<=m<=1.0f0 || throw(ArgumentError("species stage_clock_margin must lie in [0.5, 1] (negative = off; got $margin)"))
    m
end
# One stage-1 clock under the margin (host and device; m <= 0: the clock itself, bitwise).
@inline _stage_clock(m::Float32,cap::Float32)=m>0.0f0 ? m*cap : cap
function _trip_setting(s,trip_x::Real,trip_width::Real)
    iszero(trip_x) && return (0.0f0,0.0f0)
    isfinite(trip_x) && 0<trip_x<0.5 || throw(ArgumentError("species trip_x must lie in (0, 0.5) (0 = off)"))
    isfinite(trip_width) && 0<trip_width<=0.1 || throw(ArgumentError("species trip_width must lie in (0, 0.1]"))
    s.sst && s.nu_lam>0.0f0 || throw(ArgumentError("the species trip acts on the SST k and requires SST with a laminar viscosity"))
    s.wallfun==2.0f0 && throw(ArgumentError("the species trip reads the resolved wall shear; it is not defined with the wall function"))
    trip=(Float32(trip_x),Float32(trip_width))
    # At least one foil column must lie in the strip; the production driver additionally requires one on each surface.
    sum(trip_columns(s,trip))>=1 ||
        throw(ArgumentError("the trip strip [$(trip[1]), $(trip[1]+trip[2])] holds no foil column; widen it"))
    trip
end

# ---- The 10 Oct 2026 models: settings of the cache and of the driver -----------------------------------------------------
# docs/2026-10-10_REBOUD_DEFAULT_AND_SCHNERR_SAUER_DESIGN.md. The cache settings below are what build_species_ka and the restart
# validator (tools/cmesh_species_diagnostics.jl) form the configuration from; the driver settings further down are pure
# functions of an environment (a Dict or ENV), called by tools/cmesh_cav_movie.jl and unit-tested by
# tools/test_species_model_knobs.jl without running the driver.

# The Reboud treatment of a cache: (reboud_n, reboud_form, reboud_prod) with reboud_n = 0 off (no factor is ever formed), form 1
# the density form V1 of 20 Sep, form 2 the composition form V2 of 10 Oct, form 3 the void-fraction form V3 (13 Oct, the default),
# and reboud_prod 1 the production-consistent form (13 Oct: the SST k production from the corrected nu_t), 0 not.
# The bare name `reboud` is refused (review 2 of 14 Oct 2026): it meant the density form V1 from 20 Sep to 10 Oct (arm T25's
# records name it), the composition form V2 from 10 to 13 Oct and the void-fraction form V3 after, so a record naming it does not
# say which form it ran. Every form is named by its version; the refusal names all three (_REBOUD_BARE_NAME).
const _NUT_DENSITY_FORMS=(none=Int32(0),reboud_v1=Int32(1),reboud_v2=Int32(2),reboud_v3=Int32(3))
const _REBOUD_BARE_NAME=string("the bare name reboud is refused since 14 Oct 2026: it named the density form of 20 Sep (arm T25's, now ",
    "reboud_v1) until 10 Oct, the composition form (now reboud_v2) from 10 to 13 Oct and the void-fraction form (now reboud_v3) after. ",
    "Name the form: reboud_v3 (the void-fraction form, tag CMeshSpeciesReboudNutV3, the default), reboud_v2 (the composition form of ",
    "10 Oct, CMeshSpeciesReboudNutV2), reboud_v1 (the density form of 20 Sep, CMeshSpeciesReboudNutV1, which reproduces T25) or none ",
    "(the September path)")
function _nut_density_setting(nut_density::Symbol,reboud_n::Real,reboud_production::Bool=false)
    nut_density===:reboud && throw(ArgumentError(string("species nut_density=:reboud: ",_REBOUD_BARE_NAME)))
    haskey(_NUT_DENSITY_FORMS,nut_density) || throw(ArgumentError(
        "species nut_density must be :none (no correction), :reboud_v3 (the void-fraction form, the default), :reboud_v2 (the composition form of 10 Oct) or :reboud_v1 (the density form of 20 Sep); got :$nut_density"))
    if nut_density===:none
        reboud_production && throw(ArgumentError(
            "species reboud_production=true forms the SST k production from the Reboud-corrected nu_t; it requires a Reboud treatment, not nut_density=:none"))
        return (0.0f0,Int32(0),Int32(0))
    end
    isfinite(reboud_n) && reboud_n>1 || throw(ArgumentError("species reboud_n must be finite and > 1"))
    (Float32(reboud_n),_NUT_DENSITY_FORMS[nut_density],reboud_production ? Int32(1) : Int32(0))
end
# The phase-change source of a cache: (source_model, ss) with ss = (N0, D, c3, alpha_nuc). N0 and D are rounded to Float32, the
# values the configuration tag prints, and c3 = 3 cbrt(4 pi N0/3) and alpha_nuc = V_nuc/(1 + V_nuc), V_nuc = pi N0 D^3/6, are formed
# from them once, in Float64, so a tag reproduces its cache exactly.
function _source_model_setting(model::Symbol,n0,dnuc,source_weight::Symbol)
    model in (:merkle,:schnerr_sauer) || throw(ArgumentError("species source_model must be :merkle or :schnerr_sauer; got :$model"))
    if model===:merkle
        n0===nothing && dnuc===nothing || throw(ArgumentError(
            "ss_n0 and ss_dnuc are Schnerr-Sauer parameters; the Merkle source does not read them (pass source_model=:schnerr_sauer, or leave them unset)"))
        return (Int32(0),(0.0,0.0,0.0,0.0))
    end
    source_weight===:alpha || throw(ArgumentError(
        "species source_weight=:beta weights the Merkle source; it is refused with source_model=:schnerr_sauer, whose composition factors are its own"))
    n0 isa Real && dnuc isa Real && isfinite(n0) && n0>0 && isfinite(dnuc) && dnuc>0 || throw(ArgumentError(
        "source_model=:schnerr_sauer requires finite positive ss_n0 (nuclei per chord^3) and ss_dnuc (nucleus diameter in chords); there is no module default"))
    N0=Float32(n0);D=Float32(dnuc)
    isfinite(N0) && N0>0 && isfinite(D) && D>0 || throw(ArgumentError("Schnerr-Sauer ss_n0 and ss_dnuc must be representable in Float32 and positive"))
    N=Float64(N0);d=Float64(D)
    vnuc=pi*N*d^3/6;an=vnuc/(1+vnuc);c3=3*cbrt(4pi*N/3)
    isfinite(c3) && c3>0 && isfinite(an) && 0<an<1e-2 || throw(ArgumentError(
        "the Schnerr-Sauer nucleus seed alpha_nuc = $an (ss_n0 = $N0 per chord^3, ss_dnuc = $D chords) must lie in (0, 1e-2): the seed would no longer be a trace"))
    (Int32(1),(N,d,c3,an))
end
# The phase-change source term is evaluated (its term, its clock, its ledger entry and the trial's subtraction) when the model
# can produce a nonzero source: the Schnerr-Sauer source always (its seed vaporises pure liquid below p_v), the Merkle source when
# either rate is positive. For Merkle this is the Boolean every site tested before 10 Oct 2026 (bitwise); it replaces the raw
# `t.kprod>0 || t.kdest>0` predicate at all fifteen sites (tools/check_code_map.jl refuses a raw one in src/cmesh).
@inline _reactive(c)=c.source_model==Int32(1) || c.tem.kprod>0.0f0 || c.tem.kdest>0.0f0

const _NUT_DENSITY_GUARD=string("TOPO_SPECIES=1 with TOPO_SPECIES_NUT_DENSITY unset is refused (10 Oct 2026): the species default of the ",
    "Reboud correction changed from none to the void-fraction form (tag CMeshSpeciesReboudNutV3, n = 10), and every launcher ",
    "written before that date relies on the old default by omission. Set one of: TOPO_SPECIES_NUT_DENSITY=none (no correction, ",
    "the September path: it reproduces every archived arm), TOPO_SPECIES_NUT_DENSITY=reboud_v3 (the new treatment, ",
    "the void-fraction form; reboud_v2 and reboud_v1 name the forms of 10 Oct and 20 Sep), or ",
    "TOPO_SPECIES_DEFAULTS=2026-10-10 (accept every default of this epoch). ",
    "memory/project_docs/2026-10-10_REBOUD_DEFAULT_AND_SCHNERR_SAUER_DESIGN.md, sections 1.4 and 6")
"""The driver's Reboud treatment (10 Oct 2026; design note sections 1.4 and 6), a pure function of an environment `env` (ENV or a
Dict). `TOPO_SPECIES_NUT_DENSITY` in (none, reboud_v3, reboud_v2, reboud_v1): reboud_v3 the void-fraction form V3 (13 Oct),
reboud_v2 the composition form of 10 Oct, reboud_v1 the density form of 20 Sep; the bare `reboud` is refused (14 Oct 2026: it named
each form in turn, _REBOUD_BARE_NAME); exponent `TOPO_REBOUD_N` (default 10). A species run (`conservative`) that leaves it unset is
refused unless it declares `TOPO_SPECIES_DEFAULTS=2026-10-10`, the only epoch this driver accepts; it then gets the epoch's
default, reboud_v3. Returns `(density, reboud_n, set_by)` with `set_by` :knob or :epoch
(the legacy path, TOPO_SPECIES=0, reads neither and gets :none)."""
function _species_nut_density_setting(env,conservative::Bool)
    epoch=get(env,"TOPO_SPECIES_DEFAULTS",nothing)
    epoch===nothing || epoch==DEFAULTS_EPOCH || throw(ArgumentError(
        "TOPO_SPECIES_DEFAULTS must be $DEFAULTS_EPOCH, the only defaults epoch this driver knows (got \"$epoch\")"))
    nraw=get(env,"TOPO_REBOUD_N","10")
    n=tryparse(Float32,nraw)
    n===nothing && throw(ArgumentError("TOPO_REBOUD_N must be a number (got \"$nraw\")"))
    raw=get(env,"TOPO_SPECIES_NUT_DENSITY",nothing)
    if raw===nothing
        conservative || return (density=:none,reboud_n=n,set_by=:legacy)
        epoch===nothing && throw(ArgumentError(_NUT_DENSITY_GUARD))
        density=:reboud_v3;how=:epoch
    else
        raw=="reboud" && throw(ArgumentError(string("TOPO_SPECIES_NUT_DENSITY=reboud: ",_REBOUD_BARE_NAME)))
        raw in ("none","reboud_v3","reboud_v2","reboud_v1") || throw(ArgumentError(
            "TOPO_SPECIES_NUT_DENSITY must be none, reboud_v3 (the void-fraction form), reboud_v2 (the composition form of 10 Oct) or reboud_v1 (the density form of 20 Sep); got \"$raw\""))
        density=Symbol(raw);how=:knob
    end
    density!==:none && !conservative && throw(ArgumentError("TOPO_SPECIES_NUT_DENSITY=$raw requires TOPO_SPECIES=1"))
    density!==:none && !(isfinite(n) && n>1) && throw(ArgumentError("TOPO_REBOUD_N must be finite and > 1 (got $nraw)"))
    (density=density,reboud_n=n,set_by=how)
end
"""The driver's production-consistent Reboud form (13 Oct 2026; design note section 6), a pure function of an environment:
`TOPO_SPECIES_REBOUD_PRODUCTION` in (0, 1), default 0 (the SST k production from the uncorrected nu_t, the 20 Sep convention);
1 forms it from the Reboud-corrected nu_t. Refused with TOPO_SPECIES=0 and with the treatment `density` none. Returns a Bool."""
function _species_reboud_production_setting(env,conservative::Bool,density::Symbol)
    raw=get(env,"TOPO_SPECIES_REBOUD_PRODUCTION","0")
    raw in ("0","1") || throw(ArgumentError("TOPO_SPECIES_REBOUD_PRODUCTION must be 0 or 1 (got \"$raw\")"))
    raw=="0" && return false
    conservative || throw(ArgumentError("TOPO_SPECIES_REBOUD_PRODUCTION=1 requires TOPO_SPECIES=1"))
    density===:none && throw(ArgumentError(
        "TOPO_SPECIES_REBOUD_PRODUCTION=1 forms the SST k production from the Reboud-corrected nu_t; it requires a Reboud treatment (TOPO_SPECIES_NUT_DENSITY is none)"))
    true
end
"""The driver's phase-change source model (10 Oct 2026; design note section 2.5), a pure function of an environment.
`TOPO_SPECIES_SOURCE_MODEL` in (merkle, schnerr_sauer), default merkle. Schnerr-Sauer reads `TOPO_SPECIES_SS_N0` [m^-3, default
1.6e13], `TOPO_SPECIES_SS_DNUC` [m, default 2e-6] and `TOPO_SPECIES_SS_CHORD` [m, required: the physical chord the run
represents], converted to the solver's chord units (N0 = n0 c^3, D = d/c). Refused: with TOPO_SPECIES=0, with
TOPO_SPECIES_SOURCE_WEIGHT=beta, with TOPO_SPECIES_IMEX=1 and with an explicit TOPO_CPROD or TOPO_CDEST (Merkle's rates, which it
does not read); the three Schnerr-Sauer knobs are refused with Merkle. Returns `(model, n0, dnuc, chord, n0_m3, dnuc_m,
n0_default, dnuc_default)`, the solver-unit values `n0`, `dnuc` (nothing with Merkle), and whether each of n0 and d_nuc took its
default (13 Oct 2026: the defaults are OpenFOAM's tutorial calibration constants, not a measurement of any tunnel's nuclei, and
the banner says so when they are used)."""
function _species_source_model_setting(env,conservative::Bool)
    raw=get(env,"TOPO_SPECIES_SOURCE_MODEL","merkle")
    raw in ("merkle","schnerr_sauer") || throw(ArgumentError("TOPO_SPECIES_SOURCE_MODEL must be merkle or schnerr_sauer (got \"$raw\")"))
    ssknobs=("TOPO_SPECIES_SS_N0","TOPO_SPECIES_SS_DNUC","TOPO_SPECIES_SS_CHORD")
    if raw=="merkle"
        for k in ssknobs
            haskey(env,k) && throw(ArgumentError(
                "$k is a Schnerr-Sauer parameter; it requires TOPO_SPECIES_SOURCE_MODEL=schnerr_sauer (the Merkle source would not read it: a dormant knob)"))
        end
        return (model=:merkle,n0=nothing,dnuc=nothing,chord=nothing,n0_m3=nothing,dnuc_m=nothing,n0_default=false,dnuc_default=false)
    end
    conservative || throw(ArgumentError("TOPO_SPECIES_SOURCE_MODEL=schnerr_sauer requires TOPO_SPECIES=1"))
    get(env,"TOPO_SPECIES_SOURCE_WEIGHT","alpha")=="beta" && throw(ArgumentError(
        "TOPO_SPECIES_SOURCE_MODEL=schnerr_sauer is refused with TOPO_SPECIES_SOURCE_WEIGHT=beta: the weighting is the Merkle source's"))
    get(env,"TOPO_SPECIES_IMEX","0")=="1" && throw(ArgumentError(
        "TOPO_SPECIES_SOURCE_MODEL=schnerr_sauer is not supported with TOPO_SPECIES_IMEX=1 in version 1: the line-implicit step's implicit source Jacobian is a finite difference through the Merkle source"))
    for k in ("TOPO_CPROD","TOPO_CDEST")
        haskey(env,k) && throw(ArgumentError(
            "$k is a Merkle rate; it is refused with TOPO_SPECIES_SOURCE_MODEL=schnerr_sauer, which does not read it (unset it)"))
    end
    haskey(env,"TOPO_SPECIES_SS_CHORD") || throw(ArgumentError(
        "TOPO_SPECIES_SS_CHORD (the physical chord in metres the run represents; 0.1524 for the Caltech foils) is required with TOPO_SPECIES_SOURCE_MODEL=schnerr_sauer; it has no default"))
    num(k,d)=(v=tryparse(Float64,get(env,k,d)); v===nothing ? throw(ArgumentError("$k must be a number (got \"$(env[k])\")")) : v)
    chord=num("TOPO_SPECIES_SS_CHORD","");n0m=num("TOPO_SPECIES_SS_N0","1.6e13");dm=num("TOPO_SPECIES_SS_DNUC","2.0e-6")
    isfinite(chord) && chord>0 || throw(ArgumentError("TOPO_SPECIES_SS_CHORD must be a finite positive length in metres (got $chord)"))
    isfinite(n0m) && n0m>0 || throw(ArgumentError("TOPO_SPECIES_SS_N0 must be a finite positive number of nuclei per m^3 (got $n0m)"))
    isfinite(dm) && dm>0 || throw(ArgumentError("TOPO_SPECIES_SS_DNUC must be a finite positive nucleus diameter in metres (got $dm)"))
    n0=n0m*chord^3;d=dm/chord
    _source_model_setting(:schnerr_sauer,n0,d,:alpha)   # the module's own refusals (Float32 range, alpha_nuc < 1e-2)
    (model=:schnerr_sauer,n0=n0,dnuc=d,chord=chord,n0_m3=n0m,dnuc_m=dm,
     n0_default=!haskey(env,"TOPO_SPECIES_SS_N0"),dnuc_default=!haskey(env,"TOPO_SPECIES_SS_DNUC"))
end
"""The driver's free gas (10 Oct 2026; design note section 3.2), a pure function of an environment: `TOPO_SPECIES_FREE_GAS`,
the free-gas volume fraction of the incoming liquid at the free-stream static pressure, default 0 (off, bitwise). Refused:
outside [0, 0.05), with TOPO_SPECIES=0 and with a nonzero TOPO_LQ (the direct form of the same term). Returns phi_ref."""
function _species_free_gas_setting(env,conservative::Bool)
    haskey(env,"TOPO_SPECIES_FREE_GAS") || return 0.0
    raw=env["TOPO_SPECIES_FREE_GAS"];phi=tryparse(Float64,raw)
    phi===nothing && throw(ArgumentError("TOPO_SPECIES_FREE_GAS must be a number (got \"$raw\")"))
    isfinite(phi) && 0<=phi<0.05 || throw(ArgumentError("TOPO_SPECIES_FREE_GAS must lie in [0, 0.05) (got $raw)"))
    iszero(phi) && return 0.0
    conservative || throw(ArgumentError("TOPO_SPECIES_FREE_GAS requires TOPO_SPECIES=1"))
    lq=tryparse(Float64,get(env,"TOPO_LQ","0"))
    lq===nothing || lq==0 || throw(ArgumentError(
        "TOPO_SPECIES_FREE_GAS and TOPO_LQ are mutually exclusive: both set the TEM's lq (TOPO_LQ directly, TOPO_SPECIES_FREE_GAS from the free-stream gas fraction)"))
    lq===nothing && throw(ArgumentError("TOPO_LQ must be a number"))
    phi
end
"""The driver's reconstruction retreat (15 Oct 2026, option H; memory/project_docs/_records/2026-10-15_recon_h/DESIGN.md), a pure function of an
environment: `TOPO_SPECIES_RECON_RETREAT` in (halving, continuous), default halving (bitwise the path before). continuous is refused
with TOPO_SPECIES=0 and without TOPO_SPECIES_RECON=1 (no reconstruction to retreat in). The line-implicit step and the steady mode
read the same reconstruction and take it as the explicit step does. `recon` is the driver's TOPO_SPECIES_RECON reading.
Returns :halving or :continuous."""
function _species_recon_retreat_setting(env,conservative::Bool,recon::Bool)
    raw=get(env,"TOPO_SPECIES_RECON_RETREAT","halving")
    raw in ("halving","continuous") || throw(ArgumentError("TOPO_SPECIES_RECON_RETREAT must be halving or continuous (got \"$raw\")"))
    raw=="halving" && return :halving
    conservative || throw(ArgumentError("TOPO_SPECIES_RECON_RETREAT=continuous requires TOPO_SPECIES=1"))
    recon || throw(ArgumentError("TOPO_SPECIES_RECON_RETREAT=continuous is the reconstruction's phase-bound retreat; it requires TOPO_SPECIES_RECON=1"))
    :continuous
end
"""The movie-tag suffixes of the three settings (10 Oct 2026), in the order the driver appends them: the Schnerr-Sauer source
`_ssn<n0>d<d>c<chord>` (SI values, %.3g), the free gas `_fg<phi>` (%g) and the Reboud treatment (`_rbv<n>` the void-fraction form
(13 Oct, the default), `_rbc<n>` the composition form of 10 Oct, `_reboud<n>` the density form of 20 Sep (T25's), nothing with
none), followed by `p` with the production-consistent form (`production`, 13 Oct). Merkle, no free gas and none add nothing:
archived names."""
function _species_model_tag_suffix(nut,src,phi::Float64;production::Bool=false)
    sfx=src.model===:schnerr_sauer ? @sprintf("_ssn%.3gd%.3gc%.3g",src.n0_m3,src.dnuc_m,src.chord) : ""
    phi>0 && (sfx*=@sprintf("_fg%g",phi))
    nut.density===:reboud_v3 && (sfx*="_rbv"*string(round(Int,nut.reboud_n)))
    nut.density===:reboud_v2 && (sfx*="_rbc"*string(round(Int,nut.reboud_n)))
    nut.density===:reboud_v1 && (sfx*="_reboud"*string(round(Int,nut.reboud_n)))
    production && nut.density!==:none && (sfx*="p")
    sfx
end
"""The free-gas conversion (10 Oct 2026; design note section 3.2): the free-gas volume fraction `phi` of the incoming liquid at the
free-stream static pressure, to the TEM's `lq` (the liquid phase's free-gas volume fraction at p = p_v, isothermal, a fixed gas mass
per unit liquid mass: the liquid's specific volume v_l = (1/rho_l)[1 - (p - p_v)/(rho_l a2l) + lq (p_v/p - 1)]). At the free stream
phi = lq p_v rho_inf/(x rho_l,sat), x the guarded ambient root; substituted into build_tem's exact-reference quadratic,
    rho_l,sat^2 (1 + (x - p_v) phi/p_v) - rho_inf rho_l,sat + rho_inf (x - p_v)/a2l = 0   (dense root),   lq = phi x rho_l,sat/(p_v rho_inf).
`pv`, `a2l`, `psm` are the TEM's realised Float32 constants (promoted), `q = rho_inf U^2/2` and `sigma` the requested cavitation
number, as build_tem's inversion forms them; `free_gas_lq(phi, s, t, sigma)` reads them from a carrier and an lq = 0 TEM. phi = 0
gives lq = 0.0 exactly. Returns (lq, rho_l_sat, x)."""
function free_gas_lq(phi::Float64;pv::Float64,a2l::Float64,psm::Float64,sigma::Float64,q::Float64,rho_inf::Float64)
    iszero(phi) && return (lq=0.0,rho_l_sat=NaN,x=NaN)
    isfinite(phi) && 0<phi<0.05 || throw(ArgumentError("the free-gas fraction must lie in (0, 0.05) (got $phi)"))
    target=pv+sigma*q;x=target-psm*psm/(4target)
    k1=1+(x-pv)*phi/pv
    disc=rho_inf^2-4k1*rho_inf*(x-pv)/a2l
    isfinite(disc) && disc>0 || throw(ArgumentError("free gas phi=$phi: the exact-reference pressure has no dense liquid branch"))
    rl=(rho_inf+sqrt(disc))/(2k1)
    (lq=phi*x*rl/(pv*rho_inf),rho_l_sat=rl,x=x)
end
function free_gas_lq(phi::Float64,s,t,sigma::Float64)
    t.lq==0.0f0 || throw(ArgumentError("free_gas_lq reads the realised constants of an lq = 0 TEM"))
    U=sqrt(Float64(s.uxi)^2+Float64(s.uyi)^2)
    free_gas_lq(phi;pv=Float64(t.p_v),a2l=Float64(t.a2l),psm=Float64(t.psm),sigma=sigma,q=0.5*Float64(s.rho_inf)*U^2,
                rho_inf=Float64(s.rho_inf))
end
struct SpeciesNumericalRejection <: Exception
    stage::Int
    reason::Symbol
    dt::Float32
end
struct SpeciesRetryExhausted <: Exception
    halvings::Int
    rejection::SpeciesNumericalRejection
end
Base.showerror(io::IO,e::SpeciesNumericalRejection)=print(io,"species stage ",e.stage," rejected ",e.reason," at dt=",e.dt)
Base.showerror(io::IO,e::SpeciesRetryExhausted)=print(io,"species retry exhausted after ",e.halvings,
    " halvings; base state restored; ",e.rejection)
include(joinpath(@__DIR__,"cmesh_species_sst.jl"))
include(joinpath(@__DIR__,"cmesh_species_reconstruction.jl"))

"""Additional authoritative species state, bound to one solver/TEM pair.

`conserved.U` stores Float64 rho, rho*u, rho*v and q. `q` is its interior
partial-density view, not current void fraction. Shared Float32 primitives are
derived caches; geometry and
configuration belong to the bound solver and must remain fixed. Do not edit the
cache arrays. Save `species_checkpoint(cache)` alongside the complete solver/TEM
state; rebuilding from alpha alone deliberately initializes a new species state
and loses the previous precise conserved state.
"""
mutable struct SpeciesKA{S,T,A,M,B,L,R,Q,C,K,V,BB,CF,BD,RB,ST}
    const solver::S
    const tem::T
    const wallmode::Int32
    const turbulence::Union{Nothing,SpeciesSSTKA{A}}
    const q::Q
    const q0::Q
    const qtrial::Q
    const conserved::C
    const Fi::A
    const Fj::A
    const source::A
    const source_dt::A
    const pressure::A
    const status::A
    const ledger::L
    const admission::V # deferred-admission flag/clock slots (UInt32, 2*ADMISSION_SLOTS): scratch, never rollback or checkpoint state
    const banks::BB # scratch, never rollback or checkpoint state
    const readback::RB # reusable host/device admission + ledger packet
    const step_limit::V # UInt32 bits of max_dt, updated outside optional graph replay
    const reconstruction::R
    const kinetic::K # nothing (AUSM faces) or the kinetic half-mass/clock workspace below
    const backup::B
    const configuration::CF
    const bindings::BD
    const geometry::NTuple{11,A}
    const boundary_kinds::M
    const geometry_sha256::String
    const inlet_present::Bool # static: any segmented inlet face, fixed with the bound geometry
    const eos_single::Bool # single-precision EOS arithmetic on the Float64 state (2026-09-12)
    const source_beta::Bool # phase-change source weighted by the represented vapour volume fraction (2026-09-19)
    const source_model::Int32 # phase-change source model (10 Oct 2026): 0 Merkle (the TEM's kprod/kdest), 1 Schnerr-Sauer
    const ss::NTuple{4,Float64} # Schnerr-Sauer (N0 per chord^3, D in chords, c3 = 3 cbrt(4 pi N0/3), alpha_nuc); zeros with Merkle
    const reboud_n::Float32 # Reboud correction exponent of the published nu_t, 0 = off (2026-09-20)
    const reboud_form::Int32 # Reboud form (10 Oct 2026): 0 off, 1 the density form V1 of 20 Sep, 2 the composition form V2, 3 the void-fraction form V3 (13 Oct)
    const reboud_prod::Int32 # production-consistent Reboud form (13 Oct 2026): 1 the SST k production from the corrected nu_t, 0 not
    const trip::NTuple{2,Float32} # prescribed trip (wall station x/c, strip width), (0, 0) = off (2026-09-30)
    const sst_climit::Float32 # limited SST non-orthogonal diffusion correction (2026-10-02); < 0 = off (the full correction)
    const stage_margin::Float32 # stage-1 margin on the source, viscous and SST clocks (2026-10-03); <= 0 = off
    # The strip block of the strip trailing-edge topology (5 Oct 2026, memory/project_docs/2026-10-05_STRIP_TOPOLOGY_PLAN.md section 3.1):
    # nothing on every cache built without one (its type parameter is then Nothing, so every `c.strip === nothing` test
    # is resolved at compile time and the default path is today's), or the SpeciesStripKA of cmesh_species_strip.jl.
    const strip::ST
    steps::Int
    elapsed::Float64
    last_mass_boundary::Float64
    last_species_boundary::Float64
    last_species_source::Float64
    valid::Bool
    last_retries::Int
    last_rejection::Union{Nothing,SpeciesNumericalRejection}
    fast_rejections::Int # execution telemetry, not physical/restart state
    diagnostic_replays::Int
end

# Published flow/composition/derived fields, including every ghost. RK, residual,
# limiter/status/ledger and TEM alpha_new buffers are work space, not rollback state.
# TEM.res is not written by this path (the projection's res argument is unused).
function _rollback_arrays(s,t,conserved,pressure,turbulence)
    base=(s.rho,s.ux,s.uy,s.T,s.pexc,s.am,t.alpha,conserved.U,pressure,s.dtf)
    turbulence===nothing ? base : (base...,s.tk,s.tom,s.nut,turbulence.rk,turbulence.rw)
end
# With a strip bound (strip topology, plan section 4.1 item 3 and 4.6) the strip block's own published arrays and the
# pair's two U_nb buffers follow the C block's, in the order of _strip_rollback_arrays (cmesh_species_strip.jl), so one
# copy launch backs up and restores both blocks; without one the tuple is today's.
_rollback_arrays(c)=c.strip===nothing ? _rollback_arrays(c.solver,c.tem,c.conserved,c.pressure,c.turbulence) :
    (_rollback_arrays(c.solver,c.tem,c.conserved,c.pressure,c.turbulence)...,_strip_rollback_arrays(c.strip)...)

# Launch fusion (2026-09-12): the rollback backup and restore copied the ten
# (fifteen with SST) state arrays with one device copy each per accepted step.
# One kernel now copies them all: thread i copies element i of every array whose
# length reaches i (linear indexing; the tuple recursion unrolls at compile time,
# as in _geometry_equal). Bitwise a copy; the synchronisation is unchanged.
@inline _copy_tuple!(dest::Tuple{},src::Tuple{},i)=nothing
@inline function _copy_tuple!(dest::Tuple,src::Tuple,i)
    d=first(dest); s=first(src)
    @inbounds i<=length(s) && (d[i]=s[i])
    _copy_tuple!(Base.tail(dest),Base.tail(src),i)
end
@kernel function _k_copy_state!(dest,src)
    i=@index(Global,Linear)
    _copy_tuple!(dest,src,i)
end
function _copy_step_state!(dest,src,backend;sync::Bool=true)
    length(dest)==length(src) && all(size(dest[k])==size(src[k]) && eltype(dest[k])==eltype(src[k]) for k in eachindex(src)) ||
        throw(DimensionMismatch("rollback arrays changed shape or element type"))
    _k_copy_state!(backend)(dest,src;ndrange=maximum(length,src))
    # The backup is enqueued ahead of every attempt kernel on the one stream, so the attempt's own
    # host read drains it; only the restore keeps the drain, so a faulting restore is caught where it
    # happens (rebuild, 22 Sep 2026).
    sync && KA.synchronize(backend)
    nothing
end

function _step_metadata(c)
    a=c.turbulence
    (steps=c.steps,elapsed=c.elapsed,dt=c.solver.last_global_dt,
     mass=c.last_mass_boundary,species=c.last_species_boundary,source=c.last_species_source,
     sst=a===nothing ? nothing : (a.last_k_boundary,a.last_w_boundary,a.last_k_source,a.last_w_source))
end
function _restore_step!(c,metadata)
    _copy_step_state!(_rollback_arrays(c),c.backup,c.solver.backend)
    c.steps=metadata.steps;c.elapsed=metadata.elapsed;c.solver.last_global_dt=metadata.dt
    c.last_mass_boundary=metadata.mass;c.last_species_boundary=metadata.species;c.last_species_source=metadata.source
    if c.turbulence!==nothing
        a=c.turbulence
        a.last_k_boundary,a.last_w_boundary,a.last_k_source,a.last_w_source=metadata.sst
    end
    # a strip pair (strip topology): the strip block's arrays and U_nb came back with the C block's above (one rollback
    # tuple); its carrier's step and its accepted-step mirror follow the pair's
    c.strip===nothing || (c.strip.solver.last_global_dt=metadata.dt;c.strip.steps=metadata.steps)
    c.valid=true
    nothing
end

_reject_stage(c,stage,reason,message,retryable)=retryable ?
    throw(SpeciesNumericalRejection(stage,reason,c.solver.last_global_dt)) : throw(ArgumentError(message))

# Kinetic (KFVS) face workspace, allocated by `build_species_ka(...; kinetic=true)`.
# The carrier flux kernels write the area-integrated half masses M+ (>= 0) and
# M- (<= 0) of the common-gauge kinetic face and its operational clock rate at
# every face, including the exact zero halves at impermeable walls and the
# characteristic mdot splits at the pressure ends and inlet. Scratch only: it is
# regenerated at every residual evaluation and is never checkpoint/rollback state.
function _kinetic_workspace(s)
    z(dims...)=fill!(similar(s.rho,dims...),0.0f0)
    (half_i=z(s.NI+1,s.NJ,2),half_j=z(s.NI,s.NJ+1,2),rate_i=z(s.NI+1,s.NJ),rate_j=z(s.NI,s.NJ+1))
end

# Kinetic operational clock: the sum of the four face rates written by the flux
# kernels (particle support |c_x|<=3 plus mean face velocity, density-contrast
# acoustic allowance and pressure-weight velocity sensitivity) plus the final
# outward mass drain over the cell mass. A refused kinetic face (nonfinite rate)
# yields a zero cap that the caller reports or rejects; nothing is clipped.
@kernel function _k_species_kinetic_dt!(cap,@Const(rate_i),@Const(rate_j),@Const(Fi),@Const(Fj),
                                        @Const(rho),@Const(vol),cfl::Float32,NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        r=rho[i+g,j+g]
        rate=rate_i[i,j]+rate_i[i+1,j]+rate_j[i,j]+rate_j[i,j+1]
        outward=max(Fi[i+1,j],0.0f0)+max(-Fi[i,j],0.0f0)+
                max(Fj[i,j+1],0.0f0)+max(-Fj[i,j],0.0f0)
        rate+=outward/r
        dt=cfl*vol[i,j]/rate
        cap[i,j]=isfinite(rate) && rate>0.0f0 && isfinite(dt) && dt>0.0f0 ? dt : 0.0f0
    end
end

# Frozen-coefficient AUSM face estimate in speed units. The acoustic term uses
# both states and a density-contrast allowance; Kp/fa bounds pressure-driven
# mass transport and Ku bounds velocity diffusion in the pressure traction.
# The P5 velocity derivative is at most 15/16 for the actual alpha range.
# This is an operational clock, not a full nonlinear/mesh stability proof.
@inline function _species_face_speed(rL,uL,vL,cL,pL,rR,uR,vR,cR,pR,r,sx,sy,minf2)
    S=sqrt(sx*sx+sy*sy)
    if !(isfinite(S) && S>0.0f0 && isfinite(rL) && rL>0.0f0 &&
         isfinite(rR) && rR>0.0f0 && isfinite(r) && r>0.0f0 &&
         isfinite(cL) && cL>0.0f0 && isfinite(cR) && cR>0.0f0 &&
         isfinite(pL) && pL>0.0f0 && isfinite(pR) && pR>0.0f0)
        return Inf32
    end
    nx=sx/S;ny=sy/S;unL=uL*nx+vL*ny;unR=uR*nx+vR*ny
    af=0.5f0*(cL+cR);rf=0.5f0*(rL+rR)
    Mb2=0.5f0*(unL*unL+unR*unR)/(af*af)
    Mo2=minf2==0.0f0 ? 1.0f0 : min(1.0f0,max(Mb2,minf2))
    Mo=sqrt(Mo2);fa=Mo*(2.0f0-Mo)
    alp=0.1875f0*(-4.0f0+5.0f0*fa*fa)
    Pp=CMeshSolverKA._ausm_P5p(unL/af,alp)
    Pm=CMeshSolverKA._ausm_P5m(unR/af,alp)
    # Either density can become the donor at the zero-mass-flux kink.
    pressure_diffusion=(0.25f0/fa)*max(1.0f0-Mb2,0.0f0)*
        (max(rL,rR)/rf)*(max(cL*cL,cR*cR)/af)
    velocity_diffusion=0.75f0*abs(Pp*Pm)*(rL+rR)*(fa*af)/r
    pressure_split=(15.0f0/16.0f0)*(pL+pR)/(af*r)
    acoustic=max(cL,cR)*sqrt(max(rL,rR)/min(rL,rR))
    speed=max(abs(unL),abs(unR))+acoustic+pressure_diffusion+velocity_diffusion+pressure_split
    good=isfinite(Mb2) && isfinite(fa) && fa>0.0f0 && isfinite(speed) && speed>0.0f0
    good ? S*speed : Inf32
end

@inline function _species_face_rate(rho,ux,uy,am,p,iL,jL,iR,jR,r,sx,sy,minf2)
    @inbounds _species_face_speed(rho[iL,jL],ux[iL,jL],uy[iL,jL],am[iL,jL],p[iL,jL],
        rho[iR,jR],ux[iR,jR],uy[iR,jR],am[iR,jR],p[iR,jR],r,sx,sy,minf2)
end

@kernel function _k_species_face_dt!(cap,@Const(rho),@Const(ux),@Const(uy),@Const(am),@Const(p),
                                     @Const(Fi),@Const(Fj),@Const(vol),
                                     @Const(Six),@Const(Siy),@Const(Sjx),@Const(Sjy),
                                     cfl::Float32,minf2::Float32,NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        ia=i+g;ja=j+g;r=rho[ia,ja]
        rate=_species_face_rate(rho,ux,uy,am,p,ia-1,ja,ia,ja,r,Six[i,j],Siy[i,j],minf2)+
             _species_face_rate(rho,ux,uy,am,p,ia,ja,ia+1,ja,r,Six[i+1,j],Siy[i+1,j],minf2)+
             _species_face_rate(rho,ux,uy,am,p,ia,ja-1,ia,ja,r,Sjx[i,j],Sjy[i,j],minf2)+
             _species_face_rate(rho,ux,uy,am,p,ia,ja,ia,ja+1,r,Sjx[i,j+1],Sjy[i,j+1],minf2)
        # Use post-override fluxes, including characteristic/open boundaries.
        # This extra drain guard remains active even when a split coefficient
        # vanishes or a boundary replaces the numerical AUSM flux.
        outward=max(Fi[i+1,j],0.0f0)+max(-Fi[i,j],0.0f0)+
                max(Fj[i,j+1],0.0f0)+max(-Fj[i,j],0.0f0)
        rate+=outward/r
        dt=cfl*vol[i,j]/rate
        cap[i,j]=isfinite(rate) && rate>0.0f0 && isfinite(dt) && dt>0.0f0 ? dt : 0.0f0
    end
end

# Match the laminar conductances used by the carrier face kernels. In particular,
# a light cell next to dense fluid must use rho_face/rho_cell in its clock.
# This is an operational bound, not a positivity proof on arbitrary skew grids.
@inline function _viscous_conductance(rf,nu,sx,sy,dn)
    S=sqrt(sx*sx+sy*sy)
    isfinite(rf) && rf>0.0f0 && isfinite(S) && S>0.0f0 &&
        isfinite(dn) && dn>0.0f0 ? rf*nu*S/dn : Inf32
end

@kernel function _k_species_viscous_dt!(cap,@Const(rho),@Const(nut),@Const(vol),@Const(xc),@Const(yc),
                                        @Const(Six),@Const(Siy),@Const(Sjx),@Const(Sjy),@Const(ux),@Const(uy),
                                        nu::Float32,wallmode::Int32,wf::Int32,sst_on::Int32,fullvisc::Int32,
                                        NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32,ctx)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        ia=i+g; ja=j+g; r=rho[ia,ja]; conductance=0.0f0
        if i>1
            dx=xc[i,j]-xc[i-1,j]; dy=yc[i,j]-yc[i-1,j]
            dn=sqrt(dx*dx+dy*dy)+1.0f-30
            nf=sst_on==Int32(1) ? nu+0.5f0*(nut[ia-1,ja]+nut[ia,ja]) : nu
            conductance+=_viscous_conductance(0.5f0*(r+rho[ia-1,ja]),nf,Six[i,j],Siy[i,j],dn)
        elseif ctx!==nothing && ctx.iwall==Int32(1) && wallmode==Int32(2)
            # strip topology (plan 4.1): the strip's i = 1 face is the base, a no-slip wall on an i face; the wall term
            # of the j = 1 foil wall below on that face (the wall function is refused with a strip)
            S=sqrt(Six[1,j]*Six[1,j]+Siy[1,j]*Siy[1,j])
            dn=0.5f0*vol[1,j]/S+1.0f-30
            conductance+=(4.0f0/3.0f0)*_viscous_conductance(r,nu,Six[1,j],Siy[1,j],dn)
        end
        if i<NI
            dx=xc[i+1,j]-xc[i,j]; dy=yc[i+1,j]-yc[i,j]
            dn=sqrt(dx*dx+dy*dy)+1.0f-30
            nf=sst_on==Int32(1) ? nu+0.5f0*(nut[ia,ja]+nut[ia+1,ja]) : nu
            conductance+=_viscous_conductance(0.5f0*(r+rho[ia+1,ja]),nf,Six[i+1,j],Siy[i+1,j],dn)
        end
        if j>1
            dx=xc[i,j]-xc[i,j-1]; dy=yc[i,j]-yc[i,j-1]
            dn=sqrt(dx*dx+dy*dy)+1.0f-30
            nf=sst_on==Int32(1) ? nu+0.5f0*(nut[ia,ja-1]+nut[ia,ja]) : nu
            conductance+=_viscous_conductance(0.5f0*(r+rho[ia,ja-1]),nf,Sjx[i,j],Sjy[i,j],dn)
        elseif i<=NIW || i>NIW+NIF
            S=sqrt(Sjx[i,1]*Sjx[i,1]+Sjy[i,1]*Sjy[i,1])
            if ctx===nothing
                ip=NI+1-i;dx=xc[i,1]-xc[ip,1];dy=yc[i,1]-yc[ip,1]
            else
                # strip topology: the cell across this interface face lies in the other block (static table)
                st,fc=CMeshSolverKA._strip_station(i,NIW,NIF,ctx.block)
                dx=xc[i,1]-ctx.c_nb[st,1,fc];dy=yc[i,1]-ctx.c_nb[st,2,fc]
            end
            dn=sqrt(dx*dx+dy*dy)+1.0f-30
            nf=sst_on==Int32(1) ? nu+0.5f0*(nut[ia,ja-1]+nut[ia,ja]) : nu
            conductance+=_viscous_conductance(0.5f0*(r+rho[ia,ja-1]),nf,Sjx[i,1],Sjy[i,1],dn)
        elseif wallmode==Int32(2)
            S=sqrt(Sjx[i,1]*Sjx[i,1]+Sjy[i,1]*Sjy[i,1])
            dn=0.5f0*vol[i,1]/S+1.0f-30
            if wf==Int32(1)
                # Species all-y+ wall function (26 Sep 2026): the wall shear rho*u_tau(U)^2 has linearised
                # stiffness 2y+/(u+ + y+ du+/dy+) times the laminar rho*nu/y (cmesh_wallfun.jl); charge at least
                # the 4/3 of the Stokes wall it replaces.
                nx=Sjx[i,1]/S; ny=Sjy[i,1]/S; uw=ux[ia,ja]; vw=uy[ia,ja]; un=uw*nx+vw*ny
                utx=uw-un*nx; uty=vw-un*ny
                _,yp,_=CMeshSolverKA.wf_species_state(sqrt(utx*utx+uty*uty),dn,nu)
                conductance+=max(4.0f0/3.0f0,CMeshSolverKA.wf_species_clock_factor(yp))*
                    _viscous_conductance(r,nu,Sjx[i,1],Sjy[i,1],dn)
            else
            # The no-slip Stokes wall tensor has maximum eigenvalue 4/3.
            conductance+=(4.0f0/3.0f0)*_viscous_conductance(r,nu,Sjx[i,1],Sjy[i,1],dn)
            end
        end
        if j<NJ
            dx=xc[i,j+1]-xc[i,j]; dy=yc[i,j+1]-yc[i,j]
            dn=sqrt(dx*dx+dy*dy)+1.0f-30
            nf=sst_on==Int32(1) ? nu+0.5f0*(nut[ia,ja]+nut[ia,ja+1]) : nu
            conductance+=_viscous_conductance(0.5f0*(r+rho[ia,ja+1]),nf,Sjx[i,j+1],Sjy[i,j+1],dn)
        elseif ctx!==nothing && ctx.block==Int32(1)
            # strip topology: the strip's j = NJ+1 face is the upper interface; the cell above is the upper leg's
            # row-1 cell, the ghost (the same conductance as the leg's own j = 1 face)
            dx=ctx.c_nb[i,1,2]-xc[i,j]; dy=ctx.c_nb[i,2,2]-yc[i,j]
            dn=sqrt(dx*dx+dy*dy)+1.0f-30
            nf=sst_on==Int32(1) ? nu+0.5f0*(nut[ia,ja]+nut[ia,ja+1]) : nu
            conductance+=_viscous_conductance(0.5f0*(r+rho[ia,ja+1]),nf,Sjx[i,j+1],Sjy[i,j+1],dn)
        end
        # the exact pressure ends; with a strip pair's context only the outlet sides of its mask (the strip's i = 1
        # side is the base wall, charged above)
        if fullvisc==Int32(1) && nu>0f0 && (i==1 || i==NI) &&
           (ctx===nothing || ((ctx.sides>>(i==1 ? Int32(0) : Int32(1)))&Int32(1))==Int32(1))
            # Same ghost material/eddy viscosity and displacement as the exact
            # end-face traction. 4/3 bounds the compact-correction tensor norm;
            # it is not a spectral bound for lagged GG/characteristic feedback.
            side=i==1 ? Int32(1) : Int32(2);f=side==1 ? Int32(1) : NI+Int32(1)
            sg=side==1 ? -1f0 : 1f0;ig=side==1 ? g : NI+g+Int32(1)
            dx,dy,_,_=CMeshSolverKA.CMeshPressureOutlet._pressure_end_geometry(
                Six[f,j],Siy[f,j],Sjx[i,j],Sjy[i,j],Sjx[i,j+1],Sjy[i,j+1],sg)
            nf=nu+(sst_on==Int32(1) ? nut[ig,ja] : 0f0)
            conductance+=CMeshSolverKA.CMeshPressureOutlet._pressure_end_conductance(
                rho[ig,ja],nf,Six[f,j],Siy[f,j],dx,dy)
        end
        # The outer slip/inlet boundary retains its existing zero viscous flux.
        mass=r*vol[i,j]; Lnu=conductance/mass
        good=isfinite(mass) && mass>0.0f0 && isfinite(Lnu) && Lnu>=0.0f0
        cap[i,j]=good ? (Lnu>0.0f0 ? 0.25f0/Lnu : Inf32) : 0.0f0
    end
end

# Transform the existing TEM alpha rate into partial-density units at fixed rho.
# The explicit source clock includes the EOS pressure feedback, not just E+D.
# This local bound is an accuracy/stability restriction; the stage admissibility
# checks remain authoritative and no out-of-bounds q is clipped.
@inline _phase_source(r,q,ep,kprod,kdest)=_phase_source(r,q,ep,kprod,kdest,Int32(0))
@inline function _phase_source(r,q,ep,kprod,kdest,bw::Int32)
    pv,a2l,rl,rv,psm,lq,_=ep
    a=_alpha(q/r,rl,rv); rs=rl*(1.0f0-a)+rv*a
    raw,dr,_=CMeshCavTEM.CMeshEOSMix._pmix_vl(pv,a,r,rs,a2l,rl,rv,lq)
    liquid_volume=lq==0.0f0 ? (1.0f0-(raw-pv)/(rl*a2l))/rl :
                             (1.0f0-(raw-pv)/(rl*a2l)-(raw-pv)*lq/raw)/rl
    vapor_volume=pv/(rv*raw)
    if !(0.0f0<liquid_volume<vapor_volume)
        # A vapour-free cell at or above the saturation pressure has no evaporation
        # and nothing to condense: its source is identically zero and needs no clock
        # (2026-09-12; refused before, which stopped the stiffer-liquid arms at the
        # leading-edge stagnation pressure). Cells holding vapour keep the refusal:
        # their isothermal vapour would be denser than the liquid at this pressure.
        return (q<=0 && raw>=pv) ? (0.0f0,Inf32) : (0.0f0,0.0f0)
    end
    sq=sqrt(raw*raw+psm*psm); guard=0.5f0*(1.0f0+raw/sq)
    p=0.5f0*(raw+sq)
    E=kprod*max(pv-p,0.0f0); D=kdest*max(p-pv,0.0f0)
    # Weight (2026-09-19, DEFAULT_SOURCE_WEIGHT): bw==1 uses the represented vapour volume
    # fraction beta from the phase specific volumes above (both positive here); bw==0 is the
    # September path, arithmetically the same expression in a and therefore bitwise unchanged.
    w=bw==Int32(1) ? _source_beta(r,q,liquid_volume,vapor_volume) : a
    S=E*(1.0f0-w)-D*w
    gamma=r*rl*rv/(rs*rs)*S
    # Unclipped implicit derivative. The third _pmix_vl return is a legacy
    # iteration coefficient and cannot bound the true source pressure feedback.
    # (26 Sep 2026) The liquid factor carries the lq compliance as _pmix_vl's dpda does,
    # ((1-lq) - raw*Lp); it read (1 - raw*Lp), the lq=0 form, and under-bounded the clock
    # for dilute vapour well above p_v at lq>0. 1.0f0-0.0f0 is exact: lq=0 is bitwise unchanged.
    pa=guard*dr*r*((rl-rv)*raw-r*(raw-pv)*((1.0f0-lq)-raw/(rl*a2l)))/(raw*rs)
    feedback=p<pv ? kprod*(1.0f0-w) : (p>pv ? kdest*w : max(kprod*(1.0f0-w),kdest*w))
    rate=E+D+abs(pa)*feedback+2.0f0*(rl-rv)/rs*abs(S)
    isfinite(pa) && isfinite(rate) && rate>=0.0f0 || return (gamma,0.0f0)
    cap=rate>0.0f0 ? 0.25f0/rate : Inf32
    # Divide before scaling: quartering an admitted subnormal partial density
    # can round to zero although its finite drain time is representable.
    if gamma>0.0f0
        cap=min(cap,0.25f0*((r-q)/gamma))
    elseif gamma<0.0f0
        cap=min(cap,0.25f0*(q/(-gamma)))
    end
    (gamma,cap)
end

# ---- Schnerr-Sauer phase-change source (10 Oct 2026, opt-in; memory/project_docs/2026-10-10_REBOUD_DEFAULT_AND_SCHNERR_SAUER_DESIGN.md
# section 2) ------------------------------------------------------------------------------------------------------------------
# The OpenFOAM Foundation v11 form (compressibleCavitation/SchnerrSauer: Cv = Cc = 1, the 0.01 pSat regularisation, the nucleus
# seed alpha_nuc), written as a reference-composition rate S = Da/Dt on the existing source structure. A fixed number N0 of
# spherical nuclei per unit liquid volume (per chord^3), each of radius R(a), 3/R(a) = c3 [(1-a)/(a + alpha_nuc)]^(1/3) with
# c3 = 3 (4 pi N0/3)^(1/3), grow or collapse at the inertia-free Rayleigh speed
#     V(d) = sqrt(2/(3 rl)) d/sqrt(|d| + 0.01 p_v),   d = p_v - p   (p the guarded pressure, as the Merkle source uses),
#     S = K(a) V(d),   K_v = (a + alpha_nuc)(1-a) 3/R  (d > 0, vaporisation),   K_c = a (1-a) 3/R  (d <= 0, condensation),
# converted to the partial-density source exactly as Merkle's is, gamma = r rl rv/rs^2 S (q = r Y(a), Y'(a) = rl rv/rs^2). At an
# uncompressed mixture, r = rho_s(a), gamma is OpenFOAM's vapour mass rate (4e-14 in the design note's hand values); S, a
# volume-fraction rate, does not depend on the density ratio. Conservation is structural, as for Merkle: the source moves mass
# between q and r - q, and the vapour ledger (entry 17) sums the same Float32 `source` array the trial kernel subtracts.
# The admission lines are the Merkle source's, verbatim; then a vapour-free cell at or above saturation returns (0, Inf):
# S = K_c(0) V = 0 exactly, and the 3/R_nuc condensation stiffness of a cell with nothing to condense is not a time scale (without
# this return it would set a meaningless clock in every liquid cell). Since 13 Oct 2026 (tag V2) "vapour-free" is q <= SS_TRACE_Y r,
# decided on the Float64 state (r, q) the kernel passes, identically in both precisions; at the Merkle admission's own (0, Inf)
# return the same trace test replaces q <= 0. Below p_v pure liquid vaporises from the seed,
# S(0) = alpha_nuc (3/R_nuc) V: the nuclei's surface area per unit volume times the wall speed. The clock bounds |dgamma/dq| by the
# three terms of the Merkle clock, |dK/da| |V| + |pa| K |dV/dd| + 2 (rl - rv)/rs |S| (K = max(K_v, K_c) at d == 0), with
#     dK_v/da = (3/R) [(2/3)(1-a) - (4/3)(a + alpha_nuc)],   dK_c/da = (3/R) [(1-a) - a(1-a)/(3(a + alpha_nuc)) - (4/3) a],
#     dV/dd   = sqrt(2/(3 rl)) (|d| + 2 eps)/(2 (|d| + eps)^(3/2)),   eps = 0.01 p_v,
# then 0.25/rate and the Merkle source's depletion caps, divided before scaling. At a = 1, 3/R = 0: no liquid, no source, no
# clock. Generic in the precision of its inputs (every literal exact in both, the 0.01 through oftype). The Merkle functions
# above are not edited, which keeps the default bitwise.
@inline function _ss_terms(a,p,pv,rl,rv,rs,pa,c3,an)
    w=one(a)-a; aa=a+an
    R3=c3*cbrt(w/aa)                              # 3/R(a)
    d=pv-p; ad=abs(d); e=oftype(pv,0.01)*pv; de=ad+e
    C=sqrt(2/(3*rl))
    V=C*d/sqrt(de)                                # signed Rayleigh wall speed: > 0 grows, < 0 collapses
    dV=C*(ad+2*e)/(2*de*sqrt(de))                 # dV/dd > 0
    Kv=aa*w*R3; Kc=a*w*R3
    K,dK=d>0 ? (Kv,R3*((2*w-4*aa)/3)) : (Kc,R3*(w-a*w/(3*aa)-4*a/3))
    S=K*V
    Kfb=d==0 ? max(Kv,Kc) : K
    rate=abs(dK)*abs(V)+abs(pa)*Kfb*dV+2*(rl-rv)/rs*abs(S)
    (S,rate)
end
# The Schnerr-Sauer source and clock in the input precision (Float64 on the Float64 state in the kernel). sp = (c3, alpha_nuc).
@inline function _phase_source_ss(r,q,ep,sp)
    pv,a2l,rl,rv,psm,lq,_=ep
    trace=q<=SS_TRACE_Y*r                         # at most the trace level of vapour (13 Oct 2026; Float64 in the kernel)
    a=_alpha(q/r,rl,rv); rs=rl*(1.0f0-a)+rv*a
    raw,dr,_=CMeshCavTEM.CMeshEOSMix._pmix_vl(pv,a,r,rs,a2l,rl,rv,lq)
    liquid_volume=lq==0.0f0 ? (1.0f0-(raw-pv)/(rl*a2l))/rl :
                             (1.0f0-(raw-pv)/(rl*a2l)-(raw-pv)*lq/raw)/rl
    vapor_volume=pv/(rv*raw)
    if !(0.0f0<liquid_volume<vapor_volume)
        return (trace && raw>=pv) ? (0.0f0,Inf32) : (0.0f0,0.0f0)   # the Merkle source's admission, the trace test for q <= 0
    end
    sq=sqrt(raw*raw+psm*psm); guard=0.5f0*(1.0f0+raw/sq)
    p=0.5f0*(raw+sq)
    (trace && p>=pv) && return (0.0f0,Inf32)     # no vapour beyond a trace at or above saturation: nothing to condense, no clock
    pa=guard*dr*r*((rl-rv)*raw-r*(raw-pv)*((1.0f0-lq)-raw/(rl*a2l)))/(raw*rs)   # the Merkle source's lq-aware feedback
    S,rate=_ss_terms(a,p,pv,rl,rv,rs,pa,oftype(pv,sp[1]),oftype(pv,sp[2]))
    gamma=r*rl*rv/(rs*rs)*S
    isfinite(pa) && isfinite(rate) && rate>=0.0f0 || return (gamma,0.0f0)
    cap=rate>0.0f0 ? 0.25f0/rate : Inf32
    if gamma>0.0f0
        cap=min(cap,0.25f0*((r-q)/gamma))
    elseif gamma<0.0f0
        cap=min(cap,0.25f0*(q/(-gamma)))
    end
    (gamma,cap)
end

# The phase-change source kernel. sm (10 Oct 2026): 1 the Schnerr-Sauer source with sp = (c3, alpha_nuc), any other value the
# Merkle source with kp, kd and the weighting bw, through the existing calls, unchanged.
@kernel function _k_species_source!(source,source_dt,@Const(U),@Const(vol),
                                    ep::NTuple{7,Float64},kp::Float32,kd::Float32,single::Int32,bw::Int32,
                                    sm::Int32,sp::NTuple{2,Float64},
                                    NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        local gamma::Float64; local cap::Float64
        if sm==Int32(1)
            if single==Int32(1)
                gamma32,cap32=_phase_source_ss_single(U[i,j,1],U[i,j,4],ep,sp)
                gamma=Float64(gamma32);cap=Float64(cap32)
            else
                gamma,cap=_phase_source_ss(U[i,j,1],U[i,j,4],ep,sp)
            end
        elseif single==Int32(1)
            gamma32,cap32=_phase_source_single(U[i,j,1],U[i,j,4],ep,kp,kd,bw)
            gamma=Float64(gamma32);cap=Float64(cap32)
        else
            gamma,cap=_phase_source(U[i,j,1],U[i,j,4],ep,kp,kd,bw)
        end
        weighted=Float32(gamma*Float64(vol[i,j]))
        source[i,j]=weighted
        source_dt[i,j]=isfinite(gamma) && isfinite(weighted) && cap>0.0 ? cap : 0.0
    end
end

@inline _Y(a,rl,rv) = a*rv/((1.0f0-a)*rl+a*rv)
@inline _alpha(y,rl,rv) = y*rl/(rv*(1.0f0-y)+y*rl)
# Represented vapour volume fraction from the phase specific volumes the source already forms:
# the same arithmetic as tools/cmesh_species_diagnostics.jl _cmesh_species_phase_volumes
# (vapor = q v_v, liquid = (r-q) v_l, beta = vapor/(liquid+vapor)), at the source's own (raw) pressure.
@inline _source_beta(r,q,liquid_volume,vapor_volume) = (vap=q*vapor_volume; vap/((r-q)*liquid_volume+vap))

# The phase-mass interpretation requires positive raw pressure and positive
# liquid reference volume. Check before an invalid alpha enters the EOS root.
#
# ★ SOUND-SPEED FLOOR REFERENCE (2026-09-24). The last clause is a NUMERICAL guard against a
# degenerate mixture sound speed, not a physical bound -- the Wallis mixture sound speed legitimately
# collapses far below BOTH branches inside a cavity, and reproducing that collapse is the point of this
# EOS. It was written `c2 >= afloor*a2l`, i.e. as a fraction of LIQUID STIFFNESS. That is dormant for a
# soft liquid (realised floor 0.59 m/s, six times BELOW the physical mixture minimum) and becomes a
# barrier for a physical one: the water directive's 628x stiffening carried the floor to 14.9 m/s, four
# times ABOVE that minimum, so it refused exactly the states cavitation produces and wedged every
# stiff-liquid march at 600-750 steps (memory/project_docs/2026-09-24_THE_WEDGE_IS_A_SOUND_SPEED_FLOOR.md, measured on
# this predicate by S0903_RUNS/wedge_predicate_limit_20260924.jl: admissible density deficit 0.0011%
# against 0.2447% reached).
# The reference is now `p_v/rho_l_sat`, the CAVITATION PRESSURE SCALE, which is invariant to both the
# liquid stiffening and the density ratio -- the two knobs the water directive moved -- so the realised
# floor no longer tracks a parameter it was never calibrated against. (The legacy TEM projection kernels
# keep their own a2l-referenced floor with its own recorded reasoning; that path is not production for
# the Species solver, and its banner at cmesh_cav_tem.jl:1690 explains why it rejected the a2v scale.)
#
# Pass the admitted runtime lq (exactly zero) through the same EOS arithmetic as
# the shared pressure/projection kernels. A literal zero specializes away terms
# and changes GPU rounding enough to fail sound-speed coherence near saturation.
@inline function _phase_state(r,a,pv,a2l,rl,rv,psm,afloor,lq)
    if !(isfinite(r) && r > 0.0f0 && isfinite(a) && 0.0f0 <= a <= 1.0f0)
        return (false,zero(r),zero(r))
    end
    raw,dr,_ = CMeshCavTEM.CMeshEOSMix._pmix_vl(pv,a,r,rl*(1.0f0-a)+rv*a,a2l,rl,rv,lq)
    sq=sqrt(raw*raw+psm*psm)
    p=0.5f0*(raw+sq)
    c2=0.5f0*dr*(1.0f0+raw/sq)
    good=isfinite(raw) && raw>0.0f0 && 1.0f0-(raw-pv)/(rl*a2l)>0.0f0 &&
         isfinite(p) && p>=1.0f-8 && isfinite(c2) && c2>0.0f0 && c2>=afloor*(pv/rl)
    (good,p,sqrt(max(c2,0.0f0)))
end

# Single-precision EOS of an authoritative Float64 phase state (2026-09-12, speed-first
# on FP32 GPUs). The liquid density excess over saturation is the only cancellation
# of the local-pressure EOS; it is formed here in Float64 and the quadratic, the
# smoothing and the sound speed run in Float32 through _pmix_vl_d. Same admission
# conditions as _phase_state. `ep` may be the Float32 or the Float64 reference tuple.
@inline function _phase_state_single(r64::Float64,a::Float32,ep)
    pv,a2l,rl,rv,psm,lq,afloor=map(Float32,ep)
    if !(isfinite(r64) && r64 > 0.0 && isfinite(a) && 0.0f0 <= a <= 1.0f0)
        return (false,0.0f0,0.0f0)
    end
    d=Float32(Float64(rl)-r64);r=Float32(r64)
    raw,dr,_ = CMeshCavTEM.CMeshEOSMix._pmix_vl_d(pv,a,r,d,a2l,rl,rv,lq)
    sq=sqrt(raw*raw+psm*psm)
    # Defensive identity for raw < 0, where raw + sq would cancel in Float32:
    # raw + sq = psm^2/(sq - raw). Same values as the Float64 path to Float32
    # rounding on both branches.
    # NOTE (2026-09-17 audit): this branch is UNREACHABLE. _pmix_vl_d returns the
    # NON-NEGATIVE root (cmesh_eos_mix.jl: 'Nonnegative root of ...'), so raw >= 0
    # always, and raw == 0 exactly at a = lq = 0 and sufficiently low rho -- which
    # the `raw > 0` admission below then refuses. The model represents NO liquid
    # tension: there is no negative-pressure branch to be stable in. Retained as
    # defensive code; the comment it replaces implied a tension range that does
    # not exist, and that reading cost real time in the 17 September cavitation work.
    ssum=raw>=0.0f0 ? raw+sq : psm*psm/(sq-raw)
    p=0.5f0*ssum
    c2=0.5f0*dr*(ssum/sq)
    good=isfinite(raw) && raw>0.0f0 && 1.0f0-(raw-pv)/(rl*a2l)>0.0f0 &&
         isfinite(p) && p>=1.0f-8 && isfinite(c2) && c2>0.0f0 && c2>=afloor*(pv/rl)
    (good,p,sqrt(max(c2,0.0f0)))
end
@inline function _phase_state_single_y(r64::Float64,q64::Float64,ep)
    rl=Float32(ep[3]);rv=Float32(ep[4])
    isfinite(q64) && 0.0<=q64<=r64 || return (false,0.0f0,0.0f0)
    _phase_state_single(r64,_alpha(Float32(q64)/Float32(r64),rl,rv),ep)
end
# Single-precision form of _phase_source: same rate, cap and admission logic in Float32.
@inline _phase_source_single(r64::Float64,q64::Float64,ep,kprod::Float32,kdest::Float32)=_phase_source_single(r64,q64,ep,kprod,kdest,Int32(0))
@inline function _phase_source_single(r64::Float64,q64::Float64,ep,kprod::Float32,kdest::Float32,bw::Int32)
    pv,a2l,rl,rv,psm,lq,_=map(Float32,ep)
    isfinite(r64) && r64>0.0 && isfinite(q64) && 0.0<=q64<=r64 || return (0.0f0,0.0f0)
    r=Float32(r64);q=Float32(q64);y=q/r
    a=_alpha(y,rl,rv); rs=rl*(1.0f0-a)+rv*a
    d=Float32(Float64(rl)-r64)
    raw,dr,_=CMeshCavTEM.CMeshEOSMix._pmix_vl_d(pv,a,r,d,a2l,rl,rv,lq)
    liquid_volume=lq==0.0f0 ? (1.0f0-(raw-pv)/(rl*a2l))/rl :
                             (1.0f0-(raw-pv)/(rl*a2l)-(raw-pv)*lq/raw)/rl
    vapor_volume=pv/(rv*raw)
    if !(0.0f0<liquid_volume<vapor_volume)
        # A vapour-free cell at or above the saturation pressure has no evaporation
        # and nothing to condense: its source is identically zero and needs no clock
        # (2026-09-12; refused before, which stopped the stiffer-liquid arms at the
        # leading-edge stagnation pressure). Cells holding vapour keep the refusal:
        # their isothermal vapour would be denser than the liquid at this pressure.
        return (q<=0 && raw>=pv) ? (0.0f0,Inf32) : (0.0f0,0.0f0)
    end
    sq=sqrt(raw*raw+psm*psm)
    ssum=raw>=0.0f0 ? raw+sq : psm*psm/(sq-raw) # defensive; raw<0 is unreachable (see note above)
    guard=0.5f0*(ssum/sq)
    p=0.5f0*ssum
    E=kprod*max(pv-p,0.0f0); D=kdest*max(p-pv,0.0f0)
    # Weight (2026-09-19): see _phase_source; bw==0 is bitwise the September path.
    w=bw==Int32(1) ? _source_beta(r,q,liquid_volume,vapor_volume) : a
    S=E*(1.0f0-w)-D*w
    gamma=r*rl*rv/(rs*rs)*S
    pa=guard*dr*r*((rl-rv)*raw-r*(raw-pv)*((1.0f0-lq)-raw/(rl*a2l)))/(raw*rs)  # lq-aware; see _phase_source
    feedback=p<pv ? kprod*(1.0f0-w) : (p>pv ? kdest*w : max(kprod*(1.0f0-w),kdest*w))
    rate=E+D+abs(pa)*feedback+2.0f0*(rl-rv)/rs*abs(S)
    isfinite(pa) && isfinite(rate) && rate>=0.0f0 || return (gamma,0.0f0)
    cap=rate>0.0f0 ? 0.25f0/rate : Inf32
    if gamma>0.0f0
        cap=min(cap,0.25f0*((r-q)/gamma))
    elseif gamma<0.0f0
        cap=min(cap,0.25f0*(q/(-gamma)))
    end
    (gamma,cap)
end
# Single-precision form of _phase_source_ss (10 Oct 2026): Float32 on the Float64 state, the admission lines and EOS evaluation
# of _phase_source_single verbatim, then the Schnerr-Sauer terms in Float32 (c3 and alpha_nuc rounded to Float32).
@inline function _phase_source_ss_single(r64::Float64,q64::Float64,ep,sp)
    pv,a2l,rl,rv,psm,lq,_=map(Float32,ep)
    isfinite(r64) && r64>0.0 && isfinite(q64) && 0.0<=q64<=r64 || return (0.0f0,0.0f0)
    trace=q64<=SS_TRACE_Y*r64                     # the trace test of the double form, on the same Float64 state (13 Oct 2026)
    r=Float32(r64);q=Float32(q64);y=q/r
    a=_alpha(y,rl,rv); rs=rl*(1.0f0-a)+rv*a
    d=Float32(Float64(rl)-r64)
    raw,dr,_=CMeshCavTEM.CMeshEOSMix._pmix_vl_d(pv,a,r,d,a2l,rl,rv,lq)
    liquid_volume=lq==0.0f0 ? (1.0f0-(raw-pv)/(rl*a2l))/rl :
                             (1.0f0-(raw-pv)/(rl*a2l)-(raw-pv)*lq/raw)/rl
    vapor_volume=pv/(rv*raw)
    if !(0.0f0<liquid_volume<vapor_volume)
        return (trace && raw>=pv) ? (0.0f0,Inf32) : (0.0f0,0.0f0)
    end
    sq=sqrt(raw*raw+psm*psm)
    ssum=raw>=0.0f0 ? raw+sq : psm*psm/(sq-raw) # defensive; raw<0 is unreachable (see _phase_state_single)
    guard=0.5f0*(ssum/sq)
    p=0.5f0*ssum
    (trace && p>=pv) && return (0.0f0,Inf32)
    pa=guard*dr*r*((rl-rv)*raw-r*(raw-pv)*((1.0f0-lq)-raw/(rl*a2l)))/(raw*rs)
    S,rate=_ss_terms(a,p,pv,rl,rv,rs,pa,Float32(sp[1]),Float32(sp[2]))
    gamma=r*rl*rv/(rs*rs)*S
    isfinite(pa) && isfinite(rate) && rate>=0.0f0 || return (gamma,0.0f0)
    cap=rate>0.0f0 ? 0.25f0/rate : Inf32
    if gamma>0.0f0
        cap=min(cap,0.25f0*((r-q)/gamma))
    elseif gamma<0.0f0
        cap=min(cap,0.25f0*(q/(-gamma)))
    end
    (gamma,cap)
end

# Initialize once from the admitted shared state. The initial q bits preserve
# the existing Float32 composition conversion; later accepted U never reloads
# from the shared primitive caches. Momentum is initialized without a Float32
# multiply/divide round trip.
@kernel function _k_species_init!(U,@Const(rho),@Const(ux),@Const(uy),@Const(alpha),
                                  rl::Float32,rv::Float32,NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        r=Float64(rho[i+g,j+g])
        U[i,j,1]=r;U[i,j,2]=r*Float64(ux[i+g,j+g]);U[i,j,3]=r*Float64(uy[i+g,j+g])
        U[i,j,4]=Float64(rho[i+g,j+g]*_Y(alpha[i+g,j+g],rl,rv))
    end
end

@kernel function _k_species_check!(status,@Const(U),@Const(rho),@Const(ux),@Const(uy),
                                   @Const(T),@Const(pexc),@Const(am),@Const(alpha),@Const(pressure),@Const(kinds),
                                   ep::NTuple{7,Float32},coherent::Int32,initial::Int32,single::Int32,
                                   NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32,NX::Int32,NY::Int32,ctx)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NX && j<=NY
        pv,a2l,rl,rv,psm,lq,afloor=ep
        r=rho[i,j];a=alpha[i,j];u=ux[i,j];v=uy[i,j];t=T[i,j]
        # ctx: nothing (today's donor map), or a strip pair's block context: an interface ghost's donor is the other
        # block's cell, read from the pair's Float64 buffer U_nb (plan section 4.1 item 3)
        ii,jj,src=CMeshSolverKA._donor3(i,j,kinds,NI,NJ,NIW,NIF,g,ctx)
        if initial==Int32(0) && ii>0
            r64,q64=(ctx!==nothing && src!=Int32(0)) ? (ctx.U_nb[ii,jj,1,src],ctx.U_nb[ii,jj,4,src]) : (U[ii,jj,1],U[ii,jj,4])
            a64=_alpha(q64/r64,Float64(rl),Float64(rv))
            if single==Int32(1)
                good,p32,c32=_phase_state_single(r64,Float32(a64),ep)
                p=Float64(p32);c=Float64(c32)
            else
                good,p,c=_phase_state(r64,a64,Float64(pv),Float64(a2l),Float64(rl),Float64(rv),
                                      Float64(psm),Float64(afloor),Float64(lq))
            end
            good=good && isfinite(q64) && 0<=q64<=r64 && r==Float32(r64) && a==Float32(a64)
        else
            good,p32,c32=_phase_state(r,a,pv,a2l,rl,rv,psm,afloor,lq)
            p=Float64(p32);c=Float64(c32)
            # Characteristic ghosts prescribe physical pressure and obtain a
            # Float32 density by inverse EOS. For a stiff liquid, forward EOS
            # of that rounded density need not recover the prescribed pressure
            # within the primitive pedestal's tolerance. Certify the inverse
            # against its authoritative pressure cache instead. This changes
            # no state, tolerance, interior/donor check, or boundary operator.
            # Unknown outer kinds and initialization retain the old invariant.
            characteristic = initial==Int32(0) && ii==Int32(0) &&
                (i<=g || i>NI+g ||
                 (j>NJ+g && g<i<=NI+g && kinds[i-g]==0x01))
            if characteristic
                pb=pressure[i,j]
                good=good && isfinite(pb) && pb>0.5f0*psm &&
                    CMeshSolverKA.CMeshPressureOutlet._rhomix_vl(pb,a,pv,a2l,rl,rv,psm,lq)==r
                p=Float64(pb)
            end
        end
        good=good && isfinite(u) && isfinite(v) && isfinite(u*u+v*v) &&
             isfinite(t) && t>0.0f0 && isfinite(pexc[i,j]) && isfinite(am[i,j])
        if good && coherent==Int32(1)
            represented=r*CMeshCavTEM.TEM_CS2*t+pexc[i,j]
            tol=64.0f0*eps(Float32)*max(abs(r*CMeshCavTEM.TEM_CS2*t),abs(pexc[i,j]),p)
            good=abs(represented-p)<=tol && abs(am[i,j]-c)<=64.0f0*eps(Float32)*max(c,1.0f-3)
        end
        if good && g<i<=NI+g && g<j<=NJ+g
            ci=i-g;cj=j-g;r64=U[ci,cj,1];q64=U[ci,cj,4];mx=U[ci,cj,2];my=U[ci,cj,3]
            if initial==Int32(1)
                expected=r*_Y(a,rl,rv)
                tol=max(1.0f-12,32.0f0*eps(Float32)*max(abs(q64),abs(expected)))
                good=good && abs(q64-expected)<=tol
            end
            # After initialization alpha is only a rounded derived cache. Its
            # exact prescribed cast was checked above; inverting that cast can
            # amplify error near the vapour endpoint at high density ratio.
            good=good && isfinite(q64) && 0<=q64<=r64 &&
                 isfinite(mx) && isfinite(my) && r==Float32(r64) &&
                 u==Float32(mx/r64) && v==Float32(my/r64)
        end
        status[i,j]=good ? 0.0f0 : 1.0f0
    end
end

@inline function _donor_Y(U,rho,alpha,i,j,NI,NJ,g,rl,rv)
    if g<i<=NI+g && g<j<=NJ+g
        Float32(U[i-g,j-g,4]/U[i-g,j-g,1])
    else
        _Y(alpha[i,j],rl,rv)
    end
end

# Vapor face transport. AUSM faces (half===nothing) carry q with the net mass flux
# and the donor's mass fraction. Kinetic faces carry q with BOTH half streams,
# `Fq = Y_L*M+ + Y_R*M-`, the species moment of the same kinetic populations; on a
# constant-pressure contact this keeps the phase-volume identity that the net-donor
# form cannot (2026-09-10). Y donors are identical in both forms: reconstructed
# face states inside, authoritative U / ghost alpha / wake partner elsewhere.
@kernel function _k_species_flux_i!(Fq,@Const(Fmass),@Const(U),@Const(rho),@Const(alpha),
                                    rl::Float32,rv::Float32,face_states,half,NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI+1 && j<=NJ
        f=Fmass[i,j]; id=f>=0.0f0 ? i+g-1 : i+g
        if half!==nothing
            if face_states!==nothing && 2<=i<=NI
                yl=face_states[i-1,j,7,2]; yr=face_states[i,j,7,1]
            else
                yl=_donor_Y(U,rho,alpha,i+g-1,j+g,NI,NJ,g,rl,rv)
                yr=_donor_Y(U,rho,alpha,i+g,j+g,NI,NJ,g,rl,rv)
            end
            # Explicit fused form: identical rounding on CPU and GPU (no compiler contraction).
            Fq[i,j]=fma(half[i,j,1],yl,half[i,j,2]*yr)
        elseif face_states!==nothing && 2<=i<=NI
            y=f>=0.0f0 ? face_states[i-1,j,7,2] : face_states[i,j,7,1]
            Fq[i,j]=f*y
        else
            Fq[i,j]=f*_donor_Y(U,rho,alpha,id,j+g,NI,NJ,g,rl,rv)
        end
    end
end

@kernel function _k_species_flux_j!(Fq,@Const(Fmass),@Const(U),@Const(rho),@Const(alpha),
                                    rl::Float32,rv::Float32,face_states,half,NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32,ctx)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ+1
        f=Fmass[i,j]; jd=f>=0.0f0 ? j+g-1 : j+g
        if half!==nothing
            if face_states!==nothing && 2<=j<=NJ
                yl=face_states[i,j-1,7,4]; yr=face_states[i,j,7,3]
            elseif j==1 && (i<=NIW || i>NIW+NIF)
                # The wake ghost belongs to a real partner cell: its authoritative Y.
                ip=NI+Int32(1)-i
                yl=Float32(U[ip,1,4]/U[ip,1,1])
                yr=_donor_Y(U,rho,alpha,i+g,j+g,NI,NJ,g,rl,rv)
            else
                yl=_donor_Y(U,rho,alpha,i+g,j+g-1,NI,NJ,g,rl,rv)
                yr=_donor_Y(U,rho,alpha,i+g,j+g,NI,NJ,g,rl,rv)
            end
            Fq[i,j]=fma(half[i,j,1],yl,half[i,j,2]*yr)
        elseif face_states!==nothing && 2<=j<=NJ
            y=f>=0.0f0 ? face_states[i,j-1,7,4] : face_states[i,j,7,3]
            Fq[i,j]=f*y
        elseif j==1 && f>=0.0f0 && (i<=NIW || i>NIW+NIF)
            # The wake ghost belongs to a real partner cell. Use its authoritative
            # partial density, not an alpha round trip that changes face inventory.
            if ctx===nothing
                ip=NI+Int32(1)-i
                Fq[i,j]=f*Float32(U[ip,1,4]/U[ip,1,1])
            else
                # strip topology (plan 4.1 item 3): the interface ghost is the other block's cell; its authoritative
                # U from the pair's buffer, the same Float64 quotient its own block forms
                st,fc=CMeshSolverKA._strip_station(i,NIW,NIF,ctx.block)
                Fq[i,j]=f*Float32(ctx.U_nb[st,1,4,fc]/ctx.U_nb[st,1,1,fc])
            end
        elseif ctx!==nothing && ctx.block==Int32(1) && j==NJ+Int32(1) && f<0.0f0
            # the strip's upper interface with inflow from the upper leg's row-1 cell (the ghost): its authoritative U
            Fq[i,j]=f*Float32(ctx.U_nb[i,1,4,2]/ctx.U_nb[i,1,1,2])
        else
            Fq[i,j]=f*_donor_Y(U,rho,alpha,i+g,jd,NI,NJ,g,rl,rv)
        end
    end
end

# Evaluate the existing SSPRK stage with precise authoritative state. Promote
# each stored face BEFORE subtraction; promoting an already-rounded residual
# cannot recover shared-face telescoping or its small conservative increments.
@inline function _species_divergence64(Fi,Fj,i,j)
    Float64(Fi[i+1,j])-Float64(Fi[i,j])+Float64(Fj[i,j+1])-Float64(Fj[i,j])
end
@kernel function _k_species_trial!(trial,status,@Const(U),@Const(U0),
                                   @Const(Fir),@Const(Fjr),@Const(Fix),@Const(Fjx),
                                   @Const(Fiy),@Const(Fjy),@Const(Fiq),@Const(Fjq),@Const(source),
                                   @Const(dtf),@Const(vol),w::Float32,reactive::Int32,
                                   ep::NTuple{7,Float64},single::Int32,NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        fac=Float64(dtf[i,j])/Float64(vol[i,j]);weight=Float64(w)
        dr=_species_divergence64(Fir,Fjr,i,j)
        dx=_species_divergence64(Fix,Fjx,i,j)
        dy=_species_divergence64(Fiy,Fjy,i,j)
        dq=_species_divergence64(Fiq,Fjq,i,j)-(reactive!=Int32(0) ? Float64(source[i,j]) : 0.0)
        rhs=(dr,dx,dy,dq)
        for k in 1:4
            trial[i,j,k]=U0[i,j,k]+weight*((U[i,j,k]-U0[i,j,k])-fac*rhs[k])
        end
        r=trial[i,j,1];mx=trial[i,j,2];my=trial[i,j,3];qn=trial[i,j,4]
        good=isfinite(r) && r>0 && isfinite(mx) && isfinite(my) && isfinite(qn) && 0<=qn<=r
        if good
            u=mx/r;v=my/r;pv,A,rl,rv,psm,lq,afloor=ep
            branch=single==Int32(1) ? _phase_state_single(r,Float32(_alpha(qn/r,rl,rv)),ep)[1] :
                                      _phase_state(r,_alpha(qn/r,rl,rv),pv,A,rl,rv,psm,afloor,lq)[1]
            good=branch && isfinite(u) && isfinite(v) && isfinite(Float32(u)^2+Float32(v)^2) &&
                 isfinite(Float32(r)) && Float32(r)>0f0
        end
        status[i,j]=good ? 0f0 : 1f0
    end
end

# Commit the already-admitted state once; q is a view of U's fourth component.
@kernel function _k_species_commit!(U,rho,ux,uy,alpha,@Const(trial),rl::Float64,rv::Float64,
                                    NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        for k in 1:4;U[i,j,k]=trial[i,j,k];end
        r=trial[i,j,1]
        rho[i+g,j+g]=Float32(r)
        ux[i+g,j+g]=Float32(trial[i,j,2]/r);uy[i+g,j+g]=Float32(trial[i,j,3]/r)
        alpha[i+g,j+g]=Float32(_alpha(trial[i,j,4]/r,rl,rv))
    end
end

_scalars(x)=Tuple((k,getfield(x,k)) for k in fieldnames(typeof(x))
                 if getfield(x,k) isa Number && k!=:last_global_dt)
# A Float32 in a configuration tag (10 Oct 2026): Julia's shortest round-trip digits with the exponent written `e` (Julia writes a
# Float32 exponent as `f`), so the tag reads as a plain number and parse(Float32, ...) gives the cached value back.
_tagnum(x::Float32)=replace(string(x),r"f(?=-?\d+$)"=>"e")
# Keywords (10 Oct 2026), each defaulting to the tuple every existing positional caller forms: `reboud_form` (1, the density
# form V1 of 20 Sep, named by reboud_n > 0 alone; 2 the composition form V2; 3 the void-fraction form V3, 13 Oct), `reboud_prod`
# (13 Oct: 1 the production-consistent form, its own tag after the Reboud tag; 0 none), `source_model` (0 Merkle; 1 Schnerr-Sauer,
# tag from `ss`). The free-gas tag follows t.lq > 0 (lq is also among the TEM scalars of `base`). `recon_retreat` (15 Oct 2026):
# the reconstruction's retreat code, 0 (halving, no tag) unless a builder passes its cache's code; it never reads the module default.
function _configuration(s,t,reconstruction::Bool=false,kinetic::Bool=false,eos_single::Bool=false,
                        source_beta::Bool=false,reboud_n::Float32=0.0f0,trip::NTuple{2,Float32}=(0.0f0,0.0f0),
                        sst_climit::Float32=-1.0f0,stage_margin::Float32=-1.0f0;
                        reboud_form::Int32=Int32(1),reboud_prod::Int32=Int32(0),source_model::Int32=Int32(0),
                        ss::NTuple{4,Float64}=(0.0,0.0,0.0,0.0),
                        recon_retreat::Int32=Int32(0))
    base=(_scalars(s),_scalars(t))
    limiter=reconstruction ? (base...,s.sst ? :CMeshSpeciesMUSCLPhaseLimiterSSTDiffusionV2 :
                                            :CMeshSpeciesMUSCLPhaseLimiterV1) : base
    # The continuous phase-bound retreat (15 Oct 2026, option H): changes the face states (and so the fluxes and the face cap)
    # of every cell whose faces leave the phase bounds at the limiter's theta; bound right after the limiter it modifies, so a
    # restart never silently switches it. The default (halving) adds nothing: every archived tuple is unchanged.
    recon_retreat in (Int32(0),Int32(1)) || throw(ArgumentError("species recon_retreat code must be 0 or 1"))
    recon_retreat==Int32(1) && !reconstruction && throw(ArgumentError("species recon_retreat=:continuous requires reconstruction=true"))
    spatial=recon_retreat==Int32(1) ? (limiter...,RECON_RETREAT_TAG) : limiter
    # The kinetic face family is bound in the configuration so a restart can never
    # silently switch between AUSM and KFVS faces on the same conserved state.
    faces=kinetic ? (spatial...,:CMeshSpeciesKineticFaceV1) : spatial
    # The fully turbulent closure changes both production and wall omega. Bind
    # its identity before restoring state; old hybrid-SST checkpoints are not
    # compatible. The final tag also identifies authoritative Float64 storage.
    # V2 (2026-09-11): k floor correction with accounting at commit; V1 caches
    # rejected every step once a cell reached the 1e-12 floor.
    model=s.sst ? (faces...,:CMeshSpeciesSST2003mV2) : faces
    state=(model...,:CMeshSpeciesConservedFloat64V1)
    viscous=s.nu_lam>0f0 && s.fullvisc==Int32(1) ? (state...,:CMeshSpeciesViscousPressureEndsV1) : state
    # Single-precision EOS arithmetic (2026-09-12) changes published pressure and
    # sound speed by Float32 rounding; bind it so a restart never mixes precisions.
    precision=eos_single ? (viscous...,:CMeshSpeciesSingleEOSV1) : viscous
    # Beta-weighted phase-change source (2026-09-19): changes the source in every vapour-holding
    # cell away from saturation, so bind it; a restart never silently switches weighting.
    weighted=source_beta ? (precision...,:CMeshSpeciesBetaWeightedSourceV1) : precision
    # Schnerr-Sauer source (10 Oct 2026): a different phase-change source in every cell holding vapour or below p_v; bound with
    # its nuclei density and diameter (the Float32 values the cache's ss holds). The tag's version fixes Cv = Cc = 1 and the
    # 0.01 p_v regularisation (_ss_terms) and, since V2 (13 Oct 2026), the trace level SS_TRACE_Y; V1 (613dabb) had no trace level.
    sourced=source_model==Int32(1) ?
        (weighted...,Symbol("CMeshSpeciesSchnerrSauerV2/n0=",_tagnum(Float32(ss[1])),"/d=",_tagnum(Float32(ss[2])))) : weighted
    # Free gas carried by the liquid (10 Oct 2026, TOPO_SPECIES_FREE_GAS or TOPO_LQ): the TEM's lq, named so that a free-gas
    # file reads as what it is; lq is also a TEM scalar of `base`.
    gassed=t.lq>0.0f0 ? (sourced...,Symbol("CMeshSpeciesFreeGasV1/lq=",_tagnum(t.lq))) : sourced
    # Reboud correction (2026-09-20; on by default since 10 Oct 2026, the void-fraction form V3 since 13 Oct): rescales the
    # published eddy viscosity in every vapour-holding cell (V3, V2) or mixture-density cell (V1); bound with its form and
    # exponent so a restart never silently switches the turbulence treatment; the production-consistent form adds its own tag.
    rtag=reboud_form==Int32(3) ? Symbol("CMeshSpeciesReboudNutV3/n=",reboud_n) :
         reboud_form==Int32(2) ? Symbol("CMeshSpeciesReboudNutV2/n=",reboud_n) : Symbol("CMeshSpeciesReboudNutV1/n=",reboud_n)
    turbulent=reboud_n>0.0f0 ? (reboud_prod==Int32(1) ? (gassed...,rtag,:CMeshSpeciesReboudProductionV1) : (gassed...,rtag)) : gassed
    # All-y+ wall function (2026-09-26): changes the wall shear, the wall cell's omega and k wall flux; the
    # solver scalar wallfun already enters `base`, and the tag names the law.
    walled=s.wallfun==2.0f0 ? (turbulent...,:CMeshSpeciesWallFunctionReichardtV1) : turbulent
    # Prescribed trip (2026-09-30): changes k in the trip strip at every stage; bound with its station and width so a
    # restart never silently switches the boundary layer's transition treatment.
    tripped=trip[2]>0.0f0 ? (walled...,Symbol("CMeshSpeciesTripV1/x=",trip[1],"/w=",trip[2])) : walled
    # Limited SST non-orthogonal diffusion correction (2026-10-02): changes the k and omega diffusion fluxes and their clock
    # wherever a face correction exceeds its bound; bound with the limit so a restart never silently switches it.
    # V2 (2026-10-03): the clock's allowance is charged on the corrected faces only (V1 charged every face). The fluxes
    # are V1's; the clock, and so the step sequence, differs at the cells with an uncorrected face of nonzero coefficient
    # (the j = 1 wake-cut and no-slip wall rows), so V1 checkpoints are refused (_configuration_mismatch names them).
    limited=sst_climit>=0.0f0 ? (tripped...,Symbol("CMeshSpeciesSSTCorrectionLimitV2/lambda=",sst_climit)) : tripped
    # Stage-1 clock margin (2026-10-03): a time-step setting, bound with its value so a restart never silently switches it.
    stage_margin>0.0f0 ? (limited...,Symbol("CMeshSpeciesStageClockMarginV1/m=",stage_margin)) : limited
end
# The refusal message of a configuration mismatch. A checkpoint of the limited correction's first clock
# (CMeshSpeciesSSTCorrectionLimitV1, 2 Oct 2026) is refused like any other mismatch; the message names it, so a V1 file
# offered to this build reads as what it is. Any other mismatch keeps the bare message.
# (10 Oct 2026) With the current tuple: when the saved and current tuples differ ONLY in their Reboud tags (the form tag and, since
# 13 Oct, the production tag), the message names both treatments and the setting that restores the file (an archived checkpoint
# carries no Reboud tag and is refused under the new default; nut_density=:none, driver TOPO_SPECIES_NUT_DENSITY=none, restores it).
_is_reboud_tag(x)=x isa Symbol && startswith(String(x),"CMeshSpeciesReboud")
_is_ss_tag(x)=x isa Symbol && startswith(String(x),"CMeshSpeciesSchnerrSauer")
const _REBOUD_FORM_TEXT=(("CMeshSpeciesReboudNutV3/","the void-fraction-form Reboud correction",:reboud_v3,"reboud_v3"),
                         ("CMeshSpeciesReboudNutV2/","the composition-form Reboud correction of 10 Oct",:reboud_v2,"reboud_v2"),
                         ("CMeshSpeciesReboudNutV1/","the density-form Reboud correction of 20 Sep",:reboud_v1,"reboud_v1"))
function _reboud_treatment_text(tags)
    isempty(tags) && return ("no Reboud correction","nut_density=:none (driver TOPO_SPECIES_NUT_DENSITY=none)")
    names=String.(tags)
    k=findfirst(x->startswith(x,"CMeshSpeciesReboudNut"),names)
    k===nothing && return (join(names," + "),"the treatment these tags name")
    tag=names[k]; n=split(tag,"/n=")[end]; prod=any(==("CMeshSpeciesReboudProductionV1"),names)
    f=findfirst(x->startswith(tag,x[1]),_REBOUD_FORM_TEXT)
    f===nothing && return (join(names," + "),"the treatment these tags name")
    _,text,sym,raw=_REBOUD_FORM_TEXT[f]
    (string(text," ",tag,prod ? " with the SST k production from the corrected nu_t (CMeshSpeciesReboudProductionV1)" : ""),
     string("nut_density=:",sym,", reboud_n=",n,prod ? ", reboud_production=true" : "",
            " (driver TOPO_SPECIES_NUT_DENSITY=",raw,", TOPO_REBOUD_N=",n,prod ? ", TOPO_SPECIES_REBOUD_PRODUCTION=1" : "",")"))
end
function _configuration_mismatch(message::AbstractString,saved,current=nothing)
    saved isa Tuple || return message
    k=findfirst(x->x isa Symbol && startswith(String(x),"CMeshSpeciesSSTCorrectionLimitV1/"),saved)
    k===nothing || return string(message,": the checkpoint carries ",saved[k]," (V1, whose clock charged the limiter allowance on every face, ",
           "the uncorrected wake-cut and wall faces included); this build's limited correction is ",
           "CMeshSpeciesSSTCorrectionLimitV2 (the allowance on the corrected faces only) and V1 checkpoints are not converted")
    current isa Tuple || return message
    # (15 Oct 2026, option H, integration review F5) when the tuples differ ONLY in the continuous-retreat tag, the message names
    # both retreats and the knob; nothing is converted (the face states, and so the trajectory, differ between the two)
    rt(x)=x===RECON_RETREAT_TAG
    if any(rt,saved)!=any(rt,current) && isequal(Tuple(x for x in saved if !rt(x)),Tuple(x for x in current if !rt(x)))
        name(t)=any(rt,t) ? "the continuous phase-bound retreat (CMeshSpeciesContinuousPhaseRetreatV1)" :
                            "the halving phase retreat (no retreat tag)"
        return string(message,": the checkpoint carries ",name(saved),"; this build has ",name(current),
            "; restore the file with the retreat it carries (recon_retreat=:",any(rt,saved) ? "continuous" : "halving",
            ", driver TOPO_SPECIES_RECON_RETREAT=",any(rt,saved) ? "continuous" : "halving",")")
    end
    # (14 Oct 2026, review 2) when the tuples differ ONLY in their Schnerr-Sauer tag (Merkle against Schnerr-Sauer, the trace
    # level's V1 of 613dabb against V2, or other nuclei), the message names both sources; nothing is converted
    ss=filter(_is_ss_tag,collect(saved)); cs=filter(_is_ss_tag,collect(current))
    if ss!=cs && isequal(Tuple(x for x in saved if !_is_ss_tag(x)),Tuple(x for x in current if !_is_ss_tag(x)))
        src(t)=isempty(t) ? "the Merkle source (no Schnerr-Sauer tag)" : string("the Schnerr-Sauer source ",join(String.(t)," + "))
        return string(message,": the checkpoint carries ",src(ss),"; this build has ",src(cs),
            " (a CMeshSpeciesSchnerrSauerV1 file predates the trace level of 13 Oct 2026 and is not converted; other nuclei are another source); ",
            "restore the file with the source model and nuclei it carries, on the code that wrote it for V1")
    end
    sr=filter(_is_reboud_tag,collect(saved)); cr=filter(_is_reboud_tag,collect(current))
    (sr!=cr && length(sr)<=2 && length(cr)<=2 &&
     isequal(Tuple(x for x in saved if !_is_reboud_tag(x)),Tuple(x for x in current if !_is_reboud_tag(x)))) || return message
    have,_=_reboud_treatment_text(cr); was,setting=_reboud_treatment_text(sr)
    string(message,": the checkpoint carries ",was,"; this build has ",have,
           " (the species default since 10 Oct 2026 is the Reboud correction, in its void-fraction form CMeshSpeciesReboudNutV3 since 13 Oct); restore the file with ",setting)
end
function _array_bindings(x)
    names=Tuple(k for k in fieldnames(typeof(x)) if getfield(x,k) isa AbstractArray)
    NamedTuple{names}(map(k->getfield(x,k),names))
end
_bindings(s,t)=(_array_bindings(s),_array_bindings(t))
# Compare fields directly: constructing and iterating the heterogeneous tuples
# on every step used >500 KB of transient host allocation on a production cache.
# The persisted configuration format and all mutation checks stay unchanged.
@generated function _scalars_match(x::X,saved::S) where {X,S<:Tuple}
    names=filter(k->fieldtype(X,k)<:Number && k!=:last_global_dt,fieldnames(X))
    checks=[:(getfield(x,$(QuoteNode(k)))==saved[$i][2]) for (i,k) in enumerate(names)]
    foldr((a,b)->:($a && $b),checks;init=:(true))
end
@generated function _bindings_match(x,b::NamedTuple{names}) where {names}
    foldr((k,rest)->:(getfield(x,$(QuoteNode(k)))===getfield(b,$(QuoteNode(k))) && $rest),names;init=:(true))
end
@generated function _array_layout(x,::Val{names},shape,reference) where {names}
    checks=[:(size(getfield(x,$(QuoteNode(k))))==shape && typeof(getfield(x,$(QuoteNode(k))))===typeof(reference)) for k in names]
    foldr((a,b)->:($a && $b),checks;init=:(true))
end
_geometry(s)=(s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,s.xc,s.yc,
              s.outer_ref_r,s.outer_ref_mx,s.outer_ref_my,s.outer_ref_e)

function _geometry_hash(s)
    io=IOBuffer()
    for a in (_geometry(s)...,s.outer_face_kind)
        host=Array(a)
        print(io,eltype(host),':',size(host),'\n')
        write(io,reinterpret(UInt8,vec(host)))
    end
    bytes2hex(sha256(take!(io)))
end

# Compare static geometry on its own backend: no full-grid host transfer on each
# step. The portable SHA256 is computed only when initializing/restoring a cache.
# 2026-09-12: the comparison is unrolled over the tuple at compile time (a runtime
# tuple index made the GPU kernel about 1.9 ms per step on the foil) and the
# launch range is the longest array's length (previously (NI+1)(NJ+1), which
# already covered every array: xc/yc are NI x NJ). Same bit comparison, same flags.
@inline _geometry_bits_equal(a,b,k)=k<=length(a) ? reinterpret(UInt32,a[k])==reinterpret(UInt32,b[k]) : true
@inline _geometry_equal(current::Tuple{},saved::Tuple{},k)=true
@inline _geometry_equal(current::Tuple,saved::Tuple,k)=
    _geometry_bits_equal(first(current),first(saved),k) && _geometry_equal(Base.tail(current),Base.tail(saved),k)
@kernel function _k_species_geometry!(status,current,saved,@Const(kinds),@Const(oldkinds),n::Int32)
    k=@index(Global,Linear)
    @inbounds if k<=n
        good=_geometry_equal(current,saved,k)
        if k<=length(kinds)
            good=good && kinds[k]==oldkinds[k]
        end
        status[k]=good ? 0.0f0 : 1.0f0
    end
end

# `strip_carrier=true` admits a strip topology's strip carrier (outer_mode 2, plan section 3.2), only through
# _admit_strip (cmesh_species_strip.jl): the same checks, with no channel reference to validate (it has no outer boundary).
function _admit(s,t,wallmode::Int32=Int32(0);strip_carrier::Bool=false)
    CMeshSolverKA.GLOBAL_DT[]==1 || throw(ArgumentError("species transport requires global physical timesteps"))
    s.cav==4 && s.outer_mode==(strip_carrier ? 2 : 1) && s.outlet_mode==1 && s.ff_mix==1 ||
        throw(ArgumentError(strip_carrier ? "the strip carrier requires cav4, the strip interface tags (outer_mode 2) and its mixture pressure outlet" :
                                            "species transport requires cav4, segmented channel and mixture pressure outlets"))
    wallmode in (Int32(0),Int32(1),Int32(2)) && s.wall_flux==(wallmode!=Int32(0)) ||
        throw(ArgumentError("species wallmode must be 0 (transparent, wall_flux=false), 1 (slip) or 2 (noslip); impermeable modes require wall_flux=true"))
    isfinite(s.nu_lam) && s.nu_lam>=0.0f0 ||
        throw(ArgumentError("species laminar kinematic viscosity must be finite and nonnegative"))
    all(iszero,(s.suth,s.wf_larsson,s.sponge_sigma,s.irsj,s.irs_ac,s.sfd,s.acou,
        s.emix,s.latent,s.gam,s.keps)) ||
        throw(ArgumentError("species transport excludes the Larsson term, temperature-dependent viscosity, filters and auxiliary energy"))
    (s.wallfun==0.0f0 || (s.wallfun==2.0f0 && wallmode==Int32(2) && s.sst && s.nu_lam>0.0f0)) ||
        throw(ArgumentError("species wall treatment must be wallfun 0 (wall-resolved) or 2 (all-y+ wall function: no-slip wall, SST, positive viscosity); the legacy log law (1) is refused"))
    t.eos==1 && t.wallis==1 && t.vaploc==1 && t.omega==1 && t.dtboost==1 ||
        throw(ArgumentError("species transport requires local-pressure EOS and unrelaxed unit TEM clock"))
    isfinite(t.kprod) && t.kprod>=0 && isfinite(t.kdest) && t.kdest>=0 ||
        throw(ArgumentError("species phase-change rates must be finite and nonnegative"))
    # lq ADMITTED (2026-09-24, the stiff-liquid wedge remedy): the TIER-3 dissolved-gas liquid
    # compliance is carried by the SAME closed-form local-pressure root the species EOS already
    # evaluates -- every kernel consumes it through outlet_eos slot 6 (_phase_state/_phase_state_single/
    # _pmix_vl[_d]/_rhomix_vl), lq=0 is bitwise legacy (tools/verify_tier3_lq.jl GATE 1, 402,201 states),
    # and the phase-volume diagnostics inversion carries the matching term. It must be finite and
    # nonnegative; every OTHER alternate TEM/thermal/bubble mode stays excluded.
    isfinite(t.lq) && t.lq>=0 ||
        throw(ArgumentError("species transport requires finite nonnegative lq"))
    all(iszero,(t.p3,t.enf,t.src_smooth,t.src_jac,t.advimp,t.gs,t.arecon,
        t.ncg,t.dk_cmin2,t.vapbound,t.rp,t.tvp,t.slip,t.cloudint,t.poly,t.hk,t.wallnuc,
        t.btke,t.pvturb,t.pvvort,t.tfree,t.tmix)) ||
        throw(ArgumentError("species transport excludes alternate TEM/thermal/bubble modes"))
    all(isfinite,s.outlet_eos) && t.rho_l_sat>t.rho_v_sat>0 && t.p_v>0 && t.a2l>0 && t.psm>=0 ||
        throw(ArgumentError("invalid species EOS parameters"))
    t.alpha===s.outlet_alpha && size(t.alpha)==size(s.rho) && eltype(s.rho)==Float32 &&
        typeof(t.alpha)==typeof(s.rho) && KA.get_backend(s.rho)==s.backend ||
        throw(ArgumentError("species solver/TEM arrays must share Float32 shape, backend and outlet ownership"))
    s.g>=3 && s.NI>=s.g && s.NJ>=s.g && size(s.vol)==(s.NI,s.NJ) ||
        throw(ArgumentError("invalid species grid dimensions"))
    _array_layout(s,Val((:rho,:ux,:uy,:T,:pexc,:am,:rk_r,:rk_mx,:rk_my,:rk_E,:yvf)),(s.NI+2s.g,s.NJ+2s.g),s.rho) &&
        _array_layout(s,Val((:dtf,:vol,:xc,:yc,:dc_r,:dc_mx,:dc_my,:dc_e,:outer_ref_r,:outer_ref_mx,:outer_ref_my,:outer_ref_e)),(s.NI,s.NJ),s.rho) &&
        _array_layout(s,Val((:Fi_r,:Fi_mx,:Fi_my,:Fi_e,:Six,:Siy)),(s.NI+1,s.NJ),s.rho) &&
        _array_layout(s,Val((:Fj_r,:Fj_mx,:Fj_my,:Fj_e,:Sjx,:Sjy)),(s.NI,s.NJ+1),s.rho) ||
        throw(ArgumentError("species carrier array shape/backend changed"))
    size(t.alpha_new)==size(t.alpha) && typeof(t.alpha_new)==typeof(s.rho) ||
        throw(ArgumentError("species TEM scratch shape/backend changed"))
    if s.nu_lam>0.0f0 && s.fullvisc==Int32(1)
        _array_layout(s,Val((:gux,:guy,:gvx,:gvy,:gTx,:gTy)),(s.NI,s.NJ),s.rho) ||
            throw(ArgumentError("species viscous gradient shape/backend changed"))
    end
    s.sst && _admit_sst(s)
    strip_carrier ? CMeshSolverKA._validate_channel_compat_ka!(s) : CMeshSolverKA._validate_channel_ka!(s)
    CMeshSolverKA.CMeshPressureOutlet.check_pressure_outlet_tem(s,t)
    nothing
end

function _phase_owned_status!(c;coherent=true,initial=false,status=c.status)
    s,t=c.solver,c.tem; ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g)
    nx,ny=Int32.(size(s.rho))
    _k_species_check!(s.backend)(status,c.conserved.U,s.rho,s.ux,s.uy,s.T,s.pexc,s.am,t.alpha,c.pressure,s.outer_face_kind,
        s.outlet_eos,Int32(coherent),Int32(initial),Int32(c.eos_single),ni,nj,Int32(s.NIW),Int32(s.NIF),g,nx,ny,_strip_ctx(c);ndrange=size(s.rho))
end

function _check_phase_status!(c;coherent=true)
    s,t=c.solver,c.tem; ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g)
    if maximum(c.status)!=0
        # Read only the first rejected cell, only on failure and before rollback.
        # Reporting the actual stored values distinguishes boundary bookkeeping
        # from a nonphysical interior state without weakening either check.
        ij=findfirst(!iszero,Array(c.status));i,j=Tuple(ij)
        at(a,ii=i,jj=j)=only(Array(@view a[ii:ii,jj:jj]))
        values=(rho=at(s.rho),ux=at(s.ux),uy=at(s.uy),T=at(s.T),
                alpha=at(t.alpha),pexc=at(s.pexc),am=at(s.am),
                q=(g<i<=ni+g && g<j<=nj+g ? at(c.q,i-g,j-g) : missing))
        throw(ArgumentError("invalid phase/EOS/cache state at cell ($i,$j)$(_block_label(c)), coherent=$coherent, " *
                            "accepted_steps=$(c.steps), stored=$values"))
    end
end

function _check_state!(c;coherent=true,initial=false)
    _phase_owned_status!(c;coherent,initial)
    c.turbulence===nothing && return _check_phase_status!(c;coherent)
    try
        _check_sst_layout!(c)
    catch
        # No SST write occurred: retain phase precedence over layout errors.
        _check_phase_status!(c;coherent)
        rethrow()
    end
    try
        _sst_owned_status!(c)
    catch
        # A failed launch may have touched scratch; recover phase attribution.
        _phase_owned_status!(c;coherent,initial)
        _check_phase_status!(c;coherent)
        rethrow()
    end
    if maximum(c.status)!=0
        # Both producers write only private status. Reevaluate the unchanged
        # phase inputs only on failure, preserving its first-cell diagnostic.
        _phase_owned_status!(c;coherent,initial)
        _check_phase_status!(c;coherent)
        throw(ArgumentError("invalid or stale authoritative SST state/geometry"))
    end
    true
end

# Deferred variant of _check_state!: the same phase/SST kernels write the shared
# status; its maximum goes to one device slot. Host-only layout errors still
# throw here; the immediate replay re-classifies any failure with its precedence.
# Deferred admission: the status kernels write the flag slot's bank slice; the
# batch reduction at the next batch point forms the slot.
function _check_state_deferred!(c,status;coherent=true)
    _phase_owned_status!(c;coherent,status)
    if c.turbulence!==nothing
        _check_sst_layout!(c)
        _sst_owned_status!(c;status)
    end
    nothing
end

"""Initialize conservative species state, or explicitly restore its checkpoint.

The solver/TEM must already be initialized on its EOS and have seeded pressure
boundaries. A fresh cache checks that input, retains its initial rho/q bits and
publishes derived fields from the precise conserved state without advancing time.
A /2 checkpoint contains authoritative Float64 rho/momentum/q: restore matching
solver/TEM primitives and geometry first. CPU/device transfer is allowed;
configuration or conserved/cache mismatch is rejected. Old formats are not
converted. The wallmode is fixed for the cache lifetime. `kinetic=true` selects
the common-gauge KFVS face (half-stream vapor transport, direct pressure weights,
kinetic clock) on the same admitted states; its identity is bound in the
configuration, so AUSM and kinetic checkpoints are mutually rejected.
`sst_correction_limit=lambda` in [0, 1] selects the limited SST non-orthogonal
diffusion correction (V2 clock: the allowance on the corrected faces only), and
`stage_clock_margin=m` in (0, 1] caps the step's proposal at m times the
stage-1 phase-source, viscous and SST clocks; both default off (negative) and
both are bound in the configuration.
`strip=(solver=ss, tem=ts)` builds a strip-topology pair (cmesh_species_strip.jl,
memory/project_docs/2026-10-05_STRIP_TOPOLOGY_PLAN.md): `s, t` the C block's carrier and TEM, `ss, ts`
the strip's (a KSol from build_strip_metrics, outer_mode 2, with the C block's scalars, and
its TEMState, its pressure outlet bound); the pair's build sequence initialises, projects and
exchanges both blocks before their checks, and the configuration names the topology. Default
nothing: no strip, today's cache bitwise. A strip pair restores only a CMeshSpeciesKA/3
checkpoint (`species_checkpoint` of a pair: the /2 entries and `strip = (conserved,
turbulence)`; restore both carriers' published state first) and a cache without a strip only
/2, each refusal naming the topology; kinetic faces are refused with a strip. Its step
(`step_species_ka!`) runs through either admission (the deferred one by the accumulate form
of plan section 4.6: the strip's own banks reduced atomically into the pair's slots).
`nut_density` (10 Oct 2026; forms of 13 Oct): `:reboud_v3` (the default, the void-fraction form V3
of the Reboud correction), `:reboud_v2` (the composition form of 10 Oct), `:reboud_v1` (the
density form of 20 Sep) or `:none` (no correction, the September path; it restores every archived
checkpoint), exponent `reboud_n`; `reboud_production=true` (13 Oct, opt-in) forms the SST k
production from the corrected nu_t, refused with `:none`.
`source_model` (10 Oct 2026): `:merkle` (the default, the TEM's rates) or `:schnerr_sauer`
with `ss_n0` (nuclei per chord^3) and `ss_dnuc` (nucleus diameter in chords), both required
and refused with Merkle; refused with `source_weight=:beta`. Each is bound in the configuration.
"""
function build_species_ka(s::CMeshSolverKA.KSol,t::CMeshCavTEM.TEMState;
                          checkpoint=nothing,wallmode::Int32=Int32(0),reconstruction::Bool=false,
                          kinetic::Bool=false,eos_precision::Symbol=DEFAULT_EOS_PRECISION[],
                          source_weight::Symbol=DEFAULT_SOURCE_WEIGHT[],
                          nut_density::Symbol=DEFAULT_NUT_DENSITY[],reboud_n::Real=DEFAULT_REBOUD_N[],
                          reboud_production::Bool=false,
                          source_model::Symbol=DEFAULT_SOURCE_MODEL[],ss_n0=nothing,ss_dnuc=nothing,
                          trip_x::Real=0.0,trip_width::Real=DEFAULT_TRIP_WIDTH,
                          reduction_tile::Int=0,sst_correction_limit::Real=-1.0,stage_clock_margin::Real=-1.0,
                          recon_retreat::Union{Nothing,Symbol}=nothing,strip=nothing)
    if strip!==nothing
        # Strip topology (plan section 3.3): what version 1 refuses with a strip, before anything is touched.
        strip isa NamedTuple && keys(strip)==(:solver,:tem) || throw(ArgumentError(
            "strip must be (solver=<the strip's KSol>, tem=<its TEMState>)"))
        # the types here, before the pair's configuration reads the strip carrier's fields (_admit_strip's text)
        strip.solver isa CMeshSolverKA.KSol && strip.tem isa CMeshCavTEM.TEMState || throw(ArgumentError(
            "strip=(solver=..., tem=...) must hold the strip's KSol and TEMState"))
        kinetic && throw(ArgumentError(
            "the strip topology does not support kinetic faces (kinetic=true, TOPO_SPECIES_FLUX=kfvs) in version 1 ($STRIP_TOPOLOGY_TAG): the base wall needs the half-mass zeros on an i face; production is AUSM+-up"))
    end
    # (plan section 4.7) a CMeshSpeciesKA/3 checkpoint restores only into a strip pair and a strip pair only a /3 one; the
    # refusal names the topology before any other comparison (a /2 checkpoint of a cache without a strip passes unchanged)
    checkpoint===nothing || _checkpoint_topology(checkpoint,strip)
    _validate_reduction_tile(reduction_tile)
    eos_precision in (:double,:single) || throw(ArgumentError("species eos_precision must be :double or :single"))
    source_weight in (:alpha,:beta) || throw(ArgumentError("species source_weight must be :alpha or :beta"))
    rn,rform,rprod=_nut_density_setting(nut_density,reboud_n,reboud_production)
    sm,ssp=_source_model_setting(source_model,ss_n0,ss_dnuc,source_weight)
    trip=_trip_setting(s,trip_x,trip_width)
    climit=_sst_climit_setting(s,reconstruction,sst_correction_limit)
    smargin=_stage_margin_setting(stage_clock_margin)
    retreat=_recon_retreat_setting(reconstruction,_recon_retreat_choice(reconstruction,recon_retreat))
    eos_single=eos_precision===:single
    source_beta=source_weight===:beta
    configuration=_configuration(s,t,reconstruction,kinetic,eos_single,source_beta,rn,trip,climit,smargin;
                                 reboud_form=rform,reboud_prod=rprod,source_model=sm,ss=ssp,recon_retreat=retreat)
    # A strip pair appends the strip carrier's and its TEM's scalars and the topology tag (plan section 4.7, critique A
    # 7.2); a cache without a strip keeps the tuple as it is.
    strip===nothing || (configuration=_pair_configuration(configuration,strip.solver,strip.tem))
    checkpoint===nothing || (hasproperty(checkpoint,:configuration) &&
        checkpoint.configuration==configuration) ||
        throw(ArgumentError(_configuration_mismatch("species checkpoint configuration/model mismatch",
            hasproperty(checkpoint,:configuration) ? checkpoint.configuration : nothing,configuration)))
    if checkpoint!==nothing && strip!==nothing
        # a strip pair's CMeshSpeciesKA/3: the schema and both blocks' conserved state (cmesh_species_strip.jl)
        _strip_checkpoint_schema(checkpoint,s,strip.solver,wallmode)
    elseif checkpoint!==nothing
        checkpoint isa NamedTuple && keys(checkpoint)==(:format,:configuration,:geometry_sha256,
            :wallmode,:conserved,:steps,:elapsed,:turbulence) && checkpoint.format=="CMeshSpeciesKA/2" ||
            throw(ArgumentError("species checkpoint format/schema mismatch; old formats are not converted"))
        checkpoint.conserved isa Array{Float64,3} && size(checkpoint.conserved)==(s.NI,s.NJ,4) &&
            all(isfinite,checkpoint.conserved) && checkpoint.wallmode===wallmode &&
            checkpoint.steps isa Int && checkpoint.steps>=0 && checkpoint.elapsed isa Float64 &&
            isfinite(checkpoint.elapsed) && checkpoint.elapsed>=0 ||
            throw(ArgumentError("species checkpoint conserved state/metadata mismatch"))
        U=checkpoint.conserved
        all(U[i,j,1]>0 && 0<=U[i,j,4]<=U[i,j,1] for j in 1:s.NJ,i in 1:s.NI) ||
            throw(ArgumentError("species checkpoint conserved phase bounds"))
    end
    _admit(s,t,wallmode)
    CMeshSolverKA.CMeshPressureOutlet.check_pressure_end_geometry(s)
    # Setup/restore checks the static range once; every owned step compares its
    # exact saved bits in _k_species_geometry! before any state mutation.
    all(isfinite,s.vol) && minimum(s.vol)>0 || throw(ArgumentError("species cell volumes must be finite and positive"))
    ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g)
    z(m,n)=fill!(similar(s.rho,m,n),0.0f0)
    if !s.sst && checkpoint!==nothing && hasproperty(checkpoint,:turbulence) && checkpoint.turbulence!==nothing
        throw(ArgumentError("cannot restore an SST checkpoint with turbulence disabled"))
    end
    turbulence=_build_sst(s,checkpoint)
    packed()=fill!(similar(s.rho,Float64,s.NI,s.NJ,4),0.0)
    conserved=(U=packed(),U0=packed(),Utrial=packed())
    q=@view conserved.U[:,:,4];q0=@view conserved.U0[:,:,4];qtrial=@view conserved.Utrial[:,:,4]
    pressure=z(size(s.rho)...)
    ledger=fill!(similar(s.rho,Float64,21,3),0.0)
    workspace=reconstruction ? _build_reconstruction(s;retreat) : nothing
    faces=kinetic ? _kinetic_workspace(s) : nothing
    admission=fill!(similar(s.rho,UInt32,2*ADMISSION_SLOTS),0x00000000)
    # The strip block (plan section 3.1), admitted and allocated beside the C block, its banks beside the pair's one admission
    # vector (plan 4.6); nothing without one.
    x=strip===nothing ? nothing : build_species_strip(s,t,strip.solver,strip.tem;wallmode,reconstruction,eos_single,
                                                      source_beta,reboud_n=rn,sst_climit=climit,stage_margin=smargin,
                                                      admission,reduction_tile,checkpoint,source_model=sm,ssp,reboud_form=rform,
                                                      reboud_prod=rprod,recon_retreat=retreat)
    backup=x===nothing ? map(similar,_rollback_arrays(s,t,conserved,pressure,turbulence)) :
                         (map(similar,_rollback_arrays(s,t,conserved,pressure,turbulence))...,x.backup...)
    inlet_present=any(==(0x01),Array(s.outer_face_kind))
    readback=_build_readback(admission)
    step_limit=fill!(similar(admission,1),reinterpret(UInt32,Inf32))
    c=SpeciesKA(s,t,wallmode,turbulence,q,q0,qtrial,conserved,z(s.NI+1,s.NJ),
        z(s.NI,s.NJ+1),z(s.NI,s.NJ),z(s.NI,s.NJ),pressure,z(size(s.rho)...),ledger,admission,_build_banks(s;reduction_tile),readback,step_limit,workspace,faces,backup,configuration,_bindings(s,t),
        map(copy,_geometry(s)),copy(s.outer_face_kind),x===nothing ? _geometry_hash(s) : _geometry_hash_pair(s,x.solver),
        inlet_present,eos_single,source_beta,sm,ssp,rn,rform,rprod,trip,climit,smargin,x,0,0.0,0.0,0.0,0.0,true,0,nothing,0,0)
    if x!==nothing
        # The pair's build sequence (plan section 4.2; critique A 9): the strip carrier's ghosts are made admissible and
        # coherent by a first pair projection and exchange before the pair's first check; its final check is below. With a
        # CMeshSpeciesKA/3 checkpoint, the pair's restore (plan 4.7): both blocks' U and the clock, then the same projections.
        checkpoint===nothing ? _strip_build_sequence!(c) : _strip_restore!(c,checkpoint)
        _check_state!(c)
        _check_state!(x)
        return c
    end
    if checkpoint===nothing
        _k_species_init!(s.backend)(conserved.U,s.rho,s.ux,s.uy,t.alpha,t.rho_l_sat,t.rho_v_sat,ni,nj,g;ndrange=(s.NI,s.NJ))
        _check_state!(c;initial=true)
        # Preserve the initial authoritative bits while publishing the precise
        # q/r-derived alpha cache before its first EOS projection.
        copyto!(conserved.Utrial,conserved.U)
        _k_species_commit!(s.backend)(conserved.U,s.rho,s.ux,s.uy,t.alpha,conserved.Utrial,
            Float64(t.rho_l_sat),Float64(t.rho_v_sat),ni,nj,g;ndrange=(s.NI,s.NJ))
        _project!(c)
    else
        saved_wallmode=hasproperty(checkpoint,:wallmode) ? checkpoint.wallmode : Int32(0)
        checkpoint.format=="CMeshSpeciesKA/2" && checkpoint.configuration==c.configuration &&
            saved_wallmode===c.wallmode &&
            size(checkpoint.conserved)==size(conserved.U) && eltype(checkpoint.conserved)==Float64 &&
            checkpoint.steps>=0 && isfinite(checkpoint.elapsed) && checkpoint.elapsed>=0 ||
            throw(ArgumentError("species checkpoint format/configuration/state mismatch"))
        checkpoint.geometry_sha256==c.geometry_sha256 ||
            throw(ArgumentError("species checkpoint geometry/boundary/reference mismatch"))
        copyto!(conserved.U,checkpoint.conserved); c.steps=checkpoint.steps; c.elapsed=checkpoint.elapsed
        # A restored cache enters its first residual published, like a fresh one (rebuild, 22 Sep 2026): the
        # residuals no longer project for themselves. The restart tool then overwrites this with the saved
        # rollback payload, which is the same state.
        _project!(c)
    end
    _check_state!(c)
    c
end

function _check_owner(c)
    _check_owner_host(c)
    _geometry_status!(c)
    maximum(c.status)==0 || throw(ArgumentError("species geometry/boundary/reference arrays changed in place"))
    c.strip===nothing || _strip_geometry_status!(c) # the strip block's geometry likewise (strip topology)
end
# Host-only owner admission (no device read); the geometry comparison follows.
function _check_owner_host(c)
    c.valid || throw(ArgumentError("species cache is invalid after a failed step; restore a complete checkpoint"))
    _scalars_match(c.solver,c.configuration[1]) && _scalars_match(c.tem,c.configuration[2]) &&
        _bindings_match(c.solver,c.bindings[1]) && _bindings_match(c.tem,c.bindings[2]) ||
        throw(ArgumentError("species configuration or bound arrays changed; initialize/restore explicitly"))
    all(a isa AbstractArray{Float64,3} && size(a)==(c.solver.NI,c.solver.NJ,4) &&
        KA.get_backend(a)==c.solver.backend for a in values(c.conserved)) ||
        throw(ArgumentError("species conserved workspace shape/type/backend changed"))
    size(c.ledger)==(21,3) && eltype(c.ledger)==Float64 && KA.get_backend(c.ledger)==c.solver.backend ||
        throw(ArgumentError("species ledger scratch shape/backend changed"))
    size(c.admission)==(2*ADMISSION_SLOTS,) && eltype(c.admission)==UInt32 && KA.get_backend(c.admission)==c.solver.backend ||
        throw(ArgumentError("species admission scratch shape/backend changed"))
    length(c.step_limit)==1 && length(c.readback.device)==_READBACK_WORDS &&
        length(c.readback.host)==_READBACK_WORDS && length(c.readback.admission_only)==2*ADMISSION_SLOTS &&
        size(c.readback.ledger_only)==(21,3) || throw(ArgumentError("species readback scratch shape changed"))
    # Reject configuration changes before the carrier's stale-reference checker
    # can invalidate its channel metadata. Restoring an unchanged cache stays safe.
    _admit(c.solver,c.tem,c.wallmode)
    c.strip===nothing || _check_strip_owner_host(c) # the strip carrier's scalars, bindings and admission (plan 4.7)
    nothing
end
function _geometry_status!(c;status=c.status)
    status===c.status && fill!(c.status,0.0f0) # a bank slice is zero beyond the compared range by construction
    geometry=_geometry(c.solver)
    n=max(maximum(length,geometry),length(c.solver.outer_face_kind))
    n<=length(status) || throw(ArgumentError("species status scratch is smaller than the geometry it must compare"))
    _k_species_geometry!(c.solver.backend)(status,geometry,c.geometry,c.solver.outer_face_kind,
        c.boundary_kinds,Int32(n);ndrange=n)
end

"""Return authoritative conserved state; save complete solver/TEM state too. A strip pair returns its CMeshSpeciesKA/3
record (cmesh_species_strip.jl, plan section 4.7); every other cache CMeshSpeciesKA/2, as before."""
function species_checkpoint(c::SpeciesKA)
    c.strip===nothing || return _species_checkpoint_pair(c)   # resolved at compile time without a strip
    _check_owner(c); _check_state!(c)
    (format="CMeshSpeciesKA/2",configuration=c.configuration,geometry_sha256=c.geometry_sha256,
        wallmode=c.wallmode,conserved=Array(c.conserved.U),steps=c.steps,elapsed=c.elapsed,turbulence=_sst_checkpoint(c.turbulence))
end

function _alpha_bc!(s,t)
    ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g)
    CMeshCavTEM._k_tem_bc!(s.backend,CMeshCavTEM.WG)(t.alpha,ni,Int32(s.NIW),Int32(s.NIF),nj,g;ndrange=(s.NI,s.g))
    CMeshCavTEM._k_tem_bc_outer!(s.backend,CMeshCavTEM.WG)(t.alpha,s.outer_face_kind,ni,nj,g;ndrange=(s.NI,s.g))
    CMeshCavTEM._k_tem_bc_io!(s.backend,CMeshCavTEM.WG)(t.alpha,ni,nj,g;ndrange=(s.NJ+2s.g,s.g))
end

# ---- Fused boundary fill (launch fusion, 2026-09-12) ----------------------------------
# The conservative step applied the legacy ghost sequence `_alpha_bc!` (three
# launches) followed by `applyBC_ka!` (seven launches) at every stage residual
# and at the final fill, and `_alpha_bc!` alone at every projection: 46 launches
# per accepted step of a launch-bound step. Every row kernel of that sequence
# (wake wall, far field, outer slip override, channel auxiliary rows, alpha
# wall/outer) writes only ghost rows and reads only interior rows, except the
# channel auxiliary pressure, which reads the far-field ghost density and
# temperature of its own column and layer; the column kernels (outlet copy,
# channel auxiliary columns, alpha columns) read those ghost rows. One (column,
# layer) kernel therefore reproduces the row sequence per thread in the legacy
# order, with the far-field state recomputed per layer (identical arithmetic),
# and one (row, layer) kernel reproduces the column sequence. The legacy corner
# fill is omitted: the outlet copy overwrote every cell it wrote, in the same
# launch order. The species path never passes the cavitation slip blend
# (cavslip = 0), so the no-slip branch is the plain reflection. The legacy
# kernels and `applyBC_ka!` are unchanged for the other paths; the maintained
# battery asserts that the fused fill equals the legacy sequence bit for bit
# on both backends, on every branch.
const _BC_CS2=CMeshSolverKA.CS2
@inline function _alpha_rows_cell!(alpha,outer_face_kind,i,k,NI::Int32,NIW::Int32,NIF::Int32,NJ::Int32,g::Int32)
    @inbounds begin
        if i<=NIW || i>NIW+NIF                     # wake cut: partner exchange (no sign flip)
            alpha[i+g,g+Int32(1)-k]=alpha[NI+Int32(1)-i+g,k+g]
        else                                       # solid wall: zero-gradient
            alpha[i+g,g+Int32(1)-k]=alpha[i+g,g+k]
        end
        jd=NJ+g+k
        alpha[i+g,jd]=0.0f0                        # far field: pure liquid (legacy _k_tem_bc!)
        face_kind=outer_face_kind[i]               # legacy _k_tem_bc_outer!, always applied
        if face_kind==0x01
            alpha[i+g,jd]=0.0f0
        elseif face_kind>=0x02
            alpha[i+g,jd]=alpha[i+g,NJ+g+Int32(1)-k]
        end
    end
    nothing
end
@kernel function _k_species_bc_rows!(rho,ux,uy,T,pexc,am,alpha,@Const(xc),@Const(yc),@Const(Sjx),@Const(Sjy),
        @Const(outer_face_kind),wallmode::Int32,rho_inf::Float32,uxi::Float32,uyi::Float32,T_inf::Float32,
        gam_v::Float32,xv::Float32,yv::Float32,m2::Float32,betav::Float32,ca::Float32,sa::Float32,drot::Float32,
        ff_mix::Int32,ff_p_ref::Float32,ff_a2m::Float32,ff_Tmin::Float32,ff_Tmax::Float32,
        outer::Int32,with_alpha::Int32,NI::Int32,NIW::Int32,NIF::Int32,NJ::Int32,g::Int32)
    i,k=@index(Global,NTuple)
    @inbounds if i<=NI && k<=g
        jb=g+Int32(1)-k                            # ghost row below the wall/wake cut, layer k
        jd=NJ+g+k                                  # ghost row above the outer boundary, layer k
        # 1. wake wall (legacy _k_bc_wake_wall!, cavslip = 0)
        if i<=NIW || i>NIW+NIF
            ip=NI+Int32(1)-i
            rho[i+g,jb]=rho[ip+g,k+g]; ux[i+g,jb]=ux[ip+g,k+g]; uy[i+g,jb]=uy[ip+g,k+g]; T[i+g,jb]=T[ip+g,k+g]
        elseif wallmode==Int32(0)
            rho[i+g,jb]=rho[i+g,g+k]; ux[i+g,jb]=ux[i+g,g+k]; uy[i+g,jb]=uy[i+g,g+k]; T[i+g,jb]=T[i+g,g+k]
        elseif wallmode==Int32(1)
            Sx=Sjx[i,1]; Sy=Sjy[i,1]; Sm=sqrt(Sx*Sx+Sy*Sy); nxw=Sx/Sm; nyw=Sy/Sm
            u0=ux[i+g,g+k]; v0=uy[i+g,g+k]; un=u0*nxw+v0*nyw
            rho[i+g,jb]=rho[i+g,g+k]; ux[i+g,jb]=u0-2.0f0*un*nxw; uy[i+g,jb]=v0-2.0f0*un*nyw; T[i+g,jb]=T[i+g,g+k]
        else
            u0=ux[i+g,g+k]; v0=uy[i+g,g+k]
            rho[i+g,jb]=rho[i+g,g+k]; T[i+g,jb]=T[i+g,g+k]; ux[i+g,jb]=-u0; uy[i+g,jb]=-v0
        end
        # 2. far field (legacy _k_bc_farfield!, one state per column, written to layer k)
        dux=0.0f0; duy=0.0f0
        if gam_v!=0.0f0
            dx=xc[i,NJ]-xv; dy=yc[i,NJ]-yv
            dxs=dx*ca+dy*sa; dys=-dx*sa+dy*ca
            r2=dxs*dxs+dys*dys+1.0f-6
            Fc=betav/(1.0f0-m2*(dys*dys/r2))
            c=gam_v*Fc/(6.2831853f0*r2)
            dus=c*dys; dvs=-c*dxs
            dux=dus*ca-dvs*sa; duy=dus*sa+dvs*ca
        end
        uxt=uxi+dux; uyt=uyi+duy
        gam_eos=(5.0f0+drot)/(3.0f0+drot); gm1=gam_eos-1.0f0; Rgas=_BC_CS2
        Sx=Sjx[i,NJ+1]; Sy=Sjy[i,NJ+1]; Sm=sqrt(Sx*Sx+Sy*Sy)
        nxb=Sx/Sm; nyb=Sy/Sm; txb=-nyb; tyb=nxb
        rd=rho[i+g,NJ+g]; uxd=ux[i+g,NJ+g]; uyd=uy[i+g,NJ+g]; Td=T[i+g,NJ+g]
        cd_=sqrt(gam_eos*Rgas*max(Td,1.0f-4)); cinf=sqrt(gam_eos*Rgas*T_inf)
        qnd=uxd*nxb+uyd*nyb
        qni=uxt*nxb+uyt*nyb
        rb=rd; uxb=uxd; uyb=uyd; Tb=Td
        if ff_mix==Int32(1)
            am_r=sqrt(ff_a2m); Zc=rho_inf*am_r
            p_d=rd*_BC_CS2*Td+pexc[i+g,NJ+g]
            if qnd<=-am_r
                rb=rho_inf; uxb=uxt; uyb=uyt
                Tbb=ff_p_ref/(rho_inf*_BC_CS2)
                Tb=Tbb<ff_Tmin ? ff_Tmin : (Tbb>ff_Tmax ? ff_Tmax : Tbb)
            elseif qnd<am_r
                Rout=qnd+p_d/Zc
                Rin=qni-ff_p_ref/Zc
                qnb=0.5f0*(Rout+Rin)
                p_b=0.5f0*Zc*(Rout-Rin)
                p_b=p_b<1.0f-8 ? 1.0f-8 : p_b
                rbb=rho_inf+(p_b-ff_p_ref)/ff_a2m
                rb=rbb<1.0f-4 ? 1.0f-4 : rbb
                utb=qnb>=0.0f0 ? (uxd*txb+uyd*tyb) : (uxt*txb+uyt*tyb)
                uxb=qnb*nxb+utb*txb; uyb=qnb*nyb+utb*tyb
                Tbb=p_b/(rb*_BC_CS2)
                Tb=Tbb<ff_Tmin ? ff_Tmin : (Tbb>ff_Tmax ? ff_Tmax : Tbb)
            end
        elseif qnd<=-cd_
            rb=rho_inf; uxb=uxt; uyb=uyt; Tb=T_inf
        elseif qnd<cd_
            Rout=qnd+2.0f0*cd_/gm1
            Rin=qni-2.0f0*cinf/gm1
            qnb=0.5f0*(Rout+Rin)
            cb=0.25f0*gm1*(Rout-Rin); cb=cb<1.0f-4 ? 1.0f-4 : cb
            sB=0.0f0; utb=0.0f0
            if qnb>=0.0f0
                sB=(rd*Rgas*Td)/(rd^gam_eos); utb=uxd*txb+uyd*tyb
            else
                sB=(rho_inf*Rgas*T_inf)/(rho_inf^gam_eos); utb=uxt*txb+uyt*tyb
            end
            rb=(cb*cb/(gam_eos*sB))^(1.0f0/gm1)
            Tb=(sB*rb^gam_eos)/(rb*Rgas)
            uxb=qnb*nxb+utb*txb; uyb=qnb*nyb+utb*tyb
        end
        rho[i+g,jd]=rb; ux[i+g,jd]=uxb; uy[i+g,jd]=uyb; T[i+g,jd]=Tb
        if outer!=Int32(0)
            # 3. segmented-channel slip override (legacy _k_bc_outer_slip!)
            if outer_face_kind[i]>=0x02
                js=NJ+g+Int32(1)-k
                Sx=Sjx[i,NJ+Int32(1)]; Sy=Sjy[i,NJ+Int32(1)]
                Sm=sqrt(Sx*Sx+Sy*Sy); nx=Sx/Sm; ny=Sy/Sm
                u0=ux[i+g,js]; v0=uy[i+g,js]; un=u0*nx+v0*ny
                rho[i+g,jd]=rho[i+g,js]
                ux[i+g,jd]=u0-2.0f0*un*nx
                uy[i+g,jd]=v0-2.0f0*un*ny
                T[i+g,jd]=T[i+g,js]
                pexc[i+g,jd]=pexc[i+g,js]
                am[i+g,jd]=am[i+g,js]
            end
            # 4. channel auxiliary rows (legacy _k_bc_channel_aux_j!)
            if i<=NIW || i>NIW+NIF
                ip=NI+Int32(1)-i
                pexc[i+g,jb]=pexc[ip+g,k+g]
                am[i+g,jb]=am[ip+g,k+g]
            else
                pexc[i+g,jb]=pexc[i+g,g+k]
                am[i+g,jb]=am[i+g,g+k]
            end
            if outer_face_kind[i]==0x01 && ff_mix==Int32(1)
                Sx=Sjx[i,NJ+Int32(1)]; Sy=Sjy[i,NJ+Int32(1)]
                Sm=sqrt(Sx*Sx+Sy*Sy)
                qnd=(ux[i+g,NJ+g]*Sx+uy[i+g,NJ+g]*Sy)/Sm
                am_r=sqrt(ff_a2m)
                if qnd>=am_r
                    pexc[i+g,jd]=pexc[i+g,NJ+g]
                    am[i+g,jd]=am[i+g,NJ+g]
                else
                    rb=rho[i+g,jd]
                    pb=ff_p_ref+ff_a2m*(rb-rho_inf)
                    pexc[i+g,jd]=pb-rb*_BC_CS2*T[i+g,jd]
                    am[i+g,jd]=am_r
                end
            end
        end
        # 5. alpha rows (legacy _k_tem_bc! and _k_tem_bc_outer!)
        with_alpha!=Int32(0) && _alpha_rows_cell!(alpha,outer_face_kind,i,k,NI,NIW,NIF,NJ,g)
    end
end
@kernel function _k_species_bc_cols!(rho,ux,uy,T,pexc,am,alpha,outer::Int32,with_alpha::Int32,NI::Int32,NJ::Int32,g::Int32)
    jj,k=@index(Global,NTuple)                     # jj over all rows incl. ghosts, k in 1..g
    @inbounds if jj<=NJ+2g && k<=g
        il=g+Int32(1)-k; ir=NI+g+k; i1=g+Int32(1); iN=NI+g
        rho[il,jj]=rho[i1,jj]; ux[il,jj]=ux[i1,jj]; uy[il,jj]=uy[i1,jj]; T[il,jj]=T[i1,jj]     # legacy _k_bc_outlet!
        rho[ir,jj]=rho[iN,jj]; ux[ir,jj]=ux[iN,jj]; uy[ir,jj]=uy[iN,jj]; T[ir,jj]=T[iN,jj]
        if outer!=Int32(0)                                                                  # legacy _k_bc_channel_aux_io!
            pexc[il,jj]=pexc[i1,jj]; am[il,jj]=am[i1,jj]; pexc[ir,jj]=pexc[iN,jj]; am[ir,jj]=am[iN,jj]
        end
        if with_alpha!=Int32(0)                                                             # legacy _k_tem_bc_io!
            alpha[il,jj]=alpha[i1,jj]; alpha[ir,jj]=alpha[iN,jj]
        end
    end
end
"""Fill every carrier ghost (and, with `alpha=true`, every alpha ghost) in two launches.

Bitwise the legacy `_alpha_bc!` + `applyBC_ka!(s,wallmode;defer_pressure_outlet=true)`
sequence of the conservative step (`alpha=false`: `applyBC_ka!` alone). The
pressure outlet stays deferred to the mixture pressure boundary, as before.
"""
# Column launches (rebuild, 22 Sep 2026). With a pressure outlet, every i-ghost cell -- corners included -- is
# rewritten by the outlet threads of the fused pressure kernel that always follows these fills, before any
# kernel reads it; the column launches' writes are dead there. They run only when no outlet owns the columns
# (the legacy far-field configuration). The species path requires the outlet, so it never launches them.
_columns_owned_by_outlet(s)=s.outlet_mode!=Int32(0)
function _species_bc!(s,t,wallmode::Int32;alpha::Bool,columns::Bool=!_columns_owned_by_outlet(s))
    ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g)
    outer=Int32(s.outer_mode!=Int32(0)); wa=Int32(alpha)
    _k_species_bc_rows!(s.backend)(s.rho,s.ux,s.uy,s.T,s.pexc,s.am,t.alpha,s.xc,s.yc,s.Sjx,s.Sjy,s.outer_face_kind,
        wallmode,s.rho_inf,s.uxi,s.uyi,s.T_inf,s.gam,s.xv,s.yv,s.m2,s.beta,s.ca,s.sa,s.drot,
        s.ff_mix,s.ff_p_ref,s.ff_a2m,s.ff_Tmin,s.ff_Tmax,outer,wa,ni,Int32(s.NIW),Int32(s.NIF),nj,g;ndrange=(s.NI,s.g))
    columns && _k_species_bc_cols!(s.backend)(s.rho,s.ux,s.uy,s.T,s.pexc,s.am,t.alpha,outer,wa,ni,nj,g;ndrange=(s.NJ+2s.g,s.g))
    nothing
end
# The projection publishes (rebuild, 22 Sep 2026). After every commit -- and once at build -- it fills every
# ghost of the seven primitives from the committed interior and projects p/T/am from U over the whole ghosted
# field, inlet and outlet included. It is the only producer of ghost state on the species path: the stage
# residuals and the post-step check consume it, and nothing between a projection and the next residual writes
# U or an interior primitive, so the fill and the pressure trio the residuals used to re-run on entry
# re-derived the same bits (projection_idempotence_checks in the suite asserts exactly that on the production
# foil). `batched`: the deferred attempt's pressure and outlet status go to the admission vector's next batch;
# the immediate attempt keeps the host checks.
# With a strip bound this is the C block's own projection (its trio skips the leg columns' interface ghosts); the strip
# block's is _project_strip!, and the pair's sequence, ending in the exchange, is _project_pair! (cmesh_species_strip.jl).
function _project!(c;batched::Bool=false)
    s,t=c.solver,c.tem
    _species_bc!(s,t,c.wallmode;alpha=true)
    CMeshSolverKA._mixture_pressure_bc!(s,c.pressure;conservative_sst=s.sst,conserved=c.conserved.U,eos_single=c.eos_single,
        inlet_present=c.inlet_present,pressure_slot=batched ? :batched : nothing,status_slot=batched ? :batched : nothing,
        strip_ctx=_strip_ctx(c))
end

function _source_terms!(c;slot=nothing)
    s,t=c.solver,c.tem
    target=slot===nothing ? c.source_dt : slot     # deferred: the clock slot's bank slice
    _k_species_source!(s.backend)(c.source,target,c.conserved.U,s.vol,Float64.(s.outlet_eos),t.kprod,t.kdest,
        Int32(c.eos_single),Int32(c.source_beta),c.source_model,(c.ss[3],c.ss[4]),Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=(s.NI,s.NJ))
    slot===nothing || return nothing
    cap=minimum(c.source_dt)
    if !(cap>0.0f0)
        i,j=Tuple(findfirst(x->!(x>0.0f0),Array(c.source_dt)))
        at(a,ii=i,jj=j)=only(Array(@view a[ii:ii,jj:jj]))
        values=(q=at(c.q),rho=at(s.rho,i+s.g,j+s.g),
                alpha=at(t.alpha,i+s.g,j+s.g),source=at(c.source),cap=at(c.source_dt))
        throw(ArgumentError("nonfinite or unresolved species phase-change source at cell ($i,$j), " *
                            "accepted_steps=$(c.steps), stored=$values"))
    end
    cap
end

# source_dt is serially reused after any source calculation; source itself is
# untouched so the residual and phase-source ledger retain their stage values.
function _viscous_cap!(c;slot=nothing)
    s=c.solver
    target=slot===nothing ? c.source_dt : slot
    _k_species_viscous_dt!(s.backend)(target,s.rho,s.nut,s.vol,s.xc,s.yc,s.Six,s.Siy,s.Sjx,s.Sjy,s.ux,s.uy,
        s.nu_lam,c.wallmode,_wall_function(s),Int32(s.sst),s.fullvisc,Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF),Int32(s.g),
        _strip_ctx(c);ndrange=(s.NI,s.NJ))
    slot===nothing || return nothing
    cap=minimum(c.source_dt)
    cap>0.0f0 || throw(ArgumentError("invalid laminar face conductance or unresolved viscous clock"))
    cap
end

function _mixture_cap!(c,cfl::Float32;stage::Int=1,slot=nothing)
    k=c.kinetic
    if k!==nothing
        s=c.solver
        target=slot===nothing ? c.source_dt : slot
        _k_species_kinetic_dt!(s.backend)(target,k.rate_i,k.rate_j,s.Fi_r,s.Fj_r,s.rho,s.vol,cfl,
            Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=(s.NI,s.NJ))
        slot===nothing || return nothing
        cap=minimum(c.source_dt)
        if !(isfinite(cap) && cap>0.0f0)
            # A refused kinetic face (nonpositive gauge population or pressure weight)
            # on the accepted base state cannot be repaired by a smaller step; on a
            # later stage the stage state itself depends on dt, so retry by halving.
            stage>1 && throw(SpeciesNumericalRejection(stage,:kinetic_face,s.last_global_dt))
            i,j=Tuple(findfirst(x->!(x>0.0f0),Array(c.source_dt)))
            at(a,ii=i,jj=j)=only(Array(@view a[ii:ii,jj:jj]))
            values=(rho=at(s.rho,i+s.g,j+s.g),ux=at(s.ux,i+s.g,j+s.g),uy=at(s.uy,i+s.g,j+s.g),
                    p=at(c.pressure,i+s.g,j+s.g),am=at(s.am,i+s.g,j+s.g),
                    rate_w=at(k.rate_i,i,j),rate_e=at(k.rate_i,i+1,j),rate_s=at(k.rate_j,i,j),rate_n=at(k.rate_j,i,j+1))
            throw(ArgumentError("refused kinetic face or unresolved kinetic clock at cell ($i,$j), " *
                                "accepted_steps=$(c.steps), stored=$values"))
        end
        return cap
    end
    c.reconstruction===nothing || return _reconstructed_face_cap!(c,c.reconstruction,cfl;slot)
    s=c.solver
    # Identical reference convention to the actual carrier face dispatch.
    a2_liq=s.ff_mix==Int32(1) ? s.ff_a2m : s.am_ref*s.am_ref
    minf2=Float32((s.uxi*s.uxi+s.uyi*s.uyi)/max(a2_liq,1.0f-6))
    isfinite(minf2) && minf2>=0.0f0 || throw(ArgumentError("invalid mixture face reference Mach number"))
    target=slot===nothing ? c.source_dt : slot
    _k_species_face_dt!(s.backend)(target,s.rho,s.ux,s.uy,s.am,c.pressure,s.Fi_r,s.Fj_r,
        s.vol,s.Six,s.Siy,s.Sjx,s.Sjy,cfl,minf2,Int32(s.NI),Int32(s.NJ),Int32(s.g);
        ndrange=(s.NI,s.NJ))
    slot===nothing || return nothing
    cap=minimum(c.source_dt)
    isfinite(cap) && cap>0.0f0 || throw(ArgumentError("invalid AUSM face state or unresolved mixture clock"))
    cap
end

# Retain scalar CPU summation and the GPU scalar sum's mapper/output shape.
# dims returns its device singleton without a host scalar read. Existing tiny
# reduction allocations remain; copy each result into reusable stage storage.
_ledger_sum!(dest,src,::KA.CPU)=fill!(dest,sum(Float64,src))
_ledger_sum!(dest,src,backend)=copyto!(dest,sum(Float64,src;dims=ntuple(identity,ndims(src))))

function _ledger_faces!(c,Fi,Fj,offset,stage)
    l=c.ledger;b=c.solver.backend
    _ledger_sum!(@view(l[offset+1:offset+1,stage]),@view(Fi[1,:]),b)
    _ledger_sum!(@view(l[offset+2:offset+2,stage]),@view(Fi[end,:]),b)
    _ledger_sum!(@view(l[offset+3:offset+3,stage]),@view(Fj[:,1]),b)
    _ledger_sum!(@view(l[offset+4:offset+4,stage]),@view(Fj[:,end]),b)
end

# Fused deterministic stage ledger (2026-09-12, speed-first): one launch replaces the
# nineteen per-stage `sum(; dims)` reductions (each a kernel, an allocation and a
# copy), a second launch after the SST commit replaces the k-floor sum. One
# workgroup per entry accumulates a fixed strided subset in Float64 and reduces it
# through a static tree, so every entry is bitwise reproducible run to run and
# identical on the CPU and GPU backends. The summation association differs from the
# CUDA.jl mapreduce it replaces (documented change of a diagnostic; the ledgers are
# never restored state). Inactive entries keep the attempt's zero fill.
# `accumulate` (strip topology, version 1.5, 7 Oct 2026; memory/project_docs/2026-10-05_STRIP_TOPOLOGY_PLAN.md section 4.8): 0 writes
# each active entry (`=`, every launch of a cache without a strip and the C block's launch of a strip pair: the same
# value, bitwise); 1 adds the entry to the value already there (`+=`, the strip block's launch on its own flux arrays,
# after the C block's on the same stream: one writer per entry per launch, so the sum is race-free on every backend).
const _LEDGER_WG=256
@inline _ledger_entry_length(e,NI,NJ)= (e==1 || e==2 || e==5 || e==6 || e==9 || e==10 || e==13 || e==14) ? NJ :
                                       ((e==3 || e==4 || e==7 || e==8 || e==11 || e==12 || e==15 || e==16) ? NI : NI*NJ)
@inline function _ledger_entry_value(e,i,Fir,Fjr,Fi,Fj,Fik,Fjk,Fiw,Fjw,src,sk,sw,kfl,wpn,NI,NJ)
    @inbounds if e==1; Float64(Fir[1,i]) elseif e==2; Float64(Fir[NI+1,i]) elseif e==3; Float64(Fjr[i,1]) elseif e==4; Float64(Fjr[i,NJ+1])
    elseif e==5; Float64(Fi[1,i]) elseif e==6; Float64(Fi[NI+1,i]) elseif e==7; Float64(Fj[i,1]) elseif e==8; Float64(Fj[i,NJ+1])
    elseif e==9; Float64(Fik[1,i]) elseif e==10; Float64(Fik[NI+1,i]) elseif e==11; Float64(Fjk[i,1]) elseif e==12; Float64(Fjk[i,NJ+1])
    elseif e==13; Float64(Fiw[1,i]) elseif e==14; Float64(Fiw[NI+1,i]) elseif e==15; Float64(Fjw[i,1]) elseif e==16; Float64(Fjw[i,NJ+1])
    elseif e==17; Float64(src[i]) elseif e==18; Float64(sk[i]) elseif e==19; Float64(sw[i]) elseif e==20; Float64(kfl[i])
    else; Float64(wpn[i]) end # e==21: wall-function omega pin (2026-09-26), nonzero only with s.wallfun == 2
end
@kernel function _k_ledger!(ledger,stage::Int32,mask::UInt32,accumulate::Int32,@Const(Fir),@Const(Fjr),@Const(Fi),@Const(Fj),
                            @Const(Fik),@Const(Fjk),@Const(Fiw),@Const(Fjw),@Const(src),@Const(sk),@Const(sw),@Const(kfl),
                            @Const(wpn),NI::Int32,NJ::Int32)
    li=@index(Local,Linear);grp=@index(Group,Linear)
    e=Int32(grp)
    buf=@localmem Float64 (_LEDGER_WG,)
    acc=0.0
    if (mask>>(e-Int32(1)))&0x00000001==0x00000001
        n=_ledger_entry_length(e,NI,NJ)
        i=Int32(li)
        while i<=n
            acc+=_ledger_entry_value(e,i,Fir,Fjr,Fi,Fj,Fik,Fjk,Fiw,Fjw,src,sk,sw,kfl,wpn,NI,NJ); i+=Int32(_LEDGER_WG)
        end
    end
    @inbounds buf[li]=acc
    @synchronize
    li=@index(Local,Linear); @inbounds li<=128 && (buf[li]+=buf[li+128]); @synchronize
    li=@index(Local,Linear); @inbounds li<=64 && (buf[li]+=buf[li+64]); @synchronize
    li=@index(Local,Linear); @inbounds li<=32 && (buf[li]+=buf[li+32]); @synchronize
    li=@index(Local,Linear); @inbounds li<=16 && (buf[li]+=buf[li+16]); @synchronize
    li=@index(Local,Linear); @inbounds li<=8 && (buf[li]+=buf[li+8]); @synchronize
    li=@index(Local,Linear); @inbounds li<=4 && (buf[li]+=buf[li+4]); @synchronize
    li=@index(Local,Linear); @inbounds li<=2 && (buf[li]+=buf[li+2]); @synchronize
    li=@index(Local,Linear);grp=@index(Group,Linear);e=Int32(grp)
    @inbounds if li==1 && (mask>>(e-Int32(1)))&0x00000001==0x00000001
        if accumulate==Int32(1)
            ledger[e,stage]+=buf[1]+buf[2]
        else
            ledger[e,stage]=buf[1]+buf[2]
        end
    end
end
const _LEDGER_CARRIER=0x000000ff; const _LEDGER_SST_FACES=0x0000ff00; const _LEDGER_SOURCE=0x00010000
const _LEDGER_SST_SOURCES=0x00060000; const _LEDGER_KFLOOR=0x00080000; const _LEDGER_WPIN=0x00100000
# After each stage commit: the k floor corrections, and the wall-function omega pin when it is on.
_ledger_post_commit(c)=_LEDGER_KFLOOR | (_wall_function(c.solver)==Int32(1) ? _LEDGER_WPIN : 0x00000000)
function _ledger_fused!(c,stage,mask::UInt32)
    s=c.solver;a=c.turbulence
    Fik,Fjk,Fiw,Fjw,sk,sw,kfl,wpn=a===nothing ? (c.Fi,c.Fj,c.Fi,c.Fj,c.source,c.source,c.source,c.source) :
                                                (a.Fik,a.Fjk,a.Fiw,a.Fjw,a.sk,a.sw,a.kfloor,a.wpin)
    _k_ledger!(s.backend,_LEDGER_WG)(c.ledger,Int32(stage),mask,Int32(0),s.Fi_r,s.Fj_r,c.Fi,c.Fj,Fik,Fjk,Fiw,Fjw,c.source,sk,sw,kfl,wpn,
        Int32(s.NI),Int32(s.NJ);ndrange=21*_LEDGER_WG)
    nothing
end
function _ledger_stage!(c,stage,reactive)
    mask=_LEDGER_CARRIER | (reactive ? _LEDGER_SOURCE : 0x00000000) |
         (c.turbulence!==nothing ? (_LEDGER_SST_FACES | _LEDGER_SST_SOURCES) : 0x00000000)
    _ledger_fused!(c,stage,mask)
end

_ledger_boundary(l,k,stage)=l[k+1,stage]-l[k+2,stage]+l[k+3,stage]-l[k+4,stage]
function _publish_ledger!(c,reactive,l=nothing)
    # One host transfer after final numerical/clock admission. Preserve the old
    # host boundary association, Float64 weights and stage-order additions.
    if l===nothing
        l=c.readback.ledger_only
        copyto!(l,c.ledger)
    end
    a=c.turbulence
    mass=0.0;species=0.0;source=0.0;kb=0.0;wb=0.0;ks=0.0;ws=0.0
    for stage in 1:3
        weight=Float64(2.0f0/3.0f0)*(stage==3 ? 1.0 : 0.25)*Float64(c.solver.last_global_dt)
        mass+=weight*_ledger_boundary(l,0,stage)
        species+=weight*_ledger_boundary(l,4,stage)
        reactive && (source+=weight*l[17,stage])
        if a!==nothing
            kb+=weight*_ledger_boundary(l,8,stage);wb+=weight*_ledger_boundary(l,12,stage)
            ks+=weight*l[18,stage];ws+=weight*l[19,stage]
            # k floor corrections enter the committed state after each stage; the
            # convex SSPRK combination carries them to the step with weights
            # 1/6, 2/3 and 1 (not the flux weights). They are reported in the k source.
            ks+=(stage==1 ? 1.0/6.0 : stage==2 ? 2.0/3.0 : 1.0)*l[20,stage]
            # The wall-function omega pin (2026-09-26) enters the committed state the same way; it is
            # reported in the omega source. Only formed when the wall function is on (bitwise otherwise).
            _wall_function(c.solver)==Int32(1) && (ws+=(stage==1 ? 1.0/6.0 : stage==2 ? 2.0/3.0 : 1.0)*l[21,stage])
        end
    end
    c.last_mass_boundary=mass;c.last_species_boundary=species;c.last_species_source=source
    if a!==nothing
        a.last_k_boundary=kb;a.last_w_boundary=wb;a.last_k_source=ks;a.last_w_source=ws
    end
    nothing
end

function _publish_no_ledger!(c)
    c.last_mass_boundary=NaN;c.last_species_boundary=NaN;c.last_species_source=NaN
    a=c.turbulence
    a===nothing || (a.last_k_boundary=NaN;a.last_w_boundary=NaN;a.last_k_source=NaN;a.last_w_source=NaN)
    nothing
end

function _check_stage!(c,w::Float32;retryable::Bool=false,stage::Int=0,slot=nothing)
    s=c.solver; ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g)
    status=slot===nothing ? c.status : slot        # deferred: the flag slot's bank slice
    slot===nothing && fill!(c.status,0.0f0)
    _k_species_trial!(s.backend)(c.conserved.Utrial,status,c.conserved.U,c.conserved.U0,
        s.Fi_r,s.Fj_r,s.Fi_mx,s.Fj_mx,s.Fi_my,s.Fj_my,c.Fi,c.Fj,c.source,
        s.dtf,s.vol,w,Int32(_reactive(c)),Float64.(s.outlet_eos),Int32(c.eos_single),ni,nj;ndrange=(s.NI,s.NJ))
    slot===nothing || return nothing
    maximum(c.status)==0 || _reject_stage(c,stage,:phase_trial,
        "species stage rejected a bound/floor/nonphysical EOS state; reduce dt from a restored checkpoint",retryable)
end

"""Advance the restricted species/flow system; PC remains the default.

An explicitly reconstructed cache uses geometry-based admitted face states for
carrier and interior species fluxes, with the same RK/source/SST/retry stages.
Its method tag is bound in checkpoint configuration. The reconstruction workspace
is regenerated scratch/static geometry, not additional authoritative state.

All original step_ka! and reconstruction settings remain unchanged. This method
uses constant-state mixture fluxes by default and admitted reconstructed interior
faces when requested. EOS is synchronized at every SSPRK stage, including the
existing TEM evaporation/condensation rate in q units.
A kinetic cache (`build_species_ka(...; kinetic=true)`) evaluates the common-gauge
KFVS face on the same admitted states: vapor moves with both kinetic half streams,
physical pressure enters through the discrete pressure weights, walls/ends/inlet
keep their exact overrides, and the clock is the kinetic operational face rate.
SST scalars keep net-donor advection on the kinetic carrier mass flux (their
transport needs no phase-volume identity). Refused kinetic faces reject the stage.
The conservative path uses actual boundary fluxes without the legacy frozen
freestream-residual forcing. Reference arrays remain part of the bound geometry.
Phase change is explicit and its pressure-feedback clock can reduce the global
step. This is not a stiff-source integrator. An additional actual-face AUSM clock
includes mixture sound speeds, density contrast, pressure/velocity diffusion and
the final outward mass flux. It retains the original prepare_dt cap as a further
guard and checks each stage; it does not certify nonlinear stability or speedup.
Laminar viscosity uses the carrier's
constant kinematic nu and face-averaged density, with an operational dt<=1/(4Lnu)
cap checked after each stage boundary fill. This is not a stability proof for
arbitrary nonorthogonal meshes or a calibrated phase-viscosity mixing law.
Wall modes 1/2 reuse analytic impermeable slip/noslip fluxes. With s.sst=true,
rho*k and rho*omega use the same stage mass flux, density-weighted diffusion and
fully turbulent SST-2003m coefficients and wall values. Reconstructed caches
also reconstruct interior k/omega advection with independent bounded slopes.
Eddy viscosity is algebraic;
there is no iteration relaxation or separate SST clock. A k value that decays
below its 1e-12 admission floor within a stage is restored to the floor at the
commit with the added amount ledgered in the k source (SST-2003m V2); omega
keeps a strict floor. Turbulence extensions,
Wall functions and temperature-dependent viscosity remain excluded. The Reboud eddy-viscosity
correction is on by default since 10 Oct 2026, in its void-fraction form since 13 Oct (`nut_density=:reboud_v3`;
`:none` is the September path, `:reboud_v2` the composition form of 10 Oct, `:reboud_v1` the density form
of 20 Sep, `reboud_production=true` the opt-in production-consistent form; see cmesh_species_sst.jl),
and the phase change is the Merkle source by default or, opt-in, the Schnerr-Sauer source
(`source_model=:schnerr_sauer`, `_phase_source_ss`), through the same clock, admission and ledger.
The carrier temperature is a barotropic pedestal, not a physical energy model.
Numerical trial/late-stage clock rejection restores the exact published state
and retries with half the attempted step, up to eight halvings. A cache built
with `stage_clock_margin=m` proposes m times its stage-1 source, viscous and SST
clocks, so that a slowly falling binding clock passes its stage checks. Exhaustion
throws SpeciesRetryExhausted with the base state, time and ledgers restored and
the cache valid. Unexpected errors restore where possible, invalidate and
propagate. RK/residual/status/TEM alpha_new scratch is not restored. last_retries
and last_rejection describe this call, including exhaustion; they are not
physical-time or source contributions. Backup arrays are allocated once.
"""
function step_species_ka!(c::SpeciesKA,cfl::Float32;max_dt::Float32=Inf32,wallmode::Int32=c.wallmode,
                          admission::Symbol=DEFAULT_ADMISSION[],executor=nothing)
    # Strip topology (plan sections 4.2 and 4.6): a strip pair steps through _step_pair_once! (immediate) or
    # _enqueue_pair_deferred! (deferred, the accumulate admission), both in cmesh_species_strip.jl. S7's stage observer
    # (STRIP_STAGE_OBSERVER, a test instrument) runs host code inside the attempt, which a captured graph would not replay;
    # so does S6's copy observer (STRIP_COPY_OBSERVER, before the S6 fallback's copies, plan 4.3), and so does the fallback's
    # precedence probe (STRIP_FLUX_PROBE, after the C block's flux kernels and after each copy).
    c.strip===nothing || executor===nothing || STRIP_STAGE_OBSERVER[]===nothing || throw(ArgumentError(
        "a device executor cannot run a strip pair's attempt while STRIP_STAGE_OBSERVER[] is set ($STRIP_TOPOLOGY_TAG): a captured graph would not replay the observer; clear it"))
    c.strip===nothing || executor===nothing || STRIP_COPY_OBSERVER[]===nothing || throw(ArgumentError(
        "a device executor cannot run a strip pair's attempt while STRIP_COPY_OBSERVER[] is set ($STRIP_TOPOLOGY_TAG): a captured graph would not replay the observer; clear it"))
    c.strip===nothing || executor===nothing || STRIP_FLUX_PROBE[]===nothing || throw(ArgumentError(
        "a device executor cannot run a strip pair's attempt while STRIP_FLUX_PROBE[] is set ($STRIP_TOPOLOGY_TAG): a captured graph would not replay the probe; clear it"))
    admission in (:immediate,:deferred) || throw(ArgumentError("species admission must be :immediate or :deferred"))
    executor===nothing || admission===:deferred || throw(ArgumentError("a device executor requires deferred admission"))
    wallmode==c.wallmode || throw(ArgumentError("species wallmode is fixed by its cache; rebuild or restore explicitly"))
    isfinite(cfl) && 0<cfl<=CFL_MAX[] && max_dt>0 || throw(ArgumentError("invalid species CFL or time cap"))
    if admission===:deferred
        # Host-only owner admission precedes any state change exactly as below; the
        # geometry and state kernels join the deferred flags and are replayed on failure.
        _check_owner_host(c)
    else
        _check_owner(c); _check_state!(c)
        c.strip===nothing || _check_state!(c.strip)   # the strip block's published state likewise (strip topology)
    end
    isfinite(c.elapsed) && c.elapsed>=0 && 0<=c.steps<typemax(Int) || throw(ArgumentError("invalid species physical clock"))
    metadata=_step_metadata(c)
    previous=(retries=c.last_retries,rejection=c.last_rejection)
    try
        _copy_step_state!(c.backup,_rollback_arrays(c),c.solver.backend;sync=false)
    catch
        c.valid=false
        rethrow()
    end
    c.last_retries=0;c.last_rejection=nothing
    cap=max_dt
    for halvings in 0:8
        c.last_retries=halvings
        try
            if admission===:deferred
                _step_species_replayed!(c,cfl,metadata;max_dt=cap,wallmode,executor)
            else
                _step_species_once!(c,cfl;max_dt=cap,wallmode)
            end
            c.last_retries=halvings
            return nothing
        catch err
            if err isa _PreAttemptFailure
                # Deferred mode only: the owner/state admission that precedes an
                # immediate attempt failed in the replay. The base state is restored,
                # the cache stays valid and the previous retry record is retained,
                # exactly as when the immediate mode throws before its attempt.
                try
                    _restore_step!(c,metadata)
                catch restore_error
                    c.valid=false
                    throw(CompositeException(Any[err.error,restore_error]))
                end
                c.last_retries=previous.retries;c.last_rejection=previous.rejection
                throw(err.error)
            end
            try
                _restore_step!(c,metadata)
            catch restore_error
                c.valid=false
                throw(CompositeException(Any[err,restore_error]))
            end
            if err isa SpeciesNumericalRejection
                c.last_retries=halvings;c.last_rejection=err
                next_dt=0.5f0*err.dt
                if halvings==8 || !(isfinite(next_dt) && 0.0f0<next_dt<err.dt) ||
                   metadata.elapsed+Float64(next_dt)==metadata.elapsed
                    throw(SpeciesRetryExhausted(halvings,err))
                end
                cap=min(max_dt,next_dt)
            else
                c.valid=false
                rethrow()
            end
        end
    end
    error("unreachable species retry state")
end

# One attempt uses the existing RK/source/flux operations. Only the public
# wrapper owns rollback and numerical retries; configuration admission is first.
function _step_species_once!(c::SpeciesKA,cfl::Float32;max_dt::Float32,wallmode::Int32=c.wallmode)
    # a strip pair's attempt (strip topology, cmesh_species_strip.jl); without a strip this test is resolved at compile time
    c.strip===nothing || return _step_pair_once!(c,cfl;max_dt,wallmode)
    s,t=c.solver,c.tem; ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g)
    try
        viscous=s.nu_lam>0.0f0
        # All clocks see the published ghosts and EOS of the last projection (build, previous step or restore).
        _stage_residual!(c,wallmode)
        _check_state!(c;coherent=false)
        CMeshSolverKA.prepare_dt_ka!(s,cfl;max_dt=max_dt)
        # Leave margin for the reconstructed state's later-stage clock. The
        # caller max_dt and every subsequent stage/refusal check are unchanged.
        dt=c.reconstruction===nothing ?
            min(s.last_global_dt,_mixture_cap!(c,cfl)) :
            min(s.last_global_dt,CLOCK_MARGIN[]*_mixture_cap!(c,cfl))
        fill!(s.dtf,dt);s.last_global_dt=dt
        reactive=_reactive(c)
        # The stage-1 clocks of the state this step starts from; with a stage clock margin (opt-in, 3 Oct 2026) the
        # proposal takes that fraction of each, as _k_species_fill_dt!/_deferred_decision do (off: the clock, bitwise).
        if reactive
            dt=min(s.last_global_dt,_stage_clock(c.stage_margin,_source_terms!(c)))
            fill!(s.dtf,dt);s.last_global_dt=dt
        end
        if viscous
            dt=min(s.last_global_dt,_stage_clock(c.stage_margin,_viscous_cap!(c)))
            fill!(s.dtf,dt);s.last_global_dt=dt
        end
        if s.sst
            dt=min(s.last_global_dt,_stage_clock(c.stage_margin,_sst_residual!(c)))
            fill!(s.dtf,dt);s.last_global_dt=dt
        end
        copyto!(c.conserved.U0,c.conserved.U)
        s.sst && _sst_snapshot!(c)
        LEDGER_ENABLED[] && fill!(c.ledger,0.0) # rebuilt for every attempt, never restored as published state
        for (stage,(w,b)) in enumerate(((1.0f0,0.0f0),(0.25f0,0.75f0),(2.0f0/3.0f0,1.0f0/3.0f0)))
            # Stage 1 reuses the admitted state, clocks and residuals evaluated
            # above. Snapshots/dt fills modify none of their inputs; source and
            # SST residual/flux arrays survive the shared source_dt scratch use.
            if stage>1
                _stage_residual!(c,wallmode)
                _check_state!(c;coherent=false)
                cap=_mixture_cap!(c,cfl;stage)
                s.last_global_dt<=cap*(1.0f0+16.0f0*eps(Float32)) ||
                    throw(SpeciesNumericalRejection(stage,:mixture_clock,s.last_global_dt))
            end
            face_states=c.reconstruction===nothing ? nothing : c.reconstruction.states
            half_i=c.kinetic===nothing ? nothing : c.kinetic.half_i
            half_j=c.kinetic===nothing ? nothing : c.kinetic.half_j
            _k_species_flux_i!(s.backend)(c.Fi,s.Fi_r,c.conserved.U,s.rho,t.alpha,t.rho_l_sat,t.rho_v_sat,
                face_states,half_i,ni,nj,g;ndrange=(s.NI+1,s.NJ))
            _k_species_flux_j!(s.backend)(c.Fj,s.Fj_r,c.conserved.U,s.rho,t.alpha,t.rho_l_sat,t.rho_v_sat,
                face_states,half_j,ni,nj,Int32(s.NIW),Int32(s.NIF),g,_strip_ctx(c);ndrange=(s.NI,s.NJ+1))
            if stage>1 && reactive
                cap=_source_terms!(c)
                s.last_global_dt<=cap*(1.0f0+16.0f0*eps(Float32)) ||
                    throw(SpeciesNumericalRejection(stage,:source_clock,s.last_global_dt))
            end
            if stage>1 && viscous
                cap=_viscous_cap!(c)
                s.last_global_dt<=cap*(1.0f0+16.0f0*eps(Float32)) ||
                    throw(SpeciesNumericalRejection(stage,:viscous_clock,s.last_global_dt))
            end
            if stage>1 && s.sst
                cap=_sst_residual!(c)
                s.last_global_dt<=cap*(1.0f0+16.0f0*eps(Float32)) ||
                    throw(SpeciesNumericalRejection(stage,:sst_clock,s.last_global_dt))
            end
            LEDGER_ENABLED[] && _ledger_stage!(c,stage,reactive)
            _check_stage!(c,w;retryable=true,stage)
            if s.sst
                _sst_trial!(c,w;retryable=true,stage)
            end
            _k_species_commit!(s.backend)(c.conserved.U,s.rho,s.ux,s.uy,t.alpha,c.conserved.Utrial,
                Float64(t.rho_l_sat),Float64(t.rho_v_sat),ni,nj,g;ndrange=(s.NI,s.NJ))
            if s.sst
                _sst_commit!(c)
                LEDGER_ENABLED[] && _ledger_fused!(c,stage,_ledger_post_commit(c))
            end
            _project!(c)
        end
        s.sst && _sst_prepare!(c)
        _check_state!(c)
        next_time=c.elapsed+Float64(s.last_global_dt)
        isfinite(next_time) && next_time>c.elapsed || error("species physical clock cannot advance")
        LEDGER_ENABLED[] ? _publish_ledger!(c,reactive) : _publish_no_ledger!(c)
        c.steps+=1; c.elapsed=next_time
    catch
        c.valid=false
        rethrow()
    end
    nothing
end

# ---- Deferred admission (2026-09-12) -------------------------------------------------------
# The immediate attempt above reads a status maximum or clock minimum on the host
# after almost every launch: about sixty stream drains per accepted step on the
# production foil, leaving the GPU idle more than half of the time. The deferred
# attempt issues exactly the same launches in the same order with every status
# flag and clock cap reduced into one device slot vector, fills the global step on
# the device from those slots, and reads the vector once at the end. Admitted
# attempts publish the identical state (every kernel, its inputs and the exact
# Float32 minimum are unchanged). Any flagged failure, or any backend exception
# raised by a launch that consumed a flagged state, restores the base state and
# checks the restored base state first. Recorded numerical failures use the same
# ordered rejection reason and dt directly; uncertain flags/backend errors replay
# with immediate checks. REPLAY_REJECTIONS restores replay for every failure.
struct _PreAttemptFailure <: Exception
    error::Any
end
const _SLOT_GEOMETRY=1; const _SLOT_STATE0=2; const _SLOT_DTFIELD=3; const _SLOT_POST=52
_stage_base(stage)=4+16*(stage-1) # 16 slots per SSPRK stage block (stage 1 holds the pre-stage evaluation)
const _S_SST_PREPARE=1; const _S_RECON=2; const _S_FLUX_PRESSURE=5 # 3, 4: the residual-time trio's, retired 22 Sep 2026
const _S_STATE=6; const _S_MIX=7; const _S_SOURCE=8; const _S_VISC=9; const _S_SST_FACES=10; const _S_SST_CAP=11
const _S_TRIAL=12; const _S_SST_TRIAL=13; const _S_PROJECT_PRESSURE=14; const _S_PROJECT_OUTLET=15
_slot(c,k)=(c.admission,Int32(k)) # flag slot k; the clock helper writes the cap at ADMISSION_SLOTS+k
function _residual_slots(c,base)
    (sst_prepare=c.solver.sst ? _status_slice(c,base+_S_SST_PREPARE) : nothing,
     recon=c.reconstruction===nothing ? nothing : :batched,flux_pressure=:batched)
end

# The residual of one SSPRK stage on the published state (rebuild, 22 Sep 2026). The ghost fill and the
# pressure trio no longer run here: the projection that precedes every residual produced them (see
# _project!). What the stage itself needs: the SST coefficients and ghosts from the committed tk/tom, the
# face reconstruction, and the residual body. One function for the immediate attempt (`slots === nothing`,
# host checks) and the deferred one (device slots), for all three species configurations.
function _stage_residual!(c,wallmode;slots=nothing)
    s=c.solver;r=c.reconstruction
    s.sst && _sst_prepare!(c;slot=slots===nothing ? nothing : slots.sst_prepare)
    r===nothing || _reconstruct!(c,r;slot=slots===nothing ? nothing : slots.recon)
    CMeshSolverKA._raw_residual_body_ka!(s,wallmode,Int32(s.wall_flux),true,c.pressure;
        face_states=r===nothing ? nothing : r.states,kinetic=c.kinetic,inlet_present=c.inlet_present,
        pressure_slot=slots===nothing ? nothing : slots.flux_pressure,store_divergence=false,
        strip_ctx=_strip_ctx(c),outlet_sides=_outlet_sides(c))
    nothing
end

function _residual_deferred!(c,wallmode,base)
    _stage_residual!(c,wallmode;slots=_residual_slots(c,base))
    _check_state_deferred!(c,_status_slice(c,base+_S_STATE);coherent=false)
end

# ---- Batched slot reductions (launch fusion, 2026-09-12) --------------------------------
# Every deferred flag or clock producer used to write the shared status or clock
# scratch and reduce it into its slot with one launch each: 52 launches per
# accepted step. The producers now write their own slice of a status bank
# (ghosted size, one slice per banked flag slot) or of a clock bank (interior
# size, one slice per banked clock slot), and one workgroup-reduced kernel per
# batch point reduces every listed slot at once: one workgroup per slot, a
# strided loop, a static tree and one direct write (the maximum of the 0/1 flags,
# the minimum of the positive-Float32 bit patterns of the clocks; NaN or
# negative caps map to 0, the immediate rejection). Reductions are exact, so the
# slots are bitwise those of the per-slot launches. Each slice has exactly one
# producer, and every producer overwrites the whole range it owns, so the banks
# are zeroed once at allocation and never again. The dt field, the pressure
# field and the outlet status keep their legacy arrays (the legacy functions
# take the `:batched` sentinel and skip their per-slot reduction) and are reduced
# by the same kernel from those arrays. Eight batches per attempt: the
# pre-attempt flags, one per stage residual (with the dt field in stage 1), one
# per stage projection, one for the post checks.
const _BANK_KIND_STATUS=Int32(0); const _BANK_KIND_CLOCK=Int32(1); const _BANK_KIND_DTF=Int32(2)
const _BANK_KIND_PRESSURE=Int32(3); const _BANK_KIND_OUTLET=Int32(4); const _BANK_KIND_RECON=Int32(5)
const _BANK_STATUS_IDS=(_SLOT_GEOMETRY,_SLOT_STATE0,
    (b+o for b in (_stage_base(1),_stage_base(2),_stage_base(3)) for o in (_S_SST_PREPARE,_S_STATE,_S_SST_FACES,_S_TRIAL,_S_SST_TRIAL))...,
    _SLOT_POST+3,_SLOT_POST+4)
const _BANK_CLOCK_IDS=((b+o for b in (_stage_base(1),_stage_base(2),_stage_base(3)) for o in (_S_MIX,_S_SOURCE,_S_VISC,_S_SST_CAP))...,)
const _BATCH_WG=256
const _READBACK_WORDS=2*ADMISSION_SLOTS+2*21*3
function _build_readback(admission)
    host=zeros(UInt32,_READBACK_WORDS)
    flags=@view host[1:2*ADMISSION_SLOTS]
    ledger=reshape(reinterpret(Float64,@view(host[2*ADMISSION_SLOTS+1:end])),21,3)
    (;device=similar(admission,_READBACK_WORDS),host,flags,ledger,
      admission_only=zeros(UInt32,2*ADMISSION_SLOTS),ledger_only=zeros(Float64,21,3))
end
@kernel function _k_pack_readback!(packet,@Const(flags),@Const(ledger))
    i=@index(Global,Linear)
    @inbounds if i<=2*ADMISSION_SLOTS
        packet[i]=flags[i]
    elseif i<=_READBACK_WORDS
        k=i-2*ADMISSION_SLOTS
        bits=reinterpret(UInt64,ledger[(k+1)÷2])
        packet[i]=UInt32((bits>>(isodd(k) ? 0 : 32))&0xffffffff)
    end
end
function _read_attempt!(c)
    r=c.readback
    if LEDGER_ENABLED[]
        copyto!(r.host,r.device)
        return r.flags
    end
    copyto!(r.admission_only,c.admission)
    r.admission_only
end
function _build_banks(s;reduction_tile::Int=0)
    smap=zeros(Int32,ADMISSION_SLOTS); cmap=zeros(Int32,ADMISSION_SLOTS)
    for (n,id) in enumerate(_BANK_STATUS_IDS); smap[id]=Int32(n); end
    for (n,id) in enumerate(_BANK_CLOCK_IDS); cmap[id]=Int32(n); end
    status=fill!(similar(s.rho,size(s.rho)...,length(_BANK_STATUS_IDS)),0.0f0)
    clock=fill!(similar(s.rho,s.NI,s.NJ,length(_BANK_CLOCK_IDS)),0.0f0)
    (;status,clock,smap,cmap,specs=Dict{Any,Any}(),reduction=_build_reduction_workspace(s,reduction_tile))
end
function _status_slice(c,id)
    k=c.banks.smap[id]; k>0 || throw(ArgumentError("flag slot $id is not banked"))
    @view c.banks.status[:,:,k]
end
function _clock_slice(c,id)
    k=c.banks.cmap[id]; k>0 || throw(ArgumentError("clock slot $id is not banked"))
    @view c.banks.clock[:,:,k]
end
# A batch spec is a device Int32 matrix (id, kind, slice) per listed slot.
function _batch_spec(c,ids)
    spec=zeros(Int32,3,length(ids))
    for (n,(id,kind)) in enumerate(ids)
        slice=kind==_BANK_KIND_STATUS ? c.banks.smap[id] : kind==_BANK_KIND_CLOCK ? c.banks.cmap[id] : Int32(0)
        (kind==_BANK_KIND_STATUS || kind==_BANK_KIND_CLOCK) && slice==0 && throw(ArgumentError("slot $id is not banked"))
        spec[1,n]=Int32(id); spec[2,n]=kind; spec[3,n]=slice
    end
    d=similar(c.admission,Int32,3,length(ids)); copyto!(d,spec)
    (;m=d,n=Int32(length(ids)))
end
# The listed slots are exactly those the immediate checks would have evaluated
# for this configuration; an unlisted slot keeps its admitted/unwritten value.
function _batch_specs(c,reactive::Bool,viscous::Bool)
    key=(reactive,viscous)
    haskey(c.banks.specs,key) && return c.banks.specs[key]
    s=c.solver; sst=s.sst; recon=c.reconstruction!==nothing; inlet=c.inlet_present; outlet=s.outlet_mode!=Int32(0)
    S=_BANK_KIND_STATUS; C=_BANK_KIND_CLOCK; P=_BANK_KIND_PRESSURE; O=_BANK_KIND_OUTLET
    residual(b)=vcat(sst ? [(b+_S_SST_PREPARE,S)] : [],recon ? [(b+_S_RECON,_BANK_KIND_RECON)] : [],
        inlet ? [(b+_S_FLUX_PRESSURE,P)] : [],
        [(b+_S_STATE,S),(b+_S_MIX,C)],reactive ? [(b+_S_SOURCE,C)] : [],viscous ? [(b+_S_VISC,C)] : [],
        sst ? [(b+_S_SST_FACES,S),(b+_S_SST_CAP,C)] : [])
    project(b)=vcat([(b+_S_TRIAL,S)],sst ? [(b+_S_SST_TRIAL,S)] : [],inlet ? [(b+_S_PROJECT_PRESSURE,P)] : [],
        outlet ? [(b+_S_PROJECT_OUTLET,O)] : [])
    specs=(;residual=ntuple(k->_batch_spec(c,vcat(k==1 ? [(_SLOT_DTFIELD,_BANK_KIND_DTF)] : [],residual(_stage_base(k)))),3),
        project=ntuple(k->_batch_spec(c,project(_stage_base(k))),3),
        post=_batch_spec(c,vcat(sst ? [(_SLOT_POST+3,S)] : [],[(_SLOT_POST+4,S)])))
    c.banks.specs[key]=specs
end
@inline _bank_cap_bits(x::Float32)=(isnan(x) || x<0.0f0) ? 0x00000000 : reinterpret(UInt32,x)
@inline _batch_ismax(kind::Int32)=kind==_BANK_KIND_STATUS || kind==_BANK_KIND_OUTLET || kind==_BANK_KIND_RECON
@kernel function _k_slot_batch!(A,@Const(sbank),@Const(cbank),@Const(dtf),@Const(p),@Const(outlet),@Const(rstat),@Const(spec),
                                ns::Int32,nc::Int32,ndtf::Int32,np::Int32,nout::Int32,nr::Int32)
    li=@index(Local,Linear); grp=@index(Group,Linear)
    buf=@localmem UInt32 (_BATCH_WG,)
    @inbounds begin
        kind=spec[2,grp]; slice=spec[3,grp]
        v=_batch_ismax(kind) ? 0x00000000 : 0xffffffff
        i=Int32(li)
        if kind==_BANK_KIND_STATUS
            off=(slice-Int32(1))*ns
            while i<=ns; v=max(v,sbank[off+i]!=0.0f0 ? 0x00000001 : 0x00000000); i+=Int32(_BATCH_WG); end
        elseif kind==_BANK_KIND_CLOCK
            off=(slice-Int32(1))*nc
            while i<=nc; v=min(v,_bank_cap_bits(cbank[off+i])); i+=Int32(_BATCH_WG); end
        elseif kind==_BANK_KIND_DTF
            while i<=ndtf; v=min(v,_bank_cap_bits(dtf[i])); i+=Int32(_BATCH_WG); end
        elseif kind==_BANK_KIND_PRESSURE
            while i<=np; v=min(v,_bank_cap_bits(p[i])); i+=Int32(_BATCH_WG); end
        elseif kind==_BANK_KIND_OUTLET
            while i<=nout; v=max(v,outlet[i]!=0.0f0 ? 0x00000001 : 0x00000000); i+=Int32(_BATCH_WG); end
        else
            while i<=nr; v=max(v,rstat[i]!=0.0f0 ? 0x00000001 : 0x00000000); i+=Int32(_BATCH_WG); end
        end
        buf[li]=v
    end
    @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=128 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+128]) : min(buf[li],buf[li+128])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=64 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+64]) : min(buf[li],buf[li+64])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=32 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+32]) : min(buf[li],buf[li+32])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=16 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+16]) : min(buf[li],buf[li+16])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=8 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+8]) : min(buf[li],buf[li+8])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=4 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+4]) : min(buf[li],buf[li+4])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear); @inbounds li<=2 && (buf[li]=_batch_ismax(spec[2,grp]) ? max(buf[li],buf[li+2]) : min(buf[li],buf[li+2])); @synchronize
    li=@index(Local,Linear); grp=@index(Group,Linear)
    @inbounds if li==1
        id=spec[1,grp]; kind=spec[2,grp]
        if _batch_ismax(kind)
            A[id]=max(buf[1],buf[2])
        else
            A[ADMISSION_SLOTS+id]=min(buf[1],buf[2])
        end
    end
end
function _slot_batch!(c,spec)
    s=c.solver; b=c.banks
    rstat=c.reconstruction===nothing ? c.pressure : c.reconstruction.status   # never read without a reconstruction
    if b.reduction===nothing
    _k_slot_batch!(s.backend,_BATCH_WG)(c.admission,b.status,b.clock,s.dtf,c.pressure,s.outlet_status,rstat,spec.m,
        Int32(size(s.rho,1)*size(s.rho,2)),Int32(s.NI*s.NJ),Int32(length(s.dtf)),Int32(length(c.pressure)),
        Int32(length(s.outlet_status)),Int32(length(rstat));ndrange=Int(spec.n)*_BATCH_WG)
    else
        _slot_batch_tiled!(c,spec,rstat)
    end
    nothing
end

include(joinpath(@__DIR__,"cmesh_species_reduction_tiles.jl"))

# Global step from the device slots: min(minimum(dtf),max_dt), then the (factored)
# mixture cap and the enabled source/viscous/SST caps, exactly as the immediate
# attempt forms it on the host (min is exact; the single multiply is IEEE on both).
# smargin: the stage-1 clock margin of the source/viscous/SST caps (3 Oct 2026; <= 0 off, the caps themselves).
@kernel function _k_species_fill_dt!(dtf,@Const(A),@Const(step_limit),factor::Float32,smargin::Float32,base::Int32,
                                     reactive::Int32,viscous::Int32,sst::Int32,NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        cap(k)=reinterpret(Float32,A[ADMISSION_SLOTS+k]) # clock slots hold positive Float32 bit patterns
        dt=min(cap(3),reinterpret(Float32,step_limit[1]))
        dt=min(dt,factor*cap(base+7))
        reactive==Int32(1) && (dt=min(dt,_stage_clock(smargin,cap(base+8))))
        viscous==Int32(1) && (dt=min(dt,_stage_clock(smargin,cap(base+9))))
        sst==Int32(1) && (dt=min(dt,_stage_clock(smargin,cap(base+11))))
        dtf[i,j]=dt
    end
end

# Host evaluation of the slot vector in the order of the immediate checks. Returns
# the admitted global step, a numerical rejection, or nothing (diagnostic replay).
# Unwritten flag slots read 0 (admitted);
# clock and pressure slots are consulted only where the immediate attempt would
# have evaluated them. `margin`: the cache's stage-1 clock margin (stage_margin;
# <= 0 off), applied to the stage-1 source/viscous/SST caps as _k_species_fill_dt! does.
function _deferred_decision(h,max_dt::Float32,recon::Bool,reactive::Bool,viscous::Bool,sst::Bool,inlet::Bool,kinetic::Bool=false;
                            margin::Float32=-1.0f0)
    # h: UInt32 slots; flags at k (0 = admitted), clocks at ADMISSION_SLOTS+k as
    # positive Float32 bit patterns (0 = rejected cap, 0xffffffff = never written).
    flag(k)=h[k]==0x00000000
    cap(k)=reinterpret(Float32,h[ADMISSION_SLOTS+k])
    positive(k)=cap(k)>0.0f0
    clock(k)=isfinite(cap(k)) && cap(k)>0.0f0
    pressure(k)=!inlet || positive(k)
    residual(b)=flag(b+_S_SST_PREPARE) && flag(b+_S_RECON) && pressure(b+_S_FLUX_PRESSURE) && flag(b+_S_STATE)
    b1=_stage_base(1)
    residual(b1) || return nothing
    dt=min(cap(_SLOT_DTFIELD),max_dt)
    isfinite(dt) && dt>0.0f0 || return nothing
    clock(b1+_S_MIX) || return nothing
    dt=min(dt,recon ? CLOCK_MARGIN[]*cap(b1+_S_MIX) : cap(b1+_S_MIX))
    if reactive
        positive(b1+_S_SOURCE) || return nothing
        dt=min(dt,_stage_clock(margin,cap(b1+_S_SOURCE)))
    end
    if viscous
        positive(b1+_S_VISC) || return nothing
        dt=min(dt,_stage_clock(margin,cap(b1+_S_VISC)))
    end
    if sst
        flag(b1+_S_SST_FACES) && positive(b1+_S_SST_CAP) || return nothing
        dt=min(dt,_stage_clock(margin,cap(b1+_S_SST_CAP)))
    end
    tol=1.0f0+16.0f0*eps(Float32)
    for stage in 1:3
        b=_stage_base(stage)
        if stage>1
            residual(b) || return nothing
            if !clock(b+_S_MIX)
                return kinetic ? SpeciesNumericalRejection(stage,:kinetic_face,dt) : nothing
            end
            dt<=cap(b+_S_MIX)*tol || return SpeciesNumericalRejection(stage,:mixture_clock,dt)
            if reactive
                positive(b+_S_SOURCE) || return nothing
                dt<=cap(b+_S_SOURCE)*tol || return SpeciesNumericalRejection(stage,:source_clock,dt)
            end
            if viscous
                positive(b+_S_VISC) || return nothing
                dt<=cap(b+_S_VISC)*tol || return SpeciesNumericalRejection(stage,:viscous_clock,dt)
            end
            if sst
                flag(b+_S_SST_FACES) && positive(b+_S_SST_CAP) || return nothing
                dt<=cap(b+_S_SST_CAP)*tol || return SpeciesNumericalRejection(stage,:sst_clock,dt)
            end
        end
        flag(b+_S_TRIAL) || return SpeciesNumericalRejection(stage,:phase_trial,dt)
        sst && !flag(b+_S_SST_TRIAL) && return SpeciesNumericalRejection(stage,:sst_trial,dt)
        pressure(b+_S_PROJECT_PRESSURE) && flag(b+_S_PROJECT_OUTLET) || return nothing
    end
    flag(_SLOT_POST+3) && flag(_SLOT_POST+4) || return nothing
    dt
end
function _admit_deferred(h,max_dt::Float32,recon::Bool,reactive::Bool,viscous::Bool,sst::Bool,inlet::Bool;margin::Float32=-1.0f0)
    result=_deferred_decision(h,max_dt,recon,reactive,viscous,sst,inlet;margin)
    result isa Float32 ? result : nothing
end

# Enqueue one speculative deferred attempt without reading or publishing on the
# host. The finalizer admits it; a rejected attempt is restored by the wrapper.
function _enqueue_species_deferred!(c::SpeciesKA,cfl::Float32;wallmode::Int32)
    # a strip pair's deferred attempt (strip topology, cmesh_species_strip.jl; the graph executor captures this function, so
    # it captures the pair's); without a strip this test is resolved at compile time
    c.strip===nothing || return _enqueue_pair_deferred!(c,cfl;wallmode)
    s,t=c.solver,c.tem; ni,nj,g=Int32(s.NI),Int32(s.NJ),Int32(s.g)
    A=c.admission
    fill!(@view(A[1:ADMISSION_SLOTS]),0x00000000)                    # flags: 0 = admitted
    fill!(@view(A[ADMISSION_SLOTS+1:2*ADMISSION_SLOTS]),0xffffffff)  # clocks: minimum of positive Float32 bit patterns
    viscous=s.nu_lam>0.0f0
    reactive=_reactive(c)
    recon=c.reconstruction!==nothing
    specs=_batch_specs(c,reactive,viscous)
    # Rebuild (22 Sep 2026): the per-step geometry tamper guard and the pre-attempt state check are
    # gone. Geometry is immutable during a run (it is still verified at checkpoint and restore), and the
    # state entering an attempt is the one the previous attempt's post check admitted, or a restored
    # backup that was admitted when it was taken. Removing them also removes the mid-step host read that
    # drained the stream five launches into every step.
    base=_stage_base(1)
    _residual_deferred!(c,wallmode,base)
    CMeshSolverKA.prepare_dt_ka!(s,cfl;slot=:batched)
    _mixture_cap!(c,cfl;slot=_clock_slice(c,base+_S_MIX))
    reactive && _source_terms!(c;slot=_clock_slice(c,base+_S_SOURCE))
    viscous && _viscous_cap!(c;slot=_clock_slice(c,base+_S_VISC))
    s.sst && _sst_residual!(c;slot=_clock_slice(c,base+_S_SST_CAP),faces_slot=_status_slice(c,base+_S_SST_FACES))
    _slot_batch!(c,specs.residual[1])
    _k_species_fill_dt!(s.backend)(s.dtf,A,c.step_limit,recon ? CLOCK_MARGIN[] : 1.0f0,c.stage_margin,Int32(base),
        Int32(reactive),Int32(viscous),Int32(s.sst),ni,nj;ndrange=(s.NI,s.NJ))
    copyto!(c.conserved.U0,c.conserved.U)
    s.sst && _sst_snapshot!(c)
    LEDGER_ENABLED[] && fill!(c.ledger,0.0)
    for (stage,(w,b)) in enumerate(((1.0f0,0.0f0),(0.25f0,0.75f0),(2.0f0/3.0f0,1.0f0/3.0f0)))
        base=_stage_base(stage)
        if stage>1
            _residual_deferred!(c,wallmode,base)
            _mixture_cap!(c,cfl;stage,slot=_clock_slice(c,base+_S_MIX))
        end
        face_states=c.reconstruction===nothing ? nothing : c.reconstruction.states
        half_i=c.kinetic===nothing ? nothing : c.kinetic.half_i
        half_j=c.kinetic===nothing ? nothing : c.kinetic.half_j
        _k_species_flux_i!(s.backend)(c.Fi,s.Fi_r,c.conserved.U,s.rho,t.alpha,t.rho_l_sat,t.rho_v_sat,
            face_states,half_i,ni,nj,g;ndrange=(s.NI+1,s.NJ))
        _k_species_flux_j!(s.backend)(c.Fj,s.Fj_r,c.conserved.U,s.rho,t.alpha,t.rho_l_sat,t.rho_v_sat,
            face_states,half_j,ni,nj,Int32(s.NIW),Int32(s.NIF),g,_strip_ctx(c);ndrange=(s.NI,s.NJ+1))
        if stage>1
            reactive && _source_terms!(c;slot=_clock_slice(c,base+_S_SOURCE))
            viscous && _viscous_cap!(c;slot=_clock_slice(c,base+_S_VISC))
            s.sst && _sst_residual!(c;slot=_clock_slice(c,base+_S_SST_CAP),faces_slot=_status_slice(c,base+_S_SST_FACES))
            _slot_batch!(c,specs.residual[stage])
        end
        LEDGER_ENABLED[] && _ledger_stage!(c,stage,reactive)
        _check_stage!(c,w;retryable=true,stage,slot=_status_slice(c,base+_S_TRIAL))
        s.sst && _sst_trial!(c,w;retryable=true,stage,slot=_status_slice(c,base+_S_SST_TRIAL))
        _k_species_commit!(s.backend)(c.conserved.U,s.rho,s.ux,s.uy,t.alpha,c.conserved.Utrial,
            Float64(t.rho_l_sat),Float64(t.rho_v_sat),ni,nj,g;ndrange=(s.NI,s.NJ))
        if s.sst
            _sst_commit!(c)
            LEDGER_ENABLED[] && _ledger_fused!(c,stage,_ledger_post_commit(c))
        end
        _project!(c;batched=true)
        _slot_batch!(c,specs.project[stage])
    end
    s.sst && _sst_prepare!(c;slot=_status_slice(c,_SLOT_POST+3))
    _check_state_deferred!(c,_status_slice(c,_SLOT_POST+4))
    _slot_batch!(c,specs.post)
    LEDGER_ENABLED[] && _k_pack_readback!(s.backend)(c.readback.device,A,c.ledger;ndrange=_READBACK_WORDS)
    nothing
end
function _finish_species_deferred!(c,max_dt::Float32)
    s,t=c.solver,c.tem
    recon=c.reconstruction!==nothing;reactive=_reactive(c);viscous=s.nu_lam>0.0f0
    h=_read_attempt!(c) # one preallocated host read, including accounting when enabled
    dt=_admit_deferred(h,max_dt,recon,reactive,viscous,s.sst,c.inlet_present;margin=c.stage_margin)
    dt===nothing && return false
    next_time=c.elapsed+Float64(dt)
    isfinite(next_time) && next_time>c.elapsed || return false
    s.last_global_dt=dt
    c.strip===nothing || (c.strip.solver.last_global_dt=dt)   # a strip pair: the strip carrier holds the pair's step too
    LEDGER_ENABLED[] ? _publish_ledger!(c,reactive,c.readback.ledger) : _publish_no_ledger!(c)
    c.steps+=1; c.elapsed=next_time
    c.strip===nothing || (c.strip.steps=c.steps)
    true
end
function _step_species_deferred!(c::SpeciesKA,cfl::Float32;max_dt::Float32,wallmode::Int32,executor=nothing)
    fill!(c.step_limit,reinterpret(UInt32,max_dt))
    if executor===nothing
        _enqueue_species_deferred!(c,cfl;wallmode)
    else
        executor(c,cfl,wallmode)
    end
    _finish_species_deferred!(c,max_dt)
end

function _step_species_replayed!(c::SpeciesKA,cfl::Float32,metadata;max_dt::Float32,wallmode::Int32,executor=nothing)
    failure=nothing
    ok=try
        _step_species_deferred!(c,cfl;max_dt,wallmode,executor)
    catch err
        err isa InterruptException && rethrow()
        failure=err
        false
    end
    ok && return nothing
    _restore_step!(c,metadata)
    try
        _check_owner(c); _check_state!(c)
        c.strip===nothing || _check_state!(c.strip)   # a strip pair: the strip block's restored state likewise
    catch err
        throw(_PreAttemptFailure(err))
    end
    if failure===nothing && !REPLAY_REJECTIONS[]
        h=LEDGER_ENABLED[] ? c.readback.flags : c.readback.admission_only
        t=c.tem;s=c.solver
        rejection=_deferred_decision(h,max_dt,c.reconstruction!==nothing,_reactive(c),
                                     s.nu_lam>0f0,s.sst,c.inlet_present,c.kinetic!==nothing;margin=c.stage_margin)
        # a strip pair (strip topology, version 1.5): a trial rejection names the strip when its own trial failed (plan 4.6;
        # _strip_attribute, cmesh_species_strip.jl); the decision itself is unchanged
        c.strip===nothing || (rejection=_strip_attribute(h,rejection))
        if rejection isa SpeciesNumericalRejection
            c.fast_rejections+=1
            throw(rejection)
        end
    end
    c.diagnostic_replays+=1
    _step_species_once!(c,cfl;max_dt,wallmode)
    error("deferred species admission flagged a failure that the immediate replay did not reproduce"*
          (failure===nothing ? "" : "; deferred exception: "*sprint(showerror,failure)))
end

# Steady pseudo-time mode (speed-up plan W2, 30 Sep 2026): opt-in local time stepping for wetted steady targets.
# Definitions only; nothing above calls it (memory/project_docs/CAVITATION_CODE_MAP.md).
include(joinpath(@__DIR__,"cmesh_species_steady.jl"))
# Strip trailing-edge topology (5 Oct 2026, memory/project_docs/2026-10-05_STRIP_TOPOLOGY_PLAN.md): the second block, its data model and
# its coupling; reached only through build_species_ka(...; strip=...) (memory/project_docs/CAVITATION_CODE_MAP.md).
include(joinpath(@__DIR__,"cmesh_species_strip.jl"))
end
