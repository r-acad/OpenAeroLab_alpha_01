using YAML
using Printf

if !isdefined(@__MODULE__, :CaseIDContract)
    include(joinpath(@__DIR__, "case_id_contract.jl"))
end

global CFG = Dict()
global CASE_DIR = ""
global STL_FILENAME = ""
global STL_FILE = ""
global STL_FILENAMES = String[]    # All STL filenames (always a vector)
global STL_FILES = String[]        # Full paths to all STL files
global STL_SCALE = 1.0
global OUT_DIR_NAME = "RESULTS"
global OUT_DIR = ""
global SURFACE_RESOLUTION = 200
global NUM_LEVELS_CONFIG = 0
global MINIMUM_FACET_SIZE = 0.0
# Explicit STL subdivision on/off switch (2026-06-04). Default true preserves
# legacy behaviour (subdivision driven by minimum_facet_size). When set false,
# subdivision is FORCED off regardless of minimum_facet_size — the raw STL
# triangles are used as-is. Essential for huge meshes (e.g. the A350) where
# subdivision is the dominant, unnecessary cost. Set via basic.subdivision.
global SUBDIVISION_ENABLED = true

global SYMMETRY_MODE = :off
global SYMMETRIC_ANALYSIS = false
global HALF_Y_DOMAIN = false
global GROUND_ANALYSIS = false
global GROUND_UP_AXIS = :z
global ALPHA_DEG = 0.0
global BETA_DEG = 0.0
global REFERENCE_AREA_FULL_MODEL = 0.0
global REFERENCE_AREA_CONFIG = 0.0
global REFERENCE_CHORD_CONFIG = 0.0
global REFERENCE_LENGTH_FOR_MESHING = 0.0
global REFERENCE_DIMENSION = :x

global FLUID_DENSITY = 1.225
global FLUID_KINEMATIC_VISCOSITY = 1.5e-5
global FLOW_VELOCITY = 10.0

global STEPS = 1000
global RAMP_STEPS = 1500
global RAMP_PROFILE = :half_cosine
global RAMP_START_FRACTION = 0.0f0
global OUTPUT_FREQ = 100
global SLICE_OUTPUT_FREQ = 100
global SLICE_COSMETIC_FILTERING = false  # Default to no spatial smoothing; contrast remaps in derived views like schlieren are still allowed.
# Per-iteration slice field-set. :production (lean, default) or :all (full
# debug set — every field × every plane). Set `advanced.slice.field_set: "all"`
# for solver-debug runs. 2026-05-29.
global SLICE_FIELD_SET = :production
global DOMAIN_LENGTH_COVERED = 0.0  # 0.0 = use 'steps' directly; >0 = compute steps from domain traversals

global OUTPUT_DENSITY = true
global OUTPUT_VELOCITY = true
global OUTPUT_VEL_MAG = true
global OUTPUT_VORTICITY = true
global OUTPUT_OBSTACLE = true
global OUTPUT_LEVEL = true
global OUTPUT_BOUZIDI = true
# Pass174c: extend VTK to mirror slice fields. mach + pressure + wall_dist are
# cheap derived/lookup fields; schlieren needs a per-cell |∇ρ| gradient that
# falls back to one-sided diff at block edges (so it has small artefacts at
# AMR boundaries, like the slice version).
global OUTPUT_MACH = true
global OUTPUT_PRESSURE = true
global OUTPUT_WALL_DISTANCE = true
global OUTPUT_SCHLIEREN = true
global OUTPUT_STARTUP_SLICES = true
global OUTPUT_STARTUP_VOLUME = true

global U_TARGET = 0.08f0
# Ramped inlet velocity for the current batch. Updated by main.jl before
# each timestep batch. Read by _post_restrict_sponge_enforce! so that the
# AMR-sponge target respects the ramp (otherwise sponge slams sponge cells
# to full freestream speed on step 1, creating a large velocity discontinuity
# that prints as corner artifacts in the outlet+wall region).
const U_INLET_CURRENT = Ref{Float32}(0.0f0)
global C_SMAGO = 0.20f0
global TAU_MIN = 0.501f0
global TAU_SAFETY_FACTOR = 1.0f0
global INLET_TURBULENCE_INTENSITY = 0.01f0
global COLLISION_OPERATOR = :cumulant

global NU_SGS_BACKGROUND = 0.0f0
global SPONGE_BLEND_DISTRIBUTIONS = false
global TEMPORAL_INTERPOLATION = true
global INTERFACE_MACRO_TEMPORAL_INTERPOLATION = false
# ── B1 Pillar 3 (2026-05-28) — Cubic Hermite (C²) temporal interpolation ──
# When true, the AMR halo prolongation uses a 4-anchor cubic Lagrange
# polynomial in time {oldest, older, old, new} at times {-2dt_c, -dt_c, 0,
# +dt_c} evaluated at the fine substep time tw·dt_c. Raises temporal
# continuity from C¹ (quadratic, Pass176/180) to C². Requires allocation
# of `rho_oldest`, `vel_oldest`, `temperature_oldest` arrays in BlockLevel
# (gated on this flag — see src/geometry/blocks.jl). Off by default.
# The temporal-interpolation piece of the SOTA_ROADMAP "STANDING DIRECTIVE
# — absolute best-in-class AMR interface treatment" Pillar 3. The other
# half (RK4-consistent sub-cycling with parent_history depth 4) is a
# separate, larger refactor and remains future work.
global INTERFACE_CUBIC_TEMPORAL = false
# B1 P3 RK4-equivalent — quintic Lagrange (C³) temporal interpolation. Adds a
# 5th historical anchor `*_oldest2` at t = -3·dt_c, raising temporal continuity
# at AMR halos from C² (cubic) to C³ (quintic). Requires INTERFACE_CUBIC_TEMPORAL
# (it shares the anchor chain) AND INTERFACE_MACRO_TEMPORAL_INTERPOLATION. Off by
# default — when false the *_oldest2 arrays stay 1×1×1×1 stubs and every kernel's
# use_quintic_temporal flag is 0, so the quintic branch is skipped (byte-identical).
global INTERFACE_QUINTIC_TEMPORAL = false
# ── HIGH-ORDER AMR-CONTINUITY SCHEDULE (2026-06-23) ──────────────────────────
# The C4 B-spline (space) + C3 quintic-Lagrange (time) AMR-interface interpolation
# gives invisible seams but is ~3-4× slower per step (B-spline prefilter + 4×4×4
# stencil + 5-anchor temporal reads, every step). The high-order continuity only
# MATTERS once the flow has developed gradients across the interface — during an
# impulsive/ramping transient the cheap C0 path is fine. This schedule lets the
# solver run cheap C0 during spin-up and ESCALATE to gold-standard C4/C3 toward
# steady state (or use it always / never). Modes:
#   :always       — C4/C3 from step 0 (max continuity throughout; for unsteady /
#                   time-accurate runs with no steady state). Legacy behavior.
#   :off          — never escalate; pure C0 trilinear (fastest).
#   :steady_state — C0 during the ramp, switch to C4/C3 once u_curr/U_TARGET ≥
#                   the threshold (ramp essentially complete) AND past a settle
#                   delay. Best of both: fast spin-up + invisible seams at convergence.
# Only gates the C4/C3 ESCALATION; the underlying feature flags
# (interface_cubic/quintic_temporal, quintic_bspline_prolongation) must still be on.
global INTERFACE_HIGH_ORDER_SCHEDULE = :always
global INTERFACE_HIGH_ORDER_U_THRESHOLD = 0.98f0  # u_curr/U_TARGET to switch on (steady_state mode)
global INTERFACE_HIGH_ORDER_SETTLE_STEPS = Int32(0)  # extra steps after threshold before switching

# Shared gate (used by both timestep.jl per-step flags AND the solver_control.jl
# B-spline prefilter call) — is the high-order C4/C3 path active RIGHT NOW given
# the current ramp velocity? The velocity-ratio criterion is the primary signal
# (once u/U_target ≥ threshold the ramp is essentially complete). The per-step
# settle-steps refinement is applied additionally in timestep.jl where the step
# counter is in scope; the prefilter uses the velocity criterion alone (it is a
# coarse-step amortized op, so a few extra prefilter calls before the exact
# switch step are negligible).
@inline function high_order_continuity_active(u_curr::Float32)
    INTERFACE_HIGH_ORDER_SCHEDULE === :always && return true
    INTERFACE_HIGH_ORDER_SCHEDULE === :off && return false
    # :steady_state
    return U_TARGET > 0.0f0 && (u_curr / U_TARGET) >= INTERFACE_HIGH_ORDER_U_THRESHOLD
end

global INTERFACE_TEMPORAL_WEIGHT_BIAS = 0.0f0
global INTERFACE_TEMPORAL_WEIGHT_BIAS_RISE_U_START = 0.0f0
global INTERFACE_TEMPORAL_WEIGHT_BIAS_RISE_U_END = 0.0f0
global INTERFACE_TEMPORAL_WEIGHT_BIAS_DECAY_U_START = 1.0f0
global INTERFACE_TEMPORAL_WEIGHT_BIAS_DECAY_U_END = 1.0f0
global RESTRICT_FINE_TO_PARENT = true
global INTERFACE_ENTROPIC_H_SCALE = 0.0f0
global INTERFACE_FILTERING = false

global CUMULANT_OMEGA_BULK      = 1.0f0
global CUMULANT_OMEGA_3         = 1.0f0
global CUMULANT_OMEGA_4         = 1.0f0
global CUMULANT_ADAPTIVE_OMEGA4 = true
global CUMULANT_LAMBDA_PARAM    = Float32(1.0/6.0)
global CUMULANT_LIMITER         = :factored

const VALID_CUMULANT_LIMITERS = Set([:none, :factored, :positivity])

# Entropic KBC parameters
global ENTROPIC_ALPHA_SOLVER = :taylor      # :taylor (default), :bisection, :newton, :approximate
global ENTROPIC_ALPHA_MAX_ITER = 10         # Max iterations for alpha solver
global ENTROPIC_ALPHA_TOLERANCE = 1.0f-5    # Convergence tolerance (newton only)
global ENTROPIC_SIGMA = 0.0f0              # h-component damping: 0=off (pure KBC), 1=full regularization

const VALID_ENTROPIC_SOLVERS = Set([:bisection, :newton, :approximate, :taylor])

# Lattice type: D3Q27 (default) or D3Q39 (multi-speed for supersonic)
global LATTICE_TYPE = :d3q27
global N_DIRS = Int32(27)
global NUMERICAL_EQUILIBRIUM = false

# PonD per-cell co-moving conservative kinetic-FV solver (path A; opt-in, default OFF).
# Enable via `advanced.numerics.pond.enabled: true`. When off, every existing case is
# byte-identical. The PonD FV path evolves conserved MOMENTS (ρ,ρu) via shared face fluxes,
# NOT populations via stream-collide — it is dispatched separately from the LBM operators.
global POND_ENABLED = false

# Phase C (2026-06-24): automatic regime selection. Populated in load_case_configuration.
global REGIME_MA = 0.0
global REGIME_RE = 0.0
global REGIME_SELECTION = (regime = :incompressible, lattice = :d3q27, operator = :cumulant,
                           lattice_auto = true, operator_auto = true)

# 2026-06-24 consolidation: D3Q19 retired (seed kept in lattice.jl + kernel_regularization_d3q19.jl).
# Two active solvers — D3Q27 (cumulant, Ma≤0.6) and D3Q39 (entropic, Ma>0.6).
const VALID_LATTICE_TYPES = Set([:d3q27, :d3q39, :pond])

# Compressible mode (Phase 3): sets Ma_lattice = Ma_physical for true supersonic flow.
# When enabled, u_lattice is auto-computed from the physical Mach number so that
# the lattice flow is genuinely supersonic. Requires D3Q39 + entropic.
# The unshifted D3Q39 with numerical equilibrium is stable up to Ma ≈ 1.6
# (Latt, Coreixas et al., Phil Trans R Soc A, 2020).
global COMPRESSIBLE_MODE = false
global SPEED_OF_SOUND_PHYSICAL = 343.0  # m/s, air at standard conditions

# Shifted lattice (Phase 3.2): Galilean transform for Ma > 1.6
# Shifts the D3Q39 velocity set by integer vector (Ux, Uy, Uz).
# Weights, OPP, mirror mappings, entropy, alpha solver are ALL unchanged
# (Frapolli/Chikatamarla/Karlin, PRL 2016). Only C_X, C_Y, C_Z get shifted.
# Extends Ma_max from ~1.6 to ~2.5 (U=1) or ~4.0 (U=2).
global LATTICE_SHIFT_AUTO = false
global LATTICE_SHIFT_X = Int32(0)
global LATTICE_SHIFT_Y = Int32(0)
global LATTICE_SHIFT_Z = Int32(0)
global LOCAL_LATTICE_SHIFT_ENABLED = false
global LOCAL_LATTICE_SHIFT_U_MAX = Int32(1)
global LOCAL_LATTICE_SHIFT_THRESHOLD = 0.5f0
global LOCAL_LATTICE_SHIFT_HYSTERESIS = 0.1f0
global LOCAL_LATTICE_SHIFT_WALL_GUARD_CELLS = 0.0f0
global LOCAL_LATTICE_SHIFT_WALL_ZERO_GUARD_CELLS = 0.0f0
global LOCAL_LATTICE_SHIFT_ZERO_ALLOWED_X_MIN_CELLS = 0.0f0
global LOCAL_LATTICE_SHIFT_WAKE_ZERO_DEFICIT_FRAC = 0.0f0
global LOCAL_LATTICE_SHIFT_HIGH_SPEED_FRAC = 0.60f0
# STATIC GEOMETRIC FRAME MAP (2026-06-27): when > 0, the per-block frame is assigned
# PURELY by wall distance — U=0 within this many cells of the body (the bow-shock standoff
# + stagnation shell), U=u_max everywhere else — instead of the velocity-driven scoring.
# This breaks the dynamic map's upstream-marching collapse for detached bow shocks. Tune
# to ≈ the natural standoff distance. 0 = legacy velocity-driven map.
global LOCAL_LATTICE_SHIFT_GEOMETRIC_BAND_CELLS = 0.0f0

# Compressibility correction (Phase 1): extends valid Ma range to ~0.6–0.9
# by including higher-order velocity terms in the equilibrium that are
# normally truncated at O(u²).  No extra memory or distributions required.
global COMPRESSIBILITY_CORRECTION = false

# Diagnostic: compute non-equilibrium stress tensor Π_neq at slice export time.
# Π_neq,αβ = Σ_k c_kα c_kβ (f_k − f_eq_k) — drives viscous stress and identifies
# boundary layers and shock fronts. Transient GPU allocation, no BlockLevel impact.
# Basis for future Palabos-style variable-ω and stress-based force integration.
global COMPUTE_PI_NEQ = false

# Phase B: Variable ν_sgs from local Π_neq (Palabos variable-ω idea via OpenLB
# per-cell-ω architecture). When enabled:
#   1. Persistent pi_neq field allocated in BlockLevel (6 Float32 per cell = ~24 B/cell)
#   2. Refreshed every timestep after collision
#   3. Collision kernels read per-cell ν_sgs = max(floor, C_π·|Π_neq^pre|/(ρ·cs²))
# Allows global nu_sgs_background to be reduced ~10× while keeping shock stability.
global VARIABLE_NU_SGS = false
global VARIABLE_NU_SGS_C_PI = 0.2f0    # Stress-viscosity coupling constant
global VARIABLE_NU_SGS_FLOOR = 0.002f0 # Minimum ν_sgs in smooth regions (replaces nu_sgs_background when active)

# Knudsen-adaptive τ (Latt, Coreixas, Beny, Parmigiani, Phil Trans R Soc 2020, Eq. 2.28).
# When enabled, the collision kernel computes a local Knudsen-number proxy from
# the already-available Π_neq field: ε ≈ ‖Π_neq‖ / (ρ·cs²), and applies a
# piecewise-constant multiplier α(ε) ∈ {1, 1.05, 1.35, 1/τ} to τ. This replaces
# the continuous variable_nu_sgs mechanism with the canonical Latt threshold
# structure; smooth flow (ε < 0.01) sees α=1 (no extra dissipation), shock /
# boundary-layer regions (ε ≥ 1) see α = 1/τ which caps τ at 1 (maximum stability).
#
# The "pure" Latt 2020 form uses ‖Δf‖/‖f_eq‖ (all 39 populations, not just 2nd
# moment). The full version requires f_eq to be available before τ_eff, which
# is a kernel-order refactor. This Π_neq-proxy version captures the same
# qualitative behaviour (activates near BLs/shocks, idle in smooth flow) and is
# a direct drop-in: pi_neq is already always-on via STRESS_TENSOR_TRANSFER.
global KNUDSEN_ADAPTIVE_TAU = false

# Π_neq persistent field used as the coarse↔fine stress carrier.
# When true: pi_neq is allocated on every level, refreshed every timestep from
# post-collision f, and consumed by the entropic prolongation as the stress mode
# s_k = project_pi_neq_stress_mode(Π̄_neq, c_k) — replacing the population-based
# mirror decomposition s_k = ½(f_neq_k + f_neq_k̄). This is moment-based transfer:
# Π_neq is a smooth 6-component tensor (averaged 2nd moment of f_neq), far more
# stable to interpolate than raw f_k populations which can carry high-frequency
# noise amplified by cubic overshoots.
# Defaults to true for entropic (required for robust AMR coupling at high Mach).
global STRESS_TENSOR_TRANSFER = true

# Phase B4: adaptive ENTROPIC_SIGMA from local |Π_neq|. When enabled, the
# h-component damping varies per cell:
#   sigma_cell = sigma_max − (sigma_max − sigma_min) · clamp(|Π_neq^pre|/pi_ref, 0, 1)
# So smooth regions (small |Π_neq|) get HIGH sigma (kills checkerboard noise) and
# physics regions (shocks, BL) get LOW sigma (preserves real small-scale content).
# Entropic D3Q39 kernel only. Requires VARIABLE_NU_SGS=true (shares pi_neq field).
global VARIABLE_SIGMA = false
global VARIABLE_SIGMA_MIN = 0.3f0   # Minimum damping (applied where |Π_neq| >> pi_ref)
global VARIABLE_SIGMA_MAX = 0.95f0  # Maximum damping (applied in smooth freestream)
global VARIABLE_SIGMA_PI_REF = 0.01f0  # Normalization threshold for |Π_neq^pre|

# Phase B4: adaptive sigma_even in THERMAL DDF kernel (opposite convention from flow):
#   sigma_even = sigma_th_min + (sigma_th_max − sigma_th_min) · quadratic_blend(|Π_neq^pre|/pi_ref)
# Low sigma_even = pure mirror (kills even g-modes, kills noise).
# Smooth regions → sigma_th_min; physics regions → sigma_th_max (preserves energy).
global VARIABLE_SIGMA_THERMAL = false
global VARIABLE_SIGMA_TH_MIN = 0.0f0    # Freestream: pure mirror
global VARIABLE_SIGMA_TH_MAX = 0.15f0   # Physics regions: partial mirror
global VARIABLE_SIGMA_TH_PI_REF = 0.025f0  # Same threshold as flow by default

# Phase C1: Xu-Sagaut dual-equilibrium sponge (ported from OpenLB).
# When enabled, replaces the aggressive pre-collision distribution blending with
# a gentle post-collision additive forcing:
#   f_out[k] = f_collided[k] + sp·(f_eq_ref[k] − f_eq_local[k])
# f_eq_ref is the pre-computed inlet equilibrium (constant reference state).
# This keeps physical distributions intact while gently pulling them toward the
# reference freestream. Cleaner wake decay at outlet than replace-style blending.
global SPONGE_XU_SAGAUT = false

# Double Distribution Function (Phase 2/4b): thermal/energy transport
# Adds N_DIRS_THERMAL extra distributions (g) for energy field.
# Phase 4b (Frapolli): g uses same D3Q39 lattice as f for supersonic flows.
# When disabled, solver behaves identically to standard isothermal LBM.
global DDF_ENABLED = false
global DDF_PRANDTL = 0.71
global DDF_T_INLET = 1.0
global DDF_T_WALL = 1.0
global DDF_T_INITIAL = 1.0
global DDF_WALL_BC = :adiabatic   # :adiabatic or :isothermal
global DDF_TURBULENT_PRANDTL = 0.85  # Pr_t for SGS thermal diffusivity (κ_sgs = ν_eddy / Pr_t)
global DDF_CV = 2.5               # C_v for diatomic air (5/2). Monatomic = 3/2.
global DDF_GAMMA = 1.4            # γ = Cp/Cv = (Cv+1)/Cv for ideal gas
global N_DIRS_THERMAL = Int32(27) # Number of thermal DDF directions (27 or 39)
global DDF_FEEDBACK_TO_FLOW = true
global DDF_FEEDBACK_RELAXATION = 1.0f0
# AUDIT FIX (2026-06-03 Bug B part 2b): for Riemann/shock-tube cases the flow is
# near-stationary between waves (Ma≈0), so the Mach-gated thermal feedback zeros
# out and the per-side IC temperature drifts back toward T_inlet. When this flag
# is on AND the IC is riemann_x, the feedback is applied at all Mach so the
# physical per-side T persists. Default false ⇒ all aero cases byte-identical.
global DDF_SHOCK_TUBE_COUPLING = false
# AUDIT FIX (2026-06-03): explicit upper bound for the Frapolli T-recovery clamp.
# The auto formula caps T at ~2×T_inlet (for a shock tube where u_inlet≈0), which
# CLIPS high-T Riemann states (Lax left T=7.93 → clamped to ~2.0). Set to the
# case's max per-side T (with margin) for shock-tube cases. 0 = auto (aero unchanged).
global DDF_T_ADMISSIBLE_MAX = 0.0f0

# Thermal post-recovery laplacian filter (mitigates AMR-interface T banding).
# Applied AFTER the thermal kernel recovers T from E_kin each timestep.
# T' = (1-α)·T + (α/6)·Σ_neighbors T. α∈[0.0, 0.15] typical; 0.05 is safe default.
global THERMAL_FILTER_ENABLED = false
global THERMAL_FILTER_ALPHA = 0.05f0

global AUTO_LEVELS = false
global MAX_LEVELS = 12
global MIN_COARSE_BLOCKS = 4
global WALL_MODEL_ENABLED = false
global WALL_MODEL_TYPE = :equilibrium
global WALL_MODEL_YPLUS_TARGET = 30.0
global WALL_MODEL_MAX_DISTANCE_CELLS = 10.0f0
# SOTA B6: Compressible wall model (Van Driest transformation)
# When false (default), the legacy incompressible equilibrium wall model is used.
# When true, applies Van Driest 1956 velocity transformation u_VD = u·√(ρ/ρ_w)
# scaled by the local-to-wall density ratio (ρ_w ≈ ρ·T/T_w for ideal gas at
# constant p), and uses ρ_w in the wall-stress formula τ_w = ρ_w·u_τ². Required
# for high-Re compressible BLs. Foundation for Larsson 2010 NEQ-WM (pressure-
# gradient extension is documented future work).
global WALL_MODEL_COMPRESSIBLE = false

# ── SOTA B6-ext: Larsson 2010 NEQ-WM pressure-gradient correction ──
# Adds `+ (y / (ρ_w · u_τ²)) · ∂p/∂x_stream` to the Van-Driest-transformed
# log-law residual. Requires WALL_MODEL_COMPRESSIBLE = true (the term is
# computed inside the Van-Driest branch). Off by default. Schur-bounded to
# |Δu⁺| ≤ 0.5·u⁺ to prevent runaway at stagnation. See
# src/collision/kernel_larsson_pgr.jl for the derivation.
global WALL_MODEL_LARSSON_PGR = false

global DOMAIN_UPSTREAM = 0.75
global DOMAIN_DOWNSTREAM = 1.5
global DOMAIN_LATERAL = 0.75
global DOMAIN_HEIGHT = 0.75
global DOMAIN_SPAN_OVERRIDE = -1.0
global DOMAIN_OBJECT_CENTER_X_FRACTION = nothing
# Optional absolute placement of the transformed/scaled body AABB center in
# resolved-domain metres. Nothing preserves each axis's existing placement.
global DOMAIN_OBJECT_CENTER_X_M = nothing
global DOMAIN_OBJECT_CENTER_Y_M = nothing
global DOMAIN_OBJECT_CENTER_Z_M = nothing
global SPONGE_THICKNESS = 0.10f0
global SPONGE_PROFILE = :cosine      # :cosine (raised cosine) or :tanh (Palabos-style)
global SPONGE_TANH_SIGMA = 0.18f0    # Normalized width of tanh transition (fraction of thickness)
global SPONGE_OUTLET_STRENGTH = 1.0f0
global SPONGE_INLET_STRENGTH = 0.05f0
global SPONGE_TRANSVERSE_STRENGTH = 0.10f0
global OUTLET_BOUNDARY_MODE = Int32(0)  # 0=legacy bool path, 1=freestream, 2=zero-gradient, 3=interior extrapolate

global BLOCK_SIZE_CONFIG = 8
# Mesh-efficiency default (2026-05-30): tightened from 2→1 block. Verified on the
# Ma=0.8395 sphere (MARGIN_m2 vs MARGIN_m1): Cd agrees within ~1.3% through the full
# developing transient, Cl/Cs identical, no divergence — while the fine-level halo
# collapses (L2 body-halo 624→176 blocks, −72%; L3 1148→796, −31%; total 2.25→2.07M
# cells). One block = 8 fine cells is ample donor + physical buffer around the body.
# Per-case `advanced.refinement.margin*` overrides are still respected.
global REFINEMENT_MARGIN = 1
global REFINEMENT_MARGIN_UPSTREAM = 1
global REFINEMENT_MARGIN_DOWNSTREAM = 1
global REFINEMENT_MARGIN_SPAN = 1
global REFINEMENT_MARGIN_UP = 1
global REFINEMENT_MARGIN_DOWN = 1
# Distance-band refinement (2026-06-01): when band_mode == :distance, the per-level
# directional halo thickness is a fixed PHYSICAL distance (meters), converted to
# block layers at each level via ceil(dist / (dx_level·BLOCK_SIZE)). Coarse levels
# (big blocks) then use FEWER layers → thinner coarse shells. :blocks (default) keeps
# the legacy fixed-block-count margins above.
global REFINEMENT_BAND_MODE = :blocks               # :blocks | :distance
global REFINEMENT_BAND_UPSTREAM   = 0.0             # meters (used when :distance)
global REFINEMENT_BAND_DOWNSTREAM = 0.0
global REFINEMENT_BAND_SPAN       = 0.0
global REFINEMENT_BAND_UP         = 0.0
global REFINEMENT_BAND_DOWN       = 0.0
global REFINEMENT_STRATEGY = :geometry_first
global REFINEMENT_MARKER_ONLY = false
global ENABLE_WAKE_REFINEMENT = false
global WAKE_REFINEMENT_LENGTH = 0.25
global WAKE_REFINEMENT_WIDTH_FACTOR = 0.1
# Mesh-efficiency (2026-05-30): anisotropic coarse-level wake pad. Default false
# (legacy isotropic pad). When true, the per-level wake growth pad keeps full
# downstream reach but scales upstream/lateral pad → thinner coarse-level wake.
global WAKE_ANISOTROPIC_PAD = false
global WAKE_PAD_UPSTREAM_SCALE = 0.25
global WAKE_PAD_LATERAL_SCALE = 0.5
# Obstacle standoff + wake cap (Fix, 2026-06-03). 0 = legacy (byte-identical default).
global REFINEMENT_BODY_SHELL_BLOCKS = 0   # >0 = guaranteed N-block isotropic finest-level standoff (hugging shell)
# CORRECTNESS FLOOR (2026-06-29): minimum guaranteed isotropic standoff between a
# real-wall (non-marker) obstacle surface and any AMR box edge. The finest level
# always hugs the body with at least this many blocks of separation, even when
# body_shell_blocks is left at 0 — so a box edge can never touch/cross the body.
# body_shell_blocks ≥ this is respected as-is; marker-only geometry is exempt.
global MIN_OBSTACLE_STANDOFF_BLOCKS = 1
# Sharp-edge (feature-edge) local refinement (2026-06-24). Detects STL edges whose
# adjacent-triangle face normals differ by more than SHARP_EDGE_ANGLE_THRESHOLD_DEG
# (a wing LE/TE, a crease, a corner) and forces EXTRA finest-level blocks around
# them (+SHARP_EDGE_MARGIN_BLOCKS hugging margin), on top of the uniform body shell.
# Default OFF (0.0 threshold) ⇒ byte-identical mesh. Lattice-agnostic, CPU-only.
global SHARP_EDGE_REFINE = false
global SHARP_EDGE_ANGLE_THRESHOLD_DEG = 0.0  # >0 enables; angle BETWEEN adjacent triangle normals (deg). ~30 = typical crease; TE cusp ≫ this.
global SHARP_EDGE_MARGIN_BLOCKS = 1          # extra finest-level blocks dilated around sharp-edge footprint
global WAKE_MAX_LEVEL = 0                  # >0 = wake refinement only at lvl <= this (kills finest-level wake slab)
global WAKE_REFINEMENT_HEIGHT_FACTOR = 0.1
global FEATHER_DEPTH = 0
global FEATHER_NU_MAX = 0.005f0
global FEATHER_X_ONLY = false
global AMR_REGULARIZATION_DEPTH = 0
# Pass178: tensor-product axis-degraded fallback at AMR boundaries is now
# hardcoded-skipped in physics_interpolation.jl (the gate is `if false`).
# A global flag here triggered an InvalidIRError when accessed inside the
# GPU kernel — Julia globals can't be JIT-compiled to PTX. If the original
# behavior is needed, change the `if false` in physics_interpolation.jl
# back to `if !accept_cubic_macros` (around line 1396).
global AMR_INTERFACE_SHELL_SYNC = false
global AMR_SHELL_EQ_SHIFT = true
global AMR_SHELL_WELL_BALANCED_REBUILD = false
global AMR_SHELL_SYNC_INCLUDE_X = false
global AMR_SHELL_POPULATION_BLEND_X = 1.0f0
global AMR_SHELL_RHO_ALPHA_X_SINGLE = 0.04f0
global AMR_SHELL_RHO_ALPHA_X_MULTI = 0.02f0
global AMR_SHELL_RHO_ALPHA_T_SINGLE = 0.70f0
global AMR_SHELL_RHO_ALPHA_T_MULTI = 0.28f0
global AMR_SHELL_VEL_ALPHA_X_SINGLE = 0.05f0
global AMR_SHELL_VEL_ALPHA_X_MULTI = 0.035f0
global AMR_SHELL_VEL_ALPHA_T = 0.12f0
global AMR_SHELL_VEL_NORMAL_ONLY = false
global AMR_SHELL_DRHO_CAP_X_ABS = 5.0f-6
global AMR_SHELL_DRHO_CAP_X_REL = 3.0f-5
global AMR_SHELL_DRHO_CAP_T_ABS = 2.0f-5
global AMR_SHELL_DRHO_CAP_T_REL = 1.5f-4
global AMR_SHELL_DU_CAP_X_ABS = 1.0f-5
global AMR_SHELL_DU_CAP_X_REL = 5.0f-3
global AMR_SHELL_DU_CAP_T_ABS = 2.0f-5
global AMR_SHELL_DU_CAP_T_REL = 2.0f-2
global AMR_SHELL_FLUX_ALPHA_X = 0.0f0
global AMR_SHELL_FLUX_ALPHA_T = 0.0f0
global AMR_SHELL_FLUX_DU_CAP_ABS = 2.0f-5
global AMR_SHELL_FLUX_DU_CAP_REL = 1.0f-2
global AMR_SHELL_MOMFLUX_ALPHA_X = 0.0f0
global AMR_SHELL_MOMFLUX_DRHO_CAP_ABS = 5.0f-6
global AMR_SHELL_MOMFLUX_DRHO_CAP_REL = 3.0f-5
global AMR_SHELL_MOMFLUX_STRESS_ALPHA_X = 0.0f0
global AMR_SHELL_MOMFLUX_STRESS_CAP_ABS = 1.0f-5
global AMR_SHELL_MOMFLUX_STRESS_CAP_REL = 3.0f-5
global AMR_SHELL_FACE_WEIGHT_X = 0.30f0
global AMR_SHELL_FACE_POP_SHIFT = false
global AMR_SHELL_FACE_POP_ALPHA = 1.0f0
global AMR_SHELL_FACE_POP_SIDE = 1
global AMR_SHELL_FACE_STRESS_SHIFT = false
global AMR_SHELL_FACE_STRESS_ALPHA = 0.5f0
global AMR_SHELL_COUPLED_REFLUX = false
global AMR_SHELL_COUPLED_REFLUX_ALPHA = 0.20f0
global AMR_SHELL_PAIRED_REFLUX = false
global AMR_SHELL_PAIRED_REFLUX_ALPHA = 0.20f0
global AMR_SHELL_PAIRED_REFLUX_MASS_ALPHA = 1.0f0
global AMR_SHELL_PAIRED_REFLUX_MOM_ALPHA = 1.0f0
global AMR_SHELL_PAIRED_REFLUX_MOM_ALPHA_INITIAL = 1.0f0
global AMR_SHELL_PAIRED_REFLUX_MOM_ALPHA_HOLD_STEPS = 0
global AMR_SHELL_PAIRED_REFLUX_PARENT_FRACTION = 0.50f0
global AMR_SHELL_PAIRED_REFLUX_MASS_PARENT_FRACTION = 0.50f0
global AMR_SHELL_PAIRED_REFLUX_MASS_CAP_SCALE = 1.0f0
global AMR_SHELL_PAIRED_REFLUX_MASS_CAP_INITIAL = 1.0f0
global AMR_SHELL_PAIRED_REFLUX_MASS_CAP_HOLD_STEPS = 0
global AMR_SHELL_PAIRED_REFLUX_FACE_CONSERVATIVE = false
global AMR_SHELL_PAIRED_REFLUX_COUPLED_CAPS = false
global AMR_SHELL_PAIRED_REFLUX_COUPLED_MASS_CAPS = false
global AMR_SHELL_PAIRED_REFLUX_MACRO_CONSISTENT_CAPS = false
global AMR_SHELL_PAIRED_REFLUX_MACRO_PAIRWISE_CAPS = false
global AMR_SHELL_PAIRED_REFLUX_BUDGET_MEAN_PRESERVING = false
global AMR_SHELL_PAIRED_REFLUX_SHARED_STATE = false
global AMR_SHELL_PAIRED_REFLUX_SHARED_STATE_RHO_ALPHA = 1.0f0
global AMR_SHELL_PAIRED_REFLUX_HIGHPASS = false
global AMR_SHELL_PAIRED_REFLUX_HIGHPASS_BLEND = 1.0f0
global AMR_SHELL_PAIRED_REFLUX_LOWPASS = false
global AMR_SHELL_PAIRED_REFLUX_LOWPASS_BLEND = 0.0f0
global AMR_SHELL_PAIRED_REFLUX_SECOND_MOMENT = false
global AMR_SHELL_PAIRED_REFLUX_SECOND_MOMENT_BLEND = 1.0f0
global AMR_SHELL_PAIRED_REFLUX_POST_STRESS = false
global AMR_SHELL_PAIRED_REFLUX_POST_STRESS_ALPHA = 0.35f0
global AMR_SHELL_PAIRED_REFLUX_POST_STRESS_SOFT_LIMIT = false
global AMR_SHELL_HIGHPASS_DAMPER_RHO_ALPHA = 0.0f0
global AMR_SHELL_HIGHPASS_DAMPER_DRHO_CAP_ABS = 0.0f0
global AMR_SHELL_HIGHPASS_DAMPER_DRHO_CAP_REL = 0.0f0
global AMR_HALO_FULLCELL_RECOMPOSITION = false
global AMR_HALO_FULLCELL_MACRO_ALPHA = 1.0f0
global AMR_HALO_FULLCELL_POP_ALPHA = 0.95f0
global AMR_PROLONGATION_MONOTONE_VELOCITY = false
global AMR_HALO_TEMPORAL_MACRO_CLOSURE = false
global AMR_HALO_EQ_PROJECTION = true
global AMR_HALO_CONSTRAINED_COMPLETION = false
global AMR_RUNTIME_FACE_BUDGET_REFLUX = false
global AMR_RUNTIME_FACE_BUDGET_REFLUX_ALPHA = 0.10f0
global AMR_RUNTIME_FACE_BUDGET_REFLUX_MAX_PARENT_LEVEL = 1
global AMR_RUNTIME_FACE_BUDGET_REFLUX_X_ONLY = true
global AMR_GPU_FACE_BUDGET_REFLUX = false
global AMR_GPU_FACE_BUDGET_REFLUX_ALPHA = 0.10f0
global AMR_GPU_FACE_BUDGET_REFLUX_MASS_ALPHA = 1.0f0
global AMR_GPU_FACE_BUDGET_REFLUX_MOM_ALPHA = 1.0f0
global AMR_GPU_FACE_BUDGET_REFLUX_DRHO_CAP_ABS = 5.0f-6
global AMR_GPU_FACE_BUDGET_REFLUX_DRHO_CAP_REL = 3.0f-5
global AMR_GPU_FACE_BUDGET_REFLUX_MOM_CAP_ABS = 1.0f-5
global AMR_GPU_FACE_BUDGET_REFLUX_MOM_CAP_REL = 3.0f-5
global AMR_GPU_FACE_BUDGET_REFLUX_NEUTRAL_MOM = false
global AMR_GPU_FACE_BUDGET_REFLUX_MAX_PARENT_LEVEL = 1
global AMR_POPULATION_STREAM_REGISTER = false
global AMR_POPULATION_STREAM_REGISTER_FACE_RESOLVED = false
global AMR_POPULATION_STREAM_REGISTER_APPLY = false
global AMR_POPULATION_STREAM_REGISTER_APPLY_ALPHA = 1.0f0
global AMR_POPULATION_STREAM_REGISTER_APPLY_MAX_PARENT_LEVEL = 2
global AMR_POPULATION_STREAM_REGISTER_FACE_APPLY = false
global AMR_POPULATION_STREAM_REGISTER_FACE_APPLY_ALPHA = 1.0f0
global AMR_POPULATION_STREAM_REGISTER_FACE_APPLY_MAX_PARENT_LEVEL = 2
global AMR_POPULATION_STREAM_REGISTER_XFACE_HIGHPASS_APPLY = false
global AMR_POPULATION_STREAM_REGISTER_XFACE_HIGHPASS_APPLY_ALPHA = 0.25f0
global AMR_POPULATION_STREAM_REGISTER_XFACE_HIGHPASS_APPLY_MAX_PARENT_LEVEL = 2
global AMR_POPULATION_STREAM_REGISTER_XFACE_GLOBAL_APPLY = false
global AMR_POPULATION_STREAM_REGISTER_XFACE_GLOBAL_APPLY_ALPHA = 1.0f0
global AMR_POPULATION_STREAM_REGISTER_XFACE_GLOBAL_APPLY_MAX_PARENT_LEVEL = 2
global AMR_SHELL_RECALIBRATION = false
global AMR_POST_RESTRICT_SPONGE = true
global AMR_PARTIAL_FRONTIER_SYNC = false
global AMR_ENFORCE_Z_SYMMETRY = false
global AMR_ENFORCE_Y_SYMMETRY = false
global AMR_THERMAL_NONEQ_TRANSFER = false
# Decimation f_neq filter (Lagrava 2012 §3.3 q-filter, two passes). Default OFF
# per Dorschner 2016 (entropic LBM doesn't need it; the entropic α stabilizer
# handles fine-scale noise) and Schukmann 2025 (the filter INCREASES vc-ov OASPL
# by 6.5 dB). Re-enable only for BGK-family operators without entropic α / σ damping.
global AMR_RESTRICTION_PI_NEQ_FILTER = false
# Pass 265 (Tier 1 #1e) — number of moment-correction passes after the f_eq +
# τ·f_neq restriction reconstruction. Default 2 reproduces pass 187+ behavior;
# raising to 4-6 tightens the conservation residual at the cost of more
# arithmetic (and risks pass-186's positivity-floor asymmetry if pushed too high).
# Each pass: re-sum f, compute (ρ, j) deficit, solve 4×4 Gram, apply correction.
global AMR_RESTRICTION_CORRECTION_PASSES = Int32(2)
# Pass 265 (Tier 1 #1e, complementary to CORRECTION_PASSES). When true, the
# restriction kernel applies an exact 4×4 Gram correction to the parent f_eq
# (mass + 3 momentum) *before* adding the τ-rescaled stress part. The
# 13-moment Newton solver has ~1e-5 tolerance on the conserved moments; this
# closes that residual to round-off, so the subsequent 2-pass redistribution
# has nothing left to correct in freestream. Default off, opt-in.
global AMR_RESTRICTION_CONSERVATIVE_FEQ = false
# SOTA Active Workstream P1 (2026-05-27): conservative ρE restriction across AMR.
# When enabled, upgrades the 4-moment (mass + 3·momentum) correction to a
# 5-moment correction that also enforces parent kinetic energy = (1/8)·Σ_children
# Σ_k f_k·|c|². Closes the last conservation gap in the AMR fine→coarse
# transfer. Requires conservative_feq=true and Frapolli thermal (DDF) enabled
# (the parent kinetic-energy target is only computed when restrict_thermal=1).
# Closed-form solution exploits D3Q39 lattice isotropy — no extra Gram solve.
global AMR_RESTRICTION_CONSERVATIVE_RHOE = false
# ── B1 Pillar 2: 13-moment-exact (mortar-element) restriction ──
# When true (D3Q39 only), the AMR fine→coarse restriction replaces the
# legacy f_eq + projected stress carrier with a single closed-form solve
# that drives ALL 13 Hermite moments of the parent populations (mass +
# 3·momentum + 6·stress + 3·energy flux) to the τ-rescaled averaged child
# moments, to Float32 round-off. This is the strict-conservation upgrade
# over the 5-moment correction (`restriction_conservative_rhoe`) and the
# pillar 2 contribution of the B1 mortar-element AMR coupling workstream.
# Off by default; opt in via `restriction_conservative_13mom: true` in
# the case config. Implies `restriction_conservative_feq: true` because
# the legacy stress-carrier path is bypassed entirely on the flow side.
global AMR_RESTRICTION_CONSERVATIVE_13MOM = false
# ── GOLD-STANDARD STRICT CONSERVATION (2026-06-17): D3Q19 4-moment exact ──
# When true (D3Q19 only), the fine→coarse restriction adds a 4-moment (mass +
# 3·momentum) correction that drives the parent populations to EXACT conservation
# to FP32 round-off — the D3Q19 analogue of the D3Q39 13-moment-exact path.
# v1 applies to full-coverage parents; frontier cells keep current behavior.
# Default false ⇒ byte-identical.
global AMR_RESTRICTION_CONSERVATIVE_D3Q19 = false
# ── GOLD-STANDARD STRICT CONSERVATION (2026-06-23): D3Q27 cumulant 4-moment exact ──
# The D3Q27 cumulant analogue of the above. D3Q27 restriction reconstructs the
# parent from the polynomial equilibrium (round-trip-consistent with the cumulant
# collision) but leaves a small mass+momentum residual after the positivity floor,
# worst at ragged frontier cells. When true, the SAME n_dirs-generic 4-moment Gram
# correction (solve_shell_moment4) the D3Q19 path uses drives the parent populations
# to EXACT ρ + 3·momentum. D3Q27 is a full isotropic lattice (Σ w_k c_α c_β = cs²δ),
# so the {1, c} correction is orthogonal to the traceless stress — shear untouched.
# Default false ⇒ byte-identical. The kernel reuses the restriction_conservative_d3q19
# parameter (now a generic "apply the 4-moment Gram correction" flag fed per-lattice).
global AMR_RESTRICTION_CONSERVATIVE_D3Q27 = false
# C2/C4 quintic B-spline AMR prolongation (2026-06-04). When true (D3Q27 only;
# gated on LATTICE_TYPE == :d3q27 in timestep.jl), the coarse→fine macro
# interpolation uses the per-coarse-step B-spline prefilter
# + compact 6-tap eval → C4 continuity + exact 2:1 conservation. Default off.
global QUINTIC_BSPLINE_PROLONGATION = false
# ── Gendre 2017 (2026-06-28) — AMR-interface normal-gradient correction ──
# When true (D3Q39 entropic only), the coarse→fine transition cells receive the
# Gendre directional-splitting correction ΔFα = −c⊥α·∂⊥fα that restores the
# advective gradient NORMAL to the refinement interface — the term standard
# stream-and-collide drops at the seam, and the documented source of the weak
# spurious-acoustic box artifact at subsonic Mach. Operator-independent (Gendre
# §III.B). Exactly zero on a uniform freestream ⇒ byte-identical to off there.
# Default off.
global GENDRE_NORMAL_GRADIENT = false
# ── Intensive-T thermal advection fix (2026-06-29) ──
# The Frapolli g carries the energy density ρT; streaming it across a density
# gradient spuriously heats the field (root-caused for the Ma0.9 freestream T-drift
# 1.0→2.29, validated offline). When true, the thermal streaming rescales each
# advected g by ρ_local/ρ_upstream → advects the INTENSIVE temperature. Byte-identical
# where ρ is uniform. Default off pending shock-tube (Sod/Lax) regression.
global THERMAL_INTENSIVE_ADVECTION = false
# ── B3 (2026-05-28) — Float64 halo arithmetic (restriction side) ──
# When true, the AMR fine→coarse restriction (restrict_child_state_to_parent_kernel!)
# runs the 8-child accumulation + 39-direction moment sum in Float64 internally
# (promotes Float32 reads to Float64, demotes back to Float32 at return). Bulk
# distributions stay Float32 → zero VRAM impact. Drops the coarse-side seam
# residual toward FP32-storage ULP. Off by default.
global INTERFACE_FP64_HALO = false
# HRR-style hybrid stress reconstruction (Coreixas 2017, Jacob-Malaspinas-Sagaut 2018,
# Astoul 2020, Schukmann 2025). When active, the projection-based 2nd-order
# non-equilibrium coefficient a_1^(2) is blended with a finite-difference strain-rate
# version: a_1^(2) = σ·a_1^(2),PR + (1-σ)·a_1^(2),FD, where a_1^(2),FD = -ρ·c_s²·(2τ-1)·S
# (Chapman-Enskog continuum stress). This suppresses non-hydrodynamic 2nd-order modes
# that alias at AMR interfaces and emit spurious acoustic noise (Astoul 2020 main result).
# σ=1 → pure projection (legacy entropic). σ=0.98 → recommended for high-Re per Schukmann 2025.
# Only the entropic D3Q39 kernel currently uses HRR-hybrid; other kernels accept-and-ignore.
global HRR_HYBRID_ENABLED = false
global HRR_SIGMA = 1.0f0
# Pass 261 — per-level σ_HRR override. Empty (default) → all levels use HRR_SIGMA.
# When provided in config as `advanced.numerics.hrr_sigma_per_level: [σ_L1, σ_L2, σ_L3, ...]`,
# each level reads its own value. Targets the ring-mode on the finest level
# without softening L1/L2 stress. Missing entries fall back to HRR_SIGMA.
global HRR_SIGMA_PER_LEVEL = Float32[]

# ── Recursive Regularization (RR-LBM) — Latt-Chopard 2006, Coreixas 2017,
#    Coreixas-Latt 2019, Schukmann 2025 ──
# Pure-projection regularization through Hermite order 3 for D3Q39. After the
# collision step, post-collision populations are projected onto the physically
# representable Hermite subspace (mass + momentum + stress + heat-flux). The
# 4-dimensional ghost subspace — containing the (-1)^(y+z) checkerboard
# eigenmode visible in case 001-006 wakes — is annihilated by construction.
# Mass, momentum, stress, and heat-flux are preserved exactly by Hermite
# orthogonality. Only the D3Q39 entropic kernel implements this.
global RR_LBM_ENABLED = false
global RR_LBM_MIN_LEVEL = 4   # apply only on levels >= this (default L4 only)
# "finest only" mode (2026-06-29): when recursive_regularization.min_level is the string
# "finest" (or an integer <= 0), RR-LBM is applied to the FINEST level only, whatever the
# actual level count is at runtime (robust to AMR level clamping) — the gate then tests
# level_id == NUM_LEVELS_CONFIG instead of level_id >= RR_LBM_MIN_LEVEL.
global RR_LBM_FINEST_ONLY = false

# ── Spectral Vanishing Viscosity (SVV) — Tadmor 1989 / Karamanos-Karniadakis
#    2000 / Pasquetti 2006 ──
# Y-Z bi-Laplacian damping of spatial high-wavenumber modes that have nonzero
# macroscopic moments (and therefore escape Hermite projection / RR-LBM). The
# bi-Laplacian eigenvalue scales as k⁴ — 47× selectivity between Nyquist
# (k=π) and 4-cell wavelength (k=π/4). Conservation correction (4-moment
# redistribution, Lagrava 2012 §3.5) preserves ρ and ρu exactly per cell.
# Recommended ε for ONERA M6 transonic / SR=224: 0.002 (kills Y-Z Nyquist in
# ~50 steps, < 0.2 %/step damping at resolved scales).
global SVV_YZ_ENABLED = false
global SVV_YZ_EPSILON = 0.0f0
global SVV_YZ_MIN_LEVEL = 4   # apply only on levels >= this (default L4 only)

# SOTA #5 (2026-05-29): streamwise-X SVV — kills the L1 streamwise velocity/Mach
# banding. Inverse gating vs Y-Z: applied on COARSE levels (level_id <=
# SVV_X_MAX_LEVEL, default 1 = L1 only) where the lattice ν is smallest.
global SVV_X_ENABLED = false
global SVV_X_EPSILON = 0.0f0
global SVV_X_MAX_LEVEL = 1   # apply only on levels <= this (default L1 only)

# ── Two-way EOS coupling (SOTA Active Workstream A3 / P2) ─────────────────
# When false (default), the f-equilibrium uses the LINEARIZED ideal-gas form
# `p = ρ·cs²·T` — an implicit R = cs² regardless of the configured γ and cv.
# This is the legacy behavior and works well for transonic Ma ≲ 1, where the
# error from linearization is small (~0.7 % of physical pressure at Ma=0.84).
# When true, the kernel uses the TRUE thermodynamic EOS:
#   p = (γ−1)·ρ·cv·T
# with γ from `thermal.gamma` and cv from `thermal.cv`. This is required for
# any defensible supersonic claim (the linearization error grows to ~5 % at
# Ma=1.5, >15 % at Ma=2.0).
# The lattice's discrete cs² remains the polynomial-Hermite reference (it's a
# lattice-geometry constant, distinct from the fluid EOS); only the moment
# targets for the 13-moment Newton solver change. See SOTA_ROADMAP P2.
global EOS_THERMODYNAMIC = false
# AUDIT FIX (2026-06-03): configurable macroscopic realizability clamp bounds.
# Legacy hard-coded values (far-bulk 0.42/2.10, near-wall 0.35/2.40) clamp out
# legitimately-low-density states (Sod ρ_R=0.125 → 0.42), breaking canonical
# shock-tube/Riemann benchmarks. Defaults preserve the legacy aero behavior;
# canonical low-density cases set e.g. macro_density_floor: 0.05.
global MACRO_DENSITY_FLOOR = 0.42f0      # far-bulk floor
global MACRO_DENSITY_CEIL = 2.10f0       # far-bulk ceiling
global MACRO_DENSITY_FLOOR_NEAR = 0.35f0 # near-wall floor
global MACRO_DENSITY_CEIL_NEAR = 2.40f0  # near-wall ceiling
# Adaptive operator (2026-05-30): on cells with local Ma < ADAPTIVE_MA_THRESH AND
# away from AMR halos/walls, the heavy 13-mom Newton equilibrium is replaced by the
# cheap O(u⁴) polynomial equilibrium — PROVEN equivalent to ≤1% up to Ma_local~0.3
# (velocity-sweep test 2026-05-30), so smooth cells get the same f_eq without the
# iterative exp-heavy solve. Default off; threshold conservative (0.25 « the 0.3
# divergence point). The α-solver is already auto-skipped for smooth cells.
global ADAPTIVE_OPERATOR = false
global ADAPTIVE_MA_THRESH = 0.25f0

# ── Total-energy source terms (SOTA #1, 2026-05-29) ─────────────────────
# When true (D3Q39 + Frapolli only), the thermal collision adds the two
# compressible energy-equation source terms that the passive g-relaxation
# omits: pressure-work −p·∇·u and viscous heating Φ = 2ρν(S:S − ⅓(∇·u)²).
# These are injected into Σg (the non-kinetic energy carrier) so that, per the
# Frapolli closure T=(E_kin−ρ|u|²+2Σg)/(2ρCv), the per-step temperature gains
# ρCv·ΔT = −p·∇·u + Φ. This is REQUIRED physics for the Ma≤2 supersonic target
# (shock heating error: ~0.7% at Ma=0.84, ~25% at Ma=2.0), so the loader
# DEFAULTS IT ON whenever the Frapolli model is live (DDF + D3Q39). See the
# default-resolution logic near the DDF_ENABLED block.
global ENERGY_SOURCE_TERMS = false

# ── Ducros shock-sensor dissipation (SOTA #3 robustness, 2026-05-29) ────
# Per-cell normalized density gradient s=|∇ρ|/ρ·dx; a smoothstep over
# [lo,hi] raises the local ν_sgs floor by boost·nu_max only at strong
# compressive jumps. Suppresses Gibbs/carbuncle at shocks (needed for
# stable Ma≳1.6) while leaving smooth flow untouched. Default-on for
# compressible D3Q39. lo/hi calibrated to the existing diagnostic sensor.
global SHOCK_SENSOR_ENABLED = false
global SHOCK_SENSOR_NU_MAX = 0.02
global SHOCK_SENSOR_THRESH_LO = 0.15
global SHOCK_SENSOR_THRESH_HI = 0.30

# ── Level-independent lattice-ν floor (2026-05-29) ──────────────────────
# Minimum lattice ν applied on EVERY level (NOT scaled by amr_sgs_scale).
# The amr_sgs_scale machinery gives coarse levels small lattice ν for clean
# AMR coupling, but that leaves L1 under-damped against streamwise/checkerboard
# LATTICE modes (observed as growing velocity/Mach banding in the L1 freestream
# of case 030). This floor lifts the coarse levels while staying inert on fine
# levels. 0 = off. ~1e-3 is a light, mode-killing value at u_lat~0.7.
global NU_SGS_LATTICE_FLOOR = 0.0

# ── Dynamic SGS (Germano-Lilly) — SOTA Active Workstream B2 ─────────────
# When false (default), variable_nu_sgs uses the hand-tuned `c_pi` constant.
# When true, the entropic D3Q39 kernel computes a per-cell dynamic Smagorinsky
# coefficient Cs² ∈ [0, 0.25] via the Germano (1991) identity and Lilly (1992)
# least-squares closure, and uses ν_sgs = Cs²·Δ²·|S| (Smagorinsky form). The
# coefficient auto-zeros in freestream and recovers ~0.04-0.10 in BL/wake,
# removing the per-case manual `c_pi` tuning knob. See
# src/collision/kernel_dynamic_sgs.jl for the derivation.
global DYNAMIC_SGS_ENABLED = false

# ── NSCBC outflow BC (SOTA A4) — Poinsot-Lele characteristic outlet ──────
# When false (default), the +x outlet uses the legacy sponge / freestream
# blend. When true, an NSCBC-LODI relaxation BC runs on the L1 +x face cells
# after collision each step. Replaces the reflective sponge with a true
# non-reflecting BC. Required for any defensible aeroacoustic or supersonic
# claim. See src/collision/kernel_nscbc_outlet.jl for the algorithm.
global OUTLET_NSCBC_ENABLED = false
global OUTLET_NSCBC_P_TARGET = 1.0f0       # lattice pressure target (= ρ_ref · cs²)
global OUTLET_NSCBC_SIGMA = 0.27f0         # Rudy-Strikwerda 1980 LODI relaxation σ
global OUTLET_NSCBC_NEQ_BLEND = 0.5f0      # how much of the interior f_neq to preserve at face

# ── Global mass-conservation controller (2026-05-29) ────────────────────
# Root-cause fix for the over-constrained-BC mass pump (hard-velocity inlet +
# impermeable lateral walls + pressure-only outlet ⇒ no ∮ρu·n=0). Proportional
# controller: lowers the NSCBC outlet p_target in proportion to the domain mass
# surplus so the LODI relaxation bleeds the excess. Requires NSCBC outlet on.
# Default ON for compressible — without it, fixed-velocity-inlet cases pump.
global MASS_CONTROL_ENABLED = false
global MASS_CONTROL_GAIN = 2.0f0           # p_target drop per unit relative surplus
global MASS_CONTROL_BASELINE_STEP = 8      # capture baseline mass after this many substeps
# 2026-05-30 — mass-control MODE. The original :nscbc_ptarget path moves the
# NSCBC outlet p_target by the surplus; but the NSCBC characteristic outlet
# itself detonates at Ma=0.84 (corner interaction with lateral walls — proven
# by cases 044/044a, both blew up @step152 with NSCBC on, while the plain
# zerogradient-outlet 042 stayed stable). The :rescale mode is decoupled from
# the outlet BC entirely: it gently rescales the L1 populations by
# s = clamp(M_target/M_now, 1-ε, 1+ε) each step. Since f scales ρ and j
# together, the velocity field u = j/ρ is UNCHANGED — it bleeds only the
# excess mass. Unconditionally stable (no BC, no characteristics, no corners).
global MASS_CONTROL_MODE = :nscbc_ptarget   # :nscbc_ptarget | :rescale
global MASS_CONTROL_RESCALE_EPS = 1.0f-2    # max per-step |Δρ|/ρ for :rescale mode (2026-05-31: 1e-3→1e-2 so the P-controller can actually outpace a Ma~0.84 mass pump; still a bounded safety rail, u unchanged)

global BOUNDARY_METHOD = :bounce_back
global TRANSVERSE_FREESTREAM_BOUNDARY = false
global TRANSVERSE_FREESTREAM_BLEND = 1.0f0
global OUTLET_FREESTREAM_BOUNDARY = false
global OUTLET_FREESTREAM_BLEND = 1.0f0
global INLET_DENSITY_MODE = Int32(0)  # 0=freestream rho, 1=local boundary rho
global SPAN_PERIODIC_Y = false
# ══ PERIODIC-Y AMR (branch lbm-periodic-amr, 27 Oct 2026) — BEGIN globals block ══
# (parsed in the marked block at the end of _load_case_configuration_values!)
# Periodic-Y AMR (opt-in): spanwise periodicity on EVERY level
# (advanced.boundary.periodic_y_amr: true, requires periodic_y: true). Off ⇒ the
# legacy rule (periodic Y only for single-level runs) is unchanged.
global PERIODIC_Y_AMR = false
# Optional: extrude every refined level's (x, z) footprint over the whole Y
# period (advanced.refinement.span_uniform_levels: true; periodic_y_amr only).
global SPAN_UNIFORM_LEVELS = false
# Optional raw-state dump (advanced.diagnostics.raw_state_dump_freq: N > 0):
# per-level rho/vel/obstacle/block coordinates written as raw little-endian
# binaries every N steps (at batch ends) and at the final step. Off by default.
global RAW_STATE_DUMP_FREQ = 0
# Optional (periodic_y_amr only): shift the placed body along Y by this many
# metres (advanced.domain.periodic_y_body_shift). On a periodic axis any
# placement is legitimate; the STL must then cover the period as it should
# appear (e.g. an extrusion longer than the span, or two copies one period
# apart for a body on the seam) — geometry outside [0, Ly) is not wrapped.
global PERIODIC_Y_BODY_SHIFT = 0.0
"True when the CPU domain build wraps Y on every level (periodic_y AND periodic_y_amr)."
periodic_y_amr_active() = SPAN_PERIODIC_Y && PERIODIC_Y_AMR
# ══ PERIODIC-Y AMR — END globals block ══
# ══ Z-FACE WALLS (branch lbm-zwalls, 27 Oct 2026) — globals ══
# advanced.boundary.z_walls: the top/bottom (z-face) wall condition.
#   absent     → :legacy  — the mirror ghosts plus the transverse sponge on the z faces
#                           (advanced.domain.sponge_transverse_strength), unchanged;
#   "slip"     → :slip    — mirror (free-slip) ghosts and NO sponge on the z faces;
#   "no_slip"  → :no_slip — halfway bounce-back (stationary wall) and NO sponge on the z faces.
# Z_WALL_BC is the kernel code passed to the flow kernels (0 legacy, 1 slip, 2 no_slip).
# Parsed and checked in the marked block after the periodic-Y AMR parse block.
global Z_WALLS_MODE = :legacy
global Z_WALL_BC = Int32(0)
"One-line description of the z-face (top/bottom) wall condition, for the banner and the analysis summary."
function z_walls_summary(mode::Symbol=Z_WALLS_MODE)
    mode === :slip && return "slip (advanced.boundary.z_walls: \"slip\"): exact specular (free-slip) walls, no z-face sponge"
    mode === :no_slip && return "no-slip (advanced.boundary.z_walls: \"no_slip\"): stationary halfway bounce-back walls, no z-face sponge"
    return @sprintf("legacy (advanced.boundary.z_walls absent): mirror walls + transverse sponge on the z faces (strength %.3f)",
                    Float64(SPONGE_TRANSVERSE_STRENGTH))
end
# ══ Z-FACE WALLS — END globals ══
# ══ LEVEL VISCOSITY (branch lbm-taufix, 28 Oct 2026) — globals ══
# advanced.refinement.level_viscosity: the relaxation time of the coarser AMR levels.
#   absent / "legacy" → :legacy   — tau_l = 1/2 + (tau_f - 1/2)·2^(L-l): with dt ∝ dx each coarser level carries 4× the
#                                    physical viscosity of the next finer one; unchanged, byte-identical;
#   "physical"        → :physical — tau_l = 1/2 + (tau_f - 1/2)/2^(L-l): the same physical viscosity on every level
#                                    (Filippova-Hänel / Dupuis-Chopard), floored at TAU_MIN (reported per level); the
#                                    level-coupling rescalings of f_neq use the pre-collision ratio (tau dt)_c/(tau dt)_f.
# LEVEL_VISC_PHYSICAL is the kernel code (0 legacy, 1 physical). Parsed after the z-walls parse block.
global LEVEL_VISCOSITY_MODE = :legacy
global LEVEL_VISC_PHYSICAL = Int32(0)
# ══ LEVEL VISCOSITY — END globals ══
# Non-reflecting inlet (2026-05-29): 1.0 = legacy fully-reflective (pure f_eq
# imposition); <1.0 blends in the interior-extrapolated population so acoustic
# waves transmit through the inlet instead of reflecting (kills streamwise
# density banding seeded by the reflective inlet).
global INLET_NEQ_BLEND = 1.0f0
global BOUZIDI_LEVELS = 1
global Q_MIN_THRESHOLD = 0.001f0
global BOUZIDI_Q_GEOMETRY = :stl_ray
# One-shot startup symmetry report of Bouzidi wall links per direction and
# Z-hemisphere. CPU only, no hot-path cost. See bouzidi_diagnostic.jl.
global BOUZIDI_SYMMETRY_REPORT = false
# Pass 192 — Per-cell (k, x_ff) audit of the Bouzidi q<0.5 behind-cell
# fallback. CPU only, one-shot at level setup, no hot-path cost.
# Quantifies how many cells per direction silently fall back to
# f_ff = f_k because the cell behind is solid / OOB / no-neighbor.
# See bouzidi_ff_diagnostic.jl.
global BOUZIDI_FF_AUDIT = false
global BOUZIDI_MISSED_LINK_AUDIT = false
global BOUZIDI_MISSED_LINK_FIX = :off
global BOUZIDI_MISSED_LINK_FIX_MODE = Int32(0)
global BOUZIDI_MISSED_LINK_EQ_BLEND = Float32(0.0)
global BOUZIDI_MISSED_LINK_DELTA_REL_CAP = Float32(0.08)
global BOUZIDI_MISSED_LINK_DELTA_ABS_CAP = Float32(2.0e-4)
global BOUZIDI_WALL_POP_LIMITER_REL_CAP = Float32(0.0)
global BOUZIDI_WALL_POP_LIMITER_ABS_CAP = Float32(0.0)
# Pass 193 — q<0.5 behind-cell fallback mode in the GPU Bouzidi kernel.
# When x_ff is solid / OOB / no-neighbor:
#   :halfway_bb (mode 0, default) — legacy. f_ff = f_k, which makes the
#       BFL interpolation silently collapse to plain halfway bounce-back
#       at the affected (cell, direction). Continuous in q only if every
#       wall neighbor falls back identically. On wing-skin geometry with
#       stair-stepped Bouzidi q-maps, the per-link fallback firing pattern
#       is direction-pair-asymmetric across adjacent wall cells, which
#       injects a macroscopic-mode (rho + linear-momentum) Nyquist signal
#       that is invisible to s-mode (tau_min) and h-mode (variable_sigma)
#       damping by construction. This is the third channel diagnosed in
#       PASS192_DIAGNOSIS.md.
#   :wall_eq (mode 1) — pass193 candidate. f_ff = w_k * rho_f using the
#       local boundary-cell density. This is the no-slip equilibrium
#       projection at q -> 0 in the lab frame and is continuous across
#       neighboring wall cells regardless of which links fall back.
#   :bounce_back (mode 2) — pass195 candidate. For genuinely uncovered
#       double-missing links, mirror the local post-collision opposite
#       population. This keeps non-equilibrium pair content instead of
#       projecting the wall-adjacent cell all the way to equilibrium.
global BOUZIDI_FF_FALLBACK = :halfway_bb
global BOUZIDI_FF_FALLBACK_MODE = Int32(0)
global BOUZIDI_SHIFTED_WALL_CORRECTION = true
global BOUZIDI_SHIFTED_WALL_CORRECTION_SCALE = 1.0f0
global BOUZIDI_SHIFTED_WALL_CORRECTION_RAMP_STEPS = Int32(0)
global BOUZIDI_SHIFTED_WALL_CORRECTION_CAP_REL = 0.25f0
global BOUZIDI_SHIFTED_WALL_CORRECTION_CAP_ABS = 0.0f0

# ── B7 — Gold-standard curved-wall BC ladder ──
# `bc_scheme` selects the wall-link reconstruction:
#   :bouzidi_linear  — legacy 2-point linear BFL 2001 (default)
#   :ymls_quadratic  — Tier A: 3-point quadratic (BFL 2001 / YMLS 2003)
#   :marson_enriched — Tier B: YMLS + Marson 2021 isotropy-correcting
#                      consistency term (single-cell form, PRE 103,
#                      053308). Genuine "best-in-world" target — no other
#                      LBM solver ships this. See SOTA_ROADMAP "B7" and
#                      "STANDING DIRECTIVE — gold-standard wall BC".
global BC_SCHEME = :bouzidi_linear
# Grad-reconstruction wall BC knobs (2026-06-05); set by the parser below.
global GRAD_WALL_PINEQ_SCALE = 1.0f0
global GRAD_WALL_VELOCITY_MODE = Int32(0)

global FORCE_COMPUTATION_ENABLED = false
global FORCE_OUTPUT_FREQ_CONFIG = 0
global FORCE_OUTPUT_FREQ = 0
global MOMENT_CENTER_CONFIG = [0.0, 0.0, 0.0]
global NET_FORCE_EXPORT_ENABLED = false

global DIAG_FREQ = 100
global STABILITY_CHECK_ENABLED = true
global PRINT_TAU_WARNING = true
global THERMAL_INTERFACE_DIAGNOSTICS = false
global AMR_INTERFACE_DIAGNOSTICS = false
# Default: amr_seam_*.csv emitted only at t_step ∈ {100, 200, 400}. When
# `advanced.diagnostics.amr_seam_all_steps: true`, emit at every slice step
# so the free-stream preservation CI gate has a dense time series. See
# PREPOST/freestream_gate_check.jl and case 013.
global AMR_SEAM_DIAGNOSTIC_ALL_STEPS = false
# ── AMR validation gates 1 + 3 (13-mom conservation + H-theorem) ─────
# Emits per-fine→coarse-hop CSVs `amr_conservation_*.csv` and
# `amr_entropy_*.csv` to RESULTS/SLICES. Activated by config flag
# `diagnostics.amr_validation_gates: true`. Diagnostic frequency
# controlled by `diagnostics.amr_validation_freq` (default = slice_output_freq).
# CSVs consumed by:
#   - src/PREPOST/amr_conservation_check.jl
#   - src/PREPOST/amr_entropy_check.jl
# See cases 018 (conservation + entropy) and 019 (Spiegel vortex).
global AMR_VALIDATION_GATES = false
global AMR_VALIDATION_FREQ = 0
# ── AMR validation gate 2 (Spiegel isentropic vortex) ────────────────
# Emits `vortex_metrics_*.csv` with peak |u - U_inlet| per level per
# step. Consumed by src/PREPOST/spiegel_vortex_check.jl.
global VORTEX_METRICS_ENABLED = false
# Sod shock-tube centerline profile (case 050). Emits transverse-averaged
# `sod_profile_*.csv` (x,rho,ux,p,T). Consumed by src/PREPOST/sod_check.jl.
global SOD_CENTERLINE_ENABLED = false
global AMR_FULL_DIAGNOSTICS_FREQ = 0
global AMR_FACE_REGISTER_DIAGNOSTICS = false
global AMR_RUNTIME_FACE_BUDGET_DIAGNOSTICS = false
global MASS_CONSERVATION_DIAGNOSTICS = false
# Pass 190 — Compressible audit diagnostic (read-only). When true, the
# diagnostic tick emits a per-step CSV row with the Frapolli conservation
# moments (mass, momentum, total energy) and per-level Mach/T statistics.
# See docs/2026-05-16_compressible_sota_plan.md.
global COMPRESSIBLE_AUDIT_DIAGNOSTICS = false
global TRANSONIC_WAKE_DIAGNOSTICS = false
global TRANSONIC_WAKE_DIAG_X_MIN_CELLS = 0.0f0
global TRANSONIC_WAKE_DIAG_MAX_X_DIAMETERS = 3.0f0
global TRANSONIC_WAKE_OVERSPEED_WARN_FRAC = 0.03f0
global STABILITY_ABORT_ENABLED = false
global STABILITY_ABORT_RHO_MIN = 0.0f0
global STABILITY_ABORT_RHO_MAX = 0.0f0
global STABILITY_ABORT_U_MAX = 0.0f0
global STABILITY_ABORT_MASS_REL = 0.0f0
global STABILITY_ABORT_WAKE_SHEAR_OVERSPEED = 0.0f0
global STABILITY_ABORT_WAKE_SHEAR_FRAC = 0.0f0
global STABILITY_ABORT_BASE_PRESSURE_REL_MIN = 0.0f0
global STABILITY_ABORT_BASE_PRESSURE_REL_MAX = 0.0f0

# Pass 192a — Initial-condition selector. Default :uniform reproduces the
# legacy single-state initialization at machine precision (the Riemann
# branch is never visited). :riemann_x splits the X axis at the configured
# physical position and initializes left/right cells with the configured
# (rho, ux, uy, uz, T) states. Used for the Sod / Riemann validation
# cases (Pass 192b onward); production aero cases should leave this at
# :uniform. :gaussian_x/y/z add smooth acoustic-pulse microscopes for
# outlet/AMR-orientation studies.
global INITIAL_CONDITION_TYPE = :uniform   # :uniform | :riemann_x | :gaussian_x | :gaussian_y | :gaussian_z
global IC_GAUSSIAN_AXIS = 1
global IC_UNIFORM_RHO = 1.0
global IC_UNIFORM_UX = 0.0
global IC_UNIFORM_UY = 0.0
global IC_UNIFORM_UZ = 0.0
global IC_UNIFORM_T = 1.0
global IC_GAUSSIAN_CENTER_X = 0.25
global IC_GAUSSIAN_CENTER_Y = 0.0
global IC_GAUSSIAN_CENTER_Z = 0.0
global IC_GAUSSIAN_SIGMA_X = 0.04
global IC_GAUSSIAN_SIGMA_Y = 0.04
global IC_GAUSSIAN_SIGMA_Z = 0.04
global IC_GAUSSIAN_RHO_AMPLITUDE = 1.0e-3
global IC_GAUSSIAN_UX_AMPLITUDE = 0.0
global IC_GAUSSIAN_UY_AMPLITUDE = 0.0
global IC_GAUSSIAN_UZ_AMPLITUDE = 0.0
global IC_GAUSSIAN_T_AMPLITUDE = 0.0
global IC_GAUSSIAN_ELLIPSOIDAL = false
global IC_RIEMANN_SPLIT_X    = 0.5         # physical meters
global IC_RIEMANN_LEFT_RHO   = 1.0
global IC_RIEMANN_LEFT_UX    = 0.0
global IC_RIEMANN_LEFT_UY    = 0.0
global IC_RIEMANN_LEFT_UZ    = 0.0
global IC_RIEMANN_LEFT_T     = 1.0
global IC_RIEMANN_RIGHT_RHO  = 0.125
global IC_RIEMANN_RIGHT_UX   = 0.0
global IC_RIEMANN_RIGHT_UY   = 0.0
global IC_RIEMANN_RIGHT_UZ   = 0.0
global IC_RIEMANN_RIGHT_T    = 0.8

# ── Spiegel isentropic vortex IC (Gate 2 — AMR validation) ──────────
# 2D Gaussian vortex superimposed on a uniform freestream. An exact
# stationary solution of the Euler equations for an inviscid isentropic
# flow — the vortex translates with the freestream at constant strength.
# Any decay observed is purely numerical: ideal test for AMR fidelity.
#
#   r²   = ((x - x_c)/R)² + ((y - y_c)/R)²
#   δu_x = -((y - y_c)/R) · A · exp((1 - r²)/2)
#   δu_y =  ((x - x_c)/R) · A · exp((1 - r²)/2)
#   δT   = -(γ-1)·A²/2 · exp(1 - r²)
#   ρ    = ρ_∞ · (T/T_∞)^(1/(γ-1))    -- isentropic
#
# Parameters:
#   IC_SPIEGEL_CENTER_X/Y/Z  — vortex center in physical meters
#   IC_SPIEGEL_RADIUS        — vortex characteristic length R [m]
#   IC_SPIEGEL_AMPLITUDE     — A, dimensionless strength (≈ 0.1 for safe
#                              subsonic test; higher = stronger but with
#                              acoustic radiation if compressible enough)
global IC_SPIEGEL_CENTER_X    = 0.5
global IC_SPIEGEL_CENTER_Y    = 0.0
global IC_SPIEGEL_CENTER_Z    = 0.0
global IC_SPIEGEL_RADIUS      = 0.1
global IC_SPIEGEL_AMPLITUDE   = 0.05

global GPU_ASYNC_DEPTH = 8
global USE_STREAMS = true
global PREFETCH_NEIGHBORS = true

# Multi-GPU (Strategy C: replicated coarse + partitioned fine)
global MULTI_GPU_ENABLED = false
global GPU_SPLIT_LEVEL = 2   # Levels 1..split_level are replicated, rest partitioned

# 2026-06-24 consolidation: regularized_bgk retired (was the D3Q19 operator).
# :cumulant = D3Q27, :entropic = D3Q39.
const VALID_COLLISION_OPERATORS = Set([:cumulant, :entropic])
const VALID_RAMP_PROFILES = Set([:linear, :smoothstep, :smootherstep, :smoothstep7, :half_cosine])


function safe_get(dict, keys...; default=nothing)
    current = dict
    for (i, key) in enumerate(keys)
        if current === nothing || !isa(current, Dict) || !haskey(current, key)
            return default !== nothing ? default : error("Missing config key: $(join(keys[1:i], " → "))")
        end
        current = current[key]
    end
    return current === nothing && default !== nothing ? default : current
end

function config_has_path(dict, keys...)
    current = dict
    for key in keys
        if current === nothing || !isa(current, Dict) || !haskey(current, key)
            return false
        end
        current = current[key]
    end
    return true
end

# Resolve a feature flag whose DEFAULT depends on the active solver preset.
# An explicit per-case value (anywhere on the path) always wins; otherwise the
# preset-specific default is used. `pristine` and `vanilla` give the two
# baselines. Call AFTER SOLVER_PRESET is set.
function preset_flag(cfg, keys...; pristine, vanilla)
    if config_has_path(cfg, keys...)
        return safe_get(cfg, keys...)
    end
    return SOLVER_PRESET == "vanilla" ? vanilla : pristine
end

function parse_symmetric_analysis_mode(raw_value)
    if raw_value isa Bool
        return raw_value ? :mirror : :off
    elseif raw_value isa String
        value = lowercase(strip(raw_value))
        value in ("false", "off", "none", "no") && return :off
        value in ("true", "mirror", "symmetry", "symmetric") && return :mirror
        value == "ground" && return :ground
    end

    error("Invalid advanced.refinement.symmetric_analysis=$(repr(raw_value)). Valid values: false, true, or \"ground\".")
end

function symmetry_mode_summary(mode::Symbol=SYMMETRY_MODE)
    mode == :mirror && return "mirror symmetry at y=0 (slip wall)"
    mode == :ground && return "ground plane at y=0 (no-slip wall)"
    return "off (full domain)"
end

function parse_ground_up_axis(raw_value)
    axis = lowercase(strip(string(raw_value)))
    axis in ("x", "y", "z") || error("Invalid basic.ground_up_axis=$(repr(raw_value)). Valid values: \"x\", \"y\", or \"z\".")
    return Symbol(axis)
end

function parse_optional_object_center_m(config, axis::String)
    key = "object_center_$(axis)_m"
    config_has_path(config, "advanced", "domain", key) || return nothing
    raw = safe_get(config, "advanced", "domain", key)
    raw === nothing && return nothing
    raw isa Real && !(raw isa Bool) ||
        error("advanced.domain.$key must be a finite number of metres")
    value = Float64(raw)
    isfinite(value) || error("advanced.domain.$key must be a finite number of metres")
    return value
end

function load_case_configuration(runtime::CaseIDContract.RuntimeSelection;
                                 # lbm-io-opts (27 Oct 2026): the case's own root (<root>/CASES/<id>), as
                                 # setup_multilevel_domain; equals the solver root for the default queue.
                                 root_dir::AbstractString=dirname(dirname(runtime.case.case_dir)))
    CaseIDContract.verify_runtime_inputs(root_dir, runtime)
    selection = runtime.case
    parsed_config = YAML.load(selection.config_text)
    parsed_config isa AbstractDict || error("Captured config.yaml must parse to a mapping")
    CaseIDContract.verify_runtime_config(runtime, parsed_config)
    validated_case_id = CaseIDContract.validate_case_id(
        selection.case_id; source="captured load_case_configuration case ID")
    global CASE_DIR = selection.case_dir
    config_path = selection.config_path

    println("[Init] Loading: $validated_case_id")
    global CFG = parsed_config

    return _load_case_configuration_values!(validated_case_id, config_path, runtime)
end

function load_case_configuration(selection::CaseIDContract.CaseSelection;
                                 root_dir::AbstractString=dirname(dirname(selection.case_dir)))   # lbm-io-opts
    CaseIDContract.verify_case_selection(root_dir, selection)
    parsed_config = YAML.load(selection.config_text)
    parsed_config isa AbstractDict || error("Captured config.yaml must parse to a mapping")
    runtime = CaseIDContract.capture_runtime_selection(root_dir, selection, parsed_config)
    return load_case_configuration(runtime; root_dir=root_dir)
end

function load_case_configuration(case_folder_name)
    validated_case_id = CaseIDContract.validate_case_id(
        case_folder_name; source="load_case_configuration case ID")
    selection = CaseIDContract.capture_case_selection(
        joinpath(@__DIR__, "..", ".."), validated_case_id;
        source="load_case_configuration case ID")
    return load_case_configuration(selection)
end


function _load_case_configuration_values!(validated_case_id, config_path,
                                          runtime::CaseIDContract.RuntimeSelection)
    global STL_FILENAMES = [stl.declared for stl in runtime.stls]
    # ── SOLVER PRESET (2026-06-09; quick/pristine tiers 2026-06-10b;
    #     DEFAULT = pristine since 2026-06-11) ────────────────────────────────
    # `advanced.solver_preset` selects a feature baseline so users can switch
    # modes with one key, while still overriding any individual flag explicitly.
    #   "pristine" (DEFAULT) — the production D3Q19 mode: all AMR-accuracy
    #       features + the D3Q19 SEAM GATES (bspline-always halo macro,
    #       residual parent-fill, shared-bspline tile): AMR interface ≤ the
    #       monolith reference (Z155/Z197: 17× seam reduction, below the bulk
    #       plain-field jump) at ~1.3× wall vs quick at production scale.
    #       Why the slower mode is the default: quick-vs-pristine FORCES
    #       diverge materially (Z198/Z199 sphere transient: ΔCd up to 0.34,
    #       pristine in the physical range) — the quick-mode seam family is
    #       NOT aero-neutral. The gates are COMPILE-TIME consts set by the
    #       main.jl preset bootstrap BEFORE module load; env vars override.
    #   "quick" — the pre-2026-06-11 byte-identical legacy default: same
    #       feature set WITHOUT the seam gates (B-spline prolongation,
    #       cubic/quintic temporal, WALE, SVV, RR, feather, kernel split).
    #       Fast preview; carries the known box-edge seam artifact family
    #       (box outlines / corner quads at AMR interfaces). The recorded
    #       baselines of the Z15x/Z18x/Z19x reproducer cases pin this key
    #       explicitly.
    #   "vanilla"  — lean incompressible D3Q19: the accuracy features OFF + the
    #       single monolithic stream-collide kernel (d3q19_amr_interior_split
    #       false). MEASURED ~1.5× faster than the split AND a clean AMR
    #       interface. Use for incompressible aero where the extra AMR-coupling
    #       sophistication is not needed.
    # An explicit per-case flag ALWAYS wins over the preset baseline.
    global SOLVER_PRESET = lowercase(string(safe_get(CFG, "advanced", "solver_preset"; default="pristine")))
    if !(SOLVER_PRESET in ("quick", "pristine", "vanilla"))
        error("Unknown solver_preset '$SOLVER_PRESET'; valid values are quick, pristine, or vanilla.")
    end
    println("[Init] Solver preset: $SOLVER_PRESET")
    # Per-case cross-check against the compile-time seam gates (set by the
    # main.jl bootstrap from the FIRST case of the run, or by env override).
    # A mismatch is not fatal — the case runs with the gates as compiled —
    # but it must be loud: a 'pristine' case without the gates does NOT get
    # the seam guarantee, and a 'quick' case with them is ~1.55× slower and
    # not byte-identical to legacy.
    let seam_gates_on = !_D3Q19_BSPLINE_MACRO_TIME_CONSISTENT &&
                        _D3Q19_RESIDUAL_PARENT_FILL && _D3Q19_BSPLINE_TILE
        if SOLVER_PRESET == "pristine" && !seam_gates_on
            @warn "solver_preset=pristine but the compile-time D3Q19 seam gates are NOT all in their pristine state (bspline_macro_time_consistent=$(_D3Q19_BSPLINE_MACRO_TIME_CONSISTENT), residual_parent_fill=$(_D3Q19_RESIDUAL_PARENT_FILL), bspline_tile=$(_D3Q19_BSPLINE_TILE)) — env override or mixed-preset multi-case run. This case runs WITHOUT the full seam guarantee."
        elseif SOLVER_PRESET != "pristine" && seam_gates_on
            @warn "solver_preset=$(SOLVER_PRESET) but the compile-time D3Q19 seam gates are in their PRISTINE state (env override or mixed-preset multi-case run). This case runs ~1.55× slower than quick and is not byte-identical to legacy."
        end
    end

    global STL_FILES = [stl.path for stl in runtime.stls]
    # Backward-compatible aliases (first entry)
    global STL_FILENAME = STL_FILENAMES[1]
    global STL_FILE = STL_FILES[1]
    global STL_SCALE = Float64(safe_get(CFG, "basic", "stl_scale"))
    global OUT_DIR_NAME = runtime.output.leaf
    global OUT_DIR = runtime.output.path
    global SURFACE_RESOLUTION = Int(safe_get(CFG, "basic", "surface_resolution"))
    global NUM_LEVELS_CONFIG = Int(safe_get(CFG, "basic", "num_levels"))
    global MINIMUM_FACET_SIZE = Float64(safe_get(CFG, "basic", "minimum_facet_size"; default=0.0))
    # Explicit subdivision switch. When false, force MINIMUM_FACET_SIZE = 0 so the
    # subdivide_mesh(min_size <= 0 → return mesh) short-circuit skips subdivision
    # entirely (raw STL triangles used as-is). Backward-compatible: absent key →
    # true → MINIMUM_FACET_SIZE unchanged.
    global SUBDIVISION_ENABLED = Bool(safe_get(CFG, "basic", "subdivision"; default=true))
    if !SUBDIVISION_ENABLED && MINIMUM_FACET_SIZE > 0.0
        @info "[Config] subdivision: false → forcing minimum_facet_size 0 (was $(MINIMUM_FACET_SIZE)); raw STL triangles used as-is."
        global MINIMUM_FACET_SIZE = 0.0
    end

    # ── NEW mesh block (v1.11) — auto-sizing path ──
    # When mesh.min_dx > 0, the solver takes this as the finest-level cell size
    # and derives SURFACE_RESOLUTION / num_levels / domain extents / VRAM fit
    # itself (see compute_domain_auto_extents in physics_scaling.jl). When
    # mesh.min_dx ≤ 0 (legacy), the old basic.surface_resolution path is used.
    # mesh.num_levels, when present, overrides basic.num_levels.
    global MESH_MIN_DX = Float64(safe_get(CFG, "mesh", "min_dx"; default=0.0))
    if Int(safe_get(CFG, "mesh", "num_levels"; default=0)) > 0
        global NUM_LEVELS_CONFIG = Int(safe_get(CFG, "mesh", "num_levels"; default=0))
    end
    global MESH_VRAM_BUDGET_GB = Float64(safe_get(CFG, "mesh", "vram_budget_gb"; default=0.0))
    global MESH_SAFETY_FACTOR  = Float64(safe_get(CFG, "mesh", "safety_factor"; default=1.15))
    
    local raw_symmetry_mode = safe_get(CFG, "advanced", "refinement", "symmetric_analysis"; default=false)
    local symmetry_mode = parse_symmetric_analysis_mode(raw_symmetry_mode)
    global SYMMETRY_MODE = symmetry_mode
    global SYMMETRIC_ANALYSIS = symmetry_mode == :mirror
    global HALF_Y_DOMAIN = symmetry_mode != :off
    global GROUND_ANALYSIS = symmetry_mode == :ground
    global GROUND_UP_AXIS = parse_ground_up_axis(safe_get(CFG, "basic", "ground_up_axis"; default="z"))
    global ALPHA_DEG = Float64(safe_get(CFG, "basic", "alpha"; default=0.0))
    global BETA_DEG  = Float64(safe_get(CFG, "basic", "beta"; default=0.0))
    if SYMMETRIC_ANALYSIS && BETA_DEG != 0.0
        println("[Init] ⚠ beta=$(BETA_DEG)° ignored because symmetric_analysis=true (forcing beta=0)")
        global BETA_DEG = 0.0
    end
    global REFERENCE_AREA_FULL_MODEL = Float64(safe_get(CFG, "basic", "reference_area_of_full_model"; default=0.0))
    # AUDIT FIX (2026-06-03): REFERENCE_AREA_CONFIG must be the FULL-model area in
    # ALL modes. For symmetric_analysis the integrated half-domain forces are
    # already DOUBLED to full-model forces (surface.jl integrate_surface_forces!),
    # and the auto-area fallback (physics_scaling.jl) likewise returns the FULL
    # area (half-span × ·2). Halving the user-supplied full area here while the
    # forces were doubled produced coefficients 2× too large. Use the full area
    # as-given so Cd/Cl/Cs = F_full / (q·A_full) is correct in both branches.
    global REFERENCE_AREA_CONFIG = REFERENCE_AREA_FULL_MODEL
    global REFERENCE_CHORD_CONFIG = Float64(safe_get(CFG, "basic", "reference_chord"; default=0.0))
    global REFERENCE_LENGTH_FOR_MESHING = Float64(safe_get(CFG, "basic", "reference_length_for_meshing"; default=0.0))
    global REFERENCE_DIMENSION = Symbol(safe_get(CFG, "basic", "reference_dimension"; default="x"))
    
    global FLUID_DENSITY = Float64(safe_get(CFG, "basic", "fluid", "density"; default=1.225))
    global FLUID_KINEMATIC_VISCOSITY = Float64(safe_get(CFG, "basic", "fluid", "kinematic_viscosity"; default=1.5e-5))
    global FLOW_VELOCITY = Float64(safe_get(CFG, "basic", "flow", "velocity"; default=10.0))

    # ── Phase C (2026-06-24): automatic regime selection ──────────────────────
    # From the user's velocity + characteristic length + fluid, compute Ma & Re and
    # auto-pick {lattice, collision_operator}. Any explicitly-set key wins (override).
    # Runs here because all of its inputs (velocity, length, viscosity, sos) are now
    # read; LATTICE_TYPE / COLLISION_OPERATOR are finalized from REGIME_SELECTION below.
    local _sos_regime = Float64(safe_get(CFG, "basic", "fluid", "speed_of_sound"; default=343.0))
    local _len_regime = REFERENCE_CHORD_CONFIG > 0.0 ? REFERENCE_CHORD_CONFIG :
                        (REFERENCE_LENGTH_FOR_MESHING > 0.0 ? REFERENCE_LENGTH_FOR_MESHING : 1.0)
    global REGIME_MA = FLOW_VELOCITY / _sos_regime
    global REGIME_RE = FLOW_VELOCITY * _len_regime / max(FLUID_KINEMATIC_VISCOSITY, 1.0e-30)
    global REGIME_SELECTION = resolve_regime_solver(CFG, REGIME_MA, REGIME_RE)

    global DOMAIN_LENGTH_COVERED = Float64(safe_get(CFG, "basic", "simulation", "domain_length_covered"; default=0.0))
    global STEPS = Int(safe_get(CFG, "basic", "simulation", "steps"; default=0))
    global RAMP_STEPS = Int(safe_get(CFG, "basic", "simulation", "ramp_steps"))
    local ramp_profile_raw = replace(lowercase(String(safe_get(CFG, "basic", "simulation", "ramp_profile"; default="half_cosine"))), "-" => "_")
    local ramp_profile = Symbol(ramp_profile_raw == "cosine" ? "half_cosine" : ramp_profile_raw)
    if !(ramp_profile in VALID_RAMP_PROFILES)
        error("Unknown basic.simulation.ramp_profile: \"$ramp_profile_raw\". Valid: $(join(sort(collect(VALID_RAMP_PROFILES)), ", "))")
    end
    global RAMP_PROFILE = ramp_profile
    global RAMP_START_FRACTION = Float32(safe_get(CFG, "basic", "simulation", "ramp_start_fraction"; default=0.0))
    (0.0f0 <= RAMP_START_FRACTION <= 1.0f0) ||
        error("basic.simulation.ramp_start_fraction must be between 0 and 1")
    global OUTPUT_FREQ = Int(safe_get(CFG, "basic", "simulation", "output_freq"))
    global SLICE_OUTPUT_FREQ = Int(safe_get(CFG, "basic", "simulation", "slice_output_freq"; default=OUTPUT_FREQ))
    global SLICE_COSMETIC_FILTERING = Bool(safe_get(CFG, "advanced", "slice", "cosmetic_filtering"; default=false))
    # SLICE EXPORT MODE (2026-06-22, user request): "debug" = export ALL slice fields
    # + the amr_seam_*.csv diagnostics; "production" = export only the 7 inspection
    # fields (velocity, velocity_highpass, vorticity, pressure, density, mach,
    # schlieren) and NO seam CSVs. Accepts the legacy `field_set` key too.
    let _sm_raw = lowercase(string(safe_get(CFG, "advanced", "slice", "mode";
                       default=safe_get(CFG, "advanced", "slice", "field_set"; default="production"))))
        global SLICE_FIELD_SET = _sm_raw == "debug" ? :all :
                                 _sm_raw == "production" ? :production :
                                 Symbol(_sm_raw)   # allow legacy :geometric/:amr_debug etc.
        if !(SLICE_FIELD_SET in (:all, :production, :geometric, :amr_debug))
            @warn "Unknown advanced.slice.mode '$_sm_raw' — falling back to 'production'."
            global SLICE_FIELD_SET = :production
        end
        println("[Init]    Slice export mode: $_sm_raw ⇒ $SLICE_FIELD_SET")
    end
    # lbm-io-opts (27 Oct 2026): opt-in slice/IO keys and build-only mode; the parser and
    # its LBMIO_* globals live in src/io/lbm_io_opts.jl (defaults = legacy behaviour).
    lbmio_load_options!(CFG)
    
    global OUTPUT_DENSITY = safe_get(CFG, "basic", "simulation", "output_fields", "density"; default=true)
    global OUTPUT_VELOCITY = safe_get(CFG, "basic", "simulation", "output_fields", "velocity"; default=true)
    global OUTPUT_VEL_MAG = safe_get(CFG, "basic", "simulation", "output_fields", "velocity_magnitude"; default=true)
    global OUTPUT_VORTICITY = safe_get(CFG, "basic", "simulation", "output_fields", "vorticity"; default=true)
    global OUTPUT_OBSTACLE = safe_get(CFG, "basic", "simulation", "output_fields", "obstacle"; default=true)
    global OUTPUT_LEVEL = safe_get(CFG, "basic", "simulation", "output_fields", "level"; default=true)
    global OUTPUT_BOUZIDI = safe_get(CFG, "basic", "simulation", "output_fields", "bouzidi"; default=true)
    global OUTPUT_MACH = safe_get(CFG, "basic", "simulation", "output_fields", "mach"; default=true)
    global OUTPUT_PRESSURE = safe_get(CFG, "basic", "simulation", "output_fields", "pressure"; default=true)
    global OUTPUT_WALL_DISTANCE = safe_get(CFG, "basic", "simulation", "output_fields", "wall_distance"; default=true)
    global OUTPUT_SCHLIEREN = safe_get(CFG, "basic", "simulation", "output_fields", "schlieren"; default=true)
    global OUTPUT_STARTUP_SLICES = Bool(safe_get(CFG, "basic", "simulation", "output_startup_slices"; default=true))
    global OUTPUT_STARTUP_VOLUME = Bool(safe_get(CFG, "basic", "simulation", "output_startup_volume"; default=true))
    
    # Phase C: operator from the regime selector (user-pinned value respected inside it).
    global COLLISION_OPERATOR = REGIME_SELECTION.operator
    if !(COLLISION_OPERATOR in VALID_COLLISION_OPERATORS)
        error("Unknown collision_operator: \"$COLLISION_OPERATOR\". Valid options: $(join(sort(collect(VALID_COLLISION_OPERATORS)), ", "))")
    end

    global U_TARGET = Float32(safe_get(CFG, "advanced", "numerics", "u_lattice"; default=0.08))
    global C_SMAGO = Float32(safe_get(CFG, "advanced", "numerics", "c_wale"; default=0.20))
    global TAU_MIN = Float32(safe_get(CFG, "advanced", "numerics", "tau_min"; default=0.501))
    global TAU_SAFETY_FACTOR = Float32(safe_get(CFG, "advanced", "numerics", "tau_safety_factor"; default=1.0))
    global INLET_TURBULENCE_INTENSITY = Float32(safe_get(CFG, "advanced", "numerics", "inlet_turbulence_intensity"; default=0.01))

    local _nu_sgs_bg_config = Float32(safe_get(CFG, "advanced", "numerics", "nu_sgs_background"; default=0.0))
    global NU_SGS_BACKGROUND = _nu_sgs_bg_config
    global SPONGE_BLEND_DISTRIBUTIONS = safe_get(CFG, "advanced", "numerics", "sponge_blend_distributions"; default=false)
    global TEMPORAL_INTERPOLATION = safe_get(CFG, "advanced", "numerics", "temporal_interpolation"; default=true)
    global INTERFACE_MACRO_TEMPORAL_INTERPOLATION = safe_get(CFG, "advanced", "numerics", "interface_macro_temporal_interpolation"; default=false)
    # B1 P3 — cubic Hermite temporal interpolation (extends quadratic Pass176/180)
    global INTERFACE_CUBIC_TEMPORAL = safe_get(CFG, "advanced", "refinement", "interface_cubic_temporal"; default=false)
    # B1 P3 RK4-equivalent — quintic Lagrange (C³) temporal interpolation. Needs the
    # cubic anchor chain; auto-enable cubic (+warn) if quintic is requested alone.
    global INTERFACE_QUINTIC_TEMPORAL = safe_get(CFG, "advanced", "refinement", "interface_quintic_temporal"; default=false)
    if INTERFACE_QUINTIC_TEMPORAL && !INTERFACE_CUBIC_TEMPORAL
        @warn "refinement.interface_quintic_temporal=true requires interface_cubic_temporal (shared anchor chain); auto-enabling cubic."
        global INTERFACE_CUBIC_TEMPORAL = true
    end
    if INTERFACE_QUINTIC_TEMPORAL && !INTERFACE_MACRO_TEMPORAL_INTERPOLATION
        @warn "refinement.interface_quintic_temporal=true requires numerics.interface_macro_temporal_interpolation=true; the quintic branch only fires on the macro-temporal path."
    end
    if INTERFACE_QUINTIC_TEMPORAL
        @info "B1 P3 — quintic Lagrange (C³) AMR temporal interpolation ACTIVE (5 anchors: oldest2,oldest,older,old,new)."
    end
    # High-order AMR-continuity SCHEDULE (2026-06-23): when to pay the ~3-4× cost
    # of C4/C3. :always | :off | :steady_state (cheap C0 during ramp, escalate near
    # steady state). Accepts string or symbol; defaults to :always (legacy behavior).
    local _hosched = safe_get(CFG, "advanced", "refinement", "high_order_schedule"; default="always")
    global INTERFACE_HIGH_ORDER_SCHEDULE = Symbol(lowercase(string(_hosched)))
    if !(INTERFACE_HIGH_ORDER_SCHEDULE in (:always, :off, :steady_state))
        @warn "refinement.high_order_schedule=\"$_hosched\" invalid; using :always. Valid: always | off | steady_state"
        global INTERFACE_HIGH_ORDER_SCHEDULE = :always
    end
    global INTERFACE_HIGH_ORDER_U_THRESHOLD = Float32(safe_get(CFG, "advanced", "refinement", "high_order_u_threshold"; default=0.98))
    global INTERFACE_HIGH_ORDER_SETTLE_STEPS = Int32(safe_get(CFG, "advanced", "refinement", "high_order_settle_steps"; default=0))
    if INTERFACE_HIGH_ORDER_SCHEDULE != :always
        @info "High-order AMR continuity SCHEDULE = $INTERFACE_HIGH_ORDER_SCHEDULE" *
              (INTERFACE_HIGH_ORDER_SCHEDULE == :steady_state ?
               " (C0 during ramp → C4/C3 once u/U_target ≥ $INTERFACE_HIGH_ORDER_U_THRESHOLD + $INTERFACE_HIGH_ORDER_SETTLE_STEPS settle steps)" :
               " (C4/C3 escalation disabled — pure C0 trilinear, fastest)")
    end
    global INTERFACE_TEMPORAL_WEIGHT_BIAS = Float32(safe_get(CFG, "advanced", "numerics", "interface_temporal_weight_bias"; default=0.0))
    global INTERFACE_TEMPORAL_WEIGHT_BIAS_RISE_U_START = Float32(safe_get(CFG, "advanced", "numerics", "interface_temporal_weight_bias_rise_u_start"; default=0.0))
    global INTERFACE_TEMPORAL_WEIGHT_BIAS_RISE_U_END = Float32(safe_get(CFG, "advanced", "numerics", "interface_temporal_weight_bias_rise_u_end"; default=0.0))
    global INTERFACE_TEMPORAL_WEIGHT_BIAS_DECAY_U_START = Float32(safe_get(CFG, "advanced", "numerics", "interface_temporal_weight_bias_decay_u_start"; default=1.0))
    global INTERFACE_TEMPORAL_WEIGHT_BIAS_DECAY_U_END = Float32(safe_get(CFG, "advanced", "numerics", "interface_temporal_weight_bias_decay_u_end"; default=1.0))
    if INTERFACE_TEMPORAL_WEIGHT_BIAS_RISE_U_START < 0.0f0 || INTERFACE_TEMPORAL_WEIGHT_BIAS_RISE_U_START > 1.0f0
        error("interface_temporal_weight_bias_rise_u_start must be in [0,1], got $INTERFACE_TEMPORAL_WEIGHT_BIAS_RISE_U_START")
    end
    if INTERFACE_TEMPORAL_WEIGHT_BIAS_RISE_U_END < 0.0f0 || INTERFACE_TEMPORAL_WEIGHT_BIAS_RISE_U_END > 1.0f0
        error("interface_temporal_weight_bias_rise_u_end must be in [0,1], got $INTERFACE_TEMPORAL_WEIGHT_BIAS_RISE_U_END")
    end
    if INTERFACE_TEMPORAL_WEIGHT_BIAS_RISE_U_END < INTERFACE_TEMPORAL_WEIGHT_BIAS_RISE_U_START
        error("interface_temporal_weight_bias_rise_u_end must be >= interface_temporal_weight_bias_rise_u_start")
    end
    if INTERFACE_TEMPORAL_WEIGHT_BIAS_DECAY_U_START < 0.0f0 || INTERFACE_TEMPORAL_WEIGHT_BIAS_DECAY_U_START > 1.0f0
        error("interface_temporal_weight_bias_decay_u_start must be in [0,1], got $INTERFACE_TEMPORAL_WEIGHT_BIAS_DECAY_U_START")
    end
    if INTERFACE_TEMPORAL_WEIGHT_BIAS_DECAY_U_END < 0.0f0 || INTERFACE_TEMPORAL_WEIGHT_BIAS_DECAY_U_END > 1.0f0
        error("interface_temporal_weight_bias_decay_u_end must be in [0,1], got $INTERFACE_TEMPORAL_WEIGHT_BIAS_DECAY_U_END")
    end
    if INTERFACE_TEMPORAL_WEIGHT_BIAS_DECAY_U_END < INTERFACE_TEMPORAL_WEIGHT_BIAS_DECAY_U_START
        error("interface_temporal_weight_bias_decay_u_end must be >= interface_temporal_weight_bias_decay_u_start")
    end
    global RESTRICT_FINE_TO_PARENT = safe_get(CFG, "advanced", "numerics", "restrict_fine_to_parent"; default=true)
    global INTERFACE_ENTROPIC_H_SCALE = Float32(safe_get(CFG, "advanced", "numerics", "interface_entropic_h_scale"; default=0.0))

    global CUMULANT_OMEGA_BULK      = Float32(safe_get(CFG, "advanced", "numerics", "cumulant", "omega_bulk"; default=1.0))
    global CUMULANT_OMEGA_3         = Float32(safe_get(CFG, "advanced", "numerics", "cumulant", "omega_3"; default=1.0))
    global CUMULANT_OMEGA_4         = Float32(safe_get(CFG, "advanced", "numerics", "cumulant", "omega_4"; default=1.0))
    global CUMULANT_ADAPTIVE_OMEGA4 = safe_get(CFG, "advanced", "numerics", "cumulant", "adaptive_omega_4"; default=true)
    global CUMULANT_LAMBDA_PARAM    = Float32(safe_get(CFG, "advanced", "numerics", "cumulant", "lambda_param"; default=1.0/6.0))

    local limiter_str = safe_get(CFG, "advanced", "numerics", "cumulant", "limiter"; default="factored")
    global CUMULANT_LIMITER = Symbol(limiter_str)
    if !(CUMULANT_LIMITER in VALID_CUMULANT_LIMITERS)
        error("Unknown cumulant limiter: \"$CUMULANT_LIMITER\". Valid: $(join(sort(collect(VALID_CUMULANT_LIMITERS)), ", "))")
    end

    # ── Entropic KBC parameters ──
    local alpha_solver_str = safe_get(CFG, "advanced", "numerics", "entropic", "alpha_solver"; default="bisection")
    global ENTROPIC_ALPHA_SOLVER = Symbol(alpha_solver_str)
    if !(ENTROPIC_ALPHA_SOLVER in VALID_ENTROPIC_SOLVERS)
        error("Unknown entropic alpha_solver: \"$ENTROPIC_ALPHA_SOLVER\". Valid: $(join(sort(collect(VALID_ENTROPIC_SOLVERS)), ", "))")
    end
    global ENTROPIC_ALPHA_MAX_ITER = Int(safe_get(CFG, "advanced", "numerics", "entropic", "alpha_max_iter"; default=10))
    global ENTROPIC_ALPHA_TOLERANCE = Float32(safe_get(CFG, "advanced", "numerics", "entropic", "alpha_tolerance"; default=1.0e-5))
    global ENTROPIC_SIGMA = Float32(safe_get(CFG, "advanced", "numerics", "entropic", "sigma"; default=0.0))

    global COMPRESSIBILITY_CORRECTION = safe_get(CFG, "advanced", "numerics", "compressibility_correction"; default=false)

    # Diagnostic: compute Π_neq at slice export time (zero cost when disabled)
    global COMPUTE_PI_NEQ = safe_get(CFG, "advanced", "diagnostics", "compute_pi_neq"; default=false)

    # Phase B: variable ν_sgs from local Π_neq. When true, allocates persistent
    # pi_neq field in BlockLevel and collision kernels read per-cell SGS viscosity.
    global VARIABLE_NU_SGS = safe_get(CFG, "advanced", "numerics", "variable_nu_sgs", "enabled"; default=false)
    # Π_neq as primary stress carrier across AMR levels. Default: on for entropic
    # (the prolongation will route through the tensor-projection path instead of
    # mirror-decomposing raw populations, which is far more stable for high-Ma runs).
    global STRESS_TENSOR_TRANSFER = safe_get(CFG, "advanced", "numerics", "stress_tensor_transfer";
                                             default=(COLLISION_OPERATOR == :entropic))

    # Knudsen-adaptive τ (Latt 2020 Eq. 2.28). When true, the collision kernel
    # applies Latt's piecewise-constant α(ε) multiplier to τ using Π_neq as a
    # Knudsen-number proxy. Disabled by default (paper-quality Tier A-2 knob).
    global KNUDSEN_ADAPTIVE_TAU = safe_get(CFG, "advanced", "numerics", "knudsen_adaptive_tau";
                                           default=false)
    global VARIABLE_NU_SGS_C_PI = Float32(safe_get(CFG, "advanced", "numerics", "variable_nu_sgs", "c_pi"; default=0.2))
    local _floor_config = Float32(safe_get(CFG, "advanced", "numerics", "variable_nu_sgs", "floor"; default=0.002))
    local _variable_nu_sgs_mach_floor_boost = safe_get(CFG, "advanced", "numerics", "variable_nu_sgs", "mach_floor_boost"; default=true)
    global VARIABLE_NU_SGS_FLOOR = _floor_config

    # Phase B4: adaptive ENTROPIC_SIGMA (h-damping varies per cell)
    global VARIABLE_SIGMA = safe_get(CFG, "advanced", "numerics", "variable_sigma", "enabled"; default=false)
    global VARIABLE_SIGMA_MIN = Float32(safe_get(CFG, "advanced", "numerics", "variable_sigma", "sigma_min"; default=0.3))
    global VARIABLE_SIGMA_MAX = Float32(safe_get(CFG, "advanced", "numerics", "variable_sigma", "sigma_max"; default=0.95))
    global VARIABLE_SIGMA_PI_REF = Float32(safe_get(CFG, "advanced", "numerics", "variable_sigma", "pi_ref"; default=0.01))

    # Phase B4: adaptive sigma_even for thermal DDF kernel
    global VARIABLE_SIGMA_THERMAL = safe_get(CFG, "advanced", "numerics", "variable_sigma_thermal", "enabled"; default=false)
    global VARIABLE_SIGMA_TH_MIN = Float32(safe_get(CFG, "advanced", "numerics", "variable_sigma_thermal", "sigma_min"; default=0.0))
    global VARIABLE_SIGMA_TH_MAX = Float32(safe_get(CFG, "advanced", "numerics", "variable_sigma_thermal", "sigma_max"; default=0.15))
    global VARIABLE_SIGMA_TH_PI_REF = Float32(safe_get(CFG, "advanced", "numerics", "variable_sigma_thermal", "pi_ref"; default=0.025))

    # Phase C1: Xu-Sagaut sponge (replaces pre-collision distribution blending)
    global SPONGE_XU_SAGAUT = safe_get(CFG, "advanced", "numerics", "sponge_xu_sagaut"; default=false)

    # Force computation method: "prandtl" (1D wall-law estimate, default) or
    # "stress_tensor" (projects σ_ij = −(τ−0.5)/(τ−1)·Π_neq^post,ij onto surface normal).
    # stress_tensor is O(ε²) vs Prandtl's O(ε); required for accurate supersonic drag.
    local _fm_str = String(safe_get(CFG, "advanced", "forces", "method"; default="prandtl"))
    global FORCE_METHOD = Symbol(_fm_str)
    if !(FORCE_METHOD in Set([:prandtl, :stress_tensor]))
        error("Unknown forces.method: \"$_fm_str\". Valid: prandtl, stress_tensor")
    end

    # ── Lattice type (Phase C: from the regime selector; user-pinned value respected) ──
    global LATTICE_TYPE = REGIME_SELECTION.lattice
    # PonD opt-in (default OFF ⇒ byte-identical existing cases). When enabled it overrides
    # the regime lattice choice: PonD uses the 39-velocity set for the 2nd-order Grad closure.
    global POND_ENABLED = safe_get(CFG, "advanced", "numerics", "pond", "enabled"; default=false)
    if POND_ENABLED
        global LATTICE_TYPE = :pond
    end
    if !(LATTICE_TYPE in VALID_LATTICE_TYPES)
        error("Unknown lattice type: \"$LATTICE_TYPE\". Valid: $(join(sort(collect(VALID_LATTICE_TYPES)), ", "))")
    end
    global N_DIRS = (LATTICE_TYPE == :d3q39 || LATTICE_TYPE == :pond) ? Int32(39) : Int32(27)
    global NUMERICAL_EQUILIBRIUM = (LATTICE_TYPE == :d3q39)  # PonD uses closed-form Grad, not the Newton eq
    if LATTICE_TYPE == :d3q39 && COLLISION_OPERATOR != :entropic
        error("D3Q39 lattice requires collision_operator: entropic. " *
              "The cumulant operator uses D3Q27-specific structure (Chimera transform).")
    end
    if LATTICE_TYPE == :d3q27 && COLLISION_OPERATOR != :cumulant
        error("D3Q27 lattice requires collision_operator: cumulant " *
              "(the legacy entropic/regularized_bgk D3Q27 operators were retired 2026-06-24).")
    end

    # ── Phase 3.9: Compressibility-regime safety guard ──
    # Given the user's physical flow velocity + speed of sound, check that the
    # lattice choice is appropriate. D3Q27 (cs²=1/3) is stable up to Ma ≈ 0.3
    # without the O(u⁴) correction; D3Q39 (cs²=2/3, entropic) covers Ma 0 → ~1.6
    # unshifted and up to ~2.5 with a Galilean shift. Catch user error at config
    # load rather than at timestep 50 when the sim explodes. We read speed_of_sound
    # directly from CFG here because SPEED_OF_SOUND_PHYSICAL is set a few blocks
    # below — this check runs before that global is assigned.
    local _sos_check = Float64(safe_get(CFG, "basic", "fluid", "speed_of_sound"; default=343.0))
    local _Ma_phys_autocheck = FLOW_VELOCITY / _sos_check
    local _variable_nu_sgs_explicit = config_has_path(CFG, "advanced", "numerics", "variable_nu_sgs", "enabled")
    if !_variable_nu_sgs_explicit && LATTICE_TYPE == :d3q39 && _Ma_phys_autocheck >= 0.6 && SURFACE_RESOLUTION <= 256
        global VARIABLE_NU_SGS = true
        @printf("[Stability] Auto-enabled variable_nu_sgs for D3Q39 Ma=%.3f, SR=%d (pass-168 default rule)\n",
                _Ma_phys_autocheck, SURFACE_RESOLUTION)
    end
    # Phase C (2026-06-24): D3Q27 high-Re sibling of the pass-168 rule. At aircraft
    # Reynolds numbers the WALE-only SGS under-dissipates the unresolved cascade; the
    # Π_neq adaptive SGS (ported to cumulant in Phase B) is the gold standard. Auto-on
    # only when (a) the user didn't set variable_nu_sgs.enabled, (b) D3Q27, and (c) Re
    # is high (REGIME_RE_HIGH_RE_THRESHOLD). User-set value always wins.
    if !_variable_nu_sgs_explicit && LATTICE_TYPE == :d3q27 && REGIME_RE >= REGIME_RE_HIGH_RE_THRESHOLD
        global VARIABLE_NU_SGS = true
        @printf("[Stability] Auto-enabled variable_nu_sgs for D3Q27 high-Re (Re=%.3g ≥ %.0e); the gold-standard Π_neq adaptive SGS.\n",
                REGIME_RE, REGIME_RE_HIGH_RE_THRESHOLD)
    end
    if _Ma_phys_autocheck > 0.3 && LATTICE_TYPE == :d3q27 && !COMPRESSIBILITY_CORRECTION
        println("[Regime] ⚠ Ma_physical = $(round(_Ma_phys_autocheck, digits=3)) > 0.3 with D3Q27 and no compressibility_correction.")
        println("[Regime]    D3Q27 O(u²) equilibrium develops significant compressibility bias above Ma=0.3.")
        println("[Regime]    Recommendation: set advanced.numerics.lattice: \"d3q39\" and collision_operator: \"entropic\",")
        println("[Regime]    or enable advanced.numerics.compressibility_correction: true to extend D3Q27 to Ma ≈ 0.6.")
    end
    if _Ma_phys_autocheck > 0.8 && LATTICE_TYPE == :d3q27
        error("Ma_physical = $(round(_Ma_phys_autocheck, digits=3)) > 0.8 with D3Q27. " *
              "D3Q27 cannot stably resolve transonic flow; switch to lattice: \"d3q39\" " *
              "and collision_operator: \"entropic\".")
    end
    if _Ma_phys_autocheck > 1.6 && LATTICE_TYPE == :d3q39
        local _shift_configured = max(abs(Int32(safe_get(CFG, "advanced", "numerics", "shifted_lattice", "shift_x"; default=0))),
                                       abs(Int32(safe_get(CFG, "advanced", "numerics", "shifted_lattice", "shift_y"; default=0))),
                                       abs(Int32(safe_get(CFG, "advanced", "numerics", "shifted_lattice", "shift_z"; default=0))))
        local _shift_auto = Bool(safe_get(CFG, "advanced", "numerics", "shifted_lattice", "auto"; default=false))
        if _shift_configured == 0 && !_shift_auto
            println("[Regime] ⚠ Ma_physical = $(round(_Ma_phys_autocheck, digits=3)) > 1.6 without shifted lattice.")
            println("[Regime]    Unshifted D3Q39 becomes unstable beyond Ma ≈ 1.6 (Latt 2020 Fig. 3).")
            println("[Regime]    Recommendation: set advanced.numerics.shifted_lattice.auto: true.")
        end
    end

    # ── Compressible mode (Phase 3) ──
    # When enabled, u_lattice is auto-computed so Ma_lattice = Ma_physical.
    # This enables true supersonic flow with shock waves on the D3Q39 lattice.
    global COMPRESSIBLE_MODE = safe_get(CFG, "advanced", "numerics", "compressible_mode"; default=false)
    global SPEED_OF_SOUND_PHYSICAL = Float64(safe_get(CFG, "basic", "fluid", "speed_of_sound"; default=343.0))

    if COMPRESSIBLE_MODE
        if LATTICE_TYPE != :d3q39
            error("compressible_mode requires lattice: d3q39. " *
                  "The D3Q27 lattice cannot resolve supersonic flows.")
        end
        if COLLISION_OPERATOR != :entropic
            error("compressible_mode requires collision_operator: entropic. " *
                  "The entropic H-theorem provides nonlinear stability for supersonic flows.")
        end
        local _Ma_phys_comp = FLOW_VELOCITY / SPEED_OF_SOUND_PHYSICAL
        # NOTE: CS2_PHYSICS is still at default (1/3) here — it gets updated to 2/3
        # in main.jl AFTER config loading. Use LATTICE_TYPE to compute cs² directly.
        local _cs2_comp = LATTICE_TYPE == :d3q39 ? 2.0/3.0 : 1.0/3.0
        local _cs_lattice = sqrt(_cs2_comp)
        global U_TARGET = Float32(_Ma_phys_comp * _cs_lattice)
        @printf("[Compressible] Ma_physical = %.4f, cs_lattice = %.6f\n", _Ma_phys_comp, _cs_lattice)
        @printf("[Compressible] Auto u_lattice = %.6f (Ma_lattice = Ma_physical)\n", U_TARGET)
        if _Ma_phys_comp > 1.6
            println("[Compressible] ⚠ WARNING: Ma > 1.6 may exceed unshifted D3Q39 stability limit")
            println("[Compressible]   Consider shifted_lattice.auto: true for Ma > 1.6 (Phase 3.2)")
        end
        # Prior note: temporal_interpolation was force-disabled for compressible D3Q39
        # runs due to a "non-physical ringing" observation. Per Lagrava 2012 Sec 3.5,
        # time interpolation is mandatory for correct coarse-fine coupling — disabling
        # it propagates phase error per refinement level and left the innermost level
        # stuck at u≈0 in 3-level AMR runs. The override is removed so the config's
        # setting is honoured; if ringing re-appears, it should be diagnosed rather
        # than masked.
        if TEMPORAL_INTERPOLATION
            println("[Compressible] temporal_interpolation = true (config) — Lagrava 2012 prescription for compressible D3Q39")
        end
    end

    # ── Mach-adaptive ν_sgs transonic hardening ──
    # In the transonic regime (Ma ≈ 0.8–1.2), weak embedded shocks don't trigger
    # the entropic H-theorem strongly enough, and variable_nu_sgs Π_neq mechanism
    # barely activates. This leaves cells at τ ≈ 0.5 with no dissipation margin.
    # A Gaussian bump centered at Ma=1.0 adds ~0.012 extra ν_sgs, giving
    # τ_floor ≈ 0.518 — sufficient for the entropic solver. Decays quickly:
    # negligible at Ma=0.5 and Ma≥1.5 where other mechanisms provide stability.
    # Must run AFTER COMPRESSIBLE_MODE and U_TARGET are finalized (above).
    if COMPRESSIBLE_MODE
        local _cs_lat_stab = LATTICE_TYPE == :d3q39 ? sqrt(2.0/3.0) : sqrt(1.0/3.0)
        local _Ma_lat_stab = Float64(U_TARGET) / _cs_lat_stab
        # This is a transonic safety knob, not a low-Mach AMR-validation model.
        # Without the explicit Ma window, Ma=0.10 still receives ν_sgs≈4.4e-3,
        # which masks synthetic smooth-flow AMR errors and changes the test.
        # 2026-06-01: keep this boost mild. A 0.05 floor overdamped the
        # transonic cylinder wake and suppressed shear-layer roll-up.
        local _transonic_window = (0.5 <= _Ma_lat_stab <= 1.5)
        local _transonic_bump = _transonic_window ? 0.004 * exp(-3.0 * (_Ma_lat_stab - 1.0)^2) : 0.0
        local _local_stability_active = VARIABLE_NU_SGS || KNUDSEN_ADAPTIVE_TAU || VARIABLE_SIGMA || VARIABLE_SIGMA_THERMAL
        # Boost variable_nu_sgs floor (for cases with Π_neq-based SGS)
        if VARIABLE_NU_SGS && _variable_nu_sgs_mach_floor_boost && _transonic_bump > Float64(_floor_config)
            global VARIABLE_NU_SGS_FLOOR = Float32(_transonic_bump)
            @printf("[Stability] Mach-adaptive ν_sgs floor: %.4f → %.4f (Ma_lat=%.3f, transonic boost)\n",
                    _floor_config, VARIABLE_NU_SGS_FLOOR, _Ma_lat_stab)
        elseif VARIABLE_NU_SGS && !_variable_nu_sgs_mach_floor_boost && _transonic_bump > Float64(_floor_config)
            @printf("[Stability] Mach-adaptive variable_nu_sgs floor boost disabled: keeping %.4f (Ma_lat=%.3f, requested %.4f)\n",
                    VARIABLE_NU_SGS_FLOOR, _Ma_lat_stab, _transonic_bump)
        end
        # Boost background ν_sgs (for cases without variable_nu_sgs)
        if !_local_stability_active && _transonic_bump > Float64(_nu_sgs_bg_config)
            global NU_SGS_BACKGROUND = Float32(_transonic_bump)
            @printf("[Stability] Mach-adaptive ν_sgs background: %.4f → %.4f (Ma_lat=%.3f, transonic boost)\n",
                    _nu_sgs_bg_config, NU_SGS_BACKGROUND, _Ma_lat_stab)
        elseif _local_stability_active && _transonic_bump > Float64(_nu_sgs_bg_config)
            @printf("[Stability] Mach-adaptive nu_sgs background skipped: local stabilizers active (Ma_lat=%.3f, requested %.4f)\n",
                    _Ma_lat_stab, _transonic_bump)
        end
    end

    # ── Shifted lattice (Phase 3.2) ──
    # Galilean transform: shift D3Q39 velocities by integer vector U = (Ux, Uy, Uz).
    # This is the key to supersonic LBM beyond Ma ≈ 1.6.
    global LATTICE_SHIFT_AUTO = safe_get(CFG, "advanced", "numerics", "shifted_lattice", "auto"; default=false)
    local _shift_x = Int32(safe_get(CFG, "advanced", "numerics", "shifted_lattice", "shift_x"; default=0))
    local _shift_y = Int32(safe_get(CFG, "advanced", "numerics", "shifted_lattice", "shift_y"; default=0))
    local _shift_z = Int32(safe_get(CFG, "advanced", "numerics", "shifted_lattice", "shift_z"; default=0))
    local _local_shift_requested_cfg = Bool(safe_get(CFG, "advanced", "numerics", "local_lattice_shift", "enabled"; default=false))

    # Supersonic-lattice SAFETY NET (2026-06-01): when the LATTICE flow is
    # supersonic (u_lattice ≥ ~1) under compressible D3Q39, the UNSHIFTED
    # equilibrium is wildly inaccurate (the velocity set isn't centered on the
    # freestream) and the run collapses. The Galilean shift U=round(u_lattice)
    # recenters it and is THE make-or-break setting for supersonic — Ma=1.5
    # cylinder runs clean WITH it and collapses WITHOUT it (verified Z_009).
    # So we auto-apply it whenever supersonic, even if the user didn't set
    # `auto: true` or a manual shift — unless the user EXPLICITLY set a manual
    # shift (then respect it). This protects users from forgetting the one
    # setting that makes supersonic work.
    # Threshold 0.5: apply the shift whenever round(u_lattice) ≥ 1, i.e. whenever
    # the nearest-integer shift is nonzero. This covers SUPERSONIC (u_lat>1) AND
    # TRANSONIC (e.g. Ma=0.84 → u_lat≈0.685 rounds to 1). 2026-06-01: transonic
    # was the unsolved frontier — Ma=0.84 collapsed at the SHOULDER where the flow
    # goes LOCALLY supersonic and the unshifted equilibrium is mis-centered. Forcing
    # shift_x=1 makes Ma=0.84 converge to a stable steady state with the shoulder
    # shock captured (verified Z_013: 1200 steps, rho_min steady 0.66). So the shift
    # is the fix for BOTH regimes; the threshold is just "round(u_lat) ≥ 1".
    local _u_lat_needs_shift = COMPRESSIBLE_MODE && LATTICE_TYPE == :d3q39 &&
                               abs(Float64(U_TARGET)) >= 0.5
    local _manual_shift_set = (_shift_x != 0 || _shift_y != 0 || _shift_z != 0)
    local _Ma_for_shift_auto = COMPRESSIBLE_MODE ? (FLOW_VELOCITY / SPEED_OF_SOUND_PHYSICAL) : 0.0
    if _local_shift_requested_cfg && _manual_shift_set
        error("local_lattice_shift.enabled=true uses an unshifted base lattice; do not also set shifted_lattice.shift_x/y/z. " *
              "Use local_lattice_shift.u_max/threshold to control local frames.")
    end
    local _auto_local_shift = (LATTICE_SHIFT_AUTO || _local_shift_requested_cfg) && _u_lat_needs_shift &&
                              !_manual_shift_set && _Ma_for_shift_auto <= 1.6
    local _auto_shift_allowed = LATTICE_SHIFT_AUTO && _u_lat_needs_shift &&
                                !_manual_shift_set && !_auto_local_shift
    if _auto_local_shift
        global LATTICE_SHIFT_X = Int32(0)
        global LATTICE_SHIFT_Y = Int32(0)
        global LATTICE_SHIFT_Z = Int32(0)
        @printf("[Shifted] Auto local shift: unshifted base lattice U=(0,0,0), per-block |U| up to %d (u_lattice=%.3f, Ma=%.3f)\n",
                Int32(max(round(abs(Float64(U_TARGET))), 1)), Float64(U_TARGET), _Ma_for_shift_auto)
    elseif _auto_shift_allowed && COMPRESSIBLE_MODE &&
       LATTICE_TYPE == :d3q39 && !_manual_shift_set
        # Auto-compute: U_x = round(u_lattice), U_y = U_z = 0
        # Centers the equilibrium accuracy around the freestream velocity
        global LATTICE_SHIFT_X = Int32(round(Float64(U_TARGET)))
        global LATTICE_SHIFT_Y = Int32(0)
        global LATTICE_SHIFT_Z = Int32(0)
        if _u_lat_needs_shift && !LATTICE_SHIFT_AUTO
            @printf("[Shifted] AUTO-APPLIED lattice shift: U = (%d, 0, 0) (u_lattice=%.3f, round≥1). Centers the D3Q39 equilibrium for transonic/supersonic; required for stability (the shoulder pocket goes locally supersonic at transonic Ma).\n",
                    LATTICE_SHIFT_X, Float64(U_TARGET))
        else
            @printf("[Shifted] Auto shift: U = (%d, %d, %d)\n", LATTICE_SHIFT_X, LATTICE_SHIFT_Y, LATTICE_SHIFT_Z)
        end
    else
        global LATTICE_SHIFT_X = _shift_x
        global LATTICE_SHIFT_Y = _shift_y
        global LATTICE_SHIFT_Z = _shift_z
    end

    local _max_shift = max(abs(LATTICE_SHIFT_X), abs(LATTICE_SHIFT_Y), abs(LATTICE_SHIFT_Z))
    if _max_shift > 0
        if LATTICE_TYPE != :d3q39
            error("Shifted lattice requires lattice: d3q39.")
        end
        if COLLISION_OPERATOR != :entropic
            error("Shifted lattice requires collision_operator: entropic.")
        end
        # Max |c'| = 3 + max_shift. Must fit within BLOCK_SIZE for single-hop streaming.
        local _max_c_shifted = 3 + _max_shift
        if _max_c_shifted >= BLOCK_SIZE_CONFIG
            error("Shift |U|=$_max_shift too large: max |c'| = $_max_c_shifted >= BLOCK_SIZE=$BLOCK_SIZE_CONFIG. " *
                  "Reduce shift or increase block_size.")
        end
        @printf("[Shifted] Lattice shift: U = (%d, %d, %d), max |c'| = %d\n",
                LATTICE_SHIFT_X, LATTICE_SHIFT_Y, LATTICE_SHIFT_Z, _max_c_shifted)
    end

    # Local shifted-lattice probe (2026-06-01): first increment toward per-block
    # Galilean recentering. This computes and outputs a diagnostic block-wise
    # integer Ux field, but does not yet alter streaming/equilibrium.
    global LOCAL_LATTICE_SHIFT_ENABLED = _local_shift_requested_cfg || _auto_local_shift
    global LOCAL_LATTICE_SHIFT_U_MAX = Int32(safe_get(CFG, "advanced", "numerics", "local_lattice_shift", "u_max"; default=max(Int(round(abs(Float64(U_TARGET)))), 1)))
    global LOCAL_LATTICE_SHIFT_THRESHOLD = Float32(safe_get(CFG, "advanced", "numerics", "local_lattice_shift", "threshold"; default=0.5))
    global LOCAL_LATTICE_SHIFT_HYSTERESIS = Float32(safe_get(CFG, "advanced", "numerics", "local_lattice_shift", "hysteresis"; default=0.1))
    global LOCAL_LATTICE_SHIFT_WALL_GUARD_CELLS = Float32(safe_get(CFG, "advanced", "numerics", "local_lattice_shift", "wall_guard_cells"; default=0.0))
    global LOCAL_LATTICE_SHIFT_WALL_ZERO_GUARD_CELLS = Float32(safe_get(CFG, "advanced", "numerics", "local_lattice_shift", "wall_zero_guard_cells"; default=0.0))
    global LOCAL_LATTICE_SHIFT_ZERO_ALLOWED_X_MIN_CELLS = Float32(safe_get(CFG, "advanced", "numerics", "local_lattice_shift", "zero_allowed_x_min_cells"; default=0.0))
    global LOCAL_LATTICE_SHIFT_WAKE_ZERO_DEFICIT_FRAC = Float32(safe_get(CFG, "advanced", "numerics", "local_lattice_shift", "wake_zero_deficit_frac"; default=0.0))
    global LOCAL_LATTICE_SHIFT_HIGH_SPEED_FRAC = Float32(safe_get(CFG, "advanced", "numerics", "local_lattice_shift", "high_speed_frac"; default=0.60))
    global LOCAL_LATTICE_SHIFT_GEOMETRIC_BAND_CELLS = Float32(safe_get(CFG, "advanced", "numerics", "local_lattice_shift", "geometric_band_cells"; default=0.0))
    if LOCAL_LATTICE_SHIFT_ENABLED
        if LATTICE_TYPE != :d3q39 || !COMPRESSIBLE_MODE
            error("local_lattice_shift requires compressible_mode: true with lattice: d3q39.")
        end
        if LOCAL_LATTICE_SHIFT_U_MAX < 0
            error("local_lattice_shift.u_max must be non-negative.")
        end
        if LOCAL_LATTICE_SHIFT_WALL_GUARD_CELLS < 0.0f0
            error("local_lattice_shift.wall_guard_cells must be non-negative.")
        end
        if LOCAL_LATTICE_SHIFT_WALL_ZERO_GUARD_CELLS < 0.0f0
            error("local_lattice_shift.wall_zero_guard_cells must be non-negative.")
        end
        if LOCAL_LATTICE_SHIFT_ZERO_ALLOWED_X_MIN_CELLS < 0.0f0
            error("local_lattice_shift.zero_allowed_x_min_cells must be non-negative.")
        end
        if LOCAL_LATTICE_SHIFT_WAKE_ZERO_DEFICIT_FRAC < 0.0f0 || LOCAL_LATTICE_SHIFT_WAKE_ZERO_DEFICIT_FRAC > 1.0f0
            error("local_lattice_shift.wake_zero_deficit_frac must be in [0, 1].")
        end
        if LOCAL_LATTICE_SHIFT_HIGH_SPEED_FRAC < 0.0f0 || LOCAL_LATTICE_SHIFT_HIGH_SPEED_FRAC > 1.0f0
            error("local_lattice_shift.high_speed_frac must be in [0, 1].")
        end
        if LOCAL_LATTICE_SHIFT_GEOMETRIC_BAND_CELLS < 0.0f0
            error("local_lattice_shift.geometric_band_cells must be non-negative.")
        end
        if LOCAL_LATTICE_SHIFT_GEOMETRIC_BAND_CELLS > 0.0f0
            @printf("[LocalShift] STATIC GEOMETRIC frame map: U=0 within %.1f cells of the body, U=%d elsewhere (velocity scoring bypassed)\n",
                    LOCAL_LATTICE_SHIFT_GEOMETRIC_BAND_CELLS, LOCAL_LATTICE_SHIFT_U_MAX)
        end
        @printf("[LocalShift] Active D3Q39 local 3D frame enabled: base U=(%d,%d,%d) u_max=%d threshold=%.3f hysteresis=%.3f wall_guard=%.2f wall_zero_guard=%.2f cells zero_x_min=%.1f cells wake_zero_frac=%.3f high_frac=%.3f\n",
                LATTICE_SHIFT_X, LATTICE_SHIFT_Y, LATTICE_SHIFT_Z,
                LOCAL_LATTICE_SHIFT_U_MAX, LOCAL_LATTICE_SHIFT_THRESHOLD,
                LOCAL_LATTICE_SHIFT_HYSTERESIS, LOCAL_LATTICE_SHIFT_WALL_GUARD_CELLS,
                LOCAL_LATTICE_SHIFT_WALL_ZERO_GUARD_CELLS,
                LOCAL_LATTICE_SHIFT_ZERO_ALLOWED_X_MIN_CELLS,
                LOCAL_LATTICE_SHIFT_WAKE_ZERO_DEFICIT_FRAC,
                LOCAL_LATTICE_SHIFT_HIGH_SPEED_FRAC)
    end

    # Validate Ma limit with/without shift
    if COMPRESSIBLE_MODE
        local _Ma_check = FLOW_VELOCITY / SPEED_OF_SOUND_PHYSICAL
        if _Ma_check > 2.0 && _max_shift == 0
            error("Ma_physical = $(_Ma_check) exceeds D3Q39 unshifted limit (~2.0). " *
                  "Enable shifted_lattice.auto: true or set manual shift values.")
        end
    end

    # Double Distribution Function (thermal field)
    # DDF is automatically enabled when:
    #   (a) compressibility_correction is true (D3Q27 Phase 1+2), OR
    #   (b) compressible_mode is true (D3Q39 supersonic — Phase 4 two-way thermal coupling)
    # In compressible mode, the thermal field provides the energy equation needed
    # for correct Rankine-Hugoniot shock relations (p = ρ·cs²·T).
    # Override: set advanced.thermal.enabled: false to disable DDF even in compressible mode
    # (saves ~60% VRAM but loses energy conservation across shocks).
    local _ddf_auto = COMPRESSIBILITY_CORRECTION || COMPRESSIBLE_MODE
    local _ddf_override = Bool(safe_get(CFG, "advanced", "thermal", "enabled"; default=true))
    global DDF_ENABLED = _ddf_auto && _ddf_override
    if DDF_ENABLED && LOCAL_LATTICE_SHIFT_ENABLED && LATTICE_TYPE == :d3q39
        error("[Thermal] D3Q39 Frapolli thermal is not yet compatible with local_lattice_shift: " *
              "the thermal kernel still streams and reconstructs in one global shifted frame. " *
              "Disable advanced.thermal.enabled or disable local_lattice_shift until thermal g-populations are block-frame aware.")
    end
    if _ddf_auto && !_ddf_override
        println("[Thermal] DDF disabled by config (advanced.thermal.enabled: false)")
        println("[Thermal] ⚠ Energy equation inactive — shock Rankine-Hugoniot relations may be inaccurate")
    end

    # SOTA #1: total-energy source terms (pressure-work −p·∇·u + viscous heating Φ).
    # These are NOT optional for compressible flow — they are the energy-equation
    # source terms the passive g-relaxation omits. Shock heating error grows from
    # ~0.7% at Ma=0.84 to ~25% at Ma=2.0, so for the Ma≤2 supersonic target they
    # are required physics. DEFAULT ON whenever the Frapolli energy model is live
    # (DDF + D3Q39). Explicit `advanced.numerics.energy_source_terms: false`
    # disables them (e.g. for A/B comparison or isolating an instability).
    local _frapolli_live = DDF_ENABLED && (LATTICE_TYPE == :d3q39)
    global ENERGY_SOURCE_TERMS = Bool(safe_get(CFG, "advanced", "numerics",
                                       "energy_source_terms"; default=_frapolli_live))
    if ENERGY_SOURCE_TERMS && !_frapolli_live
        println("[Thermal] ⚠ energy_source_terms requested but Frapolli model inactive (needs DDF + D3Q39) — ignored at runtime")
    elseif ENERGY_SOURCE_TERMS
        println("[Thermal] Energy source terms ENABLED (pressure-work + viscous heating) — required for Ma≳1.2 shock heating")
    elseif _frapolli_live
        println("[Thermal] ⚠ Energy source terms DISABLED by config — shock heating will be under-predicted at high Ma")
    end

    # SOTA #3: Ducros shock-sensor dissipation. Default ON for compressible
    # D3Q39 (needed for stable Ma≳1.6); explicit enabled:false disables.
    local _shock_default = COMPRESSIBLE_MODE && (LATTICE_TYPE == :d3q39)
    global SHOCK_SENSOR_ENABLED = Bool(safe_get(CFG, "advanced", "numerics",
                                        "shock_sensor", "enabled"; default=_shock_default))
    global SHOCK_SENSOR_NU_MAX = Float64(safe_get(CFG, "advanced", "numerics",
                                        "shock_sensor", "nu_max"; default=0.02))
    global SHOCK_SENSOR_THRESH_LO = Float64(safe_get(CFG, "advanced", "numerics",
                                        "shock_sensor", "thresh_lo"; default=0.15))
    global SHOCK_SENSOR_THRESH_HI = Float64(safe_get(CFG, "advanced", "numerics",
                                        "shock_sensor", "thresh_hi"; default=0.30))
    if SHOCK_SENSOR_ENABLED && (LATTICE_TYPE == :d3q39)
        println("[Stability] Ducros shock-sensor dissipation ENABLED (nu_max=$(SHOCK_SENSOR_NU_MAX), band=[$(SHOCK_SENSOR_THRESH_LO),$(SHOCK_SENSOR_THRESH_HI)])")
    end

    # Level-independent lattice-ν floor (kills L1 streamwise checkerboard).
    # Default-on for compressible D3Q39 at a light value; explicit setting wins.
    local _latfloor_default = (COMPRESSIBLE_MODE && LATTICE_TYPE == :d3q39) ? 0.001 : 0.0
    global NU_SGS_LATTICE_FLOOR = Float64(safe_get(CFG, "advanced", "numerics",
                                        "variable_nu_sgs", "lattice_floor"; default=_latfloor_default))
    if NU_SGS_LATTICE_FLOOR > 0.0
        println("[Stability] Level-independent lattice-ν floor = $(NU_SGS_LATTICE_FLOOR) (all levels; suppresses L1 lattice modes)")
    end
    if DDF_ENABLED
        global DDF_PRANDTL = Float64(safe_get(CFG, "advanced", "thermal", "prandtl"; default=0.71))
        global DDF_T_INLET = Float64(safe_get(CFG, "advanced", "thermal", "t_inlet"; default=1.0))
        global DDF_T_WALL = Float64(safe_get(CFG, "advanced", "thermal", "t_wall"; default=1.0))
        global DDF_T_INITIAL = Float64(safe_get(CFG, "advanced", "thermal", "t_initial"; default=1.0))
        local ddf_wall_bc_str = safe_get(CFG, "advanced", "thermal", "wall_bc"; default="adiabatic")
        global DDF_WALL_BC = Symbol(ddf_wall_bc_str)
        if !(DDF_WALL_BC in Set([:adiabatic, :isothermal]))
            error("Unknown thermal wall_bc: \"$DDF_WALL_BC\". Valid: adiabatic, isothermal")
        end
        global DDF_TURBULENT_PRANDTL = Float64(safe_get(CFG, "advanced", "thermal", "turbulent_prandtl"; default=0.85))
        # Ratio of specific heats γ = Cp/Cv. For diatomic air, γ = 1.4.
        global DDF_GAMMA = Float64(safe_get(CFG, "advanced", "thermal", "gamma"; default=1.4))
        # C_v: specific heat at constant volume (lattice units). 5/2 for diatomic air.
        # Controls Frapolli g_eq coefficient (Cv - D/2) and temperature recovery.
        global DDF_CV = Float64(safe_get(CFG, "advanced", "thermal", "cv"; default=2.5))
        global DDF_FEEDBACK_TO_FLOW = Bool(safe_get(CFG, "advanced", "thermal", "feedback_to_flow"; default=true))
        global DDF_FEEDBACK_RELAXATION = clamp(Float32(safe_get(CFG, "advanced", "thermal", "feedback_relaxation"; default=1.0)), 0.0f0, 1.0f0)
        global DDF_SHOCK_TUBE_COUPLING = Bool(safe_get(CFG, "advanced", "thermal", "shock_tube_coupling"; default=false))
        global DDF_T_ADMISSIBLE_MAX = Float32(safe_get(CFG, "advanced", "thermal", "t_admissible_max"; default=0.0))
        # Phase 4b: When D3Q39 flow + DDF enabled, thermal g uses D3Q39 (Frapolli model).
        # This gives correct compressible energy equation without explicit source terms.
        global N_DIRS_THERMAL = (LATTICE_TYPE == :d3q39) ? N_DIRS : Int32(27)
        if LATTICE_TYPE == :d3q39
            @printf("[Thermal] Frapolli D3Q39 model: g uses %d directions (same as flow)\n", N_DIRS_THERMAL)
            cs2_th = (LATTICE_TYPE == :d3q39) ? 2.0/3.0 : 1.0/3.0
            # g_eq coefficient is Cv − 1.5·R, where R is the EOS pressure coeff.
            # Linearized EOS: R = cs²_thermal. Thermodynamic EOS (eos_thermodynamic
            # on): R = (γ−1)·Cv. The collision/thermal/AMR kernels select R via the
            # eos_R_flow sentinel; show both so the banner is not misleading.
            R_lin = cs2_th
            R_thermo = (DDF_GAMMA - 1.0) * DDF_CV
            @printf("[Thermal] C_v = %.2f, γ = %.2f, g_eq coeff (Cv-1.5·R): linearized R=cs²=%.3f → %.3f ; thermodynamic R=(γ-1)·Cv=%.3f → %.3f\n",
                    DDF_CV, DDF_GAMMA, R_lin, DDF_CV - 1.5 * R_lin, R_thermo, DDF_CV - 1.5 * R_thermo)
        end
    else
        global DDF_PRANDTL = 0.71
        global DDF_T_INLET = 1.0
        global DDF_T_WALL = 1.0
        global DDF_T_INITIAL = 1.0
        global DDF_WALL_BC = :adiabatic
        global DDF_TURBULENT_PRANDTL = 0.85
        global DDF_GAMMA = 1.4
        global DDF_CV = 2.5
        global N_DIRS_THERMAL = Int32(27)
        global DDF_FEEDBACK_TO_FLOW = true
        global DDF_FEEDBACK_RELAXATION = 1.0f0
        global DDF_SHOCK_TUBE_COUPLING = false
        global DDF_T_ADMISSIBLE_MAX = 0.0f0
    end

    # Thermal T-filter (post-recovery 6-neighbor laplacian). Only meaningful when DDF is active.
    global THERMAL_FILTER_ENABLED = DDF_ENABLED && Bool(safe_get(CFG, "advanced", "numerics", "thermal_filter", "enabled"; default=false))
    global THERMAL_FILTER_ALPHA = clamp(Float32(safe_get(CFG, "advanced", "numerics", "thermal_filter", "alpha"; default=0.05)), 0.0f0, 0.5f0)
    if THERMAL_FILTER_ENABLED
        @printf("[Thermal] T-filter (post-recovery laplacian) ENABLED, α=%.3f\n", THERMAL_FILTER_ALPHA)
    end

    global AUTO_LEVELS = safe_get(CFG, "advanced", "high_re", "auto_levels"; default=false)
    global MAX_LEVELS = Int(safe_get(CFG, "advanced", "high_re", "max_levels"; default=12))
    global MIN_COARSE_BLOCKS = Int(safe_get(CFG, "advanced", "high_re", "min_coarse_blocks"; default=4))
    global WALL_MODEL_ENABLED = safe_get(CFG, "advanced", "high_re", "wall_model", "enabled"; default=false)
    global WALL_MODEL_TYPE = Symbol(safe_get(CFG, "advanced", "high_re", "wall_model", "type"; default="equilibrium"))
    global WALL_MODEL_YPLUS_TARGET = Float64(safe_get(CFG, "advanced", "high_re", "wall_model", "y_plus_target"; default=30.0))
    global WALL_MODEL_MAX_DISTANCE_CELLS = Float32(safe_get(CFG, "advanced", "high_re", "wall_model", "max_distance_cells"; default=10.0))
    WALL_MODEL_MAX_DISTANCE_CELLS > 0.0f0 ||
        error("advanced.high_re.wall_model.max_distance_cells must be > 0")
    # SOTA B6: compressible wall model (Van Driest)
    global WALL_MODEL_COMPRESSIBLE = safe_get(CFG, "advanced", "high_re",
                                              "wall_model", "compressible_correction"; default=false)
    # SOTA B6-ext: Larsson 2010 NEQ-WM pressure-gradient correction (requires Van Driest)
    global WALL_MODEL_LARSSON_PGR = safe_get(CFG, "advanced", "high_re",
                                             "wall_model", "larsson_pgr"; default=false)
    if WALL_MODEL_LARSSON_PGR && !WALL_MODEL_COMPRESSIBLE
        @warn "wall_model.larsson_pgr enabled but compressible_correction is false; on the D3Q39 path the PG correction is silently inactive on the Van-Driest branch."
    end


    global DOMAIN_UPSTREAM = Float64(safe_get(CFG, "advanced", "domain", "upstream"; default=0.75))
    global DOMAIN_DOWNSTREAM = Float64(safe_get(CFG, "advanced", "domain", "downstream"; default=1.5))
    global DOMAIN_LATERAL = Float64(safe_get(CFG, "advanced", "domain", "lateral"; default=0.75))
    global DOMAIN_HEIGHT = Float64(safe_get(CFG, "advanced", "domain", "height"; default=0.75))
    # Optional active-domain span override in the solver Y direction. This is
    # useful for quasi-2D cylinders: generate a closed, longer STL for robust
    # ray tests, but keep its end caps outside the computational span.
    global DOMAIN_SPAN_OVERRIDE = Float64(safe_get(CFG, "advanced", "domain", "span_override"; default=-1.0))
    if DOMAIN_SPAN_OVERRIDE == -1.0
        global DOMAIN_SPAN_OVERRIDE = Float64(safe_get(CFG, "advanced", "domain", "y_size_override"; default=-1.0))
    end
    if DOMAIN_SPAN_OVERRIDE == 0.0 || DOMAIN_SPAN_OVERRIDE < -1.0
        error("advanced.domain.span_override must be positive, or omitted/negative to disable.")
    end
    # ── NEW chord-relative domain knobs (v1.11) ──
    # When any of these is > 0, the auto-sizer uses it as the *initial* extent
    # in units of the obstacle's streamwise AABB (chord proxy) and may then
    # shrink or grow the value during the VRAM fit loop. The corresponding
    # legacy key (advanced.domain.upstream, etc.) is used as an upper cap on
    # the final extent so existing cases stay bounded. A negative value disables
    # auto-sizing for that axis.
    global DOMAIN_UPSTREAM_CHORDS   = Float64(safe_get(CFG, "advanced", "domain", "upstream_chords";   default=-1.0))
    global DOMAIN_DOWNSTREAM_CHORDS = Float64(safe_get(CFG, "advanced", "domain", "downstream_chords"; default=-1.0))
    global DOMAIN_LATERAL_CHORDS    = Float64(safe_get(CFG, "advanced", "domain", "lateral_chords";    default=-1.0))
    global DOMAIN_HEIGHT_CHORDS     = Float64(safe_get(CFG, "advanced", "domain", "height_chords";     default=-1.0))
    global DOMAIN_BODY_HALO_CHORDS  = Float64(safe_get(CFG, "advanced", "domain", "body_halo_chords";  default=0.5))
    local raw_object_center_x_fraction = Float64(safe_get(CFG, "advanced", "domain", "object_center_x_fraction"; default=-1.0))
    if raw_object_center_x_fraction >= 0.0
        0.0 < raw_object_center_x_fraction < 1.0 || error("advanced.domain.object_center_x_fraction must be between 0 and 1 (exclusive), got $raw_object_center_x_fraction")
        global DOMAIN_OBJECT_CENTER_X_FRACTION = raw_object_center_x_fraction
    else
        global DOMAIN_OBJECT_CENTER_X_FRACTION = nothing
    end
    global DOMAIN_OBJECT_CENTER_X_M = parse_optional_object_center_m(CFG, "x")
    global DOMAIN_OBJECT_CENTER_Y_M = parse_optional_object_center_m(CFG, "y")
    global DOMAIN_OBJECT_CENTER_Z_M = parse_optional_object_center_m(CFG, "z")
    DOMAIN_OBJECT_CENTER_X_M !== nothing && DOMAIN_OBJECT_CENTER_X_FRACTION !== nothing &&
        error("advanced.domain.object_center_x_m and object_center_x_fraction cannot both be set")
    DOMAIN_OBJECT_CENTER_Y_M !== nothing && SYMMETRIC_ANALYSIS &&
        error("advanced.domain.object_center_y_m cannot move a body off the y=0 symmetry plane; omit it")
    # Sponge defaults tuned for multi-level AMR:
    #  - tanh with σ=0.25 is smoother at both edges than a half-cosine and
    #    avoids the hard color step that shows up at the outlet in density
    #    slices when cosine lives on L1 (156 mm) coarse cells.
    #  - 0.15·L_ref thickness (up from 0.10) gives the wake ~50 % more
    #    distance to decay before hitting the hard BC, reducing the density
    #    jump at the sponge entry plane.
    global SPONGE_THICKNESS = Float32(safe_get(CFG, "advanced", "domain", "sponge_thickness"; default=0.15))
    global SPONGE_PROFILE = Symbol(lowercase(String(safe_get(CFG, "advanced", "domain", "sponge_profile"; default="tanh"))))
    global SPONGE_TANH_SIGMA = Float32(safe_get(CFG, "advanced", "domain", "sponge_tanh_sigma"; default=0.25))
    global SPONGE_OUTLET_STRENGTH = Float32(safe_get(CFG, "advanced", "domain", "sponge_outlet_strength"; default=1.0))
    global SPONGE_INLET_STRENGTH = Float32(safe_get(CFG, "advanced", "domain", "sponge_inlet_strength"; default=0.05))
    global SPONGE_TRANSVERSE_STRENGTH = Float32(safe_get(CFG, "advanced", "domain", "sponge_transverse_strength"; default=0.10))
    
    global BLOCK_SIZE_CONFIG = Int(safe_get(CFG, "advanced", "refinement", "block_size"; default=8))
    global REFINEMENT_MARGIN = Int(safe_get(CFG, "advanced", "refinement", "margin"; default=1))  # 2026-05-30: tightened 2→1 (see global default note)
    global REFINEMENT_MARGIN_UPSTREAM = Int(safe_get(CFG, "advanced", "refinement", "margin_upstream"; default=REFINEMENT_MARGIN))
    global REFINEMENT_MARGIN_DOWNSTREAM = Int(safe_get(CFG, "advanced", "refinement", "margin_downstream"; default=max(0, REFINEMENT_MARGIN - 1)))
    global REFINEMENT_MARGIN_SPAN = Int(safe_get(CFG, "advanced", "refinement", "margin_span"; default=REFINEMENT_MARGIN))
    global REFINEMENT_MARGIN_UP = Int(safe_get(CFG, "advanced", "refinement", "margin_up"; default=REFINEMENT_MARGIN))
    global REFINEMENT_MARGIN_DOWN = Int(safe_get(CFG, "advanced", "refinement", "margin_down"; default=max(0, REFINEMENT_MARGIN - 1)))
    # Distance-band mode (per-level physical-thickness halo). Bands in meters.
    global REFINEMENT_BAND_MODE = Symbol(lowercase(String(safe_get(CFG, "advanced", "refinement", "band_mode"; default="blocks"))))
    global REFINEMENT_BAND_UPSTREAM   = Float64(safe_get(CFG, "advanced", "refinement", "band_upstream";   default=0.0))
    global REFINEMENT_BAND_DOWNSTREAM = Float64(safe_get(CFG, "advanced", "refinement", "band_downstream"; default=0.0))
    global REFINEMENT_BAND_SPAN       = Float64(safe_get(CFG, "advanced", "refinement", "band_span";       default=0.0))
    global REFINEMENT_BAND_UP         = Float64(safe_get(CFG, "advanced", "refinement", "band_up";         default=0.0))
    global REFINEMENT_BAND_DOWN       = Float64(safe_get(CFG, "advanced", "refinement", "band_down";       default=0.0))
    REFINEMENT_BAND_MODE == :distance && println("[Init]    Refinement band mode: DISTANCE (up=$(REFINEMENT_BAND_UPSTREAM) down=$(REFINEMENT_BAND_DOWNSTREAM) span=$(REFINEMENT_BAND_SPAN) up=$(REFINEMENT_BAND_UP) dn=$(REFINEMENT_BAND_DOWN) m) — coarse levels get thinner shells")
    REFINEMENT_MARGIN_UPSTREAM < 0 && error("advanced.refinement.margin_upstream must be >= 0")
    REFINEMENT_MARGIN_DOWNSTREAM < 0 && error("advanced.refinement.margin_downstream must be >= 0")
    REFINEMENT_MARGIN_SPAN < 0 && error("advanced.refinement.margin_span must be >= 0")
    REFINEMENT_MARGIN_UP < 0 && error("advanced.refinement.margin_up must be >= 0")
    REFINEMENT_MARGIN_DOWN < 0 && error("advanced.refinement.margin_down must be >= 0")
    # AMR feather band — extra viscous dissipation in cells adjacent to
    # each level's boundary, to damp the O(1e-4) density seam injected
    # by the prolongation-streaming mix before it propagates inward.
    # feather_depth: band thickness in cells (0 disables).
    # feather_nu_max: max extra ν added at the boundary face (lattice units).
    # feather_x_only: restrict the band to streamwise ±x AMR faces.
    global FEATHER_DEPTH = Int(safe_get(CFG, "advanced", "refinement", "feather_depth"; default=0))
    global FEATHER_NU_MAX = Float32(safe_get(CFG, "advanced", "refinement", "feather_nu_max"; default=0.005))
    global FEATHER_X_ONLY = safe_get(CFG, "advanced", "refinement", "feather_x_only"; default=false)
    global COARSE_SIDE_FEATHER_SCALE = Float32(safe_get(CFG, "advanced", "refinement", "coarse_side_feather_scale"; default=0.0))
    # Depth for AMR-interface h-mode cleanup in the collision operator. Unlike
    # feather_depth, this does not add viscosity by itself; it only computes the
    # interface weight used by the HRR/KBC sigma branch when parent/fine states
    # measurably disagree.
    global AMR_REGULARIZATION_DEPTH = Int(safe_get(CFG, "advanced", "refinement", "interface_regularization_depth"; default=0))
    # These two switches isolate the core AMR restriction/prolongation pair
    # from solver-specific shell/sponge patches during interface debugging.
    global AMR_INTERFACE_SHELL_SYNC = safe_get(CFG, "advanced", "refinement", "interface_shell_sync"; default=false)
    # Equilibrium-shift to restore Σf = ρ after shell-sync macro updates.
    # the parent macros without rebuilding populations. Without this shift the
    # parent shell exposes an inconsistent (ρ, u, T) vs Σf state to prolongation,
    # which prints a 1-cell streamwise band at every AMR boundary (Pe≈80 along x
    # stretches the O(1e-4) ρ inconsistency into a visible seam). Default on; can
    # be disabled to A/B test the seam contribution from this specific path.
    global AMR_SHELL_EQ_SHIFT = safe_get(CFG, "advanced", "refinement", "interface_shell_eq_shift"; default=true)
    global AMR_SHELL_WELL_BALANCED_REBUILD = safe_get(CFG, "advanced", "refinement", "interface_shell_well_balanced_rebuild"; default=false)
    global AMR_SHELL_MOMENT_SHIFT = safe_get(CFG, "advanced", "refinement", "interface_shell_moment_shift"; default=false)
    global AMR_SHELL_SYNC_INCLUDE_X = safe_get(CFG, "advanced", "refinement", "interface_shell_sync_include_x"; default=false)
    global AMR_SHELL_POPULATION_BLEND_X = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_population_blend_x"; default=1.0))
    global AMR_SHELL_RHO_ALPHA_X_SINGLE = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_rho_alpha_x_single"; default=0.04))
    global AMR_SHELL_RHO_ALPHA_X_MULTI = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_rho_alpha_x_multi"; default=0.02))
    global AMR_SHELL_RHO_ALPHA_T_SINGLE = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_rho_alpha_t_single"; default=0.70))
    global AMR_SHELL_RHO_ALPHA_T_MULTI = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_rho_alpha_t_multi"; default=0.28))
    global AMR_SHELL_VEL_ALPHA_X_SINGLE = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_vel_alpha_x_single"; default=0.05))
    global AMR_SHELL_VEL_ALPHA_X_MULTI = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_vel_alpha_x_multi"; default=0.035))
    global AMR_SHELL_VEL_ALPHA_T = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_vel_alpha_t"; default=0.12))
    global AMR_SHELL_VEL_NORMAL_ONLY = safe_get(CFG, "advanced", "refinement", "interface_shell_vel_normal_only"; default=false)
    global AMR_SHELL_DRHO_CAP_X_ABS = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_drho_cap_x_abs"; default=5.0e-6))
    global AMR_SHELL_DRHO_CAP_X_REL = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_drho_cap_x_rel"; default=3.0e-5))
    global AMR_SHELL_DRHO_CAP_T_ABS = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_drho_cap_t_abs"; default=2.0e-5))
    global AMR_SHELL_DRHO_CAP_T_REL = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_drho_cap_t_rel"; default=1.5e-4))
    global AMR_SHELL_DU_CAP_X_ABS = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_du_cap_x_abs"; default=1.0e-5))
    global AMR_SHELL_DU_CAP_X_REL = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_du_cap_x_rel"; default=5.0e-3))
    global AMR_SHELL_DU_CAP_T_ABS = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_du_cap_t_abs"; default=2.0e-5))
    global AMR_SHELL_DU_CAP_T_REL = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_du_cap_t_rel"; default=2.0e-2))
    global AMR_SHELL_FLUX_ALPHA_X = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_flux_alpha_x"; default=0.0))
    global AMR_SHELL_FLUX_ALPHA_T = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_flux_alpha_t"; default=0.0))
    global AMR_SHELL_FLUX_DU_CAP_ABS = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_flux_du_cap_abs"; default=2.0e-5))
    global AMR_SHELL_FLUX_DU_CAP_REL = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_flux_du_cap_rel"; default=1.0e-2))
    global AMR_SHELL_MOMFLUX_ALPHA_X = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_momflux_alpha_x"; default=0.0))
    global AMR_SHELL_MOMFLUX_DRHO_CAP_ABS = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_momflux_drho_cap_abs"; default=5.0e-6))
    global AMR_SHELL_MOMFLUX_DRHO_CAP_REL = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_momflux_drho_cap_rel"; default=3.0e-5))
    global AMR_SHELL_MOMFLUX_STRESS_ALPHA_X = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_momflux_stress_alpha_x"; default=0.0))
    global AMR_SHELL_MOMFLUX_STRESS_CAP_ABS = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_momflux_stress_cap_abs"; default=1.0e-5))
    global AMR_SHELL_MOMFLUX_STRESS_CAP_REL = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_momflux_stress_cap_rel"; default=3.0e-5))
    global AMR_SHELL_FACE_WEIGHT_X = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_face_weight_x"; default=0.30))
    global AMR_SHELL_FACE_POP_SHIFT = safe_get(CFG, "advanced", "refinement", "interface_shell_face_population_shift"; default=false)
    global AMR_SHELL_FACE_POP_ALPHA = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_face_population_alpha"; default=1.0))
    global AMR_SHELL_FACE_POP_SIDE = Int(safe_get(CFG, "advanced", "refinement", "interface_shell_face_population_side"; default=1))
    (AMR_SHELL_FACE_POP_SIDE == 1 || AMR_SHELL_FACE_POP_SIDE == -1) ||
        error("advanced.refinement.interface_shell_face_population_side must be 1 (toward fine) or -1 (from fine)")
    global AMR_SHELL_FACE_STRESS_SHIFT = safe_get(CFG, "advanced", "refinement", "interface_shell_face_stress_shift"; default=false)
    global AMR_SHELL_FACE_STRESS_ALPHA = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_face_stress_alpha"; default=0.5))
    global AMR_SHELL_FACE_STRESS_REL_CAP = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_face_stress_rel_cap"; default=0.6))
    global AMR_SHELL_FACE_STRESS_ABS_CAP = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_face_stress_abs_cap"; default=2.0e-5))
    global AMR_SHELL_COUPLED_REFLUX = safe_get(CFG, "advanced", "refinement", "interface_shell_coupled_reflux"; default=false)
    global AMR_SHELL_COUPLED_REFLUX_ALPHA = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_coupled_reflux_alpha"; default=0.20))
    global AMR_SHELL_PAIRED_REFLUX = safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux"; default=false)
    global AMR_SHELL_PAIRED_REFLUX_ALPHA = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_alpha"; default=0.20))
    global AMR_SHELL_PAIRED_REFLUX_MASS_ALPHA = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_mass_alpha"; default=1.0))
    global AMR_SHELL_PAIRED_REFLUX_MOM_ALPHA = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_mom_alpha"; default=1.0))
    global AMR_SHELL_PAIRED_REFLUX_MOM_ALPHA_INITIAL = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_mom_alpha_initial"; default=AMR_SHELL_PAIRED_REFLUX_MOM_ALPHA))
    global AMR_SHELL_PAIRED_REFLUX_MOM_ALPHA_HOLD_STEPS = Int(safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_mom_alpha_hold_steps"; default=0))
    global AMR_SHELL_PAIRED_REFLUX_PARENT_FRACTION = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_parent_fraction"; default=1.0 / 3.0))
    global AMR_SHELL_PAIRED_REFLUX_MASS_PARENT_FRACTION = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_mass_parent_fraction"; default=AMR_SHELL_PAIRED_REFLUX_PARENT_FRACTION))
    global AMR_SHELL_PAIRED_REFLUX_MASS_CAP_SCALE = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_mass_cap_scale"; default=1.0))
    global AMR_SHELL_PAIRED_REFLUX_MASS_CAP_INITIAL = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_mass_cap_initial"; default=AMR_SHELL_PAIRED_REFLUX_MASS_CAP_SCALE))
    global AMR_SHELL_PAIRED_REFLUX_MASS_CAP_HOLD_STEPS = Int(safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_mass_cap_hold_steps"; default=0))
    global AMR_SHELL_PAIRED_REFLUX_FACE_CONSERVATIVE = safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_face_conservative"; default=false)
    global AMR_SHELL_PAIRED_REFLUX_COUPLED_CAPS = safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_coupled_caps"; default=false)
    global AMR_SHELL_PAIRED_REFLUX_COUPLED_MASS_CAPS = safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_coupled_mass_caps"; default=false)
    global AMR_SHELL_PAIRED_REFLUX_MACRO_CONSISTENT_CAPS = safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_macro_consistent_caps"; default=false)
    global AMR_SHELL_PAIRED_REFLUX_MACRO_PAIRWISE_CAPS = safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_macro_pairwise_caps"; default=false)
    global AMR_SHELL_PAIRED_REFLUX_BUDGET_MEAN_PRESERVING = safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_budget_mean_preserving"; default=false)
    global AMR_SHELL_PAIRED_REFLUX_SHARED_STATE = safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_shared_state"; default=false)
    global AMR_SHELL_PAIRED_REFLUX_SHARED_STATE_RHO_ALPHA = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_shared_state_rho_alpha"; default=1.0))
    global AMR_SHELL_PAIRED_REFLUX_HIGHPASS = safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_highpass"; default=false)
    global AMR_SHELL_PAIRED_REFLUX_HIGHPASS_BLEND = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_highpass_blend"; default=1.0))
    global AMR_SHELL_PAIRED_REFLUX_LOWPASS = safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_lowpass"; default=false)
    global AMR_SHELL_PAIRED_REFLUX_LOWPASS_BLEND = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_lowpass_blend"; default=0.0))
    global AMR_SHELL_PAIRED_REFLUX_SECOND_MOMENT = safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_second_moment"; default=false)
    global AMR_SHELL_PAIRED_REFLUX_SECOND_MOMENT_BLEND = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_second_moment_blend"; default=1.0))
    global AMR_SHELL_PAIRED_REFLUX_POST_STRESS = safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_post_stress"; default=false)
    global AMR_SHELL_PAIRED_REFLUX_POST_STRESS_ALPHA = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_post_stress_alpha"; default=0.35))
    global AMR_SHELL_PAIRED_REFLUX_POST_STRESS_SOFT_LIMIT = safe_get(CFG, "advanced", "refinement", "interface_shell_paired_reflux_post_stress_soft_limit"; default=false)
    global AMR_SHELL_HIGHPASS_DAMPER = safe_get(CFG, "advanced", "refinement", "interface_shell_highpass_damper"; default=false)
    global AMR_SHELL_HIGHPASS_DAMPER_ALPHA = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_highpass_damper_alpha"; default=0.0))
    global AMR_SHELL_HIGHPASS_DAMPER_CAP_ABS = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_highpass_damper_cap_abs"; default=0.0))
    global AMR_SHELL_HIGHPASS_DAMPER_CAP_REL = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_highpass_damper_cap_rel"; default=0.0))
    global AMR_SHELL_HIGHPASS_DAMPER_DEPTH = Int(safe_get(CFG, "advanced", "refinement", "interface_shell_highpass_damper_depth"; default=1))
    global AMR_SHELL_HIGHPASS_DAMPER_TRANSVERSE_ONLY = safe_get(CFG, "advanced", "refinement", "interface_shell_highpass_damper_transverse_only"; default=true)
    global AMR_SHELL_HIGHPASS_DAMPER_RHO_ALPHA = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_highpass_damper_rho_alpha"; default=0.0))
    global AMR_SHELL_HIGHPASS_DAMPER_DRHO_CAP_ABS = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_highpass_damper_drho_cap_abs"; default=0.0))
    global AMR_SHELL_HIGHPASS_DAMPER_DRHO_CAP_REL = Float32(safe_get(CFG, "advanced", "refinement", "interface_shell_highpass_damper_drho_cap_rel"; default=0.0))
    global AMR_HALO_FULLCELL_RECOMPOSITION = safe_get(CFG, "advanced", "refinement", "amr_halo_fullcell_recomposition"; default=false)
    global AMR_HALO_FULLCELL_MACRO_ALPHA = Float32(safe_get(CFG, "advanced", "refinement", "amr_halo_fullcell_macro_alpha"; default=1.0))
    global AMR_HALO_FULLCELL_POP_ALPHA = Float32(safe_get(CFG, "advanced", "refinement", "amr_halo_fullcell_pop_alpha"; default=0.95))
    global AMR_PROLONGATION_MONOTONE_VELOCITY = safe_get(CFG, "advanced", "refinement", "amr_prolongation_monotone_velocity"; default=false)
    global AMR_HALO_TEMPORAL_MACRO_CLOSURE = safe_get(CFG, "advanced", "refinement", "amr_halo_temporal_macro_closure"; default=false)
    global AMR_HALO_EQ_PROJECTION = safe_get(CFG, "advanced", "refinement", "amr_halo_eq_projection"; default=true)
    global AMR_HALO_CONSTRAINED_COMPLETION = safe_get(CFG, "advanced", "refinement", "amr_halo_constrained_completion"; default=false)
    global AMR_RUNTIME_FACE_BUDGET_REFLUX = safe_get(CFG, "advanced", "refinement", "runtime_face_budget_reflux"; default=false)
    global AMR_RUNTIME_FACE_BUDGET_REFLUX_ALPHA = Float32(safe_get(CFG, "advanced", "refinement", "runtime_face_budget_reflux_alpha"; default=0.10))
    global AMR_RUNTIME_FACE_BUDGET_REFLUX_MAX_PARENT_LEVEL = Int(safe_get(CFG, "advanced", "refinement", "runtime_face_budget_reflux_max_parent_level"; default=1))
    global AMR_RUNTIME_FACE_BUDGET_REFLUX_X_ONLY = safe_get(CFG, "advanced", "refinement", "runtime_face_budget_reflux_x_only"; default=true)
    global AMR_GPU_FACE_BUDGET_REFLUX = safe_get(CFG, "advanced", "refinement", "gpu_face_budget_reflux"; default=false)
    global AMR_GPU_FACE_BUDGET_REFLUX_ALPHA = Float32(safe_get(CFG, "advanced", "refinement", "gpu_face_budget_reflux_alpha"; default=0.10))
    global AMR_GPU_FACE_BUDGET_REFLUX_MASS_ALPHA = Float32(safe_get(CFG, "advanced", "refinement", "gpu_face_budget_reflux_mass_alpha"; default=1.0))
    global AMR_GPU_FACE_BUDGET_REFLUX_MOM_ALPHA = Float32(safe_get(CFG, "advanced", "refinement", "gpu_face_budget_reflux_mom_alpha"; default=1.0))
    global AMR_GPU_FACE_BUDGET_REFLUX_DRHO_CAP_ABS = Float32(safe_get(CFG, "advanced", "refinement", "gpu_face_budget_reflux_drho_cap_abs"; default=5.0e-6))
    global AMR_GPU_FACE_BUDGET_REFLUX_DRHO_CAP_REL = Float32(safe_get(CFG, "advanced", "refinement", "gpu_face_budget_reflux_drho_cap_rel"; default=3.0e-5))
    global AMR_GPU_FACE_BUDGET_REFLUX_MOM_CAP_ABS = Float32(safe_get(CFG, "advanced", "refinement", "gpu_face_budget_reflux_mom_cap_abs"; default=1.0e-5))
    global AMR_GPU_FACE_BUDGET_REFLUX_MOM_CAP_REL = Float32(safe_get(CFG, "advanced", "refinement", "gpu_face_budget_reflux_mom_cap_rel"; default=3.0e-5))
    global AMR_GPU_FACE_BUDGET_REFLUX_NEUTRAL_MOM = safe_get(CFG, "advanced", "refinement", "gpu_face_budget_reflux_neutral_mom"; default=false)
    global AMR_GPU_FACE_BUDGET_REFLUX_MAX_PARENT_LEVEL = Int(safe_get(CFG, "advanced", "refinement", "gpu_face_budget_reflux_max_parent_level"; default=1))
    global AMR_POPULATION_STREAM_REGISTER_APPLY = safe_get(CFG, "advanced", "refinement", "population_stream_register_apply"; default=false)
    global AMR_POPULATION_STREAM_REGISTER_APPLY_ALPHA = Float32(safe_get(CFG, "advanced", "refinement", "population_stream_register_apply_alpha"; default=1.0))
    global AMR_POPULATION_STREAM_REGISTER_APPLY_MAX_PARENT_LEVEL = Int(safe_get(CFG, "advanced", "refinement", "population_stream_register_apply_max_parent_level"; default=2))
    global AMR_POPULATION_STREAM_REGISTER_FACE_APPLY = safe_get(CFG, "advanced", "refinement", "population_stream_register_face_apply"; default=false)
    global AMR_POPULATION_STREAM_REGISTER_FACE_APPLY_ALPHA = Float32(safe_get(CFG, "advanced", "refinement", "population_stream_register_face_apply_alpha"; default=1.0))
    global AMR_POPULATION_STREAM_REGISTER_FACE_APPLY_MAX_PARENT_LEVEL = Int(safe_get(CFG, "advanced", "refinement", "population_stream_register_face_apply_max_parent_level"; default=2))
    global AMR_POPULATION_STREAM_REGISTER_XFACE_HIGHPASS_APPLY = safe_get(CFG, "advanced", "refinement", "population_stream_register_xface_highpass_apply"; default=false)
    global AMR_POPULATION_STREAM_REGISTER_XFACE_HIGHPASS_APPLY_ALPHA = Float32(safe_get(CFG, "advanced", "refinement", "population_stream_register_xface_highpass_apply_alpha"; default=0.25))
    global AMR_POPULATION_STREAM_REGISTER_XFACE_HIGHPASS_APPLY_MAX_PARENT_LEVEL = Int(safe_get(CFG, "advanced", "refinement", "population_stream_register_xface_highpass_apply_max_parent_level"; default=2))
    global AMR_POPULATION_STREAM_REGISTER_XFACE_GLOBAL_APPLY = safe_get(CFG, "advanced", "refinement", "population_stream_register_xface_global_apply"; default=false)
    global AMR_POPULATION_STREAM_REGISTER_XFACE_GLOBAL_APPLY_ALPHA = Float32(safe_get(CFG, "advanced", "refinement", "population_stream_register_xface_global_apply_alpha"; default=1.0))
    global AMR_POPULATION_STREAM_REGISTER_XFACE_GLOBAL_APPLY_MAX_PARENT_LEVEL = Int(safe_get(CFG, "advanced", "refinement", "population_stream_register_xface_global_apply_max_parent_level"; default=2))
    global AMR_POPULATION_STREAM_REGISTER =
        safe_get(CFG, "advanced", "refinement", "population_stream_register"; default=false) ||
        AMR_POPULATION_STREAM_REGISTER_APPLY ||
        AMR_POPULATION_STREAM_REGISTER_FACE_APPLY ||
        AMR_POPULATION_STREAM_REGISTER_XFACE_HIGHPASS_APPLY ||
        AMR_POPULATION_STREAM_REGISTER_XFACE_GLOBAL_APPLY
    global AMR_POPULATION_STREAM_REGISTER_FACE_RESOLVED =
        AMR_POPULATION_STREAM_REGISTER &&
        (safe_get(CFG, "advanced", "refinement", "population_stream_register_face_resolved"; default=false) ||
         AMR_POPULATION_STREAM_REGISTER_FACE_APPLY ||
         AMR_POPULATION_STREAM_REGISTER_XFACE_HIGHPASS_APPLY ||
         AMR_POPULATION_STREAM_REGISTER_XFACE_GLOBAL_APPLY)
    global AMR_SHELL_RECALIBRATION = safe_get(CFG, "advanced", "refinement", "interface_shell_recalibration"; default=false)
    for (name, value) in (
        ("interface_shell_rho_alpha_x_single", AMR_SHELL_RHO_ALPHA_X_SINGLE),
        ("interface_shell_rho_alpha_x_multi", AMR_SHELL_RHO_ALPHA_X_MULTI),
        ("interface_shell_rho_alpha_t_single", AMR_SHELL_RHO_ALPHA_T_SINGLE),
        ("interface_shell_rho_alpha_t_multi", AMR_SHELL_RHO_ALPHA_T_MULTI),
        ("interface_shell_vel_alpha_x_single", AMR_SHELL_VEL_ALPHA_X_SINGLE),
        ("interface_shell_vel_alpha_x_multi", AMR_SHELL_VEL_ALPHA_X_MULTI),
        ("interface_shell_vel_alpha_t", AMR_SHELL_VEL_ALPHA_T),
        ("interface_shell_drho_cap_x_abs", AMR_SHELL_DRHO_CAP_X_ABS),
        ("interface_shell_drho_cap_x_rel", AMR_SHELL_DRHO_CAP_X_REL),
        ("interface_shell_drho_cap_t_abs", AMR_SHELL_DRHO_CAP_T_ABS),
        ("interface_shell_drho_cap_t_rel", AMR_SHELL_DRHO_CAP_T_REL),
        ("interface_shell_du_cap_x_abs", AMR_SHELL_DU_CAP_X_ABS),
        ("interface_shell_du_cap_x_rel", AMR_SHELL_DU_CAP_X_REL),
        ("interface_shell_du_cap_t_abs", AMR_SHELL_DU_CAP_T_ABS),
        ("interface_shell_du_cap_t_rel", AMR_SHELL_DU_CAP_T_REL),
        ("interface_shell_flux_alpha_x", AMR_SHELL_FLUX_ALPHA_X),
        ("interface_shell_flux_alpha_t", AMR_SHELL_FLUX_ALPHA_T),
        ("interface_shell_flux_du_cap_abs", AMR_SHELL_FLUX_DU_CAP_ABS),
        ("interface_shell_flux_du_cap_rel", AMR_SHELL_FLUX_DU_CAP_REL),
        ("interface_shell_momflux_alpha_x", AMR_SHELL_MOMFLUX_ALPHA_X),
        ("interface_shell_momflux_drho_cap_abs", AMR_SHELL_MOMFLUX_DRHO_CAP_ABS),
        ("interface_shell_momflux_drho_cap_rel", AMR_SHELL_MOMFLUX_DRHO_CAP_REL),
        ("interface_shell_momflux_stress_alpha_x", AMR_SHELL_MOMFLUX_STRESS_ALPHA_X),
        ("interface_shell_momflux_stress_cap_abs", AMR_SHELL_MOMFLUX_STRESS_CAP_ABS),
        ("interface_shell_momflux_stress_cap_rel", AMR_SHELL_MOMFLUX_STRESS_CAP_REL),
        ("interface_shell_population_blend_x", AMR_SHELL_POPULATION_BLEND_X),
        ("interface_shell_face_weight_x", AMR_SHELL_FACE_WEIGHT_X),
        ("interface_shell_face_population_alpha", AMR_SHELL_FACE_POP_ALPHA),
        ("interface_shell_face_stress_alpha", AMR_SHELL_FACE_STRESS_ALPHA),
        ("interface_shell_face_stress_rel_cap", AMR_SHELL_FACE_STRESS_REL_CAP),
        ("interface_shell_coupled_reflux_alpha", AMR_SHELL_COUPLED_REFLUX_ALPHA),
        ("interface_shell_paired_reflux_alpha", AMR_SHELL_PAIRED_REFLUX_ALPHA),
        ("interface_shell_paired_reflux_mass_alpha", AMR_SHELL_PAIRED_REFLUX_MASS_ALPHA),
        ("interface_shell_paired_reflux_parent_fraction", AMR_SHELL_PAIRED_REFLUX_PARENT_FRACTION),
        ("interface_shell_paired_reflux_mass_parent_fraction", AMR_SHELL_PAIRED_REFLUX_MASS_PARENT_FRACTION),
        ("interface_shell_paired_reflux_mass_cap_scale", AMR_SHELL_PAIRED_REFLUX_MASS_CAP_SCALE),
        ("interface_shell_paired_reflux_mass_cap_initial", AMR_SHELL_PAIRED_REFLUX_MASS_CAP_INITIAL),
        ("interface_shell_paired_reflux_highpass_blend", AMR_SHELL_PAIRED_REFLUX_HIGHPASS_BLEND),
        ("interface_shell_paired_reflux_lowpass_blend", AMR_SHELL_PAIRED_REFLUX_LOWPASS_BLEND),
        ("interface_shell_paired_reflux_second_moment_blend", AMR_SHELL_PAIRED_REFLUX_SECOND_MOMENT_BLEND),
        ("interface_shell_paired_reflux_post_stress_alpha", AMR_SHELL_PAIRED_REFLUX_POST_STRESS_ALPHA),
        ("interface_shell_paired_reflux_shared_state_rho_alpha", AMR_SHELL_PAIRED_REFLUX_SHARED_STATE_RHO_ALPHA),
        ("interface_shell_highpass_damper_alpha", AMR_SHELL_HIGHPASS_DAMPER_ALPHA),
        ("interface_shell_highpass_damper_rho_alpha", AMR_SHELL_HIGHPASS_DAMPER_RHO_ALPHA),
        ("amr_halo_fullcell_macro_alpha", AMR_HALO_FULLCELL_MACRO_ALPHA),
        ("amr_halo_fullcell_pop_alpha", AMR_HALO_FULLCELL_POP_ALPHA),
        ("runtime_face_budget_reflux_alpha", AMR_RUNTIME_FACE_BUDGET_REFLUX_ALPHA),
        ("gpu_face_budget_reflux_alpha", AMR_GPU_FACE_BUDGET_REFLUX_ALPHA),
        ("gpu_face_budget_reflux_mass_alpha", AMR_GPU_FACE_BUDGET_REFLUX_MASS_ALPHA),
    )
        (value >= 0.0f0 && value <= 1.0f0) || error("advanced.refinement.$name must be in [0, 1]")
    end
    (AMR_GPU_FACE_BUDGET_REFLUX_MOM_ALPHA >= -8.0f0 &&
     AMR_GPU_FACE_BUDGET_REFLUX_MOM_ALPHA <= 8.0f0) ||
        error("advanced.refinement.gpu_face_budget_reflux_mom_alpha must be in [-8, 8]")
    if AMR_SHELL_PAIRED_REFLUX_HIGHPASS && AMR_SHELL_PAIRED_REFLUX_LOWPASS
        error("advanced.refinement.interface_shell_paired_reflux_highpass and interface_shell_paired_reflux_lowpass are mutually exclusive")
    end
    (AMR_SHELL_PAIRED_REFLUX_MOM_ALPHA >= -8.0f0 &&
     AMR_SHELL_PAIRED_REFLUX_MOM_ALPHA <= 8.0f0) ||
        error("advanced.refinement.interface_shell_paired_reflux_mom_alpha must be in [-8, 8]")
    (AMR_SHELL_PAIRED_REFLUX_MOM_ALPHA_INITIAL >= -8.0f0 &&
     AMR_SHELL_PAIRED_REFLUX_MOM_ALPHA_INITIAL <= 8.0f0) ||
        error("advanced.refinement.interface_shell_paired_reflux_mom_alpha_initial must be in [-8, 8]")
    AMR_SHELL_PAIRED_REFLUX_MOM_ALPHA_HOLD_STEPS >= 0 ||
        error("advanced.refinement.interface_shell_paired_reflux_mom_alpha_hold_steps must be >= 0")
    AMR_GPU_FACE_BUDGET_REFLUX_DRHO_CAP_ABS >= 0.0f0 ||
        error("advanced.refinement.gpu_face_budget_reflux_drho_cap_abs must be >= 0")
    AMR_GPU_FACE_BUDGET_REFLUX_DRHO_CAP_REL >= 0.0f0 ||
        error("advanced.refinement.gpu_face_budget_reflux_drho_cap_rel must be >= 0")
    AMR_GPU_FACE_BUDGET_REFLUX_MOM_CAP_ABS >= 0.0f0 ||
        error("advanced.refinement.gpu_face_budget_reflux_mom_cap_abs must be >= 0")
    AMR_GPU_FACE_BUDGET_REFLUX_MOM_CAP_REL >= 0.0f0 ||
        error("advanced.refinement.gpu_face_budget_reflux_mom_cap_rel must be >= 0")
    AMR_SHELL_PAIRED_REFLUX_MASS_CAP_HOLD_STEPS >= 0 ||
        error("advanced.refinement.interface_shell_paired_reflux_mass_cap_hold_steps must be >= 0")
    AMR_SHELL_FACE_STRESS_ABS_CAP >= 0.0f0 ||
        error("advanced.refinement.interface_shell_face_stress_abs_cap must be >= 0")
    AMR_SHELL_HIGHPASS_DAMPER_CAP_ABS >= 0.0f0 ||
        error("advanced.refinement.interface_shell_highpass_damper_cap_abs must be >= 0")
    AMR_SHELL_HIGHPASS_DAMPER_CAP_REL >= 0.0f0 ||
        error("advanced.refinement.interface_shell_highpass_damper_cap_rel must be >= 0")
    AMR_SHELL_HIGHPASS_DAMPER_DEPTH >= 1 ||
        error("advanced.refinement.interface_shell_highpass_damper_depth must be >= 1")
    0.0f0 <= AMR_SHELL_HIGHPASS_DAMPER_RHO_ALPHA <= 1.0f0 ||
        error("advanced.refinement.interface_shell_highpass_damper_rho_alpha must be in [0, 1]")
    AMR_SHELL_HIGHPASS_DAMPER_DRHO_CAP_ABS >= 0.0f0 ||
        error("advanced.refinement.interface_shell_highpass_damper_drho_cap_abs must be >= 0")
    AMR_SHELL_HIGHPASS_DAMPER_DRHO_CAP_REL >= 0.0f0 ||
        error("advanced.refinement.interface_shell_highpass_damper_drho_cap_rel must be >= 0")
    AMR_RUNTIME_FACE_BUDGET_REFLUX_MAX_PARENT_LEVEL >= 1 ||
        error("advanced.refinement.runtime_face_budget_reflux_max_parent_level must be >= 1")
    AMR_GPU_FACE_BUDGET_REFLUX_MAX_PARENT_LEVEL >= 1 ||
        error("advanced.refinement.gpu_face_budget_reflux_max_parent_level must be >= 1")
    (AMR_POPULATION_STREAM_REGISTER_APPLY_ALPHA >= 0.0f0 &&
     AMR_POPULATION_STREAM_REGISTER_APPLY_ALPHA <= 1.0f0) ||
        error("advanced.refinement.population_stream_register_apply_alpha must be in [0, 1]")
    AMR_POPULATION_STREAM_REGISTER_APPLY_MAX_PARENT_LEVEL >= 1 ||
        error("advanced.refinement.population_stream_register_apply_max_parent_level must be >= 1")
    (AMR_POPULATION_STREAM_REGISTER_FACE_APPLY_ALPHA >= 0.0f0 &&
     AMR_POPULATION_STREAM_REGISTER_FACE_APPLY_ALPHA <= 1.0f0) ||
        error("advanced.refinement.population_stream_register_face_apply_alpha must be in [0, 1]")
    AMR_POPULATION_STREAM_REGISTER_FACE_APPLY_MAX_PARENT_LEVEL >= 1 ||
        error("advanced.refinement.population_stream_register_face_apply_max_parent_level must be >= 1")
    (AMR_POPULATION_STREAM_REGISTER_XFACE_HIGHPASS_APPLY_ALPHA >= 0.0f0 &&
     AMR_POPULATION_STREAM_REGISTER_XFACE_HIGHPASS_APPLY_ALPHA <= 1.0f0) ||
        error("advanced.refinement.population_stream_register_xface_highpass_apply_alpha must be in [0, 1]")
    AMR_POPULATION_STREAM_REGISTER_XFACE_HIGHPASS_APPLY_MAX_PARENT_LEVEL >= 1 ||
        error("advanced.refinement.population_stream_register_xface_highpass_apply_max_parent_level must be >= 1")
    (AMR_POPULATION_STREAM_REGISTER_XFACE_GLOBAL_APPLY_ALPHA >= 0.0f0 &&
     AMR_POPULATION_STREAM_REGISTER_XFACE_GLOBAL_APPLY_ALPHA <= 1.0f0) ||
        error("advanced.refinement.population_stream_register_xface_global_apply_alpha must be in [0, 1]")
    AMR_POPULATION_STREAM_REGISTER_XFACE_GLOBAL_APPLY_MAX_PARENT_LEVEL >= 1 ||
        error("advanced.refinement.population_stream_register_xface_global_apply_max_parent_level must be >= 1")
    global AMR_POST_RESTRICT_SPONGE = safe_get(CFG, "advanced", "refinement", "post_restrict_sponge"; default=true)
    global AMR_PARTIAL_FRONTIER_SYNC = safe_get(CFG, "advanced", "refinement", "partial_frontier_sync"; default=false)
    global AMR_ENFORCE_Z_SYMMETRY = safe_get(CFG, "advanced", "refinement", "enforce_z_symmetry"; default=false)
    global AMR_ENFORCE_Y_SYMMETRY = safe_get(CFG, "advanced", "refinement", "enforce_y_symmetry"; default=false)
    global AMR_THERMAL_NONEQ_TRANSFER = safe_get(CFG, "advanced", "refinement", "thermal_noneq_transfer"; default=false)
    global AMR_RESTRICTION_PI_NEQ_FILTER = safe_get(CFG, "advanced", "refinement", "restriction_pi_neq_filter"; default=false)
    # Pass 265 (Tier 1 #1e). Read the moment-correction pass count; clamp to [1, 8].
    let n_corr = Int(safe_get(CFG, "advanced", "refinement", "restriction_correction_passes"; default=2))
        (n_corr >= 1 && n_corr <= 8) ||
            error("advanced.refinement.restriction_correction_passes must be in [1, 8] (got $n_corr)")
        global AMR_RESTRICTION_CORRECTION_PASSES = Int32(n_corr)
    end
    global AMR_RESTRICTION_CONSERVATIVE_FEQ = safe_get(CFG, "advanced", "refinement",
                                                       "restriction_conservative_feq"; default=false)
    # SOTA P1: conservative ρE restriction. Requires conservative_feq=true.
    global AMR_RESTRICTION_CONSERVATIVE_RHOE = safe_get(CFG, "advanced", "refinement",
                                                       "restriction_conservative_rhoe"; default=false)
    # B1 Pillar 2 — 13-moment-exact (mortar-element) restriction
    global AMR_RESTRICTION_CONSERVATIVE_13MOM = safe_get(CFG, "advanced", "refinement",
                                                         "restriction_conservative_13mom"; default=false)
    if AMR_RESTRICTION_CONSERVATIVE_13MOM
        # The new path bypasses the legacy 4-/5-mom correction on f_eq, but it
        # depends on the same numerical-equilibrium machinery; implicitly turn
        # the conservative-feq guard on so the diagnostics stay consistent.
        global AMR_RESTRICTION_CONSERVATIVE_FEQ = true
    end
    # GOLD-STANDARD STRICT CONSERVATION (2026-06-17): D3Q19 4-moment exact restriction
    global AMR_RESTRICTION_CONSERVATIVE_D3Q19 = safe_get(CFG, "advanced", "refinement",
                                                         "restriction_conservative_d3q19"; default=false)
    global AMR_RESTRICTION_CONSERVATIVE_D3Q27 = safe_get(CFG, "advanced", "refinement",
                                                         "restriction_conservative_d3q27"; default=false)
    global QUINTIC_BSPLINE_PROLONGATION = safe_get(CFG, "advanced", "refinement",
                                                   "quintic_bspline_prolongation"; default=false)
    QUINTIC_BSPLINE_PROLONGATION &&
        println("[Init]    C2/C4 quintic B-spline AMR prolongation: ON (D3Q27 / D3Q39)")
    global THERMAL_INTENSIVE_ADVECTION = safe_get(CFG, "advanced", "thermal",
                                                  "intensive_advection"; default=false)
    THERMAL_INTENSIVE_ADVECTION &&
        println("[Init]    Intensive-T thermal advection: ON (rescale streamed g by ρ_local/ρ_upstream — fixes density-gradient spurious heating)")
    global GENDRE_NORMAL_GRADIENT = safe_get(CFG, "advanced", "refinement",
                                             "gendre_normal_gradient"; default=false)
    if GENDRE_NORMAL_GRADIENT
        if LATTICE_TYPE == :d3q39
            println("[Init]    Gendre directional-splitting normal-gradient AMR correction: ON (spurious-acoustic seam fix)")
        else
            @warn "refinement.gendre_normal_gradient is D3Q39-entropic only — inert on $(LATTICE_TYPE); ignoring."
            global GENDRE_NORMAL_GRADIENT = false
        end
    end
    # B3 — FP64 halo arithmetic (restriction side): 8-child accumulation +
    # 39-direction moment sum in restrict_child_state_to_parent_kernel! run in
    # Float64 internally. No prerequisite.
    global INTERFACE_FP64_HALO = safe_get(CFG, "advanced", "refinement",
                                          "fp64_halo"; default=false)
    if INTERFACE_FP64_HALO
        @info "refinement.fp64_halo enabled — restriction-side FP64 active."
    end
    global HRR_HYBRID_ENABLED = safe_get(CFG, "advanced", "numerics", "hrr_hybridization"; default=false)
    global HRR_SIGMA = Float32(safe_get(CFG, "advanced", "numerics", "hrr_sigma"; default=1.0))
    # Pass 261 — per-level σ_HRR override list. Empty when key is absent.
    let hrr_pl_raw = safe_get(CFG, "advanced", "numerics", "hrr_sigma_per_level"; default=nothing)
        if hrr_pl_raw === nothing
            global HRR_SIGMA_PER_LEVEL = Float32[]
        else
            global HRR_SIGMA_PER_LEVEL = Float32[Float32(x) for x in hrr_pl_raw]
            for (i, σ) in enumerate(HRR_SIGMA_PER_LEVEL)
                (σ > 0.0f0 && σ <= 1.0f0) ||
                    error("advanced.numerics.hrr_sigma_per_level[$i] must be in (0, 1]")
            end
        end
    end
    FEATHER_DEPTH < 0 && error("advanced.refinement.feather_depth must be >= 0")
    (COARSE_SIDE_FEATHER_SCALE >= 0.0f0 && COARSE_SIDE_FEATHER_SCALE <= 1.0f0) ||
        error("advanced.refinement.coarse_side_feather_scale must be in [0, 1]")
    AMR_REGULARIZATION_DEPTH < 0 && error("advanced.refinement.interface_regularization_depth must be >= 0")
    # Recursive Regularization (RR-LBM) — only valid on D3Q39 entropic; ignored otherwise.
    global RR_LBM_ENABLED = safe_get(CFG, "advanced", "numerics",
                                     "recursive_regularization", "enabled"; default=false)
    local _rr_minlvl_raw = safe_get(CFG, "advanced", "numerics",
                                    "recursive_regularization", "min_level"; default=4)
    if (_rr_minlvl_raw isa AbstractString && lowercase(strip(_rr_minlvl_raw)) == "finest") ||
       (_rr_minlvl_raw isa Number && _rr_minlvl_raw <= 0)
        # "finest only" — robust to the actual runtime level count (incl. AMR clamping).
        global RR_LBM_FINEST_ONLY = true
        global RR_LBM_MIN_LEVEL = Int32(1)   # placeholder; the gate uses NUM_LEVELS_CONFIG
        RR_LBM_ENABLED && println("[Init]    RR-LBM: FINEST LEVEL ONLY (min_level=\"finest\" ⇒ level_id == num_levels at runtime)")
    else
        global RR_LBM_FINEST_ONLY = false
        global RR_LBM_MIN_LEVEL = Int32(_rr_minlvl_raw)
        RR_LBM_MIN_LEVEL < 1 && error("advanced.numerics.recursive_regularization.min_level must be >= 1, or the string \"finest\"")
    end
    # SVV Y-Z bi-Laplacian (Tadmor 1989 / Karamanos-Karniadakis 2000) — D3Q39 entropic only.
    global SVV_YZ_ENABLED = safe_get(CFG, "advanced", "numerics",
                                     "svv_yz", "enabled"; default=false)
    global SVV_YZ_EPSILON = Float32(safe_get(CFG, "advanced", "numerics",
                                     "svv_yz", "epsilon"; default=0.002))
    global SVV_YZ_MIN_LEVEL = Int32(safe_get(CFG, "advanced", "numerics",
                                     "svv_yz", "min_level"; default=4))
    SVV_YZ_MIN_LEVEL < 1 && error("advanced.numerics.svv_yz.min_level must be >= 1")
    SVV_YZ_EPSILON < 0.0f0 && error("advanced.numerics.svv_yz.epsilon must be >= 0")
    # SOTA #5: streamwise-X SVV (coarse-level banding cure).
    global SVV_X_ENABLED = safe_get(CFG, "advanced", "numerics",
                                    "svv_x", "enabled"; default=false)
    global SVV_X_EPSILON = Float32(safe_get(CFG, "advanced", "numerics",
                                    "svv_x", "epsilon"; default=0.002))
    global SVV_X_MAX_LEVEL = Int32(safe_get(CFG, "advanced", "numerics",
                                    "svv_x", "max_level"; default=1))
    SVV_X_EPSILON < 0.0f0 && error("advanced.numerics.svv_x.epsilon must be >= 0")
    if SVV_X_ENABLED && (LATTICE_TYPE == :d3q39)
        println("[Stability] Streamwise-X SVV ENABLED on levels ≤ $(SVV_X_MAX_LEVEL) (ε=$(SVV_X_EPSILON)) — L1 banding cure")
    end
    # SOTA A3: two-way EOS coupling. Default false = legacy linearized p = ρ·cs²·T.
    # When true, uses p = (γ-1)·ρ·cv·T everywhere the 13-moment solver computes
    # variance targets, and in the slice/VTK pressure outputs.
    global EOS_THERMODYNAMIC = safe_get(CFG, "advanced", "numerics",
                                        "eos_thermodynamic"; default=false)
    # AUDIT FIX (2026-06-03): configurable macro realizability clamp bounds.
    # Defaults = legacy hard-coded values (byte-identical for existing aero cases).
    # Canonical low-density cases (Sod/Lax/Riemann) set macro_density_floor: 0.05.
    global MACRO_DENSITY_FLOOR = Float32(safe_get(CFG, "advanced", "numerics",
                                         "macro_density_floor"; default=0.42))
    global MACRO_DENSITY_CEIL = Float32(safe_get(CFG, "advanced", "numerics",
                                        "macro_density_ceil"; default=2.10))
    global MACRO_DENSITY_FLOOR_NEAR = Float32(safe_get(CFG, "advanced", "numerics",
                                              "macro_density_floor_near"; default=0.35))
    global MACRO_DENSITY_CEIL_NEAR = Float32(safe_get(CFG, "advanced", "numerics",
                                             "macro_density_ceil_near"; default=2.40))
    (MACRO_DENSITY_FLOOR > 0.0f0 && MACRO_DENSITY_FLOOR < MACRO_DENSITY_CEIL) ||
        error("advanced.numerics.macro_density_floor must be >0 and < macro_density_ceil")
    global ADAPTIVE_OPERATOR = Bool(safe_get(CFG, "advanced", "numerics",
                                        "adaptive_operator", "enabled"; default=false))
    global ADAPTIVE_MA_THRESH = Float32(safe_get(CFG, "advanced", "numerics",
                                        "adaptive_operator", "ma_threshold"; default=0.25))
    if ADAPTIVE_OPERATOR
        println("[Perf] Adaptive operator ENABLED: cheap O(u⁴) poly f_eq for Ma_local<$(ADAPTIVE_MA_THRESH) (away from AMR halo/wall) — skips Newton on smooth cells")
    end
    # SOTA B2: dynamic Smagorinsky (Germano-Lilly). Phase C: when variable_nu_sgs was
    # auto-enabled (high-Re D3Q27 or pass-168 D3Q39) and the user didn't set
    # dynamic_germano, default it ON — it's the self-calibrating gold-standard form of
    # the SGS we just turned on (removes the hand-tuned c_pi). User-set value wins.
    local _germano_explicit = config_has_path(CFG, "advanced", "numerics", "variable_nu_sgs", "dynamic_germano")
    local _germano_default = (!_germano_explicit && VARIABLE_NU_SGS &&
                              (REGIME_SELECTION.lattice_auto || REGIME_SELECTION.operator_auto))
    global DYNAMIC_SGS_ENABLED = safe_get(CFG, "advanced", "numerics",
                                          "variable_nu_sgs", "dynamic_germano"; default=_germano_default)
    if _germano_default
        println("[Stability] Auto-enabled dynamic Germano-Lilly SGS (variable_nu_sgs is on; self-calibrating c_d).")
    end
    # SOTA A4: NSCBC outflow BC
    global OUTLET_NSCBC_ENABLED = safe_get(CFG, "advanced", "boundary",
                                           "outlet_nscbc", "enabled"; default=false)
    global OUTLET_NSCBC_P_TARGET = Float32(safe_get(CFG, "advanced", "boundary",
                                           "outlet_nscbc", "p_target"; default=1.0))
    global OUTLET_NSCBC_SIGMA = Float32(safe_get(CFG, "advanced", "boundary",
                                           "outlet_nscbc", "sigma_relax"; default=0.27))
    global OUTLET_NSCBC_NEQ_BLEND = Float32(safe_get(CFG, "advanced", "boundary",
                                           "outlet_nscbc", "neq_blend"; default=0.5))
    # Global mass-conservation controller. Default ON when NSCBC outlet + compressible
    # (fixed-velocity inlet cases pump without it). Explicit setting wins.
    local _mass_ctrl_default = OUTLET_NSCBC_ENABLED && COMPRESSIBLE_MODE
    global MASS_CONTROL_ENABLED = Bool(safe_get(CFG, "advanced", "boundary",
                                       "mass_control", "enabled"; default=_mass_ctrl_default))
    global MASS_CONTROL_GAIN = Float32(safe_get(CFG, "advanced", "boundary",
                                       "mass_control", "gain"; default=2.0))
    global MASS_CONTROL_BASELINE_STEP = Int(safe_get(CFG, "advanced", "boundary",
                                       "mass_control", "baseline_step"; default=8))
    global MASS_CONTROL_MODE = Symbol(safe_get(CFG, "advanced", "boundary",
                                       "mass_control", "mode"; default="nscbc_ptarget"))
    global MASS_CONTROL_RESCALE_EPS = Float32(safe_get(CFG, "advanced", "boundary",
                                       "mass_control", "rescale_eps"; default=1.0e-3))
    if MASS_CONTROL_ENABLED
        println("[Stability] Global mass-conservation controller ENABLED (mode=$(MASS_CONTROL_MODE), gain=$(MASS_CONTROL_GAIN), eps=$(MASS_CONTROL_RESCALE_EPS), baseline@step $(MASS_CONTROL_BASELINE_STEP)) — fixes over-constrained-BC mass pump")
    end
    global REFINEMENT_STRATEGY = Symbol(safe_get(CFG, "advanced", "refinement", "strategy"; default="geometry_first"))
    global REFINEMENT_MARKER_ONLY = safe_get(CFG, "advanced", "refinement", "marker_only"; default=false)
    global ENABLE_WAKE_REFINEMENT = safe_get(CFG, "advanced", "refinement", "wake_enabled"; default=false)
    global WAKE_REFINEMENT_LENGTH = Float64(safe_get(CFG, "advanced", "refinement", "wake_length"; default=0.25))
    global WAKE_REFINEMENT_WIDTH_FACTOR = Float64(safe_get(CFG, "advanced", "refinement", "wake_width_factor"; default=0.1))
    global WAKE_REFINEMENT_HEIGHT_FACTOR = Float64(safe_get(CFG, "advanced", "refinement", "wake_height_factor"; default=0.1))
    global WAKE_ANISOTROPIC_PAD = safe_get(CFG, "advanced", "refinement", "wake_anisotropic_pad"; default=false)
    global WAKE_PAD_UPSTREAM_SCALE = Float64(safe_get(CFG, "advanced", "refinement", "wake_pad_upstream_scale"; default=0.25))
    global WAKE_PAD_LATERAL_SCALE = Float64(safe_get(CFG, "advanced", "refinement", "wake_pad_lateral_scale"; default=0.5))
    global REFINEMENT_BODY_SHELL_BLOCKS = Int(safe_get(CFG, "advanced", "refinement", "body_shell_blocks"; default=0))
    REFINEMENT_BODY_SHELL_BLOCKS < 0 && error("advanced.refinement.body_shell_blocks must be >= 0")
    global WAKE_MAX_LEVEL = Int(safe_get(CFG, "advanced", "refinement", "wake_max_level"; default=0))
    WAKE_MAX_LEVEL < 0 && error("advanced.refinement.wake_max_level must be >= 0")
    REFINEMENT_BODY_SHELL_BLOCKS > 0 &&
        println("[Init]    Obstacle standoff: body_shell_blocks=$REFINEMENT_BODY_SHELL_BLOCKS (guaranteed hugging separation)")
    WAKE_MAX_LEVEL > 0 &&
        println("[Init]    Wake cap: wake_max_level=$WAKE_MAX_LEVEL (no finest-level wake slab)")
    # Sharp-edge (feature-edge) local refinement.
    global SHARP_EDGE_ANGLE_THRESHOLD_DEG = Float64(safe_get(CFG, "advanced", "refinement", "sharp_edge_angle_threshold_deg"; default=0.0))
    SHARP_EDGE_ANGLE_THRESHOLD_DEG < 0.0 && error("advanced.refinement.sharp_edge_angle_threshold_deg must be >= 0")
    global SHARP_EDGE_MARGIN_BLOCKS = Int(safe_get(CFG, "advanced", "refinement", "sharp_edge_margin_blocks"; default=1))
    SHARP_EDGE_MARGIN_BLOCKS < 0 && error("advanced.refinement.sharp_edge_margin_blocks must be >= 0")
    global SHARP_EDGE_REFINE = SHARP_EDGE_ANGLE_THRESHOLD_DEG > 0.0
    SHARP_EDGE_REFINE &&
        println("[Init]    Sharp-edge refinement: angle>$(SHARP_EDGE_ANGLE_THRESHOLD_DEG)° between adjacent triangles → +$SHARP_EDGE_MARGIN_BLOCKS finest-level block margin")
    
    global BOUNDARY_METHOD = Symbol(safe_get(CFG, "advanced", "boundary", "method"; default="bouzidi"))
    global TRANSVERSE_FREESTREAM_BOUNDARY = safe_get(CFG, "advanced", "boundary", "transverse_freestream"; default=false)
    global TRANSVERSE_FREESTREAM_BLEND = Float32(safe_get(CFG, "advanced", "boundary", "transverse_freestream_blend"; default=1.0))
    (TRANSVERSE_FREESTREAM_BLEND < 0.0f0 || TRANSVERSE_FREESTREAM_BLEND > 1.0f0) &&
        error("advanced.boundary.transverse_freestream_blend must be in [0, 1]")
    global OUTLET_FREESTREAM_BOUNDARY = safe_get(CFG, "advanced", "boundary", "outlet_freestream"; default=false)
    outlet_mode_raw = safe_get(CFG, "advanced", "boundary", "outlet_mode"; default=OUTLET_FREESTREAM_BOUNDARY ? "freestream" : "zerogradient")
    outlet_mode_sym = Symbol(lowercase(String(outlet_mode_raw)))
    outlet_mode_sym in (:legacy, :freestream, :zerogradient, :extrapolate) ||
        error("advanced.boundary.outlet_mode must be \"legacy\", \"freestream\", \"zerogradient\", or \"extrapolate\" (got \"$outlet_mode_raw\")")
    global OUTLET_BOUNDARY_MODE = outlet_mode_sym == :freestream ? Int32(1) :
                                  outlet_mode_sym == :zerogradient ? Int32(2) :
                                  outlet_mode_sym == :extrapolate ? Int32(3) : Int32(0)
    # AUDIT FIX (2026-06-03): `outlet_mode: extrapolate` (mode 3) shares this
    # blend knob with `freestream` (mode 1). The legacy default of 1.0 silently
    # turned "extrapolate" into a full freestream-equilibrium clamp (the interior
    # extrapolation was discarded entirely), breaking shock-tube / low-back-
    # pressure / supersonic-outlet cases. Make the default MODE-AWARE: when the
    # user selects `extrapolate` and does NOT explicitly set the blend, default
    # it to 0.0 (true zero-gradient extrapolation). `freestream` mode keeps the
    # 1.0 default so existing aero cases are byte-identical. An explicit
    # `outlet_freestream_blend:` in the case file is always respected.
    local _ofb_sentinel = -1.0
    local _ofb_raw = Float64(safe_get(CFG, "advanced", "boundary", "outlet_freestream_blend"; default=_ofb_sentinel))
    if _ofb_raw == _ofb_sentinel
        # Not set by the user → choose a physically-correct default per mode.
        _ofb_raw = (OUTLET_BOUNDARY_MODE == Int32(3)) ? 0.0 : 1.0
        if OUTLET_BOUNDARY_MODE == Int32(3)
            println("[Boundary] outlet_mode=extrapolate with no explicit outlet_freestream_blend → defaulting to 0.0 (true extrapolation)")
        end
    end
    global OUTLET_FREESTREAM_BLEND = Float32(_ofb_raw)
    (OUTLET_FREESTREAM_BLEND < 0.0f0 || OUTLET_FREESTREAM_BLEND > 1.0f0) &&
        error("advanced.boundary.outlet_freestream_blend must be in [0, 1]")
    inlet_density_raw = safe_get(CFG, "advanced", "boundary", "inlet_density"; default="freestream")
    inlet_density_sym = Symbol(lowercase(String(inlet_density_raw)))
    inlet_density_sym in (:freestream, :local) ||
        error("advanced.boundary.inlet_density must be \"freestream\" or \"local\" (got \"$inlet_density_raw\")")
    global INLET_DENSITY_MODE = inlet_density_sym == :local ? Int32(1) : Int32(0)
    global SPAN_PERIODIC_Y = Bool(safe_get(CFG, "advanced", "boundary", "periodic_y"; default=false))
    if SPAN_PERIODIC_Y
        println("[Boundary] Spanwise Y periodic topology ENABLED")
        # (periodic-Y AMR: the warning below does not apply when periodic_y_amr is set; see the marked block)
        if NUM_LEVELS_CONFIG != 1 && !(safe_get(CFG, "advanced", "boundary", "periodic_y_amr"; default=false) === true)
            @warn "boundary.periodic_y currently applies only to single-level runs; AMR levels will use non-periodic topology."
        end
    end
    # Non-reflecting inlet blend. Default 0.9 for compressible (lets ~10% of the
    # acoustic wave transmit out per step → kills the reflective standing wave)
    # while keeping the inlet velocity well-imposed. 1.0 = legacy reflective.
    local _inlet_neq_default = COMPRESSIBLE_MODE ? 0.9 : 1.0
    global INLET_NEQ_BLEND = Float32(safe_get(CFG, "advanced", "boundary",
                                     "inlet_neq_blend"; default=_inlet_neq_default))
    if INLET_NEQ_BLEND < 1.0f0
        println("[Stability] Non-reflecting inlet ENABLED (neq_blend=$(INLET_NEQ_BLEND)) — transmits acoustic waves, kills reflective banding")
    end
    global BOUZIDI_LEVELS = Int(safe_get(CFG, "advanced", "boundary", "bouzidi_levels"; default=1))
    global Q_MIN_THRESHOLD = Float32(safe_get(CFG, "advanced", "boundary", "q_min_threshold"; default=0.001))
    q_geometry_raw = safe_get(CFG, "advanced", "boundary", "q_geometry"; default="stl_ray")
    q_geometry_sym = Symbol(lowercase(String(q_geometry_raw)))
    q_geometry_sym in (:stl_ray, :sphere_analytic) ||
        error("advanced.boundary.q_geometry must be \"stl_ray\" or \"sphere_analytic\" (got \"$q_geometry_raw\")")
    global BOUZIDI_Q_GEOMETRY = q_geometry_sym
    global BOUZIDI_SYMMETRY_REPORT = safe_get(CFG, "advanced", "boundary", "symmetry_report"; default=false)
    global BOUZIDI_FF_AUDIT = safe_get(CFG, "advanced", "boundary", "ff_audit"; default=false)
    global BOUZIDI_MISSED_LINK_AUDIT = safe_get(CFG, "advanced", "boundary", "missed_link_audit"; default=false)
    missed_link_fix_raw = safe_get(CFG, "advanced", "boundary", "missed_link_fix"; default="off")
    missed_link_fix_sym = Symbol(missed_link_fix_raw)
    missed_link_fix_sym in (:off, :wall_eq, :bounce_back, :blend, :bounded_blend) ||
        error("advanced.boundary.missed_link_fix must be \"off\", \"wall_eq\", \"bounce_back\", \"blend\", or \"bounded_blend\" (got \"$missed_link_fix_raw\")")
    global BOUZIDI_MISSED_LINK_FIX = missed_link_fix_sym
    global BOUZIDI_MISSED_LINK_FIX_MODE = missed_link_fix_sym == :wall_eq ? Int32(1) :
                                           missed_link_fix_sym == :bounce_back ? Int32(2) :
                                           missed_link_fix_sym == :blend ? Int32(3) :
                                           missed_link_fix_sym == :bounded_blend ? Int32(4) : Int32(0)
    global BOUZIDI_MISSED_LINK_EQ_BLEND = Float32(safe_get(CFG, "advanced", "boundary", "missed_link_eq_blend"; default=0.35))
    (BOUZIDI_MISSED_LINK_EQ_BLEND < 0.0f0 || BOUZIDI_MISSED_LINK_EQ_BLEND > 1.0f0) &&
        error("advanced.boundary.missed_link_eq_blend must be in [0, 1]")
    global BOUZIDI_MISSED_LINK_DELTA_REL_CAP = Float32(safe_get(CFG, "advanced", "boundary", "missed_link_delta_rel_cap"; default=0.08))
    global BOUZIDI_MISSED_LINK_DELTA_ABS_CAP = Float32(safe_get(CFG, "advanced", "boundary", "missed_link_delta_abs_cap"; default=2.0e-4))
    BOUZIDI_MISSED_LINK_DELTA_REL_CAP >= 0.0f0 ||
        error("advanced.boundary.missed_link_delta_rel_cap must be >= 0")
    BOUZIDI_MISSED_LINK_DELTA_ABS_CAP >= 0.0f0 ||
        error("advanced.boundary.missed_link_delta_abs_cap must be >= 0")
    global BOUZIDI_WALL_POP_LIMITER_REL_CAP = Float32(safe_get(CFG, "advanced", "boundary", "wall_pop_limiter_rel_cap"; default=0.0))
    global BOUZIDI_WALL_POP_LIMITER_ABS_CAP = Float32(safe_get(CFG, "advanced", "boundary", "wall_pop_limiter_abs_cap"; default=0.0))
    BOUZIDI_WALL_POP_LIMITER_REL_CAP >= 0.0f0 ||
        error("advanced.boundary.wall_pop_limiter_rel_cap must be >= 0")
    BOUZIDI_WALL_POP_LIMITER_ABS_CAP >= 0.0f0 ||
        error("advanced.boundary.wall_pop_limiter_abs_cap must be >= 0")
    if BOUZIDI_WALL_POP_LIMITER_REL_CAP > 0.0f0 || BOUZIDI_WALL_POP_LIMITER_ABS_CAP > 0.0f0
        @printf("[Stability] Bouzidi wall-pop limiter ENABLED (rel=%.3f abs=%.3e)\n",
                BOUZIDI_WALL_POP_LIMITER_REL_CAP, BOUZIDI_WALL_POP_LIMITER_ABS_CAP)
    end
    # Pass193+: q<0.5 behind-cell fallback mode. "halfway_bb" preserves the
    # pre-pass193 behavior; "wall_eq" replaces the silent halfway-BB collapse
    # with the local no-slip equilibrium projection; "bounce_back" preserves
    # the local non-equilibrium pair content when the geometric support is
    # missing. See PASS192_DIAGNOSIS.md and PASS196_DIAGNOSIS.md.
    ff_fb_raw = safe_get(CFG, "advanced", "boundary", "ff_fallback"; default="halfway_bb")
    ff_fb_sym = Symbol(ff_fb_raw)
    ff_fb_sym in (:halfway_bb, :wall_eq, :bounce_back) ||
        error("advanced.boundary.ff_fallback must be \"halfway_bb\", \"wall_eq\", or \"bounce_back\" (got \"$ff_fb_raw\")")
    global BOUZIDI_FF_FALLBACK = ff_fb_sym
    global BOUZIDI_FF_FALLBACK_MODE = ff_fb_sym == :wall_eq ? Int32(1) :
                                      ff_fb_sym == :bounce_back ? Int32(2) : Int32(0)
    global BOUZIDI_SHIFTED_WALL_CORRECTION = safe_get(CFG, "advanced", "boundary",
                                                       "shifted_wall_correction"; default=true)
    global BOUZIDI_SHIFTED_WALL_CORRECTION_SCALE = Float32(safe_get(CFG, "advanced", "boundary",
                                                                    "shifted_wall_correction_scale"; default=1.0))
    global BOUZIDI_SHIFTED_WALL_CORRECTION_RAMP_STEPS = Int32(safe_get(CFG, "advanced", "boundary",
                                                                       "shifted_wall_correction_ramp_steps"; default=0))
    global BOUZIDI_SHIFTED_WALL_CORRECTION_CAP_REL = Float32(safe_get(CFG, "advanced", "boundary",
                                                                      "shifted_wall_correction_cap_rel"; default=0.25))
    global BOUZIDI_SHIFTED_WALL_CORRECTION_CAP_ABS = Float32(safe_get(CFG, "advanced", "boundary",
                                                                      "shifted_wall_correction_cap_abs"; default=0.0))
    (BOUZIDI_SHIFTED_WALL_CORRECTION_SCALE < 0.0f0 || BOUZIDI_SHIFTED_WALL_CORRECTION_SCALE > 1.0f0) &&
        error("advanced.boundary.shifted_wall_correction_scale must be in [0, 1]")
    BOUZIDI_SHIFTED_WALL_CORRECTION_CAP_REL >= 0.0f0 ||
        error("advanced.boundary.shifted_wall_correction_cap_rel must be >= 0")
    BOUZIDI_SHIFTED_WALL_CORRECTION_CAP_ABS >= 0.0f0 ||
        error("advanced.boundary.shifted_wall_correction_cap_abs must be >= 0")
    BOUZIDI_SHIFTED_WALL_CORRECTION_RAMP_STEPS >= Int32(0) ||
        error("advanced.boundary.shifted_wall_correction_ramp_steps must be >= 0")

    # ── B7 — Gold-standard wall BC ladder selector ──
    bc_scheme_raw = safe_get(CFG, "advanced", "boundary", "bc_scheme"; default="bouzidi_linear")
    bc_scheme_sym = Symbol(bc_scheme_raw)
    bc_scheme_sym in (:bouzidi_linear, :ymls_quadratic, :marson_enriched, :grad) ||
        error("advanced.boundary.bc_scheme must be \"bouzidi_linear\", \"ymls_quadratic\", \"marson_enriched\", or \"grad\" (got \"$bc_scheme_raw\")")
    global BC_SCHEME = bc_scheme_sym
    # ── Grad-reconstruction (regularized) wall BC knobs (2026-06-05) ──
    # bc_scheme: "grad" rebuilds all wall-cell pops from (ρ, u_wall, Π_neq) — q-free,
    # normal-free, axis-agnostic; cures the Bouzidi curved-wall Nyquist ring + axis
    # hot spots. grad_pineq_scale∈[0,1]: keep-wall-shear (1=full Grad, 0=diffusive).
    # grad_velocity_mode: "zero" = hard no-slip at boundary cells; "wall_dist_blend"
    # = relax toward no-slip by sub-cell wall proximity (recovers the standoff).
    global GRAD_WALL_PINEQ_SCALE = Float32(safe_get(CFG, "advanced", "boundary", "grad_pineq_scale"; default=1.0))
    local _gvm = String(safe_get(CFG, "advanced", "boundary", "grad_velocity_mode"; default="zero"))
    # "zero" = hard no-slip on every staircased shell cell (mode 0; legacy);
    # "wall_dist_blend" = blend by 3-valued cell-CENTER distance (mode 1; weak);
    # "q_blend" = blend by the bouzidi per-link sub-cell q (mode 2; DE-STAIRCASES
    #             the curved-wall Nyquist ring — recommended for curved bodies).
    global GRAD_WALL_VELOCITY_MODE = _gvm == "q_blend"         ? Int32(2) :
                                     _gvm == "wall_dist_blend" ? Int32(1) : Int32(0)
    if BC_SCHEME == :grad
        println("[Init]    Wall BC: Grad reconstruction (regularized) — q-free/normal-free/axis-agnostic; pineq_scale=$(GRAD_WALL_PINEQ_SCALE), vel_mode=$(_gvm)")
    end

    global FORCE_COMPUTATION_ENABLED = safe_get(CFG, "advanced", "forces", "enabled"; default=true)
    global FORCE_OUTPUT_FREQ_CONFIG = Int(safe_get(CFG, "advanced", "forces", "output_freq"; default=0))
    global MOMENT_CENTER_CONFIG = safe_get(CFG, "advanced", "forces", "moment_center"; default=[0.25, 0.0, 0.0])
    global NET_FORCE_EXPORT_ENABLED = safe_get(CFG, "advanced", "forces", "net_force_export"; default=false)
    
    global DIAG_FREQ = Int(safe_get(CFG, "advanced", "diagnostics", "freq"; default=500))
    global STABILITY_CHECK_ENABLED = safe_get(CFG, "advanced", "diagnostics", "stability_check"; default=true)
    global PRINT_TAU_WARNING = safe_get(CFG, "advanced", "diagnostics", "print_tau_warning"; default=true)
    global THERMAL_INTERFACE_DIAGNOSTICS = safe_get(CFG, "advanced", "diagnostics", "thermal_interface_diagnostics"; default=false)
    global AMR_INTERFACE_DIAGNOSTICS = safe_get(CFG, "advanced", "diagnostics", "amr_interface_diagnostics"; default=false)
    global AMR_SEAM_DIAGNOSTIC_ALL_STEPS = safe_get(CFG, "advanced", "diagnostics", "amr_seam_all_steps"; default=false)
    global AMR_VALIDATION_GATES = safe_get(CFG, "advanced", "diagnostics", "amr_validation_gates"; default=false)
    global AMR_VALIDATION_FREQ = Int(safe_get(CFG, "advanced", "diagnostics", "amr_validation_freq"; default=0))
    global VORTEX_METRICS_ENABLED = safe_get(CFG, "advanced", "diagnostics", "vortex_metrics"; default=false)
    global SOD_CENTERLINE_ENABLED = safe_get(CFG, "advanced", "diagnostics", "sod_centerline"; default=false)
    if AMR_VALIDATION_GATES && AMR_VALIDATION_FREQ == 0
        # Fall back to slice_output_freq if unset.
        global AMR_VALIDATION_FREQ = Int(safe_get(CFG, "basic", "simulation", "slice_output_freq"; default=20))
    end
    global AMR_FULL_DIAGNOSTICS_FREQ = Int(safe_get(CFG, "advanced", "diagnostics", "amr_full_diagnostics_freq"; default=0))
    global AMR_FACE_REGISTER_DIAGNOSTICS = safe_get(CFG, "advanced", "diagnostics", "amr_face_register_diagnostics"; default=false)
    global AMR_RUNTIME_FACE_BUDGET_DIAGNOSTICS = safe_get(CFG, "advanced", "diagnostics", "amr_runtime_face_budget_diagnostics"; default=false)
    global MASS_CONSERVATION_DIAGNOSTICS = safe_get(CFG, "advanced", "diagnostics", "mass_conservation_diagnostics"; default=false)
    global COMPRESSIBLE_AUDIT_DIAGNOSTICS = safe_get(CFG, "advanced", "diagnostics", "compressible_audit"; default=false)
    global TRANSONIC_WAKE_DIAGNOSTICS = Bool(safe_get(CFG, "advanced", "diagnostics", "transonic_wake", "enabled"; default=false))
    global TRANSONIC_WAKE_DIAG_X_MIN_CELLS = Float32(safe_get(CFG, "advanced", "diagnostics", "transonic_wake", "x_min_cells"; default=0.0))
    global TRANSONIC_WAKE_DIAG_MAX_X_DIAMETERS = Float32(safe_get(CFG, "advanced", "diagnostics", "transonic_wake", "max_x_diameters"; default=3.0))
    global TRANSONIC_WAKE_OVERSPEED_WARN_FRAC = Float32(safe_get(CFG, "advanced", "diagnostics", "transonic_wake", "overspeed_warn_frac"; default=0.03))
    global STABILITY_ABORT_ENABLED = Bool(safe_get(CFG, "advanced", "diagnostics", "stability_abort", "enabled"; default=false))
    global STABILITY_ABORT_RHO_MIN = Float32(safe_get(CFG, "advanced", "diagnostics", "stability_abort", "rho_min"; default=0.0))
    global STABILITY_ABORT_RHO_MAX = Float32(safe_get(CFG, "advanced", "diagnostics", "stability_abort", "rho_max"; default=0.0))
    global STABILITY_ABORT_U_MAX = Float32(safe_get(CFG, "advanced", "diagnostics", "stability_abort", "u_max"; default=0.0))
    global STABILITY_ABORT_MASS_REL = Float32(safe_get(CFG, "advanced", "diagnostics", "stability_abort", "mass_rel"; default=0.0))
    global STABILITY_ABORT_WAKE_SHEAR_OVERSPEED = Float32(safe_get(CFG, "advanced", "diagnostics", "stability_abort", "wake_shear_overspeed"; default=0.0))
    global STABILITY_ABORT_WAKE_SHEAR_FRAC = Float32(safe_get(CFG, "advanced", "diagnostics", "stability_abort", "wake_shear_frac"; default=0.0))
    global STABILITY_ABORT_BASE_PRESSURE_REL_MIN = Float32(safe_get(CFG, "advanced", "diagnostics", "stability_abort", "base_pressure_rel_min"; default=0.0))
    global STABILITY_ABORT_BASE_PRESSURE_REL_MAX = Float32(safe_get(CFG, "advanced", "diagnostics", "stability_abort", "base_pressure_rel_max"; default=0.0))
    TRANSONIC_WAKE_DIAG_X_MIN_CELLS < 0.0f0 &&
        error("advanced.diagnostics.transonic_wake.x_min_cells must be >= 0")
    TRANSONIC_WAKE_DIAG_MAX_X_DIAMETERS <= 0.0f0 &&
        error("advanced.diagnostics.transonic_wake.max_x_diameters must be > 0")
    TRANSONIC_WAKE_OVERSPEED_WARN_FRAC < 0.0f0 &&
        error("advanced.diagnostics.transonic_wake.overspeed_warn_frac must be >= 0")
    for (name, val) in (
        ("rho_min", STABILITY_ABORT_RHO_MIN),
        ("rho_max", STABILITY_ABORT_RHO_MAX),
        ("u_max", STABILITY_ABORT_U_MAX),
        ("mass_rel", STABILITY_ABORT_MASS_REL),
        ("wake_shear_overspeed", STABILITY_ABORT_WAKE_SHEAR_OVERSPEED),
        ("wake_shear_frac", STABILITY_ABORT_WAKE_SHEAR_FRAC),
        ("base_pressure_rel_min", STABILITY_ABORT_BASE_PRESSURE_REL_MIN),
        ("base_pressure_rel_max", STABILITY_ABORT_BASE_PRESSURE_REL_MAX),
    )
        val < 0.0f0 && error("advanced.diagnostics.stability_abort.$name must be >= 0")
    end
    if TRANSONIC_WAKE_DIAGNOSTICS
        @printf("[Diag] Transonic wake diagnostics ENABLED (x_min=%.1f cells, x_max=%.2fD, overspeed_warn=%.3f)\n",
                TRANSONIC_WAKE_DIAG_X_MIN_CELLS,
                TRANSONIC_WAKE_DIAG_MAX_X_DIAMETERS,
                TRANSONIC_WAKE_OVERSPEED_WARN_FRAC)
    end
    if STABILITY_ABORT_ENABLED
        @printf("[Diag] Stability abort ENABLED (rho_min=%.3f rho_max=%.3f u_max=%.3f mass_rel=%.3f wake_over=%.3f wake_frac=%.3f pbase=[%.3f, %.3f])\n",
                STABILITY_ABORT_RHO_MIN, STABILITY_ABORT_RHO_MAX,
                STABILITY_ABORT_U_MAX, STABILITY_ABORT_MASS_REL,
                STABILITY_ABORT_WAKE_SHEAR_OVERSPEED,
                STABILITY_ABORT_WAKE_SHEAR_FRAC,
                STABILITY_ABORT_BASE_PRESSURE_REL_MIN,
                STABILITY_ABORT_BASE_PRESSURE_REL_MAX)
    end

    # Pass 192a — Initial-condition selector and Riemann state. Defaults
    # reproduce the legacy single-uniform-state initialization (the
    # solver behavior is bit-identical when type == :uniform).
    local _ic_type_str = lowercase(String(safe_get(CFG, "advanced", "initial_condition", "type"; default="uniform")))
    _ic_type_str in ("uniform", "riemann_x", "gaussian_x", "gaussian_y", "gaussian_z", "spiegel_vortex") ||
        error("advanced.initial_condition.type must be \"uniform\", \"riemann_x\", \"gaussian_x/y/z\", or \"spiegel_vortex\" (got \"$_ic_type_str\")")
    global INITIAL_CONDITION_TYPE = _ic_type_str == "riemann_x" ? :riemann_x :
                                    _ic_type_str == "gaussian_x" ? :gaussian_x :
                                    _ic_type_str == "gaussian_y" ? :gaussian_y :
                                    _ic_type_str == "gaussian_z" ? :gaussian_z :
                                    _ic_type_str == "spiegel_vortex" ? :spiegel_vortex : :uniform
    local _ic_uniform_freestream = Bool(safe_get(CFG, "advanced", "initial_condition", "uniform", "use_freestream_velocity"; default=false))
    local _ic_uniform_ramp_start = Bool(safe_get(CFG, "advanced", "initial_condition", "uniform", "use_ramp_start_velocity"; default=false))
    (_ic_uniform_freestream && _ic_uniform_ramp_start) &&
        error("advanced.initial_condition.uniform cannot set both use_freestream_velocity and use_ramp_start_velocity")
    global IC_UNIFORM_RHO = Float64(safe_get(CFG, "advanced", "initial_condition", "uniform", "rho"; default=1.0))
    global IC_UNIFORM_UX = _ic_uniform_freestream ? Float64(U_TARGET) :
                           _ic_uniform_ramp_start ? Float64(U_TARGET) * Float64(RAMP_START_FRACTION) :
                           Float64(safe_get(CFG, "advanced", "initial_condition", "uniform", "ux"; default=0.0))
    global IC_UNIFORM_UY = Float64(safe_get(CFG, "advanced", "initial_condition", "uniform", "uy"; default=0.0))
    global IC_UNIFORM_UZ = Float64(safe_get(CFG, "advanced", "initial_condition", "uniform", "uz"; default=0.0))
    global IC_UNIFORM_T = Float64(safe_get(CFG, "advanced", "initial_condition", "uniform", "T"; default=1.0))
    if INITIAL_CONDITION_TYPE === :uniform && _ic_uniform_freestream
        @printf("[Init]    Initial condition: uniform freestream  rho=%.4f u=(%.6f,%.6f,%.6f) T=%.4f\n",
                IC_UNIFORM_RHO, IC_UNIFORM_UX, IC_UNIFORM_UY, IC_UNIFORM_UZ, IC_UNIFORM_T)
    elseif INITIAL_CONDITION_TYPE === :uniform && _ic_uniform_ramp_start
        @printf("[Init]    Initial condition: uniform ramp-start  rho=%.4f u=(%.6f,%.6f,%.6f) T=%.4f\n",
                IC_UNIFORM_RHO, IC_UNIFORM_UX, IC_UNIFORM_UY, IC_UNIFORM_UZ, IC_UNIFORM_T)
    end
    if INITIAL_CONDITION_TYPE in (:gaussian_x, :gaussian_y, :gaussian_z)
        global IC_GAUSSIAN_AXIS = INITIAL_CONDITION_TYPE === :gaussian_y ? 2 :
                                  INITIAL_CONDITION_TYPE === :gaussian_z ? 3 : 1
        global IC_GAUSSIAN_CENTER_X = Float64(safe_get(CFG, "advanced", "initial_condition", "gaussian", "center_x"; default=0.25))
        global IC_GAUSSIAN_CENTER_Y = Float64(safe_get(CFG, "advanced", "initial_condition", "gaussian", "center_y"; default=IC_GAUSSIAN_CENTER_X))
        global IC_GAUSSIAN_CENTER_Z = Float64(safe_get(CFG, "advanced", "initial_condition", "gaussian", "center_z"; default=IC_GAUSSIAN_CENTER_X))
        global IC_GAUSSIAN_SIGMA_X = Float64(safe_get(CFG, "advanced", "initial_condition", "gaussian", "sigma_x"; default=0.04))
        global IC_GAUSSIAN_SIGMA_Y = Float64(safe_get(CFG, "advanced", "initial_condition", "gaussian", "sigma_y"; default=IC_GAUSSIAN_SIGMA_X))
        global IC_GAUSSIAN_SIGMA_Z = Float64(safe_get(CFG, "advanced", "initial_condition", "gaussian", "sigma_z"; default=IC_GAUSSIAN_SIGMA_X))
        global IC_GAUSSIAN_RHO_AMPLITUDE = Float64(safe_get(CFG, "advanced", "initial_condition", "gaussian", "rho_amplitude"; default=1.0e-3))
        global IC_GAUSSIAN_UX_AMPLITUDE = Float64(safe_get(CFG, "advanced", "initial_condition", "gaussian", "ux_amplitude"; default=0.0))
        global IC_GAUSSIAN_UY_AMPLITUDE = Float64(safe_get(CFG, "advanced", "initial_condition", "gaussian", "uy_amplitude"; default=0.0))
        global IC_GAUSSIAN_UZ_AMPLITUDE = Float64(safe_get(CFG, "advanced", "initial_condition", "gaussian", "uz_amplitude"; default=0.0))
        global IC_GAUSSIAN_T_AMPLITUDE = Float64(safe_get(CFG, "advanced", "initial_condition", "gaussian", "T_amplitude"; default=0.0))
        local _gaussian_profile = lowercase(String(safe_get(CFG, "advanced", "initial_condition", "gaussian", "profile"; default="axis")))
        _gaussian_profile in ("axis", "ellipsoid") ||
            error("advanced.initial_condition.gaussian.profile must be \"axis\" or \"ellipsoid\" (got \"$_gaussian_profile\")")
        global IC_GAUSSIAN_ELLIPSOIDAL = _gaussian_profile == "ellipsoid"
        IC_GAUSSIAN_SIGMA_X <= 0.0 && error("advanced.initial_condition.gaussian.sigma_x must be > 0")
        IC_GAUSSIAN_SIGMA_Y <= 0.0 && error("advanced.initial_condition.gaussian.sigma_y must be > 0")
        IC_GAUSSIAN_SIGMA_Z <= 0.0 && error("advanced.initial_condition.gaussian.sigma_z must be > 0")
        center_axis = IC_GAUSSIAN_AXIS == 2 ? IC_GAUSSIAN_CENTER_Y :
                      IC_GAUSSIAN_AXIS == 3 ? IC_GAUSSIAN_CENTER_Z : IC_GAUSSIAN_CENTER_X
        sigma_axis = IC_GAUSSIAN_AXIS == 2 ? IC_GAUSSIAN_SIGMA_Y :
                     IC_GAUSSIAN_AXIS == 3 ? IC_GAUSSIAN_SIGMA_Z : IC_GAUSSIAN_SIGMA_X
        @printf("[Init]    Initial condition: %s/%s  center=%.6f m sigma=%.6f m  d_rho=%.3e d_u=(%.3e,%.3e,%.3e) dT=%.3e\n",
                _ic_type_str, _gaussian_profile, center_axis, sigma_axis,
                IC_GAUSSIAN_RHO_AMPLITUDE,
                IC_GAUSSIAN_UX_AMPLITUDE, IC_GAUSSIAN_UY_AMPLITUDE, IC_GAUSSIAN_UZ_AMPLITUDE,
                IC_GAUSSIAN_T_AMPLITUDE)
    end
    if INITIAL_CONDITION_TYPE === :riemann_x
        global IC_RIEMANN_SPLIT_X    = Float64(safe_get(CFG, "advanced", "initial_condition", "riemann", "split_x"; default=0.5))
        global IC_RIEMANN_LEFT_RHO   = Float64(safe_get(CFG, "advanced", "initial_condition", "riemann", "left", "rho"; default=1.0))
        global IC_RIEMANN_LEFT_UX    = Float64(safe_get(CFG, "advanced", "initial_condition", "riemann", "left", "ux";  default=0.0))
        global IC_RIEMANN_LEFT_UY    = Float64(safe_get(CFG, "advanced", "initial_condition", "riemann", "left", "uy";  default=0.0))
        global IC_RIEMANN_LEFT_UZ    = Float64(safe_get(CFG, "advanced", "initial_condition", "riemann", "left", "uz";  default=0.0))
        global IC_RIEMANN_LEFT_T     = Float64(safe_get(CFG, "advanced", "initial_condition", "riemann", "left", "T";   default=1.0))
        global IC_RIEMANN_RIGHT_RHO  = Float64(safe_get(CFG, "advanced", "initial_condition", "riemann", "right", "rho"; default=0.125))
        global IC_RIEMANN_RIGHT_UX   = Float64(safe_get(CFG, "advanced", "initial_condition", "riemann", "right", "ux";  default=0.0))
        global IC_RIEMANN_RIGHT_UY   = Float64(safe_get(CFG, "advanced", "initial_condition", "riemann", "right", "uy";  default=0.0))
        global IC_RIEMANN_RIGHT_UZ   = Float64(safe_get(CFG, "advanced", "initial_condition", "riemann", "right", "uz";  default=0.0))
        global IC_RIEMANN_RIGHT_T    = Float64(safe_get(CFG, "advanced", "initial_condition", "riemann", "right", "T";   default=0.8))
        @printf("[Init]    Initial condition: riemann_x  split_x=%.6f m\n", IC_RIEMANN_SPLIT_X)
        @printf("[Init]      Left:  rho=%.4f u=(%.4f,%.4f,%.4f) T=%.4f\n",
                IC_RIEMANN_LEFT_RHO, IC_RIEMANN_LEFT_UX, IC_RIEMANN_LEFT_UY, IC_RIEMANN_LEFT_UZ, IC_RIEMANN_LEFT_T)
        @printf("[Init]      Right: rho=%.4f u=(%.4f,%.4f,%.4f) T=%.4f\n",
                IC_RIEMANN_RIGHT_RHO, IC_RIEMANN_RIGHT_UX, IC_RIEMANN_RIGHT_UY, IC_RIEMANN_RIGHT_UZ, IC_RIEMANN_RIGHT_T)
    end
    if INITIAL_CONDITION_TYPE === :spiegel_vortex
        global IC_SPIEGEL_CENTER_X  = Float64(safe_get(CFG, "advanced", "initial_condition", "spiegel", "center_x"; default=0.5))
        global IC_SPIEGEL_CENTER_Y  = Float64(safe_get(CFG, "advanced", "initial_condition", "spiegel", "center_y"; default=0.0))
        global IC_SPIEGEL_CENTER_Z  = Float64(safe_get(CFG, "advanced", "initial_condition", "spiegel", "center_z"; default=0.0))
        global IC_SPIEGEL_RADIUS    = Float64(safe_get(CFG, "advanced", "initial_condition", "spiegel", "radius"; default=0.1))
        global IC_SPIEGEL_AMPLITUDE = Float64(safe_get(CFG, "advanced", "initial_condition", "spiegel", "amplitude"; default=0.05))
        IC_SPIEGEL_RADIUS > 0.0 || error("advanced.initial_condition.spiegel.radius must be > 0")
        @printf("[Init]    Initial condition: spiegel_vortex  center=(%.4f,%.4f,%.4f) m R=%.4f m A=%.4f\n",
                IC_SPIEGEL_CENTER_X, IC_SPIEGEL_CENTER_Y, IC_SPIEGEL_CENTER_Z,
                IC_SPIEGEL_RADIUS, IC_SPIEGEL_AMPLITUDE)
    end
    global FORCE_OUTPUT_FREQ = FORCE_OUTPUT_FREQ_CONFIG == 0 ? OUTPUT_FREQ : FORCE_OUTPUT_FREQ_CONFIG
    
    global GPU_ASYNC_DEPTH = Int(safe_get(CFG, "advanced", "gpu", "async_depth"; default=8))
    global USE_STREAMS = safe_get(CFG, "advanced", "gpu", "use_streams"; default=true)
    # (D3Q19 workgroup + AMR-interior-split config retired 2026-06-24 with the D3Q19 solver.)
    global PREFETCH_NEIGHBORS = safe_get(CFG, "advanced", "gpu", "prefetch_neighbors"; default=true)
    global MULTI_GPU_ENABLED = safe_get(CFG, "advanced", "gpu", "multi_gpu"; default=false)
    global GPU_SPLIT_LEVEL = Int(safe_get(CFG, "advanced", "gpu", "split_level"; default=2))
    
    if isdefined(Main, :DOMAIN_PARAMS)
        DOMAIN_PARAMS.initialized = false
    end

    # ══ PERIODIC-Y AMR (branch lbm-periodic-amr, 27 Oct 2026) — BEGIN parse block ══
    # All periodic-Y AMR keys are parsed and checked here, in one block.
    global PERIODIC_Y_AMR = Bool(safe_get(CFG, "advanced", "boundary", "periodic_y_amr"; default=false))
    if PERIODIC_Y_AMR && !SPAN_PERIODIC_Y
        error("advanced.boundary.periodic_y_amr: true requires advanced.boundary.periodic_y: true.")
    end
    PERIODIC_Y_AMR &&
        println("[Boundary] Periodic-Y AMR ENABLED: every refinement level wraps in Y " *
                "(neighbour tables, prolongation/restriction stencils, refinement closures)")
    # The kernels' periodic-Y wrap is a COMPILE-TIME gate (_PERIODIC_Y_AMR,
    # env LUDWIG_PERIODIC_Y_AMR, set by preset_bootstrap.jl from the queued
    # cases before the kernels are compiled). A case must agree with it: a
    # periodic case on unwrapped kernels would fall back to edge treatment at the
    # seam; a non-periodic case on wrapped kernels would wrap a physical boundary.
    if PERIODIC_Y_AMR != _PERIODIC_Y_AMR
        error("advanced.boundary.periodic_y_amr=$(PERIODIC_Y_AMR) but the solver kernels were compiled with " *
              "LUDWIG_PERIODIC_Y_AMR=$(_PERIODIC_Y_AMR ? "1" : "0"). The gate is set automatically from the queued " *
              "cases (src/preset_bootstrap.jl); do not set the env var by hand, and do not queue periodic_y_amr " *
              "and non-periodic_y_amr cases in the same run.")
    end
    global SPAN_UNIFORM_LEVELS = Bool(safe_get(CFG, "advanced", "refinement", "span_uniform_levels"; default=false))
    if SPAN_UNIFORM_LEVELS && !PERIODIC_Y_AMR
        error("advanced.refinement.span_uniform_levels: true requires advanced.boundary.periodic_y_amr: true.")
    end
    global PERIODIC_Y_BODY_SHIFT = Float64(safe_get(CFG, "advanced", "domain", "periodic_y_body_shift"; default=0.0))
    if PERIODIC_Y_BODY_SHIFT != 0.0 && !PERIODIC_Y_AMR
        error("advanced.domain.periodic_y_body_shift requires advanced.boundary.periodic_y_amr: true.")
    end
    global RAW_STATE_DUMP_FREQ = Int(safe_get(CFG, "advanced", "diagnostics", "raw_state_dump_freq"; default=0))
    RAW_STATE_DUMP_FREQ < 0 && error("advanced.diagnostics.raw_state_dump_freq must be >= 0")
    # Refuse configurations periodic-Y AMR does not support. Each of these either
    # imposes a physical Y boundary (half domain / ground / specular-mirror ghosts)
    # or is an experimental AMR coupling whose kernels were not audited for the
    # periodic seam. Refuse loudly rather than run with an unverified seam.
    if PERIODIC_Y_AMR
        local _pyamr_bad = String[]
        HALF_Y_DOMAIN && push!(_pyamr_bad, "refinement.symmetric_analysis (half-domain / ground: a physical y=0 boundary)")
        MULTI_GPU_ENABLED && push!(_pyamr_bad, "gpu.multi_gpu")
        _AMR_MIRROR_GHOST_INTERP && push!(_pyamr_bad, "env LUDWIG_AMR_MIRROR_GHOST=1 (specular y/z ghosts)")
        AMR_INTERFACE_SHELL_SYNC && push!(_pyamr_bad, "refinement.interface_shell_sync")
        AMR_SHELL_HIGHPASS_DAMPER && push!(_pyamr_bad, "refinement.interface_shell_highpass_damper")
        AMR_RUNTIME_FACE_BUDGET_REFLUX && push!(_pyamr_bad, "refinement.runtime_face_budget_reflux")
        AMR_GPU_FACE_BUDGET_REFLUX && push!(_pyamr_bad, "refinement.gpu_face_budget_reflux")
        AMR_POPULATION_STREAM_REGISTER && push!(_pyamr_bad, "refinement.population_stream_register*")
        AMR_HALO_FULLCELL_RECOMPOSITION && push!(_pyamr_bad, "refinement.halo_fullcell_recomposition")
        GENDRE_NORMAL_GRADIENT && push!(_pyamr_bad, "refinement.gendre_normal_gradient")
        LOCAL_LATTICE_SHIFT_ENABLED && push!(_pyamr_bad, "numerics.local_lattice_shift")
        BOUZIDI_Q_GEOMETRY == :sphere_analytic &&
            push!(_pyamr_bad, "boundary.q_geometry: sphere_analytic (centre/radius from the bounding box; use stl_ray)")
        AMR_ENFORCE_Y_SYMMETRY &&
            push!(_pyamr_bad, "refinement.enforce_y_symmetry (a mid-plane mirror is not a symmetry of a periodic axis; " *
                              "use refinement.span_uniform_levels)")
        if !isempty(_pyamr_bad)
            error("advanced.boundary.periodic_y_amr: true is not supported together with: " *
                  join(_pyamr_bad, "; ") * ". Turn these off for a periodic-Y AMR run.")
        end
        DOMAIN_SPAN_OVERRIDE > 0.0 ||
            @info "[Boundary] periodic_y_amr without advanced.domain.span_override: the Y period is the STL Y extent " *
                  "+ 2·lateral, snapped up to whole coarsest blocks (printed at the domain build)."
        SPAN_UNIFORM_LEVELS &&
            println("[Boundary] span_uniform_levels: every refined level is extruded over the whole Y period")
    end
    # ══ PERIODIC-Y AMR — END parse block ══

    # ══ Z-FACE WALLS (branch lbm-zwalls, 27 Oct 2026) — parse block ══
    # advanced.boundary.z_walls: "slip" | "no_slip"; absent = the legacy z faces
    # (mirror + transverse sponge), byte-identical. Set per case (every load).
    if config_has_path(CFG, "advanced", "boundary", "z_walls")
        local _zw_raw = CFG["advanced"]["boundary"]["z_walls"]
        (_zw_raw isa AbstractString && _zw_raw in ("slip", "no_slip")) ||
            error("advanced.boundary.z_walls must be \"slip\" or \"no_slip\" (got $(repr(_zw_raw))); " *
                  "leave the key out for the legacy z faces (mirror + transverse sponge).")
        global Z_WALLS_MODE = Symbol(_zw_raw)
    else
        global Z_WALLS_MODE = :legacy
    end
    global Z_WALL_BC = Z_WALLS_MODE === :no_slip ? Int32(2) : Z_WALLS_MODE === :slip ? Int32(1) : Int32(0)
    if Z_WALLS_MODE !== :legacy
        local _zw_bad = String[]
        (LATTICE_TYPE == :d3q27 && COLLISION_OPERATOR == :cumulant) ||
            push!(_zw_bad, "lattice $(LATTICE_TYPE) / collision_operator $(COLLISION_OPERATOR) (implemented in the " *
                           "D3Q27 cumulant kernel only; the D3Q39 entropic kernel takes the argument as an unused stub)")
        DDF_ENABLED &&
            push!(_zw_bad, "the thermal DDF (compressibility_correction / compressible_mode: its kernel keeps mirror z faces)")
        TRANSVERSE_FREESTREAM_BOUNDARY == true &&
            push!(_zw_bad, "boundary.transverse_freestream (a far-field blend on the same faces)")
        HALF_Y_DOMAIN &&
            push!(_zw_bad, "refinement.symmetric_analysis (half domain / ground: the y=0 plane's edges with the z faces are not defined for z_walls)")
        (Z_WALLS_MODE === :no_slip && _AMR_MIRROR_GHOST_INTERP) &&
            push!(_zw_bad, "env LUDWIG_AMR_MIRROR_GHOST=1 (its specular z ghosts keep the tangential velocity: a slip wall)")
        if !isempty(_zw_bad)
            error("advanced.boundary.z_walls: \"$(Z_WALLS_MODE)\" is not supported together with: " *
                  join(_zw_bad, "; ") * ".")
        end
    end
    # ══ Z-FACE WALLS — END parse block ══

    # ══ LEVEL VISCOSITY (branch lbm-taufix, 28 Oct 2026) — parse block ══
    if config_has_path(CFG, "advanced", "refinement", "level_viscosity")
        local _lv_raw = CFG["advanced"]["refinement"]["level_viscosity"]
        (_lv_raw isa AbstractString && _lv_raw in ("legacy", "physical")) ||
            error("advanced.refinement.level_viscosity must be \"legacy\" or \"physical\" (got $(repr(_lv_raw))).")
        global LEVEL_VISCOSITY_MODE = Symbol(_lv_raw)
    else
        global LEVEL_VISCOSITY_MODE = :legacy
    end
    global LEVEL_VISC_PHYSICAL = LEVEL_VISCOSITY_MODE === :physical ? Int32(1) : Int32(0)
    if LEVEL_VISCOSITY_MODE === :physical
        local _lv_bad = String[]
        (LATTICE_TYPE == :d3q27 && COLLISION_OPERATOR == :cumulant) ||
            push!(_lv_bad, "lattice $(LATTICE_TYPE) / collision_operator $(COLLISION_OPERATOR) (the level-coupling " *
                           "rescaling is adapted in the D3Q27 cumulant path only)")
        DDF_ENABLED &&
            push!(_lv_bad, "the thermal DDF (tau_g keeps the legacy level rule; its coupling is not adapted)")
        MULTI_GPU_ENABLED && push!(_lv_bad, "gpu.multi_gpu (not verified)")
        if !isempty(_lv_bad)
            error("advanced.refinement.level_viscosity: \"physical\" is not supported together with: " *
                  join(_lv_bad, "; ") * ".")
        end
    end
    # ══ LEVEL VISCOSITY — END parse block ══

    println("[Init] ═══════════════════════════════════════════════════════")
    println("[Init] SOLVER SETTINGS:")
    println("[Init]    Half-domain mode: $(symmetry_mode_summary())")
    if GROUND_ANALYSIS
        println("[Init]       Source STL up-axis: $(uppercase(string(GROUND_UP_AXIS)))")
    end
    if ALPHA_DEG != 0.0 || BETA_DEG != 0.0
        @printf("[Init]    Model rotation: α=%.2f° (around Y), β=%.2f° (around Z)\n", ALPHA_DEG, BETA_DEG)
    end
    println("[Init]    Lattice: $LATTICE_TYPE ($N_DIRS directions)")
    println("[Init]    Collision operator: $COLLISION_OPERATOR")
    @printf("[Init]    Background ν_sgs: %.6f → τ_eff_min ≈ %.4f\n", NU_SGS_BACKGROUND, 0.5 + 3*NU_SGS_BACKGROUND)
    println("[Init]    Sponge f-blending: $SPONGE_BLEND_DISTRIBUTIONS")
    println("[Init]    Inlet ramp profile: $RAMP_PROFILE")
    @printf("[Init]    Inlet ramp start fraction: %.3f\n", RAMP_START_FRACTION)
    println("[Init]    Temporal interpolation: $TEMPORAL_INTERPOLATION")
    println("[Init]    Interface macro temporal interpolation: $INTERFACE_MACRO_TEMPORAL_INTERPOLATION")
    println("[Init]    Interface temporal weight bias: $INTERFACE_TEMPORAL_WEIGHT_BIAS")
    println("[Init]    Interface temporal bias rise U/Utarget: $INTERFACE_TEMPORAL_WEIGHT_BIAS_RISE_U_START -> $INTERFACE_TEMPORAL_WEIGHT_BIAS_RISE_U_END")
    println("[Init]    Interface temporal bias decay U/Utarget: $INTERFACE_TEMPORAL_WEIGHT_BIAS_DECAY_U_START -> $INTERFACE_TEMPORAL_WEIGHT_BIAS_DECAY_U_END")
    println("[Init]    Fine-to-parent restriction: $RESTRICT_FINE_TO_PARENT")
    println("[Init]    AMR interface shell sync: $AMR_INTERFACE_SHELL_SYNC")
    println("[Init]    AMR shell eq-shift (post-update consistency): $AMR_SHELL_EQ_SHIFT")
    println("[Init]    AMR shell well-balanced rebuild: $AMR_SHELL_WELL_BALANCED_REBUILD")
    println("[Init]    AMR shell sync includes x-links: $AMR_SHELL_SYNC_INCLUDE_X")
    @printf("[Init]    AMR shell x-population blend: %.3f\n", AMR_SHELL_POPULATION_BLEND_X)
    println("[Init]    AMR shell high-pass damper: $AMR_SHELL_HIGHPASS_DAMPER")
    if AMR_SHELL_HIGHPASS_DAMPER
        @printf("[Init]    AMR shell high-pass damper alpha/cap_abs/cap_rel/depth/transverse_only: %.3f / %.2e / %.2e / %d / %s\n",
                AMR_SHELL_HIGHPASS_DAMPER_ALPHA,
                AMR_SHELL_HIGHPASS_DAMPER_CAP_ABS,
                AMR_SHELL_HIGHPASS_DAMPER_CAP_REL,
                AMR_SHELL_HIGHPASS_DAMPER_DEPTH,
                string(AMR_SHELL_HIGHPASS_DAMPER_TRANSVERSE_ONLY))
        @printf("[Init]    AMR shell density damper rho_alpha/drho_cap_abs/drho_cap_rel: %.3f / %.2e / %.2e\n",
                AMR_SHELL_HIGHPASS_DAMPER_RHO_ALPHA,
                AMR_SHELL_HIGHPASS_DAMPER_DRHO_CAP_ABS,
                AMR_SHELL_HIGHPASS_DAMPER_DRHO_CAP_REL)
    end
    @printf("[Init]    AMR halo full-cell recomposition: %s, macro/pop alpha=%.3f/%.3f\n",
            AMR_HALO_FULLCELL_RECOMPOSITION,
            AMR_HALO_FULLCELL_MACRO_ALPHA,
            AMR_HALO_FULLCELL_POP_ALPHA)
    println("[Init]    AMR prolongation monotone velocity: $AMR_PROLONGATION_MONOTONE_VELOCITY")
    println("[Init]    AMR halo temporal macro closure: $AMR_HALO_TEMPORAL_MACRO_CLOSURE")
    println("[Init]    AMR halo equilibrium projection: $AMR_HALO_EQ_PROJECTION")
    println("[Init]    AMR halo constrained kinetic completion: $AMR_HALO_CONSTRAINED_COMPLETION")
    println("[Init]    AMR restriction f_neq filter (Lagrava box, default off): $AMR_RESTRICTION_PI_NEQ_FILTER")
    println("[Init]    AMR restriction STRICT-CONSERVATIVE D3Q19 4-moment (mass+3mom exact, default off): $AMR_RESTRICTION_CONSERVATIVE_D3Q19")
    println("[Init]    AMR restriction STRICT-CONSERVATIVE D3Q27 4-moment (mass+3mom exact, default off): $AMR_RESTRICTION_CONSERVATIVE_D3Q27")
    println("[Init]    AMR restriction moment-correction passes: $AMR_RESTRICTION_CORRECTION_PASSES")
    println("[Init]    AMR restriction conservative f_eq (Tier 1 #1e, exact ρ+j after Newton): $AMR_RESTRICTION_CONSERVATIVE_FEQ")
    println("[Init]    AMR restriction conservative f_eq (Tier 1 #1e): $AMR_RESTRICTION_CONSERVATIVE_FEQ")
    @printf("[Init]    HRR-hybrid stress (Astoul/Schukmann): %s, σ_HRR=%.4f\n", HRR_HYBRID_ENABLED, HRR_SIGMA)
    if !isempty(HRR_SIGMA_PER_LEVEL)
        # Pass 261/262: per-level σ_HRR override. List one σ per level (1-indexed).
        # Missing levels (i > length) fall back to the global HRR_SIGMA above.
        sigma_str = join((@sprintf("L%d=%.4f", i, σ) for (i, σ) in enumerate(HRR_SIGMA_PER_LEVEL)), ", ")
        @printf("[Init]    HRR-σ per level override: %s (missing levels use σ_HRR=%.4f)\n", sigma_str, HRR_SIGMA)
    end
    if !isempty(HRR_SIGMA_PER_LEVEL)
        @printf("[Init]    HRR σ per level (pass 261 override): %s\n",
                join((@sprintf("L%d=%.3f", i, σ) for (i, σ) in enumerate(HRR_SIGMA_PER_LEVEL)), ", "))
    end
    println("[Init]    AMR partial frontier sync: $AMR_PARTIAL_FRONTIER_SYNC")
    println("[Init]    AMR post-restrict sponge: $AMR_POST_RESTRICT_SPONGE")
    println("[Init]    AMR interface regularization depth: $AMR_REGULARIZATION_DEPTH")
    println("[Init]    Interface entropic h-scale: $INTERFACE_ENTROPIC_H_SCALE")
    println("[Init]    Top/bottom (z-face) walls: $(z_walls_summary())")
    println("[Init]    Level viscosity: ", LEVEL_VISCOSITY_MODE === :physical ?
            "physical (advanced.refinement.level_viscosity): tau_l - 1/2 = (tau_f - 1/2)/2^(L-l), floored at TAU_MIN" :
            "legacy: tau_l - 1/2 = (tau_f - 1/2)·2^(L-l) (each coarser level 4× the physical viscosity of the next finer)")
    println("[Init]    Transverse freestream boundary: $TRANSVERSE_FREESTREAM_BOUNDARY")
    println("[Init]    Transverse freestream blend: $TRANSVERSE_FREESTREAM_BLEND")
    println("[Init]    Outlet freestream boundary: $OUTLET_FREESTREAM_BOUNDARY")
    println("[Init]    Outlet boundary mode: $OUTLET_BOUNDARY_MODE")
    println("[Init]    Outlet freestream blend: $OUTLET_FREESTREAM_BLEND")
    println("[Init]    Inlet density mode: $(INLET_DENSITY_MODE == Int32(1) ? "local" : "freestream")")
    println("[Init]    Wall model enabled: $WALL_MODEL_ENABLED")
    if WALL_MODEL_ENABLED
        @printf("[Init]    Wall model type: %s, max distance: %.3f cells\n",
                string(WALL_MODEL_TYPE), WALL_MODEL_MAX_DISTANCE_CELLS)
    end
    if length(STL_FILENAMES) > 1
        println("[Init]    STL files: $(length(STL_FILENAMES)) objects")
        for (si, fn) in enumerate(STL_FILENAMES)
            println("[Init]       [$si] $fn")
        end
    end
    if !SUBDIVISION_ENABLED
        println("[Init]    STL subdivision: OFF (subdivision: false) — raw triangles used as-is")
    elseif MINIMUM_FACET_SIZE > 0.0
        @printf("[Init]    STL subdivision: ON, minimum facet size: %.6f m\n", MINIMUM_FACET_SIZE)
    else
        println("[Init]    STL subdivision: OFF (minimum_facet_size = 0) — raw triangles used as-is")
    end
    if COLLISION_OPERATOR == :cumulant
        println("[Init]    Cumulant settings:")
        @printf("[Init]       ω_bulk=%.3f, ω₃=%.3f, ω₄=%.3f\n", CUMULANT_OMEGA_BULK, CUMULANT_OMEGA_3, CUMULANT_OMEGA_4)
        println("[Init]       Adaptive ω₄: $CUMULANT_ADAPTIVE_OMEGA4" * (CUMULANT_ADAPTIVE_OMEGA4 ? @sprintf(" (Λ=%.4f)", CUMULANT_LAMBDA_PARAM) : ""))
        println("[Init]       Limiter: $CUMULANT_LIMITER")
    elseif COLLISION_OPERATOR == :entropic
        println("[Init]    Entropic KBC settings:")
        println("[Init]       α solver: $ENTROPIC_ALPHA_SOLVER")
        println("[Init]       Max iterations: $ENTROPIC_ALPHA_MAX_ITER")
        if ENTROPIC_ALPHA_SOLVER == :newton
            @printf("[Init]       Tolerance: %.1e\n", ENTROPIC_ALPHA_TOLERANCE)
        end
        if ENTROPIC_SIGMA > 0.0f0
            @printf("[Init]       Sigma (h-damping): %.3f\n", ENTROPIC_SIGMA)
        end
    end
    println("[Init]    Compressibility correction: $COMPRESSIBILITY_CORRECTION")
    if COMPRESSIBLE_MODE
        println("[Init]    *** COMPRESSIBLE MODE: Ma_lattice = Ma_physical ***")
    end
    if max(abs(LATTICE_SHIFT_X), abs(LATTICE_SHIFT_Y), abs(LATTICE_SHIFT_Z)) > 0
        @printf("[Init]    Shifted lattice: U = (%d, %d, %d), max |c'| = %d\n",
                LATTICE_SHIFT_X, LATTICE_SHIFT_Y, LATTICE_SHIFT_Z,
                3 + max(abs(LATTICE_SHIFT_X), abs(LATTICE_SHIFT_Y), abs(LATTICE_SHIFT_Z)))
    end
    local _cs2_lat = LATTICE_TYPE == :d3q39 ? 2.0/3.0 : 1.0/3.0
    local _Ma_lattice = Float64(U_TARGET) / sqrt(_cs2_lat)
    local _Ma_physical = FLOW_VELOCITY / SPEED_OF_SOUND_PHYSICAL
    @printf("[Init]    Mach number: Ma_physical = %.4f, Ma_lattice = %.4f (cs²=%.4f)\n", _Ma_physical, _Ma_lattice, _cs2_lat)
    if COMPRESSIBLE_MODE && _Ma_lattice > 1.0
        println("[Init]    SUPERSONIC: bow shocks expected in lattice space")
    end
    if _Ma_lattice > 0.3 && !COMPRESSIBILITY_CORRECTION && !COMPRESSIBLE_MODE && LATTICE_TYPE == :d3q27
        println("[Init]    ⚠ WARNING: Ma_lattice > 0.3 without compressibility correction!")
    end
    if LATTICE_TYPE == :d3q39
        println("[Init]    Numerical equilibrium: ENABLED (13-moment Newton-Raphson)")
    end
    if DDF_ENABLED
        println("[Init]       Thermal DDF: ENABLED (auto)")
        @printf("[Init]          Prandtl = %.3f\n", DDF_PRANDTL)
        @printf("[Init]          T_inlet = %.3f, T_wall = %.3f, T_initial = %.3f\n", DDF_T_INLET, DDF_T_WALL, DDF_T_INITIAL)
        println("[Init]          Wall BC: $DDF_WALL_BC")
        println("[Init]          Feedback to flow: $DDF_FEEDBACK_TO_FLOW")
        @printf("[Init]          Feedback relaxation: %.3f\n", DDF_FEEDBACK_RELAXATION)
    end
    if NET_FORCE_EXPORT_ENABLED
        println("[Init]    Net force export: ENABLED (two-sided probing)")
    end
    @printf("[Init]    Refinement halo: upstream=%d, downstream=%d, span=%d, up=%d, down=%d blocks\n",
            REFINEMENT_MARGIN_UPSTREAM, REFINEMENT_MARGIN_DOWNSTREAM,
            REFINEMENT_MARGIN_SPAN, REFINEMENT_MARGIN_UP, REFINEMENT_MARGIN_DOWN)
    AMR_ENFORCE_Z_SYMMETRY && println("[Init]    AMR z-mirror topology closure: ENABLED")
    AMR_ENFORCE_Y_SYMMETRY && println("[Init]    AMR y-mirror topology closure: ENABLED")
    println("[Init] ═══════════════════════════════════════════════════════")
end
