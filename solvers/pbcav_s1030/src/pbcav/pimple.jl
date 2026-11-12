# =====================================================================================================================
# pimple.jl -- the step of plan 4.7 in its psi = 0 limit (design A's algorithm, laminar), with rollback, typed rejections
# and regrowth, and the march. Stage P0b, 14 Oct 2026.
#
# THE STEP n -> n+1 (backward Euler; level 1 the iterate, level 2 = n, never written during the step):
#   0. the iterate starts from u, v, p and F extrapolated linearly from n and n-1, x^(0) = x^n + (dt/dt_prev)(x^n - x^{n-1})
#      (`extrapolate`, default on from the second step; rho and q start at level n: the mass is transported in 1.5).
#   1. outer iteration m = 1 .. N_max:
#      1.1 (SST: stage P2)
#      1.2 (phase predictor: single phase in P0b; the mixture mass is transported in 1.5 with the final flux)
#      1.3 the momentum predictor: assembled at u^(m-1), p^(m-1), F^(m-1); its unrelaxed residual is measured there; relaxed
#          by lambda_u except on the final outer iteration; `mom_sweeps` alternating coloured line sweeps (j then i);
#      1.4 `n_correctors` PISO correctors: H^s, F* and D (rhie_chow_flux!), the pressure equation, the Schaffer V(1,1)
#          hierarchy updated at the first corrector by the policy of `mg_update!` (review F5, plan 24: by default built on
#          the host once per march with its numeric plan, then refreshed on the backend when the pressure diagonal has
#          moved by more than mg_diag_delta), FCG to a relative forcing tolerance
#          (`p_tol_inner`) -- the last corrector of the final outer iteration to the ABSOLUTE tolerance that bounds
#          |Theta| by theta_tol/10 in every cell -- and the conservative corrections;
#      1.5 the mixture mass transported in flux form with the final flux; Theta = rho/rho_l - 1;
#      1.6 stop when every absolute criterion holds on the iterate (checked at the start of the next outer iteration):
#          momentum residual <= mom_tol (units rho U^2 / L, increment form: assemble_momentum.jl), max |Theta| <= theta_tol,
#          every Krylov solve converged; then one final outer iteration without relaxation (OpenFOAM's PIMPLE) closes the
#          step, and the FINAL ITERATE is checked again (review F6, 14 Oct): its momentum residual (re-assembled there) and
#          its |Theta| must meet the same tolerances (a miss sends the loop on from that iterate, and the next final
#          iteration waits until R has fallen a further decade below the last trigger, within N_max), and the
#          continuity-error term is measured on it.
#   2. commit, or reject with a typed reason (REJECTION_REASONS): the iterate is restored from level n bitwise and dt halved
#      by the march; after `regrow_after` consecutive accepted steps dt grows by `regrow_factor` per accepted step up to the
#      configured dt. :outer_predicted when the outer residual's contraction over iterations 3-6 predicts no convergence
#      within N_max (checked once, at the start of iteration 7); :linear_failed when an FCG solve misses its tolerance in
#      p_maxit iterations; :nonfinite; :bounds when (rho - q)/rho_l < -bound_tol in any cell of the final iterate;
#      :outer_stalled (review F4, 14 Oct) when a step that would be rejected as :outer_not_converged or :outer_predicted has
#      its last three outer residuals R all within a decade of the tolerances (0.1 <= R <= 10) and no new low among them
#      (their smallest >= 0.9 times the smallest of the three before): the tolerances sit at a Float64 floor
#      (corrected 14 Oct, re-review R3: in a transient the momentum floor is about eps rho |u|/dt, the time term a^t (u - u^n)
#      resolved to one ulp of u, so halving dt RAISES it; it vanishes only at u = u^n, a steady state; the Theta floor
#      scales with the cell Courant number). The march halves dt once
#      on it and refuses, with that reason, on a second consecutive one; it also refuses after `max_halvings` consecutive
#      rejections of any kind instead of halving dt forty times.
#   3. admission: the ledgers (ledger.jl) after each committed step.
# =====================================================================================================================

"""The typed rejection reasons of plan 4.7 step 2; under SST (W1b) also :turb_linear_failed (a k-omega line solve missed its
goal after its continuation; :linear_failed stays the flow's FGMRES or FCG) and :turb_not_converged (turb_coupling = :step
with a finite turb_step_tol not met in turb_step_maxit updates)."""
const REJECTION_REASONS = (:outer_not_converged, :outer_predicted, :outer_stalled, :linear_failed, :bounds, :nonfinite,
                           :turb_linear_failed, :turb_not_converged)

"""
    PBConfig(; mu, dt, kw...)

The numerical and physical settings of a P0b run (nondimensional or SI, consistently): rho_l, mu; dt; lambda_u (0.8);
n_outer_max (12), n_correctors (2), mom_sweeps (4); the absolute stopping tolerances mom_tol (1e-6, in rho U^2/L),
theta_tol (1e-9) -- a registered gate sets them at ten times the measured Float64 floor (plan 4.7 as amended 14 Oct);
p_tol_inner (1e-2, the relative forcing of the intermediate pressure solves), p_maxit (100);
nonorth, REQUIRED, no default (review F6: 1 the full non-orthogonal correction, 0 off, 0 < lambda_n < 1 limited; plan 4.1
chooses the production value from the MMS norms, and until then every caller states its own); skew (true: the
skewness-corrected face values of plan 4.1; false only for the suite's negative examples); convection (THE DEFAULT since
plan Amendment 8 (28.4, 15 Oct 2026) is :linear_upwind_unlimited, plan 4.1's form LUW with conv_gradient :unit;
:linear_upwind, the limited form of 4.1 before Amendment 8, and :upwind stay selectable); conv_gradient (THE DEFAULT is
:unit, unit weights on the least-squares stencil for the convection increment only: plan 26, R2 of 15 Oct 2026, the LUW
form; :inverse_distance, the least-squares gradient with weights 1/|d|^2 that the other deferred terms use, was the default
before Amendment 8); p_ref_cell (0: cell 1), p_ref (the reference pressure of a closed domain); U_ref, L_ref (the residual's units);
mms (NoMMS() or a field); backend; predict_reject (true); extrapolate (true: step 0 of 4.7); regrow_after (5),
regrow_factor (1.2); bound_tol (1e-6); relax_form (:patankar, production: r = (1/lambda - 1) a; :inviscid (r on a minus the
viscous two-point diagonal), :temporal (r on a^t) and :patankar_gen (Patankar through the general kernels) are
DIAGNOSTIC forms of review F2's probes, plan 23.1, not production options).
THE MULTIGRID HIERARCHY (review F5; plan 24): mg_update (:build, the host build at every outer iteration, the core's;
:refresh, the symbolic phase once per march and the device refresh `mg_refresh!` thereafter), and when to refresh:
mg_every_outer (k: every k outer iterations, counted across steps), or mg_every_steps (s > 0: at the first outer
iteration of every s-th step only), or mg_diag_delta (delta > 0: when the pressure diagonal has changed by more than delta
relative in some cell since the last refresh); at most one of the last two may be set; mg_coarse, the refresh's
coarsest inverse (:gj on the device, or :host, the build's LAPACK on a host copy; `mg_refresh!`). THE DEFAULT, chosen
by plan 24.5's rule from the CPU measurements (plan 24.6): :refresh with mg_diag_delta 1e-3 and mg_coarse :host -- one
host build per march, then a refresh whenever some cell's pressure diagonal has moved by more than 0.1% (on N60R's march
17% of the outer iterations, with the FCG and outer counts of a refresh every outer iteration; a contrast jump refreshes
at once), the coarsest inverse being the build's own arithmetic, so a refresh equals a host build bit for bit.
THE COUPLED STEP (plan 5.3; plan 27, 15 Oct 2026; linsolve_coupled.jl; ADOPTED by plan Amendment 8, 28.3, as the step of
every remaining stage, and THE DEFAULT since then): coupling (:coupled, the (u, v, p) system of each outer iteration solved
by FGMRES(cpl_restart), right-preconditioned by design C's SIMPLEC block preconditioner, with no under-relaxation; or
:segregated, the predictor-corrector loop above, the default before Amendment 8, kept bitwise as the reference and the
fallback); its settings cpl_eta (the FGMRES forcing term, relative; default 1e-1, plan 27.4's choice adopted in 28.3; it was
1e-2 before Amendment 8), cpl_abs (an absolute goal on the tolerance-scaled 2-norm), cpl_maxit (iterations per solve
before :linear_failed), cpl_au (SIMPLEC's alpha_u in the preconditioner only), cpl_sweeps (alternating momentum line sweeps
per application), cpl_vcycles (V(1,1) cycles on the Schur operator per application) and cpl_newton (true: J's momentum rows
carry the convection's derivative in the face flux, Newton in the convecting flux; false: Picard, the flux lagged);
cpl_reproject is a DIAGNOSTIC of development probe E (plan 27.1), not a production option. lambda_u, n_correctors and
mom_sweeps are not read on the coupled path.
THE FACE VELOCITY OF THE MOMENTUM INTERPOLATION (plan 28.6, review F3; 15 Oct 2026; assemble_momentum.jl, `_rc_lin_term`):
rc_weight (:linear, THE DEFAULT since the F3 fix: the steady face velocity is the linear interpolant of the two cells'
velocities, the coefficient-jump term added to Bartholomew's form; :coefficient, Bartholomew's form as built, whose steady
face velocity is the a^s-weighted mean, wrong by O(h) where a^s jumps at a velocity wall, which left the wall cells'
pressure error at O(mu), not converging; the default before the fix).
THE MOMENTUM INTERPOLATION ITSELF (plan 28.11, the odd-even norm's negative example; 15 Oct 2026): rhie_chow (:on, THE
DEFAULT, bitwise the path before the switch: the face flux by momentum interpolation; :off, a NEGATIVE EXAMPLE, never a
production option: the plain linear face velocity F_f = (w u_P + (1 - w) u_N) . S_f (the volumetric flux; its mass flux is
rho_l F_f), the owner's u_P . S_b on a pressure face, no pressure in the face flux, so continuity is written on linearly
interpolated velocities and the pressure decouples odd from even; D, the corrector and the coupled preconditioner are
unchanged, and on the coupled path J's continuity rows carry no pressure dependence).
THE DEFAULTS BEFORE AMENDMENT 8: `PBConfig(; PB_DEFAULTS_BEFORE_A8..., kw...)` is, bit for bit, what `PBConfig(; kw...)`
was before plan Amendment 8 (the segregated step, cpl_eta 1e-2, the limited :linear_upwind on :inverse_distance, Bartholomew's
coefficient-weighted face velocity); the suite and every reproduction of a record made before it use it.
THE CLOSURE-DEFECT DRIVE IN A DOMAIN WITHOUT A PRESSURE FACE (development, 15 Oct 2026; records
p0b/tg_stall/; OPT-IN, not adopted by any registration): theta_drive (:full, THE DEFAULT, bitwise the path before the option:
the pressure equation's right side carries V Theta^n/dt in every cell; :zero_mean: where the domain has no pressure face
(bc.npdir = 0: a closed or periodic box, the cavity), the drive is V (Theta^n - Theta^n_mean)/dt, Theta^n_mean the volume
mean of Theta^n; with a pressure face it is :full). Why: flux-form transport conserves the domain's total defect
sum V Theta exactly, so only round-off changes it, and it drifts; under :full the system is then inconsistent by that total,
and the pressure reference row absorbs it, so the reference cell's Theta is the total over that one cell's volume (about nc
times the mean), which no outer iteration can reduce, and the step stalls once it passes theta_tol (Taylor-Green n 128 at
t 0.444, records p0b/tg_stall/). Under :zero_mean the system is consistent and the conserved total stays spread over every
cell (|Theta| = |mean| + the solve's residual). The acceptance criterion (max |Theta| <= theta_tol) is unchanged.
"""
Base.@kwdef struct PBConfig{M<:MMSField,B}
    rho_l::Float64 = 1.0
    mu::Float64
    dt::Float64
    lambda_u::Float64 = 0.8
    n_outer_max::Int = 12
    n_correctors::Int = 2
    mom_sweeps::Int = 4
    mom_tol::Float64 = 1e-6
    theta_tol::Float64 = 1e-9
    p_tol_inner::Float64 = 1e-2
    p_maxit::Int = 100
    nonorth::Float64
    skew::Bool = true
    convection::Symbol = :linear_upwind_unlimited
    conv_gradient::Symbol = :unit
    p_ref_cell::Int = 0
    p_ref::Float64 = 0.0
    U_ref::Float64 = 1.0
    L_ref::Float64 = 1.0
    mms::M = NoMMS()
    backend::B = CPU()
    predict_reject::Bool = true
    extrapolate::Bool = true
    regrow_after::Int = 5
    regrow_factor::Float64 = 1.2
    bound_tol::Float64 = 1e-6
    relax_form::Symbol = :patankar
    mg_update::Symbol = :refresh
    mg_every_outer::Int = 1
    mg_every_steps::Int = 0
    mg_diag_delta::Float64 = 1e-3
    mg_coarse::Symbol = :host
    coupling::Symbol = :coupled
    cpl_eta::Float64 = 1e-1
    cpl_abs::Float64 = 0.1
    cpl_restart::Int = 30
    cpl_maxit::Int = 300
    cpl_au::Float64 = 0.8
    cpl_sweeps::Int = 1
    cpl_vcycles::Int = 1
    cpl_newton::Bool = true
    cpl_reproject::Bool = false
    cpl_visc_nonorth::Symbol = :jacobian
    rc_weight::Symbol = :linear
    rhie_chow::Symbol = :on
    theta_drive::Symbol = :full
    turbulence::Symbol = :laminar
    sst_tu::Float64 = 0.001
    sst_mut_ratio::Float64 = 0.1
    wall_omega::Symbol = :v1
    turb_coupling::Symbol = :step
    turb_tol::Float64 = 1e-8
    turb_lin_rtol::Float64 = 1e-2
    turb_sweeps_max::Int = 50
    turb_nonorth::Float64 = 1.0
    turb_nonorth_split::Bool = true
    turb_step_tol::Float64 = Inf
    turb_step_maxit::Int = 20
    turb_step_relax::Float64 = 1.0
    turb_outer_ramp::Int = 0
    turb_outer_tol::Float64 = 0.0
    turb_mut::Symbol = :frozen
    turb_mut_base::Symbol = :extrapolated     # 27 Oct 2026 (records air/mutbase/): the state the frozen mu_t is formed at
    turb_omega_nonorth::Symbol = :wall_off
    turb_omega_bound::Float64 = 0.2
    k_min::Float64 = 1e-20
    w_min::Float64 = 1e-8
    # W7, speed (plan Amendment 13, section 33; 22 Oct 2026; records air/w7_speed/): every default is the path before W7, bit for
    # bit; each option is opt-in until its registered adoption rule is met (W7_SPEED_REGISTRATION.md)
    sync::Symbol = :all
    line_solver::Symbol = :thomas
    prec_precision::Symbol = :f64
    kw_precision::Symbol = :f64
    cpl_gs::Symbol = :mgs
    graphs::Symbol = :off
    dot::Symbol = :chunk
    speed_dev::Bool = false
    domain::Symbol = :unset   # W7, the merge review's M2: :freeair (set by freeair_config) or :unset; no arithmetic
end

# THE SPEED SETTINGS (W7, plan Amendment 13, section 33; 22 Oct 2026). None changes the discretisation or a stopping rule.
#   sync            :all (THE DEFAULT) every device synchronisation of a step runs; :reads, on a GPU only the host reads drain the
#                   stream (mesh_faces.jl, pb_sync: no result changes; on the CPU both are the same)
#   line_solver     :thomas (THE DEFAULT, one work item per line) or :pcr, the workgroup parallel cyclic reduction of plan 5.5
#                   (linsolve_lines.jl), for every line solve: the multigrid's smoothing, the coupled preconditioner's momentum
#                   sweeps and the k-omega solves (changes the line solves' rounding)
#   prec_precision  :f64 (THE DEFAULT) or :f32: the coupled preconditioner (its momentum line sweeps and the pressure V-cycle) in
#                   Float32, its input and output Float64 (FGMRES, the residuals and every convergence test stay Float64: the
#                   flexible outer iteration absorbs the inexact preconditioner); the line factors are formed in Float64 and
#                   rounded (linsolve_lines.jl), the coarsest level's dense product accumulates in Float64
#   kw_precision    :f64 (THE DEFAULT) or :f32: the k-omega line Gauss-Seidel's corrections in Float32 (the residual, the iterate,
#                   the stopping test Float64: an iterative refinement, so the solve still converges to the Float64 system)
#   cpl_gs          :mgs (THE DEFAULT): FGMRES's modified Gram-Schmidt with each coefficient read to the host; :mgs_dev, the same
#                   operations in the same order with the coefficients kept on the device and the Hessenberg column read once
#                   per iteration (bit for bit on the CPU); :cgs2, classical Gram-Schmidt twice with fused multi-dots, 9 launches
#                   and one host read per iteration whatever the Krylov dimension (linsolve_coupled.jl; rounding differs)
#   graphs          :off (THE DEFAULT) or :on: on a GPU, the coupled preconditioner and the operator J captured as CUDA graphs
#                   (recaptured when the hierarchy, its companion or the reference row changes) and replayed, so each costs one
#                   launch on the host instead of one per kernel (linsolve_coupled.jl; needs sync = :reads and a graph backend
#                   installed by the driver, pb_graph_backend!; on the CPU it does nothing). The same kernels, in the same order,
#                   on the same arrays: no result changes
#   dot             :chunk (THE DEFAULT, 1,024 consecutive entries per work item) or :tree (a coalesced workgroup tree per
#                   2,048-entry block, linsolve_krylov.jl): deterministic and backend-independent either way; the two orders
#                   differ by rounding
#   speed_dev       false (THE DEFAULT) or true: the acknowledgement of a DEVELOPMENT run with the PROVISIONAL CUDA settings (the
#                   third W7 review's G1): on a non-CPU backend, while PB_SPEED_PROVISIONAL holds (:cuda, :air), a configuration with
#                   any SPEED_GATED field off its ungated values is refused by check_speed_config (so at every entry point)
#                   unless speed_dev = true; freeair_config sets it from PB_SPEED_DEV, the W7 scripts set it explicitly and print
#                   it. It changes no arithmetic
#   domain          :unset (THE DEFAULT) or :freeair: the case class the PROVISIONAL gate keys on (the merge review of cf838408,
#                   M2). freeair_config sets :freeair; a configuration built any other way (the tunnel, water, a script's
#                   PBConfig) is :unset. On a non-CPU backend a configuration counts as constant-density free air only when
#                   it is :freeair AND classified density-neutral (`sst_density_refusals` empty): its gate is then the
#                   (:cuda, :f32) row; every other configuration's is the (:cuda, :water) row (`config_gate_class`). It changes
#                   no arithmetic and is printed in no receipt
const W7_SPEED_VALUES = (sync=(:all, :reads), line_solver=(:thomas, :pcr), prec_precision=(:f64, :f32),
                         kw_precision=(:f64, :f32), cpl_gs=(:mgs, :mgs_dev, :cgs2), graphs=(:off, :on),
                         dot=(:chunk, :tree))
"""Check W7's speed settings (pimple_step! calls it). THE FLOAT32 PRECONDITIONER IS ADMITTED ONLY AT A UNIFORM DENSITY (the
W7 review's finding 4, 22 Oct 2026; fail-closed): prec_precision = :f32 is refused when the configuration is not classified
density-neutral by SST's classification (`sst_density_refusals`: every field classified, so a phase law, a compressibility
or a mixture option added later refuses it until classified), whatever the domain and however it was set; and every step
checks the state itself (`check_f32_state`). P0a's precision rule: the Float32 cycle failed at density contrast 4.3e4."""
function check_speed_config(cfg)
    for (f, vals) in pairs(W7_SPEED_VALUES)
        getproperty(cfg, f) in vals || throw(ArgumentError("PBConfig.$f must be one of $(join(vals, ", ")) (got :$(getproperty(cfg, f)))"))
    end
    (cfg.graphs === :on && !(cfg.backend isa CPU)) &&
        (cfg.sync === :reads || throw(ArgumentError("PBConfig.graphs = :on needs sync = :reads (a stream cannot be synchronised while it is captured)"));
         PB_GRAPH[] === nothing && throw(ArgumentError("PBConfig.graphs = :on: no graph backend installed (pb_graph_backend!, set by the driver after loading CUDA)")))
    cfg.domain in (:unset, :freeair) || throw(ArgumentError("PBConfig.domain must be :unset or :freeair (got :$(cfg.domain))"))
    # THE PROVISIONAL GATE ON THE CONFIGURATION THAT RUNS (the third W7 review's G1): whatever built it -- the driver's knobs,
    # freeair_config with another backend, a script's PBConfig -- a non-CPU configuration with a gated value needs speed_dev;
    # which values are gated follows the configuration's class (the merge review's M1-M2: config_gate_class, speed_gate_row)
    bad = config_gate_refusals(cfg)
    isempty(bad) || throw(ArgumentError("PBConfig $(join(bad, ", ")) on a non-CPU backend ($(typeof(cfg.backend))) is refused: " *
        PB_SPEED_PROVISIONAL[first(speed_gate_row(:cuda, config_gate_class(cfg)))] * ". Set speed_dev = true (PB_SPEED_DEV=1 " *
        "through the knobs) to acknowledge a development run, and print it with the reading (the third W7 review's G1)"))
    if cfg.prec_precision === :f32
        why = sst_density_refusals(cfg)
        isempty(why) || throw(ArgumentError("PBConfig.prec_precision = :f32 is refused: the Float32 preconditioner is admitted " *
            "only at a uniform density (P0a: the Float32 cycle failed at density contrast 4.3e4; the W7 review's finding 4, " *
            "fail-closed); " * join(why, "; ")))
    end
    nothing
end

"""The class of a configuration for the PROVISIONAL gate (the merge review's M2): :air for constant-density free air --
domain = :freeair (freeair_config's) and classified density-neutral (`sst_density_refusals` empty) -- and :water for every
other configuration (the tunnel, water, a script's PBConfig), so a script-built CUDA water configuration meets the water
row, not the lifted air row."""
config_gate_class(cfg) = (cfg.domain === :freeair && isempty(sst_density_refusals(cfg))) ? :air : :water

"""The gated fields of a configuration that the PROVISIONAL gate refuses (empty when it may run): on a non-CPU backend,
without cfg.speed_dev, every field gated by the row that governs the configuration's class (`speed_gate_row(:cuda,
config_gate_class(cfg))`: the water row's SPEED_GATED, or for free air the (:cuda, :f32) row's SPEED_GATED_F32; the knob's
name lower-cased, e.g. PB_LINE_SOLVER -> line_solver) whose value is not one of its ungated values."""
function config_gate_refusals(cfg)
    (cfg.backend isa CPU || cfg.speed_dev) && return String[]
    gr = speed_gate_row(:cuda, config_gate_class(cfg))
    gr === nothing && return String[]
    bad = String[]
    for (k, ok) in gr[2]
        f = Symbol(lowercase(k[4:end])); v = getproperty(cfg, f)
        v in ok || push!(bad, "$f=$v")
    end
    bad
end

"""The state check of the Float32 preconditioner (prec_precision = :f32; the W7 review's finding 4): at the start of every
step the level-n density must be uniform (q = 0 and max |rho/rho_l - 1| <= SST_THETA_MAX, measured on the state as SST's
check does), else the step is refused (ArgumentError). Two device reductions."""
function check_f32_state(st::PBState, m::PBMesh, w, cfg)
    be = cfg.backend
    qn = pb_maxabs(lev(st.q, 2); backend=be)
    _launch!(_k_rho_dev!, be, m.nc, w.scr, lev(st.rho, 2), cfg.rho_l)
    dev = pb_maxabs(w.scr; backend=be)
    _f32_state_verdict(dev, qn)
end
"""The same rule without a work space (host copies of the state's level n): the parity entry points, whose states are the
CPU snapshots (the second W7 review's R2)."""
function check_f32_state(st::PBState, cfg)
    q = Array(lev(st.q, 2)); r = Array(lev(st.rho, 2))
    _f32_state_verdict(maximum(x -> abs(x/cfg.rho_l - 1), r; init=0.0), maximum(abs, q; init=0.0))
end
function _f32_state_verdict(dev, qn)
    (dev <= SST_THETA_MAX && qn == 0) || throw(ArgumentError("PBConfig.prec_precision = :f32 is refused on this state: its " *
        "density is not uniform (max |rho/rho_l - 1| = $dev, SST_THETA_MAX $SST_THETA_MAX; max |q| = $qn): the Float32 " *
        "preconditioner is admitted only at a uniform density (the W7 review's finding 4)"))
    nothing
end

"""Run f() with W7's module modes (PB_SYNC, PB_DOT) set from cfg after `check_speed_config`, and restore the previous modes
afterwards, whatever f does (the W7 review's finding 10: before, a step's modes persisted into whatever ran next)."""
function with_speed_modes(f, cfg)
    s0 = PB_SYNC[]; d0 = PB_DOT[]
    check_speed_config(cfg); pb_sync_mode!(cfg.sync); pb_dot_mode!(cfg.dot)
    try
        return f()
    finally
        PB_SYNC[] = s0; PB_DOT[] = d0
    end
end

# THE TURBULENCE MODEL'S SETTINGS (W1, plan Amendment 12, section 32; 21 Oct 2026; assemble_komega.jl, turbulence.jl).
# OPT-IN: `turbulence` (:laminar, THE DEFAULT, bit for bit the path before W1: nothing below is read; :sst, SST-2003m); with
# :sst, the inflow turbulence `sst_tu` (intensity) and `sst_mut_ratio` (mu_t/mu), whose k and omega (sst_freestream at U_ref)
# enter at inflow faces and with backflow; `wall_omega` (:v1, option W-a, 60 nu/(beta1 d1^2); :auto, W-b, a stub until
# stage P2); `turb_coupling`:
#   :step, THE DEFAULT since W1b (review of W1, F2; W1's scored runs used it): the k-omega update after the flow's accepted
#     iterate (its flux, rho^{n+1}), k and omega lagged one step in the momentum (CORRECTED by W6, the review of the integration
#     F10 d: before W6 this said "mu_t lagged one step", but mu_t was re-formed from the iterate's strain at every outer
#     iteration; since W6 that is turb_mut = :iterate, and the default :frozen forms it once at the step's start, below).
#     `turb_step_tol` sets the per-step k-omega error:
#     Inf (THE DEFAULT) one update, the steady march's pseudo-time form (W1's, bit for bit; a converged backward-Euler
#     k-omega step need not exist at a large dt while k grows, since the lagged production is what keeps the update
#     positive and bounded); a finite value repeats the update (turb_update!, then the solve) until the k-omega relative
#     residual re-assembled at the new iterate, with its time term, is within it, at most `turb_step_maxit` updates, else the
#     step is rejected :turb_not_converged and dt halved (the time-accurate form of k and omega: register the tolerance;
#     CORRECTED by W6b, the review of W6, R5: under turb_mut = :frozen, the default since W6, these repetitions never reach
#     the momentum, whose mu_t was formed at the step's first outer iterate, so a finite turb_step_tol alone does not make
#     the coupled step time-accurate in mu_t; see turb_mut below, and register :outer or a dt-convergence pair for URANS).
#     The step's info reports the updates (`turb_iters`) and the final residual (`turb_res`, measured on every step);
#   :outer: the k-omega solve after the flow update in every outer iteration, a step accepted only when the k-omega relative
#     residual at the iterate is within `turb_tol` as well. It cannot start from rest (the review's probe R3: from a uniform
#     start the frozen momentum operator moves with mu_t while k grows by orders of magnitude, and dt collapses), so it is
#     REFUSED from a uniform turbulence field (k within SST_OUTER_UNIFORM_RTOL of cell 1's everywhere: the documented accident,
#     `init_turbulence!`'s uniform start; since the integration a tolerance, not exact equality: re-review of W1b, R5) in a
#     domain with a no-slip wall, unless `turb_outer_ramp` > 0: the first turb_outer_ramp committed steps of the state
#     (st.nstep < turb_outer_ramp) then take the :step treatment, and :outer the rest;
# `turb_mut` (work item W6, a robust start for steady SST in free air; 22 Oct 2026; records air/w6_start/): how the momentum's
#   mu_t is formed within a step that takes the :step treatment (k and omega are frozen within such a step either way; under
#   :outer's treatment the iterate's mu_t is re-formed at every outer iteration whatever this says, since its k-omega solve
#   needs the iterate's terms):
#   :iterate (W1's, bit for bit; THE DEFAULT BEFORE W6): `turb_update!` re-forms mu_t (and the face mu_eff) from the strain rate of every
#     outer iterate, so the momentum operator moves with the iterate. Through the SST limiter a1 k / max(a1 omega, S F2) that
#     is a Picard loop on mu_t(S) which the coupled Newton step does not linearise; W6 measured it setting the outer
#     contraction to about 0.5 per solve at the trailing-edge wake cells of the NACA 0012 at Re 1e6 (mu_t/mu 36), with dt
#     collapsed to 3e-4 at t 0.58 while mu_t moved within each step at the residual's own rate (dev_w6_diag_*);
#   :frozen, THE DEFAULT SINCE W6: mu_t and mu_eff formed ONCE per step, at its first outer iteration (k^n, omega^n and the
#     step's start iterate),
#     and held through the step's coupled solves; on acceptance mu_t is re-formed at the accepted iterate before the k-omega
#     update, so the k-omega step is W1's (its production from the accepted iterate's strain) and the committed mu_t is that
#     of the new k, omega and the accepted velocity, as under :iterate. THE STEADY FIXED POINT IS THE SAME: at a steady state
#     the step's start iterate is the committed state, so the frozen mu_t is the state's own, and the steady stop's residuals
#     (steady_residual, turbulence_steady_residual) re-form mu_t at the committed state under either form. Only the
#     transient differs (mu_t lagged by the step's start). W6 measured (records air/w6_start/): TMR's plate L0 STEADY in the
#     same 140 steps with 932 coupled solves against 1,549 and no rejection against 6, C_f(0.97) W1's to 5.3e-11, the two steady
#     states within 7.9e-10 in u; the review of the integration's F2 case (the first SST step of the NACA 0012 at Re 1e5,
#     alpha 5) converges in 20, 12 and 9 solves at dt 0.25, 1e-2 and 2.44e-4 where :iterate takes 28, 23 and does not converge
#     in 40 (the laminar step's 24, 11, 9): with mu_t held, an SST step gets easier as dt falls, as a backward-Euler step must.
#     TIME ACCURACY (W6b; the review of W6, R5): under :frozen the momentum is EXPLICIT in mu_t within a step. mu_t is formed
#     from k^n, omega^n and the step's start iterate, which is the EXTRAPOLATED velocity when cfg.extrapolate (the default),
#     and nothing the step does afterwards (its solves, or the k-omega repetitions of a finite turb_step_tol) feeds back into
#     the momentum's mu_t. The splitting error is O(dt), the order of backward Euler, so it is acceptable for the steady
#     pseudo-time march and for a first-order URANS; but it is a lag of the whole of mu_t (not only of k and omega, as under
#     :iterate), and it would hold any later second-order time scheme (the three state levels anticipate a variable-step
#     BDF2, state.jl) to first order. A URANS registration (A4) must therefore name either turb_coupling = :outer (mu_t
#     re-formed and k and omega solved at every outer iteration, with `turb_outer_tol`) or, under :step, a measured
#     dt-convergence pair of its shedding quantity;
# `turb_mut_base` (27 Oct 2026, branch pbcav-mutbase; records air/mutbase/; the A2F_a15 diagnosis,
#   docs/_records/2026-10-23_a2/diagnosis_A2F_a15/): THE STATE THE FROZEN mu_t IS FORMED AT, at the step's first outer iteration;
#   read only under turb_mut = :frozen in a step that takes the :step treatment (`frozen_mut_level`):
#   :extrapolated, THE DEFAULT (W6's, bit for bit): the step's start iterate, level 1 as `_step_start!` leaves it, i.e. u, v and
#     Fb extrapolated linearly from levels n and n-1 when cfg.extrapolate (k and omega are level n's: never extrapolated);
#   :committed: the committed state, level n: u, v, Fb, k and omega of level 2 (`turb_update!(...; level=2)`). Only the mu_t
#     formation changes: the flow's start iterate is still extrapolated, the step's mu_t is still written to level 1 and the face
#     viscosities, and on acceptance mu_t is re-formed at the accepted iterate as before.
#   WHY (the diagnosis, sections 4 and 6): where the SST stress limiter is active the stress is nearly independent of the strain
#   (tau = a1 rho k), so a mu_t that follows the strain cancels part of the momentum's restoring force on a grid-scale streamwise
#   wave; a step that converges its momentum under a mu_t formed at its start iterate x0 maps that wave's error as e^(n+1) = G x0,
#   a Picard iteration with |G| about 0.6 on A2F_a15. With the extrapolated start x0 = 2e^n - e^(n-1) the recurrence g^2 =
#   G (2g - 1) has a neutral travelling mode where |G| = 1/|2e^(i theta) - 1| (A2F_a15: a 5.7-column wave travelling one column
#   per step, the steady residual held at 0.82-0.90 for 2,000 steps with the loads settled to 3e-8); with x0 = e^n the error
#   contracts by |G| per step. THE STEADY FIXED POINT IS THE SAME (at a steady state levels n and n-1 agree). Without extrapolation
#   (st.nstep = 0, dt_prev = 0 or cfg.extrapolate = false) level 1 is level n at the first outer iteration, and the two forms are
#   bit for bit. REFUSED with turb_mut = :iterate (no step forms a frozen mu_t there, so the setting would be ignored); admitted
#   with turb_coupling = :outer (it applies to the :step-treated steps of the ramp);
# `turb_outer_tol` (W6, 22 Oct 2026; the review of the integration, F3 b): `:outer`'s per-step k-omega ACCEPTANCE tolerance on its
#   own: 0, THE DEFAULT, is turb_tol (bit for bit the path before it). Before it turb_tol was both :outer's acceptance and the line
#   solves' goal floor (0.1 turb_tol), so an :outer run with a steady stop had to accept every step to below ten times the steady
#   rule (the default 1e-8 is five decades stricter than W1's :outer, 1e-3; the review measured 499 steps on plate L0 after a
#   10-step ramp against :step's 140). With turb_outer_tol set, turb_tol stays the solves' floor and the acceptance is
#   turb_outer_tol, e.g. W1's 1e-3 with turb_tol 1e-8 (W6's probe dev_w6_outer_tol). Not read under :step;
# `turb_omega_nonorth` (W6, 22 Oct 2026): the deferred non-orthogonal diffusion of OMEGA (k keeps turb_nonorth's form whatever
#   this says): :full (W1b's, bit for bit: omega's correction as k's, turb_nonorth with its split; THE DEFAULT BEFORE W6);
#   :wall_off, THE DEFAULT SINCE W6, no omega correction on an internal face either of whose cells has a no-slip face
#   (TurbWork.wfmask; two-point there; bit for bit :full wherever those faces carry no correction; on TMR's plate L0, whose
#   masked faces have |k|/|S| <= 1.7e-8, omega differs by <= 9.4e-8 after 10 steps and the steady C_f(0.97) is W1's to 1e-12);
#   :limited,
#   omega's correction on every face clamped to the two-point flux, |c| <= |Gw (w_N - w_P)| (W1's limiter at lambda 0.5,
#   omega only). WHY (W6's measurement, dev_w6_te_*): near a wall omega follows 6 nu/(beta1 y^2), so the least-squares omega
#   gradient of a wall-adjacent cell carries the singularity (3.3e7 on the wall face, 2.7e6 and 4.6e5 in the first two cells of
#   the NACA 0012 at Re 1e6, dn1 1e-5); interpolated onto the faces between the last foil cells and the first wake cells behind
#   the sharp trailing edge it gives a net correction about 1e4 times the rest of those cells' equation (Cw -3.86 against a
#   right side of 4e-4), so their discrete omega equation has no positive solution, the positivity split's diagonal share
#   -C/omega^m drives omega to the floor w_min and holds it there (diagonal 3.9e8 at omega 1e-8), and mu_t in those cells is set
#   by the stress limiter alone (with turb_mut = :iterate that made the outer iteration's mu_t loop non-contractive there and
#   collapsed dt to 3e-4 at Re 1e6). Both options change omega only where the correction is formed from wall-singular gradients
#   (:wall_off) or exceeds the two-point flux (:limited). :wall_off removed every omega floor event of that case (W6's arm F);
#   :limited is W1's limiter on omega alone, kept selectable (the review of W1's F1 found the limiter inconsistent on smooth
#   fields of distorted meshes, which :wall_off avoids away from walls);
# `turb_omega_bound` (W6b, 22 Oct 2026; the review of W6, R1; records air/w6b_start2/): THE DEFAULT 0.2 since W6b (0, W6's: no
#   bound; PB_DEFAULTS_BEFORE_W6B). Whatever turb_omega_nonorth leaves, omega's NET negative deferred correction C_P of a cell
#   is bounded below by -turb_omega_bound Q_P, Q_P the cell's steady positive budget: the omega production, the positive cross
#   diffusion and the boundary inflow on its right side, plus sum_N |a_N| omega_N^m (the time term and the destruction's
#   linearisation excluded, so the bound does not depend on dt and cannot let omega decay by a fixed factor per step);
#   `TurbWork.wcut` holds what it removed. It bounds the positivity split's net correction, so it acts under the split only
#   (turb_nonorth = 1 with turb_nonorth_split, the default); W1's limited form (turb_nonorth < 1, PB_DEFAULTS_BEFORE_W1B) and
#   the unsplit form do not read it (no refusal: the suite's W1 checks run them at the default). So, as far as the
#   non-orthogonal correction goes, the cell's steady equation has a positive solution, transport and destruction balancing at
#   least (1 - bound) Q_P, and the positivity split's share -C/omega^m can no longer hold omega at the floor (the negative cross
#   diffusion, put on the diagonal in the same Patankar form -cr/omega^m, is neither in Q_P nor bounded: the claim holds where
#   |cr| V < (1 - bound) Q_P; no lane shows a trap from it; the review of W6b, F5). WHY: on A2's wall spacing (W4b's mesh, dn1 1e-6, Re 6e6) the trap :wall_off removed on W6's meshes returned one row
#   up the first wake column (rows 5-12 at the floor by step 32; the review's R1): the correction enters there through the
#   tilted column's own j-faces and the lateral faces to the foil column, whose least-squares omega gradients still carry the
#   wall singularity; no face mask by rows is robust to the spacing, while a bound on the cell's net correction against its own
#   budget is. Where the correction is a consistent, resolved term it is a small fraction of the budget and the bound does not
#   act (bit for bit the unbounded form there). THE FIXED POINT (the review of W6b, F1 c): the bounded scheme's steady fixed
#   point does not depend on dt; it is the UNBOUNDED scheme's only where the bound is inactive at convergence. Measured (W6b's suite and
#   the review's probes P1, P2, P4; development, not data): inactive at STEADY on TMR's plate L0, on W6's lane re1e5_a5_v2 and
#   on A2 at alpha 0 (it acted there in the start only: 7 of 546 steps, the first wake cells); ACTIVE at STEADY on W6's trap mesh under turb_omega_nonorth = :full (the review's P1: cut/Q 0.477 in
#   the two first wake cells, C_D 1.2e-4 relative from the bound-0 continuation, which never reaches STEADY), where the bounded
#   answer is the scheme's. So a reading states the bound's census at its committed state (freeair_drive!'s result line,
#   `omega_bound_census!`). The value: from the review's A2 state at step 20, 0.5 still let the column's omega fall
#   (297 -> 9.5 in four steps, mu_t/mu 491, as without the bound), 0.2 held it and it recovered to the next column's level
#   (491 -> 1.5e3 in five steps, mu_t/mu about 45, the bound acting in two cells); a smaller value acts in more cells;
# `turb_tol` (THE DEFAULT 1e-8 since the W1+W2 integration, 22 Oct 2026; W1 and W1b shipped 1e-3; this heading restored by W6b,
# the review of W6, R6): under :outer the step's
# k-omega acceptance tolerance; it sets the line solve's goal floor (`_kw_goal_floor`): 0.1 turb_tol under :outer, 0.1 min(turb_tol,
# turb_step_tol) under :step (corrected wording, W6; the review of the integration, F3: the min holds under :step only)
# (`_kw_goal_floor`), below which a solve returns without a sweep. At 1e-3 that floor (1e-4) froze k and omega in a steady march
# (re-review of W1b, R1: plate L0, 518 of 600 steps without a sweep, the steady k residual held at 9.98e-5 while the momentum
# reached W1's rule). 1e-8 is W1's registered value (its floor 1e-9 lies two decades below W1's steady k-omega rule 1e-7), so
# the default is now a tolerance, not an iteration count; a forced sweep per step was the alternative, rejected because it
# would change W1's arithmetic below the floor and still not bound the steady residual. A steady march under SST must also
# register its k-omega rule (`pbcav_march!`'s turb_steady_tol), which must lie above the goal floor (refused otherwise);
# the line solve's `turb_lin_rtol` (relative reduction), `turb_sweeps_max` (alternating j/i sweeps per solve; a solve that
# misses its goal is continued once for three times that budget more, counted, then the step is rejected :turb_linear_failed),
# `turb_nonorth` (the deferred non-orthogonal diffusion: 1, THE DEFAULT since W1b, the full correction with the positivity
# split of `turb_nonorth_split` (true); 0 < lambda < 1 W1's limited form, inconsistent, kept to reproduce W1; 0 none;
# assemble_komega.jl's header); the floors `k_min`, `w_min` (in the case's units; counted and ledgered). SST takes the coupled
# step (coupling = :coupled) and relax_form :patankar; the segregated loop refuses it. SST REQUIRES A UNIFORM DENSITY (W1b,
# review F3): `check_turbulence_config` refuses every setting that is not classified as density-neutral (fail-closed:
# SST_DENSITY_NEUTRAL_FIELDS, SST_DENSITY_CHECKS), and pimple_step! refuses a state whose density is not uniform.

"""The largest density deviation max |rho/rho_l - 1| (and theta_tol) admitted with SST (W1b, review F3): the k-omega
transport stores rho_l k and rho_l omega, scales its sources by rho_l and forms mu_t = rho_l nu_t, against a time term in the
mixture density; the two agree to (1 + max |rho/rho_l - 1|), which must stay at the incompressible single-phase limit's
round-off. The state check measures rho/rho_l - 1 itself, not the step's closure defect Theta (re-review of W1b, R4: Theta is
rho/rho_l - 1 only while the closure is psi = 0; stage P1 redefines it against an EOS density)."""
const SST_THETA_MAX = 1e-6
"""The relative tolerance of the `:outer` from-rest refusal (re-review of W1b, R5): the turbulence field counts as the uniform
start when max_P |k_P - k_1| <= SST_OUTER_UNIFORM_RTOL |k_1| (before the integration: exact equality, so one cell's k
multiplied by 1 + 1e-9 escaped the refusal and collapsed as probe R3 did). A developed field departs from cell 1's k by
orders of magnitude, so 1e-3 cannot refuse one."""
const SST_OUTER_UNIFORM_RTOL = 1e-3

"""THE PBConfig FIELDS THAT CANNOT MAKE THE DENSITY NON-UNIFORM (W1b, review F3). SST is admitted only when every field of the
configuration is either here or in SST_DENSITY_CHECKS: a field added later (a liquid or vapour compressibility, a phase law, a
mass-transfer model, a gas content, a mixture option of stage P1 or P5a) is unclassified and so REFUSES SST until its stage
classifies it, and the suite (testset W1b) fails while any PBConfig field is unclassified. Classify a field here only if no
value of it can make rho differ from rho_l anywhere."""
const SST_DENSITY_NEUTRAL_FIELDS = (:rho_l, :mu, :dt, :lambda_u, :n_outer_max, :n_correctors, :mom_sweeps, :mom_tol,
    :p_tol_inner, :p_maxit, :nonorth, :skew, :p_ref_cell, :p_ref, :U_ref, :L_ref, :backend,
    :predict_reject, :extrapolate, :regrow_after, :regrow_factor, :bound_tol, :mg_every_outer,
    :mg_every_steps, :mg_diag_delta, :cpl_eta, :cpl_abs, :cpl_restart, :cpl_maxit, :cpl_au,
    :cpl_sweeps, :cpl_vcycles, :cpl_newton, :cpl_reproject, :sst_tu,
    :sst_mut_ratio, :turb_tol, :turb_lin_rtol, :turb_sweeps_max, :turb_nonorth,
    :turb_nonorth_split, :turb_step_tol, :turb_step_maxit, :turb_step_relax, :turb_outer_ramp, :turb_outer_tol, :turb_omega_bound, :k_min, :w_min,
    :speed_dev,                   # W7 (the third review's G1): an acknowledgement flag, no arithmetic
    :domain)                      # W7 (the merge review's M2): the gate's case-class label, no arithmetic
# (W6, 22 Oct 2026: turb_mut and turb_omega_nonorth are enumerated fields, classified by value in SST_DENSITY_ADMITTED_VALUES
# below)
"""THE ENUMERATED (Symbol) PBConfig FIELDS AND THE VALUES ADMITTED WITH SST (re-review of W1b, R4; the W1+W2 integration, 22 Oct
2026): a field whose value selects a model or a form is classified BY VALUE, not by name, so a later stage that adds a value
(a mixture relaxation form, a compressible coupling, a phase-change convection scheme, a drive) to one of them is refused under
SST until that value is classified here. Each list is every value the code accepts today, all of which keep rho = rho_l."""
const SST_DENSITY_ADMITTED_VALUES = (convection=(:linear_upwind, :linear_upwind_unlimited, :linear_upwind_vanalbada, :upwind),
    conv_gradient=(:inverse_distance, :unit), relax_form=(:patankar, :inviscid, :temporal, :patankar_gen),
    mg_update=(:refresh, :build), mg_coarse=(:host, :gj), coupling=(:coupled, :segregated), rc_weight=(:linear, :coefficient),
    rhie_chow=(:on, :off), theta_drive=(:full, :zero_mean), turbulence=(:laminar, :sst), wall_omega=(:v1, :auto),
    turb_coupling=(:step, :outer), turb_mut=(:iterate, :frozen), turb_mut_base=(:extrapolated, :committed),
    turb_omega_nonorth=(:full, :wall_off, :limited),
    cpl_visc_nonorth=(:jacobian, :deferred), W7_SPEED_VALUES...)   # W7's speed settings: linear-solver forms only, each keeps rho = rho_l
_sst_admits(f::Symbol) = cfg -> getproperty(cfg, f) in getproperty(SST_DENSITY_ADMITTED_VALUES, f)
"""The PBConfig fields whose value decides whether the density stays uniform (W1b, review F3; re-review R4), each with the
predicate that must hold for SST: the MMS field must declare itself density-neutral (`mms_density_neutral`, false for an
undeclared field), theta_tol must not exceed SST_THETA_MAX, and every enumerated field must hold one of its admitted values
(SST_DENSITY_ADMITTED_VALUES)."""
const SST_DENSITY_CHECKS = (; mms = cfg -> mms_density_neutral(cfg.mms), theta_tol = cfg -> cfg.theta_tol <= SST_THETA_MAX,
                            (f => _sst_admits(f) for f in keys(SST_DENSITY_ADMITTED_VALUES))...)

"""
    sst_density_refusals(cfg) -> Vector{String}

Why the configuration may make the density non-uniform (empty when SST is admissible; W1b, review F3): every property of
`cfg` (a PBConfig, or any object with the same properties) is looked up in SST_DENSITY_NEUTRAL_FIELDS and SST_DENSITY_CHECKS;
an unclassified property is a refusal (fail-closed), and so is a check that does not hold.
"""
function sst_density_refusals(cfg)
    why = String[]
    for f in propertynames(cfg)
        f in SST_DENSITY_NEUTRAL_FIELDS && continue
        if haskey(SST_DENSITY_CHECKS, f)
            SST_DENSITY_CHECKS[f](cfg) || push!(why, "$f = $(repr(getproperty(cfg, f))) can make the density non-uniform")
        else
            push!(why, "the setting $f is not classified as density-neutral (SST_DENSITY_NEUTRAL_FIELDS / SST_DENSITY_CHECKS)")
        end
    end
    why
end

"Check a configuration's turbulence settings (pimple_step! calls it when turbulence is not :laminar)."
function check_turbulence_config(cfg)
    cfg.turbulence in (:laminar, :sst) ||
        throw(ArgumentError("PBConfig.turbulence must be :laminar or :sst (got :$(cfg.turbulence))"))
    cfg.turbulence === :laminar && return nothing
    cfg.coupling === :coupled || throw(ArgumentError("PBConfig.turbulence = :sst takes the coupled step (coupling = :coupled); " *
                                                     "the segregated loop has no k-omega solve (W1 builds it on the coupled step)"))
    cfg.turb_coupling in (:outer, :step) ||
        throw(ArgumentError("PBConfig.turb_coupling must be :outer or :step (got :$(cfg.turb_coupling))"))
    cfg.turb_mut in (:iterate, :frozen) ||
        throw(ArgumentError("PBConfig.turb_mut must be :iterate or :frozen (got :$(cfg.turb_mut))"))
    cfg.turb_mut_base in (:extrapolated, :committed) ||
        throw(ArgumentError("PBConfig.turb_mut_base must be :extrapolated or :committed (got :$(cfg.turb_mut_base))"))
    (cfg.turb_mut_base === :committed && cfg.turb_mut !== :frozen) &&
        throw(ArgumentError("PBConfig.turb_mut_base = :committed needs turb_mut = :frozen (got :$(cfg.turb_mut)): under :iterate " *
                            "mu_t is re-formed at every outer iterate, no step forms a frozen mu_t, and the setting would be ignored"))
    cfg.turb_omega_nonorth in (:full, :wall_off, :limited) ||
        throw(ArgumentError("PBConfig.turb_omega_nonorth must be :full, :wall_off or :limited (got :$(cfg.turb_omega_nonorth))"))
    (cfg.turb_tol > 0 && 0 < cfg.turb_lin_rtol < 1 && cfg.turb_sweeps_max >= 1 && 0 <= cfg.turb_nonorth <= 1 &&
     cfg.k_min > 0 && cfg.w_min > 0) ||
        throw(ArgumentError("PBConfig: turb_tol > 0, 0 < turb_lin_rtol < 1, turb_sweeps_max >= 1, 0 <= turb_nonorth <= 1, " *
                            "k_min > 0 and w_min > 0 required"))
    (0 <= cfg.turb_omega_bound < 1) ||
        throw(ArgumentError("PBConfig.turb_omega_bound must lie in [0, 1) (0: no bound; got $(cfg.turb_omega_bound))"))
    (cfg.turb_outer_tol >= 0 && isfinite(cfg.turb_outer_tol)) ||
        throw(ArgumentError("PBConfig.turb_outer_tol must be >= 0 and finite (0: turb_tol; got $(cfg.turb_outer_tol))"))
    (cfg.turb_step_tol > 0 && !isnan(cfg.turb_step_tol) && cfg.turb_step_maxit >= 1 && 0 < cfg.turb_step_relax <= 1 && cfg.turb_outer_ramp >= 0) ||
        throw(ArgumentError("PBConfig: turb_step_tol > 0 (Inf: one update per step), turb_step_maxit >= 1, 0 < turb_step_relax <= 1 and " *
                            "turb_outer_ramp >= 0 required"))
    why = sst_density_refusals(cfg)
    isempty(why) || throw(ArgumentError("PBConfig.turbulence = :sst is refused: it requires a uniform density rho = rho_l " *
        "(review of W1, F3: the k-omega transport stores rho_l k, scales its sources by rho_l and forms mu_t = rho_l nu_t; the " *
        "mixture-density form is a registered item of stage P1 or P5a); " * join(why, "; ")))
    nothing
end

"""The start of an SST step (pimple_step!, W1b): the state's level n must have a uniform density (review F3: q = 0 and
max |rho/rho_l - 1| <= SST_THETA_MAX, both measured here on the state itself, independent of the step's closure defect Theta:
re-review of W1b, R4), and `turb_coupling = :outer` is refused from a uniform turbulence field (k within
SST_OUTER_UNIFORM_RTOL of cell 1's: re-review R5) in a domain with a no-slip wall unless the step falls in turb_outer_ramp
(review F2). Two or three device reductions and one one-element read."""
function check_turbulence_state(st::PBState, m::PBMesh, w, cfg)
    be = cfg.backend
    qn = pb_maxabs(lev(st.q, 2); backend=be)
    _launch!(_k_rho_dev!, be, m.nc, w.scr, lev(st.rho, 2), cfg.rho_l)
    dev = pb_maxabs(w.scr; backend=be)
    (dev <= SST_THETA_MAX && qn == 0) || throw(ArgumentError("PBConfig.turbulence = :sst is refused on this state: its " *
        "density is not uniform (max |rho/rho_l - 1| = $dev, SST_THETA_MAX $SST_THETA_MAX; max |q| = $qn); review of W1, F3"))
    if cfg.turb_coupling === :outer && st.nstep >= cfg.turb_outer_ramp && w.turb.nwall > 0
        rk = lev(st.rk, 2); r1 = _host(view(rk, 1:1))[1]
        _launch!(_k_absdiff_const!, be, m.nc, w.scr, rk, r1)
        pb_maxabs(w.scr; backend=be) <= SST_OUTER_UNIFORM_RTOL*abs(r1) && throw(ArgumentError("PBConfig.turb_coupling = :outer " *
            "is refused from a uniform turbulence field (k within $(SST_OUTER_UNIFORM_RTOL) of cell 1's everywhere) in a domain " *
            "with a no-slip wall (the impulsive start: review of W1, F2, probe R3 -- dt collapses while k grows by orders of " *
            "magnitude): use turb_coupling = :step (the default), or turb_outer_ramp = N so that the first N committed steps " *
            "take the :step treatment"))
    end
    nothing
end
@kernel function _k_absdiff_const!(out, a, c::Float64)
    i = @index(Global)
    @inbounds out[i] = abs(a[i] - c)
end
@kernel function _k_rho_dev!(out, rho, rho_l::Float64)
    i = @index(Global)
    @inbounds out[i] = abs(rho[i]/rho_l - 1)
end

"""The PBConfig defaults that plan Amendment 8 (28.3, 28.4) and the F3 fix (28.6) changed, at their values before the change:
`PBConfig(; PB_DEFAULTS_BEFORE_A8..., kw...)` reproduces the configuration `PBConfig(; kw...)` gave before them, bit for bit
(the segregated path kept as the reference and the fallback; plan 28.3)."""
const PB_DEFAULTS_BEFORE_A8 = (coupling=:segregated, cpl_eta=1e-2, convection=:linear_upwind, conv_gradient=:inverse_distance,
                               rc_weight=:coefficient)
"""The SST defaults that W1b and the W1+W2 integration changed, at W1's values (re-review of W1b, R6; the A8 convention):
`PBConfig(; PB_DEFAULTS_BEFORE_W1B..., kw...)` is W1's configuration (turb_coupling :outer, the limited non-orthogonal k-omega
diffusion 0.5, turb_tol 1e-3). W1's scored runs named turb_tol 1e-8 and turb_coupling :step themselves; its instrument was
`wall_shear(...; form = :two_point)` (the default form is :assembled since W1b)."""
const PB_DEFAULTS_BEFORE_W1B = (turb_coupling=:outer, turb_nonorth=0.5, turb_tol=1e-3)
"""The SST default the W1+W2 integration changed (22 Oct 2026; re-review of W1b, R1), at W1b's value: `PBConfig(;
PB_DEFAULTS_BEFORE_INT..., kw...)` is W1b's configuration (turb_tol 1e-3)."""
const PB_DEFAULTS_BEFORE_INT = (turb_tol=1e-3,)
"""The SST defaults W6 changed (22 Oct 2026; records air/w6_start/), at their values before it (the A8 convention):
`PBConfig(; PB_DEFAULTS_BEFORE_W6..., kw...)` is the integration's configuration (mu_t re-formed at every outer iteration of a
step, the full non-orthogonal correction of omega at walls). It restores the integration's ARITHMETIC, step for step (with
PB_DEFAULTS_BEFORE_INT and PB_DEFAULTS_BEFORE_W1B as before for the earlier ones), NOT its recorded OUTCOMES (CORRECTED by W6b,
the review of W6, R4): the steady stop's k-omega reading (`turbulence_steady_residual`) is normalised by the unsplit diagonal
since W6, with no switch back, so a run whose stop the split diagonal hid does not stop where it did. The review replayed the
integration's smoke lane v2 a0 with these settings: all 550 rows' march columns byte for byte the integration's, but steady_w
301.9 where the integration read 3.65e-8, and REACHED at the 560-step limit with four cells at the floor instead of STEADY
(both STEADY smoke lanes of the integration are such runs). That is the purpose of the change. A record whose stop no split
share affected reproduces its outcome as well (TMR's plate L0: W6 measured the split shares at most 5e-10 of the unsplit
diagonal there, and the suite's W1 and INT plate checks pass)."""
const PB_DEFAULTS_BEFORE_W6 = (turb_mut=:iterate, turb_omega_nonorth=:full, turb_omega_bound=0.0)
"""The SST default W6b changed (22 Oct 2026; records air/w6b_start2/), at W6's value: `PBConfig(; PB_DEFAULTS_BEFORE_W6B...,
kw...)` is W6's configuration (no bound on omega's non-orthogonal correction). With PB_DEFAULTS_BEFORE_W6 (which since W6b
also clears the bound) the integration's arithmetic. Laminar configurations read neither."""
const PB_DEFAULTS_BEFORE_W6B = (turb_omega_bound=0.0,)

"""
    check_turb_steady_tol(cfg, turb_steady_tol)

The k-omega steady rule of an SST steady stop (re-review of W1b, R1; the W1+W2 integration): it must be registered (a number
> 0, not `nothing`), and the k-omega line solve's goal floor, below which a solve takes no sweep, must lie below it, else the
rule could only be met by accident (W1b's turb_tol 1e-3 froze k and omega at a steady residual of 1e-4). ArgumentError
otherwise. The goal floor is `_kw_goal_floor`: 0.1 turb_tol under turb_coupling = :outer, 0.1 min(turb_tol, turb_step_tol)
under :step (W6's corrected wording; the review of the integration, F3: the min holds under :step only).
THE RULE AND THE GUARD MEASURE DIFFERENT THINGS (the review of the integration, F9): the goal floor bounds a per-step relative
residual WITH the time term, the rule a steady residual WITHOUT it; at a small pseudo-time step a field can freeze (no sweep)
while its steady residual sits above a rule this guard admitted. That gives a stall (REACHED), never a false STEADY.
"""
function check_turb_steady_tol(cfg, tst)
    tst === nothing && throw(ArgumentError("PBConfig.turbulence = :sst with a steady stop (steady_tol > 0) needs a registered " *
        "k-omega steady rule turb_steady_tol (the k and omega steady residuals, turbulence_steady_residual): the momentum's " *
        "steady residual alone cannot see a frozen turbulence field (re-review of W1b, R1)"))
    (tst isa Real && tst > 0 && !isnan(tst)) || throw(ArgumentError("turb_steady_tol must be a number > 0 (got $tst)"))
    gf = _kw_goal_floor(cfg)
    gf < tst || throw(ArgumentError("turb_steady_tol = $tst lies at or below the k-omega line solve's goal floor " *
        (cfg.turb_coupling === :outer ? "0.1 turb_tol (turb_coupling = :outer)" : "0.1 min(turb_tol, turb_step_tol) (turb_coupling = :step)") *
        " = $gf, below which a solve takes no sweep: the k-omega field would freeze above the " *
        "rule (re-review of W1b, R1); register turb_tol below 10 turb_steady_tol"))
    nothing
end

"The hierarchy policy's state (review F5): host builds and refreshes done, the counters since the last one, the
pressure diagonal at the last one (for mg_diag_delta)."
mutable struct MGPolicyState{V}
    builds::Int; refreshes::Int
    outer_since::Int; steps_since::Int
    dref::V; dscr::V
end

"""
    PBWork

The step's work space on the backend (gradients, the momentum and pressure systems, the momentum interpolation, the line
families, the Krylov work, the current multigrid hierarchy, the outer-iteration's previous iterate).
"""
mutable struct PBWork{V,FS,LF,KW}
    lsq::LSQGeom{V}
    gux::V; guy::V; gvx::V; gvy::V; gpx::V; gpy::V; ggx::V; ggy::V
    gdlx::V; gdly::V; gdx::V; gdy::V
    ubf::V; vbf::V; pbf::V; pbf2::V
    msx::FS; msy::FS; psys::FS
    a::V; at::V; as_::V; bsx::V; bsy::V; Hx::V; Hy::V; hbx::V; hby::V; hxx::V; hxy::V; hyx::V; hyy::V; res::V
    fcx::V; fcy::V; bdg::V; bfx::V; bfy::V
    Fs::V; Df::V; Fbs::V; Db::V
    uprev::V; vprev::V; Fprev::V; Fbprev::V
    dp::V; theta::V; sdiag::V; srhs::V; scr::V; fscr::V
    rw::V; zw::V
    fj::LF; fi::LF
    kw::KW
    mg::Any
    vmin::Float64
    cont_step::Float64
    nonpos_step::Int
    av::V; arel::V
    mgs::MGPolicyState{V}
    cux::V; cuy::V; cvx::V; cvy::V
    cpl::Any
    tbar::Float64            # the volume mean of Theta^n taken off the drive (theta_drive = :zero_mean), else 0
    turb::Any                # the SST work (TurbWork, assemble_komega.jl) under turbulence = :sst, else nothing (W1)
    muf::V; mub::V           # the momentum's face mu_eff under SST (internal, boundary; one-element and unread on the laminar path)
end

"""
    pbwork(mesh, bc, cfg) -> PBWork

Allocate the work space (and the mesh's least-squares geometry for the condition set `bc`, and its line colours).
"""
function pbwork(m::PBMesh, bc, cfg::PBConfig)
    be = cfg.backend
    check_on_backend(be, m, bc isa BCSet ? bc : nothing; what="pbwork: ")    # W3: a host mesh under a device configuration
    z(n) = KernelAbstractions.zeros(be, Float64, n)
    nc, nf, nb = m.nc, m.nf, m.nb
    msx = FaceSystem(m; backend=be)
    msy = FaceSystem(msx.diag, msx.upper, msx.lower, z(nc))        # shares the matrix, its own right side
    PBWork(lsq_geom(m, bc; backend=be),
           z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc),
           z(nb), z(nb), z(nb), z(nb),
           msx, msy, FaceSystem(m; backend=be),
           z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc), z(nc),
           z(nf), z(nf), z(nb), z(nb), z(nb),
           z(nf), z(nf), z(nb), z(nb),
           z(nc), z(nc), z(nf), z(nb),
           z(nc), z(nc), z(nc), z(nc), z(nc), z(max(nf, nb)),
           z(nc), z(nc),
           line_family(m, :j; backend=be), line_family(m, :i; backend=be),
           KrylovWork(nc; backend=be), nothing, minimum(_host(m.vol)), 0.0, 0, z(nc), z(nc),
           MGPolicyState(0, 0, 0, 0, z(nc), z(nc)), z(nc), z(nc), z(nc), z(nc), nothing, 0.0,
           cfg.turbulence === :sst ? turbwork(m, bc, cfg) : nothing,
           cfg.turbulence === :sst ? _filled(be, max(nf, 1), cfg.mu) : z(1),
           cfg.turbulence === :sst ? _filled(be, max(nb, 1), cfg.mu) : z(1))
end
"A backend vector of n entries equal to x (the SST face mu_eff start at mu: W1b, review N4, so a momentum assembly before the
first turb_update! is laminar, not inviscid; every present path calls turb_update! first)."
_filled(be, n::Integer, x::Float64) = (v = KernelAbstractions.zeros(be, Float64, n); fill!(v, x); v)

"""
    theta_drive_mean(w, mesh, bc, cfg) -> Float64

The volume mean of Theta^n (w.theta, set from level 2 at the start of the step) that the closure-defect drive takes off
under `theta_drive = :zero_mean` in a domain without a pressure face (PBConfig's docstring); 0.0 under :full (the default)
or with a pressure face. Two scalar reductions.
"""
function theta_drive_mean(w, m::PBMesh, bc, cfg)
    cfg.theta_drive === :full && return 0.0
    cfg.theta_drive === :zero_mean ||
        throw(ArgumentError("PBConfig.theta_drive must be :full or :zero_mean (got :$(cfg.theta_drive))"))
    bc.npdir == 0 || return 0.0
    _launch!(_k_mul3!, cfg.backend, m.nc, w.scr, w.theta, m.vol)
    pb_sum(w.scr; backend=cfg.backend)/pb_sum(m.vol; backend=cfg.backend)
end

# ---- the projection start (plan 4.7, Start; review F2, 14 Oct) ------------------------------------------------------
@kernel function _k_start_bflux!(Fb, bkind, ub, vb, bSx, bSy)
    b = @index(Global)
    @inbounds begin
        k = bkind[b]
        if is_velocity_bc(k)
            Fb[b] = ub[b]*bSx[b] + vb[b]*bSy[b]
        elseif k == UInt8(BC_SLIP)
            Fb[b] = 0.0
        end
    end
end
@kernel function _k_start_vel!(u, v, p, gx, gy, rho_l::Float64, Uref::Float64, bern::Int32)
    c = @index(Global)
    @inbounds begin
        uu = u[c] - gx[c]; vv = v[c] - gy[c]; u[c] = uu; v[c] = vv
        bern == Int32(1) && (p[c] = 0.5*rho_l*(Uref*Uref - (uu*uu + vv*vv)))
    end
end

"""
    project_start!(st, mesh, bc, cfg; work=nothing, tol=1e-12, maxit=200, bernoulli=true) -> NamedTuple

The registered start of a wetted run from a given field (plan 4.7, Start, as amended for review F2, 14 Oct 2026): one
incompressible projection of the state's level 1 (set it first, e.g. `init_state!` with uniform flow). The boundary
fluxes are set by the conditions (velocity faces u_b . S, slip faces 0, pressure outlets the owner's value), then
  sum_f |S|^2/(S.d) (phi_P - phi_N) + sum_outlets |S_b|^2/(S_b.d_b) phi_P = - sum_f F0_f(out)      (phi = 0 at the outlets)
is solved (the pressure equation's own operator with D = 1, FCG with the Schaffer cycle, to `tol` relative), and
F = F0 - |S|^2/(S.d) (phi_N - phi_P), u = u0 - (grad phi)^GG: the face flux is discretely divergence-free (to the solve's
residual) and has no flux through the walls; the cell velocity is the potential flow without circulation. With
`bernoulli` the pressure is set to 1/2 rho_l (U_ref^2 - |u|^2) (the steady potential flow's Bernoulli pressure, zero where
|u| = U_ref; `bernoulli_U` replaces U_ref there, 24 Oct 2026: the ramp start projects a flow of speed r0 U_ref), otherwise
left. Every time level is set to the result. Returns the FCG count, the relative residual, and
max |div F|/V before and after (units U/L).
"""
project_start!(st::PBState, m::PBMesh, bc, cfg::PBConfig; kw...) =
    with_speed_modes(() -> _project_start!(st, m, bc, cfg; kw...), cfg)   # W7: the modes set from cfg and restored after
function _project_start!(st::PBState, m::PBMesh, bc, cfg::PBConfig; work=nothing, tol::Real=1e-12, maxit::Integer=200,
                         bernoulli::Bool=true, bernoulli_U::Real=cfg.U_ref)
    be = cfg.backend
    w = work === nothing ? pbwork(m, bc, cfg) : work
    F = lev(st.F, 1); Fb = lev(st.Fb, 1)
    _launch!(_k_start_bflux!, be, m.nb, Fb, bc.kind, bc.ub, bc.vb, m.bSx, m.bSy)
    pb_sync(be)
    copyto!(w.Fs, F); copyto!(w.Fbs, Fb); fill!(w.Df, 1.0); fill!(w.Db, 1.0)
    fill!(w.sdiag, 0.0); fill!(w.srhs, 0.0)
    rhon = lev(st.rho, 2); copyto!(w.scr, rhon); fill!(rhon, cfg.rho_l)        # Theta^n = 0 for the projection
    dts = st.dt; st.dt = cfg.dt                                                 # (the Theta^n/dt term: 0/dt, never 0/0)
    assemble_pressure!(w, m, st, bc, cfg, 0.0)
    st.dt = dts; copyto!(rhon, w.scr)
    div0 = _maxdiv(w, m, cfg)
    h = build_schaffer_mg(w.psys, m; backend=be)
    fill!(w.dp, 0.0)
    r = fcg!(w.dp, w.psys, m, (z, rr) -> mg_vcycle!(z, h, rr); tol, maxit, work=w.kw, backend=be)
    r.converged || error("project_start!: the projection solve did not converge in $maxit FCG iterations (rel $(r.rel))")
    _launch!(_k_corr_face!, be, m.nf, F, w.Fs, w.Df, m.fa, w.dp, m.own, m.nei)
    _launch!(_k_corr_bface!, be, m.nb, Fb, w.Fbs, w.Db, m.ba, w.dp, bc.kind, m.bown)
    pressure_gradients!(w.gdlx, w.gdly, w.gdx, w.gdy, w.pbf2, w.dp, w, m, bc, cfg; correction=true)
    _launch!(_k_start_vel!, be, m.nc, lev(st.ux, 1), lev(st.uy, 1), lev(st.p, 1), w.gdx, w.gdy, cfg.rho_l, Float64(bernoulli_U),
             Int32(bernoulli ? 1 : 0))
    pb_sync(be)
    copyto!(w.Fs, F); copyto!(w.Fbs, Fb)
    div1 = _maxdiv(w, m, cfg)
    copy_level!(st, 2, 1); copy_level!(st, 3, 1)
    (fcg=r.iters, rel=r.rel, div_before=div0, div_after=div1)
end
@kernel function _k_div!(out, F, Fb, cface, nf::Int32, vol)
    c = @index(Global)
    @inbounds begin
        s = 0.0
        for k in 1:4
            code = cface[k, c]
            if code > nf
                s += Fb[code - nf]
            elseif code > 0
                s += F[code]
            else
                s -= F[-code]
            end
        end
        out[c] = abs(s)/vol[c]
    end
end
"max_P |sum_f F(out)|/V_P of the fluxes in w.Fs, w.Fbs."
function _maxdiv(w, m::PBMesh, cfg)
    _launch!(_k_div!, cfg.backend, m.nc, w.scr, w.Fs, w.Fbs, m.cface, Int32(m.nf), m.vol)
    pb_maxabs(w.scr; backend=cfg.backend)
end

@kernel function _k_extrap!(x1, x2, x3, r::Float64)
    i = @index(Global)
    @inbounds x1[i] = x2[i] + r*(x2[i] - x3[i])
end
@kernel function _k_theta!(theta, rho, rho_l::Float64)
    c = @index(Global)
    @inbounds theta[c] = rho[c]/rho_l - 1
end
@kernel function _k_nonpos!(out, D, n0::Int32)
    i = @index(Global)
    @inbounds out[i] = (i <= n0 && !(D[i] > 0)) ? 1.0 : 0.0
end
@kernel function _k_conterr!(out, rho1, rhon, F, Fb, cface, nf::Int32, vol, rho_l::Float64, dt::Float64)
    c = @index(Global)
    @inbounds begin
        s = 0.0
        for k in 1:4
            code = cface[k, c]
            if code > nf
                s += Fb[code - nf]
            elseif code > 0
                s += F[code]
            else
                s -= F[-code]
            end
        end
        out[c] = abs((rho1[c] - rhon[c])*vol[c]/dt + rho_l*s)
    end
end
@kernel function _k_bounds!(out, rho, q, rho_l::Float64)
    c = @index(Global)
    @inbounds out[c] = min((rho[c] - q[c])/rho_l, 0.0)
end

"""The outer residual has stalled within a decade of the tolerances (header, :outer_stalled): the last three R in
[0.1, 10], and their smallest no lower than 0.9 times the smallest of the three before (no new low by 10% in three
iterations). A slow but steady contraction (0.92 per iteration was measured on the dt 100, lambda 0.7 box, which a first
form of this rule, "the last at least half of the one three before", took for a stall) is not a stall."""
function _stalled(R::AbstractVector{Float64})
    length(R) >= 6 || return false
    all(r -> 0.1 <= r <= 10, @view R[end-2:end]) && minimum(@view R[end-2:end]) >= 0.9*minimum(@view R[end-5:end-3])
end
"A non-convergence rejection is :outer_stalled when the residual history shows the floor (_stalled), else `reason`."
_nonconv(reason::Symbol, R) = _stalled(R) ? :outer_stalled : reason

@kernel function _k_reldiff!(out, a, b)
    i = @index(Global)
    @inbounds out[i] = abs(a[i] - b[i])/abs(b[i])
end

"""
    mg_update!(w, mesh, cfg, first_outer::Bool)

The pressure hierarchy for the current outer iteration (called at its first corrector, after the pressure equation is
assembled; review F5, plan 24): under mg_update = :build a host build of w.psys; under :refresh the host build with its
numeric plan the first time (or when w.mg has no plan), and then `mg_refresh!` when the policy says it is due
(mg_every_outer, mg_every_steps at a step's first outer iteration, or mg_diag_delta), else the hierarchy is kept.
"""
function mg_update!(w::PBWork, m::PBMesh, cfg::PBConfig, first_outer::Bool)
    be = cfg.backend; s = w.mgs
    fast!(h) = mg_fast_update!(h, cfg.prec_precision === :f32 ? Float32 : Float64, cfg.line_solver)   # W7 (none by default)
    if cfg.mg_update === :build
        w.mg = build_schaffer_mg(w.psys, m; backend=be); s.builds += 1; fast!(w.mg)
        return nothing
    end
    cfg.mg_update === :refresh || throw(ArgumentError("PBConfig.mg_update must be :build or :refresh (got $(cfg.mg_update))"))
    (cfg.mg_every_steps > 0 && cfg.mg_diag_delta > 0) &&
        throw(ArgumentError("PBConfig: set at most one of mg_every_steps and mg_diag_delta"))
    cfg.mg_every_outer >= 1 || throw(ArgumentError("PBConfig.mg_every_outer must be >= 1"))
    if w.mg === nothing || w.mg.plan === nothing
        w.mg = build_schaffer_mg(w.psys, m; backend=be, plan=true); s.builds += 1; fast!(w.mg)
        s.outer_since = 0; s.steps_since = 0; copyto!(s.dref, w.psys.diag)
        return nothing
    end
    s.outer_since += 1; first_outer && (s.steps_since += 1)
    due = if cfg.mg_every_steps > 0
        first_outer && s.steps_since >= cfg.mg_every_steps
    elseif cfg.mg_diag_delta > 0
        _launch!(_k_reldiff!, be, m.nc, s.dscr, w.psys.diag, s.dref)
        pb_maxabs(s.dscr; backend=be) > cfg.mg_diag_delta
    else
        s.outer_since >= cfg.mg_every_outer
    end
    if due
        mg_refresh!(w.mg, w.psys; coarse=cfg.mg_coarse); s.refreshes += 1; fast!(w.mg)
        s.outer_since = 0; s.steps_since = 0; copyto!(s.dref, w.psys.diag)
    end
    nothing
end

"Count the non-positive D values of the current corrector (internal faces; pressure-Dirichlet boundary faces)."
function _count_nonpos_D(w, m::PBMesh, bc, cfg)
    be = cfg.backend
    _launch!(_k_nonpos!, be, m.nf, w.fscr, w.Df, Int32(m.nf)); n = round(Int, pb_sum(view(w.fscr, 1:m.nf); backend=be))
    if bc.npdir > 0
        _launch!(_k_nonpos!, be, m.nb, w.fscr, w.Db, Int32(m.nb))
        _launch!(_k_mask_pdir!, be, m.nb, w.fscr, bc.kind)
        n += round(Int, pb_sum(view(w.fscr, 1:m.nb); backend=be))
    end
    n
end
@kernel function _k_mask_pdir!(out, bkind)
    b = @index(Global)
    @inbounds is_pressure_bc(bkind[b]) || (out[b] = 0.0)
end

"""
    _step_start!(st, cfg) -> st

The iterate a step starts its outer iterations from, `pimple_step!`'s prologue: level 1 restored from level 2, then u, v, p, F,
Fb extrapolated from levels 2-3 at dt/dt_prev when cfg.extrapolate (st.nstep >= 1, dt_prev > 0). ONE FUNCTION for the step and
for `parity_production`'s level-3 re-evaluation under turb_mut = :frozen (which forms the step's mu_t at this iterate), so the
two cannot drift apart (the review of W6b, F10: the parity tool had a hand copy; the same kernels in the same order, so the
step is bit for bit what it was).
"""
function _step_start!(st::PBState, cfg)
    be = cfg.backend
    # W7 (the merge with W6b: the prologue is one function, so the step and the parity tool share these): the speed settings'
    # checks, the PROVISIONAL gate and speed_dev included (check_speed_config; the entry points, with_speed_modes, set and
    # restore the sync and dot modes around the call), and the Float32 preconditioner's uniform-density rule on level n (the
    # host form: no work space here)
    check_speed_config(cfg)
    restore_level!(st)
    cfg.prec_precision === :f32 && check_f32_state(st, cfg)
    if cfg.extrapolate && st.nstep >= 1 && st.dt_prev > 0
        r = st.dt/st.dt_prev
        for fld in (:ux, :uy, :p, :F, :Fb)
            A = getfield(st, fld)
            _launch!(_k_extrap!, be, size(A, 1), lev(A, 1), lev(A, 2), lev(A, 3), r)
        end
        pb_sync(be)
    end
    st
end

"""
    frozen_mut_level(cfg, outer_turb::Bool) -> Int

The time level `turb_update!` reads when it forms the frozen mu_t of a step at its first outer iteration (`turb_mut_base`, the
SST settings' header): 2 (level n, the committed state) under turb_mut = :frozen with turb_mut_base = :committed in a step that
takes the :step treatment (`outer_turb` false); 1 otherwise (the step's start iterate: W6's, the default). ONE FUNCTION for
coupled_outer! and for the parity tool (`parity_step!`, `parity_production`), so the three cannot drift apart.
"""
frozen_mut_level(cfg, outer_turb::Bool) =
    (cfg.turb_mut === :frozen && !outer_turb && cfg.turb_mut_base === :committed) ? 2 : 1

"""
    pimple_step!(st, mesh, bc, w, cfg) -> (accepted::Bool, reason::Symbol, info::NamedTuple)

One step n -> n+1 (plan 4.7) of size st.dt from level 2. `reason` is `:accepted` or one of REJECTION_REASONS; on a
rejection the iterate (level 1) is restored from level 2 bitwise. `info`: outer iterations, the residual history R (on an
accepted step its last entry is the final iterate's), the FCG counts, the final |Theta|, `mom` the momentum residual OF
THE FINAL ITERATE (re-assembled there) and `mom_pre` the one measured at the start of the final outer iteration.
"""
pimple_step!(st::PBState, m::PBMesh, bc, w::PBWork, cfg::PBConfig) =
    with_speed_modes(() -> _pimple_step!(st, m, bc, w, cfg), cfg)   # W7: the modes set from cfg and restored after
function _pimple_step!(st::PBState, m::PBMesh, bc, w::PBWork, cfg::PBConfig)
    be = cfg.backend
    t_new = st.t + st.dt
    _step_start!(st, cfg)
    u = lev(st.ux, 1); v = lev(st.uy, 1)
    _launch!(_k_theta!, be, m.nc, w.theta, lev(st.rho, 2), cfg.rho_l)
    theta = pb_maxabs(w.theta; backend=be)
    w.tbar = theta_drive_mean(w, m, bc, cfg)       # 0 unless theta_drive = :zero_mean in a domain without a pressure face
    if cfg.turbulence !== :laminar                 # W1 (opt-in)
        check_turbulence_config(cfg)
        w.turb === nothing && throw(ArgumentError("PBConfig.turbulence = :sst needs a work space built for it (pbwork with that configuration)"))
        check_turbulence_state(st, m, w, cfg)               # W1b: a uniform density (F3); :outer not from rest (F2)
        turb_step_reset!(w.turb)
    end
    cfg.coupling === :segregated || return coupled_outer!(st, m, bc, w, cfg, t_new, theta)   # plan 5.3, 27 (opt-in)
    kry_ok = true; final = false; Rhist = Float64[]; kits = Int[]; resmom = NaN; fgate = 1.0
    Rm = Float64[]; Rt = Float64[]                 # the two parts of R per outer iteration (review_b1, R7; bookkeeping only)
    ffail = Ref(:none)                             # the residual the last final re-check failed on (R7; bookkeeping only)
    w.cont_step = 0.0; w.nonpos_step = 0
    reject(reason) = (restore_level!(st); return (false, reason, (outer=length(Rhist), R=Rhist, fcg=kits, theta=theta, mom=resmom,
                                                                   mom_pre=resmom, Rm=Rm, Rt=Rt, final_fail=ffail[])))
    for it in 1:cfg.n_outer_max
        copyto!(w.uprev, u); copyto!(w.vprev, v); copyto!(w.Fprev, lev(st.F, 1)); copyto!(w.Fbprev, lev(st.Fb, 1))
        resmom = assemble_momentum!(w, m, st, bc, cfg, t_new)
        isfinite(resmom) || return reject(:nonfinite)
        R = max(resmom/cfg.mom_tol, theta/cfg.theta_tol); push!(Rhist, R); push!(Rm, resmom/cfg.mom_tol); push!(Rt, theta/cfg.theta_tol)
        if !final && R <= fgate && kry_ok
            final = true
        end
        if !final && cfg.predict_reject && it == 7 && length(Rhist) >= 7
            rate = (Rhist[7]/Rhist[4])^(1/3)
            more = rate < 1 ? log(Rhist[7])/(-log(rate)) : Inf
            (it - 1 + more + 1 > cfg.n_outer_max) && return reject(_nonconv(:outer_predicted, Rhist))
        end
        lam = final ? 1.0 : cfg.lambda_u
        relax_momentum!(w, m, cfg, lam)
        line_tridiag!(w.fj, w.msx, m; backend=be); line_tridiag!(w.fi, w.msx, m; backend=be)
        for _ in 1:cfg.mom_sweeps
            for (x, sys) in ((u, w.msx), (v, w.msy))
                line_gs_sweep!(x, sys, m, w.fj, w.rw, w.zw; backend=be)
                line_gs_sweep!(x, sys, m, w.fi, w.rw, w.zw; backend=be)
            end
        end
        kry_ok = true
        for k in 1:cfg.n_correctors
            momentum_H!(w, m, st, cfg)
            k > 1 && pressure_gradients!(w.gpx, w.gpy, w.ggx, w.ggy, w.pbf, lev(st.p, 1), w, m, bc, cfg)
            rhie_chow_flux!(w, m, st, bc, cfg, lam)
            w.nonpos_step += _count_nonpos_D(w, m, bc, cfg)
            assemble_pressure!(w, m, st, bc, cfg, t_new; tbar=w.tbar)
            k == 1 && mg_update!(w, m, cfg, it == 1)
            fill!(w.dp, 0.0)
            last = final && k == cfg.n_correctors
            tol = last ? 0.0 : cfg.p_tol_inner
            atol = last ? 0.1*cfg.theta_tol*w.vmin/st.dt : 0.0
            h = w.mg
            r = fcg!(w.dp, w.psys, m, (z, rr) -> mg_vcycle!(z, h, rr); tol, atol, maxit=cfg.p_maxit, work=w.kw, backend=be)
            push!(kits, r.iters)
            r.converged || return reject(:linear_failed)
            correct_fluxes!(w, m, st, bc, cfg, lam)
        end
        transport_mass!(w, m, st, cfg, t_new)
        theta = pb_maxabs(w.theta; backend=be)
        fin = max(pb_maxabs(u; backend=be), pb_maxabs(v; backend=be), pb_maxabs(lev(st.p, 1); backend=be), theta)
        isfinite(fin) || return reject(:nonfinite)
        if final
            _launch!(_k_bounds!, be, m.nc, w.scr, lev(st.rho, 1), lev(st.q, 1), cfg.rho_l)
            pb_maxabs(w.scr; backend=be) > cfg.bound_tol && return reject(:bounds)
            # the committed iterate's continuity-error term (review F6: was read from the iterate before this one)
            _launch!(_k_conterr!, be, m.nc, w.scr, lev(st.rho, 1), lev(st.rho, 2), lev(st.F, 1), lev(st.Fb, 1), m.cface,
                     Int32(m.nf), m.vol, cfg.rho_l, st.dt)
            w.cont_step = pb_sum(w.scr; backend=be)*st.dt
            # the final iterate re-checked (review F6): its own momentum residual against the same tolerance. A miss is not
            # a rejection: the outer loop goes on from this iterate (relaxed again), and the next final iteration is tried
            # only once the iterate has gained a further decade (fgate / 10: without it a final unrelaxed iteration that
            # amplifies R slightly above 1 alternates with relaxed ones that bring it just below, measured on the dt 100,
            # lambda 0.7 box); the acceptance tolerances never change. Within N_max. The next iteration's residual is this
            # iterate's, so it is not recorded twice.
            resfin = assemble_momentum!(w, m, st, bc, cfg, t_new)
            isfinite(resfin) || return reject(:nonfinite)
            if theta <= cfg.theta_tol && resfin <= cfg.mom_tol
                push!(Rhist, max(resfin/cfg.mom_tol, theta/cfg.theta_tol)); push!(Rm, resfin/cfg.mom_tol); push!(Rt, theta/cfg.theta_tol)
                return (true, :accepted, (outer=it, R=Rhist, fcg=kits, theta=theta, mom=resfin, mom_pre=resmom, Rm=Rm, Rt=Rt,
                                          final_fail=ffail[]))
            end
            ffail[] = theta > cfg.theta_tol ? (resfin > cfg.mom_tol ? :both : :theta) : :momentum
            final = false; fgate /= 10
        end
    end
    reject(_nonconv(:outer_not_converged, Rhist))
end

"""
    pbcav_march!(st, mesh, bc, cfg; t_end=Inf, max_steps=typemax(Int), steady_tol=0.0, work=nothing, ledger=PBLedger(),
                 on_step=nothing, max_halvings=10, stop=nothing, steady_ok=nothing, turb_steady_tol=nothing)
        -> (st, ledger, history)

March from the state's level 2 (set its time levels first, e.g. `init_state!`): steps of cfg.dt, rejections (dt halved),
regrowth, the ledgers after every committed step. Stops at t_end, after max_steps committed steps, or, with
steady_tol > 0, when the committed state's STEADY RESIDUAL (`steady_residual`: the momentum residual with the time term
removed, in rho U^2/L per unit volume; review F4) falls to steady_tol; `steady` in the history is that residual (NaN
when steady_tol = 0). `on_step(st, info, ledger)` is called after each committed step. Refuses (error, with the reason
and the tally) after `max_halvings` consecutive rejections, and on a second consecutive :outer_stalled. `stop(st)`, when
given, is called after every attempt (committed or rejected) and ends the march when it returns true (`pbcav_run!`'s wall
cap, plan 28.3); `nothing` (the default) leaves the march as before, bit for bit. `steady_ok(st)` (W2, 21 Oct 2026, free air's
point-vortex correction), when given, must also return true for the steady stop (called after `on_step`, only once the steady
residual has fallen to steady_tol); `nothing` (the default) leaves the stop as before, bit for bit. `pre_step(st)` (24 Oct 2026,
the ramp start of freeair.jl), when given, is called before every attempt, committed or rejected, with st.t + st.dt the time the
attempt reaches (the ramp sets its far field there); `nothing` (the default) leaves the march as before, bit for bit.
UNDER SST (the W1+W2 integration, 22 Oct 2026; re-review of W1b, R1) a steady stop (steady_tol > 0) also needs the k and omega
STEADY RESIDUALS (`turbulence_steady_residual`, relative, the time term removed) at or below the registered rule
`turb_steady_tol`, measured after every committed step and recorded as `steady_k`, `steady_w` in its history record; the rule
must be given and must lie above the k-omega solve's goal floor (`check_turb_steady_tol`; refused before the first step).
The laminar path is unchanged (its records carry no new field).
"""
pbcav_march!(st::PBState, m::PBMesh, bc, cfg::PBConfig; kw...) =
    with_speed_modes(() -> _pbcav_march!(st, m, bc, cfg; kw...), cfg)   # W7: the modes held for the march, restored after
function _pbcav_march!(st::PBState, m::PBMesh, bc, cfg::PBConfig; t_end::Real=Inf, max_steps::Integer=typemax(Int),
                       steady_tol::Real=0.0, work=nothing, ledger::PBLedger=PBLedger(), on_step=nothing,
                       max_halvings::Integer=10, stop=nothing, steady_ok=nothing, turb_steady_tol=nothing, pre_step=nothing,
                       control=nothing)
    be = cfg.backend
    sst_steady = cfg.turbulence === :sst && steady_tol > 0
    sst_steady && check_turb_steady_tol(cfg, turb_steady_tol)
    w = work === nothing ? pbwork(m, bc, cfg) : work
    ledger.nsteps == 0 && ledger_init!(ledger, m, st, w, cfg)
    st.dt = st.dt > 0 ? min(st.dt, cfg.dt) : cfg.dt
    hist = NamedTuple[]; streak = control === nothing ? 0 : Int(get(control,"streak",0))
    nrej = control === nothing ? 0 : Int(get(control,"nrej",0)); nstall = control === nothing ? 0 : Int(get(control,"nstall",0))
    nacc = 0; tally = control === nothing ? Dict{Symbol,Int}() : Dict(Symbol(k)=>Int(v) for (k,v) in get(control,"tally",Dict()))
    save_control() = control === nothing ? nothing : merge!(control,Dict("streak"=>streak,"nrej"=>nrej,"nstall"=>nstall,"tally"=>Dict(string(k)=>v for (k,v) in tally)))
    while (!isfinite(t_end) || st.t < t_end - 1e-12*max(1.0, abs(t_end))) && nacc < max_steps
        isfinite(t_end) && (st.dt = min(st.dt, t_end - st.t))
        pre_step === nothing || pre_step(st)        # the attempt's boundary values (freeair.jl's ramp start); none by default
        acc, reason, info = pimple_step!(st, m, bc, w, cfg)
        if !acc
            ledger.rejections[reason] = get(ledger.rejections, reason, 0) + 1
            tally[reason] = get(tally, reason, 0) + 1
            rej = rejection_record(reason, info, st, m, bc, w, cfg); ledger.last_rejection = rej      # review_b1, R7
            nrej += 1; streak = 0; nstall = reason === :outer_stalled ? nstall + 1 : 0
            nstall >= 2 && error("pbcav_march!: refused at t = $(st.t), dt = $(st.dt): the outer residual stalled within a " *
                                 "decade of the tolerances in $nstall consecutive attempts (:outer_stalled, last R " *
                                 "$(isempty(info.R) ? NaN : info.R[end]), stalled residual $(rej.stalled)): mom_tol $(cfg.mom_tol) or theta_tol " *
                                 "$(cfg.theta_tol) sits at the Float64 floor (in a transient the momentum floor is about eps rho |u|/dt, which a " *
                                 "smaller dt raises; the Theta floor scales with the Courant number); register " *
                                 "the tolerances at ten times the measured floor (plan 4.7); " * rejection_text(rej, cfg))
            nrej > max_halvings && error("pbcav_march!: refused at t = $(st.t) after $nrej consecutive rejections " *
                                         "(max_halvings $max_halvings; reasons $(tally); last $reason, stalled residual $(rej.stalled)); " *
                                         rejection_text(rej, cfg))
            st.dt /= 2
            save_control()
            stop !== nothing && stop(st) && break
            continue
        end
        nrej = 0; nstall = 0; empty!(tally)
        commit_level!(st); ledger_update!(ledger, m, st, w, cfg); nacc += 1
        steady = steady_tol > 0 ? steady_residual(st, m, bc, w, cfg) : NaN
        rec = merge(info, (t=st.t, dt=st.dt_prev, steady=steady))
        tsteady = true
        if sst_steady                              # the k-omega steady rule (re-review of W1b, R1)
            sk, sw = turbulence_steady_residual(st, m, bc, w, cfg)
            tsteady = sk <= turb_steady_tol && sw <= turb_steady_tol
            rec = merge(rec, (steady_k=sk, steady_w=sw))
        end
        push!(hist, rec)
        on_step === nothing || on_step(st, rec, ledger)
        streak += 1
        streak >= cfg.regrow_after && st.dt < cfg.dt && (st.dt = min(cfg.dt, st.dt*cfg.regrow_factor))
        save_control()
        steady_tol > 0 && steady <= steady_tol && tsteady && (steady_ok === nothing || steady_ok(st)) && break
        stop !== nothing && stop(st) && break
    end
    st, ledger, hist
end

"""
    pbcav_run!(st, mesh, bc, cfg; wall_cap, t_end=Inf, steady_tol=0.0, max_steps=typemax(Int), segment=600.0, work=nothing,
               ledger=PBLedger(), on_step=nothing, max_halvings=10, clock=time, steady_ok=nothing, turb_steady_tol=nothing)
        -> NamedTuple

ONE REGISTERED RUN under plan Amendment 8's standing rules (28.3, 28.4 item 6; 15 Oct 2026): a single march
(`pbcav_march!`), never a second one. Outcomes (`outcome`):
  :STEADY       steady_tol > 0 and the steady residual fell to it (under SST also the k and omega steady residuals to the
                registered turb_steady_tol: `pbcav_march!`; the W1+W2 integration, re-review of W1b R1);
  :REACHED      t_end (or max_steps) reached (with steady_tol > 0: reached without a steady state, the budget spent);
  :NOT_REACHED  the march was refused (`refusal`: its message; `tally`: the rejection reasons of the march's ledger); a
                multigrid build or refresh refusal (an ArgumentError naming a refused pivot) is a refusal too;
  :INCOMPLETE   the registered wall cap `wall_cap` (seconds) was reached: the march stops between attempts at the first
                check after the cap (checks every `segment` seconds, at most 600 s or 1% of the cap, plan 28.3), with the
                committed t, the committed steps and the last dt recorded. Never a refusal, never NOT_REACHED.
THERE IS NO RERUN. Plan 25.2 and 25.6's "rerun once at dt/4" is removed (28.4 item 6, review F5): the march starts at
min(st.dt, cfg.dt) and regrows only up to cfg.dt, so once the failed march's dt had fallen to dt/4 and stayed there, the rerun
replayed it bit for bit (three refusals identical in t and dt). A refused level reads NOT_REACHED with its refusal and tally.
The fields: outcome, t, nstep (committed steps of this run), dt (the state's dt at the end), wall (seconds), refusal (the
march's message IN FULL), tally, history, ledger, marches (always 1), and (review_b1, R7, 15 Oct 2026) the refusal's parts as
separate fields: `reason` (the typed reason of the refused attempt, :mg_refused for a multigrid build or refresh refusal,
:none when the run was not refused), `stalled` (which residual kept that attempt from acceptance: :momentum, :theta, :both,
the segregated loop's final re-check failures :momentum_final, :theta_final, :both_final, or :linear, :nonfinite, :bounds,
:none; `rejection_record`) and `last_rejection` (the record of this run's last rejected attempt, refused or not, or nothing).
"""
function pbcav_run!(st::PBState, m::PBMesh, bc, cfg::PBConfig; wall_cap::Real, t_end::Real=Inf, steady_tol::Real=0.0,
                    max_steps::Integer=typemax(Int), segment::Real=600.0, work=nothing, ledger::PBLedger=PBLedger(),
                    on_step=nothing, max_halvings::Integer=10, clock=time, steady_ok=nothing, turb_steady_tol=nothing,
                    pre_step=nothing, control=nothing, stop_requested=nothing)
    (wall_cap > 0 && isfinite(wall_cap)) || throw(ArgumentError("pbcav_run!: a registered wall cap (> 0 s) is required"))
    cfg.turbulence === :sst && steady_tol > 0 && check_turb_steady_tol(cfg, turb_steady_tol)
    seg = min(Float64(segment), 600.0, 0.01*wall_cap)
    rej0 = ledger.last_rejection
    t0 = clock(); nextcheck = Ref(t0 + seg); capped = Ref(false); requested = Ref(false); n0 = st.nstep; last_steady = Ref(NaN)
    function stop(s)
        if stop_requested!==nothing && stop_requested(s)
            requested[]=true
            return true
        end
        tn = clock()
        tn < nextcheck[] && return false
        while nextcheck[] <= tn; nextcheck[] += seg; end
        tn - t0 >= wall_cap && (capped[] = true)
        capped[]
    end
    last_turb = Ref(true)                          # the k-omega steady rule at the last committed step (SST; else true)
    cb = (s, rec, l) -> (last_steady[] = rec.steady;
                         haskey(rec, :steady_k) && (last_turb[] = rec.steady_k <= turb_steady_tol && rec.steady_w <= turb_steady_tol);
                         on_step === nothing || on_step(s, rec, l))
    hist = NamedTuple[]; refusal = ""; mgref = false
    outcome = try
        _, _, hist = pbcav_march!(st, m, bc, cfg; t_end, max_steps, steady_tol, work, ledger, on_step=cb, max_halvings, stop,
                                  steady_ok, turb_steady_tol, pre_step, control)
        if requested[]
            :STOPPED
        elseif capped[]
            :INCOMPLETE
        elseif steady_tol > 0 && last_steady[] <= steady_tol && last_turb[] && (steady_ok === nothing || steady_ok(st))
            :STEADY
        else
            :REACHED
        end
    catch e
        march_ref = e isa ErrorException && startswith(e.msg, "pbcav_march!: refused")
        mgref = e isa ArgumentError && occursin("pivot", e.msg) && occursin("refused", e.msg)
        (march_ref || mgref) || rethrow()
        refusal = e.msg
        :NOT_REACHED
    end
    rej = ledger.last_rejection === rej0 ? nothing : ledger.last_rejection      # this run's last rejected attempt
    reason = outcome !== :NOT_REACHED ? :none : mgref ? :mg_refused : rej === nothing ? :none : rej.reason
    stalled = outcome !== :NOT_REACHED ? :none : mgref ? :linear : rej === nothing ? :none : rej.stalled
    (outcome=outcome, t=st.t, nstep=st.nstep - n0, dt=st.dt, wall=clock() - t0, refusal=refusal,
     tally=copy(ledger.rejections), history=hist, ledger=ledger, marches=1, reason=reason, stalled=stalled, last_rejection=rej)
end

# ---- the record of a rejected attempt (review_b1, R7, 15 Oct 2026) ------------------------------------------------------
"""
    rejection_record(reason, info, st, mesh, bc, w, cfg) -> NamedTuple

What kept one rejected attempt from acceptance, as separate fields (review_b1, R7: the run scripts cut the refusal message at
240 characters, and the records lost which residual had stalled). Read after the step returned, from its `info` and the
last iterate's Theta (w.theta; the rejection restores the state only): `reason`, `stalled` (:linear for :linear_failed,
:nonfinite, :bounds; otherwise the residual(s) above tolerance at the last iterate: :momentum, :theta, :both; on the
segregated loop, when every relaxed iterate met both tolerances but the final unrelaxed iterate's re-check failed and R then
never fell the further decade the next final iteration waits for, the residual that re-check failed on: :momentum_final,
:theta_final, :both_final; :none otherwise),
the attempt's t (the committed time it started from) and dt, its outer iterations, its last R and the two parts R_mom =
momentum residual / mom_tol and R_theta = max |Theta| / theta_tol (`Rm`, `Rt`, with their histories), the last linear
solve's iterations (FCG on the segregated loop, FGMRES on the coupled step) and, on the coupled step, its estimated final
scaled residual; max |Theta|; in a domain without a pressure face (bc.npdir = 0), the pressure reference cell, Theta there,
the domain's total defect sum V Theta (which flux-form transport conserves to round-off) and the reference cell's volume.
Scalar reductions and two one-element reads; nothing here changes the state or the numerics.
"""
function rejection_record(reason::Symbol, info, st::PBState, m::PBMesh, bc, w, cfg)
    be = cfg.backend
    Rmh = haskey(info, :Rm) ? copy(info.Rm) : Float64[]; Rth = haskey(info, :Rt) ? copy(info.Rt) : Float64[]
    Rm = isempty(Rmh) ? NaN : Rmh[end]; Rt = isempty(Rth) ? NaN : Rth[end]
    ff = haskey(info, :final_fail) ? info.final_fail : :none
    Ru = haskey(info, :Rturb) && !isempty(info.Rturb) ? info.Rturb[end] : 0.0        # W1: the k-omega residual (0 laminar)
    stalled = reason === :linear_failed ? :linear : reason === :nonfinite ? :nonfinite : reason === :bounds ? :bounds :
              reason in (:turb_linear_failed, :turb_not_converged) ? :turbulence :
              (Rm > 1 && Rt > 1) ? :both : Rm > 1 ? :momentum : Rt > 1 ? :theta : Ru > 1 ? :turbulence :
              ff === :none ? :none : Symbol(ff, :_final)
    thmax = pb_maxabs(w.theta; backend=be)
    rref = bc.npdir == 0 ? (cfg.p_ref_cell > 0 ? cfg.p_ref_cell : 1) : 0
    thref = rref > 0 ? _host(view(w.theta, rref:rref))[1] : NaN
    vref = rref > 0 ? _host(view(m.vol, rref:rref))[1] : NaN
    _launch!(_k_mul3!, be, m.nc, w.scr, w.theta, m.vol); tot = pb_sum(w.scr; backend=be)
    lin = isempty(info.fcg) ? 0 : info.fcg[end]
    linres = haskey(info, :linres) && !isempty(info.linres) ? info.linres[end] : NaN
    (reason=reason, stalled=stalled, t=st.t, dt=st.dt, outer=info.outer, R=isempty(info.R) ? NaN : info.R[end], Rm=Rm, Rt=Rt,
     Rm_hist=Rmh, Rt_hist=Rth, linear_its=lin, linres=linres, theta_max=thmax, ref_cell=rref, theta_ref=thref, theta_total=tot,
     vol_ref=vref)
end

"The refusal message's account of a rejection record (appended in full; `rejection_record`)."
function rejection_text(r, cfg)
    s = @sprintf("the last attempt (from t %.10g at dt %.6g; %s after %d outer iterations): stalled residual %s; ", r.t, r.dt,
                 r.reason, r.outer, r.stalled) *
        @sprintf("R_mom %.6g (momentum residual %.4e, mom_tol %.3g), ", r.Rm, r.Rm*cfg.mom_tol, cfg.mom_tol) *
        @sprintf("R_theta %.6g (max |Theta| %.4e, theta_tol %.3g)", r.Rt, r.theta_max, cfg.theta_tol)
    if r.ref_cell > 0
        s *= @sprintf("; no pressure face: Theta in the pressure reference cell %d is %.4e%s; ", r.ref_cell, r.theta_ref,
                      abs(r.theta_ref) == r.theta_max ? " (the largest)" : "") *
             @sprintf("the domain's total defect sum V Theta is %.4e, which over that cell's volume %.4e is %.4e ", r.theta_total,
                      r.vol_ref, r.theta_total/r.vol_ref) *
             "(flux-form transport conserves the total to round-off)"
    end
    s * @sprintf("; the last linear solve: %d iterations", r.linear_its) *
        (isnan(r.linres) ? "" : @sprintf(", estimated final scaled residual %.3e", r.linres))
end
pbcav_march!(state, mesh, config) = _nyi(:pbcav_march!, "stage P3 for the driver's PB_STOP=run; verification cases call pbcav_march!(st, mesh, bc, cfg::PBConfig)")

# ---- the odd-even pressure norm (plan 28.11, 10.2, NP-F7; 15 Oct 2026): a HOST DIAGNOSTIC, never in a step -------------
"""
    face_hops(mesh) -> Vector{Int}

Each cell's distance in face hops from the boundary: 1 for a cell with a boundary face, then 1 + the smallest of its face
neighbours'; typemax(Int) ÷ 2 everywhere on a mesh without boundary faces (the doubly periodic box).
"""
function face_hops(m::PBMesh)
    own = _host(m.own); nei = _host(m.nei); h = fill(typemax(Int) ÷ 2, m.nc)
    for q in _host(m.bown); h[q] = 1; end
    changed = m.nb > 0
    while changed
        changed = false
        for f in eachindex(own)
            a, b = own[f], nei[f]
            h[a] + 1 < h[b] && (h[b] = h[a] + 1; changed = true)
            h[b] + 1 < h[a] && (h[a] = h[b] + 1; changed = true)
        end
    end
    h
end

"""
    oddeven_operators(mesh, bc, p, cfg; work=nothing) -> (Lc, Lw, Lc_written)

The two discrete Laplacians of plan 28.11 on a cell pressure p (host vector), per unit volume, host vectors:
  Lc  the COMPACT one, (1/V_P) sum_f [ |S_f|^2/(S_f.d_f) (p_N - p_P) + nonorth k_f . (grad p)bar_f ], the face pressure
      gradient the momentum interpolation uses (`_k_rc_face!`: its two-point part and its non-orthogonal part with the
      linearly interpolated least-squares gradient; on a boundary face p_N is the solver's own face pressure, the condition's
      value on a pressure face, else the owner's extrapolation, and the non-orthogonal part takes the owner's gradient);
  Lw  the WIDE one, (1/V_P) sum_f S_f . (grad p)_f, (grad p)_f the skewness-corrected face interpolation (`face_value`, each
      component with its internal least-squares gradient) of the solver's Green-Gauss gradient; (grad p)_b = (grad p)_P at a
      boundary face;
  Lc_written  the compact one exactly as 28.11 writes it, the two-point part alone (no k_f term).
WHY Lc CARRIES THE k_f TERM (a correction of 28.11's written formula, made before any datum and labelled): 28.11 registers
the check "L_c and L_w both vanish for a linear p on the distorted box"; the two-point part alone does not (for a linear p,
sum_f |S|^2/(S.d) d . grad p differs from sum_f S . grad p = 0 wherever d is not parallel to S), so the written formula cannot
meet its own registered check. With the momentum interpolation's own non-orthogonal part it does, and (L_c - L_w) p is then
sum_f fa_f [(p_N - p_P) - d_f . (grad p)_f], the momentum interpolation's dissipation. Where k_f = 0 (every face of a uniform
box: the cavity) the two are identical; Lc_written is returned so the written form is always reported beside it.
"""
function oddeven_operators(m::PBMesh, bc, p::AbstractVector, cfg; work=nothing)
    w = work === nothing ? pbwork(m, bc, cfg) : work
    be = cfg.backend; G = w.lsq
    pv = similar(w.gpx); copyto!(pv, Float64.(p))
    pressure_gradients!(w.gpx, w.gpy, w.ggx, w.ggy, w.pbf, pv, w, m, bc, cfg)
    hxx = similar(w.gpx); hxy = similar(w.gpx); hyx = similar(w.gpx); hyy = similar(w.gpx)
    for (g1, g2, phi) in ((hxx, hxy, w.ggx), (hyx, hyy, w.ggy))
        _launch!(_k_lsq_grad!, be, m.nc, g1, g2, phi, w.ubf, G.i11, G.i12, G.i22, m.cface, m.own, m.nei, m.dx, m.dy,
                 m.bdx, m.bdy, bc.kind, Int32(m.nf), LSQ_INT, Int32(0), Int32(0))
    end
    pb_sync(be)
    H(x) = _host(x)
    ph = H(pv); gx = H(w.gpx); gy = H(w.gpy); ggx = H(w.ggx); ggy = H(w.ggy); pbf = H(w.pbf)
    hxx = H(hxx); hxy = H(hxy); hyx = H(hyx); hyy = H(hyy)
    own = H(m.own); nei = H(m.nei); fw = H(m.fw); fa = H(m.fa); kx = H(m.kx); ky = H(m.ky); skx = H(m.skx); sky = H(m.sky)
    Sx = H(m.Sx); Sy = H(m.Sy); bown = H(m.bown); ba = H(m.ba); bkx = H(m.bkx); bky = H(m.bky); bSx = H(m.bSx); bSy = H(m.bSy)
    V = H(m.vol); kn = Float64(cfg.nonorth); sk = cfg.skew ? 1.0 : 0.0
    Lc = zeros(m.nc); Lw = zeros(m.nc); Ll = zeros(m.nc)
    for f in 1:m.nf
        P = own[f]; N = nei[f]; ww = fw[f]; w1 = 1 - ww
        tp = fa[f]*(ph[N] - ph[P])
        np_ = kn*(kx[f]*(ww*gx[P] + w1*gx[N]) + ky[f]*(ww*gy[P] + w1*gy[N]))
        gfx = face_value(ggx, hxx, hxy, f, own, nei, fw, skx, sky, sk)
        gfy = face_value(ggy, hyx, hyy, f, own, nei, fw, skx, sky, sk)
        wd = Sx[f]*gfx + Sy[f]*gfy
        Lc[P] += tp + np_; Lc[N] -= tp + np_; Ll[P] += tp; Ll[N] -= tp; Lw[P] += wd; Lw[N] -= wd
    end
    for b in 1:m.nb
        P = bown[b]
        tp = ba[b]*(pbf[b] - ph[P]); np_ = kn*(bkx[b]*gx[P] + bky[b]*gy[P])
        Lc[P] += tp + np_; Ll[P] += tp; Lw[P] += bSx[b]*ggx[P] + bSy[b]*ggy[P]
    end
    (Lc ./ V, Lw ./ V, Ll ./ V)
end

"N_oe = ||(L_c - L_w) p|| / ||L_c p|| over the cells of `mask`, volume-weighted L2 (plan 28.11); NaN on an empty mask."
function oddeven_ratio(Lc, Lw, V, mask)
    any(mask) || return NaN
    sqrt(sum(V[mask] .* (Lc[mask] .- Lw[mask]).^2)/sum(V[mask].*(Lc[mask].^2)))
end

"""
    oddeven_norm(mesh, bc, p, cfg; work=nothing, extra=nothing) -> NamedTuple

Plan 28.11's odd-even norm of p: `scored` over the cells at least three face hops from every boundary (`face_hops` >= 3;
every cell of a mesh without boundary), `all` over every cell, `extra` over the scored cells that are also in the host mask
`extra` (the cavity's figure without the lid-corner squares), and `written` / `written_all` the same with 28.11's written
compact operator (the two-point part alone; see `oddeven_operators`). Also the number of scored cells.
"""
function oddeven_norm(m::PBMesh, bc, p::AbstractVector, cfg; work=nothing, extra=nothing)
    Lc, Lw, Ll = oddeven_operators(m, bc, p, cfg; work)
    V = _host(m.vol); sc = face_hops(m) .>= 3; al = trues(m.nc)
    ex = extra === nothing ? falses(m.nc) : (sc .& extra)
    (scored=oddeven_ratio(Lc, Lw, V, sc), all=oddeven_ratio(Lc, Lw, V, al), extra=oddeven_ratio(Lc, Lw, V, ex),
     written=oddeven_ratio(Ll, Lw, V, sc), written_all=oddeven_ratio(Ll, Lw, V, al), nscored=count(sc))
end
