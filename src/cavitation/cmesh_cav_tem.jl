# ---------------------------------------------------------------------------------------------
# PATH STATUS: **MIXED -- read this before using anything here.**
#   * CONFIGURATION half is LIVE in production: build_species_ka(s, tem; ...) reads p_v,
#     rho_l_sat, rho_v_sat and the rate constants from this module. A wrong docstring here has
#     already cost a correction (tem_timescales was wrong by ~50x for the branch the driver uses).
#   * STEPPING half (tem_step!) is LEGACY ONLY. The production branch never calls it.
# Map and dispatch: memory/project_docs/CAVITATION_CODE_MAP.md (checked by tools/check_code_map.jl).
# ---------------------------------------------------------------------------------------------
# Transported cavitation composition and algebraic mixture-pressure projection.
# For eos=1, vaploc=1, lq=0, alpha is consistently a reference-volume composition:
#   Rs=(1-a)*rho_l_sat+a*rho_v_sat, Y=a*rho_v_sat/Rs is vapour mass fraction.
# Current vapour volume generally differs from a away from raw saturation pressure.
# The shared local-pressure EOS is in cmesh_eos_mix.jl; pressure is represented as
# rho*CS2*T+pexc, with T a bounded carrier pedestal, not a thermodynamic temperature.
# The default commit overwrites the marched pedestal, so this is not a closed
# conservative thermal-energy model.
#
# cons=0 and cons=2 approximate Da/Dt=S in smooth flow; cons=1 instead transports
# div(a*u). These separate scalar stencils do not conserve rho*Y with the carrier
# mass flux. With optional physics off, S=kprod*max(p_v-p,0)*(1-a)
# -kdest*max(p-p_v,0)*a. The source compares guarded p with the raw parameter p_v.
# build_tem selects rate normalization (U/q or U/p_v); these scalar coefficients
# are not automatically a physical Merkle mass-transfer normalization.
# The point-implicit source and segregated flow/TEM/SST schedule do not inherit
# third-order time accuracy from the carrier SSPRK stages. Legacy paths are kept
# for reproducibility. cmesh_cav_species.jl supplies a separate restricted,
# conservative-species path with stage EOS projection and resolved phase change.
module CMeshCavTEM

using KernelAbstractions
using Printf                 # ★ the Rayleigh-Plesset realised-state banner in build_tem uses @sprintf
const KA = KernelAbstractions

include(joinpath(@__DIR__, "..", "cmesh", "cmesh_cav_config_contract.jl"))
using .CMeshCavConfigContract: validate_vaploc

export TEMState, build_tem, tem_step!, tem_project!, tem_hist_cycle!, tem_seed!, TEM_CS2, tem_ratio, tem_timescales, tem_ambient, tem_residual, tem_residual_report, tem_retune_sigma!, tem_cprofile, tem_cprofile_dens, tem_eos_name

const TEM_CS2 = 0.6666667f0
const WG = (256, 1)
# ★★★★ Rayleigh-Plesset bubble dynamics + Schnerr-Sauer nuclei (2026-08-01). Kept in its own file so
# it is CPU-unit-testable with no GPU (tools/test_rayleigh_plesset.jl validates it against the Rayleigh
# collapse time and the Minnaert frequency, both to ratio 1.00000) and so the sweep kernel does not
# grow a large inline branch -- this codebase has a documented GPU-JIT @inline SIZE-THRESHOLD failure
# from exactly that (CLAUDE.md, "GPU JIT failure on enlarged entropic D3Q39 kernel signature").
include(joinpath(@__DIR__, "..", "cmesh", "cmesh_rayleigh_plesset.jl"))
# ★ DIAGNOSTIC-ONLY toggle for the P3 bring-up failure attribution (2026-07-22 gate-results section
# of the design doc, pre-registered ladder step 2). NOT a production mode — never quote physics from
# it. TEM_P3_DIAG=1: solve the full 2x2 but apply Delta_alpha ONLY (skip the rho overwrite) —
# separates the rho-overwrite double-march channel from the alpha channel in one warm-up probe.
const P3_DIAG = Int32(parse(Int, get(ENV, "TEM_P3_DIAG", "0")))

mutable struct TEMState{A}
    alpha::A          # (NI+2g, NJ+2g) void fraction, ghosts for the upwind stencil
    alpha_new::A      # (NI+2g, NJ+2g) sweep scratch (commit copies back)
    un::A             # (NI, NJ) dual-time BDF2 history: alpha^n
    um::A             # (NI, NJ) dual-time BDF2 history: alpha^(n-1)
    p_v::Float32
    a2l::Float32
    rho_l_sat::Float32
    rho_v_sat::Float32
    kprod::Float32
    kdest::Float32
    Tmin::Float32
    Tmax::Float32
    a2v::Float32      # vapor-branch c^2 = p_v/rho_v_sat (ideal-gas vapor => bounds cavity density)
    wallis::Int32     # 0 = mass-weighted harmonic (legacy), 1 = DENSITY-weighted Wallis (Stage 1d)
    psm::Float32      # smooth-max width [pressure units] for the vapor-plateau floor
    eos::Int32        # 0 = legacy raw-linear a2l (STIFF: a2l/q~50, blows Cp up during the alpha lag)
                      # 1 = Wallis harmonic a2m(alpha) + smooth-max vapor floor (physical, DEFAULT)
    # Numerical sound-speed floor: legacy projection uses am >= sqrt(am_floor_frac*a2l).
    # Conservative Species instead rejects a state whose EOS derivative is below
    # this bound. Zero removes that artificial minimum while retaining c^2 > 0.
    # The default 1e-4 preserves legacy behavior; it is not a water calibration.
    # The 200 m/s floor in dbnsCavitatingFoam (arXiv:2009.04744, Eqs. 18-20)
    # applies to a numerical interface speed in the pressure-difference flux.
    # It does not prescribe a lower bound for the physical mixture EOS derivative.
    am_floor_frac::Float32
    # SMOOTHING WIDTH for the Merkle sign switch [pressure units]. 0 = hard switch (bitwise legacy).
    # The hard switch makes the source slope jump by kprod/kdest at dpv = 0, which diverges the dual-time
    # inner iteration once the rate is stiff AND asymmetric. A natural scale is a small fraction of p_v.
    src_smooth::Float32
    # 1 = add the PRESSURE-FEEDBACK term to the point-implicit diagonal (the full source Jacobian).
    # 0 = legacy diagonal, which omits the dominant dE/dalpha = -kprod*dp/dalpha contribution.
    src_jac::Int32
    # SEGREGATED-COUPLING UNDER-RELAXATION. Each inner iteration runs step_ka! (flow) THEN tem_step!
    # (alpha) -- a Picard iteration over two STRONGLY coupled fields: alpha sets p via the mixture EOS
    # (dp/dalpha = a2m*(rho_l-rho_v) ~ 0.905) and p drives alpha via the Merkle source. Segregated
    # iteration on strong two-way coupling is a classic divergence mechanism, and under-relaxation is the
    # standard remedy. omega = 1 => legacy (no relaxation), bitwise unchanged.
    omega::Float32
    # ★ P2a (2026-07-21): Kunz phasic-continuity-enforcing coupling (Kunz et al. C&F 29 (2000) Eq. 1c/9;
    # the (alpha/(rho*beta^2))*dp/dtau pseudo-time term whose PRESENCE his Eq. 15-16 eigenanalysis proves
    # necessary for volume-fraction/density-ratio-INDEPENDENT eigenvalues). Discrete form in the sweep:
    # den += cE, num += cE*a0 - cR*drho_flow with cE = a0*dpda/(rho*b2), cR = a0*dpdr/(rho*b2),
    # b2 = max(beta2, |u|^2), drho_flow = the flow step's just-applied pseudo-density change (rho - rk_r).
    # Vanishes identically at pseudo-convergence => fixed point untouched. enf=0 bitwise legacy.
    enf::Int32
    beta2::Float32    # k_beta*U_inf^2 (Kunz beta^2/U_inf^2 ~ 10); per-cell floor of the beta^2 scale
    # ★ P3a (2026-07-21): 1 = CONSERVATIVE alpha transport div(alpha*u) via upwinded face volume-fluxes.
    # The legacy non-conservative u.grad(alpha) silently drops alpha*div(u) — nonzero exactly in cavity
    # growth/collapse regions — and conservative form is the P2a-lesson PREREQUISITE for any Kunz-Gamma
    # coupling. NOTE: with the source treatment unchanged this is a MODEL-FORM change (Kunz Eq. 5 pairs
    # u.grad(alpha) with a rho_m/(rho_l*rho_v) source coefficient); judge by stability character +
    # cavity plausibility, and re-validate physics before any claim. 0 = bitwise legacy.
    cons::Int32
    # ★ P3 (2026-07-22): the FULL-GAMMA coupled (rho, alpha) Delta-form implicit-source path. Design +
    # adversarial review: memory/project_docs/2026-07-22_P3_GAMMA_DESIGN_draft.md — READ ITS HEADER BANNER; the four
    # blocking fixes B1-B4 are implemented in _k_p3_couple! (carrier-clock A11 augmentation, dpdr_G
    # floor + trust region, two-buffer alpha write, sponge-band rho-write mask). 0 = production path,
    # bitwise. 1 = coupled path: sweep+commit replaced by _k_p3_couple!; omega RETIRED; conservative
    # transport + Kunz enforcing entries + jac3 all inside the 2x2 matrix; preconditioned dtau_p
    # (CFL_p = cflp); the explicit flow march keeps the carrier dt untouched (lesson L3).
    p3::Int32
    cflp::Float32     # CFL_p of the preconditioned coupled rows (ramp 0.2 -> 5, one rung at a time)
    # ★★★★ TIER 0 (2026-08-05): alpha FACE RECONSTRUCTION order. 0 = first-order upwind (bitwise
    # legacy). 1 = minmod MUSCL. The flow runs WENO5-Z while alpha ran FIRST-ORDER UPWIND, i.e.
    # ~1500x the physical viscosity in numerical diffusion at Re=8e5 -- the nominated reason a strong
    # re-entrant jet cuts nothing, since severing a sheet needs a liquid bridge and first-order upwind
    # cannot hold one. Minmod, NOT raw WENO: alpha is BOUNDED in [0,1] and an unlimited scheme
    # overshoots with the existing clamp hiding it. Spec: memory/project_docs/2026-08-05_TIER0_ARECON_IMPLEMENTATION_SPEC.md
    arecon::Int32
    # PER-CELL |alpha change| from the last commit. There was NO alpha residual anywhere: the shed tool
    # reports rstar, which is the FLOW residual, so a tight rstar with an UNCONVERGED alpha looks like
    # success. MEASURED consequence: under-relaxation drove rstar to 1.0e-04 (control level) while the
    # cavity collapsed (lcav 0.987 -> 0.088) -- alpha was lagging further behind at every omega and the
    # flow was converging onto a state consistent with an under-developed void.
    res::A
    # ★ NCG (2026-07-28): nominal free non-condensable-gas volume fraction (frozen content), entering
    # ONLY as a third Wallis compressibility term ncg/p_v in _a2m_mix. 0 = bitwise legacy (no gas).
    # Requires eos=1 && wallis=1; build_tem recalibrates rho_l_sat so sigma keeps its meaning.
    ncg::Float32
    # ★ DK (2026-07-28): Delannoy-Kueny-class c(alpha) profile (see _a2m_mix). dk_cmin2 = cmin^2 in
    # code units (0 = bitwise legacy Wallis); dk_a2v = vapour-end stiffness cap. Requires
    # eos=1 && wallis=1; mutually exclusive with ncg.
    dk_cmin2::Float32
    dk_a2v::Float32
    # ★ VOID PSEUDO-TIME BOOST (2026-07-29): cap on dtl_alpha/dtf. 1 = bitwise legacy (the void
    # marched on the carrier acoustic step, which froze the near-wall cells ~1000x).
    dtboost::Float32
    # ★★ VAPOUR-SIDE EOS BOUND (2026-07-29, audit round 2 — memory/project_docs/2026-07-29_AUDIT_ROUND2_VERDICT.md).
    # >0 engages `_pmix_vb`: p >= a2v*rho (the ideal-gas vapour law) at EVERY EOS evaluation site.
    # 0 = bitwise legacy (the unbounded frozen-alpha linearisation, which crosses zero at
    # rho = rho_v_sat/2 and diverges negative below it). APPENDED AT THE END on purpose — the
    # constructor is positional, so inserting mid-struct silently misassigns every following field.
    vapbound::Float32
    # ★★★ LOCAL-PRESSURE VAPOUR TERM (2026-07-30). >0 replaces the FROZEN-p_v vapour term a/p_v of the
    # wallis=1 Wallis sum with the correct isothermal identity rho_v*a_v^2 = p at the LOCAL pressure,
    # i.e. a/p, and solves the resulting IMPLICIT mixture EOS EXACTLY in closed form (`_pmix_vl`).
    # This is not a floor: the alpha=1 limit DEGENERATES to the ideal gas p = a2v*rho, so the vapour
    # bound this campaign twice failed to impose is DERIVED, not imposed. 0 = bitwise legacy.
    # APPENDED AT THE END for the same positional-constructor reason as `vapbound`.
    vaploc::Float32
    # ★★★ DUAL-TIME VOID PSEUDO-STEP BOOST (2026-07-31). `dtboost` above is GATED to dtp==0, so the
    # void's pseudo-step in the TIME-ACCURATE march is the flow's acoustic dtf. That gate is why the
    # inner iteration cannot contract in the TRANSPORT region. From the sweep below the iteration is
    # alpha^(m+1) = num/den, so the per-sweep contraction factor is
    #     rho = 1/den = 1/(1 + dtl*(E + D + jacp + 1.5/dtp))
    # and convergence requires dtl*Lambda >~ 1. Inside the cavity the stiff source supplies
    # Lambda ~ 4e3 (the pressure-feedback jacp) and the iteration converges. Where vapour is ADVECTED
    # -- E = D = jacp = 0, i.e. exactly where a shed cloud lives -- the only diagonal term left is
    # 1.5/dtp, and dtl*1.5/dtp ~ 3e-4..3e-2. That is 1e2-1e4 sweeps per decade, at NINNER = 100.
    # ⇒ the void converges where the SOURCE dominates and not at all where ADVECTION does, which is
    # precisely the observed split (correct steady cavity, no shedding) and predicts the refinement
    # paradox for free, since refining the wall shrinks dtf and therefore dtl.
    # 1 = BITWISE LEGACY (gate intact). >1 allows the boost in dual time as well, still bounded by the
    # void's own advective CFL, and still FIXED-POINT-PRESERVING (dtl cancels identically at the fixed
    # point -- proven for both the steady and the dual-time form).
    # ⚠ THE dtp==0 GATE WAS INSTALLED ON A MEASURED FAILURE (line ~960: dtboost=1000 in dual time gave
    # V 400x, rstar 640x control). That failure is real and is the coupled-iteration stability limit --
    # relaxing alpha far faster than p destabilises the segregated split, which is the Nordstrom
    # eigenvalue condition failing. But it was measured ONLY at the extreme; 3, 10, 30 were never
    # tried. This field exists to find whether a stable window exists between "does not converge" and
    # "decouples". If none does, that IS the coupled-solver justification, measured rather than argued.
    dtb_dt::Float32
    # ★★★ POINT-IMPLICIT OUTFLOW (2026-07-31). Blend factor on moving the advection OUTFLOW coefficient
    # onto the point-implicit diagonal. 0 = BITWISE LEGACY (advection fully explicit, as it has been for
    # the whole campaign); 1 = outflow fully implicit. Fixed-point-preserving EXACTLY (proof at the use
    # site), so only the relaxation path changes, never the answer.
    # This is the NO-SEQUENCING half of the swept-transport fix the floor analysis nominates: the void
    # residual's floor depends on `cflp` alone (NINNER inert across 8x at three cflp values, Cprod
    # non-monotone, rstar flat) as floor ~ cflp^-0.707, and `cflp` is capped near 1 because the
    # transport is EXPLICIT. Making the outflow implicit raises that cap.
    # APPENDED AT THE END for the same positional-constructor reason as `vapbound`/`vaploc`/`dtb_dt`.
    advimp::Float32
    # ★★★ LINE GAUSS-SEIDEL IN i (2026-07-31). 0 = BITWISE LEGACY (Jacobi; ndrange (NI,NJ), one cell
    # per thread). 1 = ndrange (1,NJ), one thread per j-line marching i sequentially and reading the
    # ALREADY-UPDATED upwind neighbour, so information crosses a whole line in ONE sweep instead of one
    # cell per iteration. For first-order upwind the streamwise system is LOWER BIDIAGONAL, so the
    # forward march IS the exact implicit solve -- a forward substitution, not a Thomas solve.
    # Fixed point unchanged: at convergence alpha_new == alpha, so reading either is identical.
    # APPENDED AT THE END for the same positional-constructor reason as every flag before it.
    gs::Int32
    # ★★★★ RAYLEIGH-PLESSET BUBBLE DYNAMICS + SCHNERR-SAUER NUCLEI (2026-08-01).
    # Physics and its validation live in src/cmesh/cmesh_rayleigh_plesset.jl and
    # tools/test_rayleigh_plesset.jl (Rayleigh collapse time and Minnaert period both reproduced to
    # ratio 1.00000). Everything below is the STATE that model needs; the model itself is not
    # duplicated here.
    #
    # rp = 0 => BITWISE LEGACY MERKLE. The branch in `_k_tem_sweep!` is not taken, `rdot` is never
    # read or written, and the arrays below are 1x1 stubs. rp = 1 => the alpha source becomes
    # (dalpha/dR)*Rdot with Rdot carrying its own transported second-order dynamics.
    #
    # ⚠ WHAT CHANGES BESIDES THE SOURCE. Under Merkle the pressure acts on the alpha equation directly
    # and supplies the dominant point-implicit diagonal term jacp ~ 4e3 -- the term that makes the void
    # iteration contract at all. Under R-P the pressure acts on the Rdot equation instead, so the alpha
    # equation's pressure-feedback Jacobian is genuinely ZERO and the stiffness MOVES. `src_jac` is
    # therefore inert on the alpha equation when rp > 0, by construction and not by oversight; the Rdot
    # equation carries its own point-implicit treatment. Expect `ares` to behave differently and DO NOT
    # compare it across the two models as though it measured the same thing.
    rp::Float32
    rdot::A           # (NI+2g, NJ+2g) bubble-wall radial velocity -- the field that carries the INERTIA
    rdot_new::A       # (NI+2g, NJ+2g) sweep scratch
    rdot_n::A         # (NI, NJ) dual-time BDF2 history: rdot^n
    rdot_m::A         # (NI, NJ) dual-time BDF2 history: rdot^(n-1)
    rp_n0::Float32    # nuclei number density per unit LIQUID volume [1/chord^3]
    rp_R0::Float32    # nucleus seed radius [chord] -- also the floor on R, and the gas reference radius
    rp_S::Float32     # surface tension coefficient S [pressure*length]
    rp_pg0::Float32   # non-condensable gas partial pressure at R = R0. ★ THE RESTORING FORCE: with
                      #   pg0 = 0 the bubble has NO natural frequency (omega_0^2 <= 0) and is a runaway,
                      #   not an oscillator. See T6 of the unit test.
    rp_kappa::Float32 # polytropic exponent for that gas (1 isothermal, 1.4 adiabatic diatomic)
    rp_nu::Float32    # liquid kinematic viscosity used by the R-P damping term 4 nu Rdot/R
    # ★★★ THERMAL DEPRESSION OF THE VAPOUR PRESSURE (Clausius-Clapeyron, 2026-08-01).
    # p_v(T) = p_v_ref * exp( -(L/R_v) * (1/T - 1/T_ref) ). Evaporation cools the interface, which
    # LOWERS p_v, which suppresses further evaporation -- the thermal-delay effect that governs
    # cavitation in thermosensitive fluids (cryogens, hot water) and is the reason LOX/LH2 turbopump
    # cavitation looks nothing like cold-water cavitation.
    # 0 = bitwise legacy (constant p_v).
    # ⚠⚠ SCOPE, STATED HERE SO IT IS NOT OVER-READ. This modifies the SOURCE driving potential only,
    # NOT the EOS reference state. `rho_l_sat` and `rho_v_sat` are calibrated at the reference p_v in
    # build_tem; making them cell-local means re-deriving the saturation LINE, which needs a real
    # liquid energy equation -- Tier 3. Worse, the T this reads is the CARRIER PEDESTAL railed to
    # [Tmin,Tmax], not a thermodynamic liquid temperature. So this term is STRUCTURALLY present and
    # dimensionally correct but CANNOT BE QUANTITATIVELY TRUSTED until Tier 3 supplies a real T.
    # For cold water the effect is negligible anyway (the implementation plan says so explicitly),
    # which is precisely why it is safe to land it in this partial form and wrong to quote it.
    tvp::Float32
    tvp_LRv::Float32  # L/R_v in lattice temperature units
    tvp_Tref::Float32 # reference temperature at which p_v == the build-time p_v
    # ★★★ DRIFT-FLUX PHASE SLIP (2026-08-01). The homogeneous assumption forces vapour and liquid to
    # share ONE velocity field. Real vapour, being ~100x lighter here (1000x in water), is accelerated
    # by a pressure gradient far more strongly than the liquid, so a cloud does not simply convect with
    # the mixture. Algebraic drift-flux closure: the relative velocity is the pressure gradient acting
    # on the density difference over a drag relaxation time,
    #       u_r = -slip_tau * (1/rho_v_sat - 1/rho) * grad(p)
    # and the void equation gains the drift flux  div( alpha (1-alpha) u_r ).  The alpha(1-alpha)
    # factor is the standard drift-flux weighting: it vanishes in both pure phases, where a RELATIVE
    # velocity is meaningless.
    # 0 = bitwise legacy (homogeneous, no slip).
    # ⚠ The implementation plan judges slip NOT to be the shedding discriminator, on the good grounds
    # that published homogeneous solvers shed. It is implemented because it is real two-phase physics
    # we did not have, not because it is expected to unlock anything.
    slip::Float32
    slip_tau::Float32 # drag relaxation time [lattice time]
    # ★★★ COLLECTIVE / CLOUD BUBBLE DYNAMICS (d'Agostino & Brennen 1989, 2026-08-02).
    # Scales Brennen's cloud interaction parameter beta = alpha(1-alpha)(A/R)^2, with the cloud
    # radius A taken as the LOCAL void-gradient length alpha/|grad alpha|. The bubble-wall velocity is
    # then divided by sqrt(1+beta): a bubble inside a cluster must displace the surrounding BUBBLY
    # medium rather than pure liquid, so it responds more slowly.
    # 0 = bitwise legacy (beta = 0, factor exactly 1).
    # ⚠ THIS IS THE ONLY ABSENT PHYSICS ITEM WHOSE TIMESCALE IS NEAR THE FLOW'S -- collective modes
    # land at 0.026-0.26 CTU against a 3.2 CTU shedding period, where a single 1 um bubble is at
    # 1e-5 CTU and provably irrelevant. And unlike the Cprod lag ladder (which moved every cell by the
    # same factor and produced no oscillation at any rung), this lag is SPATIALLY VARYING: dense cores
    # slow down, dilute edges do not. The uniform ladder could not produce that and did not test it.
    cloudint::Float32
    # ★ S3c POLYDISPERSE NUCLEI: power-law spectrum n(R) ~ R^-poly_e on [poly_Rmin, poly_Rmax].
    # Only nuclei above the Laplace critical radius R_crit = 2S/(p_v-p) can be activated, so
    # inception becomes CONTINUOUS in suction depth instead of a single threshold. 0 = legacy.
    poly::Float32; poly_e::Float32; poly_Rmin::Float32; poly_Rmax::Float32
    # ★ S3d HERTZ-KNUDSEN: kinetic ceiling on the interfacial mass flux. 0 = legacy (unbounded).
    hk::Float32
    # ★ S3e HETEROGENEOUS WALL NUCLEATION: surface excess decaying into the bulk. 0 = legacy.
    wallnuc::Float32; wall_decay::Float32
    # ★ S3f BUBBLE-INDUCED TKE production coefficient. WIRED into the SST k equation (2026-08-02) via
    # the precomputed `s.btke_src` array -- see the KSol field comment for why an array and not a
    # direct call. 0 = legacy (the array stays zero, so the k equation is bitwise unchanged).
    btke::Float32
    # ★ S3k SINGHAL TURBULENT VAPOUR THRESHOLD (Singhal et al. 2002, "full cavitation model").
    # Turbulent pressure fluctuations let vapour form ABOVE the static saturation pressure:
    #       p_v_eff = p_v + 0.5 * p_turb ,      p_turb = 0.39 * rho * k
    # so the coefficient applied here is 0.195*pvturb*rho*k, with pvturb = 1 reproducing the published
    # constant exactly and 0 = bitwise legacy.
    #
    # WHY IT MATTERS HERE, and why it is not just another inert term. p_v = 2.333e-3 in this campaign's
    # units. In the closure shear layer -- exactly where the re-entrant jet is generated and where a
    # cloud would be severed -- k is large, and 0.195*rho*k is COMPARABLE TO p_v ITSELF at only ~10%
    # turbulence intensity. This is the one standard cavitation-model term that is both absent here and
    # spatially concentrated where the missing dynamics live: it raises the effective driving in the
    # shear layer WITHOUT touching the attached sheet, which is the asymmetry every other term lacked.
    # It is also in every commercial code (Fluent, CFX, STAR-CCM+), so its absence is a genuine gap and
    # not an exotic addition.
    # ⚠ It reads the SST `k`, which is a real transported turbulence quantity here -- unlike `T`, which
    # is a railed pressure pedestal. So this term does NOT inherit the thermal-p_v caveat.
    pvturb::Float32
    # ★★★ S3k -- VORTEX THRESHOLD PRESSURE (Le & Tran, JAFM 15(2) 551-561, 2022, Eq. 14).
    #
    #     p_v* = p_v + C_v * mu_eff * |dv/dx - du/dy| ,   C_v = 10 their reference value
    #
    # `pvvort` IS C_v. 0 => the term is never read and the physics is bitwise legacy.
    #
    # WHY, on positive evidence (battery 6/6b, pre-registered): our thickness deficit is not a RATE
    # deficit. Over Cprod 100 -> 3000 the cavity reaches PURE VAPOUR (amax 0.751 -> 0.9999) while
    # e/c stays FLAT at ~0.038-0.040 against Le/Franc/Michel's e/c > 0.05. Rate densifies the layer;
    # it does not thicken it. Le & Tran report the two threshold forms differ in exactly that respect:
    # the Singhal form (`pvturb` above) evaporates "only at the leading edge and behind the cavity
    # trailing edge"; the vortex form evaporates "at the leading edge AND ALONG THE ATTACHED CAVITY
    # SURFACE". Their case is NACA0015 at alpha = 8 deg, Re 1.3e6, grid 309x85 -- our angle of attack
    # on a comparable grid.
    #
    # ⚠ C_v = 10 is THEIR value in THEIR normalisation and does NOT transfer. `s.pvort` carries the
    # kinematic group (nu+nu_t)*|Omega| and is written unconditionally, so the realised magnitude of
    # rho*(nu+nu_t)*|Omega| / p_v can be MEASURED at pvvort = 0 before any value of C_v is chosen.
    # That is the admissibility check this campaign now requires before a source term goes on a GPU.
    pvvort::Float32
    # ★★★ S3g STEP 1 -- STOP DISCARDING THE MARCHED ENERGY. `tfree = 0` is the legacy behaviour:
    # `_k_tem_commit!` recomputes T from the MIXTURE PRESSURE every inner iteration,
    #       T = clamp(p/(rho*cs2), Tmin, Tmax)
    # which throws away the temperature `_k_update!` just marched from the CONSERVED ENERGY
    # (E0 = rho*(eint*T + |u|^2/2)). So the solver HAS an energy equation and the cavitation layer
    # overwrites its output -- T is half a pressure representation, and Tmin/Tmax are clamps on a
    # PRESSURE, not on a temperature.
    #
    # MEASURED CONSEQUENCE (tools/measure_pedestal_rail.jl): T is railed at Tmin = 1.2 in every state
    # sampled, so the PDP split p = rho*cs2*T + pexc becomes a difference of two large numbers --
    # 294x cancellation at the CAVITY CLOSURE, which is precisely where the re-entrant jet is
    # generated. ~2.5 of Float32's ~7 decimal digits are consumed there.
    #
    # `tfree > 0` keeps the marched T and lets `pexc` carry only the residual. It does NOT change the
    # total pressure -- pexc is redefined against the new pedestal in the same line -- so this is a
    # REPRESENTATION change, not a physics change, and its whole purpose is to shrink the
    # cancellation. It is the smallest testable increment toward the two-population energy equation.
    #
    # ⚠ NOT a shedding fix and not claimed as one: Frikha's incompressible barotropic solver has no
    # energy equation at all and sheds. This is a correctness-and-precision item that also unblocks
    # thermal p_v, Van Driest, and the density ratio.
    # ⚠ UNTESTED AT WRITE TIME. The marched T under the mixture EOS may not be a meaningful
    # temperature; if it is not, pexc could get WORSE. That is exactly why it is a knob and why the
    # acceptance test is the measured cancellation ratio, not an opinion.
    tfree::Float32
    # ★★★★ TIER 3 (2026-08-05) -- `lq`: NON-LINEAR LIQUID BRANCH. 0 = BITWISE LEGACY, verified
    # against the unmodified module over 402,201 (alpha, rho) states in all three returned values
    # (tools/verify_tier3_lq.jl GATE 1, which carries a self-test proving the comparator can fail).
    #
    # Gives the liquid a compliance that GROWS as the pressure falls, 𝓛(p) = 1/(rho_l_sat*a2l) + lq/p,
    # i.e. dissolved non-condensable gas of volume fraction `lq` held at the LOCAL pressure -- the
    # local-pressure form of the frozen-p_v `ncg` term (Singhal 2002). Derivation, and proofs that
    # properties P1/P3/P5 and the monotonicity and Jacobian-clamp invariants all survive, are in the
    # `_pmix_vl` banner in cmesh_eos_mix.jl.
    #
    # WHY IT EXISTS. Stiffening the liquid EXTINGUISHES the cavity -- measured on the a2l ladder,
    # amax 0.992 / 0.742 / 0.000 at c_liq/U = 5 / 10 / 15 -- and that is a THEOREM for a linear
    # branch, not a bug (docs/2026-08-05_TIER3_...md §1). `lq` is the second knob that breaks the
    # weld: MEASURED at c_liq/U = 15, where the cavity died completely, lq = 0.005 restores the
    # near-p_v compliance to 2.56x the legacy value while the liquid stays 3x STIFFER than legacy at
    # ambient. That is the decoupling, and a linear branch provably cannot provide it.
    lq::Float32
    # ★★★★ TIER 4 (2026-08-05) -- THERMODYNAMIC MIXTURE TEMPERATURE. 0 = BITWISE LEGACY.
    #
    # THE DEFECT, in the solver's own code. `_k_tem_commit!` ends with
    #     Tn = tfree > 0 ? T[ia,ja] : p/(r*TEM_CS2);   Tn = clamp(Tn, Tmin, Tmax);   T[ia,ja] = Tn
    # so the carrier temperature is HALF A PRESSURE REPRESENTATION, and Tmin/Tmax are clamps on a
    # PRESSURE. `tfree` removes the overwrite but NOT the clamp -- which is the recorded, mechanical
    # reason latent heat measured EXACTLY inert twice: evaporative cooling is erased by the next line.
    #
    # WHY THE CLAMP CANNOT SIMPLY BE REMOVED, and why this is therefore a TWO-POPULATION problem.
    # The carrier T is a LATTICE quantity: Tmax enforces the 3:1 density-ratio cap the KFVS
    # equilibrium needs, and Tmin keeps the pedestal positive. Deleting the clamp breaks the carrier.
    # The fix is not to unrail the carrier but to STOP ASKING ONE FIELD TO BE TWO THINGS.
    #
    # WHAT THIS ADDS. A separate THERMODYNAMIC temperature, carried as a DEVIATION `dTm` from the
    # reference `tvp_Tref`, so that:
    #   - it is NOT clamped to the carrier band, and can therefore go cold under evaporation;
    #   - it is zero-initialised and zero in the ghost band, which means "ambient" with no
    #     initialisation pass and no new boundary condition -- the class of omission that has voided
    #     runs here before (new transported variable + old initialisation/BCs);
    #   - it carries the LATENT HEAT of the phase change it is coupled to;
    #   - it feeds the Clausius-Clapeyron p_v, which until now read the RAILED CARRIER PEDESTAL. The
    #     `tvp` docstring says exactly that: "it reads the railed carrier pedestal, not a
    #     thermodynamic T ... NOT quotable until Tier 3/4". This is what makes tvp quotable.
    #
    # tmix_lat lumps L*rho_v/c_p into one coefficient, so the source is -(tmix_lat/rho)*S_alpha with
    # S_alpha = E*(1-alpha) - D*alpha the volumetric vapour production rate: evaporation ABSORBS heat
    # and cools the cell, condensation warms it. tmix_lat = 0 with tmix = 1 is a pure passive-scalar
    # control arm -- the transport without the thermodynamics -- which is the right null test.
    tmix::Float32
    tmix_lat::Float32
    dTm::A            # (NI+2g, NJ+2g) thermodynamic temperature DEVIATION from tvp_Tref
    dTm_new::A        # (NI+2g, NJ+2g) Jacobi scratch (in-place would be a GPU race)
end

"""Build the TEM material and phase-change state.

The default retains the historical linear ambient-density approximation and
`rho_inf/rho_v_sat` ratio convention, including the optional kinetic vapor floor.
With `exact_reference=true`, the local-pressure liquid EOS is inverted for
`p_ambient = p_v + sigma*q_inf`, including its pressure smoothing, and `ratio`
means `rho_l_sat/rho_v_sat`. This option requires the plain local-pressure EOS
and no vapor floor. Material constants are stored in Float32, so the realized
pressure must still be read from `tem_ambient`; precision is limited at low Mach.
Neither convention calibrates physical material properties or phase-change rates.
"""
function build_tem(s; sigma, Cprod::Float64=1.0, Cdest::Float64=1.0, ratio::Float64=3.0,
                   vfloor::Bool=true, Tmin::Float32=0.3f0, Tmax::Float32=3.0f0,
                   eos::Int=1, cmin2_frac::Float64=1e-3, psm_frac::Float64=0.05,
                   a2l_frac::Float64=1.4, eps_pv::Float64=-1.0, wallis::Int=0,
                   am_floor_frac::Float64=1e-4, rate_ref_pv::Bool=false,
                   src_smooth_frac::Float64=0.0, src_jac::Int=0, omega::Float64=1.0,
                   enforce::Int=0, kbeta::Float64=10.0, conserv::Int=0,
                   p3::Int=0, cflp::Float64=0.2, arecon::Int=0, ncg::Float64=0.0,
                   dk_cmin_U::Float64=0.0, dk_a2v_U::Float64=10.0, dtboost::Float64=1.0,
                   # No safe universal default exists: vaploc=1 is the repaired production branch
                   # but requires wallis=1, while many historical experiments deliberately used 0.
                   # Every caller must choose explicitly so omission cannot abort late or change EOS.
                   vapbound::Float64=0.0, vaploc::Float64, dtb_dt::Float64=1.0,
                   advimp::Float64=0.0, gs::Int=0,
                   # ★★★★ Rayleigh-Plesset / Schnerr-Sauer. rp=0 => BITWISE LEGACY MERKLE.
                   # Defaults below are the standard Schnerr-Sauer water values converted to this
                   # solver's units by the caller-supplied chord; they are only USED when rp>0.
                   rp::Float64=0.0, rp_n0::Float64=-1.0, rp_R0::Float64=-1.0,
                   rp_S::Float64=0.0, rp_pg0::Float64=0.0, rp_kappa::Float64=1.4,
                   tvp::Float64=0.0, tvp_LRv::Float64=-1.0, tvp_Tref::Float64=-1.0,
                   slip::Float64=0.0, slip_tau::Float64=0.0, cloudint::Float64=0.0,
                   poly::Float64=0.0, poly_e::Float64=3.0, poly_Rmin::Float64=-1.0,
                   poly_Rmax::Float64=-1.0, hk::Float64=0.0, wallnuc::Float64=0.0,
                   wall_decay::Float64=3.0, btke::Float64=0.0, pvturb::Float64=0.0, pvvort::Float64=0.0,
                   tfree::Float64=0.0, chord_m::Float64=0.150, lq::Float64=0.0,
                   tmix::Float64=0.0, tmix_lat::Float64=0.0, exact_reference::Bool=false)
    vaploc = validate_vaploc(vaploc)
    # lq ADMITTED under exact_reference (2026-09-24, wedge remedy): the inversion below now carries
    # the lq compliance term itself (bitwise the lq=0 arithmetic when lq==0), so the requested sigma
    # keeps its meaning at lq>0. ncg stays excluded (the closed-form root does not carry it).
    exact_reference && !(eos==1 && wallis==1 && vaploc==1.0 && lq>=0.0 && ncg==0.0 && !vfloor) &&
        throw(ArgumentError("exact reference requires plain local-pressure EOS and vfloor=false"))
    U = sqrt(Float64(s.uxi)^2 + Float64(s.uyi)^2)
    q = 0.5*Float64(s.rho_inf)*U^2
    p_inf = Float64(s.rho_inf)*Float64(TEM_CS2)*Float64(s.T_inf)
    # Positive eps_pv selects p_v/(sigma*q); negative eps_pv retains the
    # historical pedestal-pressure convention. This is a parameterization,
    # not a temperature-dependent physical saturation-pressure law.
    p_v = eps_pv >= 0.0 ? eps_pv*sigma*q : p_inf - sigma*q
    # Liquid stiffness in solver pressure/density units. The default 1.4 uses
    # the legacy acoustic scale, not calibrated water properties. Actual sound
    # speed depends on the EOS state; read it from the mixture derivative.
    # Stored alpha is reference composition, not current vapor volume fraction.
    a2l = a2l_frac*Float64(TEM_CS2)
    # Historical linear ambient-density approximation. The exact_reference
    # option below replaces it by inversion of the guarded local-pressure EOS.
    rho_l_sat = Float64(s.rho_inf) - sigma*q/a2l
    # For nonzero lq, preserve the legacy lq=0 raw ambient root while changing
    # liquid compliance. This differs from exact_reference calibration, which
    # targets the requested guarded ambient pressure and requires lq=0.
    tmix >= 0.0 || error("build_tem: tmix must be >= 0")
    if tmix > 0.0
        tvp > 0.0 || error("build_tem: tmix>0 without tvp>0 would transport a thermodynamic " *
                           "temperature that NOTHING READS -- the dormant-knob failure mode. " *
                           "Enable tvp so p_v actually responds to it, or leave tmix at 0.")
    end
    lq >= 0.0 || error("build_tem: lq must be >= 0 -- a negative lq makes a_eff negative, which " *
                       "breaks the discriminant proof (disc = B^2 + 4A|C| needs C <= 0) and can " *
                       "produce a silent NaN on GPU. See _pmix_vl property P1.")
    if lq > 0.0
        vaploc > 0.0 || error("build_tem: lq>0 requires vaploc>0 -- the non-linear liquid branch " *
                              "lives in the closed-form local-pressure root; the frozen-p_v path " *
                              "does not carry it, and enabling it there would be silently inert.")
        ncg == 0.0 || error("build_tem: lq and ncg are mutually exclusive (both are gas compliance; " *
                            "ncg holds it at p_v, lq at the local pressure -- carrying both would " *
                            "double-count the same physics).")
    end
    # ★ lq + exact_reference ORDER (2026-09-24). The legacy lq Newton below preserved the legacy raw
    # ambient root; when exact_reference is requested it would run FIRST and then be silently
    # OVERWRITTEN by the exact-reference inversion, which used the lq=0 algebra -- a wrong ambient
    # root at lq>0 without any error. The exact-reference inversion now carries the lq term itself
    # (bitwise the lq=0 arithmetic when lq==0), and the legacy Newton is skipped under it.
    if lq > 0.0 && !exact_reference
        # Legacy realised ambient root (lq = 0 branch), in Float64.
        D_leg = Float64(s.rho_inf) - rho_l_sat
        p_t   = p_v + D_leg*rho_l_sat*a2l/Float64(s.rho_inf)
        p_t > p_v || error("build_tem: ambient root p_t=$p_t is not above p_v=$p_v; lq calibration " *
                           "is undefined there (the liquid is not in tension at the far field).")
        # Solve  x = rho_inf - (p_t - p_v)*rho_inf*(1/(x*a2l) + lq/p_t)  by Newton, from the legacy x.
        # f(x)  = x - rho_inf + (p_t-p_v)*rho_inf*(1/(x*a2l) + lq/p_t)
        # f'(x) = 1 - (p_t-p_v)*rho_inf/(x^2*a2l)
        let K = (p_t - p_v)*Float64(s.rho_inf), x = rho_l_sat
            for _ in 1:100
                fx  = x - Float64(s.rho_inf) + K*(1.0/(x*a2l) + lq/p_t)
                dfx = 1.0 - K/(x*x*a2l)
                abs(dfx) > 1e-12 || error("build_tem: lq calibration Newton hit a zero derivative")
                xn = x - fx/dfx
                xn > 0.0 || error("build_tem: lq calibration drove rho_l_sat non-positive (lq=$lq " *
                                  "is too large for this operating point)")
                abs(xn - x) <= 1e-14*max(1.0, abs(xn)) && (x = xn; break)
                x = xn
            end
            # VERIFY the calibration rather than trusting the iteration: re-evaluate the root and
            # require it to reproduce p_t. A silent miss here is exactly the class of defect this
            # block exists to prevent.
            Lp_n = 1.0/(x*a2l); D_n = Float64(s.rho_inf) - x
            p_chk = let L = Lp_n, aef = lq, r = Float64(s.rho_inf)
                A_ = r*L; B_ = x - r*(1.0-lq) - r*p_v*L; C_ = -r*aef*p_v
                sq = sqrt(B_*B_ - 4.0*A_*C_)
                A_ > 0.0 ? (B_ >= 0.0 ? C_/(-0.5*(B_+sq)) : (-0.5*(B_-sq))/A_) : (-C_/B_)
            end
            abs(p_chk - p_t) <= 1e-9*max(p_t, 1e-30) ||
                error("build_tem: lq re-calibration FAILED to preserve the ambient root " *
                      "(target $p_t, got $p_chk). Refusing to run on a moved far field.")
            @info "TIER 3 lq=$lq: rho_l_sat re-calibrated $(rho_l_sat) -> $(x); ambient root " *
                  "preserved at p=$p_t (D $(D_leg) -> $(D_n))"
            rho_l_sat = x
        end
    end
    if exact_reference
        # Use the values that will actually be stored, while retaining the
        # requested p_v and source-rate normalization. At alpha=0 the raw EOS is
        # z=p_v+rho_l_sat*a2l*(rho_inf-rho_l_sat)/rho_inf. Invert its pressure
        # guard first, then choose the dense quadratic branch continuously
        # connected to rho_l_sat=rho_inf as the pressure difference vanishes.
        ri=Float64(s.rho_inf); pv=Float64(Float32(p_v)); A=Float64(Float32(a2l))
        sm=Float64(Float32(psm_frac*min(p_v,sigma*q)))
        all(isfinite,(ri,q,sigma,pv,A,sm,ratio)) && ri>0 && q>0 && sigma>0 &&
            pv>0 && A>0 && sm>=0 && ratio>1 ||
            throw(ArgumentError("invalid exact-reference operating point or material constants"))
        target=pv+Float64(sigma)*q
        raw=target-sm*sm/(4target)
        # ★ lq-AWARE INVERSION (2026-09-24). At alpha=0 the guarded root's denominator is
        # 1 - (x-pv)*(1/(rho_l_sat*a2l) + lq/x); requiring rho(x=raw at rho=rho_inf) gives the
        # quadratic rho_l_sat^2 - ri*(1-k)*rho_l_sat + ri*(raw-pv)/A = 0 with k = (raw-pv)*lq/raw.
        # At lq=0, k=0 and every expression below is bitwise the previous lq=0 arithmetic
        # (multiplication by 1.0 and (1-0.0)^2 are IEEE-exact identities).
        k=(raw-pv)*Float64(Float32(lq))/raw
        isfinite(k) && 0<=k<1 ||
            throw(ArgumentError("exact-reference lq compliance term is not representable (k=$k)"))
        d=(raw-pv)/(ri*A*(1-k)^2)
        isfinite(d) && 0<=d<0.25 ||
            throw(ArgumentError("exact-reference pressure has no dense liquid branch"))
        rho_l_sat=Float64(Float32(0.5ri*(1-k)*(1+sqrt(1-4d))))
        0.5ri*(1-k)<rho_l_sat<ri ||
            throw(ArgumentError("exact-reference liquid density is not representable"))
    end
    rho_v_req  = (exact_reference ? rho_l_sat : Float64(s.rho_inf))/ratio
    exact_reference && !(isfinite(Float32(rho_v_req)) && Float32(rho_v_req)>0 &&
                        isfinite(Float32(p_v/rho_v_req)) && Float32(p_v/rho_v_req)>0) &&
        throw(ArgumentError("exact-reference vapor density or stiffness is not representable"))
    # vfloor=true keeps a full-vapor cell's T <= Tmax (the legacy 3:1 cap). vfloor=false = PDP mode:
    # the pexc split carries the excess pressure, so rho_v_sat can be the REAL (e.g. 1000:1) vapor density.
    rho_v_floor = 1.02*p_v/(Float64(Tmax)*Float64(TEM_CS2))
    rho_v_sat = vfloor ? max(rho_v_req, rho_v_floor) : rho_v_req
    # PRESSURE-driven rates: source is k*(p_v - p), so k = C*U/q (the density form carried an
    # extra a2l because it multiplied a density difference).
    #
    # ★ RATE REFERENCE (2026-07-19). The source is k*(p_v - p) with p >= 0 (smooth-max), so the
    # EVAPORATION driving force is CAPPED AT p_v and the achievable rate is k*p_v. With k = C*U/q that is
    # C*U*p_v/q -- it SCALES WITH p_v, giving
    #     tau_evap [CTU] = q/(C*p_v)
    # legacy  (q/p_v = 0.029): tau = 0.029 CTU at C=1  -> fast, never starved
    # Tier-C  (q/p_v = 10   ): tau = 10 CTU    at C=1  -> the fluid crosses the chord TEN TIMES before
    #                                                     vapour forms. MEASURED consequence: rho collapses
    #                                                     to the 1e-4 floor while alpha stays ~0.02, i.e.
    #                                                     VACUUM, not vapour.
    # rate_ref_pv=true references the rate to p_v instead of q:  k = C*U/p_v  =>  tau [CTU] = 1/C,
    # INDEPENDENT of p_v. That is the structural fix; C then has a direct meaning (C = 30 -> 0.033 CTU).
    # MEASURED with this on (eps_pv=0.1, Tmin=1.2, wall fixed): alpha_max 0.019 -> 0.994, with 6017 cells
    # above alpha=0.20 and 248/256 wall points below ambient. The thick cavity Tier-C was after.
    # DEFAULT false => k = C*U/q, bitwise unchanged for every existing case.
    krate = rate_ref_pv ? U/max(p_v, 1e-30) : U/q
    kprod = Cprod*krate
    kdest = Cdest*krate
    # VAPOR-BRANCH STIFFNESS — must be p_v/rho_v_sat, NOT a small fraction of a2l. With this value the
    # Wallis mixture branch ALREADY reduces to the ideal gas at alpha=1:
    #   p = p_v + a2v*(rho - rho_v_sat) = p_v*rho/rho_v_sat  ->  0 as rho -> 0
    # which is the restoring pressure that BOUNDS cavity density. The old near-flat branch tended to the
    # CONSTANT p_v as rho -> 0, so nothing pulled mass back and the cavity evacuated to the numerical
    # floor (measured in 3D: supercavitation at every sigma below inception). See cmesh_cav_tem3d.jl.
    a2v = p_v/rho_v_sat
    # psm must be small vs BOTH the plateau depth (sigma*q) and the vapor scale (p_v), else the smooth-max
    # washes out whichever is smaller. min() keeps leg A (legacy p_v) bit-comparable to the committed runs.
    psm = psm_frac*min(p_v, sigma*q)                            # positivity-guard width
    # ★ NCG (2026-07-28): frozen free-gas compressibility — third Wallis term ncg/p_v (see _a2m_mix).
    # The gas softens the LIQUID branch (alpha=0), so rho_l_sat must be recalibrated with the SOFTENED
    # ambient slope a2m0 = 1/(rho_inf*S0), S0 = 1/(rho_l_sat*a2l) + ncg/p_v, or the ambient would sit at
    # p_v + a2m0*sigma*q/a2l < p_v + sigma*q and the A/B would silently confound gas content with a
    # LOWER effective cavitation number. Fixed point in rho_l_sat (contraction ~ sigma*q/a2l ~ 0.015).
    # FEASIBILITY: the linearized liquid branch must span sigma*q within a sane density drop; at this
    # operating point that caps ncg near p_v*(0.5*rho_inf... )/(sigma*q*rho_inf) — enforced below.
    if ncg > 0.0
        (eos == 1 && wallis == 1) || error("build_tem: ncg>0 requires eos=1 && wallis=1 (the gas term extends the density-weighted Wallis sum)")
        # TARGET: the LEGACY wallis=1 ambient overpressure p_ref - p_v = rho_l_sat_legacy*sigma*q/rho_inf
        # (the documented ~2% Wallis offset, tem_ambient docstring) — NOT the nominal sigma*q — so the
        # far-field pressure of an NCG leg matches its ncg=0 control EXACTLY and only compressibility
        # differs. At ncg=0 the fixed point below reproduces rho_l_sat_legacy identically.
        dp_target = rho_l_sat*sigma*q/Float64(s.rho_inf)
        for _ in 1:12
            S0 = 1.0/(rho_l_sat*a2l) + ncg/p_v
            rho_l_sat = Float64(s.rho_inf) - dp_target*Float64(s.rho_inf)*S0
        end
        if rho_l_sat <= 0.5*Float64(s.rho_inf)
            ncg_max = p_v*(0.5/(sigma*q) - 2.0/(Float64(s.rho_inf)*a2l))
            error("build_tem: ncg=$(ncg) infeasible — the NCG-softened liquid branch needs a density drop of $(round(Float64(s.rho_inf)-rho_l_sat, sigdigits=3)) to span sigma*q (rho_l_sat=$(round(rho_l_sat, sigdigits=3)) <= 0.5*rho_inf). Reduce ncg below ~$(round(ncg_max, sigdigits=2)).")
        end
    end
    # ★ DK (2026-07-28): unit conversion + validation. cmin and the vapour-end cap are given in units
    # of U (published: cmin/U = 0.28, vapour branch ~69U; start the cap ladder at 10U). At alpha=0 the
    # dk branch returns the exact wallis liquid value, so ambient/far-field parity is automatic.
    dk_cmin2 = 0.0; dk_a2v = 0.0
    if dk_cmin_U > 0.0
        (eos == 1 && wallis == 1) || error("build_tem: dk_cmin_U>0 requires eos=1 && wallis=1")
        ncg == 0.0 || error("build_tem: dk and ncg are mutually exclusive (orthogonal EOS experiments)")
        dk_cmin2 = (dk_cmin_U*U)^2
        dk_a2v  = (dk_a2v_U*U)^2
    end
    # ★★★ LOCAL-PRESSURE VAPOUR TERM (2026-07-30). The closed-form solve in `_pmix_vl` derives the
    # Wallis sum S = (1-a)/(rho_l_sat*a2l) + a/p EXACTLY — it assumes the density-weighted Wallis
    # branch and nothing else. The DK c(alpha) profile is a DIFFERENT compliance law (not a Wallis
    # sum at all) and NCG adds a third frozen term ncg/p_v that this derivation does not carry, so
    # both are excluded rather than silently ignored. `vapbound` is the REFUTED floor of 2026-07-29
    # (kept as flag-gated scaffolding): the two are alternative repairs of the same defect and must
    # never be composed.
    if vaploc > 0.0
        (eos == 1 && wallis == 1) || error("build_tem: vaploc>0 requires eos=1 && wallis=1 (the local-pressure vapour term is derived FROM the density-weighted Wallis sum)")
        ncg == 0.0 || error("build_tem: vaploc and ncg are mutually exclusive (the ncg/p_v term is not carried by the closed-form solve)")
        dk_cmin_U == 0.0 || error("build_tem: vaploc and dk are mutually exclusive (dk replaces the Wallis sum with a different c(alpha) law)")
        vapbound == 0.0 || error("build_tem: vaploc and vapbound are mutually exclusive (alternative repairs of the SAME defect; vapbound is the refuted floor)")
    end
    NI, NJ, g = s.NI, s.NJ, s.g
    mk(nx, ny) = (a = KA.allocate(s.backend, Float32, nx, ny); fill!(a, 0.0f0); a)
    # ★★★★ RAYLEIGH-PLESSET SET-UP. Allocated as 1x1 STUBS when rp==0 so the legacy path costs no VRAM.
    # UNITS. This solver's mesh is unit-chord, so a nuclei density quoted per m^3 must be multiplied by
    # chord^3 and a nucleus radius quoted in metres divided by chord. Getting this wrong is silent --
    # it just moves inception -- so the conversion is done HERE, once, from the physical values, and
    # the REALISED numbers are printed below rather than the knob values.
    #   Schnerr-Sauer standard water: n0 = 1e13 /m^3, R0 = 1e-6 m (Schnerr & Sauer 2001; the same pair
    #   ANSYS CFX and OpenFOAM's interPhaseChangeFoam ship as defaults).
    rp_n0_eff = rp_n0 > 0.0 ? rp_n0 : 1.0e13*chord_m^3
    rp_R0_eff = rp_R0 > 0.0 ? rp_R0 : 1.0e-6/chord_m
    if rp > 0.0
        # Report the realised bubble-dynamics state, in the campaign's own F3 style: never the knob.
        w2 = rp_minnaert_omega2(Float32(rp_R0_eff), Float32(rho_l_sat), Float32(rp_pg0),
                                Float32(rp_S), Float32(rp_kappa))
        a_seed = rp_alpha_of_R(Float32(rp_R0_eff), Float32(rp_n0_eff))
        @info @sprintf("RAYLEIGH-PLESSET ENGAGED: n0=%.4g /chord^3  R0=%.4g chord  S=%.4g  pg0=%.4g  kappa=%.2f",
                       rp_n0_eff, rp_R0_eff, rp_S, rp_pg0, rp_kappa)
        @info @sprintf("   seed alpha at R=R0: %.4e   |   Minnaert omega^2 = %+.4e  => %s",
                       a_seed, w2,
                       w2 > 0 ? @sprintf("natural period %.4g (lattice time)", 2pi/sqrt(w2)) :
                                "NO NATURAL FREQUENCY (pg0 too small vs surface tension): this is an " *
                                "inertial runaway, NOT an oscillator -- do not describe it as one")
        # ★ THE RATIO THAT DECIDES WHETHER R-P CAN MATTER AT ALL. See
        # memory/project_docs/2026-08-01_RP_IS_QUASI_STATIC_A_PRIORI.md: at the benchmark point with 1 um nuclei the
        # Minnaert period is ~1e-05 CTU against a 3.2 CTU shedding period -- quasi-static by five
        # orders. A run that then behaves like Merkle is CONFIRMING that, not failing.
        let (Tctu, ratio) = rp_timescale_report(Float32(rp_R0_eff), Float32(rho_l_sat), Float32(rp_pg0),
                                                Float32(rp_S), Float32(rp_kappa), U)
            if isfinite(Tctu)
                @info @sprintf("   Minnaert period = %.3e CTU; cavity transit ~0.35 CTU is %.3g x SLOWER%s",
                               Tctu, ratio,
                               ratio > 100 ? "  => R-P is QUASI-STATIC; expect Merkle-like behaviour" : "")
            end
        end
    end
    mkrp(nx, ny) = rp > 0.0 ? mk(nx, ny) : mk(1, 1)
    # Tier 4: 1x1 stubs when tmix == 0, so the legacy path costs no VRAM. mk() zero-fills, and
    # zero IS the ambient state for a deviation field -- no initialisation pass is needed.
    mktm(nx, ny) = tmix > 0.0 ? mk(nx, ny) : mk(1, 1)
    # ⛔ arecon is NOT plumbed into _k_p3_couple!, which carries its own advection. Allowing the
    # combination would silently keep FIRST ORDER on the coupled path while the banner said otherwise
    # -- the exact silent-degradation class this campaign has found nineteen of. Fail loudly instead.
    # ⛔ advimp IS DORMANT ON THE p3 PATH -- MEASURED 2026-08-05, battery 23. `advimp` appears only in
    # _k_tem_sweep!; it is not in _k_p3_couple!'s signature and never referenced in its body. B23 ran
    # cflp=2.0 and 5.0 with advimp 0 vs 1 and the step-3 contraction factors differ in the SIXTH
    # decimal (0.987995 vs 0.987992; 0.985895 vs 0.985891) -- GPU round-off, not an effect.
    # ⚠ CONSEQUENCE FOR THE RECORD: the 2026-07-31 claim that advimp turned a DIVERGING p3 rung at
    # cflp=2.0 (rho=1.00343) into a CONTRACTING one (rho=0.99920) CANNOT be attributed to advimp,
    # because advimp does not reach that kernel. Unlike arecon and tmix, this combination failed
    # SILENTLY rather than loudly, which is why it survived four months.
    (advimp > 0.0 && p3 > 0) && error("advimp>0 with p3>0 is not supported: advimp is implemented in "*
        "_k_tem_sweep! only and _k_p3_couple! neither declares nor reads it, so the knob would be "*
        "silently inert. Run them separately, or plumb advimp into the p3 kernel first.")
    # ⛔ Same silent-degradation guard as arecon: _k_p3_couple! REPLACES sweep+commit, so it would
    # never advance dTm and tmix would be dormant while the banner said otherwise.
    (tmix > 0.0 && p3 > 0) && error("tmix>0 with p3>0 is not supported: _k_p3_couple! replaces the "*
        "sweep/commit pair and does not advance the thermodynamic temperature. Run them separately.")
    (arecon > 0 && p3 > 0) && error("arecon>0 with p3>0 is not supported: _k_p3_couple! still uses " *
        "first-order upwind for alpha. Run them separately, or plumb arecon into the p3 kernel first.")
    TEMState(mk(NI+2g, NJ+2g), mk(NI+2g, NJ+2g), mk(NI, NJ), mk(NI, NJ),
             Float32(p_v), Float32(a2l), Float32(rho_l_sat), Float32(rho_v_sat),
             Float32(kprod), Float32(kdest), Tmin, Tmax,
             Float32(a2v), Int32(wallis), Float32(psm), Int32(eos), Float32(am_floor_frac),
             Float32(src_smooth_frac*p_v), Int32(src_jac), Float32(omega),
             Int32(enforce), Float32(kbeta*U^2), Int32(conserv),
             Int32(p3), Float32(cflp), Int32(arecon), mk(NI, NJ), Float32(ncg),
             Float32(dk_cmin2), Float32(dk_a2v), Float32(dtboost), Float32(vapbound),
             Float32(vaploc), Float32(dtb_dt), Float32(advimp), Int32(gs),
             # ★★★★ Rayleigh-Plesset block, appended in struct order.
             Float32(rp), mkrp(NI+2g, NJ+2g), mkrp(NI+2g, NJ+2g), mkrp(NI, NJ), mkrp(NI, NJ),
             Float32(rp_n0_eff), Float32(rp_R0_eff), Float32(rp_S), Float32(rp_pg0),
             Float32(rp_kappa), Float32(s.nu_lam),
             # thermal p_v: default L/R_v ~ 5 (water-like, dimensionless in lattice T) and T_ref = the
             # ambient carrier temperature, so at T == T_ref the exponent is exactly 0 and p_v is the
             # build-time value -- i.e. tvp>0 with an unperturbed T is a no-op, by construction.
             Float32(tvp), Float32(tvp_LRv > 0 ? tvp_LRv : 5.0),
             Float32(tvp_Tref > 0 ? tvp_Tref : Float64(s.T_inf)),
             Float32(slip), Float32(slip_tau), Float32(cloudint),
             # S3c defaults: 1 um to 100 um in CHORD units, the range real water tunnels carry.
             # ⚠ the 1 um Schnerr-Sauer standard is SUB-CRITICAL at this operating point by 9.3x
             # (R_crit = 9.28 um), so a spectrum reaching ~100 um is required for any activation.
             Float32(poly), Float32(poly_e),
             Float32(poly_Rmin > 0 ? poly_Rmin : 1.0e-6/chord_m),
             Float32(poly_Rmax > 0 ? poly_Rmax : 1.0e-4/chord_m),
             Float32(hk), Float32(wallnuc), Float32(wall_decay), Float32(btke), Float32(pvturb),
             Float32(pvvort), Float32(tfree), Float32(lq),
             Float32(tmix), Float32(tmix_lat), mktm(NI+2g, NJ+2g), mktm(NI+2g, NJ+2g))
end

tem_ratio(tem::TEMState, s) = Float64(s.rho_inf)/Float64(tem.rho_v_sat)

"""★ REALISED sound-speed profile c(alpha)/U, evaluated through the EOS THE SOLVER ACTUALLY USES —
`_pmix_vl`'s dp/drho under vaploc>0, `_a2m_mix` otherwise. `at` selects the EVALUATION DENSITY:
`:rho_inf` (default, legacy convention) or `:rho_sat` (the saturation line). ALWAYS print
`tem_cprofile_dens(tem, s, at)` next to the numbers — the convention is load-bearing (see below).

WHY THIS EXISTS (2026-07-29): the DK banner printed the REQUESTED c_min and liquid stiffness while
the code, through a rail-floor cap, delivered a liquid 40% softer — four runs were scored against a
configuration that never existed. Instrumentation must report what the solver USES. Every knob that
touches the mixture EOS must be reported through this function, never from its input arguments.

★★ 2026-07-30: this function violated its own rule. Under `vaploc>0` the six EOS sites take
`_pmix_vl`, whose dp/drho is the TOTAL derivative of the implicit root, while this function still
evaluated `_a2m_mix` — the FROZEN-p_v stiffness that vaploc exists to replace. Since c(alpha) is the
exact quantity the acoustic-contrast hypothesis is scored on, the profile is now taken from the same
helper the solver reads. The previous implementer printed a warning next to the wrong number instead;
that warning is removed from the two tools. vaploc==0 output is unchanged (same call, same operands).

★★★ 2026-07-30 (the SECOND half of the same bug — the EVALUATION DENSITY was silent). The profile
was, and by default still is, taken at the FIXED r = rho_inf = 1 for EVERY alpha. Under the legacy
`_a2m_mix` that convention was immaterial: a2m = 1/(r*S(alpha)), so rho entered only as a uniform
1/sqrt(r) factor and the profile SHAPE was rho-independent. Under `_pmix_vl` the root depends on rho
and alpha JOINTLY, so the convention became LOAD-BEARING — and it was stated nowhere. MEASURED at
sigma=1.0, production constants (CPU sweep, both helpers, Float32 verbatim):

                       a=0     1e-3    0.1     0.5     0.9     1.0    contrast
    vaploc @ rho_inf   4.900   4.890   4.345   3.705   3.150   2.236    2.19x
    legacy @ rho_inf   4.950   4.056   0.701   0.316   0.236   0.224   22.14x
    vaploc @ rho_sat   5.000   4.100   0.746   0.449   0.720   2.236   11.14x
    legacy @ rho_sat   5.000   4.100   0.746   0.449   0.720   2.236   11.14x   <== IDENTICAL

Read off the rho_inf row alone, the repair looks like a 10x COLLAPSE of mixture compliance — the
campaign's headline acoustic-contrast hypothesis MIS-SCORED. It is an artefact of the convention:
the (alpha=1, rho=1) point that dominates that row is FULL VAPOUR AT LIQUID DENSITY, where the
derived ideal gas gives p = a2v*rho ~ 100x p_v — a state the flow cannot occupy. On the SATURATION
LINE rho = rho_sat(alpha), where p == p_v by property P5 of `_pmix_vl` and where a cavity actually
sits, vaploc and legacy agree to every printed digit and BOTH give contrast 11.14x. This is the
DK-rail-floor failure mode in a new dress: a banner reporting a configuration the solver never
realises. BOTH columns are printed by the two tools, each labelled with its evaluation density.

★★★★ PRE-EXISTING, NOT INTRODUCED HERE — RAW dp/drho vs the solver's `am`. This function (and its
`_cprofile_a2` helper) returns the RAW dp/drho from `_pmix_vl` / `_a2m_mix`; the array the solver
actually reads is `am = sqrt(max(fac_sm*dpdr_raw, am_floor_frac*a2l))` (see :1112-1113, :1158-1159,
:1255-1256, :1373) — neither the psm smooth-max factor `fac_sm` nor the `am_floor_frac` floor is
applied here. This is the SAME gap already logged as AUDIT_A1 F3 / AUDIT_A4 / AUDIT_B3-16 — pre-
existing, not introduced by the 2026-07-30 edit above. It is NUMERICALLY INERT on the `:rho_sat` row
this function prints: there p = p_v > 0 so fac_sm = 0.5*(1 + 1/sqrt(1.0025)) = 0.99938 (scales c by
only 0.99969), and the floor sqrt(1e-4*0.93333)/U = 0.0500U sits far below that row's minimum
0.449U — neither convention would move a printed digit there. That inertness is a further argument
FOR reading the `:rho_sat` column.

Returns Vector{Tuple{alpha, c/U}} at the given alphas. Use `tem_eos_name(tem)` for the EOS-branch
label and `tem_cprofile_dens(tem, s, at)` for the evaluation-density label."""
function tem_cprofile(tem::TEMState, s; alphas=(0.0, 1.0e-3, 0.1, 0.5, 0.9, 1.0), at::Symbol=:rho_inf)
    U = sqrt(Float64(s.uxi)^2 + Float64(s.uyi)^2)
    if at === :rho_sat
        # rho_sat(alpha) formed EXACTLY as the sweep/commit kernels form it (same operands, same order).
        return [(Float64(a), sqrt(Float64(_cprofile_a2(tem, Float32(a),
                    tem.rho_l_sat*(1.0f0 - Float32(a)) + tem.rho_v_sat*Float32(a))))/U) for a in alphas]
    end
    at === :rho_inf || error("tem_cprofile: `at` must be :rho_inf or :rho_sat (got :$at)")
    # ★ the operands and their association are IDENTICAL to the pre-2026-07-30 body — `_cprofile_a2`'s
    # vaploc==0 arm IS that exact `_a2m_mix` call — hence bitwise; the default column (which every
    # historical banner was compared against) is bitwise unchanged for BOTH branches.
    r = Float32(s.rho_inf)
    [(Float64(a), sqrt(Float64(_cprofile_a2(tem, Float32(a), r)))/U) for a in alphas]
end

"""Printable EVALUATION DENSITY of a `tem_cprofile` column. Exists so the convention can never again
be silent: under vaploc>0 the c(alpha) SHAPE depends on it (2.19x vs 11.14x contrast at the SAME
constants — see `tem_cprofile`). Print this next to every c(alpha) profile."""
function tem_cprofile_dens(tem::TEMState, s, at::Symbol=:rho_inf)
    at === :rho_inf && return string("rho = rho_inf = ", round(Float64(s.rho_inf), digits=4),
        " FIXED for all alpha — OFF the saturation line for a>0")
    at === :rho_sat && return string("rho = rho_sat(a) = (1-a)*", round(Float64(tem.rho_l_sat), digits=4),
        " + a*", round(Float64(tem.rho_v_sat), digits=4), " — SATURATION LINE, p == p_v (where a cavity sits)")
    error("tem_cprofile_dens: `at` must be :rho_inf or :rho_sat (got :$at)")
end

@inline function _cprofile_a2(tem::TEMState, a::Float32, r::Float32)
    if tem.vaploc > 0.0f0
        rho_sat = tem.rho_l_sat*(1.0f0-a) + tem.rho_v_sat*a
        _, dpdr, _ = _pmix_vl(tem.p_v, a, r, rho_sat, tem.a2l, tem.rho_l_sat, tem.rho_v_sat, tem.lq)
        return dpdr
    end
    return _a2m_mix(a, r, tem.a2l, tem.a2v, tem.rho_l_sat, tem.p_v,
                    tem.wallis, tem.ncg, tem.dk_cmin2, tem.dk_a2v)
end

"""Name of the EOS helper `tem_cprofile` (and the six kernel sites) actually evaluated — for banner
labels, so a printed c(alpha) can never be attributed to the wrong branch again."""
tem_eos_name(tem::TEMState) = tem.vaploc > 0.0f0 ?
    "_pmix_vl dp/drho — LOCAL-p vapour term, exact implicit root" : "_a2m_mix — frozen-p_v Wallis"

"""★ σ-RAMP SUPPORT (2026-07-29, the published establishment protocol — LEGI/CAV2003: converge
non-cavitating, then lower the outlet pressure SLOWLY to the target σ). Retunes the σ-derived scalar
fields of an EXISTING TEMState in place (alpha state and history are PRESERVED — that is the point;
`build_tem` would zero them). Mirrors build_tem's derived-constant block exactly; pass the SAME
kwargs the state was built with. v1 restriction: ncg==0 and dk off (their recalibrations are not
mirrored here; enforced). Caller must refresh the far field afterwards:
`pr,a2m = tem_ambient(tem,s); set_farfield_mixture!(s; p_ref=pr, a2m=a2m, ...)` and update
`src_smooth` if it was set as a fraction of p_v."""
function tem_retune_sigma!(tem::TEMState, s; sigma, Cprod::Float64, Cdest::Float64, ratio::Float64,
                           vfloor::Bool, eps_pv::Float64, psm_frac::Float64=0.05,
                           rate_ref_pv::Bool=false)
    # ncg NEEDS the unmirrored fixed-point recalibration of rho_l_sat => still guarded. dk is
    # sigma-INDEPENDENT (dk_cmin2/dk_a2v are set in units of U once; the liquid-end cap reads the
    # retuned rho_l_sat at evaluation time), so the legacy retune below is exact under dk (2026-07-29).
    tem.ncg == 0.0f0 ||
        error("tem_retune_sigma!: the ncg recalibration is not mirrored here (v1)")
    U = sqrt(Float64(s.uxi)^2 + Float64(s.uyi)^2)
    q = 0.5*Float64(s.rho_inf)*U^2
    p_inf = Float64(s.rho_inf)*Float64(TEM_CS2)*Float64(s.T_inf)
    p_v = eps_pv >= 0.0 ? eps_pv*sigma*q : p_inf - sigma*q
    a2l = Float64(tem.a2l)                       # a2l_frac is σ-independent; reuse the stored value
    rho_l_sat = Float64(s.rho_inf) - sigma*q/a2l
    rho_v_req = Float64(s.rho_inf)/ratio
    rho_v_floor = 1.02*p_v/(Float64(tem.Tmax)*Float64(TEM_CS2))
    rho_v_sat = vfloor ? max(rho_v_req, rho_v_floor) : rho_v_req
    krate = rate_ref_pv ? U/max(p_v, 1e-30) : U/q
    tem.p_v = Float32(p_v);           tem.rho_l_sat = Float32(rho_l_sat)
    tem.rho_v_sat = Float32(rho_v_sat)
    tem.kprod = Float32(Cprod*krate); tem.kdest = Float32(Cdest*krate)
    tem.a2v = Float32(p_v/rho_v_sat); tem.psm = Float32(psm_frac*min(p_v, sigma*q))
    nothing
end

"""L-inf residual of the VOID equation: max |alpha_new - alpha| from the last commit, BEFORE relaxation.
Distinct from the solver's rstar, which is the FLOW residual. A converged unsteady result needs BOTH."""
tem_residual(tem::TEMState) = maximum(tem.res)

"""L-inf AND L2 report of the VOID-equation residual, plus where the max lives and how many cells sit
near it. Mirrors `rstar_report_ka` for the flow side.

★ WHY THIS EXISTS. `tem_residual` is a MAX, and this campaign has now TWICE over-read an L-inf residual
as field-wide non-convergence when it was in fact owned by a single near-wall cavity cell (see
`rstar_report_ka`'s note, and the retraction in commit e1b30d4 — "it was, once again, an over-read of an
L-inf residual"). The same trap was walked into a third time on 2026-07-20, when `ares` sitting 4–6x
above the flow control was read as "the void equation is the unconverged one" — on an L-inf number, with
no L2 to say whether it was the field or one cell.

**Before concluding anything from `ares`, check `l2/linf`**: a ratio far below ~1e-2, with `frac` at the
1/N floor, means ONE cell and the field is fine. A ratio of O(0.1) means genuine field-wide
non-convergence. The two point at completely different fixes.

Returns `(linf, l2, frac_at_max, imax, jmax)`."""
function tem_residual_report(tem::TEMState)
    r = Array(tem.res)
    linf = maximum(r)
    l2 = sqrt(sum(abs2, r)/length(r))
    frac = linf > 0 ? count(>=(0.99*linf), r)/length(r) : 0.0
    idx = argmax(r)
    (linf, l2, frac, idx[1], idx[2])
end

"""STAGE 1e — the ambient mixture state `(p_ref, dp/drho)` at `(rho_inf, alpha=0)`.

This is the reference the far field must hold. It is deliberately evaluated through the SAME `_a2m_mix`
+ smooth-max the commit kernel uses, so it tracks the `eos`/`wallis` branch exactly rather than assuming
the analytic `p_v + sigma*q` (which is ~2% off under density-weighted Wallis, because the ambient
`a2m = rho_l_sat*a2l/rho` there, not `a2l`).

Feed straight into `set_farfield_mixture!`:
    p_ref, a2m = tem_ambient(tem, s); set_farfield_mixture!(s; p_ref=p_ref, a2m=a2m,
                                                            Tmin=tem.Tmin, Tmax=tem.Tmax)"""
function tem_ambient(tem::TEMState, s)
    r = Float32(s.rho_inf)
    a2m = _a2m_mix(0.0f0, r, tem.a2l, tem.a2v, tem.rho_l_sat, tem.p_v, tem.wallis, tem.ncg,
                   tem.dk_cmin2, tem.dk_a2v)
    # ★★ VAPOUR-SIDE BOUND, through the SAME helper the five kernel sites use (the ambient must never
    # disagree with the field). At (rho_inf, alpha=0) the bound is inert by construction — a2v*rho_inf
    # is orders below the liquid branch — but evaluating it here keeps the far-field reference on the
    # identical EOS. vapbound=0 takes the legacy branch, whose Float64 arithmetic is preserved
    # VERBATIM (the helper is Float32; branching, not scaling, is what makes this bitwise).
    local pmx::Float64, dpdr_raw::Float64
    # ★★★ EOS SITE 6/6 — the LOCAL-PRESSURE vapour term. At alpha=0 the vapour term vanishes (C=0),
    # so by property P4 the ambient PRESSURE is IDENTICAL to legacy at liquid density (verified: the
    # non-negative root is exactly p_v + a2m*(rho_inf - rho_l_sat)). The returned dp/drho is NOT
    # identical: `_pmix_vl` reports the TOTAL derivative of the implicit root,
    # rho_l_sat^2*a2l/rho^2 = 0.89637, against the legacy frozen a2m = rho_l_sat*a2l/rho = 0.91467
    # — 2.0% softer. That is the honest derivative of the SAME pressure function (legacy's a2m is a
    # frozen-alpha stiffness, not d(p_legacy)/drho), and it propagates to set_farfield_mixture!.
    if tem.vaploc > 0.0f0
        pmx32, dpdr32, _ = _pmix_vl(tem.p_v, 0.0f0, r, tem.rho_l_sat, tem.a2l, tem.rho_l_sat, tem.rho_v_sat, tem.lq)
        pmx = Float64(pmx32); dpdr_raw = Float64(dpdr32)
    elseif tem.vapbound > 0.0f0
        pmx32, dpdr32, _ = _pmix_vb(tem.p_v, a2m, r, tem.rho_l_sat, tem.a2v, tem.vapbound)
        pmx = Float64(pmx32); dpdr_raw = Float64(dpdr32)
    else
        pmx = Float64(tem.p_v) + Float64(a2m)*(Float64(r) - Float64(tem.rho_l_sat))
        dpdr_raw = Float64(a2m)
    end
    if tem.eos >= Int32(1)
        sq = sqrt(pmx*pmx + Float64(tem.psm)^2)
        (0.5*(pmx + sq), 0.5*dpdr_raw*(1.0 + pmx/sq))
    else
        (pmx, dpdr_raw)
    end
end

"""Phase-change relaxation timescales `(tau_evap, tau_cond)` in CHORD-TRAVEL UNITS (CTU = L/U).

This is the knob that decides whether the TEM sheds. Linearising the point-implicit Merkle source about
equilibrium: `Delta = rho_sat(alpha) - rho = (alpha_eq - alpha)*(rho_l_sat - rho_v_sat)`, so
`d(alpha)/dt ~ k*(rho_l_sat - rho_v_sat)*(alpha_eq - alpha)`, giving `tau = U/(k*drho_lv)` in CTU.

CONVENTION WARNING (2026-09-17). The paragraph below is for the DEFAULT `rate_ref_pv=false` branch,
where `krate = U/q` and the timescale depends on the operating point. **The cavitation movie driver
hard-codes `rate_ref_pv=true`** (`tools/cmesh_cav_movie.jl`), and on that branch `krate = U/p_v` and
the timescale is exactly **`tau [CTU] = 1/C`, independent of the operating point** (the comment at the
`krate` assignment says so). So for every run this campaign has made, the numbers below are wrong by
about fifty times: `Cdest = 1` gives `tau_cond` **1.0** CTU, not 0.02, and a `tau_cond` of 0.2 to 1 CTU
needs `Cdest` **1 to 5**, not 0.02 to 0.1. Verified against the driver's own printout: `tau_evap`
0.003333 = 1/300 = 1/Cprod and `tau_cond` 0.2 = 1/5 = 1/Cdest.

Measured negative result on the guidance itself: arm U25 of `S0903_RUNS/species_cav_uniform_20260917`
ran at `tau_cond` 0.2 CTU, inside the window recommended below for shedding, and produced a STEADY
attached sheet with zero detached clouds. Being in that window is not sufficient for shedding there.

Measured meaning for the NACA0015 M=0.2 case (`a2l/q ~ 50`): **Cprod=Cdest=1 gives tau ~ 0.02 CTU** —
two orders of magnitude faster than the ~2 CTU Leroux shedding period, i.e. the "finite-rate" TEM is
running in its HEM (instantaneous-barotrope) limit, which is exactly why the 2026-07-18 run settled to a
quasi-steady sheet. Metastable cloud shedding needs the CONDENSATION timescale to be a fraction of the
shedding period: `tau_cond ~ 0.2-1 CTU` => `Cdest ~ 0.02-0.1`, with evaporation left fast (`Cprod ~ 1`)."""
function tem_timescales(tem::TEMState, s)
    drho = Float64(tem.rho_l_sat) - Float64(tem.rho_v_sat)
    U = sqrt(Float64(s.uxi)^2 + Float64(s.uyi)^2)
    a2 = Float64(tem.a2l)
    (U/(Float64(tem.kprod)*a2*drho), U/(Float64(tem.kdest)*a2*drho))
end

# ---- MIXTURE EOS: DELEGATED TO THE SINGLE SOURCE (2026-08-01, Tier 0.3) --------------------------
# The 330 lines that used to sit here -- `_a2m_mix`, `_pmix_vb`, `_pmix_vl` and their derivations --
# were a VERBATIM DUPLICATE of src/cmesh/cmesh_eos_mix.jl and are now deleted, not copied.
#
# WHY THIS MATTERS MORE THAN A TIDY-UP. BOTH EOS defects this campaign has found came from exactly
# one cause: 2D and 3D each holding their own copy of the mixture EOS, and the copies drifting. The
# second one hid on 84% of the (alpha,rho) map while agreeing to 8e-06 on the saturation line, i.e.
# it was invisible to every tolerance-based check. The 3D file was repointed on 2026-07-31; this is
# the 2D side, deferred then only because a GPU queue held the file.
#
# GATED, not asserted: tools/_attic/retired_verifiers/verify_eos_shared.jl swept 49,600 states across alpha x rho x wallis x
# both vapbound settings and found 0 bitwise mismatches (re-run 2026-08-01 at this revision), and the
# deleted block was byte-diffed against lines 24-330 of the shared file with zero differences before
# removal. The verifier warns that this proves the FUNCTIONS agree, not that callers pass the same
# arguments -- the six call sites are unchanged in this commit, so their arguments are unchanged too.
include(joinpath(@__DIR__, "..", "cmesh", "cmesh_eos_mix.jl"))
using .CMeshEOSMix: _a2m_mix, _pmix_vb, _pmix_vl
include(joinpath(@__DIR__, "..", "cmesh", "cmesh_pressure_outlet.jl"))

# ---- BC: wall zero-gradient / wake-cut partner exchange / far-field liquid --------------------------
@kernel function _k_tem_bc!(alpha, NI::Int32, NIW::Int32, NIF::Int32, NJ::Int32, g::Int32)
    i, k = @index(Global, NTuple)
    @inbounds if i <= NI && k <= g
        if i <= NIW || i > NIW+NIF                 # wake cut: partner exchange (no sign flip)
            ip = NI+1-i
            alpha[i+g, g+1-k] = alpha[ip+g, k+g]
        else                                       # solid wall: zero-gradient d(alpha)/dn = 0
            alpha[i+g, g+1-k] = alpha[i+g, g+k]
        end
        alpha[i+g, NJ+g+k] = 0.0f0                  # far-field: pure liquid
    end
end

# Segmented-channel override after the legacy far-field fill: inlet remains pure liquid, while
# slip walls use a layerwise zero-normal-gradient mirror.
@kernel function _k_tem_bc_outer!(alpha, @Const(outer_face_kind),
                                  NI::Int32, NJ::Int32, g::Int32)
    i, k = @index(Global, NTuple)
    @inbounds if i <= NI && k <= g
        face_kind = outer_face_kind[i]
        jd = NJ + g + k
        if face_kind == 0x01
            alpha[i+g, jd] = 0.0f0
        elseif face_kind >= 0x02
            alpha[i+g, jd] = alpha[i+g, NJ+g+Int32(1)-k]
        end
    end
end

@kernel function _k_tem_bc_io!(alpha, NI::Int32, NJ::Int32, g::Int32)
    jj, k = @index(Global, NTuple)
    @inbounds if jj <= NJ+2g && k <= g
        alpha[g+1-k, jj]  = alpha[g+1, jj]          # i-inlet/outlet ghosts: zero-gradient
        alpha[NI+g+k, jj] = alpha[NI+g, jj]
    end
end

# ---- sweep: upwind advection + point-implicit Merkle source + dual-time BDF2 ------------------------
# ---- TIER 0: alpha face reconstruction -----------------------------------------------------------
# minmod, and the upwind-biased face value. arecon<=0 returns the upwind CELL value, which is exactly
# the legacy first-order face -- so arecon=0 is bitwise by construction, not by intention.
@inline _mmA(a::Float32, b::Float32) = (a*b <= 0.0f0) ? 0.0f0 : (abs(a) < abs(b) ? a : b)
@inline function _afaceA(aLL::Float32, aL::Float32, aR::Float32, aRR::Float32,
                         pos::Bool, arecon::Int32)
    arecon <= Int32(0) && return pos ? aL : aR
    return pos ? aL + 0.5f0*_mmA(aL - aLL, aR - aL) :
                 aR - 0.5f0*_mmA(aR - aL,  aRR - aR)
end

@kernel function _k_tem_sweep!(alpha_new, res, @Const(alpha), @Const(rho), @Const(ux), @Const(uy),
                               @Const(Six), @Const(Siy), @Const(Sjx), @Const(Sjy), @Const(vol),
                               @Const(dtf), @Const(un), @Const(um), @Const(rk_r),
                               rho_l_sat::Float32, rho_v_sat::Float32, kprod::Float32, kdest::Float32,
                               p_v::Float32, a2l::Float32, a2v::Float32, psm::Float32, wallis::Int32,
                               ncg::Float32, dk_cmin2::Float32, dk_a2v::Float32,
                               src_smooth::Float32, src_jac::Int32, enf::Int32, beta2::Float32,
                               cons::Int32, arecon::Int32, dtboost::Float32, vapbound::Float32, vaploc::Float32,
                               dtb_dt::Float32, advimp::Float32, gs::Int32, dtp::Float32,
                               # ★★★★ Rayleigh-Plesset block. rp==0 => none of it is touched.
                               rp::Float32, rdot_new, @Const(rdot), @Const(rdot_n), @Const(rdot_m),
                               rp_n0::Float32, rp_R0::Float32, rp_S::Float32, rp_pg0::Float32,
                               rp_kappa::Float32, rp_nu::Float32,
                               # ★ thermal p_v + drift-flux slip. Both inert at tvp==0 / slip==0.
                               # T and pexc are read ONLY by the slip branch, to reconstruct p at the
                               # face neighbours as p = rho*cs2*T + pexc -- the identical expression
                               # `_k_tem_commit!` writes, so no EOS re-evaluation is needed here.
                               tvp::Float32, tvp_LRv::Float32, tvp_Tref::Float32,
                               slip::Float32, slip_tau::Float32, cloudint::Float32,
                               poly::Float32, poly_e::Float32, poly_Rmin::Float32, poly_Rmax::Float32,
                               hk::Float32, wallnuc::Float32, wall_decay::Float32,
                               btke::Float32, btke_src, pvturb::Float32, @Const(tkarr),
                               pvvort::Float32, @Const(pvortarr),
                               @Const(Tf), @Const(pexc),
                               tmix::Float32, tmix_lat::Float32, dTm, dTm_new, lq::Float32,
                               outlet_mode::Int32, NI::Int32, NJ::Int32, g::Int32)
    i0, j = @index(Global, NTuple)
    # ★★★ LINE GAUSS-SEIDEL IN i (2026-07-31, `gs`). EDIT 2 of the swept-transport fix: use the
    # ALREADY-UPDATED upwind neighbour instead of the previous iterate, so information crosses a whole
    # line in ONE sweep instead of one cell per iteration.
    # gs = 0: the loop runs ONCE with i = i0 and the guard is the original `i <= NI && j <= NJ`
    #         => ndrange (NI, NJ), one cell per thread, BITWISE LEGACY.
    # gs = 1: ndrange (1, NJ), one thread per j-line marching i = 1..NI sequentially. Within a thread
    #         the write to alpha_new[i-1] precedes the read at i, so there is NO RACE; the j-direction
    #         still reads `alpha` (Jacobi in j, Gauss-Seidel in i), which is race-free too.
    # For pure first-order upwind the streamwise system is LOWER BIDIAGONAL (i couples only to i-1),
    # so this forward march IS the exact implicit solve -- a forward substitution, not a Thomas solve.
    # FIXED POINT UNCHANGED: at convergence alpha_new == alpha, so reading either is identical.
    @inbounds if j <= NJ
    for _sw in Int32(1):(gs == Int32(1) ? NI : Int32(1))
    i = gs == Int32(1) ? _sw : i0
    if i <= NI
        ia = i+g; ja = j+g
        a0 = alpha[ia, ja]
        # Upwind value for the i-direction: the just-updated one under gs=1. Guarded at i==1 because
        # alpha_new's GHOST cells are never filled (the BC kernels fill `alpha`), so the i=1 stencil
        # must fall back to `alpha` or it would read stale ghost data.
        _aim1 = (gs == Int32(1) && i >= Int32(2)) ? alpha_new[ia-1, ja] : alpha[ia-1, ja]
        r  = rho[ia, ja]
        local rate_adv::Float32                              # void-advection inverse time (dtboost)
        # ★★★ OUTFLOW COEFFICIENT (2026-07-31, `advimp`). The coefficient MULTIPLYING a0 inside `adv`.
        # The upwind advection splits as adv = cout*a0 - (inflow from upwind neighbours), and `cout` is
        # exactly the part that can be moved onto the diagonal to make the OUTFLOW implicit. Assembled
        # per branch below because the conservative and non-conservative forms have different ones.
        local cout::Float32
        rho_sat = rho_l_sat*(1.0f0-a0) + rho_v_sat*a0        # saturated mixture density at current alpha
        # PRESSURE-DRIVEN Merkle source (standard Merkle/Kunz/Schnerr-Sauer). The old DENSITY-driven form
        # (Delta = rho_sat - rho) tied alpha_eq to how much the LIQUID COMPRESSES, so with a2l/q ~ 50 the
        # void capped near 3% and could only be thickened by softening a2l, which destroyed the -Cp
        # plateau (measured). Driving on (p_v - p) DECOUPLES void from liquid stiffness: a cell below p_v
        # keeps evaporating until the vapor it creates raises p back to p_v, whatever a2l is.
        # ★★★ EOS SITE 1/6. vaploc>0 = the EXACT local-pressure-vapour solve (`_pmix_vl`); vaploc==0
        # takes the else-branch, which is the VERBATIM legacy pair => bitwise legacy. `a2m` and
        # `dpda_scale` stay in scope because the enf block below reproduces its legacy expression
        # CHARACTER-IDENTICALLY (a re-association of that product is not bitwise in Float32).
        a2m = 0.0f0; dpda_scale = 0.0f0
        local pmx::Float32, dpdr_raw::Float32, dpda_raw::Float32
        if vaploc > 0.0f0
            pmx, dpdr_raw, dpda_raw = _pmix_vl(p_v, a0, r, rho_sat, a2l, rho_l_sat, rho_v_sat, lq)
        else
        a2m = _a2m_mix(a0, r, a2l, a2v, rho_l_sat, p_v, wallis, ncg, dk_cmin2, dk_a2v)   # same mixture EOS as the commit kernel
        # ★★ VAPOUR-SIDE BOUND (see _pmix_vb). vapbound=0 => pmx is the verbatim legacy expression,
        # dpdr_raw == a2m and dpda_scale == 1, so every line below is bitwise unchanged.
        pmx, dpdr_raw, dpda_scale = _pmix_vb(p_v, a2m, r, rho_sat, a2v, vapbound)
        dpda_raw = dpda_scale*a2m*(rho_l_sat - rho_v_sat)   # the legacy jac expression, verbatim
        end
        pl  = 0.5f0*(pmx + sqrt(pmx*pmx + psm*psm))           # positivity-guarded mixture pressure
        # ★★★ CLAUSIUS-CLAPEYRON (tvp>0). p_v_loc = p_v*exp(-(L/Rv)(1/T - 1/T_ref)). Applied to the
        # SOURCE potential ONLY -- see the struct doc for why the EOS reference state is NOT moved and
        # why the T this reads is not yet a trustworthy thermodynamic temperature.
        # tvp == 0 => p_v_loc is p_v by the same `===` value, so `dpv` is bitwise unchanged.
        p_v_loc = p_v
        if tvp > 0.0f0
            # ★★★★ TIER 4. tmix == 0 reads `Tf` exactly as before => BITWISE LEGACY. tmix > 0 reads the
            # THERMODYNAMIC temperature instead: the reference plus the transported deviation, which is
            # NOT clamped to the carrier band and can therefore actually go cold under evaporation.
            # This is the line the tvp docstring was waiting on ("it reads the railed carrier pedestal,
            # not a thermodynamic T ... NOT quotable until Tier 3/4").
            Tl = tmix > 0.0f0 ? (tvp_Tref + dTm[ia, ja]) : Tf[ia, ja]
            Tl = Tl < 1.0f-3 ? 1.0f-3 : Tl
            p_v_loc = p_v*exp(-tvp*tvp_LRv*(1.0f0/Tl - 1.0f0/tvp_Tref))
        end
        # ★★★ S3k SINGHAL TURBULENT VAPOUR THRESHOLD (Singhal et al. 2002). Turbulent pressure
        # fluctuations let vapour form ABOVE the static saturation pressure:
        #       p_v_eff = p_v + 0.5*p_turb ,   p_turb = 0.39*rho*k
        # `pvturb = 1` reproduces the published constant; 0 returns p_v_loc unchanged by the same
        # `===` value, so `dpv` is bitwise legacy.
        #
        # ⚠ THIS IS THE ONE ADDED TERM WITH A SPATIAL ASYMMETRY THAT MATTERS. Every other term in this
        # session's inventory acts wherever there is vapour, so it scales the whole cavity and leaves
        # the topology alone. `k` is concentrated in the CLOSURE SHEAR LAYER -- where the re-entrant
        # jet is generated and where a cloud would be severed -- and is small under the attached sheet.
        # So this raises the effective driving exactly where the missing dynamics live.
        # Magnitude: at 10% turbulence intensity, 0.195*rho*k is comparable to p_v itself.
        #
        # ⚠ It reads the SST `k`, a genuinely transported turbulence quantity, NOT the railed pressure
        # pedestal that makes thermal p_v unquotable. This term does not inherit that caveat.
        if pvturb > 0.0f0
            kt = tkarr[ia, ja]
            p_v_loc += 0.195f0*pvturb*r*(kt > 0.0f0 ? kt : 0.0f0)
        end
        # ★★★ S3k -- LE & TRAN VORTEX THRESHOLD PRESSURE (JAFM 2022, Eq. 14):
        #     p_v* = p_v + C_v * mu_eff * |dv/dx - du/dy| ,   mu_eff = rho*(nu + nu_t)
        # `pvortarr` carries the KINEMATIC group (nu + nu_t)*|Omega|, formed in _k_sst_sweep! where
        # both factors are already live. `r` supplies rho HERE, exactly as the Singhal branch above
        # does, so the density is the one this sweep holds rather than a second copy.
        #
        # HOW THIS DIFFERS FROM THE SINGHAL BRANCH, and why both exist. Singhal keys on `k`, which is
        # largest in the CLOSURE SHEAR LAYER. This keys on nu_eff*|Omega|, which is large wherever
        # there is a sheared viscous layer -- i.e. ALONG THE WHOLE CAVITY INTERFACE, not only at its
        # ends. Le & Tran attribute exactly that difference to the two forms, and it is the difference
        # battery 6/6b showed we need: rate makes the layer DENSER, this should make it THICKER.
        #
        # ⚠ pvvort IS C_v and their C_v = 10 does NOT transfer to this normalisation. Measure
        # rho*(nu+nu_t)*|Omega| / p_v in the realised state (it is written even at pvvort = 0) before
        # choosing a value. 0 => never read => bitwise legacy.
        if pvvort > 0.0f0
            pw = pvortarr[ia, ja]
            p_v_loc += pvvort*r*(pw > 0.0f0 ? pw : 0.0f0)
        end
        dpv = p_v_loc - pl                                    # >0: under-pressured -> evaporate
        # SMOOTHED SIGN SWITCH (src_smooth > 0). The hard max(dpv,0)/max(-dpv,0) pair makes the source
        # slope jump from kprod*(1-a) to kdest*a EXACTLY at dpv = 0 -- a factor kprod/kdest (100 at the
        # production asymmetry). A point-implicit iteration whose cells straddle that kink oscillates.
        # MEASURED (cmesh_cav_tem_shed.jl, dual-time BDF2): stable when SYMMETRIC (Cprod=Cdest=300 ->
        # rstar 1.1e-4, and IMPROVING under refinement) but divergent as soon as a lag is added at the
        # same stiffness (Cdest=30 -> 1.9e2, Cdest=3 -> 3.9e4 -> NaN). Softening the rate fixed it
        # (Cprod=30,Cdest=0.3 -> 9.7e-3), confirming stiffness x asymmetry, not the ratio alone.
        # Same smooth-max form already used for the pressure, so it degrades gracefully:
        # src_smooth = 0 reproduces the hard switch EXACTLY (bitwise legacy).
        E = 0.0f0; D = 0.0f0; wE = 0.0f0; wD = 0.0f0
        if src_smooth > 0.0f0
            sw = sqrt(dpv*dpv + src_smooth*src_smooth)
            E = kprod*0.5f0*(dpv + sw)                        # -> kprod*dpv  for dpv >>  src_smooth
            D = kdest*0.5f0*(sw - dpv)                        # -> kdest*|dpv| for dpv << -src_smooth
            wE = 0.5f0*(1.0f0 + dpv/sw)                       # dE/d(dpv)/kprod
            wD = 0.5f0*(1.0f0 - dpv/sw)                       # -dD/d(dpv)/kdest
        else
            E = dpv > 0.0f0 ? kprod*dpv  : 0.0f0              # evaporation coeff (acts on 1-alpha)
            D = dpv < 0.0f0 ? kdest*(-dpv) : 0.0f0            # condensation coeff (acts on alpha)
            wE = dpv > 0.0f0 ? 1.0f0 : 0.0f0
            wD = dpv < 0.0f0 ? 1.0f0 : 0.0f0
        end
        u0 = ux[ia, ja]; v0 = uy[ia, ja]
        adv = 0.0f0
        at_pressure_outlet = outlet_mode != Int32(0) && (i == Int32(1) || i == NI)
        if cons >= Int32(1) || at_pressure_outlet
            # ★ P3a: CONSERVATIVE transport div(alpha*u)/vol — upwinded face volume-fluxes (see
            # TEMState.cons doc). Six[i,j] is the shared i-face between cells i-1 and i (GCL storage);
            # face velocity = arithmetic average of the two adjacent cell centres; alpha upwinded by
            # the face flux sign. Includes alpha*div(u) naturally — the term the legacy form drops.
            Ulo = 0.5f0*((ux[ia-1,ja]+u0)*Six[i,j]   + (uy[ia-1,ja]+v0)*Siy[i,j])
            Uhi = 0.5f0*((u0+ux[ia+1,ja])*Six[i+1,j] + (v0+uy[ia+1,ja])*Siy[i+1,j])
            Vlo = 0.5f0*((ux[ia,ja-1]+u0)*Sjx[i,j]   + (uy[ia,ja-1]+v0)*Sjy[i,j])
            Vhi = 0.5f0*((u0+ux[ia,ja+1])*Sjx[i,j+1] + (v0+uy[ia,ja+1])*Sjy[i,j+1])
            if at_pressure_outlet
                # The outlet ghosts hold the characteristic FACE state. Use that
                # same volume flux as the carrier, including when correction reverses
                # its sign. An interior/ghost average can select the opposite donor.
                if i == Int32(1)
                    Ulo = ux[ia-1,ja]*Six[i,j] + uy[ia-1,ja]*Siy[i,j]
                else
                    Uhi = ux[ia+1,ja]*Six[i+1,j] + uy[ia+1,ja]*Siy[i+1,j]
                end
            end
            alo = _afaceA(alpha[ia-2,ja], _aim1, a0, alpha[ia+1,ja], Ulo >= 0.0f0, arecon)
            ahi = _afaceA(_aim1, a0, alpha[ia+1,ja], alpha[ia+2,ja], Uhi >= 0.0f0, arecon)
            blo = _afaceA(alpha[ia,ja-2], alpha[ia,ja-1], a0, alpha[ia,ja+1], Vlo >= 0.0f0, arecon)
            bhi = _afaceA(alpha[ia,ja-1], a0, alpha[ia,ja+1], alpha[ia,ja+2], Vhi >= 0.0f0, arecon)
            adv = (Uhi*ahi - Ulo*alo + Vhi*bhi - Vlo*blo)/vol[i,j]
            rate_adv = (abs(Uhi) + abs(Ulo) + abs(Vhi) + abs(Vlo))/vol[i,j]
            # a0 enters through the OUTFLOW faces only: ahi=a0 when Uhi>=0, alo=a0 when Ulo<0, etc.
            cout = (max(Uhi, 0.0f0) + max(-Ulo, 0.0f0) +
                    max(Vhi, 0.0f0) + max(-Vlo, 0.0f0))/vol[i,j]
            if cons >= Int32(2) || (at_pressure_outlet && cons == Int32(0))
                # ★ MATCHED SOURCE RECAST (cons=2, 2026-07-22). div(alpha u) = u.grad(alpha) +
                # alpha*div(u); the validated Merkle source E(1-a)-D*a is calibrated for the
                # NON-conservative u.grad(alpha) operator, so the bare conservative form (cons=1)
                # shifts the equilibrium by alpha*div(u) — MEASURED: sigma=0.90 warm-up saturates
                # (amax=1.000 vs production 0.644), the root cause of the P3 round-1 bring-up
                # failure (design doc §GATE RESULTS). Subtracting a0*div(u) with the SAME face
                # volume-fluxes makes the operator the FLUX-FORM u.grad(alpha): telescoping
                # face transport, and the fixed point is the validated production PDE exactly.
                # (In Kunz this coupling is his row-p volume source mdot*(1/rho_l - 1/rho_v);
                # with our compressible carrier div(u) is flow-determined, hence the explicit form.)
                adv -= a0*(Uhi - Ulo + Vhi - Vlo)/vol[i,j]
                cout -= (Uhi - Ulo + Vhi - Vlo)/vol[i,j]   # the recast removes a0*div(u) too
            end
        else
            Uci = (u0*0.5f0*(Six[i,j]+Six[i+1,j]) + v0*0.5f0*(Siy[i,j]+Siy[i+1,j]))/vol[i,j]
            Ucj = (u0*0.5f0*(Sjx[i,j]+Sjx[i,j+1]) + v0*0.5f0*(Sjy[i,j]+Sjy[i,j+1]))/vol[i,j]
            if arecon <= Int32(0)
                adv = (Uci >= 0.0f0 ? Uci*(a0-_aim1) : Uci*(alpha[ia+1,ja]-a0)) +
                      (Ucj >= 0.0f0 ? Ucj*(a0-alpha[ia,ja-1]) : Ucj*(alpha[ia,ja+1]-a0))
            else
                pi_ = Uci >= 0.0f0; pj_ = Ucj >= 0.0f0
                aloN = _afaceA(alpha[ia-2,ja], _aim1, a0, alpha[ia+1,ja], pi_, arecon)
                ahiN = _afaceA(_aim1, a0, alpha[ia+1,ja], alpha[ia+2,ja], pi_, arecon)
                bloN = _afaceA(alpha[ia,ja-2], alpha[ia,ja-1], a0, alpha[ia,ja+1], pj_, arecon)
                bhiN = _afaceA(alpha[ia,ja-1], a0, alpha[ia,ja+1], alpha[ia,ja+2], pj_, arecon)
                adv = Uci*(ahiN - aloN) + Ucj*(bhiN - bloN)
            end
            rate_adv = abs(Uci) + abs(Ucj)
            # For EITHER sign the a0 coefficient is +|U|:  U>=0 gives +U*a0, U<0 gives -U*a0 = |U|*a0.
            cout = rate_adv
        end
        # ★★ VOID-EQUATION PSEUDO-TIME STEP (2026-07-29). The void equation was marched with the
        # FLOW's dtf, which is set by the CARRIER ACOUSTIC CFL ~ Delta/(3*sqrt(T)). The void equation
        # is advection + a POINT-IMPLICIT source: it is limited by |u|/Delta, not by the lattice
        # sound speed, and the source imposes no explicit step limit at all. In the first wall cell
        # the two differ by ~1000x, so the near-wall void was effectively FROZEN in pseudo-time.
        # MEASURED (memory/project_docs/2026-07-29_THE_WALL_LAYER_IS_AN_LTS_ARTIFACT.md): predicted alpha_wall
        # 1-exp(-sum dtl*E) = 0.28 (dn1=2e-5) / 0.96 (dn1=2e-4) vs measured 0.14-0.20 / 0.84-0.99,
        # and 4x the iterations lifted max wall alpha 0.197 -> 0.556 with reversed stations 13 -> 34.
        # dtboost = 1 takes dtl = dtf EXACTLY (bitwise legacy); dtboost > 1 allows up to that factor,
        # still bounded by the void's own advective CFL. Fixed-point-preserving: only the relaxation
        # PATH to Ra = 0 changes, never the fixed point.
        dtl = dtf[i, j]
        # ★ REGIME GATE (2026-07-29, MEASURED): the boost is a STEADY/LTS accelerator only. In the
        # dual-time march (dtp>0) boosting the void step while the FLOW still marches on dtf
        # decouples the operator split: alpha races to its local equilibrium against a density field
        # that has not moved, which makes more vapour, which drops rho further. MEASURED at
        # dtboost=1000, TCTU=400: V_vapour mean 6.12 (400x the unboosted state), rstar L2 4.1e-01 =
        # 640x the control, Gate B FAILED on the threshold-artefact branch. With dtp=0 the same boost
        # is pure relaxation toward the fixed point and produces the converged dense wall layer and
        # the 0.89U re-entrant jet. Mirrors the p3 path's regime gate, in the opposite direction.
        # ★★★ DUAL-TIME BOOST (2026-07-31). `dtb_dt` opens the SAME mechanism for dtp > 0, where the
        # contraction analysis in the struct docstring says it is actually needed (the transport region
        # has no source on the diagonal, so only dtl*1.5/dtp remains and it is 3e-4..3e-2). dtb_dt = 1
        # leaves `_dtb = 1` in dual time, the branch is not taken, and dtl = dtf EXACTLY => BITWISE
        # LEGACY. With dtp == 0 `_dtb` is `dtboost` and the steady path is untouched, also bitwise.
        # The advective-CFL cap is retained in BOTH regimes: the source is point-implicit and imposes
        # no explicit limit, but the advection is explicit and does.
        _dtb = dtp == 0.0f0 ? dtboost : dtb_dt
        if _dtb > 1.0f0
            dtl_adv = 0.5f0/(rate_adv + 1.0f-20)          # CFL_alpha = 0.5 on the void advection
            dtl_cap = _dtb*dtl
            dtl = dtl_adv < dtl ? dtl : (dtl_adv > dtl_cap ? dtl_cap : dtl_adv)
        end
        # ★★★ DRIFT-FLUX PHASE SLIP (2026-08-01). slip == 0 => branch not taken => BITWISE LEGACY.
        #
        # The homogeneous mixture assumption forces vapour and liquid onto ONE velocity field. Real
        # vapour is ~100x lighter here, so a pressure gradient accelerates it far harder than the
        # liquid and a detached cloud does NOT simply convect with the mixture. Algebraic drift-flux:
        #       u_r = -slip_tau * (1/rho_v_sat - 1/rho) * grad(p)
        # and the void equation gains  u_r . grad( alpha (1-alpha) ).  The alpha(1-alpha) weighting is
        # standard: a RELATIVE velocity is meaningless in either pure phase, and it vanishes there.
        #
        # p is reconstructed as rho*cs2*T + pexc -- the IDENTICAL expression `_k_tem_commit!` writes,
        # from arrays already resident -- so this costs four array reads per direction and NO extra EOS
        # evaluation. Green-Gauss gradient on the same metric projection the advective velocities use.
        #
        # ⚠ DISCRETISED IN THE NON-CONSERVATIVE FORM u_r.grad(f) rather than div(f u_r); the two differ
        # by f*div(u_r). That is the SAME approximation the base cons==0 advection already makes, so it
        # is consistent with the operator it is added to -- but under cons>=1 (conservative transport)
        # the two halves are then formally mismatched. Prefer slip with cons==0 until a conservative
        # drift flux is written, and do not silently mix them.
        if slip > 0.0f0
            pC  = r*TEM_CS2*Tf[ia,ja]     + pexc[ia,ja]
            pIp = rho[ia+1,ja]*TEM_CS2*Tf[ia+1,ja] + pexc[ia+1,ja]
            pIm = rho[ia-1,ja]*TEM_CS2*Tf[ia-1,ja] + pexc[ia-1,ja]
            pJp = rho[ia,ja+1]*TEM_CS2*Tf[ia,ja+1] + pexc[ia,ja+1]
            pJm = rho[ia,ja-1]*TEM_CS2*Tf[ia,ja-1] + pexc[ia,ja-1]
            gi  = 0.5f0*(pIp - pIm); gj = 0.5f0*(pJp - pJm)
            iv  = 1.0f0/vol[i,j]
            dpdx = (gi*0.5f0*(Six[i,j]+Six[i+1,j]) + gj*0.5f0*(Sjx[i,j]+Sjx[i,j+1]))*iv
            dpdy = (gi*0.5f0*(Siy[i,j]+Siy[i+1,j]) + gj*0.5f0*(Sjy[i,j]+Sjy[i,j+1]))*iv
            # (1/rho_v - 1/rho): POSITIVE in liquid-dominated cells (rho > rho_v_sat), so vapour is
            # driven UP the pressure gradient relative to the mixture, i.e. toward low pressure. That
            # is the correct sign -- a bubble migrates toward the suction peak, not away from it.
            # ⚠ density floor ALIGNED to the solver's own 1e-4 (CLAUDE.md: "solver density floor is
            # 1e-4"). I originally wrote 1e-6 here, which would have computed 1/rho against a
            # DIFFERENT floor from the rest of the solver -- a silent inconsistency of exactly the
            # kind the epsilon-floor audit was looking for.
            dinv = 1.0f0/rho_v_sat - 1.0f0/(r < 1.0f-4 ? 1.0f-4 : r)
            urx  = -slip*slip_tau*dinv*dpdx
            ury  = -slip*slip_tau*dinv*dpdy
            Uri = (urx*0.5f0*(Six[i,j]+Six[i+1,j]) + ury*0.5f0*(Siy[i,j]+Siy[i+1,j]))*iv
            Urj = (urx*0.5f0*(Sjx[i,j]+Sjx[i,j+1]) + ury*0.5f0*(Sjy[i,j]+Sjy[i,j+1]))*iv
            fC  = a0*(1.0f0-a0)
            fIm = alpha[ia-1,ja]*(1.0f0-alpha[ia-1,ja]); fIp = alpha[ia+1,ja]*(1.0f0-alpha[ia+1,ja])
            fJm = alpha[ia,ja-1]*(1.0f0-alpha[ia,ja-1]); fJp = alpha[ia,ja+1]*(1.0f0-alpha[ia,ja+1])
            adv += (Uri >= 0.0f0 ? Uri*(fC-fIm) : Uri*(fIp-fC)) +
                   (Urj >= 0.0f0 ? Urj*(fC-fJm) : Urj*(fJp-fC))
            # the drift raises the local transport rate, so it must enter the pseudo-step CFL bound too
            rate_adv += abs(Uri) + abs(Urj)
        end
        # ★★★★ RAYLEIGH-PLESSET (2026-08-01). rp == 0 => branch not taken => BITWISE LEGACY MERKLE.
        #
        # Placed HERE, after `dtl` and `adv` and before the num/den assembly, because it needs the
        # pseudo-step and must OVERWRITE E and D before they are consumed. Physics + validation:
        # src/cmesh/cmesh_rayleigh_plesset.jl, tools/test_rayleigh_plesset.jl.
        #
        # TWO THINGS HAPPEN:
        #  1. The alpha source E(1-a) - D*a is replaced by (dalpha/dR)*Rdot, exactly (unit test T3).
        #     Growth still lands on the (1-a) branch and collapse on the a branch, so the point-implicit
        #     structure, the clamp and the residual below are untouched.
        #  2. Rdot itself is advanced: advection + the full R-P balance, point-implicit on the terms
        #     that are genuinely dissipative. THIS is where the inertia lives, and it is why the model
        #     can overshoot where a first-order relaxation cannot.
        #
        # ⚠ wE and wD are ZEROED. They are dE/d(dpv) weights for Merkle's pressure feedback, and under
        # R-P the pressure does not act on the alpha equation at all -- it acts on the Rdot equation
        # through `Arp`. Leaving them non-zero would add a Jacobian term for a dependence that no longer
        # exists. The `src_jac` block below is correspondingly gated off for rp > 0; see the struct doc.
        if rp > 0.0f0
            rd0 = rdot[ia, ja]
            # ★ S3e HETEROGENEOUS WALL NUCLEATION then ★ S3c POLYDISPERSE ACTIVATION, in that order:
            # the surface excess sets how many nuclei are PRESENT, the Laplace barrier sets how many of
            # them are ACTIVATABLE at the local suction. Both collapse to rp_n0 exactly at knob=0.
            n_eff = rp_wall_nuclei(rp_n0, Float32(j - 1), wallnuc, wall_decay)
            n_eff = rp_poly_nactive(n_eff, dpv, rp_S, poly_Rmin, poly_Rmax, poly_e, poly)
            Rb  = rp_R_of_alpha(a0, n_eff, rp_R0)
            # ★★ rp >= 2: THE QUASI-STATIC (Schnerr-Sauer) LIMIT -- the usable setting, and the
            # physically correct one here by five orders of magnitude. No Rdot transport, no history,
            # no stiffness: Rdot is algebraic in the local pressure. See rp_rdot_quasistatic's
            # docstring and docs/2026-08-01_RP_IS_QUASI_STATIC_A_PRIORI.md.
            #
            # rp == 1: the full inertial system, SUB-CYCLED. `rp_nsub` returns how many sub-steps the
            # oscillator needs at this pseudo-step; -1 means it cannot be resolved within the cap, in
            # which case we fall back to quasi-static FOR THAT CELL. That degradation is not a dodge:
            # being unable to resolve the oscillation and the oscillation being irrelevant are the
            # same statement at these timescales, and it is what stops the explicit restoring force
            # from diverging the way the first rp=1 smoke test did.
            # ★★★ COLLECTIVE / CLOUD DYNAMICS (d'Agostino & Brennen). A bubble inside a CLUSTER must
            # displace the surrounding bubbly medium, not pure liquid, so its response is slowed by
            # 1/sqrt(1+beta) with beta = alpha(1-alpha)(A/R)^2. The cloud radius A is estimated from
            # the LOCAL void-fraction gradient length, A ~ alpha/|grad alpha| -- computed here with the
            # same Green-Gauss metric projection the drift-flux term uses.
            #
            # ⚠ Unlike the lag ladder, this lag is SPATIALLY VARYING: dense cores respond more slowly
            # than dilute edges. The uniform Cprod ladder could not produce that and therefore did not
            # test it. cloudint = 0 => beta = 0 => the factor is exactly 1 => bitwise legacy.
            # ⛔⛔ GATED TO THE INERTIAL MODEL ONLY (rp < 2), and this is a PHYSICS correction, not a
            # tuning choice. Brennen's beta screens INERTIA: a bubble in a cluster must displace the
            # surrounding bubbly medium, which adds mass and LOWERS THE NATURAL FREQUENCY. In the
            # QUASI-STATIC limit there is no inertia at all, so there is no added mass to screen, and
            # multiplying a kinematic growth rate by 1/sqrt(1+beta) is a category error.
            #
            # It is not a small one. MEASURED at the benchmark state: beta reaches 1.3e4 at alpha=1e-4
            # and 3.6e5 at alpha=1e-2 (because A/R is a CLOUD scale over a BUBBLE scale, ~4e4, and it
            # is SQUARED), so cldfac = 8.9e-3 and 1.7e-3 -- a 100-600x cut in the growth rate. The R-P
            # source alone gives alpha = 0.978 over the warm-up; with this factor applied it stalls at
            # zero. That is what produced amax = 0.000 in BOTH all-on attempts.
            # Under rp=1 the factor is correct and stays. Collective effects reach the quasi-static
            # model through the PRESSURE FIELD, which the solver already resolves.
            cldfac = 1.0f0
            if cloudint > 0.0f0 && rp < 2.0f0
                giA = 0.5f0*(alpha[ia+1,ja] - alpha[ia-1,ja])
                gjA = 0.5f0*(alpha[ia,ja+1] - alpha[ia,ja-1])
                ivA = 1.0f0/vol[i,j]
                dax = (giA*0.5f0*(Six[i,j]+Six[i+1,j]) + gjA*0.5f0*(Sjx[i,j]+Sjx[i,j+1]))*ivA
                day = (giA*0.5f0*(Siy[i,j]+Siy[i+1,j]) + gjA*0.5f0*(Sjy[i,j]+Sjy[i,j+1]))*ivA
                bet = rp_cloud_beta(a0, Rb, sqrt(dax*dax + day*day), cloudint)
                cldfac = 1.0f0/sqrt(1.0f0 + bet)
            end
            local rd_eff::Float32
            if rp >= 2.0f0
                rd_eff = cldfac*rp_rdot_quasistatic(Rb, pl, p_v_loc, rho_l_sat, rp_S, rp_pg0, rp_R0, rp_kappa)
                rdot_new[ia, ja] = rd_eff          # stored for diagnostics only; never read back
            else
                nsub = rp_nsub(dtl, rp_R0, rho_l_sat, rp_pg0, rp_S, rp_kappa, Int32(32))
                if nsub < Int32(0)
                    rd_eff = rp_rdot_quasistatic(Rb, pl, p_v_loc, rho_l_sat, rp_S, rp_pg0, rp_R0, rp_kappa)
                    rdot_new[ia, ja] = rd_eff
                else
                    rd_eff = rd0
                end
            end
            (E, D) = rp_ED(Rb, a0, rd_eff, n_eff, rp_R0)
            # ★ S3f BUBBLE-INDUCED TKE — the ONE place in the code where alpha, Rdot and R coexist, so
            # the source is computed here and handed to the SST k equation through `s.btke_src` rather
            # than duplicating the formula inside CMeshSolverKA (a separate module, included earlier).
            # Identically zero when btke = 0, so the k equation stays bitwise legacy.
            if btke > 0.0f0
                btke_src[i, j] = rp_tke_source(a0, rd_eff, Rb, btke)
            end
            # ★ S3d HERTZ-KNUDSEN. Merkle and R-P both assume the interface sits AT saturation; kinetic
            # theory bounds the molecular flux across it. Applied to BOTH coefficients so the ceiling is
            # symmetric in evaporation and condensation -- an asymmetric cap would bias the equilibrium
            # void, which is the `src_smooth` trap this campaign already paid for once.
            if hk > 0.0f0
                E = rp_hk_limit(E, a0, Rb, p_v, rho_v_sat, hk)
                D = rp_hk_limit(D, a0, Rb, p_v, rho_v_sat, hk)
            end
            wE = 0.0f0; wD = 0.0f0
            # --- Rdot transport (non-conservative upwind, same cell-centred contravariant velocities
            # the cons==0 alpha branch uses; recomputed locally so no existing branch is disturbed) ---
            Uci_r = (u0*0.5f0*(Six[i,j]+Six[i+1,j]) + v0*0.5f0*(Siy[i,j]+Siy[i+1,j]))/vol[i,j]
            Ucj_r = (u0*0.5f0*(Sjx[i,j]+Sjx[i,j+1]) + v0*0.5f0*(Sjy[i,j]+Sjy[i,j+1]))/vol[i,j]
            adv_rd = (Uci_r >= 0.0f0 ? Uci_r*(rd0 - rdot[ia-1,ja]) : Uci_r*(rdot[ia+1,ja] - rd0)) +
                     (Ucj_r >= 0.0f0 ? Ucj_r*(rd0 - rdot[ia,ja-1]) : Ucj_r*(rdot[ia,ja+1] - rd0))
            # --- the Rayleigh-Plesset balance: Rddot = Arp - Brp*rdot - Qrp -------------------------
            # `pl` is the positivity-guarded mixture pressure computed above -- the same p the Merkle
            # source saw, so the two models are driven by an identical pressure field.
            (Arp, Brp, Qrp) = rp_rhs(Rb, rd0, pl, p_v, rho_l_sat, rp_nu, rp_S, rp_pg0, rp_R0, rp_kappa)
            # Split the quadratic on the sign of rdot: implicit while the bubble GROWS (dissipative),
            # explicit while it COLLAPSES (there it is a destabilising source -- that IS Rayleigh
            # collapse -- and putting it on the diagonal would drive den below 1). See rp_rhs's
            # docstring, and the sign trap unit test T4 caught.
            Cpos = 1.5f0*rd0/Rb
            Cpos = Cpos < 0.0f0 ? 0.0f0 : Cpos
            rnum = rd0 + dtl*(Arp - (Qrp - Cpos*rd0) - adv_rd)
            rden = 1.0f0 + dtl*(Brp + Cpos)
            if dtp > 0.0f0                                   # same BDF2 dual-time form as alpha
                rnum += (dtl/(2.0f0*dtp))*(4.0f0*rdot_n[i,j] - rdot_m[i,j])
                rden += 1.5f0*dtl/dtp
            end
            rdot_new[ia, ja] = rnum/rden
        end
        # point-implicit: E*(1-alpha) = E - E*alpha (E on diag), -D*alpha (D on diag)
        # ★★★★ TIER 4 -- THERMODYNAMIC TEMPERATURE UPDATE. tmix == 0 => branch not taken, dTm and
        # dTm_new are 1x1 stubs and never touched => BITWISE LEGACY.
        #
        #     d(dTm)/dt + u.grad(dTm) = -(tmix_lat/rho) * S_alpha ,   S_alpha = E*(1-a) - D*a
        #
        # Evaporation (S_alpha > 0) COOLS the cell; condensation warms it. Discretised in the SAME
        # non-conservative upwind form as the cons==0 alpha operator, on the SAME pseudo-step `dtl`,
        # so the two fields are advanced consistently rather than on different clocks -- the defect
        # that froze the near-wall void in 2026-07-29 and that this campaign has paid for twice.
        # ⚠ The source is EXPLICIT and that is correct here, not lazy: S_alpha does not depend on dTm
        # within an iteration (E and D are computed from the pressure above), so there is no diagonal
        # term to make implicit. The tvp coupling closes the loop only across iterations.
        if tmix > 0.0f0
            t0 = dTm[ia, ja]
            Uct = (u0*0.5f0*(Six[i,j]+Six[i+1,j]) + v0*0.5f0*(Siy[i,j]+Siy[i+1,j]))/vol[i,j]
            Vct = (u0*0.5f0*(Sjx[i,j]+Sjx[i,j+1]) + v0*0.5f0*(Sjy[i,j]+Sjy[i,j+1]))/vol[i,j]
            advT = (Uct >= 0.0f0 ? Uct*(t0 - dTm[ia-1,ja]) : Uct*(dTm[ia+1,ja] - t0)) +
                   (Vct >= 0.0f0 ? Vct*(t0 - dTm[ia,ja-1]) : Vct*(dTm[ia,ja+1] - t0))
            dTm_new[ia, ja] = t0 + dtl*(-advT - tmix_lat*(E*(1.0f0-a0) - D*a0)/r)
        end
        num = a0 + dtl*(E - adv)
        den = 1.0f0 + dtl*(E + D)
        # ★★★ POINT-IMPLICIT OUTFLOW (2026-07-31, `advimp`). The SOURCE is point-implicit here but the
        # ADVECTION is entirely explicit, so the outflow coefficient multiplies the OLD a0. Splitting
        # adv = cout*a0 - (inflow) and moving the cout*a0 piece onto the diagonal makes the outflow
        # implicit, which is the standard first step and the half of the swept-transport fix that
        # needs NO SEQUENCING -- it is embarrassingly parallel and stays in this kernel.
        #
        # FIXED-POINT PRESERVING, exactly. num' = num + dtl*cout*a0, den' = den + dtl*cout, so at
        # an = a0:  a0*den' = num'  <=>  a0*den + a0*dtl*cout = num + dtl*cout*a0  <=>  a0*den = num,
        # which is the original condition. Only the relaxation path changes, never the fixed point --
        # the same property `dtboost` has, and it is what makes this safe to A/B.
        #
        # WHY IT SHOULD HELP, and the measurement it comes from: the floor the void residual stalls at
        # depends on `cflp` ALONE (NINNER inert across 8x at three cflp values; Cprod non-monotone;
        # rstar flat), as floor ~ cflp^-0.707, and `cflp` is capped near 1 because the transport is
        # EXPLICIT. Making the outflow implicit raises that cap. It also raises `den` directly, which
        # lowers rho = 1/den. See docs/2026-07-31_THE_FLOOR_IS_JACOBI_TRANSPORT_...md §0b.
        #
        # advimp = 0 => branch not taken => BITWISE LEGACY. The max(cout,0) guard matters for cons=2,
        # whose div(u) recast can drive cout negative; adding a negative to the diagonal would REDUCE
        # diagonal dominance, which is the opposite of the intent.
        if advimp > 0.0f0
            _co = advimp*(cout < 0.0f0 ? 0.0f0 : cout)
            num += dtl*_co*a0
            den += dtl*_co
        end
        # ★ FULL SOURCE JACOBIAN (src_jac >= 1). The legacy diagonal above keeps only -(E+D), i.e. it
        # treats E and D as CONSTANT w.r.t. alpha. They are not: they depend on alpha THROUGH THE PRESSURE,
        #     p = p_v + a2m*(rho - rho_sat(alpha)),   dp/dalpha = a2m*(rho_l_sat - rho_v_sat)
        # so dE/dalpha = -kprod*dp/dalpha. At the production rate (Cprod=30, eps_pv=0.1, sigma=0.70:
        # kprod = 30*U/p_v ~ 4434, dp/dalpha ~ 0.905) that is ~ -4013, while (E+D) -> 0 AT EQUILIBRIUM.
        # The legacy denominator is therefore ~1 exactly where the true one is ~1 + dtl*4013: the DOMINANT
        # term is missing. That is the poor linearisation behind the dual-time stiffness band -- measured
        # divergence for tau_cond in [0.033, 0.333] CTU (rstar up to 9e4), stable outside it in both
        # directions, which is the classic point-implicit failure window.
        #   -dS/dalpha = (E + D) + dp/dalpha*(kprod*wE*(1-a) + kdest*wD*a)
        # src_jac = 0 reproduces the legacy diagonal EXACTLY (bitwise).
        # CONSISTENCY: linearising S(a) ~ S(a0) + (a-a0)*dS/da gives a = a0 + dt*S(a0)/(1 + dt*J), which
        # expands to num = a0 + dt*E + dt*a0*J. Adding J to the DENOMINATOR ALONE biases every cell toward
        # alpha = 0 -- MEASURED: rstar fell 5.6e+02 -> 9.0e-05 (the divergence really was this term) but
        # the CAVITY VANISHED (amax 0.818 -> 0.026, lcav 0.987 -> 0.000). The legacy J = E+D is
        # self-consistent because its a0*J cancels identically; the pressure-feedback term does not.
        # ★★★★ GATED OFF UNDER RAYLEIGH-PLESSET, and this is a physics statement, not an omission.
        # Every term in this block is the linearisation of Merkle's dE/dalpha and dD/dalpha THROUGH THE
        # PRESSURE. Under rp > 0 the pressure does not enter the alpha equation at all -- it enters the
        # Rdot equation, which carries its own point-implicit treatment above. The correct
        # pressure-feedback Jacobian of the alpha equation under R-P is therefore exactly ZERO.
        # rp == 0 leaves the condition character-identical => bitwise legacy.
        if src_jac >= Int32(1) && rp <= 0.0f0
            # ★★ dp/dalpha VANISHES on the bounded vapour branch (p = a2v*rho carries no alpha).
            # dpda_scale == 1.0f0 under vapbound=0, and a multiply by 1.0f0 is IEEE-exact => bitwise.
            # ★★★ Under vaploc=0 `dpda_raw` was assigned by that IDENTICAL expression above (same
            # operands, same association) => bitwise. Under vaploc>0 it is the implicit dp/dalpha.
            dpda = dpda_raw
            # ★ P1 KINK FIX (src_jac = 2/3, 2026-07-21). Under the HARD switch (src_smooth = 0) both wE
            # and wD are 0 AT dpv == 0 exactly (strict inequalities above), so the legacy jacp VANISHES
            # at the switch point -- precisely where cavity-interface cells sit (p ~ p_v), leaving them
            # with no pressure-feedback damping at all.
            #   Mode 2: floor the diagonal with the one-sided slope MAX EVERYWHERE. MEASURED: kills the
            #   sigma=0.75/ni200 NaN divergence (JAC1 NaN@t=158 -> 0 NaN, stationary), BUT the
            #   everywhere-active damping biases the finite-NINNER coupled statistics -- sigma=0.90/ni100
            #   converged to Cl 0.60 (T1800, flat) instead of the 0.744 attractor. jacp is
            #   consistency-neutral ONLY in the converged-inner-loop limit; at finite NINNER it shifts
            #   the trajectory. G1 FAILED for mode 2.
            #   Mode 3: the KINK-LOCAL floor -- legacy jacp away from the switch (zero tax), the MAX
            #   floor only inside |dpv| < 0.10*p_v (linear hat), Jacobian-only (the SOURCE value is
            #   untouched, so no alpha-bias of the src_smooth kind can enter).
            # Modes 0/1 are bitwise-unchanged. See memory/project_docs/2026-07-21_IMPLICIT_SOURCE_BUILD_PLAN.md P1.
            jacp = if src_jac >= Int32(3)
                wband = 1.0f0 - min(abs(dpv)/(0.10f0*p_v), 1.0f0)
                max(dpda*(kprod*wE*(1.0f0-a0) + kdest*wD*a0),
                    wband*dpda*max(kprod*(1.0f0-a0), kdest*a0))
            elseif src_jac >= Int32(2)
                dpda*max(kprod*(1.0f0-a0), kdest*a0)
            else
                dpda*(kprod*wE*(1.0f0-a0) + kdest*wD*a0)
            end
            num += dtl*a0*jacp
            den += dtl*jacp
        end
        # ★ P2a: Kunz phasic-continuity-enforcing coupling (see TEMState.enf doc). PDE term
        # (a0/(rho*b2))*dp/dtau, discretised with the SAME pseudo-clock as the flow (dtl = dtf), so the
        # dtl cancels: LHS += (a0/(r*b2))*(dpdr*drho_flow + dpda*dalpha). In num/den form:
        #   den += cE,  num += cE*a0 - cR*drho_flow
        # (BOTH num and den get the cE piece -- denominator-only would be the documented alpha->0 bias
        # trap). Fixed point untouched: at pseudo-convergence drho_flow -> 0 and alpha_new -> a0, both
        # additions cancel. drho_flow = rho - rk_r = this inner iteration's net RK3 density change.
        if enf >= Int32(1)
            b2     = max(beta2, u0*u0 + v0*v0)
            sq_e   = sqrt(pmx*pmx + psm*psm)
            fac_sm = 0.5f0*(1.0f0 + pmx/sq_e)
            # ★★ same derivative rule as src_jac: cE carries dp/dalpha (scaled to 0 on the bounded
            # branch), cR carries dp/drho (a2v there, a2m otherwise). Bitwise at vapbound=0.
            # ★★★ the vaploc==0 arm is the LEGACY EXPRESSION CHARACTER-IDENTICAL (re-associating this
            # product through dpda_raw is algebraically equal but NOT bitwise in Float32).
            cE = vaploc > 0.0f0 ? a0*fac_sm*dpda_raw/(r*b2) :
                                  a0*fac_sm*dpda_scale*a2m*(rho_l_sat - rho_v_sat)/(r*b2)
            cR = a0*fac_sm*dpdr_raw/(r*b2)
            num += cE*a0 - cR*(r - rk_r[i,j])
            den += cE
        end
        if dtp > 0.0f0                                        # dual-time BDF2 (time-accurate)
            h = dtl/(2.0f0*dtp)
            num += h*(4.0f0*un[i,j] - um[i,j])
            den += 1.5f0*dtl/dtp
        end
        an = num/den
        ac = an < 0.0f0 ? 0.0f0 : (an > 1.0f0 ? 1.0f0 : an)
        alpha_new[ia, ja] = ac
        # ★★★ TRUE VOID RESIDUAL (2026-07-31). The commit kernel used to write |an - a0| and call it
        # "the true alpha residual". It is not: from num/den above,
        #     an - a0 = dtl * [ E(1-a0) - D*a0 - adv - (3a0 - 4*un + um)/(2*dtp) ] / den
        # so that quantity is the PDE residual SCALED BY THE LOCAL PSEUDO-TIMESTEP. dtl = dtf spans
        # ~1e3 across this mesh and is SMALLEST in the near-wall cavity cells, so the old measure was a
        # mesh-dependent mask, tightest exactly where the physics is -- which is why the void's failure
        # to converge was invisible for the whole campaign while `rstar` (the FLOW residual) read
        # CONVERGED. See docs/2026-07-31_ROOT_CAUSE_THE_VOID_EQUATION_IS_SOLVED_AS_A_STEADY_PROBLEM.md.
        # Multiplying back by den/dtl recovers the residual of the actual PDE, comparable across cells,
        # meshes and runs. The CLAMPED value is used so a cell legitimately railed at 0 or 1 reads
        # converged once it stops moving, rather than reporting a residual forever.
        # ⚠ `ares` therefore MEANS SOMETHING DIFFERENT from every record before this commit, and is not
        # comparable with them. The old values were not a residual, so nothing is lost.
        res[i, j] = abs(ac - a0)*den/max(dtl, 1.0f-30)
    end   # if i <= NI
    end   # for _sw  (runs exactly once when gs == 0 => bitwise legacy)
    end   # @inbounds if j <= NJ
end

# ---- commit: alpha <- alpha_new; impose the mixture pressure via the PDP pressure-split (T pedestal +
# pexc excess), so the finite-rate TEM breaks the 3:1 cap exactly like HEM.
#
# THE STIFFNESS FIX (eos=1, default). The legacy form p = p_v + a2l*(rho - rho_sat(alpha)) applies the
# LIQUID stiffness a2l across the WHOLE mixture. With a2l/q ~ 50, the finite-rate alpha lag (which is the
# whole point of the TEM) turns a 0.1 density-mismatch into dCp ~ 5 -> the Cl~8 blow-up of the 2026-07-18
# session. That form is not just stiff, it is physically WRONG: a liquid/vapor mixture does not have the
# liquid's stiffness. The correct frozen-alpha mixture compressibility is the WALLIS harmonic average of
# the phase compressibilities (Wallis 1969; the same sound-speed COLLAPSE the HEM barotrope's dp/drho
# already reproduces):
#     1/a2m(alpha) = (1-alpha)/a2l + alpha/a2v      =>  a2m -> a2l  (pure liquid, unchanged)
#                                                       a2m -> a2v  (pure vapor, soft)
#                                                       a2m ~ a2l/500 at alpha=0.5 (the collapse)
# so the SAME alpha lag now produces a BOUNDED dCp instead of an unphysical pressure spike, while the
# liquid region (alpha=0) is bit-for-bit the legacy liquid branch. A smooth-max against the vapor plateau
# p_vap = p_v + a2v*rho floors the pressure at ~p_v (guards the metastable low-alpha/low-rho corner where
# the linear branch would go strongly negative). In the fast-rate limit alpha -> alpha_eq the mixture
# branch collapses onto p = p_v, i.e. the TEM recovers the validated HEM plateau EXACTLY - so eos=1 is a
# strict generalization of Tier A, not a new model.
#
# am is likewise fixed: sqrt(dp/drho) of the ACTUAL mixture EOS (was sqrt(a2l) everywhere, which fed
# Layer C the liquid speed inside the cavity and defeated its 1/a_m preconditioning where it matters most).
# eos=0 keeps the legacy raw-linear form for A/B. --------------------------------------------------------
@kernel function _k_tem_commit!(alpha, T, pexc, am, res, @Const(alpha_new), @Const(rho),
                                omega::Float32,
                                p_v::Float32, a2l::Float32, rho_l_sat::Float32, rho_v_sat::Float32,
                                a2v::Float32, psm::Float32, eos::Int32, wallis::Int32, ncg::Float32,
                                dk_cmin2::Float32, dk_a2v::Float32,
                                am_floor_frac::Float32, vapbound::Float32, vaploc::Float32,
                                Tmin::Float32, Tmax::Float32, tfree::Float32, yvf, emix::Float32,
                                tmix::Float32, dTm, dTm_new, lq::Float32, NI::Int32, NJ::Int32, g::Int32)
    i, j = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ
        ia = i+g; ja = j+g
        # under-relax the SEGREGATED alpha update (omega<1 damps the flow<->alpha Picard loop)
        a = omega >= 1.0f0 ? alpha_new[ia, ja] : (alpha[ia, ja] + omega*(alpha_new[ia, ja] - alpha[ia, ja]))
        # (res is now written by _k_tem_sweep!, where den and dtl are in scope -- see the block there.)
        alpha[ia, ja] = a
        # ★★★★ TIER 4 -- commit the thermodynamic temperature alongside alpha. tmix == 0 => not
        # taken => BITWISE LEGACY. Committed HERE and not written in place by the sweep because the
        # sweep is a Jacobi iteration: an in-place write would be a read-write race across threads.
        # ⚠ Deliberately NOT under-relaxed by `omega` and deliberately NOT clamped. omega damps the
        # alpha<->pressure Picard loop; dTm is not in that loop. And a clamp is the entire defect this
        # tier exists to remove -- re-imposing one here would reproduce it in a new field.
        if tmix > 0.0f0
            dTm[ia, ja] = dTm_new[ia, ja]
        end
        r = rho[ia, ja]
        rho_sat = rho_l_sat*(1.0f0-a) + rho_v_sat*a
        local p::Float32, dpdr::Float32
        if eos >= Int32(1)
            # Wallis harmonic frozen-alpha stiffness. With a2v = p_v/rho_v_sat this ALREADY reduces to the
            # ideal-gas vapor law at alpha=1 (p = p_v*r/rho_v_sat -> 0 as r -> 0), which is the restoring
            # pressure that bounds cavity density. A separate vapor-plateau branch is redundant and, under
            # a smooth-MAX, actively wrong (a2v*r swamps p_mix in the liquid). The smooth-max here is now
            # only a POSITIVITY GUARD: smooth max(pmx, 0). See cmesh_cav_tem3d.jl for the measured evidence.
            # BUGFIX 2026-07-19: this line hardcoded the mass-weighted harmonic and IGNORED the `wallis`
            # flag that sits in this kernel's own signature -- while EVERY other evaluation of the mixture
            # stiffness (sweep/Merkle source, _k_pexc_ghost!, tem_ambient/far-field) goes through
            # _a2m_mix. At wallis=1 the two forms differ by orders of magnitude inside the cavity, which
            # is the entire point of the density-weighted Wallis branch.
            # WHY IT WAS INVISIBLE: _k_pexc_ghost! runs immediately after this kernel over the FULL
            # ghosted array (interior included) and recomputes pexc/am through _a2m_mix, so those two
            # outputs were silently repaired. Only T survived -- and under Tier-C T is railed at Tmin, so
            # even the wrong branch produced Tmin. The total pressure was never wrong either
            # (p_total = rho*cs2*T + pexc is reconstructed by the ghost kernel), but the SPLIT was: T
            # feeds the upwind kinetic quadrature and the dt estimate, so this is a real dissipation
            # error wherever the pedestal is UNRAILED -- e.g. killgate leg B (wallis=1, legacy p_v),
            # whose ambient T ~ 0.999 sits in band.
            # _a2m_mix at wallis=0 returns 1/((1-a)/a2l + a/a2v) EXACTLY, so the legacy path is bitwise.
            # ★★★ EOS SITE 2/6 — vaploc>0 = the exact local-pressure solve; else VERBATIM legacy.
            local pmx::Float32, dpdr_raw::Float32
            if vaploc > 0.0f0
                pmx, dpdr_raw, _ = _pmix_vl(p_v, a, r, rho_sat, a2l, rho_l_sat, rho_v_sat, lq)
            else
            a2m = _a2m_mix(a, r, a2l, a2v, rho_l_sat, p_v, wallis, ncg, dk_cmin2, dk_a2v)
            # ★★ VAPOUR-SIDE BOUND (see _pmix_vb). The psm smooth-max below stays exactly where it
            # is and is applied AFTER the bound; dpdr picks up a2v wherever the bound is active.
            pmx, dpdr_raw, _ = _pmix_vb(p_v, a2m, r, rho_sat, a2v, vapbound)
            end
            sq  = sqrt(pmx*pmx + psm*psm)
            p    = 0.5f0*(pmx + sq)
            dpdr = 0.5f0*dpdr_raw*(1.0f0 + pmx/sq)
        else
            p    = p_v + a2l*(r - rho_sat)                  # legacy raw-linear (STIFF, kept for A/B)
            dpdr = a2l
        end
        # ★★★ S3g STEP 1. tfree = 0 reproduces the legacy pedestal EXACTLY (same expression, same
        # clamp, same order of operations) so the change is bitwise-off. tfree > 0 keeps the
        # temperature `_k_update!` marched from the conserved energy instead of overwriting it with a
        # pressure ratio; `pexc` below is then formed against that pedestal, so the TOTAL pressure p
        # is unchanged either way and only the SPLIT moves.
        Tn = tfree > 0.0f0 ? T[ia, ja] : p/(r*TEM_CS2)
        Tn = Tn < Tmin ? Tmin : (Tn > Tmax ? Tmax : Tn)   # in-band pedestal
        T[ia, ja] = Tn
        pexc[ia, ja] = p - r*TEM_CS2*Tn                    # PDP excess (breaks the cap when the flux injects it)
        # ★ S3g STEP 2: publish the vapour MASS fraction for the solver's conserved-energy closure.
        # Y = alpha*rho_v_sat/rho. This kernel is the only place that owns BOTH alpha and rho_v_sat,
        # so converting here keeps rho_v_sat out of CMeshSolverKA entirely -- one fewer quantity
        # duplicated across the module boundary. emix = 0 => never written, never read, bitwise legacy.
        if emix > 0.0f0
            yv = a*rho_v_sat/(r > 1.0f-12 ? r : 1.0f-12)
            yvf[ia, ja] = yv < 0.0f0 ? 0.0f0 : (yv > 1.0f0 ? 1.0f0 : yv)
        end
        # STAGE 1b: floor at the POSITIVITY scale, NOT a2v. Flooring at a2v (= p_v/rho_v_sat = 64.8)
        # forced am >= 8.05 over the WHOLE field => TEM mixture Mach <= 0.024, so the TEM path could not
        # produce a supersonic pocket or a condensation shock, and fed Layer C a sound speed 16-40x too
        # stiff. (The committed 'M up to 2.2' came from the HEM path, which floors at cmin2.)
        amf2 = am_floor_frac*a2l
        am[ia, ja] = sqrt(dpdr < amf2 ? amf2 : dpdr)          # mixture sound speed (Layer C preconditioning)
    end
end

# ---- STAGE 1a: pexc / am in the GHOST bands ---------------------------------------------------------
# THE BUG (code-verified by the Tier-C design judge): the commit kernels run ndrange (NI,NJ) = INTERIOR
# ONLY, and NO BC kernel anywhere writes `pexc`, which is allocated zeroed. So every ghost read in the
# flux — pexc[ia-1,ja] at i=1, pexc[ia,ja-1] across the wake cut, the far-field j-ghosts — returned 0.
# That is benign ONLY while interior liquid pexc ~ 0 (today). Under Tier-C the interior pexc becomes
# O(p_inf), so every domain boundary, the wall ghost band and the wake cut would see a spurious jump of
# 9-34x sigma*q and the run would blow up — and be misdiagnosed as "Tier-C is unstable".
#
# FIX: pexc is ALGEBRAIC in (rho, alpha, T), all of which the existing BC sweeps already fill. Evaluating
#   pexc = p_mix(rho,alpha) - rho*cs2*T      (using the ACTUAL ghost T, not T_band)
# over the FULL ghosted array guarantees the flux sees the correct p_mix in the ghosts whatever T the flow
# BC put there. Over the interior T == T_band, so this reproduces the commit's write exactly.
@kernel function _k_pexc_ghost!(pexc, am, @Const(alpha), @Const(rho), @Const(T),
                                p_v::Float32, a2l::Float32, rho_l_sat::Float32, rho_v_sat::Float32,
                                a2v::Float32, psm::Float32, eos::Int32, wallis::Int32, ncg::Float32,
                                dk_cmin2::Float32, dk_a2v::Float32,
                                am_floor_frac::Float32, vapbound::Float32, vaploc::Float32,
                                lq::Float32, checked_mixture::Int32, NX::Int32, NY::Int32)
    i, j = @index(Global, NTuple)
    @inbounds if i <= NX && j <= NY
        a = alpha[i,j]
        a = a < 0.0f0 ? 0.0f0 : (a > 1.0f0 ? 1.0f0 : a)
        r = rho[i,j]
        if checked_mixture == Int32(0)
            r = r < 1.0f-6 ? 1.0f-6 : r
        end
        rho_sat = rho_l_sat*(1.0f0-a) + rho_v_sat*a
        local p::Float32, dpdr::Float32
        if eos >= Int32(1)
            # ★★★ EOS SITE 3/6 — vaploc>0 = the exact local-pressure solve; else VERBATIM legacy.
            local pmx::Float32, dpdr_raw::Float32
            if vaploc > 0.0f0
                pmx, dpdr_raw, _ = _pmix_vl(p_v, a, r, rho_sat, a2l, rho_l_sat, rho_v_sat, lq)
            else
            a2m = _a2m_mix(a, r, a2l, a2v, rho_l_sat, p_v, wallis, ncg, dk_cmin2, dk_a2v)
            pmx, dpdr_raw, _ = _pmix_vb(p_v, a2m, r, rho_sat, a2v, vapbound)   # ★★ vapour-side bound
            end
            sq  = sqrt(pmx*pmx + psm*psm)
            p    = 0.5f0*(pmx + sq)
            dpdr = 0.5f0*dpdr_raw*(1.0f0 + pmx/sq)
        else
            p = p_v + a2l*(r - rho_sat); dpdr = a2l
        end
        pexc[i,j] = p - r*TEM_CS2*T[i,j]
        amf2 = am_floor_frac*a2l
        am[i,j] = sqrt(dpdr < amf2 ? amf2 : dpdr)
    end
end

@kernel function _k_tem_hist!(un, um, @Const(alpha), NI::Int32, NJ::Int32, g::Int32)
    i, j = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ
        um[i,j] = un[i,j]; un[i,j] = alpha[i+g, j+g]
    end
end

# ---- ★★★★ RAYLEIGH-PLESSET: commit Rdot, and fill its ghosts -----------------------------------
# Separate from `_k_tem_commit!` on purpose: that kernel also rebuilds T, pexc and am from the EOS,
# none of which Rdot touches, and enlarging it would risk the documented GPU-JIT @inline size
# threshold this codebase has already been bitten by once.
#
# UNDER-RELAXATION uses the SAME `omega` as alpha. The two fields are the two halves of one
# second-order system; relaxing them at different rates would change the effective damping of the
# oscillator, i.e. it would be a physics change disguised as a solver setting.
@kernel function _k_rp_commit!(rdot, @Const(rdot_new), omega::Float32,
                               NI::Int32, NJ::Int32, g::Int32)
    i, j = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ
        ia = i+g; ja = j+g
        rdot[ia, ja] = omega >= 1.0f0 ? rdot_new[ia, ja] :
                       (rdot[ia, ja] + omega*(rdot_new[ia, ja] - rdot[ia, ja]))
    end
end

# Ghost fill for Rdot: ZERO-GRADIENT on every boundary.
# ⚠ Why zero-gradient and not zero: Rdot is a material property of the bubble population being
# advected, not a wall-tied quantity. Forcing Rdot = 0 in the ghosts would inject an artificial
# "bubbles stop growing at the boundary" condition that the upwind stencil would then advect INWARD.
# Zero-gradient lets a growing/collapsing population leave the domain without reflecting.
#
# ⚠⚠ TWO LAUNCHES, NOT ONE, AND THE ORDER MATTERS. A single kernel doing both bands has a RACE at the
# corners: the i-band's source `rdot[g+1, j]` is itself a ghost for j <= g, which the j-band is
# concurrently writing. Splitting gives a device-wide sync between them, so the j-pass fills the
# corner rows from the INTERIOR first and the i-pass then propagates already-valid values outward.
# This is the same two-launch structure the alpha BCs (`_k_tem_bc!` / `_k_tem_bc_io!`) already use,
# and for the same reason.
@kernel function _k_rp_bc_j!(rdot, NI::Int32, NJ::Int32, g::Int32)   # PASS 1: wall and far-field bands
    i, j = @index(Global, NTuple)
    @inbounds if i <= NI + 2g && j <= g
        rdot[i, j]      = rdot[i, g+1]        # inner (wall) side
        rdot[i, NJ+g+j] = rdot[i, NJ+g]       # outer (far field) side
    end
end

@kernel function _k_rp_bc_outer!(rdot, @Const(outer_face_kind),
                                 NI::Int32, NJ::Int32, g::Int32)
    i, k = @index(Global, NTuple)
    @inbounds if i <= NI && k <= g
        face_kind = outer_face_kind[i]
        jd = NJ + g + k
        if face_kind == 0x01
            rdot[i+g, jd] = 0.0f0
        elseif face_kind >= 0x02
            rdot[i+g, jd] = rdot[i+g, NJ+g+Int32(1)-k]
        end
    end
end

@kernel function _k_rp_bc_i!(rdot, NI::Int32, NJ::Int32, g::Int32)   # PASS 2: inlet / outlet bands
    i, j = @index(Global, NTuple)
    @inbounds if j <= NJ + 2g && i <= g
        rdot[i, j]      = rdot[g+1, j]        # reads a j-ghost only where PASS 1 has already filled it
        rdot[NI+g+i, j] = rdot[NI+g, j]
    end
end

@kernel function _k_rp_hist!(rn, rm, @Const(rdot), NI::Int32, NJ::Int32, g::Int32)
    i, j = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ
        rm[i,j] = rn[i,j]; rn[i,j] = rdot[i+g, j+g]
    end
end

# ============================================================================================
# ★ P3 — FULL-GAMMA COUPLED (rho, alpha) DELTA-FORM SOLVE (2026-07-22).
# Spec: memory/project_docs/2026-07-22_P3_GAMMA_DESIGN_draft.md (READ ITS HEADER BANNER — the adversarial-review
# blocking fixes B1-B4 are implemented here, and the row-scaled system is review item N1, the raw-dpda
# source diagonal N2). One kernel per inner iteration, AFTER the 3 RK3 stages, replacing the
# segregated sweep+commit pair. Per cell:
#   row p (volume-scaled mixture continuity + Chorin (1/(rho b2)) dp/dtau, dp expanded via the EOS):
#       fixed point IDENTICAL to _k_update!'s continuity row (design 4.1, verified algebra).
#   row alpha (CONSERVATIVE div(alpha u) residual + Merkle source with the jac3 floor IN-MATRIX +
#       the Kunz (alpha/(rho b2)) dp/dtau enforcing entries as PURE LHS matrix terms + BDF2):
#       the enforcing terms are legal ONLY here (P2a lesson L1 — they cancel the alpha*div(u) piece
#       of the conservative residual; zero RHS contribution at any state).
# Delta-form discipline (design 1.5): the RHS is the plain nonlinear residual pair; EVERY floor /
# Jacobian / preconditioner multiplies a Delta, so the fixed point is a property of the residuals
# alone — the entire alpha-bias defect class (src_smooth, denominator-only, mode-1 hole) is
# structurally retired. omega is RETIRED under p3 (L2: the Picard loop it damped no longer exists).
#
# The review fixes, explicitly:
#  B1/B2 — the rho row is an EXPLICIT relaxation of the carrier-stiff KFVS mass residual, so its
#     effective step (b2/dpdr_G)*dtau_p must be capped at ~one carrier dt: A11 += lam_c/(r*vol)
#     (the _k_dtfield! spectral radius; a pure Delta-side term, consistency-free). This also bounds
#     the guarded-corner (fac_sm->0, dpdr on the amf2 floor) amplification that would otherwise NaN
#     the first smoke run. A |Delta_rho| <= 0.2*rho trust region backstops it.
#  B3 — alpha is written to alpha_new (the advection stencil reads neighbor alpha), copied back by
#     _k_p3_copy! BEFORE _k_pexc_ghost!; rho/T/pexc/am writes are own-cell-only reads, race-free.
#  B4 — inside the far-field sponge band (j > j0f*NJ when spg > 0) the coupled rho write is MASKED:
#     RK3 + sponge own that band exactly as in production (the production sponge-band fixed point is
#     [update+sponge]=identity, which R_p does not represent; the band is pure liquid, alpha ~ 0).
#  N1 — the 2x2 is solved in ROW-SCALED form (row p x r*b2*dtau_p, row alpha x dtau_p): all entries
#     O(1)-O(100) in Float32, and det is computed by the EXPANDED positive-sum identity
#     (no cancellation): det = dpdr_G*(1 + dtau*J_p3*(dpda_raw-dpda_sm) + dtau*(E+D) + kdt)
#                              + s_p*(bdf_p+aug)*B22  >= dpdr_G >= amf2 > 0.
#  N2 — the source-feedback diagonal uses the RAW dpda (>= the guarded dpda_sm): strictly more
#     damping, consistency-free in Delta-form, closer to the production-validated jac3.
@kernel function _k_p3_couple!(alpha_new, rho, T, pexc, am, res,
                               @Const(alpha), @Const(ux), @Const(uy),
                               @Const(dc_r), @Const(un_r), @Const(um_r),
                               @Const(un_a), @Const(um_a),
                               @Const(Six), @Const(Siy), @Const(Sjx), @Const(Sjy), @Const(vol),
                               rho_l_sat::Float32, rho_v_sat::Float32, kprod::Float32, kdest::Float32,
                               p_v::Float32, a2l::Float32, a2v::Float32, psm::Float32, wallis::Int32,
                               ncg::Float32, dk_cmin2::Float32, dk_a2v::Float32,
                               eos::Int32, am_floor_frac::Float32, vapbound::Float32, vaploc::Float32,
                               Tmin::Float32, Tmax::Float32,
                               beta2::Float32, cflp::Float32, dtp::Float32,
                               spg::Float32, j0f::Float32, diag::Int32,
                               lq::Float32, NI::Int32, NJ::Int32, g::Int32)
    i, j = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ
        ia = i+g; ja = j+g
        a0 = alpha[ia, ja]
        r  = rho[ia, ja]
        u0 = ux[ia, ja]; v0 = uy[ia, ja]
        # ---- EOS at the k-state: the GUARDED (committed) pressure + derivatives (design 1.1) ------
        rho_sat = rho_l_sat*(1.0f0-a0) + rho_v_sat*a0
        local p::Float32, dpdr_sm::Float32, dpda_sm::Float32, dpda_raw::Float32
        if eos >= Int32(1)
            # ★★★ EOS SITE 4/6 (p3 k-state) — vaploc>0 = the exact local-pressure solve; else the
            # VERBATIM legacy pair. The det identity below needs dpda_raw >= 0: under vaploc that is
            # dp/dalpha of the implicit root, non-negative over the whole physical map (see the
            # `_pmix_vl` banner and the verification note in this kernel's dpda comment).
            local pmx::Float32, dpdr_raw::Float32
            if vaploc > 0.0f0
                pmx, dpdr_raw, dpda_raw = _pmix_vl(p_v, a0, r, rho_sat, a2l, rho_l_sat, rho_v_sat, lq)
            else
            a2m = _a2m_mix(a0, r, a2l, a2v, rho_l_sat, p_v, wallis, ncg, dk_cmin2, dk_a2v)
            # ★★ VAPOUR-SIDE BOUND (see _pmix_vb). Where it is active dp/drho = a2v and dp/dalpha = 0,
            # so BOTH dpda_raw and dpda_sm (= fac_sm*dpda_raw) are scaled by the returned factor.
            # The det identity below is UNCHANGED and still a sum of non-negative terms:
            #   dpda_raw - dpda_sm = dpda_raw*(1 - fac_sm) >= 0  (= 0 when the bound is active),
            #   B22 = dtau*Jp3*dpda_raw + 1 + dtau*(E+D) + kdt >= 1 > 0,
            #   dpdr_G >= amf2 > 0  (dpdr_raw = a2v > 0 on the bounded branch),
            # hence det >= dpdr_G >= amf2 > 0 exactly as before.
            pmx, dpdr_raw, dpda_scale = _pmix_vb(p_v, a2m, r, rho_sat, a2v, vapbound)
            dpda_raw = dpda_scale*a2m*(rho_l_sat - rho_v_sat)
            end
            sq  = sqrt(pmx*pmx + psm*psm)
            p        = 0.5f0*(pmx + sq)
            fac_sm   = 0.5f0*(1.0f0 + pmx/sq)
            dpdr_sm  = fac_sm*dpdr_raw
            dpda_sm  = fac_sm*dpda_raw
        else
            p = p_v + a2l*(r - rho_sat); dpdr_sm = a2l
            dpda_raw = a2l*(rho_l_sat - rho_v_sat); dpda_sm = dpda_raw
        end
        amf2 = am_floor_frac*a2l
        dpdr_G = dpdr_sm < amf2 ? amf2 : dpdr_sm       # matrix floor (guarded corner, design 1.1/R4)
        # ---- Merkle source at the k-state (hard switch, L4) + jac3 floor (TEM:387-390 verbatim) ---
        dpv = p_v - p
        E  = dpv > 0.0f0 ? kprod*dpv    : 0.0f0
        D  = dpv < 0.0f0 ? kdest*(-dpv) : 0.0f0
        wE = dpv > 0.0f0 ? 1.0f0 : 0.0f0
        wD = dpv < 0.0f0 ? 1.0f0 : 0.0f0
        S_a = E*(1.0f0-a0) - D*a0
        Jp  = kprod*wE*(1.0f0-a0) + kdest*wD*a0
        wband = 1.0f0 - min(abs(dpv)/(0.10f0*p_v), 1.0f0)
        Jp3 = max(Jp, wband*max(kprod*(1.0f0-a0), kdest*a0))
        # ---- conservative advection residual (the cons=1 stencil verbatim, TEM:331-344) -----------
        Ulo = 0.5f0*((ux[ia-1,ja]+u0)*Six[i,j]   + (uy[ia-1,ja]+v0)*Siy[i,j])
        Uhi = 0.5f0*((u0+ux[ia+1,ja])*Six[i+1,j] + (v0+uy[ia+1,ja])*Siy[i+1,j])
        Vlo = 0.5f0*((ux[ia,ja-1]+u0)*Sjx[i,j]   + (uy[ia,ja-1]+v0)*Sjy[i,j])
        Vhi = 0.5f0*((u0+ux[ia,ja+1])*Sjx[i,j+1] + (v0+uy[ia,ja+1])*Sjy[i,j+1])
        alo = Ulo >= 0.0f0 ? alpha[ia-1,ja] : a0
        ahi = Uhi >= 0.0f0 ? a0 : alpha[ia+1,ja]
        blo = Vlo >= 0.0f0 ? alpha[ia,ja-1] : a0
        bhi = Vhi >= 0.0f0 ? a0 : alpha[ia,ja+1]
        adv = (Uhi*ahi - Ulo*alo + Vhi*bhi - Vlo*blo)/vol[i,j]
        # ★ MATCHED SOURCE RECAST (2026-07-22, the P3 round-1 root-cause fix — see the cons=2 block
        # in _k_tem_sweep! for the full rationale): subtract a0*div(u) with the same face fluxes so
        # the operator is flux-form u.grad(alpha) and the fixed point is the VALIDATED production
        # PDE. Explicit treatment: div(u) is advection-class (O(U/L)), not rate-class stiff.
        adv -= a0*(Uhi - Ulo + Vhi - Vlo)/vol[i,j]
        # ---- preconditioned dtau_p (Kunz Eq. 26; C' = sqrt(U^2 + b2*S^2), design 3.2) -------------
        b2 = max(beta2, u0*u0 + v0*v0)
        UiW = abs(u0*Six[i,j]   + v0*Siy[i,j]);   UiE = abs(u0*Six[i+1,j] + v0*Siy[i+1,j])
        UjS = abs(u0*Sjx[i,j]   + v0*Sjy[i,j]);   UjN = abs(u0*Sjx[i,j+1] + v0*Sjy[i,j+1])
        SiE = sqrt(Six[i+1,j]^2 + Siy[i+1,j]^2); SiW = sqrt(Six[i,j]^2 + Siy[i,j]^2)
        SjN = sqrt(Sjx[i,j+1]^2 + Sjy[i,j+1]^2); SjS = sqrt(Sjx[i,j]^2 + Sjy[i,j]^2)
        Ui = 0.5f0*(UiW + UiE); Uj = 0.5f0*(UjS + UjN)
        Sbi = 0.5f0*(SiE + SiW); Sbj = 0.5f0*(SjN + SjS)
        Ci = sqrt(Ui*Ui + b2*Sbi*Sbi); Cj = sqrt(Uj*Uj + b2*Sbj*Sbj)
        dtau = cflp*vol[i,j]/max(Ui + Ci + Uj + Cj, 1.0f-30)
        # ---- B1/B2: carrier-clock diagonal augmentation (the _k_dtfield! spectral radius) ---------
        c = 3.0f0*sqrt(max(T[ia,ja], 1.0f-3))
        aug = (Ui + Uj + c*(Sbi + Sbj))/(r*vol[i,j])
        # ---- residuals (plain nonlinear; NO derivative — that is what makes the matrix free) ------
        local bdf_p::Float32, Rp::Float32, Ra::Float32, kdt::Float32
        if dtp > 0.0f0
            bdf_p = 1.5f0/(r*dtp)
            Rp = (dc_r[i,j]/vol[i,j] + (3.0f0*r - 4.0f0*un_r[i,j] + um_r[i,j])/(2.0f0*dtp))/r
            Ra = adv - S_a + (3.0f0*a0 - 4.0f0*un_a[i,j] + um_a[i,j])/(2.0f0*dtp)
            kdt = 1.5f0*dtau/dtp
        else
            bdf_p = 0.0f0
            Rp = (dc_r[i,j]/vol[i,j])/r
            Ra = adv - S_a
            kdt = 0.0f0
        end
        # ---- the ROW-SCALED 2x2 (N1): row p x (r*b2*dtau), row alpha x dtau -----------------------
        s_p = r*b2*dtau
        B11 = dpdr_G + s_p*(bdf_p + aug)
        B12 = dpda_sm
        # ★ ROUND 3 (2026-07-22): SOURCE-COUPLED variant — the Kunz enforcing weight w = a0/(r*b2)
        # is REFUTED for our architecture (probe ladder in the design doc §ROUND 2: our MASS-residual
        # row-p carries the physical BDF2 rho-evolution every inner loop, never small in
        # time-accurate marching, and B21*w*Rp integrates it into runaway cavity growth — Kunz's
        # VOLUME-continuity residual, for which the term was constructed, is small near
        # pseudo-convergence). w := 0 keeps the genuine physics coupling: B12 = dpda_sm (alpha moves
        # p in the rho row) and B21 = dpdr_G*dtau*Jp3 (rho moves the source through p — the build
        # plan's candidate (b), kink-band-local via Jp3 rather than cavity-wide via alpha/(r*b2)).
        w   = 0.0f0
        B21 = dpdr_G*(w + dtau*Jp3)
        B22 = dpda_sm*w + dtau*Jp3*dpda_raw + 1.0f0 + dtau*(E + D) + kdt
        # det via the EXPANDED positive-sum identity (no cancellation; N1):
        det = dpdr_G*(1.0f0 + dtau*Jp3*(dpda_raw - dpda_sm) + dtau*(E + D) + kdt) +
              s_p*(bdf_p + aug)*B22
        Rp_s = s_p*Rp; Ra_s = dtau*Ra
        drho = (-B22*Rp_s + B12*Ra_s)/det
        dalp = ( B21*Rp_s - B11*Ra_s)/det
        # ---- trust region (B2 backstop) + floors/clamps + sponge-band mask (B4) -------------------
        tr = 0.2f0*r
        drho = drho > tr ? tr : (drho < -tr ? -tr : drho)
        r2 = r + drho
        r2 = r2 < 1.0f-4 ? 1.0f-4 : r2                  # the production floor (SKA:693)
        in_sponge = (spg > 0.0f0) && (Float32(j) > j0f*Float32(NJ))
        # ★ ROUND 4 (2026-07-22): the rho-OVERWRITE IS OFF in the P3 formulation — the coupled 2x2
        # CONDITIONS THE ALPHA/SOURCE UPDATE ONLY. Discriminator evidence (design doc §ROUND 3):
        # with the overwrite, TA marches run away (V→15.5, far-field contamination); dalpha-only is
        # BOUNDED (V 5-7e-2 over 45 CTU, 0 NaN). The mass residual still conditions Delta_alpha
        # through B21 = dpdr_G*dtau*Jp3 (kink-band-local), the source is matrix-implicit, and the
        # in-pass commit kills the segregation lag — the parts the sigma=0.75 excursions need.
        # diag=2 re-enables the full-Gamma rho write (EXPERIMENTAL, refuted as of tonight).
        rw = (in_sponge || diag != Int32(2)) ? r : r2
        an = a0 + dalp
        an = an < 0.0f0 ? 0.0f0 : (an > 1.0f0 ? 1.0f0 : an)   # the production clamp (TEM:421)
        # ★★★ SAME RESIDUAL AS THE SEGREGATED PATH (2026-07-31). This wrote |an - a0|, the raw update
        # increment, while `_k_tem_sweep!` now writes the PDE residual. Two paths reporting different
        # quantities into the same array cannot be compared, and comparing them is exactly the open
        # question (does the preconditioned coupled clock contract where the acoustic segregated one
        # does not?). `Ra` is ALREADY FORMED above as the plain nonlinear void residual
        #     Ra = adv - S_a + (3*a0 - 4*un_a + um_a)/(2*dtp)
        # and the segregated measure is E(1-a0) - D*a0 - adv - (3a0 - 4un + um)/(2dtp) = -Ra, so the
        # two agree IDENTICALLY in absolute value. `res` is diagnostic-only -- its only readers are
        # `tem_residual` and `tem_residual_report` -- so this changes no solution, on either path.
        res[i, j] = abs(Ra)
        alpha_new[ia, ja] = an
        rho[ia, ja] = rw
        # ---- in-pass commit: T pedestal, pexc, am from the NEW state (TEM:484-502 logic) ----------
        # This is the segregation-lag kill: (rho, alpha, T, pexc, am) leave every inner iteration
        # mutually consistent at the same state.
        rho_sat2 = rho_l_sat*(1.0f0-an) + rho_v_sat*an
        local p2::Float32, dpdr2::Float32
        if eos >= Int32(1)
            # ★★★ EOS SITE 5/6 (p3 in-pass commit) — vaploc>0 = the exact local-pressure solve;
            # else VERBATIM legacy. Character-identical to sites 2/3 by construction.
            local pmx2::Float32, dpdr_raw2::Float32
            if vaploc > 0.0f0
                pmx2, dpdr_raw2, _ = _pmix_vl(p_v, an, rw, rho_sat2, a2l, rho_l_sat, rho_v_sat, lq)
            else
            a2m2 = _a2m_mix(an, rw, a2l, a2v, rho_l_sat, p_v, wallis, ncg, dk_cmin2, dk_a2v)
            pmx2, dpdr_raw2, _ = _pmix_vb(p_v, a2m2, rw, rho_sat2, a2v, vapbound)   # ★★ vapour-side bound
            end
            sq2  = sqrt(pmx2*pmx2 + psm*psm)
            p2    = 0.5f0*(pmx2 + sq2)
            dpdr2 = 0.5f0*dpdr_raw2*(1.0f0 + pmx2/sq2)
        else
            p2 = p_v + a2l*(rw - rho_sat2); dpdr2 = a2l
        end
        Tn = p2/(rw*TEM_CS2)
        Tn = Tn < Tmin ? Tmin : (Tn > Tmax ? Tmax : Tn)
        T[ia, ja] = Tn
        pexc[ia, ja] = p2 - rw*TEM_CS2*Tn
        am[ia, ja] = sqrt(dpdr2 < amf2 ? amf2 : dpdr2)
    end
end

# B3: commit alpha_new -> alpha (interior) before _k_pexc_ghost!, mirroring the production
# sweep/commit two-buffer discipline (the couple kernel's advection stencil reads neighbor alpha).
@kernel function _k_p3_copy!(alpha, @Const(alpha_new), NI::Int32, NJ::Int32, g::Int32)
    i, j = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ
        alpha[i+g, j+g] = alpha_new[i+g, j+g]
    end
end

"""Project the current barotropic composition onto the existing TEM pressure EOS.

Interior alpha is copied through distinct alpha_new scratch with unit relaxation.
Refreshes T/pexc/am and composition ghosts, including the configured pressure outlet;
that outlet may also overwrite flow ghost states. Does not read timestep/source rates,
advance transport, or update residuals, clocks, BDF2/bubble histories or thermal fields.
alpha_new is overwritten scratch. The caller must supply admissible current states.
Thermal/energy and coupled-history modes are outside this projection contract.
`defer_pressure_outlet=true` lets the conservative caller immediately supply its
direct physical-pressure boundary update without a lossy pedestal round trip.
`checked_mixture=true` evaluates the caller's admitted positive density directly,
including below 1e-6; it does not substitute the legacy ghost-projection density.
"""
function tem_project!(s, tem::TEMState;defer_pressure_outlet::Bool=false,checked_mixture::Bool=false)
    all(iszero, (s.emix, tem.tfree, tem.tmix, tem.tvp, tem.p3, tem.rp, tem.slip)) ||
        throw(ArgumentError("tem_project! requires barotropic energy and no thermal/coupled-history modes"))
    shape = (s.NI + 2s.g, s.NJ + 2s.g)
    size(tem.alpha) == shape && size(tem.alpha_new) == shape &&
        typeof(tem.alpha_new) == typeof(tem.alpha) &&
        KA.get_backend(tem.alpha) == s.backend && KA.get_backend(tem.alpha_new) == s.backend ||
        throw(ArgumentError("tem_project! requires matching alpha scratch shape and backend"))
    all(a -> size(a) == size(tem.alpha) && KA.get_backend(a) == s.backend,
        (s.rho, s.T, s.pexc, s.am)) ||
        throw(ArgumentError("tem_project! requires matching flow and composition arrays"))
    all(a -> !Base.mightalias(tem.alpha_new, a), (tem.alpha, s.rho, s.T, s.pexc, s.am)) ||
        throw(ArgumentError("tem_project! requires alpha scratch disjoint from projected state"))
    pressure_outlet = hasproperty(s, :outlet_mode) && s.outlet_mode != Int32(0)
    if pressure_outlet
        CMeshPressureOutlet.check_pressure_outlet_tem(s, tem)
    end
    b = s.backend; NI, NJ, g = Int32(s.NI), Int32(s.NJ), Int32(s.g)
    # The commit kernel reads alpha_new as @Const while writing alpha.
    copyto!(tem.alpha_new, tem.alpha)
    _k_tem_commit!(b, WG)(tem.alpha, s.T, s.pexc, s.am, tem.res, tem.alpha_new, s.rho, 1.0f0,
                          tem.p_v, tem.a2l, tem.rho_l_sat, tem.rho_v_sat,
                          tem.a2v, tem.psm, tem.eos, tem.wallis, tem.ncg,
                          tem.dk_cmin2, tem.dk_a2v, tem.am_floor_frac, tem.vapbound, tem.vaploc,
                          tem.Tmin, tem.Tmax, tem.tfree, s.yvf, s.emix, tem.tmix, tem.dTm, tem.dTm_new,
                          tem.lq, NI, NJ, g; ndrange=(s.NI, s.NJ))
    _k_tem_bc!(b, WG)(tem.alpha, NI, Int32(s.NIW), Int32(s.NIF), NJ, g; ndrange=(s.NI, s.g))
    if hasproperty(s, :outer_mode) && s.outer_mode != Int32(0)
        _k_tem_bc_outer!(b, WG)(tem.alpha, s.outer_face_kind, NI, NJ, g; ndrange=(s.NI, s.g))
    end
    _k_tem_bc_io!(b, WG)(tem.alpha, NI, NJ, g; ndrange=(s.NJ+2s.g, s.g))
    NX = Int32(s.NI + 2s.g); NY = Int32(s.NJ + 2s.g)
    _k_pexc_ghost!(b, WG)(s.pexc, s.am, tem.alpha, s.rho, s.T,
                          tem.p_v, tem.a2l, tem.rho_l_sat, tem.rho_v_sat,
                          tem.a2v, tem.psm, tem.eos, tem.wallis, tem.ncg,
                          tem.dk_cmin2, tem.dk_a2v, tem.am_floor_frac, tem.vapbound, tem.vaploc,
                          tem.lq, Int32(checked_mixture), NX, NY; ndrange=(s.NI+2s.g, s.NJ+2s.g))
    if pressure_outlet && !defer_pressure_outlet
        CMeshPressureOutlet.pressure_outlet_bc!(s)
    end
    nothing
end

"""One post-step TEM sub-iteration: advect + phase-change alpha, then impose the mixture pressure on T.
Call every inner (pseudo-time) iteration, AFTER step_ka! (mirrors hem_apply!). `dtp>0` engages the
dual-time BDF2 physical-time term (time-accurate); `dtp=0` is steady LTS relaxation."""
function tem_step!(s, tem::TEMState; dtp::Float32=0.0f0)
    # Strip topology (5 Oct 2026): the legacy TEM stepping is not defined on a strip carrier (outer_mode 2); refused
    # before any write, by name (memory/project_docs/2026-10-05_STRIP_TOPOLOGY_PLAN.md section 3.3, the legacy path).
    hasproperty(s, :outer_mode) && s.outer_mode == Int32(2) && throw(ArgumentError(
        "tem_step! (the legacy TEM stepping) does not support the strip carrier (outer_mode 2, CMeshSpeciesStripTopologyV1)"))
    if hasproperty(s, :outlet_mode) && s.outlet_mode != Int32(0)
        CMeshPressureOutlet.check_pressure_outlet_tem(s, tem)
    end
    b = s.backend; NI, NJ, g = Int32(s.NI), Int32(s.NJ), Int32(s.g)
    _k_tem_bc!(b, WG)(tem.alpha, NI, Int32(s.NIW), Int32(s.NIF), NJ, g; ndrange=(s.NI, s.g))
    if hasproperty(s, :outer_mode) && s.outer_mode != Int32(0)
        _k_tem_bc_outer!(b, WG)(tem.alpha, s.outer_face_kind, NI, NJ, g; ndrange=(s.NI, s.g))
    end
    _k_tem_bc_io!(b, WG)(tem.alpha, NI, NJ, g; ndrange=(s.NJ+2s.g, s.g))
    if hasproperty(s, :outlet_mode) && s.outlet_mode != Int32(0)
        CMeshPressureOutlet.pressure_outlet_bc!(s)
    end
    if tem.p3 >= Int32(1) && dtp > 0.0f0
        # ★ P3 coupled path — TIME-ACCURATE PHASE ONLY (the dtp>0 regime gate, 2026-07-22 round 2).
        # Kunz's Gamma system is a pseudo-steady-within-physical-step construction: each inner loop
        # starts CLOSE to converged (R_p small). The from-scratch 40000-iteration warm-up is far
        # outside that regime, and the 5-probe attribution trail (design doc §GATE RESULTS) showed
        # the R_p->alpha coupling channel (B21·R_p) integrates the persistently-nonzero development
        # mass residual into cavity saturation (amax=1.000) even with the MATCHED source — while
        # the identical matched alpha-equation on the segregated path develops the production
        # attractor exactly (cons=2/p3=0: amax=0.644). Warm-up (dtp=0) therefore runs the PRODUCTION
        # path below — bitwise-identical development — and the whole Gamma system engages whole for
        # the time-accurate march, which is its regime AND the G2 target. Not a partial adoption:
        # a regime gate on the whole formulation.
        # BC fills above are still required (the conservative stencil reads alpha ghosts);
        # _k_pexc_ghost! below runs unchanged and must see the committed alpha (the copy precedes it).
        _k_p3_couple!(b, WG)(tem.alpha_new, s.rho, s.T, s.pexc, s.am, tem.res,
                             tem.alpha, s.ux, s.uy, s.dc_r, s.un_r, s.um_r, tem.un, tem.um,
                             s.Six, s.Siy, s.Sjx, s.Sjy, s.vol,
                             tem.rho_l_sat, tem.rho_v_sat, tem.kprod, tem.kdest,
                             tem.p_v, tem.a2l, tem.a2v, tem.psm, tem.wallis,
                             tem.ncg, tem.dk_cmin2, tem.dk_a2v,
                             tem.eos, tem.am_floor_frac, tem.vapbound, tem.vaploc, tem.Tmin, tem.Tmax,
                             tem.beta2, tem.cflp, dtp,
                             Float32(s.sponge_sigma), Float32(s.sponge_j0f), P3_DIAG,
                             tem.lq, NI, NJ, g; ndrange=(s.NI, s.NJ))
        _k_p3_copy!(b, WG)(tem.alpha, tem.alpha_new, NI, NJ, g; ndrange=(s.NI, s.NJ))
    else
    _k_tem_sweep!(b, WG)(tem.alpha_new, tem.res, tem.alpha, s.rho, s.ux, s.uy,
                         s.Six, s.Siy, s.Sjx, s.Sjy, s.vol, s.dtf, tem.un, tem.um, s.rk_r,
                         tem.rho_l_sat, tem.rho_v_sat, tem.kprod, tem.kdest,
                         tem.p_v, tem.a2l, tem.a2v, tem.psm, tem.wallis, tem.ncg,
                         tem.dk_cmin2, tem.dk_a2v,
                         tem.src_smooth, tem.src_jac,
                         tem.enf, tem.beta2, tem.cons, tem.arecon, tem.dtboost, tem.vapbound, tem.vaploc,
                         tem.dtb_dt, tem.advimp, tem.gs, dtp,
                         tem.rp, tem.rdot_new, tem.rdot, tem.rdot_n, tem.rdot_m,
                         tem.rp_n0, tem.rp_R0, tem.rp_S, tem.rp_pg0, tem.rp_kappa, tem.rp_nu,
                         tem.tvp, tem.tvp_LRv, tem.tvp_Tref, tem.slip, tem.slip_tau, tem.cloudint,
                         tem.poly, tem.poly_e, tem.poly_Rmin, tem.poly_Rmax,
                         tem.hk, tem.wallnuc, tem.wall_decay, tem.btke, s.btke_src, tem.pvturb, s.tk,
                         tem.pvvort, s.pvort, s.T, s.pexc,
                         # ★ gs=1 launches ONE THREAD PER j-LINE (the i-march is inside the kernel);
                         # gs=0 keeps the per-cell ndrange => the loop runs once => bitwise legacy.
                         tem.tmix, tem.tmix_lat, tem.dTm, tem.dTm_new, tem.lq,
                         hasproperty(s, :outlet_mode) ? s.outlet_mode : Int32(0),
                         NI, NJ, g; ndrange = (tem.gs == Int32(1) ? 1 : s.NI, s.NJ))
    # ★★★★ Rayleigh-Plesset: commit Rdot and refill its ghosts, in the same inner iteration as alpha.
    # Ordered AFTER the sweep and BEFORE `_k_tem_commit!` so the pressure the next sweep sees is built
    # from an alpha that is consistent with the Rdot just committed.
    if tem.rp > 0.0f0
        _k_rp_commit!(b, WG)(tem.rdot, tem.rdot_new, tem.omega, NI, NJ, g; ndrange=(s.NI, s.NJ))
        _k_rp_bc_j!(b, WG)(tem.rdot, NI, NJ, g; ndrange=(s.NI+2s.g, s.g))   # PASS 1 (see the kernels)
        if hasproperty(s, :outer_mode) && s.outer_mode != Int32(0)
            _k_rp_bc_outer!(b, WG)(tem.rdot, s.outer_face_kind, NI, NJ, g; ndrange=(s.NI, s.g))
        end
        _k_rp_bc_i!(b, WG)(tem.rdot, NI, NJ, g; ndrange=(s.g, s.NJ+2s.g))   # PASS 2, needs PASS 1 done
    end
    _k_tem_commit!(b, WG)(tem.alpha, s.T, s.pexc, s.am, tem.res, tem.alpha_new, s.rho, tem.omega,
                          tem.p_v, tem.a2l, tem.rho_l_sat, tem.rho_v_sat,
                          tem.a2v, tem.psm, tem.eos, tem.wallis, tem.ncg,
                          tem.dk_cmin2, tem.dk_a2v, tem.am_floor_frac, tem.vapbound, tem.vaploc,
                          tem.Tmin, tem.Tmax, tem.tfree, s.yvf, s.emix, tem.tmix, tem.dTm, tem.dTm_new, tem.lq, NI, NJ, g; ndrange=(s.NI, s.NJ))
    end
    # STAGE 1a: fill pexc/am over the FULL ghosted array (interior write is identical; ghosts were 0)
    if hasproperty(s, :outlet_mode) && s.outlet_mode != Int32(0)
        # Alpha has just committed. Refresh its boundary sources before the outlet
        # inverts the EOS; otherwise its ghost density refers to the previous alpha.
        _k_tem_bc!(b, WG)(tem.alpha, NI, Int32(s.NIW), Int32(s.NIF), NJ, g; ndrange=(s.NI, s.g))
        _k_tem_bc_outer!(b, WG)(tem.alpha, s.outer_face_kind, NI, NJ, g; ndrange=(s.NI, s.g))
        _k_tem_bc_io!(b, WG)(tem.alpha, NI, NJ, g; ndrange=(s.NJ+2s.g, s.g))
    end
    NX = Int32(s.NI + 2s.g); NY = Int32(s.NJ + 2s.g)
    _k_pexc_ghost!(b, WG)(s.pexc, s.am, tem.alpha, s.rho, s.T,
                          tem.p_v, tem.a2l, tem.rho_l_sat, tem.rho_v_sat,
                          tem.a2v, tem.psm, tem.eos, tem.wallis, tem.ncg,
                          tem.dk_cmin2, tem.dk_a2v, tem.am_floor_frac, tem.vapbound, tem.vaploc,
                          tem.lq, Int32(0), NX, NY; ndrange=(s.NI+2s.g, s.NJ+2s.g))
    if hasproperty(s, :outlet_mode) && s.outlet_mode != Int32(0)
        CMeshPressureOutlet.pressure_outlet_bc!(s)
    end
    nothing
end

"""Rotate the dual-time BDF2 history (um <- un, un <- current alpha). Call ONCE per physical step,
after the inner loop converges (mirrors hist_cycle_ka!)."""
function tem_hist_cycle!(tem::TEMState, s)
    b = s.backend; NI, NJ, g = Int32(s.NI), Int32(s.NJ), Int32(s.g)
    _k_tem_hist!(b, WG)(tem.un, tem.um, tem.alpha, NI, NJ, g; ndrange=(s.NI, s.NJ))
    # ★★★★ Rdot carries its own BDF2 history. Without this the second-order system would be marched
    # with a first-order-accurate Rdot while alpha got BDF2 -- a silent order reduction in exactly the
    # variable that carries the inertia, i.e. it would quietly damp the oscillator the model exists for.
    tem.rp > 0.0f0 &&
        _k_rp_hist!(b, WG)(tem.rdot_n, tem.rdot_m, tem.rdot, NI, NJ, g; ndrange=(s.NI, s.NJ))
    nothing
end

"""Seed the BDF2 history from the current (developed) alpha so the dual-time phase starts consistent."""
tem_seed!(tem::TEMState, s) = (tem_hist_cycle!(tem, s); tem_hist_cycle!(tem, s))

end # module
