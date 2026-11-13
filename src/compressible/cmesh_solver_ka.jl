# ---------------------------------------------------------------------------------------------
# PATH STATUS: **MIXED.** Production uses this module's carrier machinery (_mixture_pressure_bc!,
# loads_ka, GLOBAL_DT). The legacy marching entry points in this same file -- step_ka! and
# sst_step_ka! (the latter carrying the LEGACY Reboud correction; the production path carries its own,
# opt-in and default off, in cmesh_species_sst.jl) -- are NOT called
# when TOPO_SPECIES=1. Presence in this file does not mean presence in production.
# Map and dispatch: memory/project_docs/CAVITATION_CODE_MAP.md (checked by tools/check_code_map.jl).
# ---------------------------------------------------------------------------------------------
# ============================================================================================
# CMESH2D KernelAbstractions solver (overnight GPU port, 2026-07-14→15) — the backend-agnostic
# sibling of cmesh_solver.jl: SAME physics (shared cmesh_flux_core.jl), every loop an @kernel,
# runs on CPU() or CUDABackend() unchanged. The plain module stays untouched as the A/B truth.
# Sync discipline (the PV_NOSYNC lesson): kernels on one backend/stream execute in-order — NO
# per-kernel synchronize; sync only before host reads (loads pulls, residual mapreduce is
# self-syncing). Wall mode: Int32 (0 transparent, 1 slip, 2 noslip) — kernels take no Symbols.
# ============================================================================================

module CMeshSolverKA

using KernelAbstractions
const KA = KernelAbstractions
# ★★★ GLOBAL (NON-LTS) TIME STEP (2026-07-31). 1 = collapse dtf to its global minimum every step, which
# makes the explicit march TIME-ACCURATE BY CONSTRUCTION -- one clock for every cell, no inner loop, no
# BDF2, no convergence question. Brute force (the step is set by the stiffest cell) and the natural
# ground truth now that today's measurements closed the dual-time relaxation route: in EVERY reachable
# configuration the oscillation stayed clocked by inner-iteration count, even with the BDF2
# physical-time term raised from 7.8% to co-equal weight. 0 = bitwise legacy (LTS, as run all campaign).
const GLOBAL_DT = Ref(parse(Int, get(ENV, "TOPO_GLOBALDT", "0")))   # Ref, not const value: the warm-up wants LTS (fast relaxation to a developed cavity) and only the SCORED MARCH wants the global clock, so the driver flips it between phases.

export KSol, build_sol_ka, set_freestream_ka!, set_air_sutherland_ka!, set_air_constant_dynamic_ka!, set_sst_ka!, set_keps_ka!, set_farfield_mixture!, step_ka!, sst_step_ka!, loads_ka, refresh_raw_residual_ka!, residual_linf_ka, residual_scaled_ka, residual_report_ka, hist_cycle_ka!, rstar_linf_ka, rstar_report_ka, sfd_seed_ka!, acoustic_filter_ka!, outer_bc_override_ka!, outer_flux_override_ka!, seed_channel_reference_ka!, clear_channel_reference_ka!

include(joinpath(@__DIR__, "..", "cmesh", "cmesh_flux_core.jl"))
include(joinpath(@__DIR__, "..", "cmesh", "cmesh_wallfun.jl"))   # S3i: equilibrium log-law wall function
include(joinpath(@__DIR__, "..", "cmesh", "cmesh_eos_mix.jl"))
include(joinpath(@__DIR__, "..", "cmesh", "cmesh_pressure_outlet.jl"))
using .CMeshPressureOutlet: set_pressure_outlet!, clear_pressure_outlet!, pressure_outlet_bc!, pressure_outlet_flux!
export set_pressure_outlet!, clear_pressure_outlet!

mutable struct KSol{A2, V, M, S2, R<:AbstractFloat}
    NI::Int; NJ::Int; NIW::Int; NIF::Int; g::Int
    Six::A2; Siy::A2; Sjx::A2; Sjy::A2; vol::A2; xc::A2; yc::A2
    rho::A2; ux::A2; uy::A2; T::A2
    rk_r::A2; rk_mx::A2; rk_my::A2; rk_E::A2
    Fi_r::A2; Fi_mx::A2; Fi_my::A2; Fi_e::A2
    Fj_r::A2; Fj_mx::A2; Fj_my::A2; Fj_e::A2
    dc_r::A2; dc_mx::A2; dc_my::A2; dc_e::A2
    dtf::A2
    cxa::V; cya::V; cza::V; wa::V
    drot::R; eint::R
    rho_inf::R; uxi::R; uyi::R; T_inf::R
    gam::R; xv::R; yv::R
    m2::R; beta::R; ca::R; sa::R
    wall_flux::Bool; nu_lam::R
    tk::A2; tom::A2; nut::A2
    tnl::A2                                        # LAGGED eddy viscosity (Olsen-Coakley; lag==1)
    sst::Bool; k_inf::R; w_inf::R
    lag::Int32                                     # 1 = nu_t lag equation drives the stresses
    lag_a0::R                                # lag rate coefficient (Olsen-Coakley 0.35)
    # ★ FBM (Johansen/Wu/Shyy 2004) filter-based hybrid RANS-LES coefficient C_Delta. <=0 disables it
    # (bitwise legacy). Published value is 1.0. See _fbm_factor for the rationale and the warning against
    # tuning it. Added 2026-07-27 after grid convergence showed URANS shedding does not survive refinement.
    fbm_cdelta::R
    # ★ PANS (Girimaji 2006) with a PRESCRIBED CONSTANT f_k = k_u/k. <=0 disables it (bitwise legacy).
    # Published bracket for cavitating hydrofoils (Ji/Luo/Huang/Wang): f_k = 0.2-0.5. See _pans_factor
    # for the derivation, the REDUCED-FORM caveat that must travel with any result, and the warning
    # against sweeping f_k. Added 2026-07-27 after FBM was measured INERT inside the cavity.
    pans_fk::R
    earsm::Int32                                   # 1 = WJ-EARSM: effective nu_t + extra anisotropy stress
    exx::A2; exy::A2; eyy::A2                      # EARSM extra stress tau_extra (interior-sized)
    wdist::A2; gammat::A2
    recon::Int32
    fullvisc::Int32                                # 0 = thin-layer (legacy bitwise), 1 = full stress tensor (A2c)
    sponge_sigma::R                          # far-field acoustic sponge strength (0 = off, bitwise legacy)
    sponge_j0f::R                            # sponge start as fraction of NJ (default 0.75)
    suth::R                                  # Sutherland S/T_inf (0 for legacy or constant dynamic)
    rhok::Int32                                    # 1 = include -2/3 rho k in the turbulent stress
    gux::A2; guy::A2; gvx::A2; gvy::A2; gTx::A2; gTy::A2   # Green-Gauss cell gradients (interior-sized)
    # dual-time BDF2 history (conservative, interior-sized): U^n and U^(n-1); inert unless step gets dtp>0
    un_r::A2; un_mx::A2; un_my::A2; un_E::A2
    um_r::A2; um_mx::A2; um_my::A2; um_E::A2
    # j-line implicit residual smoothing (IRS-J): dt from the streamwise restriction; wall-normal
    # stiffness absorbed by per-column tridiagonal solves (variable-coefficient IRS). 0 = bitwise off.
    irsj::Int32
    irs_c::A2; irs_d::A2                           # Thomas scratch (interior-sized)
    # ★ VOID-TARGETED DUAL IRS (2026-07-25, no-shortcut high-void fix). irs_ac=1 applies BOTH i- and
    # j-direction implicit residual smoothing with a smoothing coefficient ε = max(the usual CFL-anisotropy
    # ε, irs_eps0*relu(1 - a_m/a_ref)). The VOID term fires where a_m collapses (the cavity) and is 0 in the
    # liquid, so it injects implicit smoothing precisely on the pexc grid-scale (centered-pressure-force)
    # instability that diverges the large cavity -- which the carrier-c CFL ε never catches (a_m<carrier here,
    # a_ref~0.96, and CFLV=0.15 keeps carrier-ν<1). Crucially it is dt-NEUTRAL (plain lam_i+lam_j step kept:
    # NO dt reduction, so inner convergence is NOT slowed -- the exact mechanism that made the REJECTED
    # void-adaptive CFL leak dissipation and kill the certified shedding at finite NINNER). Fixed-point-
    # preserving (M*dc_smoothed = dc_raw => dc_smoothed=0 iff dc_raw=0; the tau FAS source is OFF in the
    # single-grid cavitation runs so this holds exactly). 0 = off (bitwise legacy).
    irs_ac::Int32
    irs_eps0::R                              # void-targeted IRS strength (ε = irs_eps0*relu(1-a_m/a_ref))
    # ★ ACOUSTIC FILTER (2026-07-25, the implicit-flow cure for the split-pexc NEGATIVE-COMPRESSIBILITY
    # divergence). A post-step, Delta-form, ADI line-implicit (Thomas) LOW-PASS of the pseudo-time DENSITY
    # INCREMENT delta = rho - rk_r, gated to the railed negative-compressibility core (gap = cs2*T - a_m^2 > 0),
    # off-diagonal coeff sized on the TRUE-a2m GAP Kf = max(0, cs2*T_band - a2m) (~0.8, NOT a2m~0.002 which is
    # ~400x too weak). It filters the INCREMENT (->0 at pseudo-convergence => FIXED-POINT-EXACT, strictly
    # stronger than IRS which filters the nonzero residual), is ZERO across the vapour-liquid interface (the
    # gate kills the off-diagonal there => NO cross-interface smoothing, unlike IRS), holds momentum invariant
    # (rescale u,v by rho_pre/rho_new => NO added dissipation, unlike the negcomp K_u boost), and is a
    # preconditioner not a step-cut (dtf untouched, unlike the CFL cut). Neumann/zero-row-sum => mass-conserving
    # in the core. Derived + cross-checked (3-way judge panel) 2026-07-25. 0 = off (bitwise legacy).
    acou::Int32
    acou_kfac::R                             # filter strength multiplier (bracket {0.5,1,2}); 1.0 nominal
    acou_d::A2                                     # scratch: density increment delta (dedicated so dc_r stays valid)
    # Selective Frequency Damping (converges UNSTABLE steady states; the fine-mesh steady enabler)
    sfd::Int32
    sfd_chi::R
    sfd_alpha::R
    sb_r::S2; sb_u::S2; sb_v::S2; sb_T::S2
    # PDP-KFVS pressure-split: p_mix = rho*cs2*T_band + pexc. T carries the in-band pedestal
    # T_band=clamp(p_mix/(rho*cs2),0.3,3) on the untouched quadrature; the EXCESS pexc is injected as a
    # face-normal force in the flux (breaks the 3:1 cap). cav LEVEL: 0 = off (bitwise legacy),
    # 1 = Stage-1 pexc split, 2 = +Stage-2 Layer C (a_m-scaled AUSM+-up pressure-diffusion).
    pexc::A2                                        # excess pressure p_mix - rho*cs2*T_band
    cav::Int32
    am::A2                                          # mixture sound speed a_m = sqrt(dp_mix/drho) (Layer C)
    # ★ VOID-ADAPTIVE EXPLICIT CFL (2026-07-25, high-void robustness). The explicit RK3 step sized on
    # the carrier c=3sqrt(T) (railed in-band) UNDER-estimates the stability limit inside a large cavity,
    # where the pexc pressure-split face force is stiff (scales with the mixture EOS stiffness a_m).
    # DIAGNOSED: the Leroux large cavity diverges at full CFLV but converges (rstar 5.4e-4 < control) at
    # half CFLV, and BOTH ni- and dt-refinement destabilise the full-CFLV state (coarse dissipation was
    # masking it). FIX: reduce the local pseudo-time step where a_m collapses (the cavity), by up to the
    # floor cav_cfl, using the physically-appropriate a_m signal. dtf is a pseudo-time/LTS step, so the
    # physical-time (BDF2) fixed point is UNCHANGED -- only the convergence path in the cavity. cav_cfl=0
    # => bitwise legacy. am_ref = liquid a_m (set from tem_ambient).
    cav_cfl::R
    am_ref::R
    # STAGE 1e — mixture-aware far field. The legacy _k_bc_farfield! is a characteristic BC built on the
    # ideal-gas CARRIER eos referenced to T_inf. Under the PDP split the interior T is a RAILED in-band
    # pedestal (clamped to Tmin across the whole liquid once ambient p_mix < rho*cs2*Tmin), NOT a
    # temperature -- so its carrier-entropy reconstruction rb = (cb^2/(gam*sB))^(1/(gam-1)) mixes a railed
    # Td=0.3 against T_inf=1.0 and returns rb ~ 5.6 (outflow) / 0.28 (inflow). Measured: the domain
    # pressurises until sigma_eff ~ 3, so no Tier-C case ever ran at its design cavitation number.
    # ff_mix=1 switches to barotropic characteristics on the mixture sound speed, holding the ambient at
    # ff_p_ref. 0 = bitwise legacy.
    ff_mix::Int32
    ff_p_ref::R                               # ambient mixture pressure p_mix(rho_inf, alpha=0)
    ff_a2m::R                                 # dp_mix/drho at that ambient state
    ff_Tmin::R; ff_Tmax::R              # pedestal band (must match the TEM's)
    # host mirrors of the wall-row metrics (loads/Cp without device scalar reads)
    Sjx1_h::Vector{R}; Sjy1_h::Vector{R}; vol1_h::Vector{R}
    xc1_h::Vector{R};  yc1_h::Vector{R}
    backend::Any
    # ★ S3i EQUILIBRIUM LOG-LAW WALL FUNCTION (src/cmesh/cmesh_wallfun.jl). 0 = bitwise legacy
    # no-slip resolved-gradient shear; >0 substitutes tau_w = rho u_tau^2 in the wall viscous flux.
    # ⚠ VALID ONLY AT y+ 30-100. At y+ ~ 8 the log law is INVALID and this reproduces the four
    # archived divergences -- coarsen the wall spacing alongside enabling it.
    # APPENDED AT THE END: KSol has a POSITIONAL constructor, so a mid-struct insertion silently
    # misassigns every following field. I made exactly that mistake writing this and caught it
    # before it ran -- the same trap cmesh_cav_tem.jl warns about four times.
    wallfun::R; wf_larsson::R
    # ★ S3f BUBBLE-INDUCED TURBULENT KINETIC ENERGY. Precomputed per cell by the TEM sweep (which is
    # the only place alpha, Rdot and R exist together) and simply ADDED to the SST k-production.
    #
    # WHY A PRECOMPUTED ARRAY RATHER THAN A CALL. `cmesh_solver_ka.jl` is a MODULE included BEFORE
    # `cmesh_cav_tem.jl`, so `rp_tke_source` is not in this namespace, and transliterating it here
    # would create exactly the separately-maintained duplicate that this campaign has already been
    # burned by twice (see the warning at the head of tools/topology_core.jl). One array crossing the
    # boundary keeps a single definition of the physics.
    #
    # Zero-filled and written only when `tem.btke > 0`, so with the knob off the SST k-equation is
    # bitwise unchanged.
    btke_src::A2
    # ★★★ S3g STEP 2 -- MIXTURE INTERNAL ENERGY + LATENT HEAT (2026-08-02).
    # `eint` above is a SINGLE constant specific heat used for the whole field, so the marched energy
    # knows nothing about phase. The physically correct closure for a two-phase mixture carries the
    # vapour mass fraction Y = alpha*rho_v/rho and the latent heat of vaporisation L:
    #
    #     E = rho[(1-Y) cv_l T + Y (cv_v T + L)] + rho|u|^2/2
    #     T = (e/rho - |u|^2/2 - Y L) / [(1-Y) cv_l + Y cv_v]
    #
    # ★ THE LATENT HEAT NEEDS NO SOURCE TERM. It is carried inside the conserved energy, so when a cell
    # vaporises, Y rises, the Y*L term consumes internal energy and T FALLS -- automatically, and
    # conservatively. That temperature drop is the thermal cavitation delay (the B-factor effect):
    # evaporation cools the interface, which lowers p_v, which throttles further evaporation. It is a
    # genuine negative feedback on cavity growth and this solver has never had it.
    #
    # `yvf` is the vapour MASS fraction, written by _k_tem_commit! (which owns rho_v_sat). Passing Y
    # rather than alpha keeps rho_v_sat out of the solver entirely -- one less quantity to duplicate
    # across the module boundary, and duplication across that boundary is what produced the EOS split
    # and the two banner copies.
    #
    # emix = 0 => yvf is never read, Y == 0, eint_loc == eint, the latent term vanishes, and every
    # energy expression collapses to the legacy one BITWISE.
    yvf::A2
    emix::R; eint_v::R; latent::R
    # ★ S3j k-eps RNG. 0 = SST (bitwise legacy); >0 reinterprets `tom` as epsilon.
    keps::R
    # ★★★ S3k -- VORTEX THRESHOLD PRESSURE (Le & Tran, JAFM 15(2) 2022, Eq. 14), 2026-08-02.
    #
    #     p_v* = p_v + C_v * mu_eff * |dv/dx - du/dy|
    #
    # WHY THIS TERM AND NOT ANOTHER. Battery 6/6b measured that our thickness deficit is NOT a rate
    # deficit: over Cprod 100 -> 3000 the cavity goes to PURE VAPOUR (amax 0.751 -> 0.9999) while
    # e/c stays FLAT at ~0.038-0.040 against Le/Franc/Michel's criterion e/c > 0.05. More evaporation
    # densifies the same-shaped layer; it does not thicken it. Le & Tran report that the two threshold
    # forms differ in exactly that way: the SINGHAL form (our `pvturb`) evaporates "only at the leading
    # edge and behind the cavity trailing edge", while the VORTEX form evaporates "at the leading edge
    # AND ALONG THE ATTACHED CAVITY SURFACE". Distributed interfacial evaporation is what thickens a
    # layer rather than densifying it. Their case is NACA0015 at alpha = 8 deg -- our angle of attack.
    #
    # This array carries the KINEMATIC group (nu + nu_t)*|Omega|, written by `_k_sst_sweep!` where both
    # factors are already live (Omega is computed for the SST production limiter, nut is on the same
    # line), and consumed by `_k_tem_sweep!` which applies rho -- exactly as the Singhal branch does
    # (`0.195*pvturb*r*kt`), so rho stays consistent with the density the TEM sweep holds.
    # Ghost-sized so the consumer indexes it as [ia,ja] like `tk`.
    #
    # Written unconditionally (it is a diagnostic as well as a source): with `tem.pvvort = 0` nothing
    # READS it, so the physics is bitwise legacy while the magnitude can still be measured. That is
    # deliberate -- it lets GATE 0 double as the admissibility measurement for C_v.
    pvort::A2
    # Outer-boundary segmentation. 0=far field, 1=inlet, 2=lower slip wall, 3=upper slip wall.
    # APPEND-ONLY: KSol uses the positional constructor above.
    outer_face_kind::M
    outer_mode::Int32                              # 0 = legacy all-far-field path, 1 = segmented channel
    # Discrete freestream-preservation correction for the channel operator. Channel-only full-size
    # arrays; legacy states receive 1x1 stubs and never launch the subtraction kernel.
    outer_ref_r::A2; outer_ref_mx::A2; outer_ref_my::A2; outer_ref_e::A2
    outer_ref_rho::R; outer_ref_ux::R; outer_ref_uy::R; outer_ref_T::R
    outer_ref_ready::Int32
    outer_ref_signature::UInt64
    # Opt-in fixed-static-pressure channel outlet. The alpha array aliases the attached TEM;
    # legacy states retain 1x1 stubs and never access them. Append-only positional constructor.
    outlet_mode::Int32
    outlet_p::R
    outlet_eos::NTuple{7,R}
    outlet_alpha::A2
    outlet_status::A2
    # Actual globally applied step. Zero denotes local stepping (no physical clock).
    last_global_dt::R

    # Airfoil-only air transport: 0 legacy, 1 Sutherland dynamic, 2 constant dynamic.
    air_sutherland::Int32

    # Validation opt-in: ideal-gas characteristic condition at both downstream
    # i-end boundaries. Zero preserves the historical zero-gradient outlet.
    characteristic_i_outlet::Int32
    # Opt-in airfoil SST scalar gradients, local coefficients, and shared face
    # fluxes. Appended to preserve KSol's positional constructor mapping.
    sst_gkx::A2; sst_gky::A2; sst_gwx::A2; sst_gwy::A2
    sst_sigk::A2; sst_sigw::A2; sst_nu::A2
    sst_Fik::A2; sst_Fiw::A2; sst_Fjk::A2; sst_Fjw::A2
end

function _dev(backend, x::AbstractArray{T}) where {T}
    a = KA.allocate(backend, T, size(x)...)
    copyto!(a, x); a
end

function _outer_boundary_contract_ka(met, NI::Int)
    kinds = hasproperty(met, :outer_face_kind) ?
            UInt8.(collect(met.outer_face_kind)) : fill(UInt8(0), NI)
    length(kinds) == NI || throw(ArgumentError("outer_face_kind must have NI=$NI entries"))
    provenance = hasproperty(met, :channel_provenance) ? met.channel_provenance : false
    provenance isa Bool || throw(ArgumentError("channel_provenance must be Bool"))
    # External conservative-Species runs use the very same node/metric arrays as
    # the gas solver, with every outer face characteristic. This is explicit
    # provenance, rather than a tunnel mask masquerading as an external domain.
    free_fluid = hasproperty(met, :free_fluid_provenance) ? met.free_fluid_provenance : false
    free_fluid isa Bool || throw(ArgumentError("free_fluid_provenance must be Bool"))
    if free_fluid
        !provenance && all(==(0x01), kinds) || throw(ArgumentError(
            "free-fluid Species metrics require exclusively characteristic outer faces and no tunnel provenance"))
        return kinds, Int32(1)
    end
    if !provenance
        all(iszero, kinds) || throw(ArgumentError(
            "nonzero outer_face_kind requires channel-provenance metrics from build_metrics(::ChannelMesh)"))
        return kinds, Int32(0)
    end
    # Strip topology (5 Oct 2026, memory/project_docs/2026-10-05_STRIP_TOPOLOGY_PLAN.md section 3.2): the strip block's metrics
    # (build_strip_metrics) tag every outer face OUTER_STRIP_INTERFACE = 0x04: its j = NJ+1 row is the upper interface
    # with the C block, so the block has no outer boundary at all, outer_mode 2. Only the strip's own carrier carries it;
    # the channel contract below is not applied to it, and a mixed mask is refused.
    if any(==(0x04), kinds)
        all(==(0x04), kinds) || throw(ArgumentError(
            "the strip interface tag (OUTER_STRIP_INTERFACE = 4) must cover every outer face of a strip block (outer_mode 2); a mixed mask is refused"))
        return kinds, Int32(2)
    end

    all(k -> k == 0x01 || k == 0x02 || k == 0x03, kinds) || throw(ArgumentError(
        "channel-provenance metrics require only inlet/lower-wall/upper-wall outer tags"))
    first(kinds) == 0x02 || throw(ArgumentError(
        "channel-provenance outer mask must start on the lower wall"))
    last(kinds) == 0x03 || throw(ArgumentError(
        "channel-provenance outer mask must end on the upper wall"))
    any(==(0x01), kinds) || throw(ArgumentError(
        "channel-provenance outer mask must contain an inlet segment"))
    rank(k) = k == 0x02 ? 1 : k == 0x01 ? 2 : 3
    all(rank(kinds[i]) <= rank(kinds[i+1]) for i in 1:NI-1) || throw(ArgumentError(
        "channel-provenance outer mask must be one contiguous lower -> inlet -> upper sequence"))
    kinds, Int32(1)
end

_ksol_precision_scalar(::Type{R},x::Float32) where R=R(x)
_ksol_precision_scalar(::Type{R},x::Tuple) where R=map(y->_ksol_precision_scalar(R,y),x)
_ksol_precision_scalar(::Type{R},x) where R=x
_ksol_precision(::Type{R},args...) where R=KSol(map(x->_ksol_precision_scalar(R,x),args)...)

function build_sol_ka(backend, met; drot=2.0f0, cxv, cyv, czv, wv,
                      sfd_storage_type::Type{<:AbstractFloat}=Float32,
                      scalar_type::Type{<:AbstractFloat}=Float32)
    scalar_type in (Float32,Float64)||throw(ArgumentError("scalar_type must be Float32 or Float64"))
    sfd_storage_type in (Float32, Float64) ||
        throw(ArgumentError("sfd_storage_type must be Float32 or Float64"))
    NI, NJ, g = met.NI, met.NJ, 3
    dev(x) = _dev(backend, x)
    z(nx, ny) = _dev(backend, zeros(scalar_type, nx, ny))
    all(a->eltype(a)===scalar_type,(met.Six,met.Siy,met.Sjx,met.Sjy,met.vol,met.xc,met.yc,met.wdist,met.gammat))||
        throw(ArgumentError("Solver precision requires metrics built in the same scalar_type from original nodes"))
    scalar_type===Float64&&eltype(wv)!==Float64&&throw(ArgumentError("Float64 thermal solver requires original Float64 quadrature weights"))
    # The default calls the original allocator so all existing runs retain
    # Float32 SFD state and its bitwise numerical path. The opt-in Float64
    # accumulator avoids a frozen low-alpha filter on tiny pseudo-time steps.
    zsfd(nx, ny) = sfd_storage_type === scalar_type ? z(nx, ny) :
                    dev(zeros(sfd_storage_type, nx, ny))
    outer_face_kind_h, outer_mode = _outer_boundary_contract_ka(met, NI)
    zr() = outer_mode == Int32(0) ? z(1, 1) : z(NI, NJ)
    scalar_type===Float64&&outer_mode!=0&&throw(ArgumentError("Float64 thermal precision currently supports free-air C/A+ only"))
    cv=scalar_type===Float32 ? 0.5f0*(3.0f0+drot)*CS2 : scalar_type(.5)*(scalar_type(3)+scalar_type(drot))*_gas_cs2(scalar_type)
    _ksol_precision(scalar_type,NI, NJ, met.NIW, met.NIF, g,
        dev(met.Six), dev(met.Siy), dev(met.Sjx), dev(met.Sjy), dev(met.vol), dev(met.xc), dev(met.yc),
        z(NI+2g, NJ+2g), z(NI+2g, NJ+2g), z(NI+2g, NJ+2g), z(NI+2g, NJ+2g),
        z(NI+2g, NJ+2g), z(NI+2g, NJ+2g), z(NI+2g, NJ+2g), z(NI+2g, NJ+2g),
        z(NI+1, NJ), z(NI+1, NJ), z(NI+1, NJ), z(NI+1, NJ),
        z(NI, NJ+1), z(NI, NJ+1), z(NI, NJ+1), z(NI, NJ+1),
        z(NI, NJ), z(NI, NJ), z(NI, NJ), z(NI, NJ),
        z(NI, NJ),
        dev(collect(scalar_type.(cxv))), dev(collect(scalar_type.(cyv))), dev(collect(scalar_type.(czv))), dev(collect(scalar_type.(wv))),   # collect: SVector→Vector (CuArray copyto! from SVector scalar-iterates)
        scalar_type(drot), cv,
        1.0f0, 0.0f0, 0.0f0, 1.0f0,
        0.0f0, 0.25f0, 0.0f0,
        0.0f0, 1.0f0, 1.0f0, 0.0f0, false, 0.0f0,
        z(NI+2g, NJ+2g), z(NI+2g, NJ+2g), z(NI+2g, NJ+2g),
        z(NI+2g, NJ+2g),
        false, 0.0f0, 0.0f0, Int32(0), 0.35f0,
        0.0f0,                                         # fbm_cdelta (FBM OFF => bitwise legacy)
        0.0f0,                                         # pans_fk    (PANS OFF => bitwise legacy)
        Int32(0), z(NI, NJ), z(NI, NJ), z(NI, NJ),
        dev(met.wdist), dev(met.gammat), Int32(0), Int32(0), 0.0f0, 0.75f0, 0.0f0, Int32(0),
        z(NI, NJ), z(NI, NJ), z(NI, NJ), z(NI, NJ), z(NI, NJ), z(NI, NJ),
        z(NI, NJ), z(NI, NJ), z(NI, NJ), z(NI, NJ),
        z(NI, NJ), z(NI, NJ), z(NI, NJ), z(NI, NJ),
        Int32(0), z(NI, NJ), z(NI, NJ),
        Int32(0), 0.0f0,                               # irs_ac (void-targeted dual IRS OFF => bitwise legacy), irs_eps0
        Int32(0), 1.0f0, z(NI, NJ),                     # acou (acoustic filter OFF => bitwise legacy), acou_kfac, acou_d
        Int32(0), 0.01f0, 0.001f0,
        zsfd(NI+2g, NJ+2g), zsfd(NI+2g, NJ+2g),
        zsfd(NI+2g, NJ+2g), zsfd(NI+2g, NJ+2g),
        z(NI+2g, NJ+2g), Int32(0), z(NI+2g, NJ+2g),    # pexc, cav, am (PDP pressure-split; off by default)
        0.0f0, 1.0f0,                                  # cav_cfl (void-adaptive CFL OFF), am_ref (liquid a_m)
        Int32(0), 0.0f0, 0.0f0, 0.3f0, 3.0f0,          # Stage 1e far field: OFF => bitwise legacy
        met.Sjx[:,1], met.Sjy[:,1], met.vol[:,1], met.xc[:,1], met.yc[:,1],
        backend,
        0.0f0, 0.0f0,   # wallfun, wf_larsson: OFF => bitwise legacy no-slip
        z(NI, NJ),      # btke_src: zero unless the TEM sweep writes it (tem.btke > 0)
        z(NI+2g, NJ+2g), 0.0f0, 0.0f0, 0.0f0,   # yvf, emix, eint_v, latent: OFF => bitwise legacy
        0.0f0,                                  # keps: OFF => bitwise legacy SST
        z(NI+2g, NJ+2g),                        # pvort: (nu+nu_t)*|Omega|, written always, read only if tem.pvvort>0
        dev(outer_face_kind_h), outer_mode,
        zr(), zr(), zr(), zr(), 0.0f0, 0.0f0, 0.0f0, 0.0f0, Int32(0), UInt64(0),
        Int32(0), 0.0f0, ntuple(_ -> 0.0f0, 7), z(1,1), z(2,NJ+2g), 0.0f0,
        Int32(0), Int32(0),
        z(1,1), z(1,1), z(1,1), z(1,1),
        z(1,1), z(1,1), z(1,1),
        z(1,1), z(1,1), z(1,1), z(1,1))
end

@inline _channel_sigword(x) = UInt64(reinterpret(UInt32, x))
@inline _channel_sigword(x::Int32) = UInt64(reinterpret(UInt32, x))
@inline _channel_sigword(x::Int) = UInt64(x)
@inline _channel_sigword(x::Bool) = x ? UInt64(1) : UInt64(0)

"""Fingerprint every scalar that changes the channel reference operator.

The inner-wall model is deliberately absent: the reference operator always uses a transparent foil
and no analytic foil-wall flux, so changing the physical foil treatment does not require reseeding.
"""
function _channel_signature_ka(s::KSol)
    h = UInt64(0xcbf29ce484222325)
    for x in (s.NI, s.NJ, s.NIW, s.NIF, s.g, s.outer_mode,
              s.drot, s.rho_inf, s.uxi, s.uyi, s.T_inf,
              s.m2, s.beta, s.ca, s.sa,
              s.nu_lam, s.sst, s.k_inf, s.w_inf, s.keps,
              s.earsm, s.recon, s.fullvisc, s.suth, s.rhok,
              s.cav, s.am_ref, s.ff_mix, s.ff_p_ref, s.ff_a2m,
              s.ff_Tmin, s.ff_Tmax)
        h = xor(h, _channel_sigword(x))*UInt64(0x100000001b3)
    end
    if s.outlet_mode != Int32(0)
        for x in (s.outlet_mode, s.outlet_p, s.outlet_eos...)
            h = xor(h, _channel_sigword(x))*UInt64(0x100000001b3)
        end
    end
    h
end

"""Invalidate a previously seeded channel freestream reference."""
function clear_channel_reference_ka!(s::KSol)
    s.outer_ref_ready = Int32(0)
    s.outer_ref_signature = UInt64(0)
    nothing
end

function _channel_reference_ambient_ka(s::KSol)
    s.outer_mode != Int32(0) || throw(ArgumentError("channel reference seeding requires segmented channel boundaries"))
    s.ff_mix in (Int32(0), Int32(1)) || throw(ArgumentError("channel reference requires ff_mix to be 0 or 1"))
    isfinite(s.rho_inf) && s.rho_inf > 0.0f0 || throw(ArgumentError("channel reference requires rho_inf > 0"))
    all(isfinite, (s.uxi, s.uyi, s.T_inf)) || throw(ArgumentError("channel reference freestream contains a non-finite value"))

    Tref = s.T_inf
    pexc_ref = 0.0f0
    am_ref = s.am_ref
    if s.ff_mix == Int32(1)
        isfinite(s.ff_p_ref) && s.ff_p_ref > 0.0f0 || throw(ArgumentError("channel mixture reference requires ff_p_ref > 0"))
        isfinite(s.ff_a2m) && s.ff_a2m > 0.0f0 || throw(ArgumentError("channel mixture reference requires ff_a2m > 0"))
        isfinite(s.ff_Tmin) && isfinite(s.ff_Tmax) && 0.0f0 < s.ff_Tmin <= s.ff_Tmax ||
            throw(ArgumentError("channel mixture reference requires 0 < ff_Tmin <= ff_Tmax"))
        Tb = s.ff_p_ref/(s.rho_inf*CS2)
        Tref = Tb < s.ff_Tmin ? s.ff_Tmin : (Tb > s.ff_Tmax ? s.ff_Tmax : Tb)
        pexc_ref = s.ff_p_ref - s.rho_inf*CS2*Tref
        am_ref = sqrt(s.ff_a2m)
    elseif s.cav >= Int32(2)
        isfinite(am_ref) && am_ref > 0.0f0 ||
            throw(ArgumentError("channel cav>=2 reference requires am_ref > 0 when ff_mix is off"))
    end
    isfinite(Tref) && Tref > 0.0f0 || throw(ArgumentError("channel reference temperature must be positive"))

    kref = 0.0f0
    nutref = 0.0f0
    if s.sst
        isfinite(s.k_inf) && s.k_inf >= 0.0f0 || throw(ArgumentError("channel SST reference requires k_inf >= 0"))
        isfinite(s.w_inf) && s.w_inf > 0.0f0 || throw(ArgumentError("channel SST reference requires w_inf > 0"))
        kref = s.k_inf
        nutref = s.keps > 0.0f0 ? KE_CMU*s.k_inf*s.k_inf/s.w_inf : s.k_inf/s.w_inf
        isfinite(nutref) && nutref >= 0.0f0 || throw(ArgumentError("channel SST reference eddy viscosity is invalid"))
    end
    (rho=s.rho_inf, ux=s.uxi, uy=s.uyi, T=Tref, pexc=pexc_ref, am=am_ref,
     tk=kref, nut=nutref)
end

function set_freestream_ka!(s::KSol, mach::Float64, alpha_deg::Float64; pg_correction::Bool=true)
    F=eltype(s.rho);CS2=_gas_cs2(F)
    a = sqrt((1 + 2/(3+s.drot)) * CS2)
    U = F(mach*a)
    s.ca = F(cosd(alpha_deg)); s.sa = F(sind(alpha_deg))
    s.uxi = U*s.ca; s.uyi = U*s.sa
    s.rho_inf = 1.0f0; s.T_inf = 1.0f0
    s.m2 = pg_correction ? F(mach^2) : 0.0f0
    s.beta = pg_correction ? F(sqrt(max(1 - mach^2, 1e-6))) : 1.0f0
    fill!(s.rho, s.rho_inf); fill!(s.ux, s.uxi); fill!(s.uy, s.uyi); fill!(s.T, s.T_inf)
    clear_channel_reference_ka!(s)
    nothing
end

"""Enable dynamic Sutherland air viscosity for the single-phase transonic airfoil path.

`s.T` is normalized by freestream static temperature. `s.nu_lam` remains the
freestream kinematic viscosity, so the molecular dynamic viscosity is
`rho_inf*nu_lam*_mufac(T, S/T_inf)` and the local kinematic viscosity is that
quantity divided by local density. The separate flag preserves the historical
positive-`s.suth` path when this mode is off.
"""
function set_air_sutherland_ka!(s::KSol, temperature_inf_k::Real;
                                sutherland_k::Real=110.4)
    Tinf = Float64(temperature_inf_k); S = Float64(sutherland_k)
    isfinite(Tinf) && Tinf > 0.0 ||
        throw(ArgumentError("freestream static temperature must be finite and positive"))
    isfinite(S) && S > 0.0 ||
        throw(ArgumentError("Sutherland constant must be finite and positive"))
    sr = eltype(s.rho)(S/Tinf)
    isfinite(sr) && sr > 0.0f0 ||
        throw(ArgumentError("Sutherland ratio is outside Float32 range"))
    _check_air_sutherland_admission(s)
    s.suth = sr
    s.air_sutherland = Int32(1)
    nothing
end

"""Enable constant molecular dynamic viscosity for the single-phase airfoil path.
The nondimensional value is `rho_inf*nu_lam`; local kinematic viscosity
is this value divided by local density. No Kelvin temperature is needed.
"""
function set_air_constant_dynamic_ka!(s::KSol)
    _check_air_sutherland_admission(s)
    s.suth = 0.0f0
    s.air_sutherland = Int32(2)
    nothing
end

@inline function _air_nu_local(rho, T, nu_inf,
                               rho_inf, Sr, mode::Int32)
    mode != Int32(0) ? rho_inf*nu_inf*_mufac(T,Sr)/rho : nu_inf
end

function _check_air_sutherland_admission(s::KSol)
    (s.cav == Int32(0) && s.outer_mode == Int32(0) &&
     s.outlet_mode == Int32(0) && s.ff_mix == Int32(0) &&
     s.wallfun == 0.0f0 &&
     s.irsj == Int32(0) && s.irs_ac == Int32(0) && s.earsm == Int32(0) &&
     s.keps == 0.0f0) || throw(ArgumentError(
        "dynamic air viscosity requires single-phase free-air SST, without wall functions, IRS, EARSM or k-epsilon"))
    nothing
end

"""STAGE 1e — switch the far field to mixture-aware barotropic characteristics.

`p_ref` is the ambient MIXTURE pressure `p_mix(rho_inf, alpha=0)` and `a2m` is `dp_mix/drho` there; get
both from `tem_ambient(tem, s)` so they track the eos/wallis branch exactly. Leaves every other case
bitwise-identical (`ff_mix` defaults to 0).

WHY THIS EXISTS: the legacy characteristic far field is built on the ideal-gas CARRIER eos referenced to
`T_inf`. Under the PDP split the interior `T` is a railed in-band pedestal, not a temperature, so the
carrier-entropy reconstruction returns densities off by ~5x and the domain pressurises until
`sigma_eff ~ 3`. Structurally unavoidable: `p_ref/p_inf = sigma*(1+eps)*gamma*Ma^2/2 = 0.031` at Ma 0.2,
sigma 1 -- and that ratio is invariant under `T_inf`."""
function set_farfield_mixture!(s::KSol; p_ref::Float64, a2m::Float64,
                               Tmin=0.3f0, Tmax=3.0f0)
    a2m > 0.0 || error("set_farfield_mixture!: a2m must be > 0 (got $a2m)")
    p_ref > 0.0 || error("set_farfield_mixture!: p_ref must be > 0 (got $p_ref)")
    s.ff_mix = Int32(1); s.ff_p_ref = Float32(p_ref); s.ff_a2m = Float32(a2m)
    s.ff_Tmin = Tmin; s.ff_Tmax = Tmax
    clear_channel_reference_ka!(s)
    nothing
end

function set_sst_ka!(s::KSol; tu_inf::Float64=0.001, mu_t_ratio::Float64=0.1)
    U = sqrt(Float64(s.uxi)^2 + Float64(s.uyi)^2)
    s.k_inf = eltype(s.rho)(1.5*(tu_inf*U)^2)
    s.w_inf = eltype(s.rho)(max(s.k_inf/(mu_t_ratio*Float64(s.nu_lam)), 1e-6))
    fill!(s.tk, s.k_inf); fill!(s.tom, s.w_inf); fill!(s.nut, s.k_inf/s.w_inf)
    fill!(s.tnl, s.k_inf/s.w_inf)
    s.sst = true
    clear_channel_reference_ka!(s)
    nothing
end

"""
    set_keps_ka!(s::KSol)

Convert an SST-initialised state to k-EPSILON variables. MUST be called after `set_sst_ka!` whenever
`s.keps > 0`.

★ WHY THIS EXISTS, and it was found the hard way. `set_sst_ka!` fills `tom` with `w_inf`, a SPECIFIC
dissipation rate. The k-eps branch reinterprets that same field as the DISSIPATION RATE epsilon. The
two differ by eps = Cmu k omega, which at this campaign's freestream (k_inf = 5.6e-8, w_inf = 2.32)
is a factor of 2e8. Loading omega where epsilon is expected therefore drove

    nu_t = Cmu k^2 / eps  ~  1e-16

i.e. the first k-eps runs were LAMINAR runs wearing a turbulence model. The tell was that the Singhal
term -- which is proportional to k and moved amax 0.718 -> 0.928 on its own -- became EXACTLY inert
when combined with k-eps, identical to six significant figures. A strong term cannot be exactly inert
unless it is reading zero.

⇒ Reinterpreting a transported field is not free: every initial value, every boundary value and every
derived quantity built from it has to be converted too.
"""
function set_keps_ka!(s::KSol)
    s.keps > 0.0f0 || return nothing
    eps_inf = max(KE_CMU*s.k_inf*s.w_inf, 1.0f-30)     # eps = Cmu k omega
    s.w_inf = eps_inf                                   # the BC kernels read w_inf for BOTH closures
    nut_inf = KE_CMU*s.k_inf*s.k_inf/eps_inf            # == k_inf/omega_inf, but derived consistently
    fill!(s.tom, eps_inf); fill!(s.nut, nut_inf); fill!(s.tnl, nut_inf)
    clear_channel_reference_ka!(s)
    nothing
end

# ---- BC kernels ------------------------------------------------------------------------------
# ★★ CAVITATION-AWARE WALL SLIP (2026-07-29). MOTIVATION, measured this session
# (memory/project_docs/2026-07-29_INVERTED_CAVITY_THE_WALL_LAYER.md): a homogeneous-mixture model with a RESOLVED
# viscous sublayer enforces LIQUID no-slip in cells that are physically in contact with VAPOUR. An
# attached sheet cavity means the wall touches vapour — there is no liquid boundary layer there to
# resolve. Measured consequence at y+~0.7: the wall row carries only alpha ~ 0.14 under an alpha=0.94
# core 0.024c above it (the INVERSE of Stutz-Reboud/Coutier-Delgosha, who find high void ONLY close
# to the foil), the near-wall flow stalls at u_t=+0.024U instead of reversing, and no re-entrant jet
# forms. The y+~8 mesh — the archive's only genuine detacher — instead reaches alpha 0.4-0.84 AT the
# wall, because its first cell sits in the log layer and can be evacuated.
# TREATMENT: blend the ghost state between no-slip (liquid) and slip (vapour) by the local vapour
# fraction, so the wall shear is scaled by the LIQUID fraction and vanishes as alpha -> 1:
#     u_ghost = (1-s)*(-u0) + s*(u0 - 2*un*n),   s = cavslip * alpha_eq(rho)^cavslip_p
# alpha_eq(rho) = (rls - rho)/(rls - rvs) is the EOS-equilibrium void, VERIFIED against the
# transported alpha to 1.4% in the field probe, so no alpha plumbing is needed in the BC.
# cavslip = 0 => the branch is not taken and every legacy path is bitwise.
@kernel function _k_bc_wake_wall!(rho, ux, uy, T, @Const(Sjx), @Const(Sjy),
                                  wallmode::Int32, cavslip, cavslip_p,
                                  rls, rvs,
                                  NI::Int32, NIW::Int32, NIF::Int32, g::Int32)
    i, k = @index(Global, NTuple)                     # i in 1..NI, k in 1..g
    @inbounds if i <= NI && k <= g
        if i <= NIW || i > NIW+NIF                    # wake cut: partner exchange, no sign flips
            ip = NI+1-i
            rho[i+g, g+1-k] = rho[ip+g, k+g]; ux[i+g, g+1-k] = ux[ip+g, k+g]
            uy[i+g, g+1-k]  = uy[ip+g, k+g];  T[i+g, g+1-k]  = T[ip+g, k+g]
        elseif wallmode == Int32(0)                   # transparent (gate-a stub)
            rho[i+g, g+1-k] = rho[i+g, g+k]; ux[i+g, g+1-k] = ux[i+g, g+k]
            uy[i+g, g+1-k]  = uy[i+g, g+k];  T[i+g, g+1-k]  = T[i+g, g+k]
        elseif wallmode == Int32(1)                   # slip: reflect about the wall tangent
            Sx = Sjx[i,1]; Sy = Sjy[i,1]; Sm = sqrt(Sx*Sx+Sy*Sy); nxw = Sx/Sm; nyw = Sy/Sm
            u0 = ux[i+g, g+k]; v0 = uy[i+g, g+k]; un = u0*nxw + v0*nyw
            rho[i+g, g+1-k] = rho[i+g, g+k]
            ux[i+g, g+1-k]  = u0 - 2.0f0*un*nxw
            uy[i+g, g+1-k]  = v0 - 2.0f0*un*nyw
            T[i+g, g+1-k]   = T[i+g, g+k]
        else                                          # noslip (+ optional cavitation-aware slip blend)
            u0 = ux[i+g, g+k]; v0 = uy[i+g, g+k]
            rho[i+g, g+1-k] = rho[i+g, g+k]; T[i+g, g+1-k] = T[i+g, g+k]
            if cavslip > 0.0f0
                aeq = (rls - rho[i+g, g+k])/(rls - rvs)          # EOS-equilibrium void at this cell
                aeq = aeq < 0.0f0 ? 0.0f0 : (aeq > 1.0f0 ? 1.0f0 : aeq)
                sf  = cavslip*(cavslip_p == 1.0f0 ? aeq : aeq^cavslip_p)
                sf  = sf > 1.0f0 ? 1.0f0 : sf                    # 0 = no-slip (liquid), 1 = slip (vapour)
                Sx = Sjx[i,1]; Sy = Sjy[i,1]; Sm = sqrt(Sx*Sx+Sy*Sy); nxw = Sx/Sm; nyw = Sy/Sm
                un = u0*nxw + v0*nyw
                ux[i+g, g+1-k] = (1.0f0-sf)*(-u0) + sf*(u0 - 2.0f0*un*nxw)
                uy[i+g, g+1-k] = (1.0f0-sf)*(-v0) + sf*(v0 - 2.0f0*un*nyw)
            else
                ux[i+g, g+1-k] = -u0; uy[i+g, g+1-k] = -v0
            end
        end
    end
end

@kernel function _k_bc_farfield!(rho, ux, uy, T, @Const(xc), @Const(yc), @Const(Sjx), @Const(Sjy),
                                 rho_inf, uxi, uyi, T_inf,
                                 gam_v, xv, yv, m2, betav,
                                 ca, sa, drot,
                                 @Const(pexc), ff_mix::Int32, ff_p_ref, ff_a2m,
                                 ff_Tmin, ff_Tmax,
                                 NI::Int32, NJ::Int32, g::Int32)
    CS2=_gas_cs2(eltype(rho))
    i = @index(Global, Linear)
    @inbounds if i <= NI
        # PG vortex increment at the j=NJ interior center (freestream-frame angular form)
        dux = 0.0f0; duy = 0.0f0
        if gam_v != 0.0f0
            dx = xc[i, NJ] - xv; dy = yc[i, NJ] - yv
            dxs = dx*ca + dy*sa; dys = -dx*sa + dy*ca
            r2 = dxs*dxs + dys*dys + 1.0f-6
            Fc = betav/(1.0f0 - m2*(dys*dys/r2))
            c = gam_v*Fc/(6.2831853f0*r2)
            dus = c*dys; dvs = -c*dxs
            dux = dus*ca - dvs*sa; duy = dus*sa + dvs*ca
        end
        uxt = uxi + dux; uyt = uyi + duy
        gam_eos = (5.0f0 + drot)/(3.0f0 + drot); gm1 = gam_eos - 1.0f0; Rgas = CS2
        Sx = Sjx[i,NJ+1]; Sy = Sjy[i,NJ+1]; Sm = sqrt(Sx*Sx+Sy*Sy)
        nxb = Sx/Sm; nyb = Sy/Sm; txb = -nyb; tyb = nxb
        rd = rho[i+g, NJ+g]; uxd = ux[i+g, NJ+g]; uyd = uy[i+g, NJ+g]; Td = T[i+g, NJ+g]
        cd_ = sqrt(gam_eos*Rgas*max(Td, 1.0f-4)); cinf = sqrt(gam_eos*Rgas*T_inf)
        qnd = uxd*nxb + uyd*nyb
        qni = uxt*nxb + uyt*nyb
        rb = rd; uxb = uxd; uyb = uyd; Tb = Td
        if ff_mix == Int32(1)
            # STAGE 1e — BAROTROPIC MIXTURE FAR FIELD.
            # At the far field alpha=0, so p_mix = p_mix(rho) alone and the system is barotropic Euler
            # with sound speed a_m = sqrt(dp_mix/drho). The correct invariants are the linear-acoustic
            # pair R+- = u_n +- p/(rho_ref*a_m) about the ambient state, NOT the ideal-gas ones: under PDP
            # the carrier T is a railed pedestal and carries no thermodynamic information at all.
            am_r = sqrt(ff_a2m); Zc = rho_inf*am_r          # ambient acoustic impedance
            p_d = rd*CS2*Td + pexc[i+g, NJ+g]               # interior MIXTURE pressure (pexc = PDP excess)
            if qnd <= -am_r                                  # supersonic inflow: impose the ambient
                rb = rho_inf; uxb = uxt; uyb = uyt
                Tbb = ff_p_ref/(rho_inf*CS2)
                Tb = Tbb < ff_Tmin ? ff_Tmin : (Tbb > ff_Tmax ? ff_Tmax : Tbb)
            elseif qnd < am_r                                # subsonic: 1 outgoing + 1 incoming
                Rout = qnd + p_d/Zc                          # leaves: carries the interior state
                Rin  = qni - ff_p_ref/Zc                     # enters: carries the DESIGN ambient
                qnb  = 0.5f0*(Rout + Rin)
                p_b  = 0.5f0*Zc*(Rout - Rin)
                p_b  = p_b < 1.0f-8 ? 1.0f-8 : p_b
                rbb  = rho_inf + (p_b - ff_p_ref)/ff_a2m     # linearised about the ambient state
                rb   = rbb < 1.0f-4 ? 1.0f-4 : rbb
                utb  = qnb >= 0.0f0 ? (uxd*txb + uyd*tyb) : (uxt*txb + uyt*tyb)
                uxb  = qnb*nxb + utb*txb; uyb = qnb*nyb + utb*tyb
                Tbb  = p_b/(rb*CS2)
                Tb   = Tbb < ff_Tmin ? ff_Tmin : (Tbb > ff_Tmax ? ff_Tmax : Tbb)
            end
            # supersonic outflow (qnd >= am_r): everything extrapolates -- rb/uxb/uyb/Tb keep interior
        elseif qnd <= -cd_
            rb = rho_inf; uxb = uxt; uyb = uyt; Tb = T_inf
        elseif qnd < cd_
            Rout = qnd + 2.0f0*cd_/gm1
            Rin  = qni - 2.0f0*cinf/gm1
            qnb  = 0.5f0*(Rout+Rin)
            cb   = 0.25f0*gm1*(Rout-Rin); cb = cb < 1.0f-4 ? 1.0f-4 : cb
            sB = 0.0f0; utb = 0.0f0
            if qnb >= 0.0f0
                sB = (rd*Rgas*Td)/(rd^gam_eos); utb = uxd*txb + uyd*tyb
            else
                sB = (rho_inf*Rgas*T_inf)/(rho_inf^gam_eos); utb = uxt*txb + uyt*tyb
            end
            rb  = (cb*cb/(gam_eos*sB))^(1.0f0/gm1)
            Tb  = (sB*rb^gam_eos)/(rb*Rgas)
            uxb = qnb*nxb + utb*txb; uyb = qnb*nyb + utb*tyb
        end
        for k in 1:g
            rho[i+g, NJ+g+k] = rb; ux[i+g, NJ+g+k] = uxb; uy[i+g, NJ+g+k] = uyb; T[i+g, NJ+g+k] = Tb
        end
    end
end

# Segmented-channel post-override. The legacy characteristic kernel still owns FARFIELD/INLET
# faces; only wall-labelled faces are replaced here. Mirroring every layer keeps the WENO stencil
# consistent, and reflecting only the face-normal velocity gives an exact inviscid slip state.
@kernel function _k_bc_outer_slip!(rho, ux, uy, T, pexc, am,
                                   @Const(Sjx), @Const(Sjy), @Const(outer_face_kind),
                                   NI::Int32, NJ::Int32, g::Int32)
    i, k = @index(Global, NTuple)
    @inbounds if i <= NI && k <= g && outer_face_kind[i] >= 0x02
        js = NJ + g + Int32(1) - k
        jd = NJ + g + k
        Sx = Sjx[i, NJ+Int32(1)]; Sy = Sjy[i, NJ+Int32(1)]
        Sm = sqrt(Sx*Sx + Sy*Sy); nx = Sx/Sm; ny = Sy/Sm
        u0 = ux[i+g, js]; v0 = uy[i+g, js]; un = u0*nx + v0*ny
        rho[i+g, jd] = rho[i+g, js]
        ux[i+g, jd] = u0 - 2.0f0*un*nx
        uy[i+g, jd] = v0 - 2.0f0*un*ny
        T[i+g, jd] = T[i+g, js]
        pexc[i+g, jd] = pexc[i+g, js]
        am[i+g, jd] = am[i+g, js]
    end
end

@kernel function _k_bc_channel_aux_j!(pexc, am, @Const(rho), @Const(ux), @Const(uy),
                                      @Const(T), @Const(Sjx), @Const(Sjy),
                                      @Const(outer_face_kind),
                                      ff_mix::Int32, ff_p_ref, ff_a2m,
                                      rho_inf,
                                      NI::Int32, NIW::Int32, NIF::Int32,
                                      NJ::Int32, g::Int32)
    i, k = @index(Global, NTuple)
    @inbounds if i <= NI && k <= g
        # Inner C-mesh boundary: wake-cut partner exchange, or layerwise scalar mirror on the foil.
        if i <= NIW || i > NIW+NIF
            ip = NI + Int32(1) - i
            pexc[i+g, g+Int32(1)-k] = pexc[ip+g, k+g]
            am[i+g, g+Int32(1)-k] = am[ip+g, k+g]
        else
            pexc[i+g, g+Int32(1)-k] = pexc[i+g, g+k]
            am[i+g, g+Int32(1)-k] = am[i+g, g+k]
        end

        # The characteristic kernel owns primitive inlet ghosts but cannot return p_b. Match its
        # branch split exactly: supersonic outflow extrapolates every quantity, while all other
        # mixture branches recover p_b from the density returned by the linear barotropic relation.
        if outer_face_kind[i] == 0x01 && ff_mix == Int32(1)
            jd = NJ + g + k
            Sx = Sjx[i,NJ+Int32(1)]; Sy = Sjy[i,NJ+Int32(1)]
            Sm = sqrt(Sx*Sx + Sy*Sy)
            qnd = (ux[i+g,NJ+g]*Sx + uy[i+g,NJ+g]*Sy)/Sm
            am_r = sqrt(ff_a2m)
            if qnd >= am_r
                pexc[i+g, jd] = pexc[i+g, NJ+g]
                am[i+g, jd] = am[i+g, NJ+g]
            else
                rb = rho[i+g, jd]
                pb = ff_p_ref + ff_a2m*(rb - rho_inf)
                pexc[i+g, jd] = pb - rb*CS2*T[i+g, jd]
                am[i+g, jd] = am_r
            end
        end
    end
end

@kernel function _k_bc_channel_aux_io!(pexc, am, NI::Int32, NJ::Int32, g::Int32)
    jj, k = @index(Global, NTuple)
    @inbounds if jj <= NJ+2g && k <= g
        pexc[g+Int32(1)-k, jj] = pexc[g+Int32(1), jj]
        am[g+Int32(1)-k, jj] = am[g+Int32(1), jj]
        pexc[NI+g+k, jj] = pexc[NI+g, jj]
        am[NI+g+k, jj] = am[NI+g, jj]
    end
end

"""Apply only the segmented outer slip-wall primitive override (a no-op in legacy mode)."""
function outer_bc_override_ka!(s::KSol)
    s.outer_mode == Int32(0) && return nothing
    b = s.backend; NI, NJ, g = Int32(s.NI), Int32(s.NJ), Int32(s.g)
    _k_bc_outer_slip!(b)(s.rho, s.ux, s.uy, s.T, s.pexc, s.am,
                         s.Sjx, s.Sjy, s.outer_face_kind, NI, NJ, g;
                         ndrange=(s.NI, s.g))
    nothing
end

@kernel function _k_bc_outlet!(rho, ux, uy, T, NI::Int32, NJ::Int32, g::Int32)
    jj, k = @index(Global, NTuple)                    # jj over all rows incl ghosts, k in 1..g
    @inbounds if jj <= NJ+2g && k <= g
        rho[g+1-k, jj] = rho[g+1, jj]; ux[g+1-k, jj] = ux[g+1, jj]
        uy[g+1-k, jj]  = uy[g+1, jj];  T[g+1-k, jj]  = T[g+1, jj]
        rho[NI+g+k, jj] = rho[NI+g, jj]; ux[NI+g+k, jj] = ux[NI+g, jj]
        uy[NI+g+k, jj]  = uy[NI+g, jj]; T[NI+g+k, jj]  = T[NI+g, jj]
    end
end

# The C-grid's two i ends are downstream boundaries. At subsonic outflow a
# single acoustic characteristic enters the domain and must be prescribed.
# Use the same normal ideal-gas invariants as _k_bc_farfield!, with the local
# i-face outward normal; keep interior entropy and tangential velocity for
# outflow. This kernel is opt-in and overrides only physical-row i ghosts.
@kernel function _k_bc_characteristic_i_outlet!(rho, ux, uy, T,
        @Const(xc), @Const(yc), @Const(Six), @Const(Siy),
        rho_inf, uxi, uyi, T_inf,
        gam_v, xv, yv, m2,
        betav, ca, sa, drot,
        NI::Int32, NJ::Int32, g::Int32)
    CS2=_gas_cs2(eltype(rho))
    j, side = @index(Global, NTuple)
    @inbounds if j <= NJ && side <= Int32(2)
        face = side == Int32(1) ? Int32(1) : NI+Int32(1)
        ia = side == Int32(1) ? g+Int32(1) : NI+g
        ja = j+g
        sign = side == Int32(1) ? -1.0f0 : 1.0f0
        Sx = sign*Six[face,j]; Sy = sign*Siy[face,j]
        Sm = sqrt(Sx*Sx+Sy*Sy)
        nxb = Sx/Sm; nyb = Sy/Sm; txb = -nyb; tyb = nxb
        rd = rho[ia,ja]; uxd = ux[ia,ja]; uyd = uy[ia,ja]; Td = T[ia,ja]
        uxt = uxi; uyt = uyi
        if gam_v != 0.0f0
            # Match the existing outer characteristic far state's PG vortex.
            dx = xc[side == Int32(1) ? Int32(1) : NI,j]-xv
            dy = yc[side == Int32(1) ? Int32(1) : NI,j]-yv
            dxs = dx*ca+dy*sa; dys = -dx*sa+dy*ca
            r2 = dxs*dxs+dys*dys+1.0f-6
            Fc = betav/(1.0f0-m2*(dys*dys/r2))
            c = gam_v*Fc/(6.2831853f0*r2)
            dus = c*dys; dvs = -c*dxs
            uxt += dus*ca-dvs*sa
            uyt += dus*sa+dvs*ca
        end
        gam_eos = (5.0f0+drot)/(3.0f0+drot)
        gm1 = gam_eos-1.0f0; Rgas = CS2
        cd_ = sqrt(gam_eos*Rgas*max(Td,1.0f-4))
        cinf = sqrt(gam_eos*Rgas*T_inf)
        qnd = uxd*nxb+uyd*nyb
        qni = uxt*nxb+uyt*nyb
        rb = rd; uxb = uxd; uyb = uyd; Tb = Td
        if qnd <= -cd_
            rb = rho_inf; uxb = uxt; uyb = uyt; Tb = T_inf
        elseif qnd < cd_
            Rout = qnd+2.0f0*cd_/gm1
            Rin = qni-2.0f0*cinf/gm1
            qnb = 0.5f0*(Rout+Rin)
            cb = 0.25f0*gm1*(Rout-Rin)
            cb = cb < 1.0f-4 ? 1.0f-4 : cb
            if qnb >= 0.0f0
                sB = (rd*Rgas*Td)/(rd^gam_eos)
                utb = uxd*txb+uyd*tyb
            else
                sB = (rho_inf*Rgas*T_inf)/(rho_inf^gam_eos)
                utb = uxt*txb+uyt*tyb
            end
            rb = (cb*cb/(gam_eos*sB))^(1.0f0/gm1)
            Tb = (sB*rb^gam_eos)/(rb*Rgas)
            uxb = qnb*nxb+utb*txb
            uyb = qnb*nyb+utb*tyb
        end
        for k in 1:g
            ghost = side == Int32(1) ? g+Int32(1)-k : NI+g+k
            rho[ghost,ja]=rb; ux[ghost,ja]=uxb
            uy[ghost,ja]=uyb; T[ghost,ja]=Tb
        end
    end
end

# far-field corners of the outlet ghost columns (small fixed fill)
@kernel function _k_bc_corners!(rho, ux, uy, T, rho_inf, uxi, uyi, T_inf,
                                NI::Int32, NJ::Int32, g::Int32)
    kk, k = @index(Global, NTuple)
    @inbounds if kk <= g && k <= g
        rho[kk, NJ+g+k] = rho_inf; ux[kk, NJ+g+k] = uxi; uy[kk, NJ+g+k] = uyi; T[kk, NJ+g+k] = T_inf
        rho[NI+g+kk, NJ+g+k] = rho_inf; ux[NI+g+kk, NJ+g+k] = uxi; uy[NI+g+kk, NJ+g+k] = uyi; T[NI+g+kk, NJ+g+k] = T_inf
    end
end

function applyBC_ka!(s::KSol, wallmode::Int32; cavslip=0.0f0, cavslip_p=1.0f0,
                     cavslip_rls=1.0f0, cavslip_rvs=0.0f0,
                     defer_pressure_outlet::Bool=false)
    # Strip topology (5 Oct 2026, plan sections 4.1 and 4.5; critique A 3 and 9): this legacy row/column sequence would
    # write an outlet copy on the strip's base-wall side, a far-field state on its upper-interface rows and a within-block
    # "partner" on its lower-interface rows. The strip carrier's ghosts come from the species projection and the pair
    # exchange only, so the legacy path (TOPO_SPECIES=0, step_ka!) cannot run on it.
    s.outer_mode == Int32(2) && throw(ArgumentError(
        "applyBC_ka! is not defined on the strip carrier (outer_mode 2, CMeshSpeciesStripTopologyV1): its i = 1 side is the base wall and its two rows are interfaces; the legacy path does not support the strip topology"))
    b = s.backend; g, NI, NJ = Int32(s.g), Int32(s.NI), Int32(s.NJ)
    _k_bc_wake_wall!(b)(s.rho, s.ux, s.uy, s.T, s.Sjx, s.Sjy, wallmode,
                        cavslip, cavslip_p, cavslip_rls, cavslip_rvs,
                        NI, Int32(s.NIW), Int32(s.NIF), g; ndrange=(s.NI, s.g))
    _k_bc_farfield!(b)(s.rho, s.ux, s.uy, s.T, s.xc, s.yc, s.Sjx, s.Sjy,
                       s.rho_inf, s.uxi, s.uyi, s.T_inf, s.gam, s.xv, s.yv, s.m2, s.beta, s.ca, s.sa,
                       s.drot, s.pexc, s.ff_mix, s.ff_p_ref, s.ff_a2m, s.ff_Tmin, s.ff_Tmax,
                       NI, NJ, g; ndrange=s.NI)
    outer_bc_override_ka!(s)
    if s.outer_mode != Int32(0)
        _k_bc_channel_aux_j!(b)(s.pexc, s.am, s.rho, s.ux, s.uy, s.T,
                                s.Sjx, s.Sjy, s.outer_face_kind,
                                s.ff_mix, s.ff_p_ref, s.ff_a2m, s.rho_inf,
                                NI, Int32(s.NIW), Int32(s.NIF), NJ, g;
                                ndrange=(s.NI, s.g))
    end
    _k_bc_corners!(b)(s.rho, s.ux, s.uy, s.T, s.rho_inf, s.uxi, s.uyi, s.T_inf, NI, NJ, g; ndrange=(s.g, s.g))
    _k_bc_outlet!(b)(s.rho, s.ux, s.uy, s.T, NI, NJ, g; ndrange=(s.NJ+2s.g, s.g))
    if s.characteristic_i_outlet == Int32(1)
        (s.outer_mode == Int32(0) && s.outlet_mode == Int32(0) &&
         s.cav == Int32(0) && s.ff_mix == Int32(0)) ||
            throw(ArgumentError("characteristic i outlet requires single-phase free-air mode"))
        _k_bc_characteristic_i_outlet!(b)(s.rho,s.ux,s.uy,s.T,
            s.xc,s.yc,s.Six,s.Siy,s.rho_inf,s.uxi,s.uyi,s.T_inf,
            s.gam,s.xv,s.yv,s.m2,s.beta,s.ca,s.sa,s.drot,
            NI,NJ,g;ndrange=(s.NJ,2))
    end
    if s.outer_mode != Int32(0)
        _k_bc_channel_aux_io!(b)(s.pexc, s.am, NI, NJ, g; ndrange=(s.NJ+2s.g, s.g))
    end
    defer_pressure_outlet || pressure_outlet_bc!(s)
    nothing
end

# ---- flux/update kernels -----------------------------------------------------------------------
# A2c full-stress prerequisite: Green-Gauss cell gradients of (u, v, T), interior cells, ghost
# states already filled by the BC kernels this stage (in-order stream => no explicit sync needed).
@kernel function _k_gradients!(gux, guy, gvx, gvy, gTx, gTy, @Const(ux), @Const(uy), @Const(T),
                               @Const(Six), @Const(Siy), @Const(Sjx), @Const(Sjy), @Const(vol),
                               NI::Int32, NJ::Int32, g::Int32, ctx)
    i, j = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ
        # Strip topology (5 Oct 2026, plan section 4.4; critique A 4): with a strip pair's `ctx` the cells beside an
        # interface (the C block's leg columns at j = 1, the strip's rows 1 and NS) take the distance-weighted face value
        # there, out of line; every other cell, and every cell of a launch without a context (compiled away), is today's.
        if ctx !== nothing && _strip_gg_cell(i, j, NJ, ctx)
            r6 = _strip_gg_gradients(ux, uy, T, Six, Siy, Sjx, Sjy, vol, Int32(i), Int32(j), NJ, g, ctx)
            gux[i,j] = r6[1]; guy[i,j] = r6[2]; gvx[i,j] = r6[3]; gvy[i,j] = r6[4]; gTx[i,j] = r6[5]; gTy[i,j] = r6[6]
        else
        ia = i+g; ja = j+g; iv = 1.0f0/vol[i,j]
        SxW=Six[i,j]; SyW=Siy[i,j]; SxE=Six[i+1,j]; SyE=Siy[i+1,j]
        SxS=Sjx[i,j]; SyS=Sjy[i,j]; SxN=Sjx[i,j+1]; SyN=Sjy[i,j+1]
        r = _gg_grad(ux[ia-1,ja],ux[ia+1,ja],ux[ia,ja-1],ux[ia,ja+1],ux[ia,ja],
                     SxW,SyW,SxE,SyE,SxS,SyS,SxN,SyN, iv)
        gux[i,j] = r[1]; guy[i,j] = r[2]
        r = _gg_grad(uy[ia-1,ja],uy[ia+1,ja],uy[ia,ja-1],uy[ia,ja+1],uy[ia,ja],
                     SxW,SyW,SxE,SyE,SxS,SyS,SxN,SyN, iv)
        gvx[i,j] = r[1]; gvy[i,j] = r[2]
        r = _gg_grad(T[ia-1,ja],T[ia+1,ja],T[ia,ja-1],T[ia,ja+1],T[ia,ja],
                     SxW,SyW,SxE,SyE,SxS,SyS,SxN,SyN, iv)
        gTx[i,j] = r[1]; gTy[i,j] = r[2]
        end
    end
end

@kernel function _k_flux_i!(Fi_r, Fi_mx, Fi_my, Fi_e, @Const(rho), @Const(ux), @Const(uy), @Const(T),
                            @Const(Six), @Const(Siy), @Const(xc), @Const(yc), @Const(nut), @Const(tk),
                            @Const(exx), @Const(exy), @Const(eyy),
                            @Const(gux), @Const(guy), @Const(gvx), @Const(gvy), @Const(gTx), @Const(gTy),
                            @Const(cxa), @Const(cya), @Const(cza), @Const(wa),
                            drot, nu_lam, sst_on::Int32, recon::Int32, fullvisc::Int32,
                            suth, rho_inf, air_sutherland::Int32,
                            rhok::Int32, earsm::Int32, @Const(pexc), @Const(am), cav::Int32,
                            eint, minf2, pc::Int32, @Const(pmix), direct_p::Int32, face_states,air_geometry,
                            half, rate, NI::Int32, NJ::Int32, g::Int32, ctx)
    CS2=_gas_cs2(eltype(rho))
    i, j = @index(Global, NTuple)                     # i in 1..NI+1, j in 1..NJ
    @inbounds if i <= NI+1 && j <= NJ
        # Strip topology (5 Oct 2026, plan section 4.1, the i-face wall family): `ctx` is nothing on every carrier but a
        # strip pair's (compiled away: the default path is this kernel as it was); on the strip block (ctx.iwall == 1)
        # its i = 1 faces are the trailing-edge base, an impermeable no-slip wall on an i face.
        if ctx!==nothing && ctx.iwall==Int32(1) && i==1
            wr,wmx,wmy,we=_strip_iwall_flux(rho,ux,uy,T,pexc,pmix,ctx.vol,Six,Siy,Int32(j),g,nu_lam,suth,rho_inf,air_sutherland,cav,direct_p,ctx.wallmode,fullvisc)
            Fi_r[i,j]=wr; Fi_mx[i,j]=wmx; Fi_my[i,j]=wmy; Fi_e[i,j]=we
        else
        nxr = Six[i,j]; nyr = Siy[i,j]; S = sqrt(nxr*nxr + nyr*nyr)
        nx = nxr/S; ny = nyr/S
        ia = i+g; ja = j+g
        local Fr, Fmx, Fmy, Fe
        # `half`/`rate` non-nothing selects the common-gauge KINETIC (KFVS) face on the same
        # admitted direct-pressure states (2026-09-10): half masses M+/M- (area-integrated) and
        # the operational face clock are published for the Species vapor transport and cap.
        if air_geometry!==nothing
            air_left=i==1 ? _airfoil_air_z(rho,ux,uy,T,ia-1,ja) : _airfoil_trace4(face_states,i-1,j,2)
            air_right=i==NI+1 ? _airfoil_air_z(rho,ux,uy,T,ia,ja) : _airfoil_trace4(face_states,i,j,1)
            Fr,Fmx,Fmy,Fe=_airfoil_thermal_face_flux(air_left,air_right,nx,ny,S,cxa,cya,cza,wa,drot)
        elseif face_states!==nothing && 2<=i<=NI
            l=ntuple(k->face_states[i-1,j,k,2],Val(7))
            r=ntuple(k->face_states[i,j,k,1],Val(7))
            if half!==nothing
                Fr,Fmx,Fmy,Fe,mp,mm,rt=_fvflux_kfvs_pc(l[1],l[2],l[3],l[5],l[6],
                    r[1],r[2],r[3],r[5],r[6],cxa,cya,wa,nx,ny,S)
                half[i,j,1]=mp; half[i,j,2]=mm; rate[i,j]=rt
            else
                Fr,Fmx,Fmy,Fe=_fvflux_mix_pc(l[1],l[2],l[3],l[4],l[5],
                    r[1],r[2],r[3],r[4],r[5],l[6],r[6],eint,minf2,nx,ny,S)
            end
        elseif cav >= Int32(4)
            # ★ cav level 4 (2026-09-03): direct-pressure MIXTURE face on (rho, u, p_mix, a_m):
            # AUSM+-up split by default, or the kinetic KFVS face when `half` is supplied. Either
            # replaces the pedestal KFVS carrier, the centred pexc force AND Layer C at this face.
            if pc==Int32(1) && direct_p==Int32(1)
                if half!==nothing
                    Fr,Fmx,Fmy,Fe,mp,mm,rt=_fvflux_kfvs_pc(
                        rho[ia-1,ja],ux[ia-1,ja],uy[ia-1,ja],pmix[ia-1,ja],am[ia-1,ja],
                        rho[ia,ja],ux[ia,ja],uy[ia,ja],pmix[ia,ja],am[ia,ja],cxa,cya,wa,nx,ny,S)
                    half[i,j,1]=mp; half[i,j,2]=mm; rate[i,j]=rt
                else
                    Fr,Fmx,Fmy,Fe=_fvflux_mix_pc(
                        rho[ia-1,ja],ux[ia-1,ja],uy[ia-1,ja],T[ia-1,ja],pmix[ia-1,ja],
                        rho[ia,ja],ux[ia,ja],uy[ia,ja],T[ia,ja],pmix[ia,ja],
                        am[ia-1,ja],am[ia,ja],eint,minf2,nx,ny,S)
                end
            elseif pc == Int32(1)
                pL = direct_p==Int32(1) ? pmix[ia-1,ja] : rho[ia-1,ja]*CS2*T[ia-1,ja]+pexc[ia-1,ja]
                pR = direct_p==Int32(1) ? pmix[ia,ja] : rho[ia,ja]*CS2*T[ia,ja]+pexc[ia,ja]
                Fr, Fmx, Fmy, Fe = _fvflux_mix6(
                rho[ia-1,ja],ux[ia-1,ja],uy[ia-1,ja],T[ia-1,ja], pL,
                rho[ia-1,ja],ux[ia-1,ja],uy[ia-1,ja],T[ia-1,ja], pL,
                rho[ia-1,ja],ux[ia-1,ja],uy[ia-1,ja],T[ia-1,ja], pL,
                rho[ia,  ja],ux[ia,  ja],uy[ia,  ja],T[ia,  ja], pR,
                rho[ia,  ja],ux[ia,  ja],uy[ia,  ja],T[ia,  ja], pR,
                rho[ia,  ja],ux[ia,  ja],uy[ia,  ja],T[ia,  ja], pR,
                am[ia-1,ja], am[ia,ja], eint, minf2, nx, ny, S, Int32(0))
            else
            Fr, Fmx, Fmy, Fe = _fvflux_mix6(
                rho[ia-3,ja],ux[ia-3,ja],uy[ia-3,ja],T[ia-3,ja], rho[ia-3,ja]*CS2*T[ia-3,ja]+pexc[ia-3,ja],
                rho[ia-2,ja],ux[ia-2,ja],uy[ia-2,ja],T[ia-2,ja], rho[ia-2,ja]*CS2*T[ia-2,ja]+pexc[ia-2,ja],
                rho[ia-1,ja],ux[ia-1,ja],uy[ia-1,ja],T[ia-1,ja], rho[ia-1,ja]*CS2*T[ia-1,ja]+pexc[ia-1,ja],
                rho[ia,  ja],ux[ia,  ja],uy[ia,  ja],T[ia,  ja], rho[ia,  ja]*CS2*T[ia,  ja]+pexc[ia,  ja],
                rho[ia+1,ja],ux[ia+1,ja],uy[ia+1,ja],T[ia+1,ja], rho[ia+1,ja]*CS2*T[ia+1,ja]+pexc[ia+1,ja],
                rho[ia+2,ja],ux[ia+2,ja],uy[ia+2,ja],T[ia+2,ja], rho[ia+2,ja]*CS2*T[ia+2,ja]+pexc[ia+2,ja],
                am[ia-1,ja], am[ia,ja], eint, minf2, nx, ny, S, recon)
            end
        else
            Fr, Fmx, Fmy, Fe = _fvflux_normal6(
                rho[ia-3,ja],ux[ia-3,ja],uy[ia-3,ja],T[ia-3,ja],
                rho[ia-2,ja],ux[ia-2,ja],uy[ia-2,ja],T[ia-2,ja],
                rho[ia-1,ja],ux[ia-1,ja],uy[ia-1,ja],T[ia-1,ja],
                rho[ia,  ja],ux[ia,  ja],uy[ia,  ja],T[ia,  ja],
                rho[ia+1,ja],ux[ia+1,ja],uy[ia+1,ja],T[ia+1,ja],
                rho[ia+2,ja],ux[ia+2,ja],uy[ia+2,ja],T[ia+2,ja],
                nx, ny, S, cxa, cya, cza, wa, drot, recon)
        end
        if nu_lam > 0.0f0 && i >= 2 && i <= NI        # viscous (interior i-faces)
            dxc = air_geometry===nothing ? xc[i,j]-xc[i-1,j] : -air_geometry[i,j,1,1]
            dyc = air_geometry===nothing ? yc[i,j]-yc[i-1,j] : -air_geometry[i,j,2,1]
            dn = sqrt(dxc*dxc+dyc*dyc)+1.0f-30
            nut_f = sst_on == Int32(1) ? 0.5f0*(nut[ia-1,ja]+nut[ia,ja]) : 0.0f0
            rf = 0.5f0*(rho[ia-1,ja]+rho[ia,ja])
            fmu = _mufac(0.5f0*(T[ia-1,ja]+T[ia,ja]), suth)   # Sutherland mu(T)
            cpg = 0.5f0*(5.0f0+drot)*CS2
            mu_eff = air_sutherland != Int32(0) ?
                     rho_inf*nu_lam*fmu + rf*nut_f : rf*(nu_lam*fmu+nut_f)
            k_eff = air_sutherland != Int32(0) ?
                    cpg*(rho_inf*nu_lam*fmu/0.71f0 + rf*nut_f/0.9f0) :
                    rf*cpg*(nu_lam*fmu/0.71f0 + nut_f/0.9f0)
            if fullvisc == Int32(1)                   # A2c full stress tensor
                Fvx, Fvy, Fve = _viscous_face_full(
                    gux[i-1,j],guy[i-1,j],gvx[i-1,j],gvy[i-1,j],gTx[i-1,j],gTy[i-1,j],
                    gux[i,j],  guy[i,j],  gvx[i,j],  gvy[i,j],  gTx[i,j],  gTy[i,j],
                    ux[ia-1,ja],uy[ia-1,ja],T[ia-1,ja], ux[ia,ja],uy[ia,ja],T[ia,ja],
                    dxc, dyc, nx, ny, S, mu_eff,
                    direct_p==Int32(1) ? 0.0f0 : k_eff)
                Fmx -= Fvx; Fmy -= Fvy; Fe -= Fve
            else                                      # thin-layer (legacy bitwise)
                mu = mu_eff*S/dn
                du = ux[ia,ja]-ux[ia-1,ja]; dv = uy[ia,ja]-uy[ia-1,ja]; dT = T[ia,ja]-T[ia-1,ja]
                ufx = 0.5f0*(ux[ia-1,ja]+ux[ia,ja]); ufy = 0.5f0*(uy[ia-1,ja]+uy[ia,ja])
                Fmx -= mu*du; Fmy -= mu*dv
                Fe  -= mu*(ufx*du+ufy*dv) + (direct_p==Int32(1) ? 0.0f0 : k_eff*S/dn*dT)
            end
            if rhok == Int32(1) && sst_on == Int32(1) # -2/3 rho k in the turbulent stress
                kf = 0.5f0*(tk[ia-1,ja]+tk[ia,ja])
                ufx2 = 0.5f0*(ux[ia-1,ja]+ux[ia,ja]); ufy2 = 0.5f0*(uy[ia-1,ja]+uy[ia,ja])
                Fmx += 0.6666667f0*rf*kf*nxr
                Fmy += 0.6666667f0*rf*kf*nyr
                Fe  += 0.6666667f0*rf*kf*(ufx2*nxr + ufy2*nyr)
            end
            if earsm == Int32(1) && sst_on == Int32(1) # WJ-EARSM extra anisotropy stress (F -= tau·n·S)
                e11 = 0.5f0*(exx[i-1,j]+exx[i,j]); e12 = 0.5f0*(exy[i-1,j]+exy[i,j])
                e22 = 0.5f0*(eyy[i-1,j]+eyy[i,j])
                fex = e11*nxr + e12*nyr; fey = e12*nxr + e22*nyr
                ufx3 = 0.5f0*(ux[ia-1,ja]+ux[ia,ja]); ufy3 = 0.5f0*(uy[ia-1,ja]+uy[ia,ja])
                Fmx -= fex; Fmy -= fey
                Fe  -= ufx3*fex + ufy3*fey
            end
        end
        if cav >= Int32(1) && cav < Int32(4)              # PDP pressure-split: excess p_mix as a face force (cav 1-3)
            pexf = 0.5f0*(pexc[ia-1,ja] + pexc[ia,ja])    # GCL-conservative (uniform p_exc -> zero net force)
            Fmx += pexf*nxr; Fmy += pexf*nyr
            ufx = 0.5f0*(ux[ia-1,ja]+ux[ia,ja]); ufy = 0.5f0*(uy[ia-1,ja]+uy[ia,ja])
            Fe  += pexf*(ufx*nxr + ufy*nyr)
        end
        if cav >= Int32(2) && cav < Int32(4)              # Layer C: PRECONDITIONED a_m AUSM+-up (anti-checkerboard + shock speed) (cav 2-3)
            pL = rho[ia-1,ja]*CS2*T[ia-1,ja] + pexc[ia-1,ja]
            pR = rho[ia,  ja]*CS2*T[ia,  ja] + pexc[ia,  ja]
            unL = ux[ia-1,ja]*nx + uy[ia-1,ja]*ny; unR = ux[ia,ja]*nx + uy[ia,ja]*ny
            amf = 0.5f0*(am[ia-1,ja] + am[ia,ja])
            amp = max(amf, 0.5f0*abs(unL+unR), 0.05f0)    # preconditioned sound speed: floored at |u| => bounded 1/a
            # ★ NEGCOMP STABILISATION (cav>=3, 2026-07-25). The split pexc face force is CENTERED and has
            # d(pexc)/drho = a2m - cs2*T_band < 0 in the railed cavity core (anti-diffusive, self-amplifying
            # clumping -- the confirmed large-cavity divergence). Its AUSM K_u velocity-diffusion is sized on
            # the collapsed mixture a_m (amf~0.04) => ~20x too weak to damp it. Floor the K_u speed at the
            # anti-diffusion speed sqrt(|a2m - cs2*T_band|) -- NONZERO ONLY where compressibility is negative
            # (railed core), ZERO at the unrailed closure (spares the shedding). K_p (mass) uses the original
            # amp (unchanged). cav<3 => amp_u=amp => bitwise legacy.
            amp_u = amp
            if cav >= Int32(3)
                Tf = 0.5f0*(T[ia-1,ja] + T[ia,ja])
                amp_u = max(amp, sqrt(max(0.0f0, CS2*Tf - amf*amf)))
            end
            Fr  += -0.25f0*(pR - pL)/amp*S                # K_p pressure-diffusion mass flux (anti-checkerboard, a_m-aware)
            rf   = 0.5f0*(rho[ia-1,ja]+rho[ia,ja])
            pu   = -0.5f0*rf*amp_u*(unR - unL)            # K_u velocity-diffusion (momentum; negcomp-floored at cav>=3)
            Fmx += pu*nxr; Fmy += pu*nyr
        end
        Fi_r[i,j]=Fr; Fi_mx[i,j]=Fmx; Fi_my[i,j]=Fmy; Fi_e[i,j]=Fe
        end
    end
end

# The strip block's base wall on its i = 1 face j (strip topology, 5 Oct 2026, plan section 4.1; critique B 3.1): the
# j = 1 foil-wall branch of _k_flux_j! below, transcribed to an i face without its wall-function and log-law arms (the
# all-y+ wall function is refused with a strip): no mass or energy flux, the wall pressure p_w S n of the wall cell, and
# the Stokes traction with d1 = 0.5 vol[1,j]/|S|. The stored normal (Six, Siy) of an i = 1 face points into the fluid, as
# the j = 1 normal does, and both enter the divergence with a minus sign, so the same expressions hold. Kept out of line
# (noinline): a large inline branch inside a kernel once broke GPU compilation.
@noinline function _strip_iwall_flux(rho,ux,uy,T,pexc,pmix,vol,Six,Siy,j::Int32,g::Int32,nu_lam,suth,
                                     rho_inf,air_sutherland::Int32,cav::Int32,direct_p::Int32,wallmode::Int32,fullvisc::Int32=Int32(0))
    CS2=_gas_cs2(eltype(rho))
    @inbounds begin
        nxr=Six[1,j]; nyr=Siy[1,j]; S=sqrt(nxr*nxr+nyr*nyr)
        nx=nxr/S; ny=nyr/S
        ia=g+Int32(1); ja=j+g
        pw=rho[ia,ja]*CS2*T[ia,ja]+(cav>=Int32(1) ? pexc[ia,ja] : 0.0f0)
        direct_p==Int32(1) && (pw=pmix[ia,ja])
        Fr=0.0f0; Fmx=pw*nxr; Fmy=pw*nyr; Fe=0.0f0
        if nu_lam>0.0f0 && (direct_p==Int32(0) || wallmode!=Int32(1))
            d1=0.5f0*vol[1,j]/S+1.0f-30
            muw=(air_sutherland != Int32(0) ?
                 rho_inf*nu_lam*_mufac(T[ia,ja],suth) :
                 rho[ia,ja]*nu_lam*_mufac(T[ia,ja],suth))*S/d1
            if direct_p==Int32(1) || (cav==Int32(0) && fullvisc==Int32(1))
                tx,ty=_wall_stokes_traction(ux[ia,ja],uy[ia,ja],nx,ny,muw)
                Fmx-=tx; Fmy-=ty
            else
                Fmx-=muw*ux[ia,ja]
                Fmy-=muw*uy[ia,ja]
            end
        end
        (Fr,Fmx,Fmy,Fe)
    end
end

# The station and the interface of a j = 1 interface face of column i (strip topology): on the strip (block 1) column i
# is station i of the lower interface; on the C block (block 0) a lower-leg column i <= NIW is station NIW+1-i of the lower
# interface and an upper-leg column i > NIW+NIF station i-NIW-NIF of the upper one (plan section 2's correspondence).
@inline _strip_station(i,NIW,NIF,block::Int32)=block==Int32(1) ? (Int32(i),Int32(1)) :
    (Int32(i)<=Int32(NIW) ? (Int32(NIW)+Int32(1)-Int32(i),Int32(1)) : (Int32(i)-Int32(NIW)-Int32(NIF),Int32(2)))

# Green-Gauss gradients beside the strip pair's interfaces (strip topology, 5 Oct 2026, plan section 4.4; critique A 4).
# A cell beside an interface: the C block's leg columns at j = 1 (the interface is their S face) and the strip's rows 1 (S
# face) and NS (N face). Their neighbour across the interface lies in the other block and may be 200 times taller (the
# legs' first row is dn1-thin beside the corners, the strip's edge cells h/NS), where the plain face value 0.5 (phi_c +
# phi_nb) is no longer the value at the face: the GG gradient of a linear field is then wrong by about h_nb/(4 h_c). At
# those faces the face value is the distance-weighted (h_nb phi_c + h_c phi_nb)/(h_c + h_nb), h = vol/|S| of each cell at
# the face (the own cell's from its metrics, in the operations the static table used for the neighbour's, ctx.h_nb); every
# other face of the cell keeps the plain value of _gg_grad, in its arithmetic. Both representations of an interface face
# therefore use one face value (the same products and sums, commuted). Only a strip pair's launches reach these helpers.
@inline _strip_gg_cell(i,j,NJ,ctx)=ctx.block==Int32(1) ? (j==1 || j==NJ) :
    (j==1 && (Int32(i)<=ctx.NIW || Int32(i)>ctx.NIW+ctx.NIF))
@inline function _strip_gg_one(phiW,phiE,phiS,phiN,phiC,ws::Bool,wn::Bool,hS,nS,hN,nN,
                               SxW,SyW,SxE,SyE,SxS,SyS,SxN,SyN,invvol)
    fE = 0.5f0*(phiC+phiE); fW = 0.5f0*(phiW+phiC)
    fN = wn ? (nN*phiC+hN*phiN)/(hN+nN) : 0.5f0*(phiC+phiN)
    fS = ws ? (nS*phiC+hS*phiS)/(hS+nS) : 0.5f0*(phiS+phiC)
    ((fE*SxE - fW*SxW + fN*SxN - fS*SxS)*invvol,
     (fE*SyE - fW*SyW + fN*SyN - fS*SyS)*invvol)
end
# The interface weights of cell (i, j): (ws, wn, hS, nS, hN, nN), the own and the neighbour's heights at its S and N faces.
@inline function _strip_gg_weights(Sjx,Sjy,vol,i::Int32,j::Int32,NJ::Int32,ctx)
    @inbounds begin
        ws=j==Int32(1) && (ctx.block==Int32(1) || i<=ctx.NIW || i>ctx.NIW+ctx.NIF)
        wn=ctx.block==Int32(1) && j==NJ
        hS=1.0f0;nS=1.0f0;hN=1.0f0;nN=1.0f0
        if ws
            sx=Sjx[i,j];sy=Sjy[i,j];hS=vol[i,j]/sqrt(sx*sx+sy*sy)
            st,fc=_strip_station(i,ctx.NIW,ctx.NIF,ctx.block);nS=ctx.h_nb[st,fc]
        end
        if wn
            sx=Sjx[i,j+Int32(1)];sy=Sjy[i,j+Int32(1)];hN=vol[i,j]/sqrt(sx*sx+sy*sy);nN=ctx.h_nb[i,Int32(2)]
        end
        (ws,wn,hS,nS,hN,nN)
    end
end
# The six gradients (dudx, dudy, dvdx, dvdy, dTdx, dTdy) of _k_gradients! for a cell beside an interface (out of line).
@noinline function _strip_gg_gradients(ux,uy,T,Six,Siy,Sjx,Sjy,vol,i::Int32,j::Int32,NJ::Int32,g::Int32,ctx)
    @inbounds begin
        ia = i+g; ja = j+g; iv = 1.0f0/vol[i,j]
        SxW=Six[i,j]; SyW=Siy[i,j]; SxE=Six[i+Int32(1),j]; SyE=Siy[i+Int32(1),j]
        SxS=Sjx[i,j]; SyS=Sjy[i,j]; SxN=Sjx[i,j+Int32(1)]; SyN=Sjy[i,j+Int32(1)]
        ws,wn,hS,nS,hN,nN=_strip_gg_weights(Sjx,Sjy,vol,i,j,NJ,ctx)
        gu=_strip_gg_one(ux[ia-1,ja],ux[ia+1,ja],ux[ia,ja-1],ux[ia,ja+1],ux[ia,ja],ws,wn,hS,nS,hN,nN,
                         SxW,SyW,SxE,SyE,SxS,SyS,SxN,SyN,iv)
        gv=_strip_gg_one(uy[ia-1,ja],uy[ia+1,ja],uy[ia,ja-1],uy[ia,ja+1],uy[ia,ja],ws,wn,hS,nS,hN,nN,
                         SxW,SyW,SxE,SyE,SxS,SyS,SxN,SyN,iv)
        gt=_strip_gg_one(T[ia-1,ja],T[ia+1,ja],T[ia,ja-1],T[ia,ja+1],T[ia,ja],ws,wn,hS,nS,hN,nN,
                         SxW,SyW,SxE,SyE,SxS,SyS,SxN,SyN,iv)
        (gu[1],gu[2],gv[1],gv[2],gt[1],gt[2])
    end
end

@kernel function _k_flux_j!(Fj_r, Fj_mx, Fj_my, Fj_e, @Const(rho), @Const(ux), @Const(uy), @Const(T),
                            @Const(Sjx), @Const(Sjy), @Const(xc), @Const(yc), @Const(nut), @Const(vol), @Const(tk),
                            @Const(exx), @Const(exy), @Const(eyy),
                            @Const(gux), @Const(guy), @Const(gvx), @Const(gvy), @Const(gTx), @Const(gTy),
                            @Const(cxa), @Const(cya), @Const(cza), @Const(wa),
                            drot, nu_lam, sst_on::Int32, wallflux::Int32, recon::Int32, fullvisc::Int32,
                            suth, rho_inf, air_sutherland::Int32,
                            rhok::Int32, earsm::Int32, @Const(pexc), @Const(am), cav::Int32,
                            wallfun, wf_larsson, eint, minf2, pc::Int32,
                            @Const(pmix), direct_p::Int32, wallmode::Int32, face_states,air_geometry,
                            half, rate, NI::Int32, NJ::Int32, NIW::Int32, NIF::Int32, g::Int32, ctx)
    CS2=_gas_cs2(eltype(rho))
    i, j = @index(Global, NTuple)                     # i in 1..NI, j in 1..NJ+1
    @inbounds if i <= NI && j <= NJ+1
        nxr = Sjx[i,j]; nyr = Sjy[i,j]; S = sqrt(nxr*nxr + nyr*nyr)
        nx = nxr/S; ny = nyr/S
        ia = i+g; ja = j+g
        if wallflux == Int32(1) && j == 1 && i > NIW && i <= NIW+NIF
            # analytic impermeable-wall flux + no-slip shear (adiabatic)
            pw = rho[ia, g+1]*CS2*T[ia, g+1] + (cav >= Int32(1) ? pexc[ia, g+1] : 0.0f0)   # PDP: wall carries p_mix (plateau)
            direct_p==Int32(1) && (pw=pmix[ia,g+1])
            Fr = zero(nx); Fmx = pw*nxr; Fmy = pw*nyr; Fe = zero(nx)
            if half!==nothing
                # No kinetic transport through the impermeable wall. The clock keeps the
                # same-state kinetic rate of the wall cell (acoustic reflection allowance).
                _,_,_,_,_,_,rt=_fvflux_kfvs_pc(
                    rho[ia,g+1],ux[ia,g+1],uy[ia,g+1],pmix[ia,g+1],am[ia,g+1],
                    rho[ia,g+1],ux[ia,g+1],uy[ia,g+1],pmix[ia,g+1],am[ia,g+1],cxa,cya,wa,nx,ny,S)
                half[i,j,1]=0.0f0; half[i,j,2]=0.0f0; rate[i,j]=rt
            end
            if nu_lam > 0.0f0 && (direct_p==Int32(0) || wallmode!=Int32(1))
                d1 = 0.5f0*vol[i,1]/S + 1.0f-30
                if wallfun > 1.5f0
                    # ★ SPECIES ALL-y+ WALL FUNCTION (26 Sep 2026; s.wallfun == 2, admitted only on the species
                    # path, cmesh_cav_species.jl `_admit`). Reichardt's profile gives the wall shear from the
                    # first cell's TANGENTIAL speed at every y+ (cmesh_wallfun.jl `wf_species_state`), so there is
                    # no law switch; it replaces the whole resolved-gradient traction, applied against the
                    # tangential velocity. The SST side (zero k flux, wall-law strain, omega pinned to Menter's
                    # blend) lives in cmesh_species_sst.jl; the loads and the viscous clock use the same law.
                    uw = ux[ia, g+1]; vw = uy[ia, g+1]
                    un = uw*nx + vw*ny
                    utx = uw - un*nx; uty = vw - un*ny
                    ut = sqrt(utx*utx + uty*uty)
                    if ut > 0.0f0
                        utau, _, _ = wf_species_state(ut, d1, nu_lam)
                        tauw = rho[ia, g+1]*utau*utau
                        Fmx -= tauw*S*utx/ut
                        Fmy -= tauw*S*uty/ut
                    end
                elseif wallfun > 0.0f0
                    # ★ S3i EQUILIBRIUM LOG-LAW WALL FUNCTION. Replaces the RESOLVED-GRADIENT wall
                    # shear rather than adding to it -- cmesh is a finite-volume scheme, so a wall
                    # function here is a substitution in the viscous wall flux, NOT the body-force
                    # injection the sibling LBM path uses. Getting that wrong would double-count.
                    # ⚠ VALID ONLY AT y+ 30-100. At the present dn1=2e-4 (y+ ~ 8) the log law is
                    # invalid and this will misbehave -- which is what the four archived divergences
                    # actually were. The run must coarsen the wall spacing too.
                    uw = ux[ia, g+1]; vw = uy[ia, g+1]
                    umag = sqrt(uw*uw + vw*vw) + 1.0f-30
                    rw = rho[ia, g+1]
                    nuw = nu_lam*_mufac(T[ia, g+1], suth)
                    # streamwise mixture-pressure gradient along the wall row, for the Larsson term
                    dpw = 0.0f0
                    if wf_larsson > 0.0f0 && i > 1 && i < NI
                        pm = rho[ia-1,g+1]*CS2*T[ia-1,g+1] + (cav >= Int32(1) ? pexc[ia-1,g+1] : 0.0f0)
                        pp = rho[ia+1,g+1]*CS2*T[ia+1,g+1] + (cav >= Int32(1) ? pexc[ia+1,g+1] : 0.0f0)
                        dxw = xc[i+1,1] - xc[i-1,1]
                        dpw = abs(dxw) > 1.0f-12 ? (pp - pm)/dxw : 0.0f0
                    end
                    utau = wf_utau(umag, d1, nuw, dpw, rw, wf_larsson)
                    tauw = rw*utau*utau
                    Fmx -= tauw*S*uw/umag
                    Fmy -= tauw*S*vw/umag
                else
                muw = (air_sutherland != Int32(0) ?
                       rho_inf*nu_lam*_mufac(T[ia,g+1],suth) :
                       rho[ia,g+1]*nu_lam*_mufac(T[ia,g+1],suth))*S/d1
                if direct_p==Int32(1) || (cav==Int32(0) && fullvisc==Int32(1))
                    tx,ty=_wall_stokes_traction(ux[ia,g+1],uy[ia,g+1],nx,ny,muw)
                    Fmx -= tx; Fmy -= ty
                else
                Fmx -= muw*ux[ia, g+1]
                Fmy -= muw*uy[ia, g+1]
                end
                end
            end
            Fj_r[i,j]=Fr; Fj_mx[i,j]=Fmx; Fj_my[i,j]=Fmy; Fj_e[i,j]=Fe
        else
            local Fr, Fmx, Fmy, Fe
            if air_geometry!==nothing
                if j==1
                    air_partner=NI+Int32(1)-i
                    air_left=(i<=NIW||i>NIW+NIF) ? _airfoil_trace4(face_states,air_partner,1,3) :
                        _airfoil_air_z(rho,ux,uy,T,ia,ja-1)
                    air_right=_airfoil_trace4(face_states,i,1,3)
                else
                    air_left=_airfoil_trace4(face_states,i,j-1,4)
                    air_right=j==NJ+1 ? _airfoil_air_z(rho,ux,uy,T,ia,ja) : _airfoil_trace4(face_states,i,j,3)
                end
                Fr,Fmx,Fmy,Fe=_airfoil_thermal_face_flux(air_left,air_right,nx,ny,S,cxa,cya,cza,wa,drot)
            elseif face_states!==nothing && 2<=j<=NJ
                l=ntuple(k->face_states[i,j-1,k,4],Val(7))
                r=ntuple(k->face_states[i,j,k,3],Val(7))
                if half!==nothing
                    Fr,Fmx,Fmy,Fe,mp,mm,rt=_fvflux_kfvs_pc(l[1],l[2],l[3],l[5],l[6],
                        r[1],r[2],r[3],r[5],r[6],cxa,cya,wa,nx,ny,S)
                    half[i,j,1]=mp; half[i,j,2]=mm; rate[i,j]=rt
                else
                    Fr,Fmx,Fmy,Fe=_fvflux_mix_pc(l[1],l[2],l[3],l[4],l[5],
                        r[1],r[2],r[3],r[4],r[5],l[6],r[6],eint,minf2,nx,ny,S)
                end
            elseif cav >= Int32(4)
                # ★ cav level 4 (2026-09-03): direct-pressure MIXTURE face (AUSM+-up, or the kinetic
                # KFVS face when `half` is supplied); see _k_flux_i! and _fvflux_mix6/_fvflux_kfvs_pc.
                if pc==Int32(1) && direct_p==Int32(1)
                    if half!==nothing
                        Fr,Fmx,Fmy,Fe,mp,mm,rt=_fvflux_kfvs_pc(
                            rho[ia,ja-1],ux[ia,ja-1],uy[ia,ja-1],pmix[ia,ja-1],am[ia,ja-1],
                            rho[ia,ja],ux[ia,ja],uy[ia,ja],pmix[ia,ja],am[ia,ja],cxa,cya,wa,nx,ny,S)
                        half[i,j,1]=mp; half[i,j,2]=mm; rate[i,j]=rt
                    else
                        Fr,Fmx,Fmy,Fe=_fvflux_mix_pc(
                            rho[ia,ja-1],ux[ia,ja-1],uy[ia,ja-1],T[ia,ja-1],pmix[ia,ja-1],
                            rho[ia,ja],ux[ia,ja],uy[ia,ja],T[ia,ja],pmix[ia,ja],
                            am[ia,ja-1],am[ia,ja],eint,minf2,nx,ny,S)
                    end
                elseif pc == Int32(1)
                    pL = direct_p==Int32(1) ? pmix[ia,ja-1] : rho[ia,ja-1]*CS2*T[ia,ja-1]+pexc[ia,ja-1]
                    pR = direct_p==Int32(1) ? pmix[ia,ja] : rho[ia,ja]*CS2*T[ia,ja]+pexc[ia,ja]
                    Fr, Fmx, Fmy, Fe = _fvflux_mix6(
                    rho[ia,ja-1],ux[ia,ja-1],uy[ia,ja-1],T[ia,ja-1], pL,
                    rho[ia,ja-1],ux[ia,ja-1],uy[ia,ja-1],T[ia,ja-1], pL,
                    rho[ia,ja-1],ux[ia,ja-1],uy[ia,ja-1],T[ia,ja-1], pL,
                    rho[ia,ja  ],ux[ia,ja  ],uy[ia,ja  ],T[ia,ja  ], pR,
                    rho[ia,ja  ],ux[ia,ja  ],uy[ia,ja  ],T[ia,ja  ], pR,
                    rho[ia,ja  ],ux[ia,ja  ],uy[ia,ja  ],T[ia,ja  ], pR,
                    am[ia,ja-1], am[ia,ja], eint, minf2, nx, ny, S, Int32(0))
                else
                Fr, Fmx, Fmy, Fe = _fvflux_mix6(
                    rho[ia,ja-3],ux[ia,ja-3],uy[ia,ja-3],T[ia,ja-3], rho[ia,ja-3]*CS2*T[ia,ja-3]+pexc[ia,ja-3],
                    rho[ia,ja-2],ux[ia,ja-2],uy[ia,ja-2],T[ia,ja-2], rho[ia,ja-2]*CS2*T[ia,ja-2]+pexc[ia,ja-2],
                    rho[ia,ja-1],ux[ia,ja-1],uy[ia,ja-1],T[ia,ja-1], rho[ia,ja-1]*CS2*T[ia,ja-1]+pexc[ia,ja-1],
                    rho[ia,ja  ],ux[ia,ja  ],uy[ia,ja  ],T[ia,ja  ], rho[ia,ja  ]*CS2*T[ia,ja  ]+pexc[ia,ja  ],
                    rho[ia,ja+1],ux[ia,ja+1],uy[ia,ja+1],T[ia,ja+1], rho[ia,ja+1]*CS2*T[ia,ja+1]+pexc[ia,ja+1],
                    rho[ia,ja+2],ux[ia,ja+2],uy[ia,ja+2],T[ia,ja+2], rho[ia,ja+2]*CS2*T[ia,ja+2]+pexc[ia,ja+2],
                    am[ia,ja-1], am[ia,ja], eint, minf2, nx, ny, S, recon)
                end
            else
                Fr, Fmx, Fmy, Fe = _fvflux_normal6(
                    rho[ia,ja-3],ux[ia,ja-3],uy[ia,ja-3],T[ia,ja-3],
                    rho[ia,ja-2],ux[ia,ja-2],uy[ia,ja-2],T[ia,ja-2],
                    rho[ia,ja-1],ux[ia,ja-1],uy[ia,ja-1],T[ia,ja-1],
                    rho[ia,ja  ],ux[ia,ja  ],uy[ia,ja  ],T[ia,ja  ],
                    rho[ia,ja+1],ux[ia,ja+1],uy[ia,ja+1],T[ia,ja+1],
                    rho[ia,ja+2],ux[ia,ja+2],uy[ia,ja+2],T[ia,ja+2],
                    nx, ny, S, cxa, cya, cza, wa, drot, recon)
            end
            # Strip topology (5 Oct 2026, plan section 4.1): with a strip pair's `ctx` the j = 1 faces of the C block's leg
            # columns and of the strip are interfaces whose other cell lies in the other block (its centroid from the
            # static neighbour table ctx.c_nb), and the strip's j = NJ+1 row is the upper interface (top_iface). Both
            # interfaces are thin-layer on both sides (critique A 1.1): the full-stress and EARSM branches below are gated
            # off the strip's top row as they are off j = 1. ctx === nothing: every test below is today's.
            top_iface = ctx!==nothing && ctx.block==Int32(1) && j==NJ+Int32(1)
            if nu_lam > 0.0f0 && ((j >= 2 && j <= NJ) || (j == 1 && (i <= NIW || i > NIW+NIF)) || top_iface)
                # j=1 wake columns = the WAKE-CUT faces: previously zero viscous flux — the wake sheet
                # could kink at the cut = a TE acoustic dipole radiating upstream ripples (user report)
                local dxc, dyc, dn
                if air_geometry!==nothing
                    dxc=-air_geometry[i,j,1,3];dyc=-air_geometry[i,j,2,3]
                    dn=sqrt(dxc*dxc+dyc*dyc)+1f-30
                elseif j == 1
                    # The two representations of a wake-cut face need the same
                    # viscous conductance even if their first-row volumes differ.
                    # This also applies to the ideal-gas (direct_p=0) path.
                    if ctx===nothing
                        ip=NI+Int32(1)-i
                        dxc=xc[i,1]-xc[ip,1];dyc=yc[i,1]-yc[ip,1]
                    else
                        st,fc=_strip_station(i,NIW,NIF,ctx.block)
                        dxc=xc[i,1]-ctx.c_nb[st,1,fc];dyc=yc[i,1]-ctx.c_nb[st,2,fc]
                    end
                    dn=sqrt(dxc*dxc+dyc*dyc)+1.0f-30
                elseif ctx!==nothing && top_iface
                    # the strip's upper interface: the cell above is the upper leg's row-1 cell (the ghost); the same
                    # difference (upper minus lower centroid) as the leg's own j = 1 face computes, operand for operand
                    dxc = ctx.c_nb[i,1,2]-xc[i,NJ]; dyc = ctx.c_nb[i,2,2]-yc[i,NJ]; dn = sqrt(dxc*dxc+dyc*dyc)+1.0f-30
                else
                    dxc = xc[i,j]-xc[i,j-1]; dyc = yc[i,j]-yc[i,j-1]; dn = sqrt(dxc*dxc+dyc*dyc)+1.0f-30
                end
                nut_f = sst_on == Int32(1) ? 0.5f0*(nut[ia,ja-1]+nut[ia,ja]) : 0.0f0
                rf = 0.5f0*(rho[ia,ja-1]+rho[ia,ja])
                fmu = _mufac(0.5f0*(T[ia,ja-1]+T[ia,ja]), suth)   # Sutherland mu(T)
                cpg = 0.5f0*(5.0f0+drot)*CS2
                mu_eff = air_sutherland != Int32(0) ?
                         rho_inf*nu_lam*fmu + rf*nut_f : rf*(nu_lam*fmu+nut_f)
                k_eff = air_sutherland != Int32(0) ?
                        cpg*(rho_inf*nu_lam*fmu/0.71f0 + rf*nut_f/0.9f0) :
                        rf*cpg*(nu_lam*fmu/0.71f0 + nut_f/0.9f0)
                if fullvisc == Int32(1) && j > 1 && !top_iface      # A2c full stress tensor (cut face and strip top: thin-layer)
                    Fvx, Fvy, Fve = _viscous_face_full(
                        gux[i,j-1],guy[i,j-1],gvx[i,j-1],gvy[i,j-1],gTx[i,j-1],gTy[i,j-1],
                        gux[i,j],  guy[i,j],  gvx[i,j],  gvy[i,j],  gTx[i,j],  gTy[i,j],
                        ux[ia,ja-1],uy[ia,ja-1],T[ia,ja-1], ux[ia,ja],uy[ia,ja],T[ia,ja],
                        dxc, dyc, nx, ny, S, mu_eff,
                        direct_p==Int32(1) ? 0.0f0 : k_eff)
                    Fmx -= Fvx; Fmy -= Fvy; Fe -= Fve
                else                                  # thin-layer (legacy bitwise)
                    mu = mu_eff*S/dn
                    du = ux[ia,ja]-ux[ia,ja-1]; dv = uy[ia,ja]-uy[ia,ja-1]; dT = T[ia,ja]-T[ia,ja-1]
                    ufx = 0.5f0*(ux[ia,ja-1]+ux[ia,ja]); ufy = 0.5f0*(uy[ia,ja-1]+uy[ia,ja])
                    Fmx -= mu*du; Fmy -= mu*dv
                    Fe  -= mu*(ufx*du+ufy*dv) + (direct_p==Int32(1) ? 0.0f0 : k_eff*S/dn*dT)
                end
                if rhok == Int32(1) && sst_on == Int32(1) # -2/3 rho k in the turbulent stress
                    kf = 0.5f0*(tk[ia,ja-1]+tk[ia,ja])
                    ufx2 = 0.5f0*(ux[ia,ja-1]+ux[ia,ja]); ufy2 = 0.5f0*(uy[ia,ja-1]+uy[ia,ja])
                    Fmx += 0.6666667f0*rf*kf*nxr
                    Fmy += 0.6666667f0*rf*kf*nyr
                    Fe  += 0.6666667f0*rf*kf*(ufx2*nxr + ufy2*nyr)
                end
                if earsm == Int32(1) && sst_on == Int32(1) && j > 1 && !top_iface  # EARSM extra stress (interior j-faces)
                    e11 = 0.5f0*(exx[i,j-1]+exx[i,j]); e12 = 0.5f0*(exy[i,j-1]+exy[i,j])
                    e22 = 0.5f0*(eyy[i,j-1]+eyy[i,j])
                    fex = e11*nxr + e12*nyr; fey = e12*nxr + e22*nyr
                    ufx3 = 0.5f0*(ux[ia,ja-1]+ux[ia,ja]); ufy3 = 0.5f0*(uy[ia,ja-1]+uy[ia,ja])
                    Fmx -= fex; Fmy -= fey
                    Fe  -= ufx3*fex + ufy3*fey
                end
            end
            if cav >= Int32(1) && cav < Int32(4)          # PDP pressure-split: excess p_mix as a face force (cav 1-3)
                pexf = 0.5f0*(pexc[ia,ja-1] + pexc[ia,ja])
                Fmx += pexf*nxr; Fmy += pexf*nyr
                ufx = 0.5f0*(ux[ia,ja-1]+ux[ia,ja]); ufy = 0.5f0*(uy[ia,ja-1]+uy[ia,ja])
                Fe  += pexf*(ufx*nxr + ufy*nyr)
            end
            if cav >= Int32(2) && cav < Int32(4)          # Layer C: PRECONDITIONED a_m AUSM+-up (anti-checkerboard + shock speed) (cav 2-3)
                pL = rho[ia,ja-1]*CS2*T[ia,ja-1] + pexc[ia,ja-1]
                pR = rho[ia,ja  ]*CS2*T[ia,ja  ] + pexc[ia,ja  ]
                unL = ux[ia,ja-1]*nx + uy[ia,ja-1]*ny; unR = ux[ia,ja]*nx + uy[ia,ja]*ny
                amf = 0.5f0*(am[ia,ja-1] + am[ia,ja])
                amp = max(amf, 0.5f0*abs(unL+unR), 0.05f0)
                amp_u = amp                                   # ★ NEGCOMP (cav>=3): floor K_u speed at the
                if cav >= Int32(3)                            # pexc anti-diffusion sqrt(|a2m - cs2*T_band|)
                    Tf = 0.5f0*(T[ia,ja-1] + T[ia,ja])        # (railed core only; spares closure). See flux_i.
                    amp_u = max(amp, sqrt(max(0.0f0, CS2*Tf - amf*amf)))
                end
                Fr  += -0.25f0*(pR - pL)/amp*S
                rf   = 0.5f0*(rho[ia,ja-1]+rho[ia,ja])
                pu   = -0.5f0*rf*amp_u*(unR - unL)
                Fmx += pu*nxr; Fmy += pu*nyr
            end
            Fj_r[i,j]=Fr; Fj_mx[i,j]=Fmx; Fj_my[i,j]=Fmy; Fj_e[i,j]=Fe
        end
    end
end

# Exact inviscid slip-wall flux on segmented outer-wall faces. This runs after the general j-face
# flux so neither reconstruction, numerical dissipation, nor a viscous term can leak mass/energy
# through the channel wall.
@kernel function _k_flux_outer_slip!(Fj_r, Fj_mx, Fj_my, Fj_e,
                                     @Const(rho), @Const(T), @Const(pexc),
                                     @Const(Sjx), @Const(Sjy), @Const(outer_face_kind),
                                     cav::Int32, @Const(pmix), direct_p::Int32, half,
                                     NI::Int32, NJ::Int32, g::Int32)
    CS2=_gas_cs2(eltype(rho))
    i = @index(Global, Linear)
    @inbounds if i <= NI && outer_face_kind[i] >= 0x02
        ia = i + g; ja = NJ + g
        pwall = rho[ia, ja]*CS2*T[ia, ja] + (cav >= Int32(1) ? pexc[ia, ja] : 0.0f0)
        direct_p==Int32(1) && (pwall=pmix[ia,ja])
        Fj_r[i, NJ+Int32(1)] = 0.0f0
        Fj_mx[i, NJ+Int32(1)] = pwall*Sjx[i, NJ+Int32(1)]
        Fj_my[i, NJ+Int32(1)] = pwall*Sjy[i, NJ+Int32(1)]
        Fj_e[i, NJ+Int32(1)] = 0.0f0
        if half!==nothing   # kinetic half masses follow the exact zero wall mass flux
            half[i, NJ+Int32(1), 1] = 0.0f0; half[i, NJ+Int32(1), 2] = 0.0f0
        end
    end
end

"""Apply only the exact segmented outer slip-wall flux override (a no-op in legacy mode)."""
function outer_flux_override_ka!(s::KSol;mixture_pressure=nothing,half=nothing)
    s.outer_mode == Int32(0) && return nothing
    # The strip carrier (outer_mode 2, strip topology): its j = NJ+1 faces are the upper interface, never a slip wall,
    # although their kind 4 satisfies the kernel's `>= 0x02` test; it has no outer boundary to override (plan 4.1).
    s.outer_mode == Int32(2) && return nothing
    b = s.backend; NI, NJ, g = Int32(s.NI), Int32(s.NJ), Int32(s.g)
    _k_flux_outer_slip!(b)(s.Fj_r, s.Fj_mx, s.Fj_my, s.Fj_e,
                           s.rho, s.T, s.pexc, s.Sjx, s.Sjy, s.outer_face_kind,
                           s.cav, mixture_pressure===nothing ? s.pexc : mixture_pressure,
                           Int32(mixture_pressure!==nothing), half, NI, NJ, g; ndrange=s.NI)
    nothing
end

@kernel function _k_residual!(dc_r, dc_mx, dc_my, dc_e,
                              @Const(Fi_r), @Const(Fi_mx), @Const(Fi_my), @Const(Fi_e),
                              @Const(Fj_r), @Const(Fj_mx), @Const(Fj_my), @Const(Fj_e),
                              NI::Int32, NJ::Int32)
    i, j = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ
        dc_r[i,j]  = Fi_r[i+1,j]-Fi_r[i,j]  + Fj_r[i,j+1]-Fj_r[i,j]
        dc_mx[i,j] = Fi_mx[i+1,j]-Fi_mx[i,j] + Fj_mx[i,j+1]-Fj_mx[i,j]
        dc_my[i,j] = Fi_my[i+1,j]-Fi_my[i,j] + Fj_my[i,j+1]-Fj_my[i,j]
        dc_e[i,j]  = Fi_e[i+1,j]-Fi_e[i,j]  + Fj_e[i,j+1]-Fj_e[i,j]
    end
end

@kernel function _k_subtract_channel_reference!(dc_r, dc_mx, dc_my, dc_e,
                                                @Const(ref_r), @Const(ref_mx),
                                                @Const(ref_my), @Const(ref_e),
                                                NI::Int32, NJ::Int32)
    i, j = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ
        dc_r[i,j] -= ref_r[i,j]
        dc_mx[i,j] -= ref_mx[i,j]
        dc_my[i,j] -= ref_my[i,j]
        dc_e[i,j] -= ref_e[i,j]
    end
end

@kernel function _k_dtfield!(dtf, @Const(rho), @Const(ux), @Const(uy), @Const(T), @Const(nut),
                             @Const(Six), @Const(Siy), @Const(Sjx), @Const(Sjy), @Const(vol),
                             cfl, nu_lam, rho_inf,
                             suth, air_sutherland::Int32,
                             sst_on::Int32, irsj::Int32,
                             @Const(am), cav::Int32, cav_cfl, am_ref,
                             NI::Int32, NJ::Int32, g::Int32)
    i, j = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ
        ia=i+g; ja=j+g
        u = ux[ia,ja]; v = uy[ia,ja]
        # ★★ STAGE 1c — THE MISSING HALF OF THE PRECONDITIONING (2026-07-20).
        #
        # Layer C (cav>=2) makes the FLUX propagate information at the PRECONDITIONED speed
        #     amp = max(a_m, |u|, 0.05)                      (_k_flux_i!/_k_flux_j!)
        # but this kernel sized the pseudo-time step on the UNPRECONDITIONED CARRIER c = 3*sqrt(T) = 3.29.
        # The effective CFL is then  CFL * amp/(|u| + 3.29), which VARIES ACROSS THE DOMAIN:
        #     liquid  amp ~ 0.92  =>  CFL_eff ~ 0.26 * CFL
        #     cavity  amp ~ 0.05  =>  CFL_eff ~ 0.015 * CFL      (a_m collapses to 0.010)
        # i.e. THE CAVITY MARCHES ~17x SLOWER IN PSEUDO-TIME THAN THE LIQUID. Measured consequence: the
        # steady cavitating solve plateaus at res_L2 ~ 7 and never converges even at 1,200,000 iterations,
        # with the imbalance concentrated exactly where the cavity is, while the clean foil on the same
        # mesh converges to 3.46e-06. Reaching liquid-level convergence in the cavity would need ~15M
        # iterations. This was previously filed as a PERFORMANCE item; it is not -- it is a
        # convergence-RATE EQUALISER, and the reason the cavitating runs never converged.
        #
        # Kunz et al. (2000), Computers & Fluids 29(8): the preconditioned eigensystem must be
        # "independent of density ratio and local volume fraction". Sizing dt on the carrier while the
        # flux uses a_m violates exactly that.
        #
        # ★★★ TESTED 2026-07-20 AND REFUTED — DO NOT RE-TRY THIS FORM.
        # Replacing the carrier speed with max(a_m, |u|, 0.05) gave IMMEDIATE NaN (every trace point,
        # from n=25000 on). The reason kills the whole argument above: a_m = 0.92 in the liquid vs a
        # carrier of 3.29, so dt becomes 3.6x LARGER there -- a 3.6x CFL increase, hence blow-up.
        # THE CARRIER SPEED IS NOT MERELY CONSERVATIVE, IT IS THE ACTUAL STABILITY LIMIT: the base scheme
        # is a KINETIC/KFVS quadrature on FIXED integer abscissae (|c_x| <= 3; the pedestal T enters only
        # the Hermite terms of _feqT2 and does NOT scale the particle velocities -- comment corrected
        # 2026-09-10; the 3*sqrt(T) below is the historical operational estimate, kept bitwise), so
        # information really does propagate at particle speed. Layer C's a_m enters only as an ADDED dissipation term
        # (K_p/K_u in _k_flux_i!/_k_flux_j!), not as the transport speed.
        # So dt CANNOT simply be based on a_m. If the effective-CFL disparity across the 90x sound-speed
        # range is genuinely the convergence problem, the fix is a real preconditioner (rescaling the
        # pseudo-time DERIVATIVE, Kunz/Venkateswaran style), NOT a swapped wave speed in the dt formula.
        c = 3.0f0*sqrt(max(T[ia,ja],1.0f-3))
        SiE = sqrt(Six[i+1,j]^2 + Siy[i+1,j]^2); SiW = sqrt(Six[i,j]^2 + Siy[i,j]^2)
        SjN = sqrt(Sjx[i,j+1]^2 + Sjy[i,j+1]^2); SjS = sqrt(Sjx[i,j]^2 + Sjy[i,j]^2)
        lam_i = (abs(u*Six[i,j]+v*Siy[i,j]) + abs(u*Six[i+1,j]+v*Siy[i+1,j]))*0.5f0 + c*0.5f0*(SiE+SiW)
        lam_j = (abs(u*Sjx[i,j]+v*Sjy[i,j]) + abs(u*Sjx[i,j+1]+v*Sjy[i,j+1]))*0.5f0 + c*0.5f0*(SjN+SjS)
        nueff = _air_nu_local(rho[ia,ja],T[ia,ja],nu_lam,rho_inf,suth,air_sutherland) +
                (sst_on == Int32(1) ? nut[ia,ja] : 0.0f0)
        local dt
        if irsj == Int32(1)
            dt = cfl*vol[i,j]/max(lam_i, 1.0f-30)   # streamwise restriction only (j implicit)
            if nueff > 0.0f0
                dn_i = vol[i,j]/max(SiE, SiW)
                dt = min(dt, 0.2f0*dn_i*dn_i/nueff)
            end
        else
            dt = cfl*vol[i,j]/max(lam_i+lam_j, 1.0f-30)
            if nueff > 0.0f0
                dn_min = vol[i,j]/max(max(SiE,SiW), max(SjN,SjS))
                dt = min(dt, 0.2f0*dn_min*dn_min/nueff)
            end
        end
        # ★ VOID-ADAPTIVE reduction (2026-07-25): shrink the explicit step where a_m collapses (the
        # cavity), toward the floor cav_cfl, using a_m/a_ref as the smooth cavity indicator (1 in the
        # liquid => bitwise there; -> cav_cfl in the deep cavity). Only the pseudo-time step is reduced;
        # the physical BDF2 fixed point is untouched. cav_cfl<=0 => disabled (bitwise legacy).
        if cav_cfl > 0.0f0
            amr = am[i+g, j+g]/max(am_ref, 1.0f-30)
            fac = amr < cav_cfl ? cav_cfl : (amr > 1.0f0 ? 1.0f0 : amr)
            dt *= fac
        end
        dtf[i,j] = dt
    end
end

# IRS-J batched tridiagonal (mirrors the plain _irsj_component! line-for-line): one thread per
# i-column, sequential j sweeps; aN per row in irs_d, Thomas b-prime in irs_c, d-prime in dc.
@kernel function _k_irsj!(dc, bp, an, @Const(ux), @Const(uy), @Const(T), @Const(nut),
                          @Const(Sjx), @Const(Sjy), @Const(vol), @Const(dtf),
                          nu_lam, sst_on::Int32, @Const(am), irs_ac::Int32,
                          am_ref, irs_eps0, NI::Int32, NJ::Int32, g::Int32)
    i = @index(Global, Linear)
    @inbounds if i <= NI
        ia = i + g
        for j in 1:NJ
            if j == NJ
                an[i,j] = 0.0f0
            else
                Sx = Sjx[i,j+1]; Sy = Sjy[i,j+1]; S2f = Sx*Sx + Sy*Sy
                u = ux[ia,j+g]; v = uy[ia,j+g]; c = 3.0f0*sqrt(max(T[ia,j+g],1.0f-3))
                nueff = nu_lam + (sst_on == Int32(1) ? nut[ia,j+g] : 0.0f0)
                lam = abs(u*Sx + v*Sy) + c*sqrt(S2f) + 2.0f0*nueff*S2f/vol[i,j]
                nuJ = dtf[i,j]*lam/(vol[i,j]*0.25f0)
                # ε = max(CFL-anisotropy ε, VOID-targeted ε). The void term fires where a_m collapses (the
                # cavity), injecting implicit smoothing on the pexc grid-scale instability. irs_ac==0 =>
                # evoid=0 => bitwise legacy. Fixed-point-preserving (residual smoothing).
                evoid = irs_ac == Int32(1) ? irs_eps0*max(0.0f0, 1.0f0 - am[ia,j+g]/am_ref) : 0.0f0
                an[i,j] = max(0.25f0*max(0.0f0, nuJ*nuJ - 1.0f0), evoid)
            end
        end
        # boundary faces: eps in the DIAGONAL, no off-diagonal (Dirichlet-0 residual ghost) — mirrors plain
        eps1 = begin
            Sx = Sjx[i,1]; Sy = Sjy[i,1]; S2f = Sx*Sx + Sy*Sy
            u = ux[ia,1+g]; v = uy[ia,1+g]; c = 3.0f0*sqrt(max(T[ia,1+g],1.0f-3))
            nueff = nu_lam + (sst_on == Int32(1) ? nut[ia,1+g] : 0.0f0)
            lam = abs(u*Sx + v*Sy) + c*sqrt(S2f) + 2.0f0*nueff*S2f/vol[i,1]
            nuJ = dtf[i,1]*lam/(vol[i,1]*0.25f0)
            evoid = irs_ac == Int32(1) ? irs_eps0*max(0.0f0, 1.0f0 - am[ia,1+g]/am_ref) : 0.0f0
            max(0.25f0*max(0.0f0, nuJ*nuJ - 1.0f0), evoid)
        end
        epsN1 = begin
            Sx = Sjx[i,NJ+1]; Sy = Sjy[i,NJ+1]; S2f = Sx*Sx + Sy*Sy
            u = ux[ia,NJ+g]; v = uy[ia,NJ+g]; c = 3.0f0*sqrt(max(T[ia,NJ+g],1.0f-3))
            nueff = nu_lam + (sst_on == Int32(1) ? nut[ia,NJ+g] : 0.0f0)
            lam = abs(u*Sx + v*Sy) + c*sqrt(S2f) + 2.0f0*nueff*S2f/vol[i,NJ]
            nuJ = dtf[i,NJ]*lam/(vol[i,NJ]*0.25f0)
            evoid = irs_ac == Int32(1) ? irs_eps0*max(0.0f0, 1.0f0 - am[ia,NJ+g]/am_ref) : 0.0f0
            max(0.25f0*max(0.0f0, nuJ*nuJ - 1.0f0), evoid)
        end
        bp[i,1] = 1.0f0 + eps1 + an[i,1]
        for j in 2:NJ
            aS = an[i,j-1]; aNd = j == NJ ? epsN1 : an[i,j]
            w = (-aS)/bp[i,j-1]
            bp[i,j] = (1.0f0 + aS + aNd) - w*(-aS)
            dc[i,j] = dc[i,j] - w*dc[i,j-1]
        end
        dc[i,NJ] = dc[i,NJ]/bp[i,NJ]
        for j in NJ-1:-1:1
            dc[i,j] = (dc[i,j] + an[i,j]*dc[i,j+1])/bp[i,j]
        end
    end
end

# IRS-I batched tridiagonal (i-direction mirror of _k_irsj!): one thread per j-row, sequential i
# sweeps; aE per row in `an`, Thomas b-prime in `bp`, d-prime in dc. Acoustic-aware λ (c=max(3sqrtT,a_m)
# when irs_ac==1). Same variable-coefficient IRS (ε=0.25 max(0, ν²-1)) as the J version, streamwise.
@kernel function _k_irsi!(dc, bp, an, @Const(ux), @Const(uy), @Const(T), @Const(nut),
                          @Const(Six), @Const(Siy), @Const(vol), @Const(dtf),
                          nu_lam, sst_on::Int32, @Const(am), irs_ac::Int32,
                          am_ref, irs_eps0, NI::Int32, NJ::Int32, g::Int32)
    j = @index(Global, Linear)
    @inbounds if j <= NJ
        ja = j + g
        for i in 1:NI
            if i == NI
                an[i,j] = 0.0f0
            else
                Sx = Six[i+1,j]; Sy = Siy[i+1,j]; S2f = Sx*Sx + Sy*Sy
                u = ux[i+g,ja]; v = uy[i+g,ja]; c = 3.0f0*sqrt(max(T[i+g,ja],1.0f-3))
                nueff = nu_lam + (sst_on == Int32(1) ? nut[i+g,ja] : 0.0f0)
                lam = abs(u*Sx + v*Sy) + c*sqrt(S2f) + 2.0f0*nueff*S2f/vol[i,j]
                nuI = dtf[i,j]*lam/(vol[i,j]*0.25f0)
                evoid = irs_ac == Int32(1) ? irs_eps0*max(0.0f0, 1.0f0 - am[i+g,ja]/am_ref) : 0.0f0
                an[i,j] = max(0.25f0*max(0.0f0, nuI*nuI - 1.0f0), evoid)
            end
        end
        eps1 = begin
            Sx = Six[1,j]; Sy = Siy[1,j]; S2f = Sx*Sx + Sy*Sy
            u = ux[1+g,ja]; v = uy[1+g,ja]; c = 3.0f0*sqrt(max(T[1+g,ja],1.0f-3))
            nueff = nu_lam + (sst_on == Int32(1) ? nut[1+g,ja] : 0.0f0)
            lam = abs(u*Sx + v*Sy) + c*sqrt(S2f) + 2.0f0*nueff*S2f/vol[1,j]
            nuI = dtf[1,j]*lam/(vol[1,j]*0.25f0)
            evoid = irs_ac == Int32(1) ? irs_eps0*max(0.0f0, 1.0f0 - am[1+g,ja]/am_ref) : 0.0f0
            max(0.25f0*max(0.0f0, nuI*nuI - 1.0f0), evoid)
        end
        epsN1 = begin
            Sx = Six[NI+1,j]; Sy = Siy[NI+1,j]; S2f = Sx*Sx + Sy*Sy
            u = ux[NI+g,ja]; v = uy[NI+g,ja]; c = 3.0f0*sqrt(max(T[NI+g,ja],1.0f-3))
            nueff = nu_lam + (sst_on == Int32(1) ? nut[NI+g,ja] : 0.0f0)
            lam = abs(u*Sx + v*Sy) + c*sqrt(S2f) + 2.0f0*nueff*S2f/vol[NI,j]
            nuI = dtf[NI,j]*lam/(vol[NI,j]*0.25f0)
            evoid = irs_ac == Int32(1) ? irs_eps0*max(0.0f0, 1.0f0 - am[NI+g,ja]/am_ref) : 0.0f0
            max(0.25f0*max(0.0f0, nuI*nuI - 1.0f0), evoid)
        end
        bp[1,j] = 1.0f0 + eps1 + an[1,j]
        for i in 2:NI
            aW = an[i-1,j]; aEd = i == NI ? epsN1 : an[i,j]
            w = (-aW)/bp[i-1,j]
            bp[i,j] = (1.0f0 + aW + aEd) - w*(-aW)
            dc[i,j] = dc[i,j] - w*dc[i-1,j]
        end
        dc[NI,j] = dc[NI,j]/bp[NI,j]
        for i in NI-1:-1:1
            dc[i,j] = (dc[i,j] + an[i,j]*dc[i+1,j])/bp[i,j]
        end
    end
end

# ★ ACOUSTIC FILTER kernels (2026-07-25): ADI line-implicit low-pass of the pseudo-time DENSITY INCREMENT
# delta = rho - rk_r, curing the split-pexc negative-compressibility divergence (see the KSol acou field).
# Off-diagonal beta_f = kfac*dtl^2*Kf*|S_f|^2/vol^2 with Kf = (both cells railed) ? max(0, cs2*Tf - a2mf) : 0
# (a2m = am^2). Thomas is the PURE Neumann Helmholtz (I+M), M a negative-Laplacian with ZERO row-sums (NO
# boundary diagonal term) => mass-conserving; the both-railed gate zeros the off-diagonal across the
# vapour-liquid interface (no cross-interface smoothing). Reconstruct rho = rk_r + delta_filt (floor 1e-4),
# hold momentum invariant (u,v *= rho_pre/rho_new). i-sweep then j-sweep (ADI). CS2 = 2/3 = TEM_CS2 (verified).
@kernel function _k_acou_i!(rho, ux, uy, @Const(rk_r), @Const(am), @Const(T),
                            @Const(Six), @Const(Siy), @Const(vol), @Const(dtf),
                            delta, aE, bp, kfac, NI::Int32, NJ::Int32, g::Int32)
    j = @index(Global, Linear)
    @inbounds if j <= NJ
        ja = j + g
        for i in 1:NI
            ia = i + g
            delta[i,j] = rho[ia,ja] - rk_r[ia,ja]                 # the RK3 density increment (->0 at convergence)
            if i < NI
                gapP = CS2*T[ia,ja]   - am[ia,ja]*am[ia,ja]       # cs2*T - a2m > 0 <=> railed negcomp core
                gapN = CS2*T[ia+1,ja] - am[ia+1,ja]*am[ia+1,ja]
                if gapP > 0.0f0 && gapN > 0.0f0
                    Tf   = 0.5f0*(T[ia,ja] + T[ia+1,ja])
                    a2mf = 0.5f0*(am[ia,ja]*am[ia,ja] + am[ia+1,ja]*am[ia+1,ja])
                    Kf   = CS2*Tf - a2mf; Kf = Kf > 0.0f0 ? Kf : 0.0f0
                    Sf2  = Six[i+1,j]^2 + Siy[i+1,j]^2
                    dtl  = 0.5f0*(dtf[i,j] + dtf[i+1,j])
                    aE[i,j] = kfac*dtl*dtl*Kf*Sf2/(vol[i,j]*vol[i,j])
                else
                    aE[i,j] = 0.0f0                               # interface/liquid: decouple (Neumann)
                end
            else
                aE[i,j] = 0.0f0
            end
        end
        bp[1,j] = 1.0f0 + aE[1,j]                                 # diag = 1 + aW + aE, NO boundary diagonal term
        for i in 2:NI
            aW = aE[i-1,j]; w = (-aW)/bp[i-1,j]
            bp[i,j] = (1.0f0 + aW + aE[i,j]) - w*(-aW)
            delta[i,j] = delta[i,j] - w*delta[i-1,j]
        end
        delta[NI,j] = delta[NI,j]/bp[NI,j]
        for i in NI-1:-1:1
            delta[i,j] = (delta[i,j] + aE[i,j]*delta[i+1,j])/bp[i,j]
        end
        for i in 1:NI
            ia = i + g
            r_pre = rho[ia,ja]
            r_raw = rk_r[ia,ja] + delta[i,j]
            r_new = r_raw < 1.0f-4 ? 1.0f-4 : r_raw
            sc = r_raw < 1.0f-4 ? 1.0f0 : r_pre/r_new           # hold rho*u invariant (sc=1 if floored: no u blowup)
            rho[ia,ja] = r_new; ux[ia,ja] *= sc; uy[ia,ja] *= sc
        end
    end
end

@kernel function _k_acou_j!(rho, ux, uy, @Const(rk_r), @Const(am), @Const(T),
                            @Const(Sjx), @Const(Sjy), @Const(vol), @Const(dtf),
                            delta, aN, bp, kfac, NI::Int32, NJ::Int32, g::Int32)
    i = @index(Global, Linear)
    @inbounds if i <= NI
        ia = i + g
        for j in 1:NJ
            ja = j + g
            delta[i,j] = rho[ia,ja] - rk_r[ia,ja]
            if j < NJ
                gapP = CS2*T[ia,ja]   - am[ia,ja]*am[ia,ja]
                gapN = CS2*T[ia,ja+1] - am[ia,ja+1]*am[ia,ja+1]
                if gapP > 0.0f0 && gapN > 0.0f0
                    Tf   = 0.5f0*(T[ia,ja] + T[ia,ja+1])
                    a2mf = 0.5f0*(am[ia,ja]*am[ia,ja] + am[ia,ja+1]*am[ia,ja+1])
                    Kf   = CS2*Tf - a2mf; Kf = Kf > 0.0f0 ? Kf : 0.0f0
                    Sf2  = Sjx[i,j+1]^2 + Sjy[i,j+1]^2
                    dtl  = 0.5f0*(dtf[i,j] + dtf[i,j+1])
                    aN[i,j] = kfac*dtl*dtl*Kf*Sf2/(vol[i,j]*vol[i,j])
                else
                    aN[i,j] = 0.0f0
                end
            else
                aN[i,j] = 0.0f0
            end
        end
        bp[i,1] = 1.0f0 + aN[i,1]
        for j in 2:NJ
            aS = aN[i,j-1]; w = (-aS)/bp[i,j-1]
            bp[i,j] = (1.0f0 + aS + aN[i,j]) - w*(-aS)
            delta[i,j] = delta[i,j] - w*delta[i,j-1]
        end
        delta[i,NJ] = delta[i,NJ]/bp[i,NJ]
        for j in NJ-1:-1:1
            delta[i,j] = (delta[i,j] + aN[i,j]*delta[i,j+1])/bp[i,j]
        end
        for j in 1:NJ
            ja = j + g
            r_pre = rho[ia,ja]
            r_raw = rk_r[ia,ja] + delta[i,j]
            r_new = r_raw < 1.0f-4 ? 1.0f-4 : r_raw
            sc = r_raw < 1.0f-4 ? 1.0f0 : r_pre/r_new
            rho[ia,ja] = r_new; ux[ia,ja] *= sc; uy[ia,ja] *= sc
        end
    end
end

# Driver: called BETWEEN step_ka! and tem_step!. acou==0 => bitwise legacy (never touches the field).
# Reuses irs_d (off-diagonal) + irs_c (Thomas b-prime) scratch — free when irsj==0 && irs_ac==0.
function acoustic_filter_ka!(s::KSol)
    if s.outer_mode != Int32(0) && s.acou != Int32(0)
        throw(ArgumentError("the acoustic filter is not defined for segmented channel boundaries"))
    end
    s.acou == Int32(0) && return nothing
    b = s.backend; NI, NJ, g = Int32(s.NI), Int32(s.NJ), Int32(s.g)
    _k_acou_i!(b)(s.rho, s.ux, s.uy, s.rk_r, s.am, s.T, s.Six, s.Siy, s.vol, s.dtf,
                  s.acou_d, s.irs_d, s.irs_c, s.acou_kfac, NI, NJ, g; ndrange=(s.NJ,))
    _k_acou_j!(b)(s.rho, s.ux, s.uy, s.rk_r, s.am, s.T, s.Sjx, s.Sjy, s.vol, s.dtf,
                  s.acou_d, s.irs_d, s.irs_c, s.acou_kfac, NI, NJ, g; ndrange=(s.NI,))
    nothing
end

"""
    _emix_state(yvf, ia, ja, emix, eint, eint_v) -> (eint_loc, Y)

Mixture specific heat and vapour MASS fraction for the conserved-energy closure (S3g step 2).

ONE definition, called from all three energy sites (`_k_snapshot!`, `_k_histcycle!`, `_k_update!`),
because the E->T round trip is only exact if the same closure forms and inverts the energy. Three
hand-copies of this expression is precisely the kind of divergence that produced this campaign's
duplicated EOS and its two realised-state banners.

`emix <= 0` returns `(eint, 0)` so every caller collapses to the legacy single-cv expression BITWISE.
"""
@inline function _emix_state(yvf, ia::Int, ja::Int, emix, eint, eint_v)
    emix <= 0.0f0 && return (eint, 0.0f0)
    y = yvf[ia, ja]
    y = y < 0.0f0 ? 0.0f0 : (y > 1.0f0 ? 1.0f0 : y)
    return ((1.0f0 - y)*eint + y*eint_v, y)
end

@kernel function _k_snapshot!(rk_r, rk_mx, rk_my, rk_E, @Const(rho), @Const(ux), @Const(uy), @Const(T),
                              eint, @Const(yvf), emix, eint_v,
                              latent, NI::Int32, NJ::Int32, g::Int32)
    i, j = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ
        ia=i+g; ja=j+g
        r=rho[ia,ja]; u=ux[ia,ja]; v=uy[ia,ja]
        rk_r[ia,ja]=r; rk_mx[ia,ja]=r*u; rk_my[ia,ja]=r*v
        eL, Yl = _emix_state(yvf, ia, ja, emix, eint, eint_v)
        rk_E[ia,ja]=r*(eL*T[ia,ja]+Yl*latent+0.5f0*(u*u+v*v))
    end
end

@kernel function _k_update!(rho, ux, uy, T, @Const(rk_r), @Const(rk_mx), @Const(rk_my), @Const(rk_E),
                            @Const(dc_r), @Const(dc_mx), @Const(dc_my), @Const(dc_e),
                            @Const(un_r), @Const(un_mx), @Const(un_my), @Const(un_E),
                            @Const(um_r), @Const(um_mx), @Const(um_my), @Const(um_E),
                            @Const(tau_r), @Const(tau_mx), @Const(tau_my), @Const(tau_e),
                            @Const(dtf), @Const(vol), @Const(xc), @Const(yc),
                            eint, @Const(yvf), emix, eint_v,
                            latent, wst, bst, dtp,
                            use_tau::Int32, rk_incremental::Int32,
                            spg, j0f, gam, xv, yv,
                            m2, bet, ca, sa,
                            uxi, uyi, rho_inf, T_inf,
                            preserve_ref::Int32, ref_rho, ref_ux,
                            ref_uy, ref_T,
                            checked_mixture::Int32,
                            NI::Int32, NJ::Int32, g::Int32)
    i, j = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ
        ia=i+g; ja=j+g
        r0=rho[ia,ja]; u0=ux[ia,ja]; v0=uy[ia,ja]
        eL0, Yl0 = _emix_state(yvf, ia, ja, emix, eint, eint_v)
        E0=r0*(eL0*T[ia,ja]+Yl0*latent+0.5f0*(u0*u0+v0*v0))
        hold_ref = preserve_ref == Int32(1) && dtp == 0.0f0 && use_tau == Int32(0) &&
                   dc_r[i,j] == 0.0f0 && dc_mx[i,j] == 0.0f0 &&
                   dc_my[i,j] == 0.0f0 && dc_e[i,j] == 0.0f0 &&
                   r0 == ref_rho && u0 == ref_ux && v0 == ref_uy && T[ia,ja] == ref_T &&
                   rk_r[ia,ja] == r0 && rk_mx[ia,ja] == r0*u0 && rk_my[ia,ja] == r0*v0 &&
                   rk_E[ia,ja] == E0
        # A conservative->primitive round trip is not bitwise the identity in Float32. At the
        # explicitly seeded reference only, skip that round trip when the corrected RHS is exact
        # zero. Requiring both the live state and stage-0 snapshot keeps SSPRK semantics unchanged
        # for physical cells that merely have a zero instantaneous local residual.
        if !hold_ref
        a = dtf[i,j]/vol[i,j]
        if rk_incremental == Int32(1)
            # Explicit SSPRK in increments: avoid weighting two O(1) states to retain
            # an O(dt*R) change. Float32(1/3)+Float32(2/3) is greater than one in
            # exact arithmetic; the legacy large-state products can bias small waves.
            # The host admits this branch only without dual-time/FAS/filter terms.
            # Primitive recovery below still rounds momentum and energy between stages.
            r  = rk_r[ia,ja]  + wst*((r0    - rk_r[ia,ja])  - a*dc_r[i,j])
            mx = rk_mx[ia,ja] + wst*((r0*u0 - rk_mx[ia,ja]) - a*dc_mx[i,j])
            my = rk_my[ia,ja] + wst*((r0*v0 - rk_my[ia,ja]) - a*dc_my[i,j])
            e  = rk_E[ia,ja]  + wst*((E0    - rk_E[ia,ja])  - a*dc_e[i,j])
        else
        r  = r0    - a*dc_r[i,j]
        mx = r0*u0 - a*dc_mx[i,j]
        my = r0*v0 - a*dc_my[i,j]
        e  = E0    - a*dc_e[i,j]
        if use_tau == Int32(1)
            # FAS defect-correction source, applied PER STAGE so the pseudo-steady fixed point is
            # exactly dc = tau (the once-per-step splitting injected an O(dt^2) noise floor ~1e-5)
            r  += a*tau_r[i,j]
            mx += a*tau_mx[i,j]
            my += a*tau_my[i,j]
            e  += a*tau_e[i,j]
        end
        if dtp > 0.0f0
            # dual-time BDF2 point-implicit source (mirrors the plain solver line-for-line)
            dtl = dtf[i,j]; fac = 1.0f0/(1.0f0 + 1.5f0*dtl/dtp); h = dtl/(2.0f0*dtp)
            r  = (r  + h*(4.0f0*un_r[i,j]  - um_r[i,j] ))*fac
            mx = (mx + h*(4.0f0*un_mx[i,j] - um_mx[i,j]))*fac
            my = (my + h*(4.0f0*un_my[i,j] - um_my[i,j]))*fac
            e  = (e  + h*(4.0f0*un_E[i,j]  - um_E[i,j] ))*fac
        end
        r  = bst*rk_r[ia,ja]  + wst*r
        mx = bst*rk_mx[ia,ja] + wst*mx
        my = bst*rk_my[ia,ja] + wst*my
        e  = bst*rk_E[ia,ja]  + wst*e
        end
        if checked_mixture==Int32(1)
            # The conservative barotropic caller has admitted this rho/momentum/q
            # trial. It has no energy equation: the next EOS projection sets T.
            # Physical low vapor densities and velocities must not be clipped.
            rho[ia,ja]=r;ux[ia,ja]=mx/r;uy[ia,ja]=my/r
        else
        # The legacy gas/kinetic path retains its historical safeguards.
        # This floor is a numerical safeguard, not a saturation density. The
        # local-pressure vapor EOS can represent expanded vapor below rho_v_sat.
        r = r < 1.0f-4 ? 1.0f-4 : r
        u2=mx/r; v2=my/r; sp=sqrt(u2*u2+v2*v2)
        if sp>3.5f0; f=3.5f0/sp; u2*=f; v2*=f; end
        # ★ recover T from the CONSERVED energy with the same mixture closure the snapshot used.
        # Float32 primitive recovery can still round conserved moments. The Y*L subtraction cools
        # the cell under evaporation without changing the energy definition.
        eL2, Yl2 = _emix_state(yvf, ia, ja, emix, eint, eint_v)
        T2=(e/r-0.5f0*(u2*u2+v2*v2)-Yl2*latent)/eL2; T2 = T2<1.0f-3 ? 1.0f-3 : T2
        if spg > 0.0f0                                 # far-field acoustic sponge (mirrors plain)
            j0 = j0f*Float32(NJ)
            if Float32(j) > j0
                wsp = min(spg*dtf[i,j]*((Float32(j)-j0)/(Float32(NJ)-j0))^2, 0.5f0)
                du, dv = _pg_vortex_uv(xc[i,j], yc[i,j], gam, xv, yv, m2, bet, ca, sa)
                r  -= wsp*(r  - rho_inf)
                u2 -= wsp*(u2 - (uxi + du)); v2 -= wsp*(v2 - (uyi + dv))
                T2 -= wsp*(T2 - T_inf)
            end
        end
        rho[ia,ja]=r; ux[ia,ja]=u2; uy[ia,ja]=v2; T[ia,ja]=T2
        end
        end
    end
end

# true dual-time inner residual (mass component), scaled by the LOCAL pseudo-dt so cells of vastly
# different size are comparable (raw dc/vol is 1/vol-dominated at the wall — the unscaled inf-norm
# never converges 100x): out = dtl·[dc_r/vol + (3ρ - 4ρ^n + ρ^(n-1))/(2 dtp)] = the per-pseudo-step
# density update magnitude. Driver takes maximum(abs, out).
@kernel function _k_rstar!(out, @Const(rho), @Const(dc_r), @Const(un_r), @Const(um_r), @Const(vol),
                           @Const(dtf), dtp, NI::Int32, NJ::Int32, g::Int32)
    i, j = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ
        out[i,j] = dtf[i,j]*(dc_r[i,j]/vol[i,j] + (3.0f0*rho[i+g,j+g] - 4.0f0*un_r[i,j] + um_r[i,j])/(2.0f0*dtp))
    end
end

function rstar_linf_ka(s::KSol, dtp)
    NI, NJ, g = Int32(s.NI), Int32(s.NJ), Int32(s.g)
    _k_rstar!(s.backend)(s.gTx, s.rho, s.dc_r, s.un_r, s.um_r, s.vol, s.dtf, dtp, NI, NJ, g; ndrange=(s.NI, s.NJ))
    maximum(abs, s.gTx)      # gTx is free scratch when fullvisc == 0 (asserted by callers)
end

# ★ L-inf ALONE CANNOT SEPARATE "the inner loop cannot converge" FROM "one cavity cell is bad".
# rstar_linf_ka is a MAX, and this session established (residual_report_ka) that in a cavity the L-inf
# residual is owned by a SINGLE near-wall cell. If the inner loop is well-converged in L2 and only one
# cell plateaus, the forward path (look for shedding elsewhere) is completely different from a genuine
# field-wide non-convergence (build a preconditioner first). Returns (linf, l2, frac_at_max, imax, jmax).
function rstar_report_ka(s::KSol, dtp)
    NI, NJ, g = Int32(s.NI), Int32(s.NJ), Int32(s.g)
    _k_rstar!(s.backend)(s.gTx, s.rho, s.dc_r, s.un_r, s.um_r, s.vol, s.dtf, dtp, NI, NJ, g; ndrange=(s.NI, s.NJ))
    KA.synchronize(s.backend)
    r = abs.(Array(s.gTx))
    linf = maximum(r); l2 = sqrt(sum(abs2, r)/length(r))
    frac = count(>=(0.99*linf), r)/length(r); idx = argmax(r)
    (linf, l2, frac, idx[1], idx[2])
end

@kernel function _k_histcycle!(un_r, un_mx, un_my, un_E, um_r, um_mx, um_my, um_E,
                               @Const(rho), @Const(ux), @Const(uy), @Const(T),
                               eint, @Const(yvf), emix, eint_v,
                               latent, NI::Int32, NJ::Int32, g::Int32)
    i, j = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ
        ia=i+g; ja=j+g
        um_r[i,j]=un_r[i,j]; um_mx[i,j]=un_mx[i,j]; um_my[i,j]=un_my[i,j]; um_E[i,j]=un_E[i,j]
        r=rho[ia,ja]; u=ux[ia,ja]; v=uy[ia,ja]
        un_r[i,j]=r; un_mx[i,j]=r*u; un_my[i,j]=r*v
        eL, Yl = _emix_state(yvf, ia, ja, emix, eint, eint_v)
        un_E[i,j]=r*(eL*T[ia,ja]+Yl*latent+0.5f0*(u*u+v*v))
    end
end

function hist_cycle_ka!(s::KSol)
    NI, NJ, g = Int32(s.NI), Int32(s.NJ), Int32(s.g)
    _k_histcycle!(s.backend)(s.un_r, s.un_mx, s.un_my, s.un_E, s.um_r, s.um_mx, s.um_my, s.um_E,
                             s.rho, s.ux, s.uy, s.T, s.eint, s.yvf, s.emix, s.eint_v, s.latent, NI, NJ, g; ndrange=(s.NI, s.NJ))
    nothing
end

@kernel function _k_sfd!(rho, ux, uy, T, sb_r, sb_u, sb_v, sb_T,
                         chi, alpha, NI::Int32, NJ::Int32, g::Int32)
    i, j = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ
        ia=i+g; ja=j+g
        sb_r[ia,ja] += alpha*(rho[ia,ja] - sb_r[ia,ja]); rho[ia,ja] -= chi*(rho[ia,ja] - sb_r[ia,ja])
        sb_u[ia,ja] += alpha*(ux[ia,ja]  - sb_u[ia,ja]); ux[ia,ja]  -= chi*(ux[ia,ja]  - sb_u[ia,ja])
        sb_v[ia,ja] += alpha*(uy[ia,ja]  - sb_v[ia,ja]); uy[ia,ja]  -= chi*(uy[ia,ja]  - sb_v[ia,ja])
        sb_T[ia,ja] += alpha*(T[ia,ja]   - sb_T[ia,ja]); T[ia,ja]   -= chi*(T[ia,ja]   - sb_T[ia,ja])
    end
end

function sfd_seed_ka!(s::KSol)
    copyto!(s.sb_r, s.rho); copyto!(s.sb_u, s.ux); copyto!(s.sb_v, s.uy); copyto!(s.sb_T, s.T)
    nothing
end

# Exact packed-state donor for mirrored pressure/composition ghosts. Inlet,
# i-end and corner states are left to the existing characteristic BC routines.
@inline function _mixture_conserved_donor(i,j,kinds,NI,NJ,NIW,NIF,g)
    i32=Int32(i);j32=Int32(j)
    ii=i32-g;jj=j32-g
    if Int32(1)<=ii<=NI
        if j32<=g
            jj=g+Int32(1)-j32
            if ii<=NIW || ii>NIW+NIF
                ii=NI+Int32(1)-ii
            end
        elseif j32>NJ+g && kinds[ii]>=0x02
            jj=Int32(2)*NJ+g+Int32(1)-j32
        end
        Int32(1)<=jj<=NJ && return (ii,jj)
    end
    (Int32(0),Int32(0))
end

# The block-aware twin of the donor map above (strip topology, 5 Oct 2026, plan section 4.1 item 2; critiques A 2.1-2.3,
# B 2.2), used only with a strip pair's context; the map above is untouched and stays the default path's. It returns
# (ii, jj, src): src = 0 a cell (ii, jj) of the same block's conserved U, as above; src = 1 / 2 entry (station ii,
# layer jj) of the lower / upper interface in the pair's Float64 buffer U_nb, i.e. a cell of the OTHER block. block 1 (the
# strip) dispatches BEFORE any kinds test: its kind 4 satisfies `kinds >= 0x02`, which would mirror its upper-interface
# rows (critique A 2.3). On the strip: interior; bottom / top ghost rows -> interface 1 / 2 at (column, layer); the base-
# wall side's interior rows -> the true mirror (g+1-i32, jj), layer k from cell k (critique A 2.2); its corners -> the
# interface entry of the mirrored station (station g+1-i32, critique A 2.1(a)); the outlet side -> none (characteristic).
# On the C block (block 0): the leg columns' bottom rows -> the strip, lower leg column ii at station NIW+1-ii of
# interface 1, upper leg column ii at station ii-NIW-NIF of interface 2; every other ghost exactly as the map above.
@inline function _strip_conserved_donor(i,j,kinds,NI,NJ,NIW,NIF,g,block::Int32)
    i32=Int32(i);j32=Int32(j)
    ii=i32-g;jj=j32-g
    if block==Int32(1)
        if Int32(1)<=ii<=NI
            j32<=g && return (ii,g+Int32(1)-j32,Int32(1))
            j32>NJ+g && return (ii,j32-NJ-g,Int32(2))
            return (ii,jj,Int32(0))
        elseif i32<=g
            st=g+Int32(1)-i32
            j32<=g && return (st,g+Int32(1)-j32,Int32(1))
            j32>NJ+g && return (st,j32-NJ-g,Int32(2))
            return (st,jj,Int32(0))
        end
        return (Int32(0),Int32(0),Int32(0))
    end
    if Int32(1)<=ii<=NI
        if j32<=g
            ii<=NIW && return (NIW+Int32(1)-ii,g+Int32(1)-j32,Int32(1))
            ii>NIW+NIF && return (ii-NIW-NIF,g+Int32(1)-j32,Int32(2))
            jj=g+Int32(1)-j32
        elseif j32>NJ+g && kinds[ii]>=0x02
            jj=Int32(2)*NJ+g+Int32(1)-j32
        end
        Int32(1)<=jj<=NJ && return (ii,jj,Int32(0))
    end
    (Int32(0),Int32(0),Int32(0))
end
# The donor of a kernel launched with a strip pair's context, or without one (ctx === nothing: the map above, with
# src = 0, so a default-path launch evaluates exactly today's donor and never takes an interface branch).
@inline function _donor3(i,j,kinds,NI,NJ,NIW,NIF,g,::Nothing)
    d=_mixture_conserved_donor(i,j,kinds,NI,NJ,NIW,NIF,g)
    (d[1],d[2],Int32(0))
end
@inline _donor3(i,j,kinds,NI,NJ,NIW,NIF,g,ctx)=_strip_conserved_donor(i,j,kinds,NI,NJ,NIW,NIF,g,ctx.block)

# (rebuild, 22 Sep 2026) the projection kernel's EOS, factored VERBATIM so the fused boundary kernel below can
# recompute a cell's pressure/sound in another thread with bit-identical results.
@inline function _mixture_eos_from_U(r::Float64,q::Float64,ep::NTuple{7,Float32},single::Int32)
    local pressure; local sound
    if single==Int32(1)
        pv32,A32,rl32,rv32,psm32,lq32,afloor32=ep
        y32=Float32(q)/Float32(r);a32=y32*rl32/(rv32*(1.0f0-y32)+y32*rl32)
        d=Float32(Float64(rl32)-r)
        raw32,dr32,_=CMeshEOSMix._pmix_vl_d(pv32,a32,Float32(r),d,A32,rl32,rv32,lq32)
        sq32=sqrt(raw32*raw32+psm32*psm32)
        ssum32=raw32>=0.0f0 ? raw32+sq32 : psm32*psm32/(sq32-raw32)
        pressure=0.5f0*ssum32
        sound=sqrt(max(0.5f0*dr32*(ssum32/sq32),afloor32*(pv32/rl32)))
    else
        pv,A,rl,rv,psm,lq,afloor=Float64.(ep)
        y=q/r
        a=y*rl/(rv*(1.0-y)+y*rl)
        raw,dr,_=CMeshEOSMix._pmix_vl(pv,a,r,rl*(1.0-a)+rv*a,A,rl,rv,lq)
        sq=sqrt(raw*raw+psm*psm)
        pressure=Float32(0.5*(raw+sq))
        sound=Float32(sqrt(max(0.5*dr*(1.0+raw/sq),afloor*(pv/rl))))
    end
    (pressure,sound)
end
@inline function _mixture_eos_from_alpha(a,r,ep::NTuple{7,Float32})
    pv,A,rl,rv,psm,lq,afloor=ep
    raw,dr,_=CMeshEOSMix._pmix_vl(pv,a,r,rl*(1.0f0-a)+rv*a,A,rl,rv,lq)
    sq=sqrt(raw*raw+psm*psm)
    (0.5f0*(raw+sq),sqrt(max(0.5f0*dr*(1.0f0+raw/sq),afloor*(pv/rl))))
end

@kernel function _k_mixture_pressure!(p,am,T,pexc,@Const(rho),@Const(alpha),ep::NTuple{7,Float32},
                                      conserved,single::Int32,@Const(kinds),Tmin,Tmax,
                                      NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32,
                                      NX::Int32,NY::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NX && j<=NY
        ii,jj=conserved===nothing ? (Int32(0),Int32(0)) :
            _mixture_conserved_donor(i,j,kinds,NI,NJ,NIW,NIF,g)
        if ii!=Int32(0)
            # Promoting ep cannot recover material-reference bits already lost
            # at setup. It preserves the admitted state through EOS evaluation.
            r=conserved[ii,jj,1];q=conserved[ii,jj,4]
            local pressure; local sound
            if single==Int32(1)
                # Single-precision EOS arithmetic (2026-09-12): the liquid density
                # excess over saturation is formed in Float64, the rest in Float32.
                pv32,A32,rl32,rv32,psm32,lq32,afloor32=ep
                y32=Float32(q)/Float32(r);a32=y32*rl32/(rv32*(1.0f0-y32)+y32*rl32)
                d=Float32(Float64(rl32)-r)
                raw32,dr32,_=CMeshEOSMix._pmix_vl_d(pv32,a32,Float32(r),d,A32,rl32,rv32,lq32)
                sq32=sqrt(raw32*raw32+psm32*psm32)
                # raw32 < 0 is unreachable: _pmix_vl_d returns the non-negative root, so the second
                # branch is defensive only and the model represents no liquid tension (2026-09-17 audit).
                ssum32=raw32>=0.0f0 ? raw32+sq32 : psm32*psm32/(sq32-raw32)
                pressure=0.5f0*ssum32
                sound=sqrt(max(0.5f0*dr32*(ssum32/sq32),afloor32*(pv32/rl32)))
            else
                pv,A,rl,rv,psm,lq,afloor=Float64.(ep)
                y=q/r
                a=y*rl/(rv*(1.0-y)+y*rl)
                raw,dr,_=CMeshEOSMix._pmix_vl(pv,a,r,rl*(1.0-a)+rv*a,A,rl,rv,lq)
                sq=sqrt(raw*raw+psm*psm)
                pressure=Float32(0.5*(raw+sq))
                sound=Float32(sqrt(max(0.5*dr*(1.0+raw/sq),afloor*(pv/rl))))
            end
            p[i,j]=pressure;am[i,j]=sound
            # The shared carrier still stores a nonthermal Float32 pedestal.
            # Update it in this same kernel, never by reevaluating the rounded EOS.
            temp=clamp(pressure/(rho[i,j]*CS2),Tmin,Tmax)
            T[i,j]=temp;pexc[i,j]=pressure-rho[i,j]*CS2*temp
        else
            pv,A,rl,rv,psm,lq,afloor=ep
            a=alpha[i,j];r=rho[i,j]
            raw,dr,_=CMeshEOSMix._pmix_vl(pv,a,r,rl*(1.0f0-a)+rv*a,A,rl,rv,lq)
            sq=sqrt(raw*raw+psm*psm)
            p[i,j]=0.5f0*(raw+sq)
            am[i,j]=sqrt(max(0.5f0*dr*(1.0f0+raw/sq),afloor*(pv/rl)))
        end
    end
end

# Fused pressure trio (rebuild, 22 Sep 2026). One launch over the ghosted field replaces the projection kernel,
# the inlet ghost kernel and the outlet ghost kernel. Thread (i,j) classifies its own cell and writes only it:
#   * interior, and interior columns' ghost rows with a conserved donor: the projection (p, am, T, pexc);
#   * interior columns' top ghost rows at an inlet segment: the projection's alpha/rho branch (p, am), then the
#     inlet state -- whose input is the interior top-row state that the INT thread of that column computes in
#     this same launch, so it is RECOMPUTED here from U with the same helpers rather than read;
#   * ghost columns (all rows, corners included): the alpha/rho branch, then the outlet state from the source
#     column's cell, likewise recomputed (donor EOS, or the inlet state, or the alpha/rho branch) rather than read.
# No thread reads p, am, T or pexc written by another thread of the launch; rho/ux/uy/alpha are read only where
# an earlier launch wrote them. The result matches the three-launch sequence bit for bit (parity test).
@kernel function _k_species_pressure_ghosts!(p,am,T,pexc,rho,ux,uy,alpha,status,conserved,single::Int32,
        @Const(kinds),@Const(Six),@Const(Siy),@Const(Sjx),@Const(Sjy),ep::NTuple{7,Float32},Tmin,Tmax,
        ri,ui,vi,pinf,ci,pb,
        inlet_on::Int32,outlet_on::Int32,sst_on::Int32,NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32,NX::Int32,NY::Int32,ctx)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NX && j<=NY
        i32=Int32(i);j32=Int32(j)
        if g<i32<=NI+g
            # ctx: nothing (today's donor, src = 0), or a strip pair's block context (plan section 4.1 item 2)
            ii,jj,src=_donor3(i,j,kinds,NI,NJ,NIW,NIF,g,ctx)
            if src!=Int32(0)
                # An interface ghost of a strip pair: the exchange writes the neighbour block's published state here
                # after both projections. The trio skips it rather than re-deriving it from alpha (plan 4.1 item 2).
            elseif ii!=Int32(0)
                pressure,sound=_mixture_eos_from_U(conserved[ii,jj,1],conserved[ii,jj,4],ep,single)
                p[i,j]=pressure;am[i,j]=sound
                temp=clamp(pressure/(rho[i,j]*CS2),Tmin,Tmax)
                T[i,j]=temp;pexc[i,j]=pressure-rho[i,j]*CS2*temp
            else
                pr,so=_mixture_eos_from_alpha(alpha[i,j],rho[i,j],ep)
                p[i,j]=pr;am[i,j]=so
                col=i32-g
                if inlet_on!=Int32(0) && j32>NJ+g && kinds[col]==0x01
                    jt=NJ+g
                    pt,ct=_mixture_eos_from_U(conserved[col,NJ,1],conserved[col,NJ,4],ep,single)
                    Tt=clamp(pt/(rho[i,jt]*CS2),Tmin,Tmax)
                    Sx=Sjx[col,NJ+Int32(1)];Sy=Sjy[col,NJ+Int32(1)];Sm=sqrt(Sx*Sx+Sy*Sy)
                    rb,ub,vb,tb,pbb,cb,ab,err=CMeshPressureOutlet.inlet_state(rho[i,jt],ux[i,jt],uy[i,jt],Tt,pt,ct,alpha[i,jt],
                        Sx/Sm,Sy/Sm,ri,ui,vi,pinf,ci,ep,Tmin,Tmax)
                    if err!=0.0f0 || (sst_on!=Int32(0) && ub*Sx+vb*Sy<0.0f0)
                        j32==NJ+g+Int32(1) && (p[i,j]=-1.0f0)
                    else
                        rho[i,j]=rb;ux[i,j]=ub;uy[i,j]=vb;T[i,j]=tb
                        pexc[i,j]=pbb-rb*CS2*tb;am[i,j]=cb;alpha[i,j]=ab;p[i,j]=pbb
                    end
                end
            end
        elseif ctx===nothing || ((ctx.sides>>(i32<=g ? Int32(0) : Int32(1)))&Int32(1))==Int32(1)
            # Ghost columns. With a strip pair's context only the outlet sides its mask names (the strip: sides = 2; its
            # base-wall side is written after this launch by its column fill and by the exchange, plan section 4.1).
            pr,so=_mixture_eos_from_alpha(alpha[i,j],rho[i,j],ep)
            p[i,j]=pr;am[i,j]=so
            if outlet_on!=Int32(0)
                side=i32<=g ? Int32(1) : Int32(2)
                is=side==Int32(1) ? g+Int32(1) : NI+g
                k=side==Int32(1) ? g+Int32(1)-i32 : i32-NI-g
                jr=clamp(j32-g,Int32(1),NJ)
                face=side==Int32(1) ? Int32(1) : NI+Int32(1)
                sg=side==Int32(1) ? -1.0f0 : 1.0f0
                Sx=sg*Six[face,jr];Sy=sg*Siy[face,jr];Sm=sqrt(Sx*Sx+Sy*Sy)
                # the source cell (is,j) as the three-launch sequence leaves it, recomputed
                rd=rho[is,j];ud=ux[is,j];vd=uy[is,j];ad=alpha[is,j];td=T[is,j]
                di,dj,dsrc=_donor3(is,j,kinds,NI,NJ,NIW,NIF,g,ctx)
                local pd; local cd
                if ctx!==nothing && dsrc!=Int32(0)
                    # a corner beside an interface row: the source is the neighbour block's cell, taken whole from the pair's
                    # buffer U_nb (one exchange late) as the state check derives it from U (_k_species_check!; the commit's
                    # rho = U1, u = U2/U1, v = U3/U1, alpha of y = U4/U1). The values loaded above, from this block's
                    # interface ghost row (on the C block the row fill's partner copy, the OTHER leg's row), are discarded
                    # (strip topology, plan 4.1 item 1; the stage-2 review gate's D1, 6 Oct 2026).
                    r64=ctx.U_nb[di,dj,1,dsrc];q64=ctx.U_nb[di,dj,4,dsrc]
                    y64=q64/r64;rl64=Float64(ep[3]);rv64=Float64(ep[4])
                    rd=Float32(r64);ud=Float32(ctx.U_nb[di,dj,2,dsrc]/r64);vd=Float32(ctx.U_nb[di,dj,3,dsrc]/r64)
                    ad=Float32(y64*rl64/(rv64*(1.0-y64)+y64*rl64))
                    pd,cd=_mixture_eos_from_U(r64,q64,ep,single)
                    td=clamp(pd/(rd*CS2),Tmin,Tmax)
                elseif di!=Int32(0)
                    pd,cd=_mixture_eos_from_U(conserved[di,dj,1],conserved[di,dj,4],ep,single)
                    td=clamp(pd/(rd*CS2),Tmin,Tmax)
                else
                    pd,cd=_mixture_eos_from_alpha(ad,rd,ep)
                    scol=is-g
                    if inlet_on!=Int32(0) && j32>NJ+g && kinds[scol]==0x01
                        jt=NJ+g
                        pt,ct=_mixture_eos_from_U(conserved[scol,NJ,1],conserved[scol,NJ,4],ep,single)
                        Tt=clamp(pt/(rho[is,jt]*CS2),Tmin,Tmax)
                        Sxi=Sjx[scol,NJ+Int32(1)];Syi=Sjy[scol,NJ+Int32(1)];Smi=sqrt(Sxi*Sxi+Syi*Syi)
                        rb,ub,vb,tb,pbb,cb,ab,err=CMeshPressureOutlet.inlet_state(rho[is,jt],ux[is,jt],uy[is,jt],Tt,pt,ct,alpha[is,jt],
                            Sxi/Smi,Syi/Smi,ri,ui,vi,pinf,ci,ep,Tmin,Tmax)
                        if err!=0.0f0 || (sst_on!=Int32(0) && ub*Sxi+vb*Syi<0.0f0)
                            j32==NJ+g+Int32(1) && (pd=-1.0f0)
                        else
                            rd=rb;ud=ub;vd=vb;td=tb;pd=pbb;cd=cb;ad=ab
                        end
                    end
                end
                vals=CMeshPressureOutlet.outlet_state(rd,ud,vd,td,pd,cd,ad,Sx/Sm,Sy/Sm,ui,vi,pb,ep,Tmin,Tmax,true)
                rb,ub,vb,tb,pp,cb,ab,err=vals
                if sst_on!=Int32(0) && ub*Sx+vb*Sy<0.0f0
                    err=3.0f0
                end
                k==Int32(1) && (status[side,j]=err)
                rho[i,j]=rb;ux[i,j]=ub;uy[i,j]=vb;T[i,j]=tb
                pexc[i,j]=pp-rb*CS2*tb;am[i,j]=cb;alpha[i,j]=ab;p[i,j]=pp
            end
        end
    end
end
const FUSED_PRESSURE_BC=Ref{Bool}(true)

"Fill shared physical pressure after composition/flow ghosts, then apply the characteristic outlet."
function _mixture_pressure_bc!(s,p;conservative_sst::Bool=false,conserved=nothing,eos_single::Bool=false,
                               inlet_present=nothing,pressure_slot=nothing,status_slot=nothing,
                               fused::Bool=FUSED_PRESSURE_BC[],strip_ctx=nothing)
    CMeshPressureOutlet.check_pressure_array(s,p)
    # Strip topology (5 Oct 2026, plan sections 3.3 and 4.1): a strip pair's block projects through the fused trio with
    # its context (the block-aware donor, the interface ghosts skipped, the outlet sides mask). The strip carrier
    # (outer_mode 2) has no other projection, and the unfused path has no block-aware donor, so both are refused here.
    if s.outer_mode==Int32(2) || strip_ctx!==nothing
        strip_ctx!==nothing || throw(ArgumentError(
            "the strip carrier (outer_mode 2, CMeshSpeciesStripTopologyV1) projects its pressure only with its pair's context (strip_ctx): its two rows are interfaces"))
        fused && conserved!==nothing || throw(ArgumentError(
            "the strip topology (CMeshSpeciesStripTopologyV1) does not support the unfused pressure path (FUSED_PRESSURE_BC[] = false, _k_mixture_pressure!) or a projection without the conserved state: it has no block-aware donor; use the fused trio"))
    end
    if fused && conserved!==nothing
        conserved isa AbstractArray{Float64,3} && size(conserved)==(s.NI,s.NJ,4) &&
            KernelAbstractions.get_backend(conserved)==s.backend ||
            throw(ArgumentError("conserved mixture state must be packed Float64 (NI,NJ,4) on the flow backend"))
        all(a->!Base.mightalias(conserved,a),(p,s.am,s.T,s.pexc)) ||
            throw(ArgumentError("conserved mixture state must not alias pressure projection outputs"))
        inlet_on=inlet_present===nothing ? any(==(0x01),s.outer_face_kind) : inlet_present
        inlet_on && CMeshPressureOutlet._check_inlet_reference(s)
        outlet_on=s.outlet_mode!=Int32(0)
        if outlet_on
            (s.outer_mode==Int32(1) || (s.outer_mode==Int32(2) && strip_ctx!==nothing)) &&
                s.cav==Int32(4) && s.ff_mix==Int32(1) && s.emix==0.0f0 ||
                throw(ArgumentError("pressure outlet configuration changed outside its supported regime"))
        end
        sst_on=s.sst && !conservative_sst ? Int32(1) : Int32(0)
        _k_species_pressure_ghosts!(s.backend)(p,s.am,s.T,s.pexc,s.rho,s.ux,s.uy,s.outlet_alpha,s.outlet_status,
            conserved,Int32(eos_single),s.outer_face_kind,s.Six,s.Siy,s.Sjx,s.Sjy,s.outlet_eos,s.ff_Tmin,s.ff_Tmax,
            s.rho_inf,s.uxi,s.uyi,s.ff_p_ref,sqrt(s.ff_a2m),s.outlet_p,
            Int32(inlet_on),Int32(outlet_on),sst_on,Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF),Int32(s.g),
            Int32(s.NI+2s.g),Int32(s.NJ+2s.g),strip_ctx;ndrange=size(s.rho))
        # the immediate-mode host checks of the two ghost kernels, unchanged
        if inlet_on && pressure_slot===nothing
            minimum(p)>0.0f0 ||
                throw(ArgumentError("mixture inlet rejected invalid/transonic state or unsupported turbulence inflow; restore pressure/state scratch"))
        elseif inlet_on && pressure_slot!==:batched
            CMeshPressureOutlet._slot_min!(pressure_slot,p)
        end
        if outlet_on && status_slot===nothing
            maximum(s.outlet_status)==0.0f0 ||
                throw(ArgumentError("pressure outlet encountered invalid state, supersonic inflow, transonic correction, or unsupported turbulence backflow"))
        elseif outlet_on && status_slot!==:batched
            CMeshPressureOutlet._slot_max!(status_slot,s.outlet_status)
        end
        return nothing
    end
    if conserved!==nothing
        conserved isa AbstractArray{Float64,3} && size(conserved)==(s.NI,s.NJ,4) &&
            KernelAbstractions.get_backend(conserved)==s.backend ||
            throw(ArgumentError("conserved mixture state must be packed Float64 (NI,NJ,4) on the flow backend"))
        all(a->!Base.mightalias(conserved,a),(p,s.am,s.T,s.pexc)) ||
            throw(ArgumentError("conserved mixture state must not alias pressure projection outputs"))
    end
    _k_mixture_pressure!(s.backend)(p,s.am,s.T,s.pexc,s.rho,s.outlet_alpha,s.outlet_eos,
        conserved,Int32(eos_single),s.outer_face_kind,s.ff_Tmin,s.ff_Tmax,
        Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF),Int32(s.g),
        Int32(s.NI+2s.g),Int32(s.NJ+2s.g);ndrange=size(s.rho))
    CMeshPressureOutlet.mixture_inlet_bc!(s,p;conservative_sst,inlet_present,pressure_slot)
    pressure_outlet_bc!(s;mixture_pressure=p,conservative_sst,status_slot)
end

function _raw_residual_ka!(s::KSol, wallmode::Int32, wallflux::Int32;
                           cavslip=0.0f0, cavslip_p=1.0f0,
                           cavslip_rls=1.0f0, cavslip_rvs=0.0f0,
                           cav_first_order::Bool=false, mixture_pressure=nothing,
                           conservative_sst::Bool=false,conserved=nothing,kinetic=nothing,eos_single::Bool=false,
                           inlet_present=nothing,bc_pressure_slot=nothing,outlet_status_slot=nothing,
                           flux_pressure_slot=nothing,store_divergence::Bool=true)
    cav_first_order && s.cav != Int32(4) &&
        throw(ArgumentError("first-order mixture flux requires cav=4"))
    direct_p=mixture_pressure!==nothing
    conserved!==nothing && !direct_p &&
        throw(ArgumentError("conserved pressure projection requires direct mixture pressure"))
    conservative_sst && !direct_p &&
        throw(ArgumentError("conservative SST boundary transport requires direct mixture pressure"))
    if direct_p
        cav_first_order && s.outlet_mode==Int32(1) ||
            throw(ArgumentError("direct mixture pressure requires the first-order pressure-outlet path"))
        CMeshPressureOutlet.check_pressure_array(s,mixture_pressure)
    end
    applyBC_ka!(s, wallmode; cavslip=cavslip, cavslip_p=cavslip_p,
                cavslip_rls=cavslip_rls, cavslip_rvs=cavslip_rvs,defer_pressure_outlet=direct_p)
    if direct_p
        # One shared EOS pressure field prevents cancellation against the thermal
        # carrier pedestal at every face and at the characteristic outlet.
        _mixture_pressure_bc!(s,mixture_pressure;conservative_sst,conserved,eos_single,
            inlet_present,pressure_slot=bc_pressure_slot,status_slot=outlet_status_slot)
    end
    _raw_residual_body_ka!(s,wallmode,wallflux,cav_first_order,mixture_pressure;kinetic,
        inlet_present,pressure_slot=flux_pressure_slot,store_divergence)
end

"""
    refresh_raw_residual_ka!(s::KSol; wallmode::Int32=Int32(2))

Re-evaluate the unsmoothed face fluxes and finite-volume divergence on the
*current* single-phase air state. `step_ka!` leaves these arrays at its third
pre-update RK stage; this diagnostic can be called after the completed RK,
SFD, and SST updates, at sparse monitoring points and at the endpoint. It
does not advance time or update SFD/turbulence. All four primitive arrays,
including their ghost cells, are restored bitwise before return. The face
fluxes, gradients, and `dc_*` scratch arrays are overwritten with the fresh
raw residual. The API supports both KA CPU and CUDA backends and deliberately
excludes the mixture, strip, channel, and pressure-outlet paths, whose
residual/source contracts require separate treatment.
"""
function refresh_raw_residual_ka!(s::KSol; wallmode::Int32=Int32(2))
    (s.cav == Int32(0) && s.outer_mode == Int32(0) &&
     s.outlet_mode == Int32(0) && s.ff_mix == Int32(0)) ||
        throw(ArgumentError("raw residual refresh supports single-phase air on a free-air C mesh only"))
    wallmode in (Int32(0),Int32(1),Int32(2)) ||
        throw(ArgumentError("wallmode must be 0, 1, or 2"))
    if s.air_sutherland != Int32(0)
        _check_air_sutherland_admission(s)
        (wallmode == Int32(2) && s.wall_flux) ||
            throw(ArgumentError("air Sutherland mode requires an impermeable no-slip airfoil wall"))
    end
    # Boundary conditions write primitive ghost cells. Preserve even those
    # cells so this inspection cannot alter the next step's initial state.
    KA.synchronize(s.backend)
    saved = (copy(s.rho),copy(s.ux),copy(s.uy),copy(s.T))
    try
        _raw_residual_ka!(s,wallmode,s.wall_flux ? Int32(1) : Int32(0))
        KA.synchronize(s.backend)
    finally
        for (dest,src) in zip((s.rho,s.ux,s.uy,s.T),saved)
            copyto!(dest,src)
        end
        KA.synchronize(s.backend)
    end
    nothing
end

# Private body for the conservative SST caller whose flow/EOS ghosts have
# already been admitted before preparing turbulence. The ordinary entrypoint
# retains its checks and complete BC ordering above.
# `kinetic` (2026-09-10) selects the common-gauge KFVS face on the same admitted
# direct-pressure states: a NamedTuple of Species-owned arrays half_i (NI+1,NJ,2),
# half_j (NI,NJ+1,2), rate_i (NI+1,NJ) and rate_j (NI,NJ+1) receiving the
# area-integrated half masses M+/M- and the operational face clock at every face,
# including the exact zero halves and mdot splits written by the boundary overrides.
function _raw_residual_body_ka!(s::KSol,wallmode::Int32,wallflux::Int32,
                                cav_first_order::Bool,mixture_pressure;face_states=nothing,kinetic=nothing,
                                inlet_present=nothing,pressure_slot=nothing,store_divergence::Bool=true,
                                strip_ctx=nothing,outlet_sides::Int32=Int32(3))
    # strip_ctx / outlet_sides (strip topology, 5 Oct 2026, plan section 4.1): a strip pair's block context for the two
    # flux kernels (the base wall on the strip's i = 1 faces, the interfaces' neighbour centroids, the top interface) and
    # the pressure-outlet sides (the strip's outlet is its i = NI+1 side only). Defaults: nothing and both sides, today's.
    (s.outer_mode==Int32(2) && strip_ctx===nothing) && throw(ArgumentError(
        "the strip carrier (outer_mode 2, CMeshSpeciesStripTopologyV1) evaluates its residual only with its pair's context (strip_ctx)"))
    air_geometry=nothing
    if s.recon==_AIRFOIL_MUSCL_RECON
        s.cav==0&&s.outer_mode==0&&s.outlet_mode==0&&s.ff_mix==0&&s.emix==0&&
            strip_ctx===nothing&&face_states===nothing&&kinetic===nothing&&mixture_pressure===nothing ||
            throw(ArgumentError("Geometric thermal MUSCL is separate from Species/mixture/strip reconstruction"))
        w=get(_AIRFOIL_MUSCL,s,nothing)
        w===nothing&&throw(ArgumentError("MUSCL requires its original-node workspace; configure it explicitly after restart"))
        face_states=_airfoil_muscl_prepare!(s,w)
        air_geometry=w.geometry.displacements
    end
    if face_states!==nothing
        (air_geometry!==nothing || (cav_first_order && s.cav==Int32(4) && mixture_pressure!==nothing)) ||
            throw(ArgumentError("prepared face states require admitted direct-pressure cav4 states"))
        eltype(face_states)==(air_geometry===nothing ? Float32 : eltype(s.rho)) && size(face_states)==(s.NI,s.NJ,7,4) &&
            KA.get_backend(face_states)==s.backend ||
            throw(ArgumentError("prepared face-state shape, type or backend mismatch"))
    end
    if kinetic!==nothing
        cav_first_order && s.cav==Int32(4) && mixture_pressure!==nothing ||
            throw(ArgumentError("kinetic species faces require admitted direct-pressure cav4 states"))
        kinetic isa NamedTuple && keys(kinetic)==(:half_i,:half_j,:rate_i,:rate_j) &&
            eltype(kinetic.half_i)==Float32 && size(kinetic.half_i)==(s.NI+1,s.NJ,2) &&
            eltype(kinetic.half_j)==Float32 && size(kinetic.half_j)==(s.NI,s.NJ+1,2) &&
            eltype(kinetic.rate_i)==Float32 && size(kinetic.rate_i)==(s.NI+1,s.NJ) &&
            eltype(kinetic.rate_j)==Float32 && size(kinetic.rate_j)==(s.NI,s.NJ+1) &&
            all(KA.get_backend(a)==s.backend for a in values(kinetic)) ||
            throw(ArgumentError("kinetic face workspace shape, type or backend mismatch"))
    end
    half_i=kinetic===nothing ? nothing : kinetic.half_i; rate_i=kinetic===nothing ? nothing : kinetic.rate_i
    half_j=kinetic===nothing ? nothing : kinetic.half_j; rate_j=kinetic===nothing ? nothing : kinetic.rate_j
    b = s.backend; NI, NJ, g = Int32(s.NI), Int32(s.NJ), Int32(s.g)
    sston = s.sst ? Int32(1) : Int32(0)
    direct_p=mixture_pressure!==nothing
    pmix=direct_p ? mixture_pressure : s.pexc
    if s.nu_lam > 0.0f0 && s.fullvisc == Int32(1) && air_geometry===nothing
        _k_gradients!(b)(s.gux, s.guy, s.gvx, s.gvy, s.gTx, s.gTy, s.ux, s.uy, s.T,
                         s.Six, s.Siy, s.Sjx, s.Sjy, s.vol, NI, NJ, g, strip_ctx; ndrange=(s.NI, s.NJ))
    end
    # ★ cav level 4 (mixture flux): freestream Mach^2 on the LIQUID mixture sound speed is the AUSM+-up
    # low-Mach preconditioning floor. Read only when cav >= 4; harmless otherwise.
    a2_liq = s.ff_mix == Int32(1) ? s.ff_a2m : s.am_ref*s.am_ref
    minf2 = Float32((s.uxi*s.uxi + s.uyi*s.uyi)/max(a2_liq, 1.0f-6))
    _k_flux_i!(b)(s.Fi_r, s.Fi_mx, s.Fi_my, s.Fi_e, s.rho, s.ux, s.uy, s.T,
                  s.Six, s.Siy, s.xc, s.yc, s.nut, s.tk,
                  s.exx, s.exy, s.eyy,
                  s.gux, s.guy, s.gvx, s.gvy, s.gTx, s.gTy, s.cxa, s.cya, s.cza, s.wa,
                  s.drot, s.nu_lam, sston, s.recon >= Int32(1) ? Int32(1) : Int32(0), s.fullvisc,
                  s.suth, s.rho_inf, s.air_sutherland,
                  s.rhok, s.earsm, s.pexc, s.am, s.cav, s.eint, minf2, Int32(cav_first_order),
                  pmix,Int32(direct_p),face_states,air_geometry,half_i,rate_i,
                  NI, NJ, g, strip_ctx; ndrange=(s.NI+1, s.NJ))
    _k_flux_j!(b)(s.Fj_r, s.Fj_mx, s.Fj_my, s.Fj_e, s.rho, s.ux, s.uy, s.T,
                  s.Sjx, s.Sjy, s.xc, s.yc, s.nut, s.vol, s.tk,
                  s.exx, s.exy, s.eyy,
                  s.gux, s.guy, s.gvx, s.gvy, s.gTx, s.gTy, s.cxa, s.cya, s.cza, s.wa,
                  s.drot, s.nu_lam, sston, wallflux,
                  s.recon == Int32(1) ? Int32(1) : Int32(0), s.fullvisc,
                  s.suth, s.rho_inf, s.air_sutherland,
                  s.rhok, s.earsm, s.pexc, s.am, s.cav,
                  s.wallfun, s.wf_larsson, s.eint, minf2, Int32(cav_first_order),
                  pmix,Int32(direct_p),wallmode,face_states,air_geometry,half_j,rate_j,
                  NI, NJ, Int32(s.NIW), Int32(s.NIF), g, strip_ctx; ndrange=(s.NI, s.NJ+1))
    if air_geometry!==nothing&&s.NIW>0
        _k_airfoil_shared_wake_flux!(b)(s.Fj_r,s.Fj_mx,s.Fj_my,s.Fj_e,
            NI,Int32(s.NIW);ndrange=s.NIW)
    end
    outer_flux_override_ka!(s;mixture_pressure,half=half_j)
    direct_p && CMeshPressureOutlet.mixture_inlet_flux!(s,mixture_pressure;half=half_j,inlet_present,pressure_slot)
    pressure_outlet_flux!(s;mixture_pressure,half=half_i,sides=outlet_sides)
    # The species march recomputes its divergence in Float64 from the same face arrays inside the
    # trial kernel; on that path s.dc_* was written every stage and read by nothing (rebuild, 22 Sep 2026).
    store_divergence && _k_residual!(b)(s.dc_r, s.dc_mx, s.dc_my, s.dc_e,
                    s.Fi_r, s.Fi_mx, s.Fi_my, s.Fi_e,
                    s.Fj_r, s.Fj_mx, s.Fj_my, s.Fj_e,
                    NI, NJ; ndrange=(s.NI, s.NJ))
    nothing
end

function _validate_channel_compat_ka!(s::KSol)
    s.outer_mode == Int32(0) && return nothing
    s.sponge_sigma == 0.0f0 || throw(ArgumentError("the far-field sponge is not defined for segmented channel boundaries"))
    s.acou == Int32(0) || throw(ArgumentError("the acoustic filter is not defined for segmented channel boundaries"))
    s.gam == 0.0f0 || throw(ArgumentError("the point-vortex far-field correction is not defined for segmented channel boundaries"))
    nothing
end

function _validate_channel_wall_alignment_ka!(s::KSol)
    s.outer_mode == Int32(0) && return nothing
    Umag = hypot(Float64(s.uxi), Float64(s.uyi))
    isfinite(Umag) || throw(ArgumentError("channel freestream speed is not finite"))
    Umag == 0.0 && return nothing
    kinds = Array(s.outer_face_kind)
    Sjx = Array(s.Sjx)
    Sjy = Array(s.Sjy)
    nwalls = 0
    @inbounds for i in 1:s.NI
        kind = kinds[i]
        if kind == 0x02 || kind == 0x03
            nwalls += 1
            Sx = Float64(Sjx[i,s.NJ+1]); Sy = Float64(Sjy[i,s.NJ+1])
            Smag = hypot(Sx, Sy)
            isfinite(Smag) && Smag > 0.0 || throw(ArgumentError(
                "channel outer wall face $i has an invalid normal"))
            relative_dot = abs(Float64(s.uxi)*Sx + Float64(s.uyi)*Sy)/(Umag*Smag)
            isfinite(relative_dot) && relative_dot < 1.0e-6 || throw(ArgumentError(
                "channel freestream is not tangent to outer wall face $i (relative normal dot=$relative_dot, required < 1e-6)"))
        end
    end
    # An all-characteristic free-fluid mask is admitted only through the
    # explicit provenance branch in _outer_boundary_contract_ka.
    nwalls > 0 || all(==(0x01),kinds) || throw(ArgumentError("segmented channel has no outer wall faces"))
    nothing
end

"""Seed the channel-only discrete freestream residual on scratch state.

The foil is transparent and its analytic wall flux is disabled in this reference evaluation. The
segmented outer inlet/walls remain active, so subtracting the result removes only the mesh/operator
freestream floor and cannot cancel the physical foil forcing used by later steps.
"""
function seed_channel_reference_ka!(s::KSol)
    s.outer_mode == Int32(2) && throw(ArgumentError(
        "seed_channel_reference_ka! refuses the strip carrier (outer_mode 2, CMeshSpeciesStripTopologyV1): it has no outer boundary, and the reference residual would fill its interface rows with channel ghosts; seed the C block only"))
    _validate_channel_compat_ka!(s)
    ref = _channel_reference_ambient_ka(s)
    clear_channel_reference_ka!(s)
    _validate_channel_wall_alignment_ka!(s)

    rr=similar(s.rho); ru=similar(s.ux); rv=similar(s.uy); rT=similar(s.T)
    rp=similar(s.pexc); ra=similar(s.am); rtk=similar(s.tk); rnut=similar(s.nut)
    rex=similar(s.exx); rxy=similar(s.exy); rey=similar(s.eyy)
    rgux=similar(s.gux); rguy=similar(s.guy); rgvx=similar(s.gvx); rgvy=similar(s.gvy)
    rgTx=similar(s.gTx); rgTy=similar(s.gTy)
    rFir=similar(s.Fi_r); rFimx=similar(s.Fi_mx); rFimy=similar(s.Fi_my); rFie=similar(s.Fi_e)
    rFjr=similar(s.Fj_r); rFjmx=similar(s.Fj_mx); rFjmy=similar(s.Fj_my); rFje=similar(s.Fj_e)
    fill!(rr,ref.rho); fill!(ru,ref.ux); fill!(rv,ref.uy); fill!(rT,ref.T)
    fill!(rp,ref.pexc); fill!(ra,ref.am); fill!(rtk,ref.tk); fill!(rnut,ref.nut)
    for a in (rex,rxy,rey,rgux,rguy,rgvx,rgvy,rgTx,rgTy,rFir,rFimx,rFimy,rFie,rFjr,rFjmx,rFjmy,rFje)
        fill!(a,0.0f0)
    end

    saved = (rho=s.rho,ux=s.ux,uy=s.uy,T=s.T,pexc=s.pexc,am=s.am,tk=s.tk,nut=s.nut,
             exx=s.exx,exy=s.exy,eyy=s.eyy,
             gux=s.gux,guy=s.guy,gvx=s.gvx,gvy=s.gvy,gTx=s.gTx,gTy=s.gTy,
             Fi_r=s.Fi_r,Fi_mx=s.Fi_mx,Fi_my=s.Fi_my,Fi_e=s.Fi_e,
             Fj_r=s.Fj_r,Fj_mx=s.Fj_mx,Fj_my=s.Fj_my,Fj_e=s.Fj_e,
             dc_r=s.dc_r,dc_mx=s.dc_mx,dc_my=s.dc_my,dc_e=s.dc_e,
             outlet_mode=s.outlet_mode)
    try
        # The reference removes the ambient mesh/operator floor. Including imposed
        # back pressure here would subtract its physical forcing from every step.
        s.outlet_mode=Int32(0)
        s.rho=rr; s.ux=ru; s.uy=rv; s.T=rT; s.pexc=rp; s.am=ra; s.tk=rtk; s.nut=rnut
        s.exx=rex; s.exy=rxy; s.eyy=rey
        s.gux=rgux; s.guy=rguy; s.gvx=rgvx; s.gvy=rgvy; s.gTx=rgTx; s.gTy=rgTy
        s.Fi_r=rFir; s.Fi_mx=rFimx; s.Fi_my=rFimy; s.Fi_e=rFie
        s.Fj_r=rFjr; s.Fj_mx=rFjmx; s.Fj_my=rFjmy; s.Fj_e=rFje
        s.dc_r=s.outer_ref_r; s.dc_mx=s.outer_ref_mx
        s.dc_my=s.outer_ref_my; s.dc_e=s.outer_ref_e
        _raw_residual_ka!(s, Int32(0), Int32(0))
        KA.synchronize(s.backend)
    finally
        s.outlet_mode=saved.outlet_mode
        s.rho=saved.rho; s.ux=saved.ux; s.uy=saved.uy; s.T=saved.T
        s.pexc=saved.pexc; s.am=saved.am; s.tk=saved.tk; s.nut=saved.nut
        s.exx=saved.exx; s.exy=saved.exy; s.eyy=saved.eyy
        s.gux=saved.gux; s.guy=saved.guy; s.gvx=saved.gvx; s.gvy=saved.gvy
        s.gTx=saved.gTx; s.gTy=saved.gTy
        s.Fi_r=saved.Fi_r; s.Fi_mx=saved.Fi_mx; s.Fi_my=saved.Fi_my; s.Fi_e=saved.Fi_e
        s.Fj_r=saved.Fj_r; s.Fj_mx=saved.Fj_mx; s.Fj_my=saved.Fj_my; s.Fj_e=saved.Fj_e
        s.dc_r=saved.dc_r; s.dc_mx=saved.dc_mx; s.dc_my=saved.dc_my; s.dc_e=saved.dc_e
    end
    s.outer_ref_rho=ref.rho; s.outer_ref_ux=ref.ux
    s.outer_ref_uy=ref.uy; s.outer_ref_T=ref.T
    s.outer_ref_signature=_channel_signature_ka(s)
    s.outer_ref_ready=Int32(1)
    nothing
end

function _validate_channel_ka!(s::KSol)
    _validate_channel_compat_ka!(s)
    s.outer_mode == Int32(0) && return nothing
    # The strip carrier (outer_mode 2) has no channel reference; the species admission of a strip pair skips this check
    # (cmesh_species_strip.jl, _admit_strip), so reaching it means a legacy entry (step_ka!) was given the strip carrier.
    s.outer_mode == Int32(2) && throw(ArgumentError(
        "the strip carrier (outer_mode 2, CMeshSpeciesStripTopologyV1) has no channel reference: the legacy path (step_ka!) does not support the strip topology"))
    s.outer_ref_ready == Int32(1) ||
        throw(ArgumentError("segmented channel residual reference is unseeded; call seed_channel_reference_ka!(s) after configuring the solver"))
    sig = _channel_signature_ka(s)
    if sig != s.outer_ref_signature
        clear_channel_reference_ka!(s)
        throw(ArgumentError("segmented channel residual reference is stale after a solver configuration change; reseed it"))
    end
    nothing
end

export prepare_dt_ka!

"""Compute the next stability-limited step without advancing any physical field.

Global steps may be shortened to land on an output time; `last_global_dt` records
the exact Float32 value applied to every cell. It must be accumulated by a caller,
since velocity, temperature and eddy viscosity can change this limit each step.
Local stepping records zero, because it has no single physical elapsed time.
"""
function prepare_dt_ka!(s::KSol, cfl; max_dt=Inf32, slot=nothing)
    s.air_sutherland != Int32(0) && _check_air_sutherland_admission(s)
    max_dt > 0 || throw(ArgumentError("max_dt must be positive (or +Inf)"))
    (GLOBAL_DT[] == 1 || max_dt == Inf32) ||
        throw(ArgumentError("a physical-time cap requires global stepping"))
    b = s.backend; NI, NJ, g = Int32(s.NI), Int32(s.NJ), Int32(s.g)
    sston = s.sst ? Int32(1) : Int32(0)
    _k_dtfield!(b)(s.dtf, s.rho, s.ux, s.uy, s.T, s.nut, s.Six, s.Siy, s.Sjx, s.Sjy, s.vol,
                   cfl, s.nu_lam, s.rho_inf, s.suth, s.air_sutherland,
                   sston, s.irsj, s.am, s.cav, s.cav_cfl, s.am_ref, NI, NJ, g; ndrange=(s.NI, s.NJ))
    # ★★★ GLOBAL (NON-LTS) TIME STEP — 2026-07-31. `dtf` is a LOCAL step: every cell marches on its own
    # clock, which makes the march a relaxation to steady state and NOT time-accurate. That is the
    # premise the whole dual-time wrapper exists to repair, and today's measurements closed that route
    # (the oscillation stays clocked by inner-iteration count in EVERY reachable configuration, even
    # with the BDF2 physical-time term raised to co-equal weight).
    # Collapsing dtf to its GLOBAL MINIMUM makes the explicit march time-accurate BY CONSTRUCTION:
    # one clock, no inner loop, no convergence question, no BDF2. It is brute force -- the step is set
    # by the stiffest cell -- but it is the ground truth against which every dual-time result can be
    # judged, and it is the natural fallback now that the implicit route is measured shut.
    # 0 = bitwise legacy (LTS, as the whole campaign has run).
    if slot !== nothing
        # Deferred admission (2026-09-12): the global minimum stays on the device; the
        # owner forms min(minimum(dtf), max_dt) from the slot and fills dtf itself.
        GLOBAL_DT[] == 1 || throw(ArgumentError("a deferred clock slot requires global stepping"))
        slot === :batched || CMeshPressureOutlet._slot_min!(slot, s.dtf)   # :batched (2026-09-12): reduced by the owner's next batch
        return nothing
    end
    if GLOBAL_DT[] == 1
        KernelAbstractions.synchronize(b)
        dt = min(minimum(s.dtf), max_dt)
        isfinite(dt) && dt > 0 || error("global stability step must be finite and positive")
        fill!(s.dtf, dt)
        s.last_global_dt = dt
    else
        s.last_global_dt = 0.0f0
    end
    nothing
end

function _check_rk_incremental(global_dt::Integer, dtp, tau,
                               irsj::Int32, irs_ac::Int32, sfd::Int32, acou::Int32)
    global_dt == 1 || throw(ArgumentError("rk_incremental requires the global physical timestep"))
    dtp == 0.0f0 || throw(ArgumentError("rk_incremental does not support dual-time stepping"))
    tau === nothing || throw(ArgumentError("rk_incremental does not support FAS sources"))
    irsj == 0 && irs_ac == 0 || throw(ArgumentError("rk_incremental does not support residual smoothing"))
    sfd == 0 || throw(ArgumentError("rk_incremental does not support selective frequency damping"))
    acou == 0 || throw(ArgumentError("rk_incremental does not support the acoustic filter"))
    nothing
end

# rk_incremental is an opt-in arithmetic change for the explicit global march.
# It retains the SSPRK formula while reducing cancellation around the saved state;
# it does not change the temporal order of TEM/SST or remove primitive-recovery rounding.
function step_ka!(s::KSol, cfl; wallmode::Int32=Int32(2), dtp=0.0f0,
                  tau=nothing,     # optional 4-tuple of (NI,NJ) FAS defect-correction sources
                  cavslip=0.0f0, cavslip_p=1.0f0,
                  cavslip_rls=1.0f0, cavslip_rvs=0.0f0,
                  max_dt=Inf32, rk_incremental::Bool=false,
                  rk_weight_closure::Bool=false,
                  probe_after_rk=nothing, probe_after_sfd=nothing)
    if s.air_sutherland != Int32(0)
        _check_air_sutherland_admission(s)
        (wallmode == Int32(2) && s.wall_flux) ||
            throw(ArgumentError("air Sutherland mode requires an impermeable no-slip airfoil wall"))
    end
    rk_incremental && _check_rk_incremental(GLOBAL_DT[], dtp, tau, s.irsj, s.irs_ac, s.sfd, s.acou)
    rk_incremental && rk_weight_closure &&
        throw(ArgumentError("rk_weight_closure is unused with rk_incremental"))
    _validate_channel_ka!(s)
    (max_dt == Inf32 || dtp == 0.0f0) ||
        throw(ArgumentError("a physical-time cap is only supported by the explicit global march"))
    b = s.backend; NI, NJ, g = Int32(s.NI), Int32(s.NJ), Int32(s.g)
    sston = s.sst ? Int32(1) : Int32(0)
    wf = s.wall_flux ? Int32(1) : Int32(0)
    prepare_dt_ka!(s, cfl; max_dt=max_dt)
    _k_snapshot!(b)(s.rk_r, s.rk_mx, s.rk_my, s.rk_E, s.rho, s.ux, s.uy, s.T, s.eint, s.yvf, s.emix, s.eint_v, s.latent, NI, NJ, g; ndrange=(s.NI, s.NJ))
    w3 = 2.0f0/3.0f0
    b3 = rk_weight_closure ? 1.0f0-w3 : 1.0f0/3.0f0
    for (wst, bst) in ((1.0f0, 0.0f0), (0.25f0, 0.75f0), (w3, b3))
        _raw_residual_ka!(s, wallmode, wf; cavslip=cavslip, cavslip_p=cavslip_p,
                          cavslip_rls=cavslip_rls, cavslip_rvs=cavslip_rvs)
        if s.outer_mode != Int32(0)
            _k_subtract_channel_reference!(b)(s.dc_r, s.dc_mx, s.dc_my, s.dc_e,
                                               s.outer_ref_r, s.outer_ref_mx,
                                               s.outer_ref_my, s.outer_ref_e,
                                               NI, NJ; ndrange=(s.NI, s.NJ))
        end
        if s.irsj == Int32(1)
            for dcarr in (s.dc_r, s.dc_mx, s.dc_my, s.dc_e)
                _k_irsj!(b)(dcarr, s.irs_c, s.irs_d, s.ux, s.uy, s.T, s.nut,
                           s.Sjx, s.Sjy, s.vol, s.dtf, s.nu_lam, sston, s.am, Int32(0), s.am_ref, 0.0f0, NI, NJ, g; ndrange=(s.NI,))
            end
        elseif s.irs_ac >= Int32(1)
            # VOID-TARGETED DUAL IRS (dt-neutral, fixed-point-preserving): alternating i- then j-sweep,
            # both with ε floored by irs_eps0*relu(1-a_m/a_ref) so the smoothing targets the cavity.
            # MOMENTUM+ENERGY ONLY (skip mass dc_r): the pexc pressure-split force enters ONLY the momentum
            # and energy fluxes -- the mass residual has no pexc term. Smoothing dc_r across the sharp cavity
            # interface decouples the flow's mass update from the LOCAL TEM vaporization source (which drove
            # amax 0.345->1.0 runaway in the eps0=2 test). H1/H2 discriminator.
            for dcarr in (s.dc_mx, s.dc_my, s.dc_e)
                _k_irsi!(b)(dcarr, s.irs_c, s.irs_d, s.ux, s.uy, s.T, s.nut,
                           s.Six, s.Siy, s.vol, s.dtf, s.nu_lam, sston, s.am, s.irs_ac, s.am_ref, s.irs_eps0, NI, NJ, g; ndrange=(s.NJ,))
                _k_irsj!(b)(dcarr, s.irs_c, s.irs_d, s.ux, s.uy, s.T, s.nut,
                           s.Sjx, s.Sjy, s.vol, s.dtf, s.nu_lam, sston, s.am, s.irs_ac, s.am_ref, s.irs_eps0, NI, NJ, g; ndrange=(s.NI,))
            end
        end
        _k_update!(b)(s.rho, s.ux, s.uy, s.T, s.rk_r, s.rk_mx, s.rk_my, s.rk_E,
                      s.dc_r, s.dc_mx, s.dc_my, s.dc_e,
                      s.un_r, s.un_mx, s.un_my, s.un_E, s.um_r, s.um_mx, s.um_my, s.um_E,
                      tau === nothing ? s.dc_r : tau[1], tau === nothing ? s.dc_mx : tau[2],
                      tau === nothing ? s.dc_my : tau[3], tau === nothing ? s.dc_e : tau[4],
                      s.dtf, s.vol, s.xc, s.yc, s.eint, s.yvf, s.emix, s.eint_v, s.latent, wst, bst, dtp,
                      tau === nothing ? Int32(0) : Int32(1), Int32(rk_incremental),
                      s.sponge_sigma, s.sponge_j0f, s.gam, s.xv, s.yv, s.m2, s.beta, s.ca, s.sa,
                      s.uxi, s.uyi, s.rho_inf, s.T_inf,
                      s.outer_mode != Int32(0) ? Int32(1) : Int32(0),
                      s.outer_ref_rho, s.outer_ref_ux, s.outer_ref_uy, s.outer_ref_T,
                      Int32(0), NI, NJ, g; ndrange=(s.NI, s.NJ))
    end
    # Optional host-side observation points for validation. The default path
    # remains the exact historical kernel sequence. A probe may refresh raw
    # fluxes/divergence but must leave the primitive state unchanged.
    probe_after_rk === nothing || probe_after_rk(s)
    if s.sfd == Int32(1)
        _k_sfd!(b)(s.rho, s.ux, s.uy, s.T, s.sb_r, s.sb_u, s.sb_v, s.sb_T,
                   s.sfd_chi, s.sfd_alpha, NI, NJ, g; ndrange=(s.NI, s.NJ))
    end
    probe_after_sfd === nothing || probe_after_sfd(s)
    # TEM/P3 consume flow ghosts after this routine returns. Refresh them from the final RK/SFD
    # primitives. Mode 0 deliberately retains the historical no-extra-launch path.
    if s.outer_mode != Int32(0)
        applyBC_ka!(s, wallmode; cavslip=cavslip, cavslip_p=cavslip_p,
                    cavslip_rls=cavslip_rls, cavslip_rvs=cavslip_rvs)
    end
    nothing
end

# ---- SST kernels -------------------------------------------------------------------------------
@kernel function _k_sst_bc!(tk, tom, nut, tnl, k_inf, w_inf, keps,
                            NI::Int32, NJ::Int32, NIW::Int32, NIF::Int32, g::Int32)
    i, k = @index(Global, NTuple)
    @inbounds if i <= NI && k <= g
        if i <= NIW || i > NIW+NIF
            ip = NI+1-i
            tk[i+g, g+1-k] = tk[ip+g, k+g]; tom[i+g, g+1-k] = tom[ip+g, k+g]; nut[i+g, g+1-k] = nut[ip+g, k+g]
            tnl[i+g, g+1-k] = tnl[ip+g, k+g]
        else
            tk[i+g, g+1-k] = -tk[i+g, g+k]; tom[i+g, g+1-k] = tom[i+g, g+k]; nut[i+g, g+1-k] = 0.0f0
            tnl[i+g, g+1-k] = 0.0f0
        end
        # ★ S3j: under k-eps, `tom`/`w_inf` carry EPSILON, so the far-field nu_t is Cmu k^2/eps, not
        # k/omega. Using k/omega here would inject a far-field eddy viscosity ~1e8 too small and the
        # freestream would silently go laminar.
        nutf = keps > 0.0f0 ? KE_CMU*k_inf*k_inf/max(w_inf, 1.0f-30) : k_inf/w_inf
        tk[i+g, NJ+g+k] = k_inf; tom[i+g, NJ+g+k] = w_inf; nut[i+g, NJ+g+k] = nutf
        tnl[i+g, NJ+g+k] = nutf
    end
end

@kernel function _k_sst_bc_outer_slip!(tk, tom, nut, tnl, @Const(outer_face_kind),
                                       NI::Int32, NJ::Int32, g::Int32)
    i, k = @index(Global, NTuple)
    @inbounds if i <= NI && k <= g && outer_face_kind[i] >= 0x02
        js = NJ + g + Int32(1) - k
        jd = NJ + g + k
        tk[i+g, jd] = tk[i+g, js]
        tom[i+g, jd] = tom[i+g, js]
        nut[i+g, jd] = nut[i+g, js]
        tnl[i+g, jd] = tnl[i+g, js]
    end
end

@kernel function _k_sst_bc_io!(tk, tom, nut, tnl, NI::Int32, NJ::Int32, g::Int32)
    jj, k = @index(Global, NTuple)
    @inbounds if jj <= NJ+2g && k <= g
        tk[g+1-k, jj] = tk[g+1, jj]; tom[g+1-k, jj] = tom[g+1, jj]; nut[g+1-k, jj] = nut[g+1, jj]
        tk[NI+g+k, jj] = tk[NI+g, jj]; tom[NI+g+k, jj] = tom[NI+g, jj]; nut[NI+g+k, jj] = nut[NI+g, jj]
        tnl[g+1-k, jj] = tnl[g+1, jj]; tnl[NI+g+k, jj] = tnl[NI+g, jj]
    end
end

# ★★★ S3j k-EPSILON RNG CONSTANTS (Yakhot & Orszag 1986; Yakhot, Orszag, Thangam, Gatski & Speziale
# 1992, Phys. Fluids A 4, 1510). These are the RNG values, NOT the standard k-eps ones -- Cmu is
# 0.0845 here versus the standard 0.09, and the inverse Prandtl numbers are 1.39 versus 1.0/1.3.
# Quoting standard constants for an "RNG" run is a common and invisible error, so they are named.
const KE_CMU   = 0.0845f0
const KE_C1    = 1.42f0
const KE_C2    = 1.68f0
const KE_ALPHA = 1.39f0      # alpha_k = alpha_eps, the RNG inverse Prandtl number for BOTH fields
const KE_ETA0  = 4.38f0
const KE_BETA  = 0.012f0

include(joinpath(@__DIR__, "..", "cmesh", "cmesh_sst_nonorth.jl"))

@inline _sst_wall_constraint_cell(i,j,NIW,NIF) = j <= 2 && NIW < i <= NIW+NIF

@kernel function _k_sst_sweep!(knew, wnew, nutl, tnlnew, @Const(tk), @Const(tom), @Const(nut), @Const(tnl),
                               @Const(rho), @Const(T), @Const(ux), @Const(uy), @Const(xc), @Const(yc),
                               @Const(Six), @Const(Siy), @Const(Sjx), @Const(Sjy), @Const(vol),
                               @Const(wdist), @Const(gammat), @Const(dtf), @Const(amarr), @Const(btke_src),
                               pvortl,
                               nu_inf, rho_inf, suth,
                               air_sutherland::Int32,
                               k_inf, w_inf, lag::Int32, lag_a0,
                               wilcox_xi, wilcox_mt0, keps,
                               irsj::Int32, NI::Int32, NJ::Int32, NIW::Int32, NIF::Int32, g::Int32,
                               corrected_wall_metric::Int32, nonorth::Int32,
                               @Const(Fik), @Const(Fiw), @Const(Fjk), @Const(Fjw), raw_equations::Int32)
    i, j = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ
        βs=0.09f0; a1=0.31f0
        σk1=0.85f0; σw1=0.5f0; β1=0.075f0; γ1=0.5532f0
        σk2=1.0f0;  σw2=0.856f0; β2=0.0828f0; γ2=0.4403f0
        ia=i+g; ja=j+g
        nu = _air_nu_local(rho[ia,ja],T[ia,ja],nu_inf,rho_inf,suth,air_sutherland)
        k0 = tk[ia,ja]; w0 = max(tom[ia,ja], 1.0f-6); d = max(wdist[i,j], 1.0f-8)
        xxi, yxi, xet, yet = _sst_index_metrics(
            xc, yc, Sjx, Sjy, wdist, i, j, NI, NJ, NIW, NIF, corrected_wall_metric)
        J = xxi*yet - xet*yxi; J = abs(J) < 1.0f-30 ? 1.0f-30 : J
        duxi = 0.5f0*(ux[ia+1,ja]-ux[ia-1,ja]); duet = 0.5f0*(ux[ia,ja+1]-ux[ia,ja-1])
        dvxi = 0.5f0*(uy[ia+1,ja]-uy[ia-1,ja]); dvet = 0.5f0*(uy[ia,ja+1]-uy[ia,ja-1])
        dudx = ( duxi*yet - duet*yxi)/J; dudy = (-duxi*xet + duet*xxi)/J
        dvdx = ( dvxi*yet - dvet*yxi)/J; dvdy = (-dvxi*xet + dvet*xxi)/J
        Sxy = 0.5f0*(dudy+dvdx); S2 = 2.0f0*(dudx*dudx + dvdy*dvdy + 2.0f0*Sxy*Sxy); Ssr = sqrt(S2)
        Ω = abs(dudy - dvdx)
        # ★★★ S3k -- Le & Tran (JAFM 2022) vortex threshold pressure, KINEMATIC part.
        # p_v* = p_v + C_v * mu_eff * |Omega|, and mu_eff = rho*(nu + nu_t). Store (nu + nu_t)*|Omega|
        # here -- both factors are already live for the SST production limiter one line below, so this
        # costs one add and one multiply and introduces NO new stencil or metric handling. The rho
        # factor and the C_v coefficient are applied by the CONSUMER (_k_tem_sweep!), exactly as the
        # Singhal branch does, so rho stays consistent with the density that sweep holds.
        # Written UNCONDITIONALLY so it is available as a diagnostic even when the source is off; with
        # `tem.pvvort = 0` nothing reads it and the physics is bitwise legacy.
        pvortl[ia, ja] = (nu + max(nut[ia, ja], 0.0f0)) * Ω
        sqk = sqrt(max(k0, 0.0f0))
        dkxi = 0.5f0*(tk[ia+1,ja]-tk[ia-1,ja]); dket = 0.5f0*(tk[ia,ja+1]-tk[ia,ja-1])
        dwxi = 0.5f0*(tom[ia+1,ja]-tom[ia-1,ja]); dwet = 0.5f0*(tom[ia,ja+1]-tom[ia,ja-1])
        dkdx = ( dkxi*yet - dket*yxi)/J; dkdy = (-dkxi*xet + dket*xxi)/J
        dwdx = ( dwxi*yet - dwet*yxi)/J; dwdy = (-dwxi*xet + dwet*xxi)/J
        gkw = dkdx*dwdx + dkdy*dwdy
        CDkw = max(2.0f0*σw2/w0*gkw, 1.0f-10)
        arg1 = min(max(sqk/(βs*w0*d), 500.0f0*nu/(d*d*w0)), 4.0f0*σw2*k0/(CDkw*d*d))
        arg1 = min(arg1, 4.0f0); F1 = tanh(arg1^4)
        arg2 = min(max(2.0f0*sqk/(βs*w0*d), 500.0f0*nu/(d*d*w0)), 4.0f0); F2 = tanh(arg2^2)
        # ★★★ S3j k-EPSILON RNG (Yakhot-Orszag 1992). keps > 0 reinterprets the SECOND transported
        # field `tom` as the DISSIPATION RATE epsilon instead of the specific rate omega, reusing the
        # existing advection / diffusion / point-implicit machinery unchanged. keps = 0 leaves every
        # expression below untouched => BITWISE legacy SST.
        #
        # WHY THIS CLOSURE SPECIFICALLY. Frikha, Coutier-Delgosha & Astolfi 2008 shed by a re-entrant
        # jet at St = 0.25 on a NACA66 foil at OUR EXACT alpha = 8 deg and Re = 7.96e5, running k-eps
        # RNG -- with a BAROTROPIC cavitation model strictly more slaved than ours. That makes the
        # closure a DIFFERENTIAL-REPLICATION variable against a solver known to work at our conditions,
        # which no other item on the list is.
        #
        # The RNG strain correction is the whole point: C2* = C2 + Cmu eta^3 (1 - eta/eta0)/(1 + beta
        # eta^3) with eta = S k/eps. For eta > eta0 = 4.38 -- i.e. in RAPIDLY STRAINED regions such as
        # the cavity closure shear layer -- the correction goes NEGATIVE, cutting epsilon destruction,
        # which RAISES epsilon, which LOWERS nu_t = Cmu k^2/eps. So RNG self-reduces eddy viscosity
        # exactly where the re-entrant jet forms. That is the same qualitative effect the Reboud
        # correction applies by hand, but derived rather than tuned, and keyed on STRAIN rather than
        # on void fraction.
        local nut_t, pkcap
        if keps > 0.0f0
            e0    = w0                                    # `tom` holds epsilon in this branch
            nut_t = KE_CMU*k0*k0/max(e0, 1.0f-12)
            pkcap = 10.0f0*e0                             # epsilon already carries units of k/time
        else
            nut_t = a1*k0/max(a1*w0, Ssr*F2)
            pkcap = 10.0f0*βs*k0*w0
        end
        Pk = min(nut_t*Ssr*Ω, pkcap) * gammat[i,j]
        # ★ S3f BUBBLE-INDUCED TKE. Written by the TEM sweep (the only place alpha, Rdot and R exist
        # together); identically zero when tem.btke = 0, so the k equation is bitwise legacy then.
        # Added AFTER the shear-production limiter on purpose: the limiter bounds the SHEAR cascade
        # against the local k*omega, and clipping an independent physical source with it would let a
        # well-stirred cell silently discard the bubble contribution. It is separately bounded below.
        Pk += min(btke_src[i,j], pkcap)
        σk = F1*σk1 + (1.0f0-F1)*σk2; σw = F1*σw1 + (1.0f0-F1)*σw2
        βc = F1*β1 + (1.0f0-F1)*β2;   γc = F1*γ1 + (1.0f0-F1)*γ2
        if keps > 0.0f0
            # RNG uses ONE inverse-Prandtl number for both fields (alpha_k = alpha_eps = 1.39), so the
            # SST blend is replaced wholesale rather than blended with.
            σk = KE_ALPHA; σw = KE_ALPHA
        end
        # diffusion: 4 faces unrolled (E, W, N, S); boundary faces use dn = vol/S
        Dk = 0.0f0; Dw = 0.0f0; difsum = 0.0f0
        # E face
        SxE = Six[i+1,j]; SyE = Siy[i+1,j]; SfE = sqrt(SxE*SxE+SyE*SyE)
        dnE = i+1 <= NI ? sqrt((xc[i+1,j]-xc[i,j])^2 + (yc[i+1,j]-yc[i,j])^2)+1.0f-30 : vol[i,j]/SfE+1.0f-30
        nfE = 0.5f0*(nut[ia,ja]+nut[ia+1,ja])
        if nonorth == Int32(1) && (i < NI)
            Dk += Fik[i+1,j]; Dw += Fiw[i+1,j]
        else
            Dk += (nu+σk*nfE)*SfE/dnE*(tk[ia+1,ja]-k0);  Dw += (nu+σw*nfE)*SfE/dnE*(tom[ia+1,ja]-w0)
        end
        difsum += (nu+σk*nfE)*SfE/dnE
        # W face
        SxW = Six[i,j]; SyW = Siy[i,j]; SfW = sqrt(SxW*SxW+SyW*SyW)
        dnW = i-1 >= 1 ? sqrt((xc[i-1,j]-xc[i,j])^2 + (yc[i-1,j]-yc[i,j])^2)+1.0f-30 : vol[i,j]/SfW+1.0f-30
        nfW = 0.5f0*(nut[ia,ja]+nut[ia-1,ja])
        if nonorth == Int32(1) && (i > 1)
            Dk += -Fik[i,j]; Dw += -Fiw[i,j]
        else
            Dk += (nu+σk*nfW)*SfW/dnW*(tk[ia-1,ja]-k0);  Dw += (nu+σw*nfW)*SfW/dnW*(tom[ia-1,ja]-w0)
        end
        difsum += (nu+σk*nfW)*SfW/dnW
        # N face
        SxN = Sjx[i,j+1]; SyN = Sjy[i,j+1]; SfN = sqrt(SxN*SxN+SyN*SyN)
        dnN = j+1 <= NJ ? sqrt((xc[i,j+1]-xc[i,j])^2 + (yc[i,j+1]-yc[i,j])^2)+1.0f-30 : vol[i,j]/SfN+1.0f-30
        nfN = 0.5f0*(nut[ia,ja]+nut[ia,ja+1])
        if nonorth == Int32(1) && (j < NJ)
            Dk += Fjk[i,j+1]; Dw += Fjw[i,j+1]
        else
            Dk += (nu+σk*nfN)*SfN/dnN*(tk[ia,ja+1]-k0);  Dw += (nu+σw*nfN)*SfN/dnN*(tom[ia,ja+1]-w0)
        end
        difsum += (nu+σk*nfN)*SfN/dnN
        # S face
        SxS = Sjx[i,j]; SyS = Sjy[i,j]; SfS = sqrt(SxS*SxS+SyS*SyS)
        if j > 1
            dnS = sqrt((xc[i,j-1]-xc[i,j])^2 + (yc[i,j-1]-yc[i,j])^2)+1.0f-30
        elseif i <= NIW || i > NIW+NIF
            # At the wake cut the ghost is the opposite leg's first cell.
            ip = NI+Int32(1)-i
            dnS = sqrt((xc[ip,1]-xc[i,1])^2 + (yc[ip,1]-yc[i,1])^2)+1.0f-30
        else
            dnS = vol[i,j]/SfS+1.0f-30
        end
        nfS = 0.5f0*(nut[ia,ja]+nut[ia,ja-1])
        if nonorth == Int32(1) && (j > 1 || i <= NIW || i > NIW+NIF)
            Dk += -Fjk[i,j]; Dw += -Fjw[i,j]
        else
            Dk += (nu+σk*nfS)*SfS/dnS*(tk[ia,ja-1]-k0);  Dw += (nu+σw*nfS)*SfS/dnS*(tom[ia,ja-1]-w0)
        end
        difsum += (nu+σk*nfS)*SfS/dnS
        Dk /= vol[i,j]; Dw /= vol[i,j]
        u0 = ux[ia,ja]; v0 = uy[ia,ja]
        Uci = (u0*0.5f0*(Six[i,j]+Six[i+1,j]) + v0*0.5f0*(Siy[i,j]+Siy[i+1,j]))/vol[i,j]
        Ucj = (u0*0.5f0*(Sjx[i,j]+Sjx[i,j+1]) + v0*0.5f0*(Sjy[i,j]+Sjy[i,j+1]))/vol[i,j]
        advk = (Uci >= 0.0f0 ? Uci*(k0-tk[ia-1,ja]) : Uci*(tk[ia+1,ja]-k0)) +
               (Ucj >= 0.0f0 ? Ucj*(k0-tk[ia,ja-1]) : Ucj*(tk[ia,ja+1]-k0))
        advw = (Uci >= 0.0f0 ? Uci*(w0-tom[ia-1,ja]) : Uci*(tom[ia+1,ja]-w0)) +
               (Ucj >= 0.0f0 ? Ucj*(w0-tom[ia,ja-1]) : Ucj*(tom[ia,ja+1]-w0))
        dtl = dtf[i,j]
        if nonorth == Int32(1)
            # Cross diffusion is explicit; retain a conservative scalar clock.
            dtl = min(dtl, 0.2f0*vol[i,j]/max(difsum,1.0f-30))
        end
        if irsj == Int32(1)   # SST SELF-CLOCK (mirrors plain): flow dtf is i-restricted under IRS-J
            dtl = min(dtl, 0.35f0/(abs(Uci)+abs(Ucj)+1.0f-30), 0.4f0*vol[i,j]/max(difsum,1.0f-30))
        end
        # ★ WILCOX-CLASS DILATATION DISSIPATION (2026-07-23, model-development direction A — the
        # 42dce76 lever, NOT the refuted Reboud direction: Reboud bites only in DENSE void; this
        # keys off the turbulent Mach number M_t^2 = 2k/a_m^2, which explodes exactly where the
        # MIXTURE sound speed collapses — self-localizing to the cavity/closure where the slow
        # re-entrant jet (the measured St deficit cause) lives. k-destruction augmentation only
        # (Sarkar/Wilcox form): beta* -> beta*(1 + xi*max(M_t^2 - M_t0^2, 0)). In pure liquid
        # a_m ~ 0.95, M_t ~ 0.05 << M_t0 = 0.25 => inert (bitwise when wilcox_xi = 0; near-inert
        # in the liquid when on). The omega-equation compensation (beta_c reduction) is the
        # documented follow-up variant, deliberately NOT included in the first probe.
        βs_eff = βs
        if wilcox_xi > 0.0f0
            amloc = amarr[ia,ja]
            mt2 = 2.0f0*max(k0, 0.0f0)/max(amloc*amloc, 1.0f-8)
            fmt = mt2 - wilcox_mt0*wilcox_mt0
            if fmt > 0.0f0
                βs_eff = βs*(1.0f0 + wilcox_xi*fmt)
            end
        end
        # ★ S3j: in k-eps the k sink is epsilon itself, written point-implicitly as (eps/k)*k so the
        # diagonal stays positive and the fixed point is unchanged.
        kn = keps > 0.0f0 ?
             (k0 + dtl*(Pk - advk + Dk)) / (1.0f0 + dtl*w0/max(k0, 1.0f-12)) :
             (k0 + dtl*(Pk - advk + Dk)) / (1.0f0 + dtl*βs_eff*w0)
        local wn
        if keps > 0.0f0
            # epsilon equation:  D(eps)/Dt = C1 (eps/k) Pk - C2* eps^2/k + diffusion
            # RNG strain correction, Yakhot-Orszag 1992 / Yakhot et al. 1992:
            #     eta = S k/eps ,  C2* = C2 + Cmu eta^3 (1 - eta/eta0) / (1 + beta eta^3)
            # eta > eta0 => the correction is NEGATIVE => less eps destruction => higher eps => lower
            # nu_t, self-reducing the eddy viscosity in rapidly strained regions. This is the term that
            # distinguishes RNG from standard k-eps and the reason this closure is on the list.
            kk  = max(k0, 1.0f-12)
            eta = Ssr*kk/max(w0, 1.0f-12)
            eta = eta > 1.0f3 ? 1.0f3 : eta            # guard the cubic against a stagnation spike
            e3  = eta*eta*eta
            C2s = KE_C2 + KE_CMU*e3*(1.0f0 - eta/KE_ETA0)/(1.0f0 + KE_BETA*e3)
            C2s = C2s < 0.0f0 ? 0.0f0 : C2s            # never let the sink become a source
            wn  = (w0 + dtl*(KE_C1*w0/kk*Pk - advw + Dw)) / (1.0f0 + dtl*C2s*w0/kk)
        else
            wn = (w0 + dtl*(γc/max(nut_t,1.0f-10)*Pk - advw + Dw + 2.0f0*(1.0f0-F1)*σw2/w0*gkw)) / (1.0f0 + dtl*βc*w0)
        end
        kn = clamp(kn, 1.0f-12, 1.0f0)
        wn = max(wn, 1.0f-6)
        if !(isfinite(kn) && isfinite(wn)); kn = k_inf; wn = w_inf; end
        if raw_equations == Int32(1)
            # Original primitive transport equations, before the point-implicit
            # update and positivity clipping. This diagnostic must not infer a
            # residual by subtracting rounded or clipped updated states.
            knew[ia,ja] = keps > 0.0f0 ?
                w0 + advk - Dk - Pk : βs_eff*k0*w0 + advk - Dk - Pk
            if keps > 0.0f0
                wnew[ia,ja] = C2s*w0*w0/kk + advw - Dw - KE_C1*w0/kk*Pk
            else
                wnew[ia,ja] = βc*w0*w0 + advw - Dw -
                    γc/max(nut_t,1.0f-10)*Pk - 2.0f0*(1.0f0-F1)*σw2/w0*gkw
            end
        else
            knew[ia,ja] = kn; wnew[ia,ja] = wn
        end
        # Menter SST eddy-viscosity DEFINITION nut = a1·k/max(a1·ω, S·F2) — the limiter must gate
        # the EXPORTED nut, not just production (mirrors the plain solver line-for-line)
        nueq = a1*kn/max(a1*wn, Ssr*F2)
        nutl[ia,ja] = nueq
        if lag == Int32(1)
            # LAG equation (Olsen-Coakley): D nu_hat/Dt = a0*omega*(nu_eq - nu_hat), a0=0.35 —
            # upwind advection on the contravariant velocities (Uci/Ucj above), point-implicit
            t0v = tnl[ia,ja]
            advt = (Uci >= 0.0f0 ? Uci*(t0v-tnl[ia-1,ja]) : Uci*(tnl[ia+1,ja]-t0v)) +
                   (Ucj >= 0.0f0 ? Ucj*(t0v-tnl[ia,ja-1]) : Ucj*(tnl[ia,ja+1]-t0v))
            wnc = max(wn, 1.0f-6)
            tn = (t0v + dtl*(lag_a0*wnc*nueq - advt)) / (1.0f0 + dtl*lag_a0*wnc)
            tnlnew[ia,ja] = clamp(tn, 0.0f0, 0.15f0)
        end
        if raw_equations == Int32(1)
            # Commit prescribes omega in these exact two interior wall rows.
            # There the discrete equation is the wall constraint, not the
            # transport PDE that the ordinary sweep subsequently discards.
            tnlnew[ia,ja] = wnew[ia,ja] # retain the unmasked PDE as a diagnostic
            wall_constraint = keps <= 0.0f0 && _sst_wall_constraint_cell(i,j,NIW,NIF)
            wall_omega = 6.0f0*nu/(β1*d*d)
            nutl[ia,ja] = wall_constraint ?
                (tom[ia,ja]-wall_omega)/max(abs(wall_omega),1.0f-6) : 0.0f0
            wall_constraint && (wnew[ia,ja] = 0.0f0)
        end
    end
end

# ★★ FILTER-BASED MODEL (FBM) — Johansen, Wu & Shyy 2004, "Filter-based unsteady RANS computations",
# Int. J. Heat Fluid Flow 25:10-21; applied to cavitation by Wu, Utturkar & Shyy and widely since.
#
# WHY IT IS HERE (2026-07-27). The grid-convergence study
# (memory/project_docs/2026-07-27_GRID_CONVERGENCE_SHEDDING_IS_NOT_CONVERGED.md) established that this solver's 2D cloud
# shedding COLLAPSES MONOTONICALLY under wall refinement (duty 89.8% -> 45.0% -> 4.7% -> 0.0% as y+ goes
# 8 -> 4 -> 2 -> 0.7) and survives ONLY on the two meshes whose dual-time residual EXCEEDS its own
# convergence control. A wall-resolved sigma-scan independently found NO sigma that sustains recurrent
# shedding. That is not a bug -- it is the documented limitation of the model CLASS: homogeneous URANS
# with a standard two-equation closure over-produces eddy viscosity in the cavity closure and damps the
# re-entrant jet. Reboud's density correction mitigates but does not cure it.
#
# FBM is the standard, cheapest literature remedy and was designed for exactly this flow. It rescales the
# eddy viscosity by the ratio of the local FILTER width to the turbulence length scale, so that where the
# grid can resolve the turbulence the model steps aside (LES-like) and where it cannot it reverts to RANS:
#
#     L_t = sqrt(k)/(beta_star*omega)        beta_star = 0.09
#     f   = min(1, (Delta/L_t)^(4/3))        Delta = local cell size = sqrt(vol) in 2D
#     nu_t <- nu_t * f
#
# ⚠ PUBLISHED CONSTANTS ONLY (beta_star=0.09, exponent 4/3, C_Delta=1). This is a MODEL-CLASS change, not
# constant-tuning, and it only counts as one if it is then subjected to the SAME grid-convergence and
# sigma-scan the URANS closure just failed. Do NOT tune C_Delta to produce shedding -- that would replace
# a coarse-mesh artifact with a model-tuning artifact.
# fbm_cdelta <= 0 disables it entirely => bitwise-identical to the pre-FBM solver.
@inline function _fbm_factor(kn, wn, vol_ij, cdelta)
    cdelta <= 0.0f0 && return 1.0f0
    Lt = sqrt(max(kn, 0.0f0))/max(0.09f0*wn, 1.0f-12)      # k^(1/2)/(beta* omega)
    Lt <= 1.0f-12 && return 1.0f0                           # no turbulence length scale => RANS
    dl = cdelta*sqrt(max(vol_ij, 0.0f0))                    # 2D cell size
    r  = dl/Lt
    r >= 1.0f0 && return 1.0f0                              # filter coarser than the eddies => full RANS
    return r^(4.0f0/3.0f0)
end

# ★★ PANS — PARTIALLY-AVERAGED NAVIER-STOKES with a PRESCRIBED CONSTANT f_k.
# Girimaji 2006, J. Appl. Mech. 73:413-421. Applied to cavitating hydrofoils by Ji, Luo, Huang, Wang
# et al. at f_k ~ 0.2-0.5.
#
# WHY IT IS HERE (2026-07-27), AND WHY NOT THE GRID-ADAPTIVE FORM. Grid convergence
# (memory/project_docs/2026-07-27_GRID_CONVERGENCE_SHEDDING_IS_NOT_CONVERGED.md) showed the URANS shedding does not
# survive wall refinement. FBM was the first literature remedy tried and is INERT inside the cavity
# (nu_t-weighted mean f = 0.9984, tools/probe_fbm_engagement.jl): its filter engages only where
# Delta < L_t, and in the cavity k is small => L_t is small => f = 1. Mis-SUITED, not mis-implemented.
#
# ⚠ The GRID-ADAPTIVE PANS coefficient f_k = min(1, Cmu^(-1/2)*(Delta/Lambda)^(2/3)) uses the SAME
# integral scale Lambda = k^(3/2)/eps = L_t that FBM uses, so it saturates to f_k = 1 in the cavity for
# EXACTLY the same reason and would be just as inert. It is deliberately NOT implemented -- doing so
# would have repeated the FBM failure at the cost of a full implementation. Only the PRESCRIBED
# constant f_k acts here, because it carries no length scale at all.
#
# DERIVATION. f_k = k_u/k, f_eps = eps_u/eps. Then
#     nu_u = Cmu*k_u^2/eps_u = (Cmu*k^2/eps) * f_k^2/f_eps = nu_t * f_k^2/f_eps.
# f_eps = 1 is the standard high-Re assumption (the dissipation is entirely unresolved), so the factor
# is simply f_k^2: 0.04 at f_k = 0.2, 0.25 at f_k = 0.5. Unlike Reboud (which keys off density and so
# bites only in DENSE void) and FBM (which keys off a length scale and so goes inert in the cavity),
# this acts UNIFORMLY -- including the mix/shear bins where the eddy viscosity actually peaks
# (nu_t/nu max 44.5 at 0.10<=a<0.50 and 40.5 at 0.01<=a<0.10) and where the re-entrant jet lives.
#
# ⚠⚠ REDUCED FORM -- THIS CAVEAT MUST TRAVEL WITH ANY RESULT. Full PANS also modifies the TRANSPORT
# coefficients (k-eps: C_eps2* = C_eps1 + (f_k/f_eps)*(C_eps2 - C_eps1); sigma_u = sigma*f_k^2/f_eps).
# That is NOT done here: this is a post-hoc rescale of nu_t on the same footing as Reboud and FBM, so
# k and omega still evolve with RANS coefficients and k_u is not literally the unresolved TKE. Note
# also that in _k_sst_sweep! the k-production Pk is built from a FRESHLY computed SST nu_t, not from
# the committed array -- so this rescale reaches the MOMENTUM equations and the turbulent diffusion of
# k/omega, but not Pk directly. Hence the interpretation is ASYMMETRIC:
#   - a NEGATIVE result is STRONG: the eddy viscosity seen by the momentum equations was cut up to 25x
#     exactly where the re-entrant jet lives, which approaches the nu_t -> 0 limit of the entire
#     "turbulence over-damps the jet" hypothesis. If shedding still does not appear, that hypothesis is
#     dead without needing the full transport form.
#   - a POSITIVE result does NOT license "PANS predicts shedding": a prescribed f_k is one step from a
#     uniform nu_t multiplier, so the full transport-coefficient form would have to be implemented and
#     re-swept before any such claim.
#
# ⚠ PUBLISHED VALUES ONLY. Run f_k = 0.2 AND f_k = 0.5 (the published bracket), report BOTH, tune
# NEITHER, and subject both to the SAME grid-convergence family and sigma-scan the URANS closure failed.
# Sweeping f_k until shedding appears would swap a coarse-mesh artifact for a model-tuning artifact and
# the result would be worthless.
# pans_fk <= 0 disables it entirely => bitwise-identical to the pre-PANS solver.
@inline function _pans_factor(fk)
    fk <= 0.0f0 && return 1.0f0                             # OFF => RANS, bitwise legacy
    f = fk >= 1.0f0 ? 1.0f0 : fk                            # f_k = 1 IS RANS by definition
    return f*f                                              # nu_t <- nu_t * f_k^2/f_eps, f_eps = 1
end

@kernel function _k_sst_commit!(tk, tom, nut, tnl, @Const(knew), @Const(wnew), @Const(nutl), @Const(tnlnew),
                                @Const(wdist), @Const(rho), @Const(T), @Const(vol),
                                nu_inf, rho_inf, suth,
                                air_sutherland::Int32,
                                lag::Int32, keps, reboud_n, reboud_rhov,
                                fbm_cdelta, pans_fk,
                                NI::Int32, NJ::Int32, NIW::Int32, NIF::Int32, g::Int32)
    i, j = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ
        β1=0.075f0
        ia=i+g; ja=j+g
        nu = _air_nu_local(rho[ia,ja],T[ia,ja],nu_inf,rho_inf,suth,air_sutherland)
        kn = knew[ia,ja]; wn = wnew[ia,ja]
        wallrow = _sst_wall_constraint_cell(i,j,NIW,NIF)
        if keps > 0.0f0
            # ★ S3j: nu_t = Cmu k^2/eps. `wn` holds epsilon here, so the SST wall-row analytic omega
            # and the a1 limiter (both omega-specific) do not apply and are bypassed. The near-wall
            # behaviour is instead whatever the epsilon equation plus the wall BC produce -- which is
            # why this closure wants a WALL FUNCTION, exactly as Frikha runs it, and not the
            # sublayer-resolving path.
            nut_t = KE_CMU*kn*kn/max(wn, 1.0f-12)
        elseif wallrow
            d = max(wdist[i,j], 1.0f-8)
            wn = 6.0f0*nu/(β1*d*d)
            nut_t = kn/max(wn, 1.0f-6)              # sublayer: a1·ω >> S·F2, limiter inert; use analytic ω
        else
            nut_t = nutl[ia,ja]                     # a1-limited (SST definition)
        end
        reb = 1.0f0
        if reboud_n > 0.0f0                          # Reboud 1998 cavitation eddy-viscosity reduction.
            r = rho[ia,ja]                           # a_v = void fraction from the (T-carrier-capped) density;
            av = (1.0f0 - r)/(1.0f0 - reboud_rhov); av = av < 0.0f0 ? 0.0f0 : (av > 1.0f0 ? 1.0f0 : av)
            reb = (1.0f0-av)^(reboud_n-1.0f0)        # large-density-ratio limit of f(rho)/rho: (1-a_v)^(n-1)
        end                                          # DECOUPLED from the 3:1 quadrature cap (a modeling choice)
        # FBM filter factor (1.0 when fbm_cdelta<=0 => bitwise legacy). Applied on the SAME footing as
        # Reboud: a post-hoc rescaling of nu_t, independent of how nu_t was obtained.
        fbm = _fbm_factor(kn, wn, vol[i,j], fbm_cdelta)
        # PANS constant-f_k rescale (1.0 when pans_fk<=0 => bitwise legacy). Same footing again: a
        # post-hoc nu_t factor. Composes multiplicatively with Reboud and FBM by construction --
        # Reboud bites in dense void, FBM (when non-inert) near the wall, PANS uniformly.
        pans = _pans_factor(pans_fk)
        nut_t *= reb*fbm*pans
        tk[ia,ja] = kn; tom[ia,ja] = wn
        if lag == Int32(1) && !wallrow
            tnl[ia,ja] = tnlnew[ia,ja]
            # BUGFIX 2026-07-19: the Reboud factor was computed into nut_t and then DISCARDED on this
            # branch, so with lag==1 the cavitation eddy-viscosity reduction was silently inert over the
            # whole non-wallrow field -- i.e. exactly the cavity. It is a post-hoc density correction to
            # nu_t and applies regardless of how nu_t was obtained. reb == 1.0f0 when reboud_n == 0, so
            # the non-cavitating path stays bitwise identical.
            nut[ia,ja] = tnlnew[ia,ja]*reb*fbm*pans # lagged nu_t drives the stresses directly (FBM/PANS too)
        else
            lag == Int32(1) && (tnl[ia,ja] = kn/max(wn, 1.0f-6))   # sublayer: lag inert
            nut[ia,ja] += 0.2f0*(min(nut_t, 0.15f0) - nut[ia,ja])
        end
    end
end

# WJ-EARSM stage: effective nu_t = -(beta1/2) k tau_t (overrides the commit export) + the extra
# anisotropy stress tau_extra = -rho k beta4 (S*Om*-Om*S*) into exx/exy/eyy (mirrors plain).
@kernel function _k_earsm!(nut, exx, exy, eyy, @Const(tk), @Const(tom), @Const(rho),
                           @Const(ux), @Const(uy), @Const(xc), @Const(yc),
                           @Const(Sjx), @Const(Sjy), @Const(wdist),
                           nu, NI::Int32, NJ::Int32,
                           NIW::Int32, NIF::Int32, g::Int32,
                           corrected_wall_metric::Int32)
    i, j = @index(Global, NTuple)
    @inbounds if i <= NI && j <= NJ
        βst = 0.09f0
        ia=i+g; ja=j+g
        xxi, yxi, xet, yet = _sst_index_metrics(
            xc, yc, Sjx, Sjy, wdist, i, j, NI, NJ, NIW, NIF, corrected_wall_metric)
        J = xxi*yet - xet*yxi; J = abs(J) < 1.0f-30 ? 1.0f-30 : J
        duxi = 0.5f0*(ux[ia+1,ja]-ux[ia-1,ja]); duet = 0.5f0*(ux[ia,ja+1]-ux[ia,ja-1])
        dvxi = 0.5f0*(uy[ia+1,ja]-uy[ia-1,ja]); dvet = 0.5f0*(uy[ia,ja+1]-uy[ia,ja-1])
        dudx = ( duxi*yet - duet*yxi)/J; dudy = (-duxi*xet + duet*xxi)/J
        dvdx = ( dvxi*yet - dvet*yxi)/J; dvdy = (-dvxi*xet + dvet*xxi)/J
        k0 = max(tk[ia,ja], 1.0f-12); w0 = max(tom[ia,ja], 1.0f-6)
        tau_t = max(1.0f0/(βst*w0), 6.0f0*sqrt(nu/(βst*k0*w0)))
        b1, b4, sxx, syy, sxy, oxy = _wj_earsm_2d(dudx, dudy, dvdx, dvdy, tau_t)
        # explicit-EARSM stabilization (mirrors plain): 0.2 under-relax + |tau_extra| <= 0.6 rho k
        nut[ia,ja] += 0.2f0*(clamp(-0.5f0*b1*k0*tau_t, 0.0f0, 0.15f0) - nut[ia,ja])
        ck = rho[ia,ja]*k0*b4; bnd = 0.6f0*rho[ia,ja]*k0
        exx[i,j] += 0.2f0*(clamp( 2.0f0*ck*sxy*oxy, -bnd, bnd) - exx[i,j])
        exy[i,j] += 0.2f0*(clamp(-ck*(sxx-syy)*oxy, -bnd, bnd) - exy[i,j])
        eyy[i,j] += 0.2f0*(clamp(-2.0f0*ck*sxy*oxy, -bnd, bnd) - eyy[i,j])
    end
end

function sst_step_ka!(s::KSol; reboud_n=0.0f0, reboud_rhov=0.33f0,
                      wilcox_xi=0.0f0, wilcox_mt0=0.25f0,
                      corrected_wall_metric::Bool=false,
                      nonorth_diffusion::Bool=false, raw_equations::Bool=false)
    s.air_sutherland != Int32(0) && _check_air_sutherland_admission(s)
    s.sst || return
    nonorth_diffusion && !corrected_wall_metric &&
        throw(ArgumentError("nonorthogonal SST diffusion requires corrected_wall_metric=true"))
    b = s.backend; NI, NJ, g = Int32(s.NI), Int32(s.NJ), Int32(s.g)
    _k_sst_bc!(b)(s.tk, s.tom, s.nut, s.tnl, s.k_inf, s.w_inf, s.keps, NI, NJ, Int32(s.NIW), Int32(s.NIF), g; ndrange=(s.NI, s.g))
    if s.outer_mode != Int32(0)
        _k_sst_bc_outer_slip!(b)(s.tk, s.tom, s.nut, s.tnl, s.outer_face_kind,
                                 NI, NJ, g; ndrange=(s.NI, s.g))
    end
    _k_sst_bc_io!(b)(s.tk, s.tom, s.nut, s.tnl, NI, NJ, g; ndrange=(s.NJ+2s.g, s.g))
    if nonorth_diffusion
        _ensure_sst_nonorth_scratch!(s)
        _k_sst_nonorth_prep!(b)(s.sst_gkx,s.sst_gky,s.sst_gwx,s.sst_gwy,
            s.sst_sigk,s.sst_sigw,s.sst_nu,
            s.tk,s.tom,s.rho,s.T,s.xc,s.yc,s.Sjx,s.Sjy,s.wdist,
            s.nu_lam,s.rho_inf,s.suth,s.air_sutherland,s.keps,
            NI,NJ,Int32(s.NIW),Int32(s.NIF),g,Int32(corrected_wall_metric);
            ndrange=(s.NI,s.NJ))
        _k_sst_nonorth_i!(b)(s.sst_Fik,s.sst_Fiw,
            s.tk,s.tom,s.nut,s.sst_gkx,s.sst_gky,s.sst_gwx,s.sst_gwy,
            s.sst_sigk,s.sst_sigw,s.sst_nu,s.xc,s.yc,s.Six,s.Siy,
            NI,NJ,g;ndrange=(s.NI+1,s.NJ))
        _k_sst_nonorth_j!(b)(s.sst_Fjk,s.sst_Fjw,
            s.tk,s.tom,s.nut,s.sst_gkx,s.sst_gky,s.sst_gwx,s.sst_gwy,
            s.sst_sigk,s.sst_sigw,s.sst_nu,s.xc,s.yc,s.Sjx,s.Sjy,
            NI,NJ,Int32(s.NIW),Int32(s.NIF),g;ndrange=(s.NI,s.NJ+1))
    end
    _k_sst_sweep!(b)(s.rk_r, s.rk_mx, s.rk_my, s.rk_E, s.tk, s.tom, s.nut, s.tnl,
                     s.rho, s.T, s.ux, s.uy, s.xc, s.yc,
                     s.Six, s.Siy, s.Sjx, s.Sjy, s.vol, s.wdist, s.gammat, s.dtf, s.am, s.btke_src,
                     s.pvort,
                     s.nu_lam, s.rho_inf, s.suth, s.air_sutherland,
                     s.k_inf, s.w_inf, s.lag, s.lag_a0, wilcox_xi, wilcox_mt0, s.keps,
                     s.irsj, NI, NJ, Int32(s.NIW), Int32(s.NIF), g,
                     Int32(corrected_wall_metric),Int32(nonorth_diffusion),
                     s.sst_Fik,s.sst_Fiw,s.sst_Fjk,s.sst_Fjw,Int32(raw_equations);
                     ndrange=(s.NI, s.NJ))
    raw_equations && return (k=s.rk_r,omega=s.rk_mx,
        omega_wall_constraint_relative=s.rk_my,omega_all_transport=s.rk_E)
    _k_sst_commit!(b)(s.tk, s.tom, s.nut, s.tnl, s.rk_r, s.rk_mx, s.rk_my, s.rk_E,
                      s.wdist, s.rho, s.T, s.vol,
                      s.nu_lam, s.rho_inf, s.suth, s.air_sutherland,
                      s.lag, s.keps, reboud_n, reboud_rhov, s.fbm_cdelta, s.pans_fk, NI, NJ, Int32(s.NIW), Int32(s.NIF), g; ndrange=(s.NI, s.NJ))
    if s.earsm == Int32(1)
        _k_earsm!(b)(s.nut, s.exx, s.exy, s.eyy, s.tk, s.tom, s.rho, s.ux, s.uy, s.xc, s.yc,
                     s.Sjx, s.Sjy, s.wdist, s.nu_lam, NI, NJ,
                     Int32(s.NIW), Int32(s.NIF), g,
                     Int32(corrected_wall_metric); ndrange=(s.NI, s.NJ))
    end
    nothing
end

# ---- host-side loads (small row pulls; sync happens inside Array copies) -----------------------
function loads_ka(s::KSol;mixture_pressure=nothing,wallmode::Int32=Int32(2))
    CS2=_gas_cs2(eltype(s.rho))
    if s.air_sutherland != Int32(0)
        _check_air_sutherland_admission(s)
        (mixture_pressure === nothing && wallmode == Int32(2)) ||
            throw(ArgumentError("air Sutherland loads require the no-slip gas path"))
    end
    mixture_pressure===nothing || CMeshPressureOutlet.check_pressure_array(s,mixture_pressure)
    g = s.g; r = s.NIW+1+g : s.NIW+s.NIF+g
    KA.synchronize(s.backend)
    rw = Array(view(s.rho, r, g+1)); Tw = Array(view(s.T, r, g+1))
    uw = Array(view(s.ux, r, g+1));  vw = Array(view(s.uy, r, g+1))
    # PDP wall excess pressure. MUST be `>= 1` (cav is a LEVEL: 1 = split, 2 = +Layer C), not `== 1`.
    # BUG FIXED 2026-07-18: the `== Int32(1)` form silently dropped pexc from the force integral at the
    # PRODUCTION setting cav=2, so the wall saw only the clamped pedestal rho*cs2*T_band. Inside a
    # 1000:1 cavity that pedestal is ~0.002 against a true plateau p_v~0.648 => a phantom Cp ~ -35 over
    # the whole cavity. That is the Cl~8 "TEM stiffness blow-up" recorded in the Stage-3 handover: the
    # DIAGNOSTIC was broken, not (only) the EOS. Stage-1 gates ran at cav=1 and were unaffected, which is
    # why the -Cp plateau validated while the forces did not.
    pxw = s.cav >= Int32(1) ? Array(view(s.pexc, r, g+1)) : nothing
    pmw = mixture_pressure===nothing ? nothing : Array(view(mixture_pressure,r,g+1))
    Fx = 0.0; Fy = 0.0
    for (n, i) in enumerate(s.NIW+1:s.NIW+s.NIF)
        pw = pmw===nothing ? Float64(rw[n])*Float64(CS2)*Float64(Tw[n]) + (pxw === nothing ? 0.0 : Float64(pxw[n])) : Float64(pmw[n])
        Fx -= pw*Float64(s.Sjx1_h[i]); Fy -= pw*Float64(s.Sjy1_h[i])
        if s.nu_lam > 0.0f0 && (pmw===nothing || wallmode!=Int32(1))
            Sf = sqrt(Float64(s.Sjx1_h[i])^2 + Float64(s.Sjy1_h[i])^2)
            d1 = 0.5*Float64(s.vol1_h[i])/Sf + 1e-30
            muw = s.air_sutherland != Int32(0) ?
                  Float64(s.rho_inf)*Float64(s.nu_lam)*
                  Float64(_mufac(Tw[n],s.suth))*Sf/d1 :
                  Float64(rw[n])*Float64(s.nu_lam)*Sf/d1
            if pmw!==nothing && s.wallfun > 1.5f0
                # Species all-y+ wall function (26 Sep 2026): the same Reichardt wall shear the flux kernel
                # applies (`_k_flux_j!`, `wallfun > 1.5`), evaluated in Float64 on the same Float32 state.
                nxw = Float64(s.Sjx1_h[i])/Sf; nyw = Float64(s.Sjy1_h[i])/Sf
                u1 = Float64(uw[n]); v1 = Float64(vw[n]); un1 = u1*nxw + v1*nyw
                utx = u1 - un1*nxw; uty = v1 - un1*nyw; ut1 = sqrt(utx*utx + uty*uty)
                if ut1 > 0.0
                    utau, _, _ = wf_species_state(ut1, d1, Float64(s.nu_lam))
                    tauw = Float64(rw[n])*utau*utau
                    Fx += tauw*Sf*utx/ut1; Fy += tauw*Sf*uty/ut1
                end
            elseif pmw===nothing && s.cav==Int32(0) && s.fullvisc==Int32(1) && s.wallfun==0.0f0
                # Match the resolved thermal-gas wall flux, including its molecular viscosity.
                s.air_sutherland==Int32(0) && (muw *= Float64(_mufac(Tw[n],s.suth)))
                tx,ty=_wall_stokes_traction(Float64(uw[n]),Float64(vw[n]),
                    Float64(s.Sjx1_h[i])/Sf,Float64(s.Sjy1_h[i])/Sf,muw)
                Fx += tx; Fy += ty
            elseif pmw===nothing
            Fx += muw*Float64(uw[n]); Fy += muw*Float64(vw[n])
            else
                tx,ty=_wall_stokes_traction(Float64(uw[n]),Float64(vw[n]),
                    Float64(s.Sjx1_h[i])/Sf,Float64(s.Sjy1_h[i])/Sf,muw)
                Fx += tx; Fy += ty
            end
        end
    end
    U2 = Float64(s.uxi)^2 + Float64(s.uyi)^2
    q = 0.5*Float64(s.rho_inf)*U2
    Cl = ( Fy*Float64(s.ca) - Fx*Float64(s.sa))/q
    Cd = ( Fx*Float64(s.ca) + Fy*Float64(s.sa))/q
    (Cl, Cd)
end

function residual_linf_ka(s::KSol)
    KA.synchronize(s.backend)
    maximum(abs, s.dc_r)
end

# ★ VOLUME-SCALED STEADY RESIDUAL — use THIS to judge convergence, not residual_linf_ka.
#
# residual_linf_ka returns max|dc_r|, the RAW flux divergence. dc_r has units of (mass flux x area), so on
# a C-mesh stretched from dn1=2e-4 at the wall to cells out at R_ff=25c it is dominated by the ENORMOUS
# far-field cells, while the near-wall cells -- where the physics and the forces live -- contribute a
# negligible amount even when they are badly out of balance. Dividing by the cell volume gives d(rho)/dt,
# which is mesh-independent and actually comparable between cells. (The dual-time rstar at _k_rstar!
# already does exactly this; the steady residual never got the same treatment.)
#
# WHY IT MATTERS, measured 2026-07-20 on the non-cavitating NACA0015 at 8 deg: residual_linf_ka read
# ~1.9e-05 at BOTH 40k iterations (Cl=-0.2443, Cd=-0.10274 -- a NEGATIVE drag, i.e. not a solution at
# all) and 60k (Cl=+1.3372, Cd=+0.16892). A metric that cannot separate those two states is not a
# convergence criterion.
function residual_scaled_ka(s::KSol)
    KA.synchronize(s.backend)
    maximum(abs.(s.dc_r) ./ s.vol)
end

# ★ L-inf ALONE CANNOT TELL "GLOBALLY DIVERGING" FROM "ONE BAD CELL".
# residual_scaled_ka is a MAX, so a single pathological cell owns it -- and in a cavity, where rho falls
# to ~0.01 inside tiny near-wall cells, that is exactly what one expects. The cavitating runs show this
# metric growing to 1e+02..1e+03 while the forces stay bounded and plausible (Cl ~ 0.6-1.0, Cd > 0), which
# is itself a hint that the imbalance is LOCAL rather than global.
# Returns (linf, l2, frac_above, imax, jmax):
#   l2         = RMS of |dc_r|/vol -- insensitive to a single cell, so l2 << linf means LOCALISED
#   frac_above = fraction of cells within 1% of the max -- tiny => a handful of cells, not the field
#   imax,jmax  = WHERE the max lives, so it can be matched against the cavity/interface/closure
function residual_report_ka(s::KSol)
    KA.synchronize(s.backend)
    r = Array(abs.(s.dc_r) ./ s.vol)
    linf = maximum(r)
    l2 = sqrt(sum(abs2, r)/length(r))
    frac = count(>=(0.99*linf), r)/length(r)
    idx = argmax(r)
    (linf, l2, frac, idx[1], idx[2])
end

# Experimental steady airfoil flow iteration. Kept separate from the explicit
# physical-time and cavitation entry points.
include(joinpath(@__DIR__, "cmesh_geometric_reconstruction_ka.jl"))
include(joinpath(@__DIR__, "cmesh_steady_implicit_ka.jl"))
include(joinpath(@__DIR__, "cmesh_steady_jfnk_ka.jl"))
include(joinpath(@__DIR__, "cmesh_steady_krylov_ka.jl"))

end # module
