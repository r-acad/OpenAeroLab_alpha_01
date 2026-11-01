# ---------------------------------------------------------------------------------------------
# PATH STATUS: **NOT PRODUCTION** -- the original CPU CMESH2D core, superseded by cmesh_solver_ka.jl.
# Preserved under the working rule 'preserve old solvers and experiments'. It is NOT
# reachable from the production driver, so results obtained here are NOT statements about
# the production path. Map: docs/CAVITATION_CODE_MAP.md.
# ---------------------------------------------------------------------------------------------
# ============================================================================================
# CMESH2D solver core (A1, 2026-07-14 — memory/project_docs/2026-07-14_CMESH2D_solver_plan.md §2-3)
# Curvilinear FV-GKS on the body-fitted C-mesh: ROTATED-frame KFVS kinetic flux (the pond
# _fvflux2T 39-velocity upwind quadrature evaluated in the face-normal frame — the ONLY
# axis-dependent operation in the donor is vn = ax==1 ? vx : vy, so rotation is exact w.r.t.
# the quadrature's moment-exactness order), minmod reconstruction (v1: the validated pond
# default; the s=1 branch — WENO5-Z/gauge/DUGKS/SGS deliberately NOT ported yet), SSP-RK3 in
# LTS pseudo-time. FACE-ARRAY fluxes (each face computed once => conservative by construction,
# unlike the donor's per-cell recomputation). v1 BCs = the gate-(a) set: freestream far-field,
# zero-gradient outlet, wake-cut exchange, TRANSPARENT wall stub (real wall lands in A2).
# Units: pond lattice units — cs2=2/3, R=cs2, p=rho*cs2*T, eint=0.5*(3+drot)*cs2, drot=2 (air).
# ============================================================================================

module CMeshSolver

using KernelAbstractions

export CSol, build_sol, set_freestream!, step!, residual_linf, gate_freestream, loads, set_sst!, sst_step!, hist_cycle!, sfd_seed!, outer_bc_override!, outer_flux_override!, seed_channel_reference!, clear_channel_reference!

include(joinpath(@__DIR__, "..", "cmesh", "cmesh_flux_core.jl"))

# ---- solver state --------------------------------------------------------------------------
mutable struct CSol{A2, V, M}
    NI::Int; NJ::Int; NIW::Int; NIF::Int; g::Int
    # metrics (interior-sized)
    Six::A2; Siy::A2; Sjx::A2; Sjy::A2; vol::A2; xc::A2; yc::A2
    # state, ghosted (NI+2g)x(NJ+2g)
    rho::A2; ux::A2; uy::A2; T::A2
    rk_r::A2; rk_mx::A2; rk_my::A2; rk_E::A2       # RK stage-0 conserved snapshot
    # face flux arrays: i-faces (NI+1)xNJ, j-faces NIx(NJ+1)
    Fi_r::A2; Fi_mx::A2; Fi_my::A2; Fi_e::A2
    Fj_r::A2; Fj_mx::A2; Fj_my::A2; Fj_e::A2
    dc_r::A2; dc_mx::A2; dc_my::A2; dc_e::A2       # residual sum(F·S) per cell (interior-sized)
    dtf::A2                                        # LTS local pseudo-dt (interior-sized)
    # lattice
    cxa::V; cya::V; cza::V; wa::V
    drot::Float32; eint::Float32
    # freestream (BC targets)
    rho_inf::Float32; uxi::Float32; uyi::Float32; T_inf::Float32
    # point-vortex far-field correction (Γ clockwise-positive at (xv,yv); PG compressible angular form)
    gam::Float32; xv::Float32; yv::Float32
    m2::Float32; beta::Float32                     # M∞², √(1−M∞²) — set by set_freestream!
    ca::Float32; sa::Float32                       # cos/sin α (the sin²θ term needs the freestream frame)
    wall_flux::Bool                                # true ⇒ analytic p_w wall flux on the foil j=1 faces
    nu_lam::Float32                                # laminar kinematic viscosity (chord units: U·c/Re); 0 ⇒ inviscid
    # low-Re SST k-ω (A5): ghosted transport fields + cell ν_t; sst=false ⇒ 1×1 stubs, laminar
    tk::A2; tom::A2; nut::A2
    tnl::A2                                        # LAGGED eddy viscosity (Olsen-Coakley; lag==1)
    sst::Bool; k_inf::Float32; w_inf::Float32
    lag::Int32                                     # 1 = nu_t lag equation drives the stresses
    lag_a0::Float32                                # lag rate coefficient (Olsen-Coakley 0.35)
    earsm::Int32                                   # 1 = WJ-EARSM: effective nu_t + extra anisotropy stress
    exx::A2; exy::A2; eyy::A2                      # EARSM extra stress tau_extra (interior-sized)
    wdist::A2; gammat::A2                          # wall distance + trip intermittency (interior-sized)
    recon::Int32                                   # 0 = minmod (validated default), 1 = WENO5-Z + sensor blend
    fullvisc::Int32                                # 0 = thin-layer (legacy bitwise), 1 = full stress tensor (A2c)
    sponge_sigma::Float32                          # far-field acoustic sponge strength (0 = off, bitwise legacy)
    sponge_j0f::Float32                            # sponge start as fraction of NJ (default 0.75)
    suth::Float32                                  # Sutherland S/T_inf for mu(T) (0 = off, bitwise legacy)
    rhok::Int32                                    # 1 = include -2/3 rho k in the turbulent stress
    gux::A2; guy::A2; gvx::A2; gvy::A2; gTx::A2; gTy::A2   # Green-Gauss cell gradients (interior-sized)
    # dual-time BDF2 history (conservative, interior-sized): U^n and U^(n-1); inert unless step! gets dtp>0
    un_r::A2; un_mx::A2; un_my::A2; un_E::A2
    um_r::A2; um_mx::A2; um_my::A2; um_E::A2
    # j-line implicit residual smoothing (IRS-J): dt from the STREAMWISE restriction only; the
    # wall-normal acoustic+viscous stiffness (cell AR ~1000) absorbed by per-column tridiagonal
    # solves on the residual (variable-coefficient Jameson/Martinelli IRS). 0 = off (bitwise legacy).
    irsj::Int32
    irs_c::A2; irs_d::A2                           # Thomas-algorithm scratch (interior-sized)
    # Selective Frequency Damping (Akervik 2006): converges UNSTABLE steady states by damping
    # oscillations about a low-pass-filtered state: u -= chi*(u - ubar); ubar += alpha*(u - ubar).
    # THE fine-mesh steady enabler (breathing/grey-zone modes). 0 = off (bitwise legacy).
    sfd::Int32
    sfd_chi::Float32                               # damping gain per step (~1e-2)
    sfd_alpha::Float32                             # filter update rate per step (~1e-3 = ~1000-step memory)
    sb_r::A2; sb_u::A2; sb_v::A2; sb_T::A2         # filtered primitives (ghost-sized for simplicity of copy)
    backend::Any
    # Outer-boundary segmentation. 0=far field, 1=inlet, 2=lower slip wall, 3=upper slip wall.
    outer_face_kind::M
    outer_mode::Int32                              # 0 = legacy all-far-field path, 1 = segmented channel
    outer_ref_r::A2; outer_ref_mx::A2; outer_ref_my::A2; outer_ref_e::A2
    outer_ref_rho::Float32; outer_ref_ux::Float32; outer_ref_uy::Float32; outer_ref_T::Float32
    outer_ref_ready::Int32
    outer_ref_signature::UInt64
end

function _outer_boundary_contract(met, NI::Int)
    kinds = hasproperty(met, :outer_face_kind) ?
            UInt8.(collect(met.outer_face_kind)) : fill(UInt8(0), NI)
    length(kinds) == NI || throw(ArgumentError("outer_face_kind must have NI=$NI entries"))
    provenance = hasproperty(met, :channel_provenance) ? met.channel_provenance : false
    provenance isa Bool || throw(ArgumentError("channel_provenance must be Bool"))
    if !provenance
        all(iszero, kinds) || throw(ArgumentError(
            "nonzero outer_face_kind requires channel-provenance metrics from build_metrics(::ChannelMesh)"))
        return kinds, Int32(0)
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

function build_sol(backend, met; drot::Float32=2.0f0, cxv, cyv, czv, wv)
    NI, NJ, g = met.NI, met.NJ, 3
    dev(x) = adapt_arr(backend, x)
    z(nx, ny) = dev(zeros(Float32, nx, ny))
    outer_face_kind_h, outer_mode = _outer_boundary_contract(met, NI)
    zr() = outer_mode == Int32(0) ? z(1, 1) : z(NI, NJ)
    CSol(NI, NJ, met.NIW, met.NIF, g,
        dev(met.Six), dev(met.Siy), dev(met.Sjx), dev(met.Sjy), dev(met.vol), dev(met.xc), dev(met.yc),
        z(NI+2g, NJ+2g), z(NI+2g, NJ+2g), z(NI+2g, NJ+2g), z(NI+2g, NJ+2g),
        z(NI+2g, NJ+2g), z(NI+2g, NJ+2g), z(NI+2g, NJ+2g), z(NI+2g, NJ+2g),
        z(NI+1, NJ), z(NI+1, NJ), z(NI+1, NJ), z(NI+1, NJ),
        z(NI, NJ+1), z(NI, NJ+1), z(NI, NJ+1), z(NI, NJ+1),
        z(NI, NJ), z(NI, NJ), z(NI, NJ), z(NI, NJ),
        z(NI, NJ),
        dev(Float32.(cxv)), dev(Float32.(cyv)), dev(Float32.(czv)), dev(Float32.(wv)),
        drot, 0.5f0*(3.0f0+drot)*CS2,
        1.0f0, 0.0f0, 0.0f0, 1.0f0,
        0.0f0, 0.25f0, 0.0f0,                       # Γ=0 (off) anchored at the quarter-chord
        0.0f0, 1.0f0, 1.0f0, 0.0f0, false, 0.0f0,
        z(NI+2g, NJ+2g), z(NI+2g, NJ+2g), z(NI+2g, NJ+2g),   # tk, tom, nut (allocated; inert unless sst)
        z(NI+2g, NJ+2g),                                      # tnl (lagged nut)
        false, 0.0f0, 0.0f0, Int32(0), 0.35f0,
        Int32(0), z(NI, NJ), z(NI, NJ), z(NI, NJ),
        dev(met.wdist), dev(met.gammat), Int32(0), Int32(0), 0.0f0, 0.75f0, 0.0f0, Int32(0),
        z(NI, NJ), z(NI, NJ), z(NI, NJ), z(NI, NJ), z(NI, NJ), z(NI, NJ),
        z(NI, NJ), z(NI, NJ), z(NI, NJ), z(NI, NJ),
        z(NI, NJ), z(NI, NJ), z(NI, NJ), z(NI, NJ),
        Int32(0), z(NI, NJ), z(NI, NJ),
        Int32(0), 0.01f0, 0.001f0,
        z(NI+2g, NJ+2g), z(NI+2g, NJ+2g), z(NI+2g, NJ+2g), z(NI+2g, NJ+2g),
        backend, dev(outer_face_kind_h), outer_mode,
        zr(), zr(), zr(), zr(), 0.0f0, 0.0f0, 0.0f0, 0.0f0, Int32(0), UInt64(0))
end
adapt_arr(::CPU, x) = copy(x)                       # CPU v1; GPU adapt lands with the KA-GPU pass

@inline _channel_sigword(x::Float32) = UInt64(reinterpret(UInt32, x))
@inline _channel_sigword(x::Int32) = UInt64(reinterpret(UInt32, x))
@inline _channel_sigword(x::Int) = UInt64(x)
@inline _channel_sigword(x::Bool) = x ? UInt64(1) : UInt64(0)

function _channel_signature(s::CSol)
    h = UInt64(0xcbf29ce484222325)
    for x in (s.NI, s.NJ, s.NIW, s.NIF, s.g, s.outer_mode,
              s.drot, s.rho_inf, s.uxi, s.uyi, s.T_inf,
              s.m2, s.beta, s.ca, s.sa,
              s.nu_lam, s.sst, s.k_inf, s.w_inf,
              s.earsm, s.recon, s.fullvisc, s.suth, s.rhok)
        h = xor(h, _channel_sigword(x))*UInt64(0x100000001b3)
    end
    h
end

function clear_channel_reference!(s::CSol)
    s.outer_ref_ready = Int32(0)
    s.outer_ref_signature = UInt64(0)
    nothing
end

function _channel_reference_ambient(s::CSol)
    s.outer_mode != Int32(0) || throw(ArgumentError("channel reference seeding requires segmented channel boundaries"))
    isfinite(s.rho_inf) && s.rho_inf > 0.0f0 || throw(ArgumentError("channel reference requires rho_inf > 0"))
    all(isfinite, (s.uxi,s.uyi,s.T_inf)) && s.T_inf > 0.0f0 ||
        throw(ArgumentError("channel reference freestream is invalid"))
    kref=0.0f0; nutref=0.0f0
    if s.sst
        isfinite(s.k_inf) && s.k_inf >= 0.0f0 || throw(ArgumentError("channel SST reference requires k_inf >= 0"))
        isfinite(s.w_inf) && s.w_inf > 0.0f0 || throw(ArgumentError("channel SST reference requires w_inf > 0"))
        kref=s.k_inf; nutref=s.k_inf/s.w_inf
    end
    (rho=s.rho_inf,ux=s.uxi,uy=s.uyi,T=s.T_inf,tk=kref,nut=nutref)
end

function set_freestream!(s::CSol, mach::Float64, alpha_deg::Float64; pg_correction::Bool=true)
    a = sqrt((1 + 2/ (3+s.drot)) * CS2)             # gamma*R with gamma=(Deff+2)/Deff, R=cs2 => a=sqrt(gamma*cs2)
    U = Float32(mach*a)
    s.ca = Float32(cosd(alpha_deg)); s.sa = Float32(sind(alpha_deg))
    s.uxi = U*s.ca; s.uyi = U*s.sa
    s.rho_inf = 1.0f0; s.T_inf = 1.0f0
    s.m2 = pg_correction ? Float32(mach^2) : 0.0f0
    s.beta = pg_correction ? Float32(sqrt(max(1 - mach^2, 1e-6))) : 1.0f0
    fill!(s.rho, s.rho_inf); fill!(s.ux, s.uxi); fill!(s.uy, s.uyi); fill!(s.T, s.T_inf)
    clear_channel_reference!(s)
    nothing
end

# PG point-vortex far-field increment at the j=NJ interior cell center of column i (session-verified
# form: q_θ = Γ/(2πr)·β/(1−M∞²·sin²θ), θ from the FREESTREAM axis ⇒ rotate offsets into the freestream
# frame, apply the tangential field there, rotate the velocity back). Γ=0 ⇒ (0,0) exact legacy.
@inline function _vortex_pg(s::CSol, i::Int)
    s.gam == 0.0f0 && return (0.0f0, 0.0f0)
    dx = s.xc[i, s.NJ] - s.xv; dy = s.yc[i, s.NJ] - s.yv
    dxs = dx*s.ca + dy*s.sa; dys = -dx*s.sa + dy*s.ca
    r2 = dxs*dxs + dys*dys + 1.0f-6
    Fc = s.beta/(1.0f0 - s.m2*(dys*dys/r2))
    c = s.gam*Fc/(6.2831853f0*r2)
    dus = c*dys; dvs = -c*dxs
    (dus*s.ca - dvs*s.sa, dus*s.sa + dvs*s.ca)
end

# ---- BC ghost fills (v1 gate-(a) set; A2 replaces the wall stub) -----------------------------
"""Apply only the segmented outer slip-wall primitive override (a no-op in legacy mode)."""
function outer_bc_override!(s::CSol)
    s.outer_mode == Int32(0) && return nothing
    g, NI, NJ = s.g, s.NI, s.NJ
    @inbounds for k in 1:g, i in 1:NI
        s.outer_face_kind[i] >= 0x02 || continue
        js = NJ + g + 1 - k; jd = NJ + g + k
        Sx = s.Sjx[i, NJ+1]; Sy = s.Sjy[i, NJ+1]
        Sm = sqrt(Sx*Sx + Sy*Sy); nx = Sx/Sm; ny = Sy/Sm
        u0 = s.ux[i+g, js]; v0 = s.uy[i+g, js]; un = u0*nx + v0*ny
        s.rho[i+g, jd] = s.rho[i+g, js]
        s.ux[i+g, jd] = u0 - 2.0f0*un*nx
        s.uy[i+g, jd] = v0 - 2.0f0*un*ny
        s.T[i+g, jd] = s.T[i+g, js]
    end
    nothing
end

function applyBC!(s::CSol; wall::Symbol=:transparent)
    g, NI, NJ, NIW, NIF = s.g, s.NI, s.NJ, s.NIW, s.NIF
    rho, ux, uy, T = s.rho, s.ux, s.uy, s.T
    # 1. wake-cut exchange: ghost(i, 1-k) := interior(partner=NI+1-i, k), verbatim, no sign flips
    for k in 1:g, i in 1:NI
        if i <= NIW || i > NIW+NIF
            ip = NI+1-i
            rho[i+g, g+1-k] = rho[ip+g, k+g]; ux[i+g, g+1-k] = ux[ip+g, k+g]
            uy[i+g, g+1-k]  = uy[ip+g, k+g];  T[i+g, g+1-k]  = T[ip+g, k+g]
        end
    end
    # 2. wall (airfoil columns): :transparent (gate-a stub) | :slip (u reflected about the wall
    #    tangent — inviscid, gate c) | :noslip (u negated — viscous; adiabatic dT/dn=0, dp/dn=0 both)
    for k in 1:g, i in NIW+1:NIW+NIF
        if wall == :transparent
            rho[i+g, g+1-k] = rho[i+g, g+k]; ux[i+g, g+1-k] = ux[i+g, g+k]
            uy[i+g, g+1-k]  = uy[i+g, g+k];  T[i+g, g+1-k]  = T[i+g, g+k]
        elseif wall == :slip
            Sx = s.Sjx[i,1]; Sy = s.Sjy[i,1]; Sm = sqrt(Sx*Sx+Sy*Sy); nxw = Sx/Sm; nyw = Sy/Sm
            u0 = ux[i+g, g+k]; v0 = uy[i+g, g+k]; un = u0*nxw + v0*nyw
            rho[i+g, g+1-k] = rho[i+g, g+k]
            ux[i+g, g+1-k]  = u0 - 2.0f0*un*nxw
            uy[i+g, g+1-k]  = v0 - 2.0f0*un*nyw
            T[i+g, g+1-k]   = T[i+g, g+k]
        else # :noslip
            rho[i+g, g+1-k] = rho[i+g, g+k]; ux[i+g, g+1-k] = -ux[i+g, g+k]
            uy[i+g, g+1-k]  = -uy[i+g, g+k]; T[i+g, g+1-k]  = T[i+g, g+k]
        end
    end
    # 3. outer far-field: CHARACTERISTIC (Riemann-invariant) BC on the local outward normal, with the
    #    PG point-vortex-corrected freestream as the far state (port of pond _nscbc2T_y!, generalized
    #    from the hard-coded y-normal to n̂ = S⃗_j(i,NJ+1)/|S⃗| and to qn_inf ≠ 0). A hard Dirichlet here
    #    traps the startup acoustics (measured: Cl oscillating ±0.2 with negative Cd — gate-c FAIL).
    gam_eos = (5.0f0 + s.drot)/(3.0f0 + s.drot); gm1 = gam_eos - 1.0f0; Rgas = CS2
    for i in 1:NI
        dux, duy = _vortex_pg(s, i)
        uxt = s.uxi + dux; uyt = s.uyi + duy                 # vortex-corrected far state
        Sx = s.Sjx[i,NJ+1]; Sy = s.Sjy[i,NJ+1]; Sm = sqrt(Sx*Sx+Sy*Sy)
        nxb = Sx/Sm; nyb = Sy/Sm; txb = -nyb; tyb = nxb      # outward normal + tangent
        rd = rho[i+g, NJ+g]; uxd = ux[i+g, NJ+g]; uyd = uy[i+g, NJ+g]; Td = T[i+g, NJ+g]
        cd_ = sqrt(gam_eos*Rgas*max(Td, 1.0f-4)); cinf = sqrt(gam_eos*Rgas*s.T_inf)
        qnd = uxd*nxb + uyd*nyb
        qni = uxt*nxb + uyt*nyb
        rb = rd; uxb = uxd; uyb = uyd; Tb = Td               # supersonic-outflow default: extrapolate
        if qnd <= -cd_                                        # supersonic inflow ⇒ full far state
            rb = s.rho_inf; uxb = uxt; uyb = uyt; Tb = s.T_inf
        elseif qnd < cd_                                      # subsonic ⇒ Riemann split
            Rout = qnd + 2.0f0*cd_/gm1
            Rin  = qni - 2.0f0*cinf/gm1
            qnb  = 0.5f0*(Rout+Rin)
            cb   = 0.25f0*gm1*(Rout-Rin); cb = cb < 1.0f-4 ? 1.0f-4 : cb
            local sB::Float32; local utb::Float32
            if qnb >= 0.0f0                                   # outflow: entropy + tangential from interior
                sB = (rd*Rgas*Td)/(rd^gam_eos); utb = uxd*txb + uyd*tyb
            else                                              # inflow: from the (vortex-corrected) far state
                sB = (s.rho_inf*Rgas*s.T_inf)/(s.rho_inf^gam_eos); utb = uxt*txb + uyt*tyb
            end
            rb  = (cb*cb/(gam_eos*sB))^(1.0f0/gm1)
            Tb  = (sB*rb^gam_eos)/(rb*Rgas)
            uxb = qnb*nxb + utb*txb; uyb = qnb*nyb + utb*tyb
        end
        for k in 1:g
            rho[i+g, NJ+g+k] = rb; ux[i+g, NJ+g+k] = uxb; uy[i+g, NJ+g+k] = uyb; T[i+g, NJ+g+k] = Tb
        end
    end
    outer_bc_override!(s)
    for k in 1:g, kk in 1:g   # far-field corners of the outlet ghost columns: plain freestream
        rho[kk, NJ+g+k] = s.rho_inf; ux[kk, NJ+g+k] = s.uxi; uy[kk, NJ+g+k] = s.uyi; T[kk, NJ+g+k] = s.T_inf
        rho[NI+g+kk, NJ+g+k] = s.rho_inf; ux[NI+g+kk, NJ+g+k] = s.uxi; uy[NI+g+kk, NJ+g+k] = s.uyi; T[NI+g+kk, NJ+g+k] = s.T_inf
    end
    # 4. outlet columns (i<=0, i>NI): zero-gradient — LAST so the corner ghosts see exchanged data
    for k in 1:g, jj in 1:NJ+2g
        rho[g+1-k, jj] = rho[g+1, jj]; ux[g+1-k, jj] = ux[g+1, jj]
        uy[g+1-k, jj]  = uy[g+1, jj];  T[g+1-k, jj]  = T[g+1, jj]
        rho[NI+g+k, jj] = rho[NI+g, jj]; ux[NI+g+k, jj] = ux[NI+g, jj]
        uy[NI+g+k, jj]  = uy[NI+g, jj]; T[NI+g+k, jj]  = T[NI+g, jj]
    end
    nothing
end

# ---- one SSP-RK3 LTS pseudo-step -------------------------------------------------------------
"""Apply only the exact segmented outer slip-wall flux override (a no-op in legacy mode)."""
function outer_flux_override!(s::CSol)
    s.outer_mode == Int32(0) && return nothing
    g, NI, NJ = s.g, s.NI, s.NJ
    @inbounds for i in 1:NI
        s.outer_face_kind[i] >= 0x02 || continue
        ia = i + g; ja = NJ + g
        pwall = s.rho[ia, ja]*CS2*s.T[ia, ja]
        s.Fj_r[i, NJ+1] = 0.0f0
        s.Fj_mx[i, NJ+1] = pwall*s.Sjx[i, NJ+1]
        s.Fj_my[i, NJ+1] = pwall*s.Sjy[i, NJ+1]
        s.Fj_e[i, NJ+1] = 0.0f0
    end
    nothing
end

function _fluxes!(s::CSol)
    g, NI, NJ, NIW, NIF = s.g, s.NI, s.NJ, s.NIW, s.NIF
    rho, ux, uy, T = s.rho, s.ux, s.uy, s.T
    # i-faces: face (i,j) between interior cells (i-1,j) and (i,j)
    Threads.@threads for j in 1:NJ
        for i in 1:NI+1
            nxr = s.Six[i,j]; nyr = s.Siy[i,j]; S = sqrt(nxr*nxr + nyr*nyr)
            nx = nxr/S; ny = nyr/S
            ia = i+g; ja = j+g       # face between array cols ia-1 (interior i-1) and ia (interior i)
            Fr,Fmx,Fmy,Fe = _fvflux_normal6(
                rho[ia-3,ja],ux[ia-3,ja],uy[ia-3,ja],T[ia-3,ja],  # LLL
                rho[ia-2,ja],ux[ia-2,ja],uy[ia-2,ja],T[ia-2,ja],  # LL
                rho[ia-1,ja],ux[ia-1,ja],uy[ia-1,ja],T[ia-1,ja],  # L
                rho[ia,  ja],ux[ia,  ja],uy[ia,  ja],T[ia,  ja],  # R
                rho[ia+1,ja],ux[ia+1,ja],uy[ia+1,ja],T[ia+1,ja],  # RR
                rho[ia+2,ja],ux[ia+2,ja],uy[ia+2,ja],T[ia+2,ja],  # RRR
                nx, ny, S, s.cxa, s.cya, s.cza, s.wa, s.drot,
                s.recon >= Int32(1) ? Int32(1) : Int32(0))   # recon=2: WENO in i (near-uniform spacing)
            s.Fi_r[i,j]=Fr; s.Fi_mx[i,j]=Fmx; s.Fi_my[i,j]=Fmy; s.Fi_e[i,j]=Fe
        end
    end
    # j-faces: face (i,j) between interior cells (i,j-1) and (i,j)
    Threads.@threads for j in 1:NJ+1
        for i in 1:NI
            nxr = s.Sjx[i,j]; nyr = s.Sjy[i,j]; S = sqrt(nxr*nxr + nyr*nyr)
            nx = nxr/S; ny = nyr/S
            ia = i+g; ja = j+g       # face between array rows ja-1 (interior j-1) and ja (interior j)
            Fr,Fmx,Fmy,Fe = _fvflux_normal6(
                rho[ia,ja-3],ux[ia,ja-3],uy[ia,ja-3],T[ia,ja-3],  # LLL
                rho[ia,ja-2],ux[ia,ja-2],uy[ia,ja-2],T[ia,ja-2],  # LL
                rho[ia,ja-1],ux[ia,ja-1],uy[ia,ja-1],T[ia,ja-1],  # L
                rho[ia,ja  ],ux[ia,ja  ],uy[ia,ja  ],T[ia,ja  ],  # R
                rho[ia,ja+1],ux[ia,ja+1],uy[ia,ja+1],T[ia,ja+1],  # RR
                rho[ia,ja+2],ux[ia,ja+2],uy[ia,ja+2],T[ia,ja+2],  # RRR
                nx, ny, S, s.cxa, s.cya, s.cza, s.wa, s.drot,
                s.recon == Int32(1) ? Int32(1) : Int32(0))   # recon=2: minmod in j (1.25-growth stretch breaks uniform-grid WENO)
            s.Fj_r[i,j]=Fr; s.Fj_mx[i,j]=Fmx; s.Fj_my[i,j]=Fmy; s.Fj_e[i,j]=Fe
        end
    end
    # VISCOUS face flux (A2b, thin-layer): F_mom -= μ_f·S·Δu⃗/dn, F_E -= μ_f·S·(u_f·Δu⃗)/dn + κ·S·ΔT/dn,
    # with dn = center-to-center distance and μ_f = ρ_f·ν (the pond molecular-ν face form, made
    # metric-aware). The right BL approximation on a body-fitted mesh (plan §2); full stress = later.
    if s.nu_lam > 0.0f0
        nu = s.nu_lam; Pr = 0.71f0; cpg = 0.5f0*(5.0f0+s.drot)*CS2
        fv = s.fullvisc == Int32(1)
        if fv   # A2c: Green-Gauss cell gradients (ghosts hold BC states after applyBC!)
            Threads.@threads for j in 1:NJ
                for i in 1:NI
                    ia=i+g; ja=j+g; iv = 1.0f0/s.vol[i,j]
                    SxW=s.Six[i,j]; SyW=s.Siy[i,j]; SxE=s.Six[i+1,j]; SyE=s.Siy[i+1,j]
                    SxS=s.Sjx[i,j]; SyS=s.Sjy[i,j]; SxN=s.Sjx[i,j+1]; SyN=s.Sjy[i,j+1]
                    s.gux[i,j], s.guy[i,j] = _gg_grad(ux[ia-1,ja],ux[ia+1,ja],ux[ia,ja-1],ux[ia,ja+1],ux[ia,ja],
                                                      SxW,SyW,SxE,SyE,SxS,SyS,SxN,SyN, iv)
                    s.gvx[i,j], s.gvy[i,j] = _gg_grad(uy[ia-1,ja],uy[ia+1,ja],uy[ia,ja-1],uy[ia,ja+1],uy[ia,ja],
                                                      SxW,SyW,SxE,SyE,SxS,SyS,SxN,SyN, iv)
                    s.gTx[i,j], s.gTy[i,j] = _gg_grad(T[ia-1,ja],T[ia+1,ja],T[ia,ja-1],T[ia,ja+1],T[ia,ja],
                                                      SxW,SyW,SxE,SyE,SxS,SyS,SxN,SyN, iv)
                end
            end
        end
        Threads.@threads for j in 1:NJ
            for i in 2:NI   # interior i-faces (boundary i-faces: outlet zero-gradient ⇒ Δ=0 anyway)
                ia=i+g; ja=j+g
                dxc = s.xc[i,j]-s.xc[i-1,j]; dyc = s.yc[i,j]-s.yc[i-1,j]; dn = sqrt(dxc*dxc+dyc*dyc)+1.0f-30
                Sf = sqrt(s.Six[i,j]^2 + s.Siy[i,j]^2)
                nut_f = s.sst ? 0.5f0*(s.nut[ia-1,ja]+s.nut[ia,ja]) : 0.0f0
                rf = 0.5f0*(rho[ia-1,ja]+rho[ia,ja])
                fmu = _mufac(0.5f0*(T[ia-1,ja]+T[ia,ja]), s.suth)   # Sutherland mu(T)
                if fv
                    keff = rf*cpg*(nu*fmu/Pr + nut_f/0.9f0)
                    Fvx, Fvy, Fve = _viscous_face_full(
                        s.gux[i-1,j],s.guy[i-1,j],s.gvx[i-1,j],s.gvy[i-1,j],s.gTx[i-1,j],s.gTy[i-1,j],
                        s.gux[i,j],  s.guy[i,j],  s.gvx[i,j],  s.gvy[i,j],  s.gTx[i,j],  s.gTy[i,j],
                        ux[ia-1,ja],uy[ia-1,ja],T[ia-1,ja], ux[ia,ja],uy[ia,ja],T[ia,ja],
                        dxc, dyc, s.Six[i,j]/Sf, s.Siy[i,j]/Sf, Sf, rf*(nu*fmu+nut_f), keff)
                    s.Fi_mx[i,j] -= Fvx; s.Fi_my[i,j] -= Fvy; s.Fi_e[i,j] -= Fve
                else
                    mu = rf*(nu*fmu+nut_f)*Sf/dn
                    du = ux[ia,ja]-ux[ia-1,ja]; dv = uy[ia,ja]-uy[ia-1,ja]; dT = T[ia,ja]-T[ia-1,ja]
                    ufx = 0.5f0*(ux[ia-1,ja]+ux[ia,ja]); ufy = 0.5f0*(uy[ia-1,ja]+uy[ia,ja])
                    s.Fi_mx[i,j] -= mu*du; s.Fi_my[i,j] -= mu*dv
                    s.Fi_e[i,j]  -= mu*(ufx*du+ufy*dv) + rf*cpg*(nu*fmu/Pr + nut_f/0.9f0)*Sf/dn*dT
                end
                if s.rhok == Int32(1) && s.sst       # -2/3 rho k in the turbulent stress (acts as pressure)
                    kf = 0.5f0*(s.tk[ia-1,ja]+s.tk[ia,ja])
                    ufx2 = 0.5f0*(ux[ia-1,ja]+ux[ia,ja]); ufy2 = 0.5f0*(uy[ia-1,ja]+uy[ia,ja])
                    s.Fi_mx[i,j] += 0.6666667f0*rf*kf*s.Six[i,j]
                    s.Fi_my[i,j] += 0.6666667f0*rf*kf*s.Siy[i,j]
                    s.Fi_e[i,j]  += 0.6666667f0*rf*kf*(ufx2*s.Six[i,j] + ufy2*s.Siy[i,j])
                end
                if s.earsm == Int32(1) && s.sst      # WJ-EARSM extra anisotropy stress (F -= tau·n·S)
                    e11 = 0.5f0*(s.exx[i-1,j]+s.exx[i,j]); e12 = 0.5f0*(s.exy[i-1,j]+s.exy[i,j])
                    e22 = 0.5f0*(s.eyy[i-1,j]+s.eyy[i,j])
                    fex = e11*s.Six[i,j] + e12*s.Siy[i,j]; fey = e12*s.Six[i,j] + e22*s.Siy[i,j]
                    ufx3 = 0.5f0*(ux[ia-1,ja]+ux[ia,ja]); ufy3 = 0.5f0*(uy[ia-1,ja]+uy[ia,ja])
                    s.Fi_mx[i,j] -= fex; s.Fi_my[i,j] -= fey
                    s.Fi_e[i,j]  -= ufx3*fex + ufy3*fey
                end
            end
        end
        Threads.@threads for j in 1:NJ   # interior j-faces + the WAKE-CUT faces (j=1, wake columns):
            for i in 1:NI                # the cut previously carried ZERO viscous flux — the wake sheet
                                         # could kink at the cut = a TE acoustic dipole (upstream ripples)
                j == 1 && !(i <= NIW || i > NIW+NIF) && continue    # foil j=1 = wall-flux override
                ia=i+g; ja=j+g
                Sf = sqrt(s.Sjx[i,j]^2 + s.Sjy[i,j]^2)
                if j == 1
                    # Both copies of a wake-cut face must have the same viscous
                    # conductance. The two first-row cells can have different
                    # heights on a cambered A-plus finite-TE mesh.
                    ip = NI+1-i
                    dxc = s.xc[i,1]-s.xc[ip,1]
                    dyc = s.yc[i,1]-s.yc[ip,1]
                    dn = sqrt(dxc*dxc+dyc*dyc)+1.0f-30
                else
                    dxc = s.xc[i,j]-s.xc[i,j-1]; dyc = s.yc[i,j]-s.yc[i,j-1]; dn = sqrt(dxc*dxc+dyc*dyc)+1.0f-30
                end
                nut_f = s.sst ? 0.5f0*(s.nut[ia,ja-1]+s.nut[ia,ja]) : 0.0f0
                rf = 0.5f0*(rho[ia,ja-1]+rho[ia,ja])
                fmu = _mufac(0.5f0*(T[ia,ja-1]+T[ia,ja]), s.suth)   # Sutherland mu(T)
                if fv && j > 1   # full stress needs both cell gradients; the cut face (j=1) is thin-layer
                    keff = rf*cpg*(nu*fmu/Pr + nut_f/0.9f0)
                    Fvx, Fvy, Fve = _viscous_face_full(
                        s.gux[i,j-1],s.guy[i,j-1],s.gvx[i,j-1],s.gvy[i,j-1],s.gTx[i,j-1],s.gTy[i,j-1],
                        s.gux[i,j],  s.guy[i,j],  s.gvx[i,j],  s.gvy[i,j],  s.gTx[i,j],  s.gTy[i,j],
                        ux[ia,ja-1],uy[ia,ja-1],T[ia,ja-1], ux[ia,ja],uy[ia,ja],T[ia,ja],
                        dxc, dyc, s.Sjx[i,j]/Sf, s.Sjy[i,j]/Sf, Sf, rf*(nu*fmu+nut_f), keff)
                    s.Fj_mx[i,j] -= Fvx; s.Fj_my[i,j] -= Fvy; s.Fj_e[i,j] -= Fve
                else
                    mu = rf*(nu*fmu+nut_f)*Sf/dn
                    du = ux[ia,ja]-ux[ia,ja-1]; dv = uy[ia,ja]-uy[ia,ja-1]; dT = T[ia,ja]-T[ia,ja-1]
                    ufx = 0.5f0*(ux[ia,ja-1]+ux[ia,ja]); ufy = 0.5f0*(uy[ia,ja-1]+uy[ia,ja])
                    s.Fj_mx[i,j] -= mu*du; s.Fj_my[i,j] -= mu*dv
                    s.Fj_e[i,j]  -= mu*(ufx*du+ufy*dv) + rf*cpg*(nu*fmu/Pr + nut_f/0.9f0)*Sf/dn*dT
                end
                if s.rhok == Int32(1) && s.sst       # -2/3 rho k in the turbulent stress
                    kf = 0.5f0*(s.tk[ia,ja-1]+s.tk[ia,ja])
                    ufx2 = 0.5f0*(ux[ia,ja-1]+ux[ia,ja]); ufy2 = 0.5f0*(uy[ia,ja-1]+uy[ia,ja])
                    s.Fj_mx[i,j] += 0.6666667f0*rf*kf*s.Sjx[i,j]
                    s.Fj_my[i,j] += 0.6666667f0*rf*kf*s.Sjy[i,j]
                    s.Fj_e[i,j]  += 0.6666667f0*rf*kf*(ufx2*s.Sjx[i,j] + ufy2*s.Sjy[i,j])
                end
                if s.earsm == Int32(1) && s.sst && j > 1   # EARSM extra stress (interior j-faces)
                    e11 = 0.5f0*(s.exx[i,j-1]+s.exx[i,j]); e12 = 0.5f0*(s.exy[i,j-1]+s.exy[i,j])
                    e22 = 0.5f0*(s.eyy[i,j-1]+s.eyy[i,j])
                    fex = e11*s.Sjx[i,j] + e12*s.Sjy[i,j]; fey = e12*s.Sjx[i,j] + e22*s.Sjy[i,j]
                    ufx3 = 0.5f0*(ux[ia,ja-1]+ux[ia,ja]); ufy3 = 0.5f0*(uy[ia,ja-1]+uy[ia,ja])
                    s.Fj_mx[i,j] -= fex; s.Fj_my[i,j] -= fey
                    s.Fj_e[i,j]  -= ufx3*fex + ufy3*fey
                end
            end
        end
    end
    # WALL FLUX OVERRIDE (foil columns, j=1 face): the analytic impermeable-wall flux
    # F = (0, p_w·S⃗, 0) with p_w = rho·cs2·T of the first interior cell — the port of pond's
    # staircase kinetic wall (thermal.jl:404-410; defuses the mirrored-ghost stagnation hot-spot).
    if s.wall_flux
        @inbounds for i in s.NIW+1:s.NIW+s.NIF
            ia = i+g; pw = rho[ia, g+1]*CS2*T[ia, g+1]
            s.Fj_r[i,1]  = 0.0f0
            s.Fj_mx[i,1] = pw*s.Sjx[i,1]
            s.Fj_my[i,1] = pw*s.Sjy[i,1]
            s.Fj_e[i,1]  = 0.0f0
            if s.nu_lam > 0.0f0                      # no-slip wall SHEAR (adiabatic: no heat term;
                Sf = sqrt(s.Sjx[i,1]^2 + s.Sjy[i,1]^2)   #  shear work τ·u_wall = 0)
                d1 = 0.5f0*s.vol[i,1]/Sf + 1.0f-30       # first-cell center-to-wall distance
                muw = rho[ia, g+1]*s.nu_lam*_mufac(T[ia, g+1], s.suth)*Sf/d1
                if s.fullvisc==Int32(1)
                    tx,ty=_wall_stokes_traction(ux[ia,g+1],uy[ia,g+1],s.Sjx[i,1]/Sf,s.Sjy[i,1]/Sf,muw)
                    s.Fj_mx[i,1] -= tx; s.Fj_my[i,1] -= ty
                else
                    s.Fj_mx[i,1] -= muw*ux[ia, g+1]
                    s.Fj_my[i,1] -= muw*uy[ia, g+1]
                end
            end
        end
    end
    outer_flux_override!(s)
    # residual: dc = Fi(i+1)-Fi(i) + Fj(j+1)-Fj(j)  (outward-positive with the +i/+j normals)
    Threads.@threads for j in 1:NJ
        for i in 1:NI
            s.dc_r[i,j]  = s.Fi_r[i+1,j]-s.Fi_r[i,j]  + s.Fj_r[i,j+1]-s.Fj_r[i,j]
            s.dc_mx[i,j] = s.Fi_mx[i+1,j]-s.Fi_mx[i,j] + s.Fj_mx[i,j+1]-s.Fj_mx[i,j]
            s.dc_my[i,j] = s.Fi_my[i+1,j]-s.Fi_my[i,j] + s.Fj_my[i,j+1]-s.Fj_my[i,j]
            s.dc_e[i,j]  = s.Fi_e[i+1,j]-s.Fi_e[i,j]  + s.Fj_e[i,j+1]-s.Fj_e[i,j]
        end
    end
    nothing
end

function _dtfield!(s::CSol, cfl::Float32)
    g = s.g
    Threads.@threads for j in 1:s.NJ
        for i in 1:s.NI
            ia=i+g; ja=j+g
            u = s.ux[ia,ja]; v = s.uy[ia,ja]; c = 3.0f0*sqrt(max(s.T[ia,ja],1.0f-3))  # kinetic signal bound
            SiE = sqrt(s.Six[i+1,j]^2 + s.Siy[i+1,j]^2); SiW = sqrt(s.Six[i,j]^2 + s.Siy[i,j]^2)
            SjN = sqrt(s.Sjx[i,j+1]^2 + s.Sjy[i,j+1]^2); SjS = sqrt(s.Sjx[i,j]^2 + s.Sjy[i,j]^2)
            lam_i = (abs(u*s.Six[i,j]+v*s.Siy[i,j]) + abs(u*s.Six[i+1,j]+v*s.Siy[i+1,j]))*0.5f0 + c*0.5f0*(SiE+SiW)
            lam_j = (abs(u*s.Sjx[i,j]+v*s.Sjy[i,j]) + abs(u*s.Sjx[i,j+1]+v*s.Sjy[i,j+1]))*0.5f0 + c*0.5f0*(SjN+SjS)
            nueff = s.nu_lam + (s.sst ? s.nut[i+g, j+g] : 0.0f0)
            local dt::Float32
            if s.irsj == Int32(1)
                # IRS-J: dt from the STREAMWISE restriction only (j-stiffness handled implicitly)
                dt = cfl*s.vol[i,j]/max(lam_i, 1.0f-30)
                if nueff > 0.0f0
                    dn_i = s.vol[i,j]/max(SiE, SiW)            # i-direction viscous cap only
                    dt = min(dt, 0.2f0*dn_i*dn_i/nueff)
                end
            else
                dt = cfl*s.vol[i,j]/max(lam_i+lam_j, 1.0f-30)
                if nueff > 0.0f0                               # explicit-diffusion cap (y+~1 insurance)
                    dn_min = s.vol[i,j]/max(max(SiE,SiW), max(SjN,SjS))
                    dt = min(dt, 0.2f0*dn_min*dn_min/nueff)
                end
            end
            s.dtf[i,j] = dt
        end
    end
    nothing
end

function _snapshot!(s::CSol)
    g = s.g
    Threads.@threads for j in 1:s.NJ
        for i in 1:s.NI
            ia=i+g; ja=j+g
            r=s.rho[ia,ja]; u=s.ux[ia,ja]; v=s.uy[ia,ja]
            s.rk_r[ia,ja]=r; s.rk_mx[ia,ja]=r*u; s.rk_my[ia,ja]=r*v
            s.rk_E[ia,ja]=r*(s.eint*s.T[ia,ja]+0.5f0*(u*u+v*v))
        end
    end
    nothing
end

function _update!(s::CSol, stage::Int, dtp::Float32=0.0f0; preserve_ref::Bool=false)
    g = s.g; w = stage==1 ? 1.0f0 : (stage==2 ? 0.25f0 : 2.0f0/3.0f0)
    b = stage==1 ? 0.0f0 : (stage==2 ? 0.75f0 : 1.0f0/3.0f0)
    Threads.@threads for j in 1:s.NJ
        for i in 1:s.NI
            ia=i+g; ja=j+g
            r0=s.rho[ia,ja]; u0=s.ux[ia,ja]; v0=s.uy[ia,ja]
            E0=r0*(s.eint*s.T[ia,ja]+0.5f0*(u0*u0+v0*v0))
            if preserve_ref && dtp == 0.0f0 &&
               s.dc_r[i,j] == 0.0f0 && s.dc_mx[i,j] == 0.0f0 &&
               s.dc_my[i,j] == 0.0f0 && s.dc_e[i,j] == 0.0f0 &&
               r0 == s.outer_ref_rho && u0 == s.outer_ref_ux &&
               v0 == s.outer_ref_uy && s.T[ia,ja] == s.outer_ref_T &&
               s.rk_r[ia,ja] == r0 && s.rk_mx[ia,ja] == r0*u0 &&
               s.rk_my[ia,ja] == r0*v0 && s.rk_E[ia,ja] == E0
                continue
            end
            a = s.dtf[i,j]/s.vol[i,j]
            r  = r0    - a*s.dc_r[i,j]
            mx = r0*u0 - a*s.dc_mx[i,j]
            my = r0*v0 - a*s.dc_my[i,j]
            e  = E0    - a*s.dc_e[i,j]
            if dtp > 0.0f0
                # dual-time BDF2: point-implicit source (3U - 4U^n + U^(n-1))/(2*dtp); the pseudo-
                # steady fixed point of this stage map is exactly BDF2 in physical time (Melson-style
                # diagonal treatment keeps dtl >> dtp stable in the LTS far field)
                dtl = s.dtf[i,j]; fac = 1.0f0/(1.0f0 + 1.5f0*dtl/dtp); h = dtl/(2.0f0*dtp)
                r  = (r  + h*(4.0f0*s.un_r[i,j]  - s.um_r[i,j] ))*fac
                mx = (mx + h*(4.0f0*s.un_mx[i,j] - s.um_mx[i,j]))*fac
                my = (my + h*(4.0f0*s.un_my[i,j] - s.um_my[i,j]))*fac
                e  = (e  + h*(4.0f0*s.un_E[i,j]  - s.um_E[i,j] ))*fac
            end
            # SSP-RK3 combination against the stage-0 snapshot
            r  = b*s.rk_r[ia,ja]  + w*r
            mx = b*s.rk_mx[ia,ja] + w*mx
            my = b*s.rk_my[ia,ja] + w*my
            e  = b*s.rk_E[ia,ja]  + w*e
            r = r < 1.0f-3 ? 1.0f-3 : r                     # pond primitive recovery verbatim
            u2=mx/r; v2=my/r; sp=sqrt(u2*u2+v2*v2)
            if sp>3.5f0; f=3.5f0/sp; u2*=f; v2*=f; end
            T2=(e/r-0.5f0*(u2*u2+v2*v2))/s.eint; T2 = T2<1.0f-3 ? 1.0f-3 : T2
            if s.sponge_sigma > 0.0f0                        # far-field acoustic sponge (outer j-rows)
                j0 = s.sponge_j0f*Float32(s.NJ)
                if Float32(j) > j0
                    wsp = min(s.sponge_sigma*s.dtf[i,j]*((Float32(j)-j0)/(Float32(s.NJ)-j0))^2, 0.5f0)
                    du, dv = _pg_vortex_uv(s.xc[i,j], s.yc[i,j], s.gam, s.xv, s.yv, s.m2, s.beta, s.ca, s.sa)
                    r  -= wsp*(r  - s.rho_inf)
                    u2 -= wsp*(u2 - (s.uxi + du)); v2 -= wsp*(v2 - (s.uyi + dv))
                    T2 -= wsp*(T2 - s.T_inf)
                end
            end
            s.rho[ia,ja]=r; s.ux[ia,ja]=u2; s.uy[ia,ja]=v2; s.T[ia,ja]=T2
        end
    end
    nothing
end

# IRS-J: variable-coefficient implicit residual smoothing along j (one tridiagonal per i-column,
# Martinelli/Jameson): -aS·x_{j-1} + (1+aS+aN)·x_j - aN·x_{j+1} = dc_j, with the face coefficient
# eps = 0.25·max(0, (nuJ/0.65)^2 - 1), nuJ = dtl·lam_face/vol = the LOCAL j-CFL at the enlarged
# (i-restricted) dt. lam includes the acoustic AND viscous j-spectral radii. Solves in place on dc.
function _irsj_component!(s::CSol, dc::AbstractMatrix{Float32})
    g, NI, NJ = s.g, s.NI, s.NJ
    bp = s.irs_c; an = s.irs_d                      # Thomas b-prime + per-row aN (face j+1 eps)
    Threads.@threads for i in 1:NI
        ia = i + g
        # pass 1: aN per row (eps of face j+1, evaluated with cell-j state); aS_j = aN_{j-1}
        for j in 1:NJ
            if j == NJ
                an[i,j] = 0.0f0
            else
                Sx = s.Sjx[i,j+1]; Sy = s.Sjy[i,j+1]; S2f = Sx*Sx + Sy*Sy
                u = s.ux[ia,j+g]; v = s.uy[ia,j+g]; c = 3.0f0*sqrt(max(s.T[ia,j+g],1.0f-3))
                nueff = s.nu_lam + (s.sst ? s.nut[ia,j+g] : 0.0f0)
                lam = abs(u*Sx + v*Sy) + c*sqrt(S2f) + 2.0f0*nueff*S2f/s.vol[i,j]
                nuJ = s.dtf[i,j]*lam/(s.vol[i,j]*0.25f0)
                an[i,j] = 0.25f0*max(0.0f0, nuJ*nuJ - 1.0f0)
            end
        end
        # BOUNDARY faces (wall/wake-cut at j=1, far-field at NJ+1): residual beyond the boundary is
        # ZERO (Dirichlet-0 ghost) => their eps stays in the DIAGONAL as damping, no off-diagonal.
        # Dropping it leaves those faces explicit at the enlarged dt (instant wall-acoustic blowup).
        eps1 = begin
            Sx = s.Sjx[i,1]; Sy = s.Sjy[i,1]; S2f = Sx*Sx + Sy*Sy
            u = s.ux[ia,1+g]; v = s.uy[ia,1+g]; c = 3.0f0*sqrt(max(s.T[ia,1+g],1.0f-3))
            nueff = s.nu_lam + (s.sst ? s.nut[ia,1+g] : 0.0f0)
            lam = abs(u*Sx + v*Sy) + c*sqrt(S2f) + 2.0f0*nueff*S2f/s.vol[i,1]
            nuJ = s.dtf[i,1]*lam/(s.vol[i,1]*0.25f0)
            0.25f0*max(0.0f0, nuJ*nuJ - 1.0f0)
        end
        epsN1 = begin
            Sx = s.Sjx[i,NJ+1]; Sy = s.Sjy[i,NJ+1]; S2f = Sx*Sx + Sy*Sy
            u = s.ux[ia,NJ+g]; v = s.uy[ia,NJ+g]; c = 3.0f0*sqrt(max(s.T[ia,NJ+g],1.0f-3))
            nueff = s.nu_lam + (s.sst ? s.nut[ia,NJ+g] : 0.0f0)
            lam = abs(u*Sx + v*Sy) + c*sqrt(S2f) + 2.0f0*nueff*S2f/s.vol[i,NJ]
            nuJ = s.dtf[i,NJ]*lam/(s.vol[i,NJ]*0.25f0)
            0.25f0*max(0.0f0, nuJ*nuJ - 1.0f0)
        end
        # forward sweep (lower a_j = -aS_j, diag b_j = 1+aS+aN, upper c_j = -aN_j; d-prime in dc)
        bp[i,1] = 1.0f0 + eps1 + an[i,1]
        for j in 2:NJ
            aS = an[i,j-1]; aNd = j == NJ ? epsN1 : an[i,j]
            w = (-aS)/bp[i,j-1]
            bp[i,j] = (1.0f0 + aS + aNd) - w*(-aS)  # c_{j-1} = -aN_{j-1} = -aS_j
            dc[i,j] = dc[i,j] - w*dc[i,j-1]
        end
        # back substitution
        dc[i,NJ] = dc[i,NJ]/bp[i,NJ]
        for j in NJ-1:-1:1
            dc[i,j] = (dc[i,j] + an[i,j]*dc[i,j+1])/bp[i,j]
        end
    end
    nothing
end

function _validate_channel_compat!(s::CSol)
    s.outer_mode == Int32(0) && return nothing
    s.sponge_sigma == 0.0f0 || throw(ArgumentError("the far-field sponge is not defined for segmented channel boundaries"))
    s.gam == 0.0f0 || throw(ArgumentError("the point-vortex far-field correction is not defined for segmented channel boundaries"))
    nothing
end

function _validate_channel_wall_alignment!(s::CSol)
    s.outer_mode == Int32(0) && return nothing
    Umag = hypot(Float64(s.uxi), Float64(s.uyi))
    isfinite(Umag) || throw(ArgumentError("channel freestream speed is not finite"))
    Umag == 0.0 && return nothing
    nwalls = 0
    @inbounds for i in 1:s.NI
        kind = s.outer_face_kind[i]
        if kind == 0x02 || kind == 0x03
            nwalls += 1
            Sx = Float64(s.Sjx[i,s.NJ+1]); Sy = Float64(s.Sjy[i,s.NJ+1])
            Smag = hypot(Sx, Sy)
            isfinite(Smag) && Smag > 0.0 || throw(ArgumentError(
                "channel outer wall face $i has an invalid normal"))
            relative_dot = abs(Float64(s.uxi)*Sx + Float64(s.uyi)*Sy)/(Umag*Smag)
            isfinite(relative_dot) && relative_dot < 1.0e-6 || throw(ArgumentError(
                "channel freestream is not tangent to outer wall face $i (relative normal dot=$relative_dot, required < 1e-6)"))
        end
    end
    nwalls > 0 || throw(ArgumentError("segmented channel has no outer wall faces"))
    nothing
end

"""Seed the plain-CPU channel residual reference without modifying the live flow fields."""
function seed_channel_reference!(s::CSol)
    _validate_channel_compat!(s)
    ref = _channel_reference_ambient(s)
    clear_channel_reference!(s)
    _validate_channel_wall_alignment!(s)

    rr=similar(s.rho); ru=similar(s.ux); rv=similar(s.uy); rT=similar(s.T)
    rtk=similar(s.tk); rnut=similar(s.nut)
    rex=similar(s.exx); rxy=similar(s.exy); rey=similar(s.eyy)
    rgux=similar(s.gux); rguy=similar(s.guy); rgvx=similar(s.gvx); rgvy=similar(s.gvy)
    rgTx=similar(s.gTx); rgTy=similar(s.gTy)
    rFir=similar(s.Fi_r); rFimx=similar(s.Fi_mx); rFimy=similar(s.Fi_my); rFie=similar(s.Fi_e)
    rFjr=similar(s.Fj_r); rFjmx=similar(s.Fj_mx); rFjmy=similar(s.Fj_my); rFje=similar(s.Fj_e)
    fill!(rr,ref.rho); fill!(ru,ref.ux); fill!(rv,ref.uy); fill!(rT,ref.T)
    fill!(rtk,ref.tk); fill!(rnut,ref.nut)
    for a in (rex,rxy,rey,rgux,rguy,rgvx,rgvy,rgTx,rgTy,rFir,rFimx,rFimy,rFie,rFjr,rFjmx,rFjmy,rFje)
        fill!(a,0.0f0)
    end

    saved = (rho=s.rho,ux=s.ux,uy=s.uy,T=s.T,tk=s.tk,nut=s.nut,
             exx=s.exx,exy=s.exy,eyy=s.eyy,
             gux=s.gux,guy=s.guy,gvx=s.gvx,gvy=s.gvy,gTx=s.gTx,gTy=s.gTy,
             Fi_r=s.Fi_r,Fi_mx=s.Fi_mx,Fi_my=s.Fi_my,Fi_e=s.Fi_e,
             Fj_r=s.Fj_r,Fj_mx=s.Fj_mx,Fj_my=s.Fj_my,Fj_e=s.Fj_e,
             dc_r=s.dc_r,dc_mx=s.dc_mx,dc_my=s.dc_my,dc_e=s.dc_e,
             wall_flux=s.wall_flux)
    try
        s.rho=rr; s.ux=ru; s.uy=rv; s.T=rT; s.tk=rtk; s.nut=rnut
        s.exx=rex; s.exy=rxy; s.eyy=rey
        s.gux=rgux; s.guy=rguy; s.gvx=rgvx; s.gvy=rgvy; s.gTx=rgTx; s.gTy=rgTy
        s.Fi_r=rFir; s.Fi_mx=rFimx; s.Fi_my=rFimy; s.Fi_e=rFie
        s.Fj_r=rFjr; s.Fj_mx=rFjmx; s.Fj_my=rFjmy; s.Fj_e=rFje
        s.dc_r=s.outer_ref_r; s.dc_mx=s.outer_ref_mx
        s.dc_my=s.outer_ref_my; s.dc_e=s.outer_ref_e
        s.wall_flux=false
        applyBC!(s; wall=:transparent)
        _fluxes!(s)
    finally
        s.rho=saved.rho; s.ux=saved.ux; s.uy=saved.uy; s.T=saved.T
        s.tk=saved.tk; s.nut=saved.nut
        s.exx=saved.exx; s.exy=saved.exy; s.eyy=saved.eyy
        s.gux=saved.gux; s.guy=saved.guy; s.gvx=saved.gvx; s.gvy=saved.gvy
        s.gTx=saved.gTx; s.gTy=saved.gTy
        s.Fi_r=saved.Fi_r; s.Fi_mx=saved.Fi_mx; s.Fi_my=saved.Fi_my; s.Fi_e=saved.Fi_e
        s.Fj_r=saved.Fj_r; s.Fj_mx=saved.Fj_mx; s.Fj_my=saved.Fj_my; s.Fj_e=saved.Fj_e
        s.dc_r=saved.dc_r; s.dc_mx=saved.dc_mx; s.dc_my=saved.dc_my; s.dc_e=saved.dc_e
        s.wall_flux=saved.wall_flux
    end
    s.outer_ref_rho=ref.rho; s.outer_ref_ux=ref.ux
    s.outer_ref_uy=ref.uy; s.outer_ref_T=ref.T
    s.outer_ref_signature=_channel_signature(s)
    s.outer_ref_ready=Int32(1)
    nothing
end

function _validate_channel!(s::CSol)
    _validate_channel_compat!(s)
    s.outer_mode == Int32(0) && return nothing
    s.outer_ref_ready == Int32(1) ||
        throw(ArgumentError("segmented channel residual reference is unseeded; call seed_channel_reference!(s) after configuring the solver"))
    sig = _channel_signature(s)
    if sig != s.outer_ref_signature
        clear_channel_reference!(s)
        throw(ArgumentError("segmented channel residual reference is stale after a solver configuration change; reseed it"))
    end
    nothing
end

function _subtract_channel_reference!(s::CSol)
    Threads.@threads for j in 1:s.NJ
        @inbounds for i in 1:s.NI
            s.dc_r[i,j] -= s.outer_ref_r[i,j]
            s.dc_mx[i,j] -= s.outer_ref_mx[i,j]
            s.dc_my[i,j] -= s.outer_ref_my[i,j]
            s.dc_e[i,j] -= s.outer_ref_e[i,j]
        end
    end
    nothing
end

function step!(s::CSol, cfl::Float32; wall::Symbol=:transparent, dtp::Float32=0.0f0)
    _validate_channel!(s)
    _dtfield!(s, cfl)
    _snapshot!(s)
    for stage in 1:3
        applyBC!(s; wall=wall)
        _fluxes!(s)
        s.outer_mode != Int32(0) && _subtract_channel_reference!(s)
        if s.irsj == Int32(1)
            _irsj_component!(s, s.dc_r); _irsj_component!(s, s.dc_mx)
            _irsj_component!(s, s.dc_my); _irsj_component!(s, s.dc_e)
        end
        _update!(s, stage, dtp; preserve_ref=s.outer_mode != Int32(0))
    end
    if s.sfd == Int32(1)                            # SFD post-step (primitives)
        g2, NI2, NJ2 = s.g, s.NI, s.NJ
        χ = s.sfd_chi; α = s.sfd_alpha
        Threads.@threads for j in 1:NJ2
            for i in 1:NI2
                ia=i+g2; ja=j+g2
                @inbounds begin
                    s.sb_r[ia,ja] += α*(s.rho[ia,ja] - s.sb_r[ia,ja]);  s.rho[ia,ja] -= χ*(s.rho[ia,ja] - s.sb_r[ia,ja])
                    s.sb_u[ia,ja] += α*(s.ux[ia,ja]  - s.sb_u[ia,ja]);  s.ux[ia,ja]  -= χ*(s.ux[ia,ja]  - s.sb_u[ia,ja])
                    s.sb_v[ia,ja] += α*(s.uy[ia,ja]  - s.sb_v[ia,ja]);  s.uy[ia,ja]  -= χ*(s.uy[ia,ja]  - s.sb_v[ia,ja])
                    s.sb_T[ia,ja] += α*(s.T[ia,ja]   - s.sb_T[ia,ja]);  s.T[ia,ja]   -= χ*(s.T[ia,ja]   - s.sb_T[ia,ja])
                end
            end
        end
    end
    # Refresh ghosts from the final RK/SFD state for any post-step consumer. Legacy mode keeps its
    # historical call count and arithmetic unchanged.
    s.outer_mode != Int32(0) && applyBC!(s; wall=wall)
    nothing
end

# seed the SFD filter state from the current fields (call once when enabling)
function sfd_seed!(s::CSol)
    copyto!(s.sb_r, s.rho); copyto!(s.sb_u, s.ux); copyto!(s.sb_v, s.uy); copyto!(s.sb_T, s.T)
    nothing
end

# dual-time history rotation: U^(n-1) <- U^n <- current state. Call once after set_freestream!
# (twice-seeded start = BDF1-class first step) and then once per PHYSICAL step after inner convergence.
function hist_cycle!(s::CSol)
    g = s.g
    Threads.@threads for j in 1:s.NJ
        for i in 1:s.NI
            ia=i+g; ja=j+g
            s.um_r[i,j]=s.un_r[i,j]; s.um_mx[i,j]=s.un_mx[i,j]; s.um_my[i,j]=s.un_my[i,j]; s.um_E[i,j]=s.un_E[i,j]
            r=s.rho[ia,ja]; u=s.ux[ia,ja]; v=s.uy[ia,ja]
            s.un_r[i,j]=r; s.un_mx[i,j]=r*u; s.un_my[i,j]=r*v
            s.un_E[i,j]=r*(s.eint*s.T[ia,ja]+0.5f0*(u*u+v*v))
        end
    end
    nothing
end

residual_linf(s::CSol) = maximum(abs, s.dc_r)

# ---- A5: low-Re integrate-to-wall SST k-ω (Menter 2003) ---------------------------------------
# Loosely coupled: one transport update per flow pseudo-step, on the flow's LTS clock (dtf).
# NO wall-function band (y+~1 mesh): k→0 at the wall via mirrored ghosts; ω OVERWRITTEN analytically
# on the first two wall rows ω = 6ν/(β₁·d²) (never extrapolate a 1/d² profile). Point-implicit
# destruction (the ω_wall ~1e6 scale would pin an explicit dt). Diffusion = standard FV Laplacian
# (face S/dn differences). Trip: Pk gated by gammat. ν_t under-relaxed 20%/step into `nut` which the
# viscous face flux reads as μ_eff = ρ(ν+ν_t_face). SCRATCH: rk_r/rk_mx hold k_new/ω_new during the
# sweep (free between step! calls — documented reuse, do not call inside step!).
function set_sst!(s::CSol; tu_inf::Float64=0.001, mu_t_ratio::Float64=0.1)
    U = sqrt(Float64(s.uxi)^2 + Float64(s.uyi)^2)
    s.k_inf = Float32(1.5*(tu_inf*U)^2)
    s.w_inf = Float32(max(s.k_inf/(mu_t_ratio*Float64(s.nu_lam)), 1e-6))
    fill!(s.tk, s.k_inf); fill!(s.tom, s.w_inf); fill!(s.nut, s.k_inf/s.w_inf)
    fill!(s.tnl, s.k_inf/s.w_inf)
    s.sst = true
    clear_channel_reference!(s)
    nothing
end

function sst_step!(s::CSol; corrected_wall_metric::Bool=false)
    s.sst || return
    g, NI, NJ, NIW, NIF = s.g, s.NI, s.NJ, s.NIW, s.NIF
    tk, tom, nut = s.tk, s.tom, s.nut
    βs=0.09f0; a1=0.31f0
    σk1=0.85f0; σw1=0.5f0; β1=0.075f0; γ1=0.5532f0
    σk2=1.0f0;  σw2=0.856f0; β2=0.0828f0; γ2=0.4403f0
    ν = s.nu_lam
    # ghosts for k/ω/νt: wake-cut exchange, wall (k mirror-negated ⇒ k_wall=0; ω/νt zero-gradient —
    # the analytic rows dominate), far-field freestream, outlet zero-gradient
    tnl = s.tnl
    for k in 1:g, i in 1:NI
        if i <= NIW || i > NIW+NIF
            ip = NI+1-i
            tk[i+g, g+1-k] = tk[ip+g, k+g]; tom[i+g, g+1-k] = tom[ip+g, k+g]; nut[i+g, g+1-k] = nut[ip+g, k+g]
            tnl[i+g, g+1-k] = tnl[ip+g, k+g]
        else
            tk[i+g, g+1-k] = -tk[i+g, g+k]; tom[i+g, g+1-k] = tom[i+g, g+k]; nut[i+g, g+1-k] = 0.0f0
            tnl[i+g, g+1-k] = 0.0f0
        end
        tk[i+g, NJ+g+k] = s.k_inf; tom[i+g, NJ+g+k] = s.w_inf; nut[i+g, NJ+g+k] = s.k_inf/s.w_inf
        tnl[i+g, NJ+g+k] = s.k_inf/s.w_inf
    end
    if s.outer_mode != Int32(0)
        @inbounds for k in 1:g, i in 1:NI
            s.outer_face_kind[i] >= 0x02 || continue
            js = NJ + g + 1 - k; jd = NJ + g + k
            tk[i+g, jd] = tk[i+g, js]
            tom[i+g, jd] = tom[i+g, js]
            nut[i+g, jd] = nut[i+g, js]
            tnl[i+g, jd] = tnl[i+g, js]
        end
    end
    for k in 1:g, jj in 1:NJ+2g
        tk[g+1-k, jj] = tk[g+1, jj]; tom[g+1-k, jj] = tom[g+1, jj]; nut[g+1-k, jj] = nut[g+1, jj]
        tk[NI+g+k, jj] = tk[NI+g, jj]; tom[NI+g+k, jj] = tom[NI+g, jj]; nut[NI+g+k, jj] = nut[NI+g, jj]
        tnl[g+1-k, jj] = tnl[g+1, jj]; tnl[NI+g+k, jj] = tnl[NI+g, jj]
    end
    knew = s.rk_r; wnew = s.rk_mx; nutl = s.rk_my       # scratch reuse (see docstring)
    Threads.@threads for j in 1:NJ
        for i in 1:NI
            ia=i+g; ja=j+g
            k0 = tk[ia,ja]; w0 = max(tom[ia,ja], 1.0f-6); d = max(s.wdist[i,j], 1.0f-8)
            # The corrected metric matches the state ghost location at the
            # inner row; the historical stencil remains the default.
            xxi, yxi, xet, yet = _sst_index_metrics(
                s.xc, s.yc, s.Sjx, s.Sjy, s.wdist,
                i, j, NI, NJ, s.NIW, s.NIF, Int32(corrected_wall_metric))
            J = xxi*yet - xet*yxi; J = abs(J) < 1.0f-30 ? 1.0f-30 : J
            duxi = 0.5f0*(s.ux[ia+1,ja]-s.ux[ia-1,ja]); duet = 0.5f0*(s.ux[ia,ja+1]-s.ux[ia,ja-1])
            dvxi = 0.5f0*(s.uy[ia+1,ja]-s.uy[ia-1,ja]); dvet = 0.5f0*(s.uy[ia,ja+1]-s.uy[ia,ja-1])
            dudx = ( duxi*yet - duet*yxi)/J; dudy = (-duxi*xet + duet*xxi)/J
            dvdx = ( dvxi*yet - dvet*yxi)/J; dvdy = (-dvxi*xet + dvet*xxi)/J
            Sxy = 0.5f0*(dudy+dvdx); S2 = 2.0f0*(dudx*dudx + dvdy*dvdy + 2.0f0*Sxy*Sxy); Ssr = sqrt(S2)
            Ω = abs(dudy - dvdx)
            # blending
            sqk = sqrt(max(k0, 0.0f0))
            dkxi = 0.5f0*(tk[ia+1,ja]-tk[ia-1,ja]); dket = 0.5f0*(tk[ia,ja+1]-tk[ia,ja-1])
            dwxi = 0.5f0*(tom[ia+1,ja]-tom[ia-1,ja]); dwet = 0.5f0*(tom[ia,ja+1]-tom[ia,ja-1])
            dkdx = ( dkxi*yet - dket*yxi)/J; dkdy = (-dkxi*xet + dket*xxi)/J
            dwdx = ( dwxi*yet - dwet*yxi)/J; dwdy = (-dwxi*xet + dwet*xxi)/J
            gkw = dkdx*dwdx + dkdy*dwdy
            CDkw = max(2.0f0*σw2/w0*gkw, 1.0f-10)
            arg1 = min(max(sqk/(βs*w0*d), 500.0f0*ν/(d*d*w0)), 4.0f0*σw2*k0/(CDkw*d*d))
            arg1 = min(arg1, 4.0f0); F1 = tanh(arg1^4)
            arg2 = min(max(2.0f0*sqk/(βs*w0*d), 500.0f0*ν/(d*d*w0)), 4.0f0); F2 = tanh(arg2^2)
            nut_t = a1*k0/max(a1*w0, Ssr*F2)
            # production (Kato-Launder S·Ω: kills spurious k at the shock/stagnation), trip-gated
            Pk = min(nut_t*Ssr*Ω, 10.0f0*βs*k0*w0) * s.gammat[i,j]
            σk = F1*σk1 + (1.0f0-F1)*σk2; σw = F1*σw1 + (1.0f0-F1)*σw2
            βc = F1*β1 + (1.0f0-F1)*β2;   γc = F1*γ1 + (1.0f0-F1)*γ2
            # FV Laplacian diffusion (face S/dn) + first-order upwind advection on the contravariant velocity
            Dk = 0.0f0; Dw = 0.0f0; difsum = 0.0f0
            # E/W/N/S faces: (nb, S, dn)
            for (di, dj, Sx, Sy) in ((1,0,s.Six[i+1,j],s.Siy[i+1,j]), (-1,0,s.Six[i,j],s.Siy[i,j]),
                                     (0,1,s.Sjx[i,j+1],s.Sjy[i,j+1]), (0,-1,s.Sjx[i,j],s.Sjy[i,j]))
                inb = i+di; jnb = j+dj
                Sf = sqrt(Sx*Sx+Sy*Sy)
                if 1 <= inb <= NI && 1 <= jnb <= NJ
                    dxc = s.xc[inb,jnb]-s.xc[i,j]; dyc = s.yc[inb,jnb]-s.yc[i,j]
                    dn = sqrt(dxc*dxc+dyc*dyc) + 1.0f-30
                elseif j == 1 && dj == -1 && (i <= NIW || i > NIW+NIF)
                    # The wake-cut ghost is a real cell in the paired leg, not
                    # a reflected boundary state. Match its diffusion distance.
                    ipcut = NI+1-i
                    dxc = s.xc[ipcut,1]-s.xc[i,1]
                    dyc = s.yc[ipcut,1]-s.yc[i,1]
                    dn = sqrt(dxc*dxc+dyc*dyc) + 1.0f-30
                else
                    dn = s.vol[i,j]/Sf + 1.0f-30    # boundary face: center-to-ghost ≈ the cell width
                end                                  # (clamped-center dn was 0 ⇒ S/dn=1e27 ⇒ the smoke NaN)
                nut_f = 0.5f0*(nut[ia,ja] + nut[ia+di,ja+dj])
                Dk += (ν+σk*nut_f)*Sf/dn*(tk[ia+di,ja+dj]-k0)
                Dw += (ν+σw*nut_f)*Sf/dn*(tom[ia+di,ja+dj]-w0)
                difsum += (ν+σk*nut_f)*Sf/dn
            end
            Dk /= s.vol[i,j]; Dw /= s.vol[i,j]
            u0 = s.ux[ia,ja]; v0 = s.uy[ia,ja]
            # FIRST-ORDER UPWIND advection on the contravariant velocities (index-space donor cell —
            # central advection of the ω 1/d² profile blew up instantly: NaN in the 2000-step smoke)
            Uci = (u0*0.5f0*(s.Six[i,j]+s.Six[i+1,j]) + v0*0.5f0*(s.Siy[i,j]+s.Siy[i+1,j]))/s.vol[i,j]
            Ucj = (u0*0.5f0*(s.Sjx[i,j]+s.Sjx[i,j+1]) + v0*0.5f0*(s.Sjy[i,j]+s.Sjy[i,j+1]))/s.vol[i,j]
            advk = (Uci >= 0.0f0 ? Uci*(k0-tk[ia-1,ja]) : Uci*(tk[ia+1,ja]-k0)) +
                   (Ucj >= 0.0f0 ? Ucj*(k0-tk[ia,ja-1]) : Ucj*(tk[ia,ja+1]-k0))
            advw = (Uci >= 0.0f0 ? Uci*(w0-tom[ia-1,ja]) : Uci*(tom[ia+1,ja]-w0)) +
                   (Ucj >= 0.0f0 ? Ucj*(w0-tom[ia,ja-1]) : Ucj*(tom[ia,ja+1]-w0))
            dtl = s.dtf[i,j]
            if s.irsj == Int32(1)
                # SST SELF-CLOCK: with IRS-J the flow dtf is i-restricted only (huge in the BL) —
                # the SST explicit j-diffusion/advection needs its own conservative dt
                dtl = min(dtl, 0.35f0/(abs(Uci)+abs(Ucj)+1.0f-30), 0.4f0*s.vol[i,j]/max(difsum,1.0f-30))
            end
            # point-implicit destruction
            kn = (k0 + dtl*(Pk - advk + Dk)) / (1.0f0 + dtl*βs*w0)
            wn = (w0 + dtl*(γc/max(nut_t,1.0f-10)*Pk - advw + Dw + 2.0f0*(1.0f0-F1)*σw2/w0*gkw)) / (1.0f0 + dtl*βc*w0)
            kn = clamp(kn, 1.0f-12, 1.0f0)
            wn = max(wn, 1.0f-6)
            if !(isfinite(kn) && isfinite(wn)); kn = s.k_inf; wn = s.w_inf; end
            knew[ia,ja] = kn; wnew[ia,ja] = wn
            # Menter SST eddy-viscosity DEFINITION nut = a1·k/max(a1·ω, S·F2) — the limiter must gate
            # the EXPORTED nut, not just production (APG/post-shock zones are where it bites; exporting
            # raw k/ω over-predicted post-shock mu_t => BL too full => shock aft + too-fast recovery)
            nutl[ia,ja] = a1*kn/max(a1*wn, Ssr*F2)
        end
    end
    # LAG equation (Olsen-Coakley): D nu_hat/Dt = a0*omega*(nu_eq - nu_hat), a0=0.35 — the LAGGED
    # nu_hat drives the stresses (real turbulence keeps upstream stresses ~10 delta through the
    # shock; equilibrium SST re-energizes instantly => shock too strong + recovery too fast).
    # nu_eq = nutl (the a1-limited value from the sweep). Point-implicit; upwind advection.
    tnlnew = s.rk_E                                     # scratch (free between step! calls)
    if s.lag == Int32(1)
        Threads.@threads for j in 1:NJ
            for i in 1:NI
                ia=i+g; ja=j+g
                wn = max(wnew[ia,ja], 1.0f-6); nueq = nutl[ia,ja]
                u0 = s.ux[ia,ja]; v0 = s.uy[ia,ja]
                Uci = (u0*0.5f0*(s.Six[i,j]+s.Six[i+1,j]) + v0*0.5f0*(s.Siy[i,j]+s.Siy[i+1,j]))/s.vol[i,j]
                Ucj = (u0*0.5f0*(s.Sjx[i,j]+s.Sjx[i,j+1]) + v0*0.5f0*(s.Sjy[i,j]+s.Sjy[i,j+1]))/s.vol[i,j]
                t0v = tnl[ia,ja]
                adv = (Uci >= 0.0f0 ? Uci*(t0v-tnl[ia-1,ja]) : Uci*(tnl[ia+1,ja]-t0v)) +
                      (Ucj >= 0.0f0 ? Ucj*(t0v-tnl[ia,ja-1]) : Ucj*(tnl[ia,ja+1]-t0v))
                dtl = s.dtf[i,j]; a0 = s.lag_a0
                tn = (t0v + dtl*(a0*wn*nueq - adv)) / (1.0f0 + dtl*a0*wn)
                tnlnew[ia,ja] = clamp(tn, 0.0f0, 0.15f0)
            end
        end
    end
    # commit + analytic ω wall rows + ν_t under-relax
    Threads.@threads for j in 1:NJ
        for i in 1:NI
            ia=i+g; ja=j+g
            kn = knew[ia,ja]; wn = wnew[ia,ja]
            if j <= 2 && NIW < i <= NIW+NIF
                d = max(s.wdist[i,j], 1.0f-8)
                wn = 6.0f0*ν/(β1*d*d)                   # Menter low-Re wall ω (analytic rows 1-2)
                nut_t = kn/max(wn, 1.0f-6)              # sublayer: a1·ω >> S·F2, limiter inert; use analytic ω
            else
                nut_t = nutl[ia,ja]                     # a1-limited (SST definition)
            end
            tk[ia,ja] = kn; tom[ia,ja] = wn
            if s.lag == Int32(1) && !(j <= 2 && NIW < i <= NIW+NIF)
                tnl[ia,ja] = tnlnew[ia,ja]
                nut[ia,ja] = tnlnew[ia,ja]              # lagged nu_t drives the stresses directly
            else
                s.lag == Int32(1) && (tnl[ia,ja] = kn/max(wn, 1.0f-6))   # sublayer: lag inert
                nut[ia,ja] += 0.2f0*(min(nut_t, 0.15f0) - nut[ia,ja])
            end
        end
    end
    # WJ-EARSM stage (earsm==1, overrides the nut export): effective nu_t = -(beta1/2)*k*tau_t plus
    # the EXTRA (non-Boussinesq) anisotropy stress tau_extra = -rho*k*beta4*(S*Om*-Om*S*) stored in
    # exx/exy/eyy and added at faces by _fluxes!. v1: Kato-Launder production kept; lag+earsm not combined.
    if s.earsm == Int32(1)
        βst = 0.09f0
        Threads.@threads for j in 1:NJ
            for i in 1:NI
                ia=i+g; ja=j+g
                xxi, yxi, xet, yet = _sst_index_metrics(
                    s.xc, s.yc, s.Sjx, s.Sjy, s.wdist,
                    i, j, NI, NJ, s.NIW, s.NIF, Int32(corrected_wall_metric))
                J = xxi*yet - xet*yxi; J = abs(J) < 1.0f-30 ? 1.0f-30 : J
                duxi = 0.5f0*(s.ux[ia+1,ja]-s.ux[ia-1,ja]); duet = 0.5f0*(s.ux[ia,ja+1]-s.ux[ia,ja-1])
                dvxi = 0.5f0*(s.uy[ia+1,ja]-s.uy[ia-1,ja]); dvet = 0.5f0*(s.uy[ia,ja+1]-s.uy[ia,ja-1])
                dudx = ( duxi*yet - duet*yxi)/J; dudy = (-duxi*xet + duet*xxi)/J
                dvdx = ( dvxi*yet - dvet*yxi)/J; dvdy = (-dvxi*xet + dvet*xxi)/J
                k0 = max(tk[ia,ja], 1.0f-12); w0 = max(tom[ia,ja], 1.0f-6)
                tau_t = max(1.0f0/(βst*w0), 6.0f0*sqrt(ν/(βst*k0*w0)))
                b1, b4, sxx, syy, sxy, oxy = _wj_earsm_2d(dudx, dudy, dvdx, dvdy, tau_t)
                # explicit-EARSM stabilization: 0.2 pseudo-time under-relax on ALL outputs (fixed
                # point unchanged) + realizability bound |tau_extra| <= 0.6 rho k — instant export
                # NaN'd the transonic shock (the classic explicit anisotropy instability)
                nut[ia,ja] += 0.2f0*(clamp(-0.5f0*b1*k0*tau_t, 0.0f0, 0.15f0) - nut[ia,ja])
                ck = s.rho[ia,ja]*k0*b4; bnd = 0.6f0*s.rho[ia,ja]*k0
                s.exx[i,j] += 0.2f0*(clamp( 2.0f0*ck*sxy*oxy,      -bnd, bnd) - s.exx[i,j])
                s.exy[i,j] += 0.2f0*(clamp(-ck*(sxx-syy)*oxy,      -bnd, bnd) - s.exy[i,j])
                s.eyy[i,j] += 0.2f0*(clamp(-2.0f0*ck*sxy*oxy,      -bnd, bnd) - s.eyy[i,j])
            end
        end
    end
    nothing
end

# pressure loads over the foil j=1 faces. Force on the body = −Σ p_w·S⃗_j(i,1) (S⃗_j points from the
# wall INTO the fluid; uniform p ⇒ closed-loop sum = 0 exactly). Returns (Cl, Cd) in the wind frame.
function loads(s::CSol)
    g = s.g; Fx = 0.0; Fy = 0.0
    for i in s.NIW+1:s.NIW+s.NIF
        pw = Float64(s.rho[i+g, g+1])*Float64(CS2)*Float64(s.T[i+g, g+1])
        Fx -= pw*Float64(s.Sjx[i,1]); Fy -= pw*Float64(s.Sjy[i,1])
    end
    U2 = Float64(s.uxi)^2 + Float64(s.uyi)^2
    q = 0.5*Float64(s.rho_inf)*U2                       # chord = 1 (chord-normalized mesh)
    # friction (no-slip walls): F⃗_f on the body = Σ μ·(u⃗_int/d1)·S over the foil faces
    if s.nu_lam > 0.0f0
        g2 = s.g
        for i in s.NIW+1:s.NIW+s.NIF
            Sf = sqrt(Float64(s.Sjx[i,1])^2 + Float64(s.Sjy[i,1])^2)
            d1 = 0.5*Float64(s.vol[i,1])/Sf + 1e-30
            muw = Float64(s.rho[i+g2, g2+1])*Float64(s.nu_lam)*Sf/d1
            if s.fullvisc==Int32(1)
                muw *= Float64(_mufac(s.T[i+g2,g2+1],s.suth))
                tx,ty=_wall_stokes_traction(Float64(s.ux[i+g2,g2+1]),Float64(s.uy[i+g2,g2+1]),
                    Float64(s.Sjx[i,1])/Sf,Float64(s.Sjy[i,1])/Sf,muw)
                Fx += tx; Fy += ty
            else
                Fx += muw*Float64(s.ux[i+g2, g2+1]); Fy += muw*Float64(s.uy[i+g2, g2+1])
            end
        end
    end
    Cl = ( Fy*Float64(s.ca) - Fx*Float64(s.sa))/q
    Cd = ( Fx*Float64(s.ca) + Fy*Float64(s.sa))/q
    (Cl, Cd)
end

# ---- gate (a): metric/freestream preservation ------------------------------------------------
# Uniform flow at alpha through the FULL C-mesh (wake cut active, transparent wall): the residual
# must be metric-roundoff-small and the state must not drift. Pass: max|dc_r|/(rho*a*S_ref) < 1e-5.
function gate_freestream(s::CSol, mach::Float64, alpha_deg::Float64; nsteps::Int=100, wall::Symbol=:transparent)
    set_freestream!(s, mach, alpha_deg)
    s.outer_mode != Int32(0) && seed_channel_reference!(s)
    drift0 = (copy(s.rho), copy(s.ux))
    r0 = 0.0
    for n in 1:nsteps
        step!(s, 0.5f0; wall=wall)
        n == 1 && (r0 = residual_linf(s))
    end
    a = sqrt((1 + 2/(3+s.drot))*CS2)
    Sref = maximum(sqrt.(s.Six.^2 .+ s.Siy.^2))
    rnorm = residual_linf(s)/(1.0*a*Sref)
    drift = maximum(abs, s.ux .- drift0[2])
    (rnorm=rnorm, r_first=r0, drift=drift)
end

end # module
