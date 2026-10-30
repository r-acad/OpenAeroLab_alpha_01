# ==============================================================================
# Particles-on-Demand — 3D single-level uniform-grid driver  [steps 5-6: BCs+ramp]
# ==============================================================================
# A self-contained host-side stepping driver around the validated conservative
# kinetic-FV kernels (pond_fv3d_gpu.jl), adding the machinery the prototype lacked
# (and which the handover identified as the cause of the 2D-cylinder NaNs):
#   • boundaries-from-moments — Dirichlet inlet (ρ∞,U_ramp), transmissive outlet/lateral
#   • startup ramp            — U_inlet ramps 0→U∞ over ramp_steps (no supersonic slam)
#   • immersed obstacle       — VOLUME PENALIZATION (Brinkman), NOT a staircase flux wall:
#       u ← u/(1+dt·χ/η) inside the body. Mass-conserving (only momentum is damped),
#       smooth across the surface (graded χ), and degrades gracefully — avoiding the
#       manifold-edge / staircase blowup that killed the reflective-flux prototypes.
#       The momentum removed per step IS the body force ⇒ drag falls out for free.
# All boundary/penalization ops are KA kernels ⇒ GPU-resident, backend-agnostic.
# ==============================================================================

using KernelAbstractions

# ---- Dirichlet inlet on the −x face (planes i ≤ ng): set (ρ∞, Uin, 0, 0) ----
@kernel function _pond_inlet_x!(rho, ux, uy, uz, rho_inf::Float32, Uin::Float32, ng::Int32)
    i, j, k = @index(Global, NTuple)
    @inbounds if i <= ng
        rho[i,j,k] = rho_inf; ux[i,j,k] = Uin; uy[i,j,k] = 0.0f0; uz[i,j,k] = 0.0f0
    end
end

# ---- transmissive (zero-gradient) outlet on the +x face: copy plane NX-ng from interior ----
@kernel function _pond_outlet_x!(rho, ux, uy, uz, NX::Int32, ng::Int32)
    i, j, k = @index(Global, NTuple)
    nx = Int(NX)
    @inbounds if i > nx - ng
        src = nx - ng
        rho[i,j,k] = rho[src,j,k]; ux[i,j,k] = ux[src,j,k]; uy[i,j,k] = uy[src,j,k]; uz[i,j,k] = uz[src,j,k]
    end
end

# ---- lateral FAR-FIELD on the y,z faces: prescribe freestream (ρ∞,U∞,0,0). For EXTERNAL
# aero the lateral walls must be freestream, NOT zero-gradient — zero-gradient turns the
# domain into a duct that chokes and over-densifies the whole field at supersonic Ma. ----
@kernel function _pond_farfield_yz!(rho, ux, uy, uz, rho_inf::Float32, Uin::Float32,
                                    NY::Int32, NZ::Int32, ng::Int32)
    i, j, k = @index(Global, NTuple)
    ny = Int(NY); nz = Int(NZ)
    @inbounds if (j <= ng) || (j > ny - ng) || (k <= ng) || (k > nz - ng)
        rho[i,j,k] = rho_inf; ux[i,j,k] = Uin; uy[i,j,k] = 0.0f0; uz[i,j,k] = 0.0f0
    end
end

# ---- SPONGE lateral far-field (NON-REFLECTING): over a band of `sw` cells near the y,z boundaries,
# relax (ρ,u) → freestream with a strength σ that ramps 0 (band edge) → σ_max (boundary). This ABSORBS
# the outgoing Mach/acoustic waves before they reach the boundary and reflect (the hard Dirichlet
# above reflects them ⇒ the diagonal cross-hatch). The interior (d ≥ sw) is UNTOUCHED ⇒ accuracy-
# preserving (it only removes the spurious reflected waves). ----
@kernel function _pond_sponge_yz!(rho, ux, uy, uz, rho_inf::Float32, Uin::Float32,
                                  NY::Int32, NZ::Int32, sw::Int32, smax::Float32)
    i, j, k = @index(Global, NTuple)
    ny = Int(NY); nz = Int(NZ); w = Int(sw)
    @inbounds begin
        dy = j-1 < ny-j ? j-1 : ny-j; dz = k-1 < nz-k ? k-1 : nz-k
        d = dy < dz ? dy : dz                       # distance to the nearest lateral boundary
        if d < w
            t = 1.0f0 - Float32(d)/Float32(w)       # 0 at band edge → 1 at the boundary
            σ = smax*t*t                            # quadratic ramp (smooth absorption)
            rho[i,j,k] += σ*(rho_inf - rho[i,j,k])
            ux[i,j,k]  += σ*(Uin     - ux[i,j,k])
            uy[i,j,k]  += σ*(0.0f0   - uy[i,j,k])
            uz[i,j,k]  += σ*(0.0f0   - uz[i,j,k])
        end
    end
end

# ---- QUASI-2D lateral far-field: prescribe freestream on the y faces ONLY (z is the thin extrusion
# axis, left zero-gradient by the flux clamp so the 2D solution stays uniform in z). ----
@kernel function _pond_farfield_y!(rho, ux, uy, uz, rho_inf::Float32, Uin::Float32, NY::Int32, ng::Int32)
    i, j, k = @index(Global, NTuple)
    ny = Int(NY)
    @inbounds if (j <= ng) || (j > ny - ng)
        rho[i,j,k] = rho_inf; ux[i,j,k] = Uin; uy[i,j,k] = 0.0f0; uz[i,j,k] = 0.0f0
    end
end

# ---- QUASI-2D SPONGE on the y faces ONLY (non-reflecting absorber) ----
@kernel function _pond_sponge_y!(rho, ux, uy, uz, rho_inf::Float32, Uin::Float32, NY::Int32, sw::Int32, smax::Float32)
    i, j, k = @index(Global, NTuple)
    ny = Int(NY); w = Int(sw)
    @inbounds begin
        dy = j-1 < ny-j ? j-1 : ny-j
        if dy < w
            t = 1.0f0 - Float32(dy)/Float32(w); σ = smax*t*t
            rho[i,j,k] += σ*(rho_inf - rho[i,j,k]); ux[i,j,k] += σ*(Uin - ux[i,j,k])
            uy[i,j,k]  += σ*(0.0f0 - uy[i,j,k]);    uz[i,j,k] += σ*(0.0f0 - uz[i,j,k])
        end
    end
end

# ---- volume penalization (Brinkman) inside the body: u ← u·invfac, invfac = 1/(1+dt·χ/η) ----
# `chi` is the graded solid-fraction field (1 inside, 0 outside, smooth over ~1 cell).
@kernel function _pond_penalize!(ux, uy, uz, @Const(chi), invscale::Float32)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        c = chi[i,j,k]
        if c > 0.0f0
            f = 1.0f0 / (1.0f0 + invscale * c)   # invscale = dt/η
            ux[i,j,k] *= f; uy[i,j,k] *= f; uz[i,j,k] *= f
        end
    end
end

# graded solid mask for a sphere centred at (cx,cy,cz) radius R, smoothed over `wband` cells.
function pond_sphere_chi(NX, NY, NZ, cx, cy, cz, R, dx; wband=1.0f0)
    χ = zeros(Float32, NX, NY, NZ)
    for k in 1:NZ, j in 1:NY, i in 1:NX
        x = (i-0.5f0)*dx; y = (j-0.5f0)*dx; z = (k-0.5f0)*dx
        d = sqrt((x-cx)^2 + (y-cy)^2 + (z-cz)^2)
        χ[i,j,k] = clamp((R - d)/(wband*dx) + 0.5f0, 0.0f0, 1.0f0)
    end
    return χ
end

# graded solid mask for a CYLINDER (axis along z, infinite extrusion): circle of radius R in the
# x-y plane at (cx,cy), uniform in z. Use with quasi-2D (thin NZ, z zero-gradient).
function pond_cylinder_chi(NX, NY, NZ, cx, cy, R, dx; wband=1.0f0)
    χ = zeros(Float32, NX, NY, NZ)
    for k in 1:NZ, j in 1:NY, i in 1:NX
        x = (i-0.5f0)*dx; y = (j-0.5f0)*dx
        d = sqrt((x-cx)^2 + (y-cy)^2)
        χ[i,j,k] = clamp((R - d)/(wband*dx) + 0.5f0, 0.0f0, 1.0f0)
    end
    return χ
end

"""
    PonDFVDriver — holds the device state + lattice arrays + config for a uniform-grid run.
"""
mutable struct PonDFVDriver{A,V}
    backend
    rho::A; ux::A; uy::A; uz::A             # current state (device)
    rhon::A; uxn::A; uyn::A; uzn::A         # scratch (device)
    chi::A                                  # graded solid fraction (device)
    yw::A; nxw::A; nyw::A; nzw::A            # WALL MODEL: signed wall distance + outward normal (device)
    cxa::V; cya::V; cza::V; wa::V           # lattice arrays (device)
    NX::Int; NY::Int; NZ::Int
    dx::Float32; dt::Float32; cs2::Float32
    rho_inf::Float32; U_inf::Float32
    eta::Float32                            # penalization permeability
    tau::Float32                            # DUGKS molecular relaxation time (>0 ⇒ viscous; 0 ⇒ inviscid)
    cs_smag::Float32                        # Smagorinsky LES coefficient (>0 ⇒ subgrid eddy viscosity)
    nu_wall::Float32; y_wm::Float32          # WALL MODEL: molecular ν for the log-law + near-wall band
    use_wm::Bool                            # WMLES (wall model) instead of plain no-slip penalization
    svv_eps::Float32                        # SVV hyperviscosity coefficient (>0 ⇒ k⁴-selective ripple kill)
    sponge_w::Int; sponge_max::Float32      # SPONGE lateral far-field (non-reflecting) band width + strength
    use_weno::Bool                          # WENO5-Z interface reconstruction (less-dispersive flux; needs pond_fv3d_weno.jl)
    quasi2d::Bool                           # quasi-2D: lateral far-field/sponge on y ONLY (z = thin zero-gradient extrusion)
    ramp_steps::Int; ng::Int
end

function pond_fv3d_driver(backend, dev, NX, NY, NZ; dx, dt, cs2, rho_inf, U_inf,
                          chi_host, eta=1.0f-3, tau=0.0f0, cs_smag=0.0f0, ramp_steps=400, ng=2,
                          wallinfo=nothing, nu_wall=1.0f-3, y_wm=1.5f0, svv_eps=0.0f0,
                          sponge_w=0, sponge_max=0.5f0, use_weno=false, quasi2d=false,
                          u_init_freestream=false, cxv, cyv, czv, wv)
    rho = dev(fill(Float32(rho_inf), NX, NY, NZ))
    # IC: uniform freestream (ux=U∞ everywhere) avoids the slow inlet-convection startup (the bulk flow
    # only moves ~U·dt≈0.17 cells/step) ⇒ the cylinder sees flow immediately and the bow shock forms in
    # shock-establishment time, not domain-traversal time. Else rest (u=0) + inlet ramp (legacy/sphere).
    ux0 = u_init_freestream ? Float32(U_inf) : 0.0f0
    ux  = dev(fill(ux0, NX, NY, NZ)); uy = dev(zeros(Float32,NX,NY,NZ)); uz = dev(zeros(Float32,NX,NY,NZ))
    rhon = dev(zeros(Float32,NX,NY,NZ)); uxn = dev(zeros(Float32,NX,NY,NZ))
    uyn = dev(zeros(Float32,NX,NY,NZ)); uzn = dev(zeros(Float32,NX,NY,NZ))
    chi = dev(chi_host)
    cxa = dev(Float32.(cxv)); cya = dev(Float32.(cyv)); cza = dev(Float32.(czv)); wa = dev(Float32.(wv))
    # WALL MODEL fields (wallinfo = (y, nx, ny, nz) host arrays); stubs + use_wm=false when absent.
    use_wm = wallinfo !== nothing
    yh,nxh,nyh,nzh = use_wm ? wallinfo : (zeros(Float32,1,1,1),zeros(Float32,1,1,1),zeros(Float32,1,1,1),zeros(Float32,1,1,1))
    yw=dev(yh); nxw=dev(nxh); nyw=dev(nyh); nzw=dev(nzh)
    return PonDFVDriver{typeof(rho),typeof(cxa)}(backend, rho,ux,uy,uz, rhon,uxn,uyn,uzn, chi, yw,nxw,nyw,nzw,
        cxa,cya,cza,wa, NX,NY,NZ, Float32(dx),Float32(dt),Float32(cs2),
        Float32(rho_inf),Float32(U_inf), Float32(eta), Float32(tau), Float32(cs_smag),
        Float32(nu_wall), Float32(y_wm), use_wm, Float32(svv_eps), Int(sponge_w), Float32(sponge_max), Bool(use_weno), Bool(quasi2d), ramp_steps, ng)
end

# one driver step at global step index `t` (1-based): FV update → penalize → inlet → outlet.
function pond_fv3d_advance!(d::PonDFVDriver, t::Int)
    nd = (d.NX, d.NY, d.NZ)
    # 1) conservative FV update (writes new state into the scratch arrays).
    #    τ>0 ⇒ DUGKS viscous flux (controlled physical ν); τ=0 ⇒ inviscid KFVS (validated baseline).
    if d.use_weno
        # WENO5-Z interface reconstruction (less-dispersive flux) + DUGKS τ-viscosity + LES
        pond_fv3d_step_weno!(d.backend, d.rhon,d.uxn,d.uyn,d.uzn, d.rho,d.ux,d.uy,d.uz,
                             d.cxa,d.cya,d.cza,d.wa, d.cs2,d.dt,d.dx, d.tau, d.cs_smag, d.NX,d.NY,d.NZ)
    elseif d.tau > 0.0f0 || d.cs_smag > 0.0f0
        pond_fv3d_step_visc!(d.backend, d.rhon,d.uxn,d.uyn,d.uzn, d.rho,d.ux,d.uy,d.uz,
                             d.cxa,d.cya,d.cza,d.wa, d.cs2,d.dt,d.dx, d.tau, d.cs_smag, d.NX,d.NY,d.NZ)
    else
        pond_fv3d_step!(d.backend, d.rhon,d.uxn,d.uyn,d.uzn, d.rho,d.ux,d.uy,d.uz,
                        d.cxa,d.cya,d.cza,d.wa, d.cs2,d.dt,d.dx, d.NX,d.NY,d.NZ)
    end
    d.rho,d.rhon = d.rhon,d.rho; d.ux,d.uxn = d.uxn,d.ux; d.uy,d.uyn = d.uyn,d.uy; d.uz,d.uzn = d.uzn,d.uz
    # 1b) SVV hyperviscosity: k⁴-selective ⇒ kills the 2Δ ripples, resolved scales untouched (conservative).
    if d.svv_eps > 0.0f0
        pond_apply_svv!(d.backend, d.ux,d.uy,d.uz, d.rho, d.svv_eps, d.NX,d.NY,d.NZ)
    end
    # 2) body BC: WMLES wall model (slip + modeled log-law stress ⇒ high-Re without resolving the BL),
    #    or the plain no-slip volume penalization. Wall model needs pond_fv3d_wallmodel.jl loaded.
    if d.use_wm
        pond_apply_wallmodel!(d.backend, d.rho,d.ux,d.uy,d.uz, d.yw,d.nxw,d.nyw,d.nzw,
                              d.nu_wall, d.dt, d.dx, d.y_wm, d.eta)
    else
        invscale = d.dt / d.eta
        _pond_penalize!(d.backend)(d.ux,d.uy,d.uz, d.chi, Float32(invscale); ndrange=nd)
        KernelAbstractions.synchronize(d.backend)
    end
    # 3) Dirichlet inlet with startup ramp (smooth 0→U∞)
    frac = d.ramp_steps <= 0 ? 1.0f0 : Float32(min(1.0, t / d.ramp_steps))
    Uin = d.U_inf * (0.5f0 - 0.5f0*cos(Float32(pi)*frac))   # smooth cosine ramp
    _pond_inlet_x!(d.backend)(d.rho,d.ux,d.uy,d.uz, d.rho_inf, Uin, Int32(d.ng); ndrange=nd)
    KernelAbstractions.synchronize(d.backend)
    # 4) lateral far-field: SPONGE (non-reflecting, absorbs outgoing Mach waves) if sponge_w>0,
    #    else hard freestream Dirichlet. Both keep it an EXTERNAL flow (not a choking duct).
    if d.quasi2d
        # quasi-2D: y faces only (z is the thin extrusion, kept uniform by the flux zero-gradient clamp)
        if d.sponge_w > 0
            _pond_sponge_y!(d.backend)(d.rho,d.ux,d.uy,d.uz, d.rho_inf, Uin, Int32(d.NY), Int32(d.sponge_w), d.sponge_max; ndrange=nd)
        else
            _pond_farfield_y!(d.backend)(d.rho,d.ux,d.uy,d.uz, d.rho_inf, Uin, Int32(d.NY), Int32(d.ng); ndrange=nd)
        end
    elseif d.sponge_w > 0
        _pond_sponge_yz!(d.backend)(d.rho,d.ux,d.uy,d.uz, d.rho_inf, Uin,
                                    Int32(d.NY), Int32(d.NZ), Int32(d.sponge_w), d.sponge_max; ndrange=nd)
    else
        _pond_farfield_yz!(d.backend)(d.rho,d.ux,d.uy,d.uz, d.rho_inf, Uin,
                                      Int32(d.NY), Int32(d.NZ), Int32(d.ng); ndrange=nd)
    end
    KernelAbstractions.synchronize(d.backend)
    # 5) transmissive outlet
    _pond_outlet_x!(d.backend)(d.rho,d.ux,d.uy,d.uz, Int32(d.NX), Int32(d.ng); ndrange=nd)
    KernelAbstractions.synchronize(d.backend)
    return Uin
end

# drag force on the body via the penalization momentum sink: F_x = Σ_body ρ·(1−f)·u_x / dt.
# Returns (Fx, Fy, Fz) in lattice units. Computed from the CURRENT (already-penalized) state
# would read ~0; instead we evaluate the sink from the pre-penalization velocity, so this is
# meant to be called right AFTER pond_fv3d_step! and BEFORE penalization — see test usage,
# or use the field-integral pressure-drag estimate in the test. Here: momentum-deficit form.
# Physical drag by the CONTROL-VOLUME momentum-flux theorem (the standard CFD drag, independent
# of the penalization-band details): for a box S enclosing the body, the streamwise force the
# fluid exerts on the body is  D = −∮_S (ρ u_x (u·n) + p n_x) dS,  p = ρ·cs² (isothermal).
# `mx` cells inset from each domain face keep S inside the far-field BC layer and the body.
function pond_drag_controlvolume(d::PonDFVDriver; inset::Int=8)
    rho = Array(d.rho); ux = Array(d.ux); uy = Array(d.uy); uz = Array(d.uz)
    cs2 = Float64(d.cs2); dA = Float64(d.dx)^2
    i1 = inset; i2 = d.NX - inset; j1 = inset; j2 = d.NY - inset; k1 = inset; k2 = d.NZ - inset
    p(I) = Float64(rho[I...])*cs2
    s = 0.0
    @inbounds begin
        for k in k1:k2, j in j1:j2                       # ±x faces (n = ∓x̂)
            I2=(i2,j,k); I1=(i1,j,k)
            s += (Float64(rho[I2...])*Float64(ux[I2...])^2 + p(I2)) * dA      # +x: ρuₓ² + p
            s -= (Float64(rho[I1...])*Float64(ux[I1...])^2 + p(I1)) * dA      # −x
        end
        for k in k1:k2, i in i1:i2                       # ±y faces (n = ∓ŷ); n_x=0 ⇒ only ρuₓu_y
            Jp=(i,j2,k); Jm=(i,j1,k)
            s += Float64(rho[Jp...])*Float64(ux[Jp...])*Float64(uy[Jp...]) * dA
            s -= Float64(rho[Jm...])*Float64(ux[Jm...])*Float64(uy[Jm...]) * dA
        end
        for j in j1:j2, i in i1:i2                       # ±z faces (n = ∓ẑ); only ρuₓu_z
            Kp=(i,j,k2); Km=(i,j,k1)
            s += Float64(rho[Kp...])*Float64(ux[Kp...])*Float64(uz[Kp...]) * dA
            s -= Float64(rho[Km...])*Float64(ux[Km...])*Float64(uz[Km...]) * dA
        end
    end
    return -s          # D = −∮ … (positive = drag for +x flow)
end

function pond_body_force(d::PonDFVDriver)
    # host-side integral of ρ·χ·u over the body, divided by η (steady penalization force
    # ≈ ∫ ρ χ u / η  for the Brinkman sink); robust steady-state drag proxy.
    rho = Array(d.rho); ux = Array(d.ux); uy = Array(d.uy); uz = Array(d.uz); chi = Array(d.chi)
    Fx = 0.0; Fy = 0.0; Fz = 0.0
    @inbounds for I in eachindex(chi)
        c = chi[I]
        if c > 0.0f0
            w = rho[I]*c/d.eta
            Fx += w*ux[I]; Fy += w*uy[I]; Fz += w*uz[I]
        end
    end
    return Float64(Fx), Float64(Fy), Float64(Fz)
end
