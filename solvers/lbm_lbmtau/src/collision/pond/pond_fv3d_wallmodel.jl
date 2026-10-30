# ==============================================================================
# Particles-on-Demand — equilibrium WALL MODEL (WMLES)  [the high-Re enabler]
# ==============================================================================
# The piece that lets PonD reach high Re like the production D3Q27/D3Q39: a wall model supplies the
# near-wall shear stress τ_w = ρ u_τ² from the LAW OF THE WALL using the first-fluid-cell velocity,
# so the turbulent boundary layer does NOT need to be resolved (resolving a Re~10⁶ BL is infeasible).
# Replaces the no-slip volume penalization on the SURFACE with a SLIP wall + modeled tangential stress:
#   • no penetration:  u ← u − (u·n) n   (impermeable body)
#   • modeled friction: u_∥ decelerated by τ_w/(ρ·Δ) per step, u_τ from the log-law (Spalding-seeded
#     Newton; viscous-sublayer fallback). Van Driest compressible factor √(T_w/T) ⇒ here T≈1 isothermal,
#     so vd≈1 (the scaffold is in place for a thermal extension).
# The DEEP solid interior stays fully penalized (rest, impermeable). Mirrors the production solver's
# Van Driest/Larsson equilibrium wall model (kernel_entropic_d3q39.jl). cs_wall=0 / no near-wall band
# ⇒ inactive. See memory re-ma-consistency-les.
# ==============================================================================

using KernelAbstractions

const WM_KAPPA = 0.41f0     # von Kármán
const WM_B     = 5.0f0      # log-law intercept

# precompute per-cell signed wall distance y (= r−R; <0 solid, >0 fluid) + outward unit normal, for a
# sphere centre (cx,cy,cz) radius R. Returns (y, nx, ny, nz) host arrays. (General STL: use wall_dist.)
function pond_sphere_wallinfo(NX,NY,NZ, cx,cy,cz, R, dx)
    y=zeros(Float32,NX,NY,NZ); nx=zeros(Float32,NX,NY,NZ); ny=zeros(Float32,NX,NY,NZ); nz=zeros(Float32,NX,NY,NZ)
    for k in 1:NZ, j in 1:NY, i in 1:NX
        px=(i-0.5f0)*dx-cx; py=(j-0.5f0)*dx-cy; pz=(k-0.5f0)*dx-cz
        r=sqrt(px*px+py*py+pz*pz)+1f-6
        y[i,j,k]=r-R; nx[i,j,k]=px/r; ny[i,j,k]=py/r; nz[i,j,k]=pz/r
    end
    return y,nx,ny,nz
end

# solve u_∥ = u_τ[(1/κ)ln(y u_τ/ν)+B] for u_τ (Newton, fixed iters; sublayer fallback). GPU-safe.
@inline function _u_tau(s, y, nu)
    s<1f-8 && return 0f0
    ut = sqrt(nu*s/max(y,1f-4))            # viscous-sublayer seed: τ_w=μ s/y ⇒ u_τ=√(ν s/y)
    yp = y*ut/nu
    if yp > 11.0f0                          # log layer ⇒ refine by Newton on the log-law
        ut = s/15f0                         # rough log seed
        @inbounds for _ in 1:8
            ut = ut<1f-8 ? 1f-8 : ut
            lp = (1f0/WM_KAPPA)*log(max(y*ut/nu,1.0001f0)) + WM_B
            f  = s - ut*lp
            df = -lp - 1f0/WM_KAPPA
            ut = ut - f/df
            ut = ut<1f-6 ? 1f-6 : ut
        end
    end
    return ut
end

# wall-model kernel: SOLID (y<−wband) fully penalized to rest; NEAR-WALL fluid (0<y≤y_wm) gets
# no-penetration + the modeled tangential stress; (−wband≤y≤0) surface kept at rest (impermeable).
@kernel function pond_wallmodel_kernel!(rho, ux, uy, uz, @Const(yw), @Const(nxw), @Const(nyw), @Const(nzw),
                                        nu::Float32, dt::Float32, dx::Float32, y_wm::Float32, invscale::Float32)
    i,j,k=@index(Global,NTuple)
    @inbounds begin
        y=yw[i,j,k]
        if y <= 0f0
            f = 1f0/(1f0+invscale)         # solid / surface ⇒ penalize to rest (impermeable)
            ux[i,j,k]*=f; uy[i,j,k]*=f; uz[i,j,k]*=f
        elseif y <= y_wm
            nx=nxw[i,j,k]; ny=nyw[i,j,k]; nz=nzw[i,j,k]
            ax=ux[i,j,k]; ay=uy[i,j,k]; az=uz[i,j,k]
            un = ax*nx+ay*ny+az*nz          # remove the wall-NORMAL component (no penetration)
            ax-=un*nx; ay-=un*ny; az-=un*nz
            s=sqrt(ax*ax+ay*ay+az*az)
            if s > 1f-8
                ut=_u_tau(s, y, nu)
                tauw_over_rho = ut*ut                       # τ_w/ρ = u_τ²
                dec = tauw_over_rho*dt/(dx*s)               # Δu_∥ = −τ_w/(ρ·Δ)·dt along û_∥
                fac = 1f0 - dec; fac = fac<0f0 ? 0f0 : fac  # decelerate, never reverse
                ax*=fac; ay*=fac; az*=fac
            end
            ux[i,j,k]=ax; uy[i,j,k]=ay; uz[i,j,k]=az
        end
    end
end

# host: apply the wall model (call instead of plain penalization when the WM fields are present).
function pond_apply_wallmodel!(backend, rho, ux, uy, uz, yw, nxw, nyw, nzw, nu, dt, dx, y_wm, eta)
    nd=size(ux)
    pond_wallmodel_kernel!(backend)(rho, ux, uy, uz, yw, nxw, nyw, nzw,
                                    Float32(nu), Float32(dt), Float32(dx), Float32(y_wm), Float32(dt/eta); ndrange=nd)
    KernelAbstractions.synchronize(backend); return nothing
end
