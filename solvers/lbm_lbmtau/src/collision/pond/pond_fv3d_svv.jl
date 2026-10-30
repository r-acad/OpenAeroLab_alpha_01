# ==============================================================================
# Particles-on-Demand — Spectral Vanishing Viscosity (SVV) hyperviscosity  [ripple killer]
# ==============================================================================
# Eliminates the grid-scale (2Δ) dispersive ripples the low-dissipation DUGKS sustains, WITHOUT
# touching the resolved physics. A bi-Laplacian (∇⁴) hyperviscosity is k⁴-SELECTIVE: its eigenvalue
# λ(k)=[2(1-cos k)]² is 16 at Nyquist (k=π) but only 0.34 at an 8-cell wavelength (k=π/4) — a 47×
# selectivity. So a small ε kills the unphysical 2-3Δ ripples while the well-resolved scales feel a
# negligible effective viscosity (the production solver's SVV uses the same operator + ε~0.002).
# Applied to the MOMENTUM ρu (m_new = m − ε·∇⁴m, u = m_new/ρ): ∇⁴ is a flux divergence ⇒ TOTAL
# MOMENTUM is conserved, and ρ is untouched ⇒ MASS conserved. Physical accuracy preserved. ε=0 ⇒ off.
# ==============================================================================

using KernelAbstractions

@inline _cS(i,n) = i<1 ? 1 : (i>n ? n : i)
# 1D 6th-derivative (∇⁶) stencil [1,-6,15,-20,15,-6,1] — eigenvalue [2(1-cos k)]³. MUCH more selective
# than ∇⁴: 320× between Nyquist (k=π) and an 8-cell wavelength (k=π/4) ⇒ strong 2Δ-ripple kill with
# negligible effect on the resolved scales. Sign: subtract −ε·(−∇⁶) so it DAMPS the high modes.
@inline _hyp6(a3m,a2m,a1m,a0,a1p,a2p,a3p) = -(a3m - 6.0f0*a2m + 15.0f0*a1m - 20.0f0*a0 + 15.0f0*a1p - 6.0f0*a2p + a3p)
# per-axis ∇⁶ of the momentum component (field f sampled with ρ at ±1,±2,±3); returns the damping term
@inline _ax6(r3m,f3m, r2m,f2m, r1m,f1m, m0, r1p,f1p, r2p,f2p, r3p,f3p) =
    _hyp6(r3m*f3m, r2m*f2m, r1m*f1m, m0, r1p*f1p, r2p*f2p, r3p*f3p)

@kernel function pond_svv_kernel!(ux, uy, uz, @Const(rho), eps::Float32, NX::Int32, NY::Int32, NZ::Int32)
    i,j,k = @index(Global, NTuple)
    nx=Int(NX); ny=Int(NY); nz=Int(NZ)
    @inbounds begin
        i1m=_cS(i-1,nx);i1p=_cS(i+1,nx);i2m=_cS(i-2,nx);i2p=_cS(i+2,nx);i3m=_cS(i-3,nx);i3p=_cS(i+3,nx)
        j1m=_cS(j-1,ny);j1p=_cS(j+1,ny);j2m=_cS(j-2,ny);j2p=_cS(j+2,ny);j3m=_cS(j-3,ny);j3p=_cS(j+3,ny)
        k1m=_cS(k-1,nz);k1p=_cS(k+1,nz);k2m=_cS(k-2,nz);k2p=_cS(k+2,nz);k3m=_cS(k-3,nz);k3p=_cS(k+3,nz)
        r0=rho[i,j,k]; invr=1.0f0/r0
        for comp in 1:3
            u = comp==1 ? ux : (comp==2 ? uy : uz)
            m0=r0*u[i,j,k]
            b = _ax6(rho[i3m,j,k],u[i3m,j,k], rho[i2m,j,k],u[i2m,j,k], rho[i1m,j,k],u[i1m,j,k], m0, rho[i1p,j,k],u[i1p,j,k], rho[i2p,j,k],u[i2p,j,k], rho[i3p,j,k],u[i3p,j,k]) +
                _ax6(rho[i,j3m,k],u[i,j3m,k], rho[i,j2m,k],u[i,j2m,k], rho[i,j1m,k],u[i,j1m,k], m0, rho[i,j1p,k],u[i,j1p,k], rho[i,j2p,k],u[i,j2p,k], rho[i,j3p,k],u[i,j3p,k]) +
                _ax6(rho[i,j,k3m],u[i,j,k3m], rho[i,j,k2m],u[i,j,k2m], rho[i,j,k1m],u[i,j,k1m], m0, rho[i,j,k1p],u[i,j,k1p], rho[i,j,k2p],u[i,j,k2p], rho[i,j,k3p],u[i,j,k3p])
            u[i,j,k] = (m0 - eps*b)*invr
        end
    end
end

function pond_apply_svv!(backend, ux, uy, uz, rho, eps, NX, NY, NZ)
    pond_svv_kernel!(backend)(ux,uy,uz, rho, Float32(eps), Int32(NX),Int32(NY),Int32(NZ); ndrange=(Int(NX),Int(NY),Int(NZ)))
    KernelAbstractions.synchronize(backend); return nothing
end
