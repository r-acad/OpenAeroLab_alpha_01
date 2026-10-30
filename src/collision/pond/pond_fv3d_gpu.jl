# ==============================================================================
# Particles-on-Demand — 3D conservative kinetic-FV GPU kernels   [path A, P4→3D]
# ==============================================================================
# 3D extension of the validated 2D conservative kinetic-FV scheme. Conserved lab
# moments (ρ, ρu, ρv, ρw) updated by SHARED face fluxes built from co-moving
# Grad-reconstructed, upwind, MUSCL-limited populations. Each face is computed
# deterministically from the two adjacent cells' OLD state ⇒ the shared face is
# identical on both sides ⇒ exactly conservative, no atomics, no race.
#
# REGISTER PRESSURE / OCCUPANCY: rather than one monolithic kernel computing all 6
# face fluxes inline (≈6×`_fvflux3` live ⇒ heavy register spill on consumer GPUs,
# the same failure mode CLAUDE.md notes for the 93-param entropic kernel), the update
# is SPLIT into three per-axis kernels, each computing only its 2 faces (≈2×`_fvflux3`
# live) and accumulating the flux divergence into the conserved-moment output:
#   • pond_flux_x_kernel!  — initialise output = old conserved moments − a·div_x(flux)
#   • pond_flux_y_kernel!  — output −= a·div_y(flux)
#   • pond_flux_z_kernel!  — output −= a·div_z(flux), then floor ρ and recover u = m/ρ
# All three read the SAME old @Const(rho,ux,uy,uz) for flux evaluation (operator-
# UNsplit: every flux uses the old state), so the result is identical to the fused
# kernel to round-off while keeping each kernel small (~3× fewer live flux registers).
# ==============================================================================

using KernelAbstractions

@inline _clampi3(i, n) = i < 1 ? 1 : (i > n ? n : i)
@inline _mm3(a, b) = (a*b <= 0.0f0) ? 0.0f0 : (abs(a) < abs(b) ? a : b)

# equilibrium population k at lab (ρ,ux,uy,uz) in face frame (ufx,ufy,ufz) — closed-form
# 2nd-order Grad: f = w[ρ + (j·c)/cs² + (1/2cs⁴)Σ s_αβ(c_α c_β − cs²δ_αβ)], s = ρ p⊗p, j = ρ p.
@inline function _feq3_gpu(rho, ux, uy, uz, ufx, ufy, ufz, cxk, cyk, czk, wk, cs2)
    px = ux - ufx; py = uy - ufy; pz = uz - ufz
    inv_cs2 = 1.0f0/cs2; inv_2cs4 = 0.5f0/(cs2*cs2)
    j1 = rho*px; j2 = rho*py; j3 = rho*pz
    s5 = rho*px*px; s6 = rho*py*py; s7 = rho*pz*pz
    s8 = rho*px*py; s9 = rho*px*pz; s10 = rho*py*pz
    h1 = j1*cxk + j2*cyk + j3*czk
    h2 = s5*(cxk*cxk - cs2) + s6*(cyk*cyk - cs2) + s7*(czk*czk - cs2) +
         2.0f0*(s8*cxk*cyk + s9*cxk*czk + s10*cyk*czk)
    return max(wk*(rho + inv_cs2*h1 + inv_2cs4*h2), 1.0f-12)
end

# face flux (Fρ,Fmx,Fmy,Fmz) along axis ax (1=x,2=y,3=z) from L/R states + slope neighbours.
@inline function _fvflux3(rhoL,uxL,uyL,uzL, rhoR,uxR,uyR,uzR,
                          rhoLL,uxLL,uyLL,uzLL, rhoRR,uxRR,uyRR,uzRR,
                          ax, cx_arr, cy_arr, cz_arr, w_arr, cs2)
    ufx = 0.5f0*(uxL+uxR); ufy = 0.5f0*(uyL+uyR); ufz = 0.5f0*(uzL+uzR)
    rhoLf = max(rhoL + 0.5f0*_mm3(rhoL-rhoLL, rhoR-rhoL), 1.0f-4)
    rhoRf = max(rhoR - 0.5f0*_mm3(rhoR-rhoL, rhoRR-rhoR), 1.0f-4)
    uxLf = uxL + 0.5f0*_mm3(uxL-uxLL, uxR-uxL); uxRf = uxR - 0.5f0*_mm3(uxR-uxL, uxRR-uxR)
    uyLf = uyL + 0.5f0*_mm3(uyL-uyLL, uyR-uyL); uyRf = uyR - 0.5f0*_mm3(uyR-uyL, uyRR-uyR)
    uzLf = uzL + 0.5f0*_mm3(uzL-uzLL, uzR-uzL); uzRf = uzR - 0.5f0*_mm3(uzR-uzL, uzRR-uzR)
    Frho = 0.0f0; Fmx = 0.0f0; Fmy = 0.0f0; Fmz = 0.0f0
    @inbounds for k in 1:39
        cxk = cx_arr[k]; cyk = cy_arr[k]; czk = cz_arr[k]; wk = w_arr[k]
        vx = cxk + ufx; vy = cyk + ufy; vz = czk + ufz
        vn = ax == 1 ? vx : (ax == 2 ? vy : vz)
        fk = vn >= 0.0f0 ? _feq3_gpu(rhoLf,uxLf,uyLf,uzLf,ufx,ufy,ufz,cxk,cyk,czk,wk,cs2) :
                           _feq3_gpu(rhoRf,uxRf,uyRf,uzRf,ufx,ufy,ufz,cxk,cyk,czk,wk,cs2)
        Frho += vn*fk; Fmx += vn*vx*fk; Fmy += vn*vy*fk; Fmz += vn*vz*fk
    end
    return Frho, Fmx, Fmy, Fmz
end

# ---- axis 1 (x): initialise the conserved-moment output with old moments − a·div_x ----
@kernel function pond_flux_x_kernel!(rho_o, mx_o, my_o, mz_o,
                                     @Const(rho), @Const(ux), @Const(uy), @Const(uz),
                                     @Const(cx_arr), @Const(cy_arr), @Const(cz_arr), @Const(w_arr),
                                     cs2::Float32, a::Float32, NX::Int32, NY::Int32, NZ::Int32)
    i, j, k = @index(Global, NTuple)
    nx = Int(NX)
    @inbounds begin
        iL  = _clampi3(i-1,nx); iR  = _clampi3(i+1,nx); iRR = _clampi3(i+2,nx); iLL = _clampi3(i-2,nx)
        FrR,FmxR,FmyR,FmzR = _fvflux3(rho[i,j,k],ux[i,j,k],uy[i,j,k],uz[i,j,k], rho[iR,j,k],ux[iR,j,k],uy[iR,j,k],uz[iR,j,k],
                                      rho[iL,j,k],ux[iL,j,k],uy[iL,j,k],uz[iL,j,k], rho[iRR,j,k],ux[iRR,j,k],uy[iRR,j,k],uz[iRR,j,k],
                                      1, cx_arr,cy_arr,cz_arr,w_arr,cs2)
        FrL,FmxL,FmyL,FmzL = _fvflux3(rho[iL,j,k],ux[iL,j,k],uy[iL,j,k],uz[iL,j,k], rho[i,j,k],ux[i,j,k],uy[i,j,k],uz[i,j,k],
                                      rho[iLL,j,k],ux[iLL,j,k],uy[iLL,j,k],uz[iLL,j,k], rho[iR,j,k],ux[iR,j,k],uy[iR,j,k],uz[iR,j,k],
                                      1, cx_arr,cy_arr,cz_arr,w_arr,cs2)
        rho_o[i,j,k] = rho[i,j,k]           - a*(FrR-FrL)
        mx_o[i,j,k]  = rho[i,j,k]*ux[i,j,k] - a*(FmxR-FmxL)
        my_o[i,j,k]  = rho[i,j,k]*uy[i,j,k] - a*(FmyR-FmyL)
        mz_o[i,j,k]  = rho[i,j,k]*uz[i,j,k] - a*(FmzR-FmzL)
    end
end

# ---- axis 2 (y): accumulate −a·div_y into the conserved-moment output ----
@kernel function pond_flux_y_kernel!(rho_o, mx_o, my_o, mz_o,
                                     @Const(rho), @Const(ux), @Const(uy), @Const(uz),
                                     @Const(cx_arr), @Const(cy_arr), @Const(cz_arr), @Const(w_arr),
                                     cs2::Float32, a::Float32, NX::Int32, NY::Int32, NZ::Int32)
    i, j, k = @index(Global, NTuple)
    ny = Int(NY)
    @inbounds begin
        jD  = _clampi3(j-1,ny); jU  = _clampi3(j+1,ny); jUU = _clampi3(j+2,ny); jDD = _clampi3(j-2,ny)
        FrT,FmxT,FmyT,FmzT = _fvflux3(rho[i,j,k],ux[i,j,k],uy[i,j,k],uz[i,j,k], rho[i,jU,k],ux[i,jU,k],uy[i,jU,k],uz[i,jU,k],
                                      rho[i,jD,k],ux[i,jD,k],uy[i,jD,k],uz[i,jD,k], rho[i,jUU,k],ux[i,jUU,k],uy[i,jUU,k],uz[i,jUU,k],
                                      2, cx_arr,cy_arr,cz_arr,w_arr,cs2)
        FrB,FmxB,FmyB,FmzB = _fvflux3(rho[i,jD,k],ux[i,jD,k],uy[i,jD,k],uz[i,jD,k], rho[i,j,k],ux[i,j,k],uy[i,j,k],uz[i,j,k],
                                      rho[i,jDD,k],ux[i,jDD,k],uy[i,jDD,k],uz[i,jDD,k], rho[i,jU,k],ux[i,jU,k],uy[i,jU,k],uz[i,jU,k],
                                      2, cx_arr,cy_arr,cz_arr,w_arr,cs2)
        rho_o[i,j,k] -= a*(FrT-FrB)
        mx_o[i,j,k]  -= a*(FmxT-FmxB)
        my_o[i,j,k]  -= a*(FmyT-FmyB)
        mz_o[i,j,k]  -= a*(FmzT-FmzB)
    end
end

# ---- axis 3 (z): accumulate −a·div_z, then floor ρ and recover velocity u = m/ρ ----
@kernel function pond_flux_z_kernel!(rho_o, mx_o, my_o, mz_o,
                                     @Const(rho), @Const(ux), @Const(uy), @Const(uz),
                                     @Const(cx_arr), @Const(cy_arr), @Const(cz_arr), @Const(w_arr),
                                     cs2::Float32, a::Float32, NX::Int32, NY::Int32, NZ::Int32)
    i, j, k = @index(Global, NTuple)
    nz = Int(NZ)
    @inbounds begin
        kB  = _clampi3(k-1,nz); kF  = _clampi3(k+1,nz); kFF = _clampi3(k+2,nz); kBB = _clampi3(k-2,nz)
        FrF,FmxF,FmyF,FmzF = _fvflux3(rho[i,j,k],ux[i,j,k],uy[i,j,k],uz[i,j,k], rho[i,j,kF],ux[i,j,kF],uy[i,j,kF],uz[i,j,kF],
                                      rho[i,j,kB],ux[i,j,kB],uy[i,j,kB],uz[i,j,kB], rho[i,j,kFF],ux[i,j,kFF],uy[i,j,kFF],uz[i,j,kFF],
                                      3, cx_arr,cy_arr,cz_arr,w_arr,cs2)
        FrK,FmxK,FmyK,FmzK = _fvflux3(rho[i,j,kB],ux[i,j,kB],uy[i,j,kB],uz[i,j,kB], rho[i,j,k],ux[i,j,k],uy[i,j,k],uz[i,j,k],
                                      rho[i,j,kBB],ux[i,j,kBB],uy[i,j,kBB],uz[i,j,kBB], rho[i,j,kF],ux[i,j,kF],uy[i,j,kF],uz[i,j,kF],
                                      3, cx_arr,cy_arr,cz_arr,w_arr,cs2)
        r  = rho_o[i,j,k] - a*(FrF-FrK)
        mx = mx_o[i,j,k]  - a*(FmxF-FmxK)
        my = my_o[i,j,k]  - a*(FmyF-FmyK)
        mz = mz_o[i,j,k]  - a*(FmzF-FmzK)
        r = r < 1.0f-4 ? 1.0f-4 : r
        rho_o[i,j,k] = r; mx_o[i,j,k] = mx/r; my_o[i,j,k] = my/r; mz_o[i,j,k] = mz/r
    end
end

# ------------------------------------------------------------------------------
# Host stepping helper: one conservative kinetic-FV time step. Launches the three
# per-axis kernels in sequence on `backend`, writing the new (ρ,u) into the `_o`
# arrays. The `_o` arrays must be DISTINCT from the inputs (the flux kernels read
# the old state); the caller ping-pongs. Synchronizes after each launch.
# ------------------------------------------------------------------------------
function pond_fv3d_step!(backend,
                         rho_o, ux_o, uy_o, uz_o,
                         rho, ux, uy, uz,
                         cx_arr, cy_arr, cz_arr, w_arr,
                         cs2::Float32, dt::Float32, dx::Float32,
                         NX::Integer, NY::Integer, NZ::Integer)
    a = dt/dx
    nx = Int32(NX); ny = Int32(NY); nz = Int32(NZ); nd = (Int(NX), Int(NY), Int(NZ))
    kx = pond_flux_x_kernel!(backend); ky = pond_flux_y_kernel!(backend); kz = pond_flux_z_kernel!(backend)
    kx(rho_o, ux_o, uy_o, uz_o, rho, ux, uy, uz, cx_arr, cy_arr, cz_arr, w_arr, cs2, a, nx, ny, nz; ndrange=nd)
    KernelAbstractions.synchronize(backend)
    ky(rho_o, ux_o, uy_o, uz_o, rho, ux, uy, uz, cx_arr, cy_arr, cz_arr, w_arr, cs2, a, nx, ny, nz; ndrange=nd)
    KernelAbstractions.synchronize(backend)
    kz(rho_o, ux_o, uy_o, uz_o, rho, ux, uy, uz, cx_arr, cy_arr, cz_arr, w_arr, cs2, a, nx, ny, nz; ndrange=nd)
    KernelAbstractions.synchronize(backend)
    return nothing
end
