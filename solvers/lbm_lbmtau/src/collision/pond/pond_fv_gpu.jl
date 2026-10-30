# ==============================================================================
# Particles-on-Demand — 2D conservative kinetic-FV GPU kernel   [path A, P4 PoC]
# ==============================================================================
# Backend-agnostic (KernelAbstractions) port of the validated CPU conservative
# kinetic-FV scheme (tools/test_pond_fv2d.jl): conserved lab moments (ρ, ρu, ρv)
# updated by SHARED face fluxes built from co-moving Grad-reconstructed, upwind,
# MUSCL-limited populations. One thread per cell; each cell recomputes its 4 face
# fluxes from neighbours (deterministic ⇒ the shared face is identical on both
# sides ⇒ exactly conservative, no atomics, no race). Proves the FV scheme ports
# to the GPU before wiring into the block-structured solver.
# ==============================================================================

using KernelAbstractions

@inline _clampi(i, n) = i < 1 ? 1 : (i > n ? n : i)
@inline _mm(a, b) = (a*b <= 0.0f0) ? 0.0f0 : (abs(a) < abs(b) ? a : b)

# equilibrium population k at lab (ρ,ux,uy) in face frame (ufx,ufy) — closed-form Grad.
@inline function _feq2_gpu(rho, ux, uy, ufx, ufy, cxk, cyk, czk, wk, cs2)
    px = ux - ufx; py = uy - ufy
    # 2nd-order Grad polynomial: f = w[ρ + (j·c)/cs² + (1/2cs⁴)Σ s(cc−cs²)], s = ρ p⊗p, j = ρ p
    inv_cs2 = 1.0f0/cs2; inv_2cs4 = 0.5f0/(cs2*cs2)
    j1 = rho*px; j2 = rho*py
    s5 = rho*px*px; s6 = rho*py*py; s8 = rho*px*py
    h1 = j1*cxk + j2*cyk
    h2 = s5*(cxk*cxk - cs2) + s6*(cyk*cyk - cs2) + 2.0f0*s8*cxk*cyk
    return max(wk*(rho + inv_cs2*h1 + inv_2cs4*h2), 1.0f-12)
end

# face flux (Fρ,Fmx,Fmy) along axis ax (1=x,2=y) from L/R states + their slope neighbours.
@inline function _fvflux2(rhoL,uxL,uyL, rhoR,uxR,uyR, rhoLL,uxLL,uyLL, rhoRR,uxRR,uyRR,
                          ax, cx_arr, cy_arr, cz_arr, w_arr, cs2)
    ufx = 0.5f0*(uxL+uxR); ufy = 0.5f0*(uyL+uyR)
    rhoLf = max(rhoL + 0.5f0*_mm(rhoL-rhoLL, rhoR-rhoL), 1.0f-4)
    rhoRf = max(rhoR - 0.5f0*_mm(rhoR-rhoL, rhoRR-rhoR), 1.0f-4)
    uxLf = uxL + 0.5f0*_mm(uxL-uxLL, uxR-uxL); uxRf = uxR - 0.5f0*_mm(uxR-uxL, uxRR-uxR)
    uyLf = uyL + 0.5f0*_mm(uyL-uyLL, uyR-uyL); uyRf = uyR - 0.5f0*_mm(uyR-uyL, uyRR-uyR)
    Frho = 0.0f0; Fmx = 0.0f0; Fmy = 0.0f0
    @inbounds for k in 1:39
        cxk = cx_arr[k]; cyk = cy_arr[k]; czk = cz_arr[k]; wk = w_arr[k]
        vx = cxk + ufx; vy = cyk + ufy
        vn = ax == 1 ? vx : vy
        fk = vn >= 0.0f0 ? _feq2_gpu(rhoLf,uxLf,uyLf,ufx,ufy,cxk,cyk,czk,wk,cs2) :
                           _feq2_gpu(rhoRf,uxRf,uyRf,ufx,ufy,cxk,cyk,czk,wk,cs2)
        Frho += vn*fk; Fmx += vn*vx*fk; Fmy += vn*vy*fk
    end
    return Frho, Fmx, Fmy
end

@kernel function pond_fv2d_kernel!(rho_o, mx_o, my_o, @Const(rho), @Const(ux), @Const(uy),
                                   @Const(cx_arr), @Const(cy_arr), @Const(cz_arr), @Const(w_arr),
                                   cs2::Float32, dt::Float32, dx::Float32, N::Int32)
    i, j = @index(Global, NTuple)
    nn = Int(N)
    @inbounds begin
        # neighbour indices (clamped, transmissive boundary)
        iL  = _clampi(i-1, nn); iR  = _clampi(i+1, nn); iRR = _clampi(i+2, nn); iLL = _clampi(i-2, nn)
        jD  = _clampi(j-1, nn); jU  = _clampi(j+1, nn); jUU = _clampi(j+2, nn); jDD = _clampi(j-2, nn)
        iC = i
        # --- x faces: right (i|i+1) and left (i-1|i) ---
        FrR,FmxR,FmyR = _fvflux2(rho[iC,j],ux[iC,j],uy[iC,j], rho[iR,j],ux[iR,j],uy[iR,j],
                                 rho[iL,j],ux[iL,j],uy[iL,j], rho[iRR,j],ux[iRR,j],uy[iRR,j],
                                 1, cx_arr,cy_arr,cz_arr,w_arr,cs2)
        FrL,FmxL,FmyL = _fvflux2(rho[iL,j],ux[iL,j],uy[iL,j], rho[iC,j],ux[iC,j],uy[iC,j],
                                 rho[iLL,j],ux[iLL,j],uy[iLL,j], rho[iR,j],ux[iR,j],uy[iR,j],
                                 1, cx_arr,cy_arr,cz_arr,w_arr,cs2)
        # --- y faces: top (j|j+1) and bottom (j-1|j) ---
        FrT,FmxT,FmyT = _fvflux2(rho[i,j],ux[i,j],uy[i,j], rho[i,jU],ux[i,jU],uy[i,jU],
                                 rho[i,jD],ux[i,jD],uy[i,jD], rho[i,jUU],ux[i,jUU],uy[i,jUU],
                                 2, cx_arr,cy_arr,cz_arr,w_arr,cs2)
        FrB,FmxB,FmyB = _fvflux2(rho[i,jD],ux[i,jD],uy[i,jD], rho[i,j],ux[i,j],uy[i,j],
                                 rho[i,jDD],ux[i,jDD],uy[i,jDD], rho[i,jU],ux[i,jU],uy[i,jU],
                                 2, cx_arr,cy_arr,cz_arr,w_arr,cs2)
        a = dt/dx
        r  = rho[i,j]          - a*((FrR-FrL)   + (FrT-FrB))
        mx = rho[i,j]*ux[i,j]  - a*((FmxR-FmxL) + (FmxT-FmxB))
        my = rho[i,j]*uy[i,j]  - a*((FmyR-FmyL) + (FmyT-FmyB))
        r = r < 1.0f-4 ? 1.0f-4 : r
        rho_o[i,j] = r; mx_o[i,j] = mx/r; my_o[i,j] = my/r
    end
end
