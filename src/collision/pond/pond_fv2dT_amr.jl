# ==============================================================================
# Particles-on-Demand — PURE 2D THERMAL MULTILEVEL (2-level AMR) conservative kinetic-FV  [GPU]
# ==============================================================================
# The THERMAL + √θ-GAUGE sibling of pond_fv2d_amr.jl: carries the 4th conserved variable ρE (real T,
# adiabatic shocks) AND the full options ladder of pond_fv2d_thermal.jl — √θ gauge (gauge), adiabatic
# kinetic wall (wall), DRP κ=1/3 reconstruction (drp), general-γ (drot), LES/MC (csmag) — across the
# 2:1 shared-flux mortar. It reuses the SAME `_fvflux2T` used by the single-grid thermal kernel (a plain
# @inline function, so the host mortar can call it too ⇒ no flux duplication), guaranteeing the interface
# flux is byte-consistent with the interior flux ⇒ the mortar stays exactly conservative in mass, momentum
# AND energy. Boundary (mortar/inlet/outlet) fluxes are injected at the x-edges (i=1, i=NX). Same-Δt
# (no sub-cycling — the temporal-refluxing follow-up).
# ==============================================================================

using KernelAbstractions

@kernel function pond_fv2dT_amr_kernel!(rho_o, ux_o, uy_o, T_o,
        @Const(rho), @Const(ux), @Const(uy), @Const(Temp), @Const(chi),
        @Const(lFr),@Const(lFmx),@Const(lFmy),@Const(lFE), @Const(rFr),@Const(rFmx),@Const(rFmy),@Const(rFE),
        @Const(cx_arr),@Const(cy_arr),@Const(cz_arr),@Const(w_arr),
        cs2::Float32, a::Float32, drot::Float32, csmag::Float32, gauge::Int32, wall::Int32, drp::Int32,
        NX::Int32, NY::Int32)
    i, j = @index(Global, NTuple)
    nx = Int(NX); ny = Int(NY); sa = Int32(0)
    eint = 0.5f0*(3.0f0+drot)*cs2
    # AUDIT FIX (2026-07-07): the calls below used the OLD 16-state _fvflux2T signature (pre-WENO) —
    # MethodError at first launch since the WENO/nu_coeff extension. Extended to the ±3 stencil with
    # weno=0 + nu_coeff=csmag² ⇒ behavior identical to the legacy MC/minmod path.
    wn = Int32(0); csq = csmag*csmag
    @inbounds begin
        iL=_clampi(i-1,nx); iR=_clampi(i+1,nx); iRR=_clampi(i+2,nx); iLL=_clampi(i-2,nx)
        jD=_clampi(j-1,ny); jU=_clampi(j+1,ny); jUU=_clampi(j+2,ny); jDD=_clampi(j-2,ny)
        iLLL=_clampi(i-3,nx); iRRR=_clampi(i+3,nx); jDDD=_clampi(j-3,ny); jUUU=_clampi(j+3,ny)
        # left x-face: injected mortar/inlet flux at the sub-grid edge, else interior thermal-gauge flux
        if i==1
            FrL=lFr[j]; FmxL=lFmx[j]; FmyL=lFmy[j]; FEL=lFE[j]
        else
            FrL,FmxL,FmyL,FEL = _fvflux2T(rho[iL,j],ux[iL,j],uy[iL,j],Temp[iL,j], rho[i,j],ux[i,j],uy[i,j],Temp[i,j],
                rho[iLL,j],ux[iLL,j],uy[iLL,j],Temp[iLL,j], rho[iR,j],ux[iR,j],uy[iR,j],Temp[iR,j],
                rho[iLLL,j],ux[iLLL,j],uy[iLLL,j],Temp[iLLL,j], rho[iRR,j],ux[iRR,j],uy[iRR,j],Temp[iRR,j], 1, cx_arr,cy_arr,cz_arr,w_arr,cs2,drot,csmag,gauge,sa,drp,wn,csq)
        end
        if i==nx
            FrR=rFr[j]; FmxR=rFmx[j]; FmyR=rFmy[j]; FER=rFE[j]
        else
            FrR,FmxR,FmyR,FER = _fvflux2T(rho[i,j],ux[i,j],uy[i,j],Temp[i,j], rho[iR,j],ux[iR,j],uy[iR,j],Temp[iR,j],
                rho[iL,j],ux[iL,j],uy[iL,j],Temp[iL,j], rho[iRR,j],ux[iRR,j],uy[iRR,j],Temp[iRR,j],
                rho[iLL,j],ux[iLL,j],uy[iLL,j],Temp[iLL,j], rho[iRRR,j],ux[iRRR,j],uy[iRRR,j],Temp[iRRR,j], 1, cx_arr,cy_arr,cz_arr,w_arr,cs2,drot,csmag,gauge,sa,drp,wn,csq)
        end
        FrT,FmxT,FmyT,FET = _fvflux2T(rho[i,j],ux[i,j],uy[i,j],Temp[i,j], rho[i,jU],ux[i,jU],uy[i,jU],Temp[i,jU],
            rho[i,jD],ux[i,jD],uy[i,jD],Temp[i,jD], rho[i,jUU],ux[i,jUU],uy[i,jUU],Temp[i,jUU],
            rho[i,jDD],ux[i,jDD],uy[i,jDD],Temp[i,jDD], rho[i,jUUU],ux[i,jUUU],uy[i,jUUU],Temp[i,jUUU], 2, cx_arr,cy_arr,cz_arr,w_arr,cs2,drot,csmag,gauge,sa,drp,wn,csq)
        FrB,FmxB,FmyB,FEB = _fvflux2T(rho[i,jD],ux[i,jD],uy[i,jD],Temp[i,jD], rho[i,j],ux[i,j],uy[i,j],Temp[i,j],
            rho[i,jDD],ux[i,jDD],uy[i,jDD],Temp[i,jDD], rho[i,jU],ux[i,jU],uy[i,jU],Temp[i,jU],
            rho[i,jDDD],ux[i,jDDD],uy[i,jDDD],Temp[i,jDDD], rho[i,jUU],ux[i,jUU],uy[i,jUU],Temp[i,jUU], 2, cx_arr,cy_arr,cz_arr,w_arr,cs2,drot,csmag,gauge,sa,drp,wn,csq)
        # adiabatic kinetic wall (fluid cells bordering solid): reflective face flux (0 mass, 0 energy, p_w momentum)
        if wall == 1 && chi[i,j] <= 0.5f0
            pw = rho[i,j]*cs2*Temp[i,j]
            if chi[iR,j]>0.5f0; FrR=0.0f0; FmxR=pw; FmyR=0.0f0; FER=0.0f0; end
            if chi[iL,j]>0.5f0; FrL=0.0f0; FmxL=pw; FmyL=0.0f0; FEL=0.0f0; end
            if chi[i,jU]>0.5f0; FrT=0.0f0; FmyT=pw; FmxT=0.0f0; FET=0.0f0; end
            if chi[i,jD]>0.5f0; FrB=0.0f0; FmyB=pw; FmxB=0.0f0; FEB=0.0f0; end
        end
        u0x=ux[i,j]; u0y=uy[i,j]; r0=rho[i,j]
        Ec = r0*(eint*Temp[i,j] + 0.5f0*(u0x*u0x+u0y*u0y))
        r  = r0          - a*((FrR-FrL)  + (FrT-FrB))
        mx = r0*u0x      - a*((FmxR-FmxL)+ (FmxT-FmxB))
        my = r0*u0y      - a*((FmyR-FmyL)+ (FmyT-FmyB))
        e  = Ec          - a*((FER-FEL)  + (FET-FEB))
        r = r < 1.0f-3 ? 1.0f-3 : r
        ux2=mx/r; uy2=my/r; sp=sqrt(ux2*ux2+uy2*uy2)
        if sp>3.5f0; f=3.5f0/sp; ux2*=f; uy2*=f; end
        T2 = (e/r - 0.5f0*(ux2*ux2+uy2*uy2))/eint; T2 = T2 < 1.0f-3 ? 1.0f-3 : T2
        rho_o[i,j]=r; ux_o[i,j]=ux2; uy_o[i,j]=uy2; T_o[i,j]=T2
    end
end

# host step wrapper for one thermal sub-grid (a=dt/dx). chi = body mask (marker-only for coarse grids).
function amr_stepT!(backend, rho_o,ux_o,uy_o,T_o, rho,ux,uy,Temp, chi,
                    lFr,lFmx,lFmy,lFE, rFr,rFmx,rFmy,rFE, cx,cy,cz,w,
                    cs2::Float32, dt::Float32, dx::Float32, drot::Float32, csmag::Float32,
                    gauge::Int32, wall::Int32, drp::Int32, NX, NY)
    pond_fv2dT_amr_kernel!(backend)(rho_o,ux_o,uy_o,T_o, rho,ux,uy,Temp, chi,
        lFr,lFmx,lFmy,lFE, rFr,rFmx,rFmy,rFE, cx,cy,cz,w, cs2, dt/dx, drot, csmag, gauge, wall, drp,
        Int32(NX),Int32(NY); ndrange=(Int(NX),Int(NY)))
    KernelAbstractions.synchronize(backend); return nothing
end
