# ==============================================================================
# Particles-on-Demand — PURE 2D MULTILEVEL (2-level AMR) conservative kinetic-FV  [GPU]
# ==============================================================================
# Wires a 2:1 multilevel domain into the pure-2D solver: a 3-grid stripe layout
#   coarse-left (dx=2) | FINE-middle (dx=1, holds the body+bow shock+near-wake) | coarse-right (dx=2)
# concentrates resolution where it matters and leaves the far field coarse. The two PLANAR 2:1
# interfaces use the validated SHARED-FLUX MORTAR (test_pond_amr2d_body.jl): the coarse face flux =
# average of the 2 fine face fluxes, and that SAME flux is the fine sub-grid's edge flux ⇒ the shared
# face is identical on both sides ⇒ EXACTLY conservative across the level jump (Berger-Colella refluxing).
#
# This kernel is the per-sub-grid step with BOUNDARY FLUXES INJECTED at the x-edges (i=1, i=NX): the
# left/right edge faces use the supplied mortar (or inlet/outlet) flux instead of a clamped interior
# flux; the y-edges are clamped (freestream overwrite applied post-step by the driver). The mortar
# fluxes themselves are computed on the host from the two sub-grids' interface columns (small, 1-D).
# Reuses _fvflux2/_clampi from pond_fv_gpu.jl. Same-Δt 2-level (no sub-cycling — efficiency follow-up).
# ==============================================================================

using KernelAbstractions

@kernel function pond_fv2d_amr_kernel!(rho_o, ux_o, uy_o, @Const(rho), @Const(ux), @Const(uy),
        @Const(lFr),@Const(lFmx),@Const(lFmy), @Const(rFr),@Const(rFmx),@Const(rFmy),
        @Const(cx_arr),@Const(cy_arr),@Const(cz_arr),@Const(w_arr), cs2::Float32, a::Float32, NX::Int32, NY::Int32)
    i, j = @index(Global, NTuple)
    nx = Int(NX); ny = Int(NY)
    @inbounds begin
        iL=_clampi(i-1,nx); iR=_clampi(i+1,nx); iRR=_clampi(i+2,nx); iLL=_clampi(i-2,nx)
        jD=_clampi(j-1,ny); jU=_clampi(j+1,ny); jUU=_clampi(j+2,ny); jDD=_clampi(j-2,ny)
        # left x-face: injected mortar/inlet flux at the sub-grid edge, else interior flux
        if i==1
            FrL=lFr[j]; FmxL=lFmx[j]; FmyL=lFmy[j]
        else
            FrL,FmxL,FmyL=_fvflux2(rho[iL,j],ux[iL,j],uy[iL,j], rho[i,j],ux[i,j],uy[i,j],
                                   rho[iLL,j],ux[iLL,j],uy[iLL,j], rho[iR,j],ux[iR,j],uy[iR,j], 1, cx_arr,cy_arr,cz_arr,w_arr,cs2)
        end
        # right x-face
        if i==nx
            FrR=rFr[j]; FmxR=rFmx[j]; FmyR=rFmy[j]
        else
            FrR,FmxR,FmyR=_fvflux2(rho[i,j],ux[i,j],uy[i,j], rho[iR,j],ux[iR,j],uy[iR,j],
                                   rho[iL,j],ux[iL,j],uy[iL,j], rho[iRR,j],ux[iRR,j],uy[iRR,j], 1, cx_arr,cy_arr,cz_arr,w_arr,cs2)
        end
        # y-faces (clamped; freestream overwrite applied post-step)
        FrT,FmxT,FmyT=_fvflux2(rho[i,j],ux[i,j],uy[i,j], rho[i,jU],ux[i,jU],uy[i,jU],
                               rho[i,jD],ux[i,jD],uy[i,jD], rho[i,jUU],ux[i,jUU],uy[i,jUU], 2, cx_arr,cy_arr,cz_arr,w_arr,cs2)
        FrB,FmxB,FmyB=_fvflux2(rho[i,jD],ux[i,jD],uy[i,jD], rho[i,j],ux[i,j],uy[i,j],
                               rho[i,jDD],ux[i,jDD],uy[i,jDD], rho[i,jU],ux[i,jU],uy[i,jU], 2, cx_arr,cy_arr,cz_arr,w_arr,cs2)
        r  = rho[i,j]         - a*((FrR-FrL)  + (FrT-FrB))
        mx = rho[i,j]*ux[i,j] - a*((FmxR-FmxL)+ (FmxT-FmxB))
        my = rho[i,j]*uy[i,j] - a*((FmyR-FmyL)+ (FmyT-FmyB))
        r = r < 1.0f-3 ? 1.0f-3 : r
        ux2=mx/r; uy2=my/r; sp=sqrt(ux2*ux2+uy2*uy2)
        if sp>3.0f0; f=3.0f0/sp; ux2*=f; uy2*=f; end
        rho_o[i,j]=r; ux_o[i,j]=ux2; uy_o[i,j]=uy2
    end
end

# host step wrapper: launch the AMR kernel for one sub-grid (a=dt/dx for that level).
function amr_step!(backend, rho_o,ux_o,uy_o, rho,ux,uy, lFr,lFmx,lFmy, rFr,rFmx,rFmy,
                   cx,cy,cz,w, cs2::Float32, dt::Float32, dx::Float32, NX, NY)
    pond_fv2d_amr_kernel!(backend)(rho_o,ux_o,uy_o, rho,ux,uy, lFr,lFmx,lFmy, rFr,rFmx,rFmy,
        cx,cy,cz,w, cs2, dt/dx, Int32(NX),Int32(NY); ndrange=(Int(NX),Int(NY)))
    KernelAbstractions.synchronize(backend); return nothing
end
