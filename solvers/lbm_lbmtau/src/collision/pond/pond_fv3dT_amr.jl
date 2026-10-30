# ==============================================================================
# Particles-on-Demand — 3D THERMAL MULTILEVEL (2-level AMR) conservative kinetic-FV  [GPU]
# ==============================================================================
# The 3D sibling of pond_fv2dT_amr.jl: a 3-grid stripe in X — coarse-left (dx=2) | FINE-middle (dx=1,
# body+bow shock+near wake) | coarse-right (dx=2). The two 2:1 interfaces are YZ-PLANES; each coarse face
# covers a 2×2 block of fine faces (2:1 in BOTH y and z), so the coarse flux = mean of the FOUR fine face
# fluxes and that same flux is the fine edge flux ⇒ the shared face is identical on both sides ⇒ mass,
# momentum AND energy conserved EXACTLY across the level jump (Berger–Colella refluxing, 3D). The mortar
# calls the SAME `_fvflux3T` used in the interior (all features: √θ gauge, adiabatic wall, DRP, SGS,
# shock-aware), so the interface flux is byte-consistent with the interior. Boundary (mortar/inlet/outlet)
# fluxes are injected at the x-edges (i=1, i=NX) as YZ-plane 5-component arrays. Same-Δt (no sub-cycling).
# ==============================================================================

using KernelAbstractions

@kernel function pond_fv3dT_amr_kernel!(rho_o,ux_o,uy_o,uz_o,T_o,
        @Const(rho),@Const(ux),@Const(uy),@Const(uz),@Const(Temp),@Const(chi),
        @Const(lFr),@Const(lFmx),@Const(lFmy),@Const(lFmz),@Const(lFE),
        @Const(rFr),@Const(rFmx),@Const(rFmy),@Const(rFmz),@Const(rFE),
        @Const(cx_arr),@Const(cy_arr),@Const(cz_arr),@Const(w_arr),
        cs2::Float32, a::Float32, drot::Float32, csmag::Float32, gauge::Int32, wall::Int32, shock_aware::Int32, drp::Int32,
        NX::Int32, NY::Int32, NZ::Int32)
    i,j,l = @index(Global, NTuple)
    nx=Int(NX); ny=Int(NY); nz=Int(NZ); eint=0.5f0*(3.0f0+drot)*cs2
    @inbounds begin
        iL=_clampi(i-1,nx);iR=_clampi(i+1,nx);iRR=_clampi(i+2,nx);iLL=_clampi(i-2,nx)
        jD=_clampi(j-1,ny);jU=_clampi(j+1,ny);jUU=_clampi(j+2,ny);jDD=_clampi(j-2,ny)
        kB=_clampi(l-1,nz);kF=_clampi(l+1,nz);kFF=_clampi(l+2,nz);kBB=_clampi(l-2,nz)
        # left x-face: injected mortar/inlet flux at the sub-grid edge, else interior 3D flux
        if i==1
            FxL=(lFr[j,l],lFmx[j,l],lFmy[j,l],lFmz[j,l],lFE[j,l])
        else
            FxL=_fvflux3T(rho[iL,j,l],ux[iL,j,l],uy[iL,j,l],uz[iL,j,l],Temp[iL,j,l], rho[i,j,l],ux[i,j,l],uy[i,j,l],uz[i,j,l],Temp[i,j,l],
                rho[iLL,j,l],ux[iLL,j,l],uy[iLL,j,l],uz[iLL,j,l],Temp[iLL,j,l], rho[iR,j,l],ux[iR,j,l],uy[iR,j,l],uz[iR,j,l],Temp[iR,j,l], 1, cx_arr,cy_arr,cz_arr,w_arr,cs2,drot,csmag,gauge,shock_aware,drp)
        end
        if i==nx
            FxR=(rFr[j,l],rFmx[j,l],rFmy[j,l],rFmz[j,l],rFE[j,l])
        else
            FxR=_fvflux3T(rho[i,j,l],ux[i,j,l],uy[i,j,l],uz[i,j,l],Temp[i,j,l], rho[iR,j,l],ux[iR,j,l],uy[iR,j,l],uz[iR,j,l],Temp[iR,j,l],
                rho[iL,j,l],ux[iL,j,l],uy[iL,j,l],uz[iL,j,l],Temp[iL,j,l], rho[iRR,j,l],ux[iRR,j,l],uy[iRR,j,l],uz[iRR,j,l],Temp[iRR,j,l], 1, cx_arr,cy_arr,cz_arr,w_arr,cs2,drot,csmag,gauge,shock_aware,drp)
        end
        FyR=_fvflux3T(rho[i,j,l],ux[i,j,l],uy[i,j,l],uz[i,j,l],Temp[i,j,l], rho[i,jU,l],ux[i,jU,l],uy[i,jU,l],uz[i,jU,l],Temp[i,jU,l],
            rho[i,jD,l],ux[i,jD,l],uy[i,jD,l],uz[i,jD,l],Temp[i,jD,l], rho[i,jUU,l],ux[i,jUU,l],uy[i,jUU,l],uz[i,jUU,l],Temp[i,jUU,l], 2, cx_arr,cy_arr,cz_arr,w_arr,cs2,drot,csmag,gauge,shock_aware,drp)
        FyL=_fvflux3T(rho[i,jD,l],ux[i,jD,l],uy[i,jD,l],uz[i,jD,l],Temp[i,jD,l], rho[i,j,l],ux[i,j,l],uy[i,j,l],uz[i,j,l],Temp[i,j,l],
            rho[i,jDD,l],ux[i,jDD,l],uy[i,jDD,l],uz[i,jDD,l],Temp[i,jDD,l], rho[i,jU,l],ux[i,jU,l],uy[i,jU,l],uz[i,jU,l],Temp[i,jU,l], 2, cx_arr,cy_arr,cz_arr,w_arr,cs2,drot,csmag,gauge,shock_aware,drp)
        FzR=_fvflux3T(rho[i,j,l],ux[i,j,l],uy[i,j,l],uz[i,j,l],Temp[i,j,l], rho[i,j,kF],ux[i,j,kF],uy[i,j,kF],uz[i,j,kF],Temp[i,j,kF],
            rho[i,j,kB],ux[i,j,kB],uy[i,j,kB],uz[i,j,kB],Temp[i,j,kB], rho[i,j,kFF],ux[i,j,kFF],uy[i,j,kFF],uz[i,j,kFF],Temp[i,j,kFF], 3, cx_arr,cy_arr,cz_arr,w_arr,cs2,drot,csmag,gauge,shock_aware,drp)
        FzL=_fvflux3T(rho[i,j,kB],ux[i,j,kB],uy[i,j,kB],uz[i,j,kB],Temp[i,j,kB], rho[i,j,l],ux[i,j,l],uy[i,j,l],uz[i,j,l],Temp[i,j,l],
            rho[i,j,kBB],ux[i,j,kBB],uy[i,j,kBB],uz[i,j,kBB],Temp[i,j,kBB], rho[i,j,kF],ux[i,j,kF],uy[i,j,kF],uz[i,j,kF],Temp[i,j,kF], 3, cx_arr,cy_arr,cz_arr,w_arr,cs2,drot,csmag,gauge,shock_aware,drp)
        # adiabatic kinetic wall (reflective flux) at fluid faces bordering solid
        if wall==1 && chi[i,j,l]<=0.5f0
            pw=rho[i,j,l]*cs2*Temp[i,j,l]
            if chi[iR,j,l]>0.5f0; FxR=(0.0f0,pw,0.0f0,0.0f0,0.0f0); end
            if chi[iL,j,l]>0.5f0; FxL=(0.0f0,pw,0.0f0,0.0f0,0.0f0); end
            if chi[i,jU,l]>0.5f0; FyR=(0.0f0,0.0f0,pw,0.0f0,0.0f0); end
            if chi[i,jD,l]>0.5f0; FyL=(0.0f0,0.0f0,pw,0.0f0,0.0f0); end
            if chi[i,j,kF]>0.5f0; FzR=(0.0f0,0.0f0,0.0f0,pw,0.0f0); end
            if chi[i,j,kB]>0.5f0; FzL=(0.0f0,0.0f0,0.0f0,pw,0.0f0); end
        end
        u0x=ux[i,j,l];u0y=uy[i,j,l];u0z=uz[i,j,l];r0=rho[i,j,l]
        Ec=r0*(eint*Temp[i,j,l]+0.5f0*(u0x*u0x+u0y*u0y+u0z*u0z))
        r  = r0      - a*((FxR[1]-FxL[1])+(FyR[1]-FyL[1])+(FzR[1]-FzL[1]))
        mx = r0*u0x  - a*((FxR[2]-FxL[2])+(FyR[2]-FyL[2])+(FzR[2]-FzL[2]))
        my = r0*u0y  - a*((FxR[3]-FxL[3])+(FyR[3]-FyL[3])+(FzR[3]-FzL[3]))
        mz = r0*u0z  - a*((FxR[4]-FxL[4])+(FyR[4]-FyL[4])+(FzR[4]-FzL[4]))
        e  = Ec      - a*((FxR[5]-FxL[5])+(FyR[5]-FyL[5])+(FzR[5]-FzL[5]))
        r = r < 1.0f-3 ? 1.0f-3 : r
        ux2=mx/r;uy2=my/r;uz2=mz/r; sp=sqrt(ux2*ux2+uy2*uy2+uz2*uz2)
        if sp>3.5f0; fscl=3.5f0/sp; ux2*=fscl;uy2*=fscl;uz2*=fscl; end
        T2=(e/r-0.5f0*(ux2*ux2+uy2*uy2+uz2*uz2))/eint; T2=T2<1.0f-3 ? 1.0f-3 : T2
        rho_o[i,j,l]=r; ux_o[i,j,l]=ux2; uy_o[i,j,l]=uy2; uz_o[i,j,l]=uz2; T_o[i,j,l]=T2
    end
end

function amr_stepT3!(backend, rho_o,ux_o,uy_o,uz_o,T_o, rho,ux,uy,uz,Temp, chi,
                     lFr,lFmx,lFmy,lFmz,lFE, rFr,rFmx,rFmy,rFmz,rFE, cx,cy,cz,w,
                     cs2::Float32, dt::Float32, dx::Float32, drot::Float32, csmag::Float32,
                     gauge::Int32, wall::Int32, shock_aware::Int32, drp::Int32, NX,NY,NZ)
    pond_fv3dT_amr_kernel!(backend)(rho_o,ux_o,uy_o,uz_o,T_o, rho,ux,uy,uz,Temp, chi,
        lFr,lFmx,lFmy,lFmz,lFE, rFr,rFmx,rFmy,rFmz,rFE, cx,cy,cz,w,
        cs2, dt/dx, drot, csmag, gauge, wall, shock_aware, drp, Int32(NX),Int32(NY),Int32(NZ); ndrange=(Int(NX),Int(NY),Int(NZ)))
    KernelAbstractions.synchronize(backend); return nothing
end
