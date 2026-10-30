# ==============================================================================
# Particles-on-Demand — PURE 2D conservative kinetic-FV driver  [rectangular, supersonic]
# ==============================================================================
# A genuinely 2D solver (NX×NY arrays, NO third dimension) — the cross-section of an infinite
# cylinder by construction, removing the quasi-2D thin-span ambiguity. Conserved lab moments
# (ρ, ρu, ρv) updated by SHARED co-moving Grad-reconstructed face fluxes (exactly conservative,
# no atomics). Uses the validated D3Q39 velocity set acting in the x-y plane (cs²=2/3, supersonic-
# capable) via _feq2_gpu/_fvflux2 from pond_fv_gpu.jl (must be included first).
#
# This module adds the rectangular (NX≠NY) kernel + the external-aero machinery the square PoC
# lacked: Dirichlet inlet, transmissive outlet, y far-field/sponge, cylinder volume penalization,
# uniform-freestream IC, and an optional 2D SVV (∇⁶ momentum hyperviscosity) ripple cleaner.
# The base flux is the robust upwind KFVS (dissipative ⇒ stable; its numerical Re grows with the
# in-plane resolution, so a high-res run is both stable AND effectively high-Re).
# ==============================================================================

using KernelAbstractions

# ---- rectangular 2D conservative kinetic-FV kernel (separate NX, NY; clamped/transmissive BC) ----
@kernel function pond_fv2d_rect_kernel!(rho_o, mx_o, my_o, @Const(rho), @Const(ux), @Const(uy),
                                        @Const(cx_arr), @Const(cy_arr), @Const(cz_arr), @Const(w_arr),
                                        cs2::Float32, a::Float32, NX::Int32, NY::Int32)
    i, j = @index(Global, NTuple)
    nx = Int(NX); ny = Int(NY)
    @inbounds begin
        iL=_clampi(i-1,nx); iR=_clampi(i+1,nx); iRR=_clampi(i+2,nx); iLL=_clampi(i-2,nx)
        jD=_clampi(j-1,ny); jU=_clampi(j+1,ny); jUU=_clampi(j+2,ny); jDD=_clampi(j-2,ny)
        FrR,FmxR,FmyR = _fvflux2(rho[i,j],ux[i,j],uy[i,j], rho[iR,j],ux[iR,j],uy[iR,j],
                                 rho[iL,j],ux[iL,j],uy[iL,j], rho[iRR,j],ux[iRR,j],uy[iRR,j], 1, cx_arr,cy_arr,cz_arr,w_arr,cs2)
        FrL,FmxL,FmyL = _fvflux2(rho[iL,j],ux[iL,j],uy[iL,j], rho[i,j],ux[i,j],uy[i,j],
                                 rho[iLL,j],ux[iLL,j],uy[iLL,j], rho[iR,j],ux[iR,j],uy[iR,j], 1, cx_arr,cy_arr,cz_arr,w_arr,cs2)
        FrT,FmxT,FmyT = _fvflux2(rho[i,j],ux[i,j],uy[i,j], rho[i,jU],ux[i,jU],uy[i,jU],
                                 rho[i,jD],ux[i,jD],uy[i,jD], rho[i,jUU],ux[i,jUU],uy[i,jUU], 2, cx_arr,cy_arr,cz_arr,w_arr,cs2)
        FrB,FmxB,FmyB = _fvflux2(rho[i,jD],ux[i,jD],uy[i,jD], rho[i,j],ux[i,j],uy[i,j],
                                 rho[i,jDD],ux[i,jDD],uy[i,jDD], rho[i,jU],ux[i,jU],uy[i,jU], 2, cx_arr,cy_arr,cz_arr,w_arr,cs2)
        r  = rho[i,j]         - a*((FrR-FrL)  + (FrT-FrB))
        mx = rho[i,j]*ux[i,j] - a*((FmxR-FmxL)+ (FmxT-FmxB))
        my = rho[i,j]*uy[i,j] - a*((FmyR-FmyL)+ (FmyT-FmyB))
        r = r < 1.0f-3 ? 1.0f-3 : r
        ux2=mx/r; uy2=my/r; sp=sqrt(ux2*ux2+uy2*uy2)
        if sp>3.0f0; f=3.0f0/sp; ux2*=f; uy2*=f; end          # speed backstop (low-ρ runaway guard)
        rho_o[i,j]=r; mx_o[i,j]=ux2; my_o[i,j]=uy2
    end
end

# ---- 2D boundary kernels ----
@kernel function _inlet2_x!(rho,ux,uy, rho_inf::Float32, Uin::Float32, ng::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=ng; rho[i,j]=rho_inf; ux[i,j]=Uin; uy[i,j]=0.0f0; end
end
@kernel function _outlet2_x!(rho,ux,uy, NX::Int32, ng::Int32)
    i,j=@index(Global,NTuple); nx=Int(NX)
    @inbounds if i>nx-ng; s=nx-ng; rho[i,j]=rho[s,j]; ux[i,j]=ux[s,j]; uy[i,j]=uy[s,j]; end
end
@kernel function _farfield2_y!(rho,ux,uy, rho_inf::Float32, Uin::Float32, NY::Int32, ng::Int32)
    i,j=@index(Global,NTuple); ny=Int(NY)
    @inbounds if (j<=ng)||(j>ny-ng); rho[i,j]=rho_inf; ux[i,j]=Uin; uy[i,j]=0.0f0; end
end
@kernel function _sponge2_y!(rho,ux,uy, rho_inf::Float32, Uin::Float32, NY::Int32, sw::Int32, smax::Float32)
    i,j=@index(Global,NTuple); ny=Int(NY); w=Int(sw)
    @inbounds begin
        dy = j-1 < ny-j ? j-1 : ny-j
        if dy<w; t=1.0f0-Float32(dy)/Float32(w); σ=smax*t*t
            rho[i,j]+=σ*(rho_inf-rho[i,j]); ux[i,j]+=σ*(Uin-ux[i,j]); uy[i,j]+=σ*(0.0f0-uy[i,j]); end
    end
end
@kernel function _penalize2!(ux,uy, @Const(chi), invscale::Float32)
    i,j=@index(Global,NTuple)
    @inbounds begin; c=chi[i,j]; if c>0.0f0; f=1.0f0/(1.0f0+invscale*c); ux[i,j]*=f; uy[i,j]*=f; end; end
end

# ---- 2D SVV: ∇⁶ momentum hyperviscosity (k⁶-selective ripple kill; conservative; ε=0 ⇒ off) ----
@kernel function _svv2_kernel!(ux,uy, @Const(rho), eps::Float32, NX::Int32, NY::Int32)
    i,j=@index(Global,NTuple); nx=Int(NX); ny=Int(NY)
    @inbounds begin
        i1m=_clampi(i-1,nx);i1p=_clampi(i+1,nx);i2m=_clampi(i-2,nx);i2p=_clampi(i+2,nx);i3m=_clampi(i-3,nx);i3p=_clampi(i+3,nx)
        j1m=_clampi(j-1,ny);j1p=_clampi(j+1,ny);j2m=_clampi(j-2,ny);j2p=_clampi(j+2,ny);j3m=_clampi(j-3,ny);j3p=_clampi(j+3,ny)
        r0=rho[i,j]; invr=1.0f0/r0
        for comp in 1:2
            u = comp==1 ? ux : uy; m0=r0*u[i,j]
            hx = -(rho[i3m,j]*u[i3m,j]-6f0*rho[i2m,j]*u[i2m,j]+15f0*rho[i1m,j]*u[i1m,j]-20f0*m0+15f0*rho[i1p,j]*u[i1p,j]-6f0*rho[i2p,j]*u[i2p,j]+rho[i3p,j]*u[i3p,j])
            hy = -(rho[i,j3m]*u[i,j3m]-6f0*rho[i,j2m]*u[i,j2m]+15f0*rho[i,j1m]*u[i,j1m]-20f0*m0+15f0*rho[i,j1p]*u[i,j1p]-6f0*rho[i,j2p]*u[i,j2p]+rho[i,j3p]*u[i,j3p])
            u[i,j] = (m0 - eps*(hx+hy))*invr
        end
    end
end

# graded 2D cylinder mask (circle radius R at (cx,cy), smoothed over wband cells)
function pond_cyl2d_chi(NX, NY, cx, cy, R, dx; wband=1.0f0)
    χ=zeros(Float32,NX,NY)
    for j in 1:NY, i in 1:NX
        x=(i-0.5f0)*dx; y=(j-0.5f0)*dx; d=sqrt((x-cx)^2+(y-cy)^2)
        χ[i,j]=clamp((R-d)/(wband*dx)+0.5f0, 0.0f0, 1.0f0)
    end
    return χ
end

mutable struct PonD2D{A,V}
    backend
    rho::A; ux::A; uy::A; rhon::A; uxn::A; uyn::A; chi::A
    cxa::V; cya::V; cza::V; wa::V
    NX::Int; NY::Int; dx::Float32; dt::Float32; cs2::Float32
    rho_inf::Float32; U_inf::Float32; eta::Float32
    svv_eps::Float32; sponge_w::Int; sponge_max::Float32; ng::Int
end

function pond_fv2d_driver(backend, dev, NX, NY; dx, dt, cs2, rho_inf, U_inf, chi_host,
                          eta=1f-3, svv_eps=0f0, sponge_w=0, sponge_max=0.5f0, ng=3,
                          u_init_freestream=true, cxv, cyv, czv, wv)
    ux0 = u_init_freestream ? Float32(U_inf) : 0.0f0
    rho=dev(fill(Float32(rho_inf),NX,NY)); ux=dev(fill(ux0,NX,NY)); uy=dev(zeros(Float32,NX,NY))
    rhon=dev(zeros(Float32,NX,NY)); uxn=dev(zeros(Float32,NX,NY)); uyn=dev(zeros(Float32,NX,NY))
    chi=dev(chi_host); cxa=dev(Float32.(cxv)); cya=dev(Float32.(cyv)); cza=dev(Float32.(czv)); wa=dev(Float32.(wv))
    return PonD2D{typeof(rho),typeof(cxa)}(backend, rho,ux,uy, rhon,uxn,uyn, chi, cxa,cya,cza,wa,
        NX,NY, Float32(dx),Float32(dt),Float32(cs2), Float32(rho_inf),Float32(U_inf),Float32(eta),
        Float32(svv_eps), Int(sponge_w), Float32(sponge_max), ng)
end

function pond_fv2d_advance!(d::PonD2D, t::Int)
    nd=(d.NX,d.NY)
    pond_fv2d_rect_kernel!(d.backend)(d.rhon,d.uxn,d.uyn, d.rho,d.ux,d.uy, d.cxa,d.cya,d.cza,d.wa,
        d.cs2, d.dt/d.dx, Int32(d.NX),Int32(d.NY); ndrange=nd)
    KernelAbstractions.synchronize(d.backend)
    d.rho,d.rhon=d.rhon,d.rho; d.ux,d.uxn=d.uxn,d.ux; d.uy,d.uyn=d.uyn,d.uy
    if d.svv_eps>0f0
        _svv2_kernel!(d.backend)(d.ux,d.uy, d.rho, d.svv_eps, Int32(d.NX),Int32(d.NY); ndrange=nd)
        KernelAbstractions.synchronize(d.backend)
    end
    _penalize2!(d.backend)(d.ux,d.uy, d.chi, Float32(d.dt/d.eta); ndrange=nd); KernelAbstractions.synchronize(d.backend)
    _inlet2_x!(d.backend)(d.rho,d.ux,d.uy, d.rho_inf, d.U_inf, Int32(d.ng); ndrange=nd); KernelAbstractions.synchronize(d.backend)
    if d.sponge_w>0
        _sponge2_y!(d.backend)(d.rho,d.ux,d.uy, d.rho_inf, d.U_inf, Int32(d.NY), Int32(d.sponge_w), d.sponge_max; ndrange=nd)
    else
        _farfield2_y!(d.backend)(d.rho,d.ux,d.uy, d.rho_inf, d.U_inf, Int32(d.NY), Int32(d.ng); ndrange=nd)
    end
    KernelAbstractions.synchronize(d.backend)
    _outlet2_x!(d.backend)(d.rho,d.ux,d.uy, Int32(d.NX), Int32(d.ng); ndrange=nd); KernelAbstractions.synchronize(d.backend)
    return nothing
end
