# ==============================================================================
# Particles-on-Demand — 3D NESTED-BOX AMR (D3Q27 philosophy)  [GPU]
# ==============================================================================
# A coarse BASE grid L1 (dx=2) covers the WHOLE domain; a fine BOX L2 (dx=1) wraps the body in ALL three
# directions, nested inside L1. This is the block-refinement philosophy of the production D3Q27 solver
# (vs the coarse|fine|coarse X-stripe): the fine cells sit only where they are needed (body + bow shock +
# near wake), so for a given VRAM the sphere is resolved far finer, and the coarse base renders the whole
# domain. Coupling per step (Berger-Oliger, 2 levels, same Δt):
#   (1) PROLONG  L1 → L2 ghost halo (2 layers) — piecewise-constant refinement (each L1 cell → its 2³ fine
#       sub-cells) supplies L2's boundary condition;
#   (2) STEP L2 over its interior (the standard single-grid flux kernel, reading the halo) + penalize;
#   (3) STEP L1 over the whole domain + penalize + domain BCs;
#   (4) RESTRICT L2 interior → the L1 cells it covers — averages the CONSERVED variables (ρ, ρu, ρE) over
#       each 2³ block ⇒ mass/momentum/energy consistent, so a uniform stream is preserved exactly.
# L2 has a g=2 ghost halo (enough for the ±2 reconstruction stencil). The box occupies L1 cells
# [ci0:ci0+nbx-1]×[cj0:cj0+nby-1]×[ck0:ck0+nbz-1]; L2 interior index = g + 2·(box-local) .
# ==============================================================================

using KernelAbstractions

# Catmull-Rom cubic weights (C¹) for the 4-point stencil [base-1,base,base+1,base+2] at fraction t∈[0,1).
@inline function _cr4(t::Float32)
    t2=t*t; t3=t2*t
    return (-0.5f0*t3+t2-0.5f0*t, 1.5f0*t3-2.5f0*t2+1.0f0, -1.5f0*t3+2.0f0*t2+0.5f0*t, 0.5f0*t3-0.5f0*t2)
end
@inline _cl6(x,n)=x<1 ? 1 : (x>n ? n : x)
# Roadmap 1.4: 6-point QUINTIC LAGRANGE weights for the stencil offsets (-2,-1,0,1,2,3) at fraction t∈[0,1)
# from node 0. 5th-order accurate ⇒ reproduces polynomials up to degree 5 EXACTLY (constant, linear,
# quadratic, cubic, quartic, quintic) ⇒ a smooth ρ/u/T gradient prolongs with essentially ZERO box-edge step,
# the visible AMR-seam residual dropping from the cubic's O(h⁴) to O(h⁶). Partition of unity (Σw=1 ∀t).
@inline function _ql6(t::Float32)
    tm2=t+2f0; tm1=t+1f0; t0=t; tp1=t-1f0; tp2=t-2f0; tp3=t-3f0
    return ( (tm1*t0*tp1*tp2*tp3)/(-120f0),   # offset -2  (denom ∏(-2-x_m) = -120)
             (tm2*t0*tp1*tp2*tp3)/(  24f0),   # offset -1  (24)
             (tm2*tm1*tp1*tp2*tp3)/(-12f0),   # offset  0  (-12)
             (tm2*tm1*t0*tp2*tp3)/(  12f0),   # offset +1  (12)
             (tm2*tm1*t0*tp1*tp3)/( -24f0),   # offset +2  (-24)
             (tm2*tm1*t0*tp1*tp2)/( 120f0) )  # offset +3  (120)
end
# PROLONG: fill L2's ghost halo (the g-layer border) from L1 by TRICUBIC (Catmull-Rom, C¹) refinement.
# A cell-centred fine sub-cell sits at L1-continuous coordinate X = ci0 − ¼ + ½·(fine-index): the two fine
# sub-cells of an L1 cell are at ∓¼ of its centre. We interpolate the PRIMITIVES (ρ,u,T) with a 4×4×4 cubic
# stencil (the halo is a boundary condition, not a conserved region, so pointwise high-order is exact-safe).
# This lifts the spatial level coupling from piecewise-constant (C⁻¹, the visible box-edge staircase) to C¹.
@kernel function prolong_l1l2!(l2r,l2ux,l2uy,l2uz,l2T,
        @Const(l1r),@Const(l1ux),@Const(l1uy),@Const(l1uz),@Const(l1T),
        ci0::Int32,cj0::Int32,ck0::Int32, g::Int32, N1X::Int32,N1Y::Int32,N1Z::Int32, N2X::Int32,N2Y::Int32,N2Z::Int32)
    i2,j2,l2=@index(Global,NTuple); n2x=Int(N2X);n2y=Int(N2Y);n2z=Int(N2Z);gg=Int(g); n1x=Int(N1X);n1y=Int(N1Y);n1z=Int(N1Z)
    @inbounds begin
        isghost = (i2<=gg)||(i2>n2x-gg)||(j2<=gg)||(j2>n2y-gg)||(l2<=gg)||(l2>n2z-gg)
        if isghost
            Xf=Float32(ci0)-0.25f0+0.5f0*Float32(i2-gg-1); Yf=Float32(cj0)-0.25f0+0.5f0*Float32(j2-gg-1); Zf=Float32(ck0)-0.25f0+0.5f0*Float32(l2-gg-1)
            bx=floor(Int,Xf); by=floor(Int,Yf); bz=floor(Int,Zf)
            wx=_cr4(Xf-bx); wy=_cr4(Yf-by); wz=_cr4(Zf-bz)
            sr=0.0f0;sx=0.0f0;sy=0.0f0;sz=0.0f0;sT=0.0f0
            for dk in 0:3
                ck=_cl6(bz-1+dk,n1z); wk=wz[dk+1]
                for dj in 0:3
                    cj=_cl6(by-1+dj,n1y); wjk=wk*wy[dj+1]
                    for di in 0:3
                        ci=_cl6(bx-1+di,n1x); ww=wjk*wx[di+1]
                        sr+=ww*l1r[ci,cj,ck]; sx+=ww*l1ux[ci,cj,ck]; sy+=ww*l1uy[ci,cj,ck]; sz+=ww*l1uz[ci,cj,ck]; sT+=ww*l1T[ci,cj,ck]
                    end
                end
            end
            l2r[i2,j2,l2]=sr; l2ux[i2,j2,l2]=sx; l2uy[i2,j2,l2]=sy; l2uz[i2,j2,l2]=sz; l2T[i2,j2,l2]=(sT<1.0f-3 ? 1.0f-3 : sT)
        end
    end
end

# Roadmap 1.4: QUINTIC (6³-point Lagrange, 5th-order) sibling of prolong_l1l2!. Same halo fill, but the 4×4×4
# Catmull-Rom stencil is replaced by a 6×6×6 quintic Lagrange stencil (offsets -2..+3 per axis). Separate
# kernel (not a branch) to keep each kernel small for the GPU JIT. Default path stays cubic; select via flag.
@kernel function prolong_l1l2_q5!(l2r,l2ux,l2uy,l2uz,l2T,
        @Const(l1r),@Const(l1ux),@Const(l1uy),@Const(l1uz),@Const(l1T),
        ci0::Int32,cj0::Int32,ck0::Int32, g::Int32, N1X::Int32,N1Y::Int32,N1Z::Int32, N2X::Int32,N2Y::Int32,N2Z::Int32)
    i2,j2,l2=@index(Global,NTuple); n2x=Int(N2X);n2y=Int(N2Y);n2z=Int(N2Z);gg=Int(g); n1x=Int(N1X);n1y=Int(N1Y);n1z=Int(N1Z)
    @inbounds begin
        isghost = (i2<=gg)||(i2>n2x-gg)||(j2<=gg)||(j2>n2y-gg)||(l2<=gg)||(l2>n2z-gg)
        if isghost
            Xf=Float32(ci0)-0.25f0+0.5f0*Float32(i2-gg-1); Yf=Float32(cj0)-0.25f0+0.5f0*Float32(j2-gg-1); Zf=Float32(ck0)-0.25f0+0.5f0*Float32(l2-gg-1)
            bx=floor(Int,Xf); by=floor(Int,Yf); bz=floor(Int,Zf)
            wx=_ql6(Xf-bx); wy=_ql6(Yf-by); wz=_ql6(Zf-bz)
            sr=0.0f0;sx=0.0f0;sy=0.0f0;sz=0.0f0;sT=0.0f0
            for dk in 0:5
                ck=_cl6(bz-2+dk,n1z); wk=wz[dk+1]
                for dj in 0:5
                    cj=_cl6(by-2+dj,n1y); wjk=wk*wy[dj+1]
                    for di in 0:5
                        ci=_cl6(bx-2+di,n1x); ww=wjk*wx[di+1]
                        sr+=ww*l1r[ci,cj,ck]; sx+=ww*l1ux[ci,cj,ck]; sy+=ww*l1uy[ci,cj,ck]; sz+=ww*l1uz[ci,cj,ck]; sT+=ww*l1T[ci,cj,ck]
                    end
                end
            end
            l2r[i2,j2,l2]=sr; l2ux[i2,j2,l2]=sx; l2uy[i2,j2,l2]=sy; l2uz[i2,j2,l2]=sz; l2T[i2,j2,l2]=(sT<1.0f-3 ? 1.0f-3 : sT)
        end
    end
end

# FEATHER: damp the spurious waves the coarse flux radiates where it straddles the box interface (the
# restricted fine-truth cells inside vs the coarse-evolved cells outside differ, so the L1 flux across the
# box face sees a step and rings). A tapered Laplacian smoothing in a shell of width `fw` straddling the box
# surface — strongest at the face, fading to zero at ±fw — dissipates the interface reflection at its source.
# Reads from the scratch copy `*s` (race-free), writes only band cells of the live L1 arrays. The band sits
# far from the body (deep inside the box), so it never touches the L2 forces; it only cleans the coarse base.
@inline _boxout(i,a,b)= i<a ? (a-i) : (i>b ? (i-b) : 0)
@kernel function feather_l1!(r,ux,uy,uz,T, @Const(rs),@Const(uxs),@Const(uys),@Const(uzs),@Const(Ts),
        ci0::Int32,i1::Int32,cj0::Int32,j1::Int32,ck0::Int32,k1::Int32, fw::Int32, alpha::Float32, N1X::Int32,N1Y::Int32,N1Z::Int32)
    i,j,l=@index(Global,NTuple); nx=Int(N1X);ny=Int(N1Y);nz=Int(N1Z); f=Int(fw)
    @inbounds begin
        dxo=_boxout(i,Int(ci0),Int(i1)); dyo=_boxout(j,Int(cj0),Int(j1)); dzo=_boxout(l,Int(ck0),Int(k1))
        outside = dxo>0||dyo>0||dzo>0
        w=0.0f0
        if outside
            d=max(dxo,dyo,dzo); if d<=f; w=1.0f0-Float32(d-1)/Float32(f); end
        else
            din=min(i-Int(ci0),Int(i1)-i,j-Int(cj0),Int(j1)-j,l-Int(ck0),Int(k1)-l); if din<f; w=1.0f0-Float32(din)/Float32(f); end
        end
        if w>0.0f0
            im=max(i-1,1);ip=min(i+1,nx);jm=max(j-1,1);jp=min(j+1,ny);lm=max(l-1,1);lp=min(l+1,nz); aw=alpha*w
            r[i,j,l] =rs[i,j,l] +aw*((rs[im,j,l]+rs[ip,j,l]+rs[i,jm,l]+rs[i,jp,l]+rs[i,j,lm]+rs[i,j,lp])/6f0 -rs[i,j,l])
            ux[i,j,l]=uxs[i,j,l]+aw*((uxs[im,j,l]+uxs[ip,j,l]+uxs[i,jm,l]+uxs[i,jp,l]+uxs[i,j,lm]+uxs[i,j,lp])/6f0-uxs[i,j,l])
            uy[i,j,l]=uys[i,j,l]+aw*((uys[im,j,l]+uys[ip,j,l]+uys[i,jm,l]+uys[i,jp,l]+uys[i,j,lm]+uys[i,j,lp])/6f0-uys[i,j,l])
            uz[i,j,l]=uzs[i,j,l]+aw*((uzs[im,j,l]+uzs[ip,j,l]+uzs[i,jm,l]+uzs[i,jp,l]+uzs[i,j,lm]+uzs[i,j,lp])/6f0-uzs[i,j,l])
            T[i,j,l] =Ts[i,j,l] +aw*((Ts[im,j,l]+Ts[ip,j,l]+Ts[i,jm,l]+Ts[i,jp,l]+Ts[i,j,lm]+Ts[i,j,lp])/6f0 -Ts[i,j,l])
        end
    end
end

# RESTRICT: overwrite the L1 box cells with the conserved-variable average of each covered 2³ L2 block.
@kernel function restrict_l2l1!(l1r,l1ux,l1uy,l1uz,l1T,
        @Const(l2r),@Const(l2ux),@Const(l2uy),@Const(l2uz),@Const(l2T),
        ci0::Int32,cj0::Int32,ck0::Int32, g::Int32, eint::Float32, NBX::Int32,NBY::Int32,NBZ::Int32)
    a,b,c=@index(Global,NTuple); gg=Int(g)
    @inbounds begin
        ci=Int(ci0)+a-1; cj=Int(cj0)+b-1; ck=Int(ck0)+c-1
        i0=gg+2*(a-1); j0=gg+2*(b-1); k0=gg+2*(c-1)
        sr=0.0f0;smx=0.0f0;smy=0.0f0;smz=0.0f0;sE=0.0f0
        for dk in 1:2, dj in 1:2, di in 1:2
            i2=i0+di; j2=j0+dj; l2=k0+dk
            r=l2r[i2,j2,l2]; ux=l2ux[i2,j2,l2]; uy=l2uy[i2,j2,l2]; uz=l2uz[i2,j2,l2]; T=l2T[i2,j2,l2]
            sr+=r; smx+=r*ux; smy+=r*uy; smz+=r*uz; sE+=r*(eint*T+0.5f0*(ux*ux+uy*uy+uz*uz))
        end
        r1=sr*0.125f0; ux1=smx/sr; uy1=smy/sr; uz1=smz/sr
        T1=(sE/sr-0.5f0*(ux1*ux1+uy1*uy1+uz1*uz1))/eint
        l1r[ci,cj,ck]=r1; l1ux[ci,cj,ck]=ux1; l1uy[ci,cj,ck]=uy1; l1uz[ci,cj,ck]=uz1; l1T[ci,cj,ck]=(T1<1.0f-3 ? 1.0f-3 : T1)
    end
end
