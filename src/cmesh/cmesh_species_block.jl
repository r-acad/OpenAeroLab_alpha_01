# ---- Block wall-normal line operator (speed-up plan W3a, 30 Sep 2026) ----------------------------------------------
# Why (memory/project_docs/2026-09-30_W3A_BLOCK_LINE_OPERATOR.md): the scalar line relaxation (cmesh_species_steady_line.jl) couples
# each conserved component to its j neighbours through one symmetric, diffusion-like coefficient, a face-speed bound. A
# diffusion operator vanishes like k^2 at long wavelengths, while the wall-normal waves it has to absorb (acoustics,
# advection) vanish like k: at a large wall-normal Courant number a smooth wave is amplified rather than damped by
# (I + dt W)^-1 (I + dt J). The operator here is the Jacobian of the scheme's own first-order j faces:
#   * per j face, the 4x4 blocks dF/dU_L and dF/dU_R of the production AUSM+-up mixture face (a Float64 transcription of
#     _fvflux_mix_face; U = (rho, rho u, rho v, q)), by one-sided differences of relative step 1e-6 through the
#     production EOS _phase_state, plus the thin-layer viscous traction (4/3 on the normal component); the foil wall
#     face is the wall's own flux (the wall cell's pressure, the Stokes traction), the outer face is linearised in its
#     interior cell with the ghost held fixed;
#   * one block-tridiagonal system per j line: foil columns from the wall to the outer boundary; each wake-column pair
#     (w, NI+1-w) through the cut, whose face is linearised once, from column w's side, so the implicit correction
#     telescopes across it as across every interior face;
#   * each stage solves (V/dt I + J) D = rhs (rhs the stage's flux divergence minus source, as the trial kernel forms
#     it) and takes U - D: a linearly implicit (W-method) stage. With J = 0 it is the explicit stage.
# The turbulence scalars get the matching scalar operator: the upwind derivative of their j advection by the carrier
# mass flux, their j diffusion, and the source-stiffness bound `rate` point-implicit in the diagonal.
# Definitions only: nothing in the physical-time step reaches this file.

const _BLK_REL = 1.0e-6

# 4x4 blocks as row-major NTuple{16,Float64}.
@inline _b4(r,c)=(r-1)*4+c
@inline _b4zero()=ntuple(_->0.0,Val(16))
@inline _b4eye(d::Float64)=ntuple(k->(k==1 || k==6 || k==11 || k==16) ? d : 0.0,Val(16))
@inline _b4add(A,B)=ntuple(k->A[k]+B[k],Val(16))
@inline _b4sub(A,B)=ntuple(k->A[k]-B[k],Val(16))
@inline _b4neg(A)=ntuple(k->-A[k],Val(16))
@inline function _b4mul(A,B)
    ntuple(Val(16)) do k
        r=(k-1)>>2; c=(k-1)&3
        A[4r+1]*B[c+1]+A[4r+2]*B[c+5]+A[4r+3]*B[c+9]+A[4r+4]*B[c+13]
    end
end
@inline _b4vec(A,x)=ntuple(r->A[4r-3]*x[1]+A[4r-2]*x[2]+A[4r-1]*x[3]+A[4r]*x[4],Val(4))
@inline _b4load(J,i,j)=ntuple(k->@inbounds(J[i,j,k]),Val(16))
# Inverse by the adjugate (the cofactor expansion of MESA's gluInvertMatrix; valid for either storage order).
@inline function _b4inv(m)
    m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15=m
    i0=m5*m10*m15-m5*m11*m14-m9*m6*m15+m9*m7*m14+m13*m6*m11-m13*m7*m10
    i4=-m4*m10*m15+m4*m11*m14+m8*m6*m15-m8*m7*m14-m12*m6*m11+m12*m7*m10
    i8=m4*m9*m15-m4*m11*m13-m8*m5*m15+m8*m7*m13+m12*m5*m11-m12*m7*m9
    i12=-m4*m9*m14+m4*m10*m13+m8*m5*m14-m8*m6*m13-m12*m5*m10+m12*m6*m9
    i1=-m1*m10*m15+m1*m11*m14+m9*m2*m15-m9*m3*m14-m13*m2*m11+m13*m3*m10
    i5=m0*m10*m15-m0*m11*m14-m8*m2*m15+m8*m3*m14+m12*m2*m11-m12*m3*m10
    i9=-m0*m9*m15+m0*m11*m13+m8*m1*m15-m8*m3*m13-m12*m1*m11+m12*m3*m9
    i13=m0*m9*m14-m0*m10*m13-m8*m1*m14+m8*m2*m13+m12*m1*m10-m12*m2*m9
    i2=m1*m6*m15-m1*m7*m14-m5*m2*m15+m5*m3*m14+m13*m2*m7-m13*m3*m6
    i6=-m0*m6*m15+m0*m7*m14+m4*m2*m15-m4*m3*m14-m12*m2*m7+m12*m3*m6
    i10=m0*m5*m15-m0*m7*m13-m4*m1*m15+m4*m3*m13+m12*m1*m7-m12*m3*m5
    i14=-m0*m5*m14+m0*m6*m13+m4*m1*m14-m4*m2*m13-m12*m1*m6+m12*m2*m5
    i3=-m1*m6*m11+m1*m7*m10+m5*m2*m11-m5*m3*m10-m9*m2*m7+m9*m3*m6
    i7=m0*m6*m11-m0*m7*m10-m4*m2*m11+m4*m3*m10+m8*m2*m7-m8*m3*m6
    i11=-m0*m5*m11+m0*m7*m9+m4*m1*m11-m4*m3*m9-m8*m1*m7+m8*m3*m5
    i15=m0*m5*m10-m0*m6*m9-m4*m1*m10+m4*m2*m9+m8*m1*m6-m8*m2*m5
    d=1.0/(m0*i0+m1*i4+m2*i8+m3*i12)
    (i0*d,i1*d,i2*d,i3*d,i4*d,i5*d,i6*d,i7*d,i8*d,i9*d,i10*d,i11*d,i12*d,i13*d,i14*d,i15*d)
end

# Primitive state of a conserved Float64 state through the production EOS (the double-precision path).
@inline function _blk_prim(r,mx,my,q,ep::NTuple{7,Float64})
    pv,a2l,rl,rv,psm,lq,afloor=ep
    if !(isfinite(r) && r>0.0 && isfinite(q) && 0.0<=q<=r && isfinite(mx) && isfinite(my))
        return (false,0.0,0.0,0.0,0.0,0.0)
    end
    y=q/r
    good,p,c=_phase_state(r,_alpha(y,rl,rv),pv,a2l,rl,rv,psm,afloor,lq)
    (good,mx/r,my/r,Float64(p),Float64(c),y)
end

# The first-order AUSM+-up mixture face in Float64 (the arithmetic of _fvflux_mix_face, without the energy flux), with
# the vapour partial density carried by the donor's mass fraction as _k_species_flux_j! does.
@inline function _blk_face(rL,uL,vL,pL,cL,yL,rR,uR,vR,pR,cR,yR,minf2::Float64,nx::Float64,ny::Float64,S::Float64)
    unL=uL*nx+vL*ny; utL=-uL*ny+vL*nx; unR=uR*nx+vR*ny; utR=-uR*ny+vR*nx
    af=0.5*(cL+cR); Mb2=0.5*(unL*unL+unR*unR)/(af*af)
    Mo2=minf2==0.0 ? 1.0 : min(1.0,max(Mb2,minf2)); Mo=sqrt(Mo2); fa=Mo*(2.0-Mo)
    ML=unL/af; MR=unR/af; rf=0.5*(rL+rR)
    Mh=CMeshSolverKA._ausm_M4p(ML)+CMeshSolverKA._ausm_M4m(MR)-(0.25/fa)*max(1.0-Mb2,0.0)*(pR-pL)/(rf*af*af)
    mdot=af*Mh*(Mh>0.0 ? rL : rR)
    alp=0.1875*(-4.0+5.0*fa*fa)
    Pp=CMeshSolverKA._ausm_P5p(ML,alp); Pm=CMeshSolverKA._ausm_P5m(MR,alp)
    pf=Pp*pL+Pm*pR-0.75*Pp*Pm*(rL+rR)*(fa*af)*(unR-unL)
    unU=mdot>0.0 ? unL : unR; utU=mdot>0.0 ? utL : utR; yU=mdot>0.0 ? yL : yR
    Fmn=mdot*unU+pf; Fmt=mdot*utU
    (mdot*S,(nx*Fmn-ny*Fmt)*S,(ny*Fmn+nx*Fmt)*S,mdot*yU*S)
end

# One-sided difference step of component k (q moves inward from its bounds).
@inline function _blk_step(U,k,c)
    r=U[1]
    k==1 && return _BLK_REL*r
    k==4 && return U[4]+_BLK_REL*r<=r ? _BLK_REL*r : -_BLK_REL*r
    _BLK_REL*max(abs(U[k]),r*c)
end
@inline _blk_perturb(U,k,h)=ntuple(m->m==k ? U[m]+h : U[m],Val(4))

# Thin-layer viscous traction of a face between velocity states L and R (conductance mu = rho_f nu_f S/dn):
# F_m = -mu [(u_R - u_L) + ((u_R - u_L).n / 3) n]; its derivatives in the conserved variables of the side with sign sg
# (+1 for R, -1 for L) are added into J.
@inline function _blk_visc_add(J,mu,sg,r,u,v,nx,ny)
    un=u*nx+v*ny; f=sg*mu/r
    J=Base.setindex(J,J[_b4(2,1)]+f*(u+un*nx/3.0),_b4(2,1))
    J=Base.setindex(J,J[_b4(2,2)]-f*(1.0+nx*nx/3.0),_b4(2,2))
    J=Base.setindex(J,J[_b4(2,3)]-f*(nx*ny/3.0),_b4(2,3))
    J=Base.setindex(J,J[_b4(3,1)]+f*(v+un*ny/3.0),_b4(3,1))
    J=Base.setindex(J,J[_b4(3,2)]-f*(nx*ny/3.0),_b4(3,2))
    Base.setindex(J,J[_b4(3,3)]-f*(1.0+ny*ny/3.0),_b4(3,3))
end

# The committed Float64 state of the other block's cell across a strip-pair interface face (station st; fc 1 the lower
# interface, 2 the upper): layer 1 of the exchange's buffer U_nb (NIW, g, 4, 2), cmesh_species_strip.jl.
@inline _blk_unb4(Unb,st,fc)=(Unb[st,1,1,fc],Unb[st,1,2,fc],Unb[st,1,3,fc],Unb[st,1,4,fc])

# The face Jacobians of every j face (i, j), j = 1..NJ+1, in the production orientation (L = the j-1 side, R = the j
# side), area-integrated like the fluxes. A state the EOS refuses leaves its columns zero (a weaker, still consistent W).
# Strip topology (stage 4, 7 Oct 2026; memory/project_docs/2026-10-05_STRIP_TOPOLOGY_PLAN.md section 5 item 2, critique A 5): with the strip
# block's context (ctx.block == 1) its j = 1 face is the lower interface, whose left state is the lower leg's row-1 cell, and
# its j = NJ+1 face the upper interface, whose right state is the upper leg's row-1 cell: both the other block's committed
# Float64 U from the exchange's buffer ctx.U_nb (as the cut reads the partner's U[ip, 1], never a ghost), the viscous centre
# distance from the static table ctx.c_nb, and the outer branch (JR = 0, the ghost held fixed) yields to the upper interface.
# So each interface face is linearised once, by the strip's launch; the wake-line twin reads those Jacobians for both cells
# of the face. ctx === nothing (every cache without a strip) or the C block's context (block 0): today's arithmetic, the C
# block's leg columns linearised across the within-block cut as before (numbers the twin never reads).
@kernel function _k_blk_face_jac!(JL,JR,@Const(U),@Const(rho),@Const(ux),@Const(uy),@Const(alpha),@Const(p),@Const(am),
                                  @Const(nut),@Const(vol),@Const(xc),@Const(yc),@Const(Sjx),@Const(Sjy),
                                  ep::NTuple{7,Float64},minf2::Float64,nu::Float32,wallmode::Int32,sst_on::Int32,
                                  NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32,ctx)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ+1
        sx=Float64(Sjx[i,j]); sy=Float64(Sjy[i,j]); S=sqrt(sx*sx+sy*sy); nx=sx/S; ny=sy/S
        rl=ep[3]; rv=ep[4]
        jl=_b4zero(); jr=_b4zero()
        sb=ctx!==nothing && ctx.block==Int32(1)   # the strip block of a strip pair (compiled away without a context)
        if j==1 && NIW<i<=NIW+NIF
            # foil wall face: F = (0, p_w S n - traction, 0) on the wall cell (the right state)
            U1=(U[i,1,1],U[i,1,2],U[i,1,3],U[i,1,4])
            ok,_,_,pw,cw,_=_blk_prim(U1[1],U1[2],U1[3],U1[4],ep)
            if ok
                for k in 1:4
                    h=_blk_step(U1,k,cw); Up=_blk_perturb(U1,k,h)
                    okp,_,_,pp,_,_=_blk_prim(Up[1],Up[2],Up[3],Up[4],ep)
                    dp=okp ? (pp-pw)/h : 0.0
                    jr=Base.setindex(jr,dp*sx,_b4(2,k)); jr=Base.setindex(jr,dp*sy,_b4(3,k))
                end
                if nu>0.0f0 && wallmode==Int32(2)
                    # Stokes wall traction rho nu S/d1 (u + (u.n/3) n) = nu S/d1 (m + (m.n/3) n): linear in m
                    mu=Float64(nu)*S/(0.5*Float64(vol[i,1])/S)
                    jr=Base.setindex(jr,jr[_b4(2,2)]-mu*(1.0+nx*nx/3.0),_b4(2,2))
                    jr=Base.setindex(jr,jr[_b4(2,3)]-mu*(nx*ny/3.0),_b4(2,3))
                    jr=Base.setindex(jr,jr[_b4(3,2)]-mu*(nx*ny/3.0),_b4(3,2))
                    jr=Base.setindex(jr,jr[_b4(3,3)]-mu*(1.0+ny*ny/3.0),_b4(3,3))
                end
            end
        else
            outer=j==NJ+1 && !sb                  # the strip's j = NJ+1 row is the upper interface, not an outer boundary
            ip=NI+Int32(1)-i
            # left: the cell below, or the wake partner across the cut (the strip: the lower leg's row-1 cell, U_nb); right: the
            # cell above, or the fixed outer ghost (the strip's top row: the upper leg's row-1 cell, U_nb)
            UL=j==1 ? (sb ? _blk_unb4(ctx.U_nb,i,Int32(1)) : (U[ip,1,1],U[ip,1,2],U[ip,1,3],U[ip,1,4])) :
                (U[i,j-1,1],U[i,j-1,2],U[i,j-1,3],U[i,j-1,4])
            okL,uL,vL,pL,cL,yL=_blk_prim(UL[1],UL[2],UL[3],UL[4],ep)
            local okR::Bool,rR::Float64,uR::Float64,vR::Float64,pR::Float64,cR::Float64,yR::Float64
            UR=outer ? (0.0,0.0,0.0,0.0) : (sb && j==NJ+1 ? _blk_unb4(ctx.U_nb,i,Int32(2)) : (U[i,j,1],U[i,j,2],U[i,j,3],U[i,j,4]))
            if outer
                ia=i+g; jg=NJ+g+Int32(1)
                rR=Float64(rho[ia,jg]); uR=Float64(ux[ia,jg]); vR=Float64(uy[ia,jg])
                pR=Float64(p[ia,jg]); cR=Float64(am[ia,jg]); yR=Float64(_Y(alpha[ia,jg],Float32(rl),Float32(rv)))
                okR=isfinite(rR) && rR>0.0 && isfinite(pR) && pR>0.0 && isfinite(cR) && cR>0.0 && isfinite(uR) && isfinite(vR)
            else
                rR=UR[1]
                okR,uR,vR,pR,cR,yR=_blk_prim(UR[1],UR[2],UR[3],UR[4],ep)
            end
            if okL && okR
                F0=_blk_face(UL[1],uL,vL,pL,cL,yL,rR,uR,vR,pR,cR,yR,minf2,nx,ny,S)
                for k in 1:4
                    h=_blk_step(UL,k,cL); Up=_blk_perturb(UL,k,h)
                    okp,up,vp,pp,cp,yp=_blk_prim(Up[1],Up[2],Up[3],Up[4],ep)
                    if okp
                        Fp=_blk_face(Up[1],up,vp,pp,cp,yp,rR,uR,vR,pR,cR,yR,minf2,nx,ny,S)
                        for r in 1:4; jl=Base.setindex(jl,(Fp[r]-F0[r])/h,_b4(r,k)); end
                    end
                    if !outer
                        h=_blk_step(UR,k,cR); Up=_blk_perturb(UR,k,h)
                        okp,up,vp,pp,cp,yp=_blk_prim(Up[1],Up[2],Up[3],Up[4],ep)
                        if okp
                            Fp=_blk_face(UL[1],uL,vL,pL,cL,yL,Up[1],up,vp,pp,cp,yp,minf2,nx,ny,S)
                            for r in 1:4; jr=Base.setindex(jr,(Fp[r]-F0[r])/h,_b4(r,k)); end
                        end
                    end
                end
                if nu>0.0f0 && !outer
                    ia=i+g; ja=j+g
                    # the centre distance across the face; at the strip's interfaces from the static table of the other block's
                    # cell centroids (ctx.c_nb), as the strip's flux kernel forms its interface conductance
                    dx=j==1 ? Float64(xc[i,1])-(sb ? Float64(ctx.c_nb[i,1,1]) : Float64(xc[ip,1])) :
                        (sb && j==NJ+1 ? Float64(ctx.c_nb[i,1,2])-Float64(xc[i,NJ]) : Float64(xc[i,j])-Float64(xc[i,j-1]))
                    dy=j==1 ? Float64(yc[i,1])-(sb ? Float64(ctx.c_nb[i,2,1]) : Float64(yc[ip,1])) :
                        (sb && j==NJ+1 ? Float64(ctx.c_nb[i,2,2])-Float64(yc[i,NJ]) : Float64(yc[i,j])-Float64(yc[i,j-1]))
                    dn=sqrt(dx*dx+dy*dy)+1.0e-30
                    nf=Float64(nu)+(sst_on==Int32(1) ? 0.5*(Float64(nut[ia,ja-1])+Float64(nut[ia,ja])) : 0.0)
                    mu=0.5*(UL[1]+UR[1])*nf*S/dn
                    jl=_blk_visc_add(jl,mu,-1.0,UL[1],uL,vL,nx,ny)
                    jr=_blk_visc_add(jr,mu,1.0,UR[1],uR,vR,nx,ny)
                end
            end
        end
        # The vapour row keeps only its own upwind derivative, d(mdot Y_donor)/dq_donor (30 Sep, found by the first
        # cavitating CPU runs): its derivatives in rho and momentum (Y dmdot/dU) couple a vapour-free cell at a cavity's
        # edge to its neighbours' flow increments, so that its q moved although its own q right-hand side was exactly zero,
        # and q < 0 was reached at any dt (retries exhausted down to 4e-8). Without them the q row of every block, and
        # through the block Thomas elimination of the whole line, is [0 0 0 x]: q solves a scalar upwind M-matrix system,
        # positivity-preserving as the explicit donor flux is. The pressure coupling to q stays in the flow rows' q column.
        jl=Base.setindex(Base.setindex(Base.setindex(jl,0.0,_b4(4,1)),0.0,_b4(4,2)),0.0,_b4(4,3))
        jr=Base.setindex(Base.setindex(Base.setindex(jr,0.0,_b4(4,1)),0.0,_b4(4,2)),0.0,_b4(4,3))
        for k in 1:16; JL[i,j,k]=jl[k]; JR[i,j,k]=jr[k]; end
    end
end

# Cell m of block line l and its direction: +1 when the line runs towards +j (foil columns; column w of a wake pair),
# -1 on column NI+1-w, which a wake line descends from the outer boundary to the cut.
@inline function _blk_line_cell(l,m,NI,NJ,NIW,NIF)
    l<=NIF && return (NIW+l,m,Int32(1))
    w=l-NIF; ip=NI+Int32(1)-w
    m<=NJ ? (ip,NJ+Int32(1)-m,Int32(-1)) : (w,m-NJ,Int32(1))
end

# The flux-Jacobian part of cell (i,j)'s block row: the diagonal block and the couplings to the cell below (south: j-1,
# or the partner across the cut) and above (north: j+1). The cell is the right state of its south face and the left
# state of its north face; the cut face is column w's.
@inline function _blk_row(JL,JR,i,j,NI,NJ,NIW,NIF)
    As=_b4zero(); An=_b4zero()
    if j>1 || NIW<i<=NIW+NIF
        B=_b4neg(_b4load(JR,i,j))
        j>1 && (As=_b4neg(_b4load(JL,i,j)))
    elseif i<=NIW
        B=_b4neg(_b4load(JR,i,1)); As=_b4neg(_b4load(JL,i,1))
    else
        w=NI+Int32(1)-i
        B=_b4load(JL,w,1); As=_b4load(JR,w,1)
    end
    B=_b4add(B,_b4load(JL,i,j+1))
    j<NJ && (An=_b4load(JR,i,j+1))
    (B,As,An)
end

# Block Thomas algorithm along each line, in place in X (rhs in, increment out); CP holds the modified super-diagonal
# blocks. d0 = V/dtf per cell. GS (NI x NJ x 2, or nothing) is the implicit phase source's Jacobian V dgamma/d(r, q), which
# enters the q row of the diagonal block with a minus sign (the source is subtracted in the right-hand side). One thread
# per line.
@kernel function _k_blk_line_solve!(X,CP,@Const(JL),@Const(JR),@Const(dtf),@Const(vol),GS,
                                    NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32)
    lg=@index(Global,Linear)
    l=Int32(lg)
    @inbounds if l<=NIF+NIW
        M=l<=NIF ? NJ : Int32(2)*NJ
        cprev=_b4zero(); xprev=(0.0,0.0,0.0,0.0)
        for m in Int32(1):M
            i,j,dir=_blk_line_cell(l,m,NI,NJ,NIW,NIF)
            B,As,An=_blk_row(JL,JR,i,j,NI,NJ,NIW,NIF)
            B=_b4add(B,_b4eye(Float64(vol[i,j])/Float64(dtf[i,j])))
            if GS!==nothing
                B=Base.setindex(B,B[_b4(4,1)]-GS[i,j,1],_b4(4,1)); B=Base.setindex(B,B[_b4(4,4)]-GS[i,j,2],_b4(4,4))
            end
            Ap=dir==Int32(1) ? As : An; Cn=dir==Int32(1) ? An : As
            Bp=m==Int32(1) ? B : _b4sub(B,_b4mul(Ap,cprev))
            Bi=_b4inv(Bp)
            cp=_b4mul(Bi,Cn)
            for k in 1:16; CP[i,j,k]=cp[k]; end
            rhs=(X[i,j,1],X[i,j,2],X[i,j,3],X[i,j,4])
            if m>Int32(1)
                a=_b4vec(Ap,xprev); rhs=(rhs[1]-a[1],rhs[2]-a[2],rhs[3]-a[3],rhs[4]-a[4])
            end
            xm=_b4vec(Bi,rhs)
            for k in 1:4; X[i,j,k]=xm[k]; end
            cprev=cp; xprev=xm
        end
        xnext=xprev
        for m in M-Int32(1):Int32(-1):Int32(1)
            i,j,_=_blk_line_cell(l,m,NI,NJ,NIW,NIF)
            t=_b4vec(_b4load(CP,i,j),xnext)
            xm=(X[i,j,1]-t[1],X[i,j,2]-t[2],X[i,j,3]-t[3],X[i,j,4]-t[4])
            for k in 1:4; X[i,j,k]=xm[k]; end
            xnext=xm
        end
    end
end

# Factorisation reuse (30 Sep 2026, execution-only): the line matrix is frozen within a step (JL, JR, the source block
# and dtf do not change between its stages), so stage 1 factorises it and stores, per cell, the inverse of the modified
# diagonal block (BI) and the sub-diagonal coupling in line order (AP); stages 2 and 3 only substitute. The arithmetic
# of a substitution is the factorising sweep's own (xm = Bi (rhs - Ap xprev); x_m -= C'_m x_{m+1}), so the result is
# bitwise that of refactorising.
@kernel function _k_blk_line_factor!(X,CP,BI,AP,@Const(JL),@Const(JR),@Const(dtf),@Const(vol),GS,
                                     NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32)
    lg=@index(Global,Linear)
    l=Int32(lg)
    @inbounds if l<=NIF+NIW
        M=l<=NIF ? NJ : Int32(2)*NJ
        cprev=_b4zero(); xprev=(0.0,0.0,0.0,0.0)
        for m in Int32(1):M
            i,j,dir=_blk_line_cell(l,m,NI,NJ,NIW,NIF)
            B,As,An=_blk_row(JL,JR,i,j,NI,NJ,NIW,NIF)
            B=_b4add(B,_b4eye(Float64(vol[i,j])/Float64(dtf[i,j])))
            if GS!==nothing
                B=Base.setindex(B,B[_b4(4,1)]-GS[i,j,1],_b4(4,1)); B=Base.setindex(B,B[_b4(4,4)]-GS[i,j,2],_b4(4,4))
            end
            Ap=dir==Int32(1) ? As : An; Cn=dir==Int32(1) ? An : As
            Bp=m==Int32(1) ? B : _b4sub(B,_b4mul(Ap,cprev))
            Bi=_b4inv(Bp)
            cp=_b4mul(Bi,Cn)
            for k in 1:16; CP[i,j,k]=cp[k]; BI[i,j,k]=Bi[k]; AP[i,j,k]=Ap[k]; end
            rhs=(X[i,j,1],X[i,j,2],X[i,j,3],X[i,j,4])
            if m>Int32(1)
                a=_b4vec(Ap,xprev); rhs=(rhs[1]-a[1],rhs[2]-a[2],rhs[3]-a[3],rhs[4]-a[4])
            end
            xm=_b4vec(Bi,rhs)
            for k in 1:4; X[i,j,k]=xm[k]; end
            cprev=cp; xprev=xm
        end
        xnext=xprev
        for m in M-Int32(1):Int32(-1):Int32(1)
            i,j,_=_blk_line_cell(l,m,NI,NJ,NIW,NIF)
            t=_b4vec(_b4load(CP,i,j),xnext)
            xm=(X[i,j,1]-t[1],X[i,j,2]-t[2],X[i,j,3]-t[3],X[i,j,4]-t[4])
            for k in 1:4; X[i,j,k]=xm[k]; end
            xnext=xm
        end
    end
end
@kernel function _k_blk_line_substitute!(X,@Const(CP),@Const(BI),@Const(AP),NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32)
    lg=@index(Global,Linear)
    l=Int32(lg)
    @inbounds if l<=NIF+NIW
        M=l<=NIF ? NJ : Int32(2)*NJ
        xprev=(0.0,0.0,0.0,0.0)
        for m in Int32(1):M
            i,j,_=_blk_line_cell(l,m,NI,NJ,NIW,NIF)
            rhs=(X[i,j,1],X[i,j,2],X[i,j,3],X[i,j,4])
            if m>Int32(1)
                a=_b4vec(_b4load(AP,i,j),xprev); rhs=(rhs[1]-a[1],rhs[2]-a[2],rhs[3]-a[3],rhs[4]-a[4])
            end
            xm=_b4vec(_b4load(BI,i,j),rhs)
            for k in 1:4; X[i,j,k]=xm[k]; end
            xprev=xm
        end
        xnext=xprev
        for m in M-Int32(1):Int32(-1):Int32(1)
            i,j,_=_blk_line_cell(l,m,NI,NJ,NIW,NIF)
            t=_b4vec(_b4load(CP,i,j),xnext)
            xm=(X[i,j,1]-t[1],X[i,j,2]-t[2],X[i,j,3]-t[3],X[i,j,4]-t[4])
            for k in 1:4; X[i,j,k]=xm[k]; end
            xnext=xm
        end
    end
end

# The implicit face corrections dF = JL D_L + JR D_R of the solved increments, per j face in the production orientation
# (the wall face on its cell only, the outer face on its inner cell, the cut face of column NI+1-w column w's negated),
# and the increment rewritten as a flux difference, D = dt/V (rhs - (dF north - dF south)): the physical-time step's
# conservation is then that of the explicit stage whatever the accuracy of the line solve (exact solve: the same D).
@kernel function _k_blk_face_corr!(dF,@Const(X),@Const(JL),@Const(JR),NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ+1
        xs(ii,jj)=(X[ii,jj,1],X[ii,jj,2],X[ii,jj,3],X[ii,jj,4])
        local f::NTuple{4,Float64}
        if j==1 && NIW<i<=NIW+NIF
            f=_b4vec(_b4load(JR,i,1),xs(i,1))
        elseif j==1
            w=i<=NIW ? i : NI+Int32(1)-i; ip=NI+Int32(1)-w
            a=_b4vec(_b4load(JL,w,1),xs(ip,1)); b=_b4vec(_b4load(JR,w,1),xs(w,1))
            sg=i<=NIW ? 1.0 : -1.0
            f=(sg*(a[1]+b[1]),sg*(a[2]+b[2]),sg*(a[3]+b[3]),sg*(a[4]+b[4]))
        elseif j==NJ+1
            f=_b4vec(_b4load(JL,i,NJ+1),xs(i,NJ))
        else
            a=_b4vec(_b4load(JL,i,j),xs(i,j-1)); b=_b4vec(_b4load(JR,i,j),xs(i,j))
            f=(a[1]+b[1],a[2]+b[2],a[3]+b[3],a[4]+b[4])
        end
        for k in 1:4; dF[i,j,k]=f[k]; end
    end
end
@kernel function _k_blk_flux_increment!(X,@Const(R),@Const(dF),@Const(dtf),@Const(vol),GS,NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        fac=Float64(dtf[i,j])/Float64(vol[i,j])
        src=GS===nothing ? 0.0 : GS[i,j,1]*X[i,j,1]+GS[i,j,2]*X[i,j,4]   # the implicit source's linear part, local
        for k in 1:3; X[i,j,k]=fac*(R[i,j,k]-(dF[i,j+1,k]-dF[i,j,k])); end
        X[i,j,4]=fac*(R[i,j,4]-(dF[i,j+1,4]-dF[i,j,4])+src)
    end
end
# rhs in L.X -> the conservative increment in L.X (L.R keeps the right-hand side, L.dF the face corrections)
function _blk_line_solve_conservative!(c,L;GS=nothing,factor::Bool=true)
    s=c.solver;ni,nj,niw,nif=Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF)
    copyto!(L.R,L.X)
    _blk_line_solve!(c,L;GS,factor)
    _k_blk_face_corr!(s.backend)(L.dF,L.X,L.JL,L.JR,ni,nj,niw,nif;ndrange=(s.NI,s.NJ+1))
    _k_blk_flux_increment!(s.backend)(L.X,L.R,L.dF,s.dtf,s.vol,GS,ni,nj;ndrange=(s.NI,s.NJ))
    nothing
end

# The turbulence scalars' line coefficients in the form _k_line_solve! takes (row: (d0 + lamS + lamN) D - lamS D_south
# - lamN D_north = rhs): the couplings are the upwind derivatives of the j advection by the carrier mass flux plus the
# diffusion conductances, per unit density of the neighbour; d0 carries the rest of the diagonal (V/dt, V rate and this
# cell's own advection/diffusion derivatives) less the two couplings.
# Strip topology (stage 4, 7 Oct 2026; plan section 5 item 4, critique A 3 and 5): `top_interface` = 1 on the strip block of a
# strip pair, whose j = NJ+1 face is the upper interface: its top row couples north to the upper leg's row-1 cell, `rn` the
# exchange-written ghost copy of that cell's density (every existing launch passes 0, a branch on a constant: the outer
# boundary takes no north coupling, as before). `ctx`: a strip pair's block context (nothing without one): the interface
# conductances through the strip twins of the SST diffusion (_sst_j_diff_strip: the other block's centroid from the static
# table, its sigma_k, sigma_w from sig_nb), as the explicit SST flux kernels take them; ctx === nothing compiles to today's.
@kernel function _k_blk_sst_coeffs!(lamSk,lamNk,lamSw,lamNw,d0k,d0w,@Const(dtf),@Const(Fmj),@Const(rate),
                                    @Const(rho),@Const(nut),@Const(sigk),@Const(sigw),@Const(xc),@Const(yc),
                                    @Const(Sjx),@Const(Sjy),@Const(vol),nu::Float32,wallmode::Int32,
                                    NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32,wf::Int32,top_interface::Int32,ctx)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        ia=i+g;ja=j+g;r=rho[ia,ja]
        sk,sw=ctx===nothing ? _sst_j_diff(rho,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,nu,wallmode,i,j,NI,NJ,NIW,NIF,g,wf) :
                              _sst_j_diff_strip(rho,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,nu,wallmode,i,j,NI,NJ,NIW,NIF,g,wf,ctx)
        nk,nw=ctx===nothing ? _sst_j_diff(rho,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,nu,wallmode,i,j+1,NI,NJ,NIW,NIF,g,wf) :
                              _sst_j_diff_strip(rho,nut,sigk,sigw,xc,yc,Sjx,Sjy,vol,nu,wallmode,i,j+1,NI,NJ,NIW,NIF,g,wf,ctx)
        fs=Fmj[i,j];fn=Fmj[i,j+1]
        wall=j==1 && NIW<i<=NIW+NIF
        rs=rho[ia,ja-1];rn=rho[ia,ja+1]   # below: the cell, or across the cut the partner's ghost copy
        lsk=wall ? 0.0f0 : (max(fs,0.0f0)+sk)/rs; lsw=wall ? 0.0f0 : (max(fs,0.0f0)+sw)/rs
        north=j<NJ || top_interface==Int32(1)   # the strip's top row: the upper interface (the gate lifted for it alone)
        lnk=north ? (nk-min(fn,0.0f0))/rn : 0.0f0; lnw=north ? (nw-min(fn,0.0f0))/rn : 0.0f0
        dk=(sk-min(fs,0.0f0)+max(fn,0.0f0)+nk)/r
        dw=(sw-min(fs,0.0f0)+max(fn,0.0f0)+nw)/r
        dt=dtf[i,j]; base=(dt>0.0f0 ? vol[i,j]/dt : Inf32)+vol[i,j]*rate[i,j]
        lamSk[i,j]=lsk;lamNk[i,j]=lnk;lamSw[i,j]=lsw;lamNw[i,j]=lnw
        d0k[i,j]=base+dk-lsk-lnk; d0w[i,j]=base+dw-lsw-lnw
    end
end

# The Jacobians of the current published state (the stage-1 state of an attempt). `c`: a cache, or the strip block of a strip
# pair (its context then gives its two interface faces their one linearisation; plan section 5 item 2); a cache without a
# strip passes ctx = nothing, today's kernel.
function _blk_jacobians!(c,L,wallmode)
    s=c.solver
    a2=s.ff_mix==Int32(1) ? s.ff_a2m : s.am_ref*s.am_ref
    minf2=Float64(Float32((s.uxi*s.uxi+s.uyi*s.uyi)/max(a2,1.0f-6)))
    _k_blk_face_jac!(s.backend)(L.JL,L.JR,c.conserved.U,s.rho,s.ux,s.uy,c.tem.alpha,c.pressure,s.am,s.nut,s.vol,s.xc,s.yc,
        s.Sjx,s.Sjy,Float64.(s.outlet_eos),minf2,s.nu_lam,wallmode,Int32(s.sst),
        Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF),Int32(s.g),_strip_ctx(c);ndrange=(s.NI,s.NJ+1))
    nothing
end
# top_interface 0: the outer boundary takes no north coupling (the strip block's launch, with 1, is _blk_sst_coeffs_strip! in
# cmesh_species_strip.jl); the C block of a strip pair passes its context for its leg columns' interface conductances
function _blk_sst_coeffs!(c,L,wallmode)
    s=c.solver;a=c.turbulence
    _k_blk_sst_coeffs!(s.backend)(L.lamSk,L.lamNk,L.lamSw,L.lamNw,L.d0k,L.d0w,s.dtf,s.Fj_r,a.rate,s.rho,s.nut,
        a.sigk,a.sigw,s.xc,s.yc,s.Sjx,s.Sjy,s.vol,s.nu_lam,wallmode,Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF),
        Int32(s.g),_wall_function(s),Int32(0),_strip_ctx(c);ndrange=(s.NI,s.NJ))
    nothing
end
function _blk_line_solve!(c,L;GS=nothing,factor::Bool=true)
    s=c.solver
    if factor   # stage 1: factorise (and solve); stages 2-3 of the same step: substitute with the stored factors
        _k_blk_line_factor!(s.backend)(L.X,L.CP,L.BI,L.AP,L.JL,L.JR,s.dtf,s.vol,GS,Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF);
            ndrange=s.NIF+s.NIW)
    else
        _k_blk_line_substitute!(s.backend)(L.X,L.CP,L.BI,L.AP,Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF);
            ndrange=s.NIF+s.NIW)
    end
    nothing
end

# The implicit phase source's Jacobian (W3a, second stage): GS[.,.,2] = V dgamma/dq, a one-sided difference of relative
# step 1e-6 through the production source _phase_source (double precision; the EOS pressure feedback on the source's own
# driving force, the stiff part of its clock, is thereby included). The source changes q only, so it enters the q row of
# the block. Its rho derivative, GS[.,.,1], is kept ZERO for the same reason the vapour row keeps only its own derivative
# (see _k_blk_face_jac!): with it the q row would couple to the rho increment and lose its positivity. The kernel slot is
# kept so the solve's source block remains general. A nonfinite difference leaves its entry zero.
@kernel function _k_blk_source_jac!(GS,@Const(U),@Const(vol),ep::NTuple{7,Float64},kp::Float64,kd::Float64,bw::Int32,
                                    NI::Int32,NJ::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI && j<=NJ
        r=U[i,j,1];q=U[i,j,4];V=Float64(vol[i,j])
        g0,_=_phase_source(r,q,ep,kp,kd,bw)
        hq=q+_BLK_REL*r<=r ? _BLK_REL*r : -_BLK_REL*r
        gq,_=_phase_source(r,q+hq,ep,kp,kd,bw)
        b=V*(Float64(gq)-Float64(g0))/hq
        GS[i,j,1]=0.0
        GS[i,j,2]=isfinite(b) ? b : 0.0
    end
end
function _blk_source_jac!(c,L)
    s=c.solver;t=c.tem
    _k_blk_source_jac!(s.backend)(L.GS,c.conserved.U,s.vol,Float64.(s.outlet_eos),Float64(t.kprod),Float64(t.kdest),
        Int32(c.source_beta),Int32(s.NI),Int32(s.NJ);ndrange=(s.NI,s.NJ))
    nothing
end

function _block_workspace(s)
    base=_line_workspace(s)
    z64(dims...)=fill!(similar(s.dtf,Float64,dims...),0.0)
    merge(base,(;JL=z64(s.NI,s.NJ+1,16),JR=z64(s.NI,s.NJ+1,16),CP=z64(s.NI,s.NJ,16),
                 R=z64(s.NI,s.NJ,4),dF=z64(s.NI,s.NJ+1,4),GS=z64(s.NI,s.NJ,2),BI=z64(s.NI,s.NJ,16),AP=z64(s.NI,s.NJ,16)))
end
