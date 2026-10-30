# ==============================================================================
# Particles-on-Demand — 2D THERMAL NESTED-BOX AMR (D3Q27 philosophy, √θ-gauge sibling) [GPU]
# ==============================================================================
# The 2D-thermal port of pond_fv3d_nested.jl. A coarse BASE grid L1 (dx=2) covers the whole domain; a
# fine BOX L2 (dx=1) wraps the airfoil, nested inside L1. This is what finally gives the near-wall
# resolution the coarse UNIFORM grid lacked (see transonic-aircraft-loads-goal): the fine cells sit only
# where the boundary layer + suction pocket + shock live, so for a given VRAM the wall is resolved 2×
# (or 4× with a 3rd level) while the coarse base cheaply renders the far-field.
#
# Both L1 and L2 are ordinary `PonD2DT` grids stepped by the SAME `pond_fv2dT_advance!` (SSP-RK3 + WENO5-Z
# + SVV + molecular-ν + SA/WMLES). L2 is built with `nested=true`, so its `applyBC!` PINS a prolongation-
# filled ghost halo (the `h*` buffers) into the ng-border each RK stage instead of imposing the domain
# inlet/outlet/farfield BCs. The coarse base supplies that halo by prolongation once per coarse step.
#
# Coupling per step (Berger-Oliger, 2 levels, same Δt — the temporal-refluxing sub-cycle is the follow-up):
#   (1) PROLONG  L1^n → L2 halo (bicubic Catmull-Rom, C¹) — fills L2's ng-border BC from the coarse state;
#   (2) PIN      the halo into L2's live border (so stage-1 flux reads it);
#   (3) STEP L2  over its interior (advance!, nested — re-pins the halo each RK stage) + penalize + wall;
#   (4) STEP L1  over the whole domain (advance!, normal domain BCs) + penalize;
#   (5) RESTRICT L2 interior → the L1 cells it covers — averages the CONSERVED variables (ρ, ρu, ρv, ρE)
#       over each 2² fine block ⇒ mass/momentum/energy consistent, so a uniform stream is preserved exactly.
# The box occupies L1 cells [ci0:ci0+nbx-1]×[cj0:cj0+nby-1]; L2 interior index = ng + 2·(box-local−1) + {1,2}.
# ==============================================================================

using KernelAbstractions

# Catmull-Rom cubic weights (C¹) for the 4-point stencil [base-1,base,base+1,base+2] at fraction t∈[0,1).
@inline function _cr4n(t::Float32)
    t2=t*t; t3=t2*t
    return (-0.5f0*t3+t2-0.5f0*t, 1.5f0*t3-2.5f0*t2+1.0f0, -1.5f0*t3+2.0f0*t2+0.5f0*t, 0.5f0*t3-0.5f0*t2)
end
@inline _clin(x,n)= x<1 ? 1 : (x>n ? n : x)

# PROLONG: fill L2's ng-border ghost halo from L1 by BICUBIC (Catmull-Rom, C¹) refinement, into the h* buffers.
# A cell-centred fine sub-cell sits at L1-continuous coordinate X = ci0 − ¼ + ½·(fine-local-index): the two fine
# sub-cells of an L1 cell are at ∓¼ of its centre. We interpolate the PRIMITIVES (ρ,u,v,T) with a 4×4 cubic
# stencil (the halo is a boundary condition, not a conserved region, so pointwise high-order is exact-safe).
@kernel function prolong_l1l2_2dT!(hrho,hux,huy,hT,
        @Const(l1r),@Const(l1ux),@Const(l1uy),@Const(l1T),
        ci0::Int32,cj0::Int32, g::Int32, N1X::Int32,N1Y::Int32, N2X::Int32,N2Y::Int32)
    i2,j2=@index(Global,NTuple); n2x=Int(N2X);n2y=Int(N2Y);gg=Int(g); n1x=Int(N1X);n1y=Int(N1Y)
    @inbounds begin
        isghost = (i2<=gg)||(i2>n2x-gg)||(j2<=gg)||(j2>n2y-gg)
        if isghost
            Xf=Float32(ci0)-0.25f0+0.5f0*Float32(i2-gg-1); Yf=Float32(cj0)-0.25f0+0.5f0*Float32(j2-gg-1)
            bx=floor(Int,Xf); by=floor(Int,Yf)
            wx=_cr4n(Xf-Float32(bx)); wy=_cr4n(Yf-Float32(by))
            sr=0.0f0;sx=0.0f0;sy=0.0f0;sT=0.0f0
            for dj in 0:3
                cj=_clin(by-1+dj,n1y); wj=wy[dj+1]
                for di in 0:3
                    ci=_clin(bx-1+di,n1x); ww=wj*wx[di+1]
                    sr+=ww*l1r[ci,cj]; sx+=ww*l1ux[ci,cj]; sy+=ww*l1uy[ci,cj]; sT+=ww*l1T[ci,cj]
                end
            end
            hrho[i2,j2]=sr; hux[i2,j2]=sx; huy[i2,j2]=sy; hT[i2,j2]=(sT<1.0f-3 ? 1.0f-3 : sT)
        end
    end
end

# PROLONG k/ω (REVIEW FIX 2026-07-09): 2-field sibling of the flow prolongation for NESTED+SST grids —
# fills the child's hk/hom halo from the parent's tk/tom so the SST border pin carries the REAL incoming
# turbulence state (the freestream pin laminarized a TE patch's mid-BL inflow). No floors here (k can be
# ≪1e-3); the consumer floors/NaN-guards. Bicubic like the flow halo; zeroth-order in time (closure field).
@kernel function prolong_kw_l1l2_2dT!(hk,hom, @Const(l1k),@Const(l1w),
        ci0::Int32,cj0::Int32, g::Int32, N1X::Int32,N1Y::Int32, N2X::Int32,N2Y::Int32)
    i2,j2=@index(Global,NTuple); n2x=Int(N2X);n2y=Int(N2Y);gg=Int(g); n1x=Int(N1X);n1y=Int(N1Y)
    @inbounds begin
        isghost = (i2<=gg)||(i2>n2x-gg)||(j2<=gg)||(j2>n2y-gg)
        if isghost
            Xf=Float32(ci0)-0.25f0+0.5f0*Float32(i2-gg-1); Yf=Float32(cj0)-0.25f0+0.5f0*Float32(j2-gg-1)
            bx=floor(Int,Xf); by=floor(Int,Yf)
            wx=_cr4n(Xf-Float32(bx)); wy=_cr4n(Yf-Float32(by))
            sk=0.0f0;sw=0.0f0
            for dj in 0:3
                cj=_clin(by-1+dj,n1y); wj=wy[dj+1]
                for di in 0:3
                    ci=_clin(bx-1+di,n1x); ww=wj*wx[di+1]
                    sk+=ww*l1k[ci,cj]; sw+=ww*l1w[ci,cj]
                end
            end
            hk[i2,j2]=sk; hom[i2,j2]=sw
        end
    end
end

# RESTRICT: overwrite the L1 box cells with the conserved-variable average of each covered 2² L2 block.
@kernel function restrict_l2l1_2dT!(l1r,l1ux,l1uy,l1T,
        @Const(l2r),@Const(l2ux),@Const(l2uy),@Const(l2T),
        ci0::Int32,cj0::Int32, g::Int32, eint::Float32, NBX::Int32,NBY::Int32)
    a,b=@index(Global,NTuple); gg=Int(g)
    @inbounds begin
        ci=Int(ci0)+a-1; cj=Int(cj0)+b-1
        i0=gg+2*(a-1); j0=gg+2*(b-1)
        sr=0.0f0;smx=0.0f0;smy=0.0f0;sE=0.0f0
        for dj in 1:2, di in 1:2
            i2=i0+di; j2=j0+dj
            r=l2r[i2,j2]; ux=l2ux[i2,j2]; uy=l2uy[i2,j2]; T=l2T[i2,j2]
            sr+=r; smx+=r*ux; smy+=r*uy; sE+=r*(eint*T+0.5f0*(ux*ux+uy*uy))
        end
        r1=sr*0.25f0; ux1=smx/sr; uy1=smy/sr
        T1=(sE/sr-0.5f0*(ux1*ux1+uy1*uy1))/eint
        l1r[ci,cj]=r1; l1ux[ci,cj]=ux1; l1uy[ci,cj]=uy1; l1T[ci,cj]=(T1<1.0f-3 ? 1.0f-3 : T1)
    end
end

# ==============================================================================
# FAS MULTIGRID transfer kernels (2026-07-14, docs/2026-07-14_FAS_phase1_plan.md)
# τ-corrected coarse smoothing on the rectangular sub-chain: the parent relaxes R_H(U_H)=τ with
# τ = R_H(I·U_h) − I·(R_h(U_h)−τ_h), and the correction dU = U_H − Ū (Ū = snapshot at restriction time)
# is prolonged ADDITIVELY into the fine INTERIOR. All ops in CONSERVED variables (forced by dc_* being
# conserved rates). Fixed-point invariance: at a converged fine state R−τ≡0 cascades ⇒ τ_H cancels the
# coarse defect exactly ⇒ dU≡0 ⇒ PV_FAS on converges to the SAME state as off.
# ==============================================================================

# FAS: restrict the (τ-corrected) fine residual RATES to the parent's covered cells — plain 2×2 mean of
# the 4 already-CONSERVED rate components (no eint, no prim reconstruction — unlike restrict_l2l1_2dT!),
# scaled by r=dx_H/dx_h (PV_FAS_R; affects convergence RATE only, never the fixed point). STAGES I·R into
# the parent's fas_tau_* (finalized to τ by _fas_tau_finalize!).
@kernel function _fas_restrict_rate!(pt_r, pt_mx, pt_my, pt_e,
        @Const(cr), @Const(cmx), @Const(cmy), @Const(ce),
        ci0::Int32, cj0::Int32, g::Int32, r::Float32)
    a,b=@index(Global,NTuple); gg=Int(g)
    @inbounds begin
        ci=Int(ci0)+a-1; cj=Int(cj0)+b-1
        i0=gg+2*(a-1); j0=gg+2*(b-1)
        sr=0f0;smx=0f0;smy=0f0;se=0f0
        for dj in 1:2, di in 1:2
            i2=i0+di; j2=j0+dj
            sr+=cr[i2,j2]; smx+=cmx[i2,j2]; smy+=cmy[i2,j2]; se+=ce[i2,j2]
        end
        f=0.25f0*r
        pt_r[ci,cj]=sr*f; pt_mx[ci,cj]=smx*f; pt_my[ci,cj]=smy*f; pt_e[ci,cj]=se*f
    end
end

# FAS: snapshot the parent's live primitives → CONSERVED baseline Ū over the covered box EXPANDED by a
# 2-cell rim (clamped). The rim matters: the bicubic correction stencil straddles the box edge, and an
# unsnapshotted rim cell would make dU = cons(U_H) − 0 = O(1) garbage there. On the rim Ū = the parent's
# own pre-coarse-smooth state (the standard FAS baseline: dU there = the parent's own smoothing change).
# MUST run immediately after _tree_restrict! (Ū = I·U_h on covered), BEFORE any coarse smoothing.
@kernel function _fas_snapshot_ub!(ub_r, ub_mx, ub_my, ub_e,
        @Const(rho), @Const(ux), @Const(uy), @Const(T),
        eint::Float32, ci0::Int32, cj0::Int32, N1X::Int32, N1Y::Int32)
    a,b=@index(Global,NTuple); n1x=Int(N1X); n1y=Int(N1Y)
    @inbounds begin
        ci=Int(ci0)-2+a-1; cj=Int(cj0)-2+b-1                 # expanded box: a∈1..nbx+4, b∈1..nby+4
        ci = ci<1 ? 1 : (ci>n1x ? n1x : ci); cj = cj<1 ? 1 : (cj>n1y ? n1y : cj)
        rr=rho[ci,cj]; uxx=ux[ci,cj]; uyy=uy[ci,cj]; tt=T[ci,cj]
        ub_r[ci,cj]=rr; ub_mx[ci,cj]=rr*uxx; ub_my[ci,cj]=rr*uyy
        ub_e[ci,cj]=rr*(eint*tt+0.5f0*(uxx*uxx+uyy*uyy))
    end
end

# FAS: finalize τ = R_H(I·U_h) − staged(I·R_h) on the covered box (parent dc_* holds R_H(I·U_h) from
# pond_fv2dT_residual! run on the freshly-restricted state; fas_tau_* holds the staged I·R_h). Rect-chain
# levels are all-fluid (the wall lives only on the cut-cell leaf, excluded from FAS) ⇒ no solid mask
# needed; τ outside the covered box is never written and stays 0 (boxes are static).
@kernel function _fas_tau_finalize!(tr, tmx, tmy, te,
        @Const(dc_r), @Const(dc_mx), @Const(dc_my), @Const(dc_e),
        ci0::Int32, cj0::Int32)
    a,b=@index(Global,NTuple)
    @inbounds begin
        ci=Int(ci0)+a-1; cj=Int(cj0)+b-1
        tr[ci,cj] =dc_r[ci,cj] -tr[ci,cj];  tmx[ci,cj]=dc_mx[ci,cj]-tmx[ci,cj]
        tmy[ci,cj]=dc_my[ci,cj]-tmy[ci,cj]; te[ci,cj] =dc_e[ci,cj] -te[ci,cj]
    end
end

# FMG GRID SEQUENCING (2026-07-14, brute-speed program): initialize a newly-ACTIVATED level's ENTIRE grid
# (interior + halo) by bicubic Catmull-Rom prolongation of the parent's primitives — the FMG cold-start
# transfer (clone of prolong_l1l2_2dT! WITHOUT the isghost guard, writing the live state directly). The
# slow circulation mode is traversed on the cheap coarse chain; each activation starts the new level at
# the interpolated converged-coarse state instead of freestream. One-shot per activation (host-driven).
@kernel function _fmg_prolong_interior!(frho, fux, fuy, fT,
        @Const(pr), @Const(pux), @Const(puy), @Const(pT),
        ci0::Int32, cj0::Int32, g::Int32, N1X::Int32, N1Y::Int32)
    i2,j2=@index(Global,NTuple); gg=Int(g); n1x=Int(N1X); n1y=Int(N1Y)
    @inbounds begin
        Xf=Float32(ci0)-0.25f0+0.5f0*Float32(i2-gg-1); Yf=Float32(cj0)-0.25f0+0.5f0*Float32(j2-gg-1)
        bx=floor(Int,Xf); by=floor(Int,Yf)
        wx=_cr4n(Xf-Float32(bx)); wy=_cr4n(Yf-Float32(by))
        sr=0.0f0;sx=0.0f0;sy=0.0f0;sT=0.0f0
        for dj in 0:3
            cj=_clin(by-1+dj,n1y); wj=wy[dj+1]
            for di in 0:3
                ci=_clin(bx-1+di,n1x); ww=wj*wx[di+1]
                sr+=ww*pr[ci,cj]; sx+=ww*pux[ci,cj]; sy+=ww*puy[ci,cj]; sT+=ww*pT[ci,cj]
            end
        end
        sr = sr<1.0f-3 ? 1.0f-3 : sr
        frho[i2,j2]=sr; fux[i2,j2]=sx; fuy[i2,j2]=sy; fT[i2,j2]=(sT<1.0f-3 ? 1.0f-3 : sT)
    end
end

# FAS: THE genuinely new transfer — prolong the coarse CONSERVED correction dU = cons(U_H) − Ū additively
# into the fine INTERIOR (the existing prolong fills only the ghost halo and overwrites; the halo pin and
# this write hit disjoint index sets). Bicubic Catmull-Rom on dU (smooth, →0 at convergence), then add to
# the fine conserved state and re-primitive with the exact floors of rect_update_lts!.
@kernel function _fas_prolong_add_interior!(frho, fux, fuy, fT,
        @Const(pr), @Const(pux), @Const(puy), @Const(pT),          # coarse CURRENT primitives (post coarse-smooth)
        @Const(ub_r), @Const(ub_mx), @Const(ub_my), @Const(ub_e),  # coarse baseline Ū (conserved, incl. 2-cell rim)
        ci0::Int32, cj0::Int32, g::Int32, eint::Float32,
        N1X::Int32, N1Y::Int32, N2X::Int32, N2Y::Int32)
    i2,j2=@index(Global,NTuple); n2x=Int(N2X);n2y=Int(N2Y);gg=Int(g); n1x=Int(N1X);n1y=Int(N1Y)
    @inbounds begin
        isint = (i2>gg)&&(i2<=n2x-gg)&&(j2>gg)&&(j2<=n2y-gg)
        if isint
            Xf=Float32(ci0)-0.25f0+0.5f0*Float32(i2-gg-1); Yf=Float32(cj0)-0.25f0+0.5f0*Float32(j2-gg-1)
            bx=floor(Int,Xf); by=floor(Int,Yf)
            wx=_cr4n(Xf-Float32(bx)); wy=_cr4n(Yf-Float32(by))
            dr=0f0; dmx=0f0; dmy=0f0; de=0f0
            for dj in 0:3
                cj=_clin(by-1+dj,n1y); wj=wy[dj+1]
                for di in 0:3
                    ci=_clin(bx-1+di,n1x); ww=wj*wx[di+1]
                    cr=pr[ci,cj]; cx=pux[ci,cj]; cy=puy[ci,cj]; ct=pT[ci,cj]
                    Ec=cr*(eint*ct+0.5f0*(cx*cx+cy*cy))
                    dr +=ww*(cr    - ub_r[ci,cj]);  dmx+=ww*(cr*cx - ub_mx[ci,cj])
                    dmy+=ww*(cr*cy - ub_my[ci,cj]); de +=ww*(Ec    - ub_e[ci,cj])
                end
            end
            fr=frho[i2,j2]; fx=fux[i2,j2]; fy=fuy[i2,j2]; ft=fT[i2,j2]
            Efc=fr*(eint*ft+0.5f0*(fx*fx+fy*fy))
            r = fr + dr;  mx = fr*fx + dmx;  my = fr*fy + dmy;  e = Efc + de
            r = r<1.0f-3 ? 1.0f-3 : r                              # floors ≡ rect_update_lts!
            u2=mx/r; v2=my/r; sp=sqrt(u2*u2+v2*v2); if sp>3.5f0; f=3.5f0/sp; u2*=f; v2*=f; end
            T2=(e/r-0.5f0*(u2*u2+v2*v2))/eint; T2 = T2<1.0f-3 ? 1.0f-3 : T2
            frho[i2,j2]=r; fux[i2,j2]=u2; fuy[i2,j2]=v2; fT[i2,j2]=T2
        end
    end
end

# N-LEVEL nesting (G7 recursion, 2026-07-07). L[1]=coarsest base (domain BCs) … L[end]=finest box.
# box[l]=(ci0,cj0,nbx,nby) places level l+1 inside level l (same convention as the 2-level step).
# Coupling per coupled step, pairwise identical to the 2-level scheme:
#   (1) prolong+pin every child halo from its PARENT's t^n state (coarsest-first);
#   (2) advance every level one step of ITS OWN dt (finest-first, matching the 2-level order) — the
#       caller must build the level dts so all levels advance the SAME physical time:
#       dt_l = dt_fine·(dx_fine/dx_l) in each grid's own cell units (the PV_L1SYNC convention), and
#       nu_mol_l = nu_mol_fine·(dx_fine/dx_l)²;
#   (3) conserved restriction child→parent, finest-first (so corrections cascade down to the base).
function pond_fv2dT_nestedN_step!(L::Vector{<:PonD2DT}, box::Vector{NTuple{4,Int}}, t::Int)
    n=length(L); @assert length(box)==n-1
    for l in 1:n-1
        ci0,cj0,nbx,nby = box[l]; C=L[l+1]; P=L[l]; g=Int32(C.ng); nd2=(Int(C.NX),Int(C.NY))
        prolong_l1l2_2dT!(C.backend)(C.hrho,C.hux,C.huy,C.hT, P.rho,P.ux,P.uy,P.T,
            Int32(ci0),Int32(cj0), g, Int32(P.NX),Int32(P.NY), Int32(C.NX),Int32(C.NY); ndrange=nd2)
        KernelAbstractions.synchronize(C.backend)
        _pin_halo2T!(C.backend)(C.rho,C.ux,C.uy,C.T, C.hrho,C.hux,C.huy,C.hT, Int32(C.NX),Int32(C.NY), g; ndrange=nd2)
        KernelAbstractions.synchronize(C.backend)
    end
    for l in n:-1:1
        pond_fv2dT_advance!(L[l], t)
    end
    for l in n-1:-1:1
        ci0,cj0,nbx,nby = box[l]; C=L[l+1]; P=L[l]; g=Int32(C.ng)
        eint = 0.5f0*(3.0f0+P.drot)*P.cs2
        restrict_l2l1_2dT!(P.backend)(P.rho,P.ux,P.uy,P.T, C.rho,C.ux,C.uy,C.T,
            Int32(ci0),Int32(cj0), g, eint, Int32(nbx),Int32(nby); ndrange=(nbx,nby))
        KernelAbstractions.synchronize(P.backend)
    end
    return nothing
end

# ==============================================================================
# TREE AMR (chain→tree generalization, 2026-07-09) — HUGGING PATCHES at LE/TE
# ==============================================================================
# Generalizes the linear chain (one child per level) to a TREE: any node may carry multiple
# NON-OVERLAPPING children (siblings couple only through their parent — classic Berger-Oliger patch
# AMR; the LE and TE patches are far apart, so no sibling exchange exists by construction).
#
# TEMPORAL SUB-CYCLING (the load-bearing design point): the shared-physical-dt convention of the chain
# (all levels advance the same dt, coarse levels at low CFL) CANNOT extend to a patch at S=dx_fine/2 —
# the patch's lattice CFL would reach ~1.0 (c_max·a: GDT is sized for CFL 0.5 at S=1). Patches therefore
# take substeps=2 Berger-Oliger sub-steps per coupled step, each at HALF the physical dt (patch driver
# built with dt=GDT, i.e. the same 'a' as the finest full level ⇒ CFL 0.5), with the halo LINEARLY
# TIME-INTERPOLATED between the parent's pre-step (t^n → the h* buffers, phase 1) and post-step
# (t^{n+1} → the h2 side buffers, phase 4) states: substep A pins ½(h+h2) [=t^{n+½}], substep B pins h2.
# The blend is written INTO the h* buffers so advance!'s per-RK-stage re-pin reads it unchanged.
#
# Ordering per coupled step:
#   phase 1: prolong+pin every child halo from its parent's t^n state (all states pre-step ⇒ any order);
#   phase 2: advance substeps==1 nodes deepest-first (byte-identical to the chain's n:-1:1);
#   phase 3: restrict substeps==1 children into parents deepest-first (byte-identical to the chain);
#   phase 4: for each substeps==2 LEAF: prolong parent^{n+1}→h2, blend-pin, substep A, pin h2,
#            substep B, restrict into parent. (The patch's correction reaches the grandparent on the
#            NEXT step's phase-3 cascade — one-step staleness inside a region the parent restriction
#            overwrites anyway; accepted and documented.)
# With no substeps==2 nodes and parent[i]=i-1 this reproduces pond_fv2dT_nestedN_step! EXACTLY.
mutable struct PondTreeNode
    parent::Int              # 0 for the root
    pbox::NTuple{4,Int}      # (ci0,cj0,nbx,nby) in the PARENT's index space ((0,0,0,0) for root)
    substeps::Int            # 1 = shared-dt chain level; 2 = Berger-Oliger sub-cycled patch (leaf only)
    h2::Any                  # substeps==2 only: NamedTuple (rho,ux,uy,T) of device halo buffers @ t^{n+1}
end
PondTreeNode(parent,pbox,substeps)=PondTreeNode(parent,pbox,substeps,nothing)

function _tree_depths(T::Vector{PondTreeNode})
    n=length(T); d=zeros(Int,n)
    for i in 1:n; p=T[i].parent; d[i]= p==0 ? 0 : (@assert p<i "parents must precede children"; d[p]+1); end
    d
end
@inline function _tree_prolong_pin!(C::PonD2DT, P::PonD2DT, pbox; pin::Bool=true, dst=nothing)
    ci0,cj0,_,_=pbox; g=Int32(C.ng); nd=(Int(C.NX),Int(C.NY))
    hr,hx,hy,hT = dst===nothing ? (C.hrho,C.hux,C.huy,C.hT) : (dst.rho,dst.ux,dst.uy,dst.T)
    prolong_l1l2_2dT!(C.backend)(hr,hx,hy,hT, P.rho,P.ux,P.uy,P.T,
        Int32(ci0),Int32(cj0), g, Int32(P.NX),Int32(P.NY), Int32(C.NX),Int32(C.NY); ndrange=nd)
    _psync(C.backend)
    if pin
        _pin_halo2T!(C.backend)(C.rho,C.ux,C.uy,C.T, hr,hx,hy,hT, Int32(C.NX),Int32(C.NY), g; ndrange=nd)
        _psync(C.backend)
    end
end
@inline function _tree_restrict!(C::PonD2DT, P::PonD2DT, pbox)
    ci0,cj0,nbx,nby=pbox; g=Int32(C.ng)
    eint=0.5f0*(3.0f0+P.drot)*P.cs2
    restrict_l2l1_2dT!(P.backend)(P.rho,P.ux,P.uy,P.T, C.rho,C.ux,C.uy,C.T,
        Int32(ci0),Int32(cj0), g, eint, Int32(nbx),Int32(nby); ndrange=(nbx,nby))
    _psync(P.backend)
end
function pond_fv2dT_tree_step!(G::Vector{<:PonD2DT}, T::Vector{PondTreeNode}, t::Int)
    n=length(G); @assert length(T)==n
    ord=sortperm(_tree_depths(T); rev=true)                       # deepest-first (stable ⇒ chain order exact)
    for i in 2:n                                                  # phase 1 (all parent states are t^n)
        T[i].parent==0 && continue
        C=G[i]; P=G[T[i].parent]
        _tree_prolong_pin!(C, P, T[i].pbox)
        if C.sst==Int32(1) && P.sst==Int32(1) && size(C.hk,1)>1   # k/ω halo (parent must also carry SST)
            C.sst_halo=Int32(1)                                   # arm the border pin (stays 0 when the parent has no k/ω ⇒ legacy borders, byte-identity)
            ci0,cj0,_,_=T[i].pbox
            prolong_kw_l1l2_2dT!(C.backend)(C.hk,C.hom, P.tk,P.tom,
                Int32(ci0),Int32(cj0), Int32(C.ng), Int32(P.NX),Int32(P.NY), Int32(C.NX),Int32(C.NY); ndrange=(Int(C.NX),Int(C.NY)))
            KernelAbstractions.synchronize(C.backend)
        end
    end
    for i in ord                                                  # phase 2
        T[i].substeps==1 && pond_fv2dT_advance!(G[i], t)
    end
    for i in ord                                                  # phase 3
        (T[i].substeps==1 && T[i].parent!=0) || continue
        _tree_restrict!(G[i], G[T[i].parent], T[i].pbox)
    end
    for i in ord                                                  # phase 4: sub-cycled patches (leaves)
        T[i].substeps==2 || continue
        @assert T[i].h2!==nothing "substeps==2 node needs h2 buffers"
        C=G[i]; P=G[T[i].parent]; h2=T[i].h2
        _tree_prolong_pin!(C, P, T[i].pbox; pin=false, dst=h2)    # parent^{n+1} → h2 (no pin yet)
        @. C.hrho = 0.5f0*C.hrho + 0.5f0*h2.rho
        @. C.hux = 0.5f0*C.hux + 0.5f0*h2.ux
        @. C.huy = 0.5f0*C.huy + 0.5f0*h2.uy
        @. C.hT = 0.5f0*C.hT + 0.5f0*h2.T
        pond_fv2dT_advance!(C, t)                                 # substep A (pins t^{n+½} halo each stage)
        copyto!(C.hrho,h2.rho); copyto!(C.hux,h2.ux); copyto!(C.huy,h2.uy); copyto!(C.hT,h2.T)
        pond_fv2dT_advance!(C, t)                                 # substep B (pins t^{n+1} halo)
        _tree_restrict!(C, P, T[i].pbox)
    end
    return nothing
end

# Host: one coupled 2-level step. The fine box occupies L1 cells [ci0:ci0+nbx-1]×[cj0:cj0+nby-1];
# L2 must be sized NX2=2·nbx+2·ng, NY2=2·nby+2·ng and built with nested=true.
function pond_fv2dT_nested_step!(L1::PonD2DT, L2::PonD2DT, ci0::Int, cj0::Int, nbx::Int, nby::Int, t::Int)
    g=Int32(L2.ng); nd2=(Int(L2.NX),Int(L2.NY))
    # (1) prolong L1^n → L2 halo buffers
    prolong_l1l2_2dT!(L2.backend)(L2.hrho,L2.hux,L2.huy,L2.hT, L1.rho,L1.ux,L1.uy,L1.T,
        Int32(ci0),Int32(cj0), g, Int32(L1.NX),Int32(L1.NY), Int32(L2.NX),Int32(L2.NY); ndrange=nd2)
    KernelAbstractions.synchronize(L2.backend)
    # (2) pin the halo into L2's live border (so the stage-1 flux reads a filled halo)
    _pin_halo2T!(L2.backend)(L2.rho,L2.ux,L2.uy,L2.T, L2.hrho,L2.hux,L2.huy,L2.hT, Int32(L2.NX),Int32(L2.NY), g; ndrange=nd2)
    KernelAbstractions.synchronize(L2.backend)
    # (3) step the fine box (nested: applyBC! re-pins the halo each RK stage)
    pond_fv2dT_advance!(L2, t)
    # (4) step the coarse base (whole domain, normal BCs)
    pond_fv2dT_advance!(L1, t)
    # (5) restrict L2 interior → L1 covered cells (conserved average ⇒ freestream-exact)
    eint = 0.5f0*(3.0f0+L1.drot)*L1.cs2
    restrict_l2l1_2dT!(L1.backend)(L1.rho,L1.ux,L1.uy,L1.T, L2.rho,L2.ux,L2.uy,L2.T,
        Int32(ci0),Int32(cj0), g, eint, Int32(nbx),Int32(nby); ndrange=(nbx,nby))
    KernelAbstractions.synchronize(L1.backend)
    return nothing
end
