# =====================================================================================================
# PHASE 0 step 4b (implicit roadmap, 2026-07-11) — the COMPOSITE-GRID STATE.
#
# The 4-level nested solution as ONE abstract vector for the steady/implicit tier (JFNK, Phase 1-2)
# and, later, the adjoint. Panel BLOCKER semantics: covered-coarse cells and child ghost-halo bands
# are NOT degrees of freedom — they are restriction/prolongation images, re-derived by
# `composite_sync!` from the true DOFs; L1 BC/sponge bands, solid cells, merged-sliver members and
# TE-patch χ cells are likewise excluded from the inner product (the MASK is the DOF indicator).
#
# Design constraints honoured here:
#  - ABSTRACT interface: everything below consumes (levels, masks) — the 3D block-sparse port and the
#    Enzyme adjoint swap the storage, not the calling code.
#  - FP64 reductions (dot/norm) regardless of the Float32 state (roadmap: "FP64 for all inner
#    products/reductions").
#  - No flat gather/scatter: vectors live level-shaped (GPU-friendly); the mask enters multiplicatively.
#  - HONESTY: cget!/cset! move ALL cells; the mask governs the INNER PRODUCT only. Dependent cells are
#    re-derived by sync!; L1 BC-band cells are carried as pinned Dirichlet data (the operator wrapper
#    of Phase 1 re-imposes BCs on set — not this layer's job). dyn-SGS cd / SST wm are FROZEN input
#    state of R (rate-ization audit, Phase-0 doc).
# =====================================================================================================

struct CompositeVec{A}
    f::Vector{NTuple{4,A}}          # per level: (r, ux, uy, T)-shaped device arrays
end

struct Composite2T{A}
    L::Vector{PonD2DT}              # the levels (state lives HERE — the composite wraps, not copies)
    parent::Vector{Int}             # tree: parent index per level (0 for root)
    pbox::Vector{NTuple{4,Int}}     # child box in the parent's index space ((0,0,0,0) for root)
    mask::Vector{A}                 # per-level Float32 0/1 DOF indicator (device)
    mask_h::Vector{Matrix{Float32}} # host copies (diagnostics/argmax loops — single source of truth)
    ndof::Int                       # Σ mask (cells; ×4 fields = vector length)
end

# ---- mask construction: THE definition of the composite DOF set -------------------------------------
# root_jband: extra masked rows top/bottom on the root (sponge_w); chi_h: finest-level χ (TE patch);
# lam_h/tgt_h: finest-level cut-cell volume fraction + merge target map (nothing on rect levels).
function composite_masks(L::Vector{<:PonD2DT}, parent::Vector{Int}, pbox::Vector{NTuple{4,Int}};
                         root_jband::Int=0, lam_h=nothing, tgt_h=nothing, chi_h=nothing, dev=identity)
    n=length(L); mh=Vector{Matrix{Float32}}(undef,n)
    for l in 1:n
        d=L[l]; NXl=Int(d.NX); NYl=Int(d.NY); g=Int(d.ng)
        M=zeros(Float32,NXl,NYl)
        ilo=g+1; ihi=NXl-g; jlo=g+1; jhi=NYl-g
        if parent[l]==0 && root_jband>0
            jlo=max(jlo,root_jband+1); jhi=min(jhi,NYl-root_jband)
        end
        @inbounds for j in jlo:jhi, i in ilo:ihi; M[i,j]=1f0; end
        for c in 1:n                                    # cells covered by ANY child box are images
            parent[c]==l || continue
            ci0,cj0,nbx,nby = pbox[c]
            @inbounds for j in max(cj0,1):min(cj0+nby-1,NYl), i in max(ci0,1):min(ci0+nbx-1,NXl)
                M[i,j]=0f0
            end
        end
        if l==n
            lam_h!==nothing && @inbounds for j in 1:NYl, i in 1:NXl
                lam_h[i,j]<=1f-6 && (M[i,j]=0f0)        # solid
            end
            tgt_h!==nothing && @inbounds for j in 1:NYl, i in 1:NXl
                tgt_h[i,j]!=Int32(0) && (M[i,j]=0f0)    # merged sliver member (host carries the group)
            end
            chi_h!==nothing && @inbounds for j in 1:NYl, i in 1:NXl
                chi_h[i,j]>1f-3 && (M[i,j]=0f0)         # TE-patch χ (penalization not yet in R)
            end
        end
        mh[l]=M
    end
    mh, [dev(mh[l]) for l in 1:n], sum(m->round(Int,sum(Float64.(m))), mh)
end

function composite_build(L::Vector{<:PonD2DT}, parent::Vector{Int}, pbox::Vector{NTuple{4,Int}};
                         kwargs...)
    mh, md, ndof = composite_masks(L, parent, pbox; kwargs...)
    Composite2T{typeof(md[1])}(L, parent, pbox, md, mh, ndof)
end

# ---- C2 FIX (engineering block B, 2026-07-12): merge-group consistency for the Newton path -----------
# The cut-cell merge groups (sliver members absorbed into a host) were composite-inconsistent in two
# ways, recorded as known defect C2 at Phase-2 review:
#   (a) crates! handed Newton the host's RAW divergence Dc — but the smoother/explicit tier's group
#       fixed point is Σ Λᵢ·scᵢ·Dcᵢ = 0 (host rate sliver-scaled by sc, members by 1). Newton was
#       driving a DIFFERENT zero set at every cut cell. _mg_rg_fixup! deposits the volume-weighted
#       group residual R_g = (Λc·sc·Dc_host + Σ Λs·Dc_member)/(Λc+ΣΛs) at the host (the group's DOF).
#   (b) cset!/sync left MEMBERS stale after a Newton update (mask excludes them; restriction/prolongation
#       never touches them) — the next residual read inconsistent member states in member-neighbor
#       fluxes. _mg_member_sync! mirrors host state onto members, matching the smoother's own
#       post-update write (identical primitive state to host and all members).
# Race-free in place: members have tgt≠0 (their threads no-op); each member has exactly ONE host;
# hosts are never members. sc formula copied verbatim from _frd_merge2T_lts!.
@kernel function _mg_rg_fixup!(vr,vmx,vmy,ve, @Const(lam), @Const(tgt), @Const(links), @Const(nlk), NX::Int32)
    i,j=@index(Global,NTuple); nx=Int(NX)
    @inbounds begin
        Λc=lam[i,j]
        if Λc>1f-6 && tgt[i,j]==Int32(0)
            nl=Int(nlk[i,j])
            sc = Λc<0.6f0 ? min(1.0f0, 2.0f0*Λc) : 1.0f0
            if nl>0 || sc<1.0f0
                sw=Λc
                sr=Λc*sc*vr[i,j]; smx=Λc*sc*vmx[i,j]; smy=Λc*sc*vmy[i,j]; se=Λc*sc*ve[i,j]
                for k in 1:nl
                    li=Int(links[i,j,k]); ii=((li-1)%nx)+1; jj=((li-1)÷nx)+1
                    Λs=lam[ii,jj]; sw+=Λs
                    sr+=Λs*vr[ii,jj]; smx+=Λs*vmx[ii,jj]; smy+=Λs*vmy[ii,jj]; se+=Λs*ve[ii,jj]
                end
                invw=1.0f0/sw
                vr[i,j]=sr*invw; vmx[i,j]=smx*invw; vmy[i,j]=smy*invw; ve[i,j]=se*invw
            end
        end
    end
end
@kernel function _mg_member_sync!(rho,ux,uy,Temp, @Const(lam), @Const(tgt), @Const(links), @Const(nlk), NX::Int32)
    i,j=@index(Global,NTuple); nx=Int(NX)
    @inbounds begin
        Λc=lam[i,j]
        if Λc>1f-6 && tgt[i,j]==Int32(0)
            nl=Int(nlk[i,j])
            if nl>0
                r=rho[i,j]; u=ux[i,j]; v=uy[i,j]; t=Temp[i,j]
                for k in 1:nl
                    li=Int(links[i,j,k]); ii=((li-1)%nx)+1; jj=((li-1)÷nx)+1
                    rho[ii,jj]=r; ux[ii,jj]=u; uy[ii,jj]=v; Temp[ii,jj]=t
                end
            end
        end
    end
end
# is the merge machinery live on this level? (stub arrays are 1×1)
_mg_live(d) = d.ghost_wall==Int32(1) && size(d.mg_tgt,1)==Int(d.NX)

# ---- dependent-cell synchronisation: covered-coarse ← restriction; child halos ← prolongation --------
# Order matters: restrict DEEPEST-FIRST (parents see current children), then prolong TOP-DOWN
# (halos come from already-synced parents). After sync! the state is composite-consistent: every
# masked-out dependent cell is exactly the image of DOFs — including merge MEMBERS (C2 fix (b)).
function composite_sync!(C::Composite2T)
    n=length(C.L)
    for l in n:-1:2; _tree_restrict!(C.L[l], C.L[C.parent[l]], C.pbox[l]); end
    for l in 2:n;    _tree_prolong_pin!(C.L[l], C.L[C.parent[l]], C.pbox[l]); end
    d=C.L[n]
    if _mg_live(d)
        _mg_member_sync!(d.backend)(d.rho,d.ux,d.uy,d.T, d.lam, d.mg_tgt,d.mg_links,d.mg_nlk,
            Int32(d.NX); ndrange=(d.NX,d.NY))
        KernelAbstractions.synchronize(d.backend)
    end
    return nothing
end

# ---- the residual as a composite operator: R(U) into the levels' dc_* rate arrays --------------------
# (cd/wm frozen input state — see pond_fv2dT_residual! docstring + the Phase-0 rate-ization audit.)
function composite_residual!(C::Composite2T)
    composite_sync!(C)
    for l in 1:length(C.L); pond_fv2dT_residual!(C.L[l]); end
    return nothing
end

# ---- vector allocation / state transfer --------------------------------------------------------------
_alloc_like(a)=similar(a)
function CompositeVec(C::Composite2T)
    CompositeVec([(_alloc_like(d.rho),_alloc_like(d.ux),_alloc_like(d.uy),_alloc_like(d.T)) for d in C.L])
end
function cget!(v::CompositeVec, C::Composite2T)         # state → vector (all cells; mask rules the dot)
    for (l,d) in enumerate(C.L)
        copyto!(v.f[l][1],d.rho); copyto!(v.f[l][2],d.ux); copyto!(v.f[l][3],d.uy); copyto!(v.f[l][4],d.T)
    end; v
end
function cset!(C::Composite2T, v::CompositeVec)         # vector → state (then re-derive dependents)
    for (l,d) in enumerate(C.L)
        copyto!(d.rho,v.f[l][1]); copyto!(d.ux,v.f[l][2]); copyto!(d.uy,v.f[l][3]); copyto!(d.T,v.f[l][4])
    end
    composite_sync!(C); C
end
function crates!(v::CompositeVec, C::Composite2T)       # R(U) → vector (from dc_*; call after composite_residual!)
    for (l,d) in enumerate(C.L)
        copyto!(v.f[l][1],d.dc_r); copyto!(v.f[l][2],d.dc_mx); copyto!(v.f[l][3],d.dc_my); copyto!(v.f[l][4],d.dc_e)
    end
    d=C.L[end]                                          # C2 fix (a): host residual ← Λ·sc-weighted group rate
    if _mg_live(d)
        _mg_rg_fixup!(d.backend)(v.f[end][1],v.f[end][2],v.f[end][3],v.f[end][4],
            d.lam, d.mg_tgt,d.mg_links,d.mg_nlk, Int32(d.NX); ndrange=(d.NX,d.NY))
        _psync(d.backend)     # PV_NOSYNC-elidable: only device-side consumers follow (same-stream ordered); host reads go through cnorm's self-syncing mapreduce
    end
    v
end

# ---- BLAS-1 on the composite (FP64 reductions; masked inner products) --------------------------------
function caxpy!(a::Real, x::CompositeVec, y::CompositeVec)      # y ← a·x + y  (all cells; cheap, mask-free)
    af=Float32(a)
    for l in 1:length(x.f), c in 1:4; y.f[l][c] .= af.*x.f[l][c] .+ y.f[l][c]; end
    y
end
function cscale!(a::Real, x::CompositeVec)
    af=Float32(a)
    for l in 1:length(x.f), c in 1:4; x.f[l][c] .*= af; end
    x
end
ccopy!(dst::CompositeVec, src::CompositeVec) = (for l in 1:length(src.f), c in 1:4
    copyto!(dst.f[l][c], src.f[l][c]) end; dst)
function cmask!(x::CompositeVec, C::Composite2T)        # zero all non-DOF components (Newton updates
    for l in 1:length(x.f), c in 1:4                    # must not touch BC bands/solid/chi/halo cells)
        x.f[l][c] .*= C.mask[l]
    end
    x
end
function cdot(C::Composite2T, x::CompositeVec, y::CompositeVec)::Float64
    s=0.0
    for l in 1:length(x.f)
        m=C.mask[l]
        for c in 1:4
            s += mapreduce((a,b,w)->Float64(a)*Float64(b)*Float64(w), +, x.f[l][c], y.f[l][c], m; init=0.0)
        end
    end
    s
end
cnorm(C::Composite2T, x::CompositeVec)::Float64 = sqrt(cdot(C,x,x))
# per-level, per-component masked L2/L∞ (diagnostics; FP64)
function cnorms_by_level(C::Composite2T, x::CompositeVec)
    out=NTuple{2,Float64}[]
    for l in 1:length(x.f), c in 1:4
        m=C.mask[l]
        s2=mapreduce((a,w)->Float64(a)*Float64(a)*Float64(w), +, x.f[l][c], m; init=0.0)
        mx=mapreduce((a,w)->Float64(abs(a))*Float64(w), max, x.f[l][c], m; init=0.0)
        n=max(round(Int,sum(Float64.(C.mask_h[l]))),1)
        push!(out,(sqrt(s2/n),mx))
    end
    out
end
