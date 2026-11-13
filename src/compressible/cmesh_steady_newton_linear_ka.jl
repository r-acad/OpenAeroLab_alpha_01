# Linear algebra of the pseudo-transient Newton-Krylov airfoil march (cmesh_steady_newton_ka.jl). Experimental and
# opt-in: nothing here is called by the default solver paths. Included into module CMeshSolverKA.
#
# Contents: the local pseudo-time diagonal and the assembled 5-point 4x4 block operator (the matrix of the kinetic
# block workspace); a right-preconditioned restarted GMRES on flat host vectors; the forward finite-difference
# product of the true residual's Jacobian; alternating zebra block-line relaxation (wall-normal lines continued
# through the wake cut, then streamwise lines); a Galerkin agglomerated coarse operator solved by sparse LU; and
# first-order Roe convection blocks for the preconditioner matrix.
# The line and face loops are KernelAbstractions kernels; GMRES, the coarse solve and the packing are host code,
# so the march is a CPU method until those parts have device forms.

using SparseArrays

# Local pseudo-time diagonal V/dtau.
#   mode 0: (lam_i + lam_j)/CFL, the usual two-direction explicit-stability scale (set by the SHORT cell side);
#   mode 1: 2*min(lam_i, lam_j)/CFL, the LONG cell side: identical in isotropic cells, aspect-ratio times larger
#           steps in stretched boundary-layer cells. Admissible because both directions are implicit.
const NEWTON_D0_MODE=Ref(0)
@kernel function _k_newton_d0!(d0,@Const(ux),@Const(uy),@Const(T),@Const(Six),@Const(Siy),
    @Const(Sjx),@Const(Sjy),cfl,mode::Int32,NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI&&j<=NJ
        ia=i+g;ja=j+g;u=ux[ia,ja];v=uy[ia,ja]
        c=3*sqrt(max(T[ia,ja],oftype(u,1e-3)))
        siw=hypot(Six[i,j],Siy[i,j]);sie=hypot(Six[i+1,j],Siy[i+1,j])
        sjs=hypot(Sjx[i,j],Sjy[i,j]);sjn=hypot(Sjx[i,j+1],Sjy[i,j+1])
        li=(abs(u*Six[i,j]+v*Siy[i,j])+abs(u*Six[i+1,j]+v*Siy[i+1,j])+c*(siw+sie))/2
        lj=(abs(u*Sjx[i,j]+v*Sjy[i,j])+abs(u*Sjx[i,j+1]+v*Sjy[i,j+1])+c*(sjs+sjn))/2
        d0[i,j]=(mode==Int32(1) ? 2*min(li,lj) : li+lj)/cfl
    end
end
function newton_d0!(d0,s::KSol,cfl)
    _k_newton_d0!(s.backend)(d0,s.ux,s.uy,s.T,s.Six,s.Siy,s.Sjx,s.Sjy,eltype(d0)(cfl),Int32(NEWTON_D0_MODE[]),
        Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=(s.NI,s.NJ))
    d0
end

# y = A x on the 5-point block stencil, including the wake-cut coupling at j = 1.
@kernel function _k_newton_matvec!(y1,y2,y3,y4,@Const(x1),@Const(x2),@Const(x3),@Const(x4),
    @Const(li),@Const(ui),@Const(lj),@Const(dg),@Const(uj),
    NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32)
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI&&j<=NJ
        B=eltype(dg);x=(x1,x2,x3,x4)
        acc=_airfoil_block4(dg,i,j)*_airfoil_vec4(x,i,j,B)
        if i>1;acc+=_airfoil_block4(li,i,j)*_airfoil_vec4(x,i-Int32(1),j,B);end
        if i<NI;acc+=_airfoil_block4(ui,i,j)*_airfoil_vec4(x,i+Int32(1),j,B);end
        if j>1
            acc+=_airfoil_block4(lj,i,j)*_airfoil_vec4(x,i,j-Int32(1),B)
        elseif i<=NIW||i>NIW+NIF
            acc+=_airfoil_block4(lj,i,j)*_airfoil_vec4(x,NI+Int32(1)-i,Int32(1),B)
        end
        if j<NJ;acc+=_airfoil_block4(uj,i,j)*_airfoil_vec4(x,i,j+Int32(1),B);end
        y1[i,j]=acc[1];y2[i,j]=acc[2];y3[i,j]=acc[3];y4[i,j]=acc[4]
    end
end
function newton_matvec!(y,x,s::KSol,kw)
    _k_newton_matvec!(s.backend)(y...,x...,kw.lower_i,kw.upper_i,kw.lower_j,kw.diagonal,kw.upper_j,
        Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF);ndrange=(s.NI,s.NJ))
    y
end

# Host sparse copy of the block matrix (diagnostic exact solves; CPU arrays only).
function newton_sparse(kw,NI::Int,NJ::Int,NIW::Int,NIF::Int)
    n=4*NI*NJ;idx(i,j,c)=((j-1)*NI+(i-1))*4+c
    I=Int[];J=Int[];V=Float64[];cap=20*n÷4*4;sizehint!(I,cap);sizehint!(J,cap);sizehint!(V,cap)
    add!(blk,i,j,i2,j2)=for c2 in 1:4,c1 in 1:4
        push!(I,idx(i,j,c1));push!(J,idx(i2,j2,c2));push!(V,blk[c1,c2,i,j])
    end
    for j in 1:NJ,i in 1:NI
        add!(kw.diagonal,i,j,i,j)
        i>1&&add!(kw.lower_i,i,j,i-1,j)
        i<NI&&add!(kw.upper_i,i,j,i+1,j)
        if j>1
            add!(kw.lower_j,i,j,i,j-1)
        elseif i<=NIW||i>NIW+NIF
            add!(kw.lower_j,i,j,NI+1-i,1)
        end
        j<NJ&&add!(kw.upper_j,i,j,i,j+1)
    end
    sparse(I,J,V,n,n)
end
newton_pack(x)=vec(permutedims(cat(x...;dims=3),(3,1,2)))
function newton_unpack!(x,v,NI,NJ)
    a=reshape(v,4,NI,NJ)
    for c in 1:4;x[c].=@view a[c,:,:];end
    x
end

# Conservative trial U = U0 - alpha*delta (primitive arrays written; ghosts untouched).
function newton_trial!(s::KSol,backup,delta,alpha)
    _k_implicit_line_trial!(s.backend)(s.rho,s.ux,s.uy,s.T,backup...,delta...,
        s.eint,eltype(s.rho)(alpha),Int32(s.NI),Int32(s.NJ),Int32(s.g);ndrange=(s.NI,s.NJ))
end

# Largest alpha <= 1 keeping the density and total-energy change below `fraction` in every cell.
function newton_physical_alpha(s::KSol,backup,delta,fraction)
    g=s.g;ir=g+1:g+s.NI;jr=g+1:g+s.NJ;eint=s.eint;f=eltype(s.rho)(fraction)
    mapreduce((r,u,v,t,dr,de)->min(one(r),f*r/max(abs(dr),oftype(r,1e-30)),
            f*r*(eint*t+(u*u+v*v)/2)/max(abs(de),oftype(r,1e-30))),
        min,view(backup[1],ir,jr),view(backup[2],ir,jr),view(backup[3],ir,jr),
        view(backup[4],ir,jr),delta[1],delta[4])
end

# Scaled steady residual RMS per equation (same scaling as the public diagnostics).
function newton_flow_l2(s::KSol)
    r=max(Float64(s.rho_inf),1e-12);u=max(hypot(Float64(s.uxi),Float64(s.uyi)),1e-3)
    e=max(Float64(s.eint*s.T_inf),u*u);scales=(r,r*u,r*u,r*e);n=length(s.vol)
    raw=(s.dc_r,s.dc_mx,s.dc_my,s.dc_e)
    ntuple(k->sqrt(mapreduce((a,v)->abs2(Float64(a)/v),+,raw[k],s.vol)/n)/scales[k],4)
end


# ---- right-preconditioned GMRES -------------------------------------------------------------------------
const _NEWTON_KRYLOV_POOL=Dict{Int,Vector{Vector{Float64}}}()
struct _NewtonStride;pool::Vector{Vector{Float64}};n::Int;phase::Int;end
function Base.getindex(p::_NewtonStride,k::Int)
    slot=1+2*(k-1)+p.phase+1                 # slot 1 is the work vector
    while length(p.pool)<slot;push!(p.pool,Vector{Float64}(undef,p.n));end
    p.pool[slot]
end
# Krylov vectors are reused between calls and created only when an iteration first needs them
# (a fresh basis of 60 vectors of a 173k-cell mesh is several hundred MB of page faults per solve).
struct _NewtonPoolView;pool::Vector{Vector{Float64}};offset::Int;n::Int;end
function Base.getindex(p::_NewtonPoolView,k::Int)
    while length(p.pool)<p.offset+k;push!(p.pool,Vector{Float64}(undef,p.n));end
    p.pool[p.offset+k]
end
"Right-preconditioned restarted GMRES on flat vectors. A!(y,x) and P!(z,v) are callables. Returns (x, iterations, true relative residual history)."
function newton_gmres(A!,P!,b::Vector{Float64};tol=1e-2,restart=60,maxit=120)
    n=length(b);x=zeros(n);r=copy(b);beta0=norm(b);hist=Float64[1.0];total=0
    beta0==0&&return x,0,hist
    pool=get!(()->Vector{Float64}[],_NEWTON_KRYLOV_POOL,n)
    # interleaved slots so that short solves touch few vectors: w, then V[1], Z[1], V[2], Z[2], ...
    w=_NewtonPoolView(pool,0,n)[1]
    V=_NewtonStride(pool,n,0);Z=_NewtonStride(pool,n,1)
    while total<maxit
        beta=norm(r);beta<=tol*beta0&&break
        V[1].=r./beta;H=zeros(restart+1,restart);g=zeros(restart+1);g[1]=beta
        cs=zeros(restart);sn=zeros(restart);k=0
        for j in 1:restart
            P!(Z[j],V[j]);A!(w,Z[j]);total+=1
            for i in 1:j;H[i,j]=dot(w,V[i]);w.-=H[i,j].*V[i];end
            H[j+1,j]=norm(w);V[j+1].=w./H[j+1,j]
            for i in 1:j-1
                t=cs[i]*H[i,j]+sn[i]*H[i+1,j];H[i+1,j]=-sn[i]*H[i,j]+cs[i]*H[i+1,j];H[i,j]=t
            end
            d=hypot(H[j,j],H[j+1,j]);cs[j]=H[j,j]/d;sn[j]=H[j+1,j]/d;H[j,j]=d;H[j+1,j]=0
            g[j+1]=-sn[j]*g[j];g[j]=cs[j]*g[j];k=j
            push!(hist,abs(g[j+1])/beta0)
            (abs(g[j+1])<=tol*beta0||total>=maxit)&&break
        end
        yk=UpperTriangular(H[1:k,1:k])\g[1:k]
        for i in 1:k;x.+=yk[i].*Z[i];end
        A!(w,x);r.=b.-w
        push!(hist,norm(r)/beta0)   # true residual at the restart
    end
    x,total,hist
end



# ---- Jacobian action ------------------------------------------------------------------------------------
"Forward finite-difference action of (d0 + dR/dU) on a conservative direction; `base` is the residual at the backed-up state."
function newton_fd_jv_fwd!(out,v,s::KSol,backup,base,d0;rel=1e-7)
    r=Float64(s.rho_inf);u=max(hypot(Float64(s.uxi),Float64(s.uyi)),1e-3)
    E=r*(Float64(s.eint*s.T_inf)+0.5*u*u)
    scaled=max(maximum(abs,v[1])/r,maximum(abs,v[2])/(r*u),maximum(abs,v[3])/(r*u),maximum(abs,v[4])/E)
    scaled==0&&(foreach(o->fill!(o,0),out);return out)
    h=rel/scaled
    newton_trial!(s,backup,v,-h);_raw_residual_ka!(s,Int32(2),Int32(1))
    cur=_airfoil_residual(s)
    for k in 1:4;@. out[k]=(cur[k]-base[k])/h+d0*v[k];end
    for (d,b) in zip((s.rho,s.ux,s.uy,s.T),backup);copyto!(d,b);end
    out
end

# ---- block-line relaxation ------------------------------------------------------------------------------
struct NewtonLines{A,C}
    jdinv::A;jc::A;idinv::A;ic::A
    r::NTuple{4,C}
    jcolor::Vector{Int32};njcolor::Int;colcolor::Vector{Int32}
end

function newton_lines_init(s::KSol,kw)
    b4()=similar(kw.diagonal)
    NIF,NIW=s.NIF,s.NIW;nl=NIF+NIW
    # Greedy colouring of the j-lines: foil columns l=1..NIF, wake pairs NIF+w.
    nb(l)=l<=NIF ? (l==1 ? (NIW>0 ? NIF+NIW : 0) : l-1, l==NIF ? (NIW>0 ? NIF+NIW : 0) : l+1) :
        (l-NIF>1 ? l-1 : 0, l-NIF<NIW ? l+1 : 0, l-NIF==NIW ? 1 : 0, l-NIF==NIW ? NIF : 0)
    color=fill(Int32(-1),nl)
    for l in 1:nl
        used=Set(color[n] for n in nb(l) if n>0)
        c=Int32(0);while c in used;c+=Int32(1);end
        color[l]=c
    end
    for l in 1:nl,n in nb(l);n>0&&color[n]==color[l]&&error("line colouring failed");end
    colcolor=Int32[i<=NIW ? color[NIF+i] : i>NIW+NIF ? color[NIF+s.NI+1-i] : color[i-NIW] for i in 1:s.NI]
    NewtonLines(b4(),b4(),b4(),b4(),ntuple(_->similar(s.dtf),4),color,Int(maximum(color))+1,colcolor)
end

@kernel function _k_newton_jline_factor!(dinv,cmat,@Const(lower),@Const(diagonal),@Const(upper),
    NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32)
    l=@index(Global,Linear)
    @inbounds if l<=NIW+NIF
        B=eltype(diagonal);count=l<=NIF ? NJ : Int32(2)*NJ
        prev=zero(SMatrix{4,4,B,16})
        for m in Int32(1):count
            i,j,lo,hi=_airfoil_block_cut_cell(l,m,lower,upper,NI,NJ,NIW,NIF)
            di=inv(_airfoil_block4(diagonal,i,j)-lo*prev)
            ci=m<count ? di*hi : zero(SMatrix{4,4,B,16})
            _airfoil_block4_store!(dinv,di,i,j);_airfoil_block4_store!(cmat,ci,i,j)
            prev=ci
        end
    end
end

@kernel function _k_newton_iline_factor!(dinv,cmat,@Const(lower),@Const(diagonal),@Const(upper),NI::Int32,NJ::Int32)
    j=@index(Global,Linear)
    @inbounds if j<=NJ
        B=eltype(diagonal);prev=zero(SMatrix{4,4,B,16})
        for i in Int32(1):NI
            lo=i>1 ? _airfoil_block4(lower,i,j) : zero(SMatrix{4,4,B,16})
            di=inv(_airfoil_block4(diagonal,i,j)-lo*prev)
            ci=i<NI ? di*_airfoil_block4(upper,i,j) : zero(SMatrix{4,4,B,16})
            _airfoil_block4_store!(dinv,di,i,j);_airfoil_block4_store!(cmat,ci,i,j)
            prev=ci
        end
    end
end

function newton_lines_factor!(L::NewtonLines,s::KSol,kw)
    ni,nj,niw,nif=Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF)
    _k_newton_jline_factor!(s.backend)(L.jdinv,L.jc,kw.lower_j,kw.diagonal,kw.upper_j,ni,nj,niw,nif;ndrange=s.NIW+s.NIF)
    _k_newton_iline_factor!(s.backend)(L.idinv,L.ic,kw.lower_i,kw.diagonal,kw.upper_i,ni,nj;ndrange=s.NJ)
    L
end

# r = b - A x
@kernel function _k_newton_residual!(r1,r2,r3,r4,@Const(b1),@Const(b2),@Const(b3),@Const(b4),
    @Const(x1),@Const(x2),@Const(x3),@Const(x4),
    @Const(li),@Const(ui),@Const(lj),@Const(dg),@Const(uj),NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,
    @Const(colcolor),mode::Int32,c::Int32)
    # mode 0: every cell; 1: only columns of j-line colour c; 2: only rows of parity c.
    i,j=@index(Global,NTuple)
    @inbounds if i<=NI&&j<=NJ&&(mode==Int32(0)||(mode==Int32(1)&&colcolor[i]==c)||(mode==Int32(2)&&(j&Int32(1))==c))
        B=eltype(dg);x=(x1,x2,x3,x4)
        acc=_airfoil_block4(dg,i,j)*_airfoil_vec4(x,i,j,B)
        if i>1;acc+=_airfoil_block4(li,i,j)*_airfoil_vec4(x,i-Int32(1),j,B);end
        if i<NI;acc+=_airfoil_block4(ui,i,j)*_airfoil_vec4(x,i+Int32(1),j,B);end
        if j>1
            acc+=_airfoil_block4(lj,i,j)*_airfoil_vec4(x,i,j-Int32(1),B)
        elseif i<=NIW||i>NIW+NIF
            acc+=_airfoil_block4(lj,i,j)*_airfoil_vec4(x,NI+Int32(1)-i,Int32(1),B)
        end
        if j<NJ;acc+=_airfoil_block4(uj,i,j)*_airfoil_vec4(x,i,j+Int32(1),B);end
        r1[i,j]=b1[i,j]-acc[1];r2[i,j]=b2[i,j]-acc[2];r3[i,j]=b3[i,j]-acc[3];r4[i,j]=b4[i,j]-acc[4]
    end
end

# x += L_j^-1 r on the j-lines of one colour.
@kernel function _k_newton_jline_solve!(x1,x2,x3,x4,r1,r2,r3,r4,@Const(dinv),@Const(cmat),
    @Const(lower),@Const(upper),@Const(color),c::Int32,NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32)
    l=@index(Global,Linear)
    @inbounds if l<=NIW+NIF&&color[l]==c
        B=eltype(dinv);x=(x1,x2,x3,x4);r=(r1,r2,r3,r4)
        count=l<=NIF ? NJ : Int32(2)*NJ
        prev=zero(SVector{4,B})
        for m in Int32(1):count
            i,j,lo,_=_airfoil_block_cut_cell(l,m,lower,upper,NI,NJ,NIW,NIF)
            y=_airfoil_block4(dinv,i,j)*(_airfoil_vec4(r,i,j,B)-lo*prev)
            _airfoil_vec4_store!(r,y,i,j);prev=y
        end
        i,j,_,_=_airfoil_block_cut_cell(l,count,lower,upper,NI,NJ,NIW,NIF)
        next=_airfoil_vec4(r,i,j,B)
        x1[i,j]+=next[1];x2[i,j]+=next[2];x3[i,j]+=next[3];x4[i,j]+=next[4]
        for m in count-Int32(1):Int32(-1):Int32(1)
            i,j,_,_=_airfoil_block_cut_cell(l,m,lower,upper,NI,NJ,NIW,NIF)
            next=_airfoil_vec4(r,i,j,B)-_airfoil_block4(cmat,i,j)*next
            x1[i,j]+=next[1];x2[i,j]+=next[2];x3[i,j]+=next[3];x4[i,j]+=next[4]
        end
    end
end

# x += L_i^-1 r on the i-lines (rows) of one parity.
@kernel function _k_newton_iline_solve!(x1,x2,x3,x4,r1,r2,r3,r4,@Const(dinv),@Const(cmat),
    @Const(lower),c::Int32,NI::Int32,NJ::Int32)
    j=@index(Global,Linear)
    @inbounds if j<=NJ&&(j&Int32(1))==c
        B=eltype(dinv);r=(r1,r2,r3,r4)
        prev=zero(SVector{4,B})
        for i in Int32(1):NI
            y=i>1 ? _airfoil_block4(dinv,i,j)*(_airfoil_vec4(r,i,j,B)-_airfoil_block4(lower,i,j)*prev) :
                    _airfoil_block4(dinv,i,j)*_airfoil_vec4(r,i,j,B)
            _airfoil_vec4_store!(r,y,i,j);prev=y
        end
        next=_airfoil_vec4(r,NI,j,B)
        x1[NI,j]+=next[1];x2[NI,j]+=next[2];x3[NI,j]+=next[3];x4[NI,j]+=next[4]
        for i in NI-Int32(1):Int32(-1):Int32(1)
            next=_airfoil_vec4(r,i,j,B)-_airfoil_block4(cmat,i,j)*next
            x1[i,j]+=next[1];x2[i,j]+=next[2];x3[i,j]+=next[3];x4[i,j]+=next[4]
        end
    end
end

"x = P(b): `sweeps` alternating zebra line relaxations from x = 0. directions: :ji (both), :j or :i."
function newton_line_precond!(x,b,s::KSol,kw,L::NewtonLines;sweeps::Int=2,directions::Symbol=:ji)
    bk=s.backend;ni,nj,niw,nif=Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF)
    for k in 1:4;fill!(x[k],0);end
    for n in 1:sweeps;newton_line_smooth!(x,b,s,kw,L;zero_start=n==1,directions);end
    x
end

"Full residual r = b - A x into L.r."
function newton_block_residual!(L::NewtonLines,b,x,s::KSol,kw;mode::Int=0,c::Int=0)
    _k_newton_residual!(s.backend)(L.r...,b...,x...,kw.lower_i,kw.upper_i,kw.lower_j,kw.diagonal,kw.upper_j,
        Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF),L.colcolor,Int32(mode),Int32(c);ndrange=(s.NI,s.NJ))
    L.r
end

"One alternating zebra line sweep continuing from the current x. Each colour only needs the residual on its own lines."
function newton_line_smooth!(x,b,s::KSol,kw,L::NewtonLines;zero_start::Bool=false,directions::Symbol=:ji)
    bk=s.backend;ni,nj,niw,nif=Int32(s.NI),Int32(s.NJ),Int32(s.NIW),Int32(s.NIF)
    first=zero_start
    if directions!==:i
        for c in 0:L.njcolor-1
            if first
                for k in 1:4;copyto!(L.r[k],b[k]);end
            else
                newton_block_residual!(L,b,x,s,kw;mode=1,c)
            end
            first=false
            _k_newton_jline_solve!(bk)(x...,L.r...,L.jdinv,L.jc,kw.lower_j,kw.upper_j,L.jcolor,Int32(c),
                ni,nj,niw,nif;ndrange=s.NIW+s.NIF)
        end
    end
    if directions!==:j
        for c in 0:1
            if first
                for k in 1:4;copyto!(L.r[k],b[k]);end
            else
                newton_block_residual!(L,b,x,s,kw;mode=2,c)
            end
            first=false
            _k_newton_iline_solve!(bk)(x...,L.r...,L.idinv,L.ic,kw.lower_i,Int32(c),ni,nj;ndrange=s.NJ)
        end
    end
    x
end

# Field views of a flat vector ordered (i,j,component), so Krylov vectors need no copies.
newton_fields(v::AbstractVector,NI::Int,NJ::Int)=ntuple(k->reshape(view(v,(k-1)*NI*NJ+1:k*NI*NJ),NI,NJ),4)


# ---- two-level cycle ------------------------------------------------------------------------------------
mutable struct NewtonCoarse
    fi::Int;fj::Int;NI::Int;NJ::Int;NIW::Int;NIF::Int
    diagonal::Array{Float64,4};lower_i::Array{Float64,4};upper_i::Array{Float64,4}
    lower_j::Array{Float64,4};upper_j::Array{Float64,4}
    rc::Array{Float64,3};F
end
function newton_coarse_init(s::KSol;fi::Int=4,fj::Int=4)
    all(x->x%fi==0,(s.NI,s.NIW,s.NIF))&&s.NJ%fj==0||throw(ArgumentError("mesh not divisible by the agglomeration"))
    NI,NJ=s.NI÷fi,s.NJ÷fj;z()=zeros(4,4,NI,NJ)
    NewtonCoarse(fi,fj,NI,NJ,s.NIW÷fi,s.NIF÷fi,z(),z(),z(),z(),z(),zeros(4,NI,NJ),nothing)
end

function _newton_block_add!(dst,I,J,src,i,j)
    @inbounds for c in 1:16;dst[c+16*((I-1)+size(dst,3)*(J-1))]+=src[c+16*((i-1)+size(src,3)*(j-1))];end
end

"Galerkin coarse operator from the fine blocks, then its LU."
function newton_coarse_build!(C::NewtonCoarse,s::KSol,kw)
    fi,fj=C.fi,C.fj;NI,NJ,NIW,NIF=s.NI,s.NJ,s.NIW,s.NIF
    for a in (C.diagonal,C.lower_i,C.upper_i,C.lower_j,C.upper_j);fill!(a,0);end
    dg,li,ui,lj,uj=kw.diagonal,kw.lower_i,kw.upper_i,kw.lower_j,kw.upper_j
    Threads.@threads for J in 1:C.NJ
        for j in (J-1)*fj+1:J*fj,i in 1:NI
            I=(i-1)÷fi+1
            _newton_block_add!(C.diagonal,I,J,dg,i,j)
            if i>1;_newton_block_add!((i-2)÷fi+1==I ? C.diagonal : C.lower_i,I,J,li,i,j);end
            if i<NI;_newton_block_add!(i÷fi+1==I ? C.diagonal : C.upper_i,I,J,ui,i,j);end
            if j>1
                _newton_block_add!((j-2)÷fj+1==J ? C.diagonal : C.lower_j,I,J,lj,i,j)
            elseif i<=NIW||i>NIW+NIF
                _newton_block_add!(C.lower_j,I,J,lj,i,j)          # the cut partner's aggregate
            end
            if j<NJ;_newton_block_add!(j÷fj+1==J ? C.diagonal : C.upper_j,I,J,uj,i,j);end
        end
    end
    C.F=lu(newton_sparse(C,C.NI,C.NJ,C.NIW,C.NIF))
    C
end

"x += P * Ac^-1 * R * r  (r: tuple of fine residual fields)."
function newton_coarse_correct!(x,r,C::NewtonCoarse)
    fi,fj=C.fi,C.fj;rc=C.rc
    Threads.@threads for J in 1:C.NJ
        @inbounds for I in 1:C.NI,c in 1:4
            acc=0.0;rk=r[c]
            for j in (J-1)*fj+1:J*fj,i in (I-1)*fi+1:I*fi;acc+=rk[i,j];end
            rc[c,I,J]=acc
        end
    end
    e=reshape(C.F\vec(rc),4,C.NI,C.NJ)
    Threads.@threads for J in 1:C.NJ
        @inbounds for I in 1:C.NI,c in 1:4
            v=e[c,I,J];xk=x[c]
            for j in (J-1)*fj+1:J*fj,i in (I-1)*fi+1:I*fi;xk[i,j]+=v;end
        end
    end
    x
end

"x = P(b): pre-smooth, coarse correction, post-smooth (two-grid V(1,1) cycle from x = 0)."
function newton_twogrid_precond!(x,b,s::KSol,kw,L::NewtonLines,C::NewtonCoarse;pre::Int=1,post::Int=1)
    for k in 1:4;fill!(x[k],0);end
    for n in 1:pre;newton_line_smooth!(x,b,s,kw,L;zero_start=n==1);end
    if pre==0
        for k in 1:4;copyto!(L.r[k],b[k]);end
    else
        newton_block_residual!(L,b,x,s,kw)
    end
    newton_coarse_correct!(x,L.r,C)
    for _ in 1:post;newton_line_smooth!(x,b,s,kw,L);end
    x
end


# ---- first-order Roe blocks -----------------------------------------------------------------------------
@inline function _newton_euler_jacobian(q,nx,ny,g1)
    B=eltype(q);r=q[1];u=q[2]/r;v=q[3]/r;q2=u*u+v*v
    p=g1*(q[4]-r*q2/2);H=(q[4]+p)/r;un=u*nx+v*ny;phi=g1*q2/2
    SMatrix{4,4,B,16}(
        0,           nx*phi-u*un,        ny*phi-v*un,        (phi-H)*un,
        nx,          un+(1-g1)*u*nx,     v*nx-g1*u*ny,       H*nx-g1*u*un,
        ny,          u*ny-g1*v*nx,       un+(1-g1)*v*ny,     H*ny-g1*v*un,
        0,           g1*nx,              g1*ny,              (1+g1)*un)
end

# |A| (unit normal) at the arithmetic-average state, eigenvalues floored at delta*(|un|+c).
@inline function _newton_roe_abs(ql,qr,nx,ny,g1,delta)
    B=eltype(ql)
    rl=ql[1];ul=ql[2]/rl;vl=ql[3]/rl;Hl=(ql[4]+g1*(ql[4]-rl*(ul*ul+vl*vl)/2))/rl
    rr=qr[1];ur=qr[2]/rr;vr=qr[3]/rr;Hr=(qr[4]+g1*(qr[4]-rr*(ur*ur+vr*vr)/2))/rr
    u=(ul+ur)/2;v=(vl+vr)/2;H=(Hl+Hr)/2;q2=u*u+v*v
    c=sqrt(max(g1*(H-q2/2),B(1e-12)));un=u*nx+v*ny
    floor=delta*(abs(un)+c)
    l1=max(abs(un),floor);l2=max(abs(un+c),floor);l3=max(abs(un-c),floor)
    a=(l2+l3)/2-l1;b=(l2-l3)/2
    R1=SVector{4,B}(1,u,v,H);N=SVector{4,B}(0,nx,ny,un)
    L1=SVector{4,B}(q2/2,-u,-v,1);Mv=SVector{4,B}(-un,nx,ny,0)
    l1*SMatrix{4,4,B,16}(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1)+
        R1*(a*g1/(c*c)*L1+b/c*Mv)'+N*(a*Mv+b*g1/c*L1)'
end

@inline function _newton_roe_face_jacobians(ql,qr,nxr,nyr,g1,delta)
    area=hypot(nxr,nyr);nx=nxr/area;ny=nyr/area
    ab=_newton_roe_abs(ql,qr,nx,ny,g1,delta)*area
    (_newton_euler_jacobian(ql,nxr,nyr,g1)+ab)/2,(_newton_euler_jacobian(qr,nxr,nyr,g1)-ab)/2
end

@kernel function _k_newton_roe_i_faces!(left,right,@Const(rho),@Const(ux),@Const(uy),@Const(T),
    @Const(Six),@Const(Siy),@Const(xc),@Const(yc),@Const(nut),
    @Const(gux),@Const(guy),@Const(gvx),@Const(gvy),@Const(gTx),@Const(gTy),
    drot,eint,nu_lam,rho_inf,suth,air_sutherland::Int32,sst_on::Int32,delta,NI::Int32,NJ::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    B=eltype(left)
    @inbounds if i<=NI+1&&j<=NJ
        ia=i+g;ja=j+g;g1=_gas_cs2(B)/eint
        ql=_airfoil_conserved4(rho[ia-1,ja],ux[ia-1,ja],uy[ia-1,ja],T[ia-1,ja],eint,B)
        qr=_airfoil_conserved4(rho[ia,ja],ux[ia,ja],uy[ia,ja],T[ia,ja],eint,B)
        a,b=_newton_roe_face_jacobians(ql,qr,Six[i,j],Siy[i,j],g1,delta)
        if 2<=i<=NI&&nu_lam>0
            gl=SVector{6,B}(gux[i-1,j],guy[i-1,j],gvx[i-1,j],gvy[i-1,j],gTx[i-1,j],gTy[i-1,j])
            gr=SVector{6,B}(gux[i,j],guy[i,j],gvx[i,j],gvy[i,j],gTx[i,j],gTy[i,j])
            mu,ke=_airfoil_face_transport(rho[ia-1,ja],rho[ia,ja],T[ia-1,ja],T[ia,ja],nut[ia-1,ja],nut[ia,ja],
                nu_lam,rho_inf,suth,air_sutherland,sst_on,drot,B)
            va,vb=_airfoil_fullvisc_face_jacobians(ql,qr,gl,gr,xc[i,j]-xc[i-1,j],yc[i,j]-yc[i-1,j],Six[i,j],Siy[i,j],mu,ke,eint)
            a+=va;b+=vb
        end
        _airfoil_block4_store!(left,a,i,j);_airfoil_block4_store!(right,b,i,j)
    end
end

@kernel function _k_newton_roe_j_faces!(left,right,@Const(rho),@Const(ux),@Const(uy),@Const(T),
    @Const(Sjx),@Const(Sjy),@Const(xc),@Const(yc),@Const(nut),
    @Const(gux),@Const(guy),@Const(gvx),@Const(gvy),@Const(gTx),@Const(gTy),
    drot,eint,nu_lam,rho_inf,suth,air_sutherland::Int32,sst_on::Int32,delta,
    NI::Int32,NJ::Int32,NIW::Int32,NIF::Int32,g::Int32)
    i,j=@index(Global,NTuple)
    B=eltype(left)
    @inbounds if i<=NI&&j<=NJ+1
        ia=i+g;ja=j+g;g1=_gas_cs2(B)/eint
        if j==1&&NIW<i<=NIW+NIF
            _airfoil_block4_store!(left,zero(SMatrix{4,4,B,16}),i,j)
            _airfoil_block4_store!(right,_airfoil_wall_pressure_jacobian(ux[ia,ja],uy[ia,ja],Sjx[i,j],Sjy[i,j],eint,B),i,j)
        else
            ql=_airfoil_conserved4(rho[ia,ja-1],ux[ia,ja-1],uy[ia,ja-1],T[ia,ja-1],eint,B)
            qr=_airfoil_conserved4(rho[ia,ja],ux[ia,ja],uy[ia,ja],T[ia,ja],eint,B)
            a,b=_newton_roe_face_jacobians(ql,qr,Sjx[i,j],Sjy[i,j],g1,delta)
            if 2<=j<=NJ&&nu_lam>0
                gl=SVector{6,B}(gux[i,j-1],guy[i,j-1],gvx[i,j-1],gvy[i,j-1],gTx[i,j-1],gTy[i,j-1])
                gr=SVector{6,B}(gux[i,j],guy[i,j],gvx[i,j],gvy[i,j],gTx[i,j],gTy[i,j])
                mu,ke=_airfoil_face_transport(rho[ia,ja-1],rho[ia,ja],T[ia,ja-1],T[ia,ja],nut[ia,ja-1],nut[ia,ja],
                    nu_lam,rho_inf,suth,air_sutherland,sst_on,drot,B)
                va,vb=_airfoil_fullvisc_face_jacobians(ql,qr,gl,gr,xc[i,j]-xc[i,j-1],yc[i,j]-yc[i,j-1],Sjx[i,j],Sjy[i,j],mu,ke,eint)
                a+=va;b+=vb
            end
            _airfoil_block4_store!(left,a,i,j);_airfoil_block4_store!(right,b,i,j)
        end
    end
end

"Assemble the 5-point block matrix with first-order Roe convection (same storage and viscous/boundary assembly as the kinetic build)."
function newton_roe_build!(s::KSol,w,d0;delta=0.1)
    b=s.backend;ni=Int32(s.NI);nj=Int32(s.NJ);g=Int32(s.g);B=eltype(w.diagonal)
    _k_newton_roe_i_faces!(b)(w.face_i_left,w.face_i_right,s.rho,s.ux,s.uy,s.T,s.Six,s.Siy,s.xc,s.yc,s.nut,
        s.gux,s.guy,s.gvx,s.gvy,s.gTx,s.gTy,s.drot,s.eint,s.nu_lam,s.rho_inf,s.suth,s.air_sutherland,Int32(s.sst),
        B(delta),ni,nj,g;ndrange=(s.NI+1,s.NJ))
    _k_newton_roe_j_faces!(b)(w.face_j_left,w.face_j_right,s.rho,s.ux,s.uy,s.T,s.Sjx,s.Sjy,s.xc,s.yc,s.nut,
        s.gux,s.guy,s.gvx,s.gvy,s.gTx,s.gTy,s.drot,s.eint,s.nu_lam,s.rho_inf,s.suth,s.air_sutherland,Int32(s.sst),
        B(delta),ni,nj,Int32(s.NIW),Int32(s.NIF),g;ndrange=(s.NI,s.NJ+1))
    _k_airfoil_kinetic_coeffs!(b)(w.lower_i,w.upper_i,w.lower_j,w.upper_j,w.diagonal,
        w.face_i_left,w.face_i_right,w.face_j_left,w.face_j_right,d0,
        s.rho,s.T,s.nut,s.Six,s.Siy,s.Sjx,s.Sjy,s.vol,
        s.nu_lam,s.rho_inf,s.suth,s.air_sutherland,Int32(s.sst),Int32(1),ni,nj,Int32(s.NIW),Int32(s.NIF),g;ndrange=(s.NI,s.NJ))
    nothing
end
