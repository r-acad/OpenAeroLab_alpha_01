# ==============================================================================
# Particles-on-Demand — SPARSE, SHAPE-HUGGING block selection  [roadmap 2.1]
# ==============================================================================
# The block-structured FV (pond_fv3d_blocks.jl) operates on a LIST of blocks coupled by a
# neighbour table — blocks not in the list simply do not exist or compute. The dense driver
# (pond_fv3d_blocks_driver.jl) instantiates ALL Px·Py·Pz blocks (a full cuboid). This module
# selects ONLY the blocks in a thin shell hugging the body surface (+ optional wake), like the
# production D3Q27 solver's sparse active-block mesh — so for a given resolution the body is
# wrapped in far fewer cells than a dense cuboid (the freestream corners are never instantiated).
#
# Selection: a block is SOLID if it contains any χ>0 cell; ACTIVE if it is within `shell` blocks
# (Chebyshev) of a solid block — a shape-hugging shell of uniform thickness on ALL sides — with an
# optional downstream (+x) `wake` extension within the body's transverse footprint. The sparse
# neighbour table maps each active block's 6 face neighbours to active indices (0 ⇒ that neighbour
# is not instantiated ⇒ the block FV's zero-gradient boundary halo, the same as a domain edge; in
# the full multi-level solver 2.1-B this 0 becomes a prolong from the coarse level).
# ==============================================================================

using KernelAbstractions

# per-block SOLID flag: true if the block contains any χ>0 cell.
function block_solid_flags(chi_mono, Px,Py,Pz, BS)
    solid = falses(Px,Py,Pz)
    @inbounds for bz in 1:Pz, by in 1:Py, bx in 1:Px
        f=false
        for lz in 1:BS, ly in 1:BS, lx in 1:BS
            if chi_mono[(bx-1)*BS+lx,(by-1)*BS+ly,(bz-1)*BS+lz] > 0f0; f=true; break; end
        end
        solid[bx,by,bz]=f
    end
    return solid
end

# shape-hugging selection: dilate the solid blocks by `shell` (Chebyshev) into a uniform shell,
# then extend `wake` blocks downstream (+x) within the body's (y,z) footprint. Returns a BitArray.
function select_shell_blocks(solid::AbstractArray{Bool,3}, Px,Py,Pz; shell::Int=2, wake::Int=0)
    active = falses(Px,Py,Pz)
    @inbounds for bz in 1:Pz, by in 1:Py, bx in 1:Px
        hit=false
        for dz in -shell:shell, dy in -shell:shell, dx in -shell:shell
            x=bx+dx; y=by+dy; z=bz+dz
            if 1<=x<=Px && 1<=y<=Py && 1<=z<=Pz && solid[x,y,z]; hit=true; break; end
        end
        active[bx,by,bz]=hit
    end
    if wake>0                                   # extend downstream within the body transverse span
        @inbounds for bz in 1:Pz, by in 1:Py, bx in 1:Px
            if solid[bx,by,bz]
                for w in 1:wake; x=bx+w; if x<=Px; active[x,by,bz]=true; end; end
            end
        end
    end
    return active
end

# assemble the sparse block set: active index map, coordinate list, 6-neighbour table (0=inactive),
# and per-block global origin. Reuses the block-FV neighbour convention [+x,−x,+y,−y,+z,−z].
function sparse_block_set(active::AbstractArray{Bool,3}, Px,Py,Pz, BS)
    id = zeros(Int, Px,Py,Pz); coords = NTuple{3,Int}[]
    @inbounds for bz in 1:Pz, by in 1:Py, bx in 1:Px
        if active[bx,by,bz]; push!(coords,(bx,by,bz)); id[bx,by,bz]=length(coords); end
    end
    nb=length(coords); nbr=zeros(Int32,nb,6); org=zeros(Int32,nb,3)
    act(x,y,z) = (1<=x<=Px && 1<=y<=Py && 1<=z<=Pz) ? id[x,y,z] : 0
    @inbounds for (b,(bx,by,bz)) in enumerate(coords)
        nbr[b,1]=Int32(act(bx+1,by,bz)); nbr[b,2]=Int32(act(bx-1,by,bz))
        nbr[b,3]=Int32(act(bx,by+1,bz)); nbr[b,4]=Int32(act(bx,by-1,bz))
        nbr[b,5]=Int32(act(bx,by,bz+1)); nbr[b,6]=Int32(act(bx,by,bz-1))
        org[b,1]=Int32((bx-1)*BS); org[b,2]=Int32((by-1)*BS); org[b,3]=Int32((bz-1)*BS)
    end
    return (id=id, coords=coords, nbr=nbr, org=org, nb=nb)
end

# pack a monolithic (Nx,Ny,Nz) field into padded blocks for ONLY the active coordinate list.
function pack_sparse(field, coords, BS, H)
    nb=length(coords); pad=BS+2H; out=zeros(Float32, pad,pad,pad, nb)
    @inbounds for (b,(bx,by,bz)) in enumerate(coords)
        for lz in 1:BS, ly in 1:BS, lx in 1:BS
            out[lx+H,ly+H,lz+H,b] = field[(bx-1)*BS+lx,(by-1)*BS+ly,(bz-1)*BS+lz]
        end
    end
    return out
end

# build a PonDBlocksDriver (same struct/advance! as the dense path) over a SPARSE active block set.
function pond_blocks_driver_sparse(backend, dev, Px,Py,Pz, BS, set; dx,dt,cs2, rho_inf,U_inf, chi_mono, eta, ramp, ng, cxv,cyv,czv,wv)
    H=POND_H; nb=set.nb; Nx,Ny,Nz = Px*BS,Py*BS,Pz*BS
    rho0=fill(Float32(rho_inf), Nx,Ny,Nz); zero3=zeros(Float32, Nx,Ny,Nz)
    r  = dev(pack_sparse(rho0, set.coords, BS,H))
    ux = dev(pack_sparse(zero3,set.coords, BS,H)); uy=dev(pack_sparse(zero3,set.coords, BS,H)); uz=dev(pack_sparse(zero3,set.coords, BS,H))
    ro=similar(r); mxo=similar(r); myo=similar(r); mzo=similar(r)
    chi = dev(pack_sparse(chi_mono, set.coords, BS,H))
    nbr = dev(set.nbr); org = dev(set.org)
    cxa=dev(Float32.(cxv)); cya=dev(Float32.(cyv)); cza=dev(Float32.(czv)); wa=dev(Float32.(wv))
    return PonDBlocksDriver{typeof(r),typeof(cxa),typeof(nbr)}(backend, r,ux,uy,uz, ro,mxo,myo,mzo, chi,
        nbr,org, cxa,cya,cza,wa, Px,Py,Pz,BS,nb,Nx,Ny,Nz, Float32(dx),Float32(dt),Float32(cs2),
        Float32(rho_inf),Float32(U_inf),Float32(eta), ramp, ng)
end

# ==============================================================================
# 2.1-B: multi-level coupling operators (coarse base ⇄ sparse fine shell), 2:1 refinement.
# The fine region begins at coarse cell C0 (1-based) so a fine cell of GLOBAL index gf maps to the continuous
# coarse coordinate  cc = (C0 − 0.75) + 0.5·gf  (cell-centred 2:1: fine cells 2m−1,2m sit under coarse C0+m−1).
# ==============================================================================
@inline function _tri1(A, x,y,z, nx,ny,nz)                 # trilinear sample of a coarse array at (x,y,z)
    xc = x<1f0 ? 1f0 : (x>Float32(nx) ? Float32(nx) : x)
    yc = y<1f0 ? 1f0 : (y>Float32(ny) ? Float32(ny) : y)
    zc = z<1f0 ? 1f0 : (z>Float32(nz) ? Float32(nz) : z)
    i0=floor(Int,xc);j0=floor(Int,yc);k0=floor(Int,zc); i1=min(i0+1,nx);j1=min(j0+1,ny);k1=min(k0+1,nz)
    fx=xc-i0;fy=yc-j0;fz=zc-k0; gx=1f0-fx;gy=1f0-fy;gz=1f0-fz
    @inbounds return gz*(gy*(gx*A[i0,j0,k0]+fx*A[i1,j0,k0])+fy*(gx*A[i0,j1,k0]+fx*A[i1,j1,k0])) +
                     fz*(gy*(gx*A[i0,j0,k1]+fx*A[i1,j0,k1])+fy*(gx*A[i0,j1,k1]+fx*A[i1,j1,k1]))
end
# 6-point quintic Lagrange weights (offsets −2..+3), 5th-order ⇒ reproduces polynomials to degree 5 exactly.
@inline function _ql6w(t::Float32)
    tm2=t+2f0;tm1=t+1f0;t0=t;tp1=t-1f0;tp2=t-2f0;tp3=t-3f0
    return ((tm1*t0*tp1*tp2*tp3)/(-120f0),(tm2*t0*tp1*tp2*tp3)/(24f0),(tm2*tm1*tp1*tp2*tp3)/(-12f0),
            (tm2*tm1*t0*tp2*tp3)/(12f0),(tm2*tm1*t0*tp1*tp3)/(-24f0),(tm2*tm1*t0*tp1*tp2)/(120f0))
end
@inline _cl(x,n)= x<1 ? 1 : (x>n ? n : x)
@inline function _quint1(A, x,y,z, nx,ny,nz)               # 6³ quintic-Lagrange sample of a coarse array
    xc = x<1f0 ? 1f0 : (x>Float32(nx) ? Float32(nx) : x)
    yc = y<1f0 ? 1f0 : (y>Float32(ny) ? Float32(ny) : y)
    zc = z<1f0 ? 1f0 : (z>Float32(nz) ? Float32(nz) : z)
    bx=floor(Int,xc);by=floor(Int,yc);bz=floor(Int,zc); wx=_ql6w(xc-bx);wy=_ql6w(yc-by);wz=_ql6w(zc-bz)
    s=0f0
    @inbounds for dk in 0:5
        kk=_cl(bz-2+dk,nz); wk=wz[dk+1]
        for dj in 0:5
            jj=_cl(by-2+dj,ny); wjk=wk*wy[dj+1]
            for di in 0:5
                ii=_cl(bx-2+di,nx); s+=wjk*wx[di+1]*A[ii,jj,kk]
            end
        end
    end
    return s
end
# ---- GENUINE C4: interpolating quintic B-SPLINE (degree-5 ⇒ C4 by construction) --------------------------
# Unlike quintic Lagrange (C0 at nodes), the quintic B-spline is C4. Interpolating form = a separable prefilter
# (2 poles, causal+anticausal recursion per axis) that turns the coarse macros into B-spline COEFFICIENTS, then
# a 6³ B-spline evaluation. Proven C4 (derivative jumps 0..4 ≈ 0) + quintic-exact + freestream-exact.
const _BS_Z1 = -0.430575347099973f0
const _BS_Z2 = -0.043096288203265f0
const _BS_GAIN = Float32((1-Float64(_BS_Z1))*(1-1/Float64(_BS_Z1))*(1-Float64(_BS_Z2))*(1-1/Float64(_BS_Z2)))
@inline function _qbw(w::Float32)                         # Thévenaz quintic B-spline weights, w∈[0,1)
    w2=w*w; W5=(1f0/120f0)*w*w2*w2
    w2-=w; w4=w2*w2; ww=w-0.5f0; t=w2*(w2-3f0)
    W0=(1f0/24f0)*(0.2f0+w2+w4)-W5
    t0=(1f0/24f0)*(w2*(w2-5f0)+9.2f0); t1=(-1f0/12f0)*ww*(t+4f0); W2=t0+t1; W3=t0-t1
    s0=(1f0/16f0)*(1.8f0-t); s1=(1f0/24f0)*ww*(w4-w2-5f0); W1=s0+s1; W4=s0-s1
    return (W0,W1,W2,W3,W4,W5)
end
# prefilter recursions inlined per axis (NO closures — GPU-safe). Each thread owns one line; 2 poles applied
# sequentially: causal (truncated-mirror init) + anticausal (mirror init).
@kernel function _prefilter_x!(c, N::Int32)
    j,k=@index(Global,NTuple); n=Int(N); h=min(n,30)
    @inbounds begin
        z=_BS_Z1
        zk=z; s=c[1,j,k]; for m in 2:h; s+=zk*c[m,j,k]; zk*=z; end; c[1,j,k]=s
        for i in 2:n; c[i,j,k]+=z*c[i-1,j,k]; end
        c[n,j,k]=(z/(z*z-1f0))*(z*c[n-1,j,k]+c[n,j,k]); for i in n-1:-1:1; c[i,j,k]=z*(c[i+1,j,k]-c[i,j,k]); end
        z=_BS_Z2
        zk=z; s=c[1,j,k]; for m in 2:h; s+=zk*c[m,j,k]; zk*=z; end; c[1,j,k]=s
        for i in 2:n; c[i,j,k]+=z*c[i-1,j,k]; end
        c[n,j,k]=(z/(z*z-1f0))*(z*c[n-1,j,k]+c[n,j,k]); for i in n-1:-1:1; c[i,j,k]=z*(c[i+1,j,k]-c[i,j,k]); end
    end
end
@kernel function _prefilter_y!(c, N::Int32)
    i,k=@index(Global,NTuple); n=Int(N); h=min(n,30)
    @inbounds begin
        z=_BS_Z1
        zk=z; s=c[i,1,k]; for m in 2:h; s+=zk*c[i,m,k]; zk*=z; end; c[i,1,k]=s
        for j in 2:n; c[i,j,k]+=z*c[i,j-1,k]; end
        c[i,n,k]=(z/(z*z-1f0))*(z*c[i,n-1,k]+c[i,n,k]); for j in n-1:-1:1; c[i,j,k]=z*(c[i,j+1,k]-c[i,j,k]); end
        z=_BS_Z2
        zk=z; s=c[i,1,k]; for m in 2:h; s+=zk*c[i,m,k]; zk*=z; end; c[i,1,k]=s
        for j in 2:n; c[i,j,k]+=z*c[i,j-1,k]; end
        c[i,n,k]=(z/(z*z-1f0))*(z*c[i,n-1,k]+c[i,n,k]); for j in n-1:-1:1; c[i,j,k]=z*(c[i,j+1,k]-c[i,j,k]); end
    end
end
@kernel function _prefilter_z!(c, N::Int32)
    i,j=@index(Global,NTuple); n=Int(N); h=min(n,30)
    @inbounds begin
        z=_BS_Z1
        zk=z; s=c[i,j,1]; for m in 2:h; s+=zk*c[i,j,m]; zk*=z; end; c[i,j,1]=s
        for k in 2:n; c[i,j,k]+=z*c[i,j,k-1]; end
        c[i,j,n]=(z/(z*z-1f0))*(z*c[i,j,n-1]+c[i,j,n]); for k in n-1:-1:1; c[i,j,k]=z*(c[i,j,k+1]-c[i,j,k]); end
        z=_BS_Z2
        zk=z; s=c[i,j,1]; for m in 2:h; s+=zk*c[i,j,m]; zk*=z; end; c[i,j,1]=s
        for k in 2:n; c[i,j,k]+=z*c[i,j,k-1]; end
        c[i,j,n]=(z/(z*z-1f0))*(z*c[i,j,n-1]+c[i,j,n]); for k in n-1:-1:1; c[i,j,k]=z*(c[i,j,k+1]-c[i,j,k]); end
    end
end
@inline function _bspline3(C, x,y,z, nx,ny,nz)           # 6³ interpolating quintic B-spline eval of coeffs C
    xc=x<1f0 ? 1f0 : (x>Float32(nx) ? Float32(nx) : x); yc=y<1f0 ? 1f0 : (y>Float32(ny) ? Float32(ny) : y); zc=z<1f0 ? 1f0 : (z>Float32(nz) ? Float32(nz) : z)
    ix=floor(Int,xc);iy=floor(Int,yc);iz=floor(Int,zc); wx=_qbw(xc-ix);wy=_qbw(yc-iy);wz=_qbw(zc-iz); s=0f0
    @inbounds for dk in 0:5
        kk=_cl(iz+dk-2,nz); wk=wz[dk+1]
        for dj in 0:5
            jj=_cl(iy+dj-2,ny); wjk=wk*wy[dj+1]
            for di in 0:5; ii=_cl(ix+di-2,nx); s+=wjk*wx[di+1]*C[ii,jj,kk]; end
        end
    end
    return s
end
# host helper: coarse macros (cr,cux,cuy,cuz) -> B-spline coefficient arrays (kr,kx,ky,kz). Copy·gain, then 3
# separable prefilter sweeps per field. Call once per step before the B-spline prolong.
function bspline_coeffs!(backend, coeff, coarse, Ncx,Ncy,Ncz)
    kr,kx,ky,kz=coeff; cr,cux,cuy,cuz=coarse
    for (dst,src) in ((kr,cr),(kx,cux),(ky,cuy),(kz,cuz))
        # separable: each axis is an independent 1D prefilter ⇒ apply the 1D gain ONCE PER AXIS (3× total)
        dst .= src .* _BS_GAIN
        _prefilter_x!(backend)(dst, Int32(Ncx); ndrange=(Ncy,Ncz)); KernelAbstractions.synchronize(backend)
        dst .*= _BS_GAIN
        _prefilter_y!(backend)(dst, Int32(Ncy); ndrange=(Ncx,Ncz)); KernelAbstractions.synchronize(backend)
        dst .*= _BS_GAIN
        _prefilter_z!(backend)(dst, Int32(Ncz); ndrange=(Ncx,Ncy)); KernelAbstractions.synchronize(backend)
    end
end

# PROLONG coarse→fine HALO: fill every padded HALO cell of every sparse block from the coarse level (2:1
# trilinear). Interior cells untouched. Faces with an ACTIVE neighbour are overwritten afterwards by the
# same-level halo exchange; faces with an INACTIVE neighbour (nbr=0) keep this coarse-sourced value ⇒ the
# shell's far-field comes from the coarse base instead of a spurious zero-gradient.
@kernel function prolong_coarse_halo!(fr,fux,fuy,fuz, @Const(cr),@Const(cux),@Const(cuy),@Const(cuz),
        @Const(org), off::Float32, Ncx::Int32,Ncy::Int32,Ncz::Int32, BS::Int32, H::Int32)
    i,j,k,b=@index(Global,NTuple); bs=Int(BS); h=Int(H)
    @inbounds begin
        ishalo = i<=h || i>bs+h || j<=h || j>bs+h || k<=h || k>bs+h
        if ishalo
            ccx=off+0.5f0*Float32(org[b,1]+(i-h)); ccy=off+0.5f0*Float32(org[b,2]+(j-h)); ccz=off+0.5f0*Float32(org[b,3]+(k-h))
            nx=Int(Ncx);ny=Int(Ncy);nz=Int(Ncz)      # roadmap 1.4-class: 6³ quintic (5th-order) coarse→fine halo
            fr[i,j,k,b]=_quint1(cr,ccx,ccy,ccz,nx,ny,nz)
            fux[i,j,k,b]=_quint1(cux,ccx,ccy,ccz,nx,ny,nz)
            fuy[i,j,k,b]=_quint1(cuy,ccx,ccy,ccz,nx,ny,nz)
            fuz[i,j,k,b]=_quint1(cuz,ccx,ccy,ccz,nx,ny,nz)
        end
    end
end
# GENUINE C4 sibling: identical halo fill but evaluates the interpolating quintic B-SPLINE on the PREFILTERED
# coefficient arrays (kr,kx,ky,kz from bspline_coeffs!) ⇒ C4 continuity at the AMR interface.
@kernel function prolong_coarse_halo_bs!(fr,fux,fuy,fuz, @Const(kr),@Const(kx),@Const(ky),@Const(kz),
        @Const(org), off::Float32, Ncx::Int32,Ncy::Int32,Ncz::Int32, BS::Int32, H::Int32)
    i,j,k,b=@index(Global,NTuple); bs=Int(BS); h=Int(H)
    @inbounds begin
        ishalo = i<=h || i>bs+h || j<=h || j>bs+h || k<=h || k>bs+h
        if ishalo
            ccx=off+0.5f0*Float32(org[b,1]+(i-h)); ccy=off+0.5f0*Float32(org[b,2]+(j-h)); ccz=off+0.5f0*Float32(org[b,3]+(k-h))
            nx=Int(Ncx);ny=Int(Ncy);nz=Int(Ncz)
            fr[i,j,k,b]=_bspline3(kr,ccx,ccy,ccz,nx,ny,nz)
            fux[i,j,k,b]=_bspline3(kx,ccx,ccy,ccz,nx,ny,nz)
            fuy[i,j,k,b]=_bspline3(ky,ccx,ccy,ccz,nx,ny,nz)
            fuz[i,j,k,b]=_bspline3(kz,ccx,ccy,ccz,nx,ny,nz)
        end
    end
end
# RESTRICT fine→coarse: each coarse cell covered by a sparse fine block ← conservative average of its 8 fine
# children (ρ and ρu averaged ⇒ mass+momentum consistent, uniform flow preserved). Indexed over the BS/2 coarse
# cells per block per axis. C0 = the coarse cell where the fine region begins (1-based).
@kernel function restrict_to_coarse!(cr,cux,cuy,cuz, @Const(fr),@Const(fux),@Const(fuy),@Const(fuz),
        @Const(org), C0::Int32, BS::Int32, H::Int32)
    cx,cy,cz,b=@index(Global,NTuple); h=Int(H); c0=Int(C0)
    @inbounds begin
        gcx=c0 + (org[b,1]>>1) + (cx-1); gcy=c0 + (org[b,2]>>1) + (cy-1); gcz=c0 + (org[b,3]>>1) + (cz-1)
        fi=2*cx-1+h; fj=2*cy-1+h; fk=2*cz-1+h                # first fine child (padded)
        sr=0f0; smx=0f0; smy=0f0; smz=0f0
        for dk in 0:1, dj in 0:1, di in 0:1
            r=fr[fi+di,fj+dj,fk+dk,b]; sr+=r; smx+=r*fux[fi+di,fj+dj,fk+dk,b]; smy+=r*fuy[fi+di,fj+dj,fk+dk,b]; smz+=r*fuz[fi+di,fj+dj,fk+dk,b]
        end
        r8=sr*0.125f0; cr[gcx,gcy,gcz]=r8; cux[gcx,gcy,gcz]=smx*0.125f0/r8; cuy[gcx,gcy,gcz]=smy*0.125f0/r8; cuz[gcx,gcy,gcz]=smz*0.125f0/r8
    end
end

# ---- Roadmap 2.2 (3D port): SDF ghost-cell curved wall for the sparse BLOCK FV (removes the voxel staircase).
# Per padded block, near-surface SOLID cells (−band<φ≤0) are set to the no-slip reflection of the fluid at the
# IMAGE point (cell + 2|φ|·n̂ in block-local coords, trilinear-sampled from the padded block); deep interior
# inert. SDF/normals are packed padded per block (pack_sparse). Run in place on the fine state each step.
@inline function _tlblk(A, x,y,z, b, pad)
    xc=x<1f0 ? 1f0 : (x>Float32(pad) ? Float32(pad) : x); yc=y<1f0 ? 1f0 : (y>Float32(pad) ? Float32(pad) : y); zc=z<1f0 ? 1f0 : (z>Float32(pad) ? Float32(pad) : z)
    i0=floor(Int,xc);j0=floor(Int,yc);k0=floor(Int,zc); i1=min(i0+1,pad);j1=min(j0+1,pad);k1=min(k0+1,pad)
    fx=xc-i0;fy=yc-j0;fz=zc-k0; gx=1f0-fx;gy=1f0-fy;gz=1f0-fz
    @inbounds return gz*(gy*(gx*A[i0,j0,k0,b]+fx*A[i1,j0,k0,b])+fy*(gx*A[i0,j1,k0,b]+fx*A[i1,j1,k0,b])) +
                     fz*(gy*(gx*A[i0,j0,k1,b]+fx*A[i1,j0,k1,b])+fy*(gx*A[i0,j1,k1,b]+fx*A[i1,j1,k1,b]))
end
@kernel function ghost_wall3_blk!(rho,ux,uy,uz, @Const(sdf),@Const(wnx),@Const(wny),@Const(wnz), band::Float32, slip::Int32, BS::Int32,H::Int32)
    lx,ly,lz,b=@index(Global,NTuple); h=Int(H); pad=Int(BS)+2h; i=lx+h;j=ly+h;k=lz+h
    @inbounds begin
        φ=sdf[i,j,k,b]
        if φ<=0f0 && φ>-band
            n1=wnx[i,j,k,b];n2=wny[i,j,k,b];n3=wnz[i,j,k,b]; d=-φ
            xi=Float32(i)+2f0*d*n1; yi=Float32(j)+2f0*d*n2; zi=Float32(k)+2f0*d*n3
            uI=_tlblk(ux,xi,yi,zi,b,pad); vI=_tlblk(uy,xi,yi,zi,b,pad); wI=_tlblk(uz,xi,yi,zi,b,pad)
            rho[i,j,k,b]=_tlblk(rho,xi,yi,zi,b,pad)
            if slip==Int32(1)                                        # SLIP: flip only the wall-normal component
                un=uI*n1+vI*n2+wI*n3
                ux[i,j,k,b]=uI-2f0*un*n1; uy[i,j,k,b]=vI-2f0*un*n2; uz[i,j,k,b]=wI-2f0*un*n3
            else                                                     # NO-SLIP: mirror the full velocity (u=0 at wall)
                ux[i,j,k,b]=-uI; uy[i,j,k,b]=-vI; uz[i,j,k,b]=-wI
            end
        elseif φ<=-band
            ux[i,j,k,b]=0f0; uy[i,j,k,b]=0f0; uz[i,j,k,b]=0f0; rho[i,j,k,b]=1f0
        end
    end
end

# sparse halo exchange: overwrite ONLY the active-neighbour halos (inactive faces keep the coarse-prolonged
# value). Same face convention as the dense halo, but no zero-gradient else-branch.
@kernel function halo_x_keep!(rho,ux,uy,uz, @Const(nbr), BS::Int32, H::Int32)
    jt,kt,b=@index(Global,NTuple); bs=Int(BS); h=Int(H)
    @inbounds begin
        jj=jt+h; kk=kt+h; np=nbr[b,1]; nm=nbr[b,2]
        if np>0; for s in 1:h; di=bs+h+s; si=h+s; rho[di,jj,kk,b]=rho[si,jj,kk,np];ux[di,jj,kk,b]=ux[si,jj,kk,np];uy[di,jj,kk,b]=uy[si,jj,kk,np];uz[di,jj,kk,b]=uz[si,jj,kk,np]; end; end
        if nm>0; for s in 1:h; di=h+1-s; si=bs+h+1-s; rho[di,jj,kk,b]=rho[si,jj,kk,nm];ux[di,jj,kk,b]=ux[si,jj,kk,nm];uy[di,jj,kk,b]=uy[si,jj,kk,nm];uz[di,jj,kk,b]=uz[si,jj,kk,nm]; end; end
    end
end
@kernel function halo_y_keep!(rho,ux,uy,uz, @Const(nbr), BS::Int32, H::Int32)
    it,kt,b=@index(Global,NTuple); bs=Int(BS); h=Int(H)
    @inbounds begin
        ii=it+h; kk=kt+h; np=nbr[b,3]; nm=nbr[b,4]
        if np>0; for s in 1:h; dj=bs+h+s; sj=h+s; rho[ii,dj,kk,b]=rho[ii,sj,kk,np];ux[ii,dj,kk,b]=ux[ii,sj,kk,np];uy[ii,dj,kk,b]=uy[ii,sj,kk,np];uz[ii,dj,kk,b]=uz[ii,sj,kk,np]; end; end
        if nm>0; for s in 1:h; dj=h+1-s; sj=bs+h+1-s; rho[ii,dj,kk,b]=rho[ii,sj,kk,nm];ux[ii,dj,kk,b]=ux[ii,sj,kk,nm];uy[ii,dj,kk,b]=uy[ii,sj,kk,nm];uz[ii,dj,kk,b]=uz[ii,sj,kk,nm]; end; end
    end
end
@kernel function halo_z_keep!(rho,ux,uy,uz, @Const(nbr), BS::Int32, H::Int32)
    it,jt,b=@index(Global,NTuple); bs=Int(BS); h=Int(H)
    @inbounds begin
        ii=it+h; jj=jt+h; np=nbr[b,5]; nm=nbr[b,6]
        if np>0; for s in 1:h; dk=bs+h+s; sk=h+s; rho[ii,jj,dk,b]=rho[ii,jj,sk,np];ux[ii,jj,dk,b]=ux[ii,jj,sk,np];uy[ii,jj,dk,b]=uy[ii,jj,sk,np];uz[ii,jj,dk,b]=uz[ii,jj,sk,np]; end; end
        if nm>0; for s in 1:h; dk=h+1-s; sk=bs+h+1-s; rho[ii,jj,dk,b]=rho[ii,jj,sk,nm];ux[ii,jj,dk,b]=ux[ii,jj,sk,nm];uy[ii,jj,dk,b]=uy[ii,jj,sk,nm];uz[ii,jj,dk,b]=uz[ii,jj,sk,nm]; end; end
    end
end

# ---- C3 temporal (sub-cycling): 4-anchor CUBIC Lagrange in time (reproduces cubic-in-time ⇒ C3), replacing
# the linear (C0) halo interpolation. Anchors = coarse states at t=−2,−1,0,+1 (coarse steps); eval at τ∈[0,1]
# (the newest interval). One field per call (keeps the arg count sane); weights precomputed on host via _ctw.
@inline function _ctw(τ::Float32)                        # cubic Lagrange weights for anchors (−2,−1,0,1) at τ
    a=τ+2f0; b=τ+1f0; c=τ; d=τ-1f0
    return ((b*c*d)*(-1f0/6f0), (a*c*d)*(0.5f0), (a*b*d)*(-0.5f0), (a*b*c)*(1f0/6f0))
end
@kernel function clerp1!(o, @Const(am2),@Const(am1),@Const(a0),@Const(a1), w2::Float32,w1::Float32,w0::Float32,wp::Float32)
    i,j,k=@index(Global,NTuple)
    @inbounds o[i,j,k]=w2*am2[i,j,k]+w1*am1[i,j,k]+w0*a0[i,j,k]+wp*a1[i,j,k]
end

# ---- multi-level chaining (roadmap 2.1, ≥4 levels): a sparse level's data is mirrored into a MONOLITHIC
# "backing" array over its region so the next-finer level can sample it with the same monolithic→sparse
# prolong. unpack4! scatters sparse blocks → backing; pack4! gathers backing → sparse blocks (after restrict).
@kernel function unpack4!(mr,mx,my,mz, @Const(br),@Const(bux),@Const(buy),@Const(buz), @Const(org), BS::Int32,H::Int32)
    lx,ly,lz,b=@index(Global,NTuple); h=Int(H)
    @inbounds begin
        gi=org[b,1]+lx; gj=org[b,2]+ly; gk=org[b,3]+lz
        mr[gi,gj,gk]=br[lx+h,ly+h,lz+h,b]; mx[gi,gj,gk]=bux[lx+h,ly+h,lz+h,b]; my[gi,gj,gk]=buy[lx+h,ly+h,lz+h,b]; mz[gi,gj,gk]=buz[lx+h,ly+h,lz+h,b]
    end
end
@kernel function pack4!(br,bux,buy,buz, @Const(mr),@Const(mx),@Const(my),@Const(mz), @Const(org), BS::Int32,H::Int32)
    lx,ly,lz,b=@index(Global,NTuple); h=Int(H)
    @inbounds begin
        gi=org[b,1]+lx; gj=org[b,2]+ly; gk=org[b,3]+lz
        br[lx+h,ly+h,lz+h,b]=mr[gi,gj,gk]; bux[lx+h,ly+h,lz+h,b]=mx[gi,gj,gk]; buy[lx+h,ly+h,lz+h,b]=my[gi,gj,gk]; buz[lx+h,ly+h,lz+h,b]=mz[gi,gj,gk]
    end
end

# one COUPLED fine step: prolong coarse→halos, then active-neighbour halo exchange (inactive faces keep the
# coarse value), then the 3 per-axis block flux kernels. `st` is the OLD fine state, `out` the new (ping-pong).
function pond_fv3d_sparse_step!(backend, out, st, nbr, org, coarse, off::Float32, Ncx,Ncy,Ncz,
                                cx,cy,cz,w, cs2::Float32, dt::Float32, dx::Float32, BS::Integer, nb::Integer)
    a=dt/dx; h=POND_H; bsi=Int32(BS); hi=Int32(h); pad=BS+2h
    ro,mxo,myo,mzo=out; r,uxs,uys,uzs=st; cr,cux,cuy,cuz=coarse
    prolong_coarse_halo!(backend)(r,uxs,uys,uzs, cr,cux,cuy,cuz, org, off, Int32(Ncx),Int32(Ncy),Int32(Ncz), bsi,hi; ndrange=(pad,pad,pad,Int(nb)))
    KernelAbstractions.synchronize(backend)
    tang=(Int(BS),Int(BS),Int(nb))
    halo_x_keep!(backend)(r,uxs,uys,uzs, nbr, bsi,hi; ndrange=tang)
    halo_y_keep!(backend)(r,uxs,uys,uzs, nbr, bsi,hi; ndrange=tang)
    halo_z_keep!(backend)(r,uxs,uys,uzs, nbr, bsi,hi; ndrange=tang)
    KernelAbstractions.synchronize(backend)
    full=(Int(BS),Int(BS),Int(BS),Int(nb))
    pond_block_flux_x!(backend)(ro,mxo,myo,mzo, r,uxs,uys,uzs, cx,cy,cz,w, cs2, Float32(a), hi; ndrange=full)
    KernelAbstractions.synchronize(backend)
    pond_block_flux_y!(backend)(ro,mxo,myo,mzo, r,uxs,uys,uzs, cx,cy,cz,w, cs2, Float32(a), hi; ndrange=full)
    KernelAbstractions.synchronize(backend)
    pond_block_flux_z!(backend)(ro,mxo,myo,mzo, r,uxs,uys,uzs, cx,cy,cz,w, cs2, Float32(a), hi; ndrange=full)
    KernelAbstractions.synchronize(backend)
    return nothing
end

# GENUINE C4 variant: identical to pond_fv3d_sparse_step! but the halo is prolonged with the quintic B-SPLINE
# from the PREFILTERED coefficient arrays `coeff` (call bspline_coeffs!(coarse)->coeff first). C4 AMR interface.
function pond_fv3d_sparse_step_bs!(backend, out, st, nbr, org, coeff, off::Float32, Ncx,Ncy,Ncz,
                                   cx,cy,cz,w, cs2::Float32, dt::Float32, dx::Float32, BS::Integer, nb::Integer)
    a=dt/dx; h=POND_H; bsi=Int32(BS); hi=Int32(h); pad=BS+2h
    ro,mxo,myo,mzo=out; r,uxs,uys,uzs=st; kr,kx,ky,kz=coeff
    prolong_coarse_halo_bs!(backend)(r,uxs,uys,uzs, kr,kx,ky,kz, org, off, Int32(Ncx),Int32(Ncy),Int32(Ncz), bsi,hi; ndrange=(pad,pad,pad,Int(nb)))
    KernelAbstractions.synchronize(backend)
    tang=(Int(BS),Int(BS),Int(nb))
    halo_x_keep!(backend)(r,uxs,uys,uzs, nbr, bsi,hi; ndrange=tang)
    halo_y_keep!(backend)(r,uxs,uys,uzs, nbr, bsi,hi; ndrange=tang)
    halo_z_keep!(backend)(r,uxs,uys,uzs, nbr, bsi,hi; ndrange=tang)
    KernelAbstractions.synchronize(backend)
    full=(Int(BS),Int(BS),Int(BS),Int(nb))
    pond_block_flux_x!(backend)(ro,mxo,myo,mzo, r,uxs,uys,uzs, cx,cy,cz,w, cs2, Float32(a), hi; ndrange=full)
    KernelAbstractions.synchronize(backend)
    pond_block_flux_y!(backend)(ro,mxo,myo,mzo, r,uxs,uys,uzs, cx,cy,cz,w, cs2, Float32(a), hi; ndrange=full)
    KernelAbstractions.synchronize(backend)
    pond_block_flux_z!(backend)(ro,mxo,myo,mzo, r,uxs,uys,uzs, cx,cy,cz,w, cs2, Float32(a), hi; ndrange=full)
    KernelAbstractions.synchronize(backend)
    return nothing
end
