# ==============================================================================
# Particles-on-Demand — BLOCK-STRUCTURED sphere driver  [production-layout path]
# ==============================================================================
# Runs a full supersonic sphere case on the PRODUCTION-style block layout (padded
# (BS+2H)³ × n_blocks arrays + neighbour table), i.e. the validated cross-block
# halo-exchange kinetic-FV (pond_fv3d_blocks_step!) rather than the monolithic array.
# This is the bridge to the real solver: once this reproduces the monolithic bow shock
# on the block decomposition, plugging into the actual BlockLevel is "just" sourcing the
# block fields from BlockLevel.rho/vel (via neighbour_table) instead of a config-built grid.
#
# Block-aware boundaries operate on the padded interior cells via each block's GLOBAL
# origin: Dirichlet inlet (−x, ramped), freestream far-field (y,z), penalization body;
# the supersonic outlet (+x) is the halo's zero-gradient at the domain boundary (automatic).
# ==============================================================================

using KernelAbstractions

# pack a monolithic (Nx,Ny,Nz) scalar field into padded blocks (interior local 1:BS → +H)
function _blk_pack(field, Px,Py,Pz, BS, H)
    nb = Px*Py*Pz; pad = BS+2H
    out = zeros(Float32, pad,pad,pad,nb)
    bidx(bx,by,bz) = bx + (by-1)*Px + (bz-1)*Px*Py
    @inbounds for bz in 1:Pz, by in 1:Py, bx in 1:Px
        b = bidx(bx,by,bz)
        for lz in 1:BS, ly in 1:BS, lx in 1:BS
            out[lx+H,ly+H,lz+H,b] = field[(bx-1)*BS+lx, (by-1)*BS+ly, (bz-1)*BS+lz]
        end
    end
    return out
end
# unpack padded blocks → monolithic (Nx,Ny,Nz)
function _blk_unpack(padded, Px,Py,Pz, BS, H)
    Nx,Ny,Nz = Px*BS, Py*BS, Pz*BS
    out = zeros(Float32, Nx,Ny,Nz)
    bidx(bx,by,bz) = bx + (by-1)*Px + (bz-1)*Px*Py
    @inbounds for bz in 1:Pz, by in 1:Py, bx in 1:Px
        b = bidx(bx,by,bz)
        for lz in 1:BS, ly in 1:BS, lx in 1:BS
            out[(bx-1)*BS+lx,(by-1)*BS+ly,(bz-1)*BS+lz] = padded[lx+H,ly+H,lz+H,b]
        end
    end
    return out
end
# 6-neighbour table [+x,-x,+y,-y,+z,-z] (0 = domain boundary) for a dense Px×Py×Pz block grid
function _blk_nbr(Px,Py,Pz)
    nb = Px*Py*Pz; nbr = zeros(Int32, nb, 6)
    bidx(bx,by,bz) = bx + (by-1)*Px + (bz-1)*Px*Py
    for bz in 1:Pz, by in 1:Py, bx in 1:Px
        b = bidx(bx,by,bz)
        nbr[b,1] = bx<Px ? bidx(bx+1,by,bz) : 0; nbr[b,2] = bx>1 ? bidx(bx-1,by,bz) : 0
        nbr[b,3] = by<Py ? bidx(bx,by+1,bz) : 0; nbr[b,4] = by>1 ? bidx(bx,by-1,bz) : 0
        nbr[b,5] = bz<Pz ? bidx(bx,by,bz+1) : 0; nbr[b,6] = bz>1 ? bidx(bx,by,bz-1) : 0
    end
    return nbr
end
# per-block global origin (gi0,gj0,gk0) so kernels can recover global coords
function _blk_origins(Px,Py,Pz, BS)
    nb = Px*Py*Pz; org = zeros(Int32, nb, 3)
    bidx(bx,by,bz) = bx + (by-1)*Px + (bz-1)*Px*Py
    for bz in 1:Pz, by in 1:Py, bx in 1:Px
        b = bidx(bx,by,bz); org[b,1]=(bx-1)*BS; org[b,2]=(by-1)*BS; org[b,3]=(bz-1)*BS
    end
    return org
end

# ---- Dirichlet inlet (−x): global gi ≤ ng ----
@kernel function _blk_inlet!(rho,ux,uy,uz, @Const(org), rho_inf::Float32, Uin::Float32, ng::Int32, H::Int32)
    lx,ly,lz,b = @index(Global, NTuple)
    h=Int(H)
    @inbounds if org[b,1]+lx <= ng
        i,j,k = lx+h, ly+h, lz+h
        rho[i,j,k,b]=rho_inf; ux[i,j,k,b]=Uin; uy[i,j,k,b]=0f0; uz[i,j,k,b]=0f0
    end
end
# ---- freestream far-field (y,z faces) ----
@kernel function _blk_farfield!(rho,ux,uy,uz, @Const(org), rho_inf::Float32, Uin::Float32,
                                Ny::Int32, Nz::Int32, ng::Int32, H::Int32)
    lx,ly,lz,b = @index(Global, NTuple)
    h=Int(H); ny=Int(Ny); nz=Int(Nz)
    gj = org[b,2]+ly; gk = org[b,3]+lz
    @inbounds if gj<=ng || gj>ny-ng || gk<=ng || gk>nz-ng
        i,j,k = lx+h, ly+h, lz+h
        rho[i,j,k,b]=rho_inf; ux[i,j,k,b]=Uin; uy[i,j,k,b]=0f0; uz[i,j,k,b]=0f0
    end
end
# ---- volume penalization (body): u ← u/(1+dt·χ/η) ----
@kernel function _blk_penalize!(ux,uy,uz, @Const(chi), invscale::Float32, H::Int32)
    lx,ly,lz,b = @index(Global, NTuple)
    h=Int(H); i,j,k = lx+h, ly+h, lz+h
    @inbounds begin
        c = chi[i,j,k,b]
        if c > 0f0
            f = 1f0/(1f0+invscale*c); ux[i,j,k,b]*=f; uy[i,j,k,b]*=f; uz[i,j,k,b]*=f
        end
    end
end

mutable struct PonDBlocksDriver{A,V,I}
    backend; r::A; ux::A; uy::A; uz::A; ro::A; mxo::A; myo::A; mzo::A; chi::A
    nbr::I; org::I; cxa::V; cya::V; cza::V; wa::V
    Px::Int; Py::Int; Pz::Int; BS::Int; nb::Int; Nx::Int; Ny::Int; Nz::Int
    dx::Float32; dt::Float32; cs2::Float32; rho_inf::Float32; U_inf::Float32; eta::Float32
    ramp::Int; ng::Int
end

# build the block driver from a monolithic IC (ρ=rho_inf, u=0) + a monolithic χ mask
function pond_blocks_driver(backend, dev, Px,Py,Pz, BS; dx,dt,cs2, rho_inf,U_inf, chi_mono, eta, ramp, ng, cxv,cyv,czv,wv)
    H = POND_H; nb=Px*Py*Pz; Nx,Ny,Nz = Px*BS,Py*BS,Pz*BS
    rho0 = fill(Float32(rho_inf), Nx,Ny,Nz); zero3 = zeros(Float32, Nx,Ny,Nz)
    r  = dev(_blk_pack(rho0, Px,Py,Pz,BS,H))
    ux = dev(_blk_pack(zero3,Px,Py,Pz,BS,H)); uy=dev(_blk_pack(zero3,Px,Py,Pz,BS,H)); uz=dev(_blk_pack(zero3,Px,Py,Pz,BS,H))
    ro=similar(r); mxo=similar(r); myo=similar(r); mzo=similar(r)
    chi = dev(_blk_pack(chi_mono, Px,Py,Pz,BS,H))
    nbr = dev(_blk_nbr(Px,Py,Pz)); org = dev(_blk_origins(Px,Py,Pz,BS))
    cxa=dev(Float32.(cxv)); cya=dev(Float32.(cyv)); cza=dev(Float32.(czv)); wa=dev(Float32.(wv))
    return PonDBlocksDriver{typeof(r),typeof(cxa),typeof(nbr)}(backend, r,ux,uy,uz, ro,mxo,myo,mzo, chi,
        nbr,org, cxa,cya,cza,wa, Px,Py,Pz,BS,nb,Nx,Ny,Nz, Float32(dx),Float32(dt),Float32(cs2),
        Float32(rho_inf),Float32(U_inf),Float32(eta), ramp, ng)
end

function pond_blocks_advance!(d::PonDBlocksDriver, t::Int)
    H = Int32(POND_H); nd = (d.BS,d.BS,d.BS,d.nb)
    # conservative FV step over the block decomposition (halo-exchange handles cross-block)
    pond_fv3d_blocks_step!(d.backend, (d.ro,d.mxo,d.myo,d.mzo), (d.r,d.ux,d.uy,d.uz), d.nbr,
                           d.cxa,d.cya,d.cza,d.wa, d.cs2,d.dt,d.dx, d.BS, d.nb)
    d.r,d.ro = d.ro,d.r; d.ux,d.mxo = d.mxo,d.ux; d.uy,d.myo = d.myo,d.uy; d.uz,d.mzo = d.mzo,d.uz
    # penalization body
    _blk_penalize!(d.backend)(d.ux,d.uy,d.uz, d.chi, Float32(d.dt/d.eta), H; ndrange=nd)
    KernelAbstractions.synchronize(d.backend)
    # ramped Dirichlet inlet + freestream far-field (supersonic outlet = halo zero-gradient)
    frac = d.ramp<=0 ? 1f0 : Float32(min(1.0, t/d.ramp)); Uin = d.U_inf*(0.5f0-0.5f0*cos(Float32(pi)*frac))
    _blk_inlet!(d.backend)(d.r,d.ux,d.uy,d.uz, d.org, d.rho_inf, Uin, Int32(d.ng), H; ndrange=nd)
    KernelAbstractions.synchronize(d.backend)
    _blk_farfield!(d.backend)(d.r,d.ux,d.uy,d.uz, d.org, d.rho_inf, Uin, Int32(d.Ny), Int32(d.Nz), Int32(d.ng), H; ndrange=nd)
    KernelAbstractions.synchronize(d.backend)
    return Uin
end

# pull the density field back to a monolithic array (for slices / metrics)
pond_blocks_rho(d::PonDBlocksDriver) = _blk_unpack(Array(d.r), d.Px,d.Py,d.Pz,d.BS,POND_H)
