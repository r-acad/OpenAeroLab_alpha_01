# ==============================================================================
# Particles-on-Demand — block-structured cross-block conservative kinetic-FV  [step 4]
# ==============================================================================
# Ports the validated split per-axis kinetic-FV (pond_fv3d_gpu.jl) onto the
# production block-structured layout (fields are (BS+2H)³ × n_blocks padded arrays,
# H=2 halo). Cross-block coupling is a HALO EXCHANGE: before each step, every block's
# 2-cell face halos are filled from its FACE neighbours (via a neighbour table), so the
# per-axis flux kernels read neighbour-block state through the halo and never branch on
# block topology. Because each shared block face is reconstructed from the SAME two
# cells on both sides, the scheme is exactly conservative across block boundaries — the
# block-decomposed result reproduces the monolithic single-array result to round-off.
#
# AMR 2:1 interfaces (future): the halo fill becomes a restrict (fine→coarse) / prolong
# (coarse→fine), and the shared-flux conservation across the jump is enforced by summing
# the 4 fine face fluxes = the coarse face flux. The uniform (same-level) cross-block
# coupling here is the foundation; the AMR flux-matching layers on top of it.
#
# Neighbour table convention (this module, 6 face neighbours per block):
#   nbr[b, 1]=+x  nbr[b, 2]=-x  nbr[b, 3]=+y  nbr[b, 4]=-y  nbr[b, 5]=+z  nbr[b, 6]=-z
#   value = 1-based block index, or 0 = domain boundary (⇒ zero-gradient halo, matching
#   the monolithic kernel's clamp). Production maps these 6 from its 27-stencil table:
#   dir=(dx+1)+(dy+1)*3+(dz+1)*9+1 ⇒ +x=15 -x=13 +y=17 -y=11 +z=23 -z=5.
# ==============================================================================

using KernelAbstractions

const POND_H = 2                 # halo thickness (MUSCL ±2 stencil)

# ---- halo exchange along x: fill the low/high x-halo slabs of every block ----
# Padded layout: interior local cell l∈1:BS sits at padded index l+H. Halo low = 1:H,
# halo high = BS+H+1 : BS+2H. Tangential (y,z) range is the interior H+1 : H+BS.
@kernel function pond_halo_x!(rho, ux, uy, uz, @Const(nbr), BS::Int32, H::Int32)
    jt, kt, b = @index(Global, NTuple)          # tangential interior indices 1:BS, block b
    bs = Int(BS); h = Int(H)
    @inbounds begin
        jj = jt + h; kk = kt + h
        np = nbr[b, 1]; nm = nbr[b, 2]           # +x, -x neighbours
        # high-x halo (cells bs+h+1 .. bs+2h) ← neighbour +x interior 1..h (padded h+1..2h)
        for s in 1:h
            di = bs + h + s
            if np > 0
                si = h + s                       # neighbour interior cell s ⇒ padded h+s
                rho[di,jj,kk,b]=rho[si,jj,kk,np]; ux[di,jj,kk,b]=ux[si,jj,kk,np]
                uy[di,jj,kk,b]=uy[si,jj,kk,np];   uz[di,jj,kk,b]=uz[si,jj,kk,np]
            else                                 # domain boundary ⇒ replicate own edge (zero-grad)
                ei = bs + h
                rho[di,jj,kk,b]=rho[ei,jj,kk,b]; ux[di,jj,kk,b]=ux[ei,jj,kk,b]
                uy[di,jj,kk,b]=uy[ei,jj,kk,b];   uz[di,jj,kk,b]=uz[ei,jj,kk,b]
            end
        end
        # low-x halo (cells 1..h) ← neighbour -x interior bs-h+1..bs (padded bs..bs+h-1... )
        for s in 1:h
            di = h + 1 - s                        # h, h-1, ..., 1
            if nm > 0
                si = bs + h + 1 - s               # neighbour interior cell bs+1-s ⇒ padded bs+h+1-s
                rho[di,jj,kk,b]=rho[si,jj,kk,nm]; ux[di,jj,kk,b]=ux[si,jj,kk,nm]
                uy[di,jj,kk,b]=uy[si,jj,kk,nm];   uz[di,jj,kk,b]=uz[si,jj,kk,nm]
            else
                ei = h + 1
                rho[di,jj,kk,b]=rho[ei,jj,kk,b]; ux[di,jj,kk,b]=ux[ei,jj,kk,b]
                uy[di,jj,kk,b]=uy[ei,jj,kk,b];   uz[di,jj,kk,b]=uz[ei,jj,kk,b]
            end
        end
    end
end

@kernel function pond_halo_y!(rho, ux, uy, uz, @Const(nbr), BS::Int32, H::Int32)
    it, kt, b = @index(Global, NTuple)
    bs = Int(BS); h = Int(H)
    @inbounds begin
        ii = it + h; kk = kt + h
        np = nbr[b, 3]; nm = nbr[b, 4]
        for s in 1:h
            dj = bs + h + s
            if np > 0
                sj = h + s
                rho[ii,dj,kk,b]=rho[ii,sj,kk,np]; ux[ii,dj,kk,b]=ux[ii,sj,kk,np]
                uy[ii,dj,kk,b]=uy[ii,sj,kk,np];   uz[ii,dj,kk,b]=uz[ii,sj,kk,np]
            else
                ej = bs + h
                rho[ii,dj,kk,b]=rho[ii,ej,kk,b]; ux[ii,dj,kk,b]=ux[ii,ej,kk,b]
                uy[ii,dj,kk,b]=uy[ii,ej,kk,b];   uz[ii,dj,kk,b]=uz[ii,ej,kk,b]
            end
        end
        for s in 1:h
            dj = h + 1 - s
            if nm > 0
                sj = bs + h + 1 - s
                rho[ii,dj,kk,b]=rho[ii,sj,kk,nm]; ux[ii,dj,kk,b]=ux[ii,sj,kk,nm]
                uy[ii,dj,kk,b]=uy[ii,sj,kk,nm];   uz[ii,dj,kk,b]=uz[ii,sj,kk,nm]
            else
                ej = h + 1
                rho[ii,dj,kk,b]=rho[ii,ej,kk,b]; ux[ii,dj,kk,b]=ux[ii,ej,kk,b]
                uy[ii,dj,kk,b]=uy[ii,ej,kk,b];   uz[ii,dj,kk,b]=uz[ii,ej,kk,b]
            end
        end
    end
end

@kernel function pond_halo_z!(rho, ux, uy, uz, @Const(nbr), BS::Int32, H::Int32)
    it, jt, b = @index(Global, NTuple)
    bs = Int(BS); h = Int(H)
    @inbounds begin
        ii = it + h; jj = jt + h
        np = nbr[b, 5]; nm = nbr[b, 6]
        for s in 1:h
            dk = bs + h + s
            if np > 0
                sk = h + s
                rho[ii,jj,dk,b]=rho[ii,jj,sk,np]; ux[ii,jj,dk,b]=ux[ii,jj,sk,np]
                uy[ii,jj,dk,b]=uy[ii,jj,sk,np];   uz[ii,jj,dk,b]=uz[ii,jj,sk,np]
            else
                ek = bs + h
                rho[ii,jj,dk,b]=rho[ii,jj,ek,b]; ux[ii,jj,dk,b]=ux[ii,jj,ek,b]
                uy[ii,jj,dk,b]=uy[ii,jj,ek,b];   uz[ii,jj,dk,b]=uz[ii,jj,ek,b]
            end
        end
        for s in 1:h
            dk = h + 1 - s
            if nm > 0
                sk = bs + h + 1 - s
                rho[ii,jj,dk,b]=rho[ii,jj,sk,nm]; ux[ii,jj,dk,b]=ux[ii,jj,sk,nm]
                uy[ii,jj,dk,b]=uy[ii,jj,sk,nm];   uz[ii,jj,dk,b]=uz[ii,jj,sk,nm]
            else
                ek = h + 1
                rho[ii,jj,dk,b]=rho[ii,jj,ek,b]; ux[ii,jj,dk,b]=ux[ii,jj,ek,b]
                uy[ii,jj,dk,b]=uy[ii,jj,ek,b];   uz[ii,jj,dk,b]=uz[ii,jj,ek,b]
            end
        end
    end
end

# ---- per-axis FV on padded blocks (reuse _fvflux3/_feq3_gpu from pond_fv3d_gpu.jl) ----
# Interior cell (it,jt,kt) of block b → padded (it+H, jt+H, kt+H, b). Neighbour reads
# land in the (filled) halo for block-edge cells, so NO clamp is needed.
@kernel function pond_block_flux_x!(rho_o, mx_o, my_o, mz_o,
                                    @Const(rho), @Const(ux), @Const(uy), @Const(uz),
                                    @Const(cx), @Const(cy), @Const(cz), @Const(w),
                                    cs2::Float32, a::Float32, H::Int32)
    it, jt, kt, b = @index(Global, NTuple)
    h = Int(H); i = it + h; j = jt + h; k = kt + h
    @inbounds begin
        FrR,FmxR,FmyR,FmzR = _fvflux3(rho[i,j,k,b],ux[i,j,k,b],uy[i,j,k,b],uz[i,j,k,b], rho[i+1,j,k,b],ux[i+1,j,k,b],uy[i+1,j,k,b],uz[i+1,j,k,b],
                                      rho[i-1,j,k,b],ux[i-1,j,k,b],uy[i-1,j,k,b],uz[i-1,j,k,b], rho[i+2,j,k,b],ux[i+2,j,k,b],uy[i+2,j,k,b],uz[i+2,j,k,b],
                                      1, cx,cy,cz,w,cs2)
        FrL,FmxL,FmyL,FmzL = _fvflux3(rho[i-1,j,k,b],ux[i-1,j,k,b],uy[i-1,j,k,b],uz[i-1,j,k,b], rho[i,j,k,b],ux[i,j,k,b],uy[i,j,k,b],uz[i,j,k,b],
                                      rho[i-2,j,k,b],ux[i-2,j,k,b],uy[i-2,j,k,b],uz[i-2,j,k,b], rho[i+1,j,k,b],ux[i+1,j,k,b],uy[i+1,j,k,b],uz[i+1,j,k,b],
                                      1, cx,cy,cz,w,cs2)
        rho_o[i,j,k,b] = rho[i,j,k,b]             - a*(FrR-FrL)
        mx_o[i,j,k,b]  = rho[i,j,k,b]*ux[i,j,k,b] - a*(FmxR-FmxL)
        my_o[i,j,k,b]  = rho[i,j,k,b]*uy[i,j,k,b] - a*(FmyR-FmyL)
        mz_o[i,j,k,b]  = rho[i,j,k,b]*uz[i,j,k,b] - a*(FmzR-FmzL)
    end
end

@kernel function pond_block_flux_y!(rho_o, mx_o, my_o, mz_o,
                                    @Const(rho), @Const(ux), @Const(uy), @Const(uz),
                                    @Const(cx), @Const(cy), @Const(cz), @Const(w),
                                    cs2::Float32, a::Float32, H::Int32)
    it, jt, kt, b = @index(Global, NTuple)
    h = Int(H); i = it + h; j = jt + h; k = kt + h
    @inbounds begin
        FrT,FmxT,FmyT,FmzT = _fvflux3(rho[i,j,k,b],ux[i,j,k,b],uy[i,j,k,b],uz[i,j,k,b], rho[i,j+1,k,b],ux[i,j+1,k,b],uy[i,j+1,k,b],uz[i,j+1,k,b],
                                      rho[i,j-1,k,b],ux[i,j-1,k,b],uy[i,j-1,k,b],uz[i,j-1,k,b], rho[i,j+2,k,b],ux[i,j+2,k,b],uy[i,j+2,k,b],uz[i,j+2,k,b],
                                      2, cx,cy,cz,w,cs2)
        FrB,FmxB,FmyB,FmzB = _fvflux3(rho[i,j-1,k,b],ux[i,j-1,k,b],uy[i,j-1,k,b],uz[i,j-1,k,b], rho[i,j,k,b],ux[i,j,k,b],uy[i,j,k,b],uz[i,j,k,b],
                                      rho[i,j-2,k,b],ux[i,j-2,k,b],uy[i,j-2,k,b],uz[i,j-2,k,b], rho[i,j+1,k,b],ux[i,j+1,k,b],uy[i,j+1,k,b],uz[i,j+1,k,b],
                                      2, cx,cy,cz,w,cs2)
        rho_o[i,j,k,b] -= a*(FrT-FrB)
        mx_o[i,j,k,b]  -= a*(FmxT-FmxB)
        my_o[i,j,k,b]  -= a*(FmyT-FmyB)
        mz_o[i,j,k,b]  -= a*(FmzT-FmzB)
    end
end

@kernel function pond_block_flux_z!(rho_o, mx_o, my_o, mz_o,
                                    @Const(rho), @Const(ux), @Const(uy), @Const(uz),
                                    @Const(cx), @Const(cy), @Const(cz), @Const(w),
                                    cs2::Float32, a::Float32, H::Int32)
    it, jt, kt, b = @index(Global, NTuple)
    h = Int(H); i = it + h; j = jt + h; k = kt + h
    @inbounds begin
        FrF,FmxF,FmyF,FmzF = _fvflux3(rho[i,j,k,b],ux[i,j,k,b],uy[i,j,k,b],uz[i,j,k,b], rho[i,j,k+1,b],ux[i,j,k+1,b],uy[i,j,k+1,b],uz[i,j,k+1,b],
                                      rho[i,j,k-1,b],ux[i,j,k-1,b],uy[i,j,k-1,b],uz[i,j,k-1,b], rho[i,j,k+2,b],ux[i,j,k+2,b],uy[i,j,k+2,b],uz[i,j,k+2,b],
                                      3, cx,cy,cz,w,cs2)
        FrK,FmxK,FmyK,FmzK = _fvflux3(rho[i,j,k-1,b],ux[i,j,k-1,b],uy[i,j,k-1,b],uz[i,j,k-1,b], rho[i,j,k,b],ux[i,j,k,b],uy[i,j,k,b],uz[i,j,k,b],
                                      rho[i,j,k-2,b],ux[i,j,k-2,b],uy[i,j,k-2,b],uz[i,j,k-2,b], rho[i,j,k+1,b],ux[i,j,k+1,b],uy[i,j,k+1,b],uz[i,j,k+1,b],
                                      3, cx,cy,cz,w,cs2)
        r  = rho_o[i,j,k,b] - a*(FrF-FrK)
        mx = mx_o[i,j,k,b]  - a*(FmxF-FmxK)
        my = my_o[i,j,k,b]  - a*(FmyF-FmyK)
        mz = mz_o[i,j,k,b]  - a*(FmzF-FmzK)
        r = r < 1.0f-4 ? 1.0f-4 : r
        rho_o[i,j,k,b] = r; mx_o[i,j,k,b] = mx/r; my_o[i,j,k,b] = my/r; mz_o[i,j,k,b] = mz/r
    end
end

# host stepping helper: halo-exchange (all 6 faces, on the OLD state) then the 3 per-axis
# flux kernels (which read the OLD haloed state, write the new interior). BS×BS×BS×nb ndrange.
function pond_fv3d_blocks_step!(backend, out, st, nbr, cx,cy,cz,w, cs2::Float32, dt::Float32, dx::Float32, BS::Integer, nb::Integer)
    a = dt/dx; h = POND_H
    bsi = Int32(BS); hi = Int32(h)
    tang = (Int(BS), Int(BS), Int(nb))      # tangential face ndrange (jt,kt,b) etc.
    full = (Int(BS), Int(BS), Int(BS), Int(nb))
    ro,mxo,myo,mzo = out; r,uxs,uys,uzs = st
    # halo exchange on the OLD state
    pond_halo_x!(backend)(r,uxs,uys,uzs, nbr, bsi, hi; ndrange=tang)
    pond_halo_y!(backend)(r,uxs,uys,uzs, nbr, bsi, hi; ndrange=tang)
    pond_halo_z!(backend)(r,uxs,uys,uzs, nbr, bsi, hi; ndrange=tang)
    KernelAbstractions.synchronize(backend)
    # per-axis conservative update
    pond_block_flux_x!(backend)(ro,mxo,myo,mzo, r,uxs,uys,uzs, cx,cy,cz,w, cs2, Float32(a), hi; ndrange=full)
    KernelAbstractions.synchronize(backend)
    pond_block_flux_y!(backend)(ro,mxo,myo,mzo, r,uxs,uys,uzs, cx,cy,cz,w, cs2, Float32(a), hi; ndrange=full)
    KernelAbstractions.synchronize(backend)
    pond_block_flux_z!(backend)(ro,mxo,myo,mzo, r,uxs,uys,uzs, cx,cy,cz,w, cs2, Float32(a), hi; ndrange=full)
    KernelAbstractions.synchronize(backend)
    return nothing
end
