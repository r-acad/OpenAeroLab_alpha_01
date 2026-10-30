# ==============================================================================
# Particles-on-Demand — AMR cross-level transfer kernels (prolongation + restriction)
# ==============================================================================
# The GPU data-transfer half of the 2-level AMR coupling (the conservative interface FLUX is the
# validated mortar — pond_fv3d_amr3d.jl). These fill the fine grid's halo from the coarse grid
# (PROLONGATION, coarse→fine, trilinear ⇒ exact for linear fields) and average the fine solution
# back onto the covered coarse cells (RESTRICTION, fine→coarse, 8-cell average ⇒ exactly
# conservative). 2:1 refinement (dx_f = dx_c/2). Backend-agnostic (KernelAbstractions).
# ==============================================================================

using KernelAbstractions

@inline _clA(i,n) = i<1 ? 1 : (i>n ? n : i)

# PROLONGATION (coarse → fine, trilinear). Fine cell (i,j,k) [1-based] lies under coarse cell
# (I,J,K)=ceil(./2). Within the coarse cell the two fine subcells per axis sit at ±0.25 of the coarse
# spacing from the coarse centre ⇒ weights 0.75(self)+0.25(neighbour). Trilinear = product of axes.
@kernel function pond_prolong_kernel!(fine, @Const(coarse), NCX::Int32,NCY::Int32,NCZ::Int32)
    i,j,k = @index(Global, NTuple)
    ncx=Int(NCX); ncy=Int(NCY); ncz=Int(NCZ)
    @inbounds begin
        I=(i+1)>>1; J=(j+1)>>1; K=(k+1)>>1                 # parent coarse cell (ceil(i/2))
        # neighbour direction per axis: odd fine idx ⇒ −0.25 side (neighbour = I−1); even ⇒ +0.25 (I+1)
        Inb = isodd(i) ? _clA(I-1,ncx) : _clA(I+1,ncx)
        Jnb = isodd(j) ? _clA(J-1,ncy) : _clA(J+1,ncy)
        Knb = isodd(k) ? _clA(K-1,ncz) : _clA(K+1,ncz)
        a=0.75f0; b=0.25f0                                  # self / neighbour weights
        # trilinear: Σ over the 2×2×2 {self,neighbour} per axis with weights a/b
        v = 0.0f0
        @inbounds for (kk,wk) in ((K,a),(Knb,b)), (jj,wj) in ((J,a),(Jnb,b)), (ii,wi) in ((I,a),(Inb,b))
            v += wi*wj*wk*coarse[ii,jj,kk]
        end
        fine[i,j,k] = v
    end
end

# RESTRICTION (fine → coarse, 8-cell average ⇒ exactly conservative: 8·avg·dx_f³ = Σfine·dx_f³,
# and dx_c³ = 8 dx_f³ ⇒ coarse·dx_c³ = Σfine·dx_f³).
@kernel function pond_restrict_kernel!(coarse, @Const(fine), NCX::Int32,NCY::Int32,NCZ::Int32)
    I,J,K = @index(Global, NTuple)
    @inbounds begin
        i0=2I-1; j0=2J-1; k0=2K-1
        s = fine[i0,j0,k0]+fine[i0+1,j0,k0]+fine[i0,j0+1,k0]+fine[i0+1,j0+1,k0]+
            fine[i0,j0,k0+1]+fine[i0+1,j0,k0+1]+fine[i0,j0+1,k0+1]+fine[i0+1,j0+1,k0+1]
        coarse[I,J,K] = 0.125f0*s
    end
end

# host helpers
function pond_prolong!(backend, fine, coarse, ncx,ncy,ncz)
    pond_prolong_kernel!(backend)(fine, coarse, Int32(ncx),Int32(ncy),Int32(ncz); ndrange=(2ncx,2ncy,2ncz))
    KernelAbstractions.synchronize(backend); return nothing
end
function pond_restrict!(backend, coarse, fine, ncx,ncy,ncz)
    pond_restrict_kernel!(backend)(coarse, fine, Int32(ncx),Int32(ncy),Int32(ncz); ndrange=(ncx,ncy,ncz))
    KernelAbstractions.synchronize(backend); return nothing
end
