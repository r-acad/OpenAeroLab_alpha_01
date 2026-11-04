# D3Q19 lattice: velocity set, weights, opposite and mirrored directions, the equilibrium and the Hermite tensors
# used by the regularised collision. Opposite directions are adjacent pairs (2-3, 4-5, ...) so that bounce-back
# and mirror lookups are table reads. cs^2 = 1/3. The tables are checked against the main solver's
# `build_d3q19_static` (src/collision/lattice.jl) by tools/tests/q19_octree_lattice.jl.

const Q = 19
const CS2 = 1 / 3

const CX = (0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0)
const CY = (0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1)
const CZ = (0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1)
const WT = (1 / 3, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18,
            1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36)

"Index of the direction with velocity (cx, cy, cz); 0 when it is not in the set."
function direction_index(cx::Integer, cy::Integer, cz::Integer)
    for i in 1:Q
        CX[i] == cx && CY[i] == cy && CZ[i] == cz && return i
    end
    0
end

const OPP = ntuple(i -> direction_index(-CX[i], -CY[i], -CZ[i]), Q)
const MIRROR_X = ntuple(i -> direction_index(-CX[i], CY[i], CZ[i]), Q)
const MIRROR_Y = ntuple(i -> direction_index(CX[i], -CY[i], CZ[i]), Q)
const MIRROR_Z = ntuple(i -> direction_index(CX[i], CY[i], -CZ[i]), Q)

"Second-order equilibrium of direction i for the given density, velocity and |u|^2."
@inline function feq(i::Int, rho::T, ux::T, uy::T, uz::T, usq::T) where {T<:AbstractFloat}
    cu = T(CX[i]) * ux + T(CY[i]) * uy + T(CZ[i]) * uz
    T(WT[i]) * rho * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end

"Density and velocity of a population tuple (the velocity in lattice units of the cell's level)."
@inline function macros(f::NTuple{Q,T}) where {T<:AbstractFloat}
    rho = zero(T); mx = zero(T); my = zero(T); mz = zero(T)
    @inbounds for i in 1:Q
        rho += f[i]; mx += T(CX[i]) * f[i]; my += T(CY[i]) * f[i]; mz += T(CZ[i]) * f[i]
    end
    inv = one(T) / rho
    rho, mx * inv, my * inv, mz * inv
end

"""
Second-order non-equilibrium moments (Pi_xx, Pi_yy, Pi_zz, Pi_xy, Pi_xz, Pi_yz) of f about the equilibrium at
(rho, u): the a2 Hermite coefficients of the regularisation.
"""
@inline function pi_neq(f::NTuple{Q,T}, rho::T, ux::T, uy::T, uz::T) where {T<:AbstractFloat}
    usq = ux * ux + uy * uy + uz * uz
    pxx = zero(T); pyy = zero(T); pzz = zero(T); pxy = zero(T); pxz = zero(T); pyz = zero(T)
    @inbounds for i in 2:Q
        fn = f[i] - feq(i, rho, ux, uy, uz, usq)
        cx = T(CX[i]); cy = T(CY[i]); cz = T(CZ[i])
        pxx += cx * cx * fn; pyy += cy * cy * fn; pzz += cz * cz * fn
        pxy += cx * cy * fn; pxz += cx * cz * fn; pyz += cy * cz * fn
    end
    (pxx, pyy, pzz, pxy, pxz, pyz)
end

"""
Regularised non-equilibrium population of direction i from the second-order moments p = (xx, yy, zz, xy, xz, yz)
and the velocity (third-order recursive terms; the xyz and aaa Hermite tensors vanish on D3Q19):
fneq_i = w_i [ 4.5 H2_i : a2 + 13.5 sum_{a != b} H3_{i,aab} a3_{aab} ],  a3_{aab} = 2 u_a a2_{ab} + u_b a2_{aa}.
"""
@inline fneq_regularised(i::Int, p::NTuple{6,T}, ux::T, uy::T, uz::T) where {T<:AbstractFloat} = fneq_regularised(i, p, a3_terms(p, ux, uy, uz))

"The six third-order recursive coefficients (xxy, xxz, yyx, yyz, zzx, zzy), computed once per cell, not per direction."
@inline function a3_terms(p::NTuple{6,T}, ux::T, uy::T, uz::T) where {T<:AbstractFloat}
    pxx, pyy, pzz, pxy, pxz, pyz = p
    (T(2) * ux * pxy + uy * pxx, T(2) * ux * pxz + uz * pxx,
     T(2) * uy * pxy + ux * pyy, T(2) * uy * pyz + uz * pyy,
     T(2) * uz * pxz + ux * pzz, T(2) * uz * pyz + uy * pzz)
end

"The regularised non-equilibrium population of direction i from the second-order moments and the cell's a3 terms."
@inline function fneq_regularised(i::Int, p::NTuple{6,T}, a3::NTuple{6,T}) where {T<:AbstractFloat}
    pxx, pyy, pzz, pxy, pxz, pyz = p
    axxy, axxz, ayyx, ayyz, azzx, azzy = a3
    cx = T(CX[i]); cy = T(CY[i]); cz = T(CZ[i]); third = T(1) / T(3)
    h2 = (cx * cx - third) * pxx + (cy * cy - third) * pyy + (cz * cz - third) * pzz +
         T(2) * (cx * cy * pxy + cx * cz * pxz + cy * cz * pyz)
    h3 = cy * (cx * cx - third) * axxy + cz * (cx * cx - third) * axxz +
         cx * (cy * cy - third) * ayyx + cz * (cy * cy - third) * ayyz +
         cx * (cz * cz - third) * azzx + cy * (cz * cz - third) * azzy
    T(WT[i]) * (T(4.5) * h2 + T(13.5) * h3)
end
