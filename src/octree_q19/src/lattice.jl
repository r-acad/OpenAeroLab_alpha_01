# D3Q19 lattice: velocity set, weights, opposite and mirrored directions, and the equilibrium of one direction (the
# collision itself is in closed form in kernel.jl). Opposite directions are adjacent pairs (2-3, 4-5, ...) so that bounce-back
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
