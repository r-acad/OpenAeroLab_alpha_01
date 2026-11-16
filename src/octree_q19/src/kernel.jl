# The cell kernel in closed form, written once for a scalar and for the eight cells of an oct at a time.
#
# Every function here takes `V`, either a floating-point scalar or a SIMD vector of eight of them (`Vec{8,T}`: the
# eight children of an oct are stored contiguously per direction, so one oct is one vector per population). The
# arithmetic of a lane is the arithmetic of the scalar form, operation for operation: a cell computes the same
# numbers whether it is stepped alone or with its oct (tools/tests/q19_kernel.jl checks this bit for bit).
#
# Populations are shifted (f_s = f - w_i). The moments are taken from sums and differences of opposite pairs:
#   drho = sum f_s,  m = sum c f_s,  P_ab = sum c_a c_b f_s,  Pi_neq_ab = P_ab - drho/3 delta_ab - rho u_a u_b,
# and the post-collision state is rebuilt pair by pair from an even and an odd part,
#   f_i = w_i (E_i + O_i),  f_opp(i) = w_i (E_i - O_i),
#   E_i = drho + rho (4.5 (c.u)^2 - 1.5 u^2) + (1 - omega) 4.5 H2_i : a2,   O_i = 3 rho c.u + (1 - omega) 13.5 H3_i . a3.
# No equilibrium is formed as "feq - w": the shifted form keeps the digits that subtraction lost in single precision.
using SIMD

"Switch of the eight-at-a-time step (on by default); off, every cell is stepped one by one (the reference of tools/tests/q19_kernel.jl)."
const VECTOR_STEP = Ref(true)
"Streaming (non-temporal) stores of whole octs on levels whose population buffer exceeds STREAMING_MIN values."
const STREAMING_STORES = Ref(true)
const STREAMING_MIN = Ref(8_000_000)
@inline _eltype(::Type{T}) where {T<:AbstractFloat} = T
@inline _eltype(::Type{Vec{N,T}}) where {N,T} = T
@inline _select(c::Bool, a, b) = ifelse(c, a, b)
@inline _select(c::Vec{N,Bool}, a::Vec{N,T}, b::Vec{N,T}) where {N,T} = vifelse(c, a, b)

"drho, momentum (mx, my, mz) and the second moments P = (xx, yy, zz, xy, xz, yz) of shifted populations."
@inline function moments_q19(f::NTuple{Q,V}) where V
    @inbounds begin
        sx = f[2] + f[3]; dx = f[2] - f[3]
        sy = f[4] + f[5]; dy = f[4] - f[5]
        sz = f[6] + f[7]; dz = f[6] - f[7]
        sxy = f[8] + f[9]; dxy = f[8] - f[9]          # (1, 1, 0)
        sxY = f[10] + f[11]; dxY = f[10] - f[11]      # (1, -1, 0)
        sxz = f[12] + f[13]; dxz = f[12] - f[13]      # (1, 0, 1)
        sxZ = f[14] + f[15]; dxZ = f[14] - f[15]      # (1, 0, -1)
        syz = f[16] + f[17]; dyz = f[16] - f[17]      # (0, 1, 1)
        syZ = f[18] + f[19]; dyZ = f[18] - f[19]      # (0, 1, -1)
        exy = sxy + sxY; exz = sxz + sxZ; eyz = syz + syZ
        drho = (f[1] + ((sx + sy) + sz)) + ((exy + exz) + eyz)
        mx = dx + ((dxy + dxY) + (dxz + dxZ))
        my = dy + ((dxy - dxY) + (dyz + dyZ))
        mz = dz + ((dxz - dxZ) + (dyz - dyZ))
        P = (sx + (exy + exz), sy + (exy + eyz), sz + (exz + eyz), sxy - sxY, sxz - sxZ, syz - syZ)
    end
    drho, mx, my, mz, P
end

"Second-order non-equilibrium moments about the equilibrium at (rho, u)."
@inline function pineq_q19(drho::V, rho::V, ux::V, uy::V, uz::V, P::NTuple{6,V}) where V
    d3 = drho * _eltype(V)(1 / 3)
    rx = rho * ux; ry = rho * uy; rz = rho * uz
    (P[1] - d3 - rx * ux, P[2] - d3 - ry * uy, P[3] - d3 - rz * uz, P[4] - rx * uy, P[5] - rx * uz, P[6] - ry * uz)
end

"""
Shifted populations of the regularised state with density 1 + drho, velocity u, second-order non-equilibrium
moments `a2` and the factor `keep` on the non-equilibrium part (1 - omega after a collision, 0 for the
equilibrium). The third-order recursive terms a3_aab = 2 u_a a2_ab + u_b a2_aa are included.
"""
@inline function populations_q19(drho::V, rho::V, ux::V, uy::V, uz::V, a2::NTuple{6,V}, keep) where V
    E = _eltype(V)
    pxx, pyy, pzz, pxy, pxz, pyz = a2
    A = drho - E(1.5) * rho * (ux * ux + uy * uy + uz * uz)
    k2 = E(4.5) * keep; k3 = E(13.5) * keep
    r3 = E(3) * rho; r45 = E(4.5) * rho
    t = (pxx + pyy + pzz) * E(1 / 3)
    two = E(2); t3 = E(1 / 3); t23 = E(2 / 3)
    axxy = two * ux * pxy + uy * pxx; axxz = two * ux * pxz + uz * pxx
    ayyx = two * uy * pxy + ux * pyy; ayyz = two * uy * pyz + uz * pyy
    azzx = two * uz * pxz + ux * pzz; azzy = two * uz * pyz + uy * pzz
    w1 = E(1 / 18); w2 = E(1 / 36)
    @inline pair(w, cu, h2, h3) = (e = A + r45 * cu * cu + k2 * h2; o = r3 * cu + k3 * h3; (w * (e + o), w * (e - o)))
    f1 = E(1 / 3) * (A - k2 * t)
    f2, f3 = pair(w1, ux, pxx - t, -(ayyx + azzx) * t3)
    f4, f5 = pair(w1, uy, pyy - t, -(axxy + azzy) * t3)
    f6, f7 = pair(w1, uz, pzz - t, -(axxz + ayyz) * t3)
    hxy = pxx + pyy - t; hxz = pxx + pzz - t; hyz = pyy + pzz - t
    f8, f9 = pair(w2, ux + uy, hxy + two * pxy, t23 * (axxy + ayyx) - t3 * (azzx + azzy))
    f10, f11 = pair(w2, ux - uy, hxy - two * pxy, t23 * (ayyx - axxy) - t3 * (azzx - azzy))
    f12, f13 = pair(w2, ux + uz, hxz + two * pxz, t23 * (axxz + azzx) - t3 * (ayyx + ayyz))
    f14, f15 = pair(w2, ux - uz, hxz - two * pxz, t23 * (azzx - axxz) - t3 * (ayyx - ayyz))
    f16, f17 = pair(w2, uy + uz, hyz + two * pyz, t23 * (ayyz + azzy) - t3 * (axxy + axxz))
    f18, f19 = pair(w2, uy - uz, hyz - two * pyz, t23 * (azzy - ayyz) - t3 * (axxy - axxz))
    (f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19)
end

"The shifted equilibrium of (1 + drho, u)."
@inline function equilibrium_q19(drho::V, rho::V, ux::V, uy::V, uz::V) where V
    z = zero(V)
    populations_q19(drho, rho, ux, uy, uz, (z, z, z, z, z, z), zero(_eltype(V)))
end

"True in the lanes where a population is at or below the positivity floor (1e-6 of its rest weight)."
@inline function below_floor(out::NTuple{Q,V}) where V
    E = _eltype(V)
    l0 = E(-(1 - 1e-6) / 3); l1 = E(-(1 - 1e-6) / 18); l2 = E(-(1 - 1e-6) / 36)
    bad = out[1] < l0
    bad |= (out[2] < l1) | (out[3] < l1) | (out[4] < l1) | (out[5] < l1) | (out[6] < l1) | (out[7] < l1)
    bad |= (out[8] < l2) | (out[9] < l2) | (out[10] < l2) | (out[11] < l2) | (out[12] < l2) | (out[13] < l2)
    bad |= (out[14] < l2) | (out[15] < l2) | (out[16] < l2) | (out[17] < l2) | (out[18] < l2) | (out[19] < l2)
    bad
end

"""
Recursive-regularised collision without the positivity safeguard: the post-collision populations and
(drho, rho, u). `sigma` blends the measured second-order moments with `pi_fd` (HRR; 1 is plain RR); `omega` may
differ per lane. `F` is the Guo body force (lattice units), used when `forced`: u = (m + F/2) / rho, and the
source w_i [1.5 c.F + (1 - omega/2)(9 (c.u)(c.F) - 3 u.F)] is added (the first-order part carries the other
half of the force: the regularised non-equilibrium has no momentum).
"""
@inline function collide_q19(f::NTuple{Q,V}, omega::V, sigma, pi_fd::NTuple{6,V}, F::NTuple{3,E}, forced::Bool) where {V,E}
    drho, mx, my, mz, P = moments_q19(f)
    rho = drho + one(E)
    inv = one(E) / rho
    hF = forced ? E(0.5) : zero(E)
    ux = (mx + hF * F[1]) * inv; uy = (my + hF * F[2]) * inv; uz = (mz + hF * F[3]) * inv
    p = pineq_q19(drho, rho, ux, uy, uz, P)
    s1 = one(E) - sigma
    a2 = sigma == one(E) ? p : (sigma * p[1] + s1 * pi_fd[1], sigma * p[2] + s1 * pi_fd[2], sigma * p[3] + s1 * pi_fd[3],
                               sigma * p[4] + s1 * pi_fd[4], sigma * p[5] + s1 * pi_fd[5], sigma * p[6] + s1 * pi_fd[6])
    base = populations_q19(drho, rho, ux, uy, uz, a2, one(E) - omega)
    out = if forced
        pre = one(E) - E(0.5) * omega
        uF = ux * F[1] + uy * F[2] + uz * F[3]
        ntuple(Val(Q)) do i
            cx = E(CX[i]); cy = E(CY[i]); cz = E(CZ[i])
            cu = cx * ux + cy * uy + cz * uz; cF = cx * F[1] + cy * F[2] + cz * F[3]
            @inbounds base[i] + E(WT[i]) * (E(1.5) * cF + pre * (E(9) * cu * cF - E(3) * uF))
        end
    else
        base
    end
    out, drho, rho, ux, uy, uz
end

# ---------------------------------------------------------------- the eight lanes of an oct
"The 27 neighbour octs of oct `o` as a tuple; a missing neighbour reads the oct itself (only lanes that are masked out can refer to it)."
@inline neighbour_octs(onbr::Matrix{Int32}, o::Int) = ntuple(a -> (n = Int(@inbounds onbr[a, o]); ifelse(n == 0, o, n)), Val(27))

"""
The eight values that the cells of an oct pull along direction `J` from an (8, R, nocts) array given as a vector:
row `row` of the source octs `nb`, `stride` = 8 R values per oct. The lanes are assembled from the one, two or
four source octs of the direction by compile-time shuffles.
"""
@generated function pull_lanes(a::Vector{T}, nb::NTuple{27,Int}, row::Int, stride::Int, ::Val{J}) where {T,J}
    offs = Int[]
    for k in 1:8
        s = Int(SRC_OFF[k, J]); s in offs || push!(offs, s)
    end
    src(k) = findfirst(==(Int(SRC_OFF[k, J])), offs)
    child(k) = Int(SRC_CHILD[k, J])
    loads = [:($(Symbol(:v, s)) = vload(Vec{8,T}, a, base + stride * (nb[$(offs[s])] - 1))) for s in eachindex(offs)]
    body = if length(offs) == 1
        all(child(k) == k for k in 1:8) ? :(v1) : :(shufflevector(v1, v1, Val($(Tuple(child(k) - 1 for k in 1:8)))))
    elseif length(offs) == 2
        :(shufflevector(v1, v2, Val($(Tuple((src(k) - 1) * 8 + child(k) - 1 for k in 1:8)))))
    else
        length(offs) == 4 || error("unexpected source count")
        mA = Tuple(src(k) <= 2 ? (src(k) - 1) * 8 + child(k) - 1 : 0 for k in 1:8)
        mB = Tuple(src(k) >= 3 ? (src(k) - 3) * 8 + child(k) - 1 : 0 for k in 1:8)
        mR = Tuple(src(k) <= 2 ? k - 1 : 8 + k - 1 for k in 1:8)
        :(shufflevector(shufflevector(v1, v2, Val($mA)), shufflevector(v3, v4, Val($mB)), Val($mR)))
    end
    quote
        Base.@_inline_meta
        base = 1 + 8 * (row - 1)
        @inbounds begin
            $(loads...)
        end
        $body
    end
end

"The 19 pulled population vectors of oct `nb[self]`."
@generated function pull_populations(a::Vector{T}, nb::NTuple{27,Int}) where T
    Expr(:block, Expr(:meta, :inline), Expr(:tuple, [:(pull_lanes(a, nb, $j, $(8Q), Val($j))) for j in 1:Q]...))
end

"Central-difference velocity gradient of the eight cells of an oct from the stored velocity (8, 3, nocts), g[a, b] = d u_b / d x_a."
@inline function gradient_lanes(u::Vector{T}, nb::NTuple{27,Int}) where T
    h = T(0.5)
    # the value at +x is the pull pattern of the direction with c = (-1, 0, 0), and so on
    @inline d(a, Jp, Jm) = h * (pull_lanes(u, nb, a, 24, Jp) - pull_lanes(u, nb, a, 24, Jm))
    (d(1, Val(3), Val(2)), d(2, Val(3), Val(2)), d(3, Val(3), Val(2)),
     d(1, Val(5), Val(4)), d(2, Val(5), Val(4)), d(3, Val(5), Val(4)),
     d(1, Val(7), Val(6)), d(2, Val(7), Val(6)), d(3, Val(7), Val(6)))
end

const LANE_BITS = Vec{8,Int32}((1, 2, 4, 8, 16, 32, 64, 128))
"Lane mask of the set bits of `m`."
@inline lane_mask(m::UInt8) = (Vec{8,Int32}(Int32(m)) & LANE_BITS) != Int32(0)
