# Coupling between levels (DESIGN.md sections 5 and 6.3, amendment A4) and the recursive step.
#
# Buffers: after `step_level!` + `swap_buffers!`, a level's `f` holds the post-collision state at its new time
# and `g` the one at its previous time; both stay until its next step. The coarse level steps first, then its
# child twice. The child's ghosts for its first sub-step were filled at the end of the previous cycle from the
# coarse state at the coarse old time (and initially by `couple_initial!`); between the sub-steps they are filled
# at the half time (a blend of the coarse old and new states; covered donors hold the half-time fine average,
# restricted just before, and enter unblended); after the second sub-step the fine state is restricted at the new
# time and the ghosts are filled from the coarse new state for the next cycle. Transfers go through moments: rho,
# u and the second-order non-equilibrium moments on cubic 4x4x4 stencils (trilinear or the children's average where
# the stencil is incomplete, see generator.jl), rescaled by tau_f / (2 tau_c) (coarse to fine) or its inverse, and
# the populations are rebuilt as post-collision-equivalent states at the receiving level's omega. The moments of
# every donor or source cell are computed once per transfer into a buffer; the stencil sums read the buffer, eight
# of the ten moments at a time as one SIMD vector.

"rho, u and Pi_neq of slot (k, o) from a population buffer."
@inline function slot_moments(f::Array{T,3}, k::Int, o::Int) where T
    fs = ntuple(i -> (@inbounds f[k, i, o]), Val(Q))
    drho, mx, my, mz, P = moments_q19(fs)
    rho = drho + one(T); inv = one(T) / rho
    ux = mx * inv; uy = my * inv; uz = mz * inv
    rho, ux, uy, uz, pineq_q19(drho, rho, ux, uy, uz, P)
end

"Rebuild a post-collision-equivalent shifted population tuple from moments at relaxation rate omega."
@inline function rebuild(rho::T, ux::T, uy::T, uz::T, p::NTuple{6,T}, omega::T) where T
    drho = rho - one(T)
    out = populations_q19(drho, rho, ux, uy, uz, p, one(T) - omega)
    below_floor(out) ? positive_shifted(out, equilibrium_q19(drho, rho, ux, uy, uz)) : out
end

# Factor turning the non-equilibrium moments of a post-collision state into those of its pre-collision state,
# 1 / (1 - omega); near omega = 1 the post-collision non-equilibrium carries no information and the transfer
# drops it (the stress across the interface is then slightly off; tau about 1 is outside the LES range of use).
@inline pre_collision_factor(omega::T) where T = abs(one(T) - omega) > T(0.05) ? one(T) / (one(T) - omega) : zero(T)

"The weighted sum of the ten moments over the 64 nodes of stencil `j`: the first eight as a vector, then the last two. A zero index is an unused node; stencil 0 is empty."
@inline function stencil_sum(mv::Vector{T}, ix::Matrix{Int32}, w::Matrix{Float32}, j::Int) where T
    acc = zero(Vec{8,T}); a9 = zero(T); a10 = zero(T)
    j == 0 && return acc, a9, a10
    @inbounds for n in 1:64
        d = Int(ix[n, j]); d == 0 && continue
        wn = T(w[n, j]); b = 10 * (d - 1)
        acc += wn * vload(Vec{8,T}, mv, b + 1); a9 += wn * mv[b + 9]; a10 += wn * mv[b + 10]
    end
    acc, a9, a10
end

"The restriction sum of slot `ri`: the cubic weights over the 64 sources, or the average of the admissible inner eight. The flag is false when there is nothing to restrict."
@inline function restrict_sum(mv::Vector{T}, six::Matrix{Int32}, wcubic::NTuple{64,T}, cubic::Bool, ri::Int) where T
    acc = zero(Vec{8,T}); a9 = zero(T); a10 = zero(T)
    ri == 0 && return acc, a9, a10, false
    @inbounds if cubic
        for n in 1:64
            b = 10 * (Int(six[n, ri]) - 1); wn = wcubic[n]
            acc += wn * vload(Vec{8,T}, mv, b + 1); a9 += wn * mv[b + 9]; a10 += wn * mv[b + 10]
        end
    else
        cnt = 0
        for n in STENCIL_INNER
            d = Int(six[n, ri]); d == 0 && continue
            b = 10 * (d - 1)
            acc += vload(Vec{8,T}, mv, b + 1); a9 += mv[b + 9]; a10 += mv[b + 10]
            cnt += 1
        end
        cnt == 0 && return acc, a9, a10, false
        inv = one(T) / T(cnt)
        acc *= inv; a9 *= inv; a10 *= inv
    end
    acc, a9, a10, true
end

"rho, u and Pi_neq of the eight slots of oct `o` (ten lane vectors) from a population buffer given as a vector: `slot_moments` of every lane."
@inline function oct_moments(fvec::Vector{T}, o::Int) where T
    base = 152 * (o - 1)
    fv = ntuple(j -> (@inbounds vload(Vec{8,T}, fvec, base + 8 * (j - 1) + 1)), Val(Q))
    drho, mx, my, mz, P = moments_q19(fv)
    rho = drho + one(T); inv = one(T) / rho
    ux = mx * inv; uy = my * inv; uz = mz * inv
    p = pineq_q19(drho, rho, ux, uy, uz, P)
    (rho, ux, uy, uz, p[1], p[2], p[3], p[4], p[5], p[6])
end

"Store the listed lanes of ten moment vectors into the columns `idx[k, g]` of the buffer."
@inline function store_lanes!(m::Matrix{T}, idx::Matrix{Int32}, g::Int, vals::NTuple{10,Vec{8,T}}) where T
    @inbounds for k in 1:8
        d = Int(idx[k, g]); d == 0 && continue
        for i in 1:10; m[i, d] = vals[i][k]; end
    end
    nothing
end

"""
Rebuild the eight slots of oct `o` from the stencil sums of its lanes (moments 1 to 8 as a vector and the last two,
per lane) and write the lanes of `bits` into (f, u, rho): the non-equilibrium moments are scaled by `scale`, the
populations rebuilt at `omega` (as `rebuild`, lane for lane; an oct in which a lane needs the positivity safeguard
is written slot by slot).
"""
@inline function rebuild_oct!(fv::Vector{T}, f::Array{T,3}, uv::Vector{T}, rv::Vector{T}, o::Int, bits::UInt8,
                              sums::NTuple{8,Tuple{Vec{8,T},T,T}}, scale::T, omega::T) where T
    @inline lane(i) = Vec{8,T}(ntuple(k -> (@inbounds sums[k][1][i]), Val(8)))
    rho = lane(1); ux = lane(2); uy = lane(3); uz = lane(4)
    p = (scale * lane(5), scale * lane(6), scale * lane(7), scale * lane(8),
         scale * Vec{8,T}(ntuple(k -> (@inbounds sums[k][2]), Val(8))), scale * Vec{8,T}(ntuple(k -> (@inbounds sums[k][3]), Val(8))))
    out = populations_q19(rho - one(T), rho, ux, uy, uz, p, one(T) - omega)
    m = lane_mask(bits)
    ib = 152 * (o - 1); i8 = 1 + 8 * (o - 1); i24 = 1 + 24 * (o - 1)
    @inbounds if any(below_floor(out) & m)
        for k in 1:8
            (bits >> (k - 1)) & 0x01 == 0x01 || continue
            fs = rebuild(rho[k], ux[k], uy[k], uz[k], (p[1][k], p[2][k], p[3][k], p[4][k], p[5][k], p[6][k]), omega)
            for i in 1:Q; f[k, i, o] = fs[i]; end
        end
    elseif bits == 0xff
        for j in 1:Q; vstore(out[j], fv, ib + 1 + 8 * (j - 1)); end
    else
        for j in 1:Q; vstore(out[j], fv, ib + 1 + 8 * (j - 1), m); end
    end
    @inbounds if bits == 0xff
        vstore(ux, uv, i24); vstore(uy, uv, i24 + 8); vstore(uz, uv, i24 + 16); vstore(rho, rv, i8)
    else
        vstore(ux, uv, i24, m); vstore(uy, uv, i24 + 8, m); vstore(uz, uv, i24 + 16, m); vstore(rho, rv, i8, m)
    end
    nothing
end

"""
Fill the ghost slots of level `l` from level `l - 1` at `weight` along the coarse step: 1 = the coarse new buffer
(`f`), 0 = the old one (`g`), 1/2 = the blend (covered donors, which hold the fine average restricted at the half
time, enter unblended from `f`). The donor moments are interpolated with the stored stencil weights, the
non-equilibrium moments rescaled to the fine level, the populations rebuilt at the fine omega. Writes the fine
level's read buffers (f, u, rho). Both stages work on the eight slots of an oct at a time.
"""
function fill_ghosts!(st::State{T}, l::Int; weight::T=one(T)) where T
    Lf = st.mesh.levels[l]; Sf = st.levels[l]; Sc = st.levels[l - 1]
    isempty(Lf.ghost_slot) && return st
    fnewv = vec(Sc.f); foldv = vec(Sc.g)
    # molecular relaxation times of both levels (lattice units of each): tau_f - 1/2 = 2 (tau_c - 1/2)
    tau_c = one(T) / Sc.omega; tau_f = one(T) / Sf.omega
    scale = tau_f / (T(2) * tau_c) * pre_collision_factor(Sc.omega)
    omega_f = Sf.omega
    w1 = weight; w0 = one(T) - weight
    dm = Sf.donor_m; dmv = vec(dm); dix = Sf.ghost_dix; gw = Lf.ghost_w
    ff = Sf.f; ffv = vec(Sf.f); fuv = vec(Sf.u); frv = vec(Sf.rho)
    dg = Sf.donor_groups; docts = dg.octs; didx = dg.idx; dcov = dg.flag
    gg = Sf.ghost_groups; gocts = gg.octs; gidx = gg.idx; gbits = gg.flag
    nt = Threads.nthreads()
    # (the two thread loops capture their own range tables: a binding assigned twice would be boxed, and the loops
    # would run dynamically typed, 500 MB of allocation per step)
    # stage 1: the moments of every donor cell at the requested time, by donor oct
    ranges1 = chunk_ranges(length(docts), nt)
    @threads :static for t in 1:nt
        @inbounds for g in ranges1[t]
            o = Int(docts[g])
            if w1 == zero(T)
                store_lanes!(dm, didx, g, oct_moments(foldv, o))
            elseif w0 == zero(T)
                store_lanes!(dm, didx, g, oct_moments(fnewv, o))
            else
                mn = oct_moments(fnewv, o); mo = oct_moments(foldv, o); cov = lane_mask(dcov[g])
                store_lanes!(dm, didx, g, ntuple(i -> vifelse(cov, mn[i], w0 * mo[i] + w1 * mn[i]), Val(10)))
            end
        end
    end
    # stage 2: the stencil sums, the rescale of the non-equilibrium, the rebuild, by ghost oct
    ranges2 = chunk_ranges(length(gocts), nt)
    @threads :static for t in 1:nt
        @inbounds for g in ranges2[t]
            sums = ntuple(k -> stencil_sum(dmv, dix, gw, Int(gidx[k, g])), Val(8))
            rebuild_oct!(ffv, ff, fuv, frv, Int(gocts[g]), gbits[g], sums, scale, omega_f)
        end
    end
    st
end

"""
Restrict level `l + 1` into the restricted slots of level `l`: moments from the fine level's current `f` on the
cubic stencil around the coarse centre (or the admissible children averaged), the non-equilibrium moments rescaled
by 2 tau_c / tau_f, the populations rebuilt at the coarse omega. Writes the coarse level's `f`, `u`, `rho`. Both
stages work on the eight slots of an oct at a time.
"""
function restrict!(st::State{T}, l::Int) where T
    Lc = st.mesh.levels[l]; Sc = st.levels[l]; Sf = st.levels[l + 1]
    isempty(Lc.restrict_slot) && return st
    tau_c = one(T) / Sc.omega; tau_f = one(T) / Sf.omega
    scale = T(2) * tau_c / tau_f * pre_collision_factor(Sf.omega)
    omega_c = Sc.omega
    rm = Sc.rsrc_m; rmv = vec(rm); six = Sc.restrict_six; ffv = vec(Sf.f)
    cf = Sc.f; cfv = vec(Sc.f); cuv = vec(Sc.u); crv = vec(Sc.rho)
    sg = Sc.rsrc_groups; socts = sg.octs; sidx = sg.idx
    rg = Sc.restrict_groups; rocts = rg.octs; ridx = rg.idx; rcubic = rg.flag
    wcubic = ntuple(n -> T(RESTRICT_W[n]), Val(64))
    nt = Threads.nthreads()
    ranges1 = chunk_ranges(length(socts), nt)
    @threads :static for t in 1:nt
        @inbounds for g in ranges1[t]
            store_lanes!(rm, sidx, g, oct_moments(ffv, Int(socts[g])))
        end
    end
    ranges2 = chunk_ranges(length(rocts), nt)
    @threads :static for t in 1:nt
        @inbounds for g in ranges2[t]
            cub = rcubic[g]
            full = ntuple(k -> restrict_sum(rmv, six, wcubic, (cub >> (k - 1)) & 0x01 == 0x01, Int(ridx[k, g])), Val(8))
            bits = 0x00
            for k in 1:8; full[k][4] && (bits |= UInt8(1) << (k - 1)); end
            bits == 0x00 && continue
            sums = ntuple(k -> (full[k][1], full[k][2], full[k][3]), Val(8))
            rebuild_oct!(cfv, cf, cuv, crv, Int(rocts[g]), bits, sums, scale, omega_c)
        end
    end
    st
end

"""
After a swap, the new read buffer's ghost slots hold the fill of two steps ago; copy the last fill from the old
buffer so that a finer level reading this level's ghosts as (trilinear, near-wall) donors sees a state at most one
step old until the next fill. The finest level's ghosts are never donors.
"""
function copy_ghosts!(st::State{T}, l::Int) where T
    Lv = st.mesh.levels[l]; S = st.levels[l]
    (l == nlevels(st.mesh) || isempty(Lv.ghost_slot)) && return st
    nt = Threads.nthreads(); ranges = chunk_ranges(length(Lv.ghost_slot), nt)
    @threads :static for t in 1:nt
        @inbounds for gi in ranges[t]
            s = Int(Lv.ghost_slot[gi]); k = slot_k(s); o = slot_oct(s)
            for i in 1:Q; S.f[k, i, o] = S.g[k, i, o]; end
            S.u[k, 1, o] = S.v[k, 1, o]; S.u[k, 2, o] = S.v[k, 2, o]; S.u[k, 3, o] = S.v[k, 3, o]; S.rho[k, o] = S.rhob[k, o]
        end
    end
    st
end

"""
One step of level `l` and, recursively, two of each finer level; the fine state is restricted after each sub-step
and the fine ghosts are filled for the next one. `collect` accumulates the momentum exchange of every level in
coarse-lattice units (a factor 8^(1-l) per level).
"""
function step_recursive!(st::State{T}, l::Int; collect::Bool=false) where T
    L = nlevels(st.mesh)
    step_level!(st, l; collect=collect, scale=8.0^(1 - l))
    swap_buffers!(st.levels[l])
    copy_ghosts!(st, l)
    if l < L
        step_recursive!(st, l + 1; collect=collect)     # the child's ghosts hold the coarse old time
        restrict!(st, l)                                # the half-time fine state into the covered slots
        fill_ghosts!(st, l + 1; weight=T(0.5))
        step_recursive!(st, l + 1; collect=collect)
        restrict!(st, l)                                # the new time
        fill_ghosts!(st, l + 1; weight=one(T))          # the child's ghosts for the next cycle
    end
    st
end

"Initial consistency after `initialise!`: restrict every level into its parent and fill every level's ghosts."
function couple_initial!(st::State{T}) where T
    L = nlevels(st.mesh)
    for l in L-1:-1:1
        restrict!(st, l)
        # the restriction wrote the coarse `f`; mirror it into `g` so that a time blend at step 1 is consistent
        copyto!(st.levels[l].g, st.levels[l].f); copyto!(st.levels[l].v, st.levels[l].u); copyto!(st.levels[l].rhob, st.levels[l].rho)
    end
    for l in 2:L
        fill_ghosts!(st, l; weight=one(T))
        copyto!(st.levels[l].g, st.levels[l].f); copyto!(st.levels[l].v, st.levels[l].u); copyto!(st.levels[l].rhob, st.levels[l].rho)
    end
    st
end
