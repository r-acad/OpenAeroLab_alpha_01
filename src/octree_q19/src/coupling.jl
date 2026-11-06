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
# every donor or source cell are computed once per transfer into a buffer; the stencil sums read the buffer.

"rho, u and Pi_neq of slot (k, o) from a population buffer."
@inline function slot_moments(f::Array{T,3}, k::Int, o::Int) where T
    fs = ntuple(i -> (@inbounds f[k, i, o]), Val(Q))
    rho, ux, uy, uz = macros_shifted(fs)
    p = pi_neq_shifted(fs, rho, ux, uy, uz)
    rho, ux, uy, uz, p
end

"The moments of slot (k, o) of f into column d of the buffer m."
@inline function store_moments!(m::Matrix{T}, d::Int, f::Array{T,3}, k::Int, o::Int) where T
    rho, ux, uy, uz, p = slot_moments(f, k, o)
    @inbounds begin
        m[1, d] = rho; m[2, d] = ux; m[3, d] = uy; m[4, d] = uz
        m[5, d] = p[1]; m[6, d] = p[2]; m[7, d] = p[3]; m[8, d] = p[4]; m[9, d] = p[5]; m[10, d] = p[6]
    end
    nothing
end

"Rebuild a post-collision-equivalent shifted population tuple from moments at relaxation rate omega."
@inline function rebuild(rho::T, ux::T, uy::T, uz::T, p::NTuple{6,T}, omega::T) where T
    usq = ux * ux + uy * uy + uz * uz
    keep = one(T) - omega
    a3 = a3_terms(p, ux, uy, uz)
    eq = ntuple(i -> feq_shifted(i, rho, ux, uy, uz, usq), Val(Q))
    out = ntuple(i -> (@inbounds eq[i] + keep * fneq_regularised(i, p, a3)), Val(Q))
    positive_shifted(out, eq)
end

# Factor turning the non-equilibrium moments of a post-collision state into those of its pre-collision state,
# 1 / (1 - omega); near omega = 1 the post-collision non-equilibrium carries no information and the transfer
# drops it (the stress across the interface is then slightly off; tau about 1 is outside the LES range of use).
@inline pre_collision_factor(omega::T) where T = abs(one(T) - omega) > T(0.05) ? one(T) / (one(T) - omega) : zero(T)

"""
Fill the ghost slots of level `l` from level `l - 1` at `weight` along the coarse step: 1 = the coarse new buffer
(`f`), 0 = the old one (`g`), 1/2 = the blend (covered donors, which hold the fine average restricted at the half
time, enter unblended from `f`). The donor moments are interpolated with the stored stencil weights, the
non-equilibrium moments rescaled to the fine level, the populations rebuilt at the fine omega. Writes the fine
level's read buffers (f, u, rho).
"""
function fill_ghosts!(st::State{T}, l::Int; weight::T=one(T)) where T
    Lf = st.mesh.levels[l]; Lc = st.mesh.levels[l - 1]; Sf = st.levels[l]; Sc = st.levels[l - 1]
    isempty(Lf.ghost_slot) && return st
    fnew = Sc.f; fold = Sc.g; ckind = Lc.kind
    # molecular relaxation times of both levels (lattice units of each): tau_f - 1/2 = 2 (tau_c - 1/2)
    tau_c = one(T) / Sc.omega; tau_f = one(T) / Sf.omega
    scale = tau_f / (T(2) * tau_c) * pre_collision_factor(Sc.omega)
    omega_f = Sf.omega
    w1 = weight; w0 = one(T) - weight
    donors = Sf.donor_cells; dm = Sf.donor_m; dix = Sf.ghost_dix; gw = Lf.ghost_w
    nt = Threads.nthreads()
    # (the two thread loops capture their own range tables: a binding assigned twice would be boxed, and the loops
    # would run dynamically typed, 500 MB of allocation per step)
    # stage 1: the moments of every donor cell at the requested time
    ranges1 = chunk_ranges(length(donors), nt)
    @threads :static for t in 1:nt
        @inbounds for d in ranges1[t]
            s = Int(donors[d]); k = slot_k(s); o = slot_oct(s)
            if w1 == zero(T)
                store_moments!(dm, d, fold, k, o)
            elseif w0 == zero(T) || ckind[k, o] == K_COVERED
                store_moments!(dm, d, fnew, k, o)
            else
                r1, u1, v1, z1, p1 = slot_moments(fnew, k, o)
                r0, u0, v0, z0, p0 = slot_moments(fold, k, o)
                dm[1, d] = w0 * r0 + w1 * r1; dm[2, d] = w0 * u0 + w1 * u1; dm[3, d] = w0 * v0 + w1 * v1; dm[4, d] = w0 * z0 + w1 * z1
                dm[5, d] = w0 * p0[1] + w1 * p1[1]; dm[6, d] = w0 * p0[2] + w1 * p1[2]; dm[7, d] = w0 * p0[3] + w1 * p1[3]
                dm[8, d] = w0 * p0[4] + w1 * p1[4]; dm[9, d] = w0 * p0[5] + w1 * p1[5]; dm[10, d] = w0 * p0[6] + w1 * p1[6]
            end
        end
    end
    # stage 2: the stencil sums, the rescale of the non-equilibrium, the rebuild
    ranges2 = chunk_ranges(length(Lf.ghost_slot), nt)
    @threads :static for t in 1:nt
        @inbounds for gi in ranges2[t]
            s = Int(Lf.ghost_slot[gi]); k = slot_k(s); o = slot_oct(s)
            # scalar accumulators only: a tuple rebuilt by a closure that captures itself is boxed
            rho = zero(T); ux = zero(T); uy = zero(T); uz = zero(T)
            a1 = zero(T); a2 = zero(T); a3 = zero(T); a4 = zero(T); a5 = zero(T); a6 = zero(T)
            for n in 1:64
                d = Int(dix[n, gi]); d == 0 && continue
                wn = T(gw[n, gi])
                rho += wn * dm[1, d]; ux += wn * dm[2, d]; uy += wn * dm[3, d]; uz += wn * dm[4, d]
                a1 += wn * dm[5, d]; a2 += wn * dm[6, d]; a3 += wn * dm[7, d]; a4 += wn * dm[8, d]; a5 += wn * dm[9, d]; a6 += wn * dm[10, d]
            end
            pf = (scale * a1, scale * a2, scale * a3, scale * a4, scale * a5, scale * a6)
            fs = rebuild(rho, ux, uy, uz, pf, omega_f)
            for i in 1:Q; Sf.f[k, i, o] = fs[i]; end
            Sf.u[k, 1, o] = ux; Sf.u[k, 2, o] = uy; Sf.u[k, 3, o] = uz; Sf.rho[k, o] = rho
        end
    end
    st
end

"""
Restrict level `l + 1` into the restricted slots of level `l`: moments from the fine level's current `f` on the
cubic stencil around the coarse centre (or the admissible children averaged), the non-equilibrium moments rescaled
by 2 tau_c / tau_f, the populations rebuilt at the coarse omega. Writes the coarse level's `f`, `u`, `rho`.
"""
function restrict!(st::State{T}, l::Int) where T
    Lc = st.mesh.levels[l]; Sc = st.levels[l]; Sf = st.levels[l + 1]
    isempty(Lc.restrict_slot) && return st
    tau_c = one(T) / Sc.omega; tau_f = one(T) / Sf.omega
    scale = T(2) * tau_c / tau_f * pre_collision_factor(Sf.omega)
    omega_c = Sc.omega
    srcs = Sc.rsrc_cells; rm = Sc.rsrc_m; six = Sc.restrict_six; cubic = Lc.restrict_cubic; ff = Sf.f
    nt = Threads.nthreads()
    ranges1 = chunk_ranges(length(srcs), nt)
    @threads :static for t in 1:nt
        @inbounds for d in ranges1[t]
            s = Int(srcs[d]); store_moments!(rm, d, ff, slot_k(s), slot_oct(s))
        end
    end
    ranges2 = chunk_ranges(length(Lc.restrict_slot), nt)
    @threads :static for t in 1:nt
        @inbounds for ri in ranges2[t]
            s = Int(Lc.restrict_slot[ri]); k = slot_k(s); o = slot_oct(s)
            rho = zero(T); ux = zero(T); uy = zero(T); uz = zero(T)
            a1 = zero(T); a2 = zero(T); a3 = zero(T); a4 = zero(T); a5 = zero(T); a6 = zero(T)
            if cubic[ri]
                for n in 1:64
                    d = Int(six[n, ri]); wn = T(RESTRICT_W[n])
                    rho += wn * rm[1, d]; ux += wn * rm[2, d]; uy += wn * rm[3, d]; uz += wn * rm[4, d]
                    a1 += wn * rm[5, d]; a2 += wn * rm[6, d]; a3 += wn * rm[7, d]; a4 += wn * rm[8, d]; a5 += wn * rm[9, d]; a6 += wn * rm[10, d]
                end
            else
                cnt = 0
                for n in STENCIL_INNER
                    d = Int(six[n, ri]); d == 0 && continue
                    rho += rm[1, d]; ux += rm[2, d]; uy += rm[3, d]; uz += rm[4, d]
                    a1 += rm[5, d]; a2 += rm[6, d]; a3 += rm[7, d]; a4 += rm[8, d]; a5 += rm[9, d]; a6 += rm[10, d]
                    cnt += 1
                end
                cnt == 0 && continue
                inv = one(T) / T(cnt)
                rho *= inv; ux *= inv; uy *= inv; uz *= inv; a1 *= inv; a2 *= inv; a3 *= inv; a4 *= inv; a5 *= inv; a6 *= inv
            end
            pc = (scale * a1, scale * a2, scale * a3, scale * a4, scale * a5, scale * a6)
            fs = rebuild(rho, ux, uy, uz, pc, omega_c)
            for i in 1:Q; Sc.f[k, i, o] = fs[i]; end
            Sc.u[k, 1, o] = ux; Sc.u[k, 2, o] = uy; Sc.u[k, 3, o] = uz; Sc.rho[k, o] = rho
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
