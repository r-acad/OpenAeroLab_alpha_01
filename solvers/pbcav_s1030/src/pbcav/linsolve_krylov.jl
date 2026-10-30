# =====================================================================================================================
# linsolve_krylov.jl -- the Krylov methods (plan 5.1): Notay's flexible CG for the SPD pressure equation (T0'), FGMRES(10)
# for T1/T2 (stage P1), BiCGStab for the transport equations (a fallback, not built: plan 5.4 makes line sweeps the
# primary). Float64, deterministic two-pass reductions, the scalars kept on the device. Stage P0b, 14 Oct 2026.
#
# DETERMINISTIC REDUCTIONS. `pb_dot` sums fixed chunks of PB_DOT_CHUNK entries, one work item per chunk in index order
# (pass 1), then one work item sums the chunk partials in index order into a device scalar (pass 2). The sum order depends
# only on the length, never on the backend's scheduling, and no atomics are used (plan 5.5).
# DEVICE SCALARS. FCG keeps its scalars (z.q_old, p.q, p.r, r.r, ...) in a small device vector; the update kernels read
# them there, so an iteration needs no host read. The host reads r.r once every `check_every` iterations to test
# convergence (plan 5.5: every four or five on the GPU; 1 here, so the counts are exact iteration counts, as P0a's).
# Test reference: Krylov.jl 0.10.6 (src/pbcav/test/Project.toml), its CG with the same fixed SPD preconditioner.
# KernelAbstractions issue 652: no kernel here uses @Const (see PBCav.jl).
# =====================================================================================================================

"Entries summed serially by one work item in pass 1 of `pb_dot`."
const PB_DOT_CHUNK = 1024

@kernel function _k_dot_partial!(part, x, y, n::Int32, chunk::Int32)
    k = @index(Global)
    @inbounds begin
        a = (k - 1)*chunk + 1; b = min(k*chunk, n); s = 0.0
        for i in a:b; s += x[i]*y[i]; end
        part[k] = s
    end
end
@kernel function _k_sum_into!(sc, part, nparts::Int32, slot::Int32)
    t = @index(Global)
    @inbounds begin
        s = 0.0
        for k in Int32(1):nparts; s += part[k]; end
        sc[slot] = s
    end
end

"Work space of the reductions and of FCG for vectors of length n on `backend`."
struct KrylovWork{V<:AbstractVector{Float64}}
    n::Int
    part::V
    sc::V                 # device scalars: 1 z.q_old, 2 p_old.q_old, 3 p.q, 4 p.r, 5 r.r, 6 b.b
    r::V; z::V; p::V; q::V; pold::V; qold::V
end
function KrylovWork(n::Integer; backend=CPU())
    z(m) = KernelAbstractions.zeros(backend, Float64, m)
    KrylovWork(Int(n), z(cld(n, PB_DOT_CHUNK)), z(8), z(n), z(n), z(n), z(n), z(n), z(n))
end

"x.y into the device scalar sc[slot] (two passes)."
function _dot_dev!(sc, slot::Integer, x, y, part; backend=CPU())
    n = length(x)
    if PB_DOT[] === :tree                  # W7 (opt-in): the coalesced workgroup tree, same partial count bound
        nb = cld(n, PB_TREE_BLOCK)
        _launch_wg!(_k_dot_tree!, backend, nb*PB_TREE_WG, PB_TREE_WG, part, x, y, Int32(n))
        _launch_wg!(_k_sum_tree!, backend, PB_TREE_WG, PB_TREE_WG, sc, part, Int32(nb), Int32(slot))
        return nothing
    end
    np = cld(n, PB_DOT_CHUNK)
    _launch!(_k_dot_partial!, backend, np, part, x, y, Int32(n), Int32(PB_DOT_CHUNK))
    _launch!(_k_sum_into!, backend, 1, sc, part, Int32(np), Int32(slot))
    nothing
end

# ---- W7 (speed; plan Amendment 13; opt-in, PBConfig.dot): THE TREE DOT ---------------------------------------------------
# The chunked dot above gives one work item 1,024 consecutive entries: on a GPU the loads of a warp are 8 kB apart (one
# memory transaction each) and only cld(n, 1024) work items run (352 for the 360,192 entries of a coupled vector at 120k
# cells). The tree form gives each workgroup of PB_TREE_WG = 256 work items a block of PB_TREE_BLOCK = 2,048 consecutive
# entries, work item t summing entries t, t + 256, ... (coalesced), then a fixed binary tree in local memory; pass 2 is one
# workgroup summing the block partials the same way. The order depends only on n, never on the scheduling, and it is the
# same on every backend (deterministic, no atomics), but it is not the chunked order: the sums differ by rounding (class N).
# PB_TREE_BLOCK equals PB_DOT_CHUNK x 2, so the partial buffers sized for the chunked form are large enough.
"The dot form (PBConfig.dot, set by each step): :chunk (THE DEFAULT, the form before W7) or :tree."
const PB_DOT = Ref(:chunk)
function pb_dot_mode!(mode::Symbol)
    mode in (:chunk, :tree) || throw(ArgumentError("PBConfig.dot must be :chunk or :tree (got :$mode)"))
    PB_DOT[] = mode
end
const PB_TREE_WG = 256
const PB_TREE_BLOCK = 2048
@kernel function _k_dot_tree!(part, x, y, n::Int32)
    g = @index(Group, Linear)
    t = @index(Local, Linear)
    sh = @localmem Float64 (256,)
    @inbounds begin
        base = (g - 1)*2048; s = 0.0
        for k in 0:7
            i = base + k*256 + t
            i <= n && (s += x[i]*y[i])
        end
        sh[t] = s
    end
    @synchronize
    for st in 1:8
        @inbounds begin
            wd = 256 >> st
            t <= wd && (sh[t] += sh[t + wd])
        end
        @synchronize
    end
    @inbounds t == 1 && (part[g] = sh[1])
end
@kernel function _k_sum_tree!(sc, part, nparts::Int32, slot::Int32)
    t = @index(Local, Linear)
    sh = @localmem Float64 (256,)
    @inbounds begin
        s = 0.0; k = t
        while k <= nparts
            s += part[k]; k += 256
        end
        sh[t] = s
    end
    @synchronize
    for st in 1:8
        @inbounds begin
            wd = 256 >> st
            t <= wd && (sh[t] += sh[t + wd])
        end
        @synchronize
    end
    @inbounds t == 1 && (sc[slot] = sh[1])
end

"""
    pb_dot(x, y; backend=CPU()) -> Float64

The deterministic two-pass dot product (the reduction every Krylov method uses), read back to the host.
"""
function pb_dot(x, y; backend=CPU())
    length(x) == length(y) || throw(DimensionMismatch("pb_dot: lengths $(length(x)) and $(length(y))"))
    part = KernelAbstractions.zeros(backend, Float64, max(cld(length(x), PB_DOT_CHUNK), 1))
    sc = KernelAbstractions.zeros(backend, Float64, 1)
    _dot_dev!(sc, 1, x, y, part; backend)
    pb_sync(backend)
    _host(sc)[1]
end

"r = b - A x (face system, four-face gather)."
@kernel function _k_fs_resid!(r, b, x, dg, up, lo, cface, own, nei, nf::Int32)
    c = @index(Global)
    @inbounds begin
        acc = dg[c]*x[c]
        for k in 1:4
            code = cface[k, c]
            if 0 < code <= nf
                acc += up[code]*x[nei[code]]
            elseif code < 0
                acc += lo[-code]*x[own[-code]]
            end
        end
        r[c] = b[c] - acc
    end
end
"p = z - (sc[1]/sc[2]) pold, or p = z on the first iteration (first = 1)."
@kernel function _k_fcg_dir!(p, z, pold, sc, first::Int32)
    i = @index(Global)
    @inbounds p[i] = first == Int32(1) ? z[i] : z[i] - (sc[1]/sc[2])*pold[i]
end
"x += (sc[4]/sc[3]) p; r -= (sc[4]/sc[3]) q."
@kernel function _k_fcg_update!(x, r, p, q, sc)
    i = @index(Global)
    @inbounds begin
        al = sc[4]/sc[3]
        x[i] += al*p[i]; r[i] -= al*q[i]
    end
end
@kernel function _k_copy_slot!(sc, dst::Int32, src::Int32)
    t = @index(Global)
    @inbounds sc[dst] = sc[src]
end

"""
    fcg!(x, sys::FaceSystem, mesh, M!; tol=1e-9, atol=0.0, maxit=200, check_every=1, work=nothing, backend=CPU(),
         record=false) -> NamedTuple

Notay's FCG(1) for the SPD system `sys` (A x = sys.rhs), from the given x, preconditioned by `M!(z, r)` (z = M r; M may vary
between iterations). Stops when ||r||_2 <= max(tol ||b||_2, atol), checked every `check_every` iterations. Returns
`(iters, converged, rel, res, b_norm, hist)`: `rel` = ||r||/||b|| and `res` = ||r|| at the end, `hist` the relative
residual at each check (with `record`). With a fixed SPD M it equals PCG in exact arithmetic (Notay 2000).
"""
function fcg!(x, sys::FaceSystem, m::PBMesh, M!; tol::Real=1e-9, atol::Real=0.0, maxit::Integer=200,
              check_every::Integer=1, work=nothing, backend=CPU(), record::Bool=false)
    n = m.nc
    w = work === nothing ? KrylovWork(n; backend) : work
    w.n == n || throw(DimensionMismatch("fcg!: work space for $(w.n) cells, system has $n"))
    fs_args = (sys.diag, sys.upper, sys.lower, m.cface, m.own, m.nei, Int32(m.nf))
    _launch!(_k_fs_resid!, backend, n, w.r, sys.rhs, x, fs_args...)
    _dot_dev!(w.sc, 6, sys.rhs, sys.rhs, w.part; backend)
    _dot_dev!(w.sc, 5, w.r, w.r, w.part; backend)
    pb_sync(backend)
    s = _host(w.sc); bn = sqrt(s[6]); rn = sqrt(s[5])
    goal = max(tol*bn, atol)
    hist = Float64[]
    record && push!(hist, bn > 0 ? rn/bn : rn)
    (rn <= goal || bn == 0) && return (iters=0, converged=true, rel=bn > 0 ? rn/bn : 0.0, res=rn, b_norm=bn, hist=hist)
    it = 0; conv = false
    while it < maxit
        it += 1
        M!(w.z, w.r)
        it > 1 && _dot_dev!(w.sc, 1, w.z, w.qold, w.part; backend)
        _launch!(_k_fcg_dir!, backend, n, w.p, w.z, w.pold, w.sc, Int32(it == 1 ? 1 : 0))
        facesys_mul!(w.q, sys, w.p, m; backend)
        _dot_dev!(w.sc, 3, w.p, w.q, w.part; backend)
        _dot_dev!(w.sc, 4, w.p, w.r, w.part; backend)
        _launch!(_k_fcg_update!, backend, n, x, w.r, w.p, w.q, w.sc)
        _launch!(_k_copy_slot!, backend, 1, w.sc, Int32(2), Int32(3))
        copyto!(w.pold, w.p); copyto!(w.qold, w.q)
        if it % check_every == 0 || it == maxit
            _dot_dev!(w.sc, 5, w.r, w.r, w.part; backend)
            pb_sync(backend)
            s = _host(w.sc); rn = sqrt(s[5])
            isfinite(rn) || break
            record && push!(hist, rn/bn)
            rn <= goal && (conv = true; break)
        end
    end
    pb_sync(backend)
    rn = sqrt(_host(w.sc)[5])
    (iters=it, converged=conv, rel=rn/bn, res=rn, b_norm=bn, hist=hist)
end

"FGMRES(m), right-preconditioned. Stage P1."
fgmres!(x, sys, mesh, M!; m=10, tol, maxit) = _nyi(:fgmres!, "stage P1")
"BiCGStab for the transport equations: a fallback (plan 5.4); not built while the line sweeps suffice."
bicgstab!(x, sys, mesh, M!; tol, maxit) = _nyi(:bicgstab!, "stage P1 (a fallback of plan 5.4; the line sweeps are primary)")

# ---- the other deterministic reductions (sum, max |.|), same two-pass chunking ----------------------------------------
@kernel function _k_red_partial!(part, x, n::Int32, chunk::Int32, op::Int32)
    k = @index(Global)
    @inbounds begin
        a = (k - 1)*chunk + 1; b = min(k*chunk, n); s = 0.0
        if op == Int32(1)
            for i in a:b; s += x[i]; end
        else
            for i in a:b; v = abs(x[i]); s = (v > s || isnan(v)) ? v : s; end
        end
        part[k] = s
    end
end
@kernel function _k_red_final!(out, part, nparts::Int32, op::Int32)
    t = @index(Global)
    @inbounds begin
        s = 0.0
        if op == Int32(1)
            for k in Int32(1):nparts; s += part[k]; end
        else
            for k in Int32(1):nparts; v = part[k]; s = (v > s || isnan(v)) ? v : s; end
        end
        out[1] = s
    end
end
function _reduce(x, op::Int32; backend=CPU())
    n = length(x); n == 0 && return 0.0
    np = cld(n, PB_DOT_CHUNK)
    part = KernelAbstractions.zeros(backend, Float64, np); out = KernelAbstractions.zeros(backend, Float64, 1)
    _launch!(_k_red_partial!, backend, np, part, x, Int32(n), Int32(PB_DOT_CHUNK), op)
    _launch!(_k_red_final!, backend, 1, out, part, Int32(np), op)
    pb_sync(backend)
    _host(out)[1]
end
"Deterministic two-pass sum (fixed chunks, fixed order; no atomics)."
pb_sum(x; backend=CPU()) = _reduce(x, Int32(1); backend)
"max |x_i| (NaN if any entry is NaN), two-pass."
pb_maxabs(x; backend=CPU()) = _reduce(x, Int32(2); backend)
