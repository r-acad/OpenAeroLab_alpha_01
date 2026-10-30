# =====================================================================================================================
# linsolve_mg.jl -- the pressure preconditioner chosen by stage P0a (plan 5.1-5.2; P0a record README section 3):
# Schaffer's semicoarsening multigrid across the j-lines with line-solve interpolation, V(1,1), j-line smoothing by
# colours (coarse lines, then fine), the `:fsep` coarse/fine split, Galerkin coarse operators, WHOLE CYCLE IN FLOAT64
# (decision P0a-1), the coarsest line by an explicit Float64 inverse up to about 700 cells (decision P0a-3), and the
# build-time refusal of any non-positive line-factor or interpolation pivot. Stage P0b, 14 Oct 2026.
#
# PORT SOURCE. docs/_records/2026-10-13_pressure_based/p0a/schaffer/p0a_step3_schaffer_20261014.jl (its partner maps,
# connections, fold mirror, line-solve weights and C/F split) and the shared harness p0a_common_20261014.jl (its line
# factor, line solve, smoother order and V-cycle), transcribed operation for operation:
#   * the hierarchy's integer data (lines, split, colours, partner maps, connections, fold mirrors) are identical;
#   * every floating-point operation of the build is the reference's, in the reference's order: the line factors, the
#     Thomas solves of the weights, the Galerkin product (P'A)P accumulated row by row in increasing inner index (the order
#     SparseArrays' column-oriented product uses for each entry), the symmetrisation (A + A')/2 and the zero drop. The suite
#     (tools/test_pbcav.jl) checks every level's operator against the reference's BIT FOR BIT on N60R and N60Rx2;
#   * the cycle's arithmetic is the reference's (residual b - A x gathered in increasing column order, Thomas line solves,
#     x + z, P'r, x + P xc) except the coarsest dense product, a plain row sum here and BLAS in the reference: the cycle's
#     output therefore agrees to round-off, not bitwise (tested).
# The study's other variants (V(2,2), alternating i/j smoothing, the K-cycle) are not ported; the greedy split
# (`split=:committed`) is kept only so the build's refusal of it can be tested.
#
# WHERE IT RUNS. The CYCLE is KernelAbstractions kernels on the hierarchy's backend: per smoothing colour a residual and a
# batched line solve (one work item per line, Thomas along it), per level visit a residual, the restriction and the
# prolongation with the correction, and one coarsest kernel (a dense product or a line solve) -- the P0a KERNEL_RULE count,
# 147 kernels per cycle on N60R and 162 on N60Rx2, counted by `h.launches`. No host loop over cells runs in a cycle.
# THE BUILD runs on the host (integer graph work and the Galerkin products), reading the operator back once. Since review
# F5 (14 Oct 2026, plan 24) it runs once per march, recording its numeric plan (`plan = true`), and `mg_refresh!`
# recomputes every value from a new operator with KernelAbstractions kernels on the fixed sparsity ("THE NUMERIC REFRESH",
# at the end of this file); the coarsest level's inverse is the one host step left (by default; a device Gauss-Jordan is
# the option). The suite measures that the split, the colours and the partner maps are the same at contrasts 1 and 4.3e4
# (plan 5.2's "the line split and the partner maps depend only on the mesh, P0b to confirm"), and F5's probes that they
# stay the same along N60R's march and across a contrast change on its real operator.
# Runtime dependencies: none beyond LinearAlgebra (inv of the coarsest matrix); the sparse algebra is the host CSR below.
# =====================================================================================================================

"Largest coarsest line solved by an explicit Float64 inverse (decision P0a-3: about 700 cells)."
const MG_COARSE_INVERSE_MAX = 700

# ---- a minimal host CSR (rows sorted by column; no run-time SparseArrays, plan 6.3) ---------------------------------
"""
    HCSR

Host compressed sparse rows: `n` rows, `m` columns, `ptr` (n+1), `col` (sorted within a row), `val`.
"""
struct HCSR
    n::Int; m::Int
    ptr::Vector{Int}; col::Vector{Int}; val::Vector{Float64}
end

"COO -> CSR, duplicates summed in input order (as `sparse(I, J, V)` does), columns sorted within each row."
function hcsr_coo(I::AbstractVector{<:Integer}, J::AbstractVector{<:Integer}, V::AbstractVector{Float64}, n::Int, m::Int)
    nz = length(I)
    cnt = zeros(Int, n + 1)
    for t in 1:nz; cnt[I[t]+1] += 1; end
    ptr0 = cumsum(vcat(1, cnt[2:end])); fill = copy(ptr0)
    cs = Vector{Int}(undef, nz); vs = Vector{Float64}(undef, nz)
    for t in 1:nz; r = I[t]; cs[fill[r]] = J[t]; vs[fill[r]] = V[t]; fill[r] += 1; end
    ptr = ones(Int, n + 1); col = Int[]; val = Float64[]
    sizehint!(col, nz); sizehint!(val, nz)
    for r in 1:n
        a = ptr0[r]; b = ptr0[r+1] - 1
        if a <= b
            perm = sortperm(view(cs, a:b); alg=Base.Sort.MergeSort)          # stable: duplicates stay in input order
            last = 0
            for q in perm
                c = cs[a+q-1]; v = vs[a+q-1]
                if c == last
                    val[end] += v
                else
                    push!(col, c); push!(val, v); last = c
                end
            end
        end
        ptr[r+1] = length(col) + 1
    end
    HCSR(n, m, ptr, col, val)
end

"The entry (r, c), zero when not stored (binary search)."
function hget(A::HCSR, r::Int, c::Int)
    lo = A.ptr[r]; hi = A.ptr[r+1] - 1
    while lo <= hi
        mid = (lo + hi) >>> 1; cm = A.col[mid]
        cm == c && return A.val[mid]
        cm < c ? (lo = mid + 1) : (hi = mid - 1)
    end
    0.0
end

"Transpose (values copied, no arithmetic)."
function htranspose(A::HCSR)
    cnt = zeros(Int, A.m + 1)
    for c in A.col; cnt[c+1] += 1; end
    ptr = cumsum(vcat(1, cnt[2:end])); fill = copy(ptr)
    col = Vector{Int}(undef, length(A.col)); val = Vector{Float64}(undef, length(A.val))
    for r in 1:A.n, q in A.ptr[r]:A.ptr[r+1]-1
        c = A.col[q]; col[fill[c]] = r; val[fill[c]] = A.val[q]; fill[c] += 1
    end
    HCSR(A.m, A.n, ptr, col, val)
end

"""
    hmul(A, B) -> A*B

Row by row: C[i,j] = sum over k of A[i,k] B[k,j], accumulated in increasing k (A's row order), the first term assigned.
This is the order SparseArrays' column-oriented product accumulates each entry in, so the products agree bit for bit.
"""
function hmul(A::HCSR, B::HCSR)
    A.m == B.n || throw(DimensionMismatch("hmul: $(A.n)x$(A.m) * $(B.n)x$(B.m)"))
    acc = zeros(B.m); seen = falses(B.m); cols = Int[]
    ptr = ones(Int, A.n + 1); col = Int[]; val = Float64[]
    for i in 1:A.n
        empty!(cols)
        for q in A.ptr[i]:A.ptr[i+1]-1
            k = A.col[q]; a = A.val[q]
            for e in B.ptr[k]:B.ptr[k+1]-1
                j = B.col[e]
                if seen[j]
                    acc[j] += a*B.val[e]
                else
                    seen[j] = true; acc[j] = a*B.val[e]; push!(cols, j)
                end
            end
        end
        sort!(cols)
        for j in cols; push!(col, j); push!(val, acc[j]); seen[j] = false; end
        ptr[i+1] = length(col) + 1
    end
    HCSR(A.n, B.m, ptr, col, val)
end

"(A + A')/2 with exact zeros dropped (the reference's `Ac = (Ac + Ac')/2; dropzeros!(Ac)`)."
function hsym_drop(A::HCSR)
    T = htranspose(A)
    ptr = ones(Int, A.n + 1); col = Int[]; val = Float64[]
    for r in 1:A.n
        a = A.ptr[r]; ae = A.ptr[r+1] - 1; t = T.ptr[r]; te = T.ptr[r+1] - 1
        while a <= ae || t <= te
            ca = a <= ae ? A.col[a] : typemax(Int); ct = t <= te ? T.col[t] : typemax(Int)
            if ca == ct
                v = (A.val[a] + T.val[t])/2; c = ca; a += 1; t += 1
            elseif ca < ct
                v = (A.val[a] + 0.0)/2; c = ca; a += 1     # sparse + sparse adds the missing entry as a stored value
            else
                v = (0.0 + T.val[t])/2; c = ct; t += 1
            end
            v == 0 || (push!(col, c); push!(val, v))
        end
        ptr[r+1] = length(col) + 1
    end
    HCSR(A.n, A.m, ptr, col, val)
end

"Host product y = A x (for setup checks and the suite)."
function hmatvec(A::HCSR, x::AbstractVector{Float64})
    y = zeros(A.n)
    for r in 1:A.n
        s = 0.0
        for q in A.ptr[r]:A.ptr[r+1]-1; s += A.val[q]*x[A.col[q]]; end
        y[r] = s
    end
    y
end
"The dense matrix of a (small) HCSR."
function hdense(A::HCSR)
    D = zeros(A.n, A.m)
    for r in 1:A.n, q in A.ptr[r]:A.ptr[r+1]-1; D[r, A.col[q]] = A.val[q]; end
    D
end

"""
    facesys_hcsr(sys::FaceSystem, mesh::PBMesh) -> HCSR

The host CSR of a face system: diagonal plus, per internal face, `upper` at (own, nei) and `lower` at (nei, own). Read back
from the backend once (setup).
"""
function facesys_hcsr(sys::FaceSystem, m::PBMesh)
    dg = _host(sys.diag); up = _host(sys.upper); lo = _host(sys.lower); own = _host(m.own); nei = _host(m.nei)
    nf = m.nf; nc = m.nc
    I = Vector{Int}(undef, 2nf + nc); J = similar(I); V = Vector{Float64}(undef, 2nf + nc)
    for q in 1:nf
        I[2q-1] = own[q]; J[2q-1] = nei[q]; V[2q-1] = up[q]
        I[2q] = nei[q]; J[2q] = own[q]; V[2q] = lo[q]
    end
    for c in 1:nc; I[2nf+c] = c; J[2nf+c] = c; V[2nf+c] = dg[c]; end
    hcsr_coo(I, J, V, nc, nc)
end

# ---- the reference's line machinery (p0a_common_20261014.jl, verbatim arithmetic) -----------------------------------
"Line factor (a = coupling to the previous cell, den the Thomas pivots, cp the modified super-diagonal) of `idx` in A."
function _linefac(A::HCSR, idx::Vector{Int})
    n = length(idx); a = zeros(n); b = zeros(n); c = zeros(n)
    for q in 1:n
        r = idx[q]; b[q] = hget(A, r, r)
        q > 1 && (a[q] = hget(A, r, idx[q-1])); q < n && (c[q] = hget(A, r, idx[q+1]))
    end
    den = zeros(n); cp = zeros(n)
    den[1] = b[1]; cp[1] = c[1]/den[1]
    for q in 2:n; den[q] = b[q] - a[q]*cp[q-1]; cp[q] = c[q]/den[q]; end
    a, den, cp
end
_lineof(lines, n) = (lo = zeros(Int, n); for (l, L) in enumerate(lines), r in L; lo[r] = l; end; lo)
function _line_graph(A::HCSR, lines)
    lo = _lineof(lines, A.n); nb = [Set{Int}() for _ in lines]
    for r in 1:A.n, q in A.ptr[r]:A.ptr[r+1]-1
        c = A.col[q]; (lo[r] != lo[c] && lo[r] > 0 && lo[c] > 0) && push!(nb[lo[c]], lo[r])
    end
    nb, lo
end
function _greedy_colours(nb)
    col = zeros(Int, length(nb))
    for l in eachindex(nb)
        used = Set(col[q] for q in nb[l] if col[q] > 0); c = 1
        while c in used; c += 1; end
        col[l] = c
    end
    col
end
_positions(lines, n) = (p = zeros(Int, n); for L in lines, (q, r) in enumerate(L); p[r] = q; end; p)

# ---- the Schaffer construction (p0a_step3_schaffer_20261014.jl, verbatim logic) -------------------------------------
"Level-1 partner map: pm[r] = [(neighbouring line, partner cell)], the strongest off-line coupling of r into each line."
function _partners_fine(A::HCSR, lo)
    n = A.n; pm = [Tuple{Int,Int}[] for _ in 1:n]; multi = 0
    for r in 1:n
        best = Dict{Int,Tuple{Int,Float64}}(); cnt = Dict{Int,Int}()
        for q in A.ptr[r]:A.ptr[r+1]-1
            c = A.col[q]; c == r && continue; L2 = lo[c]; L2 == lo[r] && continue
            w = abs(A.val[q]); cnt[L2] = get(cnt, L2, 0) + 1
            (haskey(best, L2) && best[L2][2] >= w) || (best[L2] = (c, w))
        end
        for (L2, cw) in sort(collect(best)); push!(pm[r], (L2, cw[1])); end
        multi += count(>(1), values(cnt))
    end
    pm, multi
end
"Coarse partner map by composition through the removed F lines and through the fold mirror (C - F - fold - F - C)."
function _partners_coarse(pm, lo, isC, lc, cidx, nc, mir)
    pmc = [Tuple{Int,Int}[] for _ in 1:nc]
    for rf in eachindex(pm)
        cidx[rf] > 0 || continue
        r = cidx[rf]; l = lo[rf]
        for (L2, p) in pm[rf]
            isC[L2] && push!(pmc[r], (lc[L2], cidx[p]))
        end
        for (L2, p) in pm[rf]
            isC[L2] && continue
            for (L3, p2) in pm[p]; (isC[L3] && L3 != l) && push!(pmc[r], (lc[L3], cidx[p2])); end
            pq = mir[p]; pq > 0 || continue
            for (L3, p2) in pm[pq]
                (isC[L3] && L3 != l && !any(L == L3 for (L, _) in pm[p])) && push!(pmc[r], (lc[L3], cidx[p2]))
            end
        end
        unique!(pmc[r]); sort!(pmc[r])
    end
    multi = 0
    for r in 1:nc, k in 2:length(pmc[r]); pmc[r][k][1] == pmc[r][k-1][1] && (multi += 1); end
    pmc, multi
end
"The fold mirror of each cell (position len+1-q on a fold line, the wake lines), 0 elsewhere."
function _mirrors(lines, fold, n)
    mir = zeros(Int, n)
    for (l, L) in enumerate(lines); fold[l] || continue; m = length(L); for q in 1:m; mir[L[q]] = L[m+1-q]; end; end
    mir
end
"Thomas solve of tridiag(a, d, c) x = g; returns (x, smallest pivot)."
function _thomas(a, d, c, g)
    m = length(d); cp = zeros(m); x = zeros(m)
    den = d[1]; pmin = den; cp[1] = m > 1 ? c[1]/den : 0.0; x[1] = g[1]/den
    for q in 2:m
        den = d[q] - a[q]*cp[q-1]; pmin = min(pmin, den)
        q < m && (cp[q] = c[q]/den); x[q] = (g[q] - a[q]*x[q-1])/den
    end
    for q in m-1:-1:1; x[q] -= cp[q]*x[q+1]; end
    x, pmin
end

"""
    _thomas_cyclic(a, d, c, beta, alpha, g) -> (x, pmin)

THE CYCLIC LINE SOLVE OF THE PIVOT FIX (plan 28.6; 15 Oct 2026). Solves the cyclic tridiagonal system whose rows are
tridiag(a, d, c) plus the closing entries beta at (1, m) and alpha at (m, 1), by Sherman-Morrison on the opened line
(Numerical Recipes' `cyclic`, gamma = -d[1]): T' = tridiag(a, d', c) with d'[1] = d[1] - gamma, d'[m] = d[m] -
alpha beta/gamma, one factorisation, the solves T' x = g and T' z = (gamma, 0, ..., 0, alpha), then
x -= (x[1] + beta x[m]/gamma)/D z with D = 1 + z[1] + beta z[m]/gamma. With alpha = beta and the cyclic matrix S symmetric,
T' = S + d[1] w w' (w = (1, 0, ..., 0, -beta/d[1])), so T' is positive definite when S is, and D = 1 - d[1] w'T'^-1 w > 0
exactly when S is. `pmin` is the smallest pivot of T', or D when D is not positive (a refusal either way).
The operations are those of the device refresh's `_k_rf_psolve!` (its cyclic branch), in the same order.
"""
function _thomas_cyclic(a, d, c, beta::Float64, alpha::Float64, g)
    m = length(d)
    gam = -d[1]
    b1 = d[1] - gam; bm = d[m] - alpha*beta/gam
    cp = zeros(m); x = zeros(m); z = zeros(m)
    den = b1; pmin = den; cp[1] = c[1]/den; x[1] = g[1]/den; z[1] = gam/den
    for q in 2:m
        bq = q == m ? bm : d[q]
        den = bq - a[q]*cp[q-1]; pmin = min(pmin, den)
        q < m && (cp[q] = c[q]/den)
        x[q] = (g[q] - a[q]*x[q-1])/den
        uq = q == m ? alpha : 0.0
        z[q] = (uq - a[q]*z[q-1])/den
    end
    for q in m-1:-1:1; x[q] -= cp[q]*x[q+1]; z[q] -= cp[q]*z[q+1]; end
    D = 1.0 + z[1] + beta*z[m]/gam
    fact = (x[1] + beta*x[m]/gam)/D
    for q in 1:m; x[q] -= fact*z[q]; end
    D > 0 || (pmin = min(pmin, D))
    x, pmin
end

"Per-level interpolation census (the reference's PStats, the fields the suite and the refusal read); `ncyc` counts the
Schaffer solves taken by the cyclic line solve of the pivot fix (plan 28.6)."
mutable struct MGPStats
    nF::Int; lumpIn::Float64; lumpFF::Float64; pmin::Float64; npair::Int; nmulti::Int; nnopart::Int
    nneg::Int; nvirt::Int; nfold::Int; ncyc::Int
end
MGPStats() = MGPStats(0, 0.0, 0.0, Inf, 0, 0, 0, 0, 0, 0, 0)

"""Schaffer's line-solve interpolation on one level (the reference's schaffer_P). `breakage` = :none, or :half (every
weight halved: a deliberately broken interpolation, the suite's negative example). With `rec` (an MGPRec) the operands
of every floating-point operation are recorded, by index, in the order it is done (review F5: the numeric plan of the
device refresh); the arithmetic is the same with or without it.
THE PIVOT FIX (plan 28.6, 15 Oct 2026; record p0b/fixes_a8/). `cyc` (one Bool per line, default none) marks the cyclic lines
(a doubly periodic box's j-lines). On such a line of at least 3 cells the closing coupling (first cell to last) is an in-line
non-adjacent entry, which the line operator LUMPS into the two end cells' diagonals; at a coarse level, where the line's
couplings to the C lines are weak and partly positive, that lumping can make the tridiagonal matrix indefinite (probe pivot
A: n 64, distortion 0.1, level 6, pivot -4.24e-5, while the line's own cyclic matrix is positive definite). Where, and only
where, the lumped Thomas solve has a non-positive or non-finite pivot on a cyclic line, the weights are solved instead with
the closing entries kept as such (`_thomas_cyclic`, the end diagonals un-lumped as d - v); every other solve is the
reference's, so a hierarchy the old build made is unchanged bit for bit."""
function _schaffer_P(A::HCSR, lines, isC, cidx, lo, pos, pm, fold; breakage::Symbol=:none, rec=nothing,
                     cyc=falses(length(lines)))
    rec === nothing || breakage === :none ||
        throw(ArgumentError("_schaffer_P: a numeric plan is recorded only for breakage = :none"))
    n = A.n; nc = maximum(cidx); st = MGPStats()
    I = Int[]; J = Int[]; V = Float64[]
    for l in eachindex(lines)
        idx = lines[l]; m = length(idx)
        if isC[l]
            for r in idx; push!(I, r); push!(J, cidx[r]); push!(V, 1.0); rec === nothing || push!(rec.src, 0); end
            continue
        end
        st.nF += m
        a = zeros(m); d = zeros(m); c = zeros(m); cpl = Dict{Int,Vector{Tuple{Int,Int,Float64}}}()
        cple = Dict{Int,Vector{Int}}()                                   # (recording) the entry of each cpl item
        for q in 1:m, e in A.ptr[idx[q]]:A.ptr[idx[q]+1]-1     # A symmetric: row idx[q] = column idx[q]
            col = A.col[e]; v = A.val[e]; L2 = lo[col]
            if L2 == l
                dq = pos[col] - q
                if dq == 0;      d[q] += v; rec === nothing || push!(get!(() -> Int[], rec.dl, idx[q]), e)
                elseif dq == -1; a[q] += v; rec === nothing || push!(get!(() -> Int[], rec.al, idx[q]), e)
                elseif dq == 1;  c[q] += v; rec === nothing || push!(get!(() -> Int[], rec.cl, idx[q]), e)
                else;            d[q] += v; st.lumpIn += abs(v); rec === nothing || push!(get!(() -> Int[], rec.dl, idx[q]), e)
                end
            elseif isC[L2]
                push!(get!(() -> Tuple{Int,Int,Float64}[], cpl, L2), (q, col, v))
                rec === nothing || push!(get!(() -> Int[], cple, L2), e)
            else
                d[q] += v; st.lumpFF += abs(v); rec === nothing || push!(get!(() -> Int[], rec.dl, idx[q]), e)
            end
        end
        # the closing entries of a cyclic line (plan 28.6): their positions in A (0 when the line is not cyclic)
        e1m = 0; em1 = 0
        if cyc[l] && m >= 3
            e1m = hpos(A, idx[1], idx[m]); em1 = hpos(A, idx[m], idx[1])
            (e1m > 0 && em1 > 0) || (e1m = 0; em1 = 0)
        end
        for L2 in sort(collect(keys(cpl)))
            st.npair += 1; mL = length(lines[L2])
            pairs = Tuple{Int,Int}[]
            for q in 1:m, (L, p) in pm[idx[q]]; L == L2 && push!(pairs, (q, pos[p])); end
            if isempty(pairs)
                st.nnopart += 1; best = Dict{Int,Tuple{Int,Float64}}()
                for (q, col, v) in cpl[L2]; (haskey(best, q) && best[q][2] >= abs(v)) || (best[q] = (pos[col], abs(v))); end
                pairs = [(q, pv[1]) for (q, pv) in sort(collect(best))]
            end
            S = Set(pairs); keyof = Dict{Tuple{Int,Int},Tuple{Int,Int}}()
            for (q, pp) in pairs
                s = ((q-1, pp-1) in S || (q+1, pp+1) in S) ? 1 : (((q-1, pp+1) in S || (q+1, pp-1) in S) ? -1 : 0)
                keyof[(q, pp)] = (s, pp - s*q)
            end
            conns = sort(unique(collect(values(keyof)))); length(conns) > 1 && (st.nmulti += 1)
            act = [Dict{Int,Int}() for _ in conns]; kidx = Dict(k => i for (i, k) in enumerate(conns))
            for (q, pp) in pairs; act[kidx[keyof[(q, pp)]]][q] = pp; end
            vpos(k, q) = conns[k][1] == 0 ? conns[k][2] : conns[k][1]*q + conns[k][2]
            rhs = [zeros(m) for _ in conns]
            rlist = [Tuple{Int,Int}[] for _ in conns]                   # (recording) (q, entry) per connection
            for (t, (q, col, v)) in enumerate(cpl[L2])
                dbest = typemax(Int); kb = 1
                for k in eachindex(conns); dk = abs(pos[col] - vpos(k, q)); dk < dbest && (dbest = dk; kb = k); end
                rhs[kb][q] += v
                rec === nothing || push!(rlist[kb], (q, cple[L2][t]))
            end
            for k in eachindex(conns)
                w, pmin = _thomas(a, d, c, -rhs[k])
                if !(pmin > 0 && isfinite(pmin)) && e1m > 0           # the pivot fix (plan 28.6): the cyclic line solve
                    v1 = A.val[e1m]; vm = A.val[em1]
                    dc = copy(d); dc[1] = d[1] - v1; dc[m] = d[m] - vm
                    w, pmin = _thomas_cyclic(a, dc, c, v1, vm, -rhs[k]); st.ncyc += 1
                end
                st.pmin = min(st.pmin, pmin)
                breakage === :half && (w .*= 0.5)
                wbase = 0
                if rec !== nothing
                    wbase = rec.nw; rec.nw += m; push!(rec.sline, l); push!(rec.srhs, rlist[k]); push!(rec.scyc, (e1m, em1))
                end
                for q in 1:m
                    wq = w[q]
                    if wq == 0
                        rec === nothing || (rec.zskip += 1)
                        continue
                    end
                    wq < 0 && (st.nneg += 1)
                    if haskey(act[k], q)
                        tp = act[k][q]
                    elseif fold[l] && haskey(act[k], m + 1 - q)
                        tp = act[k][m+1-q]; st.nfold += 1
                    else
                        tp = vpos(k, q); st.nvirt += 1
                    end
                    tpc = clamp(tp, 1, mL)
                    push!(I, idx[q]); push!(J, cidx[lines[L2][tpc]]); push!(V, wq)
                    rec === nothing || push!(rec.src, wbase + q)
                end
            end
        end
    end
    rec === nothing || (append!(rec.I, I); append!(rec.J, J))
    hcsr_coo(I, J, V, n, nc), st
end

function _cf_split(nb, split::Symbol)
    nl = length(nb); isC = falses(nl); decided = falses(nl)
    for l in 1:nl
        decided[l] && continue
        isC[l] = true; decided[l] = true
        for q in nb[l]; decided[q] = true; end
    end
    if split === :fsep
        for l in 1:nl
            isC[l] && continue
            any(q < l && !isC[q] for q in nb[l]) && (isC[l] = true)
        end
    else
        split === :committed || throw(ArgumentError("split must be :fsep (the baseline) or :committed (refused by construction)"))
    end
    isC
end
function _cf_colour_sets(nb, isC)
    sets = Vector{Vector{Int}}()
    for want in (true, false)
        ids = findall(==(want), isC); p = Dict(l => q for (q, l) in enumerate(ids))
        col = _greedy_colours([Set(p[q] for q in nb[l] if haskey(p, q)) for l in ids])
        isempty(col) || append!(sets, [ids[findall(==(cc), col)] for cc in 1:maximum(col)])
    end
    sets
end
function _single_line_tridiagonal(A::HCSR, lines)
    length(lines) == 1 || return false
    p = _positions(lines, A.n)
    all(abs(p[A.col[q]] - p[r]) <= 1 for r in 1:A.n for q in A.ptr[r]:A.ptr[r+1]-1)
end

# ---- the device hierarchy ----------------------------------------------------------------------------------------
"""
    MGLevel

One level on the backend: the operator (CSR `ptr, col, val`), its lines (CSR `lptr, lcell`) with their Thomas factors
(`la, lden, lcp`, aligned with `lcell`), the smoothing colours (`colours`: one device array of line ids per colour, coarse
lines first), the interpolation `P` (n x nc, CSR) and restriction `R` = P' (nc x n, CSR), the work vectors, and on the
coarsest level the coarse solve: `:dense` (`Minv`, Float64 explicit inverse) or `:line` (one tridiagonal line, Thomas).
Host-side records for the suite: `hA` (the operator), `isC`, `pm_hash`, `census`.
"""
struct MGLevel{VF,VI,MF}
    n::Int
    ptr::VI; col::VI; val::VF
    lptr::VI; lcell::VI; la::VF; lden::VF; lcp::VF
    nlines::Int
    colours::Vector{VI}
    Pptr::VI; Pcol::VI; Pval::VF
    Rptr::VI; Rcol::VI; Rval::VF
    x::VF; b::VF; r::VF; z::VF
    coarse::Symbol
    Minv::MF
    hA::HCSR
    isC::BitVector
    pm_hash::UInt64
    census::NamedTuple
end

"""
    MGHierarchy

The levels, the cycle's launch counter `launches` (incremented by every kernel the cycle launches) and the build's
summary (`coarse_kind`, `min_line_pivot`, `min_interp_pivot`).
"""
mutable struct MGHierarchy{B}
    backend::B
    levels::Vector{MGLevel}
    nu::Int
    launches::Int
    coarse_kind::String
    min_line_pivot::Float64
    min_interp_pivot::Float64
    plan::Any                 # the numeric plan of the device refresh (review F5), or nothing (built with plan = false)
    refreshes::Int            # numeric refreshes done (mg_refresh!)
    refresh_launches::Int     # kernels those refreshes launched
    fast::Any                 # W7: the cycle's Float32 and/or workgroup-PCR companion (MGFast), or nothing (the cycle as built)
end

Base.length(h::MGHierarchy) = length(h.levels)

"""
    build_schaffer_mg(sys::FaceSystem, mesh::PBMesh; nu=1, split=:fsep, backend=CPU(), maxlev=40, breakage=:none)
    build_schaffer_mg(A::HCSR, jlines, nfold; ...)

The `Sch V11 j` hierarchy of the operator: the mesh's j-lines (the first `nfold` lines are the fold lines, the C-mesh's wake
lines: `mesh.NIW`; none on a box), the `:fsep` split, Schaffer's line-solve interpolation with the cut fold and the strip
maps kept on every level, Galerkin coarse operators, down to one line, solved by Thomas when it is tridiagonal, else by an
explicit Float64 inverse when it has at most MG_COARSE_INVERSE_MAX cells. REFUSES (ArgumentError) any non-positive or
non-finite line-factor pivot on any level, any non-positive interpolation pivot, and a coarsest line that has neither an
admissible line solve nor an admissible inverse. Only `nu = 1` and j-line smoothing are ported (the P0a baseline).
With `plan = true` (the face-system form only) the build also records its numeric plan (the symbolic phase of review F5,
see "THE NUMERIC REFRESH" below), so that `mg_refresh!` can recompute every value on the backend from a new face system
with the same sparsity; the hierarchy's values are the same with or without it.
"""
function build_schaffer_mg(sys::FaceSystem, m::PBMesh; plan::Bool=false, kw...)
    A = facesys_hcsr(sys, m)
    ptr = _host(m.jl_ptr); cell = _host(m.jl_cell)
    lines = [Int.(cell[ptr[l]:ptr[l+1]-1]) for l in 1:length(ptr)-1]
    build_schaffer_mg(A, lines, m.family === :cmesh ? m.NIW : 0; gather=plan ? _facesys_gather_plan(m) : nothing,
                      cyclic=_host(m.jl_cyclic) .!= 0, kw...)
end

function build_schaffer_mg(A::HCSR, jlines::Vector{Vector{Int}}, nfold::Integer; nu::Integer=1, split::Symbol=:fsep,
                           backend=CPU(), maxlev::Integer=40, breakage::Symbol=:none, gather=nothing, cyclic=nothing)
    plan = gather !== nothing
    cyc = cyclic === nothing ? falses(length(jlines)) : BitVector(cyclic)
    length(cyc) == length(jlines) || throw(ArgumentError("build_schaffer_mg: one cyclic flag per line"))
    precs = Any[]
    nu == 1 || throw(ArgumentError("only V(1,1) is ported (the P0a baseline Sch V11 j); got nu = $nu"))
    split in (:fsep, :committed) || throw(ArgumentError("split must be :fsep or :committed"))
    sort(vcat(jlines...)) == collect(1:A.n) || throw(ArgumentError("the j-lines must partition the cells"))
    levels = MGLevel[]
    Al = A; Ll = jlines; fold = [l <= nfold for l in eachindex(Ll)]
    pm, _ = _partners_fine(Al, _lineof(Ll, Al.n))
    ckind = ""; minline = Inf; mininterp = Inf
    for lev in 1:maxlev
        n = Al.n
        facs = [_linefac(Al, L) for L in Ll]
        # BAFFLES (24 Oct 2026): a cut that splits the domain (a full-length baffle) leaves components whose lines coarsen
        # separately; once every remaining line is coupled to no other there is nothing to coarsen, and that level is the
        # coarsest (before baffles every hierarchy reached one line first: this level was then a refused interpolation pivot,
        # so no hierarchy that built before changes). The line graph is integer work, the same as below.
        nb, lo = _line_graph(Al, Ll)
        isolated = length(Ll) > 1 && all(isempty, nb)
        coarsest = length(Ll) <= 1 || lev == maxlev || isolated
        # the line factors the cycle uses: every smoothing level's, and the coarsest's only when it is solved as one line
        # (a non-tridiagonal coarsest line -- the fold -- is solved by its explicit inverse, its tridiagonal part unused)
        if !coarsest || _single_line_tridiagonal(Al, Ll)
            for (l, (_, den, _)) in enumerate(facs)
                pmn = minimum(den)
                (pmn > 0 && all(isfinite, den)) ||
                    throw(ArgumentError(@sprintf("build_schaffer_mg: level %d line %d has a non-positive line-factor pivot %.3e (refused: the cycle would be indefinite)", lev, l, pmn)))
                minline = min(minline, pmn)
            end
        end
        if coarsest
            if _single_line_tridiagonal(Al, Ll)
                ck = :line; Minv = zeros(0, 0); ckind = "one line, Thomas ($n cells)"
            elseif n <= MG_COARSE_INVERSE_MAX
                Dn = hdense(Al)
                isposdef(Symmetric(Dn)) ||
                    throw(ArgumentError("build_schaffer_mg: the coarsest operator ($n cells) is not positive definite (refused)"))
                ck = :dense; Minv = inv(Dn); ckind = "dense Float64 inverse ($n cells, $(length(Ll)) line$(length(Ll) > 1 ? "s" : ""))"
                all(isfinite, Minv) || throw(ArgumentError("build_schaffer_mg: the coarsest inverse is not finite"))
            else
                throw(ArgumentError("build_schaffer_mg: the coarsest level ($n cells, $(length(Ll)) lines) is neither one tridiagonal line nor at most $(MG_COARSE_INVERSE_MAX) cells (no admissible coarse solve, plan 5.2)"))
            end
            push!(levels, _mglevel(backend, Al, Ll, facs, ck === :line ? [[1]] : Vector{Vector{Int}}(), nothing, ck, Minv,
                                   falses(length(Ll)), pm, (n=n, lines=length(Ll), st=nothing)))
            plan && push!(precs, _plan_lines(Al, Ll, nothing; check=(ck === :line)))
            break
        end
        isC = _cf_split(nb, split); pos = _positions(Ll, n)
        cidx = zeros(Int, n); nc = 0; clines = Vector{Vector{Int}}(); lc = zeros(Int, length(Ll))
        for l in eachindex(Ll)
            isC[l] || continue
            cl = Int[]; for r in Ll[l]; nc += 1; cidx[r] = nc; push!(cl, nc); end
            push!(clines, cl); lc[l] = length(clines)
        end
        recP = plan ? MGPRec() : nothing
        P, st = _schaffer_P(Al, Ll, isC, cidx, lo, pos, pm, fold; breakage=breakage, rec=recP, cyc=cyc)
        (st.pmin > 0 && isfinite(st.pmin)) ||
            throw(ArgumentError(@sprintf("build_schaffer_mg: level %d has a non-positive interpolation pivot %.3e (refused)", lev, st.pmin)))
        mininterp = min(mininterp, st.pmin)
        R = htranspose(P)
        T = hmul(R, Al); Craw = hmul(T, P); Ac = hsym_drop(Craw)          # = hsym_drop(hmul(hmul(R, Al), P)), named
        push!(levels, _mglevel(backend, Al, Ll, facs, _cf_colour_sets(nb, isC), P, :none, zeros(0, 0), isC, pm,
                               (n=n, lines=length(Ll), st=st)))
        plan && push!(precs, _plan_level(Al, Ll, P, R, T, Craw, Ac, recP))
        pm, _ = _partners_coarse(pm, lo, isC, lc, cidx, nc, _mirrors(Ll, fold, n))
        Al = Ac; Ll = clines; fold = fold[isC]; cyc = cyc[isC]
    end
    h = MGHierarchy(backend, levels, Int(nu), 0, ckind, minline, mininterp, nothing, 0, 0, nothing)
    plan && (h.plan = _mg_device_plan(backend, h, precs, gather))
    h
end

function _mglevel(backend, A::HCSR, lines, facs, sets, P, ck, Minv, isC, pm, census)
    d(v) = _dev(backend, v)
    lptr = Int32[1]; lcell = Int32[]; la = Float64[]; lden = Float64[]; lcp = Float64[]
    for (L, (a, den, cp)) in zip(lines, facs)
        append!(lcell, Int32.(L)); append!(la, a); append!(lden, den); append!(lcp, cp); push!(lptr, Int32(length(lcell) + 1))
    end
    cols = [d(Int32.(s)) for s in sets]
    if P === nothing
        Pp = Int32[1]; Pc = Int32[]; Pv = Float64[]; Rp = Int32[1]; Rc = Int32[]; Rv = Float64[]
    else
        R = htranspose(P)
        Pp = Int32.(P.ptr); Pc = Int32.(P.col); Pv = P.val; Rp = Int32.(R.ptr); Rc = Int32.(R.col); Rv = R.val
    end
    n = A.n
    MGLevel(n, d(Int32.(A.ptr)), d(Int32.(A.col)), d(A.val), d(lptr), d(lcell), d(la), d(lden), d(lcp), length(lines),
            cols, d(Pp), d(Pc), d(Pv), d(Rp), d(Rc), d(Rv),
            _zeros(backend, Float64, n), _zeros(backend, Float64, n), _zeros(backend, Float64, n), _zeros(backend, Float64, n),
            ck, d(Minv), A, BitVector(isC), hash(pm), census)
end

# ---- the cycle's kernels -----------------------------------------------------------------------------------------
"r = b - A x (row gather in increasing column order); with zero_x = 1, x = 0 and r = b (a zero initial guess)."
@kernel function _k_csr_resid!(r, b, x, ptr, col, val, zero_x::Int32)
    i = @index(Global)
    @inbounds begin
        if zero_x == Int32(1)
            x[i] = 0.0; r[i] = b[i]
        else
            acc = zero(eltype(val))
            for q in ptr[i]:ptr[i+1]-Int32(1); acc += val[q]*x[col[q]]; end
            r[i] = b[i] - acc
        end
    end
end
"y = M x (CSR)."
@kernel function _k_csr_mul!(y, x, ptr, col, val)
    i = @index(Global)
    @inbounds begin
        acc = zero(eltype(val))
        for q in ptr[i]:ptr[i+1]-Int32(1); acc += val[q]*x[col[q]]; end
        y[i] = acc
    end
end
"x += M xc (CSR; the sum formed first, then added: the reference's x .+= P*xc)."
@kernel function _k_csr_muladd!(x, xc, ptr, col, val)
    i = @index(Global)
    @inbounds begin
        acc = zero(eltype(val))
        for q in ptr[i]:ptr[i+1]-Int32(1); acc += val[q]*xc[col[q]]; end
        x[i] += acc
    end
end
"y = M x, M dense (the coarsest inverse)."
@kernel function _k_dense_mv!(y, M, x, n::Int32)
    i = @index(Global)
    @inbounds begin
        acc = 0.0
        for j in Int32(1):n; acc += M[i, j]*x[j]; end
        y[i] = acc
    end
end
"""One work item per line of a colour: z = T^-1 r along the line (Thomas, the reference's linesolve!), then x += z on the
line's cells (assign = 1: x = z)."""
@kernel function _k_line_solve!(x, z, r, lines, lptr, lcell, la, lden,
                                lcp, assign::Int32)
    t = @index(Global)
    @inbounds begin
        l = lines[t]; a0 = lptr[l]; a1 = lptr[l+1] - Int32(1)
        c = lcell[a0]; z[c] = r[c]/lden[a0]
        for q in a0+Int32(1):a1
            c = lcell[q]; z[c] = (r[c] - la[q]*z[lcell[q-Int32(1)]])/lden[q]
        end
        for q in a1-Int32(1):Int32(-1):a0
            c = lcell[q]; z[c] -= lcp[q]*z[lcell[q+Int32(1)]]
        end
        if assign == Int32(1)
            for q in a0:a1; c = lcell[q]; x[c] = z[c]; end
        else
            for q in a0:a1; c = lcell[q]; x[c] += z[c]; end
        end
    end
end

function _mg_launch!(h::MGHierarchy, kern, n, args...)
    _launch!(kern, h.backend, n, args...); h.launches += 1
    nothing
end

"Smooth level `L` with the colour sequence in `order` (x, b given; the first residual zeroes x when `zero_first`)."
function _mg_smooth!(h, L::MGLevel, x, b, order, zero_first::Bool)
    first = zero_first
    for ci in order
        set = L.colours[ci]
        _mg_launch!(h, _k_csr_resid!, L.n, L.r, b, x, L.ptr, L.col, L.val, Int32(first ? 1 : 0))
        _mg_launch!(h, _k_line_solve!, length(set), x, L.z, L.r, set, L.lptr, L.lcell, L.la, L.lden, L.lcp, Int32(0))
        first = false
    end
    first
end

function _mg_cycle!(h::MGHierarchy, k::Int, x, b)
    L = h.levels[k]
    if L.coarse === :dense
        _mg_launch!(h, _k_dense_mv!, L.n, x, L.Minv, b, Int32(L.n))
        return nothing
    elseif L.coarse === :line
        _mg_launch!(h, _k_line_solve!, 1, x, L.z, b, L.colours[1], L.lptr, L.lcell, L.la, L.lden, L.lcp, Int32(1))
        return nothing
    end
    nc = length(L.colours); pre = 1:nc; post = nc:-1:1
    zfirst = true
    for _ in 1:h.nu; zfirst = _mg_smooth!(h, L, x, b, pre, zfirst); end
    Lc = h.levels[k+1]
    _mg_launch!(h, _k_csr_resid!, L.n, L.r, b, x, L.ptr, L.col, L.val, Int32(zfirst ? 1 : 0))
    _mg_launch!(h, _k_csr_mul!, Lc.n, Lc.b, L.r, L.Rptr, L.Rcol, L.Rval)
    _mg_cycle!(h, k + 1, Lc.x, Lc.b)
    _mg_launch!(h, _k_csr_muladd!, L.n, x, Lc.x, L.Pptr, L.Pcol, L.Pval)
    for _ in 1:h.nu; _mg_smooth!(h, L, x, b, post, false); end
    nothing
end

"""
    mg_vcycle!(z, h::MGHierarchy, r) -> z

One V(1,1) cycle from a zero initial guess: z = M r (z and r on the hierarchy's backend; r is not modified).
"""
function mg_vcycle!(z, h::MGHierarchy, r)
    length(z) == length(r) == h.levels[1].n || throw(DimensionMismatch("mg_vcycle!: vectors must have $(h.levels[1].n) entries"))
    if h.fast === nothing
        _mg_cycle!(h, 1, z, r)
    else                                   # W7 (opt-in): the cycle in the companion's precision and line solver
        F = h.fast; L1 = F.levels[1]
        pb_convert!(L1.b, r; backend=h.backend)
        _mg_cycle_fast!(h, F, 1, L1.x, L1.b)
        pb_convert!(z, L1.x; backend=h.backend)
    end
    pb_sync(h.backend)
    z
end

# =====================================================================================================================
# W7 (speed; plan Amendment 13, section 33; 22 Oct 2026): THE CYCLE'S COMPANION, opt-in (MGHierarchy.fast; nothing by
# default, and then nothing here runs). It holds, per level, the operator, P, R, the line factors and the work vectors in the
# cycle's element type T (Float32 under prec_precision = :f32; Float64 for the PCR alone, the arrays then shared with the
# level), the line tridiagonal's diagonal and upper coupling (for the workgroup PCR: read from the operator's own entries,
# no reconstruction), and runs the same V(1,1) cycle -- the same colours, the same order, the same transfers -- with the
# line solves by `solver`. With T = Float64 and solver = :thomas it is the cycle above operation for operation (the suite
# checks it bit for bit). The coarsest dense inverse stays Float64 (its product accumulates in Float64 and is rounded to T on
# store). The values are refreshed from the hierarchy by `mg_fast_update!` after every build and refresh (rounded copies;
# the Float64 hierarchy, its refresh and its refusals are unchanged).
# CORRECTION (22 Oct 2026, the W7 review's finding 1): with T = Float32 (i) the PCR eliminates in Float64, on the level's
# Float64 line tridiagonal `fam64` (ta = the level's la, td and tc read from its Float64 operator, den/cp its Float64
# factors; the arrays shared with the level), never in Float32 (`line_solve_set!`); (ii) every residual is the excess form
# r_i = b_i - E_i x_i - sum_q val_q (x_col(q) - x_i), E the level operator's row sums formed in Float64 and rounded
# (`_k_csr_resid_ex!`): the pressure levels' row margins (2e-8 and below on A2 and CH) lie below Float32's resolution, so the
# plain Float32 product cancelled and lost the near-null space (the cycle differed from Float64 by 1.2e-2 on A2).
# =====================================================================================================================
struct MGFastLevel{VF,VI}
    val::VF; Pval::VF; Rval::VF
    fam::LineFamily               # the level's lines (lptr, lcell, colours shared) with ta = la, td, tc, den = lden, cp = lcp in T
    lnext::VI                     # per line position: the next position on the same line, or 0 (host-built once)
    x::VF; b::VF; r::VF; z::VF
    fam64::Any                    # T = Float32 with the PCR: the level's Float64 line tridiagonal (LineFamily), else nothing
    ex::Any                       # T = Float32: the operator's row sums (Float64-formed, rounded; length n), else nothing
end
mutable struct MGFast
    T::DataType
    solver::Symbol
    levels::Vector{MGFastLevel}
    syncs::Int
end

"The line tridiagonal's diagonal and upper coupling of every line position, from the level's CSR operator (Float64)."
@kernel function _k_mg_line_dc!(td, tc, lcell, lnext, ptr, col, val)
    q = @index(Global)
    @inbounds begin
        c = lcell[q]; nq = lnext[q]; cn = nq > 0 ? lcell[nq] : Int32(0)
        d = 0.0; u = 0.0
        for e in ptr[c]:ptr[c+1]-Int32(1)
            j = col[e]
            j == c && (d = val[e])
            j == cn && (u = val[e])
        end
        td[q] = convert(eltype(td), d); tc[q] = convert(eltype(tc), u)
    end
end

function _mgfast_level(be, L::MGLevel, ::Type{T}, solver::Symbol=:thomas) where {T}
    z(n) = KernelAbstractions.zeros(be, T, n)
    lp = _host(L.lptr); nl = length(lp) - 1; npos = Int(lp[end] - 1)
    nxt = zeros(Int32, npos)
    for l in 1:nl, q in lp[l]:lp[l+1]-2; nxt[q] = Int32(q + 1); end
    shared = T === Float64
    val = shared ? L.val : z(length(L.val)); Pv = shared ? L.Pval : z(length(L.Pval)); Rv = shared ? L.Rval : z(length(L.Rval))
    la = shared ? L.la : z(npos); lden = shared ? L.lden : z(npos); lcp = shared ? L.lcp : z(npos)
    cols = isempty(L.colours) ? typeof(L.lptr)[] : L.colours          # (a dense coarsest level has no colours)
    maxlen = maximum(diff(lp); init=0)
    # (a Float32 level's own td/tc are not used: the PCR eliminates on fam64, Float32 Thomas needs ta, den and cp only)
    fam = LineFamily(:mg, nl, maxlen, L.lptr, L.lcell, L.lptr, L.lcell, cols, la, z(shared ? npos : 0), z(shared ? npos : 0),
                     lden, lcp)
    x, b, r, zz = shared ? (L.x, L.b, L.r, L.z) : (z(L.n), z(L.n), z(L.n), z(L.n))
    z64(n) = KernelAbstractions.zeros(be, Float64, n)
    fam64 = (!shared && solver === :pcr) ?
        LineFamily(:mg, nl, maxlen, L.lptr, L.lcell, L.lptr, L.lcell, cols, L.la, z64(npos), z64(npos), L.lden, L.lcp) : nothing
    ex = shared ? nothing : z(L.n)
    MGFastLevel(val, Pv, Rv, fam, _dev(be, nxt), x, b, r, zz, fam64, ex)
end

"""The excess residual of a CSR level (W7's correction; the section header): r_i = b_i - E_i x_i - sum_q val_q (x_col(q)
- x_i) (the diagonal entry contributes zero), E the row sums; with zero_x = 1, x = 0 and r = b."""
@kernel function _k_csr_resid_ex!(r, b, x, ptr, col, val, ex, zero_x::Int32)
    i = @index(Global)
    @inbounds begin
        if zero_x == Int32(1)
            x[i] = zero(eltype(x)); r[i] = b[i]
        else
            xi = x[i]; acc = ex[i]*xi
            for q in ptr[i]:ptr[i+1]-Int32(1); acc += val[q]*(x[col[q]] - xi); end
            r[i] = b[i] - acc
        end
    end
end
"The row sums of a CSR operator (Float64 values), accumulated in Float64 and rounded to ex's element type."
@kernel function _k_csr_rowsum!(ex, ptr, val)
    i = @index(Global)
    @inbounds begin
        acc = 0.0
        for q in ptr[i]:ptr[i+1]-Int32(1); acc += val[q]; end
        ex[i] = convert(eltype(ex), acc)
    end
end
"The level's residual into FL.r: the excess form when the companion carries row sums (T = Float32), else the plain one."
function _mg_resid_fast!(h, L::MGLevel, FL::MGFastLevel, b, x, zero_x::Bool)
    if FL.ex === nothing
        _mg_launch!(h, _k_csr_resid!, L.n, FL.r, b, x, L.ptr, L.col, FL.val, Int32(zero_x ? 1 : 0))
    else
        _mg_launch!(h, _k_csr_resid_ex!, L.n, FL.r, b, x, L.ptr, L.col, FL.val, FL.ex, Int32(zero_x ? 1 : 0))
    end
    nothing
end

"""
    mg_fast_update!(h, T, solver) -> h

Make (when absent or of another kind) and load the hierarchy's W7 companion: T = Float32 or Float64, solver = :thomas or
:pcr; with (Float64, :thomas) the companion is removed (h.fast = nothing: the cycle as built). Called after every build and
refresh of the hierarchy (mg_update!).
"""
function mg_fast_update!(h::MGHierarchy, ::Type{T}, solver::Symbol) where {T}
    be = h.backend
    if T === Float64 && solver === :thomas
        h.fast = nothing; return h
    end
    F = h.fast
    if F === nothing || F.T !== T || F.solver !== solver
        F = MGFast(T, solver, MGFastLevel[_mgfast_level(be, L, T, solver) for L in h.levels], 0)
        h.fast = F
    end
    for (L, FL) in zip(h.levels, F.levels)
        if T !== Float64
            pb_convert!(FL.val, L.val; backend=be); pb_convert!(FL.Pval, L.Pval; backend=be); pb_convert!(FL.Rval, L.Rval; backend=be)
            pb_convert!(FL.fam.ta, L.la; backend=be); pb_convert!(FL.fam.den, L.lden; backend=be); pb_convert!(FL.fam.cp, L.lcp; backend=be)
            _launch!(_k_csr_rowsum!, be, L.n, FL.ex, L.ptr, L.val)
        end
        if solver === :pcr                 # the PCR's Float64 tridiagonal: the level's own (T = Float64) or fam64 (T = Float32)
            pf = FL.fam64 === nothing ? FL.fam : FL.fam64
            _launch!(_k_mg_line_dc!, be, length(FL.lnext), pf.td, pf.tc, L.lcell, FL.lnext, L.ptr, L.col, L.val)
        end
    end
    F.syncs += 1
    pb_sync(be)
    h
end

function _mg_smooth_fast!(h, L::MGLevel, FL::MGFastLevel, solver, x, b, order, zero_first::Bool)
    first = zero_first
    for ci in order
        set = L.colours[ci]
        _mg_resid_fast!(h, L, FL, b, x, first)
        line_solve_set!(x, FL.z, FL.r, set, FL.fam, Int32(0); backend=h.backend, solver, fam64=FL.fam64); h.launches += 1
        first = false
    end
    first
end

function _mg_cycle_fast!(h::MGHierarchy, F::MGFast, k::Int, x, b)
    L = h.levels[k]; FL = F.levels[k]
    if L.coarse === :dense
        _mg_launch!(h, _k_dense_mv!, L.n, x, L.Minv, b, Int32(L.n))
        return nothing
    elseif L.coarse === :line
        line_solve_set!(x, FL.z, b, L.colours[1], FL.fam, Int32(1); backend=h.backend, solver=F.solver, fam64=FL.fam64)
        h.launches += 1
        return nothing
    end
    nc = length(L.colours); pre = 1:nc; post = nc:-1:1
    zfirst = true
    for _ in 1:h.nu; zfirst = _mg_smooth_fast!(h, L, FL, F.solver, x, b, pre, zfirst); end
    Lc = h.levels[k+1]; FLc = F.levels[k+1]
    _mg_resid_fast!(h, L, FL, b, x, zfirst)
    _mg_launch!(h, _k_csr_mul!, Lc.n, FLc.b, FL.r, L.Rptr, L.Rcol, FL.Rval)
    _mg_cycle_fast!(h, F, k + 1, FLc.x, FLc.b)
    _mg_launch!(h, _k_csr_muladd!, L.n, x, FLc.x, L.Pptr, L.Pcol, FL.Pval)
    for _ in 1:h.nu; _mg_smooth_fast!(h, L, FL, F.solver, x, b, post, false); end
    nothing
end

"The P0a KERNEL_RULE count of one cycle (2 per colour per sweep, 3 per non-coarsest level, 1 coarsest)."
function mg_kernel_rule(h::MGHierarchy)
    s = 0
    for L in h.levels
        s += L.coarse === :none ? 2*h.nu*2*length(L.colours) + 3 : 1
    end
    s
end

# =====================================================================================================================
# THE NUMERIC REFRESH (review F5, 14 Oct 2026; plan 24.1, option O1). The build above is split in two:
#   * THE SYMBOLIC PHASE: `build_schaffer_mg(sys, mesh; plan = true)` runs the host build once and records, beside it, the
#     index lists of every floating-point operation of its numeric part, in the build's own order (the level-1 operator's
#     gather from the face system, each level's line factors and the lumped F-line diagonals, every Thomas solve of
#     Schaffer's weights with its right side, P's duplicate sums, R = P', the two Galerkin products (R A) P accumulated as
#     `hmul` accumulates, the symmetrisation). Everything the build decides from the mesh or from the first operator's
#     values -- the lines, the split, the colours, the partner maps, every pattern -- is frozen with it.
#   * THE NUMERIC PHASE: `mg_refresh!(h, sys)` recomputes every value of the hierarchy from a new face system with the same
#     sparsity, as KernelAbstractions kernels on the hierarchy's backend (no host loop over cells): one gather kernel for
#     level 1; per level one kernel over lines (factors and lumped diagonals), one over Schaffer's solves (one work item per
#     F line and connection, Thomas along it), one over P's entries (P and R together), two sparse products, one
#     symmetrisation; on a dense coarsest level a densify kernel and one Gauss-Jordan sweep per pivot; one segment-minimum
#     kernel for the build's refusals, read back once.
# EXACTNESS. On the CPU backend every operator, line factor, P and R value equals what a host build of the new operator
# gives, bit for bit, whenever that build would take the same symbolic decisions (the same partner maps, no weight or entry
# that is exactly zero in one and not the other): the same operations in the same order (Julia does not contract a*b + c,
# and no kernel here uses @fastmath). The coarsest inverse is Gauss-Jordan here and LAPACK in the build, equal to round-off.
# On CUDA the compiler may contract a*b + c into an FMA, so device parity is to round-off (stage P3).
# REFUSALS. A non-positive or non-finite line-factor pivot (on the levels the build checks), interpolation pivot or
# Gauss-Jordan pivot of the coarsest operator throws an ArgumentError, as the build does.
# =====================================================================================================================

"The recording of one level's `_schaffer_P` (plan = true)."
mutable struct MGPRec
    dl::Dict{Int,Vector{Int}}     # per F-line cell: the entries summed into the lumped diagonal d, in order
    al::Dict{Int,Vector{Int}}     # per F-line cell: the entry of a (the previous cell on its line)
    cl::Dict{Int,Vector{Int}}     # per F-line cell: the entry of c (the next cell on its line)
    sline::Vector{Int}            # per Thomas solve: its F line
    srhs::Vector{Vector{Tuple{Int,Int}}}   # per solve: (position, entry) added into its right side, in order
    scyc::Vector{Tuple{Int,Int}}  # per solve: the closing entries (1, m) and (m, 1) of a cyclic line, or (0, 0) (plan 28.6)
    nw::Int                       # weights recorded so far (the solves' lengths summed)
    src::Vector{Int}              # per COO entry of P: 0 (the C cell's 1.0) or its weight's index
    I::Vector{Int}; J::Vector{Int}
    zskip::Int                    # weights exactly zero, skipped by the build (outside the plan's pattern)
end
MGPRec() = MGPRec(Dict{Int,Vector{Int}}(), Dict{Int,Vector{Int}}(), Dict{Int,Vector{Int}}(), Int[],
                  Vector{Tuple{Int,Int}}[], Tuple{Int,Int}[], 0, Int[], Int[], Int[], 0)

"The CSR position of (r, c) in A, 0 when it is not stored."
function hpos(A::HCSR, r::Int, c::Int)
    lo = A.ptr[r]; hi = A.ptr[r+1] - 1
    while lo <= hi
        mid = (lo + hi) >>> 1; cm = A.col[mid]
        cm == c && return mid
        cm < c ? (lo = mid + 1) : (hi = mid - 1)
    end
    0
end

"""The plan of `hcsr_coo(I, J, V, n, m)`: its pattern (`ptr`, `col`) and, per entry, the COO indices summed into it in
`hcsr_coo`'s order (`sptr`, `sidx`: the first assigned, the others added in turn)."""
function hcsr_coo_plan(I::AbstractVector{<:Integer}, J::AbstractVector{<:Integer}, n::Int, m::Int)
    nz = length(I)
    cnt = zeros(Int, n + 1)
    for t in 1:nz; cnt[I[t]+1] += 1; end
    ptr0 = cumsum(vcat(1, cnt[2:end])); fill = copy(ptr0)
    cs = Vector{Int}(undef, nz); ts = Vector{Int}(undef, nz)
    for t in 1:nz; r = I[t]; cs[fill[r]] = J[t]; ts[fill[r]] = t; fill[r] += 1; end
    ptr = ones(Int, n + 1); col = Int[]; sstart = Int[]; sidx = Int[]
    for r in 1:n
        a = ptr0[r]; b = ptr0[r+1] - 1
        if a <= b
            perm = sortperm(view(cs, a:b); alg=Base.Sort.MergeSort); last = 0
            for q in perm
                c = cs[a+q-1]; t = ts[a+q-1]
                if c == last
                    push!(sidx, t)
                else
                    push!(col, c); push!(sstart, length(sidx) + 1); push!(sidx, t); last = c
                end
            end
        end
        ptr[r+1] = length(col) + 1
    end
    ptr, col, vcat(sstart, length(sidx) + 1), sidx
end

"Per entry q of A, its position in `htranspose(A)` (the transpose's values are copies)."
function htranspose_plan(A::HCSR)
    cnt = zeros(Int, A.m + 1)
    for c in A.col; cnt[c+1] += 1; end
    fill = cumsum(vcat(1, cnt[2:end])); pos = Vector{Int}(undef, length(A.col))
    for r in 1:A.n, q in A.ptr[r]:A.ptr[r+1]-1
        c = A.col[q]; pos[q] = fill[c]; fill[c] += 1
    end
    pos
end

"""The plan of `hmul(A, B)`: its pattern and, per entry, the pairs (index into A.val, index into B.val) in `hmul`'s
accumulation order (`pptr`, `pa`, `pb`): the first product assigned, the others added in turn."""
function hmul_plan(A::HCSR, B::HCSR)
    A.m == B.n || throw(DimensionMismatch("hmul_plan"))
    slot = zeros(Int, B.m); cols = Int[]; la = Vector{Int}[]; lb = Vector{Int}[]
    ptr = ones(Int, A.n + 1); col = Int[]; pptr = [1]; pa = Int[]; pb = Int[]
    for i in 1:A.n
        empty!(cols); nl = 0
        for q in A.ptr[i]:A.ptr[i+1]-1
            k = A.col[q]
            for e in B.ptr[k]:B.ptr[k+1]-1
                j = B.col[e]
                if slot[j] == 0
                    nl += 1; slot[j] = nl; push!(cols, j)
                    nl > length(la) && (push!(la, Int[]); push!(lb, Int[]))
                    empty!(la[nl]); empty!(lb[nl])
                end
                push!(la[slot[j]], q); push!(lb[slot[j]], e)
            end
        end
        for s in sortperm(cols)
            j = cols[s]; push!(col, j); append!(pa, la[s]); append!(pb, lb[s]); push!(pptr, length(pa) + 1); slot[j] = 0
        end
        ptr[i+1] = length(col) + 1
    end
    ptr, col, pptr, pa, pb
end

"""The plan of `hsym_drop(A)` on A's current values: per entry of the result, the indices into A.val of A[r, c] and of
A[c, r] (0 when not stored), and the number of entries dropped as exactly zero."""
function hsym_plan(A::HCSR)
    T = htranspose(A); tp = htranspose_plan(A); tinv = similar(tp); for q in eachindex(tp); tinv[tp[q]] = q; end
    ptr = ones(Int, A.n + 1); col = Int[]; i1 = Int[]; i2 = Int[]; drops = 0
    for r in 1:A.n
        a = A.ptr[r]; ae = A.ptr[r+1] - 1; t = T.ptr[r]; te = T.ptr[r+1] - 1
        while a <= ae || t <= te
            ca = a <= ae ? A.col[a] : typemax(Int); ct = t <= te ? T.col[t] : typemax(Int)
            if ca == ct
                v = (A.val[a] + T.val[t])/2; c = ca; j1 = a; j2 = tinv[t]; a += 1; t += 1
            elseif ca < ct
                v = (A.val[a] + 0.0)/2; c = ca; j1 = a; j2 = 0; a += 1
            else
                v = (0.0 + T.val[t])/2; c = ct; j1 = 0; j2 = tinv[t]; t += 1
            end
            if v == 0
                drops += 1
            else
                push!(col, c); push!(i1, j1); push!(i2, j2)
            end
        end
        ptr[r+1] = length(col) + 1
    end
    ptr, col, i1, i2, drops
end

"""The plan of `facesys_hcsr(sys, mesh)`: its pattern and, per entry, the face-system values summed into it in order, as
codes (1..nf upper[f], nf+1..2nf lower[f - nf], 2nf+1.. diag[c - 2nf])."""
function _facesys_gather_plan(m::PBMesh)
    own = _host(m.own); nei = _host(m.nei); nf = m.nf; nc = m.nc
    I = Vector{Int}(undef, 2nf + nc); J = similar(I)
    for q in 1:nf; I[2q-1] = own[q]; J[2q-1] = nei[q]; I[2q] = nei[q]; J[2q] = own[q]; end
    for c in 1:nc; I[2nf+c] = c; J[2nf+c] = c; end
    ptr, col, sptr, sidx = hcsr_coo_plan(I, J, nc, nc)
    code = [t <= 2nf ? (isodd(t) ? (t + 1) ÷ 2 : nf + t ÷ 2) : t for t in sidx]
    (ptr=ptr, col=col, sptr=sptr, code=code, nc=nc, nf=nf)
end

"The line-factor plan of a level (positions in the level's lcell order: ib, ia, ic) and its lumped-diagonal lists."
function _plan_lines(A::HCSR, lines, rec; check::Bool=true)
    ncl = sum(length, lines); ib = zeros(Int, ncl); ia = zeros(Int, ncl); ic = zeros(Int, ncl)
    dptr = ones(Int, ncl + 1); dsrc = Int[]; g = 0
    for L in lines
        m = length(L)
        for q in 1:m
            g += 1; r = L[q]
            ib[g] = hpos(A, r, r); ib[g] > 0 || throw(ArgumentError("_plan_lines: cell $r has no stored diagonal"))
            q > 1 && (ia[g] = hpos(A, r, L[q-1])); q < m && (ic[g] = hpos(A, r, L[q+1]))
            if rec !== nothing && haskey(rec.dl, r)
                append!(dsrc, rec.dl[r])
                # Schaffer's a and c are the line factor's entries (the refresh reads them through ia and ic)
                get(rec.al, r, Int[]) == (ia[g] > 0 ? [ia[g]] : Int[]) ||
                    error("_plan_lines: Schaffer's a of cell $r is not the line factor's")
                get(rec.cl, r, Int[]) == (ic[g] > 0 ? [ic[g]] : Int[]) ||
                    error("_plan_lines: Schaffer's c of cell $r is not the line factor's")
            end
            dptr[g+1] = length(dsrc) + 1
        end
    end
    (ib=ib, ia=ia, ic=ic, dptr=dptr, dsrc=dsrc, check=check, nlines=length(lines))
end

"One non-coarsest level's plan: its lines, Schaffer's solves, P (and R), the Galerkin products and the symmetrisation."
function _plan_level(A::HCSR, lines, P::HCSR, R::HCSR, T::HCSR, Craw::HCSR, Ac::HCSR, rec::MGPRec)
    lf = _plan_lines(A, lines, rec; check=true)
    ns = length(rec.sline); lens = [length(lines[l]) for l in rec.sline]
    wofs = Vector{Int}(undef, ns); o = 1; for s in 1:ns; wofs[s] = o; o += lens[s]; end
    o - 1 == rec.nw || error("_plan_level: weight offsets")
    gptr = ones(Int, ns + 1); gq = Int[]; ge = Int[]
    for s in 1:ns
        for (q, e) in rec.srhs[s]; push!(gq, q); push!(ge, e); end
        gptr[s+1] = length(gq) + 1
    end
    pptr, pcol, sptr, sidx = hcsr_coo_plan(rec.I, rec.J, P.n, P.m)
    (pptr == P.ptr && pcol == P.col) || error("_plan_level: P's plan has not P's pattern")
    psrc = rec.src[sidx]
    rpos = htranspose_plan(P)
    tptr, tcol, tpp, ta, tb = hmul_plan(R, A); (tptr == T.ptr && tcol == T.col) || error("_plan_level: R A's pattern")
    cptr, ccol, cpp, ca, cb = hmul_plan(T, P); (cptr == Craw.ptr && ccol == Craw.col) || error("_plan_level: (R A) P's pattern")
    sp, scol, si1, si2, drops = hsym_plan(Craw); (sp == Ac.ptr && scol == Ac.col) || error("_plan_level: the symmetrisation's pattern")
    length(rec.scyc) == ns || error("_plan_level: one closing-entry pair per solve")
    merge(lf, (sline=rec.sline, wofs=wofs, nw=rec.nw, gptr=gptr, gq=gq, ge=ge, sptr=sptr, psrc=psrc, rpos=rpos,
               ce1=first.(rec.scyc), cem=last.(rec.scyc),
               tpp=tpp, ta=ta, tb=tb, nt=length(T.val), cpp=cpp, ca=ca, cb=cb, ncr=length(Craw.val), si1=si1, si2=si2,
               zskip=rec.zskip, drops=drops))
end

"The plan on the backend: Int32 index arrays, the scratch, the pivot segments (lines, solves, coarsest)."
function _mg_device_plan(backend, h::MGHierarchy, precs, gather)
    d(v) = _dev(backend, Int32.(v)); zf(n) = _zeros(backend, Float64, max(n, 1))
    nlev = length(h.levels); length(precs) == nlev || error("_mg_device_plan: $(length(precs)) level plans for $nlev levels")
    nl = sum(p.nlines for p in precs); nsol = sum(haskey(p, :sline) ? length(p.sline) : 0 for p in precs)
    Lc = h.levels[end]; ng = Lc.coarse === :dense ? Lc.n : 0
    lev = Any[]; loff = 0; soff = nl                     # the solves' segment follows the lines'
    for (k, p) in enumerate(precs)
        L = h.levels[k]
        base = (ib=d(p.ib), ia=d(p.ia), ic=d(p.ic), dptr=d(p.dptr), dsrc=d(p.dsrc), dbuf=zf(length(p.ib)),
                check=Int32(p.check), loff=Int32(loff), nlines=p.nlines)
        loff += p.nlines
        if L.coarse === :none
            q = (sline=d(p.sline), wofs=d(p.wofs), gptr=d(p.gptr), gq=d(p.gq), ge=d(p.ge), nsolve=length(p.sline),
                 wbuf=zf(p.nw), cpbuf=zf(p.nw), ubuf=zf(p.nw), ce1=d(p.ce1), cem=d(p.cem), sptr=d(p.sptr), psrc=d(p.psrc), rpos=d(p.rpos), np=length(p.rpos),
                 tpp=d(p.tpp), ta=d(p.ta), tb=d(p.tb), tval=zf(p.nt), nt=p.nt,
                 cpp=d(p.cpp), ca=d(p.ca), cb=d(p.cb), craw=zf(p.ncr), ncr=p.ncr,
                 si1=d(p.si1), si2=d(p.si2), nac=length(p.si1), soff=Int32(soff), zskip=p.zskip, drops=p.drops)
            soff += length(p.sline)
            push!(lev, merge(base, q))
        elseif L.coarse === :dense
            push!(lev, merge(base, (M0=KernelAbstractions.zeros(backend, Float64, L.n, L.n),
                                    M1=KernelAbstractions.zeros(backend, Float64, L.n, L.n))))
        else
            push!(lev, base)
        end
    end
    segptr = d([1, 1 + nl, 1 + nl + nsol, 1 + nl + nsol + ng])
    (gather=(sptr=d(gather.sptr), code=d(gather.code), nnz=length(gather.col), nc=gather.nc, nf=Int32(gather.nf)),
     levels=lev, piv=zf(nl + nsol + ng), segptr=segptr, goff=Int32(nl + nsol), pmin=_zeros(backend, Float64, 3),
     nseg=(nl, nsol, ng),
     zskip=sum(haskey(l, :zskip) ? l.zskip : 0 for l in lev), drops=sum(haskey(l, :drops) ? l.drops : 0 for l in lev))
end

# ---- the refresh's kernels -------------------------------------------------------------------------------------------
"Level 1's operator from the face system (facesys_hcsr's duplicate sums, in its order)."
@kernel function _k_rf_gather!(val, dg, up, lo, sptr, code, nf::Int32)
    e = @index(Global)
    @inbounds begin
        t0 = sptr[e]; c = code[t0]
        acc = c <= nf ? up[c] : (c <= Int32(2)*nf ? lo[c - nf] : dg[c - Int32(2)*nf])
        for t in t0+Int32(1):sptr[e+1]-Int32(1)
            c = code[t]
            acc += c <= nf ? up[c] : (c <= Int32(2)*nf ? lo[c - nf] : dg[c - Int32(2)*nf])
        end
        val[e] = acc
    end
end
"""One work item per line: the line factors (as `_linefac`) and the lumped diagonal of Schaffer's F-line solves; the
line's smallest pivot into piv (-Inf if one is not finite; +Inf on a level the build does not check)."""
@kernel function _k_rf_linefac!(la, lden, lcp, dbuf, val, lptr, ib, ia, ic, dptr, dsrc, piv, loff::Int32, check::Int32)
    l = @index(Global)
    @inbounds begin
        a0 = lptr[l]; a1 = lptr[l+1] - Int32(1)
        pmin = Inf; ok = true; cprev = 0.0
        for g in a0:a1
            b = val[ib[g]]
            av = ia[g] > Int32(0) ? val[ia[g]] : 0.0
            cv = ic[g] > Int32(0) ? val[ic[g]] : 0.0
            den = g == a0 ? b : b - av*cprev
            cp = cv/den
            la[g] = av; lden[g] = den; lcp[g] = cp; cprev = cp
            pmin = min(pmin, den); ok &= isfinite(den)
            s = 0.0
            for t in dptr[g]:dptr[g+1]-Int32(1); s += val[dsrc[t]]; end
            dbuf[g] = s
        end
        piv[loff + l] = check == Int32(1) ? (ok ? pmin : -Inf) : Inf
    end
end
"""One work item per Schaffer solve: its right side (the coupled entries, summed in order, negated) and the Thomas solve
of (a, d, c) in place (`_thomas`'s operations); the solve's smallest pivot into piv. THE PIVOT FIX (plan 28.6): on a
cyclic line (ce1[s] > 0, the positions of its closing entries (1, m) and (m, 1)) whose lumped solve has a non-positive or
non-finite pivot, the right side is gathered again and the cyclic line solve `_thomas_cyclic` is done instead, operation
for operation (u the scratch of its second solve), so the refresh equals a host build of the new operator bit for bit."""
@kernel function _k_rf_psolve!(x, cpb, u, val, dbuf, ia, ic, lptr, sline, wofs, gptr, gq, ge, ce1, cem, piv, soff::Int32)
    s = @index(Global)
    @inbounds begin
        l = sline[s]; a0 = lptr[l]; m = lptr[l+1] - a0; o = wofs[s]
        for q in Int32(0):m-Int32(1); x[o+q] = 0.0; end
        for t in gptr[s]:gptr[s+1]-Int32(1); x[o+gq[t]-Int32(1)] += val[ge[t]]; end
        for q in Int32(0):m-Int32(1); x[o+q] = -x[o+q]; end
        den = dbuf[a0]; pmin = den
        c1 = ic[a0] > Int32(0) ? val[ic[a0]] : 0.0
        cpb[o] = m > Int32(1) ? c1/den : 0.0
        x[o] = x[o]/den
        for q in Int32(2):m
            g = a0 + q - Int32(1)
            aq = ia[g] > Int32(0) ? val[ia[g]] : 0.0
            den = dbuf[g] - aq*cpb[o+q-Int32(2)]
            pmin = min(pmin, den)
            if q < m
                cq = ic[g] > Int32(0) ? val[ic[g]] : 0.0
                cpb[o+q-Int32(1)] = cq/den
            end
            x[o+q-Int32(1)] = (x[o+q-Int32(1)] - aq*x[o+q-Int32(2)])/den
        end
        for q in m-Int32(1):Int32(-1):Int32(1)
            x[o+q-Int32(1)] -= cpb[o+q-Int32(1)]*x[o+q]
        end
        if !(pmin > 0 && isfinite(pmin)) && ce1[s] > Int32(0)
            v1 = val[ce1[s]]; vm = val[cem[s]]
            for q in Int32(0):m-Int32(1); x[o+q] = 0.0; end
            for t in gptr[s]:gptr[s+1]-Int32(1); x[o+gq[t]-Int32(1)] += val[ge[t]]; end
            for q in Int32(0):m-Int32(1); x[o+q] = -x[o+q]; end
            d1 = dbuf[a0] - v1; dm = dbuf[a0+m-Int32(1)] - vm
            gam = -d1
            b1 = d1 - gam; bm = dm - vm*v1/gam
            den = b1; pmin = den
            c1 = ic[a0] > Int32(0) ? val[ic[a0]] : 0.0
            cpb[o] = c1/den; x[o] = x[o]/den; u[o] = gam/den
            for q in Int32(2):m
                g = a0 + q - Int32(1)
                aq = ia[g] > Int32(0) ? val[ia[g]] : 0.0
                bq = q == m ? bm : dbuf[g]
                den = bq - aq*cpb[o+q-Int32(2)]
                pmin = min(pmin, den)
                if q < m
                    cq = ic[g] > Int32(0) ? val[ic[g]] : 0.0
                    cpb[o+q-Int32(1)] = cq/den
                end
                x[o+q-Int32(1)] = (x[o+q-Int32(1)] - aq*x[o+q-Int32(2)])/den
                uq = q == m ? vm : 0.0
                u[o+q-Int32(1)] = (uq - aq*u[o+q-Int32(2)])/den
            end
            for q in m-Int32(1):Int32(-1):Int32(1)
                x[o+q-Int32(1)] -= cpb[o+q-Int32(1)]*x[o+q]
                u[o+q-Int32(1)] -= cpb[o+q-Int32(1)]*u[o+q]
            end
            D = 1.0 + u[o] + v1*u[o+m-Int32(1)]/gam
            fact = (x[o] + v1*x[o+m-Int32(1)]/gam)/D
            for q in Int32(0):m-Int32(1); x[o+q] -= fact*u[o+q]; end
            D > 0 || (pmin = min(pmin, D))
        end
        piv[soff + s] = pmin
    end
end
"P's values (duplicates summed in order; 0 = a C cell's 1.0) and R's, its transpose's, at the mapped positions."
@kernel function _k_rf_pval!(Pv, Rv, w, sptr, psrc, rpos)
    e = @index(Global)
    @inbounds begin
        t0 = sptr[e]; s0 = psrc[t0]
        acc = s0 == Int32(0) ? 1.0 : w[s0]
        for t in t0+Int32(1):sptr[e+1]-Int32(1)
            st = psrc[t]; acc += st == Int32(0) ? 1.0 : w[st]
        end
        Pv[e] = acc; Rv[rpos[e]] = acc
    end
end
"One entry of a sparse product on its plan: the first pair's product assigned, the others added in turn (as `hmul`)."
@kernel function _k_rf_spgemm!(out, Av, Bv, pptr, pa, pb)
    e = @index(Global)
    @inbounds begin
        t0 = pptr[e]; acc = Av[pa[t0]]*Bv[pb[t0]]
        for t in t0+Int32(1):pptr[e+1]-Int32(1); acc += Av[pa[t]]*Bv[pb[t]]; end
        out[e] = acc
    end
end
"The symmetrisation (A + A')/2 on its plan (as `hsym_drop`), into the next level's operator."
@kernel function _k_rf_sym!(out, raw, i1, i2)
    e = @index(Global)
    @inbounds begin
        j1 = i1[e]; j2 = i2[e]
        if j1 > Int32(0) && j2 > Int32(0)
            out[e] = (raw[j1] + raw[j2])/2
        elseif j1 > Int32(0)
            out[e] = (raw[j1] + 0.0)/2
        else
            out[e] = (0.0 + raw[j2])/2
        end
    end
end
"The dense matrix of a level's operator (as `hdense`), one row per work item."
@kernel function _k_rf_dense!(M, val, ptr, col, n::Int32)
    r = @index(Global)
    @inbounds begin
        for j in Int32(1):n; M[r, j] = 0.0; end
        for q in ptr[r]:ptr[r+1]-Int32(1); M[r, col[q]] = val[q]; end
    end
end
"""One Gauss-Jordan sweep on pivot k, out of place (B from A): after the n sweeps B = A^-1 (no pivoting: the operator is
SPD, and its pivots are recorded in piv for the refusal)."""
@kernel function _k_rf_gj!(B, A, piv, k::Int32, n::Int32, goff::Int32)
    t = @index(Global)
    @inbounds begin
        i = Int32((t - 1) % n + 1); j = Int32((t - 1) ÷ n + 1)
        p = A[k, k]
        if i == k && j == k
            B[i, j] = 1/p; piv[goff + k] = p
        elseif i == k
            B[i, j] = A[k, j]/p
        elseif j == k
            B[i, j] = -A[i, k]/p
        else
            B[i, j] = A[i, j] - A[i, k]*(A[k, j]/p)
        end
    end
end
"The smallest value of each pivot segment (one work item per segment; +Inf when empty)."
@kernel function _k_rf_segmin!(out, piv, segptr)
    t = @index(Global)
    @inbounds begin
        s = Inf
        for i in segptr[t]:segptr[t+1]-Int32(1); s = min(s, piv[i]); end
        out[t] = s
    end
end

"""
    mg_refresh!(h::MGHierarchy, sys::FaceSystem; coarse=:host, timer=nothing) -> h

Recompute every value of the hierarchy (each level's operator, line factors, P, R, the coarsest inverse) from the face
system `sys`, which must have the sparsity the hierarchy's plan was made from (the same mesh), on the hierarchy's backend
(review F5; plan 24.1, O1). Needs a hierarchy built with `plan = true`. Refuses, as the build does, any non-positive or
non-finite line-factor pivot (on the levels the build checks), interpolation pivot, or coarsest Gauss-Jordan pivot.
One host read (the three pivot minima). `coarse` (a dense coarsest level only): :gj, the device Gauss-Jordan sweeps (plan
24.1 as registered); :host, the build's own arithmetic on a host copy of the coarsest operator (one read-back of its
values, LAPACK's inverse, one upload; bit for bit the build's inverse on the CPU; written after M2's data, plan 24.6).
THE DEFAULT is :host, the decided and registered coarsest inverse (plan 24.6; PBConfig.mg_coarse). Until the re-review's R6
(14 Oct 2026) it was :gj, the inverse that decision rejected. `timer`, a Dict{Symbol,Float64}, accumulates the seconds of each kernel class
(synchronising after each launch: for measurement only).
"""
function mg_refresh!(h::MGHierarchy, sys::FaceSystem; coarse::Symbol=:host, timer=nothing)
    coarse in (:gj, :host) || throw(ArgumentError("mg_refresh!: coarse must be :gj or :host (got $coarse)"))
    pl = h.plan
    pl === nothing && throw(ArgumentError("mg_refresh!: the hierarchy has no numeric plan (build it with plan = true)"))
    be = h.backend; g = pl.gather
    (length(sys.diag) == g.nc && length(sys.upper) == g.nf) ||
        throw(DimensionMismatch("mg_refresh!: the face system is not the plan's ($(g.nc) cells, $(g.nf) faces)"))
    function go(cls::Symbol, kern, n, args...)
        if timer === nothing
            _launch!(kern, be, n, args...)
        else
            t0 = time_ns(); _launch!(kern, be, n, args...); KernelAbstractions.synchronize(be)
            timer[cls] = get(timer, cls, 0.0) + (time_ns() - t0)*1e-9
            timer[Symbol(cls, :_n)] = get(timer, Symbol(cls, :_n), 0.0) + 1
        end
        h.refresh_launches += n > 0 ? 1 : 0
        nothing
    end
    L1 = h.levels[1]; coarse_bad = nothing
    go(:gather, _k_rf_gather!, g.nnz, L1.val, sys.diag, sys.upper, sys.lower, g.sptr, g.code, g.nf)
    for k in eachindex(h.levels)
        L = h.levels[k]; p = pl.levels[k]
        go(:linefac, _k_rf_linefac!, p.nlines, L.la, L.lden, L.lcp, p.dbuf, L.val, L.lptr, p.ib, p.ia, p.ic, p.dptr,
           p.dsrc, pl.piv, p.loff, p.check)
        if L.coarse === :none
            Ln = h.levels[k+1]
            go(:psolve, _k_rf_psolve!, p.nsolve, p.wbuf, p.cpbuf, p.ubuf, L.val, p.dbuf, p.ia, p.ic, L.lptr, p.sline,
               p.wofs, p.gptr, p.gq, p.ge, p.ce1, p.cem, pl.piv, p.soff)
            go(:pval, _k_rf_pval!, p.np, L.Pval, L.Rval, p.wbuf, p.sptr, p.psrc, p.rpos)
            go(:RA, _k_rf_spgemm!, p.nt, p.tval, L.Rval, L.val, p.tpp, p.ta, p.tb)
            go(:RAP, _k_rf_spgemm!, p.ncr, p.craw, p.tval, L.Pval, p.cpp, p.ca, p.cb)
            go(:sym, _k_rf_sym!, p.nac, Ln.val, p.craw, p.si1, p.si2)
        elseif L.coarse === :dense && coarse === :host
            t0 = time_ns()
            pb_sync(be)
            Ah = HCSR(L.hA.n, L.hA.m, L.hA.ptr, L.hA.col, _host(L.val))      # the coarsest operator, read back once
            Dn = hdense(Ah)
            # a failure here is reported after the pivot checks below, so the first defect named is the finest level's,
            # as in the build (which checks level by level)
            if !isposdef(Symmetric(Dn))
                coarse_bad = "mg_refresh!: the coarsest operator ($(L.n) cells) is not positive definite (refused)"
            else
                Mi = inv(Dn)
                all(isfinite, Mi) || (coarse_bad = "mg_refresh!: the coarsest inverse is not finite (refused)")
                coarse_bad === nothing && copyto!(L.Minv, Mi)
            end
            pb_sync(be)
            if timer !== nothing
                timer[:host_inv] = get(timer, :host_inv, 0.0) + (time_ns() - t0)*1e-9
                timer[:host_inv_n] = get(timer, :host_inv_n, 0.0) + 1
            end
        elseif L.coarse === :dense
            n = Int32(L.n)
            go(:dense, _k_rf_dense!, L.n, p.M0, L.val, L.ptr, L.col, n)
            src, dst = p.M0, p.M1
            for kk in Int32(1):n
                go(:gj, _k_rf_gj!, L.n*L.n, dst, src, pl.piv, kk, n, pl.goff)
                src, dst = dst, src
            end
            if timer === nothing
                copyto!(L.Minv, src)
            else
                t0 = time_ns(); copyto!(L.Minv, src); KernelAbstractions.synchronize(be)
                timer[:gj] = get(timer, :gj, 0.0) + (time_ns() - t0)*1e-9
                timer[:copy_n] = get(timer, :copy_n, 0.0) + 1
            end
            h.refresh_launches += 1
        end
    end
    go(:check, _k_rf_segmin!, 3, pl.pmin, pl.piv, pl.segptr)
    pb_sync(be)
    pm = _host(pl.pmin)
    # (pm > 0 refuses NaN, -Inf (a non-finite line pivot) and non-positive; an empty segment's +Inf passes)
    (pm[1] > 0 && (isfinite(pm[1]) || pl.nseg[1] == 0)) ||
        throw(ArgumentError(@sprintf("mg_refresh!: a non-positive line-factor pivot %.3e (refused: the cycle would be indefinite)", pm[1])))
    (pm[2] > 0 && (isfinite(pm[2]) || pl.nseg[2] == 0)) ||
        throw(ArgumentError(@sprintf("mg_refresh!: a non-positive interpolation pivot %.3e (refused)", pm[2])))
    (coarse === :host || (pm[3] > 0 && (isfinite(pm[3]) || pl.nseg[3] == 0))) ||
        throw(ArgumentError(@sprintf("mg_refresh!: the coarsest operator is not positive definite (Gauss-Jordan pivot %.3e; refused)", pm[3])))
    coarse_bad === nothing || throw(ArgumentError(coarse_bad))
    h.min_line_pivot = pm[1]; h.min_interp_pivot = pm[2]; h.refreshes += 1
    h
end
