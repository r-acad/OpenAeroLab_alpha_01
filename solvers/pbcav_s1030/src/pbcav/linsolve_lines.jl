# =====================================================================================================================
# linsolve_lines.jl -- line solves along the mesh's j-lines and i-lines (plan 5.4-5.5): coloured line Gauss-Seidel and the
# Thomas algorithm (one work item per line; the CPU reference and, until stage P3's workgroup PCR, the GPU form too).
# Float64 here (the momentum sweeps' Float32 option is a stage-P3 cost item). Stage P0b, 14 Oct 2026. W7 (plan 33, 22 Oct
# 2026): the workgroup PCR and the Float32 line copies, opt-in, at the end of this file.
#
# A LINE FAMILY is the mesh's CSR line list (mesh.jl_* or mesh.il_*). Its tridiagonal T_l of a face system holds, for
# consecutive cells c_q, c_{q+1} joined by internal face f, the row coefficients on each other (upper[f] or lower[f],
# by which side owns f) and the diagonal; a cyclic line's closing face is NOT in T_l (it stays an off-line coupling, so
# the sweep is still exact Gauss-Seidel on the other couplings).
# COLOURS. The lines of a family are coloured greedily on their adjacency graph (two lines adjacent when an internal face
# joins them), once per mesh, on the host (integers only). A sweep relaxes the colours in order: residual r = b - A x,
# T_l z = r_l on every line of the colour in parallel, x += z -- the same smoother the multigrid uses.
# KernelAbstractions issue 652: no kernel here uses @Const (see PBCav.jl).
# =====================================================================================================================

"""
    LineFamily

A line family on the backend: `ptr, cell` (lines, CSR), `fptr, face` (the faces between consecutive cells), `colours`
(one device array of line ids per colour), `nlines`, and the tridiagonal storage `ta, td, tc` aligned with `cell`
(ta coupling to the previous cell, td the diagonal, tc to the next; den/cp the Thomas factors).
"""
struct LineFamily{VI,VF}
    family::Symbol
    nlines::Int
    maxlen::Int               # the longest line, cells (W7: the workgroup PCR sizes its shared memory by it)
    ptr::VI; cell::VI; fptr::VI; face::VI
    colours::Vector{VI}
    ta::VF; td::VF; tc::VF; den::VF; cp::VF
end

"Greedy colouring of a line family (host, integers; once per mesh)."
function _line_colours(m::PBMesh, ptr::Vector{Int32}, cell::Vector{Int32})
    nl = length(ptr) - 1
    lo = zeros(Int, m.nc)
    for l in 1:nl, q in ptr[l]:ptr[l+1]-1; lo[cell[q]] = l; end
    own = _host(m.own); nei = _host(m.nei)
    nb = [Set{Int}() for _ in 1:nl]
    for f in 1:m.nf
        a = lo[own[f]]; b = lo[nei[f]]
        (a > 0 && b > 0 && a != b) && (push!(nb[a], b); push!(nb[b], a))
    end
    col = zeros(Int, nl)
    for l in 1:nl
        used = Set(col[q] for q in nb[l] if col[q] > 0); c = 1
        while c in used; c += 1; end
        col[l] = c
    end
    [Int32.(findall(==(c), col)) for c in 1:maximum(col; init=0)]
end

"""
    line_family(mesh, family::Symbol; backend=CPU()) -> LineFamily

`family` = :j (mesh.jl_*) or :i (mesh.il_*).
"""
function line_family(m::PBMesh, family::Symbol; backend=CPU())
    family in (:j, :i) || throw(ArgumentError("line family must be :j or :i"))
    ptr, cell, fptr, face = family === :j ? (m.jl_ptr, m.jl_cell, m.jl_fptr, m.jl_face) :
                                            (m.il_ptr, m.il_cell, m.il_fptr, m.il_face)
    cols = _line_colours(m, _host(ptr), _host(cell))
    n = length(cell); z() = KernelAbstractions.zeros(backend, Float64, n)
    hp = _host(ptr)
    LineFamily(family, length(ptr) - 1, maximum(diff(hp); init=0), ptr, cell, fptr, face, [_dev(backend, c) for c in cols],
               z(), z(), z(), z(), z())
end

"Fill a family's tridiagonal (and its Thomas factors) from a face system: one work item per line."
@kernel function _k_line_tridiag!(ta, td, tc, den, cp, ptr, cell, fptr, face, dg, up, lo, own)
    l = @index(Global)
    @inbounds begin
        a0 = ptr[l]; a1 = ptr[l+1] - Int32(1); f0 = fptr[l]
        for q in a0:a1
            c = cell[q]; td[q] = dg[c]; ta[q] = 0.0; tc[q] = 0.0
        end
        for q in a0:a1-Int32(1)
            f = face[f0 + (q - a0)]; c = cell[q]
            if own[f] == c                  # c owns f: its row's coefficient on the next cell is upper[f]
                tc[q] = up[f]; ta[q+Int32(1)] = lo[f]
            else
                tc[q] = lo[f]; ta[q+Int32(1)] = up[f]
            end
        end
        d = td[a0]; den[a0] = d; cpv = tc[a0]/d; cp[a0] = cpv
        for q in a0+Int32(1):a1
            d = td[q] - ta[q]*cpv; den[q] = d; cpv = tc[q]/d; cp[q] = cpv
        end
    end
end

"""Thomas on the lines listed in `lines` (one work item each): z = T^-1 r on the line; then x += z (mode 0), x = z (mode 1),
or only z (mode 2). The recurrences are carried in registers."""
@kernel function _k_line_thomas!(x, z, r, lines, ptr, cell, ta, den, cp, mode::Int32)
    t = @index(Global)
    @inbounds begin
        l = lines[t]; a0 = ptr[l]; a1 = ptr[l+1] - Int32(1)
        zp = r[cell[a0]]/den[a0]; z[cell[a0]] = zp
        for q in a0+Int32(1):a1
            zq = (r[cell[q]] - ta[q]*zp)/den[q]; z[cell[q]] = zq; zp = zq
        end
        zn = zp
        for q in a1-Int32(1):Int32(-1):a0
            zq = z[cell[q]] - cp[q]*zn; z[cell[q]] = zq; zn = zq
        end
        if mode == Int32(0)
            for q in a0:a1; c = cell[q]; x[c] += z[c]; end
        elseif mode == Int32(1)
            for q in a0:a1; c = cell[q]; x[c] = z[c]; end
        end
    end
end

"Load a family's tridiagonal from `sys` (call once per assembled system)."
function line_tridiag!(fam::LineFamily, sys::FaceSystem, m::PBMesh; backend=CPU())
    _launch!(_k_line_tridiag!, backend, fam.nlines, fam.ta, fam.td, fam.tc, fam.den, fam.cp, fam.ptr, fam.cell, fam.fptr,
             fam.face, sys.diag, sys.upper, sys.lower, m.own)
    pb_sync(backend)
    fam
end

"""
    thomas_lines!(z, fam::LineFamily, r; backend=CPU())

Solve T_l z_l = r_l on every line of the family (the tridiagonal loaded by `line_tridiag!`). The plan's interface name
`thomas_lines!(z, sys, r, mesh, family)` is the method below.
"""
function thomas_lines!(z, fam::LineFamily, r; backend=CPU())
    alll = _dev(backend, Int32.(1:fam.nlines))
    _launch!(_k_line_thomas!, backend, fam.nlines, z, z, r, alll, fam.ptr, fam.cell, fam.ta, fam.den, fam.cp, Int32(2))
    pb_sync(backend)
    z
end
function thomas_lines!(z, sys::FaceSystem, r, m::PBMesh, family::Symbol; backend=CPU())
    fam = line_tridiag!(line_family(m, family; backend), sys, m; backend)
    thomas_lines!(z, fam, r; backend)
end

"""
    line_gs_sweep!(x, sys::FaceSystem, mesh, fam::LineFamily, r_work; backend=CPU(), reverse=false)

One coloured line Gauss-Seidel sweep of `sys` along the family (its tridiagonal loaded by `line_tridiag!` from the same
system): for each colour, r = b - A x, then T_l z = r_l and x += z on every line of the colour.
"""
function line_gs_sweep!(x, sys::FaceSystem, m::PBMesh, fam::LineFamily, rw, zw; backend=CPU(), reverse::Bool=false)
    order = reverse ? (length(fam.colours):-1:1) : (1:length(fam.colours))
    for ci in order
        set = fam.colours[ci]
        _launch!(_k_fs_resid!, backend, m.nc, rw, sys.rhs, x, sys.diag, sys.upper, sys.lower, m.cface, m.own, m.nei,
                 Int32(m.nf))
        _launch!(_k_line_thomas!, backend, length(set), x, zw, rw, set, fam.ptr, fam.cell, fam.ta, fam.den, fam.cp, Int32(0))
    end
    pb_sync(backend)
    x
end

# =====================================================================================================================
# W7 (speed; plan Amendment 13, section 33; 22 Oct 2026): THE LINE SOLVER AND ITS PRECISION, opt-in. Nothing above changes:
# `line_gs_sweep!` without `solver` is the Thomas sweep above, bit for bit.
#   * THE WORKGROUP PCR (plan 5.5): one workgroup per line, the line's tridiagonal (ta, td, tc) and right side in
#     @localmem, ceil(log2 NM) parallel-cyclic-reduction steps (Hockney's reduction in its parallel form; Zhang, Cohen and
#     Owens 2010), each a read phase into @private registers and a write phase, separated by @synchronize; then
#     z = d/b per cell. NM, the shared length, is the family's longest line rounded up to a power of two (at most
#     PCR_NMAX = 1024 in Float64, 2048 in Float32: a longer family takes Thomas); each of the W = min(NM, 256) work items holds NM/W cells. Exact in
#     exact arithmetic (it is Gaussian elimination in another order), so it differs from Thomas only by rounding; stable
#     for the diagonally dominant and SPD line systems the solver forms. On the GPU a line costs about log2 NM shared-memory
#     steps instead of one dependent global-memory round trip per cell (Thomas's one work item per line). On the CPU it is
#     correct but slower than Thomas (O(n log n) work, a workgroup loop per phase): CPU runs keep :thomas.
#   * THE PRECISION OF A LINE SOLVE: the factors and the work vector of a family may be Float32 (`line_family_like`,
#     loaded from the Float64 family by `line_copy!`, the factors formed in Float64 and rounded: the cancellation-prone
#     recurrence d_q - a_q c_{q-1}/d_{q-1} is evaluated in Float64, so the rounded Thomas factors carry no cancellation
#     error). The kernels below convert the right side to the factors' type on load and the solution back to x's type on
#     store, so a Float64 residual can drive a Float32 correction (mixed precision: the k-omega solves) or the whole sweep
#     can be Float32 (the coupled preconditioner).
#   * CORRECTION (22 Oct 2026, after the W7 review's finding 1): THE PCR ELIMINATES IN FLOAT64 ONLY. Before, a Float32
#     family ran the PCR in Float32 on its rounded (ta, td, tc): the reduction b' = b - c a/b - a c/b then cancels in Float32,
#     and rounding td alone destroys a relative row margin below about 6e-8 (wall-normal lines of high-aspect-ratio cells:
#     A2's and CH's pressure levels lost 2-5 orders, 5.5e-2 against Float32 Thomas's 2e-6). Now a Float32 family asks for the
#     PCR with its Float64 family `fam64` (the same lines; its tridiagonal formed in Float64): the elimination runs on fam64
#     in Float64, the Float32 right side converted on load and the correction rounded to x's type on store; without fam64,
#     or above PCR_NMAX, a Float32 family takes Float32 Thomas (factors from Float64). No Float32 PCR exists.
#   * THE EXCESS RESIDUAL (the plan's row-excess form, 5.2; same correction): a Float32 residual r = b - A x on a rounded
#     operator cancels when the row margin is below Float32's resolution, and loses the operator's near-null space (A 1 = the
#     row sums E, small). `_k_fs_resid_ex!` forms r_c = b_c - E_c x_c - sum_f a_f (x_nb - x_c), E_c the row sum formed in
#     Float64 (`_k_fs_rowsum!`) and rounded: the same operator algebraically, with its row sums carried exactly.
# =====================================================================================================================
"The longest line the workgroup PCR takes (shared memory: 4 NM Float64 entries, 32 kB; the PCR runs in Float64 only)."
const PCR_NMAX = 1024
"The line solvers (`line_solver` of PBConfig): :thomas (the default, one work item per line) or :pcr."
const LINE_SOLVERS = (:thomas, :pcr)

"""Thomas with the factors' element type T = eltype(den) (W7): z = T^-1 r on each listed line, the right side converted to T
on load; then x += z (mode 0), x = z (mode 1), or only z (mode 2). With every array Float64 it is `_k_line_thomas!`'s
arithmetic."""
@kernel function _k_line_thomas_t!(x, z, r, lines, ptr, cell, ta, den, cp, mode::Int32)
    t = @index(Global)
    @inbounds begin
        T = eltype(den)
        l = lines[t]; a0 = ptr[l]; a1 = ptr[l+1] - Int32(1)
        zp = convert(T, r[cell[a0]])/den[a0]; z[cell[a0]] = zp
        for q in a0+Int32(1):a1
            zq = (convert(T, r[cell[q]]) - ta[q]*zp)/den[q]; z[cell[q]] = zq; zp = zq
        end
        zn = zp
        for q in a1-Int32(1):Int32(-1):a0
            zq = z[cell[q]] - cp[q]*zn; z[cell[q]] = zq; zn = zq
        end
        if mode == Int32(0)
            for q in a0:a1; c = cell[q]; x[c] += z[c]; end
        elseif mode == Int32(1)
            for q in a0:a1; c = cell[q]; x[c] = z[c]; end
        end
    end
end

"""The workgroup PCR (W7; the section header above): workgroup g solves line lines[g]; NM the shared length (a power of
two, at least the line's length), NPT = NM/W cells per work item, LG = log2 NM reduction steps. The tridiagonal is
(ta, td, tc) aligned with `cell` (ta[a0] = tc[a1] = 0), in the element type T of td (Float64: `line_solve_set!` launches
nothing else); the right side r is converted to T on load. Then x += z (mode 0), x = z (mode 1), or z only (mode 2), each
sum formed in T and rounded once to the stored array's type."""
@kernel function _k_line_pcr!(x, z, r, lines, ptr, cell, ta, td, tc, mode::Int32, ::Val{NM}, ::Val{NPT},
                              ::Val{LG}) where {NM, NPT, LG}
    # (KernelAbstractions' CPU form runs each phase between two @synchronize as its own loop over the work items, so every
    # per-item value a phase needs -- the line, its start and length -- is recomputed in it; T and W are uniform.)
    g = @index(Group, Linear)
    li = @index(Local, Linear)
    @uniform T = eltype(td)
    @uniform W = @groupsize()[1]
    sa = @localmem T (NM,)
    sb = @localmem T (NM,)
    sc = @localmem T (NM,)
    sd = @localmem T (NM,)
    na = @private T (NPT,)
    nb = @private T (NPT,)
    nc = @private T (NPT,)
    nd = @private T (NPT,)
    @inbounds begin
        l = lines[g]; a0 = ptr[l]; n = ptr[l+1] - a0
        for k in 1:NPT
            q = li + (k - 1)*W
            if q <= n
                gq = a0 + q - Int32(1)
                sa[q] = ta[gq]; sb[q] = td[gq]; sc[q] = tc[gq]; sd[q] = convert(T, r[cell[gq]])
            end
        end
    end
    @synchronize
    for st in 0:LG-1
        @inbounds begin
            l = lines[g]; n = ptr[l+1] - ptr[l]; s = 1 << st
            for k in 1:NPT
                q = li + (k - 1)*W
                if q <= n
                    bq = sb[q]; dq = sd[q]; qm = q - s; qp = q + s
                    if qm >= 1
                        k1 = sa[q]/sb[qm]
                        na[k] = -sa[qm]*k1; bq -= sc[qm]*k1; dq -= sd[qm]*k1
                    else
                        na[k] = zero(T)
                    end
                    if qp <= n
                        k2 = sc[q]/sb[qp]
                        nc[k] = -sc[qp]*k2; bq -= sa[qp]*k2; dq -= sd[qp]*k2
                    else
                        nc[k] = zero(T)
                    end
                    nb[k] = bq; nd[k] = dq
                end
            end
        end
        @synchronize
        @inbounds begin
            l = lines[g]; n = ptr[l+1] - ptr[l]
            for k in 1:NPT
                q = li + (k - 1)*W
                if q <= n
                    sa[q] = na[k]; sb[q] = nb[k]; sc[q] = nc[k]; sd[q] = nd[k]
                end
            end
        end
        @synchronize
    end
    @inbounds begin
        l = lines[g]; a0 = ptr[l]; n = ptr[l+1] - a0
        for k in 1:NPT
            q = li + (k - 1)*W
            if q <= n
                c = cell[a0 + q - Int32(1)]; zq = sd[q]/sb[q]
                if mode == Int32(0)
                    x[c] = convert(eltype(x), x[c] + zq)
                elseif mode == Int32(1)
                    x[c] = convert(eltype(x), zq)
                else
                    z[c] = convert(eltype(z), zq)
                end
            end
        end
    end
end

"""The PCR launch shape (Float64) of a family whose longest line has `maxlen` cells: (NM, W, NPT, LG), or nothing above
the shared-memory limit PCR_NMAX (4 NM Float64 entries, 32 kB)."""
function pcr_shape(maxlen::Integer)
    maxlen <= PCR_NMAX || return nothing
    NM = max(32, nextpow(2, max(Int(maxlen), 1))); W = min(NM, 256)
    (NM, W, NM ÷ W, trailing_zeros(NM))
end

"""
    pcr_family(fam, fam64) -> LineFamily or nothing

The family whose tridiagonal a PCR solve of `fam`'s lines eliminates: `fam` itself when it is Float64, else `fam64` (its
Float64 family: the same line arrays, refused otherwise), or nothing (Float32 Thomas) without fam64 or above PCR_NMAX.
"""
function pcr_family(fam::LineFamily, fam64)
    pf = eltype(fam.td) === Float64 ? fam : fam64
    pf === nothing && return nothing
    (eltype(pf.td) === Float64 && pf.ptr === fam.ptr && pf.cell === fam.cell) ||
        throw(ArgumentError("line_solve_set!: fam64 must be the Float64 family of the same lines"))
    pcr_shape(pf.maxlen) === nothing ? nothing : pf
end

"""Solve the lines listed in `set` (a device array of line ids) of a family with `solver` (:thomas or :pcr): z = T^-1 r,
then x += z (mode 0), x = z (mode 1) or z only (mode 2). Thomas runs in the family's precision (Float64, or Float32 with
factors formed in Float64); the PCR in Float64 only, on `fam` when it is Float64, else on its Float64 family `fam64`
(`pcr_family`; without one, or above PCR_NMAX, Thomas)."""
function line_solve_set!(x, z, r, set, fam::LineFamily, mode::Int32; backend=CPU(), solver::Symbol=:thomas, fam64=nothing)
    nl = length(set); nl == 0 && return nothing
    pf = solver === :pcr ? pcr_family(fam, fam64) : nothing
    if pf === nothing
        if eltype(fam.den) === Float64 && eltype(r) === Float64 && eltype(z) === Float64
            _launch!(_k_line_thomas!, backend, nl, x, z, r, set, fam.ptr, fam.cell, fam.ta, fam.den, fam.cp, mode)
        else
            _launch!(_k_line_thomas_t!, backend, nl, x, z, r, set, fam.ptr, fam.cell, fam.ta, fam.den, fam.cp, mode)
        end
    else
        NM, W, NPT, LG = pcr_shape(pf.maxlen)
        _launch_wg!(_k_line_pcr!, backend, nl*W, W, x, z, r, set, pf.ptr, pf.cell, pf.ta, pf.td, pf.tc, mode,
                    Val(NM), Val(NPT), Val(LG))
    end
    nothing
end

"""The excess residual (W7; the section header): r_c = b_c - E_c x_c - sum_f a_f (x_nb - x_c) over the cell's internal
faces, E_c the row sum (`_k_fs_rowsum!`); algebraically `_k_fs_resid!`'s r = b - A x."""
@kernel function _k_fs_resid_ex!(r, b, x, ex, up, lo, cface, own, nei, nf::Int32)
    c = @index(Global)
    @inbounds begin
        xc = x[c]; acc = ex[c]*xc
        for k in 1:4
            code = cface[k, c]
            if 0 < code <= nf
                acc += up[code]*(x[nei[code]] - xc)
            elseif code < 0
                acc += lo[-code]*(x[own[-code]] - xc)
            end
        end
        r[c] = b[c] - acc
    end
end
"""The row sums E_c = diag_c + sum_f a_f of a face system (the couplings `_k_fs_resid!` applies), accumulated in Float64
and rounded to ex's element type."""
@kernel function _k_fs_rowsum!(ex, dg, up, lo, cface, nf::Int32)
    c = @index(Global)
    @inbounds begin
        acc = Float64(dg[c])
        for k in 1:4
            code = cface[k, c]
            if 0 < code <= nf
                acc += Float64(up[code])
            elseif code < 0
                acc += Float64(lo[-code])
            end
        end
        ex[c] = convert(eltype(ex), acc)
    end
end
"Load `ex` (length nc, any float type) with the row sums of the face system `sys` (Float64)."
fs_rowsum!(ex, sys::FaceSystem, m::PBMesh; backend=CPU()) =
    (_launch!(_k_fs_rowsum!, backend, m.nc, ex, sys.diag, sys.upper, sys.lower, m.cface, Int32(m.nf)); ex)

"""
    line_gs_sweep!(x, sys, mesh, fam, rw, zw, solver; backend=CPU(), reverse=false)

W7's form of the coloured line Gauss-Seidel sweep: the line solves by `solver` (:thomas or :pcr) and in the family's
precision (`fam` may be a Float32 copy, `line_family_like`; `sys`, x and the residual work rw may be Float64 or Float32: the
residual is formed in rw's type, the correction in the family's). `line_gs_sweep!` without `solver` is the Thomas sweep,
bit for bit; with solver = :thomas and every array Float64, this method is that sweep's arithmetic. `fam64`: a Float32
family's Float64 family, on which the PCR then eliminates (`line_solve_set!`); `ex`: the system's row sums
(`fs_rowsum!`), when given the residual is the excess form `_k_fs_resid_ex!` (W7's correction of 22 Oct 2026).
"""
function line_gs_sweep!(x, sys::FaceSystem, m::PBMesh, fam::LineFamily, rw, zw, solver::Symbol; backend=CPU(),
                        reverse::Bool=false, fam64=nothing, ex=nothing)
    order = reverse ? (length(fam.colours):-1:1) : (1:length(fam.colours))
    for ci in order
        set = fam.colours[ci]
        if ex === nothing
            _launch!(_k_fs_resid!, backend, m.nc, rw, sys.rhs, x, sys.diag, sys.upper, sys.lower, m.cface, m.own, m.nei,
                     Int32(m.nf))
        else
            _launch!(_k_fs_resid_ex!, backend, m.nc, rw, sys.rhs, x, ex, sys.upper, sys.lower, m.cface, m.own, m.nei,
                     Int32(m.nf))
        end
        line_solve_set!(x, zw, rw, set, fam, Int32(0); backend, solver, fam64)
    end
    pb_sync(backend)
    x
end

"""
    line_family_like(fam, T; backend) -> LineFamily

A copy of a family's structure (the same line, face and colour arrays) with its tridiagonal and factor storage in element
type T (Float32 for W7's mixed precision); load it from the Float64 family with `line_copy!`.
"""
function line_family_like(fam::LineFamily, ::Type{T}; backend=CPU()) where {T}
    z() = KernelAbstractions.zeros(backend, T, length(fam.cell))
    LineFamily(fam.family, fam.nlines, fam.maxlen, fam.ptr, fam.cell, fam.fptr, fam.face, fam.colours, z(), z(), z(), z(), z())
end
@kernel function _k_convert!(y, x)
    i = @index(Global)
    @inbounds y[i] = convert(eltype(y), x[i])
end
"y = x converted to y's element type (elementwise; W7's precision copies)."
pb_convert!(y, x; backend=CPU()) = (length(y) == length(x) || throw(DimensionMismatch("pb_convert!: $(length(y)) and $(length(x)) entries"));
                                    _launch!(_k_convert!, backend, length(y), y, x); y)
"Load the tridiagonal and the Thomas factors of `src` (Float64, loaded by `line_tridiag!`) into its copy `dst`, rounded."
function line_copy!(dst::LineFamily, src::LineFamily; backend=CPU())
    for f in (:ta, :td, :tc, :den, :cp); pb_convert!(getfield(dst, f), getfield(src, f); backend); end
    dst
end

"""
    pcr_lines!(z, sys::FaceSystem, r, mesh, family; backend=CPU()) -> z

Plan 5.5's interface: the workgroup PCR on every line of the family (:j or :i) of `sys` (a Float64 system; built here, for
tests: the solver keeps its families and calls `line_solve_set!`). Lines longer than PCR_NMAX take Thomas.
"""
function pcr_lines!(z, sys::FaceSystem, r, m::PBMesh, family::Symbol; backend=CPU())
    fam = line_tridiag!(line_family(m, family; backend), sys, m; backend)
    alll = _dev(backend, Int32.(1:fam.nlines))
    line_solve_set!(z, z, r, alll, fam, Int32(2); backend, solver=:pcr)
    pb_sync(backend)
    z
end
