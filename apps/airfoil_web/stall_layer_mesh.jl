# Opt-in "stall layer" for the external-flow airfoil meshes (21 Oct 2026). Included in CFDBackend.
# Changes only interior node positions along the existing grid columns; the wall row, the wake cut row,
# the outer boundary and the tangential distribution are the parent's, exactly.

"""Monotone cubic Hermite interpolation (Fritsch-Carlson) of v(s) at the points q (s increasing)."""
function _pchip(s::Vector{Float64}, v::Vector{Float64}, q::Vector{Float64})
    n = length(s); h = diff(s); d = diff(v) ./ h
    m = zeros(n)
    m[1] = d[1]; m[n] = d[n-1]
    for k in 2:n-1
        m[k] = d[k-1] * d[k] <= 0 ? 0.0 :
            (w1 = 2h[k] + h[k-1]; w2 = h[k] + 2h[k-1]; (w1 + w2) / (w1 / d[k-1] + w2 / d[k]))
    end
    out = similar(q)
    for (t, x) in enumerate(q)
        k = clamp(searchsortedlast(s, x), 1, n - 1)
        hk = h[k]; u = (x - s[k]) / hk
        h00 = (1 + 2u) * (1 - u)^2; h10 = u * (1 - u)^2; h01 = u^2 * (3 - 2u); h11 = u^2 * (u - 1)
        out[t] = h00 * v[k] + h10 * hk * m[k] + h01 * v[k+1] + h11 * hk * m[k+1]
    end
    out
end

"""Geometric ratio r continuing from a cell of height a: n cells a r, a r^2, ..., a r^n summing to L
(r = 1 when L <= n a)."""
function _continued_ratio(a::Float64, n::Int, L::Float64)
    L <= n * a * (1 + 1e-12) && return 1.0
    total(r) = a * r * (r^n - 1) / (r - 1)
    lo, hi = 1.0 + 1e-12, 2.0
    while total(hi) < L; hi *= 1.5; end
    for _ in 1:200
        r = 0.5(lo + hi); total(r) < L ? (lo = r) : (hi = r)
    end
    0.5(lo + hi)
end

"""
    _stall_layer_remap(mesh; h_cap=2.5e-3, y_plateau=0.2, n_inner=64, n_outer=64)

Rebuild every grid column with three sections and the same row counts in every column:
an inner section of `n_inner` rows growing geometrically from the column's own first spacing (the parent's, so
the wall spacing and the wake cut's first row are unchanged) to a cell height h_c = max(h_cap, first spacing); a
plateau of `n_p` uniform rows from there to y_p = max(y_plateau, its start + n_p h_c), with n_p set so the foil
columns' plateau cells are about h_cap; an outer section of `n_outer` rows growing geometrically from the
plateau's cell height to the column's end (the parent's outer node). Nodes are placed on each parent column by
monotone cubic interpolation in arc length. Equal row counts keep each grid line a smooth function of the
column's first spacing and length, so the lines do not jump where the first spacing changes (the trailing edge).
Returns (X, Y, NJ, info).
"""
function _stall_layer_remap(X::AbstractMatrix, Y::AbstractMatrix, NIW::Int, NIF::Int;
                            h_cap::Float64=2.5e-3, y_plateau::Float64=0.2, n_inner::Int=64, n_outer::Int=64)
    NI1, NJ1 = size(X)
    s_of(i) = vcat(0.0, cumsum([hypot(X[i, j+1] - X[i, j], Y[i, j+1] - Y[i, j]) for j in 1:NJ1-1]))
    # plateau row count from the foil columns: from the inner section's end to y_plateau at about h_cap
    foil = NIW+1:NIW+NIF+1
    t_in(h1) = h1 >= h_cap ? n_inner * h1 :
        (r = (h_cap / h1)^(1 / n_inner); h1 * (r^n_inner - 1) / (r - 1))
    t_in_foil = maximum(t_in(s_of(i)[2]) for i in foil)
    n_p = max(8, round(Int, (y_plateau - t_in_foil) / h_cap))
    NJn = n_inner + n_p + n_outer
    Xn = zeros(NI1, NJn + 1); Yn = zeros(NI1, NJn + 1)
    rmax_out = 0.0; rmax_in = 0.0
    for i in 1:NI1
        s = s_of(i); S = s[end]; h1 = s[2]
        hc = max(h_cap, h1)
        # inner: geometric from h1 to hc over n_inner rows (ratio 1 when h1 >= h_cap)
        rin = h1 >= h_cap ? 1.0 : (hc / h1)^(1 / (n_inner - 1))
        d = zeros(NJn + 1); h = h1
        for j in 1:n_inner; d[j+1] = d[j] + h; h *= rin; end
        # plateau: uniform to y_p
        yp = max(y_plateau, d[n_inner+1] + n_p * hc)
        hp = (yp - d[n_inner+1]) / n_p
        for j in n_inner+1:n_inner+n_p; d[j+1] = d[j] + hp; end
        # outer: geometric from hp to S
        L = S - d[n_inner+n_p+1]
        L > n_outer * hp || error("stall layer: column $i is too short for the plateau (length $S)")
        rout = _continued_ratio(hp, n_outer, L)
        hh = hp * rout
        for j in n_inner+n_p+1:NJn; d[j+1] = d[j] + hh; hh *= rout; end
        d[end] = S   # exact end
        rmax_out = max(rmax_out, rout); rmax_in = max(rmax_in, rin)
        xs = _pchip(s, Float64.(X[i, :]), d); ys = _pchip(s, Float64.(Y[i, :]), d)
        xs[1] = X[i, 1]; ys[1] = Y[i, 1]; xs[end] = X[i, end]; ys[end] = Y[i, end]
        Xn[i, :] = xs; Yn[i, :] = ys
    end
    (X=Xn, Y=Yn, NJ=NJn, info=Dict{String,Any}("h_cap"=>h_cap, "y_plateau"=>y_plateau, "n_inner"=>n_inner,
        "n_plateau"=>n_p, "n_outer"=>n_outer, "rows"=>NJn, "max_inner_ratio"=>rmax_in, "max_outer_ratio"=>rmax_out,
        "version"=>"stall_layer_v1",
        "scope"=>"interior nodes moved along the parent's grid columns; wall row, wake-cut row, outer boundary and tangential distribution unchanged"))
end

"""
    _stall_layer_remap_common(X, Y, NIW, NIF; h_cap=2.5e-3, y_plateau=0.2, n_inner=64, n_outer=64, n_keep=12)

The stall layer with one distance distribution for every column (stall_layer_v2, for the C+H mesh, 21 Oct 2026).
The reference D grows geometrically over `n_inner` rows from the foil's median first spacing to h_cap; each column
keeps the parent's own rows exactly up to row `n_keep` (the generator's frozen boundary-layer rows: the wall spacing,
the interface rows and their first spacing are the parent's), then blends to D with a smoothstep weight that reaches
D at row `n_inner`; the plateau (uniform rows to y_plateau) is then the same in every column, and the outer section
grows geometrically to the column's end. Version 1 grows each column from its own first spacing, so where that
spacing changes over a few narrow columns (the C+H near wake: 6e-6 c at the corner, 3e-5 c along the interface, in
columns 2.5e-5 c wide) its rows step between neighbours and shear the cells; here the rows away from the wall do not
depend on the column. Returns (X, Y, NJ, info).
"""
function _stall_layer_remap_common(X::AbstractMatrix, Y::AbstractMatrix, NIW::Int, NIF::Int;
                                   h_cap::Float64=2.5e-3, y_plateau::Float64=0.2, n_inner::Int=64,
                                   n_outer::Int=64, n_keep::Int=12)
    NI1, NJ1 = size(X)
    2 <= n_keep < n_inner < NJ1 || error("stall layer: need 2 <= n_keep < n_inner < the parent's rows")
    s_of(i) = vcat(0.0, cumsum([hypot(X[i, j+1] - X[i, j], Y[i, j+1] - Y[i, j]) for j in 1:NJ1-1]))
    foil = NIW+1:NIW+NIF+1
    h1 = sort([s_of(i)[2] for i in foil]); h1ref = h1[cld(length(h1), 2)]
    rin = h1ref >= h_cap ? 1.0 : (h_cap / h1ref)^(1 / (n_inner - 1))
    D = zeros(n_inner + 1); h = h1ref
    for j in 1:n_inner; D[j+1] = D[j] + h; h *= rin; end
    n_p = max(8, round(Int, (y_plateau - D[end]) / h_cap))
    yp = max(y_plateau, D[end] + n_p * h_cap); hp = (yp - D[end]) / n_p
    NJn = n_inner + n_p + n_outer
    Xn = zeros(NI1, NJn + 1); Yn = zeros(NI1, NJn + 1)
    rmax_out = 0.0; minstep = Inf
    for i in 1:NI1
        s = s_of(i); S = s[end]
        d = zeros(NJn + 1)
        for j in 0:n_inner
            t = clamp((j - n_keep) / (n_inner - n_keep), 0.0, 1.0)
            w = 1 - t * t * (3 - 2t)
            d[j+1] = w * s[j+1] + (1 - w) * D[j+1]
        end
        for j in n_inner+1:n_inner+n_p; d[j+1] = d[j] + hp; end
        L = S - d[n_inner+n_p+1]
        L > n_outer * hp || error("stall layer: column $i is too short for the plateau (length $S)")
        rout = _continued_ratio(hp, n_outer, L); hh = hp * rout
        for j in n_inner+n_p+1:NJn; d[j+1] = d[j] + hh; hh *= rout; end
        d[end] = S
        steps = diff(d); minimum(steps) > 0 || error("stall layer: column $i is not monotone (row $(argmin(steps)))")
        minstep = min(minstep, minimum(steps[n_keep+1:end] ./ steps[n_keep:end-1]))
        rmax_out = max(rmax_out, rout)
        xs = _pchip(s, Float64.(X[i, :]), d); ys = _pchip(s, Float64.(Y[i, :]), d)
        # the kept rows are the parent's nodes exactly (the cubic is evaluated at its own knots)
        for j in 1:n_keep+1; xs[j] = X[i, j]; ys[j] = Y[i, j]; end
        xs[end] = X[i, end]; ys[end] = Y[i, end]
        Xn[i, :] = xs; Yn[i, :] = ys
    end
    (X=Xn, Y=Yn, NJ=NJn, info=Dict{String,Any}("h_cap"=>h_cap, "y_plateau"=>y_plateau, "n_inner"=>n_inner,
        "n_keep"=>n_keep, "n_plateau"=>n_p, "n_outer"=>n_outer, "rows"=>NJn, "reference_first_spacing"=>h1ref,
        "inner_ratio"=>rin, "max_outer_ratio"=>rmax_out, "min_blend_height_ratio"=>minstep,
        "version"=>"stall_layer_v2_common",
        "scope"=>"interior nodes moved along the parent's grid columns; rows 0..n_keep, the outer boundary and the tangential distribution unchanged"))
end

"Apply the stall layer to a ChannelMesh, returning a ChannelMesh with the new rows (`common=true`: version 2)."
function _stall_layer_mesh(mesh; common::Bool=false, kwargs...)
    r = common ? _stall_layer_remap_common(mesh.X, mesh.Y, mesh.NIW, mesh.NIF; kwargs...) :
        _stall_layer_remap(mesh.X, mesh.Y, mesh.NIW, mesh.NIF; kwargs...)
    m = CMeshGen.ChannelMesh(mesh.NI, r.NJ, mesh.NIW, mesh.NIF, r.X, r.Y, mesh.dn1,
        fill(r.info["max_outer_ratio"], length(mesh.growth)), mesh.channel, mesh.outer_face_kind, mesh.strip)
    (mesh=m, info=r.info)
end
