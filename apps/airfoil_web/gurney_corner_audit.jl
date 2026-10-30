# RETIRED 28 Oct 2026: the plate-INSERTION mesh (the reshaped-mesh Gurney flap) is no longer used by the app. The Gurney flap is a
# baffle in the section's own C+H mesh (PBCav PB_BAFFLE; lowspeed_ch_backend.jl). This file is not included by cfd_backend.jl any
# more; it stays here only so that the archived validation scripts under validation/gurney_demo_20261023 and
# validation/gurney_smooth_20261024 still load it. Do not use it for new work.
# The Gurney plate's corner cells (23 Oct 2026): the cells that touch each tip and each root, and the worst cell near
# each, measured with version 1's CMeshQualityMetrics (passed in as `Q`, so this file needs no include of its own):
# smallest corner angle, aspect ratio ar_p, largest face skew/d and largest face non-orthogonality theta over the cell's
# faces. Geometry only; nothing is changed.

"""    gurney_corner_table(Q, X, Y, NIW, NIF, Xs, Ys; corners, radius) -> Vector{Dict}

`corners`: (name, block-C wall index of the corner node, h_side) for each tip and root. For each corner: the C cells
sharing the corner node in the first row, the H cell at a tip (lower tip: H (1,1); upper tip: H (1,NS)), and the worst
values over every cell (C and H) whose centroid lies within `radius` (default 0.25 h_side) of the corner."""
function gurney_corner_table(Q, X, Y, NIW::Int, NIF::Int, Xs, Ys; corners, radius=nothing)
    blocks = [(X, Y), (Xs, Ys)]
    cells = [Q.block_cells(b...) for b in blocks]
    faces, _ = Q.mesh_faces(blocks, cells; NIW=NIW, NIF=NIF, strip=true)
    cellfaces = Dict{Tuple{Int,Int,Int},Vector{Int}}()
    for (k, f) in enumerate(faces)
        push!(get!(cellfaces, (f.Pb, f.Pi, f.Pj), Int[]), k)
        f.Nb > 0 && push!(get!(cellfaces, (f.Nb, f.Ni, f.Nj), Int[]), k)
    end
    function cellrec(b, i, j)
        cb = cells[b]; fk = get(cellfaces, (b, i, j), Int[])
        Dict{String,Any}("block"=>b == 1 ? "C" : "H", "i"=>i, "j"=>j,
            "min_angle_deg"=>cb.minang[i, j], "aspect_ratio"=>cb.ar_p[i, j],
            "max_skew_over_d"=>isempty(fk) ? 0.0 : maximum(faces[k].skew for k in fk),
            "max_nonorthogonality_deg"=>isempty(fk) ? 0.0 : maximum(faces[k].theta for k in fk))
    end
    NS = size(Xs, 2) - 1
    out = Dict{String,Any}[]
    for (name, ic, hside) in corners
        r = radius === nothing ? 0.25hside : Float64(radius)
        cx, cy = X[ic, 1], Y[ic, 1]
        touching = [cellrec(1, ic - 1, 1), cellrec(1, ic, 1)]
        occursin("lower tip", name) && push!(touching, cellrec(2, 1, 1))
        occursin("upper tip", name) && push!(touching, cellrec(2, 1, NS))
        worst = Dict{String,Any}("min_angle_deg"=>180.0, "aspect_ratio"=>0.0, "max_skew_over_d"=>0.0, "max_nonorthogonality_deg"=>0.0, "cells"=>0)
        for (b, cb) in enumerate(cells), j in axes(cb.area, 2), i in axes(cb.area, 1)
            hypot(cb.xc[i, j] - cx, cb.yc[i, j] - cy) <= r || continue
            c = cellrec(b, i, j); worst["cells"] += 1
            worst["min_angle_deg"] = min(worst["min_angle_deg"], c["min_angle_deg"])
            worst["aspect_ratio"] = max(worst["aspect_ratio"], c["aspect_ratio"])
            worst["max_skew_over_d"] = max(worst["max_skew_over_d"], c["max_skew_over_d"])
            worst["max_nonorthogonality_deg"] = max(worst["max_nonorthogonality_deg"], c["max_nonorthogonality_deg"])
        end
        push!(out, Dict{String,Any}("corner"=>name, "x"=>cx, "y"=>cy, "radius"=>r, "touching_cells"=>touching, "worst_within_radius"=>worst))
    end
    out
end

"The corners of a plate mesh from gurney_plate_mesh's records: (name, C wall index, h of that side)."
function gurney_corner_list(r)
    c = Tuple{String,Int,Float64}[]
    r.lower === nothing || (push!(c, ("lower tip", r.lower["tip_index"], r.lower["plate_height"])); push!(c, ("lower root", r.lower["root_index"], r.lower["plate_height"])))
    r.upper === nothing || (push!(c, ("upper tip", r.upper["tip_index"], r.upper["plate_height"])); push!(c, ("upper root", r.upper["root_index"], r.upper["plate_height"])))
    c
end
