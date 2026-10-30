# Regional mesh-quality audit of a C+H mesh around its finite base (23 Oct 2026, Gurney flap work).
# Included inside CFDBackend after cmesh_gen.jl. Geometry only: it reads node arrays and never changes them.
#
# The global CMeshQualityV1 gates (quality_summary / quality_gates) decide admission. This audit reports, for the
# Gurney flap's base, where the worst cells are: the C block near either base corner and the H block behind the base.
# Metrics (the gates' own definitions, CMeshQualityMetrics): aspect ratio ar_p (principal lengths), face skew/d,
# face non-orthogonality theta (true centroids), the smallest cell corner angle and the largest neighbour volume ratio
# (the growth from one cell to the next across a face).

"""    gurney_regional_audit(X, Y, NIW, NIF, strip; radius=nothing) -> Dict

`radius` (chords) defaults to max(3 base heights, 0.02 c): C cells whose true centroid lies within it of either base
corner form the corner regions. Every strip cell forms the H region."""
function gurney_regional_audit(X::AbstractMatrix, Y::AbstractMatrix, NIW::Int, NIF::Int, strip; radius=nothing)
    Q = CMeshGen.CMeshQualityMetrics
    blocks = [(X, Y), (strip.X, strip.Y)]
    cells = [Q.block_cells(b...) for b in blocks]
    faces, _ = Q.mesh_faces(blocks, cells; NIW=NIW, NIF=NIF, strip=true)
    lower = (X[NIW+1, 1], Y[NIW+1, 1]); upper = (X[NIW+NIF+1, 1], Y[NIW+NIF+1, 1])
    base = hypot(upper[1] - lower[1], upper[2] - lower[2])
    r = radius === nothing ? max(3base, 0.02) : Float64(radius)
    C = cells[1]
    near(c, i, j) = hypot(C.xc[i, j] - c[1], C.yc[i, j] - c[2]) <= r
    mid = NIW + NIF ÷ 2                      # the C block's lower half (lower wake leg and lower surface) ends here
    inregion = Dict{String,Function}(
        "lower_corner_C"=>(b, i, j) -> b == 1 && i <= mid && near(lower, i, j),
        "upper_corner_C"=>(b, i, j) -> b == 1 && i > mid && near(upper, i, j),
        "H_block"=>(b, i, j) -> b == 2,
        "C_H_interfaces"=>(b, i, j) -> false)
    out = Dict{String,Any}("radius"=>r, "base_height"=>base,
        "scope"=>"worst values per region: C cells within the radius of each base corner, every H cell, and the two C/H interfaces; metrics as CMeshQualityMetrics (ar_p, face skew/d, face non-orthogonality, minimum corner angle, neighbour volume ratio)")
    for name in ("lower_corner_C", "upper_corner_C", "H_block", "C_H_interfaces")
        f = inregion[name]
        ar = 0.0; ang = 180.0; arloc = nothing; angloc = nothing; ncell = 0
        for (b, cb) in enumerate(cells), j in axes(cb.area, 2), i in axes(cb.area, 1)
            f(b, i, j) || continue
            ncell += 1
            cb.ar_p[i, j] > ar && (ar = cb.ar_p[i, j]; arloc = (b, i, j, cb.xc[i, j], cb.yc[i, j]))
            cb.minang[i, j] < ang && (ang = cb.minang[i, j]; angloc = (b, i, j, cb.xc[i, j], cb.yc[i, j]))
        end
        th = 0.0; sk = 0.0; vr = 1.0; thloc = nothing; skloc = nothing; nface = 0
        for fc in faces
            hit = name == "C_H_interfaces" ? fc.kind in (:ifl, :ifu) : (f(fc.Pb, fc.Pi, fc.Pj) || f(fc.Nb, fc.Ni, fc.Nj))
            hit || continue
            nface += 1
            fc.theta > th && (th = fc.theta; thloc = (fc.kind, fc.xf, fc.yf))
            fc.skew > sk && (sk = fc.skew; skloc = (fc.kind, fc.xf, fc.yf)); vr = max(vr, fc.vr)
        end
        if name == "C_H_interfaces"
            for fc in faces
                fc.kind in (:ifl, :ifu) || continue
                for (b, i, j) in ((fc.Pb, fc.Pi, fc.Pj), (fc.Nb, fc.Ni, fc.Nj))
                    cb = cells[b]; ncell += 1
                    cb.ar_p[i, j] > ar && (ar = cb.ar_p[i, j]; arloc = (b, i, j, cb.xc[i, j], cb.yc[i, j]))
                    cb.minang[i, j] < ang && (ang = cb.minang[i, j]; angloc = (b, i, j, cb.xc[i, j], cb.yc[i, j]))
                end
            end
        end
        out[name] = Dict{String,Any}("cells"=>ncell, "faces"=>nface, "max_aspect_ratio"=>ar,
            "max_skew_over_distance"=>sk, "max_nonorthogonality_deg"=>th, "min_corner_angle_deg"=>ang,
            "max_neighbour_volume_ratio"=>vr,
            "worst_aspect_ratio_at"=>arloc === nothing ? "none" : Dict("block"=>arloc[1] == 1 ? "C" : "H", "i"=>arloc[2], "j"=>arloc[3], "x"=>arloc[4], "y"=>arloc[5]),
            "worst_angle_at"=>angloc === nothing ? "none" : Dict("block"=>angloc[1] == 1 ? "C" : "H", "i"=>angloc[2], "j"=>angloc[3], "x"=>angloc[4], "y"=>angloc[5]),
            "worst_nonorthogonality_at"=>thloc === nothing ? "none" : Dict("face"=>String(thloc[1]), "x"=>thloc[2], "y"=>thloc[3]),
            "worst_skew_at"=>skloc === nothing ? "none" : Dict("face"=>String(skloc[1]), "x"=>skloc[2], "y"=>skloc[3]))
    end
    # where are the global extremes (the gates' theta_max and minang)?
    thg = argmax(fc -> fc.theta, faces)
    out["global_worst_nonorthogonality"] = Dict{String,Any}("deg"=>thg.theta, "face"=>String(thg.kind), "x"=>thg.xf, "y"=>thg.yf,
        "owner"=>Dict("block"=>thg.Pb == 1 ? "C" : "H", "i"=>thg.Pi, "j"=>thg.Pj))
    best = (180.0, 1, 1, 1)
    for (b, cb) in enumerate(cells), j in axes(cb.area, 2), i in axes(cb.area, 1)
        cb.minang[i, j] < best[1] && (best = (cb.minang[i, j], b, i, j))
    end
    cb = cells[best[2]]
    out["global_min_corner_angle"] = Dict{String,Any}("deg"=>best[1], "block"=>best[2] == 1 ? "C" : "H", "i"=>best[3], "j"=>best[4],
        "x"=>cb.xc[best[3], best[4]], "y"=>cb.yc[best[3], best[4]])
    out
end
gurney_regional_audit(m; kw...) = gurney_regional_audit(m.X, m.Y, m.NIW, m.NIF, m.strip; kw...)
