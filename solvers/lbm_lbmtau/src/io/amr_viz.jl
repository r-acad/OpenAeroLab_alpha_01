using Printf

"""
    export_amr_viz(grids, mesh_offset, output_dir; mesh=nothing, case_name="")

Export the AMR block structure, Bouzidi boundary cells, and (optionally) the STL
surface mesh of every level into a SINGLE JSON file
(`<output_dir>/amr_viz/amr_data.json`) so the standalone Babylon.js viewer at
`PREPOST/amr_viewer.html` can open it via its file picker.

The file layout:

    {
      "format_version": 3,
      "case_name": "...",
      "n_levels": N,
      "mesh_offset": [x,y,z],
      "domain_nx/ny/nz": ...,
      "domain_size": [Lx,Ly,Lz],
      "block_size": 8,
      "geometry": {
        "n_triangles": T,
        "vertices": [x1,y1,z1, x2,y2,z2, ...],   # 3*T*3 floats (flat)
        "min": [xmin,ymin,zmin],
        "max": [xmax,ymax,zmax]
      },
      "levels": [
        {
          "level_id": 1,
          "dx": ...,
          "block_phys_size": ...,
          "n_blocks": K,
          "blocks": [bx1,by1,bz1, bx2,by2,bz2, ...],
          "n_bouzidi_cells": M,
          "bouzidi_cells": [gx1,gy1,gz1, gx2,gy2,gz2, ...]
        },
        ...
      ]
    }

Bouzidi cells are stored as flat global cell indices
`global = (block_idx - 1) * BLOCK_SIZE + local_idx`. Cell physical size = dx
(not bs*dx). The viewer renders them as cell-sized cubes on the surface.

Geometry vertices are stored in DOMAIN coordinates (mesh_offset already added)
so they line up with the AMR blocks and Bouzidi cells without further offsets.
"""
function export_amr_viz(grids, mesh_offset, output_dir::String;
                        mesh=nothing, case_name::String="")
    viz_dir = joinpath(output_dir, "amr_viz")
    isdir(viz_dir) || mkpath(viz_dir)

    # Clean up obsolete files from the old multi-file format.
    for stale in ("summary.json",)
        p = joinpath(viz_dir, stale)
        isfile(p) && rm(p; force=true)
    end
    for f in readdir(viz_dir)
        if startswith(f, "level_") && endswith(f, ".json")
            rm(joinpath(viz_dir, f); force=true)
        end
    end

    n_levels = length(grids)
    mx, my, mz = Float64(mesh_offset[1]), Float64(mesh_offset[2]), Float64(mesh_offset[3])
    bs = BLOCK_SIZE

    fname = joinpath(viz_dir, "amr_data.json")
    open(fname, "w") do io
        print(io, "{")
        @printf(io, "\"format_version\":3,")
        # Escape any backslashes/quotes in case_name (Windows paths sometimes carry them).
        safe_case = replace(replace(case_name, "\\" => "\\\\"), "\"" => "\\\"")
        @printf(io, "\"case_name\":\"%s\",", safe_case)
        @printf(io, "\"n_levels\":%d,", n_levels)
        @printf(io, "\"mesh_offset\":[%.10e,%.10e,%.10e],", mx, my, mz)

        # Domain extent from level 1
        l1 = grids[1]
        dx1 = Float64(l1.dx)
        @printf(io, "\"domain_nx\":%d,", l1.grid_dim_x * bs)
        @printf(io, "\"domain_ny\":%d,", l1.grid_dim_y * bs)
        @printf(io, "\"domain_nz\":%d,", l1.grid_dim_z * bs)
        @printf(io, "\"domain_size\":[%.10e,%.10e,%.10e],",
                l1.grid_dim_x * bs * dx1, l1.grid_dim_y * bs * dx1, l1.grid_dim_z * bs * dx1)
        @printf(io, "\"block_size\":%d,", bs)

        # STL geometry (optional). Vertices are placed into domain coords by
        # adding mesh_offset, so the viewer can render them in the same frame
        # as the AMR blocks and Bouzidi cells.
        if mesh !== nothing && isdefined(mesh, :triangles) && !isempty(mesh.triangles)
            n_tri = length(mesh.triangles)
            xmin =  Inf; ymin =  Inf; zmin =  Inf
            xmax = -Inf; ymax = -Inf; zmax = -Inf
            print(io, "\"geometry\":{")
            @printf(io, "\"n_triangles\":%d,", n_tri)
            print(io, "\"vertices\":[")
            for (ti, tri) in enumerate(mesh.triangles)
                ti > 1 && print(io, ",")
                # tri = ((x1,y1,z1),(x2,y2,z2),(x3,y3,z3))
                for vi in 1:3
                    v = tri[vi]
                    x = Float64(v[1]) + mx
                    y = Float64(v[2]) + my
                    z = Float64(v[3]) + mz
                    xmin = min(xmin, x); ymin = min(ymin, y); zmin = min(zmin, z)
                    xmax = max(xmax, x); ymax = max(ymax, y); zmax = max(zmax, z)
                    vi > 1 && print(io, ",")
                    @printf(io, "%.6e,%.6e,%.6e", x, y, z)
                end
            end
            print(io, "],")
            @printf(io, "\"min\":[%.6e,%.6e,%.6e],", xmin, ymin, zmin)
            @printf(io, "\"max\":[%.6e,%.6e,%.6e]", xmax, ymax, zmax)
            print(io, "},")
            @printf("[AMR Viz] Geometry: %d triangles, bounds=[%.3f,%.3f,%.3f]→[%.3f,%.3f,%.3f]\n",
                    n_tri, xmin, ymin, zmin, xmax, ymax, zmax)
        else
            print(io, "\"geometry\":null,")
        end

        print(io, "\"levels\":[")
        for (lvl_idx, level) in enumerate(grids)
            coords = level.active_block_coords
            n_blocks = length(coords)
            dx = Float64(level.dx)
            bs_phys = bs * dx
            n_bouzidi = (level.bouzidi_enabled ? Int(level.n_boundary_cells) : 0)

            lvl_idx > 1 && print(io, ",")
            print(io, "{")
            @printf(io, "\"level_id\":%d,", lvl_idx)
            @printf(io, "\"dx\":%.10e,", dx)
            @printf(io, "\"block_phys_size\":%.10e,", bs_phys)
            @printf(io, "\"n_blocks\":%d,", n_blocks)
            print(io, "\"blocks\":[")
            for (i, (bx, by, bz)) in enumerate(coords)
                i > 1 && print(io, ",")
                @printf(io, "%d,%d,%d", bx, by, bz)
            end
            print(io, "]")
            @printf(io, ",\"n_bouzidi_cells\":%d", n_bouzidi)
            print(io, ",\"bouzidi_cells\":[")
            if n_bouzidi > 0
                cb = level.bouzidi_cell_block
                cxv = level.bouzidi_cell_x
                cyv = level.bouzidi_cell_y
                czv = level.bouzidi_cell_z
                for i in 1:n_bouzidi
                    b_idx = Int(cb[i])
                    (bx, by, bz) = coords[b_idx]
                    gx = (bx - 1) * bs + Int(cxv[i])
                    gy = (by - 1) * bs + Int(cyv[i])
                    gz = (bz - 1) * bs + Int(czv[i])
                    i > 1 && print(io, ",")
                    @printf(io, "%d,%d,%d", gx, gy, gz)
                end
            end
            print(io, "]")
            print(io, "}")
            @printf("[AMR Viz] Level %d: %d blocks, %d bouzidi cells, dx=%.4e m\n",
                    lvl_idx, n_blocks, n_bouzidi, dx)
        end
        print(io, "]")
        print(io, "}")
    end
    @printf("[AMR Viz] Wrote %s — open PREPOST/amr_viewer.html and load this file\n", fname)
end
