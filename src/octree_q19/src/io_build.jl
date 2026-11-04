# Build-only outputs: the files the app reads after "Create mesh" (plan section 3.10, DESIGN.md section 4 step 10):
# mesh.oct, RESULTS/build_only_summary.toml (the app's keys plus an `octree` table), RESULTS/amr_viz/amr_data.json
# (format 4: leaf boxes per level), RESULTS/SLICES/step_000000/plane_<tag>_000000.lbmslice (level and obstacle),
# RESULTS/provenance.json.
include(joinpath(@__DIR__, "..", "..", "io", "lbm_slice_format.jl"))   # the shared slice format, src/io/lbm_slice_format.jl
using .LBMSliceFormat

const SOLVER_NAME = "LUDWIG D3Q19 CPU octree (src/octree_q19)"
const AMR_VIZ_FORMAT = 4

"Bytes of a stored slot at the run's precision: two population buffers, two velocity buffers, kind, cut index, its share of the oct table."
bytes_per_slot(::Type{T}) where T = (2 * Q + 8) * sizeof(T) + 1 + 4 + 27 * 4 / 8

function plane_sample_limit(cfg)
    n=_get(cfg.raw,"octree_q19","outputs","max_plane_samples";default=4_000_000)
    n isa Integer && 10_000<=n<=16_000_000 || error("max_plane_samples must be an integer in 10000..16000000")
    Int(n)
end

"Conservative run allowance: mesh storage, state/coupling buffers, thread forces and bounded export work."
function estimated_run_bytes(cfg,mesh)
    T=cfg.precision;bytes=Float64(Base.summarysize(mesh))
    for (l,Lv) in enumerate(mesh.levels)
        slots=8Lv.nocts
        bytes+=slots*(2Q+8)*sizeof(T)+4Lv.nocts+length(Lv.cut_slot)*(5Q+sizeof(T))
        for (table,donor_slots) in ((Lv.ghost_donor,l>1 ? 8mesh.levels[l-1].nocts : 0),
                                   (Lv.restrict_src,l<length(mesh.levels) ? 8mesh.levels[l+1].nocts : 0))
            bytes+=4length(table)+min(length(table),donor_slots)*(4+10sizeof(T))
        end
    end
    bytes+=24max(1,length(mesh.triangles))*(Threads.nthreads()+1)
    # Native planes are sampled sequentially; VTK builds point/connectivity arrays.
    exports=60plane_sample_limit(cfg)+(cfg.outputs.volume_vtk ? 512mesh.counts["fluid_total"] : 0)
    ceil(Int,1.25bytes+exports+512_000_000)
end

"The placed triangles of a case: STL scaled, rotated by alpha (about y) and beta (about z) about the STL bounds centre as the existing solver does, then translated by the offset."
function placed_triangles(cfg::CaseConfig)
    tri, dropped = read_stl(cfg.stl_path)
    s = cfg.stl_scale
    tri = [(vmul(t[1], s), vmul(t[2], s), vmul(t[3], s)) for t in tri]
    # a ground case whose STL is z-up is turned to the solver's y-up frame as src/geometry/geometry.jl does: (x, y, z) -> (x, z, -y)
    if cfg.mesh.faces[3] == :ground && lowercase(String(something(_get(cfg.raw, "basic", "ground_up_axis"), "y"))) == "z"
        tri = [((t[1][1], t[1][3], -t[1][2]), (t[2][1], t[2][3], -t[2][2]), (t[3][1], t[3][3], -t[3][2])) for t in tri]
    end
    if cfg.flow.alpha_deg != 0 || cfg.flow.beta_deg != 0
        lo, hi = bounds(tri); c = vmul(vadd(lo, hi), 0.5)
        tri = [(rotate_alpha_beta(vsub(t[1], c), cfg.flow.alpha_deg, cfg.flow.beta_deg), rotate_alpha_beta(vsub(t[2], c), cfg.flow.alpha_deg, cfg.flow.beta_deg),
                rotate_alpha_beta(vsub(t[3], c), cfg.flow.alpha_deg, cfg.flow.beta_deg)) for t in tri]
        tri = [(vadd(t[1], c), vadd(t[2], c), vadd(t[3], c)) for t in tri]
    end
    tri = [(vadd(t[1], cfg.offset), vadd(t[2], cfg.offset), vadd(t[3], cfg.offset)) for t in tri]
    tri, dropped
end

"Rotation of a point by alpha degrees about +y (positive alpha raises the trailing edge: nose down about y) then beta about +z, as src/geometry/geometry.jl."
@inline function rotate_alpha_beta(v::V3, alpha_deg::Float64, beta_deg::Float64)
    a = deg2rad(alpha_deg); b = deg2rad(beta_deg)
    ca, sa = cos(a), sin(a); cb, sb = cos(b), sin(b)
    x1 = ca * v[1] + sa * v[3]; y1 = v[2]; z1 = -sa * v[1] + ca * v[3]
    (cb * x1 - sb * y1, sb * x1 + cb * y1, z1)
end

"Run the build-only path of a case: mesh and the app's files. Returns the mesh."
function build_only_case(cfg::CaseConfig)
    t0 = time()
    out = joinpath(cfg.dir, "RESULTS"); mkpath(out)
    println("[BuildOnly] $(SOLVER_NAME): $(cfg.name), $(cfg.mesh.levels) levels, finest dx $(cfg.mesh.dx_finest) m, threads $(Threads.nthreads())")
    tri, dropped = placed_triangles(cfg)
    println("[BuildOnly] geometry: $(length(tri)) triangles ($dropped degenerate dropped), offset $(round.(cfg.offset, digits=6)) m")
    flush(stdout)
    t1 = time()
    mesh = generate(tri, cfg.mesh; offset=cfg.offset, config_sha=cfg.config_sha, stl_sha=cfg.stl_sha)
    build_s = time() - t1
    for p in mesh.counts["per_level"]
        println(@sprintf("[BuildOnly] L%d: dx=%.6g m, %d octs (%d real), fluid %d, solid %d, covered %d, ghost %d, wall cells %d, cut links %d",
                p["level"], p["dx_m"], p["octs"], p["real_octs"], p["fluid"], p["solid"], p["covered"], p["ghost"], p["wall_cells"], p["cut_links"]))
    end
    rep = mesh.counts["reports"]
    println(@sprintf("[BuildOnly] mesh built in %.1f s: %d fluid cells, W %.4g; leak cells %d, unreached surface area %.2f %%, missing-partner links %d",
            build_s, mesh.counts["fluid_total"], mesh.counts["work_W"], rep["leak_cells"], 100 * rep["unreached_triangle_area_fraction"], rep["missing_partner_links"]))
    flush(stdout)
    sha = write_mesh(joinpath(cfg.dir, "mesh.oct"), mesh)
    println("[BuildOnly] mesh.oct written, sha256 $(sha[1:16])")
    scaling = lattice_scaling(cfg)
    write_amr_viz(out, cfg, mesh)
    println("[AMR Viz] Wrote $(joinpath(out, "amr_viz", "amr_data.json"))")
    plane_files = export_planes(mesh, cfg, scaling, 0, joinpath(out, "SLICES"); with_fields=false)
    write_provenance(out, cfg, sha)
    write_build_summary(joinpath(out, "build_only_summary.toml"), cfg, mesh, scaling, build_s, plane_files, sha)
    println(@sprintf("[BuildOnly] done in %.1f s; summary: %s", time() - t0, joinpath(out, "build_only_summary.toml")))
    flush(stdout)
    mesh
end

function write_build_summary(path, cfg::CaseConfig, mesh::OctMesh, sc::LatticeScaling, build_s, plane_files, mesh_sha)
    per = mesh.counts["per_level"]; L = nlevels(mesh)
    bps = bytes_per_slot(cfg.precision)
    levels = Dict{String,Any}[]
    total_cells = 0; total_blocks = 0; total_bytes = 0.0
    for p in per
        cells = p["fluid"] + p["solid"]; blocks = p["real_octs"]; bytes = round(Int, 8 * p["octs"] * bps)
        total_cells += cells; total_blocks += blocks; total_bytes += bytes
        push!(levels, Dict{String,Any}("level" => p["level"], "dx_m" => p["dx_m"], "tau" => sc.tau[p["level"]],
            "active_blocks" => blocks, "cells" => cells, "fluid_cells" => p["fluid"], "obstacle_cells" => p["solid"],
            "bouzidi_boundary_cells" => p["wall_cells"], "bytes_allocated" => bytes,
            "octs" => p["octs"], "ghost_cells" => p["ghost"], "covered_cells" => p["covered"], "cut_links" => p["cut_links"], "restricted_cells" => p["restricted"]))
    end
    root = mesh.counts["root_cells"]
    steps = cfg.outputs.steps
    summary = Dict{String,Any}(
        "case" => cfg.name, "build_only" => true, "solver" => SOLVER_NAME, "mesh" => "octree",
        "written_utc" => Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS") * "Z",
        "domain_build_seconds" => build_s,
        "lattice" => "d3q19", "collision_operator" => String(cfg.numerics.collision),
        "num_levels" => L,
        "coarse_grid_cells" => [root[1], root[2], root[3]],
        "domain_size_m" => collect(mesh.domain), "mesh_offset_m" => collect(mesh.offset),
        "reynolds" => sc.reynolds, "u_physical_m_s" => cfg.flow.velocity, "u_lattice_target" => sc.u_lattice,
        "time_scale_s" => sc.dt_finest, "time_per_step_s" => sc.dt_finest * 2.0^(L - 1), "velocity_scale_m_s" => sc.velocity_scale,
        "steps_configured" => steps,
        "total_blocks" => total_blocks, "total_cells" => total_cells, "bytes_allocated_total" => round(Int, total_bytes),
        "estimated_run_bytes_total" => estimated_run_bytes(cfg,mesh),
        "memory_note" => "bytes_allocated is the basic slot allocation only. estimated_run_bytes_total includes mesh, density/population/velocity buffers, conservative coupling buffers, thread forces, bounded native planes, optional volume VTK work, 25% reserve and 512 MB overhead. RAM, no GPU.",
        "levels" => levels,
        "plane_files" => [replace(relpath(f, dirname(path)), "\\" => "/") for f in plane_files],
        "geometry_triangles" => length(mesh.triangles),
        "octree" => Dict{String,Any}("mesh_file" => "mesh.oct", "mesh_sha256" => mesh_sha, "fluid_cells_total" => mesh.counts["fluid_total"],
            "stored_slots" => mesh.counts["stored_slots"], "work_W" => mesh.counts["work_W"], "band_fine" => cfg.mesh.band_fine, "band" => cfg.mesh.band,
            "surface_level" => cfg.mesh.surface_level, "seal_cavities" => cfg.mesh.seal_cavities, "wake_length" => cfg.mesh.wake_length, "wake_level" => mesh.counts["wake_level"],
            "periodic" => collect(mesh.periodic), "faces" => [String(f) for f in cfg.mesh.faces], "precision" => string(cfg.precision),
            "reports" => mesh.counts["reports"], "timings_s" => mesh.counts["timings_s"], "distance_queries" => mesh.counts["distance_queries"]),
    )
    tmp = path * ".tmp"
    open(tmp, "w") do io; TOML.print(io, summary) end
    mv(tmp, path; force=true)
    summary
end

"Fluid leaves of a level merged into axis-aligned boxes (0-based inclusive cell ranges): runs along x, then y, then z."
function leaf_boxes(L::OctLevel)
    cells = Tuple{Int,Int,Int}[]
    for o in 1:L.nocts
        L.real[o] || continue
        for k in 1:8
            L.kind[k, o] == K_FLUID && push!(cells, slot_cell(L, slot_id(k, o)))
        end
    end
    sort!(cells; by=c -> (c[3], c[2], c[1]))
    segs = NTuple{4,Int}[]          # (i0, i1, j, k)
    i = 1
    while i <= length(cells)
        j = i
        while j < length(cells) && cells[j + 1][2] == cells[i][2] && cells[j + 1][3] == cells[i][3] && cells[j + 1][1] == cells[j][1] + 1
            j += 1
        end
        push!(segs, (cells[i][1], cells[j][1], cells[i][2], cells[i][3])); i = j + 1
    end
    sort!(segs; by=s -> (s[4], s[1], s[2], s[3]))
    strips = NTuple{5,Int}[]        # (i0, i1, j0, j1, k)
    i = 1
    while i <= length(segs)
        j = i
        while j < length(segs) && segs[j + 1][4] == segs[i][4] && segs[j + 1][1] == segs[i][1] && segs[j + 1][2] == segs[i][2] && segs[j + 1][3] == segs[j][3] + 1
            j += 1
        end
        push!(strips, (segs[i][1], segs[i][2], segs[i][3], segs[j][3], segs[i][4])); i = j + 1
    end
    sort!(strips; by=s -> (s[1], s[2], s[3], s[4], s[5]))
    boxes = NTuple{6,Int}[]         # (i0, i1, j0, j1, k0, k1)
    i = 1
    while i <= length(strips)
        j = i
        while j < length(strips) && strips[j + 1][1:4] == strips[i][1:4] && strips[j + 1][5] == strips[j][5] + 1
            j += 1
        end
        push!(boxes, (strips[i][1], strips[i][2], strips[i][3], strips[i][4], strips[i][5], strips[j][5])); i = j + 1
    end
    boxes
end

"amr_viz/amr_data.json, format 4: per level the fluid-leaf boxes and the wall cells; the geometry in domain coordinates."
function write_amr_viz(out::AbstractString, cfg::CaseConfig, mesh::OctMesh)
    dir = joinpath(out, "amr_viz"); mkpath(dir)
    path = joinpath(dir, "amr_data.json")
    lo, hi = bounds(mesh.triangles)
    open(path * ".tmp", "w") do io
        print(io, "{\"format_version\":", AMR_VIZ_FORMAT, ",\"mesh\":\"octree\",\"solver\":\"octree_q19\",\"case_name\":", repr(cfg.name),
              ",\"n_levels\":", nlevels(mesh), ",\"mesh_offset\":[", join(mesh.offset, ","), "],\"domain_size\":[", join(mesh.domain, ","), "]",
              ",\"root_cells\":[", join(mesh.counts["root_cells"], ","), "],\"periodic\":[", join(mesh.periodic, ","), "]")
        print(io, ",\"geometry\":{\"n_triangles\":", length(mesh.triangles), ",\"min\":[", join(lo, ","), "],\"max\":[", join(hi, ","), "],\"vertices\":[")
        first = true
        for t in mesh.triangles, v in t, a in 1:3
            first || print(io, ","); first = false
            print(io, Float32(v[a]))
        end
        print(io, "]}")
        print(io, ",\"levels\":[")
        for (l, L) in enumerate(mesh.levels)
            boxes = leaf_boxes(L); kc = kind_counts(L)
            l > 1 && print(io, ",")
            print(io, "{\"level_id\":", l, ",\"dx\":", L.dx, ",\"n_cells\":[", join(L.n, ","), "],\"n_octs\":", L.nocts, ",\"n_fluid\":", kc["fluid"], ",\"n_solid\":", kc["solid"],
                  ",\"box_rule\":\"[i0,i1,j0,j1,k0,k1] inclusive 0-based cell index ranges of fluid leaves; cell (i,j,k) spans [i*dx,(i+1)*dx]\",\"n_boxes\":", length(boxes), ",\"boxes\":[")
            for (bi, b) in enumerate(boxes)
                bi > 1 && print(io, ",")
                print(io, b[1], ",", b[2], ",", b[3], ",", b[4], ",", b[5], ",", b[6])
            end
            print(io, "],\"n_wall_cells\":", length(L.cut_slot), ",\"wall_cells\":[")
            for (ci, s) in enumerate(L.cut_slot)
                c = slot_cell(L, Int(s)); ci > 1 && print(io, ",")
                print(io, c[1], ",", c[2], ",", c[3])
            end
            print(io, "]}")
        end
        print(io, "]}")
    end
    mv(path * ".tmp", path; force=true)
    path
end

"Level and obstacle (and later field) planes at the finest spacing over the domain, in the shared lbmslice format."
function export_planes(mesh::OctMesh, cfg::CaseConfig, sc::LatticeScaling, step::Int, slices_dir::AbstractString; with_fields::Bool=false, sampler=nothing)
    out = joinpath(slices_dir, @sprintf("step_%06d", step)); mkpath(out)
    L = nlevels(mesh); finest_h = finest(mesh).dx
    written = String[]
    axes = Dict(:x => 1, :y => 2, :z => 3); names = ("x", "y", "z")
    for pl in cfg.outputs.planes
        an = axes[pl.normal]; a1, a2 = an == 1 ? (2, 3) : an == 2 ? (1, 3) : (1, 2)
        limit=plane_sample_limit(cfg)
        stride=max(1,ceil(Int,sqrt((mesh.domain[a1]/finest_h)*(mesh.domain[a2]/finest_h)/limit)))
        h=finest_h*stride;n1=ceil(Int,mesh.domain[a1]/h);n2=ceil(Int,mesh.domain[a2]/h)
        while n1*n2>limit;stride+=1;h=finest_h*stride;n1=ceil(Int,mesh.domain[a1]/h);n2=ceil(Int,mesh.domain[a2]/h);end
        physical_coord=clamp(pl.position_m,0,mesh.domain[an]);coord=min(physical_coord,prevfloat(mesh.domain[an]))
        stride>1 && println("[SliceData] plane sampling coarsened by $stride to retain at most $limit samples; solver mesh unchanged")
        level = zeros(UInt8, n1, n2); obstacle = zeros(UInt8, n1, n2)
        fields = Dict{String,Matrix{Float32}}()
        nt = Threads.nthreads(); ranges = chunk_ranges(n2, nt)
        Threads.@threads :static for t in 1:nt
            for k in ranges[t], i in 1:n1
                p = ntuple(a -> a == an ? coord : a == a1 ? (i - 0.5) * h : (k - 0.5) * h, 3)
                l, s = locate_leaf(mesh, p)
                level[i, k] = UInt8(l)
                obstacle[i, k] = (l == 0 || mesh.levels[l].kind[slot_k(s), slot_oct(s)] != K_FLUID) ? 0x01 : 0x00
            end
        end
        arrays = Any[("level", level, "-", "octree level of the leaf at the node (0 = none)"), ("obstacle", obstacle, "-", "1 = solid or no leaf")]
        if with_fields && sampler !== nothing
            sampler(mesh, an, coord, a1, a2, n1, n2, h, arrays)
        end
        header = Dict{String,Any}(
            "case" => cfg.name, "solver" => SOLVER_NAME, "lattice" => "d3q19", "collision_operator" => String(cfg.numerics.collision),
            "plane" => names[a1] * names[a2], "plane_tag" => pl.tag, "axis1" => names[a1], "axis2" => names[a2], "normal_axis" => names[an],
            "normal_coord_m" => physical_coord, "normal_target_m" => pl.position_m, "normal_sample_m"=>coord,"normal_index_finest" => round(Int, physical_coord / finest_h),
            "sampling_stride"=>stride,"spacing_requested_m"=>finest_h,"max_plane_samples"=>limit,
            "plane_moved_to_cross_obstacle" => false, "plane_crosses_obstacle" => any(==(0x01), obstacle),
            "frame" => "domain", "frame_note" => "domain coordinates: a point of the alpha/beta-rotated STL is at its STL position + mesh_offset_m",
            "mesh_offset_m" => collect(mesh.offset), "domain_size_m" => collect(mesh.domain),
            "origin_m" => [0.0, 0.0], "spacing_m" => h, "spacing_level" => L, "spacing_requested_level" => L, "window_mode" => "domain",
            "window_m" => [0.0, n1 * h, 0.0, n2 * h],
            "node_center_rule" => "centre of node (i, k) = (origin_m[1] + (i - 0.5) * spacing_m, origin_m[2] + (k - 0.5) * spacing_m), i = 1..n1 along axis1, k = 1..n2 along axis2",
            "layout" => "column-major, element (i, k) at linear index i + (k - 1) * n1; little-endian",
            "combine_rule" => "octree_leaf_injection", "combine_note" => "each node takes the value of the octree leaf containing the node point; level 0 = outside the domain",
            "num_levels" => L, "level_dx_m" => [lv.dx for lv in mesh.levels], "coarse_grid_cells" => collect(mesh.counts["root_cells"]), "levels_used" => collect(1:L),
            "step" => step, "state_step" => step, "time_s" => step * sc.dt_finest * 2.0^(L - 1), "time_scale_s" => sc.dt_finest, "time_per_step_s" => sc.dt_finest * 2.0^(L - 1),
            "time_note" => "time_s = state_step * time_per_step_s (one solver step is one coarsest-level step)",
            "u_inlet_lattice" => sc.u_lattice, "u_target_lattice" => sc.u_lattice, "velocity_scale_m_s" => sc.velocity_scale,
            "velocity_note" => "u, v, w, velocity_magnitude = lattice velocity * velocity_scale_m_s",
            "u_ref_m_s" => cfg.flow.velocity, "rho_ref_kg_m3" => cfg.flow.density, "reference_chord_m" => cfg.reference_chord,
            "pressure_ref_lattice" => 0.0, "dynamic_pressure_ref_lattice" => 0.5 * cfg.flow.density * cfg.flow.velocity^2,
            "pressure_note" => "Pressure is gauge Pa; legacy reference keys above also use Pa. Cp = pressure / dynamic_pressure_ref_lattice.", "with_fields" => with_fields)
        fname = joinpath(out, @sprintf("plane_%s_%06d.lbmslice", pl.tag, step))
        write_lbmslice(fname, header, arrays)
        push!(written, fname)
        println(@sprintf("[SliceData] %s: %s (%dx%d nodes, h=%.6g m)", pl.tag, fname, n1, n2, h))
    end
    written
end

function write_provenance(out::AbstractString, cfg::CaseConfig, mesh_sha::AbstractString)
    d = Dict{String,Any}("case" => cfg.name, "solver" => SOLVER_NAME, "package_version" => "0.1.0", "julia" => string(VERSION),
        "config_sha256" => cfg.config_sha, "stl_sha256" => cfg.stl_sha, "mesh_sha256" => mesh_sha, "threads" => Threads.nthreads(),
        "written_utc" => Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS") * "Z", "precision" => string(cfg.precision))
    open(joinpath(out, "provenance.json"), "w") do io
        print(io, "{", join(["$(repr(k)):$(v isa AbstractString ? repr(v) : v)" for (k, v) in d], ","), "}")
    end
end
