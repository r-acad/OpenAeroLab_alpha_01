// # FILE: .\src\bouzidi_setup.jl
using Base.Threads
using Printf
using StaticArrays

"""
BOUZIDI_SETUP.JL - Preprocessing and Initialization
"""

"""
    build_block_triangle_map_for_bouzidi - Build spatial hash of triangles per block
"""
function build_block_triangle_map_for_bouzidi(mesh, 
                                              sorted_blocks::Vector{Tuple{Int,Int,Int}},
                                              dx::Float64,
                                              mesh_offset::SVector{3,Float64},
                                              block_size::Int)
    block_tris = [Int[] for _ in 1:length(sorted_blocks)]
    bs = block_size
    
    b_lookup = Dict{Tuple{Int,Int,Int}, Int}()
    for (i, coord) in enumerate(sorted_blocks)
        b_lookup[coord] = i
    end
    
    max_link_len = 1.0
    if isdefined(Main, :C_X) && length(C_X) > 1
        max_link_len = maximum(sqrt(Float64(C_X[k])^2 + Float64(C_Y[k])^2 + Float64(C_Z[k])^2)
                               for k in 1:length(C_X))
    end
    margin = dx * (max_link_len + 1.0)
    
    for (t_idx, tri) in enumerate(mesh.triangles)
        v1 = SVector{3,Float64}(tri[1])
        v2 = SVector{3,Float64}(tri[2])
        v3 = SVector{3,Float64}(tri[3])
        
        tri_min = min.(v1, min.(v2, v3))
        tri_max = max.(v1, max.(v2, v3))
        
        min_pt = tri_min + mesh_offset
        max_pt = tri_max + mesh_offset
        
        min_b = floor.(Int, (min_pt .- margin) ./ (bs * dx)) .+ 1
        max_b = floor.(Int, (max_pt .+ margin) ./ (bs * dx)) .+ 1
        
        for bz in max(1, min_b[3]):max_b[3]
            for by in max(1, min_b[2]):max_b[2]
                for bx in max(1, min_b[1]):max_b[1]
                    if haskey(b_lookup, (bx, by, bz))
                        idx = b_lookup[(bx, by, bz)]
                        push!(block_tris[idx], t_idx)
                    end
                end
            end
        end
    end
    
    return block_tris
end

# Per-block q-map worker. Extracted verbatim from the original serial loop body
# so the parallel (@threads :static) and serial paths are byte-identical. Uses
# Threads.threadid() to index the per-thread accumulators — SAFE under :static
# scheduling (iteration pinned to a fixed thread for the whole call). The
# q_map_cpu / tri_map_cpu writes are indexed by b_idx → disjoint across blocks.
function _bouzidi_qmap_block!(
        b_idx::Int, level_active_coords, block_tri_map, mesh, dx::Float64,
        mesh_offset::SVector{3,Float64}, bs::Int,
        q_geometry::Symbol, q_frame_count::Int, n_dirs::Int,
        cx_frames, cy_frames, cz_frames,
        require_solid_neighbor::Bool, obstacle_arr, block_lookup,
        sphere_center::SVector{3,Float64}, sphere_radius::Float64,
        q_map_cpu, tri_map_cpu,
        thread_boundary_cells, link_hits, link_kept, link_rejected)
    tid = Threads.threadid()
    (bx, by, bz) = level_active_coords[b_idx]
    relevant_tris_indices = block_tri_map[b_idx]

    if isempty(relevant_tris_indices) && q_geometry != :sphere_analytic
        return
    end
    if tid > length(thread_boundary_cells)
        return
    end

    local_triangles = q_geometry == :sphere_analytic ?
                      eltype(mesh.triangles)[] :
                      [mesh.triangles[t_idx] for t_idx in relevant_tris_indices]

    for lz in 1:bs, ly in 1:bs, lx in 1:bs
        if require_solid_neighbor && obstacle_arr[lx, ly, lz, b_idx]
            continue
        end

        px = ((bx - 1) * bs + lx - 0.5) * dx
        py = ((by - 1) * bs + ly - 0.5) * dx
        pz = ((bz - 1) * bs + lz - 0.5) * dx
        cell_center = SVector(px, py, pz)

        has_boundary = false
        for frame_idx in 1:q_frame_count
            cx = cx_frames[frame_idx]
            cy = cy_frames[frame_idx]
            cz = cz_frames[frame_idx]
            if q_geometry == :sphere_analytic
                q_vals, tri_idxs = compute_q_for_sphere_cell(cell_center, dx,
                                                             sphere_center, sphere_radius,
                                                             cx, cy, cz)
            else
                q_vals, tri_idxs = compute_q_for_cell(cell_center, dx, local_triangles,
                                                      mesh_offset, cx, cy, cz, relevant_tris_indices)
            end

            frame_offset = (frame_idx - 1) * n_dirs
            for k in 1:n_dirs
                if q_vals[k] > 0.0
                    link_hits[tid] += 1
                    keep_link = true
                    if require_solid_neighbor
                        nx = lx + Int(cx[k])
                        ny = ly + Int(cy[k])
                        nz = lz + Int(cz[k])
                        nb_idx = b_idx
                        if nx < 1 || nx > bs || ny < 1 || ny > bs || nz < 1 || nz > bs
                            nbx = bx + (nx < 1 ? -1 : (nx > bs ? 1 : 0))
                            nby = by + (ny < 1 ? -1 : (ny > bs ? 1 : 0))
                            nbz = bz + (nz < 1 ? -1 : (nz > bs ? 1 : 0))
                            nb_idx = get(block_lookup, (nbx, nby, nbz), 0)
                            nx = nx < 1 ? nx + bs : (nx > bs ? nx - bs : nx)
                            ny = ny < 1 ? ny + bs : (ny > bs ? ny - bs : ny)
                            nz = nz < 1 ? nz + bs : (nz > bs ? nz - bs : nz)
                        end
                        keep_link = nb_idx > 0 && obstacle_arr[nx, ny, nz, nb_idx]
                    end

                    if keep_link
                        q_slot = frame_offset + k
                        q_map_cpu[lx, ly, lz, b_idx, q_slot] = Float16(q_vals[k])
                        tri_map_cpu[lx, ly, lz, b_idx, q_slot] = tri_idxs[k]
                        has_boundary = true
                        link_kept[tid] += 1
                    else
                        link_rejected[tid] += 1
                    end
                end
            end
        end

        if has_boundary
            push!(thread_boundary_cells[tid], (Int32(b_idx), Int32(lx), Int32(ly), Int32(lz)))
        end
    end
    return
end

"""
    compute_bouzidi_qmap_sparse(level_active_coords, mesh, dx, mesh_offset, block_size, obstacle_arr=nothing)

Compute Q-values and build sparse boundary cell coordinate lists.
Also computes the mapping from boundary links to triangle indices.

Returns: (q_map, cell_block, cell_x, cell_y, cell_z, tri_map, n_boundary_cells)
"""
function compute_bouzidi_qmap_sparse(level_active_coords::Vector{Tuple{Int,Int,Int}},
                                     mesh,
                                     dx::Float64,
                                     mesh_offset::SVector{3,Float64},
                                     block_size::Int,
                                     obstacle_arr=nothing;
                                     q_geometry::Symbol=:stl_ray)
    
    println("[Bouzidi] Computing Q-values and surface mapping for $(length(level_active_coords)) blocks...")
    
    n_blocks = length(level_active_coords)
    bs = block_size
    
    # Build lattice velocity vectors for ray casting.  In local shifted-lattice
    # mode the wall geometry must be queried in the same block frame that the
    # Bouzidi kernel will use at runtime; otherwise U=0 wall blocks read q-values
    # for the reference shifted rays and the boundary becomes nonphysical.
    n_dirs = Int(N_DIRS)
    array_shift_x = Int32(C_X[1])
    array_shift_y = Int32(C_Y[1])
    array_shift_z = Int32(C_Z[1])
    base_cx = [Int32(C_X[k]) - array_shift_x for k in 1:n_dirs]
    base_cy = [Int32(C_Y[k]) - array_shift_y for k in 1:n_dirs]
    base_cz = [Int32(C_Z[k]) - array_shift_z for k in 1:n_dirs]
    q_frame_count = (LOCAL_LATTICE_SHIFT_ENABLED && LATTICE_TYPE == :d3q39) ?
                    max(Int(LOCAL_LATTICE_SHIFT_U_MAX) + 1, 1) : 1
    cx_frames = [Vector{Int32}(undef, n_dirs) for _ in 1:q_frame_count]
    cy_frames = [Vector{Int32}(undef, n_dirs) for _ in 1:q_frame_count]
    cz_frames = [Vector{Int32}(undef, n_dirs) for _ in 1:q_frame_count]
    for frame_idx in 1:q_frame_count
        frame_x = q_frame_count == 1 ? array_shift_x : Int32(frame_idx - 1)
        frame_y = array_shift_y
        frame_z = array_shift_z
        for k in 1:n_dirs
            cx_frames[frame_idx][k] = base_cx[k] + frame_x
            cy_frames[frame_idx][k] = base_cy[k] + frame_y
            cz_frames[frame_idx][k] = base_cz[k] + frame_z
        end
    end
    q_dirs_total = n_dirs * q_frame_count

    q_map_cpu = zeros(Float16, bs, bs, bs, n_blocks, q_dirs_total)
    tri_map_cpu = zeros(Int32, bs, bs, bs, n_blocks, q_dirs_total)
    if q_frame_count > 1
        @printf("[Bouzidi] Local-shift q-map: %d x-frames, %d stored wall-link directions\n",
                q_frame_count, q_dirs_total)
    end
    
    max_tid = Threads.nthreads()
    if isdefined(Threads, :threadpoolsize)
        try
            max_tid += Threads.threadpoolsize(:interactive)
        catch
        end
    end
    
    thread_boundary_cells = [Vector{NTuple{4,Int32}}() for _ in 1:max_tid]
    
    block_tri_map = build_block_triangle_map_for_bouzidi(mesh, level_active_coords,
                                                         dx, mesh_offset, bs)
    # Always-bound defaults (the worker takes them as args; only used when
    # q_geometry == :sphere_analytic, but must be defined for the stl_ray path).
    sphere_center = SVector(0.0, 0.0, 0.0)
    sphere_radius = 0.0
    if q_geometry == :sphere_analytic
        min_pt = SVector{3,Float64}(mesh.min_bounds) + mesh_offset
        max_pt = SVector{3,Float64}(mesh.max_bounds) + mesh_offset
        # SYMMETRY FIX (2026-06-05): under HALF_Y_DOMAIN the mesh is sliced to the
        # half y∈[0,R], so 0.5*(min+max) puts the sphere center at R/2 (=0.25) —
        # OFF the y=0 mirror plane — and the Bouzidi q-map is then built around an
        # off-plane sphere → asymmetric near-plane wall links → the runaway u_y.
        # This is the SAME bug as apply_analytic_sphere_obstacle! (domain_generation.jl);
        # the q-map had its OWN duplicate center calc that the earlier fix missed (it
        # is what produced the "[Bouzidi] q-geometry ... center y=0.250000" line while
        # the obstacle mask correctly showed y=0.000000). Force the y-center onto the
        # plane and take the radius from the full in-plane (X/Z) extent.
        if @isdefined(HALF_Y_DOMAIN) && HALF_Y_DOMAIN
            sphere_radius = 0.5 * max(max_pt[1] - min_pt[1], max_pt[3] - min_pt[3])
            sphere_center = SVector(0.5 * (min_pt[1] + max_pt[1]),
                                    min_pt[2],                       # symmetry plane y=0
                                    0.5 * (min_pt[3] + max_pt[3]))
        else
            sphere_center = 0.5 * (min_pt + max_pt)
            sphere_radius = 0.5 * maximum(max_pt - min_pt)
        end
        @printf("[Bouzidi] q-geometry: analytic sphere center=(%.6f, %.6f, %.6f), R=%.6f\n",
                sphere_center[1], sphere_center[2], sphere_center[3], sphere_radius)
    elseif q_geometry != :stl_ray
        error("Unknown Bouzidi q_geometry: $q_geometry")
    end
    block_lookup = Dict{Tuple{Int,Int,Int}, Int}()
    for (i, coord) in enumerate(level_active_coords)
        block_lookup[coord] = i
    end
    # Periodic-Y AMR (27 Oct 2026): a wall link that crosses the Y seam must find
    # its solid neighbour in the wrapped block (no-op when periodic_y_amr is off).
    _add_periodic_y_ghost_keys!(block_lookup, _level_by_max(dx))
    require_solid_neighbor = obstacle_arr !== nothing
    link_hits = zeros(Int, max_tid)
    link_kept = zeros(Int, max_tid)
    link_rejected = zeros(Int, max_tid)
    
    # ── Parallel q-map construction (2026-06-03) ────────────────────────────
    # Originally serial (see git history): threaded writes once caused a
    # hard-to-diagnose Windows/CUDA startup failure in LOCAL SHIFTED-LATTICE
    # cases (D3Q39 local shift). Root cause was the classic Julia footgun of
    # indexing per-thread buffers by `threadid()` while tasks can MIGRATE
    # between threads (threadid() is not stable across yields on the default
    # dynamic scheduler). The fix is `@threads :static`, which pins each
    # iteration to a fixed thread for the whole body, making `threadid()`
    # stable and the per-thread accumulators (thread_boundary_cells[tid],
    # link_*[tid]) race-free. The q_map_cpu / tri_map_cpu writes are indexed
    # by b_idx (the loop variable), so different iterations touch DISJOINT
    # slices — no data race regardless of scheduling.
    #
    # This turns the serial q-map (the dominant cost of the domain build on
    # LARGE meshes, e.g. the 39 MB A250 aircraft) into a ~Ncore speedup.
    #
    # SAFETY: the historically-fragile local-shift path is kept SERIAL (the
    # `use_threads` gate) so its proven-good behavior is byte-identical; only
    # the common single-frame path is parallelized.
    use_threads = (Threads.nthreads() > 1) && (q_frame_count == 1)
    if use_threads
        Threads.@threads :static for b_idx in 1:n_blocks
            _bouzidi_qmap_block!(
                b_idx, level_active_coords, block_tri_map, mesh, dx, mesh_offset, bs,
                q_geometry, q_frame_count, n_dirs,
                cx_frames, cy_frames, cz_frames,
                require_solid_neighbor, obstacle_arr, block_lookup,
                sphere_center, sphere_radius,
                q_map_cpu, tri_map_cpu,
                thread_boundary_cells, link_hits, link_kept, link_rejected)
        end
    else
        for b_idx in 1:n_blocks
            _bouzidi_qmap_block!(
                b_idx, level_active_coords, block_tri_map, mesh, dx, mesh_offset, bs,
                q_geometry, q_frame_count, n_dirs,
                cx_frames, cy_frames, cz_frames,
                require_solid_neighbor, obstacle_arr, block_lookup,
                sphere_center, sphere_radius,
                q_map_cpu, tri_map_cpu,
                thread_boundary_cells, link_hits, link_kept, link_rejected)
        end
    end
    if false  # ── legacy inline serial body (kept for diff reference) ──
    for b_idx in 1:n_blocks
        tid = Threads.threadid()
        (bx, by, bz) = level_active_coords[b_idx]
        relevant_tris_indices = block_tri_map[b_idx]
        
        if isempty(relevant_tris_indices) && q_geometry != :sphere_analytic
            continue
        end

        if tid > length(thread_boundary_cells)
             continue 
        end
        
        local_triangles = q_geometry == :sphere_analytic ?
                          eltype(mesh.triangles)[] :
                          [mesh.triangles[t_idx] for t_idx in relevant_tris_indices]
        
        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            if require_solid_neighbor && obstacle_arr[lx, ly, lz, b_idx]
                continue
            end

            px = ((bx - 1) * bs + lx - 0.5) * dx
            py = ((by - 1) * bs + ly - 0.5) * dx
            pz = ((bz - 1) * bs + lz - 0.5) * dx
            cell_center = SVector(px, py, pz)
            
            # Now returns Tuple(q_vals, tri_indices). For sphere validation,
            # use an analytic zero-crossing so the q-map and voxel mask cannot
            # disagree because of STL facet/ray misses.
            has_boundary = false
            for frame_idx in 1:q_frame_count
                cx = cx_frames[frame_idx]
                cy = cy_frames[frame_idx]
                cz = cz_frames[frame_idx]
                if q_geometry == :sphere_analytic
                    q_vals, tri_idxs = compute_q_for_sphere_cell(cell_center, dx,
                                                                 sphere_center, sphere_radius,
                                                                 cx, cy, cz)
                else
                    q_vals, tri_idxs = compute_q_for_cell(cell_center, dx, local_triangles,
                                                          mesh_offset, cx, cy, cz, relevant_tris_indices)
                end

                frame_offset = (frame_idx - 1) * n_dirs
                for k in 1:n_dirs
                    if q_vals[k] > 0.0
                        link_hits[tid] += 1
                        keep_link = true
                        if require_solid_neighbor
                            nx = lx + Int(cx[k])
                            ny = ly + Int(cy[k])
                            nz = lz + Int(cz[k])
                            nb_idx = b_idx
                            if nx < 1 || nx > bs || ny < 1 || ny > bs || nz < 1 || nz > bs
                                nbx = bx + (nx < 1 ? -1 : (nx > bs ? 1 : 0))
                                nby = by + (ny < 1 ? -1 : (ny > bs ? 1 : 0))
                                nbz = bz + (nz < 1 ? -1 : (nz > bs ? 1 : 0))
                                nb_idx = get(block_lookup, (nbx, nby, nbz), 0)
                                nx = nx < 1 ? nx + bs : (nx > bs ? nx - bs : nx)
                                ny = ny < 1 ? ny + bs : (ny > bs ? ny - bs : ny)
                                nz = nz < 1 ? nz + bs : (nz > bs ? nz - bs : nz)
                            end
                            keep_link = nb_idx > 0 && obstacle_arr[nx, ny, nz, nb_idx]
                        end

                        if keep_link
                            q_slot = frame_offset + k
                            q_map_cpu[lx, ly, lz, b_idx, q_slot] = Float16(q_vals[k])
                            tri_map_cpu[lx, ly, lz, b_idx, q_slot] = tri_idxs[k]
                            has_boundary = true
                            link_kept[tid] += 1
                        else
                            link_rejected[tid] += 1
                        end
                    end
                end
            end
            
            if has_boundary
                push!(thread_boundary_cells[tid], (Int32(b_idx), Int32(lx), Int32(ly), Int32(lz)))
            end
        end
    end
    end  # close `if false` legacy-body block

    all_boundary_cells = NTuple{4,Int32}[]
    for cells in thread_boundary_cells
        append!(all_boundary_cells, cells)
    end
    
    n_boundary = length(all_boundary_cells)
    total_cells = n_blocks * bs^3
    
    cell_block = Vector{Int32}(undef, n_boundary)
    cell_x = Vector{Int32}(undef, n_boundary)
    cell_y = Vector{Int32}(undef, n_boundary)
    cell_z = Vector{Int32}(undef, n_boundary)
    
    for (i, (b, x, y, z)) in enumerate(all_boundary_cells)
        cell_block[i] = b
        cell_x[i] = x
        cell_y[i] = y
        cell_z[i] = z
    end
    
    qmap_mem_mb = sizeof(q_map_cpu) / 1024^2
    sparse_mem_kb = (sizeof(cell_block) + sizeof(cell_x) + sizeof(cell_y) + sizeof(cell_z)) / 1024
    
    println("[Bouzidi] Found $n_boundary boundary cells ($(round(100*n_boundary/total_cells, digits=2))%)")
    if require_solid_neighbor
        hits = sum(link_hits)
        kept = sum(link_kept)
        rejected = sum(link_rejected)
        keep_pct = hits > 0 ? 100 * kept / hits : 0.0
        println("[Bouzidi] Link topology filter: kept $kept / $hits q-links ($(round(keep_pct, digits=1))%), rejected $rejected grazing links")
    end
    println("[Bouzidi] Q-map: $(round(qmap_mem_mb, digits=1)) MB, Sparse coords: $(round(sparse_mem_kb, digits=1)) KB")
    
    return q_map_cpu, cell_block, cell_x, cell_y, cell_z, tri_map_cpu, n_boundary
end
