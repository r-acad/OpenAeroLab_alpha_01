# FILE: src/domain_generation.jl
using Base.Threads
using LinearAlgebra
using StaticArrays

"""
DOMAIN_GENERATION.JL - Voxelization, Sponge Layers, and Geometry Processing
"""

# ── Periodic-Y AMR (27 Oct 2026) ──────────────────────────────────────────────
# Block-coordinate Dicts used by the CPU setup passes (wall distances, Bouzidi
# solid-neighbour test) look up the neighbour block at by±1 directly. On the
# periodic Y axis add ghost keys so (bx, 0, bz) resolves to the block at
# by = by_max and (bx, by_max+1, bz) to the block at by = 1; the callers already
# wrap the local cell index. No-op unless periodic_y_amr is active.
function _add_periodic_y_ghost_keys!(lookup::Dict{Tuple{Int,Int,Int},Int}, by_max::Int)
    periodic_y_amr_active() || return lookup
    by_max > 0 || return lookup
    extra = Pair{Tuple{Int,Int,Int},Int}[]
    for ((bx, by, bz), i) in lookup
        by == 1 && push!(extra, (bx, by_max + 1, bz) => i)
        by == by_max && push!(extra, (bx, 0, bz) => i)
    end
    for p in extra
        lookup[p.first] = p.second
    end
    return lookup
end

"Grid Y block count of the level with cell size `dx` (the periodic Y period in blocks)."
_level_by_max(dx::Real) = DOMAIN_PARAMS.initialized ?
    round(Int, DOMAIN_PARAMS.domain_size[2] / (BLOCK_SIZE * Float64(dx))) : 0

function build_block_triangle_map(mesh::Geometry.SolverMesh,
                                  sorted_blocks::Vector{Tuple{Int, Int, Int}}, 
                                  dx::Float64,
                                  mesh_offset::SVector{3,Float64})
    block_tris = [Int[] for _ in 1:length(sorted_blocks)]
    bs = BLOCK_SIZE
    bx_max = isempty(sorted_blocks) ? 0 : maximum(coord[1] for coord in sorted_blocks)
    by_max = isempty(sorted_blocks) ? 0 : maximum(coord[2] for coord in sorted_blocks)
    bz_max = isempty(sorted_blocks) ? 0 : maximum(coord[3] for coord in sorted_blocks)
    
    b_lookup = Dict{Tuple{Int, Int, Int}, Int}()
    for (i, coord) in enumerate(sorted_blocks)
        b_lookup[coord] = i
    end
    
    margin = dx * 2

    for (t_idx, tri) in enumerate(mesh.triangles)
        _, _, _, min_pt, max_pt = triangle_bounds_with_offset(tri, mesh_offset)
        min_bx, max_bx, min_by, max_by, min_bz, max_bz =
            block_index_bounds_from_aabb(min_pt, max_pt, bs * dx, bx_max, by_max, bz_max;
                                         margin=margin)

        for bz in min_bz:max_bz
            for by in min_by:max_by
                for bx in min_bx:max_bx
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

function voxelize_blocks!(obstacle_arr, sorted_blocks, mesh::Geometry.SolverMesh, 
                          dx::Float64, mesh_offset::SVector{3,Float64})
    block_tri_map = build_block_triangle_map(mesh, sorted_blocks, dx, mesh_offset)
    
    println("[Domain] SAT Surface Marking...")
    box_half = SVector(0.75, 0.75, 0.75) * dx
    
    @threads for i in 1:length(sorted_blocks)
        (bx, by, bz) = sorted_blocks[i]
        relevant_tris = block_tri_map[i]
        if isempty(relevant_tris)
            continue
        end
        
        for lz in 1:BLOCK_SIZE, ly in 1:BLOCK_SIZE, lx in 1:BLOCK_SIZE
            px = ((bx - 1) * BLOCK_SIZE + lx - 0.5) * dx
            py = ((by - 1) * BLOCK_SIZE + ly - 0.5) * dx
            pz = ((bz - 1) * BLOCK_SIZE + lz - 0.5) * dx
            center = SVector(px, py, pz)
            
            is_shell = false
            for tid in relevant_tris
                t = mesh.triangles[tid]
                v1 = t[1] .+ mesh_offset
                v2 = t[2] .+ mesh_offset
                v3 = t[3] .+ mesh_offset
                
                if triangle_intersects_aabb(center, box_half, v1, v2, v3)
                    is_shell = true
                    break
                end
            end
            
            if is_shell
                obstacle_arr[lx, ly, lz, i] = true
            end
        end
    end
end

function apply_analytic_sphere_obstacle!(obstacle_arr,
                                         sorted_blocks,
                                         mesh::Geometry.SolverMesh,
                                         dx::Float64,
                                         mesh_offset::SVector{3,Float64})
    min_pt = SVector{3,Float64}(mesh.min_bounds) + mesh_offset
    max_pt = SVector{3,Float64}(mesh.max_bounds) + mesh_offset
    # SYMMETRY FIX (2026-06-04): under HALF_Y_DOMAIN the mesh has been sliced to the
    # half y∈[0, R], so the bbox y-center is R/2 (e.g. 0.25 for R=0.5) — NOT the
    # physical sphere center, which sits ON the symmetry plane y=0 (= min_pt.y after
    # placement). The legacy `center = 0.5(min+max)` therefore stamped a full sphere
    # mis-centered half a radius ABOVE the plane (reaching y=+0.75, dipping to y=-0.25
    # below the wall) — the obstacle no longer matched the auto-sliced half-sphere
    # triangle mesh and broke the symmetric setup. Force the y-center to the plane and
    # derive the radius from the full (X/Z) extent (the in-plane diameter is the true
    # full diameter; the sliced y-extent is only the half-radius). Full-domain runs are
    # unchanged (the else branch is byte-identical to the legacy formula).
    if HALF_Y_DOMAIN
        radius = 0.5 * max(max_pt[1] - min_pt[1], max_pt[3] - min_pt[3])
        center = SVector{3,Float64}(0.5 * (min_pt[1] + max_pt[1]),
                                    min_pt[2],                       # symmetry plane y=0
                                    0.5 * (min_pt[3] + max_pt[3]))
    else
        center = 0.5 * (min_pt + max_pt)
        radius = 0.5 * maximum(max_pt - min_pt)
    end
    r2 = radius * radius
    bs = BLOCK_SIZE

    @threads for b_idx in 1:length(sorted_blocks)
        (bx, by, bz) = sorted_blocks[b_idx]
        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            px = ((bx - 1) * bs + lx - 0.5) * dx
            py = ((by - 1) * bs + ly - 0.5) * dx
            pz = ((bz - 1) * bs + lz - 0.5) * dx
            p = SVector(px, py, pz)
            obstacle_arr[lx, ly, lz, b_idx] = sum((p - center) .^ 2) <= r2
        end
    end

    solid_count = count(identity, obstacle_arr)
    @printf("[Domain] Analytic sphere obstacle mask: center=(%.6f, %.6f, %.6f), R=%.6f, solid voxels=%d\n",
            center[1], center[2], center[3], radius, solid_count)
end

function perform_flood_fill!(obstacle_arr, sorted_blocks, block_ptr, 
                             grid_dim_x, grid_dim_y, grid_dim_z)
    println("[Domain] Flood Fill...")
    if isempty(sorted_blocks)
        println("[Domain] No active blocks; skipping flood fill.")
        return
    end
    visited = falses(size(obstacle_arr))
    q_block = Int32[]
    q_idx = Int32[]
    
    min_x_block = minimum(b[1] for b in sorted_blocks)
    
    for b_idx in 1:length(sorted_blocks)
        (bx, by, bz) = sorted_blocks[b_idx]
        if bx == min_x_block
            for z in 1:BLOCK_SIZE, y in 1:BLOCK_SIZE, x in 1:BLOCK_SIZE
                if !obstacle_arr[x, y, z, b_idx]
                    visited[x, y, z, b_idx] = true
                    push!(q_block, b_idx)
                    push!(q_idx, x + (y-1)*BLOCK_SIZE + (z-1)*BLOCK_SIZE*BLOCK_SIZE)
                end
            end
        end
    end
    
    head = 1
    dx_arr = [1, -1, 0, 0, 0, 0]
    dy_arr = [0, 0, 1, -1, 0, 0]
    dz_arr = [0, 0, 0, 0, 1, -1]
    bs = BLOCK_SIZE
    bs2 = bs*bs
    
    function has_block_local(bx, by, bz)
        if bx < 1 || bx > grid_dim_x || by < 1 || by > grid_dim_y || bz < 1 || bz > grid_dim_z
            return false
        end
        return block_ptr[bx, by, bz] > 0
    end
    
    while head <= length(q_block)
        b_curr = q_block[head]
        idx_curr = q_idx[head]
        head += 1
        
        rem = idx_curr - 1
        lz = rem ÷ bs2 + 1
        rem = rem % bs2
        ly = rem ÷ bs + 1
        lx = rem % bs + 1
        
        (bx, by, bz) = sorted_blocks[b_curr]
        
        for i in 1:6
            nlx = lx + dx_arr[i]
            nly = ly + dy_arr[i]
            nlz = lz + dz_arr[i]
            
            if nlx >= 1 && nlx <= bs && nly >= 1 && nly <= bs && nlz >= 1 && nlz <= bs
                if !visited[nlx, nly, nlz, b_curr] && !obstacle_arr[nlx, nly, nlz, b_curr]
                    visited[nlx, nly, nlz, b_curr] = true
                    push!(q_block, b_curr)
                    push!(q_idx, nlx + (nly-1)*bs + (nlz-1)*bs2)
                end
            else
                nbx = bx + dx_arr[i]
                nby = _topo_wrap_by(by + dy_arr[i], grid_dim_y)   # periodic-Y AMR wrap (identity when off)
                nbz = bz + dz_arr[i]

                if has_block_local(nbx, nby, nbz)
                    nb_idx = block_ptr[nbx, nby, nbz]
                    wlx = (nlx < 1) ? bs : (nlx > bs ? 1 : nlx)
                    wly = (nly < 1) ? bs : (nly > bs ? 1 : nly)
                    wlz = (nlz < 1) ? bs : (nlz > bs ? 1 : nlz)
                    
                    if !visited[wlx, wly, wlz, nb_idx] && !obstacle_arr[wlx, wly, wlz, nb_idx]
                        visited[wlx, wly, wlz, nb_idx] = true
                        push!(q_block, nb_idx)
                        push!(q_idx, wlx + (wly-1)*bs + (wlz-1)*bs2)
                    end
                end
            end
        end
    end
    
    filled_count = 0
    for i in 1:length(obstacle_arr)
        if !obstacle_arr[i] && !visited[i]
            obstacle_arr[i] = true
            filled_count += 1
        end
    end
    println("[Domain] Filled $filled_count interior voxels.")
end

@inline function smooth_sponge_profile(x::Float64, thickness::Float64)
    if x <= 0.0
        return 1.0
    elseif x >= thickness
        return 0.0
    else
        return 0.5 * (1.0 + cos(π * x / thickness))
    end
end

# ──────────────────────────────────────────────────────────────────
# Palabos-style tanh sponge profile (spongeZones3D.hh).
# Transition centered at x = thickness/2 with steepness controlled by sigma
# (fraction of thickness). Smaller sigma = sharper transition.
# s(0)=1, s(thickness)=0, with C∞-smooth tanh interior.
# ──────────────────────────────────────────────────────────────────
@inline function tanh_sponge_profile(x::Float64, thickness::Float64, sigma::Float64)
    if x <= 0.0
        return 1.0
    elseif x >= thickness
        return 0.0
    else
        arg = (x - 0.5 * thickness) / (sigma * thickness)
        return 0.5 * (1.0 - tanh(arg))
    end
end

@inline function sponge_profile(x::Float64, thickness::Float64)
    if SPONGE_PROFILE === :tanh
        return tanh_sponge_profile(x, thickness, Float64(SPONGE_TANH_SIGMA))
    else
        return smooth_sponge_profile(x, thickness)
    end
end

@inline function combine_sponge_strength(current::Float64, added::Float64)
    cur = clamp(current, 0.0, 1.0)
    add = clamp(added, 0.0, 1.0)
    # Smooth saturated union of independent sponge bands.
    # Using max(cur, add) leaves a ridge where two bands cross because the
    # active source switches abruptly from one face to the other. The union
    # blend stays monotone, never exceeds 1, and removes that corner kink.
    return 1.0 - (1.0 - cur) * (1.0 - add)
end

function apply_sponge!(sponge_arr, sorted_blocks, params, lvl_scale::Int)
    println("[Domain] Applying Sponge Layers...")
    if isempty(sorted_blocks)
        println("[Domain] No active blocks; skipping sponge layers.")
        return
    end
    dx = params.dx_coarse / lvl_scale
    
    Lx = params.domain_size[1]
    Ly = params.domain_size[2]
    Lz = params.domain_size[3]
    
    # Interpret sponge thickness in units of reference length, matching the
    # case configuration and analysis summary. Using a fraction of the whole
    # domain length makes the outflow sponge intrude too far upstream in long
    # tunnels and prints a vertical seam into downstream slice diagnostics.
    sponge_ref = params.reference_length * Float64(SPONGE_THICKNESS)
    outlet_thickness = min(Lx, params.reference_length * max(Float64(SPONGE_THICKNESS), 0.08))
    
    inlet_thickness = min(Lx, params.reference_length * 0.02)
    y_sponge_thickness = min(Ly / 2.0, sponge_ref * 0.5)
    z_sponge_thickness = min(Lz / 2.0, sponge_ref * 0.5)
    
    outlet_start = Lx - outlet_thickness
    y_top_start = Ly - y_sponge_thickness
    z_back_start = Lz - z_sponge_thickness
    
    outlet_strength = Float64(SPONGE_OUTLET_STRENGTH)
    inlet_strength = Float64(SPONGE_INLET_STRENGTH)
    wall_strength = Float64(SPONGE_TRANSVERSE_STRENGTH)
    # advanced.boundary.z_walls (lbm-zwalls, 27 Oct 2026): with "slip" or "no_slip" the z faces
    # are walls and carry NO sponge (the y-face, inlet and outlet sponges are unchanged). Key
    # absent (:legacy) ⇒ the z-face sponge below runs exactly as before.
    z_face_sponge = Z_WALLS_MODE === :legacy
    z_face_sponge || println("[Domain] z_walls=$(Z_WALLS_MODE): no sponge on the z faces")

    for i in 1:length(sorted_blocks)
        (bx, by, bz) = sorted_blocks[i]
        
        for lz in 1:BLOCK_SIZE, ly in 1:BLOCK_SIZE, lx in 1:BLOCK_SIZE
            px = ((bx - 1) * BLOCK_SIZE + lx - 0.5) * dx
            py = ((by - 1) * BLOCK_SIZE + ly - 0.5) * dx
            pz = ((bz - 1) * BLOCK_SIZE + lz - 0.5) * dx
            
            sponge_val = 0.0
            
            if px > outlet_start
                dist_from_boundary = px - outlet_start
                s = sponge_profile(outlet_thickness - dist_from_boundary, outlet_thickness)
                sponge_val = combine_sponge_strength(sponge_val, s * outlet_strength)
            end
            
            if px < inlet_thickness
                s = sponge_profile(px, inlet_thickness)
                sponge_val = combine_sponge_strength(sponge_val, s * inlet_strength)
            end
            
            if !SPAN_PERIODIC_Y && !HALF_Y_DOMAIN && py < y_sponge_thickness
                s = sponge_profile(py, y_sponge_thickness)
                sponge_val = combine_sponge_strength(sponge_val, s * wall_strength)
            end
            
            if !SPAN_PERIODIC_Y && py > y_top_start
                dist_from_boundary = py - y_top_start
                s = sponge_profile(y_sponge_thickness - dist_from_boundary, y_sponge_thickness)
                sponge_val = combine_sponge_strength(sponge_val, s * wall_strength)
            end
            
            if z_face_sponge && pz < z_sponge_thickness
                s = sponge_profile(pz, z_sponge_thickness)
                sponge_val = combine_sponge_strength(sponge_val, s * wall_strength)
            end

            if z_face_sponge && pz > z_back_start
                dist_from_boundary = pz - z_back_start
                s = sponge_profile(z_sponge_thickness - dist_from_boundary, z_sponge_thickness)
                sponge_val = combine_sponge_strength(sponge_val, s * wall_strength)
            end
            
            sponge_arr[lx, ly, lz, i] = Float32(sponge_val)
        end
    end
    
    sponge_cells = count(sponge_arr .> 0.0f0)
    total_cells = length(sponge_arr)
    max_sponge = maximum(sponge_arr)
    @printf("[Domain] Sponge (%s): %.1f%% cells affected, max strength = %.3f\n",
            SPONGE_PROFILE, 100.0 * sponge_cells / total_cells, max_sponge)
end


function compute_wall_distances!(wall_dist_arr, sorted_blocks, obstacle_arr, 
                                 mesh::Geometry.SolverMesh, dx::Float64, 
                                 mesh_offset::SVector{3,Float64})
    println("[Domain] Computing wall distances...")
    bs = BLOCK_SIZE
    n_blocks = length(sorted_blocks)
    
    block_lookup = Dict{Tuple{Int,Int,Int}, Int}()
    for (i, coord) in enumerate(sorted_blocks)
        block_lookup[coord] = i
    end
    _add_periodic_y_ghost_keys!(block_lookup, _level_by_max(dx))   # periodic-Y AMR (no-op when off)

    near_wall_count = 0
    
    @threads for b_idx in 1:n_blocks
        (bx, by, bz) = sorted_blocks[b_idx]
        
        for lz in 1:bs, ly in 1:bs, lx in 1:bs
            if obstacle_arr[lx, ly, lz, b_idx]
                continue  
            end
            
            is_near_wall = false
            min_dist = 100.0f0
            
            for dz in -1:1, dy in -1:1, dx_off in -1:1
                if dx_off == 0 && dy == 0 && dz == 0
                    continue
                end
                
                nx, ny, nz = lx + dx_off, ly + dy, lz + dz
                nb_idx = b_idx
                
                if nx < 1 || nx > bs || ny < 1 || ny > bs || nz < 1 || nz > bs
                    nbx = bx + (nx < 1 ? -1 : (nx > bs ? 1 : 0))
                    nby = by + (ny < 1 ? -1 : (ny > bs ? 1 : 0))
                    nbz = bz + (nz < 1 ? -1 : (nz > bs ? 1 : 0))
                    
                    if haskey(block_lookup, (nbx, nby, nbz))
                        nb_idx = block_lookup[(nbx, nby, nbz)]
                        nx = nx < 1 ? nx + bs : (nx > bs ? nx - bs : nx)
                        ny = ny < 1 ? ny + bs : (ny > bs ? ny - bs : ny)
                        nz = nz < 1 ? nz + bs : (nz > bs ? nz - bs : nz)
                    else
                        continue
                    end
                end
                
                if obstacle_arr[nx, ny, nz, nb_idx]
                    is_near_wall = true
                    dist = sqrt(Float32(dx_off^2 + dy^2 + dz^2)) * Float32(dx)
                    min_dist = min(min_dist, dist)
                end
            end
            
            if is_near_wall
                wall_dist_arr[lx, ly, lz, b_idx] = min_dist
                near_wall_count += 1
            end
        end
    end
    
    println("[Domain] Found $near_wall_count near-wall cells")
end
