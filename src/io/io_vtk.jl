using WriteVTK
using Printf
using Base.Threads


function export_merged_mesh_sync(t_step, grids, out_dir, backend;
                                 cx_gpu=nothing, cy_gpu=nothing, cz_gpu=nothing, w_gpu=nothing,
                                 include_level::Bool=true)
    log_walltime("VTK Export START for step $t_step")
    vtk_start = time()

    # Optional Π_neq diagnostic — same gating rule as slice exporter.
    compute_pineq = COMPUTE_PI_NEQ && cx_gpu !== nothing && cy_gpu !== nothing &&
                    cz_gpu !== nothing && w_gpu !== nothing
    shift_x_f = Float32(LATTICE_SHIFT_X)
    shift_y_f = Float32(LATTICE_SHIFT_Y)
    shift_z_f = Float32(LATTICE_SHIFT_Z)
    
    level_blocks = [Set{Tuple{Int,Int,Int}}() for _ in 1:length(grids)]
    for lvl in 1:length(grids)
        for b in grids[lvl].active_block_coords
            push!(level_blocks[lvl], b)
        end
    end
    
    valid_blocks = Vector{NamedTuple{(:lvl, :b_idx, :bx, :by, :bz), Tuple{Int, Int, Int, Int, Int}}}()
    
    for lvl in 1:length(grids)
        next_blocks = lvl < length(grids) ? level_blocks[lvl+1] : nothing
        for (b_idx, (bx, by, bz)) in enumerate(grids[lvl].active_block_coords)
            should_export = true
            if next_blocks !== nothing
                children = 0
                for dbz in 0:1, dby in 0:1, dbx in 0:1
                    if (2*bx-1+dbx, 2*by-1+dby, 2*bz-1+dbz) in next_blocks
                        children += 1
                    end
                end
                if children == 8
                    should_export = false
                end
            end
            if should_export
                push!(valid_blocks, (lvl=lvl, b_idx=b_idx, bx=bx, by=by, bz=bz))
            end
        end
    end
    
    if isempty(valid_blocks); return; end
    
    level_data = Dict{Int, NamedTuple}()
    ddf_active = DDF_ENABLED
    for lvl in 1:length(grids)
        level = grids[lvl]
        rho_cpu = Array(level.rho)
        # Level 1 uses global step parity; levels 2+ always end on odd sub-step → vel
        vel_cpu = (lvl == 1 && iseven(t_step)) ? Array(level.vel_temp) : Array(level.vel)
        obs_cpu = Array(level.obstacle)
        temp_cpu = (ddf_active && has_thermal(level)) ? Array(level.temperature) : nothing
        sponge_cpu = hasproperty(level, :sponge) ? Array(level.sponge) : nothing

        # Π_neq magnitude per cell (τ-corrected to pre-collision form, invariant across levels)
        pinq_mag_cpu = nothing
        # (ET/AA-pattern single-buffer path removed 2026-06-24 with the D3Q19 retirement.)
        if compute_pineq
            f_current = (lvl == 1 && iseven(t_step)) ? level.f_temp : level.f
            pi_neq_gpu = compute_pi_neq_for_level!(level, f_current,
                                                   cx_gpu, cy_gpu, cz_gpu, w_gpu, backend;
                                                   shift_x=shift_x_f, shift_y=shift_y_f, shift_z=shift_z_f)
            pi_neq_cpu = Array(pi_neq_gpu)
            tau_lvl = Float32(level.tau)
            denom = tau_lvl - 1.0f0
            pre_scale = (abs(denom) < 1.0f-3) ? 1.0f0 : (tau_lvl / denom)
            abs_scale = abs(pre_scale)
            n_blocks_lvl = size(pi_neq_cpu, 4)
            pinq_mag_cpu = Array{Float32}(undef, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n_blocks_lvl)
            @inbounds for b in 1:n_blocks_lvl, lz in 1:BLOCK_SIZE, ly in 1:BLOCK_SIZE, lx in 1:BLOCK_SIZE
                pxx = pi_neq_cpu[lx, ly, lz, b, 1]
                pyy = pi_neq_cpu[lx, ly, lz, b, 2]
                pzz = pi_neq_cpu[lx, ly, lz, b, 3]
                pxy = pi_neq_cpu[lx, ly, lz, b, 4]
                pxz = pi_neq_cpu[lx, ly, lz, b, 5]
                pyz = pi_neq_cpu[lx, ly, lz, b, 6]
                mag_post = sqrt(pxx*pxx + pyy*pyy + pzz*pzz +
                                2.0f0*(pxy*pxy + pxz*pxz + pyz*pyz))
                pinq_mag_cpu[lx, ly, lz, b] = abs_scale * mag_post
            end
        end

        # wall_dist sentinel-cell-band field (see domain_generation.jl: only
        # populated for fluid cells adjacent to the wall; far-field cells and
        # obstacle cells keep the sentinel value 100.0f0). Threshold display in
        # ParaView to >0 && <50 to hide the sentinel.
        wall_dist_cpu = (OUTPUT_WALL_DISTANCE && level.wall_dist !== nothing && length(level.wall_dist) > 1) ?
                        Array(level.wall_dist) : nothing

        level_data[lvl] = (rho=rho_cpu, vel=vel_cpu, obs=obs_cpu, dx=Float32(level.dx),
                           temperature=temp_cpu, pi_neq_mag=pinq_mag_cpu,
                           sponge=sponge_cpu,
                           wall_dist=wall_dist_cpu)
    end
    
    n_total = length(valid_blocks)
    n_pts = (BLOCK_SIZE + 1)^3
    n_cells = BLOCK_SIZE^3
    
    points = Matrix{Float32}(undef, 3, n_total * n_pts)
    cells = Vector{MeshCell}(undef, n_total * n_cells)
    rho_arr = Vector{Float32}(undef, n_total * n_cells)
    vel_mat = Matrix{Float32}(undef, 3, n_total * n_cells)
    obst_arr = Vector{UInt8}(undef, n_total * n_cells)
    level_arr = Vector{Int32}(undef, n_total * n_cells)
    temp_arr = (ddf_active || (EOS_THERMODYNAMIC && LATTICE_TYPE == :d3q39)) ?
               Vector{Float32}(undef, n_total * n_cells) : nothing
    pinq_arr = compute_pineq ? Vector{Float32}(undef, n_total * n_cells) : nothing
    # Slice-mirror fields: mach + pressure + wall_distance are cheap pointwise;
    # vorticity (3D vector, units 1/dt) and schlieren (|∇ρ|, units 1/m) need
    # central-difference gradients. We compute the gradient WITHIN a block and
    # fall back to one-sided diff at block edges (so values at block boundaries
    # have small bias; ParaView users who care can run a Gradient filter on
    # Density/Velocity instead to get AMR-aware results).
    mach_arr     = OUTPUT_MACH      ? Vector{Float32}(undef, n_total * n_cells) : nothing
    press_arr    = OUTPUT_PRESSURE  ? Vector{Float32}(undef, n_total * n_cells) : nothing
    wdist_arr    = OUTPUT_WALL_DISTANCE ? Vector{Float32}(undef, n_total * n_cells) : nothing
    vort_mat     = OUTPUT_VORTICITY ? Matrix{Float32}(undef, 3, n_total * n_cells) : nothing
    schlieren_arr= OUTPUT_SCHLIEREN ? Vector{Float32}(undef, n_total * n_cells) : nothing
    cs2_phys     = Float32(CS2_PHYSICS)
    pressure_coeff = Float32(diagnostic_pressure_coeff())
    u_ref_vtk = abs(Float64(U_INLET_CURRENT[])) > 0.0 ?
                abs(Float64(U_INLET_CURRENT[])) :
                abs(Float64(U_TARGET))
    
    Threads.@threads for i in 1:n_total
        blk = valid_blocks[i]
        data = level_data[blk.lvl]
        dx = data.dx
        
        pt_off = (i - 1) * n_pts
        cell_off = (i - 1) * n_cells
        off_x = Float32((blk.bx - 1) * BLOCK_SIZE)
        off_y = Float32((blk.by - 1) * BLOCK_SIZE)
        off_z = Float32((blk.bz - 1) * BLOCK_SIZE)
        
        for pz in 0:BLOCK_SIZE, py in 0:BLOCK_SIZE, px in 0:BLOCK_SIZE
            pidx = pt_off + 1 + px + py*(BLOCK_SIZE+1) + pz*(BLOCK_SIZE+1)^2
            points[1, pidx] = (off_x + px) * dx
            points[2, pidx] = (off_y + py) * dx
            points[3, pidx] = (off_z + pz) * dx
        end
        
        stride_y = BLOCK_SIZE + 1
        stride_z = (BLOCK_SIZE + 1)^2
        
        inv_dx = 1.0f0 / dx
        for z in 1:BLOCK_SIZE, y in 1:BLOCK_SIZE, x in 1:BLOCK_SIZE
            cidx = cell_off + x + (y-1)*BLOCK_SIZE + (z-1)*BLOCK_SIZE^2
            pt_base = pt_off + 1 + (x-1) + (y-1)*stride_y + (z-1)*stride_z
            cells[cidx] = MeshCell(VTKCellTypes.VTK_VOXEL,
                (pt_base, pt_base+1, pt_base+stride_y, pt_base+stride_y+1,
                 pt_base+stride_z, pt_base+stride_z+1, pt_base+stride_z+stride_y, pt_base+stride_z+stride_y+1))

            vx = data.vel[x, y, z, blk.b_idx, 1]
            vy = data.vel[x, y, z, blk.b_idx, 2]
            vz = data.vel[x, y, z, blk.b_idx, 3]
            vel_mat[1, cidx] = vx
            vel_mat[2, cidx] = vy
            vel_mat[3, cidx] = vz
            rho_val = data.rho[x, y, z, blk.b_idx]
            rho_arr[cidx] = rho_val
            obst_arr[cidx] = data.obs[x, y, z, blk.b_idx] ? 0x01 : 0x00
            level_arr[cidx] = Int32(blk.lvl)
            stored_T = data.temperature !== nothing ? data.temperature[x, y, z, blk.b_idx] : NaN32
            sp = data.sponge !== nothing ? data.sponge[x, y, z, blk.b_idx] : 0.0f0
            t_val = isfinite(stored_T) ? stored_T :
                    Float32(diagnostic_effective_temperature(
                        rho_val, vx, vy, vz, NaN, sp, u_ref_vtk))
            if temp_arr !== nothing
                temp_arr[cidx] = t_val
            end
            if pinq_arr !== nothing && data.pi_neq_mag !== nothing
                pinq_arr[cidx] = data.pi_neq_mag[x, y, z, blk.b_idx]
            end
            if mach_arr !== nothing
                mach_arr[cidx] = sqrt(vx*vx + vy*vy + vz*vz) /
                                  sqrt(max(cs2_phys * t_val, 1.0f-12))
            end
            if press_arr !== nothing
                press_arr[cidx] = rho_val * pressure_coeff * t_val
            end
            if wdist_arr !== nothing
                wdist_arr[cidx] = data.wall_dist !== nothing ?
                                  data.wall_dist[x, y, z, blk.b_idx] : NaN32
            end
            # ── Block-local central-difference gradients ──
            # 2-point central if interior, one-sided at the block face. Does
            # NOT cross block boundaries, so values at block edges have a
            # known small bias; rerun ParaView's Gradient filter on Velocity
            # / Density if you need fully AMR-aware derivatives.
            if vort_mat !== nothing || schlieren_arr !== nothing
                # X-gradient
                if 1 < x < BLOCK_SIZE
                    inv_2dx = 0.5f0 * inv_dx
                    dvx_dx = (data.vel[x+1, y, z, blk.b_idx, 1] - data.vel[x-1, y, z, blk.b_idx, 1]) * inv_2dx
                    dvy_dx = (data.vel[x+1, y, z, blk.b_idx, 2] - data.vel[x-1, y, z, blk.b_idx, 2]) * inv_2dx
                    dvz_dx = (data.vel[x+1, y, z, blk.b_idx, 3] - data.vel[x-1, y, z, blk.b_idx, 3]) * inv_2dx
                    drho_dx = (data.rho[x+1, y, z, blk.b_idx] - data.rho[x-1, y, z, blk.b_idx]) * inv_2dx
                elseif x == 1
                    dvx_dx = (data.vel[x+1, y, z, blk.b_idx, 1] - vx) * inv_dx
                    dvy_dx = (data.vel[x+1, y, z, blk.b_idx, 2] - vy) * inv_dx
                    dvz_dx = (data.vel[x+1, y, z, blk.b_idx, 3] - vz) * inv_dx
                    drho_dx = (data.rho[x+1, y, z, blk.b_idx] - rho_val) * inv_dx
                else
                    dvx_dx = (vx - data.vel[x-1, y, z, blk.b_idx, 1]) * inv_dx
                    dvy_dx = (vy - data.vel[x-1, y, z, blk.b_idx, 2]) * inv_dx
                    dvz_dx = (vz - data.vel[x-1, y, z, blk.b_idx, 3]) * inv_dx
                    drho_dx = (rho_val - data.rho[x-1, y, z, blk.b_idx]) * inv_dx
                end
                # Y-gradient
                if 1 < y < BLOCK_SIZE
                    inv_2dx = 0.5f0 * inv_dx
                    dvx_dy = (data.vel[x, y+1, z, blk.b_idx, 1] - data.vel[x, y-1, z, blk.b_idx, 1]) * inv_2dx
                    dvy_dy = (data.vel[x, y+1, z, blk.b_idx, 2] - data.vel[x, y-1, z, blk.b_idx, 2]) * inv_2dx
                    dvz_dy = (data.vel[x, y+1, z, blk.b_idx, 3] - data.vel[x, y-1, z, blk.b_idx, 3]) * inv_2dx
                    drho_dy = (data.rho[x, y+1, z, blk.b_idx] - data.rho[x, y-1, z, blk.b_idx]) * inv_2dx
                elseif y == 1
                    dvx_dy = (data.vel[x, y+1, z, blk.b_idx, 1] - vx) * inv_dx
                    dvy_dy = (data.vel[x, y+1, z, blk.b_idx, 2] - vy) * inv_dx
                    dvz_dy = (data.vel[x, y+1, z, blk.b_idx, 3] - vz) * inv_dx
                    drho_dy = (data.rho[x, y+1, z, blk.b_idx] - rho_val) * inv_dx
                else
                    dvx_dy = (vx - data.vel[x, y-1, z, blk.b_idx, 1]) * inv_dx
                    dvy_dy = (vy - data.vel[x, y-1, z, blk.b_idx, 2]) * inv_dx
                    dvz_dy = (vz - data.vel[x, y-1, z, blk.b_idx, 3]) * inv_dx
                    drho_dy = (rho_val - data.rho[x, y-1, z, blk.b_idx]) * inv_dx
                end
                # Z-gradient
                if 1 < z < BLOCK_SIZE
                    inv_2dx = 0.5f0 * inv_dx
                    dvx_dz = (data.vel[x, y, z+1, blk.b_idx, 1] - data.vel[x, y, z-1, blk.b_idx, 1]) * inv_2dx
                    dvy_dz = (data.vel[x, y, z+1, blk.b_idx, 2] - data.vel[x, y, z-1, blk.b_idx, 2]) * inv_2dx
                    dvz_dz = (data.vel[x, y, z+1, blk.b_idx, 3] - data.vel[x, y, z-1, blk.b_idx, 3]) * inv_2dx
                    drho_dz = (data.rho[x, y, z+1, blk.b_idx] - data.rho[x, y, z-1, blk.b_idx]) * inv_2dx
                elseif z == 1
                    dvx_dz = (data.vel[x, y, z+1, blk.b_idx, 1] - vx) * inv_dx
                    dvy_dz = (data.vel[x, y, z+1, blk.b_idx, 2] - vy) * inv_dx
                    dvz_dz = (data.vel[x, y, z+1, blk.b_idx, 3] - vz) * inv_dx
                    drho_dz = (data.rho[x, y, z+1, blk.b_idx] - rho_val) * inv_dx
                else
                    dvx_dz = (vx - data.vel[x, y, z-1, blk.b_idx, 1]) * inv_dx
                    dvy_dz = (vy - data.vel[x, y, z-1, blk.b_idx, 2]) * inv_dx
                    dvz_dz = (vz - data.vel[x, y, z-1, blk.b_idx, 3]) * inv_dx
                    drho_dz = (rho_val - data.rho[x, y, z-1, blk.b_idx]) * inv_dx
                end
                if vort_mat !== nothing
                    vort_mat[1, cidx] = dvz_dy - dvy_dz
                    vort_mat[2, cidx] = dvx_dz - dvz_dx
                    vort_mat[3, cidx] = dvy_dx - dvx_dy
                end
                if schlieren_arr !== nothing
                    schlieren_arr[cidx] = sqrt(drho_dx*drho_dx + drho_dy*drho_dy + drho_dz*drho_dz)
                end
            end
        end
    end
    
    replace!(rho_arr, NaN32 => 0f0, Inf32 => 0f0, -Inf32 => 0f0)
    replace!(vel_mat, NaN32 => 0f0, Inf32 => 0f0, -Inf32 => 0f0)
    if temp_arr !== nothing
        replace!(temp_arr, NaN32 => 0f0, Inf32 => 0f0, -Inf32 => 0f0)
    end
    if pinq_arr !== nothing
        replace!(pinq_arr, NaN32 => 0f0, Inf32 => 0f0, -Inf32 => 0f0)
    end
    if mach_arr !== nothing
        replace!(mach_arr, NaN32 => 0f0, Inf32 => 0f0, -Inf32 => 0f0)
    end
    if press_arr !== nothing
        replace!(press_arr, NaN32 => 0f0, Inf32 => 0f0, -Inf32 => 0f0)
    end
    # WallDistance: leave sentinel 100.0f0 intact so the user can threshold it
    # in ParaView (display only values < 50 to see the band). NaN→0 just in
    # case some level had no wall_dist array allocated.
    if wdist_arr !== nothing
        replace!(wdist_arr, NaN32 => 0f0, Inf32 => 0f0, -Inf32 => 0f0)
    end
    if vort_mat !== nothing
        replace!(vort_mat, NaN32 => 0f0, Inf32 => 0f0, -Inf32 => 0f0)
    end
    if schlieren_arr !== nothing
        replace!(schlieren_arr, NaN32 => 0f0, Inf32 => 0f0, -Inf32 => 0f0)
    end

    filename = @sprintf("%s/flow_%06d", out_dir, t_step)
    try
        vtk_grid(filename, points, cells; compress=true, append=false) do vtk
            if OUTPUT_DENSITY; vtk["Density"] = rho_arr; end
            if OUTPUT_VELOCITY; vtk["Velocity"] = vel_mat; end
            if OUTPUT_VEL_MAG; vtk["VelocityMagnitude"] = sqrt.(vel_mat[1,:].^2 .+ vel_mat[2,:].^2 .+ vel_mat[3,:].^2); end
            if OUTPUT_OBSTACLE; vtk["Obstacle"] = obst_arr; end
            if OUTPUT_LEVEL && include_level; vtk["Level"] = level_arr; end
            if temp_arr !== nothing; vtk["Temperature"] = temp_arr; end
            if pinq_arr !== nothing; vtk["PiNeqMag"] = pinq_arr; end
            if mach_arr !== nothing; vtk["Mach"] = mach_arr; end
            if press_arr !== nothing; vtk["Pressure"] = press_arr; end
            if wdist_arr !== nothing; vtk["WallDistance"] = wdist_arr; end
            if vort_mat !== nothing; vtk["Vorticity"] = vort_mat; end
            if schlieren_arr !== nothing; vtk["Schlieren"] = schlieren_arr; end
        end
    catch e
        println("[Error] VTK write failed: $e")
    end
    
    log_walltime(@sprintf("VTK Export END (%.2fM cells, %.1fs)", n_total*n_cells/1e6, time()-vtk_start))
end
