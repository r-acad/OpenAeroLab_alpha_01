"""
    report_bouzidi_missed_links(...)

Pass 193. One-shot CPU audit for Bouzidi boundary cells where a lattice
    direction has no valid q-link (`q <= q_min`), the opposite q-link is also
    invalid, and the normal streaming upstream cell (`x_f - c_k`) is solid.
    If `q[opp_k]` is valid, the Bouzidi kernel writes population k through the
    opposite link, so it is not a missed streaming population.

Outputs (in OUT_DIR):
  - bouzidi_missed_link_levelN_direction.csv
  - bouzidi_missed_link_levelN_cells.csv

Diagnostic-only. No solver behavior change.
"""
function report_bouzidi_missed_links(
    level_id::Int,
    bouzidi_q_map::AbstractArray{<:Real, 5},
    bouzidi_cell_block::AbstractVector{<:Integer},
    bouzidi_cell_x::AbstractVector{<:Integer},
    bouzidi_cell_y::AbstractVector{<:Integer},
    bouzidi_cell_z::AbstractVector{<:Integer},
    n_boundary_cells::Int,
    obstacle_arr::AbstractArray{Bool, 4},
    nb_table::AbstractMatrix{<:Integer},
    sorted_blocks::AbstractVector,
    block_size::Int,
    cx_arr::AbstractVector{<:Integer},
    cy_arr::AbstractVector{<:Integer},
    cz_arr::AbstractVector{<:Integer},
    opp_arr::AbstractVector{<:Integer},
    q_min::Float32,
    output_dir::AbstractString,
)
    n_boundary_cells > 0 || return

    n_dirs = length(cx_arr)
    @assert size(bouzidi_q_map, 5) == n_dirs "q_map last dim must match n_dirs"

    n_valid = zeros(Int, n_dirs)
    n_skipped = zeros(Int, n_dirs)
    n_skipped_covered_by_opp = zeros(Int, n_dirs)
    n_uncovered = zeros(Int, n_dirs)
    n_upstream_fluid = zeros(Int, n_dirs)
    n_upstream_solid = zeros(Int, n_dirs)
    n_upstream_oob = zeros(Int, n_dirs)
    n_upstream_nbr_fluid = zeros(Int, n_dirs)
    n_upstream_nbr_solid = zeros(Int, n_dirs)

    cell_idx_buf = Int[]
    cell_xg_buf = Int[]
    cell_yg_buf = Int[]
    cell_zg_buf = Int[]
    cell_n_skipped_buf = Int[]
    cell_n_solid_buf = Int[]
    cell_dirs_buf = String[]

    bs = block_size

    @inbounds for ci in 1:n_boundary_cells
        b_idx = Int(bouzidi_cell_block[ci])
        bx = sorted_blocks[b_idx][1]
        by = sorted_blocks[b_idx][2]
        bz = sorted_blocks[b_idx][3]
        lx = Int(bouzidi_cell_x[ci])
        ly = Int(bouzidi_cell_y[ci])
        lz = Int(bouzidi_cell_z[ci])

        skipped_this = 0
        solid_this = 0
        solid_dirs = Int[]

        for k in 1:n_dirs
            q = Float32(bouzidi_q_map[lx, ly, lz, b_idx, k])
            if q > q_min && q <= 1.0f0
                n_valid[k] += 1
                continue
            end

            n_skipped[k] += 1
            opp_k = Int(opp_arr[k])
            q_opp = Float32(bouzidi_q_map[lx, ly, lz, b_idx, opp_k])
            if q_opp > q_min && q_opp <= 1.0f0
                n_skipped_covered_by_opp[k] += 1
                continue
            end

            n_uncovered[k] += 1
            skipped_this += 1

            # Streaming into direction k at x_f samples x_f - c_k.
            nx = lx - Int(cx_arr[k])
            ny = ly - Int(cy_arr[k])
            nz_ = lz - Int(cz_arr[k])

            if nx >= 1 && nx <= bs && ny >= 1 && ny <= bs && nz_ >= 1 && nz_ <= bs
                if obstacle_arr[nx, ny, nz_, b_idx]
                    n_upstream_solid[k] += 1
                    solid_this += 1
                    push!(solid_dirs, k)
                else
                    n_upstream_fluid[k] += 1
                end
            else
                off_x = nx < 1 ? -1 : (nx > bs ? 1 : 0)
                off_y = ny < 1 ? -1 : (ny > bs ? 1 : 0)
                off_z = nz_ < 1 ? -1 : (nz_ > bs ? 1 : 0)
                dir_idx = (off_x + 1) + (off_y + 1) * 3 + (off_z + 1) * 9 + 1
                nb_block = Int(nb_table[b_idx, dir_idx])

                if nb_block > 0
                    nnx = nx < 1 ? nx + bs : (nx > bs ? nx - bs : nx)
                    nny = ny < 1 ? ny + bs : (ny > bs ? ny - bs : ny)
                    nnz = nz_ < 1 ? nz_ + bs : (nz_ > bs ? nz_ - bs : nz_)
                    if obstacle_arr[nnx, nny, nnz, nb_block]
                        n_upstream_nbr_solid[k] += 1
                        solid_this += 1
                        push!(solid_dirs, k)
                    else
                        n_upstream_nbr_fluid[k] += 1
                    end
                else
                    n_upstream_oob[k] += 1
                end
            end
        end

        if solid_this > 0
            push!(cell_idx_buf, ci)
            push!(cell_xg_buf, (bx - 1) * bs + lx)
            push!(cell_yg_buf, (by - 1) * bs + ly)
            push!(cell_zg_buf, (bz - 1) * bs + lz)
            push!(cell_n_skipped_buf, skipped_this)
            push!(cell_n_solid_buf, solid_this)
            push!(cell_dirs_buf, join(solid_dirs, ";"))
        end
    end

    tot_skipped = sum(n_skipped)
    tot_uncovered = sum(n_uncovered)
    tot_solid = sum(n_upstream_solid) + sum(n_upstream_nbr_solid)
    n_cells_with_solid = length(cell_idx_buf)
    frac = tot_uncovered > 0 ? Float64(tot_solid) / Float64(tot_uncovered) : 0.0

    println()
    @printf("[BouzidiMissed] Level %d: %d boundary cells, %d skipped q-links, %d uncovered skipped q-links, %d solid-upstream missed links\n",
            level_id, n_boundary_cells, tot_skipped, tot_uncovered, tot_solid)
    @printf("[BouzidiMissed]   solid-upstream missed-link frac: %.6f\n", frac)
    @printf("[BouzidiMissed]   cells with >=1 solid-upstream skipped link: %d / %d = %.3f%%\n",
            n_cells_with_solid, n_boundary_cells,
            100.0 * Float64(n_cells_with_solid) / Float64(n_boundary_cells))

    dir_path = joinpath(output_dir, @sprintf("bouzidi_missed_link_level%d_direction.csv", level_id))
    open(dir_path, "w") do io
        println(io, "k,cx,cy,cz,opp_k,n_valid_q,n_skipped_q,n_skipped_covered_by_opp,n_uncovered_skipped,n_upstream_fluid,n_upstream_nbr_fluid,n_upstream_solid,n_upstream_nbr_solid,n_upstream_oob_no_nbr,n_missed_solid,missed_frac_uncovered")
        for k in 1:n_dirs
            missed = n_upstream_solid[k] + n_upstream_nbr_solid[k]
            kfrac = n_uncovered[k] > 0 ? Float64(missed) / Float64(n_uncovered[k]) : 0.0
            @printf(io, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.8f\n",
                    k, Int(cx_arr[k]), Int(cy_arr[k]), Int(cz_arr[k]), Int(opp_arr[k]),
                    n_valid[k], n_skipped[k], n_skipped_covered_by_opp[k], n_uncovered[k],
                    n_upstream_fluid[k], n_upstream_nbr_fluid[k],
                    n_upstream_solid[k], n_upstream_nbr_solid[k],
                    n_upstream_oob[k], missed, kfrac)
        end
    end
    println("[BouzidiMissed]   Wrote per-direction CSV: ", dir_path)

    cell_path = joinpath(output_dir, @sprintf("bouzidi_missed_link_level%d_cells.csv", level_id))
    open(cell_path, "w") do io
        println(io, "cell_idx,x_global,y_global,z_global,n_uncovered_skipped,n_missed_solid,solid_dirs")
        for i in 1:n_cells_with_solid
            @printf(io, "%d,%d,%d,%d,%d,%d,%s\n",
                    cell_idx_buf[i], cell_xg_buf[i], cell_yg_buf[i], cell_zg_buf[i],
                    cell_n_skipped_buf[i], cell_n_solid_buf[i], cell_dirs_buf[i])
        end
    end
    println("[BouzidiMissed]   Wrote per-cell CSV: ", cell_path,
            "  (", n_cells_with_solid, " cells)")

    println("[BouzidiMissed]   Top missed-link directions:")
    println("[BouzidiMissed]   k  | (cx,cy,cz) | uncover | solid | nbr_solid | frac")
    println("[BouzidiMissed]   ---|------------|---------|-------|-----------|------")
    order = sortperm(1:n_dirs; by = k -> n_upstream_solid[k] + n_upstream_nbr_solid[k], rev = true)
    shown = 0
    for k in order
        missed = n_upstream_solid[k] + n_upstream_nbr_solid[k]
        missed > 0 || continue
        kfrac = n_uncovered[k] > 0 ? Float64(missed) / Float64(n_uncovered[k]) : 0.0
        @printf("[BouzidiMissed]   %2d | (%+d,%+d,%+d) | %7d | %5d | %9d | %.4f\n",
                k, Int(cx_arr[k]), Int(cy_arr[k]), Int(cz_arr[k]),
                n_uncovered[k], n_upstream_solid[k], n_upstream_nbr_solid[k], kfrac)
        shown += 1
        shown >= 10 && break
    end
    println()
    return
end
