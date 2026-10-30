"""
    report_bouzidi_link_symmetry(...)

Print a one-shot startup summary of Bouzidi wall-link counts for one level,
classified by lattice direction, by Z-hemisphere relative to the median Bouzidi
cell Z, and by behind-support kind (fluid, solid, out-of-range). Diagnostic
only — runs on CPU at level setup, no hot-path cost.

Motivation: pass162/163 left the YZ-slice alternating vorticity pattern
unchanged near the wing. Item 2 of the 2026-05-13 handoff
(`docs/2026-05-13_solver_status_handoff.md`) calls for a Bouzidi link-support
report to expose whether wall treatment itself is asymmetric across the wing
upper/lower surface.

The "behind" cell is the one sampled when the Bouzidi q<0.5 branch reads
`f_post_collision[x_ff]`. The direction stored in q_map for link k means the
wall is hit moving along c_k from the fluid cell; the behind sample is one
cell further along the OPP_k direction (i.e. one cell upstream of the fluid
cell along the inward-pointing normal).
"""
function report_bouzidi_link_symmetry(
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
)
    n_boundary_cells > 0 || return

    n_dirs = length(cx_arr)
    @assert size(bouzidi_q_map, 5) == n_dirs "q_map last dim must match n_dirs"

    # Collect global Z (in cell units) of each Bouzidi cell to find the median
    # split for the upper/lower hemisphere bucket. Use the global Z coordinate
    # (block_z - 1) * block_size + local_z so the split is geometry-independent.
    cell_z_global = Vector{Int}(undef, n_boundary_cells)
    @inbounds for ci in 1:n_boundary_cells
        b_idx = Int(bouzidi_cell_block[ci])
        bz = sorted_blocks[b_idx][3]
        lz = Int(bouzidi_cell_z[ci])
        cell_z_global[ci] = (bz - 1) * block_size + lz
    end
    z_median = isempty(cell_z_global) ? 0 : Int(round(median_int(cell_z_global)))

    # Bucket structure: dictionary keyed by direction label, with counts split
    # into (above, below) hemispheres and (fluid, solid, oob) behind-support.
    # Use fixed direction labels for D3Q27 / D3Q39 cardinal axes plus an
    # "other" bucket for shells with |c|>1 or oblique velocities.
    labels = ("+x", "-x", "+y", "-y", "+z", "-z", "other")
    # buckets[label][hemisphere][support] => count
    # hemisphere: 1 = above, 2 = below
    # support:    1 = fluid, 2 = solid (in-block obstacle), 3 = oob (no nb block)
    buckets = Dict{String, Array{Int, 2}}()
    for lab in labels
        buckets[lab] = zeros(Int, 2, 3)
    end
    # Separate counter for q>=0.5 links (no behind sample needed)
    high_q = Dict{String, Array{Int, 1}}()
    for lab in labels
        high_q[lab] = zeros(Int, 2)
    end

    @inline function _direction_label(cx::Int, cy::Int, cz::Int)
        # Cardinal axis = exactly one nonzero component with |c|=1
        ax = abs(cx); ay = abs(cy); az = abs(cz)
        if ax == 1 && ay == 0 && az == 0
            return cx > 0 ? "+x" : "-x"
        elseif ay == 1 && ax == 0 && az == 0
            return cy > 0 ? "+y" : "-y"
        elseif az == 1 && ax == 0 && ay == 0
            return cz > 0 ? "+z" : "-z"
        else
            return "other"
        end
    end

    bs = block_size
    total_q_low = 0
    total_q_high = 0
    total_links = 0

    @inbounds for ci in 1:n_boundary_cells
        b_idx = Int(bouzidi_cell_block[ci])
        lx = Int(bouzidi_cell_x[ci])
        ly = Int(bouzidi_cell_y[ci])
        lz = Int(bouzidi_cell_z[ci])
        zg = cell_z_global[ci]
        hemi = zg >= z_median ? 1 : 2

        for k in 1:n_dirs
            q = Float32(bouzidi_q_map[lx, ly, lz, b_idx, k])
            q > q_min && q <= 1.0f0 || continue
            total_links += 1

            cx = Int(cx_arr[k]); cy = Int(cy_arr[k]); cz = Int(cz_arr[k])
            lab = _direction_label(cx, cy, cz)

            if q < 0.5f0
                total_q_low += 1
                # Behind cell = x_f + c_{opp_k}
                opp_k = Int(opp_arr[k])
                ncx = Int(cx_arr[opp_k]); ncy = Int(cy_arr[opp_k]); ncz = Int(cz_arr[opp_k])
                nx = lx + ncx; ny = ly + ncy; nz_ = lz + ncz

                support = 3  # default = oob
                if nx >= 1 && nx <= bs && ny >= 1 && ny <= bs && nz_ >= 1 && nz_ <= bs
                    support = obstacle_arr[nx, ny, nz_, b_idx] ? 2 : 1
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
                        support = obstacle_arr[nnx, nny, nnz, nb_block] ? 2 : 1
                    end
                end

                buckets[lab][hemi, support] += 1
            else
                total_q_high += 1
                high_q[lab][hemi] += 1
            end
        end
    end

    println()
    @printf("[BouzidiSym] Level %d: %d boundary cells, %d active wall links\n",
            level_id, n_boundary_cells, total_links)
    @printf("[BouzidiSym]   z_global median = %d (hemisphere split)\n", z_median)
    @printf("[BouzidiSym]   q<0.5 (behind-sampled): %d   q>=0.5 (mirror): %d\n",
            total_q_low, total_q_high)
    println("[BouzidiSym]   Direction | Hemi  |   q<0.5 fluid /   solid /     oob |  q>=0.5")
    println("[BouzidiSym]   ----------|-------|----------------------------------|--------")
    for lab in labels
        for (hemi_i, hemi_name) in ((1, "above"), (2, "below"))
            n_f, n_s, n_o = buckets[lab][hemi_i, 1], buckets[lab][hemi_i, 2], buckets[lab][hemi_i, 3]
            n_hq = high_q[lab][hemi_i]
            (n_f + n_s + n_o + n_hq) > 0 || continue
            @printf("[BouzidiSym]   %-9s | %-5s |    %8d / %8d / %8d | %8d\n",
                    lab, hemi_name, n_f, n_s, n_o, n_hq)
        end
    end

    # Asymmetry summary: cardinal axes only, q<0.5 fluid+solid+oob totals.
    println("[BouzidiSym]   Hemisphere asymmetry on cardinal q<0.5 links (above-below)/total:")
    println("[BouzidiSym]   Direction | total above | total below | (a-b)/total |  solid frac above / below")
    println("[BouzidiSym]   ----------|-------------|-------------|-------------|----------------------------")
    for lab in ("+x", "-x", "+y", "-y", "+z", "-z")
        b = buckets[lab]
        ta = b[1, 1] + b[1, 2] + b[1, 3]
        tb = b[2, 1] + b[2, 2] + b[2, 3]
        tot = ta + tb
        tot > 0 || continue
        sa = ta > 0 ? Float64(b[1, 2] + b[1, 3]) / ta : 0.0
        sb = tb > 0 ? Float64(b[2, 2] + b[2, 3]) / tb : 0.0
        asym = (ta - tb) / max(tot, 1)
        @printf("[BouzidiSym]   %-9s | %11d | %11d | %+9.4f   |  %7.4f / %7.4f\n",
                lab, ta, tb, asym, sa, sb)
    end
    println()
    return
end

# Plain integer median (no Statistics dep needed at this point in load order).
@inline function median_int(v::AbstractVector{<:Integer})
    isempty(v) && return 0
    s = sort(v)
    n = length(s)
    return n % 2 == 1 ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) ÷ 2
end
