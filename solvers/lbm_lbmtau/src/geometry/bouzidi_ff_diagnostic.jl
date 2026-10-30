"""
    report_bouzidi_ff_fallback(...)

Pass 192. One-shot CPU audit of the Bouzidi q<0.5 "behind-cell" fallback in
[bouzidi_kernel.jl:58-79]. For every Bouzidi cell on this level, for every
direction k with `q_min < q[k] < 0.5`, classify the support of `x_ff`
(the cell at x_f + c_{opp_k}), which is what the kernel reads as `f_ff`:

  - intra_fluid       same block, not obstacle                — good (full BFL)
  - nbr_fluid         neighbor block, not obstacle            — good (full BFL)
  - intra_solid       same block, IS obstacle                 — FALLBACK (f_ff := f_k)
  - nbr_solid         neighbor block, IS obstacle             — FALLBACK
  - oob_no_neighbor   outside block AND no neighbor block     — FALLBACK

The fallback collapses the q<0.5 branch to `f_out[opp_k] = f_k`, i.e. a
direction-pair-asymmetric copy. When fallback hits some cells along a
wall but not others, the wall BC injects a sparse cell-to-cell pattern
into the macroscopic mass and momentum field — content variable_sigma
(h-mode damping) cannot reach, because the injection is not in the
mirror-antisymmetric channel.

Outputs (in OUT_DIR):
  - bouzidi_ff_audit_levelN_direction.csv   per-direction counts
  - bouzidi_ff_audit_levelN_cells.csv       per-cell fallback fingerprint
                                            (only cells with ≥1 fallback)
  - stdout summary table

Diagnostic-only. No solver behavior change.
"""
function report_bouzidi_ff_fallback(
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

    # Per-direction counters: q_low_total, ff_intra_fluid, ff_nbr_fluid,
    # ff_intra_solid, ff_nbr_solid, ff_oob_no_nbr.
    n_q_low      = zeros(Int, n_dirs)
    n_q_high     = zeros(Int, n_dirs)
    n_intra_f    = zeros(Int, n_dirs)
    n_nbr_f      = zeros(Int, n_dirs)
    n_intra_s    = zeros(Int, n_dirs)
    n_nbr_s      = zeros(Int, n_dirs)
    n_oob_no_nbr = zeros(Int, n_dirs)

    # Per-cell fallback fingerprint, only stored if ≥1 link in this cell
    # triggers fallback. (cell_idx, x_global, y_global, z_global,
    # n_links_total, n_links_q_low, n_fallback)
    cell_idx_buf      = Int[]
    cell_xg_buf       = Int[]
    cell_yg_buf       = Int[]
    cell_zg_buf       = Int[]
    cell_n_links_buf  = Int[]
    cell_n_qlow_buf   = Int[]
    cell_n_fb_buf     = Int[]

    bs = block_size

    @inbounds for ci in 1:n_boundary_cells
        b_idx = Int(bouzidi_cell_block[ci])
        bx = sorted_blocks[b_idx][1]
        by = sorted_blocks[b_idx][2]
        bz = sorted_blocks[b_idx][3]
        lx = Int(bouzidi_cell_x[ci])
        ly = Int(bouzidi_cell_y[ci])
        lz = Int(bouzidi_cell_z[ci])

        n_links_this   = 0
        n_qlow_this    = 0
        n_fb_this      = 0

        for k in 1:n_dirs
            q = Float32(bouzidi_q_map[lx, ly, lz, b_idx, k])
            q > q_min && q <= 1.0f0 || continue
            n_links_this += 1

            if q >= 0.5f0
                n_q_high[k] += 1
                continue
            end

            # q < 0.5 branch — reads f_ff at x_f + c_{opp_k}
            n_q_low[k] += 1
            n_qlow_this += 1
            opp_k = Int(opp_arr[k])
            ncx = Int(cx_arr[opp_k]); ncy = Int(cy_arr[opp_k]); ncz = Int(cz_arr[opp_k])
            nx = lx + ncx; ny = ly + ncy; nz_ = lz + ncz

            if nx >= 1 && nx <= bs && ny >= 1 && ny <= bs && nz_ >= 1 && nz_ <= bs
                # Inside the current block
                if obstacle_arr[nx, ny, nz_, b_idx]
                    n_intra_s[k] += 1
                    n_fb_this += 1
                else
                    n_intra_f[k] += 1
                end
            else
                # Outside block — check neighbor block table
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
                        n_nbr_s[k] += 1
                        n_fb_this += 1
                    else
                        n_nbr_f[k] += 1
                    end
                else
                    n_oob_no_nbr[k] += 1
                    n_fb_this += 1
                end
            end
        end

        if n_fb_this > 0
            push!(cell_idx_buf,     ci)
            push!(cell_xg_buf,      (bx - 1) * bs + lx)
            push!(cell_yg_buf,      (by - 1) * bs + ly)
            push!(cell_zg_buf,      (bz - 1) * bs + lz)
            push!(cell_n_links_buf, n_links_this)
            push!(cell_n_qlow_buf,  n_qlow_this)
            push!(cell_n_fb_buf,    n_fb_this)
        end
    end

    # --- Totals across all directions ---
    tot_q_low      = sum(n_q_low)
    tot_q_high     = sum(n_q_high)
    tot_intra_f    = sum(n_intra_f)
    tot_nbr_f      = sum(n_nbr_f)
    tot_intra_s    = sum(n_intra_s)
    tot_nbr_s      = sum(n_nbr_s)
    tot_oob        = sum(n_oob_no_nbr)
    tot_fb         = tot_intra_s + tot_nbr_s + tot_oob
    n_cells_with_fb = length(cell_idx_buf)

    fb_link_frac = tot_q_low > 0 ? Float64(tot_fb) / Float64(tot_q_low) : 0.0
    fb_cell_frac = Float64(n_cells_with_fb) / Float64(n_boundary_cells)

    println()
    @printf("[BouzidiFF] Level %d: %d boundary cells, %d q<0.5 links, %d q>=0.5 links\n",
            level_id, n_boundary_cells, tot_q_low, tot_q_high)
    @printf("[BouzidiFF]   q<0.5 fallback total: %d / %d = %.3f%% of q<0.5 links\n",
            tot_fb, tot_q_low, 100.0 * fb_link_frac)
    @printf("[BouzidiFF]     intra_solid: %d  nbr_solid: %d  oob_no_nbr: %d\n",
            tot_intra_s, tot_nbr_s, tot_oob)
    @printf("[BouzidiFF]     (vs intra_fluid: %d  nbr_fluid: %d)\n",
            tot_intra_f, tot_nbr_f)
    @printf("[BouzidiFF]   Cells with ≥1 fallback link: %d / %d = %.3f%%\n",
            n_cells_with_fb, n_boundary_cells, 100.0 * fb_cell_frac)

    # --- Per-direction CSV: one row per discrete velocity ---
    dir_path = joinpath(output_dir,
                        @sprintf("bouzidi_ff_audit_level%d_direction.csv", level_id))
    open(dir_path, "w") do io
        println(io, "k,cx,cy,cz,opp_k,n_q_low,n_q_high,n_intra_fluid,n_nbr_fluid,",
                    "n_intra_solid,n_nbr_solid,n_oob_no_nbr,n_fallback,fb_frac_q_low")
        for k in 1:n_dirs
            opp_k = Int(opp_arr[k])
            cx = Int(cx_arr[k]); cy = Int(cy_arr[k]); cz = Int(cz_arr[k])
            fb = n_intra_s[k] + n_nbr_s[k] + n_oob_no_nbr[k]
            qlow = n_q_low[k]
            frac = qlow > 0 ? Float64(fb) / Float64(qlow) : 0.0
            @printf(io, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f\n",
                    k, cx, cy, cz, opp_k,
                    qlow, n_q_high[k],
                    n_intra_f[k], n_nbr_f[k],
                    n_intra_s[k], n_nbr_s[k], n_oob_no_nbr[k],
                    fb, frac)
        end
    end
    println("[BouzidiFF]   Wrote per-direction CSV: ", dir_path)

    # --- Per-cell CSV: only cells with ≥1 fallback link ---
    cell_path = joinpath(output_dir,
                         @sprintf("bouzidi_ff_audit_level%d_cells.csv", level_id))
    open(cell_path, "w") do io
        println(io, "cell_idx,x_global,y_global,z_global,n_links,n_links_q_low,n_fallback,fb_frac_cell")
        for i in 1:n_cells_with_fb
            nl = cell_n_links_buf[i]
            nq = cell_n_qlow_buf[i]
            nf = cell_n_fb_buf[i]
            frac = nq > 0 ? Float64(nf) / Float64(nq) : 0.0
            @printf(io, "%d,%d,%d,%d,%d,%d,%d,%.6f\n",
                    cell_idx_buf[i],
                    cell_xg_buf[i], cell_yg_buf[i], cell_zg_buf[i],
                    nl, nq, nf, frac)
        end
    end
    println("[BouzidiFF]   Wrote per-cell fallback CSV: ", cell_path,
            "  (", n_cells_with_fb, " cells)")

    # --- Per-direction stdout table for the strongest offenders ---
    println("[BouzidiFF]   Top fallback-rate directions (q<0.5 only):")
    println("[BouzidiFF]   k  | (cx,cy,cz) | q_low |  intra_s |   nbr_s | oob_no_nbr |   fb |  frac")
    println("[BouzidiFF]   ---|------------|-------|----------|---------|------------|------|------")
    # Sort by fallback fraction (descending), only directions with q_low > 0.
    order = sortperm(1:n_dirs;
                     by = k -> (n_q_low[k] == 0 ? -1.0 :
                                Float64(n_intra_s[k] + n_nbr_s[k] + n_oob_no_nbr[k]) /
                                Float64(n_q_low[k])),
                     rev = true)
    shown = 0
    for k in order
        n_q_low[k] > 0 || continue
        fb = n_intra_s[k] + n_nbr_s[k] + n_oob_no_nbr[k]
        frac = Float64(fb) / Float64(n_q_low[k])
        @printf("[BouzidiFF]   %2d | (%+d,%+d,%+d) | %5d | %8d | %7d | %10d | %4d | %.3f\n",
                k, Int(cx_arr[k]), Int(cy_arr[k]), Int(cz_arr[k]),
                n_q_low[k], n_intra_s[k], n_nbr_s[k], n_oob_no_nbr[k], fb, frac)
        shown += 1
        shown >= 8 && break
    end
    println()
    return
end
