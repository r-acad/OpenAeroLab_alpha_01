# ==============================================================================
# AMR validation gates — CPU-side diagnostic emitters
# ==============================================================================
#
# Supports the SOTA "STANDING DIRECTIVE — absolute best-in-class AMR interface
# treatment" gating verification:
#
#   Gate 1  : 13-moment conservation per fine→coarse hop
#             Asserts ‖M_parent − Σ_8children/8‖∞ ≤ tol  for each moment.
#             Emits  amr_conservation_NNNNNN_L<child>to<parent>.csv
#
#   Gate 3  : discrete H-theorem `Σ_k f_k log(f_k/w_k)` across AMR transitions
#             Asserts the parent-cell H ≤ averaged-children H (entropy is
#             non-increasing — the restriction does not inject information).
#             Emits  amr_entropy_NNNNNN_L<child>to<parent>.csv
#
# Both gates share the same data path (one GPU→CPU pull of parent f and
# child f per diagnostic step) so they are computed together in
# `dump_amr_validation_diagnostic!`. Activated by config flag
# `diagnostics.amr_validation_gates: true` and run every
# `diagnostics.amr_validation_freq` steps.
#
# The functions are CPU-side and intentionally simple — they exist to be
# the GROUND-TRUTH against which the (much more complex) GPU kernels are
# validated. Speed is not critical; correctness is.
# ==============================================================================

using Printf

"""
    _compute_13_moments_cpu(f, cx, cy, cz, cs2; shift_x=0, shift_y=0, shift_z=0)

CPU-only sibling of `compute_13_moments_of_populations_d3q39` in
physics_utils.jl. Returns NTuple{13, Float64} (promoted from Float32
storage so the moment sum doesn't lose precision).
"""
function _compute_13_moments_cpu(f::AbstractVector,
                                  cx::AbstractVector, cy::AbstractVector, cz::AbstractVector,
                                  cs2::Float64;
                                  shift_x::Float64=0.0, shift_y::Float64=0.0, shift_z::Float64=0.0)
    m = zeros(Float64, 13)
    dplus2_cs2 = 5.0 * cs2
    @inbounds for k in 1:length(f)
        cxk = Float64(cx[k]) - shift_x
        cyk = Float64(cy[k]) - shift_y
        czk = Float64(cz[k]) - shift_z
        c2k = cxk*cxk + cyk*cyk + czk*czk
        efl = c2k - dplus2_cs2
        fk  = Float64(f[k])
        m[1]  += fk
        m[2]  += fk * cxk
        m[3]  += fk * cyk
        m[4]  += fk * czk
        m[5]  += fk * (cxk*cxk - cs2)
        m[6]  += fk * (cyk*cyk - cs2)
        m[7]  += fk * (czk*czk - cs2)
        m[8]  += fk * cxk * cyk
        m[9]  += fk * cxk * czk
        m[10] += fk * cyk * czk
        m[11] += fk * cxk * efl
        m[12] += fk * cyk * efl
        m[13] += fk * czk * efl
    end
    return m
end

"""
    _compute_H_cpu(f, w)

Discrete H-functional `Σ_k f_k log(f_k / w_k)`. Returns Float64.
Uses a small floor `1e-30` to keep log finite on degenerate cells; in
the relevant interface region populations are far from zero so the
floor doesn't bias the measurement.
"""
function _compute_H_cpu(f::AbstractVector, w::AbstractVector)
    H = 0.0
    @inbounds for k in 1:length(f)
        fk = max(Float64(f[k]), 1.0e-30)
        wk = Float64(w[k])
        H += fk * log(fk / wk)
    end
    return H
end

"""
    dump_amr_validation_diagnostic!(parent_level, child_level, output_dir, t_step,
                                     cx_gpu, cy_gpu, cz_gpu, w_gpu, cs2_phys;
                                     shift_x=0.0, shift_y=0.0, shift_z=0.0)

Compute and emit gate-1 (13-moment conservation) and gate-3 (H-theorem)
diagnostics for the fine→coarse coupling between `child_level` and
`parent_level`. The measurement is done CPU-side after pulling back
the populations from GPU — slow but exact.

For each parent cell flagged `amr_covered=true` (i.e. it has fine
children), compute:
  - The 13 Hermite moments of the parent populations
  - The 13 Hermite moments of the average of 8 child populations
  - The H-functional of both
Then aggregate per-moment max/L1 deltas and per-cell ΔH.

Two CSVs are written:
  - amr_conservation_NNNNNN_L<child>to<parent>.csv  (per-moment statistics)
  - amr_entropy_NNNNNN_L<child>to<parent>.csv       (per-cell H statistics)

The checker scripts `src/PREPOST/amr_conservation_check.jl` and
`src/PREPOST/amr_entropy_check.jl` consume these.
"""
function dump_amr_validation_diagnostic!(parent_level, child_level,
                                          output_dir::String, t_step::Int,
                                          cx_gpu, cy_gpu, cz_gpu, w_gpu,
                                          cs2_phys::Real;
                                          shift_x::Real=0.0,
                                          shift_y::Real=0.0,
                                          shift_z::Real=0.0)
    isdir(output_dir) || mkpath(output_dir)
    n_parent_blocks = length(parent_level.active_block_coords)
    n_parent_blocks == 0 && return

    # Pull data back from GPU. Heavy but acceptable for periodic diagnostic.
    parent_f_cpu = Array(parent_level.f)
    child_f_cpu  = Array(child_level.f)
    parent_amr_covered_cpu = Array(parent_level.amr_covered)
    child_ptr_cpu = Array(child_level.block_pointer)
    parent_active = parent_level.active_block_coords  # Vector{Tuple{Int,Int,Int}}
    cx_cpu = Array(cx_gpu)
    cy_cpu = Array(cy_gpu)
    cz_cpu = Array(cz_gpu)
    w_cpu  = Array(w_gpu)

    n_dirs = size(parent_f_cpu, 5)
    cs2 = Float64(cs2_phys)
    sx = Float64(shift_x); sy = Float64(shift_y); sz = Float64(shift_z)
    half_bs = BLOCK_SIZE ÷ 2

    # Per-moment accumulators (13 moments)
    max_abs_delta = zeros(Float64, 13)
    sum_abs_delta = zeros(Float64, 13)
    sum_abs_ref   = zeros(Float64, 13)
    count_cells = 0

    # H-theorem accumulators
    sum_dH = 0.0           # signed: H_parent - H_child_avg
    sum_abs_dH = 0.0
    max_abs_dH = 0.0
    n_dH_violations = 0    # count of cells where H_parent > H_child_avg (entropy injection)
    max_violation = 0.0    # worst single-cell H increase
    count_h = 0

    f_parent = zeros(Float64, n_dirs)
    f_avg    = zeros(Float64, n_dirs)
    f_child  = zeros(Float64, n_dirs)

    cdim = size(child_ptr_cpu)

    @inbounds for (pb_idx, (pbx, pby, pbz)) in enumerate(parent_active)
        cb_base_x = 2*pbx - 1
        cb_base_y = 2*pby - 1
        cb_base_z = 2*pbz - 1
        for z in 1:BLOCK_SIZE, y in 1:BLOCK_SIZE, x in 1:BLOCK_SIZE
            parent_amr_covered_cpu[x, y, z, pb_idx] || continue
            cbx_off = (x > half_bs) ? 1 : 0
            cby_off = (y > half_bs) ? 1 : 0
            cbz_off = (z > half_bs) ? 1 : 0
            cbx = cb_base_x + cbx_off
            cby = cb_base_y + cby_off
            cbz = cb_base_z + cbz_off
            (cbx < 1 || cbx > cdim[1]) && continue
            (cby < 1 || cby > cdim[2]) && continue
            (cbz < 1 || cbz > cdim[3]) && continue
            cb_idx = child_ptr_cpu[cbx, cby, cbz]
            cb_idx == 0 && continue

            x_in_half = (x > half_bs) ? x - half_bs : x
            y_in_half = (y > half_bs) ? y - half_bs : y
            z_in_half = (z > half_bs) ? z - half_bs : z
            cx_base = 2*x_in_half - 1
            cy_base = 2*y_in_half - 1
            cz_base = 2*z_in_half - 1

            # Compute averaged-child populations and per-child H.
            fill!(f_avg, 0.0)
            H_child_sum = 0.0
            for dz in 0:1, dy in 0:1, dx in 0:1
                ccx = cx_base + dx
                ccy = cy_base + dy
                ccz = cz_base + dz
                for k in 1:n_dirs
                    fk = Float64(child_f_cpu[ccx, ccy, ccz, cb_idx, k])
                    f_avg[k] += fk
                    f_child[k] = fk
                end
                # H of this child cell
                Hc = 0.0
                for k in 1:n_dirs
                    fk = max(f_child[k], 1.0e-30)
                    Hc += fk * log(fk / Float64(w_cpu[k]))
                end
                H_child_sum += Hc
            end
            for k in 1:n_dirs
                f_avg[k] *= 0.125  # /8
            end
            H_child_avg = H_child_sum * 0.125

            # Parent populations
            for k in 1:n_dirs
                f_parent[k] = Float64(parent_f_cpu[x, y, z, pb_idx, k])
            end

            # 13 moments of both
            mom_p = _compute_13_moments_cpu(f_parent, cx_cpu, cy_cpu, cz_cpu, cs2;
                                             shift_x=sx, shift_y=sy, shift_z=sz)
            mom_c = _compute_13_moments_cpu(f_avg, cx_cpu, cy_cpu, cz_cpu, cs2;
                                             shift_x=sx, shift_y=sy, shift_z=sz)

            for i in 1:13
                delta = abs(mom_p[i] - mom_c[i])
                if delta > max_abs_delta[i]; max_abs_delta[i] = delta; end
                sum_abs_delta[i] += delta
                sum_abs_ref[i]   += abs(mom_c[i])
            end
            count_cells += 1

            # H-theorem
            H_p = _compute_H_cpu(f_parent, w_cpu)
            dH = H_p - H_child_avg
            sum_dH += dH
            ad = abs(dH)
            sum_abs_dH += ad
            if ad > max_abs_dH; max_abs_dH = ad; end
            if dH > 0.0
                n_dH_violations += 1
                if dH > max_violation; max_violation = dH; end
            end
            count_h += 1
        end
    end

    # ── Emit gate-1 (13-moment conservation) CSV ──
    tag = @sprintf("L%dto%d", child_level.level_id, parent_level.level_id)
    cons_path = joinpath(output_dir,
        @sprintf("amr_conservation_%06d_%s.csv", t_step, tag))
    open(cons_path, "w") do io
        println(io, "# 13-moment conservation across fine→coarse restriction")
        println(io, "# child level = $(child_level.level_id), parent level = $(parent_level.level_id)")
        println(io, "# t_step = $t_step, n_parent_cells_covered = $count_cells")
        println(io, "moment,max_abs_delta,sum_abs_delta,mean_abs_delta,sum_abs_ref,relative_l1")
        moment_labels = ["mass", "jx", "jy", "jz",
                         "pxx-cs2", "pyy-cs2", "pzz-cs2",
                         "pxy", "pxz", "pyz",
                         "qx", "qy", "qz"]
        for i in 1:13
            mean_d = count_cells > 0 ? sum_abs_delta[i] / count_cells : 0.0
            rel    = sum_abs_ref[i] > 1.0e-30 ? sum_abs_delta[i] / sum_abs_ref[i] : 0.0
            @printf(io, "M%02d_%s,%.10e,%.10e,%.10e,%.10e,%.10e\n",
                    i, moment_labels[i],
                    max_abs_delta[i], sum_abs_delta[i], mean_d,
                    sum_abs_ref[i], rel)
        end
    end

    # ── Emit gate-3 (H-theorem) CSV ──
    ent_path = joinpath(output_dir,
        @sprintf("amr_entropy_%06d_%s.csv", t_step, tag))
    open(ent_path, "w") do io
        println(io, "# H-theorem across fine→coarse restriction")
        println(io, "# H = Σ_k f_k log(f_k / w_k)")
        println(io, "# child level = $(child_level.level_id), parent level = $(parent_level.level_id)")
        println(io, "# t_step = $t_step")
        println(io, "metric,value")
        mean_dH = count_h > 0 ? sum_dH / count_h : 0.0
        mean_abs_dH = count_h > 0 ? sum_abs_dH / count_h : 0.0
        @printf(io, "count_cells,%d\n", count_h)
        @printf(io, "sum_dH_parent_minus_child_avg,%.10e\n", sum_dH)
        @printf(io, "mean_dH_parent_minus_child_avg,%.10e\n", mean_dH)
        @printf(io, "sum_abs_dH,%.10e\n", sum_abs_dH)
        @printf(io, "mean_abs_dH,%.10e\n", mean_abs_dH)
        @printf(io, "max_abs_dH,%.10e\n", max_abs_dH)
        @printf(io, "n_violations,%d\n", n_dH_violations)
        @printf(io, "violation_fraction,%.10e\n", count_h > 0 ? n_dH_violations / count_h : 0.0)
        @printf(io, "max_violation_dH,%.10e\n", max_violation)
    end

    return nothing
end

"""
    dump_vortex_metrics_diagnostic!(grids, output_dir, t_step, U_inlet)

Gate 2 diagnostic — Spiegel isentropic vortex measurement.

Walks all levels and records the peak |u_vec - U_inlet·x̂| over the
entire active domain, plus the position of that peak and the total
enstrophy on the finest level. The output CSV `vortex_metrics_NNNNNN.csv`
is consumed by `src/PREPOST/spiegel_vortex_check.jl` which asserts the
peak perturbation is preserved within tolerance as the vortex advects
across AMR boundaries.

This is a coarse but model-agnostic measurement — it doesn't need to
know where the vortex actually is (the field tells us). For a clean
isentropic vortex propagating at U_inlet, the peak |δu| should stay
within ~0.5% over many AMR crossings; a degraded scheme will show
monotonic decay.
"""
# dump_sod_centerline_diagnostic!(grids, output_dir, t_step, mesh_offset_x; eos_R)
#
# 1D centerline profile for the Sod shock tube (case 050). Averages rho, u_x, T
# over the transverse (y,z) cells at each global-x lattice index and writes
# sod_profile_NNNNNN.csv with columns x, rho, ux, p, T. Pressure uses the
# thermodynamic EOS p = eos_R*rho*T (eos_R = (gamma-1)*cv); pass eos_R<0 to fall
# back to p = CS2*rho (linearized). Single-level only (Sod has no AMR).
function dump_sod_centerline_diagnostic!(grids, output_dir::String, t_step::Int,
                                         mesh_offset_x::Real; eos_R::Real=-1.0)
    isdir(output_dir) || mkpath(output_dir)
    level = grids[1]
    n_blocks = length(level.active_block_coords)
    n_blocks == 0 && return nothing

    rho_cpu  = Array(level.rho)
    vel_cpu  = Array(level.vel)
    has_T    = (length(level.temperature) > 1)
    T_cpu    = has_T ? Array(level.temperature) : nothing

    dx = Float64(level.dx)
    offx = Float64(mesh_offset_x)

    # Accumulate per global-x bin (averaging over y,z and all blocks at that x).
    sums = Dict{Int, NTuple{4, Float64}}()  # gx -> (rho, ux, T, count)
    for b_idx in 1:n_blocks
        bx = level.active_block_coords[b_idx][1]
        for z in 1:BLOCK_SIZE, y in 1:BLOCK_SIZE, x in 1:BLOCK_SIZE
            gx = (bx - 1) * BLOCK_SIZE + x
            r  = Float64(rho_cpu[x, y, z, b_idx])
            ux = Float64(vel_cpu[x, y, z, b_idx, 1])
            T  = has_T ? Float64(T_cpu[x, y, z, b_idx]) : 1.0
            acc = get(sums, gx, (0.0, 0.0, 0.0, 0.0))
            sums[gx] = (acc[1] + r, acc[2] + ux, acc[3] + T, acc[4] + 1.0)
        end
    end

    out_path = joinpath(output_dir, @sprintf("sod_profile_%06d.csv", t_step))
    open(out_path, "w") do io
        println(io, "# Sod centerline profile (transverse-averaged) at step $t_step")
        println(io, "# x,rho,ux,p,T")
        for gx in sort(collect(keys(sums)))
            (sr, su, sT, cnt) = sums[gx]
            r = sr / cnt; ux = su / cnt; T = sT / cnt
            xc = (gx - 0.5) * dx + offx          # cell-center physical x
            p  = eos_R > 0 ? eos_R * r * T : Float64(CS2_PHYSICS) * r
            @printf(io, "%.8f,%.10e,%.10e,%.10e,%.10e\n", xc, r, ux, p, T)
        end
    end
    @printf("[Sod] Wrote centerline profile %s (%d x-stations)\n", out_path, length(sums))
    return nothing
end

function dump_vortex_metrics_diagnostic!(grids, output_dir::String, t_step::Int,
                                          U_inlet::Real;
                                          U_inlet_y::Real=0.0, U_inlet_z::Real=0.0)
    isdir(output_dir) || mkpath(output_dir)
    Ux = Float64(U_inlet)
    Uy = Float64(U_inlet_y)
    Uz = Float64(U_inlet_z)

    out_path = joinpath(output_dir, @sprintf("vortex_metrics_%06d.csv", t_step))
    open(out_path, "w") do io
        println(io, "# Spiegel vortex metrics — peak |u - U_inlet| per level")
        println(io, "# U_inlet = ($Ux, $Uy, $Uz)")
        println(io, "level,n_cells,peak_du_mag,peak_x_block,peak_y_block,peak_z_block,peak_x_cell,peak_y_cell,peak_z_cell,sum_du_sq")
        for lvl_id in 1:length(grids)
            level = grids[lvl_id]
            n_blocks = length(level.active_block_coords)
            n_blocks == 0 && continue
            vel_cpu = Array(level.vel)
            peak = 0.0
            peak_pos = (0, 0, 0, 0, 0, 0)
            sum_du2 = 0.0
            n_cells = 0
            for b_idx in 1:n_blocks
                bcoord = level.active_block_coords[b_idx]
                for z in 1:BLOCK_SIZE, y in 1:BLOCK_SIZE, x in 1:BLOCK_SIZE
                    du_x = Float64(vel_cpu[x, y, z, b_idx, 1]) - Ux
                    du_y = Float64(vel_cpu[x, y, z, b_idx, 2]) - Uy
                    du_z = Float64(vel_cpu[x, y, z, b_idx, 3]) - Uz
                    mag2 = du_x*du_x + du_y*du_y + du_z*du_z
                    sum_du2 += mag2
                    n_cells += 1
                    if mag2 > peak
                        peak = mag2
                        peak_pos = (bcoord[1], bcoord[2], bcoord[3], x, y, z)
                    end
                end
            end
            @printf(io, "%d,%d,%.10e,%d,%d,%d,%d,%d,%d,%.10e\n",
                    level.level_id, n_cells, sqrt(peak),
                    peak_pos[1], peak_pos[2], peak_pos[3],
                    peak_pos[4], peak_pos[5], peak_pos[6],
                    sum_du2)
        end
    end
    return nothing
end

# ──────────────────────────────────────────────────────────────────────
# dump_vortex_field_diagnostic!(grids, output_dir, t_step, mesh_offset_x, mesh_offset_y; eos_R)
#
# AUDIT ADD (2026-06-03): full XY-plane field dump for the isentropic-vortex
# CONVERGENCE study (the operator-accuracy benchmark). Unlike the metrics
# diagnostic (which sums |u-U_inlet| over the whole domain and is boundary-
# confounded), this writes the per-cell (x, y, rho, ux, uy, p, T) at the
# mid-z plane so a checker can compute the INTERIOR L1/L2 error vs the exact
# advected vortex and extract the observed order of accuracy across resolutions.
# Single-level only (the convergence study uses no AMR to isolate the operator).
# Pressure: p = eos_R*rho*T (eos_R=(γ-1)cv) when eos_R>0, else p = CS2_PHYSICS*rho*T.
# ──────────────────────────────────────────────────────────────────────
function dump_vortex_field_diagnostic!(grids, output_dir::String, t_step::Int,
                                       mesh_offset_x::Real, mesh_offset_y::Real;
                                       eos_R::Real=-1.0)
    isdir(output_dir) || mkpath(output_dir)
    level = grids[1]
    n_blocks = length(level.active_block_coords)
    n_blocks == 0 && return nothing

    rho_cpu = Array(level.rho)
    vel_cpu = Array(level.vel)
    has_T   = (length(level.temperature) > 1)
    T_cpu   = has_T ? Array(level.temperature) : nothing

    dx = Float64(level.dx)
    offx = Float64(mesh_offset_x)
    offy = Float64(mesh_offset_y)
    # cs2 fallback for the linearized pressure form (CS2_PHYSICS is a module global)
    cs2 = Float64(CS2_PHYSICS)
    zmid = (BLOCK_SIZE ÷ 2) + 1   # mid-z cell (vortex is uniform in z)

    out_path = joinpath(output_dir, @sprintf("vortex_field_%06d.csv", t_step))
    open(out_path, "w") do io
        println(io, "# Isentropic-vortex XY field (mid-z plane) at step $t_step")
        println(io, "# x,y,rho,ux,uy,p,T")
        for b_idx in 1:n_blocks
            bx = level.active_block_coords[b_idx][1]
            by = level.active_block_coords[b_idx][2]
            for yy in 1:BLOCK_SIZE, xx in 1:BLOCK_SIZE
                gx = (bx - 1) * BLOCK_SIZE + xx
                gy = (by - 1) * BLOCK_SIZE + yy
                xphys = offx + (Float64(gx) - 0.5) * dx
                yphys = offy + (Float64(gy) - 0.5) * dx
                r  = Float64(rho_cpu[xx, yy, zmid, b_idx])
                ux = Float64(vel_cpu[xx, yy, zmid, b_idx, 1])
                uy = Float64(vel_cpu[xx, yy, zmid, b_idx, 2])
                T  = has_T ? Float64(T_cpu[xx, yy, zmid, b_idx]) : 1.0
                p  = (eos_R > 0.0) ? eos_R * r * T : cs2 * r * T
                @printf(io, "%.6f,%.6f,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                        xphys, yphys, r, ux, uy, p, T)
            end
        end
    end
    return nothing
end
