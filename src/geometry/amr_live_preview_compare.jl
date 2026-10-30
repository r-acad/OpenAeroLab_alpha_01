"""
AMR_LIVE_PREVIEW_COMPARE.JL

Diagnostic-only comparison between staged recomposition preview populations and
the live populations currently sitting on the AMR interface.

This closes the loop from "the preview conserves its own moments" to "the
preview predicts a population state that is closer/farther than the live
interface state". No solver arrays are modified.
"""

const AMR_LIVE_STAGE_BLEND_WEIGHTS = (0.0f0, 0.25f0, 0.5f0, 0.75f0, 1.0f0)

function _amr_live_average_populations!(avg::Vector{Float32},
                                        f_cpu,
                                        obs_cpu,
                                        block_idx::Int,
                                        xr,
                                        yr,
                                        zr,
                                        n_dirs::Int)
    fill!(avg, 0.0f0)
    n_cells = 0

    @inbounds for lz in zr, ly in yr, lx in xr
        obs_cpu[lx, ly, lz, block_idx] && continue
        n_cells += 1
        for k in 1:n_dirs
            avg[k] += Float32(f_cpu[lx, ly, lz, block_idx, k])
        end
    end

    if n_cells == 0
        return false
    end

    inv_n = 1.0f0 / Float32(n_cells)
    @inbounds for k in 1:n_dirs
        avg[k] *= inv_n
    end
    return true
end

function _amr_live_population_delta_moments(preview::Vector{Float32},
                                            live::Vector{Float32},
                                            entity::AMRInterfaceEntity,
                                            n_dirs::Int)
    shift_x = Float32(LATTICE_SHIFT_X)
    shift_y = Float32(LATTICE_SHIFT_Y)
    shift_z = Float32(LATTICE_SHIFT_Z)
    cs2 = Float32(LATTICE_TYPE == :d3q39 ? 2.0f0 / 3.0f0 : 1.0f0 / 3.0f0)

    abs_df_sum = 0.0
    df_sq_sum = 0.0
    max_abs_df = 0.0
    drho = 0.0
    djx = 0.0
    djy = 0.0
    djz = 0.0
    dpxx = 0.0
    dpyy = 0.0
    dpzz = 0.0
    dpxy = 0.0
    dpxz = 0.0
    dpyz = 0.0
    face_normal_dj = 0.0

    sx = Float64(entity.sx)
    sy = Float64(entity.sy)
    sz = Float64(entity.sz)

    @inbounds for k in 1:n_dirs
        df = Float64(preview[k] - live[k])
        cx_shifted = Float64(C_X[k])
        cy_shifted = Float64(C_Y[k])
        cz_shifted = Float64(C_Z[k])
        cx_rel = cx_shifted - Float64(shift_x)
        cy_rel = cy_shifted - Float64(shift_y)
        cz_rel = cz_shifted - Float64(shift_z)

        abs_df = abs(df)
        abs_df_sum += abs_df
        df_sq_sum += df * df
        max_abs_df = max(max_abs_df, abs_df)
        drho += df
        djx += df * cx_shifted
        djy += df * cy_shifted
        djz += df * cz_shifted
        dpxx += df * (cx_rel * cx_rel - Float64(cs2))
        dpyy += df * (cy_rel * cy_rel - Float64(cs2))
        dpzz += df * (cz_rel * cz_rel - Float64(cs2))
        dpxy += df * cx_rel * cy_rel
        dpxz += df * cx_rel * cz_rel
        dpyz += df * cy_rel * cz_rel

        if entity.topology == Int8(1)
            face_normal_dj += df * (sx * cx_shifted + sy * cy_shifted + sz * cz_shifted)
        end
    end

    return (
        mean_abs_df = abs_df_sum / max(n_dirs, 1),
        rms_df = sqrt(df_sq_sum / max(n_dirs, 1)),
        max_abs_df = max_abs_df,
        abs_drho = abs(drho),
        abs_dj = sqrt(djx*djx + djy*djy + djz*djz),
        abs_dpi = sqrt(dpxx*dpxx + dpyy*dpyy + dpzz*dpzz +
                       2.0 * (dpxy*dpxy + dpxz*dpxz + dpyz*dpyz)),
        abs_face_normal_dj = abs(face_normal_dj)
    )
end

function _summarize_amr_live_preview_matrix(registry::AMRInterfaceRegistry,
                                            buffers::AMRStagedTransferBuffers,
                                            state_matrix::Matrix{Float32},
                                            parent_level::BlockLevel,
                                            child_level::BlockLevel;
                                            state::Symbol=:custom)
    child_f_cpu = Array(child_level.f)
    child_obs_cpu = Array(child_level.obstacle)
    parent_f_cpu = Array(parent_level.f)
    parent_obs_cpu = Array(parent_level.obstacle)
    parent_ptr_cpu = Array(parent_level.block_pointer)

    n_dirs = min(Int(N_DIRS), size(child_f_cpu, 5), size(parent_f_cpu, 5))
    bs = BLOCK_SIZE
    preview = zeros(Float32, n_dirs)
    child_live = zeros(Float32, n_dirs)
    parent_live = zeros(Float32, n_dirs)

    n_child_pairs = 0
    child_mean_abs_df_sum = 0.0
    child_rms_df_sum = 0.0
    child_max_abs_df = 0.0
    child_abs_drho_sum = 0.0
    child_abs_dj_sum = 0.0
    child_abs_dpi_sum = 0.0
    face_abs_normal_dj_sum = 0.0

    n_parent_pairs = 0
    parent_mean_abs_df_sum = 0.0
    parent_abs_drho_sum = 0.0
    parent_abs_dj_sum = 0.0

    n_face = 0
    n_edge = 0
    n_corner = 0
    child_mean_abs_df_face_sum = 0.0
    child_mean_abs_df_edge_sum = 0.0
    child_mean_abs_df_corner_sum = 0.0

    @inbounds for idx in eachindex(registry.entities)
        (buffers.coarse_valid[idx] && buffers.child_valid[idx]) || continue
        entity = registry.entities[idx]
        _amr_recomp_preview_populations!(preview, state_matrix, idx)

        child_idx = Int(entity.child_block_index)
        xr = _amr_decomp_cell_range(entity.sx, bs)
        yr = _amr_decomp_cell_range(entity.sy, bs)
        zr = _amr_decomp_cell_range(entity.sz, bs)
        if _amr_live_average_populations!(
            child_live, child_f_cpu, child_obs_cpu, child_idx, xr, yr, zr, n_dirs
        )
            d = _amr_live_population_delta_moments(preview, child_live, entity, n_dirs)
            n_child_pairs += 1
            child_mean_abs_df_sum += d.mean_abs_df
            child_rms_df_sum += d.rms_df
            child_max_abs_df = max(child_max_abs_df, d.max_abs_df)
            child_abs_drho_sum += d.abs_drho
            child_abs_dj_sum += d.abs_dj
            child_abs_dpi_sum += d.abs_dpi
            if entity.topology == Int8(1)
                n_face += 1
                face_abs_normal_dj_sum += d.abs_face_normal_dj
                child_mean_abs_df_face_sum += d.mean_abs_df
            elseif entity.topology == Int8(2)
                n_edge += 1
                child_mean_abs_df_edge_sum += d.mean_abs_df
            elseif entity.topology == Int8(3)
                n_corner += 1
                child_mean_abs_df_corner_sum += d.mean_abs_df
            end
        end

        if entity.parent_inside_active
            pbx = Int(entity.parent_bx)
            pby = Int(entity.parent_by)
            pbz = Int(entity.parent_bz)
            if _amr_registry_active(parent_ptr_cpu, pbx, pby, pbz)
                parent_idx = Int(parent_ptr_cpu[pbx, pby, pbz])
                pxr = _amr_decomp_parent_child_half_range(entity.child_bx, entity.sx, bs)
                pyr = _amr_decomp_parent_child_half_range(entity.child_by, entity.sy, bs)
                pzr = _amr_decomp_parent_child_half_range(entity.child_bz, entity.sz, bs)
                if _amr_live_average_populations!(
                    parent_live, parent_f_cpu, parent_obs_cpu,
                    parent_idx, pxr, pyr, pzr, n_dirs
                )
                    d_parent = _amr_live_population_delta_moments(preview, parent_live, entity, n_dirs)
                    n_parent_pairs += 1
                    parent_mean_abs_df_sum += d_parent.mean_abs_df
                    parent_abs_drho_sum += d_parent.abs_drho
                    parent_abs_dj_sum += d_parent.abs_dj
                end
            end
        end
    end

    return (
        state = state,
        n_child_pairs = n_child_pairs,
        n_parent_pairs = n_parent_pairs,
        mean_abs_df_child = n_child_pairs > 0 ? child_mean_abs_df_sum / n_child_pairs : 0.0,
        mean_rms_df_child = n_child_pairs > 0 ? child_rms_df_sum / n_child_pairs : 0.0,
        max_abs_df_child = child_max_abs_df,
        mean_abs_drho_child = n_child_pairs > 0 ? child_abs_drho_sum / n_child_pairs : 0.0,
        mean_abs_dj_child = n_child_pairs > 0 ? child_abs_dj_sum / n_child_pairs : 0.0,
        mean_abs_dpi_child = n_child_pairs > 0 ? child_abs_dpi_sum / n_child_pairs : 0.0,
        mean_abs_face_normal_dj = n_face > 0 ? face_abs_normal_dj_sum / n_face : 0.0,
        n_face_pairs = n_face,
        n_edge_pairs = n_edge,
        n_corner_pairs = n_corner,
        mean_abs_df_face = n_face > 0 ? child_mean_abs_df_face_sum / n_face : 0.0,
        mean_abs_df_edge = n_edge > 0 ? child_mean_abs_df_edge_sum / n_edge : 0.0,
        mean_abs_df_corner = n_corner > 0 ? child_mean_abs_df_corner_sum / n_corner : 0.0,
        mean_abs_df_parent = n_parent_pairs > 0 ? parent_mean_abs_df_sum / n_parent_pairs : 0.0,
        mean_abs_drho_parent = n_parent_pairs > 0 ? parent_abs_drho_sum / n_parent_pairs : 0.0,
        mean_abs_dj_parent = n_parent_pairs > 0 ? parent_abs_dj_sum / n_parent_pairs : 0.0
    )
end

function summarize_amr_live_preview_comparison(registry::AMRInterfaceRegistry,
                                               buffers::AMRStagedTransferBuffers,
                                               parent_level::BlockLevel,
                                               child_level::BlockLevel;
                                               state::Symbol=:t12)
    state_matrix = _amr_recomp_state_matrix(buffers, state)
    return _summarize_amr_live_preview_matrix(
        registry, buffers, state_matrix, parent_level, child_level; state=state)
end

function write_amr_live_preview_compare_csv_header(filepath::String)
    open(filepath, "w") do io
        println(io,
            "step,parent_level,child_level,state,n_child_pairs,n_parent_pairs," *
            "mean_abs_df_child,mean_rms_df_child,max_abs_df_child," *
            "mean_abs_drho_child,mean_abs_dj_child,mean_abs_dpi_child," *
            "mean_abs_face_normal_dj,n_face_pairs,n_edge_pairs,n_corner_pairs," *
            "mean_abs_df_face,mean_abs_df_edge,mean_abs_df_corner," *
            "mean_abs_df_parent,mean_abs_drho_parent,mean_abs_dj_parent")
    end
end

function append_amr_live_preview_compare_csv_row(filepath::String, step::Int,
                                                 parent_level::Int, child_level::Int,
                                                 summary)
    if !isfile(filepath)
        write_amr_live_preview_compare_csv_header(filepath)
    end
    open(filepath, "a") do io
        @printf(io,
            "%d,%d,%d,%s,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",
            step,
            parent_level,
            child_level,
            String(summary.state),
            summary.n_child_pairs,
            summary.n_parent_pairs,
            summary.mean_abs_df_child,
            summary.mean_rms_df_child,
            summary.max_abs_df_child,
            summary.mean_abs_drho_child,
            summary.mean_abs_dj_child,
            summary.mean_abs_dpi_child,
            summary.mean_abs_face_normal_dj,
            summary.n_face_pairs,
            summary.n_edge_pairs,
            summary.n_corner_pairs,
            summary.mean_abs_df_face,
            summary.mean_abs_df_edge,
            summary.mean_abs_df_corner,
            summary.mean_abs_df_parent,
            summary.mean_abs_drho_parent,
            summary.mean_abs_dj_parent)
    end
end

@inline _amr_live_stage_moment_score(summary) =
    summary.mean_abs_drho_child + summary.mean_abs_dj_child + summary.mean_abs_dpi_child

@inline _amr_live_stage_ratio(num::Float64, den::Float64) =
    den > 0.0 ? num / den : 0.0

function _amr_live_blended_state(buffers::AMRStagedTransferBuffers,
                                 weight_t1::Float32)
    state = similar(buffers.coarse_t1)
    w1 = clamp(weight_t1, 0.0f0, 1.0f0)
    w0 = 1.0f0 - w1
    @inbounds for i in eachindex(state)
        state[i] = w0 * buffers.coarse_t0[i] + w1 * buffers.coarse_t1[i]
    end
    return state
end

function summarize_amr_live_stage_selector(registry::AMRInterfaceRegistry,
                                           buffers::AMRStagedTransferBuffers,
                                           parent_level::BlockLevel,
                                           child_level::BlockLevel;
                                           t12_summary=nothing)
    t1 = summarize_amr_live_preview_comparison(
        registry, buffers, parent_level, child_level; state=:t1)
    t12 = t12_summary === nothing ?
        summarize_amr_live_preview_comparison(
            registry, buffers, parent_level, child_level; state=:t12) :
        t12_summary

    t1_moment = _amr_live_stage_moment_score(t1)
    t12_moment = _amr_live_stage_moment_score(t12)
    preferred_df = t12.mean_abs_df_child < t1.mean_abs_df_child ? :t12 : :t1
    preferred_moment = t12_moment < t1_moment ? :t12 : :t1
    preferred_face_normal =
        t12.mean_abs_face_normal_dj < t1.mean_abs_face_normal_dj ? :t12 : :t1

    return (
        n_child_pairs = min(t1.n_child_pairs, t12.n_child_pairs),
        n_face_pairs = min(t1.n_face_pairs, t12.n_face_pairs),
        n_edge_pairs = min(t1.n_edge_pairs, t12.n_edge_pairs),
        n_corner_pairs = min(t1.n_corner_pairs, t12.n_corner_pairs),
        t1_mean_abs_df_child = t1.mean_abs_df_child,
        t12_mean_abs_df_child = t12.mean_abs_df_child,
        t1_mean_abs_drho_child = t1.mean_abs_drho_child,
        t12_mean_abs_drho_child = t12.mean_abs_drho_child,
        t1_mean_abs_dj_child = t1.mean_abs_dj_child,
        t12_mean_abs_dj_child = t12.mean_abs_dj_child,
        t1_mean_abs_dpi_child = t1.mean_abs_dpi_child,
        t12_mean_abs_dpi_child = t12.mean_abs_dpi_child,
        t1_mean_abs_face_normal_dj = t1.mean_abs_face_normal_dj,
        t12_mean_abs_face_normal_dj = t12.mean_abs_face_normal_dj,
        t1_moment_score = t1_moment,
        t12_moment_score = t12_moment,
        df_t12_over_t1 = _amr_live_stage_ratio(
            t12.mean_abs_df_child, t1.mean_abs_df_child),
        moment_t12_over_t1 = _amr_live_stage_ratio(t12_moment, t1_moment),
        face_normal_t12_over_t1 = _amr_live_stage_ratio(
            t12.mean_abs_face_normal_dj, t1.mean_abs_face_normal_dj),
        preferred_df = preferred_df,
        preferred_moment = preferred_moment,
        preferred_face_normal = preferred_face_normal
    )
end

function summarize_amr_live_stage_blend_sweep(registry::AMRInterfaceRegistry,
                                              buffers::AMRStagedTransferBuffers,
                                              parent_level::BlockLevel,
                                              child_level::BlockLevel;
                                              weights=AMR_LIVE_STAGE_BLEND_WEIGHTS)
    rows = NamedTuple[]
    for weight in weights
        w = Float32(weight)
        state_matrix = _amr_live_blended_state(buffers, w)
        summary = _summarize_amr_live_preview_matrix(
            registry, buffers, state_matrix, parent_level, child_level; state=:blend)
        push!(rows, (
            weight_t1 = Float64(w),
            n_child_pairs = summary.n_child_pairs,
            n_parent_pairs = summary.n_parent_pairs,
            n_face_pairs = summary.n_face_pairs,
            n_edge_pairs = summary.n_edge_pairs,
            n_corner_pairs = summary.n_corner_pairs,
            mean_abs_df_child = summary.mean_abs_df_child,
            mean_rms_df_child = summary.mean_rms_df_child,
            max_abs_df_child = summary.max_abs_df_child,
            mean_abs_drho_child = summary.mean_abs_drho_child,
            mean_abs_dj_child = summary.mean_abs_dj_child,
            mean_abs_dpi_child = summary.mean_abs_dpi_child,
            mean_abs_face_normal_dj = summary.mean_abs_face_normal_dj,
            mean_abs_df_face = summary.mean_abs_df_face,
            mean_abs_df_edge = summary.mean_abs_df_edge,
            mean_abs_df_corner = summary.mean_abs_df_corner,
            mean_abs_df_parent = summary.mean_abs_df_parent,
            mean_abs_drho_parent = summary.mean_abs_drho_parent,
            mean_abs_dj_parent = summary.mean_abs_dj_parent,
            moment_score = _amr_live_stage_moment_score(summary)
        ))
    end
    return rows
end

function summarize_amr_live_stage_blend_best(rows)
    isempty(rows) && return (
        n_weights = 0,
        best_df_weight = 0.0,
        best_df = 0.0,
        best_moment_weight = 0.0,
        best_moment = 0.0,
        best_face_weight = 0.0,
        best_face_normal = 0.0
    )

    best_df = rows[1]
    best_moment = rows[1]
    best_face = rows[1]
    for row in rows
        row.mean_abs_df_child < best_df.mean_abs_df_child && (best_df = row)
        row.moment_score < best_moment.moment_score && (best_moment = row)
        row.mean_abs_face_normal_dj < best_face.mean_abs_face_normal_dj && (best_face = row)
    end

    return (
        n_weights = length(rows),
        best_df_weight = best_df.weight_t1,
        best_df = best_df.mean_abs_df_child,
        best_moment_weight = best_moment.weight_t1,
        best_moment = best_moment.moment_score,
        best_face_weight = best_face.weight_t1,
        best_face_normal = best_face.mean_abs_face_normal_dj
    )
end

function write_amr_live_stage_selector_csv_header(filepath::String)
    open(filepath, "w") do io
        println(io,
            "step,parent_level,child_level,n_child_pairs,n_face_pairs,n_edge_pairs,n_corner_pairs," *
            "t1_mean_abs_df_child,t12_mean_abs_df_child," *
            "t1_mean_abs_drho_child,t12_mean_abs_drho_child," *
            "t1_mean_abs_dj_child,t12_mean_abs_dj_child," *
            "t1_mean_abs_dpi_child,t12_mean_abs_dpi_child," *
            "t1_mean_abs_face_normal_dj,t12_mean_abs_face_normal_dj," *
            "t1_moment_score,t12_moment_score," *
            "df_t12_over_t1,moment_t12_over_t1,face_normal_t12_over_t1," *
            "preferred_df,preferred_moment,preferred_face_normal")
    end
end

function append_amr_live_stage_selector_csv_row(filepath::String, step::Int,
                                                parent_level::Int, child_level::Int,
                                                summary)
    if !isfile(filepath)
        write_amr_live_stage_selector_csv_header(filepath)
    end
    open(filepath, "a") do io
        @printf(io,
            "%d,%d,%d,%d,%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%s,%s,%s\n",
            step,
            parent_level,
            child_level,
            summary.n_child_pairs,
            summary.n_face_pairs,
            summary.n_edge_pairs,
            summary.n_corner_pairs,
            summary.t1_mean_abs_df_child,
            summary.t12_mean_abs_df_child,
            summary.t1_mean_abs_drho_child,
            summary.t12_mean_abs_drho_child,
            summary.t1_mean_abs_dj_child,
            summary.t12_mean_abs_dj_child,
            summary.t1_mean_abs_dpi_child,
            summary.t12_mean_abs_dpi_child,
            summary.t1_mean_abs_face_normal_dj,
            summary.t12_mean_abs_face_normal_dj,
            summary.t1_moment_score,
            summary.t12_moment_score,
            summary.df_t12_over_t1,
            summary.moment_t12_over_t1,
            summary.face_normal_t12_over_t1,
            String(summary.preferred_df),
            String(summary.preferred_moment),
            String(summary.preferred_face_normal))
    end
end

function amr_live_preview_compare_csv_path(amr_interface_csv_path::String)
    return joinpath(dirname(amr_interface_csv_path), "amr_live_preview_compare_history.csv")
end

function amr_live_stage_selector_csv_path(amr_interface_csv_path::String)
    return joinpath(dirname(amr_interface_csv_path), "amr_live_stage_selector_history.csv")
end

function write_amr_live_stage_blend_sweep_csv_header(filepath::String)
    open(filepath, "w") do io
        println(io,
            "step,parent_level,child_level,weight_t1,n_child_pairs,n_parent_pairs," *
            "n_face_pairs,n_edge_pairs,n_corner_pairs," *
            "mean_abs_df_child,mean_rms_df_child,max_abs_df_child," *
            "mean_abs_drho_child,mean_abs_dj_child,mean_abs_dpi_child," *
            "mean_abs_face_normal_dj,mean_abs_df_face,mean_abs_df_edge,mean_abs_df_corner," *
            "mean_abs_df_parent,mean_abs_drho_parent,mean_abs_dj_parent,moment_score")
    end
end

function append_amr_live_stage_blend_sweep_csv_rows(filepath::String, step::Int,
                                                    parent_level::Int, child_level::Int,
                                                    rows)
    if !isfile(filepath)
        write_amr_live_stage_blend_sweep_csv_header(filepath)
    end
    open(filepath, "a") do io
        for row in rows
            @printf(io,
                "%d,%d,%d,%.6f,%d,%d,%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",
                step,
                parent_level,
                child_level,
                row.weight_t1,
                row.n_child_pairs,
                row.n_parent_pairs,
                row.n_face_pairs,
                row.n_edge_pairs,
                row.n_corner_pairs,
                row.mean_abs_df_child,
                row.mean_rms_df_child,
                row.max_abs_df_child,
                row.mean_abs_drho_child,
                row.mean_abs_dj_child,
                row.mean_abs_dpi_child,
                row.mean_abs_face_normal_dj,
                row.mean_abs_df_face,
                row.mean_abs_df_edge,
                row.mean_abs_df_corner,
                row.mean_abs_df_parent,
                row.mean_abs_drho_parent,
                row.mean_abs_dj_parent,
                row.moment_score)
        end
    end
end

function amr_live_stage_blend_sweep_csv_path(amr_interface_csv_path::String)
    return joinpath(dirname(amr_interface_csv_path), "amr_live_stage_blend_sweep_history.csv")
end
