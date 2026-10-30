"""
AMR_STAGED_TRANSFER.JL

Diagnostic-only Palabos-style staged AMR transfer view.

For each AMRInterfaceEntity this builds coarse decomposed snapshots:

- t0: previous parent macro state, using current Pi_neq when no Pi history exists;
- t1: current parent decomposed state from AMRDecomposedTransferBuffers;
- t12: linear half-time state, 0.5*(t0 + t1).

The staged state is compared against the child-side decomposed interface state,
but it is never applied to the solver.
"""

struct AMRStagedTransferBuffers
    parent_level_id::Int
    child_level_id::Int
    coarse_t0::Matrix{Float32}
    coarse_t1::Matrix{Float32}
    coarse_t12::Matrix{Float32}
    child_state::Matrix{Float32}
    coarse_valid::BitVector
    child_valid::BitVector
    temporal_available::Bool
    pi_temporal_available::Bool
end

function build_amr_staged_transfer_buffers(registry::AMRInterfaceRegistry,
                                           parent_level::BlockLevel,
                                           child_level::BlockLevel,
                                           decomp_buffers::AMRDecomposedTransferBuffers)
    n_entities = length(registry.entities)
    coarse_t0 = zeros(Float32, AMR_DECOMP_STATE_SIZE, n_entities)
    coarse_t1 = copy(decomp_buffers.parent_state)
    coarse_t12 = zeros(Float32, AMR_DECOMP_STATE_SIZE, n_entities)
    coarse_valid = copy(decomp_buffers.parent_valid)

    temporal_available = has_macro_temporal_storage(parent_level)
    parent_ptr_cpu = Array(parent_level.block_pointer)
    parent_obs_cpu = Array(parent_level.obstacle)
    parent_rho_t0 = temporal_available ? Array(parent_level.rho_old) : Array(parent_level.rho)
    parent_vel_t0 = temporal_available ? Array(parent_level.vel_old) : Array(parent_level.vel)
    parent_has_temp = has_thermal(parent_level)
    parent_temp_t0 = parent_has_temp ?
        (temporal_available ? Array(parent_level.temperature_old) : Array(parent_level.temperature)) :
        nothing
    parent_has_pi = has_pi_neq(parent_level)
    parent_pi_cpu = parent_has_pi ? Array(parent_level.pi_neq) : nothing

    bs = BLOCK_SIZE
    @inbounds for (idx, entity) in enumerate(registry.entities)
        if entity.parent_inside_active
            pbx = Int(entity.parent_bx)
            pby = Int(entity.parent_by)
            pbz = Int(entity.parent_bz)
            if _amr_registry_active(parent_ptr_cpu, pbx, pby, pbz)
                parent_idx = Int(parent_ptr_cpu[pbx, pby, pbz])
                pxr = _amr_decomp_parent_child_half_range(entity.child_bx, entity.sx, bs)
                pyr = _amr_decomp_parent_child_half_range(entity.child_by, entity.sy, bs)
                pzr = _amr_decomp_parent_child_half_range(entity.child_bz, entity.sz, bs)
                t0_vals, t0_valid = _amr_decomp_average_state(
                    parent_rho_t0,
                    parent_vel_t0,
                    parent_obs_cpu,
                    parent_temp_t0,
                    parent_pi_cpu,
                    parent_idx,
                    pxr,
                    pyr,
                    pzr,
                    parent_has_temp,
                    parent_has_pi
                )
                _amr_decomp_store!(coarse_t0, idx, t0_vals)
                coarse_valid[idx] = coarse_valid[idx] && t0_valid
            end
        end

        if coarse_valid[idx]
            for i in 1:AMR_DECOMP_STATE_SIZE
                coarse_t12[i, idx] = 0.5f0 * (coarse_t0[i, idx] + coarse_t1[i, idx])
            end
        end
    end

    return AMRStagedTransferBuffers(
        registry.parent_level_id,
        registry.child_level_id,
        coarse_t0,
        coarse_t1,
        coarse_t12,
        decomp_buffers.child_state,
        coarse_valid,
        decomp_buffers.child_valid,
        temporal_available,
        false
    )
end

function build_amr_staged_transfer_buffers(registry::AMRInterfaceRegistry,
                                           parent_level::BlockLevel,
                                           child_level::BlockLevel)
    decomp_buffers = build_amr_decomposed_transfer_buffers(registry, parent_level, child_level)
    return build_amr_staged_transfer_buffers(registry, parent_level, child_level, decomp_buffers)
end

@inline function _amr_stage_scalar_absdiff(a::Matrix{Float32}, b::Matrix{Float32},
                                           field::Int, idx::Int)
    return abs(Float64(a[field, idx] - b[field, idx]))
end

function summarize_amr_staged_transfer_buffers(registry::AMRInterfaceRegistry,
                                               buffers::AMRStagedTransferBuffers)
    n_pairs = 0
    temporal_drho_sum = 0.0
    temporal_dj_sum = 0.0
    temporal_dT_sum = 0.0

    child_t1_drho_sum = 0.0
    child_t12_drho_sum = 0.0
    child_t1_dj_sum = 0.0
    child_t12_dj_sum = 0.0
    child_t1_dT_sum = 0.0
    child_t12_dT_sum = 0.0
    child_t1_dpi_sum = 0.0
    child_t12_dpi_sum = 0.0

    face_child_t12_drho_sum = 0.0
    edge_child_t12_drho_sum = 0.0
    corner_child_t12_drho_sum = 0.0
    n_face = 0
    n_edge = 0
    n_corner = 0

    @inbounds for idx in eachindex(registry.entities)
        (buffers.coarse_valid[idx] && buffers.child_valid[idx]) || continue
        entity = registry.entities[idx]
        n_pairs += 1

        temporal_drho_sum += _amr_stage_scalar_absdiff(
            buffers.coarse_t1, buffers.coarse_t0, AMR_DECOMP_RHO, idx)
        temporal_dj_sum += _amr_decomp_jdiff(buffers.coarse_t0, buffers.coarse_t1, idx)
        temporal_dT_sum += _amr_stage_scalar_absdiff(
            buffers.coarse_t1, buffers.coarse_t0, AMR_DECOMP_T, idx)

        child_t1_drho = _amr_stage_scalar_absdiff(
            buffers.child_state, buffers.coarse_t1, AMR_DECOMP_RHO, idx)
        child_t12_drho = _amr_stage_scalar_absdiff(
            buffers.child_state, buffers.coarse_t12, AMR_DECOMP_RHO, idx)
        child_t1_dj = _amr_decomp_jdiff(buffers.coarse_t1, buffers.child_state, idx)
        child_t12_dj = _amr_decomp_jdiff(buffers.coarse_t12, buffers.child_state, idx)
        child_t1_dT = _amr_stage_scalar_absdiff(
            buffers.child_state, buffers.coarse_t1, AMR_DECOMP_T, idx)
        child_t12_dT = _amr_stage_scalar_absdiff(
            buffers.child_state, buffers.coarse_t12, AMR_DECOMP_T, idx)
        child_t1_dpi = _amr_decomp_pidiff(buffers.coarse_t1, buffers.child_state, idx)
        child_t12_dpi = _amr_decomp_pidiff(buffers.coarse_t12, buffers.child_state, idx)

        child_t1_drho_sum += child_t1_drho
        child_t12_drho_sum += child_t12_drho
        child_t1_dj_sum += child_t1_dj
        child_t12_dj_sum += child_t12_dj
        child_t1_dT_sum += child_t1_dT
        child_t12_dT_sum += child_t12_dT
        child_t1_dpi_sum += child_t1_dpi
        child_t12_dpi_sum += child_t12_dpi

        if entity.topology == Int8(1)
            n_face += 1
            face_child_t12_drho_sum += child_t12_drho
        elseif entity.topology == Int8(2)
            n_edge += 1
            edge_child_t12_drho_sum += child_t12_drho
        elseif entity.topology == Int8(3)
            n_corner += 1
            corner_child_t12_drho_sum += child_t12_drho
        end
    end

    return (
        n_pairs = n_pairs,
        temporal_available = buffers.temporal_available,
        pi_temporal_available = buffers.pi_temporal_available,
        mean_abs_parent_temporal_drho = n_pairs > 0 ? temporal_drho_sum / n_pairs : 0.0,
        mean_abs_parent_temporal_dj = n_pairs > 0 ? temporal_dj_sum / n_pairs : 0.0,
        mean_abs_parent_temporal_dT = n_pairs > 0 ? temporal_dT_sum / n_pairs : 0.0,
        mean_abs_child_t1_drho = n_pairs > 0 ? child_t1_drho_sum / n_pairs : 0.0,
        mean_abs_child_t12_drho = n_pairs > 0 ? child_t12_drho_sum / n_pairs : 0.0,
        mean_abs_child_t1_dj = n_pairs > 0 ? child_t1_dj_sum / n_pairs : 0.0,
        mean_abs_child_t12_dj = n_pairs > 0 ? child_t12_dj_sum / n_pairs : 0.0,
        mean_abs_child_t1_dT = n_pairs > 0 ? child_t1_dT_sum / n_pairs : 0.0,
        mean_abs_child_t12_dT = n_pairs > 0 ? child_t12_dT_sum / n_pairs : 0.0,
        mean_abs_child_t1_dpi = n_pairs > 0 ? child_t1_dpi_sum / n_pairs : 0.0,
        mean_abs_child_t12_dpi = n_pairs > 0 ? child_t12_dpi_sum / n_pairs : 0.0,
        n_face_pairs = n_face,
        n_edge_pairs = n_edge,
        n_corner_pairs = n_corner,
        mean_abs_child_t12_drho_face = n_face > 0 ? face_child_t12_drho_sum / n_face : 0.0,
        mean_abs_child_t12_drho_edge = n_edge > 0 ? edge_child_t12_drho_sum / n_edge : 0.0,
        mean_abs_child_t12_drho_corner = n_corner > 0 ? corner_child_t12_drho_sum / n_corner : 0.0
    )
end

function write_amr_staged_transfer_csv_header(filepath::String)
    open(filepath, "w") do io
        println(io,
            "step,parent_level,child_level,n_pairs,temporal_available,pi_temporal_available," *
            "parent_temporal_drho,parent_temporal_dj,parent_temporal_dT," *
            "child_t1_drho,child_t12_drho,child_t1_dj,child_t12_dj," *
            "child_t1_dT,child_t12_dT,child_t1_dpi,child_t12_dpi," *
            "n_face_pairs,n_edge_pairs,n_corner_pairs," *
            "child_t12_drho_face,child_t12_drho_edge,child_t12_drho_corner")
    end
end

function append_amr_staged_transfer_csv_row(filepath::String, step::Int,
                                            parent_level::Int, child_level::Int,
                                            summary)
    if !isfile(filepath)
        write_amr_staged_transfer_csv_header(filepath)
    end
    open(filepath, "a") do io
        @printf(io,
            "%d,%d,%d,%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%d,%d,%d,%.9e,%.9e,%.9e\n",
            step,
            parent_level,
            child_level,
            summary.n_pairs,
            summary.temporal_available ? 1 : 0,
            summary.pi_temporal_available ? 1 : 0,
            summary.mean_abs_parent_temporal_drho,
            summary.mean_abs_parent_temporal_dj,
            summary.mean_abs_parent_temporal_dT,
            summary.mean_abs_child_t1_drho,
            summary.mean_abs_child_t12_drho,
            summary.mean_abs_child_t1_dj,
            summary.mean_abs_child_t12_dj,
            summary.mean_abs_child_t1_dT,
            summary.mean_abs_child_t12_dT,
            summary.mean_abs_child_t1_dpi,
            summary.mean_abs_child_t12_dpi,
            summary.n_face_pairs,
            summary.n_edge_pairs,
            summary.n_corner_pairs,
            summary.mean_abs_child_t12_drho_face,
            summary.mean_abs_child_t12_drho_edge,
            summary.mean_abs_child_t12_drho_corner)
    end
end

function amr_staged_transfer_csv_path(amr_interface_csv_path::String)
    return joinpath(dirname(amr_interface_csv_path), "amr_staged_transfer_history.csv")
end
