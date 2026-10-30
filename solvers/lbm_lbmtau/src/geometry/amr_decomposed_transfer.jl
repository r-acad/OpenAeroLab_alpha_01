"""
AMR_DECOMPOSED_TRANSFER.JL

Diagnostic-only decomposed AMR transfer buffers keyed by AMRInterfaceEntity.

The buffer layout is the first scaffold for a Palabos-style staged transfer:
conserved variables and non-equilibrium stress are gathered into compact arrays
with one column per interface registry entity. No solver state is modified.
"""

const AMR_DECOMP_RHO = 1
const AMR_DECOMP_JX  = 2
const AMR_DECOMP_JY  = 3
const AMR_DECOMP_JZ  = 4
const AMR_DECOMP_T   = 5
const AMR_DECOMP_PIXX = 6
const AMR_DECOMP_PIYY = 7
const AMR_DECOMP_PIZZ = 8
const AMR_DECOMP_PIXY = 9
const AMR_DECOMP_PIXZ = 10
const AMR_DECOMP_PIYZ = 11
const AMR_DECOMP_STATE_SIZE = 11

struct AMRDecomposedTransferBuffers
    parent_level_id::Int
    child_level_id::Int
    parent_state::Matrix{Float32}
    child_state::Matrix{Float32}
    parent_valid::BitVector
    child_valid::BitVector
end

@inline function _amr_decomp_cell_range(side::Int8, block_size::Int)
    side < 0 && return 1:1
    side > 0 && return block_size:block_size
    return 1:block_size
end

@inline function _amr_decomp_parent_child_half_range(child_b::Int32,
                                                     side::Int8,
                                                     block_size::Int)
    half = block_size ÷ 2
    low_half = isodd(Int(child_b))
    lo = low_half ? 1 : half + 1
    hi = low_half ? half : block_size

    # Match the child-side face/edge/corner subset with the corresponding
    # parent half-block layer. Whole-block parent averages hide exactly the
    # topology jump this diagnostic is meant to expose.
    side < 0 && return lo:lo
    side > 0 && return hi:hi
    return lo:hi
end

function _amr_decomp_average_state(
    rho_cpu,
    vel_cpu,
    obs_cpu,
    temp_cpu,
    pi_cpu,
    block_idx::Int,
    xr,
    yr,
    zr,
    has_temp::Bool,
    has_pi::Bool
)
    n = 0
    rho_sum = 0.0f0
    jx_sum = 0.0f0
    jy_sum = 0.0f0
    jz_sum = 0.0f0
    t_sum = 0.0f0
    pxx_sum = 0.0f0
    pyy_sum = 0.0f0
    pzz_sum = 0.0f0
    pxy_sum = 0.0f0
    pxz_sum = 0.0f0
    pyz_sum = 0.0f0

    @inbounds for lz in zr, ly in yr, lx in xr
        obs_cpu[lx, ly, lz, block_idx] && continue
        rho = Float32(rho_cpu[lx, ly, lz, block_idx])
        ux = Float32(vel_cpu[lx, ly, lz, block_idx, 1])
        uy = Float32(vel_cpu[lx, ly, lz, block_idx, 2])
        uz = Float32(vel_cpu[lx, ly, lz, block_idx, 3])

        n += 1
        rho_sum += rho
        jx_sum += rho * ux
        jy_sum += rho * uy
        jz_sum += rho * uz
        t_sum += has_temp ? Float32(temp_cpu[lx, ly, lz, block_idx]) : 1.0f0

        if has_pi
            pxx_sum += Float32(pi_cpu[lx, ly, lz, block_idx, 1])
            pyy_sum += Float32(pi_cpu[lx, ly, lz, block_idx, 2])
            pzz_sum += Float32(pi_cpu[lx, ly, lz, block_idx, 3])
            pxy_sum += Float32(pi_cpu[lx, ly, lz, block_idx, 4])
            pxz_sum += Float32(pi_cpu[lx, ly, lz, block_idx, 5])
            pyz_sum += Float32(pi_cpu[lx, ly, lz, block_idx, 6])
        end
    end

    if n == 0
        return ntuple(_ -> 0.0f0, AMR_DECOMP_STATE_SIZE), false
    end

    inv_n = 1.0f0 / Float32(n)
    return (
        rho_sum * inv_n,
        jx_sum * inv_n,
        jy_sum * inv_n,
        jz_sum * inv_n,
        t_sum * inv_n,
        pxx_sum * inv_n,
        pyy_sum * inv_n,
        pzz_sum * inv_n,
        pxy_sum * inv_n,
        pxz_sum * inv_n,
        pyz_sum * inv_n
    ), true
end

@inline function _amr_decomp_store!(state::Matrix{Float32}, col::Int, values)
    @inbounds for i in 1:AMR_DECOMP_STATE_SIZE
        state[i, col] = values[i]
    end
end

"""
    build_amr_decomposed_transfer_buffers(registry, parent_level, child_level)

Build compact decomposed transfer buffers keyed exactly like
`registry.entities`. Parent state is a block-local parent donor average. Child
state is an average over the child-side face/edge/corner subset represented by
the registry entity.
"""
function build_amr_decomposed_transfer_buffers(registry::AMRInterfaceRegistry,
                                               parent_level::BlockLevel,
                                               child_level::BlockLevel)
    n_entities = length(registry.entities)
    parent_state = zeros(Float32, AMR_DECOMP_STATE_SIZE, n_entities)
    child_state = zeros(Float32, AMR_DECOMP_STATE_SIZE, n_entities)
    parent_valid = falses(n_entities)
    child_valid = falses(n_entities)

    parent_ptr_cpu = Array(parent_level.block_pointer)
    parent_rho_cpu = Array(parent_level.rho)
    parent_vel_cpu = Array(parent_level.vel)
    parent_obs_cpu = Array(parent_level.obstacle)
    parent_has_temp = has_thermal(parent_level)
    parent_temp_cpu = parent_has_temp ? Array(parent_level.temperature) : nothing
    parent_has_pi = has_pi_neq(parent_level)
    parent_pi_cpu = parent_has_pi ? Array(parent_level.pi_neq) : nothing

    child_rho_cpu = Array(child_level.rho)
    child_vel_cpu = Array(child_level.vel)
    child_obs_cpu = Array(child_level.obstacle)
    child_has_temp = has_thermal(child_level)
    child_temp_cpu = child_has_temp ? Array(child_level.temperature) : nothing
    child_has_pi = has_pi_neq(child_level)
    child_pi_cpu = child_has_pi ? Array(child_level.pi_neq) : nothing

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
                pvals, pvalid = _amr_decomp_average_state(
                    parent_rho_cpu,
                    parent_vel_cpu,
                    parent_obs_cpu,
                    parent_temp_cpu,
                    parent_pi_cpu,
                    parent_idx,
                    pxr,
                    pyr,
                    pzr,
                    parent_has_temp,
                    parent_has_pi
                )
                _amr_decomp_store!(parent_state, idx, pvals)
                parent_valid[idx] = pvalid
            end
        end

        child_idx = Int(entity.child_block_index)
        xr = _amr_decomp_cell_range(entity.sx, bs)
        yr = _amr_decomp_cell_range(entity.sy, bs)
        zr = _amr_decomp_cell_range(entity.sz, bs)
        cvals, cvalid = _amr_decomp_average_state(
            child_rho_cpu,
            child_vel_cpu,
            child_obs_cpu,
            child_temp_cpu,
            child_pi_cpu,
            child_idx,
            xr,
            yr,
            zr,
            child_has_temp,
            child_has_pi
        )
        _amr_decomp_store!(child_state, idx, cvals)
        child_valid[idx] = cvalid
    end

    return AMRDecomposedTransferBuffers(
        registry.parent_level_id,
        registry.child_level_id,
        parent_state,
        child_state,
        parent_valid,
        child_valid
    )
end

@inline function _amr_decomp_jdiff(parent_state::Matrix{Float32},
                                   child_state::Matrix{Float32},
                                   idx::Int)
    djx = child_state[AMR_DECOMP_JX, idx] - parent_state[AMR_DECOMP_JX, idx]
    djy = child_state[AMR_DECOMP_JY, idx] - parent_state[AMR_DECOMP_JY, idx]
    djz = child_state[AMR_DECOMP_JZ, idx] - parent_state[AMR_DECOMP_JZ, idx]
    return sqrt(Float64(djx*djx + djy*djy + djz*djz))
end

@inline function _amr_decomp_pidiff(parent_state::Matrix{Float32},
                                    child_state::Matrix{Float32},
                                    idx::Int)
    dxx = child_state[AMR_DECOMP_PIXX, idx] - parent_state[AMR_DECOMP_PIXX, idx]
    dyy = child_state[AMR_DECOMP_PIYY, idx] - parent_state[AMR_DECOMP_PIYY, idx]
    dzz = child_state[AMR_DECOMP_PIZZ, idx] - parent_state[AMR_DECOMP_PIZZ, idx]
    dxy = child_state[AMR_DECOMP_PIXY, idx] - parent_state[AMR_DECOMP_PIXY, idx]
    dxz = child_state[AMR_DECOMP_PIXZ, idx] - parent_state[AMR_DECOMP_PIXZ, idx]
    dyz = child_state[AMR_DECOMP_PIYZ, idx] - parent_state[AMR_DECOMP_PIYZ, idx]
    return sqrt(Float64(dxx*dxx + dyy*dyy + dzz*dzz + 2.0f0*(dxy*dxy + dxz*dxz + dyz*dyz)))
end

function summarize_amr_decomposed_transfer_buffers(registry::AMRInterfaceRegistry,
                                                   buffers::AMRDecomposedTransferBuffers)
    n_pairs = 0
    n_face_pairs = 0
    n_edge_pairs = 0
    n_corner_pairs = 0

    abs_drho_sum = 0.0
    drho_sq_sum = 0.0
    abs_dj_sum = 0.0
    dj_sq_sum = 0.0
    abs_dT_sum = 0.0
    dT_sq_sum = 0.0
    abs_dpi_sum = 0.0
    dpi_sq_sum = 0.0
    abs_drho_face_sum = 0.0
    abs_drho_edge_sum = 0.0
    abs_drho_corner_sum = 0.0
    abs_dj_face_sum = 0.0
    abs_dj_edge_sum = 0.0
    abs_dj_corner_sum = 0.0
    abs_dpi_face_sum = 0.0
    abs_dpi_edge_sum = 0.0
    abs_dpi_corner_sum = 0.0

    @inbounds for idx in eachindex(registry.entities)
        (buffers.parent_valid[idx] && buffers.child_valid[idx]) || continue
        entity = registry.entities[idx]

        drho = Float64(buffers.child_state[AMR_DECOMP_RHO, idx] -
                       buffers.parent_state[AMR_DECOMP_RHO, idx])
        abs_drho = abs(drho)
        dj = _amr_decomp_jdiff(buffers.parent_state, buffers.child_state, idx)
        dT = Float64(buffers.child_state[AMR_DECOMP_T, idx] -
                     buffers.parent_state[AMR_DECOMP_T, idx])
        abs_dT = abs(dT)
        dpi = _amr_decomp_pidiff(buffers.parent_state, buffers.child_state, idx)

        n_pairs += 1
        abs_drho_sum += abs_drho
        drho_sq_sum += drho * drho
        abs_dj_sum += dj
        dj_sq_sum += dj * dj
        abs_dT_sum += abs_dT
        dT_sq_sum += dT * dT
        abs_dpi_sum += dpi
        dpi_sq_sum += dpi * dpi

        if entity.topology == Int8(1)
            n_face_pairs += 1
            abs_drho_face_sum += abs_drho
            abs_dj_face_sum += dj
            abs_dpi_face_sum += dpi
        elseif entity.topology == Int8(2)
            n_edge_pairs += 1
            abs_drho_edge_sum += abs_drho
            abs_dj_edge_sum += dj
            abs_dpi_edge_sum += dpi
        elseif entity.topology == Int8(3)
            n_corner_pairs += 1
            abs_drho_corner_sum += abs_drho
            abs_dj_corner_sum += dj
            abs_dpi_corner_sum += dpi
        end
    end

    return (
        n_pairs = n_pairs,
        n_face_pairs = n_face_pairs,
        n_edge_pairs = n_edge_pairs,
        n_corner_pairs = n_corner_pairs,
        mean_abs_drho = n_pairs > 0 ? abs_drho_sum / n_pairs : 0.0,
        rms_drho = n_pairs > 0 ? sqrt(drho_sq_sum / n_pairs) : 0.0,
        mean_abs_dj = n_pairs > 0 ? abs_dj_sum / n_pairs : 0.0,
        rms_dj = n_pairs > 0 ? sqrt(dj_sq_sum / n_pairs) : 0.0,
        mean_abs_dT = n_pairs > 0 ? abs_dT_sum / n_pairs : 0.0,
        rms_dT = n_pairs > 0 ? sqrt(dT_sq_sum / n_pairs) : 0.0,
        mean_abs_dpi = n_pairs > 0 ? abs_dpi_sum / n_pairs : 0.0,
        rms_dpi = n_pairs > 0 ? sqrt(dpi_sq_sum / n_pairs) : 0.0,
        mean_abs_drho_face = n_face_pairs > 0 ? abs_drho_face_sum / n_face_pairs : 0.0,
        mean_abs_drho_edge = n_edge_pairs > 0 ? abs_drho_edge_sum / n_edge_pairs : 0.0,
        mean_abs_drho_corner = n_corner_pairs > 0 ? abs_drho_corner_sum / n_corner_pairs : 0.0,
        mean_abs_dj_face = n_face_pairs > 0 ? abs_dj_face_sum / n_face_pairs : 0.0,
        mean_abs_dj_edge = n_edge_pairs > 0 ? abs_dj_edge_sum / n_edge_pairs : 0.0,
        mean_abs_dj_corner = n_corner_pairs > 0 ? abs_dj_corner_sum / n_corner_pairs : 0.0,
        mean_abs_dpi_face = n_face_pairs > 0 ? abs_dpi_face_sum / n_face_pairs : 0.0,
        mean_abs_dpi_edge = n_edge_pairs > 0 ? abs_dpi_edge_sum / n_edge_pairs : 0.0,
        mean_abs_dpi_corner = n_corner_pairs > 0 ? abs_dpi_corner_sum / n_corner_pairs : 0.0
    )
end

function write_amr_decomposed_transfer_csv_header(filepath::String)
    open(filepath, "w") do io
        println(io,
            "step,parent_level,child_level," *
            "n_pairs,n_face_pairs,n_edge_pairs,n_corner_pairs," *
            "mean_abs_drho,rms_drho,mean_abs_dj,rms_dj," *
            "mean_abs_dT,rms_dT,mean_abs_dpi,rms_dpi," *
            "mean_abs_drho_face,mean_abs_drho_edge,mean_abs_drho_corner," *
            "mean_abs_dj_face,mean_abs_dj_edge,mean_abs_dj_corner," *
            "mean_abs_dpi_face,mean_abs_dpi_edge,mean_abs_dpi_corner")
    end
end

function append_amr_decomposed_transfer_csv_row(filepath::String, step::Int, parent_level::Int,
                                                child_level::Int, summary)
    if !isfile(filepath)
        write_amr_decomposed_transfer_csv_header(filepath)
    end
    open(filepath, "a") do io
        @printf(io,
            "%d,%d,%d,%d,%d,%d,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",
            step,
            parent_level,
            child_level,
            summary.n_pairs,
            summary.n_face_pairs,
            summary.n_edge_pairs,
            summary.n_corner_pairs,
            summary.mean_abs_drho,
            summary.rms_drho,
            summary.mean_abs_dj,
            summary.rms_dj,
            summary.mean_abs_dT,
            summary.rms_dT,
            summary.mean_abs_dpi,
            summary.rms_dpi,
            summary.mean_abs_drho_face,
            summary.mean_abs_drho_edge,
            summary.mean_abs_drho_corner,
            summary.mean_abs_dj_face,
            summary.mean_abs_dj_edge,
            summary.mean_abs_dj_corner,
            summary.mean_abs_dpi_face,
            summary.mean_abs_dpi_edge,
            summary.mean_abs_dpi_corner)
    end
end

function amr_decomposed_transfer_csv_path(amr_interface_csv_path::String)
    return joinpath(dirname(amr_interface_csv_path), "amr_decomposed_transfer_history.csv")
end
