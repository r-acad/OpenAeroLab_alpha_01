"""
MULTI_GPU.JL — Strategy C: Replicated Coarse + Partitioned Fine

Both GPUs hold complete copies of all grid arrays.
- Coarse levels (1..split_level): computed redundantly on both GPUs
- Fine levels (split_level+1..N): each GPU computes half the blocks,
  then results are merged via P2P copy over NVLink.
"""

using CUDA
using KernelAbstractions
using Adapt

mutable struct MultiGPUState
    n_gpus::Int
    device_ids::Vector{Int}                      # CUDA device IDs [0, 1]
    grids::Vector{Vector}                        # grids[gpu_idx][level] — GPU-resident BlockLevels
    lattice_arrays::Vector{NTuple{7,Any}}        # Per-GPU (cx,cy,cz,w,opp,my,mz)
    split_level::Int                             # Levels 1..split replicated, rest partitioned
    backends::Vector{CUDABackend}                # One backend per GPU
end

"""
    setup_multi_gpu(cpu_grids, split_level) -> MultiGPUState

Enable P2P access, upload grid data to both GPUs, build lattice arrays on each.
`cpu_grids` should be the CPU-side BlockLevel vector (before any adapt).
"""
function setup_multi_gpu(cpu_grids::Vector, split_level::Int)
    n_gpus = Int(CUDA.ndevices())
    @assert n_gpus >= 2 "Multi-GPU requires at least 2 CUDA devices, found $n_gpus"

    device_ids = [0, 1]
    println("[MultiGPU] Setting up $n_gpus GPUs: devices $device_ids")

    # Enable peer-to-peer access
    for i in device_ids, j in device_ids
        if i != j
            if CUDA.can_access_peer(CuDevice(i), CuDevice(j))
                try
                    # Get context of peer device, then enable access from source
                    CUDA.device!(j)
                    peer_ctx = CUDA.context()
                    CUDA.device!(i)
                    CUDA.enable_peer_access(peer_ctx)
                    println("[MultiGPU] P2P: GPU $i -> GPU $j enabled")
                catch e
                    println("[MultiGPU] P2P: GPU $i -> GPU $j already enabled or failed: $e")
                end
            else
                @warn "P2P access not available between GPU $i and GPU $j — performance will be reduced"
            end
        end
    end

    # Upload grids to both GPUs
    all_grids = Vector{Vector}(undef, 2)
    all_lattice = Vector{NTuple{7,Any}}(undef, 2)
    all_backends = Vector{CUDABackend}(undef, 2)

    for (gpu_idx, dev_id) in enumerate(device_ids)
        CUDA.device!(dev_id)
        backend = CUDABackend()
        all_backends[gpu_idx] = backend

        println("[MultiGPU] Uploading grids to GPU $dev_id...")
        gpu_grids = [adapt(backend, g) for g in cpu_grids]
        all_grids[gpu_idx] = gpu_grids

        lat = build_lattice_arrays_gpu(backend)
        all_lattice[gpu_idx] = lat
    end

    # Reset to GPU 0
    CUDA.device!(0)

    mgpu = MultiGPUState(2, device_ids, all_grids, all_lattice, split_level, all_backends)

    # Print memory summary
    for (gpu_idx, dev_id) in enumerate(device_ids)
        CUDA.device!(dev_id)
        free_mem, total_mem = CUDA.memory_info()
        Printf.@printf("[MultiGPU] GPU %d: %.2f GB free / %.2f GB total\n",
                        dev_id, free_mem/1024^3, total_mem/1024^3)
    end
    CUDA.device!(0)

    return mgpu
end

"""
    sync_gpu!(dev_id)

Synchronize a specific GPU device.
"""
function sync_gpu!(dev_id::Int)
    CUDA.device!(dev_id)
    CUDA.synchronize()
end

"""
    sync_all_gpus!(mgpu)

Synchronize all GPUs.
"""
function sync_all_gpus!(mgpu::MultiGPUState)
    for dev_id in mgpu.device_ids
        sync_gpu!(dev_id)
    end
end

"""
    merge_level_arrays!(mgpu, level_idx)

After a partitioned timestep, merge results between GPUs via P2P copyto!.
GPU 0 computed blocks 1..n_half, GPU 1 computed blocks n_half+1..n_blocks.
Each GPU copies its computed portion to the other GPU.

Only merges arrays that were written by the collision kernel:
f_out (or f/f_temp based on parity), vel_out, rho, f_post_collision,
and thermal arrays (g_out, temperature, kinetic_energy) if active.
"""
function merge_level_arrays!(mgpu::MultiGPUState, level_idx::Int, t_sub::Int)
    g1 = mgpu.grids[1][level_idx]  # GPU 0's level
    g2 = mgpu.grids[2][level_idx]  # GPU 1's level

    n_blocks = length(g1.active_block_coords)
    if n_blocks <= 1; return; end

    n_half = n_blocks ÷ 2
    BS = BLOCK_SIZE

    # Determine which buffers are "out" based on ping-pong parity
    if iseven(t_sub)
        f_out_1, f_out_2 = g1.f_temp, g2.f_temp
        vel_out_1, vel_out_2 = g1.vel_temp, g2.vel_temp
    else
        f_out_1, f_out_2 = g1.f, g2.f
        vel_out_1, vel_out_2 = g1.vel, g2.vel
    end

    # GPU 0 computed blocks 1..n_half → copy to GPU 1's arrays
    # GPU 1 computed blocks n_half+1..n_blocks → copy to GPU 0's arrays
    n_dirs = Int(N_DIRS)

    # --- Copy GPU 0's results (blocks 1..n_half) to GPU 1 ---
    # f_out
    src_f_0 = view(f_out_1, :, :, :, 1:n_half, :)
    dst_f_0 = view(f_out_2, :, :, :, 1:n_half, :)
    copyto!(dst_f_0, src_f_0)

    # rho
    src_rho_0 = view(g1.rho, :, :, :, 1:n_half)
    dst_rho_0 = view(g2.rho, :, :, :, 1:n_half)
    copyto!(dst_rho_0, src_rho_0)

    # vel_out
    src_vel_0 = view(vel_out_1, :, :, :, 1:n_half, :)
    dst_vel_0 = view(vel_out_2, :, :, :, 1:n_half, :)
    copyto!(dst_vel_0, src_vel_0)

    # f_post_collision (needed for Bouzidi)
    if g1.bouzidi_enabled && g1.n_boundary_cells > 0
        src_fpc_0 = view(g1.f_post_collision, :, :, :, 1:n_half, :)
        dst_fpc_0 = view(g2.f_post_collision, :, :, :, 1:n_half, :)
        copyto!(dst_fpc_0, src_fpc_0)
    end

    # --- Copy GPU 1's results (blocks n_half+1..n_blocks) to GPU 0 ---
    b_start = n_half + 1

    src_f_1 = view(f_out_2, :, :, :, b_start:n_blocks, :)
    dst_f_1 = view(f_out_1, :, :, :, b_start:n_blocks, :)
    copyto!(dst_f_1, src_f_1)

    src_rho_1 = view(g2.rho, :, :, :, b_start:n_blocks)
    dst_rho_1 = view(g1.rho, :, :, :, b_start:n_blocks)
    copyto!(dst_rho_1, src_rho_1)

    src_vel_1 = view(vel_out_2, :, :, :, b_start:n_blocks, :)
    dst_vel_1 = view(vel_out_1, :, :, :, b_start:n_blocks, :)
    copyto!(dst_vel_1, src_vel_1)

    if g1.bouzidi_enabled && g1.n_boundary_cells > 0
        src_fpc_1 = view(g2.f_post_collision, :, :, :, b_start:n_blocks, :)
        dst_fpc_1 = view(g1.f_post_collision, :, :, :, b_start:n_blocks, :)
        copyto!(dst_fpc_1, src_fpc_1)
    end

    # --- Thermal DDF merge ---
    ddf_active = DDF_ENABLED && has_thermal(g1)
    if ddf_active
        if iseven(t_sub)
            g_out_1, g_out_2 = g1.g_temp, g2.g_temp
        else
            g_out_1, g_out_2 = g1.g, g2.g
        end

        # GPU 0 → GPU 1 (blocks 1..n_half)
        copyto!(view(g_out_2, :, :, :, 1:n_half, :),
                view(g_out_1, :, :, :, 1:n_half, :))
        copyto!(view(g2.temperature, :, :, :, 1:n_half),
                view(g1.temperature, :, :, :, 1:n_half))
        copyto!(view(g2.kinetic_energy, :, :, :, 1:n_half),
                view(g1.kinetic_energy, :, :, :, 1:n_half))

        # GPU 1 → GPU 0 (blocks n_half+1..n_blocks)
        copyto!(view(g_out_1, :, :, :, b_start:n_blocks, :),
                view(g_out_2, :, :, :, b_start:n_blocks, :))
        copyto!(view(g1.temperature, :, :, :, b_start:n_blocks),
                view(g2.temperature, :, :, :, b_start:n_blocks))
        copyto!(view(g1.kinetic_energy, :, :, :, b_start:n_blocks),
                view(g2.kinetic_energy, :, :, :, b_start:n_blocks))

        # g_post_collision merge (for thermal Bouzidi)
        if g1.bouzidi_enabled && g1.n_boundary_cells > 0
            copyto!(view(g2.g_post_collision, :, :, :, 1:n_half, :),
                    view(g1.g_post_collision, :, :, :, 1:n_half, :))
            copyto!(view(g1.g_post_collision, :, :, :, b_start:n_blocks, :),
                    view(g2.g_post_collision, :, :, :, b_start:n_blocks, :))
        end
    end
end

"""
    init_equilibrium_multi_gpu!(mgpu, w_gpu_per_device, params)

Initialize equilibrium on both GPUs (same as single-GPU but on each device).
Called from main.jl after setup_multi_gpu.
"""
function init_equilibrium_multi_gpu!(mgpu::MultiGPUState, params)
    @kernel function init_eq_mgpu!(f, f_temp, f_old, rho_old, vel_old, W,
                                    has_old::Int32, n_dirs::Int32,
                                    u_init_x::Float32, u_init_y::Float32, u_init_z::Float32)
        x, y, z, b = @index(Global, NTuple)
        @inbounds begin
            for k in 1:n_dirs
                f[x, y, z, b, k] = W[k]
                f_temp[x, y, z, b, k] = W[k]
            end
            if has_old == Int32(1)
                for k in 1:n_dirs; f_old[x, y, z, b, k] = W[k]; end
                rho_old[x, y, z, b] = 1.0f0
                vel_old[x, y, z, b, 1] = u_init_x
                vel_old[x, y, z, b, 2] = u_init_y
                vel_old[x, y, z, b, 3] = u_init_z
            end
        end
    end

    @kernel function init_thermal_eq_mgpu!(g, g_temp, g_old, g_post_collision,
                                            temperature, temperature_old, kinetic_energy,
                                            W, T_init::Float32, has_old::Int32, has_bouzidi::Int32,
                                            n_dirs_g::Int32, coeff_g::Float32)
        x, y, z, b = @index(Global, NTuple)
        @inbounds begin
            for k in 1:n_dirs_g
                g_eq = coeff_g * T_init * W[k]
                g[x, y, z, b, k] = g_eq
                g_temp[x, y, z, b, k] = g_eq
            end
            temperature[x, y, z, b] = T_init
            temperature_old[x, y, z, b] = T_init
            kinetic_energy[x, y, z, b] = 0.0f0
            if has_old == Int32(1)
                for k in 1:n_dirs_g; g_old[x, y, z, b, k] = coeff_g * T_init * W[k]; end
            end
            if has_bouzidi == Int32(1)
                for k in 1:n_dirs_g; g_post_collision[x, y, z, b, k] = coeff_g * T_init * W[k]; end
            end
        end
    end

    for (gpu_idx, dev_id) in enumerate(mgpu.device_ids)
        CUDA.device!(dev_id)
        backend = mgpu.backends[gpu_idx]
        _, _, _, w_gpu, _, _, _ = mgpu.lattice_arrays[gpu_idx]
        grids = mgpu.grids[gpu_idx]

        for lvl in 1:length(grids)
            level = grids[lvl]
            n = length(level.active_block_coords)
            if n > 0
                has_old = has_temporal_storage(level) ? Int32(1) : Int32(0)
                # vel_old stores lab-frame velocity; f=w corresponds to u_lab=0
                init_eq_mgpu!(backend)(level.f, level.f_temp, level.f_old,
                                       level.rho_old, level.vel_old, w_gpu, has_old, N_DIRS,
                                       0.0f0, 0.0f0, 0.0f0,
                                       ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n))

                if DDF_ENABLED && has_thermal(level)
                    level.tau_g = params.tau_g_levels[lvl]
                    has_old_g = (length(level.g_old) > N_DIRS_THERMAL) ? Int32(1) : Int32(0)
                    has_bouz_g = (level.bouzidi_enabled && level.n_boundary_cells > 0) ? Int32(1) : Int32(0)
                    coeff_g_init = (LATTICE_TYPE == :d3q39) ? Float32(DDF_CV - 1.5 * Float64(CS2_PHYSICS)) : 1.0f0
                    init_thermal_eq_mgpu!(backend)(level.g, level.g_temp, level.g_old, level.g_post_collision,
                                                    level.temperature, level.temperature_old, level.kinetic_energy,
                                                    w_gpu, Float32(DDF_T_INITIAL), has_old_g, has_bouz_g,
                                                    N_DIRS_THERMAL, coeff_g_init,
                                                    ndrange=(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, n))
                end
            end
        end
        KernelAbstractions.synchronize(backend)
    end

    CUDA.device!(0)
end
