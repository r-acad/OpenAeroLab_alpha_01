# ==============================================================================
# RAW STATE DUMP (27 Oct 2026, opt-in diagnostic)
# ==============================================================================
# `advanced.diagnostics.raw_state_dump_freq: N` (N > 0) writes the macroscopic
# state of every level as raw little-endian binaries every N steps (at batch
# ends, after materialize_current_state_for_io!) and at the final step, into
#   <output_dir>/raw_state/step_<NNNNNN>/level_<L>.bin
# Default 0 ⇒ nothing is written and nothing else changes.
#
# Layout of level_<L>.bin (all little-endian):
#   Int32  magic = 0x4C42_4D31 ("LBM1")
#   Int32  level_id, n_blocks, block_size, grid_dim_x, grid_dim_y, grid_dim_z
#   Float64 dx
#   Int32  coords[3, n_blocks]                    (bx, by, bz) per block
#   Float32 rho[bs, bs, bs, n_blocks]
#   Float32 vel[bs, bs, bs, n_blocks, 3]
#   UInt8  obstacle[bs, bs, bs, n_blocks]          (1 = solid)
# Readers: memory/project_docs/_records/2026-10-27_lbm_periodic_amr/tools/raw_state.jl.
# ==============================================================================

function dump_raw_state(grids, output_dir::AbstractString, step::Integer)
    dir = joinpath(output_dir, "raw_state", @sprintf("step_%06d", step))
    mkpath(dir)
    for level in grids
        n = length(level.active_block_coords)
        bs = Int(BLOCK_SIZE)
        rho = Array{Float32}(Array(level.rho))
        vel = Array{Float32}(Array(level.vel))
        obs = Array(level.obstacle)
        coords = zeros(Int32, 3, n)
        for (i, (bx, by, bz)) in enumerate(level.active_block_coords)
            coords[1, i] = Int32(bx); coords[2, i] = Int32(by); coords[3, i] = Int32(bz)
        end
        path = joinpath(dir, @sprintf("level_%d.bin", level.level_id))
        open(path, "w") do io
            write(io, Int32(0x4C424D31))
            write(io, Int32(level.level_id), Int32(n), Int32(bs),
                  Int32(level.grid_dim_x), Int32(level.grid_dim_y), Int32(level.grid_dim_z))
            write(io, Float64(level.dx))
            write(io, coords)
            write(io, rho)
            write(io, vel)
            write(io, UInt8.(obs))
        end
    end
    println("[RawState] step $step: $(length(grids)) level(s) written to $dir")
    return dir
end
