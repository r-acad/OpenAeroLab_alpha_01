"""
OctreeQ19: the CPU-only D3Q19 lattice Boltzmann solver of Aero Lab on a cell-level octree mesh.

No CUDA, no KernelAbstractions. Plan and design: 02_PRIVATE_PROJECT_DEVELOPMENT/memory/project_docs/
2026-11-03_CPU_OCTREE_D3Q19_PLAN.md and _records/2026-11-03_cpu_octree_q19/DESIGN.md.
"""
module OctreeQ19

using Printf, LinearAlgebra, TOML, SHA, Dates
import WriteVTK

include("lattice.jl")
include("geometry.jl")
include("octree.jl")
include("generator.jl")
include("config.jl")
include("io_build.jl")
include("collision.jl")
include("state.jl")
include("walls.jl")
include("stream.jl")
include("coupling.jl")
include("io_run.jl")
include("run.jl")

end
