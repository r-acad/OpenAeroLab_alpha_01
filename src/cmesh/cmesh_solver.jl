# Compatibility path for existing scripts. The legacy CPU solver lives in src/compressible/.
include(joinpath(@__DIR__, "..", "compressible", "cmesh_solver.jl"))
