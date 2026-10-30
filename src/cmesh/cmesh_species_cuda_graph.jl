# Optional CUDA-only executor. Include after cmesh_cav_species.jl and CUDA.
# Physical state, admission, rollback, clocks and accounting remain owned by
# step_species_ka!. Only its successful device operation sequence is captured.
module CMeshSpeciesCUDAGraph
using CUDA
import ..CMeshCavSpecies
const P=CMeshCavSpecies
export SpeciesGraphExecutor

mutable struct SpeciesGraphExecutor{C}
    const cache::C
    key::Union{Nothing,Tuple{Float32,Int32,Bool,Float32,Int32}}
    graph::Union{Nothing,CUDA.CuGraphExec}
    captures::Int
end
function SpeciesGraphExecutor(c::P.SpeciesKA)
    c.solver.backend isa CUDA.CUDABackend || throw(ArgumentError("species graphs require CUDA"))
    SpeciesGraphExecutor(c,nothing,nothing,0)
end
function (e::SpeciesGraphExecutor)(c,cfl::Float32,wallmode::Int32)
    c===e.cache || throw(ArgumentError("species graph belongs to another cache"))
    key=P._species_execution_key(c,cfl,wallmode)
    if e.key!==key || e.graph===nothing
        # Precompile and populate lazy batch specs before capture. The public
        # wrapper already saved the complete base state in c.backup. Warmup is
        # speculative, never published, and restored before the real attempt.
        metadata=P._step_metadata(c)
        try
            P._enqueue_species_deferred!(c,cfl;wallmode)
            CUDA.synchronize()
        finally
            P._restore_step!(c,metadata)
        end
        graph=CUDA.capture() do
            P._enqueue_species_deferred!(c,cfl;wallmode)
        end
        e.graph=CUDA.instantiate(graph)
        e.key=key
        e.captures+=1
    end
    # max_dt lives in c.step_limit, written before this call. Changing output
    # landing times and retry limits therefore never reuses a stale scalar.
    CUDA.launch(e.graph)
    nothing
end
end
