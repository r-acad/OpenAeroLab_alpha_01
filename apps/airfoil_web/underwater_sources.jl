module UnderwaterSourceProvenance
using SHA
export source_paths, capture_sources, verify_sources!, sources_current, write_snapshot

function source_paths(worker;projection=false)
    app=dirname(abspath(worker));src=normpath(joinpath(app,"..","..","src","cmesh"))
    names=["cmesh_free_surface_airfoil.jl","cmesh_free_surface_airfoil_initialization.jl",
        "cmesh_free_surface.jl","cmesh_free_surface_hllc.jl","cmesh_free_surface_ausm.jl",
        "cmesh_free_surface_hydrostatic.jl","cmesh_free_surface_cartesian.jl","cmesh_free_surface_gate.jl",
        "cmesh_free_surface_ka.jl","cmesh_free_surface_thornber.jl","cmesh_gen.jl",
        "cmesh_quality_metrics.jl","cmesh_strip_refinement.jl","airfoil_conditioning.jl"]
    append!(names,projection ? ["cmesh_free_surface_projection.jl","cmesh_free_surface_projection_ka.jl"] : ["cmesh_free_surface_airfoil_ka.jl"])
    paths=vcat([abspath(worker),abspath(@__FILE__),joinpath(app,"protocol.jl"),joinpath(app,"full_field_artifact.jl")],joinpath.(Ref(src),names))
    projection && push!(paths,joinpath(app,"cuda_projection_triangular.jl"))
    paths
end

function capture_sources(paths)
    length(unique(basename.(paths)))==length(paths) || error("source snapshot has duplicate basenames")
    bytes=Dict(basename(p)=>read(p) for p in paths)
    hashes=Dict(name=>bytes2hex(sha256(value)) for (name,value) in bytes)
    (;paths=abspath.(paths),bytes,hashes)
end
sources_current(captured)=all(isfile(p) && read(p)==captured.bytes[basename(p)] for p in captured.paths)
function verify_sources!(captured)
    sources_current(captured) || error("underwater source changed after capture; loaded definitions refused")
    nothing
end
function write_snapshot(captured,out)
    mkpath(out)
    for (name,bytes) in captured.bytes;write(joinpath(out,name),bytes);end
    nothing
end
end
