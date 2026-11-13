# Narrow file-load provenance for isolated RANS workers. This is a guarded
# startup cohort, not a Julia method fingerprint or a general include loader.
using SHA

const _RANS_RESTART_LOADED_SOURCES = Ref{Union{Nothing,Dict{String,Any}}}(nothing)
const _RANS_RESTART_SOURCE_PROVENANCE = "worker_file_load_snapshot_v1"

function _rans_restart_capture_solver_sources(src)
    folder=joinpath(src,"compressible")
    names=sort!(unique(vcat(["cmesh_solver.jl","cmesh_solver_ka.jl",
        "cmesh_geometric_reconstruction_ka.jl"],
        filter(name->startswith(name,"cmesh_steady_")&&endswith(name,".jl"),readdir(folder)))))
    Dict(name=>bytes2hex(open(sha256,joinpath(folder,name))) for name in names)
end

function _rans_restart_capture_source_snapshot(src=SRC;
    native_dir=joinpath(@__DIR__,"..","runtime"))
    Dict{String,Any}(
        "solver_sha256"=>_rans_restart_capture_solver_sources(src),
        "geometry_sha256"=>Dict(path=>bytes2hex(open(sha256,joinpath(src,split(path,'/')...)))
            for path in ("cmesh/cmesh_metrics.jl","collision/lattice.jl")),
        "flux_sha256"=>Dict(name=>bytes2hex(open(sha256,joinpath(src,"cmesh",name)))
            for name in ("cmesh_flux_core.jl","cmesh_sst_nonorth.jl","cmesh_wallfun.jl",
                         "cmesh_eos_mix.jl","cmesh_pressure_outlet.jl","cmesh_species_sst.jl")),
        "native_sha256"=>Dict(name=>bytes2hex(open(sha256,joinpath(native_dir,name)))
            for name in ("gas_strip_prototype.jl","gas_strip_sst_prototype.jl","gas_strip_sst_steady.jl")))
end

function _rans_restart_bind_source_snapshot!(before;src=SRC,
    native_dir=joinpath(@__DIR__,"..","runtime"))
    _RANS_RESTART_LOADED_SOURCES[]===nothing || error("RANS worker source snapshot is already bound")
    after=_rans_restart_capture_source_snapshot(src;native_dir)
    before==after || error("RANS numerical files changed while worker sources were loading; restart the worker")
    _RANS_RESTART_LOADED_SOURCES[]=deepcopy(before)
    nothing
end

function _rans_restart_loaded_source_identity()
    source=_RANS_RESTART_LOADED_SOURCES[]
    source===nothing && error("RANS worker source loading has not established a verified snapshot")
    deepcopy(source) # A returned checkpoint identity must not mutate the binding.
end

_rans_restart_solver_sources() = _rans_restart_loaded_source_identity()["solver_sha256"]
