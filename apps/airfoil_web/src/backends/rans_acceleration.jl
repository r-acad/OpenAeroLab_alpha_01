module RANSAcceleration

using SHA

export policy, configure_compressible!

const METHODS=("standard","irsj","newton")
# "newton": the implicit Newton march on the marched mesh (CPU, Float64). It has its own worker and configuration
# (backends/newton_backend.jl, workers/newton_worker.jl).
const NEWTON_LABEL="implicit Newton-Krylov march (CPU, Float64, marched mesh)"

function numerical_source_receipt()
    root=normpath(joinpath(@__DIR__,"..","..","..",".."))
    paths=vcat([joinpath("src","compressible",name) for name in readdir(joinpath(root,"src","compressible"))
        if startswith(name,"cmesh_steady_")&&endswith(name,".jl")],
        [joinpath("src","compressible","cmesh_solver_ka.jl"),
         joinpath("src","compressible","cmesh_geometric_reconstruction_ka.jl"),joinpath("src","cmesh","cmesh_flux_core.jl"),
         joinpath("src","cmesh","cmesh_wallfun.jl"),joinpath("src","cmesh","cmesh_metrics.jl"),joinpath("src","collision","lattice.jl")])
    Dict(replace(path,'\\'=>'/')=>bytes2hex(open(sha256,joinpath(root,path))) for path in paths)
end

"Select an actual iteration method; never silently substitute a different solver."
function configure_compressible!(cfg,flow;coupled=false)
    method=String(get(flow,"rans_method",get(cfg,"irsj",false) ? "irsj" : "standard"))
    method in METHODS || throw(ArgumentError("Unknown steady RANS method: $method"))
    method!="standard" && get(cfg,"time_mode","steady")!="steady" &&
        throw(ArgumentError("The selected RANS iteration method requires steady mode"))
    coupled && method=="newton" && throw(ArgumentError(
        "The Implicit Newton method has its own worker and marched mesh; it cannot be configured for the coupled C+H driver."))
    cfg["rans_method"]=method
    cfg["rans_acceleration"]=method!="standard"
    labels=Dict("standard"=>"standard explicit", "irsj"=>"j-line implicit residual smoothing","newton"=>NEWTON_LABEL)
    cfg["rans_acceleration_method"]=labels[method]
    cfg["irsj"]=method=="irsj"
    cfg
end

"""Steady RANS acceleration selected by the airfoil app. The default preserves legacy numerics.
The low-speed pressure solver is already implicit; its accelerated setting raises only
the requested pseudo-time ceiling, leaving its nonlinear rejection/retry in charge.
These settings are experimental and do not imply a measured wall-clock speedup."""
function policy(enabled::Bool, family::Symbol; physical::Bool=false)
    family in (:compressible, :compressible_ch, :lowspeed, :lowspeed_ch, :species_air) ||
        throw(ArgumentError("unsupported steady RANS family: $family"))
    enabled && physical && throw(ArgumentError("RANS convergence acceleration applies to steady pseudo-time only"))
    method = !enabled ? "standard" : family in (:compressible, :compressible_ch) ?
        "j-line implicit residual smoothing" : family in (:lowspeed, :lowspeed_ch) ?
        "adaptive implicit backward Euler, requested pseudo-time ceiling 0.5 c/U" :
        "wall-normal line implicit relaxation"
    (; enabled, method,
       irsj=enabled && family in (:compressible, :compressible_ch),
       pseudo_dt=enabled && family in (:lowspeed, :lowspeed_ch) ? 0.5 : 0.25,
       species_line=enabled && family === :species_air)
end

end
