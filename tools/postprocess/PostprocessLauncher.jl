module LudwigPostprocess

using TOML, UUIDs

export launch, available

const DIRECTORY = @__DIR__
available() = isfile(joinpath(DIRECTORY, "render.jl")) && isfile(joinpath(DIRECTORY, "Project.toml"))

_toml(x::Nothing) = "unavailable"
_toml(x::AbstractDict) = Dict{String,Any}(String(k) => _toml(v) for (k,v) in pairs(x))
_toml(x::AbstractVector) = [_toml(v) for v in x]
_toml(x::Union{AbstractString,Bool,Integer}) = x
_toml(x::AbstractFloat) = isfinite(x) ? x : string(x)
_toml(x) = string(x)

"""
    launch(request; log_path, wait=false) -> Process

Run the headless CairoMakie postprocessor in its own Julia environment. The
request uses source_directory/destination_directory plus report, charts,
render_3d, chart_format, surface_quantity, camera and image dimensions.
With wait=true, inspect success(process) and destination/manifest.json; an
optional plot failure does not change the solver's completion status.
"""
function launch(request::AbstractDict; log_path::AbstractString, wait::Bool=false)
    available() || error("The Makie postprocessor is unavailable")
    data = Dict{String,Any}(String(k)=>v for (k,v) in pairs(request))
    for key in ("source_directory", "destination_directory")
        haskey(data,key) || throw(ArgumentError("Missing $key for Makie postprocessing"))
        path = abspath(String(data[key]))
        key == "source_directory" && !isdir(path) && throw(ArgumentError("Postprocess source directory does not exist"))
        data[key] = path
    end
    mkpath(data["destination_directory"])
    log = abspath(String(log_path)); mkpath(dirname(log))
    request_file = joinpath(dirname(log), "makie_request_$(uuid4()).toml")
    open(request_file,"w") do io
        TOML.print(io,_toml(data);sorted=true)
    end
    julia = joinpath(Sys.BINDIR,Base.julia_exename())
    cmd = Cmd(`$julia --startup-file=no --project=$DIRECTORY -t 1 $(joinpath(DIRECTORY,"render.jl")) $request_file`; dir=DIRECTORY,windows_hide=true)
    output = open(log,"w")
    process = try
        run(pipeline(cmd;stdout=output,stderr=output);wait=false)
    catch
        close(output); rethrow()
    end
    if wait
        try Base.wait(process) finally close(output) end
    else
        @async try Base.wait(process) finally close(output) end
    end
    process
end

end
