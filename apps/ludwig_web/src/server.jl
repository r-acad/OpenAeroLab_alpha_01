module LudwigWeb

using HTTP

include(joinpath(@__DIR__, "..", "..", "airfoil_web", "src", "server.jl"))
include(joinpath(@__DIR__, "aero_backend.jl"))

export serve, app_handler

const APP_DIR = normpath(joinpath(@__DIR__, ".."))
const STATIC_DIR = joinpath(APP_DIR, "static")
const ASSETS = Dict(
    "/workbench-static/aero_reference.js" => ("aero_reference.js", "application/javascript; charset=utf-8"),
    "/workbench-static/home.css" => ("home.css", "text/css; charset=utf-8"),
    "/workbench-static/solver_catalog.css" => ("solver_catalog.css", "text/css; charset=utf-8"),
    "/workbench-static/aero_lab.css" => ("aero_lab.css", "text/css; charset=utf-8"),
    "/workbench-static/aero_lab.js" => ("aero_lab.js", "application/javascript; charset=utf-8"),
    "/workbench-static/aero_mesh_inspector.js" => ("aero_mesh_inspector.js", "application/javascript; charset=utf-8"),
    "/workbench-static/aero_sections.js" => ("aero_sections.js", "application/javascript; charset=utf-8"),
    "/workbench-static/aero_result_browser.js" => ("aero_result_browser.js", "application/javascript; charset=utf-8"),
    "/workbench-static/aero_studies.js" => ("aero_studies.js", "application/javascript; charset=utf-8"),
    "/workbench-static/aero_results.js" => ("aero_results.js", "application/javascript; charset=utf-8"),
    "/workbench-static/aero_layout.js" => ("aero_layout.js", "application/javascript; charset=utf-8"),
    "/workbench-static/aero_model_tools.js" => ("aero_model_tools.js", "application/javascript; charset=utf-8"),
    "/workbench-static/vendor/babylonjs.loaders.min.js" => (joinpath("vendor", "babylonjs.loaders.min.js"), "application/javascript; charset=utf-8"),
)

function _asset(relative::AbstractString, mime::AbstractString)
    path = joinpath(STATIC_DIR, relative)
    isfile(path) || return HTTP.Response(404, "Asset unavailable")
    HTTP.Response(200, ["Content-Type" => mime, "Cache-Control" => "no-store"], read(path))
end

function app_handler(req::HTTP.Request)
    path = String(HTTP.URI(req.target).path)
    method = String(req.method)
    if startswith(path, "/api/aero/")
        return AeroBackend.handle(req, path)
    elseif method == "GET" && path in ("/", "/index.html")
        return _asset("index.html", "text/html; charset=utf-8")
    elseif method == "GET" && path in ("/solvers", "/solvers/")
        return _asset("solver_catalog.html", "text/html; charset=utf-8")
    elseif method == "GET" && path in ("/aero-lab", "/aero-lab/")
        return _asset("aero_lab.html", "text/html; charset=utf-8")
    elseif method == "GET" && path in ("/airfoil-lab", "/airfoil-lab/")
        return AirfoilWeb._static_response("/")
    elseif method == "GET" && haskey(ASSETS, path)
        file, mime = ASSETS[path]
        return _asset(file, mime)
    elseif startswith(path, "/common-static/") || startswith(path, "/static/") || startswith(path, "/api/") ||
           path in ("/favicon.svg", "/app.js", "/app.css")
        return AirfoilWeb.app_handler(req)
    end
    HTTP.Response(404, ["Content-Type" => "text/plain; charset=utf-8"], "Not found")
end

function serve(; host::String="127.0.0.1", port::Int=8058, on_listening::Function=()->nothing)
    host in ("127.0.0.1", "localhost", "::1") || throw(ArgumentError("Workbench binds to loopback only"))
    mkpath(AirfoilWeb.RUN_DIR)
    AirfoilWeb._start_worker()
    server = HTTP.serve!(app_handler, host, port; verbose=false)
    println("Ludwig Workbench at http://$host:$port/")
    flush(stdout)
    try
        on_listening()
    catch err
        @warn "Browser launch failed; the workbench is still running" exception=err
    end
    wait(server)
end

end
