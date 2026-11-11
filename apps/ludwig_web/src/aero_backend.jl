module AeroBackend

using HTTP, JSON3, UUIDs, Dates, Printf, TOML, Base64, SHA

const PUBLIC_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const PRIVATE_ROOT = normpath(get(ENV, "LUDWIG_PRIVATE_ROOT",
    joinpath(PUBLIC_ROOT, "..", "02_PRIVATE_PROJECT_DEVELOPMENT")))
const RUNTIME_ROOT = joinpath(PRIVATE_ROOT, "results", "aero_lab")
const CASES_ROOT = joinpath(RUNTIME_ROOT, "CASES")
const MODEL_ROOT = joinpath(RUNTIME_ROOT, "models")
const JOBS_ROOT = joinpath(RUNTIME_ROOT, "jobs")
const MAIN_SOLVER = PUBLIC_ROOT
const PINNED_SOLVER = joinpath(PUBLIC_ROOT, "solvers", "lbm_lbmtau")
const Q19_SOLVER = joinpath(PUBLIC_ROOT, "src", "octree_q19")
const JOB_LOCK = ReentrantLock()
const JOB_PROCESSES = Dict{String,Any}()

include(joinpath(PUBLIC_ROOT, "apps", "common", "NativeDialogs.jl"))

include(joinpath(PUBLIC_ROOT, "apps", "airfoil_web", "src", "gpu_lock.jl"))
include(joinpath(PUBLIC_ROOT, "src", "io", "lbm_slice_format.jl"))
include(joinpath(PUBLIC_ROOT, "tools", "postprocess", "PostprocessLauncher.jl"))
include(joinpath(@__DIR__, "aero", "geometry.jl"))
include(joinpath(@__DIR__, "aero", "outputs.jl"))
include(joinpath(@__DIR__, "aero", "reference.jl"))
include(joinpath(@__DIR__, "aero", "pond.jl"))
include(joinpath(@__DIR__, "aero", "q19.jl"))
include(joinpath(@__DIR__, "aero", "case.jl"))
include(joinpath(@__DIR__, "aero", "mesh_helpers.jl"))
include(joinpath(@__DIR__, "aero", "artifacts.jl"))
include(joinpath(@__DIR__, "aero", "jobs.jl"))
include(joinpath(@__DIR__, "aero", "q19_refinement.jl"))
include(joinpath(@__DIR__, "aero", "storage.jl"))
include(joinpath(@__DIR__, "aero", "studies.jl"))
include(joinpath(@__DIR__, "aero", "model_assets.jl"))
include(joinpath(@__DIR__, "aero", "volume_cuts.jl"))
include(joinpath(@__DIR__, "aero", "results_import.jl"))
include(joinpath(@__DIR__, "aero", "diagnostics.jl"))
include(joinpath(@__DIR__, "aero", "view_captures.jl"))

function _response(status::Integer, body)
    HTTP.Response(status, ["Content-Type" => "application/json; charset=utf-8",
                           "Cache-Control" => "no-store"], JSON3.write(body))
end
_bad(message, status::Integer=400) = _response(status, Dict("ok" => false, "error" => string(message)))

function _plain(x)
    x isa AbstractDict && return Dict{String,Any}(String(k) => _plain(v) for (k, v) in pairs(x))
    x isa AbstractVector && return Any[_plain(v) for v in x]
    x
end

function _body(req::HTTP.Request)
    length(req.body) <= 1_000_000 || throw(ArgumentError("JSON request exceeds 1 MB"))
    parsed = try
        JSON3.read(String(copy(req.body)))
    catch
        throw(ArgumentError("Request body is not valid JSON"))
    end
    x = _plain(parsed)
    x isa AbstractDict || throw(ArgumentError("Expected a JSON object"))
    x
end

function _number(x, name; lo=-Inf, hi=Inf)
    x isa Real && !(x isa Bool) && isfinite(x) && lo <= x <= hi ||
        throw(ArgumentError("$name must be a finite number in [$lo, $hi]"))
    Float64(x)
end

function _integer(x, name; lo=1, hi=typemax(Int))
    v = _number(x, name; lo=lo, hi=hi)
    isinteger(v) || throw(ArgumentError("$name must be a whole number"))
    Int(v)
end

function _vec3(x, name; lo=-Inf, hi=Inf)
    x isa AbstractVector && length(x) == 3 || throw(ArgumentError("$name must contain three numbers"))
    ntuple(i -> _number(x[i], "$name[$i]"; lo=lo, hi=hi), 3)
end

function _id(x)
    s = string(x)
    occursin(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", s) ||
        throw(ArgumentError("Invalid resource ID"))
    s
end

"""Handle only /api/aero/* routes from the top-level server."""
function handle(req::HTTP.Request, path::AbstractString)::HTTP.Response
    method = String(req.method)
    route = split(String(path), '?'; limit=2)[1]
    try
        route == "/api/aero/server-info" && method == "GET" && return server_info()
        route == "/api/aero/capabilities" && method == "GET" && return capabilities()
        route == "/api/aero/output-folders" && method == "GET" && return output_folders(req)
        route == "/api/aero/select-folder" && method == "POST" && return _response(200,
            NativeDialogs.choose_folder(req,joinpath(PRIVATE_ROOT,"TEMP_DELETEABLE","native_dialogs")))
        route == "/api/aero/models" && method == "POST" && return upload_model(req)
        route == "/api/aero/mesh" && method == "POST" && return create_mesh(req)
        route == "/api/aero/studies" && method == "POST" && return save_study(req)
        route == "/api/aero/studies/recent" && method == "GET" && return recent_studies()
        route == "/api/aero/studies/open-recent" && method == "POST" && return open_recent_study(req)
        route == "/api/aero/studies/import" && method == "POST" && return import_study(req)
        route == "/api/aero/studies/open-native" && method == "POST" && return open_study_native(req)
        route == "/api/aero/studies/choose-save" && method == "POST" && return choose_study_save_path(req)
        route == "/api/aero/results/import" && method == "POST" && return import_results(req)
        route == "/api/aero/results/upload" && method == "POST" && return upload_results(req)
        if (m=match(r"^/api/aero/diagnostics/([^/]+)$",route))!==nothing && method=="GET"
            return request_diagnostic(_id(m[1]))
        end
        if (m = match(r"^/api/aero/results/([^/]+)$", route)) !== nothing && method == "GET"
            return imported_result_response(_id(m[1]))
        end
        if (m = match(r"^/api/aero/results/([^/]+)/surface$", route)) !== nothing && method == "GET"
            return imported_surface(_id(m[1]))
        end
        if (m = match(r"^/api/aero/results/([^/]+)/files/([A-Za-z0-9_.-]+)$", route)) !== nothing && method == "GET"
            return imported_result_file(_id(m[1]),m[2])
        end
        if (m = match(r"^/api/aero/results/([^/]+)/amr$", route)) !== nothing && method == "GET"
            return imported_amr(_id(m[1]))
        end
        if (m = match(r"^/api/aero/results/([^/]+)/cut$", route)) !== nothing && method == "GET"
            return imported_cut(_id(m[1]),req)
        end
        if (m = match(r"^/api/aero/studies/([^/]+)/file$", route)) !== nothing && method == "GET"
            return study_file(_id(m[1]))
        end
        if (m = match(r"^/api/aero/models/([^/]+)/file$", route)) !== nothing && method == "GET"
            return model_file(_id(m[1]))
        end
        if (m = match(r"^/api/aero/models/([^/]+)/source$", route)) !== nothing
            method == "POST" && return save_model_asset(_id(m[1]),req)
            method == "GET" && return model_asset(_id(m[1]))
        end
        if (m = match(r"^/api/aero/jobs/([^/]+)$", route)) !== nothing && method == "GET"
            return job_response(_id(m[1]))
        end
        if (m = match(r"^/api/aero/jobs/([^/]+)/run$", route)) !== nothing && method == "POST"
            return start_run(_id(m[1]),req)
        end
        if (m = match(r"^/api/aero/jobs/([^/]+)/output-estimate$", route)) !== nothing && method == "POST"
            return output_storage_estimate(_id(m[1]),req)
        end
        if (m = match(r"^/api/aero/jobs/([^/]+)/cancel$", route)) !== nothing && method == "POST"
            return cancel_job(_id(m[1]),req)
        end
        if (m=match(r"^/api/aero/jobs/([^/]+)/view-captures$",route))!==nothing
            method=="POST"&&return request_view_capture(_id(m[1]),req)
            method=="GET"&&return pending_view_capture(_id(m[1]),req)
        end
        if (m=match(r"^/api/aero/jobs/([^/]+)/view-captures/([^/]+)$",route))!==nothing
            method=="GET"&&return view_capture_status(_id(m[1]),_id(m[2]))
            method=="POST"&&return accept_view_capture(_id(m[1]),_id(m[2]),req)
        end
        if (m=match(r"^/api/aero/jobs/([^/]+)/view-captures/([^/]+)/image$",route))!==nothing&&method=="GET"
            return view_capture_image(_id(m[1]),_id(m[2]))
        end
        if (m=match(r"^/api/aero/jobs/([^/]+)/view-captures/([^/]+)/error$",route))!==nothing&&method=="POST"
            return fail_view_capture(_id(m[1]),_id(m[2]),req)
        end
        if (m = match(r"^/api/aero/jobs/([^/]+)/slice/([^/]+)$", route)) !== nothing && method == "GET"
            return slice_json(_id(m[1]), m[2], req)
        end
        if (m = match(r"^/api/aero/jobs/([^/]+)/volume-cut$", route)) !== nothing && method == "GET"
            return volume_cut_json(_id(m[1]), req)
        end
        if (m = match(r"^/api/aero/jobs/([^/]+)/files/(.+)$", route)) !== nothing && method == "GET"
            return job_file(_id(m[1]), m[2])
        end
        _bad("Unknown Aero Lab route", 404)
    catch e
        e isa ArgumentError && return _bad(sprint(showerror, e))
        _request_failure(route,method,e,catch_backtrace())
    end
end

end # module
