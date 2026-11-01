module AirfoilJXBackend

using Dates
using Downloads
using SHA
include(joinpath(@__DIR__, "..", "references", "cavitation_reference_cases.jl"))
using .AirfoilCavitationCases

const WORKSPACE_ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
const PRIVATE_ROOT = normpath(get(ENV, "LUDWIG_PRIVATE_ROOT",
    joinpath(WORKSPACE_ROOT, "..", "02_PRIVATE_PROJECT_DEVELOPMENT")))
const JXFOIL_ROOT = joinpath(
    WORKSPACE_ROOT,
    "third_party", "JXfoil",
)
isfile(joinpath(JXFOIL_ROOT, "src", "JXfoil.jl")) ||
    error("JXfoil source tree was not found at $JXFOIL_ROOT")
JXFOIL_ROOT in LOAD_PATH || pushfirst!(LOAD_PATH, JXFOIL_ROOT)
import JXfoil

export airfoil_catalog, resolve_airfoil, solve_point, solve_sweep

const UIUC_INDEX_URL = "https://m-selig.ae.illinois.edu/ads/coord_database.html"
const UIUC_BASE_URL = "https://m-selig.ae.illinois.edu"
const CACHE_DIR = joinpath(PRIVATE_ROOT, "TEMP_DELETEABLE", "build", "airfoil_web_cache")
const INDEX_CACHE = joinpath(CACHE_DIR, "uiuc_index.html")
const COORD_CACHE = joinpath(CACHE_DIR, "uiuc")
const REFERENCE_GEOMETRY_DIR = normpath(joinpath(@__DIR__, "..", "..", "data", "references"))
const ONERA_GEOMETRY_TERMS = "The ONERA OAT15A geometry remains the intellectual property of ONERA. Refer to it as the ONERA OAT15A, acknowledge ONERA whenever sharing it, and include these conditions when sharing."

"Validate portable source coordinates; no file paths or executable payloads."
function study_coordinates(data)
    data isa AbstractDict || throw(ArgumentError("Invalid embedded airfoil coordinates"))
    x,y = get(data,"x",nothing),get(data,"y",nothing)
    x isa AbstractVector && y isa AbstractVector && 8<=length(x)<=25000 && length(x)==length(y) ||
        throw(ArgumentError("Embedded airfoil needs 8 to 25000 paired coordinates"))
    all(v isa Real && !(v isa Bool) && isfinite(v) && abs(v)<=100 for v in Iterators.flatten((x,y))) ||
        throw(ArgumentError("Airfoil coordinates must be finite chord coordinates"))
    maximum(x)-minimum(x)>0.01 || throw(ArgumentError("Airfoil chord is degenerate"))
    Dict("x"=>Float64.(x),"y"=>Float64.(y))
end

function resolve_airfoil(source::AbstractDict; offline=false)
    kind,id = String(source["kind"]),String(source["id"])
    if kind == "uiuc" && haskey(source,"coordinates")
        _safe_uiuc_id(id)
        coordinates = study_coordinates(source["coordinates"])
        foil = JXfoil.Airfoil(id,coordinates["x"],coordinates["y"])
        return (airfoil=foil,metadata=Dict{String,Any}("source"=>kind,"id"=>id,"name"=>id,
            "source_url"=>UIUC_BASE_URL*"/ads/coord/"*id,"embedded"=>true,
            "x"=>copy(foil.x),"y"=>copy(foil.y)))
    end
    resolve_airfoil(kind,id;offline=offline)
end
const LOCAL_FALLBACK = Dict(
    "e387.dat" => joinpath(
        PRIVATE_ROOT, "Xfoil_code_2026_10_18_BASELINE", "JXfoil_REPOSITORY", "runs", "e387.dat"
    ),
)

_finite(x::Real) = isfinite(x) ? Float64(x) : nothing
_finite(xs::AbstractVector{<:Real}) = [_finite(x) for x in xs]

function _download(url::AbstractString, destination::AbstractString)
    mkpath(dirname(destination))
    temporary = tempname(dirname(destination))
    try
        Downloads.download(url, temporary; timeout = 20)
        filesize(temporary) > 0 || throw(ArgumentError("empty download from $url"))
        mv(temporary, destination; force = true)
    finally
        isfile(temporary) && rm(temporary; force = true)
    end
    return destination
end

function _safe_uiuc_id(id::AbstractString)
    value = strip(String(id))
    occursin(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.dat$"i, value) ||
        throw(ArgumentError("UIUC id must be a .dat filename from the catalog"))
    return lowercase(value)
end

function _uiuc_url(href::AbstractString)
    s = strip(String(href))
    url = if startswith(s, "/")
        UIUC_BASE_URL * s
    elseif startswith(s, "https://")
        s
    elseif startswith(s, "./")
        UIUC_BASE_URL * "/ads/" * s[3:end]
    else
        UIUC_BASE_URL * "/ads/" * s
    end
    startswith(url, UIUC_BASE_URL * "/ads/") ||
        throw(ArgumentError("UIUC catalog supplied a URL outside its data site"))
    return url
end

function _html_text(s::AbstractString)
    result = replace(String(s), r"<[^>]*>" => " ")
    result = replace(
        result, "&nbsp;" => " ", "&amp;" => "&", "&quot;" => "\"",
        "&#39;" => "'", "&lt;" => "<", "&gt;" => ">",
    )
    return strip(replace(result, r"\s+" => " "))
end

function _parse_catalog(html::AbstractString)
    items = Dict{String,Any}[]
    seen = Set{String}()
    for line in eachsplit(html, '\n')
        m = match(r"<a[^>]+href=\"([^\"]+\.dat)\"[^>]*>([^<]+)</a>(.*)$"i, line)
        isnothing(m) && continue
        id = try
            _safe_uiuc_id(_html_text(m.captures[2]))
        catch
            continue
        end
        id in seen && continue
        url = try
            _uiuc_url(m.captures[1])
        catch
            continue
        end
        push!(seen, id)
        description = _html_text(m.captures[3])
        startswith(description, "-") && (description = strip(description[2:end]))
        push!(items, Dict{String,Any}(
            "id" => id,
            "name" => replace(id, r"\.dat$" => ""),
            "description" => description,
            "dat_url" => url,
            "source_url" => UIUC_INDEX_URL,
            "cached" => isfile(joinpath(COORD_CACHE, id)) ||
                (haskey(LOCAL_FALLBACK, id) && isfile(LOCAL_FALLBACK[id])),
        ))
    end
    sort!(items; by = item -> item["id"])
    return items
end

function _fallback_catalog()
    return [Dict{String,Any}(
        "id" => id,
        "name" => replace(id, r"\.dat$" => ""),
        "description" => "Local copy of a UIUC airfoil",
        "dat_url" => UIUC_BASE_URL * "/ads/coord/" * id,
        "source_url" => UIUC_INDEX_URL,
        "cached" => true,
    ) for (id, path) in LOCAL_FALLBACK if isfile(path)]
end

"""
    airfoil_catalog(; refresh=false, offline=false, query="", limit=2000)

Return UIUC catalog entries. A previously fetched index remains usable offline; a
small local fallback is available before the first fetch. The UIUC site is only
contacted when an uncached or explicitly refreshed catalog is requested.
"""
function airfoil_catalog(;
    refresh::Bool = false,
    offline::Bool = false,
    query::AbstractString = "",
    limit::Integer = 2000,
)
    limit >= 0 || throw(ArgumentError("catalog limit must be nonnegative"))
    fresh_cache = isfile(INDEX_CACHE) &&
        (time() - stat(INDEX_CACHE).mtime <= 24 * 60 * 60)
    if !offline && (refresh || !fresh_cache)
        try
            _download(UIUC_INDEX_URL, INDEX_CACHE)
        catch err
            isfile(INDEX_CACHE) || @warn "UIUC catalog unavailable; using local fallback" exception = err
        end
    end
    items = isfile(INDEX_CACHE) ? _parse_catalog(read(INDEX_CACHE, String)) : _fallback_catalog()
    isempty(items) && (items = _fallback_catalog())
    needle = lowercase(strip(String(query)))
    if !isempty(needle)
        filter!(item -> occursin(needle, lowercase(item["id"] * " " * item["description"])), items)
    end
    return items[1:min(Int(limit), length(items))]
end

function _naca_code(source_id::AbstractString)
    value = uppercase(strip(String(source_id)))
    startswith(value, "NACA") && (value = strip(value[5:end]))
    occursin(r"^\d{4}$", value) ||
        throw(ArgumentError("JXfoil's built-in NACA generator supports four-digit codes only"))
    return parse(Int, value)
end

"""
    resolve_airfoil(source_kind, source_id; offline=false)

Return `(airfoil, metadata)`. `source_kind` is `naca4`, `uiuc`, `rae2822`, or
`oat15a`, or `cavitation_reference`. UIUC IDs are catalog filenames, never arbitrary paths. Cavitation
case IDs resolve only checksum-bound local benchmark geometries. The reference
geometries come from NASA NPARC and the AIAA DPW/ONERA published coordinate
tables (see validation/data/README.md). Coordinates are retained for CFD and
for plotting the geometry.
"""
function resolve_airfoil(
    source_kind::AbstractString,
    source_id::AbstractString;
    offline::Bool = false,
)
    kind = lowercase(strip(String(source_kind)))
    if kind == "naca4" || kind == "naca"
        code = _naca_code(source_id)
        foil = JXfoil.naca4_airfoil(code; n = 161)
        id = lpad(string(code), 4, '0')
        metadata = Dict{String,Any}(
            "source" => "naca4", "id" => id, "name" => foil.name,
            "source_url" => nothing, "dat_url" => nothing,
            "x" => copy(foil.x), "y" => copy(foil.y),
        )
        return (airfoil = foil, metadata = metadata)
    elseif kind == "cavitation_reference"
        preset = AirfoilCavitationCases.preset(source_id)
        path = AirfoilCavitationCases.geometry_path(source_id)
        coordinates = AirfoilCavitationCases.geometry_coordinates(source_id)
        foil = JXfoil.Airfoil(String(preset["geometry_name"]), coordinates.x, coordinates.y)
        metadata = Dict{String,Any}(
            "source"=>kind, "id"=>preset["id"], "name"=>foil.name,
            "description"=>preset["description"], "reference_case"=>preset,
            "source_geometry_path"=>relpath(path, WORKSPACE_ROOT),
            "source_geometry_sha256"=>bytes2hex(sha256(read(path))),
            "source_url"=>get(preset, "source_url", nothing),
            "x"=>copy(foil.x), "y"=>copy(foil.y))
        return (airfoil=foil, metadata=metadata)
    elseif kind == "rae2822"
        lowercase(strip(String(source_id))) == "rae2822" ||
            throw(ArgumentError("RAE 2822 id must be rae2822"))
        foil = JXfoil.load_airfoil(joinpath(REFERENCE_GEOMETRY_DIR, "rae2822_agard.dat"))
        metadata = Dict{String,Any}(
            "source" => "rae2822", "id" => "rae2822", "name" => "RAE 2822",
            "source_url" => "https://www.grc.nasa.gov/www/wind/valid/raetaf/raetaf.html",
            "dat_url" => "https://www.grc.nasa.gov/www/wind/valid/raetaf/geom.txt",
            "description" => "AGARD AR-138 Table 6.1 ordinates via the NASA NPARC archive",
            "x" => copy(foil.x), "y" => copy(foil.y),
        )
        return (airfoil = foil, metadata = metadata)
    elseif kind == "oat15a"
        lowercase(strip(String(source_id))) == "oat15a" ||
            throw(ArgumentError("ONERA OAT15A id must be oat15a"))
        loaded = JXfoil.load_airfoil(joinpath(REFERENCE_GEOMETRY_DIR, "onera_oat15a.dat"))
        foil = JXfoil.Airfoil("ONERA OAT15A", loaded.x, loaded.y)
        metadata = Dict{String,Any}(
            "source" => "oat15a", "id" => "oat15a", "name" => "ONERA OAT15A",
            "source_url" => "https://www.aiaa-dpw.org/geometry.html#oat",
            "dat_url" => "https://media.githubusercontent.com/media/dpw-committee/aiaa-dpw/main/Geometry/ONERA-OAT15A-090524.zip",
            "description" => "ONERA 230 mm chord geometry released 2024-09-05 for AIAA DPW-8",
            "attribution" => ONERA_GEOMETRY_TERMS,
            "x" => copy(foil.x), "y" => copy(foil.y),
        )
        return (airfoil = foil, metadata = metadata)
    elseif kind == "uiuc"
        id = _safe_uiuc_id(source_id)
        entries = airfoil_catalog(; offline = offline)
        match_index = findfirst(item -> item["id"] == id, entries)
        entry = isnothing(match_index) ? nothing : entries[match_index]
        local_path = joinpath(COORD_CACHE, id)
        if !isfile(local_path)
            fallback = get(LOCAL_FALLBACK, id, "")
            if isfile(fallback)
                local_path = fallback
            else
                offline && throw(ArgumentError("UIUC airfoil $id is not cached"))
                url = isnothing(entry) ? UIUC_BASE_URL * "/ads/coord/" * id : entry["dat_url"]
                _download(url, local_path)
            end
        end
        foil = JXfoil.load_airfoil(local_path)
        metadata = Dict{String,Any}(
            "source" => "uiuc", "id" => id, "name" => foil.name,
            "source_url" => UIUC_INDEX_URL,
            "dat_url" => isnothing(entry) ? UIUC_BASE_URL * "/ads/coord/" * id : entry["dat_url"],
            "description" => isnothing(entry) ? "" : entry["description"],
            "x" => copy(foil.x), "y" => copy(foil.y),
        )
        return (airfoil = foil, metadata = metadata)
    end
    throw(ArgumentError("unsupported airfoil source '$source_kind'"))
end

function _provenance(p)
    return Dict{String,Any}(
        "backend" => string(p.backend), "selection" => string(p.selection),
        "converged" => p.converged, "status" => string(p.status),
        "failure_message" => p.failure_message, "iterations" => p.iterations,
        "selected_iteration" => p.selected_iteration,
        "selected_residual" => _finite(p.selected_residual),
    )
end

function _surface_payload(surface, transition_xc)
    n = minimum((length(surface.xc), length(surface.ue), length(surface.primary),
        length(surface.secondary), length(surface.mode)))
    return Dict{String,Any}(
        "x" => _finite(surface.xc[1:n]),
        "ue" => _finite(surface.ue[1:n]),
        "dstar" => [_finite(p.d) for p in surface.primary[1:n]],
        "theta" => [_finite(p.t) for p in surface.primary[1:n]],
        "cf" => [_finite(s.cf) for s in surface.secondary[1:n]],
        "h" => [_finite(s.h) for s in surface.secondary[1:n]],
        "n_factor" => [_finite(p.ampl) for p in surface.primary[1:n]],
        "mode" => [string(m) for m in surface.mode[1:n]],
        "transition_x" => _finite(transition_xc),
    )
end

function _surface_cp(result)
    inv = result.inviscid
    ue = try
        JXfoil._panel_ue_from_surfaces(inv, result.upper, result.lower)
    catch
        nothing
    end
    n = min(length(inv.xcp), length(inv.ycp), length(inv.cp),
        isnothing(ue) ? length(inv.xcp) : length(ue))
    n == 0 && return Dict("upper" => Any[], "lower" => Any[])
    ile = argmin(view(inv.xcp, 1:n))
    a = [Dict{String,Any}(
        "x" => _finite(inv.xcp[i]), "y" => _finite(inv.ycp[i]),
        "cp" => isnothing(ue) ? nothing : _finite(1.0 - ue[i]^2),
        "cp_inviscid" => _finite(inv.cp[i]),
    ) for i in 1:ile]
    b = [Dict{String,Any}(
        "x" => _finite(inv.xcp[i]), "y" => _finite(inv.ycp[i]),
        "cp" => isnothing(ue) ? nothing : _finite(1.0 - ue[i]^2),
        "cp_inviscid" => _finite(inv.cp[i]),
    ) for i in (ile + 1):n]
    mean_a = sum(Float64(item["y"]) for item in a if item["y"] !== nothing) / max(length(a), 1)
    mean_b = sum(Float64(item["y"]) for item in b if item["y"] !== nothing) / max(length(b), 1)
    upper, lower = mean_a >= mean_b ? (a, b) : (b, a)
    sort!(upper; by = item -> something(item["x"], Inf))
    sort!(lower; by = item -> something(item["x"], Inf))
    return Dict("upper" => upper, "lower" => lower)
end

function _unsupported(alpha_deg, mach)
    return Dict{String,Any}(
        "backend" => "jxfoil", "alpha_deg" => Float64(alpha_deg),
        "status" => "unsupported", "converged" => false,
        "message" => "JXfoil's public coupled solver is incompressible; Mach $mach requires the CFD backend.",
        "cl" => nothing, "cd" => nothing, "cm" => nothing, "cm_c4" => nothing,
        "cp" => Dict("upper" => Any[], "lower" => Any[]),
        "fields" => Dict{String,Any}(), "provenance" => Dict{String,Any}(),
    )
end

function _error_result(alpha_deg, err)
    return Dict{String,Any}(
        "backend" => "jxfoil", "alpha_deg" => Float64(alpha_deg),
        "status" => "error", "converged" => false,
        "message" => sprint(showerror, err),
        "cl" => nothing, "cd" => nothing, "cm" => nothing, "cm_c4" => nothing,
        "cp" => Dict("upper" => Any[], "lower" => Any[]),
        "fields" => Dict{String,Any}(), "provenance" => Dict{String,Any}(),
    )
end

function _point_payload(result; include_profiles::Bool = true)
    p = result.provenance
    payload = Dict{String,Any}(
        "backend" => "jxfoil", "alpha_deg" => _finite(result.alpha_deg),
        "reynolds" => _finite(result.reynolds), "ncrit" => _finite(result.ncrit),
        "status" => p.converged ? "ok" : "nonconverged",
        "converged" => p.converged,
        "message" => p.converged ? "JXfoil coupled solve converged." :
            (isempty(p.failure_message) ? "JXfoil did not converge." : p.failure_message),
        "cl" => _finite(result.cl), "cd" => _finite(result.cd),
        "cm" => _finite(result.cm_c4), "cm_c4" => _finite(result.cm_c4),
        "cdp" => nothing, "top_xtr" => _finite(result.top_xtr),
        "bot_xtr" => _finite(result.bot_xtr),
        "coupling_iterations" => result.coupling_iterations,
        "coupling_residual" => _finite(result.coupling_residual),
        "provenance" => _provenance(p),
        "fields" => Dict{String,Any}(),
        "available_fields" => String[],
    )
    if include_profiles
        payload["cp"] = _surface_cp(result)
        payload["boundary_layer"] = Dict(
            "upper" => _surface_payload(result.upper, result.top_xtr),
            "lower" => _surface_payload(result.lower, result.bot_xtr),
        )
    else
        payload["cp"] = Dict("upper" => Any[], "lower" => Any[])
    end
    return payload
end

"""Run one incompressible JXfoil operating point and return wire-ready data."""
function solve_point(
    foil::JXfoil.Airfoil,
    alpha_deg::Real;
    mach::Real = 0.0,
    reynolds::Real = 1.0e5,
    ncrit::Real = 9.0,
    repanel::Bool = true,
    viscal_max_iter::Integer = 80,
    include_profiles::Bool = true,
)
    isfinite(alpha_deg) || throw(ArgumentError("alpha must be finite"))
    isfinite(mach) && mach >= 0 || throw(ArgumentError("Mach must be finite and nonnegative"))
    isfinite(reynolds) && reynolds > 0 || throw(ArgumentError("Reynolds number must be positive"))
    isfinite(ncrit) && ncrit > 0 || throw(ArgumentError("Ncrit must be positive"))
    mach > 1.0e-6 && return _unsupported(alpha_deg, mach)
    try
        result = JXfoil.solve_coupled_case(
            foil, alpha_deg; reynolds = reynolds, ncrit = ncrit,
            repanel = repanel, viscal_max_iter = viscal_max_iter,
        )
        return _point_payload(result; include_profiles = include_profiles)
    catch err
        return _error_result(alpha_deg, err)
    end
end

"""Run a continuation polar. Individual failures are represented as point statuses."""
function solve_sweep(
    foil::JXfoil.Airfoil,
    alphas::AbstractVector{<:Real};
    mach::Real = 0.0,
    reynolds::Real = 1.0e5,
    ncrit::Real = 9.0,
    repanel::Bool = true,
    viscal_max_iter::Integer = 80,
    include_profiles::Bool = true,
)
    alpha_list = sort!(unique(Float64.(alphas)))
    length(alpha_list) <= 200 || throw(ArgumentError("alpha sweep exceeds 200 points"))
    all(isfinite, alpha_list) || throw(ArgumentError("alpha values must be finite"))
    isfinite(mach) && mach >= 0 || throw(ArgumentError("Mach must be finite and nonnegative"))
    isfinite(reynolds) && reynolds > 0 || throw(ArgumentError("Reynolds number must be positive"))
    isfinite(ncrit) && ncrit > 0 || throw(ArgumentError("Ncrit must be positive"))
    if mach > 1.0e-6
        return Dict{String,Any}(
            "backend" => "jxfoil", "status" => "unsupported",
            "points" => [_unsupported(alpha, mach) for alpha in alpha_list],
        )
    end
    points = try
        results = JXfoil.solve_coupled_polar(
            foil, alpha_list; reynolds = reynolds, ncrit = ncrit,
            repanel = repanel, viscal_max_iter = viscal_max_iter,
        )
        [_point_payload(result; include_profiles = include_profiles) for result in results]
    catch
        # A single geometry or numerical exception must not erase the whole polar.
        [solve_point(
            foil, alpha; mach = mach, reynolds = reynolds, ncrit = ncrit,
            repanel = repanel, viscal_max_iter = viscal_max_iter,
            include_profiles = include_profiles,
        ) for alpha in alpha_list]
    end
    return Dict{String,Any}(
        "backend" => "jxfoil",
        "status" => all(p -> p["status"] == "ok", points) ? "ok" : "partial",
        "points" => points,
    )
end

end # module
