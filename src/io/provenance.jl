# ==============================================================================
# Provenance metadata for reproducibility
# ==============================================================================
# Captures all information needed by an external reader to reproduce a result:
#   - Solver version + git commit hash + dirty flag
#   - Build timestamp + Julia + CUDA versions
#   - Resolved configuration (every parameter as actually used at runtime)
#   - Host info (hostname, GPU model, GPU count, CUDA driver)
#   - Wall-clock + per-stage timings
#
# Two output products:
#   1. provenance.json — machine-readable, lives next to convergence.csv
#   2. provenance.txt  — human-readable banner appended to the run log
# ==============================================================================

using Dates
using Printf

# Solver semantic version. Bump on any release-worthy change. The git commit
# hash captures fine-grained provenance between version bumps.
const LUDWIG_VERSION = "1.10-dev"

"""
    git_provenance() -> NamedTuple

Capture git status as `(commit, branch, dirty, status_lines)`. If the project
isn't a git repo, returns sentinel values so the run still proceeds.
"""
function git_provenance()
    project_root = dirname(dirname(@__DIR__))
    git_marker = joinpath(project_root, ".git")
    if !(isdir(git_marker) || isfile(git_marker))
        return (
            commit_full = "non-git",
            commit = "non-git",
            branch = "non-git",
            dirty = false,
            n_dirty_files = 0,
        )
    end
    try
        commit_full = try
            strip(read(`git -C $project_root rev-parse HEAD`, String))
        catch
            "non-git"
        end
        # String slicing returns a SubString on current Julia releases. Keep the public provenance
        # schema concrete and stable so JSON/text consumers never see a version-dependent view type.
        commit_short = length(commit_full) >= 8 ? String(commit_full[1:8]) : String(commit_full)
        branch = try
            strip(read(`git -C $project_root rev-parse --abbrev-ref HEAD`, String))
        catch
            "non-git"
        end
        # `--porcelain` returns nothing on a clean repo, lines on a dirty one.
        status_raw = try
            strip(read(`git -C $project_root status --porcelain`, String))
        catch
            ""
        end
        dirty = !isempty(status_raw)
        n_dirty_files = dirty ? count(==('\n'), status_raw) + 1 : 0
        return (
            commit_full = commit_full,
            commit = commit_short,
            branch = branch,
            dirty = dirty,
            n_dirty_files = n_dirty_files,
        )
    catch
        return (
            commit_full = "unknown",
            commit = "unknown",
            branch = "unknown",
            dirty = false,
            n_dirty_files = 0,
        )
    end
end

"""
    host_provenance() -> NamedTuple

Capture host and accelerator info: hostname, OS, Julia version, CUDA version,
GPU model and count.
"""
function host_provenance()
    hostname = try
        Sys.gethostname()
    catch
        "unknown"
    end
    julia_ver = string(VERSION)
    os_str = "$(Sys.KERNEL) $(Sys.MACHINE)"
    cuda_ver = "n/a"
    gpu_models = String[]
    n_gpus = 0
    try
        if isdefined(Main, :CUDA) && CUDA.functional()
            cuda_ver = string(CUDA.version())
            n_gpus = length(CUDA.devices())
            for dev in CUDA.devices()
                push!(gpu_models, CUDA.name(dev))
            end
        end
    catch
        # CUDA introspection failed; leave defaults.
    end
    return (
        hostname = hostname,
        os = os_str,
        julia_version = julia_ver,
        cuda_version = cuda_ver,
        n_gpus = n_gpus,
        gpu_models = gpu_models,
    )
end

"""
    write_provenance(output_dir::String, case_name::String, config_dict::Any;
                     extra::Dict = Dict()) -> Nothing

Write `provenance.json` and `provenance.txt` to `output_dir`. `config_dict`
is the resolved config (the parsed YAML) — it is captured verbatim so a
future reader has every parameter that was actually used.
"""
function write_provenance(output_dir::String, case_name::String, config_dict;
                          extra::Dict = Dict{String,Any}())
    git = git_provenance()
    host = host_provenance()
    now_dt = Dates.now()

    # JSON-compatible dict. Avoid embedding non-serialisable Julia objects.
    payload = Dict{String,Any}(
        "ludwig_version" => LUDWIG_VERSION,
        "timestamp_utc" => string(Dates.now(Dates.UTC)),
        "timestamp_local" => string(now_dt),
        "case_name" => case_name,
        "git" => Dict(
            "commit" => git.commit_full,
            "commit_short" => git.commit,
            "branch" => git.branch,
            "dirty" => git.dirty,
            "n_dirty_files" => git.n_dirty_files,
        ),
        "host" => Dict(
            "hostname" => host.hostname,
            "os" => host.os,
            "julia_version" => host.julia_version,
            "cuda_version" => host.cuda_version,
            "n_gpus" => host.n_gpus,
            "gpu_models" => host.gpu_models,
        ),
        "config" => _make_jsonable(config_dict),
        "extra" => _make_jsonable(extra),
    )

    isdir(output_dir) || mkpath(output_dir)
    open(joinpath(output_dir, "provenance.json"), "w") do io
        _write_json(io, payload, 0)
    end

    open(joinpath(output_dir, "provenance.txt"), "w") do io
        println(io, "═"^72)
        println(io, "  LUDWIG run provenance")
        println(io, "═"^72)
        @printf(io, "  Solver version : %s\n", LUDWIG_VERSION)
        @printf(io, "  Timestamp      : %s (local) | %s (UTC)\n",
                now_dt, Dates.now(Dates.UTC))
        @printf(io, "  Case           : %s\n", case_name)
        @printf(io, "  Output dir     : %s\n", output_dir)
        println(io)
        println(io, "  Git")
        @printf(io, "    branch       : %s\n", git.branch)
        @printf(io, "    commit       : %s\n", git.commit_full)
        if git.dirty
            @printf(io, "    DIRTY        : %d uncommitted change(s)\n", git.n_dirty_files)
        else
            println(io, "    clean        : yes")
        end
        println(io)
        println(io, "  Host")
        @printf(io, "    hostname     : %s\n", host.hostname)
        @printf(io, "    OS           : %s\n", host.os)
        @printf(io, "    Julia        : %s\n", host.julia_version)
        @printf(io, "    CUDA         : %s\n", host.cuda_version)
        @printf(io, "    GPUs         : %d\n", host.n_gpus)
        for (i, gm) in enumerate(host.gpu_models)
            @printf(io, "      [%d]        : %s\n", i, gm)
        end
        println(io)
        println(io, "  Config: see provenance.json for the full resolved config.")
        if !isempty(extra)
            println(io)
            println(io, "  Extra")
            for (k, v) in extra
                @printf(io, "    %-12s : %s\n", string(k), string(v))
            end
        end
        println(io, "═"^72)
    end
    return nothing
end

# ──────────────────────────────────────────────────────────────────────
# Minimal JSON writer (avoids adding a dependency just for provenance).
# Handles Dict, AbstractVector, String, Real, Bool, Nothing.
# ──────────────────────────────────────────────────────────────────────

_make_jsonable(x::Dict) = Dict(string(k) => _make_jsonable(v) for (k, v) in x)
_make_jsonable(x::AbstractVector) = [_make_jsonable(v) for v in x]
_make_jsonable(x::AbstractString) = String(x)
_make_jsonable(x::Symbol) = string(x)
_make_jsonable(x::Bool) = x
_make_jsonable(x::Real) = x
_make_jsonable(::Nothing) = nothing
_make_jsonable(x) = string(x)  # fallback

function _json_escape(s::AbstractString)
    out = IOBuffer()
    for c in s
        if c == '"'
            print(out, "\\\"")
        elseif c == '\\'
            print(out, "\\\\")
        elseif c == '\n'
            print(out, "\\n")
        elseif c == '\r'
            print(out, "\\r")
        elseif c == '\t'
            print(out, "\\t")
        elseif UInt32(c) < 0x20
            @printf(out, "\\u%04x", UInt32(c))
        else
            print(out, c)
        end
    end
    return String(take!(out))
end

function _write_json(io::IO, x::Dict, indent::Int)
    pad = "  "^indent
    pad_inner = "  "^(indent + 1)
    if isempty(x)
        print(io, "{}")
        return
    end
    println(io, "{")
    keys_list = collect(keys(x))
    for (i, k) in enumerate(keys_list)
        @printf(io, "%s\"%s\": ", pad_inner, _json_escape(string(k)))
        _write_json(io, x[k], indent + 1)
        println(io, i == length(keys_list) ? "" : ",")
    end
    print(io, pad, "}")
end

function _write_json(io::IO, x::AbstractVector, indent::Int)
    if isempty(x)
        print(io, "[]")
        return
    end
    pad = "  "^indent
    pad_inner = "  "^(indent + 1)
    println(io, "[")
    for (i, v) in enumerate(x)
        print(io, pad_inner)
        _write_json(io, v, indent + 1)
        println(io, i == length(x) ? "" : ",")
    end
    print(io, pad, "]")
end

_write_json(io::IO, x::AbstractString, _::Int) = print(io, "\"", _json_escape(x), "\"")
_write_json(io::IO, x::Bool, _::Int) = print(io, x ? "true" : "false")
_write_json(io::IO, x::Real, _::Int) = isfinite(x) ? print(io, x) : print(io, "null")
_write_json(io::IO, ::Nothing, _::Int) = print(io, "null")
_write_json(io::IO, x, indent::Int) = _write_json(io, string(x), indent)

"""
    print_provenance_banner(io::IO = stdout)

Print a 1-line provenance banner suitable for the head of a run log.
"""
function print_provenance_banner(io::IO = stdout)
    git = git_provenance()
    host = host_provenance()
    @printf(io, "[Provenance] LUDWIG %s | git %s%s | %s | %d GPU%s\n",
            LUDWIG_VERSION,
            git.commit,
            git.dirty ? "*" : "",
            host.hostname,
            host.n_gpus,
            host.n_gpus == 1 ? "" : "s")
end
