# cmesh_mesh_quality_receipt.jl -- the CMeshQualityV1 mesh option at the driver and in the readers (15 Oct 2026).
#
# The generator's quality option (src/cmesh/cmesh_gen.jl, `quality = mesh_quality_v1(...)`, tag CMeshQualityV1;
# memory/project_docs/_records/2026-10-15_mesh_quality/implement/README.md) reaches a run through four things, all here so that the driver
# and every instrument that rebuilds a mesh use one definition:
#   1. the environment knobs (`mesh_quality_env_option`): TOPO_MESH_QUALITY=v1 turns it on (unset: off, the exact legacy
#      call); the overrides TOPO_MESH_Q_PRESET (air; strip and sharp are the topology's own and only confirm it),
#      TOPO_MESH_Q_WAKE, TOPO_MESH_Q_RELAX, TOPO_MESH_Q_OUTER (each a comma list or "off") and TOPO_MESH_Q_STRIP (Q-D:
#      "rho,nb_ref"; unset or "off": no Q-D) are refused without it and outside the generator's guards;
#   2. the receipt line `MESH QUALITY RECEIPT: mesh_version=CMeshQualityV1 params=<canonical string> | <readings>`, printed
#      by the driver after the MESH COORDINATE RECEIPT only when the option is on; its settings half is the canonical
#      parameter string, which `CMeshGen.parse_mesh_quality` turns back into the identical option, so a reader rebuilds the
#      mesh from the log (and the coordinate receipt binds the rebuild);
#   3. the checkpoint metadata `mesh_quality = "CMeshQualityV1/<params>"` and its restart guard (both or neither, equal);
#   4. `mesh_quality_reader_refusal`, the named refusal of a reader that does not rebuild quality meshes.
# The readings half is for people: never passed to the generator.
module CMeshMeshQualityReceipt
using Printf
export MESH_QUALITY_PREFIX, MESH_QUALITY_ENV_KNOBS, mesh_quality_env_option, mesh_quality_receipt_line,
       parse_mesh_quality_receipt, mesh_quality_receipt_from_lines, mesh_quality_restart_guard, mesh_quality_reader_refusal

const MESH_QUALITY_PREFIX = "MESH QUALITY RECEIPT: "
const MESH_QUALITY_ENV_KNOBS = ("TOPO_MESH_Q_PRESET", "TOPO_MESH_Q_WAKE", "TOPO_MESH_Q_RELAX", "TOPO_MESH_Q_OUTER", "TOPO_MESH_Q_STRIP")

_list(knob, raw) = raw == "off" ? :off : begin
    v = tryparse.(Float64, split(raw, ','))
    any(isnothing, v) && throw(ArgumentError("$knob must be a comma list of numbers or off (got \"$raw\")"))
    Tuple(Float64.(v))
end

"""
    mesh_quality_env_option(env, G; strip, n_base, channel) -> nothing | G.MeshQualityV1

The option the environment asks for, built by the generator module `G` (CMeshGen), or nothing when TOPO_MESH_QUALITY is
unset (then no override may be set). The preset follows the topology: :strip with the strip trailing edge, :sharp with a
sharp edge (n_base = 0); TOPO_MESH_Q_PRESET=air selects the air meshes' preset on a sharp edge (strip and sharp may be named
only for their own topology). Refused: any value of TOPO_MESH_QUALITY but v1, an override without it, an A-plus blunt edge
(n_base > 0 without the strip), a run without a channel, and anything the generator's guards refuse.
"""
function mesh_quality_env_option(env, G::Module; strip::Bool, n_base::Integer, channel::Bool)
    raw = get(env, "TOPO_MESH_QUALITY", "")
    set = [k for k in MESH_QUALITY_ENV_KNOBS if haskey(env, k)]
    if isempty(raw)
        isempty(set) || throw(ArgumentError("$(join(set, ", ")) require TOPO_MESH_QUALITY=v1 (the CMeshQualityV1 mesh option; unset: off)"))
        return nothing
    end
    raw == "v1" || throw(ArgumentError("TOPO_MESH_QUALITY must be v1 or unset (got \"$raw\")"))
    channel || throw(ArgumentError("TOPO_MESH_QUALITY=v1 requires TOPO_CHANNEL_H > 0 (the option is the channel branch's)"))
    n_base > 0 && !strip && throw(ArgumentError("TOPO_MESH_QUALITY=v1 does not support the A-plus blunt construction " *
        "(TOPO_MESH_TE_BASE > 0 without TOPO_MESH_TE_TOPOLOGY=strip)"))
    own = strip ? :strip : :sharp
    pr = get(env, "TOPO_MESH_Q_PRESET", "")
    preset = isempty(pr) ? own : Symbol(pr)
    preset in (:strip, :sharp, :air) || throw(ArgumentError("TOPO_MESH_Q_PRESET must be air (or the topology's own strip/sharp; got \"$pr\")"))
    (preset === :air ? own === :sharp : preset === own) || throw(ArgumentError(
        "TOPO_MESH_Q_PRESET=$pr does not fit this mesh's topology ($(own)): air is a sharp edge's preset"))
    kw = (;)
    haskey(env, "TOPO_MESH_Q_WAKE")  && (kw = merge(kw, (wake=_list("TOPO_MESH_Q_WAKE", env["TOPO_MESH_Q_WAKE"]),)))
    haskey(env, "TOPO_MESH_Q_RELAX") && (kw = merge(kw, (relax=_list("TOPO_MESH_Q_RELAX", env["TOPO_MESH_Q_RELAX"]),)))
    haskey(env, "TOPO_MESH_Q_OUTER") && (kw = merge(kw, (outer=_list("TOPO_MESH_Q_OUTER", env["TOPO_MESH_Q_OUTER"]),)))
    if haskey(env, "TOPO_MESH_Q_STRIP") && env["TOPO_MESH_Q_STRIP"] != "off"
        v = split(env["TOPO_MESH_Q_STRIP"], ',')
        length(v) == 2 && tryparse(Float64, v[1]) !== nothing && tryparse(Int, v[2]) !== nothing ||
            throw(ArgumentError("TOPO_MESH_Q_STRIP must be rho,nb_ref (e.g. 1.6,8) or off (got \"$(env["TOPO_MESH_Q_STRIP"])\")"))
        kw = merge(kw, (strip=(parse(Float64, v[1]), parse(Int, v[2])),))
    end
    for (k, v) in pairs(kw)
        k === :strip && continue
        v === :off || length(v) == (k === :relax ? 5 : 2) || throw(ArgumentError(
            "TOPO_MESH_Q_$(uppercase(string(k))) needs $(k === :relax ? "start,ar,ratio,tau_hiar,skew" : k === :wake ? "ds_max,far_growth" : "ar,q") or off"))
    end
    G.mesh_quality_v1(preset; kw...)
end

_tok(x::Integer) = string(x)
_tok(x::Real) = isfinite(x) ? @sprintf("%.7g", x) : string(x)
_tok(x::Nothing) = "none"
_tok(x::AbstractString) = (occursin(r"[\s|=]", x) && throw(ArgumentError("receipt token \"$x\" is not a single token")); x)

"""
    mesh_quality_receipt_line(tag, readings) -> String

The driver's receipt line: `MESH QUALITY RECEIPT: mesh_version=CMeshQualityV1 params=<canonical> | k=v ...`. `tag` is
`CMeshGen.mesh_quality_tag(q)`; `readings` a NamedTuple of numbers, strings or nothing (printed as single tokens, seven
significant digits).
"""
function mesh_quality_receipt_line(tag::AbstractString, readings::NamedTuple)
    m = match(r"^(CMeshQualityV\d+)/(.+)$", tag)
    m === nothing && throw(ArgumentError("not a mesh quality tag: $tag"))
    r = join((string(k, "=", _tok(v)) for (k, v) in pairs(readings)), " ")
    MESH_QUALITY_PREFIX * "mesh_version=" * m.captures[1] * " params=" * m.captures[2] * " | " * r
end

"""
    parse_mesh_quality_receipt(line) -> (tag, version, params, readings::Dict{String,String})

Inverse of `mesh_quality_receipt_line` (the readings as strings). Refuses a malformed line.
"""
function parse_mesh_quality_receipt(line::AbstractString)
    l = lstrip(line)
    startswith(l, MESH_QUALITY_PREFIX) || throw(ArgumentError("not a mesh quality receipt"))
    halves = split(l[length(MESH_QUALITY_PREFIX)+1:end], " | "; limit=2)
    length(halves) == 2 || throw(ArgumentError("malformed mesh quality receipt (no ' | ' separator)"))
    m = match(r"^mesh_version=(\S+) params=(\S+)$", strip(halves[1]))
    m === nothing && throw(ArgumentError("malformed mesh quality receipt settings: $(halves[1])"))
    rd = Dict{String,String}()
    for p in split(strip(halves[2]))
        a = split(p, '='; limit=2)
        length(a) == 2 || throw(ArgumentError("malformed mesh quality reading \"$p\""))
        haskey(rd, a[1]) && throw(ArgumentError("duplicated mesh quality reading $(a[1])"))
        rd[String(a[1])] = String(a[2])
    end
    (tag=string(m.captures[1], "/", m.captures[2]), version=String(m.captures[1]), params=String(m.captures[2]), readings=rd)
end

"The receipt of a run log: nothing when it printed none (no quality option), the parsed receipt for one, an error for more."
function mesh_quality_receipt_from_lines(lines)
    found = filter(l -> startswith(lstrip(l), MESH_QUALITY_PREFIX), lines)
    isempty(found) && return nothing
    length(found) == 1 || throw(ArgumentError("mesh quality receipt is duplicated"))
    parse_mesh_quality_receipt(only(found))
end

"""
    mesh_quality_restart_guard(metadata, active_tag)

Bind a restart (or a reader's checkpoint) to the mesh quality option, as te_base_restart_guard binds the trailing-edge
settings: with no active option (nothing) the checkpoint metadata must not carry `mesh_quality`; with one it must carry
exactly its tag.
"""
function mesh_quality_restart_guard(metadata, active_tag)
    # (finalisation, 15 Oct 2026, quality review F5) the stored value is read from a NamedTuple (the driver's and the
    # instruments' metadata) or from a Dict keyed by :mesh_quality or "mesh_quality" alike; before, a Dict carrying an equal
    # tag was refused as "lacks"
    marked, stored = if metadata === nothing
        (false, nothing)
    elseif metadata isa AbstractDict
        haskey(metadata, :mesh_quality) ? (true, metadata[:mesh_quality]) :
            haskey(metadata, "mesh_quality") ? (true, metadata["mesh_quality"]) : (false, nothing)
    else
        hasproperty(metadata, :mesh_quality) ? (true, getproperty(metadata, :mesh_quality)) : (false, nothing)
    end
    if active_tag === nothing
        marked && throw(ArgumentError("a CMeshQualityV1 checkpoint (its metadata names mesh_quality) requires the matching mesh option (TOPO_MESH_QUALITY)"))
        return nothing
    end
    marked || throw(ArgumentError("checkpoint lacks the mesh quality option of this run ($active_tag)"))
    string(stored) == active_tag || throw(ArgumentError(
        "checkpoint mesh quality option differs from this run's: $stored vs $active_tag"))
    nothing
end

"""
    mesh_quality_reader_refusal(text, reader)

The named refusal of a reader that rebuilds one mesh by its own generator call and does not carry the quality option: a log
(text) with a MESH QUALITY RECEIPT is refused before its geometry is compared, naming the option.
"""
function mesh_quality_reader_refusal(text::AbstractString, reader::AbstractString)
    occursin(MESH_QUALITY_PREFIX, text) && throw(ArgumentError("$reader does not rebuild CMeshQualityV1 meshes: this run's log " *
        "carries a MESH QUALITY RECEIPT (TOPO_MESH_QUALITY=v1); use the wall instruments (tools/cmesh_wall_diagnostics.jl, " *
        "which read the receipt) or teach $reader the option"))
    nothing
end

end # module CMeshMeshQualityReceipt
