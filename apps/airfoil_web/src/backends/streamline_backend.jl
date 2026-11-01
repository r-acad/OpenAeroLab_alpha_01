# =====================================================================================================================
# streamline_backend.jl -- streamline-aligned CFD meshes in the app (DEVELOPMENT, NOT VALIDATED; 24 Oct 2026).
# Included inside CFDBackend. Design: STREAMLINE_ALIGNMENT_DESIGN.md.
#
# Opt-in request keys (absent: every request, input file and mesh is exactly as before):
#   cfd_streamline_align      true to align the CFD mesh with JXfoil's wake path and stagnation streamline at the point's
#                             own angle of attack;
#   cfd_streamline_mode       "viscous" (default: JXfoil's coupled solution through its displacement body) or "inviscid";
#   cfd_streamline_fallback   true: a refused viscous path (not converged; trailing-edge separation over 0.10 c) falls back
#                             to the inviscid path, labelled; false (default): the point is refused with the reason;
#   cfd_streamline_admission  "strict" (default: every gate that passed still passes and no failing gate gets worse) or
#                             "demo" (convex cells, conforming C/H interfaces and an unchanged wall first spacing; every
#                             gate's change is reported, none refuses). Demo-admitted meshes are labelled so.
# The JXfoil step runs in the app server (STREAMLINE_PROVIDER, set by server.jl: CFDBackend's solver workers never load
# JXfoil); it writes a hashed path file, and the solver worker applies mesh_streamline_morph.jl to its own mesh nodes:
# the low-speed C+H worker (PBCav blocks, with or without a Gurney plate) and the compressible C+H worker (ch_streamline.jl,
# finite trailing edge, no plate)
# (Create mesh and Run run the same worker code on the same file, so they build the same nodes; the mesh SHA-256 shows it).
# =====================================================================================================================
const STREAMLINE_MORPH_MODULE = normpath(joinpath(@__DIR__, "..", "mesh", "mesh_streamline_morph.jl"))
const STREAMLINE_QUALITY_MODULE = normpath(joinpath(@__DIR__, "..", "..", "..", "..", "src", "cmesh", "cmesh_quality_metrics.jl"))
"The server's JXfoil path step: (source, flow, alpha, dir; mode, fallback) -> Dict with path, sha256, source, source_label."
const STREAMLINE_PROVIDER = Ref{Any}(nothing)
const STREAMLINE_DEV_LABEL = "Streamline-aligned mesh (JXfoil, development, not validated)"

streamline_enabled(flow::AbstractDict) = _boolean(flow, "cfd_streamline_align", false)

function streamline_settings(flow::AbstractDict)
    mode = String(_get(flow, "cfd_streamline_mode", "viscous"))
    mode in ("viscous", "inviscid") || throw(ArgumentError("cfd_streamline_mode must be viscous or inviscid"))
    adm = String(_get(flow, "cfd_streamline_admission", "strict"))
    adm in ("strict", "demo") || throw(ArgumentError("cfd_streamline_admission must be strict or demo"))
    (; mode, admission=adm, fallback=_boolean(flow, "cfd_streamline_fallback", false))
end

const STREAMLINE_CH_ONLY = "Streamline alignment runs on the C+H mesh of a finite trailing edge (TE topology Auto); this section's CFD mesh is not C+H (a sharp trailing edge, or Legacy A-plus). Untick the alignment, or use a section with a finite trailing edge."
const STREAMLINE_NO_COMPRESSIBLE_GURNEY = "Streamline alignment with a Gurney flap is offered on the low-speed CFD only: the Gurney flap is a baffle, which only the low-speed solver has (the compressible C+H solver has no interior walls)."

"Validate the keys of a request (server side, before any work); refuses the alignment where no backend supports it."
function streamline_check_request(flow::AbstractDict, backends)
    haskey(flow, "cfd_streamline_align") || return nothing
    flow["cfd_streamline_align"] isa Bool || throw(ArgumentError("cfd_streamline_align must be boolean"))
    streamline_enabled(flow) || return nothing
    s = streamline_settings(flow)
    ls = any(==("lowspeed"), backends); tr = any(==("transonic"), backends)
    any(==("cavitation"), backends) && throw(ArgumentError(
        "Streamline alignment is not offered for the Species / cavitation CFD (its driver builds its own mesh). Untick it, or choose the low-speed or compressible CFD."))
    ls || tr || throw(ArgumentError(
        "Streamline-aligned meshes are available for the low-speed CFD (C+H mesh: a Gurney flap, or the native-base control) and for the compressible CFD on a finite trailing edge (C+H); choose one of them."))
    ls && !_lowspeed_ch_selected(flow) && throw(ArgumentError(
        "Streamline alignment runs on the low-speed CFD's C+H mesh: add a Gurney flap (or the native-base C+H control, lowspeed_topology = ch_native)."))
    if tr
        gurney_enabled(flow) && throw(ArgumentError(STREAMLINE_NO_COMPRESSIBLE_GURNEY))
        String(_get(flow, "cfd_te_topology", "aplus")) == "auto" || throw(ArgumentError(
            "Streamline alignment on the compressible CFD runs on the C+H mesh: select TE topology Auto (a finite trailing edge)."))
    end
    s
end

"""The worker's `streamline` table: the server computes and hashes the JXfoil paths at `alpha` for the CLEAN source section.
With a Gurney flap (a baffle since 28 Oct 2026: the mesh keeps the native base, so the wake path needs no move to a taller base)
the notes say that JXfoil cannot model the plate and that the baffle is cut AFTER the alignment, on the aligned mesh. `nothing`
when the alignment is off."""
function streamline_worker_cfg(source::AbstractDict, flow::AbstractDict, alpha::Real, dir::AbstractString; gurney=nothing)
    streamline_enabled(flow) || return nothing
    s = streamline_settings(flow)
    STREAMLINE_PROVIDER[] === nothing && throw(ArgumentError(
        "Streamline alignment needs JXfoil in the app server; this process has no streamline provider. Restart run.cmd."))
    rec = STREAMLINE_PROVIDER[](source, flow, Float64(alpha), dir; mode=s.mode, fallback=s.fallback)
    params = Dict{String,Any}()
    note = @sprintf("Mesh aligned with JXfoil's %s at alpha %.4g deg, Re %.3g (path SHA-256 %s).", rec["source_label"], alpha,
                    Float64(_get(flow, "reynolds", 3e6)), String(rec["sha256"])[1:12])
    mach = Float64(_get(flow, "mach", 0.0))
    mach > 0 && (note *= @sprintf(" JXfoil's paths are incompressible (computed at Mach 0); the flow runs at Mach %.3g.", mach))
    if gurney !== nothing
        note *= " JXfoil cannot model the Gurney plate: the paths are the CLEAN section's (no plate), so they do not follow the wake the plate displaces (a lower plate turns it downward). The plate is a baffle cut in the mesh AFTER the alignment, on the aligned nodes; it never moves the wall row (the aligned wake cut is on it) and acts only within about 0.05 c of the trailing-edge corner."
    end
    Dict{String,Any}("paths"=>replace(String(rec["path"]), '\\'=>'/'), "paths_sha256"=>String(rec["sha256"]),
        "morph_module"=>replace(STREAMLINE_MORPH_MODULE, '\\'=>'/'), "quality_module"=>replace(STREAMLINE_QUALITY_MODULE, '\\'=>'/'),
        "admission"=>s.admission, "params"=>params, "section_note"=>note,
        "path_source"=>String(rec["source"]), "path_label"=>String(rec["source_label"]))
end

"The result notes of an aligned point (from the mesh's own alignment record)."
function streamline_point_notes(rec)
    rec isa AbstractDict || return String[]
    adm = rec["admission"]
    gates = [g for g in adm["gate_changes"] if g["worse"]]
    gtxt = isempty(gates) ? "no gate got worse" :
        "gates worse than on the unaligned mesh: " * join([@sprintf("%s %.4g -> %.4g (limit %.4g)", g["gate"], g["before"], g["after"], g["limit"]) for g in gates], "; ")
    [STREAMLINE_DEV_LABEL * ". " * String(get(rec, "section_note", "")),
     @sprintf("Alignment admission: %s (%s), TE amplitude %.3g, LE amplitude %.3g; %s. %s", uppercase(String(adm["rule"])), adm["stage"],
              adm["te_amplitude"], adm["le_amplitude"], gtxt, String(adm["label"])),
     "Comparability: an aligned mesh is a different mesh family from the unaligned one (tag $(rec["tag"])); compare aligned and unaligned results as a mesh sensitivity, not as a validation."]
end
