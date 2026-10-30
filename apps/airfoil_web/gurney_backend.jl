# Gurney flap integration of the CFD backends (23 Oct 2026, the plate design; the BAFFLE since 28 Oct 2026). Included inside
# CFDBackend after ch_mesh.jl, gurney.jl and gurney_mesh_audit.jl. With the flap off nothing here runs.
#
# The flap is a zero-thickness plate perpendicular to the chord at the trailing edge, exactly sharp, symmetric = h/2 up and h/2
# down (gurney.jl). The user's rulings of 27 Oct 2026: it is modelled as a BAFFLE -- a wall inside the section's own normal
# finite-trailing-edge C+H mesh along the C block's grid line from the trailing-edge corner -- and never by reshaping the mesh.
# Only the low-speed solver (PBCav, version 2) has interior walls, so:
#   low-speed (PBCav):        the native C+H mesh of the section (the native-base control's) and PB_BAFFLE=gurney:side=...,h=...
#                             (lowspeed_ch_backend.jl, lowspeed_ch_worker.jl; the solver's quality gate may refuse a height);
#   compressible (transonic): refused -- the compressible C+H solver has no interior walls (the earlier plate-insertion mesh,
#                             gurney_mesh.jl, is retired: no longer included, kept only for archived validation scripts);
#   cavitation:               refused -- the Species driver (tools/cmesh_cav_movie.jl) builds its own mesh and has no interior walls;
#   XFOIL / JXfoil:           the clean section, labelled as the no-flap reference (they cannot model the plate);
#   LBM 3D:                   its own THICK plate (lbm_geometry.jl), unchanged.

using .AirfoilGurney

"The request's Gurney settings, or nothing when the flap is off (ArgumentError on a bad value)."
gurney_request(flow::AbstractDict) = AirfoilGurney.gurney_settings(flow)
gurney_enabled(flow::AbstractDict) = gurney_request(flow) !== nothing

const GURNEY_TUNNEL_REFUSAL = "The Gurney flap is not available for the Species (cavitation) backends. The flap is modelled as a baffle, a zero-thickness wall inside PBCav's C+H mesh; the Species water-tunnel run regenerates its mesh inside the production driver (tools/cmesh_cav_movie.jl), which has no interior walls and is not changed by the app, and the air, free-fluid and free-surface paths do not mesh C+H. Use the low-speed CFD (PBCav)."
const GURNEY_COMPRESSIBLE_REFUSAL = "The Gurney flap is available in the low-speed CFD only (PBCav, version 2). It is modelled as a baffle, a zero-thickness wall along a grid line of the section's own finite-trailing-edge C+H mesh, and the baffle exists only in the low-speed solver; the compressible C+H solver has no interior walls. The earlier compressible approach, reshaping its mesh to take the plate, is retired (its meshes failed air geometry gates the clean host passes). Choose the low-speed CFD for a Gurney flap."

"""Refuse every combination the Gurney flap does not support, with the reason (called by the server's validation).
`backends` are the request's backend names; `flow` its flow map (with the server's normalized cav_* keys); `mesh` false for the
geometry-only preview (no mesh and no run: the plate is only drawn)."""
function gurney_check_request(flow::AbstractDict, backends; mesh::Bool=true)
    g = gurney_request(flow)
    g === nothing && return nothing
    te = _get(flow, "cfd_te_thickness", 0.0)
    (te isa Real && te > 0) && throw(ArgumentError(
        "The Gurney flap and the CFD trailing-edge thickness override both set the trailing edge: clear the CFD trailing-edge thickness (the plate stands on the section's own trailing edge)."))
    if any(b -> b in ("transonic", "cavitation", "lowspeed"), backends)
        String(_get(flow, "cfd_te_topology", "auto")) == "auto" || throw(ArgumentError(
            "The Gurney flap is meshed in the conforming C+H topology (TE topology Auto). Legacy A-plus with diagonal fans is deprecated for finite trailing edges and is not offered for the Gurney flap."))
        String(_get(flow, "cfd_mesh", "aplus_nested_refined")) == "legacy_sharp_cmesh" && throw(ArgumentError(
            "The Gurney flap needs a finite base; the Legacy sharp C mesh closes the trailing edge. Choose the refined shared mesh profile."))
    end
    "cavitation" in backends && throw(ArgumentError(GURNEY_TUNNEL_REFUSAL))
    # a geometry-only preview (mesh=false) only draws the plate, whatever backend is selected; a mesh or a run is refused
    (mesh && "transonic" in backends) && throw(ArgumentError(GURNEY_COMPRESSIBLE_REFUSAL))
    g
end

"""The host section of a Gurney request, or nothing when the flap is off: (af::AirfoilPts in wrap order, metadata).
`source` is the request's airfoil map (kind, id); `path` the prepared coordinate file of the job or preview. The metadata
carry the plate (gurney_plate: tips, base, root angles), the trailing-edge trim and the support. THIS IS THE DRAWING'S SECTION
(the preview, the plate's record, the LBM's host): the low-speed CFD meshes the section's native contour (not trimmed) and
cuts the baffle in it (lowspeed_ch_backend.jl)."""
function gurney_section(source::AbstractDict, path::AbstractString, flow::AbstractDict)
    g = gurney_request(flow)
    g === nothing && return nothing
    kind = String(_get(source, "kind", ""))
    if kind == "naca4"
        code = String(_get(source, "id", ""))
        occursin(r"^[0-9]{4}$", code) || throw(ArgumentError("invalid NACA four-digit code"))
        base = CMeshGen.naca4_points(parse(Int, code))
        support = "NACA $code four-digit equations at 4097 cosine-spaced points (the CFD meshes' own support)"
        source_points = length(base.x)
    else
        src = CMeshGen.read_airfoil_dat(path)
        base = CMeshGen.condition_airfoil(src).airfoil
        support = "natural C2 reconstruction of the $(length(src.x)) source coordinates ($(length(base.x)) stations, every source point kept)"
        source_points = length(src.x)
    end
    x, y, trim = AirfoilGurney.gurney_trim_te(base.x, base.y)
    plate = AirfoilGurney.gurney_plate(x, y; height=g.height, mode=g.mode)
    meta = plate.metadata
    meta["trailing_edge_trim"] = trim
    meta["support"] = support; meta["source_points"] = source_points; meta["source_kind"] = kind
    meta["source_id"] = String(_get(source, "id", ""))
    meta["note"] = AirfoilGurney.gurney_note(meta) * (trim["trimmed"] ? "; " * trim["note"] : "")
    (af=CMeshGen.AirfoilPts(x, y), metadata=meta)
end

"""Write the host section in Selig order with 17 significant digits and return the path. Both the app's and the PBCav
snapshot's read_airfoil_dat return the wrap-order contour of this file bit for bit; the app's reader is checked here."""
function write_gurney_section(sec, dir::AbstractString; name::AbstractString="gurney_host_section.dat")
    mkpath(dir)
    f = joinpath(dir, name)
    m = sec.metadata
    open(f, "w") do io
        @printf(io, "Gurney flap host section (the preview's drawing section; the flap is a baffle in the CFD mesh): %s %s, h/c %.6g, %s (%s)\n",
                m["source_kind"], m["source_id"], m["gurney_height"], m["gurney_mode"], AirfoilGurney.GURNEY_VERSION)
        for k in reverse(eachindex(sec.af.x))
            @printf(io, "%.17g %.17g\n", sec.af.x[k], sec.af.y[k])
        end
    end
    back = CMeshGen.read_airfoil_dat(f)
    back.x == sec.af.x && back.y == sec.af.y || error("the Gurney host section file does not read back bit for bit: $f")
    m["file"] = basename(f); m["file_sha256"] = bytes2hex(open(sha256, f))
    f
end

"The host section, written into `dir`, and its read-back contour: (path, af, metadata), or nothing when the flap is off."
function gurney_section_file(source::AbstractDict, path::AbstractString, flow::AbstractDict, dir::AbstractString)
    sec = gurney_section(source, path, flow)
    sec === nothing && return nothing
    f = write_gurney_section(sec, dir)
    (path=f, af=CMeshGen.read_airfoil_dat(f), metadata=sec.metadata)
end

"The label of the panel codes' curves in a Gurney job, and the caveat every point carries."
const GURNEY_PANEL_LABEL = "no-flap reference (XFOIL cannot model the plate)"
const GURNEY_PANEL_CAVEAT = "XFOIL and JXfoil cannot model the Gurney plate: their curves are the clean section's, the no-flap reference."
gurney_point_note(meta) = String(get(meta, "note", AirfoilGurney.gurney_note(meta)))

"""The note of a Gurney flap that is a baffle (the low-speed CFD): the plate as the user defined it, and what the mesh is."""
function gurney_baffle_note(meta::AbstractDict)
    mode = String(meta["gurney_mode"])
    where_ = mode == "symmetric" ? "symmetric: a T, h/2 above the upper and h/2 below the lower trailing-edge point" :
             mode == "lower" ? "on the lower (pressure) side: from the lower trailing-edge point down h" : "on the upper (suction) side: from the upper trailing-edge point up h"
    @sprintf("CFD geometry change: Gurney flap modelled as a baffle, a zero-thickness plate perpendicular to the chord at the trailing edge, exactly sharp (%s), h/c = %.4g, in the section's own finite-trailing-edge C+H mesh (a wall along the grid line from the corner; the mesh and the airfoil, with its native base of %.5f c, are not reshaped)",
        where_, meta["gurney_height"], meta["source_te_gap"])
end

"The /api/health description of the Gurney flap option."
gurney_capabilities() = Dict{String,Any}("available"=>true, "version"=>AirfoilGurney.GURNEY_VERSION, "model"=>"baffle",
    "modes"=>collect(AirfoilGurney.GURNEY_MODES), "height_range"=>collect(AirfoilGurney.GURNEY_HEIGHT_RANGE),
    "refine"=>Dict{String,Any}("request_key"=>"gurney_refine", "levels"=>collect(AirfoilGurney.GURNEY_REFINE_LEVELS), "default_api"=>"off",
        "knobs"=>Dict{String,Any}(k=>v for (k, v) in AirfoilGurney.GURNEY_REFINE_KNOBS), "api_revision"=>17,
        "model"=>"rows added around the plate tip and wake, with density transferred from the plate root into the flap-height recirculation region; the finite-TE H block is coarsened across its gap while its interfaces and edge cells stay fixed (PBCav PB_BAFFLE_REFINE): tip cells are about 1/q of the section's own; admitted by convexity, corner and face angles, cell-ratio and aspect-ratio checks, else the section's own mesh with the reason in the baffle record"),
    "geometry"=>"a zero-thickness plate perpendicular to the chord at the trailing edge, exact sharp root and tip corners; symmetric = h/2 above and h/2 below; modelled as a baffle (an interior wall) on the section's own C+H mesh",
    "gate"=>"the solver refuses a plate outside the mesh's admitted range of h (on the demo mesh of NACA 0012: 5.75e-4 to 0.0726 c one side, 1.15e-3 to 0.145 c symmetric); the range for the current mesh is returned by Create mesh",
    "backends"=>Dict("xfoil"=>GURNEY_PANEL_LABEL, "jxfoil"=>GURNEY_PANEL_LABEL,
        "transonic"=>"refused: the baffle exists only in the low-speed solver",
        "lowspeed"=>"PBCav C+H free-air mesh, the plate a baffle (snapshot $(basename(LOWSPEED_PINS["s1028"].snapshot)), $(LOWSPEED_PINS["s1028"].commit)); ramp start, PB_TURB_MUT_BASE=committed",
        "cavitation"=>"refused: the Species driver builds its own mesh"))
