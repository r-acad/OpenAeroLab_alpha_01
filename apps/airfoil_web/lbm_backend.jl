# =====================================================================================================================
# lbm_backend.jl -- app orchestration of the "LBM 3D (LUDWIG)" backend (27 Oct 2026). Included inside CFDBackend after
# lbm_geometry.jl and lbm_case.jl. The HTTP process loads no LBM code: Create mesh is pure geometry here (the section,
# its extrusion and STL, and a replica of the solver's single-level lattice arithmetic); Run starts lbm_worker.jl, which
# takes the shared GPU lock and runs the pinned LUDWIG snapshot's src/main.jl as it is. The solver profile (lbm_case.jl:
# "lbmio", the lbm-io-opts snapshot, or "lbmapp", the first path) decides the outputs: "lbmio" gives numeric mid-plane
# fields (the app's viewer files num/<field>/s<step>.f16 and a fixed-scale video) and the solver's own build-only domain
# check; "lbmapp" gives the solver's slice pictures.
#
# Request keys (flow map; all optional, defaults in AirfoilLBMCase.LBM_DEFAULTS): lbm_span_ratio, lbm_cells_per_chord
# (finest-level cells per chord), lbm_ctu, lbm_ramp_ctu, lbm_frames, lbm_upstream, lbm_downstream, lbm_height,
# lbm_plate_cells, lbm_wall_model, lbm_spoiler (+ lbm_spoiler_hinge, lbm_spoiler_length, lbm_spoiler_deflection), and the
# Gurney keys gurney_height / gurney_mode shared with the other CFD backends (a THICK plate here: lbm_plate_cells cells).
# Reynolds is the request's reynolds; Mach is not an input (the LBM runs at lattice Mach 0.139, incompressible regime).
# =====================================================================================================================
using .AirfoilLBMGeometry
using .AirfoilLBMCase

const LBM_LABEL = "LBM 3D (LUDWIG)"
const LBM_THREADS = 8
lbm_worker_path() = get(ENV, "AIRFOIL_LBM_WORKER", joinpath(@__DIR__, "lbm_worker.jl"))

"Validated LBM settings of a request (ArgumentError with the reason). Called by the server's request validation."
function lbm_configuration(flow::AbstractDict)
    s = AirfoilLBMCase.lbm_settings(flow)
    sp = AirfoilLBMGeometry.spoiler_settings(flow)
    s["spoiler"] = sp === nothing ? nothing : Dict("hinge"=>sp.hinge, "length"=>sp.length, "deflection"=>sp.deflection)
    te = _get(flow, "cfd_te_thickness", 0.0)
    (te isa Real && te > 0) && throw(ArgumentError(
        "The LBM 3D backend extrudes the section's own trailing edge: clear the CFD trailing-edge thickness override."))
    s
end

lbm_capabilities() = Dict{String,Any}("label"=>LBM_LABEL,
    "solver"=>"LUDWIG lattice-Boltzmann, D3Q27 cumulant, single level, spanwise periodic (development; not validated for airfoils)",
    "commit"=>AirfoilLBMCase.lbm_commit()[1:8], "snapshot"=>AirfoilLBMCase.lbm_snapshot(), "profile"=>AirfoilLBMCase.lbm_profile_name(),
    "profile_label"=>AirfoilLBMCase.lbm_profile().label,
    "mid_plane"=>AirfoilLBMCase.lbm_io() ? "numeric fields (|u|/U, vorticity, Cp, u, w, v) on fixed colour scales" : "the solver's slice pictures",
    "available"=>isfile(joinpath(AirfoilLBMCase.lbm_snapshot(), "src", "main.jl")), "defaults"=>AirfoilLBMCase.LBM_DEFAULTS,
    "gpu"=>"RTX 3080 through the shared GPU lock (the run waits while another owner holds it)",
    "memory_limit_gb"=>AirfoilLBMCase.LBM_GPU_LIMIT_GB, "levels"=>AirfoilLBMCase.lbm_pamr() ? AirfoilLBMCase.LBM_LEVELS_MAX : 1,
    "friction_note"=>AirfoilLBMCase.LBM_FRICTION_NOTE, "wall_position_note"=>AirfoilLBMCase.LBM_WALL_POSITION_NOTE, "wall_model_label"=>AirfoilLBMCase.LBM_WALL_MODEL_LABEL,
    "z_walls"=>AirfoilLBMCase.lbm_zwalls() ? ["slip", "no_slip"] : ["slip"],
    "level_viscosity"=>AirfoilLBMCase.lbm_levelvisc(), "level_viscosity_tooltip"=>AirfoilLBMCase.LBM_LEVELVISC_TOOLTIP,
    "time_fix"=>get(AirfoilLBMCase.lbm_profile(), :timefix, false) === true,
    "z_walls_note"=>(AirfoilLBMCase.lbm_zwalls() ? "" : "no-slip needs the solver update (lbm-zwalls). ") * AirfoilLBMCase.LBM_ZWALLS_NOTE,
    "estimate"=>Dict{String,Any}("bytes_per_cell"=>AirfoilLBMCase.lbm_io() ? AirfoilLBMCase.LBM_BYTES_PER_CELL_IO : AirfoilLBMCase.LBM_BYTES_PER_CELL_APP,
        "fixed_gb"=>AirfoilLBMCase.LBM_FIXED_GB, "mlups"=>AirfoilLBMCase.LBM_MLUPS, "amr_bytes_per_cell"=>AirfoilLBMCase.LBM_AMR_BYTES_PER_CELL,
        "amr_bytes_per_wall_cell"=>AirfoilLBMCase.LBM_AMR_BYTES_PER_WALL_CELL,
        "amr_step_model"=>Dict{String,Any}("rate_per_s"=>[AirfoilLBMCase.LBM_MLUPS*1e6, 67.4e6, 28.5e6, 28.5e6], "fixed_s"=>[0.0, 0.0452, 0.143, 0.286],
            "per"=>"solver step (one coarsest-level step); work = sum over levels of cells x 2^(level-1)"),
        "export_seconds_per_node"=>AirfoilLBMCase.LBM_EXPORT_SECONDS_PER_NODE, "basis"=>"measured on the integration cases (27 Oct 2026)"),
    "refinement"=>AirfoilLBMCase.lbm_pamr() ? "1-$(AirfoilLBMCase.LBM_LEVELS_MAX) levels with periodic-Y AMR (the span snapped to whole coarsest blocks)" :
                  "single level only until the solver's periodic-Y AMR is pinned for the app (branch lbm-periodic-amr)",
    "video"=>isfile(AirfoilLBMCase.lbm_ffmpeg()) ? "ffmpeg H.264 MP4" : "ffmpeg missing: frames only")

"""The host contour of an LBM request in wrap order (lower TE -> LE -> upper TE), and the Gurney settings: with a Gurney
flap the app's Gurney host section (gurney_section: the source section, its trailing-edge points brought onto one x
station); else the NACA equations or the conditioned coordinate file, as the other CFD meshes use."""
function _lbm_host(source::AbstractDict, path::AbstractString, flow::AbstractDict)
    g = gurney_request(flow)
    if g !== nothing
        sec = gurney_section(source, path, flow)
        return sec.af.x, sec.af.y, (height=g.height, mode=g.mode), "Gurney host section: " * String(sec.metadata["support"])
    end
    kind = String(_get(source, "kind", ""))
    if kind == "naca4"
        code = String(_get(source, "id", ""))
        af = CMeshGen.naca4_points(parse(Int, code))
        return af.x, af.y, nothing, "NACA $code four-digit equations at 4097 cosine-spaced points"
    end
    src = CMeshGen.read_airfoil_dat(path)
    af = CMeshGen.condition_airfoil(src).airfoil
    af.x, af.y, nothing, "natural C2 reconstruction of the $(length(src.x)) source coordinates"
end

"""Build the LBM geometry and case of one angle into `dir`: lbm_section.stl, config.yaml and the viewer payload
(lbm_lattice.msgpack). Create mesh and Run call this identically, so the STL and the case are the same bytes."""
function _lbm_build(source::AbstractDict, path::AbstractString, flow::AbstractDict, alpha::Real, dir::AbstractString;
                    note::AbstractString="")
    G = AirfoilLBMGeometry
    s = lbm_configuration(flow)
    N = s["cells_per_chord"]; dx = 1.0/N
    x, y, gur, support = _lbm_host(source, path, flow)
    sp = s["spoiler"] === nothing ? nothing : (hinge=s["spoiler"]["hinge"], length=s["spoiler"]["length"], deflection=s["spoiler"]["deflection"])
    poly, smeta = G.lbm_section(x, y; ds=0.5dx, thickness=s["plate_thickness"], gurney=gur, spoiler=sp)
    # the STL covers the period and runs past both periodic faces (at least two coarsest cells with several levels);
    # its side walls are split at y = 0 and y = span (lbm_extrude's stations), so no side triangle straddles a face
    ext_cells = max(6, 2^s["levels"])
    V, T, emeta = G.lbm_extrude(poly; span=s["span"], ext=ext_cells*dx, dy_station=2dx)
    closed = G.stl_check_closed(T)
    closed.closed || error("the extruded section is not closed ($(closed.bad_edges) open edges)")
    mkpath(dir)
    stl = joinpath(dir, "lbm_section.stl")
    G.write_stl_binary(stl, V, T; header="Airfoil Lab LBM 3D section (lbm_geometry.jl)")
    tris = G.read_stl_binary(stl)
    lat = G.lbm_lattice(tris; alpha_lbm_deg=-Float64(alpha), cells_per_chord=N, span_cells=s["span_cells"],
                        upstream=s["upstream"], downstream=s["downstream"], height=s["height"], levels=s["levels"])
    lat.ny == s["span_cells"] || error("the resolved span ($(lat.ny) cells) differs from the requested $(s["span_cells"])")
    poly32 = [(Float64(Float32(p[1])), Float64(Float32(p[2]))) for p in poly]
    placed = G.placed_polygon(poly32, lat)
    win = G.obstacle_window(placed, lat)
    if s["levels"] > 1
        # the finest shell must stay out of the solver's farfield wall buffer (a refined level would touch a wall; the
        # solver only warns): refuse with the largest shell that fits
        maxshell, face = G.lbm_max_shell_blocks(placed, lat)
        s["shell_blocks"] <= maxshell || throw(ArgumentError(maxshell >= 1 ?
            "The finest-level shell of $(s["shell_blocks"]) blocks would reach the solver's wall buffer at the $(face): use lbm_shell_blocks <= $(maxshell) or enlarge the domain (lbm_height, lbm_upstream, lbm_downstream)." :
            "The domain leaves no room for a refined level between the body and the $(face) (the solver's 4-block wall buffer at $(N) cells per chord): enlarge the domain (lbm_height, lbm_upstream, lbm_downstream) or use fewer levels."))
    end
    perimeter = sum(hypot(poly[i][1] - poly[mod1(i + 1, length(poly))][1], poly[i][2] - poly[mod1(i + 1, length(poly))][2]) for i in eachindex(poly))
    level_cells = AirfoilLBMCase.lbm_amr_level_cells(s, lat; perimeter=perimeter, area=smeta["area"])
    pxs = first.(placed); pzs = last.(placed)
    window = AirfoilLBMCase.lbm_export_window(s, (minimum(pxs), minimum(pzs)), (maximum(pxs), maximum(pzs)), lat.Lx, lat.Lz)
    if window !== nothing     # whole finest cells, as the solver snaps it outward
        window = [floor(window[1]/dx + 1e-9)*dx, ceil(window[2]/dx - 1e-9)*dx, floor(window[3]/dx + 1e-9)*dx, ceil(window[4]/dx - 1e-9)*dx]
    end
    plane_nodes = window === nothing ? lat.nx*lat.nz : round(Int, (window[2] - window[1])/dx)*round(Int, (window[4] - window[3])/dx)
    est = AirfoilLBMCase.lbm_estimate(s, lat.nx, lat.ny, lat.nz; level_cells=s["levels"] > 1 ? level_cells : nothing, plane_nodes=plane_nodes)
    est["fits"] || throw(ArgumentError(@sprintf(
        "This LBM case needs about %.1f GB of GPU memory (per-cell allocation measured on the integration cases, plus 0.7 GB; %s cells, %d x %d x %d), above the %.0f GB limit of the %.0f GB card: reduce the cells per chord, the span or the domain.",
        est["memory_gb"], _lbm_count(est["cells"]), lat.nx, lat.ny, lat.nz, AirfoilLBMCase.LBM_GPU_LIMIT_GB, AirfoilLBMCase.LBM_GPU_MEMORY_GB)))
    io = AirfoilLBMCase.lbm_io()
    write(joinpath(dir, "config.yaml"), AirfoilLBMCase.lbm_case_yaml(s, alpha; note=note, io=io, export_window=window))
    yplus = est["yplus_estimate"]
    notes = String[
        s["levels"] == 1 ?
            "LBM 3D (LUDWIG, snapshot $(AirfoilLBMCase.lbm_commit()[1:8]), profile $(AirfoilLBMCase.lbm_profile_name())): D3Q27 cumulant lattice Boltzmann on ONE uniform level, $(N) cells per chord; development path, not validated for airfoils." :
            @sprintf("LBM 3D (LUDWIG, snapshot %s, profile %s): D3Q27 cumulant lattice Boltzmann on %d levels with periodic-Y AMR, %d cells per chord at the finest level (%.4g at the coarsest); every level spans the period; a %d-block finest shell around the body, the wake refined to level %d; gold-standard level coupling. Development path, not validated for airfoils.",
                     AirfoilLBMCase.lbm_commit()[1:8], AirfoilLBMCase.lbm_profile_name(), s["levels"], N, s["coarsest_cells_per_chord"], s["shell_blocks"], s["levels"] - 1),
        @sprintf("The section is extruded along the span (the solver's Y axis) by %d finest cells = %.4g c (requested %.4g c, snapped to whole %s of %d cells); both span faces are periodic (%s).", s["span_cells"], s["span"], s["span_ratio_requested"],
                 s["levels"] == 1 ? "8-cell blocks" : "coarsest-level blocks", s["span_quantum_cells"], AirfoilLBMCase.lbm_uses_pamr(s) ? "advanced.boundary.periodic_y_amr" : "advanced.boundary.periodic_y"),
        @sprintf("Domain: %.3g c upstream of the body, %.3g c downstream, %.3g c above and below; inlet: velocity; outlet: extrapolation with a 0.2 c sponge; top and bottom: %s.", s["upstream"], s["downstream"], s["height"],
                 s["z_walls"] == "no_slip" ? "NO-SLIP walls (stationary halfway bounce-back, no sponge)" :
                 s["zwalls_option"] ? "SLIP walls (advanced.boundary.z_walls: an exact specular reflection, no z-face sponge)" :
                                      "SLIP walls (the solver's legacy local mirror walls without the z-face sponge; their wall row lags 14-34 %, fixed in the solver's z_walls option)"),
        AirfoilLBMCase.LBM_ZWALLS_NOTE,
        @sprintf("Re = %.4g (chord); lattice speed %.3g (lattice Mach %.3f); tau = %.5f%s. Mach is not an input: the solver runs incompressible-regime D3Q27 at this lattice Mach.", s["reynolds"], s["u_lattice"], s["lattice_mach"], s["tau"],
                 s["tau_clipped"] ? @sprintf(" is below the solver's floor %.3f, which it uses instead: the effective Reynolds number is about %.3g", AirfoilLBMCase.LBM_TAU_MIN, s["reynolds_effective"]) : ""),
        @sprintf("Wall: Grad-reconstruction Bouzidi wall on the STL; first-cell y+ estimate %.2g%s.", yplus, s["wall_model"] ? "; the wall-model switch is ON but inactive in the current solver (it changes nothing there; not validated)" : " (wall model off: a resolved wall is assumed)"),
        "Angle of attack: the solver rotates the STL about +Y (basic.alpha = -alpha: LUDWIG's positive alpha raises the trailing edge); lift is +Z, drag +X.",
        AirfoilLBMCase.LBM_FRICTION_NOTE,
        AirfoilLBMCase.LBM_WALL_POSITION_NOTE,
        "Force coefficients: the solver's surface integration over the triangles inside the periodic span, reference area chord x span; CM is not reported (the solver's moment centre is not the quarter chord of the rotated section).",
        io ? "Mid-plane results are the solver's own numeric fields on the XZ plane through mid-span (advanced.slice.export_data: the cell values of its finest level, no interpolation), shown and filmed on FIXED colour scales: |u|/U, vorticity (counter-clockwise positive), Cp against the inlet state, u/U, w/U and the spanwise v/U." :
             "Mid-plane results are the solver's own slice images (XZ plane through mid-span); the solver writes no numeric slice data, so the fields are pictures with their own colour scales."]
    gur === nothing || push!(notes, @sprintf("Gurney flap: a THICK plate (%.3g cells = %.4g c) perpendicular to the chord at the trailing edge, h/c = %.4g, %s; its back face on the base line x = x_TE (the immersed boundary needs a resolvable thickness, unlike the zero-thickness plate of the other CFD backends).",
                                             s["plate_cells"], s["plate_thickness"], gur.height, gur.mode))
    sp === nothing || push!(notes, @sprintf("Spoiler: a flat plate %.3g cells (%.4g c) thick hinged on the upper surface at x/c = %.3g, length %.3g c, deflected %.3g deg from the local surface tangent; its rear face runs from the hinge, its front face meets the surface %.4g c upstream; the cove under it is fluid.",
                                            s["plate_cells"], s["plate_thickness"], sp.hinge, sp.length, sp.deflection, smeta["spoiler"]["front_root_offset"]))
    s["tau_clipped"] && push!(notes, "Re is above what this lattice resolves at the solver's tau floor; raise the cells per chord or lower Re for the requested viscosity.")
    if s["levels"] > 1
        lt = AirfoilLBMCase.lbm_level_taus(s)
        push!(notes, s["level_viscosity"] == "physical" ?
            "Level viscosity: PHYSICAL on the coarse levels (advanced.refinement.level_viscosity; EXPERIMENTAL: its app-length stability screen (24 c/U) did not pass, the growth of the C_L fluctuation amplitude differing from the legacy twin's (1.5 x against 0.96 x), with no numerical instability seen; results with it are not validated. Observation from that screen: the peak velocity on the coarsest level after 8 c/U was 9-20 % above the legacy twin's). " *
            join([@sprintf("Level %d: tau %.6f%s, %.3g x the physical viscosity", t.level, t.tau, t.floored ? " (floored at tau_min)" : "", t.ratio) for t in lt], "; ") * "." :
            "Level viscosity: the solver's legacy level rule gives each coarse level 4^(L-l) x the physical viscosity (measured 4.005 x one level up: the level-viscosity record). " *
            join([@sprintf("Level %d: tau %.6f, %.3g x physical", t.level, t.tau, t.ratio) for t in lt], "; ") * "." *
            (s["levelvisc_option"] ? " The experimental option \"Physical viscosity on coarse levels\" lowers it (off by default)." : " The solver snapshot has no option to change it."))
    end
    s["levels"] > 1 && push!(notes, @sprintf("Cells per level (coarsest first; an ESTIMATE scaled from the one measured periodic-AMR case, record T8; the solver's own build gives the counts): %s; about %.1f GB and %.2g s per finest-level step.",
                                             join(_lbm_count.(level_cells), " / "), est["memory_gb"], est["seconds_per_step"]))
    window === nothing || push!(notes, @sprintf("The numeric mid-plane covers x %.3g-%.3g m, z %.3g-%.3g m of the domain at the finest spacing (each node the cell value of the finest level there).", window...))
    payload = Dict{String,Any}("kind"=>"lbm_lattice", "version"=>"lbm_lattice_v1", "label"=>LBM_LABEL, "alpha_deg"=>Float64(alpha),
        "basic_alpha_deg"=>-Float64(alpha),
        "section"=>Dict("x"=>first.(poly), "z"=>last.(poly)), "section_meta"=>smeta, "support"=>support,
        "placed"=>Dict("x"=>first.(placed), "z"=>last.(placed)),
        "domain"=>Dict("Lx"=>lat.Lx, "Ly"=>lat.Ly, "Lz"=>lat.Lz, "nx"=>lat.nx, "ny"=>lat.ny, "nz"=>lat.nz, "dx"=>lat.dx,
                       "bx"=>lat.bx, "by"=>lat.by, "bz"=>lat.bz, "cells"=>lat.cells, "blocks"=>lat.blocks, "block_size"=>8,
                       "offset"=>collect(lat.offset), "rotation_origin"=>collect(lat.origin), "levels"=>lat.levels,
                       "dx_coarse"=>lat.dx_coarse, "coarse_cells"=>[lat.ncx, lat.ncy, lat.ncz], "block_m"=>lat.block_m,
                       "level_cells_estimate"=>level_cells, "export_window"=>window,
                       "periodic"=>"y", "mid_span_y"=>lat.Ly/2,
                       "faces"=>Dict("x_min"=>"inlet (velocity)", "x_max"=>"outlet (extrapolation + sponge)", "y_min"=>"periodic", "y_max"=>"periodic",
                                     "z_min"=>"slip wall + freestream sponge", "z_max"=>"slip wall + freestream sponge")),
        "span"=>Dict("cells"=>s["span_cells"], "length"=>s["span"], "requested_ratio"=>s["span_ratio_requested"],
                     "stl_y"=>emeta["stations"], "extension"=>6dx),
        "mask"=>Dict("i0"=>win.i0, "k0"=>win.k0, "ni"=>win.ni, "nk"=>win.nk, "solid"=>win.solid, "shell"=>win.shell,
                     "cells"=>Int.(win.mask), "rule"=>"solver voxelization: a cell is solid when its 1.5 dx box touches the surface or it is enclosed"),
        "stl"=>Dict("file"=>stl, "triangles"=>length(T), "vertices"=>length(V), "closed"=>closed.closed,
                    "side_triangles"=>emeta["side_triangles"], "cap_triangles"=>emeta["cap_triangles"]),
        "settings"=>s, "estimate"=>est, "notes"=>notes, "config"=>joinpath(dir, "config.yaml"),
        "profile"=>AirfoilLBMCase.lbm_profile_name(), "commit"=>AirfoilLBMCase.lbm_commit())
    write(joinpath(dir, "lbm_lattice.msgpack"), AirfoilWire.encode(_lbm_clean(payload), :msgpack))
    (payload=payload, settings=s, lattice=lat, estimate=est, notes=notes)
end

_lbm_count(n) = n >= 1e6 ? @sprintf("%.2f M", n/1e6) : string(n)
"msgpack-safe copy (NamedTuples to Dicts, nothing kept, non-finite numbers to nothing)."
_lbm_clean(x::AbstractDict) = Dict{String,Any}(string(k)=>_lbm_clean(v) for (k, v) in x)
_lbm_clean(x::NamedTuple) = Dict{String,Any}(string(k)=>_lbm_clean(v) for (k, v) in pairs(x))
_lbm_clean(x::AbstractVector) = eltype(x) <: Real && !(eltype(x) <: Bool) ? [isfinite(v) ? v : nothing for v in x] : Any[_lbm_clean(v) for v in x]
_lbm_clean(x::Tuple) = Any[_lbm_clean(v) for v in x]
_lbm_clean(x::AbstractFloat) = isfinite(x) ? Float64(x) : nothing
_lbm_clean(x) = x

"""Create mesh for the LBM backend: the section, its STL and the lattice, at the form's angle, in a few seconds (no
solver process: the lattice is the replica of the solver's arithmetic, checked against the solver's own domain record at
Run, and on the "lbmio" profile also by the API-only domain check, the solver's build-only mode)."""
function preview_lbm_mesh(path::AbstractString, flow::AbstractDict, outdir::AbstractString; source::AbstractDict=Dict())
    alpha = _number(flow, "alpha_deg", 0.0; lo=-30, hi=30)
    b = _lbm_build(source, path, flow, alpha, outdir; note="Create mesh preview (not run)")
    meta = Dict{String,Any}("geometry_support"=>"LBM 3D: " * b.payload["support"], "lbm"=>true,
        "mesh_cells"=>b.estimate["cells"], "lbm_note"=>"lattice from the app's replica of the solver's domain arithmetic")
    (lbm=b.payload, metadata=meta)
end

"The section for a geometry-only preview (no lattice): the polygon with the thick plates at the default resolution."
function lbm_section_preview(path::AbstractString, flow::AbstractDict; source::AbstractDict=Dict())
    s = lbm_configuration(flow)
    x, y, gur, support = _lbm_host(source, path, flow)
    sp = s["spoiler"] === nothing ? nothing : (hinge=s["spoiler"]["hinge"], length=s["spoiler"]["length"], deflection=s["spoiler"]["deflection"])
    dx = 1.0/s["cells_per_chord"]
    poly, meta = AirfoilLBMGeometry.lbm_section(x, y; ds=0.5dx, thickness=s["plate_thickness"], gurney=gur, spoiler=sp)
    Dict{String,Any}("x"=>first.(poly), "z"=>last.(poly), "meta"=>_lbm_clean(meta), "support"=>support)
end

"Kill a recorded solver pid if it is still a running julia process (the worker was stopped before it could)."
function _lbm_stop_recorded_solver(dir::AbstractString, log::Function)
    f = joinpath(dir, "lbm_solver.pid")
    isfile(f) || return
    pid = tryparse(Int, strip(read(f, String)))
    pid === nothing && return
    alive = try
        occursin("julia", lowercase(read(`tasklist /FI "PID eq $pid" /NH`, String)))
    catch
        false
    end
    alive || return
    _deliver_log(log, "Stopping the LBM solver left running by its worker (recorded pid $pid).")
    try run(`taskkill /PID $pid /T /F`) catch end
end
"Remove the GPU lock a killed LBM worker left behind (only one whose owner.txt names that worker's pid)."
function _lbm_release_if_pid!(pid::Integer)
    f = joinpath(AirfoilGPULock.LOCK_DIR, AirfoilGPULock.OWNER_FILE)
    isfile(f) || return false
    text = try read(f, String) catch; return false end
    occursin("Airfoil Lab LBM 3D worker", text) && occursin("; pid $(pid);", text) || return false
    rm(AirfoilGPULock.LOCK_DIR; recursive=true, force=true)
    true
end

"Run the worker and turn its lines into job events (ITERATION, GPU_WAIT/GPU_ACQUIRED, LBM_FRAME, log lines)."
function _lbm_log_process(cmd::Cmd, dir::AbstractString, progress::Function, log::Function, alpha::Float64, total::Integer)
    logpath = joinpath(dir, "worker.log")
    open(logpath, "w") do io
        output = Pipe()
        proc = run(pipeline(cmd; stdout=output, stderr=output); wait=false)
        close(output.in)
        wpid = getpid(proc)
        gpu_wait = Ref{Union{Nothing,String}}(nothing)
        reader = @async for line in eachline(output)
            println(io, line); flush(io)
            v = split(line, ',')
            if length(v) == 8 && v[1] == "ITERATION"
                n = tryparse(Int, v[2]); tot = tryparse(Int, v[3]); r = tryparse(Float64, v[4])
                cl = tryparse(Float64, v[7]); cd = tryparse(Float64, v[8])
                if n !== nothing && tot !== nothing && all(x -> x !== nothing && isfinite(x), (r, cl, cd))
                    progress(Dict{String,Any}("event"=>"iteration", "backend"=>"lbm", "alpha_deg"=>alpha, "iteration"=>n,
                        "total_iterations"=>tot, "residual"=>r, "residual_scaled"=>r, "residual_linf"=>r, "residual_l2"=>r,
                        "residual_kind"=>"LBM force change per sample, |dCL| + |dCD|", "cl"=>cl, "cd"=>cd))
                end
            elseif startswith(line, "GPU_WAIT,")
                owner = String(line[10:end])
                gpu_wait[] == owner || _deliver_log(log, "Waiting for the GPU (held by $(owner)); the worker retries every 60 s and never breaks the lock.")
                gpu_wait[] = owner
                progress(Dict("event"=>"gpu_wait", "backend"=>"lbm", "alpha_deg"=>alpha, "owner"=>owner))
            elseif startswith(line, "GPU_ACQUIRED,")
                gpu_wait[] = nothing
                _deliver_log(log, "GPU lock taken: $(line[14:end]).")
                progress(Dict("event"=>"gpu_acquired", "backend"=>"lbm", "alpha_deg"=>alpha, "owner"=>String(line[14:end])))
            elseif length(v) == 4 && v[1] == "LBM_FRAME" && tryparse(Int, v[2]) !== nothing
                progress(Dict{String,Any}("event"=>"lbm_frame", "backend"=>"lbm", "alpha_deg"=>alpha, "iteration"=>parse(Int, v[2]),
                    "velocity"=>joinpath(dir, v[3]), "vorticity"=>joinpath(dir, v[4])))
            elseif length(v) == 8 && v[1] == "LBM_NFRAME" && tryparse(Int, v[2]) !== nothing
                st = parse(Int, v[2]); nums = tryparse.(Float64, v[3:8])
                if all(x -> x !== nothing, nums)
                    files = Dict{String,Any}(f.key => joinpath(dir, "num", f.key, @sprintf("s%07d.f16", st)) for f in AirfoilLBMRender.LBM_NUM_FIELDS)
                    filter!(kv -> isfile(last(kv)), files)
                    progress(Dict{String,Any}("event"=>"lbm_frame", "backend"=>"lbm", "alpha_deg"=>alpha, "iteration"=>st,
                        "numeric"=>Dict{String,Any}("step"=>st, "ctu"=>nums[1], "n1"=>Int(nums[2]), "n2"=>Int(nums[3]),
                            "origin_m"=>[nums[4], nums[5]], "spacing_m"=>nums[6], "files"=>files, "mask"=>joinpath(dir, "num", "mask.u8"),
                            "fields"=>_lbm_field_meta())))
                end
            else
                _deliver_log(log, line)
            end
        end
        progress(Dict("event"=>"started", "backend"=>"lbm", "alpha_deg"=>alpha, "pid"=>wpid))
        started = time(); heartbeat = started
        while process_running(proc)
            sleep(1)
            if time() - heartbeat >= 10
                ev = Dict{String,Any}("event"=>"running", "backend"=>"lbm", "alpha_deg"=>alpha, "elapsed_seconds"=>round(Int, time() - started))
                gpu_wait[] === nothing || (ev["gpu_wait"] = gpu_wait[])
                progress(ev); heartbeat = time()
            end
        end
        wait(proc); wait(reader)
        if !success(proc)
            _lbm_stop_recorded_solver(dir, log)
            _lbm_release_if_pid!(wpid) && _deliver_log(log, "Released the GPU lock that LBM worker pid $(wpid) left behind.")
        end
        success(proc)
    end
end

"The numeric mid-plane fields with their fixed colour scales (lbm_render.jl), for the viewer."
_lbm_field_meta() = [Dict{String,Any}("key"=>f.key, "label"=>f.label, "cmap"=>f.cmap, "range"=>collect(f.range),
                                      "mid"=>isfinite(f.mid) ? f.mid : nothing, "units"=>f.units) for f in AirfoilLBMRender.LBM_NUM_FIELDS]

"The video title of a point (ASCII: the frames' bitmap font)."
function _lbm_title(source, s, alpha, flow)
    kind = String(_get(source, "kind", "")); id = String(_get(source, "id", ""))
    name = kind == "naca4" ? "NACA $(id)" : isempty(id) ? "airfoil" : uppercase(id)
    g = gurney_request(flow)
    extra = g === nothing ? "" : @sprintf("  Gurney h/c %.3g", g.height)
    s["spoiler"] === nothing || (extra *= @sprintf("  spoiler %.3g c at %.3g c, %.3g deg", s["spoiler"]["length"], s["spoiler"]["hinge"], s["spoiler"]["deflection"]))
    t = @sprintf("LBM 3D (LUDWIG)  %s%s  alpha %.2f deg  Re %.4g  %d cells/c%s  span %.3g c  %s walls", name, extra, alpha, s["reynolds"], s["cells_per_chord"],
                 s["levels"] > 1 ? " ($(s["levels"]) levels$(s["level_viscosity"] == "physical" ? ", physical coarse nu (exp.)" : ""))" : "", s["span"], s["z_walls"] == "no_slip" ? "no-slip" : "slip")
    String(map(c -> isascii(c) ? c : '?', t))
end

function _lbm_point(case, path, flow, alpha, dir, progress, log)
    profile = AirfoilLBMCase.lbm_profile_name()
    snap, head = AirfoilLBMCase.lbm_snapshot_check(profile)
    source = _get(case, "airfoil", Dict())
    jobid = basename(dirname(dirname(normpath(dir))))
    suffix = replace(first(jobid, 8) * "_" * replace(basename(dir), r"[^A-Za-z0-9]+" => "_"), r"_+" => "_")
    b = _lbm_build(source, path, flow, alpha, dir; note="app job $(jobid), point $(basename(dir))")
    s = b.settings; est = b.estimate
    progress(Dict{String,Any}("event"=>"lbm_lattice", "backend"=>"lbm", "alpha_deg"=>Float64(alpha), "lbm"=>b.payload))
    s["cpu_run"] && !AirfoilLBMCase.lbm_io(profile) && throw(ArgumentError("lbm_cpu_run needs the lbm-io-opts solver profile (the first snapshot's CPU backend does not march)"))
    cap = s["wall_cap_s"] > 0 ? s["wall_cap_s"] : s["domain_check"] ? 1800.0 : clamp(4est["total_seconds"], 1800.0, 172800.0)
    cfg = Dict{String,Any}("outdir"=>replace(dir, '\\' => '/'), "domain_check"=>s["domain_check"], "case_suffix"=>suffix,
        "steps"=>s["steps"], "steps_per_ctu"=>s["steps_per_ctu"], "wall_cap_s"=>cap, "threads"=>LBM_THREADS,
        "max_lock_wait_s"=>s["max_lock_wait_s"], "lock_settle_s"=>s["lock_settle_s"], "u_lattice"=>s["u_lattice"], "profile"=>profile, "cpu_run"=>s["cpu_run"],
        "title"=>_lbm_title(source, s, alpha, flow))
    open(io -> TOML.print(io, cfg), joinpath(dir, "input.toml"), "w")
    L = b.lattice.levels
    grid = L == 1 ? @sprintf("%d x %d x %d = %s cells", b.lattice.nx, b.lattice.ny, b.lattice.nz, _lbm_count(b.lattice.cells)) :
        @sprintf("%d levels, coarsest grid %d x %d x %d, about %s cells in all (%s, coarsest first, estimated)", L, b.lattice.ncx, b.lattice.ncy, b.lattice.ncz,
                 _lbm_count(est["cells"]), join(_lbm_count.(est["level_cells"]), " / "))
    _deliver_log(log, @sprintf("Starting LBM 3D (LUDWIG %s): %s (%d cells per chord%s, span %d cells), %d steps (%.3g CTU, %.0f steps per CTU), %d mid-plane frames; estimated %.1f GB GPU memory and %.0f min (%s). %s",
        head[1:8], grid, s["cells_per_chord"], L > 1 ? " at the finest level" : "", s["span_cells"],
        s["steps"], s["ctu"], s["steps_per_ctu"], s["frames"], est["memory_gb"], est["total_seconds"]/60, est["basis"],
        s["domain_check"] ? (AirfoilLBMCase.lbm_io(profile) ? "DOMAIN CHECK ONLY: the solver's build-only mode builds its domain on the CPU (no GPU, no march)." :
                                                            "DOMAIN CHECK ONLY: the solver builds its domain on the CPU and is stopped (no GPU, no march).") :
        s["cpu_run"] ? "CPU RUN (a test): the solver marches on its CPU backend, without the GPU lock." : "It queues for the shared GPU lock."))
    julia = joinpath(Sys.BINDIR, Base.julia_exename())
    cmd = Cmd(`$julia --startup-file=no $(lbm_worker_path()) $(joinpath(dir, "input.toml"))`;
              env=filter(kv -> !startswith(first(kv), "LUDWIG_"), copy(ENV)))
    t0 = time()
    ok = _lbm_log_process(cmd, dir, progress, log, Float64(alpha), s["steps"])
    rf = joinpath(dir, "result.toml")
    isfile(rf) || throw(ErrorException("LBM worker failed before writing its result; see worker.log and solver.log"))
    r = TOML.parsefile(rf)
    (ok && r["outcome"] in ("completed", "wall_cap", "domain_check")) ||
        throw(ErrorException("LBM run failed: $(get(r, "message", "see solver.log"))"))
    r["outcome"] == "domain_check" && return _collect_lbm_domain_check(dir, Float64(alpha), b, r, time() - t0)
    _collect_lbm_output(dir, Float64(alpha), b, r, time() - t0)
end

"""The solver's mid-span obstacle plane (num/mask.u8, "lbmio") against the app's replica mask window (lbm_geometry.jl
obstacle_window): compared cell by cell when the plane is the single level's own cell grid; also the solver's solid
cells outside the window. Returns nothing without a plane."""
function _lbm_mask_compare(dir, grid, b)
    f = joinpath(dir, "num", "mask.u8")
    (grid isa AbstractDict && isfile(f)) || return nothing
    n1 = Int(grid["n1"]); n2 = Int(grid["n2"])
    m = read(f)
    length(m) == n1*n2 || return Dict{String,Any}("compared"=>false, "reason"=>"the mask file has $(length(m)) bytes, not $(n1*n2)")
    m = reshape(m, n1, n2)
    lat = b.lattice
    org = Float64.(grid["origin_m"]); h = Float64(grid["spacing_m"])
    io_ = org ./ lat.dx; ioff = round.(Int, io_)
    w = b.payload["mask"]; i0 = w["i0"]; k0 = w["k0"]; ni = w["ni"]; nk = w["nk"]
    # the header's spacing and origin come from the solver's Float32 level spacing: equal to ~1e-7 relative
    same = abs(h - lat.dx) <= 1e-6*lat.dx && maximum(abs.(io_ .- ioff)) < 1e-3 &&
           ioff[1] <= i0 - 1 && ioff[2] <= k0 - 1 && ioff[1] + n1 >= i0 + ni - 1 && ioff[2] + n2 >= k0 + nk - 1
    same || return Dict{String,Any}("compared"=>false, "reason"=>@sprintf("the solver's plane (%d x %d, spacing %.6g, origin %s) does not hold the app's mask window on the finest cell grid (dx %.6g)", n1, n2, h, org, lat.dx))
    lv = joinpath(dir, "num", "level.u8")
    levels = isfile(lv) && filesize(lv) == n1*n2 ? reshape(read(lv), n1, n2) : fill(UInt8(lat.levels), n1, n2)
    app = reshape(Bool.(w["cells"]), ni, nk)
    solver_win = falses(ni, nk); diff = 0; coarse = 0
    for k in 1:nk, i in 1:ni
        a1 = i0 + i - 1 - ioff[1]; a2 = k0 + k - 1 - ioff[2]
        sv = m[a1, a2] == 0x01
        solver_win[i, k] = sv
        levels[a1, a2] == lat.levels || (coarse += 1)
        diff += sv != app[i, k]
    end
    total = count(==(0x01), m); inside = count(solver_win)
    only_solver = count(solver_win .& .!app); only_app = count(app .& .!solver_win)
    Dict{String,Any}("compared"=>true, "matches"=>diff == 0 && total == inside, "differing_cells"=>diff + (total - inside),
        "solver_only_cells"=>only_solver + (total - inside), "app_only_cells"=>only_app, "solver_solid"=>total,
        "app_solid"=>count(app), "solver_solid_outside_window"=>total - inside, "window"=>[i0, k0, ni, nk],
        "solver_cells"=>Int.(vec(solver_win)), "plane_y_m"=>get(grid, "normal_coord_m", nothing),
        "window_cells_not_finest"=>coarse, "plane_origin_cells"=>ioff,
        "rule"=>"the solver's obstacle flags on its mid-span XZ plane (lbm-io-opts) against the app's replica of its voxelization")
end

"""The lattice of the solver's own records against the app's replica: the plane header's coarse grid and mesh offset
("lbmio") or amr_viz/amr_data.json ("lbmapp"); returns (checks, offset difference, match)."""
function _lbm_lattice_checks(sl::AbstractDict, b)
    lat = b.lattice
    cg = get(sl, "coarse_grid_cells", nothing)
    checks = cg isa AbstractVector && length(cg) == 3 ?
        Dict{String,Any}("coarse_nx"=>(cg[1], lat.ncx), "coarse_ny"=>(cg[2], lat.ncy), "coarse_nz"=>(cg[3], lat.ncz)) :
        Dict{String,Any}("domain_nx"=>(get(sl, "domain_nx", -1), lat.nx), "domain_ny"=>(get(sl, "domain_ny", -1), lat.ny), "domain_nz"=>(get(sl, "domain_nz", -1), lat.nz))
    # the level count the solver ran (it may clamp num_levels by the domain, with a warning only)
    haskey(sl, "num_levels") && (checks["num_levels"] = (sl["num_levels"], lat.levels))
    off = get(sl, "mesh_offset_plane", get(sl, "mesh_offset", Float64[]))
    offset_err = off isa AbstractVector && length(off) == 3 ? maximum(abs.(Float64.(off) .- collect(lat.offset))) : NaN
    checks, offset_err, all(v[1] == v[2] for v in values(checks)) && isfinite(offset_err) && offset_err < 1e-6
end

"""The point of a domain check (API-only, the app's tests): the solver's own domain record ("lbmio": its build-only
summary and step-0 mid-span obstacle plane; "lbmapp": amr_data.json and the step-0 slice pictures) against the app's
lattice replica and mask; no flow, no coefficients."""
function _collect_lbm_domain_check(dir, alpha, b, r, elapsed)
    sl = Dict{String,Any}(string(k)=>v for (k, v) in get(r, "lattice_solver", Dict()))
    checks, offset_err, match_ = _lbm_lattice_checks(sl, b)
    numeric = get(r, "numeric", Dict())
    grid = get(numeric, "grid", nothing)
    mask = _lbm_mask_compare(dir, grid, b)
    build = get(r, "build_only", Dict())
    io = !isempty(build)
    ny = b.lattice.ny
    obstacle3d = nothing
    if io && !isempty(get(build, "levels", Any[]))
        lv = build["levels"][end]          # the finest level (the only one on a single-level case)
        solid_plane = mask === nothing ? nothing : get(mask, "solver_solid", nothing)
        obstacle3d = Dict{String,Any}("obstacle_cells"=>lv["obstacle_cells"], "plane_solid_x_span"=>solid_plane === nothing ? nothing : solid_plane*ny,
            "uniform_along_span"=>solid_plane !== nothing && lv["obstacle_cells"] == solid_plane*ny,
            "bouzidi_boundary_cells"=>lv["bouzidi_boundary_cells"], "cells"=>lv["cells"], "bytes_allocated"=>lv["bytes_allocated"])
    end
    mask_ok = mask === nothing || get(mask, "matches", false) === true
    logtxt = isfile(joinpath(dir, "solver.log")) ? read(joinpath(dir, "solver.log"), String) : ""
    standoff = [String(m.match) for m in eachmatch(r"\[Standoff\][^\n]*farfield wall buffer[^\n]*", logtxt)]
    floors = [String(m.match) for m in eachmatch(r"\[Levels\] level \d+: physical tau[^\n]*floored[^\n]*", logtxt)]
    lbm = merge(Dict{String,Any}(b.payload), Dict{String,Any}("solver_lattice"=>sl, "lattice_matches_solver"=>match_,
        "offset_max_difference"=>offset_err, "case_dir"=>r["case_dir"], "mask_check"=>mask, "build_only"=>build,
        "obstacle_check"=>obstacle3d,
        "startup_slices"=>Dict{String,Any}(k=>joinpath(dir, v) for (k, v) in get(r, "startup_slices", Dict()))))
    lbm["stl"]["file"] = joinpath(dir, "lbm_section.stl")
    masktxt = mask === nothing ? "" : get(mask, "compared", false) === true ?
        (mask["matches"] ? @sprintf(" Its mid-span obstacle mask equals the app's replica cell by cell (%d solid cells).", mask["solver_solid"]) :
                           @sprintf(" Its mid-span obstacle mask DIFFERS from the app's replica in %d cells (%d solver-only, %d app-only).", mask["differing_cells"], mask["solver_only_cells"], mask["app_only_cells"])) :
        " The mid-span masks were not compared: $(mask["reason"])."
    spantxt = obstacle3d === nothing ? "" : obstacle3d["uniform_along_span"] === true ?
        @sprintf(" Its finest level's 3D obstacle cells (%d) are the plane's %d solid cells times the %d span cells: the voxelized body is uniform along the span.", obstacle3d["obstacle_cells"], mask["solver_solid"], ny) :
        @sprintf(" Its 3D obstacle cells (%d) are NOT the plane's solid cells times the span (%s).", obstacle3d["obstacle_cells"], string(obstacle3d["plane_solid_x_span"]))
    Dict{String,Any}("backend"=>"lbm", "alpha_deg"=>alpha, "status"=>"domain_check", "cl"=>nothing, "cd"=>nothing, "cm"=>nothing,
        "convergence"=>(match_ ? @sprintf("Domain check: the solver's own domain (%d x %d x %d cells, mesh offset within %.1e) matches the app's lattice.", b.lattice.nx, b.lattice.ny, b.lattice.nz, offset_err) :
                                 "Domain check: the solver's domain DIFFERS from the app's lattice: $(checks), offset difference $(offset_err).") * masktxt * spantxt,
        "validation"=>"domain check only (no flow)", "fields"=>Dict{String,Any}("kind"=>io ? "lbm_numeric" : "lbm_images"), "lbm"=>lbm,
        "standoff_warnings"=>standoff, "level_floors"=>floors,
        "notes"=>vcat(b.notes, [io ? "DOMAIN CHECK: the solver's build-only mode built its domain on the CPU and stopped before any time step; not a flow result." :
                                     "DOMAIN CHECK: the solver built its domain on the CPU and was stopped before marching; not a flow result."],
                      isempty(standoff) ? String[] : ["DOMAIN CHECK FAILED: a refined level reaches the solver's farfield wall buffer: " * first(standoff)],
                      ["Solver: " * f for f in floors]),
        "iteration_history"=>Any[], "cp"=>Dict("upper"=>Any[], "lower"=>Any[]),
        "artifacts"=>Dict{String,Any}("log"=>joinpath(dir, "solver.log"), "config"=>joinpath(dir, "config.yaml"), "stl"=>joinpath(dir, "lbm_section.stl")),
        "diagnostics"=>Dict{String,Any}("solver_outcome"=>"domain_check", "lattice_checks"=>Dict(k=>collect(v) for (k, v) in checks),
            "offset_max_difference"=>offset_err, "matches"=>match_ && mask_ok && isempty(standoff), "lattice_matches"=>match_, "mask_matches"=>mask_ok,
            "standoff_clear"=>isempty(standoff),
            "mask_check"=>mask === nothing ? nothing : Dict(k=>v for (k, v) in mask if k != "solver_cells"), "obstacle_check"=>obstacle3d,
            "build_only"=>build, "point_wall_seconds"=>elapsed, "case_dir"=>r["case_dir"], "profile"=>get(r, "profile", "lbmapp")))
end

"Window statistics of the force history after the ramp: the last third (at least 3 samples)."
function _lbm_force_window(hist, ramp_steps)
    post = [h for h in hist if h.step > ramp_steps]
    isempty(post) && (post = hist)
    n = length(post); m = max(min(3, n), cld(n, 3))
    w = post[end-m+1:end]
    mean_(f) = sum(f(h) for h in w)/length(w)
    halves = length(w) >= 4 ? (w[1:fld(length(w), 2)], w[fld(length(w), 2)+1:end]) : (w, w)
    hm(v, f) = sum(f(h) for h in v)/length(v)
    (cl=mean_(h -> h.cl), cd=mean_(h -> h.cd), cl_min=minimum(h.cl for h in w), cl_max=maximum(h.cl for h in w),
     cd_min=minimum(h.cd for h in w), cd_max=maximum(h.cd for h in w), samples=length(w), first_step=w[1].step, last_step=w[end].step,
     cl_halves=(hm(halves[1], h -> h.cl), hm(halves[2], h -> h.cl)), cd_halves=(hm(halves[1], h -> h.cd), hm(halves[2], h -> h.cd)))
end

function _collect_lbm_output(dir, alpha, b, r, elapsed)
    s = b.settings
    hist = NamedTuple[]
    hf = joinpath(dir, "forces_history.csv")
    for row in (isfile(hf) ? _csv_rows(hf) : Dict{String,String}[])
        st = tryparse(Int, row["step"]); cl = _float(row["cl"]); cd = _float(row["cd"])
        (st === nothing || cl === nothing || cd === nothing) && continue
        push!(hist, (step=st, ctu=something(_float(row["ctu"]), NaN), cl=cl, cd=cd, cdp=something(_float(row["cdp"]), NaN), cdv=something(_float(row["cdv"]), NaN)))
    end
    isempty(hist) && throw(ErrorException("the LBM run wrote no force samples"))
    w = _lbm_force_window(hist, s["ramp_steps"])
    rel(a, b) = abs(a - b)/max(abs(b), 1e-6)
    judged = w.samples >= 4          # fewer samples give no two halves to compare
    settled = judged && rel(w.cl_halves...) <= 0.02 && rel(w.cd_halves...) <= 0.05
    status = r["outcome"] == "completed" ? "completed" : "not_converged"
    convergence = AirfoilLBMCase.LBM_FRICTION_NOTE * " " * AirfoilLBMCase.LBM_WALL_POSITION_NOTE * " " * @sprintf("LBM run %s at step %d of %d (%.3g convective times; %s top and bottom walls). CL and CD are the means of the last %d force samples after the ramp (steps %d-%d): CL %.4f (range %.4f to %.4f), CD %.5f (range %.5f to %.5f); the two halves of that window differ by %.1f%% in CL and %.1f%% in CD, so the loads are %s. Grid and span independence are not established.",
        r["outcome"] == "completed" ? "completed" : "stopped by its wall cap (diagnostic)", get(r, "last_step", 0), s["steps"], get(r, "last_step", 0)/s["steps_per_ctu"],
        get(s, "z_walls", "slip") == "no_slip" ? "no-slip" : "slip",
        w.samples, w.first_step, w.last_step, w.cl, w.cl_min, w.cl_max, w.cd, w.cd_min, w.cd_max,
        100rel(w.cl_halves...), 100rel(w.cd_halves...), settled ? "settled to that tolerance (2% CL, 5% CD)" :
        judged ? "still changing or unsteady" : "not judged (fewer than 4 samples in the window: run longer)")
    stride = max(1, cld(length(hist), 1500))
    history = [Dict{String,Any}("iteration"=>h.step, "residual"=>(k == 1 ? 1.0 : max(abs(h.cl - hist[k-1].cl) + abs(h.cd - hist[k-1].cd), 1e-12)),
        "cl"=>h.cl, "cd"=>h.cd, "t_ctu"=>h.ctu) for (k, h) in enumerate(hist) if (k - 1) % stride == 0 || k == length(hist)]
    abspath_(rel) = joinpath(dir, rel)
    frames = Dict{String,Any}()
    for (fld, fr) in get(r, "frames", Dict())
        frames[fld] = Dict{String,Any}("steps"=>fr["steps"], "files"=>[abspath_(f) for f in fr["files"]],
                                       "ctu"=>[st/s["steps_per_ctu"] for st in fr["steps"]])
    end
    videos = Dict{String,Any}(fld=>merge(Dict{String,Any}(string(k)=>x for (k, x) in v), Dict{String,Any}("file"=>abspath_(v["file"])))
                              for (fld, v) in get(r, "videos", Dict()))   # every key of the worker's record (label, colour range, source)
    startup = Dict{String,Any}(k=>abspath_(v) for (k, v) in get(r, "startup_slices", Dict()))
    perf = Dict{String,Any}(string(k)=>v for (k, v) in get(r, "performance", Dict()))
    # the solver's MLUPS is its total cells (every level) x solver steps / time, so total cells / MLUPS is the time per
    # SOLVER step (one coarsest step; W7): one level, the lattice's cells; several,
    # the solver's own total ("[Info] Total cells", in M with two decimals) or else the app's estimate
    tcm = get(get(r, "lattice_solver", Dict()), "total_cells_millions", NaN)
    cells = b.lattice.levels == 1 ? b.lattice.cells : (tcm isa Real && isfinite(tcm) && tcm > 0 ? round(Int, tcm*1e6) : b.estimate["cells"])
    perf["cells_counted"] = cells
    ml = get(perf, "mlups_compute_median", NaN)
    if ml isa Real && isfinite(ml) && ml > 0
        perf["seconds_per_step_compute"] = cells/(ml*1e6)
        perf["seconds_per_ctu_compute"] = cells/(ml*1e6)*s["steps_per_ctu"]
    end
    mw = get(perf, "march_wall_s", NaN)
    if mw isa Real && isfinite(mw)
        perf["seconds_per_step_wall"] = mw/s["steps"]; perf["seconds_per_ctu_wall"] = mw/s["ctu"]
    end
    solver_lattice = Dict{String,Any}(string(k)=>v for (k, v) in get(r, "lattice_solver", Dict()))
    _, offset_err, match_ = _lbm_lattice_checks(solver_lattice, b)
    haskey(solver_lattice, "coarse_grid_cells") || (match_ = all(get(solver_lattice, k, -1) == v for (k, v) in (("domain_nx", b.lattice.nx), ("domain_ny", b.lattice.ny), ("domain_nz", b.lattice.nz))))
    numeric = Dict{String,Any}(string(k)=>v for (k, v) in get(r, "numeric", Dict()))
    mask = _lbm_mask_compare(dir, get(numeric, "grid", nothing), b)
    if !isempty(numeric)
        numeric["files"] = Dict{String,Any}(k=>[abspath_(f) for f in v] for (k, v) in get(numeric, "files", Dict()))
        numeric["mask"] = abspath_(numeric["mask"])
        haskey(numeric, "level") && (numeric["level"] = abspath_(numeric["level"]))
        numeric["ctu_from_steps"] = [st/s["steps_per_ctu"] for st in numeric["steps"]]
    end
    posters = Dict{String,Any}(k=>abspath_(v) for (k, v) in get(r, "posters", Dict()))
    notes = copy(b.notes)
    tc = get(numeric, "time_check", nothing)
    if tc isa AbstractDict && get(tc, "compared", false) === true
        mm = get(tc, "mismatches", Any[])
        nd = get(tc, "max_relative_difference_to_nominal", NaN)
        push!(notes, isempty(mm) ? @sprintf("The solver's own time (per solver step) equals the app's step-based time at every frame (1e-9 relative, with the solver's Float32 lattice speed; the nominal times shown differ from it by at most %.2g relative).", nd) :
              "WARNING: the solver's own time differs from the app's step-based time at $(length(mm)) frame(s), the first: $(first(mm)). The app's times come from the step count.")
    end
    push!(notes, isempty(solver_lattice) ? "The solver's own domain record (amr_viz/amr_data.json) was not found; the lattice shown is the app's replica." :
          match_ ? @sprintf("The solver's own domain record matches the app's lattice: %d x %d x %d cells.", b.lattice.nx, b.lattice.ny, b.lattice.nz) :
          "The solver's own domain record DIFFERS from the app's lattice replica: solver $(get(solver_lattice, "domain_nx", "?")) x $(get(solver_lattice, "domain_ny", "?")) x $(get(solver_lattice, "domain_nz", "?")), app $(b.lattice.nx) x $(b.lattice.ny) x $(b.lattice.nz).")
    logtxt = isfile(joinpath(dir, "solver.log")) ? read(joinpath(dir, "solver.log"), String) : ""
    for m in eachmatch(r"\[Standoff\][^\n]*farfield wall buffer[^\n]*", logtxt)
        push!(notes, "WARNING from the solver: a refined level reaches the farfield wall buffer: " * String(m.match))
    end
    for m in eachmatch(r"\[Levels\] level \d+: physical tau[^\n]*floored[^\n]*", logtxt)
        push!(notes, "Solver: " * String(m.match))
    end
    nl = get(solver_lattice, "num_levels", b.lattice.levels)
    nl == b.lattice.levels || push!(notes, "The solver ran $(nl) refinement level(s) instead of the requested $(b.lattice.levels): it reduces num_levels by the domain size with a warning only (see solver.log, \"num_levels clamped\"). The lattice shown is the requested one.")
    mask === nothing || push!(notes, get(mask, "compared", false) === true ?
        (mask["matches"] ? @sprintf("The solver's own mid-span obstacle mask equals the app's replica cell by cell (%d solid cells).", mask["solver_solid"]) :
                           @sprintf("The solver's mid-span obstacle mask DIFFERS from the app's replica in %d cells.", mask["differing_cells"])) :
        "The mid-span masks were not compared: $(mask["reason"]).")
    get(r, "domain_check", false) && push!(notes, "DOMAIN CHECK: the solver built its domain on the CPU and was stopped before marching; not a flow result.")
    get(r, "cpu_run", false) === true && push!(notes, "CPU RUN (a test of the app's pipeline): the solver marched on its CPU backend; the timings are not those of the GPU.")
    push!(notes, isempty(numeric) ? "Solver case: $(r["case_dir"]) (pinned LBM snapshot, commit $(first(String(r["commit"]), 8))); the solver's full-size slice PNGs for every frame are kept there." :
        "Solver case: $(r["case_dir"]) (pinned LBM snapshot, commit $(first(String(r["commit"]), 8)), profile $(get(r, "profile", "?"))); the solver's numeric mid-plane files (.lbmslice, Float32, every frame) are kept there.")
    lbm = merge(Dict{String,Any}(b.payload), Dict{String,Any}("frames"=>frames, "videos"=>videos, "startup_slices"=>startup,
        "numeric"=>numeric, "posters"=>posters, "mask_check"=>mask, "profile"=>get(r, "profile", "lbmapp"),
        "performance"=>perf, "solver_lattice"=>solver_lattice, "lattice_matches_solver"=>match_, "case_dir"=>r["case_dir"],
        "solver_summary"=>get(r, "solver_summary", Dict()), "window"=>Dict(string(k)=>v for (k, v) in pairs(w)), "settled"=>settled,
        "config"=>joinpath(dir, "config.yaml")))
    lbm["stl"]["file"] = joinpath(dir, "lbm_section.stl")
    Dict{String,Any}("backend"=>"lbm", "alpha_deg"=>alpha, "status"=>status, "convergence"=>convergence,
        "validation"=>"LBM 3D (LUDWIG) development path, not validated for airfoils. " * AirfoilLBMCase.LBM_FRICTION_NOTE * " " * AirfoilLBMCase.LBM_WALL_POSITION_NOTE,
        "cl"=>w.cl, "cd"=>w.cd, "cm"=>nothing, "iteration_history"=>history, "cp"=>Dict("upper"=>Any[], "lower"=>Any[]),
        "fields"=>Dict{String,Any}("kind"=>isempty(numeric) ? "lbm_images" : "lbm_numeric"), "notes"=>notes, "lbm"=>lbm,
        "geometry"=>Dict("support"=>b.payload["support"], "points"=>length(b.payload["section"]["x"])),
        "artifacts"=>Dict{String,Any}("log"=>joinpath(dir, "solver.log"), "worker_log"=>joinpath(dir, "worker.log"),
            "config"=>joinpath(dir, "config.yaml"), "stl"=>joinpath(dir, "lbm_section.stl"), "forces_csv"=>hf,
            "result_toml"=>joinpath(dir, "result.toml"), "input_toml"=>joinpath(dir, "input.toml"),
            "videos"=>Dict(k=>v["file"] for (k, v) in videos)),
        "diagnostics"=>Dict{String,Any}("solver"=>"LUDWIG LBM (D3Q27 cumulant, single level, periodic span)", "commit"=>r["commit"],
            "case_dir"=>r["case_dir"], "solver_outcome"=>r["outcome"], "steps"=>get(r, "last_step", 0), "max_steps"=>s["steps"],
            "ctu"=>get(r, "last_step", 0)/s["steps_per_ctu"], "cells"=>cells, "lattice"=>[b.lattice.nx, b.lattice.ny, b.lattice.nz],
            "cells_per_chord"=>s["cells_per_chord"], "span_cells"=>s["span_cells"], "reynolds"=>s["reynolds"],
            "tau"=>s["tau"], "tau_used"=>s["tau_used"], "lattice_mach"=>s["lattice_mach"], "force_samples"=>length(hist),
            "window_samples"=>w.samples, "settled"=>settled, "point_wall_seconds"=>elapsed, "performance"=>perf,
            "estimate"=>b.estimate, "time_mode"=>"lbm",
            "clock"=>b.lattice.levels == 1 ? "LBM time steps (one step = dx/u_lattice in lattice units)" :
                     "LBM solver steps: one solver step is one coarsest-level step = $(2^(b.lattice.levels - 1)) finest steps; times from the step count (the solver's printed time is 2^(L-1) x too small, W7)",
            "friction_drag_note"=>AirfoilLBMCase.LBM_FRICTION_NOTE, "wall_position_note"=>AirfoilLBMCase.LBM_WALL_POSITION_NOTE, "wall_model_label"=>AirfoilLBMCase.LBM_WALL_MODEL_LABEL,
            "profile"=>get(r, "profile", "lbmapp"), "cpu_run"=>get(r, "cpu_run", false), "z_walls"=>get(s, "z_walls", "slip"), "levels"=>s["levels"], "level_viscosity"=>get(s, "level_viscosity", "legacy"), "mask_check"=>mask === nothing ? nothing : Dict(k=>v for (k, v) in mask if k != "solver_cells"),
            "offset_max_difference"=>offset_err,
            "convergence_guard"=>Dict("status"=>settled ? "lbm_window_settled" : "lbm_window_unsettled", "reason"=>convergence)))
end
