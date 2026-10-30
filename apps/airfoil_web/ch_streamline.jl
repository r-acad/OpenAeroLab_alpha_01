# =====================================================================================================================
# ch_streamline.jl -- streamline alignment of the compressible C+H mesh (DEVELOPMENT, NOT VALIDATED; 24 Oct 2026).
# Included inside CFDBackend after streamline_backend.jl, in the app server (Create mesh) and in the C+H worker (Run): both
# call `_ch_streamline_apply` on the mesh `_build_ch_mesh` returned, with the same hashed path file, so they build the same
# nodes. Absent `streamline` (the default), nothing here runs and the mesh is bitwise the previous one.
# The Gurney plate on the compressible path is not aligned (its Run is refused); see streamline_check_request.
# Design: STREAMLINE_ALIGNMENT_DESIGN.md, section 7.1.
# =====================================================================================================================
include(joinpath(@__DIR__, "mesh_streamline_morph.jl"))     # module MeshStreamlineMorph (Base + SHA + TOML)

"The C+H mesh's own quality report, as `_build_ch_mesh` admits it (strip post-checks and the air gates)."
function _ch_streamline_mesh(m, X, Y, Xs, Ys)
    s = m.strip
    CMeshGen.ChannelMesh(m.NI, m.NJ, m.NIW, m.NIF, X, Y, m.dn1, m.growth, m.channel, m.outer_face_kind,
                         CMeshGen.StripBlock(s.NI, s.NJ, s.n_base, Xs, Ys, s.t, s.dn1_i))
end

"""Align a built C+H mesh (`(;mesh, metadata)` of `_build_ch_mesh`) with the JXfoil paths of the worker table `st`
(streamline_worker_cfg). Returns `(;mesh, metadata, streamlines)`: the morphed mesh, the metadata with the air and
cavitating gates measured again on it and the alignment record, and the overlay for the mesh payload."""
function _ch_streamline_apply(built, st::AbstractDict)
    MSM = MeshStreamlineMorph; G = CMeshGen; Q = G.CMeshQualityMetrics
    m = built.mesh; m.strip === nothing && throw(ArgumentError("streamline alignment needs the C+H mesh (finite trailing edge)"))
    paths = MSM.read_streamline_paths(String(st["paths"]); expected=String(st["paths_sha256"]))
    params = MSM.params_from(get(st, "params", Dict{String,Any}()))
    function quality(X, Y, Xs, Ys)
        r = try
            G._paired_strip_quality_report(_ch_streamline_mesh(m, X, Y, Xs, Ys); check_air_gates=false)
        catch err
            return Dict{String,Any}("postcheck_error"=>sprint(showerror, err), "gates"=>Any[])
        end
        Dict{String,Any}("gates"=>[Dict{String,Any}("gate"=>g[1], "value"=>g[3], "limit"=>g[4], "pass"=>g[5]) for g in r.gates])
    end
    X2, Y2, Xs2, Ys2, rep = MSM.admit_morph(m.X, m.Y, m.NIW, m.NIF, m.strip.X, m.strip.Y, paths; quality, params,
                                            rule=Symbol(String(st["admission"])))
    m2 = _ch_streamline_mesh(m, X2, Y2, Xs2, Ys2)
    md = deepcopy(built.metadata)
    # the gates measured again on the aligned nodes, exactly as _build_ch_mesh records them
    measured = G._paired_strip_quality_report(m2; check_air_gates=false); s = measured.summary
    cavity = Q.quality_gates(s; cavitating=true, strip_grade=true)
    md["quality_passed"] = measured.passed
    md["quality_status"] = measured.passed ? "passed_air_geometry" : "refused_air_geometry"
    md["quality_failures"] = measured.failed
    md["cavitation_quality_passed"] = all(r[5] for r in cavity); md["cavitation_quality_failures"] = Q.gate_failures(cavity)
    md["maximum_face_nonorthogonality_deg"] = s.theta_max
    haskey(md, "refinement") && (md["refinement"]["air_gates_passed"] = measured.passed;
        md["refinement"]["air_gate_failures"] = measured.failed; md["refinement"]["admitted"] = measured.passed)
    md["quality_report"] = Dict("air_reference_gates"=>Dict("passed"=>measured.passed, "failed"=>measured.failed),
        "geometric_gate"=>Dict("passed"=>true), "cells"=>s.cells,
        "global"=>Dict("maximum_face_nonorthogonality_deg"=>s.theta_max, "minimum_corner_angle_deg"=>s.minang,
            "maximum_aspect_ratio"=>s.ar_max, "maximum_neighbor_volume_ratio"=>s.vr_max),
        "interfaces"=>Dict("maximum_volume_ratio"=>s.vr_junction,
            "maximum_nonorthogonality_deg"=>s.theta_junction, "maximum_skew_over_distance"=>s.skew_junction))
    md["mesh_family"] = String(md["mesh_family"]) * " + streamline-aligned (JXfoil, development)"
    md["streamline_alignment"] = Dict{String,Any}("aligned"=>true, "tag"=>rep["tag"], "path_source"=>paths["source"],
        "path_label"=>paths["source_label"], "path_sha256"=>paths["sha256"], "alpha_deg"=>paths["alpha_deg"],
        "reynolds"=>paths["reynolds"], "fallback_reason"=>get(paths, "fallback_reason", ""),
        "cl_viscous"=>get(get(paths, "viscous", Dict()), "cl", nothing), "cl_inviscid"=>paths["cl_inviscid"], "cl_field"=>paths["cl_field"],
        "section_note"=>String(get(st, "section_note", "")), "admission"=>rep["admission"], "params"=>rep["params_used"],
        "te"=>rep["te"], "le"=>rep["le"], "source_nodes_sha256"=>rep["source_sha256"], "morphed_nodes_sha256"=>rep["morphed_sha256"],
        "foil_first_spacing_max_rel_change"=>rep["foil_first_spacing_max_rel_change"],
        "wake_first_spacing_max_rel_change"=>rep["wake_first_spacing_max_rel_change"],
        "air_gates_before"=>built.metadata["quality_failures"], "air_gates_after"=>measured.failed)
    # the overlay: the wake path the mesh follows (8 c of arc) and the stagnation streamline (3 c), at most 800 points each
    w = MSM.aligned_wake_path(m.X, m.Y, m.NIW, m.NIF, m.strip.X, m.strip.Y, paths["wake"]; params)
    thin(x, y, sv, L) = (k = findlast(<=(L), sv); idx = unique(vcat(collect(1:max(1, cld(k, 800)):k), k)); Dict("x"=>Float64.(x[idx]), "y"=>Float64.(y[idx])))
    sw = zeros(length(w.x)); for q in 2:length(w.x); sw[q] = sw[q-1] + hypot(w.x[q] - w.x[q-1], w.y[q] - w.y[q-1]); end
    stg = paths["stagnation_streamline"]
    overlay = Dict{String,Any}("wake"=>thin(w.x, w.y, sw, 8.0), "stagnation"=>thin(stg["x"], stg["y"], stg["s"], 3.0),
        "stagnation_point"=>[Float64(paths["stagnation"]["x"]), Float64(paths["stagnation"]["y"])],
        "source"=>paths["source"], "label"=>paths["source_label"], "alpha_deg"=>paths["alpha_deg"],
        "admission"=>rep["admission"]["rule"], "note"=>String(get(st, "section_note", "")))
    adm = rep["admission"]
    line = @sprintf("STREAMLINE ALIGNMENT: %s path (sha256 %s), %s admission: %s, TE amplitude %.3f, LE amplitude %.3f, %d attempts; gates worse: %s; nodes sha256 %s",
        String(paths["source"]), String(paths["sha256"])[1:12], adm["rule"], adm["stage"], adm["te_amplitude"], adm["le_amplitude"],
        length(adm["attempts"]), isempty(adm["gates_worse"]) ? "none" : join(adm["gates_worse"], ", "), String(rep["morphed_sha256"])[1:12])
    (; mesh=m2, metadata=md, streamlines=overlay, line)
end
