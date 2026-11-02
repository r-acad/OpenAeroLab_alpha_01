# Airfoil Web's external-flow use of the cavitation C-grid construction.
# Include inside CFDBackend after CMeshGen and CMeshMetrics are loaded. The
# geometry is generated with the channel branch because that is where the
# A-plus finite trailing-edge base and the curved corner columns are defined.
# The gas solve remains an external-flow solve: the raw-array metrics overload
# deliberately tags *every* outer face as characteristic FARFIELD.

include(joinpath(@__DIR__, "nested_refined_mesh.jl"))
include(joinpath(@__DIR__, "mesh_volume_redistribution.jl"))
include(joinpath(@__DIR__, "normal_collar.jl"))

"Sharp C mesh with locally concentrated surface columns around the leading edge."
function _sharp_nose_mesh(af::CMeshGen.AirfoilPts; n_foil::Int, n_wake::Int,
                          n_normal::Int, dn1::Float64, le_ds_scale::Float64=1.0)
    0 < le_ds_scale <= 1 || throw(ArgumentError("le_ds_scale must lie in (0, 1]"))
    kappa=CMeshGen.airfoil_curvature(af)
    xmin=minimum(af.x)
    le_kappa=maximum(abs(kappa[i]) for i in eachindex(kappa) if af.x[i] <= xmin+0.02)
    le_ds=clamp(0.03le_ds_scale/max(le_kappa,1.0),4dn1,0.0006)
    # A weak continuous curvature monitor plus a measured nose interval adds
    # columns at the nose without leaving an unresolved gap at mid-chord.
    mesh=CMeshGen.generate_cmesh(af;n_foil,n_wake,n_normal,dn1,
        growth_max=1.25,R_ff=30.0,L_wake=25.0,smooth_iters=60,bl_freeze=12,
        surface_evaluator=:natural_c2,surface_distribution=:curvature,
        surface_curvature_weight=0.001,le_ds)
    (;mesh,le_ds,le_kappa)
end

"Project the refined foil onto the same C2 contour used by geometry preview."
function _transonic_curve_wall!(mesh, curve, n_base::Int)
    # In a C+H layout the entire base belongs to H, so C.NIF counts only
    # airfoil surface cells. Keep both interface corner nodes fixed.
    c_base=hasproperty(mesh,:strip) && mesh.strip !== nothing ? 0 : n_base
    first_surface=mesh.NIW+c_base+1
    last_surface=mesh.NIW+mesh.NIF-c_base+1
    max_shift=0.0
    for i in first_surface:last_surface
        # The end points and real, straight finite base are exact constraints.
        i in (first_surface,last_surface) && continue
        p=CMeshGen._airfoil_closest(curve,mesh.X[i,1],mesh.Y[i,1])
        dx=p.x-mesh.X[i,1]; dy=p.y-mesh.Y[i,1]
        max_shift=max(max_shift,hypot(dx,dy))
        x0,y0=mesh.X[i,1],mesh.Y[i,1]
        for j in 1:mesh.NJ+1
            distance=hypot(mesh.X[i,j]-x0,mesh.Y[i,j]-y0)
            # All viscous layers move together. Fade at a physical distance,
            # rather than over a fixed number of tiny high-Re wall layers.
            t=clamp(distance/0.15,0.0,1.0)
            weight=1-t^3*(10-15t+6t^2)
            mesh.X[i,j]+=weight*dx
            mesh.Y[i,j]+=weight*dy
        end
        mesh.X[i,1]=p.x; mesh.Y[i,1]=p.y
    end
    max_shift
end

"Remove abrupt tangential cell-width jumps in the finite trailing-edge collar."
function _transonic_te_redistribute!(mesh; sweeps::Int=80)
    X,Y=mesh.X,mesh.Y
    maximum_shift=0.0
    for _ in 1:sweeps
        for parity in 0:1, j in 14:mesh.NJ, i in 2+parity:2:mesh.NI
            # Keep the resolved viscous wall, leading edge, wake cut and outer
            # contour fixed. The collar's blend turns off smoothly in space.
            abs(X[i,j]-1.0) < 0.15 && abs(Y[i,j]) < 0.10 || continue
            a=hypot(X[i,j]-X[i-1,j],Y[i,j]-Y[i-1,j])
            b=hypot(X[i+1,j]-X[i,j],Y[i+1,j]-Y[i,j])
            max(a,b) > 1.55min(a,b) || continue
            tx=X[i+1,j]-X[i-1,j]; ty=Y[i+1,j]-Y[i-1,j]
            tangent=hypot(tx,ty)
            tangent>0 || continue
            wx=cospi(0.5*(X[i,j]-1.0)/0.15)^2
            wy=cospi(0.5*Y[i,j]/0.10)^2
            wj=min(1.0,(j-13)/12)
            shift=clamp(0.15*(b-a),-0.05min(a,b),0.05min(a,b))*wx*wy*wj
            X[i,j]+=shift*tx/tangent
            Y[i,j]+=shift*ty/tangent
            maximum_shift=max(maximum_shift,abs(shift))
        end
    end
    maximum_shift
end

"""Measured quadrilateral and face quality, including the first 12 foil rows.

These are geometric checks and diagnostics, not evidence of grid independence.
Face nonorthogonality and skew use the same definitions as CMeshQualityV1.
"""
function _transonic_quality_report(mesh)
    hasproperty(mesh,:strip) && mesh.strip !== nothing && throw(ArgumentError(
        "the single-block transonic quality report cannot assess a C+H mesh; use the two-block face metrics including both interfaces and the base"))
    Q = CMeshGen.CMeshQualityMetrics
    corner_defects, worst_corner_jacobian, worst_corner_location =
        CMeshGen._convexity_defects(mesh.X, mesh.Y, mesh.NI, 1:mesh.NJ)
    corner_defects == 0 && worst_corner_jacobian > 0 ||
        error("transonic mesh has $corner_defects non-convex cells; worst corner Jacobian " *
              "$worst_corner_jacobian at $worst_corner_location")
    cells = Q.block_cells(mesh.X, mesh.Y)
    faces, boundary = Q.mesh_faces([(mesh.X, mesh.Y)], [cells];
        NIW=mesh.NIW, NIF=mesh.NIF, strip=false)
    summary = Q.quality_summary(mesh.X, mesh.Y, mesh.NIW, mesh.NIF;
        band_j=min(40,mesh.NJ), nb=min(3,mesh.NJ))
    foil_range = mesh.NIW+1:mesh.NIW+mesh.NIF
    near_j = 1:min(12,mesh.NJ)
    near(i,j) = i in foil_range && j in near_j
    near_ar = vec(cells.ar_p[foil_range, near_j])
    near_angle = vec(cells.minang[foil_range, near_j])
    near_jratio = vec(cells.jratio[foil_range, near_j])
    near_faces = [f for f in faces if
        (f.Pb == 1 && near(f.Pi,f.Pj)) ||
        (f.Nb == 1 && near(f.Ni,f.Nj))]
    near_theta = [f.theta for f in near_faces]
    near_skew = [f.skew for f in near_faces]
    wake_cell_range = vcat(collect(1:mesh.NIW),
        collect(mesh.NIW+mesh.NIF+1:mesh.NI))
    wake_first_ar = cells.ar_p[wake_cell_range,1]
    wake_faces = [f for f in faces if f.kind === :cut ||
        (f.Pj <= 2 && f.Pi in wake_cell_range) ||
        (f.Nj <= 2 && f.Ni in wake_cell_range)]
    cut_faces = [f for f in faces if f.kind === :cut]
    cut_skew_max = maximum(f.skew for f in cut_faces)
    wake_ar_max = maximum(wake_first_ar)
    wall_theta = [b[5] for b in boundary if b[1] === :foil]
    percentile(v,q) = (s = sort!(collect(Float64,v)); s[max(1,ceil(Int,q*length(s)))])
    worst_skew_face = near_faces[argmax(near_skew)]
    worst_angle_face = near_faces[argmax(near_theta)]
    global_worst_angle_face = faces[argmax([f.theta for f in faces])]
    global_worst_skew_face = faces[argmax([f.skew for f in faces])]
    global_worst_volume_face = faces[argmax([f.vr for f in faces])]
    solver_worst_angle_face = faces[argmax([f.theta_v for f in faces])]
    solver_worst_skew_face = faces[argmax([f.skew_v for f in faces])]
    face_location(f) = Dict("kind"=>String(f.kind),
        "owner_i"=>f.Pi, "owner_j"=>f.Pj,
        "neighbor_i"=>f.Ni, "neighbor_j"=>f.Nj,
        "x"=>f.xf, "y"=>f.yf)
    air_gates = Q.quality_gates(summary; cavitating=false)
    report = Dict{String,Any}(
        "definition"=>"CMeshQualityV1 geometric quadrilateral and finite-volume face metrics; near foil means first 12 wall-normal cell rows, including trailing-edge base",
        "cells"=>summary.cells,
        "nonconvex_cells"=>corner_defects,
        "minimum_corner_jacobian"=>worst_corner_jacobian,
        "minimum_corner_jacobian_cell"=>[worst_corner_location[1],worst_corner_location[2]],
        "geometric_gate"=>Dict("passed"=>true,
            "criteria"=>"strictly positive corner Jacobians, convex cells, finite metrics and positive area; wake-cut coincidence is checked separately"),
        "air_reference_gates"=>Dict(
            "passed"=>all(row[5] for row in air_gates),
            "failed"=>Q.gate_failures(air_gates),
            "applicability"=>"comparative CMeshQualityV1 air thresholds; this A-plus external-flow topology does not implement that preset's mesh construction"),
        "provisional_wake_gates"=>Dict(
            "passed"=>cut_skew_max <= 4.0 && wake_ar_max <= 1500.0,
            "maximum_true_centroid_cut_skew_over_distance"=>cut_skew_max,
            "maximum_first_row_wake_aspect_ratio"=>wake_ar_max,
            "cut_skew_limit"=>4.0,
            "wake_aspect_ratio_limit"=>1500.0,
            "scope"=>"provisional geometric screening limits; not CFD grid convergence or experimental validation"),
        "global"=>Dict(
            "minimum_corner_angle_deg"=>summary.minang,
            "minimum_scaled_corner_jacobian"=>summary.sj_min,
            "maximum_aspect_ratio"=>summary.ar_max,
            "maximum_face_nonorthogonality_deg"=>summary.theta_max,
            "worst_nonorthogonality_face"=>face_location(global_worst_angle_face),
            "median_face_nonorthogonality_deg"=>summary.theta_p50,
            "maximum_face_skew_over_centroid_distance"=>summary.skew_max,
            "worst_skew_face"=>face_location(global_worst_skew_face),
            "maximum_solver_center_nonorthogonality_deg"=>solver_worst_angle_face.theta_v,
            "worst_solver_center_nonorthogonality_face"=>face_location(solver_worst_angle_face),
            "maximum_solver_center_skew_over_centroid_distance"=>solver_worst_skew_face.skew_v,
            "worst_solver_center_skew_face"=>face_location(solver_worst_skew_face),
            "maximum_openfoam_normalized_skew"=>summary.skew_of_max,
            "maximum_neighbor_volume_ratio"=>summary.vr_max,
            "worst_neighbor_volume_ratio_face"=>face_location(global_worst_volume_face),
            "maximum_wall_normal_growth_first_40"=>summary.rn_band,
            "maximum_wall_normal_growth_elsewhere"=>summary.rn_else,
            "maximum_foil_tangential_ratio"=>summary.rt_foil),
        "near_foil"=>Dict(
            "cells"=>length(near_ar),
            "aspect_ratio_median"=>percentile(near_ar,0.5),
            "aspect_ratio_p95"=>percentile(near_ar,0.95),
            "aspect_ratio_max"=>maximum(near_ar),
            "minimum_corner_angle_deg"=>minimum(near_angle),
            "minimum_corner_jacobian_ratio"=>minimum(near_jratio),
            "face_nonorthogonality_median_deg"=>percentile(near_theta,0.5),
            "face_nonorthogonality_p95_deg"=>percentile(near_theta,0.95),
            "face_nonorthogonality_max_deg"=>maximum(near_theta),
            "worst_nonorthogonality_face"=>face_location(worst_angle_face),
            "face_skew_median"=>percentile(near_skew,0.5),
            "face_skew_p95"=>percentile(near_skew,0.95),
            "face_skew_max"=>maximum(near_skew),
            "worst_skew_face"=>face_location(worst_skew_face),
            "wall_face_nonorthogonality_p95_deg"=>percentile(wall_theta,0.95),
            "wall_face_nonorthogonality_max_deg"=>maximum(wall_theta)),
        "wake_first_rows"=>Dict(
            "definition"=>"both wake legs, first two wall-normal rows; includes the paired wake cut",
            "first_row_aspect_ratio_p95"=>percentile(wake_first_ar,0.95),
            "first_row_aspect_ratio_max"=>maximum(wake_first_ar),
            "geometric_center_nonorthogonality_max_deg"=>maximum(f.theta for f in wake_faces),
            "solver_center_nonorthogonality_max_deg"=>maximum(f.theta_v for f in wake_faces),
            "geometric_center_skew_max"=>maximum(f.skew for f in wake_faces),
            "solver_center_skew_max"=>maximum(f.skew_v for f in wake_faces),
            "cut_geometric_center_skew_max"=>cut_skew_max,
            "cut_solver_center_skew_max"=>maximum(f.skew_v for f in cut_faces),
            "cut_geometric_center_nonorthogonality_max_deg"=>maximum(f.theta for f in cut_faces),
            "cut_solver_center_nonorthogonality_max_deg"=>maximum(f.theta_v for f in cut_faces)))
    for group in (report["global"], report["near_foil"]), (name,value) in group
        value isa Real && !isfinite(value) && error("nonfinite mesh quality metric $name")
    end
    report["global"]["minimum_corner_angle_deg"] > 0 || error("mesh has a collapsed corner angle")
    report["global"]["minimum_scaled_corner_jacobian"] > 0 || error("mesh has a nonpositive corner Jacobian")
    report
end

function _build_transonic_mesh(af::CMeshGen.AirfoilPts;
        n_foil::Int, n_wake::Int, n_normal::Int, dn1::Float64,
        alpha_deg::Float64, trip_upper::Float64, trip_lower::Float64,
        n_base::Int=4, far_extent::Float64=30.0,
        farfield_angle_deg::Union{Nothing,Float64}=nothing,
        growth_max::Float64=1.25, smooth_iters::Int=80, bl_freeze::Int=12,
        wake_blend_fraction::Float64=0.25,
        wake_first_spacing_growth::Float64=1.0,
        wake_first_spacing_max::Float64=0.0,
        wake_first_spacing_delay_cells::Int=0,
        mesh_refinement::Symbol=:none,
        nested_auto_wake_growth::Bool=true,
        condition_geometry::Bool=true,
        cfd_te_thickness::Float64=0.0,
        stall_layer::Bool=false,
        le_ds_scale::Float64=1.0, normal_collar::Bool=true)
    far_extent >= 10 || throw(ArgumentError("far_extent must be at least 10 chords"))
    n_base >= 1 || throw(ArgumentError("n_base must be positive"))
    outer_angle = isnothing(farfield_angle_deg) ? alpha_deg : farfield_angle_deg
    isfinite(outer_angle) && -90 <= outer_angle <= 90 ||
        throw(ArgumentError("farfield_angle_deg must be finite and within [-90, 90] degrees"))
    conditioned=CMeshGen.condition_airfoil(af;
        trailing_edge_thickness=cfd_te_thickness)
    source_af=af
    af=condition_geometry ? conditioned.airfoil : af
    curve=CMeshGen._airfoil_spline(af)
    curvature=condition_geometry ? conditioned.curvature : CMeshGen.airfoil_curvature(af)
    # A bounded angular turn resolves the small leading-edge radius. Use an
    # actual curvature measurement rather than the airfoil's name or thickness.
    xmin=minimum(af.x)
    le_curvature=maximum(abs(curvature[k]) for k in eachindex(curvature) if af.x[k] <= xmin+0.02)
    # le_ds_scale (21 Oct 2026, opt-in; 1 is the established rule exactly) scales the nose target spacing.
    0 < le_ds_scale <= 1 || throw(ArgumentError("le_ds_scale must lie in (0, 1]"))
    le_ds=clamp(0.06*le_ds_scale/max(le_curvature,1.0),8dn1,0.001)
    gap = hypot(af.x[1] - af.x[end], af.y[1] - af.y[end])
    blunt = gap > 1e-8
    # G8 requires each base interval to be at least dn1. A small, real base
    # must be resolved rather than silently replacing the user's geometry.
    nb = blunt ? min(n_base, max(1, floor(Int, gap/(2dn1)))) : 0
    wall_dn1 = blunt ? min(dn1, 0.9 * gap / (2nb)) : dn1
    blunt && wall_dn1 < dn1/8 && throw(ArgumentError(
        "trailing-edge gap $gap is below the resolvable scale of the requested wall spacing $dn1; " *
        "refine the CFD wall spacing or use a section with a resolved trailing edge"))
    te_ds = blunt ? 1.5 * gap / (2nb) : 0.0
    spec = CMeshGen.ChannelSpec(; height=2far_extent,
        xi_in=0.5 - far_extent, xi_out=0.5 + far_extent,
        xi_ref=0.5, ref=(0.5, 0.0), angle_deg=outer_angle)
    mesh_refinement in (:none,:aplus_nested_refined) ||
        throw(ArgumentError("mesh_refinement must be :none or :aplus_nested_refined"))
    generate(growth,blend,maxspacing,delay; nose_anchor_fillet=0.0) = CMeshGen.generate_cmesh(af;
        n_foil=n_foil, n_wake=n_wake, n_normal=n_normal,
        dn1=wall_dn1, growth_max=growth_max,
        R_ff=far_extent, L_wake=far_extent,
        smooth_iters=smooth_iters, bl_freeze=bl_freeze,
        # The finite-base ray construction needs its smooth two-ended surface
        # distribution. Resolve its nose by curvature-driven nested children,
        # without changing the base-to-surface Vinokur join.
        le_ds=condition_geometry && !blunt ? le_ds : 0.0,
        channel=spec, n_base=nb, te_ds=te_ds, te_topology=:aplus,
        wake_blend_fraction=blend,
        wake_first_spacing_growth=growth,
        wake_first_spacing_max=maxspacing,
        wake_first_spacing_delay_cells=delay,nose_anchor_fillet)
    actual_blend = mesh_refinement === :none ? wake_blend_fraction : 0.10
    actual_growth = wake_first_spacing_growth
    actual_delay = wake_first_spacing_delay_cells
    actual_maxspacing = mesh_refinement === :none ? wake_first_spacing_max :
                        (wake_first_spacing_max > 0 ? wake_first_spacing_max : 0.01)
    refinement_metadata = Dict{String,Any}()
    wall_projection_shift=0.0
    mesh = if mesh_refinement === :none
        candidate=generate(actual_growth,actual_blend,actual_maxspacing,actual_delay)
        condition_geometry && (wall_projection_shift=_transonic_curve_wall!(candidate,curve,nb))
        candidate
    else
        # Search tested topology families in a deterministic order, then
        # check each realised mesh. The gap only sets the trial order: no
        # source name bypasses the measured air and wake quality gates.
        auto_growth = nested_auto_wake_growth && wake_first_spacing_growth == 1.0 &&
                      wake_first_spacing_delay_cells == 0
        scaled_delay(n) = clamp(round(Int,n*n_wake/64),0,n_wake-1)
        # Larger-grid and high-Reynolds-number finite-base profiles are
        # screened by the same air/cut gates. No named section bypasses a gate.
        measured_extra = if auto_growth && blunt && gap < 0.004 && wall_dn1 < 1e-5 && n_wake == 64
            ((1.125,16,true),(1.12,14,true))
        elseif auto_growth && blunt && gap < 0.004 && wall_dn1 < 1e-5 && n_wake == 96
            ((1.12,45,true),)
        elseif auto_growth && blunt && gap < 0.004 && n_wake == 96
            ((1.12,51,true),)
        elseif auto_growth && blunt && gap < 0.004 && n_wake == 128
            ((1.12,85,true),)
        elseif auto_growth && blunt && gap >= 0.004 && n_wake == 96
            ((1.07,0,true),)
        else
            ()
        end
        preferred_base = !auto_growth ?
            ((wake_first_spacing_growth,wake_first_spacing_delay_cells,false),) :
            !blunt ? ((1.12,0,false),(1.10,0,false),(1.08,0,true),
                      (1.04,0,true)) :
            gap >= 0.004 ? ((1.08,0,true),(1.125,scaled_delay(12),false),
                            (1.12,scaled_delay(10),false),(1.04,0,true)) :
            ((1.12,scaled_delay(26),true),
             (1.12,scaled_delay(26),false),
             (1.12,scaled_delay(24),false),
             (1.12,scaled_delay(28),false),
             (1.10,scaled_delay(20),true),
             (1.08,0,true),(1.04,0,true),
             (1.10,scaled_delay(12),true))
        preferred = (measured_extra..., preferred_base...)
        profiles = !blunt ? ((0.4,0.3,0),(0.3,0.15,0),(0.3,0.1,0),(1.0,0.0,4)) :
            ((0.3,0.15,0),(0.3,0.1,0),(0.3,0.05,0),(0.4,0.3,0),
             (1.0,0.0,3),(1.0,0.0,4))
        selected = nothing
        refusals = String[]
        Q = CMeshGen.CMeshQualityMetrics
        # Preserve every previously admitted profile and its trial order.
        # Only after all of those refuse, try a generic smooth nose join and
        # a feasible graded subdivision with a wider outlet correction. No
        # airfoil name or failed quality threshold changes this admission.
        trials = Any[(growth,delay,split,0.0,2,false,profiles)
                     for (growth,delay,split) in preferred]
        if condition_geometry && blunt
            fallback_profiles = ((0.3,0.15,0),(1.0,0.0,4))
            for (growth,delay) in ((1.12,scaled_delay(22)),(1.12,scaled_delay(20)),
                                   (1.125,scaled_delay(22)),(1.125,scaled_delay(24)),
                                   (1.13,scaled_delay(22)))
                push!(trials,(growth,delay,true,0.09,3,true,fallback_profiles))
            end
        end
        for (growth,delay,second_split,fillet,minimum_children,early_outlet,trial_profiles) in trials
            source_mesh = try
                generate(growth,actual_blend,actual_maxspacing,delay;nose_anchor_fillet=fillet)
            catch err
                push!(refusals,"growth=$growth delay=$delay generator: " * sprint(showerror,err))
                continue
            end
            for (scale,slope,uniform_factor) in trial_profiles
                try
                    candidate, meta, _ = _nested_refine_candidate(source_mesh,nb;
                        scale,slope,uniform_factor,minimum_children,
                        leading_edge_spacing=condition_geometry && blunt ? le_ds : 0.0)
                    if second_split
                        candidate = _nested_uniform_bisect(candidate)
                        meta["refined_n_foil"] *= 2
                        meta["refined_n_wake"] *= 2
                        meta["refined_n_base"] *= 2
                    end
                    if condition_geometry
                        wall_projection_shift=_transonic_curve_wall!(candidate,curve,meta["refined_n_base"])
                        meta["original_foil_j1j2_preserved_before_curve_projection"]=
                            meta["original_foil_j1j2_preserved"]
                        meta["original_foil_j1j2_preserved"]=false
                        meta["wall_representation"]="shared source-interpolating C2 curve; refined wall nodes projected before geometric gates"
                    end
                    if condition_geometry && blunt
                        pinnedX=copy(candidate.X[:,1:2]);pinnedY=copy(candidate.Y[:,1:2])
                        meta["te_collar_tangential_redistribution_max_step"]=
                            _transonic_te_redistribute!(candidate)
                        balance=_transonic_balance_volumes!(candidate)
                        meta["te_collar_volume_balance_sweeps"]=balance.sweeps
                        meta["te_collar_volume_balance_limited_sweeps"]=balance.limited_sweeps
                        meta["te_collar_volume_balance_max_step"]=balance.maximum_shift
                        all(isequal.(candidate.X[:,1:2],pinnedX)) &&
                            all(isequal.(candidate.Y[:,1:2],pinnedY)) ||
                            error("trailing-edge quality repair moved the wall or first-cell layer")
                        meta["quality_repair_preserves_wall_and_first_cell_layer"]=true
                    end
                    candidate.NI*candidate.NJ <= 2_000_000 ||
                        error("refined mesh exceeds two million cells")
                    beforeX,beforeY = copy(candidate.X),copy(candidate.Y)
                    patch = _nested_outlet_patch!(candidate;
                        ncols=(uniform_factor>0 ? 4uniform_factor : 4)*(second_split ? 2 : 1),
                        sweeps=early_outlet ? 20 : 10,
                        start_row=early_outlet ? round(Int,60*candidate.NJ/112) : nothing,
                        full_row=early_outlet ? round(Int,95*candidate.NJ/112) : nothing)
                    contour_preserved = all(isequal.(candidate.X[:,end],beforeX[:,end])) &&
                                        all(isequal.(candidate.Y[:,end],beforeY[:,end]))
                    contour_preserved || error("nested outlet patch moved the farfield contour")
                    defects,worst_jac,_ = CMeshGen._convexity_defects(
                        candidate.X,candidate.Y,candidate.NI,1:candidate.NJ)
                    defects == 0 && worst_jac > 0 ||
                        error("nested candidate has $defects nonconvex cells")
                    summary = Q.quality_summary(candidate.X,candidate.Y,
                                                candidate.NIW,candidate.NIF)
                    failed = Q.gate_failures(Q.quality_gates(summary;cavitating=false))
                    isempty(failed) || error(join(failed,"; "))
                    wake_quality = _nested_wake_cut_metrics(candidate)
                    wake_quality.skew <= 4.0 || error(
                        "true-centroid wake-cut skew/d=$(wake_quality.skew) exceeds provisional 4.0 at x/c=$(wake_quality.worst_x)")
                    wake_quality.ar <= 1500.0 || error(
                        "first-row wake aspect ratio=$(wake_quality.ar) exceeds provisional 1500.0")
                    meta["farfield_source_contour_preserved"] = contour_preserved
                    meta["outlet_patch_start_row"] = patch.start
                    meta["outlet_patch_full_row"] = patch.full
                    meta["outlet_patch_columns_per_side"] = patch.ncols
                    meta["outlet_patch_sweeps"] = patch.sweeps
                    meta["outlet_patch_gain"] = patch.gain
                    meta["maximum_outlet_interior_displacement_chords"] =
                        maximum(hypot.(candidate.X.-beforeX,candidate.Y.-beforeY))
                    meta["nested_wake_growth_auto_selected"] = auto_growth
                    meta["nested_second_i_bisection"] = second_split
                    meta["nested_profile_family"] = fillet > 0 ? "generic_c1_nose_fallback_v1" : "existing_profiles"
                    meta["nose_anchor_fillet_chords"] = fillet
                    meta["original_source_viscous_layers_pinned"] = fillet > 0 ? 12 : 0
                    meta["provisional_wake_cut_skew_over_distance"] = wake_quality.skew
                    meta["provisional_first_row_wake_aspect_ratio"] = wake_quality.ar
                    selected = candidate
                    refinement_metadata = meta
                    actual_growth = growth
                    actual_delay = delay
                    break
                catch err
                    push!(refusals,"growth=$growth delay=$delay split=$second_split scale=$scale slope=$slope: " *
                                   sprint(showerror,err))
                end
            end
            selected === nothing || break
        end
        selected === nothing && error("nested A-plus mesh failed the comparative air and provisional wake gates: " *
                                      join(refusals," | "))
        selected
    end
    mesh.strip === nothing || error("the gas solver does not support the two-block TE strip")
    # Opt-in stall layer (21 Oct 2026, stall_layer_mesh.jl): cells no taller than 2.5e-3 c out to 0.2 c around the
    # foil and the near wake, for separated shear layers; the gates below judge the remapped mesh.
    stall_info = nothing
    if stall_layer
        sl = _stall_layer_mesh(mesh); mesh = sl.mesh; stall_info = sl.info
    end
    collar_info = normal_collar ? _normal_collar!(mesh) : Dict{String,Any}("applied"=>false,"reason"=>"disabled")
    # Passing mesh directly would attach the cavitation channel's slip-wall
    # and inlet tags to this external-flow case. The raw-array overload
    # preserves the gas solver's characteristic far-field boundary everywhere.
    metrics = CMeshMetrics.build_metrics(mesh.X, mesh.Y, mesh.NIW, mesh.NIF;
        trip_upper=trip_upper, trip_lower=trip_lower, trip_side=:wall_order)
    all(==(CMeshGen.OUTER_FARFIELD), metrics.outer_face_kind) ||
        error("external transonic mesh has a non-farfield outer face")
    !metrics.channel_provenance || error("external transonic metrics acquired channel BC provenance")
    quality = CMeshGen.mesh_quality(mesh)
    quality.nfold == 0 && quality.area_min > 0 ||
        error("transonic mesh has folded or nonpositive-area cells")
    quality_report = _transonic_quality_report(mesh)
    mesh_refinement === :aplus_nested_refined &&
        !quality_report["provisional_wake_gates"]["passed"] &&
        error("nested mesh failed the provisional wake-cut skew/aspect gates")
    partner_distance = maximum(hypot(mesh.X[i,1] - mesh.X[mesh.NI+2-i,1],
                                     mesh.Y[i,1] - mesh.Y[mesh.NI+2-i,1])
                               for i in 1:mesh.NIW+1)
    partner_distance <= 1e-12 || error("transonic wake-cut nodes do not coincide")
    foil_indices = mesh.NIW+1:mesh.NIW+mesh.NIF
    actual_nb = mesh_refinement === :none ? nb : refinement_metadata["refined_n_base"]
    surface_indices = first(foil_indices)+actual_nb:last(foil_indices)-actual_nb
    base_indices = actual_nb == 0 ? Int[] :
        vcat(collect(first(foil_indices):first(foil_indices)+actual_nb-1),
             collect(last(foil_indices)-actual_nb+1:last(foil_indices)))
    expected_surface = mesh_refinement === :none ? n_foil :
                       refinement_metadata["refined_n_foil"]
    length(surface_indices) == expected_surface ||
        error("transonic surface index count differs from the selected mesh")
    metadata = Dict{String,Any}(
        "source_te_gap"=>hypot(source_af.x[1]-source_af.x[end],source_af.y[1]-source_af.y[end]),
        "cfd_te_gap"=>gap, "transform"=>condition_geometry ? conditioned.metadata["transform"] : "none",
        "conditioning"=>condition_geometry ? conditioned.metadata : Dict("conditioning"=>"disabled"),
        "geometry_version"=>condition_geometry ? "source_c2_curvature_v1" : "source_polygon_legacy",
        "leading_edge_target_spacing"=>condition_geometry ? le_ds : 0.0,
        "leading_edge_max_curvature"=>le_curvature,
        "maximum_wall_curve_projection_chords"=>wall_projection_shift,
        "conditioned_x"=>copy(af.x),"conditioned_y"=>copy(af.y),
        "wall_x"=>copy(mesh.X[mesh.NIW+1:mesh.NIW+mesh.NIF+1,1]),
        "wall_y"=>copy(mesh.Y[mesh.NIW+1:mesh.NIW+mesh.NIF+1,1]),
        "mesh_family"=>mesh_refinement === :none ?
            "CMeshGen.ChannelMesh A-plus geometry with external-flow farfield metrics" :
            "CMeshGen.ChannelMesh nested-i A-plus geometry with external-flow farfield metrics",
        "mesh_topology"=>mesh_refinement === :none ?
            (blunt ? "aplus_v1 finite base" : "sharp") :
            (blunt ? "aplus_nested_refined_v1 finite base" : "aplus_nested_refined_v1 sharp"),
        "mesh_refinement"=>String(mesh_refinement),
        "n_base"=>actual_nb, "requested_n_base"=>n_base,
        "te_ds"=>te_ds, "wall_dn1"=>wall_dn1,
        "far_extent"=>far_extent, "farfield_height"=>spec.height,
        "farfield_xi_in"=>spec.xi_in, "farfield_xi_out"=>spec.xi_out,
        "farfield_angle_deg"=>outer_angle,
        "wake_blend_fraction"=>actual_blend,
        "wake_first_spacing_growth"=>actual_growth,
        "wake_first_spacing_max"=>actual_maxspacing,
        "wake_first_spacing_delay_cells"=>actual_delay,
        "trip_side"=>"wall_order_chord_projection",
        "outer_boundary_condition"=>"characteristic farfield on every outer face",
        "wake_partner_max_distance"=>partner_distance,
        "mesh_area_min"=>quality.area_min, "mesh_folds"=>quality.nfold,
        "quality_report"=>quality_report)
    merge!(metadata,refinement_metadata)
    metadata["normal_collar"] = collar_info
    stall_info === nothing || (metadata["stall_layer"] = stall_info;
        metadata["mesh_family"] *= " + stall layer v1 (cells <= 2.5e-3 c to 0.2 c)")
    (; mesh, metrics, metadata, foil_indices, surface_indices, base_indices)
end
