# Implicit Newton (CPU) steady method of the compressible backend, on the marched regular mesh. Reached only when a
# request says rans_method = "newton".
# Included inside CFDBackend. No solver package is loaded in the HTTP process: this file holds the request rules, the
# mesh (pure geometry, shared by Create mesh and the worker), the worker launch and the collection of its files.
#
#   request      rans_method = "newton" (steady only). The mesh is always the marched one.
#   topology    the section after the app's usual conditioning (C2 reconstruction, cfd_te_thickness): a closed
#                trailing edge runs one C block (newton_cold_run), an open one runs C+H (newton_pair_cold_run).
#   worker       workers/newton_worker.jl, flag --newton-worker: CPU, Float64, cold start with mesh sequencing 4-2-1.
#   files        one block: the files _collect_transonic_output reads; two blocks: the files of the app's C+H points.
# Development result, not validated.

const NEWTON_METHOD="newton"
const NEWTON_MESH="marched"
# (surface cells, wake cells per side, wall-normal cells); every count is divisible by 4 (mesh sequencing 4-2-1).
const NEWTON_MESH_SIZES=Dict("standard"=>(768,128,96),"fine"=>(1152,192,128),"very_fine"=>(1536,256,160))
const NEWTON_FINE_STEP_CAP=600
const NEWTON_COARSE_STEPS=200
const NEWTON_DEFAULT_STEPS=300

_newton_selected(flow)=String(_get(flow,"rans_method","standard"))==NEWTON_METHOD

"""Request rules of the Implicit Newton method; called by the server's request validation and again before a mesh
or a run. Returns nothing; a request that names another method is left exactly as it was."""
function newton_check_request(flow::AbstractDict,backends)
    _newton_selected(flow) || return nothing
    "transonic" in backends || throw(ArgumentError("The Implicit Newton method belongs to the compressible CFD backend"))
    "cavitation" in backends && throw(ArgumentError(
        "The Implicit Newton method and its marched mesh are not available to the Species / cavitation backend. Run the Implicit Newton method with the compressible backend alone."))
    String(_get(flow,"cfd_time_mode","steady"))=="steady" || throw(ArgumentError(
        "The Implicit Newton method is a steady method. Select Steady pseudo-time."))
    model=lowercase(String(_get(flow,"cfd_viscosity_model","constant_kinematic")))
    model in ("constant_kinematic","sutherland","constant_dynamic") ||
        throw(ArgumentError("cfd_viscosity_model must be constant_kinematic, sutherland, or constant_dynamic"))
    String(_get(flow,"cfd_resolution","standard")) in keys(NEWTON_MESH_SIZES) ||
        throw(ArgumentError("cfd_resolution must be standard, fine, or very_fine"))
    streamline_enabled(flow) && throw(ArgumentError(
        "Streamline alignment applies to the app's own C+H mesh; it is not available with the marched mesh of the Implicit Newton method."))
    gurney_enabled(flow) && throw(ArgumentError(
        "The Gurney flap is a baffle of the low-speed CFD; it is not available with the Implicit Newton method."))
    nothing
end

"The section as the CFD meshes it: equation NACA or the prepared file, then the app's conditioning with cfd_te_thickness."
function _newton_section(source,path,flow)
    naca=_get(source,"kind","")=="naca4"
    af=naca ? CMeshGen.naca4_points(parse(Int,String(_get(source,"id","")))) : CMeshGen.read_airfoil_dat(path)
    thickness=_number(flow,"cfd_te_thickness",0.;lo=0,hi=.05)
    conditioned=CMeshGen.condition_airfoil(af;trailing_edge_thickness=thickness)
    f=conditioned.airfoil
    gap=hypot(f.x[1]-f.x[end],f.y[1]-f.y[end])
    (;af,naca,thickness,gap,blunt=gap>1e-8,source_gap=hypot(af.x[1]-af.x[end],af.y[1]-af.y[end]),
      transform=String(conditioned.metadata["transform"]))
end

function _newton_mesh_sizes(flow,resolution)
    sizes=get(NEWTON_MESH_SIZES,resolution,nothing)
    sizes===nothing && throw(ArgumentError("cfd_resolution must be standard, fine, or very_fine"))
    # The same hidden validation overrides as the other compressible meshes; absent, the profile's sizes.
    n=(_integer(flow,"cfd_n_foil",sizes[1];lo=64,hi=4096),_integer(flow,"cfd_n_wake",sizes[2];lo=16,hi=1024),
       _integer(flow,"cfd_n_normal",sizes[3];lo=16,hi=512))
    all(x->x%4==0,n) || throw(ArgumentError(
        "The marched mesh of the Implicit Newton method needs surface, wake and wall-normal cell counts divisible by 4 (mesh sequencing 4-2-1)"))
    n
end

function _newton_quality_single(mesh)
    try
        report=_transonic_quality_report(mesh)
        # Wording only: the shared report describes its air thresholds for the nested C mesh; say what they are for this one.
        report["air_reference_gates"]["applicability"]="comparative CMeshQualityV1 air thresholds, as reported for the app's nested-refinement profile; the marched mesh is built by another rule"
        report
    catch err
        Dict{String,Any}("available"=>false,"reason"=>first(sprint(showerror,err),400),
            "geometric_gate"=>Dict("passed"=>true,"criteria"=>"positive cell areas, checked by the mesher"))
    end
end

# The measured two-block face metrics and air reference gates the app's own C+H mesh reports (ch_mesh.jl).
function _newton_quality_pair(mesh)
    Q=CMeshGen.CMeshQualityMetrics
    try
        s=Q.quality_summary(mesh.X,mesh.Y,mesh.NIW,mesh.NIF;strip=mesh.strip)
        gates=Q.quality_gates(s;cavitating=false,strip_grade=true)
        failed=Q.gate_failures(gates)
        (;passed=isempty(failed),failed=String.(failed),status=isempty(failed) ? "passed_air_geometry" : "outside_air_geometry_limits",
          report=Dict{String,Any}("air_reference_gates"=>Dict("passed"=>isempty(failed),"failed"=>String.(failed),
                "applicability"=>"comparative CMeshQualityV1 air thresholds of the app's C+H mesh; the marched mesh is built by another rule"),
            "geometric_gate"=>Dict("passed"=>true,"criteria"=>"positive cell areas in both blocks, checked by the mesher"),"cells"=>s.cells,
            "global"=>Dict("maximum_face_nonorthogonality_deg"=>s.theta_max,"minimum_corner_angle_deg"=>s.minang,
                "maximum_aspect_ratio"=>s.ar_max,"maximum_neighbor_volume_ratio"=>s.vr_max),
            "interfaces"=>Dict("maximum_volume_ratio"=>s.vr_junction,
                "maximum_nonorthogonality_deg"=>s.theta_junction,"maximum_skew_over_distance"=>s.skew_junction)))
    catch err
        reason="mesh quality metrics unavailable: "*first(sprint(showerror,err),300)
        (;passed=false,failed=[reason],status="not_assessed",
          report=Dict{String,Any}("available"=>false,"reason"=>reason,
            "geometric_gate"=>Dict("passed"=>true,"criteria"=>"positive cell areas in both blocks, checked by the mesher")))
    end
end

"""Build the marched mesh of the Implicit Newton method. Pure geometry: Create mesh and the worker both call it, so
the preview is the mesh of the run.

Returns (; mesh, metadata). `mesh` = (X, Y, NI, NJ, NIW, NIF, dn1, strip); `strip` is nothing for a closed trailing edge
(one C block) and the H block in the app's C+H layout (row 1 on the lower leg, column 1 on the base) otherwise.
"""
function _build_newton_mesh(af;reynolds::Real,alpha_deg::Real,farfield_angle_deg::Real=alpha_deg,
                            cfd_te_thickness::Real=0.0,n_foil::Int,n_wake::Int,n_normal::Int,
                            te_ds::Real=1e-3,wake_aspect::Real=200.0)
    conditioned=CMeshGen.condition_airfoil(af;trailing_edge_thickness=cfd_te_thickness)
    # Without a thickness override the mesher receives the source section, as in the benchmark runs (it conditions
    # the contour itself); with one it receives the section the conditioning produced.
    section=cfd_te_thickness>0 ? conditioned.airfoil : af
    cf=0.026*reynolds^(-1/7)
    dn1=0.7/(reynolds*sqrt(cf/2))          # the wall spacing of the app's other compressible meshes (y+ about 0.7)
    built=CMeshGen.marched_airfoil_mesh(section;n_foil,n_wake,n_normal,dn1=Float64(dn1),alpha_deg=Float64(alpha_deg),
        farfield_angle_deg=Float64(farfield_angle_deg),te_ds=Float64(te_ds),wake_aspect=Float64(wake_aspect))
    m=built.mesh;info=built.info;st=m.strip;two=st!==nothing
    mesh=(;X=m.X,Y=m.Y,NI=m.NI,NJ=m.NJ,NIW=m.NIW,NIF=m.NIF,dn1=m.dn1,
        strip=two ? (;X=st.X,Y=st.Y,NI=st.NI,NJ=st.NJ,n_base=st.NJ÷2,gap=st.gap) : nothing)
    foil=conditioned.airfoil
    metadata=Dict{String,Any}(
        "mesh_family"=>two ? "marched regular C block + H strip behind the base" : "marched regular C block",
        "mesh_topology"=>two ? "marched_C_H" : "marched_C",
        "mesh_blocks"=>two ? 2 : 1,
        "outer_boundary_condition"=>"characteristic farfield",
        "n_base"=>two ? st.NJ÷2 : 0,
        "source_te_gap"=>hypot(af.x[1]-af.x[end],af.y[1]-af.y[end]),
        "cfd_te_gap"=>hypot(foil.x[1]-foil.x[end],foil.y[1]-foil.y[end]),
        "transform"=>String(conditioned.metadata["transform"]),"conditioning"=>conditioned.metadata,
        "wall_dn1"=>m.dn1,"estimated_yplus"=>reynolds*m.dn1*sqrt(cf/2),
        "surface_cells"=>m.NIF,"wake_cells_per_side"=>m.NIW,"normal_cells"=>m.NJ,
        "strip_cross_cells"=>two ? st.NJ : 0,
        "cells"=>m.NI*m.NJ+(two ? st.NI*st.NJ : 0),
        "trailing_edge_spacing"=>info.te_ds,"leading_edge_spacing"=>info.le_ds,
        "wake_first_row_aspect_limit"=>Float64(wake_aspect),"farfield_distance_chords"=>info.far,
        "farfield_angle_deg"=>Float64(farfield_angle_deg),"minimum_cell_area"=>info.area_min,
        "solver_supported"=>true,"solver_backends"=>["transonic"],
        "solver_support_message"=>"Marched regular mesh of the Implicit Newton (CPU) steady method. Development mesh, not validated.",
        "qualification"=>"marched regular mesh; development, not validated")
    if two
        quality=_newton_quality_pair(mesh)
        metadata["quality_report"]=quality.report
        metadata["quality_passed"]=quality.passed
        metadata["quality_status"]=quality.status
        metadata["quality_failures"]=quality.failed
        metadata["quality_scope"]="geometry and the air reference limits of the app's C+H mesh only; not CFD convergence"
    else
        metadata["quality_report"]=_newton_quality_single(mesh)
    end
    (;mesh,metadata)
end

"The app's own mesh writers: the structured payload for one block, the C+H payload (kind structured_blocks) for two."
_newton_write_mesh_artifacts(mesh,metadata,outdir,resolution)=mesh.strip===nothing ?
    _write_mesh_artifacts(mesh,metadata,outdir,resolution) : _write_ch_mesh_artifacts(mesh,metadata,outdir,resolution)

_newton_geometry_support(naca)=naca ?
    "NACA four-digit equations evaluated at 4097 cosine-spaced support points; C2 surface evaluation by the marched mesher" :
    "C2 interpolation of source coordinates"

function _newton_preview_mesh(path::AbstractString,flow::AbstractDict,outdir::AbstractString;source::AbstractDict=Dict())
    newton_check_request(flow,["transonic"])
    section=_newton_section(source,path,flow)
    resolution=String(_get(flow,"cfd_resolution","standard"))
    n_foil,n_wake,n_normal=_newton_mesh_sizes(flow,resolution)
    alpha=_number(flow,"alpha_deg",0.0;lo=-90,hi=90)
    built=_build_newton_mesh(section.af;reynolds=_number(flow,"reynolds",300_000.0;lo=1e3,hi=1e9),alpha_deg=alpha,
        farfield_angle_deg=_transonic_mesh_angle(flow,alpha),cfd_te_thickness=section.thickness,
        n_foil,n_wake,n_normal)
    built.metadata["geometry_support"]=_newton_geometry_support(section.naca)
    payload=_newton_write_mesh_artifacts(built.mesh,built.metadata,outdir,resolution)
    payload["artifact"]=joinpath(outdir,"mesh_full.msgpack")
    payload["preview_artifact"]=joinpath(outdir,"mesh_preview.msgpack")
    (;mesh=payload,metadata=built.metadata)
end

"""The worker's input for one angle. The settings are those of the benchmark runs of 13-14 Nov 2026; only the mesh
size (cfd_resolution) and the fine-level step budget (cfd_steps, at most $NEWTON_FINE_STEP_CAP) follow the request."""
function _newton_config(case,path,flow,alpha,dir;mesh_anchor=nothing,warmdir=nothing)
    newton_check_request(flow,["transonic"])
    source=_get(case,"airfoil",Dict())
    section=_newton_section(source,path,flow)
    resolution=String(_get(flow,"cfd_resolution","standard"))
    n_foil,n_wake,n_normal=_newton_mesh_sizes(flow,resolution)
    requested_steps=_integer(flow,"cfd_steps",NEWTON_DEFAULT_STEPS;lo=1,hi=2_000_000)
    fine_steps=min(requested_steps,NEWTON_FINE_STEP_CAP)
    fully=_boolean(flow,"cfd_fully_turbulent",false)
    cfg=Dict{String,Any}(
        "geometry"=>String(path),"source_geometry"=>String(path),"outdir"=>String(dir),
        "mesh_mode"=>NEWTON_MESH,"resolution"=>resolution,
        "mach"=>_number(flow,"mach",0.73;lo=0.02,hi=0.98),
        "reynolds"=>_number(flow,"reynolds",6.5e6;lo=1e3,hi=1e9),
        "alpha_deg"=>Float64(alpha),
        # The wake direction: the mesh-angle rule of the single-block compressible mesh (the first angle of a sweep).
        "cfd_farfield_angle_deg"=>_transonic_mesh_angle(flow,mesh_anchor===nothing ? alpha : mesh_anchor),
        "time_mode"=>"steady",
        # The method runs on the CPU in Float64 whatever the request's GPU switch says; the key is read at worker load.
        "gpu"=>false,"precision"=>"float64","requested_gpu"=>_boolean(flow,"cfd_gpu",false),
        "rans_method"=>NEWTON_METHOD,"rans_acceleration"=>true,
        "rans_acceleration_method"=>RANSAcceleration.NEWTON_LABEL,
        "newton_topology"=>section.blunt ? "c_plus_h" : "c",
        "n_foil"=>n_foil,"n_wake"=>n_wake,"n_normal"=>n_normal,"te_ds"=>1e-3,"wake_aspect"=>200.0,
        "cfd_te_thickness"=>section.thickness,"source_te_gap"=>section.source_gap,"cfd_te_gap"=>section.gap,
        # Forced trips on the surface (C block), the app's defaults when none are given; the strip has no trip.
        "trip_upper"=>fully ? -0.05 : _number(flow,"trip_upper",0.03;lo=0.0,hi=1.0),
        "trip_lower"=>fully ? -0.05 : _number(flow,"trip_lower",0.05;lo=0.0,hi=1.0),
        "fully_turbulent"=>fully,
        "viscosity_model"=>lowercase(String(_get(flow,"cfd_viscosity_model","constant_kinematic"))),
        "freestream_temperature_k"=>_number(flow,"cfd_freestream_temperature_k",288.15;lo=100.0,hi=2000.0),
        "tu_inf"=>0.001,"mu_t_ratio"=>0.1,
        "limiter"=>"venkat_chord","limiter_smooth_coefficient"=>0.2,
        # Step budget: the fine level takes at most min(cfd_steps, 600) steps, the two coarse levels 200 each.
        "steps"=>fine_steps,"requested_cfd_steps"=>requested_steps,
        "newton_levels"=>[4,2,1],"newton_nsteps"=>[NEWTON_COARSE_STEPS,NEWTON_COARSE_STEPS,fine_steps],
        "newton_cflmax"=>[1e3,1e3,1e3],"newton_load_tol"=>[1e-3,3e-4,1e-4],"newton_window"=>[10,15,20],
        "newton_lintol"=>0.3,"newton_maxit"=>60,"newton_growth"=>1.5,"newton_every"=>2,"newton_vortex_relax"=>0.3,
        "numerical_source_sha256"=>RANSAcceleration.numerical_source_receipt())
    if section.naca
        code=String(_get(source,"id",""))
        occursin(r"^[0-9]{4}$",code) || throw(ArgumentError("invalid NACA four-digit code"))
        cfg["naca4_code"]=parse(Int,code)
    end
    mesh_anchor===nothing || (cfg["polar_mesh_anchor_deg"]=cfg["cfd_farfield_angle_deg"])
    # Every point starts cold. A checkpoint of the previous angle, if one is offered, is recorded and not used.
    if warmdir!==nothing
        warm=joinpath(warmdir,"state_checkpoint.jls")
        isfile(warm) && (cfg["warm_start_requested"]=warm)
    end
    cfg
end

# Threads of the Newton worker: one per physical core by default (half the logical processors where they are paired;
# the march was measured at 16 threads on a 16-core, 32-thread machine and gains nothing from the second thread of a
# core). LUDWIG_NEWTON_THREADS sets another count, or "auto" for Julia's own choice.
function _newton_threads()
    value=strip(get(ENV,"LUDWIG_NEWTON_THREADS",""))
    occursin(r"^(auto|[1-9][0-9]{0,2})$",value) ? String(value) : string(max(1,Sys.CPU_THREADS>=4 ? Sys.CPU_THREADS÷2 : Sys.CPU_THREADS))
end

function _newton_point(case,path,flow,alpha,dir,progress,log,warmdir=nothing;mesh_anchor=nothing)
    cfg=_newton_config(case,path,flow,alpha,dir;mesh_anchor,warmdir)
    two=cfg["newton_topology"]=="c_plus_h"
    _deliver_log(log,"Implicit Newton (CPU, Float64) on the marched mesh: $(two ? "finite trailing edge, two blocks (C+H)" : "closed trailing edge, one C block"); cold start, mesh sequencing 4-2-1, at most $(cfg["steps"]) fine-level steps. Development method, not validated.")
    cfg["requested_gpu"] && _deliver_log(log,"The GPU switch does not apply to the Implicit Newton method; it runs on the CPU.")
    haskey(cfg,"warm_start_requested") && _deliver_log(log,"The Implicit Newton method starts cold at every angle; the previous angle's checkpoint is not used.")
    cfg_path=joinpath(dir,"input.toml")
    open(cfg_path,"w") do io;TOML.print(io,cfg);end
    julia=joinpath(Sys.BINDIR,Base.julia_exename())
    worker=joinpath(@__DIR__,"cfd_backend.jl")
    threads=_newton_threads()
    cmd=`$julia --project=$SRC --startup-file=no -t $threads $worker --newton-worker $cfg_path`
    ok=_log_process(cmd,joinpath(dir,"solver.log"),progress,log,"transonic",Float64(alpha);
        residual_kind="Newton march: RMS continuity residual, freestream-scaled; all six equations in newton_history.csv",
        every_history_row=true)
    ok || throw(ErrorException("Implicit Newton worker failed: $(_solver_failure_summary(joinpath(dir,"solver.log"))); see solver.log"))
    two && return _collect_newton_pair_output(dir,Float64(alpha))
    _collect_transonic_output(dir,alpha,Dict{String,Any}("source_te_gap"=>cfg["source_te_gap"],
        "cfd_te_gap"=>cfg["cfd_te_gap"],"transform"=>"none","path"=>String(path),"mesh_mode"=>NEWTON_MESH))
end

"""The point dictionary of a two-block Newton run, from the files the app's C+H points have. The status follows the
march's own rule only: "completed" when the worker recorded that CL and CD settled on the fine level
(convergence_guard.status = "early_stop_eligible"), "not_converged" otherwise."""
function _collect_newton_pair_output(dir,alpha)
    meta=TOML.parsefile(joinpath(dir,"result.toml"))
    mesh,fields,cp,history=_read_block_point_files(dir,meta)
    guard=get(meta,"convergence_guard",Dict{String,Any}("status"=>"unavailable","reason"=>"the worker recorded no stop guard"))
    settled=get(guard,"status","")=="early_stop_eligible"
    diagnostics=Dict{String,Any}(string(k)=>v for (k,v) in meta)
    merge!(diagnostics,Dict("mesh_cells"=>fields["full_cell_count"],"convergence_guard"=>guard,
        "mesh_quality_passed"=>get(mesh,"quality_passed",false),"physical_time"=>false,"validated"=>false))
    point=Dict{String,Any}("backend"=>"transonic","alpha_deg"=>alpha,
        "status"=>settled ? "completed" : "not_converged",
        "cl"=>meta["cl"],"cd"=>meta["cd"],"cm"=>meta["cm"],
        "convergence"=>get(meta,"convergence_summary",get(guard,"reason","inspect the iteration history")),
        "validation"=>get(meta,"validation_summary","development result; not validated"),
        "geometry"=>get(meta,"geometry",get(mesh,"geometry_metadata",Dict())),
        "mesh"=>mesh,"fields"=>fields,"cp"=>cp,"iteration_history"=>history,
        "diagnostics"=>diagnostics,"notes"=>String.(get(meta,"point_notes",String[])),
        "artifacts"=>Dict("mesh_full"=>joinpath(dir,"mesh_full.msgpack"),
            "mesh_preview"=>joinpath(dir,"mesh_preview.msgpack"),
            "field_preview"=>fields["artifact"],"field_full"=>fields["full_artifact"],
            "cp_csv"=>joinpath(dir,meta["cp_csv"]),"iteration_csv"=>joinpath(dir,meta["iteration_csv"]),
            "newton_history_csv"=>joinpath(dir,meta["newton_history_csv"]),
            "wall_force_csv"=>joinpath(dir,meta["wall_force_csv"]),
            "result_toml"=>joinpath(dir,"result.toml"),"log"=>joinpath(dir,"solver.log")))
    _compact_ch_transport(point)
end
