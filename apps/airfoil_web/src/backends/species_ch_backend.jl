# Conservative Species on a finite (blunt) trailing edge in a free stream: the request, the mesh preview, the launch
# and the collection of the two-block (C+H) run. Included inside CFDBackend after species_backend.jl. The worker is
# workers/species_ch_worker.jl; the production water-tunnel driver has its own two-block run (species_backend.jl).

# The optional C+H mesh settings of ch_mesh.jl, passed on only when the request names them.
const _SPECIES_CH_MESH_KEYS=("cfd_stall_layer","cfd_le_ds_scale","cfd_ch_pair_factor","cfd_ch_te_multiple",
    "cfd_ch_foil_cells","cfd_ch_surface_distribution","cfd_ch_h_cross")

"""The C+H mesh of a Species run in a free stream: the compressible solver's own C+H mesh (`_build_ch_mesh`), so
Create mesh and both solvers build the same nodes, with the Species boundary description in its metadata. The
cavitating geometry limits are recorded in the metadata (`cavitation_quality_passed`); they do not refuse the mesh."""
function _species_ch_mesh(af,flow,resolution)
    built=_build_ch_mesh(af,flow;resolution);metadata=built.metadata
    metadata["solver_supported"]=true
    metadata["solver_backends"]=["cavitation"]
    metadata["solver_support_message"]="Species C+H Run in a free stream: the conservative Species pair (C block around foil and wake, H block behind the base), physical time. Geometry limits are recorded, not a convergence or validation certificate."
    metadata["outer_boundary_condition"]="mixture characteristic outer contour; fixed-static-pressure downstream ends of both blocks (acoustically reflecting)"
    metadata["species_boundary_conditions"]=_species_boundary_conditions("free_fluid")
    (;mesh=built.mesh,metadata)
end

_species_ch_airfoil(source,path)=_get(source,"kind","")=="naca4" ?
    CMeshGen.naca4_points(parse(Int,String(_get(source,"id","")))) : CMeshGen.read_airfoil_dat(path)

"Create mesh for Species air or the free-fluid domain on a finite trailing edge: the nodes the Run will use."
function _preview_species_ch_mesh(path,flow,outdir;source=Dict())
    refusal=species_finite_edge_refusal(flow;run=false)
    refusal===nothing || throw(ArgumentError(refusal))
    resolution=String(_get(flow,"cfd_resolution","standard"))
    built=_species_ch_mesh(_species_ch_airfoil(source,path),flow,resolution)
    payload=_write_ch_mesh_artifacts(built.mesh,built.metadata,outdir,resolution)
    (;mesh=payload,metadata=built.metadata)
end

"The worker input of a two-block Species point; every refusal of the request is raised here, before any process starts."
function _species_ch_config(case,path,flow,alpha,dir)
    material,domain=cavitation_configuration(flow)
    refusal=species_finite_edge_refusal(flow)
    refusal===nothing || throw(ArgumentError(refusal))
    streamline_enabled(flow) && throw(ArgumentError("Streamline alignment is an option of the compressible and low-speed C+H runs; the Species C+H run uses the unaligned mesh."))
    _boolean(flow,"rans_acceleration",false) && throw(ArgumentError(
        "RANS acceleration is the steady Species air iteration on one C block; the C+H pair runs in physical time."))
    resolution=String(_get(flow,"cfd_resolution","standard"))
    resolution in ("standard","fine","very_fine") || throw(ArgumentError("cfd_resolution must be standard, fine, or very_fine"))
    for key in ("cfd_n_foil","cfd_n_wake","cfd_n_normal")
        _get(flow,key,nothing)===nothing || throw(ArgumentError(
            "$key sizes the single C block of a closed trailing edge; the C+H mesh takes its sizes from the resolution profile"))
    end
    duration=_number(flow,"cav_duration_ctu",10.0;lo=0.001,hi=1000)
    cfg=Dict{String,Any}(
        "geometry"=>String(path),"source_geometry"=>String(path),"outdir"=>String(dir),
        "mesh_mode"=>SPECIES_CH_MESH,"resolution"=>resolution,
        "cav_fluid"=>material,"cav_domain"=>domain,
        "cav_air_mode"=>material=="air" ? String(_get(flow,"cav_air_mode","steady")) : "physical",
        "mach"=>_number(flow,"mach",0.2;lo=0.02,hi=0.3),
        "reynolds"=>_number(flow,"reynolds",3e6;lo=1e3,hi=1e9),
        "alpha_deg"=>Float64(alpha),"sigma"=>_number(flow,"sigma",1.25;lo=0.01,hi=20),
        "cfd_te_thickness"=>_number(flow,"cfd_te_thickness",0.;lo=0,hi=.05),
        # The outer frame of Create mesh (ch_mesh.jl): zero unless the request turns it.
        "cfd_farfield_angle_deg"=>_number(flow,"cfd_farfield_angle_deg",0.;lo=-45,hi=45),
        "steps"=>_integer(flow,"cfd_steps",500_000;lo=1,hi=2_000_000),
        "gpu"=>_boolean(flow,"cfd_gpu",true),
        "cfl"=>_number(flow,"cfd_cfl",0.45;lo=0.01,hi=0.7),
        "auto_stop"=>_boolean(flow,"cfd_auto_stop",true),
        "mass_ceiling"=>_number(flow,"cfd_mass_ceiling",0.1;lo=1e-6,hi=1000.0),
        "mass_net_limit"=>_number(flow,"cfd_mass_net_limit",0.001;lo=1e-8,hi=1.0),
        "trip_upper"=>_number(flow,"trip_upper",0.05;lo=0.0,hi=1.0),
        "trip_lower"=>_number(flow,"trip_lower",0.05;lo=0.0,hi=1.0),
        "cav_duration_ctu"=>duration,
        "burnin_ctu"=>_number(flow,"cfd_burnin_ctu",0.5duration;lo=0.0,hi=10000),
        "block_ctu"=>_number(flow,"cfd_block_ctu",5.0;lo=0.001,hi=1000),
        "physical_time_refined"=>_boolean(flow,"cfd_physical_time_refined",false),
        "cav_history_stride"=>_integer(flow,"cav_history_stride",200;lo=1,hi=1000000),
        # The finite-base step proposal of the production tunnel runs: 0.99 of the stage-1 source, viscous and
        # SST clocks, which keeps the later stages inside their own clocks.
        "cav_stage_clock_margin"=>_number(flow,"cav_stage_clock_margin",.99;lo=.5,hi=1.))
    merge!(cfg,_species_model_settings(flow;alpha))
    for key in _SPECIES_CH_MESH_KEYS
        value=_get(flow,key,nothing)
        value===nothing || (cfg[key]=value)
    end
    source=_get(case,"airfoil",Dict())
    if _get(source,"kind","")=="naca4"
        code=String(_get(source,"id",""))
        occursin(r"^[0-9]{4}$",code) || throw(ArgumentError("invalid NACA four-digit code"))
        cfg["naca4_code"]=parse(Int,code)
    end
    cfg
end

"Run one Species point on a finite trailing edge: the shared Species worker on the C+H pair."
function _species_ch_point(case,path,flow,alpha,dir,progress,log)
    cfg=_species_ch_config(case,path,flow,alpha,dir)
    _deliver_log(log,"Starting the conservative Species C+H pair in a free stream ($(cfg["cav_fluid"]), physical time, $(cfg["gpu"] ? "GPU" : "CPU")).")
    cfg_path=joinpath(dir,"input.toml")
    open(cfg_path,"w") do io;TOML.print(io,cfg);end
    julia=joinpath(Sys.BINDIR,Base.julia_exename())
    worker=joinpath(@__DIR__,"cfd_backend.jl")
    cmd=`$julia --project=$SRC --startup-file=no -t auto $worker --species-worker $cfg_path`
    ok=_log_process(cmd,joinpath(dir,"solver.log"),progress,log,"cavitation",Float64(alpha))
    ok || throw(ErrorException("Species C+H worker failed: $(_solver_failure_summary(joinpath(dir,"solver.log"))); see solver.log"))
    _collect_species_ch_output(dir,Float64(alpha))
end

"Read the files of a finished two-block Species point into the point the server sends (both blocks' fields kept)."
function _collect_species_ch_output(dir,alpha)
    meta=TOML.parsefile(joinpath(dir,"result.toml"))
    mesh,fields,cp,_=_read_block_point_files(dir,meta)
    # The history with the physical time of every record, as the single-block Species point returns it, bounded
    # for the job response (the complete record stays in iteration_history.csv).
    rows=_csv_rows(joinpath(dir,meta["iteration_csv"]));stride=max(1,cld(length(rows),1200))
    hist=[Dict{String,Any}(k=>parse(k=="iteration" ? Int : Float64,v) for (k,v) in rows[n])
        for n in unique(vcat(collect(1:stride:length(rows)),length(rows)))]
    for h in hist;h["residual"]=h["residual_scaled"];end
    stats=meta["time_statistics"]
    status=!get(stats,"sampling_adequate",false) ? "insufficient_sampling" :
        !get(stats,"physical_mean_eligible",false) ? "unqualified_physical_mean" : "completed"
    notes=String[meta["energy_model"],meta["trip_model"],"SST closure; no transition/stall validation.",
        "The carrier temperature field is an EOS pedestal, not a thermodynamic temperature.",
        "Two blocks: the C block around foil and wake and the H block behind the base; the coefficients include the base."]
    get(mesh,"quality_passed",false) || push!(notes,"The mesh has failed air geometry limits; inspect the mesh quality diagnostics.")
    point=Dict{String,Any}("backend"=>"cavitation","alpha_deg"=>alpha,"status"=>status,
        "cl"=>meta["cl"],"cd"=>meta["cd"],"cm"=>meta["cm"],"geometry"=>meta["geometry"],"mesh"=>mesh,"cp"=>cp,
        "iteration_history"=>hist,
        "convergence"=>"physical time; sampling status: $(get(stats,"status","insufficient_sampling"))",
        "validation"=>meta["validation"],"fields"=>fields,"diagnostics"=>meta,"notes"=>notes,
        "artifacts"=>Dict("mesh_full"=>joinpath(dir,"mesh_full.msgpack"),"mesh_preview"=>joinpath(dir,"mesh_preview.msgpack"),
            "field_preview"=>fields["artifact"],"field_full"=>fields["full_artifact"],
            "cp_csv"=>joinpath(dir,meta["cp_csv"]),"iteration_csv"=>joinpath(dir,meta["iteration_csv"]),
            "wall_force_csv"=>joinpath(dir,meta["wall_force_csv"]),
            "result_toml"=>joinpath(dir,"result.toml"),"log"=>joinpath(dir,"solver.log")))
    _compact_ch_transport(point)
end
