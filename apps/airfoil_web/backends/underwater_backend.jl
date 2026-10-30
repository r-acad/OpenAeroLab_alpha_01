# App integration of the explicit, physical-time, nonlinear water/air worker.
# Included inside CFDBackend. Preview is host-only and never initializes CUDA.
include(joinpath(@__DIR__,"..","underwater_worker.jl"))

const UNDERWATER_PROJECTION_SCOPE="Experimental incompressible three-constituent nonlinear free surface; conservative volume/momentum transport and adjoint pressure projection; no viscosity, SST, phase change, acoustic waves or stall validation"

function _underwater_config(path,flow,outdir;source=Dict(),alpha=_number(flow,"alpha_deg",0.))
    abs(alpha)<=45 || throw(ArgumentError("experimental underwater body grid currently supports alpha within ±45 degrees"))
    _number(flow,"cfd_te_thickness",0.;lo=0,hi=.05)==0 || throw(ArgumentError("the underwater development grid preserves the native trailing edge; set the optional CFD trailing-edge override to zero"))
    cfg=Dict{String,Any}("geometry"=>String(path),"outdir"=>String(outdir),
        "formulation"=>String(_get(flow,"cav_fs_formulation","incompressible_projection")),
        "alpha_deg"=>Float64(alpha),"gpu"=>_boolean(flow,"cfd_gpu",true),
        "chord_m"=>_number(flow,"cav_chord_m",1.;lo=.001,hi=1000),
        "submergence_chords"=>_number(flow,"cav_depth_chords",1.;lo=.05,hi=100),
        "speed_m_s"=>_number(flow,"cav_speed_m_s",3.;lo=.01,hi=100),
        "water_depth_chords"=>_number(flow,"cav_water_depth_chords",4.;lo=.2,hi=200),
        "air_height_chords"=>_number(flow,"cav_air_height_chords",2.;lo=.2,hi=100),
        "duration_seconds"=>_number(flow,"cav_fs_duration_seconds",String(_get(flow,"cav_fs_formulation","incompressible_projection"))=="incompressible_projection" ? .1 : .002;lo=1e-8,hi=100),
        "max_steps"=>_integer(flow,"cav_fs_max_steps",2000;lo=1,hi=10_000_000),
        "n_foil"=>_integer(flow,"cav_fs_n_foil",96;lo=32,hi=1024),
        "n_normal"=>_integer(flow,"cav_fs_n_normal",24;lo=12,hi=256),
        "history_stride"=>_integer(flow,"cav_fs_history_stride",50;lo=1,hi=10000),
        "frame_stride"=>_integer(flow,"cav_fs_frame_stride",500;lo=1,hi=100000),
        "initial_vapor_fraction"=>_number(flow,"cav_fs_initial_vapor_fraction",1e-6;lo=0,hi=.01),
        "outer_bc"=>"open","flux"=>"thornber")
    cfg["formulation"] in ("incompressible_projection","barotropic_explicit") ||
        throw(ArgumentError("cav_fs_formulation must be incompressible_projection or barotropic_explicit"))
    if cfg["formulation"]=="incompressible_projection"
        cfg["dt_max"]=_number(flow,"cav_fs_dt_max_seconds",.0001;lo=1e-8,hi=.01)
    end
    cfg["water_depth_chords"]>cfg["submergence_chords"]+.3 ||
        throw(ArgumentError("water depth must exceed airfoil submergence by at least 0.3 chord"))
    iseven(cfg["n_foil"]) || throw(ArgumentError("cav_fs_n_foil must be even"))
    if _get(source,"kind","")=="naca4"
        cfg["naca4_code"]=parse(Int,String(_get(source,"id","")))
        cfg["geometry"]="" # Equations at the actual grid stations, not a coarse DAT resample.
    end
    cfg
end

function preview_underwater_mesh(path,flow,outdir;source=Dict())
    cfg=_underwater_config(path,flow,outdir;source)
    W=UnderwaterAirfoilWorker;M=W.CMeshFreeSurfaceAirfoil
    g=M.build_airfoil_grid(;geometry=cfg["geometry"],code=Int(get(cfg,"naca4_code",12)),
        n_foil=cfg["n_foil"],n_normal=cfg["n_normal"],chord_m=cfg["chord_m"],
        submergence_chords=cfg["submergence_chords"],speed_m_s=cfg["speed_m_s"],
        water_depth_chords=cfg["water_depth_chords"],air_height_chords=cfg["air_height_chords"],
        alpha_deg=cfg["alpha_deg"],initial_vapor_fraction=cfg["initial_vapor_fraction"])
    mesh=W.mesh_payload(g);mkpath(outdir)
    mesh["geometry_metadata"]["formulation"]=cfg["formulation"]
    for name in ("mesh_preview.msgpack","mesh_full.msgpack")
        write(joinpath(outdir,name),AirfoilWire.encode(mesh,:msgpack))
    end
    mesh["artifact"]=joinpath(outdir,"mesh_full.msgpack")
    mesh["preview_artifact"]=joinpath(outdir,"mesh_preview.msgpack")
    (;mesh,metadata=mesh["geometry_metadata"])
end

function _underwater_point(case,path,flow,alpha,dir,progress,log)
    cfg=_underwater_config(path,flow,dir;source=case["airfoil"],alpha)
    projection=cfg["formulation"]=="incompressible_projection"
    scope=projection ? UNDERWATER_PROJECTION_SCOPE : UnderwaterAirfoilWorker.SCOPE
    mkpath(dir);config_path=joinpath(dir,"underwater.toml")
    open(config_path,"w") do io;TOML.print(io,cfg);end
    _deliver_log(log,scope)
    _deliver_log(log,"Physical-time startup: chord=$(cfg["chord_m"]) m, speed=$(cfg["speed_m_s"]) m/s, depth=$(cfg["submergence_chords"]) chord; duration=$(cfg["duration_seconds"]) s, maximum $(cfg["max_steps"]) steps. Mach, Reynolds and sigma are not model inputs.")
    julia=joinpath(Sys.BINDIR,Base.julia_exename())
    worker=joinpath(@__DIR__,"..",projection ? "underwater_projection_worker.jl" : "underwater_worker.jl")
    cmd=`$julia --project=$SRC --startup-file=no -t 2 $worker $config_path`
    ok=_log_process(cmd,joinpath(dir,"solver.log"),progress,log,"cavitation",Float64(alpha);
        physical_seconds_per_ctu=cfg["chord_m"]/cfg["speed_m_s"],
        residual_kind=projection ? "volume_flux_divergence" : "mass_rate")
    ok || throw(ErrorException("nonlinear underwater worker failed; see solver.log"))
    meta=TOML.parsefile(joinpath(dir,"result.toml"))
    mesh=AirfoilWire.decode(read(joinpath(dir,"mesh_preview.msgpack")),:msgpack)
    mesh["artifact"]=joinpath(dir,"mesh_full.msgpack")
    mesh["preview_artifact"]=joinpath(dir,"mesh_preview.msgpack")
    field=AirfoilWire.decode(read(joinpath(dir,"field_full.msgpack")),:msgpack)
    cp=Dict{String,Any}("upper"=>Any[],"lower"=>Any[],"kind"=>"instantaneous pressure minus hydrostatic reference")
    for row in _csv_rows(joinpath(dir,"surface_cp.csv"))
        side=row["surface"];haskey(cp,side) || continue
        push!(cp[side],[_float(row["x"]),_float(row["cp"])])
    end
    history=Any[]
    for raw in _csv_rows(joinpath(dir,"iteration_history.csv"))
        row=Dict{String,Any}("iteration"=>parse(Int,raw["iteration"]))
        for k in ("residual_linf","residual_scaled","residual_l2","cl","cd","cm","mass_net_frac","t_ctu","t_seconds","dt_seconds","pressure_residual","volume_closure")
            haskey(raw,k) || continue
            row[k]=_float(raw[k])
        end
        row["residual"]=row["residual_scaled"];push!(history,row)
    end
    diagnostics=copy(meta)
    diagnostics["time_mode"]="physical";diagnostics["clock"]="physical seconds"
    diagnostics["formulation"]=cfg["formulation"]
    diagnostics["residual_kind"]=projection ? "volume_flux_divergence" : "mass_rate"
    diagnostics["residual_quantity"]=get(meta,"residual_quantity","mass rate; kg/m³/s")
    diagnostics["elapsed_seconds"]=meta["elapsed_seconds_physical"]
    diagnostics["requested_duration_seconds"]=meta["duration_seconds_requested"]
    diagnostics["requested_duration_ctu"]=cfg["duration_seconds"]*cfg["speed_m_s"]/cfg["chord_m"]
    diagnostics["physical_duration_reached"]=meta["duration_reached"]
    diagnostics["field_time_scope"]=projection ? "instantaneous inviscid physical state; no developed wave or polar qualification" : "instantaneous acoustic startup; no developed wave or polar qualification"
    status=get(meta,"minimum_pressure_pa",0.)>0 ? "experimental_physical_state" : "invalid_physical_state"
    Dict{String,Any}("backend"=>"cavitation","alpha_deg"=>alpha,"status"=>status,
        "cl"=>meta["cl"],"cd"=>meta["cd"],"cm"=>meta["cm"],"convergence"=>diagnostics["field_time_scope"],
        "validation"=>scope,"coefficient_method"=>meta["coefficient_method"],
        "native_geometry"=>Dict("x"=>mesh["wall_x"],"y"=>mesh["wall_y"],"units"=>"m"),
        "mesh"=>mesh,"cp"=>cp,"iteration_history"=>history,"diagnostics"=>diagnostics,
        "fields"=>Dict("kind"=>"samples","samples"=>field["samples"],"sample_shape"=>[meta["sampled_ni"],meta["sampled_nj"]],
            "sample_stride"=>1,"coordinate_units"=>"m","units"=>Dict("pressure"=>"Pa","density"=>"kg/m³","u"=>"m/s","v"=>"m/s","temperature"=>"K","hydrostatic_pressure"=>"Pa","dynamic_pressure"=>"Pa"),
            "full_artifact"=>joinpath(dir,"field_full.msgpack"),"full_cell_count"=>meta["full_field_cells"],
            "mesh_sha256"=>meta["full_field_mesh_sha256"],"full_iteration"=>meta["steps"]),
        "artifacts"=>Dict("log"=>joinpath(dir,"solver.log"),"result"=>joinpath(dir,"result.toml"),
            "iteration_csv"=>joinpath(dir,"iteration_history.csv"),"mesh_full"=>joinpath(dir,"mesh_full.msgpack"),
            "field_full"=>joinpath(dir,"field_full.msgpack"),"cp_csv"=>joinpath(dir,"surface_cp.csv")),
        "notes"=>[scope,
            "The resolved air-water interface evolves nonlinearly. Vapor nuclei are transported without evaporation or condensation.",
            "This annular development mesh is independent of the transonic C+H mesh. Slip-wall pressure loads exclude viscosity and are not stall predictions.",
            "Reaching max_steps is not reaching the requested physical duration; inspect actual seconds. Pressure projection solves an incompressible equation set; it does not accelerate a compressible clock.",
            projection ? "The pressure outlet and air roof are not nonreflecting wave boundaries. Instantaneous inviscid loads remain unqualified." : "Incoming hydrostatic reservoir and extrapolated outlet can reflect waves. Large initial acoustic loads are unqualified startup transients."])
end
