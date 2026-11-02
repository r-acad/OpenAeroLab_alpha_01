# Native case integration. Uses the same setup_multilevel_domain as LBM, then
# retains only fluid leaf cells for thermal finite volumes. No LBM populations
# are uploaded, evolved, or interpreted as thermal conserved variables.
isdefined(@__MODULE__,:PondLBMMesh) || include("pond_lbm_mesh.jl")
isdefined(@__MODULE__,:PondThermalFV) || include("pond_thermal_fv.jl")
isdefined(@__MODULE__,:PondThermalIO) || include("pond_thermal_io.jl")
using .PondLBMMesh, .PondThermalFV
using Serialization, TOML, Dates, Printf

function _thermal_boundary_codes(symmetric,ground,wall)
    yminus=ground ? (wall=="noslip" ? POND_NOSLIP : POND_SLIP) : symmetric ? POND_SLIP : POND_FARFIELD
    (POND_FARFIELD,POND_FARFIELD,yminus,POND_FARFIELD,POND_FARFIELD,POND_FARFIELD)
end

function _thermal_full_loads(force,moment,origin,symmetric)
    symmetric || return (;force,moment)
    # Reflection through Y=0, expressed about the same arbitrary moment origin.
    (;force=[2force[1],0.,2force[3]],moment=[-2origin[2]*force[3],2moment[2],2origin[2]*force[1]])
end

function _thermal_csv(path,header,row)
    mkpath(dirname(path));fresh=!isfile(path)
    open(path,"a") do io
        fresh && println(io,join(header,','))
        println(io,join(row,','));flush(io)
    end
end

function _thermal_pngs(files,folder)
    for path in files
        header,arrays=PondThermalIO.LBMSliceFormat.read_lbmslice(path)
        h=header["spacing_m"];ox,oy=header["origin_m"]
        for field in ("mach","pressure")
            a=arrays[field];x=ox .+ ((1:size(a,1)).-.5).*h;y=oy .+ ((1:size(a,2)).-.5).*h
            fig=Plots.heatmap(x,y,permutedims(a);aspect_ratio=:equal,
                xlabel=header["axis1"]*" (m)",ylabel=header["axis2"]*" (m)",
                title="Experimental thermal FV | $field | step $(header["step"])",
                color=:viridis,size=(1000,700))
            Plots.savefig(fig,joinpath(folder,replace(basename(path),".lbmslice"=>"_$field.png")))
        end
    end
end

function run_pond_thermal_case!(runtime_selection,output_claim,sim_start_datetime)
    CaseIDContract.verify_runtime_inputs(STARTUP_SELECTION.root_dir,runtime_selection)
    CaseIDContract.verify_claimed_output(runtime_selection,output_claim)
    started=time();outdir=output_claim.target.path;mkpath(outdir)
    cfg=safe_get(CFG,"advanced","numerics","pond_thermal";default=Dict())
    gamma=Float64(get(cfg,"gamma",1.4));temperature=Float64(get(cfg,"temperature_k",288.15))
    gas_constant=Float64(get(cfg,"gas_constant",287.05));cfl=Float64(get(cfg,"cfl",.25))
    prandtl=Float64(get(cfg,"prandtl",safe_get(CFG,"advanced","thermal","prandtl";default=.72)))
    isfinite(prandtl) && prandtl>0 || error("Thermal kinetic FV Prandtl number must be positive and finite")
    1<gamma<=5/3 && 100<=temperature<=2000 && gas_constant>0 || error("Invalid thermal ideal-gas parameters")
    sound=sqrt(gamma*gas_constant*temperature);mach=FLOW_VELOCITY/sound
    .05<=mach<=1.5 || error("Experimental thermal FV admits freestream Mach 0.05 to 1.5")
    wall=String(get(cfg,"wall_model","slip"));wall in ("slip","noslip") || error("wall_model must be slip or noslip")
    thermal_wall=lowercase(String(safe_get(CFG,"advanced","thermal","wall_bc";default="adiabatic")))
    thermal_wall=="adiabatic" || error("Thermal kinetic FV currently supports adiabatic walls; requested thermal.wall_bc=$thermal_wall")
    outlet=lowercase(String(safe_get(CFG,"advanced","boundary","outlet_mode";default="extrapolate")))
    outlet in ("extrapolate","characteristic") || error("Thermal kinetic FV uses characteristic freestream/outflow boundaries; outlet_mode=$outlet is not supported")
    zwalls=lowercase(String(safe_get(CFG,"advanced","boundary","z_walls";default="legacy")))
    zwalls=="legacy" || error("Thermal kinetic FV does not yet implement explicit z_walls=$zwalls; use its free-air/ground/symmetry domain options")
    INITIAL_CONDITION_TYPE==:uniform || error("The native thermal FV runner currently supports uniform/ramped freestream initialization, not advanced.initial_condition.type=$(INITIAL_CONDITION_TYPE). Custom verification states use the ThermalFV API.")
    uniform=safe_get(CFG,"advanced","initial_condition","uniform";default=Dict())
    all(k->!haskey(uniform,k),("ux","uy","uz","temperature","T")) && get(uniform,"rho",1)==1 ||
        error("Custom native uniform density, velocity or temperature is not supported by thermal FV; use its freestream inputs or the ThermalFV API")
    !get(uniform,"use_ramp_start_velocity",false) && get(uniform,"use_freestream_velocity",true) ||
        error("Thermal FV startup is selected by its ramp_steps setting; legacy initial-velocity overrides are unsupported")
    mu=wall=="noslip" ? FLUID_KINEMATIC_VISCOSITY/sound : 0.
    wall=="noslip" && mu<=0 && error("Laminar no-slip requires positive molecular viscosity")
    println("[ThermalFV] Experimental analytic Maxwellian KFVS; total energy; shared AMR face fluxes.")
    println("[ThermalFV] Wall approximation: staircase voxel faces; $(wall=="slip" ? "inviscid slip" : "laminar no-slip, no turbulence model").")
    println("[ThermalFV] Outer faces use characteristic freestream/outflow; Y-min uses the selected symmetry/ground condition. LBM relaxation, collision and sponge settings do not apply to FV evolution.")
    println("[ThermalFV] Building the native LBM Cartesian hierarchy on CPU...");flush(stdout)
    grids,geometry,_,_,_=setup_multilevel_domain(runtime_selection,output_claim;num_levels=NUM_LEVELS_CONFIG)
    params=get_domain_params()
    export_amr_viz(grids,params.mesh_offset,outdir;mesh=geometry,case_name=basename(CASE_DIR))
    codes=_thermal_boundary_codes(SYMMETRIC_ANALYSIS,GROUND_ANALYSIS,wall)
    periodic=(false,SPAN_PERIODIC_Y,false)
    solid=wall=="slip" ? POND_SLIP : POND_NOSLIP
    mesh=build_pond_lbm_mesh(grids;boundary_codes=Tuple(codes),solid_boundary=solid,periodic)
    nc=length(mesh.x);nf=length(mesh.owner)
    @printf("[ThermalFV] Native cells reused: %d fluid leaves, %d shared face patches, %d levels.\n",nc,nf,length(grids))
    index=build_point_index(mesh);findcell=(x,y,z)->find_leaf(index,x,y,z)
    surface_map=PondThermalIO.surface_sample_map(mesh,geometry,params.mesh_offset,findcell)
    # The leaf mesh owns its block-local index; discard all LBM distribution arrays.
    grids=nothing;GC.gc()
    device=lowercase(get(ENV,"POND_THERMAL_DEVICE","auto"))
    device in ("auto","cpu","cuda") || error("POND_THERMAL_DEVICE must be auto, cpu, or cuda")
    gpu=device!="cpu" && CUDA.functional()
    device=="cuda" && !gpu && error("CUDA requested but unavailable")
    backend=gpu ? CUDA.CUDABackend() : KernelAbstractions.CPU()
    gpu && CUDA.allowscalar(false)
    dev=gpu ? CUDA.CuArray : copy
    if gpu
        # This allowance includes face topology, gradients, RK buffers and fluxes.
        # Aero Lab applies the same coefficient to all allocated (not just leaf) cells.
        needed=Int128(1800)*nc
        free=CUDA.available_memory()
        needed<.9free || error("Thermal FV needs approximately $(round(Float64(needed)/1e9;digits=2)) GB free for this mesh; available $(round(free/1e9;digits=2)) GB. Increase surface cell size.")
    end
    ramp_steps=max(0,Int(safe_get(CFG,"basic","simulation","ramp_steps";default=0)))
    initial=ramp_steps>0 ? ((i,x,y,z)->(1.,0.,0.,0.,1/gamma)) : nothing
    s=ThermalFV(mesh;backend,dev,gamma,freestream=(1.,ramp_steps>0 ? 0. : mach,0.,0.,1/gamma),initial,mu,prandtl,cfl,periodic)
    scales=(rho=Float64(params.rho_physical),sound=sound,speed=Float64(params.u_physical),gamma=gamma,gas_constant=gas_constant)
    nsteps=STEPS;nsteps>0 || error("Thermal FV requires a positive iteration count")
    every=max(1,DIAG_FREQ);field_every=max(1,SLICE_OUTPUT_FREQ)
    vtk_enabled=OUTPUT_FREQ<=nsteps
    planes=LBMIO_SLICE_PLANES===nothing ? lbmio_legacy_planes() : LBMIO_SLICE_PLANES
    planes=[(;tag=p.tag,plane=p.plane==:default ? (GROUND_ANALYSIS ? :xy : :xz) : p.plane,coord=p.coord) for p in planes]
    csvdir=joinpath(outdir,"csv");vtkdir=joinpath(outdir,"VTK");mkpath(vtkdir)
    origin=Tuple(Float64.(params.moment_center))
    summary=Dict{String,Any}("solver"=>"thermal_kinetic_fv","flux"=>"analytic Maxwellian half-space KFVS",
        "experimental"=>true,"converged"=>false,"convergence_assessed"=>false,"status"=>"running",
        "wall_model"=>wall,"wall_geometry"=>"staircase voxel; not a cut-cell boundary",
        "turbulence_model"=>"none","gamma"=>gamma,"prandtl"=>prandtl,"mach"=>mach,"temperature_k"=>temperature,
        "cells"=>nc,"faces"=>nf,"backend"=>gpu ? "CUDA" : "CPU","num_levels"=>NUM_LEVELS_CONFIG,
        "gpu_array_bytes"=>gpu ? memory_bytes(s) : 0,"state_array_bytes"=>memory_bytes(s),
        "reference_area_m2"=>params.reference_area,"reference_chord_m"=>params.reference_chord,
        "ramp_steps"=>ramp_steps,"startup"=>ramp_steps>0 ? "initially at rest; incoming freestream velocity ramps linearly" : "uniform freestream",
        "moment_origin_m"=>collect(origin),"steps_requested"=>nsteps,"mesh"=>mesh.metadata,
        "load_convention"=>SYMMETRIC_ANALYSIS ? "mirrored full-model loads divided by full reference area" : "solved-domain loads divided by reference area",
        "limitations"=>"Experimental inviscid/laminar equations on voxel walls. No turbulence model or automatic steady-convergence declaration. Shock and boundary-layer grid convergence remain necessary.")
    write_provenance(outdir,basename(CASE_DIR),CFG;extra=Dict("solver"=>"thermal_kinetic_fv","experimental"=>true))
    totals0=conserved_totals(s);previous=Array(s.q);last_cd=0.;last_report=time()
    function save_summary(status)
        summary["status"]=status;summary["steps"]=s.steps;summary["time_s"]=s.time/sound
        summary["wall_seconds"]=time()-started;summary["rejected_steps"]=s.rejected_steps
        summary["conserved_totals_initial"]=totals0;summary["conserved_totals_final"]=conserved_totals(s)
        summary["note"]="Physical-time integration; completion means the requested steps finished, not that a steady solution converged."
        lbmio_write_build_summary_atomic(joinpath(outdir,"thermal_fv_summary.toml"),summary)
    end
    function save_fields(;volume=false,fields=true)
        residual!(s);KernelAbstractions.synchronize(backend)
        P=primitives(s);time_s=s.time/sound
        surface=@sprintf("surface_%06d.lbmp",s.steps)
        if fields
            PondThermalIO.write_surface(joinpath(vtkdir,surface),geometry,params.mesh_offset,surface_map,P,scales;
                step=s.steps,time_s,moment_origin=origin,reference_area=params.reference_area,reference_chord=params.reference_chord)
            if get(cfg,"export_surface_vtk",true)
                PondThermalIO.write_surface_vtk(joinpath(vtkdir,@sprintf("surface_%06d",s.steps)),geometry,params.mesh_offset,surface_map,P,scales;step=s.steps,time_s)
            end
            if LBMIO_EXPORT_DATA || LBMIO_SLICE_PNG
                folder=joinpath(outdir,"SLICES",@sprintf("step_%06d",s.steps))
                files=PondThermalIO.write_planes(folder,mesh,P,scales,planes,findcell;step=s.steps,time_s,offset=params.mesh_offset,max_cells=min(LBMIO_EXPORT_MAX_CELLS,262144),gradient=Array(s.gradient))
                LBMIO_SLICE_PNG && _thermal_pngs(files,folder)
            end
        end
        volume && PondThermalIO.write_volume(joinpath(vtkdir,@sprintf("flow_%06d",s.steps)),mesh,P,scales;step=s.steps,time_s,gradient=Array(s.gradient))
        saved=fields ? (volume ? "surface, recorded planes and volume" : "surface and recorded fields") : "volume"
        println("[ThermalFV] Saved $saved at step $(s.steps), t=$(round(time_s;digits=7)) s");flush(stdout)
    end
    function report()
        residual!(s);KernelAbstractions.synchronize(backend)
        flux=Array(s.flux);Q=Array(s.q);P=primitives(s)
        loads=PondThermalIO.wall_loads(mesh,flux,scales,origin)
        qref=.5scales.rho*scales.speed^2*params.reference_area
        force=loads.force;moment=loads.moment
        summary["solved_domain_force_N"]=copy(force);summary["solved_domain_moment_Nm"]=copy(moment)
        force,moment=_thermal_full_loads(force,moment,origin,SYMMETRIC_ANALYSIS)
        cd=force[1]/qref;cl=force[GROUND_ANALYSIS ? 2 : 3]/qref
        cs=force[GROUND_ANALYSIS ? 3 : 2]/qref
        eq=zeros(5);change=zeros(5)
        for i in 1:nc,m in 1:5
            v=0.
            for k in mesh.cell_face_offsets[i]:mesh.cell_face_offsets[i+1]-1
                sf=mesh.cell_faces[k];v+=sign(sf)*flux[abs(sf),m]
            end
            eq[m]+=v*v/mesh.volume[i]
            change[m]+=abs2(Q[i,m]-previous[i,m])*mesh.volume[i]
        end
        normscale=sqrt(sum(mesh.volume));eq=sqrt.(eq)./normscale;change=sqrt.(change)./normscale
        elapsed=time()-started
        _thermal_csv(joinpath(csvdir,"convergence.csv"),
            ("Step","Time_phys_s","Residual_mass","Residual_mx","Residual_my","Residual_mz","Residual_energy","Change_mass","Change_energy","CL","CD","Dt_s","Rejected_steps"),
            (s.steps,s.time/sound,eq...,change[1],change[5],cl,cd,s.last_dt/sound,s.rejected_steps))
        _thermal_csv(joinpath(csvdir,"forces.csv"),
            ("Step","Time_phys_s","Fx_N","Fy_N","Fz_N","Mx","My","Mz","Cd","Cl","Cs","Cmx","Cmy","Cmz"),
            (s.steps,s.time/sound,force...,moment...,cd,cl,cs,(moment./(qref*params.reference_chord))...))
        _thermal_csv(joinpath(csvdir,"resource_history.csv"),("Step","Elapsed_s","Iter_per_s","State_array_bytes","RAM_available_bytes"),
            (s.steps,elapsed,s.steps/max(elapsed,1e-9),memory_bytes(s),Sys.free_memory()))
        @printf("[ThermalFV] step %d/%d t=%.7g s dt=%.4g s CL=%+.6f CD=%+.6f mass_RMS=%.4e energy_RMS=%.4e rejects=%d\n",s.steps,nsteps,s.time/sound,s.last_dt/sound,cl,cd,eq[1],eq[5],s.rejected_steps)
        flush(stdout);previous=Q;last_cd=cd;last_report=time()
        summary["last_residuals"]=eq;summary["last_forces_N"]=force;summary["last_moments_Nm"]=moment
        summary["CL"]=cl;summary["CD"]=cd;summary["CS"]=cs
        save_summary("running")
    end
    try
        report();save_fields()
        for _ in 1:nsteps
            if ramp_steps>0
                T=eltype(s.q);factor=min(1.,(s.steps+1)/ramp_steps)
                s.freestream=T.((1.,mach*factor,0.,0.,1/gamma))
            end
            advance!(s)
            (s.steps%every==0 || s.steps==nsteps || time()-last_report>15) && report()
            fields_due=s.steps%field_every==0 || s.steps==nsteps
            volume_due=vtk_enabled && (s.steps%OUTPUT_FREQ==0 || s.steps==nsteps)
            (fields_due || volume_due) && save_fields(volume=volume_due,fields=fields_due)
        end
        # Restart/inspection state in the permanent case output, with enough
        # topology and scales to identify the exact mesh used by the fields.
        checkpoint=joinpath(outdir,"thermal_state.jls")
        open(checkpoint*".partial","w") do io
            serialize(io,(version=1,solver="thermal_kinetic_fv",mesh=mesh,conserved=Array(s.q),scales=scales,time=s.time,steps=s.steps))
        end
        Base.Filesystem.rename(checkpoint*".partial",checkpoint)
        save_summary("completed_steps")
    catch err
        summary["error"]=sprint(showerror,err);save_summary("failed");rethrow()
    end
    println("[ThermalFV] Requested steps completed. Steady convergence was not assessed; inspect residuals and fields.")
    BatchOutcomeContract.pond_case_outcome(true;case_name=basename(CASE_DIR),re_number=params.re_number,
        ma_physical=mach,total_cells=nc,wall_time=time()-started,steps=s.steps,start_time=sim_start_datetime,
        end_time=Dates.now(),output_dir=outdir,cd_cv=last_cd,n_samples=1)
end
