module UnderwaterProjectionWorker
using TOML, Serialization, SHA, Printf
const STANDALONE=abspath(PROGRAM_FILE)==abspath(@__FILE__)
STANDALONE && (println("UNDERWATER_SETUP,Loading Julia mesh, material and projection definitions.");flush(stdout))
const _SOURCE_HELPER=joinpath(@__DIR__,"..","runtime","underwater_sources.jl")
const _SOURCE_HELPER_BYTES=read(_SOURCE_HELPER)
include(_SOURCE_HELPER)
using .UnderwaterSourceProvenance
const SOURCE_CLOSURE=capture_sources(source_paths(@__FILE__;projection=true))
SOURCE_CLOSURE.bytes[basename(_SOURCE_HELPER)]==_SOURCE_HELPER_BYTES || error("source provenance helper changed while loading")
include(joinpath(@__DIR__,"..","..","..","..","src","cmesh","cmesh_free_surface_projection.jl"))
if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    Base.include(CMeshFreeSurfaceProjection,joinpath(@__DIR__,"..","..","..","..","src","cmesh","cmesh_free_surface_projection_ka.jl"))
end
using .CMeshFreeSurfaceProjection
using .CMeshFreeSurfaceProjection.CMeshFreeSurfaceAirfoil
using .CMeshFreeSurfaceProjection.CMeshFreeSurfaceAirfoil.CMeshFreeSurface
include(joinpath(@__DIR__,"..","protocol.jl"));using .AirfoilWire
include(joinpath(@__DIR__,"..","full_field_artifact.jl"));using .FullFieldArtifact
const GPU_REQUESTED=abspath(PROGRAM_FILE)==abspath(@__FILE__) && length(ARGS)==1 && get(TOML.parsefile(ARGS[1]),"gpu",false)
if GPU_REQUESTED
    println("UNDERWATER_SETUP,Loading CUDA libraries and checking the selected runtime.");flush(stdout)
    @eval import CUDA
    include(joinpath(@__DIR__,"..","runtime","cuda_projection_triangular.jl"))
end
const SCOPE="Experimental incompressible three-constituent nonlinear free surface; conservative volume/momentum transport and adjoint pressure projection; no viscosity, SST, phase change, acoustic waves or stall validation"
setup(message)=(println("UNDERWATER_SETUP,",message);flush(stdout))
physical_duration_reached(elapsed,target)=target-elapsed<=64max(eps(target),eps(elapsed))

function constant_density_hydrostatic_reference(s)
    g=s.grid;wet=(1-g.vapor_seed)*s.phase_density[1]+g.vapor_seed*s.phase_density[2]
    [101325.0-g.gravity*(g.yc[i,j]-g.water_level)*(g.yc[i,j]<g.water_level ? wet : s.phase_density[3]) for i in 1:g.NI,j in 1:g.NJ]
end

function mesh_payload(g)
    Dict{String,Any}("kind"=>"structured","ni"=>g.NI+1,"nj"=>g.NJ+1,"full_ni"=>g.NI+1,"full_nj"=>g.NJ+1,
        "i_indices"=>collect(1:g.NI+1),"j_indices"=>collect(1:g.NJ+1),"x"=>vec(g.X),"y"=>vec(g.Y),
        "wall_x"=>g.X[:,1],"wall_y"=>g.Y[:,1],"wall_i_start"=>1,"wall_i_end"=>g.NI+1,
        "foil_surface_cells"=>g.body_surface_cells,"wake_cells_per_side"=>0,"normal_cells"=>g.NJ,
        "stride_i"=>1,"stride_j"=>1,"base_faces_per_side"=>1,"periodic_i"=>true,
        "units"=>"m","node_order"=>"j-major, i-fastest","resolution"=>"underwater development",
        "quality"=>Dict("min_corner_sine"=>g.min_corner_sine,"folded_cells"=>0,"nonconvex_cells"=>0),
        "geometry_metadata"=>Dict("mesh_topology"=>"periodic annular O-grid","body_surface_cells"=>g.body_surface_cells,
            "cp_station_coordinates"=>"body-chordx/c",
            "mesh_family"=>"inviscid body-fitted annular mesh; exact finite-TE wall","qualification"=>"development mesh, not admitted viscous C+H grid","units"=>"m"))
end

function export_fields(s,out,iteration;filename="field_full.msgpack")
    g=s.grid;rho=CMeshFreeSurfaceProjection._rho(s);vel=CMeshFreeSurfaceProjection._velocity(s)
    hydrostatic=constant_density_hydrostatic_reference(s)
    qref=max(.5s.phase_density[1]*g.speed^2,eps())
    function vort(i,j)
        im=mod1(i-1,g.NI);ip=mod1(i+1,g.NI);jm=max(1,j-1);jp=min(g.NJ,j+1)
        xi=g.xc[ip,j]-g.xc[im,j];yi=g.yc[ip,j]-g.yc[im,j];xj=g.xc[i,jp]-g.xc[i,jm];yj=g.yc[i,jp]-g.yc[i,jm];det=xi*yj-xj*yi
        abs(det)<1e-20 ? 0. : ((vel[ip,j,2]-vel[im,j,2])*yj-(vel[i,jp,2]-vel[i,jm,2])*yi+(vel[ip,j,1]-vel[im,j,1])*xj-(vel[i,jp,1]-vel[i,jm,1])*xi)/det
    end
    at(i,j)=(g.xc[i,j],g.yc[i,j],0.,s.pressure[i,j]/qref,hydrostatic[i,j]+s.pressure[i,j],rho[i,j],vel[i,j,1],vel[i,j,2],293.15,vort(i,j),0.)
    path=joinpath(out,filename)
    field=FullFieldArtifact.write_full_field(at,path,g.NI,g.NJ,joinpath(out,"mesh_full.msgpack"),max(1,iteration))
    payload=AirfoilWire.decode(read(path),:msgpack);payload["scope"]=SCOPE;payload["units"]="SI";payload["mach_scope"]="not applicable to incompressible constituent equations; zero is a display placeholder"
    arrays=payload["samples"]
    for (k,name) in enumerate(("beta_water","beta_vapor","beta_air"));arrays[name]=vec(s.mass[:,:,k]./s.phase_density[k]);end
    arrays["hydrostatic_pressure"]=vec(hydrostatic);arrays["dynamic_pressure"]=vec(s.pressure)
    write(path,AirfoilWire.encode(payload,:msgpack))
    columns=vcat(collect(FullFieldArtifact.COLUMNS),["beta_water","beta_air","beta_vapor","hydrostatic_pressure","dynamic_pressure"])
    if filename=="field_full.msgpack"
        open(joinpath(out,"field_samples.csv"),"w") do io
            println(io,join(columns,','))
            for n in 1:g.NI*g.NJ;println(io,join((arrays[k][n] for k in columns),','));end
        end
        open(joinpath(out,"surface_cp.csv"),"w") do io
            println(io,"surface,x,cp")
            for i in 1:g.NI
                side=i>g.body_surface_cells ? "base" : i<=g.body_surface_cells÷2 ? "lower" : "upper"
                x,_=body_chord_coordinates(g,.5(g.X[i,1]+g.X[i+1,1]),.5(g.Y[i,1]+g.Y[i+1,1]))
                println(io,side,',',x,',',s.pressure[i,1]/qref)
            end
        end
    end
    (;field,columns)
end

function run_underwater_projection_worker(config_path)
    entry_time=time();setup("Verifying loaded source snapshots and flow configuration.")
    verify_sources!(SOURCE_CLOSURE)
    cfg=TOML.parsefile(config_path);get(cfg,"formulation","incompressible_projection")=="incompressible_projection" || error("wrong projection formulation")
    out=abspath(cfg["outdir"]);mkpath(out);gpu=Bool(get(cfg,"gpu",false));m=MaterialModel()
    gpu && !isdefined(@__MODULE__,:CUDA) && error("CUDA must be loaded before entering the underwater worker")
    hashes=copy(SOURCE_CLOSURE.hashes);write_snapshot(SOURCE_CLOSURE,joinpath(out,"source_snapshot"))
    setup("Building the finite trailing-edge body mesh and hydrostatic reference.")
    g=build_airfoil_grid(;code=Int(get(cfg,"naca4_code",12)),geometry=haskey(cfg,"naca4_code") ? "" : String(get(cfg,"geometry","")),
        n_foil=Int(get(cfg,"n_foil",96)),n_normal=Int(get(cfg,"n_normal",24)),chord_m=Float64(get(cfg,"chord_m",1.)),
        submergence_chords=Float64(get(cfg,"submergence_chords",1.)),water_depth_chords=Float64(get(cfg,"water_depth_chords",4.)),
        air_height_chords=Float64(get(cfg,"air_height_chords",2.)),upstream_chords=Float64(get(cfg,"upstream_chords",3.)),
        downstream_chords=Float64(get(cfg,"downstream_chords",5.)),alpha_deg=Float64(get(cfg,"alpha_deg",0.)),
        speed_m_s=Float64(get(cfg,"speed_m_s",3.)),initial_vapor_fraction=Float64(get(cfg,"initial_vapor_fraction",1e-6)),model=m)
    mesh=mesh_payload(g)
    for name in ("mesh_full.msgpack","mesh_preview.msgpack");write(joinpath(out,name),AirfoilWire.encode(mesh,:msgpack));end
    println("MESH_READY,mesh_preview.msgpack");flush(stdout)
    duration=Float64(get(cfg,"duration_seconds",.01));maxsteps=Int(get(cfg,"max_steps",2000));cfl=Float64(get(cfg,"cfl",.35))
    dtmax=Float64(get(cfg,"dt_max",.0001));stride=Int(get(cfg,"history_stride",20));frame_stride=Int(get(cfg,"frame_stride",200))
    duration>0 && maxsteps>=1 && stride>=1 && frame_stride>=1 && dtmax>0 || error("invalid projection duration/steps/strides")
    initialization=Symbol(get(cfg,"initialization","source_potential"));outer_bc=Symbol(get(cfg,"outer_bc","open"))
    setup("Preparing the initial flow and CPU adjoint pressure operator.")
    s=build_projection(g,m;outer_bc,initialization)
    hydrostatic=constant_density_hydrostatic_reference(s)
    setup(gpu ? "Initializing CUDA operators and pressure preconditioner." : "Initializing CPU pressure factorization.")
    ka=gpu ? build_projection_ka(CUDA.CUDABackend(),s;cuda=true,
        sparse_device=A->CUDA.CUSPARSE.CuSparseMatrixCSR(A),
        triangular_solve=make_cuda_projection_triangular_solver(),
        preconditioner=Symbol(get(cfg,"pressure_preconditioner","initial_cholesky"))) : nothing
    verify_sources!(SOURCE_CLOSURE)
    setup(gpu ? "CUDA operators ready on $(CUDA.name(CUDA.device())); starting physical time integration." : "CPU operators ready; starting physical time integration.")
    println("UNDERWATER_PROJECTION,",SCOPE);flush(stdout)
    setup_seconds=time()-entry_time
    initialmass=projection_mass(s);began=time();stop="max_steps";loads=projection_loads(s);frames=Any[];dt=0.;maxdiv=0.;maxclosure=0.
    export_fields(s,out,0;filename="field_initial.msgpack")
    open(joinpath(out,"iteration_history.csv"),"w") do io
        println(io,"iteration,residual_linf,residual_scaled,residual_l2,cl,cd,cm,mass_net_frac,t_ctu,t_seconds,dt_seconds,pressure_residual,volume_closure")
        for n in 1:maxsteps
            if isfile(joinpath(out,"STOP_REQUESTED"));stop="user_stop";break;end
            now=gpu ? ka.elapsed : s.elapsed
            if physical_duration_reached(now,duration);stop="duration_reached";break;end
            r=gpu ? projection_step_ka!(ka;dt_cap=min(dtmax,duration-now),cfl,
                pressure_rtol=Float64(get(cfg,"pressure_rtol",1e-10)),pressure_atol=Float64(get(cfg,"pressure_atol",1e-12)),
                pressure_max_iterations=Int(get(cfg,"pressure_max_iterations",4000))) : projection_step!(s;dt_cap=min(dtmax,duration-now),cfl)
            dt=r.dt;now=gpu ? ka.elapsed : s.elapsed
            reached=physical_duration_reached(now,duration)
            inspect=n==1 || n%stride==0 || n%frame_stride==0 || n==maxsteps || reached
            if inspect
                gpu && (s=projection_host(ka))
                loads=projection_loads(s);div=projection_divergence(s);raw=maximum(abs,div);scale=g.chord/max(g.speed,.01)
                scaled=raw*scale;l2=sqrt(sum(abs2,div)/length(div))*scale;closure=projection_volume_error(s)
                maxdiv=max(maxdiv,raw);maxclosure=max(maxclosure,closure)
                ledger=projection_mass(s)-initialmass+s.boundary_mass_integral
                net=maximum(abs,ledger)/max(sum(abs,initialmass),1.)
                println(io,join((n,raw,scaled,l2,loads.cl,loads.cd,loads.cm,net,s.elapsed*g.speed/g.chord,s.elapsed,dt,s.pressure_residual,closure),','));flush(io)
                println("ITERATION,",join((n,maxsteps,scaled,raw,l2,loads.cl,loads.cd,net),','))
                @printf("PHYSICAL_TIME,%d,%.12g,%.12g,%.12g\n",n,s.elapsed*g.speed/g.chord,duration*max(g.speed,.01)/g.chord,dt*max(g.speed,.01)/g.chord)
                @printf("UNDERWATER_PROJECTION_TIME n=%d physical=%.8g/%.8g s dt=%.4g s div=%.3e wall=%.1f s\n",n,s.elapsed,duration,dt,raw,time()-began);flush(stdout)
                minimum(hydrostatic.+s.pressure)>0 || (stop="negative_absolute_pressure_requires_cavitation_model";break)
            end
            if n%frame_stride==0 || n==maxsteps || reached
                filename="field_frame_"*lpad(string(n),8,'0')*".msgpack";export_fields(s,out,n;filename)
                push!(frames,Dict("filename"=>filename,"iteration"=>n,"time_seconds"=>s.elapsed))
                serialize(joinpath(out,"state.jls"),(;state=s,hashes))
            end
            if reached;stop="duration_reached";break;end
        end
    end
    gpu && (s=projection_host(ka));loads=projection_loads(s);field=export_fields(s,out,s.steps)
    finalmass=projection_mass(s);balance=finalmass-initialmass+s.boundary_mass_integral
    changed=!sources_current(SOURCE_CLOSURE)
    changed && error("projection source changed during run; result refused")
    meta=Dict{String,Any}("backend"=>"cavitation","solver"=>"Nonlinear incompressible three-constituent adjoint pressure projection",
        "formulation"=>"incompressible_projection","qualification"=>SCOPE,"status"=>"experimental_physical_state","stop_reason"=>stop,
        "cl"=>loads.cl,"cd"=>loads.cd,"cm"=>loads.cm,"coefficient_method"=>"instantaneous inviscid dynamic pressure force; hydrostatic reference excluded",
        "cp_station_coordinates"=>"body-chordx/c","moment_reference"=>Dict("body_x_c"=>.25,"body_y_c"=>0.,"world_x_m"=>g.moment_x,"world_y_m"=>g.moment_y),
        "clock"=>"physical time","steps"=>s.steps,"max_steps"=>maxsteps,"elapsed_seconds_physical"=>s.elapsed,
        "duration_seconds_requested"=>duration,"duration_reached"=>physical_duration_reached(s.elapsed,duration),"last_dt_seconds"=>dt,"elapsed_ctu"=>s.elapsed*g.speed/g.chord,
        "wall_seconds"=>time()-began,"gpu"=>gpu,"gpu_pressure_solver"=>gpu ? "native CUDA sparse matrix-free preconditioned CG of B M^-1 B^T" : "CPU sparse Cholesky of B M^-1 B^T",
        "pressure_preconditioner"=>gpu ? ka.preconditioner_name : "direct CPU Cholesky",
        "converged"=>false,"stall_qualified"=>false,"fluid"=>"water_vapor_air","domain"=>"underwater","alpha_deg"=>get(cfg,"alpha_deg",0.),
        "chord_m"=>g.chord,"speed_m_s"=>g.speed,"submergence_chords"=>get(cfg,"submergence_chords",1.),"gravity_m_s2"=>g.gravity,
        "phase_density_order"=>["water","vapor","air"],"constant_phase_densities_kg_m3"=>collect(s.phase_density),
        "phase_change"=>false,"viscosity"=>false,"surface_tension"=>false,"turbulence"=>false,"energy_equation"=>false,
        "acoustic_waves"=>false,"mach_scope"=>"not applicable to incompressible constituent equations; no reduced sound-speed model",
        "initial_vapor_volume_fraction_in_water"=>g.vapor_seed,"vapor_source"=>"independent conservative transport of initial nuclei; no vapor production or compressible bubble expansion",
        "hydrostatic_reference"=>"piecewise linear from constant constituent densities and the prescribed initial water/vapor mixture",
        "prescribed_quantities"=>Dict("temperature_kelvin"=>293.15),"unavailable_quantities"=>["mach","nut"],
        "boundary_conditions"=>Dict("body"=>"impermeable slip wall; exact zero constituent flux","floor"=>"slip wall",
            "inlet"=>"prescribed hydrostatic-composition reservoir velocity","outlet_and_air_roof"=>"fixed dynamic pressure; not a nonreflecting wave boundary"),
        "mass_integrals_order"=>["liquid","vapor","air"],"initial_mass_integrals"=>initialmass,"final_mass_integrals"=>finalmass,
        "integrated_outward_mass"=>s.boundary_mass_integral,"mass_balance_errors"=>balance,"relative_mass_balance_errors"=>balance./max.(abs.(initialmass),1.),
        "wall_volume_flux_max_m2_s"=>maximum(abs,s.qj[:,1]),
        "wall_constituent_mass_flux_max_kg_m_s"=>maximum(abs(s.qj[i,1])*s.mass[i,1,k] for i in 1:g.NI,k in 1:3),"minimum_pressure_pa"=>minimum(hydrostatic.+s.pressure),
        "residual_quantity"=>"divergence of volume flux; 1/s","residual_scaled"=>maximum(abs,projection_divergence(s))*g.chord/max(g.speed,.01),
        "maximum_volume_divergence_linf_s_inv"=>maxdiv,"maximum_volume_closure_error"=>maxclosure,"pressure_operator_residual"=>s.pressure_residual,
        "state_format"=>"IncompressibleConstituentProjection/1","source_sha256"=>hashes,"source_changed_during_run"=>changed,
        "source_capture_stage"=>"before source includes; verified after module load and before time integration",
        "setup_wall_seconds"=>setup_seconds,"total_wall_seconds"=>time()-entry_time,
        "field_frames"=>frames,"full_field_cells"=>g.NI*g.NJ,"full_field_mesh_sha256"=>field.field["mesh_sha256"],
        "sampled_ni"=>g.NI,"sampled_nj"=>g.NJ,"sample_stride"=>1,"field_columns"=>field.columns,"geometry"=>mesh["geometry_metadata"])
    open(joinpath(out,"result.toml"),"w") do io;TOML.print(io,meta);end
    serialize(joinpath(out,"state.jls"),(;state=s,hashes))
    println("UNDERWATER_PROJECTION_COMPLETE stop=",stop," actual_physical_seconds=",s.elapsed," steps=",s.steps);flush(stdout)
    meta
end
verify_sources!(SOURCE_CLOSURE)
end
if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    length(ARGS)==1 || error("usage: underwater_projection_worker.jl CONFIG.toml")
    UnderwaterProjectionWorker.run_underwater_projection_worker(ARGS[1])
end
