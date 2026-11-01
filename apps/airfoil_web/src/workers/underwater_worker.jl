module UnderwaterAirfoilWorker

using TOML, Serialization, SHA, Printf, Dates
const STANDALONE=abspath(PROGRAM_FILE)==abspath(@__FILE__)
STANDALONE && (println("UNDERWATER_SETUP,Loading Julia mesh, material and flow definitions.");flush(stdout))
const _SOURCE_HELPER=joinpath(@__DIR__,"..","runtime","underwater_sources.jl")
const _SOURCE_HELPER_BYTES=read(_SOURCE_HELPER)
include(_SOURCE_HELPER)
using .UnderwaterSourceProvenance
const SOURCE_CLOSURE=capture_sources(source_paths(@__FILE__))
SOURCE_CLOSURE.bytes[basename(_SOURCE_HELPER)]==_SOURCE_HELPER_BYTES || error("source provenance helper changed while loading")
include(joinpath(@__DIR__,"..","..","..","..","src","cmesh","cmesh_free_surface_airfoil.jl"))
if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    Base.include(CMeshFreeSurfaceAirfoil,joinpath(@__DIR__,"..","..","..","..","src","cmesh","cmesh_free_surface_airfoil_ka.jl"))
end
using .CMeshFreeSurfaceAirfoil
using .CMeshFreeSurfaceAirfoil.CMeshFreeSurface
include(joinpath(@__DIR__,"..","protocol.jl"))
using .AirfoilWire
include(joinpath(@__DIR__,"..","full_field_artifact.jl"))
using .FullFieldArtifact

const SCOPE="Experimental inviscid, barotropic three-constituent nonlinear free-surface airfoil; no viscosity, SST, energy equation, phase change or stall validation"
const GPU_REQUESTED=abspath(PROGRAM_FILE)==abspath(@__FILE__) && length(ARGS)==1 && get(TOML.parsefile(ARGS[1]),"gpu",false)
if GPU_REQUESTED
    println("UNDERWATER_SETUP,Loading CUDA libraries and checking the selected runtime.");flush(stdout)
    @eval import CUDA
end
setup(message)=(println("UNDERWATER_SETUP,",message);flush(stdout))

function mesh_payload(g)
    Dict{String,Any}("kind"=>"structured","ni"=>g.NI+1,"nj"=>g.NJ+1,
        "full_ni"=>g.NI+1,"full_nj"=>g.NJ+1,"i_indices"=>collect(1:g.NI+1),"j_indices"=>collect(1:g.NJ+1),
        "x"=>vec(g.X),"y"=>vec(g.Y),"wall_x"=>g.X[:,1],"wall_y"=>g.Y[:,1],
        "wall_i_start"=>1,"wall_i_end"=>g.NI+1,"foil_surface_cells"=>g.NI,
        "wake_cells_per_side"=>0,"normal_cells"=>g.NJ,"stride_i"=>1,"stride_j"=>1,
        "base_faces_per_side"=>1,"node_order"=>"j-major, i-fastest","periodic_i"=>true,
        "units"=>"m","resolution"=>"underwater development",
        "quality"=>Dict("min_corner_sine"=>g.min_corner_sine,"folded_cells"=>0,"nonconvex_cells"=>0),
        "geometry_metadata"=>Dict("mesh_family"=>"body-fitted annular O-grid; exact finite-TE wall","mesh_topology"=>"periodic annular O-grid",
            "cp_station_coordinates"=>"body-chordx/c",
            "body_surface_cells"=>g.body_surface_cells,"wall_cells_including_base"=>g.NI,"units"=>"m",
            "qualification"=>"inviscid development mesh; not the admitted viscous C+H mesh"))
end

function _vorticity(U,g,i,j)
    im=mod1(i-1,g.NI);ip=mod1(i+1,g.NI);jm=max(1,j-1);jp=min(g.NJ,j+1)
    qm=primitive(ntuple(k->U[im,j,k],5));qp=primitive(ntuple(k->U[ip,j,k],5))
    qjm=primitive(ntuple(k->U[i,jm,k],5));qjp=primitive(ntuple(k->U[i,jp,k],5))
    xi=g.xc[ip,j]-g.xc[im,j];yi=g.yc[ip,j]-g.yc[im,j]
    xj=g.xc[i,jp]-g.xc[i,jm];yj=g.yc[i,jp]-g.yc[i,jm];det=xi*yj-xj*yi
    abs(det)<1e-20 ? 0. : ((qp.v-qm.v)*yj-(qjp.v-qjm.v)*yi+(qp.u-qm.u)*xj-(qjp.u-qjm.u)*xi)/det
end

function export_fields(U,g,m,out,iteration;filename="field_full.msgpack")
    qref=max(.5m.rho_l_ref*g.speed^2,eps())
    function at(i,j)
        state=ntuple(k->U[i,j,k],5);q=primitive(state,m)
        (g.xc[i,j],g.yc[i,j],hypot(q.u,q.v)/q.c,(q.p-g.pc0[i,j])/qref,
            q.p,q.rho,q.u,q.v,293.15,_vorticity(U,g,i,j),0.)
    end
    path=joinpath(out,filename)
    field=FullFieldArtifact.write_full_field(at,path,g.NI,g.NJ,joinpath(out,"mesh_full.msgpack"),max(1,iteration))
    payload=AirfoilWire.decode(read(path),:msgpack)
    payload["scope"]=SCOPE;payload["units"]="SI";payload["time_scope"]="instantaneous physical state"
    arrays=payload["samples"]
    arrays["beta_water"]=[primitive(ntuple(k->U[i,j,k],5),m).volume_fractions.liquid for j in 1:g.NJ for i in 1:g.NI]
    arrays["beta_air"]=[primitive(ntuple(k->U[i,j,k],5),m).volume_fractions.air for j in 1:g.NJ for i in 1:g.NI]
    arrays["beta_vapor"]=[primitive(ntuple(k->U[i,j,k],5),m).volume_fractions.vapor for j in 1:g.NJ for i in 1:g.NI]
    arrays["hydrostatic_pressure"]=[g.pc0[i,j] for j in 1:g.NJ for i in 1:g.NI]
    write(path,AirfoilWire.encode(payload,:msgpack))
    columns=vcat(collect(FullFieldArtifact.COLUMNS),["beta_water","beta_air","beta_vapor","hydrostatic_pressure"])
    if filename=="field_full.msgpack"
        open(joinpath(out,"field_samples.csv"),"w") do io
            println(io,join(columns,','))
            for n in 1:g.NI*g.NJ;println(io,join((arrays[k][n] for k in columns),','));end
        end
        open(joinpath(out,"surface_cp.csv"),"w") do io
            println(io,"surface,x,cp")
            for i in 1:g.NI
                state=CMeshFreeSurfaceAirfoil._face_state(ntuple(k->U[i,1,k],5),g.pc0[i,1],g.pj0[i,1],m)
                x,_=body_chord_coordinates(g,.5(g.X[i,1]+g.X[i+1,1]),.5(g.Y[i,1]+g.Y[i+1,1]))
                side=i>g.body_surface_cells ? "base" : i<=div(g.body_surface_cells,2) ? "lower" : "upper"
                println(io,side,',',x,',',(mixture_pressure(state,m)-g.pj0[i,1])/qref)
            end
        end
    end
    (;field,columns)
end

function run_underwater_worker(config_path::AbstractString)
    entry_time=time();setup("Verifying loaded source snapshots and flow configuration.")
    verify_sources!(SOURCE_CLOSURE)
    cfg=TOML.parsefile(config_path);out=abspath(cfg["outdir"]);mkpath(out)
    source_paths=SOURCE_CLOSURE.paths;hashes=copy(SOURCE_CLOSURE.hashes)
    write_snapshot(SOURCE_CLOSURE,joinpath(out,"source_snapshot"))
    gpu=Bool(get(cfg,"gpu",false))
    gpu && !isdefined(@__MODULE__,:CUDA) && error("CUDA must be loaded before entering the underwater worker")
    code=Int(get(cfg,"naca4_code",12));geometry=String(get(cfg,"geometry",""))
    m=MaterialModel()
    setup("Building the finite trailing-edge body mesh and hydrostatic reference.")
    g=build_airfoil_grid(;code,geometry,n_foil=Int(get(cfg,"n_foil",96)),n_normal=Int(get(cfg,"n_normal",24)),
        chord_m=Float64(get(cfg,"chord_m",1.)),submergence_chords=Float64(get(cfg,"submergence_chords",1.)),
        water_depth_chords=Float64(get(cfg,"water_depth_chords",4.)),air_height_chords=Float64(get(cfg,"air_height_chords",2.)),
        upstream_chords=Float64(get(cfg,"upstream_chords",3.)),downstream_chords=Float64(get(cfg,"downstream_chords",5.)),
        alpha_deg=Float64(get(cfg,"alpha_deg",0.)),speed_m_s=Float64(get(cfg,"speed_m_s",3.)),
        radial_growth=Float64(get(cfg,"radial_growth",1.22)),initial_vapor_fraction=Float64(get(cfg,"initial_vapor_fraction",1e-6)),model=m)
    mesh=mesh_payload(g)
    for name in ("mesh_full.msgpack","mesh_preview.msgpack");write(joinpath(out,name),AirfoilWire.encode(mesh,:msgpack));end
    println("MESH_READY,mesh_preview.msgpack");flush(stdout)
    duration=Float64(get(cfg,"duration_seconds",.002));steps=Int(get(cfg,"max_steps",2000))
    stride=Int(get(cfg,"history_stride",50));frame_stride=Int(get(cfg,"frame_stride",500))
    cfl=Float64(get(cfg,"cfl",.35));dt_cap=Float64(get(cfg,"dt_max",1.))
    duration>0 && steps>=1 && stride>=1 && frame_stride>=1 || error("positive underwater duration, max_steps and strides required")
    outer_bc=Symbol(get(cfg,"outer_bc","open"));outer_bc in (:wall,:open) || error("outer_bc must be open or wall")
    flux=String(get(cfg,"flux","thornber"));flux in ("hllc","thornber") || error("flux must be hllc or thornber")
    face_flux=flux=="thornber" ? CMeshFreeSurfaceAirfoil.CMeshFreeSurface.thornber_hllc_flux : hllc_flux
    initialization=String(get(cfg,"initialization","source_potential"))
    initialization in ("uniform","source_potential") || error("initialization must be uniform or source_potential")
    setup("Preparing the initial flow and conservative CPU workspace.")
    U,initialization_receipt=initialization=="source_potential" ? source_panel_initial_state(g,m) :
        (initial_state(g,m),Dict{String,Any}("method"=>"impulsive uniform velocity; acoustic startup may dominate"))
    w=AirfoilWorkspace(g);initial_mass=material_mass(U,g)
    setup(gpu ? "Initializing CUDA flow operators." : "Initializing CPU flow operators.")
    ka=gpu ? build_airfoil_ka(CUDA.CUDABackend(),U,g,m;outer_bc,flux=Symbol(flux)) : nothing
    verify_sources!(SOURCE_CLOSURE)
    setup(gpu ? "CUDA operators ready on $(CUDA.name(CUDA.device())); starting physical time integration." : "CPU operators ready; starting physical time integration.")
    println("UNDERWATER,",SCOPE);flush(stdout)
    setup_seconds=time()-entry_time
    elapsed=0.;actual=0;integrated_outflow=zeros(3);began=time();stop_reason="max_steps";loads=airfoil_loads(U,g,m)
    last_residual=0.;last_dt=0.;frames=Any[]
    export_fields(U,g,m,out,0;filename="field_initial.msgpack")
    open(joinpath(out,"iteration_history.csv"),"w") do io
        println(io,"iteration,residual_linf,residual_scaled,residual_l2,cl,cd,cm,mass_net_frac,t_ctu,t_seconds,dt_seconds")
        for n in 1:steps
            if isfile(joinpath(out,"STOP_REQUESTED"));stop_reason="user_stop";break;end
            step=gpu ? airfoil_step_ka!(ka;dt_cap=min(dt_cap,duration-elapsed),cfl) :
                airfoil_step!(U,w,g,m;dt_cap=min(dt_cap,duration-elapsed),cfl,outer_bc,face_flux)
            elapsed+=step.dt;last_dt=step.dt;actual=n
            integrated_outflow .+= step.dt.*step.outward_mass_rate
            if n==1 || n%stride==0 || n==steps || elapsed>=duration
                gpu && (U=Array(ka.U))
                loads=airfoil_loads(U,g,m)
                scale=m.rho_l_ref*max(g.speed,.01)/g.chord
                raw=step.residual;scaled=raw/scale
                residual_array=gpu ? Array(ka.residual) : w.residual
                l2=sqrt(sum(abs2,view(residual_array,:,:,1))/length(g.volume))/scale
                last_residual=scaled;net=sum(step.outward_mass_rate)/(m.rho_l_ref*max(g.speed,.01)*g.chord)
                t_ctu=elapsed*g.speed/g.chord
                println(io,join((n,raw,scaled,l2,loads.cl,loads.cd,loads.cm,net,t_ctu,elapsed,step.dt),','));flush(io)
                println("ITERATION,",join((n,steps,scaled,raw,l2,loads.cl,loads.cd,net),','));flush(stdout)
                @printf("PHYSICAL_TIME,%d,%.12g,%.12g,%.12g\n",n,t_ctu,duration*max(g.speed,.01)/g.chord,step.dt*max(g.speed,.01)/g.chord)
                @printf("UNDERWATER_TIME n=%d physical=%.9g/%.9g s dt=%.4g s wall=%.1f s\n",n,elapsed,duration,step.dt,time()-began);flush(stdout)
            end
            if n%frame_stride==0 || n==steps || elapsed>=duration
                gpu && (U=Array(ka.U))
                name="field_frame_"*lpad(string(n),8,'0')*".msgpack"
                export_fields(U,g,m,out,n;filename=name)
                push!(frames,Dict("filename"=>name,"iteration"=>n,"time_seconds"=>elapsed))
                serialize(joinpath(out,"state.jls"),(;U,elapsed,steps=n,hashes))
            end
            if elapsed>=duration;stop_reason="duration_reached";break;end
        end
    end
    gpu && (U=Array(ka.U))
    exported=export_fields(U,g,m,out,actual);final_mass=material_mass(U,g)
    wall_flux=gpu ? Array(ka.Fj)[:,1,[1,4,5]] : w.fj[:,1,[1,4,5]]
    wall_flux_max=maximum(abs,wall_flux)
    minimum_pressure=minimum(mixture_pressure(ntuple(k->U[i,j,k],5),m) for i in 1:g.NI,j in 1:g.NJ)
    source_changed=!sources_current(SOURCE_CLOSURE)
    source_changed && error("underwater source changed during calculation; result refused")
    mass_error=final_mass.-initial_mass.+integrated_outflow
    meta=Dict{String,Any}("backend"=>"cavitation","solver"=>"Nonlinear body-fitted three-constituent HLLC",
        "status"=>"experimental_physical_state","stop_reason"=>stop_reason,"qualification"=>SCOPE,
        "cl"=>loads.cl,"cd"=>loads.cd,"cm"=>loads.cm,"coefficient_method"=>"instantaneous inviscid pressure force minus exact discrete hydrostatic reference",
        "cp_station_coordinates"=>"body-chordx/c","moment_reference"=>Dict("body_x_c"=>.25,"body_y_c"=>0.,"world_x_m"=>g.moment_x,"world_y_m"=>g.moment_y),
        "hydrostatic_force_x_n_m"=>loads.hydrostatic_force_x,"hydrostatic_force_y_n_m"=>loads.hydrostatic_force_y,
        "steps"=>actual,"max_steps"=>steps,"clock"=>"physical time","elapsed_seconds_physical"=>elapsed,
        "duration_seconds_requested"=>duration,"duration_reached"=>elapsed>=duration,"last_dt_seconds"=>last_dt,
        "elapsed_ctu"=>elapsed*g.speed/g.chord,"wall_seconds"=>time()-began,"gpu"=>gpu,
        "residual_scaled"=>last_residual,"converged"=>false,"stall_qualified"=>false,
        "chord_m"=>g.chord,"speed_m_s"=>g.speed,"submergence_chords"=>get(cfg,"submergence_chords",1.),
        "fluid"=>"water_vapor_air","domain"=>"underwater","gravity_m_s2"=>g.gravity,
        "water_sound_speed_m_s"=>sqrt(m.c_l2),"rho_water_kg_m3"=>m.rho_l_ref,
        "phase_change"=>false,"viscosity"=>false,"surface_tension"=>false,"turbulence"=>false,"energy_equation"=>false,
        "initial_vapor_volume_fraction_in_water"=>g.vapor_seed,"vapor_source"=>"independent conservative transport of explicitly seeded nuclei; no phase change or vapor creation",
        "initialization"=>initialization_receipt,
        "face_flux"=>flux=="thornber" ? "HLLC with Thornber low-Mach velocity-jump correction; physical acoustic clock" : "HLLC",
        "boundary_conditions"=>Dict("body"=>"impermeable slip wall; exactly zero constituent mass flux",
            "outer"=>outer_bc===:wall ? "closed slip wall" : "hydrostatic reservoir on incoming flow; extrapolation on outgoing flow; not a nonreflecting wave outlet"),
        "initial_mass_integrals"=>initial_mass,"final_mass_integrals"=>final_mass,"integrated_outward_mass"=>integrated_outflow,
        "mass_balance_errors"=>mass_error,"relative_mass_balance_errors"=>mass_error./max.(abs.(initial_mass),1.),
        "wall_constituent_mass_flux_max_kg_m_s"=>wall_flux_max,"minimum_pressure_pa"=>minimum_pressure,
        "material_mass_order"=>["liquid","vapor","air"],"state_format"=>STATE_FORMAT,"source_sha256"=>hashes,
        "source_changed_during_run"=>source_changed,"field_frames"=>frames,"full_field_cells"=>g.NI*g.NJ,
        "source_capture_stage"=>"before source includes; verified after module load and before time integration",
        "setup_wall_seconds"=>setup_seconds,"total_wall_seconds"=>time()-entry_time,
        "full_field_mesh_sha256"=>exported.field["mesh_sha256"],"sampled_ni"=>g.NI,"sampled_nj"=>g.NJ,"sample_stride"=>1,
        "field_columns"=>exported.columns,"geometry"=>mesh["geometry_metadata"],"alpha_deg"=>get(cfg,"alpha_deg",0.))
    open(joinpath(out,"result.toml"),"w") do io;TOML.print(io,meta);end
    serialize(joinpath(out,"state.jls"),(;U,elapsed,steps=actual,hashes))
    println("UNDERWATER_COMPLETE stop=",stop_reason," actual_physical_seconds=",elapsed," steps=",actual);flush(stdout)
    meta
end

verify_sources!(SOURCE_CLOSURE)

end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    length(ARGS)==1 || error("usage: julia --project=src underwater_worker.jl CONFIG.toml")
    UnderwaterAirfoilWorker.run_underwater_worker(ARGS[1])
end
