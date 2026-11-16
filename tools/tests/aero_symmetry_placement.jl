# julia --project=third_party/JXfoil tools/tests/aero_symmetry_placement.jl
using Test, HTTP, JSON3
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
const SCRATCH=joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(ROOT,"..","02_PRIVATE_PROJECT_DEVELOPMENT")),"TEMP_DELETEABLE","symmetry_placement_tests")
mkpath(SCRATCH)
const FIXTURE=mktempdir(SCRATCH)
ENV["LUDWIG_PRIVATE_ROOT"]=joinpath(FIXTURE,"app")
include(joinpath(ROOT,"apps","ludwig_web","src","aero_backend.jl"))
push!(LOAD_PATH,joinpath(ROOT,"src","octree_q19"))
using OctreeQ19, YAML
const A=AeroBackend
include(joinpath(@__DIR__,"q19_octree_geometry_helpers.jl"))
const original=sphere_tris(8,16;r=.25)
A._write_stl(joinpath(FIXTURE,"sphere.stl"),original)
const uploaded=A.upload_model(HTTP.Request("POST","/api/aero/models",["Content-Type"=>"application/octet-stream"],read(joinpath(FIXTURE,"sphere.stl"))))
const MODEL=JSON3.read(String(uploaded.body),Dict{String,Any})["model_id"]
const SETTINGS=Dict{String,Any}("model_id"=>MODEL,"solver"=>"q19_octree","lattice"=>"d3q19",
    "transform"=>Dict("scale"=>[1.,1.,1.],"rotation_deg"=>[0.,0.,0.],"translation"=>[0.,0.,0.]),
    "domain"=>Dict{String,Any}("upstream"=>1.,"downstream"=>1.,"lateral"=>1.,"height"=>1.,"boundary"=>"symmetry","sizing"=>"fixed"),
    "surface_dx"=>.125,"levels"=>1,"mesh_sizing"=>"manual","target_vram_percent"=>80,
    "surface_triangle_max_edge_m"=>1.,"geometry_handling"=>"two_sided_open",
    "octree"=>Dict("band_fine"=>6,"band"=>4,"wake_length"=>0.,"wake_level_offset"=>1,"threads"=>2),
    "flow"=>Dict("alpha"=>0.,"beta"=>0.,"velocity"=>10.,"density"=>1.225,"kinematic_viscosity"=>.001,"speed_of_sound"=>343.),
    "steps"=>10,"display_every"=>10,"cuts"=>Any[],"outputs"=>Dict{String,Any}())

@testset "Default and explicit symmetry placement" begin
    # A request that names no placement puts the prepared half model's minimum Y edge on the plane.
    default,triangles=A._validate_case(SETTINGS)
    @test default["domain"]["symmetry_placement"]=="wall"
    @test minimum(p[2] for t in A._solver_model_triangles(default,triangles) for p in t)≈0 atol=1e-10
    for mode in ("wall","center","none")
        request=deepcopy(SETTINGS);request["domain"]["symmetry_placement"]=mode
        if mode=="none"
            request["domain"]["symmetry_offset_y_m"]=.8
            request["transform"]["translation"][2]=.2
        end
        case,triangles=A._validate_case(request)
        staged=A._solver_model_triangles(case,triangles)
        bounds=A._bounds(staged)
        expected=mode=="wall" ? 0. : mode=="center" ? -.25 : .75
        @test bounds["min"][2]≈expected atol=1e-7
        @test A._symmetry_placement(case["domain"])==mode
        @test case["solved_surface_area_m2"]≈case["surface_area_m2"]*(mode=="center" ? .5 : 1.) rtol=1e-7
        yaml=A._yaml_case(case,case["bounds"];build_only=true)
        raw=YAML.load(yaml)
        @test get(raw["advanced"]["domain"],"preserve_symmetry_y",false)==(mode=="none")
        @test !haskey(raw["advanced"]["domain"],"object_center_y_m")
        # The real native parser must preserve the staged Y position.
        folder=joinpath(FIXTURE,mode);mkpath(folder)
        A._write_stl(joinpath(folder,"model.stl"),staged);write(joinpath(folder,"config.yaml"),yaml)
        cfg=OctreeQ19.load_case(folder)
        @test cfg.offset[2]≈0 atol=1e-7
        @test cfg.mesh.domain[2]>=bounds["max"][2]+case["domain"]["lateral"]-1e-6
        saved=A.save_study(HTTP.Request("POST","/api/aero/studies",[],JSON3.write(Dict("name"=>"Placement "*mode,"case"=>case))))
        @test saved.status==201
        saved_id=JSON3.read(String(saved.body),Dict{String,Any})["study_id"]
        document=JSON3.read(String(A.study_file(saved_id).body),Dict{String,Any})
        loaded=A.import_study(HTTP.Request("POST","/api/aero/studies/import",[],JSON3.write(document)))
        @test loaded.status==201
        reopened=JSON3.read(String(loaded.body),Dict{String,Any})["case"]
        @test reopened["domain"]["symmetry_placement"]==mode
        @test reopened["transform"]["translation"]==case["transform"]["translation"]
        @test A._same_study_setup(reopened,case)
        if mode=="none"
            @test reopened["domain"]["symmetry_offset_y_m"]==.8
            @test A._solver_model_triangles(reopened,triangles)==staged
            changed=deepcopy(case);changed["domain"]["symmetry_offset_y_m"]+=.1
            @test !A._same_study_setup(case,changed)
        else
            # Only the wall placement is the default of a domain that names none.
            unnamed=deepcopy(case)
            for key in ("domain","requested_domain");delete!(unnamed[key],"symmetry_placement");end
            @test A._same_study_setup(unnamed,case)==(mode=="wall")
        end
    end
    for invalid in ("invalid",true,1)
        request=deepcopy(SETTINGS);request["domain"]["symmetry_placement"]=invalid
        @test_throws ArgumentError A._validate_case(request)
    end
    for offset in (-1.,Inf,NaN,true)
        request=deepcopy(SETTINGS);request["domain"]["symmetry_placement"]="none";request["domain"]["symmetry_offset_y_m"]=offset
        @test_throws ArgumentError A._validate_case(request)
    end
    request=deepcopy(SETTINGS);request["domain"]["symmetry_placement"]="none";request["domain"]["symmetry_offset_y_m"]=0.
    centered,triangles=A._validate_case(request)
    @test A._solver_model_triangles(centered,triangles)==triangles # no recentering or clipping at import/build
    # The removed centring flag is not read: without a placement choice the default applies.
    flag_only=deepcopy(SETTINGS);flag_only["domain"]["symmetry_center_y"]=true
    flagged=first(A._validate_case(flag_only))
    @test flagged["domain"]["symmetry_placement"]=="wall" && !haskey(flagged["domain"],"symmetry_center_y")
    # A study with a symmetry plane must name its placement; the validated case always does.
    unplaced=deepcopy(default);delete!(unplaced["domain"],"symmetry_placement")
    @test_throws ArgumentError A._check_study_setup(unplaced)
    @test A._check_study_setup(deepcopy(default)) isa AbstractDict
    # The pinned lbmtau snapshot is no longer an Aero Lab solver; the refusal says so.
    request["solver"]="lbmtau";request["lattice"]="d3q27"
    refusal=try A._validate_case(request);"" catch error;sprint(showerror,error) end
    @test occursin("lbmtau",refusal) && occursin("no longer offered",refusal)
    request["solver"]="unknown"
    @test_throws ArgumentError A._validate_case(request)
end

@testset "Q19 time step and output resource settings" begin
    defaults=A._q19_settings(SETTINGS)
    @test defaults["u_lattice"]==.08
    @test defaults["hrr_sigma"]==.98
    @test defaults["wall_function"]===false
    @test defaults["checkpoint_keep"]==2
    @test defaults["max_plane_samples"]==4_000_000
    # Settings the solver no longer has are neither stored nor refused.
    @test isempty(intersect(keys(defaults),("collision","wall_law","absorbing_faces","ramp_steps")))
    retired=deepcopy(SETTINGS);retired["octree"]=merge(Dict{String,Any}(retired["octree"]),Dict("collision"=>"rr","absorbing_faces"=>"outlet","ramp_steps"=>10,"wall_law"=>"spalding"))
    @test A._q19_settings(retired)==defaults
    chosen=Dict("u_lattice"=>.05,"hrr_sigma"=>.9,"wall_function"=>false,"checkpoint_keep"=>1,"max_plane_samples"=>1_000_000)
    request=deepcopy(SETTINGS);request["octree"]=merge(Dict{String,Any}(request["octree"]),chosen)
    case,triangles=A._validate_case(request)
    @test all(case["octree"][key]==value for (key,value) in chosen)
    yaml=A._yaml_case(case,case["bounds"];build_only=true)
    raw=YAML.load(yaml)
    @test raw["octree_q19"]["numerics"]==Dict("hrr_sigma"=>.9,"sgs"=>"wale","u_lattice"=>.05)
    @test raw["octree_q19"]["walls"]["wall_function"]==Dict("enabled"=>false,"sampling"=>"normal_1","filter_steps"=>50,"pressure_gradient"=>false)
    @test !haskey(raw["basic"]["simulation"],"ramp_steps") && !haskey(raw["advanced"],"numerics")
    folder=joinpath(FIXTURE,"resources");mkpath(folder)
    A._write_stl(joinpath(folder,"model.stl"),A._solver_model_triangles(case,triangles))
    write(joinpath(folder,"config.yaml"),yaml)
    cfg=OctreeQ19.load_case(folder)
    @test cfg.numerics.u_lattice==.05
    @test cfg.numerics.hrr_sigma==.9
    @test cfg.walls.wall_function===false
    @test cfg.outputs.checkpoint_keep==1
    @test OctreeQ19.plane_sample_limit(cfg)==1_000_000
    saved=A.save_study(HTTP.Request("POST","/api/aero/studies",[],JSON3.write(Dict("name"=>"Q19 resources","case"=>case))))
    @test saved.status==201
    saved_id=JSON3.read(String(saved.body),Dict{String,Any})["study_id"]
    document=JSON3.read(String(A.study_file(saved_id).body),Dict{String,Any})
    loaded=A.import_study(HTTP.Request("POST","/api/aero/studies/import",[],JSON3.write(document)))
    @test loaded.status==201
    reopened=JSON3.read(String(loaded.body),Dict{String,Any})["case"]
    @test all(reopened["octree"][key]==value for (key,value) in chosen)
    @test A._same_study_setup(reopened,case)
    for (key,value) in (("checkpoint_keep",2),("max_plane_samples",2_000_000))
        changed=deepcopy(case);changed["octree"][key]=value
        @test A._same_study_setup(case,changed) # old fields remain usable after storage/display edits
    end
    for (key,value) in (("u_lattice",.08),("hrr_sigma",.98),("wall_function",true))
        changed=deepcopy(case);changed["octree"][key]=value
        @test !A._same_study_setup(case,changed)
    end
    # A case that names no octree numerics equals one that names the defaults.
    unnamed=first(A._validate_case(SETTINGS));explicit=deepcopy(unnamed)
    for key in keys(chosen);pop!(unnamed["octree"],key,nothing);end
    @test A._same_study_setup(unnamed,explicit)
    for (key,values) in (("u_lattice",(0.,.121,Inf,NaN,true)),("checkpoint_keep",(0,11,1.5,true)),
                         ("hrr_sigma",(.49,1.01,NaN,true)),("wall_function",("on",1)),("max_plane_samples",(9999,16_000_001,10_000.5,true)))
        for value in values
            invalid=deepcopy(SETTINGS);invalid["octree"]=Dict{String,Any}(invalid["octree"]);invalid["octree"][key]=value
            @test_throws ArgumentError A._validate_case(invalid)
        end
    end
end

@testset "Native Q19 fields do not subdivide solver geometry" begin
    request=deepcopy(SETTINGS);request["surface_triangle_max_edge_m"]=.0001
    case,triangles=A._validate_case(request)
    @test case["surface_subdivision"]["applied"]===false
    @test case["surface_subdivision"]["refined_triangles_before_symmetry"]==length(original)
    @test case["surface_subdivision"]["gpu_surface_allowance_gb"]==0
    @test !haskey(case,"surface_triangle_max_edge_m")
    yaml=YAML.load(A._yaml_case(case,case["bounds"];build_only=true))
    # The octree solver has no surface subdivision: its case file carries neither key.
    @test !haskey(yaml["basic"],"subdivision") && !haskey(yaml["basic"],"minimum_facet_size")
    changed=deepcopy(case);changed["surface_triangle_max_edge_m"]=.05
    @test A._same_study_setup(case,changed)
    @test_throws ArgumentError A._surface_subdivision_preview(triangles,.0001)
end

@testset "Open half-model admission uses placed Y=0" begin
    # Five sides of a box: the missing Y-min cap is the intended mirror opening.
    p=[(-.5,0.,-.5),(.5,0.,-.5),(.5,1.,-.5),(-.5,1.,-.5),
       (-.5,0.,.5),(.5,0.,.5),(.5,1.,.5),(-.5,1.,.5)]
    faces=((1,4,3,2),(5,6,7,8),(1,5,8,4),(2,3,7,6),(4,8,7,3))
    tris=A.Triangle[]
    for (a,b,c,d) in faces;push!(tris,(p[a],p[b],p[c]),(p[a],p[c],p[d]));end
    A._write_stl(joinpath(FIXTURE,"open_half.stl"),tris)
    response=A.upload_model(HTTP.Request("POST","/api/aero/models",["Content-Type"=>"application/octet-stream"],read(joinpath(FIXTURE,"open_half.stl"))))
    model=JSON3.read(String(response.body),Dict{String,Any})["model_id"]
    for (mode,offset,move,accepted) in (("wall",0.,0.,true),("center",0.,0.,false),
                                      ("none",.5,0.,true),("none",.25,.25,true),
                                      ("none",1.5,0.,false),("none",.5,.2,false))
        request=deepcopy(SETTINGS);request["model_id"]=model;request["geometry_handling"]="closed_solid"
        request["domain"]["symmetry_placement"]=mode;request["domain"]["symmetry_offset_y_m"]=offset
        request["transform"]["translation"][2]=move
        case,transformed=A._validate_case(request)
        @test !case["geometry_audit"]["outward_orientation_verified"]
        @test case["geometry_audit"]["symmetry_boundary_verified"]==accepted
        actual=Dict{String,Any}("levels"=>[Dict("bouzidi_boundary_cells"=>10000,"dx_m"=>case["surface_dx"])])
        A._assess_mesh!(actual,case)
        @test actual["run_blocked_by_geometry"]==!accepted
        @test actual["quality"]["runnable"]==accepted
        case["geometry_handling"]="two_sided_open";A._assess_mesh!(actual,case)
        @test !actual["run_blocked_by_geometry"]
        @test actual["geometry_override_active"]==!accepted
    end
end

# Exercise the real shared main/thermal placement function without allocating a GPU lattice.
module NativePlacement
include(joinpath(@__DIR__,"..","..","src","solver","physics_scaling.jl"))
MESH_MIN_DX=0.;REFERENCE_LENGTH_FOR_MESHING=1.;REFERENCE_DIMENSION=:x
REFERENCE_CHORD_CONFIG=1.;REFERENCE_AREA_CONFIG=1.;MOMENT_CENTER_CONFIG=(.25,0.,0.)
FLOW_VELOCITY=10.;FLUID_KINEMATIC_VISCOSITY=.001;FLUID_DENSITY=1.225
SURFACE_RESOLUTION=20;U_TARGET=.08;CS2_PHYSICS=1f0/3;CS4_PHYSICS=1f0/9;TAU_MIN=.501
NU_SGS_BACKGROUND=0.;VARIABLE_NU_SGS=false;VARIABLE_NU_SGS_FLOOR=0.
DOMAIN_UPSTREAM=4.;DOMAIN_DOWNSTREAM=5.;DOMAIN_LATERAL=4.;DOMAIN_HEIGHT=4.;DOMAIN_SPAN_OVERRIDE=-1.
GROUND_ANALYSIS=false;HALF_Y_DOMAIN=true;SYMMETRIC_ANALYSIS=true;DOMAIN_PRESERVE_SYMMETRY_Y=true
PERIODIC_Y_AMR=false;SPAN_PERIODIC_Y=false;PERIODIC_Y_BODY_SHIFT=0.
NUM_LEVELS_CONFIG=2;AUTO_LEVELS=false;MAX_LEVELS=7;BLOCK_SIZE_CONFIG=8;MIN_COARSE_BLOCKS=4
DOMAIN_OBJECT_CENTER_X_M=nothing;DOMAIN_OBJECT_CENTER_Y_M=nothing;DOMAIN_OBJECT_CENTER_Z_M=nothing;DOMAIN_OBJECT_CENTER_X_FRACTION=nothing
LEVEL_VISCOSITY_MODE=:legacy;DDF_ENABLED=false;TEMPORAL_INTERPOLATION=false;WALL_MODEL_ENABLED=false
end

@testset "Main/thermal native Y preservation" begin
    for (lo,hi) in ((.75,1.25),(-.25,.25),(-.1,.4))
        result=redirect_stdout(devnull) do
            NativePlacement.compute_domain_from_mesh((-.5,lo,-.5),(.5,hi,.5))
        end
        @test result.mesh_offset[2]==0.
        @test result.domain_size[2]>=hi+NativePlacement.DOMAIN_LATERAL
        @test result.num_levels==2
    end
    @test_throws ErrorException NativePlacement.compute_domain_from_mesh((-.5,-1.,-.5),(.5,-.5,.5))
    NativePlacement.DOMAIN_PRESERVE_SYMMETRY_Y=false
    legacy=redirect_stdout(devnull) do
        NativePlacement.compute_domain_from_mesh((-.5,.75,-.5),(.5,1.25,.5))
    end
    @test legacy.mesh_offset[2]==-.75
end

@testset "Solved facet area and backend-specific coarse domain" begin
    triangle=((0.,-1.,0.),(2.,1.,0.),(0.,1.,0.))
    @test A._positive_y_surface_area([triangle],0.)≈1.5
    @test A._positive_y_surface_area([triangle],1.)≈2.
    @test A._positive_y_surface_area([triangle],-1.)≈0.
    @test A._positive_y_surface_area([triangle],.5)≈1.875
    request=deepcopy(SETTINGS);request["levels"]=3;request["domain"]["symmetry_placement"]="center"
    case=first(A._validate_case(request))
    h0=case["surface_dx"]*2.0^(case["levels"]-1)
    limits=A._mesh_limits(case;available_ram_bytes=8e9)
    @test limits["valid"]
    @test limits["pyramid_fits"]
    @test minimum(A._estimate(case)["coarse_grid_cells"])<32
    @test all(iseven,A._estimate(case)["coarse_grid_cells"])
    @test limits["coarse_cells"]==prod(A._estimate(case)["coarse_grid_cells"])
    @test limits["dense_mask_bytes"]==limits["coarse_cells"]*sum(8.0^k for k in 0:2)/8
    # CPU allowance still rejects real memory pressure, without a GPU-shaped domain.
    @test !A._mesh_limits(case;available_ram_bytes=100e6)["valid"]
    gpu=deepcopy(request);gpu["solver"]="main";gpu["lattice"]="d3q27"
    @test_throws ArgumentError A._validate_case(gpu)
    for boundary in ("symmetry","ground","free_air"),sizing in ("minimum","proportions")
        domain=Dict{String,Any}("boundary"=>boundary,"sizing"=>sizing,"upstream"=>.2,"downstream"=>.3,"lateral"=>.2,"height"=>.2,"symmetry_placement"=>"center","proportions"=>[3.,2.,1.])
        before=deepcopy(domain);cpu=A._expand_minimum_domain!(domain,[1.,.5,.25],.05,5;octree=true)
        @test all(domain[k]>=before[k] for k in ("upstream","downstream","lateral","height"))
        @test cpu["minimum_axis_size_m"]==2*.05*16
        @test all(cpu["resolved_size_m"].>=cpu["requested_size_m"])
        if boundary=="symmetry"
            @test cpu["resolved_size_m"][2]≈.5*min(cpu["resolved_size_m"][[1,3]]...)
        end
        gpudomain=deepcopy(before);gpu=A._expand_minimum_domain!(gpudomain,[1.,.5,.25],.05,5)
        @test gpu["minimum_axis_size_m"]==32*.05*16
        @test all(gpu["resolved_size_m"].>=32*.05*16)
    end
end
