# Portable CPU regression: generated sphere, native startup fields, resume, and independent output schedules.
# julia --startup-file=no -t 2 --project=third_party/JXfoil tools/tests/q19_output_cadence.jl
using Test,JSON3,TOML,Dates,HTTP
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
push!(LOAD_PATH,joinpath(ROOT,"src","octree_q19"))
using OctreeQ19,YAML
const O=OctreeQ19
const SCRATCH=joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(ROOT,"..","02_PRIVATE_PROJECT_DEVELOPMENT")),"TEMP_DELETEABLE","q19_tests")
mkpath(SCRATCH)
const RUN_FOLDER=haskey(ENV,"Q19_TEST_OUTPUT_ROOT") ? mkpath(ENV["Q19_TEST_OUTPUT_ROOT"]) : mktempdir(SCRATCH)
ENV["LUDWIG_PRIVATE_ROOT"]=joinpath(RUN_FOLDER,"isolated_app")
include(joinpath(ROOT,"apps","ludwig_web","src","workbench_revision.jl"))
include(joinpath(ROOT,"apps","ludwig_web","src","aero_backend.jl"))
include(joinpath(@__DIR__,"q19_octree_geometry_helpers.jl"))
const A=AeroBackend
const FIXTURE=joinpath(RUN_FOLDER,"fixture");mkpath(FIXTURE)
A._write_stl(joinpath(FIXTURE,"model.stl"),sphere_tris(8,16;r=.25))
const uploaded=A.upload_model(HTTP.Request("POST","/api/aero/models",["Content-Type"=>"application/octet-stream"],read(joinpath(FIXTURE,"model.stl"))))
@test uploaded.status==201
const MODEL=JSON3.read(String(uploaded.body),Dict{String,Any})["model_id"]
const SETTINGS=Dict{String,Any}("model_id"=>MODEL,"solver"=>"q19_octree","lattice"=>"d3q19",
    "transform"=>Dict("scale"=>[1.,1.,1.],"rotation_deg"=>[0.,0.,0.],"translation"=>[0.,0.,0.]),
    "domain"=>Dict("upstream"=>1.,"downstream"=>1.,"lateral"=>1.,"height"=>1.,"boundary"=>"free_air","sizing"=>"fixed"),
    "surface_dx"=>.125,"levels"=>1,"mesh_sizing"=>"manual","target_vram_percent"=>80,
    "octree"=>Dict("band_fine"=>6,"band"=>4,"wake_length"=>0.,"wake_level_offset"=>1,"threads"=>2),
    "flow"=>Dict("alpha"=>0.,"beta"=>0.,"velocity"=>10.,"density"=>1.225,"kinematic_viscosity"=>.001,"speed_of_sound"=>343.),
    "steps"=>1000,"display_every"=>50,"frames"=>20,"cuts"=>Any[],"outputs"=>Dict{String,Any}())
const CASE=first(A._validate_case(SETTINGS))
write(joinpath(FIXTURE,"config.yaml"),A._yaml_case(CASE,CASE["bounds"];build_only=false))
const OLD=Dict("case"=>CASE,"run_case_dir"=>FIXTURE)

@testset "Opt-in cavity sealing through native and portable studies" begin
    @test A._q19_settings(Dict{String,Any}())["seal_cavities"]===false
    for invalid in ("false",1,nothing)
        @test_throws ArgumentError A._q19_settings(Dict("octree"=>Dict("seal_cavities"=>invalid)))
    end
    cases=Dict{Bool,Any}()
    for seal in (false,true)
        request=deepcopy(SETTINGS);request["octree"]["seal_cavities"]=seal
        case=first(A._validate_case(request));cases[seal]=case
        @test case["octree"]["seal_cavities"]===seal
        yaml=A._yaml_case(case,case["bounds"];build_only=false)
        @test YAML.load(yaml)["octree_q19"]["mesh"]["seal_cavities"]===seal
        dir=joinpath(RUN_FOLDER,"seal_"*string(seal));mkpath(dir)
        cp(joinpath(FIXTURE,"model.stl"),joinpath(dir,"model.stl"));write(joinpath(dir,"config.yaml"),yaml)
        @test O.load_case(dir).mesh.seal_cavities===seal
        response=A.save_study(HTTP.Request("POST","/api/aero/studies",[],JSON3.write(Dict("name"=>"Sealing test","case"=>case))))
        @test response.status==201
        saved=JSON3.read(String(response.body),Dict{String,Any})
        document=JSON3.read(String(A.study_file(saved["study_id"]).body),Dict{String,Any})
        @test document["case"]["octree"]["seal_cavities"]===seal
        loaded=A.import_study(HTTP.Request("POST","/api/aero/studies/import",[],JSON3.write(document)))
        @test loaded.status==201
        @test JSON3.read(String(loaded.body),Dict{String,Any})["case"]["octree"]["seal_cavities"]===seal
    end
    @test !A._same_study_setup(cases[false],cases[true])
    @test !A._same_study_setup(cases[true],cases[false])
    legacy=deepcopy(cases[false]);delete!(legacy["octree"],"seal_cavities")
    @test A._same_study_setup(legacy,cases[false])
end

@testset "Q19 diagnostics independent of field cadence" begin
    case=deepcopy(OLD["case"]);case["steps"]=1000
    for every in (50,100,37)
        case["display_every"]=every
        raw=YAML.load(A._yaml_case(case,case["bounds"];build_only=false))
        @test raw["basic"]["simulation"]["slice_output_freq"]==every
        @test raw["advanced"]["forces"]["output_freq"]==gcd(10,every)
        @test raw["advanced"]["diagnostics"]["freq"]==gcd(10,every)
        @test raw["basic"]["simulation"]["output_startup_slices"]===true
    end
    case["solver"]="main";case["lattice"]="d3q27";case["display_every"]=50
    raw=YAML.load(A._yaml_case(case,case["bounds"];build_only=false))
    @test raw["advanced"]["forces"]["output_freq"]==50
    @test raw["advanced"]["diagnostics"]["freq"]==50
end
function exercise(startup)
    folder=joinpath(RUN_FOLDER,startup ? "startup_on" : "startup_off")
    ispath(folder)&&error("Test output already exists: $folder")
    mkpath(folder)
    oldrun=OLD["run_case_dir"]
    cp(joinpath(oldrun,"model.stl"),joinpath(folder,"model.stl"))
    raw=YAML.load_file(joinpath(oldrun,"config.yaml"))
    merge!(raw["basic"]["simulation"],Dict("steps"=>13,"ramp_steps"=>4,"output_freq"=>14,"slice_output_freq"=>6,"output_startup_slices"=>startup,"build_only"=>false))
    raw["advanced"]["forces"]["output_freq"]=2;raw["advanced"]["diagnostics"]["freq"]=2
    raw["octree_q19"]["checkpoint"]["every_minutes"]=0
    YAML.write_file(joinpath(folder,"config.yaml"),raw)
    cfg=O.load_case(folder);@test cfg.outputs.startup_slices==startup
    open(joinpath(folder,"solver.log"),"w") do io
        redirect_stdout(io) do;O.run_case(cfg);end
    end
    slices=joinpath(folder,"RESULTS","SLICES")
    steps=sort(parse.(Int,replace.(readdir(slices),"step_"=>"")))
    @test steps==[0,6,12,13]
    for step in steps
        dir=joinpath(slices,"step_"*lpad(string(step),6,'0'))
        for file in readdir(dir;join=true)
            header,arrays=O.LBMSliceFormat.read_lbmslice(file)
            @test header["convergence_assessed"]===false
            @test header["state_kind"]==(step==0 ? (startup ? "initialization" : "mesh") : "evolved")
            @test header["with_fields"]==(step>0||startup)
            if step==0&&startup
                @test header["u_inlet_lattice"]==0
                @test header["u_target_lattice"]==.08
                @test maximum(abs,vcat(filter(isfinite,vec(arrays["u"])),[0.0]))==0
            end
        end
    end
    for name in ("forces.csv","convergence.csv")
        lines=readlines(joinpath(folder,"RESULTS","csv",name))
        @test [parse(Int,first(split(line,','))) for line in lines[2:end]]==[1,2,4,6,8,10,12,13]
    end
    @test occursin("[InitialFields]",read(joinpath(folder,"solver.log"),String))==startup
    @test TOML.parsefile(joinpath(folder,"RESULTS","octree_run_summary.toml"))["convergence_assessed"]===false
    if startup
        initial=first(readdir(joinpath(slices,"step_000000");join=true));modified=mtime(initial)
        open(joinpath(folder,"resume.log"),"w") do io
            redirect_stdout(io) do;O.run_case(cfg;resume=true);end
        end
        @test !occursin("[InitialFields]",read(joinpath(folder,"resume.log"),String))
        @test mtime(initial)==modified
    end
end
@testset "Native initial fields on/off and independent schedules" begin
    exercise(true);exercise(false)
end
write(joinpath(RUN_FOLDER,"startup_cadence_checks.toml"),"passed = true\nchecked_utc = \"$(now(UTC))\"\ncoarse_steps = 13\nfield_steps = [0,6,12,13]\ndiagnostics_steps = [1,2,4,6,8,10,12,13]\n")
