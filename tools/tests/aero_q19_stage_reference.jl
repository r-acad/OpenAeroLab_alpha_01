using Test, HTTP, JSON3
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
push!(LOAD_PATH,joinpath(ROOT,"src","octree_q19"))
using YAML, OctreeQ19
const SCRATCH=joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(ROOT,"..","02_PRIVATE_PROJECT_DEVELOPMENT")),"TEMP_DELETEABLE","aero_q19_stage_reference_tests")
mkpath(SCRATCH);const FIXTURE=mktempdir(SCRATCH;prefix="reference_")
ENV["LUDWIG_PRIVATE_ROOT"]=joinpath(FIXTURE,"app")
include(joinpath(ROOT,"apps","ludwig_web","src","aero_backend.jl"))
include(joinpath(@__DIR__,"q19_octree_geometry_helpers.jl"))
const A=AeroBackend
A._write_stl(joinpath(FIXTURE,"model.stl"),sphere_tris(8,16;r=.25))
response=A.upload_model(HTTP.Request("POST","/api/aero/models",["Content-Type"=>"application/octet-stream"],read(joinpath(FIXTURE,"model.stl"))))
const MODEL=JSON3.read(String(response.body),Dict{String,Any})["model_id"]
const SETTINGS=Dict{String,Any}("model_id"=>MODEL,"solver"=>"q19_octree","lattice"=>"d3q19",
    "transform"=>Dict("scale"=>[1.,1.,1.],"rotation_deg"=>[0.,0.,0.],"translation"=>[0.,0.,0.]),
    "domain"=>Dict("upstream"=>1.,"downstream"=>1.,"lateral"=>1.,"height"=>1.,"boundary"=>"free_air","sizing"=>"fixed"),
    "surface_dx"=>.125,"levels"=>1,"mesh_sizing"=>"manual","target_vram_percent"=>80,
    "octree"=>Dict("band_fine"=>6,"band"=>4,"wake_length"=>0.,"wake_level_offset"=>1,"threads"=>2),
    "flow"=>Dict("alpha"=>0.,"beta"=>0.,"velocity"=>10.,"density"=>1.225,"kinematic_viscosity"=>.001,"speed_of_sound"=>343.),
    "steps"=>1000,"display_every"=>50,"frames"=>20,"cuts"=>Any[],"outputs"=>Dict{String,Any}())

@testset "Q19 optional reference length for staged remeshing" begin
    @test !haskey(A._q19_settings(Dict()),"stage_reference_length_m")
    for invalid in (0.,-1.,NaN,Inf,"invalid",true)
        @test_throws ArgumentError A._q19_settings(Dict("octree"=>Dict("stage_reference_length_m"=>invalid)))
    end
    base=first(A._validate_case(SETTINGS));baseline=YAML.load(A._yaml_case(base,base["bounds"];build_only=true))
    prior=baseline["basic"]["reference_length_for_meshing"]
    altered=deepcopy(SETTINGS);altered["octree"]["stage_reference_length_m"]=prior+.01
    stage=first(A._validate_case(altered));updated=YAML.load(A._yaml_case(stage,stage["bounds"];build_only=true))
    @test stage["octree"]["stage_reference_length_m"]==prior+.01
    @test updated["basic"]["reference_length_for_meshing"]==prior+.01
    for key in ("upstream","downstream","lateral","height")
        @test isapprox(baseline["advanced"]["domain"][key]*prior,updated["advanced"]["domain"][key]*(prior+.01);rtol=1e-10)
    end
    ordinary=YAML.load(A._yaml_case(base,base["bounds"];build_only=true))
    @test ordinary["basic"]["reference_length_for_meshing"]==prior
    stage["octree"]["stage_reference_length_m"]=prior*100
    @test_throws ArgumentError A._yaml_case(stage,stage["bounds"];build_only=true)
    # Actual Q19 configs at adjacent levels preserve the physical setup and
    # coarse timestep, not just the displayed dimensionless clearances.
    finer=deepcopy(SETTINGS);finer["surface_dx"]=.0625;finer["levels"]=2
    finer["octree"]["stage_reference_length_m"]=prior
    refined=first(A._validate_case(finer))
    for (tag,c) in (("source",base),("target",refined))
        dir=joinpath(FIXTURE,tag);mkpath(dir)
        cp(joinpath(FIXTURE,"model.stl"),joinpath(dir,"model.stl"))
        write(joinpath(dir,"config.yaml"),A._yaml_case(c,c["bounds"];build_only=true))
    end
    source=OctreeQ19.load_case(joinpath(FIXTURE,"source");build_only=true)
    target=OctreeQ19.load_case(joinpath(FIXTURE,"target");build_only=true)
    @test source.reference_length==target.reference_length
    @test source.mesh.domain==target.mesh.domain
    @test all(isapprox.(source.offset,target.offset;rtol=0,atol=1e-9))
    @test source.sponge_thickness*source.reference_length==target.sponge_thickness*target.reference_length
    @test source.mesh.dx_finest*2^(source.mesh.levels-1)==target.mesh.dx_finest*2^(target.mesh.levels-1)
    s=OctreeQ19.lattice_scaling(source);t=OctreeQ19.lattice_scaling(target)
    @test isapprox(s.dt_finest*2^(source.mesh.levels-1),t.dt_finest*2^(target.mesh.levels-1);rtol=1e-12)
end
