# julia --project=third_party/JXfoil tools/tests/aero_wall_surface.jl
using Test, SHA, TOML, UUIDs, HTTP, JSON3
const ROOT=normpath(joinpath(@__DIR__,"..",".."))
const SCRATCH=joinpath(ROOT,"..","02_PRIVATE_PROJECT_DEVELOPMENT","TEMP_DELETEABLE","aero_wall_tests")
mkpath(SCRATCH);const FIXTURE=mktempdir(SCRATCH);ENV["LUDWIG_PRIVATE_ROOT"]=FIXTURE
include(joinpath(ROOT,"apps","ludwig_web","src","aero_backend.jl"))
const A=AeroBackend
body(response)=JSON3.read(String(response.body),Dict{String,Any})
function geometry_bytes()
    io=IOBuffer();write(io,codeunits("LBMWGEO\0"),UInt32(1),UInt32(36),UInt64(3))
    for i in 1:3
        write(io,Float32(i/10),0f0,0f0,0f0,0f0,1f0,.1f0,UInt32(1),UInt8(3),UInt8(2),UInt16(i==2 ? 3 : 1))
    end
    take!(io)
end
const GEO=geometry_bytes();const DIGEST=bytes2hex(sha256(GEO))
const GEONAME="surface_wall_geometry_"*DIGEST*".lbmwg"
function frame_bytes(;digest=DIGEST,step=12,q=2.)
    io=IOBuffer();write(io,codeunits("LBMWFLD\0"),UInt32(1),UInt32(1),UInt64(3),codeunits(digest),Int64(step),.25,1.,2.,q,Float32[2,-3,NaN]);take!(io)
end
const ID=string(uuid4());const VTK=joinpath(A.CASES_ROOT,ID*"_run","RESULTS","VTK");mkpath(VTK)
const FRAME=joinpath(VTK,"surface_000012.lbmw");write(joinpath(VTK,GEONAME),GEO);write(FRAME,frame_bytes())
A._save_job(Dict("job_id"=>ID,"status"=>"cancelled","case"=>Dict("outputs"=>Dict())))
@testset "Exact native wall identity and bounded metadata" begin
    h=A._wall_surface_metadata(FRAME;verify=true)
    @test h["samples"]==3 && h["step"]==12
    @test h["geometry_sha256"]==DIGEST
    @test h["representation"]=="wall_samples" && h["geometry_frame"]=="domain"
    @test h["q_reference_pa"]==2
    @test A._wall_geometry_bounds(joinpath(VTK,GEONAME))["min"]≈[.1,0,0]
    @test A._artifact_complete(FRAME)
    @test A._artifact_complete(joinpath(VTK,GEONAME))
    @test A._export_category(FRAME)=="surfaces"
    @test A._selected_export(FRAME,Dict("surface_lbmp"=>true))
    @test !A._selected_export(FRAME,Dict("surface_lbmp"=>false))
    @test A._selected_export(joinpath(VTK,GEONAME),Dict("surface_lbmp"=>true))
    for (name,bytes) in (("truncated.lbmw",frame_bytes()[1:end-1]),("reference.lbmw",frame_bytes(q=3.)),("hash.lbmw",frame_bytes(digest="f"^64)))
        path=joinpath(FIXTURE,name);write(path,bytes)
        @test !A._artifact_complete(path)
        @test_throws ArgumentError A._wall_surface_metadata(path;verify=true)
    end
    corrupt=joinpath(FIXTURE,"corrupt");mkpath(corrupt);write(joinpath(corrupt,"surface_000012.lbmw"),frame_bytes())
    altered=copy(GEO);altered[25]⊻=1;write(joinpath(corrupt,GEONAME),altered)
    @test_throws ArgumentError A._wall_surface_metadata(joinpath(corrupt,"surface_000012.lbmw");verify=true)
    catalog=A.discover_artifacts(ID)
    @test endswith(catalog["surface_latest"],"surface_000012.lbmw")
    @test catalog["surface_latest"]==catalog["surface_native_latest"]
    @test catalog["wall_surface_latest_info"]["geometry_sha256"]==DIGEST
    @test endswith(catalog["wall_surface_latest_info"]["geometry_url"],GEONAME)
    @test length(catalog["wall_surfaces"])==1
    @test length(catalog["surfaces"])==1 # static geometry is setup, not another field
    @test catalog["availability"]["surface"]["available"]
end
@testset "Active atomic wall and VTK files are not completed results" begin
    inprogress=("surface_000016.lbmw.tmp.123.abcdef", "surface_wall_samples_000016.vtp.tmp.456.1f",
                "checkpoint_000000016.bin.tmp", "render.png.123.partial")
    for name in inprogress
        path=joinpath(VTK,name);write(path,"unfinished payload")
        @test A._artifact_in_progress(path)
        @test !A._artifact_complete(path)
    end
    ordinary=joinpath(VTK,"notes.tmp.123.nothex");write(ordinary,"ordinary completed file")
    @test !A._artifact_in_progress(ordinary) && A._artifact_complete(ordinary)
    catalog=A.discover_artifacts(ID)
    active=Set(entry["name"] for entry in catalog["in_progress_files"] if entry["phase"]=="run")
    complete=Set(entry["name"] for entry in catalog["files"] if entry["phase"]=="run")
    @test all(name->"RESULTS/VTK/"*name in active,inprogress)
    @test all(name->!("RESULTS/VTK/"*name in complete),inprogress)
    @test "RESULTS/VTK/notes.tmp.123.nothex" in complete
    @test endswith(catalog["surface_latest"],"surface_000012.lbmw")
end
@testset "Native import remains binary, portable and explicit" begin
    options=Dict{String,Any}("coordinates"=>"auto")
    imported=A._store_result_sources([(basename(FRAME),FRAME,nothing)],options)
    @test imported["surface"]["samples"]==3
    @test imported["surface"]["triangles"]==0
    @test imported["surface"]["representation"]=="wall_samples"
    @test Set(imported["fields"])==Set(["pressure","cp"])
    @test length(imported["files"])==2
    id=imported["id"];descriptor=body(A.imported_surface(id))
    @test descriptor["representation"]=="wall_samples"
    @test descriptor["surface_offset"]==[0.,0.,0.]
    @test descriptor["geometry_sha256"]==DIGEST
    @test isempty(descriptor["fields"]["pressure"]) # no enormous JSON expansion or facet averaging
    @test length(JSON3.write(descriptor))<4000
    @test A.imported_result_file(id,GEONAME).body==GEO
    frame_name=imported["surface"]["file"]
    @test A.imported_result_file(id,frame_name).body==frame_bytes()
    @test_throws ArgumentError A.imported_result_file(id,"../"*GEONAME)
    @test_throws ArgumentError A.imported_result_file(id,"unknown.lbmwg")
    empty!(A.RESULT_IMPORT_CACHE)
    @test body(A.imported_surface(id))["geometry_sha256"]==DIGEST
    @test A.handle(HTTP.Request("GET",descriptor["geometry_url"]),descriptor["geometry_url"]).body==GEO
    geometry_only=A._store_result_sources([(GEONAME,"Uploaded geometry",GEO)],options)
    @test geometry_only["surface"]===nothing
    @test any(message->occursin("matching .lbmw",message),geometry_only["warnings"])
    attached=A._store_result_sources([("surface_000012.lbmw","Uploaded frame",frame_bytes())],Dict("dataset_id"=>geometry_only["id"]))
    @test attached["surface"]["samples"]==3
    @test length(A._result_input_files(VTK))==1 # frame brings its own exact static geometry
end
