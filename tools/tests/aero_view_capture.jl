# julia --project=third_party/JXfoil tools/tests/aero_view_capture.jl
using Test,HTTP,JSON3,UUIDs
const REPOSITORY=normpath(joinpath(@__DIR__,"..",".."))
const SCRATCH=joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(REPOSITORY,"..","02_PRIVATE_PROJECT_DEVELOPMENT")),"TEMP_DELETEABLE","view_capture_tests")
mkpath(SCRATCH);const FIXTURE=mktempdir(SCRATCH)
ENV["LUDWIG_PRIVATE_ROOT"]=joinpath(FIXTURE,"private")
include(joinpath(REPOSITORY,"apps","ludwig_web","src","aero_backend.jl"))
const A=AeroBackend
const ID=string(uuid4());const OTHER=string(uuid4());const CASE=joinpath(FIXTURE,"case");mkpath(CASE)
A._save_job(Dict{String,Any}("job_id"=>ID,"status"=>"cancelled","run_case_dir"=>CASE))
const FRAME="/api/aero/jobs/$ID/files/run/surface_000012.lbmw"
const DIGEST=repeat("a",64)
function writeframe(step=12)
    open(joinpath(CASE,"surface_000012.lbmw"),"w") do io
        write(io,codeunits("LBMWFLD\0"),UInt32(1),UInt32(1),UInt64(3),codeunits(DIGEST),Int64(step),zeros(Float64,4),Float32[1,2,3])
    end
end
writeframe()
jsonpost(path,data)=HTTP.Request("POST",path,["Content-Type"=>"application/json"],JSON3.write(data))
decode(response)=JSON3.read(String(response.body),Dict{String,Any})
function newrequest()
    path="/api/aero/jobs/$ID/view-captures"
    response=A.handle(jsonpost(path,Dict("frame_url"=>FRAME,"width"=>400,"height"=>300)),path)
    @test response.status==202
    decode(response)
end
function png_header_fixture(width=400,height=300)
    # Transport fixture exercises identity, dimensions and bounded image acceptance;
    # actual Babylon PNG encoding is handled by the browser, not this fixture.
    b=zeros(UInt8,57);b[1:8]=[0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a]
    b[13:16]=codeunits("IHDR");b[end-7:end-4]=codeunits("IEND")
    for (i,v) in ((17,width),(21,height)),j in 0:3;b[i+j]=UInt8((v>>(8*(3-j)))&0xff);end
    b
end
@testset "Aero Lab renderer images retain native frame identity" begin
    bad="/api/aero/jobs/$ID/view-captures"
    for source in (replace(FRAME,ID=>OTHER),"https://example.invalid/surface.lbmw",replace(FRAME,"surface_"=>"../surface_"))
        @test A.handle(jsonpost(bad,Dict("frame_url"=>source)),bad).status==400
    end
    writeframe(0);@test A.handle(jsonpost(bad,Dict("frame_url"=>FRAME)),bad).status==400;writeframe()
    request=newrequest();token=request["request_id"]
    @test request["step"]==12&&request["sample_count"]==3&&request["geometry_sha256"]==DIGEST
    @test A.handle(jsonpost(bad,Dict("frame_url"=>FRAME)),bad).status==400
    @test decode(A.pending_view_capture(ID))["request"]["request_id"]==token
    route="$bad/$token"
    headers=["Content-Type"=>"image/png","X-Aero-Step"=>"12","X-Aero-Samples"=>"3","X-Aero-Geometry"=>DIGEST,"X-Aero-Frame"=>request["frame_sha256"],"X-Aero-Representation"=>"wall_samples","X-Aero-Field"=>"cp"]
    for (name,value) in (("X-Aero-Step","8"),("X-Aero-Samples","2"),("X-Aero-Geometry",repeat("b",64)),("X-Aero-Frame",repeat("b",64)),("X-Aero-Representation","facet_averages"),("X-Aero-Field","pressure"))
        changed=[key=>(key==name ? value : v) for (key,v) in headers]
        @test A.handle(HTTP.Request("POST",route,changed,png_header_fixture()),route).status==400
    end
    for bytes in (UInt8[1,2,3],png_header_fixture(800,300),zeros(UInt8,A.VIEW_CAPTURE_MAX_BYTES+1))
        @test A.handle(HTTP.Request("POST",route,headers,bytes),route).status==400
    end
    open(joinpath(CASE,"surface_000012.lbmw"),"r+") do io;seekend(io);seek(io,position(io)-1);write(io,UInt8(1));end
    @test A.handle(HTTP.Request("POST",route,headers,png_header_fixture()),route).status==400
    writeframe()
    @test A.handle(HTTP.Request("POST",route,headers,png_header_fixture()),route).status==200
    saved=decode(A.view_capture_status(ID,token));@test saved["status"]=="complete"
    @test length(saved["image_sha256"])==64&&isfile(saved["image_path"])
    @test A.view_capture_image(ID,token).body==png_header_fixture()
    @test A.handle(HTTP.Request("POST",route,headers,png_header_fixture()),route).status==400
    @test decode(A.pending_view_capture(ID))["request"]===nothing
    @test A.view_capture_image(ID,string(uuid4())).status==404
    expired=newrequest();file=A._view_capture_path(ID,expired["request_id"]);expired["expires_at"]=time()-1;A._write_view_capture(file,expired)
    @test decode(A.view_capture_status(ID,expired["request_id"]))["status"]=="expired"
    @test decode(A.pending_view_capture(ID))["request"]===nothing
    error_request=newrequest();error_id=error_request["request_id"]
    response=A.pending_view_capture(ID,HTTP.Request("GET",bad,["X-Aero-View-Client"=>"viewport"]))
    @test decode(response)["request"]["request_id"]==error_id
    @test isfile(joinpath(A._view_capture_root(ID),"browser.json"))
    failure_route="$bad/$error_id/error"
    @test A.handle(jsonpost(failure_route,Dict("error"=>"Renderer lost its canvas")),failure_route).status==200
    @test decode(A.view_capture_status(ID,error_id))["status"]=="error"
    @test decode(A.pending_view_capture(ID))["request"]===nothing
end
println("View capture fixtures: ",FIXTURE)
