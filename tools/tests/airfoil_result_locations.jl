using Test,Dates,TOML,UUIDs
const REPOSITORY=dirname(dirname(@__DIR__))
const SCRATCH=joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(dirname(REPOSITORY),"02_PRIVATE_PROJECT_DEVELOPMENT")),"TEMP_DELETEABLE","airfoil_result_location_tests")
mkpath(SCRATCH);ENV["LUDWIG_PRIVATE_ROOT"]=mktempdir(SCRATCH)
include(joinpath(REPOSITORY,"apps","airfoil_web","src","server.jl"))
const A=AirfoilWeb;const L=A.AirfoilResultLocations
const HTTP=A.HTTP
function call(method,path,data=nothing)
    headers=["Content-Type"=>"application/msgpack","Accept"=>"application/msgpack"]
    request=HTTP.Request(method,path,headers,data===nothing ? UInt8[] : A.AirfoilWire.encode(data,:msgpack))
    response=A.app_handler(request)
    value=try A.AirfoilWire.decode(response.body,:msgpack) catch;nothing end
    response,value
end

case(;study="",parent="")=Dict{String,Any}("airfoil"=>Dict("kind"=>"naca4","id"=>"2412"),
    "backends"=>["jxfoil"],"flow"=>Dict("reynolds"=>3e6,"alpha_deg"=>4.,"mach"=>0.),
    "outputs"=>Dict("folder"=>parent),"study_path"=>study)

@testset "Timestamped native result folders follow the saved study" begin
    root=joinpath(ENV["LUDWIG_PRIVATE_ROOT"],"lookup");mkpath(root)
    study_parent=joinpath(ENV["LUDWIG_PRIVATE_ROOT"],"my study");mkpath(study_parent)
    study=joinpath(study_parent,"section.ludwig-study");write(study,"test fixture")
    id=string(uuid4());clock=DateTime(2026,11,12,9,8,7,6)
    target=L.create_directory(root,id;study_path=study,clock)
    @test dirname(target)==realpath(study_parent)
    @test basename(target)=="20261112_090807_006_UTC_airfoil_"*id
    @test L.registered_directory(root,id)==target
    @test isdir(target)
    sentinel=joinpath(target,"native-field.vtu");write(sentinel,"original native field")
    @test_throws ArgumentError L.create_directory(root,id;study_path=study,clock)
    @test read(sentinel,String)=="original native field"
    @test L.registered_directory(root,string(uuid4()))===nothing
    @test_throws ArgumentError L.registered_directory(root,"../outside")
    @test_throws ArgumentError L.create_directory(root,string(uuid4());parent="relative")
    @test_throws ArgumentError L.create_directory(root,string(uuid4());study_path=study*".missing")
    pointer=joinpath(root,id,L.POINTER_FILE);original=read(pointer)
    open(io->TOML.print(io,Dict("job_id"=>id,"directory"=>study_parent)),pointer,"w")
    @test_throws ArgumentError L.registered_directory(root,id)
    write(pointer,original)
    @test L.registered_directory(root,id)==target
end

@testset "Actual API save, enqueue, reopen, cancel and download use registered folders" begin
    # Queue-only test: no CFD, panel run, subprocess or GPU is started.
    A.WORKER_STARTED[]=true
    parent=joinpath(ENV["LUDWIG_PRIVATE_ROOT"],"saved studies");mkpath(parent)
    study=joinpath(parent,"NACA2412.ludwig-study")
    saved=A._study_save(Dict("path"=>study,"etag"=>nothing,"request"=>case()))
    @test saved["path"]==study
    @test A._study_job_path(case(study=study))==study
    @test_throws ArgumentError A._study_job_path(case(study=study*".missing"))
    @test_throws ArgumentError A._study_job_path(Dict("study_path"=>42))
    ids=String[]
    explicit=joinpath(ENV["LUDWIG_PRIVATE_ROOT"],"chosen results")
    for (request,expected) in ((case(study=study),parent),(case(),A.RUN_DIR),(case(study=study,parent=explicit),explicit))
        response,data=call("POST","/api/jobs",request)
        @test response.status==202
        id=data["id"];push!(ids,id)
        dir=A._safe_job_dir(id)
        @test data["results_directory"]==dir
        @test dirname(dir)==realpath(expected)
        @test endswith(basename(dir),"_airfoil_"*id)
        @test A._get_job(id)["results_directory"]==dir
        @test isfile(joinpath(dir,A.JOB_RECORD_FILE))
        @test A._saved_job(id)["id"]==id
        write(joinpath(dir,"field.vtu"),"native field data")
        downloaded,_=call("GET","/api/jobs/$id/files/field.vtu")
        @test downloaded.status==200
        @test String(downloaded.body)=="native field data"
        forbidden,_=call("GET","/api/jobs/$id/files/../result_location.toml")
        @test forbidden.status!=200
        cancelled,_=call("POST","/api/jobs/$id/cancel",Dict())
        @test cancelled.status==202
        @test A._get_job(id)["status"]=="cancelled"
        @test A._saved_job(id)["status"]=="cancelled"
    end
    recent=A._recent_jobs()
    @test all(id->any(row->row["id"]==id,recent),ids)
    legacy_id=string(uuid4());legacy=joinpath(A.RUN_DIR,legacy_id);mkpath(legacy)
    A.JOBS[legacy_id]=Dict{String,Any}("id"=>legacy_id,"status"=>"completed","case"=>A._validated_case(case()))
    A._persist_job(legacy_id)
    @test A._safe_job_dir(legacy_id)==legacy
    @test A._saved_job(legacy_id)["id"]==legacy_id
    @test isfile(joinpath(legacy,A.JOB_RECORD_FILE))
    @test any(row->row["id"]==legacy_id,A._recent_jobs())
    pointer=joinpath(A.RUN_DIR,first(ids),L.POINTER_FILE);original=read(pointer)
    write(pointer,"damaged lookup metadata")
    @test !any(row->row["id"]==first(ids),A._recent_jobs())
    @test any(row->row["id"]==last(ids),A._recent_jobs())
    write(pointer,original)
    @test any(row->row["id"]==first(ids),A._recent_jobs())
    denied,_=call("POST","/api/jobs",case(study=study*".missing"))
    @test denied.status==400
    @test length(A.JOBS)==4
    linked=A._study_save(Dict("path"=>study,"etag"=>saved["etag"],"request"=>case(),"results"=>Dict("job_id"=>first(ids))))
    @test linked["study"]["results"]["directory"]==A._safe_job_dir(first(ids))
    @test A._study_load(study)["results_available"]
    health,data=call("GET","/api/health")
    @test health.status==200
    @test data["api_revision"]>=27
    @test data["result_storage"]["study_folder_default"]
    asset,_=call("GET","/static/scripts/workflow.js")
    @test asset.status==200
    @test occursin("data-workflow-target",String(asset.body))
end

@testset "Nested links cannot escape a registered study result folder" begin
    # All targets and links are generated private test fixtures. Directory links
    # are junctions on Windows and symlinks on Unix; no administrator is needed.
    request=case(parent=joinpath(ENV["LUDWIG_PRIVATE_ROOT"],"linked result parents"))
    response,data=call("POST","/api/jobs",request)
    @test response.status==202
    id=data["id"];root=A._safe_job_dir(id)
    outside=root*"_outside";mkpath(outside)
    write(joinpath(outside,"secret.bin"),"outside this job")
    nested=joinpath(root,"nested");mkpath(nested)
    escape=joinpath(nested,"escape")
    symlink(outside,escape;dir_target=true)
    @test realpath(escape)==realpath(outside)
    denied,_=call("GET","/api/jobs/$id/files/nested/escape/secret.bin")
    @test denied.status==403
    @test !occursin("outside this job",String(denied.body))
    allowed=joinpath(root,"allowed");mkpath(allowed)
    write(joinpath(allowed,"field.vtu"),"inside this job")
    alias=joinpath(nested,"inside")
    symlink(allowed,alias;dir_target=true)
    downloaded,_=call("GET","/api/jobs/$id/files/nested/inside/field.vtu")
    @test downloaded.status==200
    @test String(downloaded.body)=="inside this job"
    write(joinpath(allowed,"clip.mp4"),UInt8.(0:31))
    ranged=A._asset_response(id,"nested/inside/clip.mp4";range="bytes=2-5")
    @test ranged.status==206
    @test ranged.body==UInt8[2,3,4,5]
    @test dirname(root)!=A.RUN_DIR
    # A legacy UUID directory cannot itself be replaced with an external link.
    legacy_id=string(uuid4())
    symlink(outside,joinpath(A.RUN_DIR,legacy_id);dir_target=true)
    legacy_denied,_=call("GET","/api/jobs/$legacy_id/files/secret.bin")
    @test legacy_denied.status==403
    @test !occursin("outside this job",String(legacy_denied.body))
    terminal,_=call("POST","/api/jobs/$id/cancel",Dict())
    @test terminal.status==202
end
