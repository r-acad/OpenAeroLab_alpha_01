using Test
const REPOSITORY=dirname(dirname(@__DIR__))
include(joinpath(REPOSITORY,"apps","airfoil_web","src","protocol.jl"))
include(joinpath(REPOSITORY,"apps","airfoil_web","src","full_field_artifact.jl"))
using .AirfoilWire, .FullFieldArtifact
const FS=FullFieldArtifact.FieldSnapshots
const SCRATCH=joinpath(FS.scratch_root(),"tests")
mkpath(SCRATCH)

@testset "Wall-clock scheduling, no iteration coupling" begin
    clock=Ref(0.0);scheduler=FS.Schedule(;clock=()->clock[])
    writes=Ref(0);record()=(writes[]+=1)
    for step in 1:1000
        clock[]=step*.1
        @test !FS.save_if_due!(record,scheduler;log=_ -> nothing)
    end
    clock[]=599.999;@test !FS.save_if_due!(record,scheduler)
    clock[]=600.;@test FS.save_if_due!(record,scheduler;log=_ -> nothing)
    @test writes[]==1
    clock[]=1800.;@test FS.save_if_due!(record,scheduler;log=_ -> nothing)
    @test !FS.save_if_due!(record,scheduler)
    @test scheduler.next_due==2400.
    @test writes[]==2
    off=FS.Schedule(0;clock=()->clock[])
    clock[]=1e9;@test !FS.save_if_due!(record,off)
    for invalid in (true,-1,Inf,NaN,86401,"600")
        @test_throws ArgumentError FS.Schedule(invalid)
    end
    clock[]=0.;failed=FS.Schedule(2;clock=()->clock[]);messages=String[]
    clock[]=2.;@test !FS.save_if_due!(() -> error("disk full"),failed;log=s->push!(messages,s))
    @test failed.failed==1 && failed.saved==0
    @test occursin("not saved",only(messages))
    @test !FS.save_if_due!(() -> error("must not retry immediately"),failed)
    clock[]=4.;@test FS.save_if_due!(record,failed;log=_ -> nothing)
end

@testset "Complete artifacts, chunk publication, low-disk failure" begin
    mktempdir(SCRATCH) do dir
        mesh=joinpath(dir,"mesh.msgpack");write(mesh,UInt8[1,2,3])
        at(i,j)=ntuple(k->Float64(i+j+k),11)
        path=joinpath(dir,"live_field_000001.msgpack")
        info=FullFieldArtifact.write_full_field(at,path,4,3,mesh,1)
        f=AirfoilWire.decode(read(path),:msgpack)
        @test info["bytes"]==filesize(path)>0
        @test f["iteration"]==1 && length(f["samples"]["cp"])==12
        multipart=joinpath(dir,"live_field_000002.msgpack")
        parts=FullFieldArtifact.write_full_field(at,multipart,4,10,mesh,2;max_bytes=9000)
        manifest=AirfoilWire.decode(read(multipart),:msgpack)
        @test manifest["kind"]=="structured_cells_manifest"
        @test parts["part_count"]>1
        @test all(p->filesize(joinpath(dir,p["path"]))==p["bytes"],manifest["parts"])
        failed=joinpath(dir,"live_field_000003.msgpack")
        @test_throws ErrorException FullFieldArtifact.write_full_field(
            (i,j)->j>3 ? error("bad field") : at(i,j),failed,4,10,mesh,3;max_bytes=9000)
        @test !isfile(failed)
        @test !any(n->startswith(n,"live_field_000003"),readdir(dir))
        refused=joinpath(dir,"no-space.msgpack")
        @test_throws ArgumentError FS.atomic_write(refused,UInt8[1];reserve_bytes=typemax(Int)-1)
        @test !isfile(refused)
        FS.atomic_write(refused,UInt8[1,2,3];reserve_bytes=0)
        FS.atomic_write(refused,UInt8[4,5];reserve_bytes=0)
        @test read(refused)==UInt8[4,5]
        @test startswith(FS.scratch_root(),joinpath(dirname(REPOSITORY),"02_PRIVATE_PROJECT_DEVELOPMENT","TEMP_DELETEABLE")) || haskey(ENV,"LUDWIG_PRIVATE_ROOT")
    end
end

const APP_TEST_PRIVATE=mktempdir(SCRATCH)
withenv("LUDWIG_PRIVATE_ROOT"=>APP_TEST_PRIVATE) do
    include(joinpath(REPOSITORY,"apps","airfoil_web","src","server.jl"))
end
const A=AirfoilWeb
@testset "Backend preferences and persistent storage" begin
    @test A._output_preferences(Dict())["snapshot_interval_minutes"]==10
    @test A._output_preferences(Dict("outputs"=>Dict("snapshot_interval_minutes"=>0.5)))["snapshot_interval_minutes"]==0.5
    for value in (-1,true,Inf,"10",1441)
        @test_throws ArgumentError A._output_preferences(Dict("outputs"=>Dict("snapshot_interval_minutes"=>value)))
    end
    input=Dict("airfoil"=>Dict("kind"=>"naca4","id"=>"0012"),"backends"=>["transonic"],
        "flow"=>Dict("mach"=>.7,"reynolds"=>6e6,"alpha_deg"=>2.79),"display_every"=>100,
        "outputs"=>Dict("snapshot_interval_minutes"=>2.5))
    case=A._validated_case(input)
    @test case["flow"]["snapshot_interval_seconds"]==150.
    @test case["flow"]["cfd_history_stride"]==100
    @test !haskey(case["flow"],"lowspeed_live_every")
    @test relpath(A.RUN_DIR,A.PRIVATE_ROOT)==joinpath("results","airfoil_runs")
    @test startswith(relpath(A.PREVIEW_DIR,A.PRIVATE_ROOT),"TEMP_DELETEABLE")
    cfg=A.CFDBackend.lowspeed_configuration(Dict("reynolds"=>6e6,"snapshot_interval_seconds"=>7.))
    @test cfg["snapshot_interval_seconds"]==7.
end

@testset "Native checkpoints follow numeric export preferences and download as binary" begin
    settings=A._output_preferences(Dict())
    @test A._selected_export("state_checkpoint.pbrestart",settings)
    @test A._selected_export("STATE_CHECKPOINT.PBRESTART",settings)
    @test !A._selected_export("state_checkpoint.pbrestart",merge(settings,Dict("numeric_slices"=>false)))
    @test A._output_capabilities()["lowspeed"]["volume_vtk"]===true
    id=string(A.uuid4());dir=A._safe_job_dir(id);mkpath(dir)
    bytes=Vector{UInt8}(codeunits("PBRestart/1 test download\n"))
    write(joinpath(dir,"state_checkpoint.pbrestart"),bytes)
    response=A.app_handler(A.HTTP.Request("GET","/api/jobs/$id/files/state_checkpoint.pbrestart"))
    @test response.status==200
    @test A.HTTP.header(response,"Content-Type")=="application/octet-stream"
    @test response.body==bytes
    @test A._asset_response(id,"../outside.pbrestart").status==403
    @test A._asset_response(id,"missing.pbrestart").status==404
end
