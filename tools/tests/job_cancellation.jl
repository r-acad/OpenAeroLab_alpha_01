using Test, UUIDs, TOML
const REPOSITORY=normpath(joinpath(@__DIR__,"..",".."))
const SCRATCH=normpath(joinpath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(REPOSITORY,"..","02_PRIVATE_PROJECT_DEVELOPMENT")),"TEMP_DELETEABLE","job_cancellation_tests"))
mkpath(SCRATCH)
const FIXTURE=mktempdir(SCRATCH)
ENV["LUDWIG_PRIVATE_ROOT"]=FIXTURE
include(joinpath(REPOSITORY,"apps","airfoil_web","src","server.jl"))
const A=AirfoilWeb
const JC=A.AirfoilJobControl

function test_job(status="queued";sweep=false)
    case=A._validated_case(Dict("airfoil"=>Dict("kind"=>"naca4","id"=>"0012"),"backends"=>["lowspeed"],
        "flow"=>Dict("reynolds"=>3e6,"alpha_deg"=>4.,"lowspeed_resolution"=>"demo","lowspeed_max_steps"=>20),
        "sweep"=>Dict("enabled"=>sweep,"start"=>4.,"stop"=>6.,"step"=>2.)))
    id=string(uuid4())
    lock(A.JOB_LOCK) do
        A.JOBS[id]=Dict{String,Any}("id"=>id,"status"=>status,"message"=>status,"progress"=>0.,"case"=>case)
        A.JOB_CONTROLS[id]=JC.CancellationToken(marker=joinpath(A._safe_job_dir(id),"cancel.requested"),grace_seconds=1.)
        A.JOB_LOGS[id]=Dict{String,Any}[];A.JOB_LOG_SEQ[id]=0
        A.JOB_HISTORY[id]=Dict{String,Any}[];A.JOB_HISTORY_SEQ[id]=0
    end
    id
end
cancel_response(id)=A.app_handler(A.HTTP.Request("POST","/api/jobs/$id/cancel",["Accept"=>"application/msgpack"]))
payload(response)=A.AirfoilWire.decode(copy(response.body),:msgpack)

@testset "Queued and terminal cancellation API retains artifacts" begin
    id=test_job();dir=A._safe_job_dir(id);mkpath(dir)
    saved=joinpath(dir,"saved_field.bin");write(saved,UInt8[1,2,3])
    response=cancel_response(id)
    @test response.status==202
    @test payload(response)["status"]=="cancelled"
    @test payload(response)["cancel_requested"]===true
    @test read(saved)==UInt8[1,2,3]
    A._run_job(id) # A queued cancellation must never start the solver.
    @test A._get_job(id)["status"]=="cancelled"
    @test cancel_response(id).status==200
    @test A._saved_job(id)["status"]=="cancelled"
    done=test_job("completed")
    @test payload(cancel_response(done))["status"]=="completed"
    @test !JC.requested(A.JOB_CONTROLS[done])
    @test cancel_response(string(uuid4())).status==404
    @test cancel_response("bad-id").status==404
end

function pid_active(pid)
    if Sys.iswindows()
        handle=ccall((:OpenProcess,"kernel32"),Ptr{Cvoid},(UInt32,Int32,UInt32),0x1000,0,pid)
        handle==C_NULL && return false
        try
            code=Ref{UInt32}(0)
            ccall((:GetExitCodeProcess,"kernel32"),Int32,(Ptr{Cvoid},Ref{UInt32}),handle,code)!=0 && code[]==259
        finally
            ccall((:CloseHandle,"kernel32"),Int32,(Ptr{Cvoid},),handle)
        end
    else
        ccall(:kill,Int32,(Int32,Int32),pid,0)==0
    end
end

@testset "Active GPU-wait worker and descendants stop; sweep and unrelated jobs do not run" begin
    grandchild=joinpath(FIXTURE,"grandchild.jl")
    write(grandchild,"write(ARGS[1],string(getpid()));sleep(90)\n")
    worker=joinpath(FIXTURE,"waiting_worker.jl")
    write(worker,"using TOML\ncfg=TOML.parsefile(ARGS[1]);dir=cfg[\"outdir\"]\n"*
        "write(joinpath(dir,\"saved_field.bin\"),UInt8[9,8,7])\n"*
        "run(pipeline(`\$(Base.julia_cmd()) --startup-file=no \$("*repr(grandchild)*") \$(joinpath(dir,\"grandchild.pid\"))`;stdout=devnull,stderr=devnull);wait=false)\n"*
        "write(joinpath(dir,\"worker_started.txt\"),string(getpid()))\nprintln(\"GPU_WAIT,another test owner\");flush(stdout);sleep(90)\n")
    sentinel=run(pipeline(JC.process_command(`$(Base.julia_cmd()) --startup-file=no $grandchild $(joinpath(FIXTURE,"sentinel.pid"))`);stdout=devnull,stderr=devnull);wait=false)
    sentinel_owned=JC.track!(sentinel,nothing)
    lockdir=A.CFDBackend.AirfoilGPULock.LOCK_DIR
    @test startswith(lockdir,FIXTURE)
    mkpath(lockdir);write(joinpath(lockdir,"owner.txt"),"another GPU owner's lock\n")
    try
        snapshot=joinpath(REPOSITORY,"solvers","pbcav_s1030")
        withenv("AIRFOIL_LOWSPEED_WORKER"=>worker,"AIRFOIL_LOWSPEED_CH_WORKER"=>worker,
                "AIRFOIL_LOWSPEED_SNAPSHOT"=>snapshot,"AIRFOIL_LOWSPEED_CH_SNAPSHOT"=>snapshot) do
            id=test_job(;sweep=true)
            task=@async A._run_job(id)
            dir=A._safe_job_dir(id)
            find_marker()=[joinpath(folder,name) for (folder,_,files) in walkdir(dir) for name in files if name=="grandchild.pid"]
            @test timedwait(()->(isdir(dir)&&!isempty(find_marker()))||istaskdone(task),45.;pollint=.1)==:ok
            markers=find_marker()
            if isempty(markers)
                println(read(joinpath(dir,"job.log"),String))
                error("worker did not start")
            end
            @test length(markers)==1
            childpid=parse(Int,read(only(markers),String))
            @test pid_active(childpid)
            response=cancel_response(id)
            @test response.status==202
            @test payload(response)["status"] in ("cancelling","cancelled")
            @test timedwait(()->istaskdone(task),15.;pollint=.1)==:ok
            wait(task)
            job=A._get_job(id)
            @test job["status"]=="cancelled"
            @test !haskey(job,"error")
            @test !pid_active(childpid)
            @test length(find_marker())==1 # No second operating point started.
            @test process_running(sentinel)
            @test read(joinpath(dirname(only(markers)),"saved_field.bin"))==UInt8[9,8,7]
            @test A._saved_job(id)["status"]=="cancelled"
            @test read(joinpath(lockdir,"owner.txt"),String)=="another GPU owner's lock\n"
        end
    finally
        foreach(JC.cancel!,values(A.JOB_CONTROLS))
        JC.finish!(sentinel_owned);wait(sentinel)
        isdir(lockdir) && rm(lockdir;recursive=true)
    end
end

@testset "Cooperative cancellation saves state and releases only its GPU lock" begin
    ready=joinpath(FIXTURE,"cooperative_ready.pid")
    saved=joinpath(FIXTURE,"cooperative_checkpoint.bin")
    lockdir=A.CFDBackend.AirfoilGPULock.LOCK_DIR
    script=joinpath(FIXTURE,"cooperative_worker.jl")
    write(script,"dir=ARGS[3];mkpath(dir)\n"*
        "write(joinpath(dir,\"owner.txt\"),\"Airfoil Lab low-speed CFD worker; pid \$(getpid()); cooperative test\\n\")\n"*
        "write(ARGS[1],string(getpid()))\nwhile !isfile(ENV[\"AIRFOIL_CANCEL_FILE\"]);sleep(.02);end\n"*
        "write(ARGS[2],UInt8[4,5,6]);println(\"cooperative checkpoint saved\");flush(stdout)\n")
    token=JC.CancellationToken(marker=joinpath(FIXTURE,"cooperative.cancel"),grace_seconds=10.)
    task=@async JC.with_token(token) do
        A.CFDBackend._log_process(`$(Base.julia_cmd()) --startup-file=no $script $ready $saved $lockdir`,
            joinpath(FIXTURE,"cooperative.log"),_->nothing,_->nothing,"lowspeed",4.)
    end
    try
        @test timedwait(()->isfile(ready)||istaskdone(task),20.;pollint=.1)==:ok
        @test isfile(ready)
        JC.cancel!(token)
        @test isfile(token.marker)
        @test timedwait(()->istaskdone(task),8.;pollint=.1)==:ok
        @test fetch(task)===true
        @test read(saved)==UInt8[4,5,6]
        @test !isdir(lockdir)
    finally
        JC.cancel!(token)
    end
end
