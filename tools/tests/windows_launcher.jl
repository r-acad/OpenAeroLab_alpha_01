using Test, UUIDs
const REPO=normpath(joinpath(@__DIR__,"..",".."))
include(joinpath(REPO,"tools","install","launch_windows.jl"))
const Launcher=OpenAeroLabWindowsLauncher
const PRIVATE=abspath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(REPO,"..","02_PRIVATE_PROJECT_DEVELOPMENT")))
const SCRATCH=joinpath(PRIVATE,"TEMP_DELETEABLE","windows_launcher_tests",string(uuid4()))
mkpath(SCRATCH)

@testset "Live console and persistent combined startup log" begin
    child=joinpath(SCRATCH,"startup fixture.jl")
    write(child,"println(\"stdout marker\"); flush(stdout); println(stderr,\"stderr marker\"); flush(stderr); println(join(ARGS,\"|\")); exit(7)\n")
    log=joinpath(SCRATCH,"private","runtime","launch_logs","failure.log")
    console=IOBuffer()
    code=Launcher.launch_logged(`$(Base.julia_cmd()) --startup-file=no $child "argument with spaces" tail`,log;console)
    @test code==7
    visible=String(take!(console));retained=read(log,String)
    @test occursin("stdout marker",visible)
    @test occursin("stderr marker",visible)
    @test occursin("argument with spaces|tail",visible)
    @test occursin(visible,retained)
    @test occursin("exit code 7",retained)
end

if Sys.iswindows()
    @testset "Native launcher forwarding and durable error log" begin
        fixture=joinpath(SCRATCH,"repository (with spaces)")
        install=joinpath(fixture,"tools","install");mkpath(install)
        for file in ("launch_windows.cmd","launch_windows.jl","resolve_julia.cmd","julia_version.txt")
            cp(joinpath(REPO,"tools","install",file),joinpath(install,file))
        end
        entry=joinpath(fixture,"apps","ludwig_web","run.jl");mkpath(dirname(entry))
        write(entry,"println(\"CLI stdout: \",join(ARGS,\"|\")); println(stderr,\"CLI stderr startup failure\"); exit(7)\n")
        project=joinpath(fixture,"third_party","JXfoil");mkpath(project)
        write(joinpath(project,"Project.toml"),"[deps]\n")
        wrapper=joinpath(SCRATCH,"invoke_launcher.cmd")
        write(wrapper,"@echo off\r\nsetlocal DisableDelayedExpansion\r\nset OA_APP=workbench\r\ncall \"$(joinpath(install,"launch_windows.cmd"))\" \"argument with spaces\" tail\r\nexit /b %ERRORLEVEL%\r\n")
        private=joinpath(SCRATCH,"native private")
        command=Cmd(Cmd([ENV["COMSPEC"],"/d","/c",wrapper]);ignorestatus=true,windows_hide=true)
        console=IOBuffer()
        child=run(pipeline(addenv(command,"LUDWIG_JULIA"=>joinpath(Sys.BINDIR,"julia.exe"),"LUDWIG_PRIVATE_ROOT"=>private,"LUDWIG_WEB_NO_PAUSE"=>"1");stdout=console,stderr=console))
        visible=String(take!(console))
        @test child.exitcode==7
        @test occursin("CLI stdout: argument with spaces|tail",visible)
        @test occursin("CLI stderr startup failure",visible)
        @test occursin("Startup / server log:",visible)
        files=readdir(joinpath(private,"runtime","launch_logs");join=true)
        @test length(files)==1
        retained=read(only(files),String)
        @test occursin("CLI stdout: argument with spaces|tail",retained)
        @test occursin("CLI stderr startup failure",retained)
        @test occursin("exit code 7",retained)
    end

    @testset "Interrupted launcher terminates its own process tree" begin
        pidfile=joinpath(SCRATCH,"grandchild.pid")
        entry=joinpath(SCRATCH,"long running fixture.jl")
        write(entry,raw"""
        grandchild=run(`$(Base.julia_cmd()) --startup-file=no -e "sleep(30)"`;wait=false)
        write(ARGS[1]*".tmp",string(getpid(grandchild)))
        mv(ARGS[1]*".tmp",ARGS[1])
        sleep(30)
        """)
        ownchild=Ref{Any}(nothing)
        callback=child->begin
            ownchild[]=child
            deadline=time()+10
            while !isfile(pidfile) && time()<deadline; sleep(0.05); end
            isfile(pidfile) || error("Child did not initialize its descendant fixture")
            throw(InterruptException())
        end
        log=joinpath(SCRATCH,"interrupted.log")
        code=Launcher.launch_logged(`$(Base.julia_cmd()) --startup-file=no $entry $pidfile`,log;console=IOBuffer(),on_started=callback)
        @test code==130
        @test !process_running(ownchild[])
        pid=parse(Int,read(pidfile,String))
        tasklist=joinpath(ENV["SystemRoot"],"System32","tasklist.exe")
        listing=read(Cmd(Cmd([tasklist,"/FI","PID eq $pid","/FO","CSV","/NH"]);windows_hide=true),String)
        @test !occursin("\"$pid\"",listing)
        @test occursin("Interrupted by user",read(log,String))
        @test occursin("exit code 130",read(log,String))
    end
end
println("Disposable launcher fixtures: ",SCRATCH)
