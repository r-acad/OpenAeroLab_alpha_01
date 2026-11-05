module OpenAeroLabWindowsLauncher
using Dates, UUIDs

function stop_child!(child)
    process_running(child) || return
    if Sys.iswindows()
        # Only the process created by this launcher and its descendants are
        # stopped. Closing the console must not leave solver children behind.
        taskkill=joinpath(get(ENV,"SystemRoot",raw"C:\Windows"),"System32","taskkill.exe")
        command=Cmd(Cmd([taskkill,"/PID",string(getpid(child)),"/T","/F"]);ignorestatus=true,windows_hide=true)
        try run(pipeline(command,stdout=devnull,stderr=devnull)) catch end
    end
    process_running(child) && kill(child)
    wait(child)
end

"""Stream both child output channels to the console and a persistent log."""
function launch_logged(command, logpath; console=stdout, on_started=child->nothing)
    mkpath(dirname(logpath))
    pipe=Pipe()
    child=nothing
    interrupted=false
    exitcode=1
    open(logpath,"w") do log
        println(log,"OpenAeroLab launch started at ",now(UTC)," UTC");flush(log)
        try
            child=run(pipeline(ignorestatus(command),stdout=pipe,stderr=pipe);wait=false)
            close(pipe.in)
            on_started(child)
            while !eof(pipe)
                bytes=readavailable(pipe)
                write(log,bytes);flush(log)
                write(console,bytes);flush(console)
            end
            wait(child)
            exitcode=child.exitcode
        catch err
            if err isa InterruptException
                interrupted=true
                println(log,"\nInterrupted by user; stopping the application and its child processes.")
            else
                showerror(log,err,catch_backtrace());println(log)
                rethrow()
            end
        finally
            # Cleanup is also required when spawning, reading or writing fails.
            Base.disable_sigint() do
                child===nothing || stop_child!(child)
                isopen(pipe.in) && close(pipe.in)
                isopen(pipe.out) && close(pipe.out)
            end
            exitcode=interrupted ? 130 : exitcode
            println(log,"\nOpenAeroLab launch ended at ",now(UTC)," UTC; exit code ",exitcode)
            flush(log)
        end
    end
    exitcode
end

function main(args=ARGS)
    length(args)>=3 || error("Usage: launch_windows.jl APP PROJECT ENTRY [application arguments...]")
    app,project,entry=args[1:3]
    app in ("workbench","airfoil","solver") || error("Unknown OpenAeroLab application: $app")
    isfile(entry) || error("Application entry point is missing: $entry. Extract the complete repository.")
    repository=normpath(joinpath(@__DIR__,"..",".."))
    private=abspath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(repository,"..","02_PRIVATE_PROJECT_DEVELOPMENT")))
    stamp=Dates.format(now(UTC),"yyyymmdd_HHMMSS")*"_"*string(uuid4())
    logpath=joinpath(private,"runtime","launch_logs",app*"_"*stamp*".log")
    println("Startup / server log: ",logpath);flush(stdout)
    executable=joinpath(Sys.BINDIR,Sys.iswindows() ? "julia.exe" : "julia")
    command=Cmd(Cmd([executable,"--startup-file=no","--project="*project,"--threads=auto",entry,args[4:end]...]);windows_hide=Sys.iswindows())
    code=launch_logged(command,logpath)
    code==0 || println(stderr,"OpenAeroLab exited with code $code. Details: $logpath")
    code
end
end

if abspath(PROGRAM_FILE)==@__FILE__
    try
        exit(OpenAeroLabWindowsLauncher.main())
    catch err
        showerror(stderr,err);println(stderr)
        exit(1)
    end
end
