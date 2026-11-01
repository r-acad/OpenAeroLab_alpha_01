include(joinpath(@__DIR__,"src","launcher.jl"))
using .WorkbenchLauncher

preferred_port = parse(Int,get(ENV,"LUDWIG_WEB_PORT","8058"))
1 <= preferred_port <= 65535 || error("LUDWIG_WEB_PORT must be between 1 and 65535")
public_root = normpath(joinpath(@__DIR__,"..",".."))
marker = WorkbenchLauncher.marker_path(public_root)
open_on_start = get(ENV,"LUDWIG_WEB_NO_BROWSER","") != "1"

function open_workbench(url::String)
    if open_on_start
        WorkbenchLauncher.open_browser(url)
    else
        println("Automatic browser opening is disabled (LUDWIG_WEB_NO_BROWSER=1). Open $url")
        flush(stdout)
    end
end

lock_ref = Ref{Any}(WorkbenchLauncher.startup_lock(marker))
started = Ref(false)
function release_startup_lock!()
    lock_ref[] === nothing && return
    close(lock_ref[])
    lock_ref[] = nothing
end

try
    existing = WorkbenchLauncher.find_current_server(marker,public_root,preferred_port)
    if existing !== nothing
        release_startup_lock!()
        println("Reusing current Ludwig Workbench at $existing")
        flush(stdout)
        open_workbench(existing)
    else
        println("Starting Ludwig Workbench; loading the application…")
        flush(stdout)
        include(joinpath(@__DIR__,"src","server.jl"))
        started[]=false
        for port in preferred_port:min(preferred_port+19,65535)
            url="http://127.0.0.1:$port/"
            try
                listening = () -> begin
                    try
                        WorkbenchLauncher.write_current_server!(marker,public_root,port)
                    finally
                        release_startup_lock!()
                    end
                    open_workbench(url)
                end
                Base.invokelatest(LudwigWeb.serve;port,on_listening=listening)
                started[]=true
                break
            catch err
                if occursin("EADDRINUSE",sprint(showerror,err))
                    if WorkbenchLauncher.is_current_server(port,public_root)
                        release_startup_lock!()
                        println("Reusing current Ludwig Workbench at $url")
                        flush(stdout)
                        open_workbench(url)
                        started[]=true
                        break
                    end
                    println("Port $port belongs to another or older server; trying the next port.")
                else
                    rethrow()
                end
            end
        end
        started[] || error("No free workbench port found")
    end
finally
    release_startup_lock!()
end
