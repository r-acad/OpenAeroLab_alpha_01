include(joinpath(@__DIR__, "server.jl"))
using .AirfoilWeb

preferred_port = parse(Int, get(ENV, "AIRFOIL_WEB_PORT", "8058"))
1 <= preferred_port <= 65535 || error("AIRFOIL_WEB_PORT must be between 1 and 65535")

function open_browser(url::String)
    Sys.iswindows() || return
    occursin(r"^http://127\.0\.0\.1:[0-9]{1,5}/$",url) || error("Browser URL must be the local Airfoil Lab server")
    # A browser's Windows DDE handler can block ShellExecuteW indefinitely.
    # Keep that native call out of Julia's server thread. The hidden helper
    # launches the user's default browser while Julia continues serving.
    command=Cmd(Cmd(["powershell.exe","-NoProfile","-NonInteractive","-WindowStyle","Hidden",
        "-Command","Start-Process -FilePath '$url'"]);windows_hide=true)
    process=run(command;wait=false)
    @async begin
        wait(process)
        success(process) || @warn "Default browser launch failed; open the printed URL manually" url
    end
    println("Requested default browser at $url")
    flush(stdout)
end

open_on_start = get(ENV, "AIRFOIL_WEB_NO_BROWSER", "") != "1"
started = false
for port in preferred_port:min(preferred_port + 19, 65535)
    url = "http://127.0.0.1:$port/"
    try
        AirfoilWeb.serve(; port=port,
            on_listening=open_on_start ? ()->open_browser(url) : ()->nothing)
        global started = true
        break
    catch err
        if occursin("EADDRINUSE", sprint(showerror, err))
            println("Port $port is busy; trying the next port.")
        else
            rethrow()
        end
    end
end
started || error("No free port was found from $preferred_port through $(min(preferred_port + 19, 65535))")
