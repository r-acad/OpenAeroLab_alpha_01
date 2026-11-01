include(joinpath(@__DIR__, "src", "server.jl"))
using .AirfoilWeb

preferred_port = parse(Int, get(ENV, "AIRFOIL_WEB_PORT", "8058"))
1 <= preferred_port <= 65535 || error("AIRFOIL_WEB_PORT must be between 1 and 65535")

include(joinpath(@__DIR__,"..","common","Browser.jl"))
open_browser(url::String) = DesktopBrowser.open_browser(url)

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
