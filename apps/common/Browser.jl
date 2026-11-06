module DesktopBrowser
"Open a loopback web app with the operating system's default browser."
function open_browser(url::AbstractString; timeout::Real=10, command::Union{Nothing,Cmd}=nothing)
    address=match(r"^http://127\.0\.0\.1:([0-9]{1,5})/(?:aero-lab(?:\?job=[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}(?:&reporting=1)?)?)?\z",url)
    address!==nothing&&1<=parse(Int,address[1])<=65535 || throw(ArgumentError("Browser URL must be a local workbench or Aero Lab case"))
    timeout>0 || throw(ArgumentError("Browser timeout must be positive"))
    if command===nothing
        if Sys.iswindows()
            command=Cmd(Cmd(["powershell.exe","-NoProfile","-NonInteractive","-WindowStyle","Hidden",
                "-Command","Start-Process -FilePath '$url' -ErrorAction Stop"]);windows_hide=true)
        elseif Sys.isapple()
            command=`open $url`
        else
            opener=Sys.which("xdg-open")
            if opener===nothing
                println("Open $url in your browser (xdg-open is unavailable).");return false
            end
            command=`$opener $url`
        end
    end
    try
        process=run(command;wait=false)
        if timedwait(()->process_exited(process),timeout;pollint=.05)===:timed_out
            @warn "Browser handoff timed out; open the printed URL manually" url
            return false
        end
        success(process) || (@warn "Could not open the browser; open the printed URL manually" url; return false)
        println("Browser handoff completed for $url");flush(stdout);true
    catch err
        @warn "Browser could not be opened; the service remains available" url exception=err
        false
    end
end
end
