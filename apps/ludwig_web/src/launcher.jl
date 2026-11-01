module WorkbenchLauncher

using HTTP, JSON3, Dates, UUIDs
using FileWatching: Pidfile
include(joinpath(@__DIR__,"workbench_revision.jl"))

function marker_path(public_root::AbstractString)
    private_root = normpath(get(ENV,"LUDWIG_PRIVATE_ROOT",
        joinpath(public_root,"..","02_PRIVATE_PROJECT_DEVELOPMENT")))
    joinpath(private_root,"TEMP_DELETEABLE","ludwig_web","current_server.json")
end

_valid_port(port) = port isa Integer && !(port isa Bool) && 1 <= port <= 65535
_url(port::Integer) = "http://127.0.0.1:$port/"

"Wait for the browser handoff before a reused-server launcher is allowed to exit."
function open_browser(url::AbstractString; timeout::Real=10,
                      command::Union{Nothing,Cmd}=nothing)
    occursin(r"^http://127\.0\.0\.1:[0-9]{1,5}/$",url) ||
        throw(ArgumentError("Browser URL must be the local workbench"))
    Sys.iswindows() || return false
    timeout > 0 || throw(ArgumentError("Browser handoff timeout must be positive"))
    if command===nothing
        script="\$ErrorActionPreference = 'Stop'; Start-Process -FilePath '$url' -ErrorAction Stop"
        command=Cmd(Cmd(["powershell.exe","-NoProfile","-NonInteractive","-WindowStyle","Hidden",
            "-Command",script]);windows_hide=true)
    end
    try
        process=run(command;wait=false)
        if timedwait(()->process_exited(process),timeout;pollint=0.05)===:timed_out
            @warn "Browser handoff did not finish in time; open the workbench URL manually" url
            return false
        end
        if !success(process)
            @warn "Default browser could not be opened; open the workbench URL manually" url
            return false
        end
        println("Browser handoff completed for $url")
        flush(stdout)
        true
    catch error
        @warn "Default browser could not be opened; the workbench remains available" url exception=error
        false
    end
end

"Check the live API, rather than trusting a stale PID, URL or saved marker."
function is_current_server(port::Integer, public_root::AbstractString)
    _valid_port(port) || return false
    try
        response = HTTP.get(_url(port)*"api/aero/server-info";
            status_exception=false,readtimeout=1,connect_timeout=1,retry=false)
        response.status == 200 || return false
        data = JSON3.read(String(response.body))
        data isa AbstractDict || return false
        get(data,"api_revision",nothing) == WORKBENCH_API_REVISION &&
            get(data,"repository_id",nothing) == _workbench_repository_id(public_root)
    catch error
        # An unavailable/obsolete server is expected. A programming/API error
        # must stay visible instead of silently creating another server.
        (error isa TypeError || error isa MethodError) && rethrow()
        false
    end
end

function find_current_server(marker::AbstractString, public_root::AbstractString, preferred_port::Integer;
                             probe::Function=is_current_server)
    ports = Int[]
    if isfile(marker)
        try
            data = JSON3.read(read(marker,String))
            port = get(data,"port",nothing)
            if _valid_port(port) && get(data,"api_revision",nothing) == WORKBENCH_API_REVISION &&
               get(data,"repository_id",nothing) == _workbench_repository_id(public_root)
                push!(ports,Int(port))
            end
        catch
            # An interrupted or obsolete marker must never prevent a launch.
        end
    end
    append!(ports,preferred_port:min(preferred_port+19,65535))
    for port in unique(ports)
        probe(port,public_root) && return _url(port)
    end
    nothing
end

function write_current_server!(marker::AbstractString,public_root::AbstractString,port::Integer)
    _valid_port(port) || throw(ArgumentError("Invalid workbench port"))
    mkpath(dirname(marker))
    data=Dict("api_revision"=>WORKBENCH_API_REVISION,
        "repository_id"=>_workbench_repository_id(public_root),"public_root"=>normpath(abspath(public_root)),
        "port"=>port,"url"=>_url(port),"pid"=>getpid(),
        "started_utc"=>Dates.format(Dates.now(Dates.UTC),"yyyy-mm-ddTHH:MM:SSZ"))
    temporary = String(marker)*".$(uuid4()).tmp"
    write(temporary,JSON3.write(data))
    Base.Filesystem.rename(temporary,String(marker))
    data
end

function startup_lock(marker::AbstractString)
    mkpath(dirname(marker))
    # Released as soon as the new server is listening, so concurrent launchers
    # see and reuse it. A dead launcher's advisory PID lock can expire safely.
    Pidfile.mkpidlock(String(marker)*".startup.lock";stale_age=180,refresh=30,poll_interval=1)
end

end
