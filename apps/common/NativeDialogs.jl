module NativeDialogs

using HTTP, JSON3

const DIALOG_LOCK = ReentrantLock()
const SCRIPT = joinpath(@__DIR__, "native_folder_dialog.ps1")

"Only a same-origin JSON request may open an interactive desktop dialog."
function _validate_request(req)
    headers = Dict(lowercase(String(k)) => String(v) for (k,v) in req.headers)
    startswith(lowercase(get(headers,"content-type","")),"application/json") ||
        throw(ArgumentError("Folder selection requires a JSON request"))
    origin = get(headers,"origin","")
    host = get(headers,"host","")
    if !isempty(origin)
        uri = HTTP.URI(origin)
        uri.scheme == "http" && lowercase(String(uri.host)) in ("localhost","127.0.0.1","::1") ||
            throw(ArgumentError("Folder selection is available only from the local app"))
        !isempty(host) && lowercase(origin) != "http://"*lowercase(host) &&
            throw(ArgumentError("Folder selection must use the current app origin"))
    end
    length(req.body) <= 8192 || throw(ArgumentError("Folder selection request is too large"))
    # String(::Vector{UInt8}) takes ownership and empties the vector. Validation
    # must leave the HTTP body available to the route that consumes it next.
    raw = try JSON3.read(String(copy(req.body))) catch; throw(ArgumentError("Invalid folder selection request")) end
    raw isa AbstractDict || throw(ArgumentError("Folder selection requires an object"))
    path = get(raw,"path","")
    path isa AbstractString && length(path)<=2000 && !occursin('\0',path) ||
        throw(ArgumentError("Invalid starting folder"))
    title = get(raw,"title","Choose folder")
    title isa AbstractString && length(title)<=160 && !occursin('\0',title) ||
        throw(ArgumentError("Invalid dialog title"))
    (String(path),String(title))
end

function _windows_folder(path,title,runtime)
    Sys.iswindows() || throw(ArgumentError("The native folder dialog is currently available on Windows; enter a folder path directly on this system"))
    mkpath(runtime)
    result, io = mktemp(runtime); close(io)
    try
        powershell = joinpath(get(ENV,"SystemRoot",raw"C:\Windows"),"System32","WindowsPowerShell","v1.0","powershell.exe")
        cmd = `$powershell -NoProfile -NonInteractive -STA -WindowStyle Hidden -ExecutionPolicy Bypass -File $SCRIPT`
        process = run(ignorestatus(addenv(cmd,"LUDWIG_DIALOG_PATH"=>path,
            "LUDWIG_DIALOG_TITLE"=>title,"LUDWIG_DIALOG_RESULT"=>result));wait=false)
        if timedwait(() -> process_exited(process),600;pollint=.1) == :timed_out
            kill(process); wait(process)
            throw(ArgumentError("Folder selection timed out; choose Browse again"))
        end
        success(process) || throw(ArgumentError("Windows could not open the folder dialog; enter a folder path directly"))
        data = JSON3.read(read(result,String))
        get(data,"ok",false) || throw(ArgumentError(string(get(data,"error","Folder selection failed"))))
        cancelled = get(data,"cancelled",false) === true
        selected = get(data,"path",nothing)
        if !cancelled
            selected isa AbstractString && isabspath(selected) && isdir(selected) ||
                throw(ArgumentError("The selected folder is unavailable"))
        end
        Dict("ok"=>true,"cancelled"=>cancelled,"path"=>cancelled ? nothing : String(selected))
    finally
        isfile(result) && rm(result;force=true)
    end
end

function choose_folder(req,runtime; picker=_native_folder)
    path,title = _validate_request(req)
    trylock(DIALOG_LOCK) || throw(ArgumentError("A folder dialog is already open; finish or cancel it first"))
    try
        picker(path,title,runtime)
    finally
        unlock(DIALOG_LOCK)
    end
end

function _windows_file(path,title,runtime,mode)
    Sys.iswindows() || throw(ArgumentError("Native study file dialogs require Windows"))
    mode in ("open","save") || throw(ArgumentError("Invalid file dialog mode"))
    mkpath(runtime);result,io=mktemp(runtime);close(io)
    try
        powershell=joinpath(get(ENV,"SystemRoot",raw"C:\Windows"),"System32","WindowsPowerShell","v1.0","powershell.exe")
        script=joinpath(@__DIR__,"native_file_dialog.ps1")
        cmd=`$powershell -NoProfile -NonInteractive -STA -WindowStyle Hidden -ExecutionPolicy Bypass -File $script`
        process=run(ignorestatus(addenv(cmd,"LUDWIG_DIALOG_PATH"=>path,"LUDWIG_DIALOG_TITLE"=>title,
            "LUDWIG_DIALOG_RESULT"=>result,"LUDWIG_DIALOG_MODE"=>mode));wait=false)
        if timedwait(()->process_exited(process),600;pollint=.1)==:timed_out
            kill(process);wait(process);throw(ArgumentError("File selection timed out"))
        end
        success(process) || throw(ArgumentError("Windows could not open the study file dialog"))
        data=JSON3.read(read(result,String));get(data,"ok",false) || throw(ArgumentError(string(get(data,"error","File selection failed"))))
        cancelled=get(data,"cancelled",false)===true;chosen=get(data,"path",nothing)
        if !cancelled
            chosen isa AbstractString && isabspath(chosen) || throw(ArgumentError("Invalid selected file"))
            lowercase(splitext(chosen)[2])==".ludwig-study" || throw(ArgumentError("Choose a .ludwig-study file"))
            (mode=="open" ? isfile(chosen) : isdir(dirname(chosen))) || throw(ArgumentError("Selected study path is unavailable"))
        end
        Dict("ok"=>true,"cancelled"=>cancelled,"path"=>cancelled ? nothing : String(chosen))
    finally
        isfile(result) && rm(result;force=true)
    end
end

function choose_file(req,runtime;mode="open",picker=_native_file)
    path,title=_validate_request(req)
    trylock(DIALOG_LOCK) || throw(ArgumentError("A file dialog is already open; finish or cancel it first"))
    try
        picker(path,title,runtime,mode)
    finally
        unlock(DIALOG_LOCK)
    end
end

"System pickers use argument arrays: titles and paths are never shell code."
function _desktop_selection(path,title,mode)
    command = if Sys.isapple()
        script=joinpath(@__DIR__,"native_dialog.applescript")
        folder=isdir(path) ? path : isdir(dirname(path)) ? dirname(path) : homedir()
        filename=mode=="save"&&!isempty(path)&&!isdir(path) ? basename(path) : "Aero study.ludwig-study"
        `osascript $script $mode $folder $title $filename`
    elseif Sys.which("zenity")!==nothing
        args=String[Sys.which("zenity"),"--file-selection","--title=$title"]
        !isempty(path) && push!(args,"--filename=$path")
        mode=="folder" ? push!(args,"--directory") : mode=="save" ? append!(args,["--save","--confirm-overwrite","--file-filter=Study | *.ludwig-study"]) : push!(args,"--file-filter=Study | *.ludwig-study")
        Cmd(args)
    elseif Sys.which("kdialog")!==nothing
        flag=mode=="folder" ? "--getexistingdirectory" : mode=="save" ? "--getsavefilename" : "--getopenfilename"
        args=String[Sys.which("kdialog"),"--title",title,flag,isempty(path) ? homedir() : path]
        mode!="folder" && push!(args,"*.ludwig-study|OpenAeroLab Study")
        Cmd(args)
    else
        throw(ArgumentError("No desktop file picker is installed. Install zenity or kdialog, or enter the destination path directly / use the browser file selector."))
    end
    output=IOBuffer();errors=IOBuffer()
    process=run(pipeline(ignorestatus(command),stdout=output,stderr=errors);wait=false)
    if timedwait(()->process_exited(process),600;pollint=.1)==:timed_out
        kill(process);wait(process);throw(ArgumentError("File selection timed out"))
    end
    chosen=strip(String(take!(output)))
    # AppleScript returns an empty selection for cancellation. GTK/Qt use exit 1.
    cancelled=(success(process)&&isempty(chosen))||(!Sys.isapple()&&process.exitcode==1&&isempty(String(take!(errors))))
    !success(process)&&!cancelled && throw(ArgumentError("The desktop file dialog failed; enter a path directly"))
    cancelled && return Dict("ok"=>true,"cancelled"=>true,"path"=>nothing)
    mode=="save" && isempty(splitext(chosen)[2]) && (chosen*=".ludwig-study")
    isabspath(chosen) || throw(ArgumentError("The selected path must be absolute"))
    if mode=="folder"
        isdir(chosen) || throw(ArgumentError("Selected folder is unavailable"))
    else
        lowercase(splitext(chosen)[2])==".ludwig-study" || throw(ArgumentError("Choose a .ludwig-study file"))
        (mode=="open" ? isfile(chosen) : isdir(dirname(chosen))) || throw(ArgumentError("Selected study path is unavailable"))
    end
    Dict("ok"=>true,"cancelled"=>false,"path"=>chosen)
end
_native_folder(path,title,runtime) = Sys.iswindows() ? _windows_folder(path,title,runtime) : _desktop_selection(path,title,"folder")
_native_file(path,title,runtime,mode) = Sys.iswindows() ? _windows_file(path,title,runtime,mode) : _desktop_selection(path,title,mode)

end
