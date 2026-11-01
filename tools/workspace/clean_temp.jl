# Dry run by default. Stop the apps/bridge, then pass --delete to clear scratch.
using JSON3
const PUBLIC=dirname(dirname(@__DIR__))
const PRIVATE=abspath(get(ENV,"LUDWIG_PRIVATE_ROOT",joinpath(dirname(PUBLIC),"02_PRIVATE_PROJECT_DEVELOPMENT")))
const TEMP_ROOT=joinpath(PRIVATE,"TEMP_DELETEABLE")
within(p,b)=let r=relpath(abspath(p),abspath(b)); !isabspath(r) && r!="." && first(splitpath(r))!=".." end
function alive(pid)
    pid isa Integer && pid>0 || return false
    if Sys.iswindows()
        handle=ccall((:OpenProcess,"kernel32"),Ptr{Cvoid},(UInt32,Int32,UInt32),0x1000,0,pid)
        handle==C_NULL && return false
        code=Ref{UInt32}(0)
        try
            ccall((:GetExitCodeProcess,"kernel32"),Int32,(Ptr{Cvoid},Ref{UInt32}),handle,code)!=0 && code[]==259
        finally
            ccall((:CloseHandle,"kernel32"),Int32,(Ptr{Cvoid},),handle)
        end
    else
        ccall(:kill,Cint,(Cint,Cint),pid,0)==0
    end
end
function main()
    isdir(TEMP_ROOT)||return
    islink(TEMP_ROOT)&&error("Scratch root is a link; refusing cleanup")
    delete="--delete" in ARGS
    if delete
        for marker in (joinpath(TEMP_ROOT,"ludwig_web","current_server.json"),
                       joinpath(PRIVATE,"runtime","telegram_reply_bridge","heartbeat.json"))
            isfile(marker)||continue
            data=JSON3.read(read(marker,String),Dict{String,Any})
            alive(get(data,"pid",0)) && error("Stop the running application before deleting scratch: $marker")
        end
    end
    listing=read(Cmd(["git","-C",PUBLIC,"worktree","list","--porcelain"]),String)
    worktrees=[strip(line[10:end]) for line in split(listing,'\n') if startswith(line,"worktree ")]
    for name in readdir(TEMP_ROOT)
        p=joinpath(TEMP_ROOT,name)
        within(p,TEMP_ROOT)||error("Unsafe cleanup path: $p")
        if any(w->normpath(w)==normpath(p)||within(w,p),worktrees)
            println("KEEP registered Git worktree: ",p);continue
        end
        if islink(p) || (isdir(p) && any(islink(joinpath(d,n)) for (d,ds,fs) in walkdir(p;follow_symlinks=false) for n in vcat(ds,fs)))
            println("KEEP directory link requiring review: ",p);continue
        end
        within(realpath(p),realpath(TEMP_ROOT))||error("Resolved cleanup path escapes scratch: $p")
        println(delete ? "DELETE " : "Would delete ",p)
        delete && rm(p;recursive=isdir(p),force=true)
    end
    delete || println("Dry run only. Important data belongs in private results/studies/validation/runtime, never here.")
end
abspath(PROGRAM_FILE)==abspath(@__FILE__) && main()
