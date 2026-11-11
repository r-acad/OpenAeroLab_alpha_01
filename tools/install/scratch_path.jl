module OpenAeroLabScratch

"The private setup scratch directory; all paths resolve inside the configured private root."
function scratch_directory(private::AbstractString)
    joinpath(abspath(private),"TEMP_DELETEABLE","setup")
end

"Ask Windows for an existing short-name alias; never create a drive mapping or change settings."
function windows_short_path(path::AbstractString)
    Sys.iswindows()||return nothing
    needed=ccall((:GetShortPathNameW,"kernel32"),UInt32,(Cwstring,Ptr{UInt16},UInt32),path,C_NULL,0)
    needed==0&&return nothing
    buffer=Vector{UInt16}(undef,needed)
    count=ccall((:GetShortPathNameW,"kernel32"),UInt32,(Cwstring,Ptr{UInt16},UInt32),path,buffer,needed)
    count==0&&return nothing
    transcode(String,buffer[1:count])
end

function choose_scratch_path(path::AbstractString;shortener=windows_short_path)
    candidate=try shortener(path) catch; nothing end
    if candidate isa AbstractString && isdir(candidate)
        same=try Base.Filesystem.samefile(path,candidate) catch; false end
        same && return (path=String(candidate),shortened=length(candidate)<length(path))
    end
    (path=String(path),shortened=false)
end

"Prepare TEMP before loading Pkg: Julia's own first precompile also uses this path."
function prepare_windows_scratch!(;private=get(ENV,"LUDWIG_PRIVATE_ROOT",normpath(joinpath(@__DIR__,"..","..","..","02_PRIVATE_PROJECT_DEVELOPMENT"))),
        shortener=windows_short_path,io=stderr)
    scratch=scratch_directory(private);mkpath(scratch)
    chosen=choose_scratch_path(scratch;shortener)
    if length(scratch)>=140
        # Some Julia/LibGit2 operations canonicalize Windows short names again.
        println(io,"The setup temporary directory has a long path. If Julia reports a path-length error, set LUDWIG_PRIVATE_ROOT to a shorter writable private-project location and rerun setup.")
    end
    for key in ("TEMP","TMP","TMPDIR");ENV[key]=chosen.path;end
    (path=chosen.path,canonical=scratch,shortened=chosen.shortened)
end
end
