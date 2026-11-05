module OpenAeroLabRuntimePromotion
using UUIDs, Dates

function inside(path, root)
    path = normpath(abspath(path)); root = normpath(abspath(root))
    relative = relpath(path, root)
    !isabspath(relative) && relative != "." && first(splitpath(relative)) != ".."
end

function reject_links(path, root)
    current = abspath(path)
    while inside(current, root)
        islink(current) && error("A runtime installation path contains a link: $current")
        current = dirname(current)
    end
end

function version_ok(binary, version)
    command = Cmd(Cmd([binary, "--startup-file=no", "--version"]); windows_hide=true)
    try
        strip(read(pipeline(command; stderr=devnull), String)) == "julia version $version"
    catch
        false
    end
end

function promote(private_root, staging, version)
    private_root = abspath(private_root); staging = abspath(staging)
    required = strip(read(joinpath(@__DIR__, "julia_version.txt"), String))
    version == required || error("The requested runtime differs from the pinned Julia version.")
    setup_root = joinpath(private_root, "TEMP_DELETEABLE", "setup")
    inside(staging, setup_root) || error("Staging must be inside the selected private TEMP_DELETEABLE/setup folder.")
    dirname(staging) == setup_root || error("Unexpected staging location.")
    reject_links(staging, private_root)
    source = joinpath(staging, "u")
    source_binary = joinpath(source, "julia-$version", "bin", "julia.exe")
    version_ok(source_binary, version) || error("The staged Julia runtime cannot start; the existing runtime was not changed.")
    runtime = joinpath(private_root, "runtime", "julia-$version")
    inside(runtime, private_root) || error("Runtime location escaped the private directory.")
    reject_links(runtime, private_root)
    prepared = joinpath(staging, "ready")
    ispath(prepared) && error("Prepared runtime staging already exists; retry with fresh staging.")
    # The bootstrap currently runs from source. Copy before renaming, so an
    # executable/DLL file lock on Windows cannot interrupt final promotion.
    println("Preparing the runtime independently of the running bootstrap...")
    cp(source, prepared; follow_symlinks=false)
    backup = nothing
    try
        if ispath(runtime)
            isdir(runtime) || error("The runtime destination is occupied by a file: $runtime")
            backup = joinpath(private_root, "runtime", "julia-$version.previous-$(uuid4())")
            inside(backup, private_root) || error("Backup location escaped the private directory.")
            mv(runtime, backup)
            println("Previous runtime preserved at $backup")
        end
        mv(prepared, runtime)
        version_ok(joinpath(runtime, "julia-$version", "bin", "julia.exe"), version) ||
            error("Julia could not start in its final directory. Ask IT to approve this application path.")
    catch err
        # Preserve a failed new installation for diagnosis and restore the old
        # installation. No recursive deletes occur during recovery.
        if ispath(runtime) && !ispath(prepared)
            failed = joinpath(staging, "failed-runtime")
            inside(failed, setup_root) || error("Recovery destination escaped staging.")
            mv(runtime, failed)
            println("Failed new runtime preserved at $failed")
        end
        if backup !== nothing && ispath(backup) && !ispath(runtime)
            mv(backup, runtime)
            println("Previous runtime restored at $runtime")
        end
        rethrow(err)
    end
    println("Julia $version validated at $runtime on $(now(UTC)) UTC.")
    backup === nothing || println("The previous runtime remains available for inspection at $backup.")
    runtime
end
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        length(ARGS) == 3 || error("Usage: promote_runtime.jl PRIVATE_ROOT STAGING VERSION")
        OpenAeroLabRuntimePromotion.promote(ARGS...)
    catch err
        showerror(stderr, err, catch_backtrace()); println(stderr)
        exit(1)
    end
end
